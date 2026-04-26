#!/usr/bin/env python3
"""Kronos feature extractor — supports DAEMON mode and one-shot mode.

Features produced (v18-DE3 inputs):
    kronos_max_high_above, kronos_min_low_below, kronos_pred_atr_30bar,
    kronos_dir_move, kronos_close_vs_entry, kronos_inf_time_s

Modes:
    --daemon    Long-lived process; loads model once, then loops:
                stdin: one JSON request per line
                stdout: {"event":"loading"} then {"event":"ready"} then
                        one JSON response line per request.
                Request schema (one line):
                    {"bars":[{"ts":..,"open":..,...,"volume":..}, ...],
                     "entry_price": 6000.0,
                     "side": "LONG"   # optional
                    }
                Special command:
                    {"cmd":"shutdown"}  -> exits cleanly.
    (default)   One-shot mode (legacy): reads a single JSON payload from
                stdin (multi-line OK), prints features as the LAST line of
                stdout, exits.

Tunable via env (used in BOTH modes):
    KRONOS_SAMPLE_COUNT  (default "1")  — number of forecast samples.
                                          The legacy script used 1; we keep
                                          1 as default to preserve behavior
                                          unless the bot opts in.
    KRONOS_PRED_LEN      (default "30") — forecast horizon in bars.
                                          Legacy script used 30; we keep 30
                                          as default to preserve behavior.

Designed to be called via subprocess by the live bot from the .kronos_venv
venv. Keeps PyTorch / Kronos out of the bot's main process so OMP threading
collisions are impossible.

Exit codes (one-shot mode only):
    0  success — JSON features printed on last line
    1  bad input / not enough bars
    2  Kronos load / inference error
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

# OMP / threading isolation — critical when called from the bot. The bot
# may itself be using multiple threads; Kronos in this subprocess must
# not contend with that.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Resolve repo root (this file lives at <root>/tools/kronos_predict_features.py).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "kronos_external"))

CTX_BARS = 512
PRED_LEN = int(os.environ.get("KRONOS_PRED_LEN", "30"))
SAMPLE_COUNT = int(os.environ.get("KRONOS_SAMPLE_COUNT", "1"))
MIN_BARS = 100  # Kronos needs >=100 bars of context


def _emit_error(stage: str, err: Exception, code: int = 2) -> int:
    payload = {"error": stage, "exc": str(err)}
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()
    return code


def _build_df(bars_in):
    """Validate + build the OHLCV DataFrame. Raises ValueError on bad input."""
    import pandas as pd
    if not isinstance(bars_in, list) or len(bars_in) < MIN_BARS:
        raise ValueError(
            f"not_enough_bars n={len(bars_in) if isinstance(bars_in, list) else 0} need={MIN_BARS}"
        )
    df = pd.DataFrame(bars_in)
    # ts column REQUIRED to build x_timestamp. Accept "ts" or "timestamp".
    if "ts" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" not in df.columns:
        raise ValueError("bars_missing_ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"bars_missing_col:{col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df.iloc[-CTX_BARS:].copy()
    if len(df) < MIN_BARS:
        raise ValueError(f"after_clean_too_few_bars n={len(df)}")
    df["amount"] = df["volume"] * df["close"]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _run_inference(predictor, df, entry_price):
    """Run Kronos inference and return features dict."""
    import pandas as pd
    import torch
    x_ts = pd.Series(df.index)
    last_ts = df.index[-1]
    y_ts = pd.Series(pd.date_range(start=last_ts + pd.Timedelta(minutes=1),
                                   periods=PRED_LEN, freq="1min"))
    t0 = time.time()
    with torch.no_grad():
        pred_df = predictor.predict(
            df=df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=PRED_LEN,
            T=1.0,
            top_p=0.9,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )
    inf_time = time.time() - t0

    high_max = float(pred_df["high"].max())
    low_min = float(pred_df["low"].min())
    final_close = float(pred_df["close"].iloc[-1])
    atr_30 = float((pred_df["high"] - pred_df["low"]).mean())
    return {
        "kronos_pred_atr_30bar": atr_30,
        "kronos_dir_move": final_close - entry_price,
        "kronos_max_high_above": high_max - entry_price,
        "kronos_min_low_below": entry_price - low_min,
        "kronos_close_vs_entry": final_close - entry_price,
        "kronos_inf_time_s": inf_time,
    }


def _load_predictor():
    """Load model + tokenizer + predictor. Returns predictor."""
    import torch
    torch.set_num_threads(1)
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=CTX_BARS, device="cpu")
    return predictor


def _daemon_main() -> int:
    """Long-lived daemon. Reads JSON requests from stdin, writes responses."""
    sys.stdout.write(json.dumps({"event": "loading"}) + "\n")
    sys.stdout.flush()
    try:
        predictor = _load_predictor()
    except Exception as e:
        sys.stdout.write(json.dumps({"event": "load_error", "exc": str(e)}) + "\n")
        sys.stdout.flush()
        sys.stderr.write(traceback.format_exc())
        return 2
    sys.stdout.write(json.dumps({
        "event": "ready",
        "sample_count": SAMPLE_COUNT,
        "pred_len": PRED_LEN,
    }) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stdout.write(json.dumps({
                "kronos_failed": f"bad_json:{str(e)[:160]}"
            }) + "\n")
            sys.stdout.flush()
            continue
        if isinstance(req, dict) and req.get("cmd") == "shutdown":
            sys.stdout.write(json.dumps({"event": "shutdown"}) + "\n")
            sys.stdout.flush()
            return 0
        try:
            bars_in = req.get("bars") or []
            entry_price = float(req["entry_price"])
            df = _build_df(bars_in)
            feats = _run_inference(predictor, df, entry_price)
            sys.stdout.write(json.dumps(feats) + "\n")
            sys.stdout.flush()
        except Exception as e:
            # Stay alive on per-request errors.
            sys.stdout.write(json.dumps({
                "kronos_failed": str(e)[:200]
            }) + "\n")
            sys.stdout.flush()
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
    return 0


def _oneshot_main() -> int:
    # ----- 1) read stdin -----
    try:
        raw = sys.stdin.read()
        if not raw or not raw.strip():
            print(json.dumps({"error": "empty_stdin"}))
            return 1
        payload = json.loads(raw)
    except Exception as e:
        return _emit_error("stdin_parse", e, code=1)

    try:
        bars_in = payload.get("bars") or []
        entry_price = float(payload["entry_price"])
    except Exception as e:
        return _emit_error("payload_validate", e, code=1)

    # ----- 2) heavy imports -----
    try:
        import pandas as pd  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        return _emit_error("imports", e)

    # ----- 3) build DF -----
    try:
        df = _build_df(bars_in)
    except Exception as e:
        return _emit_error("df_build", e, code=1)

    # ----- 4) load + infer -----
    try:
        predictor = _load_predictor()
    except Exception as e:
        return _emit_error("model_load", e)

    try:
        feats = _run_inference(predictor, df, entry_price)
    except Exception as e:
        return _emit_error("inference", e)

    # last line of stdout = JSON features.
    sys.stdout.write(json.dumps(feats) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    if "--daemon" in sys.argv[1:]:
        return _daemon_main()
    return _oneshot_main()


if __name__ == "__main__":
    sys.exit(main())

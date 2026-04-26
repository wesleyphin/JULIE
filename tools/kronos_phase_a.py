"""Kronos Phase A — extract forecast features for 50 random DE3 candidates and
compute univariate correlations against is_winner.

Hard rules:
- CPU only
- OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1
- Process in batches of 10, gc.collect after each
- RAM check before every batch: if free < 1500MB pause+retry, then ABORT and save partial.

Outputs:
  artifacts/kronos_phase_a_50sample.parquet
  /tmp/kronos_phase_a_report.md
"""
from __future__ import annotations

import os
import sys
import gc
import time
import json
import math
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = "/Users/wes/Downloads/JULIE001"
sys.path.insert(0, os.path.join(ROOT, "kronos_external"))

import psutil
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from model import Kronos, KronosTokenizer, KronosPredictor


CORPUS_PATH = os.path.join(ROOT, "artifacts", "v12_training_corpus.parquet")
BARS_PATH = os.path.join(ROOT, "es_master_outrights.parquet")
OUT_PARQUET = os.path.join(ROOT, "artifacts", "kronos_phase_a_50sample.parquet")
REPORT_PATH = "/tmp/kronos_phase_a_report.md"

N_SAMPLE = 50
RANDOM_STATE = 42
CTX_BARS = 512
PRED_LEN = 30
SAMPLE_COUNT = 5
BATCH = 10
RAM_FLOOR_MB = 1500

PEAK_USED = {"max_used_mb": 0, "min_free_mb": 10**9}


def ram_mb() -> int:
    return psutil.virtual_memory().available // 1024 // 1024


def used_mb() -> int:
    p = psutil.Process(os.getpid())
    return p.memory_info().rss // 1024 // 1024


def track_ram(label: str = "") -> tuple[int, int]:
    free = ram_mb()
    used = used_mb()
    PEAK_USED["max_used_mb"] = max(PEAK_USED["max_used_mb"], used)
    PEAK_USED["min_free_mb"] = min(PEAK_USED["min_free_mb"], free)
    if label:
        print(f"  [{label}] free={free} MB, used={used} MB")
    return free, used


def wait_or_abort_on_ram(label: str) -> bool:
    """Return True if OK to proceed. False if abort."""
    free, _ = track_ram(label)
    if free >= RAM_FLOOR_MB:
        return True
    print(f"  [{label}] free RAM {free} MB < floor {RAM_FLOOR_MB}; pausing 5s and retrying once")
    time.sleep(5)
    gc.collect()
    free, _ = track_ram(label + "-retry")
    if free >= RAM_FLOOR_MB:
        return True
    print(f"  [{label}] still low ({free} MB); ABORT")
    return False


def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS_PATH)
    de3_fr = df[(df["family"] == "de3") & (df["allowed_by_friend_rule"] == True)].copy()
    de3_fr = de3_fr.reset_index(drop=True)
    return de3_fr


def load_bars_indexed(symbols_needed: list[str]) -> dict[str, pd.DataFrame]:
    """Load bars only for symbols we need. Returns {symbol: df with ts index}."""
    print(f"Loading bars for {len(symbols_needed)} symbols: {symbols_needed}")
    df = pd.read_parquet(
        BARS_PATH,
        columns=["open", "high", "low", "close", "volume", "symbol"],
    )
    # timestamp is index per schema
    df = df[df["symbol"].isin(symbols_needed)].copy()
    print(f"  bars rows after symbol filter: {len(df):,}")
    out = {}
    for sym in symbols_needed:
        sub = df[df["symbol"] == sym].copy()
        sub = sub.drop(columns=["symbol"])
        sub = sub.sort_index()
        # Ensure single-tz datetimeindex
        out[sym] = sub
        print(f"   {sym}: {len(sub):,} bars  range {sub.index.min()} .. {sub.index.max()}")
    return out


def get_window(bars: pd.DataFrame, end_ts: pd.Timestamp, n: int) -> pd.DataFrame | None:
    """Return last n bars strictly before end_ts."""
    # Ensure tz alignment
    if bars.index.tz is None and end_ts.tzinfo is not None:
        end_ts_use = end_ts.tz_convert(None)
    else:
        end_ts_use = end_ts
    sub = bars.loc[:end_ts_use]
    # exclude bars exactly at end_ts (the entry bar itself)
    if len(sub) and sub.index[-1] == end_ts_use:
        sub = sub.iloc[:-1]
    if len(sub) < 100:
        return None
    return sub.iloc[-n:].copy()


def normalize_bars(window: pd.DataFrame) -> pd.DataFrame:
    """Ensure required cols and add 'amount' if missing."""
    df = window[["open", "high", "low", "close", "volume"]].copy().astype(float)
    df["amount"] = df["volume"] * df["close"]
    # Drop NaN rows defensively
    df = df.dropna()
    return df


def build_y_timestamps(last_ts: pd.Timestamp, n: int) -> pd.Series:
    # 1-min cadence forecast horizon
    idx = pd.date_range(start=last_ts + pd.Timedelta(minutes=1), periods=n, freq="1min")
    return pd.Series(idx)


def kronos_features(pred_df: pd.DataFrame, entry_price: float, inf_time_s: float) -> dict:
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
        "kronos_inf_time_s": inf_time_s,
    }


def main() -> int:
    t_start = time.time()
    print(f"[init] free={ram_mb()} MB, used={used_mb()} MB")

    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  total DE3+friend: {len(corpus)}")
    samp = corpus.sample(n=N_SAMPLE, random_state=RANDOM_STATE).copy().reset_index(drop=True)
    print(f"  sampled {len(samp)} candidates (random_state={RANDOM_STATE})")

    # is_winner label
    samp["is_winner"] = (samp["net_pnl_after_haircut"] > 0).astype(int)
    print(f"  winners in sample: {samp['is_winner'].sum()} / {len(samp)}")

    # Symbols we need
    symbols_needed = sorted(samp["contract"].dropna().unique().tolist())

    print("Loading bars...")
    bars_by_sym = load_bars_indexed(symbols_needed)
    track_ram("post-bars-load")

    print("Loading Kronos model + tokenizer (CPU)...")
    t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=CTX_BARS, device="cpu")
    print(f"  loaded in {time.time()-t0:.1f}s")
    track_ram("post-model-load")

    rows = []
    skipped = []
    proc = 0
    aborted = False

    for batch_start in range(0, len(samp), BATCH):
        ok = wait_or_abort_on_ram(f"pre-batch-{batch_start}")
        if not ok:
            aborted = True
            break

        batch = samp.iloc[batch_start: batch_start + BATCH]
        for idx, row in batch.iterrows():
            try:
                contract = row["contract"]
                ts = row["ts"]
                entry_price = float(row["entry_price"])
                bars = bars_by_sym.get(contract)
                if bars is None or len(bars) == 0:
                    skipped.append({"idx": int(idx), "reason": "no_bars_for_symbol", "contract": contract})
                    continue
                window = get_window(bars, ts, CTX_BARS)
                if window is None:
                    skipped.append({"idx": int(idx), "reason": "lt_100_bars", "contract": contract, "ts": str(ts)})
                    continue
                df_win = normalize_bars(window)
                if len(df_win) < 100:
                    skipped.append({"idx": int(idx), "reason": "lt_100_bars_post_clean", "contract": contract})
                    continue

                x_ts = pd.Series(df_win.index)
                # Build forecast timestamps stepping 1 min from last bar
                y_ts = build_y_timestamps(df_win.index[-1], PRED_LEN)

                t0 = time.time()
                with torch.no_grad():
                    pred_df = predictor.predict(
                        df=df_win,
                        x_timestamp=x_ts,
                        y_timestamp=y_ts,
                        pred_len=PRED_LEN,
                        T=1.0,
                        top_p=0.9,
                        sample_count=SAMPLE_COUNT,
                        verbose=False,
                    )
                inf_time = time.time() - t0

                feats = kronos_features(pred_df, entry_price, inf_time)
                feats.update({
                    "idx": int(idx),
                    "ts": ts,
                    "contract": contract,
                    "side": row["side"],
                    "entry_price": entry_price,
                    "net_pnl_after_haircut": float(row["net_pnl_after_haircut"]),
                    "is_winner": int(row["is_winner"]),
                    "n_ctx_bars": int(len(df_win)),
                })
                rows.append(feats)
                proc += 1
                del pred_df
            except Exception as e:
                tb = traceback.format_exc()
                skipped.append({"idx": int(idx), "reason": f"exception: {e}", "trace": tb[:500]})
                print(f"  ! exception at idx={idx}: {e}")

        gc.collect()
        free, used = track_ram(f"end-batch-{batch_start} (proc={proc})")

    elapsed = time.time() - t_start
    print(f"\nDone. processed={proc}, skipped={len(skipped)}, total_elapsed={elapsed:.1f}s, aborted={aborted}")

    # Save partial/full results
    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_parquet(OUT_PARQUET, index=False)
        print(f"Wrote {OUT_PARQUET}: {len(out_df)} rows")
    else:
        out_df = pd.DataFrame()
        print("No rows to save")

    # Compute correlations
    feat_cols = [
        "kronos_pred_atr_30bar",
        "kronos_dir_move",
        "kronos_max_high_above",
        "kronos_min_low_below",
        "kronos_close_vs_entry",
    ]
    corrs = {}
    if len(out_df) > 5:
        for c in feat_cols:
            try:
                corrs[c] = float(out_df[c].corr(out_df["is_winner"]))
            except Exception:
                corrs[c] = float("nan")
    else:
        corrs = {c: float("nan") for c in feat_cols}

    # Build report
    abs_max_corr = max((abs(v) for v in corrs.values() if not (v is None or math.isnan(v))), default=0.0)
    decision = "STOP"
    reasons = []
    mean_inf = float(out_df["kronos_inf_time_s"].mean()) if len(out_df) else float("nan")
    if aborted:
        reasons.append("aborted due to RAM floor")
    if abs_max_corr < 0.05:
        reasons.append(f"all |corr| < 0.05 (max abs = {abs_max_corr:.4f})")
    if not math.isnan(mean_inf) and mean_inf > 10.0:
        reasons.append(f"mean per-candidate inference {mean_inf:.2f}s > 10s")
    if PEAK_USED["min_free_mb"] < 1500:
        reasons.append(f"min free RAM {PEAK_USED['min_free_mb']} MB < 1500 MB floor")
    if not reasons and (abs_max_corr >= 0.08) and not math.isnan(mean_inf) and mean_inf < 5.0:
        decision = "PROCEED"
    elif not reasons:
        decision = "MARGINAL"

    lines = []
    lines.append("# Kronos Phase A Report\n")
    lines.append(f"- corpus: {CORPUS_PATH}")
    lines.append(f"- bars: {BARS_PATH}")
    lines.append(f"- random_state: {RANDOM_STATE}")
    lines.append(f"- ctx_bars: {CTX_BARS}, pred_len: {PRED_LEN}, sample_count: {SAMPLE_COUNT}")
    lines.append(f"- batch size: {BATCH}, RAM floor: {RAM_FLOOR_MB} MB")
    lines.append("")
    lines.append("## Run stats")
    lines.append(f"- candidates sampled: {N_SAMPLE}")
    lines.append(f"- successful inferences: {proc}")
    lines.append(f"- skipped: {len(skipped)}")
    lines.append(f"- total elapsed: {elapsed:.1f}s")
    lines.append(f"- mean inference time: {mean_inf:.2f}s" if not math.isnan(mean_inf) else "- mean inference time: n/a")
    lines.append(f"- aborted: {aborted}")
    lines.append("")
    lines.append("## RAM stats")
    lines.append(f"- peak process RSS: {PEAK_USED['max_used_mb']} MB")
    lines.append(f"- min free system RAM: {PEAK_USED['min_free_mb']} MB")
    lines.append("")
    if len(out_df):
        lines.append("## Feature distributions")
        desc = out_df[feat_cols + ["kronos_inf_time_s"]].describe().T[["min", "50%", "max", "std", "mean"]]
        lines.append("```")
        lines.append(desc.to_string())
        lines.append("```")
        lines.append("")
        lines.append("## Univariate correlations vs is_winner")
        for c, v in corrs.items():
            lines.append(f"- {c}: {v:+.4f}")
        lines.append(f"\nMax |corr|: {abs_max_corr:.4f}")
    else:
        lines.append("(no output rows)")
    lines.append("")
    lines.append("## Skipped breakdown")
    if skipped:
        from collections import Counter
        c = Counter([s.get("reason", "?").split(":")[0] for s in skipped])
        for k, v in c.most_common():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"**{decision}**")
    if reasons:
        lines.append("\nStop reasons:")
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")
    lines.append("## Phase B feasibility")
    if not math.isnan(mean_inf):
        full_eta_s = mean_inf * 1599
        lines.append(f"- per-candidate: {mean_inf:.2f}s")
        lines.append(f"- 1599-candidate ETA: {full_eta_s/60:.1f} min ({full_eta_s/3600:.2f} hr)")
        if full_eta_s > 4 * 3600:
            lines.append("- WARNING: > 4 hr; consider reducing sample_count")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote report: {REPORT_PATH}")
    print(f"\nDECISION: {decision}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fast AetherFlow signal-replay → simulated trade log.

The live-loop backtest path takes ~28s/bar because it re-runs every
strategy + filter + regime classifier per bar. AF doesn't need any of
that to TRAIN A G GATE on its trades — we only need:

  1. The list of bars where the *deployed* AF model would have fired,
     at the threshold the live bot actually uses.
  2. Each signal's bracket (tp_dist, sl_dist, horizon_bars, side).
  3. A simple bracket simulation (high/low scan over the next N bars)
     to determine PnL.

This bypasses the live loop entirely and processes the full 13.5-month
window in minutes instead of months.

Output:
  backtest_reports/af_fast_replay/<run_name>/closed_trades.json
  backtest_reports/af_fast_replay/<run_name>/summary.json

Usage:
  python scripts/signal_gate/fast_aetherflow_replay.py \
      --start 2025-03-01 --end 2026-04-20 \
      --run-name af_full_2025_2026
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aetherflow_base_cache import (  # noqa: E402
    DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    resolve_full_manifold_base_features_path,
    validate_full_manifold_base_features_path,
)
from aetherflow_features import BASE_FEATURE_COLUMNS  # noqa: E402
from config import CONFIG  # noqa: E402

NY = "America/New_York"
PARQUET_DEFAULT = ROOT / "es_master_outrights.parquet"
POINT_VALUE = 5.0  # MES = $5/pt

# Front-month roll map — we use the symbol that's the active contract on
# the entry bar. (Same map as train_per_strategy_models.py)
ROLL_MAP = [
    (pd.Timestamp("2025-01-01", tz=NY), "ESH5"),
    (pd.Timestamp("2025-03-17", tz=NY), "ESM5"),
    (pd.Timestamp("2025-06-16", tz=NY), "ESU5"),
    (pd.Timestamp("2025-09-15", tz=NY), "ESZ5"),
    (pd.Timestamp("2025-12-15", tz=NY), "ESH6"),
    (pd.Timestamp("2026-03-16", tz=NY), "ESM6"),
]


def active_symbol(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    best = ROLL_MAP[0][1]
    for d, s in ROLL_MAP:
        if ts >= d:
            best = s
        else:
            break
    return best


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _entry_window_config() -> tuple[bool, int, int]:
    exec_cfg = dict(CONFIG.get("BACKTEST_EXECUTION", {}) or {})
    enabled = bool(exec_cfg.get("enforce_no_new_entries_window", False))
    start_minutes = (
        max(0, min(23, _coerce_int(exec_cfg.get("no_new_entries_start_hour_et", 16), 16))) * 60
        + max(0, min(59, _coerce_int(exec_cfg.get("no_new_entries_start_minute_et", 0), 0)))
    )
    end_minutes = (
        max(0, min(23, _coerce_int(exec_cfg.get("no_new_entries_end_hour_et", 18), 18))) * 60
        + max(0, min(59, _coerce_int(exec_cfg.get("no_new_entries_end_minute_et", 0), 0)))
    )
    return enabled, start_minutes, end_minutes


def _minutes_in_half_open_window(minutes_et: int, start_minutes: int, end_minutes: int) -> bool:
    if start_minutes == end_minutes:
        return False
    if start_minutes < end_minutes:
        return start_minutes <= minutes_et < end_minutes
    return minutes_et >= start_minutes or minutes_et < end_minutes


def _signal_time_blocked_live_window(ts: pd.Timestamp) -> bool:
    enabled, start_minutes, end_minutes = _entry_window_config()
    if not enabled:
        return False
    ts_et = pd.Timestamp(ts)
    if ts_et.tzinfo is None:
        ts_et = ts_et.tz_localize(NY)
    else:
        ts_et = ts_et.tz_convert(NY)
    minutes_et = int(ts_et.hour) * 60 + int(ts_et.minute)
    return _minutes_in_half_open_window(minutes_et, start_minutes, end_minutes)


def load_bars_for_window(parquet_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Pull MES bars covering [start, end] using the front-month roll map.
    Returns a single concatenated, sorted, deduped df indexed by timestamp."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY)
    elif str(df.index.tz) != NY:
        df.index = df.index.tz_convert(NY)
    # Need warmup before start (AF wants ~320 bars min); pad with 7 days
    pad = pd.Timedelta(days=10)
    win = df.loc[(df.index >= start - pad) & (df.index <= end)]
    if "symbol" not in win.columns:
        return win[["open", "high", "low", "close", "volume"]].copy()
    # Pick front-month per timestamp
    parts = []
    boundaries = ROLL_MAP + [(pd.Timestamp("2030-01-01", tz=NY), "ZZZZ")]
    for i in range(len(boundaries) - 1):
        cut_lo, sym = boundaries[i]
        cut_hi, _ = boundaries[i + 1]
        if cut_hi < (start - pad) or cut_lo > end:
            continue
        sub = win.loc[(win.index >= cut_lo) & (win.index < cut_hi) & (win["symbol"] == sym)]
        if not sub.empty:
            parts.append(sub[["open", "high", "low", "close", "volume"]])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _resolve_path(path_value: str | Path | None, fallback: str | Path | None = None) -> Path:
    text = str(path_value or "").strip()
    if text:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()
    if fallback is None:
        raise SystemExit("Required path not provided")
    path = Path(str(fallback)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_base_features_for_window(base_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    validate_full_manifold_base_features_path(base_path, required_columns=BASE_FEATURE_COLUMNS)
    data = pd.read_parquet(base_path)
    if data.empty:
        return pd.DataFrame()
    data = data.sort_index()
    data = data.loc[(data.index >= start) & (data.index <= end)]
    if data.empty:
        return pd.DataFrame()
    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def simulate_bracket(
    bars: pd.DataFrame,
    entry_idx: int,
    side: str,
    tp_dist: float,
    sl_dist: float,
    horizon_bars: int,
    use_time_stop: bool,
) -> dict:
    """Walk forward from entry_idx+1, look for TP or SL hit each bar.
    Returns dict with exit_idx, exit_price, exit_source, pnl_points.
    Conservative tie-break: if a single bar's H/L could hit both TP and SL,
    assume SL hits first."""
    if entry_idx + 1 >= len(bars):
        return {"exit_idx": entry_idx, "exit_price": float(bars.iloc[entry_idx]["close"]),
                "exit_source": "no_data", "pnl_points": 0.0}
    entry_price = float(bars.iloc[entry_idx + 1]["open"])  # market on next bar
    if side == "LONG":
        tp_price = entry_price + tp_dist
        sl_price = entry_price - sl_dist
    else:
        tp_price = entry_price - tp_dist
        sl_price = entry_price + sl_dist

    end = min(len(bars), entry_idx + 1 + horizon_bars)
    for j in range(entry_idx + 1, end):
        h = float(bars.iloc[j]["high"])
        l = float(bars.iloc[j]["low"])
        # Conservative — if both hit in same bar, prefer SL
        sl_hit = (side == "LONG" and l <= sl_price) or (side == "SHORT" and h >= sl_price)
        tp_hit = (side == "LONG" and h >= tp_price) or (side == "SHORT" and l <= tp_price)
        if sl_hit and tp_hit:
            pnl_pts = -sl_dist
            return {"exit_idx": j, "exit_price": sl_price, "exit_source": "stop_pessimistic",
                    "pnl_points": pnl_pts, "entry_price": entry_price}
        if sl_hit:
            return {"exit_idx": j, "exit_price": sl_price, "exit_source": "stop",
                    "pnl_points": -sl_dist, "entry_price": entry_price}
        if tp_hit:
            return {"exit_idx": j, "exit_price": tp_price, "exit_source": "take",
                    "pnl_points": tp_dist, "entry_price": entry_price}
    # Time stop or end-of-data
    j = end - 1
    exit_price = float(bars.iloc[j]["close"])
    if side == "LONG":
        pnl_pts = exit_price - entry_price
    else:
        pnl_pts = entry_price - exit_price
    return {"exit_idx": j, "exit_price": exit_price,
            "exit_source": "horizon_time_stop" if use_time_stop else "horizon",
            "pnl_points": pnl_pts, "entry_price": entry_price}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--source", default=str(PARQUET_DEFAULT))
    p.add_argument(
        "--base-features",
        default="",
        help="Full precomputed manifold base cache. Defaults to AETHERFLOW_STRATEGY.backtest_base_features_file "
             f"or {DEFAULT_FULL_MANIFOLD_BASE_FEATURES}.",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--threshold-override", type=float, default=None,
                   help="Override AF threshold (default: use configured)")
    p.add_argument("--size-override", type=int, default=None)
    p.add_argument(
        "--recompute-candidate-features",
        action="store_true",
        help="Rebuild candidate features from bars instead of using AF's precomputed backtest feature path.",
    )
    p.add_argument(
        "--ignore-live-entry-window",
        action="store_true",
        help="Do not enforce the live RISK no-new-entries window during replay.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("af_fast")

    start_ts = pd.Timestamp(args.start, tz=NY)
    end_ts = pd.Timestamp(args.end, tz=NY) + pd.Timedelta(hours=23, minutes=59)

    logger.info("Loading bars %s -> %s from %s", start_ts, end_ts, args.source)
    bars = load_bars_for_window(Path(args.source), start_ts, end_ts)
    if bars.empty:
        raise SystemExit("No bars loaded for window")
    logger.info("Loaded %d bars (with %d days of warmup pad)", len(bars), 10)

    logger.info("Loading AetherFlowStrategy...")
    from aetherflow_strategy import AetherFlowStrategy
    af = AetherFlowStrategy()
    if not af.model_loaded:
        raise SystemExit("AF model not loaded — check config + model_file path")
    logger.info("AF loaded. threshold=%.3f min_conf=%.3f size=%d families=%s",
                af.threshold, af.min_confidence, af.size, af._candidate_family_names())
    live_window_enabled, start_minutes, end_minutes = _entry_window_config()
    logger.info(
        "Live entry window enforcement: enabled=%s active_replay=%s window=%02d:%02d-%02d:%02d ET",
        live_window_enabled,
        not bool(args.ignore_live_entry_window) and live_window_enabled,
        start_minutes // 60,
        start_minutes % 60,
        end_minutes // 60,
        end_minutes % 60,
    )

    logger.info("Computing AF replay signals...")
    t0 = datetime.now()
    if args.recompute_candidate_features:
        logger.info("AF replay mode: recompute_candidate_features=True")
        from aetherflow_features import build_feature_frame, build_manifold_feature_frame

        logger.info("Building manifold base frame ONCE (explicit recompute path)...")
        base_features = build_manifold_feature_frame(bars)
        logger.info("Manifold base frame built: %d rows", len(base_features))

        candidate_frames = []
        for family in af._candidate_family_names():
            logger.info("Building AF candidate frame for family=%s", family)
            feats = build_feature_frame(
                base_features=base_features,
                preferred_setup_families={family},
            )
            if feats.empty:
                continue
            feats = feats.loc[
                (feats["setup_family"].astype(str) == str(family))
                & (pd.to_numeric(feats.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0)
            ].copy()
            if feats.empty:
                continue
            feats["aetherflow_confidence"] = af._compute_probabilities(feats)
            try:
                from aetherflow_strategy import _selection_score, _regime_name_from_row
            except ImportError:
                from aetherflow_strategy import _selection_score, _regime_name_from_row  # noqa
            feats["manifold_regime_name"] = feats.apply(
                lambda row: _regime_name_from_row(row.to_dict()), axis=1
            )
            feats["selection_score"] = [
                _selection_score(
                    float(c),
                    af._policy_for_family(family, row_dict) or {},
                )
                for row_dict, c in zip(feats.to_dict("records"), feats["aetherflow_confidence"].tolist())
            ]
            candidate_frames.append(feats)
        if not candidate_frames:
            raise SystemExit("AF precompute returned 0 rows — strategy didn't fire on this window")
        merged = pd.concat(candidate_frames, axis=0).sort_index()
        sigs = af._select_signal_rows(merged)
    else:
        base_features_path = resolve_full_manifold_base_features_path(
            args.base_features or af.backtest_base_features_path or DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
        )
        logger.info(
            "AF replay mode: using full precomputed manifold base cache from %s",
            base_features_path,
        )
        base_features = load_base_features_for_window(
            base_features_path,
            start_ts,
            end_ts,
        )
        if base_features.empty:
            raise SystemExit(
                f"No precomputed manifold base rows found in {base_features_path} for {start_ts} -> {end_ts}"
            )
        sigs = af.build_backtest_df_from_base_features(
            base_features,
            start_time=start_ts,
            end_time=end_ts,
        )
    dt_pre = (datetime.now() - t0).total_seconds()
    logger.info("AF precompute done in %.1fs: %d candidate signal rows", dt_pre, len(sigs))
    if sigs.empty:
        raise SystemExit("AF precompute returned 0 rows after selection")

    # Trim to actual replay window (precompute may include warmup hits)
    sigs = sigs.loc[(sigs.index >= start_ts) & (sigs.index <= end_ts)]
    logger.info("After trim to [%s, %s]: %d signals", start_ts, end_ts, len(sigs))

    if sigs.empty:
        raise SystemExit("No signals in target window after trim")

    # Build a sorted bar timestamp index for fast lookup
    bar_index = bars.index
    bar_index_ns = bar_index.asi8

    # Each row in `sigs` is a "signal payload" dict-row.  But it does NOT yet
    # contain tp_dist/sl_dist/horizon — those come from `_signal_from_row`.
    # We re-call that to get the live-bot signal shape.
    threshold = float(args.threshold_override) if args.threshold_override is not None else af.threshold

    # NOTE: rows in `sigs` are ALREADY the output of _signal_from_row — they
    # contain tp_dist/sl_dist/horizon_bars/side/confidence directly. Use those
    # fields; do NOT re-call _signal_from_row (different input shape).
    closed = []
    skipped_thr = 0
    skipped_no_bar = 0
    skipped_live_window = 0
    for ts, row in sigs.iterrows():
        if (not bool(args.ignore_live_entry_window)) and _signal_time_blocked_live_window(ts):
            skipped_live_window += 1
            continue
        prob = float(row.get("aetherflow_confidence", row.get("confidence", 0.0)) or 0.0)
        if prob < threshold:
            skipped_thr += 1
            continue
        side = str(row.get("side", "")).upper()
        tp_dist = float(row.get("tp_dist", 0.0) or 0.0)
        sl_dist = float(row.get("sl_dist", 0.0) or 0.0)
        horizon_bars = int(row.get("horizon_bars", 0) or 0)
        if not side or tp_dist <= 0 or sl_dist <= 0 or horizon_bars <= 0:
            skipped_no_bar += 1
            continue
        try:
            entry_idx = int(bar_index.searchsorted(ts))
        except Exception:
            skipped_no_bar += 1
            continue
        if entry_idx + 1 >= len(bars):
            skipped_no_bar += 1
            continue
        sim = simulate_bracket(
            bars,
            entry_idx,
            side,
            tp_dist,
            sl_dist,
            horizon_bars,
            bool(row.get("use_horizon_time_stop", False)),
        )
        sig_size = int(row.get("size", af.size) or af.size)
        size = int(args.size_override or sig_size)
        per_contract_pnl_dollars = sim["pnl_points"] * POINT_VALUE
        pnl_dollars = per_contract_pnl_dollars * size

        exit_idx = sim["exit_idx"]
        exit_ts = bar_index[exit_idx]
        entry_ts_used = bar_index[entry_idx + 1]
        closed.append({
            "strategy": "AetherFlow",
            "sub_strategy": str(row.get("aetherflow_setup_family") or ""),
            "side": side,
            "size": size,
            "entry_price": float(sim["entry_price"]),
            "exit_price": float(sim["exit_price"]),
            "entry_time": entry_ts_used.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "source": sim["exit_source"],
            "pnl_points": float(sim["pnl_points"]),
            "pnl_dollars": float(pnl_dollars),
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "horizon_bars": horizon_bars,
            "confidence": prob,
            "threshold": float(threshold),
            "regime": row.get("aetherflow_regime"),
        })
    logger.info(
        "Loop summary: scored=%d trades=%d skipped_live_window=%d skipped_thr=%d skipped_other=%d",
        len(sigs),
        len(closed),
        skipped_live_window,
        skipped_thr,
        skipped_no_bar,
    )

    out_dir = ROOT / "backtest_reports" / "af_fast_replay" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "closed_trades.json").write_text(json.dumps(closed, indent=2))

    # Summary
    if closed:
        wins = sum(1 for t in closed if t["pnl_dollars"] > 0)
        pnls = np.array([t["pnl_dollars"] for t in closed])
        summary = {
            "range_start": str(start_ts),
            "range_end": str(end_ts),
            "signals_pre_threshold": int(len(sigs)),
            "trades_after_threshold": int(len(closed)),
            "skipped_live_entry_window": int(skipped_live_window),
            "wins": int(wins),
            "losses": int(len(closed) - wins),
            "winrate": float(wins / len(closed)),
            "net_pnl": float(pnls.sum()),
            "mean_pnl": float(pnls.mean()),
            "median_pnl": float(np.median(pnls)),
            "max_win": float(pnls.max()),
            "max_loss": float(pnls.min()),
            "exit_source_counts": {
                k: sum(1 for t in closed if t["source"] == k)
                for k in {t["source"] for t in closed}
            },
            "regime_counts": {
                str(k): sum(1 for t in closed if t.get("regime") == k)
                for k in {t.get("regime") for t in closed}
            },
        }
    else:
        summary = {"trades_after_threshold": 0, "note": "no trades after threshold"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logger.info("WROTE %s (%d trades)", out_dir / "closed_trades.json", len(closed))
    logger.info("Summary: %s", json.dumps(summary, default=str)[:600])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt


ENTRY_SHAPE_COLUMNS = [
    "de3_entry_ret1_atr",
    "de3_entry_body_pos1",
    "de3_entry_lower_wick_ratio",
    "de3_entry_upper_wick_ratio",
    "de3_entry_upper1_ratio",
    "de3_entry_body1_ratio",
    "de3_entry_close_pos1",
    "de3_entry_flips5",
    "de3_entry_down3",
    "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20",
    "de3_entry_atr14",
    # 30-bar chart-context features (added to let DE3 v4 learn the
    # bounce/dip-fade pattern natively that filter F currently catches).
    # Validated conditionally on 2025 136-day + April 2026 OOS.
    "de3_entry_velocity_30",      # pts/min over last 30 bars (1-min bars)
    "de3_entry_dist_low30_atr",   # (entry - 30-bar low) / ATR14
    "de3_entry_dist_high30_atr",  # (30-bar high - entry) / ATR14
    "de3_entry_ret30_atr",        # 30-bar return / ATR14 (scale-free velocity)
]


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _resolve_output(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _parse_bool(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series([], dtype=bool)
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _parse_et_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(bt.NY_TZ)


def _load_chosen_decisions(path: Path) -> pd.DataFrame:
    usecols = [
        "decision_id",
        "timestamp",
        "chosen",
        "rank",
        "timeframe",
        "strategy_type",
        "side_considered",
        "de3_v4_selected_lane",
        "de3_v4_route_confidence",
        "edge_points",
        "runtime_rank_score",
        "structural_score",
        "de3_v4_selected_variant_id",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    chosen_mask = _parse_bool(df.get("chosen", pd.Series(False, index=df.index)))
    out = df.loc[chosen_mask].copy()
    out["decision_id"] = out.get("decision_id", "").astype(str)
    out["timestamp_et"] = _parse_et_timestamps(out.get("timestamp", pd.Series([], dtype=object)))
    out["timeframe"] = out.get("timeframe", "").astype(str).str.strip()
    out["strategy_type"] = out.get("strategy_type", "").astype(str).str.strip()
    out["side_considered"] = out.get("side_considered", "").astype(str).str.strip().str.lower()
    out["de3_v4_selected_lane"] = out.get("de3_v4_selected_lane", "").astype(str).str.strip()
    out["de3_v4_selected_variant_id"] = out.get("de3_v4_selected_variant_id", "").astype(str).str.strip()
    out["rank"] = pd.to_numeric(out.get("rank", 0), errors="coerce").fillna(0).astype(int)
    out = out[out["timestamp_et"].notna()].copy()
    out = out.sort_values(["decision_id", "rank"], kind="mergesort")
    out = out.drop_duplicates(subset=["decision_id"], keep="first")
    return out


def _prepare_symbol_df(source_path: Path, start_time, end_time, symbol_mode: str, symbol_method: str) -> pd.DataFrame:
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No rows found in the source file.")
    source_df = df[df.index <= end_time]
    if source_df.empty:
        raise SystemExit("No rows found before the requested end time.")
    symbol_df = source_df
    if "symbol" in source_df.columns:
        if symbol_mode != "single":
            symbol_df, _, _ = bt.apply_symbol_mode(source_df, symbol_mode, symbol_method)
            if symbol_df.empty:
                raise SystemExit("No rows found after auto symbol selection.")
        else:
            preferred_symbol = bt.CONFIG.get("TARGET_SYMBOL")
            symbol = bt.choose_symbol(source_df, preferred_symbol)
            symbol_df = source_df[source_df["symbol"] == symbol]
            if symbol_df.empty:
                raise SystemExit("No rows found for the selected symbol.")
        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    lookback_start = start_time - pd.Timedelta(days=2)
    symbol_df = symbol_df[symbol_df.index >= lookback_start].copy()
    return symbol_df


def _compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ENTRY_SHAPE_COLUMNS, index=df.index)
    open_s = pd.to_numeric(df["open"], errors="coerce")
    high_s = pd.to_numeric(df["high"], errors="coerce")
    low_s = pd.to_numeric(df["low"], errors="coerce")
    close_s = pd.to_numeric(df["close"], errors="coerce")
    volume_s = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce")

    prev_open = open_s.shift(1)
    prev_high = high_s.shift(1)
    prev_low = low_s.shift(1)
    prev_close = close_s.shift(1)
    prev_volume = volume_s.shift(1)

    bar_range = (prev_high - prev_low).clip(lower=1e-9)
    body = prev_close - prev_open

    prev_close_for_tr = close_s.shift(1)
    tr = pd.concat(
        [
            (high_s - low_s).abs(),
            (high_s - prev_close_for_tr).abs(),
            (low_s - prev_close_for_tr).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean().shift(1)

    close_diff = close_s.diff()
    sign = np.sign(close_diff)
    sign_prev = sign.shift(1)
    flips5 = ((sign * sign_prev) < 0).astype(float).rolling(4, min_periods=4).sum().shift(1)
    down3 = (close_diff < 0).astype(float).rolling(3, min_periods=3).sum().shift(1)

    range10 = (
        high_s.rolling(10, min_periods=10).max()
        - low_s.rolling(10, min_periods=10).min()
    ).shift(1)
    low5 = low_s.rolling(5, min_periods=5).min().shift(1)
    high5 = high_s.rolling(5, min_periods=5).max().shift(1)
    vol20 = volume_s.rolling(20, min_periods=20).mean().shift(1)

    # 30-bar chart-context (for bounce/dip-fade pattern learning).
    low30 = low_s.rolling(30, min_periods=30).min().shift(1)
    high30 = high_s.rolling(30, min_periods=30).max().shift(1)
    ret30_pts = prev_close - close_s.shift(30)

    feature_df = pd.DataFrame(
        {
            "de3_entry_ret1_atr": body / atr14,
            "de3_entry_body_pos1": body / bar_range,
            "de3_entry_lower_wick_ratio": (np.minimum(prev_open, prev_close) - prev_low) / bar_range,
            "de3_entry_upper_wick_ratio": (prev_high - np.maximum(prev_open, prev_close)) / bar_range,
            "de3_entry_upper1_ratio": (prev_high - np.maximum(prev_open, prev_close)) / bar_range,
            "de3_entry_body1_ratio": body.abs() / bar_range,
            "de3_entry_close_pos1": (prev_close - prev_low) / bar_range,
            "de3_entry_flips5": flips5,
            "de3_entry_down3": down3,
            "de3_entry_range10_atr": range10 / atr14,
            "de3_entry_dist_low5_atr": (prev_close - low5) / atr14,
            "de3_entry_dist_high5_atr": (high5 - prev_close) / atr14,
            "de3_entry_vol1_rel20": prev_volume / vol20,
            "de3_entry_atr14": atr14,
            # 30-bar chart context — lets the model see "bouncing dead-cat"
            # and "failed dip" patterns the 5-bar features can't reach.
            "de3_entry_velocity_30": ret30_pts / 30.0,  # raw pts per 1-min bar
            "de3_entry_dist_low30_atr": (prev_close - low30) / atr14,
            "de3_entry_dist_high30_atr": (high30 - prev_close) / atr14,
            "de3_entry_ret30_atr": ret30_pts / atr14,
        },
        index=df.index,
    )

    # Need 30 bars of history for the 30-bar features + 14 for ATR14 +
    # the shift-1 = minimum 31 valid rows, but 41 was the previous margin
    # to accommodate rolling warm-ups for flips5 etc. 30+14 = 44, so bump.
    required_rows = 44
    if len(feature_df) > 0:
        valid_mask = pd.Series(np.arange(len(feature_df)) >= (required_rows - 1), index=feature_df.index)
        feature_df.loc[~valid_mask, ENTRY_SHAPE_COLUMNS] = np.nan
    feature_df.index.name = "timestamp_et"
    return feature_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a chosen-row DE3 current-pool dataset with reconstructed "
            "entry-shape features from the outright 1m source."
        )
    )
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME, help="Parquet/CSV source path.")
    parser.add_argument(
        "--decisions-csv",
        default="reports/de3_current_pool_2011_2024.csv",
        help="Existing DE3 current-pool decisions CSV.",
    )
    parser.add_argument(
        "--out-csv",
        default="reports/de3_current_pool_2011_2024_chosen_shape.csv",
        help="Output CSV path for chosen decisions enriched with shape features.",
    )
    parser.add_argument(
        "--symbol-mode",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"),
        help="single, auto_by_day, or another supported backtest symbol mode.",
    )
    parser.add_argument(
        "--symbol-method",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"),
        help="Auto symbol selection method.",
    )
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    decisions_path = _resolve_path(str(args.decisions_csv))
    out_path = _resolve_output(str(args.out_csv))

    chosen_df = _load_chosen_decisions(decisions_path)
    if chosen_df.empty:
        raise SystemExit("No chosen rows found in the decisions CSV.")

    start_time = chosen_df["timestamp_et"].min()
    end_time = chosen_df["timestamp_et"].max()
    symbol_df = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(args.symbol_mode or "single").strip().lower(),
        str(args.symbol_method or "volume").strip().lower(),
    )
    feature_df = _compute_feature_frame(symbol_df)

    merged = chosen_df.merge(
        feature_df.reset_index(),
        on="timestamp_et",
        how="left",
    )
    coverage = float(merged["de3_entry_close_pos1"].notna().mean()) if not merged.empty else 0.0
    merged = merged.drop(columns=["timestamp_et"], errors="ignore")
    merged.to_csv(out_path, index=False)

    summary = {
        "source_path": str(source_path),
        "decisions_csv": str(decisions_path),
        "out_csv": str(out_path),
        "rows": int(len(merged)),
        "feature_coverage": float(coverage),
        "start_time": str(start_time),
        "end_time": str(end_time),
        "symbol_mode": str(args.symbol_mode or "").strip().lower(),
        "symbol_method": str(args.symbol_method or "").strip().lower(),
        "feature_columns": list(ENTRY_SHAPE_COLUMNS),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"rows={summary['rows']}")
    print(f"feature_coverage={summary['feature_coverage']:.4f}")
    print(f"out_csv={out_path}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()

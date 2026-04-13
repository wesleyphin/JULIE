import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from de3_v4_decision_policy_trainer import _directionless_strategy_style
from tools.build_de3_chosen_shape_dataset import _compute_feature_frame


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _resolve_output_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    try:
        raw = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(raw):
        return float(default)
    return float(raw)


def _force_flat_deadline(entry_ts: pd.Timestamp) -> pd.Timestamp:
    deadline = entry_ts.normalize() + pd.Timedelta(
        hours=int(bt.BACKTEST_FORCE_FLAT_HOUR_ET),
        minutes=int(bt.BACKTEST_FORCE_FLAT_MINUTE_ET),
    )
    if entry_ts > deadline:
        deadline += pd.Timedelta(days=1)
    return deadline


def _prepare_symbol_df(
    source_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    symbol_mode: str,
    symbol_method: str,
) -> pd.DataFrame:
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No source rows found.")
    selection_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    source_df = df[(df.index >= selection_start) & (df.index <= end_time)].copy()
    if source_df.empty:
        raise SystemExit("No source rows found inside requested range.")
    symbol_df, _, _ = bt.apply_symbol_mode(source_df, symbol_mode, symbol_method)
    if symbol_df.empty:
        raise SystemExit("Auto-by-day symbol selection produced no rows.")
    if "symbol" in symbol_df.columns:
        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    return symbol_df.sort_index(kind="mergesort").copy()


def _pick_top_side_row(grp: pd.DataFrame, side_name: str) -> Optional[pd.Series]:
    side_df = grp[grp["side_considered"] == side_name].copy()
    if side_df.empty:
        return None
    side_df["candidate_rank_num"] = pd.to_numeric(
        side_df.get("candidate_rank_before_adjustments"),
        errors="coerce",
    )
    side_df["rank_num"] = pd.to_numeric(side_df.get("rank"), errors="coerce")
    side_df["final_score_num"] = pd.to_numeric(side_df.get("final_score"), errors="coerce")
    side_df["edge_points_num"] = pd.to_numeric(side_df.get("edge_points"), errors="coerce")
    side_df["structural_score_num"] = pd.to_numeric(side_df.get("structural_score"), errors="coerce")
    side_df["rank_key"] = side_df["candidate_rank_num"].where(side_df["candidate_rank_num"] > 0, side_df["rank_num"])
    side_df["rank_key"] = side_df["rank_key"].where(side_df["rank_key"] > 0, 999999)
    side_df = side_df.sort_values(
        ["rank_key", "final_score_num", "edge_points_num", "structural_score_num"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    return side_df.iloc[0]


def _simulate_trade_points_fast(
    *,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    ts_index: pd.DatetimeIndex,
    decision_pos: int,
    side_name: str,
    sl_dist: float,
    tp_dist: float,
) -> Optional[Dict[str, Any]]:
    entry_pos = int(decision_pos) + 1
    if entry_pos >= len(open_arr):
        return None
    entry_price = _coerce_float(open_arr[entry_pos], float("nan"))
    if not math.isfinite(entry_price):
        return None
    max_horizon = 240
    last_pos = min(len(open_arr) - 1, entry_pos + max_horizon)
    if bool(bt.BACKTEST_FORCE_FLAT_AT_TIME):
        deadline = _force_flat_deadline(pd.Timestamp(ts_index[entry_pos]))
        deadline_pos = int(ts_index.searchsorted(deadline, side="right")) - 1
        if deadline_pos < entry_pos:
            return None
        last_pos = min(last_pos, deadline_pos)
    highs = high_arr[entry_pos : last_pos + 1]
    lows = low_arr[entry_pos : last_pos + 1]
    side_upper = str(side_name or "").strip().upper()
    if side_upper == "LONG":
        tp_level = float(entry_price) + float(tp_dist)
        sl_level = float(entry_price) - float(sl_dist)
        tp_hits = np.flatnonzero(highs >= tp_level)
        sl_hits = np.flatnonzero(lows <= sl_level)
    else:
        tp_level = float(entry_price) - float(tp_dist)
        sl_level = float(entry_price) + float(sl_dist)
        tp_hits = np.flatnonzero(lows <= tp_level)
        sl_hits = np.flatnonzero(highs >= sl_level)
    first_tp = int(tp_hits[0]) if tp_hits.size else None
    first_sl = int(sl_hits[0]) if sl_hits.size else None
    if first_tp is not None and first_sl is not None:
        if first_tp == first_sl:
            pnl_points = -float(sl_dist)
        elif first_tp < first_sl:
            pnl_points = float(tp_dist)
        else:
            pnl_points = -float(sl_dist)
    elif first_tp is not None:
        pnl_points = float(tp_dist)
    elif first_sl is not None:
        pnl_points = -float(sl_dist)
    else:
        exit_price = _coerce_float(close_arr[last_pos], float("nan"))
        if not math.isfinite(exit_price):
            return None
        pnl_points = float(bt.compute_pnl_points(side_upper, float(entry_price), float(exit_price)))
    return {
        "entry_pos": int(entry_pos),
        "entry_time": str(pd.Timestamp(ts_index[entry_pos]).isoformat()),
        "entry_price": float(entry_price),
        "pnl_points": float(pnl_points),
    }


def build_dataset(
    *,
    decisions_path: Path,
    source_path: Path,
    output_csv: Path,
    output_summary: Path,
    symbol_mode: str,
    symbol_method: str,
    start: str = "",
    end: str = "",
) -> None:
    decision_usecols = [
        "decision_id",
        "timestamp",
        "session",
        "side_considered",
        "chosen",
        "rank",
        "candidate_rank_before_adjustments",
        "sub_strategy",
        "timeframe",
        "strategy_type",
        "ctx_session_substate",
        "ctx_hour_et",
        "ctx_volatility_regime",
        "ctx_price_location",
        "final_score",
        "edge_points",
        "structural_score",
        "sl",
        "tp",
        "de3_v4_route_confidence",
    ]
    decisions = pd.read_csv(decisions_path, usecols=lambda c: c in decision_usecols)
    if decisions.empty:
        raise SystemExit("No decision rows loaded.")
    decisions["decision_id"] = decisions["decision_id"].astype(str)
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], errors="coerce", utc=True).dt.tz_convert(bt.NY_TZ)
    decisions["side_considered"] = decisions["side_considered"].astype(str).str.strip().str.lower()
    decisions["chosen"] = decisions["chosen"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    decisions = decisions[decisions["timestamp"].notna()].copy()
    decisions = decisions[decisions["side_considered"].isin({"long", "short"})].copy()
    if start:
        start_ts = pd.Timestamp(str(start))
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(bt.NY_TZ)
        else:
            start_ts = start_ts.tz_convert(bt.NY_TZ)
        decisions = decisions[decisions["timestamp"] >= start_ts].copy()
    if end:
        end_ts = pd.Timestamp(str(end))
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(bt.NY_TZ)
        else:
            end_ts = end_ts.tz_convert(bt.NY_TZ)
        decisions = decisions[decisions["timestamp"] <= end_ts].copy()
    if decisions.empty:
        raise SystemExit("No valid decision rows after normalization.")

    start_time = decisions["timestamp"].min()
    end_time = decisions["timestamp"].max() + pd.Timedelta(days=2)
    market_df = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        symbol_mode=symbol_mode,
        symbol_method=symbol_method,
    )
    feature_df = _compute_feature_frame(market_df)
    ts_index = market_df.index
    ts_to_pos = {pd.Timestamp(ts): idx for idx, ts in enumerate(ts_index)}
    open_arr = market_df["open"].to_numpy(dtype=float, copy=False)
    high_arr = market_df["high"].to_numpy(dtype=float, copy=False)
    low_arr = market_df["low"].to_numpy(dtype=float, copy=False)
    close_arr = market_df["close"].to_numpy(dtype=float, copy=False)

    rows: List[Dict[str, Any]] = []
    missing_market_ts = 0
    skipped_no_entry = 0
    one_side_decisions = 0
    both_side_decisions = 0
    for decision_id, grp in decisions.groupby("decision_id", sort=False):
        long_row = _pick_top_side_row(grp, "long")
        short_row = _pick_top_side_row(grp, "short")
        if long_row is None and short_row is None:
            continue
        if long_row is not None and short_row is not None:
            both_side_decisions += 1
        else:
            one_side_decisions += 1
        base_row = long_row if long_row is not None else short_row
        decision_ts = pd.Timestamp(base_row.get("timestamp"))
        decision_pos = ts_to_pos.get(decision_ts)
        if decision_pos is None:
            missing_market_ts += 1
            continue
        if decision_ts not in feature_df.index:
            missing_market_ts += 1
            continue
        local_features = feature_df.loc[decision_ts]

        def _sim_for(row: Optional[pd.Series], side_text: str) -> Optional[Dict[str, Any]]:
            if row is None:
                return None
            sl_dist = max(_coerce_float(row.get("sl"), 0.0), float(bt.MIN_SL))
            tp_dist = max(_coerce_float(row.get("tp"), 0.0), float(bt.MIN_TP))
            return _simulate_trade_points_fast(
                open_arr=open_arr,
                high_arr=high_arr,
                low_arr=low_arr,
                close_arr=close_arr,
                ts_index=ts_index,
                decision_pos=int(decision_pos),
                side_name=side_text,
                sl_dist=float(sl_dist),
                tp_dist=float(tp_dist),
            )

        long_sim = _sim_for(long_row, "LONG")
        short_sim = _sim_for(short_row, "SHORT")
        if long_row is not None and long_sim is None:
            skipped_no_entry += 1
            continue
        if short_row is not None and short_sim is None:
            skipped_no_entry += 1
            continue

        chosen_rows = grp[grp["chosen"] == True]
        chosen_side = ""
        chosen_sub_strategy = ""
        if not chosen_rows.empty:
            chosen_row = chosen_rows.iloc[0]
            chosen_side = str(chosen_row.get("side_considered", "") or "").strip().lower()
            chosen_sub_strategy = str(chosen_row.get("sub_strategy", "") or "").strip()

        long_points = float(long_sim["pnl_points"]) if isinstance(long_sim, dict) else float("nan")
        short_points = float(short_sim["pnl_points"]) if isinstance(short_sim, dict) else float("nan")
        long_available = bool(isinstance(long_row, pd.Series) and isinstance(long_sim, dict))
        short_available = bool(isinstance(short_row, pd.Series) and isinstance(short_sim, dict))
        best_action = "no_trade"
        best_value = 0.0
        if long_available and math.isfinite(long_points) and long_points > best_value:
            best_action = "long"
            best_value = float(long_points)
        if short_available and math.isfinite(short_points) and short_points > best_value:
            best_action = "short"
            best_value = float(short_points)
        side_pattern = "both" if long_available and short_available else ("long_only" if long_available else "short_only")
        long_style = _directionless_strategy_style(long_row.get("strategy_type", "") if long_row is not None else "")
        short_style = _directionless_strategy_style(short_row.get("strategy_type", "") if short_row is not None else "")
        long_rank = _coerce_float(
            long_row.get("candidate_rank_before_adjustments", long_row.get("rank")) if long_row is not None else float("nan"),
            float("nan"),
        )
        short_rank = _coerce_float(
            short_row.get("candidate_rank_before_adjustments", short_row.get("rank")) if short_row is not None else float("nan"),
            float("nan"),
        )
        long_score = _coerce_float(long_row.get("final_score") if long_row is not None else float("nan"), float("nan"))
        short_score = _coerce_float(short_row.get("final_score") if short_row is not None else float("nan"), float("nan"))
        long_edge = _coerce_float(long_row.get("edge_points") if long_row is not None else float("nan"), float("nan"))
        short_edge = _coerce_float(short_row.get("edge_points") if short_row is not None else float("nan"), float("nan"))
        long_struct = _coerce_float(long_row.get("structural_score") if long_row is not None else float("nan"), float("nan"))
        short_struct = _coerce_float(short_row.get("structural_score") if short_row is not None else float("nan"), float("nan"))
        baseline_points = 0.0
        if chosen_side == "long" and long_available and math.isfinite(long_points):
            baseline_points = float(long_points)
        elif chosen_side == "short" and short_available and math.isfinite(short_points):
            baseline_points = float(short_points)
        rows.append(
            {
                "decision_id": str(decision_id),
                "timestamp": str(decision_ts.isoformat()),
                "year": int(decision_ts.year),
                "session": str(base_row.get("session", "") or "").strip().lower(),
                "ctx_session_substate": str(base_row.get("ctx_session_substate", "") or "").strip().lower(),
                "ctx_hour_et": pd.to_numeric(base_row.get("ctx_hour_et"), errors="coerce"),
                "ctx_volatility_regime": str(base_row.get("ctx_volatility_regime", "") or "").strip().lower(),
                "ctx_price_location": pd.to_numeric(base_row.get("ctx_price_location"), errors="coerce"),
                "side_pattern": str(side_pattern),
                "chosen_side": str(chosen_side),
                "chosen_sub_strategy": str(chosen_sub_strategy),
                "baseline_points": float(baseline_points),
                "best_action": str(best_action),
                "best_action_points": float(best_value),
                "long_available": bool(long_available),
                "short_available": bool(short_available),
                "long_sub_strategy": str(long_row.get("sub_strategy", "") or "") if long_row is not None else "",
                "long_timeframe": str(long_row.get("timeframe", "") or "").strip().lower() if long_row is not None else "",
                "long_strategy_type": str(long_row.get("strategy_type", "") or "").strip().lower() if long_row is not None else "",
                "long_strategy_style": str(long_style),
                "long_rank": float(long_rank) if math.isfinite(long_rank) else np.nan,
                "long_final_score": float(long_score) if math.isfinite(long_score) else np.nan,
                "long_edge_points": float(long_edge) if math.isfinite(long_edge) else np.nan,
                "long_structural_score": float(long_struct) if math.isfinite(long_struct) else np.nan,
                "long_route_confidence": pd.to_numeric(long_row.get("de3_v4_route_confidence"), errors="coerce") if long_row is not None else np.nan,
                "long_pnl_points": float(long_points) if math.isfinite(long_points) else np.nan,
                "short_sub_strategy": str(short_row.get("sub_strategy", "") or "") if short_row is not None else "",
                "short_timeframe": str(short_row.get("timeframe", "") or "").strip().lower() if short_row is not None else "",
                "short_strategy_type": str(short_row.get("strategy_type", "") or "").strip().lower() if short_row is not None else "",
                "short_strategy_style": str(short_style),
                "short_rank": float(short_rank) if math.isfinite(short_rank) else np.nan,
                "short_final_score": float(short_score) if math.isfinite(short_score) else np.nan,
                "short_edge_points": float(short_edge) if math.isfinite(short_edge) else np.nan,
                "short_structural_score": float(short_struct) if math.isfinite(short_struct) else np.nan,
                "short_route_confidence": pd.to_numeric(short_row.get("de3_v4_route_confidence"), errors="coerce") if short_row is not None else np.nan,
                "short_pnl_points": float(short_points) if math.isfinite(short_points) else np.nan,
                "de3_entry_ret1_atr": pd.to_numeric(local_features.get("de3_entry_ret1_atr"), errors="coerce"),
                "de3_entry_body_pos1": pd.to_numeric(local_features.get("de3_entry_body_pos1"), errors="coerce"),
                "de3_entry_lower_wick_ratio": pd.to_numeric(local_features.get("de3_entry_lower_wick_ratio"), errors="coerce"),
                "de3_entry_upper_wick_ratio": pd.to_numeric(local_features.get("de3_entry_upper_wick_ratio"), errors="coerce"),
                "de3_entry_upper1_ratio": pd.to_numeric(local_features.get("de3_entry_upper1_ratio"), errors="coerce"),
                "de3_entry_body1_ratio": pd.to_numeric(local_features.get("de3_entry_body1_ratio"), errors="coerce"),
                "de3_entry_close_pos1": pd.to_numeric(local_features.get("de3_entry_close_pos1"), errors="coerce"),
                "de3_entry_flips5": pd.to_numeric(local_features.get("de3_entry_flips5"), errors="coerce"),
                "de3_entry_down3": pd.to_numeric(local_features.get("de3_entry_down3"), errors="coerce"),
                "de3_entry_range10_atr": pd.to_numeric(local_features.get("de3_entry_range10_atr"), errors="coerce"),
                "de3_entry_dist_low5_atr": pd.to_numeric(local_features.get("de3_entry_dist_low5_atr"), errors="coerce"),
                "de3_entry_dist_high5_atr": pd.to_numeric(local_features.get("de3_entry_dist_high5_atr"), errors="coerce"),
                "de3_entry_vol1_rel20": pd.to_numeric(local_features.get("de3_entry_vol1_rel20"), errors="coerce"),
                "de3_entry_atr14": pd.to_numeric(local_features.get("de3_entry_atr14"), errors="coerce"),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("Decision-side dataset build produced no rows.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    summary = {
        "decisions_csv": str(decisions_path),
        "source_path": str(source_path),
        "rows_total": int(len(decisions)),
        "decision_count": int(out_df["decision_id"].nunique()),
        "one_side_decisions": int(one_side_decisions),
        "both_side_decisions": int(both_side_decisions),
        "missing_market_timestamp_count": int(missing_market_ts),
        "skipped_no_entry_count": int(skipped_no_entry),
        "chosen_side_distribution": out_df["chosen_side"].astype(str).value_counts(dropna=False).to_dict(),
        "best_action_distribution": out_df["best_action"].astype(str).value_counts(dropna=False).to_dict(),
        "baseline_points_total": float(pd.to_numeric(out_df["baseline_points"], errors="coerce").fillna(0.0).sum()),
        "best_action_points_total": float(pd.to_numeric(out_df["best_action_points"], errors="coerce").fillna(0.0).sum()),
        "potential_points_uplift": float(
            pd.to_numeric(out_df["best_action_points"], errors="coerce").fillna(0.0).sum()
            - pd.to_numeric(out_df["baseline_points"], errors="coerce").fillna(0.0).sum()
        ),
    }
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"decision_side_dataset={output_csv}")
    print(f"summary_json={output_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DE3 decision-level side dataset with local tape features and simulated side outcomes."
    )
    parser.add_argument("--decisions", default="reports/de3_current_pool_2011_2024.csv")
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument("--output-csv", default="reports/de3_decision_side_dataset.csv")
    parser.add_argument("--output-summary", default="reports/de3_decision_side_dataset_summary.json")
    parser.add_argument(
        "--symbol-mode",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"),
    )
    parser.add_argument(
        "--symbol-method",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"),
    )
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    args = parser.parse_args()

    build_dataset(
        decisions_path=_resolve_path(str(args.decisions)),
        source_path=_resolve_path(str(args.source)),
        output_csv=_resolve_output_path(str(args.output_csv)),
        output_summary=_resolve_output_path(str(args.output_summary)),
        symbol_mode=str(args.symbol_mode or "single").strip().lower(),
        symbol_method=str(args.symbol_method or "volume").strip().lower(),
        start=str(args.start or "").strip(),
        end=str(args.end or "").strip(),
    )


if __name__ == "__main__":
    main()

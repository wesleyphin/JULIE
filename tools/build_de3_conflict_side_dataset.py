import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _force_flat_deadline(entry_ts: pd.Timestamp) -> pd.Timestamp:
    deadline = entry_ts.normalize() + pd.Timedelta(
        hours=int(bt.BACKTEST_FORCE_FLAT_HOUR_ET),
        minutes=int(bt.BACKTEST_FORCE_FLAT_MINUTE_ET),
    )
    if entry_ts > deadline:
        deadline += pd.Timedelta(days=1)
    return deadline


def _prepare_symbol_df(source_path: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No source rows found.")
    selection_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    source_df = df[(df.index >= selection_start) & (df.index <= end_time)].copy()
    if source_df.empty:
        raise SystemExit("No source rows found inside requested range.")
    symbol_df, _, _ = bt.apply_symbol_mode(source_df, "auto_by_day", "volume")
    if symbol_df.empty:
        raise SystemExit("Auto-by-day symbol selection produced no rows.")
    if "symbol" in symbol_df.columns:
        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    return symbol_df.sort_index(kind="mergesort").copy()


def _pick_top_side_row(grp: pd.DataFrame, side_name: str) -> Optional[pd.Series]:
    side_df = grp[grp["side_considered"] == side_name].copy()
    if side_df.empty:
        return None
    side_df["rank_num"] = pd.to_numeric(side_df.get("rank"), errors="coerce")
    side_df["final_score_num"] = pd.to_numeric(side_df.get("final_score"), errors="coerce")
    side_df["edge_points_num"] = pd.to_numeric(side_df.get("edge_points"), errors="coerce")
    side_df = side_df.sort_values(
        ["rank_num", "final_score_num", "edge_points_num"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return side_df.iloc[0]


def _simulate_candidate(
    *,
    market_df: pd.DataFrame,
    ts_to_pos: Dict[pd.Timestamp, int],
    decision_ts: pd.Timestamp,
    side_name: str,
    sl_dist: float,
    tp_dist: float,
) -> Optional[Dict[str, Any]]:
    decision_pos = ts_to_pos.get(decision_ts)
    if decision_pos is None:
        return None
    entry_pos = int(decision_pos) + 1
    if entry_pos >= len(market_df):
        return None
    entry_bar = market_df.iloc[entry_pos]
    try:
        entry_price = float(entry_bar["open"])
    except Exception:
        return None
    if not pd.notna(entry_price):
        return None

    max_horizon = 240
    if bool(bt.BACKTEST_FORCE_FLAT_AT_TIME):
        deadline = _force_flat_deadline(pd.Timestamp(market_df.index[entry_pos]))
        last_pos = int(market_df.index.searchsorted(deadline, side="right")) - 1
        if last_pos < entry_pos:
            return None
        max_horizon = max(0, last_pos - entry_pos)

    pnl_points = bt.simulate_trade_points(
        market_df,
        entry_pos,
        str(side_name).strip().upper(),
        float(entry_price),
        float(max(sl_dist, bt.MIN_SL)),
        float(max(tp_dist, bt.MIN_TP)),
        int(max_horizon),
        True,
        "close",
    )
    return {
        "entry_pos": int(entry_pos),
        "entry_time": str(pd.Timestamp(market_df.index[entry_pos]).isoformat()),
        "entry_price": float(entry_price),
        "pnl_points": float(pnl_points),
    }


def build_dataset(
    *,
    decisions_path: Path,
    source_path: Path,
    output_csv: Path,
    output_summary: Path,
) -> None:
    decision_usecols = [
        "decision_id",
        "timestamp",
        "session",
        "side_considered",
        "chosen",
        "rank",
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
    ]
    decisions = pd.read_csv(decisions_path, usecols=lambda c: c in decision_usecols)
    if decisions.empty:
        raise SystemExit("No decision rows loaded.")
    decisions["decision_id"] = decisions["decision_id"].astype(str)
    decisions["timestamp"] = (
        pd.to_datetime(decisions["timestamp"], errors="coerce", utc=True)
        .dt.tz_convert(bt.NY_TZ)
    )
    decisions["side_considered"] = decisions["side_considered"].astype(str).str.strip().str.lower()
    decisions["chosen"] = decisions["chosen"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    decisions = decisions[decisions["timestamp"].notna()].copy()
    decisions = decisions[decisions["side_considered"].isin({"long", "short"})].copy()
    if decisions.empty:
        raise SystemExit("No valid decision rows after normalization.")

    side_counts = decisions.groupby("decision_id", dropna=False)["side_considered"].nunique()
    both_side_ids = set(side_counts[side_counts > 1].index.astype(str).tolist())
    conflict_decisions = decisions[decisions["decision_id"].isin(both_side_ids)].copy()
    if conflict_decisions.empty:
        raise SystemExit("No both-side DE3 decisions found.")

    start_time = conflict_decisions["timestamp"].min()
    end_time = conflict_decisions["timestamp"].max() + pd.Timedelta(days=2)
    market_df = _prepare_symbol_df(source_path, start_time, end_time)
    ts_to_pos = {pd.Timestamp(ts): idx for idx, ts in enumerate(market_df.index)}

    rows: List[Dict[str, Any]] = []
    missing_market_ts = 0
    skipped_no_entry = 0
    for decision_id, grp in conflict_decisions.groupby("decision_id", sort=False):
        long_row = _pick_top_side_row(grp, "long")
        short_row = _pick_top_side_row(grp, "short")
        if long_row is None or short_row is None:
            continue
        decision_ts = pd.Timestamp(long_row.get("timestamp"))
        if decision_ts not in ts_to_pos:
            missing_market_ts += 1
            continue
        long_sim = _simulate_candidate(
            market_df=market_df,
            ts_to_pos=ts_to_pos,
            decision_ts=decision_ts,
            side_name="LONG",
            sl_dist=float(pd.to_numeric(long_row.get("sl"), errors="coerce") or 0.0),
            tp_dist=float(pd.to_numeric(long_row.get("tp"), errors="coerce") or 0.0),
        )
        short_sim = _simulate_candidate(
            market_df=market_df,
            ts_to_pos=ts_to_pos,
            decision_ts=decision_ts,
            side_name="SHORT",
            sl_dist=float(pd.to_numeric(short_row.get("sl"), errors="coerce") or 0.0),
            tp_dist=float(pd.to_numeric(short_row.get("tp"), errors="coerce") or 0.0),
        )
        if long_sim is None or short_sim is None:
            skipped_no_entry += 1
            continue

        chosen_rows = grp[grp["chosen"] == True]
        chosen_side = ""
        chosen_sub_strategy = ""
        if not chosen_rows.empty:
            chosen_row = chosen_rows.iloc[0]
            chosen_side = str(chosen_row.get("side_considered", "") or "").strip().lower()
            chosen_sub_strategy = str(chosen_row.get("sub_strategy", "") or "").strip()

        long_points = float(long_sim["pnl_points"])
        short_points = float(short_sim["pnl_points"])
        if long_points > short_points:
            best_side = "long"
        elif short_points > long_points:
            best_side = "short"
        else:
            best_side = "tie"

        best_positive_side = "none"
        if max(long_points, short_points) > 0.0:
            best_positive_side = "long" if long_points >= short_points else "short"

        rows.append(
            {
                "decision_id": str(decision_id),
                "timestamp": str(decision_ts.isoformat()),
                "session": str(long_row.get("session", "") or ""),
                "ctx_session_substate": str(long_row.get("ctx_session_substate", "") or ""),
                "ctx_hour_et": pd.to_numeric(long_row.get("ctx_hour_et"), errors="coerce"),
                "ctx_volatility_regime": str(long_row.get("ctx_volatility_regime", "") or ""),
                "ctx_price_location": pd.to_numeric(long_row.get("ctx_price_location"), errors="coerce"),
                "long_sub_strategy": str(long_row.get("sub_strategy", "") or ""),
                "long_timeframe": str(long_row.get("timeframe", "") or ""),
                "long_strategy_type": str(long_row.get("strategy_type", "") or ""),
                "long_rank": pd.to_numeric(long_row.get("rank"), errors="coerce"),
                "long_final_score": pd.to_numeric(long_row.get("final_score"), errors="coerce"),
                "long_edge_points": pd.to_numeric(long_row.get("edge_points"), errors="coerce"),
                "long_structural_score": pd.to_numeric(long_row.get("structural_score"), errors="coerce"),
                "long_sl": pd.to_numeric(long_row.get("sl"), errors="coerce"),
                "long_tp": pd.to_numeric(long_row.get("tp"), errors="coerce"),
                "long_entry_time": str(long_sim["entry_time"]),
                "long_entry_price": float(long_sim["entry_price"]),
                "long_pnl_points": float(long_points),
                "short_sub_strategy": str(short_row.get("sub_strategy", "") or ""),
                "short_timeframe": str(short_row.get("timeframe", "") or ""),
                "short_strategy_type": str(short_row.get("strategy_type", "") or ""),
                "short_rank": pd.to_numeric(short_row.get("rank"), errors="coerce"),
                "short_final_score": pd.to_numeric(short_row.get("final_score"), errors="coerce"),
                "short_edge_points": pd.to_numeric(short_row.get("edge_points"), errors="coerce"),
                "short_structural_score": pd.to_numeric(short_row.get("structural_score"), errors="coerce"),
                "short_sl": pd.to_numeric(short_row.get("sl"), errors="coerce"),
                "short_tp": pd.to_numeric(short_row.get("tp"), errors="coerce"),
                "short_entry_time": str(short_sim["entry_time"]),
                "short_entry_price": float(short_sim["entry_price"]),
                "short_pnl_points": float(short_points),
                "pnl_advantage_points": float(long_points - short_points),
                "best_side": str(best_side),
                "best_positive_side": str(best_positive_side),
                "chosen_side": str(chosen_side),
                "chosen_sub_strategy": str(chosen_sub_strategy),
                "chosen_side_is_best": bool(chosen_side in {"long", "short"} and chosen_side == best_side),
                "chosen_side_is_best_positive": bool(
                    chosen_side in {"long", "short"} and chosen_side == best_positive_side
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("Conflict dataset build produced no rows.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    chosen_best_rate = float(out_df["chosen_side_is_best"].fillna(False).astype(bool).mean())
    chosen_best_positive_rate = float(
        out_df["chosen_side_is_best_positive"].fillna(False).astype(bool).mean()
    )
    positive_opportunity = out_df[out_df["best_positive_side"].astype(str).isin({"long", "short"})].copy()
    chosen_side_points = []
    for row in out_df.to_dict("records"):
        chosen_side = str(row.get("chosen_side", "") or "")
        if chosen_side == "long":
            chosen_side_points.append(float(row.get("long_pnl_points", 0.0) or 0.0))
        elif chosen_side == "short":
            chosen_side_points.append(float(row.get("short_pnl_points", 0.0) or 0.0))
        else:
            chosen_side_points.append(0.0)
    optimal_points = [
        max(float(row.get("long_pnl_points", 0.0) or 0.0), float(row.get("short_pnl_points", 0.0) or 0.0), 0.0)
        for row in out_df.to_dict("records")
    ]
    summary = {
        "decisions_csv": str(decisions_path),
        "source_path": str(source_path),
        "rows_total": int(len(decisions)),
        "conflict_rows_total": int(len(conflict_decisions)),
        "conflict_decisions": int(len(out_df)),
        "both_side_decision_share": float(len(out_df) / max(1, side_counts.shape[0])),
        "missing_market_timestamp_count": int(missing_market_ts),
        "skipped_no_entry_count": int(skipped_no_entry),
        "chosen_side_is_best_rate": float(chosen_best_rate),
        "chosen_side_is_best_positive_rate": float(chosen_best_positive_rate),
        "chosen_side_total_points": float(sum(chosen_side_points)),
        "optimal_nonnegative_total_points": float(sum(optimal_points)),
        "potential_points_uplift_nonnegative": float(sum(optimal_points) - sum(chosen_side_points)),
        "best_side_distribution": out_df["best_side"].value_counts(dropna=False).to_dict(),
        "best_positive_side_distribution": out_df["best_positive_side"].value_counts(dropna=False).to_dict(),
    }
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"conflict_dataset={output_csv}")
    print(f"summary_json={output_summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DE3 both-side conflict dataset with simulated side outcomes.")
    parser.add_argument("--decisions", default="reports/de3_current_pool_2011_2024.csv")
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument("--output-csv", default="reports/de3_conflict_side_dataset.csv")
    parser.add_argument("--output-summary", default="reports/de3_conflict_side_dataset_summary.json")
    args = parser.parse_args()

    build_dataset(
        decisions_path=_resolve_path(str(args.decisions)),
        source_path=_resolve_path(str(args.source)),
        output_csv=(ROOT / str(args.output_csv)).resolve() if not Path(str(args.output_csv)).is_absolute() else Path(str(args.output_csv)).resolve(),
        output_summary=(ROOT / str(args.output_summary)).resolve() if not Path(str(args.output_summary)).is_absolute() else Path(str(args.output_summary)).resolve(),
    )


if __name__ == "__main__":
    main()

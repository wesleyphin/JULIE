import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _distribution(series: pd.Series) -> dict:
    s = _to_numeric(series).dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p10": float(q.get(0.10)),
        "p25": float(q.get(0.25)),
        "p50": float(q.get(0.50)),
        "p75": float(q.get(0.75)),
        "p90": float(q.get(0.90)),
        "max": float(s.max()),
    }


def _safe_float(val):
    try:
        out = float(val)
    except Exception:
        return np.nan
    if not np.isfinite(out):
        return np.nan
    return out


def _prepare_decisions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "edge_points",
        "edge_gap_points",
        "runtime_rank_score",
        "structural_score",
        "final_score",
        "bucket_score",
        "stop_like_share",
        "loss_share",
        "profitable_block_ratio",
        "worst_block_avg_pnl",
        "worst_block_pf",
        "realized_pnl",
        "mfe",
        "mae",
    ):
        if col in out.columns:
            out[col] = _to_numeric(out[col])
    if "chosen" in out.columns:
        out["chosen"] = _to_bool(out["chosen"])
    else:
        out["chosen"] = False
    if "abstained" in out.columns:
        out["abstained"] = _to_bool(out["abstained"])
    else:
        out["abstained"] = False
    if "decision_id" in out.columns:
        out["decision_id"] = out["decision_id"].astype(str)
    if "sub_strategy" in out.columns:
        out["sub_strategy"] = out["sub_strategy"].astype(str)
    if "realized_exit_type" in out.columns:
        out["realized_exit_type"] = out["realized_exit_type"].astype(str)
    return out


def _load_trade_attribution(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trade attribution file not found: {path}")
    df = pd.read_csv(path)
    if "decision_id" not in df.columns:
        raise ValueError("Trade attribution CSV must include 'decision_id'")
    out = df.copy()
    out["decision_id"] = out["decision_id"].astype(str)
    for col in ("trade_id", "realized_pnl", "mfe", "mae"):
        if col in out.columns:
            out[col] = _to_numeric(out[col])
    if "realized_exit_type" in out.columns:
        out["realized_exit_type"] = out["realized_exit_type"].astype(str)
    return out


def _merge_trade_outcomes(decisions: pd.DataFrame, trades: pd.DataFrame | None) -> pd.DataFrame:
    out = decisions.copy()
    if trades is None or trades.empty:
        if "trade_id" not in out.columns:
            out["trade_id"] = np.nan
        return out
    trade_cols = ["decision_id", "trade_id", "realized_exit_type", "realized_pnl", "mfe", "mae"]
    for col in trade_cols:
        if col not in trades.columns:
            trades[col] = np.nan
    first_trade = trades[trade_cols].drop_duplicates(subset=["decision_id"], keep="first")
    out = out.merge(first_trade, on="decision_id", how="left", suffixes=("", "_trade"))
    for col in ("trade_id", "realized_exit_type", "realized_pnl", "mfe", "mae"):
        trade_col = f"{col}_trade"
        if trade_col in out.columns:
            if col in out.columns:
                out[col] = out[col].where(out[col].notna(), out[trade_col])
                out = out.drop(columns=[trade_col], errors="ignore")
            else:
                out = out.rename(columns={trade_col: col})
    return out


def _build_bucket_attribution(chosen_exec: pd.DataFrame) -> pd.DataFrame:
    if chosen_exec.empty:
        return pd.DataFrame(
            columns=[
                "sub_strategy",
                "trades",
                "pnl",
                "profit_factor",
                "stop_rate",
                "stop_gap_rate",
                "avg_mae",
                "avg_mfe",
            ]
        )
    rows: list[dict] = []
    for bucket, grp in chosen_exec.groupby("sub_strategy", dropna=False):
        pnl = _to_numeric(grp["realized_pnl"]).fillna(0.0)
        gross_win = float(pnl[pnl > 0].sum())
        gross_loss_abs = float((-pnl[pnl < 0]).sum())
        pf = (gross_win / gross_loss_abs) if gross_loss_abs > 0 else (np.inf if gross_win > 0 else 0.0)
        exit_type = grp["realized_exit_type"].astype(str).str.lower()
        rows.append(
            {
                "sub_strategy": str(bucket),
                "trades": int(len(grp)),
                "pnl": float(pnl.sum()),
                "profit_factor": float(pf),
                "stop_rate": float((exit_type == "stop").mean()),
                "stop_gap_rate": float((exit_type == "stop_gap").mean()),
                "avg_mae": float(_to_numeric(grp["mae"]).mean()),
                "avg_mfe": float(_to_numeric(grp["mfe"]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pnl", "profit_factor"], ascending=[False, False], kind="stable")
    return out


def _compute_threshold_sensitivity(chosen_exec: pd.DataFrame) -> dict:
    result: dict[str, object] = {"executed_trades": int(len(chosen_exec)), "metrics": {}}
    if chosen_exec.empty:
        return result
    metric_names = ["edge_points", "edge_gap_points", "runtime_rank_score", "structural_score"]
    pnl = _to_numeric(chosen_exec["realized_pnl"]).fillna(0.0)
    for metric in metric_names:
        if metric not in chosen_exec.columns:
            continue
        vals = _to_numeric(chosen_exec[metric])
        valid = vals.dropna()
        if valid.empty:
            continue
        quantiles = sorted({float(valid.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75)})
        sweeps = []
        for floor in quantiles:
            mask = vals < floor
            impacted = chosen_exec.loc[mask]
            kept = chosen_exec.loc[~mask]
            impacted_pnl = float(_to_numeric(impacted["realized_pnl"]).fillna(0.0).sum())
            kept_pnl = float(_to_numeric(kept["realized_pnl"]).fillna(0.0).sum())
            sweeps.append(
                {
                    "floor": float(floor),
                    "impacted_trades": int(mask.sum()),
                    "impacted_pnl": impacted_pnl,
                    "kept_trades": int((~mask).sum()),
                    "kept_pnl": kept_pnl,
                    "impacted_avg_pnl": float(impacted_pnl / max(1, int(mask.sum()))),
                }
            )
        suggested_floor = None
        for item in sweeps:
            if item["impacted_trades"] > 0 and item["impacted_avg_pnl"] < 0 and item["kept_trades"] > 0:
                suggested_floor = item["floor"]
        result["metrics"][metric] = {
            "quantile_floors_tested": sweeps,
            "suggested_floor_if_goal_is_to_remove_negative_ev_tail": suggested_floor,
        }
    return result


def analyze(decisions_path: Path, trade_path: Path | None, out_dir: Path, edge_tolerance: float, close_rank_gap: float):
    decision_usecols = [
        "decision_id",
        "timestamp",
        "chosen",
        "abstained",
        "sub_strategy",
        "edge_points",
        "edge_gap_points",
        "runtime_rank_score",
        "structural_score",
        "trade_id",
        "realized_exit_type",
        "realized_pnl",
        "mfe",
        "mae",
    ]
    decisions_raw = pd.read_csv(
        decisions_path,
        usecols=lambda c: c in decision_usecols,
    )
    decisions = _prepare_decisions(decisions_raw)
    trades = _load_trade_attribution(trade_path) if trade_path is not None else None
    merged = _merge_trade_outcomes(decisions, trades)

    merged["runtime_rank_score"] = _to_numeric(merged.get("runtime_rank_score", pd.Series(dtype=float)))
    merged["structural_score"] = _to_numeric(merged.get("structural_score", pd.Series(dtype=float)))
    merged["edge_points"] = _to_numeric(merged.get("edge_points", pd.Series(dtype=float)))

    chosen = merged[merged["chosen"] == True].copy()
    chosen_exec = chosen[chosen["trade_id"].notna()].copy()
    chosen_exec["realized_pnl"] = _to_numeric(chosen_exec["realized_pnl"])
    chosen_exec["realized_exit_type"] = chosen_exec["realized_exit_type"].astype(str)

    wins = chosen_exec[chosen_exec["realized_pnl"] > 0]
    losses = chosen_exec[chosen_exec["realized_pnl"] < 0]
    stop_like = chosen_exec[
        chosen_exec["realized_exit_type"].str.lower().isin({"stop", "stop_gap"})
    ]
    checked_chosen = int(len(chosen))
    chosen_with_max = chosen.copy()
    if not chosen_with_max.empty:
        chosen_with_max["group_max_struct"] = merged.groupby("decision_id", dropna=False)["structural_score"].transform("max").reindex(chosen_with_max.index)
        not_best_structural = int(
            (
                chosen_with_max["structural_score"].notna()
                & chosen_with_max["group_max_struct"].notna()
                & (chosen_with_max["structural_score"] < (chosen_with_max["group_max_struct"] - 1e-12))
            ).sum()
        )
    else:
        not_best_structural = 0

    ordered = merged.sort_values(
        ["decision_id", "runtime_rank_score"],
        ascending=[True, False],
        kind="stable",
    )
    top2 = ordered.groupby("decision_id", dropna=False, sort=False).head(2).copy()
    if not top2.empty:
        top2["rank_in_decision"] = top2.groupby("decision_id", dropna=False).cumcount()
        pivot = top2.pivot_table(
            index="decision_id",
            columns="rank_in_decision",
            values="runtime_rank_score",
            aggfunc="first",
        )
        if 0 in pivot.columns and 1 in pivot.columns:
            top2_gap = (pivot[0] - pivot[1]).dropna()
            close_top2_checked = int(len(top2_gap))
            close_top2_count = int((top2_gap <= float(close_rank_gap)).sum())
            top2_gap_map = top2_gap.to_dict()
        else:
            close_top2_checked = 0
            close_top2_count = 0
            top2_gap_map = {}
    else:
        close_top2_checked = 0
        close_top2_count = 0
        top2_gap_map = {}

    chosen_exec_decisions = set(chosen_exec["decision_id"].astype(str).tolist())
    close_top2_executed_checked = int(
        sum(1 for did in chosen_exec_decisions if did in top2_gap_map)
    )
    close_top2_executed = int(
        sum(1 for did in chosen_exec_decisions if (did in top2_gap_map and float(top2_gap_map[did]) <= float(close_rank_gap)))
    )
    swap_candidate_count = 0
    swap_checked = 0
    swap_examples: list[dict] = []

    grouped = merged.groupby("decision_id", dropna=False, sort=False)
    for _, chosen_row in chosen_exec.iterrows():
        decision_id = str(chosen_row.get("decision_id", ""))
        if not decision_id:
            continue
        try:
            grp = grouped.get_group(decision_id)
        except Exception:
            continue
        swap_checked += 1
        chosen_edge = _safe_float(chosen_row.get("edge_points"))
        chosen_struct = _safe_float(chosen_row.get("structural_score"))
        if not np.isfinite(chosen_struct):
            continue
        alts = grp[grp["sub_strategy"].astype(str) != str(chosen_row.get("sub_strategy", ""))].copy()
        if np.isfinite(chosen_edge):
            alts = alts[np.abs(_to_numeric(alts["edge_points"]) - chosen_edge) <= float(edge_tolerance)]
        alts = alts[_to_numeric(alts["structural_score"]) > chosen_struct]
        if alts.empty:
            continue
        swap_candidate_count += 1
        best_alt = alts.sort_values("structural_score", ascending=False, kind="stable").iloc[0]
        if len(swap_examples) < 25:
            swap_examples.append(
                {
                    "decision_id": str(decision_id),
                    "chosen_sub_strategy": str(chosen_row.get("sub_strategy", "")),
                    "chosen_edge": _safe_float(chosen_edge),
                    "chosen_structural": _safe_float(chosen_struct),
                    "alt_sub_strategy": str(best_alt.get("sub_strategy", "")),
                    "alt_edge": _safe_float(best_alt.get("edge_points")),
                    "alt_structural": _safe_float(best_alt.get("structural_score")),
                }
            )

    bucket_df = _build_bucket_attribution(chosen_exec)
    threshold_sensitivity = _compute_threshold_sensitivity(chosen_exec)

    summary = {
        "inputs": {
            "decisions_csv": str(decisions_path),
            "trade_attribution_csv": str(trade_path) if trade_path is not None else None,
            "edge_tolerance": float(edge_tolerance),
            "close_runtime_rank_gap": float(close_rank_gap),
        },
        "counts": {
            "decision_rows": int(len(merged)),
            "decisions": int(merged["decision_id"].nunique()) if "decision_id" in merged.columns else 0,
            "chosen_rows": int(len(chosen)),
            "executed_chosen_rows": int(len(chosen_exec)),
        },
        "distributions": {
            "edge_points": {
                "wins": _distribution(wins["edge_points"]) if "edge_points" in wins.columns else _distribution(pd.Series(dtype=float)),
                "losses": _distribution(losses["edge_points"]) if "edge_points" in losses.columns else _distribution(pd.Series(dtype=float)),
                "stop_or_stop_gap": _distribution(stop_like["edge_points"]) if "edge_points" in stop_like.columns else _distribution(pd.Series(dtype=float)),
            },
            "edge_gap_points": {
                "wins": _distribution(wins["edge_gap_points"]) if "edge_gap_points" in wins.columns else _distribution(pd.Series(dtype=float)),
                "losses": _distribution(losses["edge_gap_points"]) if "edge_gap_points" in losses.columns else _distribution(pd.Series(dtype=float)),
                "stop_or_stop_gap": _distribution(stop_like["edge_gap_points"]) if "edge_gap_points" in stop_like.columns else _distribution(pd.Series(dtype=float)),
            },
        },
        "selection_quality": {
            "chosen_not_best_structural_score_count": int(not_best_structural),
            "chosen_not_best_structural_score_rate": float(not_best_structural / max(1, checked_chosen)),
            "decisions_checked": int(checked_chosen),
            "top2_runtime_rank_close_count": int(close_top2_count),
            "top2_runtime_rank_close_rate": float(close_top2_count / max(1, close_top2_checked)),
            "top2_runtime_rank_close_checked": int(close_top2_checked),
            "executed_top2_runtime_rank_close_count": int(close_top2_executed),
            "executed_top2_runtime_rank_close_rate": float(close_top2_executed / max(1, close_top2_executed_checked)),
            "executed_top2_runtime_rank_close_checked": int(close_top2_executed_checked),
        },
        "counterfactual_swap": {
            "executed_decisions_checked": int(swap_checked),
            "decisions_with_structurally_better_alt_within_edge_tolerance": int(swap_candidate_count),
            "rate": float(swap_candidate_count / max(1, swap_checked)),
            "examples": swap_examples,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "de3_decisions_summary.json"
    bucket_path = out_dir / "de3_bucket_attribution.csv"
    sensitivity_path = out_dir / "de3_threshold_sensitivity.json"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    bucket_df.to_csv(bucket_path, index=False)
    sensitivity_path.write_text(json.dumps(threshold_sensitivity, indent=2, ensure_ascii=True), encoding="utf-8")

    return summary_path, bucket_path, sensitivity_path


def main():
    parser = argparse.ArgumentParser(description="Analyze DE3 decision journal exports.")
    parser.add_argument("--decisions", required=True, help="Path to de3_decisions.csv")
    parser.add_argument(
        "--trade_attribution",
        default="",
        help="Optional path to DE3 trade attribution CSV. If omitted, analyzer uses fields in decision CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="./reports",
        help="Output directory for analyzer outputs (default: ./reports).",
    )
    parser.add_argument(
        "--edge_tolerance",
        type=float,
        default=0.15,
        help="Edge-points tolerance for counterfactual swap checks (default: 0.15).",
    )
    parser.add_argument(
        "--close_runtime_gap",
        type=float,
        default=0.10,
        help="Top-2 runtime rank gap threshold to mark a close call (default: 0.10).",
    )
    args = parser.parse_args()

    decisions_path = Path(args.decisions).expanduser()
    trade_path = Path(args.trade_attribution).expanduser() if str(args.trade_attribution or "").strip() else None
    out_dir = Path(args.out_dir).expanduser()
    summary_path, bucket_path, sensitivity_path = analyze(
        decisions_path=decisions_path,
        trade_path=trade_path,
        out_dir=out_dir,
        edge_tolerance=float(args.edge_tolerance),
        close_rank_gap=float(args.close_runtime_gap),
    )
    print(f"Summary JSON: {summary_path}")
    print(f"Bucket attribution CSV: {bucket_path}")
    print(f"Threshold sensitivity JSON: {sensitivity_path}")


if __name__ == "__main__":
    main()

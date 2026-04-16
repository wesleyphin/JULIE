import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NY_TZ = "America/New_York"


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _find_column(df: pd.DataFrame, *candidates: str) -> str:
    normalized = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        match = normalized.get(str(candidate).strip().lower())
        if match:
            return match
    raise KeyError(f"Missing required column. Tried: {candidates}")


def _inverse_side(side: str) -> str:
    side_text = str(side or "").strip().upper()
    if side_text == "LONG":
        return "SHORT"
    if side_text == "SHORT":
        return "LONG"
    return side_text


def _derive_original_signal(df: pd.DataFrame) -> pd.Series:
    signal_col = _find_column(df, "Signal", "side")
    reverted_col = _find_column(df, "Reverted", "reverted")
    signal_side = df[signal_col].astype(str).str.upper().str.strip()
    reverted = df[reverted_col].astype(str).str.lower().isin({"true", "1", "yes"})
    return pd.Series(
        np.where(reverted.to_numpy(dtype=bool), signal_side.map(_inverse_side).to_numpy(dtype=object), signal_side.to_numpy(dtype=object)),
        index=df.index,
        dtype=object,
    )


def _policy_stats(df: pd.DataFrame) -> pd.DataFrame:
    combo_col = _find_column(df, "Combo key", "combo_key")
    rule_col = _find_column(df, "Rule ID", "rule_id")
    pnl_col = _find_column(df, "Net P&L USD", "pnl_net")
    time_col = _find_column(df, "Date and time", "entry_time")

    working = df.copy()
    working["combo_key"] = working[combo_col].astype(str)
    working["rule_id"] = working[rule_col].astype(str)
    working["pnl_net"] = working[pnl_col].astype(float)
    working["original_signal"] = _derive_original_signal(working)
    working["entry_time"] = pd.to_datetime(working[time_col], utc=True).dt.tz_convert(NY_TZ)
    working["year"] = working["entry_time"].dt.year.astype(int)

    group_cols = ["combo_key", "original_signal", "rule_id"]
    grouped = working.groupby(group_cols, dropna=False)
    stats = grouped.agg(
        trades=("pnl_net", "size"),
        equity=("pnl_net", "sum"),
        avg_trade=("pnl_net", "mean"),
        wins=("pnl_net", lambda values: int((values > 0).sum())),
        losses=("pnl_net", lambda values: int((values <= 0).sum())),
    )
    gross_profit = working.loc[working["pnl_net"] > 0].groupby(group_cols)["pnl_net"].sum()
    gross_loss = working.loc[working["pnl_net"] < 0].groupby(group_cols)["pnl_net"].sum()
    yearly = working.groupby(group_cols + ["year"])["pnl_net"].sum().unstack(fill_value=0.0)

    stats["gross_profit"] = gross_profit
    stats["gross_loss"] = gross_loss
    stats = stats.fillna({"gross_profit": 0.0, "gross_loss": 0.0})
    stats["profit_factor"] = stats.apply(
        lambda row: (row["gross_profit"] / abs(row["gross_loss"]))
        if float(row["gross_loss"]) < 0.0
        else (float("inf") if float(row["gross_profit"]) > 0.0 else 0.0),
        axis=1,
    )
    stats["positive_years"] = (yearly > 0.0).sum(axis=1)
    stats["negative_years"] = (yearly < 0.0).sum(axis=1)
    for year in sorted(yearly.columns.tolist()):
        stats[f"year_{int(year)}"] = yearly[int(year)]
    return stats.sort_values(["equity", "trades"], ascending=[True, False])


def _should_prune(row: pd.Series, args) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    trades = int(row["trades"])
    equity = float(row["equity"])
    pf = float(row["profit_factor"])
    positive_years = int(row["positive_years"])

    negative_fail = trades >= int(args.min_trades) and equity <= float(args.negative_equity_cutoff)
    if negative_fail:
        reasons.append("negative_equity")

    weak_fail = False
    if str(args.mode) == "negative_and_weak":
        weak_fail = (
            trades >= int(args.min_trades)
            and equity <= float(args.weak_equity_cutoff)
            and pf <= float(args.weak_profit_factor_cutoff)
            and positive_years <= int(args.weak_positive_years_cutoff)
        )
        if weak_fail and not negative_fail:
            reasons.append("weak_edge")

    return bool(negative_fail or weak_fail), reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-only pruner for dense RegimeAdaptive side policies based on executed trade CSV statistics."
    )
    parser.add_argument("--artifact", default="artifacts/regimeadaptive_v14_dense_balanced/latest.json")
    parser.add_argument("--trade-csv", required=True)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--mode", choices=["negative_only", "negative_and_weak"], default="negative_only")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--negative-equity-cutoff", type=float, default=0.0)
    parser.add_argument("--weak-equity-cutoff", type=float, default=250.0)
    parser.add_argument("--weak-profit-factor-cutoff", type=float, default=1.05)
    parser.add_argument("--weak-positive-years-cutoff", type=int, default=1)
    parser.add_argument("--write-latest", action="store_true")
    args = parser.parse_args()

    artifact_path = _resolve_path(str(args.artifact))
    trade_csv_path = _resolve_path(str(args.trade_csv))
    artifact_root = _resolve_path(
        str(args.artifact_root) or f"artifacts/regimeadaptive_pruned_{str(args.mode).strip().lower()}"
    )

    if not artifact_path.is_file():
        raise SystemExit(f"Artifact not found: {artifact_path}")
    if not trade_csv_path.is_file():
        raise SystemExit(f"Trade CSV not found: {trade_csv_path}")

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    signal_policies = payload.get("signal_policies", {})
    if not isinstance(signal_policies, dict) or not signal_policies:
        raise SystemExit("Artifact has no signal_policies to prune.")

    trades = pd.read_csv(trade_csv_path)
    stats = _policy_stats(trades)

    pruned_rows: list[dict] = []
    for index, row in stats.iterrows():
        combo_key, original_signal, rule_id = index
        should_prune, reasons = _should_prune(row, args)
        if not should_prune:
            continue
        side_map = signal_policies.get(str(combo_key))
        if not isinstance(side_map, dict):
            continue
        record = side_map.get(str(original_signal).upper())
        if not isinstance(record, dict):
            continue
        record_rule_id = str(record.get("rule_id", "") or "").strip()
        if record_rule_id and str(rule_id).strip() and record_rule_id != str(rule_id).strip():
            continue
        updated_record = dict(record)
        updated_record["policy"] = "skip"
        side_map[str(original_signal).upper()] = updated_record
        pruned_rows.append(
            {
                "combo_key": str(combo_key),
                "original_signal": str(original_signal).upper(),
                "rule_id": str(rule_id),
                "trades": int(row["trades"]),
                "equity": float(row["equity"]),
                "profit_factor": None if not np.isfinite(float(row["profit_factor"])) else float(row["profit_factor"]),
                "positive_years": int(row["positive_years"]),
                "negative_years": int(row["negative_years"]),
                "reasons": list(reasons),
            }
        )

    artifact_root.mkdir(parents=True, exist_ok=True)

    payload["created_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
    payload["version"] = f"{str(payload.get('version', 'regimeadaptive')).strip()}_pruned_{str(args.mode).strip().lower()}"
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["policy_pruning"] = {
        "source_artifact_path": str(artifact_path),
        "source_trade_csv_path": str(trade_csv_path),
        "mode": str(args.mode),
        "min_trades": int(args.min_trades),
        "negative_equity_cutoff": float(args.negative_equity_cutoff),
        "weak_equity_cutoff": float(args.weak_equity_cutoff),
        "weak_profit_factor_cutoff": float(args.weak_profit_factor_cutoff),
        "weak_positive_years_cutoff": int(args.weak_positive_years_cutoff),
        "pruned_policy_count": int(len(pruned_rows)),
        "pruned_rows": pruned_rows,
    }
    payload["metadata"] = metadata

    artifact_out = artifact_root / "regimeadaptive_pruned_artifact.json"
    artifact_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if bool(args.write_latest):
        (artifact_root / "latest.json").write_text(artifact_out.read_text(encoding="utf-8"), encoding="utf-8")

    report_out = artifact_root / "pruning_report.json"
    report_out.write_text(
        json.dumps(
            {
                "source_artifact_path": str(artifact_path),
                "source_trade_csv_path": str(trade_csv_path),
                "mode": str(args.mode),
                "policy_count": int(len(stats)),
                "pruned_policy_count": int(len(pruned_rows)),
                "pruned_rows": pruned_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    stats_out = artifact_root / "policy_stats.csv"
    stats_reset = stats.reset_index()
    stats_reset.to_csv(stats_out, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"artifact={artifact_out}")
    print(f"report={report_out}")
    print(f"stats_csv={stats_out}")
    print(f"pruned_policy_count={len(pruned_rows)}")


if __name__ == "__main__":
    main()

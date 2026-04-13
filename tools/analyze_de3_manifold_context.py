import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "yes", "y"})


def _side_bias_neutral_or_match(side_bias: pd.Series, side: pd.Series) -> pd.Series:
    bias = pd.to_numeric(side_bias, errors="coerce").fillna(0.0).astype(int)
    sign = np.where(side.astype(str).str.upper().eq("LONG"), 1, -1)
    return (bias == 0) | (bias == sign)


def _side_bias_strict_match(side_bias: pd.Series, side: pd.Series) -> pd.Series:
    bias = pd.to_numeric(side_bias, errors="coerce").fillna(0.0).astype(int)
    sign = np.where(side.astype(str).str.upper().eq("LONG"), 1, -1)
    return bias == sign


def _summarize(df: pd.DataFrame) -> dict:
    trades = int(len(df))
    pnl = pd.to_numeric(df.get("pnl_net"), errors="coerce").fillna(0.0)
    wins = int((pnl > 0.0).sum())
    losses = int((pnl < 0.0).sum())
    gross_profit = float(pnl[pnl > 0.0].sum())
    gross_loss = float(-pnl[pnl < 0.0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0.0 else None
    equity = pnl.cumsum()
    max_dd = float((equity.cummax() - equity).max()) if trades else 0.0
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": float((wins / trades) * 100.0) if trades else 0.0,
        "pnl_net": float(pnl.sum()),
        "avg_trade": float(pnl.mean()) if trades else 0.0,
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "max_drawdown": max_dd,
    }


def _load_report(report_path: Path, label: str) -> pd.DataFrame:
    obj = json.loads(report_path.read_text(encoding="utf-8"))
    trades = pd.DataFrame(obj.get("trade_log") or [])
    if trades.empty:
        raise RuntimeError(f"No trade_log rows found in {report_path}")
    trades["replay_label"] = label
    trades["source_report"] = str(report_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["pnl_net"] = pd.to_numeric(trades["pnl_net"], errors="coerce").fillna(0.0)
    trades = trades.sort_values(["entry_time", "trade_id"], kind="stable").reset_index(drop=True)
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze manifold-context filters on DE3 trade logs from shadow-manifold backtests."
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Report spec in LABEL=PATH form. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        default="backtest_reports/de3_manifold_context_analysis",
        help="Output directory for CSV/JSON results.",
    )
    args = parser.parse_args()

    if not args.report:
        raise SystemExit("Provide at least one --report LABEL=PATH value.")

    frames = []
    for item in args.report:
        label, sep, raw_path = str(item or "").partition("=")
        label = label.strip()
        raw_path = raw_path.strip()
        if not sep or not label or not raw_path:
            raise SystemExit(f"Invalid --report value '{item}'. Use LABEL=PATH.")
        report_path = Path(raw_path).expanduser()
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        if not report_path.is_file():
            raise SystemExit(f"Report not found: {report_path}")
        frames.append(_load_report(report_path, label))

    df = pd.concat(frames, ignore_index=True)
    required_cols = [
        "regime_manifold_allow_style",
        "regime_manifold_no_trade",
        "regime_manifold_side_bias",
        "regime_manifold_stress",
        "regime_manifold_dispersion",
        "regime_manifold_regime",
        "side",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise SystemExit(
            "Input reports do not contain manifold shadow fields. Missing: " + ", ".join(missing)
        )

    df["allow_style"] = _coerce_bool(df["regime_manifold_allow_style"])
    df["no_trade"] = _coerce_bool(df["regime_manifold_no_trade"])
    df["side_bias_neutral_or_match"] = _side_bias_neutral_or_match(
        df["regime_manifold_side_bias"], df["side"]
    )
    df["side_bias_strict_match"] = _side_bias_strict_match(
        df["regime_manifold_side_bias"], df["side"]
    )
    df["stress"] = pd.to_numeric(df["regime_manifold_stress"], errors="coerce")
    df["dispersion"] = pd.to_numeric(df["regime_manifold_dispersion"], errors="coerce")
    df["regime"] = df["regime_manifold_regime"].astype(str)

    not_chop_spiral = df["regime"].ne("CHOP_SPIRAL")
    not_rotational = df["regime"].ne("ROTATIONAL_TURBULENCE")
    trend_only = df["regime"].eq("TREND_GEODESIC")
    dispersed_only = df["regime"].eq("DISPERSED")
    not_chop_or_rotation = not_chop_spiral & not_rotational
    natural_rules = {
        "baseline": pd.Series(True, index=df.index),
        "not_rotational": not_rotational,
        "not_chop_spiral": not_chop_spiral,
        "not_chop_or_rotation": not_chop_or_rotation,
        "trend_only": trend_only,
        "dispersed_only": dispersed_only,
        "trend_only_stress_0.80": trend_only & (df["stress"] <= 0.80),
        "not_chop_spiral_stress_0.15": not_chop_spiral & (df["stress"] <= 0.15),
        "not_chop_spiral_stress_0.60": not_chop_spiral & (df["stress"] <= 0.60),
        "not_chop_or_rotation_stress_0.60": not_chop_or_rotation & (df["stress"] <= 0.60),
        "not_chop_spiral_stress_0.15_disp_0.95": (
            not_chop_spiral & (df["stress"] <= 0.15) & (df["dispersion"] <= 0.95)
        ),
        "allow_style": df["allow_style"],
        "allow_style_neutral_bias": df["allow_style"] & df["side_bias_neutral_or_match"],
        "allow_style_strict_bias": df["allow_style"] & df["side_bias_strict_match"],
        "allow_style_stress_0.60": df["allow_style"] & (df["stress"] <= 0.60),
        "allow_style_stress_0.65": df["allow_style"] & (df["stress"] <= 0.65),
        "allow_style_disp_0.55": df["allow_style"] & (df["dispersion"] <= 0.55),
        "allow_style_neutral_bias_stress_0.60": (
            df["allow_style"] & df["side_bias_neutral_or_match"] & (df["stress"] <= 0.60)
        ),
        "allow_style_neutral_bias_disp_0.55": (
            df["allow_style"] & df["side_bias_neutral_or_match"] & (df["dispersion"] <= 0.55)
        ),
    }

    baseline_by_label = {
        label: _summarize(part)
        for label, part in df.groupby("replay_label", sort=False)
    }
    baseline_combined = _summarize(df)

    natural_rows = []
    all_labels = list(df["replay_label"].drop_duplicates())
    for rule_name, mask in natural_rules.items():
        kept_all = df.loc[mask]
        combined = _summarize(kept_all)
        combined["rule"] = rule_name
        combined["replay_label"] = "combined"
        combined["retention"] = float(len(kept_all) / len(df)) if len(df) else 0.0
        natural_rows.append(combined)
        for label in all_labels:
            part = df.loc[df["replay_label"] == label]
            kept = part.loc[mask.loc[part.index]]
            row = _summarize(kept)
            row["rule"] = rule_name
            row["replay_label"] = label
            row["retention"] = float(len(kept) / len(part)) if len(part) else 0.0
            base = baseline_by_label[label]
            row["delta_pnl_net"] = float(row["pnl_net"] - base["pnl_net"])
            row["delta_max_drawdown"] = float(row["max_drawdown"] - base["max_drawdown"])
            row["delta_avg_trade"] = float(row["avg_trade"] - base["avg_trade"])
            natural_rows.append(row)

    sweep_rows = []
    base_masks = {
        "all": (~df["no_trade"]),
        "not_rotational": (~df["no_trade"]) & not_rotational,
        "not_chop_spiral": (~df["no_trade"]) & not_chop_spiral,
        "not_chop_or_rotation": (~df["no_trade"]) & not_chop_or_rotation,
        "trend_only": (~df["no_trade"]) & trend_only,
        "dispersed_only": (~df["no_trade"]) & dispersed_only,
        "allow_style": df["allow_style"] & (~df["no_trade"]),
    }
    bias_modes = {
        "off": pd.Series(True, index=df.index),
        "neutral_or_match": df["side_bias_neutral_or_match"],
        "strict_match": df["side_bias_strict_match"],
    }
    stress_caps = [None, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65]
    dispersion_caps = [None, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    for base_name, base_mask in base_masks.items():
        for bias_name, bias_mask in bias_modes.items():
            for stress_cap in stress_caps:
                for disp_cap in dispersion_caps:
                    mask = base_mask & bias_mask
                    rule_parts = [base_name, bias_name]
                    if stress_cap is not None:
                        mask = mask & (df["stress"] <= stress_cap)
                        rule_parts.append(f"stress<={stress_cap:.2f}")
                    if disp_cap is not None:
                        mask = mask & (df["dispersion"] <= disp_cap)
                        rule_parts.append(f"disp<={disp_cap:.2f}")
                    rule_name = "|".join(rule_parts)
                    row = _summarize(df.loc[mask])
                    row["rule"] = rule_name
                    row["replay_label"] = "combined"
                    row["retention"] = float(mask.mean()) if len(mask) else 0.0
                    for label in all_labels:
                        part = df.loc[df["replay_label"] == label]
                        kept = part.loc[mask.loc[part.index]]
                        stats = _summarize(kept)
                        row[f"{label}_trades"] = int(stats["trades"])
                        row[f"{label}_retention"] = float(len(kept) / len(part)) if len(part) else 0.0
                        row[f"{label}_pnl_net"] = float(stats["pnl_net"])
                        row[f"{label}_avg_trade"] = float(stats["avg_trade"])
                        row[f"{label}_max_drawdown"] = float(stats["max_drawdown"])
                        base = baseline_by_label[label]
                        row[f"{label}_delta_pnl_net"] = float(stats["pnl_net"] - base["pnl_net"])
                        row[f"{label}_delta_max_drawdown"] = float(stats["max_drawdown"] - base["max_drawdown"])
                    sweep_rows.append(row)

    regime_rows = []
    for (label, regime), part in df.groupby(["replay_label", "regime"], sort=False):
        row = _summarize(part)
        row["replay_label"] = label
        row["regime"] = regime
        regime_rows.append(row)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    natural_df = pd.DataFrame(natural_rows)
    sweep_df = pd.DataFrame(sweep_rows)
    regime_df = pd.DataFrame(regime_rows)

    natural_path = out_dir / "natural_rules.csv"
    sweep_path = out_dir / "sweep_rules.csv"
    regime_path = out_dir / "regime_breakdown.csv"
    summary_path = out_dir / "summary.json"

    natural_df.to_csv(natural_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    summary = {
        "baseline_by_label": baseline_by_label,
        "baseline_combined": baseline_combined,
        "natural_rules_path": str(natural_path),
        "sweep_rules_path": str(sweep_path),
        "regime_breakdown_path": str(regime_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"natural_rules={natural_path}")
    print(f"sweep_rules={sweep_path}")
    print(f"regime_breakdown={regime_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_ID_TO_NAME = {
    0: "TREND_GEODESIC",
    1: "CHOP_SPIRAL",
    2: "DISPERSED",
    3: "ROTATIONAL_TURBULENCE",
}

SESSION_ID_TO_NAME = {
    -1: "OFF",
    0: "ASIA",
    1: "LONDON",
    2: "NY_AM",
    3: "NY_PM",
}


def _read_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex in {path}")
    out = df.copy()
    out["regime_name"] = (
        pd.to_numeric(out.get("manifold_regime_id"), errors="coerce")
        .round()
        .astype("Int64")
        .map(REGIME_ID_TO_NAME)
        .fillna("UNKNOWN")
    )
    out["session_name"] = (
        pd.to_numeric(out.get("session_id"), errors="coerce")
        .round()
        .astype("Int64")
        .map(SESSION_ID_TO_NAME)
        .fillna("UNKNOWN")
    )
    out["year"] = out.index.tz_convert("America/New_York").year
    return out


def _pct(series: pd.Series) -> pd.Series:
    total = float(series.sum())
    if total <= 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series.astype(float) / total) * 100.0


def _metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "manifold_R",
        "manifold_alignment",
        "manifold_smoothness",
        "manifold_stress",
        "manifold_dispersion",
        "manifold_risk_mult",
    ]
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    rows = []
    for metric in metrics:
        vals = pd.to_numeric(df.get(metric), errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "scope": "overall",
                "group": "ALL",
                "metric": metric,
                "count": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                **{f"q{int(q*100):02d}": float(vals.quantile(q)) for q in quantiles},
            }
        )
        for regime_name, part in df.groupby("regime_name", sort=False):
            regime_vals = pd.to_numeric(part.get(metric), errors="coerce").dropna()
            if regime_vals.empty:
                continue
            rows.append(
                {
                    "scope": "regime",
                    "group": str(regime_name),
                    "metric": metric,
                    "count": int(regime_vals.shape[0]),
                    "mean": float(regime_vals.mean()),
                    "std": float(regime_vals.std()),
                    "min": float(regime_vals.min()),
                    "max": float(regime_vals.max()),
                    **{f"q{int(q*100):02d}": float(regime_vals.quantile(q)) for q in quantiles},
                }
            )
    return pd.DataFrame(rows)


def _threshold_pass_rates(df: pd.DataFrame) -> pd.DataFrame:
    alignment = pd.to_numeric(df.get("manifold_alignment"), errors="coerce").fillna(0.0)
    smoothness = pd.to_numeric(df.get("manifold_smoothness"), errors="coerce").fillna(0.0)
    stress = pd.to_numeric(df.get("manifold_stress"), errors="coerce").fillna(0.0)
    dispersion = pd.to_numeric(df.get("manifold_dispersion"), errors="coerce").fillna(0.0)
    side_bias = pd.to_numeric(df.get("manifold_side_bias"), errors="coerce").fillna(0.0)
    no_trade = pd.to_numeric(df.get("manifold_no_trade"), errors="coerce").fillna(0.0) > 0.5

    checks = {
        "alignment>=0.62": alignment >= 0.62,
        "alignment<=0.45": alignment <= 0.45,
        "alignment>=0.45": alignment >= 0.45,
        "smoothness>=0.60": smoothness >= 0.60,
        "smoothness<=0.45": smoothness <= 0.45,
        "dispersion<=0.45": dispersion <= 0.45,
        "dispersion>=0.65": dispersion >= 0.65,
        "dispersion<=0.95": dispersion <= 0.95,
        "stress<=0.15": stress <= 0.15,
        "stress>=0.60": stress >= 0.60,
        "stress>=0.85": stress >= 0.85,
        "side_bias!=0": side_bias != 0.0,
        "no_trade": no_trade,
        "trend_geodesic_raw": (alignment >= 0.62) & (smoothness >= 0.60) & (dispersion <= 0.45),
        "dispersed_raw": (dispersion >= 0.65) & (alignment <= 0.45),
        "chop_spiral_raw": (stress >= 0.60) | (smoothness <= 0.45),
    }
    rows = []
    total = float(len(df))
    for name, mask in checks.items():
        count = int(mask.sum())
        rows.append(
            {
                "check": name,
                "count": count,
                "rate": (count / total) * 100.0 if total > 0.0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("rate", ascending=False, kind="stable")


def _regime_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["regime_name"].value_counts(dropna=False, sort=False)
    out = counts.rename_axis("regime_name").reset_index(name="count")
    out["pct"] = _pct(out["count"])
    return out.sort_values(["count", "regime_name"], ascending=[False, True], kind="stable")


def _session_regime_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["session_name", "regime_name"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["session_total"] = counts.groupby("session_name")["count"].transform("sum")
    counts["pct_within_session"] = np.where(
        counts["session_total"] > 0,
        (counts["count"].astype(float) / counts["session_total"].astype(float)) * 100.0,
        0.0,
    )
    return counts.sort_values(
        ["session_name", "count", "regime_name"],
        ascending=[True, False, True],
        kind="stable",
    )


def _year_regime_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["year", "regime_name"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["year_total"] = counts.groupby("year")["count"].transform("sum")
    counts["pct_within_year"] = np.where(
        counts["year_total"] > 0,
        (counts["count"].astype(float) / counts["year_total"].astype(float)) * 100.0,
        0.0,
    )
    return counts.sort_values(
        ["year", "count", "regime_name"],
        ascending=[True, False, True],
        kind="stable",
    )


def _side_bias_counts(df: pd.DataFrame) -> pd.DataFrame:
    side_bias = pd.to_numeric(df.get("manifold_side_bias"), errors="coerce").fillna(0.0).astype(int)
    counts = side_bias.value_counts(dropna=False, sort=False)
    out = counts.rename_axis("side_bias").reset_index(name="count")
    out["pct"] = _pct(out["count"])
    return out.sort_values("side_bias", kind="stable")


def _regime_transition_counts(df: pd.DataFrame) -> pd.DataFrame:
    regime = df["regime_name"].astype(str)
    prev_regime = regime.shift(1)
    mask = prev_regime.notna()
    trans = (
        pd.DataFrame({"prev_regime": prev_regime[mask], "regime_name": regime[mask]})
        .groupby(["prev_regime", "regime_name"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    trans["prev_total"] = trans.groupby("prev_regime")["count"].transform("sum")
    trans["pct_from_prev"] = np.where(
        trans["prev_total"] > 0,
        (trans["count"].astype(float) / trans["prev_total"].astype(float)) * 100.0,
        0.0,
    )
    return trans.sort_values(
        ["prev_regime", "count", "regime_name"],
        ascending=[True, False, True],
        kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RegimeManifold calibration from cached feature parquet.")
    parser.add_argument(
        "--features",
        default="manifold_strategy_features_clean_full.parquet",
        help="Parquet path containing manifold feature columns.",
    )
    parser.add_argument(
        "--out-dir",
        default="backtest_reports/manifold_regime_calibration_audit",
        help="Output directory for audit artifacts.",
    )
    parser.add_argument("--start", default="", help="Optional ET start timestamp/date filter.")
    parser.add_argument("--end", default="", help="Optional ET end timestamp/date filter.")
    args = parser.parse_args()

    features_path = Path(args.features).expanduser()
    if not features_path.is_absolute():
        features_path = Path.cwd() / features_path
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_features(features_path)
    if str(args.start or "").strip():
        start = pd.Timestamp(str(args.start)).tz_localize("America/New_York") if pd.Timestamp(str(args.start)).tzinfo is None else pd.Timestamp(str(args.start)).tz_convert("America/New_York")
        df = df.loc[df.index.tz_convert("America/New_York") >= start]
    if str(args.end or "").strip():
        end = pd.Timestamp(str(args.end)).tz_localize("America/New_York") if pd.Timestamp(str(args.end)).tzinfo is None else pd.Timestamp(str(args.end)).tz_convert("America/New_York")
        df = df.loc[df.index.tz_convert("America/New_York") <= end]
    if df.empty:
        raise SystemExit("No rows left after filtering.")

    regime_counts = _regime_counts(df)
    session_regime_counts = _session_regime_counts(df)
    year_regime_counts = _year_regime_counts(df)
    threshold_pass_rates = _threshold_pass_rates(df)
    metric_summary = _metric_summary(df)
    side_bias_counts = _side_bias_counts(df)
    transition_counts = _regime_transition_counts(df)

    regime_counts_path = out_dir / "regime_counts.csv"
    session_counts_path = out_dir / "session_regime_counts.csv"
    year_counts_path = out_dir / "year_regime_counts.csv"
    threshold_path = out_dir / "threshold_pass_rates.csv"
    metric_path = out_dir / "metric_summary.csv"
    side_bias_path = out_dir / "side_bias_counts.csv"
    transition_path = out_dir / "regime_transitions.csv"
    summary_path = out_dir / "summary.json"

    regime_counts.to_csv(regime_counts_path, index=False)
    session_regime_counts.to_csv(session_counts_path, index=False)
    year_regime_counts.to_csv(year_counts_path, index=False)
    threshold_pass_rates.to_csv(threshold_path, index=False)
    metric_summary.to_csv(metric_path, index=False)
    side_bias_counts.to_csv(side_bias_path, index=False)
    transition_counts.to_csv(transition_path, index=False)

    summary = {
        "features_path": str(features_path),
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "regime_counts_path": str(regime_counts_path),
        "session_regime_counts_path": str(session_counts_path),
        "year_regime_counts_path": str(year_counts_path),
        "threshold_pass_rates_path": str(threshold_path),
        "metric_summary_path": str(metric_path),
        "side_bias_counts_path": str(side_bias_path),
        "regime_transitions_path": str(transition_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary={summary_path}")
    print(f"rows={len(df)}")
    print(f"range={df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()

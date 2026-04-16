import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_manifold_engine import classify_signal_style


REGIME_ID_TO_NAME = {
    0: "TREND_GEODESIC",
    1: "CHOP_SPIRAL",
    2: "DISPERSED",
    3: "ROTATIONAL_TURBULENCE",
}


def _allow_style(style: str, row: pd.Series) -> bool:
    if style == "fade":
        return bool(row.get("manifold_allow_fade", 0.0) >= 0.5 or row.get("manifold_allow_mean_reversion", 0.0) >= 0.5)
    if style == "trend":
        return bool(row.get("manifold_allow_trend", 0.0) >= 0.5)
    if style == "breakout":
        return bool(row.get("manifold_allow_breakout", 0.0) >= 0.5)
    return bool(row.get("manifold_allow_mean_reversion", 0.0) >= 0.5)


def _load_features(path: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    cols = [
        "manifold_R",
        "manifold_alignment",
        "manifold_smoothness",
        "manifold_stress",
        "manifold_dispersion",
        "manifold_side_bias",
        "manifold_risk_mult",
        "manifold_no_trade",
        "manifold_regime_id",
        "manifold_allow_trend",
        "manifold_allow_mean_reversion",
        "manifold_allow_breakout",
        "manifold_allow_fade",
        "manifold_R_pct",
        "manifold_alignment_pct",
        "manifold_smoothness_pct",
        "manifold_stress_pct",
        "manifold_dispersion_pct",
    ]
    feat = pd.read_parquet(path, columns=cols)
    if not isinstance(feat.index, pd.DatetimeIndex):
        raise SystemExit(f"Expected DatetimeIndex in feature parquet: {path}")
    window = feat.loc[(feat.index >= start_ts) & (feat.index <= end_ts)].copy()
    if window.empty:
        raise SystemExit("No feature rows found inside requested report time range.")
    return window


def _restamp_report(report_path: Path, features: pd.DataFrame, out_dir: Path) -> Path:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    trades = pd.DataFrame(payload.get("trade_log") or [])
    if trades.empty:
        raise SystemExit(f"No trade_log rows in report: {report_path}")

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    feat = features.reindex(trades["entry_time"], method="ffill")
    if feat.isna().all(axis=1).any():
        missing = int(feat.isna().all(axis=1).sum())
        raise SystemExit(f"Missing calibrated feature rows for {missing} trade timestamps in {report_path}")

    feat = feat.reset_index(drop=True)
    trades = trades.reset_index(drop=True)
    regime_names = (
        pd.to_numeric(feat.get("manifold_regime_id"), errors="coerce")
        .round()
        .astype("Int64")
        .map(REGIME_ID_TO_NAME)
        .fillna("UNKNOWN")
    )

    trades["regime_manifold_regime"] = regime_names
    trades["regime_manifold_R"] = pd.to_numeric(feat["manifold_R"], errors="coerce").fillna(0.0)
    trades["regime_manifold_alignment"] = pd.to_numeric(feat["manifold_alignment"], errors="coerce").fillna(0.0)
    trades["regime_manifold_smoothness"] = pd.to_numeric(feat["manifold_smoothness"], errors="coerce").fillna(0.0)
    trades["regime_manifold_stress"] = pd.to_numeric(feat["manifold_stress"], errors="coerce").fillna(0.0)
    trades["regime_manifold_dispersion"] = pd.to_numeric(feat["manifold_dispersion"], errors="coerce").fillna(0.0)
    trades["regime_manifold_side_bias"] = pd.to_numeric(feat["manifold_side_bias"], errors="coerce").fillna(0.0).astype(int)
    trades["regime_manifold_no_trade"] = pd.to_numeric(feat["manifold_no_trade"], errors="coerce").fillna(0.0) >= 0.5
    trades["regime_manifold_risk_mult"] = pd.to_numeric(feat["manifold_risk_mult"], errors="coerce").fillna(1.0)
    trades["regime_manifold_alignment_pct"] = pd.to_numeric(feat["manifold_alignment_pct"], errors="coerce").fillna(0.0)
    trades["regime_manifold_smoothness_pct"] = pd.to_numeric(feat["manifold_smoothness_pct"], errors="coerce").fillna(0.0)
    trades["regime_manifold_stress_pct"] = pd.to_numeric(feat["manifold_stress_pct"], errors="coerce").fillna(0.0)
    trades["regime_manifold_dispersion_pct"] = pd.to_numeric(feat["manifold_dispersion_pct"], errors="coerce").fillna(0.0)

    allow_rows = feat[
        [
            "manifold_allow_trend",
            "manifold_allow_mean_reversion",
            "manifold_allow_breakout",
            "manifold_allow_fade",
        ]
    ]
    style_values = []
    allow_style_values = []
    allow_raw_values = []
    for idx, trade in trades.iterrows():
        style = classify_signal_style(trade.get("strategy"), trade.get("sub_strategy"))
        allow_row = allow_rows.iloc[idx]
        allow_raw = {
            "trend": bool(float(allow_row.get("manifold_allow_trend", 0.0) or 0.0) >= 0.5),
            "mean_reversion": bool(float(allow_row.get("manifold_allow_mean_reversion", 0.0) or 0.0) >= 0.5),
            "breakout": bool(float(allow_row.get("manifold_allow_breakout", 0.0) or 0.0) >= 0.5),
            "fade": bool(float(allow_row.get("manifold_allow_fade", 0.0) or 0.0) >= 0.5),
        }
        style_values.append(style)
        allow_style_values.append(_allow_style(style, allow_row))
        allow_raw_values.append(allow_raw)
    trades["regime_manifold_style"] = style_values
    trades["regime_manifold_allow_style"] = allow_style_values
    trades["regime_manifold_allow_raw"] = allow_raw_values

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / report_path.name
    payload["trade_log"] = json.loads(trades.to_json(orient="records", date_format="iso"))
    payload["calibrated_manifold_source"] = str(out_dir)
    payload["calibrated_manifold_features"] = str(features.attrs.get("source_path", ""))
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Restamp DE3 reports with calibrated Manifold context from cached feature parquet.")
    parser.add_argument("--features", default="manifold_strategy_features_clean_full_robust.parquet")
    parser.add_argument("--report", action="append", default=[], help="Repeatable report path.")
    parser.add_argument("--out-dir", default="backtest_reports/de3_manifold_shadow_restamped")
    args = parser.parse_args()

    if not args.report:
        raise SystemExit("Provide at least one --report path.")

    report_paths = []
    for raw in args.report:
        report_path = Path(raw).expanduser()
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        if not report_path.is_file():
            raise SystemExit(f"Report not found: {report_path}")
        report_paths.append(report_path)

    min_entry = None
    max_entry = None
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        trades = pd.DataFrame(payload.get("trade_log") or [])
        if trades.empty:
            continue
        entry_ts = pd.to_datetime(trades["entry_time"], utc=True)
        cur_min = entry_ts.min()
        cur_max = entry_ts.max()
        min_entry = cur_min if min_entry is None else min(min_entry, cur_min)
        max_entry = cur_max if max_entry is None else max(max_entry, cur_max)
    if min_entry is None or max_entry is None:
        raise SystemExit("No entry_time values found in reports.")

    features_path = Path(args.features).expanduser()
    if not features_path.is_absolute():
        features_path = Path.cwd() / features_path
    features = _load_features(features_path, min_entry, max_entry)
    features.attrs["source_path"] = str(features_path)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    for report_path in report_paths:
        out_path = _restamp_report(report_path, features, out_dir)
        print(f"restamped={out_path}")


if __name__ == "__main__":
    main()

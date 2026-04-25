import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES  # noqa: E402
from tools.run_aetherflow_live_policy_search import _configure_strategy, _build_variant_signals  # noqa: E402
from tools.backtest_aetherflow_direct import _load_base_features, _prepare_symbol_df, _resolve_source, _simulate  # noqa: E402
from tools.run_aetherflow_deploy_policy_search import _load_eval_windows, _window_bounds  # noqa: E402


def _json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AetherFlow hidden sub-regime, family-specific, and phase-memory hypotheses.")
    parser.add_argument("--windows-file", default="configs/aetherflow_eval_windows_20260421.json")
    parser.add_argument("--major-windows-file", default="configs/aetherflow_eval_windows_2011_2024_full.json")
    parser.add_argument("--policy-file", default="configs/aetherflow_current_live_policy_20260421.json")
    parser.add_argument(
        "--base-features",
        default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--history-buffer-days", type=int, default=14)
    parser.add_argument("--top-buckets", type=int, default=8)
    parser.add_argument("--top-filters", type=int, default=8)
    parser.add_argument("--output-dir", default="backtest_reports/aetherflow_hypotheses_20260422")
    return parser.parse_args()


def _phase_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_index(kind="mergesort").copy()
    regime_id = pd.to_numeric(work.get("manifold_regime_id"), errors="coerce").fillna(-1).round().astype(int)
    regime_change = regime_id.ne(regime_id.shift(1)).fillna(True)
    group_id = regime_change.cumsum()
    work["phase_regime_run_bars"] = group_id.groupby(group_id).cumcount() + 1
    work["phase_regime_flip_count_10"] = regime_change.astype(int).rolling(10, min_periods=1).sum()

    def _series(name: str) -> pd.Series:
        if name in work.columns:
            raw = work[name]
        else:
            raw = pd.Series(0.0, index=work.index)
        return pd.to_numeric(raw, errors="coerce").fillna(0.0)

    for col in [
        "manifold_alignment_pct",
        "manifold_smoothness_pct",
        "manifold_stress_pct",
        "manifold_dispersion_pct",
        "d_alignment_3",
        "d_stress_3",
        "d_dispersion_3",
        "flow_agreement",
        "flow_mag_slow",
        "pressure_imbalance_30",
    ]:
        series = _series(col)
        work[f"phase_{col}_mean_5"] = series.rolling(5, min_periods=1).mean()
        work[f"phase_{col}_trend_5"] = series - series.shift(5).fillna(series.iloc[0] if len(series) else 0.0)
    return work


def _load_strategy(policy_file: Path):
    payload = json.loads(policy_file.read_text(encoding="utf-8"))
    variant = payload["variants"][0]
    params = config.CONFIG["AETHERFLOW_STRATEGY"]
    strategy = _configure_strategy(
        model_file=Path(params["model_file"]),
        thresholds_file=Path(params["thresholds_file"]),
        metrics_file=Path(params["metrics_file"]),
        variant=variant,
    )
    return strategy, variant, params


def _load_trade_feature_frame(
    *,
    strategy,
    source_path: Path,
    base_features_path: Path,
    windows: list[dict[str, Any]],
    history_buffer_days: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for window in windows:
        start_time, end_time = _window_bounds(str(window["start"]), str(window["end"]))
        symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
            source_path,
            start_time,
            end_time,
            "auto_by_day",
            "volume",
            int(history_buffer_days),
        )
        base_features = _load_base_features(
            base_features_path,
            pd.Timestamp(symbol_df.index.min()),
            pd.Timestamp(symbol_df.index.max()),
        )
        base_features = _phase_features(base_features)
        signals = _build_variant_signals(strategy, base_features)
        stats = _simulate(
            df=symbol_df,
            signals=signals,
            start_time=start_time,
            end_time=end_time,
            use_horizon_time_stop=False,
            allow_same_side_add_ons=False,
            max_same_side_legs=1,
        )
        trade_log = pd.DataFrame(stats.get("trade_log", []) or [])
        if trade_log.empty:
            continue
        trade_log["window_label"] = str(window["label"])
        trade_log["entry_time_utc"] = pd.to_datetime(trade_log["entry_time"], utc=True)
        trade_log["signal_time_utc"] = trade_log["entry_time_utc"] - pd.Timedelta(minutes=1)
        signals = signals.copy()
        signals["signal_time_utc"] = pd.to_datetime(signals.index, utc=True)
        merged = trade_log.merge(signals.reset_index(drop=True), on="signal_time_utc", how="left", suffixes=("_trade", "_sig"))
        merged["symbol"] = symbol
        merged["symbol_distribution"] = str(symbol_distribution)
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _bucket_key_cols() -> list[str]:
    return ["aetherflow_setup_family", "session", "aetherflow_regime", "side"]


def _major_weak_buckets(trades: pd.DataFrame, top_buckets: int) -> pd.DataFrame:
    major_labels = ["y2011_2024_full", "y2025_full", "y2026_fresh_post_jan26"]
    scoped = trades.loc[trades["window_label"].isin(major_labels)].copy()
    grouped = (
        scoped.groupby(_bucket_key_cols() + ["window_label"], dropna=False)["pnl_net"]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=_bucket_key_cols(),
        columns="window_label",
        values="pnl_net",
        fill_value=0.0,
    ).reset_index()
    for label in major_labels:
        if label not in pivot.columns:
            pivot[label] = 0.0
    pivot["negative_windows"] = (pivot[major_labels] < 0.0).sum(axis=1)
    pivot["major_total_net"] = pivot[major_labels].sum(axis=1)
    pivot["recent_april_net"] = 0.0
    april = (
        trades.loc[trades["window_label"].eq("y2026_april_recent")]
        .groupby(_bucket_key_cols(), dropna=False)["pnl_net"]
        .sum()
        .reset_index(name="recent_april_net")
    )
    pivot = pivot.merge(april, on=_bucket_key_cols(), how="left", suffixes=("", "_april"))
    pivot["recent_april_net"] = pd.to_numeric(pivot["recent_april_net"], errors="coerce").fillna(0.0)
    pivot = pivot.sort_values(
        ["negative_windows", "major_total_net", "recent_april_net"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return pivot.head(int(top_buckets)).reset_index(drop=True)


def _candidate_filters(bucket: pd.DataFrame) -> list[dict[str, Any]]:
    defs: list[dict[str, Any]] = []

    def add(name: str, col: str, kind: str, thresholds: list[float], hypothesis: str):
        for threshold in thresholds:
            defs.append(
                {
                    "name": f"{name}_{str(threshold).replace('.', 'p').replace('-', 'm')}",
                    "column": col,
                    "kind": kind,
                    "threshold": float(threshold),
                    "hypothesis": hypothesis,
                }
            )

    add("align_pct_min", "manifold_alignment_pct", "min", [0.45, 0.50, 0.55, 0.60], "hidden_subregime")
    add("smooth_pct_min", "manifold_smoothness_pct", "min", [0.35, 0.45, 0.55], "hidden_subregime")
    add("stress_pct_max", "manifold_stress_pct", "max", [0.15, 0.25, 0.40, 0.60], "hidden_subregime")
    add("disp_pct_max", "manifold_dispersion_pct", "max", [0.55, 0.70, 0.85, 0.95], "hidden_subregime")
    add("flow_agreement_min", "flow_agreement", "min", [0.50, 0.70, 0.90], "family_specific")
    add("flow_mag_slow_min", "flow_mag_slow", "min", [0.30, 0.50, 0.70], "family_specific")
    add("setup_strength_min", "setup_strength", "min", [0.40, 0.50, 0.60], "family_specific")
    add("d_align_min", "d_alignment_3", "min", [-0.05, 0.00, 0.05, 0.10, 0.20], "family_specific")
    add("pressure30_min", "pressure_imbalance_30", "min", [-0.50, -0.25, 0.00, 0.25], "family_specific")
    add("phase_run_min", "phase_regime_run_bars", "min", [2, 4, 8, 12], "phase_memory")
    add("phase_flip10_max", "phase_regime_flip_count_10", "max", [1, 2, 3], "phase_memory")
    add("phase_align_mean5_min", "phase_manifold_alignment_pct_mean_5", "min", [0.45, 0.50, 0.55], "phase_memory")
    add("phase_stress_mean5_max", "phase_manifold_stress_pct_mean_5", "max", [0.20, 0.35, 0.50], "phase_memory")
    add("phase_align_trend5_min", "phase_manifold_alignment_pct_trend_5", "min", [-0.05, 0.00, 0.05, 0.10], "phase_memory")
    add("phase_stress_trend5_max", "phase_manifold_stress_pct_trend_5", "max", [0.00, 0.05, 0.10], "phase_memory")
    add("phase_dalign_mean5_min", "phase_d_alignment_3_mean_5", "min", [-0.05, 0.00, 0.05, 0.10], "phase_memory")
    return defs


def _apply_filter(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    if rule["column"] in df.columns:
        raw = df[rule["column"]]
    else:
        raw = pd.Series(np.nan, index=df.index)
    series = pd.to_numeric(raw, errors="coerce")
    if rule["kind"] == "min":
        return series >= float(rule["threshold"])
    return series <= float(rule["threshold"])


def _evaluate_bucket_filters(bucket_df: pd.DataFrame, filter_defs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bucket_df.empty:
        return pd.DataFrame()
    major_labels = ["y2011_2024_full", "y2025_full", "y2026_fresh_post_jan26", "y2026_april_recent", "y2026_jan_validation"]
    for rule in filter_defs:
        mask_keep = _apply_filter(bucket_df, rule).fillna(False)
        removed = bucket_df.loc[~mask_keep].copy()
        kept = bucket_df.loc[mask_keep].copy()
        row: dict[str, Any] = {
            "hypothesis": rule["hypothesis"],
            "filter_name": rule["name"],
            "column": rule["column"],
            "kind": rule["kind"],
            "threshold": float(rule["threshold"]),
            "bucket_trades": int(len(bucket_df)),
            "kept_trades": int(len(kept)),
            "removed_trades": int(len(removed)),
            "removed_net_total": float(removed["pnl_net"].sum()),
        }
        score = 0.0
        for label in major_labels:
            removed_net = float(removed.loc[removed["window_label"].eq(label), "pnl_net"].sum())
            row[f"{label}_removed_net"] = removed_net
            if label == "y2026_fresh_post_jan26":
                score += -removed_net * 1.0
            elif label == "y2026_april_recent":
                score += -removed_net * 0.8
            elif label == "y2025_full":
                score += -removed_net * 0.4
            elif label == "y2011_2024_full":
                score += -removed_net * 0.2
            elif label == "y2026_jan_validation":
                score += -removed_net * 0.4
        row["score"] = float(score)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["hypothesis", "score", "y2026_fresh_post_jan26_removed_net", "y2026_april_recent_removed_net"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    return out


def main() -> None:
    args = _parse_args()
    out_dir = (ROOT / str(args.output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_file = (ROOT / str(args.policy_file)).resolve()
    strategy, variant, params = _load_strategy(policy_file)
    source_path = _resolve_source = __import__("tools.backtest_aetherflow_direct", fromlist=["_resolve_source"])._resolve_source(str(args.source))
    base_features_path = (ROOT / str(args.base_features)).resolve()
    windows = _load_eval_windows(years_text="", windows_file=str((ROOT / str(args.windows_file)).resolve()))
    major_windows = _load_eval_windows(years_text="", windows_file=str((ROOT / str(args.major_windows_file)).resolve()))

    trades = _load_trade_feature_frame(
        strategy=strategy,
        source_path=source_path,
        base_features_path=base_features_path,
        windows=major_windows + windows,
        history_buffer_days=int(args.history_buffer_days),
    )
    if trades.empty:
        raise RuntimeError("No trades produced for analysis.")
    trades["pnl_net"] = pd.to_numeric(trades["pnl_net"], errors="coerce").fillna(0.0)
    trades.to_csv(out_dir / "trades_with_features.csv", index=False)

    weak_buckets = _major_weak_buckets(trades, int(args.top_buckets))
    weak_buckets.to_csv(out_dir / "weak_buckets.csv", index=False)

    filter_rows: list[pd.DataFrame] = []
    top_recommendations: list[dict[str, Any]] = []
    for _, bucket in weak_buckets.iterrows():
        key_mask = pd.Series(True, index=trades.index)
        for col in _bucket_key_cols():
            key_mask &= trades[col].astype(str).eq(str(bucket[col]))
        bucket_df = trades.loc[key_mask].copy()
        filter_defs = _candidate_filters(bucket_df)
        evaluated = _evaluate_bucket_filters(bucket_df, filter_defs)
        if evaluated.empty:
            continue
        for col in _bucket_key_cols():
            evaluated[col] = bucket[col]
        bucket_name = "__".join(str(bucket[col]).replace("/", "_") for col in _bucket_key_cols())
        evaluated.to_csv(out_dir / f"filters_{bucket_name}.csv", index=False)
        filter_rows.append(evaluated)
        for hypothesis in ["hidden_subregime", "family_specific", "phase_memory"]:
            scoped = evaluated.loc[evaluated["hypothesis"].eq(hypothesis)].head(int(args.top_filters))
            if not scoped.empty:
                top_recommendations.extend(scoped.head(2).to_dict("records"))

    combined_filters = pd.concat(filter_rows, ignore_index=True) if filter_rows else pd.DataFrame()
    if not combined_filters.empty:
        combined_filters.to_csv(out_dir / "all_filter_candidates.csv", index=False)

    summary = {
        "policy_file": str(policy_file),
        "model_file": str(params["model_file"]),
        "thresholds_file": str(params["thresholds_file"]),
        "weak_buckets": weak_buckets.to_dict("records"),
        "top_recommendations": top_recommendations,
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()

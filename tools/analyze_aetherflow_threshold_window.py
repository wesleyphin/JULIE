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

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES  # noqa: E402
import backtest_mes_et as bt  # noqa: E402
from aetherflow_features import build_feature_frame, resolve_setup_params  # noqa: E402
from aetherflow_strategy import (  # noqa: E402
    AetherFlowStrategy,
    _context_block_reason,
    _params_block_reason,
    _regime_name_from_row,
    _selection_score,
)
from tools.backtest_aetherflow_direct import (  # noqa: E402
    _load_base_features,
    _prepare_symbol_df,
    _resolve_source,
    _session_name,
    _simulate,
)


def _json_safe(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        out = float(value)
    except Exception:
        return int(default)
    return int(round(out)) if np.isfinite(out) else int(default)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _build_candidate_frame(
    *,
    strategy: AetherFlowStrategy,
    base_features: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    features = build_feature_frame(
        base_features=base_features,
        preferred_setup_families=strategy.allowed_setup_families,
    )
    if features.empty:
        return pd.DataFrame()
    features = features.loc[(features.index >= start_time) & (features.index <= end_time)].copy()
    features = features.loc[
        pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0
    ].copy()
    if features.empty:
        return pd.DataFrame()
    features["aetherflow_confidence"] = strategy._compute_probabilities(features)
    features["manifold_regime_name"] = features.apply(lambda row: _regime_name_from_row(row.to_dict()), axis=1)

    enriched_rows: list[dict] = []
    enriched_index: list[pd.Timestamp] = []
    for ts, row in features.iterrows():
        row_dict = row.to_dict()
        policy = strategy._policy_for_family(str(row_dict.get("setup_family", "") or ""), row_dict)
        if policy is None:
            row_dict["policy_threshold"] = np.nan
            row_dict["selection_score"] = np.nan
            row_dict["primary_reason"] = "setup_family_blocked"
            row_dict["secondary_reason"] = ""
            row_dict["session_name"] = _session_name(pd.Timestamp(ts))
            enriched_rows.append(row_dict)
            enriched_index.append(ts)
            continue

        params_row = pd.Series(dict(row_dict))
        if policy.get("sl_mult_override") is not None:
            params_row["setup_sl_mult"] = float(policy.get("sl_mult_override"))
        if policy.get("tp_mult_override") is not None:
            params_row["setup_tp_mult"] = float(policy.get("tp_mult_override"))
        if policy.get("horizon_bars_override") is not None:
            params_row["setup_horizon_bars"] = int(policy.get("horizon_bars_override"))
        params = resolve_setup_params(params_row)

        primary_reason = str(strategy._row_block_reason(row_dict) or "")
        secondary_reason = ""
        if primary_reason == "below_threshold":
            secondary_reason = str(
                _context_block_reason(row_dict, policy) or _params_block_reason(row_dict, policy, params=params) or ""
            )

        row_dict["policy_threshold"] = float(policy.get("threshold", strategy.threshold) or strategy.threshold)
        row_dict["selection_score"] = float(
            _selection_score(float(row_dict.get("aetherflow_confidence", 0.0) or 0.0), policy)
        )
        row_dict["primary_reason"] = primary_reason or "passed"
        row_dict["secondary_reason"] = secondary_reason
        row_dict["session_name"] = _session_name(pd.Timestamp(ts))
        enriched_rows.append(row_dict)
        enriched_index.append(ts)

    if not enriched_rows:
        return pd.DataFrame()
    return pd.DataFrame(enriched_rows, index=pd.DatetimeIndex(enriched_index))


def _build_signal_payloads(
    *,
    strategy: AetherFlowStrategy,
    candidate_df: pd.DataFrame,
    include_primary_reasons: set[str],
    require_no_secondary_reason: bool,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    work = candidate_df.loc[candidate_df["primary_reason"].astype(str).isin(sorted(include_primary_reasons))].copy()
    if require_no_secondary_reason:
        work = work.loc[work["secondary_reason"].astype(str).eq("")]
    if work.empty:
        return pd.DataFrame()

    signal_rows: list[dict] = []
    signal_index: list[pd.Timestamp] = []
    for ts, row in work.iterrows():
        row_dict = row.to_dict()
        policy = strategy._policy_for_family(str(row_dict.get("setup_family", "") or ""), row_dict)
        if policy is None:
            continue
        row_dict["entry_mode"] = str(policy.get("entry_mode", "market_next_bar") or "market_next_bar")
        row_dict["use_horizon_time_stop"] = bool(policy.get("use_horizon_time_stop", False))
        row_dict["aetherflow_threshold"] = float(policy.get("threshold", strategy.threshold) or strategy.threshold)
        if policy.get("sl_mult_override") is not None:
            row_dict["sl_mult_override"] = float(policy.get("sl_mult_override"))
        if policy.get("tp_mult_override") is not None:
            row_dict["tp_mult_override"] = float(policy.get("tp_mult_override"))
        if policy.get("horizon_bars_override") is not None:
            row_dict["horizon_bars_override"] = int(policy.get("horizon_bars_override"))
        early_exit = dict(policy.get("early_exit", {}) or {})
        if early_exit:
            row_dict["early_exit_enabled"] = bool(early_exit.get("enabled", False))
            row_dict["early_exit_exit_if_not_green_by"] = int(early_exit.get("exit_if_not_green_by", 0) or 0)
            row_dict["early_exit_max_profit_crosses"] = int(early_exit.get("max_profit_crosses", 0) or 0)
        signal_rows.append(row_dict)
        signal_index.append(ts)

    if not signal_rows:
        return pd.DataFrame()

    signals = pd.DataFrame(signal_rows, index=pd.DatetimeIndex(signal_index))
    signals = signals.sort_values(
        by=["selection_score", "aetherflow_confidence", "setup_strength"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    signals = signals.loc[~signals.index.duplicated(keep="first")].sort_index(kind="mergesort")
    return signals


def _simulate_bucket(
    *,
    symbol_df: pd.DataFrame,
    signals: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> dict:
    if signals.empty:
        return {
            "signals": 0,
            "equity": 0.0,
            "trades": 0,
            "winrate": 0.0,
            "max_drawdown": 0.0,
            "trade_log": [],
        }
    first_signal = signals.iloc[0].to_dict()
    use_horizon_time_stop = bool(first_signal.get("use_horizon_time_stop", False))
    stats = _simulate(
        df=symbol_df,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
        use_horizon_time_stop=use_horizon_time_stop,
    )
    return {
        "signals": int(len(signals)),
        "equity": float(stats.get("equity", 0.0) or 0.0),
        "trades": int(stats.get("trades", 0) or 0),
        "winrate": float(stats.get("winrate", 0.0) or 0.0),
        "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
        "trade_log": list(stats.get("trade_log", []) or []),
    }


def _simulate_standalone_rows(
    *,
    symbol_df: pd.DataFrame,
    rows_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    out_rows: list[dict] = []
    for ts, row in rows_df.iterrows():
        single_df = pd.DataFrame([row.to_dict()], index=pd.DatetimeIndex([ts]))
        stats = _simulate(
            df=symbol_df,
            signals=single_df,
            start_time=start_time,
            end_time=end_time,
            use_horizon_time_stop=bool(row.get("use_horizon_time_stop", False)),
        )
        trade_log = list(stats.get("trade_log", []) or [])
        trade = dict(trade_log[0]) if trade_log else {}
        out_rows.append(
            {
                "signal_time": pd.Timestamp(ts).isoformat(),
                "family": str(row.get("setup_family", "") or ""),
                "side": "LONG" if _coerce_float(row.get("candidate_side", 0.0), 0.0) > 0.0 else "SHORT",
                "session": str(row.get("session_name", "") or ""),
                "regime": str(row.get("manifold_regime_name", "") or ""),
                "confidence": _coerce_float(row.get("aetherflow_confidence", 0.0), 0.0),
                "threshold": _coerce_float(row.get("policy_threshold", np.nan), np.nan),
                "margin_to_threshold": _coerce_float(row.get("aetherflow_confidence", 0.0), 0.0)
                - _coerce_float(row.get("policy_threshold", 0.0), 0.0),
                "primary_reason": str(row.get("primary_reason", "") or ""),
                "secondary_reason": str(row.get("secondary_reason", "") or ""),
                "selection_score": _coerce_float(row.get("selection_score", 0.0), 0.0),
                "pnl_net": trade.get("pnl_net"),
                "exit_reason": trade.get("exit_reason"),
                "bars_held": trade.get("bars_held"),
            }
        )
    return pd.DataFrame(out_rows)


def _correlation_block(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "pearson_conf_pnl": None,
            "spearman_conf_pnl": None,
            "avg_conf_win": None,
            "avg_conf_loss": None,
            "net_pnl_sum": 0.0,
        }
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    pnl = pd.to_numeric(df["pnl_net"], errors="coerce")
    winners = df.loc[pnl > 0.0]
    losers = df.loc[pnl <= 0.0]
    if len(df) >= 2:
        pearson = float(conf.corr(pnl, method="pearson"))
        spearman = float(conf.corr(pnl, method="spearman"))
    else:
        pearson = None
        spearman = None
    return {
        "rows": int(len(df)),
        "pearson_conf_pnl": pearson,
        "spearman_conf_pnl": spearman,
        "avg_conf_win": float(pd.to_numeric(winners["confidence"], errors="coerce").mean()) if len(winners) else None,
        "avg_conf_loss": float(pd.to_numeric(losers["confidence"], errors="coerce").mean()) if len(losers) else None,
        "net_pnl_sum": float(pd.to_numeric(df["pnl_net"], errors="coerce").fillna(0.0).sum()),
    }


def _bucket_trade_summary(stats: dict) -> dict:
    trade_log = list(stats.get("trade_log", []) or [])
    return {
        "signals": int(stats.get("signals", 0) or 0),
        "trades": int(stats.get("trades", 0) or 0),
        "equity": float(stats.get("equity", 0.0) or 0.0),
        "winrate": float(stats.get("winrate", 0.0) or 0.0),
        "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
        "trade_log": [
            {
                "entry_time": item.get("entry_time"),
                "family": item.get("aetherflow_setup_family"),
                "side": item.get("side"),
                "conf": item.get("aetherflow_confidence"),
                "pnl_net": item.get("pnl_net"),
                "exit_reason": item.get("exit_reason"),
            }
            for item in trade_log
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AetherFlow below-threshold behavior over a date window.")
    parser.add_argument(
        "--base-features",
        default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--report-dir", default="backtest_reports/aetherflow_threshold_window_audit")
    parser.add_argument("--history-buffer-days", type=int, default=14)
    args = parser.parse_args()

    start_time = bt.parse_user_datetime(str(args.start), bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(str(args.end), bt.NY_TZ, is_end=True)
    if start_time > end_time:
        raise SystemExit("Start must be before end.")

    strategy = AetherFlowStrategy()
    if not strategy.model_loaded or strategy.model_bundle is None:
        raise SystemExit("AetherFlowStrategy failed to load its configured live bundle.")

    source_path = _resolve_source(bt.DEFAULT_CSV_NAME)
    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single").strip().lower(),
        str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume").strip().lower(),
        int(args.history_buffer_days),
    )
    base_features = _load_base_features(
        ROOT / str(args.base_features),
        pd.Timestamp(symbol_df.index.min()),
        pd.Timestamp(symbol_df.index.max()),
    )
    candidate_df = _build_candidate_frame(
        strategy=strategy,
        base_features=base_features,
        start_time=start_time,
        end_time=end_time,
    )
    if candidate_df.empty:
        raise SystemExit("No AetherFlow candidates found in the requested window.")

    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    day_values = sorted({pd.Timestamp(ts).date() for ts in pd.DatetimeIndex(candidate_df.index)})
    day_rows: list[dict] = []
    standalone_frames: list[pd.DataFrame] = []

    for day_value in day_values:
        day_start = pd.Timestamp(day_value, tz=bt.NY_TZ)
        day_end = day_start + pd.Timedelta(hours=23, minutes=59)
        day_candidates = candidate_df.loc[
            (candidate_df.index >= day_start) & (candidate_df.index <= min(day_end, end_time))
        ].copy()
        if day_candidates.empty:
            continue

        passed_signals = _build_signal_payloads(
            strategy=strategy,
            candidate_df=day_candidates,
            include_primary_reasons={"passed"},
            require_no_secondary_reason=False,
        )
        threshold_only_signals = _build_signal_payloads(
            strategy=strategy,
            candidate_df=day_candidates,
            include_primary_reasons={"below_threshold"},
            require_no_secondary_reason=True,
        )
        all_below_signals = _build_signal_payloads(
            strategy=strategy,
            candidate_df=day_candidates,
            include_primary_reasons={"below_threshold"},
            require_no_secondary_reason=False,
        )

        passed_stats = _simulate_bucket(
            symbol_df=symbol_df,
            signals=passed_signals,
            start_time=day_start,
            end_time=min(day_end, end_time),
        )
        threshold_only_stats = _simulate_bucket(
            symbol_df=symbol_df,
            signals=threshold_only_signals,
            start_time=day_start,
            end_time=min(day_end, end_time),
        )
        all_below_stats = _simulate_bucket(
            symbol_df=symbol_df,
            signals=all_below_signals,
            start_time=day_start,
            end_time=min(day_end, end_time),
        )
        combined_threshold_only_signals = pd.concat([passed_signals, threshold_only_signals]).sort_index(kind="mergesort")
        combined_threshold_only_signals = combined_threshold_only_signals.loc[
            ~combined_threshold_only_signals.index.duplicated(keep="first")
        ]
        combined_threshold_only_stats = _simulate_bucket(
            symbol_df=symbol_df,
            signals=combined_threshold_only_signals,
            start_time=day_start,
            end_time=min(day_end, end_time),
        )

        day_threshold_only_candidates = day_candidates.loc[
            day_candidates["primary_reason"].astype(str).eq("below_threshold")
            & day_candidates["secondary_reason"].astype(str).eq("")
        ].copy()
        standalone_frame = _simulate_standalone_rows(
            symbol_df=symbol_df,
            rows_df=threshold_only_signals,
            start_time=day_start,
            end_time=min(day_end, end_time),
        )
        if not standalone_frame.empty:
            standalone_frame["trade_day"] = str(day_value)
            standalone_frames.append(standalone_frame)

        day_rows.append(
            {
                "trade_day": str(day_value),
                "candidate_rows": int(len(day_candidates)),
                "passed_rows": int(day_candidates["primary_reason"].astype(str).eq("passed").sum()),
                "below_threshold_rows": int(day_candidates["primary_reason"].astype(str).eq("below_threshold").sum()),
                "threshold_only_rows": int(len(day_threshold_only_candidates)),
                "secondary_blocked_below_threshold_rows": int(
                    day_candidates["primary_reason"].astype(str).eq("below_threshold").sum() - len(day_threshold_only_candidates)
                ),
                "current_policy_signals": int(len(passed_signals)),
                "current_policy_equity": float(passed_stats["equity"]),
                "current_policy_trades": int(passed_stats["trades"]),
                "threshold_only_signals": int(len(threshold_only_signals)),
                "threshold_only_equity": float(threshold_only_stats["equity"]),
                "threshold_only_trades": int(threshold_only_stats["trades"]),
                "combined_current_plus_threshold_only_signals": int(len(combined_threshold_only_signals)),
                "combined_current_plus_threshold_only_equity": float(combined_threshold_only_stats["equity"]),
                "combined_current_plus_threshold_only_trades": int(combined_threshold_only_stats["trades"]),
                "all_below_signals": int(len(all_below_signals)),
                "all_below_equity": float(all_below_stats["equity"]),
                "all_below_trades": int(all_below_stats["trades"]),
            }
        )

    day_summary_df = pd.DataFrame(day_rows).sort_values("trade_day", kind="mergesort")
    standalone_df = pd.concat(standalone_frames, ignore_index=True) if standalone_frames else pd.DataFrame()
    below_rows_df = candidate_df.loc[candidate_df["primary_reason"].astype(str).eq("below_threshold")].copy()
    below_rows_df["trade_day"] = [str(pd.Timestamp(ts).date()) for ts in pd.DatetimeIndex(below_rows_df.index)]
    below_rows_df["signal_time"] = [pd.Timestamp(ts).isoformat() for ts in pd.DatetimeIndex(below_rows_df.index)]
    below_rows_export = below_rows_df[
        [
            "trade_day",
            "signal_time",
            "setup_family",
            "session_name",
            "manifold_regime_name",
            "candidate_side",
            "aetherflow_confidence",
            "policy_threshold",
            "selection_score",
            "primary_reason",
            "secondary_reason",
            "setup_strength",
        ]
    ].copy()
    below_rows_export.rename(
        columns={
            "setup_family": "family",
            "manifold_regime_name": "regime",
            "candidate_side": "side_num",
            "aetherflow_confidence": "confidence",
            "policy_threshold": "threshold",
        },
        inplace=True,
    )
    below_rows_export["margin_to_threshold"] = (
        pd.to_numeric(below_rows_export["confidence"], errors="coerce")
        - pd.to_numeric(below_rows_export["threshold"], errors="coerce")
    )

    overall_corr = _correlation_block(standalone_df)
    by_family_corr = {}
    if not standalone_df.empty:
        for family_name, fam_df in standalone_df.groupby("family", dropna=False):
            by_family_corr[str(family_name)] = _correlation_block(fam_df.copy())

    secondary_counts = (
        below_rows_export["secondary_reason"].replace("", "none").value_counts().to_dict()
        if not below_rows_export.empty
        else {}
    )

    summary = {
        "window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "symbol": symbol,
            "symbol_distribution": symbol_distribution,
        },
        "policy": {
            "threshold": float(strategy.threshold),
            "min_confidence": float(strategy.min_confidence),
            "allowed_setup_families": sorted(str(item) for item in (strategy.allowed_setup_families or set())),
            "hazard_block_regimes": sorted(str(item) for item in (strategy.hazard_block_regimes or set())),
        },
        "totals": {
            "days": int(len(day_summary_df)),
            "candidate_rows": int(len(candidate_df)),
            "passed_rows": int(candidate_df["primary_reason"].astype(str).eq("passed").sum()),
            "below_threshold_rows": int(candidate_df["primary_reason"].astype(str).eq("below_threshold").sum()),
            "threshold_only_rows": int(
                (
                    candidate_df["primary_reason"].astype(str).eq("below_threshold")
                    & candidate_df["secondary_reason"].astype(str).eq("")
                ).sum()
            ),
            "secondary_blocked_below_threshold_rows": int(
                (
                    candidate_df["primary_reason"].astype(str).eq("below_threshold")
                    & candidate_df["secondary_reason"].astype(str).ne("")
                ).sum()
            ),
            "current_policy_equity_sum": float(pd.to_numeric(day_summary_df["current_policy_equity"], errors="coerce").fillna(0.0).sum()) if not day_summary_df.empty else 0.0,
            "threshold_only_equity_sum": float(pd.to_numeric(day_summary_df["threshold_only_equity"], errors="coerce").fillna(0.0).sum()) if not day_summary_df.empty else 0.0,
            "combined_current_plus_threshold_only_equity_sum": float(
                pd.to_numeric(day_summary_df["combined_current_plus_threshold_only_equity"], errors="coerce").fillna(0.0).sum()
            ) if not day_summary_df.empty else 0.0,
            "all_below_equity_sum": float(pd.to_numeric(day_summary_df["all_below_equity"], errors="coerce").fillna(0.0).sum()) if not day_summary_df.empty else 0.0,
            "days_threshold_only_beats_current": int(
                (
                    pd.to_numeric(day_summary_df["combined_current_plus_threshold_only_equity"], errors="coerce").fillna(0.0)
                    > pd.to_numeric(day_summary_df["current_policy_equity"], errors="coerce").fillna(0.0)
                ).sum()
            ) if not day_summary_df.empty else 0,
            "overall_threshold_only_correlation": overall_corr,
            "threshold_only_correlation_by_family": by_family_corr,
            "secondary_reason_counts": secondary_counts,
        },
        "best_threshold_only_days": (
            day_summary_df.sort_values("threshold_only_equity", ascending=False, kind="mergesort")
            .head(10)
            .to_dict("records")
            if not day_summary_df.empty
            else []
        ),
        "worst_threshold_only_days": (
            day_summary_df.sort_values("threshold_only_equity", ascending=True, kind="mergesort")
            .head(10)
            .to_dict("records")
            if not day_summary_df.empty
            else []
        ),
    }

    day_summary_path = report_dir / "day_summary.csv"
    below_rows_path = report_dir / "below_threshold_rows.csv"
    standalone_path = report_dir / "threshold_only_standalone.csv"
    summary_path = report_dir / "summary.json"

    day_summary_df.to_csv(day_summary_path, index=False)
    below_rows_export.to_csv(below_rows_path, index=False)
    standalone_df.to_csv(standalone_path, index=False)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    print(f"day_summary_csv={day_summary_path}")
    print(f"below_threshold_rows_csv={below_rows_path}")
    print(f"threshold_only_standalone_csv={standalone_path}")
    print(f"summary_json={summary_path}")
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()

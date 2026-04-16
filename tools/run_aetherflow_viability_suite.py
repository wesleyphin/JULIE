import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES
import backtest_mes_et as bt
from aetherflow_features import build_feature_frame
from aetherflow_model_bundle import predict_bundle_probabilities
from aetherflow_strategy import REGIME_ID_TO_NAME
from tools.backtest_aetherflow_direct import (
    _load_base_features,
    _load_model_bundle,
    _prepare_symbol_df,
    _resolve_source,
    _simulate,
)

VALID_FAMILIES = {
    "compression_release",
    "aligned_flow",
    "exhaustion_reversal",
    "transition_burst",
}


def _resolve_fold_artifact_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip()).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    normalized = str(raw_path or "").replace("/", "\\")
    marker = "\\artifacts\\"
    idx = normalized.lower().find(marker)
    if idx >= 0:
        candidate = ROOT / normalized[idx + 1 :]
        if candidate.exists():
            return candidate.resolve()
    marker = "artifacts\\"
    idx = normalized.lower().find(marker)
    if idx >= 0:
        candidate = ROOT / normalized[idx:]
        if candidate.exists():
            return candidate.resolve()
    candidate = (ROOT / path).resolve() if not path.is_absolute() else path
    return candidate


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _test_bounds(test_years: list[int]) -> tuple[str, str]:
    years = sorted(int(y) for y in test_years)
    return f"{years[0]:04d}-01-01", f"{years[-1]:04d}-12-31"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _json_safe(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _trade_pnl(trade: dict) -> float:
    for key in ("pnl_net", "pnl_dollars", "pnl"):
        value = _safe_float(trade.get(key), float("nan"))
        if math.isfinite(value):
            return float(value)
    return 0.0


def _profit_factor(pnls: list[float]) -> float:
    arr = np.asarray(pnls, dtype=float)
    if arr.size <= 0:
        return 0.0
    gross_profit = float(np.sum(arr[arr > 0.0]))
    gross_loss = float(-np.sum(arr[arr < 0.0]))
    if gross_loss <= 1e-12:
        return 999.0 if gross_profit > 0.0 else 0.0
    return float(gross_profit / gross_loss)


def _daily_sharpe_sortino(day_pnls: np.ndarray) -> tuple[float, float]:
    if day_pnls.size <= 1:
        return 0.0, 0.0
    daily_mean = float(np.mean(day_pnls))
    daily_std = float(np.std(day_pnls, ddof=1))
    daily_sharpe = float((daily_mean / daily_std) * math.sqrt(252.0)) if daily_std > 1e-12 else 0.0
    downside = np.minimum(day_pnls, 0.0)
    downside_rms = float(math.sqrt(float(np.mean(np.square(downside)))))
    daily_sortino = float((daily_mean / downside_rms) * math.sqrt(252.0)) if downside_rms > 1e-12 else 0.0
    return daily_sharpe, daily_sortino


def _max_drawdown_from_day_pnls(day_pnls: np.ndarray) -> float:
    if day_pnls.size <= 0:
        return 0.0
    equity_curve = np.cumsum(day_pnls)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity_curve)))
    drawdowns = running_peak[1:] - equity_curve
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def _bootstrap_metric_summary(trade_log: list[dict], simulations: int, seed: int) -> dict:
    day_trade_pnls: dict[str, list[float]] = defaultdict(list)
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        day_key = bt._trade_timestamp_to_ny_day(trade.get("exit_time") or trade.get("entry_time"))
        if not day_key:
            continue
        day_trade_pnls[day_key].append(_trade_pnl(trade))

    if not day_trade_pnls:
        return {
            "enabled": True,
            "status": "empty",
            "simulations": int(simulations),
            "trade_days": 0,
        }

    rng = np.random.default_rng(int(seed))
    ordered_days = sorted(day_trade_pnls.keys())
    day_trade_lists = [day_trade_pnls[key] for key in ordered_days]
    day_nets = np.asarray([float(sum(pnls)) for pnls in day_trade_lists], dtype=float)
    day_count = int(len(day_trade_lists))

    net_pnls = np.empty(int(simulations), dtype=float)
    profit_factors = np.empty(int(simulations), dtype=float)
    sharpes = np.empty(int(simulations), dtype=float)
    sortinos = np.empty(int(simulations), dtype=float)
    max_drawdowns = np.empty(int(simulations), dtype=float)

    for idx in range(int(simulations)):
        sampled_idx = rng.integers(0, day_count, size=day_count, endpoint=False)
        sampled_day_nets = day_nets[sampled_idx]
        sampled_trade_pnls: list[float] = []
        for item in sampled_idx.tolist():
            sampled_trade_pnls.extend(day_trade_lists[int(item)])
        net_pnls[idx] = float(np.sum(sampled_day_nets))
        profit_factors[idx] = float(_profit_factor(sampled_trade_pnls))
        sharpe, sortino = _daily_sharpe_sortino(sampled_day_nets)
        sharpes[idx] = float(sharpe)
        sortinos[idx] = float(sortino)
        max_drawdowns[idx] = float(_max_drawdown_from_day_pnls(sampled_day_nets))

    def _metric(values: np.ndarray, *, threshold: float | None = None) -> dict:
        out = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
        if threshold is not None:
            out["probability_above_threshold"] = float(np.mean(values > float(threshold)))
        return out

    return {
        "enabled": True,
        "status": "ok",
        "simulations": int(simulations),
        "seed": int(seed),
        "trade_days": int(day_count),
        "net_pnl": _metric(net_pnls, threshold=0.0),
        "profit_factor": _metric(profit_factors, threshold=1.0),
        "daily_sharpe": _metric(sharpes, threshold=1.0),
        "daily_sortino": _metric(sortinos, threshold=1.0),
        "max_drawdown": _metric(max_drawdowns),
    }


def _trade_counts(trade_log: list[dict], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        value = str(trade.get(field) or "UNKNOWN").strip().upper()
        counts[value] += 1
    return dict(sorted(((key, int(val)) for key, val in counts.items()), key=lambda kv: (-kv[1], kv[0])))


def _load_walkforward_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds", []) or []
    if not folds:
        raise RuntimeError(f"No folds found in {path}")
    return payload


def _normalize_int_set(value) -> list[int] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(float(item)))
        except Exception:
            continue
    deduped = sorted({item for item in out})
    return deduped or None


def _normalize_str_set(value) -> list[str] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out = []
    for item in items:
        text = str(item).strip().upper()
        if text:
            out.append(text)
    deduped = sorted({item for item in out})
    return deduped or None


def _normalize_break_even_config(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("break_even must be an object when provided.")
    return {
        "enabled": bool(value.get("enabled", False)),
        "trigger_pct": max(0.0, _safe_float(value.get("trigger_pct"), 0.0)),
        "buffer_ticks": max(0, int(round(_safe_float(value.get("buffer_ticks"), 0.0)))),
        "trail_pct": max(0.0, _safe_float(value.get("trail_pct"), 0.0)),
        "activate_on_next_bar": bool(value.get("activate_on_next_bar", True)),
    }


def _normalize_early_exit_config(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("early_exit must be an object when provided.")
    return {
        "enabled": bool(value.get("enabled", False)),
        "exit_if_not_green_by": max(0, int(round(_safe_float(value.get("exit_if_not_green_by"), 0.0)))),
        "max_profit_crosses": max(0, int(round(_safe_float(value.get("max_profit_crosses"), 0.0)))),
    }


def _normalize_family_policy_mapping(policy, *, allow_match_fields: bool, allow_rules: bool) -> dict:
    if not isinstance(policy, dict):
        raise RuntimeError("Family policy must be an object.")
    raw = dict(policy or {})
    out: dict[str, Any] = {}

    if "threshold" in raw:
        out["threshold"] = None if raw.get("threshold") is None else float(raw.get("threshold"))
    if "allowed_session_ids" in raw:
        out["allowed_session_ids"] = _normalize_int_set(raw.get("allowed_session_ids"))
    if "allowed_regimes" in raw:
        out["allowed_regimes"] = _normalize_str_set(raw.get("allowed_regimes"))
    if "blocked_regimes" in raw:
        out["blocked_regimes"] = _normalize_str_set(raw.get("blocked_regimes"))
    if "break_even" in raw:
        out["break_even"] = _normalize_break_even_config(raw.get("break_even")) or {}
    if "early_exit" in raw:
        out["early_exit"] = _normalize_early_exit_config(raw.get("early_exit")) or {}

    if allow_match_fields:
        if "name" in raw and str(raw.get("name", "") or "").strip():
            out["name"] = str(raw.get("name", "") or "").strip()
        if "match_session_ids" in raw:
            out["match_session_ids"] = _normalize_int_set(raw.get("match_session_ids"))
        if "match_regimes" in raw:
            out["match_regimes"] = _normalize_str_set(raw.get("match_regimes"))

    if allow_rules:
        raw_rules = raw.get("policy_rules", raw.get("rules"))
        if raw_rules is not None:
            if not isinstance(raw_rules, list):
                raise RuntimeError("policy_rules must be a list when provided.")
            out["policy_rules"] = [
                _normalize_family_policy_mapping(rule, allow_match_fields=True, allow_rules=False)
                for rule in raw_rules
                if isinstance(rule, dict)
            ]

    return out


def _merge_family_policy_layers(*layers: dict) -> dict:
    merged: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for key, value in layer.items():
            if key in {"policy_rules", "rules", "match_session_ids", "match_regimes", "name"}:
                continue
            if isinstance(value, dict):
                merged[key] = dict(value)
            elif isinstance(value, list):
                merged[key] = list(value)
            else:
                merged[key] = value
    return merged


def _rule_match_mask(features: pd.DataFrame, rule: dict) -> pd.Series:
    mask = pd.Series(True, index=features.index, dtype=bool)
    if "match_session_ids" in rule:
        match_session_ids = rule.get("match_session_ids")
        if match_session_ids:
            session_series = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
            mask &= session_series.isin(match_session_ids)
    if "match_regimes" in rule:
        match_regimes = rule.get("match_regimes")
        if match_regimes:
            mask &= features["manifold_regime_name"].isin(match_regimes)
    return mask


def _apply_family_policy_frame(
    features: pd.DataFrame,
    *,
    policy: dict,
    default_threshold: float,
    rule_name: str | None,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    scoped = features.copy()

    allowed_session_ids = policy.get("allowed_session_ids")
    if allowed_session_ids:
        session_series = pd.to_numeric(scoped.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
        scoped = scoped.loc[session_series.isin(allowed_session_ids)]
    if scoped.empty:
        return pd.DataFrame()

    allowed_regimes = policy.get("allowed_regimes")
    blocked_regimes = policy.get("blocked_regimes")
    if allowed_regimes:
        scoped = scoped.loc[scoped["manifold_regime_name"].isin(allowed_regimes)]
    if blocked_regimes:
        scoped = scoped.loc[~scoped["manifold_regime_name"].isin(blocked_regimes)]
    if scoped.empty:
        return pd.DataFrame()

    threshold = float(policy.get("threshold") if policy.get("threshold") is not None else default_threshold)
    scoped["policy_threshold"] = float(threshold)
    scoped = scoped.loc[scoped["aetherflow_confidence"] >= float(threshold)]
    if scoped.empty:
        return pd.DataFrame()

    break_even_cfg = policy.get("break_even") or {}
    early_exit_cfg = policy.get("early_exit") or {}
    if break_even_cfg:
        scoped["break_even_enabled"] = bool(break_even_cfg.get("enabled", False))
        scoped["break_even_trigger_pct"] = float(break_even_cfg.get("trigger_pct", 0.0) or 0.0)
        scoped["break_even_buffer_ticks"] = int(break_even_cfg.get("buffer_ticks", 0) or 0)
        scoped["break_even_trail_pct"] = float(break_even_cfg.get("trail_pct", 0.0) or 0.0)
        scoped["break_even_activate_on_next_bar"] = bool(break_even_cfg.get("activate_on_next_bar", True))
    if early_exit_cfg:
        scoped["early_exit_enabled"] = bool(early_exit_cfg.get("enabled", False))
        scoped["early_exit_exit_if_not_green_by"] = int(early_exit_cfg.get("exit_if_not_green_by", 0) or 0)
        scoped["early_exit_max_profit_crosses"] = int(early_exit_cfg.get("max_profit_crosses", 0) or 0)
    if rule_name:
        scoped["policy_rule_name"] = str(rule_name)
    return scoped


def _normalize_family_policies(raw: dict) -> dict[str, dict]:
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("Variant must define a non-empty family_policies mapping.")
    normalized: dict[str, dict] = {}
    for family_name, policy in raw.items():
        family_key = str(family_name or "").strip()
        if family_key not in VALID_FAMILIES:
            raise RuntimeError(f"Unknown AetherFlow family in policy: {family_key}")
        normalized[family_key] = _normalize_family_policy_mapping(
            policy,
            allow_match_fields=False,
            allow_rules=True,
        )
    return normalized


def _load_variants(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_variants = payload.get("variants", payload if isinstance(payload, list) else []) or []
    if not isinstance(raw_variants, list) or not raw_variants:
        raise RuntimeError(f"No variants found in {path}")
    variants: list[dict] = []
    for item in raw_variants:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        variants.append(
            {
                "name": name,
                "description": str(item.get("description", "") or ""),
                "family_policies": _normalize_family_policies(item.get("family_policies", {})),
                "selection_mode": str(item.get("selection_mode", "highest_confidence") or "highest_confidence"),
            }
        )
    if not variants:
        raise RuntimeError(f"No usable variants found in {path}")
    return variants


def _prepare_family_signals(
    *,
    family_name: str,
    policy: dict,
    base_features: pd.DataFrame,
    model,
    feature_columns: list[str],
    default_threshold: float,
) -> pd.DataFrame:
    features = build_feature_frame(
        base_features=base_features,
        preferred_setup_families={family_name},
    )
    if features.empty:
        return pd.DataFrame()
    features = features.loc[
        (features["setup_family"].astype(str) == str(family_name))
        & (pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0)
    ].copy()
    if features.empty:
        return pd.DataFrame()

    features["aetherflow_confidence"] = predict_bundle_probabilities(model, features)
    regime_names = pd.to_numeric(features.get("manifold_regime_id"), errors="coerce").fillna(-1).round().astype(int)
    regime_names = regime_names.map(REGIME_ID_TO_NAME).fillna("").astype(str).str.upper()
    features["manifold_regime_name"] = regime_names
    base_policy = {
        key: value
        for key, value in dict(policy or {}).items()
        if key not in {"policy_rules", "rules"}
    }
    policy_rules = list(dict(policy or {}).get("policy_rules", []) or [])

    scoped_frames: list[pd.DataFrame] = []
    remaining = features.copy()
    for rule in policy_rules:
        match_mask = _rule_match_mask(remaining, rule)
        if not bool(match_mask.any()):
            continue
        matched = remaining.loc[match_mask].copy()
        remaining = remaining.loc[~match_mask].copy()
        rule_name = str(rule.get("name", "") or "").strip() or None
        effective_policy = _merge_family_policy_layers(base_policy, rule)
        scoped = _apply_family_policy_frame(
            matched,
            policy=effective_policy,
            default_threshold=float(default_threshold),
            rule_name=rule_name,
        )
        if not scoped.empty:
            scoped_frames.append(scoped)

    base_scoped = _apply_family_policy_frame(
        remaining,
        policy=base_policy,
        default_threshold=float(default_threshold),
        rule_name=None,
    )
    if not base_scoped.empty:
        scoped_frames.append(base_scoped)
    if not scoped_frames:
        return pd.DataFrame()
    merged = pd.concat(scoped_frames, axis=0).sort_index()
    merged["family_policy_name"] = str(family_name)
    return merged


def _merge_variant_signals(frames: list[pd.DataFrame], selection_mode: str) -> pd.DataFrame:
    usable = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable, axis=0).sort_index()
    mode = str(selection_mode or "highest_confidence").strip().lower()
    if mode == "highest_setup_strength":
        merged = merged.sort_values(
            by=["aetherflow_confidence", "setup_strength"],
            ascending=[False, False],
            kind="mergesort",
        )
    else:
        merged = merged.sort_values(
            by=["aetherflow_confidence", "setup_strength"],
            ascending=[False, False],
            kind="mergesort",
        )
    merged = merged.loc[~merged.index.duplicated(keep="first")]
    return merged.sort_index()


def _variant_summary(stats: dict) -> dict:
    return {
        "equity": float(stats.get("equity", 0.0) or 0.0),
        "trades": int(stats.get("trades", 0) or 0),
        "wins": int(stats.get("wins", 0) or 0),
        "losses": int(stats.get("losses", 0) or 0),
        "winrate": float(stats.get("winrate", 0.0) or 0.0),
        "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
        "gross_profit": float(stats.get("gross_profit", 0.0) or 0.0),
        "gross_loss": float(stats.get("gross_loss", 0.0) or 0.0),
        "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0),
        "avg_trade_net": float(stats.get("avg_trade_net", 0.0) or 0.0),
        "trade_sqn": float(stats.get("trade_sqn", 0.0) or 0.0),
        "trade_sharpe_like": float(stats.get("trade_sharpe_like", 0.0) or 0.0),
        "daily_sharpe": float(stats.get("daily_sharpe", 0.0) or 0.0),
        "daily_sortino": float(stats.get("daily_sortino", 0.0) or 0.0),
        "trading_days": int(stats.get("trading_days", 0) or 0),
    }


def _aggregate_variant_results(fold_results: list[dict]) -> dict:
    if not fold_results:
        return {}
    summaries = [result.get("summary", {}) or {} for result in fold_results]
    equities = np.asarray([_safe_float(item.get("equity"), 0.0) for item in summaries], dtype=float)
    trades = np.asarray([int(item.get("trades", 0) or 0) for item in summaries], dtype=int)
    avg_trade_net = np.asarray([_safe_float(item.get("avg_trade_net"), 0.0) for item in summaries], dtype=float)
    profit_factors = np.asarray([_safe_float(item.get("profit_factor"), 0.0) for item in summaries], dtype=float)
    drawdowns = np.asarray([_safe_float(item.get("max_drawdown"), 0.0) for item in summaries], dtype=float)
    sharpe = np.asarray([_safe_float(item.get("daily_sharpe"), 0.0) for item in summaries], dtype=float)
    sortino = np.asarray([_safe_float(item.get("daily_sortino"), 0.0) for item in summaries], dtype=float)
    mc_prob = np.asarray(
        [
            _safe_float(
                ((result.get("monte_carlo_trade_day_bootstrap", {}) or {}).get("net_pnl", {}) or {}).get("probability_above_threshold"),
                0.0,
            )
            for result in fold_results
        ],
        dtype=float,
    )
    return {
        "fold_count": int(len(fold_results)),
        "positive_folds": int(np.sum(equities > 0.0)),
        "negative_folds": int(np.sum(equities < 0.0)),
        "total_equity": float(np.sum(equities)),
        "mean_equity": float(np.mean(equities)),
        "median_equity": float(np.median(equities)),
        "total_trades": int(np.sum(trades)),
        "weighted_avg_trade_net": float(np.sum(equities) / max(1, int(np.sum(trades)))),
        "mean_avg_trade_net": float(np.mean(avg_trade_net)),
        "mean_profit_factor": float(np.mean(profit_factors)),
        "worst_max_drawdown": float(np.max(drawdowns)),
        "mean_daily_sharpe": float(np.mean(sharpe)),
        "mean_daily_sortino": float(np.mean(sortino)),
        "mean_mc_day_bootstrap_prob_above_zero": float(np.mean(mc_prob)),
    }


def _run_fold_variant(
    *,
    fold: dict,
    variant: dict,
    source_path: Path,
    base_features_path: Path,
    symbol_mode: str,
    symbol_method: str,
    history_buffer_days: int,
    output_dir: Path,
    mc_simulations: int,
    mc_seed: int,
    skip_report_export: bool,
    use_horizon_time_stop: bool,
) -> dict:
    artifacts = fold.get("artifacts") or {}
    model_path = _resolve_fold_artifact_path(str(artifacts.get("model_file", "") or ""))
    thresholds_path = _resolve_fold_artifact_path(str(artifacts.get("thresholds_file", "") or ""))
    if not model_path.exists() or not thresholds_path.exists():
        raise RuntimeError(f"Missing fold artifacts for {fold.get('fold')}: {model_path} / {thresholds_path}")

    test_years = [int(y) for y in (fold.get("test_years") or [])]
    start_text, end_text = _test_bounds(test_years)
    start_time = bt.parse_user_datetime(start_text, bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(end_text, bt.NY_TZ, is_end=True)
    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(symbol_mode or "single").strip().lower(),
        str(symbol_method or "volume").strip().lower(),
        int(history_buffer_days),
    )
    model, feature_columns, default_threshold, _ = _load_model_bundle(model_path, thresholds_path)
    base_features = _load_base_features(base_features_path, pd.Timestamp(symbol_df.index.min()), pd.Timestamp(symbol_df.index.max()))

    family_frames = []
    for family_name, policy in (variant.get("family_policies", {}) or {}).items():
        family_frames.append(
            _prepare_family_signals(
                family_name=str(family_name),
                policy=dict(policy or {}),
                base_features=base_features,
                model=model,
                feature_columns=list(feature_columns),
                default_threshold=float(default_threshold),
            )
        )
    signals = _merge_variant_signals(family_frames, str(variant.get("selection_mode", "highest_confidence")))
    stats = _simulate(
        df=symbol_df,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
        use_horizon_time_stop=bool(use_horizon_time_stop),
    )
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    fold_dir = output_dir / str(variant["name"]) / str(fold["fold"])
    fold_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path | None = None
    if not bool(skip_report_export):
        report_path = bt.save_backtest_report(
            stats,
            symbol,
            start_time,
            end_time,
            output_dir=fold_dir,
        )
    trade_log = stats.get("trade_log", []) or []
    mc_trade_order = bt._build_monte_carlo_summary(
        trade_log,
        stats,
        simulations=max(1, int(mc_simulations)),
        seed=int(mc_seed),
        starting_balance=float(bt.BACKTEST_MONTE_CARLO_START_BALANCE),
    )
    mc_day_bootstrap = _bootstrap_metric_summary(
        trade_log,
        simulations=max(1, int(mc_simulations)),
        seed=int(mc_seed),
    )
    result = {
        "fold": str(fold.get("fold", "")),
        "test_years": test_years,
        "model_file": str(model_path),
        "thresholds_file": str(thresholds_path),
        "symbol": symbol,
        "symbol_distribution": symbol_distribution,
        "signals": int(len(signals)),
        "summary": _variant_summary(stats),
        "trade_families": _trade_counts(trade_log, "aetherflow_setup_family"),
        "trade_regimes": _trade_counts(trade_log, "aetherflow_regime"),
        "trade_sessions": _trade_counts(trade_log, "session"),
        "monte_carlo_trade_order": mc_trade_order,
        "monte_carlo_trade_day_bootstrap": mc_day_bootstrap,
        "report_path": str(report_path) if report_path is not None else "",
    }
    lightweight_path = fold_dir / "summary.json"
    lightweight_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    result["summary_path"] = str(lightweight_path)
    print(
        f"[suite] {variant['name']} {fold.get('fold')} trades={result['summary']['trades']} "
        f"equity={result['summary']['equity']:.2f} pf={result['summary']['profit_factor']:.3f} "
        f"mc_day_p>0={_safe_float(((mc_day_bootstrap.get('net_pnl', {}) or {}).get('probability_above_threshold')), 0.0):.3f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact AetherFlow OOS family-viability backtests with Monte Carlo summaries."
    )
    parser.add_argument(
        "--walkforward-report",
        default="artifacts/aetherflow_walkforward_fullrange_v2_2024plus_seedfix/walkforward_report.json",
        help="AetherFlow walkforward report containing per-fold artifacts.",
    )
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME, help="Parquet/CSV source path.")
    parser.add_argument("--base-features", default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    parser.add_argument(
        "--variants-file",
        default="configs/aetherflow_viability_candidates.json",
        help="JSON file defining candidate family policies.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names to run, or 'all'.",
    )
    parser.add_argument(
        "--folds",
        default="all",
        help="Comma-separated fold names to run, or 'all'.",
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
    parser.add_argument("--history-buffer-days", type=int, default=14)
    parser.add_argument(
        "--output-dir",
        default="backtest_reports/aetherflow_viability_suite",
        help="Directory for suite summaries and optional report exports.",
    )
    parser.add_argument("--mc-simulations", type=int, default=2000)
    parser.add_argument("--mc-seed", type=int, default=1337)
    parser.add_argument("--skip-report-export", action="store_true")
    parser.add_argument("--use-horizon-time-stop", action="store_true")
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    base_features_path = _resolve_path(str(args.base_features), DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    walkforward_report = _resolve_path(str(args.walkforward_report), "artifacts/aetherflow_walkforward_fullrange_v2_2024plus_seedfix/walkforward_report.json")
    variants_file = _resolve_path(str(args.variants_file), "configs/aetherflow_viability_candidates.json")
    output_dir = _resolve_path(str(args.output_dir), "backtest_reports/aetherflow_viability_suite")
    output_dir.mkdir(parents=True, exist_ok=True)

    walkforward_payload = _load_walkforward_report(walkforward_report)
    folds = list(walkforward_payload.get("folds", []) or [])
    if str(args.folds).strip().lower() != "all":
        wanted_folds = {item.strip() for item in str(args.folds).split(",") if item.strip()}
        folds = [fold for fold in folds if str(fold.get("fold", "")) in wanted_folds]
    if not folds:
        raise RuntimeError("No matching folds selected.")

    variants = _load_variants(variants_file)
    if str(args.variants).strip().lower() != "all":
        wanted_variants = {item.strip() for item in str(args.variants).split(",") if item.strip()}
        variants = [variant for variant in variants if variant["name"] in wanted_variants]
    if not variants:
        raise RuntimeError("No matching variants selected.")

    created_at = datetime.now(bt.NY_TZ).strftime("%Y%m%d_%H%M%S")
    suite_results: list[dict] = []
    for variant in variants:
        fold_results = []
        for fold in folds:
            fold_results.append(
                _run_fold_variant(
                    fold=fold,
                    variant=variant,
                    source_path=source_path,
                    base_features_path=base_features_path,
                    symbol_mode=str(args.symbol_mode),
                    symbol_method=str(args.symbol_method),
                    history_buffer_days=int(args.history_buffer_days),
                    output_dir=output_dir,
                    mc_simulations=int(args.mc_simulations),
                    mc_seed=int(args.mc_seed),
                    skip_report_export=bool(args.skip_report_export),
                    use_horizon_time_stop=bool(args.use_horizon_time_stop),
                )
            )
        variant_payload = {
            "name": str(variant["name"]),
            "description": str(variant.get("description", "") or ""),
            "family_policies": _json_safe(dict(variant.get("family_policies", {}) or {})),
            "selection_mode": str(variant.get("selection_mode", "highest_confidence")),
            "aggregate": _aggregate_variant_results(fold_results),
            "fold_results": fold_results,
        }
        suite_results.append(variant_payload)

    suite_payload = {
        "created_at": datetime.now(bt.NY_TZ).isoformat(),
        "walkforward_report": str(walkforward_report),
        "source": str(source_path),
        "base_features": str(base_features_path),
        "variants_file": str(variants_file),
        "folds": [str(fold.get("fold", "")) for fold in folds],
        "variants": suite_results,
    }
    suite_path = output_dir / f"aetherflow_viability_suite_{created_at}.json"
    suite_path.write_text(json.dumps(_json_safe(suite_payload), indent=2), encoding="utf-8")
    print(f"suite_report={suite_path}", flush=True)


if __name__ == "__main__":
    main()

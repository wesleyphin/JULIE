import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _resolve_run_dir(path_arg: str) -> Path:
    def is_dist_run_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "config.json").exists()
            and (path / "artifact_index.json").exists()
        )

    def collect_candidates(root: Path) -> list[Path]:
        out: list[Path] = []
        if is_dist_run_dir(root):
            out.append(root)
        try:
            level1 = [p for p in root.iterdir() if p.is_dir()]
        except Exception:
            return out
        for child in level1:
            if is_dist_run_dir(child):
                out.append(child)
            try:
                for nested in child.iterdir():
                    if nested.is_dir() and is_dist_run_dir(nested):
                        out.append(nested)
            except Exception:
                continue
        return out

    roots = [Path(path_arg).expanduser(), ROOT / Path(path_arg)]
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if is_dist_run_dir(resolved):
            return resolved
        if resolved.is_dir():
            candidates = collect_candidates(resolved)
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[0]
    raise SystemExit(f"MLPhysics dist run dir not found or invalid: {path_arg}")


def _prepare_symbol_df(source_path: Path, start_time, end_time, symbol_mode: str, symbol_method: str):
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No rows found in the source file.")

    source_df = df[df.index <= end_time]
    if source_df.empty:
        raise SystemExit("No rows found before the requested end time.")

    symbol = None
    symbol_distribution = {}
    symbol_df = source_df
    if "symbol" in source_df.columns:
        if symbol_mode != "single":
            symbol_df, auto_label, _ = bt.apply_symbol_mode(source_df, symbol_mode, symbol_method)
            if symbol_df.empty:
                raise SystemExit("No rows found after auto symbol selection.")
            selected_range_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
            if selected_range_df.empty:
                raise SystemExit("No rows found in the selected range after auto symbol selection.")
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()
            symbol = auto_label
        else:
            preferred_symbol = bt.CONFIG.get("TARGET_SYMBOL")
            symbol = bt.choose_symbol(source_df, preferred_symbol)
            symbol_df = source_df[source_df["symbol"] == symbol]
            if symbol_df.empty:
                raise SystemExit("No rows found for the selected symbol.")
            selected_range_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
            if selected_range_df.empty:
                raise SystemExit("No rows found in the selected range for the selected symbol.")
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()

        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    else:
        selected_range_df = source_df[(source_df.index >= start_time) & (source_df.index <= end_time)]
        if selected_range_df.empty:
            raise SystemExit("No rows found in the selected range.")
        symbol = "AUTO"

    source_attrs = getattr(source_df, "attrs", {}) or {}
    symbol_df = bt.attach_backtest_symbol_context(
        symbol_df,
        symbol,
        symbol_mode,
        source_key=source_attrs.get("source_cache_key"),
        source_label=source_attrs.get("source_label"),
        source_path=source_attrs.get("source_path"),
    )
    return symbol_df, symbol, symbol_distribution


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out.get(key) or {}, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


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


def _variant_definitions() -> list[dict[str, Any]]:
    common_gate_relax = {
        "ML_PHYSICS_EV_DECISION": {
            "min_trade_gate_prob": None,
            "max_trade_gate_prob": None,
        },
        "ML_PHYSICS_EV_DECISION_SESSION_OVERRIDES": {
            "ASIA": {
                "enabled": False,
                "use_model_predictions": False,
                "require_trade_gate": True,
                "require_threshold_gate": True,
                "min_trade_gate_prob": None,
                "max_trade_gate_prob": None,
            },
            "LONDON": {"min_trade_gate_prob": None, "max_trade_gate_prob": None},
            "NY_AM": {"min_trade_gate_prob": None, "max_trade_gate_prob": None},
            "NY_PM": {"min_trade_gate_prob": None, "max_trade_gate_prob": None},
        },
        "ML_PHYSICS_GATE_HARD_LIMITS": {"enabled": False},
    }
    clamp040 = {
        "ML_PHYSICS_DIST_GATE_THRESHOLD_CLAMP": {
            "enabled": True,
            "default": {"min": 0.35, "max": 0.40},
            "sessions": {
                "ASIA": {
                    "LONG": {"max": 0.40},
                    "SHORT": {"max": 0.40},
                },
                "LONDON": {
                    "LONG": {"max": 0.40},
                    "SHORT": {"max": 0.40},
                },
                "NY_AM": {
                    "LONG": {"max": 0.40},
                    "SHORT": {"max": 0.40},
                },
                "NY_PM": {
                    "LONG": {"max": 0.40},
                    "SHORT": {"max": 0.40},
                },
            },
        },
    }
    clamp035 = {
        "ML_PHYSICS_DIST_GATE_THRESHOLD_CLAMP": {
            "enabled": True,
            "default": {"min": 0.35, "max": 0.35},
            "sessions": {
                "ASIA": {
                    "LONG": {"max": 0.35},
                    "SHORT": {"max": 0.35},
                },
                "LONDON": {
                    "LONG": {"max": 0.35},
                    "SHORT": {"max": 0.35},
                },
                "NY_AM": {
                    "LONG": {"max": 0.35},
                    "SHORT": {"max": 0.35},
                },
                "NY_PM": {
                    "LONG": {"max": 0.35},
                    "SHORT": {"max": 0.35},
                },
            },
        },
    }
    gate_off = {
        "ML_PHYSICS_DIST_GATE_THRESHOLD_CLAMP": {
            "enabled": True,
            "default": {"min": 0.0, "max": 0.0},
            "sessions": {
                "ASIA": {
                    "LONG": {"min": 0.0, "max": 0.0},
                    "SHORT": {"min": 0.0, "max": 0.0},
                },
                "LONDON": {
                    "LONG": {"min": 0.0, "max": 0.0},
                    "SHORT": {"min": 0.0, "max": 0.0},
                },
                "NY_AM": {
                    "LONG": {"min": 0.0, "max": 0.0},
                    "SHORT": {"min": 0.0, "max": 0.0},
                },
                "NY_PM": {
                    "LONG": {"min": 0.0, "max": 0.0},
                    "SHORT": {"min": 0.0, "max": 0.0},
                },
            },
        },
    }
    return [
        {
            "name": "baseline_current_runtime",
            "description": "Current filterless runtime with the selected dist run.",
            "overrides": {},
        },
        {
            "name": "learned_gate_thresholds",
            "description": "Keep tradeability gate, but stop forcing runtime gate floors/caps above the learned session thresholds.",
            "overrides": common_gate_relax,
        },
        {
            "name": "learned_gate_plus_full_session_sides",
            "description": "Use learned gate thresholds and stop runtime side/session pruning for normal-profile trades.",
            "overrides": _deep_merge(
                common_gate_relax,
                {
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                },
            ),
        },
        {
            "name": "learned_gate_plus_full_session_sides_no_runtime_ev_floor",
            "description": "Use learned gate thresholds, restore full session/side coverage, and remove the runtime absolute-EV floor.",
            "overrides": _deep_merge(
                common_gate_relax,
                {
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                },
            ),
        },
        {
            "name": "artifact_gate_clamp040_no_floor_filter_full_session_sides_no_runtime_ev_floor",
            "description": "Clamp artifact gate thresholds to 0.40 max across sessions, disable runtime floor bracket filtering, restore full session-side coverage, and remove the runtime EV floor.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, clamp040),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                },
            ),
        },
        {
            "name": "artifact_gate_clamp035_no_floor_filter_full_session_sides_no_runtime_ev_floor",
            "description": "Clamp artifact gate thresholds to 0.35 max across sessions, disable runtime floor bracket filtering, restore full session-side coverage, and remove the runtime EV floor.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, clamp035),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                },
            ),
        },
        {
            "name": "artifact_gate_off_no_floor_filter_full_session_sides_no_runtime_ev_floor",
            "description": "Force artifact gate thresholds to zero, disable runtime floor bracket filtering, restore full session-side coverage, and remove the runtime EV floor.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                },
            ),
        },
        {
            "name": "artifact_gate_clamp040_ny_short_only_no_runtime_ev_floor",
            "description": "Clamp artifact gate thresholds to 0.40 max, disable runtime floor/EV filters, block wide-runner brackets, and only allow NY_AM/NY_PM shorts in the normal profile.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, clamp040),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER": {"enabled": True, "sessions": ["OFF"]},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "ASIA": {"allowed_sides": []},
                            "LONDON": {"allowed_sides": []},
                            "NY_AM": {"allowed_sides": ["SHORT"]},
                            "NY_PM": {"allowed_sides": ["SHORT"]},
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_ny_short_only_no_runtime_ev_floor",
            "description": "Force artifact gate thresholds to zero, disable runtime floor/EV filters, block wide-runner brackets, and only allow NY_AM/NY_PM shorts in the normal profile.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER": {"enabled": True, "sessions": ["OFF"]},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "ASIA": {"allowed_sides": []},
                            "LONDON": {"allowed_sides": []},
                            "NY_AM": {"allowed_sides": ["SHORT"]},
                            "NY_PM": {"allowed_sides": ["SHORT"]},
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_ny_pm_short_ev30_gate020",
            "description": "Force artifact gate thresholds to zero, block wide-runner brackets, require NY_PM short-only flow, a 0.20 gate margin floor, and 30-point runtime EV minimum.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_MIN_GATE_MARGIN": {
                        "default": 0.0,
                        "sessions": {
                            "NY_PM": 0.20,
                        },
                    },
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {
                        "enabled": True,
                        "default": 0.0,
                        "sessions": {
                            "NY_PM": 30.0,
                        },
                    },
                    "ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER": {"enabled": True, "sessions": ["OFF"]},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "ASIA": {"allowed_sides": []},
                            "LONDON": {"allowed_sides": []},
                            "NY_AM": {"allowed_sides": []},
                            "NY_PM": {"allowed_sides": ["SHORT"]},
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_no_hyst_ny_pm_short_ev30_gate020",
            "description": "Force artifact gate thresholds to zero, disable hysteresis, block wide-runner brackets, require NY_PM short-only flow, a 0.20 gate margin floor, and 30-point runtime EV minimum.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_HYSTERESIS": {
                        "enabled": False,
                    },
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_MIN_GATE_MARGIN": {
                        "default": 0.0,
                        "sessions": {
                            "NY_PM": 0.20,
                        },
                    },
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {
                        "enabled": True,
                        "default": 0.0,
                        "sessions": {
                            "NY_PM": 30.0,
                        },
                    },
                    "ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER": {"enabled": True, "sessions": ["OFF"]},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "ASIA": {"allowed_sides": []},
                            "LONDON": {"allowed_sides": []},
                            "NY_AM": {"allowed_sides": []},
                            "NY_PM": {"allowed_sides": ["SHORT"]},
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_entry_policy_ny_pm_short_ev30_gate020",
            "description": "Force artifact gate thresholds to zero and apply a late-stage entry policy that only allows NY_PM shorts with EV>=30, gate_prob>=0.20, and no runner brackets.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "min_ev_abs": 30.0,
                                "min_gate_prob": 0.20,
                                "allow_runner": False,
                            },
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_entry_policy_ny_pm_short_ev30_gate020_bt_session_only",
            "description": "Force artifact gate thresholds to zero, disable non-NY_PM sessions in backtest, and apply a late-stage NY_PM short-only entry policy with EV>=30, gate_prob>=0.20, and no runner brackets.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_BACKTEST_DISABLED_SESSIONS": ["ASIA", "LONDON", "NY_AM"],
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": None,
                        "sessions": {
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "min_ev_abs": 30.0,
                                "min_gate_prob": 0.20,
                                "allow_runner": False,
                            },
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_entry_policy_ny_pm_short_ev30_gate020_bt_session_only_tue_fri",
            "description": "Same NY_PM short-only late entry policy, but only trade Tuesdays and Fridays.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_BACKTEST_DISABLED_SESSIONS": ["ASIA", "LONDON", "NY_AM"],
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": None,
                        "sessions": {
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "allowed_weekdays": ["Tuesday", "Friday"],
                                "min_ev_abs": 30.0,
                                "min_gate_prob": 0.20,
                                "allow_runner": False,
                            },
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_entry_policy_ny_pm_short_ev30_gate020_bt_session_only_mon_tue_fri",
            "description": "Same NY_PM short-only late entry policy, but only trade Mondays, Tuesdays, and Fridays.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_BACKTEST_DISABLED_SESSIONS": ["ASIA", "LONDON", "NY_AM"],
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": None,
                        "sessions": {
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "allowed_weekdays": ["Monday", "Tuesday", "Friday"],
                                "min_ev_abs": 30.0,
                                "min_gate_prob": 0.20,
                                "allow_runner": False,
                            },
                        },
                    },
                },
            ),
        },
        {
            "name": "live_candidate_fri_ev35_gate025",
            "description": "Live-like candidate: zero artifact gate clamp, no runtime EV/floor filter, and an entry policy that only allows Friday NY_PM shorts with EV>=35, gate_prob>=0.25, and no runner brackets.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "allowed_weekdays": ["Friday"],
                                "min_ev_abs": 35.0,
                                "min_gate_prob": 0.25,
                                "allow_runner": False,
                            },
                        },
                    },
                },
            ),
        },
        {
            "name": "live_candidate_asia_thu40_gate025_plus_nypm_fri35_gate025",
            "description": "Live-like candidate: ASIA Thursday longs with EV>=40 and gate_prob>=0.25, plus Friday NY_PM shorts with EV>=35 and gate_prob>=0.25; zero artifact gate clamp and no runtime EV/floor filter.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": [],
                        "sessions": {
                            "ASIA": {
                                "allowed_sides": ["LONG"],
                                "allowed_weekdays": ["Thursday"],
                                "min_ev_abs": 40.0,
                                "min_gate_prob": 0.25,
                                "allow_runner": False,
                            },
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "allowed_weekdays": ["Friday"],
                                "min_ev_abs": 35.0,
                                "min_gate_prob": 0.25,
                                "allow_runner": False,
                            },
                        },
                    },
                },
            ),
        },
        {
            "name": "artifact_gate_off_entry_policy_ny_pm_short_ev30_gate020_bt_session_only_sl075_tp200",
            "description": "Same NY_PM short-only late entry policy, but tighten the NY_PM bracket policy to SL 0.75 / TP 2.0.",
            "overrides": _deep_merge(
                _deep_merge(common_gate_relax, gate_off),
                {
                    "ML_PHYSICS_BACKTEST_DISABLED_SESSIONS": ["ASIA", "LONDON", "NY_AM"],
                    "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER": {"enabled": False},
                    "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS": {"enabled": False},
                    "ML_PHYSICS_DIST_NORMAL_PROFILE": {"enabled": False},
                    "ML_PHYSICS_DIST_ENTRY_POLICY": {
                        "enabled": True,
                        "default_allowed_sides": None,
                        "sessions": {
                            "NY_PM": {
                                "allowed_sides": ["SHORT"],
                                "min_ev_abs": 30.0,
                                "min_gate_prob": 0.20,
                                "allow_runner": False,
                            },
                        },
                    },
                    "ML_PHYSICS_DIST_NORMAL_BRACKET_POLICY": {
                        "enabled": True,
                        "default": {"sl": 1.25, "tp": 1.5},
                        "sessions": {
                            "NY_PM": {
                                "default": {"sl": 0.75, "tp": 2.0},
                                "regimes": {
                                    "high": {"sl": 0.75, "tp": 2.0},
                                    "normal": {"sl": 0.75, "tp": 2.0},
                                    "low": {"sl": 0.75, "tp": 2.0},
                                },
                            },
                        },
                    },
                },
            ),
        },
    ]


def _run_variant(
    *,
    name: str,
    description: str,
    overrides: dict,
    base_cfg: dict,
    symbol_df,
    symbol: str,
    symbol_distribution: dict,
    start_time,
    end_time,
    dist_run_dir: Path,
    output_dir: Path,
    mc_simulations: int,
    mc_seed: int,
    skip_report_export: bool,
) -> dict[str, Any]:
    print(f"[suite] starting variant={name}", flush=True)
    cfg = copy.deepcopy(base_cfg)
    cfg = _deep_merge(
        cfg,
        {
            "ML_PHYSICS_DIST_RUN_DIR": str(dist_run_dir),
            "BACKTEST_SPEED_PROFILE": {"enabled": False},
            "BACKTEST_ML_DIAGNOSTICS": {
                "enabled": True,
                "include_no_signal": True,
                "max_records": 0,
            },
            "BACKTEST_ML_ONLY_DIAGNOSTIC_PROFILE": {"enabled": False},
            "ML_PHYSICS_OPT": {
                "enabled": True,
                "mode": "backtest",
                "prediction_cache": True,
                "overwrite_cache": False,
                "dist_precomputed_file": "",
                "dist_precomputed_strict": False,
            },
        },
    )
    cfg = _deep_merge(cfg, overrides or {})
    bt.CONFIG.clear()
    bt.CONFIG.update(cfg)

    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    stats = bt.run_backtest(
        symbol_df,
        start_time,
        end_time,
        enabled_strategies={"MLPhysicsStrategy"},
        enabled_filters=set(),
    )
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    report_path: Path | None = None
    if not bool(skip_report_export):
        report_path = bt.save_backtest_report(
            stats,
            symbol,
            start_time,
            end_time,
            output_dir=variant_dir,
        )
    trade_log = stats.get("trade_log", []) or []
    risk = bt._compute_backtest_risk_metrics(trade_log)
    mc = bt._build_monte_carlo_summary(
        trade_log,
        stats,
        simulations=max(1, int(mc_simulations)),
        seed=int(mc_seed),
        starting_balance=float(bt.BACKTEST_MONTE_CARLO_START_BALANCE),
    )
    ext_mc = _bootstrap_metric_summary(
        trade_log,
        simulations=max(1, int(mc_simulations)),
        seed=int(mc_seed),
    )
    ml_diag_summary = copy.deepcopy(stats.get("ml_diagnostics_summary", {}) or {})
    session_counts = _trade_counts(trade_log, "session")
    side_counts = _trade_counts(trade_log, "side")
    summary = {
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
    result = {
        "name": name,
        "description": description,
        "report_path": str(report_path) if report_path is not None else "",
        "summary": summary,
        "risk_metrics": risk,
        "trade_sessions": session_counts,
        "trade_sides": side_counts,
        "ml_diagnostics_summary": ml_diag_summary,
        "monte_carlo_trade_order": mc,
        "monte_carlo_trade_day_bootstrap": ext_mc,
    }
    # Keep the filename short so summary-only runs do not trip Windows path limits.
    lightweight_path = variant_dir / "summary.json"
    lightweight_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(lightweight_path)
    print(
        f"[suite] completed variant={name} trades={summary['trades']} equity={summary['equity']:.2f} "
        f"pf={summary['profit_factor']:.3f} sharpe={summary['daily_sharpe']:.3f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a focused MLPhysics viability ablation suite in true filterless backtest mode."
    )
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME, help="Parquet/CSV source path.")
    parser.add_argument("--start", required=True, help="Start datetime: YYYY-MM-DD or YYYY-MM-DD HH:MM")
    parser.add_argument("--end", required=True, help="End datetime: YYYY-MM-DD or YYYY-MM-DD HH:MM")
    parser.add_argument("--dist-run-dir", required=True, help="MLPhysics dist run directory to evaluate.")
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
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_reports/mlphysics_viability_suite",
        help="Directory for reports and summaries.",
    )
    parser.add_argument("--mc-simulations", type=int, default=2000, help="Monte Carlo simulations per variant.")
    parser.add_argument("--mc-seed", type=int, default=1337, help="Monte Carlo RNG seed.")
    parser.add_argument(
        "--skip-report-export",
        action="store_true",
        help="Skip heavyweight report export and only write lightweight summary JSON files.",
    )
    args = parser.parse_args()

    dist_run_dir = _resolve_run_dir(str(args.dist_run_dir))
    source_path = _resolve_source(str(args.source))
    start_time = bt.parse_user_datetime(str(args.start), bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(str(args.end), bt.NY_TZ, is_end=True)
    if start_time > end_time:
        raise SystemExit("Start must be before end.")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = copy.deepcopy(bt.CONFIG)
    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(args.symbol_mode or "single").strip().lower(),
        str(args.symbol_method or "volume").strip().lower(),
    )

    variant_defs = _variant_definitions()
    if str(args.variants).strip().lower() != "all":
        allowed = {item.strip() for item in str(args.variants).split(",") if item.strip()}
        variant_defs = [item for item in variant_defs if item["name"] in allowed]
    if not variant_defs:
        raise SystemExit("No matching variants selected.")

    timestamp = datetime.now(bt.NY_TZ).strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []
    try:
        for variant in variant_defs:
            result = _run_variant(
                name=str(variant["name"]),
                description=str(variant["description"]),
                overrides=dict(variant.get("overrides") or {}),
                base_cfg=base_cfg,
                symbol_df=symbol_df,
                symbol=symbol,
                symbol_distribution=symbol_distribution,
                start_time=start_time,
                end_time=end_time,
                dist_run_dir=dist_run_dir,
                output_dir=output_dir,
                mc_simulations=int(args.mc_simulations),
                mc_seed=int(args.mc_seed),
                skip_report_export=bool(args.skip_report_export),
            )
            results.append(result)
            bt.CONFIG.clear()
            bt.CONFIG.update(copy.deepcopy(base_cfg))
    finally:
        bt.CONFIG.clear()
        bt.CONFIG.update(base_cfg)

    suite_payload = {
        "created_at": datetime.now(bt.NY_TZ).isoformat(),
        "source": str(source_path),
        "dist_run_dir": str(dist_run_dir),
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "symbol": symbol,
        "symbol_distribution": symbol_distribution,
        "variants": results,
    }
    suite_path = output_dir / f"mlphysics_viability_suite_{timestamp}.json"
    suite_path.write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")
    print(f"suite_report={suite_path}", flush=True)


if __name__ == "__main__":
    main()

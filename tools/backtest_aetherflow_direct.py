import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import (
    DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    resolve_full_manifold_base_features_path,
    validate_full_manifold_base_features_path,
)
import backtest_mes_et as bt
from aetherflow_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, build_feature_frame
from aetherflow_model_bundle import (
    bundle_feature_columns,
    normalize_model_bundle,
    predict_bundle_probabilities,
)
from aetherflow_strategy import AetherFlowStrategy, REGIME_ID_TO_NAME, augment_aetherflow_phase_features
from bot_state import trading_day_start
from config import CONFIG
from tools.backtest_regimeadaptive_robust import _write_converted_csv


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _prepare_symbol_df(source_path: Path, start_time, end_time, symbol_mode: str, symbol_method: str, history_buffer_days: int):
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
    buffer_start = pd.Timestamp(start_time) - pd.Timedelta(days=max(3, int(history_buffer_days)))
    symbol_df = symbol_df.loc[(symbol_df.index >= buffer_start) & (symbol_df.index <= end_time)]
    return symbol_df, symbol, symbol_distribution


def _load_model_bundle(model_path: Path, thresholds_path: Path) -> tuple[object, list[str], float, dict]:
    with model_path.open("rb") as fh:
        bundle = normalize_model_bundle(pickle.load(fh))
    feature_columns = bundle_feature_columns(bundle)
    threshold = float(bundle.get("threshold", 0.58) or 0.58)
    policy = {
        "allowed_setup_families": [],
        "hazard_block_regimes": [],
    }
    if thresholds_path.exists():
        try:
            payload = json.loads(thresholds_path.read_text())
            threshold = float(payload.get("threshold", threshold) or threshold)
            feat_cols = payload.get("feature_columns")
            if isinstance(feat_cols, list) and feat_cols:
                requested = {str(col) for col in feat_cols if str(col).strip()}
                feature_columns = [col for col in bundle_feature_columns(bundle) if col in requested] or feature_columns
            policy["allowed_setup_families"] = [
                str(item).strip()
                for item in (payload.get("allowed_setup_families", []) or [])
                if str(item).strip()
            ]
            policy["hazard_block_regimes"] = [
                str(item).strip().upper()
                for item in (payload.get("hazard_block_regimes", []) or [])
                if str(item).strip()
            ]
        except Exception:
            pass
    return bundle, feature_columns, threshold, policy


def _load_base_features(path: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    validate_full_manifold_base_features_path(path, BASE_FEATURE_COLUMNS)
    data = pd.read_parquet(path, columns=sorted(set(BASE_FEATURE_COLUMNS)))
    data = data.loc[(data.index >= start_time) & (data.index <= end_time)]
    if data.empty:
        raise RuntimeError("No cached manifold base rows in the requested range.")
    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _session_name(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


def _build_signal_frame(
    *,
    symbol_df: pd.DataFrame,
    base_features_path: Path,
    model_path: Path,
    thresholds_path: Path,
    threshold_override: float | None,
    min_confidence_override: float | None,
    allow_setups: set[str] | None,
    block_regimes: set[str],
    allowed_session_ids: set[int] | None,
    use_runtime_policy: bool = False,
) -> pd.DataFrame:
    base = _load_base_features(base_features_path, pd.Timestamp(symbol_df.index.min()), pd.Timestamp(symbol_df.index.max()))

    if use_runtime_policy:
        runtime_strategy = AetherFlowStrategy()
        metrics_cfg_path = Path(str(CONFIG.get("AETHERFLOW_STRATEGY", {}).get("metrics_file", runtime_strategy.metrics_path))).expanduser()
        if not metrics_cfg_path.is_absolute():
            metrics_cfg_path = ROOT / metrics_cfg_path
        runtime_strategy.model_path = Path(model_path)
        runtime_strategy.thresholds_path = Path(thresholds_path)
        runtime_strategy.metrics_path = metrics_cfg_path
        runtime_strategy.model = None
        runtime_strategy.model_bundle = None
        runtime_strategy.model_loaded = False
        runtime_strategy._precomputed_df = pd.DataFrame()
        runtime_strategy._precomputed_lookup = {}
        runtime_strategy._precomputed_known_timestamps = set()
        runtime_strategy._load_artifacts()
        if not runtime_strategy.model_loaded or runtime_strategy.model_bundle is None:
            raise RuntimeError(f"AetherFlow model failed to load: {model_path}")
        if threshold_override is not None:
            runtime_strategy.threshold_override = float(threshold_override)
            runtime_strategy.threshold = float(threshold_override)
        if min_confidence_override is not None:
            runtime_strategy.min_confidence = float(min_confidence_override)
        if allow_setups:
            runtime_strategy.allowed_setup_families = set(str(item) for item in allow_setups if str(item).strip())
        if block_regimes:
            runtime_strategy.hazard_block_regimes = {str(item).strip().upper() for item in block_regimes if str(item).strip()}
        if allowed_session_ids and not runtime_strategy.family_policies:
            runtime_strategy.allowed_session_ids = {int(x) for x in allowed_session_ids}
        return runtime_strategy.build_backtest_df_from_base_features(base)

    bundle, feature_columns, threshold, policy = _load_model_bundle(model_path, thresholds_path)
    effective_threshold = float(threshold_override if threshold_override is not None else threshold)
    if min_confidence_override is not None:
        effective_threshold = max(effective_threshold, float(min_confidence_override))
    if not allow_setups:
        allow_setups = {
            str(item).strip()
            for item in (policy.get("allowed_setup_families", []) or [])
            if str(item).strip()
        } or None
    if not block_regimes:
        block_regimes = {
            str(item).strip().upper()
            for item in (policy.get("hazard_block_regimes", []) or [])
            if str(item).strip()
        }

    features = build_feature_frame(
        base_features=base,
        preferred_setup_families=allow_setups,
    )
    if features.empty:
        return pd.DataFrame()
    features = augment_aetherflow_phase_features(features)

    if allowed_session_ids:
        sess = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
        features = features.loc[sess.isin(sorted(allowed_session_ids))]
    features = features.loc[pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0]
    if allow_setups:
        features = features.loc[features["setup_family"].astype(str).isin(sorted(allow_setups))]
    if features.empty:
        return pd.DataFrame()

    features = features.copy()
    features["aetherflow_confidence"] = predict_bundle_probabilities(bundle, features)
    features["manifold_regime_name"] = features["manifold_regime_id"].round().astype(int).map(REGIME_ID_TO_NAME).fillna("")
    if block_regimes:
        features = features.loc[~features["manifold_regime_name"].astype(str).str.upper().isin(sorted(block_regimes))]
    features = features.loc[features["aetherflow_confidence"] >= float(effective_threshold)]
    if features.empty:
        return pd.DataFrame()
    return features


def _normalize_aetherflow_risk_governor(raw_cfg: dict | None) -> dict:
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}

    def _float_or(default: float, value) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return float(out) if math.isfinite(out) else float(default)

    def _int_or(default: int, value) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            return int(default)

    return {
        "enabled": bool(cfg.get("enabled", False)),
        "daily_loss_stop_usd": max(0.0, _float_or(0.0, cfg.get("daily_loss_stop_usd", 0.0))),
        "min_loss_trades": max(0, _int_or(0, cfg.get("min_loss_trades", 0))),
        "block_new_entries_rest_of_day": bool(cfg.get("block_new_entries_rest_of_day", True)),
    }


def _resolve_aetherflow_signal_size(signal: dict | None) -> tuple[int, float]:
    base_size = max(1, int(CONFIG.get("AETHERFLOW_STRATEGY", {}).get("size", 5) or 5))
    if not isinstance(signal, dict):
        return int(base_size), 1.0
    try:
        raw_multiplier = float(
            signal.get(
                "aetherflow_size_multiplier",
                signal.get("size_multiplier", 1.0),
            )
            or 1.0
        )
    except Exception:
        raw_multiplier = 1.0
    if not math.isfinite(raw_multiplier) or raw_multiplier <= 0.0:
        raw_multiplier = 1.0
    target_size = int(round(float(base_size) * float(raw_multiplier)))
    return max(1, target_size), float(raw_multiplier)


def _aetherflow_trade_day_key(ts: pd.Timestamp) -> pd.Timestamp:
    dt_value = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts).to_pydatetime()
    return pd.Timestamp(trading_day_start(dt_value))


def _aetherflow_backtest_execution_cfg() -> dict:
    cfg = dict(CONFIG.get("BACKTEST_EXECUTION", {}) or {})
    af_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    override = af_cfg.get("direct_backtest_execution", {})
    if isinstance(override, dict):
        cfg.update(override)
    return cfg


def _normalize_aetherflow_drawdown_size_scaling(raw_cfg: dict | None = None) -> dict:
    af_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    if isinstance(raw_cfg, dict):
        cfg = dict(raw_cfg)
        source = "override"
    elif isinstance(af_cfg.get("drawdown_size_scaling"), dict):
        cfg = dict(af_cfg.get("drawdown_size_scaling") or {})
        source = "aetherflow_strategy"
    else:
        exec_cfg = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
        cfg = {
            "enabled": exec_cfg.get("drawdown_size_scaling_enabled", False),
            "start_usd": exec_cfg.get("drawdown_size_scaling_start_usd", 0.0),
            "max_usd": exec_cfg.get("drawdown_size_scaling_max_usd", 0.0),
            "base_contracts": exec_cfg.get("drawdown_size_scaling_base_contracts", af_cfg.get("size", 5)),
            "min_contracts": exec_cfg.get("drawdown_size_scaling_min_contracts", 1),
        }
        source = "backtest_execution"

    def _float_or(default: float, value) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return float(out) if math.isfinite(out) else float(default)

    def _int_or(default: int, value) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            return int(default)

    start_usd = max(0.0, _float_or(0.0, cfg.get("start_usd", cfg.get("drawdown_size_scaling_start_usd", 0.0))))
    max_usd = max(
        start_usd,
        _float_or(start_usd, cfg.get("max_usd", cfg.get("drawdown_size_scaling_max_usd", start_usd))),
    )
    base_contracts = max(
        1,
        _int_or(
            int(af_cfg.get("size", 5) or 5),
            cfg.get("base_contracts", cfg.get("drawdown_size_scaling_base_contracts", af_cfg.get("size", 5))),
        ),
    )
    min_contracts = max(
        1,
        _int_or(1, cfg.get("min_contracts", cfg.get("drawdown_size_scaling_min_contracts", 1))),
    )
    if min_contracts > base_contracts:
        min_contracts = int(base_contracts)
    return {
        "enabled": bool(cfg.get("enabled", cfg.get("drawdown_size_scaling_enabled", False))),
        "start_usd": float(start_usd),
        "max_usd": float(max_usd),
        "base_contracts": int(base_contracts),
        "min_contracts": int(min_contracts),
        "source": source,
    }


def _apply_aetherflow_drawdown_size_cap(
    requested_size: int,
    current_realized_dd: float,
    scaling_cfg: dict | None,
) -> tuple[int, dict]:
    cfg = scaling_cfg if isinstance(scaling_cfg, dict) else _normalize_aetherflow_drawdown_size_scaling()
    requested = max(1, int(requested_size))
    start_usd = max(0.0, float(cfg.get("start_usd", 0.0) or 0.0))
    max_usd = max(start_usd, float(cfg.get("max_usd", start_usd) or start_usd))
    configured_base_contracts = max(1, int(cfg.get("base_contracts", requested) or requested))
    base_contracts = max(1, int(max(requested, configured_base_contracts)))
    min_contracts = max(1, int(cfg.get("min_contracts", 1) or 1))
    if min_contracts > base_contracts:
        min_contracts = int(base_contracts)
    realized_dd = max(0.0, float(current_realized_dd or 0.0))

    diagnostics = {
        "drawdown_size_scaling_enabled": bool(cfg.get("enabled", False)),
        "drawdown_size_source": str(cfg.get("source", "") or ""),
        "drawdown_size_realized_dd_usd": float(realized_dd),
        "drawdown_size_progress": 0.0,
        "drawdown_size_cap": int(base_contracts),
        "drawdown_size_requested": int(requested),
        "drawdown_size_applied": False,
        "drawdown_size_step_usd": 0.0,
        "drawdown_size_base_contracts": int(base_contracts),
        "drawdown_size_min_contracts": int(min_contracts),
    }
    if not bool(cfg.get("enabled", False)) or max_usd <= start_usd or base_contracts <= min_contracts:
        diagnostics["drawdown_size_scaling_enabled"] = False
        return int(requested), diagnostics

    contract_range = max(1, int(base_contracts) - int(min_contracts))
    span_usd = max(1e-9, float(max_usd) - float(start_usd))
    step_usd = span_usd / float(contract_range)
    if realized_dd <= start_usd:
        dd_progress = 0.0
        drawdown_size_cap = int(base_contracts)
    elif realized_dd >= max_usd:
        dd_progress = 1.0
        drawdown_size_cap = int(min_contracts)
    else:
        dd_above = realized_dd - start_usd
        dd_progress = min(1.0, dd_above / span_usd)
        bucket = int(dd_above / step_usd)
        bucket = max(0, min(int(contract_range), int(bucket)))
        drawdown_size_cap = max(int(min_contracts), int(base_contracts) - int(bucket))

    applied_size = min(int(requested), int(drawdown_size_cap), int(base_contracts))
    applied_size = max(int(min_contracts), int(applied_size))
    diagnostics.update(
        {
            "drawdown_size_progress": float(dd_progress),
            "drawdown_size_cap": int(drawdown_size_cap),
            "drawdown_size_applied": bool(applied_size < requested),
            "drawdown_size_step_usd": float(step_usd),
        }
    )
    return int(applied_size), diagnostics


def _simulate_single_position(
    *,
    df: pd.DataFrame,
    signals: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    use_horizon_time_stop: bool,
    risk_governor_override: dict | None = None,
) -> dict:
    exec_cfg = _aetherflow_backtest_execution_cfg()
    gap_fills = bool(exec_cfg.get("gap_fills", True))
    no_entry_window = bool(exec_cfg.get("enforce_no_new_entries_window", True))
    no_entry_start = int(exec_cfg.get("no_new_entries_start_hour_et", 16) or 16)
    no_entry_start_minute = int(exec_cfg.get("no_new_entries_start_minute_et", 0) or 0)
    no_entry_end = int(exec_cfg.get("no_new_entries_end_hour_et", 18) or 18)
    no_entry_end_minute = int(exec_cfg.get("no_new_entries_end_minute_et", 0) or 0)
    force_flat_enabled = bool(exec_cfg.get("force_flat_at_time", True))
    force_flat_hour = int(exec_cfg.get("force_flat_hour_et", 16) or 16)
    force_flat_minute = int(exec_cfg.get("force_flat_minute_et", 0) or 0)

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
    fee_per_contract_rt = fees_per_side * 2.0
    tick_size = float(getattr(bt, "TICK_SIZE", 0.25) or 0.25)
    aetherflow_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    drawdown_size_cfg = _normalize_aetherflow_drawdown_size_scaling()
    risk_governor_cfg = _normalize_aetherflow_risk_governor(
        risk_governor_override if isinstance(risk_governor_override, dict) and risk_governor_override else aetherflow_cfg.get("risk_governor", {})
    )

    index = pd.DatetimeIndex(df.index)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    hours = index.hour.to_numpy(dtype=np.int16)
    minutes = index.minute.to_numpy(dtype=np.int16)
    in_range = np.asarray((index >= start_time) & (index <= end_time), dtype=bool)
    positions = np.flatnonzero(in_range)
    if positions.size == 0:
        raise RuntimeError("No bars in requested range.")

    holiday_dates = bt._build_closed_holiday_dates_et(index)
    date_arr = np.array(index.date, dtype=object)
    holiday_closed_arr = np.isin(date_arr, np.array(sorted(holiday_dates), dtype=object)) if holiday_dates else np.zeros(len(df), dtype=bool)
    holiday_session_block = np.asarray([_session_name(ts) in {"ASIA", "LONDON", "NY_AM", "NY_PM"} for ts in index], dtype=bool)
    holiday_flat_arr = holiday_closed_arr & holiday_session_block

    signal_lookup = {int(ts.value): row for ts, row in zip(pd.DatetimeIndex(signals.index), signals.to_dict("records"))}

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    wins = 0
    losses = 0
    trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_log: list[dict] = []
    exit_reason_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    break_even_armed_trade_count = 0
    break_even_stop_update_count = 0
    early_exit_close_count = 0
    risk_governor_blocked_days: set[str] = set()
    risk_governor_blocked_signals = 0
    risk_governor_state = {
        "trade_day": None,
        "realized_pnl": 0.0,
        "loss_trades": 0,
        "blocked": False,
        "blocked_reason": "",
        "blocked_at": None,
    }

    active = False
    pending_entry = None
    active_trade: dict = {}

    no_entry_start_total = (int(no_entry_start) * 60) + int(no_entry_start_minute)
    no_entry_end_total = (int(no_entry_end) * 60) + int(no_entry_end_minute)

    def _coerce_float(value, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    def _coerce_int(value, default: int = 0) -> int:
        return int(round(_coerce_float(value, float(default))))

    def _coerce_bool(value, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        try:
            if pd.isna(value):
                return bool(default)
        except Exception:
            pass
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    def _ensure_risk_governor_day(ts: pd.Timestamp) -> None:
        trade_day = _aetherflow_trade_day_key(pd.Timestamp(ts))
        if risk_governor_state["trade_day"] == trade_day:
            return
        risk_governor_state["trade_day"] = trade_day
        risk_governor_state["realized_pnl"] = 0.0
        risk_governor_state["loss_trades"] = 0
        risk_governor_state["blocked"] = False
        risk_governor_state["blocked_reason"] = ""
        risk_governor_state["blocked_at"] = None

    def _risk_governor_blocks_new_entries(ts: pd.Timestamp) -> bool:
        _ensure_risk_governor_day(ts)
        return bool(
            risk_governor_cfg.get("enabled", False)
            and risk_governor_cfg.get("block_new_entries_rest_of_day", True)
            and risk_governor_state.get("blocked", False)
        )

    def _minute_window_blocked(current_hour: int, current_minute: int) -> bool:
        if not no_entry_window:
            return False
        current_total = (int(current_hour) * 60) + int(current_minute)
        if no_entry_start_total == no_entry_end_total:
            return False
        if no_entry_start_total < no_entry_end_total:
            return bool(no_entry_start_total <= current_total < no_entry_end_total)
        return bool(current_total >= no_entry_start_total or current_total < no_entry_end_total)

    def close_trade(exit_price: float, exit_time: pd.Timestamp, reason: str, exit_bar_index: int) -> None:
        nonlocal active, pending_entry, active_trade, equity, peak, max_drawdown, wins, losses, trades, gross_profit, gross_loss, early_exit_close_count
        pnl_points = (
            float(exit_price) - float(active_trade["entry_price"])
            if int(active_trade["side_num"]) > 0
            else float(active_trade["entry_price"]) - float(exit_price)
        )
        fee_paid = float(fee_per_contract_rt) * int(active_trade["size"])
        pnl_dollars = pnl_points * float(point_value) * int(active_trade["size"])
        pnl_net = pnl_dollars - fee_paid
        equity += pnl_net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        gross_profit += max(pnl_net, 0.0)
        gross_loss += min(pnl_net, 0.0)
        wins += int(pnl_net > 0.0)
        losses += int(pnl_net <= 0.0)
        trades += 1
        if str(reason) == "early_exit":
            early_exit_close_count += 1
        _ensure_risk_governor_day(exit_time)
        risk_governor_state["realized_pnl"] = float(risk_governor_state.get("realized_pnl", 0.0) or 0.0) + float(pnl_net)
        if pnl_net < 0.0:
            risk_governor_state["loss_trades"] = int(risk_governor_state.get("loss_trades", 0) or 0) + 1
        if (
            bool(risk_governor_cfg.get("enabled", False))
            and not bool(risk_governor_state.get("blocked", False))
            and float(risk_governor_cfg.get("daily_loss_stop_usd", 0.0) or 0.0) > 0.0
            and float(risk_governor_state.get("realized_pnl", 0.0) or 0.0) <= -float(risk_governor_cfg.get("daily_loss_stop_usd", 0.0) or 0.0)
            and int(risk_governor_state.get("loss_trades", 0) or 0) >= int(risk_governor_cfg.get("min_loss_trades", 0) or 0)
        ):
            risk_governor_state["blocked"] = True
            risk_governor_state["blocked_reason"] = f"daily_loss_stop_{int(round(float(risk_governor_cfg.get('daily_loss_stop_usd', 0.0) or 0.0)))}"
            risk_governor_state["blocked_at"] = pd.Timestamp(exit_time)
            risk_governor_blocked_days.add(str(pd.Timestamp(risk_governor_state["trade_day"]).date()))
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        session_counts[active_trade["session"]] = session_counts.get(active_trade["session"], 0) + 1
        trade_log.append(
            {
                "trade_id": int(trades),
                "strategy": "AetherFlowStrategy",
                "sub_strategy": None,
                "side": "LONG" if int(active_trade["side_num"]) > 0 else "SHORT",
                "entry_time": pd.Timestamp(active_trade["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(exit_time).isoformat(),
                "entry_price": float(active_trade["entry_price"]),
                "exit_price": float(exit_price),
                "size": int(active_trade["size"]),
                "pnl_points": float(round(pnl_points, 6)),
                "pnl_dollars": float(round(pnl_dollars, 2)),
                "pnl_net": float(round(pnl_net, 2)),
                "fee_paid": float(round(fee_paid, 2)),
                "sl_dist": float(active_trade["sl_dist"]),
                "tp_dist": float(active_trade["tp_dist"]),
                "mfe_points": float(round(active_trade.get("mfe_points", 0.0), 6)),
                "mae_points": float(round(active_trade.get("mae_points", 0.0), 6)),
                "entry_mode": "aetherflow_direct",
                "vol_regime": "BYPASS",
                "exit_reason": str(reason),
                "bars_held": int(active_trade.get("bars_held", 0)),
                "session": str(active_trade["session"]),
                "aetherflow_setup_family": str(active_trade.get("setup_family", "") or ""),
                "aetherflow_confidence": float(active_trade.get("confidence", 0.0) or 0.0),
                "aetherflow_regime": str(active_trade.get("regime_name", "") or ""),
                "aetherflow_horizon_bars": int(active_trade.get("horizon_bars", 0) or 0),
                "aetherflow_use_horizon_time_stop": bool(active_trade.get("use_horizon_time_stop", False)),
                "aetherflow_entry_mode": str(active_trade.get("entry_mode", "") or ""),
                "aetherflow_entry_intrabar_fill": bool(active_trade.get("entry_intrabar_fill", False)),
                "aetherflow_size_multiplier": float(active_trade.get("size_multiplier", 1.0) or 1.0),
                "drawdown_size_scaling_enabled": bool(active_trade.get("drawdown_size_scaling_enabled", False)),
                "drawdown_size_source": str(active_trade.get("drawdown_size_source", "") or ""),
                "drawdown_size_realized_dd_usd": float(active_trade.get("drawdown_size_realized_dd_usd", 0.0) or 0.0),
                "drawdown_size_progress": float(active_trade.get("drawdown_size_progress", 0.0) or 0.0),
                "drawdown_size_cap": int(active_trade.get("drawdown_size_cap", active_trade.get("size", 0)) or 0),
                "drawdown_size_requested": int(active_trade.get("drawdown_size_requested", active_trade.get("requested_size", active_trade.get("size", 0))) or 0),
                "drawdown_size_applied": bool(active_trade.get("drawdown_size_applied", False)),
                "drawdown_size_step_usd": float(active_trade.get("drawdown_size_step_usd", 0.0) or 0.0),
                "aetherflow_break_even_enabled": bool(active_trade.get("break_even_enabled", False)),
                "aetherflow_break_even_applied": bool(active_trade.get("break_even_applied", False)),
                "aetherflow_break_even_move_count": int(active_trade.get("break_even_move_count", 0) or 0),
                "aetherflow_break_even_last_stop_price": (
                    None
                    if active_trade.get("break_even_last_stop_price") is None
                    else float(active_trade.get("break_even_last_stop_price"))
                ),
                "aetherflow_early_exit_enabled": bool(active_trade.get("early_exit_enabled", False)),
                "aetherflow_profit_crosses": int(active_trade.get("profit_crosses", 0) or 0),
            }
        )
        active = False
        active_trade = {}
        pending_entry = None

    def update_excursions(bar_high: float, bar_low: float) -> None:
        if not active:
            return
        entry_price = float(active_trade["entry_price"])
        if int(active_trade["side_num"]) > 0:
            favorable = max(0.0, float(bar_high) - entry_price)
            adverse = max(0.0, entry_price - float(bar_low))
        else:
            favorable = max(0.0, entry_price - float(bar_low))
            adverse = max(0.0, float(bar_high) - entry_price)
        active_trade["mfe_points"] = max(float(active_trade.get("mfe_points", 0.0)), favorable)
        active_trade["mae_points"] = max(float(active_trade.get("mae_points", 0.0)), adverse)

    def align_stop_price_to_tick(price: float, side_num: int) -> float:
        if not math.isfinite(price) or tick_size <= 0.0:
            return float(price)
        scaled = float(price) / float(tick_size)
        if int(side_num) > 0:
            return round(math.floor(scaled + 1e-9) * float(tick_size), 10)
        return round(math.ceil(scaled - 1e-9) * float(tick_size), 10)

    def apply_break_even_stop_update(trade: dict, new_stop_price: float, *, from_pending: bool) -> bool:
        nonlocal break_even_armed_trade_count, break_even_stop_update_count
        if not isinstance(trade, dict):
            return False
        side_num = int(trade.get("side_num", 0) or 0)
        if side_num == 0:
            return False
        current_stop_price = float(trade.get("current_stop_price", math.nan))
        if not math.isfinite(current_stop_price):
            return False
        target_stop_price = align_stop_price_to_tick(float(new_stop_price), side_num)
        entry_price = float(trade.get("entry_price", math.nan))
        tp_dist = float(trade.get("tp_dist", math.nan))
        if math.isfinite(entry_price) and math.isfinite(tp_dist) and tp_dist > 0.0 and tick_size > 0.0:
            take_price = entry_price + tp_dist if side_num > 0 else entry_price - tp_dist
            if side_num > 0:
                max_stop_price = align_stop_price_to_tick(take_price - tick_size, side_num)
                if math.isfinite(max_stop_price):
                    target_stop_price = min(target_stop_price, max_stop_price)
            else:
                min_stop_price = align_stop_price_to_tick(take_price + tick_size, side_num)
                if math.isfinite(min_stop_price):
                    target_stop_price = max(target_stop_price, min_stop_price)
        improved = (
            target_stop_price > current_stop_price + 1e-12
            if side_num > 0
            else target_stop_price < current_stop_price - 1e-12
        )
        if not improved:
            if from_pending:
                trade["break_even_pending_stop_price"] = None
            return False
        already_applied = bool(trade.get("break_even_applied", False))
        trade["current_stop_price"] = float(target_stop_price)
        trade["break_even_last_stop_price"] = float(target_stop_price)
        trade["break_even_armed"] = True
        trade["break_even_applied"] = True
        trade["break_even_move_count"] = int(trade.get("break_even_move_count", 0) or 0) + 1
        if from_pending:
            trade["break_even_pending_stop_price"] = None
        if not already_applied:
            break_even_armed_trade_count += 1
        break_even_stop_update_count += 1
        return True

    def apply_pending_break_even_stop_update(trade: dict) -> None:
        if not isinstance(trade, dict):
            return
        pending_stop_price = trade.get("break_even_pending_stop_price")
        if pending_stop_price is None:
            return
        try:
            pending_stop_price = float(pending_stop_price)
        except Exception:
            trade["break_even_pending_stop_price"] = None
            return
        if not math.isfinite(pending_stop_price):
            trade["break_even_pending_stop_price"] = None
            return
        apply_break_even_stop_update(trade, pending_stop_price, from_pending=True)

    def stage_break_even_stop_update(trade: dict) -> None:
        if not isinstance(trade, dict) or not bool(trade.get("break_even_enabled", False)):
            return
        side_num = int(trade.get("side_num", 0) or 0)
        entry_price = float(trade.get("entry_price", math.nan))
        current_stop_price = float(trade.get("current_stop_price", math.nan))
        tp_dist = float(trade.get("tp_dist", math.nan))
        if side_num == 0 or not math.isfinite(entry_price) or not math.isfinite(current_stop_price) or not math.isfinite(tp_dist) or tp_dist <= 0.0:
            return
        trigger_pct = max(0.0, float(trade.get("break_even_trigger_pct", 0.0) or 0.0))
        trail_pct = max(0.0, float(trade.get("break_even_trail_pct", 0.0) or 0.0))
        buffer_ticks = max(0, int(round(float(trade.get("break_even_buffer_ticks", 0) or 0))))
        activate_on_next_bar = bool(trade.get("break_even_activate_on_next_bar", True))
        mfe_points = max(0.0, float(trade.get("mfe_points", 0.0) or 0.0))
        trigger_points = max(0.0, tp_dist * trigger_pct)
        if trigger_points > 0.0 and mfe_points + 1e-9 < trigger_points:
            return
        buffer_points = float(buffer_ticks) * float(tick_size)
        locked_points = max(buffer_points, mfe_points * trail_pct if trail_pct > 0.0 else 0.0)
        candidate_stop_price = entry_price + locked_points if side_num > 0 else entry_price - locked_points
        candidate_stop_price = align_stop_price_to_tick(candidate_stop_price, side_num)
        improved = (
            candidate_stop_price > current_stop_price + 1e-12
            if side_num > 0
            else candidate_stop_price < current_stop_price - 1e-12
        )
        if not improved:
            return
        trade["break_even_armed"] = True
        trade["break_even_locked_points"] = float(locked_points)
        if activate_on_next_bar:
            pending_stop_price = trade.get("break_even_pending_stop_price")
            if pending_stop_price is not None:
                try:
                    pending_stop_price = float(pending_stop_price)
                except Exception:
                    pending_stop_price = None
            better_pending = (
                pending_stop_price is None
                or (
                    candidate_stop_price > pending_stop_price + 1e-12
                    if side_num > 0
                    else candidate_stop_price < pending_stop_price - 1e-12
                )
            )
            if better_pending:
                trade["break_even_pending_stop_price"] = float(candidate_stop_price)
            return
        apply_break_even_stop_update(trade, candidate_stop_price, from_pending=False)

    def check_early_exit(current_price: float) -> bool:
        if not active or not bool(active_trade.get("early_exit_enabled", False)):
            return False
        side_num = int(active_trade.get("side_num", 0) or 0)
        entry_price = float(active_trade.get("entry_price", math.nan))
        if side_num == 0 or not math.isfinite(entry_price):
            return False
        is_green = current_price > entry_price if side_num > 0 else current_price < entry_price
        was_green = active_trade.get("was_green")
        if was_green is not None and bool(was_green) != bool(is_green):
            active_trade["profit_crosses"] = int(active_trade.get("profit_crosses", 0) or 0) + 1
        active_trade["was_green"] = bool(is_green)
        exit_if_not_green_by = max(0, int(active_trade.get("early_exit_exit_if_not_green_by", 0) or 0))
        max_profit_crosses = max(0, int(active_trade.get("early_exit_max_profit_crosses", 0) or 0))
        if exit_if_not_green_by > 0 and int(active_trade.get("bars_held", 0) or 0) >= exit_if_not_green_by and not is_green:
            return True
        if int(active_trade.get("profit_crosses", 0) or 0) > max_profit_crosses:
            return True
        return False

    def activate_trade_from_signal(signal: dict, *, entry_time: pd.Timestamp, entry_bar_index: int, entry_price: float, intrabar_fill: bool) -> None:
        nonlocal active, pending_entry, active_trade
        side_num = _coerce_int(signal.get("candidate_side", 0), 0)
        if side_num == 0:
            raw_side = str(signal.get("side", "") or "").strip().upper()
            if raw_side == "LONG":
                side_num = 1
            elif raw_side == "SHORT":
                side_num = -1
        if side_num == 0:
            pending_entry = None
            return
        atr14 = max(_coerce_float(signal.get("atr14", 1.0), 1.0), 1e-9)
        setup_sl_mult = _coerce_float(signal.get("setup_sl_mult", 1.0), 1.0)
        setup_tp_mult = _coerce_float(signal.get("setup_tp_mult", 2.0), 2.0)
        sl_mult = _coerce_float(signal.get("sl_mult_override", setup_sl_mult), setup_sl_mult)
        tp_mult = _coerce_float(signal.get("tp_mult_override", setup_tp_mult), setup_tp_mult)
        signal_sl_dist = _coerce_float(signal.get("sl_dist", math.nan), math.nan)
        if math.isfinite(signal_sl_dist) and signal_sl_dist > 0.0:
            sl_dist = float(signal_sl_dist)
        else:
            sl_dist = float(np.clip(sl_mult * atr14, 1.0, 8.0))
        signal_tp_dist = _coerce_float(signal.get("tp_dist", math.nan), math.nan)
        if math.isfinite(signal_tp_dist) and signal_tp_dist > 0.0:
            tp_dist = float(signal_tp_dist)
        else:
            tp_dist = float(np.clip(tp_mult * atr14, max(sl_dist * 1.2, 1.5), 16.0))
        current_stop_price = float(entry_price) - sl_dist if side_num > 0 else float(entry_price) + sl_dist
        resolved_size, size_multiplier = _resolve_aetherflow_signal_size(signal)
        requested_size = int(resolved_size)
        resolved_size, drawdown_size_diag = _apply_aetherflow_drawdown_size_cap(
            requested_size,
            max(0.0, float(peak) - float(equity)),
            drawdown_size_cfg,
        )
        horizon_bars = _coerce_int(
            signal.get(
                "horizon_bars_override",
                signal.get("horizon_bars", signal.get("setup_horizon_bars", 0.0)),
            ),
            0,
        )
        active_trade = {
            "entry_time": entry_time,
            "entry_bar_index": int(entry_bar_index),
            "entry_price": float(entry_price),
            "entry_intrabar_fill": bool(intrabar_fill),
            "side_num": side_num,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "sl_mult": float(sl_mult),
            "tp_mult": float(tp_mult),
            "size": int(resolved_size),
            "requested_size": int(requested_size),
            "size_multiplier": float(size_multiplier),
            **drawdown_size_diag,
            "bars_held": 0,
            "mfe_points": 0.0,
            "mae_points": 0.0,
            "setup_family": str(signal.get("setup_family", signal.get("aetherflow_setup_family", "")) or ""),
            "confidence": _coerce_float(signal.get("confidence", signal.get("aetherflow_confidence", 0.0)), 0.0),
            "regime_name": str(signal.get("manifold_regime_name", signal.get("aetherflow_regime", "")) or ""),
            "session": _session_name(entry_time),
            "horizon_bars": horizon_bars,
            "use_horizon_time_stop": _coerce_bool(signal.get("use_horizon_time_stop", use_horizon_time_stop), bool(use_horizon_time_stop)),
            "entry_mode": str(signal.get("entry_mode", "market_next_bar") or "market_next_bar"),
            "current_stop_price": float(current_stop_price),
            "break_even_enabled": _coerce_bool(signal.get("break_even_enabled", False), False),
            "break_even_activate_on_next_bar": _coerce_bool(signal.get("break_even_activate_on_next_bar", True), True),
            "break_even_trigger_pct": _coerce_float(signal.get("break_even_trigger_pct", 0.0), 0.0),
            "break_even_buffer_ticks": max(0, _coerce_int(signal.get("break_even_buffer_ticks", 0), 0)),
            "break_even_trail_pct": _coerce_float(signal.get("break_even_trail_pct", 0.0), 0.0),
            "break_even_armed": False,
            "break_even_applied": False,
            "break_even_move_count": 0,
            "break_even_last_stop_price": None,
            "break_even_pending_stop_price": None,
            "break_even_locked_points": 0.0,
            "early_exit_enabled": _coerce_bool(signal.get("early_exit_enabled", False), False),
            "early_exit_exit_if_not_green_by": max(0, _coerce_int(signal.get("early_exit_exit_if_not_green_by", 0), 0)),
            "early_exit_max_profit_crosses": max(0, _coerce_int(signal.get("early_exit_max_profit_crosses", 0), 0)),
            "profit_crosses": 0,
            "was_green": None,
        }
        active = True
        pending_entry = None

    for i in positions:
        ts = index[i]
        bar_open = float(opens[i])
        bar_high = float(highs[i])
        bar_low = float(lows[i])
        bar_close = float(closes[i])
        holiday_closed_now = bool(holiday_flat_arr[i])
        entry_window_blocked = _minute_window_blocked(hours[i], minutes[i])
        force_flat_now = bool(force_flat_enabled and hours[i] == force_flat_hour and minutes[i] >= force_flat_minute)

        if pending_entry is not None and not active:
            if holiday_closed_now or entry_window_blocked or _risk_governor_blocks_new_entries(ts):
                pending_entry = None
            else:
                signal_bar_index = _coerce_int(pending_entry.get("_signal_bar_index", i - 1), i - 1)
                bars_since_signal = max(0, int(i - signal_bar_index))
                entry_mode = str(pending_entry.get("_entry_mode", "market_next_bar") or "market_next_bar").strip().lower()
                if entry_mode in {"market", "market_next_bar", "next_bar_market"}:
                    activate_trade_from_signal(
                        pending_entry,
                        entry_time=ts,
                        entry_bar_index=int(i),
                        entry_price=float(bar_open),
                        intrabar_fill=False,
                    )
                elif entry_mode == "pullback_limit":
                    entry_window_bars = max(1, _coerce_int(pending_entry.get("_entry_window_bars", 2), 2))
                    if bars_since_signal > entry_window_bars:
                        pending_entry = None
                    else:
                        side_num = _coerce_int(pending_entry.get("candidate_side", 0), 0)
                        if side_num == 0:
                            pending_entry = None
                        else:
                            limit_price = pending_entry.get("_entry_limit_price")
                            if limit_price is None:
                                atr14 = max(_coerce_float(pending_entry.get("atr14", 1.0), 1.0), 1e-9)
                                pullback_atr = max(0.0, _coerce_float(pending_entry.get("entry_pullback_atr", 0.0), 0.0))
                                if pullback_atr <= 0.0:
                                    activate_trade_from_signal(
                                        pending_entry,
                                        entry_time=ts,
                                        entry_bar_index=int(i),
                                        entry_price=float(bar_open),
                                        intrabar_fill=False,
                                    )
                                    limit_price = None
                                else:
                                    pullback_points = float(pullback_atr) * float(atr14)
                                    raw_limit_price = float(bar_open) - pullback_points if side_num > 0 else float(bar_open) + pullback_points
                                    limit_price = align_stop_price_to_tick(raw_limit_price, side_num)
                                    pending_entry["_entry_limit_price"] = float(limit_price)
                                    pending_entry["_entry_reference_price"] = float(bar_open)
                            if pending_entry is not None and limit_price is not None and not active:
                                fill_price = None
                                intrabar_fill = False
                                if side_num > 0:
                                    if gap_fills and bar_open <= float(limit_price):
                                        fill_price = float(bar_open)
                                    elif bar_low <= float(limit_price):
                                        fill_price = float(limit_price)
                                        intrabar_fill = True
                                else:
                                    if gap_fills and bar_open >= float(limit_price):
                                        fill_price = float(bar_open)
                                    elif bar_high >= float(limit_price):
                                        fill_price = float(limit_price)
                                        intrabar_fill = True
                                if fill_price is not None:
                                    activate_trade_from_signal(
                                        pending_entry,
                                        entry_time=ts,
                                        entry_bar_index=int(i),
                                        entry_price=float(fill_price),
                                        intrabar_fill=bool(intrabar_fill),
                                    )
                                elif bars_since_signal >= entry_window_bars:
                                    pending_entry = None
                else:
                    activate_trade_from_signal(
                        pending_entry,
                        entry_time=ts,
                        entry_bar_index=int(i),
                        entry_price=float(bar_open),
                        intrabar_fill=False,
                    )

        if active:
            if bool(active_trade.get("entry_intrabar_fill", False)) and int(active_trade.get("entry_bar_index", -1)) == int(i):
                active_trade["entry_intrabar_fill"] = False
                continue
            update_excursions(bar_high, bar_low)
            active_trade["bars_held"] = int(i - active_trade["entry_bar_index"] + 1)
            apply_pending_break_even_stop_update(active_trade)
            stage_break_even_stop_update(active_trade)
            side = "LONG" if int(active_trade["side_num"]) > 0 else "SHORT"
            if side == "LONG":
                stop_price = float(active_trade.get("current_stop_price", float(active_trade["entry_price"]) - float(active_trade["sl_dist"])))
                take_price = float(active_trade["entry_price"]) + float(active_trade["tp_dist"])
                if holiday_closed_now or force_flat_now:
                    close_trade(bar_open, ts, "holiday_flat" if holiday_closed_now else "force_flat", i)
                    continue
                if gap_fills and bar_open <= stop_price:
                    close_trade(bar_open, ts, "stop_gap", i)
                    continue
                if gap_fills and bar_open >= take_price:
                    close_trade(bar_open, ts, "take_gap", i)
                    continue
                hit_stop = bar_low <= stop_price
                hit_take = bar_high >= take_price
            else:
                stop_price = float(active_trade.get("current_stop_price", float(active_trade["entry_price"]) + float(active_trade["sl_dist"])))
                take_price = float(active_trade["entry_price"]) - float(active_trade["tp_dist"])
                if holiday_closed_now or force_flat_now:
                    close_trade(bar_open, ts, "holiday_flat" if holiday_closed_now else "force_flat", i)
                    continue
                if gap_fills and bar_open >= stop_price:
                    close_trade(bar_open, ts, "stop_gap", i)
                    continue
                if gap_fills and bar_open <= take_price:
                    close_trade(bar_open, ts, "take_gap", i)
                    continue
                hit_stop = bar_high >= stop_price
                hit_take = bar_low <= take_price

            if hit_stop and hit_take:
                exit_price, exit_reason = bt._resolve_sl_tp_conflict(side, bar_open, bar_close, stop_price, take_price)
                close_trade(float(exit_price), ts, str(exit_reason), i)
                continue
            if hit_stop:
                close_trade(stop_price, ts, "stop", i)
                continue
            if hit_take:
                close_trade(take_price, ts, "take", i)
                continue

            if bool(active_trade.get("use_horizon_time_stop", use_horizon_time_stop)) and int(active_trade.get("horizon_bars", 0) or 0) > 0 and int(active_trade["bars_held"]) >= int(active_trade["horizon_bars"]):
                close_trade(bar_close, ts, "horizon", i)
                continue
            if check_early_exit(bar_close):
                close_trade(bar_close, ts, "early_exit", i)
                continue

        if active or pending_entry is not None:
            continue
        if holiday_closed_now or entry_window_blocked or i >= (len(df) - 1):
            continue
        if _risk_governor_blocks_new_entries(ts):
            signal = signal_lookup.get(int(ts.value))
            if signal is not None:
                risk_governor_blocked_signals += 1
            continue

        signal = signal_lookup.get(int(ts.value))
        if signal is None:
            continue
        pending_entry = dict(signal)
        entry_mode = str(pending_entry.get("entry_mode", "market_next_bar") or "market_next_bar").strip().lower()
        pending_entry["_signal_bar_index"] = int(i)
        pending_entry["_entry_mode"] = entry_mode
        pending_entry["_entry_window_bars"] = max(
            1,
            _coerce_int(
                pending_entry.get(
                    "entry_window_bars",
                    1 if entry_mode in {"market", "market_next_bar", "next_bar_market"} else 2,
                ),
                1,
            ),
        )
        pending_entry["_entry_limit_price"] = None

    if active:
        close_trade(float(closes[positions[-1]]), index[positions[-1]], "end_of_range", int(positions[-1]))

    risk_metrics = bt._compute_backtest_risk_metrics(trade_log)
    summary = {
        "equity": float(round(equity, 2)),
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": float((wins / trades) * 100.0) if trades > 0 else 0.0,
        "max_drawdown": float(round(max_drawdown, 2)),
        "gross_profit": float(round(gross_profit, 2)),
        "gross_loss": float(round(gross_loss, 2)),
        "trade_log": trade_log,
        "exit_reason_counts": exit_reason_counts,
        "session_counts": session_counts,
        "selection": {"strategies": ["AetherFlowStrategy"], "filters": []},
        "symbol_mode": "direct_aetherflow",
        "fast_mode": {"enabled": True, "bar_stride": 1, "skip_mfe_mae": False},
        "entry_window_block_enabled": bool(no_entry_window),
        "entry_window_block_hours_et": [int(no_entry_start), int(no_entry_end)],
        "force_flat_enabled": bool(force_flat_enabled),
        "force_flat_time_et": f"{force_flat_hour:02d}:{force_flat_minute:02d}",
        "bar_minutes": 1,
        "drawdown_size_scaling_enabled": bool(drawdown_size_cfg.get("enabled", False)),
        "drawdown_size_scaling_source": str(drawdown_size_cfg.get("source", "") or ""),
        "drawdown_size_scaling_start_usd": float(drawdown_size_cfg.get("start_usd", 0.0) or 0.0),
        "drawdown_size_scaling_max_usd": float(drawdown_size_cfg.get("max_usd", 0.0) or 0.0),
        "drawdown_size_scaling_base_contracts": int(drawdown_size_cfg.get("base_contracts", 0) or 0),
        "drawdown_size_scaling_min_contracts": int(drawdown_size_cfg.get("min_contracts", 0) or 0),
        "aetherflow_risk_governor_enabled": bool(risk_governor_cfg.get("enabled", False)),
        "aetherflow_risk_governor_daily_loss_stop_usd": float(risk_governor_cfg.get("daily_loss_stop_usd", 0.0) or 0.0),
        "aetherflow_risk_governor_min_loss_trades": int(risk_governor_cfg.get("min_loss_trades", 0) or 0),
        "aetherflow_risk_governor_blocked_days": int(len(risk_governor_blocked_days)),
        "aetherflow_risk_governor_blocked_signals": int(risk_governor_blocked_signals),
        "break_even_armed_trades": int(break_even_armed_trade_count),
        "break_even_stop_updates": int(break_even_stop_update_count),
        "early_exit_closes": int(early_exit_close_count),
        "report": "",
    }
    summary.update(risk_metrics)
    return summary


def _simulate(
    *,
    df: pd.DataFrame,
    signals: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    use_horizon_time_stop: bool,
    allow_same_side_add_ons: bool = False,
    max_same_side_legs: int = 1,
    risk_governor_override: dict | None = None,
) -> dict:
    if bool(allow_same_side_add_ons) and int(max_same_side_legs) > 1:
        from tools.backtest_aetherflow_multi_leg import simulate_same_side_add_ons

        return simulate_same_side_add_ons(
            df=df,
            signals=signals,
            start_time=start_time,
            end_time=end_time,
            use_horizon_time_stop=use_horizon_time_stop,
            max_same_side_legs=int(max_same_side_legs),
            risk_governor_override=risk_governor_override,
        )
    return _simulate_single_position(
        df=df,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
        use_horizon_time_stop=use_horizon_time_stop,
        risk_governor_override=risk_governor_override,
    )


def _converted_trade_log_for_montecarlo(trade_log: list[dict], gate_threshold: float) -> list[dict]:
    out: list[dict] = []
    for trade in trade_log:
        row = dict(trade)
        row["combo_key"] = str(trade.get("aetherflow_setup_family", "") or "")
        row["rule_id"] = str(trade.get("aetherflow_regime", "") or "")
        row["gate_prob"] = trade.get("aetherflow_confidence")
        row["gate_threshold"] = float(gate_threshold)
        out.append(row)
    return out


def main() -> None:
    configured_aetherflow_cfg = dict(CONFIG.get("AETHERFLOW_STRATEGY", {}) or {})
    parser = argparse.ArgumentParser(description="Direct AetherFlow backtest using cached manifold features and setup horizons.")
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument(
        "--base-features",
        default=str(configured_aetherflow_cfg.get("backtest_base_features_file", DEFAULT_FULL_MANIFOLD_BASE_FEATURES) or DEFAULT_FULL_MANIFOLD_BASE_FEATURES),
        help="Full precomputed manifold base cache. Defaults to the canonical promoted AF full-range manifold cache.",
    )
    parser.add_argument(
        "--model-file",
        default=str(configured_aetherflow_cfg.get("model_file", "model_aetherflow_fullrange_v2.pkl") or "model_aetherflow_fullrange_v2.pkl"),
    )
    parser.add_argument(
        "--thresholds-file",
        default=str(configured_aetherflow_cfg.get("thresholds_file", "aetherflow_thresholds_fullrange_v2.json") or "aetherflow_thresholds_fullrange_v2.json"),
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol-mode", default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"))
    parser.add_argument("--symbol-method", default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"))
    parser.add_argument("--report-dir", default="backtest_reports/aetherflow_direct")
    parser.add_argument("--history-buffer-days", type=int, default=14)
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--min-confidence-override", type=float, default=None)
    parser.add_argument("--allow-setup", action="append", default=None)
    parser.add_argument("--block-regime", action="append", default=None)
    parser.add_argument("--use-horizon-time-stop", action="store_true")
    parser.add_argument("--allow-same-side-add-ons", action="store_true")
    parser.add_argument("--max-same-side-legs", type=int, default=1)
    args = parser.parse_args()

    source_path = _resolve_source(args.source)
    start_time = bt.parse_user_datetime(str(args.start), bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(str(args.end), bt.NY_TZ, is_end=True)
    if start_time > end_time:
        raise SystemExit("Start must be before end.")

    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(args.symbol_mode or "single").strip().lower(),
        str(args.symbol_method or "volume").strip().lower(),
        int(args.history_buffer_days),
    )

    allow_setups = {str(x).strip() for x in (args.allow_setup or []) if str(x).strip()}
    if args.block_regime:
        block_regimes = {str(x).strip().upper() for x in (args.block_regime or []) if str(x).strip()}
    else:
        block_regimes = {
            str(x).strip().upper()
            for x in (CONFIG.get("AETHERFLOW_STRATEGY", {}).get("hazard_block_regimes", []) or [])
            if str(x).strip()
        }
    configured_allowed_session_ids = configured_aetherflow_cfg.get("allowed_session_ids")
    allowed_session_ids = (
        {int(x) for x in (configured_allowed_session_ids or [])}
        if configured_allowed_session_ids
        else None
    )

    configured_threshold_override = CONFIG.get("AETHERFLOW_STRATEGY", {}).get("threshold_override", None)
    try:
        configured_threshold_override = (
            float(configured_threshold_override)
            if configured_threshold_override is not None
            else None
        )
    except Exception:
        configured_threshold_override = None
    configured_min_conf = float(CONFIG.get("AETHERFLOW_STRATEGY", {}).get("min_confidence", 0.0) or 0.0)
    threshold_override = (
        float(args.threshold_override)
        if args.threshold_override is not None
        else configured_threshold_override
    )
    effective_threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(max(configured_min_conf, args.min_confidence_override if args.min_confidence_override is not None else 0.0))
    )
    configured_model_path = (ROOT / str(configured_aetherflow_cfg.get("model_file", ""))).resolve()
    configured_thresholds_path = (ROOT / str(configured_aetherflow_cfg.get("thresholds_file", ""))).resolve()
    selected_model_path = (ROOT / str(args.model_file)).resolve()
    selected_thresholds_path = (ROOT / str(args.thresholds_file)).resolve()
    use_runtime_policy = bool(
        selected_model_path == configured_model_path
        and selected_thresholds_path == configured_thresholds_path
        and args.threshold_override is None
        and args.min_confidence_override is None
        and not args.allow_setup
        and not args.block_regime
    )
    signals = _build_signal_frame(
        symbol_df=symbol_df,
        base_features_path=resolve_full_manifold_base_features_path(args.base_features),
        model_path=selected_model_path,
        thresholds_path=selected_thresholds_path,
        threshold_override=threshold_override,
        min_confidence_override=effective_threshold,
        allow_setups=allow_setups or None,
        block_regimes=block_regimes,
        allowed_session_ids=allowed_session_ids,
        use_runtime_policy=use_runtime_policy,
    )
    if isinstance(signals, pd.DataFrame) and not signals.empty:
        signals = signals.loc[(signals.index >= start_time) & (signals.index <= end_time)].copy()
    stats = _simulate(
        df=symbol_df,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
        use_horizon_time_stop=bool(args.use_horizon_time_stop),
        allow_same_side_add_ons=bool(args.allow_same_side_add_ons),
        max_same_side_legs=int(args.max_same_side_legs),
    )
    stats["symbol_mode"] = str(args.symbol_mode or "").strip().lower()
    stats["symbol_distribution"] = symbol_distribution

    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = bt.save_backtest_report(stats, symbol, start_time, end_time, output_dir=report_dir)
    csv_path = report_dir / f"converted_{report_path.stem}.csv"
    _write_converted_csv(csv_path, _converted_trade_log_for_montecarlo(stats.get("trade_log", []) or [], effective_threshold))

    print(f"report={report_path}")
    print(f"csv_path={csv_path}")
    print(f"symbol={symbol}")
    print("selected_strategies=['AetherFlowStrategy']")
    print("selected_filters=[]")
    print(f"policy_mode={'runtime_config' if use_runtime_policy else 'threshold_only'}")
    print(f"equity={stats.get('equity')}")
    print(f"trades={stats.get('trades')}")
    print(f"winrate={stats.get('winrate')}")
    print(f"max_drawdown={stats.get('max_drawdown')}")
    print(f"signals={len(signals)}")


if __name__ == "__main__":
    main()

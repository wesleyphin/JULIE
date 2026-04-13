import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import backtest_mes_et as bt
import data_cache
from config import CONFIG
from de3_v2_scoring import (
    compute_structural_rank_fields,
    compute_oos_block_stats,
    evaluate_grid_metrics,
    evaluate_single_combo,
    gpu_backend_available,
    select_plateau_candidate,
    stability_weighted_score,
)
from train_dynamic_engine3 import _build_trades as _build_trades_base, _resample_df


TICK_SIZE = 0.25
NY_TZ = bt.NY_TZ


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h{minutes:02d}m"


def _parse_date(value: Optional[str], *, is_end: bool = False) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    has_time = ("T" in raw) or (":" in raw)
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    if is_end and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _filter_range(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


def _round_to_tick(value: float, tick: float = TICK_SIZE) -> float:
    if tick <= 0:
        return float(value)
    return float(round(float(value) / float(tick)) * float(tick))


def _safe_profit_factor(gross_win: float, gross_loss: float, *, cap: float = 10.0) -> float:
    gross_win = float(gross_win)
    gross_loss = float(gross_loss)
    if gross_loss <= 0.0:
        if gross_win <= 0.0:
            return 0.0
        return float(cap)
    pf = gross_win / gross_loss
    if not math.isfinite(pf) or pf < 0.0:
        return 0.0
    return float(min(pf, cap))


def _oos_trade_quality(pnls: np.ndarray, *, sl_dist: float, tp_dist: float) -> Dict[str, float]:
    arr = np.asarray(pnls, dtype=float)
    n = int(arr.size)
    if n <= 0:
        return {
            "count_trades": 0,
            "count_stop_like": 0,
            "count_take_like": 0,
            "stop_like_share": 0.0,
            "take_like_share": 0.0,
            "loss_share": 0.0,
            "pnl_p10": 0.0,
            "pnl_p05": 0.0,
            "pnl_p01": 0.0,
        }
    tol = max(1e-9, float(TICK_SIZE) * 0.25)
    stop_like = np.isclose(arr, -float(sl_dist), atol=tol, rtol=0.0)
    take_like = np.isclose(arr, float(tp_dist), atol=tol, rtol=0.0)
    stop_count = int(np.sum(stop_like))
    take_count = int(np.sum(take_like))
    loss_share = float(np.mean(arr < 0.0))
    return {
        "count_trades": int(n),
        "count_stop_like": stop_count,
        "count_take_like": take_count,
        "stop_like_share": float(stop_count / n),
        "take_like_share": float(take_count / n),
        "loss_share": float(loss_share),
        "pnl_p10": float(np.percentile(arr, 10)),
        "pnl_p05": float(np.percentile(arr, 5)),
        "pnl_p01": float(np.percentile(arr, 1)),
    }


def _build_tp_pairs(
    sl_list: Sequence[float],
    rr_list: Sequence[float],
    *,
    min_tp: float,
    max_tp: float,
    tick_size: float,
) -> List[Tuple[float, float]]:
    pairs = []
    for sl in sl_list:
        for rr in rr_list:
            tp = _round_to_tick(float(sl) * float(rr), tick_size)
            if tp < float(min_tp) or tp > float(max_tp):
                continue
            pairs.append((float(sl), float(tp)))
    return sorted(set(pairs))


def _build_resample_cache_key(
    source: Path,
    df: pd.DataFrame,
    symbol_mode: str,
    symbol_method: str,
) -> str:
    start = df.index.min().isoformat() if not df.empty else ""
    end = df.index.max().isoformat() if not df.empty else ""
    token = f"{source.resolve()}|{len(df)}|{start}|{end}|{symbol_mode}|{symbol_method}"
    # Use a stable hash so cache keys remain reusable across Python processes.
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:20]


def _load_or_build_resamples(
    df_1m: pd.DataFrame,
    *,
    cache_dir: Optional[Path],
    cache_key: str,
    use_cache: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    p5 = cache_dir / f"de3v2_{cache_key}_5m.parquet" if cache_dir else None
    p15 = cache_dir / f"de3v2_{cache_key}_15m.parquet" if cache_dir else None

    df_5m = None
    df_15m = None
    if use_cache and p5 and p15 and p5.exists() and p15.exists():
        try:
            df_5m = pd.read_parquet(p5)
            df_15m = pd.read_parquet(p15)
            logging.info("Loaded cached resamples: %s | %s", p5, p15)
        except Exception as exc:
            logging.warning("Resample cache load failed: %s", exc)
            df_5m, df_15m = None, None

    if df_5m is None or df_15m is None:
        t0 = time.time()
        df_5m = _resample_df(df_1m, 5)
        df_15m = _resample_df(df_1m, 15)
        logging.info(
            "Built resamples in %s (5m=%s rows, 15m=%s rows)",
            _format_duration(time.time() - t0),
            len(df_5m),
            len(df_15m),
        )
        if use_cache and p5 and p15:
            try:
                df_5m.to_parquet(p5, index=True)
                df_15m.to_parquet(p15, index=True)
            except Exception as exc:
                logging.warning("Resample cache write failed: %s", exc)

    return df_5m, df_15m


def _normalize_family_tag(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    clean_chars: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            clean_chars.append(ch)
        elif ch in {" ", "|", ":"}:
            clean_chars.append("_")
    return "".join(clean_chars).strip("_")


def _strategy_id_for_row(
    *,
    tf_label: str,
    session: str,
    stype: str,
    family_tag: str,
    thresh: float,
    sl: float,
    tp: float,
) -> str:
    parts = [
        str(tf_label or "").strip(),
        str(session or "").strip(),
        str(stype or "").strip(),
    ]
    family_tag_norm = _normalize_family_tag(family_tag)
    if family_tag_norm:
        parts.append(family_tag_norm)
    parts.extend(
        [
            f"T{_format_threshold_component(thresh)}",
            f"SL{_format_threshold_component(sl)}",
            f"TP{_format_threshold_component(tp)}",
        ]
    )
    return "_".join([p for p in parts if p])


def _build_runtime_strategy_fields(
    *,
    tf_label: str,
    session: str,
    stype: str,
    family_tag: str,
    thresh: float,
    sl: float,
    tp: float,
    family_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    family_tag_norm = _normalize_family_tag(family_tag)
    fields: Dict[str, object] = {
        "strategy_id": _strategy_id_for_row(
            tf_label=tf_label,
            session=session,
            stype=stype,
            family_tag=family_tag_norm,
            thresh=thresh,
            sl=sl,
            tp=tp,
        )
    }
    if not family_tag_norm:
        return fields

    fields["FamilyTag"] = family_tag_norm
    profile = dict(family_profile or {})
    trigger_side = str(profile.get("candle_side", "") or "").strip().lower()
    if trigger_side:
        fields["TriggerCandleSide"] = trigger_side

    profile_key_map = {
        "min_body_ratio": "TriggerMinBodyRatio",
        "max_body_ratio": "TriggerMaxBodyRatio",
        "min_close_pos1": "TriggerMinClosePos1",
        "max_close_pos1": "TriggerMaxClosePos1",
        "min_upper_wick_ratio": "TriggerMinUpperWickRatio",
        "max_upper_wick_ratio": "TriggerMaxUpperWickRatio",
        "min_lower_wick_ratio": "TriggerMinLowerWickRatio",
        "max_lower_wick_ratio": "TriggerMaxLowerWickRatio",
        "min_body_thresh_ratio": "TriggerMinBodyThreshRatio",
        "max_body_thresh_ratio": "TriggerMaxBodyThreshRatio",
    }
    for src_key, dst_key in profile_key_map.items():
        if src_key not in profile:
            continue
        try:
            value = float(profile.get(src_key))
        except Exception:
            continue
        if math.isfinite(value):
            fields[dst_key] = float(value)
    return fields


def _format_threshold_component(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "0"
    rounded = round(val)
    if abs(val - rounded) <= 1e-9:
        return str(int(rounded))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _default_family_profiles() -> List[Dict[str, Any]]:
    missing_short_sessions = ["03-06", "06-09", "18-21", "21-24"]
    return [
        {
            "name": "bear_impulse_short_mom",
            "family_tag": "BearImpulse",
            "enabled": True,
            "strategy_type": "Short_Mom",
            "candle_side": "red",
            "apply_sessions": list(missing_short_sessions),
            "min_body_ratio": 0.58,
            "max_close_pos1": 0.18,
            "max_lower_wick_ratio": 0.12,
            "min_body_thresh_ratio": 1.25,
        },
        {
            "name": "bull_exhaust_short_rev",
            "family_tag": "BullExhaust",
            "enabled": True,
            "strategy_type": "Short_Rev",
            "candle_side": "green",
            "apply_sessions": list(missing_short_sessions),
            "min_upper_wick_ratio": 0.30,
            "max_body_ratio": 0.62,
            "max_close_pos1": 0.82,
            "min_body_thresh_ratio": 1.10,
        },
    ]


def _resolve_family_profiles(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    family_cfg = cfg.get("family_profiles", {}) or {}
    if not bool(family_cfg.get("enabled", False)):
        return []
    raw_profiles = family_cfg.get("profiles", None)
    profiles = raw_profiles if isinstance(raw_profiles, list) and raw_profiles else _default_family_profiles()
    clean: List[Dict[str, Any]] = []
    for raw in profiles:
        if not isinstance(raw, dict):
            continue
        if not bool(raw.get("enabled", True)):
            continue
        family_tag = _normalize_family_tag(raw.get("family_tag", raw.get("name", "")))
        strategy_type = str(raw.get("strategy_type", "") or "").strip()
        candle_side = str(raw.get("candle_side", "") or "").strip().lower()
        if not family_tag or not strategy_type or candle_side not in {"red", "green"}:
            continue
        profile = dict(raw)
        profile["family_tag"] = family_tag
        profile["strategy_type"] = strategy_type
        profile["candle_side"] = candle_side
        profile["apply_sessions"] = [
            str(item).strip()
            for item in (raw.get("apply_sessions", []) if isinstance(raw.get("apply_sessions", []), (list, tuple, set)) else [])
            if str(item).strip()
        ]
        clean.append(profile)
    return clean


def _build_trades_with_family_profiles(
    df_tf: pd.DataFrame,
    df_1m_index: np.ndarray,
    thresholds: Sequence[float],
    *,
    family_profiles: Sequence[Dict[str, Any]],
    label: str = "",
) -> Dict[Tuple[str, str, float, str], List[tuple]]:
    base_trades = _build_trades_base(df_tf, df_1m_index, list(thresholds), label=label)
    out: Dict[Tuple[str, str, float, str], List[tuple]] = defaultdict(list)
    for (session, stype, thresh), items in base_trades.items():
        if items:
            out[(str(session), str(stype), float(thresh), "")].extend(items)

    if df_tf.empty or not family_profiles:
        return out

    times = df_tf.index
    times_ny = times.tz_convert(NY_TZ) if times.tz is not None else times.tz_localize(NY_TZ)
    times_ns = times.values.astype("datetime64[ns]").astype("int64")
    entry_pos_1m_all = np.searchsorted(df_1m_index, times_ns, side="left")

    open_arr = df_tf["open"].to_numpy()
    high_arr = df_tf["high"].to_numpy()
    low_arr = df_tf["low"].to_numpy()
    close_arr = df_tf["close"].to_numpy()
    min_thresh = min(thresholds)
    thresholds_sorted = np.array(sorted(thresholds), dtype=float)
    threshold_prefix = [tuple() for _ in range(len(thresholds_sorted) + 1)]
    running: List[float] = []
    for k, v in enumerate(thresholds_sorted, start=1):
        running.append(float(v))
        threshold_prefix[k] = tuple(running)

    hours = times_ny.hour
    session_idx = (hours // 3).astype(int)
    session_labels = np.array(
        ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"],
        dtype=object,
    )
    session_bucket = session_labels[session_idx]
    body = close_arr[:-1] - open_arr[:-1]
    abs_body = np.abs(body)
    valid_mask = abs_body > min_thresh
    if not np.any(valid_mask):
        return out

    family_counts: Counter = Counter()
    for prev_i in np.nonzero(valid_mask)[0]:
        i = prev_i + 1
        body_i = float(body[prev_i])
        abs_body_i = float(abs_body[prev_i])
        if abs_body_i <= 0.0 or body_i == 0.0:
            continue
        entry_pos_1m = int(entry_pos_1m_all[i])
        if entry_pos_1m >= len(df_1m_index):
            continue
        session = str(session_bucket[i] or "")
        prev_open = float(open_arr[prev_i])
        prev_close = float(close_arr[prev_i])
        prev_high = float(high_arr[prev_i])
        prev_low = float(low_arr[prev_i])
        prev_range = float(prev_high - prev_low)
        if not math.isfinite(prev_range) or prev_range <= 1e-9:
            continue
        body_ratio = float(abs_body_i / prev_range)
        close_pos1 = float((prev_close - prev_low) / prev_range)
        upper_wick_ratio = float((prev_high - max(prev_open, prev_close)) / prev_range)
        lower_wick_ratio = float((min(prev_open, prev_close) - prev_low) / prev_range)
        is_red = bool(body_i < 0.0)
        is_green = bool(body_i > 0.0)
        active_thresholds = threshold_prefix[int(np.searchsorted(thresholds_sorted, abs_body_i, side="left"))]
        if not active_thresholds:
            continue

        entry_time_ns = int(times_ns[i])
        entry_price = float(open_arr[i])
        entry_tuple = (entry_pos_1m, i, entry_price, entry_time_ns)

        for profile in family_profiles:
            apply_sessions = profile.get("apply_sessions", [])
            if apply_sessions and session not in apply_sessions:
                continue
            candle_side = str(profile.get("candle_side", "") or "").strip().lower()
            if candle_side == "red" and not is_red:
                continue
            if candle_side == "green" and not is_green:
                continue
            family_tag = str(profile.get("family_tag", "") or "").strip()
            strategy_type = str(profile.get("strategy_type", "") or "").strip()
            min_body_ratio = float(profile.get("min_body_ratio", float("-inf")) or float("-inf"))
            max_body_ratio = float(profile.get("max_body_ratio", float("inf")) or float("inf"))
            min_close_pos1 = float(profile.get("min_close_pos1", float("-inf")) or float("-inf"))
            max_close_pos1 = float(profile.get("max_close_pos1", float("inf")) or float("inf"))
            min_upper_wick_ratio = float(profile.get("min_upper_wick_ratio", float("-inf")) or float("-inf"))
            max_upper_wick_ratio = float(profile.get("max_upper_wick_ratio", float("inf")) or float("inf"))
            min_lower_wick_ratio = float(profile.get("min_lower_wick_ratio", float("-inf")) or float("-inf"))
            max_lower_wick_ratio = float(profile.get("max_lower_wick_ratio", float("inf")) or float("inf"))
            if body_ratio < min_body_ratio or body_ratio > max_body_ratio:
                continue
            if close_pos1 < min_close_pos1 or close_pos1 > max_close_pos1:
                continue
            if upper_wick_ratio < min_upper_wick_ratio or upper_wick_ratio > max_upper_wick_ratio:
                continue
            if lower_wick_ratio < min_lower_wick_ratio or lower_wick_ratio > max_lower_wick_ratio:
                continue
            for thresh in active_thresholds:
                body_thresh_ratio = float(abs_body_i / max(1e-9, float(thresh)))
                min_body_thresh_ratio = float(
                    profile.get("min_body_thresh_ratio", float("-inf")) or float("-inf")
                )
                max_body_thresh_ratio = float(
                    profile.get("max_body_thresh_ratio", float("inf")) or float("inf")
                )
                if body_thresh_ratio < min_body_thresh_ratio or body_thresh_ratio > max_body_thresh_ratio:
                    continue
                out[(session, strategy_type, float(thresh), family_tag)].append(entry_tuple)
                family_counts[(family_tag, session, strategy_type)] += 1

    if family_counts:
        summary = ", ".join(
            [
                f"{fam}:{sess}:{stype}={cnt}"
                for (fam, sess, stype), cnt in sorted(family_counts.items())
            ]
        )
        logging.info("DE3 v2 family-profile candidates %s | %s", label or "build", summary)
    return out


def _profile_override_int(
    family_profile: Optional[Dict[str, Any]],
    key: str,
    default: int,
    *,
    min_value: Optional[int] = None,
) -> int:
    if not isinstance(family_profile, dict) or key not in family_profile:
        return int(default)
    try:
        value = int(family_profile.get(key))
    except Exception:
        return int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    return int(value)


def _profile_override_float(
    family_profile: Optional[Dict[str, Any]],
    key: str,
    default: Optional[float],
    *,
    min_value: Optional[float] = None,
) -> Optional[float]:
    if not isinstance(family_profile, dict) or key not in family_profile:
        return default
    try:
        value = float(family_profile.get(key))
    except Exception:
        return default
    if not math.isfinite(value):
        return default
    if min_value is not None:
        value = max(float(min_value), value)
    return float(value)


def _profile_override_bool(
    family_profile: Optional[Dict[str, Any]],
    key: str,
    default: bool = False,
) -> bool:
    if not isinstance(family_profile, dict) or key not in family_profile:
        return bool(default)
    raw = family_profile.get(key)
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _profile_override_text(
    family_profile: Optional[Dict[str, Any]],
    key: str,
    default: str,
    *,
    allowed: Optional[Set[str]] = None,
) -> str:
    if not isinstance(family_profile, dict) or key not in family_profile:
        return str(default)
    text = str(family_profile.get(key) or "").strip()
    if not text:
        return str(default)
    if allowed is not None and text not in allowed:
        return str(default)
    return text


def _family_selection_overrides(
    *,
    family_profile: Optional[Dict[str, Any]],
    plateau_cfg: Dict[str, Any],
    scoring_cfg: Dict[str, Any],
    min_train_trades: int,
    min_oos_trades: int,
) -> Dict[str, Any]:
    local_plateau_cfg = dict(plateau_cfg or {})
    local_scoring_cfg = dict(scoring_cfg or {})
    effective_min_train_trades = int(min_train_trades)
    effective_min_oos_trades = int(min_oos_trades)
    fallback_to_best_score = False
    fallback_min_score = None

    if isinstance(family_profile, dict):
        effective_min_train_trades = _profile_override_int(
            family_profile,
            "selection_min_train_trades",
            min_train_trades,
            min_value=1,
        )
        effective_min_oos_trades = _profile_override_int(
            family_profile,
            "selection_min_oos_trades",
            min_oos_trades,
            min_value=1,
        )
        local_plateau_cfg["min_neighbors"] = _profile_override_int(
            family_profile,
            "selection_plateau_min_neighbors",
            int(local_plateau_cfg.get("min_neighbors", 4) or 4),
            min_value=0,
        )
        local_plateau_cfg["neighbor_def"] = _profile_override_text(
            family_profile,
            "selection_plateau_neighbor_def",
            str(local_plateau_cfg.get("neighbor_def", "adjacent_grid") or "adjacent_grid"),
            allowed={"adjacent_grid"},
        )
        plateau_min_score = _profile_override_float(
            family_profile,
            "selection_plateau_min_score",
            float(local_plateau_cfg.get("min_plateau_score", 0.0) or 0.0),
        )
        if plateau_min_score is not None:
            local_plateau_cfg["min_plateau_score"] = float(plateau_min_score)

        scoring_key_map = {
            "selection_min_oos_win_rate": "min_oos_win_rate",
            "selection_max_oos_stop_share": "max_oos_stop_share",
            "selection_max_oos_tail_p10_abs": "max_oos_tail_p10_abs",
            "selection_min_oos_profitable_blocks": "min_oos_profitable_blocks",
            "selection_min_oos_profitable_block_ratio": "min_oos_profitable_block_ratio",
            "selection_min_oos_profit_factor": "min_oos_profit_factor",
            "selection_max_oos_drawdown_norm": "max_oos_drawdown_norm",
        }
        for profile_key, scoring_key in scoring_key_map.items():
            if profile_key not in family_profile:
                continue
            current_value = local_scoring_cfg.get(scoring_key, None)
            if isinstance(current_value, int) and not isinstance(current_value, bool):
                local_scoring_cfg[scoring_key] = _profile_override_int(
                    family_profile,
                    profile_key,
                    int(current_value),
                    min_value=0,
                )
            else:
                current_float = None if current_value is None else float(current_value)
                local_scoring_cfg[scoring_key] = _profile_override_float(
                    family_profile,
                    profile_key,
                    current_float,
                    min_value=0.0,
                )

        fallback_to_best_score = _profile_override_bool(
            family_profile,
            "selection_fallback_to_best_score",
            False,
        )
        fallback_min_score = _profile_override_float(
            family_profile,
            "selection_fallback_min_score",
            None,
        )

    return {
        "plateau_cfg": local_plateau_cfg,
        "scoring_cfg": local_scoring_cfg,
        "min_train_trades": int(effective_min_train_trades),
        "min_oos_trades": int(effective_min_oos_trades),
        "fallback_to_best_score": bool(fallback_to_best_score),
        "fallback_min_score": fallback_min_score,
    }


def _select_best_candidate_fallback(
    rows: Sequence[Dict[str, float]],
    *,
    min_score: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    best: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, float, float, float]] = None
    min_score_value = float(min_score) if min_score is not None else None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            score = float(row.get("score", float("-inf")) or float("-inf"))
        except Exception:
            score = float("-inf")
        if min_score_value is not None and score < float(min_score_value):
            continue
        key = (
            float(score),
            float(row.get("avg_pnl", 0.0) or 0.0),
            float(row.get("profit_factor", 0.0) or 0.0),
            float(row.get("win_rate", 0.0) or 0.0),
        )
        if best is None or best_key is None or key > best_key:
            best = dict(row)
            best_key = key
    if best is None:
        return None
    best["selected_by"] = "best_score_fallback"
    return best


def _build_tasks(
    prepared_5m: Dict[Tuple[str, str, float, str], List[tuple]],
    prepared_15m: Dict[Tuple[str, str, float, str], List[tuple]],
) -> List[Tuple[str, str, str, float, str, List[tuple]]]:
    tasks: List[Tuple[str, str, str, float, str, List[tuple]]] = []
    for (session, stype, thresh, family_tag), items in prepared_5m.items():
        if items:
            tasks.append(("5min", session, stype, float(thresh), str(family_tag or ""), items))
    for (session, stype, thresh, family_tag), items in prepared_15m.items():
        if items:
            tasks.append(("15min", session, stype, float(thresh), str(family_tag or ""), items))
    tasks.sort(key=lambda x: (x[0], x[1], x[2], float(x[3]), x[4]))
    return tasks


def _task_key(task: Tuple[str, str, str, float, str, List[tuple]]) -> str:
    tf_label, session, stype, thresh, family_tag, _ = task
    family_tag_norm = _normalize_family_tag(family_tag)
    family_token = family_tag_norm if family_tag_norm else "base"
    return f"{str(tf_label)}|{str(session)}|{str(stype)}|{float(thresh):.8f}|{family_token}"


def _soft_diversity_select(
    items: Sequence[Dict[str, object]],
    *,
    max_keep: int,
    cfg: Dict[str, object],
    global_sl_counts: Counter,
    global_tp_counts: Counter,
    global_combo_counts: Counter,
) -> List[Dict[str, object]]:
    enabled = bool(cfg.get("enabled", False))
    max_keep = max(1, int(max_keep))
    if not items:
        return []
    if not enabled or len(items) <= 1:
        return list(items[:max_keep])

    try:
        combo_penalty = float(cfg.get("combo_penalty", 0.08) or 0.08)
    except Exception:
        combo_penalty = 0.08
    try:
        sl_penalty = float(cfg.get("sl_penalty", 0.02) or 0.02)
    except Exception:
        sl_penalty = 0.02
    try:
        tp_penalty = float(cfg.get("tp_penalty", 0.02) or 0.02)
    except Exception:
        tp_penalty = 0.02
    try:
        global_scale = float(cfg.get("global_scale", 0.5) or 0.5)
    except Exception:
        global_scale = 0.5

    remaining = list(items)
    selected: List[Dict[str, object]] = []
    local_sl_counts: Counter = Counter()
    local_tp_counts: Counter = Counter()
    local_combo_counts: Counter = Counter()

    while remaining and len(selected) < max_keep:
        best_i = 0
        best_adj = float("-inf")
        best_tiebreak = float("-inf")
        for i, row in enumerate(remaining):
            try:
                sl = float(row.get("Best_SL", 0.0) or 0.0)
            except Exception:
                sl = 0.0
            try:
                tp = float(row.get("Best_TP", 0.0) or 0.0)
            except Exception:
                tp = 0.0
            try:
                base_score = float(row.get("StructuralScore", row.get("Score", 0.0)) or 0.0)
            except Exception:
                base_score = 0.0
            try:
                tiebreak = float((row.get("OOS") or {}).get("avg_pnl", row.get("WorstBlockAvgPnL", 0.0)) or 0.0)
            except Exception:
                tiebreak = 0.0

            combo_key = (sl, tp)
            penalty = 0.0
            penalty += combo_penalty * (
                float(local_combo_counts.get(combo_key, 0))
                + (global_scale * float(global_combo_counts.get(combo_key, 0)))
            )
            penalty += sl_penalty * (
                float(local_sl_counts.get(sl, 0))
                + (global_scale * float(global_sl_counts.get(sl, 0)))
            )
            penalty += tp_penalty * (
                float(local_tp_counts.get(tp, 0))
                + (global_scale * float(global_tp_counts.get(tp, 0)))
            )
            adjusted = float(base_score - penalty)

            if adjusted > best_adj or (adjusted == best_adj and tiebreak > best_tiebreak):
                best_adj = adjusted
                best_tiebreak = tiebreak
                best_i = i

        chosen = remaining.pop(best_i)
        try:
            sl = float(chosen.get("Best_SL", 0.0) or 0.0)
        except Exception:
            sl = 0.0
        try:
            tp = float(chosen.get("Best_TP", 0.0) or 0.0)
        except Exception:
            tp = 0.0
        combo_key = (sl, tp)
        local_sl_counts[sl] += 1
        local_tp_counts[tp] += 1
        local_combo_counts[combo_key] += 1
        global_sl_counts[sl] += 1
        global_tp_counts[tp] += 1
        global_combo_counts[combo_key] += 1

        chosen = dict(chosen)
        chosen["diversity_adjusted_score"] = float(best_adj)
        selected.append(chosen)

    return selected


def _prune_dominated_rows(
    rows: Sequence[Dict[str, object]],
    *,
    scoring_cfg: Dict[str, object],
    preserve_family_buckets: bool = False,
) -> List[Dict[str, object]]:
    enabled = bool(scoring_cfg.get("dominance_pruning_enabled", False))
    if not enabled or not rows:
        return list(rows)

    try:
        avg_tol = float(scoring_cfg.get("dominance_avg_pnl_tolerance", 0.10) or 0.10)
    except Exception:
        avg_tol = 0.10
    try:
        score_tol = float(scoring_cfg.get("dominance_score_tolerance", 0.50) or 0.50)
    except Exception:
        score_tol = 0.50
    try:
        dd_tol = float(scoring_cfg.get("dominance_dd_tolerance", 0.05) or 0.05)
    except Exception:
        dd_tol = 0.05
    require_lower_thresh = bool(scoring_cfg.get("dominance_require_lower_or_equal_thresh", True))

    grouped: Dict[Tuple[str, str, str, float, float], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        try:
            sl = float(row.get("Best_SL", 0.0) or 0.0)
        except Exception:
            sl = 0.0
        try:
            tp = float(row.get("Best_TP", 0.0) or 0.0)
        except Exception:
            tp = 0.0
        key = (
            str(row.get("TF", "")),
            str(row.get("Session", "")),
            str(row.get("Type", "")),
            (
                _normalize_family_tag(row.get("FamilyTag", row.get("family_tag", "")))
                if preserve_family_buckets
                else ""
            ),
            sl,
            tp,
        )
        grouped[key].append(row)

    kept: List[Dict[str, object]] = []
    pruned = 0
    for _, items in grouped.items():
        if len(items) <= 1:
            kept.extend(items)
            continue

        dominated: set[int] = set()
        for i, a in enumerate(items):
            if i in dominated:
                continue
            try:
                a_thresh = float(a.get("Thresh", 0.0) or 0.0)
            except Exception:
                a_thresh = 0.0
            a_oos = a.get("OOS") or {}
            a_avg = float(a_oos.get("avg_pnl", 0.0) or 0.0)
            a_sh = float(a_oos.get("sharpe_like", 0.0) or 0.0)
            a_tr = int(a_oos.get("trades", 0) or 0)
            a_dd = float(a_oos.get("max_oos_drawdown_norm", 0.0) or 0.0)
            a_score = float(a.get("Score", 0.0) or 0.0)
            for j, b in enumerate(items):
                if i == j or j in dominated:
                    continue
                try:
                    b_thresh = float(b.get("Thresh", 0.0) or 0.0)
                except Exception:
                    b_thresh = 0.0
                if require_lower_thresh and a_thresh > b_thresh:
                    continue
                b_oos = b.get("OOS") or {}
                b_avg = float(b_oos.get("avg_pnl", 0.0) or 0.0)
                b_sh = float(b_oos.get("sharpe_like", 0.0) or 0.0)
                b_tr = int(b_oos.get("trades", 0) or 0)
                b_dd = float(b_oos.get("max_oos_drawdown_norm", 0.0) or 0.0)
                b_score = float(b.get("Score", 0.0) or 0.0)

                avg_close_or_better = a_avg >= (b_avg * (1.0 - max(0.0, avg_tol)))
                score_close_or_better = a_score >= (b_score - max(0.0, score_tol))
                better_shape = (a_sh >= b_sh) and (a_tr >= b_tr)
                better_tail = a_dd <= (b_dd + max(0.0, dd_tol))
                if avg_close_or_better and score_close_or_better and better_shape and better_tail:
                    dominated.add(j)

        for idx, row in enumerate(items):
            if idx not in dominated:
                kept.append(row)
            else:
                pruned += 1

    if pruned > 0:
        logging.info("DE3 v2 dominance pruning removed %d near-duplicate candidate(s).", int(pruned))
    return kept


def _resolve_checkpoint_path(out_path: Path, cfg: Dict[str, object], override: Optional[str]) -> Path:
    if override:
        p = Path(str(override))
    else:
        cp_cfg = cfg.get("checkpoint", {}) or {}
        path_cfg = cp_cfg.get("path")
        if path_cfg:
            p = Path(str(path_cfg))
        else:
            p = out_path.with_name(f"{out_path.stem}_checkpoint.json")
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _build_resume_run_key(
    *,
    source_path: Path,
    mode: str,
    cache_key: str,
    split_train_end: Optional[pd.Timestamp],
    split_valid_start: Optional[pd.Timestamp],
    split_valid_end: Optional[pd.Timestamp],
    purge_bars: int,
    thresholds: Sequence[float],
    sl_list: Sequence[float],
    rr_list: Sequence[float],
    min_tp: float,
    max_tp: float,
    execution_rules: Dict[str, object],
    tp_pairs: Sequence[Tuple[float, float]],
    task_keys: Sequence[str],
    config_signature: Optional[Dict[str, object]] = None,
) -> str:
    payload = {
        "source": str(source_path.resolve()),
        "mode": str(mode),
        "cache_key": str(cache_key),
        "train_end": split_train_end.isoformat() if split_train_end is not None else None,
        "valid_start": split_valid_start.isoformat() if split_valid_start is not None else None,
        "valid_end": split_valid_end.isoformat() if split_valid_end is not None else None,
        "purge_bars": int(purge_bars),
        "thresholds": [float(x) for x in thresholds],
        "sl_list": [float(x) for x in sl_list],
        "rr_list": [float(x) for x in rr_list],
        "min_tp": float(min_tp),
        "max_tp": float(max_tp),
        "execution_rules": dict(execution_rules or {}),
        "tp_pairs": [(float(sl), float(tp)) for sl, tp in tp_pairs],
        "task_keys": list(task_keys),
        "config_signature": dict(config_signature or {}),
    }
    msg = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(msg.encode("utf-8")).hexdigest()


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    run_key: str,
    records: List[Dict[str, Any]],
    mode: str,
    source_path: Path,
    out_path: Path,
    completed: int,
    total: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": dt.datetime.now(NY_TZ).isoformat(),
        "run_key": str(run_key),
        "mode": str(mode),
        "source": str(source_path),
        "out_path": str(out_path),
        "completed": int(completed),
        "total": int(total),
        "records": records,
    }
    tmp = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp.replace(checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    run_key: str,
    valid_task_keys: Set[str],
) -> Tuple[Set[str], List[Dict[str, object]], List[Dict[str, Any]]]:
    if not checkpoint_path.exists():
        return set(), [], []
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("DE3 v2 checkpoint read failed (%s): %s", checkpoint_path, exc)
        return set(), [], []
    if str(payload.get("run_key", "")) != str(run_key):
        logging.info("DE3 v2 checkpoint run key mismatch; ignoring checkpoint: %s", checkpoint_path)
        return set(), [], []

    done_keys: Set[str] = set()
    rows: List[Dict[str, object]] = []
    records: List[Dict[str, Any]] = []
    for rec in payload.get("records", []) or []:
        if not isinstance(rec, dict):
            continue
        task_key = str(rec.get("task_key", ""))
        if not task_key or task_key not in valid_task_keys:
            continue
        if task_key in done_keys:
            continue
        done_keys.add(task_key)
        row_obj = rec.get("row")
        if isinstance(row_obj, dict):
            rows.append(row_obj)
            records.append({"task_key": task_key, "row": row_obj})
        elif isinstance(row_obj, list):
            row_list = [x for x in row_obj if isinstance(x, dict)]
            if row_list:
                rows.extend(row_list)
                records.append({"task_key": task_key, "row": row_list})
            else:
                records.append({"task_key": task_key, "row": None})
        else:
            records.append({"task_key": task_key, "row": None})
    return done_keys, rows, records


def _filter_trade_entries_by_hour_window(
    trades: Dict[Tuple[str, str, float, str], List[tuple]],
    *,
    blocked_start_hour_et: int,
    blocked_end_hour_et: int,
) -> Tuple[Dict[Tuple[str, str, float, str], List[tuple]], int]:
    if not trades:
        return trades, 0
    staged: List[Tuple[Tuple[str, str, float, str], List[tuple], np.ndarray]] = []
    total = 0
    for key, items in trades.items():
        if not items:
            continue
        entry_ns = np.fromiter((int(it[3]) for it in items), dtype=np.int64, count=len(items))
        staged.append((key, items, entry_ns))
        total += int(len(entry_ns))
    if total <= 0:
        return {}, 0

    all_entry_ns = np.empty(total, dtype=np.int64)
    cur = 0
    for _, _, entry_ns in staged:
        n = len(entry_ns)
        all_entry_ns[cur : cur + n] = entry_ns
        cur += n

    unique_entry_ns, inverse = np.unique(all_entry_ns, return_inverse=True)
    entry_hours_unique = pd.to_datetime(unique_entry_ns, utc=True).tz_convert(NY_TZ).hour.to_numpy(dtype=np.int16, copy=False)

    start_h = int(blocked_start_hour_et) % 24
    end_h = int(blocked_end_hour_et) % 24
    if start_h == end_h:
        blocked_unique = np.zeros(len(entry_hours_unique), dtype=bool)
    elif start_h < end_h:
        blocked_unique = (entry_hours_unique >= start_h) & (entry_hours_unique < end_h)
    else:
        blocked_unique = (entry_hours_unique >= start_h) | (entry_hours_unique < end_h)

    out: Dict[Tuple[str, str, float, str], List[tuple]] = {}
    dropped = 0
    cur = 0
    for key, items, entry_ns in staged:
        n = len(entry_ns)
        idx = inverse[cur : cur + n]
        cur += n
        blocked = blocked_unique[idx]
        if not bool(np.any(blocked)):
            out[key] = items
            continue
        keep_mask = ~blocked
        kept = [it for it, keep in zip(items, keep_mask) if bool(keep)]
        dropped += int(len(items) - len(kept))
        if kept:
            out[key] = kept
    return out, dropped


def _cap_prepared_trade_end_to_force_flat(
    prepared: Dict[Tuple[str, str, float, str], List[tuple]],
    *,
    df_1m_index: np.ndarray,
    df_tf_index: np.ndarray,
    force_flat_hour_et: int,
) -> Tuple[Dict[Tuple[str, str, float, str], List[tuple]], int]:
    if not prepared:
        return prepared, 0

    out: Dict[Tuple[str, str, float, str], List[tuple]] = {}
    adjusted = 0
    force_flat_hour_et = int(force_flat_hour_et) % 24
    one_day_ns = int(pd.Timedelta(days=1).value)

    staged: List[
        Tuple[
            Tuple[str, str, float, str],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    total = 0
    for key, items in prepared.items():
        if not items:
            continue
        n = len(items)
        entry_pos_1m = np.fromiter((int(t[0]) for t in items), dtype=np.int64, count=n)
        entry_pos_tf = np.fromiter((int(t[1]) for t in items), dtype=np.int64, count=n)
        entry_price = np.fromiter((float(t[2]) for t in items), dtype=float, count=n)
        entry_time_ns = np.fromiter((int(t[3]) for t in items), dtype=np.int64, count=n)
        end_pos_1m = np.fromiter((int(t[4]) for t in items), dtype=np.int64, count=n)
        end_pos_tf = np.fromiter((int(t[5]) for t in items), dtype=np.int64, count=n)
        staged.append((key, entry_pos_1m, entry_pos_tf, entry_price, entry_time_ns, end_pos_1m, end_pos_tf))
        total += n
    if total <= 0:
        return out, 0

    all_entry_ns = np.empty(total, dtype=np.int64)
    cur = 0
    for _, _, _, _, entry_time_ns, _, _ in staged:
        n = len(entry_time_ns)
        all_entry_ns[cur : cur + n] = entry_time_ns
        cur += n

    unique_entry_ns, inverse = np.unique(all_entry_ns, return_inverse=True)
    entry_dt_et = pd.to_datetime(unique_entry_ns, utc=True).tz_convert(NY_TZ)
    force_dt_et = entry_dt_et.normalize() + pd.Timedelta(hours=force_flat_hour_et)
    force_ns = force_dt_et.asi8.copy()
    entry_ns_et = entry_dt_et.asi8
    force_ns = np.where(entry_ns_et >= force_ns, force_ns + one_day_ns, force_ns)
    force_dt64 = force_ns.astype("datetime64[ns]")
    force_end_1m_unique = np.searchsorted(df_1m_index, force_dt64, side="left") - 1
    force_end_tf_unique = np.searchsorted(df_tf_index, force_dt64, side="left") - 1

    cur = 0
    for key, entry_pos_1m, entry_pos_tf, entry_price, entry_time_ns, end_pos_1m, end_pos_tf in staged:
        n = len(entry_time_ns)
        map_idx = inverse[cur : cur + n]
        cur += n
        force_end_1m = force_end_1m_unique[map_idx]
        force_end_tf = force_end_tf_unique[map_idx]

        capped_end_1m = np.minimum(end_pos_1m, force_end_1m)
        capped_end_tf = np.minimum(end_pos_tf, force_end_tf)
        capped_end_1m = np.maximum(entry_pos_1m, capped_end_1m)
        capped_end_tf = np.maximum(entry_pos_tf, capped_end_tf)

        adjusted += int(np.count_nonzero(capped_end_1m != end_pos_1m))
        out[key] = [
            (
                int(entry_pos_1m[i]),
                int(entry_pos_tf[i]),
                float(entry_price[i]),
                int(entry_time_ns[i]),
                int(capped_end_1m[i]),
                int(capped_end_tf[i]),
            )
            for i in range(n)
        ]
    return out, adjusted


def _build_arrays(df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "open_1m": df_1m["open"].to_numpy(),
        "high_1m": df_1m["high"].to_numpy(),
        "low_1m": df_1m["low"].to_numpy(),
        "close_1m": df_1m["close"].to_numpy(),
        "open_5m": df_5m["open"].to_numpy(),
        "high_5m": df_5m["high"].to_numpy(),
        "low_5m": df_5m["low"].to_numpy(),
        "close_5m": df_5m["close"].to_numpy(),
        "open_15m": df_15m["open"].to_numpy(),
        "high_15m": df_15m["high"].to_numpy(),
        "low_15m": df_15m["low"].to_numpy(),
        "close_15m": df_15m["close"].to_numpy(),
    }


def _prepare_trade_metadata_fast(
    trades: Dict[Tuple[str, str, float, str], List[tuple]],
    *,
    df_1m_index: np.ndarray,
    df_tf_index: np.ndarray,
    max_horizon_1m: int,
    max_horizon_tf: int,
    limit_to_session: bool,
    label: str = "",
) -> Dict[Tuple[str, str, float, str], List[tuple]]:
    prepared: Dict[Tuple[str, str, float, str], List[tuple]] = {}
    total_items = sum(len(items) for items in trades.values())
    if total_items <= 0:
        return prepared

    processed = 0
    t0 = time.time()
    last_log = t0
    log_every = 200_000

    max_1m = len(df_1m_index) - 1
    max_tf = len(df_tf_index) - 1
    one_day_ns = int(pd.Timedelta(days=1).value)

    for key, items in trades.items():
        if not items:
            continue

        n = len(items)
        entry_pos_1m = np.fromiter((int(t[0]) for t in items), dtype=np.int64, count=n)
        entry_pos_tf = np.fromiter((int(t[1]) for t in items), dtype=np.int64, count=n)
        entry_price = np.fromiter((float(t[2]) for t in items), dtype=float, count=n)
        entry_time_ns = np.fromiter((int(t[3]) for t in items), dtype=np.int64, count=n)

        entry_pos_1m = np.clip(entry_pos_1m, 0, max_1m)
        entry_pos_tf = np.clip(entry_pos_tf, 0, max_tf)

        if limit_to_session:
            entry_dt_et = pd.to_datetime(entry_time_ns, utc=True).tz_convert(NY_TZ)
            hour = entry_dt_et.hour.to_numpy(dtype=np.int16, copy=False)
            session_end_hour = ((hour // 3) + 1) * 3
            day_offset = (session_end_hour >= 24).astype(np.int64)
            session_end_hour = (session_end_hour % 24).astype(np.int64)

            midnight_ns = entry_dt_et.normalize().asi8
            end_time_ns = midnight_ns + (day_offset * one_day_ns) + (session_end_hour * 3_600_000_000_000)
            end_dt64 = end_time_ns.astype("datetime64[ns]")

            session_end_1m = np.searchsorted(df_1m_index, end_dt64, side="left") - 1
            session_end_tf = np.searchsorted(df_tf_index, end_dt64, side="left") - 1
            session_end_1m = np.clip(session_end_1m, 0, max_1m)
            session_end_tf = np.clip(session_end_tf, 0, max_tf)

            end_pos_1m = np.minimum(entry_pos_1m + int(max_horizon_1m), session_end_1m)
            end_pos_tf = np.minimum(entry_pos_tf + int(max_horizon_tf), session_end_tf)
        else:
            end_pos_1m = np.minimum(entry_pos_1m + int(max_horizon_1m), max_1m)
            end_pos_tf = np.minimum(entry_pos_tf + int(max_horizon_tf), max_tf)

        end_pos_1m = np.maximum(entry_pos_1m, end_pos_1m)
        end_pos_tf = np.maximum(entry_pos_tf, end_pos_tf)

        prepared[key] = [
            (
                int(entry_pos_1m[i]),
                int(entry_pos_tf[i]),
                float(entry_price[i]),
                int(entry_time_ns[i]),
                int(end_pos_1m[i]),
                int(end_pos_tf[i]),
            )
            for i in range(n)
        ]

        processed += n
        now = time.time()
        if (processed % log_every) == 0 or (now - last_log) >= 30:
            elapsed = max(1e-9, now - t0)
            rate = float(processed) / elapsed
            remaining = max(0, int(total_items) - int(processed))
            eta_sec = (remaining / rate) if rate > 0 else 0.0
            logging.info(
                "Trade metadata %s: %s/%s | elapsed=%s | ETA=%s | %.0f rows/sec",
                label or "build",
                processed,
                total_items,
                _format_duration(elapsed),
                _format_duration(eta_sec),
                rate,
            )
            last_log = now

    logging.info(
        "Trade metadata %s complete: %s items in %.2fs",
        label or "build",
        total_items,
        time.time() - t0,
    )
    return prepared


def _select_arrays(
    arrays: Dict[str, np.ndarray],
    tf_label: str,
    trade_resolution: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if str(trade_resolution).lower() == "1m":
        return arrays["open_1m"], arrays["high_1m"], arrays["low_1m"], arrays["close_1m"]
    if tf_label == "5min":
        return arrays["open_5m"], arrays["high_5m"], arrays["low_5m"], arrays["close_5m"]
    return arrays["open_15m"], arrays["high_15m"], arrays["low_15m"], arrays["close_15m"]


def _split_items_fixed(
    items: Sequence[tuple],
    *,
    train_end_ns: int,
    valid_start_ns: int,
    valid_end_ns: Optional[int],
    purge_ns: int,
    entry_time_ns_arr: Optional[np.ndarray] = None,
    assume_sorted: bool = False,
) -> Tuple[List[tuple], List[tuple]]:
    if not items:
        return [], []

    train_cut = int(train_end_ns - purge_ns)
    valid_cut = int(valid_start_ns + purge_ns)

    if entry_time_ns_arr is None:
        entry_time_ns_arr = np.fromiter((int(it[3]) for it in items), dtype=np.int64, count=len(items))
    elif len(entry_time_ns_arr) != len(items):
        entry_time_ns_arr = np.fromiter((int(it[3]) for it in items), dtype=np.int64, count=len(items))

    sorted_ok = bool(assume_sorted)
    if not sorted_ok:
        sorted_ok = (len(entry_time_ns_arr) <= 1) or bool(np.all(entry_time_ns_arr[1:] >= entry_time_ns_arr[:-1]))
    if sorted_ok:
        train_right = int(np.searchsorted(entry_time_ns_arr, train_cut, side="right"))
        valid_left = int(np.searchsorted(entry_time_ns_arr, valid_cut, side="left"))
        if valid_end_ns is None:
            valid_right = len(items)
        else:
            valid_right = int(np.searchsorted(entry_time_ns_arr, int(valid_end_ns), side="right"))

        if isinstance(items, list):
            return items[:train_right], items[valid_left:valid_right]
        item_list = list(items)
        return item_list[:train_right], item_list[valid_left:valid_right]

    train_items = [it for it in items if int(it[3]) <= train_cut]
    if valid_end_ns is None:
        valid_items = [it for it in items if int(it[3]) >= valid_cut]
    else:
        valid_items = [it for it in items if valid_cut <= int(it[3]) <= int(valid_end_ns)]
    return train_items, valid_items


def _evaluate_train_valid_candidate(
    *,
    tf_label: str,
    session: str,
    stype: str,
    thresh: float,
    family_tag: str = "",
    family_profile: Optional[Dict[str, Any]] = None,
    train_items: Sequence[tuple],
    valid_items: Sequence[tuple],
    tp_pairs: Sequence[Tuple[float, float]],
    sl_values: Sequence[float],
    tp_values: Sequence[float],
    arrays: Dict[str, np.ndarray],
    trade_resolution: str,
    assume_sl_first: bool,
    sl_tp_conflict: str,
    exit_at_horizon: str,
    min_train_trades: int,
    min_oos_trades: int,
    plateau_cfg: Dict[str, object],
    scoring_cfg: Dict[str, float],
    selected_by: str = "plateau",
    acceleration: str = "cpu",
    exclude_combos: Optional[Set[Tuple[float, float]]] = None,
) -> Optional[Dict[str, object]]:
    if not train_items or not valid_items:
        return None

    selection_overrides = _family_selection_overrides(
        family_profile=family_profile,
        plateau_cfg=plateau_cfg,
        scoring_cfg=scoring_cfg,
        min_train_trades=min_train_trades,
        min_oos_trades=min_oos_trades,
    )
    effective_plateau_cfg = selection_overrides["plateau_cfg"]
    effective_scoring_cfg = selection_overrides["scoring_cfg"]
    effective_min_train_trades = int(selection_overrides["min_train_trades"])
    effective_min_oos_trades = int(selection_overrides["min_oos_trades"])
    fallback_to_best_score = bool(selection_overrides["fallback_to_best_score"])
    fallback_min_score = selection_overrides["fallback_min_score"]
    selected_mode = str(selected_by)

    side = "LONG" if stype.startswith("Long") else "SHORT"
    open_, high, low, close = _select_arrays(arrays, tf_label, trade_resolution)

    train_grid = evaluate_grid_metrics(
        train_items,
        tp_pairs,
        open_,
        high,
        low,
        close,
        side,
        trade_resolution=trade_resolution,
        assume_sl_first=assume_sl_first,
        sl_tp_conflict=sl_tp_conflict,
        exit_at_horizon=exit_at_horizon,
        min_trades_for_score=effective_min_train_trades,
        acceleration=acceleration,
    )
    if not train_grid:
        return None

    eligible = [
        row
        for row in train_grid
        if int(row.get("trades", 0) or 0) >= int(effective_min_train_trades)
    ]
    if exclude_combos:
        filtered: List[Dict[str, object]] = []
        for row in eligible:
            try:
                key = (float(row.get("sl", 0.0) or 0.0), float(row.get("tp", 0.0) or 0.0))
            except Exception:
                key = (0.0, 0.0)
            if key in exclude_combos:
                continue
            filtered.append(row)
        eligible = filtered
    if not eligible:
        return None

    chosen = select_plateau_candidate(
        eligible,
        sl_values,
        tp_values,
        min_neighbors=int(effective_plateau_cfg.get("min_neighbors", 4) or 4),
        neighbor_def=str(effective_plateau_cfg.get("neighbor_def", "adjacent_grid") or "adjacent_grid"),
        min_plateau_score=float(effective_plateau_cfg.get("min_plateau_score", 0.0) or 0.0),
    )
    if chosen is None and fallback_to_best_score:
        chosen = _select_best_candidate_fallback(
            eligible,
            min_score=fallback_min_score,
        )
        if chosen is not None:
            selected_mode = str(chosen.get("selected_by", "best_score_fallback") or "best_score_fallback")
    if chosen is None:
        return None

    sl_dist = float(chosen.get("sl", 0.0))
    tp_dist = float(chosen.get("tp", 0.0))
    valid_eval = evaluate_single_combo(
        valid_items,
        sl_dist,
        tp_dist,
        open_,
        high,
        low,
        close,
        side,
        trade_resolution=trade_resolution,
        assume_sl_first=assume_sl_first,
        sl_tp_conflict=sl_tp_conflict,
        exit_at_horizon=exit_at_horizon,
        acceleration=acceleration,
    )
    valid_summary = dict(valid_eval["summary"])
    if int(valid_summary.get("trades", 0) or 0) < int(effective_min_oos_trades):
        return None
    if float(valid_summary.get("avg_pnl", 0.0) or 0.0) <= 0.0:
        return None
    quality = _oos_trade_quality(valid_eval["pnls"], sl_dist=sl_dist, tp_dist=tp_dist)
    valid_pnls = np.asarray(valid_eval["pnls"], dtype=float)
    gross_win_valid = float(np.sum(valid_pnls[valid_pnls > 0.0])) if valid_pnls.size else 0.0
    gross_loss_valid = float(-np.sum(valid_pnls[valid_pnls < 0.0])) if valid_pnls.size else 0.0
    min_oos_win_rate_cfg = effective_scoring_cfg.get("min_oos_win_rate", None)
    if min_oos_win_rate_cfg is not None:
        try:
            min_oos_win_rate = float(min_oos_win_rate_cfg)
        except Exception:
            min_oos_win_rate = float("nan")
        if np.isfinite(min_oos_win_rate) and min_oos_win_rate > 0.0:
            if float(valid_summary.get("win_rate", 0.0) or 0.0) < float(min_oos_win_rate):
                return None
    max_oos_stop_share_cfg = effective_scoring_cfg.get("max_oos_stop_share", None)
    if max_oos_stop_share_cfg is not None:
        try:
            max_oos_stop_share = float(max_oos_stop_share_cfg)
        except Exception:
            max_oos_stop_share = float("nan")
        if np.isfinite(max_oos_stop_share) and max_oos_stop_share > 0.0:
            if float(quality.get("stop_like_share", 0.0) or 0.0) > float(max_oos_stop_share):
                return None
    max_oos_tail_p10_abs_cfg = effective_scoring_cfg.get("max_oos_tail_p10_abs", None)
    if max_oos_tail_p10_abs_cfg is not None:
        try:
            max_oos_tail_p10_abs = float(max_oos_tail_p10_abs_cfg)
        except Exception:
            max_oos_tail_p10_abs = float("nan")
        if np.isfinite(max_oos_tail_p10_abs) and max_oos_tail_p10_abs > 0.0:
            p10_loss_mag = max(0.0, -float(quality.get("pnl_p10", 0.0) or 0.0))
            if p10_loss_mag > float(max_oos_tail_p10_abs):
                return None

    block_stats = compute_oos_block_stats(
        valid_eval["entry_time_ns"],
        valid_eval["pnls"],
        block_freq="Q",
    )
    profitable_blocks = int(block_stats.get("positive_blocks", 0) or 0)
    total_blocks = int(len(block_stats.get("blocks", []) or []))
    min_oos_profitable_blocks = int(effective_scoring_cfg.get("min_oos_profitable_blocks", 0) or 0)
    if min_oos_profitable_blocks > 0 and profitable_blocks < min_oos_profitable_blocks:
        return None
    min_oos_profitable_block_ratio_cfg = effective_scoring_cfg.get("min_oos_profitable_block_ratio", None)
    if min_oos_profitable_block_ratio_cfg is not None and total_blocks > 0:
        try:
            min_oos_profitable_block_ratio = float(min_oos_profitable_block_ratio_cfg)
        except Exception:
            min_oos_profitable_block_ratio = float("nan")
        if (
            np.isfinite(min_oos_profitable_block_ratio)
            and min_oos_profitable_block_ratio > 0.0
            and (float(profitable_blocks) / float(total_blocks)) < float(min_oos_profitable_block_ratio)
        ):
            return None
    min_oos_profit_factor_cfg = effective_scoring_cfg.get("min_oos_profit_factor", None)
    if min_oos_profit_factor_cfg is not None:
        try:
            min_oos_profit_factor = float(min_oos_profit_factor_cfg)
        except Exception:
            min_oos_profit_factor = float("nan")
        if np.isfinite(min_oos_profit_factor) and min_oos_profit_factor > 0.0:
            if float(valid_summary.get("profit_factor", 0.0) or 0.0) < float(min_oos_profit_factor):
                return None
    mean_avg = float(block_stats.get("mean_avg_pnl", 0.0) or 0.0)
    std_avg = float(block_stats.get("std_avg_pnl", 0.0) or 0.0)
    max_dd_norm = float(block_stats.get("max_drawdown_norm", 0.0) or 0.0)
    max_oos_dd_norm_cfg = effective_scoring_cfg.get("max_oos_drawdown_norm", None)
    if max_oos_dd_norm_cfg is not None:
        try:
            max_oos_dd_norm_cap = float(max_oos_dd_norm_cfg)
        except Exception:
            max_oos_dd_norm_cap = float("nan")
        if np.isfinite(max_oos_dd_norm_cap) and max_oos_dd_norm_cap > 0.0:
            if float(max_dd_norm) > float(max_oos_dd_norm_cap):
                return None
    stability = stability_weighted_score(
        mean_avg,
        std_avg,
        max_dd_norm,
        lambda_std=float(effective_scoring_cfg.get("lambda_std", 0.7) or 0.7),
        gamma_dd=float(effective_scoring_cfg.get("gamma_dd", 0.3) or 0.3),
    )
    stop_like_share = float(quality.get("stop_like_share", 0.0) or 0.0)
    loss_share = float(quality.get("loss_share", 0.0) or 0.0)
    pnl_p10 = float(quality.get("pnl_p10", 0.0) or 0.0)
    tail_p10_loss_mag = max(0.0, -pnl_p10)
    gamma_stop_share = float(effective_scoring_cfg.get("gamma_stop_share", 0.0) or 0.0)
    gamma_loss_share = float(effective_scoring_cfg.get("gamma_loss_share", 0.0) or 0.0)
    gamma_tail_p10 = float(effective_scoring_cfg.get("gamma_tail_p10", 0.0) or 0.0)
    stability_penalty = (
        (gamma_stop_share * stop_like_share)
        + (gamma_loss_share * loss_share)
        + (gamma_tail_p10 * tail_p10_loss_mag)
    )
    stability -= float(stability_penalty)
    if not math.isfinite(stability):
        return None

    train_summary = {
        "trades": int(chosen.get("trades", 0) or 0),
        "wins": int(chosen.get("wins", 0) or 0),
        "win_rate": float(chosen.get("win_rate", 0.0) or 0.0),
        "avg_pnl": float(chosen.get("avg_pnl", 0.0) or 0.0),
        "profit_factor": float(chosen.get("profit_factor", 0.0) or 0.0),
        "total_pnl": float(chosen.get("total_pnl", 0.0) or 0.0),
        "score_train": float(chosen.get("score", 0.0) or 0.0),
    }

    record = {
        "TF": tf_label,
        "Session": session,
        "Type": stype,
        "Thresh": float(thresh),
        "Best_SL": float(sl_dist),
        "Best_TP": float(tp_dist),
        "Opt_WR": float(train_summary["win_rate"]),
        "Trades": int(train_summary["trades"]),
        "Avg_PnL": float(train_summary["avg_pnl"]),
        "Score": float(stability),
        "Recent": {
            "trades": int(valid_summary.get("trades", 0) or 0),
            "wins": int(valid_summary.get("wins", 0) or 0),
            "win_rate": float(valid_summary.get("win_rate", 0.0) or 0.0),
            "avg_pnl": float(valid_summary.get("avg_pnl", 0.0) or 0.0),
            "profit_factor": float(valid_summary.get("profit_factor", 0.0) or 0.0),
        },
        "Selected_by": str(selected_mode),
        "plateau_cluster_score": float(chosen.get("plateau_cluster_score", float("nan"))),
        "plateau_neighbors": int(chosen.get("plateau_neighbors", 0) or 0),
        "Train": train_summary,
        "OOS": {
            "trades": int(valid_summary.get("trades", 0) or 0),
            "wins": int(valid_summary.get("wins", 0) or 0),
            "win_rate": float(valid_summary.get("win_rate", 0.0) or 0.0),
            "avg_pnl": float(valid_summary.get("avg_pnl", 0.0) or 0.0),
            "profit_factor": float(valid_summary.get("profit_factor", 0.0) or 0.0),
            "total_pnl": float(valid_summary.get("total_pnl", 0.0) or 0.0),
            "gross_win": float(gross_win_valid),
            "gross_loss": float(gross_loss_valid),
            "stability_score": float(stability),
            "mean_oos_avg_pnl": float(mean_avg),
            "std_oos_avg_pnl": float(std_avg),
            "max_drawdown": float(block_stats.get("max_drawdown", 0.0) or 0.0),
            "max_oos_drawdown_norm": float(max_dd_norm),
            "sharpe_like": float(block_stats.get("sharpe_like", 0.0) or 0.0),
            "count_positive_blocks": int(profitable_blocks),
            "count_total_blocks": int(total_blocks),
            "count_stop_like": int(quality.get("count_stop_like", 0) or 0),
            "count_take_like": int(quality.get("count_take_like", 0) or 0),
            "stop_like_share": float(stop_like_share),
            "take_like_share": float(quality.get("take_like_share", 0.0) or 0.0),
            "loss_share": float(loss_share),
            "pnl_p10": float(pnl_p10),
            "pnl_p05": float(quality.get("pnl_p05", 0.0) or 0.0),
            "pnl_p01": float(quality.get("pnl_p01", 0.0) or 0.0),
            "stability_penalty": float(stability_penalty),
        },
        "oos_blocks_stats": block_stats,
    }
    record.update(
        _build_runtime_strategy_fields(
            tf_label=tf_label,
            session=session,
            stype=stype,
            family_tag=family_tag,
            thresh=float(thresh),
            sl=float(sl_dist),
            tp=float(tp_dist),
            family_profile=family_profile,
        )
    )
    return record


def _build_rolling_windows(
    index: pd.DatetimeIndex,
    *,
    train_years: int,
    valid_years: int,
    step_years: int,
) -> List[Dict[str, pd.Timestamp]]:
    if index.empty:
        return []
    start = pd.Timestamp(index.min()).tz_convert("US/Eastern")
    end = pd.Timestamp(index.max()).tz_convert("US/Eastern")

    windows: List[Dict[str, pd.Timestamp]] = []
    cur = pd.Timestamp(start.year, 1, 1, tz="US/Eastern")
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(years=int(train_years)) - pd.Timedelta(minutes=1)
        valid_start = train_end + pd.Timedelta(minutes=1)
        valid_end = valid_start + pd.DateOffset(years=int(valid_years)) - pd.Timedelta(minutes=1)
        if valid_start > end:
            break
        valid_end = min(valid_end, end)
        if train_start < end and train_end > start and valid_start <= valid_end:
            windows.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                }
            )
        cur = cur + pd.DateOffset(years=int(step_years))
        if cur > end:
            break
    return windows


def _evaluate_candidate_rolling(
    *,
    tf_label: str,
    session: str,
    stype: str,
    thresh: float,
    family_tag: str = "",
    family_profile: Optional[Dict[str, Any]] = None,
    items: Sequence[tuple],
    windows: Sequence[Dict[str, pd.Timestamp]],
    purge_ns: int,
    tp_pairs: Sequence[Tuple[float, float]],
    sl_values: Sequence[float],
    tp_values: Sequence[float],
    arrays: Dict[str, np.ndarray],
    trade_resolution: str,
    assume_sl_first: bool,
    sl_tp_conflict: str,
    exit_at_horizon: str,
    min_train_trades: int,
    min_oos_trades: int,
    min_profitable_blocks: int,
    plateau_cfg: Dict[str, object],
    scoring_cfg: Dict[str, float],
    acceleration: str = "cpu",
) -> Optional[Dict[str, object]]:
    if not items:
        return None

    selection_overrides = _family_selection_overrides(
        family_profile=family_profile,
        plateau_cfg=plateau_cfg,
        scoring_cfg=scoring_cfg,
        min_train_trades=min_train_trades,
        min_oos_trades=min_oos_trades,
    )
    effective_scoring_cfg = selection_overrides["scoring_cfg"]
    effective_min_profitable_blocks = _profile_override_int(
        family_profile,
        "selection_min_profitable_blocks",
        min_profitable_blocks,
        min_value=0,
    )

    entry_time_ns = np.fromiter((int(it[3]) for it in items), dtype=np.int64, count=len(items))
    if len(entry_time_ns) > 1 and not bool(np.all(entry_time_ns[1:] >= entry_time_ns[:-1])):
        order = np.argsort(entry_time_ns, kind="stable")
        items_sorted = [items[int(i)] for i in order]
        entry_time_ns_sorted = entry_time_ns[order]
    else:
        items_sorted = items if isinstance(items, list) else list(items)
        entry_time_ns_sorted = entry_time_ns

    fold_records: List[Dict[str, object]] = []
    combo_counts: Counter = Counter()
    combo_fold_scores: Dict[Tuple[float, float], List[float]] = defaultdict(list)

    for fold in windows:
        train_end_ns = int(pd.Timestamp(fold["train_end"]).value)
        valid_start_ns = int(pd.Timestamp(fold["valid_start"]).value)
        valid_end_ns = int(pd.Timestamp(fold["valid_end"]).value)
        train_items, valid_items = _split_items_fixed(
            items_sorted,
            train_end_ns=train_end_ns,
            valid_start_ns=valid_start_ns,
            valid_end_ns=valid_end_ns,
            purge_ns=purge_ns,
            entry_time_ns_arr=entry_time_ns_sorted,
            assume_sorted=True,
        )
        rec = _evaluate_train_valid_candidate(
            tf_label=tf_label,
            session=session,
            stype=stype,
            thresh=thresh,
            family_tag=family_tag,
            family_profile=family_profile,
            train_items=train_items,
            valid_items=valid_items,
            tp_pairs=tp_pairs,
            sl_values=sl_values,
            tp_values=tp_values,
            arrays=arrays,
            trade_resolution=trade_resolution,
            assume_sl_first=assume_sl_first,
            sl_tp_conflict=sl_tp_conflict,
            exit_at_horizon=exit_at_horizon,
            min_train_trades=min_train_trades,
            min_oos_trades=min_oos_trades,
            plateau_cfg=plateau_cfg,
            scoring_cfg=scoring_cfg,
            selected_by="plateau",
            acceleration=acceleration,
        )
        if rec is None:
            continue
        fold_rec = {
            "fold": {
                "train_start": pd.Timestamp(fold["train_start"]).isoformat(),
                "train_end": pd.Timestamp(fold["train_end"]).isoformat(),
                "valid_start": pd.Timestamp(fold["valid_start"]).isoformat(),
                "valid_end": pd.Timestamp(fold["valid_end"]).isoformat(),
            },
            "Best_SL": float(rec["Best_SL"]),
            "Best_TP": float(rec["Best_TP"]),
            "OOS": rec["OOS"],
            "Train": rec["Train"],
            "Score": float(rec["Score"]),
        }
        fold_records.append(fold_rec)
        combo_key = (float(rec["Best_SL"]), float(rec["Best_TP"]))
        combo_counts[combo_key] += 1
        combo_fold_scores[combo_key].append(float(rec["Score"]))

    if not fold_records:
        return None

    profitable_blocks = int(sum(1 for fr in fold_records if float(fr["OOS"]["total_pnl"]) > 0.0))
    if profitable_blocks < int(effective_min_profitable_blocks):
        return None

    if combo_counts:
        def _combo_rank(item: Tuple[Tuple[float, float], int]) -> Tuple[int, float]:
            key, cnt = item
            mean_score = float(np.mean(combo_fold_scores.get(key, [0.0])))
            return (cnt, mean_score)

        final_combo = sorted(combo_counts.items(), key=_combo_rank, reverse=True)[0][0]
    else:
        final_combo = (float(fold_records[-1]["Best_SL"]), float(fold_records[-1]["Best_TP"]))

    fold_avg = np.asarray([float(fr["OOS"]["avg_pnl"]) for fr in fold_records], dtype=float)
    fold_dd_norm = np.asarray([float(fr["OOS"]["max_oos_drawdown_norm"]) for fr in fold_records], dtype=float)
    fold_wr = np.asarray([float(fr["OOS"].get("win_rate", 0.0) or 0.0) for fr in fold_records], dtype=float)
    fold_stop_share = np.asarray([float(fr["OOS"].get("stop_like_share", 0.0) or 0.0) for fr in fold_records], dtype=float)
    fold_loss_share = np.asarray([float(fr["OOS"].get("loss_share", 0.0) or 0.0) for fr in fold_records], dtype=float)
    fold_p10 = np.asarray([float(fr["OOS"].get("pnl_p10", 0.0) or 0.0) for fr in fold_records], dtype=float)
    mean_avg = float(np.mean(fold_avg)) if len(fold_avg) else 0.0
    std_avg = float(np.std(fold_avg)) if len(fold_avg) else 0.0
    worst_dd_norm = float(np.max(fold_dd_norm)) if len(fold_dd_norm) else 0.0
    mean_wr = float(np.mean(fold_wr)) if len(fold_wr) else 0.0
    mean_stop_share = float(np.mean(fold_stop_share)) if len(fold_stop_share) else 0.0
    mean_loss_share = float(np.mean(fold_loss_share)) if len(fold_loss_share) else 0.0
    worst_p10 = float(np.min(fold_p10)) if len(fold_p10) else 0.0
    mean_p10 = float(np.mean(fold_p10)) if len(fold_p10) else 0.0
    min_oos_win_rate_cfg = effective_scoring_cfg.get("min_oos_win_rate", None)
    if min_oos_win_rate_cfg is not None:
        try:
            min_oos_win_rate = float(min_oos_win_rate_cfg)
        except Exception:
            min_oos_win_rate = float("nan")
        if np.isfinite(min_oos_win_rate) and min_oos_win_rate > 0.0:
            if float(mean_wr) < float(min_oos_win_rate):
                return None
    max_oos_stop_share_cfg = effective_scoring_cfg.get("max_oos_stop_share", None)
    if max_oos_stop_share_cfg is not None:
        try:
            max_oos_stop_share = float(max_oos_stop_share_cfg)
        except Exception:
            max_oos_stop_share = float("nan")
        if np.isfinite(max_oos_stop_share) and max_oos_stop_share > 0.0:
            if float(mean_stop_share) > float(max_oos_stop_share):
                return None
    max_oos_tail_p10_abs_cfg = effective_scoring_cfg.get("max_oos_tail_p10_abs", None)
    if max_oos_tail_p10_abs_cfg is not None:
        try:
            max_oos_tail_p10_abs = float(max_oos_tail_p10_abs_cfg)
        except Exception:
            max_oos_tail_p10_abs = float("nan")
        if np.isfinite(max_oos_tail_p10_abs) and max_oos_tail_p10_abs > 0.0:
            worst_p10_loss_mag = max(0.0, -float(worst_p10))
            if worst_p10_loss_mag > float(max_oos_tail_p10_abs):
                return None
    max_oos_dd_norm_cfg = effective_scoring_cfg.get("max_oos_drawdown_norm", None)
    if max_oos_dd_norm_cfg is not None:
        try:
            max_oos_dd_norm_cap = float(max_oos_dd_norm_cfg)
        except Exception:
            max_oos_dd_norm_cap = float("nan")
        if np.isfinite(max_oos_dd_norm_cap) and max_oos_dd_norm_cap > 0.0:
            if float(worst_dd_norm) > float(max_oos_dd_norm_cap):
                return None
    stability = stability_weighted_score(
        mean_avg,
        std_avg,
        worst_dd_norm,
        lambda_std=float(effective_scoring_cfg.get("lambda_std", 0.7) or 0.7),
        gamma_dd=float(effective_scoring_cfg.get("gamma_dd", 0.3) or 0.3),
    )
    mean_p10_loss_mag = max(0.0, -float(mean_p10))
    gamma_stop_share = float(effective_scoring_cfg.get("gamma_stop_share", 0.0) or 0.0)
    gamma_loss_share = float(effective_scoring_cfg.get("gamma_loss_share", 0.0) or 0.0)
    gamma_tail_p10 = float(effective_scoring_cfg.get("gamma_tail_p10", 0.0) or 0.0)
    stability_penalty = (
        (gamma_stop_share * float(mean_stop_share))
        + (gamma_loss_share * float(mean_loss_share))
        + (gamma_tail_p10 * float(mean_p10_loss_mag))
    )
    stability -= float(stability_penalty)
    if not math.isfinite(stability):
        return None

    total_trades = int(sum(int(fr["OOS"]["trades"]) for fr in fold_records))
    total_wins = int(sum(int(fr["OOS"]["wins"]) for fr in fold_records))
    total_pnl = float(sum(float(fr["OOS"]["total_pnl"]) for fr in fold_records))
    total_stop_like = int(sum(int(fr["OOS"].get("count_stop_like", 0) or 0) for fr in fold_records))
    total_take_like = int(sum(int(fr["OOS"].get("count_take_like", 0) or 0) for fr in fold_records))
    gross_win = 0.0
    gross_loss = 0.0
    fold_blocks: List[Dict[str, object]] = []
    max_fold_drawdown = 0.0
    for fr in fold_records:
        oos_fold = fr["OOS"]
        gross_win += float(oos_fold.get("gross_win", 0.0) or 0.0)
        gross_loss += float(oos_fold.get("gross_loss", 0.0) or 0.0)
        max_fold_drawdown = max(max_fold_drawdown, float(oos_fold.get("max_drawdown", 0.0) or 0.0))
        fold_meta = fr.get("fold", {}) if isinstance(fr.get("fold"), dict) else {}
        valid_start_text = str(fold_meta.get("valid_start", "") or "")
        valid_end_text = str(fold_meta.get("valid_end", "") or "")
        block_label = valid_start_text[:10]
        if valid_start_text and valid_end_text:
            try:
                start_year = pd.Timestamp(valid_start_text).year
                end_year = pd.Timestamp(valid_end_text).year
                block_label = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
            except Exception:
                block_label = valid_start_text[:10]
        fold_blocks.append(
            {
                "block": str(block_label),
                "trades": int(oos_fold.get("trades", 0) or 0),
                "wins": int(oos_fold.get("wins", 0) or 0),
                "win_rate": float(oos_fold.get("win_rate", 0.0) or 0.0),
                "avg_pnl": float(oos_fold.get("avg_pnl", 0.0) or 0.0),
                "total_pnl": float(oos_fold.get("total_pnl", 0.0) or 0.0),
                "profit_factor": float(oos_fold.get("profit_factor", 0.0) or 0.0),
            }
        )

    agg_pf = _safe_profit_factor(gross_win, gross_loss)
    agg_wr = (total_wins / total_trades) if total_trades else 0.0
    agg_avg = (total_pnl / total_trades) if total_trades else 0.0
    agg_stop_share = (float(total_stop_like) / float(total_trades)) if total_trades else 0.0
    agg_take_share = (float(total_take_like) / float(total_trades)) if total_trades else 0.0

    record = {
        "TF": tf_label,
        "Session": session,
        "Type": stype,
        "Thresh": float(thresh),
        "Best_SL": float(final_combo[0]),
        "Best_TP": float(final_combo[1]),
        "Opt_WR": float(agg_wr),
        "Trades": int(total_trades),
        "Avg_PnL": float(agg_avg),
        "Score": float(stability),
        "Recent": {
            "trades": int(total_trades),
            "wins": int(total_wins),
            "win_rate": float(agg_wr),
            "avg_pnl": float(agg_avg),
            "profit_factor": float(agg_pf),
        },
        "Selected_by": "plateau",
        "Train": {
            "folds": len(fold_records),
        },
        "OOS": {
            "trades": int(total_trades),
            "wins": int(total_wins),
            "win_rate": float(agg_wr),
            "avg_pnl": float(agg_avg),
            "profit_factor": float(agg_pf),
            "total_pnl": float(total_pnl),
            "gross_win": float(gross_win),
            "gross_loss": float(gross_loss),
            "stability_score": float(stability),
            "mean_oos_avg_pnl": float(mean_avg),
            "std_oos_avg_pnl": float(std_avg),
            "max_drawdown": float(max_fold_drawdown),
            "max_oos_drawdown_norm": float(worst_dd_norm),
            "sharpe_like": float((mean_avg / (std_avg + 1e-12)) if len(fold_avg) > 1 else 0.0),
            "count_positive_blocks": int(profitable_blocks),
            "count_total_blocks": int(len(fold_records)),
            "count_stop_like": int(total_stop_like),
            "count_take_like": int(total_take_like),
            "stop_like_share": float(agg_stop_share),
            "take_like_share": float(agg_take_share),
            "loss_share": float(mean_loss_share),
            "pnl_p10": float(mean_p10),
            "pnl_p10_worst": float(worst_p10),
            "stability_penalty": float(stability_penalty),
        },
        "oos_blocks_stats": {
            "mode": "rolling",
            "folds": fold_records,
            "blocks": fold_blocks,
            "mean_avg_pnl": float(mean_avg),
            "std_avg_pnl": float(std_avg),
            "count_positive_blocks": int(profitable_blocks),
            "positive_blocks": int(profitable_blocks),
            "max_drawdown": float(max_fold_drawdown),
            "max_drawdown_norm": float(worst_dd_norm),
            "sharpe_like": float((mean_avg / (std_avg + 1e-12)) if len(fold_avg) > 1 else 0.0),
        },
    }
    record.update(
        _build_runtime_strategy_fields(
            tf_label=tf_label,
            session=session,
            stype=stype,
            family_tag=family_tag,
            thresh=float(thresh),
            sl=float(final_combo[0]),
            tp=float(final_combo[1]),
            family_profile=family_profile,
        )
    )
    return record


def _write_validation_report(
    out_path: Path,
    all_candidates: Sequence[Dict[str, object]],
    final_strategies: Sequence[Dict[str, object]],
) -> Tuple[Path, Path]:
    report_json = out_path.with_name(f"{out_path.stem}_validation_report.json")
    report_csv = out_path.with_name(f"{out_path.stem}_validation_report.csv")

    final_list = list(final_strategies)
    oos_avg = [float((r.get("OOS") or {}).get("avg_pnl", 0.0) or 0.0) for r in final_list]
    oos_pf = [float((r.get("OOS") or {}).get("profit_factor", 0.0) or 0.0) for r in final_list]
    oos_sharpe = [float((r.get("OOS") or {}).get("sharpe_like", 0.0) or 0.0) for r in final_list]
    scores = [float(r.get("Score", 0.0) or 0.0) for r in final_list]
    structural_scores = [float(r.get("StructuralScore", r.get("Score", 0.0)) or 0.0) for r in final_list]

    def _dist(vals: Sequence[float]) -> Dict[str, float]:
        if not vals:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size <= 0:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    top = sorted(final_list, key=lambda x: float(x.get("Score", 0.0) or 0.0), reverse=True)[:25]
    report = {
        "generated_at": dt.datetime.now(NY_TZ).isoformat(),
        "total_candidates_considered": int(len(all_candidates)),
        "kept_strategies": int(len(final_list)),
        "distributions": {
            "oos_avg_pnl": _dist(oos_avg),
            "oos_profit_factor": _dist(oos_pf),
            "oos_sharpe_like": _dist(oos_sharpe),
            "stability_score": _dist(scores),
            "structural_score": _dist(structural_scores),
        },
        "top_strategies_by_stability": [
            {
                "TF": r.get("TF"),
                "Session": r.get("Session"),
                "Type": r.get("Type"),
                "FamilyTag": r.get("FamilyTag"),
                "strategy_id": r.get("strategy_id"),
                "Thresh": r.get("Thresh"),
                "Best_SL": r.get("Best_SL"),
                "Best_TP": r.get("Best_TP"),
                "Score": r.get("Score"),
                "StructuralScore": r.get("StructuralScore"),
                "StructuralPass": r.get("StructuralPass"),
                "ProfitableBlockRatio": r.get("ProfitableBlockRatio"),
                "WorstBlockAvgPnL": r.get("WorstBlockAvgPnL"),
                "WorstBlockPF": r.get("WorstBlockPF"),
                "BlockAvgPnLStd": r.get("BlockAvgPnLStd"),
                "TailP10Scaled": r.get("TailP10Scaled"),
                "OOS": r.get("OOS"),
            }
            for r in top
        ],
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    rows = []
    for r in final_list:
        o = r.get("OOS") or {}
        rows.append(
            {
                "TF": r.get("TF"),
                "Session": r.get("Session"),
                "Type": r.get("Type"),
                "FamilyTag": r.get("FamilyTag"),
                "strategy_id": r.get("strategy_id"),
                "Thresh": r.get("Thresh"),
                "Best_SL": r.get("Best_SL"),
                "Best_TP": r.get("Best_TP"),
                "Score": r.get("Score"),
                "StructuralScore": r.get("StructuralScore"),
                "StructuralPass": r.get("StructuralPass"),
                "OOS_profitable_block_ratio": r.get("ProfitableBlockRatio"),
                "OOS_worst_block_avg_pnl": r.get("WorstBlockAvgPnL"),
                "OOS_worst_block_pf": r.get("WorstBlockPF"),
                "OOS_block_avg_pnl_std": r.get("BlockAvgPnLStd"),
                "OOS_tail_p10_scaled": r.get("TailP10Scaled"),
                "OOS_trades": o.get("trades"),
                "OOS_win_rate": o.get("win_rate"),
                "OOS_avg_pnl": o.get("avg_pnl"),
                "OOS_profit_factor": o.get("profit_factor"),
                "OOS_sharpe_like": o.get("sharpe_like"),
                "OOS_max_dd_norm": o.get("max_oos_drawdown_norm"),
                "OOS_stop_like_share": o.get("stop_like_share"),
                "OOS_take_like_share": o.get("take_like_share"),
                "OOS_loss_share": o.get("loss_share"),
                "OOS_pnl_p10": o.get("pnl_p10"),
                "OOS_pnl_p10_worst": o.get("pnl_p10_worst"),
                "OOS_stability_penalty": o.get("stability_penalty"),
            }
        )
    pd.DataFrame(rows).to_csv(report_csv, index=False)
    return report_json, report_csv


def _default_v2_config() -> Dict[str, object]:
    return {
        "enabled": False,
        "db_path": "dynamic_engine3_strategies_v2.json",
        "mode": "fixed_split",
        "train_end": "2024-12-31",
        "valid_start": "2025-01-01",
        "valid_end": None,
        "purge_bars": 200,
        "plateau": {
            "enabled": True,
            "min_neighbors": 4,
            "neighbor_def": "adjacent_grid",
            "min_plateau_score": 0.0,
        },
        "scoring": {
            "lambda_std": 0.7,
            "gamma_dd": 0.3,
            "min_oos_trades": 50,
            "min_profitable_blocks": 2,
            "min_train_trades": 30,
            # Fixed-split OOS consistency and edge-quality filters.
            "min_oos_profitable_blocks": 0,
            "min_oos_profitable_block_ratio": None,
            "min_oos_profit_factor": None,
            "min_oos_win_rate": None,
            "max_oos_stop_share": None,
            "max_oos_tail_p10_abs": None,
            # Optional hard cap on OOS drawdown-normalized tail risk.
            # Candidates exceeding this are rejected before final selection.
            "max_oos_drawdown_norm": None,
            # Soft tail-risk penalties (subtracted from stability score).
            "gamma_stop_share": 0.0,
            "gamma_loss_share": 0.0,
            "gamma_tail_p10": 0.0,
            # Optional structural pruning: within same TF/session/type + bracket,
            # drop higher-threshold near-duplicates when a lower-threshold candidate
            # has similar edge but better shape/coverage.
            "dominance_pruning_enabled": False,
            "dominance_avg_pnl_tolerance": 0.10,
            "dominance_score_tolerance": 0.50,
            "dominance_dd_tolerance": 0.05,
            "dominance_require_lower_or_equal_thresh": True,
        },
        "robust_ranking": {
            "enabled": True,
            "min_oos_trades": 80,
            "min_profitable_block_ratio": 0.60,
            "min_worst_block_avg_pnl": -0.25,
            "min_worst_block_pf": 0.90,
            "max_oos_drawdown_norm": 0.80,
            "max_stop_like_share": 0.50,
            "max_loss_share": 0.65,
            "max_tail_p10_abs_sl_mult": 1.00,
            "trade_conf_tau": 100,
            "weights": {
                "avg_pnl": 1.50,
                "profit_factor": 1.00,
                "win_rate": 0.50,
                "trade_confidence": 0.60,
                "profitable_block_ratio": 1.00,
                "worst_block_avg_pnl": 0.85,
                "worst_block_pf": 0.35,
                "drawdown_norm": -1.10,
                "stop_like_share": -0.90,
                "loss_share": -0.70,
                "tail_p10": -0.80,
                "block_std": -0.80,
                "sharpe_like": 0.20,
            },
            "runtime_weights": {
                "edge_points": 0.35,
                "edge_gap": 0.20,
                "structural_score": 0.30,
                "bucket_score": 0.10,
                "confidence": 0.05,
                "ambiguity_penalty": -0.15,
                "concentration_penalty": -0.10,
            },
            "runtime_abstain": {
                "enabled": True,
                "min_edge_points": 0.16,
                "min_edge_gap_points": 0.12,
                "min_structural_score": -1.00,
                "min_runtime_rank_score": 0.08,
            },
            "log_top_k": 3,
            "log_decisions": True,
        },
        "search_space": {
            "thresholds": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            "sl_list": [3, 4, 5, 6, 8, 10, 12, 15],
            "rr_list": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
            "max_per_bucket": 6,
        },
        "diversity": {
            "enabled": False,
            # Soft penalties reduce repeated SL/TP usage during final top-N selection.
            # These are score-space penalties, not hard blocks.
            "combo_penalty": 0.08,
            "sl_penalty": 0.02,
            "tp_penalty": 0.02,
            "global_scale": 0.5,
            # Fixed-split only: evaluate up to N unique plateau candidates per task
            # and let final selection apply diversity scoring across these options.
            "candidates_per_task": 1,
            # Keep alternates near the top plateau quality to avoid weak tails.
            "max_cluster_drop": 0.30,
        },
        "family_profiles": {
            # Experimental additive families are off by default so the legacy
            # v2 rebuild path remains unchanged unless explicitly enabled.
            "enabled": False,
            "profiles": [],
            # When enabled, family-tagged rows are selected in their own final
            # TF/session/type bucket instead of competing directly with base rows.
            "separate_selection_buckets": False,
        },
        "rolling": {
            "train_years": 5,
            "valid_years": 1,
            "step_years": 1,
        },
        "checkpoint": {
            "enabled": False,
            "path": None,
            "every_tasks": 25,
            "resume": True,
            "delete_on_success": False,
        },
        "symbol_mode": "auto_by_day",
        "symbol_method": "volume",
        "workers": 1,
        "acceleration": "cpu",  # cpu | gpu | auto
        "trade_resolution": "1m",
        "max_horizon": 180,
        "limit_to_session": True,
        "exit_at_horizon": "close",
        "assume_sl_first": False,
        "sl_tp_conflict": "ohlc",
        "execution": {
            "enforce_no_new_entries_window": True,
            "no_new_entries_start_hour_et": 16,
            "no_new_entries_end_hour_et": 18,
            "force_flat_at_hour_enabled": True,
            "force_flat_hour_et": 16,
        },
        "min_tp": 4.0,
        "max_tp": 30.0,
    }


def _merge_v2_config() -> Dict[str, object]:
    base = _default_v2_config()
    user = CONFIG.get("DE3_V2", {}) or {}

    def _deep_merge(dst: Dict[str, object], src: Dict[str, object]) -> Dict[str, object]:
        out = dict(dst)
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
            else:
                out[k] = v
        return out

    return _deep_merge(base, user)


def generate_de3_v2(
    *,
    source_path: Path,
    out_path: Path,
    cfg: Dict[str, object],
    cache_dir: Optional[Path],
    use_cache: bool,
    checkpoint_path_override: Optional[str] = None,
    checkpoint_every_override: Optional[int] = None,
    checkpoint_resume_override: Optional[bool] = None,
    checkpoint_enabled_override: Optional[bool] = None,
) -> Dict[str, object]:
    t_start = time.time()
    mode = str(cfg.get("mode", "fixed_split") or "fixed_split").lower()
    if mode not in {"fixed_split", "rolling"}:
        raise ValueError(f"Unsupported DE3 v2 mode: {mode}")

    logging.info("Stage: load bars")
    df = data_cache.load_bars(source_path, cache_dir=cache_dir, use_cache=use_cache)
    if df.empty:
        raise RuntimeError("No bars loaded from source.")

    symbol_mode = str(cfg.get("symbol_mode", "auto_by_day") or "auto_by_day")
    symbol_method = str(cfg.get("symbol_method", "volume") or "volume")
    logging.info("Applying symbol mode: %s (%s)", symbol_mode, symbol_method)
    df, symbol_mode_actual, symbol_map = bt.apply_symbol_mode(df, symbol_mode, symbol_method)
    if df.empty:
        raise RuntimeError("No bars left after symbol mode.")

    split_train_end = _parse_date(str(cfg.get("train_end")) if cfg.get("train_end") is not None else None, is_end=True)
    split_valid_start = _parse_date(str(cfg.get("valid_start")) if cfg.get("valid_start") is not None else None, is_end=False)
    split_valid_end = _parse_date(str(cfg.get("valid_end")) if cfg.get("valid_end") is not None else None, is_end=True)
    if split_train_end is None or split_valid_start is None:
        raise ValueError("DE3 v2 fixed split requires train_end and valid_start.")
    if split_valid_end is not None and split_valid_end <= split_valid_start:
        raise ValueError("valid_end must be greater than valid_start.")

    data_start = None
    data_end = split_valid_end
    df = _filter_range(df, data_start, data_end)

    search_space = cfg.get("search_space", {}) or {}
    thresholds = [float(x) for x in (search_space.get("thresholds") or [])]
    sl_list = [float(x) for x in (search_space.get("sl_list") or [])]
    rr_list = [float(x) for x in (search_space.get("rr_list") or [])]
    max_per_bucket = int(search_space.get("max_per_bucket", 6) or 6)
    diversity_cfg = cfg.get("diversity", {}) or {}

    min_tp = float(cfg.get("min_tp", 4.0) or 4.0)
    max_tp = float(cfg.get("max_tp", 30.0) or 30.0)
    tp_pairs = _build_tp_pairs(sl_list, rr_list, min_tp=min_tp, max_tp=max_tp, tick_size=TICK_SIZE)
    if not thresholds or not tp_pairs:
        raise RuntimeError("Empty thresholds or SL/TP search space.")
    execution_cfg = cfg.get("execution", {}) or {}
    enforce_no_new_entries_window = bool(execution_cfg.get("enforce_no_new_entries_window", True))
    no_new_entries_start_hour_et = int(execution_cfg.get("no_new_entries_start_hour_et", 16) or 16)
    no_new_entries_end_hour_et = int(execution_cfg.get("no_new_entries_end_hour_et", 18) or 18)
    force_flat_at_hour_enabled = bool(execution_cfg.get("force_flat_at_hour_enabled", True))
    force_flat_hour_et = int(execution_cfg.get("force_flat_hour_et", 16) or 16)
    execution_rules = {
        "enforce_no_new_entries_window": bool(enforce_no_new_entries_window),
        "no_new_entries_start_hour_et": int(no_new_entries_start_hour_et),
        "no_new_entries_end_hour_et": int(no_new_entries_end_hour_et),
        "force_flat_at_hour_enabled": bool(force_flat_at_hour_enabled),
        "force_flat_hour_et": int(force_flat_hour_et),
    }

    sl_values = sorted({float(sl) for sl, _ in tp_pairs})
    tp_values = sorted({float(tp) for _, tp in tp_pairs})
    family_profiles = _resolve_family_profiles(cfg)
    preserve_family_buckets = bool((cfg.get("family_profiles", {}) or {}).get("separate_selection_buckets", False))
    family_profile_map = {
        str(profile.get("family_tag", "") or "").strip(): dict(profile)
        for profile in family_profiles
        if str(profile.get("family_tag", "") or "").strip()
    }

    cache_key = _build_resample_cache_key(source_path, df, symbol_mode_actual, symbol_method)
    df_5m, df_15m = _load_or_build_resamples(
        df,
        cache_dir=cache_dir,
        cache_key=cache_key,
        use_cache=use_cache,
    )
    if df_5m.empty or df_15m.empty:
        raise RuntimeError("Resampled bars are empty.")

    index_1m = df.index.values.astype("datetime64[ns]")
    logging.info("Building candidate trade lists...")
    trades_5m = _build_trades_with_family_profiles(
        df_5m,
        index_1m,
        thresholds,
        family_profiles=family_profiles,
        label="de3v2-5m",
    )
    trades_15m = _build_trades_with_family_profiles(
        df_15m,
        index_1m,
        thresholds,
        family_profiles=family_profiles,
        label="de3v2-15m",
    )
    if enforce_no_new_entries_window:
        trades_5m, dropped_5m = _filter_trade_entries_by_hour_window(
            trades_5m,
            blocked_start_hour_et=no_new_entries_start_hour_et,
            blocked_end_hour_et=no_new_entries_end_hour_et,
        )
        trades_15m, dropped_15m = _filter_trade_entries_by_hour_window(
            trades_15m,
            blocked_start_hour_et=no_new_entries_start_hour_et,
            blocked_end_hour_et=no_new_entries_end_hour_et,
        )
        logging.info(
            "DE3 v2 entry window filter applied [%02d:00-%02d:00 ET): dropped %s 5m + %s 15m candidates",
            int(no_new_entries_start_hour_et) % 24,
            int(no_new_entries_end_hour_et) % 24,
            int(dropped_5m),
            int(dropped_15m),
        )

    max_horizon = int(cfg.get("max_horizon", 180) or 180)
    max_horizon_1m = max(1, int(max_horizon))
    max_horizon_5m = max(1, int(math.ceil(max_horizon / 5)))
    max_horizon_15m = max(1, int(math.ceil(max_horizon / 15)))
    limit_to_session = bool(cfg.get("limit_to_session", True))

    prepared_5m = _prepare_trade_metadata_fast(
        trades_5m,
        df_1m_index=index_1m,
        df_tf_index=df_5m.index.values.astype("datetime64[ns]"),
        max_horizon_1m=max_horizon_1m,
        max_horizon_tf=max_horizon_5m,
        limit_to_session=limit_to_session,
        label="de3v2-5m",
    )
    prepared_15m = _prepare_trade_metadata_fast(
        trades_15m,
        df_1m_index=index_1m,
        df_tf_index=df_15m.index.values.astype("datetime64[ns]"),
        max_horizon_1m=max_horizon_1m,
        max_horizon_tf=max_horizon_15m,
        limit_to_session=limit_to_session,
        label="de3v2-15m",
    )
    if force_flat_at_hour_enabled:
        prepared_5m, adjusted_5m = _cap_prepared_trade_end_to_force_flat(
            prepared_5m,
            df_1m_index=index_1m,
            df_tf_index=df_5m.index.values.astype("datetime64[ns]"),
            force_flat_hour_et=force_flat_hour_et,
        )
        prepared_15m, adjusted_15m = _cap_prepared_trade_end_to_force_flat(
            prepared_15m,
            df_1m_index=index_1m,
            df_tf_index=df_15m.index.values.astype("datetime64[ns]"),
            force_flat_hour_et=force_flat_hour_et,
        )
        logging.info(
            "DE3 v2 force-flat cap applied @%02d:00 ET: adjusted %s 5m + %s 15m horizons",
            int(force_flat_hour_et) % 24,
            int(adjusted_5m),
            int(adjusted_15m),
        )
    tasks = _build_tasks(prepared_5m, prepared_15m)
    logging.info("Total DE3 v2 candidate buckets: %s", len(tasks))
    task_keys_all = [_task_key(task) for task in tasks]

    arrays = _build_arrays(df, df_5m, df_15m)
    trade_resolution = str(cfg.get("trade_resolution", "1m") or "1m")
    assume_sl_first = bool(cfg.get("assume_sl_first", False))
    sl_tp_conflict = str(
        cfg.get(
            "sl_tp_conflict",
            ((CONFIG.get("BACKTEST_EXECUTION", {}) or {}).get("sl_tp_conflict", "ohlc")),
        )
        or "ohlc"
    ).lower()
    exit_at_horizon = str(cfg.get("exit_at_horizon", "close") or "close")
    purge_bars = int(cfg.get("purge_bars", 200) or 200)
    purge_ns = int(pd.Timedelta(minutes=purge_bars).value)

    plateau_cfg = cfg.get("plateau", {}) or {}
    scoring_cfg = cfg.get("scoring", {}) or {}
    robust_ranking_cfg = cfg.get("robust_ranking", {}) or {}
    min_train_trades = int(scoring_cfg.get("min_train_trades", 30) or 30)
    min_oos_trades = int(scoring_cfg.get("min_oos_trades", 50) or 50)
    min_profitable_blocks = int(scoring_cfg.get("min_profitable_blocks", 2) or 2)
    try:
        diversity_candidates_per_task = int((diversity_cfg or {}).get("candidates_per_task", 1) or 1)
    except Exception:
        diversity_candidates_per_task = 1
    diversity_candidates_per_task = max(1, min(4, diversity_candidates_per_task))
    try:
        diversity_max_cluster_drop = float((diversity_cfg or {}).get("max_cluster_drop", 0.30) or 0.30)
    except Exception:
        diversity_max_cluster_drop = 0.30

    acceleration = str(cfg.get("acceleration", "cpu") or "cpu").lower()
    if acceleration not in {"cpu", "gpu", "auto"}:
        acceleration = "cpu"
    if acceleration == "auto":
        acceleration = "gpu" if gpu_backend_available() else "cpu"
    if acceleration == "gpu" and not gpu_backend_available():
        logging.warning("DE3 v2 GPU acceleration requested but CuPy is not available; falling back to CPU.")
        acceleration = "cpu"

    workers_cfg = int(cfg.get("workers", 1) or 1)
    if workers_cfg <= 0:
        workers_cfg = int(os.cpu_count() or 1)
    workers = max(1, workers_cfg)
    if acceleration == "gpu" and workers > 1:
        # Multiple workers each contending for one GPU tends to slow this path.
        workers = 1
    logging.info("DE3 v2 execution: mode=%s acceleration=%s workers=%s", mode, acceleration, workers)

    cp_cfg = cfg.get("checkpoint", {}) or {}
    checkpoint_enabled = bool(cp_cfg.get("enabled", False))
    if checkpoint_enabled_override is not None:
        checkpoint_enabled = bool(checkpoint_enabled_override)
    checkpoint_every = int(cp_cfg.get("every_tasks", 25) or 25)
    if checkpoint_every_override is not None:
        checkpoint_every = max(1, int(checkpoint_every_override))
    checkpoint_resume = bool(cp_cfg.get("resume", True))
    if checkpoint_resume_override is not None:
        checkpoint_resume = bool(checkpoint_resume_override)
    checkpoint_delete_on_success = bool(cp_cfg.get("delete_on_success", False))
    checkpoint_path = _resolve_checkpoint_path(out_path, cfg, checkpoint_path_override)
    config_signature = {
        "trade_resolution": str(trade_resolution),
        "assume_sl_first": bool(assume_sl_first),
        "sl_tp_conflict": str(sl_tp_conflict),
        "exit_at_horizon": str(exit_at_horizon),
        "min_train_trades": int(min_train_trades),
        "min_oos_trades": int(min_oos_trades),
        "min_profitable_blocks": int(min_profitable_blocks),
        "max_per_bucket": int(max_per_bucket),
        "plateau": dict(plateau_cfg or {}),
        "scoring": dict(scoring_cfg or {}),
        "robust_ranking": dict(robust_ranking_cfg or {}),
        "diversity": dict(diversity_cfg or {}),
        "family_profiles": dict(cfg.get("family_profiles", {}) or {}),
        "acceleration": str(acceleration),
        "workers": int(workers),
    }
    run_key = _build_resume_run_key(
        source_path=source_path,
        mode=mode,
        cache_key=cache_key,
        split_train_end=split_train_end,
        split_valid_start=split_valid_start,
        split_valid_end=split_valid_end,
        purge_bars=purge_bars,
        thresholds=thresholds,
        sl_list=sl_list,
        rr_list=rr_list,
        min_tp=min_tp,
        max_tp=max_tp,
        execution_rules=execution_rules,
        tp_pairs=tp_pairs,
        task_keys=task_keys_all,
        config_signature=config_signature,
    )

    done_keys: Set[str] = set()
    checkpoint_records: List[Dict[str, Any]] = []
    rows: List[Dict[str, object]] = []
    if checkpoint_enabled and checkpoint_resume:
        done_keys, rows, checkpoint_records = _load_checkpoint(
            checkpoint_path,
            run_key=run_key,
            valid_task_keys=set(task_keys_all),
        )
        if done_keys:
            logging.info(
                "Resuming from checkpoint: %s completed tasks, %s rows restored (%s)",
                len(done_keys),
                len(rows),
                checkpoint_path,
            )
    pending_tasks = [task for task in tasks if _task_key(task) not in done_keys]

    total = len(tasks)
    eval_start = time.time()
    last_log = eval_start
    done = len(done_keys)

    def _checkpoint_save(done_count: int) -> None:
        if not checkpoint_enabled:
            return
        _save_checkpoint(
            checkpoint_path,
            run_key=run_key,
            records=checkpoint_records,
            mode=mode,
            source_path=source_path,
            out_path=out_path,
            completed=done_count,
            total=total,
        )

    def _record_completed(
        task: Tuple[str, str, str, float, str, List[tuple]],
        rec: Optional[object],
        done_count: int,
    ) -> None:
        row_payload: Optional[object] = None
        if isinstance(rec, dict):
            rows.append(rec)
            row_payload = rec
        elif isinstance(rec, list):
            row_list = [x for x in rec if isinstance(x, dict)]
            if row_list:
                rows.extend(row_list)
                row_payload = row_list
            else:
                row_payload = None
        if checkpoint_enabled:
            checkpoint_records.append({"task_key": _task_key(task), "row": row_payload})
            if done_count == total or (checkpoint_every > 0 and (done_count % checkpoint_every) == 0):
                _checkpoint_save(done_count)

    def _log_progress(done: int, force: bool = False) -> float:
        nonlocal last_log
        now = time.time()
        if not force and done != total and (now - last_log) < 15:
            return now
        elapsed = now - eval_start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = ((total - done) / rate) if rate > 0 else 0.0
        logging.info(
            "Progress: %s/%s (%.1f%%) | elapsed=%s | ETA=%s",
            done,
            total,
            (done / total * 100.0) if total else 100.0,
            _format_duration(elapsed),
            _format_duration(eta),
        )
        last_log = now
        return now

    if done > 0:
        _log_progress(done, force=True)

    if mode == "rolling":
        rolling_cfg = cfg.get("rolling", {}) or {}
        windows = _build_rolling_windows(
            df.index,
            train_years=int(rolling_cfg.get("train_years", 5) or 5),
            valid_years=int(rolling_cfg.get("valid_years", 1) or 1),
            step_years=int(rolling_cfg.get("step_years", 1) or 1),
        )
        if not windows:
            raise RuntimeError("No rolling windows available for DE3 v2 rolling mode.")
        logging.info("Rolling mode windows: %s", len(windows))

        def _rolling_task(task: Tuple[str, str, str, float, str, List[tuple]]) -> Optional[Dict[str, object]]:
            tf_label, session, stype, thresh, family_tag, items = task
            family_profile = family_profile_map.get(str(family_tag or "").strip(), None)
            return _evaluate_candidate_rolling(
                tf_label=tf_label,
                session=session,
                stype=stype,
                thresh=thresh,
                family_tag=str(family_tag or ""),
                family_profile=family_profile,
                items=items,
                windows=windows,
                purge_ns=purge_ns,
                tp_pairs=tp_pairs,
                sl_values=sl_values,
                tp_values=tp_values,
                arrays=arrays,
                trade_resolution=trade_resolution,
                assume_sl_first=assume_sl_first,
                sl_tp_conflict=sl_tp_conflict,
                exit_at_horizon=exit_at_horizon,
                min_train_trades=min_train_trades,
                min_oos_trades=min_oos_trades,
                min_profitable_blocks=min_profitable_blocks,
                plateau_cfg=plateau_cfg,
                scoring_cfg=scoring_cfg,
                acceleration=acceleration,
            )

        if workers > 1 and len(pending_tasks) > 1:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_rolling_task, task): task for task in pending_tasks}
                for fut in cf.as_completed(futs):
                    task = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception:
                        tf_label, session, stype, thresh, family_tag, _ = task
                        logging.exception(
                            "DE3 v2 rolling task failed: TF=%s Session=%s Type=%s Thresh=%s Family=%s",
                            tf_label,
                            session,
                            stype,
                            thresh,
                            family_tag or "base",
                        )
                        rec = None
                    done += 1
                    _record_completed(task, rec, done)
                    _log_progress(done, force=(done == total))
        else:
            for task in pending_tasks:
                try:
                    rec = _rolling_task(task)
                except Exception:
                    _checkpoint_save(done)
                    raise
                done += 1
                _record_completed(task, rec, done)
                _log_progress(done, force=(done == total))
    else:
        train_end_ns = int(split_train_end.value)
        valid_start_ns = int(split_valid_start.value)
        valid_end_ns = int(split_valid_end.value) if split_valid_end is not None else None

        def _fixed_task(task: Tuple[str, str, str, float, str, List[tuple]]) -> Optional[object]:
            tf_label, session, stype, thresh, family_tag, items = task
            family_profile = family_profile_map.get(str(family_tag or "").strip(), None)
            train_items, valid_items = _split_items_fixed(
                items,
                train_end_ns=train_end_ns,
                valid_start_ns=valid_start_ns,
                valid_end_ns=valid_end_ns,
                purge_ns=purge_ns,
            )
            if not train_items or not valid_items:
                return None

            records: List[Dict[str, object]] = []
            excluded_combos: Set[Tuple[float, float]] = set()
            best_cluster: Optional[float] = None

            for idx in range(int(diversity_candidates_per_task)):
                rec = _evaluate_train_valid_candidate(
                    tf_label=tf_label,
                    session=session,
                    stype=stype,
                    thresh=thresh,
                    family_tag=str(family_tag or ""),
                    family_profile=family_profile,
                    train_items=train_items,
                    valid_items=valid_items,
                    tp_pairs=tp_pairs,
                    sl_values=sl_values,
                    tp_values=tp_values,
                    arrays=arrays,
                    trade_resolution=trade_resolution,
                    assume_sl_first=assume_sl_first,
                    sl_tp_conflict=sl_tp_conflict,
                    exit_at_horizon=exit_at_horizon,
                    min_train_trades=min_train_trades,
                    min_oos_trades=min_oos_trades,
                    plateau_cfg=plateau_cfg,
                    scoring_cfg=scoring_cfg,
                    selected_by="plateau",
                    acceleration=acceleration,
                    exclude_combos=excluded_combos,
                )
                if rec is None:
                    break

                try:
                    cluster_score = float(rec.get("plateau_cluster_score", float("nan")))
                except Exception:
                    cluster_score = float("nan")
                if idx == 0:
                    best_cluster = cluster_score
                elif (
                    best_cluster is not None
                    and math.isfinite(best_cluster)
                    and math.isfinite(cluster_score)
                    and cluster_score < (best_cluster - float(diversity_max_cluster_drop))
                ):
                    break

                rec = dict(rec)
                rec["diversity_task_rank"] = int(idx + 1)
                records.append(rec)
                excluded_combos.add((float(rec["Best_SL"]), float(rec["Best_TP"])))

            if not records:
                return None
            if len(records) == 1:
                return records[0]
            return records

        if workers > 1 and len(pending_tasks) > 1:
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_fixed_task, task): task for task in pending_tasks}
                for fut in cf.as_completed(futs):
                    task = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception:
                        tf_label, session, stype, thresh, family_tag, _ = task
                        logging.exception(
                            "DE3 v2 fixed task failed: TF=%s Session=%s Type=%s Thresh=%s Family=%s",
                            tf_label,
                            session,
                            stype,
                            thresh,
                            family_tag or "base",
                        )
                        rec = None
                    done += 1
                    _record_completed(task, rec, done)
                    _log_progress(done, force=(done == total))
        else:
            for task in pending_tasks:
                try:
                    rec = _fixed_task(task)
                except Exception:
                    _checkpoint_save(done)
                    raise
                done += 1
                _record_completed(task, rec, done)
                _log_progress(done, force=(done == total))

    if total > 0:
        _log_progress(done, force=True)

    rows = _prune_dominated_rows(
        rows,
        scoring_cfg=scoring_cfg,
        preserve_family_buckets=preserve_family_buckets,
    )
    if rows:
        for row in rows:
            try:
                row.update(compute_structural_rank_fields(row, robust_ranking_cfg))
            except Exception:
                # Keep generator resilient; fallback to legacy score if structural
                # diagnostics fail for any single row.
                row["StructuralScore"] = float(row.get("Score", 0.0) or 0.0)
                row["StructuralPass"] = True
                row["StructuralTrustReason"] = "structural_compute_failed"

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["TF"]),
                str(row["Session"]),
                str(row["Type"]),
                (
                    _normalize_family_tag(row.get("FamilyTag", row.get("family_tag", "")))
                    if preserve_family_buckets
                    else ""
                ),
            )
        ].append(row)

    final_strategies: List[Dict[str, object]] = []
    global_sl_counts: Counter = Counter()
    global_tp_counts: Counter = Counter()
    global_combo_counts: Counter = Counter()
    for key in sorted(grouped.keys()):
        items = grouped[key]
        items.sort(
            key=lambda x: (
                float(x.get("StructuralScore", float("-inf")) or float("-inf")),
                float((x.get("OOS") or {}).get("avg_pnl", 0.0) or 0.0),
                float((x.get("OOS") or {}).get("profit_factor", 0.0) or 0.0),
                float((x.get("OOS") or {}).get("win_rate", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected_items = _soft_diversity_select(
            items,
            max_keep=max_per_bucket,
            cfg=diversity_cfg,
            global_sl_counts=global_sl_counts,
            global_tp_counts=global_tp_counts,
            global_combo_counts=global_combo_counts,
        )
        final_strategies.extend(selected_items)

    payload = {
        "version": "v2",
        "generated_at": dt.datetime.now(NY_TZ).isoformat(),
        "source_csv": str(source_path),
        "symbol_mode": symbol_mode_actual,
        "symbol_map_size": len(symbol_map) if symbol_map else 0,
        "date_range": {
            "start": df.index.min().isoformat() if not df.empty else None,
            "end": df.index.max().isoformat() if not df.empty else None,
        },
        "settings": {
            "thresholds": [float(x) for x in thresholds],
            "sl_list": [float(x) for x in sl_list],
            "rr_list": [float(x) for x in rr_list],
            "min_tp": float(min_tp),
            "max_tp": float(max_tp),
            "min_trades": int(min_oos_trades),
            "min_win_rate": 0.0,
            "min_avg_pnl": 0.0,
            "recent_start": str(split_valid_start.date()) if split_valid_start is not None else None,
            "recent_end": str(split_valid_end.date()) if split_valid_end is not None else None,
            "recent_mode": "validation_only",
            "folds": None,
            "min_fold_trades": None,
            "min_fold_win_rate": None,
            "min_fold_avg_pnl": None,
            "min_positive_folds": None,
            "min_positive_ratio": None,
            "loro": False,
            "min_regime_trades": None,
            "min_regime_avg_pnl": None,
            "min_positive_regimes": None,
            "max_per_bucket": int(max_per_bucket),
            "max_horizon": int(max_horizon),
            "limit_to_session": bool(limit_to_session),
            "exit_at_horizon": str(exit_at_horizon),
            "assume_sl_first": bool(assume_sl_first),
            "sl_tp_conflict": str(sl_tp_conflict),
            "trade_resolution": str(trade_resolution),
            "workers": int(workers),
            "acceleration": str(acceleration),
            "mode": str(mode),
            "execution": execution_rules,
            "plateau": plateau_cfg,
            "scoring": scoring_cfg,
            "robust_ranking": robust_ranking_cfg,
            "diversity": diversity_cfg,
        },
        "split": {
            "mode": mode,
            "train_end": split_train_end.isoformat() if split_train_end is not None else None,
            "valid_start": split_valid_start.isoformat() if split_valid_start is not None else None,
            "valid_end": split_valid_end.isoformat() if split_valid_end is not None else None,
            "purge_bars": int(purge_bars),
        },
        "strategies": final_strategies,
        "summary": {
            "total_candidates": len(rows),
            "total_strategies": len(final_strategies),
            "elapsed_seconds": float(time.time() - t_start),
        },
    }

    _checkpoint_save(done)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    report_json, report_csv = _write_validation_report(out_path, rows, final_strategies)
    logging.info("Wrote DE3 v2 DB: %s", out_path)
    logging.info("Wrote DE3 v2 validation report: %s", report_json)
    logging.info("Wrote DE3 v2 validation csv: %s", report_csv)
    if checkpoint_enabled and checkpoint_delete_on_success and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logging.info("Removed DE3 v2 checkpoint after success: %s", checkpoint_path)
        except Exception as exc:
            logging.warning("Failed to remove DE3 v2 checkpoint (%s): %s", checkpoint_path, exc)
    logging.info("Total elapsed: %s", _format_duration(time.time() - t_start))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DynamicEngine3 v2 strategy DB.")
    parser.add_argument("--source", "--csv", dest="source", default="es_master_outrights.parquet", help="Source 1m bars CSV/parquet.")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to CONFIG['DE3_V2']['db_path']).")
    parser.add_argument("--mode", choices=("fixed_split", "rolling"), default=None, help="Override DE3 v2 mode.")
    parser.add_argument("--workers", type=int, default=None, help="Override DE3 v2 worker count (<=0 => auto CPU cores).")
    parser.add_argument(
        "--acceleration",
        choices=("cpu", "gpu", "auto"),
        default=None,
        help="Execution backend for scoring loops. GPU requires CuPy.",
    )
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for bars/resamples.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    parser.add_argument("--checkpoint", action="store_true", help="Enable checkpoint/resume file writes.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Force-disable checkpointing.")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint JSON path (defaults near output file).")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Write checkpoint every N completed tasks.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint and start fresh.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    cfg = _merge_v2_config()
    if args.mode:
        cfg["mode"] = args.mode
    if args.workers is not None:
        cfg["workers"] = int(args.workers)
    if args.acceleration is not None:
        cfg["acceleration"] = str(args.acceleration)

    source = Path(args.source)
    if not source.exists():
        source = Path(__file__).resolve().parent / source
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    out_value = args.out if args.out is not None else str(cfg.get("db_path", "dynamic_engine3_strategies_v2.json"))
    out_path = Path(out_value)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    checkpoint_enabled_override = None
    if args.checkpoint and args.no_checkpoint:
        raise ValueError("Cannot set both --checkpoint and --no-checkpoint.")
    if args.checkpoint:
        checkpoint_enabled_override = True
    elif args.no_checkpoint:
        checkpoint_enabled_override = False
    checkpoint_resume_override = None
    if args.resume and args.no_resume:
        raise ValueError("Cannot set both --resume and --no-resume.")
    if args.resume:
        checkpoint_resume_override = True
    elif args.no_resume:
        checkpoint_resume_override = False

    generate_de3_v2(
        source_path=source,
        out_path=out_path,
        cfg=cfg,
        cache_dir=cache_dir,
        use_cache=not bool(args.no_cache),
        checkpoint_path_override=args.checkpoint_path,
        checkpoint_every_override=args.checkpoint_every,
        checkpoint_resume_override=checkpoint_resume_override,
        checkpoint_enabled_override=checkpoint_enabled_override,
    )


if __name__ == "__main__":
    main()

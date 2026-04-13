import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import CONFIG
from de3_v4_schema import build_family_id, safe_float, safe_int


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_FIT_TRADES_PATH = "reports/de3_decisions_fresh_current_live_2011_2024_trade_attribution.csv"
DEFAULT_OOS_TRADES_PATH = "reports/de3_decisions_2025_current_live_trade_attribution.csv"
DEFAULT_SCOPE_PRIORITY = [
    "exact",
    "regime_conf",
    "regime_substate",
    "session_substate",
]
_SUB_STRATEGY_RE = re.compile(
    r"^(?P<tf>[^_]+)_(?P<session>\d{2}-\d{2})_(?P<typ>.+?)_T(?P<thresh>-?\d+(?:\.\d+)?)_SL(?P<sl>-?\d+(?:\.\d+)?)_TP(?P<tp>-?\d+(?:\.\d+)?)$"
)


def _resolve_path(raw_path: Any) -> Path:
    path = Path(str(raw_path or "").strip())
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _norm_pair(sl: Any, tp: Any) -> Optional[Tuple[float, float]]:
    sl_val = round(safe_float(sl, 0.0), 4)
    tp_val = round(safe_float(tp, 0.0), 4)
    if sl_val <= 0.0 or tp_val <= 0.0:
        return None
    return (sl_val, tp_val)


def _pair_key(pair: Tuple[float, float]) -> str:
    return f"SL{pair[0]:g}_TP{pair[1]:g}"


def _split_type_and_tag(type_text: str) -> Tuple[str, str]:
    text = str(type_text or "").strip()
    for prefix in ("Long_Rev", "Short_Rev", "Long_Mom", "Short_Mom"):
        if text == prefix:
            return prefix, ""
        if text.startswith(prefix + "_"):
            return prefix, text[len(prefix) + 1 :]
    return text, ""


def _parse_sub_strategy(sub_strategy: Any) -> Optional[Dict[str, Any]]:
    text = str(sub_strategy or "").strip()
    if not text:
        return None
    match = _SUB_STRATEGY_RE.match(text)
    if not match:
        return None
    tf = str(match.group("tf") or "").strip()
    session = str(match.group("session") or "").strip()
    full_type = str(match.group("typ") or "").strip()
    strategy_type, family_tag = _split_type_and_tag(full_type)
    threshold = safe_float(match.group("thresh"), 0.0)
    pair = _norm_pair(match.group("sl"), match.group("tp"))
    if pair is None:
        return None
    family_id = build_family_id(
        timeframe=tf,
        session=session,
        strategy_type=strategy_type,
        threshold=threshold,
        family_tag=family_tag,
    )
    return {
        "sub_strategy": text,
        "timeframe": tf,
        "session": session,
        "strategy_type": strategy_type,
        "family_tag": family_tag,
        "threshold": float(threshold),
        "family_id": str(family_id),
        "pair": pair,
    }


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, closed="left", label="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0)
    return out


def _augment_5m_features(
    df_5m: pd.DataFrame,
    *,
    atr_period: int,
    atr_median_window: int,
    price_loc_window: int,
) -> pd.DataFrame:
    if df_5m is None or df_5m.empty:
        return df_5m
    out = df_5m.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(max(2, atr_period)), adjust=False).mean()
    atr_med = atr.rolling(max(2, atr_median_window), min_periods=max(2, atr_median_window)).median()
    high_roll = out["high"].rolling(max(2, price_loc_window), min_periods=max(2, price_loc_window)).max()
    low_roll = out["low"].rolling(max(2, price_loc_window), min_periods=max(2, price_loc_window)).min()
    denom = (high_roll - low_roll).replace(0.0, np.nan)
    out["atr_5m"] = atr
    out["atr_5m_median"] = atr_med
    out["price_location"] = ((out["close"] - low_roll) / denom).clip(lower=0.0, upper=1.0)
    return out


def _session_substate(ts: pd.Timestamp, session: str) -> str:
    try:
        hour_et = int(pd.Timestamp(ts).tz_convert(NY_TZ).hour)
    except Exception:
        hour_et = int(pd.Timestamp(ts).hour)
    try:
        start_hour = int(str(session or "").split("-", 1)[0])
    except Exception:
        return ""
    rel_hour = (hour_et - start_hour) % 24
    if rel_hour < 1:
        return "open"
    if rel_hour < 2:
        return "mid"
    return "late"


def _context_bucket_from_runtime_fields(context: Dict[str, Any]) -> Dict[str, str]:
    vol = str(context.get("volatility_regime", "normal") or "normal").strip().lower()
    comp = str(context.get("compression_expansion_regime", "neutral") or "neutral").strip().lower()
    conf = str(context.get("confidence_band", "mid") or "mid").strip().lower()
    substate = str(context.get("session_substate", "") or "").strip().lower()
    if not substate:
        substate = "unknown"
    return {
        "volatility_regime": vol,
        "compression_expansion_regime": comp,
        "confidence_band": conf,
        "session_substate": substate,
    }


def _compute_context_for_time(
    df_5m: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    session: str,
) -> Dict[str, Any]:
    default = {
        "volatility_regime": "normal",
        "compression_expansion_regime": "neutral",
        "confidence_band": "mid",
        "session_substate": _session_substate(decision_time, session),
    }
    if df_5m is None or df_5m.empty:
        return default
    ts = pd.Timestamp(decision_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    idx = df_5m.index
    try:
        pos = int(idx.searchsorted(ts, side="right")) - 1
    except Exception:
        return default
    if pos < 0 or pos >= len(df_5m):
        return default
    row = df_5m.iloc[pos]
    atr_5m = safe_float(row.get("atr_5m"), np.nan)
    atr_med = safe_float(row.get("atr_5m_median"), np.nan)
    price_loc = safe_float(row.get("price_location"), 0.5)
    if not np.isfinite(atr_5m) or atr_5m <= 0.0 or not np.isfinite(atr_med) or atr_med <= 0.0:
        return default
    atr_ratio = float(atr_5m / atr_med)
    if atr_ratio >= 1.10:
        vol = "high"
        comp = "expanding"
    elif atr_ratio <= 0.90:
        vol = "low"
        comp = "compressed"
    else:
        vol = "normal"
        comp = "neutral"
    if price_loc >= 0.70:
        conf = "high"
    elif price_loc <= 0.30:
        conf = "low"
    else:
        conf = "mid"
    return {
        "volatility_regime": str(vol),
        "compression_expansion_regime": str(comp),
        "confidence_band": str(conf),
        "session_substate": _session_substate(ts, session),
        "atr_5m": float(atr_5m),
        "atr_5m_median": float(atr_med),
        "atr_ratio": float(atr_ratio),
        "price_location": float(price_loc),
    }


def _stats_from_points(points: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in points if np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "trades": 0.0,
            "net_points": 0.0,
            "avg_points": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
    gross_pos = float(arr[arr > 0.0].sum())
    gross_neg = float(-arr[arr < 0.0].sum())
    if gross_neg <= 1e-12:
        pf = 999.0 if gross_pos > 0.0 else 0.0
    else:
        pf = float(gross_pos / gross_neg)
    return {
        "trades": float(arr.size),
        "net_points": float(arr.sum()),
        "avg_points": float(arr.mean()),
        "win_rate": float(np.mean(arr > 0.0)),
        "profit_factor": float(pf),
    }


def _score_option(stats: Dict[str, float]) -> float:
    trades = max(1.0, float(stats.get("trades", 0.0) or 0.0))
    reliability = min(1.0, trades / 80.0)
    avg_points = float(stats.get("avg_points", 0.0) or 0.0)
    pf = float(stats.get("profit_factor", 0.0) or 0.0)
    win_rate = float(stats.get("win_rate", 0.0) or 0.0)
    pf_term = min(2.5, max(-1.0, pf - 1.0))
    return float(avg_points + (0.30 * reliability * pf_term) + (0.20 * reliability * (win_rate - 0.5)))


def _scope_key(scope: str, context: Dict[str, Any]) -> str:
    buckets = _context_bucket_from_runtime_fields(context)
    vol = buckets["volatility_regime"]
    comp = buckets["compression_expansion_regime"]
    conf = buckets["confidence_band"]
    substate = buckets["session_substate"]
    if scope == "exact":
        return f"{vol}|{comp}|{conf}|{substate}"
    if scope == "regime_conf":
        return f"{vol}|{comp}|{conf}"
    if scope == "regime_substate":
        return f"{vol}|{comp}|{substate}"
    if scope == "session_substate":
        return str(substate)
    return ""


def _simulate_option_points(
    *,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    entry_pos: int,
    exit_pos: int,
    side: str,
    entry_price: float,
    sl: float,
    tp: float,
    mfe: Optional[float],
    mae: Optional[float],
) -> float:
    side_norm = str(side or "").strip().upper()
    mfe_val = safe_float(mfe, np.nan)
    mae_val = safe_float(mae, np.nan)
    tp_possible = bool(np.isfinite(mfe_val) and mfe_val >= tp - 1e-9)
    sl_possible = bool(np.isfinite(mae_val) and mae_val >= sl - 1e-9)

    if tp_possible and not sl_possible:
        return float(tp)
    if sl_possible and not tp_possible:
        return float(-sl)
    if not tp_possible and not sl_possible:
        exit_price = float(close_arr[exit_pos])
        if side_norm == "SHORT":
            return float(entry_price - exit_price)
        return float(exit_price - entry_price)

    for pos in range(entry_pos, exit_pos + 1):
        hi = float(high_arr[pos])
        lo = float(low_arr[pos])
        if side_norm == "SHORT":
            hit_tp = lo <= entry_price - tp
            hit_sl = hi >= entry_price + sl
        else:
            hit_tp = hi >= entry_price + tp
            hit_sl = lo <= entry_price - sl
        if hit_tp and hit_sl:
            is_green = float(close_arr[pos]) >= float(open_arr[pos])
            stop_first = bool(is_green) if side_norm != "SHORT" else bool(not is_green)
            return float(-sl) if stop_first else float(tp)
        if hit_tp:
            return float(tp)
        if hit_sl:
            return float(-sl)

    exit_price = float(close_arr[exit_pos])
    if side_norm == "SHORT":
        return float(entry_price - exit_price)
    return float(exit_price - entry_price)


def _prepare_trade_rows(
    *,
    csv_path: Path,
    family_options: Dict[str, List[Tuple[float, float]]],
    bars_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if not csv_path.exists() or bars_1m.empty:
        return []
    idx = bars_1m.index
    open_arr = bars_1m["open"].to_numpy(dtype=float)
    high_arr = bars_1m["high"].to_numpy(dtype=float)
    low_arr = bars_1m["low"].to_numpy(dtype=float)
    close_arr = bars_1m["close"].to_numpy(dtype=float)
    prepared: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = _parse_sub_strategy(row.get("sub_strategy"))
            if not parsed:
                continue
            family_id = str(parsed.get("family_id", "") or "")
            options = family_options.get(family_id)
            if not options or len(options) < 2:
                continue
            actual_pair = parsed.get("pair")
            if actual_pair is None:
                continue
            try:
                entry_ts = pd.Timestamp(row.get("entry_time"))
                exit_ts = pd.Timestamp(row.get("exit_time"))
            except Exception:
                continue
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize(NY_TZ)
            else:
                entry_ts = entry_ts.tz_convert(NY_TZ)
            if exit_ts.tzinfo is None:
                exit_ts = exit_ts.tz_localize(NY_TZ)
            else:
                exit_ts = exit_ts.tz_convert(NY_TZ)
            try:
                entry_pos = int(idx.searchsorted(entry_ts, side="left"))
                exit_pos = int(idx.searchsorted(exit_ts, side="right")) - 1
            except Exception:
                continue
            if entry_pos >= len(idx) or exit_pos < entry_pos or exit_pos < 0:
                continue
            if idx[entry_pos] != entry_ts:
                continue
            exit_pos = min(exit_pos, len(idx) - 1)
            entry_price = float(open_arr[entry_pos])
            decision_time = entry_ts - pd.Timedelta(minutes=1)
            context = _compute_context_for_time(
                df_5m,
                decision_time=decision_time,
                session=str(parsed.get("session", "") or ""),
            )
            option_points: Dict[Tuple[float, float], float] = {}
            mfe = safe_float(row.get("mfe"), np.nan)
            mae = safe_float(row.get("mae"), np.nan)
            side = str(row.get("side", "") or "").strip().upper()
            for pair in options:
                option_points[pair] = _simulate_option_points(
                    open_arr=open_arr,
                    high_arr=high_arr,
                    low_arr=low_arr,
                    close_arr=close_arr,
                    entry_pos=entry_pos,
                    exit_pos=exit_pos,
                    side=side,
                    entry_price=entry_price,
                    sl=float(pair[0]),
                    tp=float(pair[1]),
                    mfe=mfe,
                    mae=mae,
                )
            if actual_pair not in option_points:
                option_points[actual_pair] = _simulate_option_points(
                    open_arr=open_arr,
                    high_arr=high_arr,
                    low_arr=low_arr,
                    close_arr=close_arr,
                    entry_pos=entry_pos,
                    exit_pos=exit_pos,
                    side=side,
                    entry_price=entry_price,
                    sl=float(actual_pair[0]),
                    tp=float(actual_pair[1]),
                    mfe=mfe,
                    mae=mae,
                )
            prepared.append(
                {
                    "family_id": family_id,
                    "actual_pair": actual_pair,
                    "context": context,
                    "option_points": option_points,
                }
            )
    return prepared


def _pick_best_option(
    rows: List[Dict[str, Any]],
    options: List[Tuple[float, float]],
) -> Tuple[Optional[Tuple[float, float]], Dict[Tuple[float, float], Dict[str, float]]]:
    stats_by_pair: Dict[Tuple[float, float], Dict[str, float]] = {}
    best_pair: Optional[Tuple[float, float]] = None
    best_score: Optional[Tuple[float, float, float]] = None
    for pair in options:
        stats = _stats_from_points(
            row["option_points"].get(pair)
            for row in rows
            if pair in (row.get("option_points") or {})
        )
        stats_by_pair[pair] = stats
        rank = (
            _score_option(stats),
            float(stats.get("avg_points", 0.0) or 0.0),
            float(stats.get("profit_factor", 0.0) or 0.0),
        )
        if best_score is None or rank > best_score:
            best_pair = pair
            best_score = rank
    return best_pair, stats_by_pair


def _serialize_stats(stats: Dict[str, float]) -> Dict[str, Any]:
    return {
        "trades": int(round(float(stats.get("trades", 0.0) or 0.0))),
        "net_points": float(stats.get("net_points", 0.0) or 0.0),
        "avg_points": float(stats.get("avg_points", 0.0) or 0.0),
        "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
        "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0),
    }


def _build_family_selector(
    *,
    fit_rows: List[Dict[str, Any]],
    oos_rows: List[Dict[str, Any]],
    family_options: Dict[str, List[Tuple[float, float]]],
    selector_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    family_min_support = max(10, safe_int(selector_cfg.get("family_min_support", 60), 60))
    context_min_support = max(8, safe_int(selector_cfg.get("context_min_support", 20), 20))
    min_delta_avg_points = safe_float(selector_cfg.get("min_delta_avg_points", 0.15), 0.15)
    max_pf_drawdown = safe_float(selector_cfg.get("max_pf_drawdown", 0.10), 0.10)
    scope_priority = [
        str(v).strip()
        for v in selector_cfg.get("scope_priority", DEFAULT_SCOPE_PRIORITY)
        if str(v).strip()
    ] or list(DEFAULT_SCOPE_PRIORITY)

    fit_by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    oos_by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in fit_rows:
        fit_by_family[str(row.get("family_id", ""))].append(row)
    for row in oos_rows:
        oos_by_family[str(row.get("family_id", ""))].append(row)

    families: Dict[str, Any] = {}
    changed_fit_trades = 0
    for family_id, options in family_options.items():
        if len(options) < 2:
            continue
        family_fit_rows = fit_by_family.get(family_id, [])
        if len(family_fit_rows) < family_min_support:
            continue
        baseline_stats = _stats_from_points(
            row["option_points"].get(row["actual_pair"]) for row in family_fit_rows
        )
        best_pair, stats_by_pair = _pick_best_option(family_fit_rows, options)
        if best_pair is None:
            continue
        best_stats = stats_by_pair.get(best_pair, {})
        delta_avg = float(best_stats.get("avg_points", 0.0) - baseline_stats.get("avg_points", 0.0))
        delta_pf = float(best_stats.get("profit_factor", 0.0) - baseline_stats.get("profit_factor", 0.0))
        allow_global = bool(
            float(best_stats.get("trades", 0.0)) >= family_min_support
            and best_pair != Counter(row["actual_pair"] for row in family_fit_rows).most_common(1)[0][0]
            and delta_avg >= min_delta_avg_points
            and float(best_stats.get("profit_factor", 0.0)) >= (float(baseline_stats.get("profit_factor", 0.0)) - max_pf_drawdown)
        )

        overrides_by_scope: Dict[str, Dict[str, Any]] = {}
        reference_mode = "baseline_actual_mix"
        reference_pair = None
        if allow_global:
            reference_mode = "family_global_default"
            reference_pair = best_pair

        for scope in scope_priority:
            scope_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in family_fit_rows:
                scope_key = _scope_key(scope, row.get("context", {}))
                if scope_key:
                    scope_rows[scope_key].append(row)
            scope_payload: Dict[str, Any] = {}
            for scope_key, bucket_rows in scope_rows.items():
                if len(bucket_rows) < context_min_support:
                    continue
                bucket_best_pair, bucket_stats_by_pair = _pick_best_option(bucket_rows, options)
                if bucket_best_pair is None:
                    continue
                if reference_pair is not None:
                    reference_stats = bucket_stats_by_pair.get(reference_pair, _stats_from_points([]))
                else:
                    reference_stats = _stats_from_points(
                        row["option_points"].get(row["actual_pair"]) for row in bucket_rows
                    )
                bucket_best_stats = bucket_stats_by_pair.get(bucket_best_pair, {})
                if bucket_best_pair == reference_pair:
                    continue
                delta_bucket_avg = float(
                    bucket_best_stats.get("avg_points", 0.0) - reference_stats.get("avg_points", 0.0)
                )
                if delta_bucket_avg < min_delta_avg_points:
                    continue
                if float(bucket_best_stats.get("profit_factor", 0.0)) < (
                    float(reference_stats.get("profit_factor", 0.0)) - max_pf_drawdown
                ):
                    continue
                scope_payload[scope_key] = {
                    "sl": float(bucket_best_pair[0]),
                    "tp": float(bucket_best_pair[1]),
                    "support_trades": int(len(bucket_rows)),
                    "fit_stats": _serialize_stats(bucket_best_stats),
                    "reference_stats": _serialize_stats(reference_stats),
                    "delta_avg_points": float(delta_bucket_avg),
                }
            if scope_payload:
                overrides_by_scope[scope] = scope_payload

        families[family_id] = {
            "options": [{"sl": float(pair[0]), "tp": float(pair[1])} for pair in options],
            "global_default": (
                {
                    "sl": float(best_pair[0]),
                    "tp": float(best_pair[1]),
                    "support_trades": int(len(family_fit_rows)),
                    "fit_stats": _serialize_stats(best_stats),
                    "baseline_stats": _serialize_stats(baseline_stats),
                    "delta_avg_points": float(delta_avg),
                    "delta_profit_factor": float(delta_pf),
                }
                if allow_global
                else {}
            ),
            "context_overrides": overrides_by_scope,
            "fit_support_trades": int(len(family_fit_rows)),
            "fit_baseline_stats": _serialize_stats(baseline_stats),
        }
        if allow_global:
            changed_fit_trades += sum(1 for row in family_fit_rows if row["actual_pair"] != best_pair)

    return {
        "enabled": bool(families),
        "scope_priority": scope_priority,
        "families": families,
        "fit_summary": {
            "fit_rows": int(len(fit_rows)),
            "families_considered": int(len([k for k, v in family_options.items() if len(v) >= 2])),
            "families_selected": int(len(families)),
            "fit_trades_changed_by_global_defaults": int(changed_fit_trades),
        },
    }


def _selector_pair_for_row(
    selector: Dict[str, Any],
    row: Dict[str, Any],
) -> Tuple[Tuple[float, float], str]:
    actual_pair = row.get("actual_pair")
    if not isinstance(selector, dict) or not bool(selector.get("enabled", False)):
        return actual_pair, "baseline"
    family_id = str(row.get("family_id", "") or "")
    family_payload = (selector.get("families", {}) or {}).get(family_id, {})
    if not isinstance(family_payload, dict):
        return actual_pair, "baseline"
    scope_priority = selector.get("scope_priority", DEFAULT_SCOPE_PRIORITY)
    context = row.get("context", {}) if isinstance(row.get("context"), dict) else {}
    for scope in scope_priority:
        scope_rows = (family_payload.get("context_overrides", {}) or {}).get(scope, {})
        if not isinstance(scope_rows, dict):
            continue
        match_key = _scope_key(str(scope), context)
        scope_payload = scope_rows.get(match_key, {})
        pair = _norm_pair(scope_payload.get("sl"), scope_payload.get("tp")) if isinstance(scope_payload, dict) else None
        if pair is not None:
            return pair, f"context:{scope}"
    global_default = family_payload.get("global_default", {})
    pair = _norm_pair(global_default.get("sl"), global_default.get("tp")) if isinstance(global_default, dict) else None
    if pair is not None:
        return pair, "global"
    return actual_pair, "baseline"


def _evaluate_family_selector(
    selector: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_points: List[float] = []
    selected_points: List[float] = []
    mode_counts: Counter[str] = Counter()
    changed_trades = 0
    by_family: Dict[str, Dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "changed": 0.0, "baseline": 0.0, "selected": 0.0})

    for row in rows:
        actual_pair = row.get("actual_pair")
        option_points = row.get("option_points", {}) if isinstance(row.get("option_points"), dict) else {}
        if actual_pair not in option_points:
            continue
        selected_pair, mode = _selector_pair_for_row(selector, row)
        if selected_pair not in option_points:
            selected_pair = actual_pair
            mode = "baseline"
        base_val = float(option_points[actual_pair])
        sel_val = float(option_points[selected_pair])
        baseline_points.append(base_val)
        selected_points.append(sel_val)
        mode_counts[mode] += 1
        family_id = str(row.get("family_id", "") or "")
        fam = by_family[family_id]
        fam["trades"] += 1.0
        fam["baseline"] += base_val
        fam["selected"] += sel_val
        if selected_pair != actual_pair:
            changed_trades += 1
            fam["changed"] += 1.0

    baseline_stats = _stats_from_points(baseline_points)
    selected_stats = _stats_from_points(selected_points)
    top_families = sorted(
        (
            {
                "family_id": family_id,
                "trades": int(round(values["trades"])),
                "changed_trades": int(round(values["changed"])),
                "baseline_net_points": float(values["baseline"]),
                "selected_net_points": float(values["selected"]),
                "delta_net_points": float(values["selected"] - values["baseline"]),
            }
            for family_id, values in by_family.items()
            if values["changed"] > 0
        ),
        key=lambda item: (item["delta_net_points"], item["changed_trades"]),
        reverse=True,
    )[:20]
    return {
        "baseline": _serialize_stats(baseline_stats),
        "selected": _serialize_stats(selected_stats),
        "delta_avg_points": float(selected_stats.get("avg_points", 0.0) - baseline_stats.get("avg_points", 0.0)),
        "delta_profit_factor": float(selected_stats.get("profit_factor", 0.0) - baseline_stats.get("profit_factor", 0.0)),
        "changed_trades": int(changed_trades),
        "selection_mode_counts": dict(mode_counts),
        "top_changed_families": top_families,
    }


def _default_source_data_path() -> Path:
    de3_v4_cfg = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4"), dict) else {}
    training_data_cfg = de3_v4_cfg.get("training_data", {}) if isinstance(de3_v4_cfg.get("training_data"), dict) else {}
    return _resolve_path(training_data_cfg.get("parquet_path", "es_master_outrights.parquet"))


def _load_bars_for_selector() -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_path = _default_source_data_path()
    if not source_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_parquet(source_path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
        df.index = ts
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(), pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    df = df.sort_index()
    cols = [col for col in ["open", "high", "low", "close", "volume"] if col in df.columns]
    bars_1m = df.loc[:, cols].copy()
    bars_5m = _augment_5m_features(
        _resample_ohlcv(bars_1m, "5min"),
        atr_period=20,
        atr_median_window=390,
        price_loc_window=20,
    )
    return bars_1m, bars_5m


def train_de3_v4_bracket_module(
    *,
    dataset: Dict[str, Any],
    lane_output: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    training_cfg = cfg if isinstance(cfg, dict) else {}
    bracket_cfg = (
        training_cfg.get("bracket_module", {})
        if isinstance(training_cfg.get("bracket_module"), dict)
        else {}
    )
    rows = (
        lane_output.get("all_variant_rows_scored", [])
        if isinstance(lane_output.get("all_variant_rows_scored"), list)
        else dataset.get("variants", [])
    )

    enable_alternative_modes = bool(bracket_cfg.get("enable_alternative_modes", True))
    conservative_sl_mult = safe_float(bracket_cfg.get("conservative_sl_multiplier", 0.92), 0.92)
    conservative_tp_mult = safe_float(bracket_cfg.get("conservative_tp_multiplier", 0.85), 0.85)
    aggressive_sl_mult = safe_float(bracket_cfg.get("aggressive_sl_multiplier", 1.08), 1.08)
    aggressive_tp_mult = safe_float(bracket_cfg.get("aggressive_tp_multiplier", 1.18), 1.18)

    bracket_defaults: Dict[str, Dict[str, Any]] = {}
    bracket_modes: Dict[str, Dict[str, Dict[str, float]]] = {}
    family_options_raw: Dict[str, set[Tuple[float, float]]] = defaultdict(set)

    for row in rows:
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id", "") or "")
        family_id = str(row.get("family_id", "") or "")
        pair = _norm_pair(row.get("best_sl"), row.get("best_tp"))
        if not variant_id or pair is None:
            continue
        support_trades = max(0, safe_int(row.get("support_trades", 0), 0))
        bracket_defaults[variant_id] = {
            "sl": float(pair[0]),
            "tp": float(pair[1]),
            "support_trades": int(support_trades),
        }
        mode_map = {
            "canonical": {"sl": float(pair[0]), "tp": float(pair[1])},
        }
        if enable_alternative_modes:
            conservative = _norm_pair(pair[0] * conservative_sl_mult, pair[1] * conservative_tp_mult)
            aggressive = _norm_pair(pair[0] * aggressive_sl_mult, pair[1] * aggressive_tp_mult)
            if conservative is not None and conservative != pair:
                mode_map["conservative"] = {"sl": float(conservative[0]), "tp": float(conservative[1])}
            if aggressive is not None and aggressive != pair:
                mode_map["aggressive"] = {"sl": float(aggressive[0]), "tp": float(aggressive[1])}
        bracket_modes[variant_id] = mode_map
        if family_id:
            family_options_raw[family_id].add(pair)

    family_options = {
        family_id: sorted(list(pairs))
        for family_id, pairs in family_options_raw.items()
        if len(pairs) >= 2
    }
    fit_trades_path = _resolve_path(bracket_cfg.get("fit_trade_attribution_csv", DEFAULT_FIT_TRADES_PATH))
    oos_trades_path = _resolve_path(bracket_cfg.get("oos_trade_attribution_csv", DEFAULT_OOS_TRADES_PATH))

    family_selector: Dict[str, Any] = {"enabled": False, "scope_priority": list(DEFAULT_SCOPE_PRIORITY), "families": {}}
    selector_report: Dict[str, Any] = {
        "fit_trade_attribution_csv": str(fit_trades_path),
        "oos_trade_attribution_csv": str(oos_trades_path),
        "family_option_count": int(len(family_options)),
        "fit_rows_used": 0,
        "oos_rows_used": 0,
        "oos_evaluation": {},
    }
    if bool(bracket_cfg.get("enable_family_bracket_selector", True)) and family_options:
        try:
            bars_1m, bars_5m = _load_bars_for_selector()
            fit_rows = _prepare_trade_rows(
                csv_path=fit_trades_path,
                family_options=family_options,
                bars_1m=bars_1m,
                df_5m=bars_5m,
            )
            oos_rows = _prepare_trade_rows(
                csv_path=oos_trades_path,
                family_options=family_options,
                bars_1m=bars_1m,
                df_5m=bars_5m,
            )
            family_selector = _build_family_selector(
                fit_rows=fit_rows,
                oos_rows=oos_rows,
                family_options=family_options,
                selector_cfg=bracket_cfg,
            )
            selector_report["fit_rows_used"] = int(len(fit_rows))
            selector_report["oos_rows_used"] = int(len(oos_rows))
            selector_report["fit_summary"] = dict(family_selector.get("fit_summary", {}))
            selector_report["oos_evaluation"] = _evaluate_family_selector(family_selector, oos_rows)
            selector_report["selected_family_count"] = int(
                len((family_selector.get("families", {}) or {}))
            )
        except Exception as exc:
            selector_report["error"] = str(exc)
            family_selector = {"enabled": False, "scope_priority": list(DEFAULT_SCOPE_PRIORITY), "families": {}}

    bracket_training_report = {
        "variant_count": int(len(bracket_defaults)),
        "adaptive_mode_variant_count": int(
            len([v for v in bracket_modes.values() if isinstance(v, dict) and len(v) > 1])
        ),
        "family_bracket_selector": selector_report,
    }

    return {
        "bracket_defaults": bracket_defaults,
        "bracket_modes": bracket_modes,
        "family_bracket_selector": family_selector,
        "bracket_training_report": bracket_training_report,
    }

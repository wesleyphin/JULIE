from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest_symbol_context import apply_symbol_mode, choose_symbol
from ml_physics_legacy_experiment_common import build_feature_frame, normalize_ohlcv_frame


PT_TZ = ZoneInfo("America/Los_Angeles")
NY_TZ = ZoneInfo("America/New_York")
BANK_STEP_POINTS = 12.5

LEVEL_ORDER = ["Prev_Sess", "Q1", "Mid_ORB", "Morn_ORB", "Prev_Day", "Sess_Max"]
LEVEL_ORDER_INDEX = {name: idx for idx, name in enumerate(LEVEL_ORDER)}

MACRO_HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MACRO_NAME_BY_HOUR = {hour: f"Macro_{idx:02d}" for idx, hour in enumerate(MACRO_HOURS, start=1)}
MACRO_INDEX_BY_HOUR = {hour: idx for idx, hour in enumerate(MACRO_HOURS, start=1)}

TOPSTEP_EXIT_HOUR_PT = 13
TOPSTEP_EXIT_MINUTE_PT = 10
TOPSTEP_EXIT_LABEL = "13:10_PT"

COMBO_CATALOG_COLUMNS = ["Macro", "High_Breach", "Low_Breach", "Open_Ref", "Session_Window", "Topstep_Exit"]
MODEL_CATEGORICAL_COLUMNS = [
    "macro_name",
    "session_name",
    "session_window",
    "open_ref_name",
    "high_breach_combo",
    "low_breach_combo",
    "event_family",
    "bank_bias",
    "bank_interaction_source",
    "abs_bank_nearest_level",
    "abs_bank_level_cluster",
    "rel_bank_nearest_level",
    "rel_bank_level_cluster",
]


def canonical_combo_label(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split("+") if part.strip()]
    if not parts:
        return ""
    parts = sorted(parts, key=lambda item: LEVEL_ORDER_INDEX.get(item, 10_000))
    deduped: list[str] = []
    for part in parts:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    return "+".join(deduped)


def combo_key_from_fields(
    macro_name: str,
    high_breach: str,
    low_breach: str,
    open_ref_name: str,
    session_window: str,
    topstep_exit: str = TOPSTEP_EXIT_LABEL,
) -> str:
    return "|".join(
        [
            str(macro_name or "").strip(),
            canonical_combo_label(high_breach),
            canonical_combo_label(low_breach),
            str(open_ref_name or "").strip(),
            str(session_window or "").strip(),
            str(topstep_exit or "").strip(),
        ]
    )


def level_combo_label(level_names: list[str]) -> str:
    parts = sorted(
        [str(name).strip() for name in level_names if str(name).strip()],
        key=lambda item: LEVEL_ORDER_INDEX.get(item, 10_000),
    )
    if not parts:
        return ""
    return "+".join(parts[:2])


def session_name_from_pt_hour(hour_pt: int) -> str:
    hour = int(hour_pt)
    if hour >= 18:
        return "Asia"
    if hour < 6:
        return "London"
    if hour < 12:
        return "NY"
    return "PM"


def session_window_from_macro_index(macro_index: int) -> str:
    return "Asia_to_NY" if int(macro_index) <= 18 else "PM_Only"


def macro_window_context(hour_pt: int, minute_pt: int) -> tuple[Optional[str], int, Optional[int]]:
    hour = int(hour_pt)
    minute = int(minute_pt)
    macro_hour: Optional[int] = None
    minute_offset: Optional[int] = None
    if minute >= 50 and hour in MACRO_NAME_BY_HOUR:
        macro_hour = hour
        minute_offset = minute - 50
    elif minute <= 10:
        prev_hour = (hour - 1) % 24
        if prev_hour in MACRO_NAME_BY_HOUR:
            macro_hour = prev_hour
            minute_offset = 10 + minute
    if macro_hour is None:
        return None, 0, None
    return MACRO_NAME_BY_HOUR[macro_hour], int(MACRO_INDEX_BY_HOUR[macro_hour]), minute_offset


def next_topstep_exit(timestamp_like: Any) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize(PT_TZ)
    else:
        ts = ts.tz_convert(PT_TZ)
    exit_ts = ts.normalize() + pd.Timedelta(hours=TOPSTEP_EXIT_HOUR_PT, minutes=TOPSTEP_EXIT_MINUTE_PT)
    if ts >= exit_ts:
        exit_ts = exit_ts + pd.Timedelta(days=1)
    return exit_ts


def load_combo_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path)
    catalog.columns = [str(col).strip() for col in catalog.columns]
    missing = [col for col in COMBO_CATALOG_COLUMNS if col not in catalog.columns]
    if missing:
        raise ValueError(f"Combo catalog missing required columns: {missing}")
    for column in COMBO_CATALOG_COLUMNS:
        catalog[column] = catalog[column].astype(str).str.strip()
    catalog["High_Breach"] = catalog["High_Breach"].map(canonical_combo_label)
    catalog["Low_Breach"] = catalog["Low_Breach"].map(canonical_combo_label)
    catalog["combo_key"] = catalog.apply(
        lambda row: combo_key_from_fields(
            row["Macro"],
            row["High_Breach"],
            row["Low_Breach"],
            row["Open_Ref"],
            row["Session_Window"],
            row["Topstep_Exit"],
        ),
        axis=1,
    )
    catalog = catalog.drop_duplicates(subset=["combo_key"]).reset_index(drop=True)
    return catalog


def _load_csv_source(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline()
        second = handle.readline()
        needs_skip = "Time Series" in first and "Date" in second

    frame = pd.read_csv(path, skiprows=1 if needs_skip else 0)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    ts_col = next((col for col in ("ts_event", "timestamp", "datetime", "date", "time") if col in frame.columns), None)
    if ts_col is None:
        raise ValueError(f"CSV missing timestamp column: {path}")

    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.replace('"', "", regex=False)
                .str.replace(",", "", regex=False)
            )
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    dt_index = pd.to_datetime(frame[ts_col], errors="coerce", utc=True)
    valid_mask = ~dt_index.isna()
    frame = frame.loc[valid_mask].copy()
    dt_index = pd.DatetimeIndex(dt_index.loc[valid_mask]).tz_convert(NY_TZ)
    frame.index = dt_index
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    return frame


def load_market_data(
    source_path: Path,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    symbol_mode: str = "auto",
    symbol_method: str = "volume",
) -> pd.DataFrame:
    if source_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source_path)
    else:
        frame = _load_csv_source(source_path)

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Source did not produce a DatetimeIndex: {source_path}")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(NY_TZ)
    else:
        frame.index = frame.index.tz_convert(NY_TZ)
    frame = frame.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]

    if "symbol" in frame.columns:
        if str(symbol_mode or "auto").lower() in {"single", "fixed"}:
            selected_symbol = choose_symbol(frame.rename(columns={"symbol": "symbol"}), None)
            frame = frame.loc[frame["symbol"].astype(str) == str(selected_symbol)].copy()
        else:
            filtered, _, _ = apply_symbol_mode(frame.rename(columns={"symbol": "symbol"}), symbol_mode, symbol_method)
            frame = filtered.copy()
        frame = frame.drop(columns=["symbol"], errors="ignore")

    frame = normalize_ohlcv_frame(frame)
    if start:
        start_ts = pd.Timestamp(start)
        start_ts = start_ts.tz_localize(NY_TZ) if start_ts.tzinfo is None else start_ts.tz_convert(NY_TZ)
        frame = frame.loc[frame.index >= start_ts]
    if end:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            if len(str(end).strip()) <= 10:
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            end_ts = end_ts.tz_localize(NY_TZ)
        else:
            end_ts = end_ts.tz_convert(NY_TZ)
        frame = frame.loc[frame.index <= end_ts]

    return frame.dropna(subset=["open", "high", "low", "close"])


def _map_prices_at_times(price_series: pd.Series, target_times: pd.DatetimeIndex) -> np.ndarray:
    aligned = price_series.reindex(target_times)
    if not aligned.isna().any():
        return aligned.to_numpy(dtype=float)
    index = price_series.index
    positions = index.searchsorted(target_times, side="left")
    values = []
    for pos, existing in zip(positions.tolist(), aligned.tolist()):
        if existing is not None and np.isfinite(existing):
            values.append(float(existing))
        elif 0 <= pos < len(price_series):
            values.append(float(price_series.iloc[pos]))
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _session_true_open_timestamp(trade_day: Any, session_name: str) -> pd.Timestamp:
    base = pd.Timestamp(trade_day)
    base = base.tz_localize(PT_TZ) if base.tzinfo is None else base.tz_convert(PT_TZ)
    if session_name == "Asia":
        return base - pd.Timedelta(days=1) + pd.Timedelta(hours=19, minutes=30)
    if session_name == "London":
        return base + pd.Timedelta(hours=1, minutes=30)
    if session_name == "NY":
        return base + pd.Timedelta(hours=7, minutes=30)
    return base + pd.Timedelta(hours=13, minutes=30)


def build_indicator_state_frame(market_df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_ohlcv_frame(market_df)
    if work.empty:
        return pd.DataFrame()

    state = work.copy()
    index_ny = pd.DatetimeIndex(state.index).tz_convert(NY_TZ)
    index_pt = index_ny.tz_convert(PT_TZ)
    state["ts_pt"] = index_pt
    state["pt_hour"] = index_pt.hour
    state["pt_min"] = index_pt.minute
    state["pt_date"] = pd.Index(index_pt.date)

    trade_day_base = pd.to_datetime(pd.Index(index_pt.date))
    trade_day_shift = np.where(index_pt.hour >= 18, 1, 0)
    trade_day = trade_day_base + pd.to_timedelta(trade_day_shift, unit="D")
    state["trade_day"] = pd.Index(trade_day.date)
    state["session_name"] = [session_name_from_pt_hour(hour) for hour in index_pt.hour]
    state["session_key"] = state["trade_day"].astype(str) + "|" + state["session_name"].astype(str)

    session_summary = (
        state.groupby("session_key", sort=False)
        .agg(
            session_high=("high", "max"),
            session_low=("low", "min"),
            first_ts=("ts_pt", "min"),
            trade_day=("trade_day", "first"),
            session_name=("session_name", "first"),
        )
        .sort_values("first_ts")
    )
    session_summary["prev_session_high"] = session_summary["session_high"].shift(1)
    session_summary["prev_session_low"] = session_summary["session_low"].shift(1)
    session_summary["true_open_ts_pt"] = session_summary.apply(
        lambda row: _session_true_open_timestamp(row["trade_day"], row["session_name"]),
        axis=1,
    )
    session_summary["true_open_ts_ny"] = pd.DatetimeIndex(session_summary["true_open_ts_pt"]).tz_convert(NY_TZ)
    session_summary["true_open"] = _map_prices_at_times(state["open"], session_summary["true_open_ts_ny"])

    state = state.join(
        session_summary[
            ["prev_session_high", "prev_session_low", "true_open_ts_ny", "true_open"]
        ],
        on="session_key",
    )
    state["prev_session_mid"] = (state["prev_session_high"] + state["prev_session_low"]) / 2.0
    state["session_high_so_far"] = state.groupby("session_key", sort=False)["high"].cummax()
    state["session_low_so_far"] = state.groupby("session_key", sort=False)["low"].cummin()

    q1_mask = state.index.to_series().le(state["true_open_ts_ny"]).to_numpy(dtype=bool)
    q1_high = state["high"].where(q1_mask)
    q1_low = state["low"].where(q1_mask)
    state["q1_high"] = q1_high.groupby(state["session_key"], sort=False).cummax()
    state["q1_low"] = q1_low.groupby(state["session_key"], sort=False).cummin()
    state["q1_high"] = state.groupby("session_key", sort=False)["q1_high"].ffill()
    state["q1_low"] = state.groupby("session_key", sort=False)["q1_low"].ffill()
    state["q1_mid"] = (state["q1_high"] + state["q1_low"]) / 2.0
    true_open_live_mask = state.index.to_series().ge(state["true_open_ts_ny"]).to_numpy(dtype=bool)
    state.loc[~true_open_live_mask, "true_open"] = np.nan

    daily_open_source = (
        state.loc[(state["pt_hour"] == 15) & (state["pt_min"] == 0), ["pt_date", "open"]]
        .drop_duplicates(subset=["pt_date"], keep="first")
        .set_index("pt_date")["open"]
    )
    daily_ref_base = pd.to_datetime(pd.Index(index_pt.date))
    daily_ref_shift = np.where(index_pt.hour < 15, -1, 0)
    daily_ref_dates = daily_ref_base + pd.to_timedelta(daily_ref_shift, unit="D")
    state["daily_open_ref_date"] = pd.Index(daily_ref_dates.date)
    state["daily_open"] = state["daily_open_ref_date"].map(daily_open_source)

    day_summary = (
        state.groupby("pt_date", sort=True)
        .agg(day_high=("high", "max"), day_low=("low", "min"))
        .sort_index()
    )
    day_summary["prev_day_high"] = day_summary["day_high"].shift(1)
    day_summary["prev_day_low"] = day_summary["day_low"].shift(1)
    state["prev_day_high"] = state["pt_date"].map(day_summary["prev_day_high"])
    state["prev_day_low"] = state["pt_date"].map(day_summary["prev_day_low"])
    state["prev_day_mid"] = (state["prev_day_high"] + state["prev_day_low"]) / 2.0

    mid_mask = (state["pt_hour"] == 0) & (state["pt_min"] < 30)
    state["mid_orb_high"] = state["high"].where(mid_mask).groupby(state["trade_day"], sort=False).cummax()
    state["mid_orb_low"] = state["low"].where(mid_mask).groupby(state["trade_day"], sort=False).cummin()
    state["mid_orb_high"] = state.groupby("trade_day", sort=False)["mid_orb_high"].ffill()
    state["mid_orb_low"] = state.groupby("trade_day", sort=False)["mid_orb_low"].ffill()
    state["mid_orb_mid"] = (state["mid_orb_high"] + state["mid_orb_low"]) / 2.0

    morn_mask = (state["pt_hour"] == 6) & (state["pt_min"] >= 30)
    state["morn_orb_high"] = state["high"].where(morn_mask).groupby(state["trade_day"], sort=False).cummax()
    state["morn_orb_low"] = state["low"].where(morn_mask).groupby(state["trade_day"], sort=False).cummin()
    state["morn_orb_high"] = state.groupby("trade_day", sort=False)["morn_orb_high"].ffill()
    state["morn_orb_low"] = state.groupby("trade_day", sort=False)["morn_orb_low"].ffill()
    state["morn_orb_mid"] = (state["morn_orb_high"] + state["morn_orb_low"]) / 2.0

    state["trade_day_high_so_far"] = state.groupby("trade_day", sort=False)["high"].cummax()
    state["trade_day_low_so_far"] = state.groupby("trade_day", sort=False)["low"].cummin()

    pm_mask = state["pt_hour"] >= 12
    state["pm_only_high_so_far"] = state["high"].where(pm_mask).groupby(state["trade_day"], sort=False).cummax()
    state["pm_only_low_so_far"] = state["low"].where(pm_mask).groupby(state["trade_day"], sort=False).cummin()
    state["pm_only_high_so_far"] = state.groupby("trade_day", sort=False)["pm_only_high_so_far"].ffill()
    state["pm_only_low_so_far"] = state.groupby("trade_day", sort=False)["pm_only_low_so_far"].ffill()

    state["session_anchor_price"] = state.groupby("session_key", sort=False)["open"].transform("first")

    prev_close = state["close"].shift(1)
    true_range = pd.concat(
        [
            state["high"] - state["low"],
            (state["high"] - prev_close).abs(),
            (state["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    state["atr_pts"] = true_range.rolling(window=14, min_periods=5).mean()

    macro_context = [
        macro_window_context(hour_pt, minute_pt)
        for hour_pt, minute_pt in zip(state["pt_hour"].tolist(), state["pt_min"].tolist())
    ]
    state["macro_window_name"] = [item[0] for item in macro_context]
    state["macro_window_index"] = [int(item[1]) if item[1] else 0 for item in macro_context]
    state["macro_window_minute_offset"] = [item[2] for item in macro_context]
    state["macro_window_active"] = state["macro_window_index"].gt(0).astype(int)

    abs_nearest_bank = np.round(state["close"].to_numpy(dtype=float) / BANK_STEP_POINTS) * BANK_STEP_POINTS
    state["nearest_abs_bank"] = abs_nearest_bank
    state["dist_nearest_abs_bank_pts"] = state["close"].to_numpy(dtype=float) - abs_nearest_bank
    state["dist_nearest_abs_bank_atr"] = state["dist_nearest_abs_bank_pts"] / state["atr_pts"]
    abs_touch_meta = [
        _bank_touch_or_cross(low_value, high_value, anchor=0.0)
        for low_value, high_value in zip(
            state["low"].to_numpy(dtype=float),
            state["high"].to_numpy(dtype=float),
        )
    ]
    state["abs_bank_range_touch"] = [item[0] for item in abs_touch_meta]
    state["abs_bank_span_count"] = [item[1] for item in abs_touch_meta]
    state["abs_bank_touched_level"] = [item[2] for item in abs_touch_meta]
    state["abs_bank_open_close_cross"] = (
        np.floor(state["open"].to_numpy(dtype=float) / BANK_STEP_POINTS)
        != np.floor(state["close"].to_numpy(dtype=float) / BANK_STEP_POINTS)
    ).astype(int)

    rel_anchor = state["session_anchor_price"].to_numpy(dtype=float)
    rel_nearest_bank = rel_anchor + np.round((state["close"].to_numpy(dtype=float) - rel_anchor) / BANK_STEP_POINTS) * BANK_STEP_POINTS
    state["nearest_rel_bank"] = rel_nearest_bank
    state["dist_nearest_rel_bank_pts"] = state["close"].to_numpy(dtype=float) - rel_nearest_bank
    state["dist_nearest_rel_bank_atr"] = state["dist_nearest_rel_bank_pts"] / state["atr_pts"]
    rel_touch_meta = [
        _bank_touch_or_cross(low_value, high_value, anchor=anchor_value)
        for low_value, high_value, anchor_value in zip(
            state["low"].to_numpy(dtype=float),
            state["high"].to_numpy(dtype=float),
            rel_anchor,
        )
    ]
    state["rel_bank_range_touch"] = [item[0] for item in rel_touch_meta]
    state["rel_bank_span_count"] = [item[1] for item in rel_touch_meta]
    state["rel_bank_touched_level"] = [item[2] for item in rel_touch_meta]
    state["rel_bank_open_close_cross"] = (
        np.floor((state["open"].to_numpy(dtype=float) - rel_anchor) / BANK_STEP_POINTS)
        != np.floor((state["close"].to_numpy(dtype=float) - rel_anchor) / BANK_STEP_POINTS)
    ).astype(int)
    state["abs_rel_bank_gap_pts"] = (state["nearest_abs_bank"] - state["nearest_rel_bank"]).abs()
    state["abs_rel_bank_confluence"] = state["abs_rel_bank_gap_pts"].le(1.0).astype(int)

    feature_frame = build_feature_frame(state[["open", "high", "low", "close", "volume"]])
    for column in feature_frame.columns:
        state[column] = feature_frame.reindex(state.index)[column]

    return state


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _finite_or_none(value: Any) -> Optional[float]:
    out = _safe_float(value, float("nan"))
    return None if not np.isfinite(out) else float(out)


def _bank_touch_or_cross(
    low_value: float,
    high_value: float,
    *,
    anchor: float = 0.0,
    step: float = BANK_STEP_POINTS,
) -> tuple[int, int, float]:
    if not (np.isfinite(low_value) and np.isfinite(high_value) and np.isfinite(anchor) and step > 0.0):
        return 0, 0, float("nan")
    low_norm = (float(low_value) - float(anchor)) / float(step)
    high_norm = (float(high_value) - float(anchor)) / float(step)
    low_bucket = int(np.floor(low_norm))
    high_bucket = int(np.floor(high_norm))
    touch = int(high_bucket > low_bucket)
    first_level = float("nan")
    if touch:
        first_level = float(anchor + high_bucket * step)
    else:
        tol = 1e-9
        if abs(low_norm - round(low_norm)) <= tol:
            touch = 1
            first_level = float(anchor + round(low_norm) * step)
        elif abs(high_norm - round(high_norm)) <= tol:
            touch = 1
            first_level = float(anchor + round(high_norm) * step)
    span_count = max(0, high_bucket - low_bucket)
    return int(touch), int(span_count), float(first_level)


def _bank_event_family(row: pd.Series) -> str:
    abs_cross = int(_safe_float(row.get("abs_bank_open_close_cross"), 0.0))
    rel_cross = int(_safe_float(row.get("rel_bank_open_close_cross"), 0.0))
    abs_touch = int(_safe_float(row.get("abs_bank_range_touch"), 0.0))
    rel_touch = int(_safe_float(row.get("rel_bank_range_touch"), 0.0))
    confluence = int(_safe_float(row.get("abs_rel_bank_confluence"), 0.0))
    if abs_cross and rel_cross:
        return "dual_cross"
    if abs_touch and rel_touch:
        return "dual_touch"
    if abs_cross:
        return "abs_cross"
    if rel_cross:
        return "rel_cross"
    if abs_touch:
        return "abs_touch"
    if rel_touch:
        return "rel_touch"
    if confluence:
        return "bank_confluence"
    return ""


def _bank_interaction_source(row: pd.Series) -> str:
    abs_touch = int(_safe_float(row.get("abs_bank_range_touch"), 0.0))
    rel_touch = int(_safe_float(row.get("rel_bank_range_touch"), 0.0))
    if abs_touch and rel_touch:
        return "absolute+relative"
    if abs_touch:
        return "absolute"
    if rel_touch:
        return "relative"
    return ""


def _bank_bias(row: pd.Series) -> str:
    abs_dist = _safe_float(row.get("dist_nearest_abs_bank_pts"))
    rel_dist = _safe_float(row.get("dist_nearest_rel_bank_pts"))
    abs_sign = 0 if not np.isfinite(abs_dist) or abs(abs_dist) <= 1e-9 else (1 if abs_dist > 0.0 else -1)
    rel_sign = 0 if not np.isfinite(rel_dist) or abs(rel_dist) <= 1e-9 else (1 if rel_dist > 0.0 else -1)
    if abs_sign > 0 and rel_sign > 0:
        return "above"
    if abs_sign < 0 and rel_sign < 0:
        return "below"
    if abs_sign == 0 and rel_sign == 0:
        return "on_bank"
    return "mixed"


def _finite_level_list(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        numeric = _safe_float(value)
        if np.isfinite(numeric):
            out.append(float(numeric))
    return out


def _confluence_stats(
    anchor_level: float,
    reference_levels: list[float],
    *,
    atr_pts: float,
) -> dict[str, float]:
    if not (np.isfinite(anchor_level) and reference_levels and np.isfinite(atr_pts) and atr_pts > 0.0):
        return {
            "nearest_dist_pts": float("nan"),
            "nearest_dist_atr": float("nan"),
            "count_025atr": 0.0,
            "count_050atr": 0.0,
            "count_100atr": 0.0,
        }
    distances = np.abs(np.asarray(reference_levels, dtype=float) - float(anchor_level))
    dist_atr = distances / float(atr_pts)
    return {
        "nearest_dist_pts": float(np.min(distances)) if distances.size else float("nan"),
        "nearest_dist_atr": float(np.min(dist_atr)) if dist_atr.size else float("nan"),
        "count_025atr": float(np.sum(dist_atr <= 0.25)),
        "count_050atr": float(np.sum(dist_atr <= 0.50)),
        "count_100atr": float(np.sum(dist_atr <= 1.00)),
    }


def _canonical_level_cluster(level_names: list[str], *, limit: int = 2) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_name in level_names:
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    return "+".join(cleaned[: max(int(limit), 1)])


def _named_confluence_stats(
    anchor_level: float,
    reference_pairs: list[tuple[str, float]],
    *,
    atr_pts: float,
    cluster_threshold_atr: float = 0.50,
    max_cluster_levels: int = 2,
) -> dict[str, Any]:
    if not (np.isfinite(anchor_level) and reference_pairs and np.isfinite(atr_pts) and atr_pts > 0.0):
        return {
            "nearest_dist_pts": float("nan"),
            "nearest_dist_atr": float("nan"),
            "count_025atr": 0.0,
            "count_050atr": 0.0,
            "count_100atr": 0.0,
            "nearest_level": "",
            "level_cluster": "",
        }

    ranked: list[tuple[float, float, int, str]] = []
    for order_idx, (raw_name, raw_value) in enumerate(reference_pairs):
        name = str(raw_name or "").strip()
        value = _safe_float(raw_value)
        if not name or not np.isfinite(value):
            continue
        dist_pts = abs(float(value) - float(anchor_level))
        dist_atr = dist_pts / float(atr_pts)
        ranked.append((dist_atr, dist_pts, int(order_idx), name))

    if not ranked:
        return {
            "nearest_dist_pts": float("nan"),
            "nearest_dist_atr": float("nan"),
            "count_025atr": 0.0,
            "count_050atr": 0.0,
            "count_100atr": 0.0,
            "nearest_level": "",
            "level_cluster": "",
        }

    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    dist_atr_values = np.asarray([item[0] for item in ranked], dtype=float)
    dist_pts_values = np.asarray([item[1] for item in ranked], dtype=float)
    nearest_level = ranked[0][3]
    cluster_names = [item[3] for item in ranked if item[0] <= float(cluster_threshold_atr)]
    if not cluster_names:
        cluster_names = [nearest_level]
    level_cluster = _canonical_level_cluster(cluster_names, limit=max_cluster_levels)
    return {
        "nearest_dist_pts": float(np.min(dist_pts_values)),
        "nearest_dist_atr": float(np.min(dist_atr_values)),
        "count_025atr": float(np.sum(dist_atr_values <= 0.25)),
        "count_050atr": float(np.sum(dist_atr_values <= 0.50)),
        "count_100atr": float(np.sum(dist_atr_values <= 1.00)),
        "nearest_level": nearest_level,
        "level_cluster": level_cluster,
    }


def _reference_level_pairs(
    row: pd.Series,
    *,
    window_high: float,
    window_low: float,
) -> list[tuple[str, float]]:
    window_mid = float("nan")
    if np.isfinite(window_high) and np.isfinite(window_low):
        window_mid = (float(window_high) + float(window_low)) / 2.0

    raw_pairs = [
        ("True_Open", row.get("true_open")),
        ("Daily_Open", row.get("daily_open")),
        ("Prev_Sess_High", row.get("prev_session_high")),
        ("Prev_Sess_Low", row.get("prev_session_low")),
        ("Prev_Sess_Mid", row.get("prev_session_mid")),
        ("Q1_High", row.get("q1_high")),
        ("Q1_Low", row.get("q1_low")),
        ("Q1_Mid", row.get("q1_mid")),
        ("Mid_ORB_High", row.get("mid_orb_high")),
        ("Mid_ORB_Low", row.get("mid_orb_low")),
        ("Mid_ORB_Mid", row.get("mid_orb_mid")),
        ("Morn_ORB_High", row.get("morn_orb_high")),
        ("Morn_ORB_Low", row.get("morn_orb_low")),
        ("Morn_ORB_Mid", row.get("morn_orb_mid")),
        ("Prev_Day_High", row.get("prev_day_high")),
        ("Prev_Day_Low", row.get("prev_day_low")),
        ("Prev_Day_Mid", row.get("prev_day_mid")),
        ("Sess_Max_High", window_high),
        ("Sess_Max_Low", window_low),
        ("Sess_Max_Mid", window_mid),
    ]
    pairs: list[tuple[str, float]] = []
    for raw_name, raw_value in raw_pairs:
        numeric = _safe_float(raw_value)
        if np.isfinite(numeric):
            pairs.append((str(raw_name), float(numeric)))
    return pairs


def _level_values_for_row(row: pd.Series, window_high: float, window_low: float) -> dict[str, tuple[float, float]]:
    return {
        "Prev_Sess": (_safe_float(row.get("prev_session_high")), _safe_float(row.get("prev_session_low"))),
        "Q1": (_safe_float(row.get("q1_high")), _safe_float(row.get("q1_low"))),
        "Mid_ORB": (_safe_float(row.get("mid_orb_high")), _safe_float(row.get("mid_orb_low"))),
        "Morn_ORB": (_safe_float(row.get("morn_orb_high")), _safe_float(row.get("morn_orb_low"))),
        "Prev_Day": (_safe_float(row.get("prev_day_high")), _safe_float(row.get("prev_day_low"))),
        "Sess_Max": (float(window_high), float(window_low)),
    }


def infer_breach_combo(
    *,
    open_ref: float,
    current_price: float,
    window_high: float,
    window_low: float,
    level_values: dict[str, tuple[float, float]],
    side: str,
) -> tuple[str, list[str], dict[str, int], dict[str, float]]:
    candidates: list[tuple[float, str]] = []
    flags: dict[str, int] = {name: 0 for name in LEVEL_ORDER}
    levels: dict[str, float] = {}

    for name in LEVEL_ORDER:
        upper_level, lower_level = level_values.get(name, (float("nan"), float("nan")))
        if side == "high":
            level = float(upper_level)
            if not np.isfinite(level) or level <= open_ref:
                continue
            if window_high < level:
                continue
        else:
            level = float(lower_level)
            if not np.isfinite(level) or level >= open_ref:
                continue
            if window_low > level:
                continue
        flags[name] = 1
        levels[name] = level
        candidates.append((abs(current_price - level), name))

    if not candidates:
        return "", [], flags, levels

    candidates.sort(key=lambda item: (item[0], LEVEL_ORDER_INDEX.get(item[1], 10_000)))
    selected = [name for _, name in candidates[:2]]
    return level_combo_label(selected), selected, flags, levels


def build_topstep_combo_dataset(
    market_df: pd.DataFrame,
    combo_catalog: pd.DataFrame,
) -> pd.DataFrame:
    state = build_indicator_state_frame(market_df)
    if state.empty:
        return pd.DataFrame()

    valid_combo_keys = set(combo_catalog["combo_key"].astype(str))
    macro_mask = state["pt_min"].eq(50) & state["pt_hour"].isin(MACRO_HOURS)
    macro_rows = state.loc[macro_mask].copy()
    if macro_rows.empty:
        return pd.DataFrame()

    high_arr = state["high"].to_numpy(dtype=float)
    low_arr = state["low"].to_numpy(dtype=float)
    close_arr = state["close"].to_numpy(dtype=float)
    index_ny = pd.DatetimeIndex(state.index)

    records: list[dict[str, Any]] = []
    position_lookup = {ts: pos for pos, ts in enumerate(index_ny)}

    base_feature_columns = [
        column
        for column in state.columns
        if column not in {"ts_pt", "pt_date", "trade_day", "session_name", "session_key", "daily_open_ref_date"}
        and pd.api.types.is_numeric_dtype(state[column])
    ]

    for timestamp, row in macro_rows.iterrows():
        entry_pos = position_lookup.get(timestamp)
        if entry_pos is None:
            continue

        macro_name = MACRO_NAME_BY_HOUR.get(int(row["pt_hour"]))
        macro_index = int(MACRO_INDEX_BY_HOUR.get(int(row["pt_hour"]), 0))
        if not macro_name or macro_index <= 0:
            continue
        session_window = session_window_from_macro_index(macro_index)
        exit_ts_pt = next_topstep_exit(row["ts_pt"])
        exit_ts_ny = exit_ts_pt.tz_convert(NY_TZ)
        exit_pos = int(index_ny.searchsorted(exit_ts_ny, side="left"))
        if exit_pos <= entry_pos or exit_pos >= len(index_ny):
            continue

        future_high_max = float(np.max(high_arr[entry_pos + 1 : exit_pos + 1]))
        future_low_min = float(np.min(low_arr[entry_pos + 1 : exit_pos + 1]))
        exit_price = float(close_arr[exit_pos])
        entry_price = float(row["close"])

        window_high = _safe_float(
            row["pm_only_high_so_far"] if session_window == "PM_Only" else row["trade_day_high_so_far"]
        )
        window_low = _safe_float(
            row["pm_only_low_so_far"] if session_window == "PM_Only" else row["trade_day_low_so_far"]
        )
        if not (np.isfinite(window_high) and np.isfinite(window_low)):
            continue

        for open_ref_name, open_ref_value in (
            ("Daily_Open", row.get("daily_open")),
            ("True_Open", row.get("true_open")),
        ):
            open_ref = _safe_float(open_ref_value)
            if not np.isfinite(open_ref):
                continue

            level_values = _level_values_for_row(row, window_high, window_low)
            high_combo, high_selected, high_flags, _high_levels = infer_breach_combo(
                open_ref=open_ref,
                current_price=entry_price,
                window_high=window_high,
                window_low=window_low,
                level_values=level_values,
                side="high",
            )
            low_combo, low_selected, low_flags, _low_levels = infer_breach_combo(
                open_ref=open_ref,
                current_price=entry_price,
                window_high=window_high,
                window_low=window_low,
                level_values=level_values,
                side="low",
            )
            if not high_combo or not low_combo:
                continue

            combo_key = combo_key_from_fields(
                macro_name,
                high_combo,
                low_combo,
                open_ref_name,
                session_window,
                TOPSTEP_EXIT_LABEL,
            )
            if combo_key not in valid_combo_keys:
                continue

            atr_pts = _safe_float(row.get("atr_pts"), 1.0)
            if not np.isfinite(atr_pts) or atr_pts <= 0.0:
                atr_pts = max(float(row["high"] - row["low"]), 0.25)

            record: dict[str, Any] = {
                "entry_time": timestamp.isoformat(),
                "exit_time": index_ny[exit_pos].isoformat(),
                "entry_pos": int(entry_pos),
                "exit_pos": int(exit_pos),
                "trade_day": str(row["trade_day"]),
                "macro_name": macro_name,
                "macro_index": int(macro_index),
                "session_name": str(row["session_name"]),
                "session_window": session_window,
                "open_ref_name": open_ref_name,
                "high_breach_combo": high_combo,
                "low_breach_combo": low_combo,
                "combo_key": combo_key,
                "entry_price": entry_price,
                "exit_price_to_topstep": exit_price,
                "future_high_max": future_high_max,
                "future_low_min": future_low_min,
                "atr_pts": atr_pts,
                "price_vs_open_ref_pts": entry_price - open_ref,
                "price_vs_open_ref_atr": (entry_price - open_ref) / atr_pts,
                "window_range_pts": window_high - window_low,
                "window_range_atr": (window_high - window_low) / atr_pts,
                "window_extension_up_pts": window_high - open_ref,
                "window_extension_dn_pts": open_ref - window_low,
                "window_extension_up_atr": (window_high - open_ref) / atr_pts,
                "window_extension_dn_atr": (open_ref - window_low) / atr_pts,
                "high_breach_count": int(sum(high_flags.values())),
                "low_breach_count": int(sum(low_flags.values())),
                "high_selected_count": int(len(high_selected)),
                "low_selected_count": int(len(low_selected)),
                "long_exit_pnl_pts": exit_price - entry_price,
                "short_exit_pnl_pts": entry_price - exit_price,
                "long_mfe_pts": future_high_max - entry_price,
                "long_mae_pts": entry_price - future_low_min,
                "short_mfe_pts": entry_price - future_low_min,
                "short_mae_pts": future_high_max - entry_price,
                "long_win": 1 if exit_price > entry_price else 0,
                "short_win": 1 if exit_price < entry_price else 0,
            }

            for level_name in LEVEL_ORDER:
                upper_level, lower_level = level_values[level_name]
                upper_level = _finite_or_none(upper_level)
                lower_level = _finite_or_none(lower_level)
                record[f"upper_breached_{level_name.lower()}"] = int(high_flags[level_name])
                record[f"lower_breached_{level_name.lower()}"] = int(low_flags[level_name])
                record[f"dist_upper_{level_name.lower()}_pts"] = (
                    None if upper_level is None else upper_level - entry_price
                )
                record[f"dist_lower_{level_name.lower()}_pts"] = (
                    None if lower_level is None else entry_price - lower_level
                )
                record[f"dist_upper_{level_name.lower()}_atr"] = (
                    None if upper_level is None else (upper_level - entry_price) / atr_pts
                )
                record[f"dist_lower_{level_name.lower()}_atr"] = (
                    None if lower_level is None else (entry_price - lower_level) / atr_pts
                )

            for column in base_feature_columns:
                record[column] = row.get(column)

            records.append(record)

    if not records:
        return pd.DataFrame()

    dataset = pd.DataFrame.from_records(records)
    dataset["entry_ts"] = pd.to_datetime(dataset["entry_time"], errors="coerce", utc=True)
    dataset = dataset.sort_values("entry_ts").reset_index(drop=True)
    dataset = dataset.drop(columns=["entry_ts"])
    return dataset


def build_topstep_combo_bank_event_dataset(
    market_df: pd.DataFrame,
    combo_catalog: pd.DataFrame,
) -> pd.DataFrame:
    state = build_indicator_state_frame(market_df)
    if state.empty:
        return pd.DataFrame()

    valid_combo_keys = set(combo_catalog["combo_key"].astype(str))
    event_mask = state["macro_window_active"].astype(bool) & (
        state["abs_bank_range_touch"].astype(bool)
        | state["rel_bank_range_touch"].astype(bool)
        | state["abs_bank_open_close_cross"].astype(bool)
        | state["rel_bank_open_close_cross"].astype(bool)
        | state["abs_rel_bank_confluence"].astype(bool)
    )
    event_rows = state.loc[event_mask].copy()
    if event_rows.empty:
        return pd.DataFrame()

    high_arr = state["high"].to_numpy(dtype=float)
    low_arr = state["low"].to_numpy(dtype=float)
    close_arr = state["close"].to_numpy(dtype=float)
    index_ny = pd.DatetimeIndex(state.index)
    position_lookup = {ts: pos for pos, ts in enumerate(index_ny)}
    records: list[dict[str, Any]] = []

    base_feature_columns = [
        column
        for column in state.columns
        if column not in {"ts_pt", "pt_date", "trade_day", "session_name", "session_key", "daily_open_ref_date"}
        and pd.api.types.is_numeric_dtype(state[column])
    ]

    for timestamp, row in event_rows.iterrows():
        entry_pos = position_lookup.get(timestamp)
        if entry_pos is None:
            continue
        macro_name = str(row.get("macro_window_name") or "").strip()
        macro_index = int(_safe_float(row.get("macro_window_index"), 0.0))
        if not macro_name or macro_index <= 0:
            continue
        event_family = _bank_event_family(row)
        if not event_family:
            continue
        bank_interaction_source = _bank_interaction_source(row)
        bank_bias = _bank_bias(row)

        session_window = session_window_from_macro_index(macro_index)
        exit_ts_pt = next_topstep_exit(row["ts_pt"])
        exit_ts_ny = exit_ts_pt.tz_convert(NY_TZ)
        exit_pos = int(index_ny.searchsorted(exit_ts_ny, side="left"))
        if exit_pos <= entry_pos or exit_pos >= len(index_ny):
            continue

        future_high_max = float(np.max(high_arr[entry_pos + 1 : exit_pos + 1]))
        future_low_min = float(np.min(low_arr[entry_pos + 1 : exit_pos + 1]))
        exit_price = float(close_arr[exit_pos])
        entry_price = float(row["close"])

        window_high = _safe_float(
            row["pm_only_high_so_far"] if session_window == "PM_Only" else row["trade_day_high_so_far"]
        )
        window_low = _safe_float(
            row["pm_only_low_so_far"] if session_window == "PM_Only" else row["trade_day_low_so_far"]
        )
        if not (np.isfinite(window_high) and np.isfinite(window_low)):
            continue

        level_values = _level_values_for_row(row, window_high, window_low)
        atr_pts = _safe_float(row.get("atr_pts"), 1.0)
        if not np.isfinite(atr_pts) or atr_pts <= 0.0:
            atr_pts = max(float(row["high"] - row["low"]), 0.25)

        for open_ref_name, open_ref_value in (
            ("Daily_Open", row.get("daily_open")),
            ("True_Open", row.get("true_open")),
        ):
            open_ref = _safe_float(open_ref_value)
            if not np.isfinite(open_ref):
                continue
            reference_pairs = _reference_level_pairs(
                row,
                window_high=window_high,
                window_low=window_low,
            )
            abs_bank_level_stats = _named_confluence_stats(
                _safe_float(row.get("nearest_abs_bank")),
                reference_pairs,
                atr_pts=atr_pts,
            )
            rel_bank_level_stats = _named_confluence_stats(
                _safe_float(row.get("nearest_rel_bank")),
                reference_pairs,
                atr_pts=atr_pts,
            )
            best_bank_level_nearest_dist_atr = min(
                _safe_float(abs_bank_level_stats["nearest_dist_atr"], float("inf")),
                _safe_float(rel_bank_level_stats["nearest_dist_atr"], float("inf")),
            )
            if not np.isfinite(best_bank_level_nearest_dist_atr):
                best_bank_level_nearest_dist_atr = float("nan")
            shared_bank_nearest_level = int(
                bool(abs_bank_level_stats["nearest_level"])
                and abs_bank_level_stats["nearest_level"] == rel_bank_level_stats["nearest_level"]
            )
            abs_bank_vs_open_ref_steps = (
                (_safe_float(row.get("nearest_abs_bank")) - float(open_ref)) / BANK_STEP_POINTS
            )
            rel_bank_vs_open_ref_steps = (
                (_safe_float(row.get("nearest_rel_bank")) - float(open_ref)) / BANK_STEP_POINTS
            )
            abs_bank_touched_offset_steps = (
                (_safe_float(row.get("abs_bank_touched_level")) - float(open_ref)) / BANK_STEP_POINTS
            )
            rel_bank_touched_offset_steps = (
                (_safe_float(row.get("rel_bank_touched_level")) - float(open_ref)) / BANK_STEP_POINTS
            )

            high_combo, high_selected, high_flags, _high_levels = infer_breach_combo(
                open_ref=open_ref,
                current_price=entry_price,
                window_high=window_high,
                window_low=window_low,
                level_values=level_values,
                side="high",
            )
            low_combo, low_selected, low_flags, _low_levels = infer_breach_combo(
                open_ref=open_ref,
                current_price=entry_price,
                window_high=window_high,
                window_low=window_low,
                level_values=level_values,
                side="low",
            )
            if not high_combo or not low_combo:
                continue

            combo_key = combo_key_from_fields(
                macro_name,
                high_combo,
                low_combo,
                open_ref_name,
                session_window,
                TOPSTEP_EXIT_LABEL,
            )
            if combo_key not in valid_combo_keys:
                continue

            record: dict[str, Any] = {
                "entry_time": timestamp.isoformat(),
                "exit_time": index_ny[exit_pos].isoformat(),
                "entry_pos": int(entry_pos),
                "exit_pos": int(exit_pos),
                "trade_day": str(row["trade_day"]),
                "macro_name": macro_name,
                "macro_index": int(macro_index),
                "session_name": str(row["session_name"]),
                "session_window": session_window,
                "open_ref_name": open_ref_name,
                "high_breach_combo": high_combo,
                "low_breach_combo": low_combo,
                "combo_key": combo_key,
                "event_family": event_family,
                "bank_bias": bank_bias,
                "bank_interaction_source": bank_interaction_source,
                "abs_bank_nearest_level": abs_bank_level_stats["nearest_level"],
                "abs_bank_level_cluster": abs_bank_level_stats["level_cluster"],
                "rel_bank_nearest_level": rel_bank_level_stats["nearest_level"],
                "rel_bank_level_cluster": rel_bank_level_stats["level_cluster"],
                "entry_price": entry_price,
                "exit_price_to_topstep": exit_price,
                "future_high_max": future_high_max,
                "future_low_min": future_low_min,
                "atr_pts": atr_pts,
                "abs_bank_vs_open_ref_steps": abs_bank_vs_open_ref_steps,
                "rel_bank_vs_open_ref_steps": rel_bank_vs_open_ref_steps,
                "abs_rel_bank_gap_steps": _safe_float(row.get("abs_rel_bank_gap_pts")) / BANK_STEP_POINTS,
                "abs_bank_touched_offset_steps": abs_bank_touched_offset_steps,
                "rel_bank_touched_offset_steps": rel_bank_touched_offset_steps,
                "abs_bank_level_nearest_dist_pts": abs_bank_level_stats["nearest_dist_pts"],
                "abs_bank_level_nearest_dist_atr": abs_bank_level_stats["nearest_dist_atr"],
                "abs_bank_level_count_025atr": abs_bank_level_stats["count_025atr"],
                "abs_bank_level_count_050atr": abs_bank_level_stats["count_050atr"],
                "abs_bank_level_count_100atr": abs_bank_level_stats["count_100atr"],
                "rel_bank_level_nearest_dist_pts": rel_bank_level_stats["nearest_dist_pts"],
                "rel_bank_level_nearest_dist_atr": rel_bank_level_stats["nearest_dist_atr"],
                "rel_bank_level_count_025atr": rel_bank_level_stats["count_025atr"],
                "rel_bank_level_count_050atr": rel_bank_level_stats["count_050atr"],
                "rel_bank_level_count_100atr": rel_bank_level_stats["count_100atr"],
                "best_bank_level_nearest_dist_atr": best_bank_level_nearest_dist_atr,
                "combined_bank_level_count_050atr": (
                    float(abs_bank_level_stats["count_050atr"]) + float(rel_bank_level_stats["count_050atr"])
                ),
                "shared_bank_nearest_level": shared_bank_nearest_level,
                "price_vs_open_ref_pts": entry_price - open_ref,
                "price_vs_open_ref_atr": (entry_price - open_ref) / atr_pts,
                "window_range_pts": window_high - window_low,
                "window_range_atr": (window_high - window_low) / atr_pts,
                "window_extension_up_pts": window_high - open_ref,
                "window_extension_dn_pts": open_ref - window_low,
                "window_extension_up_atr": (window_high - open_ref) / atr_pts,
                "window_extension_dn_atr": (open_ref - window_low) / atr_pts,
                "high_breach_count": int(sum(high_flags.values())),
                "low_breach_count": int(sum(low_flags.values())),
                "high_selected_count": int(len(high_selected)),
                "low_selected_count": int(len(low_selected)),
                "long_exit_pnl_pts": exit_price - entry_price,
                "short_exit_pnl_pts": entry_price - exit_price,
                "long_mfe_pts": future_high_max - entry_price,
                "long_mae_pts": entry_price - future_low_min,
                "short_mfe_pts": entry_price - future_low_min,
                "short_mae_pts": future_high_max - entry_price,
                "long_win": 1 if exit_price > entry_price else 0,
                "short_win": 1 if exit_price < entry_price else 0,
            }

            for level_name in LEVEL_ORDER:
                upper_level, lower_level = level_values[level_name]
                upper_level = _finite_or_none(upper_level)
                lower_level = _finite_or_none(lower_level)
                record[f"upper_breached_{level_name.lower()}"] = int(high_flags[level_name])
                record[f"lower_breached_{level_name.lower()}"] = int(low_flags[level_name])
                record[f"dist_upper_{level_name.lower()}_pts"] = (
                    None if upper_level is None else upper_level - entry_price
                )
                record[f"dist_lower_{level_name.lower()}_pts"] = (
                    None if lower_level is None else entry_price - lower_level
                )
                record[f"dist_upper_{level_name.lower()}_atr"] = (
                    None if upper_level is None else (upper_level - entry_price) / atr_pts
                )
                record[f"dist_lower_{level_name.lower()}_atr"] = (
                    None if lower_level is None else (entry_price - lower_level) / atr_pts
                )

            for column in base_feature_columns:
                record[column] = row.get(column)

            records.append(record)

    if not records:
        return pd.DataFrame()

    dataset = pd.DataFrame.from_records(records)
    dataset["entry_ts"] = pd.to_datetime(dataset["entry_time"], errors="coerce", utc=True)
    dataset = dataset.sort_values("entry_ts").reset_index(drop=True)
    dataset = dataset.drop(columns=["entry_ts"])
    return dataset


def feature_column_sets(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    target_columns = {
        "entry_time",
        "exit_time",
        "entry_pos",
        "exit_pos",
        "trade_day",
        "entry_price",
        "exit_price_to_topstep",
        "future_high_max",
        "future_low_min",
        "long_exit_pnl_pts",
        "short_exit_pnl_pts",
        "long_mfe_pts",
        "long_mae_pts",
        "short_mfe_pts",
        "short_mae_pts",
        "long_win",
        "short_win",
    }
    categorical_columns = [column for column in MODEL_CATEGORICAL_COLUMNS if column in dataset.columns]
    feature_columns = [column for column in dataset.columns if column not in target_columns]
    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    return feature_columns, categorical_columns, numeric_columns


def simulate_bracket_trade(
    *,
    side: str,
    entry_price: float,
    entry_pos: int,
    exit_pos: int,
    tp_dist: float,
    sl_dist: float,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    fee_points: float,
) -> dict[str, Any]:
    side_key = str(side or "").strip().upper()
    tp = max(float(tp_dist), 0.25)
    sl = max(float(sl_dist), 0.25)
    if exit_pos <= entry_pos:
        raise ValueError("exit_pos must be greater than entry_pos")

    target_price = entry_price + tp if side_key == "LONG" else entry_price - tp
    stop_price = entry_price - sl if side_key == "LONG" else entry_price + sl

    for pos in range(int(entry_pos) + 1, int(exit_pos) + 1):
        bar_high = float(high_arr[pos])
        bar_low = float(low_arr[pos])
        if side_key == "LONG":
            hit_target = bar_high >= target_price
            hit_stop = bar_low <= stop_price
        else:
            hit_target = bar_low <= target_price
            hit_stop = bar_high >= stop_price

        if hit_target and hit_stop:
            return {
                "exit_pos": int(pos),
                "exit_price": stop_price,
                "exit_reason": "both_hit_same_bar_stop_first",
                "pnl_points_gross": -sl,
                "pnl_points_net": -sl - float(fee_points),
            }
        if hit_target:
            return {
                "exit_pos": int(pos),
                "exit_price": target_price,
                "exit_reason": "target",
                "pnl_points_gross": tp,
                "pnl_points_net": tp - float(fee_points),
            }
        if hit_stop:
            return {
                "exit_pos": int(pos),
                "exit_price": stop_price,
                "exit_reason": "stop",
                "pnl_points_gross": -sl,
                "pnl_points_net": -sl - float(fee_points),
            }

    mark_exit_price = float(close_arr[int(exit_pos)])
    gross = mark_exit_price - entry_price if side_key == "LONG" else entry_price - mark_exit_price
    return {
        "exit_pos": int(exit_pos),
        "exit_price": mark_exit_price,
        "exit_reason": "topstep_exit",
        "pnl_points_gross": float(gross),
        "pnl_points_net": float(gross) - float(fee_points),
    }

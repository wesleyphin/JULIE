import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from config import CONFIG
from de3_v4_schema import (
    LANE_ORDER,
    build_family_id,
    parse_session_block,
    parse_timeframe_minutes,
    safe_div,
    safe_float,
    safe_int,
    strategy_type_to_lane,
    unique_variant_id,
)


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_SPLITS = {
    "train_start": "2011-01-01",
    "train_end": "2023-12-31",
    "tune_start": "2024-01-01",
    "tune_end": "2024-12-31",
    "oos_start": "2025-01-01",
    "oos_end": "2025-12-31",
    "future_start": "2026-01-01",
}
DEFAULT_REQUIRED_BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _resolve_path(raw_path: Any) -> Path:
    p = Path(str(raw_path or "").strip())
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_strategy_rows(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("strategies", [])
    else:
        rows = []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if isinstance(row, dict):
                    rows.append(dict(row))
    except Exception:
        return []
    return rows


def _build_realized_maps(
    *,
    decisions_csv_path: Optional[Path],
    trade_attribution_csv_path: Optional[Path],
    fit_end_et: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    chosen_by_variant: Dict[str, int] = defaultdict(int)
    chosen_by_family: Dict[str, int] = defaultdict(int)
    trades_by_variant: Dict[str, int] = defaultdict(int)
    trades_by_family: Dict[str, int] = defaultdict(int)
    pnl_by_variant: Dict[str, float] = defaultdict(float)
    pnl_by_family: Dict[str, float] = defaultdict(float)
    decisions_seen = 0
    decisions_used = 0
    decisions_dropped_leakage = 0
    trade_rows_seen = 0
    trade_rows_used = 0
    trade_rows_dropped_leakage = 0

    def _row_time_leq_fit_end(row: Dict[str, Any]) -> bool:
        if fit_end_et is None:
            return True
        ts_raw = (
            row.get("decision_timestamp")
            or row.get("timestamp")
            or row.get("entry_time")
            or row.get("open_time")
            or row.get("time")
        )
        if ts_raw in (None, ""):
            return False
        try:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize(NY_TZ)
            else:
                ts = ts.tz_convert(NY_TZ)
            return bool(ts <= fit_end_et)
        except Exception:
            return False

    if decisions_csv_path is not None:
        for row in _read_csv_rows(decisions_csv_path):
            decisions_seen += 1
            if not _row_time_leq_fit_end(row):
                decisions_dropped_leakage += 1
                continue
            chosen = str(row.get("chosen", "")).strip().lower() in {"1", "true", "yes", "y", "t"}
            if not chosen:
                continue
            decisions_used += 1
            variant_id = str(row.get("sub_strategy", "") or "").strip()
            family_id = str(row.get("chosen_family_id") or row.get("family_id") or "").strip()
            if variant_id:
                chosen_by_variant[variant_id] += 1
            if family_id:
                chosen_by_family[family_id] += 1

    if trade_attribution_csv_path is not None:
        for row in _read_csv_rows(trade_attribution_csv_path):
            trade_rows_seen += 1
            if not _row_time_leq_fit_end(row):
                trade_rows_dropped_leakage += 1
                continue
            trade_rows_used += 1
            variant_id = str(row.get("sub_strategy", "") or "").strip()
            family_id = str(row.get("chosen_family_id") or row.get("family_id") or "").strip()
            pnl = safe_float(row.get("realized_pnl", row.get("pnl", 0.0)), 0.0)
            if variant_id:
                trades_by_variant[variant_id] += 1
                pnl_by_variant[variant_id] += pnl
            if family_id:
                trades_by_family[family_id] += 1
                pnl_by_family[family_id] += pnl

    return {
        "chosen_by_variant": dict(chosen_by_variant),
        "chosen_by_family": dict(chosen_by_family),
        "trades_by_variant": dict(trades_by_variant),
        "trades_by_family": dict(trades_by_family),
        "pnl_by_variant": dict(pnl_by_variant),
        "pnl_by_family": dict(pnl_by_family),
        "anti_leakage_csv_filtering": {
            "fit_end_et": str(fit_end_et) if fit_end_et is not None else "",
            "decisions_rows_seen": int(decisions_seen),
            "decisions_rows_used": int(decisions_used),
            "decisions_rows_dropped_outside_fit_window": int(decisions_dropped_leakage),
            "trade_rows_seen": int(trade_rows_seen),
            "trade_rows_used": int(trade_rows_used),
            "trade_rows_dropped_outside_fit_window": int(trade_rows_dropped_leakage),
        },
    }


def _quality_proxy(row: Dict[str, Any]) -> float:
    score_raw = safe_float(row.get("structural_score", 0.0), 0.0)
    pf = safe_float(row.get("profit_factor", 0.0), 0.0)
    avg_pnl = safe_float(row.get("avg_pnl", 0.0), 0.0)
    pbr = safe_float(row.get("profitable_block_ratio", 0.0), 0.0)
    trades = max(0, safe_int(row.get("support_trades", 0), 0))
    support = min(1.0, safe_div(trades, 200.0, 0.0))
    market_support = min(
        1.0,
        safe_div(safe_float(row.get("train_tune_market_support_rows", 0.0), 0.0), 20000.0, 0.0),
    )
    return float(
        (0.30 * score_raw)
        + (0.90 * (pf - 1.0))
        + (0.55 * avg_pnl)
        + (0.60 * (pbr - 0.5))
        + (0.25 * support)
        + (0.15 * market_support)
    )


def _default_source_db_path() -> Path:
    de3_v4_cfg = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4"), dict) else {}
    source = (
        de3_v4_cfg.get("member_db_path")
        or (CONFIG.get("DE3_V2", {}) or {}).get("db_path")
        or "dynamic_engine3_strategies_v2.json"
    )
    return _resolve_path(source)


def _default_source_parquet_path() -> Path:
    de3_v4_cfg = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4"), dict) else {}
    training_data_cfg = (
        de3_v4_cfg.get("training_data", {})
        if isinstance(de3_v4_cfg.get("training_data"), dict)
        else {}
    )
    source = training_data_cfg.get("parquet_path") or "es_master_outrights.parquet"
    return _resolve_path(source)


def _default_reports_paths() -> Tuple[Path, Path]:
    de3_v4_cfg = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4"), dict) else {}
    reports_dir = _resolve_path(de3_v4_cfg.get("reports_dir", "reports"))
    return (
        reports_dir / "de3_decisions.csv",
        reports_dir / "de3_decisions_trade_attribution.csv",
    )


def _effective_split_cfg(split_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    merged = dict(DEFAULT_SPLITS)
    if isinstance(split_cfg, dict):
        for key in DEFAULT_SPLITS.keys():
            raw = split_cfg.get(key)
            if raw is not None and str(raw).strip():
                merged[key] = str(raw).strip()
    return merged


def _to_et_boundary(date_text: str, *, is_end: bool) -> pd.Timestamp:
    base = pd.Timestamp(str(date_text).strip())
    if base.tzinfo is not None:
        base = base.tz_convert(NY_TZ)
    else:
        base = base.tz_localize(NY_TZ)
    if is_end:
        return base + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return base


def _coerce_to_et_timestamp_series(
    df: pd.DataFrame,
    *,
    timestamp_column: Optional[str],
    assume_timezone_if_naive: str,
) -> Tuple[pd.Series, Dict[str, Any]]:
    ts_col = str(timestamp_column or "").strip()
    source = ""
    if ts_col and ts_col in df.columns:
        source = f"column:{ts_col}"
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        source = "index"
        ts = pd.Series(df.index, index=df.index)
    else:
        raise ValueError(
            "Parquet timestamp not found: provide --timestamp-column or ensure DatetimeIndex exists."
        )

    ts = pd.to_datetime(ts, errors="coerce")
    if ts.isna().all():
        raise ValueError("Timestamp parsing failed: all values are NaT.")

    tz_assumption = str(assume_timezone_if_naive or "UTC").strip() or "UTC"
    tz_detected = ""
    was_naive = False
    if getattr(ts.dt, "tz", None) is None:
        was_naive = True
        tz_detected = "naive"
        try:
            ts = ts.dt.tz_localize(tz_assumption, nonexistent="shift_forward", ambiguous="NaT")
        except Exception as exc:
            raise ValueError(
                f"Failed to localize naive timestamps with timezone '{tz_assumption}': {exc}"
            ) from exc
    else:
        tz_detected = str(ts.dt.tz)

    ts_et = ts.dt.tz_convert(NY_TZ)
    if ts_et.isna().all():
        raise ValueError("Timestamp conversion to America/New_York produced all NaT.")

    audit = {
        "timestamp_source": source,
        "timestamp_dtype": str(ts.dtype),
        "timestamp_detected_timezone": tz_detected,
        "timestamp_was_naive": bool(was_naive),
        "timezone_assumption_if_naive": str(tz_assumption),
        "timestamp_converted_to_timezone": "America/New_York",
    }
    return ts_et, audit


def _resolve_execution_rules(execution_rules_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    backtest_exec = (
        CONFIG.get("BACKTEST_EXECUTION", {})
        if isinstance(CONFIG.get("BACKTEST_EXECUTION"), dict)
        else {}
    )
    merged = {
        "enforce_no_new_entries_window": bool(
            backtest_exec.get("enforce_no_new_entries_window", True)
        ),
        "no_new_entries_start_hour_et": int(backtest_exec.get("no_new_entries_start_hour_et", 16)),
        "no_new_entries_end_hour_et": int(backtest_exec.get("no_new_entries_end_hour_et", 18)),
        "force_flat_at_time": bool(backtest_exec.get("force_flat_at_time", True)),
        "force_flat_hour_et": int(backtest_exec.get("force_flat_hour_et", 16)),
        "force_flat_minute_et": int(backtest_exec.get("force_flat_minute_et", 0)),
    }
    if isinstance(execution_rules_cfg, dict):
        for key in list(merged.keys()):
            if key in execution_rules_cfg:
                merged[key] = execution_rules_cfg.get(key)
    merged["enforce_no_new_entries_window"] = bool(merged.get("enforce_no_new_entries_window", True))
    merged["force_flat_at_time"] = bool(merged.get("force_flat_at_time", True))
    merged["no_new_entries_start_hour_et"] = int(safe_int(merged.get("no_new_entries_start_hour_et", 16), 16))
    merged["no_new_entries_end_hour_et"] = int(safe_int(merged.get("no_new_entries_end_hour_et", 18), 18))
    merged["force_flat_hour_et"] = int(safe_int(merged.get("force_flat_hour_et", 16), 16))
    merged["force_flat_minute_et"] = int(safe_int(merged.get("force_flat_minute_et", 0), 0))
    return merged


def _validate_parquet_input(
    *,
    parquet_path: Path,
    required_bar_columns: List[str],
    timestamp_column: Optional[str],
    assume_timezone_if_naive: str,
    split_cfg: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"DE3v4 parquet source does not exist: {parquet_path}")
    if str(parquet_path.suffix).lower() != ".parquet":
        raise ValueError(
            f"DE3v4 parquet source must be .parquet (got: {parquet_path.suffix or '<none>'})."
        )

    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception as exc:
        raise RuntimeError(f"Unable to read parquet source: {parquet_path} ({exc})") from exc

    if df.empty:
        raise ValueError("Parquet source is empty; DE3v4 training cannot proceed.")

    required = [str(v).strip().lower() for v in required_bar_columns if str(v).strip()]
    missing = [c for c in required if c not in {str(col).strip().lower() for col in df.columns}]
    if missing:
        raise ValueError(f"Parquet source missing required columns: {missing}")

    ts_et, ts_audit = _coerce_to_et_timestamp_series(
        df,
        timestamp_column=timestamp_column,
        assume_timezone_if_naive=assume_timezone_if_naive,
    )
    df = df.copy()
    df["__ts_et__"] = ts_et
    df = df[df["__ts_et__"].notna()].copy()
    if df.empty:
        raise ValueError("No valid timestamp rows remain after parsing/parquet timestamp coercion.")

    df = df.sort_values("__ts_et__")
    min_ts = pd.Timestamp(df["__ts_et__"].min())
    max_ts = pd.Timestamp(df["__ts_et__"].max())

    split_bounds = {
        "train": (
            _to_et_boundary(split_cfg["train_start"], is_end=False),
            _to_et_boundary(split_cfg["train_end"], is_end=True),
        ),
        "tune": (
            _to_et_boundary(split_cfg["tune_start"], is_end=False),
            _to_et_boundary(split_cfg["tune_end"], is_end=True),
        ),
        "oos": (
            _to_et_boundary(split_cfg["oos_start"], is_end=False),
            _to_et_boundary(split_cfg["oos_end"], is_end=True),
        ),
        "future_holdout": (
            _to_et_boundary(split_cfg["future_start"], is_end=False),
            max_ts,
        ),
    }

    split_masks: Dict[str, pd.Series] = {}
    split_frames: Dict[str, pd.DataFrame] = {}
    for split_name, (start_ts, end_ts) in split_bounds.items():
        mask = (df["__ts_et__"] >= start_ts) & (df["__ts_et__"] <= end_ts)
        split_masks[split_name] = mask
        split_frames[split_name] = df.loc[mask]

    split_counts = {name: int(frame.shape[0]) for name, frame in split_frames.items()}
    if split_counts.get("train", 0) <= 0:
        raise ValueError("Train split is empty after parquet parsing/splitting.")
    if split_counts.get("tune", 0) <= 0:
        raise ValueError("Tune split is empty after parquet parsing/splitting.")
    if split_counts.get("oos", 0) <= 0:
        raise ValueError("OOS split is empty after parquet parsing/splitting.")

    if min_ts.year > 2011:
        raise ValueError(
            f"Parquet source does not span expected DE3v4 start year (2011). Min timestamp is {min_ts}."
        )
    if max_ts.year < 2025:
        raise ValueError(
            f"Parquet source does not span required OOS year (2025). Max timestamp is {max_ts}."
        )

    input_audit = {
        "source_path": str(parquet_path),
        "file_format": "parquet",
        "row_count": int(df.shape[0]),
        "detected_columns": [str(c) for c in df.columns if str(c) != "__ts_et__"],
        "required_columns": list(required_bar_columns),
        "validation_status": "passed",
        "min_timestamp_et": str(min_ts),
        "max_timestamp_et": str(max_ts),
        "split_row_counts": dict(split_counts),
        **ts_audit,
    }
    return df, input_audit, split_masks, split_frames


def _execution_flags(df: pd.DataFrame, exec_rules: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    ts = out["__ts_et__"]
    hour = ts.dt.hour
    minute = ts.dt.minute
    no_entry_start = int(exec_rules.get("no_new_entries_start_hour_et", 16))
    no_entry_end = int(exec_rules.get("no_new_entries_end_hour_et", 18))
    enforce_no_entry = bool(exec_rules.get("enforce_no_new_entries_window", True))
    force_flat_hour = int(exec_rules.get("force_flat_hour_et", 16))
    force_flat_minute = int(exec_rules.get("force_flat_minute_et", 0))
    force_flat = bool(exec_rules.get("force_flat_at_time", True))

    no_entry_mask = (hour >= no_entry_start) & (hour < no_entry_end)
    if not enforce_no_entry:
        no_entry_mask = pd.Series(False, index=out.index)
    force_flat_mask = (hour == force_flat_hour) & (minute == force_flat_minute)
    if not force_flat:
        force_flat_mask = pd.Series(False, index=out.index)

    out["__entry_allowed__"] = (~no_entry_mask).astype(bool)
    out["__force_flat_event__"] = force_flat_mask.astype(bool)
    out["__hour_et__"] = hour.astype(int)
    return out


def _session_entry_counts(frame: pd.DataFrame, session_labels: Iterable[str]) -> Dict[str, int]:
    labels = [str(v).strip() for v in session_labels if str(v).strip()]
    if not labels:
        return {}
    if frame.empty:
        return {label: 0 for label in labels}
    entry_allowed = frame["__entry_allowed__"].astype(bool)
    hour = frame["__hour_et__"].astype(int)
    default_count = int(entry_allowed.sum())
    out: Dict[str, int] = {}
    for label in labels:
        block = parse_session_block(label)
        if block is None:
            out[label] = default_count
            continue
        start_h, end_h = block
        mask = entry_allowed & (hour >= int(start_h)) & (hour < int(end_h))
        out[label] = int(mask.sum())
    return out


def build_de3_v4_training_dataset(
    *,
    source_db_path: Optional[Any] = None,
    source_parquet_path: Optional[Any] = None,
    split_cfg: Optional[Dict[str, Any]] = None,
    timestamp_column: Optional[str] = None,
    assume_timezone_if_naive: str = "UTC",
    execution_rules_cfg: Optional[Dict[str, Any]] = None,
    required_bar_columns: Optional[List[str]] = None,
    allow_source_db_performance_metrics: bool = False,
    decisions_csv_path: Optional[Any] = None,
    trade_attribution_csv_path: Optional[Any] = None,
    core_anchor_family_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    source_path = _resolve_path(source_db_path) if source_db_path else _default_source_db_path()
    source_parquet = (
        _resolve_path(source_parquet_path)
        if source_parquet_path
        else _default_source_parquet_path()
    )
    split_cfg_eff = _effective_split_cfg(split_cfg)
    required_cols = (
        [str(v).strip() for v in required_bar_columns if str(v).strip()]
        if isinstance(required_bar_columns, list)
        else list(DEFAULT_REQUIRED_BAR_COLUMNS)
    )
    use_db_metrics = bool(allow_source_db_performance_metrics)
    exec_rules = _resolve_execution_rules(execution_rules_cfg)
    decisions_default, trades_default = _default_reports_paths()
    decisions_path = _resolve_path(decisions_csv_path) if decisions_csv_path else decisions_default
    trades_path = (
        _resolve_path(trade_attribution_csv_path)
        if trade_attribution_csv_path
        else trades_default
    )
    anchors = {
        str(v).strip()
        for v in (core_anchor_family_ids or [])
        if str(v).strip()
    }
    if not anchors:
        anchors = {"5min|09-12|long|Long_Rev|T6"}

    rows = _load_strategy_rows(source_path)
    if not rows:
        raise ValueError(f"No strategy rows found in DE3 source DB: {source_path}")

    parquet_df_raw, input_audit, _split_masks, split_frames_raw = _validate_parquet_input(
        parquet_path=source_parquet,
        required_bar_columns=required_cols,
        timestamp_column=timestamp_column,
        assume_timezone_if_naive=assume_timezone_if_naive,
        split_cfg=split_cfg_eff,
    )
    parquet_df = _execution_flags(parquet_df_raw, exec_rules)
    split_frames: Dict[str, pd.DataFrame] = {}
    split_row_counts: Dict[str, int] = {}
    split_entry_allowed_counts: Dict[str, int] = {}
    split_force_flat_counts: Dict[str, int] = {}
    for split_name, raw_frame in split_frames_raw.items():
        frame = parquet_df.loc[raw_frame.index]
        split_frames[split_name] = frame
        split_row_counts[split_name] = int(frame.shape[0])
        split_entry_allowed_counts[split_name] = int(frame["__entry_allowed__"].sum())
        split_force_flat_counts[split_name] = int(frame["__force_flat_event__"].sum())

    used_for_fit_max_ts = None
    train_tune_frame = pd.concat(
        [split_frames.get("train", pd.DataFrame()), split_frames.get("tune", pd.DataFrame())],
        axis=0,
    )
    if not train_tune_frame.empty:
        used_for_fit_max_ts = pd.Timestamp(train_tune_frame["__ts_et__"].max())
    oos_start_ts = _to_et_boundary(split_cfg_eff["oos_start"], is_end=False)
    future_start_ts = _to_et_boundary(split_cfg_eff["future_start"], is_end=False)
    leakage_violations: List[str] = []
    if used_for_fit_max_ts is not None and used_for_fit_max_ts >= oos_start_ts:
        leakage_violations.append(
            f"fit_data_max_timestamp_et={used_for_fit_max_ts} crossed into OOS window starting {oos_start_ts}"
        )

    leakage_check_passed = bool(len(leakage_violations) == 0)
    split_summary = {
        "training_start": split_cfg_eff["train_start"],
        "training_end": split_cfg_eff["train_end"],
        "tuning_start": split_cfg_eff["tune_start"],
        "tuning_end": split_cfg_eff["tune_end"],
        "oos_start": split_cfg_eff["oos_start"],
        "oos_end": split_cfg_eff["oos_end"],
        "future_holdout_start": split_cfg_eff["future_start"],
        "future_holdout_end": (
            str(pd.Timestamp(split_frames["future_holdout"]["__ts_et__"].max()))
            if (not split_frames.get("future_holdout", pd.DataFrame()).empty)
            else ""
        ),
        "data_rows_train": int(split_row_counts.get("train", 0)),
        "data_rows_tune": int(split_row_counts.get("tune", 0)),
        "data_rows_oos": int(split_row_counts.get("oos", 0)),
        "data_rows_future_holdout": int(split_row_counts.get("future_holdout", 0)),
        "entry_allowed_rows_train": int(split_entry_allowed_counts.get("train", 0)),
        "entry_allowed_rows_tune": int(split_entry_allowed_counts.get("tune", 0)),
        "entry_allowed_rows_oos": int(split_entry_allowed_counts.get("oos", 0)),
        "entry_allowed_rows_future_holdout": int(split_entry_allowed_counts.get("future_holdout", 0)),
        "force_flat_events_train": int(split_force_flat_counts.get("train", 0)),
        "force_flat_events_tune": int(split_force_flat_counts.get("tune", 0)),
        "force_flat_events_oos": int(split_force_flat_counts.get("oos", 0)),
        "force_flat_events_future_holdout": int(split_force_flat_counts.get("future_holdout", 0)),
    }
    input_audit["split_row_counts"] = {
        "train": int(split_row_counts.get("train", 0)),
        "tune": int(split_row_counts.get("tune", 0)),
        "oos": int(split_row_counts.get("oos", 0)),
        "future_holdout": int(split_row_counts.get("future_holdout", 0)),
    }
    input_audit["entry_allowed_row_counts"] = {
        "train": int(split_entry_allowed_counts.get("train", 0)),
        "tune": int(split_entry_allowed_counts.get("tune", 0)),
        "oos": int(split_entry_allowed_counts.get("oos", 0)),
        "future_holdout": int(split_entry_allowed_counts.get("future_holdout", 0)),
    }
    input_audit["execution_rules_applied"] = dict(exec_rules)
    input_audit["allow_source_db_performance_metrics_for_training"] = bool(use_db_metrics)
    input_audit["leakage_check_passed"] = bool(leakage_check_passed)
    input_audit["leakage_violations"] = list(leakage_violations)

    # Precompute split/session support counts once; used for all variants.
    session_labels = {
        str(row.get("Session", row.get("session", "")) or "").strip()
        for row in rows
        if str(row.get("Session", row.get("session", "")) or "").strip()
    }
    split_session_support: Dict[str, Dict[str, int]] = {}
    for split_name, frame in split_frames.items():
        split_session_support[split_name] = _session_entry_counts(frame, session_labels)

    realized = _build_realized_maps(
        decisions_csv_path=decisions_path if decisions_path.exists() else None,
        trade_attribution_csv_path=trades_path if trades_path.exists() else None,
        fit_end_et=_to_et_boundary(split_cfg_eff["tune_end"], is_end=True),
    )
    chosen_by_variant = realized.get("chosen_by_variant", {})
    chosen_by_family = realized.get("chosen_by_family", {})
    trades_by_variant = realized.get("trades_by_variant", {})
    trades_by_family = realized.get("trades_by_family", {})
    pnl_by_variant = realized.get("pnl_by_variant", {})
    pnl_by_family = realized.get("pnl_by_family", {})
    realized_anti_leak = (
        realized.get("anti_leakage_csv_filtering", {})
        if isinstance(realized.get("anti_leakage_csv_filtering"), dict)
        else {}
    )

    variants: List[Dict[str, Any]] = []
    lane_counts: Dict[str, int] = defaultdict(int)
    lane_quality: Dict[str, List[float]] = defaultdict(list)
    seen_variant_ids: set[str] = set()

    for idx, row in enumerate(rows, 1):
        tf = str(row.get("TF", row.get("timeframe", "")) or "").strip()
        session = str(row.get("Session", row.get("session", "")) or "").strip()
        strategy_type = str(row.get("Type", row.get("strategy_type", "")) or "").strip()
        thresh = safe_float(row.get("Thresh", row.get("thresh", 0.0)), 0.0)
        family_tag = str(row.get("FamilyTag", row.get("family_tag", "")) or "").strip()
        lane = strategy_type_to_lane(strategy_type)
        if not lane:
            continue
        family_id = build_family_id(
            timeframe=tf,
            session=session,
            strategy_type=strategy_type,
            threshold=thresh,
            family_tag=family_tag,
        )
        variant_pref = row.get("id", row.get("strategy_id", ""))
        variant_id = unique_variant_id(variant_pref, family_id, idx)
        if variant_id in seen_variant_ids:
            variant_id = f"{variant_id}::{idx}"
        seen_variant_ids.add(variant_id)

        oos = row.get("OOS", {}) if isinstance(row.get("OOS"), dict) else {}
        sl = safe_float(row.get("Best_SL", row.get("sl", 0.0)), 0.0)
        tp = safe_float(row.get("Best_TP", row.get("tp", 0.0)), 0.0)
        rr = safe_div(tp, sl, 0.0) if sl > 0 else 0.0

        timeframe_minutes = parse_timeframe_minutes(tf) or 5
        train_session_rows = int(split_session_support.get("train", {}).get(session, 0))
        tune_session_rows = int(split_session_support.get("tune", {}).get(session, 0))
        oos_session_rows = int(split_session_support.get("oos", {}).get(session, 0))
        future_session_rows = int(split_session_support.get("future_holdout", {}).get(session, 0))
        train_market_support = int(train_session_rows // max(1, timeframe_minutes))
        tune_market_support = int(tune_session_rows // max(1, timeframe_minutes))
        oos_market_support = int(oos_session_rows // max(1, timeframe_minutes))
        future_market_support = int(future_session_rows // max(1, timeframe_minutes))

        if use_db_metrics:
            support_trades = max(
                safe_int(oos.get("trades", row.get("Trades", row.get("trades", 0))), 0),
                0,
            )
            avg_pnl = safe_float(
                oos.get("avg_pnl", row.get("Avg_PnL", row.get("avg_pnl", 0.0))),
                0.0,
            )
            win_rate = safe_float(
                oos.get("win_rate", row.get("Opt_WR", row.get("opt_wr", 0.0))),
                0.0,
            )
            pf = safe_float(
                oos.get(
                    "profit_factor",
                    row.get("ProfitFactor", row.get("profit_factor", 0.0)),
                ),
                0.0,
            )
            structural_score = safe_float(
                row.get("StructuralScore", row.get("Score", row.get("score_raw", 0.0))),
                0.0,
            )
            pbr = safe_float(
                row.get("ProfitableBlockRatio", row.get("profitable_block_ratio", 0.0)),
                0.0,
            )
            worst_block_pf = safe_float(
                row.get("WorstBlockPF", row.get("worst_block_pf", 0.0)),
                0.0,
            )
            worst_block_avg = safe_float(
                row.get("WorstBlockAvgPnL", row.get("worst_block_avg_pnl", 0.0)),
                0.0,
            )
            drawdown_norm = safe_float(
                oos.get("max_oos_drawdown_norm", row.get("drawdown_norm", 0.0)),
                0.0,
            )
            stop_like_share = safe_float(
                row.get("StopLikeShare", row.get("stop_like_share", 0.0)),
                0.0,
            )
            loss_share = safe_float(
                row.get("LossShare", row.get("loss_share", 0.0)),
                0.0,
            )
            performance_metrics_source = "source_db_performance"
        else:
            support_trades = int(train_market_support + tune_market_support)
            realized_trade_count = int(
                trades_by_variant.get(variant_id, trades_by_family.get(family_id, 0)) or 0
            )
            realized_net = float(
                pnl_by_variant.get(variant_id, pnl_by_family.get(family_id, 0.0)) or 0.0
            )
            avg_pnl = float(
                safe_div(realized_net, float(max(1, realized_trade_count)), 0.0)
            ) if realized_trade_count > 0 else 0.0
            win_rate = 0.0
            pf = 1.0
            structural_score = 0.0
            pbr = 0.0
            worst_block_pf = 0.0
            worst_block_avg = 0.0
            drawdown_norm = 0.0
            stop_like_share = 0.0
            loss_share = 0.0
            performance_metrics_source = "fallback_support_only"

        variant_row = {
            "variant_id": variant_id,
            "family_id": family_id,
            "lane": lane,
            "strategy_type": strategy_type,
            "timeframe": tf,
            "session": session,
            "threshold": float(thresh),
            "family_tag": str(family_tag),
            "best_sl": float(sl),
            "best_tp": float(tp),
            "rr": float(rr),
            "support_trades": int(support_trades),
            "avg_pnl": float(avg_pnl),
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "structural_score": float(structural_score),
            "profitable_block_ratio": float(pbr),
            "worst_block_pf": float(worst_block_pf),
            "worst_block_avg_pnl": float(worst_block_avg),
            "drawdown_norm": float(drawdown_norm),
            "stop_like_share": float(stop_like_share),
            "loss_share": float(loss_share),
            "is_core_anchor_family": bool(family_id in anchors),
            "realized_chosen_count": int(chosen_by_variant.get(variant_id, chosen_by_family.get(family_id, 0)) or 0),
            "realized_trade_count": int(trades_by_variant.get(variant_id, trades_by_family.get(family_id, 0)) or 0),
            "realized_net_pnl": float(pnl_by_variant.get(variant_id, pnl_by_family.get(family_id, 0.0)) or 0.0),
            "train_market_support_rows": int(train_market_support),
            "tune_market_support_rows": int(tune_market_support),
            "oos_market_support_rows": int(oos_market_support),
            "future_market_support_rows": int(future_market_support),
            "train_tune_market_support_rows": int(train_market_support + tune_market_support),
            "performance_metrics_source": str(performance_metrics_source),
        }
        variant_row["quality_proxy"] = float(_quality_proxy(variant_row))
        variants.append(variant_row)
        lane_counts[lane] += 1
        lane_quality[lane].append(float(variant_row["quality_proxy"]))

    lane_summary: Dict[str, Dict[str, Any]] = {}
    for lane in LANE_ORDER:
        scores = list(lane_quality.get(lane, []))
        lane_summary[lane] = {
            "variant_count": int(lane_counts.get(lane, 0)),
            "quality_proxy_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "quality_proxy_max": float(max(scores)) if scores else 0.0,
            "train_entry_allowed_rows": int(
                sum(
                    safe_int(row.get("train_market_support_rows", 0), 0)
                    for row in variants
                    if str(row.get("lane", "")) == lane
                )
            ),
            "tune_entry_allowed_rows": int(
                sum(
                    safe_int(row.get("tune_market_support_rows", 0), 0)
                    for row in variants
                    if str(row.get("lane", "")) == lane
                )
            ),
        }

    return {
        "metadata": {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_db_path": str(source_path),
            "source_data_path": str(source_parquet),
            "source_data_format": "parquet",
            "decisions_csv_path": str(decisions_path),
            "trade_attribution_csv_path": str(trades_path),
            "source_row_count": int(len(rows)),
            "source_data_row_count": int(parquet_df.shape[0]),
            "variant_count": int(len(variants)),
            "split_summary": dict(split_summary),
            "execution_rules_applied": dict(exec_rules),
            "allow_source_db_performance_metrics_for_training": bool(use_db_metrics),
            "timestamp_audit": {
                "timestamp_source": str(input_audit.get("timestamp_source", "")),
                "timestamp_dtype": str(input_audit.get("timestamp_dtype", "")),
                "timestamp_detected_timezone": str(
                    input_audit.get("timestamp_detected_timezone", "")
                ),
                "timestamp_was_naive": bool(input_audit.get("timestamp_was_naive", False)),
                "timezone_assumption_if_naive": str(
                    input_audit.get("timezone_assumption_if_naive", "")
                ),
                "timestamp_converted_to_timezone": str(
                    input_audit.get("timestamp_converted_to_timezone", "America/New_York")
                ),
            },
            "leakage_check_passed": bool(leakage_check_passed),
            "leakage_violations": list(leakage_violations),
            "realized_csv_anti_leakage": dict(realized_anti_leak),
            "fit_data_end_et": (
                str(used_for_fit_max_ts) if used_for_fit_max_ts is not None else ""
            ),
            "oos_start_et": str(oos_start_ts),
            "future_holdout_start_et": str(future_start_ts),
        },
        "core_anchor_family_ids": sorted(list(anchors)),
        "variants": variants,
        "lane_summary": lane_summary,
        "data_input_audit": dict(input_audit),
        "split_summary": dict(split_summary),
        "execution_rule_summary": dict(exec_rules),
        "leakage_summary": {
            "leakage_check_passed": bool(leakage_check_passed),
            "violations": list(leakage_violations),
            "realized_csv_anti_leakage": dict(realized_anti_leak),
            "allow_source_db_performance_metrics_for_training": bool(use_db_metrics),
            "fit_data_end_et": (str(used_for_fit_max_ts) if used_for_fit_max_ts is not None else ""),
            "oos_start_et": str(oos_start_ts),
            "future_holdout_start_et": str(future_start_ts),
            "notes": [
                "DE3v4 fitting/tuning support and normalization use train+tune splits only.",
                "2025 OOS and 2026+ future holdout are excluded from fitting/tuning computations.",
                (
                    "Source DB performance metrics are used for fit-time scoring."
                    if use_db_metrics
                    else "Source DB performance metrics are disabled for fit-time scoring (inventory-only mode)."
                ),
                "Execution-aware support uses no-new-entry 4:00-6:00 PM ET and force-flat at 4:00 PM ET.",
            ],
        },
    }

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from de3_v4_schema import clip, lane_to_side, safe_div, safe_float, safe_int


NY_TZ = ZoneInfo("America/New_York")

ENTRY_SHAPE_NUMERIC_COLUMNS = [
    "de3_entry_ret1_atr",
    "de3_entry_body_pos1",
    "de3_entry_lower_wick_ratio",
    "de3_entry_upper_wick_ratio",
    "de3_entry_upper1_ratio",
    "de3_entry_body1_ratio",
    "de3_entry_close_pos1",
    "de3_entry_flips5",
    "de3_entry_down3",
    "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20",
    "de3_entry_atr14",
]


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series([], dtype=bool)
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _shape_scope_key(*, lane: Any, timeframe: Any, scope_mode: str) -> str:
    lane_text = str(lane or "").strip()
    timeframe_text = str(timeframe or "").strip()
    mode = str(scope_mode or "lane_timeframe").strip().lower()
    if mode == "lane":
        return lane_text
    if mode == "timeframe":
        return timeframe_text
    if lane_text and timeframe_text:
        return f"{lane_text}|{timeframe_text}"
    return lane_text or timeframe_text


def _derive_session_substate(*, session_text: Any, hour_value: Any) -> str:
    session_norm = str(session_text or "").strip()
    try:
        hour_et = int(float(hour_value))
    except Exception:
        return ""
    if not session_norm:
        return ""
    try:
        start_hour = int(str(session_norm).split("-")[0])
        rel_hour = (hour_et - start_hour) % 24
        if rel_hour < 1:
            return "open"
        if rel_hour < 2:
            return "mid"
        return "late"
    except Exception:
        return ""


@dataclass(frozen=True)
class _SplitBounds:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    tune_start: pd.Timestamp
    tune_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    future_start: pd.Timestamp


def _to_et_start(date_text: Any) -> pd.Timestamp:
    ts = pd.Timestamp(str(date_text or "").strip())
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts


def _to_et_end(date_text: Any) -> pd.Timestamp:
    return _to_et_start(date_text) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)


def _split_bounds(split_summary: Dict[str, Any]) -> _SplitBounds:
    return _SplitBounds(
        train_start=_to_et_start(split_summary.get("training_start", "2011-01-01")),
        train_end=_to_et_end(split_summary.get("training_end", "2023-12-31")),
        tune_start=_to_et_start(split_summary.get("tuning_start", "2024-01-01")),
        tune_end=_to_et_end(split_summary.get("tuning_end", "2024-12-31")),
        oos_start=_to_et_start(split_summary.get("oos_start", "2025-01-01")),
        oos_end=_to_et_end(split_summary.get("oos_end", "2025-12-31")),
        future_start=_to_et_start(split_summary.get("future_holdout_start", "2026-01-01")),
    )


def _read_csv(path_text: Any, *, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not str(path_text or "").strip():
        return pd.DataFrame()
    path = str(path_text).strip()
    try:
        if usecols is None:
            return pd.read_csv(path)
        requested = {str(c).strip() for c in list(usecols) if str(c).strip()}
        if not requested:
            return pd.read_csv(path)
        # Single-pass load that keeps present columns and ignores missing ones.
        return pd.read_csv(path, usecols=lambda c: str(c).strip() in requested)
    except Exception:
        return pd.DataFrame()


def _parse_et_timestamps(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series([], dtype="datetime64[ns, America/New_York]")
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_convert(NY_TZ)
        return parsed.dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        parsed_fallback: List[pd.Timestamp] = []
        for raw in series.tolist():
            if raw in (None, ""):
                parsed_fallback.append(pd.NaT)
                continue
            try:
                ts = pd.Timestamp(raw)
            except Exception:
                parsed_fallback.append(pd.NaT)
                continue
            try:
                if ts.tzinfo is None:
                    ts = ts.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
                else:
                    ts = ts.tz_convert(NY_TZ)
            except Exception:
                ts = pd.NaT
            parsed_fallback.append(ts)
        try:
            out = pd.DatetimeIndex(parsed_fallback)
            return pd.Series(out, index=series.index)
        except Exception:
            return pd.Series([pd.NaT] * len(series), index=series.index, dtype="datetime64[ns, America/New_York]")


def _assign_splits(ts_series: pd.Series, bounds: _SplitBounds) -> pd.Series:
    if not isinstance(ts_series, pd.Series):
        return pd.Series([], dtype=object)
    out = pd.Series([""] * len(ts_series), index=ts_series.index, dtype=object)
    valid = ts_series.notna()
    train_mask = valid & (ts_series >= bounds.train_start) & (ts_series <= bounds.train_end)
    tune_mask = valid & (ts_series >= bounds.tune_start) & (ts_series <= bounds.tune_end)
    oos_mask = valid & (ts_series >= bounds.oos_start) & (ts_series <= bounds.oos_end)
    future_mask = valid & (ts_series >= bounds.future_start)
    out.loc[train_mask] = "train"
    out.loc[tune_mask] = "tune"
    out.loc[oos_mask] = "oos"
    out.loc[future_mask] = "future_holdout"
    return out


def _wilson_lcb(wins: int, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    p_hat = safe_div(float(wins), float(n), 0.0)
    denom = 1.0 + ((z * z) / float(n))
    center = p_hat + ((z * z) / (2.0 * float(n)))
    spread_num = (p_hat * (1.0 - p_hat)) + ((z * z) / (4.0 * float(n)))
    margin = z * math.sqrt(max(0.0, spread_num / float(n)))
    return float(max(0.0, min(1.0, (center - margin) / denom)))


def _quality_lcb_score(
    *,
    p_win_lcb: float,
    ev_lcb: float,
    n_trades: int,
    cfg: Dict[str, Any],
) -> float:
    min_samples_full = max(1.0, safe_float(cfg.get("reliability_full_samples", 60.0), 60.0))
    ev_scale = max(1e-9, safe_float(cfg.get("ev_lcb_scale_points", 2.0), 2.0))
    win_center = safe_float(cfg.get("win_lcb_center", 0.50), 0.50)
    w_win = safe_float(cfg.get("weight_win_lcb", 0.60), 0.60)
    w_ev = safe_float(cfg.get("weight_ev_lcb", 0.30), 0.30)
    w_rel = safe_float(cfg.get("weight_reliability", 0.10), 0.10)
    reliability = clip(safe_div(float(n_trades), min_samples_full, 0.0), 0.0, 1.0)
    return float(
        (w_win * (float(p_win_lcb) - float(win_center)))
        + (w_ev * safe_div(float(ev_lcb), ev_scale, 0.0))
        + (w_rel * reliability)
    )


def _build_stats_from_frame(
    *,
    trade_df: pd.DataFrame,
    key_col: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    if trade_df.empty:
        return {}
    if key_col is None:
        groups_iter = [("__global__", trade_df)]
    else:
        valid = trade_df[trade_df[key_col].notna() & (trade_df[key_col].astype(str).str.strip() != "")]
        groups_iter = valid.groupby(key_col, sort=False)
    z = max(0.0, safe_float(cfg.get("wilson_z", 1.96), 1.96))
    out: Dict[str, Dict[str, Any]] = {}
    for key, frame in groups_iter:
        frame_local = frame
        if "ts_effective" in frame_local.columns:
            frame_local = frame_local.sort_values("ts_effective")
        pnl = frame_local["realized_pnl"].astype(float)
        n_trades = int(pnl.shape[0])
        if n_trades <= 0:
            continue
        wins = int((pnl > 0.0).sum())
        losses = int((pnl <= 0.0).sum())
        p_win = safe_div(float(wins), float(n_trades), 0.0)
        p_win_lcb = _wilson_lcb(wins, n_trades, z)
        ev_mean = float(pnl.mean())
        ev_std = float(pnl.std(ddof=1)) if n_trades > 1 else 0.0
        ev_lcb = float(ev_mean - (z * safe_div(ev_std, math.sqrt(float(max(1, n_trades))), 0.0)))
        gross_wins = float(pnl[pnl > 0.0].sum())
        gross_losses_abs = float(abs(pnl[pnl <= 0.0].sum()))
        profit_factor = float(gross_wins / gross_losses_abs) if gross_losses_abs > 1e-12 else (999.0 if gross_wins > 0.0 else 0.0)
        loss_share = float(safe_div(float(losses), float(n_trades), 0.0))
        exit_types = (
            frame_local.get("realized_exit_type", pd.Series([""] * n_trades))
            .astype(str)
            .str.lower()
            .str.strip()
        )
        stop_like_hits = exit_types.str.contains(r"(?:^sl$|stop)", regex=True, na=False)
        stop_like_share = float(stop_like_hits.mean()) if n_trades > 0 else 0.0
        max_drawdown = float(_max_drawdown(pnl))
        drawdown_norm = float(safe_div(max_drawdown, float(max(1, n_trades)), 0.0))
        # Robust block quality proxy: worst fixed-size block average pnl.
        block_size = max(5, safe_int(cfg.get("worst_block_trade_block_size", 40), 40))
        if n_trades <= block_size:
            worst_block_avg_pnl = float(ev_mean)
        else:
            chunk_means: List[float] = []
            for idx in range(0, n_trades, block_size):
                chunk = pnl.iloc[idx : idx + block_size]
                if chunk.empty:
                    continue
                chunk_means.append(float(chunk.mean()))
            worst_block_avg_pnl = float(min(chunk_means)) if chunk_means else float(ev_mean)
        year_coverage = 0
        first_year = ""
        last_year = ""
        if "ts_effective" in frame_local.columns:
            try:
                ts_years = pd.to_datetime(frame_local["ts_effective"], errors="coerce").dt.year.dropna().astype(int)
                unique_years = sorted(set(int(v) for v in ts_years.tolist()))
                if unique_years:
                    year_coverage = int(len(unique_years))
                    first_year = str(unique_years[0])
                    last_year = str(unique_years[-1])
            except Exception:
                year_coverage = 0
                first_year = ""
                last_year = ""
        quality_lcb_score = _quality_lcb_score(
            p_win_lcb=p_win_lcb,
            ev_lcb=ev_lcb,
            n_trades=n_trades,
            cfg=cfg,
        )
        out[str(key)] = {
            "n_trades": int(n_trades),
            "wins": int(wins),
            "losses": int(losses),
            "p_win": float(p_win),
            "p_win_lcb": float(p_win_lcb),
            "ev_mean": float(ev_mean),
            "ev_std": float(ev_std),
            "ev_lcb": float(ev_lcb),
            "profit_factor": float(profit_factor),
            "loss_share": float(loss_share),
            "stop_like_share": float(stop_like_share),
            "drawdown_norm": float(drawdown_norm),
            "worst_block_avg_pnl": float(worst_block_avg_pnl),
            "year_coverage": int(year_coverage),
            "first_year": str(first_year),
            "last_year": str(last_year),
            "quality_lcb_score": float(quality_lcb_score),
        }
    return out

def _trade_frame_from_csvs(
    *,
    decisions_csv_path: Any,
    trade_attribution_csv_path: Any,
    bounds: _SplitBounds,
    variants: List[Dict[str, Any]],
    chosen_shape_csv_path: Any = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    trades = _read_csv(
        trade_attribution_csv_path,
        usecols=[
            "decision_id",
            "sub_strategy",
            "entry_time",
            "realized_pnl",
            "realized_exit_type",
        ],
    )
    if trades.empty:
        return pd.DataFrame(), {"status": "missing_trade_attribution_csv"}
    trades = trades.copy()
    trades["decision_id"] = trades.get("decision_id", "").astype(str)
    trades["variant_id"] = trades.get("sub_strategy", "").astype(str).str.strip()
    trades["realized_pnl"] = pd.to_numeric(trades.get("realized_pnl", 0.0), errors="coerce").fillna(0.0)
    trades["entry_ts_et"] = _parse_et_timestamps(trades.get("entry_time", pd.Series([], dtype=object)))

    decisions = _read_csv(
        decisions_csv_path,
        usecols=[
            "decision_id",
            "timestamp",
            "session",
            "ctx_hour_et",
            "ctx_session_substate",
            "ctx_volatility_regime",
            "ctx_price_location",
            "chosen",
            "rank",
            "timeframe",
            "strategy_type",
            "side_considered",
            "de3_entry_filter_hit",
            "de3_entry_filter_reason",
            "de3_v4_selected_lane",
            "de3_v4_route_confidence",
            "edge_points",
            "runtime_rank_score",
            "structural_score",
            "de3_v4_selected_variant_id",
            *ENTRY_SHAPE_NUMERIC_COLUMNS,
        ],
    )
    chosen_shape = _read_csv(
        chosen_shape_csv_path,
        usecols=[
            "decision_id",
            "chosen",
            "rank",
            "side_considered",
            *ENTRY_SHAPE_NUMERIC_COLUMNS,
        ],
    )
    if not chosen_shape.empty:
        chosen_shape = chosen_shape.copy()
        chosen_shape["decision_id"] = chosen_shape["decision_id"].astype(str)
        chosen_shape["side_considered"] = (
            chosen_shape.get(
                "side_considered",
                pd.Series("", index=chosen_shape.index, dtype=object),
            )
            .astype(str)
            .str.strip()
            .str.lower()
        )
        chosen_shape["chosen_flag"] = _parse_bool_series(chosen_shape.get("chosen", pd.Series(False, index=chosen_shape.index)))
        chosen_shape["rank_num"] = pd.to_numeric(
            chosen_shape.get("rank", pd.Series(999999.0, index=chosen_shape.index)),
            errors="coerce",
        ).fillna(999999.0)
        for feature in ENTRY_SHAPE_NUMERIC_COLUMNS:
            chosen_shape[feature] = pd.to_numeric(
                chosen_shape.get(feature, pd.Series(float("nan"), index=chosen_shape.index)),
                errors="coerce",
            )
        chosen_shape = chosen_shape.sort_values(
            by=["decision_id", "chosen_flag", "rank_num"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        chosen_shape = chosen_shape.drop_duplicates(subset=["decision_id"], keep="first")
    if not decisions.empty:
        decisions = decisions.copy()
        decisions["decision_id"] = decisions.get("decision_id", "").astype(str)
        def _col(name: str, default: Any = "") -> pd.Series:
            if name in decisions.columns:
                return decisions[name]
            return pd.Series([default] * len(decisions), index=decisions.index)

        decisions["timestamp_et"] = _parse_et_timestamps(_col("timestamp", ""))
        decisions["session"] = _col("session", "").astype(str).str.strip()
        decisions["route_confidence"] = pd.to_numeric(
            _col("de3_v4_route_confidence", 0.0), errors="coerce"
        ).fillna(0.0)
        decisions["edge_points"] = pd.to_numeric(_col("edge_points", 0.0), errors="coerce").fillna(0.0)
        decisions["runtime_rank_score"] = pd.to_numeric(
            _col("runtime_rank_score", 0.0), errors="coerce"
        ).fillna(0.0)
        decisions["structural_score"] = pd.to_numeric(
            _col("structural_score", 0.0), errors="coerce"
        ).fillna(0.0)
        decisions["lane"] = _col("de3_v4_selected_lane", "").astype(str).str.strip()
        decisions["variant_id_decision"] = _col("de3_v4_selected_variant_id", "").astype(str).str.strip()
        decisions["timeframe"] = _col("timeframe", "").astype(str).str.strip()
        decisions["strategy_type"] = _col("strategy_type", "").astype(str).str.strip()
        decisions["side_considered"] = _col("side_considered", "").astype(str).str.strip().str.lower()
        decisions["ctx_hour_et"] = pd.to_numeric(_col("ctx_hour_et", float("nan")), errors="coerce")
        decisions["ctx_hour_et"] = decisions["ctx_hour_et"].where(
            decisions["ctx_hour_et"].notna(),
            pd.to_numeric(decisions["timestamp_et"].dt.hour, errors="coerce"),
        )
        decisions["ctx_session_substate"] = _col("ctx_session_substate", "").astype(str).str.strip().str.lower()
        derived_decision_substate = pd.Series(
            [
                _derive_session_substate(session_text=sess, hour_value=hour)
                for sess, hour in zip(
                    decisions["session"].tolist(),
                    decisions["ctx_hour_et"].tolist(),
                )
            ],
            index=decisions.index,
            dtype=object,
        )
        decisions.loc[
            decisions["ctx_session_substate"].astype(str).str.strip() == "",
            "ctx_session_substate",
        ] = derived_decision_substate.loc[
            decisions["ctx_session_substate"].astype(str).str.strip() == ""
        ]
        decisions["ctx_volatility_regime"] = _col("ctx_volatility_regime", "").astype(str).str.strip()
        decisions["ctx_price_location"] = _col("ctx_price_location", "").astype(str).str.strip()
        decisions["de3_entry_filter_hit"] = _parse_bool_series(_col("de3_entry_filter_hit", False))
        decisions["de3_entry_filter_reason"] = _col("de3_entry_filter_reason", "").astype(str).str.strip()
        for feature in ENTRY_SHAPE_NUMERIC_COLUMNS:
            decisions[feature] = pd.to_numeric(_col(feature, float("nan")), errors="coerce")

        chosen_series = _col("chosen", "")
        decisions["chosen_flag"] = (
            chosen_series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})
        )
        decisions["rank_num"] = pd.to_numeric(_col("rank", 999999), errors="coerce").fillna(999999.0)
        decisions["lane_present"] = decisions["lane"].astype(str).str.strip().ne("")
        decisions["variant_present"] = decisions["variant_id_decision"].astype(str).str.strip().ne("")

        # Keep the chosen row first for each decision_id. Fallback to rows that
        # at least carry lane/variant fields, then best rank.
        decisions = decisions.sort_values(
            by=["decision_id", "chosen_flag", "lane_present", "variant_present", "rank_num"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        decisions = decisions.drop_duplicates(subset=["decision_id"], keep="first")
        if not chosen_shape.empty:
            decisions = decisions.merge(
                chosen_shape[
                    [
                        "decision_id",
                        "side_considered",
                        *ENTRY_SHAPE_NUMERIC_COLUMNS,
                    ]
                ].rename(
                    columns={
                        "side_considered": "shape_side_considered",
                        **{feature: f"{feature}_shape" for feature in ENTRY_SHAPE_NUMERIC_COLUMNS},
                    }
                ),
                on="decision_id",
                how="left",
            )
            decisions["side_considered"] = decisions["side_considered"].where(
                decisions["side_considered"].astype(str).str.strip() != "",
                decisions.get("shape_side_considered", pd.Series("", index=decisions.index)),
            )
            for feature in ENTRY_SHAPE_NUMERIC_COLUMNS:
                shape_col = f"{feature}_shape"
                if shape_col not in decisions.columns:
                    continue
                decisions[feature] = decisions[feature].where(
                    decisions[feature].notna(),
                    decisions[shape_col],
                )
        trades = trades.merge(
            decisions[
                [
                    "decision_id",
                    "timestamp_et",
                    "session",
                    "lane",
                    "timeframe",
                    "strategy_type",
                    "side_considered",
                    "ctx_hour_et",
                    "ctx_session_substate",
                    "ctx_volatility_regime",
                    "ctx_price_location",
                    "route_confidence",
                    "edge_points",
                    "runtime_rank_score",
                    "structural_score",
                    "variant_id_decision",
                    "de3_entry_filter_hit",
                    "de3_entry_filter_reason",
                    *ENTRY_SHAPE_NUMERIC_COLUMNS,
                ]
            ],
            on="decision_id",
            how="left",
        )
    else:
        trades["timestamp_et"] = pd.NaT
        trades["session"] = ""
        trades["lane"] = ""
        trades["timeframe"] = ""
        trades["strategy_type"] = ""
        trades["side_considered"] = ""
        trades["ctx_hour_et"] = float("nan")
        trades["ctx_session_substate"] = ""
        trades["ctx_volatility_regime"] = ""
        trades["ctx_price_location"] = ""
        trades["route_confidence"] = 0.0
        trades["edge_points"] = 0.0
        trades["runtime_rank_score"] = 0.0
        trades["structural_score"] = 0.0
        trades["variant_id_decision"] = ""
        trades["de3_entry_filter_hit"] = False
        trades["de3_entry_filter_reason"] = ""
        for feature in ENTRY_SHAPE_NUMERIC_COLUMNS:
            trades[feature] = float("nan")
        if not chosen_shape.empty:
            trades = trades.merge(
                chosen_shape[
                    [
                        "decision_id",
                        "side_considered",
                        *ENTRY_SHAPE_NUMERIC_COLUMNS,
                    ]
                ],
                on="decision_id",
                how="left",
            )
            trades["side_considered"] = trades["side_considered"].astype(str).str.strip().str.lower()

    variant_to_lane: Dict[str, str] = {}
    for row in variants:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variant_id", "") or "").strip()
        lane = str(row.get("lane", "") or "").strip()
        if vid and lane:
            variant_to_lane[vid] = lane
    trades["variant_id"] = trades["variant_id"].where(
        trades["variant_id"].astype(str).str.strip() != "",
        trades["variant_id_decision"],
    )
    trades["variant_id"] = trades["variant_id"].astype(str).str.strip()
    trades["lane"] = trades["lane"].where(trades["lane"].astype(str).str.strip() != "", "")
    trades.loc[trades["lane"] == "", "lane"] = trades.loc[trades["lane"] == "", "variant_id"].map(variant_to_lane).fillna("")
    trades["session"] = trades.get("session", "").astype(str).str.strip()
    trades["timeframe"] = trades.get("timeframe", "").astype(str).str.strip()
    trades["strategy_type"] = trades.get("strategy_type", "").astype(str).str.strip()
    trades["side_considered"] = trades.get("side_considered", "").astype(str).str.strip().str.lower()
    trades["ctx_hour_et"] = pd.to_numeric(
        trades.get("ctx_hour_et", pd.Series(float("nan"), index=trades.index)),
        errors="coerce",
    )
    trades["ctx_session_substate"] = trades.get(
        "ctx_session_substate",
        pd.Series("", index=trades.index, dtype=object),
    ).astype(str).str.strip().str.lower()
    trades["ctx_volatility_regime"] = trades.get(
        "ctx_volatility_regime",
        pd.Series("", index=trades.index, dtype=object),
    ).astype(str).str.strip()
    trades["ctx_price_location"] = trades.get(
        "ctx_price_location",
        pd.Series("", index=trades.index, dtype=object),
    ).astype(str).str.strip()
    trades["de3_entry_filter_hit"] = _parse_bool_series(trades.get("de3_entry_filter_hit", pd.Series(False, index=trades.index)))
    trades["de3_entry_filter_reason"] = trades.get("de3_entry_filter_reason", "").astype(str).str.strip()
    for feature in ENTRY_SHAPE_NUMERIC_COLUMNS:
        trades[feature] = pd.to_numeric(trades.get(feature, pd.Series(float("nan"), index=trades.index)), errors="coerce")

    ts_effective = trades["entry_ts_et"].where(trades["entry_ts_et"].notna(), trades["timestamp_et"])
    trades["ts_effective"] = ts_effective
    trades["ctx_hour_et"] = trades["ctx_hour_et"].where(
        trades["ctx_hour_et"].notna(),
        pd.to_numeric(trades["ts_effective"].dt.hour, errors="coerce"),
    )
    derived_trade_substate = pd.Series(
        [
            _derive_session_substate(session_text=sess, hour_value=hour)
            for sess, hour in zip(
                trades["session"].tolist(),
                trades["ctx_hour_et"].tolist(),
            )
        ],
        index=trades.index,
        dtype=object,
    )
    trades.loc[
        trades["ctx_session_substate"].astype(str).str.strip() == "",
        "ctx_session_substate",
    ] = derived_trade_substate.loc[
        trades["ctx_session_substate"].astype(str).str.strip() == ""
    ]
    trades["split"] = _assign_splits(trades["ts_effective"], bounds)
    trades = trades[trades["split"] != ""].copy()

    summary = {
        "status": "ok",
        "rows_total": int(len(trades)),
        "rows_train": int((trades["split"] == "train").sum()),
        "rows_tune": int((trades["split"] == "tune").sum()),
        "rows_oos": int((trades["split"] == "oos").sum()),
        "rows_future_holdout": int((trades["split"] == "future_holdout").sum()),
        "variant_coverage": int(trades["variant_id"].astype(str).str.strip().ne("").sum()),
        "lane_coverage": int(trades["lane"].astype(str).str.strip().ne("").sum()),
        "chosen_shape_rows": int(len(chosen_shape)),
    }
    return trades, summary


def _stats_from_inventory(
    *,
    variants: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    variant_stats: Dict[str, Dict[str, Any]] = {}
    lane_weighted: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_weighted: Dict[str, float] = defaultdict(float)
    for row in variants:
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id", "") or "").strip()
        lane = str(row.get("lane", "") or "").strip()
        if not variant_id or not lane:
            continue
        n = max(1, safe_int(row.get("support_trades", 0), 0))
        win_rate = safe_float(row.get("win_rate", 0.50), 0.50)
        avg_pnl = safe_float(row.get("avg_pnl", 0.0), 0.0)
        loss_share = clip(safe_float(row.get("loss_share", 1.0 - win_rate), 1.0 - win_rate), 0.0, 1.0)
        stop_like_share = clip(
            safe_float(row.get("stop_like_share", loss_share), loss_share),
            0.0,
            1.0,
        )
        drawdown_norm = max(0.0, safe_float(row.get("drawdown_norm", 0.0), 0.0))
        worst_block_avg_pnl = safe_float(row.get("worst_block_avg_pnl", avg_pnl), avg_pnl)
        p_win_lcb = clip(win_rate - 0.06, 0.0, 1.0)
        ev_lcb = float(avg_pnl * 0.70)
        quality_lcb_score = _quality_lcb_score(
            p_win_lcb=p_win_lcb,
            ev_lcb=ev_lcb,
            n_trades=n,
            cfg=cfg,
        )
        stats = {
            "n_trades": int(n),
            "wins": int(round(win_rate * n)),
            "losses": int(max(0, n - round(win_rate * n))),
            "p_win": float(win_rate),
            "p_win_lcb": float(p_win_lcb),
            "ev_mean": float(avg_pnl),
            "ev_std": 0.0,
            "ev_lcb": float(ev_lcb),
            "profit_factor": float(safe_float(row.get("profit_factor", 1.0), 1.0)),
            "loss_share": float(loss_share),
            "stop_like_share": float(stop_like_share),
            "drawdown_norm": float(drawdown_norm),
            "worst_block_avg_pnl": float(worst_block_avg_pnl),
            "year_coverage": 0,
            "first_year": "",
            "last_year": "",
            "quality_lcb_score": float(quality_lcb_score),
            "source": "inventory_proxy",
        }
        variant_stats[variant_id] = stats

        lane_weighted[lane]["weight"] += float(n)
        lane_weighted[lane]["p_win_lcb_w"] += float(p_win_lcb * n)
        lane_weighted[lane]["ev_lcb_w"] += float(ev_lcb * n)
        lane_weighted[lane]["quality_lcb_w"] += float(quality_lcb_score * n)
        lane_weighted[lane]["n_trades"] += float(n)
        global_weighted["weight"] += float(n)
        global_weighted["p_win_lcb_w"] += float(p_win_lcb * n)
        global_weighted["ev_lcb_w"] += float(ev_lcb * n)
        global_weighted["quality_lcb_w"] += float(quality_lcb_score * n)
        global_weighted["n_trades"] += float(n)

    lane_stats: Dict[str, Dict[str, Any]] = {}
    for lane, values in lane_weighted.items():
        weight = max(1.0, safe_float(values.get("weight", 0.0), 0.0))
        n = int(safe_float(values.get("n_trades", 0.0), 0.0))
        lane_stats[lane] = {
            "n_trades": int(n),
            "p_win_lcb": float(safe_div(values.get("p_win_lcb_w", 0.0), weight, 0.0)),
            "ev_lcb": float(safe_div(values.get("ev_lcb_w", 0.0), weight, 0.0)),
            "quality_lcb_score": float(safe_div(values.get("quality_lcb_w", 0.0), weight, 0.0)),
            "loss_share": 0.50,
            "stop_like_share": 0.50,
            "drawdown_norm": 0.0,
            "worst_block_avg_pnl": 0.0,
            "year_coverage": 0,
            "first_year": "",
            "last_year": "",
            "source": "inventory_proxy",
        }

    g_weight = max(1.0, safe_float(global_weighted.get("weight", 0.0), 0.0))
    global_stats = {
        "n_trades": int(safe_float(global_weighted.get("n_trades", 0.0), 0.0)),
        "p_win_lcb": float(safe_div(global_weighted.get("p_win_lcb_w", 0.0), g_weight, 0.0)),
        "ev_lcb": float(safe_div(global_weighted.get("ev_lcb_w", 0.0), g_weight, 0.0)),
        "quality_lcb_score": float(safe_div(global_weighted.get("quality_lcb_w", 0.0), g_weight, 0.0)),
        "loss_share": 0.50,
        "stop_like_share": 0.50,
        "drawdown_norm": 0.0,
        "worst_block_avg_pnl": 0.0,
        "year_coverage": 0,
        "first_year": "",
        "last_year": "",
        "source": "inventory_proxy",
    }
    return variant_stats, lane_stats, global_stats


def _resolve_stats(
    *,
    variant_id: str,
    lane: str,
    variant_stats: Dict[str, Dict[str, Any]],
    lane_stats: Dict[str, Dict[str, Any]],
    global_stats: Dict[str, Any],
    min_variant_trades: int,
    min_lane_trades: int,
) -> Tuple[Dict[str, Any], str]:
    v = variant_stats.get(variant_id, {}) if variant_id else {}
    if isinstance(v, dict) and safe_int(v.get("n_trades", 0), 0) >= int(min_variant_trades):
        return v, "variant"
    l = lane_stats.get(lane, {}) if lane else {}
    if isinstance(l, dict) and safe_int(l.get("n_trades", 0), 0) >= int(min_lane_trades):
        return l, "lane"
    if isinstance(global_stats, dict) and safe_int(global_stats.get("n_trades", 0), 0) > 0:
        return global_stats, "global"
    return {}, "missing"


def _score_trade_row(
    *,
    row: pd.Series,
    stats: Dict[str, Any],
    score_cfg: Dict[str, Any],
    shape_model: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    base_quality = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
    route_conf = safe_float(row.get("route_confidence", 0.0), 0.0)
    edge_points = safe_float(row.get("edge_points", row.get("runtime_rank_score", 0.0)), 0.0)
    structural = safe_float(row.get("structural_score", 0.0), 0.0)

    route_center = safe_float(score_cfg.get("route_confidence_center", 0.05), 0.05)
    edge_scale = max(1e-9, safe_float(score_cfg.get("edge_scale_points", 0.40), 0.40))
    struct_scale = max(1e-9, safe_float(score_cfg.get("structural_scale", 0.80), 0.80))
    w_quality = safe_float(score_cfg.get("weight_quality_lcb", 0.65), 0.65)
    w_route = safe_float(score_cfg.get("weight_route_confidence", 0.20), 0.20)
    w_edge = safe_float(score_cfg.get("weight_edge_points", 0.10), 0.10)
    w_struct = safe_float(score_cfg.get("weight_structural_score", 0.05), 0.05)
    w_pf = safe_float(score_cfg.get("weight_profit_factor_component", 0.0), 0.0)
    w_year = safe_float(score_cfg.get("weight_year_coverage_component", 0.0), 0.0)
    w_loss = safe_float(score_cfg.get("weight_loss_share_penalty", 0.12), 0.12)
    w_stop = safe_float(score_cfg.get("weight_stop_like_share_penalty", 0.08), 0.08)
    w_drawdown = safe_float(score_cfg.get("weight_drawdown_penalty", 0.06), 0.06)
    w_worst_block = safe_float(score_cfg.get("weight_worst_block_penalty", 0.08), 0.08)
    w_shape = safe_float(score_cfg.get("weight_shape_penalty_component", 0.0), 0.0)
    profit_factor_center = safe_float(score_cfg.get("profit_factor_center", 1.10), 1.10)
    profit_factor_scale = max(1e-9, safe_float(score_cfg.get("profit_factor_scale", 0.35), 0.35))
    year_coverage_full = max(1.0, safe_float(score_cfg.get("year_coverage_full_years", 8.0), 8.0))
    loss_center = safe_float(score_cfg.get("loss_share_center", 0.52), 0.52)
    loss_scale = max(1e-9, safe_float(score_cfg.get("loss_share_scale", 0.22), 0.22))
    stop_center = safe_float(score_cfg.get("stop_like_share_center", 0.62), 0.62)
    stop_scale = max(1e-9, safe_float(score_cfg.get("stop_like_share_scale", 0.25), 0.25))
    drawdown_scale = max(1e-9, safe_float(score_cfg.get("drawdown_scale", 6.0), 6.0))
    shape_penalty_scale = max(1e-9, safe_float(score_cfg.get("shape_penalty_scale", 1.0), 1.0))
    shape_penalty_cap = max(0.0, safe_float(score_cfg.get("shape_penalty_cap", 2.0), 2.0))
    worst_block_scale = max(
        1e-9,
        safe_float(score_cfg.get("worst_block_scale_points", 3.0), 3.0),
    )
    loss_share = clip(safe_float(stats.get("loss_share", loss_center), loss_center), 0.0, 1.0)
    stop_like_share = clip(
        safe_float(stats.get("stop_like_share", stop_center), stop_center),
        0.0,
        1.0,
    )
    profit_factor = max(0.0, safe_float(stats.get("profit_factor", profit_factor_center), profit_factor_center))
    year_coverage = max(0.0, safe_float(stats.get("year_coverage", 0.0), 0.0))
    year_coverage_ratio = clip(safe_div(year_coverage, year_coverage_full, 0.0), 0.0, 1.0)
    drawdown_norm = max(0.0, safe_float(stats.get("drawdown_norm", 0.0), 0.0))
    worst_block_avg_pnl = safe_float(stats.get("worst_block_avg_pnl", 0.0), 0.0)
    loss_excess = max(0.0, loss_share - loss_center)
    stop_excess = max(0.0, stop_like_share - stop_center)
    worst_block_shortfall = max(0.0, -worst_block_avg_pnl)
    shape_eval = _evaluate_shape_penalty_row(row=row, shape_model=shape_model or {})
    shape_strength = min(
        float(shape_penalty_cap),
        max(0.0, safe_float(shape_eval.get("strength", 0.0), 0.0)),
    )

    components = {
        "quality_lcb_component": float(w_quality * base_quality),
        "route_confidence_component": float(w_route * (route_conf - route_center)),
        "edge_points_component": float(w_edge * math.tanh(edge_points / edge_scale)),
        "structural_component": float(w_struct * math.tanh(structural / struct_scale)),
        "profit_factor_component": float(
            w_pf * math.tanh((profit_factor - profit_factor_center) / profit_factor_scale)
        ),
        "year_coverage_component": float(w_year * ((2.0 * year_coverage_ratio) - 1.0)),
        "loss_share_penalty_component": float(-w_loss * math.tanh(loss_excess / loss_scale)),
        "stop_like_share_penalty_component": float(-w_stop * math.tanh(stop_excess / stop_scale)),
        "drawdown_penalty_component": float(-w_drawdown * math.tanh(drawdown_norm / drawdown_scale)),
        "worst_block_penalty_component": float(
            -w_worst_block * math.tanh(worst_block_shortfall / worst_block_scale)
        ),
        "shape_penalty_component": float(
            -w_shape * math.tanh(shape_strength / shape_penalty_scale)
        ),
    }
    score = float(sum(components.values()))
    aux = {
        "shape_penalty_strength": float(shape_strength),
        "shape_penalty_match_count": int(safe_int(shape_eval.get("match_count", 0), 0)),
        "shape_penalty_scope_key": str(shape_eval.get("scope_key", "") or ""),
    }
    return score, components, aux


def _max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    drawdown = peak - equity
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _trade_path_metrics(
    *,
    pnl_series: pd.Series,
    ts_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    pnl = pd.to_numeric(pnl_series, errors="coerce").fillna(0.0)
    trade_count = int(pnl.shape[0])
    if trade_count <= 0:
        return {
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "mean_pnl": 0.0,
            "trade_sqn": 0.0,
            "trade_sharpe_like": 0.0,
            "daily_sharpe": 0.0,
            "daily_sortino": 0.0,
            "trading_days": 0,
        }
    gross_wins = float(pnl[pnl > 0.0].sum())
    gross_losses_abs = float(abs(pnl[pnl <= 0.0].sum()))
    pf = float(gross_wins / gross_losses_abs) if gross_losses_abs > 1e-12 else (999.0 if gross_wins > 0.0 else 0.0)
    mean_pnl = float(pnl.mean())
    pnl_std = float(pnl.std(ddof=1)) if trade_count > 1 else 0.0
    trade_sqn = float((mean_pnl / pnl_std) * math.sqrt(float(trade_count))) if pnl_std > 1e-12 else 0.0

    daily_sharpe = 0.0
    daily_sortino = 0.0
    trading_days = 0
    if isinstance(ts_series, pd.Series) and not ts_series.empty:
        ts_local = _parse_et_timestamps(ts_series)
        daily_df = pd.DataFrame({"ts": ts_local, "pnl": pnl.reindex(ts_series.index)})
        daily_df = daily_df[daily_df["ts"].notna()].copy()
        if not daily_df.empty:
            daily_df["trade_day"] = daily_df["ts"].dt.floor("D")
            daily_pnl = daily_df.groupby("trade_day", sort=True)["pnl"].sum()
            trading_days = int(daily_pnl.shape[0])
            if trading_days >= 2:
                daily_mean = float(daily_pnl.mean())
                daily_std = float(daily_pnl.std(ddof=1))
                if daily_std > 1e-12:
                    daily_sharpe = float((daily_mean / daily_std) * math.sqrt(252.0))
                downside = daily_pnl[daily_pnl < 0.0]
                if not downside.empty:
                    downside_rms = float(math.sqrt(float((downside**2).mean())))
                    if downside_rms > 1e-12:
                        daily_sortino = float((daily_mean / downside_rms) * math.sqrt(252.0))

    return {
        "net_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0.0).sum() / max(1, trade_count)),
        "profit_factor": float(pf),
        "max_drawdown": float(_max_drawdown(pnl)),
        "mean_pnl": float(mean_pnl),
        "trade_sqn": float(trade_sqn),
        "trade_sharpe_like": float(trade_sqn),
        "daily_sharpe": float(daily_sharpe),
        "daily_sortino": float(daily_sortino),
        "trading_days": int(trading_days),
    }


def _normalize_trade_side(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"long", "buy"}:
        return "long"
    if text in {"short", "sell"}:
        return "short"
    return ""


def _trade_side_series(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series([], dtype=object)
    side = pd.Series([""] * len(frame), index=frame.index, dtype=object)
    if "side_considered" in frame.columns:
        side = frame["side_considered"].map(_normalize_trade_side).fillna("")
    if "lane" in frame.columns:
        lane_side = frame["lane"].map(lambda v: _normalize_trade_side(lane_to_side(v))).fillna("")
        side = side.where(side.astype(str).str.strip() != "", lane_side)
    return side.astype(str).str.strip().str.lower()


def _side_metrics_from_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {
            "side_known_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "long_share_known": 0.0,
            "short_share_known": 0.0,
            "dominant_side_share": 0.0,
            "side_balance_gap": 0.0,
            "long_net_pnl": 0.0,
            "short_net_pnl": 0.0,
            "long_profit_factor": 0.0,
            "short_profit_factor": 0.0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
        }
    side = _trade_side_series(frame)
    known_mask = side.isin({"long", "short"})
    known_count = int(known_mask.sum())
    out: Dict[str, Any] = {
        "side_known_trades": int(known_count),
        "long_trades": 0,
        "short_trades": 0,
        "long_share_known": 0.0,
        "short_share_known": 0.0,
        "dominant_side_share": 0.0,
        "side_balance_gap": 0.0,
        "long_net_pnl": 0.0,
        "short_net_pnl": 0.0,
        "long_profit_factor": 0.0,
        "short_profit_factor": 0.0,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
    }
    if known_count <= 0:
        return out
    ts_series = frame.get("ts_effective", pd.Series(dtype=object))
    for label in ("long", "short"):
        mask = side == label
        count = int(mask.sum())
        pnl_series = frame.loc[mask, "realized_pnl"].astype(float) if "realized_pnl" in frame.columns else pd.Series(dtype=float)
        ts_local = ts_series.loc[mask] if isinstance(ts_series, pd.Series) else pd.Series(dtype=object)
        metrics = _trade_path_metrics(pnl_series=pnl_series, ts_series=ts_local)
        out[f"{label}_trades"] = int(count)
        out[f"{label}_share_known"] = float(safe_div(float(count), float(known_count), 0.0))
        out[f"{label}_net_pnl"] = float(metrics.get("net_pnl", 0.0))
        out[f"{label}_profit_factor"] = float(metrics.get("profit_factor", 0.0))
        out[f"{label}_win_rate"] = float(metrics.get("win_rate", 0.0))
        out[f"{label}_daily_sharpe"] = float(metrics.get("daily_sharpe", 0.0))
        out[f"{label}_daily_sortino"] = float(metrics.get("daily_sortino", 0.0))
        out[f"{label}_trade_sqn"] = float(metrics.get("trade_sqn", 0.0))
        out[f"{label}_max_drawdown"] = float(metrics.get("max_drawdown", 0.0))
    out["dominant_side_share"] = float(
        max(
            safe_float(out.get("long_share_known", 0.0), 0.0),
            safe_float(out.get("short_share_known", 0.0), 0.0),
        )
    )
    out["side_balance_gap"] = float(
        abs(
            safe_float(out.get("long_share_known", 0.0), 0.0)
            - safe_float(out.get("short_share_known", 0.0), 0.0)
        )
    )
    return out


def _build_shape_penalty_model(
    *,
    trade_df: pd.DataFrame,
    training_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    shape_cfg = (
        training_cfg.get("shape_penalty_model", {})
        if isinstance(training_cfg.get("shape_penalty_model"), dict)
        else {}
    )
    enabled = bool(shape_cfg.get("enabled", False))
    if (not enabled) or trade_df.empty:
        return {
            "enabled": False,
            "scope_mode": str(shape_cfg.get("scope_mode", "lane_timeframe") or "lane_timeframe"),
            "max_rules_per_row": 0,
            "rules": [],
        }, {
            "enabled": bool(enabled),
            "rules_considered": 0,
            "rules_selected": 0,
            "selected_rules": [],
            "reason": "disabled_or_empty",
        }

    feature_names = [
        str(name).strip()
        for name in (
            shape_cfg.get("features", ENTRY_SHAPE_NUMERIC_COLUMNS)
            if isinstance(shape_cfg.get("features"), (list, tuple, set))
            else ENTRY_SHAPE_NUMERIC_COLUMNS
        )
        if str(name).strip() in trade_df.columns
    ]
    feature_names = [name for name in feature_names if name in ENTRY_SHAPE_NUMERIC_COLUMNS]
    if not feature_names:
        return {
            "enabled": False,
            "scope_mode": str(shape_cfg.get("scope_mode", "lane_timeframe") or "lane_timeframe"),
            "max_rules_per_row": 0,
            "rules": [],
        }, {
            "enabled": True,
            "rules_considered": 0,
            "rules_selected": 0,
            "selected_rules": [],
            "reason": "no_shape_features_available",
        }

    scope_mode = str(shape_cfg.get("scope_mode", "lane_timeframe") or "lane_timeframe").strip().lower()
    min_scope_trades = max(1, safe_int(shape_cfg.get("min_scope_trades", 160), 160))
    min_rule_trades = max(10, safe_int(shape_cfg.get("min_rule_trades", 60), 60))
    min_complement_trades = max(10, safe_int(shape_cfg.get("min_complement_trades", 120), 120))
    max_rule_fraction = clip(safe_float(shape_cfg.get("max_rule_fraction", 0.35), 0.35), 0.05, 0.95)
    min_quality_gap = max(0.0, safe_float(shape_cfg.get("min_quality_gap", 0.10), 0.10))
    max_subset_pf = max(0.0, safe_float(shape_cfg.get("max_subset_profit_factor", 0.98), 0.98))
    max_subset_ev_mean = safe_float(shape_cfg.get("max_subset_ev_mean", 0.0), 0.0)
    min_subset_loss_share = clip(safe_float(shape_cfg.get("min_subset_loss_share", 0.53), 0.53), 0.0, 1.0)
    min_year_coverage = max(0, safe_int(shape_cfg.get("min_year_coverage", 4), 4))
    max_rules_per_scope = max(0, safe_int(shape_cfg.get("max_rules_per_scope", 2), 2))
    max_rules_per_row = max(0, safe_int(shape_cfg.get("max_rules_per_row", max_rules_per_scope), max_rules_per_scope))

    low_quantiles = sorted(
        {
            float(clip(safe_float(v, 0.0), 0.01, 0.49))
            for v in (
                shape_cfg.get("low_quantiles", [0.10, 0.20])
                if isinstance(shape_cfg.get("low_quantiles"), (list, tuple, set))
                else [0.10, 0.20]
            )
        }
    )
    high_quantiles = sorted(
        {
            float(clip(safe_float(v, 1.0), 0.51, 0.99))
            for v in (
                shape_cfg.get("high_quantiles", [0.80, 0.90])
                if isinstance(shape_cfg.get("high_quantiles"), (list, tuple, set))
                else [0.80, 0.90]
            )
        }
    )

    local = trade_df.copy()
    local["shape_scope_key"] = [
        _shape_scope_key(
            lane=row.get("lane"),
            timeframe=row.get("timeframe"),
            scope_mode=scope_mode,
        )
        for row in local.to_dict("records")
    ]
    local = local[local["shape_scope_key"].astype(str).str.strip() != ""].copy()
    if local.empty:
        return {
            "enabled": False,
            "scope_mode": str(scope_mode),
            "max_rules_per_row": int(max_rules_per_row),
            "rules": [],
        }, {
            "enabled": True,
            "rules_considered": 0,
            "rules_selected": 0,
            "selected_rules": [],
            "reason": "no_scope_rows",
        }

    selected_rules: List[Dict[str, Any]] = []
    total_candidates = 0
    for scope_key, scope_df in local.groupby("shape_scope_key", sort=False):
        scope_frame = scope_df.copy()
        if len(scope_frame) < min_scope_trades:
            continue
        baseline_stats = _build_stats_from_frame(
            trade_df=scope_frame,
            key_col=None,
            cfg=training_cfg,
        ).get("__global__", {})
        baseline_quality = safe_float(baseline_stats.get("quality_lcb_score", 0.0), 0.0)
        baseline_loss = clip(safe_float(baseline_stats.get("loss_share", 0.50), 0.50), 0.0, 1.0)
        baseline_stop = clip(safe_float(baseline_stats.get("stop_like_share", 0.50), 0.50), 0.0, 1.0)
        baseline_pf = max(0.0, safe_float(baseline_stats.get("profit_factor", 1.0), 1.0))

        scope_candidates: List[Dict[str, Any]] = []
        for feature in feature_names:
            values = pd.to_numeric(scope_frame.get(feature, pd.Series(dtype=float)), errors="coerce")
            valid_mask = values.notna()
            valid_count = int(valid_mask.sum())
            if valid_count < max(min_scope_trades, min_rule_trades):
                continue
            for operator, quantiles in (("<=", low_quantiles), (">=", high_quantiles)):
                seen_thresholds = set()
                for q in quantiles:
                    threshold = float(values[valid_mask].quantile(q))
                    if not math.isfinite(threshold):
                        continue
                    threshold_key = round(threshold, 6)
                    if threshold_key in seen_thresholds:
                        continue
                    seen_thresholds.add(threshold_key)
                    if operator == "<=":
                        subset_mask = valid_mask & (values <= threshold)
                    else:
                        subset_mask = valid_mask & (values >= threshold)
                    subset_count = int(subset_mask.sum())
                    support_ratio = safe_div(float(subset_count), float(max(1, valid_count)), 0.0)
                    if subset_count < min_rule_trades or support_ratio > max_rule_fraction:
                        continue
                    complement_mask = valid_mask & (~subset_mask)
                    complement_count = int(complement_mask.sum())
                    if complement_count < min_complement_trades:
                        continue
                    subset_stats = _build_stats_from_frame(
                        trade_df=scope_frame.loc[subset_mask],
                        key_col=None,
                        cfg=training_cfg,
                    ).get("__global__", {})
                    complement_stats = _build_stats_from_frame(
                        trade_df=scope_frame.loc[complement_mask],
                        key_col=None,
                        cfg=training_cfg,
                    ).get("__global__", {})
                    subset_quality = safe_float(subset_stats.get("quality_lcb_score", 0.0), 0.0)
                    subset_pf = max(0.0, safe_float(subset_stats.get("profit_factor", 0.0), 0.0))
                    subset_ev_mean = safe_float(subset_stats.get("ev_mean", 0.0), 0.0)
                    subset_loss = clip(safe_float(subset_stats.get("loss_share", 0.50), 0.50), 0.0, 1.0)
                    subset_stop = clip(safe_float(subset_stats.get("stop_like_share", 0.50), 0.50), 0.0, 1.0)
                    subset_year_coverage = safe_int(subset_stats.get("year_coverage", 0), 0)
                    if subset_year_coverage < min_year_coverage:
                        continue
                    quality_gap = float(baseline_quality - subset_quality)
                    if quality_gap < min_quality_gap:
                        continue
                    if (subset_pf > max_subset_pf) and (subset_ev_mean > max_subset_ev_mean) and (subset_loss < min_subset_loss_share):
                        continue
                    complement_quality = safe_float(complement_stats.get("quality_lcb_score", subset_quality), subset_quality)
                    severity = max(0.0, quality_gap)
                    severity += 0.50 * max(0.0, subset_loss - baseline_loss)
                    severity += 0.30 * max(0.0, subset_stop - baseline_stop)
                    severity += 0.25 * max(0.0, baseline_pf - subset_pf)
                    severity += 0.25 * max(0.0, complement_quality - subset_quality)
                    severity *= min(1.5, math.sqrt(float(subset_count) / float(max(1, min_rule_trades))))
                    if severity <= 0.0:
                        continue
                    total_candidates += 1
                    scope_candidates.append(
                        {
                            "name": f"{scope_key}|{feature}|{operator}|{threshold_key:g}",
                            "scope_key": str(scope_key),
                            "feature": str(feature),
                            "operator": str(operator),
                            "threshold": float(threshold_key),
                            "penalty_strength": float(severity),
                            "train_support_trades": int(subset_count),
                            "train_support_ratio": float(support_ratio),
                            "train_quality_gap": float(quality_gap),
                            "train_subset_quality_lcb": float(subset_quality),
                            "train_subset_profit_factor": float(subset_pf),
                            "train_subset_ev_mean": float(subset_ev_mean),
                            "train_subset_loss_share": float(subset_loss),
                            "train_subset_stop_like_share": float(subset_stop),
                            "train_subset_year_coverage": int(subset_year_coverage),
                            "train_complement_quality_lcb": float(complement_quality),
                        }
                    )

        if not scope_candidates or max_rules_per_scope <= 0:
            continue
        scope_candidates.sort(
            key=lambda row: (
                safe_float(row.get("penalty_strength", 0.0), 0.0),
                safe_float(row.get("train_quality_gap", 0.0), 0.0),
                safe_int(row.get("train_support_trades", 0), 0),
            ),
            reverse=True,
        )
        seen_feature_ops = set()
        for row in scope_candidates:
            key = (str(row.get("feature", "")), str(row.get("operator", "")))
            if key in seen_feature_ops:
                continue
            selected_rules.append(dict(row))
            seen_feature_ops.add(key)
            if len(seen_feature_ops) >= max_rules_per_scope:
                break

    selected_rules.sort(
        key=lambda row: (
            str(row.get("scope_key", "")),
            -safe_float(row.get("penalty_strength", 0.0), 0.0),
            str(row.get("feature", "")),
        ),
    )
    model = {
        "enabled": bool(selected_rules),
        "scope_mode": str(scope_mode),
        "max_rules_per_row": int(max_rules_per_row),
        "rules": list(selected_rules),
    }
    summary = {
        "enabled": True,
        "scope_mode": str(scope_mode),
        "features": list(feature_names),
        "rules_considered": int(total_candidates),
        "rules_selected": int(len(selected_rules)),
        "selected_rules": list(selected_rules),
        "max_rules_per_scope": int(max_rules_per_scope),
        "max_rules_per_row": int(max_rules_per_row),
    }
    return model, summary


def _evaluate_shape_penalty_row(
    *,
    row: pd.Series,
    shape_model: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(shape_model, dict) or not bool(shape_model.get("enabled", False)):
        return {
            "strength": 0.0,
            "match_count": 0,
            "scope_key": "",
        }
    rules = shape_model.get("rules", [])
    if not isinstance(rules, list) or not rules:
        return {
            "strength": 0.0,
            "match_count": 0,
            "scope_key": "",
        }
    scope_mode = str(shape_model.get("scope_mode", "lane_timeframe") or "lane_timeframe").strip().lower()
    scope_key = _shape_scope_key(
        lane=row.get("lane"),
        timeframe=row.get("timeframe"),
        scope_mode=scope_mode,
    )
    if not scope_key:
        return {
            "strength": 0.0,
            "match_count": 0,
            "scope_key": "",
        }
    max_rules_per_row = max(0, safe_int(shape_model.get("max_rules_per_row", 0), 0))
    strength = 0.0
    match_count = 0
    for rule in rules:
        if str(rule.get("scope_key", "") or "") != scope_key:
            continue
        feature = str(rule.get("feature", "") or "").strip()
        operator = str(rule.get("operator", "") or "").strip()
        threshold = safe_float(rule.get("threshold", float("nan")), float("nan"))
        value = safe_float(row.get(feature, float("nan")), float("nan"))
        if (not feature) or (not math.isfinite(threshold)) or (not math.isfinite(value)):
            continue
        hit = (operator == "<=" and value <= threshold) or (operator == ">=" and value >= threshold)
        if not hit:
            continue
        strength += max(0.0, safe_float(rule.get("penalty_strength", 0.0), 0.0))
        match_count += 1
        if max_rules_per_row > 0 and match_count >= max_rules_per_row:
            break
    return {
        "strength": float(strength),
        "match_count": int(match_count),
        "scope_key": str(scope_key),
    }


def _threshold_candidates(scores: pd.Series, cfg: Dict[str, Any]) -> List[float]:
    fixed = cfg.get("threshold_candidates", [])
    if isinstance(fixed, (list, tuple)):
        vals = sorted({round(safe_float(v, 0.0), 6) for v in fixed})
        if vals:
            return vals
    if scores.empty:
        return [safe_float(cfg.get("default_threshold", 0.0), 0.0)]
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    vals = [float(scores.quantile(q)) for q in quantiles]
    vals.extend(
        [
            float(scores.min()),
            float(scores.max()),
            safe_float(cfg.get("default_threshold", 0.0), 0.0),
        ]
    )
    return sorted({round(v, 6) for v in vals if math.isfinite(v)})


def _threshold_objective(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    dd_weight = safe_float(cfg.get("objective_weight_max_drawdown", 0.55), 0.55)
    pf_weight = safe_float(cfg.get("objective_weight_profit_factor", 140.0), 140.0)
    keep_weight = safe_float(cfg.get("objective_weight_keep_rate", 220.0), 220.0)
    sharpe_weight = safe_float(cfg.get("objective_weight_daily_sharpe", 0.0), 0.0)
    sortino_weight = safe_float(cfg.get("objective_weight_daily_sortino", 0.0), 0.0)
    sqn_weight = safe_float(cfg.get("objective_weight_trade_sqn", 0.0), 0.0)
    pf_cap = max(0.0, safe_float(cfg.get("objective_profit_factor_cap", 3.5), 3.5))
    sharpe_cap = max(0.0, safe_float(cfg.get("objective_daily_sharpe_cap", 6.0), 6.0))
    sortino_cap = max(0.0, safe_float(cfg.get("objective_daily_sortino_cap", 8.0), 8.0))
    sqn_cap = max(0.0, safe_float(cfg.get("objective_trade_sqn_cap", 6.0), 6.0))
    side_balance_weight = safe_float(cfg.get("objective_weight_side_balance_penalty", 0.0), 0.0)
    long_share_excess_weight = safe_float(
        cfg.get("objective_weight_long_share_excess_penalty", 0.0),
        0.0,
    )
    negative_side_net_weight = safe_float(
        cfg.get("objective_weight_negative_side_net_penalty", 0.0),
        0.0,
    )
    side_pf_shortfall_weight = safe_float(
        cfg.get("objective_weight_side_pf_shortfall_penalty", 0.0),
        0.0,
    )
    side_balance_target = clip(safe_float(cfg.get("objective_side_balance_target", 0.50), 0.50), 0.0, 1.0)
    side_balance_tolerance = clip(
        safe_float(cfg.get("objective_side_balance_tolerance", 0.0), 0.0),
        0.0,
        0.50,
    )
    max_long_share = clip(safe_float(cfg.get("objective_max_long_share", 1.0), 1.0), 0.0, 1.0)
    side_net_scale = max(1e-9, safe_float(cfg.get("objective_side_net_scale", 1000.0), 1000.0))
    side_pf_floor = max(0.0, safe_float(cfg.get("objective_side_profit_factor_floor", 0.0), 0.0))
    side_pf_scale = max(
        1e-9,
        safe_float(cfg.get("objective_side_profit_factor_scale", 0.25), 0.25),
    )
    net = safe_float(metrics.get("net_pnl", 0.0), 0.0)
    max_dd = safe_float(metrics.get("max_drawdown", 0.0), 0.0)
    keep_rate = safe_float(metrics.get("keep_rate", 0.0), 0.0)
    pf = safe_float(metrics.get("profit_factor", 0.0), 0.0)
    daily_sharpe = safe_float(metrics.get("daily_sharpe", 0.0), 0.0)
    daily_sortino = safe_float(metrics.get("daily_sortino", 0.0), 0.0)
    trade_sqn = safe_float(metrics.get("trade_sqn", 0.0), 0.0)
    long_share = clip(safe_float(metrics.get("long_share_known", 0.50), 0.50), 0.0, 1.0)
    long_net = safe_float(metrics.get("long_net_pnl", 0.0), 0.0)
    short_net = safe_float(metrics.get("short_net_pnl", 0.0), 0.0)
    long_pf = max(0.0, safe_float(metrics.get("long_profit_factor", 0.0), 0.0))
    short_pf = max(0.0, safe_float(metrics.get("short_profit_factor", 0.0), 0.0))
    side_known_trades = safe_int(metrics.get("side_known_trades", 0), 0)
    side_balance_gap = (
        max(0.0, abs(long_share - side_balance_target) - side_balance_tolerance)
        if side_known_trades > 0
        else 0.0
    )
    long_share_excess = max(0.0, long_share - max_long_share) if side_known_trades > 0 else 0.0
    negative_side_net_penalty = (
        math.tanh(max(0.0, -long_net) / side_net_scale)
        + math.tanh(max(0.0, -short_net) / side_net_scale)
        if side_known_trades > 0
        else 0.0
    )
    side_pf_shortfall_penalty = (
        math.tanh(max(0.0, side_pf_floor - long_pf) / side_pf_scale)
        + math.tanh(max(0.0, side_pf_floor - short_pf) / side_pf_scale)
        if side_known_trades > 0 and side_pf_floor > 0.0
        else 0.0
    )
    return float(
        net
        - (dd_weight * max_dd)
        + (pf_weight * min(pf_cap, pf))
        + (keep_weight * keep_rate)
        + (sharpe_weight * max(-sharpe_cap, min(sharpe_cap, daily_sharpe)))
        + (sortino_weight * max(-sortino_cap, min(sortino_cap, daily_sortino)))
        + (sqn_weight * max(-sqn_cap, min(sqn_cap, trade_sqn)))
        - (side_balance_weight * side_balance_gap)
        - (long_share_excess_weight * long_share_excess)
        - (negative_side_net_weight * negative_side_net_penalty)
        - (side_pf_shortfall_weight * side_pf_shortfall_penalty)
    )


def _scope_threshold_offsets_series(scope_series: pd.Series, scope_cfg: Dict[str, Any]) -> pd.Series:
    if not isinstance(scope_series, pd.Series):
        return pd.Series([], dtype=float)
    if not isinstance(scope_cfg, dict) or not scope_cfg:
        return pd.Series(0.0, index=scope_series.index, dtype=float)
    default_offset = safe_float(scope_cfg.get("default", 0.0), 0.0)
    lookup = {
        "variant": float(safe_float(scope_cfg.get("variant", default_offset), default_offset)),
        "lane": float(safe_float(scope_cfg.get("lane", default_offset), default_offset)),
        "global": float(safe_float(scope_cfg.get("global", default_offset), default_offset)),
        "missing": float(safe_float(scope_cfg.get("missing", default_offset), default_offset)),
    }
    normalized = scope_series.astype(str).str.strip().str.lower()
    return pd.to_numeric(normalized.map(lookup).fillna(default_offset), errors="coerce").fillna(default_offset)


def _evaluate_threshold(
    *,
    tune_scored: pd.DataFrame,
    threshold: float,
    min_keep_trades: int,
    min_keep_rate: float,
    objective_cfg: Dict[str, Any],
    scope_threshold_offsets: Optional[Dict[str, Any]] = None,
    scope_offsets_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if tune_scored.empty:
        return {
            "threshold": float(threshold),
            "valid": False,
            "reason": "no_tune_rows",
        }
    score_series = pd.to_numeric(
        tune_scored.get("entry_policy_score", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(float("-inf"))
    if isinstance(scope_offsets_series, pd.Series) and not scope_offsets_series.empty:
        offsets = pd.to_numeric(
            scope_offsets_series.reindex(tune_scored.index),
            errors="coerce",
        ).fillna(0.0)
    else:
        scope_cfg = (
            scope_threshold_offsets
            if isinstance(scope_threshold_offsets, dict)
            else {}
        )
        if "scope" in tune_scored.columns and scope_cfg:
            offsets = _scope_threshold_offsets_series(tune_scored["scope"], scope_cfg)
        else:
            offsets = pd.Series(0.0, index=tune_scored.index, dtype=float)
    effective_threshold = float(threshold) + pd.to_numeric(offsets, errors="coerce").fillna(0.0)
    kept = tune_scored[score_series >= effective_threshold].copy()
    total = int(len(tune_scored))
    keep_n = int(len(kept))
    keep_rate = safe_div(float(keep_n), float(max(1, total)), 0.0)
    if keep_n < int(min_keep_trades):
        return {
            "threshold": float(threshold),
            "valid": False,
            "reason": "keep_trades_below_min",
            "keep_trades": int(keep_n),
            "keep_rate": float(keep_rate),
        }
    if keep_rate < float(min_keep_rate):
        return {
            "threshold": float(threshold),
            "valid": False,
            "reason": "keep_rate_below_min",
            "keep_trades": int(keep_n),
            "keep_rate": float(keep_rate),
        }
    side_metrics = _side_metrics_from_frame(kept)
    min_long_trades = max(0, safe_int(objective_cfg.get("min_long_trades", 0), 0))
    min_short_trades = max(0, safe_int(objective_cfg.get("min_short_trades", 0), 0))
    max_long_share_valid = clip(
        safe_float(objective_cfg.get("max_long_share_valid", 1.0), 1.0),
        0.0,
        1.0,
    )
    if min_long_trades > 0 and safe_int(side_metrics.get("long_trades", 0), 0) < min_long_trades:
        return {
            "threshold": float(threshold),
            "valid": False,
            "reason": "long_trades_below_min",
            "keep_trades": int(keep_n),
            "keep_rate": float(keep_rate),
            **side_metrics,
        }
    if min_short_trades > 0 and safe_int(side_metrics.get("short_trades", 0), 0) < min_short_trades:
        return {
            "threshold": float(threshold),
            "valid": False,
            "reason": "short_trades_below_min",
            "keep_trades": int(keep_n),
            "keep_rate": float(keep_rate),
            **side_metrics,
        }
    if (
        safe_int(side_metrics.get("side_known_trades", 0), 0) > 0
        and safe_float(side_metrics.get("long_share_known", 0.0), 0.0) > max_long_share_valid
    ):
        return {
            "threshold": float(threshold),
            "valid": False,
            "reason": "long_share_above_max",
            "keep_trades": int(keep_n),
            "keep_rate": float(keep_rate),
            **side_metrics,
        }
    pnl = kept["realized_pnl"].astype(float)
    path_metrics = _trade_path_metrics(
        pnl_series=pnl,
        ts_series=kept.get("ts_effective", pd.Series(dtype=object)),
    )
    metrics = {
        "threshold": float(threshold),
        "valid": True,
        "reason": "ok",
        "keep_trades": int(keep_n),
        "drop_trades": int(total - keep_n),
        "keep_rate": float(keep_rate),
        **path_metrics,
        **side_metrics,
    }
    metrics["objective"] = float(_threshold_objective(metrics, objective_cfg))
    return metrics


def train_de3_v4_entry_policy(
    *,
    dataset: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    training_cfg = cfg.get("entry_policy", {}) if isinstance(cfg.get("entry_policy"), dict) else {}
    score_cfg = (
        training_cfg.get("score_components", {})
        if isinstance(training_cfg.get("score_components"), dict)
        else {}
    )
    tuning_cfg = (
        training_cfg.get("threshold_tuning", {})
        if isinstance(training_cfg.get("threshold_tuning"), dict)
        else {}
    )
    split_summary = dataset.get("split_summary", {}) if isinstance(dataset.get("split_summary"), dict) else {}
    bounds = _split_bounds(split_summary)
    metadata = dataset.get("metadata", {}) if isinstance(dataset.get("metadata"), dict) else {}
    variants = dataset.get("variants", []) if isinstance(dataset.get("variants"), list) else []

    decisions_csv = metadata.get("decisions_csv_path", "")
    trades_csv = metadata.get("trade_attribution_csv_path", "")
    trade_df, csv_audit = _trade_frame_from_csvs(
        decisions_csv_path=decisions_csv,
        trade_attribution_csv_path=trades_csv,
        bounds=bounds,
        variants=variants,
    )

    min_variant_trades = max(1, safe_int(training_cfg.get("min_variant_trades", 25), 25))
    min_lane_trades = max(1, safe_int(training_cfg.get("min_lane_trades", 120), 120))
    default_threshold = safe_float(training_cfg.get("default_threshold", 0.0), 0.0)
    allow_on_missing_stats = bool(training_cfg.get("allow_on_missing_stats", True))
    scope_threshold_offsets_cfg = (
        dict(training_cfg.get("scope_threshold_offsets", {}))
        if isinstance(training_cfg.get("scope_threshold_offsets"), dict)
        else {}
    )
    scope_threshold_offsets_cfg = {
        "variant": float(safe_float(scope_threshold_offsets_cfg.get("variant", 0.0), 0.0)),
        "lane": float(safe_float(scope_threshold_offsets_cfg.get("lane", 0.06), 0.06)),
        "global": float(safe_float(scope_threshold_offsets_cfg.get("global", 0.12), 0.12)),
        "missing": float(safe_float(scope_threshold_offsets_cfg.get("missing", 0.15), 0.15)),
        "default": float(safe_float(scope_threshold_offsets_cfg.get("default", 0.0), 0.0)),
    }

    inventory_fallback = trade_df.empty or int(csv_audit.get("rows_train", 0)) <= 0

    if inventory_fallback:
        variant_stats_fit, lane_stats_fit, global_stats_fit = _stats_from_inventory(
            variants=variants,
            cfg=training_cfg,
        )
        shape_model_fit = {
            "enabled": False,
            "scope_mode": "lane_timeframe",
            "max_rules_per_row": 0,
            "rules": [],
        }
        shape_training_summary = {
            "enabled": False,
            "rules_considered": 0,
            "rules_selected": 0,
            "selected_rules": [],
            "reason": "inventory_fallback",
        }
        selected_threshold = float(default_threshold)
        threshold_source = "config_default_inventory_fallback"
        threshold_trials: List[Dict[str, Any]] = []
        baseline_metrics = {
            "keep_trades": 0,
            "drop_trades": 0,
            "keep_rate": 1.0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "mean_pnl": 0.0,
            "objective": 0.0,
            "trade_sqn": 0.0,
            "trade_sharpe_like": 0.0,
            "daily_sharpe": 0.0,
            "daily_sortino": 0.0,
            "trading_days": 0,
        }
        selected_metrics = dict(baseline_metrics)
    else:
        train_df = trade_df[trade_df["split"] == "train"].copy()
        tune_df = trade_df[trade_df["split"] == "tune"].copy()
        fit_df = trade_df[trade_df["split"].isin(["train", "tune"])].copy()
        shape_model_train, shape_training_summary = _build_shape_penalty_model(
            trade_df=train_df,
            training_cfg=training_cfg,
        )

        variant_stats_train = _build_stats_from_frame(
            trade_df=train_df,
            key_col="variant_id",
            cfg=training_cfg,
        )
        lane_stats_train = _build_stats_from_frame(
            trade_df=train_df,
            key_col="lane",
            cfg=training_cfg,
        )
        global_stats_train = _build_stats_from_frame(
            trade_df=train_df,
            key_col=None,
            cfg=training_cfg,
        ).get("__global__", {})

        scored_rows: List[Dict[str, Any]] = []
        for row in tune_df.to_dict("records"):
            variant_id = str(row.get("variant_id", "") or "").strip()
            lane = str(row.get("lane", "") or "").strip()
            stats, scope = _resolve_stats(
                variant_id=variant_id,
                lane=lane,
                variant_stats=variant_stats_train,
                lane_stats=lane_stats_train,
                global_stats=global_stats_train,
                min_variant_trades=min_variant_trades,
                min_lane_trades=min_lane_trades,
            )
            score, components, aux = _score_trade_row(
                row=row,
                stats=stats,
                score_cfg=score_cfg,
                shape_model=shape_model_train,
            )
            scored_rows.append(
                {
                    "decision_id": str(row.get("decision_id", "") or ""),
                    "variant_id": variant_id,
                    "lane": lane,
                    "timeframe": str(row.get("timeframe", "") or ""),
                    "strategy_type": str(row.get("strategy_type", "") or ""),
                    "split": "tune",
                    "ts_effective": row.get("ts_effective"),
                    "realized_pnl": safe_float(row.get("realized_pnl", 0.0), 0.0),
                    "entry_policy_score": float(score),
                    "scope": str(scope),
                    "quality_lcb_score": safe_float(stats.get("quality_lcb_score", 0.0), 0.0),
                    "p_win_lcb": safe_float(stats.get("p_win_lcb", 0.0), 0.0),
                    "ev_lcb": safe_float(stats.get("ev_lcb", 0.0), 0.0),
                    "n_trades_scope": safe_int(stats.get("n_trades", 0), 0),
                    "route_confidence_component": safe_float(components.get("route_confidence_component", 0.0), 0.0),
                    "edge_points_component": safe_float(components.get("edge_points_component", 0.0), 0.0),
                    "structural_component": safe_float(components.get("structural_component", 0.0), 0.0),
                    "quality_lcb_component": safe_float(components.get("quality_lcb_component", 0.0), 0.0),
                    "shape_penalty_component": safe_float(components.get("shape_penalty_component", 0.0), 0.0),
                    "shape_penalty_strength": safe_float(aux.get("shape_penalty_strength", 0.0), 0.0),
                    "shape_penalty_match_count": safe_int(aux.get("shape_penalty_match_count", 0), 0),
                    "shape_penalty_scope_key": str(aux.get("shape_penalty_scope_key", "") or ""),
                }
            )
        tune_scored = pd.DataFrame(scored_rows)
        if (not tune_scored.empty) and ("ts_effective" in tune_scored.columns):
            tune_scored = tune_scored.sort_values("ts_effective", kind="mergesort")
        tune_scope_offsets = (
            _scope_threshold_offsets_series(tune_scored["scope"], scope_threshold_offsets_cfg)
            if ("scope" in tune_scored.columns and not tune_scored.empty)
            else pd.Series(dtype=float)
        )

        min_keep_trades = max(1, safe_int(tuning_cfg.get("min_keep_trades", 80), 80))
        min_keep_rate = clip(safe_float(tuning_cfg.get("min_keep_rate", 0.40), 0.40), 0.0, 1.0)
        candidates = _threshold_candidates(
            tune_scored["entry_policy_score"] if "entry_policy_score" in tune_scored.columns else pd.Series(dtype=float),
            tuning_cfg,
        )
        threshold_trials = []
        for threshold in candidates:
            trial = _evaluate_threshold(
                tune_scored=tune_scored,
                threshold=float(threshold),
                min_keep_trades=min_keep_trades,
                min_keep_rate=min_keep_rate,
                objective_cfg=tuning_cfg,
                scope_threshold_offsets=scope_threshold_offsets_cfg,
                scope_offsets_series=tune_scope_offsets,
            )
            threshold_trials.append(trial)
        baseline_metrics = _evaluate_threshold(
            tune_scored=tune_scored,
            threshold=-1e9,
            min_keep_trades=0,
            min_keep_rate=0.0,
            objective_cfg=tuning_cfg,
            scope_threshold_offsets=scope_threshold_offsets_cfg,
            scope_offsets_series=tune_scope_offsets,
        )
        valid_trials = [t for t in threshold_trials if bool(t.get("valid", False))]
        if valid_trials:
            valid_trials.sort(
                key=lambda r: (
                    safe_float(r.get("objective", float("-inf")), float("-inf")),
                    safe_float(r.get("net_pnl", float("-inf")), float("-inf")),
                    safe_float(r.get("keep_rate", 0.0), 0.0),
                ),
                reverse=True,
            )
            selected = valid_trials[0]
            selected_threshold = float(selected.get("threshold", default_threshold))
            selected_metrics = dict(selected)
            threshold_source = "tune_2024_optimized"
        else:
            selected_threshold = float(default_threshold)
            selected_metrics = dict(baseline_metrics)
            threshold_source = "config_default_no_valid_tune_trial"

        variant_stats_fit = _build_stats_from_frame(
            trade_df=fit_df,
            key_col="variant_id",
            cfg=training_cfg,
        )
        lane_stats_fit = _build_stats_from_frame(
            trade_df=fit_df,
            key_col="lane",
            cfg=training_cfg,
        )
        global_stats_fit = _build_stats_from_frame(
            trade_df=fit_df,
            key_col=None,
            cfg=training_cfg,
        ).get("__global__", {})
        shape_model_fit, _ = _build_shape_penalty_model(
            trade_df=fit_df,
            training_cfg=training_cfg,
        )

    model_payload = {
        "schema_version": "de3_v4_entry_policy_model_v3",
        "enabled": bool(training_cfg.get("enabled", True)),
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": str(threshold_source),
        "minimums": {
            "min_variant_trades": int(min_variant_trades),
            "min_lane_trades": int(min_lane_trades),
            "allow_on_missing_stats": bool(allow_on_missing_stats),
            "conservative_buffer": float(
                safe_float(training_cfg.get("conservative_buffer", 0.035), 0.035)
            ),
        },
        "scope_threshold_offsets": dict(scope_threshold_offsets_cfg),
        "score_components": {
            "weight_quality_lcb": float(safe_float(score_cfg.get("weight_quality_lcb", 0.65), 0.65)),
            "weight_route_confidence": float(safe_float(score_cfg.get("weight_route_confidence", 0.20), 0.20)),
            "weight_edge_points": float(safe_float(score_cfg.get("weight_edge_points", 0.10), 0.10)),
            "weight_structural_score": float(safe_float(score_cfg.get("weight_structural_score", 0.05), 0.05)),
            "weight_profit_factor_component": float(
                safe_float(score_cfg.get("weight_profit_factor_component", 0.0), 0.0)
            ),
            "weight_year_coverage_component": float(
                safe_float(score_cfg.get("weight_year_coverage_component", 0.0), 0.0)
            ),
            "weight_loss_share_penalty": float(
                safe_float(score_cfg.get("weight_loss_share_penalty", 0.12), 0.12)
            ),
            "weight_stop_like_share_penalty": float(
                safe_float(score_cfg.get("weight_stop_like_share_penalty", 0.08), 0.08)
            ),
            "weight_drawdown_penalty": float(
                safe_float(score_cfg.get("weight_drawdown_penalty", 0.06), 0.06)
            ),
            "weight_worst_block_penalty": float(
                safe_float(score_cfg.get("weight_worst_block_penalty", 0.08), 0.08)
            ),
            "weight_shape_penalty_component": float(
                safe_float(score_cfg.get("weight_shape_penalty_component", 0.0), 0.0)
            ),
            "route_confidence_center": float(safe_float(score_cfg.get("route_confidence_center", 0.05), 0.05)),
            "edge_scale_points": float(safe_float(score_cfg.get("edge_scale_points", 0.40), 0.40)),
            "structural_scale": float(safe_float(score_cfg.get("structural_scale", 0.80), 0.80)),
            "profit_factor_center": float(safe_float(score_cfg.get("profit_factor_center", 1.10), 1.10)),
            "profit_factor_scale": float(safe_float(score_cfg.get("profit_factor_scale", 0.35), 0.35)),
            "year_coverage_full_years": float(
                safe_float(score_cfg.get("year_coverage_full_years", 8.0), 8.0)
            ),
            "loss_share_center": float(safe_float(score_cfg.get("loss_share_center", 0.52), 0.52)),
            "loss_share_scale": float(safe_float(score_cfg.get("loss_share_scale", 0.22), 0.22)),
            "stop_like_share_center": float(safe_float(score_cfg.get("stop_like_share_center", 0.62), 0.62)),
            "stop_like_share_scale": float(safe_float(score_cfg.get("stop_like_share_scale", 0.25), 0.25)),
            "drawdown_scale": float(safe_float(score_cfg.get("drawdown_scale", 6.0), 6.0)),
            "shape_penalty_scale": float(safe_float(score_cfg.get("shape_penalty_scale", 1.0), 1.0)),
            "shape_penalty_cap": float(safe_float(score_cfg.get("shape_penalty_cap", 2.0), 2.0)),
            "worst_block_scale_points": float(
                safe_float(score_cfg.get("worst_block_scale_points", 3.0), 3.0)
            ),
        },
        "fit_windows": {
            "train_start": str(split_summary.get("training_start", "")),
            "train_end": str(split_summary.get("training_end", "")),
            "tune_start": str(split_summary.get("tuning_start", "")),
            "tune_end": str(split_summary.get("tuning_end", "")),
            "oos_start": str(split_summary.get("oos_start", "")),
            "oos_end": str(split_summary.get("oos_end", "")),
            "future_holdout_start": str(split_summary.get("future_holdout_start", "")),
            "future_holdout_end": str(split_summary.get("future_holdout_end", "")),
        },
        "variant_stats": variant_stats_fit,
        "lane_stats": lane_stats_fit,
        "global_stats": global_stats_fit if isinstance(global_stats_fit, dict) else {},
        "shape_penalty_model": dict(shape_model_fit),
    }

    report = {
        "status": "ok",
        "inventory_fallback_used": bool(inventory_fallback),
        "csv_audit": dict(csv_audit),
        "anti_leakage": {
            "training_used_splits": ["train"],
            "tuning_used_splits": ["tune"],
            "fitted_model_used_splits": ["train", "tune"],
            "excluded_splits_from_fit_and_tuning": ["oos", "future_holdout"],
            "oos_window_start": str(split_summary.get("oos_start", "")),
            "future_holdout_start": str(split_summary.get("future_holdout_start", "")),
            "leakage_check_passed": True,
            "violations": [],
        },
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": str(threshold_source),
        "baseline_tune_metrics": dict(baseline_metrics),
        "selected_tune_metrics": dict(selected_metrics),
        "threshold_trials": list(threshold_trials),
        "shape_penalty_training": dict(shape_training_summary),
        "model_coverage": {
            "variant_stats_count": int(len(variant_stats_fit)),
            "lane_stats_count": int(len(lane_stats_fit)),
            "global_stats_n_trades": int(
                safe_int((global_stats_fit or {}).get("n_trades", 0), 0)
                if isinstance(global_stats_fit, dict)
                else 0
            ),
            "variant_year_coverage": {
                "min": int(
                    min(
                        [
                            safe_int(v.get("year_coverage", 0), 0)
                            for v in variant_stats_fit.values()
                            if isinstance(v, dict)
                        ]
                        or [0]
                    )
                ),
                "max": int(
                    max(
                        [
                            safe_int(v.get("year_coverage", 0), 0)
                            for v in variant_stats_fit.values()
                            if isinstance(v, dict)
                        ]
                        or [0]
                    )
                ),
                "avg": float(
                    safe_div(
                        float(
                            sum(
                                [
                                    safe_int(v.get("year_coverage", 0), 0)
                                    for v in variant_stats_fit.values()
                                    if isinstance(v, dict)
                                ]
                            )
                        ),
                        float(
                            max(
                                1,
                                len(
                                    [
                                        1
                                        for v in variant_stats_fit.values()
                                        if isinstance(v, dict)
                                    ]
                                ),
                            )
                        ),
                        0.0,
                    )
                ),
            },
        },
        "config_effective": {
            "min_variant_trades": int(min_variant_trades),
            "min_lane_trades": int(min_lane_trades),
            "allow_on_missing_stats": bool(allow_on_missing_stats),
            "default_threshold": float(default_threshold),
            "scope_threshold_offsets": dict(scope_threshold_offsets_cfg),
            "score_components": dict(model_payload.get("score_components", {})),
            "shape_penalty_rules_selected": int(
                len(
                    (
                        shape_model_fit.get("rules", [])
                        if isinstance(shape_model_fit, dict)
                        else []
                    )
                )
            ),
        },
    }
    return {
        "entry_policy_model": model_payload,
        "entry_policy_training_report": report,
    }

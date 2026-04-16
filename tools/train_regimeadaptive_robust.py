import argparse
import datetime as dt
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

if sys.platform.startswith("win"):
    _platform_machine = str(os.environ.get("PROCESSOR_ARCHITECTURE", "") or "").strip()
    if _platform_machine:
        platform.machine = lambda: _platform_machine  # type: ignore[assignment]

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from tools.regimeadaptive_filterless_runner import (
    NY_TZ,
    SESSION_NAMES,
    _atr_array,
    _build_combo_arrays,
    _combo_key_from_id,
    _load_bars,
    _rolling_cache,
    _round_points_to_tick,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if text:
            out.append(int(text))
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if text:
            out.append(float(text))
    return out


def _parse_date(raw: str, is_end: bool) -> pd.Timestamp:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Date text is empty.")
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    if len(text) <= 10:
        if is_end:
            ts = ts.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts


def _build_split_map(index: pd.DatetimeIndex, valid_years: int, test_years: int) -> tuple[np.ndarray, list[str], dict]:
    years = sorted({int(ts.year) for ts in index})
    if len(years) <= max(valid_years + test_years, 2):
        raise ValueError("Not enough years in range for the requested validation/test split.")
    test_set = set(years[-test_years:])
    valid_set = set(years[-(valid_years + test_years):-test_years])
    split = np.empty(len(index), dtype=object)
    split[:] = "train"
    years_arr = index.year.to_numpy(dtype=np.int16)
    split[np.isin(years_arr, list(valid_set))] = "valid"
    split[np.isin(years_arr, list(test_set))] = "test"
    return split, ["train", "valid", "test"], {
        "train_years": [year for year in years if year not in valid_set and year not in test_set],
        "valid_years": sorted(valid_set),
        "test_years": sorted(test_set),
    }


def _build_event_frame(
    df: pd.DataFrame,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    split_labels: np.ndarray,
    sma_fast: int,
    sma_slow: int,
    cross_atr_mult: float,
    max_hold_bars: int,
    point_value: float,
    fee_per_trade: float,
) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    atr_period = int((CONFIG.get("REGIME_ADAPTIVE_TUNING", {}) or {}).get("atr_period", 20) or 20)
    atr = _atr_array(high, low, close, atr_period)
    rolling = _rolling_cache(close, [sma_fast, sma_slow])
    sma_fast_arr = rolling[sma_fast]
    sma_slow_arr = rolling[sma_slow]

    valid = (
        np.isfinite(sma_fast_arr)
        & np.isfinite(sma_slow_arr)
        & np.isfinite(atr)
        & (session_codes != (len(SESSION_NAMES) - 1))
    )
    cross_thresh = atr * float(cross_atr_mult)
    long_mask = valid & (sma_fast_arr > sma_slow_arr) & (close < (sma_fast_arr - cross_thresh))
    short_mask = valid & (sma_fast_arr < sma_slow_arr) & (close > (sma_fast_arr + cross_thresh))
    signal_side = np.zeros(len(close), dtype=np.int8)
    signal_side[long_mask] = 1
    signal_side[short_mask] = -1

    signal_idx = np.flatnonzero(signal_side != 0)
    signal_idx = signal_idx[(signal_idx + 1) < len(df)]
    signal_idx = signal_idx[(signal_idx + max_hold_bars) < len(df)]
    if signal_idx.size == 0:
        return pd.DataFrame()

    entry_idx = signal_idx + 1
    exit_idx = signal_idx + max_hold_bars
    entry_price = open_[entry_idx]
    exit_price = close[exit_idx]
    move_points = exit_price - entry_price
    long_pnl = (move_points * point_value) - fee_per_trade
    short_pnl = (-move_points * point_value) - fee_per_trade
    signal_side_sel = signal_side[signal_idx]
    normal_pnl = np.where(signal_side_sel > 0, long_pnl, short_pnl)
    reversed_pnl = np.where(signal_side_sel > 0, short_pnl, long_pnl)

    return pd.DataFrame(
        {
            "signal_idx": signal_idx.astype(np.int32),
            "entry_idx": entry_idx.astype(np.int32),
            "combo_id": combo_ids[signal_idx].astype(np.int16),
            "session_code": session_codes[signal_idx].astype(np.int8),
            "split": split_labels[signal_idx],
            "signal_side": signal_side_sel.astype(np.int8),
            "normal_pnl": normal_pnl.astype(np.float64),
            "reversed_pnl": reversed_pnl.astype(np.float64),
        }
    )


def _select_combo_policies(
    events: pd.DataFrame,
    split_names: list[str],
    prior_strength: float,
    min_total_trades: int,
    min_recent_trades: int,
    min_split_edge: float,
    min_train_edge: float,
    min_policy_advantage: float,
    instability_penalty: float,
    min_positive_oos_splits: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    session_split_means = (
        events.groupby(["session_code", "split"])[["normal_pnl", "reversed_pnl"]]
        .mean()
        .to_dict("index")
    )
    global_split_means = events.groupby("split")[["normal_pnl", "reversed_pnl"]].mean().to_dict("index")
    records: list[dict] = []

    for combo_id, group in events.groupby("combo_id", sort=False):
        session_code = int(group["session_code"].iloc[0])
        total_trades = int(len(group))
        recent_trades = int((group["split"] == "test").sum())
        combo_key = _combo_key_from_id(int(combo_id))

        policy_scores = {}
        policy_payload = {}
        for policy_name, pnl_col in (("normal", "normal_pnl"), ("reversed", "reversed_pnl")):
            split_means = {}
            split_counts = {}
            shrunk_means = {}
            for split_name in split_names:
                split_vals = group.loc[group["split"] == split_name, pnl_col].to_numpy(dtype=np.float64)
                split_n = int(split_vals.size)
                split_mean = float(split_vals.mean()) if split_n else 0.0
                prior_bucket = session_split_means.get((session_code, split_name), global_split_means.get(split_name, {}))
                prior_mean = float(prior_bucket.get(pnl_col, 0.0) or 0.0) if isinstance(prior_bucket, dict) else 0.0
                split_sum = float(split_vals.sum()) if split_n else 0.0
                shrunk_mean = (split_sum + (prior_strength * prior_mean)) / float(split_n + prior_strength)
                split_means[split_name] = split_mean
                split_counts[split_name] = split_n
                shrunk_means[split_name] = shrunk_mean
            oos_values = [shrunk_means[name] for name in ("valid", "test") if split_counts.get(name, 0) > 0]
            robust_edge = min(oos_values) if oos_values else shrunk_means.get("train", 0.0)
            stability_values = [shrunk_means[name] for name in split_names if split_counts.get(name, 0) > 0]
            instability = float(np.std(stability_values)) if len(stability_values) >= 2 else 0.0
            positive_oos = sum(1 for name in ("valid", "test") if split_counts.get(name, 0) > 0 and shrunk_means.get(name, 0.0) > min_split_edge)
            score = robust_edge - (instability_penalty * instability)
            policy_scores[policy_name] = score
            policy_payload[policy_name] = {
                "score": score,
                "robust_edge": robust_edge,
                "instability": instability,
                "positive_oos": positive_oos,
                "split_means": split_means,
                "split_counts": split_counts,
                "shrunk_means": shrunk_means,
            }

        best_policy = "normal" if policy_scores["normal"] >= policy_scores["reversed"] else "reversed"
        other_policy = "reversed" if best_policy == "normal" else "normal"
        best_payload = policy_payload[best_policy]
        score_gap = float(best_payload["score"] - policy_payload[other_policy]["score"])

        selected_policy = "skip"
        train_edge = float(best_payload["shrunk_means"].get("train", 0.0))
        if total_trades >= min_total_trades and recent_trades >= min_recent_trades:
            if train_edge >= min_train_edge and best_payload["positive_oos"] >= min_positive_oos_splits:
                if best_payload["robust_edge"] > min_split_edge and score_gap >= min_policy_advantage:
                    selected_policy = best_policy

        selected_score = float(best_payload["score"]) if selected_policy != "skip" else 0.0
        if selected_policy == "skip":
            quality_tier = 4
        elif total_trades >= 100 and selected_score >= 10.0:
            quality_tier = 1
        elif total_trades >= 60 and selected_score >= 5.0:
            quality_tier = 2
        else:
            quality_tier = 3

        records.append(
            {
                "combo_id": int(combo_id),
                "combo_key": combo_key,
                "session_code": session_code,
                "session": SESSION_NAMES[session_code],
                "policy": selected_policy,
                "quality_tier": quality_tier,
                "support_total": total_trades,
                "support_recent": recent_trades,
                "score_normal": float(policy_scores["normal"]),
                "score_reversed": float(policy_scores["reversed"]),
                "selected_score": selected_score,
                "score_gap": score_gap,
                "normal_train_mean": float(policy_payload["normal"]["split_means"].get("train", 0.0)),
                "normal_valid_mean": float(policy_payload["normal"]["split_means"].get("valid", 0.0)),
                "normal_test_mean": float(policy_payload["normal"]["split_means"].get("test", 0.0)),
                "reversed_train_mean": float(policy_payload["reversed"]["split_means"].get("train", 0.0)),
                "reversed_valid_mean": float(policy_payload["reversed"]["split_means"].get("valid", 0.0)),
                "reversed_test_mean": float(policy_payload["reversed"]["split_means"].get("test", 0.0)),
                "normal_train_shrunk": float(policy_payload["normal"]["shrunk_means"].get("train", 0.0)),
                "normal_valid_shrunk": float(policy_payload["normal"]["shrunk_means"].get("valid", 0.0)),
                "normal_test_shrunk": float(policy_payload["normal"]["shrunk_means"].get("test", 0.0)),
                "reversed_train_shrunk": float(policy_payload["reversed"]["shrunk_means"].get("train", 0.0)),
                "reversed_valid_shrunk": float(policy_payload["reversed"]["shrunk_means"].get("valid", 0.0)),
                "reversed_test_shrunk": float(policy_payload["reversed"]["shrunk_means"].get("test", 0.0)),
            }
        )

    return pd.DataFrame.from_records(records)


def _evaluate_param_set(events: pd.DataFrame, policy_df: pd.DataFrame) -> dict:
    if events.empty or policy_df.empty:
        return {
            "score": float("-inf"),
            "train_total": 0.0,
            "valid_total": 0.0,
            "test_total": 0.0,
            "selected_combos": 0,
            "selected_trades": 0,
        }
    merged = events.merge(policy_df[["combo_id", "policy"]], on="combo_id", how="left")
    merged = merged[merged["policy"].isin(["normal", "reversed"])].copy()
    if merged.empty:
        return {
            "score": float("-inf"),
            "train_total": 0.0,
            "valid_total": 0.0,
            "test_total": 0.0,
            "selected_combos": 0,
            "selected_trades": 0,
        }
    merged["selected_pnl"] = np.where(
        merged["policy"].eq("reversed"),
        merged["reversed_pnl"],
        merged["normal_pnl"],
    )
    split_totals = merged.groupby("split")["selected_pnl"].sum().to_dict()
    train_total = float(split_totals.get("train", 0.0))
    valid_total = float(split_totals.get("valid", 0.0))
    test_total = float(split_totals.get("test", 0.0))
    return {
        "score": 0.0,  # overwritten by caller with score weights
        "train_total": train_total,
        "valid_total": valid_total,
        "test_total": test_total,
        "selected_combos": int(policy_df["policy"].isin(["normal", "reversed"]).sum()),
        "selected_trades": int(len(merged)),
    }


def _simulate_event_bracket(
    signal_idx: int,
    side: int,
    sl_dist: float,
    tp_dist: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    max_hold_bars: int,
    point_value: float,
    fee_per_trade: float,
    gap_fills: bool,
) -> float:
    entry_bar = int(signal_idx) + 1
    if entry_bar >= len(opens):
        return 0.0
    entry_price = float(opens[entry_bar])
    stop_price = entry_price - sl_dist if side > 0 else entry_price + sl_dist
    take_price = entry_price + tp_dist if side > 0 else entry_price - tp_dist
    end_bar = min(len(opens) - 1, entry_bar + max_hold_bars - 1)

    for bar in range(entry_bar, end_bar + 1):
        bar_open = float(opens[bar])
        bar_high = float(highs[bar])
        bar_low = float(lows[bar])
        bar_close = float(closes[bar])
        if bar > entry_bar and gap_fills:
            if side > 0:
                if bar_open <= stop_price:
                    exit_price = stop_price
                    break
                if bar_open >= take_price:
                    exit_price = take_price
                    break
            else:
                if bar_open >= stop_price:
                    exit_price = stop_price
                    break
                if bar_open <= take_price:
                    exit_price = take_price
                    break
        hit_stop = bar_low <= stop_price if side > 0 else bar_high >= stop_price
        hit_take = bar_high >= take_price if side > 0 else bar_low <= take_price
        if hit_stop and hit_take:
            exit_price = stop_price if ((bar_close >= bar_open) == (side > 0)) else take_price
            break
        if hit_stop:
            exit_price = stop_price
            break
        if hit_take:
            exit_price = take_price
            break
    else:
        exit_price = float(closes[end_bar])

    pnl_points = exit_price - entry_price if side > 0 else entry_price - exit_price
    return (pnl_points * point_value) - fee_per_trade


def _fit_bracket_defaults(
    selected_events: pd.DataFrame,
    df: pd.DataFrame,
    max_hold_bars: int,
    point_value: float,
    fee_per_trade: float,
    min_bracket_trades: int,
    sl_candidates: list[float],
    tp_candidates: list[float],
) -> tuple[dict, dict]:
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    gap_fills = bool((CONFIG.get("BACKTEST_EXECUTION", {}) or {}).get("gap_fills", True))

    candidate_pairs = [
        (_round_points_to_tick(sl), _round_points_to_tick(tp))
        for sl in sl_candidates
        for tp in tp_candidates
        if float(tp) >= max(float(sl), 1.5)
    ]

    def choose_best(group: pd.DataFrame) -> Optional[dict]:
        if group.empty or len(group) < min_bracket_trades:
            return None
        best = None
        for sl_dist, tp_dist in candidate_pairs:
            pnl = [
                _simulate_event_bracket(
                    int(row.signal_idx),
                    int(row.selected_side),
                    sl_dist,
                    tp_dist,
                    opens,
                    highs,
                    lows,
                    closes,
                    max_hold_bars,
                    point_value,
                    fee_per_trade,
                    gap_fills,
                )
                for row in group.itertuples(index=False)
            ]
            trial = group[["split"]].copy()
            trial["pnl"] = pnl
            split_means = trial.groupby("split")["pnl"].mean().to_dict()
            split_totals = trial.groupby("split")["pnl"].sum().to_dict()
            oos_values = [float(split_means.get(name, 0.0)) for name in ("valid", "test") if name in split_means]
            robust_edge = min(oos_values) if oos_values else float(split_means.get("train", 0.0))
            score = float(split_totals.get("test", 0.0) + (0.5 * split_totals.get("valid", 0.0)))
            payload = {
                "sl": float(sl_dist),
                "tp": float(tp_dist),
                "score": score,
                "robust_edge": robust_edge,
                "support": int(len(group)),
            }
            if best is None or payload["score"] > best["score"] or (
                payload["score"] == best["score"] and payload["robust_edge"] > best["robust_edge"]
            ):
                best = payload
        return best

    global_default = {}
    for side_name, side_value in (("LONG", 1), ("SHORT", -1)):
        group = selected_events[selected_events["selected_side"] == side_value]
        best = choose_best(group)
        if best:
            global_default[side_name] = {"sl": best["sl"], "tp": best["tp"], "support": best["support"]}
        else:
            global_default[side_name] = {"sl": 2.0, "tp": 3.0, "support": 0}

    session_defaults: dict[str, dict[str, dict]] = {}
    for session_name in SESSION_NAMES[:-1]:
        session_group = selected_events[selected_events["session"] == session_name]
        if session_group.empty:
            continue
        side_map = {}
        for side_name, side_value in (("LONG", 1), ("SHORT", -1)):
            best = choose_best(session_group[session_group["selected_side"] == side_value])
            if best:
                side_map[side_name] = {"sl": best["sl"], "tp": best["tp"], "support": best["support"]}
            else:
                side_map[side_name] = dict(global_default[side_name])
        session_defaults[session_name] = side_map

    return global_default, session_defaults


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a robust RegimeAdaptive artifact with normal/reversed/skip combo policies on clean parquet."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--sma-fast-values", default="20,34,50")
    parser.add_argument("--sma-slow-values", default="200,300")
    parser.add_argument("--cross-atr-mults", default="2.0,4.0,6.0,8.0")
    parser.add_argument("--max-hold-bars", type=int, default=30)
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--min-total-trades", type=int, default=30)
    parser.add_argument("--min-recent-trades", type=int, default=5)
    parser.add_argument("--min-split-edge", type=float, default=1.0)
    parser.add_argument("--min-train-edge", type=float, default=0.25)
    parser.add_argument("--min-policy-advantage", type=float, default=1.0)
    parser.add_argument("--instability-penalty", type=float, default=0.25)
    parser.add_argument("--min-positive-oos-splits", type=int, default=2)
    parser.add_argument("--train-score-weight", type=float, default=0.25)
    parser.add_argument("--min-bracket-trades", type=int, default=25)
    parser.add_argument("--sl-candidates", default="1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--tp-candidates", default="2.0,3.0,4.0,6.0")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--write-latest", action="store_true")
    args = parser.parse_args()

    start_time = _parse_date(args.start, is_end=False)
    end_time = _parse_date(args.end, is_end=True)
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_trade = fee_per_side * 2.0 * int(args.contracts)

    df, symbol_label = _load_bars(source, args.symbol_mode, args.symbol_method)
    df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    if df.empty:
        raise SystemExit("No data in requested range.")
    combo_ids, session_codes = _build_combo_arrays(df.index)
    split_labels, split_names, split_meta = _build_split_map(df.index, int(args.valid_years), int(args.test_years))

    sma_fast_values = _parse_int_list(args.sma_fast_values)
    sma_slow_values = _parse_int_list(args.sma_slow_values)
    cross_atr_mults = _parse_float_list(args.cross_atr_mults)
    sl_candidates = _parse_float_list(args.sl_candidates)
    tp_candidates = _parse_float_list(args.tp_candidates)

    candidate_summaries = []
    best = None
    best_events = None
    best_policy_df = None
    for sma_fast in sma_fast_values:
        for sma_slow in sma_slow_values:
            if sma_fast <= 0 or sma_slow <= 0 or sma_fast >= sma_slow:
                continue
            for cross_atr_mult in cross_atr_mults:
                events = _build_event_frame(
                    df,
                    combo_ids,
                    session_codes,
                    split_labels,
                    sma_fast=sma_fast,
                    sma_slow=sma_slow,
                    cross_atr_mult=cross_atr_mult,
                    max_hold_bars=int(args.max_hold_bars),
                    point_value=point_value,
                    fee_per_trade=fee_per_trade,
                )
                if events.empty:
                    continue
                policy_df = _select_combo_policies(
                    events,
                    split_names,
                    prior_strength=float(args.prior_strength),
                    min_total_trades=int(args.min_total_trades),
                    min_recent_trades=int(args.min_recent_trades),
                    min_split_edge=float(args.min_split_edge),
                    min_train_edge=float(args.min_train_edge),
                    min_policy_advantage=float(args.min_policy_advantage),
                    instability_penalty=float(args.instability_penalty),
                    min_positive_oos_splits=int(args.min_positive_oos_splits),
                )
                summary = _evaluate_param_set(events, policy_df)
                if math.isfinite(float(summary["score"])):
                    summary["score"] = float(
                        summary["test_total"]
                        + (0.5 * summary["valid_total"])
                        + (float(args.train_score_weight) * summary["train_total"])
                    )
                summary.update(
                    {
                        "sma_fast": int(sma_fast),
                        "sma_slow": int(sma_slow),
                        "cross_atr_mult": float(cross_atr_mult),
                        "signal_count": int(len(events)),
                    }
                )
                candidate_summaries.append(summary)
                print(
                    f"sma_fast={sma_fast} sma_slow={sma_slow} cross_atr={cross_atr_mult:.2f} "
                    f"score={summary['score']:.2f} valid={summary['valid_total']:.2f} "
                    f"test={summary['test_total']:.2f} combos={summary['selected_combos']} trades={summary['selected_trades']}"
                )
                if best is None or summary["score"] > best["score"]:
                    best = summary
                    best_events = events
                    best_policy_df = policy_df

    if best is None or best_events is None or best_policy_df is None:
        raise SystemExit("No valid parameter candidates produced training events.")

    selected = best_events.merge(best_policy_df[["combo_id", "policy"]], on="combo_id", how="left")
    selected = selected[selected["policy"].isin(["normal", "reversed"])].copy()
    selected["selected_side"] = np.where(
        selected["policy"].eq("reversed"),
        -selected["signal_side"],
        selected["signal_side"],
    ).astype(np.int8)
    selected["session"] = selected["session_code"].map({idx: name for idx, name in enumerate(SESSION_NAMES)})
    global_default, session_defaults = _fit_bracket_defaults(
        selected,
        df,
        max_hold_bars=int(args.max_hold_bars),
        point_value=point_value,
        fee_per_trade=fee_per_trade,
        min_bracket_trades=int(args.min_bracket_trades),
        sl_candidates=sl_candidates,
        tp_candidates=tp_candidates,
    )

    combo_policies = {}
    for row in best_policy_df.itertuples(index=False):
        combo_policies[str(row.combo_key)] = {
            "policy": str(row.policy),
            "quality_tier": int(row.quality_tier),
            "support_total": int(row.support_total),
            "support_recent": int(row.support_recent),
            "selected_score": float(row.selected_score),
            "score_gap": float(row.score_gap),
            "score_normal": float(row.score_normal),
            "score_reversed": float(row.score_reversed),
            "normal_train_shrunk": float(row.normal_train_shrunk),
            "normal_valid_shrunk": float(row.normal_valid_shrunk),
            "normal_test_shrunk": float(row.normal_test_shrunk),
            "reversed_train_shrunk": float(row.reversed_train_shrunk),
            "reversed_valid_shrunk": float(row.reversed_valid_shrunk),
            "reversed_test_shrunk": float(row.reversed_test_shrunk),
        }

    if str(args.artifact_root or "").strip():
        artifact_root = Path(args.artifact_root).expanduser().resolve()
    else:
        artifact_root = ROOT / "artifacts" / f"regimeadaptive_robust_{dt.datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S')}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "regimeadaptive_robust_artifact.json"

    payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "version": "regimeadaptive_robust_v1",
        "source_data_path": str(source),
        "symbol": symbol_label,
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "split_meta": split_meta,
        "training_config": {
            "max_hold_bars": int(args.max_hold_bars),
            "contracts": int(args.contracts),
            "prior_strength": float(args.prior_strength),
            "min_total_trades": int(args.min_total_trades),
            "min_recent_trades": int(args.min_recent_trades),
            "min_split_edge": float(args.min_split_edge),
            "min_train_edge": float(args.min_train_edge),
            "min_policy_advantage": float(args.min_policy_advantage),
            "instability_penalty": float(args.instability_penalty),
            "min_positive_oos_splits": int(args.min_positive_oos_splits),
            "train_score_weight": float(args.train_score_weight),
            "min_bracket_trades": int(args.min_bracket_trades),
        },
        "base_rule": {
            "sma_fast": int(best["sma_fast"]),
            "sma_slow": int(best["sma_slow"]),
            "cross_atr_mult": float(best["cross_atr_mult"]),
            "atr_period": int((CONFIG.get("REGIME_ADAPTIVE_TUNING", {}) or {}).get("atr_period", 20) or 20),
            "max_hold_bars": int(args.max_hold_bars),
        },
        "candidate_summaries": candidate_summaries,
        "best_candidate": best,
        "global_default": {
            side: {"sl": float(values["sl"]), "tp": float(values["tp"]), "support": int(values.get("support", 0))}
            for side, values in global_default.items()
        },
        "session_defaults": {
            session: {
                side: {"sl": float(values["sl"]), "tp": float(values["tp"]), "support": int(values.get("support", 0))}
                for side, values in side_map.items()
            }
            for session, side_map in session_defaults.items()
        },
        "summary": {
            "selected_combo_count": int(best_policy_df["policy"].isin(["normal", "reversed"]).sum()),
            "reversed_combo_count": int((best_policy_df["policy"] == "reversed").sum()),
            "skipped_combo_count": int((best_policy_df["policy"] == "skip").sum()),
            "selected_trade_count": int(len(selected)),
        },
        "combo_policies": combo_policies,
    }

    artifact_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"artifact={artifact_path}")
    if bool(args.write_latest):
        latest_dir = ROOT / "artifacts" / "regimeadaptive_robust"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_path = latest_dir / "latest.json"
        shutil.copyfile(artifact_path, latest_path)
        print(f"latest={latest_path}")


if __name__ == "__main__":
    main()

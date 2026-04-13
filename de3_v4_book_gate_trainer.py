from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from de3_v4_entry_policy_trainer import _assign_splits, _split_bounds
from de3_v4_schema import safe_float, safe_int


BOOK_GATE_SCHEMA_VERSION = "de3_v4_book_gate_model_v2"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _normalize_book_gate_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return str(value).strip().lower()
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        numeric = float(value)
    except Exception:
        return str(value).strip().lower()
    if not math.isfinite(numeric):
        return ""
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-9:
        return str(int(rounded))
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _bucket_key_from_row(row: pd.Series, fields: List[str]) -> str:
    if not fields:
        return "__global__"
    parts: List[str] = []
    for field in fields:
        key = str(field or "").strip()
        if not key:
            continue
        value = _normalize_book_gate_value(row.get(key, ""))
        if not value:
            return ""
        parts.append(f"{key}={value}")
    return "|".join(parts)


def _extract_split_summary(base_bundle: Dict[str, Any]) -> Dict[str, Any]:
    meta = base_bundle.get("metadata", {}) if isinstance(base_bundle.get("metadata"), dict) else {}
    training_split = meta.get("training_split", {}) if isinstance(meta.get("training_split"), dict) else {}
    if training_split:
        return dict(training_split)
    decision_fit = (
        (base_bundle.get("decision_policy_training_report", {}) or {}).get("fit_windows", {})
        if isinstance(base_bundle.get("decision_policy_training_report", {}), dict)
        else {}
    )
    if isinstance(decision_fit, dict) and decision_fit:
        return {
            "training_start": decision_fit.get("train_start", "2011-01-01"),
            "training_end": decision_fit.get("train_end", "2023-12-31"),
            "tuning_start": decision_fit.get("tune_start", "2024-01-01"),
            "tuning_end": decision_fit.get("tune_end", "2024-12-31"),
            "oos_start": decision_fit.get("oos_start", "2025-01-01"),
            "oos_end": decision_fit.get("oos_end", "2025-12-31"),
            "future_holdout_start": decision_fit.get("future_holdout_start", "2026-01-01"),
            "future_holdout_end": decision_fit.get("future_holdout_end", ""),
        }
    raise ValueError("Could not extract split summary from the base bundle.")


def _decision_frame_from_exports(
    *,
    decisions_csv_path: str,
    trade_attribution_csv_path: str,
    book_name: str,
) -> pd.DataFrame:
    decision_cols = [
        "decision_id",
        "timestamp",
        "rank",
        "chosen",
        "abstained",
        "session",
        "ctx_volatility_regime",
        "ctx_chop_trend_regime",
        "ctx_compression_expansion_regime",
        "ctx_confidence_band",
        "ctx_rvol_liquidity_state",
        "ctx_session_substate",
        "ctx_price_location",
        "ctx_hour_et",
        "side_considered",
        "timeframe",
        "strategy_type",
        "sub_strategy",
        "de3_v4_selected_variant_id",
        "de3_v4_route_decision",
    ]
    decisions_path = Path(str(decisions_csv_path)).expanduser().resolve()
    if not decisions_path.is_file():
        raise FileNotFoundError(f"Decisions CSV not found: {decisions_path}")
    decisions = pd.read_csv(
        decisions_path,
        usecols=lambda c: c in set(decision_cols),
        low_memory=False,
    )
    if decisions.empty:
        raise ValueError(f"Decisions CSV is empty: {decisions_path}")
    decisions["decision_id"] = decisions["decision_id"].astype(str)
    decisions["rank"] = decisions["rank"].apply(lambda v: safe_int(v, 999999))
    decisions["chosen"] = decisions["chosen"].map(_coerce_bool)
    decisions["abstained"] = decisions["abstained"].map(_coerce_bool)
    decisions["de3_v4_selected_variant_id"] = (
        decisions.get("de3_v4_selected_variant_id", "").fillna("").astype(str).str.strip()
    )
    decisions["de3_v4_route_decision"] = (
        decisions.get("de3_v4_route_decision", "").fillna("").astype(str).str.strip()
    )
    decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], utc=True, errors="coerce")
    decisions = decisions.dropna(subset=["timestamp"]).copy()
    decisions = decisions.sort_values(["decision_id", "rank", "timestamp"])

    base_rows = decisions.drop_duplicates("decision_id", keep="first").copy()
    runtime_context_rows = decisions[
        (decisions["de3_v4_selected_variant_id"] != "")
        | (decisions["de3_v4_route_decision"] != "")
    ].drop_duplicates("decision_id", keep="first").copy()
    chosen_rows = decisions[decisions["chosen"]].drop_duplicates("decision_id", keep="first").copy()
    runtime_context_ids: set[str] = set()
    if not runtime_context_rows.empty:
        runtime_context_rows = runtime_context_rows.set_index("decision_id")
        runtime_context_ids = set(runtime_context_rows.index.astype(str))
        base_rows = base_rows.set_index("decision_id")
        for col in [
            "side_considered",
            "timeframe",
            "strategy_type",
            "sub_strategy",
            "de3_v4_selected_variant_id",
        ]:
            if col in runtime_context_rows.columns:
                base_rows[col] = runtime_context_rows[col].combine_first(base_rows.get(col))
        base_rows = base_rows.reset_index()
    if not chosen_rows.empty:
        chosen_rows = chosen_rows.set_index("decision_id")
        base_rows = base_rows.set_index("decision_id")
        for col in [
            "side_considered",
            "timeframe",
            "strategy_type",
            "sub_strategy",
            "de3_v4_selected_variant_id",
        ]:
            if col in chosen_rows.columns:
                base_rows[col] = chosen_rows[col].combine_first(base_rows.get(col))
        chosen_ids = set(chosen_rows.index.astype(str))
        base_rows["chosen"] = base_rows.index.to_series().astype(str).map(lambda v: v in chosen_ids)
        base_rows["abstained"] = (~base_rows["chosen"]).astype(bool)
        base_rows = base_rows.reset_index()
    if runtime_context_ids:
        base_rows["has_runtime_context"] = base_rows["decision_id"].astype(str).map(
            lambda v: v in runtime_context_ids
        )
    else:
        base_rows["has_runtime_context"] = False
    no_runtime_context_mask = (~base_rows["chosen"]) & (~base_rows["has_runtime_context"])
    for col in [
        "side_considered",
        "timeframe",
        "strategy_type",
        "sub_strategy",
        "de3_v4_selected_variant_id",
    ]:
        if col in base_rows.columns:
            base_rows.loc[no_runtime_context_mask, col] = ""
    base_rows = base_rows.drop(columns=["has_runtime_context"], errors="ignore")

    trade_cols = ["decision_id", "realized_pnl", "side"]
    trade_path = Path(str(trade_attribution_csv_path)).expanduser().resolve()
    if not trade_path.is_file():
        raise FileNotFoundError(f"Trade attribution CSV not found: {trade_path}")
    trades = pd.read_csv(
        trade_path,
        usecols=lambda c: c in set(trade_cols),
        low_memory=False,
    )
    if trades.empty:
        trade_agg = pd.DataFrame(columns=["decision_id", "decision_pnl", "trade_count", "trade_side"])
    else:
        trades["decision_id"] = trades["decision_id"].astype(str)
        trades["realized_pnl"] = trades["realized_pnl"].apply(lambda v: safe_float(v, 0.0))
        trade_agg = (
            trades.groupby("decision_id", as_index=False)
            .agg(
                decision_pnl=("realized_pnl", "sum"),
                trade_count=("realized_pnl", "size"),
                trade_side=("side", "first"),
            )
        )

    merged = base_rows.merge(trade_agg, on="decision_id", how="left")
    merged["decision_pnl"] = merged["decision_pnl"].fillna(0.0).astype(float)
    merged["trade_count"] = merged["trade_count"].fillna(0).astype(int)
    merged["trade_taken"] = merged["trade_count"] > 0
    merged["selected_side"] = merged["trade_side"].fillna(merged.get("side_considered")).fillna("").astype(str).str.lower()
    merged["selected_variant_id"] = (
        merged["sub_strategy"].fillna(merged.get("de3_v4_selected_variant_id")).fillna("").astype(str)
    )
    merged["timeframe"] = merged.get("timeframe", "").fillna("").astype(str).str.strip().str.lower()
    merged["strategy_type"] = merged.get("strategy_type", "").fillna("").astype(str).str.strip().str.lower()
    merged["side_considered"] = merged.get("side_considered", "").fillna("").astype(str).str.strip().str.lower()
    merged["sub_strategy"] = merged.get("sub_strategy", "").fillna("").astype(str).str.strip()
    merged["ts"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    merged = merged.dropna(subset=["ts"]).copy()
    merged["decision_key"] = merged["ts"].astype("int64").astype(str)
    for field in [
        "session",
        "ctx_volatility_regime",
        "ctx_chop_trend_regime",
        "ctx_compression_expansion_regime",
        "ctx_confidence_band",
        "ctx_rvol_liquidity_state",
        "ctx_session_substate",
        "ctx_price_location",
        "ctx_hour_et",
        "timeframe",
        "strategy_type",
        "side_considered",
    ]:
        merged[field] = merged[field].map(_normalize_book_gate_value)
    merged = merged[
        [
            "decision_id",
            "decision_key",
            "ts",
            "session",
            "ctx_volatility_regime",
            "ctx_chop_trend_regime",
            "ctx_compression_expansion_regime",
            "ctx_confidence_band",
            "ctx_rvol_liquidity_state",
            "ctx_session_substate",
            "ctx_price_location",
            "ctx_hour_et",
            "timeframe",
            "strategy_type",
            "side_considered",
            "sub_strategy",
            "decision_pnl",
            "trade_taken",
            "selected_side",
            "selected_variant_id",
        ]
    ].copy()
    merged = merged.rename(
        columns={
            "decision_pnl": f"{book_name}__decision_pnl",
            "trade_taken": f"{book_name}__trade_taken",
            "selected_side": f"{book_name}__selected_side",
            "selected_variant_id": f"{book_name}__selected_variant_id",
        }
    )
    return merged


def _profit_factor(pnls: pd.Series) -> float:
    values = [safe_float(v, 0.0) for v in pnls.tolist() if math.isfinite(safe_float(v, 0.0))]
    gross_win = sum(v for v in values if v > 0.0)
    gross_loss = -sum(v for v in values if v < 0.0)
    if gross_loss <= 0.0:
        return float("inf") if gross_win > 0.0 else 0.0
    return float(gross_win / gross_loss)


def _sqn(pnls: pd.Series) -> float:
    values = [safe_float(v, 0.0) for v in pnls.tolist() if abs(safe_float(v, 0.0)) > 1e-12]
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return float((mean / std) * math.sqrt(n))


def _daily_pnl_series(ts: pd.Series, pnl: pd.Series) -> pd.Series:
    if ts.empty:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"ts": ts, "pnl": pnl})
    df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
    return df.groupby("day")["pnl"].sum().astype(float)


def _daily_sharpe_sortino(ts: pd.Series, pnl: pd.Series) -> Dict[str, float]:
    daily = _daily_pnl_series(ts, pnl)
    if len(daily) < 2:
        return {"daily_sharpe": 0.0, "daily_sortino": 0.0}
    mean = float(daily.mean())
    std = float(daily.std(ddof=1))
    downside = daily[daily < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
    sharpe = 0.0 if std <= 1e-12 else float((mean / std) * math.sqrt(252.0))
    sortino = 0.0 if downside_std <= 1e-12 else float((mean / downside_std) * math.sqrt(252.0))
    return {"daily_sharpe": sharpe, "daily_sortino": sortino}


def _max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    peak = equity.cummax()
    return float((peak - equity).max())


def _evaluate_book(df: pd.DataFrame, book_name: str) -> Dict[str, float]:
    pnl_col = f"{book_name}__decision_pnl"
    trade_col = f"{book_name}__trade_taken"
    side_col = f"{book_name}__selected_side"
    pnls = df[pnl_col].astype(float) if pnl_col in df.columns else pd.Series(dtype=float)
    trades = df[trade_col].astype(bool) if trade_col in df.columns else pd.Series(dtype=bool)
    sides = df[side_col].astype(str).str.lower() if side_col in df.columns else pd.Series(dtype=str)
    trade_pnls = pnls[trades] if not pnls.empty and not trades.empty else pd.Series(dtype=float)
    risk = _daily_sharpe_sortino(df["ts"], pnls)
    long_trades = int(((trades) & (sides == "long")).sum()) if len(trades) else 0
    short_trades = int(((trades) & (sides == "short")).sum()) if len(trades) else 0
    trade_count = int(trades.sum()) if len(trades) else 0
    long_share = float(long_trades / trade_count) if trade_count > 0 else 0.0
    return {
        "decision_count": int(len(df)),
        "trade_count": int(trade_count),
        "net_pnl": float(pnls.sum()) if len(pnls) else 0.0,
        "avg_pnl_per_decision": float(pnls.mean()) if len(pnls) else 0.0,
        "avg_pnl_per_trade": float(trade_pnls.mean()) if len(trade_pnls) else 0.0,
        "profit_factor": float(_profit_factor(trade_pnls)),
        "sqn": float(_sqn(trade_pnls)),
        "max_drawdown": float(_max_drawdown(pnls)),
        "daily_sharpe": float(risk["daily_sharpe"]),
        "daily_sortino": float(risk["daily_sortino"]),
        "keep_rate": float(trade_count / len(df)) if len(df) else 0.0,
        "long_share": float(long_share),
        "short_share": float(1.0 - long_share) if trade_count > 0 else 0.0,
        "long_trades": int(long_trades),
        "short_trades": int(short_trades),
    }


def _simulate_gate(
    df: pd.DataFrame,
    *,
    scope_priority: List[Dict[str, Any]],
    bucket_overrides: Dict[str, Dict[str, Dict[str, Any]]],
    default_book: str,
) -> Dict[str, Any]:
    if df.empty:
        metrics = _evaluate_book(df.assign(**{f"{default_book}__decision_pnl": []}), default_book)
        return {
            "metrics": metrics,
            "selected_books": [],
            "override_count": 0,
            "override_rate": 0.0,
            "selected_book_counts": {},
        }

    selected_books: List[str] = []
    decision_pnls: List[float] = []
    selected_sides: List[str] = []
    selected_trade_taken: List[bool] = []

    for _, row in df.iterrows():
        chosen_book = str(default_book)
        for scope in scope_priority:
            fields = (
                [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
                if isinstance(scope.get("fields"), list)
                else []
            )
            scope_name = str(scope.get("name", "") or "").strip()
            if not scope_name:
                scope_name = "__".join(fields) if fields else "global"
            bucket_key = _bucket_key_from_row(row, fields)
            if not bucket_key:
                continue
            scope_map = bucket_overrides.get(scope_name, {})
            payload = scope_map.get(bucket_key, {}) if isinstance(scope_map.get(bucket_key, {}), dict) else {}
            book_name = str(payload.get("book", "") or "").strip()
            if book_name:
                chosen_book = str(book_name)
                break
        selected_books.append(chosen_book)
        decision_pnls.append(float(safe_float(row.get(f"{chosen_book}__decision_pnl", 0.0), 0.0)))
        selected_sides.append(str(row.get(f"{chosen_book}__selected_side", "") or "").strip().lower())
        selected_trade_taken.append(bool(row.get(f"{chosen_book}__trade_taken", False)))

    sim = pd.DataFrame(
        {
            "ts": df["ts"].tolist(),
            "selected_book": selected_books,
            "decision_pnl": decision_pnls,
            "trade_taken": selected_trade_taken,
            "selected_side": selected_sides,
        }
    )
    sim[f"{default_book}__decision_pnl"] = sim["decision_pnl"]
    sim[f"{default_book}__trade_taken"] = sim["trade_taken"]
    sim[f"{default_book}__selected_side"] = sim["selected_side"]
    metrics = _evaluate_book(sim, default_book)
    selected_counts = sim["selected_book"].value_counts().to_dict()
    override_count = int((sim["selected_book"] != str(default_book)).sum())
    return {
        "metrics": metrics,
        "selected_books": selected_books,
        "override_count": int(override_count),
        "override_rate": float(override_count / len(sim)) if len(sim) else 0.0,
        "selected_book_counts": {str(k): int(v) for k, v in selected_counts.items()},
    }


def _score_override_candidate(
    *,
    baseline_stats: Dict[str, float],
    candidate_stats: Dict[str, float],
    rules_cfg: Dict[str, Any],
) -> float:
    avg_delta = float(candidate_stats["avg_pnl_per_decision"] - baseline_stats["avg_pnl_per_decision"])
    pf_delta = float(candidate_stats["profit_factor"] - baseline_stats["profit_factor"])
    sharpe_delta = float(candidate_stats["daily_sharpe"] - baseline_stats["daily_sharpe"])
    long_share_reduction = float(baseline_stats["long_share"] - candidate_stats["long_share"])
    return (
        (float(safe_float(rules_cfg.get("score_weight_avg_pnl_delta", 1.0), 1.0)) * avg_delta)
        + (float(safe_float(rules_cfg.get("score_weight_profit_factor_delta", 18.0), 18.0)) * pf_delta)
        + (float(safe_float(rules_cfg.get("score_weight_daily_sharpe_delta", 2.5), 2.5)) * sharpe_delta)
        + (float(safe_float(rules_cfg.get("score_weight_long_share_reduction", 4.0), 4.0)) * long_share_reduction)
    )


def _build_bucket_overrides(
    df: pd.DataFrame,
    *,
    default_book: str,
    alt_books: List[str],
    scope_priority: List[Dict[str, Any]],
    rules_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    bucket_overrides: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if df.empty:
        return bucket_overrides

    for raw_scope in scope_priority:
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = (
            [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
            if isinstance(scope.get("fields"), list)
            else []
        )
        scope_name = str(scope.get("name", "") or "").strip()
        if not scope_name:
            scope_name = "__".join(fields) if fields else "global"
        min_decisions = max(
            1,
            safe_int(scope.get("min_decisions", rules_cfg.get("min_train_decisions", 180)), 180),
        )
        bucket_df = df.copy()
        bucket_df["_bucket_key"] = bucket_df.apply(lambda row: _bucket_key_from_row(row, fields), axis=1)
        bucket_df = bucket_df[bucket_df["_bucket_key"] != ""].copy()
        if bucket_df.empty:
            continue
        scope_map: Dict[str, Dict[str, Any]] = {}
        for bucket_key, group in bucket_df.groupby("_bucket_key", sort=False):
            if len(group) < min_decisions:
                continue
            baseline_stats = _evaluate_book(group, default_book)
            if baseline_stats["decision_count"] < min_decisions:
                continue
            best_book = ""
            best_score = float("-inf")
            best_stats: Dict[str, float] = {}
            for alt_book in alt_books:
                alt_stats = _evaluate_book(group, alt_book)
                if alt_stats["trade_count"] < max(1, safe_int(rules_cfg.get("min_alt_trades", 24), 24)):
                    continue
                if alt_stats["profit_factor"] < float(safe_float(rules_cfg.get("min_alt_profit_factor", 1.04), 1.04)):
                    continue
                avg_delta = float(alt_stats["avg_pnl_per_decision"] - baseline_stats["avg_pnl_per_decision"])
                pf_delta = float(alt_stats["profit_factor"] - baseline_stats["profit_factor"])
                if avg_delta < float(safe_float(rules_cfg.get("min_avg_pnl_delta", 1.25), 1.25)):
                    continue
                if pf_delta < float(safe_float(rules_cfg.get("min_profit_factor_delta", 0.01), 0.01)):
                    continue
                score = _score_override_candidate(
                    baseline_stats=baseline_stats,
                    candidate_stats=alt_stats,
                    rules_cfg=rules_cfg,
                )
                if score > best_score:
                    best_book = str(alt_book)
                    best_score = float(score)
                    best_stats = dict(alt_stats)
            if not best_book:
                continue
            scope_map[str(bucket_key)] = {
                "book": str(best_book),
                "score": float(best_score),
                "baseline_stats": dict(baseline_stats),
                "candidate_stats": dict(best_stats),
            }
        if scope_map:
            bucket_overrides[str(scope_name)] = scope_map
    return bucket_overrides


def _objective_score(
    *,
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    override_rate: float,
    objective_cfg: Dict[str, Any],
) -> float:
    return (
        (metrics["net_pnl"] - baseline_metrics["net_pnl"])
        + (float(safe_float(objective_cfg.get("weight_profit_factor", 4200.0), 4200.0)) * (metrics["profit_factor"] - baseline_metrics["profit_factor"]))
        + (float(safe_float(objective_cfg.get("weight_daily_sharpe", 900.0), 900.0)) * (metrics["daily_sharpe"] - baseline_metrics["daily_sharpe"]))
        + (float(safe_float(objective_cfg.get("weight_daily_sortino", 420.0), 420.0)) * (metrics["daily_sortino"] - baseline_metrics["daily_sortino"]))
        + (float(safe_float(objective_cfg.get("weight_sqn", 250.0), 250.0)) * (metrics["sqn"] - baseline_metrics["sqn"]))
        - (float(safe_float(objective_cfg.get("weight_max_drawdown", 0.85), 0.85)) * (metrics["max_drawdown"] - baseline_metrics["max_drawdown"]))
        + (float(safe_float(objective_cfg.get("weight_long_share_reduction", 1400.0), 1400.0)) * (baseline_metrics["long_share"] - metrics["long_share"]))
        - (float(safe_float(objective_cfg.get("weight_override_rate", 900.0), 900.0)) * override_rate)
    )


def train_de3_v4_book_gate(
    *,
    base_bundle: Dict[str, Any],
    books: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not books:
        raise ValueError("books is required")
    split_summary = _extract_split_summary(base_bundle)
    bounds = _split_bounds(split_summary)
    base_book = next((book for book in books if bool(book.get("is_default", False))), books[0])
    default_book = str(base_book.get("name", "") or "").strip()
    if not default_book:
        raise ValueError("Default book name is required.")
    alt_books = [str(book.get("name", "") or "").strip() for book in books if str(book.get("name", "") or "").strip() and str(book.get("name", "") or "").strip() != default_book]

    base_frame = _decision_frame_from_exports(
        decisions_csv_path=str(base_book.get("decisions_csv_path", "")),
        trade_attribution_csv_path=str(base_book.get("trade_attribution_csv_path", "")),
        book_name=default_book,
    )
    merged = base_frame.copy()
    for book in books:
        book_name = str(book.get("name", "") or "").strip()
        if not book_name or book_name == default_book:
            continue
        other = _decision_frame_from_exports(
            decisions_csv_path=str(book.get("decisions_csv_path", "")),
            trade_attribution_csv_path=str(book.get("trade_attribution_csv_path", "")),
            book_name=book_name,
        )
        other_cols = [
            "decision_key",
            f"{book_name}__decision_pnl",
            f"{book_name}__trade_taken",
            f"{book_name}__selected_side",
            f"{book_name}__selected_variant_id",
        ]
        merged = merged.merge(other[other_cols], on="decision_key", how="left")
        merged[f"{book_name}__decision_pnl"] = merged[f"{book_name}__decision_pnl"].fillna(0.0).astype(float)
        merged[f"{book_name}__trade_taken"] = merged[f"{book_name}__trade_taken"].fillna(False).astype(bool)
        merged[f"{book_name}__selected_side"] = merged[f"{book_name}__selected_side"].fillna("").astype(str)
        merged[f"{book_name}__selected_variant_id"] = merged[f"{book_name}__selected_variant_id"].fillna("").astype(str)

    merged = merged.sort_values("ts").reset_index(drop=True)
    merged["split"] = _assign_splits(merged["ts"], bounds)
    merged = merged[merged["split"] != ""].copy()

    train_df = merged[merged["split"] == "train"].copy()
    tune_df = merged[merged["split"] == "tune"].copy()
    fit_df = merged[merged["split"].isin(["train", "tune"])].copy()

    scope_priority = (
        cfg.get("scope_priority", [])
        if isinstance(cfg.get("scope_priority"), list)
        else []
    )
    rules_cfg = cfg.get("override_rules", {}) if isinstance(cfg.get("override_rules"), dict) else {}
    objective_cfg = cfg.get("tune_objective", {}) if isinstance(cfg.get("tune_objective"), dict) else {}

    train_bucket_overrides = _build_bucket_overrides(
        train_df,
        default_book=default_book,
        alt_books=alt_books,
        scope_priority=scope_priority,
        rules_cfg=rules_cfg,
    )
    tune_baseline = _evaluate_book(tune_df, default_book)
    tune_sim = _simulate_gate(
        tune_df,
        scope_priority=scope_priority,
        bucket_overrides=train_bucket_overrides,
        default_book=default_book,
    )
    tune_score = _objective_score(
        metrics=tune_sim["metrics"],
        baseline_metrics=tune_baseline,
        override_rate=float(tune_sim["override_rate"]),
        objective_cfg=objective_cfg,
    )

    fit_bucket_overrides = _build_bucket_overrides(
        fit_df,
        default_book=default_book,
        alt_books=alt_books,
        scope_priority=scope_priority,
        rules_cfg=rules_cfg,
    )
    fit_baseline = _evaluate_book(fit_df, default_book)
    fit_sim = _simulate_gate(
        fit_df,
        scope_priority=scope_priority,
        bucket_overrides=fit_bucket_overrides,
        default_book=default_book,
    )

    books_payload: Dict[str, Dict[str, Any]] = {}
    for book in books:
        book_name = str(book.get("name", "") or "").strip()
        if not book_name:
            continue
        books_payload[book_name] = {
            "label": str(book.get("label", book_name)),
            "decision_policy_model": (
                dict(book.get("decision_policy_model", {}))
                if isinstance(book.get("decision_policy_model"), dict)
                else {}
            ),
            "candidate_variant_filter": (
                dict(book.get("candidate_variant_filter", {}))
                if isinstance(book.get("candidate_variant_filter"), dict)
                else {}
            ),
        }

    model_payload = {
        "enabled": True,
        "schema_version": BOOK_GATE_SCHEMA_VERSION,
        "default_book": str(default_book),
        "books": books_payload,
        "scope_priority": list(scope_priority),
        "bucket_overrides": fit_bucket_overrides,
        "fallback_to_default_on_empty": bool(cfg.get("fallback_to_default_on_empty", True)),
        "override_rules": dict(rules_cfg),
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
        "candidate_profile_name": str(cfg.get("name", "") or ""),
    }
    report = {
        "status": "ok",
        "schema_version": BOOK_GATE_SCHEMA_VERSION,
        "candidate_profile_name": str(cfg.get("name", "") or ""),
        "default_book": str(default_book),
        "books": [str(book.get("name", "") or "") for book in books if str(book.get("name", "") or "").strip()],
        "split_summary": dict(split_summary),
        "train_rows": int(len(train_df)),
        "tune_rows": int(len(tune_df)),
        "fit_rows": int(len(fit_df)),
        "train_bucket_override_count": int(
            sum(len(scope_map) for scope_map in train_bucket_overrides.values())
        ),
        "fit_bucket_override_count": int(
            sum(len(scope_map) for scope_map in fit_bucket_overrides.values())
        ),
        "tune_baseline_metrics": dict(tune_baseline),
        "tune_gate_metrics": dict(tune_sim["metrics"]),
        "tune_selected_book_counts": dict(tune_sim["selected_book_counts"]),
        "tune_override_rate": float(tune_sim["override_rate"]),
        "tune_objective_score": float(tune_score),
        "fit_baseline_metrics": dict(fit_baseline),
        "fit_gate_metrics": dict(fit_sim["metrics"]),
        "fit_selected_book_counts": dict(fit_sim["selected_book_counts"]),
        "fit_override_rate": float(fit_sim["override_rate"]),
    }
    return {
        "book_gate_model": model_payload,
        "book_gate_training_report": report,
    }

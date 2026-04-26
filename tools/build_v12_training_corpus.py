#!/usr/bin/env python3
"""V12 corpus builder.

Starts from artifacts/v11_training_corpus_with_mfe.parquet (3,438 candidates)
and adds:
  1) **Hydrated Kalshi snapshot features** (13 columns) from
     `kalshi_history_provider` — point-in-time settlement-hour midpoint
     probability, sentiment momentum, and aggregate skew. v11 had these as
     neutral defaults; v12 actually queries the day-level archive.
  2) **DE3 shock-context features** (`ctx_shock_*`, `ctx_day_*`) from
     `de3_shock_context.compute_shock_context` — rule-mining primitives that
     §8.31 didn't have access to.

Output: artifacts/v12_training_corpus.parquet (preserves all v11 columns).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalshi_history_provider import HistoricalKalshiProvider, resolve_daily_dirs  # noqa: E402
from de3_shock_context import compute_shock_context  # noqa: E402

V11_CORPUS = ROOT / "artifacts" / "v11_training_corpus_with_mfe.parquet"
BARS_PARQUET = ROOT / "es_master_outrights.parquet"
OUT = ROOT / "artifacts" / "v12_training_corpus.parquet"
OUT_SUMMARY = ROOT / "artifacts" / "v12_corpus_summary.json"


def _safe_get_prob(provider: HistoricalKalshiProvider, strike: float) -> float:
    try:
        v = provider.get_probability(strike)
        return float(v) if v is not None and math.isfinite(float(v)) else float("nan")
    except Exception:
        return float("nan")


def hydrate_kalshi_features(corpus: pd.DataFrame) -> pd.DataFrame:
    """Hydrate 13 Kalshi snapshot features by day-level lookup. Uses
    settlement-hour midpoint probability at the strike nearest entry_price,
    plus probabilities at +/- 5/10/25 strike offsets to capture local skew."""
    daily_dirs = resolve_daily_dirs()
    print(f"[kalshi] daily_dirs: {[str(d) for d in daily_dirs]}")
    if not daily_dirs:
        print("[kalshi] WARN no daily dirs found — features will be NaN")

    provider = HistoricalKalshiProvider(daily_dirs)

    feats = {
        "k12_entry_probability": [],
        "k12_probe_probability": [],   # +5 strike
        "k12_probe_neg_probability": [],  # -5 strike
        "k12_skew_p10": [],            # entry_p - p(strike-10)
        "k12_skew_p25": [],            # entry_p - p(strike-25)
        "k12_above_5": [],             # p(strike+5)
        "k12_above_10": [],            # p(strike+10)
        "k12_below_10": [],            # p(strike-10)
        "k12_distance_to_50": [],      # |entry_p - 0.50|  (0 = uncertain market)
        "k12_momentum_5": [],
        "k12_momentum_15": [],
        "k12_window_active": [],       # 1 if hourly settlement active at ts
        "k12_data_present": [],        # 1 if any Kalshi data found
    }
    n_hyd = 0
    n_total = len(corpus)
    for ts, entry_price in zip(corpus["ts"], corpus["entry_price"]):
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("America/New_York")
        try:
            provider.set_context_time(ts_pd)
        except Exception:
            for k in feats:
                feats[k].append(float("nan"))
            continue
        # Use ES->SPX is identity here (basis_offset=0 is provider default)
        strike = float(entry_price)
        p_entry = _safe_get_prob(provider, strike)
        p_above5 = _safe_get_prob(provider, strike + 5.0)
        p_below5 = _safe_get_prob(provider, strike - 5.0)
        p_above10 = _safe_get_prob(provider, strike + 10.0)
        p_below10 = _safe_get_prob(provider, strike - 10.0)
        p_below25 = _safe_get_prob(provider, strike - 25.0)
        p_below5_alt = _safe_get_prob(provider, strike - 5.0)

        # momentum: take a couple of probability snapshots in the past minutes
        try:
            mom5 = provider.get_sentiment_momentum(strike, lookback=5)
        except Exception:
            mom5 = None
        try:
            mom15 = provider.get_sentiment_momentum(strike, lookback=15)
        except Exception:
            mom15 = None
        active_h = provider.active_settlement_hour_et()
        active = 1.0 if active_h is not None else 0.0
        data_present = 1.0 if math.isfinite(p_entry) else 0.0
        if data_present > 0.0:
            n_hyd += 1

        feats["k12_entry_probability"].append(p_entry)
        feats["k12_probe_probability"].append(p_above5)
        feats["k12_probe_neg_probability"].append(p_below5)
        feats["k12_skew_p10"].append(
            (p_entry - p_below10) if (math.isfinite(p_entry) and math.isfinite(p_below10)) else float("nan")
        )
        feats["k12_skew_p25"].append(
            (p_entry - p_below25) if (math.isfinite(p_entry) and math.isfinite(p_below25)) else float("nan")
        )
        feats["k12_above_5"].append(p_above5)
        feats["k12_above_10"].append(p_above10)
        feats["k12_below_10"].append(p_below10)
        feats["k12_distance_to_50"].append(
            abs(p_entry - 0.5) if math.isfinite(p_entry) else float("nan")
        )
        feats["k12_momentum_5"].append(float(mom5) if (mom5 is not None and math.isfinite(float(mom5))) else float("nan"))
        feats["k12_momentum_15"].append(float(mom15) if (mom15 is not None and math.isfinite(float(mom15))) else float("nan"))
        feats["k12_window_active"].append(active)
        feats["k12_data_present"].append(data_present)
    print(f"[kalshi] hydrated {n_hyd}/{n_total} = {n_hyd/max(1,n_total)*100:.1f}%")
    out = corpus.copy()
    for k, v in feats.items():
        out[k] = v
    return out


def hydrate_shock_context(corpus: pd.DataFrame) -> pd.DataFrame:
    """Compute DE3 shock-context features at each corpus timestamp.

    Pulls 1-min bars from es_master_outrights and uses the front-month contract
    pinned at signal time. This is approximate but matches what the live bot
    produces.
    """
    print("[shock] reading bars parquet (this can take a moment)...")
    bars = pd.read_parquet(BARS_PARQUET)
    if bars.index.tz is None:
        # Cannot determine — skip
        print("[shock] WARN bars tz-naive; skipping")
        return corpus
    # Group by symbol for fast lookup
    bars_idx = bars.copy()
    bars_idx["__symbol"] = bars_idx["symbol"]
    by_sym = {sym: g.sort_index() for sym, g in bars_idx.groupby("__symbol")}
    print(f"[shock] {len(by_sym)} symbols in bar data")

    new_cols_list: list[dict] = []
    n_total = len(corpus)
    n_pop = 0
    for i, (ts, contract) in enumerate(zip(corpus["ts"], corpus["contract"])):
        ts_pd = pd.Timestamp(ts)
        if ts_pd.tzinfo is None:
            ts_pd = ts_pd.tz_localize("America/New_York")
        contract = str(contract or "")
        sub = by_sym.get(contract, None)
        if sub is None or sub.empty:
            new_cols_list.append({})
            continue
        # Slice last 500 bars up to ts (incl)
        try:
            window_end = sub.loc[sub.index <= ts_pd]
        except Exception:
            new_cols_list.append({})
            continue
        if window_end.empty:
            new_cols_list.append({})
            continue
        wb = window_end.tail(500).copy()
        position = len(wb) - 1
        hour_et = ts_pd.tz_convert("America/New_York").hour
        sess_text = f"{(hour_et // 3) * 3:02d}-{((hour_et // 3) * 3 + 3) % 24:02d}"
        try:
            ctx = compute_shock_context(
                wb, position=position, session_text=sess_text,
                price_loc=None, rvol_ratio=None, recent_window=3, base_window=60,
            )
            n_pop += 1
        except Exception:
            ctx = {}
        new_cols_list.append(ctx)
        if (i + 1) % 500 == 0:
            print(f"[shock] {i+1}/{n_total}")

    # Aggregate
    keys = set()
    for d in new_cols_list:
        keys.update(d.keys())
    print(f"[shock] populated {n_pop}/{n_total} = {n_pop/max(1,n_total)*100:.1f}%; keys: {len(keys)}")
    out = corpus.copy()
    for k in sorted(keys):
        vals = []
        for d in new_cols_list:
            v = d.get(k, None)
            if v is None:
                vals.append(np.nan if k.startswith("ctx_") and (
                    "ratio" in k or "norm" in k or "score" in k or "frac" in k or "share" in k
                ) else "")
            else:
                vals.append(v)
        out[k] = vals
    return out


def main() -> None:
    print(f"[v12] reading {V11_CORPUS}")
    corpus = pd.read_parquet(V11_CORPUS)
    print(f"[v12] v11 rows: {len(corpus)}; cols: {len(corpus.columns)}")
    corpus = hydrate_kalshi_features(corpus)
    corpus = hydrate_shock_context(corpus)
    print(f"[v12] v12 rows: {len(corpus)}; cols: {len(corpus.columns)}")

    # Sanity: variance check on hydrated kalshi features
    sanity = {}
    for k in ["k12_entry_probability", "k12_skew_p10", "k12_distance_to_50"]:
        s = pd.to_numeric(corpus[k], errors="coerce")
        sanity[k] = {
            "n_valid": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUT, index=False)
    print(f"[v12] wrote {OUT}")

    summary = {
        "v11_input_rows": int(len(corpus)),
        "v12_columns": int(len(corpus.columns)),
        "kalshi_hydration_count": int(corpus["k12_data_present"].sum()),
        "kalshi_hydration_pct": float(corpus["k12_data_present"].mean() * 100.0),
        "kalshi_feature_sanity": sanity,
        "shock_context_keys": [c for c in corpus.columns if c.startswith("ctx_")],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[v12] summary: {OUT_SUMMARY}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

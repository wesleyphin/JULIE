"""Train an ML model that blocks/passes trades based on whether Kalshi's
crowd probability at the TAKE-PROFIT price suggests the trade will be
profitable.

Dual-head model:
  - Classifier (diagnostic):   P(HIT_TP) — was the trade's programmed
    take-profit actually reached?
  - Regressor (production):   E[pnl_dollars] — what was the trade's
    realized P&L, regardless of exit source?

We gate on the REGRESSOR, not the classifier. The classifier is retained
for inspection / comparison with the earlier binary-label approach.

Why regression instead of binary HIT_TP:
  A classifier trained on HIT_TP correctly identifies trades that will
  reach their programmed target, but HIT_TP is a STRICTER condition than
  "trade is profitable." Many profitable trades close early via other
  exits — manual close, reversal-signal close, Kalshi TP trail, break-
  even-armed stops. Binary HIT_TP rejects those trades during gating
  even though they make money. Rolling-origin evaluation showed a
  HIT_TP-gated overlay loses money in normal weeks (it throws out real
  profits) while helping only in bad weeks. Regressing on pnl_dollars
  directly eliminates that label mismatch.

For each historical trade with Kalshi coverage, we:
  1. Parse tp_dist from sub_strategy (e.g. '..._SL10_TP25' → tp_dist=25)
  2. Compute tp_price = entry_price ± tp_dist depending on side
  3. Look up the Kalshi settlement event covering the trade's entry hour
  4. Compute the aligned probability at tp_price via interpolation across
     the nearest strikes (mirrors _interpolated_aligned_probability in
     kalshi_trade_overlay.py)
  5. Build additional features: ladder slope near TP, strike distance,
     open interest at nearest strike, daily volume, minutes-to-settlement,
     plus the usual market-state + DE3 substrategy context
  6. Labels: HIT_TP binary + pnl_dollars continuous

Output: artifacts/signal_gate_2025/model_kalshi_tp_gate.joblib
  → payload holds BOTH classifier + regressor
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error

ROOT = Path("/Users/wes/Downloads/JULIE001")
NY = ZoneInfo("America/New_York")

# Allowlist of canonical monthly + outrageous-event replays (matches train_kalshi_ml.py)
REPLAY_ALLOWLIST = {
    "2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06",
    "2025_07", "2025_08", "2025_09", "2025_10", "2025_11", "2025_12",
    "outrageous_feb", "outrageous_apr", "outrageous_jul", "outrageous_aug",
    "outrageous_oct",
}

SCAN_ROOT = ROOT / "backtest_reports" / "full_live_replay"
KALSHI_PARQUET = ROOT / "data" / "kalshi" / "kxinxu_hourly_2025.parquet"
ES_PARQUET = ROOT / "es_master_outrights.parquet"

OUT_MODEL = ROOT / "artifacts" / "signal_gate_2025" / "model_kalshi_tp_gate.joblib"
OUT_DATA = ROOT / "artifacts" / "signal_gate_2025" / "kalshi_tp_training_data.parquet"

# Kalshi only gates during these settlement hours (matches julie001.py)
KALSHI_GATING_HOURS_ET = {12, 13, 14, 15, 16}
REGIME_WINDOW = 120
EFF_LOW = 0.05
EFF_HIGH = 0.12

# Feature schema
NUMERIC_FEATURES = [
    # TP-strike relationship (the core of this model)
    "tp_aligned_prob",        # Kalshi's P(settlement aligned with side) at tp_price
    "tp_dist_pts",            # how far TP is from entry
    "tp_prob_edge",           # tp_aligned_prob - 0.50 (edge over coinflip)
    "tp_vs_entry_prob_delta", # tp_aligned_prob - entry_aligned_prob (slope of probability)
    "nearest_strike_dist",    # |nearest strike - tp_price|
    "nearest_strike_oi",      # open interest at nearest strike
    "nearest_strike_volume",  # daily volume at nearest strike
    "ladder_slope_near_tp",   # local slope of prob vs strike around tp_price
    "minutes_to_settlement",  # how many minutes until Kalshi event resolves
    # Entry-side anchor (for scale)
    "entry_aligned_prob",
    # Market-state (from ES bars)
    "atr14_pts",
    "range_30bar_pts",
    "trend_20bar_pct",
    "vel_5bar_pts_per_min",
    # Substrategy
    "sub_tier",
    "sub_is_rev",
    "sub_is_5min",
]
CATEGORICAL_FEATURES = ["side", "regime"]

RGX_DE3_TP = re.compile(r"_SL\d+_TP(?P<tp>\d+)")
RGX_DE3_SUB = re.compile(
    r"(?P<tf>5min|15min)_.*?_(?P<direction>Long|Short)_(?P<type>Rev|Mom)_T(?P<tier>\d)"
)


def parse_tp_dist(sub_strategy: str) -> float | None:
    m = RGX_DE3_TP.search(sub_strategy or "")
    if not m:
        return None
    try:
        return float(m.group("tp"))
    except Exception:
        return None


def parse_substrategy(sub: str) -> tuple[int, bool, bool]:
    m = RGX_DE3_SUB.search(sub or "")
    if not m:
        return 0, False, False
    try:
        tier = int(m.group("tier"))
    except Exception:
        tier = 0
    return tier, m.group("type") == "Rev", m.group("tf") == "5min"


def aligned_probability(raw_prob: float, side: str) -> float:
    """Kalshi YES price = P(settle >= strike). Mirror kalshi_trade_overlay._aligned_probability."""
    if side == "LONG":
        return float(raw_prob)
    return float(1.0 - raw_prob)


def interpolated_aligned_prob(markets: list[dict], price: float, side: str) -> float | None:
    """Mirrors kalshi_trade_overlay._interpolated_aligned_probability but on
    our local (strike, probability) list extracted from the parquet."""
    if not markets or not math.isfinite(price):
        return None
    below = [r for r in markets if r["strike"] <= price]
    above = [r for r in markets if r["strike"] > price]
    if below and above:
        low, high = below[-1], above[0]
        span = float(high["strike"] - low["strike"])
        if span <= 1e-9:
            raw = float(low["prob"])
        else:
            frac = (price - low["strike"]) / span
            raw = low["prob"] * (1.0 - frac) + high["prob"] * frac
    elif below:
        raw = float(below[-1]["prob"])
    else:
        raw = float(above[0]["prob"])
    return aligned_probability(raw, side)


def pick_settlement_hour(et_hour: int) -> int | None:
    """Pick the next settlement event that covers this bar. Trades at
    10:30 ET → 11 AM event; trades at 13:05 → 14 PM event; etc."""
    for h in sorted(KALSHI_GATING_HOURS_ET):
        if et_hour < h:
            return h
    return None  # after 4 PM — no gating


def event_ticker_for(date_et: str, settlement_hour_et: int) -> str:
    """Build Kalshi event_ticker from the trade date + settlement hour."""
    dt = datetime.strptime(date_et, "%Y-%m-%d")
    return f"KXINXU-{dt.strftime('%y%b%d').upper()}H{settlement_hour_et:02d}00"


def build_markets_from_snapshot(sub_df: pd.DataFrame) -> list[dict]:
    """Convert a filtered parquet slice (one event, one snapshot_date) into
    a sorted list of {strike, prob, oi, volume} dicts."""
    out = []
    for _, r in sub_df.iterrows():
        hi = float(r["high"])
        lo = float(r["low"])
        # Midpoint of bid/ask in cents → probability 0..1
        prob = (hi + lo) / 200.0
        out.append({
            "strike": float(r["strike"]),
            "prob": prob,
            "oi": int(r["open_interest"] or 0),
            "volume": int(r["daily_volume"] or 0),
        })
    out.sort(key=lambda x: x["strike"])
    return out


class ESBarCache:
    """Load ES bars once and supply market-state features at trade entry time."""
    def __init__(self, parquet_path: Path):
        print(f"[es parquet] loading {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df = df[df.index.year >= 2024].sort_index()
        if "symbol" in df.columns and "volume" in df.columns:
            df = df.sort_values("volume", ascending=False).groupby(df.index).first().sort_index()
        self.df = df[["open", "high", "low", "close"]].copy()
        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize("UTC").tz_convert(NY)
        else:
            self.df.index = self.df.index.tz_convert(NY)
        self.closes = self.df["close"].values
        self.highs = self.df["high"].values
        self.lows = self.df["low"].values
        print(f"  {len(self.df):,} bars")

    def state_at(self, et_ts) -> dict | None:
        if et_ts.tzinfo is None:
            et_ts = et_ts.replace(tzinfo=NY)
        else:
            et_ts = et_ts.astimezone(NY)
        pos = self.df.index.searchsorted(et_ts, side="right") - 1
        if pos < REGIME_WINDOW or pos >= len(self.df):
            return None
        trs = [max(self.highs[k] - self.lows[k],
                   abs(self.highs[k] - self.closes[k - 1]),
                   abs(self.lows[k] - self.closes[k - 1])) for k in range(pos - 13, pos + 1)]
        atr14 = float(np.mean(trs))
        r30 = float(self.highs[pos - 29:pos + 1].max() - self.lows[pos - 29:pos + 1].min())
        t20 = (self.closes[pos] - self.closes[pos - 20]) / max(1.0, self.closes[pos - 20]) * 100.0
        v5 = (self.closes[pos] - self.closes[pos - 5]) / 5.0
        # Regime from 120-bar window
        win = self.closes[pos - REGIME_WINDOW + 1:pos + 1]
        rets = np.diff(win) / win[:-1]
        if len(rets) == 0:
            regime = "warmup"
        else:
            mean = rets.mean()
            var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
            vol_bp = math.sqrt(var) * 10_000.0
            abs_sum = float(np.abs(rets).sum())
            eff = float(abs(rets.sum()) / abs_sum) if abs_sum > 0 else 0.0
            if vol_bp > 3.5 and eff < EFF_LOW:
                regime = "whipsaw"
            elif eff > EFF_HIGH:
                regime = "calm_trend"
            else:
                regime = "neutral"
        return {
            "atr14_pts": atr14, "range_30bar_pts": r30,
            "trend_20bar_pct": t20, "vel_5bar_pts_per_min": v5,
            "regime": regime,
        }


def extract_features(trade: dict, kalshi_df: pd.DataFrame, bar_cache: ESBarCache) -> dict | None:
    """Build one training row from a trade + Kalshi ladder + ES bars."""
    try:
        entry_time = datetime.fromisoformat(trade["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=NY)
        et_ny = entry_time.astimezone(NY)
    except Exception:
        return None
    tp_dist = parse_tp_dist(trade.get("sub_strategy", ""))
    if tp_dist is None or tp_dist <= 0:
        return None
    side = str(trade.get("side", "")).upper()
    if side not in {"LONG", "SHORT"}:
        return None
    entry_price = float(trade.get("entry_price", 0) or 0)
    if entry_price <= 0:
        return None
    tp_price = entry_price + tp_dist if side == "LONG" else entry_price - tp_dist
    # Settlement hour
    settle_h = pick_settlement_hour(et_ny.hour)
    if settle_h is None:
        return None
    date_et = et_ny.strftime("%Y-%m-%d")
    ev_ticker = event_ticker_for(date_et, settle_h)
    # Find the parquet snapshot: snapshot_date = trade date, event_ticker match
    slice_df = kalshi_df[
        (kalshi_df["snapshot_date"] == date_et) &
        (kalshi_df["event_ticker"] == ev_ticker)
    ]
    if slice_df.empty:
        return None
    markets = build_markets_from_snapshot(slice_df)
    if len(markets) < 5:
        return None
    # Aligned probabilities
    tp_prob = interpolated_aligned_prob(markets, tp_price, side)
    entry_prob = interpolated_aligned_prob(markets, entry_price, side)
    if tp_prob is None or entry_prob is None:
        return None
    # Nearest strike to TP
    nearest = min(markets, key=lambda m: abs(m["strike"] - tp_price))
    nearest_dist = abs(nearest["strike"] - tp_price)
    # Local ladder slope: probability change per 1 point around TP (use ±5pt window)
    above_tp = [m for m in markets if tp_price < m["strike"] <= tp_price + 10]
    below_tp = [m for m in markets if tp_price - 10 <= m["strike"] <= tp_price]
    if above_tp and below_tp:
        p_up = aligned_probability(above_tp[-1]["prob"], side)
        p_dn = aligned_probability(below_tp[0]["prob"], side)
        span = max(0.5, above_tp[-1]["strike"] - below_tp[0]["strike"])
        slope = (p_up - p_dn) / span
    else:
        slope = 0.0
    # Minutes to settlement
    settle_dt = et_ny.replace(hour=settle_h, minute=0, second=0, microsecond=0)
    min_to_settle = max(0.0, (settle_dt - et_ny).total_seconds() / 60.0)
    # ES bar state
    bar_state = bar_cache.state_at(et_ny)
    if bar_state is None:
        return None
    # Substrategy
    tier, is_rev, is_5m = parse_substrategy(trade.get("sub_strategy", ""))
    # Label: hit TP (source in {take, take_gap})
    source = str(trade.get("source", "")).lower()
    hit_tp = 1 if source in {"take", "take_gap"} else 0

    return {
        "tp_aligned_prob": float(tp_prob),
        "tp_dist_pts": float(tp_dist),
        "tp_prob_edge": float(tp_prob - 0.50),
        "tp_vs_entry_prob_delta": float(tp_prob - entry_prob),
        "nearest_strike_dist": float(nearest_dist),
        "nearest_strike_oi": float(nearest["oi"]),
        "nearest_strike_volume": float(nearest["volume"]),
        "ladder_slope_near_tp": float(slope),
        "minutes_to_settlement": float(min_to_settle),
        "entry_aligned_prob": float(entry_prob),
        "atr14_pts": bar_state["atr14_pts"],
        "range_30bar_pts": bar_state["range_30bar_pts"],
        "trend_20bar_pct": bar_state["trend_20bar_pct"],
        "vel_5bar_pts_per_min": bar_state["vel_5bar_pts_per_min"],
        "sub_tier": float(tier),
        "sub_is_rev": 1.0 if is_rev else 0.0,
        "sub_is_5min": 1.0 if is_5m else 0.0,
        "side": side,
        "regime": bar_state["regime"],
        "hit_tp": hit_tp,
        "pnl_dollars": float(trade.get("pnl_dollars", 0.0) or 0.0),
        "source": source,
        "ts": et_ny.isoformat(),
    }


def collect_rows():
    print(f"[kalshi parquet] loading {KALSHI_PARQUET}")
    kalshi_df = pd.read_parquet(KALSHI_PARQUET)
    print(f"  {len(kalshi_df):,} rows")
    bar_cache = ESBarCache(ES_PARQUET)

    rows = []
    stats = {}
    for name in sorted(REPLAY_ALLOWLIST):
        replay = SCAN_ROOT / name
        ct = replay / "closed_trades.json"
        if not ct.exists():
            stats[name] = "MISSING"
            continue
        trades = json.loads(ct.read_text(encoding="utf-8"))
        kept = 0
        missed_no_tp = 0
        missed_no_kalshi = 0
        for t in trades:
            # Only DE3 trades carry sub_strategy with _TPxx
            if not str(t.get("strategy", "")).startswith("DynamicEngine3"):
                continue
            if parse_tp_dist(t.get("sub_strategy", "")) is None:
                missed_no_tp += 1
                continue
            r = extract_features(t, kalshi_df, bar_cache)
            if r is None:
                missed_no_kalshi += 1
                continue
            rows.append(r)
            kept += 1
        stats[name] = f"DE3_trades={sum(1 for t in trades if str(t.get('strategy','')).startswith('DynamicEngine3'))} kept={kept} no_tp={missed_no_tp} no_kalshi={missed_no_kalshi}"
    return rows, stats


def assemble_X(df, cat_maps=None):
    updated = dict(cat_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        if known is None:
            known = sorted(df[col].dropna().unique().tolist())
            updated[col] = known
        enc = pd.DataFrame(
            {f"{col}__{v}": (df[col] == v).astype(int) for v in known}, index=df.index
        )
        parts.append(enc)
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def rolling_origin_eval(ds, cat_maps, min_train=0.40, step=0.10):
    """Per-chunk evaluation on classifier AUC + regressor gate PnL.

    For each chunk:
      - Train classifier (hit_tp) + regressor (pnl_dollars) on pre-chunk data
      - Score the chunk
      - Report AUC and per-threshold regressor-gated PnL delta vs rule
    """
    ds_sorted = ds.sort_values("ts").reset_index(drop=True)
    n = len(ds_sorted)
    results = []
    t = min_train
    while t + step <= 1.0 + 1e-9:
        tr_end = int(n * t); te_end = min(n, int(n * (t + step)))
        if te_end - tr_end < 30: break
        tr = ds_sorted.iloc[:tr_end]; te = ds_sorted.iloc[tr_end:te_end]
        X_tr, cm = assemble_X(tr, cat_maps)
        X_te, _ = assemble_X(te, cm)
        for c in X_tr.columns:
            if c not in X_te.columns: X_te[c] = 0.0
        X_te = X_te[X_tr.columns]
        y_tr_bin = tr["hit_tp"].astype(int).values
        y_te_bin = te["hit_tp"].astype(int).values
        y_tr_pnl = tr["pnl_dollars"].values
        y_te_pnl = te["pnl_dollars"].values
        if len(set(y_tr_bin)) < 2 or len(set(y_te_bin)) < 2:
            t += step; continue
        # Classifier
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=25, random_state=42,
        )
        clf.fit(X_tr, y_tr_bin)
        p_clf = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        auc = float(roc_auc_score(y_te_bin, p_clf))
        # Regressor (the gating head)
        reg = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=25, random_state=42,
        )
        reg.fit(X_tr, y_tr_pnl)
        p_reg = reg.predict(X_te)
        rule_pnl = float(y_te_pnl.sum())
        # Regressor-gate PnL at several thresholds
        gate_deltas = {}
        for thr in (-50.0, -25.0, -10.0, 0.0, 10.0, 25.0):
            kept = p_reg > thr
            gate_pnl = float(y_te_pnl[kept].sum())
            gate_deltas[thr] = {
                "pnl": gate_pnl,
                "delta": gate_pnl - rule_pnl,
                "n_kept": int(kept.sum()),
            }
        results.append({
            "train_frac": t, "test_frac": t + step,
            "test_start": te["ts"].min(), "test_end": te["ts"].max(),
            "n_test": len(te), "hit_rate": float(y_te_bin.mean()),
            "auc_clf": auc,
            "rule_pnl": rule_pnl,
            "reg_mae": float(mean_absolute_error(y_te_pnl, p_reg)),
            "gate_deltas": gate_deltas,
        })
        t += step
    return results


def main():
    print("[kalshi_tp_ml] TP-aligned Kalshi probability ML — regressor-gated")
    rows, stats = collect_rows()
    for name in sorted(stats.keys()):
        print(f"  {name:<20}  {stats[name]}")

    if not rows:
        print("ERROR: no rows collected")
        return
    ds = pd.DataFrame(rows).dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES).reset_index(drop=True)
    print(f"\n[rows] {len(ds)}  hit_tp_rate={ds['hit_tp'].mean():.1%}  "
          f"total_pnl=${ds['pnl_dollars'].sum():+,.0f}")
    print(f"[regime]: {dict(ds['regime'].value_counts())}")
    print(f"[tp dist buckets]: {dict(ds['tp_dist_pts'].value_counts().head())}")

    X, cat_maps = assemble_X(ds)
    y_bin = ds["hit_tp"].astype(int).values
    y_pnl = ds["pnl_dollars"].values
    print(f"[features] {X.shape[1]}")

    # Rolling-origin CV — classifier AUC + regressor-gated PnL
    print(f"\n[rolling-origin CV — classifier AUC + regressor-gated PnL deltas]")
    cv = rolling_origin_eval(ds, cat_maps)
    print(f"  {'train%':>7}{'test%':>7}  {'test_start':<20}{'n':>5}{'hit_rt':>8}{'AUC':>7}"
          f"{'rule_PnL':>10}  " +
          "  ".join(f"thr>${t:<+3.0f}_Δ" for t in (-50, -25, -10, 0, 10, 25)))
    for r in cv:
        line = (f"  {r['train_frac']*100:>6.0f}%{r['test_frac']*100:>6.0f}%  "
                f"{r['test_start'][:19]:<20}{r['n_test']:>5}{r['hit_rate']:>8.1%}"
                f"{r['auc_clf']:>7.3f}{r['rule_pnl']:>+10.0f}  ")
        for t in (-50.0, -25.0, -10.0, 0.0, 10.0, 25.0):
            line += f"{r['gate_deltas'][t]['delta']:>+12.0f}  "
        print(line)

    if cv:
        mean_auc = np.mean([r['auc_clf'] for r in cv])
        total_rule = sum(r['rule_pnl'] for r in cv)
        print(f"\n  rolling-origin summary:")
        print(f"    mean classifier AUC:  {mean_auc:.3f}")
        print(f"    cumulative rule PnL across chunks: ${total_rule:+,.0f}")
        print(f"    cumulative regressor-gate Δ by threshold:")
        for t in (-50.0, -25.0, -10.0, 0.0, 10.0, 25.0):
            total_d = sum(r['gate_deltas'][t]['delta'] for r in cv)
            pos_chunks = sum(1 for r in cv if r['gate_deltas'][t]['delta'] > 0)
            total_kept = sum(r['gate_deltas'][t]['n_kept'] for r in cv)
            total_n = sum(r['n_test'] for r in cv)
            print(f"      thr > ${t:+6.0f}: Δ=${total_d:+,.0f}  "
                  f"({pos_chunks}/{len(cv)} chunks positive, "
                  f"kept {total_kept}/{total_n} = {total_kept/total_n*100:.0f}%)")

    # Fit final classifier + regressor on full data, save both
    clf_final = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=25, random_state=42,
    )
    clf_final.fit(X, y_bin)
    reg_final = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=25, random_state=42,
    )
    reg_final.fit(X, y_pnl)

    imps_clf = sorted(zip(X.columns, clf_final.feature_importances_), key=lambda t: -t[1])
    imps_reg = sorted(zip(X.columns, reg_final.feature_importances_), key=lambda t: -t[1])
    print(f"\nTop 10 classifier features (diagnostic):")
    for n_, imp in imps_clf[:10]:
        print(f"    {n_:<30} {imp:.4f}")
    print(f"\nTop 10 regressor features (production gate):")
    for n_, imp in imps_reg[:10]:
        print(f"    {n_:<30} {imp:.4f}")

    joblib.dump({
        "classifier": clf_final,
        "regressor": reg_final,
        "model_kind": "GBT_kalshi_tp_gate_clf+reg",
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_rows": len(ds),
        "rolling_origin_cv": cv,
        "rolling_origin_mean_auc": float(np.mean([r['auc_clf'] for r in cv])) if cv else None,
        "base_hit_rate": float(ds['hit_tp'].mean()),
        "allowlist": sorted(REPLAY_ALLOWLIST),
        # Gate on the regressor; default threshold is "predicted pnl > $0" but
        # tunable via JULIE_ML_KALSHI_TP_PNL_THR env var at inference time.
        "gate_mode": "regressor_pnl",
        "regressor_gate_threshold": 0.0,
        # Legacy classifier threshold retained for diagnostic scoring only
        "classifier_threshold": 0.50,
    }, OUT_MODEL)
    ds.to_parquet(OUT_DATA)
    print(f"\n[write] {OUT_MODEL}")
    print(f"[write] {OUT_DATA}")


if __name__ == "__main__":
    main()

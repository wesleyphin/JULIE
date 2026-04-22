"""Train an ML replacement for the Kalshi entry-blocker rule.

Current live rule (julie001.py _apply_kalshi_trade_overlay_to_signal):
  at signal time, compute support_score (blend of entry_probability,
  probe_probability, momentum_retention) — if score < threshold (0.45-0.55),
  block the signal. Single feature, single threshold.

ML replacement: binary classifier + pnl regressor. Given Kalshi features +
market-state context (regime, ATR, velocity, 20/30-bar ranges, DE3 sub-
strategy tier/Rev-Mom/timeframe), predict (a) P(profitable trade) and
(b) expected pnl_dollars. Learned decision surface replaces the flat
threshold.

Design decisions (iterated through v1/v2):
  - Replay ALLOWLIST — only canonical monthly + outrageous_* runs.
    Experimental variants (exp1_tp80, ny_iter*, baseline2, etc.) polluted
    v1 labels because they ran with altered filter configs.
  - Regime from parquet bars — v2 parsed `Regime transition:` logs but
    canonical replays don't actually log those lines; always returned
    'warmup'. Compute from 120-bar close window matching regime_classifier.
  - DE3 substrategy features (Rev/Mom, tier 2/3/5/7, 5min/15min).
  - Rolling-origin temporal CV (not random shuffle) — v1's 0.698 AUC was
    leaky; honest rolling-origin ~0.545.
  - Parallel regressor on pnl_dollars — binary label collapses $5 scratch
    and $200 winner into the same class; regression exploits asymmetry.

Output: artifacts/signal_gate_2025/model_kalshi_gate.joblib  (classifier
        + regressor in one payload)
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error

ROOT = Path("/Users/wes/Downloads/JULIE001")
NY = ZoneInfo("America/New_York")

REPLAY_ALLOWLIST = {
    "2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06",
    "2025_07", "2025_08", "2025_09", "2025_10", "2025_11", "2025_12",
    "outrageous_feb", "outrageous_apr", "outrageous_jul", "outrageous_aug",
    "outrageous_oct",  # outrageous_dec had 0 Kalshi views
}

SCAN_ROOT = ROOT / "backtest_reports" / "full_live_replay"
PARQUET = ROOT / "es_master_outrights.parquet"

OUT_MODEL = ROOT / "artifacts" / "signal_gate_2025" / "model_kalshi_gate.joblib"
OUT_DATA = ROOT / "artifacts" / "signal_gate_2025" / "kalshi_training_data.parquet"

# Regime classifier params (mirror regime_classifier.py)
REGIME_WINDOW = 120
EFF_LOW = 0.05
EFF_HIGH = 0.12

RGX_KALSHI_VIEW = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[INFO\] "
    r"\[(?P<ts2>[^\]]+)\] \[KALSHI_ENTRY_VIEW\].*?\| (?P<body>.*)$"
)

KALSHI_NUMERIC = [
    "entry_probability", "probe_probability", "momentum_delta",
    "momentum_retention", "support_score", "threshold",
    "probe_distance_pts", "et_hour_frac",
]
MARKET_NUMERIC = [
    "atr14_pts", "range_30bar_pts", "trend_20bar_pct",
    "dist_to_20bar_hi_pct", "dist_to_20bar_lo_pct",
    "vel_5bar_pts_per_min", "dist_to_bank_pts",
    # regime metrics from window (NEW)
    "regime_vol_bp", "regime_eff",
    # substrategy-derived (NEW)
    "sub_tier", "sub_is_rev", "sub_is_5min",
]
NUMERIC_FEATURES = KALSHI_NUMERIC + MARKET_NUMERIC
CATEGORICAL_FEATURES = ["side", "role", "regime"]  # drop 'strategy' (all DE3 now)


def parse_kv_body(body):
    out = {}
    for kv in body.split(" | "):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_log_views(log_path):
    views = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "KALSHI_ENTRY_VIEW" not in line:
                continue
            m = RGX_KALSHI_VIEW.search(line)
            if not m:
                continue
            kv = parse_kv_body(m.group("body"))
            views.append({
                "decision": kv.get("decision", ""),
                "strategy": kv.get("strategy", ""),
                "side": kv.get("side", ""),
                "entry_price": kv.get("entry_price", ""),
                "role": kv.get("role", ""),
                "entry_probability": kv.get("entry_probability"),
                "probe_price": kv.get("probe_price"),
                "probe_probability": kv.get("probe_probability"),
                "momentum_delta": kv.get("momentum_delta"),
                "momentum_retention": kv.get("momentum_retention"),
                "support_score": kv.get("support_score"),
                "threshold": kv.get("threshold"),
            })
    return views


def match_views_to_trades(views, trades):
    passes = [v for v in views if v.get("decision") == "PASS"]
    by_key = {}
    for v in passes:
        try:
            ep = round(float(v.get("entry_price", 0.0)), 2)
        except Exception:
            continue
        by_key[(v["strategy"], v["side"], ep)] = v
    matches = []
    for t in trades:
        strat = str(t.get("strategy", "")).strip()
        side = str(t.get("side", "")).upper()
        try:
            ep = round(float(t.get("entry_price", 0.0) or 0.0), 2)
        except Exception:
            continue
        v = by_key.get((strat, side, ep))
        if v is None:
            for d in (-0.25, 0.25, -0.50, 0.50):
                v = by_key.get((strat, side, round(ep + d, 2)))
                if v is not None:
                    break
        if v is not None:
            matches.append((t, v))
    return matches


class BarCache:
    def __init__(self, parquet_path):
        print(f"[parquet] loading {parquet_path}")
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
        print(f"  {len(self.df):,} bars from {self.df.index.min()} to {self.df.index.max()}")

    def features_at(self, et_ts):
        if et_ts.tzinfo is None:
            et_ts = et_ts.replace(tzinfo=NY)
        else:
            et_ts = et_ts.astimezone(NY)
        pos = self.df.index.searchsorted(et_ts, side="right") - 1
        if pos < REGIME_WINDOW or pos >= len(self.df):
            return None
        # Market state (standard)
        trs = [max(self.highs[k] - self.lows[k],
                   abs(self.highs[k] - self.closes[k - 1]),
                   abs(self.lows[k] - self.closes[k - 1])) for k in range(pos - 13, pos + 1)]
        atr14 = float(np.mean(trs))
        r30 = float(self.highs[pos - 29:pos + 1].max() - self.lows[pos - 29:pos + 1].min())
        t20 = (self.closes[pos] - self.closes[pos - 20]) / max(1.0, self.closes[pos - 20]) * 100.0
        hi20 = float(self.highs[pos - 19:pos + 1].max())
        lo20 = float(self.lows[pos - 19:pos + 1].min())
        dh = (self.closes[pos] - hi20) / max(1.0, self.closes[pos]) * 100.0
        dl = (self.closes[pos] - lo20) / max(1.0, self.closes[pos]) * 100.0
        v5 = (self.closes[pos] - self.closes[pos - 5]) / 5.0
        px = self.closes[pos]
        dist_bank = min(px - math.floor(px / 12.5) * 12.5, math.ceil(px / 12.5) * 12.5 - px)
        # Regime from 120-bar window
        win = self.closes[pos - REGIME_WINDOW + 1:pos + 1]
        rets = np.diff(win) / win[:-1]
        if len(rets) == 0:
            regime = "warmup"; vol_bp = 0.0; eff = 0.0
        else:
            mean = rets.mean()
            var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
            vol_bp = math.sqrt(var) * 10_000.0
            abs_sum = np.abs(rets).sum()
            eff = float(abs(rets.sum()) / abs_sum) if abs_sum > 0 else 0.0
            if vol_bp > 3.5 and eff < EFF_LOW:
                regime = "whipsaw"
            elif eff > EFF_HIGH:
                regime = "calm_trend"
            else:
                regime = "neutral"
        return {
            "atr14_pts": atr14, "range_30bar_pts": r30, "trend_20bar_pct": t20,
            "dist_to_20bar_hi_pct": dh, "dist_to_20bar_lo_pct": dl,
            "vel_5bar_pts_per_min": v5, "dist_to_bank_pts": float(dist_bank),
            "regime_vol_bp": float(vol_bp), "regime_eff": float(eff),
            "regime": regime,
        }


# Sub-strategy parser: DE3 sub_strategy looks like "15min_06-09_Long_Rev_T2_SL10_TP25"
RGX_DE3_SUB = re.compile(
    r"(?P<tf>5min|15min)_.*?_(?P<direction>Long|Short)_(?P<type>Rev|Mom)_T(?P<tier>\d)"
)


def parse_substrategy(sub_str):
    """Return (tier, is_rev, is_5min) — defaults 0/False/False if unparsable."""
    if not sub_str:
        return 0, False, False
    m = RGX_DE3_SUB.search(sub_str)
    if not m:
        return 0, False, False
    try:
        tier = int(m.group("tier"))
    except Exception:
        tier = 0
    return tier, m.group("type") == "Rev", m.group("tf") == "5min"


def build_features(trade, view, bar_cache):
    def fnum(x):
        try: return float(x)
        except Exception: return None
    ep = fnum(view.get("entry_price"))
    pp = fnum(view.get("probe_price"))
    if ep is None or pp is None: return None
    try:
        et = datetime.fromisoformat(trade["entry_time"])
        if et.tzinfo is None: et = et.replace(tzinfo=NY)
        et_ny = et.astimezone(NY)
    except Exception:
        return None
    mkt = bar_cache.features_at(et_ny)
    if mkt is None: return None
    tier, is_rev, is_5m = parse_substrategy(str(trade.get("sub_strategy", "") or ""))
    row = {
        "entry_probability": fnum(view.get("entry_probability")) or 0.5,
        "probe_probability": fnum(view.get("probe_probability")) or 0.5,
        "momentum_delta": fnum(view.get("momentum_delta")) or 0.0,
        "momentum_retention": fnum(view.get("momentum_retention")) or 1.0,
        "support_score": fnum(view.get("support_score")) or 0.5,
        "threshold": fnum(view.get("threshold")) or 0.45,
        "probe_distance_pts": pp - ep,
        "et_hour_frac": et_ny.hour + et_ny.minute / 60.0,
        "sub_tier": tier,
        "sub_is_rev": 1 if is_rev else 0,
        "sub_is_5min": 1 if is_5m else 0,
        **mkt,  # includes atr14_pts, range_30bar_pts, trend_20bar_pct,
                # dist_to_20bar_hi/lo_pct, vel_5bar_pts_per_min,
                # dist_to_bank_pts, regime_vol_bp, regime_eff, regime
        "side": str(trade.get("side", "")).upper(),
        "role": view.get("role") or "unknown",
        "pnl_dollars": float(trade.get("pnl_dollars", 0.0) or 0.0),
        "label": 1 if float(trade.get("pnl_dollars", 0.0) or 0.0) > 0 else 0,
        "ts": et_ny.isoformat(),
        "source_dir": view.get("_source_dir", ""),
    }
    return row


def collect_rows(bar_cache):
    rows, stats = [], {}
    for name in sorted(REPLAY_ALLOWLIST):
        replay = SCAN_ROOT / name
        ct = replay / "closed_trades.json"
        lg = replay / "topstep_live_bot.log"
        if not (ct.exists() and lg.exists()):
            stats[name] = "MISSING"; continue
        try:
            trades = json.loads(ct.read_text(encoding="utf-8"))
        except Exception:
            stats[name] = "BAD_JSON"; continue
        views = parse_log_views(lg)
        matches = match_views_to_trades(views, trades)
        kept = 0
        for t, v in matches:
            v["_source_dir"] = name
            r = build_features(t, v, bar_cache)
            if r:
                rows.append(r); kept += 1
        stats[name] = f"views={len(views)} match={len(matches)} kept={kept}"
    return rows, stats


def assemble_X(df, cat_maps=None):
    updated = dict(cat_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        if known is None:
            known = sorted(df[col].dropna().unique().tolist())
            updated[col] = known
        enc = pd.DataFrame({f"{col}__{v}": (df[col]==v).astype(int) for v in known}, index=df.index)
        parts.append(enc)
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def rolling_origin_eval(ds, cat_maps, min_train_pct=0.40, step=0.10, use_regressor=False):
    ds_sorted = ds.sort_values("ts").reset_index(drop=True)
    n = len(ds_sorted)
    results = []
    t = min_train_pct
    while t + step <= 1.0 + 1e-9:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + step)))
        if te_end - tr_end < 30: break
        tr = ds_sorted.iloc[:tr_end]; te = ds_sorted.iloc[tr_end:te_end]
        X_tr, cm = assemble_X(tr, cat_maps)
        X_te, _ = assemble_X(te, cm)
        for c in X_tr.columns:
            if c not in X_te.columns: X_te[c] = 0.0
        X_te = X_te[X_tr.columns]
        y_tr_bin = tr["label"].astype(int).values
        y_te_bin = te["label"].astype(int).values
        y_tr_pnl = tr["pnl_dollars"].values
        y_te_pnl = te["pnl_dollars"].values
        if len(set(y_tr_bin)) < 2 or len(set(y_te_bin)) < 2:
            t += step; continue
        # Classifier
        clf = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.04,
            min_samples_leaf=30, random_state=42,
        )
        clf.fit(X_tr, y_tr_bin)
        p_clf = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        auc = roc_auc_score(y_te_bin, p_clf)
        # Regressor on pnl
        reg = GradientBoostingRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.04,
            min_samples_leaf=30, random_state=42,
        )
        reg.fit(X_tr, y_tr_pnl)
        p_reg = reg.predict(X_te)
        # Eval as gate: keep trades where predicted pnl > 0 (or > $X), measure PnL
        for gate_thr in (0.0, 10.0, 25.0):
            kept_mask = p_reg > gate_thr
            if kept_mask.sum() == 0: continue
            kept_pnl = float(y_te_pnl[kept_mask].sum())
            kept_wr = float(y_te_bin[kept_mask].mean())
        # Also classifier gate at thr=0.50
        clf_kept_pnl = float(y_te_pnl[p_clf >= 0.50].sum())
        rule_pnl = float(y_te_pnl.sum())
        # Use regressor > $0 as gate
        reg_kept_pnl = float(y_te_pnl[p_reg > 0.0].sum())
        results.append({
            "train_frac": t, "test_frac": t + step,
            "test_start": te["ts"].min(), "test_end": te["ts"].max(),
            "n_test": len(te),
            "auc_clf": float(auc),
            "rule_pnl": rule_pnl,
            "clf50_pnl": clf_kept_pnl, "clf50_delta": clf_kept_pnl - rule_pnl,
            "reg0_pnl": reg_kept_pnl, "reg0_delta": reg_kept_pnl - rule_pnl,
            "reg_mae": float(mean_absolute_error(y_te_pnl, p_reg)),
        })
        t += step
    return results


def main():
    print("[kalshi_ml] regime-from-parquet + DE3 substrategy + regression head")
    print(f"[allowlist] {len(REPLAY_ALLOWLIST)} replays")
    bar_cache = BarCache(PARQUET)
    rows, stats = collect_rows(bar_cache)
    for name in sorted(stats.keys()):
        print(f"  {name:<20}  {stats[name]}")

    ds = pd.DataFrame(rows).dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES).reset_index(drop=True)
    print(f"\n[rows] {len(ds)}  win_rate={ds['label'].mean():.1%}")
    print(f"[regime dist]: {dict(ds['regime'].value_counts())}")
    print(f"[substrategy]: tier={dict(ds['sub_tier'].value_counts())} rev_frac={ds['sub_is_rev'].mean():.1%}")

    X, cat_maps = assemble_X(ds)
    y_bin = ds["label"].astype(int).values
    y_pnl = ds["pnl_dollars"].values
    print(f"[features] {X.shape[1]}  (v2 had 22, v3 should have ~25)")

    # Rolling-origin eval
    print(f"\n[rolling-origin CV — classifier AUC + regressor gate PnL]")
    cv_results = rolling_origin_eval(ds, cat_maps, min_train_pct=0.40, step=0.10)
    print(f"  {'train%':>7}{'test%':>7}  {'test_start':<20}{'n':>5}{'AUC':>7}{'rule_PnL':>10}"
          f"{'clf50_Δ':>9}{'reg>$0_Δ':>11}{'reg_MAE':>9}")
    for r in cv_results:
        print(f"  {r['train_frac']*100:>6.0f}%{r['test_frac']*100:>6.0f}%  "
              f"{r['test_start'][:19]:<20}{r['n_test']:>5}{r['auc_clf']:>7.3f}"
              f"{r['rule_pnl']:>+10.0f}{r['clf50_delta']:>+9.0f}{r['reg0_delta']:>+11.0f}{r['reg_mae']:>9.0f}")
    if cv_results:
        print(f"\n  mean AUC: {np.mean([r['auc_clf'] for r in cv_results]):.3f}")
        print(f"  mean clf@0.50 PnL delta: ${np.mean([r['clf50_delta'] for r in cv_results]):+.0f}/chunk")
        print(f"  mean reg>$0 PnL delta:   ${np.mean([r['reg0_delta'] for r in cv_results]):+.0f}/chunk")

    # Final models — train on full data
    clf_final = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.04,
        min_samples_leaf=30, random_state=42,
    )
    clf_final.fit(X, y_bin)
    reg_final = GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.04,
        min_samples_leaf=30, random_state=42,
    )
    reg_final.fit(X, y_pnl)

    imps_clf = sorted(zip(X.columns, clf_final.feature_importances_), key=lambda t: -t[1])
    imps_reg = sorted(zip(X.columns, reg_final.feature_importances_), key=lambda t: -t[1])
    print("\nTop 10 classifier features:")
    for n_, imp in imps_clf[:10]:
        print(f"    {n_:<32} {imp:.4f}")
    print("\nTop 10 regressor features:")
    for n_, imp in imps_reg[:10]:
        print(f"    {n_:<32} {imp:.4f}")

    joblib.dump({
        "classifier": clf_final,
        "regressor": reg_final,
        "model_kind": "GBT_kalshi_clf+reg",
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_rows": len(ds),
        "rolling_origin_cv": cv_results,
        "rolling_origin_mean_auc": float(np.mean([r['auc_clf'] for r in cv_results])) if cv_results else None,
        "allowlist": sorted(REPLAY_ALLOWLIST),
        "pass_threshold": 0.50,
        "regressor_gate_threshold": 0.0,
    }, OUT_MODEL)
    ds.to_parquet(OUT_DATA)
    print(f"\n[write] {OUT_MODEL}")
    print(f"[write] {OUT_DATA}")


if __name__ == "__main__":
    main()

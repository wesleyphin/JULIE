"""v2 cross-market breakout gate — real ML this time.

v1 (`train_cm_breakout_gate.py`) trained on 1977 overlay-era signals
with a noisy win/loss target and hit AUC 0.537 (barely above random).
That was an overlay-override question crammed into the wrong dataset.

v2 reframes the problem as a pure price-direction prediction:
  "Given current cross-market state, does MES move ≥X points in the
   next N minutes in direction D?"

This is a legitimate ML question with a huge dataset (810k aligned
1-min bars across MES+MNQ+VIX, Jan-2024 → Apr-2026) and a direct
target (future MES return magnitude). We train two models:

    model_cm_breakout_long.joblib   P(MES +Y pts in next N min)
    model_cm_breakout_short.joblib  P(MES -Y pts in next N min)

At inference, a LONG signal that Kalshi blocks is overridden only when
the LONG model says P ≥ threshold; same for SHORT.

Features (per bar):
    8 cross-market dims  — vix_level, vix_regime_code, vix_roc_5d,
                            mnq_ret_5m, mnq_ret_30m, mes_mnq_corr_30,
                            mes_mnq_div_pct, dxy_level
    6 MES momentum dims  — ret_5m, ret_15m, ret_30m, atr_14,
                            dist_20hi_pct, dist_20lo_pct
    4 calendar dims      — et_hour, minute_of_hour, day_of_week,
                            minutes_since_session_open
Total: 18 features per sample.

Target horizon + threshold default:
    N = 30 minutes
    Y = 5.0 MES points  (equals ≈ 0.07% on 7150 — matches typical
                         DE3 TP distance and a 'meaningful' directional
                         breakout).

Training: stratified sample of ~100k aligned bars, temporal 70/15/15
train/val/test split. GBT(n=300, d=4, lr=0.03).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"
NY = ZoneInfo("America/New_York")

MES_PARQUET = ROOT / "es_master_outrights.parquet"
MNQ_PARQUET = ROOT / "data" / "mnq_master_continuous.parquet"
VIX_PARQUET = ROOT / "data" / "vix_daily.parquet"

FORWARD_MINUTES = 30
BREAKOUT_PTS   = 5.0
SAMPLE_EVERY_N_BARS = 5   # every 5 min → ~160k samples from 810k bars

CLF_KWARGS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    min_samples_leaf=200, random_state=42,
)


def _load_front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a multi-contract bar frame to per-day front-month."""
    if "symbol" not in df.columns:
        return df.sort_index()
    # front month per day = highest-volume contract on that day
    d = df.assign(_d=df.index.date)
    day_sym = (d.groupby(["_d", "symbol"])["volume"].sum().reset_index()
                 .sort_values(["_d", "volume"], ascending=[True, False])
                 .drop_duplicates("_d", keep="first")
                 .set_index("_d")["symbol"])
    d["_front"] = d["_d"].map(day_sym)
    out = d[d["symbol"] == d["_front"]].drop(columns=["_d", "_front"]).sort_index()
    return out


def main():
    print("[load] MES master parquet…")
    mes = _load_front_month(pd.read_parquet(MES_PARQUET))
    if mes.index.tz is None:
        mes.index = mes.index.tz_localize("UTC").tz_convert(NY)
    print(f"  MES front-month bars: {len(mes):,}")

    print("[load] MNQ master parquet…")
    mnq = _load_front_month(pd.read_parquet(MNQ_PARQUET))
    if mnq.index.tz is None:
        mnq.index = mnq.index.tz_localize("UTC").tz_convert(NY)
    print(f"  MNQ front-month bars: {len(mnq):,}")

    print("[load] VIX daily parquet…")
    vix = pd.read_parquet(VIX_PARQUET).sort_index()
    if vix.index.tz is None:
        vix.index = vix.index.tz_localize("UTC").tz_convert(NY)
    print(f"  VIX bars: {len(vix):,}")

    # Restrict MES to the MNQ overlap window (2024+). Use boolean masking
    # rather than .loc[] slicing because mixed DST offsets in the MES
    # index trip pandas' slice_indexer.
    overlap_start = max(mes.index.min(), mnq.index.min())
    overlap_end   = min(mes.index.max(), mnq.index.max())
    mask = (mes.index >= overlap_start) & (mes.index <= overlap_end)
    mes = mes[mask]
    print(f"\n[align] overlap {overlap_start} → {overlap_end}  (MES bars: {len(mes):,})")

    # --- Feature + label construction ---
    print(f"\n[build] subsampling every {SAMPLE_EVERY_N_BARS} bars…")
    mes_sub = mes.iloc[::SAMPLE_EVERY_N_BARS].copy()
    print(f"  subsampled rows: {len(mes_sub):,}")

    print("[build] MES momentum features…")
    close = mes["close"].reset_index(drop=False)
    close = close.set_index("ts" if "ts" in close.columns else close.columns[0])
    # Reindex by mes's original index
    close = mes["close"]
    ret_5m  = close.pct_change(periods=5) * 100.0
    ret_15m = close.pct_change(periods=15) * 100.0
    ret_30m = close.pct_change(periods=30) * 100.0
    tr = pd.concat([
        (mes["high"] - mes["low"]).abs(),
        (mes["high"] - mes["close"].shift()).abs(),
        (mes["low"]  - mes["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=14).mean()
    hi20  = mes["high"].rolling(20, min_periods=20).max()
    lo20  = mes["low"].rolling(20, min_periods=20).min()
    dist_hi = (close - hi20) / close * 100.0
    dist_lo = (close - lo20) / close * 100.0

    # Forward label: did MES move at least BREAKOUT_PTS in direction D
    # within the next FORWARD_MINUTES?
    fwd_high = mes["high"].rolling(FORWARD_MINUTES, min_periods=1).max().shift(-FORWARD_MINUTES + 1)
    fwd_low  = mes["low"].rolling(FORWARD_MINUTES, min_periods=1).min().shift(-FORWARD_MINUTES + 1)
    long_break  = (fwd_high >= close + BREAKOUT_PTS).astype(int)
    short_break = (fwd_low  <= close - BREAKOUT_PTS).astype(int)

    # --- Cross-market features ---
    print("[build] cross-market features — this is the slow step…")
    # Compute in one pass: at each sampled bar, look up MNQ + VIX history
    mnq_close = mnq["close"]
    vix_close = vix["close"]
    # Arrays for searchsorted speed
    mnq_idx_arr = mnq.index.values
    vix_idx_arr = vix.index.values
    mnq_vals = mnq_close.values
    vix_vals = vix_close.values

    rows = []
    mes_idx = mes_sub.index
    for i, ts in enumerate(mes_idx):
        if i % 10000 == 0 and i > 0:
            print(f"  {i:,}/{len(mes_idx):,}  ({i/len(mes_idx)*100:.0f}%)")
        ts64 = np.datetime64(ts.to_numpy())
        # MNQ 30-bar window ending at ts
        p_mnq = np.searchsorted(mnq_idx_arr, ts64, side="right") - 1
        if p_mnq < 30:
            continue
        mnq_win = mnq_vals[p_mnq - 29: p_mnq + 1]
        # VIX 5-day lookback
        p_vix = np.searchsorted(vix_idx_arr, ts64, side="right") - 1
        if p_vix < 5:
            continue
        vix_now = float(vix_vals[p_vix])
        vix_5d  = float(vix_vals[p_vix - 5])
        vix_roc = (vix_now - vix_5d) / vix_5d * 100.0 if vix_5d > 0 else 0.0
        # MES 30-bar window
        p_mes = mes.index.searchsorted(ts, side="right") - 1
        if p_mes < 30:
            continue
        mes_win = mes["close"].iloc[p_mes - 29: p_mes + 1].to_numpy()
        if len(mnq_win) != 30 or len(mes_win) != 30:
            continue
        # Compute cross-market
        mnq_ret_5m  = (mnq_win[-1] - mnq_win[-6]) / mnq_win[-6] * 100.0 if mnq_win[-6] > 0 else 0.0
        mnq_ret_30m = (mnq_win[-1] - mnq_win[0]) / mnq_win[0] * 100.0 if mnq_win[0] > 0 else 0.0
        mes_ret_30m = (mes_win[-1] - mes_win[0]) / mes_win[0] * 100.0 if mes_win[0] > 0 else 0.0
        mnq_rets = np.diff(mnq_win) / mnq_win[:-1]
        mes_rets = np.diff(mes_win) / mes_win[:-1]
        if mnq_rets.std() > 1e-9 and mes_rets.std() > 1e-9:
            corr = float(np.corrcoef(mnq_rets, mes_rets)[0, 1])
            corr = corr if np.isfinite(corr) else 0.5
        else:
            corr = 0.5
        div = mes_ret_30m - mnq_ret_30m
        # Roll-day clip
        mnq_ret_5m = float(np.clip(mnq_ret_5m, -2.0, 2.0))
        mnq_ret_30m = float(np.clip(mnq_ret_30m, -5.0, 5.0))
        div = float(np.clip(div, -5.0, 5.0))
        if vix_now < 14:  vix_code = 0.0
        elif vix_now < 20: vix_code = 1.0
        elif vix_now < 30: vix_code = 2.0
        else:              vix_code = 3.0

        rows.append({
            "ts": ts,
            # cross-market
            "vix_level": vix_now, "vix_regime_code": vix_code, "vix_roc_5d": vix_roc,
            "mnq_ret_5m": mnq_ret_5m, "mnq_ret_30m": mnq_ret_30m,
            "mes_mnq_corr_30": corr, "mes_mnq_div_pct": div,
            "dxy_level": 100.0,  # stub
            # MES momentum (from precomputed series)
            "mes_ret_5m":  float(ret_5m.iloc[p_mes])  if p_mes < len(ret_5m)  else 0.0,
            "mes_ret_15m": float(ret_15m.iloc[p_mes]) if p_mes < len(ret_15m) else 0.0,
            "mes_ret_30m": float(ret_30m.iloc[p_mes]) if p_mes < len(ret_30m) else 0.0,
            "mes_atr_14":  float(atr_14.iloc[p_mes]) if p_mes < len(atr_14) else 0.0,
            "mes_dist_hi20_pct": float(dist_hi.iloc[p_mes]) if p_mes < len(dist_hi) else 0.0,
            "mes_dist_lo20_pct": float(dist_lo.iloc[p_mes]) if p_mes < len(dist_lo) else 0.0,
            # calendar
            "et_hour": float(ts.hour),
            "minute_of_hour": float(ts.minute),
            "day_of_week": float(ts.dayofweek),
            # label
            "long_break":  int(long_break.iloc[p_mes]) if p_mes < len(long_break) else 0,
            "short_break": int(short_break.iloc[p_mes]) if p_mes < len(short_break) else 0,
        })

    print(f"[build] kept {len(rows):,} rows after alignment filters")
    ds = pd.DataFrame(rows)
    # Drop any rows where labels are NaN (end of series)
    ds = ds.dropna(subset=["long_break", "short_break"]).reset_index(drop=True)
    ds.to_parquet(ARTIFACTS / "cm_breakout_v2_training.parquet", index=False)
    print(f"[write] {ARTIFACTS / 'cm_breakout_v2_training.parquet'}")
    print(f"  long_break rate:  {ds['long_break'].mean():.4f}")
    print(f"  short_break rate: {ds['short_break'].mean():.4f}")

    feat_cols = [c for c in ds.columns
                 if c not in ("ts", "long_break", "short_break")]
    X = ds[feat_cols].astype(np.float32).fillna(0.0)
    n = len(ds)

    for direction, label_col in [("long", "long_break"), ("short", "short_break")]:
        y = ds[label_col].astype(int).values
        print(f"\n=== {direction.upper()} model ===")
        print(f"  samples: {n:,}  positive rate: {y.mean():.4f}")
        # Rolling-origin A/B — 6 chunks
        rows_ab = []
        for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            tr_end = int(n * t); te_end = min(n, int(n * (t + 0.10)))
            if te_end - tr_end < 5000: continue
            if len(set(y[:tr_end])) < 2 or len(set(y[tr_end:te_end])) < 2: continue
            clf = GradientBoostingClassifier(**CLF_KWARGS).fit(X.iloc[:tr_end], y[:tr_end])
            p = clf.predict_proba(X.iloc[tr_end:te_end])[:, list(clf.classes_).index(1)]
            auc = float(roc_auc_score(y[tr_end:te_end], p))
            mask = p >= 0.60
            hit_rate = (y[tr_end:te_end][mask].sum() / mask.sum()) if mask.sum() else float("nan")
            rows_ab.append({"t": t, "n_te": te_end - tr_end, "auc": auc,
                            "p>=0.60_n": int(mask.sum()), "p>=0.60_win_rate": float(hit_rate)})
            print(f"  {int(t*100):>3}% → {int((t+0.1)*100):>3}%  n_te={te_end-tr_end:,}  AUC={auc:.3f}  "
                  f"p≥0.60: {int(mask.sum()):,} @ hit={hit_rate*100:.1f}%")
        import statistics as s
        mean_auc = s.mean(r["auc"] for r in rows_ab)
        print(f"  MEAN AUC: {mean_auc:.3f}")

        # Final model on all data
        clf_full = GradientBoostingClassifier(**CLF_KWARGS).fit(X, y)
        out = ARTIFACTS / f"model_cm_breakout_{direction}.joblib"
        joblib.dump({
            "model": clf_full,
            "model_kind": f"GBT_d4_cm_breakout_direction_{direction}_v2",
            "target": label_col,
            "direction": direction,
            "forward_minutes": FORWARD_MINUTES,
            "breakout_pts": BREAKOUT_PTS,
            "feature_names": feat_cols,
            "numeric_features": feat_cols,
            "categorical_features": [],
            "categorical_maps": {},
            "ordinal_features": [],
            "uses_cross_market": True,
            "uses_bar_encoder": False,
            "override_threshold": 0.60,
            "cv_auc_mean": mean_auc,
            "training_rows": n,
            "positive_rate": float(y.mean()),
            "ab_chunks": rows_ab,
        }, out)
        print(f"  [write] {out}")


if __name__ == "__main__":
    main()

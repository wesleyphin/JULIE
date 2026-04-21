"""Backtest all three G gates + Kalshi on October 2025.

Strategies:
  DynamicEngine3  — from 2025_10/closed_trades.json (222 trades)
  RegimeAdaptive  — from 2025_10/closed_trades.json (45 trades)
  AetherFlow      — from af_fast_replay/oct2025/closed_trades.json

Per strategy, scores each trade against:
  - Kalshi decision (12-16 ET gating, real Oct 2025 daily data)
  - G gate (strategy-specific model with mkt_regime + trend_align)

Reports:
  - Weekly breakdown (Oct 1-3, 6-10, 13-17, 20-24, 27-31)
  - Monthly totals per strategy
  - Filter stack delta (baseline vs Kalshi+G)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))

NY = ZoneInfo("America/New_York")
KALSHI_DAILY = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
KALSHI_GATING_HOURS_ET = {12, 13, 14, 15, 16}
KALSHI_ENTRY_THRESHOLD = 0.45

# Trade sources
DE3_RA_SOURCE = ROOT / "backtest_reports" / "full_live_replay" / "2025_10" / "closed_trades.json"
AF_SOURCE = ROOT / "backtest_reports" / "af_fast_replay" / "oct2025" / "closed_trades.json"

# Weekly windows — Oct 2025 had 5 partial/full weeks
WEEKS = [
    ("week1 (Oct 1-3)",  "2025-10-01", "2025-10-03"),
    ("week2 (Oct 6-10)", "2025-10-06", "2025-10-10"),
    ("week3 (Oct 13-17)","2025-10-13", "2025-10-17"),
    ("week4 (Oct 20-24)","2025-10-20", "2025-10-24"),
    ("week5 (Oct 27-31)","2025-10-27", "2025-10-31"),
]


def load_trades(path, strategy=None):
    if not path.exists():
        return []
    out = []
    for t in json.load(open(path)):
        if strategy and str(t.get("strategy")) != strategy:
            continue
        out.append(t)
    return out


def in_week(trade, start, end):
    sd = pd.Timestamp(start, tz=NY)
    ed = pd.Timestamp(end, tz=NY) + pd.Timedelta(hours=23, minutes=59)
    try:
        et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
        return sd <= pd.Timestamp(et) <= ed
    except Exception:
        return False


# v5 regime-adaptive threshold multipliers (matches signal_gate_2025.py)
_REGIME_THR_MULT = {
    "whipsaw":    0.60,
    "calm_trend": 1.05,
    "neutral":    1.0,
    "warmup":     1.0,
    "":           1.0,
}


def kalshi_decision(et_dt, entry_price, side, cache):
    h = et_dt.hour
    if h not in KALSHI_GATING_HOURS_ET:
        return ("auto_pass", None)
    d = et_dt.date().isoformat()
    if d not in cache:
        p = KALSHI_DAILY / f"{d}.parquet"
        cache[d] = pd.read_parquet(p) if p.exists() else None
    kdf = cache[d]
    if kdf is None or kdf.empty:
        return ("auto_pass", None)
    nxt = None
    for hh in (10, 11, 12, 13, 14, 15, 16):
        if hh > h:
            nxt = hh
            break
    if nxt is None:
        return ("auto_pass", None)
    sub = kdf[(kdf["settlement_hour_et"] == nxt) & (kdf["event_date"] == d)]
    if sub.empty:
        return ("auto_pass", None)
    sub = sub.copy()
    sub["dist"] = (sub["strike"] - entry_price).abs()
    row = sub.nsmallest(1, "dist").iloc[0]
    yes_prob = (float(row["high"]) + float(row["low"])) / 200.0
    aligned = yes_prob if side == "LONG" else 1.0 - yes_prob
    if aligned < KALSHI_ENTRY_THRESHOLD:
        return ("block", aligned)
    return ("pass", aligned)


def compute_features(trade, master_df, regime_fn):
    from build_de3_chosen_shape_dataset import _compute_feature_frame
    et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
    end = pd.Timestamp(et).tz_convert("UTC")
    start = end - pd.Timedelta(hours=6)
    win = master_df.loc[(master_df.index >= start) & (master_df.index <= end + pd.Timedelta(minutes=30))]
    if win.empty:
        return None, None
    sym = win["symbol"].value_counts().idxmax()
    sub = win[win["symbol"] == sym][["open", "high", "low", "close", "volume"]]
    if len(sub) < 50:
        return None, None
    feats = _compute_feature_frame(sub)
    if feats.empty:
        return None, None
    idx = feats.index.searchsorted(end)
    if idx <= 0 or idx > len(feats):
        return None, None
    feat_row = feats.iloc[idx - 1]
    if feat_row.isna().all():
        return None, None
    return feat_row, regime_fn(master_df, sym, end)


def score_g(trade, feat_row, mkt_regime, g_model, side):
    feat_names = g_model["feature_names"]
    numeric = g_model["numeric_features"]
    cat_maps = g_model.get("categorical_maps", {})
    row = {c: 0.0 for c in feat_names}
    for c in numeric:
        if c == "trend_align_ret1":
            r = feat_row.get("de3_entry_ret1_atr", 0.0)
            try:
                rf = float(r);  rf = 0.0 if not np.isfinite(rf) else rf
            except: rf = 0.0
            if rf > 0:   row[c] = 1.0 if side == "LONG" else -1.0
            elif rf < 0: row[c] = 1.0 if side == "SHORT" else -1.0
            else:        row[c] = 0.0
            continue
        v = feat_row.get(c, 0.0)
        try:
            fv = float(v);  fv = 0.0 if not np.isfinite(fv) else fv
        except: fv = 0.0
        if c in row: row[c] = fv
    et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
    h = et.hour
    sess = ("ASIA" if (18 <= h or h < 3) else "LONDON" if h < 7 else
            "NY_PRE" if h < 9 else "NY" if h < 16 else "POST")
    af_reg = str(trade.get("regime") or trade.get("aetherflow_regime") or "").strip().upper()
    if af_reg not in {"DISPERSED", "TREND_GEODESIC", "CHOP_SPIRAL", "ROTATIONAL_TURBULENCE"}:
        af_reg = ""
    cat_vals = {
        "side": side.upper(),
        "session": sess,
        "regime": af_reg,
        "mkt_regime": mkt_regime or "",
    }
    for cc, kvs in cat_maps.items():
        val = cat_vals.get(cc, "")
        for kv in kvs:
            nm = f"{cc}__{kv}"
            if nm in row and val == kv:
                row[nm] = 1
    if "et_hour" in row:
        row["et_hour"] = float(h)
    X = np.array([[row[c] for c in feat_names]])
    return float(g_model["model"].predict_proba(X)[0, 1])


def evaluate(picked):
    if not picked:
        return dict(n=0, pnl=0.0, wr=0.0)
    pnls = [float(t["pnl_dollars"]) for t in picked]
    wins = sum(1 for p in pnls if p > 0)
    return dict(n=len(pnls), pnl=sum(pnls), wr=wins / len(pnls))


def main():
    from regime_classifier import RegimeClassifier, WINDOW_BARS as _REG_WINDOW

    def mkt_regime_for(master_df, symbol, end_ts):
        start = end_ts - pd.Timedelta(minutes=_REG_WINDOW * 2)
        sub = master_df.loc[(master_df.index >= start) & (master_df.index <= end_ts) &
                            (master_df["symbol"] == symbol), "close"]
        if len(sub) < _REG_WINDOW:
            return ""
        clf = RegimeClassifier()
        last = "warmup"
        for ts, c in sub.iloc[-_REG_WINDOW * 2:].items():
            try:
                r = clf.update(ts, float(c))
                if r: last = r
            except Exception: continue
        return "" if last == "warmup" else str(last)

    print("[load] parquet")
    master = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    master = master[(master.index >= "2025-09-15") & (master.index <= "2025-11-05")]
    print(f"  {len(master):,} rows (Oct 2025 + margin)")

    # Load all three sources
    de3 = load_trades(DE3_RA_SOURCE, "DynamicEngine3")
    ra  = load_trades(DE3_RA_SOURCE, "RegimeAdaptive")
    af  = load_trades(AF_SOURCE, "AetherFlow")
    # Ensure October 2025 scope
    de3 = [t for t in de3 if t.get("entry_time","").startswith("2025-10")]
    ra  = [t for t in ra  if t.get("entry_time","").startswith("2025-10")]
    af  = [t for t in af  if t.get("entry_time","").startswith("2025-10")]
    print(f"  DE3: {len(de3)} trades  RA: {len(ra)} trades  AF: {len(af)} trades")

    # Load G models
    models = {
        "DynamicEngine3": joblib.load(ROOT / "artifacts/signal_gate_2025/model_de3.joblib"),
        "RegimeAdaptive": joblib.load(ROOT / "artifacts/signal_gate_2025/model_regimeadaptive.joblib"),
        "AetherFlow":     joblib.load(ROOT / "artifacts/signal_gate_2025/model_aetherflow.joblib"),
    }
    for s, m in models.items():
        print(f"  G[{s}]: thr={m['veto_threshold']}, features={len(m['feature_names'])}, "
              f"rows={m['training_rows']}, cv_auc={m['cv_auc_mean']:.3f}")

    all_trades = {"DynamicEngine3": de3, "RegimeAdaptive": ra, "AetherFlow": af}
    kalshi_cache = {}

    # Enrich every trade
    print("\n[enrich] computing features + Kalshi + G for every trade...")
    enriched_all = {s: [] for s in all_trades}
    for strat, trades in all_trades.items():
        if not trades: continue
        g = models[strat]
        skipped = 0
        for t in trades:
            side = str(t.get("side", "")).upper()
            feat_row, mkt_regime = compute_features(t, master, mkt_regime_for)
            if feat_row is None:
                skipped += 1
                continue
            try:
                et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
            except:
                skipped += 1
                continue
            k_dec, k_prob = kalshi_decision(et, float(t["entry_price"]), side, kalshi_cache)
            g_p = score_g(t, feat_row, mkt_regime, g, side)
            et2 = dict(t)
            et2["kalshi_decision"] = k_dec
            et2["kalshi_aligned_prob"] = k_prob
            et2["g_p_big_loss"] = g_p
            # v5: apply regime-adaptive threshold multiplier
            base_thr = g["veto_threshold"]
            mult = _REGIME_THR_MULT.get((mkt_regime or "").lower(), 1.0)
            eff_thr = base_thr * mult
            et2["g_effective_thr"] = eff_thr
            et2["g_veto"] = g_p >= eff_thr
            et2["mkt_regime"] = mkt_regime
            enriched_all[strat].append(et2)
        print(f"  {strat}: {len(enriched_all[strat])} enriched, {skipped} skipped")

    # Per-week, per-strategy scenarios
    print(f"\n\n{'='*96}\nPER-WEEK × PER-STRATEGY BREAKDOWN (October 2025)\n{'='*96}")
    week_results = {}
    for wk_name, start, end in WEEKS:
        week_results[wk_name] = {}
        print(f"\n  {wk_name}")
        for strat in all_trades:
            picked = [t for t in enriched_all[strat] if in_week(t, start, end)]
            if not picked:
                print(f"    {strat:<18} 0 trades")
                continue
            base = evaluate(picked)
            kal  = evaluate([t for t in picked if t["kalshi_decision"] != "block"])
            g_on = evaluate([t for t in picked if not t["g_veto"]])
            full = evaluate([t for t in picked if t["kalshi_decision"] != "block" and not t["g_veto"]])
            week_results[wk_name][strat] = {"base": base, "kal": kal, "g": g_on, "full": full}
            delta = full["pnl"] - base["pnl"]
            sign = "+" if delta >= 0 else ""
            print(f"    {strat:<18} baseline n={base['n']:>3} ${base['pnl']:>+9.2f} wr={base['wr']*100:>4.1f}%  "
                  f"→ full-stack n={full['n']:>3} ${full['pnl']:>+9.2f} wr={full['wr']*100:>4.1f}%  "
                  f"delta={sign}${delta:+.2f}")

    # Monthly rollup
    print(f"\n\n{'='*96}\nOCTOBER 2025 MONTHLY ROLLUP\n{'='*96}")
    print(f"\n  {'strategy':<18} {'scenario':<32} {'n':>4} {'pnl':>11} {'wr':>6}")
    print("  " + "-" * 75)
    monthly = {}
    for strat in all_trades:
        picked = enriched_all[strat]
        base = evaluate(picked)
        kal  = evaluate([t for t in picked if t["kalshi_decision"] != "block"])
        g_on = evaluate([t for t in picked if not t["g_veto"]])
        full = evaluate([t for t in picked if t["kalshi_decision"] != "block" and not t["g_veto"]])
        monthly[strat] = {"base": base, "kal": kal, "g": g_on, "full": full}
        for label, ev in [("baseline", base), ("Kalshi only", kal),
                          (f"G only (thr {models[strat]['veto_threshold']:.3f})", g_on),
                          ("Kalshi + G (FULL LIVE STACK)", full)]:
            print(f"  {strat:<18} {label:<32} {ev['n']:>4} ${ev['pnl']:>+10.2f} {ev['wr']*100:>5.1f}%")
        print()

    # Combined portfolio
    print(f"  {'COMBINED (all 3)':<18}")
    for label, key in [("baseline","base"), ("Kalshi only","kal"),
                       ("G only (per-strategy thr)","g"),
                       ("Kalshi + G (FULL LIVE STACK)","full")]:
        n = sum(monthly[s][key]["n"] for s in all_trades)
        pnl = sum(monthly[s][key]["pnl"] for s in all_trades)
        # Aggregate winrate (trade-weighted)
        wins = sum(int(round(monthly[s][key]["wr"] * monthly[s][key]["n"])) for s in all_trades)
        wr = wins / max(1, n)
        print(f"  {'':<18} {label:<32} {n:>4} ${pnl:>+10.2f} {wr*100:>5.1f}%")

    # Filter effect summary
    print(f"\n\n{'='*96}\nFILTER EFFECT — delta vs baseline (October 2025)\n{'='*96}\n")
    print(f"  {'strategy':<18} {'baseline':>11} {'+Kalshi':>11} {'+G':>11} {'+Both':>11}  "
          f"{'delta':>10}")
    print("  " + "-" * 85)
    total_base = total_full = 0
    for strat in all_trades:
        m = monthly[strat]
        db = m["base"]["pnl"]; dk = m["kal"]["pnl"]; dg = m["g"]["pnl"]; dfull = m["full"]["pnl"]
        total_base += db
        total_full += dfull
        print(f"  {strat:<18} ${db:>+10.2f} ${dk:>+10.2f} ${dg:>+10.2f} ${dfull:>+10.2f}  "
              f"${dfull-db:>+9.2f}")
    print("  " + "-" * 85)
    print(f"  {'COMBINED':<18} ${total_base:>+10.2f} {'':>11} {'':>11} ${total_full:>+10.2f}  "
          f"${total_full-total_base:>+9.2f}")

    out = Path("/tmp/test_all_gates_oct2025.json")
    out.write_text(json.dumps({
        "weeks": week_results,
        "monthly": monthly,
    }, default=str, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()

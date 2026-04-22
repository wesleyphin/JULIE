"""Parameterized monthly v5-gate test. Takes --month YYYY-MM.

Same pipeline as test_all_gates_mar2025.py / oct2025, but generic:
  - DE3+RA trades: backtest_reports/full_live_replay/{YYYY_MM}/closed_trades.json
  - AF trades:     backtest_reports/af_fast_replay/{YYYY_MM}/closed_trades.json
  - Kalshi:        data/kalshi/kxinxu_2025_daily/{YYYY-MM-DD}.parquet

Usage:
  python test_month.py --month 2025-04
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
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
_REGIME_THR_MULT = {"whipsaw": 0.60, "calm_trend": 1.05, "neutral": 1.0,
                    "warmup": 1.0, "": 1.0}
# v5.2 session-adaptive multiplier thresholds
_SESS_LENIENT_PNL    = 100.0
_SESS_AGGRESSIVE_PNL = -200.0
_SESS_LENIENT_MULT   = 1.25
_SESS_AGGRESSIVE_MULT = 0.80


def _session_mult(cum_pnl):
    if cum_pnl >= _SESS_LENIENT_PNL:    return _SESS_LENIENT_MULT
    if cum_pnl <= _SESS_AGGRESSIVE_PNL: return _SESS_AGGRESSIVE_MULT
    return 1.0


def load_trades(path, strategy=None, month_prefix=""):
    if not path.exists(): return []
    out = []
    for t in json.load(open(path)):
        if strategy and str(t.get("strategy")) != strategy:
            continue
        if month_prefix and not t.get("entry_time","").startswith(month_prefix):
            continue
        out.append(t)
    return out


def kalshi_decision(et_dt, entry_price, side, cache):
    h = et_dt.hour
    if h not in KALSHI_GATING_HOURS_ET:
        return ("auto_pass", None)
    d = et_dt.date().isoformat()
    if d not in cache:
        p = KALSHI_DAILY / f"{d}.parquet"
        cache[d] = pd.read_parquet(p) if p.exists() else None
    kdf = cache[d]
    if kdf is None or kdf.empty: return ("auto_pass", None)
    nxt = None
    for hh in (10, 11, 12, 13, 14, 15, 16):
        if hh > h: nxt = hh; break
    if nxt is None: return ("auto_pass", None)
    sub = kdf[(kdf["settlement_hour_et"] == nxt) & (kdf["event_date"] == d)]
    if sub.empty: return ("auto_pass", None)
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
    if win.empty: return None, None
    sym = win["symbol"].value_counts().idxmax()
    sub = win[win["symbol"] == sym][["open", "high", "low", "close", "volume"]]
    if len(sub) < 50: return None, None
    feats = _compute_feature_frame(sub)
    if feats.empty: return None, None
    idx = feats.index.searchsorted(end)
    if idx <= 0 or idx > len(feats): return None, None
    feat_row = feats.iloc[idx - 1]
    if feat_row.isna().all(): return None, None
    return feat_row, regime_fn(master_df, sym, end)


def compute_session_state(trades):
    by_day = defaultdict(list)
    for t in trades:
        try: et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
        except: continue
        by_day[et.date().isoformat()].append(t)
    state = {}
    for day, day_trades in by_day.items():
        day_trades.sort(key=lambda x: str(x.get("entry_time", "")))
        cum = 0.0; consec = 0; count = 0
        for t in day_trades:
            k = (str(t.get("entry_time")), t.get("side"), float(t.get("entry_price") or 0))
            state[k] = (cum, consec, count)
            pnl = float(t.get("pnl_dollars", 0) or 0)
            cum += pnl
            if pnl < 0: consec += 1
            elif pnl > 0: consec = 0
            count += 1
    return state


def score_g(trade, feat_row, mkt_regime, g_model, side, session_state):
    feat_names = g_model["feature_names"]
    numeric = g_model["numeric_features"]
    cat_maps = g_model.get("categorical_maps", {})
    row = {c: 0.0 for c in feat_names}
    k = (str(trade.get("entry_time")), trade.get("side"), float(trade.get("entry_price") or 0))
    cum_pnl, consec, count = session_state.get(k, (0.0, 0, 0))
    for c in numeric:
        if c == "trend_align_ret1":
            r = feat_row.get("de3_entry_ret1_atr", 0.0)
            try: rf = float(r); rf = 0.0 if not np.isfinite(rf) else rf
            except: rf = 0.0
            if rf > 0: row[c] = 1.0 if side == "LONG" else -1.0
            elif rf < 0: row[c] = 1.0 if side == "SHORT" else -1.0
            else: row[c] = 0.0
            continue
        if c == "cum_day_pnl_pre_trade": row[c] = float(cum_pnl); continue
        if c == "consec_losses_pre_trade": row[c] = float(consec); continue
        if c == "trades_today_pre_trade":  row[c] = float(count); continue
        v = feat_row.get(c, 0.0)
        try: fv = float(v); fv = 0.0 if not np.isfinite(fv) else fv
        except: fv = 0.0
        if c in row: row[c] = fv
    et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
    h = et.hour
    sess = ("ASIA" if (18 <= h or h < 3) else "LONDON" if h < 7 else
            "NY_PRE" if h < 9 else "NY" if h < 16 else "POST")
    af_reg = str(trade.get("regime") or trade.get("aetherflow_regime") or "").strip().upper()
    if af_reg not in {"DISPERSED", "TREND_GEODESIC", "CHOP_SPIRAL", "ROTATIONAL_TURBULENCE"}:
        af_reg = ""
    cat_vals = {"side": side.upper(), "session": sess, "regime": af_reg,
                "mkt_regime": mkt_regime or ""}
    for cc, kvs in cat_maps.items():
        val = cat_vals.get(cc, "")
        for kv in kvs:
            nm = f"{cc}__{kv}"
            if nm in row and val == kv: row[nm] = 1
    if "et_hour" in row: row["et_hour"] = float(h)
    X = np.array([[row[c] for c in feat_names]])
    return float(g_model["model"].predict_proba(X)[0, 1])


def evaluate(picked):
    if not picked: return dict(n=0, pnl=0.0, wr=0.0)
    pnls = [float(t["pnl_dollars"]) for t in picked]
    wins = sum(1 for p in pnls if p > 0)
    return dict(n=len(pnls), pnl=sum(pnls), wr=wins / len(pnls))


def run_month(month: str):
    """month = 'YYYY-MM'"""
    label = month.replace("-", "_")  # 2025_04
    DE3_RA_SRC = ROOT / "backtest_reports" / "full_live_replay" / label / "closed_trades.json"
    # AF paths historically use month-name labels (mar2025, oct2025) as well
    # as the numeric YYYY_MM format. Try numeric first, fall back to legacy.
    yyyy, mm = month.split("-")
    MONTH_NAMES = {"01":"jan","02":"feb","03":"mar","04":"apr","05":"may","06":"jun",
                   "07":"jul","08":"aug","09":"sep","10":"oct","11":"nov","12":"dec"}
    af_candidates = [
        ROOT / "backtest_reports" / "af_fast_replay" / label,
        ROOT / "backtest_reports" / "af_fast_replay" / f"{MONTH_NAMES[mm]}{yyyy}",
    ]
    AF_SRC = None
    for c in af_candidates:
        if (c / "closed_trades.json").exists():
            AF_SRC = c / "closed_trades.json"; break
    if AF_SRC is None:
        AF_SRC = af_candidates[0] / "closed_trades.json"  # will yield 0 trades cleanly

    from regime_classifier import RegimeClassifier, WINDOW_BARS as _REG_WINDOW

    def mkt_regime_for(master_df, symbol, end_ts):
        start = end_ts - pd.Timedelta(minutes=_REG_WINDOW * 2)
        sub = master_df.loc[(master_df.index >= start) & (master_df.index <= end_ts) &
                            (master_df["symbol"] == symbol), "close"]
        if len(sub) < _REG_WINDOW: return ""
        clf = RegimeClassifier()
        last = "warmup"
        for ts, c in sub.iloc[-_REG_WINDOW * 2:].items():
            try:
                r = clf.update(ts, float(c))
                if r: last = r
            except: continue
        return "" if last == "warmup" else str(last)

    # Parquet — month ± 1 month for regime warmup
    yyyy, mm = month.split("-")
    month_start = pd.Timestamp(f"{yyyy}-{mm}-01", tz="US/Eastern")
    master = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    master = master[(master.index >= month_start - pd.Timedelta(days=15)) &
                    (master.index <= month_start + pd.Timedelta(days=40))]
    de3 = load_trades(DE3_RA_SRC, "DynamicEngine3", month)
    ra  = load_trades(DE3_RA_SRC, "RegimeAdaptive", month)
    af  = load_trades(AF_SRC, "AetherFlow", month)

    _family_map = {"dynamicengine3": "de3", "aetherflow": "aetherflow",
                   "regimeadaptive": "regimeadaptive"}
    models = {}
    for strat in ["DynamicEngine3", "RegimeAdaptive", "AetherFlow"]:
        fam = _family_map[strat.lower()]
        models[strat] = joblib.load(ROOT / f"artifacts/signal_gate_2025/model_{fam}.joblib")

    all_trades = {"DynamicEngine3": de3, "RegimeAdaptive": ra, "AetherFlow": af}
    session_state = {s: compute_session_state(all_trades[s]) for s in all_trades}
    kalshi_cache = {}
    enriched_all = {s: [] for s in all_trades}
    for strat, trades in all_trades.items():
        if not trades: continue
        g = models[strat]
        base_thr = g["veto_threshold"]
        for t in trades:
            side = str(t.get("side", "")).upper()
            feat_row, mkt_regime = compute_features(t, master, mkt_regime_for)
            if feat_row is None: continue
            try: et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
            except: continue
            # v5.3: Kalshi disabled for DE3 — post-hoc analysis shows Kalshi
            # was net -$238 across 5 months on DE3, its settlement-hour logic
            # doesn't match DE3's 10/25pt bracket horizon.
            if strat == "DynamicEngine3":
                k_dec, k_prob = ("auto_pass", None)
            else:
                k_dec, k_prob = kalshi_decision(et, float(t["entry_price"]), side, kalshi_cache)
            g_p = score_g(t, feat_row, mkt_regime, g, side, session_state[strat])
            # v5.2: regime mult × session mult
            regime_mult = _REGIME_THR_MULT.get((mkt_regime or "").lower(), 1.0)
            k_key = (str(t.get("entry_time")), t.get("side"), float(t.get("entry_price") or 0))
            cum_pnl_pre, _, _ = session_state[strat].get(k_key, (0.0, 0, 0))
            sess_mult = _session_mult(cum_pnl_pre)
            eff_thr = base_thr * regime_mult * sess_mult
            et2 = dict(t)
            et2["kalshi_decision"] = k_dec
            et2["g_p_big_loss"] = g_p
            et2["g_effective_thr"] = eff_thr
            et2["g_session_mult"] = sess_mult
            et2["g_regime_mult"] = regime_mult
            et2["cum_day_pnl_pre"] = cum_pnl_pre
            et2["g_veto"] = g_p >= eff_thr
            et2["mkt_regime"] = mkt_regime
            enriched_all[strat].append(et2)

    # Monthly
    print(f"\n{'='*88}\n{month}  (counts: DE3={len(de3)}, RA={len(ra)}, AF={len(af)})\n{'='*88}")
    print(f"  {'strategy':<18} {'scenario':<30} {'n':>4} {'pnl':>11} {'wr':>6}")
    print("  " + "-" * 73)
    monthly = {}
    for strat in all_trades:
        picked = enriched_all[strat]
        base = evaluate(picked)
        kal  = evaluate([t for t in picked if t["kalshi_decision"] != "block"])
        g_on = evaluate([t for t in picked if not t["g_veto"]])
        full = evaluate([t for t in picked if t["kalshi_decision"] != "block" and not t["g_veto"]])
        monthly[strat] = {"base": base, "kal": kal, "g": g_on, "full": full}
        for label2, ev in [("baseline", base), ("Kalshi only", kal),
                          (f"G only (thr {models[strat]['veto_threshold']:.3f})", g_on),
                          ("Kalshi + G (FULL)", full)]:
            print(f"  {strat:<18} {label2:<30} {ev['n']:>4} ${ev['pnl']:>+10.2f} {ev['wr']*100:>5.1f}%")
        print()

    # Combined
    print(f"  {'COMBINED':<18}")
    for label2, key in [("baseline","base"), ("Kalshi only","kal"),
                       ("G only","g"), ("Kalshi + G (FULL)","full")]:
        n = sum(monthly[s][key]["n"] for s in all_trades)
        pnl = sum(monthly[s][key]["pnl"] for s in all_trades)
        wins = sum(int(round(monthly[s][key]["wr"] * monthly[s][key]["n"])) for s in all_trades)
        wr = wins / max(1, n)
        print(f"  {'':<18} {label2:<30} {n:>4} ${pnl:>+10.2f} {wr*100:>5.1f}%")
    print()

    # Delta table
    print(f"  {'strategy':<18} {'baseline':>11} {'full':>11}  {'delta':>10}")
    print("  " + "-" * 57)
    tot_b = tot_f = 0
    for strat in all_trades:
        m = monthly[strat]
        db = m["base"]["pnl"]; dfull = m["full"]["pnl"]
        tot_b += db; tot_f += dfull
        print(f"  {strat:<18} ${db:>+10.2f} ${dfull:>+10.2f}  ${dfull-db:>+9.2f}")
    print("  " + "-" * 57)
    print(f"  {'COMBINED':<18} ${tot_b:>+10.2f} ${tot_f:>+10.2f}  ${tot_f-tot_b:>+9.2f}")
    return monthly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/test_months.json")
    args = ap.parse_args()

    results = {}
    for m in args.months:
        results[m] = run_month(m)

    # Summary
    print(f"\n\n{'='*88}\nCROSS-MONTH v5 SUMMARY\n{'='*88}")
    print(f"  {'month':<12} {'baseline':>12} {'full-stack':>12}  {'delta':>10}")
    print("  " + "-" * 52)
    grand_b = grand_f = 0
    for m, monthly in results.items():
        tot_b = sum(monthly[s]["base"]["pnl"] for s in monthly)
        tot_f = sum(monthly[s]["full"]["pnl"] for s in monthly)
        grand_b += tot_b; grand_f += tot_f
        print(f"  {m:<12} ${tot_b:>+11.2f} ${tot_f:>+11.2f}  ${tot_f-tot_b:>+9.2f}")
    print("  " + "-" * 52)
    print(f"  {'TOTAL':<12} ${grand_b:>+11.2f} ${grand_f:>+11.2f}  ${grand_f-grand_b:>+9.2f}")

    Path(args.out).write_text(json.dumps({
        m: {s: {k: v for k, v in sv.items()} for s, sv in monthly.items()}
        for m, monthly in results.items()
    }, default=str, indent=2))
    print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()

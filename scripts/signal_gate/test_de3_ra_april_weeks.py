"""Score DE3 and RegimeAdaptive trades from replay_apr2026_p1 against
the updated G gates + Kalshi over the two target weeks:

    apr6_10:  2026-04-06 → 2026-04-10
    apr13_17: 2026-04-13 → 2026-04-17

Trades come from the already-completed live-replay run:
    backtest_reports/replay_apr2026_p1/live_loop_MES_<ts>/closed_trades.json

Every trade gets:
  - Kalshi decision (real daily snapshot from API, aligned_prob threshold 0.45)
  - G gate score (new model with regime + mkt_regime + trend_align features)
  - Scenario aggregates (baseline / Kalshi only / G only / Kalshi+G)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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

REPLAY_DIR = ROOT / "backtest_reports" / "replay_apr2026_p1" / "live_loop_MES_20260421_061829"

WEEKS = [
    ("apr6_10",  "2026-04-06", "2026-04-10"),
    ("apr13_17", "2026-04-13", "2026-04-17"),
]

STRATEGIES = ["DynamicEngine3", "RegimeAdaptive"]


def load_trades() -> list:
    trades = json.loads((REPLAY_DIR / "closed_trades.json").read_text())
    print(f"  loaded {len(trades)} trades from {REPLAY_DIR.name}")
    return trades


def filter_trades(trades, strategy, start, end):
    sd = pd.Timestamp(start, tz=NY)
    ed = pd.Timestamp(end, tz=NY) + pd.Timedelta(hours=23, minutes=59)
    out = []
    for t in trades:
        if t.get("strategy") != strategy:
            continue
        try:
            et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        if not (sd <= pd.Timestamp(et) <= ed):
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


def compute_trade_features(trade, master_df, symbol_for_ts, regime_clf_fn):
    """Recompute entry-shape + mkt_regime for a single trade."""
    from build_de3_chosen_shape_dataset import _compute_feature_frame

    et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
    end = pd.Timestamp(et).tz_convert("UTC")
    start = end - pd.Timedelta(hours=6)
    # Pick the symbol that's most common in the recent window
    window = master_df.loc[(master_df.index >= start) & (master_df.index <= end + pd.Timedelta(minutes=30))]
    if window.empty:
        return None, None, None
    sym = window["symbol"].value_counts().idxmax()
    sub = window[window["symbol"] == sym][["open", "high", "low", "close", "volume"]]
    if len(sub) < 50:
        return None, None, sym
    feats = _compute_feature_frame(sub)
    if feats.empty:
        return None, None, sym
    idx = feats.index.searchsorted(end)
    if idx <= 0 or idx > len(feats):
        return None, None, sym
    feat_row = feats.iloc[idx - 1]
    if feat_row.isna().all():
        return None, None, sym
    mkt_regime = regime_clf_fn(master_df, sym, end)
    return feat_row, mkt_regime, sym


def score_g(trade, feat_row, mkt_regime, g_model, side):
    feat_names = g_model["feature_names"]
    numeric = g_model["numeric_features"]
    cat_maps = g_model.get("categorical_maps", {})
    row = {c: 0.0 for c in feat_names}
    for c in numeric:
        if c == "trend_align_ret1":
            r = feat_row.get("de3_entry_ret1_atr", 0.0)
            try:
                rf = float(r)
                if not np.isfinite(rf): rf = 0.0
            except: rf = 0.0
            if rf > 0:
                row[c] = 1.0 if side == "LONG" else -1.0
            elif rf < 0:
                row[c] = 1.0 if side == "SHORT" else -1.0
            else:
                row[c] = 0.0
            continue
        v = feat_row.get(c, 0.0)
        try:
            fv = float(v)
            if not np.isfinite(fv): fv = 0.0
        except: fv = 0.0
        if c in row: row[c] = fv

    et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
    h = et.hour
    sess = ("ASIA" if (18 <= h or h < 3) else "LONDON" if h < 7 else
            "NY_PRE" if h < 9 else "NY" if h < 16 else "POST")
    cat_vals = {
        "side": side.upper(),
        "session": sess,
        "regime": "",  # AF manifold regime (empty for DE3/RA)
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
    p = float(g_model["model"].predict_proba(X)[0, 1])
    return p


def evaluate_set(trades):
    if not trades:
        return dict(n=0, pnl=0.0, wr=0.0)
    pnls = [float(t["pnl_dollars"]) for t in trades]
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
                if r:
                    last = r
            except Exception:
                continue
        return "" if last == "warmup" else str(last)

    print("[load] parquet")
    master = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    master = master[master.index >= "2026-01-01"]
    print(f"  {len(master):,} rows")

    print("[load] replay trades")
    trades_all = load_trades()

    # Load G models
    models = {}
    for strat in STRATEGIES:
        p = ROOT / "artifacts" / "signal_gate_2025" / f"model_{strat.lower()}.joblib"
        m = joblib.load(p)
        models[strat] = m
        print(f"  G model {strat}: thr={m['veto_threshold']}, "
              f"features={len(m['feature_names'])}, rows={m['training_rows']}, "
              f"cv_auc={m['cv_auc_mean']:.3f}")

    results = {}
    for strat in STRATEGIES:
        print(f"\n{'='*78}\nSTRATEGY: {strat}\n{'='*78}")
        g = models[strat]
        strat_results = {}
        for run_name, start, end in WEEKS:
            print(f"\n  WEEK {run_name} ({start} → {end}):")
            week_trades = filter_trades(trades_all, strat, start, end)
            print(f"    raw trades: {len(week_trades)}")
            if not week_trades:
                strat_results[run_name] = dict(trades=[], scenarios=[])
                continue

            kalshi_cache = {}
            enriched = []
            for t in week_trades:
                et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
                side = str(t.get("side", "")).upper()
                feat_row, mkt_regime, _sym = compute_trade_features(t, master, None, mkt_regime_for)
                if feat_row is None:
                    continue
                k_dec, k_prob = kalshi_decision(et, float(t["entry_price"]), side, kalshi_cache)
                g_p = score_g(t, feat_row, mkt_regime, g, side)
                et2 = dict(t)
                et2["kalshi_decision"] = k_dec
                et2["kalshi_aligned_prob"] = k_prob
                et2["g_p_big_loss"] = g_p
                et2["g_veto"] = g_p >= g["veto_threshold"]
                et2["mkt_regime"] = mkt_regime
                enriched.append(et2)

            # Scenarios
            scenarios = [
                ("baseline", lambda t: True),
                ("Kalshi only", lambda t: t["kalshi_decision"] != "block"),
                (f"G only (thr {g['veto_threshold']:.3f})", lambda t: not t["g_veto"]),
                ("Kalshi + G (FULL LIVE STACK)", lambda t: t["kalshi_decision"] != "block" and not t["g_veto"]),
            ]
            print(f"    {'scenario':<36} {'n':>4} {'pnl':>10} {'wr':>6}")
            week_output = []
            for label, selector in scenarios:
                picked = [t for t in enriched if selector(t)]
                ev = evaluate_set(picked)
                print(f"    {label:<36} {ev['n']:>4} ${ev['pnl']:>+9.2f} {ev['wr']*100:>5.1f}%")
                week_output.append(dict(label=label, **ev))

            # G precision
            vw = sum(t["pnl_dollars"] for t in enriched if t["g_veto"] and t["pnl_dollars"] > 0)
            vl = sum(t["pnl_dollars"] for t in enriched if t["g_veto"] and t["pnl_dollars"] <= 0)
            n_vw = sum(1 for t in enriched if t["g_veto"] and t["pnl_dollars"] > 0)
            n_vl = sum(1 for t in enriched if t["g_veto"] and t["pnl_dollars"] <= 0)
            g_net = -(vw + vl)
            print(f"    G precision: vetoed {n_vw}W +${vw:.2f} / {n_vl}L ${vl:.2f} | net ${g_net:+.2f}")

            # Kalshi counts
            kal_b = sum(1 for t in enriched if t["kalshi_decision"] == "block")
            kal_p = sum(1 for t in enriched if t["kalshi_decision"] == "pass")
            kal_auto = sum(1 for t in enriched if t["kalshi_decision"] == "auto_pass")
            print(f"    Kalshi:    blocks={kal_b} passes={kal_p} auto_pass={kal_auto}")

            # mkt regime mix
            mkt_mix = Counter(t.get("mkt_regime") or "none" for t in enriched)
            print(f"    mkt_regime mix: {dict(mkt_mix)}")

            strat_results[run_name] = dict(trades=enriched, scenarios=week_output,
                                            g_net=g_net, kalshi=dict(b=kal_b, p=kal_p, a=kal_auto))
        results[strat] = strat_results

    # Cross summary
    print(f"\n\n{'='*88}\nCROSS-STRATEGY CROSS-WEEK SUMMARY\n{'='*88}")
    hdr = f"{'strat':<16} {'week':<10} {'scenario':<32} {'n':>4} {'pnl':>10} {'wr':>6}"
    print(hdr)
    print("-" * len(hdr))
    for strat, wk_res in results.items():
        for wk_name, wk_data in wk_res.items():
            for s in wk_data.get("scenarios", []):
                print(f"{strat:<16} {wk_name:<10} {s['label']:<32} {s['n']:>4} ${s['pnl']:>+9.2f} {s['wr']*100:>5.1f}%")
            print()

    out = Path("/tmp/test_de3_ra_april_weeks.json")
    out.write_text(json.dumps({s: {k: {
        "scenarios": v["scenarios"],
        "g_net": v.get("g_net", 0),
        "kalshi": v.get("kalshi", {}),
    } for k, v in r.items()} for s, r in results.items()}, default=str, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()

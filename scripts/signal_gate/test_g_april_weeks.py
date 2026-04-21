"""Test the v3 G gate on two April 2026 weeks: 6-10 and 13-17.

Pipeline per week:
  1. Fast AF replay on parquet bars to produce candidate trades
  2. Kalshi post-hoc block (real API data pulled for this range)
  3. G gate scoring with model_aetherflow.joblib

Reports:
  - Per-week: baseline PnL, PnL under each filter stack (Kalshi only,
    G only, Kalshi+G, AF-loose+Kalshi+G), veto rate, winrate, DD
  - G's precision: winners-vetoed vs. losers-blocked
  - Per-signal detail on demand (the interesting ones — Kalshi blocks
    and G vetoes of winners/passes of losers)
"""
from __future__ import annotations

import json
import subprocess
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

PY = "/opt/homebrew/Caskroom/miniconda/base/envs/julie_bot_11/bin/python"
NY = ZoneInfo("America/New_York")
KALSHI_DAILY = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
KALSHI_GATING_HOURS_ET = {12, 13, 14, 15, 16}
KALSHI_ENTRY_THRESHOLD = 0.45
POINT_VALUE = 5.0
SIZE = 5  # AF default

WEEKS = [
    ("apr6_10",  "2026-04-06", "2026-04-10"),
    ("apr13_17", "2026-04-13", "2026-04-17"),
]


def run_fast_replay(start: str, end: str, run_name: str):
    """Invoke fast_aetherflow_replay.py for this window."""
    out_dir = ROOT / "backtest_reports" / "af_fast_replay" / run_name
    if (out_dir / "closed_trades.json").exists():
        print(f"  [skip fast-replay] {run_name} already exists")
        return out_dir
    cmd = [
        PY, str(ROOT / "scripts/signal_gate/fast_aetherflow_replay.py"),
        "--start", start, "--end", end,
        "--run-name", run_name,
    ]
    print(f"  [fast-replay] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_dir


def kalshi_decision_for(ts_et: datetime, entry_price: float, side: str,
                        kalshi_cache: dict):
    et_hour = ts_et.hour
    if et_hour not in KALSHI_GATING_HOURS_ET:
        return ("auto_pass", None)
    date_str = ts_et.date().isoformat()
    if date_str not in kalshi_cache:
        p = KALSHI_DAILY / f"{date_str}.parquet"
        kalshi_cache[date_str] = pd.read_parquet(p) if p.exists() else None
    kdf = kalshi_cache[date_str]
    if kdf is None or kdf.empty:
        return ("auto_pass", None)
    # next settlement hour
    next_set = None
    for h in (10, 11, 12, 13, 14, 15, 16):
        if h > et_hour:
            next_set = h
            break
    if next_set is None:
        return ("auto_pass", None)
    sub = kdf[(kdf["settlement_hour_et"] == next_set) & (kdf["event_date"] == date_str)]
    if sub.empty:
        return ("auto_pass", None)
    sub = sub.copy()
    sub["dist"] = (sub["strike"] - entry_price).abs()
    row = sub.nsmallest(1, "dist").iloc[0]
    # high/low already cents midpoints from the fetcher
    yes_prob = (float(row["high"]) + float(row["low"])) / 200.0
    aligned = yes_prob if side == "LONG" else 1.0 - yes_prob
    if aligned < KALSHI_ENTRY_THRESHOLD:
        return ("block", aligned)
    return ("pass", aligned)


def score_g_for_trade(trade: dict, g_model: dict, master_df: pd.DataFrame):
    """Compute features for a trade from parquet and score G.
    Uses the same feature pipeline as the trainer."""
    from build_de3_chosen_shape_dataset import _compute_feature_frame

    et = datetime.fromisoformat(trade["entry_time"]).astimezone(NY)
    # Pull 6h of bars before entry for feature window
    start = pd.Timestamp(et) - pd.Timedelta(hours=6)
    end = pd.Timestamp(et)
    sub = master_df.loc[
        (master_df.index >= start) & (master_df.index <= end + pd.Timedelta(minutes=30)),
        ["open", "high", "low", "close", "volume", "symbol"]
    ]
    if sub.empty:
        return None, None
    # Pick front-month symbol (the most common in this slice)
    sym = sub["symbol"].value_counts().idxmax()
    sub = sub[sub["symbol"] == sym][["open", "high", "low", "close", "volume"]]
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

    feat_names = g_model["feature_names"]
    numeric = g_model["numeric_features"]
    cat_maps = g_model.get("categorical_maps", {})
    side = str(trade["side"]).upper()
    row = {c: 0.0 for c in feat_names}

    for c in numeric:
        if c == "trend_align_ret1":
            ret1 = feat_row.get("de3_entry_ret1_atr", 0.0)
            try:
                r = float(ret1)
                if not np.isfinite(r): r = 0.0
            except: r = 0.0
            if r > 0:
                row[c] = 1.0 if side == "LONG" else -1.0
            elif r < 0:
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
    # session / regime
    h = et.hour
    sess = ("ASIA" if (18 <= h or h < 3) else "LONDON" if h < 7 else
            "NY_PRE" if h < 9 else "NY" if h < 16 else "POST")
    regime = str(trade.get("regime", "") or "").strip().upper()
    cat_vals = {"side": side, "session": sess, "regime": regime}
    for cc, kvs in cat_maps.items():
        val = cat_vals.get(cc, "")
        for kv in kvs:
            nm = f"{cc}__{kv}"
            if nm in row and val == kv:
                row[nm] = 1
    if "et_hour" in row:
        row["et_hour"] = float(h)
    X = np.array([[row[c] for c in feat_names]])
    p_big_loss = float(g_model["model"].predict_proba(X)[0, 1])
    return p_big_loss, feat_row


def compute_dd(pnls):
    cum = peak = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def evaluate_set(trades):
    pnls = [float(t["pnl_dollars"]) for t in trades]
    if not pnls:
        return dict(n=0, pnl=0.0, wr=0.0, dd=0.0, stops=0, takes=0, horizon=0)
    wins = sum(1 for p in pnls if p > 0)
    return dict(
        n=len(pnls),
        pnl=sum(pnls),
        wr=wins / len(pnls),
        dd=compute_dd(pnls),
        stops=sum(1 for t in trades if "stop" in str(t.get("source", ""))),
        takes=sum(1 for t in trades if t.get("source") == "take"),
        horizon=sum(1 for t in trades if "horizon" in str(t.get("source", ""))),
    )


def run_window(run_name: str, start: str, end: str, g_model: dict,
               master_df: pd.DataFrame):
    print(f"\n{'='*88}\nWEEK: {run_name}  ({start} → {end})\n{'='*88}")
    out_dir = run_fast_replay(start, end, run_name)
    ct_path = out_dir / "closed_trades.json"
    trades = json.loads(ct_path.read_text())
    print(f"  raw AF trades: {len(trades)}")
    if not trades:
        return None

    # Decorate each trade with Kalshi decision + G score
    kalshi_cache = {}
    for t in trades:
        et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
        kal, kal_prob = kalshi_decision_for(et, float(t["entry_price"]), t["side"], kalshi_cache)
        t["kalshi_decision"] = kal
        t["kalshi_aligned_prob"] = kal_prob
        p_bl, _ = score_g_for_trade(t, g_model, master_df)
        t["g_p_big_loss"] = p_bl
        t["g_veto"] = (p_bl is not None and p_bl >= g_model["veto_threshold"])
        t["af_confidence"] = float(t.get("confidence", 0.0))

    # Scenario aggregates
    def agg(selector, label):
        picked = [t for t in trades if selector(t)]
        ev = evaluate_set(picked)
        return dict(label=label, **ev)

    scenarios = [
        agg(lambda t: True, "baseline (take every AF trade)"),
        agg(lambda t: t["kalshi_decision"] != "block", "Kalshi only"),
        agg(lambda t: not t["g_veto"], "G only (v3 thr 0.275)"),
        agg(lambda t: t["kalshi_decision"] != "block" and not t["g_veto"], "Kalshi + G"),
        # AF strict (live threshold 0.55+)
        agg(lambda t: float(t.get("confidence", 0.0)) >= 0.55,
            "AF strict (≥0.55)"),
        agg(lambda t: float(t.get("confidence", 0.0)) >= 0.55
                     and t["kalshi_decision"] != "block" and not t["g_veto"],
            "AF strict + Kalshi + G (FULL LIVE STACK)"),
        # AF loose
        agg(lambda t: float(t.get("confidence", 0.0)) >= 0.45,
            "AF loose (≥0.45)"),
        agg(lambda t: float(t.get("confidence", 0.0)) >= 0.45
                     and t["kalshi_decision"] != "block" and not t["g_veto"],
            "AF loose + Kalshi + G"),
    ]

    # G's precision breakdown
    vw = [t for t in trades if t["g_veto"] and t["pnl_dollars"] > 0]
    vl = [t for t in trades if t["g_veto"] and t["pnl_dollars"] <= 0]
    pw = [t for t in trades if not t["g_veto"] and t["pnl_dollars"] > 0]
    pl = [t for t in trades if not t["g_veto"] and t["pnl_dollars"] <= 0]

    # Print scenarios
    print(f"\n  {'scenario':<42}  {'n':>3}  {'pnl':>10}  {'wr':>6}  {'dd':>9}  s/t/h")
    print(f"  {'-'*42:<42}  {'---':>3}  {'-'*10:>10}  {'-'*6:>6}  {'-'*9:>9}  -----")
    for s in scenarios:
        print(f"  {s['label']:<42}  {s['n']:>3}  ${s['pnl']:>+9.2f}  {s['wr']*100:>5.1f}%  "
              f"${s['dd']:>+7.2f}  {s['stops']}/{s['takes']}/{s['horizon']}")

    # G precision
    print(f"\n  G gate precision:")
    print(f"    veto_winner  (missed wins)   n={len(vw):>3}  pnl=${sum(t['pnl_dollars'] for t in vw):>+9.2f}")
    print(f"    veto_loser   (correct blocks) n={len(vl):>3}  pnl=${sum(t['pnl_dollars'] for t in vl):>+9.2f}")
    print(f"    pass_winner  (correct passes) n={len(pw):>3}  pnl=${sum(t['pnl_dollars'] for t in pw):>+9.2f}")
    print(f"    pass_loser   (missed blocks)  n={len(pl):>3}  pnl=${sum(t['pnl_dollars'] for t in pl):>+9.2f}")
    g_net = -(sum(t['pnl_dollars'] for t in vw) + sum(t['pnl_dollars'] for t in vl))
    print(f"    G net effect: ${g_net:+.2f}  ({'HELPS' if g_net > 0 else 'HURTS'})")

    # Kalshi effect
    kal_blocks = [t for t in trades if t["kalshi_decision"] == "block"]
    kal_passes = [t for t in trades if t["kalshi_decision"] == "pass"]
    print(f"\n  Kalshi effect:")
    print(f"    active BLOCKS n={len(kal_blocks):>3}  pnl=${sum(t['pnl_dollars'] for t in kal_blocks):>+9.2f}  "
          f"(wr {sum(1 for t in kal_blocks if t['pnl_dollars']>0)/max(1,len(kal_blocks))*100:.0f}%)")
    print(f"    active PASSES n={len(kal_passes):>3}  pnl=${sum(t['pnl_dollars'] for t in kal_passes):>+9.2f}  "
          f"(wr {sum(1 for t in kal_passes if t['pnl_dollars']>0)/max(1,len(kal_passes))*100:.0f}%)")
    print(f"    auto_passed   n={len(trades) - len(kal_blocks) - len(kal_passes):>3}")

    # Persist
    enriched_path = out_dir / "closed_trades_with_gates.json"
    enriched_path.write_text(json.dumps(trades, default=str, indent=2))
    print(f"\n  [write] {enriched_path}")

    return dict(run_name=run_name, scenarios=scenarios,
                 g_precision=dict(vw=len(vw), vl=len(vl), pw=len(pw), pl=len(pl), net=g_net),
                 kalshi=dict(blocks=len(kal_blocks), passes=len(kal_passes)))


def main():
    print("[load] G model")
    g_model = joblib.load(ROOT / "artifacts" / "signal_gate_2025" / "model_aetherflow.joblib")
    print(f"  thr={g_model['veto_threshold']}, features={len(g_model['feature_names'])}, "
          f"training_rows={g_model['training_rows']}, cv_auc={g_model['cv_auc_mean']:.3f}")

    print("\n[load] parquet")
    master = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    master = master[master.index >= "2026-01-01"]
    print(f"  rows={len(master):,}")

    all_results = []
    for run_name, start, end in WEEKS:
        r = run_window(run_name, start, end, g_model, master)
        if r:
            all_results.append(r)

    # Cross-week summary
    print(f"\n\n{'='*88}\nCROSS-WEEK SUMMARY (v3 G gate + real Kalshi)\n{'='*88}")
    print(f"{'week':<12} {'scenario':<42} {'n':>3} {'pnl':>10} {'wr':>6}")
    print("-" * 88)
    for r in all_results:
        for s in r["scenarios"]:
            print(f"{r['run_name']:<12} {s['label']:<42} {s['n']:>3} ${s['pnl']:>+9.2f} {s['wr']*100:>5.1f}%")
        print()


if __name__ == "__main__":
    main()

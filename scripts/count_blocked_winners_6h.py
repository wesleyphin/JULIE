#!/usr/bin/env python3
"""6h blocked-trade counterfactual — every filter, every strategy.

Same plumbing as count_blocked_winners_12h_v2.py but a 6-hour window and
dual-scale PnL reporting (size=1 + live-size = 3 for DE3/RegAdapt, 5 for
AetherFlow solo).

v6 ML bracket note: DE3 blocked signals below vol_bp=1.5 get the v6-ML
scalp bracket (3/5). Otherwise DE3 defaults to 8.25/10 (matches the live
DE3 ATR-based median). AetherFlow family-specific fixed brackets used.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict

ROOT = Path("/Users/wes/Downloads/JULIE001")
LOG = ROOT / "topstep_live_bot.log"
MES_PT = 5.0
WINDOW_HOURS = 6.0

# Live size defaults (matches observed live trading on account 15686180)
LIVE_SIZE_DEFAULT = {
    "AetherFlow":      5,    # base 5, up to 10 with solo multiplier
    "DynamicEngine3":  3,    # DE3 typical live size
    "RegimeAdaptive":  3,
    "MLPhysics":       3,
}

BRACKETS_BY_KEY = {
    ("AetherFlow", "aligned_flow"):       (4.0, 3.5),
    ("AetherFlow", "transition_burst"):   (10.0, 6.0),
    ("AetherFlow", "compression_release"):(6.0, 4.0),
    ("AetherFlow", "*"):                  (6.0, 4.0),
    ("RegimeAdaptive", "*"):              (6.0, 4.0),
    ("RegimeAdaptive", "dead_tape"):      (3.0, 5.0),
    ("DynamicEngine3", "*"):              (8.25, 10.0),
    ("DynamicEngine3", "dead_tape"):      (3.0, 5.0),
    ("MLPhysics", "*"):                   (6.0, 4.0),
    ("*", "*"):                           (6.0, 4.0),
}


def bracket_for(strategy: str, family_or_regime: str) -> tuple[float, float]:
    fam = (family_or_regime or "").lower()
    if "dead_tape" in fam:
        key = (strategy, "dead_tape")
        if key in BRACKETS_BY_KEY:
            return BRACKETS_BY_KEY[key]
    return (BRACKETS_BY_KEY.get((strategy, family_or_regime))
            or BRACKETS_BY_KEY.get((strategy, "*"))
            or BRACKETS_BY_KEY[("*", "*")])


RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
RE_BAR = re.compile(r"Bar: [\d-]+ [\d:]+ ET \| Price: (?P<px>[\d.]+)")
RE_STRATEGY_SIG_BLOCK = re.compile(
    r"\[STRATEGY_SIGNAL\].*?strategy=(?P<strategy>\S+) "
    r"\| side=(?P<side>LONG|SHORT) \| price=(?P<price>[-\d.]+).*?"
    r"status=BLOCKED.*?reason=(?P<reason>\S+).*?"
    r"combo_key=(?P<family>\S+)"
    r"(?:.*?vol_regime=(?P<regime>\w+))?"
)
RE_KALSHI_BLOCK = re.compile(
    r"Kalshi overlay blocked entry: (?P<strategy>\S+) (?P<side>LONG|SHORT) "
    r"\| role=(?P<role>\S+) \| score=(?P<score>[\d.]+)<thresh=(?P<thresh>[\d.]+) "
    r"\| reason=(?P<reason>\S+)"
)
RE_SAME_SIDE = re.compile(
    r"Ignoring same-side signal while (?P<pos>LONG|SHORT) position is already active: (?P<strategy>\S+)"
)
RE_FILTER_CHECK_BLOCK = re.compile(
    r"\[FILTER_CHECK\] ✗ BLOCK - (?P<filter>\S+) for (?P<side>\S+) "
    r"\| filter=\S+ \| side=(?:LONG|SHORT|ALL) \| passed=False "
    r"\| strategy=(?P<strategy>\S+) \| reason=(?P<reason>.*?)(?:\| Status=|$)"
)


def parse_log_window(hours: float):
    signals = []
    bars = []
    with LOG.open(errors="ignore") as fh:
        lines = fh.readlines()

    last_ts = None
    for line in reversed(lines):
        m = RE_HEADER.match(line)
        if m:
            last_ts = m.group(1); break
    last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
    cutoff = last_dt - timedelta(hours=hours)
    print(f"[window] {cutoff} → {last_dt}  (bot-local PT, {hours}h)")

    latest_bar_price = None
    for line in lines:
        m = RE_HEADER.match(line)
        if not m: continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if ts < cutoff: continue
        bm = RE_BAR.search(line)
        if bm:
            latest_bar_price = float(bm.group("px"))
            bars.append((ts, latest_bar_price))

        sm = RE_STRATEGY_SIG_BLOCK.search(line)
        if sm:
            signals.append({
                "ts": ts, "strategy": sm.group("strategy"),
                "side": sm.group("side"), "price": float(sm.group("price")),
                "reason": f"threshold:{sm.group('reason')}",
                "filter": "AetherFlowThreshold", "family": sm.group("family"),
                "regime_hint": sm.group("regime") or "",
            })
            continue
        km = RE_KALSHI_BLOCK.search(line)
        if km and latest_bar_price is not None:
            signals.append({
                "ts": ts, "strategy": km.group("strategy"),
                "side": km.group("side"), "price": latest_bar_price,
                "reason": f"kalshi_score:{km.group('score')}<{km.group('thresh')}",
                "filter": "KalshiOverlay", "family": "*", "regime_hint": "",
            })
            continue
        im = RE_SAME_SIDE.search(line)
        if im and latest_bar_price is not None:
            signals.append({
                "ts": ts, "strategy": im.group("strategy"),
                "side": im.group("pos"), "price": latest_bar_price,
                "reason": "position_already_open",
                "filter": "SameSideSuppression", "family": "*", "regime_hint": "",
            })
            continue
        fm = RE_FILTER_CHECK_BLOCK.search(line)
        if fm and latest_bar_price is not None:
            f_name = fm.group("filter")
            f_side = fm.group("side")
            if f_side == "ALL" and f_name == "NewsFilter": continue
            if f_side not in ("LONG", "SHORT"): continue
            signals.append({
                "ts": ts, "strategy": fm.group("strategy"),
                "side": f_side, "price": latest_bar_price,
                "reason": fm.group("reason").strip() or "blocked",
                "filter": f_name, "family": "*", "regime_hint": "",
            })
    return signals, bars, last_dt


def walk_forward(signal, bars, lookahead=60):
    tp, sl = bracket_for(signal["strategy"],
                          signal.get("family") or signal.get("regime_hint") or "")
    entry = signal["price"]
    side_n = +1 if signal["side"] == "LONG" else -1
    walk = [(t, p) for t, p in bars if t > signal["ts"]][:lookahead]
    if not walk:
        return {"outcome": "open", "pnl": 0.0, "bars_held": 0, "tp": tp, "sl": sl}
    tp_price = entry + side_n * tp
    sl_price = entry - side_n * sl
    for i, (t, p) in enumerate(walk, start=1):
        if side_n > 0:
            if p <= sl_price: return {"outcome":"loss","pnl":-sl*MES_PT,"bars_held":i,"tp":tp,"sl":sl}
            if p >= tp_price: return {"outcome":"win", "pnl":tp*MES_PT,"bars_held":i,"tp":tp,"sl":sl}
        else:
            if p >= sl_price: return {"outcome":"loss","pnl":-sl*MES_PT,"bars_held":i,"tp":tp,"sl":sl}
            if p <= tp_price: return {"outcome":"win", "pnl":tp*MES_PT,"bars_held":i,"tp":tp,"sl":sl}
    last_px = walk[-1][1]
    pts = (last_px - entry) * side_n
    return {"outcome": "expired", "pnl": pts * MES_PT, "bars_held": len(walk),
             "tp": tp, "sl": sl}


def main():
    signals, bars, last_dt = parse_log_window(WINDOW_HOURS)
    print(f"[parse] raw block signals: {len(signals)}  bar prints: {len(bars)}")

    seen, uniq = set(), []
    for s in signals:
        key = (s["ts"].strftime("%Y-%m-%d %H:%M"), s["strategy"],
               s["side"], round(s["price"], 2), s["filter"])
        if key in seen: continue
        seen.add(key); uniq.append(s)
    print(f"[dedup] unique blocks: {len(uniq)}")

    for s in uniq:
        s.update(walk_forward(s, bars))
        # Live-size PnL
        live_size = LIVE_SIZE_DEFAULT.get(s["strategy"], 3)
        s["live_size"] = live_size
        s["pnl_live"] = s["pnl"] * live_size

    winners = [s for s in uniq if s["pnl"] > 0]
    losers  = [s for s in uniq if s["pnl"] < 0]
    flats   = [s for s in uniq if s["outcome"] == "expired" and s["pnl"] == 0]
    still_open = [s for s in uniq if s["outcome"] == "open"]
    win_pnl_1 = sum(s["pnl"] for s in winners)
    loss_pnl_1 = sum(s["pnl"] for s in losers)
    win_pnl_L = sum(s["pnl_live"] for s in winners)
    loss_pnl_L = sum(s["pnl_live"] for s in losers)
    n_resolved = len(winners) + len(losers) + len(flats)

    print()
    print("═" * 82)
    print(f" 6-HOUR BLOCKED-TRADE COUNTERFACTUAL — all filters")
    print(f" bot-local window: {last_dt - timedelta(hours=WINDOW_HOURS)} → {last_dt}")
    print("═" * 82)
    print(f"  would-have-won   : {len(winners):>3}")
    print(f"  would-have-lost  : {len(losers):>3}")
    print(f"  flats / still-open: {len(flats)} / {len(still_open)}")
    print(f"  resolved         : {n_resolved}")
    if n_resolved > 0:
        print(f"  resolved win rate: {100*len(winners)/n_resolved:.2f}%")
    print()
    print(f"  {'bucket':<16} {'size=1':>12} {'live-size':>12}")
    print(f"  {'winners':<16} ${win_pnl_1:>+10,.2f}  ${win_pnl_L:>+10,.2f}")
    print(f"  {'losers':<16} ${loss_pnl_1:>+10,.2f}  ${loss_pnl_L:>+10,.2f}")
    print(f"  {'NET':<16} ${win_pnl_1 + loss_pnl_1:>+10,.2f}  ${win_pnl_L + loss_pnl_L:>+10,.2f}")
    net1 = win_pnl_1 + loss_pnl_1
    netL = win_pnl_L + loss_pnl_L
    msg = "filters COST money" if net1 > 0 else "filters SAVED money"
    print(f"  → {msg} (size=1: ${net1:+,.2f}  |  live: ${netL:+,.2f})")

    print()
    print("  ── By block filter ──")
    print(f"  {'filter':<22} {'n':>4} {'W':>4} {'L':>4} {'WR':>7} {'size=1 $':>11} {'live $':>11}")
    by_f = defaultdict(lambda: {"n":0,"w":0,"l":0,"p1":0.0,"pL":0.0})
    for s in uniq:
        if s["outcome"] == "open": continue
        k = s["filter"]
        by_f[k]["n"] += 1
        by_f[k]["p1"] += s["pnl"]; by_f[k]["pL"] += s["pnl_live"]
        if s["pnl"] > 0: by_f[k]["w"] += 1
        elif s["pnl"] < 0: by_f[k]["l"] += 1
    for k, v in sorted(by_f.items(), key=lambda x: -abs(x[1]["p1"])):
        wr = 100 * v["w"] / max(1, v["n"])
        print(f"  {k:<22} {v['n']:>4} {v['w']:>4} {v['l']:>4} "
              f"{wr:>6.1f}%  ${v['p1']:>+9.2f}  ${v['pL']:>+9.2f}")

    print()
    print("  ── By strategy ──")
    print(f"  {'strategy':<22} {'n':>4} {'W':>4} {'L':>4} {'WR':>7} {'size=1 $':>11} {'live $':>11}")
    by_s = defaultdict(lambda: {"n":0,"w":0,"l":0,"p1":0.0,"pL":0.0})
    for s in uniq:
        if s["outcome"] == "open": continue
        k = s["strategy"]
        by_s[k]["n"] += 1
        by_s[k]["p1"] += s["pnl"]; by_s[k]["pL"] += s["pnl_live"]
        if s["pnl"] > 0: by_s[k]["w"] += 1
        elif s["pnl"] < 0: by_s[k]["l"] += 1
    for k, v in sorted(by_s.items(), key=lambda x: -abs(x[1]["p1"])):
        wr = 100 * v["w"] / max(1, v["n"])
        print(f"  {k:<22} {v['n']:>4} {v['w']:>4} {v['l']:>4} "
              f"{wr:>6.1f}%  ${v['p1']:>+9.2f}  ${v['pL']:>+9.2f}")

    # Top 15 biggest missed winners (at LIVE SIZE — that's the real cost)
    winners.sort(key=lambda s: -s["pnl_live"])
    print()
    print(f"  ── Top 15 biggest missed winners (live-size notional) ──")
    for s in winners[:15]:
        print(f"  {s['ts']}  {s['strategy']:<14}  {s['side']:<5}  "
              f"entry=${s['price']:.2f}  TP/SL={s['tp']:.1f}/{s['sl']:.1f} "
              f"×{s['live_size']}  +${s['pnl_live']:.2f}  [{s['filter']}]")


if __name__ == "__main__":
    main()

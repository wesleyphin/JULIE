#!/usr/bin/env python3
"""12h blocked-trade counterfactual — every filter, every strategy.

Pulls ALL block events from topstep_live_bot.log in the last 12 bot-local
hours (no cutoff on filter type) and walks each forward against bar
prices with proper per-strategy + dead_tape-aware brackets:

Block types captured:
    1. [STRATEGY_SIGNAL] BLOCKED   — AetherFlow family blocks by threshold
    2. Kalshi overlay blocked entry — DE3/RegimeAdaptive overlay blocks
    3. Ignoring same-side signal   — position-size suppression
    4. [FILTER_CHECK] ✗ BLOCK …    — NewsFilter / DirectionalConflict / etc.
    5. NEWS BLACKOUT / NEWS WAIT on candidate bars

Excluded (not actual live blocks):
    - [SHADOW_GATE_2025] would_veto=True — shadow measurements, not live
    - [RL_LIVE] skipped — position-management skips, not entry blocks
    - Order-level rejects (Topstep-side rejects, not strategy blocks)

Bracket resolution:
    - AetherFlow aligned_flow:       TP=4.0 / SL=3.5  (shipped fixed)
    - AetherFlow transition_burst:   TP=10.0 / SL=6.0 (shipped fixed)
    - AetherFlow compression_release: TP=6.0 / SL=4.0 (historic)
    - RegimeAdaptive (default):      TP=6.0 / SL=4.0
    - RegimeAdaptive (dead_tape):    TP=3.0 / SL=5.0  (dead-tape rewrite)
    - DynamicEngine3:                 TP=8.25 / SL=10.0 (observed typical)
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
WINDOW_HOURS = 12.0

# Bracket table — (tp, sl) in points
BRACKETS_BY_KEY = {
    ("AetherFlow", "aligned_flow"):      (4.0, 3.5),
    ("AetherFlow", "transition_burst"):  (10.0, 6.0),
    ("AetherFlow", "compression_release"):(6.0, 4.0),
    ("AetherFlow", "*"):                 (6.0, 4.0),
    ("RegimeAdaptive", "*"):             (6.0, 4.0),
    ("RegimeAdaptive", "dead_tape"):     (3.0, 5.0),     # dead-tape rewrite
    ("DynamicEngine3", "*"):             (8.25, 10.0),
    ("MLPhysics", "*"):                  (6.0, 4.0),
    ("*", "*"):                          (6.0, 4.0),
}


def bracket_for(strategy: str, family_or_regime: str) -> tuple[float, float]:
    """Resolve (tp, sl) for a strategy + family/regime context."""
    fam = (family_or_regime or "").lower()
    if strategy == "RegimeAdaptive" and "dead_tape" in fam:
        return BRACKETS_BY_KEY[("RegimeAdaptive", "dead_tape")]
    return (BRACKETS_BY_KEY.get((strategy, family_or_regime))
            or BRACKETS_BY_KEY.get((strategy, "*"))
            or BRACKETS_BY_KEY[("*", "*")])


RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
RE_BAR    = re.compile(r"Bar: [\d-]+ [\d:]+ ET \| Price: (?P<px>[\d.]+)")

# Block patterns
RE_STRATEGY_SIG_BLOCK = re.compile(
    r"\[STRATEGY_SIGNAL\].*?strategy=(?P<strategy>\S+) "
    r"\| side=(?P<side>LONG|SHORT) "
    r"\| price=(?P<price>[-\d.]+).*?"
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
RE_NEWS_BLACKOUT = re.compile(r"📰 NEWS BLACKOUT: (?P<reason>.*)")


def parse_log_window(hours: float):
    signals = []
    bars = []  # list of (dt, price)
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
    latest_bar_time = None
    open_position_side = None   # tracks active position to tag same-side suppressions

    for line in lines:
        m = RE_HEADER.match(line)
        if not m: continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if ts < cutoff: continue

        bm = RE_BAR.search(line)
        if bm:
            latest_bar_price = float(bm.group("px"))
            latest_bar_time = ts
            bars.append((ts, latest_bar_price))

        # Track position so we know the current position side for same-side block
        if "[TRADE_PLACED]" in line:
            if "side=LONG" in line: open_position_side = "LONG"
            elif "side=SHORT" in line: open_position_side = "SHORT"
        if "[TRADE_CLOSED]" in line or "Position Sync" in line and "FLAT" in line:
            open_position_side = None

        # STRATEGY_SIGNAL BLOCKED
        sm = RE_STRATEGY_SIG_BLOCK.search(line)
        if sm:
            reg = sm.group("regime") or ""
            signals.append({
                "ts": ts, "strategy": sm.group("strategy"),
                "side": sm.group("side"), "price": float(sm.group("price")),
                "reason": f"threshold:{sm.group('reason')}",
                "filter": "AetherFlowThreshold",
                "family": sm.group("family"),
                "regime_hint": reg,
            })
            continue

        km = RE_KALSHI_BLOCK.search(line)
        if km and latest_bar_price is not None:
            signals.append({
                "ts": ts, "strategy": km.group("strategy"),
                "side": km.group("side"), "price": latest_bar_price,
                "reason": f"kalshi_score:{km.group('score')}<{km.group('thresh')}",
                "filter": "KalshiOverlay",
                "family": "*", "regime_hint": "",
            })
            continue

        im = RE_SAME_SIDE.search(line)
        if im and latest_bar_price is not None:
            signals.append({
                "ts": ts, "strategy": im.group("strategy"),
                "side": im.group("pos"),
                "price": latest_bar_price,
                "reason": "position_already_open",
                "filter": "SameSideSuppression",
                "family": "*", "regime_hint": "",
            })
            continue

        fm = RE_FILTER_CHECK_BLOCK.search(line)
        if fm and latest_bar_price is not None:
            f_name = fm.group("filter")
            f_side = fm.group("side")
            if f_side == "ALL" and f_name == "NewsFilter":
                continue   # count news blocks via coincidence with STRATEGY_SIGNAL events
            if f_side not in ("LONG", "SHORT"):
                continue
            signals.append({
                "ts": ts, "strategy": fm.group("strategy"),
                "side": f_side, "price": latest_bar_price,
                "reason": fm.group("reason").strip() or "blocked",
                "filter": f_name,
                "family": "*", "regime_hint": "",
            })
            continue

    # --- find silent/unexplained blocks: CANDIDATE that never became TRADE_PLACED
    RE_CAND = re.compile(
        r"\[STRATEGY_SIGNAL\].*?strategy=(?P<strategy>\S+) "
        r"\| side=(?P<side>LONG|SHORT) \| price=(?P<price>[-\d.]+).*?status=CANDIDATE"
    )
    RE_PLACED = re.compile(
        r"\[TRADE_PLACED\].*?strategy=(?P<strategy>\S+) "
        r"\| side=(?P<side>LONG|SHORT) \| entry=(?P<entry>[-\d.]+)"
    )
    candidates = []
    placed_set = set()
    for line in lines:
        m = RE_HEADER.match(line)
        if not m: continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if ts < cutoff: continue
        c = RE_CAND.search(line)
        if c and "BLOCKED" not in line:
            candidates.append({
                "ts": ts, "strategy": c.group("strategy"),
                "side": c.group("side"), "price": float(c.group("price")),
            })
        p = RE_PLACED.search(line)
        if p:
            minute = ts.replace(second=0)
            for dm in range(-1, 3):
                placed_set.add((minute + timedelta(minutes=dm), p.group("strategy"), p.group("side")))
    # Existing blocks already captured — index by (minute, strat, side)
    blocked_keys = set()
    for s in signals:
        blocked_keys.add((s["ts"].replace(second=0), s["strategy"], s["side"]))
        # Also add neighboring minutes so we don't double-count candidate→block
        for dm in range(-1, 2):
            blocked_keys.add((s["ts"].replace(second=0) + timedelta(minutes=dm),
                              s["strategy"], s["side"]))
    for c in candidates:
        key = (c["ts"].replace(second=0), c["strategy"], c["side"])
        if key in placed_set: continue       # actually became a trade
        if key in blocked_keys: continue     # already captured by an explicit block
        # Unexplained — add as SilentBlock
        signals.append({
            "ts": c["ts"], "strategy": c["strategy"],
            "side": c["side"], "price": c["price"],
            "reason": "candidate_never_fired",
            "filter": "SilentBlock",
            "family": "*", "regime_hint": "",
        })
    return signals, bars, last_dt


def walk_forward(signal: dict, bars: list, lookahead: int = 60) -> dict:
    tp, sl = bracket_for(signal["strategy"], signal.get("family") or signal.get("regime_hint") or "")
    entry = signal["price"]
    side_n = +1 if signal["side"] == "LONG" else -1
    walk = [(t, p) for t, p in bars if t > signal["ts"]][:lookahead]
    if not walk:
        return {"outcome": "open", "pnl": 0.0, "bars_held": 0, "tp": tp, "sl": sl}
    tp_price = entry + side_n * tp
    sl_price = entry - side_n * sl
    for i, (t, p) in enumerate(walk, start=1):
        if side_n > 0:
            if p <= sl_price:
                return {"outcome": "loss", "pnl": -sl * MES_PT, "bars_held": i,
                        "tp": tp, "sl": sl}
            if p >= tp_price:
                return {"outcome": "win",  "pnl":  tp * MES_PT, "bars_held": i,
                        "tp": tp, "sl": sl}
        else:
            if p >= sl_price:
                return {"outcome": "loss", "pnl": -sl * MES_PT, "bars_held": i,
                        "tp": tp, "sl": sl}
            if p <= tp_price:
                return {"outcome": "win",  "pnl":  tp * MES_PT, "bars_held": i,
                        "tp": tp, "sl": sl}
    last_px = walk[-1][1]
    pts = (last_px - entry) * side_n
    return {"outcome": "expired", "pnl": pts * MES_PT, "bars_held": len(walk),
            "tp": tp, "sl": sl}


def main():
    signals, bars, last_dt = parse_log_window(WINDOW_HOURS)
    print(f"[parse] raw block signals captured: {len(signals)}  bar prints: {len(bars)}")

    # Dedupe at (minute, strategy, side, price, filter)
    seen, uniq = set(), []
    for s in signals:
        key = (s["ts"].strftime("%Y-%m-%d %H:%M"), s["strategy"],
               s["side"], round(s["price"], 2), s["filter"])
        if key in seen: continue
        seen.add(key); uniq.append(s)
    print(f"[dedup] unique blocks (1/minute/filter): {len(uniq)}")

    # Resolve each
    for s in uniq:
        s.update(walk_forward(s, bars))

    # Classify
    winners = [s for s in uniq if s["outcome"] == "win" or (s["outcome"] == "expired" and s["pnl"] > 0)]
    losers  = [s for s in uniq if s["outcome"] == "loss" or (s["outcome"] == "expired" and s["pnl"] < 0)]
    flats   = [s for s in uniq if s["outcome"] == "expired" and s["pnl"] == 0]
    still_open = [s for s in uniq if s["outcome"] == "open"]

    win_pnl  = sum(s["pnl"] for s in winners)
    loss_pnl = sum(s["pnl"] for s in losers)
    total    = win_pnl + loss_pnl
    n_resolved = len(winners) + len(losers) + len(flats)

    print()
    print("═" * 74)
    print(f" 12-HOUR BLOCKED-TRADE COUNTERFACTUAL — all filters (size=1 MES)")
    print(f" bot-local window: {last_dt - timedelta(hours=WINDOW_HOURS)} → {last_dt}")
    print("═" * 74)
    print(f"  would-have-won   : {len(winners):>4}   profitable PnL  ${win_pnl:>+10,.2f}")
    print(f"  would-have-lost  : {len(losers):>4}   loss-bucket PnL ${loss_pnl:>+10,.2f}")
    print(f"  flats            : {len(flats):>4}")
    print(f"  still-open (no data) : {len(still_open)}")
    print(f"  resolved         : {n_resolved}")
    if n_resolved > 0:
        print(f"  resolved win rate: {100*len(winners)/n_resolved:.2f}%")
    print(f"  NET of all blocks: ${total:+,.2f}  "
          f"({'filters COST money' if total > 0 else 'filters SAVED money'})")

    # By filter
    print()
    print("  ── By block filter ──")
    print(f"  {'filter':<22} {'n':>4} {'W':>4} {'L':>4} {'flat':>4} {'WR':>7} {'net $':>10}")
    by_f = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "flat": 0, "pnl": 0.0})
    for s in uniq:
        if s["outcome"] == "open": continue
        k = s["filter"]
        by_f[k]["n"] += 1; by_f[k]["pnl"] += s["pnl"]
        if s["outcome"] == "win" or (s["outcome"] == "expired" and s["pnl"] > 0):
            by_f[k]["w"] += 1
        elif s["outcome"] == "loss" or (s["outcome"] == "expired" and s["pnl"] < 0):
            by_f[k]["l"] += 1
        else:
            by_f[k]["flat"] += 1
    for k, v in sorted(by_f.items(), key=lambda x: -abs(x[1]["pnl"])):
        wr = 100 * v["w"] / max(1, v["n"])
        print(f"  {k:<22} {v['n']:>4} {v['w']:>4} {v['l']:>4} {v['flat']:>4} "
              f"{wr:>6.1f}%  ${v['pnl']:>+9.2f}")

    # By strategy
    print()
    print("  ── By strategy ──")
    print(f"  {'strategy':<22} {'n':>4} {'W':>4} {'L':>4} {'WR':>7} {'net $':>10}")
    by_s = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "pnl": 0.0})
    for s in uniq:
        if s["outcome"] == "open": continue
        k = s["strategy"]
        by_s[k]["n"] += 1; by_s[k]["pnl"] += s["pnl"]
        if s["pnl"] > 0: by_s[k]["w"] += 1
        elif s["pnl"] < 0: by_s[k]["l"] += 1
    for k, v in sorted(by_s.items(), key=lambda x: -abs(x[1]["pnl"])):
        wr = 100 * v["w"] / max(1, v["n"])
        print(f"  {k:<22} {v['n']:>4} {v['w']:>4} {v['l']:>4} {wr:>6.1f}%  ${v['pnl']:>+9.2f}")

    # Top 15 biggest missed winners
    winners.sort(key=lambda s: -s["pnl"])
    print()
    print(f"  ── Top 15 biggest missed winners (size=1 notional) ──")
    for s in winners[:15]:
        print(f"  {s['ts']}  {s['strategy']:<14}  {s['side']:<5}  "
              f"entry=${s['price']:.2f}  TP/SL={s['tp']:.1f}/{s['sl']:.1f}  "
              f"+${s['pnl']:.2f}  [{s['filter']}]  reason={s['reason'][:40]}")


if __name__ == "__main__":
    main()

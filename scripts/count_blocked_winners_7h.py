#!/usr/bin/env python3
"""Count winning/profitable trades BLOCKED in the last 7 hours.

Walks topstep_live_bot.log for:
    1. [STRATEGY_SIGNAL] BLOCKED entries — AetherFlow with gate_prob + price
    2. Same-side position suppression (DE3/RegimeAdaptive signals dropped because a position was already open)
    3. Filter-level blocks (NewsFilter, etc.)

For each blocked signal with a parseable entry price + side, forward-walks
the log's Bar prints (which capture the realized price for every bar the
bot saw) using per-strategy default brackets to determine hit-TP vs hit-SL
vs neither. Counts winners and their dollar PnL.

Why log-based price walk: live_prices.parquet is stale (ends 09:37 ET),
but topstep_live_bot.log has real-time Bar lines for every minute the bot
ran, including the last 7 hours.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter

ROOT = Path("/Users/wes/Downloads/JULIE001")
LOG = ROOT / "topstep_live_bot.log"
MES_PT = 5.0

# Log time is the bot's local PT. We treat it as naive for arithmetic.
# Default brackets by strategy/family (used when the block log line lacks
# tp_dist/sl_dist — which is the common case for pre-fire BLOCKED logs):
DEFAULT_BRACKETS = {
    "AetherFlow|aligned_flow":      (4.0, 3.5),
    "AetherFlow|transition_burst":  (10.0, 6.0),
    "AetherFlow|compression_release":(6.0, 4.0),   # deprecated family
    "AetherFlow|*":                 (6.0, 4.0),
    "RegimeAdaptive|*":             (6.0, 4.0),
    "DynamicEngine3|*":             (25.0, 10.0),  # DE3 uses wider brackets
}

# Regexes
RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
RE_BAR = re.compile(r"Bar: (?P<ts>[\d-]+ [\d:]+) ET \| Price: (?P<px>[\d.]+)")
RE_STRATEGY_SIGNAL = re.compile(
    r"\[STRATEGY_SIGNAL\].*?strategy=(?P<strategy>\S+) "
    r"\| side=(?P<side>LONG|SHORT) "
    r"\| price=(?P<price>[-\d.]+).*?"
    r"status=(?P<status>\w+).*?"
    r"(?:reason=(?P<reason>\S+).*?)?"
    r"combo_key=(?P<family>\S+)"
)
RE_IGNORE_SAME_SIDE = re.compile(
    r"Ignoring same-side signal while (?P<pos>LONG|SHORT) position is already active: (?P<strategy>\S+)"
)

# Also look for the bar's ET time in log messages. The bar line times are
# in NY (ET) but the log header is bot-local PT. For simplicity we track
# the most recent Bar price and use the HEADER time for ordering.


def parse_log(window_hours: float):
    """Parse the log, returning:
        (signals, bar_prices_by_pt_time)
    signals: list of dicts with ts_pt, strategy, side, price, status, reason, family
    bar_prices_by_pt_time: sorted list of (ts_pt, price) tuples
    """
    signals = []
    bar_prices = []
    # Find last timestamp to compute cutoff
    with LOG.open(errors="ignore") as fh:
        last_ts = None
        for raw in fh:
            m = RE_HEADER.match(raw)
            if m: last_ts = m.group(1)
    if not last_ts:
        raise RuntimeError("no timestamp in log")
    last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
    cutoff = last_dt - timedelta(hours=window_hours)
    print(f"[window] {cutoff} → {last_dt}  ({window_hours}h)")

    latest_bar_price = None
    with LOG.open(errors="ignore") as fh:
        for raw in fh:
            m = RE_HEADER.match(raw)
            if not m: continue
            ts_dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if ts_dt < cutoff: continue
            line = raw.rstrip()

            bm = RE_BAR.search(line)
            if bm:
                latest_bar_price = float(bm.group("px"))
                bar_prices.append((ts_dt, latest_bar_price))
                continue

            sm = RE_STRATEGY_SIGNAL.search(line)
            if sm and sm.group("status") == "BLOCKED":
                signals.append({
                    "ts": ts_dt,
                    "strategy": sm.group("strategy"),
                    "side": sm.group("side"),
                    "price": float(sm.group("price")),
                    "reason": sm.group("reason") or "",
                    "family": sm.group("family") or "*",
                    "type": "pre_filter_block",
                })
                continue

            im = RE_IGNORE_SAME_SIDE.search(line)
            if im and latest_bar_price is not None:
                # Same-side suppression: a candidate signal was generated but
                # ignored because a position was already open. Price is the
                # current bar's price at block time.
                signals.append({
                    "ts": ts_dt,
                    "strategy": im.group("strategy"),
                    "side": im.group("pos"),  # side that was blocked == same as open position
                    "price": latest_bar_price,
                    "reason": "same_side_suppressed",
                    "family": "*",
                    "type": "same_side_suppressed",
                })

    return signals, bar_prices


def simulate(signal, bar_prices, bar_lookahead_bars: int = 60):
    """Forward-walk bar_prices from signal.ts and determine TP/SL hit."""
    key = f"{signal['strategy']}|{signal['family']}"
    if key in DEFAULT_BRACKETS:
        tp_pts, sl_pts = DEFAULT_BRACKETS[key]
    else:
        tp_pts, sl_pts = DEFAULT_BRACKETS.get(f"{signal['strategy']}|*", (6.0, 4.0))
    entry = signal["price"]
    side = +1 if signal["side"] == "LONG" else -1
    # Find bars strictly after signal.ts
    walk = [(t, p) for t, p in bar_prices if t > signal["ts"]][:bar_lookahead_bars]
    if not walk:
        return {"outcome": "no_data", "pnl": 0.0, "bars_held": 0, "tp_pts": tp_pts, "sl_pts": sl_pts}
    if side > 0:
        tp_price = entry + tp_pts
        sl_price = entry - sl_pts
    else:
        tp_price = entry - tp_pts
        sl_price = entry + sl_pts
    for i, (t, p) in enumerate(walk, start=1):
        if side > 0:
            if p <= sl_price:
                return {"outcome": "loss", "pnl": -sl_pts * MES_PT, "bars_held": i,
                        "tp_pts": tp_pts, "sl_pts": sl_pts}
            if p >= tp_price:
                return {"outcome": "win",  "pnl":  tp_pts * MES_PT, "bars_held": i,
                        "tp_pts": tp_pts, "sl_pts": sl_pts}
        else:
            if p >= sl_price:
                return {"outcome": "loss", "pnl": -sl_pts * MES_PT, "bars_held": i,
                        "tp_pts": tp_pts, "sl_pts": sl_pts}
            if p <= tp_price:
                return {"outcome": "win",  "pnl":  tp_pts * MES_PT, "bars_held": i,
                        "tp_pts": tp_pts, "sl_pts": sl_pts}
    # Expired — close at last bar
    last_px = walk[-1][1]
    pts = (last_px - entry) if side > 0 else (entry - last_px)
    return {"outcome": "expired", "pnl": pts * MES_PT, "bars_held": len(walk),
            "tp_pts": tp_pts, "sl_pts": sl_pts}


def main():
    signals, bar_prices = parse_log(window_hours=7.0)
    print(f"[parse] blocked signals found: {len(signals)}  bar prints: {len(bar_prices)}")
    print()

    # Classify: dedup signals that are near-identical (same strategy/side/price/minute)
    seen = set()
    unique = []
    for s in signals:
        key = (s["ts"].strftime("%Y-%m-%d %H:%M"), s["strategy"], s["side"],
               round(s["price"], 2))
        if key in seen: continue
        seen.add(key)
        unique.append(s)
    print(f"[dedup] unique blocked signals (1/min): {len(unique)}")

    # Simulate each
    winners, losers, flats, expired, no_data = [], [], [], [], []
    for s in unique:
        out = simulate(s, bar_prices)
        s.update(out)
        if out["outcome"] == "win": winners.append(s)
        elif out["outcome"] == "loss": losers.append(s)
        elif out["outcome"] == "expired":
            if out["pnl"] > 0: winners.append(s); s["outcome"]="expired_win"
            elif out["pnl"] < 0: losers.append(s); s["outcome"]="expired_loss"
            else: flats.append(s)
        elif out["outcome"] == "no_data": no_data.append(s)

    # Aggregate
    total_pnl = sum(s["pnl"] for s in winners) + sum(s["pnl"] for s in losers)
    win_pnl = sum(s["pnl"] for s in winners)
    loss_pnl = sum(s["pnl"] for s in losers)
    n_total = len(winners) + len(losers) + len(flats)
    wr = 100.0 * len(winners) / max(1, n_total)
    print()
    print("═" * 70)
    print(f" BLOCKED-TRADE COUNTERFACTUAL — last 7 hours (size=1 simulation)")
    print("═" * 70)
    print(f"  winners   : {len(winners):>3}   profitable PnL  ${win_pnl:>+9,.2f}")
    print(f"  losers    : {len(losers):>3}   loss-bucket PnL ${loss_pnl:>+9,.2f}")
    print(f"  flats     : {len(flats):>3}")
    print(f"  no-data   : {len(no_data):>3}  (signal too late for bar-walk)")
    print(f"  resolved  : {n_total:>3}")
    print(f"  win rate  : {wr:.2f}%")
    print(f"  NET PnL   : ${total_pnl:+.2f}")
    print()

    # Breakdown by strategy
    print(f"  By strategy:")
    by_strat = {}
    for s in winners + losers + flats:
        k = s["strategy"]
        by_strat.setdefault(k, {"n":0,"w":0,"l":0,"pnl":0.0})
        by_strat[k]["n"] += 1
        if s["outcome"].startswith("win") or s["outcome"] == "expired_win":
            by_strat[k]["w"] += 1
        elif s["outcome"].startswith("loss") or s["outcome"] == "expired_loss":
            by_strat[k]["l"] += 1
        by_strat[k]["pnl"] += s["pnl"]
    for k, v in sorted(by_strat.items(), key=lambda x: -x[1]["pnl"]):
        wr_k = 100 * v["w"] / max(1, v["n"])
        print(f"    {k:<22} n={v['n']:>3}  W={v['w']:>3}  L={v['l']:>3}  WR={wr_k:5.1f}%  PnL=${v['pnl']:+.2f}")

    # Breakdown by block reason
    print()
    print(f"  By block reason (winners + losers combined):")
    by_reason = Counter(s["reason"] for s in (winners + losers + flats))
    win_by_reason = Counter(s["reason"] for s in winners)
    for r, cnt in by_reason.most_common():
        w = win_by_reason.get(r, 0)
        wr_r = 100 * w / max(1, cnt)
        print(f"    {r or '(none)':<30} n={cnt:>3}  W={w:>3}  WR={wr_r:5.1f}%")

    # Top 10 biggest missed winners
    winners.sort(key=lambda s: -s["pnl"])
    print()
    print(f"  Top 10 biggest missed winners (size=1 notional):")
    for s in winners[:10]:
        print(f"    {s['ts']}  {s['strategy']:<14}  {s['side']:<5}  "
              f"entry=${s['price']:.2f}  TP={s['tp_pts']:.1f}/SL={s['sl_pts']:.1f}  "
              f"PnL=${s['pnl']:+.2f}  reason={s['reason']}")


if __name__ == "__main__":
    main()

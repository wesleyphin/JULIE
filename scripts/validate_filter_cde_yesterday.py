#!/usr/bin/env python3
"""Simulate filters C+D+E on yesterday (and this week) of live-bot trades.

Parses /Users/wes/Downloads/JULIE001/topstep_live_bot.log for:
- Trade close lines (Early exit closed / Trade closed (reverse))
- Regime transitions
- trend_day=<tier>/<dir> markers from FILTER_CHECK lines (to re-enact filter C)

Then applies the same C/D/E logic that's now committed to main:
- C: veto _Rev_ sub-strategies that fade an active trend day (tier>=1)
- D: cap size to 1 when regime is whipsaw or calm_trend
- E: raise cap to 3 once cum daily PnL >= $200 in a capped regime

Reports per-day base (as-traded) vs simulated P&L and caveats.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
LOG = ROOT / "topstep_live_bot.log"

NY = ZoneInfo("America/New_York")
# Host TZ is PDT per log stamps (e.g. host 06:49 PDT == bar 09:48 ET). We keep
# everything in log-host time which is just ET offset by -3h. Convert trade
# times to ET so we bucket by ET trading day.
HOST_OFFSET_HOURS = 3  # host=PDT, ET=PDT+3

# Trade close line forms:
# 2026-04-18 02:02:09,438 [INFO] 📊 Trade closed (reverse): DynamicEngine3 SHORT | Entry: 7161.00 | Exit: 7163.75 | PnL: -2.75 pts ($-44.97) | source=reverse | order_id=500286 | entry_order_id=500283 | size=3 | live_dd=$126.11 | de3_reason=
# 2026-04-18 02:02:24,173 [INFO] 📊 Early exit closed: DynamicEngine3 LONG | Entry: 7158.25 | Exit: 7163.75 | PnL: 5.50 pts ($78.78) | source=close_position | order_id=500296 | entry_order_id=500293 | size=3 | live_dd=$178.53 | de3_reason=
RGX_TRADE = re.compile(
    r"^(?P<host_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] "
    r"📊 (?:Early exit closed|Trade closed \(reverse\)): "
    r"(?P<strategy>[^\s]+) (?P<side>LONG|SHORT) \| "
    r"Entry: [\d\.]+ \| Exit: [\d\.]+ \| "
    r"PnL: [\-\d\.]+ pts \(\$(?P<pnl>-?[\d\.]+)\).*"
    r"size=(?P<size>\d+)"
)

# Regime transition line form:
# 2026-04-20 06:49:59,348 [INFO] Regime transition: warmup -> calm_trend | ... | ts=2026-04-20 09:48:00-04:00
RGX_REGIME = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)

# Filter check with trend_day marker:
# ... | trend_day=0/none  or  trend_day=2/long
RGX_FILTER_TREND = re.compile(
    r"^(?P<host_ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r".*\[FILTER_CHECK\].*trend_day=(?P<tier>\d+)/(?P<dir>long|short|none)"
)

CAP_SIZE = 1
UNLOCK_THRESHOLD = 200.0
UNLOCK_SIZE = 3
REGIME_CAPPED = {"whipsaw", "calm_trend"}


def parse_host_dt(s: str) -> datetime:
    # Naive host-local datetime (PDT). We only need it for ordering + to derive
    # the trading-ET date; compute ET by adding HOST_OFFSET_HOURS.
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_regime_timeline(log_path: Path) -> list[tuple[datetime, str]]:
    events: list[tuple[datetime, str]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Regime transition" not in line:
                continue
            m = RGX_REGIME.search(line)
            if not m:
                continue
            regime = m.group("regime").strip()
            if regime == "dead_tape":
                regime = "neutral"
            try:
                ts = parse_iso(m.group("ts"))  # tz-aware ET
            except Exception:
                continue
            events.append((ts, regime))
    events.sort(key=lambda x: x[0])
    return events


def load_trend_timeline(log_path: Path) -> list[tuple[datetime, int, str]]:
    """Return (host_dt_naive, tier, dir) sorted, from FILTER_CHECK lines."""
    events = []
    last_seen: tuple[int, str] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "trend_day=" not in line or "FILTER_CHECK" not in line:
                continue
            m = RGX_FILTER_TREND.search(line)
            if not m:
                continue
            tier = int(m.group("tier"))
            direction = m.group("dir")
            # Only keep transitions to reduce list size
            now = (tier, direction)
            if now == last_seen:
                continue
            last_seen = now
            try:
                host_dt = parse_host_dt(m.group("host_ts"))
            except Exception:
                continue
            events.append((host_dt, tier, direction))
    return events


def load_trades(log_path: Path) -> list[dict]:
    trades = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Early exit closed" not in line and "Trade closed (reverse)" not in line:
                continue
            m = RGX_TRADE.search(line)
            if not m:
                continue
            host_dt = parse_host_dt(m.group("host_ts"))
            # Lift to ET
            from datetime import timedelta
            et_naive = host_dt + timedelta(hours=HOST_OFFSET_HOURS)
            et_dt = et_naive.replace(tzinfo=NY)
            # Also extract sub-strategy (might be "DynamicEngine3" with no sub)
            sub_strat = ""
            sub_m = re.search(r"sub_strategy=([^\s|]+)", line)
            if sub_m:
                sub_strat = sub_m.group(1)
            trades.append({
                "host_dt": host_dt,
                "et_dt": et_dt,
                "et_date": et_dt.date().isoformat(),
                "strategy": m.group("strategy"),
                "sub_strategy": sub_strat,
                "side": m.group("side"),
                "pnl": float(m.group("pnl")),
                "size": int(m.group("size")),
                "line": line.strip(),
            })
    return trades


def regime_at(ts_et: datetime, events: list[tuple[datetime, str]]) -> str:
    if not events:
        return "warmup"
    keys = [e[0] for e in events]
    i = bisect_right(keys, ts_et) - 1
    if i < 0:
        return "warmup"
    return events[i][1]


def trend_at(host_dt: datetime, events: list[tuple[datetime, int, str]]) -> tuple[int, str]:
    if not events:
        return 0, "none"
    keys = [e[0] for e in events]
    i = bisect_right(keys, host_dt) - 1
    if i < 0:
        return 0, "none"
    return events[i][1], events[i][2]


def simulate(trades: list[dict], regimes, trends) -> dict:
    # Group by ET date
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["et_date"]].append(t)

    day_rows = []
    total_base = 0.0
    total_sim = 0.0

    for day, day_trades in sorted(by_day.items()):
        day_trades.sort(key=lambda t: t["et_dt"])
        base_pnl = 0.0
        sim_pnl = 0.0
        cum_sim = 0.0
        caps = unlocks = vetos = 0

        for t in day_trades:
            size = t["size"]
            pnl = t["pnl"]
            base_pnl += pnl
            per_contract = pnl / size if size > 0 else pnl

            regime = regime_at(t["et_dt"], regimes)
            tier, direction = trend_at(t["host_dt"], trends)

            # Filter C: counter-trend reversal veto.
            # The trade has "_Rev_" in its sub_strategy if it's a Rev shape,
            # but with this live log the sub_strategy field is frequently empty
            # (just "DynamicEngine3"). Without it, C can't fire. We mark
            # "maybe_veto" on any trade where tier>=1 and side fades direction.
            c_fires = False
            if tier >= 1 and direction in ("long", "short"):
                trend_dir = direction
                fades = (trend_dir == "long" and t["side"] == "SHORT") or \
                        (trend_dir == "short" and t["side"] == "LONG")
                if fades and "_Rev_" in (t["sub_strategy"] or ""):
                    c_fires = True
            if c_fires:
                vetos += 1
                sim_pnl += 0.0  # vetoed trade: no PnL at all
                # cum_sim unchanged — vetoed trades don't contribute
                continue

            # Filter D + E: regime size cap + unlock
            capped = regime in REGIME_CAPPED
            if capped and size > CAP_SIZE:
                effective_cap = CAP_SIZE
                if cum_sim >= UNLOCK_THRESHOLD:
                    effective_cap = UNLOCK_SIZE
                    unlocks += 1
                new_size = min(size, effective_cap)
                if new_size < size:
                    caps += 1
                trade_pnl = per_contract * new_size
            else:
                trade_pnl = pnl

            sim_pnl += trade_pnl
            cum_sim += trade_pnl

        day_rows.append({
            "day": day,
            "trades": len(day_trades),
            "base_pnl": round(base_pnl, 2),
            "sim_pnl": round(sim_pnl, 2),
            "delta": round(sim_pnl - base_pnl, 2),
            "caps": caps,
            "unlocks": unlocks,
            "vetos": vetos,
        })
        total_base += base_pnl
        total_sim += sim_pnl

    return {
        "days": day_rows,
        "total_base": round(total_base, 2),
        "total_sim": round(total_sim, 2),
        "total_delta": round(total_sim - total_base, 2),
    }


def format_report(result: dict, focus_day: str | None = None) -> str:
    out = ["=" * 78]
    if focus_day:
        out.append(f"FOCUS: {focus_day} (yesterday)")
    out.append("=" * 78)
    out.append(f"{'day':<12}{'trades':>7}  {'base $':>10}  {'sim $':>10}  {'delta':>10}  caps/unlocks/vetos")
    out.append("-" * 78)
    for d in result["days"]:
        focus = "  <--" if d["day"] == focus_day else ""
        out.append(
            f"{d['day']:<12}{d['trades']:>7}  "
            f"{d['base_pnl']:>+10.2f}  {d['sim_pnl']:>+10.2f}  {d['delta']:>+10.2f}  "
            f"{d['caps']}/{d['unlocks']}/{d['vetos']}{focus}"
        )
    out.append("-" * 78)
    out.append(
        f"{'TOTAL':<12}{sum(d['trades'] for d in result['days']):>7}  "
        f"{result['total_base']:>+10.2f}  {result['total_sim']:>+10.2f}  {result['total_delta']:>+10.2f}"
    )
    return "\n".join(out)


if __name__ == "__main__":
    print(f"[scan] parsing {LOG}")
    regimes = load_regime_timeline(LOG)
    trends = load_trend_timeline(LOG)
    trades = load_trades(LOG)
    print(f"[scan] regime transitions: {len(regimes)}  trend-state changes: {len(trends)}  "
          f"trade closes: {len(trades)}")

    result = simulate(trades, regimes, trends)
    print(format_report(result, focus_day="2026-04-20"))

    # Zoom in on yesterday's trade-by-trade
    print()
    print("=" * 78)
    print("YESTERDAY (2026-04-20) trade-by-trade:")
    print("=" * 78)
    for t in [tr for tr in trades if tr["et_date"] == "2026-04-20"]:
        regime = regime_at(t["et_dt"], regimes)
        tier, direction = trend_at(t["host_dt"], trends)
        print(f"  {t['et_dt'].strftime('%H:%M')}  {t['strategy']:<40}  "
              f"{t['side']:>5}  size={t['size']:>2}  "
              f"pnl={t['pnl']:>+8.2f}  regime={regime:<11}  trend_day={tier}/{direction}")

    out = ROOT / "backtest_reports" / "validate_filter_cde_yesterday.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")

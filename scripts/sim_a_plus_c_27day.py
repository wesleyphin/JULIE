#!/usr/bin/env python3
"""Re-score the A+C filter combo on the 27-day outrageous+normal set.

Produces a per-day table (day / src / cat / trend_dir / baseline / C / A350 /
A500 / A+C $350 / A+C $500 / A+C gated $350) plus the aggregate totals
previously saved to backtest_reports/sim_a_plus_c_27day.json.

Filters:
  Filter A — intraday trailing-DD circuit breaker. Once peak→cum drops by
             $350 (or $500), block every remaining trade for the day.
  Filter C — counter-trend reversal veto. If day is classified breakout with
             trend_dir, block any _Rev_ sub-strategy that fades the trend.
             (Long_Rev on trend=down OR Short_Rev on trend=up).
  A+C gated $350 — Filter A active ONLY on breakout days; chop days keep
             the full baseline day.

Day universe (same 27 days previously scored in this session):
  11 outrageous_apr days + 16 non-April days (feb, mar, may, jun, jul, aug,
  oct, dec supersets).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
REPORT_ROOT = ROOT / "backtest_reports"
FULL_LR = REPORT_ROOT / "full_live_replay"
NY = ZoneInfo("America/New_York")

# (date, category, trend_dir, source_folder)
DAY_META = [
    # outrageous_apr — 11 April target days
    ('2025-04-01', 'chop',     None,   'outrageous_apr'),
    ('2025-04-02', 'chop',     None,   'outrageous_apr'),
    ('2025-04-04', 'breakout', 'down', 'outrageous_apr'),
    ('2025-04-07', 'breakout', 'up',   'outrageous_apr'),
    ('2025-04-08', 'breakout', 'down', 'outrageous_apr'),
    ('2025-04-09', 'breakout', 'up',   'outrageous_apr'),
    ('2025-04-10', 'breakout', 'down', 'outrageous_apr'),
    ('2025-04-15', 'chop',     None,   'outrageous_apr'),
    ('2025-04-16', 'breakout', 'down', 'outrageous_apr'),
    ('2025-04-24', 'breakout', 'up',   'outrageous_apr'),
    ('2025-04-28', 'chop',     None,   'outrageous_apr'),
    # 16 non-April outrageous/normal days
    ('2025-02-27', 'breakout', 'down', 'outrageous_feb'),
    ('2025-02-28', 'breakout', 'up',   'outrageous_feb'),
    ('2025-03-04', 'chop',     None,   '2025_03_ny_iter11_deadtape'),
    ('2025-03-13', 'chop',     None,   '2025_03_ny_iter11_deadtape'),
    ('2025-05-21', 'breakout', 'down', '2025_05_ny_iter11_deadtape'),
    ('2025-06-27', 'chop',     None,   '2025_06_ny_iter11_deadtape'),
    ('2025-07-21', 'chop',     None,   'outrageous_jul'),
    ('2025-07-22', 'chop',     None,   'outrageous_jul'),
    ('2025-08-14', 'chop',     None,   'outrageous_aug'),
    ('2025-08-22', 'breakout', 'up',   'outrageous_aug'),
    ('2025-08-27', 'chop',     None,   'outrageous_aug'),
    ('2025-10-10', 'breakout', 'down', 'outrageous_oct'),
    ('2025-10-16', 'breakout', 'down', 'outrageous_oct'),
    ('2025-12-05', 'chop',     None,   'outrageous_dec'),
    ('2025-12-15', 'breakout', 'up',   'outrageous_dec'),
    ('2025-12-17', 'breakout', 'up',   'outrageous_dec'),
]

APR_TARGET = {d for d, c, t, s in DAY_META if s == 'outrageous_apr'}
DD_LIMIT = 350.0  # "violation" threshold for reporting


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_trades_for(source: str) -> list[dict]:
    """Load closed_trades.json from either full_live_replay/<source> or
    backtest_reports/<source>."""
    for base in (FULL_LR, REPORT_ROOT):
        p = base / source / "closed_trades.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"closed_trades.json not found for {source}")


def is_reversal(t: dict) -> bool:
    sub = str(t.get("sub_strategy", "") or t.get("combo_key", "") or "")
    return ("Long_Rev" in sub) or ("Short_Rev" in sub) or ("_Rev_" in sub)


def simulate_day(trades: list[dict], *, trend_dir, filter_c: bool, a_thr: float | None,
                 gate_a_to_breakout: bool = False, is_breakout: bool = False):
    """Return (pnl, dd, c_vetoes, a_blocked) after applying chosen filters."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    c_vetoes = 0
    a_blocked = 0
    a_active = a_thr is not None and ((not gate_a_to_breakout) or is_breakout)
    a_tripped = False
    for t in sorted(trades, key=lambda x: parse_ts(x["entry_time"])):
        # Filter A pre-check — once tripped, block everything else
        if a_active and a_tripped:
            a_blocked += 1
            continue
        # Filter C — block counter-trend reversals on trend days
        if filter_c and trend_dir is not None and is_reversal(t):
            side = str(t.get("side", "")).upper()
            fades = (side == "LONG" and trend_dir == "down") or (
                side == "SHORT" and trend_dir == "up"
            )
            if fades:
                c_vetoes += 1
                continue
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        cum += pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        # Filter A trip check — after this trade if trailing DD hit
        if a_active and (peak - cum) >= a_thr:
            a_tripped = True
    return round(cum, 2), round(max_dd, 2), c_vetoes, a_blocked


def main():
    # Load each source once, bucket trades by day
    by_day: dict[str, list[dict]] = defaultdict(list)
    source_of: dict[str, str] = {d: s for d, c, t, s in DAY_META}
    trend_of: dict[str, tuple[str, str]] = {
        d: (c, t) for d, c, t, s in DAY_META
    }
    source_cache: dict[str, list[dict]] = {}
    for d, cat, tr, src in DAY_META:
        if src not in source_cache:
            source_cache[src] = load_trades_for(src)
        for t in source_cache[src]:
            try:
                ts = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
            except Exception:
                continue
            if ts == d:
                by_day[d].append(t)

    CONFIGS = [
        ("baseline",         dict(filter_c=False, a_thr=None)),
        ("C only",           dict(filter_c=True,  a_thr=None)),
        ("A350 only",        dict(filter_c=False, a_thr=350.0)),
        ("A500 only",        dict(filter_c=False, a_thr=500.0)),
        ("A+C $350",         dict(filter_c=True,  a_thr=350.0)),
        ("A+C $500",         dict(filter_c=True,  a_thr=500.0)),
        ("A+C gated $350",   dict(filter_c=True,  a_thr=350.0, gate_a_to_breakout=True)),
    ]

    # Per-day scores
    rows = []
    for d, _, _, _ in DAY_META:
        cat, tr = trend_of[d]
        is_bk = (cat == "breakout")
        trades = by_day.get(d, [])
        row = {"day": d, "src": source_of[d], "cat": cat, "trend": tr or "-",
               "n": len(trades)}
        for name, cfg in CONFIGS:
            kwargs = dict(cfg)
            kwargs["is_breakout"] = is_bk
            # gate_a_to_breakout may be missing from some configs
            kwargs.setdefault("gate_a_to_breakout", False)
            pnl, dd, cv, ab = simulate_day(trades, trend_dir=tr, **kwargs)
            row[name + "_pnl"] = pnl
            row[name + "_dd"] = dd
            row[name + "_cv"] = cv
            row[name + "_ab"] = ab
        rows.append(row)

    # Aggregate totals
    agg = {}
    for name, _ in CONFIGS:
        total = sum(r[name + "_pnl"] for r in rows)
        viol = sum(1 for r in rows if r[name + "_dd"] > DD_LIMIT)
        chop_p = sum(r[name + "_pnl"] for r in rows if r["cat"] == "chop")
        brk_p = sum(r[name + "_pnl"] for r in rows if r["cat"] == "breakout")
        apr_p = sum(r[name + "_pnl"] for r in rows if r["day"] in APR_TARGET)
        non_p = sum(r[name + "_pnl"] for r in rows if r["day"] not in APR_TARGET)
        cv = sum(r[name + "_cv"] for r in rows)
        ab = sum(r[name + "_ab"] for r in rows)
        apr9 = next((r[name + "_pnl"] for r in rows if r["day"] == "2025-04-09"), 0.0)
        trades_kept = sum(r["n"] for r in rows) - ab - cv
        agg[name] = dict(
            total=round(total, 2), viol=viol,
            chop=round(chop_p, 2), brk=round(brk_p, 2),
            apr=round(apr_p, 2), non_apr=round(non_p, 2),
            trades=trades_kept, c_vetoes=cv, a_blocked=ab,
            apr9=round(apr9, 2),
        )

    # Print per-day table
    CFGS = [c[0] for c in CONFIGS]
    print("=" * 140)
    print("Per-day breakdown (PnL $; dd in parens if > $350)")
    print("=" * 140)
    hdr = f"{'day':<12}{'src':<11}{'cat':<9}{'tr':<4}{'n':>4} "
    hdr += " ".join(f"{c:>14}" for c in CFGS)
    print(hdr)
    print("-" * len(hdr))
    # April first (sorted), then non-April (sorted)
    apr_rows = sorted([r for r in rows if r["day"] in APR_TARGET], key=lambda r: r["day"])
    non_rows = sorted([r for r in rows if r["day"] not in APR_TARGET], key=lambda r: r["day"])

    def fmt_cell(pnl, dd):
        star = "*" if dd > DD_LIMIT else " "
        return f"{pnl:>+10.2f}{star} "

    def print_group(group, label):
        sub_sub = {c: 0.0 for c in CFGS}
        sub_viol = {c: 0 for c in CFGS}
        for r in group:
            src_short = r["src"].replace("outrageous_", "").replace("_ny_iter11_deadtape", "")
            line = f"{r['day']:<12}{src_short:<11}{r['cat']:<9}{r['trend']:<4}{r['n']:>4} "
            for c in CFGS:
                line += fmt_cell(r[c + "_pnl"], r[c + "_dd"])
                sub_sub[c] += r[c + "_pnl"]
                if r[c + "_dd"] > DD_LIMIT:
                    sub_viol[c] += 1
            print(line)
        sub_line = f"{label + ' sub':<12}{'':<11}{'':<9}{'':<4}{'':>4} "
        for c in CFGS:
            sub_line += f"{sub_sub[c]:>+10.2f}  "
        print("-" * len(hdr))
        print(sub_line)
        viol_line = f"{'  DD viol':<12}{'':<11}{'':<9}{'':<4}{'':>4} "
        for c in CFGS:
            viol_line += f"{sub_viol[c]:>10d}  "
        print(viol_line)
        print()

    print_group(apr_rows, "APR (11)")
    print_group(non_rows, "NON (16)")

    # Aggregate
    print("=" * 140)
    print("27-day aggregate")
    print("=" * 140)
    ah = f"{'config':<20}{'total':>11}{'DD viol':>9}{'chop $':>11}{'breakout $':>12}{'APR $':>11}{'non-APR $':>12}{'trades':>8}{'C veto':>8}{'A block':>9}{'Apr9 $':>10}"
    print(ah)
    print("-" * len(ah))
    for name, _ in CONFIGS:
        a = agg[name]
        print(f"{name:<20}{a['total']:>+11.2f}{a['viol']:>9d}"
              f"{a['chop']:>+11.2f}{a['brk']:>+12.2f}"
              f"{a['apr']:>+11.2f}{a['non_apr']:>+12.2f}"
              f"{a['trades']:>8d}{a['c_vetoes']:>8d}{a['a_blocked']:>9d}"
              f"{a['apr9']:>+10.2f}")
    # Write JSON
    out = REPORT_ROOT / "sim_a_plus_c_27day_perday.json"
    out.write_text(json.dumps({"rows": rows, "agg": agg}, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()

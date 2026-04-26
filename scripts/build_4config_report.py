#!/usr/bin/env python3
"""Build the comprehensive 4-config comparison report.

Inputs: backtest_reports/{baseline,rules,mlstack,v8}_YYYY_MM/closed_trades.json
        backtest_reports/{config}_YYYY_MM/topstep_live_bot.log

Outputs:
  artifacts/4config_report/
    summary.json              — all configs, all aggregates
    monthly_per_config.csv    — every (config × month) row
    yearly_per_config.csv     — every (config × year) row
    overall_per_config.csv    — every config (whole-period totals)
    hourly_loss_breakdown.csv — top losing hours per (config × month)
    losing_regimes.csv        — top losing regime per (config × month)
    drawdown_dips.csv         — every DD dip > $200 (visible) and > $1200 (alert)
    delta_table.csv           — vs baseline lift / WR-Δ / DD-Δ / PnL-Δ
    REPORT.md                 — human-readable narrative
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
OUT = ROOT / "artifacts/4config_report"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = ["baseline", "rules", "mlstack", "v8"]
MONTHS = [
    "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08",
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04",
]
DD_DIP_ALERT = 1200.0   # user-requested threshold
DD_DIP_VISIBLE = 200.0  # smaller dips listed for context


def load_trades(config: str, month: str) -> list[dict]:
    p = ROOT / f"backtest_reports/{config}_{month.replace('-','_')}/closed_trades.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def trade_summary(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0, "wins": 0, "losses": 0, "wr": 0, "max_dd": 0,
                "avg_pnl": 0, "biggest_win": 0, "biggest_loss": 0,
                "long_n": 0, "long_pnl": 0, "short_n": 0, "short_pnl": 0,
                "dd_dips_alert": 0, "dd_dips_visible": 0, "all_dips": []}
    n = len(trades)
    pnl = sum(t.get("pnl_dollars", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("pnl_dollars", 0) > 0)
    losses = sum(1 for t in trades if t.get("pnl_dollars", 0) < 0)
    long_t = [t for t in trades if t.get("side") == "LONG"]
    short_t = [t for t in trades if t.get("side") == "SHORT"]

    # Equity curve + DD dips
    sorted_trades = sorted(trades, key=lambda t: t.get("entry_time", ""))
    cum, running = [], 0.0
    for t in sorted_trades:
        running += t.get("pnl_dollars", 0)
        cum.append(running)

    # Find drawdown DIPS — distinct events from peak down through trough
    dips = []
    peak = 0.0
    in_dip = False
    dip_start = 0.0
    for v in cum:
        if v > peak:
            if in_dip and (peak - min_in_dip) >= DD_DIP_VISIBLE:
                dips.append({"peak": peak, "trough": min_in_dip,
                             "depth": peak - min_in_dip})
            peak = v
            in_dip = False
            min_in_dip = v
        else:
            if not in_dip:
                in_dip = True
                min_in_dip = v
            else:
                min_in_dip = min(min_in_dip, v)
    if in_dip and (peak - min_in_dip) >= DD_DIP_VISIBLE:
        dips.append({"peak": peak, "trough": min_in_dip,
                     "depth": peak - min_in_dip})

    max_dd = max((d["depth"] for d in dips), default=0.0)
    dd_dips_alert = sum(1 for d in dips if d["depth"] >= DD_DIP_ALERT)
    dd_dips_visible = len(dips)

    return {
        "n": n, "pnl": round(pnl, 2),
        "wins": wins, "losses": losses,
        "wr": round(100*wins/n, 2) if n else 0,
        "avg_pnl": round(pnl/n, 2) if n else 0,
        "max_dd": round(max_dd, 2),
        "biggest_win": round(max((t.get("pnl_dollars", 0) for t in trades), default=0), 2),
        "biggest_loss": round(min((t.get("pnl_dollars", 0) for t in trades), default=0), 2),
        "long_n": len(long_t),
        "long_pnl": round(sum(t.get("pnl_dollars", 0) for t in long_t), 2),
        "short_n": len(short_t),
        "short_pnl": round(sum(t.get("pnl_dollars", 0) for t in short_t), 2),
        "dd_dips_alert": dd_dips_alert,
        "dd_dips_visible": dd_dips_visible,
        "all_dips": dips,
    }


def hourly_loss_breakdown(trades: list[dict]) -> list[dict]:
    """Return list of (hour, n, pnl, max_loss_in_hour) sorted by losing-est."""
    by_hour: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        try:
            ts = pd.Timestamp(t.get("entry_time", ""))
            by_hour[ts.hour].append(float(t.get("pnl_dollars", 0)))
        except Exception:
            continue
    rows = []
    for h, pnls in by_hour.items():
        rows.append({
            "hour_et": h,
            "n_trades": len(pnls),
            "total_pnl": round(sum(pnls), 2),
            "loss_pnl": round(sum(p for p in pnls if p < 0), 2),
            "biggest_single_loss": round(min(pnls), 2),
            "avg_pnl": round(sum(pnls)/len(pnls), 2),
        })
    rows.sort(key=lambda r: r["loss_pnl"])  # most negative loss_pnl first
    return rows


def losing_regimes_breakdown(config: str, month: str) -> list[dict]:
    """Parse log for regime tags around losing trades."""
    log = ROOT / f"backtest_reports/{config}_{month.replace('-','_')}/topstep_live_bot.log"
    if not log.exists():
        return []
    # Read log + find regime markers around closed-trade lines.
    # Regime is logged by the regime classifier as e.g. "regime: trend_up_calm"
    # or "vol_regime=normal". For each closed trade, find the regime
    # label active at trade entry time.
    RE_REGIME = re.compile(r"vol_regime=(\w+)|regime:\s*(\w+)|engine_session=(\w+)", re.IGNORECASE)
    RE_CLOSE = re.compile(
        r"Trade closed.*?(LONG|SHORT) \| Entry: ([\d.]+).*?PnL:\s*([-\d.]+)\s*pts.*?\$([-\d.]+)"
    )
    last_regime = "unknown"
    by_regime_loss = Counter()
    by_regime_n = Counter()
    by_regime_pnl = defaultdict(float)
    with log.open(errors="ignore") as f:
        for line in f:
            rm = RE_REGIME.search(line)
            if rm:
                groups = [g for g in rm.groups() if g]
                if groups:
                    last_regime = groups[0]
            cm = RE_CLOSE.search(line)
            if cm:
                pnl = float(cm.group(4))
                by_regime_n[last_regime] += 1
                by_regime_pnl[last_regime] += pnl
                if pnl < 0:
                    by_regime_loss[last_regime] += pnl
    rows = []
    for r in by_regime_n:
        rows.append({
            "regime": r,
            "n_trades": by_regime_n[r],
            "total_pnl": round(by_regime_pnl[r], 2),
            "loss_pnl": round(by_regime_loss[r], 2),
        })
    rows.sort(key=lambda x: x["loss_pnl"])
    return rows


def main():
    rows_monthly = []
    rows_yearly = []
    rows_overall = []
    all_dips = []
    all_hourly = []
    all_regimes = []

    # Per-config × per-month
    by_config_year_trades = defaultdict(lambda: defaultdict(list))
    by_config_total_trades = defaultdict(list)
    for cfg in CONFIGS:
        for m in MONTHS:
            trades = load_trades(cfg, m)
            summary = trade_summary(trades)
            year = m.split("-")[0]
            by_config_year_trades[cfg][year].extend(trades)
            by_config_total_trades[cfg].extend(trades)

            row = {"config": cfg, "month": m, **{k: v for k, v in summary.items() if k != "all_dips"}}
            rows_monthly.append(row)

            for d in summary["all_dips"]:
                all_dips.append({
                    "config": cfg, "month": m,
                    "peak": round(d["peak"], 2),
                    "trough": round(d["trough"], 2),
                    "depth": round(d["depth"], 2),
                    "alert_threshold": d["depth"] >= DD_DIP_ALERT,
                })

            for hr in hourly_loss_breakdown(trades)[:5]:  # top 5 losing hours
                all_hourly.append({"config": cfg, "month": m, **hr})

            for rg in losing_regimes_breakdown(cfg, m)[:5]:
                all_regimes.append({"config": cfg, "month": m, **rg})

    # Per-config × per-year
    for cfg in CONFIGS:
        for year, trades in by_config_year_trades[cfg].items():
            summary = trade_summary(trades)
            rows_yearly.append({"config": cfg, "year": year,
                                **{k: v for k, v in summary.items() if k != "all_dips"}})

    # Per-config overall
    for cfg in CONFIGS:
        summary = trade_summary(by_config_total_trades[cfg])
        rows_overall.append({"config": cfg, "period": "Mar2025-Apr2026",
                             **{k: v for k, v in summary.items() if k != "all_dips"}})

    # Delta table — vs baseline
    baseline_overall = next((r for r in rows_overall if r["config"] == "baseline"), None)
    delta_rows = []
    if baseline_overall and baseline_overall["n"] > 0:
        for r in rows_overall:
            cfg = r["config"]
            if cfg == "baseline":
                continue
            delta_rows.append({
                "config": cfg,
                "Δ_n_trades": r["n"] - baseline_overall["n"],
                "Δ_pnl": round(r["pnl"] - baseline_overall["pnl"], 2),
                "Δ_wr": round(r["wr"] - baseline_overall["wr"], 2),
                "Δ_max_dd": round(r["max_dd"] - baseline_overall["max_dd"], 2),
                "Δ_dd_dips_alert": r["dd_dips_alert"] - baseline_overall["dd_dips_alert"],
                "Δ_avg_pnl": round(r["avg_pnl"] - baseline_overall["avg_pnl"], 2),
            })

    # CSVs
    pd.DataFrame(rows_monthly).to_csv(OUT / "monthly_per_config.csv", index=False)
    pd.DataFrame(rows_yearly).to_csv(OUT / "yearly_per_config.csv", index=False)
    pd.DataFrame(rows_overall).to_csv(OUT / "overall_per_config.csv", index=False)
    pd.DataFrame(all_dips).to_csv(OUT / "drawdown_dips.csv", index=False)
    pd.DataFrame(all_hourly).to_csv(OUT / "hourly_loss_breakdown.csv", index=False)
    pd.DataFrame(all_regimes).to_csv(OUT / "losing_regimes.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT / "delta_table.csv", index=False)

    # Summary JSON
    with (OUT / "summary.json").open("w") as f:
        json.dump({
            "configs": CONFIGS, "months": MONTHS,
            "monthly": rows_monthly,
            "yearly": rows_yearly,
            "overall": rows_overall,
            "delta_vs_baseline": delta_rows,
            "dd_dip_alert_threshold": DD_DIP_ALERT,
            "n_dd_dips": {r["config"]: r["dd_dips_alert"] for r in rows_overall},
        }, f, indent=2, default=str)

    # Markdown narrative
    md = []
    md.append("# 4-Config Comparison Report")
    md.append("")
    md.append("## Overall (Mar 2025 – Apr 2026)")
    md.append("")
    md.append("| config | trades | WR | PnL | avg | max_DD | DD>$1200 dips |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows_overall:
        md.append(f"| {r['config']} | {r['n']} | {r['wr']:.1f}% | "
                  f"${r['pnl']:+,.2f} | ${r['avg_pnl']:+,.2f} | "
                  f"${r['max_dd']:,.2f} | {r['dd_dips_alert']} |")
    md.append("")
    md.append("## Delta vs baseline")
    md.append("")
    md.append("| config | Δ trades | Δ PnL | Δ WR | Δ max_DD | Δ DD>$1200 dips |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in delta_rows:
        md.append(f"| {r['config']} | {r['Δ_n_trades']:+} | ${r['Δ_pnl']:+,.2f} | "
                  f"{r['Δ_wr']:+.2f}pp | ${r['Δ_max_dd']:+,.2f} | {r['Δ_dd_dips_alert']:+} |")
    md.append("")
    md.append("## Yearly")
    md.append("")
    md.append("| config | year | trades | WR | PnL | max_DD | DD>$1200 dips |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(rows_yearly, key=lambda x: (x["config"], x["year"])):
        md.append(f"| {r['config']} | {r['year']} | {r['n']} | {r['wr']:.1f}% | "
                  f"${r['pnl']:+,.2f} | ${r['max_dd']:,.2f} | {r['dd_dips_alert']} |")
    md.append("")
    md.append("## Monthly (per config)")
    md.append("")
    for cfg in CONFIGS:
        md.append(f"### {cfg}")
        md.append("")
        md.append("| month | trades | WR | PnL | max_DD | DD>$1200 |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for r in [x for x in rows_monthly if x["config"] == cfg]:
            md.append(f"| {r['month']} | {r['n']} | {r['wr']:.1f}% | "
                      f"${r['pnl']:+,.2f} | ${r['max_dd']:,.2f} | {r['dd_dips_alert']} |")
        md.append("")
    md.append("## Worst losing hours (top-3 per config × month)")
    md.append("")
    md.append("| config | month | hour ET | trades | total PnL | biggest single loss |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in sorted(all_hourly, key=lambda x: x["loss_pnl"])[:60]:
        md.append(f"| {r['config']} | {r['month']} | {r['hour_et']:02d}:00 | "
                  f"{r['n_trades']} | ${r['total_pnl']:+,.2f} | ${r['biggest_single_loss']:+,.2f} |")
    md.append("")
    md.append("## Worst losing regimes (top-3 per config × month)")
    md.append("")
    md.append("| config | month | regime | trades | total PnL |")
    md.append("|---|---|---|---:|---:|")
    for r in sorted(all_regimes, key=lambda x: x["loss_pnl"])[:60]:
        md.append(f"| {r['config']} | {r['month']} | {r['regime']} | "
                  f"{r['n_trades']} | ${r['total_pnl']:+,.2f} |")
    md.append("")
    md.append("---")
    md.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    (OUT / "REPORT.md").write_text("\n".join(md))

    print(f"[wrote] {OUT}/")
    print(f"  summary.json")
    print(f"  monthly_per_config.csv ({len(rows_monthly)} rows)")
    print(f"  yearly_per_config.csv ({len(rows_yearly)} rows)")
    print(f"  overall_per_config.csv ({len(rows_overall)} rows)")
    print(f"  delta_table.csv ({len(delta_rows)} rows)")
    print(f"  drawdown_dips.csv ({len(all_dips)} rows)")
    print(f"  hourly_loss_breakdown.csv ({len(all_hourly)} rows)")
    print(f"  losing_regimes.csv ({len(all_regimes)} rows)")
    print(f"  REPORT.md")


if __name__ == "__main__":
    main()

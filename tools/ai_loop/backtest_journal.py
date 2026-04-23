"""Backtest-consensus journal writer.

Companion to `journal.py`. Instead of parsing the live log for a single
calendar day, this reads one or more `closed_trades.json` files from
replay directories and emits an aggregate journal that gives the AI-loop
analyzer a "here's what the bot looks like across a long tape" consensus.

Output shape matches `journal.py`'s structured JSON as closely as
possible so the analyzer layer can consume it without branching:

    ai_loop_data/journals/backtest_<label>.md
    ai_loop_data/journals/backtest_<label>.json

Usage:
    python3 -m tools.ai_loop.backtest_journal \
        --label 2026_full \
        --source backtest_reports/full_live_replay/2026_jan_apr/closed_trades.json \
        --source backtest_reports/full_live_replay/2026_04_ml_stacks/closed_trades.json

With --dedupe-by-entry-ts (default), trades sharing the same (entry_time,
strategy, side, entry_price) across sources are kept once — prevents
double-counting when overlapping replays are combined.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .config import JOURNALS_DIR, ROOT
from . import price_context


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_trades(paths: list[Path], *, dedupe: bool = True) -> list[dict]:
    combined: list[dict] = []
    seen: set[tuple] = set()
    for p in paths:
        raw = json.loads(p.read_text())
        for t in raw:
            t.setdefault("_source", str(p.relative_to(ROOT) if p.is_absolute() else p))
            if dedupe:
                key = (
                    t.get("entry_time"), t.get("strategy"),
                    t.get("side"), t.get("entry_price"),
                )
                if key in seen:
                    continue
                seen.add(key)
            combined.append(t)
    # sort chronologically
    combined.sort(key=lambda t: _parse_ts(t["entry_time"]))
    return combined


def _bucket_hour(ts: datetime) -> str:
    return f"{ts.hour:02d}:00"


def _session_of(ts: datetime) -> str:
    """NY-style session buckets (assume timestamps already in ET)."""
    h = ts.hour + ts.minute / 60.0
    if 4.0 <= h < 9.5:
        return "pre_open"
    if 9.5 <= h < 12.0:
        return "morning"
    if 12.0 <= h < 14.0:
        return "lunch"
    if 14.0 <= h < 16.0:
        return "afternoon"
    if 16.0 <= h < 17.0:
        return "post_close"
    return "overnight"


def _family(sub: str) -> str:
    # pull the Rev/Mom/Break token if present
    for tag in ("Long_Rev", "Short_Rev", "Long_Mom", "Short_Mom",
                "Long_Break", "Short_Break"):
        if tag in sub:
            return tag
    return "other"


def analyze(trades: list[dict]) -> dict:
    """Compute the big blob of per-axis stats the analyzer wants."""
    if not trades:
        return {"n_trades": 0}

    # ── Overall ──────────────────────────────────────────────
    pnls = [float(t.get("pnl_dollars", 0.0) or 0.0) for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    scratches = [p for p in pnls if p == 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0.0

    # ── Streaks ──────────────────────────────────────────────
    max_w_streak = max_l_streak = 0
    cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1; max_w_streak = max(max_w_streak, cur_w); cur_l = 0
        elif p < 0:
            cur_l += 1; max_l_streak = max(max_l_streak, cur_l); cur_w = 0
        else:
            cur_w = cur_l = 0

    # ── Overall running cum & DD ─────────────────────────────
    cum = 0.0; peak = 0.0; dd = 0.0
    equity_curve = []
    for t, p in zip(trades, pnls):
        cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
        equity_curve.append(round(cum, 2))

    # ── By day ──────────────────────────────────────────────
    by_day: dict[str, list[float]] = defaultdict(list)
    for t, p in zip(trades, pnls):
        day = t["entry_time"][:10]
        by_day[day].append(p)
    day_pnls = {d: round(sum(v), 2) for d, v in by_day.items()}
    win_days = sum(1 for d in day_pnls.values() if d > 0)
    loss_days = sum(1 for d in day_pnls.values() if d < 0)
    scratch_days = sum(1 for d in day_pnls.values() if d == 0)
    # worst / best days
    best_days = sorted(day_pnls.items(), key=lambda kv: -kv[1])[:5]
    worst_days = sorted(day_pnls.items(), key=lambda kv: kv[1])[:5]

    # ── By strategy / sub-strategy / family ──────────────────
    by_strat: dict[str, list[float]] = defaultdict(list)
    by_sub: dict[str, list[float]] = defaultdict(list)
    by_fam: dict[str, list[float]] = defaultdict(list)
    by_side: dict[str, list[float]] = defaultdict(list)
    by_exit: dict[str, list[float]] = defaultdict(list)
    by_hour: dict[str, list[float]] = defaultdict(list)
    by_session: dict[str, list[float]] = defaultdict(list)

    for t, p in zip(trades, pnls):
        by_strat[t.get("strategy", "?")].append(p)
        sub = t.get("sub_strategy") or t.get("combo_key") or "?"
        by_sub[sub].append(p)
        by_fam[_family(sub)].append(p)
        by_side[str(t.get("side", "?")).upper()].append(p)
        by_exit[t.get("source", "?")].append(p)
        ts = _parse_ts(t["entry_time"])
        by_hour[_bucket_hour(ts)].append(p)
        by_session[_session_of(ts)].append(p)

    def _bucket_stats(d: dict[str, list[float]]) -> dict:
        out = {}
        for k, v in d.items():
            if not v: continue
            w = sum(1 for x in v if x > 0)
            out[k] = {
                "n": len(v),
                "pnl": round(sum(v), 2),
                "wr": round(w / len(v) * 100, 1),
                "avg": round(mean(v), 2),
                "median": round(median(v), 2),
                "best": round(max(v), 2),
                "worst": round(min(v), 2),
            }
        return out

    # ── Pattern flags (same spirit as journal.flag_patterns) ──
    flags: list[str] = []
    if max_l_streak >= 4:
        flags.append(
            f"⚠ max losing streak = {max_l_streak} trades (chop / overtrading signature)"
        )
    tiny_losses = sum(1 for p in pnls if -25 < p < 0)
    if tiny_losses / max(1, len(losses)) >= 0.30:
        flags.append(
            f"⚠ {tiny_losses}/{len(losses)} losses are small (<$25) — "
            f"SL-tightening / flip-flop signature"
        )
    # Side skew
    long_n = len(by_side.get("LONG", []))
    short_n = len(by_side.get("SHORT", []))
    if long_n + short_n > 0:
        skew = abs(long_n - short_n) / (long_n + short_n)
        if skew > 0.40:
            flags.append(
                f"⚠ side skew {long_n}L/{short_n}S ({skew*100:.0f}%) — "
                f"tape had a directional bias this window"
            )
    # Worst family
    fam_stats = _bucket_stats(by_fam)
    for fam, s in fam_stats.items():
        if s["n"] >= 20 and s["pnl"] < -200:
            flags.append(
                f"⚠ family {fam}: {s['n']} trades netted ${s['pnl']:+.0f} "
                f"(WR {s['wr']}%) — underperforming vs peers"
            )
    # Loss tail
    big_losers = [p for p in losses if p < -200]
    if len(big_losers) >= 5:
        flags.append(
            f"⚠ {len(big_losers)} trades lost ≥ $200 (sum ${sum(big_losers):+.0f}) — "
            f"the left-tail is the drag"
        )
    # Consecutive-loss cluster within a single day (cascade signature)
    cascade_days = []
    for day, v in by_day.items():
        cur = 0; mx = 0
        for p in v:
            if p < 0: cur += 1; mx = max(mx, cur)
            else: cur = 0
        if mx >= 3:
            cascade_days.append((day, mx, round(sum(v), 2)))
    if cascade_days:
        cascade_days.sort(key=lambda r: -r[1])
        top = cascade_days[:3]
        flags.append(
            f"⚠ {len(cascade_days)} days had ≥3-loss intraday cascade "
            f"(top: " + ", ".join(f"{d} x{n} net ${p:+.0f}" for d,n,p in top) + ")"
        )
    # Short-window large drawdown
    if dd > 1500:
        flags.append(
            f"⚠ peak-to-trough DD on the full tape = ${dd:,.0f} "
            f"(equity curve not monotone)"
        )

    return {
        "n_trades": len(trades),
        "n_days": len(by_day),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(wr, 1),
        "n_wins": len(wins), "n_losses": len(losses), "n_scratches": len(scratches),
        "avg_trade": round(mean(pnls), 2),
        "median_trade": round(median(pnls), 2),
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "std_trade": round(pstdev(pnls), 2) if len(pnls) > 1 else 0.0,
        "avg_win": round(mean(wins), 2) if wins else 0.0,
        "avg_loss": round(mean(losses), 2) if losses else 0.0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else float('inf'),
        "max_win_streak": max_w_streak,
        "max_loss_streak": max_l_streak,
        "max_drawdown": round(dd, 2),
        "win_days": win_days, "loss_days": loss_days, "scratch_days": scratch_days,
        "day_win_rate": round(win_days / max(1, len(by_day)) * 100, 1),
        "avg_day_pnl": round(mean(day_pnls.values()), 2),
        "best_day_pnl": round(max(day_pnls.values()), 2),
        "worst_day_pnl": round(min(day_pnls.values()), 2),
        "best_days": best_days,
        "worst_days": worst_days,
        "by_strategy": _bucket_stats(by_strat),
        "by_sub_strategy": _bucket_stats(by_sub),
        "by_family": _bucket_stats(by_fam),
        "by_side": _bucket_stats(by_side),
        "by_exit_source": _bucket_stats(by_exit),
        "by_hour": _bucket_stats(by_hour),
        "by_session": _bucket_stats(by_session),
        "pattern_flags": flags,
        "equity_curve_tail": equity_curve[-50:],
        "date_range": (trades[0]["entry_time"][:10], trades[-1]["entry_time"][:10]),
    }


def render_markdown(label: str, sources: list[Path], stats: dict) -> str:
    lines = [
        f"# Julie backtest consensus journal — {label}",
        "",
        f"Compiled: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Date range: **{stats['date_range'][0]} → {stats['date_range'][1]}** "
        f"({stats['n_days']} trading days, {stats['n_trades']} trades)",
        "",
        "## Sources",
        "",
    ]
    for p in sources:
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            rel = p
        lines.append(f"- `{rel}`")
    lines += [
        "",
        "## Headline — \"what it's going to be like\"",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Net PnL | **${stats['total_pnl']:+,.2f}** |",
        f"| Profit factor | {stats['profit_factor']:.2f} |",
        f"| Trade WR | {stats['win_rate']:.1f}% ({stats['n_wins']}W / {stats['n_losses']}L) |",
        f"| Day WR | {stats['day_win_rate']:.1f}% ({stats['win_days']}W / {stats['loss_days']}L / {stats['scratch_days']}S) |",
        f"| Avg trade | ${stats['avg_trade']:+.2f} (median ${stats['median_trade']:+.2f}, σ ${stats['std_trade']:.2f}) |",
        f"| Avg win / Avg loss | ${stats['avg_win']:+.2f} / ${stats['avg_loss']:+.2f} |",
        f"| Best / Worst trade | ${stats['best_trade']:+.2f} / ${stats['worst_trade']:+.2f} |",
        f"| Avg day PnL | ${stats['avg_day_pnl']:+.2f} (range ${stats['worst_day_pnl']:+.0f} … ${stats['best_day_pnl']:+.0f}) |",
        f"| Max drawdown (tape) | ${stats['max_drawdown']:,.2f} |",
        f"| Longest win / loss streak | {stats['max_win_streak']} / {stats['max_loss_streak']} |",
        "",
        "## Per-side breakdown",
        "",
        "| side | n | PnL | WR | avg | median | best | worst |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for side, s in stats["by_side"].items():
        lines.append(
            f"| {side} | {s['n']} | ${s['pnl']:+.2f} | {s['wr']:.1f}% | "
            f"${s['avg']:+.2f} | ${s['median']:+.2f} | ${s['best']:+.2f} | ${s['worst']:+.2f} |"
        )

    lines += ["", "## Per-family breakdown (Long/Short × Rev/Mom/Break)", ""]
    lines.append("| family | n | PnL | WR | avg |")
    lines.append("|---|---:|---:|---:|---:|")
    for fam, s in sorted(stats["by_family"].items(), key=lambda kv: -kv[1]["pnl"]):
        lines.append(
            f"| {fam} | {s['n']} | ${s['pnl']:+.2f} | {s['wr']:.1f}% | ${s['avg']:+.2f} |"
        )

    lines += ["", "## Per-sub-strategy leaderboard (top 10 by PnL)", ""]
    lines.append("| sub_strategy | n | PnL | WR | avg |")
    lines.append("|---|---:|---:|---:|---:|")
    for sub, s in sorted(stats["by_sub_strategy"].items(),
                         key=lambda kv: -kv[1]["pnl"])[:10]:
        lines.append(
            f"| {sub} | {s['n']} | ${s['pnl']:+.2f} | {s['wr']:.1f}% | ${s['avg']:+.2f} |"
        )
    lines += ["", "### Worst 5 sub-strategies (the drag)", ""]
    lines.append("| sub_strategy | n | PnL | WR | avg | worst |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sub, s in sorted(stats["by_sub_strategy"].items(),
                         key=lambda kv: kv[1]["pnl"])[:5]:
        lines.append(
            f"| {sub} | {s['n']} | ${s['pnl']:+.2f} | {s['wr']:.1f}% | "
            f"${s['avg']:+.2f} | ${s['worst']:+.2f} |"
        )

    lines += ["", "## Per-session buckets (NY hours)", ""]
    lines.append("| session | n | PnL | WR | avg |")
    lines.append("|---|---:|---:|---:|---:|")
    order = ["pre_open", "morning", "lunch", "afternoon", "post_close", "overnight"]
    for sess in order:
        if sess not in stats["by_session"]: continue
        s = stats["by_session"][sess]
        lines.append(
            f"| {sess} | {s['n']} | ${s['pnl']:+.2f} | {s['wr']:.1f}% | ${s['avg']:+.2f} |"
        )

    lines += ["", "## Exit-source breakdown", ""]
    lines.append("| exit source | n | PnL | WR | avg |")
    lines.append("|---|---:|---:|---:|---:|")
    for src, s in sorted(stats["by_exit_source"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(
            f"| {src} | {s['n']} | ${s['pnl']:+.2f} | {s['wr']:.1f}% | ${s['avg']:+.2f} |"
        )

    lines += ["", "## Best 5 days / Worst 5 days", "",
              "| rank | best day | $ | worst day | $ |",
              "|---:|---|---:|---|---:|"]
    best = stats["best_days"]; worst = stats["worst_days"]
    for i in range(max(len(best), len(worst))):
        b = best[i] if i < len(best) else ("—", 0.0)
        w = worst[i] if i < len(worst) else ("—", 0.0)
        lines.append(f"| {i+1} | {b[0]} | ${b[1]:+.2f} | {w[0]} | ${w[1]:+.2f} |")

    pc = stats.get("price_context")
    if pc and pc.get("available"):
        lines += [
            "", "## Price-regime context (from logs → parquet)", "",
            f"Days with price data on record: **{pc['n_days_with_prices']}**",
            "",
            "| subset | n | avg intraday range (pts) | avg bar-to-bar vol (pts) |",
            "|---|---:|---:|---:|",
        ]
        for label, k in (("best days", "best_days_summary"), ("worst days", "worst_days_summary")):
            s = pc.get(k) or {}
            lines.append(
                f"| {label} | {s.get('n', 0)} | "
                f"{s.get('avg_range_pts') if s.get('avg_range_pts') is not None else '—'} | "
                f"{s.get('avg_bar_vol_pts') if s.get('avg_bar_vol_pts') is not None else '—'} |"
            )
        # Highlight the single widest range day in the whole window
        per_day = pc.get("per_day", {}) or {}
        if per_day:
            widest = max(per_day.items(), key=lambda kv: kv[1].get("range_pts", 0))
            d, ctx = widest
            lines += [
                "",
                f"- Widest single-day range: **{d}** with **{ctx.get('range_pts', '?')}pt** "
                f"range ({ctx.get('trend_dir', '?')}, bar-vol {ctx.get('bar_vol_pts', '?')})",
            ]
    elif pc and not pc.get("available"):
        lines += [
            "", "## Price-regime context", "",
            f"(unavailable: {pc.get('reason','parquet missing — run `python3 -m tools.ai_loop.price_parquet_updater`')})",
        ]

    lines += ["", "## Auto-flagged patterns", ""]
    if stats["pattern_flags"]:
        for fl in stats["pattern_flags"]:
            lines.append(f"- {fl}")
    else:
        lines.append("✓ no patterns flagged — tape looks clean")

    lines += ["", "## Analyzer consensus prompt", "",
              "Use the above to answer:",
              "1. Which sub-strategies have negative expectancy AND ≥20 trades? (candidates for retirement)",
              "2. Which session buckets have negative expectancy? (candidates for trade-window restriction)",
              "3. Are LONG and SHORT symmetric, or is one side the drag?",
              "4. Do the cascade-day flags suggest a circuit breaker would help?",
              "5. Is profit factor > 1.30? (if not, no single param tweak will save it — need structural change)",
              ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="short label for output filenames (e.g. 2026_full)")
    ap.add_argument("--source", action="append", required=True,
                    help="closed_trades.json path (repeatable)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="disable dedupe-by-entry-ts")
    ap.add_argument("--out-dir", default=None,
                    help="override output directory (default: ai_loop_data/journals)")
    args = ap.parse_args()

    paths = [Path(s) for s in args.source]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"source not found: {p}")

    trades = load_trades(paths, dedupe=not args.no_dedupe)
    stats = analyze(trades)
    # Attach price-regime context (per-day OHLC/range/trend from the
    # logs-derived parquet) so downstream analyzer rules can correlate
    # trade outcomes with price action, not just time-of-day.
    stats = price_context.annotate_backtest_stats(stats)

    out_dir = Path(args.out_dir) if args.out_dir else JOURNALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"backtest_{args.label}.md"
    js_path = out_dir / f"backtest_{args.label}.json"

    md_path.write_text(render_markdown(args.label, paths, stats), encoding="utf-8")
    js_path.write_text(
        json.dumps({
            "label": args.label,
            "sources": [str(p) for p in paths],
            "generated_at": datetime.now().isoformat(),
            "stats": stats,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[backtest-journal] wrote {md_path}")
    print(f"[backtest-journal] wrote {js_path}")
    print(f"[backtest-journal] {stats['n_trades']} trades / "
          f"{stats['n_days']} days / ${stats['total_pnl']:+.2f}")


if __name__ == "__main__":
    main()

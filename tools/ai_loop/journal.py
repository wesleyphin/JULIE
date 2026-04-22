"""Layer 1 — Daily journal writer.

Reads yesterday's (or a specified date's) trading activity and emits a
human-readable markdown journal plus a structured JSON sidecar.

Inputs:
    topstep_live_bot.log    — the bot's structured log
    bot_state.json          — current circuit-breaker / live_position state
    (optional) closed_trades.json from a replay dir

Output:
    ai_loop_data/journals/YYYY-MM-DD.md    (markdown for humans)
    ai_loop_data/journals/YYYY-MM-DD.json  (structured for the analyzer)

Journal sections:
    - Session summary (PnL, WR, DD, # trades, # blocks)
    - Per-trade attribution table (entry/exit + which ML layers decided what)
    - ML-layer block-rate breakdown (how many signals each layer killed)
    - Counterfactual: what happened to blocked signals
    - Auto-flagged patterns (4+ consec losses, chop, etc.)
    - Cross-market context snapshot
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Optional

from .config import ROOT, JOURNALS_DIR

LOG_PATH = ROOT / "topstep_live_bot.log"
BOT_STATE_PATH = ROOT / "bot_state.json"


# ─── Log line parsers ────────────────────────────────────────
_RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_RE_SIGNAL_TAG = re.compile(r"\[STRATEGY_SIGNAL\]")
_RE_SIDE = re.compile(r"side=(\w+)")
_RE_PRICE = re.compile(r"price=([\d.]+)")
_RE_TP = re.compile(r"tp_dist=([\d.]+)")
_RE_SL = re.compile(r"sl_dist=([\d.]+)")
_RE_STRAT = re.compile(r"strategy=(\w[\w:]*)")
_RE_STATUS = re.compile(r"status=(\w+)")
_RE_KALSHI_VIEW = re.compile(
    r"\[KALSHI_ENTRY_VIEW\].*decision=(\w+).*entry_probability=([\d.]+).*"
    r"probe_probability=([\d.]+).*support_score=([\d.]+).*threshold=([\d.]+)"
)
_RE_KALSHI_BLOCK = re.compile(r"Kalshi overlay blocked entry:\s+(\w+)")
_RE_CM_OVERRIDE = re.compile(r"\[KALSHI_CM_OVERRIDE\].*side=(\w+)\s+(.*)")
_RE_CM_GATE_V2 = re.compile(
    r"\[CM_GATE_V2\].*side=(\w+) p_direction=([\d.]+) would_override=(\w+) active=(\w+)"
)
_RE_CM_GATE_ML = re.compile(
    r"\[CM_GATE_ML\].*side=(\w+) p_win=([\d.]+) would_override=(\w+) active=(\w+)"
)
_RE_SHADOW_RL = re.compile(
    r"\[SHADOW_RL\].*strat=(\S+).*side=(\w+).*bar=(\d+).*pnl_pts=([+\-\d.]+).*action=(\w+)"
)
_RE_RL_LIVE = re.compile(r"\[RL_LIVE\]\s+status=(\w+)\s+(?:action=(\w+))?")
_RE_RL_SLMOVE = re.compile(r"\[RL_LIVE\]\s+\S+\s+(\w+)\s+SL\s+→\s+([\d.]+)\s+\(action=(\w+)")
_RE_SHADOW_LFO = re.compile(
    r"\[SHADOW_LFO\]\s+rule=(\w+)\s+ml=(\w+)\s+p_wait=([\d.]+)\s+thr=([\d.]+).*"
    r"strat=(\S+)\s+side=(\w+)"
)
_RE_SHADOW_GATE = re.compile(
    r"\[SHADOW_GATE_2025\].*family=(\w+).*side=(\w+).*"
    r"P\(big_loss\)=([\d.]+).*thresh=([\d.]+).*would_veto=(\w+)"
)
_RE_TRADE_PLACED = re.compile(
    r"\[TRADE_PLACED\].*order_id=([\w\-]+).*strategy=(\w+).*side=(\w+).*"
    r"entry=([\d.]+).*tp=([\d.]+).*sl=([\d.]+)"
)
_RE_TRADE_CLOSED = re.compile(
    r"Trade closed.*:\s*(\w+)\s+(\w+).*Entry:\s*([\d.]+).*Exit:\s*([\d.]+).*"
    r"PnL:\s*([+\-\d.]+)\s*pts\s*\(\$([+\-\d.]+)\).*source=(\w+)"
)
_RE_HEARTBEAT = re.compile(r"💓 Heartbeat.*Price:\s*([\d.]+)")


def _ts_from_line(line: str) -> Optional[datetime]:
    m = _RE_TS.match(line)
    if not m: return None
    try: return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except Exception: return None


def parse_log_for_date(log_path: Path, target_date: date) -> dict:
    """Walk the whole log file, pick out events dated `target_date`, bucket them."""
    prefix = target_date.strftime("%Y-%m-%d")
    stats = {
        "date": prefix,
        "signals_fired": [],
        "kalshi_views": [],
        "kalshi_blocks": [],
        "cm_overrides": [],
        "cm_gate_v2": [],
        "cm_gate_ml": [],
        "shadow_rl": [],
        "rl_live": [],
        "shadow_lfo": [],
        "shadow_gate": [],
        "trades_placed": [],
        "trades_closed": [],
        "heartbeats": [],
    }
    if not log_path.exists():
        return stats
    with log_path.open() as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            ts = _ts_from_line(line)
            if ts is None: continue
            if _RE_SIGNAL_TAG.search(line):
                fs = {
                    "side": _RE_SIDE.search(line),
                    "price": _RE_PRICE.search(line),
                    "tp_dist": _RE_TP.search(line),
                    "sl_dist": _RE_SL.search(line),
                    "strategy": _RE_STRAT.search(line),
                    "status": _RE_STATUS.search(line),
                }
                if all(fs.values()):
                    stats["signals_fired"].append({
                        "ts": ts.isoformat(), "side": fs["side"].group(1),
                        "price": float(fs["price"].group(1)),
                        "tp_dist": float(fs["tp_dist"].group(1)),
                        "sl_dist": float(fs["sl_dist"].group(1)),
                        "strategy": fs["strategy"].group(1),
                        "status": fs["status"].group(1),
                    })
            if m := _RE_KALSHI_VIEW.search(line):
                stats["kalshi_views"].append({
                    "ts": ts.isoformat(), "decision": m.group(1),
                    "entry_probability": float(m.group(2)),
                    "probe_probability": float(m.group(3)),
                    "support_score": float(m.group(4)),
                    "threshold": float(m.group(5)),
                })
            if m := _RE_KALSHI_BLOCK.search(line):
                stats["kalshi_blocks"].append({
                    "ts": ts.isoformat(), "strategy": m.group(1)
                })
            if m := _RE_CM_OVERRIDE.search(line):
                stats["cm_overrides"].append({
                    "ts": ts.isoformat(), "side": m.group(1), "reason": m.group(2).strip(),
                })
            if m := _RE_CM_GATE_V2.search(line):
                stats["cm_gate_v2"].append({
                    "ts": ts.isoformat(), "side": m.group(1),
                    "p_direction": float(m.group(2)),
                    "would_override": m.group(3) == "True",
                    "active": m.group(4) == "True",
                })
            if m := _RE_CM_GATE_ML.search(line):
                stats["cm_gate_ml"].append({
                    "ts": ts.isoformat(), "side": m.group(1),
                    "p_win": float(m.group(2)),
                    "would_override": m.group(3) == "True",
                    "active": m.group(4) == "True",
                })
            if m := _RE_SHADOW_RL.search(line):
                stats["shadow_rl"].append({
                    "ts": ts.isoformat(), "strategy": m.group(1), "side": m.group(2),
                    "bar": int(m.group(3)), "pnl_pts": float(m.group(4)),
                    "action": m.group(5),
                })
            if m := _RE_RL_SLMOVE.search(line):
                stats["rl_live"].append({
                    "ts": ts.isoformat(), "type": "sl_move",
                    "side": m.group(1), "new_sl": float(m.group(2)),
                    "action": m.group(3),
                })
            elif m := _RE_RL_LIVE.search(line):
                stats["rl_live"].append({
                    "ts": ts.isoformat(), "type": "status", "status": m.group(1),
                    "action": m.group(2) or "",
                })
            if m := _RE_SHADOW_LFO.search(line):
                stats["shadow_lfo"].append({
                    "ts": ts.isoformat(), "rule": m.group(1), "ml": m.group(2),
                    "p_wait": float(m.group(3)), "threshold": float(m.group(4)),
                    "strategy": m.group(5), "side": m.group(6),
                })
            if m := _RE_SHADOW_GATE.search(line):
                stats["shadow_gate"].append({
                    "ts": ts.isoformat(), "family": m.group(1), "side": m.group(2),
                    "p_big_loss": float(m.group(3)), "threshold": float(m.group(4)),
                    "would_veto": m.group(5) == "True",
                })
            if m := _RE_TRADE_PLACED.search(line):
                stats["trades_placed"].append({
                    "ts": ts.isoformat(), "order_id": m.group(1),
                    "strategy": m.group(2), "side": m.group(3),
                    "entry": float(m.group(4)), "tp": float(m.group(5)),
                    "sl": float(m.group(6)),
                })
            if m := _RE_TRADE_CLOSED.search(line):
                stats["trades_closed"].append({
                    "ts": ts.isoformat(), "strategy": m.group(1), "side": m.group(2),
                    "entry": float(m.group(3)), "exit": float(m.group(4)),
                    "pnl_pts": float(m.group(5)), "pnl_dollars": float(m.group(6)),
                    "source": m.group(7),
                })
            if m := _RE_HEARTBEAT.search(line):
                stats["heartbeats"].append({"ts": ts.isoformat(), "price": float(m.group(1))})
    return stats


def compute_session_summary(stats: dict) -> dict:
    """Aggregate per-day rollup used in the markdown header."""
    tc = stats["trades_closed"]
    pnl = sum(t["pnl_dollars"] for t in tc)
    wins = sum(1 for t in tc if t["pnl_dollars"] > 0)
    losses = sum(1 for t in tc if t["pnl_dollars"] < 0)
    scratches = sum(1 for t in tc if t["pnl_dollars"] == 0)
    # Daily DD: running cum PnL from trades, max peak-to-trough
    cum, peak, dd = 0.0, 0.0, 0.0
    for t in sorted(tc, key=lambda x: x["ts"]):
        cum += t["pnl_dollars"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    # Block-rate by source
    n_sig = len(stats["signals_fired"])
    n_sig_blocked = sum(1 for s in stats["signals_fired"] if s["status"] == "BLOCKED")
    n_kalshi_block = len(stats["kalshi_blocks"])
    # Counterfactual block outcomes (requires forward data — stub unless
    # we have it; here we just count "still open at log end")
    cm_fire = sum(1 for c in stats["cm_overrides"])
    cm_v2_would = sum(1 for c in stats["cm_gate_v2"] if c.get("would_override"))
    return {
        "total_pnl": round(pnl, 2),
        "n_trades": len(tc),
        "n_wins": wins, "n_losses": losses, "n_scratches": scratches,
        "win_rate": round(wins / len(tc) * 100, 1) if tc else 0.0,
        "max_drawdown": round(dd, 2),
        "n_signals_fired": n_sig,
        "n_signals_blocked_strategy": n_sig_blocked,
        "n_kalshi_blocks": n_kalshi_block,
        "n_cm_overrides_fired": cm_fire,
        "n_cm_gate_v2_would_override": cm_v2_would,
    }


def compute_block_rate_breakdown(stats: dict) -> dict:
    """Per-layer block counts."""
    breakdown = Counter()
    for s in stats["signals_fired"]:
        if s["status"] == "BLOCKED":
            breakdown[f"{s['strategy']}_self_veto"] += 1
    breakdown["kalshi_overlay"] = len(stats["kalshi_blocks"])
    for sg in stats["shadow_gate"]:
        if sg["would_veto"]:
            breakdown[f"filter_g_{sg['family']}"] += 1
    return dict(breakdown)


def flag_patterns(stats: dict) -> list[str]:
    """Auto-detect suspicious patterns in today's activity."""
    flags = []
    tc = stats["trades_closed"]
    # Consecutive losses
    consec_loss = 0; max_consec = 0
    for t in sorted(tc, key=lambda x: x["ts"]):
        if t["pnl_dollars"] < 0:
            consec_loss += 1; max_consec = max(max_consec, consec_loss)
        else:
            consec_loss = 0
    if max_consec >= 4:
        flags.append(f"⚠ {max_consec} consecutive losses in one session — chop/overtrading signature")
    # Tiny-loss clustering (≤ $25 losses in quick succession = executor-forcing?)
    tiny = [t for t in tc if -25 < t["pnl_dollars"] < 0]
    if len(tiny) >= 3:
        flags.append(f"⚠ {len(tiny)} tiny losses (< $25 each) — possible SL-tightening misfire")
    # Kalshi block rate
    n_sig = len(stats["signals_fired"])
    n_kb = len(stats["kalshi_blocks"])
    if n_sig >= 50 and n_kb / max(1, n_sig) >= 0.60:
        flags.append(
            f"⚠ Kalshi blocked {n_kb}/{n_sig} = {n_kb/n_sig*100:.0f}% of signals — "
            f"either Kalshi is cautious (chop/lunch) or threshold is mis-tuned")
    # CM gate v2 fired zero but many blocks
    cm_fires = sum(1 for c in stats["cm_gate_v2"] if c.get("would_override"))
    if n_kb >= 30 and cm_fires == 0:
        flags.append(
            f"⚠ CM gate v2 fired 0 overrides across {n_kb} blocks — "
            f"threshold {0.60:.2f} may be too conservative for this tape")
    # RL status=closed (fatal SL move on underwater trade)
    rl_closed = sum(1 for r in stats["rl_live"] if r.get("status") == "closed")
    if rl_closed >= 2:
        flags.append(
            f"⚠ {rl_closed} RL 'status=closed' events — SL-move-on-underwater-trade issue "
            f"(should be zero after c5caf83 guard)")
    return flags


def render_markdown(stats: dict, summary: dict, breakdown: dict, flags: list[str]) -> str:
    date_s = stats["date"]
    lines = [
        f"# Julie daily journal — {date_s}",
        "",
        "## Session summary",
        "",
        f"| metric | value |",
        f"|---|---:|",
        f"| Realized PnL | **${summary['total_pnl']:+,.2f}** |",
        f"| Trades closed | {summary['n_trades']} |",
        f"| WR | {summary['win_rate']:.1f}% ({summary['n_wins']}W / {summary['n_losses']}L) |",
        f"| Max session DD | ${summary['max_drawdown']:,.2f} |",
        f"| Signals fired | {summary['n_signals_fired']} |",
        f"| Strategy self-blocks | {summary['n_signals_blocked_strategy']} |",
        f"| Kalshi overlay blocks | {summary['n_kalshi_blocks']} |",
        f"| CM overrides fired (hand+v2) | {summary['n_cm_overrides_fired']} |",
        f"| CM gate v2 would-override events | {summary['n_cm_gate_v2_would_override']} |",
        "",
        "## Block-rate breakdown by layer",
        "",
    ]
    if breakdown:
        lines.append("| layer | blocks |")
        lines.append("|---|---:|")
        for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("(no blocks recorded)")
    lines.extend(["", "## Trades", ""])
    tc = stats["trades_closed"]
    if tc:
        lines.append("| time | strat | side | entry | exit | PnL | source |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for t in sorted(tc, key=lambda x: x["ts"])[:40]:
            ts = t["ts"].split("T")[1][:5]
            lines.append(
                f"| {ts} | {t['strategy'][:14]} | {t['side']:5s} | "
                f"{t['entry']:.2f} | {t['exit']:.2f} | "
                f"${t['pnl_dollars']:+.2f} | {t['source']} |"
            )
    else:
        lines.append("(no closed trades)")
    lines.extend(["", "## Auto-flagged patterns", ""])
    if flags:
        for fl in flags:
            lines.append(f"- {fl}")
    else:
        lines.append("✓ no patterns flagged — session looks clean")
    # ML-layer activity footer
    lines.extend(["", "## ML-layer activity", ""])
    lines.append(f"- Shadow-LFO decisions logged: {len(stats['shadow_lfo'])}")
    lines.append(f"- Shadow-Gate evaluations logged: {len(stats['shadow_gate'])}")
    lines.append(f"- RL per-bar decisions: {len(stats['shadow_rl'])}")
    lines.append(f"- RL live status lines: {len(stats['rl_live'])}")
    lines.append(f"- CM gate v2 log lines: {len(stats['cm_gate_v2'])}")
    lines.append(f"- CM gate v1 log lines: {len(stats['cm_gate_ml'])}")
    return "\n".join(lines) + "\n"


def write_journal(target_date: date) -> tuple[Path, Path]:
    """Write journal .md and .json for target_date. Returns both paths."""
    stats = parse_log_for_date(LOG_PATH, target_date)
    summary = compute_session_summary(stats)
    breakdown = compute_block_rate_breakdown(stats)
    flags = flag_patterns(stats)
    md = render_markdown(stats, summary, breakdown, flags)
    structured = {
        "date": target_date.isoformat(),
        "summary": summary,
        "breakdown_by_layer": breakdown,
        "pattern_flags": flags,
        "raw_counts": {k: len(v) for k, v in stats.items() if isinstance(v, list)},
        # keep trades for downstream analyzer
        "trades_closed": stats["trades_closed"],
        "signals_fired": stats["signals_fired"][:500],  # cap size
        "kalshi_views": stats["kalshi_views"][:500],
        "cm_gate_v2": stats["cm_gate_v2"][:500],
    }
    out_md = JOURNALS_DIR / f"{target_date.isoformat()}.md"
    out_json = JOURNALS_DIR / f"{target_date.isoformat()}.json"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(structured, indent=2, default=str), encoding="utf-8")
    return out_md, out_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    if args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        d = date.today()
    md, js = write_journal(d)
    print(f"[journal] wrote {md}")
    print(f"[journal] wrote {js}")


if __name__ == "__main__":
    main()

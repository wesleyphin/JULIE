#!/usr/bin/env python3
"""Priority-effect backtest for the Triathlon Engine.

Isolates the priority_delta mechanism from the (already-neutralized)
size multiplier. Asks: given the rescue-queue's per-bar candidate list,
does medal-based priority reordering change which trade gets picked —
and if so, does it improve PnL?

Pipeline
--------
1. Parse the April 2026 replay log (`backtest_reports/full_live_replay/
   2026_04_ml_stacks/topstep_live_bot.log`) for every bar that emitted
   ≥ 2 GENUINELY COMPETING candidates. "Genuinely competing" = different
   strategy OR different side on the same minute-bar. Duplicate
   checkpoints of the same signal (strategy_signal + level_fill_queued)
   are collapsed into one candidate.
2. Freeze medals from pre-April 2026 data (same cutoff used by the
   size-effect backtest).
3. For each competing bar, run two sort orderings:
     a) baseline — Julie's current `_live_signal_sort_key`:
        (priority, -confidence, strategy_label, sub_strategy, side)
     b) priority-active — same tuple but with
        adjusted_priority = base_priority + medal_priority_delta_flipped
        where "flipped" = -priority_delta because in Julie's sort, LOWER
        priority sorts first (priority=0 is SENTIMENT, best; priority=2
        is STANDARD, worst). So a gold medal with priority_delta=+1 in
        the Triathlon semantics should SUBTRACT 1 from the sort priority
        to actually promote it.
   NOTE: this is the "as-intended" priority behavior. The live
   `_live_signal_sort_key` does not currently consult the
   `triathlon_priority_delta` signal-dict key at all — priority deltas
   are effectively dead code in the current live binary. This backtest
   scores the INTENDED wiring; the verdict at the end addresses both
   (keep the dead code, wire it up, or remove it entirely).
4. Match realized outcomes:
     - Baseline winner → realized via `closed_trades.json`
       (match by entry_price + side + minute-bucket).
     - If priority-active picks a different winner:
         * If that winner also appears in closed_trades (different bar),
           use its realized outcome.
         * Else counterfactual: forward-walk through live_prices.parquet
           until hitting TP or SL.
5. Report PnL / WR / MaxDD / # trades for both scenarios, and for every
   bar where the decision actually differed, show the two candidates
   and the outcome delta.

Output: `backtest_reports/triathlon_priority_oos_results.json`
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
REPLAY_LOG = ROOT / "backtest_reports/full_live_replay/2026_04_ml_stacks/topstep_live_bot.log"
CLOSED_TRADES = ROOT / "backtest_reports/full_live_replay/2026_04_ml_stacks/closed_trades.json"
LEDGER = ROOT / "ai_loop_data/triathlon/ledger.db"
OUT = ROOT / "backtest_reports/triathlon_priority_oos_results.json"

MES_POINT_VALUE = 5.0
LOOKAHEAD_BARS = 60

# Mirror the live Triathlon constants
MIN_SAMPLES = 20
GOLD_PCTILE = 0.20
SILVER_PCTILE = 0.50
BRONZE_PCTILE = 0.80
PROBATION_PCTILE = 0.80

# Intended Triathlon priority effects (what we're testing)
MEDAL_PRIORITY_DELTA = {
    "gold":      +1,
    "silver":     0,
    "bronze":    -1,
    "probation": -2,
    "unrated":    0,
}

# Time-bucket boundaries (match triathlon/__init__.py)
TIME_BUCKETS = [
    ("pre_open",    4.0,  9.5),
    ("morning",     9.5, 12.0),
    ("lunch",      12.0, 14.0),
    ("afternoon",  14.0, 16.0),
    ("post_close", 16.0, 17.0),
    ("overnight",  17.0, 28.0),
]

def time_bucket_of(h: float) -> str:
    if h < 4.0: h += 24.0
    for name, lo, hi in TIME_BUCKETS:
        if lo <= h < hi:
            return name
    return "overnight"


# ─── log parsing ──────────────────────────────────────────────
# STRATEGY_SIGNAL line:
# 2026-04-22 14:57:30,494 [INFO] [2026-04-22 14:57:30.494] [STRATEGY_SIGNAL] 📊 DynamicEngine3 generated LONG signal |
#   strategy=DynamicEngine3 | side=LONG | price=6617.50 | tp_dist=12.50 | sl_dist=10.00 | status=CANDIDATE | priority=FAST
BAR_RE = re.compile(r"\[INFO\] Bar:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET\s+\|\s+Price:\s+([\d.]+)")
SIGNAL_RE = re.compile(
    r"\[STRATEGY_SIGNAL\].*strategy=(?P<strategy>\w+).*side=(?P<side>\w+).*"
    r"price=(?P<price>[\d.]+).*tp_dist=(?P<tp>[\d.]+).*sl_dist=(?P<sl>[\d.]+).*"
    r"status=(?P<status>\w+).*priority=(?P<priority>\w+)"
)
SUB_RE = re.compile(r"combo_key=(?P<sub>\S+)")

PRIORITY_LABEL_TO_INT = {
    "SENTIMENT": 0,
    "FAST": 1,
    "STANDARD": 2,
    "LOOSE": 3,  # safety; live code treats >1 as STANDARD
}


@dataclass
class Candidate:
    bar_ts: datetime
    strategy: str
    side: str
    price: float
    tp_dist: float
    sl_dist: float
    status: str
    priority_label: str
    priority_int: int
    sub_strategy: Optional[str]
    # Populated after cell lookup
    regime: Optional[str] = None
    time_bucket: Optional[str] = None
    medal: Optional[str] = None
    medal_priority_delta: int = 0

    def cell_key(self) -> str:
        return f"{self.strategy}|{self.regime}|{self.time_bucket}"


def parse_replay_log(log_path: Path) -> dict[datetime, list[Candidate]]:
    """Walk the replay log and collect per-bar candidates.
    Deduplicates same-(strategy, side, price) entries at the same bar."""
    per_bar: dict[datetime, list[Candidate]] = defaultdict(list)
    seen_keys: dict[datetime, set[tuple]] = defaultdict(set)
    current_bar: Optional[datetime] = None
    with log_path.open() as f:
        for line in f:
            mb = BAR_RE.search(line)
            if mb:
                try:
                    current_bar = datetime.strptime(mb.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    current_bar = None
                continue
            if current_bar is None:
                continue
            m = SIGNAL_RE.search(line)
            if not m or m.group("status") != "CANDIDATE":
                continue
            # Dedup: same (strategy, side, price) at the same bar = same signal
            dedup_key = (m.group("strategy"), m.group("side"), m.group("price"))
            if dedup_key in seen_keys[current_bar]:
                continue
            seen_keys[current_bar].add(dedup_key)
            msub = SUB_RE.search(line)
            cand = Candidate(
                bar_ts=current_bar,
                strategy=m.group("strategy"),
                side=m.group("side"),
                price=float(m.group("price")),
                tp_dist=float(m.group("tp")),
                sl_dist=float(m.group("sl")),
                status=m.group("status"),
                priority_label=m.group("priority"),
                priority_int=PRIORITY_LABEL_TO_INT.get(m.group("priority"), 2),
                sub_strategy=msub.group("sub") if msub else None,
            )
            per_bar[current_bar].append(cand)
    return per_bar


# ─── regime classifier replay (lightweight) ────────────────────
def build_regime_lookup(price_df: pd.DataFrame, timestamps: list[datetime]) -> dict[datetime, str]:
    """Build {ts -> regime} by running the rolling 120-bar classifier
    on prices up to each ts. Mirrors regime_classifier._classify()."""
    import math
    WINDOW = 120
    EFF_LOW, EFF_HIGH = 0.05, 0.12
    DEAD_TAPE_VOL_BP = 1.5
    if price_df is None or price_df.empty:
        return {ts: "warmup" for ts in timestamps}
    closes = price_df["price"].astype(float).tolist()
    idx = price_df.index
    import bisect
    out: dict[datetime, str] = {}
    for ts in set(timestamps):
        ts_ny = pd.Timestamp(ts)
        if ts_ny.tzinfo is None:
            ts_ny = ts_ny.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        else:
            ts_ny = ts_ny.tz_convert("America/New_York")
        pos = idx.searchsorted(ts_ny, side="right") - 1
        if pos < WINDOW:
            out[ts] = "warmup"
            continue
        window = closes[pos + 1 - WINDOW : pos + 1]
        rets = []
        for i in range(1, len(window)):
            p0 = window[i - 1]
            if p0 > 0:
                rets.append((window[i] - p0) / p0)
        if not rets:
            out[ts] = "neutral"; continue
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        vol_bp = math.sqrt(var) * 10_000.0
        abs_sum = sum(abs(r) for r in rets)
        eff = abs(sum(rets)) / abs_sum if abs_sum > 0 else 0.0
        if vol_bp < DEAD_TAPE_VOL_BP:
            out[ts] = "dead_tape"
        elif vol_bp > 3.5 and eff < EFF_LOW:
            out[ts] = "whipsaw"
        elif eff > EFF_HIGH:
            out[ts] = "calm_trend"
        else:
            out[ts] = "neutral"
    return out


# ─── medals from pre-April data ───────────────────────────────
def load_pre_april_medals() -> dict[str, str]:
    """Load trades before 2026-04-01 from the ledger, score cells, return
    {cell_key: medal}."""
    conn = sqlite3.connect(str(LEDGER))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        """
        SELECT s.strategy, s.regime, s.time_bucket, s.size,
               o.pnl_dollars, o.bars_held
        FROM signals s JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.source_tag IN ('seed_2025','seed_2026')
          AND o.counterfactual = 0 AND s.status = 'fired'
          AND s.ts < '2026-04-01'
        """
    ))
    conn.close()
    by_cell: dict[str, list] = defaultdict(list)
    for r in rows:
        ck = f"{r['strategy']}|{r['regime']}|{r['time_bucket']}"
        by_cell[ck].append(r)

    scores = {}
    for ck, items in by_cell.items():
        if len(items) < MIN_SAMPLES:
            scores[ck] = {"rated": False}
            continue
        wins = [t for t in items if t["pnl_dollars"] > 0]
        purity = len(wins) / len(items)
        cash = sum(t["pnl_dollars"] for t in items) / max(1, sum(max(1, t["size"] or 1) for t in items))
        win_bars = [t["bars_held"] for t in wins if t["bars_held"] and t["bars_held"] > 0]
        velocity = 1.0 / statistics.median(win_bars) if win_bars else None
        if velocity is None:
            scores[ck] = {"rated": False}
            continue
        scores[ck] = {"rated": True, "purity": purity, "cash": cash, "velocity": velocity}

    rated = [ck for ck, s in scores.items() if s["rated"]]
    n_rated = len(rated)
    def rank_in(key):
        srt = sorted(rated, key=lambda ck: (-scores[ck][key], ck))
        return {ck: i + 1 for i, ck in enumerate(srt)}
    pr = rank_in("purity")
    cr = rank_in("cash")
    vr = rank_in("velocity")

    medals = {}
    for ck, s in scores.items():
        if not s["rated"]:
            medals[ck] = "unrated"; continue
        pp = pr[ck] / n_rated
        pc = cr[ck] / n_rated
        pv = vr[ck] / n_rated
        best = min(pp, pc, pv)
        if best >= PROBATION_PCTILE:
            medals[ck] = "probation"
        elif best <= GOLD_PCTILE:
            medals[ck] = "gold"
        elif best <= SILVER_PCTILE:
            medals[ck] = "silver"
        elif best <= BRONZE_PCTILE:
            medals[ck] = "bronze"
        else:
            medals[ck] = "silver"
    return medals


# ─── sort keys ────────────────────────────────────────────────
def baseline_sort_key(c: Candidate) -> tuple:
    """Mirrors julie001._live_signal_sort_key (confidence stubbed to 0
    since the log doesn't surface it; the cascade is then
    alphabetical on strategy/sub/side — stable and deterministic)."""
    return (c.priority_int, 0, c.strategy, c.sub_strategy or "", c.side)


def priority_active_sort_key(c: Candidate) -> tuple:
    """As-intended Triathlon priority: a gold medal (priority_delta=+1
    in the Triathlon spec) SUBTRACTS from the sort priority to promote
    it. Probation (delta=-2) ADDS to the sort priority to demote it."""
    adjusted_priority = c.priority_int - c.medal_priority_delta
    return (adjusted_priority, 0, c.strategy, c.sub_strategy or "", c.side)


# ─── counterfactual for priority's new winner ─────────────────
def forward_walk(
    df: pd.DataFrame,
    side: str, entry_price: float, tp_dist: float, sl_dist: float,
    entry_ts: datetime, lookahead: int = LOOKAHEAD_BARS,
) -> Optional[tuple[float, int, str]]:
    ts = pd.Timestamp(entry_ts).tz_localize("America/New_York",
                        ambiguous="NaT", nonexistent="shift_forward")
    sub = df.loc[df.index > ts].head(lookahead)
    if sub.empty:
        return None
    px = sub["price"].astype(float)
    side_up = side.upper()
    if side_up == "LONG":
        sl_p, tp_p = entry_price - sl_dist, entry_price + tp_dist
    else:
        sl_p, tp_p = entry_price + sl_dist, entry_price - tp_dist
    for i, (_, price) in enumerate(px.items(), start=1):
        if side_up == "LONG":
            if price <= sl_p: return (-sl_dist, i, "cf_sl")
            if price >= tp_p: return (tp_dist, i, "cf_tp")
        else:
            if price >= sl_p: return (-sl_dist, i, "cf_sl")
            if price <= tp_p: return (tp_dist, i, "cf_tp")
    last = float(px.iloc[-1])
    pnl = (last - entry_price) if side_up == "LONG" else (entry_price - last)
    return (pnl, len(px), "cf_expired")


# ─── baseline realized outcomes from closed_trades.json ───────
def load_closed_trades() -> list[dict]:
    return json.loads(CLOSED_TRADES.read_text())


def match_closed_trade(closed: list[dict], bar_ts: datetime, side: str, strategy: str, entry_price: float) -> Optional[dict]:
    """Find the closed_trade whose entry_time is within a few minutes
    of the bar, on the same side + strategy + entry price (±0.75pt)."""
    for t in closed:
        try:
            t_ts = datetime.fromisoformat(t["entry_time"]).replace(tzinfo=None)
        except Exception:
            continue
        if abs((t_ts - bar_ts).total_seconds()) > 180:
            continue
        if str(t.get("side", "")).upper() != side.upper():
            continue
        if str(t.get("strategy", "")) != strategy:
            continue
        try:
            if abs(float(t.get("entry_price", 0)) - entry_price) > 0.75:
                continue
        except Exception:
            continue
        return t
    return None


# ─── main ─────────────────────────────────────────────────────
def main():
    print(f"[priority-oos] parsing replay log: {REPLAY_LOG}")
    per_bar = parse_replay_log(REPLAY_LOG)
    print(f"  bars seen: {len(per_bar):,}")

    # Keep only truly-competing bars: ≥2 DIFFERENT (strategy, side) candidates
    competing_bars: dict[datetime, list[Candidate]] = {}
    for bar, cands in per_bar.items():
        unique_keys = set((c.strategy, c.side) for c in cands)
        if len(unique_keys) >= 2:
            competing_bars[bar] = cands
    print(f"  bars with genuinely-competing candidates (different strategy OR side): {len(competing_bars):,}")

    if not competing_bars:
        print("  NO COMPETING BARS — priority effects are vacuous on this tape")
        print("  verdict: priority deltas have zero PnL impact in this sample")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"competing_bars": 0, "verdict": "no_effect"}, indent=2))
        return

    # Load medals + regime lookup
    print(f"\n[priority-oos] loading pre-April 2026 medals from ledger...")
    medals = load_pre_april_medals()
    print(f"  rated cells: {sum(1 for m in medals.values() if m != 'unrated')} / {len(medals)} total")

    print(f"\n[priority-oos] loading live_prices parquet for regime classification...")
    px_path = ROOT / "ai_loop_data/live_prices.parquet"
    price_df = None
    if px_path.exists():
        price_df = pd.read_parquet(px_path)
        if price_df.index.tz is None:
            price_df.index = price_df.index.tz_localize("America/New_York",
                ambiguous="NaT", nonexistent="shift_forward")
        else:
            price_df.index = price_df.index.tz_convert("America/New_York")

    timestamps = list(competing_bars.keys())
    regime_map = build_regime_lookup(price_df, timestamps)

    # Annotate every competing candidate with regime / bucket / medal
    for bar, cands in competing_bars.items():
        regime = regime_map.get(bar, "neutral")
        h = bar.hour + bar.minute / 60.0
        tb = time_bucket_of(h)
        for c in cands:
            c.regime = regime
            c.time_bucket = tb
            c.medal = medals.get(c.cell_key(), "unrated")
            c.medal_priority_delta = MEDAL_PRIORITY_DELTA[c.medal]

    # For each competing bar, sort under both orderings and compare
    closed = load_closed_trades()
    differing_decisions: list[dict] = []
    baseline_pnls: list[float] = []
    priority_pnls: list[float] = []

    for bar, cands in sorted(competing_bars.items()):
        base_sorted = sorted(cands, key=baseline_sort_key)
        prio_sorted = sorted(cands, key=priority_active_sort_key)
        baseline_winner = base_sorted[0]
        priority_winner = prio_sorted[0]

        if (baseline_winner.strategy, baseline_winner.side, baseline_winner.price) == \
           (priority_winner.strategy, priority_winner.side, priority_winner.price):
            # No change; skip to keep the comparison focused on "where
            # priority actually shifted the decision."
            continue

        # Realize baseline winner via closed_trades
        base_trade = match_closed_trade(closed, bar, baseline_winner.side,
                                         baseline_winner.strategy, baseline_winner.price)
        base_pnl = float(base_trade.get("pnl_dollars", 0.0) or 0.0) if base_trade else 0.0
        # The priority-active winner: try to find it in closed_trades too
        # (if it was picked up on a different bar within 5 min), else CF.
        prio_trade = match_closed_trade(closed, bar, priority_winner.side,
                                         priority_winner.strategy, priority_winner.price)
        if prio_trade is not None:
            prio_pnl = float(prio_trade.get("pnl_dollars", 0.0) or 0.0)
            prio_source = f"realized:{prio_trade.get('source','?')}"
        else:
            result = forward_walk(price_df, priority_winner.side, priority_winner.price,
                                   priority_winner.tp_dist, priority_winner.sl_dist, bar)
            if result is None:
                prio_pnl = 0.0
                prio_source = "cf_no_data"
            else:
                pnl_pts, bars_held, exit = result
                prio_pnl = pnl_pts * MES_POINT_VALUE * 1  # size=1 convention
                prio_source = f"{exit}:{bars_held}bars"

        differing_decisions.append({
            "bar_ts": bar.isoformat(),
            "baseline_winner": {
                "strategy": baseline_winner.strategy, "side": baseline_winner.side,
                "price": baseline_winner.price, "medal": baseline_winner.medal,
                "priority": baseline_winner.priority_label,
                "pnl_dollars": round(base_pnl, 2),
                "source": (base_trade.get("source") if base_trade else "no_match"),
            },
            "priority_winner": {
                "strategy": priority_winner.strategy, "side": priority_winner.side,
                "price": priority_winner.price, "medal": priority_winner.medal,
                "priority": priority_winner.priority_label,
                "pnl_dollars": round(prio_pnl, 2),
                "source": prio_source,
            },
            "delta_pnl": round(prio_pnl - base_pnl, 2),
        })
        baseline_pnls.append(base_pnl)
        priority_pnls.append(prio_pnl)

    # Aggregate
    def stats(pnls):
        if not pnls: return {"n": 0, "pnl": 0, "wr": 0, "max_dd": 0, "avg": 0}
        cum = 0; peak = 0; dd = 0
        for p in pnls:
            cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
        wins = [p for p in pnls if p > 0]
        return {
            "n": len(pnls), "pnl": round(cum, 2),
            "wr": round(len(wins) / len(pnls) * 100, 2),
            "max_dd": round(dd, 2),
            "avg": round(cum / len(pnls), 2),
            "wins": len(wins), "losses": len(pnls) - len(wins),
        }

    base_stats = stats(baseline_pnls)
    prio_stats = stats(priority_pnls)

    print(f"\n═══ PRIORITY EFFECT ON APRIL 2026 HOLDOUT ═══\n")
    print(f"  total competing bars (different strategy OR side):  {len(competing_bars):,}")
    print(f"  bars where priority sort changed the winner:        {len(differing_decisions):,}")

    if not differing_decisions:
        print(f"\n  priority effect = ZERO trades shifted.")
        print(f"  verdict: neutralize priority deltas — they never fire on this tape.")
    else:
        print(f"\n  baseline (tied-break by native priority only):")
        print(f"    trades={base_stats['n']}  PnL=${base_stats['pnl']:+,.2f}  "
              f"WR={base_stats['wr']}%  MaxDD=${base_stats['max_dd']:,.0f}  "
              f"avg/trade=${base_stats['avg']:+,.2f}")
        print(f"  priority-active (medal delta applied):")
        print(f"    trades={prio_stats['n']}  PnL=${prio_stats['pnl']:+,.2f}  "
              f"WR={prio_stats['wr']}%  MaxDD=${prio_stats['max_dd']:,.0f}  "
              f"avg/trade=${prio_stats['avg']:+,.2f}")
        print(f"\n  DELTA:")
        print(f"    PnL    ${prio_stats['pnl'] - base_stats['pnl']:+,.2f}")
        print(f"    WR     {prio_stats['wr'] - base_stats['wr']:+.2f}pp")
        print(f"    MaxDD  ${prio_stats['max_dd'] - base_stats['max_dd']:+,.0f}")

        print(f"\n  per-decision detail:")
        for d in differing_decisions:
            bw, pw = d["baseline_winner"], d["priority_winner"]
            print(f"    {d['bar_ts']}")
            print(f"      baseline:  {bw['strategy']:<16} {bw['side']:<5} "
                  f"@{bw['price']:.2f}  medal={bw['medal']}  pnl=${bw['pnl_dollars']:+.2f}")
            print(f"      priority:  {pw['strategy']:<16} {pw['side']:<5} "
                  f"@{pw['price']:.2f}  medal={pw['medal']}  pnl=${pw['pnl_dollars']:+.2f}")
            print(f"      Δ=${d['delta_pnl']:+.2f}  ({pw['source']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "competing_bars": len(competing_bars),
        "differing_decisions": len(differing_decisions),
        "baseline_stats": base_stats,
        "priority_stats": prio_stats,
        "decisions": differing_decisions,
    }, indent=2))
    print(f"\n[write] {OUT}")


if __name__ == "__main__":
    main()

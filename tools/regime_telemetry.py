#!/usr/bin/env python3
"""Regime telemetry foundation — read-only data substrate for adaptive work.

This is Phase 1 of the multi-component adaptive risk-manager plan. It does
NOT modify any bot behavior — it only READS the bot log + journals + parquet
and produces:

  1. Per-trade enriched record:
       (timestamp, strategy, sub_strategy, side, regime_at_fire, vol_bp,
        designed_tp, designed_sl, fired_tp, fired_sl, was_clipped,
        actual_pnl, mfe_at_horizon, mae_at_horizon)

  2. Rolling per-(strategy, regime) realized stats:
       n_trades, win_rate, avg_pnl, mfe_p50, mfe_p75, mae_p50, mae_p75

  3. Counterfactual at multiple bracket variants:
       designed brackets, dead-tape brackets, half-of-designed, etc.

  4. JSON + markdown report at artifacts/regime_telemetry/.

Output is the substrate that future adaptive components (vol-percentile
threshold, per-side V18 threshold, bracket auto-fitter, strategy circuits)
will read from. Without this data, no adaptive system can self-calibrate.

Zero side effects on the live bot. Safe to run anytime.

Usage:
    python3 tools/regime_telemetry.py
    python3 tools/regime_telemetry.py --window-days 30
    python3 tools/regime_telemetry.py --strategy DynamicEngine3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "topstep_live_bot.log"
PARQUET = ROOT / "es_master_outrights-2.parquet"
OUT_DIR = ROOT / "artifacts" / "regime_telemetry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOLLAR_PER_PT_MES = 5.0
COMMISSION = 7.50
HORIZON_MIN_DEFAULT = 30
DEAD_TAPE_VOL_BP_RULE = 1.5  # current bot rule
DEAD_TAPE_TP_PT = 3.0
DEAD_TAPE_SL_PT = 5.0


# ---------------------------------------------------------------------------
# Log parsers
# ---------------------------------------------------------------------------
TRADE_CLOSE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"Trade closed.*?: (?P<strategy>\w+) (?P<side>\w+) \| "
    r"Entry: (?P<entry>[\d.]+) \| Exit: (?P<exit>[\d.]+) \| "
    r"PnL: (?P<pnl_pts>[-\d.]+) pts \(\$(?P<pnl_dollars>[-\d.]+)\).*"
    r"order_id=(?P<order_id>\S+).*entry_order_id=(?P<entry_order_id>\S+)"
    r" \| size=(?P<size>\d+)"
)

TRADE_PLACED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[TRADE_PLACED\] .*strategy=(?P<strategy>\S+) \| side=(?P<side>\w+) \| "
    r"entry=(?P<entry>[\d.]+) \| tp=(?P<tp>[\d.]+) \| sl=(?P<sl>[\d.]+)"
)

DEAD_TAPE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"Dead-tape override: (?P<strategy>\w+) (?P<side>\w+) \| "
    r"tp->(?P<tp_post>[\d.]+) sl->(?P<sl_post>[\d.]+) size->(?P<size_post>\d+) "
    r"BE=(?P<be>\w+) \| vol=(?P<vol_bp>[\d.]+)bp"
)

V18_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[V18_DE3 (?P<dec>BLOCKED|KEEP)\] strat=(?P<strategy>\w+) side=(?P<side>\w+) "
    r"v18_(?:block|keep)_proba=(?P<proba>[\d.]+)"
)

REGIME_TRANSITION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"Regime transition: (?P<from>\w+) -> (?P<to>\w+) \| "
    r"vol=(?P<vol>[\d.]+)bp eff=(?P<eff>[\d.]+)"
)


def parse_log(log_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Parse the bot log into structured event lists."""
    if not log_path.exists():
        print(f"[regime_telemetry] WARN: log not found at {log_path}")
        return {"closes": [], "placements": [], "dead_tape": [], "v18": [], "regime_transitions": []}

    closes, placements, dead_tape, v18, transitions = [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            for patt, sink in (
                (TRADE_CLOSE_RE, closes),
                (TRADE_PLACED_RE, placements),
                (DEAD_TAPE_RE, dead_tape),
                (V18_RE, v18),
                (REGIME_TRANSITION_RE, transitions),
            ):
                m = patt.search(line)
                if not m:
                    continue
                d = m.groupdict()
                ts = pd.Timestamp(d["ts"]).tz_localize("America/Los_Angeles")
                d["ts_pdt"] = ts
                d["ts_et"] = ts.tz_convert("US/Eastern")
                # Coerce numeric fields
                for k in ("entry", "exit", "pnl_pts", "pnl_dollars", "tp", "sl",
                          "tp_post", "sl_post", "size_post", "vol_bp", "proba", "vol", "eff"):
                    if k in d and d[k] is not None:
                        try: d[k] = float(d[k])
                        except (TypeError, ValueError): pass
                if "size" in d and d["size"] is not None:
                    try: d["size"] = int(d["size"])
                    except (TypeError, ValueError): pass
                sink.append(d)

    return {
        "closes": closes,
        "placements": placements,
        "dead_tape": dead_tape,
        "v18": v18,
        "regime_transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Trade enrichment — match each closed trade with its placement + regime
# ---------------------------------------------------------------------------
@dataclass
class EnrichedTrade:
    ts_pdt: pd.Timestamp
    strategy: str
    side: str
    entry: float
    exit: float
    pnl_pts: float
    pnl_dollars: float
    size: int
    order_id: str
    entry_order_id: str
    fire_ts_pdt: Optional[pd.Timestamp] = None
    fired_tp_dist: Optional[float] = None
    fired_sl_dist: Optional[float] = None
    dead_tape_clipped: bool = False
    vol_bp_at_fire: Optional[float] = None
    regime_at_fire: Optional[str] = None
    v18_proba: Optional[float] = None
    v18_decision: Optional[str] = None
    mfe_pts: Optional[float] = None
    mae_pts: Optional[float] = None
    horizon_close: Optional[float] = None
    cf_pnl_designed_tp: Optional[float] = None
    cf_pnl_designed_sl: Optional[float] = None
    cf_pnl_at_125_10: Optional[float] = None  # 12.5/10 counterfactual
    cf_pnl_at_25_10: Optional[float] = None   # 25/10 counterfactual

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Stringify timestamps for JSON
        for k in ("ts_pdt", "fire_ts_pdt"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


def match_placement(close: Dict, placements: List[Dict], window_min: int = 240) -> Optional[Dict]:
    """Find the TRADE_PLACED that opened the closed trade.

    Heuristics: same strategy+side, placement before close, entry-price close,
    closest-in-time wins. Window default 4 hours.
    """
    candidates = []
    for p in placements:
        if p.get("strategy") != close.get("strategy"):
            continue
        if p.get("side") != close.get("side"):
            continue
        delta = (close["ts_pdt"] - p["ts_pdt"]).total_seconds()
        if delta <= 0 or delta > window_min * 60:
            continue
        if abs(p["entry"] - close["entry"]) > 1.0:
            continue
        candidates.append((delta, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def match_dead_tape(placement: Dict, dead_tape: List[Dict], window_sec: int = 120) -> Optional[Dict]:
    """Find a Dead-tape override event matching the placement (same strategy
    + side + close timestamp). Returns the matched event or None."""
    if placement is None:
        return None
    candidates = []
    for d in dead_tape:
        if d.get("strategy") != placement.get("strategy"):
            continue
        if d.get("side") != placement.get("side"):
            continue
        delta = abs((d["ts_pdt"] - placement["ts_pdt"]).total_seconds())
        if delta > window_sec:
            continue
        candidates.append((delta, d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def match_v18(placement: Dict, v18: List[Dict], window_sec: int = 60) -> Optional[Dict]:
    if placement is None:
        return None
    candidates = []
    for v in v18:
        if v.get("side") != placement.get("side"):
            continue
        delta = abs((v["ts_pdt"] - placement["ts_pdt"]).total_seconds())
        if delta > window_sec:
            continue
        candidates.append((delta, v))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def regime_at_ts(ts_pdt: pd.Timestamp, transitions: List[Dict]) -> Optional[str]:
    """Find the bot's regime classification at a given timestamp by scanning
    Regime transition events backwards. Returns None if no prior transition."""
    most_recent = None
    for t in transitions:
        if t["ts_pdt"] <= ts_pdt:
            if most_recent is None or t["ts_pdt"] > most_recent["ts_pdt"]:
                most_recent = t
    return most_recent["to"] if most_recent else None


# ---------------------------------------------------------------------------
# Counterfactual walk-forward through OHLC parquet
# ---------------------------------------------------------------------------
class OHLCWalker:
    """Lightweight wrapper around the ES parquet for counterfactual TP/SL
    walks. Loads once, queries by time."""

    def __init__(self, parquet_path: Path = PARQUET):
        if not parquet_path.exists():
            print(f"[regime_telemetry] WARN: parquet not found at {parquet_path}")
            self.df = pd.DataFrame()
            return
        df = pd.read_parquet(parquet_path).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        # Pre-convert index to UTC-naive for fast slicing
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        self.df = df

    def walk(
        self,
        entry_ts_pdt: pd.Timestamp,
        side: str,
        entry: float,
        tp_dist: float,
        sl_dist: float,
        horizon_min: int = HORIZON_MIN_DEFAULT,
    ) -> Tuple[str, float, float, float, float]:
        """Walk forward through OHLC. Returns:
            (outcome, exit_price, pnl_pts, mfe_pts, mae_pts)
        outcome ∈ {"WIN","LOSS","TIMEOUT"}.
        mfe_pts / mae_pts are favorable / adverse excursions from entry within
        the horizon (positive numbers).
        """
        if self.df.empty:
            return "NO_DATA", entry, 0.0, 0.0, 0.0
        entry_ts_naive = entry_ts_pdt.tz_convert("UTC").tz_localize(None)
        deadline = entry_ts_naive + timedelta(minutes=horizon_min)
        window = self.df[(self.df.index > entry_ts_naive) & (self.df.index <= deadline)]
        if window.empty:
            return "NO_DATA", entry, 0.0, 0.0, 0.0

        if side == "LONG":
            tp_level, sl_level = entry + tp_dist, entry - sl_dist
        else:
            tp_level, sl_level = entry - tp_dist, entry + sl_dist

        outcome, exit_price = "TIMEOUT", entry
        for ts, bar in window.iterrows():
            hi, lo = bar["high"], bar["low"]
            if side == "LONG":
                if lo <= sl_level:
                    outcome, exit_price = "LOSS", sl_level; break
                if hi >= tp_level:
                    outcome, exit_price = "WIN", tp_level; break
            else:
                if hi >= sl_level:
                    outcome, exit_price = "LOSS", sl_level; break
                if lo <= tp_level:
                    outcome, exit_price = "WIN", tp_level; break

        if outcome == "TIMEOUT":
            exit_price = float(window.iloc[-1]["close"])

        if side == "LONG":
            mfe = max(0.0, float(window["high"].max()) - entry)
            mae = max(0.0, entry - float(window["low"].min()))
            pnl_pts = exit_price - entry
        else:
            mfe = max(0.0, entry - float(window["low"].min()))
            mae = max(0.0, float(window["high"].max()) - entry)
            pnl_pts = entry - exit_price
        return outcome, exit_price, pnl_pts, mfe, mae


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_per_strategy_regime(trades: List[EnrichedTrade]) -> Dict[str, Any]:
    """Build per-(strategy, regime) summary stats."""
    by_key: Dict[Tuple[str, str], List[EnrichedTrade]] = defaultdict(list)
    for t in trades:
        regime = t.regime_at_fire or ("dead_tape" if t.dead_tape_clipped else "unknown")
        by_key[(t.strategy, regime)].append(t)

    out = {}
    for (strategy, regime), group in by_key.items():
        pnls = [g.pnl_dollars for g in group]
        mfes = [g.mfe_pts for g in group if g.mfe_pts is not None]
        maes = [g.mae_pts for g in group if g.mae_pts is not None]
        out[f"{strategy}|{regime}"] = {
            "strategy": strategy,
            "regime": regime,
            "n": len(group),
            "n_long": sum(1 for g in group if g.side == "LONG"),
            "n_short": sum(1 for g in group if g.side == "SHORT"),
            "pnl_total": float(sum(pnls)),
            "pnl_avg": float(np.mean(pnls)) if pnls else 0.0,
            "win_rate_pct": float(sum(1 for p in pnls if p > 0) / max(1, len(pnls)) * 100),
            "mfe_p50": float(np.median(mfes)) if mfes else 0.0,
            "mfe_p75": float(np.percentile(mfes, 75)) if mfes else 0.0,
            "mfe_p90": float(np.percentile(mfes, 90)) if mfes else 0.0,
            "mae_p50": float(np.median(maes)) if maes else 0.0,
            "mae_p75": float(np.percentile(maes, 75)) if maes else 0.0,
            "mae_p90": float(np.percentile(maes, 90)) if maes else 0.0,
        }
    return out


def aggregate_dead_tape_counterfactuals(trades: List[EnrichedTrade]) -> Dict[str, Any]:
    """For dead-tape-clipped trades, summarize counterfactual savings/costs
    of bracket alternatives. Mirrors the validation we did manually."""
    clipped = [t for t in trades if t.dead_tape_clipped]
    if not clipped:
        return {"n": 0}
    by_strategy = defaultdict(list)
    for t in clipped:
        by_strategy[t.strategy].append(t)
    out = {"n_total": len(clipped), "by_strategy": {}}
    for strategy, group in by_strategy.items():
        actual = [g.pnl_dollars for g in group]
        cf_125 = [g.cf_pnl_at_125_10 for g in group if g.cf_pnl_at_125_10 is not None]
        cf_25 = [g.cf_pnl_at_25_10 for g in group if g.cf_pnl_at_25_10 is not None]
        out["by_strategy"][strategy] = {
            "n": len(group),
            "actual_total": float(sum(actual)),
            "cf_12.5_10_total": float(sum(cf_125)) if cf_125 else None,
            "cf_25_10_total": float(sum(cf_25)) if cf_25 else None,
            "delta_12.5_10_total": (float(sum(cf_125)) - float(sum(actual))) if cf_125 else None,
            "delta_25_10_total": (float(sum(cf_25)) - float(sum(actual))) if cf_25 else None,
            "verdict_12.5_10": "skip-guard would help" if cf_125 and sum(cf_125) > sum(actual) else "skip-guard would HURT",
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_telemetry(window_days: Optional[int] = None,
                     strategy_filter: Optional[str] = None,
                     ) -> Dict[str, Any]:
    print(f"[regime_telemetry] parsing {LOG.name} ...")
    events = parse_log(LOG)
    print(f"  closes:        {len(events['closes'])}")
    print(f"  placements:    {len(events['placements'])}")
    print(f"  dead_tape:     {len(events['dead_tape'])}")
    print(f"  v18:           {len(events['v18'])}")
    print(f"  transitions:   {len(events['regime_transitions'])}")
    print()

    closes = events["closes"]
    if window_days:
        cutoff = pd.Timestamp.now(tz="America/Los_Angeles") - pd.Timedelta(days=window_days)
        closes = [c for c in closes if c["ts_pdt"] >= cutoff]
        print(f"[regime_telemetry] filtered to last {window_days} days: {len(closes)} closes")
    if strategy_filter:
        closes = [c for c in closes if c.get("strategy") == strategy_filter]
        print(f"[regime_telemetry] filtered to strategy={strategy_filter}: {len(closes)} closes")
    print()

    walker = OHLCWalker()
    print(f"[regime_telemetry] OHLC parquet: {len(walker.df):,} bars")
    print()

    print("[regime_telemetry] enriching trades ...")
    enriched: List[EnrichedTrade] = []
    for c in closes:
        plc = match_placement(c, events["placements"])
        dt = match_dead_tape(plc, events["dead_tape"]) if plc else None
        v18 = match_v18(plc, events["v18"]) if plc else None
        regime = regime_at_ts(c["ts_pdt"], events["regime_transitions"])

        # Compute fired tp/sl distances from the placement
        if plc:
            if c["side"] == "LONG":
                fired_tp_dist = plc["tp"] - plc["entry"]
                fired_sl_dist = plc["entry"] - plc["sl"]
            else:
                fired_tp_dist = plc["entry"] - plc["tp"]
                fired_sl_dist = plc["sl"] - plc["entry"]
            # dead-tape signature: tp≈3.0 sl≈5.0
            is_clipped = (abs(fired_tp_dist - 3.0) < 0.1) and (abs(fired_sl_dist - 5.0) < 0.1)
        else:
            fired_tp_dist = fired_sl_dist = None
            is_clipped = False

        # MFE/MAE walk
        if plc:
            _, _, _, mfe, mae = walker.walk(
                plc["ts_pdt"], c["side"], plc["entry"],
                tp_dist=1e9, sl_dist=1e9,  # huge brackets so we just track excursion
                horizon_min=HORIZON_MIN_DEFAULT,
            )
        else:
            mfe = mae = None

        # Counterfactual at 12.5/10 if dead-tape-clipped
        cf_125 = cf_25 = None
        if is_clipped and plc:
            _, _, cf_pts_125, _, _ = walker.walk(
                plc["ts_pdt"], c["side"], plc["entry"],
                tp_dist=12.5, sl_dist=10.0,
                horizon_min=HORIZON_MIN_DEFAULT,
            )
            cf_125 = cf_pts_125 * c["size"] * DOLLAR_PER_PT_MES - COMMISSION * c["size"]
            _, _, cf_pts_25, _, _ = walker.walk(
                plc["ts_pdt"], c["side"], plc["entry"],
                tp_dist=25.0, sl_dist=10.0,
                horizon_min=HORIZON_MIN_DEFAULT,
            )
            cf_25 = cf_pts_25 * c["size"] * DOLLAR_PER_PT_MES - COMMISSION * c["size"]

        enriched.append(EnrichedTrade(
            ts_pdt=c["ts_pdt"],
            strategy=c["strategy"],
            side=c["side"],
            entry=c["entry"],
            exit=c["exit"],
            pnl_pts=c["pnl_pts"],
            pnl_dollars=c["pnl_dollars"],
            size=c["size"],
            order_id=c["order_id"],
            entry_order_id=c["entry_order_id"],
            fire_ts_pdt=plc["ts_pdt"] if plc else None,
            fired_tp_dist=fired_tp_dist,
            fired_sl_dist=fired_sl_dist,
            dead_tape_clipped=is_clipped,
            vol_bp_at_fire=dt["vol_bp"] if dt else None,
            regime_at_fire=regime,
            v18_proba=v18["proba"] if v18 else None,
            v18_decision=v18["dec"] if v18 else None,
            mfe_pts=mfe,
            mae_pts=mae,
            cf_pnl_at_125_10=cf_125,
            cf_pnl_at_25_10=cf_25,
        ))

    n_matched = sum(1 for t in enriched if t.fire_ts_pdt)
    print(f"[regime_telemetry] enriched {len(enriched)} trades  ({n_matched} matched to placement)")

    per_strat_regime = aggregate_per_strategy_regime(enriched)
    cf_summary = aggregate_dead_tape_counterfactuals(enriched)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_days": window_days,
        "strategy_filter": strategy_filter,
        "log_path": str(LOG),
        "n_total_trades": len(enriched),
        "n_matched_to_placement": n_matched,
        "n_dead_tape_clipped": sum(1 for t in enriched if t.dead_tape_clipped),
        "per_strategy_regime": per_strat_regime,
        "dead_tape_counterfactual_summary": cf_summary,
        "enriched_trades": [t.to_dict() for t in enriched],
    }


def write_outputs(report: Dict[str, Any]) -> None:
    json_path = OUT_DIR / "report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    md_path = OUT_DIR / "report.md"
    md_path.write_text(_render_markdown(report))
    csv_path = OUT_DIR / "enriched_trades.csv"
    df = pd.DataFrame(report["enriched_trades"])
    df.to_csv(csv_path, index=False)
    print(f"\n[regime_telemetry] wrote:")
    print(f"  {json_path} ({json_path.stat().st_size:,} bytes)")
    print(f"  {md_path} ({md_path.stat().st_size:,} bytes)")
    print(f"  {csv_path} ({csv_path.stat().st_size:,} bytes)")


def _render_markdown(r: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Regime Telemetry Report")
    lines.append(f"\n_Generated: {r['generated_at']}_\n")
    lines.append(f"- Total trades parsed: **{r['n_total_trades']}**")
    lines.append(f"- Matched to placement: **{r['n_matched_to_placement']}**")
    lines.append(f"- Dead-tape clipped: **{r['n_dead_tape_clipped']}**")
    if r.get("window_days"):
        lines.append(f"- Window: last **{r['window_days']}** days")
    if r.get("strategy_filter"):
        lines.append(f"- Strategy filter: **{r['strategy_filter']}**")
    lines.append("")

    lines.append("## Per-(strategy, regime) realized stats")
    lines.append("")
    lines.append("| Strategy | Regime | N | L/S | WR% | Avg PnL | MFE p50 | MFE p75 | MAE p50 | MAE p75 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, v in sorted(r["per_strategy_regime"].items()):
        lines.append(
            f"| {v['strategy']} | {v['regime']} | {v['n']} | "
            f"{v['n_long']}/{v['n_short']} | {v['win_rate_pct']:.1f}% | "
            f"${v['pnl_avg']:+.2f} | {v['mfe_p50']:.2f} | {v['mfe_p75']:.2f} | "
            f"{v['mae_p50']:.2f} | {v['mae_p75']:.2f} |"
        )
    lines.append("")

    cf = r.get("dead_tape_counterfactual_summary", {})
    if cf.get("n_total", 0):
        lines.append("## Dead-tape skip-guard counterfactual (per strategy)")
        lines.append("")
        lines.append("| Strategy | N | Actual $ | CF @12.5/10 $ | Δ @12.5/10 | CF @25/10 $ | Δ @25/10 | Verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s, v in cf.get("by_strategy", {}).items():
            lines.append(
                f"| {s} | {v['n']} | ${v['actual_total']:+.2f} | "
                f"${v.get('cf_12.5_10_total') or 0:+.2f} | "
                f"${v.get('delta_12.5_10_total') or 0:+.2f} | "
                f"${v.get('cf_25_10_total') or 0:+.2f} | "
                f"${v.get('delta_25_10_total') or 0:+.2f} | "
                f"{v.get('verdict_12.5_10', '?')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--window-days", type=int, default=None,
                   help="Restrict to trades closed within last N days")
    p.add_argument("--strategy", type=str, default=None,
                   help="Filter to one strategy (e.g., DynamicEngine3, AetherFlow)")
    args = p.parse_args()

    report = build_telemetry(window_days=args.window_days, strategy_filter=args.strategy)
    write_outputs(report)
    print(f"\n[regime_telemetry] done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""AetherFlow LONDON + NY_AM session backtest.

User asked: turn on AetherFlow for LONDON (session 1) and NY_AM (session 2),
but FIRST run a backtest showing performance.

Method:
  1. Pull all AetherFlow STRATEGY_SIGNAL log lines.
  2. Filter to session_id ∈ {1, 2}.
  3. For each signal, walk forward 30 minutes through ES OHLC parquet to
     determine TP/SL/timeout outcome at AF's designed brackets.
  4. Stratify by:
       - session (LONDON vs NY_AM)
       - block reason (today's reason codes — what would be unblocked)
       - gate_prob bucket (model confidence)
       - vol_regime (DISPERSED, CHOP_SPIRAL, TREND_GEODESIC)
  5. Compute counterfactual PnL: what would AF have made if we'd let these
     signals fire?

Two scenarios:
  A. "Pure unblock": fire every session_not_allowed-blocked signal in
     LONDON + NY_AM at AF's designed brackets.
  B. "Threshold-gated": only fire signals where gate_prob >= gate_threshold
     (filters out below_threshold blocks too — keeps model's confidence intact).

Output: artifacts/aetherflow_session_backtest/report.{md,json}
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "topstep_live_bot.log"
PARQUET = ROOT / "es_master_outrights-2.parquet"
OUT_DIR = ROOT / "artifacts" / "aetherflow_session_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOLLAR_PER_PT_MES = 5.0
COMMISSION = 7.50
HORIZON_MIN = 30
SIZE = 1  # 1 MES contract for baseline

SESSION_NAMES = {0: "ASIA", 1: "LONDON", 2: "NY_AM", 3: "NY_PM"}

AF_SIGNAL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\.]\d+ "
    r".*\[STRATEGY_SIGNAL\] .*"
    r"strategy=AetherFlow \| side=(?P<side>\w+) \| price=(?P<price>[\d.]+) \| "
    r"tp_dist=(?P<tp>[\d.]+) \| sl_dist=(?P<sl>[\d.]+) \| status=(?P<status>\w+) \|"
    r" decision=(?P<decision>\w+)(?: \| reason=(?P<reason>[^|]+))?"
)


def parse_signals() -> List[Dict]:
    sigs = []
    with open(LOG) as f:
        for line in f:
            m = AF_SIGNAL_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            ts = pd.Timestamp(d["ts"]).tz_localize("America/Los_Angeles")
            d["ts_pdt"] = ts
            d["ts_et"] = ts.tz_convert("US/Eastern")
            d["price"] = float(d["price"])
            d["tp"] = float(d["tp"])
            d["sl"] = float(d["sl"])
            d["reason"] = (d.get("reason") or "").strip()
            # Pull additional fields from the raw line
            for fkey in ("combo_key", "vol_regime", "session_id", "gate_prob", "gate_threshold"):
                m2 = re.search(rf"{fkey}=([^|]+)", line)
                if m2:
                    val = m2.group(1).strip()
                    try:
                        if fkey in ("session_id",):
                            val = int(val)
                        elif fkey in ("gate_prob", "gate_threshold"):
                            val = float(val)
                    except (ValueError, TypeError):
                        pass
                    d[fkey] = val
                else:
                    d[fkey] = None
            sigs.append(d)
    return sigs


class OHLCWalker:
    def __init__(self):
        df = pd.read_parquet(PARQUET).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        self.df = df

    def walk(self, ts_pdt, side, entry, tp_dist, sl_dist):
        """Return (outcome, exit_price, pnl_pts, mfe_pts, mae_pts)."""
        ts_naive = ts_pdt.tz_convert("UTC").tz_localize(None)
        deadline = ts_naive + timedelta(minutes=HORIZON_MIN)
        window = self.df[(self.df.index > ts_naive) & (self.df.index <= deadline)]
        if window.empty:
            return "NO_DATA", entry, 0.0, 0.0, 0.0
        if side == "LONG":
            tp_lvl, sl_lvl = entry + tp_dist, entry - sl_dist
        else:
            tp_lvl, sl_lvl = entry - tp_dist, entry + sl_dist
        outcome = "TIMEOUT"
        exit_price = entry
        for ts, bar in window.iterrows():
            hi, lo = bar["high"], bar["low"]
            if side == "LONG":
                if lo <= sl_lvl: outcome, exit_price = "LOSS", sl_lvl; break
                if hi >= tp_lvl: outcome, exit_price = "WIN", tp_lvl; break
            else:
                if hi >= sl_lvl: outcome, exit_price = "LOSS", sl_lvl; break
                if lo <= tp_lvl: outcome, exit_price = "WIN", tp_lvl; break
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


def main():
    print("Parsing AetherFlow signals from bot log ...")
    sigs = parse_signals()
    print(f"Total AF signals found: {len(sigs)}")

    # Filter to LONDON + NY_AM only
    target_sessions = {1: "LONDON", 2: "NY_AM"}
    target = [s for s in sigs if s.get("session_id") in target_sessions]
    print(f"  In LONDON or NY_AM: {len(target)}")

    # By session, status, reason
    by_sess_status = defaultdict(int)
    by_reason = defaultdict(int)
    for s in target:
        by_sess_status[(s.get("session_id"), s.get("status"))] += 1
        if s.get("status") == "BLOCKED":
            by_reason[(s.get("session_id"), s.get("reason"))] += 1
    print()
    print("=== By session × status ===")
    for (sess, st), n in sorted(by_sess_status.items()):
        print(f"  {SESSION_NAMES.get(sess, sess):<8} {st:<10} {n}")
    print()
    print("=== By session × block reason ===")
    for (sess, reason), n in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {SESSION_NAMES.get(sess, sess):<8} {reason:<35} {n}")

    # ============ Walk forward all LONDON + NY_AM signals ============
    print("\n[ Loading OHLC parquet ... ]")
    walker = OHLCWalker()
    print(f"  parquet bars: {len(walker.df):,}")

    print("\n[ Walking each signal forward 30min ... ]")
    enriched = []
    for s in target:
        outcome, exit_p, pnl_pts, mfe, mae = walker.walk(
            s["ts_pdt"], s["side"], s["price"], s["tp"], s["sl"]
        )
        if outcome == "NO_DATA":
            continue
        pnl_dollars = pnl_pts * SIZE * DOLLAR_PER_PT_MES - COMMISSION * SIZE
        enriched.append({
            "ts_et": s["ts_et"],
            "session": SESSION_NAMES.get(s["session_id"], "?"),
            "side": s["side"],
            "entry": s["price"],
            "tp_dist": s["tp"],
            "sl_dist": s["sl"],
            "status": s["status"],
            "block_reason": s["reason"] if s["status"] == "BLOCKED" else "",
            "combo_key": s.get("combo_key"),
            "vol_regime": s.get("vol_regime"),
            "gate_prob": s.get("gate_prob"),
            "gate_threshold": s.get("gate_threshold"),
            "outcome": outcome,
            "pnl_pts": pnl_pts,
            "pnl_dollars": pnl_dollars,
            "mfe_pts": mfe,
            "mae_pts": mae,
        })
    df = pd.DataFrame(enriched)
    print(f"  enriched rows: {len(df)}")

    # Many signals have tp=0/sl=0 because they were blocked before bracket
    # resolution. Use AF's designed default of 3.0/5.0 in those cases.
    df["tp_used"] = df["tp_dist"].where(df["tp_dist"] > 0, 3.0)
    df["sl_used"] = df["sl_dist"].where(df["sl_dist"] > 0, 5.0)

    # Re-walk with corrected brackets where tp/sl were 0
    print("\n[ Re-walking signals with 0/0 brackets at default 3.0/5.0 ... ]")
    fixed = []
    for r in df.to_dict("records"):
        if r["tp_dist"] > 0:
            fixed.append(r)
            continue
        ts_pdt = r["ts_et"].tz_convert("America/Los_Angeles")
        outcome, exit_p, pnl_pts, mfe, mae = walker.walk(
            ts_pdt, r["side"], r["entry"], 3.0, 5.0
        )
        if outcome == "NO_DATA":
            continue
        r["outcome"] = outcome
        r["pnl_pts"] = pnl_pts
        r["pnl_dollars"] = pnl_pts * SIZE * DOLLAR_PER_PT_MES - COMMISSION * SIZE
        r["mfe_pts"] = mfe
        r["mae_pts"] = mae
        fixed.append(r)
    df = pd.DataFrame(fixed)

    # ============ Scenarios ============
    print()
    print("=" * 80)
    print("SCENARIO A: Pure unblock — fire every AF signal in LONDON + NY_AM")
    print("=" * 80)
    sa = df.copy()
    print(f"Signals: {len(sa)}")
    sa_summary = summarize(sa, "Scenario A (pure unblock)")
    print()

    print("=" * 80)
    print("SCENARIO B: Threshold-gated — only fire when gate_prob >= gate_threshold")
    print("=" * 80)
    sb = df[(df["gate_prob"].notna()) & (df["gate_threshold"].notna()) & (df["gate_prob"] >= df["gate_threshold"])].copy()
    print(f"Signals passing threshold: {len(sb)}")
    sb_summary = summarize(sb, "Scenario B (threshold-gated)")
    print()

    print("=" * 80)
    print("SCENARIO C: Threshold + non-session blocks honored")
    print("=" * 80)
    # Keep ONLY blocks where the reason was session_not_allowed (would be unblocked).
    # Strip out hazard_blocked / regime_not_allowed / directional_vwap_too_far / signed_d_alignment etc.
    # — those were blocked for non-session reasons that we'd want to keep.
    sc_keep_reasons = {"session_not_allowed", "below_threshold", "", None}
    sc = df[df["block_reason"].isin([""] + list(sc_keep_reasons))].copy()
    sc = sc[(sc["status"] == "CANDIDATE") |
            (sc["block_reason"] == "session_not_allowed") |
            ((sc["block_reason"] == "below_threshold") & sc["gate_prob"].notna() &
             (sc["gate_prob"] >= sc["gate_threshold"]))]
    print(f"Signals: {len(sc)}")
    sc_summary = summarize(sc, "Scenario C (clean unblock)")
    print()

    print("=" * 80)
    print("STRATIFIED — Scenario A by session × side")
    print("=" * 80)
    if len(sa):
        g = sa.groupby(["session", "side"]).agg(
            n=("pnl_dollars", "count"),
            wins=("pnl_dollars", lambda s: (s > 0).sum()),
            losses=("pnl_dollars", lambda s: (s < 0).sum()),
            net_pnl=("pnl_dollars", "sum"),
            avg_pnl=("pnl_dollars", "mean"),
            wr_pct=("pnl_dollars", lambda s: (s > 0).mean() * 100),
            mfe_p75=("mfe_pts", lambda s: s.quantile(0.75)),
            mae_p75=("mae_pts", lambda s: s.quantile(0.75)),
        )
        print(g.to_string())
    print()

    print("=" * 80)
    print("STRATIFIED — Scenario A by block reason")
    print("=" * 80)
    if len(sa):
        g = sa.groupby("block_reason").agg(
            n=("pnl_dollars", "count"),
            wr_pct=("pnl_dollars", lambda s: (s > 0).mean() * 100),
            net_pnl=("pnl_dollars", "sum"),
            avg_pnl=("pnl_dollars", "mean"),
        ).sort_values("net_pnl")
        print(g.to_string())

    # Save outputs
    df.to_csv(OUT_DIR / "all_signals_walked.csv", index=False)
    summary_payload = {
        "n_total_af_signals": len(sigs),
        "n_in_target_sessions": len(target),
        "n_walked": len(df),
        "scenario_a_pure_unblock": sa_summary,
        "scenario_b_threshold_gated": sb_summary,
        "scenario_c_clean_unblock": sc_summary,
        "by_session_status": {f"{SESSION_NAMES.get(k[0],k[0])}_{k[1]}": v for k, v in by_sess_status.items()},
        "by_session_block_reason": {f"{SESSION_NAMES.get(k[0],k[0])}_{k[1]}": v for k, v in by_reason.items()},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(summary_payload, indent=2, default=str))

    # Markdown
    md = ["# AetherFlow LONDON + NY_AM session backtest", ""]
    md.append(f"_Generated: {pd.Timestamp.now()}_\n")
    md.append(f"- Total AF signals in log: **{len(sigs)}**")
    md.append(f"- LONDON + NY_AM signals: **{len(target)}**")
    md.append(f"- Walked successfully: **{len(df)}**\n")
    md.append("## Headline numbers (1 MES, $5/pt, $7.50 round-trip)\n")
    md.append(f"| Scenario | n | WR% | Net PnL | Avg/trade | Note |")
    md.append(f"|---|---|---|---|---|---|")
    for label, s in [("A: pure unblock", sa_summary), ("B: threshold-gated", sb_summary), ("C: clean unblock", sc_summary)]:
        md.append(f"| {label} | {s['n']} | {s['wr']:.1f}% | ${s['net_pnl']:+,.2f} | ${s['avg_pnl']:+.2f} | {s['note']} |")
    md.append("")
    (OUT_DIR / "report.md").write_text("\n".join(md))

    print(f"\nWrote:")
    print(f"  {OUT_DIR}/all_signals_walked.csv")
    print(f"  {OUT_DIR}/report.json")
    print(f"  {OUT_DIR}/report.md")


def summarize(df, label):
    if df.empty:
        print(f"  (empty)")
        return {"n": 0, "net_pnl": 0, "avg_pnl": 0, "wr": 0, "note": "empty"}
    n = len(df); wins = (df["pnl_dollars"] > 0).sum()
    losses = (df["pnl_dollars"] < 0).sum()
    net = df["pnl_dollars"].sum(); avg = df["pnl_dollars"].mean()
    wr = wins / n * 100
    span_days = (df["ts_et"].max() - df["ts_et"].min()).days + 1
    annualized = net * 252 / max(1, span_days)
    print(f"  Trades: {n}  |  {wins}W / {losses}L  |  WR {wr:.1f}%")
    print(f"  Net PnL: ${net:+,.2f}  |  Avg ${avg:+.2f}/trade")
    print(f"  Span: {span_days} days  |  Annualized: ${annualized:+,.0f}/yr")
    cum = df.sort_values("ts_et")["pnl_dollars"].cumsum()
    dd = (cum - cum.cummax()).min()
    print(f"  Max DD: ${abs(dd):.2f}")
    return {"n": int(n), "wins": int(wins), "losses": int(losses), "wr": float(wr),
            "net_pnl": float(net), "avg_pnl": float(avg), "max_dd": float(abs(dd)),
            "span_days": int(span_days), "annualized": float(annualized),
            "note": label}


if __name__ == "__main__":
    main()

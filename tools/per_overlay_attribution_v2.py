#!/usr/bin/env python3
"""Per-overlay attribution drill against the corrected v2 corpus.

Walks the candidate stream `full_overlay_stack_simulation_14mo_v2.parquet`
once per overlay X. For each X:
- Apply ONLY X's BLOCK / WAIT / size-mult decision.
- Re-walk the friend rule (one open trade at a time, family-aware deferral
  approximated by a single global open-position window — same convention
  used by `kalshi_gate_reconstruction_v2.py`).
- Apply $7.50/trade haircut (already baked into raw_pnl - net via pnl
  fields).
- Aggregate trades / WR / PnL / DD per month and overall.
- Compute delta vs the filterless baseline (no overlay applied).

For Filter G this is a fresh isolation (was never separately computed
at v2 level). For Kalshi/LFO/PCT/Pivot we recompute on the same friend-
rule walk so the numbers are apples-to-apples (the existing per-overlay
recon parquets use slightly different conventions).

Outputs:
- `/tmp/per_overlay_attribution_v2.json` — full attribution table
- stdout — human-readable ranking + worst-offender deep dive
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGS_PATH = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo_v2.parquet"
OUT_JSON = Path("/tmp/per_overlay_attribution_v2.json")

HORIZON = 30  # bars; matches kalshi_gate_reconstruction_v2.py


def _ts_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str[:19])


def walk_friend_rule(rows: pd.DataFrame, blocked_mask: np.ndarray) -> pd.DataFrame:
    """Walk rows in time order; skip if a prior trade is still open
    OR the candidate is blocked. blocked_mask True = block.
    Single open-position friend rule — matches the convention used by
    kalshi_gate_reconstruction_v2.py and the v2 baseline.
    """
    out = rows.copy().reset_index(drop=True)
    out["taken"] = False
    out["blocked_by_overlay"] = False
    next_open_at = None
    n = len(out)
    for i in range(n):
        ts = out.at[i, "ts_n"]
        if blocked_mask[i]:
            out.at[i, "blocked_by_overlay"] = True
            continue
        if next_open_at is not None and ts < next_open_at:
            continue
        out.at[i, "taken"] = True
        exit_ts = out.at[i, "exit_ts"]
        if pd.notna(exit_ts):
            nx = pd.Timestamp(exit_ts)
            if nx.tzinfo is not None:
                nx = nx.tz_convert("US/Eastern").tz_localize(None)
            next_open_at = nx
        else:
            next_open_at = ts + pd.Timedelta(minutes=HORIZON)
    return out


def aggregate(walk_df: pd.DataFrame) -> dict:
    taken = walk_df[walk_df["taken"]].copy()
    n = len(taken)
    wins = int((taken["raw_pnl_walk"] > 0).sum())
    wr = 100.0 * wins / n if n else 0.0
    pnl = float(taken["net_pnl"].sum())
    cum = taken.sort_values("ts_n")["net_pnl"].cumsum()
    dd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
    return {
        "trades": int(n),
        "wr": round(wr, 2),
        "pnl_net": round(pnl, 2),
        "max_dd": round(dd, 2),
    }


def per_month(walk_df: pd.DataFrame) -> list:
    taken = walk_df[walk_df["taken"]].copy()
    if taken.empty:
        return []
    taken["m"] = taken["ts_n"].dt.strftime("%Y-%m")
    out = []
    for m, sub in taken.groupby("m", sort=True):
        n = len(sub)
        wins = int((sub["raw_pnl_walk"] > 0).sum())
        wr = 100.0 * wins / n if n else 0.0
        pnl = float(sub["net_pnl"].sum())
        cum = sub.sort_values("ts_n")["net_pnl"].cumsum()
        dd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
        out.append({"month": m, "trades": n, "wr": round(wr, 2),
                    "pnl": round(pnl, 2), "dd": round(dd, 2)})
    return out


def block_mask_from_decision(df: pd.DataFrame, name: str) -> np.ndarray:
    """Return True where overlay `name` would BLOCK this candidate.
    For overlays that don't block (LFO=WAIT, PCT=size-down) this is False;
    those overlays affect PnL via per-row adjustments, not by blocking.
    """
    if name == "filter_g":
        return (df["fg_decision"] == "BLOCK").to_numpy()
    if name == "kalshi":
        return (df["k_decision"] == "BLOCK").to_numpy()
    if name in {"lfo", "pct", "pivot"}:
        return np.zeros(len(df), dtype=bool)
    raise ValueError(name)


def adjusted_net_pnl(df: pd.DataFrame, name: str) -> np.ndarray:
    """Return per-row net_pnl under overlay X being applied.

    Baseline net_pnl is `pnl_baseline`-equivalent: raw_pnl_walk - $7.50 haircut.
    But `pnl_baseline` in the parquet is conditional on taken_baseline. We
    instead recompute net_pnl per-row from raw_pnl_walk so the friend-rule
    walk can pick its own taken set.
    """
    raw = df["raw_pnl_walk"].astype(float).to_numpy()
    base_net = raw - 7.5

    if name in {"filter_g", "kalshi", "pivot"}:
        # Pure block / pass overlays: no per-row adjustment to PnL when
        # the trade is taken. (Pivot is pass-through in v2, see notes.)
        return base_net

    if name == "lfo":
        # LFO IMMEDIATE => identical to baseline.
        # LFO WAIT => trade is structurally deferred; in the recon parquet
        # `pnl_lfo_net` represents the net under WAIT semantics. We use
        # the per-row LFO recon delta to compute the WAIT-adjusted net.
        # Read once and join by (ts, strategy, side).
        return _lfo_adjusted_net(df, base_net)

    if name == "pct":
        # PCT scales size & TP. Use the recon parquet's `pnl_with_pct_net`
        # for active-PCT rows (BREAKOUT_LEAN / PIVOT_LEAN), baseline for
        # NOT_AT_LEVEL. Join by (ts, strategy, side).
        return _pct_adjusted_net(df, base_net)

    raise ValueError(name)


_LFO_RECON = None
_PCT_RECON = None


def _lfo_adjusted_net(df: pd.DataFrame, base_net: np.ndarray) -> np.ndarray:
    global _LFO_RECON
    if _LFO_RECON is None:
        _LFO_RECON = pd.read_parquet(ROOT / "artifacts" / "lfo_reconstruction_14mo_v2.parquet").copy()
        _LFO_RECON["ts_n"] = _ts_naive(_LFO_RECON["ts"])
        _LFO_RECON = _LFO_RECON.drop_duplicates(subset=["ts_n", "strategy", "side"], keep="first")
    keys = ["ts_n", "strategy", "side"]
    out = df.merge(
        _LFO_RECON[keys + ["lfo_decision", "pnl_lfo_net", "pnl_immediate"]].rename(
            columns={"lfo_decision": "lfo_decision_recon"}
        ),
        on=keys, how="left", validate="m:1",
    )
    assert len(out) == len(df), f"merge changed row count: {len(df)} -> {len(out)}"
    adj = base_net.copy()
    # Use the candidate-stream lfo_decision as primary (matches the live
    # overlay decision); fall back to recon decision when stream is missing.
    decis = out["lfo_decision"].fillna(out["lfo_decision_recon"])
    wait_mask = (decis == "WAIT").to_numpy()
    has_recon = out["pnl_lfo_net"].notna().to_numpy()
    use = wait_mask & has_recon
    adj[use] = out.loc[use, "pnl_lfo_net"].astype(float).to_numpy()
    return adj


def _pct_adjusted_net(df: pd.DataFrame, base_net: np.ndarray) -> np.ndarray:
    global _PCT_RECON
    if _PCT_RECON is None:
        _PCT_RECON = pd.read_parquet(ROOT / "artifacts" / "pct_reconstruction_14mo_v2.parquet").copy()
        _PCT_RECON["ts_n"] = _ts_naive(_PCT_RECON["ts"])
        _PCT_RECON = _PCT_RECON.drop_duplicates(subset=["ts_n", "strategy", "side"], keep="first")
    keys = ["ts_n", "strategy", "side"]
    out = df.merge(
        _PCT_RECON[keys + ["pct_decision", "pnl_with_pct_net", "pnl_no_pct"]].rename(
            columns={"pct_decision": "pct_decision_recon"}
        ),
        on=keys, how="left", validate="m:1",
    )
    assert len(out) == len(df), f"merge changed row count: {len(df)} -> {len(out)}"
    adj = base_net.copy()
    decis = out["pct_decision"].fillna(out["pct_decision_recon"]).fillna("NOT_AT_LEVEL")
    active_mask = decis.isin(["BREAKOUT_LEAN", "PIVOT_LEAN", "NEUTRAL"]).to_numpy()
    has_recon = out["pnl_with_pct_net"].notna().to_numpy()
    use = active_mask & has_recon
    adj[use] = out.loc[use, "pnl_with_pct_net"].astype(float).to_numpy()
    return adj


def run_overlay(name: str, df: pd.DataFrame) -> dict:
    blocked = block_mask_from_decision(df, name)
    df2 = df.copy()
    df2["net_pnl"] = adjusted_net_pnl(df2, name)
    walk = walk_friend_rule(df2, blocked)
    agg = aggregate(walk)
    pm = per_month(walk)
    # Counterfactual: PnL of trades blocked by this overlay had they been
    # taken (under their baseline net). Sum of base_net on blocked rows
    # that would have been taken under the no-overlay walk.
    base_net = df2["raw_pnl_walk"].astype(float).to_numpy() - 7.5
    no_block = np.zeros(len(df2), dtype=bool)
    df_base = df2.copy()
    df_base["net_pnl"] = base_net
    base_walk = walk_friend_rule(df_base, no_block)
    taken_in_baseline = base_walk["taken"].to_numpy()
    blocked_and_would_have_traded = blocked & taken_in_baseline
    edge_destroyed = float(base_net[blocked_and_would_have_traded].sum())
    blocked_winners = int(((df2["raw_pnl_walk"] > 0) & blocked_and_would_have_traded).sum())
    blocked_losers = int(((df2["raw_pnl_walk"] < 0) & blocked_and_would_have_traded).sum())
    blocked_total = int(blocked_and_would_have_traded.sum())
    block_precision_pct = (
        100.0 * blocked_losers / blocked_total if blocked_total else 0.0
    )
    return {
        "agg": agg,
        "per_month": pm,
        "blocked_total_decisions": int(blocked.sum()),
        "blocked_and_would_have_traded": blocked_total,
        "blocked_winners": blocked_winners,
        "blocked_losers": blocked_losers,
        "block_precision_pct_loser": round(block_precision_pct, 2),
        "edge_destroyed_pnl": round(edge_destroyed, 2),
        "walk": walk,
    }


def main() -> int:
    df = pd.read_parquet(SIGS_PATH).copy()
    df["ts_n"] = _ts_naive(df["ts"])
    df = df.sort_values("ts_n").reset_index(drop=True)
    print(f"[attr] candidate stream: n={len(df)}")

    # Baseline (no overlay)
    df_b = df.copy()
    df_b["net_pnl"] = df_b["raw_pnl_walk"].astype(float).to_numpy() - 7.5
    base_walk = walk_friend_rule(df_b, np.zeros(len(df_b), dtype=bool))
    base_agg = aggregate(base_walk)
    base_pm = per_month(base_walk)
    print(f"[attr] baseline: trades={base_agg['trades']} pnl={base_agg['pnl_net']} "
          f"WR={base_agg['wr']} DD={base_agg['max_dd']}")

    overlays = ["filter_g", "kalshi", "lfo", "pct", "pivot"]
    results = {"baseline": {"agg": base_agg, "per_month": base_pm}}

    for name in overlays:
        print(f"\n[attr] === overlay: {name} ===")
        r = run_overlay(name, df)
        # Compute deltas vs baseline
        d_pnl = r["agg"]["pnl_net"] - base_agg["pnl_net"]
        d_trades = r["agg"]["trades"] - base_agg["trades"]
        d_wr = r["agg"]["wr"] - base_agg["wr"]
        d_dd = r["agg"]["max_dd"] - base_agg["max_dd"]
        r["delta_vs_baseline"] = {
            "d_pnl": round(d_pnl, 2),
            "d_trades": d_trades,
            "d_wr_pp": round(d_wr, 2),
            "d_dd": round(d_dd, 2),
        }
        # Drop walk df from JSON output
        walk_df = r.pop("walk")
        results[name] = r
        print(f"  trades={r['agg']['trades']:4d}  pnl={r['agg']['pnl_net']:>10.2f}  "
              f"WR={r['agg']['wr']:5.2f}%  DD={r['agg']['max_dd']:>9.2f}  "
              f"| dPnL={d_pnl:>+10.2f}  dTrades={d_trades:>+4d}  dWR={d_wr:>+5.2f}pp  dDD={d_dd:>+9.2f}")
        print(f"  blocked_decisions={r['blocked_total_decisions']}  "
              f"blocked_and_would_have_traded={r['blocked_and_would_have_traded']}  "
              f"winners_blocked={r['blocked_winners']}  losers_blocked={r['blocked_losers']}  "
              f"loser-precision={r['block_precision_pct_loser']}%  "
              f"edge_destroyed=${r['edge_destroyed_pnl']}")
        # Persist walk for the worst-offender deep dive
        results[name]["_walk"] = walk_df

    # ---- Ranking by |dPnL| ----
    ranked = sorted(
        [(n, results[n]["delta_vs_baseline"]["d_pnl"]) for n in overlays],
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    print("\n=== RANKING BY |dPnL| ===")
    for i, (n, d) in enumerate(ranked, 1):
        print(f"  {i}. {n:>10s}  dPnL=${d:>+9.2f}")

    # ---- Worst-offender deep dive ----
    worst_name = ranked[0][0]
    worst = results[worst_name]
    walk = worst["_walk"]
    df_full = df.copy()
    df_full["net_pnl"] = df_full["raw_pnl_walk"].astype(float).to_numpy() - 7.5

    print(f"\n=== WORST OFFENDER: {worst_name} ===")
    print("  per-month dPnL vs baseline:")
    base_pm_map = {row["month"]: row for row in base_pm}
    pm_rows = []
    for row in worst["per_month"]:
        bm = base_pm_map.get(row["month"], {"pnl": 0.0, "trades": 0, "wr": 0})
        d = row["pnl"] - bm["pnl"]
        pm_rows.append({"month": row["month"], "ovl_pnl": row["pnl"],
                        "base_pnl": bm["pnl"], "delta": round(d, 2),
                        "blocked": bm["trades"] - row["trades"]})
        print(f"    {row['month']}: ovl=${row['pnl']:>+9.2f}  base=${bm['pnl']:>+9.2f}  "
              f"dPnL=${d:>+9.2f}  trades_lost={bm['trades']-row['trades']}")

    # By side / strategy: blocked rows attributable to the overlay
    blocked_mask = block_mask_from_decision(df_full, worst_name)
    base_walk_taken = base_walk["taken"].to_numpy()
    bm_and_taken = blocked_mask & base_walk_taken
    print("\n  by SIDE (of blocked-and-would-trade):")
    for side, sub in df_full[bm_and_taken].groupby("side"):
        net = (sub["raw_pnl_walk"] - 7.5).sum()
        print(f"    {side:>5s}  n={len(sub)}  edge_destroyed=${net:>+9.2f}")
    print("  by STRATEGY:")
    for strat, sub in df_full[bm_and_taken].groupby("strategy"):
        net = (sub["raw_pnl_walk"] - 7.5).sum()
        print(f"    {strat:>15s}  n={len(sub)}  edge_destroyed=${net:>+9.2f}")

    # Active fire rate: of all candidates, how many got the BLOCK decision?
    fire_rate = 100.0 * blocked_mask.sum() / len(df_full)
    print(f"\n  fire rate (BLOCK decisions / candidates): {fire_rate:.1f}%")

    # Strip walk from JSON dump
    for n in overlays:
        results[n].pop("_walk", None)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[attr] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

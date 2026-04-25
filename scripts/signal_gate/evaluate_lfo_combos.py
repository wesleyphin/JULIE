"""Evaluate gate + fill-optimization combo matrix on the honest overlap window.

This script compares the current corrected per-strategy G gate against the
fill-optimization stack on the common trade set where we have:

- realized IMMEDIATE vs WAIT PnL from ``lfo_training_data.parquet``
- enough historical bars to replay the rule-based LevelFillOptimizer
- enough strategy context to score the corrected per-strategy G gates

Combos evaluated on one common universe:
  1. ungated + immediate
  2. ungated + rule_lfo
  3. ungated + ml_lfo
  4. ungated + live_hybrid_lfo   (rule LFO, ML only cancels WAIT -> IMMEDIATE)
  5. gated + immediate
  6. gated + rule_lfo
  7. gated + ml_lfo
  8. gated + live_hybrid_lfo

Output:
  artifacts/signal_gate_2025/lfo_combo_matrix_YYYYMMDD.json
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "artifacts" / "signal_gate_2025"
PARQUET = ROOT / "es_master_outrights.parquet"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))

from bank_level_quarter_filter import BankLevelQuarterFilter  # noqa: E402
from level_fill_optimizer import (  # noqa: E402
    FILL_AT_LEVEL,
    FILL_IMMEDIATE,
    FILL_WAIT,
    LevelFillOptimizer,
)
import ml_overlay_shadow as mls  # noqa: E402
import signal_gate_2025 as sg  # noqa: E402
from structural_level_tracker import StructuralLevelTracker  # noqa: E402
from train_lfo_ml import collect_all_trades  # noqa: E402
from train_per_strategy_models import active_symbol, compute_features_for_trades  # noqa: E402

NY = ZoneInfo("America/New_York")
LFO_DATASET = ART_DIR / "lfo_training_data.parquet"
OUT_PATH = ART_DIR / f"lfo_combo_matrix_{datetime.now().strftime('%Y%m%d')}.json"


def compute_dd(trade_pnls: List[float]) -> float:
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for pnl in trade_pnls:
        cum += float(pnl)
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return float(dd)


def summarize_rows(rows: List[Dict[str, Any]], pnl_key: str) -> Dict[str, Any]:
    pnls = [float(r[pnl_key]) for r in rows]
    executed = [float(r[pnl_key]) for r in rows if not bool(r.get("blocked_by_gate", False))]
    waits = sum(1 for r in rows if r.get("final_fill_mode") == FILL_WAIT and not bool(r.get("blocked_by_gate", False)))
    vetoes = sum(1 for r in rows if bool(r.get("blocked_by_gate", False)))
    wins = sum(1 for p in executed if p > 0)
    return {
        "n_rows": int(len(rows)),
        "n_executed": int(len(executed)),
        "n_vetoed": int(vetoes),
        "n_waits": int(waits),
        "wait_rate_among_executed": float(waits / len(executed)) if executed else 0.0,
        "veto_rate": float(vetoes / len(rows)) if rows else 0.0,
        "pnl": float(sum(executed)),
        "dd": compute_dd(pnls),
        "wr": float(wins / len(executed)) if executed else 0.0,
        "avg_trade": float(sum(executed) / len(executed)) if executed else 0.0,
    }


def combo_fill_mode(row: Dict[str, Any], fill_policy: str) -> str:
    rule_mode = row["rule_fill_mode"]
    ml_mode = row["ml_fill_mode"]
    if fill_policy == "immediate":
        return FILL_IMMEDIATE
    if fill_policy == "rule_lfo":
        return rule_mode
    if fill_policy == "ml_lfo":
        return ml_mode
    if fill_policy == "live_hybrid_lfo":
        if rule_mode == FILL_WAIT and ml_mode == FILL_IMMEDIATE:
            return FILL_IMMEDIATE
        return rule_mode
    raise ValueError(f"unknown fill policy: {fill_policy}")


def pnl_for_mode(row: Dict[str, Any], mode: str) -> float:
    if mode == FILL_WAIT:
        return float(row["wait_pnl_dol"])
    return float(row["imm_pnl_dol"])


def gate_payload_for_strategy(strategy: str, payloads: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    family = sg._strategy_family(strategy)  # pylint: disable=protected-access
    return payloads.get(family)


def reconstruct_rule_lfo(
    trade_row: Dict[str, Any],
    master_df: pd.DataFrame,
    optimizer: LevelFillOptimizer,
) -> Dict[str, Any]:
    entry_et = datetime.fromisoformat(str(trade_row["entry_time"])).astimezone(NY)
    symbol = active_symbol(entry_et)
    end_utc = pd.Timestamp(entry_et).tz_convert("UTC")
    start_utc = end_utc - pd.Timedelta(hours=60)

    sub = master_df.loc[
        (master_df.index >= start_utc)
        & (master_df.index <= end_utc + pd.Timedelta(minutes=5))
        & (master_df["symbol"] == symbol),
        ["open", "high", "low", "close", "volume"],
    ]
    if len(sub) < 100:
        return {"mode": FILL_IMMEDIATE, "reason": "insufficient_history"}

    idx = sub.index.searchsorted(end_utc)
    sig_idx = idx - 2
    if sig_idx < 1 or sig_idx >= len(sub):
        return {"mode": FILL_IMMEDIATE, "reason": "signal_bar_not_found"}

    bank_filter = BankLevelQuarterFilter()
    structural_tracker = StructuralLevelTracker()

    hist = sub.iloc[: sig_idx + 1]
    for ts, bar in hist.iterrows():
        ts_et = pd.Timestamp(ts).tz_convert(NY).to_pydatetime()
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        bank_filter.update(ts_et, h, l, c)
        structural_tracker.update(ts_et, o, h, l, c)

    sig_bar = hist.iloc[-1]
    signal = {
        "side": trade_row["side"],
        "sl_dist": float(trade_row["sl_dist_pts"]),
        "tp_dist": float(trade_row["tp_dist_pts"]),
    }
    return optimizer.evaluate(
        signal=signal,
        current_price=float(sig_bar["close"]),
        structural_tracker=structural_tracker,
        bank_filter=bank_filter,
        bar_candle={
            "open": float(sig_bar["open"]),
            "high": float(sig_bar["high"]),
            "low": float(sig_bar["low"]),
            "close": float(sig_bar["close"]),
        },
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but GradientBoostingClassifier was fitted with feature names",
        category=UserWarning,
    )

    print(f"[load] {LFO_DATASET}")
    lfo_df = pd.read_parquet(LFO_DATASET).copy()
    lfo_df["entry_time"] = lfo_df["entry_time"].astype(str)
    lfo_df["side"] = lfo_df["side"].astype(str).str.upper()

    start_et = pd.Timestamp(lfo_df["entry_time"].min())
    end_et = pd.Timestamp(lfo_df["entry_time"].max())
    print(f"  rows={len(lfo_df)}  window={start_et} -> {end_et}")

    print(f"[load] {PARQUET}")
    master = pd.read_parquet(PARQUET)
    master = master[
        (master.index >= (start_et.tz_convert("UTC") - pd.Timedelta(days=7)))
        & (master.index <= (end_et.tz_convert("UTC") + pd.Timedelta(days=1)))
    ]
    print(f"  scoped bars={len(master):,}")

    print("[collect] raw trades from same sources used to build LFO dataset")
    raw_trades = collect_all_trades()
    raw_map = {}
    for t in raw_trades:
        key = (str(t.get("entry_time")), str(t.get("side", "")).upper())
        raw_map[key] = t
    lfo_keys = {(r["entry_time"], r["side"]) for _, r in lfo_df.iterrows()}
    raw_subset = [t for t in raw_trades if (str(t.get("entry_time")), str(t.get("side", "")).upper()) in lfo_keys]
    print(f"  collected={len(raw_trades)}  overlap_subset={len(raw_subset)}")

    print("[gate] compute corrected gate context on the overlap trades")
    feat_df = compute_features_for_trades(raw_subset, master).copy()
    feat_df["entry_time"] = feat_df["entry_time"].astype(str)
    feat_df["side"] = feat_df["side"].astype(str).str.upper()
    feat_df["join_key"] = feat_df["entry_time"] + "||" + feat_df["side"]
    feat_map = {k: row.to_dict() for k, row in feat_df.set_index("join_key").iterrows()}
    print(f"  gate feature rows={len(feat_df)}")

    gate_payloads = {
        "de3": joblib.load(ART_DIR / "model_de3.joblib"),
        "aetherflow": joblib.load(ART_DIR / "model_aetherflow.joblib"),
        "regimeadaptive": joblib.load(ART_DIR / "model_regimeadaptive.joblib"),
    }

    print("[ml] load current ML LFO payload")
    mls._LFO_PAYLOAD = joblib.load(ART_DIR / "model_lfo.joblib")  # pylint: disable=protected-access

    optimizer = LevelFillOptimizer()

    print("[replay] reconstruct rule LFO + corrected gate + ML LFO choices")
    enriched: List[Dict[str, Any]] = []
    skipped = 0
    for _, r in lfo_df.sort_values("entry_time").iterrows():
        key = (r["entry_time"], r["side"])
        raw_trade = raw_map.get(key)
        feat_row = feat_map.get(f"{r['entry_time']}||{r['side']}")
        if raw_trade is None or feat_row is None:
            skipped += 1
            continue

        strategy = str(raw_trade.get("strategy", r.get("strategy", "")))
        gate_payload = gate_payload_for_strategy(strategy, gate_payloads)
        if gate_payload is None:
            skipped += 1
            continue

        gate_p = sg._score_with_gate(  # pylint: disable=protected-access
            gate_payload,
            side=r["side"],
            regime=str(feat_row.get("regime", "") or ""),
            mkt_regime=str(feat_row.get("mkt_regime", "") or ""),
            et_hour=int(feat_row.get("et_hour", int(r["et_hour"]))),
            bar_features=feat_row,
        )
        if gate_p is None:
            skipped += 1
            continue

        gate_thr = float(gate_payload.get("veto_threshold", 1.0))
        gate_veto = bool(gate_p >= gate_thr)

        rule_decision = reconstruct_rule_lfo(r.to_dict(), master, optimizer)
        rule_mode = str(rule_decision.get("mode", FILL_IMMEDIATE))
        if rule_mode not in {FILL_IMMEDIATE, FILL_AT_LEVEL, FILL_WAIT}:
            rule_mode = FILL_IMMEDIATE

        ml_score = mls.score_lfo(
            signal={"side": r["side"]},
            bar_features=r.to_dict(),
            dist_to_bank_below=float(r["dist_to_bank_below"]),
            dist_to_bank_above=float(r["dist_to_bank_above"]),
            bar_range_pts=float(r["bar_range_pts"]),
            bar_close_pct_body=float(r["bar_close_pct_body"]),
            sl_dist=float(r["sl_dist_pts"]),
            tp_dist=float(r["tp_dist_pts"]),
            session=str(r["session"]),
            mkt_regime=str(r["mkt_regime"] or ""),
            et_hour=int(r["et_hour"]),
        )
        if ml_score is None:
            skipped += 1
            continue
        ml_p_wait, ml_thr = ml_score
        ml_mode = FILL_WAIT if ml_p_wait >= ml_thr else FILL_IMMEDIATE

        enriched.append(
            {
                "entry_time": r["entry_time"],
                "strategy": strategy,
                "side": r["side"],
                "session": str(r["session"]),
                "mkt_regime": str(r["mkt_regime"] or ""),
                "imm_pnl_dol": float(r["imm_pnl_dol"]),
                "wait_pnl_dol": float(r["wait_pnl_dol"]),
                "rule_fill_mode": rule_mode,
                "rule_reason": str(rule_decision.get("reason", "")),
                "ml_fill_mode": ml_mode,
                "ml_p_wait": float(ml_p_wait),
                "ml_threshold": float(ml_thr),
                "gate_p_big_loss": float(gate_p),
                "gate_threshold": float(gate_thr),
                "gate_veto": gate_veto,
            }
        )

    print(f"  enriched_rows={len(enriched)}  skipped={skipped}")
    if not enriched:
        raise SystemExit("no enriched rows")

    policies = [
        ("ungated_immediate", False, "immediate"),
        ("ungated_rule_lfo", False, "rule_lfo"),
        ("ungated_ml_lfo", False, "ml_lfo"),
        ("ungated_live_hybrid_lfo", False, "live_hybrid_lfo"),
        ("gated_immediate", True, "immediate"),
        ("gated_rule_lfo", True, "rule_lfo"),
        ("gated_ml_lfo", True, "ml_lfo"),
        ("gated_live_hybrid_lfo", True, "live_hybrid_lfo"),
    ]

    combo_results: Dict[str, Any] = {}
    for name, use_gate, fill_policy in policies:
        combo_rows = []
        for row in enriched:
            blocked = bool(use_gate and row["gate_veto"])
            final_mode = combo_fill_mode(row, fill_policy)
            realized = 0.0 if blocked else pnl_for_mode(row, final_mode)
            combo_rows.append(
                {
                    **row,
                    "blocked_by_gate": blocked,
                    "final_fill_mode": final_mode,
                    "realized_pnl_dol": realized,
                }
            )

        overall = summarize_rows(combo_rows, "realized_pnl_dol")
        by_strategy = {}
        for strategy in sorted({r["strategy"] for r in combo_rows}):
            sub = [r for r in combo_rows if r["strategy"] == strategy]
            by_strategy[strategy] = summarize_rows(sub, "realized_pnl_dol")
        combo_results[name] = {
            "overall": overall,
            "by_strategy": by_strategy,
        }

    baseline_key = "ungated_immediate"
    baseline_pnl = combo_results[baseline_key]["overall"]["pnl"]
    baseline_dd = combo_results[baseline_key]["overall"]["dd"]
    ranking = []
    for name, result in combo_results.items():
        overall = result["overall"]
        ranking.append(
            {
                "combo": name,
                "pnl": overall["pnl"],
                "dd": overall["dd"],
                "delta_vs_ungated_immediate": overall["pnl"] - baseline_pnl,
                "dd_change_vs_ungated_immediate": overall["dd"] - baseline_dd,
            }
        )
    ranking.sort(key=lambda x: (-x["pnl"], x["dd"], x["combo"]))

    result = {
        "window": {
            "entry_min": min(r["entry_time"] for r in enriched),
            "entry_max": max(r["entry_time"] for r in enriched),
            "n_rows": len(enriched),
        },
        "notes": [
            "Common comparison universe = overlap of LFO dataset rows and rows with corrected gate context.",
            "Rule LFO is reconstructed from historical bars using BankLevelQuarterFilter + StructuralLevelTracker + LevelFillOptimizer.evaluate().",
            "ML LFO uses the current live-loaded artifact model_lfo.joblib.",
            "live_hybrid_lfo matches current bot wiring: rule LFO can WAIT, ML only cancels WAIT to IMMEDIATE.",
            "Gate path uses corrected runtime semantics with static thresholds (dynamic thresholding off by default).",
        ],
        "combo_results": combo_results,
        "ranking_by_pnl": ranking,
        "diagnostics": {
            "rule_mode_counts": pd.Series([r["rule_fill_mode"] for r in enriched]).value_counts().to_dict(),
            "ml_mode_counts": pd.Series([r["ml_fill_mode"] for r in enriched]).value_counts().to_dict(),
            "gate_veto_counts": {
                "vetoed": int(sum(1 for r in enriched if r["gate_veto"])),
                "kept": int(sum(1 for r in enriched if not r["gate_veto"])),
            },
        },
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    print(f"[write] {OUT_PATH}")

    print("\nTop combos by total PnL:")
    for row in ranking[:8]:
        print(
            f"  {row['combo']:<24} pnl=${row['pnl']:+,.2f} "
            f"dd=${row['dd']:,.2f} "
            f"delta_vs_base=${row['delta_vs_ungated_immediate']:+,.2f}"
        )


if __name__ == "__main__":
    main()

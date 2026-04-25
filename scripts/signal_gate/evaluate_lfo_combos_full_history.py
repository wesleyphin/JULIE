"""Full-history gate + fill-optimization combo evaluation.

Uses the exact full-history source reports referenced by
artifacts/signal_gate_2025/strategy_artifact_compare_full_history_20260424.json
for:
  - DynamicEngine3
  - AetherFlow
  - RegimeAdaptive

For each trade we:
  1. infer the actual front-month contract symbol from the entry price/time
  2. reconstruct rule-based LFO state from the true bar stream
  3. simulate IMMEDIATE vs generic WAIT outcomes (same methodology as LFO training)
  4. score the corrected per-strategy G gate
  5. score ML LFO
  6. compare the combo matrix on one common full-history trade universe

The PnL numbers are expressed with the same per-contract dollar convention used
by the LFO trainer: pnl_points * $5.00. This keeps the WAIT-vs-IMMEDIATE
evaluation aligned with the saved ML LFO threshold.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
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
COMPARE_PATH = ART_DIR / "strategy_artifact_compare_full_history_20260424.json"
OUT_PATH = ART_DIR / f"lfo_combo_matrix_full_history_{datetime.now().strftime('%Y%m%d')}.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))
sys.path.insert(0, str(ROOT / "tools"))

from bank_level_quarter_filter import BankLevelQuarterFilter  # noqa: E402
from build_de3_chosen_shape_dataset import (  # noqa: E402
    ENTRY_SHAPE_COLUMNS,
    _compute_feature_frame,
)
from level_fill_optimizer import (  # noqa: E402
    FILL_AT_LEVEL,
    FILL_IMMEDIATE,
    FILL_WAIT,
    LevelFillOptimizer,
)
import ml_overlay_shadow as mls  # noqa: E402
from regime_classifier import RegimeClassifier  # noqa: E402
import signal_gate_2025 as sg  # noqa: E402
from structural_level_tracker import StructuralLevelTracker  # noqa: E402
from train_lfo_ml import POINT_VALUE, simulate_fill  # noqa: E402

NY = ZoneInfo("America/New_York")
PROBE_DELTAS = (
    pd.Timedelta(minutes=0),
    pd.Timedelta(minutes=-1),
    pd.Timedelta(minutes=1),
)


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


def normalize_strategy(strategy: Optional[str], fallback: str) -> str:
    s = str(strategy or "").strip()
    if not s:
        return fallback
    if s.startswith("DynamicEngine3"):
        return "DynamicEngine3"
    if s.startswith("AetherFlow"):
        return "AetherFlow"
    if s.startswith("RegimeAdaptive"):
        return "RegimeAdaptive"
    return fallback


def build_trade_key(trade: Dict[str, Any]) -> str:
    return "||".join(
        [
            str(trade["strategy"]),
            str(trade["entry_time"]),
            str(trade["side"]).upper(),
            f"{float(trade['entry_price']):.5f}",
            str(int(trade.get("size", 1) or 1)),
        ]
    )


def load_full_history_trades() -> List[Dict[str, Any]]:
    compare = json.loads(COMPARE_PATH.read_text())
    spec = {
        "DynamicEngine3": Path(compare["DynamicEngine3"]["source_report"]),
        "AetherFlow": Path(compare["AetherFlow"]["source_report"]),
        "RegimeAdaptive": Path(compare["RegimeAdaptive"]["source_report"]),
    }
    out: List[Dict[str, Any]] = []
    seen = set()
    for fallback, path in spec.items():
        data = json.load(open(path))
        for t in data.get("trade_log", []):
            strategy = normalize_strategy(t.get("strategy"), fallback)
            norm = dict(t)
            norm["strategy"] = strategy
            norm["side"] = str(t.get("side", "")).upper()
            norm["size"] = int(t.get("size", 1) or 1)
            norm["entry_price"] = float(t.get("entry_price") or 0.0)
            norm["sl_dist"] = float(t.get("sl_dist") or 0.0)
            norm["tp_dist"] = float(t.get("tp_dist") or 0.0)
            key = build_trade_key(norm)
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
    out.sort(key=lambda t: str(t.get("entry_time", "")))
    return out


def infer_symbol_for_trade(
    trade: Dict[str, Any],
    master_by_ts: pd.DataFrame,
) -> Optional[str]:
    ts_utc = pd.Timestamp(trade["entry_time"]).tz_convert("UTC")
    entry_price = float(trade["entry_price"])
    best_symbol = None
    best_score = float("inf")

    for delta in PROBE_DELTAS:
        probe = ts_utc + delta
        try:
            rows = master_by_ts.loc[probe]
        except KeyError:
            continue
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        rows = rows.copy()
        rows["score"] = np.minimum(
            (rows["open"].astype(float) - entry_price).abs(),
            (rows["close"].astype(float) - entry_price).abs(),
        )
        rows = rows.sort_values(["score", "volume"], ascending=[True, False])
        candidate = rows.iloc[0]
        score = float(candidate["score"])
        if score < best_score:
            best_score = score
            best_symbol = str(candidate["symbol"])
        if best_score <= 1.0:
            break

    if best_symbol is not None:
        return best_symbol

    window = master_by_ts.loc[ts_utc - pd.Timedelta(minutes=1): ts_utc + pd.Timedelta(minutes=1)].copy()
    if window.empty:
        return None
    window["score"] = np.minimum(
        (window["open"].astype(float) - entry_price).abs(),
        (window["close"].astype(float) - entry_price).abs(),
    )
    window = window.sort_values(["score", "volume"], ascending=[True, False])
    return str(window.iloc[0]["symbol"])


def compute_day_state(trades: List[Dict[str, Any]]) -> Dict[str, tuple[float, int, int]]:
    by_strategy_day: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        et = datetime.fromisoformat(str(t["entry_time"])).astimezone(NY)
        by_strategy_day[(t["strategy"], et.date().isoformat())].append(t)

    state: Dict[str, tuple[float, int, int]] = {}
    for _, day_trades in by_strategy_day.items():
        day_trades.sort(key=lambda x: str(x.get("entry_time", "")))
        cum = 0.0
        consec = 0
        count = 0
        for t in day_trades:
            key = build_trade_key(t)
            state[key] = (cum, consec, count)
            pnl = float(t.get("pnl_dollars", t.get("pnl_net", 0.0)) or 0.0)
            cum += pnl
            if pnl < 0:
                consec += 1
            elif pnl > 0:
                consec = 0
            count += 1
    return state


def simulate_wait_bracket_from_fill(
    bars: pd.DataFrame,
    fill_idx: int,
    fill_px: float,
    side: str,
    tp_dist: float,
    sl_dist: float,
    max_bars: int = 100,
) -> float:
    if fill_idx < 0:
        return 0.0
    if fill_idx + 1 >= len(bars):
        return 0.0
    if side == "LONG":
        tp_price = fill_px + tp_dist
        sl_price = fill_px - sl_dist
    else:
        tp_price = fill_px - tp_dist
        sl_price = fill_px + sl_dist
    end = min(len(bars), fill_idx + max_bars)
    for j in range(fill_idx, end):
        h = float(bars.iloc[j]["high"])
        l = float(bars.iloc[j]["low"])
        sl_hit = (side == "LONG" and l <= sl_price) or (side == "SHORT" and h >= sl_price)
        tp_hit = (side == "LONG" and h >= tp_price) or (side == "SHORT" and l <= tp_price)
        if sl_hit:
            return -sl_dist
        if tp_hit:
            return tp_dist
    exit_price = float(bars.iloc[end - 1]["close"])
    return (exit_price - fill_px) if side == "LONG" else (fill_px - exit_price)


def precompute_symbol_context(symbol_bars: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    feats = _compute_feature_frame(symbol_bars)
    clf = RegimeClassifier()
    labels: List[str] = []
    last = ""
    for ts, close in symbol_bars["close"].items():
        try:
            r = clf.update(ts, float(close))
            if r and r != "warmup":
                last = str(r)
        except Exception:
            pass
        labels.append(last)
    return feats, labels


def build_full_history_rows(
    trades: List[Dict[str, Any]],
    master: pd.DataFrame,
) -> List[Dict[str, Any]]:
    print("[infer] contract symbol per trade")
    master_lookup = master[["symbol", "open", "close", "volume"]]
    for i, trade in enumerate(trades):
        trade["symbol"] = infer_symbol_for_trade(trade, master_lookup)
        if (i + 1) % 5000 == 0:
            print(f"  inferred {i+1}/{len(trades)}")

    trades = [t for t in trades if t.get("symbol")]
    print(f"  inferred symbols for {len(trades)} trades")

    day_state = compute_day_state(trades)
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trades:
        by_symbol[str(t["symbol"])].append(t)

    all_rows: List[Dict[str, Any]] = []
    for symbol, sym_trades in sorted(by_symbol.items()):
        bars = master.loc[master["symbol"] == symbol, ["open", "high", "low", "close", "volume"]].copy()
        if len(bars) < 500:
            continue
        feats, regime_labels = precompute_symbol_context(bars)

        sym_trades.sort(key=lambda t: str(t["entry_time"]))
        optimizer = LevelFillOptimizer()
        bank_filter = BankLevelQuarterFilter()
        structural_tracker = StructuralLevelTracker()
        last_updated = -1

        print(f"[symbol] {symbol} trades={len(sym_trades)} bars={len(bars)}")
        for trade in sym_trades:
            entry_ts = pd.Timestamp(trade["entry_time"]).tz_convert("UTC")
            idx = bars.index.searchsorted(entry_ts)
            sig_idx = idx - 2
            if sig_idx < 1 or sig_idx >= len(bars):
                continue

            for j in range(last_updated + 1, sig_idx + 1):
                bar = bars.iloc[j]
                ts_et = pd.Timestamp(bars.index[j]).tz_convert(NY).to_pydatetime()
                bank_filter.update(ts_et, float(bar["high"]), float(bar["low"]), float(bar["close"]))
                structural_tracker.update(
                    ts_et,
                    float(bar["open"]),
                    float(bar["high"]),
                    float(bar["low"]),
                    float(bar["close"]),
                )
            last_updated = sig_idx

            feat_row = feats.iloc[min(sig_idx, len(feats) - 1)]
            if feat_row.isna().all():
                continue

            sig_bar = bars.iloc[sig_idx]
            side = str(trade["side"]).upper()
            current_close = float(sig_bar["close"])
            bar_hi = float(sig_bar["high"])
            bar_lo = float(sig_bar["low"])
            sl_dist = float(trade.get("sl_dist") or 0.0)
            tp_dist = float(trade.get("tp_dist") or 0.0)
            if sl_dist <= 0.0:
                sl_dist = 10.0
            if tp_dist <= 0.0:
                tp_dist = 25.0

            rule_decision = optimizer.evaluate(
                signal={"side": side, "sl_dist": sl_dist, "tp_dist": tp_dist},
                current_price=current_close,
                structural_tracker=structural_tracker,
                bank_filter=bank_filter,
                bar_candle={
                    "open": float(sig_bar["open"]),
                    "high": bar_hi,
                    "low": bar_lo,
                    "close": current_close,
                },
            )
            rule_mode = str(rule_decision.get("mode", FILL_IMMEDIATE))
            if rule_mode not in {FILL_IMMEDIATE, FILL_AT_LEVEL, FILL_WAIT}:
                rule_mode = FILL_IMMEDIATE

            local_start = max(0, sig_idx - 10)
            local_end = min(len(bars), sig_idx + 140)
            local = bars.iloc[local_start:local_end].copy()
            local_sig_idx = sig_idx - local_start

            imm_fill_idx, _ = simulate_fill(
                local,
                local_sig_idx,
                side,
                float(trade["entry_price"]),
                sl_dist,
                tp_dist,
                "IMMEDIATE",
            )
            wait_fill_idx, wait_fill_px = simulate_fill(
                local,
                local_sig_idx,
                side,
                float(trade["entry_price"]),
                sl_dist,
                tp_dist,
                "WAIT",
            )
            if imm_fill_idx < 0:
                continue
            imm_entry_px = float(local.iloc[imm_fill_idx]["open"])
            imm_pnl_pts = simulate_wait_bracket_from_fill(local, imm_fill_idx, imm_entry_px, side, tp_dist, sl_dist)
            wait_pnl_pts = simulate_wait_bracket_from_fill(local, wait_fill_idx, wait_fill_px, side, tp_dist, sl_dist)

            et = datetime.fromisoformat(str(trade["entry_time"])).astimezone(NY)
            dist_below = current_close - ((current_close // 12.5) * 12.5)
            dist_above = (((current_close // 12.5) + 1) * 12.5) - current_close
            dist_in_dir = dist_below if side == "LONG" else dist_above
            atr14 = float(feat_row.get("de3_entry_atr14", 0.0) or 0.0)
            ret1_raw = feat_row.get("de3_entry_ret1_atr", 0.0)
            try:
                ret1 = float(ret1_raw)
                if not np.isfinite(ret1):
                    ret1 = 0.0
            except Exception:
                ret1 = 0.0
            if ret1 > 0:
                trend_align = 1.0 if side == "LONG" else -1.0
            elif ret1 < 0:
                trend_align = 1.0 if side == "SHORT" else -1.0
            else:
                trend_align = 0.0

            regime_raw = trade.get("regime") or trade.get("aetherflow_regime") or ""
            regime = str(regime_raw or "").strip().upper()
            if regime and regime not in {"DISPERSED", "TREND_GEODESIC", "CHOP_SPIRAL", "ROTATIONAL_TURBULENCE"}:
                regime = ""

            key = build_trade_key(trade)
            cum_pnl, consec_losses, count_prior = day_state.get(key, (0.0, 0, 0))

            row = {
                "join_key": key,
                "strategy": trade["strategy"],
                "symbol": symbol,
                "entry_time": trade["entry_time"],
                "side": side,
                "entry_price": float(trade["entry_price"]),
                "size": int(trade["size"]),
                "session": str(trade.get("session") or ("ASIA" if et.hour < 3 or et.hour >= 18 else "LONDON" if et.hour < 7 else "NY_PRE" if et.hour < 9 else "NY" if et.hour < 16 else "POST")),
                "et_hour": int(et.hour),
                "mkt_regime": regime_labels[sig_idx] or "",
                "regime": regime,
                "cum_day_pnl_pre_trade": float(cum_pnl),
                "consec_losses_pre_trade": int(consec_losses),
                "trades_today_pre_trade": int(count_prior),
                "trend_align_ret1": trend_align,
                "dist_to_bank_below": float(dist_below),
                "dist_to_bank_above": float(dist_above),
                "dist_to_bank_in_dir": float(dist_in_dir),
                "bar_range_pts": float(bar_hi - bar_lo),
                "bar_close_pct_body": float((current_close - bar_lo) / max(0.01, bar_hi - bar_lo)),
                "sl_dist_pts": float(sl_dist),
                "tp_dist_pts": float(tp_dist),
                "atr_ratio_to_sl": float(atr14 / max(0.5, sl_dist)),
                "imm_pnl_dol": float(imm_pnl_pts * POINT_VALUE),
                "wait_pnl_dol": float(wait_pnl_pts * POINT_VALUE),
                "rule_fill_mode": rule_mode,
                "rule_reason": str(rule_decision.get("reason", "")),
            }
            for c in ENTRY_SHAPE_COLUMNS:
                row[c] = float(feat_row.get(c, 0.0) or 0.0)
            all_rows.append(row)

    return all_rows


def gate_payload_for_strategy(strategy: str, payloads: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    family = sg._strategy_family(strategy)  # pylint: disable=protected-access
    return payloads.get(family)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but GradientBoostingClassifier was fitted with feature names",
        category=UserWarning,
    )

    print(f"[load] {COMPARE_PATH}")
    trades = load_full_history_trades()
    print(f"  trades={len(trades)}")
    print(f"  window={trades[0]['entry_time']} -> {trades[-1]['entry_time']}")

    print(f"[load] {PARQUET}")
    master = pd.read_parquet(PARQUET, columns=["symbol", "open", "high", "low", "close", "volume"])
    print(f"  bars={len(master):,}")

    print("[build] full-history feature + LFO rows")
    rows = build_full_history_rows(trades, master)
    print(f"  built rows={len(rows)}")
    if not rows:
        raise SystemExit("no rows built")

    gate_payloads = {
        "de3": joblib.load(ART_DIR / "model_de3.joblib"),
        "aetherflow": joblib.load(ART_DIR / "model_aetherflow.joblib"),
        "regimeadaptive": joblib.load(ART_DIR / "model_regimeadaptive.joblib"),
    }
    mls._LFO_PAYLOAD = joblib.load(ART_DIR / "model_lfo.joblib")  # pylint: disable=protected-access

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        gate_payload = gate_payload_for_strategy(row["strategy"], gate_payloads)
        if gate_payload is None:
            continue
        gate_p = sg._score_with_gate(  # pylint: disable=protected-access
            gate_payload,
            side=row["side"],
            regime=row["regime"],
            mkt_regime=row["mkt_regime"],
            et_hour=int(row["et_hour"]),
            bar_features=row,
        )
        if gate_p is None:
            continue
        ml_score = mls.score_lfo(
            signal={"side": row["side"]},
            bar_features=row,
            dist_to_bank_below=float(row["dist_to_bank_below"]),
            dist_to_bank_above=float(row["dist_to_bank_above"]),
            bar_range_pts=float(row["bar_range_pts"]),
            bar_close_pct_body=float(row["bar_close_pct_body"]),
            sl_dist=float(row["sl_dist_pts"]),
            tp_dist=float(row["tp_dist_pts"]),
            session=str(row["session"]),
            mkt_regime=str(row["mkt_regime"] or ""),
            et_hour=int(row["et_hour"]),
        )
        if ml_score is None:
            continue
        ml_p_wait, ml_thr = ml_score
        ml_mode = FILL_WAIT if ml_p_wait >= ml_thr else FILL_IMMEDIATE
        enriched.append(
            {
                **row,
                "gate_p_big_loss": float(gate_p),
                "gate_threshold": float(gate_payload.get("veto_threshold", 1.0)),
                "gate_veto": bool(gate_p >= float(gate_payload.get("veto_threshold", 1.0))),
                "ml_p_wait": float(ml_p_wait),
                "ml_threshold": float(ml_thr),
                "ml_fill_mode": ml_mode,
            }
        )

    print(f"[score] enriched rows={len(enriched)}")
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
        combo_results[name] = {"overall": overall, "by_strategy": by_strategy}

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
            "Full-history trade universe comes from the exact source_report files used in strategy_artifact_compare_full_history_20260424.json.",
            "Contract symbol is inferred per trade from entry_time + entry_price using the raw outrights parquet.",
            "PnL is in the same per-contract dollar convention used by the LFO trainer: pnl_points * $5.00.",
            "Rule LFO uses the real LevelFillOptimizer decision reconstructed from the true bar stream, while WAIT payoff uses the same generic WAIT-vs-IMMEDIATE simulator used by the ML LFO trainer.",
            "Gate path uses corrected runtime semantics with static thresholds.",
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
            "symbol_count": int(len({r["symbol"] for r in enriched})),
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

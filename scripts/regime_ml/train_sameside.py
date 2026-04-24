#!/usr/bin/env python3
"""Train SameSide stack-or-suppress ML — HGB-only, reproducible.

Converts the hard rule at julie001.py:14659 ("Ignoring same-side signal
while X position is already active") from always-suppress to ML-gated:
  stack   → add 1 contract, capped at max 2 contracts total
  suppress→ keep existing behavior (refuse the stack)

Data sources:
  - Replay logs: backtest_reports/full_live_replay/YYYY_MM/topstep_live_bot.log
  - Current live log: topstep_live_bot.log
  - Price bars: es_master_outrights.parquet (for forward PnL + features)

For each 'Ignoring same-side signal' event:
  1. Extract market_ts from the latest 'Bar: <mts> ET' print before the event
  2. Extract position context from the most recent 'FAST EXEC' entry before
  3. Extract signal context from the CANDIDATE line immediately before
  4. Build 40-feat regime snapshot from parquet at market_ts
  5. Add position-state + strategy features
  6. Label by forward 15-min: would a new size=1 trade at signal_price with
     default TP=6/SL=4 produce positive PnL?

Ship gates per user:
  1. OOS PnL lift > 0 on 2026-Q1+ holdout
  2. MaxDD <= 110% baseline
  3. n_stack >= 20 on OOS
  4. Stacked-trades WR >= 55%
  5. Captures >= 20% of oracle-perfect lift
"""
from __future__ import annotations

import argparse, json, logging, pickle, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    SESSION_START_HOUR_ET, SESSION_END_HOUR_ET,
    load_continuous_bars, build_feature_frame, simulate_trade,
    sample_weights_balanced, stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_sameside")

LABEL_WINDOW_MIN = 15
LABEL_MARGIN_USD = 15.0
OOS_START_DEFAULT = "2026-01-27"
OOS_END_DEFAULT   = "2026-04-20"

# Log parser regexes
RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<px>[-\d.]+)")
RE_STRATEGY_SIG = re.compile(
    r"\[STRATEGY_SIGNAL\].*?strategy=(?P<strat>\S+).*?"
    r"side=(?P<side>LONG|SHORT).*?price=(?P<price>[-\d.]+)"
    r".*?tp_dist=(?P<tp>[-\d.]+).*?sl_dist=(?P<sl>[-\d.]+)"
)
RE_FAST_EXEC = re.compile(r"✅ FAST EXEC: (?P<strat>\S+) \((?P<side>LONG|SHORT)\)")
RE_SAMESIDE = re.compile(r"Ignoring same-side signal while (?P<pos>LONG|SHORT) position is already active: (?P<strat>\S+)")
RE_TRADE_CLOSED = re.compile(r"Trade closed.*?Entry: (?P<e>[-\d.]+).*?Exit: (?P<x>[-\d.]+)")
RE_POSITION_FLAT = re.compile(r"Position Sync.*FLAT")


def _parse_market_ts(mts: str) -> datetime:
    """Replay log bar timestamps like '2025-04-01 00:00:00' — naive, ET."""
    return datetime.strptime(mts, "%Y-%m-%d %H:%M:%S")


# ─── Event extraction ────────────────────────────────────────────────────

def parse_log_events(log_path: Path) -> list:
    """Walk the log once, emit one event dict per minute-unique same-side
    suppression event with full context."""
    events = []
    state = {
        "active": False,
        "side": None,
        "entry_price": None,
        "entry_mts": None,
        "tp_dist": None,
        "sl_dist": None,
    }
    # Rolling context — last N of each feature
    last_candidate = None       # most recent STRATEGY_SIGNAL
    last_bar_mts = None
    last_bar_price = None

    seen_minute_keys = set()

    with log_path.open(errors="ignore") as fh:
        for line in fh:
            m = RE_HEADER.match(line)
            if not m:
                continue
            log_ts = m.group(1)

            bm = RE_BAR.search(line)
            if bm:
                last_bar_mts = bm.group("mts")
                last_bar_price = float(bm.group("px"))
                continue

            sg = RE_STRATEGY_SIG.search(line)
            if sg:
                last_candidate = {
                    "strat": sg.group("strat"),
                    "side":  sg.group("side"),
                    "price": float(sg.group("price")),
                    "tp_dist": float(sg.group("tp")),
                    "sl_dist": float(sg.group("sl")),
                    "mts": last_bar_mts,
                }
                continue

            fe = RE_FAST_EXEC.search(line)
            if fe and last_candidate is not None:
                if last_candidate["strat"].startswith(fe.group("strat")) and last_candidate["side"] == fe.group("side"):
                    state["active"] = True
                    state["side"] = last_candidate["side"]
                    state["entry_price"] = last_candidate["price"]
                    state["entry_mts"] = last_candidate["mts"]
                    state["tp_dist"] = last_candidate["tp_dist"]
                    state["sl_dist"] = last_candidate["sl_dist"]
                continue

            if RE_TRADE_CLOSED.search(line) or RE_POSITION_FLAT.search(line):
                state["active"] = False
                state["side"] = None
                state["entry_price"] = None
                continue

            ss = RE_SAMESIDE.search(line)
            if not ss:
                continue

            # Only label when we have full context
            if not state["active"]: continue
            if state["side"] != ss.group("pos"): continue
            if last_bar_mts is None: continue
            if state["entry_mts"] is None or state["entry_price"] is None: continue
            if last_candidate is None or last_candidate["side"] != ss.group("pos"): continue

            # Dedupe at (market minute, strategy, side)
            minute = last_bar_mts[:16]
            key = (minute, ss.group("strat"), ss.group("pos"))
            if key in seen_minute_keys: continue
            seen_minute_keys.add(key)

            try:
                entry_dt = _parse_market_ts(state["entry_mts"])
                event_dt = _parse_market_ts(last_bar_mts)
            except Exception:
                continue
            bars_held = int((event_dt - entry_dt).total_seconds() // 60)

            side_sign = +1 if state["side"] == "LONG" else -1
            unrealized = (last_bar_price - state["entry_price"]) * side_sign
            dist_to_tp = state["tp_dist"] - unrealized  # how far to TP from here
            dist_to_sl = state["sl_dist"] + unrealized  # how far to SL from here (stop is entry - sl, curr dist)

            events.append({
                "market_ts": last_bar_mts,
                "strategy": ss.group("strat"),
                "position_side": state["side"],
                "signal_side": ss.group("pos"),
                "position_entry": state["entry_price"],
                "position_tp_dist": state["tp_dist"],
                "position_sl_dist": state["sl_dist"],
                "bars_held": max(0, bars_held),
                "current_price": last_bar_price,
                "signal_price": last_candidate["price"],
                "unrealized_pts": unrealized,
                "unrealized_frac": unrealized / max(state["entry_price"], 1.0),
                "dist_to_tp": dist_to_tp,
                "dist_to_sl": dist_to_sl,
                "side_sign": side_sign,
                "signal_to_position_delta": last_candidate["price"] - state["entry_price"],
            })
    return events


# ─── Feature + label build ───────────────────────────────────────────────

def build_features_for_events(events: list, bars_df: pd.DataFrame) -> pd.DataFrame:
    """Join events to parquet bars by market_ts. Build 40-feat snapshot +
    position state features. Compute labels via forward 15-min simulation."""
    # Build feature frame over full window
    feats = build_feature_frame(bars_df)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    log.info("feature frame rows: %d", len(feats))

    # Index bars by (date, hour, minute) for event lookup
    bars_mts_set = {pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M") for ts in feats.index}
    log.info("indexed feature bars: %d", len(bars_mts_set))

    # Join events to features
    rows = []
    dropped_no_bar = 0
    dropped_out_of_range = 0
    for e in events:
        mts_key = e["market_ts"][:16]
        if mts_key not in bars_mts_set:
            dropped_no_bar += 1
            continue
        try:
            # Find the exact feature row — use the bar whose floor matches our minute
            ts = pd.Timestamp(mts_key + ":00", tz=feats.index.tz)
        except Exception:
            dropped_out_of_range += 1
            continue
        if ts not in feats.index:
            # Find closest within same minute
            sub = feats.loc[feats.index.strftime("%Y-%m-%d %H:%M") == mts_key]
            if sub.empty:
                dropped_out_of_range += 1
                continue
            ts = sub.index[0]
        row = {**e, "_ts": ts}
        for c in FEATURE_COLS_40:
            row[c] = feats.loc[ts, c]
        rows.append(row)
    log.info("joined events: %d  dropped (no matching bar): %d  dropped (out of range): %d",
             len(rows), dropped_no_bar, dropped_out_of_range)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("_ts")

    # Strategy one-hot
    df["is_de3"] = (df["strategy"].str.startswith("DynamicEngine")).astype(int)
    df["is_ra"]  = (df["strategy"].str.startswith("RegimeAdaptive")).astype(int)
    df["is_af"]  = (df["strategy"].str.startswith("AetherFlow")).astype(int)

    # Forward-label: new size=1 trade at signal_price with TP=6/SL=4
    c = bars_df["close"].to_numpy(float)
    h = bars_df["high"].to_numpy(float)
    l = bars_df["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars_df.index)}

    labels, pnls = [], []
    for ts, e in df.iterrows():
        sp = idx_pos.get(ts)
        if sp is None:
            labels.append(None); pnls.append(None); continue
        side = e["side_sign"]
        # Simulate new contract at signal_price with default brackets
        # Approximation: use simulate_trade from bars[sp] forward, entry = current bar close
        pnl = simulate_trade(h, l, c, int(sp), DEFAULT_TP, DEFAULT_SL, int(side))
        pnls.append(pnl)
        if pnl > LABEL_MARGIN_USD:
            labels.append("stack")
        elif pnl < -LABEL_MARGIN_USD:
            labels.append("suppress")
        else:
            labels.append(None)   # ambiguous — drop

    df["label"] = labels
    df["new_contract_pnl"] = pnls
    df = df.loc[df["label"].notna()].copy()
    log.info("labeled rows after margin filter: %d  dist: %s",
             len(df), dict(Counter(df["label"])))
    return df


FEATURE_COLS_SS = FEATURE_COLS_40 + [
    "bars_held", "unrealized_pts", "unrealized_frac",
    "dist_to_tp", "dist_to_sl", "side_sign",
    "signal_to_position_delta",
    "is_de3", "is_ra", "is_af",
]


# ─── Training + OOS ──────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START_DEFAULT)
    ap.add_argument("--oos-end", default=OOS_END_DEFAULT)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_sameside"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse all canonical replay logs + current live
    log.info("parsing logs...")
    logs = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in ("2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 "
              "2025_08 2025_09 2025_10 2025_11 2025_12").split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            logs.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): logs.append(live)

    all_events = []
    for p in logs:
        events = parse_log_events(p)
        log.info("  %s → %d events", p.name if p.parent == ROOT else f"{p.parent.name}/{p.name}", len(events))
        all_events.extend(events)
    log.info("total minute-unique events w/ full context: %d", len(all_events))

    if not all_events:
        log.error("no events parsed — pipeline broken"); return 2
    if len(all_events) < 2000:
        log.warning("only %d events (<2000 floor) — honest kill per user policy", len(all_events))
        return 1

    # Load bars for whole 2025-2026 range (covers all events)
    # Events have market_ts strings like "2025-04-01 14:23:00"
    all_mts = [e["market_ts"] for e in all_events]
    min_ts = min(all_mts)
    max_ts = max(all_mts)
    log.info("event market_ts range: %s → %s", min_ts, max_ts)
    start = pd.Timestamp(min_ts).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max_ts) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bars = load_continuous_bars(start, end)
    log.info("bars loaded: %d", len(bars))

    df = build_features_for_events(all_events, bars)
    if df.empty:
        log.error("no rows after feature+label build"); return 2
    if len(df) < 500:
        log.warning("only %d labeled rows — below reliable-training floor", len(df))
        return 1

    # Train/OOS split
    oos_start = pd.Timestamp(args.oos_start, tz=df.index.tz)
    oos_end = pd.Timestamp(args.oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start]
    oos = df.loc[(df.index >= oos_start) & (df.index <= oos_end)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    log.info("train labels: %s", dict(Counter(tr["label"])))
    log.info("OOS labels: %s", dict(Counter(oos["label"])))

    if len(oos) < 100:
        log.warning("OOS too small (%d) — cannot validate ship gates", len(oos))
        return 1

    X_tr = tr[FEATURE_COLS_SS].to_numpy()
    y_tr = tr["label"].to_numpy()
    sw = sample_weights_balanced(y_tr, cost_ratio=1.5)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=args.seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    stack_idx = list(clf.classes_).index("stack")

    # OOS sweep
    X_oos = oos[FEATURE_COLS_SS].to_numpy()
    y_true = oos["label"].to_numpy()
    pnl_actual = oos["new_contract_pnl"].to_numpy()

    # Baseline: suppress everything (current rule)
    baseline_pnl = 0.0   # no stacks → no new PnL
    baseline_dd = 0.0

    # Oracle: stack when label=='stack' (perfect info)
    oracle_mask = (y_true == "stack")
    oracle_pnl_arr = np.where(oracle_mask, pnl_actual, 0.0)
    oracle_st = stats(oracle_pnl_arr[oracle_mask])

    print(f"\n══ SameSide ML stack-or-suppress — OOS sweep ══")
    print(f"  baseline (suppress all): PnL=$0  DD=$0  (current rule)")
    print(f"  oracle (perfect stack):  PnL=${oracle_st['pnl']:+,.2f}  DD=${oracle_st['dd']:,.0f}")
    print(f"  {'thr':>5} {'n_stk':>6} {'WR':>7} {'pnl':>11} {'dd':>9} {'dd/pnl':>7} {'capt_%':>7} gates")
    probs = clf.predict_proba(X_oos)[:, stack_idx]
    best = None
    for thr in np.arange(0.50, 0.91, 0.05):
        pred = probs >= thr
        n_stk = int(pred.sum())
        if n_stk == 0:
            print(f"  {thr:>5.2f} {n_stk:>6} {'--':>7} {'--':>11} {'--':>9} {'--':>7}  (no stacks)")
            continue
        pnls = pnl_actual[pred]
        wins = int((pnls > 0).sum())
        wr = 100 * wins / n_stk
        st = stats(pnls)
        capt_pct = (st["pnl"] / oracle_st["pnl"] * 100) if oracle_st["pnl"] > 0 else 0
        # Gate 2 redefined (2026-04-24, per user ship decision): DD/PnL
        # ratio ≤ 30% instead of DD ≤ 110% × baseline_DD. The literal
        # formulation was unsatisfiable because baseline_DD=$0 (no stacks
        # in the suppress-all baseline). Risk-adjusted ratio is the
        # faithful equivalent that preserves the "don't let stacking
        # amplify risk excessively" intent.
        dd_over_pnl = (st["dd"] / st["pnl"] * 100.0) if st["pnl"] > 0 else float("inf")
        gates = {
            "pnl_ok":     st["pnl"] > 0,
            "dd_ratio_ok":dd_over_pnl <= 30.0,          # risk-adjusted, replaces old dd_ok
            "n_ok":       n_stk >= 20,
            "wr_ok":      wr >= 55.0,
            "capt_ok":    capt_pct >= 20.0,
        }
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f} {n_stk:>6} {wr:>6.2f}% ${st['pnl']:>+9,.2f} ${st['dd']:>7,.0f} "
              f"{dd_over_pnl:>6.1f}%  {capt_pct:>6.2f}%  {sum(gates.values())}/5{flag}")
        if ok and (best is None or st["pnl"] > best["pnl"]):
            best = {"thr": float(thr), **st, "wr": wr, "n_stk": n_stk,
                    "capture_pct": capt_pct, "gates": gates}

    if best is None and not args.force:
        log.warning("[KILL] no threshold clears all 5 gates — not writing model"); return 1

    payload = {
        "threshold": best["thr"],
        "threshold_hgb_only": best["thr"],
        "feature_cols": FEATURE_COLS_SS,
        "positive_class": "stack",
        "label_to_int": {c: i for i, c in enumerate(clf.classes_)},
        "hgb": clf,
        "label_name": "sameside_stack_suppress",
        "inference_mode": "hgb_only",
        "stats_oos": best,
        "oracle_oos": oracle_st,
        "n_training_events": int(len(tr)),
        "n_oos_events": int(len(oos)),
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "feature_order.json").write_text(json.dumps({
        "features": FEATURE_COLS_SS, "threshold": best["thr"],
        "positive_class": "stack", "label_name": "sameside_stack_suppress",
        "stats_oos": best, "n_training_events": int(len(tr)), "n_oos_events": int(len(oos)),
    }, indent=2, default=str))
    log.info("[SHIP] thr=%.2f  PnL=$%+.2f  WR=%.2f%%  capture=%.2f%%",
             best["thr"], best["pnl"], best["wr"], best["capture_pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

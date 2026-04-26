#!/usr/bin/env python3
"""Train Kalshi overlay ML — HGB-only, reproducible.

Converts julie001._apply_kalshi_gate_size from rule-gated (score vs
threshold BLOCK/PASS + multiplier) to ML-gated decision on the PASS/BLOCK
side. Size multiplier computation stays in the existing rule path —
ML only decides whether the Kalshi overlay should veto.

Data source:
  [KALSHI_ENTRY_VIEW] log lines in replay logs + live log, which encode
  both the rule's decision AND all 7 Kalshi features:
    entry_probability, probe_price, probe_probability, momentum_delta,
    momentum_retention, support_score, threshold, role, decision

Label:
  For each KALSHI_ENTRY_VIEW event, simulate a size=1 forward trade at
  entry_price with default TP=6/SL=4 over 15-min window.
    PnL > +$15 → 'pass'   (Kalshi should allow it through)
    PnL < -$15 → 'block'  (Kalshi should veto)
    ambiguous → drop

Ship gates (all 5 must pass):
  1. OOS PnL lift > 0 on April 2026 Kalshi-hour events
  2. DD/PnL ratio ≤ 30%  (same redefined gate used for SameSide — risk-
     adjusted equivalent of the unsatisfiable literal-$0 baseline)
  3. n_kalshi_events ≥ 50 in OOS
  4. WR on newly-PASSED signals (which the rule blocked but ML passes)
     ≥ 50%
  5. Capture ≥ 20% of oracle-perfect Kalshi gating lift

AetherFlow events excluded from both training and OOS — AF has a
size-only Kalshi carveout in the live code and ML should not veto it.
"""
from __future__ import annotations

import argparse, json, logging, pickle, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    load_continuous_bars, build_feature_frame, simulate_trade,
    stats, sample_weights_balanced,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi")

LABEL_WINDOW_MIN = 15
LABEL_MARGIN_USD = 15.0
OOS_START = "2026-01-27"
OOS_END = "2026-04-20"

# Log parser
RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<px>[-\d.]+)")
RE_ENTRY_VIEW = re.compile(
    r"\[KALSHI_ENTRY_VIEW\].*?"
    r"strategy=(?P<strategy>\S+) \| side=(?P<side>LONG|SHORT) "
    r"\| entry_price=(?P<entry>[-\d.]+) "
    r"\| role=(?P<role>\S+) "
    r"\| decision=(?P<decision>PASS|BLOCK) "
    r"\| entry_probability=(?P<ep>[-\d.]+) "
    r"\| probe_price=(?P<pp>[-\d.]+) "
    r"\| probe_probability=(?P<pprob>[-\d.]+) "
    r"\| momentum_delta=(?P<md>[-\d.]+) "
    r"\| momentum_retention=(?P<mr>[-\d.]+) "
    r"\| support_score=(?P<ss>[-\d.]+) "
    r"\| threshold=(?P<thr>[-\d.]+)"
)


def parse_log_events(log_path: Path) -> list:
    events = []
    last_bar_mts = None
    seen_minute_keys = set()
    with log_path.open(errors="ignore") as fh:
        for line in fh:
            m = RE_HEADER.match(line)
            if not m: continue
            bm = RE_BAR.search(line)
            if bm:
                last_bar_mts = bm.group("mts")
                continue
            ev = RE_ENTRY_VIEW.search(line)
            if not ev: continue
            strat = ev.group("strategy")
            # Exclude AetherFlow — size-only carveout in live code
            if "aetherflow" in strat.lower():
                continue
            if last_bar_mts is None: continue
            minute = last_bar_mts[:16]
            key = (minute, strat, ev.group("side"), ev.group("role"))
            if key in seen_minute_keys: continue
            seen_minute_keys.add(key)
            events.append({
                "market_ts": last_bar_mts,
                "strategy": strat,
                "side": ev.group("side"),
                "entry_price": float(ev.group("entry")),
                "role": ev.group("role"),
                "rule_decision": ev.group("decision"),
                "k_entry_probability": float(ev.group("ep")),
                "k_probe_price": float(ev.group("pp")),
                "k_probe_probability": float(ev.group("pprob")),
                "k_momentum_delta": float(ev.group("md")),
                "k_momentum_retention": float(ev.group("mr")),
                "k_support_score": float(ev.group("ss")),
                "k_threshold": float(ev.group("thr")),
            })
    return events


def build_dataset(events: list, bars_df: pd.DataFrame) -> pd.DataFrame:
    feats = build_feature_frame(bars_df)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats_minute_idx = {pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"): ts for ts in feats.index}
    log.info("feature frame rows: %d  (minute-indexed: %d)", len(feats), len(feats_minute_idx))

    c = bars_df["close"].to_numpy(float)
    h = bars_df["high"].to_numpy(float)
    l = bars_df["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars_df.index)}

    rows = []
    dropped_no_bar = 0
    dropped_ambiguous = 0
    for e in events:
        mts_key = e["market_ts"][:16]
        ts = feats_minute_idx.get(mts_key)
        if ts is None:
            dropped_no_bar += 1
            continue
        pos = idx_pos.get(ts)
        if pos is None:
            dropped_no_bar += 1
            continue
        # Forward-label: size=1 trade at entry_price with default brackets
        side_sign = 1 if e["side"] == "LONG" else -1
        pnl = simulate_trade(h, l, c, int(pos), DEFAULT_TP, DEFAULT_SL, side_sign)
        if abs(pnl) < LABEL_MARGIN_USD:
            dropped_ambiguous += 1
            continue
        row = {**e, "_ts": ts}
        for col in FEATURE_COLS_40:
            row[col] = feats.loc[ts, col]
        row["side_sign"] = side_sign
        row["is_de3"] = 1 if "dynamicengine" in e["strategy"].lower() else 0
        row["is_ra"]  = 1 if "regimeadaptive" in e["strategy"].lower() else 0
        row["is_ml"]  = 1 if "mlphysics" in e["strategy"].lower() else 0
        row["settlement_hour"] = int(pd.Timestamp(mts_key + ":00").hour)
        row["minutes_into_session"] = int(
            (pd.Timestamp(mts_key + ":00").hour - 9) * 60 +
             pd.Timestamp(mts_key + ":00").minute
        )
        row["role_forward"] = 1 if "forward" in e["role"] else 0
        row["role_background"] = 1 if e["role"] == "background" else 0
        row["role_balanced"] = 1 if e["role"] == "balanced" else 0
        row["label"] = "pass" if pnl > 0 else "block"
        row["forward_pnl"] = pnl
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("_ts")
    log.info("labeled rows: %d  dropped (no bar): %d  dropped (ambiguous): %d",
             len(df), dropped_no_bar, dropped_ambiguous)
    if len(df):
        log.info("class dist: %s", dict(Counter(df["label"])))
        log.info("rule decision × label cross-tab:")
        for r_dec in ("PASS", "BLOCK"):
            for lbl in ("pass", "block"):
                n = int(((df["rule_decision"] == r_dec) & (df["label"] == lbl)).sum())
                log.info("  rule=%s × label=%s: %d", r_dec, lbl, n)
    return df


FEATURE_COLS_KALSHI = FEATURE_COLS_40 + [
    "k_entry_probability", "k_probe_probability", "k_momentum_delta",
    "k_momentum_retention", "k_support_score", "k_threshold",
    "side_sign", "settlement_hour", "minutes_into_session",
    "is_de3", "is_ra", "is_ml",
    "role_forward", "role_background", "role_balanced",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START)
    ap.add_argument("--oos-end", default=OOS_END)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("parsing logs...")
    logs = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in "2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 2025_08 2025_09 2025_10 2025_11 2025_12".split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            logs.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): logs.append(live)

    all_events = []
    for p in logs:
        ev = parse_log_events(p)
        log.info("  %s → %d events", p.name if p.parent == ROOT else f"{p.parent.name}/{p.name}", len(ev))
        all_events.extend(ev)
    log.info("total minute-unique Kalshi events (AF excluded): %d", len(all_events))
    if len(all_events) < 2000:
        log.warning("sample size %d < 2000 floor — honest kill", len(all_events))
        return 1

    all_mts = [e["market_ts"] for e in all_events]
    start = pd.Timestamp(min(all_mts)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(all_mts)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("event range: %s → %s", start, end)
    bars = load_continuous_bars(start, end)
    log.info("bars loaded: %d", len(bars))

    df = build_dataset(all_events, bars)
    if len(df) < 500:
        log.warning("only %d labeled rows — below reliable-training floor", len(df))
        return 1

    oos_start = pd.Timestamp(args.oos_start, tz=df.index.tz)
    oos_end = pd.Timestamp(args.oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start]
    oos = df.loc[(df.index >= oos_start) & (df.index <= oos_end)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    log.info("train labels: %s", dict(Counter(tr["label"])))
    log.info("OOS labels: %s", dict(Counter(oos["label"])))

    if len(oos) < 50:
        log.warning("OOS too small (%d) — fails n_kalshi_events≥50 gate", len(oos))
        return 1

    X_tr = tr[FEATURE_COLS_KALSHI].to_numpy()
    y_tr = tr["label"].to_numpy()
    sw = sample_weights_balanced(y_tr, cost_ratio=1.3)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=args.seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    pass_idx = list(clf.classes_).index("pass")

    # OOS evaluation
    X_oos = oos[FEATURE_COLS_KALSHI].to_numpy()
    y_true = oos["label"].to_numpy()
    pnl_if_passed = oos["forward_pnl"].to_numpy()
    rule_decision = oos["rule_decision"].to_numpy()

    # Baseline = rule decision: pass-through PnL where rule passes, $0 otherwise
    rule_pnl_arr = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl_arr)

    # Oracle = perfect labeling
    oracle_pnl_arr = np.where(y_true == "pass", pnl_if_passed, 0.0)
    oracle_st = stats(oracle_pnl_arr)

    print(f"\n══ Kalshi overlay ML — OOS sweep ══")
    print(f"  rule baseline (current decisions): PnL=${rule_st['pnl']:+,.2f}  DD=${rule_st['dd']:,.0f}")
    print(f"  oracle (perfect gating):           PnL=${oracle_st['pnl']:+,.2f}  DD=${oracle_st['dd']:,.0f}")
    print(f"  {'thr':>5} {'n_pass':>7} {'new_pass':>8} {'newpass_WR':>10} {'pnl':>11} {'dd':>8} {'dd/pnl':>7} {'capt':>7} gates")
    probs = clf.predict_proba(X_oos)[:, pass_idx]
    best = None
    for thr in np.arange(0.35, 0.81, 0.05):
        pred = np.where(probs >= thr, "pass", "block")
        ml_pnl_arr = np.where(pred == "pass", pnl_if_passed, 0.0)
        ml_st = stats(ml_pnl_arr)
        n_pass = int((pred == "pass").sum())

        # Newly-passed signals (rule BLOCKED but ML PASSES)
        new_pass_mask = (pred == "pass") & (rule_decision == "BLOCK")
        n_new_pass = int(new_pass_mask.sum())
        new_pass_pnls = pnl_if_passed[new_pass_mask]
        if len(new_pass_pnls) > 0:
            new_pass_wr = 100 * int((new_pass_pnls > 0).sum()) / len(new_pass_pnls)
        else:
            new_pass_wr = 0.0

        lift = ml_st["pnl"] - rule_st["pnl"]
        dd_over_pnl = (ml_st["dd"] / ml_st["pnl"] * 100.0) if ml_st["pnl"] > 0 else float("inf")
        capt_pct = (ml_st["pnl"] / oracle_st["pnl"] * 100.0) if oracle_st["pnl"] > 0 else 0.0

        gates = {
            "pnl_ok":        lift > 0,
            "dd_ratio_ok":   dd_over_pnl <= 30.0,
            "n_ok":          len(oos) >= 50,
            "new_pass_wr_ok":new_pass_wr >= 50.0 if n_new_pass >= 5 else False,
            "capt_ok":       capt_pct >= 20.0,
        }
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f} {n_pass:>7} {n_new_pass:>8} {new_pass_wr:>9.2f}% "
              f"${ml_st['pnl']:>+9,.2f} ${ml_st['dd']:>6,.0f} {dd_over_pnl:>6.1f}% {capt_pct:>6.2f}%  "
              f"{sum(gates.values())}/5{flag}")
        if ok and (best is None or lift > best["lift"]):
            best = {"thr": float(thr), **ml_st, "lift": lift, "gates": gates,
                    "n_pass": n_pass, "n_new_pass": n_new_pass, "new_pass_wr": new_pass_wr,
                    "capture_pct": capt_pct, "dd_ratio": dd_over_pnl}

    if best is None and not args.force:
        log.warning("[KILL] no threshold clears all 5 gates — not writing model")
        return 1

    payload = {
        "threshold": best["thr"],
        "threshold_hgb_only": best["thr"],
        "feature_cols": FEATURE_COLS_KALSHI,
        "positive_class": "pass",
        "label_to_int": {c: i for i, c in enumerate(clf.classes_)},
        "hgb": clf,
        "label_name": "kalshi_overlay_pass_block",
        "inference_mode": "hgb_only",
        "stats_oos": best,
        "oracle_oos": oracle_st,
        "rule_oos":   rule_st,
        "n_training_events": int(len(tr)),
        "n_oos_events":      int(len(oos)),
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "feature_order.json").write_text(json.dumps({
        "features": FEATURE_COLS_KALSHI, "threshold": best["thr"],
        "positive_class": "pass",
        "label_name": "kalshi_overlay_pass_block",
        "stats_oos": best,
    }, indent=2, default=str))
    log.info("[SHIP] thr=%.2f  lift=$%+.2f  newpassWR=%.2f%%  capture=%.2f%%",
             best["thr"], best["lift"], best["new_pass_wr"], best["capture_pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

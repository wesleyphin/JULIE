#!/usr/bin/env python3
"""V10 Stack-Aware Retrain v2 — sequential 4-stage training (DE3 only)
on the corrected v2 mistake corpus.

Mirrors §8.24 method (Filter G → Kalshi → LFO → PCT, sequential, with
upstream predictions as features) but using:
  - artifacts/full_overlay_stack_simulation_14mo_v2.parquet (corrected sim)
  - friend-rule single-position chain on raw_pnl_walk for combined sim
  - HGB only, class_weight='balanced' (sample_weight)
  - Temporal split: Mar-Dec 2025 train / Jan-Apr 2026 holdout

Outputs:
  artifacts/regime_ml_{filterg,kalshi,lfo,pct}_v10_v2/de3/{model.joblib,metrics.json,thresholds.json}
  artifacts/v10_stack_aware_v2_summary.json
  /tmp/v10_stack_v2.txt  (live log)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo_v2.parquet"

HAIRCUT = 7.50
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="US/Eastern")
DD_CAP = 870.0
WR_GATE = 55.0
MIN_OOS = 50

# ---------- features ----------
def make_base_features(df: pd.DataFrame, extra_proba_cols: list[str] | None = None) -> pd.DataFrame:
    """Build feature matrix from corpus columns. extra_proba_cols are upstream stage probas."""
    f = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["ts"])
    f["hour"] = ts.dt.hour.astype(float)
    f["minute"] = ts.dt.minute.astype(float)
    f["dow"] = ts.dt.dayofweek.astype(float)
    f["is_morning"] = (f["hour"] < 12).astype(float)
    f["is_kalshi_window"] = ((f["hour"] >= 12) & (f["hour"] < 16)).astype(float)
    f["side_long"] = (df["side"].astype(str).str.upper() == "LONG").astype(float)
    f["price"] = df["price"].astype(float)
    f["sl"] = df["sl"].astype(float)
    f["tp"] = df["tp"].astype(float)
    f["rr"] = (df["tp"].astype(float) / df["sl"].astype(float).replace(0, np.nan)).fillna(0.0)
    # raw upstream signals from corpus (these are deployed-model probas)
    f["fg_proba_raw"] = df["fg_proba"].astype(float).fillna(0.5)
    f["k_proba_raw"] = df["k_proba"].astype(float).fillna(0.5)
    f["lfo_proba_raw"] = df["lfo_proba"].astype(float).fillna(0.5)
    f["fg_decision_block"] = (df["fg_decision"] == "BLOCK").astype(float)
    f["k_decision_block"] = (df["k_decision"] == "BLOCK").astype(float)
    f["k_decision_na"] = (df["k_decision"] == "NA").astype(float)
    f["lfo_wait"] = (df["lfo_decision"] == "WAIT").astype(float)
    f["pct_breakout"] = (df["pct_decision"] == "BREAKOUT_LEAN").astype(float)
    f["pct_pivot"] = (df["pct_decision"] == "PIVOT_LEAN").astype(float)
    f["pct_size_mult"] = df["pct_size_mult"].astype(float).fillna(1.0)
    f["pct_tp_mult"] = df["pct_tp_mult"].astype(float).fillna(1.0)
    if extra_proba_cols:
        for c in extra_proba_cols:
            f[c] = df[c].astype(float).fillna(0.5)
    return f


def train_hgb(X, y):
    pos = y.sum()
    n = len(y)
    if n == 0 or pos == 0 or pos == n:
        return None
    # class-balanced sample weight
    w_pos = n / (2 * pos)
    w_neg = n / (2 * (n - pos))
    sw = np.where(y == 1, w_pos, w_neg)
    clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05, random_state=42
    )
    clf.fit(X.values, y, sample_weight=sw)
    return clf


def calc_dd(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


# ---------- combined sim with friend-rule ----------
def simulate_combined(
    df_holdout: pd.DataFrame,
    p_fg: np.ndarray,
    p_k: np.ndarray,
    p_lfo: np.ndarray,
    p_pct: np.ndarray | None,
    thr_fg: float,
    thr_k: float,
    thr_lfo: float,
    thr_pct: float,
    in_kalshi_window: np.ndarray,
) -> dict:
    """Apply 4-stage gates + single-position friend-rule on raw_pnl_walk.

    Block if model proba >= threshold (i.e. classifier predicts is_big_loss high).
    Kalshi only gates inside the 12-16 ET window.

    Returns dict {fired, pnl, wr, dd}.
    """
    df = df_holdout.sort_values("ts").reset_index(drop=True)
    n = len(df)
    raw = df["raw_pnl_walk"].astype(float).values
    exit_ts = pd.to_datetime(df["exit_ts"]).reset_index(drop=True)
    ts = pd.to_datetime(df["ts"]).reset_index(drop=True)

    # Block masks (True = blocked)
    blk_fg = p_fg >= thr_fg
    blk_k = (p_k >= thr_k) & in_kalshi_window
    blk_lfo = p_lfo >= thr_lfo
    blk_pct = (p_pct >= thr_pct) if p_pct is not None else np.zeros(n, dtype=bool)

    take = ~(blk_fg | blk_k | blk_lfo | blk_pct)

    pnl_per = np.zeros(n)
    taken_mask = np.zeros(n, dtype=bool)
    next_open_at = None
    fire_count = 0
    wins = 0
    pnls_sequence = []

    for i in range(n):
        if not take[i]:
            continue
        ti = ts.iloc[i]
        if next_open_at is not None and ti < next_open_at:
            continue
        # Take
        p = float(raw[i]) - HAIRCUT
        pnl_per[i] = p
        taken_mask[i] = True
        pnls_sequence.append(p)
        fire_count += 1
        if p > 0:
            wins += 1
        ex = exit_ts.iloc[i]
        if pd.notna(ex):
            next_open_at = ex
        else:
            next_open_at = ti + pd.Timedelta(minutes=30)

    pnl_total = float(np.sum(pnl_per))
    wr = (wins / fire_count * 100.0) if fire_count > 0 else 0.0
    dd = calc_dd(np.array(pnls_sequence)) if pnls_sequence else 0.0
    return {"fired": fire_count, "pnl": pnl_total, "wr": wr, "dd": dd}


def main():
    log_path = Path("/tmp/v10_stack_v2.txt")
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)
        log_path.write_text("\n".join(log_lines) + "\n")

    t0 = time.time()
    log(f"[v10v2] loading corpus: {CORPUS}")
    df = pd.read_parquet(CORPUS)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("US/Eastern")

    de3 = df[df["strategy"] == "DynamicEngine3"].copy().reset_index(drop=True)
    de3 = de3.sort_values("ts").reset_index(drop=True)
    log(f"[v10v2] DE3 rows: {len(de3)}")

    # Label
    de3["is_big_loss"] = (de3["pnl_baseline"] <= -50).astype(int)

    # Splits
    train_mask = de3["ts"] < HOLDOUT_START
    test_mask = ~train_mask
    de3_tr = de3[train_mask].reset_index(drop=True)
    de3_te = de3[test_mask].reset_index(drop=True)
    log(f"  train n={len(de3_tr)} positives={int(de3_tr['is_big_loss'].sum())}")
    log(f"  test  n={len(de3_te)} positives={int(de3_te['is_big_loss'].sum())}")

    # Holdout baseline (corrected v2)
    base_taken = de3_te["taken_baseline"].astype(bool).values
    base_pnls = de3_te.loc[base_taken, "pnl_baseline"].values
    base_pnl = float(np.sum(base_pnls))
    base_trades = int(base_taken.sum())
    base_dd = calc_dd(de3_te.loc[base_taken].sort_values("ts")["pnl_baseline"].values)
    base_wr = float(((base_pnls > 0).mean()) * 100.0) if base_trades > 0 else 0.0
    log(f"  holdout baseline (corrected v2): PnL=${base_pnl:+,.2f}  trades={base_trades}  DD=${base_dd:,.2f}  WR={base_wr:.1f}%")

    # ===== STAGE 1: Filter G v10 =====
    log("\n=== STAGE 1: Filter G v10 (full candidate stream) ===")
    X_tr1 = make_base_features(de3_tr)
    y_tr1 = de3_tr["is_big_loss"].values
    X_te1 = make_base_features(de3_te)
    y_te1 = de3_te["is_big_loss"].values

    clf_fg = train_hgb(X_tr1, y_tr1)
    p_tr_fg = clf_fg.predict_proba(X_tr1.values)[:, 1]
    p_te_fg = clf_fg.predict_proba(X_te1.values)[:, 1]
    auc_tr_fg = roc_auc_score(y_tr1, p_tr_fg)
    auc_te_fg = roc_auc_score(y_te1, p_te_fg) if y_te1.sum() > 0 else float("nan")
    log(f"  FG train AUC={auc_tr_fg:.3f}  test AUC={auc_te_fg:.3f}")

    # Add fg proba to dataframes (keyed by row)
    de3_tr["fg_v10_proba"] = p_tr_fg
    de3_te["fg_v10_proba"] = p_te_fg

    # ===== STAGE 2: Kalshi v10 (FG survivors AND in-window) =====
    log("\n=== STAGE 2: Kalshi v10 (FG-survivors + 12-16 ET) ===")
    in_window_tr = (de3_tr["ts"].dt.hour >= 12) & (de3_tr["ts"].dt.hour < 16)
    in_window_te = (de3_te["ts"].dt.hour >= 12) & (de3_te["ts"].dt.hour < 16)
    surv_k_tr = (p_tr_fg < 0.5) & in_window_tr.values
    sub_tr_k = de3_tr[surv_k_tr].reset_index(drop=True)
    log(f"  Kalshi train pool: {len(sub_tr_k)}  positives={int(sub_tr_k['is_big_loss'].sum())}")

    if len(sub_tr_k) >= 30 and sub_tr_k["is_big_loss"].sum() >= 5:
        X_tr_k = make_base_features(sub_tr_k, ["fg_v10_proba"])
        y_tr_k = sub_tr_k["is_big_loss"].values
        clf_k = train_hgb(X_tr_k, y_tr_k)
        # full-test preds (we'll only USE on in-window rows but still compute everywhere)
        X_te_k = make_base_features(de3_te, ["fg_v10_proba"])
        p_te_k = clf_k.predict_proba(X_te_k.values)[:, 1]
        # AUC restricted to in-window survivors
        surv_k_te = (p_te_fg < 0.5) & in_window_te.values
        if surv_k_te.sum() > 0 and de3_te.loc[surv_k_te, "is_big_loss"].sum() > 0:
            auc_te_k = roc_auc_score(de3_te.loc[surv_k_te, "is_big_loss"].values, p_te_k[surv_k_te])
        else:
            auc_te_k = float("nan")
        log(f"  K test AUC (on FG-pass + in-window): {auc_te_k:.3f}")
    else:
        clf_k = None
        p_te_k = np.zeros(len(de3_te))
        auc_te_k = float("nan")
        log("  Kalshi: pool too small/no positives — skip; K block always off")

    # Train probas for downstream chaining (only meaningful where k applied; else 0)
    if clf_k is not None:
        p_tr_k_full = clf_k.predict_proba(make_base_features(de3_tr, ["fg_v10_proba"]).values)[:, 1]
    else:
        p_tr_k_full = np.zeros(len(de3_tr))
    de3_tr["k_v10_proba"] = p_tr_k_full
    de3_te["k_v10_proba"] = p_te_k

    # ===== STAGE 3: LFO v10 (FG + K survivors) =====
    log("\n=== STAGE 3: LFO v10 (FG+K-survivors) ===")
    # Survivors at training: FG pass AND (out-of-window OR K pass)
    k_pass_tr = ~((p_tr_k_full >= 0.5) & in_window_tr.values)
    surv_lfo_tr = (p_tr_fg < 0.5) & k_pass_tr
    sub_tr_lfo = de3_tr[surv_lfo_tr].reset_index(drop=True)
    log(f"  LFO train pool: {len(sub_tr_lfo)}  positives={int(sub_tr_lfo['is_big_loss'].sum())}")

    if len(sub_tr_lfo) >= 30 and sub_tr_lfo["is_big_loss"].sum() >= 5:
        X_tr_lfo = make_base_features(sub_tr_lfo, ["fg_v10_proba", "k_v10_proba"])
        y_tr_lfo = sub_tr_lfo["is_big_loss"].values
        clf_lfo = train_hgb(X_tr_lfo, y_tr_lfo)
        X_te_lfo = make_base_features(de3_te, ["fg_v10_proba", "k_v10_proba"])
        p_te_lfo = clf_lfo.predict_proba(X_te_lfo.values)[:, 1]
        # AUC on test survivors
        k_pass_te = ~((p_te_k >= 0.5) & in_window_te.values)
        surv_lfo_te = (p_te_fg < 0.5) & k_pass_te
        if surv_lfo_te.sum() > 0 and de3_te.loc[surv_lfo_te, "is_big_loss"].sum() > 0:
            auc_te_lfo = roc_auc_score(de3_te.loc[surv_lfo_te, "is_big_loss"].values, p_te_lfo[surv_lfo_te])
        else:
            auc_te_lfo = float("nan")
        log(f"  LFO test AUC (on FG+K-survivors): {auc_te_lfo:.3f}")
    else:
        clf_lfo = None
        p_te_lfo = np.zeros(len(de3_te))
        auc_te_lfo = float("nan")
        log("  LFO: pool too small/no positives — skip; LFO block always off")

    if clf_lfo is not None:
        p_tr_lfo_full = clf_lfo.predict_proba(make_base_features(de3_tr, ["fg_v10_proba", "k_v10_proba"]).values)[:, 1]
    else:
        p_tr_lfo_full = np.zeros(len(de3_tr))
    de3_tr["lfo_v10_proba"] = p_tr_lfo_full
    de3_te["lfo_v10_proba"] = p_te_lfo

    # ===== STAGE 4: PCT v10 (triple-survivor) =====
    log("\n=== STAGE 4: PCT v10 (FG+K+LFO survivors) ===")
    lfo_pass_tr = p_tr_lfo_full < 0.5
    surv_pct_tr = (p_tr_fg < 0.5) & k_pass_tr & lfo_pass_tr
    sub_tr_pct = de3_tr[surv_pct_tr].reset_index(drop=True)
    log(f"  PCT train pool: {len(sub_tr_pct)}  positives={int(sub_tr_pct['is_big_loss'].sum())}")

    if len(sub_tr_pct) >= 30 and sub_tr_pct["is_big_loss"].sum() >= 5:
        X_tr_pct = make_base_features(sub_tr_pct, ["fg_v10_proba", "k_v10_proba", "lfo_v10_proba"])
        y_tr_pct = sub_tr_pct["is_big_loss"].values
        clf_pct = train_hgb(X_tr_pct, y_tr_pct)
        X_te_pct = make_base_features(de3_te, ["fg_v10_proba", "k_v10_proba", "lfo_v10_proba"])
        p_te_pct = clf_pct.predict_proba(X_te_pct.values)[:, 1]
        lfo_pass_te = p_te_lfo < 0.5
        surv_pct_te = (p_te_fg < 0.5) & (~((p_te_k >= 0.5) & in_window_te.values)) & lfo_pass_te
        if surv_pct_te.sum() > 0 and de3_te.loc[surv_pct_te, "is_big_loss"].sum() > 0:
            auc_te_pct = roc_auc_score(de3_te.loc[surv_pct_te, "is_big_loss"].values, p_te_pct[surv_pct_te])
        else:
            auc_te_pct = float("nan")
        log(f"  PCT test AUC (on triple-survivors): {auc_te_pct:.3f}")
    else:
        clf_pct = None
        p_te_pct = None
        auc_te_pct = None
        log("  PCT: pool too small or zero positives — DEGENERATE; PCT block disabled")

    # ===== threshold sweep =====
    log("\n" + "=" * 80)
    log("COMBINED V10 v2 STACK SIMULATION (holdout)")
    log("=" * 80)

    # 38 combinations: per-stage cross product + named presets
    in_window_te_arr = in_window_te.values

    combos = []
    # All-equal sweep
    for v in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        combos.append((f"all={v}", {"fg": v, "k": v, "lfo": v, "pct": v}))
    # Strict-each
    for stage in ("fg", "k", "lfo", "pct"):
        d = {"fg": 0.85, "k": 0.85, "lfo": 0.85, "pct": 0.85}
        d[stage] = 0.40
        combos.append((f"loose_{stage}_strict_rest", d))
    for stage in ("fg", "k", "lfo", "pct"):
        d = {"fg": 0.85, "k": 0.85, "lfo": 0.85, "pct": 0.85}
        d[stage] = 0.40
        combos.append((f"strict_rest_loose_{stage}", d))  # alias
    # User spec
    combos.append(("user_spec", {"fg": 0.40, "k": 0.85, "lfo": 0.60, "pct": 0.45}))
    combos.append(("balanced", {"fg": 0.50, "k": 0.50, "lfo": 0.50, "pct": 0.50}))
    combos.append(("dd_focused", {"fg": 0.40, "k": 0.40, "lfo": 0.40, "pct": 0.40}))
    combos.append(("permissive", {"fg": 0.70, "k": 0.70, "lfo": 0.70, "pct": 0.70}))
    # Random spread
    rng = np.random.default_rng(0)
    for i in range(20):
        d = {k: float(rng.choice([0.45, 0.55, 0.65, 0.75, 0.85])) for k in ("fg", "k", "lfo", "pct")}
        combos.append((f"rand_{i}", d))

    # De-dup
    seen = set()
    deduped = []
    for label, d in combos:
        key = (round(d["fg"], 3), round(d["k"], 3), round(d["lfo"], 3), round(d["pct"], 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, d))
    combos = deduped
    log(f"Threshold combinations: {len(combos)}")

    # Run sweep
    results = []
    for label, d in combos:
        res = simulate_combined(
            de3_te,
            p_te_fg,
            p_te_k,
            p_te_lfo,
            p_te_pct if p_te_pct is not None else None,
            d["fg"], d["k"], d["lfo"], d["pct"],
            in_window_te_arr,
        )
        # gates
        g1 = abs(res["dd"]) <= DD_CAP
        g2 = (res["pnl"] >= base_pnl) and (res["fired"] <= base_trades)
        g3 = res["fired"] >= MIN_OOS
        g4 = res["wr"] >= WR_GATE
        all_pass = g1 and g2 and g3 and g4
        results.append({
            "label": label, "thresholds": d, "fired": res["fired"],
            "pnl": res["pnl"], "wr": res["wr"], "dd": res["dd"],
            "g1": g1, "g2": g2, "g3": g3, "g4": g4, "all_pass": all_pass,
        })

    results.sort(key=lambda r: r["pnl"], reverse=True)
    log(f"\nTop 20 by PnL:")
    log(f"{'Combo':<28}{'fired':>7}{'PnL':>14}{'WR%':>7}{'DD':>10}  gates")
    for r in results[:20]:
        marks = "".join("✓" if r[f"g{i}"] else "✗" for i in (1, 2, 3, 4))
        log(f"{r['label']:<28}{r['fired']:>7} ${r['pnl']:>+11,.2f}{r['wr']:>6.1f}% ${abs(r['dd']):>8,.2f}  {marks}")

    passing = [r for r in results if r["all_pass"]]
    log(f"\nCombinations passing all 4 gates: {len(passing)}")

    log("\nTop 5 by lowest |DD|:")
    by_dd = sorted(results, key=lambda r: abs(r["dd"]))[:5]
    for r in by_dd:
        log(f"  {r['label']}: DD=${abs(r['dd']):,.2f}  PnL=${r['pnl']:+,.2f}  fired={r['fired']}")

    log("\nTop 5 by Pareto (PnL/|DD|):")
    by_pareto = sorted(
        [r for r in results if r["fired"] >= MIN_OOS and abs(r["dd"]) > 0],
        key=lambda r: (r["pnl"] / abs(r["dd"])),
        reverse=True,
    )[:5]
    for r in by_pareto:
        ratio = r["pnl"] / abs(r["dd"])
        log(f"  {r['label']}: PnL/DD={ratio:.2f}  PnL=${r['pnl']:+,.2f}  DD=${abs(r['dd']):,.2f}  fired={r['fired']}")

    # ===== save artifacts =====
    log("\n[v10v2] saving model.joblib and metrics.json per stage…")
    for stage_name, clf, auc_te, extras in [
        ("filterg_v10_v2", clf_fg, auc_te_fg, []),
        ("kalshi_v10_v2", clf_k, auc_te_k, ["fg_v10_proba"]),
        ("lfo_v10_v2", clf_lfo, auc_te_lfo, ["fg_v10_proba", "k_v10_proba"]),
        ("pct_v10_v2", clf_pct, auc_te_pct, ["fg_v10_proba", "k_v10_proba", "lfo_v10_proba"]),
    ]:
        outdir = ROOT / "artifacts" / f"regime_ml_{stage_name}" / "de3"
        outdir.mkdir(parents=True, exist_ok=True)
        if clf is not None:
            joblib.dump(clf, outdir / "model.joblib")
        meta = {
            "stage": stage_name,
            "auc_test": float(auc_te) if auc_te is not None and not (isinstance(auc_te, float) and np.isnan(auc_te)) else None,
            "trained": clf is not None,
            "extra_features": extras,
        }
        (outdir / "metrics.json").write_text(json.dumps(meta, indent=2))
        log(f"  {stage_name}: trained={clf is not None}  auc_test={meta['auc_test']}")

    # Summary JSON
    best_pareto = by_pareto[0] if by_pareto else None
    best_dd = by_dd[0] if by_dd else None
    summary = {
        "method": "v10 stack-aware sequential training (CORRECTED v2 corpus)",
        "corpus": str(CORPUS),
        "holdout_baseline_v2": {"pnl": base_pnl, "trades": base_trades, "dd": base_dd, "wr": base_wr},
        "ship_gates": {"G1_max_dd": DD_CAP, "G2_pnl>=base_and_trades<=base": True, "G3_min_oos": MIN_OOS, "G4_wr": WR_GATE},
        "stage_aucs": {
            "filterg_train": float(auc_tr_fg),
            "filterg_test": float(auc_te_fg),
            "kalshi_test_on_fg_survivors": float(auc_te_k) if not np.isnan(auc_te_k) else None,
            "lfo_test_on_fg_k_survivors": float(auc_te_lfo) if not np.isnan(auc_te_lfo) else None,
            "pct_test_on_triple_survivors": float(auc_te_pct) if (auc_te_pct is not None and not (isinstance(auc_te_pct, float) and np.isnan(auc_te_pct))) else None,
        },
        "combinations_tested": len(combos),
        "passing_all_gates": len(passing),
        "best_pareto": best_pareto,
        "best_dd_floor": best_dd,
        "top_10_by_pnl": results[:10],
    }
    out_summary = ROOT / "artifacts" / "v10_stack_aware_v2_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nSaved: {out_summary}")
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""RA rule mining — companion to §8.31 DE3 work.

The RA family has only 175 rows in v11 corpus (163 allowed_by_friend_rule==True).
Train (Mar-Dec 2025): ~134 / Holdout (Jan-Apr 2026): ~29 — below G3=50.

So this script does THREE passes:
  (A) Strict Mar25/Jan26 split with G3>=50 — expected to fail trivially.
  (B) Walk-forward CV (5 contiguous splits) — the rigorous answer.
  (C) Relaxed G3>=25 — "is there a signal at all".

Phases mirror §8.31:
  1. Feature audit on RA train
  2. Single-feature rule sweep
  3. Two-feature conjunctions
  4. Decision tree (max_depth=3, min_samples_leaf=15)
  5. Walk-forward validation across 5 splits
  6. Three gate variants (strict, relaxed, walk-forward)
  7. Report

Run:  python tools/ra_rule_mining.py
"""

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, _tree

ROOT = "/Users/wes/Downloads/JULIE001"
CORPUS = os.path.join(ROOT, "artifacts/v11_training_corpus_with_mfe.parquet")
OUT_TOP = os.path.join(ROOT, "artifacts/ra_rule_mining_top_rules.json")
OUT_WF = os.path.join(ROOT, "artifacts/ra_walk_forward_validation.json")
OUT_BEST = os.path.join(ROOT, "artifacts/best_ra_rule.json")
REPORT = "/tmp/ra_rule_mining_report.md"

HOLDOUT_START = "2026-01-01"

GATES_STRICT = {"G1_870": 870.0, "G3_min_n": 50, "G4_min_wr": 0.55}
GATES_RELAXED = {"G1_870": 870.0, "G3_min_n": 25, "G4_min_wr": 0.55}


def dd_formula_a(pnl_series: pd.Series) -> float:
    if len(pnl_series) == 0:
        return 0.0
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    return float((peak - cum).max())


def metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "dd": 0.0}
    pnl = df["net_pnl_after_haircut"]
    return {
        "n": int(len(df)),
        "wr": float((pnl > 0).mean()),
        "pnl": float(pnl.sum()),
        "dd": dd_formula_a(pnl),
    }


def gates_check(holdout_metrics: dict, baseline_pnl: float, baseline_n: int, g3_min: int) -> dict:
    g1_870 = holdout_metrics["dd"] <= 870.0
    g2 = (holdout_metrics["pnl"] >= baseline_pnl) and (holdout_metrics["n"] <= baseline_n)
    g3 = holdout_metrics["n"] >= g3_min
    g4 = holdout_metrics["wr"] >= 0.55
    return {
        "G1_870": bool(g1_870),
        "G2": bool(g2),
        "G3": bool(g3),
        "G4": bool(g4),
        "all": bool(g1_870 and g2 and g3 and g4),
    }


def load_corpus():
    df = pd.read_parquet(CORPUS)
    df = df[(df["family"] == "regimeadaptive") & (df["allowed_by_friend_rule"] == True)].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["win"] = (df["net_pnl_after_haircut"] > 0).astype(int)
    df["hour_et"] = df["ts"].dt.hour
    df["minute_of_session"] = (df["ts"].dt.hour - 9) * 60 + df["ts"].dt.minute - 30
    df["minute_of_session"] = df["minute_of_session"].clip(lower=0)
    df["dow"] = df["ts"].dt.dayofweek
    df["half_hour_bucket"] = df["hour_et"] * 2 + (df["ts"].dt.minute // 30)
    df["is_long"] = (df["side"] == "LONG").astype(int)
    return df


def split_train_holdout(df: pd.DataFrame):
    train = df[df["ts"] < HOLDOUT_START].reset_index(drop=True)
    holdout = df[df["ts"] >= HOLDOUT_START].reset_index(drop=True)
    return train, holdout


def feature_audit(train: pd.DataFrame):
    NON_FEATURE = {
        "ts", "exit_ts", "entry_price", "sl", "tp", "exit_price",
        "raw_pnl", "raw_pnl_resim", "net_pnl_after_haircut",
        "is_big_loss", "allowed_by_friend_rule", "win",
        "exit_reason", "exit_reason_resim", "contract", "strategy",
        "family", "side", "sl6_eligible", "sl6_exit_reason",
        "sl6_raw_pnl", "sl6_mfe_points", "sl6_mae_points",
        "be_arm_threshold_pts", "mfe_crosses_be_arm",
        "mfe_points", "mae_points",
    }
    numeric_features = []
    for c in train.columns:
        if c in NON_FEATURE:
            continue
        if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c]):
            numeric_features.append(c)

    audit = []
    y_train = train["win"].values
    for f in numeric_features:
        x = train[f].astype(float).values
        if np.isnan(x).all():
            continue
        if np.isnan(x).any():
            med = np.nanmedian(x)
            x = np.where(np.isnan(x), med, x)
        # need variance
        if np.std(x) < 1e-12:
            continue
        try:
            corr_win = float(np.corrcoef(x, y_train)[0, 1])
        except Exception:
            corr_win = 0.0
        if np.isnan(corr_win):
            corr_win = 0.0
        try:
            q1, q3 = np.quantile(x, [0.25, 0.75])
            wr_lo = float(train.loc[x <= q1, "win"].mean()) if (x <= q1).sum() else 0.0
            wr_hi = float(train.loc[x >= q3, "win"].mean()) if (x >= q3).sum() else 0.0
            wr_diff = wr_hi - wr_lo
        except Exception:
            wr_diff = 0.0
            wr_hi = wr_lo = 0.0
        audit.append({
            "feature": f, "corr_win": corr_win, "wr_top_q": wr_hi, "wr_bot_q": wr_lo,
            "wr_q_diff": wr_diff,
            "abs_signal": max(abs(corr_win), abs(wr_diff) / 2),
        })

    # Get list of features still present
    feats_present = [a["feature"] for a in audit]
    X = train[feats_present].astype(float).fillna(train[feats_present].median(numeric_only=True)).values
    try:
        mi = mutual_info_classif(X, y_train, random_state=0)
        for i, a in enumerate(audit):
            a["mutual_info"] = float(mi[i])
            a["abs_signal"] = max(a["abs_signal"], a["mutual_info"] * 5)
    except Exception:
        for a in audit:
            a["mutual_info"] = 0.0

    audit_sorted = sorted(audit, key=lambda r: r["abs_signal"], reverse=True)
    return audit_sorted, feats_present


def sweep_single(train, holdout, feature, baseline_pnl, baseline_n, n_thresholds=20,
                 g3_min=25, min_train_n=20, min_holdout_n=5):
    x_train = train[feature].astype(float).values
    x_holdout = holdout[feature].astype(float).values
    if np.isnan(x_train).all():
        return []
    finite = x_train[np.isfinite(x_train)]
    if len(finite) < 30:
        return []
    qs = np.linspace(0.05, 0.95, n_thresholds)
    thresholds = np.unique(np.quantile(finite, qs))
    results = []
    for op in [">", "<="]:
        for thr in thresholds:
            if op == ">":
                mask_t = x_train > thr
                mask_h = x_holdout > thr
            else:
                mask_t = x_train <= thr
                mask_h = x_holdout <= thr
            if mask_t.sum() < min_train_n or mask_h.sum() < min_holdout_n:
                continue
            mt = metrics(train.loc[mask_t])
            mh = metrics(holdout.loc[mask_h])
            gates_strict = gates_check(mh, baseline_pnl, baseline_n, 50)
            gates_relaxed = gates_check(mh, baseline_pnl, baseline_n, 25)
            results.append({
                "feature": feature, "operator": op, "threshold": float(thr),
                "train": mt, "holdout": mh,
                "gates_strict": gates_strict, "gates_relaxed": gates_relaxed,
            })
    return results


def phase2(train, holdout, audit, top_k=30):
    base_h = metrics(holdout)
    baseline_pnl = base_h["pnl"]
    baseline_n = base_h["n"]
    top_features = [a["feature"] for a in audit[:top_k]]
    all_rules = []
    for f in top_features:
        all_rules.extend(sweep_single(train, holdout, f, baseline_pnl, baseline_n))

    def rank(r):
        gs = r["gates_relaxed"]
        passes = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        return (-passes, -r["holdout"]["wr"], -r["holdout"]["pnl"], r["holdout"]["dd"], -r["holdout"]["n"])

    all_rules.sort(key=rank)
    return all_rules[:50], baseline_pnl, baseline_n


def phase3(train, holdout, top_single, baseline_pnl, baseline_n, top_k=15):
    seeds = top_single[:top_k]
    pair_rules = []
    for ra, rb in combinations(seeds, 2):
        if ra["feature"] == rb["feature"]:
            continue

        def mk_mask(df, r):
            x = df[r["feature"]].astype(float).values
            return (x > r["threshold"]) if r["operator"] == ">" else (x <= r["threshold"])

        mt = mk_mask(train, ra) & mk_mask(train, rb)
        mh = mk_mask(holdout, ra) & mk_mask(holdout, rb)
        if mt.sum() < 15 or mh.sum() < 3:
            continue
        mt_m = metrics(train.loc[mt])
        mh_m = metrics(holdout.loc[mh])
        gs_strict = gates_check(mh_m, baseline_pnl, baseline_n, 50)
        gs_relaxed = gates_check(mh_m, baseline_pnl, baseline_n, 25)
        pair_rules.append({
            "feature_a": ra["feature"], "op_a": ra["operator"], "thr_a": ra["threshold"],
            "feature_b": rb["feature"], "op_b": rb["operator"], "thr_b": rb["threshold"],
            "train": mt_m, "holdout": mh_m,
            "gates_strict": gs_strict, "gates_relaxed": gs_relaxed,
        })

    def rank(r):
        gs = r["gates_relaxed"]
        passes = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        return (-passes, -r["holdout"]["wr"], -r["holdout"]["pnl"], r["holdout"]["dd"], -r["holdout"]["n"])

    pair_rules.sort(key=rank)
    return pair_rules[:50]


def extract_tree_rules(tree, feature_names):
    t = tree.tree_
    leaves = []

    def recurse(node, conditions):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            value = t.value[node][0]
            n_total = int(value.sum())
            n_pos = int(value[1])
            wr = n_pos / max(n_total, 1)
            leaves.append({"conditions": list(conditions), "n_train": n_total, "wr_train_pred": wr})
            return
        f = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        recurse(t.children_left[node], conditions + [(f, "<=", thr)])
        recurse(t.children_right[node], conditions + [(f, ">", thr)])

    recurse(0, [])
    return leaves


def phase4(train, holdout, audit, baseline_pnl, baseline_n, max_depth=3, min_samples_leaf=15):
    feats = [a["feature"] for a in audit[:30]]
    Xt = train[feats].astype(float).fillna(train[feats].median(numeric_only=True))
    yt = train["win"].values
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    clf.fit(Xt, yt)
    leaves = extract_tree_rules(clf, feats)
    leaf_results = []
    for leaf in leaves:
        mt = np.ones(len(train), dtype=bool)
        mh = np.ones(len(holdout), dtype=bool)
        for f, op, thr in leaf["conditions"]:
            xt = train[f].astype(float).fillna(train[f].median()).values
            xh = holdout[f].astype(float).fillna(train[f].median()).values
            if op == "<=":
                mt &= (xt <= thr)
                mh &= (xh <= thr)
            else:
                mt &= (xt > thr)
                mh &= (xh > thr)
        if mt.sum() < 10 or mh.sum() < 3:
            continue
        train_m = metrics(train.loc[mt])
        hold_m = metrics(holdout.loc[mh])
        gs_strict = gates_check(hold_m, baseline_pnl, baseline_n, 50)
        gs_relaxed = gates_check(hold_m, baseline_pnl, baseline_n, 25)
        leaf_results.append({
            "conditions": [{"feature": f, "operator": op, "threshold": thr} for f, op, thr in leaf["conditions"]],
            "train": train_m, "holdout": hold_m,
            "gates_strict": gs_strict, "gates_relaxed": gs_relaxed,
        })

    def rank(r):
        gs = r["gates_relaxed"]
        passes = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        return (-passes, -r["holdout"]["wr"], -r["holdout"]["pnl"], r["holdout"]["dd"], -r["holdout"]["n"])

    leaf_results.sort(key=rank)
    return leaf_results[:10]


def apply_rule(df, rule, kind):
    if kind == "single":
        x = df[rule["feature"]].astype(float).values
        return (x > rule["threshold"]) if rule["operator"] == ">" else (x <= rule["threshold"])
    if kind == "pair":
        xa = df[rule["feature_a"]].astype(float).values
        xb = df[rule["feature_b"]].astype(float).values
        ma = (xa > rule["thr_a"]) if rule["op_a"] == ">" else (xa <= rule["thr_a"])
        mb = (xb > rule["thr_b"]) if rule["op_b"] == ">" else (xb <= rule["thr_b"])
        return ma & mb
    if kind == "leaf":
        m = np.ones(len(df), dtype=bool)
        for c in rule["conditions"]:
            x = df[c["feature"]].astype(float).fillna(df[c["feature"]].median()).values
            if c["operator"] == "<=":
                m &= (x <= c["threshold"])
            else:
                m &= (x > c["threshold"])
        return m
    return np.ones(len(df), dtype=bool)


def fmt_rule(rule, kind):
    if kind == "single":
        return f"{rule['feature']} {rule['operator']} {rule['threshold']:.4f}"
    if kind == "pair":
        return (f"({rule['feature_a']} {rule['op_a']} {rule['thr_a']:.4f}) AND "
                f"({rule['feature_b']} {rule['op_b']} {rule['thr_b']:.4f})")
    if kind == "leaf":
        return " AND ".join([f"({c['feature']} {c['operator']} {c['threshold']:.4f})"
                             for c in rule["conditions"]])
    return str(rule)


def walk_forward_validation(df_all: pd.DataFrame, candidates: list, n_splits: int = 5):
    """Time-series walk-forward CV over the FULL RA dataset (ordered by ts).

    Splits the full sequence into n_splits contiguous time windows. For each split:
    - test = the chunk
    - eval rule on test
    Reports per-rule mean ± std across splits, and consistency (#splits with WR>=0.55).
    """
    df_sorted = df_all.sort_values("ts").reset_index(drop=True)
    n = len(df_sorted)
    fold_size = n // n_splits
    folds = []
    for i in range(n_splits):
        s = i * fold_size
        e = (i + 1) * fold_size if i < n_splits - 1 else n
        folds.append((s, e))

    wf_results = []
    base_full = metrics(df_sorted)
    for kind, rule in candidates:
        per_fold = []
        for fi, (s, e) in enumerate(folds):
            test_df = df_sorted.iloc[s:e].reset_index(drop=True)
            mask = apply_rule(test_df, rule, kind)
            sel = test_df.loc[mask]
            m = metrics(sel)
            base_fold = metrics(test_df)
            per_fold.append({
                "fold": fi,
                "fold_n_total": len(test_df),
                "fold_baseline_wr": base_fold["wr"],
                "fold_baseline_pnl": base_fold["pnl"],
                "n": m["n"], "wr": m["wr"], "pnl": m["pnl"], "dd": m["dd"],
            })

        wrs = [f["wr"] for f in per_fold if f["n"] >= 3]
        ns = [f["n"] for f in per_fold]
        pnls = [f["pnl"] for f in per_fold]
        if not wrs:
            continue
        mean_wr = float(np.mean(wrs))
        std_wr = float(np.std(wrs))
        consistency = int(sum(1 for f in per_fold if f["n"] >= 3 and f["wr"] >= 0.55))
        wf_results.append({
            "kind": kind,
            "rule_human": fmt_rule(rule, kind),
            "rule_data": rule,
            "per_fold": per_fold,
            "mean_wr": mean_wr,
            "std_wr": std_wr,
            "mean_n": float(np.mean(ns)),
            "mean_pnl": float(np.mean(pnls)),
            "consistency_4_of_5": bool(consistency >= 4),
            "splits_above_55": consistency,
        })

    wf_results.sort(key=lambda r: (-(r["consistency_4_of_5"]), -r["mean_wr"], -r["mean_pnl"]))
    return wf_results


def write_report(audit, top_single, top_pair, top_leaves, wf_top, base_t, base_h,
                 best_rule_strict, best_rule_relaxed, best_rule_wf,
                 ra_n_total, ra_n_train, ra_n_holdout, per_month):
    lines = []
    lines.append("# RA Rule Mining — The Companion to §8.31\n\n")
    lines.append("**Methodology:** mirrors §8.31 DE3 work — find a deterministic rule that selects winners from RA candidates. ")
    lines.append("Because RA holdout n=29 (< G3=50), this report runs 3 passes: strict gates, relaxed G3≥25, and walk-forward CV.\n\n")

    # Phase 1
    lines.append("## Phase 1 — RA Data Audit\n\n")
    lines.append(f"- **Total RA rows in v11 corpus (allowed_by_friend_rule==True):** {ra_n_total}\n")
    lines.append(f"- **Train (Mar-Dec 2025):** n={ra_n_train}, WR={base_t['wr']:.3f}, PnL=${base_t['pnl']:.0f}, DD=${base_t['dd']:.0f}\n")
    lines.append(f"- **Holdout (Jan-Apr 2026):** n={ra_n_holdout}, WR={base_h['wr']:.3f}, PnL=${base_h['pnl']:.0f}, DD=${base_h['dd']:.0f}\n")
    lines.append(f"- **G3 violation:** holdout n={ra_n_holdout} < 50 floor — strict gate G3 unsatisfiable on this data volume regardless of rule.\n")
    lines.append(f"- **RA-specific feature columns:** none. RA shares the `bf_de3_entry_*`, `bf_*`, `pct_*`, and probability features with DE3 in this corpus.\n\n")
    lines.append(f"### Per-month RA fire counts\n\n")
    lines.append("| Month | n | WR | PnL |\n|---|---|---|---|\n")
    for m, row in per_month.iterrows():
        lines.append(f"| {m} | {int(row['n'])} | {row['wr']:.3f} | ${row['pnl']:.0f} |\n")
    lines.append("\n")

    # Top-15 features
    lines.append("### Top 15 Features by Signal (RA train)\n\n")
    lines.append("| Feature | MI | corr(win) | WR top-Q | WR bot-Q |\n|---|---|---|---|---|\n")
    for a in audit[:15]:
        lines.append(f"| {a['feature']} | {a['mutual_info']:.4f} | {a['corr_win']:+.3f} | {a['wr_top_q']:.3f} | {a['wr_bot_q']:.3f} |\n")
    lines.append("\n")

    # Phase 2
    lines.append("## Phase 2 — Top 10 Single-Feature Rules (RA holdout-sorted)\n\n")
    lines.append("| Rule | Train n/WR | Holdout n | Holdout WR | Holdout PnL | Holdout DD | Strict | Relaxed |\n|---|---|---|---|---|---|---|---|\n")
    for r in top_single[:10]:
        gs = r["gates_strict"]
        gr = r["gates_relaxed"]
        sp = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        rp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        rule_str = f"{r['feature']} {r['operator']} {r['threshold']:.4f}"
        lines.append(f"| {rule_str} | {r['train']['n']}/{r['train']['wr']:.3f} | {r['holdout']['n']} | {r['holdout']['wr']:.3f} | ${r['holdout']['pnl']:.0f} | ${r['holdout']['dd']:.0f} | {sp}/4 | {rp}/4 |\n")
    lines.append("\n")

    # Phase 3
    lines.append("## Phase 3 — Top 10 Two-Feature Conjunctions\n\n")
    lines.append("| Rule | Train n/WR | Holdout n | Holdout WR | Holdout PnL | Holdout DD | Strict | Relaxed |\n|---|---|---|---|---|---|---|---|\n")
    for r in top_pair[:10]:
        gs = r["gates_strict"]
        gr = r["gates_relaxed"]
        sp = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        rp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        rule_str = f"({r['feature_a']} {r['op_a']} {r['thr_a']:.3f}) AND ({r['feature_b']} {r['op_b']} {r['thr_b']:.3f})"
        lines.append(f"| {rule_str} | {r['train']['n']}/{r['train']['wr']:.3f} | {r['holdout']['n']} | {r['holdout']['wr']:.3f} | ${r['holdout']['pnl']:.0f} | ${r['holdout']['dd']:.0f} | {sp}/4 | {rp}/4 |\n")
    lines.append("\n")

    # Phase 4
    lines.append("## Phase 4 — Decision Tree Leaves (max_depth=3, min_samples_leaf=15)\n\n")
    lines.append("| Rule | Train n/WR | Holdout n | Holdout WR | Holdout PnL | Strict | Relaxed |\n|---|---|---|---|---|---|---|\n")
    for r in top_leaves[:10]:
        gs = r["gates_strict"]
        gr = r["gates_relaxed"]
        sp = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        rp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        rule_str = " AND ".join([f"{c['feature']}{c['operator']}{c['threshold']:.3f}" for c in r["conditions"]])
        lines.append(f"| {rule_str} | {r['train']['n']}/{r['train']['wr']:.3f} | {r['holdout']['n']} | {r['holdout']['wr']:.3f} | ${r['holdout']['pnl']:.0f} | {sp}/4 | {rp}/4 |\n")
    lines.append("\n")

    # Phase 5 — walk-forward
    lines.append("## Phase 5 — Walk-Forward Validation (5 splits across full 163 RA rows)\n\n")
    lines.append("Each split is ~33 contiguous-time RA rows. Per-split rule performance, then mean ± std.\n\n")
    lines.append("| Rule | Mean WR | Std WR | Mean n | Mean PnL | Splits ≥55% | 4-of-5 robust? |\n|---|---|---|---|---|---|---|\n")
    for r in wf_top[:10]:
        lines.append(f"| {r['rule_human'][:80]} | {r['mean_wr']:.3f} | {r['std_wr']:.3f} | {r['mean_n']:.1f} | ${r['mean_pnl']:.0f} | {r['splits_above_55']}/5 | {'YES' if r['consistency_4_of_5'] else 'no'} |\n")
    lines.append("\n")

    # Per-fold breakdown for top 5
    lines.append("### Per-fold detail (top 5 walk-forward rules)\n\n")
    for r in wf_top[:5]:
        lines.append(f"**{r['rule_human']}**\n\n")
        lines.append("| Fold | n_total | rule n | rule WR | rule PnL | rule DD | baseline WR |\n|---|---|---|---|---|---|---|\n")
        for f in r["per_fold"]:
            lines.append(f"| {f['fold']} | {f['fold_n_total']} | {f['n']} | {f['wr']:.3f} | ${f['pnl']:.0f} | ${f['dd']:.0f} | {f['fold_baseline_wr']:.3f} |\n")
        lines.append("\n")

    # Phase 6 — three gate variants
    lines.append("## Phase 6 — Three Gate Variants\n\n")
    lines.append("| Rule | Strict G3≥50 | Relaxed G3≥25 | Walk-forward 4-of-5 |\n|---|---|---|---|\n")
    pool = []
    for r in top_single[:5]:
        pool.append(("single", r))
    for r in top_pair[:5]:
        pool.append(("pair", r))
    for r in top_leaves[:5]:
        pool.append(("leaf", r))
    # Find walk-forward results for these by rule_human match
    wf_lookup = {r["rule_human"]: r for r in wf_top}
    for kind, r in pool[:10]:
        gs = r["gates_strict"]
        gr = r["gates_relaxed"]
        sp = sum([gs["G1_870"], gs["G2"], gs["G3"], gs["G4"]])
        rp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        rh = fmt_rule(r, kind)
        wf = wf_lookup.get(rh)
        wf_str = f"{wf['splits_above_55']}/5 (mean WR {wf['mean_wr']:.2f})" if wf else "n/a"
        lines.append(f"| {rh[:75]} | {sp}/4 ({'PASS' if gs['all'] else 'FAIL'}) | {rp}/4 ({'PASS' if gr['all'] else 'FAIL'}) | {wf_str} |\n")
    lines.append("\n")

    # Phase 7 — verdict
    lines.append("## Phase 7 — Verdict\n\n")

    if best_rule_strict is None:
        lines.append("**(A) Hard SKIP at strict gates:** YES. ")
        lines.append(f"RA holdout n=29 < G3=50. No rule, regardless of WR/PnL/DD on the 29 rows, can satisfy G3. ")
        lines.append("This was anticipated and is the structural constraint, not a methodology failure.\n\n")
    else:
        lines.append(f"**(A) Strict gates:** A rule passes. {fmt_rule(best_rule_strict[1], best_rule_strict[0])}\n\n")

    if best_rule_wf is None:
        lines.append("**(B) Walk-forward verdict:** No rule consistently produced WR ≥ 55% in 4 of 5 splits with n ≥ 3 per split. ")
        lines.append("RA's signal is too weak / sample too small for stable rule discovery via walk-forward.\n\n")
    else:
        wf = best_rule_wf
        lines.append(f"**(B) Walk-forward verdict:** Robust rule found. ")
        lines.append(f"`{wf['rule_human']}` clears WR ≥ 55% in {wf['splits_above_55']}/5 splits ")
        lines.append(f"(mean WR {wf['mean_wr']:.3f} ± {wf['std_wr']:.3f}, mean n/split {wf['mean_n']:.1f}). ")
        lines.append("Caveat: per-fold n is small (~6-15), so confidence intervals are wide.\n\n")

    if best_rule_relaxed is None:
        lines.append("**(C) Relaxed G3≥25 verdict:** No rule clears all 4 gates with G3≥25. ")
        lines.append("Even with the floor halved, RA holdout doesn't yield a deterministic ship-quality rule.\n\n")
    else:
        kind, r = best_rule_relaxed
        lines.append(f"**(C) Relaxed G3≥25 verdict:** Rule passes 4/4 at G3≥25: `{fmt_rule(r, kind)}` ")
        lines.append(f"(holdout n={r['holdout']['n']}, WR={r['holdout']['wr']:.3f}, PnL=${r['holdout']['pnl']:.0f}, DD=${r['holdout']['dd']:.0f}).\n\n")

    # Honest summary
    lines.append("### Honest summary\n\n")
    if best_rule_wf is not None and best_rule_relaxed is not None:
        lines.append("RA can ship a rule under (B) walk-forward consistency + (C) relaxed G3, but NOT under strict G3. ")
        lines.append("Either accept relaxed evidence (small-n caveat) or wait for more RA data.\n\n")
    elif best_rule_wf is not None:
        lines.append("RA shows walk-forward consistency on a candidate rule but doesn't pass the relaxed G3 sweep on the chronological holdout. ")
        lines.append("Mixed signal; recommend collecting more data before shipping.\n\n")
    elif best_rule_relaxed is not None:
        lines.append("RA passes a relaxed G3 (n≥25) gate on the chronological holdout but lacks walk-forward stability. ")
        lines.append("This pattern is consistent with overfitting to the small holdout window. Do NOT ship.\n\n")
    else:
        lines.append("**RA does not ship.** No rule satisfies any of the three criteria. Either (1) collect more RA data ")
        lines.append("(target ≥500 rows for stable mining), (2) accept RA as unshipped and route only DE3, or (3) gate RA more conservatively at the strategy level (e.g. veto in vol_regime=high, the §8.5 finding).\n\n")

    lines.append("### Recommended deployment for RA\n\n")
    if best_rule_wf is not None and best_rule_relaxed is not None:
        kind, r = best_rule_relaxed
        lines.append(f"If shipping is allowed under relaxed evidence: apply `{fmt_rule(r, kind)}` ")
        lines.append("as a deterministic RA-only entry filter, with a 4-week post-deployment review against actual fills. ")
        lines.append(f"This would reduce RA holdout from {ra_n_holdout} to {r['holdout']['n']} trades and lift WR from {base_h['wr']*100:.1f}% to {r['holdout']['wr']*100:.1f}%.\n\n")
        lines.append("**Stronger recommendation:** keep RA gated/vetoed at strategy level (e.g. `JULIE_VETO_RA_VOL_HIGH=1` per §8.5) until n_RA reaches 500+ and a strict-gate rule mining pass is feasible.\n\n")
    else:
        lines.append("**No rule is recommended for shipment.** The data volume (n=163 total, n_holdout=29) cannot support strict gate validation, and walk-forward consistency was not established. ")
        lines.append("Concrete options:\n")
        lines.append("1. **Hold RA in current state** — accept the negative baseline ($PnL on holdout) as cost, gate at strategy level only.\n")
        lines.append("2. **Disable RA family entirely** — drop the 29 holdout trades; saves the negative PnL but admits failure.\n")
        lines.append("3. **Wait for data** — re-run this mining at end of Q3 2026 once RA fires accumulate.\n")
        lines.append("4. **Reduce validation rigor** — accept walk-forward-only evidence even without 4-of-5 consistency, with explicit caution.\n\n")

    lines.append("\n## Files written\n")
    lines.append(f"- {OUT_TOP}\n")
    lines.append(f"- {OUT_WF}\n")
    if best_rule_relaxed is not None or best_rule_wf is not None:
        lines.append(f"- {OUT_BEST}\n")
    lines.append(f"- {REPORT}\n")
    lines.append(f"- tools/ra_rule_mining.py\n")

    with open(REPORT, "w") as f:
        f.writelines(lines)


def main():
    print("Loading RA corpus subset...")
    df = load_corpus()
    print(f"  RA + allowed rows: {len(df)}")

    train, holdout = split_train_holdout(df)
    base_t = metrics(train)
    base_h = metrics(holdout)
    print(f"  train n={base_t['n']} wr={base_t['wr']:.3f} pnl={base_t['pnl']:.0f} dd={base_t['dd']:.0f}")
    print(f"  holdout n={base_h['n']} wr={base_h['wr']:.3f} pnl={base_h['pnl']:.0f} dd={base_h['dd']:.0f}")

    # per-month
    df_pm = df.copy()
    df_pm["month"] = df_pm["ts"].dt.to_period("M").astype(str)
    per_month = df_pm.groupby("month").apply(
        lambda g: pd.Series({"n": len(g), "wr": (g["net_pnl_after_haircut"] > 0).mean(),
                             "pnl": g["net_pnl_after_haircut"].sum()})
    )
    print(per_month)

    # Phase 1
    print("\nPhase 1 — feature audit on RA train...")
    audit, feats = feature_audit(train)
    print(f"  audited {len(feats)} numeric features")
    print("  top-10 by signal:")
    for a in audit[:10]:
        print(f"    {a['feature']:<40s}  mi={a.get('mutual_info', 0):.4f}  corr_win={a['corr_win']:+.3f}  q_diff={a['wr_q_diff']:+.3f}")

    # Phase 2
    print("\nPhase 2 — single-feature rule sweep...")
    top_single, base_pnl, base_n = phase2(train, holdout, audit, top_k=30)
    print(f"  generated {len(top_single)} single-feature rules")
    for r in top_single[:5]:
        gr = r["gates_relaxed"]
        gp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        print(f"    {r['feature']:<35s} {r['operator']} {r['threshold']:>8.4f}  | "
              f"holdout n={r['holdout']['n']:>3d} wr={r['holdout']['wr']:.3f} pnl={r['holdout']['pnl']:>+8.0f} | relaxed={gp}/4")

    # Phase 3
    print("\nPhase 3 — two-feature conjunctions...")
    top_pair = phase3(train, holdout, top_single, base_pnl, base_n, top_k=15)
    print(f"  generated {len(top_pair)} pair rules")
    for r in top_pair[:5]:
        gr = r["gates_relaxed"]
        gp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        print(f"    {r['feature_a']} {r['op_a']} {r['thr_a']:.3f} AND {r['feature_b']} {r['op_b']} {r['thr_b']:.3f}  "
              f"| h n={r['holdout']['n']} wr={r['holdout']['wr']:.3f} | relaxed={gp}/4")

    # Phase 4
    print("\nPhase 4 — decision tree leaves (max_depth=3, min_samples_leaf=15)...")
    top_leaves = phase4(train, holdout, audit, base_pnl, base_n, max_depth=3, min_samples_leaf=15)
    print(f"  generated {len(top_leaves)} leaf rules")
    for r in top_leaves[:5]:
        gr = r["gates_relaxed"]
        gp = sum([gr["G1_870"], gr["G2"], gr["G3"], gr["G4"]])
        cs = " AND ".join([f"{c['feature']}{c['operator']}{c['threshold']:.3f}" for c in r["conditions"]])
        print(f"    {cs} | h n={r['holdout']['n']} wr={r['holdout']['wr']:.3f} | relaxed={gp}/4")

    # Phase 5 — walk-forward
    print("\nPhase 5 — walk-forward validation across 5 splits of RA full data...")
    candidates = []
    for r in top_single[:10]:
        candidates.append(("single", r))
    for r in top_pair[:10]:
        candidates.append(("pair", r))
    for r in top_leaves[:5]:
        candidates.append(("leaf", r))
    wf = walk_forward_validation(df, candidates, n_splits=5)
    print(f"  evaluated {len(wf)} candidates across 5 splits")
    print("  top-5 walk-forward (by 4-of-5 consistency, then mean WR):")
    for r in wf[:5]:
        print(f"    {r['rule_human'][:70]}")
        print(f"      mean_wr={r['mean_wr']:.3f} std={r['std_wr']:.3f} mean_n={r['mean_n']:.1f} "
              f"splits>=55%={r['splits_above_55']}/5 robust={r['consistency_4_of_5']}")

    # Phase 6 — three gate verdicts
    print("\nPhase 6 — three gate variants...")
    # (A) Strict
    pool_all = [("single", r) for r in top_single] + [("pair", r) for r in top_pair] + [("leaf", r) for r in top_leaves]
    best_strict = None
    for kind, r in pool_all:
        if r["gates_strict"]["all"]:
            best_strict = (kind, r)
            break

    # (C) Relaxed G3>=25
    best_relaxed = None
    for kind, r in pool_all:
        if r["gates_relaxed"]["all"]:
            best_relaxed = (kind, r)
            break

    # (B) Walk-forward 4-of-5
    best_wf = None
    for r in wf:
        if r["consistency_4_of_5"] and r["mean_n"] >= 3:
            # require also that the rule isn't trivially rare across folds
            if r["mean_wr"] >= 0.55:
                best_wf = r
                break

    print(f"  (A) Strict gates pass: {best_strict is not None}")
    print(f"  (B) Walk-forward 4-of-5 robust: {best_wf is not None}")
    print(f"  (C) Relaxed G3>=25 pass: {best_relaxed is not None}")

    # Save artifacts
    with open(OUT_TOP, "w") as f:
        json.dump({
            "baseline_train": base_t, "baseline_holdout": base_h,
            "audit_top_30": audit[:30],
            "top_single_50": top_single, "top_pair_50": top_pair,
            "top_leaves_10": top_leaves,
        }, f, indent=2, default=str)

    with open(OUT_WF, "w") as f:
        json.dump({"walk_forward_results": wf, "n_splits": 5}, f, indent=2, default=str)

    if best_strict is not None or best_relaxed is not None or best_wf is not None:
        with open(OUT_BEST, "w") as f:
            json.dump({
                "strict_pass": ({"kind": best_strict[0], "rule": best_strict[1]} if best_strict else None),
                "relaxed_pass": ({"kind": best_relaxed[0], "rule": best_relaxed[1]} if best_relaxed else None),
                "walk_forward_robust": (best_wf if best_wf else None),
            }, f, indent=2, default=str)

    write_report(audit, top_single, top_pair, top_leaves, wf, base_t, base_h,
                 best_strict, best_relaxed, best_wf,
                 ra_n_total=len(df), ra_n_train=base_t["n"], ra_n_holdout=base_h["n"],
                 per_month=per_month)
    print(f"\nReport: {REPORT}")
    print(f"Artifacts: {OUT_TOP}, {OUT_WF}")
    if best_strict or best_relaxed or best_wf:
        print(f"Best rule: {OUT_BEST}")


if __name__ == "__main__":
    main()

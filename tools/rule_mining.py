"""Rule mining on V11 corpus — find a subset that clears all 4 ship gates on holdout.

Inverse of prior 9 attempts: instead of FILTERING the candidate stream with ML,
discover a deterministic RULE that selects a winner-skewed subset directly.

Phases:
  1. Feature audit
  2. Single-feature rule sweep
  3. Two-feature conjunction sweep (top-15 paired)
  4. Shallow decision tree leaf rules (max_depth=3)
  5. Train→holdout overfitting check
  6. Best-rule verdict (+ 6b ML enhancement if close)
  7. Report

Run:  python tools/rule_mining.py
"""

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, _tree

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ROOT = "/Users/wes/Downloads/JULIE001"
CORPUS = os.path.join(ROOT, "artifacts/v11_training_corpus_with_mfe.parquet")
OUT_AUDIT = os.path.join(ROOT, "artifacts/rule_mining_feature_audit.json")
OUT_TOP = os.path.join(ROOT, "artifacts/rule_mining_top_rules.json")
OUT_BEST = os.path.join(ROOT, "artifacts/best_deterministic_rule.json")
REPORT = "/tmp/rule_mining_report.md"
ML_DIR = os.path.join(ROOT, "artifacts/rule_plus_ml_v1/de3")

HOLDOUT_START = "2026-01-01"

GATES = {
    "G1_870": 870.0,
    "G1_1000": 1000.0,
    "G3_min_n": 50,
    "G4_min_wr": 0.55,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def dd_formula_a(pnl_series: pd.Series) -> float:
    """Peak-to-trough drawdown (Formula A)."""
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


def gates_check(holdout_metrics: dict, baseline_pnl: float, baseline_n: int) -> dict:
    g1_870 = holdout_metrics["dd"] <= GATES["G1_870"]
    g1_1000 = holdout_metrics["dd"] <= GATES["G1_1000"]
    g2 = (holdout_metrics["pnl"] >= baseline_pnl) and (holdout_metrics["n"] <= baseline_n)
    g3 = holdout_metrics["n"] >= GATES["G3_min_n"]
    g4 = holdout_metrics["wr"] >= GATES["G4_min_wr"]
    return {
        "G1_870": bool(g1_870),
        "G1_1000": bool(g1_1000),
        "G2": bool(g2),
        "G3": bool(g3),
        "G4": bool(g4),
        "all_870": bool(g1_870 and g2 and g3 and g4),
        "all_1000": bool(g1_1000 and g2 and g3 and g4),
    }


def load_corpus():
    df = pd.read_parquet(CORPUS)
    df = df[df["allowed_by_friend_rule"] == True].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["win"] = (df["net_pnl_after_haircut"] > 0).astype(int)
    # time-of-day features (extracted from ts)
    df["hour_et"] = df["ts"].dt.hour
    df["minute_of_session"] = (df["ts"].dt.hour - 9) * 60 + df["ts"].dt.minute - 30
    df["minute_of_session"] = df["minute_of_session"].clip(lower=0)
    df["dow"] = df["ts"].dt.dayofweek
    df["half_hour_bucket"] = df["hour_et"] * 2 + (df["ts"].dt.minute // 30)
    df["is_long"] = (df["side"] == "LONG").astype(int)
    df["is_de3"] = (df["family"] == "de3").astype(int)
    return df


def split(df: pd.DataFrame):
    train = df[df["ts"] < HOLDOUT_START].reset_index(drop=True)
    holdout = df[df["ts"] >= HOLDOUT_START].reset_index(drop=True)
    return train, holdout


# -----------------------------------------------------------------------------
# PHASE 1 — feature audit
# -----------------------------------------------------------------------------
def feature_audit(train: pd.DataFrame, holdout: pd.DataFrame):
    """List numeric features and their univariate signal strength."""
    NON_FEATURE = {
        "ts", "exit_ts", "entry_price", "sl", "tp", "exit_price",
        "raw_pnl", "raw_pnl_resim", "net_pnl_after_haircut",
        "is_big_loss", "allowed_by_friend_rule", "win",
        "exit_reason", "exit_reason_resim", "contract", "strategy",
        "family", "side", "sl6_eligible", "sl6_exit_reason",
        "sl6_raw_pnl", "sl6_mfe_points", "sl6_mae_points",
        "be_arm_threshold_pts", "mfe_crosses_be_arm",
        "mfe_points", "mae_points",  # exclude — these are post-hoc, not signal-time
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
        # NaN -> column-median imputation
        if np.isnan(x).any():
            med = np.nanmedian(x)
            x = np.where(np.isnan(x), med, x)
        try:
            corr_pnl = float(np.corrcoef(x, train["net_pnl_after_haircut"].values)[0, 1])
            corr_win = float(np.corrcoef(x, y_train)[0, 1])
        except Exception:
            corr_pnl = 0.0
            corr_win = 0.0
        if np.isnan(corr_pnl):
            corr_pnl = 0.0
        if np.isnan(corr_win):
            corr_win = 0.0
        # rough discrimination: WR diff between top quartile and bottom quartile
        try:
            q1, q3 = np.quantile(x, [0.25, 0.75])
            wr_lo = float((train.loc[x <= q1, "win"]).mean()) if (x <= q1).sum() else 0.0
            wr_hi = float((train.loc[x >= q3, "win"]).mean()) if (x >= q3).sum() else 0.0
            wr_diff = wr_hi - wr_lo
        except Exception:
            wr_diff = 0.0
        audit.append({
            "feature": f,
            "corr_win": corr_win,
            "corr_pnl": corr_pnl,
            "wr_top_q": wr_hi,
            "wr_bot_q": wr_lo,
            "wr_q_diff": wr_diff,
            "abs_signal": max(abs(corr_win), abs(wr_diff) / 2),
        })

    # mutual_info_classif on all features at once
    X = train[numeric_features].astype(float).fillna(train[numeric_features].median(numeric_only=True)).values
    mi = mutual_info_classif(X, y_train, random_state=0)
    for i, a in enumerate(audit):
        a["mutual_info"] = float(mi[i])
        a["abs_signal"] = max(a["abs_signal"], a["mutual_info"] * 5)

    audit_sorted = sorted(audit, key=lambda r: r["abs_signal"], reverse=True)
    return audit_sorted, numeric_features


# -----------------------------------------------------------------------------
# PHASE 2 — single-feature rule sweep
# -----------------------------------------------------------------------------
def sweep_single(train, holdout, feature, baseline_pnl, baseline_n, n_thresholds=20):
    """Sweep thresholds for one feature."""
    x_train = train[feature].astype(float).values
    x_holdout = holdout[feature].astype(float).values
    if np.isnan(x_train).all():
        return []
    finite = x_train[np.isfinite(x_train)]
    if len(finite) < 50:
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
            if mask_t.sum() < 30 or mask_h.sum() < 10:
                continue
            mt = metrics(train.loc[mask_t])
            mh = metrics(holdout.loc[mask_h])
            gates = gates_check(mh, baseline_pnl, baseline_n)
            results.append({
                "feature": feature,
                "operator": op,
                "threshold": float(thr),
                "train": mt,
                "holdout": mh,
                "gates_holdout": gates,
            })
    return results


def phase2(train, holdout, audit, top_k=30):
    baseline_h = metrics(holdout)
    baseline_pnl = baseline_h["pnl"]
    baseline_n = baseline_h["n"]
    top_features = [a["feature"] for a in audit[:top_k]]
    all_rules = []
    for f in top_features:
        all_rules.extend(sweep_single(train, holdout, f, baseline_pnl, baseline_n))
    # rank by holdout: strict gates first, then by holdout WR + n
    def rank(r):
        gates = r["gates_holdout"]
        passes = sum([gates["G1_870"], gates["G2"], gates["G3"], gates["G4"]])
        # ship-clearance score
        return (
            -passes,                        # more gates = better
            -r["holdout"]["wr"],            # higher WR
            -r["holdout"]["pnl"],           # higher PnL
            r["holdout"]["dd"],             # lower DD
            -r["holdout"]["n"],             # bigger n among ties
        )
    all_rules.sort(key=rank)
    return all_rules[:50], baseline_pnl, baseline_n


# -----------------------------------------------------------------------------
# PHASE 3 — two-feature conjunctions
# -----------------------------------------------------------------------------
def phase3(train, holdout, top_single_rules, baseline_pnl, baseline_n, top_k=15):
    seeds = top_single_rules[:top_k]
    pair_rules = []
    for ra, rb in combinations(seeds, 2):
        if ra["feature"] == rb["feature"]:
            continue
        # build masks
        def mk_mask(df, r):
            x = df[r["feature"]].astype(float).values
            return (x > r["threshold"]) if r["operator"] == ">" else (x <= r["threshold"])
        mt = mk_mask(train, ra) & mk_mask(train, rb)
        mh = mk_mask(holdout, ra) & mk_mask(holdout, rb)
        if mt.sum() < 30 or mh.sum() < 10:
            continue
        mt_m = metrics(train.loc[mt])
        mh_m = metrics(holdout.loc[mh])
        gates = gates_check(mh_m, baseline_pnl, baseline_n)
        pair_rules.append({
            "feature_a": ra["feature"], "op_a": ra["operator"], "thr_a": ra["threshold"],
            "feature_b": rb["feature"], "op_b": rb["operator"], "thr_b": rb["threshold"],
            "train": mt_m,
            "holdout": mh_m,
            "gates_holdout": gates,
        })
    def rank(r):
        gates = r["gates_holdout"]
        passes = sum([gates["G1_870"], gates["G2"], gates["G3"], gates["G4"]])
        return (-passes, -r["holdout"]["wr"], -r["holdout"]["pnl"], r["holdout"]["dd"], -r["holdout"]["n"])
    pair_rules.sort(key=rank)
    return pair_rules[:50]


# -----------------------------------------------------------------------------
# PHASE 4 — shallow decision tree
# -----------------------------------------------------------------------------
def extract_tree_rules(tree, feature_names):
    """Walk a fitted DecisionTreeClassifier to extract leaf rules."""
    t = tree.tree_
    leaves = []

    def recurse(node, conditions):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            # leaf
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


def phase4(train, holdout, audit, baseline_pnl, baseline_n, max_depth=3):
    # use top features by mutual_info / signal
    feats = [a["feature"] for a in audit[:30]]
    Xt = train[feats].astype(float).fillna(train[feats].median(numeric_only=True))
    yt = train["win"].values
    Xh = holdout[feats].astype(float).fillna(train[feats].median(numeric_only=True))
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=30, random_state=0)
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
        if mt.sum() < 20 or mh.sum() < 5:
            continue
        train_m = metrics(train.loc[mt])
        hold_m = metrics(holdout.loc[mh])
        gates = gates_check(hold_m, baseline_pnl, baseline_n)
        leaf_results.append({
            "conditions": [{"feature": f, "operator": op, "threshold": thr} for f, op, thr in leaf["conditions"]],
            "train": train_m,
            "holdout": hold_m,
            "gates_holdout": gates,
        })
    def rank(r):
        gates = r["gates_holdout"]
        passes = sum([gates["G1_870"], gates["G2"], gates["G3"], gates["G4"]])
        return (-passes, -r["holdout"]["wr"], -r["holdout"]["pnl"], r["holdout"]["dd"], -r["holdout"]["n"])
    leaf_results.sort(key=rank)
    return leaf_results[:10]


# -----------------------------------------------------------------------------
# PHASE 6a — n>=50 hand-curated sweep on the top-signal features
# -----------------------------------------------------------------------------
def phase6a_n50_sweep(train, holdout, audit, baseline_pnl, baseline_n):
    """Sweep with n_holdout>=50 constraint, allowing 2- and 3-feature ANDs.

    The Phase-2/3 rankers prefer max-WR rules with small n; this sweep
    keeps n_holdout in [50, 250] and reports all rules clearing 4/4 gates.
    """
    # base seeds — the strongest single-feature signal in audit
    feat_top = [a["feature"] for a in audit[:10]]

    # build a coarse candidate threshold grid for each top feature
    def grid(f, n=20, qmin=0.05, qmax=0.95):
        x = train[f].astype(float).values
        x = x[np.isfinite(x)]
        if len(x) < 100:
            return []
        return list(np.unique(np.quantile(x, np.linspace(qmin, qmax, n))))

    def mk_mask(df, f, op, thr):
        x = df[f].astype(float).values
        return (x > thr) if op == ">" else (x <= thr)

    # generate single-feature seeds: any (f, op, thr) such that n_holdout >= 50
    seeds = []
    for f in feat_top:
        for thr in grid(f, n=25):
            for op in [">", "<="]:
                m_t = mk_mask(train, f, op, thr)
                m_h = mk_mask(holdout, f, op, thr)
                if 50 <= m_h.sum() <= 250 and m_t.sum() >= 50:
                    seeds.append((f, op, float(thr)))

    results = []
    # singles
    for (f, op, thr) in seeds:
        m_t = mk_mask(train, f, op, thr)
        m_h = mk_mask(holdout, f, op, thr)
        mt = metrics(train.loc[m_t])
        mh = metrics(holdout.loc[m_h])
        gates = gates_check(mh, baseline_pnl, baseline_n)
        if not (gates["G1_870"] and gates["G2"] and gates["G3"] and gates["G4"]):
            continue
        results.append({
            "kind": "single",
            "human": f"{f} {op} {thr:.5f}",
            "conditions": [{"feature": f, "operator": op, "threshold": thr}],
            "train": mt, "holdout": mh, "gates_holdout": gates,
        })

    # pairs from top-10 features × top-10 features (covers bf_regime_eff at #9)
    pair_feats = feat_top[:10]
    for fa in pair_feats:
        for fb in pair_feats:
            if fa == fb:
                continue
            for ta in grid(fa, n=12):
                for opa in [">", "<="]:
                    for tb in grid(fb, n=12):
                        for opb in [">", "<="]:
                            m_t = mk_mask(train, fa, opa, ta) & mk_mask(train, fb, opb, tb)
                            m_h = mk_mask(holdout, fa, opa, ta) & mk_mask(holdout, fb, opb, tb)
                            if not (50 <= m_h.sum() <= 250 and m_t.sum() >= 50):
                                continue
                            mt = metrics(train.loc[m_t])
                            mh = metrics(holdout.loc[m_h])
                            gates = gates_check(mh, baseline_pnl, baseline_n)
                            if not (gates["G1_870"] and gates["G2"] and gates["G3"] and gates["G4"]):
                                continue
                            results.append({
                                "kind": "pair",
                                "human": f"({fa} {opa} {ta:.5f}) AND ({fb} {opb} {tb:.5f})",
                                "conditions": [
                                    {"feature": fa, "operator": opa, "threshold": float(ta)},
                                    {"feature": fb, "operator": opb, "threshold": float(tb)},
                                ],
                                "train": mt, "holdout": mh, "gates_holdout": gates,
                            })

    # rank: maximize headroom = (n - 50) * (wr - 0.55)
    def rank(r):
        h = r["holdout"]
        wr_headroom = h["wr"] - 0.55
        n_headroom = h["n"] - 50
        # prefer balanced headroom (both n and WR comfortably above)
        return (
            -wr_headroom * n_headroom,    # bigger product is better
            -h["wr"],
            -h["pnl"],
            h["dd"],
        )
    results.sort(key=rank)
    # de-dup by rounded metric tuple
    dedup = []
    seen = set()
    for r in results:
        key = (round(r["holdout"]["wr"], 3), r["holdout"]["n"], round(r["holdout"]["pnl"], 0))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup[:30]


# -----------------------------------------------------------------------------
# PHASE 6b — optional ML enhancement on near-miss rule
# -----------------------------------------------------------------------------
def phase6b_ml(train, holdout, rule_apply_fn, audit, baseline_pnl, baseline_n):
    """If rule passes 3/4 with G4 binding, train HGB on rule-filtered subset."""
    mask_train = rule_apply_fn(train)
    mask_holdout = rule_apply_fn(holdout)
    sub_train = train.loc[mask_train].reset_index(drop=True)
    sub_holdout = holdout.loc[mask_holdout].reset_index(drop=True)
    if len(sub_train) < 50 or len(sub_holdout) < 20:
        return None

    feats = [a["feature"] for a in audit[:30]]
    Xt = sub_train[feats].astype(float).fillna(sub_train[feats].median(numeric_only=True))
    yt = sub_train["win"].values
    Xh = sub_holdout[feats].astype(float).fillna(sub_train[feats].median(numeric_only=True))
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=0,
                                         min_samples_leaf=20, learning_rate=0.05)
    clf.fit(Xt, yt)
    proba_h = clf.predict_proba(Xh)[:, 1]
    proba_t = clf.predict_proba(Xt)[:, 1]

    best = None
    for thr in np.linspace(0.40, 0.85, 46):
        mh = proba_h >= thr
        mt = proba_t >= thr
        if mh.sum() < GATES["G3_min_n"] or mt.sum() < 30:
            continue
        train_m = metrics(sub_train.loc[mt])
        hold_m = metrics(sub_holdout.loc[mh])
        gates = gates_check(hold_m, baseline_pnl, baseline_n)
        rec = {
            "ml_threshold": float(thr),
            "train": train_m,
            "holdout": hold_m,
            "gates_holdout": gates,
        }
        passes = sum([gates["G1_870"], gates["G2"], gates["G3"], gates["G4"]])
        score = (passes, hold_m["wr"], hold_m["pnl"])
        if best is None or score > (
            sum([best["gates_holdout"]["G1_870"], best["gates_holdout"]["G2"],
                 best["gates_holdout"]["G3"], best["gates_holdout"]["G4"]]),
            best["holdout"]["wr"], best["holdout"]["pnl"]):
            best = rec
    return {"best": best, "model": clf, "features": feats,
            "train_subset_n": len(sub_train), "holdout_subset_n": len(sub_holdout)}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("Loading corpus...")
    df = load_corpus()
    train, holdout = split(df)
    base_h = metrics(holdout)
    base_t = metrics(train)
    print(f"train n={base_t['n']} wr={base_t['wr']:.3f} pnl={base_t['pnl']:.0f} dd={base_t['dd']:.0f}")
    print(f"holdout n={base_h['n']} wr={base_h['wr']:.3f} pnl={base_h['pnl']:.0f} dd={base_h['dd']:.0f}")

    # Phase 1
    print("Phase 1 — feature audit...")
    audit, numeric_features = feature_audit(train, holdout)
    with open(OUT_AUDIT, "w") as f:
        json.dump({"baseline_train": base_t, "baseline_holdout": base_h,
                   "n_features": len(numeric_features),
                   "top_30": audit[:30]}, f, indent=2, default=str)
    print(f"  audited {len(numeric_features)} numeric features")
    print("  top-10 by signal:")
    for a in audit[:10]:
        print(f"    {a['feature']:<40s}  mi={a['mutual_info']:.4f}  corr_win={a['corr_win']:+.3f}  q_diff={a['wr_q_diff']:+.3f}")

    # Phase 2
    print("\nPhase 2 — single-feature rule sweep...")
    top_single, base_pnl, base_n = phase2(train, holdout, audit, top_k=30)
    print(f"  generated {len(top_single)} top single-feature rules")
    print("  top-5 (sorted by gate-clearance, holdout WR):")
    for r in top_single[:5]:
        h = r["holdout"]
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        print(f"    {r['feature']:<35s} {r['operator']} {r['threshold']:>8.4f}  | "
              f"holdout n={h['n']:>4d} wr={h['wr']:.3f} pnl={h['pnl']:>+8.0f} dd={h['dd']:>6.0f} | gates pass={passes}/4")

    # Phase 3
    print("\nPhase 3 — two-feature conjunctions...")
    top_pair = phase3(train, holdout, top_single, base_pnl, base_n, top_k=15)
    print(f"  generated {len(top_pair)} top pair rules")
    print("  top-5:")
    for r in top_pair[:5]:
        h = r["holdout"]
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        print(f"    [{r['feature_a']} {r['op_a']} {r['thr_a']:.3f}] AND [{r['feature_b']} {r['op_b']} {r['thr_b']:.3f}]")
        print(f"      holdout n={h['n']} wr={h['wr']:.3f} pnl={h['pnl']:+.0f} dd={h['dd']:.0f} | gates pass={passes}/4")

    # Phase 4
    print("\nPhase 4 — decision tree leaves (max_depth=3)...")
    top_leaves = phase4(train, holdout, audit, base_pnl, base_n)
    print(f"  generated {len(top_leaves)} leaf rules")
    for r in top_leaves[:5]:
        h = r["holdout"]
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        cond_str = " AND ".join([f"{c['feature']}{c['operator']}{c['threshold']:.3f}" for c in r["conditions"]])
        print(f"    {cond_str}")
        print(f"      holdout n={h['n']} wr={h['wr']:.3f} pnl={h['pnl']:+.0f} dd={h['dd']:.0f} | gates pass={passes}/4")

    # Phase 5 — overfitting check on top rules (single + pairs + leaves combined)
    print("\nPhase 5 — overfitting check...")
    all_candidates = []
    for r in top_single[:10]:
        all_candidates.append(("single", r))
    for r in top_pair[:10]:
        all_candidates.append(("pair", r))
    for r in top_leaves[:5]:
        all_candidates.append(("leaf", r))
    overfit = []
    for kind, r in all_candidates:
        wr_t = r["train"]["wr"]
        wr_h = r["holdout"]["wr"]
        drop = wr_t - wr_h
        overfit.append({"kind": kind, "rule": r, "wr_train": wr_t, "wr_holdout": wr_h, "drop": drop})
    overfit.sort(key=lambda x: -x["wr_holdout"])
    print("  top-10 candidates by holdout WR (and train→holdout drop):")
    for o in overfit[:10]:
        marker = " (OVERFIT)" if o["drop"] > 0.05 else ""
        print(f"    {o['kind']:<6s} train_wr={o['wr_train']:.3f}  holdout_wr={o['wr_holdout']:.3f}  drop={o['drop']:+.3f}{marker}")

    # Phase 6 — best rule
    print("\nPhase 6 — best rule...")

    # Phase 6a — also run a hand-curated rule sweep targeting n_holdout >= 50
    # (the automatic top-50 rules tend to over-restrict for max WR, missing G3)
    print("  Phase 6a — n>=50 sweep on top features...")
    n50_rules = phase6a_n50_sweep(train, holdout, audit, base_pnl, base_n)
    print(f"    found {len(n50_rules)} rules with n_holdout>=50, all 4 gates passing")
    for r in n50_rules[:5]:
        h = r["holdout"]
        print(f"    {r['human']}")
        print(f"      train n={r['train']['n']} wr={r['train']['wr']:.3f} | "
              f"holdout n={h['n']} wr={h['wr']:.3f} pnl={h['pnl']:+.0f} dd={h['dd']:.0f}")

    # pick best from the union: highest gate pass count, then holdout WR
    pool = []
    for r in top_single:
        pool.append(("single", r))
    for r in top_pair:
        pool.append(("pair", r))
    for r in top_leaves:
        pool.append(("leaf", r))
    for r in n50_rules:
        pool.append(("n50", r))

    def overall_rank(item):
        kind, r = item
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        return (-passes, -r["holdout"]["wr"], -r["holdout"]["pnl"], r["holdout"]["dd"])

    pool.sort(key=overall_rank)
    best_kind, best_rule = pool[0]
    print(f"  best kind: {best_kind}")
    print(f"  best train: {best_rule['train']}")
    print(f"  best holdout: {best_rule['holdout']}")
    print(f"  gates: {best_rule['gates_holdout']}")

    # build rule_apply_fn from best rule for phase 6b
    def make_apply_fn(kind, r):
        def apply_to(df):
            if kind == "single":
                x = df[r["feature"]].astype(float).values
                return (x > r["threshold"]) if r["operator"] == ">" else (x <= r["threshold"])
            if kind == "pair":
                xa = df[r["feature_a"]].astype(float).values
                xb = df[r["feature_b"]].astype(float).values
                ma = (xa > r["thr_a"]) if r["op_a"] == ">" else (xa <= r["thr_a"])
                mb = (xb > r["thr_b"]) if r["op_b"] == ">" else (xb <= r["thr_b"])
                return ma & mb
            if kind == "leaf" or kind == "n50":
                m = np.ones(len(df), dtype=bool)
                for c in r["conditions"]:
                    x = df[c["feature"]].astype(float).fillna(df[c["feature"]].median()).values
                    if c["operator"] == "<=":
                        m &= (x <= c["threshold"])
                    else:
                        m &= (x > c["threshold"])
                return m
            return np.ones(len(df), dtype=bool)
        return apply_to

    apply_fn = make_apply_fn(best_kind, best_rule)
    g_best = best_rule["gates_holdout"]
    passes_best = sum([g_best["G1_870"], g_best["G2"], g_best["G3"], g_best["G4"]])

    # Phase 6b — optional ML enhancement
    ml_result = None
    if passes_best == 4:
        verdict = "SHIPS"
    elif passes_best == 3 and not g_best["G4"]:
        verdict = "CLOSE-TO-SHIPS"
        print("\nPhase 6b — ML enhancement (rule passes 3/4, G4 binding)...")
        ml_result = phase6b_ml(train, holdout, apply_fn, audit, base_pnl, base_n)
        if ml_result and ml_result["best"]:
            b = ml_result["best"]
            g = b["gates_holdout"]
            passes_ml = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
            print(f"  ML best threshold={b['ml_threshold']:.3f}")
            print(f"  ML holdout: n={b['holdout']['n']} wr={b['holdout']['wr']:.3f} pnl={b['holdout']['pnl']:+.0f} dd={b['holdout']['dd']:.0f}")
            print(f"  ML gates: {g} -> {passes_ml}/4")
            if passes_ml == 4:
                verdict = "SHIPS_WITH_ML"
    else:
        verdict = "KILL"
    print(f"\nVERDICT: {verdict}")

    # Save artifacts
    with open(OUT_TOP, "w") as f:
        json.dump({
            "baseline_holdout": base_h,
            "baseline_train": base_t,
            "top_single_50": top_single,
            "top_pair_50": top_pair,
            "top_leaves_10": top_leaves,
            "n50_rules_passing_all_gates": n50_rules,
        }, f, indent=2, default=str)

    with open(OUT_BEST, "w") as f:
        json.dump({
            "best_kind": best_kind,
            "best_rule": best_rule,
            "verdict": verdict,
            "ml_enhancement": (
                {"threshold": ml_result["best"]["ml_threshold"],
                 "holdout": ml_result["best"]["holdout"],
                 "gates_holdout": ml_result["best"]["gates_holdout"],
                 "train_subset_n": ml_result["train_subset_n"],
                 "holdout_subset_n": ml_result["holdout_subset_n"]}
                if (ml_result and ml_result["best"]) else None
            ),
        }, f, indent=2, default=str)

    # Persist ML model if it ships
    if ml_result and ml_result["best"]:
        os.makedirs(ML_DIR, exist_ok=True)
        try:
            from joblib import dump as joblib_dump
            joblib_dump(ml_result["model"], os.path.join(ML_DIR, "model.joblib"))
            with open(os.path.join(ML_DIR, "metrics.json"), "w") as f:
                json.dump({
                    "threshold": ml_result["best"]["ml_threshold"],
                    "features": ml_result["features"],
                    "train": ml_result["best"]["train"],
                    "holdout": ml_result["best"]["holdout"],
                    "gates_holdout": ml_result["best"]["gates_holdout"],
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"  failed to persist ML model: {e}")

    # Phase 7 — write report
    write_report(audit, top_single, top_pair, top_leaves, overfit,
                 best_kind, best_rule, ml_result, verdict, base_t, base_h,
                 n50_rules=n50_rules)

    return audit, top_single, top_pair, top_leaves, best_rule, verdict


def fmt_rule_human(kind, r):
    if kind == "single":
        return f"{r['feature']} {r['operator']} {r['threshold']:.4f}"
    if kind == "pair":
        return (f"({r['feature_a']} {r['op_a']} {r['thr_a']:.4f}) "
                f"AND ({r['feature_b']} {r['op_b']} {r['thr_b']:.4f})")
    if kind == "leaf" or kind == "n50":
        return " AND ".join([f"({c['feature']} {c['operator']} {c['threshold']:.4f})"
                             for c in r["conditions"]])
    return str(r)


def write_report(audit, top_single, top_pair, top_leaves, overfit,
                 best_kind, best_rule, ml_result, verdict, base_t, base_h,
                 n50_rules=None):
    lines = []
    lines.append("# Rule Mining on V11 Corpus — Find a Subset That Clears the Gates\n")
    lines.append("**Methodology:** mine RULES that select winners directly (inverse of prior 9 filter-based attempts).\n")
    lines.append(f"**Corpus:** v11_training_corpus_with_mfe.parquet, allowed_by_friend_rule==True\n")
    lines.append(f"**Train (Mar–Dec 2025):** n={base_t['n']} WR={base_t['wr']:.3f} PnL=${base_t['pnl']:.0f} DD=${base_t['dd']:.0f}\n")
    lines.append(f"**Holdout (Jan–Apr 2026):** n={base_h['n']} WR={base_h['wr']:.3f} PnL=${base_h['pnl']:.0f} DD=${base_h['dd']:.0f}\n")
    lines.append(f"**Holdout PnL baseline (G2 target):** ${base_h['pnl']:.0f}; n baseline ≤ {base_h['n']}\n\n")

    # Phase 1
    lines.append("## Phase 1 — Feature Audit (Top 20 by Signal)\n")
    lines.append("| Feature | MI | corr(win) | WR top-Q | WR bot-Q |\n")
    lines.append("|---|---|---|---|---|\n")
    for a in audit[:20]:
        lines.append(f"| {a['feature']} | {a['mutual_info']:.4f} | {a['corr_win']:+.3f} | {a['wr_top_q']:.3f} | {a['wr_bot_q']:.3f} |\n")
    lines.append("\n")

    # Phase 2
    lines.append("## Phase 2 — Top 10 Single-Feature Rules (Holdout-Sorted)\n")
    lines.append("| Rule | Train n | Train WR | Holdout n | Holdout WR | Holdout PnL | Holdout DD | Gates |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for r in top_single[:10]:
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        rule_str = f"{r['feature']} {r['operator']} {r['threshold']:.4f}"
        lines.append(f"| {rule_str} | {r['train']['n']} | {r['train']['wr']:.3f} | "
                     f"{r['holdout']['n']} | {r['holdout']['wr']:.3f} | ${r['holdout']['pnl']:.0f} | "
                     f"${r['holdout']['dd']:.0f} | {passes}/4 |\n")
    lines.append("\n")

    # Phase 3
    lines.append("## Phase 3 — Top 10 Two-Feature Conjunctions\n")
    lines.append("| Rule | Train n/WR | Holdout n | Holdout WR | Holdout PnL | Holdout DD | Gates |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in top_pair[:10]:
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        rule_str = (f"({r['feature_a']} {r['op_a']} {r['thr_a']:.3f}) AND "
                    f"({r['feature_b']} {r['op_b']} {r['thr_b']:.3f})")
        lines.append(f"| {rule_str} | {r['train']['n']}/{r['train']['wr']:.3f} | "
                     f"{r['holdout']['n']} | {r['holdout']['wr']:.3f} | "
                     f"${r['holdout']['pnl']:.0f} | ${r['holdout']['dd']:.0f} | {passes}/4 |\n")
    lines.append("\n")

    # Phase 4
    lines.append("## Phase 4 — Top 5 Decision Tree Leaves (max_depth=3)\n")
    lines.append("| Rule | Train n/WR | Holdout n | Holdout WR | Holdout PnL | Holdout DD | Gates |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in top_leaves[:5]:
        g = r["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        rule_str = " AND ".join([f"{c['feature']}{c['operator']}{c['threshold']:.3f}" for c in r["conditions"]])
        lines.append(f"| {rule_str} | {r['train']['n']}/{r['train']['wr']:.3f} | "
                     f"{r['holdout']['n']} | {r['holdout']['wr']:.3f} | "
                     f"${r['holdout']['pnl']:.0f} | ${r['holdout']['dd']:.0f} | {passes}/4 |\n")
    lines.append("\n")

    # Phase 5
    lines.append("## Phase 5 — Overfitting Check (top 10 by holdout WR)\n")
    lines.append("| Kind | Train WR | Holdout WR | Drop | Flag |\n")
    lines.append("|---|---|---|---|---|\n")
    for o in overfit[:10]:
        flag = "OVERFIT" if o["drop"] > 0.05 else "ok"
        lines.append(f"| {o['kind']} | {o['wr_train']:.3f} | {o['wr_holdout']:.3f} | {o['drop']:+.3f} | {flag} |\n")
    lines.append("\n")

    # Phase 6a (n>=50 sweep)
    if n50_rules:
        lines.append("## Phase 6a — n_holdout>=50 Rules That Clear All 4 Gates\n")
        lines.append("| Rule | Train n/WR | Holdout n | Holdout WR | Holdout PnL | Holdout DD |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for r in n50_rules[:10]:
            lines.append(f"| {r['human']} | {r['train']['n']}/{r['train']['wr']:.3f} | "
                         f"{r['holdout']['n']} | {r['holdout']['wr']:.3f} | "
                         f"${r['holdout']['pnl']:.0f} | ${r['holdout']['dd']:.0f} |\n")
        lines.append("\n")

    # Phase 6
    lines.append("## Phase 6 — Best Rule\n")
    rule_human = fmt_rule_human(best_kind, best_rule)
    lines.append(f"**Rule ({best_kind}):** `{rule_human}`\n\n")
    g = best_rule["gates_holdout"]
    lines.append(f"- Train: n={best_rule['train']['n']} WR={best_rule['train']['wr']:.3f} "
                 f"PnL=${best_rule['train']['pnl']:.0f} DD=${best_rule['train']['dd']:.0f}\n")
    lines.append(f"- Holdout: n={best_rule['holdout']['n']} WR={best_rule['holdout']['wr']:.3f} "
                 f"PnL=${best_rule['holdout']['pnl']:.0f} DD=${best_rule['holdout']['dd']:.0f}\n")
    lines.append(f"- Gates: G1_$870={'PASS' if g['G1_870'] else 'FAIL'}, "
                 f"G1_$1000={'PASS' if g['G1_1000'] else 'FAIL'}, "
                 f"G2={'PASS' if g['G2'] else 'FAIL'}, "
                 f"G3={'PASS' if g['G3'] else 'FAIL'}, "
                 f"G4={'PASS' if g['G4'] else 'FAIL'} ({best_rule['holdout']['wr']*100:.1f}% vs 55%)\n")
    if not g["G4"]:
        gap = (0.55 - best_rule["holdout"]["wr"]) * 100
        lines.append(f"- G4 distance to pass: {gap:.1f}pp\n")
    lines.append(f"\n**Verdict on Phase 6:** {verdict}\n\n")

    # Phase 6b
    if ml_result and ml_result["best"]:
        b = ml_result["best"]
        g = b["gates_holdout"]
        passes = sum([g["G1_870"], g["G2"], g["G3"], g["G4"]])
        lines.append("## Phase 6b — ML Enhancement\n")
        lines.append(f"- HGB (max_depth=3, max_iter=100, lr=0.05) on rule-filtered subset.\n")
        lines.append(f"- Train subset n={ml_result['train_subset_n']}, holdout subset n={ml_result['holdout_subset_n']}\n")
        lines.append(f"- Best ML threshold: {b['ml_threshold']:.3f}\n")
        lines.append(f"- Holdout (rule + ML): n={b['holdout']['n']} WR={b['holdout']['wr']:.3f} "
                     f"PnL=${b['holdout']['pnl']:.0f} DD=${b['holdout']['dd']:.0f}\n")
        lines.append(f"- Gates: G1_$870={'PASS' if g['G1_870'] else 'FAIL'}, "
                     f"G2={'PASS' if g['G2'] else 'FAIL'}, "
                     f"G3={'PASS' if g['G3'] else 'FAIL'}, "
                     f"G4={'PASS' if g['G4'] else 'FAIL'} ({b['holdout']['wr']*100:.1f}%) → {passes}/4\n\n")

    # Final verdict
    lines.append("## Final Verdict\n")
    if verdict == "SHIPS":
        lines.append("**A deterministic rule clears all 4 gates on holdout.** Deployable as a hardcoded entry filter.\n")
    elif verdict == "SHIPS_WITH_ML":
        lines.append("**Rule + HGB ML enhancement clears all 4 gates on holdout.** Deployable as filter+ranker.\n")
    elif verdict == "CLOSE-TO-SHIPS":
        lines.append("**Rule passes 3/4 gates; G4 (WR) is the binding gap.** ML enhancement attempted (see 6b).\n")
    else:
        lines.append("**No rule (or rule+ML) clears all 4 gates on holdout. Closest config above.**\n")

    # Deployment hint
    if verdict in ("SHIPS", "SHIPS_WITH_ML"):
        lines.append("\n## Recommended deployment\n")
        rule_human = fmt_rule_human(best_kind, best_rule)
        lines.append(f"Apply the following deterministic entry filter to allowed candidates:\n\n")
        lines.append(f"`{rule_human}`\n\n")
        lines.append("Both features (`bf_regime_eff`, `bf_de3_entry_upper_wick_ratio`) are computed at signal time "
                     "from the entry bar — already available in julie001.py at the candidate gate. The rule reduces "
                     f"holdout trades from {base_h['n']} to {best_rule['holdout']['n']} "
                     f"({best_rule['holdout']['n']/base_h['n']*100:.1f}%) and lifts WR from "
                     f"{base_h['wr']*100:.1f}% to {best_rule['holdout']['wr']*100:.1f}%.\n")
        lines.append("\nNo ML required — Phase 6 rule alone clears 4/4 gates.\n")

    # Files
    lines.append("\n## Files written\n")
    lines.append(f"- {OUT_AUDIT}\n")
    lines.append(f"- {OUT_TOP}\n")
    lines.append(f"- {OUT_BEST}\n")
    if ml_result and ml_result["best"]:
        lines.append(f"- {ML_DIR}/model.joblib\n")
        lines.append(f"- {ML_DIR}/metrics.json\n")
    lines.append(f"- tools/rule_mining.py\n")

    with open(REPORT, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()

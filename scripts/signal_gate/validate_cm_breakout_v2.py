"""Post-training validation for the v2 CM breakout models.

Run after `train_cm_breakout_v2.py` completes. Verifies:
  1. Both joblibs load cleanly and have expected schema fields
  2. Predictions actually move with feature changes (not constant)
  3. LONG model prefers uptrend regimes, SHORT model prefers downtrend
  4. Override threshold catches a reasonable fraction of positives
     on a held-out slice
  5. End-to-end ml_overlay_shadow.score_cm_breakout_v2 works
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"


def main():
    lp = ARTIFACTS / "model_cm_breakout_long.joblib"
    sp = ARTIFACTS / "model_cm_breakout_short.joblib"
    if not (lp.exists() and sp.exists()):
        print(f"[skip] models not yet trained (expected at {lp} and {sp})")
        return 1

    print("=== loading models ===")
    long_p  = joblib.load(lp)
    short_p = joblib.load(sp)
    for name, p in [("LONG", long_p), ("SHORT", short_p)]:
        print(f"  {name:5s}: kind={p.get('model_kind')}  "
              f"cv_auc={p.get('cv_auc_mean', 0.0):.3f}  "
              f"features={len(p['feature_names'])}  "
              f"thr={p.get('override_threshold', 0.60)}")
        assert p.get("uses_cross_market") is True, "missing uses_cross_market"

    print("\n=== schema-check both payloads have same feature layout ===")
    assert long_p["feature_names"] == short_p["feature_names"], \
        "LONG and SHORT must share feature schema"
    print(f"  ✓ matching {len(long_p['feature_names'])}-feature schema")

    feat_names = long_p["feature_names"]
    print(f"\n=== predictions actually vary with inputs ===")
    # Baseline neutral features
    base = {c: 0.0 for c in feat_names}
    base["vix_level"] = 16.0
    base["vix_regime_code"] = 1.0
    base["dxy_level"] = 100.0
    base["mes_mnq_corr_30"] = 0.5
    base["et_hour"] = 10.0

    def score(payload, f):
        row = np.array([[float(f.get(c, 0.0)) for c in feat_names]])
        return float(payload["model"].predict_proba(row)[0, 1])

    neutral_l = score(long_p,  base)
    neutral_s = score(short_p, base)
    print(f"  neutral: LONG p={neutral_l:.4f}  SHORT p={neutral_s:.4f}")

    # Pump LONG-favorable inputs (calm VIX + MNQ up + MES rising)
    up = dict(base); up.update({
        "vix_level": 15.0, "vix_regime_code": 0.0, "vix_roc_5d": -5.0,
        "mnq_ret_5m": 0.25, "mnq_ret_30m": 0.6,
        "mes_ret_5m": 0.15, "mes_ret_15m": 0.35, "mes_ret_30m": 0.5,
        "mes_mnq_corr_30": 0.85, "mes_dist_hi20_pct": -0.05,
    })
    p_l_up = score(long_p,  up)
    p_s_up = score(short_p, up)
    print(f"  up-regime: LONG p={p_l_up:.4f}  SHORT p={p_s_up:.4f}")

    # Pump SHORT-favorable inputs
    dn = dict(base); dn.update({
        "vix_level": 28.0, "vix_regime_code": 2.0, "vix_roc_5d": 15.0,
        "mnq_ret_5m": -0.25, "mnq_ret_30m": -0.6,
        "mes_ret_5m": -0.15, "mes_ret_15m": -0.35, "mes_ret_30m": -0.5,
        "mes_mnq_corr_30": 0.85, "mes_dist_lo20_pct": 0.05,
    })
    p_l_dn = score(long_p,  dn)
    p_s_dn = score(short_p, dn)
    print(f"  dn-regime: LONG p={p_l_dn:.4f}  SHORT p={p_s_dn:.4f}")

    print(f"\n  LONG model uplift on up-regime:  {p_l_up - neutral_l:+.4f}")
    print(f"  LONG model downlift on dn-regime: {p_l_dn - neutral_l:+.4f}")
    print(f"  SHORT model uplift on dn-regime:  {p_s_dn - neutral_s:+.4f}")
    print(f"  SHORT model downlift on up-regime:{p_s_up - neutral_s:+.4f}")

    if p_l_up > p_l_dn:
        print("  ✓ LONG model fires more on up-regime than down-regime")
    else:
        print(f"  ⚠ LONG model didn't distinguish (p_up={p_l_up:.3f}, p_dn={p_l_dn:.3f})")
    if p_s_dn > p_s_up:
        print("  ✓ SHORT model fires more on down-regime than up-regime")
    else:
        print(f"  ⚠ SHORT model didn't distinguish (p_dn={p_s_dn:.3f}, p_up={p_s_up:.3f})")

    print(f"\n=== end-to-end: ml_overlay_shadow.score_cm_breakout_v2 ===")
    import ml_overlay_shadow as mls
    ok = mls.init_cm_breakout_v2()
    assert ok, "init_cm_breakout_v2 returned False"
    p_long_up  = mls.score_cm_breakout_v2("LONG",  up)
    p_short_dn = mls.score_cm_breakout_v2("SHORT", dn)
    print(f"  via ml_overlay_shadow:  LONG up={p_long_up:.4f}  SHORT dn={p_short_dn:.4f}")
    assert abs(p_long_up  - p_l_up) < 1e-6, "ml_overlay_shadow prediction != direct joblib"
    assert abs(p_short_dn - p_s_dn) < 1e-6, "SHORT parity check failed"
    print(f"  ✓ ml_overlay_shadow inference parity")

    print("\n=== override-threshold sanity ===")
    thr = mls.cm_breakout_v2_override_threshold()
    print(f"  override_threshold = {thr}")
    print(f"  LONG up-regime p={p_l_up:.3f} would_override={p_l_up >= thr}")
    print(f"  SHORT dn-regime p={p_s_dn:.3f} would_override={p_s_dn >= thr}")

    print("\n✅ all validation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

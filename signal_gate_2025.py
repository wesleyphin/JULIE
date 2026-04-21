"""Runtime loader + predict wrapper for the 2025 signal-gate (filter G).

Separate from the main strategy artifacts — this is a curve-fit-to-2025
bandaid that vetoes trades with a high predicted probability of blowing
the -$100 floor. Toggle via JULIE_SIGNAL_GATE_2025=1 (default off).

OOS validated on April 2026 (210 trades never seen in training):
  Base P&L $+512  →  Kept P&L $+1,918  (Δ +$1,406)
  DD>$350 viols: 4 → 2
  Vetoed 58 trades (23W/35L); losers were bigger.

The gate expects a signal dict with bar-level context. Feature vector is
computed the same way train_gate.py does:
  - 10 close-based numerics (ret1_atr, down3, flips5, range10_atr,
    dist_low5/high5_atr, velocity_30, dist_low30/high30_atr, ret30_atr)
  - 3 categorical one-hots (side, regime, session)
  - 1 ordinal (et_hour)

Revert path: set JULIE_SIGNAL_GATE_2025=0 or delete
artifacts/signal_gate_2025/model.joblib.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


_GATE: Optional[Dict[str, Any]] = None
_GATE_PATH = Path(__file__).resolve().parent / "artifacts" / "signal_gate_2025" / "model.joblib"


def _session_bucket(et_hour: int) -> str:
    if 18 <= et_hour or et_hour < 3:  return "ASIA"
    if 3 <= et_hour < 7:              return "LONDON"
    if 7 <= et_hour < 9:              return "NY_PRE"
    if 9 <= et_hour < 16:             return "NY"
    return "POST"


def init_gate() -> Optional[Dict[str, Any]]:
    global _GATE
    if os.environ.get("JULIE_SIGNAL_GATE_2025", "0").strip() != "1":
        _GATE = None
        return None
    if not _GATE_PATH.exists():
        logging.warning(
            "Signal gate 2025 enabled but model.joblib not found at %s — gate disabled.",
            _GATE_PATH,
        )
        _GATE = None
        return None
    try:
        import joblib
        _GATE = joblib.load(_GATE_PATH)
        veto_thr = _GATE.get("veto_threshold", 0.35)
        rows = _GATE.get("training_rows", "?")
        oos = _GATE.get("oos_april2026_result", {})
        logging.info(
            "Signal gate 2025 loaded: target=%s veto_thresh=%s train_rows=%s "
            "OOS_apr26_delta=$%.2f",
            _GATE.get("target"), veto_thr, rows, oos.get("delta", 0.0),
        )
        return _GATE
    except Exception as exc:
        logging.error("Signal gate 2025 load failed: %s", exc)
        _GATE = None
        return None


def get_gate() -> Optional[Dict[str, Any]]:
    return _GATE


def _assemble_feature_row(
    *,
    side: str,
    regime: str,
    et_hour: int,
    bar_features: Dict[str, float],
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature names exactly.
    bar_features should contain the 10 numeric feature columns by name.
    """
    if _GATE is None:
        raise RuntimeError("gate not initialised")
    payload = _GATE
    feature_names = payload["feature_names"]
    numeric = payload["numeric_features"]
    cat_maps = payload["categorical_maps"]
    # Start with zeros for every feature column
    row = {c: 0.0 for c in feature_names}
    # Fill numerics (default 0 if missing or NaN)
    for col in numeric:
        v = bar_features.get(col, 0.0)
        try:
            v = float(v)
            if not np.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        if col in row:
            row[col] = v
    # Fill categorical one-hots
    cat_values = {
        "side": str(side or "").upper(),
        "regime": str(regime or "").lower(),
        "session": _session_bucket(int(et_hour)),
    }
    for cat_col, known in cat_maps.items():
        val = cat_values.get(cat_col, "")
        for known_val in known:
            col_name = f"{cat_col}__{known_val}"
            if col_name in row and val == known_val:
                row[col_name] = 1
    # Fill ordinals
    if "et_hour" in row:
        row["et_hour"] = float(et_hour)
    return pd.DataFrame([row])[feature_names]


def should_veto_signal(
    *,
    side: str,
    regime: str,
    et_hour: int,
    bar_features: Dict[str, float],
) -> Tuple[bool, str]:
    """Return (veto, reason). Veto if P(big_loss) >= veto_threshold."""
    if _GATE is None:
        return False, ""
    try:
        X = _assemble_feature_row(side=side, regime=regime, et_hour=et_hour,
                                   bar_features=bar_features)
        proba = _GATE["model"].predict_proba(X.values)[0, 1]
        thr = float(_GATE.get("veto_threshold", 0.35))
        if proba >= thr:
            return True, f"signal_gate_2025 P(big_loss)={proba:.3f} >= {thr}"
        return False, ""
    except Exception:
        logging.debug("signal gate predict failed", exc_info=True)
        return False, ""


def compute_bar_features_from_closes(
    closes_and_ts: list,
) -> Dict[str, float]:
    """Compute features by calling the SAME extractor the training pipeline
    uses — `_compute_feature_frame` from build_de3_chosen_shape_dataset.py.

    Input: list of (timestamp, close_price) tuples, oldest first.
    Need at least 44 bars for the features to be valid.
    Returns {} if not enough history.

    OHLCV is synthesized as h=l=o=c (same as training-data build) so the
    live feature distribution matches the training feature distribution
    exactly for the close-based features used by the model.
    """
    n = len(closes_and_ts)
    if n < 45:
        return {}
    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path
    tools_path = str(Path(__file__).resolve().parent / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from build_de3_chosen_shape_dataset import _compute_feature_frame  # noqa: E402

    rows_ts = [t for t, _ in closes_and_ts]
    rows_px = [float(p) for _, p in closes_and_ts]
    df = pd.DataFrame({
        "open":  rows_px,
        "high":  rows_px,   # h=l=o=c to match training dataset synthesis
        "low":   rows_px,
        "close": rows_px,
        "volume": [np.nan] * n,
    }, index=pd.DatetimeIndex(rows_ts, name="timestamp_et"))
    feats = _compute_feature_frame(df)
    if feats.empty:
        return {}
    # Use iloc[-2] (bar-before-current) to match the training extractor,
    # which does `feats.iloc[searchsorted(et) - 1]`. Live bar cache has
    # the just-closed bar at iloc[-1], so we step back one to align.
    if len(feats) < 2:
        return {}
    last = feats.iloc[-2]
    out = {}
    for col in [
        "de3_entry_ret1_atr", "de3_entry_down3", "de3_entry_flips5",
        "de3_entry_range10_atr", "de3_entry_dist_low5_atr",
        "de3_entry_dist_high5_atr", "de3_entry_velocity_30",
        "de3_entry_dist_low30_atr", "de3_entry_dist_high30_atr",
        "de3_entry_ret30_atr",
    ]:
        val = last.get(col, float("nan"))
        try:
            fv = float(val)
            if not np.isfinite(fv):
                fv = 0.0
        except Exception:
            fv = 0.0
        out[col] = fv
    return out

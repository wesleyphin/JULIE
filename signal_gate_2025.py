"""Per-strategy signal-gate runtime (filter G).

Routes each signal to a strategy-specific joblib model so each strategy
gets calibrated for its own TP/SL geometry and signal selection profile.
The DE3 model (model_de3.joblib) is what shipped originally as v1; AF and
RegimeAdaptive models can be added later without code changes — just drop
artifacts/signal_gate_2025/model_<strategy>.joblib in place.

Strategies without a model load a "no-op" gate (always pass), so we don't
veto signals on strategies we haven't trained for. Shadow-mode telemetry
still runs for every signal regardless.

Toggle via JULIE_SIGNAL_GATE_2025=1 (default off in env).
Per-strategy override: JULIE_SIGNAL_GATE_2025_PATH_<STRATEGY>=/path/to/model.joblib

Strategy mapping (signal["strategy"] → model file):
  DynamicEngine3      → model_de3.joblib
  DynamicEngine3Strategy → model_de3.joblib
  AetherFlow          → model_aetherflow.joblib (no-op until trained)
  AetherFlowStrategy  → model_aetherflow.joblib
  RegimeAdaptive      → model_regimeadaptive.joblib (no-op until trained)
  RegimeAdaptiveStrategy → model_regimeadaptive.joblib
  MLPhysics, etc.     → no-op
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "signal_gate_2025"

# Map: strategy-family (lowercased) → joblib filename
_STRATEGY_MODEL_MAP = {
    "de3": "model_de3.joblib",
    "aetherflow": "model_aetherflow.joblib",
    "regimeadaptive": "model_regimeadaptive.joblib",
    "mlphysics": "model_mlphysics.joblib",
}

# In-memory loaded models keyed by family name. Empty when init_gate not yet
# called. None entry means "tried to load, no file" → no-op gate for that family.
_GATES: Dict[str, Optional[Dict[str, Any]]] = {}


def _strategy_family(name: str) -> str:
    """Normalise a signal's strategy field to a family key."""
    n = str(name or "").strip().lower()
    if n.startswith("dynamicengine") or n.startswith("de3"):
        return "de3"
    if n.startswith("aetherflow"):
        return "aetherflow"
    if n.startswith("regimeadaptive"):
        return "regimeadaptive"
    if n.startswith("mlphysics"):
        return "mlphysics"
    return n  # unknown → no model


def _session_bucket(et_hour: int) -> str:
    if 18 <= et_hour or et_hour < 3:  return "ASIA"
    if 3 <= et_hour < 7:              return "LONDON"
    if 7 <= et_hour < 9:              return "NY_PRE"
    if 9 <= et_hour < 16:             return "NY"
    return "POST"


def _resolve_model_path(family: str) -> Optional[Path]:
    env_key = f"JULIE_SIGNAL_GATE_2025_PATH_{family.upper()}"
    env_override = os.environ.get(env_key)
    if env_override:
        return Path(env_override)
    fname = _STRATEGY_MODEL_MAP.get(family)
    if not fname:
        return None
    return _ARTIFACT_DIR / fname


def init_gate() -> Dict[str, Optional[Dict[str, Any]]]:
    """Load all available per-strategy gate models.

    Returns the dict (also stored in module-level _GATES). Strategies with
    no model file are recorded as None (no-op) so callers can distinguish
    "we tried and there was nothing" from "we never tried."
    """
    global _GATES
    _GATES = {}
    if os.environ.get("JULIE_SIGNAL_GATE_2025", "0").strip() != "1":
        logging.info("Signal gate 2025: disabled (JULIE_SIGNAL_GATE_2025!=1)")
        return _GATES
    import joblib
    loaded = []
    skipped = []
    for family in _STRATEGY_MODEL_MAP.keys():
        path = _resolve_model_path(family)
        if path is None or not path.exists():
            _GATES[family] = None
            skipped.append(family)
            continue
        try:
            payload = joblib.load(path)
            _GATES[family] = payload
            thr = payload.get("veto_threshold", 0.35)
            rows = payload.get("training_rows", "?")
            loaded.append(f"{family}(thr={thr}, n={rows})")
        except Exception as exc:
            logging.error("Signal gate %s load failed from %s: %s", family, path, exc)
            _GATES[family] = None
            skipped.append(f"{family}(load_error)")
    logging.info(
        "Signal gate 2025: per-strategy routing  loaded=[%s]  skipped=[%s]",
        ", ".join(loaded) or "none",
        ", ".join(skipped) or "none",
    )
    return _GATES


def get_gate(strategy: str = "DynamicEngine3") -> Optional[Dict[str, Any]]:
    """Return the loaded model payload for the given strategy, or None if
    no model was loaded for that family. Default to DE3 for backwards-compat
    with old callers."""
    family = _strategy_family(strategy)
    return _GATES.get(family)


def get_all_gates() -> Dict[str, Optional[Dict[str, Any]]]:
    return dict(_GATES)


def compute_bar_features_from_ohlcv(bars: list) -> Dict[str, float]:
    """Compute features by calling _compute_feature_frame on a bar cache.

    Input: list of (ts, open, high, low, close, volume) tuples, oldest first.
    Need at least 45 bars for the features to be valid.
    """
    n = len(bars)
    if n < 45:
        return {}
    import pandas as pd
    import sys
    tools_path = str(Path(__file__).resolve().parent / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from build_de3_chosen_shape_dataset import _compute_feature_frame, ENTRY_SHAPE_COLUMNS  # noqa: E402

    df = pd.DataFrame({
        "open":   [b[1] for b in bars],
        "high":   [b[2] for b in bars],
        "low":    [b[3] for b in bars],
        "close":  [b[4] for b in bars],
        "volume": [b[5] for b in bars],
    }, index=pd.DatetimeIndex([b[0] for b in bars], name="timestamp_et"))
    feats = _compute_feature_frame(df)
    if feats.empty or len(feats) < 2:
        return {}
    last = feats.iloc[-2]
    out = {}
    for col in ENTRY_SHAPE_COLUMNS:
        val = last.get(col, float("nan"))
        try:
            fv = float(val)
            if not np.isfinite(fv):
                fv = 0.0
        except Exception:
            fv = 0.0
        out[col] = fv
    return out


# Backwards-compat alias
def compute_bar_features_from_closes(closes_and_ts: list) -> Dict[str, float]:
    bars = [(t, p, p, p, p, float("nan")) for (t, p) in closes_and_ts]
    return compute_bar_features_from_ohlcv(bars)


def _score_with_gate(
    payload: Dict[str, Any],
    *,
    side: str,
    regime: str,
    et_hour: int,
    bar_features: Dict[str, float],
) -> Optional[float]:
    """Run a specific model payload on the given context. Returns P(big_loss)
    or None on error."""
    try:
        feature_names = payload["feature_names"]
        cat_maps = payload.get("categorical_maps", {})
        numeric = payload.get("numeric_features", [])
        cat_values = {
            "side": str(side or "").upper(),
            "regime": str(regime or "").lower(),
            "session": _session_bucket(int(et_hour)),
        }
        row = {c: 0.0 for c in feature_names}
        for col in numeric:
            v = bar_features.get(col, 0.0)
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            if col in row:
                row[col] = fv
        for cat_col, known in cat_maps.items():
            val = cat_values.get(cat_col, "")
            for kv in known:
                name = f"{cat_col}__{kv}"
                if name in row and val == kv:
                    row[name] = 1
        if "et_hour" in row:
            row["et_hour"] = float(et_hour)
        X = np.array([[row[c] for c in feature_names]])
        return float(payload["model"].predict_proba(X)[0, 1])
    except Exception:
        logging.debug("gate scoring failed", exc_info=True)
        return None


def should_veto_signal(
    *,
    side: str,
    regime: str,
    et_hour: int,
    bar_features: Dict[str, float],
    strategy: str = "DynamicEngine3",
) -> Tuple[bool, str]:
    """Active-veto path. Picks the per-strategy model and applies its threshold.
    No-op (returns False) if there's no model for this strategy family."""
    family = _strategy_family(strategy)
    payload = _GATES.get(family)
    if payload is None:
        return False, ""
    p = _score_with_gate(payload, side=side, regime=regime, et_hour=et_hour,
                         bar_features=bar_features)
    if p is None:
        return False, ""
    thr = float(payload.get("veto_threshold", 0.35))
    if p >= thr:
        return True, (
            f"signal_gate_2025[{family}] P(big_loss)={p:.3f} >= {thr}"
        )
    return False, ""


def log_shadow_prediction(signal: dict, current_time_et=None) -> Optional[float]:
    """Shadow-mode telemetry: route to the per-strategy model and emit a log
    line. Returns the prediction (or None if not scoreable)."""
    if not _GATES:
        return None
    try:
        import loss_factor_guard as _lfg_mod
        import regime_classifier as _rc
        guard = _lfg_mod.get_guard()
        if guard is None or not getattr(guard, "_bar_cache", None):
            return None
        bars = list(guard._bar_cache)
        if len(bars) < 45:
            return None
        regime = _rc.current_regime()
        side = str(signal.get("side", "")).upper()
        entry_price = float(signal.get("entry_price") or signal.get("price") or 0.0)
        strategy = str(signal.get("strategy", "")).strip() or "DynamicEngine3"
        family = _strategy_family(strategy)

        last_ts = bars[-1][0]
        et_hour = 0
        try:
            from zoneinfo import ZoneInfo
            et_hour = int(last_ts.astimezone(ZoneInfo("America/New_York")).hour)
        except Exception:
            try:
                et_hour = int(last_ts.hour)
            except Exception:
                et_hour = 0

        feats = compute_bar_features_from_ohlcv(bars)
        if not feats:
            return None

        payload = _GATES.get(family)
        if payload is None:
            # No model for this family — log that we'd need one
            sub = signal.get("sub_strategy") or ""
            logging.info(
                "[SHADOW_GATE_2025] family=%s side=%s entry=%.2f size=%s regime=%s "
                "P(big_loss)=N/A would_veto=N/A reason=no_model_for_family sub=%s",
                family, side, entry_price, signal.get("size", "?"),
                regime, sub[:40],
            )
            return None

        p = _score_with_gate(
            payload, side=side, regime=regime, et_hour=et_hour, bar_features=feats,
        )
        if p is None:
            return None
        thr = float(payload.get("veto_threshold", 0.35))
        would_veto = p >= thr
        sub = signal.get("sub_strategy") or ""
        logging.info(
            "[SHADOW_GATE_2025] family=%s side=%s entry=%.2f size=%s regime=%s "
            "P(big_loss)=%.3f thresh=%.2f would_veto=%s sub=%s",
            family, side, entry_price, signal.get("size", "?"),
            regime, p, thr, would_veto, sub[:40],
        )
        return p
    except Exception:
        logging.debug("signal gate shadow log failed", exc_info=True)
        return None

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
import pandas as pd


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

_DYNAMIC_THRESHOLD_ENABLED = (
    os.environ.get("JULIE_GATE_DYNAMIC_THRESHOLD", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


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
    mkt_regime: str = "",
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
            "regime": str(regime or ""),
            "mkt_regime": str(mkt_regime or ""),
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
                if name in row and str(val).lower() == str(kv).lower():
                    row[name] = 1
        if "et_hour" in row:
            row["et_hour"] = float(et_hour)
        X = pd.DataFrame([[row[c] for c in feature_names]], columns=feature_names)
        return float(payload["model"].predict_proba(X)[0, 1])
    except Exception:
        logging.debug("gate scoring failed", exc_info=True)
        return None


# Regime-adaptive threshold multipliers. Validation on Oct 2025 showed the
# fixed-threshold gate was too lenient during whipsaw weeks (DE3 gate hurt
# -$218 on Oct 6-10) and sometimes too aggressive on calm trends. These
# multipliers are applied at runtime based on regime_classifier.current_regime()
# and tunable via env vars.
#   whipsaw    → 0.60× threshold (more aggressive veto — cuts more losers)
#   calm_trend → 1.15× threshold (slightly more lenient — lets winners run)
#   neutral    → 1.00× (unchanged)
#   warmup     → 1.00× (no regime info yet)
_REGIME_THR_MULT = {
    "whipsaw":    float(os.environ.get("JULIE_GATE_WHIPSAW_THR_MULT",    "0.60")),
    # calm_trend: was 1.15 in v5 initial, but that let too many losers through
    # on smooth-directional fade days (April 6-10 DE3: -$500 regression). 1.05
    # keeps a slight look-through for trend wins without blunting the gate.
    "calm_trend": float(os.environ.get("JULIE_GATE_CALMTREND_THR_MULT",  "1.05")),
    "neutral":    1.0,
    "warmup":     1.0,
}

# v5.2: session-adaptive threshold. Extends the regime multiplier with a
# drawdown-state multiplier. When the strategy has already made money today,
# be more lenient with the G gate (let winners run). When it's already in
# drawdown, be more aggressive (stop the bleeding).
# Based on backtest analysis of regression cases (Aug/Nov 2025):
#   cum_day_pnl > +$100 → mult 1.25 (more lenient — don't clip a winning day)
#   cum_day_pnl <= -$200 → mult 0.80 (more aggressive — catch the losers)
#   otherwise           → mult 1.00
_SESSION_LENIENT_PNL     = float(os.environ.get("JULIE_GATE_LENIENT_PNL", "100"))
_SESSION_AGGRESSIVE_PNL  = float(os.environ.get("JULIE_GATE_AGGRESSIVE_PNL", "-200"))
_SESSION_LENIENT_MULT    = float(os.environ.get("JULIE_GATE_LENIENT_MULT", "1.25"))
_SESSION_AGGRESSIVE_MULT = float(os.environ.get("JULIE_GATE_AGGRESSIVE_MULT", "0.80"))


def _session_multiplier(cum_day_pnl: float) -> float:
    if cum_day_pnl is None:
        return 1.0
    try:
        cp = float(cum_day_pnl)
    except Exception:
        return 1.0
    if cp >= _SESSION_LENIENT_PNL:
        return _SESSION_LENIENT_MULT
    if cp <= _SESSION_AGGRESSIVE_PNL:
        return _SESSION_AGGRESSIVE_MULT
    return 1.0


_EFFECTIVE_THR_FLOOR = float(os.environ.get("JULIE_GATE_EFF_THR_FLOOR", "0.25"))

# Per-(strategy × regime × time-bucket) threshold overrides produced by
# `scripts/idea1_filterg_per_cell_calibrate.py` from the seeded ledger.
# Loaded lazily; a missing file or missing cell key yields multiplier 1.0
# (no change from the existing regime × session behavior).
_PER_CELL_ACTIVE = os.environ.get("JULIE_FILTERG_PER_CELL_ACTIVE", "1").strip() not in ("0", "false", "False", "")
_PER_CELL_TABLE_PATH = Path(__file__).resolve().parent / "ai_loop_data" / "triathlon" / "filterg_threshold_overrides.json"
_PER_CELL_CACHE: Optional[Dict[str, float]] = None


def _load_per_cell_overrides() -> Dict[str, float]:
    """Lazy-load the per-cell multiplier table. Returns {} on any I/O
    or parse error (fail-closed)."""
    global _PER_CELL_CACHE
    if _PER_CELL_CACHE is not None:
        return _PER_CELL_CACHE
    if not _PER_CELL_ACTIVE:
        _PER_CELL_CACHE = {}
        return _PER_CELL_CACHE
    try:
        import json as _json
        if _PER_CELL_TABLE_PATH.exists():
            payload = _json.loads(_PER_CELL_TABLE_PATH.read_text())
            mults = payload.get("runtime_multipliers", {}) or {}
            # Keep only valid float values
            out: Dict[str, float] = {}
            for k, v in mults.items():
                try:
                    out[str(k)] = float(v)
                except (ValueError, TypeError):
                    continue
            _PER_CELL_CACHE = out
            logging.info(
                "[signal_gate_2025] per-cell threshold overrides loaded: "
                "%d cells from %s", len(out), _PER_CELL_TABLE_PATH,
            )
        else:
            _PER_CELL_CACHE = {}
    except Exception as exc:
        logging.warning("[signal_gate_2025] per-cell override load failed: %s", exc)
        _PER_CELL_CACHE = {}
    return _PER_CELL_CACHE


def _time_bucket_of_hour(et_hour: int) -> str:
    """Mirrors triathlon time-bucket logic so per-cell keys match the
    Triathlon Engine's cell coordinates."""
    h = float(et_hour)
    if h < 4.0:
        h += 24.0
    if 4.0 <= h < 9.5:   return "pre_open"
    if 9.5 <= h < 12.0:  return "morning"
    if 12.0 <= h < 14.0: return "lunch"
    if 14.0 <= h < 16.0: return "afternoon"
    if 16.0 <= h < 17.0: return "post_close"
    return "overnight"


def _per_cell_multiplier(strategy: str, regime: str, et_hour: int) -> float:
    """Look up the per-cell multiplier. Returns 1.0 when inactive, missing,
    or when the cell isn't in the override table."""
    if not _PER_CELL_ACTIVE:
        return 1.0
    table = _load_per_cell_overrides()
    if not table:
        return 1.0
    # Mirror the triathlon cell-key shape. Strategy is kept as-is (not
    # normalized to family) because the Triathlon table uses full strategy
    # labels like "DynamicEngine3" / "RegimeAdaptive".
    tb = _time_bucket_of_hour(et_hour)
    ck = f"{strategy}|{str(regime or '').lower()}|{tb}"
    return float(table.get(ck, 1.0))


def _effective_threshold(base_thr: float, regime: str = "",
                         cum_day_pnl: float = 0.0,
                         strategy: str = "", et_hour: int = 0) -> Tuple[float, float]:
    """Apply regime × session × per-cell multipliers to a gate's base threshold.
    Returns (effective, combined_multiplier).

    v5.5: apply a floor on the EFFECTIVE threshold. Whipsaw regime mult
    (0.60) on a DE3 base of 0.35 gives 0.21, which backtests showed
    over-vetoed borderline winners on high-ATR trend days. Floor at 0.25
    ensures G only fires when it's at least moderately confident
    (>= 25% P(big_loss)) regardless of regime.

    v6 (per-cell, 2026-04-23): after computing the regime × session
    multiplier, ALSO apply a per-(strategy × regime × time-bucket)
    multiplier loaded from the Idea-1 override table. Cells classified
    as "bleeding" in pre-April data receive a tighter threshold
    (mult < 1.0, more aggressive veto); cells classified as "strong"
    receive a more lenient threshold (mult > 1.0). Unrated / neutral
    cells get 1.0 and behave exactly as before. Disable with
    JULIE_FILTERG_PER_CELL_ACTIVE=0. The floor still applies.
    """
    if not _DYNAMIC_THRESHOLD_ENABLED:
        return float(base_thr), 1.0
    regime_mult = _REGIME_THR_MULT.get(str(regime or "").lower(), 1.0)
    session_mult = _session_multiplier(cum_day_pnl)
    per_cell_mult = _per_cell_multiplier(strategy, regime, et_hour) if strategy else 1.0
    mult = regime_mult * session_mult * per_cell_mult
    eff = float(base_thr) * mult
    if eff < _EFFECTIVE_THR_FLOOR:
        eff = _EFFECTIVE_THR_FLOOR
    return eff, mult


def should_veto_signal(
    *,
    side: str,
    regime: str,
    et_hour: int,
    bar_features: Dict[str, float],
    strategy: str = "DynamicEngine3",
    mkt_regime: str = "",
    cum_day_pnl: float = 0.0,
) -> Tuple[bool, str]:
    """Active-veto path. Picks the per-strategy model and applies its threshold.
    No-op (returns False) if there's no model for this strategy family.

    mkt_regime: the GLOBAL regime classifier label ("neutral" / "whipsaw" /
    "calm_trend") — used to adapt the veto threshold.
    cum_day_pnl: cumulative realized PnL for this strategy today before this
    trade. Used for session-adaptive thresholding (v5.2) — if the strategy is
    already up $100+ today, G stands down a bit; if it's down $200+, G becomes
    more aggressive.

    Callers that don't pass the new kwargs fall through at 1.0× (v4 behavior).
    """
    family = _strategy_family(strategy)
    payload = _GATES.get(family)
    if payload is None:
        return False, ""
    p = _score_with_gate(
        payload,
        side=side,
        regime=regime,
        mkt_regime=mkt_regime,
        et_hour=et_hour,
        bar_features=bar_features,
    )
    if p is None:
        return False, ""
    base_thr = float(payload.get("veto_threshold", 0.35))
    eff_thr, mult = _effective_threshold(
        base_thr, mkt_regime, cum_day_pnl,
        strategy=str(strategy or ""), et_hour=int(et_hour or 0),
    )
    if p >= eff_thr:
        mult_tag = f" x{mult:.2f}[{mkt_regime} dd=${cum_day_pnl:+.0f}]" if mult != 1.0 else ""
        return True, (
            f"signal_gate_2025[{family}] P(big_loss)={p:.3f} "
            f">= {eff_thr:.3f}{mult_tag}"
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

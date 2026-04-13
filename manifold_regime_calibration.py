from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


CALIBRATION_METRIC_COLUMNS = [
    "manifold_alignment",
    "manifold_smoothness",
    "manifold_stress",
    "manifold_dispersion",
    "manifold_R",
]

PERCENTILE_COLUMN_MAP = {
    "manifold_alignment": "manifold_alignment_pct",
    "manifold_smoothness": "manifold_smoothness_pct",
    "manifold_stress": "manifold_stress_pct",
    "manifold_dispersion": "manifold_dispersion_pct",
    "manifold_R": "manifold_R_pct",
}

REGIME_TO_ID = {
    "TREND_GEODESIC": 0,
    "CHOP_SPIRAL": 1,
    "DISPERSED": 2,
    "ROTATIONAL_TURBULENCE": 3,
}

ID_TO_REGIME = {v: k for k, v in REGIME_TO_ID.items()}

DEFAULT_PERCENTILE_GRID = [round(i / 100.0, 2) for i in range(101)]

DEFAULT_REGIME_POLICY: Dict[str, Any] = {
    "trend": {
        "alignment_pct_min": 0.80,
        "smoothness_pct_min": 0.55,
        "dispersion_pct_max": 0.35,
        "stress_pct_max": 0.80,
    },
    "rotation": {
        "stress_pct_min": 0.95,
        "dispersion_pct_min": 0.50,
    },
    "chop": {
        "stress_pct_min": 0.80,
        "smoothness_pct_max": 0.15,
    },
    "dispersed": {
        "dispersion_pct_min": 0.75,
        "alignment_pct_max": 0.55,
    },
    "side_bias": {
        "alignment_pct_min": 0.75,
        "abs_b_signed_min": 0.10,
    },
    "no_trade": {
        "stress_pct_hard": 0.985,
        "dispersion_pct_hard": 0.985,
        "alignment_pct_low": 0.20,
    },
    "risk_scale": {
        "TREND_GEODESIC": 1.10,
        "CHOP_SPIRAL": 0.92,
        "DISPERSED": 0.82,
        "ROTATIONAL_TURBULENCE": 0.60,
    },
}


def _ensure_float_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").astype(float)
    return pd.to_numeric(pd.Series(values), errors="coerce").astype(float)


def _compress_quantile_curve(values: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & np.isfinite(probs)
    values = values[finite]
    probs = probs[finite]
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=float), np.array([0.0, 1.0], dtype=float)
    order = np.argsort(values, kind="stable")
    values = values[order]
    probs = probs[order]
    uniq_values, inverse = np.unique(values, return_inverse=True)
    uniq_probs = np.zeros(len(uniq_values), dtype=float)
    for idx in range(len(uniq_values)):
        uniq_probs[idx] = float(np.max(probs[inverse == idx]))
    return uniq_values.astype(float), uniq_probs.astype(float)


def percentile_from_quantiles(values: Any, quantile_values: Any, quantile_grid: Any) -> np.ndarray:
    x = np.asarray(quantile_values, dtype=float)
    y = np.asarray(quantile_grid, dtype=float)
    xp, yp = _compress_quantile_curve(x, y)
    vals = _ensure_float_series(values).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    if xp.size <= 1:
        return np.full(vals.shape, yp[-1] if yp.size else 0.5, dtype=float)
    return np.interp(vals, xp, yp, left=float(yp[0]), right=float(yp[-1])).astype(float)


def build_calibration_payload(
    df: pd.DataFrame,
    source_path: Optional[str] = None,
    policy: Optional[Mapping[str, Any]] = None,
    percentile_grid: Optional[list[float]] = None,
) -> Dict[str, Any]:
    grid = list(percentile_grid or DEFAULT_PERCENTILE_GRID)
    payload: Dict[str, Any] = {
        "version": 1,
        "source_path": source_path,
        "metric_quantiles": {},
        "policy": json.loads(json.dumps(policy or DEFAULT_REGIME_POLICY)),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    for metric in CALIBRATION_METRIC_COLUMNS:
        series = _ensure_float_series(df.get(metric)).dropna()
        if series.empty:
            quantiles = [0.0 for _ in grid]
        else:
            quantiles = [float(series.quantile(q)) for q in grid]
        payload["metric_quantiles"][metric] = {
            "grid": list(grid),
            "values": quantiles,
        }
    calibrated = apply_calibration_frame(df, payload, preserve_existing_side_bias=True)
    counts = (
        pd.to_numeric(calibrated.get("manifold_regime_id"), errors="coerce")
        .round()
        .astype("Int64")
        .map(ID_TO_REGIME)
        .fillna("UNKNOWN")
        .value_counts(normalize=True)
    )
    payload["coverage_estimate"] = {str(k): float(v) for k, v in counts.items()}
    return payload


def load_calibration_payload(path: Any) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    cal_path = Path(path).expanduser()
    if not cal_path.is_absolute():
        cal_path = Path.cwd() / cal_path
    if not cal_path.exists():
        return None
    return json.loads(cal_path.read_text())


def save_calibration_payload(payload: Mapping[str, Any], path: Any) -> Path:
    out_path = Path(path).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _percentile_frame(df: pd.DataFrame, payload: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    metric_quantiles = payload.get("metric_quantiles", {}) or {}
    for metric, pct_col in PERCENTILE_COLUMN_MAP.items():
        spec = metric_quantiles.get(metric, {}) if isinstance(metric_quantiles, dict) else {}
        grid = spec.get("grid", DEFAULT_PERCENTILE_GRID)
        values = spec.get("values", [])
        out[pct_col] = percentile_from_quantiles(df.get(metric), values, grid)
    return out


def _assign_regimes(
    alignment_pct: np.ndarray,
    smoothness_pct: np.ndarray,
    stress_pct: np.ndarray,
    dispersion_pct: np.ndarray,
    policy: Mapping[str, Any],
) -> np.ndarray:
    trend_cfg = dict(policy.get("trend", {}) or {})
    rotation_cfg = dict(policy.get("rotation", {}) or {})
    chop_cfg = dict(policy.get("chop", {}) or {})
    dispersed_cfg = dict(policy.get("dispersed", {}) or {})

    trend_mask = (
        (alignment_pct >= float(trend_cfg.get("alignment_pct_min", 0.80)))
        & (smoothness_pct >= float(trend_cfg.get("smoothness_pct_min", 0.55)))
        & (dispersion_pct <= float(trend_cfg.get("dispersion_pct_max", 0.35)))
        & (stress_pct <= float(trend_cfg.get("stress_pct_max", 0.80)))
    )
    rotation_mask = (
        (stress_pct >= float(rotation_cfg.get("stress_pct_min", 0.95)))
        & (dispersion_pct >= float(rotation_cfg.get("dispersion_pct_min", 0.50)))
    )
    chop_mask = (
        (stress_pct >= float(chop_cfg.get("stress_pct_min", 0.80)))
        | (smoothness_pct <= float(chop_cfg.get("smoothness_pct_max", 0.15)))
    )
    dispersed_mask = (
        (dispersion_pct >= float(dispersed_cfg.get("dispersion_pct_min", 0.75)))
        & (alignment_pct <= float(dispersed_cfg.get("alignment_pct_max", 0.55)))
    )

    labels = np.full(len(alignment_pct), "DISPERSED", dtype=object)
    labels[rotation_mask] = "ROTATIONAL_TURBULENCE"
    labels[trend_mask & ~rotation_mask] = "TREND_GEODESIC"

    remaining = ~(rotation_mask | trend_mask)
    chop_score = (0.60 * stress_pct) + (0.40 * (1.0 - smoothness_pct))
    disp_score = (0.65 * dispersion_pct) + (0.35 * (1.0 - alignment_pct))

    both = remaining & chop_mask & dispersed_mask
    labels[remaining & chop_mask & ~dispersed_mask] = "CHOP_SPIRAL"
    labels[remaining & dispersed_mask & ~chop_mask] = "DISPERSED"
    labels[both] = np.where(chop_score[both] >= disp_score[both], "CHOP_SPIRAL", "DISPERSED")

    fallback = remaining & ~(chop_mask | dispersed_mask)
    labels[fallback] = np.where(chop_score[fallback] >= disp_score[fallback], "CHOP_SPIRAL", "DISPERSED")
    return labels


def _regime_allow_map(regime: str) -> Dict[str, bool]:
    if regime == "TREND_GEODESIC":
        return {"trend": True, "mean_reversion": False, "breakout": True, "fade": False}
    if regime == "CHOP_SPIRAL":
        return {"trend": False, "mean_reversion": True, "breakout": False, "fade": True}
    if regime == "DISPERSED":
        return {"trend": False, "mean_reversion": True, "breakout": False, "fade": True}
    return {"trend": False, "mean_reversion": False, "breakout": False, "fade": False}


def apply_calibration_frame(
    df: pd.DataFrame,
    payload: Mapping[str, Any],
    preserve_existing_side_bias: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    pct_frame = _percentile_frame(out, payload)
    for pct_col, values in pct_frame.items():
        out[pct_col] = values

    policy = dict(payload.get("policy", {}) or {})
    labels = _assign_regimes(
        alignment_pct=np.asarray(out["manifold_alignment_pct"], dtype=float),
        smoothness_pct=np.asarray(out["manifold_smoothness_pct"], dtype=float),
        stress_pct=np.asarray(out["manifold_stress_pct"], dtype=float),
        dispersion_pct=np.asarray(out["manifold_dispersion_pct"], dtype=float),
        policy=policy,
    )
    out["manifold_regime_id"] = [REGIME_TO_ID.get(str(label), -1) for label in labels]

    side_bias = np.zeros(len(out), dtype=float)
    if preserve_existing_side_bias and "manifold_side_bias" in out.columns:
        side_bias = _ensure_float_series(out["manifold_side_bias"]).fillna(0.0).to_numpy(dtype=float)
    out["manifold_side_bias"] = side_bias

    no_trade_cfg = dict(policy.get("no_trade", {}) or {})
    hard_no_trade = (
        (np.asarray(out["manifold_stress_pct"], dtype=float) >= float(no_trade_cfg.get("stress_pct_hard", 0.985)))
        & (np.asarray(out["manifold_dispersion_pct"], dtype=float) >= float(no_trade_cfg.get("dispersion_pct_hard", 0.985)))
        & (np.asarray(out["manifold_alignment_pct"], dtype=float) <= float(no_trade_cfg.get("alignment_pct_low", 0.20)))
    )
    no_trade = (labels == "ROTATIONAL_TURBULENCE") | hard_no_trade
    out["manifold_no_trade"] = np.where(no_trade, 1.0, 0.0)

    risk_scale_cfg = dict(policy.get("risk_scale", {}) or {})
    base_risk = _ensure_float_series(out.get("manifold_risk_mult", 1.0)).fillna(1.0).to_numpy(dtype=float)
    scale = np.array(
        [float(risk_scale_cfg.get(str(label), 1.0)) for label in labels],
        dtype=float,
    )
    out["manifold_risk_mult"] = np.clip(base_risk * scale, 0.25, 1.50)

    allow_trend = []
    allow_mean_reversion = []
    allow_breakout = []
    allow_fade = []
    for label in labels:
        allow = _regime_allow_map(str(label))
        allow_trend.append(1.0 if allow["trend"] else 0.0)
        allow_mean_reversion.append(1.0 if allow["mean_reversion"] else 0.0)
        allow_breakout.append(1.0 if allow["breakout"] else 0.0)
        allow_fade.append(1.0 if allow["fade"] else 0.0)
    out["manifold_allow_trend"] = allow_trend
    out["manifold_allow_mean_reversion"] = allow_mean_reversion
    out["manifold_allow_breakout"] = allow_breakout
    out["manifold_allow_fade"] = allow_fade
    return out


def apply_calibration_to_meta(
    meta: Mapping[str, Any],
    payload: Mapping[str, Any],
    existing_side_bias: float = 0.0,
) -> Dict[str, Any]:
    row = pd.DataFrame(
        [
            {
                "manifold_alignment": float(meta.get("alignment", 0.0) or 0.0),
                "manifold_smoothness": float(meta.get("smoothness", 0.0) or 0.0),
                "manifold_stress": float(meta.get("stress", 0.0) or 0.0),
                "manifold_dispersion": float(meta.get("dispersion", 0.0) or 0.0),
                "manifold_R": float(meta.get("R", 0.0) or 0.0),
                "manifold_risk_mult": float(meta.get("risk_mult", 1.0) or 1.0),
                "manifold_side_bias": float(existing_side_bias or 0.0),
            }
        ]
    )
    calibrated = apply_calibration_frame(row, payload, preserve_existing_side_bias=True).iloc[0]
    regime_id_raw = calibrated.get("manifold_regime_id", -1)
    if pd.isna(regime_id_raw):
        regime_id = -1
    else:
        regime_id = int(round(float(regime_id_raw)))
    regime = ID_TO_REGIME.get(regime_id, "DISPERSED")
    allow = _regime_allow_map(regime)
    out = dict(meta)
    out["regime"] = regime
    out["allow"] = allow
    out["no_trade"] = bool(float(calibrated.get("manifold_no_trade", 0.0) or 0.0) >= 0.5)
    out["risk_mult"] = float(calibrated.get("manifold_risk_mult", meta.get("risk_mult", 1.0)) or 1.0)
    out["alignment_pct"] = float(calibrated.get("manifold_alignment_pct", 0.0) or 0.0)
    out["smoothness_pct"] = float(calibrated.get("manifold_smoothness_pct", 0.0) or 0.0)
    out["stress_pct"] = float(calibrated.get("manifold_stress_pct", 0.0) or 0.0)
    out["dispersion_pct"] = float(calibrated.get("manifold_dispersion_pct", 0.0) or 0.0)
    out["R_pct"] = float(calibrated.get("manifold_R_pct", 0.0) or 0.0)
    side_cfg = dict((payload.get("policy", {}) or {}).get("side_bias", {}) or {})
    b_signed = float(meta.get("B_signed", 0.0) or 0.0)
    if abs(b_signed) >= float(side_cfg.get("abs_b_signed_min", 0.10)) and out["alignment_pct"] >= float(side_cfg.get("alignment_pct_min", 0.75)):
        out["side_bias"] = 1 if b_signed > 0.0 else -1
    else:
        out["side_bias"] = int(existing_side_bias or 0)
    return out

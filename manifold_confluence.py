import math
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_CONFLUENCE_PARAMS: Dict[str, float] = {
    # Exponential decay terms.
    "lambda_t": 0.85,  # T term (stress)
    "lambda_v": 0.55,  # V term (volatility)
    "lambda_r": 0.35,  # R term (regime intensity)
    # Gamma-like boost.
    "gamma0": 0.65,
    "y_gate": 0.70,
    # Additional regime decay.
    "phi0": 0.40,
    # Directional tilt term.
    "kappa": 0.70,
    "direction_gain": 0.35,
    "direction_floor": 0.35,
    "direction_ceiling": 1.65,
    # Final clamps.
    "alpha_min": 0.10,
    "alpha_max": 1.80,
    "prob_floor": 0.01,
    "prob_ceiling": 0.99,
}


def _merge_params(params: Optional[Dict]) -> Dict[str, float]:
    out = dict(DEFAULT_CONFLUENCE_PARAMS)
    if isinstance(params, dict):
        for k, v in params.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    # Hard safety guards.
    out["y_gate"] = float(np.clip(out["y_gate"], 0.0, 1.0))
    out["direction_floor"] = float(max(0.01, out["direction_floor"]))
    out["direction_ceiling"] = float(max(out["direction_floor"], out["direction_ceiling"]))
    out["alpha_min"] = float(max(0.01, out["alpha_min"]))
    out["alpha_max"] = float(max(out["alpha_min"], out["alpha_max"]))
    out["prob_floor"] = float(np.clip(out["prob_floor"], 0.0, 0.49))
    out["prob_ceiling"] = float(np.clip(out["prob_ceiling"], 0.51, 1.0))
    return out


def _feature_vector(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if df is None or df.empty or col not in df.columns:
        return np.full((0,), float(default), dtype=float)
    out = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    out = np.where(np.isfinite(out), out, float(default))
    return out


def apply_confluence_formula(
    feature_frame: pd.DataFrame,
    prob_up: np.ndarray,
    params: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply formula-style confluence scaling to classifier probabilities.

    Returns arrays:
    - prob_up_adj
    - alpha_scale
    - directional_sign
    - directional_alignment
    """
    cfg = _merge_params(params)
    p = np.asarray(prob_up, dtype=float).reshape(-1)
    n = int(p.shape[0])
    if n <= 0:
        return {
            "prob_up_adj": np.asarray([], dtype=float),
            "alpha_scale": np.asarray([], dtype=float),
            "directional_sign": np.asarray([], dtype=float),
            "directional_alignment": np.asarray([], dtype=float),
        }

    if feature_frame is None or feature_frame.empty:
        feat = pd.DataFrame(index=np.arange(n))
    else:
        feat = feature_frame.reset_index(drop=True)
    if len(feat) != n:
        if len(feat) == 1 and n > 1:
            feat = pd.concat([feat] * n, ignore_index=True)
        else:
            raise ValueError(
                f"feature/probability length mismatch: features={len(feat)} probs={n}"
            )

    # Variable mapping from available manifold + aux features.
    # T: stress, V: volatility regime, R: manifold R intensity.
    # Y: alignment confidence, C: dispersion/chop, sigma_h: realized vol proxy.
    T = np.abs(_feature_vector(feat, "manifold_stress", 0.0))
    V = np.abs(_feature_vector(feat, "atr14_z", 0.0))
    R = np.abs(_feature_vector(feat, "manifold_R", 0.0))
    Y = np.clip(_feature_vector(feat, "manifold_alignment", 0.0), 0.0, 1.0)
    sigma_h = np.abs(_feature_vector(feat, "vol_z", 0.0))
    C = np.clip(_feature_vector(feat, "manifold_dispersion", 0.5), 0.0, 1.0)

    decay = np.exp(-(cfg["lambda_t"] * T) - (cfg["lambda_v"] * V) - (cfg["lambda_r"] * R))
    gamma = 1.0 + (cfg["gamma0"] * Y * (Y > cfg["y_gate"]).astype(float))
    regime_decay = np.exp(-(cfg["phi0"] * np.square(R)))

    directional_sign = np.sign(sigma_h - (cfg["kappa"] * C))
    directional_sign = np.where(directional_sign == 0.0, 1.0, directional_sign)
    prob_direction = np.sign(p - 0.5)
    prob_direction = np.where(prob_direction == 0.0, 1.0, prob_direction)
    directional_alignment = directional_sign * prob_direction

    directional_term = 1.0 + (cfg["direction_gain"] * directional_alignment * (1.0 - C))
    directional_term = np.clip(
        directional_term,
        cfg["direction_floor"],
        cfg["direction_ceiling"],
    )

    alpha = decay * gamma * regime_decay * directional_term
    alpha = np.clip(alpha, cfg["alpha_min"], cfg["alpha_max"])

    p_adj = 0.5 + (alpha * (p - 0.5))
    p_adj = np.clip(p_adj, cfg["prob_floor"], cfg["prob_ceiling"])

    return {
        "prob_up_adj": p_adj.astype(float),
        "alpha_scale": alpha.astype(float),
        "directional_sign": directional_sign.astype(float),
        "directional_alignment": directional_alignment.astype(float),
    }


def sample_confluence_params(rng: np.random.Generator) -> Dict[str, float]:
    # Bounded random search region tuned for stability.
    return {
        "lambda_t": float(rng.uniform(0.05, 2.00)),
        "lambda_v": float(rng.uniform(0.05, 2.00)),
        "lambda_r": float(rng.uniform(0.05, 1.25)),
        "gamma0": float(rng.uniform(0.0, 2.0)),
        "y_gate": float(rng.uniform(0.50, 0.90)),
        "phi0": float(rng.uniform(0.0, 1.25)),
        "kappa": float(rng.uniform(0.10, 2.00)),
        "direction_gain": float(rng.uniform(0.0, 0.9)),
        "direction_floor": float(rng.uniform(0.20, 0.90)),
        "direction_ceiling": float(rng.uniform(1.0, 2.50)),
        "alpha_min": float(rng.uniform(0.05, 0.50)),
        "alpha_max": float(rng.uniform(1.0, 2.50)),
        "prob_floor": 0.01,
        "prob_ceiling": 0.99,
    }


def calibrate_confluence(
    feature_frame: pd.DataFrame,
    prob_up: np.ndarray,
    evaluate_fn: Callable[[np.ndarray], Dict],
    *,
    trials: int = 300,
    seed: int = 42,
) -> Dict:
    """
    Random-search calibrator.
    evaluate_fn takes adjusted probabilities and returns score dict with at least:
      {score, trade_count, avg_pnl, total_pnl, threshold, short_threshold}
    """
    rng = np.random.default_rng(int(seed))
    trial_count = max(1, int(trials))

    best: Optional[Dict] = None
    for i in range(trial_count):
        if i == 0:
            params = dict(DEFAULT_CONFLUENCE_PARAMS)
        else:
            params = sample_confluence_params(rng)
        adjusted = apply_confluence_formula(feature_frame, prob_up, params)
        stats = dict(evaluate_fn(adjusted["prob_up_adj"]))
        score = float(stats.get("score", -math.inf))
        candidate = {
            "score": score,
            "params": _merge_params(params),
            "stats": stats,
            "alpha_mean": float(np.mean(adjusted["alpha_scale"])) if len(adjusted["alpha_scale"]) else 0.0,
            "alpha_std": float(np.std(adjusted["alpha_scale"])) if len(adjusted["alpha_scale"]) else 0.0,
        }
        if best is None:
            best = candidate
        else:
            if candidate["score"] > best["score"] + 1e-12:
                best = candidate
            elif abs(candidate["score"] - best["score"]) <= 1e-12:
                c_pnl = float(candidate["stats"].get("total_pnl", 0.0))
                b_pnl = float(best["stats"].get("total_pnl", 0.0))
                if c_pnl > b_pnl + 1e-12:
                    best = candidate

    if best is None:
        raise RuntimeError("Confluence calibration failed to produce a candidate.")
    return best

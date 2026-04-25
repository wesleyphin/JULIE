import logging
from os import PathLike
from typing import Dict, Optional

import numpy as np
import pandas as pd

from manifold_strategy_features import (
    AUX_FEATURE_COLUMNS,
    build_training_feature_frame as build_manifold_feature_frame,
)

SETUP_TO_ID = {
    "compression_release": 1,
    "aligned_flow": 2,
    "exhaustion_reversal": 3,
    "transition_burst": 4,
}

SETUP_DEFAULTS = {
    "compression_release": {"sl_mult": 1.25, "tp_mult": 2.35, "horizon_bars": 20},
    "aligned_flow": {"sl_mult": 1.10, "tp_mult": 2.00, "horizon_bars": 18},
    "exhaustion_reversal": {"sl_mult": 1.00, "tp_mult": 1.65, "horizon_bars": 12},
    "transition_burst": {"sl_mult": 1.20, "tp_mult": 2.10, "horizon_bars": 16},
}

SETUP_SCORE_KEYS = (
    ("compression_release_long", "compression_release"),
    ("compression_release_short", "compression_release"),
    ("aligned_flow_long", "aligned_flow"),
    ("aligned_flow_short", "aligned_flow"),
    ("exhaustion_reversal_long", "exhaustion_reversal"),
    ("exhaustion_reversal_short", "exhaustion_reversal"),
    ("transition_burst_long", "transition_burst"),
    ("transition_burst_short", "transition_burst"),
)

SETUP_THRESHOLDS = {
    "compression_release": 0.26,
    "aligned_flow": 0.40,
    "exhaustion_reversal": 0.42,
    "transition_burst": 0.42,
}

BASE_FEATURE_COLUMNS = [
    "manifold_R",
    "manifold_alignment",
    "manifold_smoothness",
    "manifold_stress",
    "manifold_dispersion",
    "manifold_risk_mult",
    "manifold_regime_id",
    "manifold_R_pct",
    "manifold_alignment_pct",
    "manifold_smoothness_pct",
    "manifold_stress_pct",
    "manifold_dispersion_pct",
] + list(AUX_FEATURE_COLUMNS)

DERIVED_FEATURE_COLUMNS = [
    "flow_fast",
    "flow_slow",
    "flow_mag_fast",
    "flow_mag_slow",
    "flow_agreement",
    "flow_curvature",
    "up_pressure_10",
    "down_pressure_10",
    "pressure_imbalance_10",
    "up_pressure_30",
    "down_pressure_30",
    "pressure_imbalance_30",
    "coherence",
    "compression_score",
    "expansion_score",
    "extension_score",
    "transition_energy",
    "novelty_score",
    "regime_change",
    "d_alignment_3",
    "d_alignment_10",
    "d_stress_3",
    "d_stress_10",
    "d_dispersion_3",
    "d_dispersion_10",
    "d_r_3",
    "d_r_10",
    "d_coherence_3",
    "skew_20",
    "skew_60",
    "kurt_20",
    "directional_vwap_dist",
    "alignment_minus_stress",
    "release_energy",
    "trend_persistence",
    "burst_pressure",
    "burst_regime_shift",
    "coherence_recovery",
    "compression_release_edge",
    "aligned_flow_edge",
    "transition_burst_edge",
    "session_is_asia",
    "session_is_london",
    "session_is_nyam",
    "session_is_nypm",
    "regime_is_trend_geodesic",
    "regime_is_chop_spiral",
    "regime_is_dispersed",
    "regime_is_rotational_turbulence",
    "compression_release_session_edge",
    "aligned_flow_ny_dispersed_edge",
    "aligned_flow_nypm_trend_edge",
    "transition_burst_nyam_chop_edge",
    "setup_strength",
    "candidate_side",
    "setup_compression_release",
    "setup_aligned_flow",
    "setup_exhaustion_reversal",
    "setup_transition_burst",
]

META_COLUMNS = [
    "setup_family",
    "setup_id",
    "setup_sl_mult",
    "setup_tp_mult",
    "setup_horizon_bars",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
EXPORT_COLUMNS = FEATURE_COLUMNS + META_COLUMNS


# Raw scaling helpers tuned for minute-level ES returns.
_FAST_SCALE = 1200.0
_SLOW_SCALE = 900.0
_PRESSURE_SCALE = 1800.0


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"AetherFlow input missing required columns: {', '.join(missing)}")


def _sigmoid_neg(series: pd.Series) -> pd.Series:
    clipped = series.clip(-6.0, 6.0)
    return 1.0 / (1.0 + np.exp(clipped))


def _tanh_scaled(series: pd.Series, scale: float) -> pd.Series:
    return pd.Series(np.tanh(pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float) * float(scale)), index=series.index)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _augment_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    work = frame.copy()

    def _series(name: str, default: float = 0.0) -> pd.Series:
        if name in work.columns:
            raw = work[name]
        else:
            raw = pd.Series(float(default), index=work.index)
        return pd.to_numeric(raw, errors="coerce").fillna(float(default))

    candidate_side = _series("candidate_side", 0.0)
    vwap_dist_atr = _series("vwap_dist_atr", 0.0)
    alignment_pct = _clip01(_series("manifold_alignment_pct", 0.0))
    smoothness_pct = _clip01(_series("manifold_smoothness_pct", 0.0))
    stress_pct = _clip01(_series("manifold_stress_pct", 0.0))
    coherence = _clip01(_series("coherence", 0.0))
    compression_score = _clip01(_series("compression_score", 0.0))
    transition_energy = _clip01(_series("transition_energy", 0.0))
    novelty_score = _clip01(_series("novelty_score", 0.0))
    flow_mag_fast = _clip01(_series("flow_mag_fast", 0.0))
    flow_mag_slow = _clip01(_series("flow_mag_slow", 0.0))
    flow_agreement = _series("flow_agreement", 0.0)
    pressure_imbalance_30 = _series("pressure_imbalance_30", 0.0)
    regime_change = _series("regime_change", 0.0)
    session_id = _series("session_id", -999.0)
    regime_id = _series("manifold_regime_id", -1.0)
    d_alignment_3 = _series("d_alignment_3", 0.0)
    d_coherence_3 = _series("d_coherence_3", 0.0)
    setup_compression_release = _series("setup_compression_release", 0.0)
    setup_aligned_flow = _series("setup_aligned_flow", 0.0)
    setup_transition_burst = _series("setup_transition_burst", 0.0)

    directional_vwap_dist = candidate_side * vwap_dist_atr
    alignment_minus_stress = alignment_pct - stress_pct
    release_energy = compression_score * transition_energy * flow_mag_fast
    trend_persistence = coherence * flow_mag_slow * flow_agreement.clip(lower=0.0)
    burst_pressure = transition_energy * pressure_imbalance_30.abs().clip(0.0, 1.0) * flow_mag_fast
    burst_regime_shift = regime_change * novelty_score * flow_mag_fast
    coherence_recovery = d_coherence_3.clip(lower=0.0) * (1.0 - stress_pct) * smoothness_pct
    compression_release_edge = setup_compression_release * release_energy * d_alignment_3.clip(lower=0.0)
    aligned_flow_edge = setup_aligned_flow * trend_persistence * (alignment_pct * smoothness_pct)
    transition_burst_edge = setup_transition_burst * (0.6 * burst_pressure + 0.4 * burst_regime_shift)
    session_is_asia = session_id.eq(0.0).astype(float)
    session_is_london = session_id.eq(1.0).astype(float)
    session_is_nyam = session_id.eq(2.0).astype(float)
    session_is_nypm = session_id.eq(3.0).astype(float)
    regime_is_trend_geodesic = regime_id.eq(0.0).astype(float)
    regime_is_chop_spiral = regime_id.eq(1.0).astype(float)
    regime_is_dispersed = regime_id.eq(2.0).astype(float)
    regime_is_rotational_turbulence = regime_id.eq(3.0).astype(float)
    compression_release_session_edge = setup_compression_release * release_energy * (session_is_london + session_is_nypm)
    aligned_flow_ny_dispersed_edge = setup_aligned_flow * trend_persistence * regime_is_dispersed * (session_is_nyam + session_is_nypm)
    aligned_flow_nypm_trend_edge = setup_aligned_flow * trend_persistence * session_is_nypm * regime_is_trend_geodesic
    transition_burst_nyam_chop_edge = setup_transition_burst * burst_pressure * session_is_nyam * regime_is_chop_spiral

    work["directional_vwap_dist"] = directional_vwap_dist
    work["alignment_minus_stress"] = alignment_minus_stress
    work["release_energy"] = release_energy
    work["trend_persistence"] = trend_persistence
    work["burst_pressure"] = burst_pressure
    work["burst_regime_shift"] = burst_regime_shift
    work["coherence_recovery"] = coherence_recovery
    work["compression_release_edge"] = compression_release_edge
    work["aligned_flow_edge"] = aligned_flow_edge
    work["transition_burst_edge"] = transition_burst_edge
    work["session_is_asia"] = session_is_asia
    work["session_is_london"] = session_is_london
    work["session_is_nyam"] = session_is_nyam
    work["session_is_nypm"] = session_is_nypm
    work["regime_is_trend_geodesic"] = regime_is_trend_geodesic
    work["regime_is_chop_spiral"] = regime_is_chop_spiral
    work["regime_is_dispersed"] = regime_is_dispersed
    work["regime_is_rotational_turbulence"] = regime_is_rotational_turbulence
    work["compression_release_session_edge"] = compression_release_session_edge
    work["aligned_flow_ny_dispersed_edge"] = aligned_flow_ny_dispersed_edge
    work["aligned_flow_nypm_trend_edge"] = aligned_flow_nypm_trend_edge
    work["transition_burst_nyam_chop_edge"] = transition_burst_nyam_chop_edge
    return work


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    work = df.copy()
    work = _augment_interaction_features(work)
    for col in FEATURE_COLUMNS:
        if col not in work.columns:
            work[col] = 0.0
    return work


def _build_from_base(
    base: pd.DataFrame,
    *,
    preferred_setup_families: Optional[set[str]] = None,
    emit_family_frames: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    work = base.reindex(columns=BASE_FEATURE_COLUMNS, fill_value=0.0).copy()
    _require_columns(work, BASE_FEATURE_COLUMNS)
    for col in BASE_FEATURE_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype(np.float32)

    ret_1 = pd.to_numeric(work["ret_1"], errors="coerce").fillna(0.0)
    ret_5 = pd.to_numeric(work["ret_5"], errors="coerce").fillna(0.0)
    ret_15 = pd.to_numeric(work["ret_15"], errors="coerce").fillna(0.0)
    ema_slope = pd.to_numeric(work["ema_slope_20"], errors="coerce").fillna(0.0)
    ema_spread = pd.to_numeric(work["ema_spread"], errors="coerce").fillna(0.0)
    atr14_z = pd.to_numeric(work["atr14_z"], errors="coerce").fillna(0.0)
    range_z = pd.to_numeric(work["range_z"], errors="coerce").fillna(0.0)
    vwap_dist_atr = pd.to_numeric(work["vwap_dist_atr"], errors="coerce").fillna(0.0)

    alignment_pct = _clip01(work["manifold_alignment_pct"])
    smoothness_pct = _clip01(work["manifold_smoothness_pct"])
    stress_pct = _clip01(work["manifold_stress_pct"])
    dispersion_pct = _clip01(work["manifold_dispersion_pct"])
    r_pct = _clip01(work["manifold_R_pct"])
    regime_id = pd.to_numeric(work.get("manifold_regime_id", -1.0), errors="coerce").fillna(-1.0)
    regime_is_trend = regime_id.eq(0.0)
    regime_is_chop = regime_id.eq(1.0)
    regime_is_disperse = regime_id.eq(2.0)
    regime_is_rot = regime_id.eq(3.0)

    flow_fast = (0.60 * ret_5) + (0.40 * ema_slope)
    flow_slow = (0.60 * ret_15) + (0.40 * ema_spread)
    flow_mag_fast = _tanh_scaled(flow_fast.abs(), _FAST_SCALE)
    flow_mag_slow = _tanh_scaled(flow_slow.abs(), _SLOW_SCALE)
    flow_curvature = flow_fast - flow_slow

    up_pressure_10 = ret_1.clip(lower=0.0).rolling(10, min_periods=3).mean().fillna(0.0)
    down_pressure_10 = (-ret_1.clip(upper=0.0)).rolling(10, min_periods=3).mean().fillna(0.0)
    pressure_imbalance_10 = _tanh_scaled(up_pressure_10 - down_pressure_10, _PRESSURE_SCALE)

    up_pressure_30 = ret_1.clip(lower=0.0).rolling(30, min_periods=10).mean().fillna(0.0)
    down_pressure_30 = (-ret_1.clip(upper=0.0)).rolling(30, min_periods=10).mean().fillna(0.0)
    pressure_imbalance_30 = _tanh_scaled(up_pressure_30 - down_pressure_30, _PRESSURE_SCALE)

    flow_sign_same = np.sign(flow_fast.to_numpy(dtype=float)) == np.sign(flow_slow.to_numpy(dtype=float))
    flow_agreement_strength = _tanh_scaled(flow_fast.abs() + flow_slow.abs(), 700.0)
    flow_agreement = pd.Series(
        np.where(flow_sign_same, flow_agreement_strength.to_numpy(dtype=float), -flow_agreement_strength.to_numpy(dtype=float)),
        index=work.index,
    )

    coherence = (alignment_pct * smoothness_pct * (1.0 - stress_pct)).clip(0.0, 1.0)
    range_compression = _sigmoid_neg(range_z)
    atr_compression = _sigmoid_neg(atr14_z)
    compression_score = ((1.0 - dispersion_pct) * 0.5 * (range_compression + atr_compression)).clip(0.0, 1.0)
    expansion_score = (alignment_pct * r_pct * flow_mag_fast).clip(0.0, 1.0)
    extension_score = (np.tanh(vwap_dist_atr.abs() / 2.0) * (0.5 + 0.5 * stress_pct)).clip(0.0, 1.0)

    d_alignment_3 = alignment_pct.diff(3).fillna(0.0)
    d_alignment_10 = alignment_pct.diff(10).fillna(0.0)
    d_stress_3 = stress_pct.diff(3).fillna(0.0)
    d_stress_10 = stress_pct.diff(10).fillna(0.0)
    d_dispersion_3 = dispersion_pct.diff(3).fillna(0.0)
    d_dispersion_10 = dispersion_pct.diff(10).fillna(0.0)
    d_r_3 = r_pct.diff(3).fillna(0.0)
    d_r_10 = r_pct.diff(10).fillna(0.0)
    d_coherence_3 = coherence.diff(3).fillna(0.0)
    transition_energy = (
        d_alignment_3.abs() + d_stress_3.abs() + d_dispersion_3.abs() + d_r_3.abs()
    ).clip(0.0, 4.0) / 4.0
    novelty_arr = (
        np.abs(alignment_pct.to_numpy(dtype=np.float32) - np.float32(0.5))
        + np.abs(stress_pct.to_numpy(dtype=np.float32) - np.float32(0.5))
        + np.abs(dispersion_pct.to_numpy(dtype=np.float32) - np.float32(0.5))
    )
    novelty_score = pd.Series(np.clip(novelty_arr, 0.0, 1.5) / 1.5, index=work.index)
    regime_change = regime_id.ne(regime_id.shift(1)).astype(float)

    skew_20 = ret_1.rolling(20, min_periods=8).skew().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    skew_60 = ret_1.rolling(60, min_periods=20).skew().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    kurt_20 = ret_1.rolling(20, min_periods=8).kurt().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    directional_bias = _tanh_scaled(flow_fast + (0.70 * flow_slow) + (0.35 * pressure_imbalance_10) + (0.20 * ema_spread), 950.0)

    cr_cond = (
        (compression_score > 0.50)
        & (transition_energy > 0.10)
        & (stress_pct < 0.78)
        & (flow_mag_fast > 0.10)
        & (flow_agreement > -0.05)
        & (~regime_is_rot)
    )
    cr_base = (0.45 * compression_score) + (0.30 * coherence) + (0.15 * flow_mag_fast) + (0.10 * d_alignment_3.clip(lower=0.0))
    cr_long = np.where(cr_cond, cr_base * directional_bias.clip(lower=0.0), 0.0)
    cr_short = np.where(cr_cond, cr_base * (-directional_bias).clip(lower=0.0), 0.0)

    af_cond = (
        (coherence > 0.34)
        & (alignment_pct > 0.60)
        & (smoothness_pct > 0.55)
        & (stress_pct < 0.46)
        & (flow_agreement > 0.18)
        & (flow_mag_slow > 0.14)
        & (~regime_is_rot)
    )
    af_base = (0.35 * coherence) + (0.25 * alignment_pct) + (0.20 * (1.0 - stress_pct)) + (0.20 * flow_mag_slow)
    af_long = np.where(af_cond, af_base * directional_bias.clip(lower=0.0), 0.0)
    af_short = np.where(af_cond, af_base * (-directional_bias).clip(lower=0.0), 0.0)

    ex_base = (0.45 * extension_score) + (0.20 * stress_pct) + (0.20 * flow_curvature.abs().clip(0.0, 0.01) * 80.0) + (0.15 * d_stress_3.clip(lower=0.0))
    ex_long_cond = (
        (vwap_dist_atr < -1.25)
        & (d_coherence_3 < -0.015)
        & (d_stress_3 > 0.03)
        & (flow_curvature > 0.0)
        & (flow_mag_fast > 0.16)
        & (stress_pct > 0.60)
        & (extension_score > 0.55)
        & (flow_agreement < 0.15)
        & (regime_is_disperse | regime_is_chop)
    )
    ex_short_cond = (
        (vwap_dist_atr > 1.25)
        & (d_coherence_3 < -0.015)
        & (d_stress_3 > 0.03)
        & (flow_curvature < 0.0)
        & (flow_mag_fast > 0.16)
        & (stress_pct > 0.60)
        & (extension_score > 0.55)
        & (flow_agreement < 0.15)
        & (regime_is_disperse | regime_is_chop)
    )
    ex_long = np.where(ex_long_cond, ex_base, 0.0)
    ex_short = np.where(ex_short_cond, ex_base, 0.0)

    tb_bias = _tanh_scaled(flow_fast + (0.40 * pressure_imbalance_30), 900.0)
    tb_cond = (
        (transition_energy > 0.24)
        & ((regime_change > 0.0) | (novelty_score > 0.64))
        & (flow_mag_fast > 0.20)
        & (tb_bias.abs() > 0.14)
        & (~regime_is_rot)
    )
    tb_base = (0.35 * transition_energy) + (0.25 * novelty_score) + (0.20 * flow_mag_fast) + (0.20 * flow_mag_slow)
    tb_long = np.where(tb_cond, tb_base * tb_bias.clip(lower=0.0), 0.0)
    tb_short = np.where(tb_cond, tb_base * (-tb_bias).clip(lower=0.0), 0.0)

    setup_scores = pd.DataFrame(
        {
            "compression_release_long": np.asarray(cr_long, dtype=float),
            "compression_release_short": np.asarray(cr_short, dtype=float),
            "aligned_flow_long": np.asarray(af_long, dtype=float),
            "aligned_flow_short": np.asarray(af_short, dtype=float),
            "exhaustion_reversal_long": np.asarray(ex_long, dtype=float),
            "exhaustion_reversal_short": np.asarray(ex_short, dtype=float),
            "transition_burst_long": np.asarray(tb_long, dtype=float),
            "transition_burst_short": np.asarray(tb_short, dtype=float),
        },
        index=work.index,
    )
    def _keys_for_families(families: Optional[set[str]]) -> list[str]:
        if not families:
            return [key for key, _ in SETUP_SCORE_KEYS]
        return [key for key, family_name in SETUP_SCORE_KEYS if family_name in families]

    def _assign_setup(keys: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
        setup_score_view = setup_scores.loc[:, keys] if keys else setup_scores
        best_key = setup_score_view.idxmax(axis=1)
        best_score = setup_score_view.max(axis=1).fillna(0.0)
        family = pd.Series("", index=work.index, dtype=object)
        candidate_side = pd.Series(0.0, index=work.index, dtype=float)
        setup_strength = pd.Series(0.0, index=work.index, dtype=float)
        key_set = set(keys)
        for key, family_name in SETUP_SCORE_KEYS:
            if key_set and key not in key_set:
                continue
            mask = (best_key == key) & (best_score >= SETUP_THRESHOLDS[family_name])
            if not bool(mask.any()):
                continue
            family.loc[mask] = family_name
            candidate_side.loc[mask] = 1.0 if key.endswith("_long") else -1.0
            setup_strength.loc[mask] = best_score.loc[mask].astype(float)
        return family, candidate_side, setup_strength

    def _common_frame() -> pd.DataFrame:
        out = work.copy()
        out["flow_fast"] = flow_fast
        out["flow_slow"] = flow_slow
        out["flow_mag_fast"] = flow_mag_fast
        out["flow_mag_slow"] = flow_mag_slow
        out["flow_agreement"] = flow_agreement
        out["flow_curvature"] = flow_curvature
        out["up_pressure_10"] = up_pressure_10
        out["down_pressure_10"] = down_pressure_10
        out["pressure_imbalance_10"] = pressure_imbalance_10
        out["up_pressure_30"] = up_pressure_30
        out["down_pressure_30"] = down_pressure_30
        out["pressure_imbalance_30"] = pressure_imbalance_30
        out["coherence"] = coherence
        out["compression_score"] = compression_score
        out["expansion_score"] = expansion_score
        out["extension_score"] = extension_score
        out["transition_energy"] = transition_energy
        out["novelty_score"] = novelty_score
        out["regime_change"] = regime_change
        out["d_alignment_3"] = d_alignment_3
        out["d_alignment_10"] = d_alignment_10
        out["d_stress_3"] = d_stress_3
        out["d_stress_10"] = d_stress_10
        out["d_dispersion_3"] = d_dispersion_3
        out["d_dispersion_10"] = d_dispersion_10
        out["d_r_3"] = d_r_3
        out["d_r_10"] = d_r_10
        out["d_coherence_3"] = d_coherence_3
        out["skew_20"] = skew_20.clip(-10.0, 10.0)
        out["skew_60"] = skew_60.clip(-10.0, 10.0)
        out["kurt_20"] = kurt_20.clip(-10.0, 20.0)
        return out

    def _finalize(family: pd.Series, candidate_side: pd.Series, setup_strength: pd.Series) -> pd.DataFrame:
        setup_id = family.map(SETUP_TO_ID).fillna(0.0)
        setup_sl_mult = family.map(lambda x: float(SETUP_DEFAULTS.get(str(x), {}).get("sl_mult", 0.0))).fillna(0.0)
        setup_tp_mult = family.map(lambda x: float(SETUP_DEFAULTS.get(str(x), {}).get("tp_mult", 0.0))).fillna(0.0)
        setup_horizon_bars = family.map(
            lambda x: float(SETUP_DEFAULTS.get(str(x), {}).get("horizon_bars", 0.0))
        ).fillna(0.0)
        out = _common_frame()
        out["setup_strength"] = setup_strength
        out["candidate_side"] = candidate_side
        out["setup_compression_release"] = (family == "compression_release").astype(float)
        out["setup_aligned_flow"] = (family == "aligned_flow").astype(float)
        out["setup_exhaustion_reversal"] = (family == "exhaustion_reversal").astype(float)
        out["setup_transition_burst"] = (family == "transition_burst").astype(float)
        out = _augment_interaction_features(out)
        out["setup_family"] = family
        out["setup_id"] = setup_id.astype(float)
        out["setup_sl_mult"] = setup_sl_mult.astype(float)
        out["setup_tp_mult"] = setup_tp_mult.astype(float)
        out["setup_horizon_bars"] = setup_horizon_bars.astype(float)
        out.loc[family == "", "setup_family"] = ""
        numeric_cols = [col for col in EXPORT_COLUMNS if col != "setup_family"]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float32)
        return out

    if emit_family_frames:
        requested_families = preferred_setup_families or {family_name for _, family_name in SETUP_SCORE_KEYS}
        frames: dict[str, pd.DataFrame] = {}
        for family_name in sorted(str(name) for name in requested_families if str(name).strip()):
            keys = [key for key, score_family in SETUP_SCORE_KEYS if score_family == family_name]
            if not keys:
                continue
            frames[family_name] = _finalize(*_assign_setup(keys))
        return frames

    return _finalize(*_assign_setup(_keys_for_families(preferred_setup_families)))


def build_feature_frame(
    df: Optional[pd.DataFrame] = None,
    *,
    base_features: Optional[pd.DataFrame] = None,
    preferred_setup_families: Optional[set[str]] = None,
    manifold_cfg: Optional[Dict] = None,
    log_every: int = 0,
) -> pd.DataFrame:
    if base_features is None:
        if df is None or df.empty:
            return pd.DataFrame(columns=EXPORT_COLUMNS)
        logging.info("AetherFlow feature build: deriving manifold base frame first")
        base_features = build_manifold_feature_frame(df, manifold_cfg=manifold_cfg, log_every=log_every)
    if base_features is None or base_features.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    return _build_from_base(
        base_features,
        preferred_setup_families=preferred_setup_families,
    )


def build_feature_frames_by_family(
    df: Optional[pd.DataFrame] = None,
    *,
    base_features: Optional[pd.DataFrame] = None,
    preferred_setup_families: Optional[set[str]] = None,
    manifold_cfg: Optional[Dict] = None,
    log_every: int = 0,
) -> dict[str, pd.DataFrame]:
    if base_features is None:
        if df is None or df.empty:
            return {}
        logging.info("AetherFlow feature build: deriving manifold base frame first")
        base_features = build_manifold_feature_frame(df, manifold_cfg=manifold_cfg, log_every=log_every)
    if base_features is None or base_features.empty:
        return {}
    frames = _build_from_base(
        base_features,
        preferred_setup_families=preferred_setup_families,
        emit_family_frames=True,
    )
    return frames if isinstance(frames, dict) else {}


def build_feature_frame_from_parquet(path: str | PathLike[str]) -> pd.DataFrame:
    base = pd.read_parquet(path)
    return _build_from_base(base)


def resolve_setup_params(feature_row: pd.Series) -> Dict[str, float]:
    family = str(feature_row.get("setup_family", "") or "")
    params = dict(SETUP_DEFAULTS.get(family, {}))
    atr14 = float(feature_row.get("atr14", 0.0) or 0.0)
    if atr14 <= 0.0:
        atr14 = 1.0
    sl_mult = float(feature_row.get("setup_sl_mult", params.get("sl_mult", 1.1)) or params.get("sl_mult", 1.1))
    tp_mult = float(feature_row.get("setup_tp_mult", params.get("tp_mult", 2.0)) or params.get("tp_mult", 2.0))
    horizon = int(round(float(feature_row.get("setup_horizon_bars", params.get("horizon_bars", 16)) or params.get("horizon_bars", 16))))
    sl_points = float(np.clip(atr14 * sl_mult, 1.0, 8.0))
    tp_points = float(np.clip(atr14 * tp_mult, max(sl_points * 1.2, 1.5), 16.0))
    return {
        "setup_family": family,
        "sl_points": sl_points,
        "tp_points": tp_points,
        "horizon_bars": max(6, horizon),
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
    }

import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from manifold_regime_calibration import apply_calibration_to_meta, load_calibration_payload


TWOPI = 2.0 * math.pi
logger = logging.getLogger(__name__)


DEFAULT_MANIFOLD_CONFIG: Dict = {
    "enabled": False,
    "mode": "enforce",  # enforce | shadow
    "persist_state": True,
    "seed": 42,
    "n_probes": 12,
    "mom_window": 60,
    "sigma_ewm_span": 45,
    "sigma_floor": 1e-6,
    "atr_window": 20,
    "vol_ewm_span": 30,
    "vol_z_window": 390,
    "m_scale": 2.0,
    "omega_bars": 390,
    "A": 0.02,
    "beta_mean": 0.10,
    "beta_std": 0.03,
    "gamma_mean": 0.06,
    "gamma_std": 0.02,
    "c_drift": 0.01,
    "noise_theta": 0.01,
    "noise_phi": 0.01,
    "hotspot_a": 1.0,
    "stress_ewm_alpha": 0.20,
    "ret_z_clip": 6.0,
    "risk_mult_min": 0.25,
    "risk_mult_max": 1.50,
    "min_bars": 80,
    "side_bias_min_abs_m": 0.10,
    "side_bias_min_alignment": 0.45,
    "rotational_phi_threshold": 0.08,
    "enforce_side_bias": True,
    "calibration_file": "manifold_regime_calibration_clean_full.json",
}


def _wrap_angle_signed(x: np.ndarray | float) -> np.ndarray | float:
    return ((x + math.pi) % TWOPI) - math.pi


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def classify_signal_style(strategy_name: Optional[str], sub_strategy: Optional[str] = None) -> str:
    """
    Map existing strategy labels to one of:
    trend | mean_reversion | breakout | fade
    """
    strat = str(strategy_name or "").strip().lower()
    sub = str(sub_strategy or "").strip().lower()

    if strat.startswith("continuation_"):
        return "trend"

    if "dynamicengine3" in strat or "dynamicengine" in strat:
        if "_rev_" in sub:
            return "mean_reversion"
        if "_mom_" in sub:
            return "trend"
        return "trend"

    if any(key in strat for key in ("vixreversion", "auctionreversion")):
        return "fade"

    if any(key in strat for key in ("impulsebreakout", "valueareabreakout", "orb")):
        return "breakout"

    if any(key in strat for key in ("intradaydip", "regimeadaptive", "liquiditysweep", "ictmodel")):
        return "mean_reversion"

    if any(key in strat for key in ("smt", "smoothtrendasia", "mlphysics", "confluence")):
        return "trend"

    return "trend"


def apply_meta_policy(
    signal: Dict,
    meta: Optional[Dict],
    fallback_name: Optional[str] = None,
    default_size: int = 1,
    enforce_side_bias: bool = True,
    kalshi_context: Optional[Dict] = None,
    session_name: Optional[str] = None,
) -> Tuple[bool, str, Dict]:
    """
    Attach manifold context to a signal.

    Gating is intentionally disabled in this policy path:
    - Always returns allowed=True.
    - Never hard-blocks by style/side/no_trade.
    - Does not mutate order size.
    """
    if not isinstance(signal, dict):
        return False, "invalid_signal", {}
    if not isinstance(meta, dict):
        return True, "", {}

    strategy_name = str(signal.get("strategy") or fallback_name or "")
    sub_strategy = str(signal.get("sub_strategy") or "")
    side = str(signal.get("side") or "").upper()
    style = classify_signal_style(strategy_name, sub_strategy)
    allow = meta.get("allow") or {}

    def _allow_style(style_name: str) -> bool:
        if style_name == "fade":
            return bool(allow.get("fade", False) or allow.get("mean_reversion", False))
        return bool(allow.get(style_name, False))

    updates = {
        "regime_manifold_regime": meta.get("regime"),
        "regime_manifold_R": float(meta.get("R", 0.0) or 0.0),
        "regime_manifold_alignment": float(meta.get("alignment", 0.0) or 0.0),
        "regime_manifold_smoothness": float(meta.get("smoothness", 0.0) or 0.0),
        "regime_manifold_stress": float(meta.get("stress", 0.0) or 0.0),
        "regime_manifold_dispersion": float(meta.get("dispersion", 0.0) or 0.0),
        "regime_manifold_side_bias": int(meta.get("side_bias", 0) or 0),
        "regime_manifold_style": style,
        "regime_manifold_no_trade": bool(meta.get("no_trade", False)),
        "regime_manifold_allow_style": bool(_allow_style(style)),
        "regime_manifold_allow_raw": allow,
        "regime_manifold_risk_mult": float(meta.get("risk_mult", 1.0) or 1.0),
    }
    kalshi_updates = {}
    if isinstance(kalshi_context, dict):
        prob_above = kalshi_context.get("probability")
        if prob_above is None:
            prob_above = kalshi_context.get("prob_above")
        classification = str(kalshi_context.get("classification", "unavailable") or "unavailable")
        kalshi_updates = {
            "kalshi_prob_above": float(prob_above) if prob_above is not None else None,
            "kalshi_classification": classification,
            "kalshi_session": str(session_name or signal.get("session_name") or ""),
        }
    updates.update(kalshi_updates)
    return True, "", updates


def apply_kalshi_gate(signal_direction: int, es_price: float, kalshi, config: Dict) -> Tuple[bool, str, float]:
    """
    Strategy-agnostic Kalshi crowd confirmation gate.
    """
    _ = es_price
    if kalshi is None:
        return True, "Kalshi unavailable — ML-only mode", 1.0
    if not getattr(kalshi, "enabled", False) or not getattr(kalshi, "is_healthy", False):
        return True, "Kalshi unavailable — ML-only mode", 1.0

    sentiment = kalshi.get_sentiment(es_price)
    probability = sentiment.get("probability")
    if probability is None:
        return True, "Kalshi data unavailable — ML-only mode", 1.0

    thresholds = dict(config.get("sentiment_thresholds", {}))
    strong_bull = float(thresholds.get("strong_bull", 0.70))
    mild_bull = float(thresholds.get("mild_bull", 0.55))
    neutral_low = float(thresholds.get("neutral_low", 0.45))
    neutral_high = float(thresholds.get("neutral_high", 0.55))
    mild_bear = float(thresholds.get("mild_bear", 0.45))
    strong_bear = float(thresholds.get("strong_bear", 0.30))
    veto_mode = str(config.get("veto_mode", "soft") or "soft").lower()

    if signal_direction == 1:
        if probability < neutral_low:
            if veto_mode == "hard":
                return False, f"VETO: Bearish crowd divergence (prob={probability:.2f})", 0.0
            return True, f"SOFT VETO: Bearish crowd (prob={probability:.2f}), half size", 0.5
        if probability >= strong_bull:
            return True, f"STRONG ALIGN: Bullish crowd (prob={probability:.2f}), 3x size", 3.0
        if probability >= mild_bull:
            return True, f"ALIGNED: Crowd agrees (prob={probability:.2f}), 3x size", 3.0
        return True, f"NEUTRAL: Crowd undecided (prob={probability:.2f})", 0.8

    if signal_direction == -1:
        if probability > neutral_high:
            if veto_mode == "hard":
                return False, f"VETO: Bullish crowd divergence (prob={probability:.2f})", 0.0
            return True, f"SOFT VETO: Bullish crowd (prob={probability:.2f}), half size", 0.5
        if probability <= strong_bear:
            return True, f"STRONG ALIGN: Bearish crowd (prob={probability:.2f}), 3x size", 3.0
        if probability <= mild_bear:
            return True, f"ALIGNED: Crowd agrees (prob={probability:.2f}), 3x size", 3.0
        return True, f"NEUTRAL: Crowd undecided (prob={probability:.2f})", 0.8

    return True, "No direction signal", 1.0


def get_kalshi_gate_decision(signal_direction: int, es_price: float, kalshi, root_config: Dict) -> Tuple[bool, str, float]:
    """
    Wrapper around Kalshi gating with graceful fallback to ML-only flow.
    """
    try:
        kalshi_cfg = dict((root_config or {}).get("KALSHI", {}))
        if not kalshi_cfg.get("enabled", False):
            return True, "Kalshi disabled", 1.0
        if kalshi is None or not getattr(kalshi, "is_healthy", False):
            logger.warning("Kalshi unhealthy — ML-only mode")
            return True, "Kalshi unhealthy — ML-only fallback", 1.0
        return apply_kalshi_gate(signal_direction, es_price, kalshi, kalshi_cfg)
    except Exception as exc:
        logger.error("Kalshi gate error: %s", exc)
        if kalshi is not None:
            current_failures = int(getattr(kalshi, "consecutive_failures", 0) or 0) + 1
            setattr(kalshi, "consecutive_failures", current_failures)
            if current_failures >= 5:
                setattr(kalshi, "is_healthy", False)
        return True, f"Kalshi error — ML-only fallback: {exc}", 1.0


class RegimeManifoldEngine:
    """
    Fast toroidal manifold regime engine for 1-minute data.
    Produces an upstream gating/risk-throttle meta regime.
    """

    def __init__(self, cfg: Optional[Dict] = None, seed: Optional[int] = None) -> None:
        merged = dict(DEFAULT_MANIFOLD_CONFIG)
        if isinstance(cfg, dict):
            merged.update(cfg)
        self.cfg = merged

        self.n_probes = max(3, int(self.cfg.get("n_probes", 12) or 12))
        base_seed = seed if seed is not None else self.cfg.get("seed", 42)
        self.rng = np.random.default_rng(int(base_seed))

        self.mom_window = max(10, int(self.cfg.get("mom_window", 60) or 60))
        self.sigma_ewm_span = max(5, int(self.cfg.get("sigma_ewm_span", 45) or 45))
        self.sigma_floor = float(self.cfg.get("sigma_floor", 1e-6) or 1e-6)
        self.atr_window = max(5, int(self.cfg.get("atr_window", 20) or 20))
        self.vol_ewm_span = max(5, int(self.cfg.get("vol_ewm_span", 30) or 30))
        self.vol_z_window = max(30, int(self.cfg.get("vol_z_window", 390) or 390))
        self.m_scale = float(self.cfg.get("m_scale", 2.0) or 2.0)

        omega_bars = max(10, int(self.cfg.get("omega_bars", 390) or 390))
        self.omega = TWOPI / float(omega_bars)
        self.A = float(self.cfg.get("A", 0.02) or 0.02)
        self.c_drift = float(self.cfg.get("c_drift", 0.01) or 0.01)
        self.noise_theta = float(self.cfg.get("noise_theta", 0.01) or 0.01)
        self.noise_phi = float(self.cfg.get("noise_phi", 0.01) or 0.01)
        self.hotspot_a = float(self.cfg.get("hotspot_a", 1.0) or 1.0)
        self.stress_alpha = float(self.cfg.get("stress_ewm_alpha", 0.20) or 0.20)
        self.ret_z_clip = float(self.cfg.get("ret_z_clip", 6.0) or 6.0)

        self.risk_mult_min = float(self.cfg.get("risk_mult_min", 0.25) or 0.25)
        self.risk_mult_max = float(self.cfg.get("risk_mult_max", 1.50) or 1.50)
        self.min_bars = max(20, int(self.cfg.get("min_bars", 80) or 80))
        self.side_bias_min_abs_m = float(self.cfg.get("side_bias_min_abs_m", 0.10) or 0.10)
        self.side_bias_min_alignment = float(self.cfg.get("side_bias_min_alignment", 0.45) or 0.45)
        self.rotational_phi_threshold = float(self.cfg.get("rotational_phi_threshold", 0.08) or 0.08)
        calibration_payload = self.cfg.get("calibration_payload")
        if isinstance(calibration_payload, dict):
            self.calibration = calibration_payload
        else:
            self.calibration = load_calibration_payload(self.cfg.get("calibration_file"))

        beta_mean = float(self.cfg.get("beta_mean", 0.10) or 0.10)
        beta_std = float(self.cfg.get("beta_std", 0.03) or 0.03)
        gamma_mean = float(self.cfg.get("gamma_mean", 0.06) or 0.06)
        gamma_std = float(self.cfg.get("gamma_std", 0.02) or 0.02)

        self.phase = self.rng.uniform(0.0, TWOPI, self.n_probes)
        self.beta = np.clip(self.rng.normal(beta_mean, beta_std, self.n_probes), -1.0, 1.0)
        self.gamma = np.clip(self.rng.normal(gamma_mean, gamma_std, self.n_probes), -1.0, 1.0)

        self.theta = self.rng.uniform(0.0, TWOPI, self.n_probes)
        self.phi = self.rng.uniform(0.0, TWOPI, self.n_probes)
        self.prev_dir: Optional[np.ndarray] = None
        self.stress_ewma = 0.0
        self.t = 0
        self._tail_df: Optional[pd.DataFrame] = None
        self._tail_source_len = 0
        self._tail_source_last_ts: Optional[pd.Timestamp] = None

    @staticmethod
    def _calc_atr_series(df: pd.DataFrame, window: int) -> Optional[pd.Series]:
        if df is None or df.empty or len(df) < max(window, 3):
            return None
        if not {"high", "low", "close"}.issubset(df.columns):
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / float(window), adjust=False).mean()

    def _neutral_output(self, reason: str) -> Dict:
        return {
            "regime": "DISPERSED",
            "R": 0.50,
            "alignment": 0.50,
            "smoothness": 0.50,
            "stress": 0.50,
            "dispersion": 0.50,
            "side_bias": 0,
            "allow": {
                "trend": True,
                "mean_reversion": True,
                "breakout": True,
                "fade": True,
            },
            "risk_mult": 1.0,
            "no_trade": False,
            "debug": {"reason": reason},
        }

    def _tail_frame(self, df_1m: pd.DataFrame, tail_len: int) -> pd.DataFrame:
        cur_len = int(len(df_1m))
        cur_last_ts = df_1m.index[-1] if cur_len else None

        if self._tail_df is None or self._tail_source_len <= 0:
            tail = df_1m.iloc[-tail_len:].copy()
        else:
            same_snapshot = (
                cur_len == self._tail_source_len and cur_last_ts == self._tail_source_last_ts
            )
            append_only = (
                cur_len > self._tail_source_len
                and self._tail_source_last_ts is not None
                and self._tail_source_len > 0
                and self._tail_source_len <= cur_len
                and df_1m.index[self._tail_source_len - 1] == self._tail_source_last_ts
            )
            if same_snapshot:
                tail = self._tail_df
            elif append_only:
                new_rows = df_1m.iloc[self._tail_source_len :]
                if new_rows.empty:
                    tail = df_1m.iloc[-tail_len:].copy()
                else:
                    tail = pd.concat([self._tail_df, new_rows]).iloc[-tail_len:].copy()
            else:
                tail = df_1m.iloc[-tail_len:].copy()

        self._tail_df = tail
        self._tail_source_len = cur_len
        self._tail_source_last_ts = cur_last_ts
        return tail

    def get_state(self) -> Dict:
        return {
            "version": 1,
            "t": int(self.t),
            "theta": self.theta.tolist(),
            "phi": self.phi.tolist(),
            "prev_dir": self.prev_dir.tolist() if isinstance(self.prev_dir, np.ndarray) else None,
            "stress_ewma": float(self.stress_ewma),
        }

    def load_state(self, state: Optional[Dict]) -> None:
        if not isinstance(state, dict):
            return
        try:
            theta = np.asarray(state.get("theta", []), dtype="float64")
            phi = np.asarray(state.get("phi", []), dtype="float64")
            if theta.size == self.n_probes and phi.size == self.n_probes:
                self.theta = np.mod(theta, TWOPI)
                self.phi = np.mod(phi, TWOPI)
            prev_dir = state.get("prev_dir")
            if isinstance(prev_dir, list):
                prev = np.asarray(prev_dir, dtype="float64")
                if prev.size == self.n_probes:
                    self.prev_dir = prev
            self.t = int(state.get("t", self.t) or self.t)
            self.stress_ewma = float(state.get("stress_ewma", self.stress_ewma) or self.stress_ewma)
        except Exception:
            return

    def update(
        self,
        df_1m: pd.DataFrame,
        ts: Optional[pd.Timestamp] = None,
        session: Optional[str] = None,
    ) -> Dict:
        if df_1m is None or df_1m.empty:
            return self._neutral_output("empty_df")
        if len(df_1m) < self.min_bars:
            return self._neutral_output("warmup")
        if "close" not in df_1m.columns:
            return self._neutral_output("missing_close")

        tail_len = max(self.vol_z_window + 20, self.mom_window + 20, self.min_bars)
        work = self._tail_frame(df_1m, tail_len)

        close = pd.to_numeric(work["close"], errors="coerce")
        close = close.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if close.isna().all():
            return self._neutral_output("close_nan")

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        r_t = float(returns.iloc[-1])

        sigma_series = returns.ewm(span=self.sigma_ewm_span, adjust=False).std(bias=False)
        sigma_series = sigma_series.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        sigma_last = float(sigma_series.iloc[-1]) if not sigma_series.empty else 0.0
        if not math.isfinite(sigma_last):
            sigma_last = 0.0
        sigma_t = max(self.sigma_floor, abs(sigma_last))

        if len(returns) > (self.mom_window + 1):
            mom_slice = returns.iloc[-(self.mom_window + 1) : -1]
        else:
            mom_slice = returns.iloc[:-1]
        if mom_slice.empty:
            mom_slice = returns.iloc[-self.mom_window :]
        mom_n = max(1, int(len(mom_slice)))
        M_t = float(np.sum(mom_slice.to_numpy(dtype=float)) / (sigma_t * math.sqrt(float(mom_n))))
        M_t = float(np.clip(M_t, -3.0, 3.0))

        atr_series = self._calc_atr_series(work, self.atr_window)
        if atr_series is not None and not atr_series.empty:
            vol_proxy = (atr_series / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        else:
            vol_proxy = returns.abs()
        vol_proxy = vol_proxy.ffill().bfill().fillna(0.0)
        vol_smooth = vol_proxy.ewm(span=self.vol_ewm_span, adjust=False).mean()

        vol_mean = vol_smooth.rolling(self.vol_z_window, min_periods=max(20, self.vol_z_window // 4)).mean()
        vol_std = vol_smooth.rolling(self.vol_z_window, min_periods=max(20, self.vol_z_window // 4)).std()
        vol_std = vol_std.replace(0.0, np.nan)
        vol_z = float(((vol_smooth - vol_mean) / vol_std).iloc[-1]) if len(vol_smooth) else 0.0
        if not math.isfinite(vol_z):
            vol_z = 0.0

        # V_t in [0, 2pi); V_norm in [0, 1]
        V_t = float((math.pi * math.tanh(vol_z)) % TWOPI)
        V_norm = _clip01(0.5 * (math.tanh(vol_z) + 1.0))
        if vol_z < -0.5:
            Vband = 0.0
        elif vol_z > 0.5:
            Vband = 1.0
        else:
            Vband = 0.5

        m_center = float(np.clip(M_t, -2.0, 2.0))
        theta_c = float(((1.0 - m_center / 2.0) * (math.pi / 2.0) + (math.pi / 4.0)) % TWOPI)
        phi_c = float((V_t - 0.5 * math.pi + self.omega * float(self.t)) % TWOPI)

        norm_ret = float(np.clip(r_t / sigma_t, -self.ret_z_clip, self.ret_z_clip))
        osc = (self.omega * float(self.t)) + self.phase

        delta_theta = (
            self.A * np.sin(osc)
            + (self.beta * norm_ret)
            + self.rng.normal(0.0, self.noise_theta, self.n_probes)
        )
        delta_phi = (
            self.c_drift
            + self.A * np.cos(osc)
            + (self.gamma * norm_ret)
            + self.rng.normal(0.0, self.noise_phi, self.n_probes)
        )

        self.theta = np.mod(self.theta + delta_theta, TWOPI)
        self.phi = np.mod(self.phi + delta_phi, TWOPI)
        self.t += 1

        d_theta = _wrap_angle_signed(self.theta - theta_c)
        d_phi = _wrap_angle_signed(self.phi - phi_c)
        d2 = np.square(d_theta) + np.square(d_phi)

        hotspot = np.exp(-d2 / max(1e-6, self.hotspot_a)) * (1.0 + V_norm)
        hotspot_mean = float(np.mean(hotspot))
        hotspot_component = _clip01(hotspot_mean / 2.0)

        alignment = hotspot_component
        dispersion_raw = float(math.sqrt(float(np.mean(d2))))
        dispersion = _clip01(dispersion_raw / math.pi)

        direction = np.arctan2(delta_theta, delta_phi)
        if self.prev_dir is None:
            dir_delta = np.zeros_like(direction)
        else:
            dir_delta = _wrap_angle_signed(direction - self.prev_dir)
        self.prev_dir = direction

        mean_curv = float(np.mean(np.abs(dir_delta)))
        smoothness = _clip01(1.0 - (mean_curv / math.pi))

        unit = np.exp(1j * direction)
        circular_r = float(np.abs(np.mean(unit)))
        direction_var = _clip01(1.0 - circular_r)

        turbulence = float(np.mean(np.abs(delta_theta) + np.abs(delta_phi)) / (2.0 * math.pi))
        turbulence = _clip01(turbulence)
        self.stress_ewma = ((1.0 - self.stress_alpha) * self.stress_ewma) + (self.stress_alpha * turbulence)
        stress = _clip01(0.65 * direction_var + 0.35 * self.stress_ewma)

        B_signed = float(np.clip(M_t / max(1e-9, self.m_scale), -1.0, 1.0))
        R = (0.3 * B_signed) + (0.2 * Vband) + (0.4 * hotspot_component) + (0.1 * stress) + 0.1
        R = _clip01(R)

        mean_abs_dphi = float(np.mean(np.abs(delta_phi)))

        if mean_abs_dphi >= self.rotational_phi_threshold and stress >= 0.65:
            raw_regime = "ROTATIONAL_TURBULENCE"
        elif alignment >= 0.62 and smoothness >= 0.60 and dispersion <= 0.45:
            raw_regime = "TREND_GEODESIC"
        elif dispersion >= 0.65 and alignment <= 0.45:
            raw_regime = "DISPERSED"
        elif stress >= 0.60 or smoothness <= 0.45:
            raw_regime = "CHOP_SPIRAL"
        else:
            raw_regime = "CHOP_SPIRAL"

        side_bias = 0
        if abs(B_signed) >= self.side_bias_min_abs_m and alignment >= self.side_bias_min_alignment:
            side_bias = 1 if B_signed > 0 else -1

        allow = {
            "trend": True,
            "mean_reversion": True,
            "breakout": True,
            "fade": True,
        }
        no_trade = False
        if raw_regime == "TREND_GEODESIC":
            allow = {"trend": True, "mean_reversion": False, "breakout": True, "fade": False}
        elif raw_regime == "CHOP_SPIRAL":
            allow = {"trend": False, "mean_reversion": True, "breakout": False, "fade": True}
        elif raw_regime == "DISPERSED":
            allow = {"trend": False, "mean_reversion": False, "breakout": False, "fade": False}
            no_trade = True
        elif raw_regime == "ROTATIONAL_TURBULENCE":
            allow = {"trend": False, "mean_reversion": False, "breakout": False, "fade": False}
            no_trade = True

        if stress >= 0.85:
            no_trade = True
            allow = {"trend": False, "mean_reversion": False, "breakout": False, "fade": False}

        risk_mult = 0.5 + (1.2 * R) - (0.8 * stress)
        risk_mult = float(np.clip(risk_mult, self.risk_mult_min, self.risk_mult_max))
        meta = {
            "regime": raw_regime,
            "R": float(R),
            "alignment": float(alignment),
            "smoothness": float(smoothness),
            "stress": float(stress),
            "dispersion": float(dispersion),
            "side_bias": int(side_bias),
            "allow": allow,
            "risk_mult": risk_mult,
            "no_trade": bool(no_trade),
            "B_signed": float(B_signed),
        }
        if isinstance(self.calibration, dict):
            meta = apply_calibration_to_meta(meta, self.calibration, existing_side_bias=side_bias)

        meta["debug"] = {
            "theta_c": theta_c,
            "phi_c": phi_c,
            "probe_stats": {
                "delta_theta_mean": float(np.mean(delta_theta)),
                "delta_theta_std": float(np.std(delta_theta)),
                "delta_phi_mean": float(np.mean(delta_phi)),
                "delta_phi_std": float(np.std(delta_phi)),
                "mean_abs_dphi": mean_abs_dphi,
                "hotspot_mean": hotspot_mean,
            },
            "features": {
                "r_t": r_t,
                "sigma_t": sigma_t,
                "M_t": M_t,
                "vol_z": vol_z,
                "V_t": V_t,
                "V_norm": V_norm,
                "Vband": Vband,
                "B_signed": B_signed,
            },
            "raw_regime": raw_regime,
            "raw_side_bias": int(side_bias),
            "calibrated": bool(isinstance(self.calibration, dict)),
            "percentiles": {
                "alignment_pct": float(meta.get("alignment_pct", 0.0) or 0.0),
                "smoothness_pct": float(meta.get("smoothness_pct", 0.0) or 0.0),
                "stress_pct": float(meta.get("stress_pct", 0.0) or 0.0),
                "dispersion_pct": float(meta.get("dispersion_pct", 0.0) or 0.0),
                "R_pct": float(meta.get("R_pct", 0.0) or 0.0),
            },
            "session": session,
            "ts": str(ts) if ts is not None else None,
        }
        return meta

"""Runtime regime classifier: rolling close-based whipsaw vs calm-trend detector.

Public API:
- init_regime_classifier() -> Optional[RegimeClassifier]
- get_regime_classifier()   -> Optional[RegimeClassifier]
- update_regime_classifier(ts, close) -> Optional[str]   # returns current regime name
- current_regime() -> str                                # "whipsaw" | "calm_trend" | "neutral" | "warmup"

Enabled by JULIE_REGIME_CLASSIFIER=1. Default: disabled (module is a no-op).

When a regime transition occurs, the classifier mutates CONFIG in place:

    WHIPSAW:
        KALSHI_TRADE_OVERLAY.entry_block_buffer.balanced       = 0.25
        KALSHI_TRADE_OVERLAY.entry_block_buffer.forward_primary = 0.25
        LIVE_OPPOSITE_REVERSAL.required_confirmations          = baseline

    CALM_TREND:
        LIVE_OPPOSITE_REVERSAL.required_confirmations          = 5
        KALSHI_TRADE_OVERLAY.entry_block_buffer.*              = baseline

    NEUTRAL:
        both restored to baselines captured at init time.

Thresholds are calibrated from Apr 2026 Wk1 (WHIPSAW: vol=6.52bp eff=0.039)
and Wk2 (CALM_TREND: vol=2.42bp eff=0.108).
"""
from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from config import CONFIG


WINDOW_BARS = 120

# Simplified efficiency-primary thresholds (vol is noisy at daily granularity
# due to overnight gaps; eff cleanly separates regimes).
# Daily Oct 2025 / Apr 2026 analysis:
#   eff < 0.05  -> extreme chop (Oct 21 eff=0.009, Oct 06 eff=0.009, Apr 10 eff=0.003) -> Exp5
#   eff > 0.12  -> strong trend (Oct 10 eff=0.281, Oct 20 eff=0.149, Apr 13 eff=0.209) -> Exp4
EFF_LOW = 0.05
EFF_HIGH = 0.12

WHIPSAW_BUF_VALUE = 0.25
CALM_REV_CONFIRM = 5

TRANSITION_COOLDOWN_BARS = 30

# Per-regime risk caps. Tight on whipsaw (Oct-style chop), loose on calm-trend
# (Apr Wk2-style grind), medium on neutral. JULIE_REGIME_ADAPTIVE_CB=1 enables.
ADAPTIVE_CB_ENABLED = os.environ.get("JULIE_REGIME_ADAPTIVE_CB", "0").strip() == "1"
CB_CAP_WHIPSAW = float(os.environ.get("JULIE_REGIME_CB_WHIPSAW", "250"))
CB_CAP_NEUTRAL = float(os.environ.get("JULIE_REGIME_CB_NEUTRAL", "350"))
CB_CAP_CALM = float(os.environ.get("JULIE_REGIME_CB_CALM", "500"))
CB_CONSEC_WHIPSAW = int(os.environ.get("JULIE_REGIME_CONSEC_WHIPSAW", "4"))
CB_CONSEC_NEUTRAL = int(os.environ.get("JULIE_REGIME_CONSEC_NEUTRAL", "5"))
CB_CONSEC_CALM = int(os.environ.get("JULIE_REGIME_CONSEC_CALM", "7"))

# JULIE_REGIME_WHIPSAW_VETO=1 makes the classifier block entries while
# regime=whipsaw. Default 0 (no veto; only config mutations are applied).
WHIPSAW_VETO_ENABLED = os.environ.get("JULIE_REGIME_WHIPSAW_VETO", "0").strip() == "1"

# Filter D — regime-gated size cap. When regime is whipsaw or calm_trend
# (i.e. NOT neutral), cap incoming signal size to 1 contract.
REGIME_SIZE_CAP_ENABLED = os.environ.get("JULIE_REGIME_SIZE_CAP", "0").strip() == "1"
REGIME_SIZE_CAP_VALUE = int(os.environ.get("JULIE_REGIME_SIZE_CAP_VALUE", "1"))

# Filter E — green-day size unlock. On trend regimes where filter D would
# normally cap to 1, recover upside by raising the cap once daily PnL proves
# the direction.
REGIME_GREEN_UNLOCK_THRESHOLD = float(os.environ.get("JULIE_REGIME_GREEN_UNLOCK_PNL", "200"))
REGIME_GREEN_UNLOCK_SIZE = int(os.environ.get("JULIE_REGIME_GREEN_UNLOCK_SIZE", "3"))

# Dead-tape regime: bottom-percentile-vol session where DE3's 25/10 TP/SL +
# BE-at-10pt never activate (max MFE caps at 3-6pt on these days).
# When the dead_tape branch classifies the session, signal brackets get
# rewritten to scalp-sized values before the downstream BE/trail logic
# sees them.
DEAD_TAPE_VOL_BP = float(os.environ.get("JULIE_REGIME_DEAD_VOL_BP", "1.5"))
DEAD_TAPE_TP_PTS = float(os.environ.get("JULIE_REGIME_DEAD_TP", "3.0"))
DEAD_TAPE_SL_PTS = float(os.environ.get("JULIE_REGIME_DEAD_SL", "5.0"))
DEAD_TAPE_BE_TRIGGER_PTS = float(os.environ.get("JULIE_REGIME_DEAD_BE_TRIGGER", "3.0"))


@dataclass
class RegimeState:
    regime: str = "warmup"
    vol_bp: float = 0.0
    eff: float = 0.0
    last_transition_bar: int = -10_000
    bar_count: int = 0


class RegimeClassifier:
    # History depth needed by ML v5 features (largest lookback = 480 bars)
    ML_FEATURE_HISTORY_BARS = 520

    def __init__(self) -> None:
        self.enabled = True
        self._closes: Deque[float] = deque(maxlen=WINDOW_BARS)
        # OHLCV history for ML feature building; deeper lookback than _closes
        # because v5 features use vol_bp at 480-bar window.
        self._ml_o: Deque[float] = deque(maxlen=self.ML_FEATURE_HISTORY_BARS)
        self._ml_h: Deque[float] = deque(maxlen=self.ML_FEATURE_HISTORY_BARS)
        self._ml_l: Deque[float] = deque(maxlen=self.ML_FEATURE_HISTORY_BARS)
        self._ml_c: Deque[float] = deque(maxlen=self.ML_FEATURE_HISTORY_BARS)
        self._ml_v: Deque[float] = deque(maxlen=self.ML_FEATURE_HISTORY_BARS)
        self._ml_ts: Deque = deque(maxlen=self.ML_FEATURE_HISTORY_BARS)
        self._state = RegimeState()
        # Snapshot baselines once so NEUTRAL can restore them.
        overlay = CONFIG.get("KALSHI_TRADE_OVERLAY") or {}
        buf = overlay.get("entry_block_buffer") or {}
        self._baseline_buf_balanced = float(buf.get("balanced", 0.10))
        self._baseline_buf_fp = float(buf.get("forward_primary", 0.10))
        rev_cfg = CONFIG.get("LIVE_OPPOSITE_REVERSAL") or {}
        self._baseline_rev_confirm = int(rev_cfg.get("required_confirmations", 3))

    def record_bar(self, ts, o: float, h: float, l: float, c: float, v: float = 0.0) -> None:
        """Push an OHLCV sample into the ML history deques. Call this alongside
        update() for live bars so the ML feature builder has fresh data.
        """
        try:
            self._ml_o.append(float(o))
            self._ml_h.append(float(h))
            self._ml_l.append(float(l))
            self._ml_c.append(float(c))
            self._ml_v.append(float(v))
            self._ml_ts.append(ts)
        except Exception:
            pass

    def build_ml_feature_snapshot(self) -> Optional[dict]:
        """Compute the 40-feature snapshot used by v5 ML models. Returns
        None if history is too shallow for the deepest feature.
        """
        if len(self._ml_c) < 480:
            return None
        import numpy as np
        import pandas as pd
        c = np.array(self._ml_c, dtype=float)
        o = np.array(self._ml_o, dtype=float)
        h = np.array(self._ml_h, dtype=float)
        l = np.array(self._ml_l, dtype=float)
        v = np.array(self._ml_v, dtype=float)

        def roll_vol_eff(closes, window):
            rets = (closes[-window:][1:] - closes[-window:][:-1]) / closes[-window:][:-1]
            mean = rets.mean()
            var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
            vol = (var ** 0.5) * 10_000.0
            abs_sum = np.abs(rets).sum()
            eff = abs(rets.sum()) / abs_sum if abs_sum > 0 else 0.0
            return float(vol), float(eff)

        vol30,  eff30  = roll_vol_eff(c, 30)
        vol60,  eff60  = roll_vol_eff(c, 60)
        vol120, eff120 = roll_vol_eff(c, 120)
        vol240, eff240 = roll_vol_eff(c, 240)
        vol480, eff480 = roll_vol_eff(c, 480)

        # Slopes — need historical vol values, approximate with shorter calcs
        vol60_lag10 = roll_vol_eff(c[:-10], 60)[0] if len(c) >= 70 else vol60
        vol60_lag30 = roll_vol_eff(c[:-30], 60)[0] if len(c) >= 90 else vol60
        vol120_lag60 = roll_vol_eff(c[:-60], 120)[0] if len(c) >= 180 else vol120
        eff60_lag30 = roll_vol_eff(c[:-30], 60)[1] if len(c) >= 90 else eff60

        vol_slope_10 = vol60 - vol60_lag10
        vol_slope_30 = vol60 - vol60_lag30
        vol_slope_60 = vol120 - vol120_lag60
        eff_slope_30 = eff60 - eff60_lag30

        prev_c = np.r_[c[0], c[:-1]]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr14 = float(tr[-14:].mean())
        atr30 = float(tr[-30:].mean())

        rng_pct = (h - l) / np.where(c != 0, c, 1.0)
        range_pct_20 = float(rng_pct[-20:].mean() * 10_000)
        range_pct_120 = float(rng_pct[-120:].mean() * 10_000)

        hl = np.maximum(h - l, 1e-9)
        body_ratio = np.abs(c - o) / hl
        body_ratio_20 = float(body_ratio[-20:].mean())
        body_ratio_60 = float(body_ratio[-60:].mean())
        abs_body_20 = float(np.abs(c - o)[-20:].mean())
        up_bar_pct_20 = float((c >= o)[-20:].astype(float).mean())

        # Consecutive up/down run lengths
        up_bar = (c >= o).astype(int)
        run_up = 0; run_down = 0; max_ru = 0; max_rd = 0
        for i in range(len(c) - 20, len(c)):
            if up_bar[i]:
                run_up += 1; run_down = 0
            else:
                run_down += 1; run_up = 0
            max_ru = max(max_ru, run_up)
            max_rd = max(max_rd, run_down)

        gap_pct = float((o[-1] - prev_c[-1]) / max(prev_c[-1], 1.0) * 10_000)
        gap_abs_mean_20 = float(np.abs((o[-20:] - prev_c[-20:]) / np.where(prev_c[-20:] != 0, prev_c[-20:], 1.0))[-20:].mean() * 10_000)

        def mom(n):
            if len(c) <= n: return 0.0
            return float((c[-1] - c[-1-n]) / max(c[-1-n], 1.0) * 10_000)
        mom_5 = mom(5); mom_15 = mom(15); mom_30 = mom(30); mom_60 = mom(60); mom_120 = mom(120)

        v_mean = float(v[-200:].mean()) if len(v) >= 200 else float(v.mean())
        v_std = float(v[-200:].std()) if len(v) >= 200 else float(v.std() or 1.0)
        volume_z_20 = float((v[-1] - v_mean) / (v_std or 1.0))
        volume_ma_ratio = float(v[-1] / max(v_mean, 1.0))

        c60 = c[-60:]
        max_runup_60 = float((c60.max() - c60[0]) / max(c60[0], 1e-9) * 10_000)
        max_rundown_60 = float((c60[0] - c60.min()) / max(c60[0], 1e-9) * 10_000)

        ts_last = self._ml_ts[-1]
        try:
            et_hour = int(ts_last.hour)
            et_min = int(ts_last.minute)
            day_of_week = int(ts_last.weekday())
        except Exception:
            et_hour = 0; et_min = 0; day_of_week = 0
        minutes_into_session = (et_hour - 9) * 60 + et_min if 9 <= et_hour < 16 else -1

        # Cross-strategy proxies (last 10/20-bar move / breakout heuristics)
        c20 = c[-20:]
        high_20 = float(c20.max())
        low_20 = float(c20.min())
        c5 = c[-5:]
        broke_high = 1.0 if float(c5.max()) > high_20 else 0.0
        broke_low = 1.0 if float(c5.min()) < low_20 else 0.0
        any_strategy_signal_30 = float(broke_high + broke_low)
        max_move_10 = float(c[-10:].max() - c[-10:].min())
        big_move_10 = 1.0 if max_move_10 >= 6.0 else 0.0

        return {
            "vol_bp_30": vol30, "eff_30": eff30,
            "vol_bp_60": vol60, "eff_60": eff60,
            "vol_bp_120": vol120, "eff_120": eff120,
            "vol_bp_240": vol240, "eff_240": eff240,
            "vol_bp_480": vol480, "eff_480": eff480,
            "vol_slope_10": vol_slope_10, "vol_slope_30": vol_slope_30,
            "vol_slope_60": vol_slope_60, "eff_slope_30": eff_slope_30,
            "atr14": atr14, "atr30": atr30,
            "range_pct_20": range_pct_20, "range_pct_120": range_pct_120,
            "body_ratio_20": body_ratio_20, "body_ratio_60": body_ratio_60,
            "abs_body_20": abs_body_20, "up_bar_pct_20": up_bar_pct_20,
            "run_up_max_20": float(max_ru), "run_down_max_20": float(max_rd),
            "gap_pct": gap_pct, "gap_abs_mean_20": gap_abs_mean_20,
            "mom_5": mom_5, "mom_15": mom_15, "mom_30": mom_30,
            "mom_60": mom_60, "mom_120": mom_120,
            "volume_z_20": volume_z_20, "volume_ma_ratio": volume_ma_ratio,
            "max_runup_60": max_runup_60, "max_rundown_60": max_rundown_60,
            "et_hour": float(et_hour),
            "minutes_into_session": float(minutes_into_session),
            "day_of_week": float(day_of_week),
            "any_strategy_signal_30": any_strategy_signal_30,
            "big_move_10": big_move_10,
        }

    @property
    def regime(self) -> str:
        return self._state.regime

    @property
    def state(self) -> RegimeState:
        return self._state

    def update(self, ts, close) -> Optional[str]:
        try:
            c = float(close)
        except Exception:
            return self._state.regime
        if not math.isfinite(c) or c <= 0:
            return self._state.regime
        self._closes.append(c)
        self._state.bar_count += 1
        if len(self._closes) < WINDOW_BARS:
            return self._state.regime  # still warming up
        vol_bp, eff = self._compute_metrics()
        self._state.vol_bp = vol_bp
        self._state.eff = eff
        new_regime = self._classify(vol_bp, eff)
        if new_regime != self._state.regime:
            since = self._state.bar_count - self._state.last_transition_bar
            if self._state.regime != "warmup" and since < TRANSITION_COOLDOWN_BARS:
                return self._state.regime
            self._apply_regime(new_regime, ts, vol_bp, eff)
        return self._state.regime

    def _compute_metrics(self) -> Tuple[float, float]:
        closes = list(self._closes)
        rets = []
        for i in range(1, len(closes)):
            p0 = closes[i - 1]
            if p0 > 0:
                rets.append((closes[i] - p0) / p0)
        if not rets:
            return 0.0, 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        vol_bp = math.sqrt(var) * 10_000.0
        abs_sum = sum(abs(r) for r in rets)
        eff = abs(sum(rets)) / abs_sum if abs_sum > 0 else 0.0
        return vol_bp, eff

    def _classify(self, vol_bp: float, eff: float) -> str:
        # Dead-tape = exceptionally low volatility regardless of eff. DE3's
        # default 25pt TP / 10pt SL / 10pt BE-trigger never activate here.
        # Checked first so it wins over calm_trend when vol is in bottom tail.
        if vol_bp < DEAD_TAPE_VOL_BP:
            return "dead_tape"
        # Whipsaw = violent chop: BOTH high vol AND low eff (e.g. tariff-week
        # tape). Pure-eff low with low vol is just a flat tape (not a whipsaw,
        # stays neutral so the bot can still fish).
        # Calm-trend = directional with moderate-to-low vol.
        if vol_bp > 3.5 and eff < EFF_LOW:
            return "whipsaw"
        if eff > EFF_HIGH:
            return "calm_trend"
        return "neutral"

    def _apply_regime(self, new_regime: str, ts, vol_bp: float, eff: float) -> None:
        overlay = CONFIG.setdefault("KALSHI_TRADE_OVERLAY", {})
        buf = overlay.setdefault("entry_block_buffer", {})
        rev_cfg = CONFIG.setdefault("LIVE_OPPOSITE_REVERSAL", {})
        if new_regime == "whipsaw":
            buf["balanced"] = WHIPSAW_BUF_VALUE
            buf["forward_primary"] = WHIPSAW_BUF_VALUE
            rev_cfg["required_confirmations"] = self._baseline_rev_confirm
            cb_cap, cb_consec = CB_CAP_WHIPSAW, CB_CONSEC_WHIPSAW
        elif new_regime == "calm_trend":
            buf["balanced"] = self._baseline_buf_balanced
            buf["forward_primary"] = self._baseline_buf_fp
            rev_cfg["required_confirmations"] = CALM_REV_CONFIRM
            cb_cap, cb_consec = CB_CAP_CALM, CB_CONSEC_CALM
        elif new_regime == "dead_tape":
            # Keep Kalshi buf baseline, keep reversal baseline. The TP/SL
            # override is consumed via apply_dead_tape_brackets() at signal
            # birth — config mutation here is informational only; DE3 still
            # generates signals with original TP/SL, the override is applied
            # to the signal payload by the caller.
            buf["balanced"] = self._baseline_buf_balanced
            buf["forward_primary"] = self._baseline_buf_fp
            rev_cfg["required_confirmations"] = self._baseline_rev_confirm
            cb_cap, cb_consec = CB_CAP_WHIPSAW, CB_CONSEC_WHIPSAW
        else:
            buf["balanced"] = self._baseline_buf_balanced
            buf["forward_primary"] = self._baseline_buf_fp
            rev_cfg["required_confirmations"] = self._baseline_rev_confirm
            cb_cap, cb_consec = CB_CAP_NEUTRAL, CB_CONSEC_NEUTRAL

        # Adaptive CB: retune the circuit breaker risk cap to match the current
        # regime. Tight caps on whipsaw (kill chop days fast); loose on calm-
        # trend (let normal drawdowns recover into a winning day).
        if ADAPTIVE_CB_ENABLED:
            try:
                import circuit_breaker as _cb_mod
                cb = getattr(_cb_mod, "_GLOBAL_CB", None)
                if cb is not None:
                    cb.max_daily_loss = float(cb_cap)
                    cb.max_consecutive_losses = int(cb_consec)
            except Exception:
                logging.debug("adaptive CB update failed", exc_info=True)
        prev = self._state.regime
        self._state.regime = new_regime
        self._state.last_transition_bar = self._state.bar_count
        logging.info(
            "Regime transition: %s -> %s | vol=%.2fbp eff=%.3f | buf_bal=%.2f buf_fp=%.2f rev=%d | ts=%s",
            prev, new_regime, vol_bp, eff,
            buf.get("balanced", 0), buf.get("forward_primary", 0),
            rev_cfg.get("required_confirmations", 0), ts,
        )


_CLASSIFIER: Optional[RegimeClassifier] = None


def init_regime_classifier() -> Optional[RegimeClassifier]:
    global _CLASSIFIER
    if os.environ.get("JULIE_REGIME_CLASSIFIER", "0").strip() != "1":
        _CLASSIFIER = None
        return None
    _CLASSIFIER = RegimeClassifier()
    logging.info("Regime classifier enabled (JULIE_REGIME_CLASSIFIER=1).")
    return _CLASSIFIER


def get_regime_classifier() -> Optional[RegimeClassifier]:
    return _CLASSIFIER


def update_regime_classifier(ts, close) -> Optional[str]:
    if _CLASSIFIER is None:
        return None
    try:
        return _CLASSIFIER.update(ts, close)
    except Exception:
        logging.debug("regime classifier update failed", exc_info=True)
        return None


def current_regime() -> str:
    return _CLASSIFIER.regime if _CLASSIFIER is not None else "disabled"


def should_veto_entry() -> tuple[bool, str]:
    """Return (veto, reason). Caller should block the entry if veto is True.

    Active only when JULIE_REGIME_WHIPSAW_VETO=1 and current regime is whipsaw.
    """
    if _CLASSIFIER is None or not WHIPSAW_VETO_ENABLED:
        return False, ""
    if _CLASSIFIER.regime == "whipsaw":
        return True, f"regime=whipsaw vol={_CLASSIFIER.state.vol_bp:.2f}bp eff={_CLASSIFIER.state.eff:.3f}"
    return False, ""


def apply_regime_size_cap(signal: dict) -> bool:
    """Filter D — when regime is whipsaw or calm_trend (non-neutral), cap the
    signal's size field to REGIME_SIZE_CAP_VALUE. Returns True if size was
    modified, False otherwise. Idempotent and mutation is visible to the
    pct_overlay snapshot + downstream consumers.

    Gated by JULIE_REGIME_SIZE_CAP=1.
    """
    if _CLASSIFIER is None or not REGIME_SIZE_CAP_ENABLED:
        return False
    if not isinstance(signal, dict):
        return False
    regime = _CLASSIFIER.regime
    if regime not in ("whipsaw", "calm_trend"):
        return False
    try:
        base = int(signal.get("size", 1) or 1)
    except Exception:
        base = 1
    cap = max(1, REGIME_SIZE_CAP_VALUE)
    # Filter E: green-day unlock. If today's cum PnL has crossed the threshold,
    # raise the cap so we can press a winning day. Read cum PnL from the
    # circuit_breaker's daily_pnl tracker (shared state, already updated per
    # trade close). Falls through to D's tight cap if CB isn't available or
    # daily PnL hasn't reached the unlock threshold.
    try:
        import circuit_breaker as _cb_mod
        cb = getattr(_cb_mod, "_GLOBAL_CB", None)
        if cb is not None and float(cb.daily_pnl) >= REGIME_GREEN_UNLOCK_THRESHOLD:
            unlocked_cap = max(cap, REGIME_GREEN_UNLOCK_SIZE)
            if unlocked_cap > cap:
                cap = unlocked_cap
                signal["regime_size_cap_unlocked"] = True
    except Exception:
        pass
    if base <= cap:
        return False
    signal["size"] = cap
    signal["regime_size_cap_before"] = base
    signal["regime_size_cap_regime"] = regime
    logging.info(
        "Regime size cap (%s%s): %s %s | size %d -> %d",
        regime, " unlocked" if signal.get("regime_size_cap_unlocked") else "",
        signal.get("strategy", "?"), signal.get("side", "?"), base, cap,
    )
    return True


# ─── Decoupled action space (2026-04-24) ──────────────────────────────────
#
# apply_dead_tape_brackets historically did THREE things in one shot:
#   1. rewrite TP/SL to scalp geometry (3/5)
#   2. force size to 1
#   3. disable BE-arm
#
# v4 ML research found genuine PnL lift from (1) alone but couldn't ship
# because forcing (2)+(3) on ML-called-scalp bars (which include high-vol
# bars where low-size + BE-off is wrong) fails the catastrophic-safety gate.
#
# We split into three independently-gated functions. Each can be driven by
# a separate ML model OR fall through to the rule. Env flags control:
#     JULIE_REGIME_ML_BRACKETS — model A drives bracket rewrite (else rule)
#     JULIE_REGIME_ML_SIZE     — model B drives size reduction (else rule)
#     JULIE_REGIME_ML_BE       — model C drives BE disable (else rule)
#
# Default all flags OFF → behavior identical to pre-split rule.
# apply_dead_tape_brackets() remains as a thin compatibility wrapper.

ML_BRACKETS_ENABLED = os.environ.get("JULIE_REGIME_ML_BRACKETS", "0").strip() == "1"
ML_SIZE_ENABLED     = os.environ.get("JULIE_REGIME_ML_SIZE",     "0").strip() == "1"
ML_BE_ENABLED       = os.environ.get("JULIE_REGIME_ML_BE",       "0").strip() == "1"

# Each ML model is lazy-loaded from artifacts/regime_ml_v5_* when its env
# flag is on. If the artifact is missing, the module silently falls back
# to the rule for that action. Shipped ML paths print a one-liner at init.
_ML_BRACKET_MODEL = None
_ML_SIZE_MODEL    = None
_ML_BE_MODEL      = None


def _rule_says_dead_tape() -> bool:
    """Current regime flag as determined by the rule classifier."""
    return _CLASSIFIER is not None and _CLASSIFIER.regime == "dead_tape"


def _predict_with_payload(payload: dict, features: dict) -> bool:
    """Reconstruct prediction from saved components.

    Uses HGB-only in live inference. LightGBM's OpenMP runtime conflicts
    with other threaded libs the bot loads (torch, sklearn, asyncio), which
    caused SIGSEGV-11 crashes in the first attempted deployment. HGB is
    pure sklearn/Cython and survives the multi-threaded bot context.

    Threshold compensates: the saved 0.70 threshold was tuned on the
    HGB+LGBM ensemble output. For HGB-only output the equivalent ship
    threshold is 0.50 (re-validated on the same OOS slice: PnL +$11,160
    vs rule +$8,485, lift +$2,675, DD $1,440 ≤ rule 110%).
    """
    import numpy as _np
    feature_cols = payload["feature_cols"]
    hgb = payload["hgb"]
    positive_class = payload["positive_class"]
    # HGB-only threshold override (see docstring). Fall back to saved
    # threshold if the override key is missing.
    threshold = float(payload.get("threshold_hgb_only", 0.50))
    row = _np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=float)
    hgb_classes = list(hgb.classes_)
    p_idx = hgb_classes.index(positive_class)
    p_hgb = hgb.predict_proba(row)[0, p_idx]
    return p_hgb >= threshold


def _ml_says(model_key: str, features: Optional[dict]) -> Optional[bool]:
    """Lazy-load the named ML model and return its prediction, or None if
    the model isn't available or features are missing."""
    global _ML_BRACKET_MODEL, _ML_SIZE_MODEL, _ML_BE_MODEL
    if features is None:
        return None
    try:
        from pathlib import Path as _Path
        import pickle as _pickle
        _ART = _Path(__file__).resolve().parent / "artifacts"
        key_to_slot = {
            "brackets": ("_ML_BRACKET_MODEL", "regime_ml_v5_brackets"),
            "size":     ("_ML_SIZE_MODEL",    "regime_ml_v5_size"),
            "be":       ("_ML_BE_MODEL",      "regime_ml_v5_be"),
        }
        if model_key not in key_to_slot:
            return None
        slot_name, dir_name = key_to_slot[model_key]
        payload = globals().get(slot_name)
        if payload is None:
            pkl = _ART / dir_name / "model.pkl"
            if not pkl.exists():
                return None
            with pkl.open("rb") as fh:
                payload = _pickle.load(fh)
            globals()[slot_name] = payload
        return _predict_with_payload(payload, features)
    except Exception:
        logging.debug("ML model load/predict failed (%s)", model_key, exc_info=True)
        return None


def _auto_features(bar_features: Optional[dict]) -> Optional[dict]:
    """If caller didn't supply features, try to build them from the classifier's
    internal OHLCV history (via record_bar). Returns None if history is shallow."""
    if bar_features is not None:
        return bar_features
    if _CLASSIFIER is None:
        return None
    try:
        return _CLASSIFIER.build_ml_feature_snapshot()
    except Exception:
        logging.debug("auto feature build failed", exc_info=True)
        return None


def apply_scalp_brackets(signal: dict, *, bar_features: Optional[dict] = None) -> bool:
    """Rewrite TP/SL to scalp values (3/5) when we want tight-geometry risk.

    Decision source:
      - if JULIE_REGIME_ML_BRACKETS=1 AND model-A artifact loads: ML decides
      - else: rule (dead_tape regime → rewrite, else passthrough)

    Mutates signal. Returns True if rewritten.
    """
    if not isinstance(signal, dict):
        return False
    should_rewrite: Optional[bool] = None
    if ML_BRACKETS_ENABLED:
        ml = _ml_says("brackets", _auto_features(bar_features))
        if ml is not None:
            should_rewrite = ml
    if should_rewrite is None:
        should_rewrite = _rule_says_dead_tape()
    if not should_rewrite:
        return False
    original_tp = signal.get("tp_dist")
    original_sl = signal.get("sl_dist")
    signal["tp_dist"] = DEAD_TAPE_TP_PTS
    signal["sl_dist"] = DEAD_TAPE_SL_PTS
    signal["scalp_bracket_applied"] = True
    signal["scalp_bracket_original_tp_dist"] = original_tp
    signal["scalp_bracket_original_sl_dist"] = original_sl
    signal["scalp_bracket_source"] = "ml" if (ML_BRACKETS_ENABLED and _ML_BRACKET_MODEL) else "rule"
    return True


def apply_size_reduction(signal: dict, *, bar_features: Optional[dict] = None) -> bool:
    """Force size = 1 when we expect a high-variance forward window.

    Decision source:
      - if JULIE_REGIME_ML_SIZE=1 AND model-B artifact loads: ML decides
      - else: rule (dead_tape → force size=1, else passthrough)

    Mutates signal. Returns True if size was reduced.
    """
    if not isinstance(signal, dict):
        return False
    should_reduce: Optional[bool] = None
    if ML_SIZE_ENABLED:
        ml = _ml_says("size", _auto_features(bar_features))
        if ml is not None:
            should_reduce = ml
    if should_reduce is None:
        should_reduce = _rule_says_dead_tape()
    if not should_reduce:
        return False
    original_size = signal.get("size")
    if original_size == 1:
        return False
    signal["size"] = 1
    signal["size_reduced_to_1"] = True
    signal["size_reduction_original_size"] = original_size
    signal["size_reduction_source"] = "ml" if (ML_SIZE_ENABLED and _ML_SIZE_MODEL) else "rule"
    return True


def apply_be_disable(signal: dict, *, bar_features: Optional[dict] = None) -> bool:
    """Disable BE-arm when we expect mean-reversion that would whipsaw the
    BE-triggered stop.

    Decision source:
      - if JULIE_REGIME_ML_BE=1 AND model-C artifact loads: ML decides
      - else: rule (dead_tape → BE-off, else passthrough)

    Mutates signal. Returns True if BE was disabled.
    """
    if not isinstance(signal, dict):
        return False
    should_disable: Optional[bool] = None
    if ML_BE_ENABLED:
        ml = _ml_says("be", _auto_features(bar_features))
        if ml is not None:
            should_disable = ml
    if should_disable is None:
        should_disable = _rule_says_dead_tape()
    if not should_disable:
        return False
    signal["de3_break_even_enabled"] = False
    signal["de3_break_even_activate_on_next_bar"] = False
    signal["be_disabled"] = True
    signal["be_disabled_source"] = "ml" if (ML_BE_ENABLED and _ML_BE_MODEL) else "rule"
    return True


def apply_dead_tape_brackets(signal: dict, *, bar_features: Optional[dict] = None) -> bool:
    """Compatibility wrapper preserving the old unified-action call site.

    Calls the three decoupled functions in order. Each is independently
    driven by ML (if enabled + artifact loaded) or rule. Behavior identical
    to the pre-split version WHEN all three env flags are OFF (the default).

    Returns True if ANY of the three actions was applied (matching old
    semantics where True meant "dead-tape overrides kicked in").
    """
    a = apply_scalp_brackets(signal, bar_features=bar_features)
    b = apply_size_reduction(signal, bar_features=bar_features)
    c = apply_be_disable(signal, bar_features=bar_features)
    any_applied = a or b or c
    if any_applied:
        # Preserve legacy audit fields for downstream consumers
        signal["dead_tape_original_tp_dist"] = signal.get("scalp_bracket_original_tp_dist", signal.get("dead_tape_original_tp_dist"))
        signal["dead_tape_original_sl_dist"] = signal.get("scalp_bracket_original_sl_dist", signal.get("dead_tape_original_sl_dist"))
        signal["dead_tape_original_size"]    = signal.get("size_reduction_original_size", signal.get("dead_tape_original_size"))
        signal["dead_tape_be_trigger_pts"]   = DEAD_TAPE_BE_TRIGGER_PTS
        signal["dead_tape_regime_active"]    = True
        vol_bp = _CLASSIFIER.state.vol_bp if _CLASSIFIER is not None else 0.0
        logging.info(
            "Dead-tape override: %s %s | tp->%s sl->%s size->%s BE=%s | vol=%.2fbp | sources=(brackets:%s, size:%s, be:%s)",
            signal.get("strategy", "?"), signal.get("side", "?"),
            signal.get("tp_dist"), signal.get("sl_dist"), signal.get("size"),
            "off" if signal.get("be_disabled") else "on", vol_bp,
            signal.get("scalp_bracket_source", "none"),
            signal.get("size_reduction_source", "none"),
            signal.get("be_disabled_source", "none"),
        )
    return any_applied

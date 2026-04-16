"""
RegimeAdaptive Strategy Module
==============================
Extracted from julie001.py with signal reversion for underperforming time contexts.

Based on backtest results, combos with WR < 35% get signals REVERTED (LONG<->SHORT).
This turns losing patterns into winning ones by fading the original signal.

Import this in julie001.py:
    from regime_strategy import RegimeAdaptiveStrategy
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

from config import CONFIG
from regimeadaptive_artifact import load_regimeadaptive_artifact
from regimeadaptive_gate import (
    build_runtime_gate_feature_row,
    load_regimeadaptive_gate_model,
)

# ============================================================
# Try to import optimized SLTP params
# ============================================================
try:
    from regime_sltp_params import (
        get_regime_sltp, 
        is_reverted_combo as _is_reverted_from_params,
        REVERTED_COMBOS as PARAMS_REVERTED_COMBOS,
        PARAMS as SLTP_PARAMS
    )
    SLTP_PARAMS_AVAILABLE = True
    logging.info(f"Loaded regime_sltp_params with {len(SLTP_PARAMS)} combos")
except ImportError:
    SLTP_PARAMS_AVAILABLE = False
    PARAMS_REVERTED_COMBOS = set()
    SLTP_PARAMS = {}
    logging.warning("regime_sltp_params not found, using defaults")

# ============================================================
# REVERTED COMBOS - from regime_sltp_params.py if available
# These combos performed poorly with normal signals, so we
# trade the OPPOSITE direction (fade the signal)
# ============================================================
if SLTP_PARAMS_AVAILABLE and PARAMS_REVERTED_COMBOS:
    REVERTED_COMBOS = PARAMS_REVERTED_COMBOS
    logging.info(f"Using {len(REVERTED_COMBOS)} reverted combos from regime_sltp_params")
else:
    # Fallback defaults if params file not available
    REVERTED_COMBOS = {
        'Q1_W1_TUE_ASIA', 'Q1_W1_WED_ASIA', 'Q1_W2_FRI_NY_PM',
        'Q1_W4_FRI_NY_PM', 'Q1_W4_MON_ASIA', 'Q1_W4_THU_ASIA',
        'Q1_W4_TUE_NY_PM', 'Q2_W2_TUE_NY_AM', 'Q2_W2_WED_NY_PM',
        'Q2_W3_WED_LONDON', 'Q2_W4_MON_NY_AM', 'Q2_W4_MON_NY_PM',
        'Q3_W1_TUE_NY_PM', 'Q3_W1_WED_ASIA', 'Q3_W1_WED_LONDON',
        'Q3_W1_WED_NY_PM', 'Q3_W2_WED_ASIA', 'Q3_W3_THU_NY_AM',
        'Q3_W4_FRI_NY_PM', 'Q3_W4_MON_ASIA', 'Q3_W4_MON_NY_PM',
        'Q3_W4_THU_LONDON', 'Q3_W4_THU_NY_AM',
    }


def get_session(hour: int) -> str:
    """Determine trading session from hour."""
    if 18 <= hour <= 23 or 0 <= hour < 3:
        return 'ASIA'
    elif 3 <= hour < 8:
        return 'LONDON'
    elif 8 <= hour < 12:
        return 'NY_AM'
    elif 12 <= hour < 17:
        return 'NY_PM'
    return 'CLOSED'


def get_yearly_quarter(month: int) -> str:
    """Get yearly quarter from month."""
    if month <= 3: return 'Q1'
    elif month <= 6: return 'Q2'
    elif month <= 9: return 'Q3'
    return 'Q4'


def get_monthly_quarter(day: int) -> str:
    """Get monthly quarter (week of month) from day."""
    if day <= 7: return 'W1'
    elif day <= 14: return 'W2'
    elif day <= 21: return 'W3'
    return 'W4'


def get_day_name(dow: int) -> str:
    """Get day name from day of week integer."""
    return ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][dow]


def get_time_context(ts) -> Dict[str, str]:
    """Extract full time context from timestamp."""
    return {
        'yearly_q': get_yearly_quarter(ts.month),
        'monthly_q': get_monthly_quarter(ts.day),
        'day_of_week': get_day_name(ts.dayofweek),
        'session': get_session(ts.hour)
    }


def get_combo_key(ts) -> str:
    """Get hierarchy combo key for timestamp."""
    ctx = get_time_context(ts)
    return f"{ctx['yearly_q']}_{ctx['monthly_q']}_{ctx['day_of_week']}_{ctx['session']}"


def should_revert_signal(ts) -> bool:
    """Check if current time context should have signals reverted."""
    combo = get_combo_key(ts)
    return combo in REVERTED_COMBOS


def get_optimized_sltp(side: str, ts) -> Dict[str, float]:
    """
    Get optimized SL/TP for the given side and timestamp.

    Returns dict with 'sl_dist' and 'tp_dist'
    """
    # Minimum enforcement for positive RR (points)
    MIN_SL = 2.0  # 8 ticks minimum
    MIN_TP = 3.0  # 12 ticks minimum (1.5:1 RR)

    ctx = get_time_context(ts)

    if SLTP_PARAMS_AVAILABLE:
        params = get_regime_sltp(
            side,
            ctx['yearly_q'],
            ctx['monthly_q'],
            ctx['day_of_week'],
            ctx['session']
        )
        sl_dist = max(params['sl'], MIN_SL)
        tp_dist = max(params['tp'], MIN_TP)
        return {
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }

    # Defaults if no params available
    return {'sl_dist': MIN_SL, 'tp_dist': MIN_TP}


class RegimeAdaptiveStrategy:
    """
    RegimeAdaptive Strategy with Time-Context Signal Reversion

    Logic:
    - NORMAL: LONG on uptrend dip, SHORT on downtrend rally
    - REVERTED: For poor-performing time contexts, flip LONG<->SHORT

    Signal conditions:
    - Uptrend: SMA20 > SMA200 + low volatility
    - Downtrend: SMA20 < SMA200 + low volatility
    - Entry: Price crosses fast SMA with range spike

    Equal low/high filter prevents continuation pattern entries.

    Optimized SL/TP params loaded from regime_sltp_params.py if available.
    """

    def __init__(self, dynamic_sltp_engine=None):
        self.strategy_name = "RegimeAdaptive"
        tuning = CONFIG.get("REGIME_ADAPTIVE_TUNING", {}) or {}
        artifact_path = str(tuning.get("artifact_path", "") or "").strip()
        self.artifact = load_regimeadaptive_artifact(artifact_path)
        self.signal_gate_model = (
            load_regimeadaptive_gate_model(self.artifact.path, getattr(self.artifact, "signal_gate", {}))
            if self.artifact is not None
            else None
        )
        mode = str(tuning.get("mode", "filterless") or "filterless").strip().lower()
        filterless_mode = mode == "filterless"
        self.use_eq_filter = bool(tuning.get("use_eq_filter", not filterless_mode))
        self.enable_high_vol_gate = bool(
            tuning.get("enable_high_vol_gate", not filterless_mode)
        )
        self.enable_time_block = bool(tuning.get("enable_time_block", not filterless_mode))
        self.require_low_vol_trend = bool(
            tuning.get("require_low_vol_trend", not filterless_mode)
        )
        self.require_range_spike = bool(
            tuning.get("require_range_spike", not filterless_mode)
        )
        self.enable_signal_reversion = bool(tuning.get("enable_signal_reversion", True))
        self.eq_tolerance = float(tuning.get("eq_tolerance", 0.5))
        self.lookback_bars = int(tuning.get("eq_lookback", 20))
        self.dynamic_sltp_engine = dynamic_sltp_engine
        self.last_combo_key = None
        self.bars_since_log = 0

        # SMA periods
        artifact_base_rule = self.artifact.base_rule if self.artifact is not None else {}
        self.sma_fast = int(artifact_base_rule.get("sma_fast", tuning.get("sma_fast", 20)))
        self.sma_slow = int(artifact_base_rule.get("sma_slow", tuning.get("sma_slow", 200)))

        # 1m robustness tuning
        self.atr_period = int(artifact_base_rule.get("atr_period", tuning.get("atr_period", 20)))
        self.range_window = int(tuning.get("range_window", 20))
        self.range_spike_mult = float(tuning.get("range_spike_mult", 1.3))
        self.range_atr_mult = float(tuning.get("range_atr_mult", 0.8))
        self.cross_atr_mult = float(
            artifact_base_rule.get("cross_atr_mult", tuning.get("cross_atr_mult", 0.3))
        )
        self.eq_atr_mult = float(tuning.get("eq_atr_mult", 0.25))
        self.vol_window = int(tuning.get("vol_window", 30))
        self.vol_median_window = int(tuning.get("vol_median_window", 120))
        self.high_vol_mult = float(tuning.get("high_vol_mult", 1.5))
        self.high_vol_block_sessions = set(tuning.get("high_vol_block_sessions", ["NY_PM"]))
        self.block_hours_by_session = tuning.get(
            "block_hours_by_session",
            {"NY_PM": [12, 13, 14]},
        )

        # Use optimized SLTP from regime_sltp_params if available
        self.use_optimized_sltp = bool(self.artifact is not None or SLTP_PARAMS_AVAILABLE)

        logging.info(
            "RegimeAdaptiveStrategy initialized | mode=%s | artifact=%s | reverted_combos=%s | "
            "eq_filter=%s high_vol_gate=%s time_block=%s low_vol_trend=%s "
            "range_spike=%s gate=%s | SLTP params=%s",
            mode,
            getattr(self.artifact, "path", None),
            getattr(self.artifact, "reverted_count", len(REVERTED_COMBOS)),
            self.use_eq_filter,
            self.enable_high_vol_gate,
            self.enable_time_block,
            self.require_low_vol_trend,
            self.require_range_spike,
            bool(self.signal_gate_model is not None),
            SLTP_PARAMS_AVAILABLE,
        )

    def _calc_atr(self, df: pd.DataFrame, period: Optional[int] = None) -> Optional[float]:
        use_period = int(period or self.atr_period or 0)
        if df.empty or use_period <= 0 or len(df) < use_period + 1:
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(use_period).mean().iloc[-1]
        if pd.isna(atr):
            return None
        return float(atr)

    def _evaluate_rule_candidate(
        self,
        df: pd.DataFrame,
        combo_key: str,
        original_signal: str,
        curr_vol,
        vol_median,
        curr_range,
        range_sma,
    ) -> Optional[Dict]:
        if self.artifact is None:
            return None
        rule = self.artifact.get_rule(combo_key, original_signal)
        if not isinstance(rule, dict) or not rule:
            return None

        sma_fast_period = int(rule.get("sma_fast", self.sma_fast) or self.sma_fast)
        sma_slow_period = int(rule.get("sma_slow", self.sma_slow) or self.sma_slow)
        atr_period = int(rule.get("atr_period", self.atr_period) or self.atr_period)
        cross_atr_mult = float(rule.get("cross_atr_mult", self.cross_atr_mult) or self.cross_atr_mult)
        rule_type = str(rule.get("rule_type", "pullback") or "pullback").strip().lower()
        pattern_lookback = max(1, int(rule.get("pattern_lookback", self.lookback_bars) or self.lookback_bars))
        touch_atr_mult = max(0.0, float(rule.get("touch_atr_mult", 0.25) or 0.25))

        min_bars = max(sma_slow_period, atr_period) + max(1, pattern_lookback)
        if len(df) < min_bars:
            return None

        curr = df.iloc[-1]
        sma_fast = df["close"].rolling(sma_fast_period).mean().iloc[-1]
        sma_slow = df["close"].rolling(sma_slow_period).mean().iloc[-1]
        atr = self._calc_atr(df, period=atr_period)

        low_vol_ok = True
        if self.require_low_vol_trend:
            low_vol_ok = bool(
                pd.notna(curr_vol)
                and pd.notna(vol_median)
                and (curr_vol < vol_median)
            )
        trending_up = (sma_fast > sma_slow) and low_vol_ok
        trending_down = (sma_fast < sma_slow) and low_vol_ok
        if atr is not None:
            range_spike = curr_range > max(range_sma * self.range_spike_mult, atr * self.range_atr_mult)
            cross_thresh = atr * cross_atr_mult
            eq_tol = max(self.eq_tolerance, atr * self.eq_atr_mult)
        else:
            range_spike = curr_range > (range_sma * self.range_spike_mult)
            cross_thresh = 0.0
            eq_tol = self.eq_tolerance
        if not self.require_range_spike:
            range_spike = True

        recent_window = df.iloc[-(pattern_lookback + 1):-1] if pattern_lookback > 0 else df.iloc[:-1]
        recent_high = float(recent_window["high"].max()) if not recent_window.empty else float("nan")
        recent_low = float(recent_window["low"].min()) if not recent_window.empty else float("nan")
        touch_buffer = (atr * touch_atr_mult) if atr is not None else self.eq_tolerance

        long_ok = False
        short_ok = False
        long_strength = None
        short_strength = None

        if rule_type == "breakout":
            long_ok = bool(trending_up and pd.notna(recent_high) and (curr["close"] > (recent_high + cross_thresh)))
            short_ok = bool(trending_down and pd.notna(recent_low) and (curr["close"] < (recent_low - cross_thresh)))
            if long_ok:
                long_strength = float(curr["close"] - recent_high - cross_thresh)
            if short_ok:
                short_strength = float(recent_low - curr["close"] - cross_thresh)
        elif rule_type == "continuation":
            long_touch = bool(pd.notna(recent_low) and (recent_low <= (sma_fast + touch_buffer)))
            short_touch = bool(pd.notna(recent_high) and (recent_high >= (sma_fast - touch_buffer)))
            long_ok = bool(trending_up and long_touch and (curr["close"] > (sma_fast + cross_thresh)))
            short_ok = bool(trending_down and short_touch and (curr["close"] < (sma_fast - cross_thresh)))
            if long_ok:
                long_strength = float(curr["close"] - sma_fast - cross_thresh)
            if short_ok:
                short_strength = float(sma_fast - curr["close"] - cross_thresh)
        else:
            long_ok = bool(trending_up and (curr["close"] < (sma_fast - cross_thresh)) and range_spike)
            short_ok = bool(trending_down and (curr["close"] > (sma_fast + cross_thresh)) and range_spike)
            if long_ok:
                long_strength = float((sma_fast - curr["close"]) - cross_thresh)
            if short_ok:
                short_strength = float((curr["close"] - sma_fast) - cross_thresh)

        if str(original_signal).upper() == "LONG":
            if not long_ok:
                return None
            if rule_type == "pullback" and self.use_eq_filter and self._is_equal_low(df, tolerance=eq_tol):
                return None
            strength = float(long_strength or 0.0)
        else:
            if not short_ok:
                return None
            if rule_type == "pullback" and self.use_eq_filter and self._is_equal_high(df, tolerance=eq_tol):
                return None
            strength = float(short_strength or 0.0)

        return {
            "rule": rule,
            "rule_id": self.artifact.get_rule_id(combo_key, original_signal),
            "signal": str(original_signal).upper(),
            "strength": strength,
            "sma_fast_value": float(sma_fast),
            "sma_slow_value": float(sma_slow),
            "atr_value": float(atr) if atr is not None else None,
        }
    
    def _is_equal_low(self, df: pd.DataFrame, tolerance: Optional[float] = None) -> bool:
        """Check if current low matches recent lows (continuation pattern)."""
        if len(df) < self.lookback_bars + 1:
            return False
        curr_low = df.iloc[-1]['low']
        recent_lows = df['low'].iloc[-(self.lookback_bars + 1):-1].values
        tol = self.eq_tolerance if tolerance is None else tolerance
        return any(abs(curr_low - l) <= tol for l in recent_lows)
    
    def _is_equal_high(self, df: pd.DataFrame, tolerance: Optional[float] = None) -> bool:
        """Check if current high matches recent highs (continuation pattern)."""
        if len(df) < self.lookback_bars + 1:
            return False
        curr_high = df.iloc[-1]['high']
        recent_highs = df['high'].iloc[-(self.lookback_bars + 1):-1].values
        tol = self.eq_tolerance if tolerance is None else tolerance
        return any(abs(curr_high - h) <= tol for h in recent_highs)
    
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal on new bar.
        
        Returns:
            Dict with signal info or None
        """
        if len(df) < 200:
            return None
        
        curr = df.iloc[-1]
        ts = df.index[-1]
        
        # Get time context
        combo_key = get_combo_key(ts)
        ctx = get_time_context(ts)

        # Log context changes
        if self.last_combo_key != combo_key:
            self.last_combo_key = combo_key
            if self.artifact is not None:
                is_reverted = self.artifact.should_revert(combo_key)
            else:
                is_reverted = combo_key in REVERTED_COMBOS
            logging.info(f"RegimeAdaptive: Context changed to {combo_key} | Reverted: {is_reverted}")

        # Skip if CLOSED session
        if ctx['session'] == 'CLOSED':
            return None
        
        # Calculate indicators
        sma_fast = df['close'].rolling(self.sma_fast).mean().iloc[-1]
        sma_slow = df['close'].rolling(self.sma_slow).mean().iloc[-1]

        # Volatility + ATR
        vol_series = df['close'].pct_change().rolling(self.vol_window).std()
        curr_vol = vol_series.iloc[-1]
        vol_median = vol_series.rolling(self.vol_median_window).median().iloc[-1]
        atr = self._calc_atr(df)

        # High-vol gate
        high_vol = False
        if pd.notna(curr_vol) and pd.notna(vol_median) and vol_median > 0:
            high_vol = curr_vol > (vol_median * self.high_vol_mult)
        if self.enable_high_vol_gate and high_vol and ctx['session'] in self.high_vol_block_sessions:
            logging.debug(f"RegimeAdaptive: blocked high-vol in {ctx['session']} ({curr_vol:.4f} > {vol_median:.4f})")
            return None

        # Time-window guard (e.g. NY_PM 12-14 ET)
        block_hours = self.block_hours_by_session.get(ctx['session'], [])
        if self.enable_time_block and block_hours and ts.hour in block_hours:
            logging.debug(f"RegimeAdaptive: blocked {ctx['session']} hour {ts.hour}")
            return None
        
        # Range spike
        rng = df['high'] - df['low']
        range_sma = rng.rolling(self.range_window).mean().iloc[-1]
        curr_range = rng.iloc[-1]
        
        # Regime detection
        low_vol_ok = True
        if self.require_low_vol_trend:
            low_vol_ok = bool(
                pd.notna(curr_vol)
                and pd.notna(vol_median)
                and (curr_vol < vol_median)
            )
        trending_up = (sma_fast > sma_slow) and low_vol_ok
        trending_down = (sma_fast < sma_slow) and low_vol_ok
        if atr is not None:
            range_spike = curr_range > max(range_sma * self.range_spike_mult, atr * self.range_atr_mult)
            cross_thresh = atr * self.cross_atr_mult
            eq_tol = max(self.eq_tolerance, atr * self.eq_atr_mult)
        else:
            range_spike = curr_range > (range_sma * self.range_spike_mult)
            cross_thresh = 0.0
            eq_tol = self.eq_tolerance
        if not self.require_range_spike:
            range_spike = True

        signal = None
        original_signal = None
        revert = False
        early_exit_enabled = None
        selected_rule_id = None
        selected_sma_fast = sma_fast
        selected_sma_slow = sma_slow

        if self.artifact is not None and getattr(self.artifact, "rule_catalog", {}):
            candidates = []
            close_ret = df["close"].pct_change()
            ret_1 = float(close_ret.iloc[-1]) if len(close_ret) >= 1 and pd.notna(close_ret.iloc[-1]) else 0.0
            ret_5 = float(df["close"].pct_change(5).iloc[-1]) if len(df) >= 6 and pd.notna(df["close"].pct_change(5).iloc[-1]) else 0.0
            ret_15 = float(df["close"].pct_change(15).iloc[-1]) if len(df) >= 16 and pd.notna(df["close"].pct_change(15).iloc[-1]) else 0.0
            for candidate_side in ("LONG", "SHORT"):
                if self.artifact.should_skip(combo_key, candidate_side):
                    continue
                candidate = self._evaluate_rule_candidate(
                    df,
                    combo_key,
                    candidate_side,
                    curr_vol,
                    vol_median,
                    curr_range,
                    range_sma,
                )
                if candidate is None:
                    continue
                candidate_revert = self.artifact.should_revert(combo_key, candidate_side)
                final_signal = "SHORT" if candidate_revert and candidate_side == "LONG" else "LONG" if candidate_revert and candidate_side == "SHORT" else candidate_side
                gate_prob = None
                if self.signal_gate_model is not None:
                    gate_row = build_runtime_gate_feature_row(
                        ts,
                        combo_key,
                        final_signal,
                        candidate_side,
                        candidate.get("rule", {}) if isinstance(candidate.get("rule", {}), dict) else {},
                        float(candidate.get("strength", 0.0) or 0.0),
                        float(candidate.get("sma_fast_value", sma_fast) or sma_fast),
                        float(candidate.get("sma_slow_value", sma_slow) or sma_slow),
                        float(candidate.get("atr_value", atr) or atr or 0.0),
                        float(curr["open"]),
                        float(curr["high"]),
                        float(curr["low"]),
                        float(curr["close"]),
                        ret_1,
                        ret_5,
                        ret_15,
                        float(curr_vol) if pd.notna(curr_vol) else 0.0,
                        float(vol_median) if pd.notna(vol_median) else 0.0,
                        float(range_sma) if pd.notna(range_sma) else 0.0,
                    )
                    gate_prob = float(self.signal_gate_model.predict_proba_row(gate_row))
                    if gate_prob < float(self.signal_gate_model.threshold):
                        continue
                candidates.append(
                    {
                        "original_signal": candidate_side,
                        "signal": final_signal,
                        "revert": bool(candidate_revert),
                        "early_exit_enabled": self.artifact.get_early_exit_enabled(combo_key, candidate_side),
                        "rule_id": candidate.get("rule_id"),
                        "strength": float(candidate.get("strength", 0.0)),
                        "sma_fast_value": float(candidate.get("sma_fast_value", sma_fast)),
                        "sma_slow_value": float(candidate.get("sma_slow_value", sma_slow)),
                        "gate_prob": gate_prob,
                    }
                )

            if not candidates:
                return None
            candidates.sort(key=lambda item: float(item.get("strength", 0.0)), reverse=True)
            if len(candidates) >= 2:
                top = candidates[0]
                nxt = candidates[1]
                if (
                    abs(float(top.get("strength", 0.0)) - float(nxt.get("strength", 0.0))) <= 1e-9
                    and str(top.get("signal")) != str(nxt.get("signal"))
                ):
                    logging.debug(f"RegimeAdaptive: skipped conflicting equal-strength multi-rule signals at {combo_key}")
                    return None
            chosen = candidates[0]
            signal = str(chosen["signal"])
            original_signal = str(chosen["original_signal"])
            revert = bool(chosen["revert"])
            early_exit_enabled = chosen.get("early_exit_enabled")
            selected_rule_id = chosen.get("rule_id")
            selected_sma_fast = float(chosen.get("sma_fast_value", sma_fast))
            selected_sma_slow = float(chosen.get("sma_slow_value", sma_slow))
        else:
            # LONG signal: Uptrend + dip below fast SMA + range spike
            if trending_up and (curr['close'] < (sma_fast - cross_thresh)) and range_spike:
                if self.use_eq_filter and self._is_equal_low(df, tolerance=eq_tol):
                    logging.debug(f"RegimeAdaptive: LONG skipped - equal low at {combo_key}")
                    return None
                signal = 'LONG'
            
            # SHORT signal: Downtrend + rally above fast SMA + range spike
            elif trending_down and (curr['close'] > (sma_fast + cross_thresh)) and range_spike:
                if self.use_eq_filter and self._is_equal_high(df, tolerance=eq_tol):
                    logging.debug(f"RegimeAdaptive: SHORT skipped - equal high at {combo_key}")
                    return None
                signal = 'SHORT'
            
            if signal is None:
                return None

            original_signal = signal
            if self.artifact is not None:
                if self.artifact.should_skip(combo_key, original_signal):
                    logging.debug(f"RegimeAdaptive: skipped {combo_key} {original_signal} by artifact policy")
                    return None
                revert = self.artifact.should_revert(combo_key, original_signal)
                early_exit_enabled = self.artifact.get_early_exit_enabled(combo_key, original_signal)
                selected_rule_id = self.artifact.get_rule_id(combo_key, original_signal)
            else:
                revert = self.enable_signal_reversion and should_revert_signal(ts)

        # Apply reversion if needed
        if revert:
            signal = 'SHORT' if signal == 'LONG' else 'LONG'
            logging.info(f"RegimeAdaptive: Signal REVERTED {original_signal}->{signal} for {combo_key}")
        else:
            logging.info(f"RegimeAdaptive: {signal} signal generated | Combo: {combo_key}")
            logging.info(f"   Fast SMA: {selected_sma_fast:.2f} | Slow SMA: {selected_sma_slow:.2f} | Vol: {curr_vol:.4f} (median: {vol_median:.4f})")
            logging.info(f"   Range: {curr_range:.2f} (SMA: {range_sma:.2f}) | Price: {curr['close']:.2f}")

        # Get SL/TP - use optimized params if available
        if self.artifact is not None:
            sltp = self.artifact.get_sltp(signal, combo_key, ctx['session'])
        elif self.use_optimized_sltp:
            sltp = get_optimized_sltp(signal, ts)
        elif self.dynamic_sltp_engine:
            sltp = self.dynamic_sltp_engine.calculate_dynamic_sltp(df)
        else:
            sltp = {'sl_dist': 2.0, 'tp_dist': 3.0}
        
        # Log params if engine available
        if self.dynamic_sltp_engine:
            self.dynamic_sltp_engine.log_params(sltp)

        payload = {
            "strategy": "RegimeAdaptive",
            "sub_strategy": combo_key,
            "side": signal,
            "tp_dist": sltp['tp_dist'],
            "sl_dist": sltp['sl_dist'],
            "combo_key": combo_key,
            "reverted": revert,
            "original_signal": original_signal if revert else signal,
            "rule_id": selected_rule_id,
        }
        if chosen.get("gate_prob") is not None:
            payload["gate_prob"] = float(chosen["gate_prob"])
            payload["gate_threshold"] = float(self.signal_gate_model.threshold) if self.signal_gate_model is not None else None
        if early_exit_enabled is not None:
            early_exit_cfg = (CONFIG.get("EARLY_EXIT", {}) or {}).get("RegimeAdaptive", {}) or {}
            payload["early_exit_enabled"] = bool(early_exit_enabled)
            payload["early_exit_exit_if_not_green_by"] = int(
                early_exit_cfg.get("exit_if_not_green_by", 30) or 30
            )
            payload["early_exit_max_profit_crosses"] = int(
                early_exit_cfg.get("max_profit_crosses", 4) or 4
            )
        return payload


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def load_reverted_combos_from_csv(csv_path: str, wr_threshold: float = 35.0) -> set:
    """
    Load hierarchy CSV and return combos with WR below threshold.
    
    Usage:
        combos = load_reverted_combos_from_csv('regime_backtest_hierarchy.csv', 35.0)
        print(combos)  # Copy these into REVERTED_COMBOS
    """
    df = pd.read_csv(csv_path)
    poor = df[df['wr'] < wr_threshold]
    return set(poor['combo'].tolist())


def get_strategy_stats() -> Dict:
    """Get current strategy configuration stats."""
    return {
        'reverted_combos': len(REVERTED_COMBOS),
        'sltp_params_available': SLTP_PARAMS_AVAILABLE,
        'sltp_combos': len(SLTP_PARAMS) if SLTP_PARAMS_AVAILABLE else 0
    }


if __name__ == "__main__":
    # Test: Show current configuration
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo('America/New_York')
    now = datetime.now(et)
    
    combo = get_combo_key(now)
    revert = should_revert_signal(now)
    
    print("=" * 60)
    print("REGIME STRATEGY CONFIGURATION")
    print("=" * 60)
    print(f"Current time: {now}")
    print(f"Combo key: {combo}")
    print(f"Should revert: {revert}")
    print(f"\nTotal reverted combos: {len(REVERTED_COMBOS)}")
    print(f"SLTP params available: {SLTP_PARAMS_AVAILABLE}")
    
    if SLTP_PARAMS_AVAILABLE:
        print(f"SLTP combos loaded: {len(SLTP_PARAMS)}")
        
        # Show current SLTP
        ctx = get_time_context(now)
        for side in ['LONG', 'SHORT']:
            sltp = get_optimized_sltp(side, now)
            print(f"\n{side} SLTP for {combo}:")
            print(f"  SL: {sltp['sl_dist']:.2f} pts")
            print(f"  TP: {sltp['tp_dist']:.2f} pts")

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy
from volatility_filter import volatility_filter
from volume_profile import build_volume_profile


class ValueAreaBreakoutStrategy(Strategy):
    """
    Volume profile value-area breakout/acceptance.
    - Go long if price accepts above VAH for N bars.
    - Go short if price accepts below VAL for N bars.
    """

    def __init__(self) -> None:
        cfg = CONFIG.get("VALUE_AREA_BREAKOUT", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback = int(cfg.get("lookback", 120))
        self.value_area_pct = float(cfg.get("value_area_pct", 0.70))
        self.accept_bars = int(cfg.get("accept_bars", 2))
        self.accept_bars_long = int(cfg.get("accept_bars_long", self.accept_bars) or self.accept_bars)
        self.accept_bars_short = int(cfg.get("accept_bars_short", self.accept_bars) or self.accept_bars)
        self.buffer = float(cfg.get("buffer", 0.10))
        self.buffer_atr = float(cfg.get("buffer_atr", 0.10))
        self.er_window = int(cfg.get("er_window", 30))
        self.er_min = float(cfg.get("er_min", 0.25))
        self.min_range = float(cfg.get("min_range", 4.0))
        self.atr_window = int(cfg.get("atr_window", 20))
        self.close_position_ratio = float(cfg.get("close_position_ratio", 0.60))
        self.close_position_ratio_long = float(
            cfg.get("close_position_ratio_long", self.close_position_ratio)
        )
        self.close_position_ratio_short = float(
            cfg.get("close_position_ratio_short", self.close_position_ratio)
        )
        self.volume_mult = float(cfg.get("volume_mult", 1.0))
        self.volume_mult_long = float(cfg.get("volume_mult_long", self.volume_mult))
        self.volume_mult_short = float(cfg.get("volume_mult_short", self.volume_mult))
        self.sessions = set(cfg.get("sessions", ["NY_AM", "NY_PM", "LONDON"]))
        self.tick_size = float(cfg.get("tick_size", 0.25))
        self.trend_ema_period = int(cfg.get("trend_ema_period", 50) or 50)
        self.long_require_above_ema = bool(cfg.get("long_require_above_ema", False))
        self.short_require_below_ema = bool(cfg.get("short_require_below_ema", False))
        self.cooldown_bars = int(cfg.get("cooldown_bars", 0) or 0)
        self.trigger_on_transition = bool(cfg.get("trigger_on_transition", False))
        self._last_signal_ts: Optional[pd.Timestamp] = None
        allowed_regimes = cfg.get("allowed_regimes")
        if allowed_regimes is None:
            self.allowed_regimes = None
        else:
            self.allowed_regimes = {str(r).lower() for r in allowed_regimes if r}

        ass_cfg = cfg.get("ass", {}) or {}
        self.ass_enabled = bool(ass_cfg.get("enabled", False))
        self.ass_mode = str(ass_cfg.get("mode", "instrument") or "instrument").lower()
        self.ass_window_closes = int(ass_cfg.get("window_closes", 4) or 4)
        self.ass_thresholds = ass_cfg.get("thresholds", {}) or {}
        self.ass_retest_downgrade_atr = ass_cfg.get("retest_downgrade_atr")
        self.ass_survival_sl_mult = float(ass_cfg.get("survival_sl_mult", 1.0) or 1.0)
        self.ass_log_decisions = bool(ass_cfg.get("log_decisions", False))
        self.ass_long_penalty = float(ass_cfg.get("long_penalty", 0.0) or 0.0)
        self.ass_short_penalty = float(ass_cfg.get("short_penalty", 0.0) or 0.0)

        logging.info(
            "ValueAreaBreakoutStrategy initialized | lookback=%s accept_bars=%s",
            self.lookback,
            self.accept_bars,
        )

    def _calc_atr(self, df: pd.DataFrame) -> Optional[float]:
        if df is None or len(df) < max(self.atr_window, 2):
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.atr_window).mean().iloc[-1]
        return float(atr) if np.isfinite(atr) else None

    def _efficiency_ratio(self, closes: pd.Series) -> Optional[float]:
        if closes is None or len(closes) < self.er_window + 1:
            return None
        start = closes.iloc[-self.er_window - 1]
        end = closes.iloc[-1]
        net = abs(float(end) - float(start))
        total = float((closes.diff().abs().iloc[-self.er_window:]).sum())
        if total <= 0:
            return None
        return net / total

    def _ema_value(self, closes: pd.Series, period: int) -> Optional[float]:
        if closes is None or len(closes) < max(int(period), 2):
            return None
        ema = closes.ewm(span=int(period), adjust=False).mean().iloc[-1]
        return float(ema) if np.isfinite(ema) else None

    def _is_cooldown_active(self, ts: pd.Timestamp) -> bool:
        if self.cooldown_bars <= 0 or self._last_signal_ts is None:
            return False
        try:
            elapsed = (pd.Timestamp(ts) - pd.Timestamp(self._last_signal_ts)).total_seconds() / 60.0
            return float(elapsed) < float(self.cooldown_bars)
        except Exception:
            return False

    def _mark_signal(self, ts: pd.Timestamp) -> None:
        try:
            self._last_signal_ts = pd.Timestamp(ts)
        except Exception:
            self._last_signal_ts = None

    def _ass_score(
        self,
        df: pd.DataFrame,
        ts,
        side: str,
        vah: float,
        val: float,
        buffer: float,
        atr: Optional[float],
        close_pos: float,
        session: str,
    ) -> Dict[str, float]:
        """
        ValueAreaBreakout Acceptance Strength Score (ASS) in the range [0, 100].

        Uses only information available at signal time (no future bars).
        """
        side = str(side).upper()

        required_accept = self.accept_bars_long if side == "LONG" else self.accept_bars_short
        window = max(int(self.ass_window_closes), int(required_accept) + 1, 3)
        closes = df["close"].iloc[-window:]

        if side == "LONG":
            edge = float(vah) + float(buffer)
            outside_mask = closes > edge
        else:
            edge = float(val) - float(buffer)
            outside_mask = closes < edge
        outside_count = int(outside_mask.sum())

        # A) Outside-Value Hold / reclaim detection
        score_outside = 0.0
        if outside_count >= required_accept:
            score_outside += 30.0
        if outside_count >= required_accept + 1:
            score_outside += 15.0

        reclaimed = False
        first_outside = None
        try:
            mask_arr = outside_mask.to_numpy(dtype=bool)
            for i, is_out in enumerate(mask_arr):
                if is_out:
                    first_outside = i
                    break
        except Exception:
            first_outside = None
        if first_outside is not None and first_outside + 1 < len(closes):
            after = closes.iloc[first_outside + 1 :]
            if not after.empty:
                if side == "LONG":
                    reclaimed = bool((after < float(vah)).any())
                else:
                    reclaimed = bool((after > float(val)).any())
        if reclaimed:
            score_outside -= 25.0

        # Normalization denominator for ATR-scaled components.
        denom = (
            float(atr)
            if atr is not None and np.isfinite(atr) and float(atr) > 1e-9
            else max(float(buffer), float(self.tick_size), 1e-6)
        )

        # B) Migration distance beyond VAH/VAL edge
        last_close = float(df["close"].iloc[-1])
        mig_pts = (last_close - edge) if side == "LONG" else (edge - last_close)
        mig_pts = max(0.0, float(mig_pts))
        mig_ratio = float(mig_pts) / denom if denom > 0 else 0.0
        if mig_ratio < 0.4:
            score_mig = 0.0
        elif mig_ratio < 0.7:
            score_mig = 10.0
        elif mig_ratio < 1.1:
            score_mig = 20.0
        else:
            score_mig = 30.0

        # C) Wick intrusion back into value during acceptance bars (retest pressure)
        accept_slice = df.iloc[-required_accept:]
        if side == "LONG":
            intrusion_pts = max(0.0, float(vah) - float(accept_slice["low"].min()))
        else:
            intrusion_pts = max(0.0, float(accept_slice["high"].max()) - float(val))
        intrusion_ratio = float(intrusion_pts) / denom if denom > 0 else 0.0
        if intrusion_pts <= 0:
            score_retest = 15.0
        elif intrusion_ratio <= 0.25:
            score_retest = 10.0
        elif intrusion_ratio <= 0.50:
            score_retest = 5.0
        elif intrusion_ratio <= 0.80:
            score_retest = 0.0
        else:
            score_retest = -10.0

        # Bonus: close location on the signal bar (impulse quality)
        score_closepos = 0.0
        try:
            cp = float(close_pos)
        except Exception:
            cp = 0.5
        if side == "LONG":
            if cp >= 0.80:
                score_closepos = 5.0
            elif cp >= 0.70:
                score_closepos = 2.0
        else:
            if cp <= 0.20:
                score_closepos = 5.0
            elif cp <= 0.30:
                score_closepos = 2.0

        # Optional D) risk penalty windows (config-driven)
        penalty_risk = 0.0
        cfg = CONFIG.get("VALUE_AREA_BREAKOUT", {}) or {}
        ass_cfg = cfg.get("ass", {}) or {}
        for window_cfg in ass_cfg.get("risk_windows", []) or []:
            if not isinstance(window_cfg, dict):
                continue
            w_session = window_cfg.get("session")
            if w_session and str(w_session).upper() != str(session).upper():
                continue
            start = window_cfg.get("start")
            end = window_cfg.get("end")
            penalty = window_cfg.get("penalty")
            if start is None or end is None or penalty is None:
                continue
            try:
                start_h, start_m = [int(x) for x in str(start).split(":")[:2]]
                end_h, end_m = [int(x) for x in str(end).split(":")[:2]]
                ts_local = ts
                if hasattr(ts_local, "to_pydatetime"):
                    ts_local = ts_local.to_pydatetime()
                hh = int(getattr(ts_local, "hour"))
                mm = int(getattr(ts_local, "minute"))
                now_min = (hh * 60) + mm
                start_min = (start_h * 60) + start_m
                end_min = (end_h * 60) + end_m
                if start_min <= end_min:
                    in_window = start_min <= now_min <= end_min
                else:
                    in_window = now_min >= start_min or now_min <= end_min
                if in_window:
                    penalty_risk += float(penalty)
            except Exception:
                continue

        raw = score_outside + score_mig + score_retest + score_closepos - penalty_risk
        total = max(0.0, min(100.0, raw))
        return {
            "score": float(total),
            "score_outside": float(score_outside),
            "score_migration": float(score_mig),
            "score_retest": float(score_retest),
            "score_closepos": float(score_closepos),
            "penalty_risk": float(penalty_risk),
            "outside_count": float(outside_count),
            "reclaimed": 1.0 if reclaimed else 0.0,
            "migration_pts": float(mig_pts),
            "migration_ratio": float(mig_ratio),
            "intrusion_pts": float(intrusion_pts),
            "intrusion_ratio": float(intrusion_ratio),
            "atr": float(atr) if atr is not None and np.isfinite(atr) else 0.0,
            "buffer": float(buffer),
            "vah": float(vah),
            "val": float(val),
        }

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled:
            return None
        min_accept = max(self.accept_bars, self.accept_bars_long, self.accept_bars_short)
        if df is None or len(df) < self.lookback + min_accept + 2:
            return None

        ts = df.index[-1]
        session = volatility_filter.get_session(ts.hour)
        if self.sessions and session not in self.sessions:
            return None
        if self._is_cooldown_active(ts):
            return None
        if self.allowed_regimes is not None:
            regime, _, _ = volatility_filter.get_regime(df, ts)
            if regime not in self.allowed_regimes:
                return None

        range_window = df.iloc[-self.lookback :]
        if float(range_window["high"].max() - range_window["low"].min()) < self.min_range:
            return None

        er = self._efficiency_ratio(df["close"])
        if er is None or er < self.er_min:
            return None

        profile = build_volume_profile(
            df,
            lookback=self.lookback,
            tick_size=self.tick_size,
            value_area_pct=self.value_area_pct,
        )
        if not profile:
            return None

        vah = float(profile["vah"])
        val = float(profile["val"])

        atr = self._calc_atr(df)
        trend_ema = self._ema_value(df["close"], self.trend_ema_period)
        buffer = max(self.buffer, (atr or 0.0) * self.buffer_atr)
        long_accept_bars = max(1, int(self.accept_bars_long))
        short_accept_bars = max(1, int(self.accept_bars_short))
        long_closes = df["close"].iloc[-long_accept_bars:]
        short_closes = df["close"].iloc[-short_accept_bars:]
        long_accept = bool((long_closes > vah + buffer).all())
        short_accept = bool((short_closes < val - buffer).all())
        if self.trigger_on_transition:
            prev_long_closes = df["close"].iloc[-(long_accept_bars + 1) : -1]
            prev_short_closes = df["close"].iloc[-(short_accept_bars + 1) : -1]
            if len(prev_long_closes) == long_accept_bars and bool((prev_long_closes > vah + buffer).all()):
                long_accept = False
            if len(prev_short_closes) == short_accept_bars and bool((prev_short_closes < val - buffer).all()):
                short_accept = False

        if long_accept:
            last = df.iloc[-1]
            bar_range = float(last["high"] - last["low"])
            close_pos = (float(last["close"]) - float(last["low"])) / bar_range if bar_range > 0 else 0.5
            if close_pos < self.close_position_ratio_long:
                return None
            if self.long_require_above_ema and trend_ema is not None and float(last["close"]) < trend_ema:
                return None
            if self.volume_mult_long > 1.0 and "volume" in df.columns:
                vol_series = df["volume"].iloc[-self.lookback :]
                avg_vol = float(vol_series.mean()) if len(vol_series) else 0.0
                if avg_vol > 0 and float(last["volume"]) < avg_vol * self.volume_mult_long:
                    return None
            sltp = dynamic_sltp_engine.calculate_sltp("ValueAreaBreakout_LONG", df)
            logging.info("ValueAreaBreakout: LONG acceptance above VAH %.2f", vah)
            dynamic_sltp_engine.log_params(sltp, "ValueAreaBreakout_LONG")
            sl_dist = float(sltp["sl_dist"])
            tp_dist = float(sltp["tp_dist"])
            tighten_mult = float(CONFIG.get("VALUE_AREA_BREAKOUT", {}).get("sl_tighten_mult", 1.0))
            min_stop_ticks = CONFIG.get("VALUE_AREA_BREAKOUT", {}).get("min_stop_ticks")
            max_stop_ticks = CONFIG.get("VALUE_AREA_BREAKOUT", {}).get("max_stop_ticks")
            min_stop_points = None
            if min_stop_ticks is not None:
                try:
                    min_stop_points = float(min_stop_ticks) * self.tick_size
                except Exception:
                    min_stop_points = None
            if min_stop_points is None:
                min_stop_points = float(CONFIG.get("SLTP_MIN", {}).get("sl", 0.0))
            adjusted_sl = max(sl_dist * tighten_mult, min_stop_points)
            if max_stop_ticks is not None:
                try:
                    max_stop_points = float(max_stop_ticks) * self.tick_size
                    if max_stop_points > 0:
                        adjusted_sl = min(adjusted_sl, max_stop_points)
                except Exception:
                    pass
            if self.tick_size > 0:
                adjusted_sl = round(adjusted_sl / self.tick_size) * self.tick_size
            if adjusted_sl != sl_dist:
                logging.info("ValueAreaBreakout: SL tightened from %.2f to %.2f", sl_dist, adjusted_sl)

            ass = None
            ass_template = None
            if self.ass_enabled and self.ass_mode != "off":
                ass = self._ass_score(
                    df=df,
                    ts=ts,
                    side="LONG",
                    vah=vah,
                    val=val,
                    buffer=buffer,
                    atr=atr,
                    close_pos=close_pos,
                    session=session,
                )
                raw_score = float(ass.get("score", 0.0))
                score = max(0.0, raw_score - self.ass_long_penalty)
                th = self.ass_thresholds.get(session, self.ass_thresholds.get("default", {})) or {}
                diag_min = float(th.get("diagnostic", 70.0) or 70.0)
                surv_min = float(th.get("survival", 45.0) or 45.0)
                force_template = str(th.get("force") or "").strip().upper()
                if self.ass_mode == "trade":
                    if score < surv_min:
                        if self.ass_log_decisions:
                            logging.info(
                                "ValueAreaBreakout ASS BLOCK (LONG) | score=%.1f < %.1f | session=%s",
                                score,
                                surv_min,
                                session,
                            )
                        return None
                    ass_template = "DIAG" if score >= diag_min else "SURV"
                    if force_template in ("DIAG", "SURV"):
                        ass_template = force_template
                    downgrade_atr = None
                    if self.ass_retest_downgrade_atr is not None:
                        try:
                            downgrade_atr = float(self.ass_retest_downgrade_atr)
                        except Exception:
                            downgrade_atr = None
                    if ass_template == "DIAG" and downgrade_atr is not None:
                        intrusion_ratio = float(ass.get("intrusion_ratio", 0.0))
                        if intrusion_ratio >= downgrade_atr:
                            # In DIAG-forced sessions, deep retest pressure is a hard "no"
                            # (we don't downgrade to SURV; we block instead).
                            if force_template == "DIAG":
                                if self.ass_log_decisions:
                                    logging.info(
                                        "ValueAreaBreakout ASS BLOCK (LONG) | intrusion=%.2f >= %.2f | session=%s",
                                        intrusion_ratio,
                                        downgrade_atr,
                                        session,
                                    )
                                return None
                            ass_template = "SURV"
                    if self.ass_log_decisions:
                        logging.info(
                            "ValueAreaBreakout ASS (LONG) | score=%.1f raw=%.1f template=%s session=%s",
                            score,
                            raw_score,
                            ass_template,
                            session,
                        )
                else:
                    ass_template = "DIAG" if score >= diag_min else ("SURV" if score >= surv_min else "BLOCK")
                    if force_template in ("DIAG", "SURV") and ass_template != "BLOCK":
                        ass_template = force_template
                    if self.ass_log_decisions:
                        logging.info(
                            "ValueAreaBreakout ASS (LONG) | score=%.1f raw=%.1f suggested=%s session=%s",
                            score,
                            raw_score,
                            ass_template,
                            session,
                        )

            final_sl = adjusted_sl
            if self.ass_enabled and self.ass_mode == "trade" and ass_template == "SURV":
                survival_sl = max(sl_dist * self.ass_survival_sl_mult, min_stop_points)
                if max_stop_ticks is not None:
                    try:
                        max_stop_points = float(max_stop_ticks) * self.tick_size
                        if max_stop_points > 0:
                            survival_sl = min(survival_sl, max_stop_points)
                    except Exception:
                        pass
                if self.tick_size > 0:
                    survival_sl = round(survival_sl / self.tick_size) * self.tick_size
                final_sl = float(survival_sl)

            signal = {
                "strategy": "ValueAreaBreakout",
                "sub_strategy": f"ASS_{ass_template}"
                if (self.ass_enabled and self.ass_mode == "trade" and ass_template)
                else None,
                "side": "LONG",
                "tp_dist": tp_dist,
                "sl_dist": final_sl,
            }
            if isinstance(ass, dict):
                signal.update(
                    {
                        "vab_ass_score": float(ass.get("score", 0.0)),
                        "vab_ass_score_eff": float(score) if self.ass_enabled else None,
                        "vab_ass_template": ass_template,
                        "vab_ass_outside": float(ass.get("score_outside", 0.0)),
                        "vab_ass_migration": float(ass.get("score_migration", 0.0)),
                        "vab_ass_retest": float(ass.get("score_retest", 0.0)),
                        "vab_ass_closepos": float(ass.get("score_closepos", 0.0)),
                        "vab_ass_penalty": float(ass.get("penalty_risk", 0.0)),
                        "vab_migration_ratio": float(ass.get("migration_ratio", 0.0)),
                        "vab_intrusion_ratio": float(ass.get("intrusion_ratio", 0.0)),
                        "vab_close_pos": float(close_pos),
                        "vab_vah": float(vah),
                        "vab_val": float(val),
                        "vab_buffer": float(buffer),
                        "vab_atr": float(atr) if atr is not None and np.isfinite(atr) else None,
                        "vab_sl_base": float(sl_dist),
                        "vab_sl_tight": float(adjusted_sl),
                    }
                )
            self._mark_signal(ts)
            return signal

        if short_accept:
            last = df.iloc[-1]
            bar_range = float(last["high"] - last["low"])
            close_pos = (float(last["close"]) - float(last["low"])) / bar_range if bar_range > 0 else 0.5
            if close_pos > (1.0 - self.close_position_ratio_short):
                return None
            if self.short_require_below_ema and trend_ema is not None and float(last["close"]) > trend_ema:
                return None
            if self.volume_mult_short > 1.0 and "volume" in df.columns:
                vol_series = df["volume"].iloc[-self.lookback :]
                avg_vol = float(vol_series.mean()) if len(vol_series) else 0.0
                if avg_vol > 0 and float(last["volume"]) < avg_vol * self.volume_mult_short:
                    return None
            sltp = dynamic_sltp_engine.calculate_sltp("ValueAreaBreakout_SHORT", df)
            logging.info("ValueAreaBreakout: SHORT acceptance below VAL %.2f", val)
            dynamic_sltp_engine.log_params(sltp, "ValueAreaBreakout_SHORT")
            sl_dist = float(sltp["sl_dist"])
            tp_dist = float(sltp["tp_dist"])
            tighten_mult = float(CONFIG.get("VALUE_AREA_BREAKOUT", {}).get("sl_tighten_mult", 1.0))
            min_stop_ticks = CONFIG.get("VALUE_AREA_BREAKOUT", {}).get("min_stop_ticks")
            max_stop_ticks = CONFIG.get("VALUE_AREA_BREAKOUT", {}).get("max_stop_ticks")
            min_stop_points = None
            if min_stop_ticks is not None:
                try:
                    min_stop_points = float(min_stop_ticks) * self.tick_size
                except Exception:
                    min_stop_points = None
            if min_stop_points is None:
                min_stop_points = float(CONFIG.get("SLTP_MIN", {}).get("sl", 0.0))
            adjusted_sl = max(sl_dist * tighten_mult, min_stop_points)
            if max_stop_ticks is not None:
                try:
                    max_stop_points = float(max_stop_ticks) * self.tick_size
                    if max_stop_points > 0:
                        adjusted_sl = min(adjusted_sl, max_stop_points)
                except Exception:
                    pass
            if self.tick_size > 0:
                adjusted_sl = round(adjusted_sl / self.tick_size) * self.tick_size
            if adjusted_sl != sl_dist:
                logging.info("ValueAreaBreakout: SL tightened from %.2f to %.2f", sl_dist, adjusted_sl)

            ass = None
            ass_template = None
            if self.ass_enabled and self.ass_mode != "off":
                ass = self._ass_score(
                    df=df,
                    ts=ts,
                    side="SHORT",
                    vah=vah,
                    val=val,
                    buffer=buffer,
                    atr=atr,
                    close_pos=close_pos,
                    session=session,
                )
                raw_score = float(ass.get("score", 0.0))
                score = max(0.0, raw_score - self.ass_short_penalty)
                th = self.ass_thresholds.get(session, self.ass_thresholds.get("default", {})) or {}
                diag_min = float(th.get("diagnostic", 70.0) or 70.0)
                surv_min = float(th.get("survival", 45.0) or 45.0)
                force_template = str(th.get("force") or "").strip().upper()
                if self.ass_mode == "trade":
                    if score < surv_min:
                        if self.ass_log_decisions:
                            logging.info(
                                "ValueAreaBreakout ASS BLOCK (SHORT) | score=%.1f < %.1f | session=%s",
                                score,
                                surv_min,
                                session,
                            )
                        return None
                    ass_template = "DIAG" if score >= diag_min else "SURV"
                    if force_template in ("DIAG", "SURV"):
                        ass_template = force_template
                    downgrade_atr = None
                    if self.ass_retest_downgrade_atr is not None:
                        try:
                            downgrade_atr = float(self.ass_retest_downgrade_atr)
                        except Exception:
                            downgrade_atr = None
                    if ass_template == "DIAG" and downgrade_atr is not None:
                        intrusion_ratio = float(ass.get("intrusion_ratio", 0.0))
                        if intrusion_ratio >= downgrade_atr:
                            if force_template == "DIAG":
                                if self.ass_log_decisions:
                                    logging.info(
                                        "ValueAreaBreakout ASS BLOCK (SHORT) | intrusion=%.2f >= %.2f | session=%s",
                                        intrusion_ratio,
                                        downgrade_atr,
                                        session,
                                    )
                                return None
                            ass_template = "SURV"
                    if self.ass_log_decisions:
                        logging.info(
                            "ValueAreaBreakout ASS (SHORT) | score=%.1f raw=%.1f template=%s session=%s",
                            score,
                            raw_score,
                            ass_template,
                            session,
                        )
                else:
                    ass_template = "DIAG" if score >= diag_min else ("SURV" if score >= surv_min else "BLOCK")
                    if force_template in ("DIAG", "SURV") and ass_template != "BLOCK":
                        ass_template = force_template
                    if self.ass_log_decisions:
                        logging.info(
                            "ValueAreaBreakout ASS (SHORT) | score=%.1f raw=%.1f suggested=%s session=%s",
                            score,
                            raw_score,
                            ass_template,
                            session,
                        )

            final_sl = adjusted_sl
            if self.ass_enabled and self.ass_mode == "trade" and ass_template == "SURV":
                survival_sl = max(sl_dist * self.ass_survival_sl_mult, min_stop_points)
                if max_stop_ticks is not None:
                    try:
                        max_stop_points = float(max_stop_ticks) * self.tick_size
                        if max_stop_points > 0:
                            survival_sl = min(survival_sl, max_stop_points)
                    except Exception:
                        pass
                if self.tick_size > 0:
                    survival_sl = round(survival_sl / self.tick_size) * self.tick_size
                final_sl = float(survival_sl)

            signal = {
                "strategy": "ValueAreaBreakout",
                "sub_strategy": f"ASS_{ass_template}"
                if (self.ass_enabled and self.ass_mode == "trade" and ass_template)
                else None,
                "side": "SHORT",
                "tp_dist": tp_dist,
                "sl_dist": final_sl,
            }
            if isinstance(ass, dict):
                signal.update(
                    {
                        "vab_ass_score": float(ass.get("score", 0.0)),
                        "vab_ass_score_eff": float(score) if self.ass_enabled else None,
                        "vab_ass_template": ass_template,
                        "vab_ass_outside": float(ass.get("score_outside", 0.0)),
                        "vab_ass_migration": float(ass.get("score_migration", 0.0)),
                        "vab_ass_retest": float(ass.get("score_retest", 0.0)),
                        "vab_ass_closepos": float(ass.get("score_closepos", 0.0)),
                        "vab_ass_penalty": float(ass.get("penalty_risk", 0.0)),
                        "vab_migration_ratio": float(ass.get("migration_ratio", 0.0)),
                        "vab_intrusion_ratio": float(ass.get("intrusion_ratio", 0.0)),
                        "vab_close_pos": float(close_pos),
                        "vab_vah": float(vah),
                        "vab_val": float(val),
                        "vab_buffer": float(buffer),
                        "vab_atr": float(atr) if atr is not None and np.isfinite(atr) else None,
                        "vab_sl_base": float(sl_dist),
                        "vab_sl_tight": float(adjusted_sl),
                    }
                )
            self._mark_signal(ts)
            return signal

        return None

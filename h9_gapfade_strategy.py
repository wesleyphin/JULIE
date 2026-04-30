"""H9 Gap-Fade ML strategy — cash-open mean-reversion at 09:30 ET.

Edge (validated 2011-2026, 4757 trading days, full ES outright stack):
  - 09:30 ET cash open ≥0.5% BELOW prev RTH close → LONG, 0.30%/0.40% TP/SL.
  - 09:30 ET cash open ≥0.5% ABOVE prev RTH close → SHORT, 0.30%/0.40% TP/SL.
  - Up to 30-minute hold; if neither bracket hits, exit at the 30-min close.

Statistical signature (raw rule, no ML):
  - LONG bias on gap-down >0.5%:  mean +17.3 bps, t=+6.40σ, WR 60.7%
  - SHORT bias on gap-up >0.5%:   mean -12.3 bps, t=-5.03σ, WR 57.8%
  - 14 of 16 calendar years profitable; +$198,743 net over 15+ years
    at size=10 MES, $5/pt, $1.50 commission.

39-feature ML overlay (matches AetherFlow / DE3 depth):
  - LONG-side gating model: random-control z=+4.79σ (real edge above noise)
  - SHORT-side gating model: random-control z=+8.04σ (very strong)
  - Bundled at artifacts/h9_gapfade_ml/models.joblib

Plug-in pattern matches FibH1214Strategy / AetherFlowStrategy:
  - on_bar(df) returns a signal dict or None
  - state_for_persist() / restore_state(payload) for daily restart
  - record_trade_pnl() no-op (constant size, no adaptive layer)

Feature groups (39 total):
  - Gap geometry (5)         : gap_pct, abs_gap_pct, gap_squared, gap_z60, gap_dir
  - Pre-RTH ETH context (8)  : eth_range_pts, eth_range_pct, eth_close_loc, …
  - Trailing returns (5)     : ret_1d, 3d, 5d, 10d, 20d
  - Realized vol (5)         : rv_5d, 10d, 20d, 60d, rv_5_20_ratio
  - Range / position (5)     : prev_rth_range_pct, pos_in_60d_hi/lo, MA-distances
  - Calendar / regime (8)    : dow, dom, month, quarter, is_monday/friday/eom
  - Volume / liquidity (3)   : rth_vol_z20, eth_to_rth_volume_ratio, vol_pct_60d
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

import numpy as np
import pandas as pd

NY_TZ = "US/Eastern"
LOG = logging.getLogger("H9GapFade")

# ---- Strategy constants (match validated backtest config) ----
GAP_THRESHOLD_PCT = 0.005      # 0.5% gap threshold either side
TP_PCT = 0.0030                # 0.30% take profit
SL_PCT = 0.0040                # 0.40% stop loss
HORIZON_MIN = 30               # max trade duration
ENTRY_HOUR_ET = 9
ENTRY_MINUTE_ET = 30
# Sized for $50k Topstep ($2k trailing DD). Sized-down simulation:
#   size=2 + 3-consec-loss circuit  → worst all-time DD $1,396, ann $2,608/yr.
# Bump up only if your funded-account DD allows: at size=N the all-time DD
# is N × $958 and the 3-consec-loss circuit caps at N × $698.
FIXED_SIZE = 2                 # was 10 — sized down for funded-account safety
CIRCUIT_MAX_CONSEC_LOSSES = 3  # pause after this many consecutive losers
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "h9_gapfade_ml"
MODEL_PATH = ARTIFACT_DIR / "models.joblib"

# 39 ML features — must match the training pipeline in tools/h9_gapfade_ml.py.
ML_FEATURES = [
    # Gap geometry (5)
    "gap_pct", "abs_gap_pct", "gap_squared", "gap_z60", "gap_dir",
    # Pre-RTH ETH context (8)
    "eth_range_pts", "eth_range_pct", "eth_close_loc", "eth_close_vs_prev",
    "eth_above_prev_close", "eth_below_prev_close", "eth_open_drift",
    "eth_volume_z",
    # Trailing returns (5)
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # Realized vol (5)
    "rv_5d", "rv_10d", "rv_20d", "rv_60d", "rv_5_20_ratio",
    # Range / position (5)
    "prev_rth_range_pct", "pos_in_60d_hi", "pos_in_60d_lo",
    "ma50_dist_pct", "ma200_dist_pct",
    # Calendar / regime (8)
    "dow", "dom", "month", "quarter",
    "is_monday", "is_friday", "is_first_5d_of_month", "is_last_5d_of_month",
    # Volume / liquidity (3)
    "rth_vol_z20", "eth_to_rth_volume_ratio", "volume_pct_of_60d_max",
]


def _safe_div(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or den == 0:
        return 0.0
    return float(num / den)


def _z(x: float, mean: float, std: float) -> float:
    if std is None or not np.isfinite(std) or std <= 0:
        return 0.0
    return float((x - mean) / std)


@dataclass
class H9GapFadeStrategy:
    """H9 Gap-Fade rule with optional 39-feature ML overlay on both sides.

    Self-gated against dead_tape regime: the bot's downstream
    apply_dead_tape_brackets() rewrites tp_dist→3pt and sl_dist→5pt and
    forces size=1 whenever the regime classifier flags 'dead_tape'. With
    H9 GapFade's designed 21pt/28pt brackets, that rewrite is destructive
    (live simulation: $13k/yr → $278/yr). Rather than add skip-guards in
    julie001.py, we skip the entry entirely when dead_tape is active so
    the strategy only fires when its designed brackets will survive.
    Set ``skip_on_dead_tape=False`` to disable the gate."""
    name: str = "h9_gapfade"
    label: str = "H9 GapFade"
    enabled: bool = True
    use_ml_long: bool = True
    use_ml_short: bool = True
    # Default OFF — production deployment uses skip-guards in julie001.py
    # (apply_dead_tape_brackets and Kalshi overlay both bypass H9GapFade
    # signals) which keeps the designed 0.30%/0.40% brackets intact even
    # on dead-tape regime days. The self-gate here remains as a fallback
    # that can be flipped on if the skip-guards are ever removed.
    skip_on_dead_tape: bool = False
    fixed_size: int = FIXED_SIZE
    # Circuit breaker — funded-account safety. After N consecutive losing
    # closes the next signal is skipped (and the counter resets). Sized
    # for the worst stretches the strategy has ever seen (2021-06 to
    # 2021-10 ran 11 SL-hits in 18 fires; the 3-loss circuit caps that
    # bleed at $1,396 max DD at size=2).
    circuit_max_consec_losses: int = CIRCUIT_MAX_CONSEC_LOSSES
    consecutive_losses: int = 0
    circuit_skips: int = 0
    last_signal_day: Optional[str] = None
    last_signal_time: Optional[str] = None
    last_block_reason: Optional[str] = None
    _model_bundle: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _model_load_attempted: bool = field(default=False, init=False, repr=False)

    # --- ML overlay ---
    def _ensure_model(self) -> None:
        if self._model_load_attempted:
            return
        self._model_load_attempted = True
        if joblib is None or not MODEL_PATH.exists():
            LOG.warning("H9GapFade: ML overlay unavailable (joblib=%s, exists=%s)",
                        joblib is not None, MODEL_PATH.exists())
            return
        try:
            self._model_bundle = joblib.load(MODEL_PATH)
            n_models = len((self._model_bundle or {}).get("models") or {})
            LOG.info("H9GapFade: loaded ML overlay from %s (%d models, %d features)",
                     MODEL_PATH, n_models, len(ML_FEATURES))
        except Exception as exc:  # pragma: no cover
            LOG.warning("H9GapFade: failed to load ML overlay: %s", exc)
            self._model_bundle = None

    def _ml_predict(self, side: str, feat_vec: np.ndarray
                    ) -> tuple[Optional[float], Optional[float]]:
        if not self._model_bundle:
            return None, None
        models = self._model_bundle.get("models") or {}
        thrs = self._model_bundle.get("thresholds") or {}
        clf = models.get(side)
        thr = thrs.get(side)
        if clf is None:
            return None, None
        try:
            prob = float(clf.predict_proba(feat_vec)[:, 1][0])
        except Exception as exc:  # pragma: no cover
            LOG.debug("H9GapFade ML predict failed (%s): %s", side, exc)
            return None, None
        return prob, (float(thr) if thr is not None else None)

    # --- Gating helpers ---
    @staticmethod
    def _is_entry_bar(ts: pd.Timestamp) -> bool:
        if ts.tz is None:
            ts = ts.tz_localize(NY_TZ)
        else:
            ts = ts.tz_convert(NY_TZ)
        return (ts.hour == ENTRY_HOUR_ET and ts.minute == ENTRY_MINUTE_ET
                and ts.dayofweek < 5)

    # --- Feature builder (live) ---
    def _build_features(self, df: pd.DataFrame, today_open: float,
                          ts: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Compute the 39 features at runtime from a 1-min OHLCV df.

        Requires roughly 200+ trading days of history (≈80,000 bars) to
        populate the longest rolling features (200-day MA distance). With
        less history the long-horizon features fall back to 0 — model still
        runs but predictions degrade for early days."""
        if df is None or df.empty:
            return None
        f = df.copy()
        f.index = pd.to_datetime(f.index)
        if f.index.tz is None:
            f.index = f.index.tz_localize(NY_TZ)
        # Per-day RTH close series (used for trailing returns/vols)
        f_local = f.tz_convert(NY_TZ) if f.index.tz != NY_TZ else f
        f_local = f_local.copy()
        f_local["date"] = f_local.index.date
        f_local["hour"] = f_local.index.hour
        f_local["minute"] = f_local.index.minute
        rth_mask = (f_local["hour"] >= 9) & (f_local["hour"] < 16) & ~(
            (f_local["hour"] == 9) & (f_local["minute"] < 30))
        rth_daily = (f_local[rth_mask].groupby("date")
                       .agg(rth_close=("close", "last"),
                             rth_high=("high", "max"),
                             rth_low=("low", "min"),
                             rth_volume=("volume", "sum")))
        if len(rth_daily) < 2:
            return None
        rth_daily = rth_daily.sort_index()
        # Prior trading day stats — last full RTH day before today
        today_date = ts.tz_convert(NY_TZ).date() if ts.tz else ts.tz_localize(NY_TZ).date()
        prior_rth = rth_daily[rth_daily.index < today_date]
        if prior_rth.empty:
            return None
        prev_close = float(prior_rth["rth_close"].iloc[-1])
        prev_high = float(prior_rth["rth_high"].iloc[-1])
        prev_low = float(prior_rth["rth_low"].iloc[-1])
        prev_rth_volume = float(prior_rth["rth_volume"].iloc[-1])
        if prev_close <= 0:
            return None
        # ETH window (last close → 09:30 today)
        eth_mask = (
            (f_local.index.tz_convert(NY_TZ).date == today_date)
            & ((f_local["hour"] < 9) | ((f_local["hour"] == 9) & (f_local["minute"] < 30)))
        )
        eth_today = f_local[eth_mask]
        if eth_today.empty:
            eth_high = max(today_open, prev_close)
            eth_low = min(today_open, prev_close)
            eth_open = prev_close
            eth_close = today_open
            eth_volume = 0.0
        else:
            eth_high = float(eth_today["high"].max())
            eth_low = float(eth_today["low"].min())
            eth_open = float(eth_today["open"].iloc[0])
            eth_close = float(eth_today["close"].iloc[-1])
            eth_volume = float(eth_today["volume"].sum())
        # Trailing daily returns (use rth_close history)
        closes = prior_rth["rth_close"]
        def _ret(n):
            if len(closes) < n + 1:
                return 0.0
            return float(closes.iloc[-1] / closes.iloc[-1 - n] - 1)
        ret_1d = _ret(1); ret_3d = _ret(3); ret_5d = _ret(5)
        ret_10d = _ret(10); ret_20d = _ret(20)
        # Realized vol of daily returns
        rets = closes.pct_change().dropna()
        def _rv(n):
            if len(rets) < n:
                return 0.0
            return float(rets.iloc[-n:].std())
        rv_5d = _rv(5); rv_10d = _rv(10)
        rv_20d = _rv(20); rv_60d = _rv(60)
        rv_5_20_ratio = _safe_div(rv_5d, rv_20d)
        # Position-relative
        pos_in_60d_hi = _safe_div(prev_close, closes.iloc[-60:].max() if len(closes) >= 60 else closes.max())
        pos_in_60d_lo = _safe_div(prev_close, closes.iloc[-60:].min() if len(closes) >= 60 else closes.min())
        ma50 = closes.iloc[-50:].mean() if len(closes) >= 50 else closes.mean()
        ma200 = closes.iloc[-200:].mean() if len(closes) >= 200 else closes.mean()
        ma50_dist_pct = _safe_div(prev_close, ma50) - 1
        ma200_dist_pct = _safe_div(prev_close, ma200) - 1
        # Volume z-scores
        vol_series = prior_rth["rth_volume"]
        vol_mean20 = vol_series.iloc[-20:].mean() if len(vol_series) >= 20 else vol_series.mean()
        vol_std20 = vol_series.iloc[-20:].std() if len(vol_series) >= 20 else vol_series.std()
        rth_vol_z20 = _z(prev_rth_volume, vol_mean20, vol_std20)
        eth_to_rth_volume_ratio = _safe_div(eth_volume, prev_rth_volume)
        vol_pct_60d = _safe_div(prev_rth_volume,
                                 vol_series.iloc[-60:].max() if len(vol_series) >= 60 else vol_series.max())
        # Gap features
        gap_pct = (today_open / prev_close) - 1
        # Gap z-score over 60-day distribution
        gap_history = (rth_daily["rth_close"].pct_change().shift(-1).dropna()
                          if False else None)  # not yet — simpler:
        # Use trailing daily returns as proxy for typical move size
        gap_z60 = _z(gap_pct, rets.iloc[-60:].mean() if len(rets) >= 20 else 0.0,
                     rets.iloc[-60:].std() if len(rets) >= 20 else 0.01)
        # ETH features
        eth_range_pts = eth_high - eth_low
        eth_range_pct = _safe_div(eth_range_pts, prev_close)
        eth_close_loc = _safe_div(eth_close - eth_low, eth_range_pts) if eth_range_pts > 0 else 0.5
        eth_close_vs_prev = _safe_div(eth_close, prev_close) - 1
        eth_above_prev_close = 1 if eth_high > prev_close else 0
        eth_below_prev_close = 1 if eth_low < prev_close else 0
        eth_open_drift = _safe_div(today_open, eth_close) - 1
        eth_vol_mean20 = (rth_daily["rth_volume"].shift(-1).iloc[-20:].mean()
                            if len(rth_daily) >= 21 else eth_volume)
        eth_vol_std20 = (rth_daily["rth_volume"].shift(-1).iloc[-20:].std()
                           if len(rth_daily) >= 21 else max(eth_volume * 0.3, 1.0))
        eth_volume_z = _z(eth_volume, eth_vol_mean20, eth_vol_std20)
        # Calendar
        ts_local = ts.tz_convert(NY_TZ) if ts.tz else ts.tz_localize(NY_TZ)
        feats = {
            "gap_pct": gap_pct,
            "abs_gap_pct": abs(gap_pct),
            "gap_squared": gap_pct ** 2,
            "gap_z60": gap_z60,
            "gap_dir": int(np.sign(gap_pct)),
            "eth_range_pts": eth_range_pts,
            "eth_range_pct": eth_range_pct,
            "eth_close_loc": eth_close_loc,
            "eth_close_vs_prev": eth_close_vs_prev,
            "eth_above_prev_close": eth_above_prev_close,
            "eth_below_prev_close": eth_below_prev_close,
            "eth_open_drift": eth_open_drift,
            "eth_volume_z": eth_volume_z,
            "ret_1d": ret_1d, "ret_3d": ret_3d, "ret_5d": ret_5d,
            "ret_10d": ret_10d, "ret_20d": ret_20d,
            "rv_5d": rv_5d, "rv_10d": rv_10d, "rv_20d": rv_20d,
            "rv_60d": rv_60d, "rv_5_20_ratio": rv_5_20_ratio,
            "prev_rth_range_pct": _safe_div(prev_high - prev_low, prev_close),
            "pos_in_60d_hi": pos_in_60d_hi,
            "pos_in_60d_lo": pos_in_60d_lo,
            "ma50_dist_pct": ma50_dist_pct,
            "ma200_dist_pct": ma200_dist_pct,
            "dow": int(ts_local.dayofweek),
            "dom": int(ts_local.day),
            "month": int(ts_local.month),
            "quarter": int(ts_local.quarter),
            "is_monday": int(ts_local.dayofweek == 0),
            "is_friday": int(ts_local.dayofweek == 4),
            "is_first_5d_of_month": int(ts_local.day <= 5),
            "is_last_5d_of_month": int(ts_local.day >= 25),
            "rth_vol_z20": rth_vol_z20,
            "eth_to_rth_volume_ratio": eth_to_rth_volume_ratio,
            "volume_pct_of_60d_max": vol_pct_60d,
        }
        # Sanity: ensure every ML_FEATURES key is present
        missing = [k for k in ML_FEATURES if k not in feats]
        if missing:
            LOG.error("H9GapFade feature builder missing: %s", missing)
            return None
        return feats

    # --- Self-gate: dead_tape regime ---
    def _dead_tape_active(self) -> bool:
        """Return True if the regime classifier currently reports
        'dead_tape'. Imported lazily so this strategy module remains
        decoupled from the rest of the bot at import time."""
        if not self.skip_on_dead_tape:
            return False
        try:
            from regime_classifier import _CLASSIFIER as _RC
            return bool(_RC is not None and getattr(_RC, "regime", None) == "dead_tape")
        except Exception:  # pragma: no cover
            return False

    # --- Strategy entry point ---
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not self.enabled or df is None or df.empty:
            return None
        ts = df.index[-1]
        if not self._is_entry_bar(ts):
            return None
        day_key = (ts.tz_convert(NY_TZ) if ts.tz else ts.tz_localize(NY_TZ)).strftime("%Y-%m-%d")
        if self.last_signal_day == day_key:
            return None  # one signal per day per strategy

        # Circuit breaker: skip the next entry if N consecutive losers piled
        # up. Resets the counter so the strategy resumes on the SUBSEQUENT
        # gap-day, not the very next 09:30 bar — buys one day of cool-down.
        if self.consecutive_losses >= self.circuit_max_consec_losses:
            self.last_block_reason = (
                f"circuit_breaker_{self.consecutive_losses}_consec_losses"
            )
            self.last_signal_day = day_key  # mark today as "consumed" so we don't retry
            self.circuit_skips += 1
            self.consecutive_losses = 0
            LOG.warning(
                "H9GapFade: circuit-breaker tripped — skipped today's signal "
                "after %d consecutive losses (skip #%d)",
                self.circuit_max_consec_losses, self.circuit_skips,
            )
            return None

        # Self-gate: skip if dead_tape regime is active. The bot's
        # apply_dead_tape_brackets would rewrite our designed 21/28pt
        # brackets to 3/5pt and force size=1, gutting the edge.
        if self._dead_tape_active():
            self.last_block_reason = "dead_tape_active"
            LOG.info("H9GapFade: skipped — dead_tape regime active "
                     "(would clip designed brackets)")
            return None

        today_open = float(df["open"].iloc[-1])
        feats = self._build_features(df, today_open, ts)
        if feats is None:
            return None
        gap_pct = feats["gap_pct"]
        if abs(gap_pct) < GAP_THRESHOLD_PCT:
            return None

        side = "LONG" if gap_pct < 0 else "SHORT"
        # ML gating
        ml_prob: Optional[float] = None
        ml_thr: Optional[float] = None
        ml_blocked = False
        use_ml_for_side = (self.use_ml_long if side == "LONG" else self.use_ml_short)
        if use_ml_for_side:
            self._ensure_model()
            feat_vec = np.array([[feats[k] for k in ML_FEATURES]], dtype=float)
            ml_prob, ml_thr = self._ml_predict(side.lower(), feat_vec)
            if ml_prob is not None and ml_thr is not None and ml_prob < ml_thr:
                ml_blocked = True
                LOG.info("H9GapFade %s blocked by ML: prob=%.3f < thr=%.3f",
                         side, ml_prob, ml_thr)
        if ml_blocked:
            return None

        tp_pts = today_open * TP_PCT
        sl_pts = today_open * SL_PCT
        sub_strategy = "gap_dn_long" if side == "LONG" else "gap_up_short"
        signal: Dict[str, Any] = {
            "strategy": "H9GapFade",
            "side": side,
            "tp_dist": float(tp_pts),
            "sl_dist": float(sl_pts),
            "size": int(self.fixed_size),
            "entry_mode": "market_next_bar",
            "horizon_bars": HORIZON_MIN,
            "use_horizon_time_stop": True,
            "sub_strategy": sub_strategy,
            "combo_key": sub_strategy,
            "rule_id": f"h9_gapfade_{sub_strategy}",
            "h9_gap_pct": float(gap_pct),
            "h9_today_open": float(today_open),
            "h9_ml_prob": ml_prob,
            "h9_ml_threshold": ml_thr,
            "h9_n_features": len(ML_FEATURES),
            "confidence": float(min(1.0, abs(gap_pct) / 0.02)),
        }
        self.last_signal_day = day_key
        self.last_signal_time = ts.isoformat()
        LOG.info(
            "H9GapFade SIGNAL %s gap=%.3f%% tp=%.2f sl=%.2f size=%d ml_prob=%s",
            side, gap_pct * 100, tp_pts, sl_pts, self.fixed_size, ml_prob,
        )
        return signal

    # --- Persistence ---
    def state_for_persist(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": bool(self.enabled),
            "use_ml_long": bool(self.use_ml_long),
            "use_ml_short": bool(self.use_ml_short),
            "skip_on_dead_tape": bool(self.skip_on_dead_tape),
            "fixed_size": int(self.fixed_size),
            "circuit_max_consec_losses": int(self.circuit_max_consec_losses),
            "consecutive_losses": int(self.consecutive_losses),
            "circuit_skips": int(self.circuit_skips),
            "last_signal_day": self.last_signal_day,
            "last_signal_time": self.last_signal_time,
            "last_block_reason": self.last_block_reason,
        }

    def restore_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.enabled = bool(payload.get("enabled", self.enabled))
        self.use_ml_long = bool(payload.get("use_ml_long", self.use_ml_long))
        self.use_ml_short = bool(payload.get("use_ml_short", self.use_ml_short))
        self.skip_on_dead_tape = bool(payload.get("skip_on_dead_tape", self.skip_on_dead_tape))
        self.fixed_size = int(payload.get("fixed_size", self.fixed_size))
        self.circuit_max_consec_losses = int(payload.get(
            "circuit_max_consec_losses", self.circuit_max_consec_losses))
        self.consecutive_losses = int(payload.get("consecutive_losses", 0))
        self.circuit_skips = int(payload.get("circuit_skips", 0))
        self.last_signal_day = payload.get("last_signal_day")
        self.last_signal_time = payload.get("last_signal_time")
        self.last_block_reason = payload.get("last_block_reason")

    def reset_for_new_day(self) -> None:
        self.last_signal_day = None
        self.last_signal_time = None
        # NOTE: consecutive_losses persists across days (don't reset here).
        # That's intentional — the 2021 bleed was 4 months of mostly losers.
        # Day-level reset would defeat the circuit; we want it to count
        # losers ACROSS days until a winner resets the streak.

    def record_trade_pnl(self, _close_time, pnl_dollars) -> None:
        """Track consecutive losses for the circuit breaker.

        Called by julie001.py on every closed H9GapFade trade. A loss
        increments the counter; a win or breakeven resets it. The on_bar
        check then uses the counter at the next 09:30 bar to decide
        whether to skip.
        """
        try:
            pnl = float(pnl_dollars)
        except (TypeError, ValueError):
            return
        if pnl < 0:
            self.consecutive_losses += 1
            LOG.info("H9GapFade: trade loss recorded (consec=%d, threshold=%d)",
                     self.consecutive_losses, self.circuit_max_consec_losses)
        else:
            if self.consecutive_losses > 0:
                LOG.info("H9GapFade: trade win/flat resets consec_losses (was=%d)",
                         self.consecutive_losses)
            self.consecutive_losses = 0

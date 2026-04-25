"""
Dynamic Signal Engine 3 - JSON-backed strategy database.
Uses the same signal definition as DynamicEngine (prev candle body vs threshold).
"""

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config import CONFIG
from de3_distance_utils import (
    DEFAULT_DISTANCE_MODE,
    DEFAULT_DISTANCE_ROUNDING,
    convert_distance_pair_to_points,
    distance_mode_uses_percent,
    normalize_distance_mode,
    normalize_distance_rounding,
)


class DynamicSignalEngine3:
    """
    Signal engine with strategies loaded from a JSON file.
    Checks for signals based on:
    - Current session (3-hour blocks in US/Eastern time)
    - Previous candle body vs threshold
    - Strategy type (Long_Rev, Short_Rev, Long_Mom, Short_Mom)
    """

    def __init__(self, db_path: Optional[str] = None):
        self.et_tz = ZoneInfo("America/New_York")
        self.db_settings: dict = {}
        self.db_meta: dict = {}
        self.db_version = "v1"
        self._v2_requested = False
        self._v3_requested = False
        self._v4_requested = False
        self._fallback_db_path: Optional[Path] = None
        self._session_tf_index: Dict[str, Dict[str, list[Dict]]] = {}
        self.db_path = self._resolve_db_path(db_path)
        self.strategies = self._load_strategies()
        self._build_strategy_index()
        self._log_initialization()

    def _resolve_db_path(self, db_path: Optional[str]) -> Path:
        default_v1 = Path(CONFIG.get("DYNAMIC_ENGINE3_DB_FILE", "dynamic_engine3_strategies.json"))
        if db_path is not None:
            self.db_version = "custom"
            self._fallback_db_path = default_v1
            return Path(db_path)

        de3_version = str(CONFIG.get("DE3_VERSION", "v1") or "v1").strip().lower()
        de3_v2_cfg = CONFIG.get("DE3_V2", {}) or {}
        de3_v3_cfg = CONFIG.get("DE3_V3", {}) or {}
        de3_v4_cfg = CONFIG.get("DE3_V4", {}) or {}
        v4_enabled = bool(de3_v4_cfg.get("enabled", False))
        if v4_enabled and de3_version == "v4":
            self._v4_requested = True
            self.db_version = "v4"
            self._fallback_db_path = default_v1
            member_db_path = de3_v4_cfg.get("member_db_path") or de3_v2_cfg.get(
                "db_path",
                "dynamic_engine3_strategies_v2.json",
            )
            return Path(str(member_db_path))
        v3_enabled = bool(de3_v3_cfg.get("enabled", False))
        if v3_enabled and de3_version == "v3":
            self._v3_requested = True
            self.db_version = "v3"
            self._fallback_db_path = default_v1
            member_db_path = de3_v3_cfg.get("member_db_path") or de3_v2_cfg.get(
                "db_path",
                "dynamic_engine3_strategies_v2.json",
            )
            return Path(str(member_db_path))

        v2_enabled = bool(de3_v2_cfg.get("enabled", False))
        if v2_enabled and de3_version == "v2":
            self._v2_requested = True
            self.db_version = "v2"
            self._fallback_db_path = default_v1
            return Path(str(de3_v2_cfg.get("db_path", "dynamic_engine3_strategies_v2.json")))

        self.db_version = "v1"
        self._fallback_db_path = default_v1
        return default_v1

    @staticmethod
    def _load_payload(path: Path) -> Optional[object]:
        if not path.exists():
            # If relative path fails (CWD may have been changed by replay harness),
            # try resolving against sys.path[0] which points to ROOT.
            if not path.is_absolute() and sys.path:
                candidate = Path(sys.path[0]) / path
                if candidate.exists():
                    path = candidate
                else:
                    return None
            else:
                return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_strategies(self) -> list[dict]:
        payload = self._load_payload(self.db_path)
        if payload is None and (self._v2_requested or self._v3_requested or self._v4_requested) and self._fallback_db_path is not None:
            logging.warning(
                "DynamicEngine3 %s DB unavailable (%s). Falling back to v1 DB: %s",
                "v4" if self._v4_requested else ("v3" if self._v3_requested else "v2"),
                self.db_path,
                self._fallback_db_path,
            )
            self.db_path = self._fallback_db_path
            self.db_version = "v1-fallback"
            payload = self._load_payload(self.db_path)

        if payload is None:
            logging.warning("DynamicEngine3 DB not found/unreadable: %s", self.db_path)
            return []

        if isinstance(payload, list):
            strategies = payload
            self.db_settings = {}
            self.db_meta = {}
        elif isinstance(payload, dict):
            strategies = payload.get("strategies") or []
            settings = payload.get("settings")
            self.db_settings = settings if isinstance(settings, dict) else {}
            self.db_meta = payload
            payload_version = str(payload.get("version") or "").strip().lower()
            if payload_version:
                if self.db_version.startswith("v3") or self.db_version.startswith("v4"):
                    self.db_meta["member_payload_version"] = payload_version
                else:
                    self.db_version = payload_version
        else:
            strategies = []
            self.db_settings = {}
            self.db_meta = {}
        filtered = []
        for strat in strategies:
            if not isinstance(strat, dict):
                continue
            if not all(k in strat for k in ("TF", "Session", "Type", "Thresh", "Best_SL", "Best_TP", "Opt_WR")):
                continue
            filtered.append(strat)
        return filtered

    def _build_strategy_index(self) -> None:
        """Index strategies by session/timeframe and precompute static runtime metrics."""
        index: Dict[str, Dict[str, list[Dict]]] = {}
        for strategy in self.strategies:
            try:
                session = str(strategy.get("Session", "") or "")
                timeframe = str(strategy.get("TF", "") or "")
                if not session or not timeframe:
                    continue
                strategy["_runtime_metrics"] = self._runtime_metrics_from_strategy(strategy)
                index.setdefault(session, {}).setdefault(timeframe, []).append(strategy)
            except Exception:
                continue
        self._session_tf_index = index

    def _log_initialization(self) -> None:
        logging.info("=" * 70)
        logging.info("🚀 DYNAMIC SIGNAL ENGINE 3 - JSON STRATEGY DB")
        logging.info("=" * 70)
        logging.info("   DB path: %s", self.db_path)
        logging.info("   DB version: %s", self.db_version)
        logging.info("✅ Loaded %s strategies", len(self.strategies))

        tf_5min = len([s for s in self.strategies if s.get("TF") == "5min"])
        tf_15min = len([s for s in self.strategies if s.get("TF") == "15min"])
        logging.info("   5min strategies: %s", tf_5min)
        logging.info("   15min strategies: %s", tf_15min)
        if self.db_settings:
            logging.info(
                "   DB settings: min_trades=%s min_win_rate=%s min_avg_pnl=%s recent_mode=%s",
                self.db_settings.get("min_trades"),
                self.db_settings.get("min_win_rate"),
                self.db_settings.get("min_avg_pnl"),
                self.db_settings.get("recent_mode"),
            )
        logging.info("=" * 70)

    def get_session_from_time(self, dt_et: datetime) -> str:
        hour = dt_et.hour
        session_start = (hour // 3) * 3
        session_end = session_start + 3
        return f"{session_start:02d}-{session_end:02d}"

    @staticmethod
    def _calculate_body(open_price: float, close_price: float) -> float:
        return close_price - open_price

    @staticmethod
    def _is_green_candle(open_price: float, close_price: float) -> bool:
        return close_price > open_price

    @staticmethod
    def _is_red_candle(open_price: float, close_price: float) -> bool:
        return close_price < open_price

    @staticmethod
    def _fmt_num(value: float) -> str:
        try:
            if abs(value - round(value)) < 1e-6:
                return str(int(round(value)))
        except Exception:
            pass
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _safe_float(value, fallback: float = 0.0) -> float:
        try:
            out = float(value)
            if not math.isfinite(out):
                return float(fallback)
            return out
        except Exception:
            return float(fallback)

    @staticmethod
    def _safe_int(value, fallback: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(fallback)

    @staticmethod
    def _timeframe_minutes(timeframe: str) -> int:
        text = str(timeframe or "").strip().lower()
        if text.endswith("min"):
            text = text[:-3]
        try:
            minutes = int(float(text))
        except Exception:
            minutes = 1
        return max(1, int(minutes))

    @classmethod
    def _horizon_bars(cls, horizon_minutes, timeframe: str) -> int:
        try:
            minutes = int(float(horizon_minutes))
        except Exception:
            return 0
        if minutes <= 0:
            return 0
        return int(math.ceil(minutes / cls._timeframe_minutes(timeframe)))

    def _strategy_id_for_runtime(
        self,
        *,
        timeframe: str,
        session: str,
        strategy_type: str,
        thresh: float,
        raw_sl: float,
        raw_tp: float,
        sl_points: float,
        tp_points: float,
        distance_mode: str,
        horizon_minutes: int,
    ) -> str:
        if distance_mode_uses_percent(distance_mode):
            sl_token = f"SLPCT{self._fmt_num(raw_sl)}"
            tp_token = f"TPPCT{self._fmt_num(raw_tp)}"
        else:
            sl_token = f"SL{self._fmt_num(sl_points)}"
            tp_token = f"TP{self._fmt_num(tp_points)}"
        out = (
            f"{timeframe}_{session}_{strategy_type}"
            f"_T{self._fmt_num(thresh)}_{sl_token}_{tp_token}"
        )
        if int(horizon_minutes or 0) > 0:
            out += f"_HZ{int(horizon_minutes)}"
        return out

    @staticmethod
    def _clip01(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _strategy_trigger_profile_allows(
        self,
        strategy: Dict,
        *,
        abs_body: float,
        thresh: float,
        is_green: bool,
        is_red: bool,
        body_ratio: Optional[float],
        close_pos1: Optional[float],
        upper_wick_ratio: Optional[float],
        lower_wick_ratio: Optional[float],
    ) -> bool:
        family_tag = str(strategy.get("FamilyTag", strategy.get("family_tag", "")) or "").strip()
        if not family_tag:
            return True

        trigger_side = str(
            strategy.get("TriggerCandleSide", strategy.get("trigger_candle_side", ""))
            or ""
        ).strip().lower()
        if trigger_side == "green" and not bool(is_green):
            return False
        if trigger_side == "red" and not bool(is_red):
            return False

        def _metric_in_range(
            *,
            value: Optional[float],
            min_keys: tuple[str, ...],
            max_keys: tuple[str, ...],
        ) -> bool:
            min_bound = None
            max_bound = None
            for key in min_keys:
                if key in strategy:
                    min_bound = self._safe_float(strategy.get(key), float("nan"))
                    break
            for key in max_keys:
                if key in strategy:
                    max_bound = self._safe_float(strategy.get(key), float("nan"))
                    break
            if min_bound is None and max_bound is None:
                return True
            if value is None:
                return False
            if min_bound is not None and math.isfinite(min_bound) and value < float(min_bound):
                return False
            if max_bound is not None and math.isfinite(max_bound) and value > float(max_bound):
                return False
            return True

        body_thresh_ratio = None
        if thresh > 0.0 and math.isfinite(abs_body):
            body_thresh_ratio = float(abs_body / max(1e-9, float(thresh)))

        return all(
            [
                _metric_in_range(
                    value=body_ratio,
                    min_keys=("TriggerMinBodyRatio", "trigger_min_body_ratio"),
                    max_keys=("TriggerMaxBodyRatio", "trigger_max_body_ratio"),
                ),
                _metric_in_range(
                    value=close_pos1,
                    min_keys=("TriggerMinClosePos1", "trigger_min_close_pos1"),
                    max_keys=("TriggerMaxClosePos1", "trigger_max_close_pos1"),
                ),
                _metric_in_range(
                    value=upper_wick_ratio,
                    min_keys=("TriggerMinUpperWickRatio", "trigger_min_upper_wick_ratio"),
                    max_keys=("TriggerMaxUpperWickRatio", "trigger_max_upper_wick_ratio"),
                ),
                _metric_in_range(
                    value=lower_wick_ratio,
                    min_keys=("TriggerMinLowerWickRatio", "trigger_min_lower_wick_ratio"),
                    max_keys=("TriggerMaxLowerWickRatio", "trigger_max_lower_wick_ratio"),
                ),
                _metric_in_range(
                    value=body_thresh_ratio,
                    min_keys=("TriggerMinBodyThreshRatio", "trigger_min_body_thresh_ratio"),
                    max_keys=("TriggerMaxBodyThreshRatio", "trigger_max_body_thresh_ratio"),
                ),
            ]
        )

    def _runtime_metrics_from_strategy(self, strategy: Dict) -> Dict[str, float]:
        score_raw = self._safe_float(strategy.get("Score", strategy.get("Opt_WR", 0.0)), 0.0)
        trades_raw = max(0, self._safe_int(strategy.get("Trades", 0), 0))
        avg_pnl_raw = self._safe_float(strategy.get("Avg_PnL", 0.0), 0.0)
        opt_wr_raw = self._safe_float(strategy.get("Opt_WR", 0.0), 0.0)

        oos = strategy.get("OOS")
        if isinstance(oos, dict):
            oos_trades = max(0, self._safe_int(oos.get("trades", trades_raw), trades_raw))
            oos_wr = self._safe_float(oos.get("win_rate", opt_wr_raw), opt_wr_raw)
            oos_avg = self._safe_float(oos.get("avg_pnl", avg_pnl_raw), avg_pnl_raw)
            oos_pf = self._safe_float(oos.get("profit_factor", 0.0), 0.0)
            oos_stability = self._safe_float(oos.get("stability_score", score_raw), score_raw)
            oos_sharpe = self._safe_float(oos.get("sharpe_like", 0.0), 0.0)
            oos_dd_norm = self._safe_float(oos.get("max_oos_drawdown_norm", 0.0), 0.0)
        else:
            oos_trades = trades_raw
            oos_wr = opt_wr_raw
            oos_avg = avg_pnl_raw
            oos_pf = 0.0
            oos_stability = score_raw
            oos_sharpe = 0.0
            oos_dd_norm = 0.0

        structural_score = self._safe_float(
            strategy.get("StructuralScore", strategy.get("Score", 0.0)),
            self._safe_float(strategy.get("Score", 0.0), 0.0),
        )
        structural_pass = bool(strategy.get("StructuralPass", True))
        profitable_block_ratio = self._safe_float(strategy.get("ProfitableBlockRatio", 0.0), 0.0)
        worst_block_avg_pnl = self._safe_float(strategy.get("WorstBlockAvgPnL", 0.0), 0.0)
        worst_block_pf = self._safe_float(strategy.get("WorstBlockPF", 0.0), 0.0)
        block_avg_pnl_std = self._safe_float(strategy.get("BlockAvgPnLStd", 0.0), 0.0)
        tail_p10_scaled = self._safe_float(strategy.get("TailP10Scaled", 0.0), 0.0)

        # For v2/v3 runtimes, prefer OOS metrics for candidate ranking/gating.
        if (self.db_version.startswith("v2") or self.db_version.startswith("v3") or self.db_version.startswith("v4")) and oos_trades > 0:
            trades_raw = oos_trades
            avg_pnl_raw = oos_avg
            opt_wr_raw = oos_wr
            score_raw = oos_stability

        return {
            "score_raw": float(score_raw),
            "trades_raw": int(trades_raw),
            "avg_pnl_raw": float(avg_pnl_raw),
            "opt_wr_raw": float(opt_wr_raw),
            "oos_trades": int(oos_trades),
            "oos_win_rate": float(oos_wr),
            "oos_avg_pnl": float(oos_avg),
            "oos_profit_factor": float(oos_pf),
            "oos_stability_score": float(oos_stability),
            "oos_sharpe_like": float(oos_sharpe),
            "oos_max_drawdown_norm": float(oos_dd_norm),
            "structural_score": float(structural_score),
            "structural_pass": 1.0 if structural_pass else 0.0,
            "profitable_block_ratio": float(profitable_block_ratio),
            "worst_block_avg_pnl": float(worst_block_avg_pnl),
            "worst_block_pf": float(worst_block_pf),
            "block_avg_pnl_std": float(block_avg_pnl_std),
            "tail_p10_scaled": float(tail_p10_scaled),
        }

    def check_signals(
        self,
        current_time: datetime,
        df_5m,
        df_15m,
        *,
        emit_logs: bool = False,
    ) -> list[Dict]:
        """
        Return all triggered signals for the current session, ranked best->worst.

        This is used by the strategy wrapper to "fall back" to the next-best candidate when a
        specific sub-strategy is blocked by guards, so we don't accidentally zero out DE3's vote.
        """
        if not self.strategies:
            return []
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=self.et_tz)
        else:
            current_time = current_time.astimezone(self.et_tz)

        session = self.get_session_from_time(current_time)

        session_index = self._session_tf_index.get(session)
        if not session_index:
            return []

        runtime_cfg = CONFIG.get("DYNAMIC_ENGINE3_RUNTIME", {}) or {}
        runtime_enabled = bool(runtime_cfg.get("enabled", True))
        use_db_settings = bool(runtime_cfg.get("use_db_settings", True))
        db_recent_gate = bool(runtime_cfg.get("db_recent_gate", True))
        min_trades = max(0, self._safe_int(runtime_cfg.get("min_trades", 0), 0))
        min_score = self._safe_float(runtime_cfg.get("min_score", -1e9), -1e9)
        min_avg_pnl = self._safe_float(runtime_cfg.get("min_avg_pnl", -1e9), -1e9)
        min_win_rate = -1e9
        recent_mode = None
        v4_signal_gate_cfg = {}
        if self._v4_requested:
            de3_v4_cfg = CONFIG.get("DE3_V4", {}) or {}
            runtime_v4_cfg = (
                de3_v4_cfg.get("runtime", {})
                if isinstance(de3_v4_cfg.get("runtime"), dict)
                else {}
            )
            v4_signal_gate_cfg = (
                runtime_v4_cfg.get("signal_gate", {})
                if isinstance(runtime_v4_cfg.get("signal_gate"), dict)
                else {}
            )
            if bool(v4_signal_gate_cfg.get("enabled", False)):
                if "use_runtime_gate" in v4_signal_gate_cfg:
                    runtime_enabled = bool(v4_signal_gate_cfg.get("use_runtime_gate"))
                if "use_db_settings" in v4_signal_gate_cfg:
                    use_db_settings = bool(v4_signal_gate_cfg.get("use_db_settings"))
                if "db_recent_gate" in v4_signal_gate_cfg:
                    db_recent_gate = bool(v4_signal_gate_cfg.get("db_recent_gate"))
        v4_disable_runtime_abstain = bool(
            self._v4_requested
            and bool(v4_signal_gate_cfg.get("enabled", False))
            and bool(v4_signal_gate_cfg.get("disable_runtime_abstain", False))
        )
        if use_db_settings and self.db_settings:
            min_trades = max(0, self._safe_int(self.db_settings.get("min_trades", min_trades), min_trades))
            min_avg_pnl = self._safe_float(self.db_settings.get("min_avg_pnl", min_avg_pnl), min_avg_pnl)
            min_win_rate = self._safe_float(self.db_settings.get("min_win_rate", -1e9), -1e9)
            if db_recent_gate:
                recent_mode_raw = str(self.db_settings.get("recent_mode") or "").strip().lower()
                if recent_mode_raw in {"intersect", "union", "recent_only"}:
                    recent_mode = recent_mode_raw
                elif recent_mode_raw:
                    recent_mode = "intersect"
        if self._v4_requested and bool(v4_signal_gate_cfg.get("enabled", False)):
            min_trades = max(0, self._safe_int(v4_signal_gate_cfg.get("min_trades", min_trades), min_trades))
            min_score = self._safe_float(v4_signal_gate_cfg.get("min_score", min_score), min_score)
            min_avg_pnl = self._safe_float(v4_signal_gate_cfg.get("min_avg_pnl", min_avg_pnl), min_avg_pnl)
            min_win_rate = self._safe_float(v4_signal_gate_cfg.get("min_win_rate", min_win_rate), min_win_rate)
            recent_mode_raw = str(v4_signal_gate_cfg.get("recent_mode", recent_mode or "") or "").strip().lower()
            if recent_mode_raw in {"intersect", "union", "recent_only"}:
                recent_mode = recent_mode_raw
            elif recent_mode_raw in {"none", "off", "disabled"}:
                recent_mode = None
        score_cap = max(1e-6, self._safe_float(runtime_cfg.get("score_cap", 3.0), 3.0))
        avg_pnl_cap = max(1e-6, self._safe_float(runtime_cfg.get("avg_pnl_cap", 3.0), 3.0))
        trades_cap = max(1, self._safe_int(runtime_cfg.get("trades_cap", 1000), 1000))
        wr_floor = self._safe_float(runtime_cfg.get("win_rate_floor", 0.45), 0.45)
        wr_ceil = self._safe_float(runtime_cfg.get("win_rate_ceil", 0.75), 0.75)
        if wr_ceil <= wr_floor:
            wr_ceil = wr_floor + 1e-6
        bucket_score_cap = max(1e-6, self._safe_float(runtime_cfg.get("bucket_score_cap", 2.0), 2.0))

        weights_cfg = runtime_cfg.get("weights", {}) or {}
        w_score = max(0.0, self._safe_float(weights_cfg.get("score", 0.45), 0.45))
        w_wr = max(0.0, self._safe_float(weights_cfg.get("win_rate", 0.25), 0.25))
        w_avg = max(0.0, self._safe_float(weights_cfg.get("avg_pnl", 0.20), 0.20))
        w_trades = max(0.0, self._safe_float(weights_cfg.get("trades", 0.10), 0.10))
        w_bucket = max(0.0, self._safe_float(weights_cfg.get("bucket", 0.15), 0.15))
        w_loc = max(0.0, self._safe_float(weights_cfg.get("location", 0.05), 0.05))
        base_w_sum = w_score + w_wr + w_avg + w_trades
        if base_w_sum <= 0.0:
            w_score, w_wr, w_avg, w_trades = 0.45, 0.25, 0.20, 0.10
            base_w_sum = 1.0

        price_location = 0.5
        try:
            if df_5m is not None and len(df_5m) > 20:
                last_20 = df_5m.iloc[-20:]
                recent_high = last_20["high"].max()
                recent_low = last_20["low"].min()
                current_close = df_5m.iloc[-1]["close"]
                if recent_high > recent_low:
                    price_location = (current_close - recent_low) / (recent_high - recent_low)
                    price_location = max(0.0, min(1.0, price_location))
        except Exception as exc:
            logging.error("DynamicEngine3 price location error: %s", exc)

        triggered_signals: list[Dict] = []

        for timeframe_str, df in [("5min", df_5m), ("15min", df_15m)]:
            if df is None or len(df) < 2:
                continue

            prev_candle = df.iloc[-2]
            current_candle = df.iloc[-1]
            col_map = {col.lower(): col for col in df.columns}
            open_col = col_map.get("open")
            close_col = col_map.get("close")
            high_col = col_map.get("high")
            low_col = col_map.get("low")
            if not open_col or not close_col:
                continue

            prev_open = float(prev_candle[open_col])
            prev_close = float(prev_candle[close_col])
            body = self._calculate_body(prev_open, prev_close)
            abs_body = abs(body)
            is_green = self._is_green_candle(prev_open, prev_close)
            is_red = self._is_red_candle(prev_open, prev_close)
            body_ratio = None
            close_pos1 = None
            upper_wick_ratio = None
            lower_wick_ratio = None
            if high_col and low_col:
                try:
                    prev_high = float(prev_candle[high_col])
                    prev_low = float(prev_candle[low_col])
                    prev_range = float(prev_high - prev_low)
                    if math.isfinite(prev_range) and prev_range > 1e-9:
                        body_ratio = float(abs_body / prev_range)
                        close_pos1 = float((prev_close - prev_low) / prev_range)
                        upper_wick_ratio = float((prev_high - max(prev_open, prev_close)) / prev_range)
                        lower_wick_ratio = float((min(prev_open, prev_close) - prev_low) / prev_range)
                except Exception:
                    body_ratio = None
                    close_pos1 = None
                    upper_wick_ratio = None
                    lower_wick_ratio = None

            tf_strategies = session_index.get(timeframe_str) or []
            if not tf_strategies:
                continue

            for strategy in tf_strategies:
                strategy_type = strategy["Type"]
                thresh = float(strategy["Thresh"])
                signal = None

                if strategy_type == "Long_Rev" and is_red and abs_body > thresh:
                    signal = "LONG"
                elif strategy_type == "Short_Rev" and is_green and abs_body > thresh:
                    signal = "SHORT"
                elif strategy_type == "Long_Mom" and is_green and abs_body > thresh:
                    signal = "LONG"
                elif strategy_type == "Short_Mom" and is_red and abs_body > thresh:
                    signal = "SHORT"

                if signal:
                    if not self._strategy_trigger_profile_allows(
                        strategy,
                        abs_body=abs_body,
                        thresh=thresh,
                        is_green=is_green,
                        is_red=is_red,
                        body_ratio=body_ratio,
                        close_pos1=close_pos1,
                        upper_wick_ratio=upper_wick_ratio,
                        lower_wick_ratio=lower_wick_ratio,
                    ):
                        continue
                    min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
                    min_sl = float(min_cfg.get("sl", 1.25))
                    raw_sl = float(strategy["Best_SL"])
                    raw_tp = float(strategy["Best_TP"])
                    distance_mode = normalize_distance_mode(
                        strategy.get("DistanceMode", strategy.get("distance_mode", DEFAULT_DISTANCE_MODE))
                    )
                    distance_rounding = normalize_distance_rounding(
                        strategy.get(
                            "DistanceRounding",
                            strategy.get("distance_rounding", DEFAULT_DISTANCE_ROUNDING),
                        )
                    )
                    distance_reference_price_strategy = self._safe_float(
                        strategy.get(
                            "DistanceReferencePrice",
                            strategy.get("distance_reference_price", 0.0),
                        ),
                        0.0,
                    )
                    runtime_reference_price = 0.0
                    if distance_mode_uses_percent(distance_mode):
                        runtime_reference_price = self._safe_float(
                            current_candle.get(open_col), 0.0
                        )
                        if runtime_reference_price <= 0.0:
                            runtime_reference_price = self._safe_float(
                                current_candle.get(close_col), 0.0
                            )
                        if runtime_reference_price <= 0.0:
                            runtime_reference_price = distance_reference_price_strategy
                    sl_points, tp_points = convert_distance_pair_to_points(
                        sl_value=raw_sl,
                        tp_value=raw_tp,
                        distance_mode=distance_mode,
                        reference_price=runtime_reference_price
                        if runtime_reference_price > 0.0
                        else distance_reference_price_strategy,
                        tick_size=0.25,
                        rounding=distance_rounding,
                    )
                    if sl_points <= 0.0:
                        sl_points = self._safe_float(
                            strategy.get(
                                "Best_SL_Points_Ref",
                                strategy.get("best_sl_points_ref", raw_sl),
                            ),
                            raw_sl,
                        )
                    if tp_points <= 0.0:
                        tp_points = self._safe_float(
                            strategy.get(
                                "Best_TP_Points_Ref",
                                strategy.get("best_tp_points_ref", raw_tp),
                            ),
                            raw_tp,
                        )
                    sl_value = max(min_sl, float(sl_points))
                    tp_value = float(tp_points)
                    horizon_minutes = self._safe_int(
                        strategy.get("HorizonMinutes", strategy.get("horizon_minutes", 0)),
                        0,
                    )
                    horizon_bars = self._safe_int(
                        strategy.get("HorizonBars", strategy.get("horizon_bars", 0)),
                        0,
                    )
                    if horizon_bars <= 0:
                        horizon_bars = self._horizon_bars(horizon_minutes, timeframe_str)
                    use_horizon_time_stop = bool(
                        strategy.get(
                            "UseHorizonTimeStop",
                            strategy.get("use_horizon_time_stop", horizon_minutes > 0),
                        )
                    )
                    metrics = strategy.get("_runtime_metrics")
                    if not isinstance(metrics, dict):
                        metrics = self._runtime_metrics_from_strategy(strategy)
                        strategy["_runtime_metrics"] = metrics
                    score_raw = float(metrics["score_raw"])
                    trades_raw = int(metrics["trades_raw"])
                    avg_pnl_raw = float(metrics["avg_pnl_raw"])
                    opt_wr_raw = float(metrics["opt_wr_raw"])

                    full_ok = (
                        trades_raw >= min_trades
                        and score_raw >= min_score
                        and avg_pnl_raw >= min_avg_pnl
                        and opt_wr_raw >= min_win_rate
                    )

                    recent_ok = True
                    if recent_mode:
                        recent_ok = False
                        recent = strategy.get("Recent")
                        if isinstance(recent, dict):
                            recent_trades = max(0, self._safe_int(recent.get("trades", 0), 0))
                            recent_wr = self._safe_float(recent.get("win_rate", 0.0), 0.0)
                            recent_avg = self._safe_float(recent.get("avg_pnl", 0.0), 0.0)
                            min_recent_trades = max(5, int(min_trades / 2))
                            recent_ok = (
                                recent_trades >= min_recent_trades
                                and recent_wr >= min_win_rate
                                and recent_avg >= min_avg_pnl
                            )

                    keep_candidate = full_ok
                    if recent_mode == "recent_only":
                        keep_candidate = recent_ok
                    elif recent_mode == "union":
                        keep_candidate = bool(full_ok or recent_ok)
                    elif recent_mode == "intersect":
                        keep_candidate = bool(full_ok and recent_ok)

                    if runtime_enabled:
                        if not keep_candidate:
                            if emit_logs:
                                logging.info(
                                    "DE3 skip %s | trades=%s score=%.3f avg_pnl=%.3f wr=%.3f "
                                    "(mins: trades=%s score=%.3f avg_pnl=%.3f wr=%.3f mode=%s full_ok=%s recent_ok=%s)",
                                    strategy_type,
                                    trades_raw,
                                    score_raw,
                                    avg_pnl_raw,
                                    opt_wr_raw,
                                    min_trades,
                                    min_score,
                                    min_avg_pnl,
                                    min_win_rate,
                                    recent_mode or "none",
                                    full_ok,
                                    recent_ok,
                                )
                            continue
                    triggered_signals.append(
                        {
                            "signal": signal,
                            "sl": sl_value,
                            "tp": tp_value,
                            "opt_wr": opt_wr_raw,
                            "score_raw": score_raw,
                            "trades": trades_raw,
                            "avg_pnl": avg_pnl_raw,
                            "timeframe": timeframe_str,
                            "strategy_type": strategy_type,
                            "thresh": thresh,
                            "body": abs_body,
                            "family_tag": str(
                                strategy.get("FamilyTag", strategy.get("family_tag", "")) or ""
                            ).strip(),
                            "strategy_id": strategy.get(
                                "strategy_id",
                                strategy.get(
                                    "id",
                                    self._strategy_id_for_runtime(
                                        timeframe=timeframe_str,
                                        session=session,
                                        strategy_type=strategy_type,
                                        thresh=thresh,
                                        raw_sl=raw_sl,
                                        raw_tp=raw_tp,
                                        sl_points=sl_value,
                                        tp_points=tp_value,
                                        distance_mode=distance_mode,
                                        horizon_minutes=horizon_minutes,
                                    ),
                                ),
                            ),
                            "distance_mode": distance_mode,
                            "distance_rounding": distance_rounding,
                            "distance_reference_price_strategy": float(
                                distance_reference_price_strategy
                            ),
                            "distance_reference_price_runtime": float(
                                runtime_reference_price
                                if runtime_reference_price > 0.0
                                else distance_reference_price_strategy
                            ),
                            "distance_sl_points_ref": float(
                                self._safe_float(
                                    strategy.get(
                                        "Best_SL_Points_Ref",
                                        strategy.get("best_sl_points_ref", sl_points),
                                    ),
                                    sl_points,
                                )
                            ),
                            "distance_tp_points_ref": float(
                                self._safe_float(
                                    strategy.get(
                                        "Best_TP_Points_Ref",
                                        strategy.get("best_tp_points_ref", tp_points),
                                    ),
                                    tp_points,
                                )
                            ),
                            "horizon_minutes": int(horizon_minutes),
                            "horizon_bars": int(horizon_bars),
                            "use_horizon_time_stop": bool(use_horizon_time_stop),
                            "close_pos1": float(close_pos1) if close_pos1 is not None else None,
                            "body1_ratio": float(body_ratio) if body_ratio is not None else None,
                            "upper_wick_ratio": float(upper_wick_ratio) if upper_wick_ratio is not None else None,
                            "lower_wick_ratio": float(lower_wick_ratio) if lower_wick_ratio is not None else None,
                            "TriggerCandleSide": strategy.get(
                                "TriggerCandleSide",
                                strategy.get("trigger_candle_side", ""),
                            ),
                            "TriggerMinBodyRatio": strategy.get(
                                "TriggerMinBodyRatio",
                                strategy.get("trigger_min_body_ratio", None),
                            ),
                            "TriggerMaxBodyRatio": strategy.get(
                                "TriggerMaxBodyRatio",
                                strategy.get("trigger_max_body_ratio", None),
                            ),
                            "TriggerMinClosePos1": strategy.get(
                                "TriggerMinClosePos1",
                                strategy.get("trigger_min_close_pos1", None),
                            ),
                            "TriggerMaxClosePos1": strategy.get(
                                "TriggerMaxClosePos1",
                                strategy.get("trigger_max_close_pos1", None),
                            ),
                            "TriggerMinUpperWickRatio": strategy.get(
                                "TriggerMinUpperWickRatio",
                                strategy.get("trigger_min_upper_wick_ratio", None),
                            ),
                            "TriggerMaxUpperWickRatio": strategy.get(
                                "TriggerMaxUpperWickRatio",
                                strategy.get("trigger_max_upper_wick_ratio", None),
                            ),
                            "TriggerMinLowerWickRatio": strategy.get(
                                "TriggerMinLowerWickRatio",
                                strategy.get("trigger_min_lower_wick_ratio", None),
                            ),
                            "TriggerMaxLowerWickRatio": strategy.get(
                                "TriggerMaxLowerWickRatio",
                                strategy.get("trigger_max_lower_wick_ratio", None),
                            ),
                            "TriggerMinBodyThreshRatio": strategy.get(
                                "TriggerMinBodyThreshRatio",
                                strategy.get("trigger_min_body_thresh_ratio", None),
                            ),
                            "TriggerMaxBodyThreshRatio": strategy.get(
                                "TriggerMaxBodyThreshRatio",
                                strategy.get("trigger_max_body_thresh_ratio", None),
                            ),
                            "db_recent_mode": recent_mode or "none",
                            "db_recent_ok": bool(recent_ok),
                            "oos_trades": int(metrics["oos_trades"]),
                            "oos_win_rate": float(metrics["oos_win_rate"]),
                            "oos_avg_pnl": float(metrics["oos_avg_pnl"]),
                            "oos_profit_factor": float(metrics["oos_profit_factor"]),
                            "oos_stability_score": float(metrics["oos_stability_score"]),
                            "oos_sharpe_like": float(metrics["oos_sharpe_like"]),
                            "oos_max_drawdown_norm": float(metrics["oos_max_drawdown_norm"]),
                            "StructuralScore": float(metrics["structural_score"]),
                            "StructuralPass": bool(float(metrics["structural_pass"]) >= 0.5),
                            "ProfitableBlockRatio": float(metrics["profitable_block_ratio"]),
                            "WorstBlockAvgPnL": float(metrics["worst_block_avg_pnl"]),
                            "WorstBlockPF": float(metrics["worst_block_pf"]),
                            "BlockAvgPnLStd": float(metrics["block_avg_pnl_std"]),
                            "TailP10Scaled": float(metrics["tail_p10_scaled"]),
                        }
                    )
                    if emit_logs:
                        logging.info(
                            "✅ TRIGGER: %s on %s | Body=%.2f > Thresh=%.2f",
                            strategy_type,
                            timeframe_str,
                            abs_body,
                            thresh,
                        )

        if not triggered_signals:
            return []

        long_count = sum(1 for s in triggered_signals if s["signal"] == "LONG")
        short_count = sum(1 for s in triggered_signals if s["signal"] == "SHORT")

        for sig in triggered_signals:
            score_norm = self._clip01(sig.get("score_raw", 0.0) / score_cap)
            wr_norm = self._clip01((sig["opt_wr"] - wr_floor) / (wr_ceil - wr_floor))
            avg_norm = self._clip01(sig.get("avg_pnl", 0.0) / avg_pnl_cap)
            trades_norm = self._clip01(
                math.log1p(max(0, sig.get("trades", 0))) / math.log1p(max(1, trades_cap))
            )
            quality_base = (
                (w_score * score_norm)
                + (w_wr * wr_norm)
                + (w_avg * avg_norm)
                + (w_trades * trades_norm)
            ) / base_w_sum
            sig["quality_base"] = quality_base
            sig["quality_components"] = {
                "score_norm": score_norm,
                "wr_norm": wr_norm,
                "avg_norm": avg_norm,
                "trades_norm": trades_norm,
            }

        bucket_scores: dict[str, dict[tuple[str, str], float]] = {"LONG": {}, "SHORT": {}}
        for sig in triggered_signals:
            side = sig["signal"]
            bucket_key = (sig["timeframe"], sig["strategy_type"])
            current_best = bucket_scores[side].get(bucket_key, 0.0)
            if sig["quality_base"] > current_best:
                bucket_scores[side][bucket_key] = sig["quality_base"]

        bucket_score_raw = {side: float(sum(bucket_scores[side].values())) for side in bucket_scores}
        bucket_score_norm = {
            side: min(bucket_score_raw[side], bucket_score_cap) / bucket_score_cap for side in bucket_scores
        }

        for sig in triggered_signals:
            bucket_score = (
                bucket_score_norm["LONG"] if sig["signal"] == "LONG" else bucket_score_norm["SHORT"]
            )
            loc_score = (1.0 - price_location) if sig["signal"] == "LONG" else price_location
            quality_base = float(sig.get("quality_base", 0.0))
            final_score = (quality_base * 10.0) + (bucket_score * w_bucket * 10.0) + (loc_score * w_loc * 10.0)
            sig["final_score"] = final_score
            sig["debug_info"] = (
                f"Q:{quality_base:.3f} "
                f"S:{sig['quality_components']['score_norm']:.2f} "
                f"WR:{sig['quality_components']['wr_norm']:.2f} "
                f"A:{sig['quality_components']['avg_norm']:.2f} "
                f"T:{sig['quality_components']['trades_norm']:.2f} "
                f"B:{bucket_score:.2f} "
                f"L:{loc_score:.2f}"
            )

        triggered_signals.sort(key=lambda x: x["final_score"], reverse=True)

        abstain_cfg = runtime_cfg.get("abstain", {}) or {}
        abstain_enabled = bool(abstain_cfg.get("enabled", True))
        if v4_disable_runtime_abstain:
            abstain_enabled = False
        if abstain_enabled and triggered_signals:
            best_signal = triggered_signals[0]
            min_best_score = self._safe_float(abstain_cfg.get("min_best_score", 4.0), 4.0)
            if float(best_signal["final_score"]) < min_best_score:
                if emit_logs:
                    logging.info(
                        "DE3 abstain: best score %.3f < min_best_score %.3f",
                        float(best_signal["final_score"]),
                        min_best_score,
                    )
                return []

            side_best: dict[str, float] = {}
            for sig in triggered_signals:
                side = str(sig.get("signal", "")).upper()
                score_val = float(sig.get("final_score", 0.0))
                if side not in side_best or score_val > side_best[side]:
                    side_best[side] = score_val
            if "LONG" in side_best and "SHORT" in side_best:
                min_side_edge = self._safe_float(abstain_cfg.get("min_side_edge", 0.30), 0.30)
                side_edge = abs(side_best["LONG"] - side_best["SHORT"])
                if side_edge < min_side_edge:
                    if emit_logs:
                        logging.info(
                            "DE3 abstain: side edge %.3f < %.3f (LONG=%.3f SHORT=%.3f)",
                            side_edge,
                            min_side_edge,
                            side_best["LONG"],
                            side_best["SHORT"],
                        )
                    return []

            if len(triggered_signals) > 1:
                top = triggered_signals[0]
                second = triggered_signals[1]
                if str(top.get("signal", "")) != str(second.get("signal", "")):
                    min_top2_gap = self._safe_float(
                        abstain_cfg.get("min_top2_gap_opposite", 0.20), 0.20
                    )
                    top_gap = float(top.get("final_score", 0.0)) - float(second.get("final_score", 0.0))
                    if top_gap < min_top2_gap:
                        if emit_logs:
                            logging.info(
                                "DE3 abstain: top opposite gap %.3f < %.3f (%s vs %s)",
                                top_gap,
                                min_top2_gap,
                                top.get("strategy_id"),
                                second.get("strategy_id"),
                            )
                        return []

        if emit_logs:
            best_signal = triggered_signals[0]
            logging.info(
                "🎯 TIE-BREAK: %s signals. Counts: LONG=%s, SHORT=%s",
                len(triggered_signals),
                long_count,
                short_count,
            )
            logging.info(
                "   BucketScore raw: LONG=%.3f SHORT=%.3f (cap %.2f)",
                bucket_score_raw["LONG"],
                bucket_score_raw["SHORT"],
                bucket_score_cap,
            )
            logging.info("   Price Location: %.2f (0=Low, 1=High)", price_location)
            logging.info(
                "   Winner: %s (Score: %.3f | %s)",
                best_signal["strategy_id"],
                best_signal["final_score"],
                best_signal["debug_info"],
            )
            if len(triggered_signals) > 1:
                logging.info(
                    "   Runner-up: %s (Score: %.3f)",
                    triggered_signals[1]["strategy_id"],
                    triggered_signals[1]["final_score"],
                )

            logging.info("=" * 70)
            logging.info("🚀 FINAL SIGNAL SELECTED")
            logging.info("   Direction: %s", best_signal["signal"])
            logging.info("   Timeframe: %s", best_signal["timeframe"])
            logging.info("   Strategy: %s", best_signal["strategy_type"])
            logging.info("   Stop Loss: %.2f points", best_signal["sl"])
            logging.info("   Take Profit: %.2f points", best_signal["tp"])
            logging.info("   ID: %s", best_signal["strategy_id"])
            logging.info("=" * 70)

        return triggered_signals

    def check_signal(self, current_time: datetime, df_5m, df_15m) -> Optional[Dict]:
        ranked = self.check_signals(current_time, df_5m, df_15m, emit_logs=True)
        return ranked[0] if ranked else None


_engine_instance = None


def reset_signal_engine() -> None:
    """Drop cached singleton so next request reloads DB/config from disk."""
    global _engine_instance
    _engine_instance = None


def get_signal_engine(*, force_reload: bool = False) -> DynamicSignalEngine3:
    global _engine_instance
    if force_reload:
        _engine_instance = None
    if _engine_instance is None:
        _engine_instance = DynamicSignalEngine3()
    return _engine_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = DynamicSignalEngine3()
    logging.info("Loaded %s strategies", len(engine.strategies))

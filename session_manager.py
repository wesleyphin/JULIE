import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
from zoneinfo import ZoneInfo

from config import CONFIG

NY_TZ = ZoneInfo('America/New_York')


class SessionManager:
    """Manages the 4 Neural Networks and switches them based on NY Time."""

    def __init__(self):
        self.thresholds = self._load_json(CONFIG.get("ML_PHYSICS_THRESHOLDS_FILE"))
        self.metrics = self._load_json(CONFIG.get("ML_PHYSICS_METRICS_FILE"))
        self.guard = CONFIG.get("ML_PHYSICS_GUARD", {}) or {}
        self.guard_overrides = CONFIG.get("ML_PHYSICS_GUARD_OVERRIDES", {}) or {}
        self.vol_split = CONFIG.get("ML_PHYSICS_VOL_SPLIT", {}) or {}
        self.disabled_sessions = set()
        self.disabled_regimes = {}
        self.session_thresholds = {}
        self.brains = {}
        self.load_all_brains()

    def _load_json(self, path_value):
        if not path_value:
            return {}
        try:
            path = Path(path_value)
            if not path.exists():
                return {}
            return json.loads(path.read_text())
        except Exception as e:
            logging.warning(f"⚠️ MLPhysics: failed to load {path_value}: {e}")
            return {}

    def load_all_brains(self):
        logging.info("🧠 Initializing Neural Network Array...")
        for name, settings in CONFIG["SESSIONS"].items():
            path = settings["MODEL_FILE"]
            split_enabled = bool(self.vol_split.get("enabled")) and name in set(self.vol_split.get("sessions", []))
            model_low_path = settings.get("MODEL_FILE_LOW")
            model_high_path = settings.get("MODEL_FILE_HIGH")
            if split_enabled and (not model_low_path or not model_high_path):
                if "." in path:
                    base, ext = path.rsplit(".", 1)
                    model_low_path = model_low_path or f"{base}_low.{ext}"
                    model_high_path = model_high_path or f"{base}_high.{ext}"
                else:
                    model_low_path = model_low_path or f"{path}_low"
                    model_high_path = model_high_path or f"{path}_high"
            threshold = settings["THRESHOLD"]
            if name in self.thresholds:
                try:
                    if isinstance(self.thresholds[name], dict) and ("low" in self.thresholds[name] or "high" in self.thresholds[name]):
                        low_thr = self.thresholds[name].get("low", {}).get("threshold", threshold)
                        high_thr = self.thresholds[name].get("high", {}).get("threshold", threshold)
                        self.session_thresholds[name] = {"low": float(low_thr), "high": float(high_thr)}
                    else:
                        threshold = float(self.thresholds[name].get("threshold", threshold))
                        self.session_thresholds[name] = threshold
                except Exception:
                    self.session_thresholds[name] = threshold
            else:
                self.session_thresholds[name] = threshold

            guard_enabled = bool(self.guard.get("enabled", False))
            if guard_enabled and name in self.metrics:
                metrics = self.metrics.get(name) or {}

                if isinstance(metrics, dict) and ("low" in metrics or "high" in metrics):
                    for regime in ("low", "high"):
                        regime_metrics = metrics.get(regime) or {}
                        thresholds = self._guard_thresholds(name, regime=regime)
                        min_trades = thresholds["min_trades"]
                        min_win_rate = thresholds["min_win_rate"]
                        min_avg_pnl = thresholds["min_avg_pnl"]
                        trade_count = float(regime_metrics.get("trade_count", 0))
                        win_rate = float(regime_metrics.get("win_rate", 0.0))
                        avg_pnl = float(regime_metrics.get("avg_pnl", 0.0))
                        if trade_count < min_trades or win_rate < min_win_rate or avg_pnl < min_avg_pnl:
                            self.disabled_regimes.setdefault(name, set()).add(regime)
                            logging.warning(
                                f"⚠️ MLPhysics: {name} {regime} disabled by guard "
                                f"(trades={trade_count:.0f} win_rate={win_rate:.2%} avg_pnl={avg_pnl:.2f})"
                            )
                    if len(self.disabled_regimes.get(name, set())) >= 2:
                        self.disabled_sessions.add(name)
                else:
                    thresholds = self._guard_thresholds(name)
                    min_trades = thresholds["min_trades"]
                    min_win_rate = thresholds["min_win_rate"]
                    min_avg_pnl = thresholds["min_avg_pnl"]
                    trade_count = float(metrics.get("trade_count", 0))
                    win_rate = float(metrics.get("win_rate", 0.0))
                    avg_pnl = float(metrics.get("avg_pnl", 0.0))
                    if trade_count < min_trades or win_rate < min_win_rate or avg_pnl < min_avg_pnl:
                        self.disabled_sessions.add(name)
                        logging.warning(
                            f"⚠️ MLPhysics: {name} disabled by guard "
                            f"(trades={trade_count:.0f} win_rate={win_rate:.2%} avg_pnl={avg_pnl:.2f})"
                        )
            try:
                if split_enabled:
                    low_model = None
                    high_model = None
                    try:
                        low_model = joblib.load(model_low_path)
                        logging.info(f"  ✅ {name} Low-Vol Specialist Loaded ({model_low_path})")
                    except Exception as e:
                        logging.error(f"  ❌ Failed to load {name} low-vol ({model_low_path}): {e}")
                    try:
                        high_model = joblib.load(model_high_path)
                        logging.info(f"  ✅ {name} High-Vol Specialist Loaded ({model_high_path})")
                    except Exception as e:
                        logging.error(f"  ❌ Failed to load {name} high-vol ({model_high_path}): {e}")
                    self.brains[name] = {"low": low_model, "high": high_model}
                    logging.info(f"  ✅ {name} Vol-Split Active")
                else:
                    self.brains[name] = joblib.load(path)
                    logging.info(f"  ✅ {name} Specialist Loaded (Thresh: {threshold:.2f})")
            except Exception as e:
                logging.error(f"  ❌ Failed to load {name} ({path}): {e}")

    def _guard_thresholds(self, session_name: str, regime: Optional[str] = None):
        thresholds = {
            "min_trades": self._safe_int(self.guard.get("min_trades", 0), 0),
            "min_win_rate": self._safe_float(self.guard.get("min_win_rate", 0.0), 0.0),
            "min_avg_pnl": self._safe_float(self.guard.get("min_avg_pnl", -1e9), -1e9),
        }
        session_override = self.guard_overrides.get(session_name) or {}
        thresholds = self._apply_guard_override(thresholds, session_override)
        if regime:
            regime_override = session_override.get(regime) if isinstance(session_override, dict) else None
            if isinstance(regime_override, dict):
                thresholds = self._apply_guard_override(thresholds, regime_override)
        return thresholds

    @staticmethod
    def _apply_guard_override(thresholds: dict, override: dict):
        if not isinstance(override, dict):
            return thresholds
        if "min_trades" in override and override["min_trades"] is not None:
            thresholds["min_trades"] = SessionManager._safe_int(override["min_trades"], thresholds["min_trades"])
        if "min_win_rate" in override and override["min_win_rate"] is not None:
            thresholds["min_win_rate"] = SessionManager._safe_float(override["min_win_rate"], thresholds["min_win_rate"])
        if "min_avg_pnl" in override and override["min_avg_pnl"] is not None:
            thresholds["min_avg_pnl"] = SessionManager._safe_float(override["min_avg_pnl"], thresholds["min_avg_pnl"])
        return thresholds

    @staticmethod
    def _safe_int(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _safe_float(value, fallback):
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def get_current_setup(self, now_override: Optional[datetime.datetime] = None):
        """Returns the active Model, Threshold, SL, and TP for the current minute."""
        now_ny = now_override or datetime.datetime.now(NY_TZ)
        if now_ny.tzinfo is None:
            now_ny = now_ny.replace(tzinfo=NY_TZ)
        else:
            now_ny = now_ny.astimezone(NY_TZ)
        current_hour = now_ny.hour

        for name, settings in CONFIG["SESSIONS"].items():
            if current_hour in settings["HOURS"]:
                timeframe_minutes = settings.get("TIMEFRAME_MINUTES")
                if name in self.disabled_sessions:
                    return {
                        "name": name,
                        "model": None,
                        "threshold": self.session_thresholds.get(name, settings["THRESHOLD"]),
                        "sl": settings["SL"],
                        "tp": settings["TP"],
                        "timeframe_minutes": timeframe_minutes,
                        "disabled": True,
                    }
                model = self.brains.get(name)
                return {
                    "name": name,
                    "model": model,
                    "threshold": self.session_thresholds.get(name, settings["THRESHOLD"]),
                    "sl": settings["SL"],
                    "tp": settings["TP"],
                    "timeframe_minutes": timeframe_minutes,
                    "disabled_regimes": self.disabled_regimes.get(name, set()),
                    "split": isinstance(model, dict),
                }

        return None  # Market Closed or Gap Time

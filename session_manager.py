import datetime
import math
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
        self.wf_guard = CONFIG.get("ML_PHYSICS_WALK_FORWARD_GUARD", {}) or {}
        self.wf_guard_overrides = self.wf_guard.get("sessions", {}) or {}
        self.vol_split = CONFIG.get("ML_PHYSICS_VOL_SPLIT", {}) or {}
        self.vol_split_3way = CONFIG.get("ML_PHYSICS_VOL_SPLIT_3WAY", {}) or {}
        self.manual_disabled_regimes = CONFIG.get("ML_PHYSICS_DISABLED_REGIMES", {}) or {}
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
            split_3way = bool(self.vol_split_3way.get("enabled")) and name in set(self.vol_split_3way.get("sessions", []))
            if split_3way:
                split_enabled = True
            manual_regimes = self.manual_disabled_regimes.get(name)
            if manual_regimes:
                if isinstance(manual_regimes, (list, tuple, set)):
                    regimes = {str(r).lower() for r in manual_regimes}
                else:
                    regimes = {str(manual_regimes).lower()}
                if regimes:
                    self.disabled_regimes.setdefault(name, set()).update(regimes)
                    logging.warning(f"⚠️ MLPhysics: {name} regimes disabled by config: {sorted(regimes)}")
            model_low_path = settings.get("MODEL_FILE_LOW")
            model_high_path = settings.get("MODEL_FILE_HIGH")
            model_normal_path = settings.get("MODEL_FILE_NORMAL")
            if split_enabled:
                if "." in path:
                    base, ext = path.rsplit(".", 1)
                    model_low_path = model_low_path or f"{base}_low.{ext}"
                    model_high_path = model_high_path or f"{base}_high.{ext}"
                    if split_3way:
                        model_normal_path = model_normal_path or f"{base}_normal.{ext}"
                else:
                    model_low_path = model_low_path or f"{path}_low"
                    model_high_path = model_high_path or f"{path}_high"
                    if split_3way:
                        model_normal_path = model_normal_path or f"{path}_normal"
            threshold = settings["THRESHOLD"]
            if name in self.thresholds:
                try:
                    if isinstance(self.thresholds[name], dict):
                        regime_thresholds = {}
                        for regime in ("low", "normal", "high"):
                            if regime in self.thresholds[name]:
                                regime_thresholds[regime] = float(
                                    self.thresholds[name].get(regime, {}).get("threshold", threshold)
                                )
                        if regime_thresholds:
                            if split_3way and "normal" not in regime_thresholds:
                                regime_thresholds["normal"] = regime_thresholds.get("low", threshold)
                            self.session_thresholds[name] = regime_thresholds
                        else:
                            threshold = float(self.thresholds[name].get("threshold", threshold))
                            self.session_thresholds[name] = threshold
                    else:
                        threshold = float(self.thresholds[name].get("threshold", threshold))
                        self.session_thresholds[name] = threshold
                except Exception:
                    self.session_thresholds[name] = threshold
            else:
                self.session_thresholds[name] = threshold

            guard_enabled = bool(self.guard.get("enabled", False))
            metrics = self.metrics.get(name) or {}
            if guard_enabled and metrics:
                if isinstance(metrics, dict):
                    regime_keys = [reg for reg in ("low", "normal", "high") if reg in metrics]
                    if regime_keys:
                        for regime in regime_keys:
                            regime_metrics = metrics.get(regime) or {}
                            thresholds = self._guard_thresholds(name, regime=regime)
                            min_trades = thresholds["min_trades"]
                            min_win_rate = thresholds["min_win_rate"]
                            min_avg_pnl = thresholds["min_avg_pnl"]
                            trade_count = float(regime_metrics.get("trade_count", 0))
                            win_rate = float(regime_metrics.get("win_rate", 0.0))
                            avg_pnl = float(regime_metrics.get("avg_pnl", 0.0))
                            if (
                                trade_count < min_trades
                                or not self._win_rate_meets(min_win_rate, win_rate)
                                or avg_pnl < min_avg_pnl
                            ):
                                self.disabled_regimes.setdefault(name, set()).add(regime)
                                logging.warning(
                                    f"⚠️ MLPhysics: {name} {regime} disabled by guard "
                                    f"(trades={trade_count:.0f} win_rate={win_rate:.2%} avg_pnl={avg_pnl:.2f})"
                                )
                        if split_3way:
                            required = 3
                        else:
                            required = 2 if split_enabled else 1
                        if len(self.disabled_regimes.get(name, set())) >= required:
                            self.disabled_sessions.add(name)
                            logging.warning(f"⚠️ MLPhysics: {name} disabled by guard (all regimes)")
                    else:
                        thresholds = self._guard_thresholds(name)
                        min_trades = thresholds["min_trades"]
                        min_win_rate = thresholds["min_win_rate"]
                        min_avg_pnl = thresholds["min_avg_pnl"]
                        trade_count = float(metrics.get("trade_count", 0))
                        win_rate = float(metrics.get("win_rate", 0.0))
                        avg_pnl = float(metrics.get("avg_pnl", 0.0))
                        if (
                            trade_count < min_trades
                            or not self._win_rate_meets(min_win_rate, win_rate)
                            or avg_pnl < min_avg_pnl
                        ):
                            self.disabled_sessions.add(name)
                            logging.warning(
                                f"⚠️ MLPhysics: {name} disabled by guard "
                                f"(trades={trade_count:.0f} win_rate={win_rate:.2%} avg_pnl={avg_pnl:.2f})"
                            )
                else:
                    thresholds = self._guard_thresholds(name)
                    min_trades = thresholds["min_trades"]
                    min_win_rate = thresholds["min_win_rate"]
                    min_avg_pnl = thresholds["min_avg_pnl"]
                    trade_count = float(metrics.get("trade_count", 0))
                    win_rate = float(metrics.get("win_rate", 0.0))
                    avg_pnl = float(metrics.get("avg_pnl", 0.0))
                    if (
                        trade_count < min_trades
                        or not self._win_rate_meets(min_win_rate, win_rate)
                        or avg_pnl < min_avg_pnl
                    ):
                        self.disabled_sessions.add(name)
                        logging.warning(
                            f"⚠️ MLPhysics: {name} disabled by guard "
                            f"(trades={trade_count:.0f} win_rate={win_rate:.2%} avg_pnl={avg_pnl:.2f})"
                        )

            wf_guard_enabled = bool(self.wf_guard.get("enabled", False))
            if wf_guard_enabled:
                if isinstance(metrics, dict):
                    regime_keys = [reg for reg in ("low", "normal", "high") if reg in metrics]
                    if regime_keys:
                        for regime in regime_keys:
                            regime_metrics = metrics.get(regime) or {}
                            ok, reason = self._check_walk_forward_guard(name, regime_metrics, regime)
                            if not ok:
                                self.disabled_regimes.setdefault(name, set()).add(regime)
                                logging.warning(
                                    f"⚠️ MLPhysics: {name} {regime} disabled by walk-forward guard "
                                    f"({reason})"
                                )
                        if split_3way:
                            required = 3
                        else:
                            required = 2 if split_enabled else 1
                        if len(self.disabled_regimes.get(name, set())) >= required:
                            self.disabled_sessions.add(name)
                            logging.warning(f"⚠️ MLPhysics: {name} disabled by walk-forward guard (all regimes)")
                    else:
                        ok, reason = self._check_walk_forward_guard(name, metrics)
                        if not ok:
                            self.disabled_sessions.add(name)
                            logging.warning(
                                f"⚠️ MLPhysics: {name} disabled by walk-forward guard ({reason})"
                            )
                else:
                    ok, reason = self._check_walk_forward_guard(name, metrics)
                    if not ok:
                        self.disabled_sessions.add(name)
                        logging.warning(
                            f"⚠️ MLPhysics: {name} disabled by walk-forward guard ({reason})"
                        )
            try:
                if split_enabled:
                    low_model = None
                    normal_model = None
                    high_model = None
                    try:
                        low_model = joblib.load(model_low_path)
                        logging.info(f"  ✅ {name} Low-Vol Specialist Loaded ({model_low_path})")
                    except Exception as e:
                        logging.error(f"  ❌ Failed to load {name} low-vol ({model_low_path}): {e}")
                    if split_3way:
                        if model_normal_path:
                            try:
                                normal_model = joblib.load(model_normal_path)
                                logging.info(f"  ✅ {name} Normal-Vol Specialist Loaded ({model_normal_path})")
                            except Exception as e:
                                logging.error(f"  ❌ Failed to load {name} normal-vol ({model_normal_path}): {e}")
                        else:
                            logging.warning(f"  ⚠️ {name} normal-vol model path missing for 3-way split")
                    try:
                        high_model = joblib.load(model_high_path)
                        logging.info(f"  ✅ {name} High-Vol Specialist Loaded ({model_high_path})")
                    except Exception as e:
                        logging.error(f"  ❌ Failed to load {name} high-vol ({model_high_path}): {e}")
                    if split_3way:
                        self.brains[name] = {"low": low_model, "normal": normal_model, "high": high_model}
                    else:
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
    def _win_rate_meets(min_win_rate: float, win_rate: float) -> bool:
        # Compare win rate using ceiling on percentage points (e.g., 54.28 -> 55)
        min_pct = min_win_rate * 100.0 if min_win_rate <= 1 else min_win_rate
        win_pct = win_rate * 100.0 if win_rate <= 1 else win_rate
        return math.ceil(win_pct) >= min_pct

    def _wf_guard_settings(self, session_name: str, regime: Optional[str] = None) -> dict:
        settings = {
            "enabled": bool(self.wf_guard.get("enabled", False)),
            "require": bool(self.wf_guard.get("require", True)),
            "min_folds": self._safe_int(self.wf_guard.get("min_folds", 0), 0),
            "min_positive_folds": self._safe_int(self.wf_guard.get("min_positive_folds", 0), 0),
            "min_positive_ratio": self._safe_float(self.wf_guard.get("min_positive_ratio", 0.0), 0.0),
            "min_fold_avg_pnl": self._safe_float(self.wf_guard.get("min_fold_avg_pnl", 0.0), 0.0),
            "min_fold_trades": self._safe_int(self.wf_guard.get("min_fold_trades", 0), 0),
        }
        session_override = self.wf_guard_overrides.get(session_name) or {}
        if isinstance(session_override, dict):
            settings = self._apply_wf_override(settings, session_override)
            if regime:
                regime_override = session_override.get(regime) if isinstance(session_override, dict) else None
                if isinstance(regime_override, dict):
                    settings = self._apply_wf_override(settings, regime_override)
        return settings

    @staticmethod
    def _apply_wf_override(settings: dict, override: dict) -> dict:
        if not isinstance(override, dict):
            return settings
        for key in ("enabled", "require"):
            if key in override and override[key] is not None:
                settings[key] = bool(override[key])
        if "min_folds" in override and override["min_folds"] is not None:
            settings["min_folds"] = SessionManager._safe_int(override["min_folds"], settings["min_folds"])
        if "min_positive_folds" in override and override["min_positive_folds"] is not None:
            settings["min_positive_folds"] = SessionManager._safe_int(
                override["min_positive_folds"], settings["min_positive_folds"]
            )
        if "min_positive_ratio" in override and override["min_positive_ratio"] is not None:
            settings["min_positive_ratio"] = SessionManager._safe_float(
                override["min_positive_ratio"], settings["min_positive_ratio"]
            )
        if "min_fold_avg_pnl" in override and override["min_fold_avg_pnl"] is not None:
            settings["min_fold_avg_pnl"] = SessionManager._safe_float(
                override["min_fold_avg_pnl"], settings["min_fold_avg_pnl"]
            )
        if "min_fold_trades" in override and override["min_fold_trades"] is not None:
            settings["min_fold_trades"] = SessionManager._safe_int(
                override["min_fold_trades"], settings["min_fold_trades"]
            )
        return settings

    def _check_walk_forward_guard(
        self,
        session_name: str,
        metrics: dict,
        regime: Optional[str] = None,
    ) -> tuple[bool, str]:
        settings = self._wf_guard_settings(session_name, regime=regime)
        if not settings.get("enabled", False):
            return True, "disabled"

        wf = {}
        if isinstance(metrics, dict):
            wf = metrics.get("walk_forward") or {}

        if not wf:
            if settings.get("require", True):
                return False, "walk_forward missing"
            return True, "walk_forward missing (ignored)"

        folds_detail = wf.get("folds_detail")
        min_folds = settings.get("min_folds", 0)
        min_positive_folds = settings.get("min_positive_folds", 0)
        min_positive_ratio = settings.get("min_positive_ratio", 0.0)
        min_fold_avg_pnl = settings.get("min_fold_avg_pnl", 0.0)
        min_fold_trades = settings.get("min_fold_trades", 0)

        effective_folds = 0
        positive_folds = 0
        if isinstance(folds_detail, list) and folds_detail:
            for fold in folds_detail:
                try:
                    trade_count = int(fold.get("trade_count", 0))
                except Exception:
                    trade_count = 0
                if trade_count < min_fold_trades:
                    continue
                effective_folds += 1
                try:
                    avg_pnl = float(fold.get("avg_pnl", 0.0))
                except Exception:
                    avg_pnl = 0.0
                if avg_pnl >= min_fold_avg_pnl:
                    positive_folds += 1
        else:
            try:
                effective_folds = int(wf.get("folds", 0) or 0)
            except Exception:
                effective_folds = 0
            try:
                positive_folds = int(wf.get("positive_folds", 0) or 0)
            except Exception:
                positive_folds = 0
            if positive_folds <= 0 and effective_folds > 0:
                try:
                    positive_ratio = float(wf.get("positive_ratio", 0.0) or 0.0)
                except Exception:
                    positive_ratio = 0.0
                positive_folds = int(round(positive_ratio * effective_folds))

        positive_ratio = float(positive_folds) / float(effective_folds) if effective_folds else 0.0

        if effective_folds < min_folds:
            return False, f"folds={effective_folds} < {min_folds}"
        if positive_folds < min_positive_folds:
            return False, f"positive_folds={positive_folds} < {min_positive_folds}"
        if positive_ratio < min_positive_ratio:
            return False, f"positive_ratio={positive_ratio:.2f} < {min_positive_ratio:.2f}"
        return True, f"folds={effective_folds} positive={positive_folds} ratio={positive_ratio:.2f}"

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
    @staticmethod
    def _collapse_threshold(value, fallback):
        try:
            values = []
            for key in ("low", "normal", "high"):
                if key in value:
                    values.append(float(value.get(key, fallback)))
            if not values:
                return fallback
            return max(values)
        except Exception:
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
                threshold_value = self.session_thresholds.get(name, settings["THRESHOLD"])
                if name in self.disabled_sessions:
                    if isinstance(threshold_value, dict):
                        threshold_value = self._collapse_threshold(threshold_value, settings["THRESHOLD"])
                    return {
                        "name": name,
                        "model": None,
                        "threshold": threshold_value,
                        "sl": settings["SL"],
                        "tp": settings["TP"],
                        "timeframe_minutes": timeframe_minutes,
                        "disabled": True,
                    }
                model = self.brains.get(name)
                split = isinstance(model, dict)
                split_regimes = []
                if split:
                    split_regimes = [reg for reg in ("low", "normal", "high") if reg in model]
                if isinstance(threshold_value, dict) and not split:
                    threshold_value = self._collapse_threshold(threshold_value, settings["THRESHOLD"])
                return {
                    "name": name,
                    "model": model,
                    "threshold": threshold_value,
                    "sl": settings["SL"],
                    "tp": settings["TP"],
                    "timeframe_minutes": timeframe_minutes,
                    "disabled_regimes": self.disabled_regimes.get(name, set()),
                    "split": split,
                    "split_regimes": split_regimes,
                }

        return None  # Market Closed or Gap Time

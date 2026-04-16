import datetime
import math
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from zoneinfo import ZoneInfo

from config import CONFIG

NY_TZ = ZoneInfo('America/New_York')

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


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
        self.gate_brains = {}
        self.ev_cfg = CONFIG.get("ML_PHYSICS_EV_MODELS", {}) or {}
        self.ev_models_enabled = bool(self.ev_cfg.get("enabled", True))
        self.ev_brains = {}
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
                        if self._is_regime_thresholds(self.thresholds[name]):
                            regime_thresholds = {}
                            for regime in ("low", "normal", "high"):
                                if regime in self.thresholds[name]:
                                    regime_thresholds[regime] = self._normalize_threshold_entry(
                                        self.thresholds[name].get(regime),
                                        threshold,
                                    )
                            if regime_thresholds:
                                global_gate_threshold = None
                                try:
                                    global_gate_threshold = self.thresholds[name].get("gate_threshold")
                                except Exception:
                                    global_gate_threshold = None
                                if global_gate_threshold is not None:
                                    for reg_key, reg_entry in regime_thresholds.items():
                                        if isinstance(reg_entry, dict) and "gate_threshold" not in reg_entry:
                                            reg_entry["gate_threshold"] = global_gate_threshold
                                            regime_thresholds[reg_key] = reg_entry
                                if split_3way and "normal" not in regime_thresholds:
                                    regime_thresholds["normal"] = regime_thresholds.get("low", threshold)
                                self.session_thresholds[name] = regime_thresholds
                            else:
                                self.session_thresholds[name] = threshold
                        else:
                            self.session_thresholds[name] = self._normalize_threshold_entry(
                                self.thresholds[name],
                                threshold,
                            )
                    else:
                        threshold = float(self.thresholds[name].get("threshold", threshold))
                        self.session_thresholds[name] = threshold
                except Exception:
                    self.session_thresholds[name] = threshold
            else:
                self.session_thresholds[name] = threshold

            gate_model_path = settings.get("MODEL_FILE_GATE")
            if not gate_model_path:
                if "." in path:
                    base, ext = path.rsplit(".", 1)
                    gate_model_path = f"{base}_gate.{ext}"
                else:
                    gate_model_path = f"{path}_gate"

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
                        self.ev_brains[name] = {
                            "low": self._load_ev_models(model_low_path, name, "low"),
                            "normal": self._load_ev_models(model_normal_path, name, "normal"),
                            "high": self._load_ev_models(model_high_path, name, "high"),
                        }
                    else:
                        self.brains[name] = {"low": low_model, "high": high_model}
                        self.ev_brains[name] = {
                            "low": self._load_ev_models(model_low_path, name, "low"),
                            "high": self._load_ev_models(model_high_path, name, "high"),
                        }
                    logging.info(f"  ✅ {name} Vol-Split Active")
                else:
                    self.brains[name] = joblib.load(path)
                    self.ev_brains[name] = self._load_ev_models(path, name, None)
                    logging.info(f"  ✅ {name} Specialist Loaded (Thresh: {threshold:.2f})")
                gate_model = None
                try:
                    gate_path_obj = Path(gate_model_path)
                    if gate_path_obj.exists():
                        gate_model = joblib.load(gate_model_path)
                        logging.info(f"  ✅ {name} Tradeability Gate Loaded ({gate_model_path})")
                except Exception as e:
                    logging.error(f"  ❌ Failed to load {name} gate ({gate_model_path}): {e}")
                self.gate_brains[name] = gate_model
            except Exception as e:
                logging.error(f"  ❌ Failed to load {name} ({path}): {e}")
                self.gate_brains[name] = None
                self.ev_brains[name] = None

    @staticmethod
    def _derive_ev_model_paths(model_path) -> tuple[Path, Path]:
        model_path = Path(model_path)
        if model_path.suffix:
            return (
                model_path.with_name(f"{model_path.stem}_ev_long{model_path.suffix}"),
                model_path.with_name(f"{model_path.stem}_ev_short{model_path.suffix}"),
            )
        return Path(f"{model_path}_ev_long"), Path(f"{model_path}_ev_short")

    def _load_ev_models(self, model_path, session_name: str, regime: Optional[str]):
        payload = {"long": None, "short": None}
        if not self.ev_models_enabled or not model_path:
            return payload
        long_path, short_path = self._derive_ev_model_paths(model_path)
        for side, path_obj in (("long", long_path), ("short", short_path)):
            try:
                if path_obj.exists():
                    payload[side] = joblib.load(path_obj)
                    if regime:
                        logging.info(f"  ✅ {session_name} {regime.upper()} EV-{side.upper()} Loaded ({path_obj})")
                    else:
                        logging.info(f"  ✅ {session_name} EV-{side.upper()} Loaded ({path_obj})")
            except Exception as e:
                if regime:
                    logging.error(f"  ❌ Failed to load {session_name} {regime} EV-{side} ({path_obj}): {e}")
                else:
                    logging.error(f"  ❌ Failed to load {session_name} EV-{side} ({path_obj}): {e}")
        return payload

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
            "min_mean_fold_ev": self._safe_float(self.wf_guard.get("min_mean_fold_ev", -1e9), -1e9),
            "min_worst_fold_ev": self._safe_float(self.wf_guard.get("min_worst_fold_ev", -1e9), -1e9),
            "min_objective_score": self._safe_float(self.wf_guard.get("min_objective_score", -1e9), -1e9),
            "objective_mean_weight": self._safe_float(self.wf_guard.get("objective_mean_weight", 0.65), 0.65),
            "objective_worst_weight": self._safe_float(self.wf_guard.get("objective_worst_weight", 0.35), 0.35),
            "objective_std_penalty": self._safe_float(self.wf_guard.get("objective_std_penalty", 0.10), 0.10),
            "objective_min_fold_trades": self._safe_int(
                self.wf_guard.get("objective_min_fold_trades", self.wf_guard.get("min_fold_trades", 0)),
                0,
            ),
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
        int_keys = ("min_folds", "min_positive_folds", "min_fold_trades", "objective_min_fold_trades")
        float_keys = (
            "min_positive_ratio",
            "min_fold_avg_pnl",
            "min_mean_fold_ev",
            "min_worst_fold_ev",
            "min_objective_score",
            "objective_mean_weight",
            "objective_worst_weight",
            "objective_std_penalty",
        )
        for key in int_keys:
            if key in override and override[key] is not None:
                settings[key] = SessionManager._safe_int(override[key], settings[key])
        for key in float_keys:
            if key in override and override[key] is not None:
                settings[key] = SessionManager._safe_float(override[key], settings[key])
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
        min_mean_fold_ev = settings.get("min_mean_fold_ev", -1e9)
        min_worst_fold_ev = settings.get("min_worst_fold_ev", -1e9)
        min_objective_score = settings.get("min_objective_score", -1e9)
        objective_mean_weight = settings.get("objective_mean_weight", 0.65)
        objective_worst_weight = settings.get("objective_worst_weight", 0.35)
        objective_std_penalty = settings.get("objective_std_penalty", 0.10)
        objective_min_fold_trades = settings.get("objective_min_fold_trades", min_fold_trades)

        try:
            w_sum = float(objective_mean_weight) + float(objective_worst_weight)
        except Exception:
            w_sum = 0.0
        if w_sum <= 0:
            objective_mean_weight = 0.65
            objective_worst_weight = 0.35
        else:
            objective_mean_weight = float(objective_mean_weight) / float(w_sum)
            objective_worst_weight = float(objective_worst_weight) / float(w_sum)

        effective_folds = 0
        positive_folds = 0
        fold_evs = []
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
                try:
                    if trade_count >= objective_min_fold_trades and math.isfinite(avg_pnl):
                        fold_evs.append(avg_pnl)
                except Exception:
                    pass
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

        def _safe_float(value, fallback):
            try:
                out = float(value)
                if not math.isfinite(out):
                    return fallback
                return out
            except Exception:
                return fallback

        mean_fold_ev = _safe_float(wf.get("mean_fold_ev"), float("nan"))
        worst_fold_ev = _safe_float(wf.get("worst_fold_ev"), float("nan"))
        fold_ev_std = _safe_float(wf.get("fold_ev_std"), float("nan"))
        objective_score = _safe_float(wf.get("objective_score"), float("nan"))

        if (not math.isfinite(mean_fold_ev)) or (not math.isfinite(worst_fold_ev)):
            if not fold_evs and isinstance(folds_detail, list):
                for fold in folds_detail:
                    try:
                        trade_count = int(fold.get("trade_count", 0) or 0)
                    except Exception:
                        trade_count = 0
                    if trade_count <= 0:
                        continue
                    try:
                        avg_pnl = float(fold.get("avg_pnl", 0.0) or 0.0)
                    except Exception:
                        avg_pnl = 0.0
                    if math.isfinite(avg_pnl):
                        fold_evs.append(avg_pnl)
            if fold_evs:
                mean_fold_ev = float(sum(fold_evs) / float(len(fold_evs)))
                worst_fold_ev = float(min(fold_evs))
                fold_ev_std = float(np.std(fold_evs))

        if not math.isfinite(mean_fold_ev):
            mean_fold_ev = _safe_float(wf.get("avg_pnl"), 0.0)
        if not math.isfinite(worst_fold_ev):
            worst_fold_ev = mean_fold_ev
        if not math.isfinite(fold_ev_std):
            fold_ev_std = 0.0
        if not math.isfinite(objective_score):
            objective_score = (
                float(objective_mean_weight) * float(mean_fold_ev)
                + float(objective_worst_weight) * float(worst_fold_ev)
                - float(objective_std_penalty) * float(fold_ev_std)
            )

        if effective_folds < min_folds:
            return False, f"folds={effective_folds} < {min_folds}"
        if positive_folds < min_positive_folds:
            return False, f"positive_folds={positive_folds} < {min_positive_folds}"
        if positive_ratio < min_positive_ratio:
            return False, f"positive_ratio={positive_ratio:.2f} < {min_positive_ratio:.2f}"
        if mean_fold_ev < min_mean_fold_ev:
            return False, f"mean_fold_ev={mean_fold_ev:.4f} < {min_mean_fold_ev:.4f}"
        if worst_fold_ev < min_worst_fold_ev:
            return False, f"worst_fold_ev={worst_fold_ev:.4f} < {min_worst_fold_ev:.4f}"
        if objective_score < min_objective_score:
            return False, f"objective_score={objective_score:.4f} < {min_objective_score:.4f}"
        return True, (
            f"folds={effective_folds} positive={positive_folds} ratio={positive_ratio:.2f} "
            f"mean_ev={mean_fold_ev:.4f} worst_ev={worst_fold_ev:.4f} obj={objective_score:.4f}"
        )

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
    def _is_regime_thresholds(value) -> bool:
        if not isinstance(value, dict):
            return False
        return any(key in value for key in ("low", "normal", "high"))

    @staticmethod
    def _normalize_threshold_entry(entry, fallback):
        if isinstance(entry, dict):
            if "threshold" in entry:
                try:
                    threshold = float(entry.get("threshold", fallback))
                except Exception:
                    threshold = fallback
                normalized = {"threshold": threshold}
                if "short_threshold" in entry and entry["short_threshold"] is not None:
                    try:
                        normalized["short_threshold"] = float(entry["short_threshold"])
                    except Exception:
                        pass
                if "policy" in entry and entry["policy"] is not None:
                    normalized["policy"] = entry["policy"]
                for extra_key in (
                    "context",
                    "context_thresholds",
                    "fallback",
                    "hierarchy",
                    "trade_budget",
                    "gate_threshold",
                    "gate",
                    "ev_runtime",
                ):
                    if extra_key in entry:
                        normalized[extra_key] = entry.get(extra_key)
                return normalized
        try:
            return float(entry)
        except Exception:
            return fallback
    @staticmethod
    def _collapse_threshold(value, fallback):
        try:
            if isinstance(value, dict) and "threshold" in value:
                return value
            values = []
            for key in ("low", "normal", "high"):
                if key in value:
                    entry = value.get(key, fallback)
                    if isinstance(entry, dict):
                        entry = entry.get("threshold", fallback)
                    values.append(float(entry))
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
                        if self._is_regime_thresholds(threshold_value):
                            threshold_value = self._collapse_threshold(threshold_value, settings["THRESHOLD"])
                    return {
                        "name": name,
                        "model": None,
                        "ev_models": self.ev_brains.get(name),
                        "gate_model": self.gate_brains.get(name),
                        "gate_threshold": None,
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
                    if self._is_regime_thresholds(threshold_value):
                        threshold_value = self._collapse_threshold(threshold_value, settings["THRESHOLD"])
                gate_threshold = None
                if isinstance(threshold_value, dict):
                    try:
                        gt = threshold_value.get("gate_threshold")
                        gate_threshold = None if gt is None else float(gt)
                    except Exception:
                        gate_threshold = None
                return {
                    "name": name,
                    "model": model,
                    "ev_models": self.ev_brains.get(name),
                    "gate_model": self.gate_brains.get(name),
                    "gate_threshold": gate_threshold,
                    "threshold": threshold_value,
                    "sl": settings["SL"],
                    "tp": settings["TP"],
                    "timeframe_minutes": timeframe_minutes,
                    "disabled_regimes": self.disabled_regimes.get(name, set()),
                    "split": split,
                    "split_regimes": split_regimes,
                }

        return None  # Market Closed or Gap Time

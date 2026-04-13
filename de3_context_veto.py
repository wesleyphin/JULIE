import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class _VetoModel:
    coef: np.ndarray
    intercept: float
    calibration: dict


@dataclass
class _BucketModels:
    models: list[_VetoModel]
    n_samples: int
    n_loss: int
    n_win: int
    level: str


class DE3ContextVeto:
    """Loss-probability veto for DynamicEngine3 candidates with confidence metadata."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.feature_order: list[str] = ["atr_ratio", "price_location", "vwap_dist_atr", "sl_atr"]
        self.bucket_schema: list[str] = ["session_bucket", "timeframe", "strategy_type", "vol_regime", "thresh_bucket"]
        self.threshold: float = 0.5
        self.bucket_models: Dict[str, _BucketModels] = {}
        self.ready = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            logging.warning("DE3 veto models missing: %s", self.model_path)
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("DE3 veto models failed to load: %s", exc)
            return
        self.feature_order = payload.get("feature_order", self.feature_order) or self.feature_order
        self.bucket_schema = payload.get("bucket_schema", self.bucket_schema) or self.bucket_schema
        try:
            self.threshold = float(payload.get("threshold", self.threshold))
        except Exception:
            self.threshold = 0.5
        bucket_models = payload.get("bucket_models", {}) or {}
        for bucket_key, bucket_payload in bucket_models.items():
            models = []
            for entry in bucket_payload.get("models", []) or []:
                coef = np.asarray(entry.get("coef", []), dtype="float64")
                try:
                    intercept = float(entry.get("intercept", 0.0))
                except Exception:
                    intercept = 0.0
                calib = entry.get("calibration", {}) or {}
                models.append(_VetoModel(coef=coef, intercept=intercept, calibration=calib))
            if not models:
                continue
            try:
                n_samples = int(bucket_payload.get("n_samples", 0) or 0)
            except Exception:
                n_samples = 0
            try:
                n_loss = int(bucket_payload.get("n_loss", 0) or 0)
            except Exception:
                n_loss = 0
            try:
                n_win = int(bucket_payload.get("n_win", 0) or 0)
            except Exception:
                n_win = 0
            level = str(bucket_payload.get("level", "") or "")
            self.bucket_models[bucket_key] = _BucketModels(
                models=models,
                n_samples=n_samples,
                n_loss=n_loss,
                n_win=n_win,
                level=level,
            )
        self.ready = bool(self.bucket_models)
        if self.ready:
            logging.info("DE3 veto models loaded: %s buckets", len(self.bucket_models))

    @staticmethod
    def _sigmoid(z: float) -> float:
        try:
            return float(1.0 / (1.0 + np.exp(-z)))
        except Exception:
            return 0.5

    def _apply_calibration(self, score: float, calib: dict) -> float:
        method = str(calib.get("method", "sigmoid") or "sigmoid").lower()
        if method == "isotonic":
            x = calib.get("x_thresholds")
            y = calib.get("y_thresholds")
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and len(x) == len(y) and len(x) > 1:
                try:
                    return float(np.interp(score, np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")))
                except Exception:
                    return self._sigmoid(score)
        if method == "sigmoid":
            if "a" in calib and "b" in calib:
                try:
                    a = float(calib.get("a", 0.0))
                    b = float(calib.get("b", 0.0))
                    return float(1.0 / (1.0 + np.exp(a * score + b)))
                except Exception:
                    return self._sigmoid(score)
        return self._sigmoid(score)

    @staticmethod
    def _normalize_key_list(bucket_key: str | Sequence[str]) -> list[str]:
        if isinstance(bucket_key, str):
            return [bucket_key]
        out: list[str] = []
        for item in bucket_key:
            if item is None:
                continue
            key = str(item).strip()
            if key:
                out.append(key)
        return out

    @staticmethod
    def _clip01(value: float) -> float:
        try:
            return float(max(0.0, min(1.0, value)))
        except Exception:
            return 0.0

    def predict_loss(self, bucket_key: str | Sequence[str], features: Sequence[float]) -> Optional[dict]:
        if not self.ready:
            return None
        keys = self._normalize_key_list(bucket_key)
        if not keys:
            return None
        bucket = None
        selected_key = None
        for key in keys:
            bucket = self.bucket_models.get(key)
            if bucket is not None:
                selected_key = key
                break
        if bucket is None or selected_key is None:
            return None
        x = np.asarray(features, dtype="float64")
        if x.shape[0] == 0:
            return None
        probs = []
        for model in bucket.models:
            if model.coef.size != x.shape[0]:
                continue
            score = float(np.dot(model.coef, x) + model.intercept)
            probs.append(self._apply_calibration(score, model.calibration))
        if not probs:
            return None
        arr = np.asarray(probs, dtype="float64")
        p_loss = float(np.mean(arr))
        p_std = float(np.std(arr)) if arr.size > 1 else 0.0
        return {
            "bucket_key": selected_key,
            "p_loss": p_loss,
            "p_loss_std": p_std,
            "model_count": int(arr.size),
            "n_samples": int(bucket.n_samples),
            "n_loss": int(bucket.n_loss),
            "n_win": int(bucket.n_win),
            "level": str(bucket.level or ""),
        }

    def should_veto(
        self,
        bucket_key: str | Sequence[str],
        features: Sequence[float],
        threshold: Optional[float] = None,
        uncertainty_z: float = 1.0,
        max_std: Optional[float] = None,
        min_samples: int = 0,
    ) -> tuple[bool, Optional[dict]]:
        pred = self.predict_loss(bucket_key, features)
        if pred is None:
            return False, None
        limit = self.threshold if threshold is None else float(threshold)
        p_loss = float(pred.get("p_loss", 0.5))
        p_std = float(pred.get("p_loss_std", 0.0))
        z = float(uncertainty_z) if np.isfinite(uncertainty_z) else 1.0
        p_lcb = float(p_loss - max(0.0, z) * max(0.0, p_std))
        p_ucb = float(p_loss + max(0.0, z) * max(0.0, p_std))

        veto_hit = p_lcb > float(limit)
        if max_std is not None and np.isfinite(max_std) and p_std > float(max_std):
            veto_hit = False
        if int(min_samples or 0) > 0 and int(pred.get("n_samples", 0) or 0) < int(min_samples):
            veto_hit = False

        pred["threshold"] = float(limit)
        pred["p_loss_lcb"] = p_lcb
        pred["p_loss_ucb"] = p_ucb
        pred["veto_margin"] = float(p_lcb - float(limit))
        pred["veto_hit"] = bool(veto_hit)
        return bool(veto_hit), pred

    def evaluate_candidate_ev(
        self,
        bucket_key: str | Sequence[str],
        features: Sequence[float],
        *,
        tp_points: float,
        sl_points: float,
        uncertainty_z: float = 1.0,
        min_samples: int = 0,
        max_p_loss_std: Optional[float] = None,
        blend_empirical: bool = True,
        prior_strength: float = 300.0,
        min_ev_lcb_points: float = 0.0,
        min_ev_mean_points: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Evaluate a DE3 candidate using model-implied loss probability + uncertainty.

        EV is computed in points from candidate bracket geometry:
            EV = P(win)*TP - P(loss)*SL, where P(win)=1-P(loss)

        Uncertainty is propagated from p_loss_std:
            ev_std ~= (TP + SL) * p_loss_std
        """
        pred = self.predict_loss(bucket_key, features)
        if pred is None:
            return None

        try:
            tp = float(tp_points)
            sl = float(sl_points)
        except Exception:
            return None
        if not np.isfinite(tp) or not np.isfinite(sl) or tp <= 0.0 or sl <= 0.0:
            return None

        p_model = self._clip01(float(pred.get("p_loss", 0.5)))
        p_model_std = max(0.0, float(pred.get("p_loss_std", 0.0) or 0.0))
        n_samples = int(pred.get("n_samples", 0) or 0)
        n_loss = int(pred.get("n_loss", 0) or 0)
        n_win = int(pred.get("n_win", 0) or 0)

        # Empirical bucket prior stabilizes probabilities where calibration drifts.
        p_emp = p_model
        if n_samples > 0:
            if (n_loss + n_win) > 0:
                p_emp = self._clip01(float(n_loss) / float(n_loss + n_win))
            else:
                p_emp = self._clip01(float(n_loss) / float(max(1, n_samples)))

        blend_weight = 0.0
        if blend_empirical:
            try:
                strength = max(1e-6, float(prior_strength))
            except Exception:
                strength = 300.0
            blend_weight = float(n_samples) / float(n_samples + strength)

        p_loss = self._clip01((1.0 - blend_weight) * p_model + (blend_weight * p_emp))
        p_emp_std = float(np.sqrt(max(p_emp * (1.0 - p_emp), 0.0) / float(max(1, n_samples))))
        p_std = float(np.sqrt(((1.0 - blend_weight) ** 2) * (p_model_std ** 2) + (blend_weight ** 2) * (p_emp_std ** 2)))

        try:
            z = float(uncertainty_z)
        except Exception:
            z = 1.0
        if not np.isfinite(z) or z < 0.0:
            z = 1.0

        p_loss_lcb = self._clip01(p_loss - (z * p_std))
        p_loss_ucb = self._clip01(p_loss + (z * p_std))
        p_win = self._clip01(1.0 - p_loss)

        ev_mean = float((p_win * tp) - (p_loss * sl))
        ev_std = float((tp + sl) * max(0.0, p_std))
        ev_lcb = float(ev_mean - (z * ev_std))
        ev_ucb = float(ev_mean + (z * ev_std))

        allow = True
        reason = "ok"
        if int(min_samples or 0) > 0 and n_samples < int(min_samples):
            allow = False
            reason = f"samples {n_samples}<{int(min_samples)}"
        elif max_p_loss_std is not None and np.isfinite(max_p_loss_std) and p_std > float(max_p_loss_std):
            allow = False
            reason = f"p_loss_std {p_std:.3f}>{float(max_p_loss_std):.3f}"
        elif ev_lcb < float(min_ev_lcb_points):
            allow = False
            reason = f"ev_lcb {ev_lcb:.2f}<{float(min_ev_lcb_points):.2f}"
        elif min_ev_mean_points is not None and np.isfinite(min_ev_mean_points) and ev_mean < float(min_ev_mean_points):
            allow = False
            reason = f"ev_mean {ev_mean:.2f}<{float(min_ev_mean_points):.2f}"

        return {
            "bucket_key": str(pred.get("bucket_key") or ""),
            "level": str(pred.get("level") or ""),
            "n_samples": int(n_samples),
            "n_loss": int(n_loss),
            "n_win": int(n_win),
            "p_loss_model": float(p_model),
            "p_loss_empirical": float(p_emp),
            "p_loss_blend_weight": float(blend_weight),
            "p_loss": float(p_loss),
            "p_loss_std": float(p_std),
            "p_loss_lcb": float(p_loss_lcb),
            "p_loss_ucb": float(p_loss_ucb),
            "p_win": float(p_win),
            "ev_points": float(ev_mean),
            "ev_std_points": float(ev_std),
            "ev_lcb_points": float(ev_lcb),
            "ev_ucb_points": float(ev_ucb),
            "allow": bool(allow),
            "reason": str(reason),
        }

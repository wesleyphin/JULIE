"""Inference helper for Kalshi ML v7 model.

Used both by:
  - v7 self-eval scripts (validate the trained model)
  - live bot (`julie001._apply_kalshi_trade_overlay_to_signal` if/when wired)

The model takes a flat feature dict and returns predicted forward $PnL.
A positive prediction above pass_thr means: ML wants to OVERRIDE rule=BLOCK
to PASS. Negative below -block_thr means override rule=PASS to BLOCK.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path("/Users/wes/Downloads/JULIE001")
MODEL_PATH = ROOT / "artifacts" / "regime_ml_kalshi_v7" / "model.pkl"

log = logging.getLogger("kalshi_ml_inference")


class KalshiMLInference:
    """Lazy-loaded singleton wrapper around the v7 model pickle."""
    _instance = None

    @classmethod
    def get(cls, model_path: Optional[Path] = None) -> "KalshiMLInference":
        if cls._instance is None:
            cls._instance = cls(model_path or MODEL_PATH)
        return cls._instance

    def __init__(self, model_path: Path):
        self.path = model_path
        self.payload = None
        self.reg = None
        self.feature_cols: list[str] = []
        self.median_imputes: dict[str, float] = {}
        self.pass_thr: float = 0.0
        self.block_thr: float = 0.0
        self.horizon_min: int = 60
        self.enabled = False
        self._load()

    def _load(self):
        if not self.path.exists():
            log.warning("[kalshi-ml] model not found at %s", self.path)
            return
        try:
            with self.path.open("rb") as f:
                self.payload = pickle.load(f)
            self.reg = self.payload["reg"]
            self.feature_cols = list(self.payload["feature_cols"])
            self.median_imputes = dict(self.payload.get("median_imputes", {}))
            decision = self.payload.get("decision", {})
            self.pass_thr = float(decision.get("pass_thr", 0.0))
            self.block_thr = float(decision.get("block_thr", 0.0))
            self.horizon_min = int(self.payload.get("horizon_min", 60))
            self.enabled = True
            log.info("[kalshi-ml] loaded v7 model: %d features, "
                     "pass_thr=$%.1f block_thr=$%.1f horizon=%dmin",
                     len(self.feature_cols), self.pass_thr, self.block_thr,
                     self.horizon_min)
        except Exception as exc:
            log.exception("[kalshi-ml] failed to load model: %s", exc)
            self.enabled = False

    def predict_pnl(self, features: dict) -> Optional[float]:
        """Return predicted forward PnL ($) for a single Kalshi event.
        features must contain keys matching `self.feature_cols`. Missing
        features are imputed with the trained-time median."""
        if not self.enabled or self.reg is None: return None
        x = np.array([[
            features.get(c, self.median_imputes.get(c, 0.0))
            for c in self.feature_cols
        ]], dtype=float)
        # NaN cleanup: any NaN/inf becomes the median
        for j, c in enumerate(self.feature_cols):
            v = x[0, j]
            if not np.isfinite(v):
                x[0, j] = self.median_imputes.get(c, 0.0)
        try:
            return float(self.reg.predict(x)[0])
        except Exception as exc:
            log.warning("[kalshi-ml] predict failed: %s", exc)
            return None

    def decide_override(self, features: dict, rule_decision: str) -> tuple[str, dict]:
        """Apply binary-override decision. Returns (final_decision, metadata).

        rule_decision: 'PASS' or 'BLOCK'
        Returns:
          ('PASS', {'override': True, 'reason': '...'})  if ML overrides BLOCK→PASS
          ('BLOCK', {'override': True, 'reason': '...'}) if ML overrides PASS→BLOCK
          ('PASS'|'BLOCK', {'override': False, ...})     if ML defers to rule
        """
        if not self.enabled:
            return rule_decision, {"override": False, "reason": "ml_not_loaded"}
        pred = self.predict_pnl(features)
        if pred is None:
            return rule_decision, {"override": False, "reason": "predict_failed"}
        meta = {"pred_pnl": pred, "pass_thr": self.pass_thr, "block_thr": self.block_thr}
        if rule_decision == "BLOCK" and pred >= self.pass_thr:
            return "PASS", {**meta, "override": True,
                            "reason": f"ml_pred_pnl=${pred:+.2f}>=${self.pass_thr:.1f}_pass"}
        if rule_decision == "PASS" and pred <= -self.block_thr:
            return "BLOCK", {**meta, "override": True,
                             "reason": f"ml_pred_pnl=${pred:+.2f}<=-${self.block_thr:.1f}_block"}
        return rule_decision, {**meta, "override": False, "reason": "ml_defers_to_rule"}


__all__ = ["KalshiMLInference"]

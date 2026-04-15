from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from config import CONFIG
from strategy_base import Strategy
from services.sentiment_service import get_sentiment_state


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class TruthSocialEngine(Strategy):
    def __init__(self):
        cfg = (CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) if isinstance(CONFIG, dict) else {}) or {}
        self.strategy_name = "TruthSocialEngine"
        self.enabled = bool(cfg.get("enabled", True))
        self.pump_threshold = float(cfg.get("pump_threshold", 0.85) or 0.85)
        self.signal_max_age_seconds = max(60, int(cfg.get("signal_max_age_seconds", 1800) or 1800))
        self.quick_pump_tp_points = max(0.25, float(cfg.get("quick_pump_tp_points", 4.0) or 4.0))
        self.quick_pump_sl_points = max(0.25, float(cfg.get("quick_pump_sl_points", 2.0) or 2.0))
        self._last_emitted_post_id: Optional[str] = None

    def on_bar(self, df) -> Optional[Dict]:
        if not self.enabled:
            return None

        snapshot = get_sentiment_state()
        if not bool(snapshot.get("enabled")) or not bool(snapshot.get("healthy")):
            return None

        post_id = str(snapshot.get("latest_post_id") or "").strip()
        if not post_id or post_id == self._last_emitted_post_id:
            return None

        score = snapshot.get("sentiment_score")
        confidence = snapshot.get("finbert_confidence")
        if score is None or confidence is None:
            return None

        created_at = _parse_iso(snapshot.get("latest_post_created_at"))
        if created_at is None:
            return None
        age_seconds = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds())
        if age_seconds > float(self.signal_max_age_seconds):
            return None

        if float(score) >= self.pump_threshold:
            side = "LONG"
        elif float(score) <= -self.pump_threshold:
            side = "SHORT"
        else:
            return None

        self._last_emitted_post_id = post_id
        return {
            "strategy": self.strategy_name,
            "side": side,
            "tp_dist": float(self.quick_pump_tp_points),
            "sl_dist": float(self.quick_pump_sl_points),
            "entry_mode": "sentiment_impulse",
            "sub_strategy": "Truth Social Quick Pump",
            "combo_key": "truth_social_impulse",
            "rule_id": "truth_social_finbert",
            "priority": "SENTIMENT",
            "truth_social_post_id": post_id,
            "truth_social_post_created_at": snapshot.get("latest_post_created_at"),
            "truth_social_post_url": snapshot.get("latest_post_url"),
            "truth_social_post_text": snapshot.get("latest_post_text"),
            "sentiment_score": float(score),
            "finbert_confidence": float(confidence),
            "truth_social_trigger_reason": snapshot.get("trigger_reason"),
        }

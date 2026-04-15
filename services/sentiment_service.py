from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config import CONFIG
from event_logger import event_logger


logger = logging.getLogger("truth_social_sentiment")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


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


def _clean_post_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


@dataclass
class SentimentState:
    enabled: bool = False
    active: bool = False
    healthy: bool = False
    model_loaded: bool = False
    quantized_8bit: bool = False
    target_handle: Optional[str] = None
    source: str = "truthbrush_finbert"
    last_poll_at: Optional[str] = None
    last_analysis_at: Optional[str] = None
    latest_post_id: Optional[str] = None
    latest_post_created_at: Optional[str] = None
    latest_post_url: Optional[str] = None
    latest_post_text: Optional[str] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    finbert_confidence: Optional[float] = None
    trigger_side: Optional[str] = None
    trigger_reason: Optional[str] = None
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "enabled": bool(self.enabled),
            "active": bool(self.active),
            "healthy": bool(self.healthy),
            "model_loaded": bool(self.model_loaded),
            "quantized_8bit": bool(self.quantized_8bit),
            "target_handle": self.target_handle,
            "source": self.source,
            "last_poll_at": self.last_poll_at,
            "last_analysis_at": self.last_analysis_at,
            "latest_post_id": self.latest_post_id,
            "latest_post_created_at": self.latest_post_created_at,
            "latest_post_url": self.latest_post_url,
            "latest_post_text": self.latest_post_text,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "finbert_confidence": self.finbert_confidence,
            "trigger_side": self.trigger_side,
            "trigger_reason": self.trigger_reason,
            "last_error": self.last_error,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


_STATE_LOCK = threading.Lock()
_SENTIMENT_STATE = SentimentState()


def update_sentiment_state(**updates: Any) -> Dict[str, Any]:
    with _STATE_LOCK:
        for key, value in updates.items():
            if hasattr(_SENTIMENT_STATE, key):
                setattr(_SENTIMENT_STATE, key, value)
        return _SENTIMENT_STATE.to_dict()


def set_sentiment_state(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    with _STATE_LOCK:
        state = SentimentState()
        for key in state.to_dict().keys():
            if key in snapshot:
                setattr(state, key, snapshot.get(key))
        state.metadata = dict(snapshot.get("metadata") or {})
        global _SENTIMENT_STATE
        _SENTIMENT_STATE = state
        return _SENTIMENT_STATE.to_dict()


def get_sentiment_state() -> Dict[str, Any]:
    with _STATE_LOCK:
        return _SENTIMENT_STATE.to_dict()


class TruthSocialSentimentService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.poll_interval = max(5, int(cfg.get("poll_interval", 30) or 30))
        self.pump_threshold = max(0.0, float(cfg.get("pump_threshold", 0.85) or 0.85))
        self.emergency_exit_threshold = float(cfg.get("emergency_exit_threshold", -0.75) or -0.75)
        self.finbert_local_path = str(cfg.get("finbert_local_path", "./models/finbert") or "./models/finbert")
        self.target_handle = str(cfg.get("target_handle", "realDonaldTrump") or "realDonaldTrump").lstrip("@")
        self.signal_max_age_seconds = max(60, int(cfg.get("signal_max_age_seconds", 1800) or 1800))
        self.emergency_exit_max_age_seconds = max(
            self.signal_max_age_seconds,
            int(cfg.get("emergency_exit_max_age_seconds", 3600) or 3600),
        )
        self._api = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._quantized_8bit = False
        self._stop_event = asyncio.Event()
        self._seen_post_ids: set[str] = set()
        self._seen_post_order: list[str] = []
        self._seen_limit = 128
        self._last_seen_created_at: Optional[datetime] = None

        update_sentiment_state(
            enabled=self.enabled,
            active=False,
            healthy=False,
            model_loaded=False,
            quantized_8bit=False,
            target_handle=self.target_handle,
            source="truthbrush_finbert",
            last_error=None,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> Dict[str, Any]:
        return get_sentiment_state()

    def _mark_error(self, message: str, *, active: bool = False) -> None:
        logger.warning("Truth Social sentiment service: %s", message)
        update_sentiment_state(
            enabled=self.enabled,
            active=active,
            healthy=False,
            model_loaded=self._model is not None,
            quantized_8bit=self._quantized_8bit,
            target_handle=self.target_handle,
            last_poll_at=_iso_or_none(_utc_now()),
            last_error=str(message),
        )

    def _ensure_truthbrush_api(self):
        if self._api is not None:
            return self._api
        try:
            from truthbrush import Api
        except Exception as exc:
            raise RuntimeError(f"truthbrush import failed: {exc}") from exc

        username = os.environ.get("TRUTHSOCIAL_USERNAME")
        password = os.environ.get("TRUTHSOCIAL_PASSWORD")
        token = os.environ.get("TRUTHSOCIAL_TOKEN")
        if not token and (not username or not password):
            raise RuntimeError(
                "Truth Social credentials missing. Set TRUTHSOCIAL_TOKEN or TRUTHSOCIAL_USERNAME/TRUTHSOCIAL_PASSWORD."
            )
        self._api = Api(username=username, password=password, token=token)
        return self._api

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"FinBERT dependencies unavailable: {exc}") from exc

        local_model_path = Path(self.finbert_local_path)
        local_files_only = local_model_path.exists()
        model_source = str(local_model_path) if local_files_only else "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)

        quantized_8bit = False
        model = None
        try:
            from transformers import BitsAndBytesConfig

            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                local_files_only=local_files_only,
                quantization_config=quant_cfg,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            quantized_8bit = True
        except Exception as quant_exc:
            logger.warning("FinBERT 8-bit load failed, using standard precision fallback: %s", quant_exc)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                local_files_only=local_files_only,
            )

        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._quantized_8bit = bool(quantized_8bit)
        update_sentiment_state(
            model_loaded=True,
            quantized_8bit=self._quantized_8bit,
            last_error=None,
            metadata={"model_source": model_source},
        )

    def _post_url(self, post: Dict[str, Any]) -> Optional[str]:
        handle = str(self.target_handle or "").strip()
        post_id = str(post.get("id") or "").strip()
        if not handle or not post_id:
            return None
        return f"https://truthsocial.com/@{handle}/{post_id}"

    def _iter_new_posts(self) -> Iterable[Dict[str, Any]]:
        api = self._ensure_truthbrush_api()
        created_after = self._last_seen_created_at
        posts = list(
            api.pull_statuses(
                self.target_handle,
                replies=False,
                created_after=created_after,
                pinned=False,
            )
        )
        if not posts:
            return []

        items: list[Dict[str, Any]] = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            post_id = str(post.get("id") or "").strip()
            if not post_id or post_id in self._seen_post_ids:
                continue
            items.append(post)
        items.sort(
            key=lambda item: _parse_iso(item.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)
        )
        return items

    def _classify_text(self, text: str) -> Dict[str, Any]:
        self._ensure_model()
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("FinBERT model not loaded")

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with self._torch.no_grad():
            outputs = self._model(**encoded)
            probabilities = self._torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

        label_map = getattr(self._model.config, "id2label", {}) or {}
        labels = [str(label_map.get(index, str(index))).lower() for index in range(len(probabilities))]
        score_by_label = {label: float(prob) for label, prob in zip(labels, probabilities)}
        positive = float(score_by_label.get("positive", 0.0))
        negative = float(score_by_label.get("negative", 0.0))
        neutral = float(score_by_label.get("neutral", 0.0))
        sentiment_score = max(-1.0, min(1.0, positive - negative))

        top_label = "neutral"
        top_probability = neutral
        for label, prob in score_by_label.items():
            if prob > top_probability:
                top_label = label
                top_probability = prob

        return {
            "sentiment_score": round(float(sentiment_score), 4),
            "finbert_confidence": round(float(top_probability), 4),
            "sentiment_label": top_label,
            "probabilities": {
                "positive": round(positive, 4),
                "negative": round(negative, 4),
                "neutral": round(neutral, 4),
            },
        }

    def _trigger_payload(self, sentiment_score: float) -> tuple[Optional[str], Optional[str]]:
        if sentiment_score >= self.pump_threshold:
            return "LONG", f"Truth Social quick-pump threshold {sentiment_score:.2f} >= {self.pump_threshold:.2f}"
        if sentiment_score <= -self.pump_threshold:
            return "SHORT", f"Truth Social dump threshold {sentiment_score:.2f} <= {-self.pump_threshold:.2f}"
        return None, None

    def _remember_post(self, post_id: str, created_at: Optional[datetime]) -> None:
        if post_id not in self._seen_post_ids:
            self._seen_post_ids.add(post_id)
            self._seen_post_order.append(post_id)
        if len(self._seen_post_ids) > self._seen_limit:
            while len(self._seen_post_ids) > self._seen_limit:
                oldest = self._seen_post_order.pop(0)
                self._seen_post_ids.discard(oldest)
        if created_at is not None and (
            self._last_seen_created_at is None or created_at > self._last_seen_created_at
        ):
            self._last_seen_created_at = created_at

    async def poll_once(self) -> Dict[str, Any]:
        now = _utc_now()
        if not self.enabled:
            update_sentiment_state(
                enabled=False,
                active=False,
                healthy=False,
                last_poll_at=_iso_or_none(now),
                target_handle=self.target_handle,
            )
            return get_sentiment_state()

        try:
            posts = await asyncio.to_thread(lambda: list(self._iter_new_posts()))
        except Exception as exc:
            self._mark_error(str(exc), active=True)
            return get_sentiment_state()

        update_sentiment_state(
            enabled=True,
            active=True,
            healthy=True,
            target_handle=self.target_handle,
            last_poll_at=_iso_or_none(now),
            last_error=None,
        )

        if not posts:
            return get_sentiment_state()

        latest_snapshot = get_sentiment_state()
        for post in posts:
            post_id = str(post.get("id") or "").strip()
            created_at = _parse_iso(post.get("created_at"))
            post_text = _clean_post_text(post.get("content") or post.get("text"))
            if not post_id or not post_text:
                continue

            try:
                analysis = await asyncio.to_thread(self._classify_text, post_text)
            except Exception as exc:
                self._mark_error(str(exc), active=True)
                continue

            trigger_side, trigger_reason = self._trigger_payload(float(analysis["sentiment_score"]))
            latest_snapshot = update_sentiment_state(
                enabled=True,
                active=True,
                healthy=True,
                model_loaded=self._model is not None,
                quantized_8bit=self._quantized_8bit,
                target_handle=self.target_handle,
                last_poll_at=_iso_or_none(now),
                last_analysis_at=_iso_or_none(now),
                latest_post_id=post_id,
                latest_post_created_at=_iso_or_none(created_at),
                latest_post_url=self._post_url(post),
                latest_post_text=post_text,
                sentiment_label=str(analysis["sentiment_label"]),
                sentiment_score=float(analysis["sentiment_score"]),
                finbert_confidence=float(analysis["finbert_confidence"]),
                trigger_side=trigger_side,
                trigger_reason=trigger_reason,
                last_error=None,
                metadata={
                    "model_source": (
                        str(Path(self.finbert_local_path))
                        if Path(self.finbert_local_path).exists()
                        else "ProsusAI/finbert"
                    ),
                    **dict(analysis.get("probabilities") or {}),
                },
            )
            self._remember_post(post_id, created_at)
            event_logger.log_sentiment_event(
                "Truth Social post analyzed",
                {
                    "strategy": "TruthSocialEngine",
                    "target_handle": self.target_handle,
                    "post_id": post_id,
                    "sentiment_label": latest_snapshot.get("sentiment_label"),
                    "sentiment_score": latest_snapshot.get("sentiment_score"),
                    "finbert_confidence": latest_snapshot.get("finbert_confidence"),
                    "trigger_side": latest_snapshot.get("trigger_side"),
                },
            )
            if trigger_side:
                event_logger.log_sentiment_event(
                    "Truth Social trade trigger armed",
                    {
                        "strategy": "TruthSocialEngine",
                        "target_handle": self.target_handle,
                        "post_id": post_id,
                        "sentiment_score": latest_snapshot.get("sentiment_score"),
                        "finbert_confidence": latest_snapshot.get("finbert_confidence"),
                        "trigger_side": trigger_side,
                        "reason": trigger_reason,
                    },
                )

        return latest_snapshot

    async def run_forever(self) -> None:
        update_sentiment_state(
            enabled=self.enabled,
            active=bool(self.enabled),
            healthy=False,
            target_handle=self.target_handle,
        )
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._mark_error(str(exc), active=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                continue


def build_truth_social_sentiment_service(config: Optional[Dict[str, Any]] = None) -> TruthSocialSentimentService:
    if config is None:
        config = (CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) if isinstance(CONFIG, dict) else {}) or {}
    return TruthSocialSentimentService(config)

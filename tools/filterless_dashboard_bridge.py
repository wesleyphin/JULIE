from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_state import load_bot_state, trading_day_start as bot_trading_day_start
from config import CONFIG
from process_singleton import acquire_singleton_lock

NY_TZ = ZoneInfo("America/New_York")
DEFAULT_LOG_PATH = ROOT / "topstep_live_bot.log"
DEFAULT_STATE_PATH = ROOT / "bot_state.json"
DEFAULT_TRADE_FACTORS_PATH = ROOT / "live_trade_factors.csv"
DEFAULT_OUTPUT_PATH = ROOT / "montecarlo" / "Backtest-Simulator-main" / "public" / "filterless_live_state.json"
DEFAULT_KALSHI_SNAPSHOT_PATH = ROOT / "kalshi_live_snapshot.json"
BRIDGE_LOCK_PATH = ROOT / "logs" / "filterless_dashboard_bridge.lock"


def _config_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed)


_RISK_CONFIG = CONFIG.get("RISK") or {}
TOPSTEP_ROUND_TURN_COMMISSION = round(
    max(0.0, _config_float(_RISK_CONFIG.get("TOPSTEP_COMMISSION_ROUND_TURN_PER_CONTRACT"), 0.50)),
    2,
)
MES_ROUND_TURN_FEE = round(
    max(0.0, _config_float(_RISK_CONFIG.get("FEES_PER_SIDE"), 0.37)) * 2.0
    + TOPSTEP_ROUND_TURN_COMMISSION,
    2,
)

STRATEGY_ORDER = ["dynamic_engine3", "regime_adaptive", "ml_physics",
                  "aetherflow", "fib_h1214", "h9_gapfade"]
STRATEGY_LABELS = {
    "dynamic_engine3": "Dynamic Engine 3",
    "regime_adaptive": "RegimeAdaptive",
    "ml_physics": "ML Physics",
    "aetherflow": "AetherFlow",
    "fib_h1214": "Fibonacci",
    "h9_gapfade": "H9 GapFade",
}
FILTERLESS_LIVE_DISABLED_STRATEGIES = {
    str(value).strip().lower()
    for value in (CONFIG.get("FILTERLESS_LIVE_DISABLED_STRATEGIES", []) or [])
    if str(value).strip()
}
TRUTH_SOCIAL_CONFIG = dict(CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {})
GEMINI_CONFIG = dict(CONFIG.get("GEMINI", {}) or {})

LOG_PREFIX_RE = re.compile(
    r"^(?P<logged_at>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(?P<level>[^\]]+)\] (?P<message>.*)$"
)
STRUCTURED_EVENT_RE = re.compile(r"^\[(?P<event_ts>[^\]]+)\] \[(?P<event_type>[A-Z_]+)\] (?P<body>.*)$")
HEARTBEAT_RE = re.compile(
    r"^💓 Heartbeat #(?P<count>\d+): (?P<clock>\d{2}:\d{2}:\d{2}) \| Session: (?P<session_status>[^|]+) \| Price: (?P<price>-?\d+(?:\.\d+)?)$"
)
BAR_RE = re.compile(
    r"^Bar: (?P<bar_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>-?\d+(?:\.\d+)?)$"
)
POSITION_SYNC_RE = re.compile(
    r"^🔄 Position Sync #(?P<count>\d+): (?P<clock>\d{2}:\d{2}:\d{2}) \| "
    r"(?:(?:Status: (?P<status>FLAT))|"
    r"(?P<side>LONG|SHORT) (?P<size>\d+) @ (?P<avg_price>-?\d+(?:\.\d+)?)(?: \| OpenPnL: \$(?P<open_pnl>-?\d+(?:\.\d+)?))?)$"
)
TRADE_CLOSED_RE = re.compile(
    r"^(?:📊 )?Trade closed(?: \(reverse\))?: (?P<label>.+?) (?P<side>LONG|SHORT) \| Entry: (?P<entry>-?\d+(?:\.\d+)?) \| Exit: (?P<exit>-?\d+(?:\.\d+)?) \| PnL: (?P<pnl_points>-?\d+(?:\.\d+)?) pts \(\$(?P<pnl_dollars>-?\d+(?:\.\d+)?)\)(?: \| (?P<tail>.*))?$"
)
MANIFOLD_ABSTAIN_RE = re.compile(r"^🧭 Manifold abstain: (?P<reason>.+)$")
DE3_BREAKEVEN_ARMED_RE = re.compile(
    r"^DE3 v4 break-even armed: (?P<label>.+?) (?P<side>LONG|SHORT) -> (?P<new_sl>-?\d+(?:\.\d+)?)$"
)
READY_RE = re.compile(r"State restored\. Bot is ready", re.IGNORECASE)


def iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, "", "None"):
        return {}
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def parse_logged_at(value: str) -> Optional[datetime]:
    # Bot log timestamps are written in the host's local wall-clock time (PDT
    # on Mac, etc.), NOT in NY_TZ. Interpret them as local and convert to ET
    # so freshness checks against datetime.now(NY_TZ) don't create a phantom
    # multi-hour age gap on non-NY hosts.
    try:
        naive = datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")
        return naive.astimezone(NY_TZ)
    except ValueError:
        return None


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NY_TZ)
    return parsed.astimezone(NY_TZ)


def session_day_start(value: Optional[datetime]) -> datetime:
    ts = value or datetime.now(NY_TZ)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=NY_TZ)
    return bot_trading_day_start(ts, NY_TZ)


def canonical_strategy_id(raw_strategy: Optional[str]) -> Optional[str]:
    raw = str(raw_strategy or "").strip()
    if not raw:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", raw.lower())
    if normalized.startswith("dynamicengine"):
        return "dynamic_engine3"
    if normalized.startswith("regimeadaptive"):
        return "regime_adaptive"
    if normalized.startswith("mlphysics"):
        return "ml_physics"
    if normalized.startswith("manifold"):
        return None
    if normalized.startswith("aetherflow"):
        return "aetherflow"
    if normalized.startswith("fibh1214"):
        return "fib_h1214"
    if normalized.startswith("h9gapfade") or normalized.startswith("h9_gapfade"):
        return "h9_gapfade"
    return None


def resolve_trade_strategy_identity(
    raw_strategy: Optional[str],
    raw_label: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    canonical = canonical_strategy_id(raw_strategy) or canonical_strategy_id(raw_label)
    if canonical is not None:
        return canonical, STRATEGY_LABELS[canonical]

    label = str(raw_label or "").strip() or str(raw_strategy or "").strip()
    if not label:
        return None, None

    normalized = re.sub(r"[^a-z0-9]+", "", label.lower())
    friendly_labels = {
        "restoredliveposition": "Recovered Live Position",
        "projectxhistorybackfill": "ProjectX Backfill",
    }
    fallback_label = friendly_labels.get(normalized, label)
    fallback_key = re.sub(r"[^a-z0-9]+", "_", fallback_label.lower()).strip("_") or "recovered_trade"
    return f"recovered_{fallback_key}", fallback_label


def strategy_state_template(strategy_id: str) -> Dict[str, Any]:
    return {
        "id": strategy_id,
        "label": STRATEGY_LABELS[strategy_id],
        "status": "idle",
        "updated_at": None,
        "last_signal_time": None,
        "last_signal_side": None,
        "last_signal_price": None,
        "tp_dist": None,
        "sl_dist": None,
        "priority": None,
        "last_reason": None,
        "last_block_reason": None,
        "latest_activity": None,
        "latest_activity_time": None,
        "latest_activity_type": None,
        "latest_activity_severity": None,
        "last_trade_pnl": None,
        "last_trade_points": None,
        "last_trade_time": None,
        "last_trade_side": None,
        "last_trade_entry": None,
        "last_trade_exit": None,
        "sub_strategy": None,
        "combo_key": None,
        "rule_id": None,
        "early_exit_enabled": None,
        "gate_prob": None,
        "gate_threshold": None,
        "entry_mode": None,
        "base_session": None,
        "current_session": None,
        "vol_regime": None,
    }


def default_sentiment_metrics() -> Optional[Dict[str, Any]]:
    sentiment_enabled = bool(TRUTH_SOCIAL_CONFIG.get("enabled", False))
    if not sentiment_enabled:
        return None
    target_handle = str(
        TRUTH_SOCIAL_CONFIG.get("target_handle", "realDonaldTrump") or "realDonaldTrump"
    ).lstrip("@")
    finbert_path = str(
        TRUTH_SOCIAL_CONFIG.get("finbert_local_path", "./models/finbert") or "./models/finbert"
    )
    gemini_model = str(GEMINI_CONFIG.get("model", "gemini-3-pro-preview") or "gemini-3-pro-preview")
    gemini_api_key = str(GEMINI_CONFIG.get("api_key", "") or "").strip()
    return {
        "enabled": True,
        "active": False,
        "healthy": False,
        "model_loaded": False,
        "quantized_8bit": False,
        "target_handle": target_handle,
        "source": "rss_finbert",
        "last_poll_at": None,
        "last_analysis_at": None,
        "latest_post_id": None,
        "latest_post_created_at": None,
        "latest_post_url": None,
        "latest_post_text": None,
        "sentiment_label": None,
        "sentiment_score": None,
        "finbert_confidence": None,
        "trigger_side": None,
        "trigger_reason": None,
        "last_error": None,
        "metadata": {
            "finbert_local_path": finbert_path,
            "gemini_enabled": bool(GEMINI_CONFIG.get("enabled", False)) and bool(gemini_api_key),
            "gemini_configured": bool(gemini_api_key),
            "gemini_model": gemini_model,
            "gemini_used": False,
        },
    }


def normalize_sentiment_error_message(value: Any) -> Optional[str]:
    message = str(value or "").strip()
    if not message:
        return None
    return message


def disabled_strategy_message(strategy_id: str) -> str:
    return f"{STRATEGY_LABELS.get(strategy_id, strategy_id)} disabled in filterless live config"


def parse_pipe_details(body: str) -> tuple[str, Dict[str, str]]:
    parts = [part.strip() for part in body.split("|")]
    message = parts[0] if parts else body.strip()
    details: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        details[key.strip()] = value.strip()
    return message, details


def parse_tail_details(body: Optional[str]) -> Dict[str, str]:
    details: Dict[str, str] = {}
    text = str(body or "").strip()
    if not text:
        return details
    for part in text.split("|"):
        item = part.strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        details[key.strip()] = value.strip()
    return details


def append_limited(bucket: list[Dict[str, Any]], item: Dict[str, Any], limit: int) -> None:
    bucket.append(item)
    if len(bucket) > limit:
        del bucket[0 : len(bucket) - limit]


def set_strategy_activity(
    strategy: Dict[str, Any],
    when: Optional[datetime],
    activity_type: str,
    message: str,
    *,
    severity: str = "info",
    blocked: bool = False,
) -> None:
    timestamp = iso_or_none(when)
    strategy["latest_activity"] = message
    strategy["latest_activity_time"] = timestamp
    strategy["latest_activity_type"] = activity_type
    strategy["latest_activity_severity"] = str(severity or "info").lower()
    strategy["last_reason"] = message
    if blocked:
        strategy["last_block_reason"] = message
    else:
        strategy["last_block_reason"] = None


def _event_signature(item: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("event_type") or ""),
        str(item.get("strategy_id") or ""),
        str(item.get("message") or ""),
    )


def _parse_event_time(value: Optional[str]) -> Optional[datetime]:
    return parse_iso(value)


def should_skip_duplicate_event(bucket: list[Dict[str, Any]], item: Dict[str, Any]) -> bool:
    if not bucket:
        return False
    item_time = _parse_event_time(item.get("time"))
    for existing in reversed(bucket[-8:]):
        if _event_signature(existing) != _event_signature(item):
            continue
        existing_time = _parse_event_time(existing.get("time"))
        if existing_time is None or item_time is None:
            return True
        if abs((item_time - existing_time).total_seconds()) <= 5.0:
            return True
    return False


def format_strategy_signal_activity(event_message: str, details: Dict[str, str]) -> tuple[str, bool]:
    side = str(details.get("side") or "").strip().upper()
    if side in {"NONE", "ALL"}:
        side = ""
    status = str(details.get("status") or "").strip().upper()
    decision = str(details.get("decision") or "").strip().lower()
    reason = str(details.get("reason") or details.get("block_reason") or "").strip()
    price = safe_float(details.get("price"))
    tp_dist = safe_float(details.get("tp_dist"))
    sl_dist = safe_float(details.get("sl_dist"))
    side_text = f" {side}" if side else ""
    price_text = f" @ {price:.2f}" if price is not None else ""
    bracket_bits = []
    if tp_dist is not None:
        bracket_bits.append(f"TP {tp_dist:.2f}")
    if sl_dist is not None:
        bracket_bits.append(f"SL {sl_dist:.2f}")
    bracket_text = f" | {' | '.join(bracket_bits)}" if bracket_bits else ""
    blocked = status == "BLOCKED" or decision == "blocked" or bool(reason)
    if blocked:
        base = f"Decision blocked{side_text}{price_text}"
        if reason:
            base = f"{base} | {reason}"
        return base, True
    if status == "EXECUTED":
        return f"Executed{side_text}{price_text}{bracket_text}", False
    if status == "QUEUED":
        return f"Queued{side_text}{price_text}{bracket_text}", False
    if status == "CANDIDATE" or side_text or price_text:
        return f"Candidate{side_text}{price_text}{bracket_text}", False
    return event_message, False


def format_trade_placed_activity(details: Dict[str, str]) -> str:
    side = str(details.get("side") or "").strip().upper()
    entry = safe_float(details.get("entry")) or safe_float(details.get("price"))
    tp_price = safe_float(details.get("tp"))
    sl_price = safe_float(details.get("sl"))
    parts = [f"Trade placed {side}".strip()]
    if entry is not None:
        parts.append(f"@ {entry:.2f}")
    if tp_price is not None:
        parts.append(f"TP {tp_price:.2f}")
    if sl_price is not None:
        parts.append(f"SL {sl_price:.2f}")
    return " | ".join(parts)


def format_system_activity(strategy_id: Optional[str], message: str, details: Optional[Dict[str, str]] = None) -> str:
    text = " ".join(str(message or "").strip().split())
    details = details or {}
    if not text:
        return "Strategy ready"

    if strategy_id == "dynamic_engine3":
        sub_count_match = re.search(r"(\d+)\s+sub-strategies loaded", text, re.IGNORECASE)
        db_match = re.search(r"db_version=([a-zA-Z0-9_\\.-]+)", text)
        parts = ["DE3 ready"]
        if db_match:
            parts.append(str(db_match.group(1)))
        if sub_count_match:
            parts.append(f"{sub_count_match.group(1)} setups")
        return " | ".join(parts)

    if strategy_id == "regime_adaptive":
        return "RegimeAdaptive ready"

    if strategy_id == "ml_physics":
        if "legacy SessionManager models detached" in text:
            return "ML Physics live model active"
        return "ML Physics ready"

    if strategy_id == "aetherflow":
        threshold_match = re.search(r"threshold(?:s)?(?: loaded)?:?.*?([0-9]+\\.[0-9]+)", text, re.IGNORECASE)
        if "threshold" in text.lower():
            if threshold_match:
                return f"AetherFlow thresholds loaded | {threshold_match.group(1)}"
            return "AetherFlow thresholds loaded"
        if "model loaded" in text.lower():
            return "AetherFlow model loaded"
        if "initialized for live execution" in text.lower():
            return "AetherFlow ready"
        return "AetherFlow ready"

    head = text.split("|", 1)[0].strip()
    return head or text


def trade_identity(trade: Dict[str, Any]) -> tuple[Any, ...]:
    order_id = trade.get("order_id")
    if order_id not in (None, ""):
        return ("order", str(order_id))
    entry_order_id = trade.get("entry_order_id")
    if entry_order_id not in (None, ""):
        return ("entry_order", str(entry_order_id))
    return (
        "snapshot",
        str(trade.get("strategy_id") or ""),
        str(trade.get("side") or ""),
        round(safe_float(trade.get("exit_price")) or 0.0, 4),
        round(safe_float(trade.get("pnl_dollars")) or 0.0, 2),
    )


def trade_close_signature(trade: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(trade.get("strategy_id") or ""),
        str(trade.get("side") or ""),
        round(safe_float(trade.get("entry_price")) or 0.0, 4),
        round(safe_float(trade.get("exit_price")) or 0.0, 4),
        round(safe_float(trade.get("pnl_dollars")) or 0.0, 2),
    )


def trades_represent_same_close(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    if trade_identity(existing) == trade_identity(incoming):
        return True
    if trade_close_signature(existing) != trade_close_signature(incoming):
        return False
    existing_size = int(safe_float(existing.get("size")) or 0)
    incoming_size = int(safe_float(incoming.get("size")) or 0)
    if existing_size > 0 and incoming_size > 0 and existing_size != incoming_size:
        return False
    existing_time = parse_iso(existing.get("time"))
    incoming_time = parse_iso(incoming.get("time"))
    if existing_time is None or incoming_time is None:
        return False
    return abs((existing_time - incoming_time).total_seconds()) <= 15.0


def normalize_trade_realized_pnl(trade: Dict[str, Any]) -> Dict[str, Any]:
    trade = dict(trade)
    explicit_net = safe_float(trade.get("pnl_dollars_net"))
    explicit_fee = safe_float(trade.get("pnl_fee_dollars"))
    gross = safe_float(trade.get("pnl_dollars_gross"))
    if gross is None:
        if explicit_net is not None and explicit_fee is not None:
            gross = explicit_net + explicit_fee
        else:
            gross = safe_float(trade.get("pnl_dollars"))
    gross = float(gross or 0.0)
    size = int(max(1, safe_float(trade.get("size")) or 1))
    if explicit_fee is not None:
        fee = round(float(explicit_fee), 2)
    else:
        fee = round(float(MES_ROUND_TURN_FEE) * size, 2)
    if explicit_net is not None:
        net = round(float(explicit_net), 2)
    else:
        net = round(gross - fee, 2)
    trade["pnl_dollars_gross"] = round(gross, 2)
    trade["pnl_fee_dollars"] = fee
    trade["pnl_dollars_net"] = net
    trade["pnl_dollars"] = net
    trade["result"] = "win" if net > 0 else "loss" if net < 0 else "flat"
    return trade


def make_daily_trade_signature(trade: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        trade_identity(trade),
        str(trade.get("time") or ""),
        int(max(1, safe_float(trade.get("size")) or 1)),
    )


def track_daily_realized_pnl(
    tracker: Dict[str, Any],
    trade: Dict[str, Any],
) -> None:
    start = tracker.get("start")
    if not isinstance(start, datetime):
        return
    trade_time = parse_iso(trade.get("time"))
    if trade_time is None or trade_time < start:
        return
    signature = make_daily_trade_signature(trade)
    seen = tracker.setdefault("seen", set())
    if signature in seen:
        return
    seen.add(signature)
    tracker["net_pnl"] = round(float(tracker.get("net_pnl", 0.0) or 0.0) + float(safe_float(trade.get("pnl_dollars")) or 0.0), 2)


def compute_session_realized_pnl(trades: list[Dict[str, Any]], session_start: Optional[datetime]) -> float:
    if not isinstance(session_start, datetime):
        return 0.0
    total = 0.0
    for trade in trades:
        trade_time = parse_iso(trade.get("time"))
        if trade_time is None or trade_time < session_start:
            continue
        normalized = normalize_trade_realized_pnl(trade)
        total += float(safe_float(normalized.get("pnl_dollars")) or 0.0)
    return round(total, 2)


def upsert_trade(dashboard: Dict[str, Any], trade: Dict[str, Any], limit: int) -> None:
    trade = normalize_trade_realized_pnl(trade)
    strategy_id = str(trade.get("strategy_id") or "").strip()
    if not strategy_id:
        return
    if strategy_id in STRATEGY_LABELS:
        trade["strategy_label"] = STRATEGY_LABELS[strategy_id]
    else:
        fallback_label = str(trade.get("strategy_label") or "").strip() or "Recovered Trade"
        trade["strategy_label"] = fallback_label
    bucket = dashboard["trades"]
    for index, existing in enumerate(bucket):
        if trades_represent_same_close(existing, trade):
            merged = dict(existing)
            merged.update({key: value for key, value in trade.items() if value is not None})
            bucket[index] = merged
            break
    else:
        bucket.append(trade)
    bucket.sort(key=lambda row: str(row.get("time") or ""))
    if len(bucket) > limit:
        del bucket[0 : len(bucket) - limit]


def apply_trade_summary(
    dashboard: Dict[str, Any],
    trade: Dict[str, Any],
    *,
    updated_at: Optional[str],
    touch_runtime_state: bool,
) -> None:
    trade = normalize_trade_realized_pnl(trade)
    strategy_id = trade.get("strategy_id")
    if strategy_id not in dashboard["strategies"]:
        return
    strategy = dashboard["strategies"][strategy_id]
    if touch_runtime_state or dashboard["bot"].get("current_position") is None:
        strategy["status"] = "idle"
        strategy["updated_at"] = updated_at
    strategy["last_trade_pnl"] = safe_float(trade.get("pnl_dollars"))
    strategy["last_trade_points"] = safe_float(trade.get("pnl_points"))
    strategy["last_trade_time"] = updated_at
    strategy["last_trade_side"] = trade.get("side")
    strategy["last_trade_entry"] = safe_float(trade.get("entry_price"))
    strategy["last_trade_exit"] = safe_float(trade.get("exit_price"))
    if touch_runtime_state:
        current_position = dashboard["bot"].get("current_position")
        if isinstance(current_position, dict) and current_position.get("strategy_id") == strategy_id:
            dashboard["bot"]["current_position"] = None


def build_empty_state(log_path: Path, state_path: Path, trade_factors_path: Path) -> Dict[str, Any]:
    now = datetime.now(NY_TZ)
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "meta": {
            "log_path": str(log_path),
            "state_path": str(state_path),
            "trade_factors_path": str(trade_factors_path),
        },
        "bot": {
            "status": "offline",
            "session": None,
            "price": None,
            "trading_day_start": None,
            "last_bar_time": None,
            "last_ready_time": None,
            "last_heartbeat_time": None,
            "heartbeat_age_seconds": None,
            "session_connection_ok": False,
            "last_position_sync_time": None,
            "position_sync_status": None,
            "current_position": None,
            "price_history": [],
            "price_history_ohlc": [],
            "risk": {
                "daily_pnl": None,
                "circuit_consecutive_losses": None,
                "circuit_tripped": None,
                "long_consecutive_losses": None,
                "short_consecutive_losses": None,
                "long_blocked_until": None,
                "short_blocked_until": None,
                "reversed_bias": None,
                "hostile_day_active": None,
                "hostile_day_reason": None,
            },
            "warnings": [],
        },
        "strategies": {strategy_id: strategy_state_template(strategy_id) for strategy_id in STRATEGY_ORDER},
        "events": [],
        "trades": [],
        "kalshi_metrics": None,
        "sentiment_metrics": default_sentiment_metrics(),
        "pipeline": None,
        "daily_journals": [],
    }


def build_sentiment_metrics_from_snapshot(snapshot: Any) -> Optional[Dict[str, Any]]:
    base = default_sentiment_metrics()
    if not isinstance(snapshot, dict):
        return dict(base) if isinstance(base, dict) else None
    base_metadata = dict((base or {}).get("metadata") or {})
    snapshot_metadata = snapshot.get("metadata")
    if isinstance(snapshot_metadata, dict):
        base_metadata.update(snapshot_metadata)
    else:
        base_metadata.update(parse_json_object(snapshot_metadata))
    return {
        "enabled": bool(snapshot.get("enabled", (base or {}).get("enabled", False))),
        "active": bool(snapshot.get("active", (base or {}).get("active", False))),
        "healthy": bool(snapshot.get("healthy", (base or {}).get("healthy", False))),
        "model_loaded": bool(snapshot.get("model_loaded", (base or {}).get("model_loaded", False))),
        "quantized_8bit": bool(snapshot.get("quantized_8bit", (base or {}).get("quantized_8bit", False))),
        "target_handle": snapshot.get("target_handle") or (base or {}).get("target_handle"),
        "source": snapshot.get("source") or (base or {}).get("source"),
        "last_poll_at": snapshot.get("last_poll_at"),
        "last_analysis_at": snapshot.get("last_analysis_at"),
        "latest_post_id": snapshot.get("latest_post_id"),
        "latest_post_created_at": snapshot.get("latest_post_created_at"),
        "latest_post_url": snapshot.get("latest_post_url"),
        "latest_post_text": snapshot.get("latest_post_text"),
        "sentiment_label": snapshot.get("sentiment_label"),
        "sentiment_score": safe_float(snapshot.get("sentiment_score")),
        "finbert_confidence": safe_float(snapshot.get("finbert_confidence")),
        "trigger_side": snapshot.get("trigger_side"),
        "trigger_reason": snapshot.get("trigger_reason"),
        "last_error": normalize_sentiment_error_message(snapshot.get("last_error")),
        "metadata": base_metadata,
    }


def build_kalshi_metrics_from_snapshot(snapshot: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return None
    enabled = bool(snapshot.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "healthy": False,
            "updated_at": snapshot.get("updated_at"),
            "basis_offset": safe_float(snapshot.get("basis_offset")) or 0.0,
            "probability_60m": None,
            "probability_reference_kind": None,
            "probability_reference_side": None,
            "probability_reference_es_price": None,
            "probability_contract_es_price": None,
            "probability_contract_spx_price": None,
            "probability_contract_probability": None,
            "probability_contract_outcome": None,
            "probability_contract_distance_es": None,
            "event_ticker": None,
            "es_reference_price": None,
            "strikes": [],
        }

    basis_offset = safe_float(snapshot.get("basis_offset")) or 0.0
    spx_reference_price = safe_float(snapshot.get("spx_reference_price"))
    es_reference_price = safe_float(snapshot.get("es_reference_price"))
    if es_reference_price is None and spx_reference_price is not None:
        es_reference_price = spx_reference_price + basis_offset

    strikes: list[Dict[str, Any]] = []
    raw_strikes = snapshot.get("strikes")
    if isinstance(raw_strikes, list):
        for row in raw_strikes:
            if not isinstance(row, dict):
                continue
            strike = safe_float(row.get("strike"))
            probability = safe_float(row.get("probability"))
            if strike is None or probability is None:
                continue
            strikes.append(
                {
                    "strike": strike,
                    "probability": probability,
                    "volume": safe_float(row.get("volume")),
                    "status": row.get("status"),
                    "result": row.get("result"),
                }
            )

    return {
        "enabled": True,
        "healthy": bool(snapshot.get("healthy", True)),
        "requested": snapshot.get("requested"),
        "configured": snapshot.get("configured"),
        "observer_only": snapshot.get("observer_only"),
        "status_label": snapshot.get("status_label"),
        "status_reason": snapshot.get("status_reason"),
        "source": snapshot.get("source"),
        "updated_at": snapshot.get("updated_at"),
        "basis_offset": basis_offset,
        "probability_60m": safe_float(snapshot.get("probability_60m")),
        "probability_reference_kind": snapshot.get("probability_reference_kind"),
        "probability_reference_side": snapshot.get("probability_reference_side"),
        "probability_reference_es_price": safe_float(snapshot.get("probability_reference_es_price")),
        "probability_contract_es_price": safe_float(snapshot.get("probability_contract_es_price")),
        "probability_contract_spx_price": safe_float(snapshot.get("probability_contract_spx_price")),
        "probability_contract_probability": safe_float(snapshot.get("probability_contract_probability")),
        "probability_contract_outcome": snapshot.get("probability_contract_outcome"),
        "probability_contract_distance_es": safe_float(snapshot.get("probability_contract_distance_es")),
        "event_ticker": snapshot.get("event_ticker"),
        "es_reference_price": es_reference_price,
        "spx_reference_price": spx_reference_price,
        "trade_gating_active": snapshot.get("trade_gating_active"),
        "trade_gating_hour": snapshot.get("trade_gating_hour"),
        "strikes": strikes,
        "daily_contracts": snapshot.get("daily_contracts"),
    }


def update_state(
    dashboard: Dict[str, Any],
    *,
    kalshi_snapshot: Optional[Dict[str, Any]] = None,
    kalshi_provider: Optional[Any] = None,
) -> Dict[str, Any]:
    if kalshi_provider is not None:
        bot_price = safe_float((dashboard.get("bot") or {}).get("price"))
        current_position = (dashboard.get("bot") or {}).get("current_position")
        target_price = None
        target_side = None
        if isinstance(current_position, dict):
            target_price = safe_float(current_position.get("target_price"))
            if target_price is None:
                target_price = safe_float(current_position.get("tp_price"))
            side = str(current_position.get("side") or "").strip().upper()
            target_side = side if side in {"LONG", "SHORT"} else None
        sentiment = kalshi_provider.get_sentiment(bot_price) if bot_price is not None else {}
        target_probability = (
            kalshi_provider.get_target_probability(target_price, target_side)
            if target_price is not None
            else {}
        )
        ui_reference_prices = [ref for ref in (bot_price, target_price) if ref is not None]
        strikes = (
            kalshi_provider.get_relative_markets_for_ui(ui_reference_prices, window_size=30)
            if getattr(kalshi_provider, "enabled", False)
            else []
        )
        basis_offset = float(getattr(kalshi_provider, "basis_offset", 0.0) or 0.0)
        dashboard["kalshi_metrics"] = {
            "enabled": bool(getattr(kalshi_provider, "enabled", False)),
            "healthy": bool(getattr(kalshi_provider, "is_healthy", False)),
            "updated_at": datetime.now(NY_TZ).isoformat(),
            "basis_offset": basis_offset,
            "probability_60m": (
                safe_float((target_probability or {}).get("probability"))
                if safe_float((target_probability or {}).get("probability")) is not None
                else safe_float((sentiment or {}).get("probability"))
            ),
            "probability_reference_kind": (
                "open_position_target"
                if safe_float((target_probability or {}).get("probability")) is not None
                else "current_price"
            ),
            "probability_reference_side": target_side if safe_float((target_probability or {}).get("probability")) is not None else None,
            "probability_reference_es_price": (
                safe_float((target_probability or {}).get("reference_es"))
                if safe_float((target_probability or {}).get("probability")) is not None
                else bot_price
            ),
            "probability_contract_es_price": safe_float((target_probability or {}).get("strike_es")),
            "probability_contract_spx_price": safe_float((target_probability or {}).get("strike_spx")),
            "probability_contract_probability": safe_float((target_probability or {}).get("market_probability")),
            "probability_contract_outcome": (target_probability or {}).get("outcome_side"),
            "probability_contract_distance_es": safe_float((target_probability or {}).get("distance_es")),
            "event_ticker": kalshi_provider._current_event_ticker() if getattr(kalshi_provider, "enabled", False) else None,  # noqa: SLF001
            "es_reference_price": bot_price,
            "spx_reference_price": (
                (bot_price - basis_offset)
                if bot_price is not None
                else None
            ),
            "strikes": strikes if isinstance(strikes, list) else [],
        }
        return dashboard
    dashboard["kalshi_metrics"] = build_kalshi_metrics_from_snapshot(kalshi_snapshot)
    return dashboard


def build_kalshi_provider() -> Optional[Any]:
    kalshi_cfg = dict(CONFIG.get("KALSHI", {}) or {})
    if not kalshi_cfg:
        return None
    try:
        module = importlib.import_module("services.kalshi_provider")
        provider_class = getattr(module, "KalshiProvider", None)
        if provider_class is None:
            return None
        provider = provider_class(kalshi_cfg)
    except Exception:
        return None
    return provider


def load_trade_factor_index(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strategy_id = canonical_strategy_id(row.get("strategy"))
            if not strategy_id:
                continue
            order_details = parse_json_object(row.get("order_details_json"))
            signal_details = parse_json_object(row.get("signal_factors_json"))
            order_id = str(order_details.get("order_id") or "").strip()
            if not order_id:
                continue
            index[order_id] = {
                "size": safe_float(row.get("size")) or safe_float(order_details.get("size")),
                "entry_mode": row.get("entry_mode") or order_details.get("entry_mode"),
                "base_session": row.get("base_session"),
                "current_session": row.get("current_session"),
                "sub_strategy": row.get("sub_strategy"),
                "vol_regime": row.get("vol_regime"),
                "combo_key": signal_details.get("combo_key") or row.get("sub_strategy"),
                "rule_id": signal_details.get("rule_id"),
                "early_exit_enabled": signal_details.get("early_exit_enabled"),
                "gate_prob": safe_float(signal_details.get("gate_prob")),
                "gate_threshold": safe_float(signal_details.get("gate_threshold")),
            }
    return index


def summarize_trade_close_details(
    close_details: Any,
    *,
    fallback_gross: Optional[float] = None,
    fallback_size: Optional[float] = None,
) -> Dict[str, Any]:
    details = parse_json_object(close_details)
    gross = safe_float(details.get("pnl_dollars_gross"))
    fee = safe_float(details.get("pnl_fee_dollars"))
    net = safe_float(details.get("pnl_dollars_net"))

    gross_from_rows = 0.0
    fee_from_rows = 0.0
    fee_contracts = 0
    gross_found = False
    fee_found = False
    raw_close_rows = details.get("raw_close_rows")
    if isinstance(raw_close_rows, list):
        for row in raw_close_rows:
            if not isinstance(row, dict):
                continue
            pnl_value = safe_float(row.get("profitAndLoss"))
            if pnl_value is not None:
                gross_from_rows += float(pnl_value)
                gross_found = True
            row_size = abs(int(safe_float(row.get("size")) or 0))
            if row_size > 0:
                fee_contracts += row_size
            commissions = safe_float(row.get("commissions"))
            exchange_fees = safe_float(row.get("fees"))
            if commissions is not None or exchange_fees is not None:
                fee_from_rows += float(commissions or 0.0) + float(exchange_fees or 0.0)
                fee_found = True

    if gross is None:
        if gross_found:
            gross = gross_from_rows
        else:
            gross = fallback_gross
    if fee is None and fee_found:
        if fee_contracts <= 0:
            fee_contracts = max(0, int(safe_float(fallback_size) or 0))
        fee = fee_from_rows + (TOPSTEP_ROUND_TURN_COMMISSION * float(fee_contracts))
    if net is None and gross is not None and fee is not None:
        net = gross - fee

    return {
        "close_details": details,
        "pnl_dollars_gross": round(float(gross), 2) if gross is not None else None,
        "pnl_fee_dollars": round(float(fee), 2) if fee is not None else None,
        "pnl_dollars_net": round(float(net), 2) if net is not None else None,
    }


def prefer_trade_close_record(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    existing_fee = safe_float(existing.get("pnl_fee_dollars"))
    incoming_fee = safe_float(incoming.get("pnl_fee_dollars"))
    if (incoming_fee is not None) != (existing_fee is not None):
        return incoming_fee is not None

    existing_source = str(existing.get("source") or "").strip().lower()
    incoming_source = str(incoming.get("source") or "").strip().lower()
    if (incoming_source == "projectx_trade_history") != (existing_source == "projectx_trade_history"):
        return incoming_source == "projectx_trade_history"

    existing_time = parse_iso(existing.get("time"))
    incoming_time = parse_iso(incoming.get("time"))
    if incoming_time is not None and (existing_time is None or incoming_time >= existing_time):
        return True
    return False


def load_trade_close_index(path: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            close_order_id = str(row.get("close_order_id") or "").strip()
            entry_order_id = str(row.get("entry_order_id") or "").strip()
            if not close_order_id and not entry_order_id:
                continue
            close_summary = summarize_trade_close_details(
                row.get("close_details_json"),
                fallback_gross=safe_float(row.get("pnl_dollars")),
                fallback_size=safe_float(row.get("size")),
            )
            record = {
                "time": row.get("close_time") or row.get("event_time"),
                "source": row.get("close_source") or row.get("source"),
                "size": safe_float(row.get("size")),
                "entry_price": safe_float(row.get("entry_price")),
                "exit_price": safe_float(row.get("exit_price")),
                "pnl_points": safe_float(row.get("pnl_points")),
                "pnl_dollars_gross": close_summary.get("pnl_dollars_gross"),
                "pnl_fee_dollars": close_summary.get("pnl_fee_dollars"),
                "pnl_dollars_net": close_summary.get("pnl_dollars_net"),
                "close_details": close_summary.get("close_details"),
            }
            for key in (
                ("order", close_order_id),
                ("entry_order", entry_order_id),
            ):
                if not key[1]:
                    continue
                existing = index.get(key)
                if existing is None or prefer_trade_close_record(existing, record):
                    index[key] = record
    return index


def enrich_trade_from_close_index(
    trade: Dict[str, Any],
    trade_close_index: Dict[tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    enriched = dict(trade)
    record = None
    order_id = str(enriched.get("order_id") or "").strip()
    entry_order_id = str(enriched.get("entry_order_id") or "").strip()
    for key in (
        ("order", order_id),
        ("entry_order", entry_order_id),
    ):
        if not key[1]:
            continue
        record = trade_close_index.get(key)
        if record is not None:
            break
    if record is None:
        return enriched

    for field in (
        "size",
        "entry_price",
        "exit_price",
        "pnl_points",
    ):
        if enriched.get(field) in (None, "") and record.get(field) is not None:
            enriched[field] = record.get(field)

    for field in (
        "pnl_dollars_gross",
        "pnl_fee_dollars",
        "pnl_dollars_net",
    ):
        if record.get(field) is not None:
            enriched[field] = record.get(field)

    if record.get("source") and not enriched.get("close_source"):
        enriched["close_source"] = record.get("source")
    if record.get("close_details") and not enriched.get("close_details"):
        enriched["close_details"] = record.get("close_details")
    return enriched


def apply_bot_state_snapshot(
    dashboard: Dict[str, Any],
    persisted_state: Dict[str, Any],
    *,
    max_trades: int,
    trade_close_index: Dict[tuple[str, str], Dict[str, Any]],
    daily_tracker: Optional[Dict[str, Any]] = None,
) -> None:
    if not persisted_state:
        dashboard["bot"]["warnings"].append("bot_state.json not found or unreadable")
        return

    persisted_ts = parse_iso(persisted_state.get("timestamp"))
    bank_filter = persisted_state.get("bank_filter") or {}
    circuit = persisted_state.get("circuit_breaker") or {}
    directional = persisted_state.get("directional_loss_blocker") or {}
    hostile = persisted_state.get("hostile_day") or {}
    runtime_base_session = persisted_state.get("base_session")
    runtime_current_session = persisted_state.get("current_session") or persisted_state.get("session")

    dashboard["bot"]["session"] = (
        bank_filter.get("current_session")
        or (persisted_state.get("rejection_filter") or {}).get("current_session_name")
        or runtime_current_session
        or runtime_base_session
        or dashboard["bot"].get("session")
    )
    trading_day_start_value = parse_iso(persisted_state.get("trading_day_start"))
    if trading_day_start_value is not None:
        dashboard["bot"]["trading_day_start"] = iso_or_none(trading_day_start_value)
    dashboard["bot"]["risk"] = {
        "daily_pnl": safe_float(circuit.get("daily_pnl")),
        "circuit_consecutive_losses": circuit.get("consecutive_losses"),
        "circuit_tripped": circuit.get("is_tripped"),
        "long_consecutive_losses": directional.get("long_consecutive_losses"),
        "short_consecutive_losses": directional.get("short_consecutive_losses"),
        "long_blocked_until": directional.get("long_blocked_until"),
        "short_blocked_until": directional.get("short_blocked_until"),
        "reversed_bias": directional.get("reversed_bias"),
        "hostile_day_active": hostile.get("hostile_day_active"),
        "hostile_day_reason": hostile.get("hostile_day_reason"),
    }
    sentiment_snapshot = build_sentiment_metrics_from_snapshot(persisted_state.get("sentiment"))
    dashboard["sentiment_metrics"] = sentiment_snapshot

    pipeline_snapshot = persisted_state.get("pipeline")
    if isinstance(pipeline_snapshot, dict):
        dashboard["pipeline"] = pipeline_snapshot

    ohlc_snapshot = persisted_state.get("price_history_ohlc")
    if isinstance(ohlc_snapshot, list) and ohlc_snapshot:
        dashboard["bot"]["price_history_ohlc"] = ohlc_snapshot
    else:
        # Sidecar fallback: tools/tick_chart_poller.py writes 15-sec OHLC
        # bars to artifacts/tick_chart_ohlc.json. Read it if the in-bot
        # tick stream hasn't populated bot_state.json yet.
        sidecar_path = ROOT / "artifacts" / "tick_chart_ohlc.json"
        if sidecar_path.exists():
            try:
                sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if isinstance(sidecar_payload, list) and sidecar_payload:
                    dashboard["bot"]["price_history_ohlc"] = sidecar_payload
            except (OSError, json.JSONDecodeError):
                pass

    live_position = persisted_state.get("live_position")
    live_position_ts = None
    latest_logged_sync = parse_iso(dashboard["bot"].get("last_position_sync_time"))
    if isinstance(live_position, dict):
        live_position_ts = parse_iso(live_position.get("updated_at")) or persisted_ts
    live_position_is_stale = (
        isinstance(live_position, dict)
        and live_position_ts is not None
        and latest_logged_sync is not None
        and latest_logged_sync > live_position_ts
    )
    if live_position_is_stale:
        dashboard["bot"]["warnings"].append(
            "Ignoring stale live_position snapshot older than latest broker position sync"
        )

    if isinstance(live_position, dict) and not live_position_is_stale:
        side = str(live_position.get("side") or "").strip().upper()
        size = safe_float(live_position.get("size"))
        if side in {"LONG", "SHORT"} and size not in (None, 0):
            strategy_id = canonical_strategy_id(
                live_position.get("strategy_id")
                or live_position.get("strategy")
                or live_position.get("strategy_label")
            )
            strategy_label = STRATEGY_LABELS.get(strategy_id, "Open Position")
            current_position = dashboard["bot"].get("current_position")
            if not isinstance(current_position, dict):
                current_position = {
                    "strategy_id": strategy_id or "unknown",
                    "strategy_label": strategy_label,
                }
            current_position_opened_at = parse_iso(current_position.get("opened_at"))
            live_position_opened_at = parse_iso(live_position.get("opened_at"))
            current_position_size = safe_float(current_position.get("size"))
            preserve_log_trade_metadata = (
                current_position_opened_at is not None
                and live_position_opened_at is not None
                and current_position_opened_at > live_position_opened_at
                and str(current_position.get("side") or "").strip().upper() == side
                and current_position_size not in (None, 0)
                and size not in (None, 0)
                and int(current_position_size) == int(size)
            )
            if strategy_id is not None:
                current_position["strategy_id"] = strategy_id
                current_position["strategy_label"] = STRATEGY_LABELS.get(strategy_id, strategy_label)
                strategy = dashboard["strategies"].get(strategy_id)
                if strategy is not None:
                    strategy["status"] = "in_trade"
                    strategy["updated_at"] = live_position.get("updated_at") or strategy.get("updated_at")
            for key in (
                "source",
                "side",
                "size",
                "avg_price",
                "entry_price",
                "signal_entry_price",
                "opened_at",
                "order_id",
                "rule_id",
                "combo_key",
                "early_exit_enabled",
                "gate_prob",
                "gate_threshold",
                "kalshi_gate_applied",
                "kalshi_gate_reason",
                "kalshi_gate_multiplier",
                "kalshi_trade_overlay_applied",
                "kalshi_trade_overlay_reason",
                "kalshi_trade_overlay_role",
                "kalshi_trade_overlay_mode",
                "kalshi_trade_overlay_forward_weight",
                "kalshi_curve_informative",
                "kalshi_entry_probability",
                "kalshi_entry_support_score",
                "kalshi_entry_threshold",
                "kalshi_tp_anchor_price",
                "kalshi_tp_anchor_probability",
                "kalshi_tp_trail_enabled",
                "entry_mode",
                "base_session",
                "current_session",
                "session",
                "sub_strategy",
                "vol_regime",
                "open_pnl_points",
                "open_pnl_dollars",
                "point_value",
                "stop_price",
                "sl_price",
                "target_price",
                "tp_price",
            ):
                if preserve_log_trade_metadata and key in {
                    "opened_at",
                    "order_id",
                    "rule_id",
                    "combo_key",
                    "early_exit_enabled",
                    "gate_prob",
                    "gate_threshold",
                    "kalshi_gate_applied",
                    "kalshi_gate_reason",
                    "kalshi_gate_multiplier",
                    "kalshi_trade_overlay_applied",
                    "kalshi_trade_overlay_reason",
                    "kalshi_trade_overlay_role",
                    "kalshi_trade_overlay_mode",
                    "kalshi_trade_overlay_forward_weight",
                    "kalshi_curve_informative",
                    "kalshi_entry_probability",
                    "kalshi_entry_support_score",
                    "kalshi_entry_threshold",
                    "kalshi_tp_anchor_price",
                    "kalshi_tp_anchor_probability",
                    "kalshi_tp_trail_enabled",
                    "entry_mode",
                    "base_session",
                    "current_session",
                    "session",
                    "sub_strategy",
                    "vol_regime",
                    "signal_entry_price",
                    "stop_price",
                    "sl_price",
                    "target_price",
                    "tp_price",
                }:
                    continue
                if key in live_position and live_position.get(key) is not None:
                    current_position[key] = live_position.get(key)
            for numeric_key in (
                "gate_prob",
                "gate_threshold",
                "kalshi_gate_multiplier",
                "kalshi_trade_overlay_forward_weight",
                "kalshi_entry_probability",
                "kalshi_entry_support_score",
                "kalshi_entry_threshold",
                "kalshi_tp_anchor_price",
                "kalshi_tp_anchor_probability",
            ):
                numeric_value = safe_float(current_position.get(numeric_key))
                if numeric_value is not None:
                    current_position[numeric_key] = numeric_value
            avg_price = safe_float(live_position.get("avg_price"))
            if avg_price is not None:
                current_position["avg_price"] = avg_price
                current_position["entry_price"] = avg_price
            dashboard["bot"]["current_position"] = current_position
            if strategy_id in dashboard["strategies"]:
                strategy = dashboard["strategies"][strategy_id]
                for key in (
                    "sub_strategy",
                    "combo_key",
                    "rule_id",
                    "early_exit_enabled",
                    "gate_prob",
                    "gate_threshold",
                    "kalshi_gate_applied",
                    "kalshi_gate_reason",
                    "kalshi_gate_multiplier",
                    "kalshi_trade_overlay_applied",
                    "kalshi_trade_overlay_reason",
                    "kalshi_trade_overlay_role",
                    "kalshi_trade_overlay_mode",
                    "kalshi_trade_overlay_forward_weight",
                    "kalshi_curve_informative",
                    "kalshi_entry_probability",
                    "kalshi_entry_support_score",
                    "kalshi_entry_threshold",
                    "kalshi_tp_anchor_price",
                    "kalshi_tp_anchor_probability",
                    "kalshi_tp_trail_enabled",
                    "entry_mode",
                    "base_session",
                    "current_session",
                    "session",
                    "vol_regime",
                ):
                    if current_position.get(key) is not None:
                        strategy[key] = current_position.get(key)
            updated_at = live_position.get("updated_at")
            if isinstance(updated_at, str) and updated_at:
                dashboard["bot"]["last_position_sync_time"] = updated_at
            side_label = side
            if size is not None:
                side_label = f"{side_label} {int(size)}"
            if avg_price is not None:
                side_label = f"{side_label} @ {avg_price:.2f}"
            dashboard["bot"]["position_sync_status"] = side_label
    elif "live_position" in persisted_state and live_position is None:
        last_sync = parse_iso(dashboard["bot"].get("last_position_sync_time"))
        ready = parse_iso(dashboard["bot"].get("last_ready_time"))
        grace_elapsed = (
            persisted_ts is not None
            and (ready is None or (persisted_ts - ready).total_seconds() >= 20.0)
        )
        newer_than_sync = persisted_ts is not None and (last_sync is None or persisted_ts >= last_sync)
        if grace_elapsed and newer_than_sync:
            current_position = dashboard["bot"].get("current_position")
            if isinstance(current_position, dict):
                strategy_id = current_position.get("strategy_id")
                if strategy_id in dashboard["strategies"]:
                    strategy = dashboard["strategies"][strategy_id]
                    strategy["status"] = "idle"
                    strategy["updated_at"] = iso_or_none(persisted_ts)
            dashboard["bot"]["current_position"] = None
            dashboard["bot"]["position_sync_status"] = "FLAT"
            dashboard["bot"]["last_position_sync_time"] = iso_or_none(persisted_ts)
            dashboard["bot"]["warnings"].append("Cleared stale open position from older logs using bot_state flat snapshot")

    recent_closed_trades = persisted_state.get("recent_closed_trades")
    if isinstance(recent_closed_trades, list):
        for row in recent_closed_trades:
            if not isinstance(row, dict):
                continue
            strategy_id, strategy_label = resolve_trade_strategy_identity(
                row.get("strategy_id")
                or row.get("strategy")
                or row.get("strategy_label"),
                row.get("strategy_label") or row.get("strategy"),
            )
            if strategy_id is None or strategy_label is None:
                continue
            trade_time = parse_iso(
                row.get("time")
                or row.get("closed_at")
                or row.get("exit_time")
                or row.get("updated_at")
            )
            trade = {
                "time": iso_or_none(trade_time),
                "strategy_id": strategy_id,
                "strategy_label": strategy_label,
                "side": str(row.get("side") or "").strip().upper() or None,
                "size": int(safe_float(row.get("size")) or 0) or None,
                "entry_price": safe_float(row.get("entry_price")),
                "signal_entry_price": safe_float(row.get("signal_entry_price")),
                "exit_price": safe_float(row.get("exit_price")),
                "pnl_points": safe_float(row.get("pnl_points")),
                "pnl_dollars": safe_float(row.get("pnl_dollars")),
                "pnl_dollars_gross": safe_float(row.get("pnl_dollars_gross")),
                "pnl_fee_dollars": safe_float(row.get("pnl_fee_dollars")),
                "pnl_dollars_net": safe_float(row.get("pnl_dollars_net")),
                "result": row.get("result"),
                "source": row.get("source"),
                "order_id": row.get("order_id"),
                "entry_order_id": row.get("entry_order_id"),
                "opened_at": row.get("opened_at"),
            }
            trade = enrich_trade_from_close_index(trade, trade_close_index)
            upsert_trade(dashboard, trade, max_trades)
            if isinstance(daily_tracker, dict):
                track_daily_realized_pnl(daily_tracker, trade)
            apply_trade_summary(
                dashboard,
                trade,
                updated_at=trade["time"],
                touch_runtime_state=False,
            )


def enrich_position(position: Optional[Dict[str, Any]], bot: Dict[str, Any]) -> None:
    if not position:
        return
    source = str(position.get("source") or "").strip().lower()
    current_price = safe_float(position.get("current_price"))
    if current_price is None:
        current_price = safe_float(bot.get("price"))
    entry_price = safe_float(position.get("entry_price"))
    if entry_price is None:
        entry_price = safe_float(position.get("avg_price"))
    if entry_price is None:
        entry_price = safe_float(position.get("signal_entry_price"))
    side = position.get("side")
    size = safe_float(position.get("size"))
    point_value = safe_float(position.get("point_value"))
    if point_value is None or point_value <= 0.0:
        point_value = 5.0
    stop_price = safe_float(position.get("sl_price"))
    target_price = safe_float(position.get("tp_price"))
    broker_open_pnl = safe_float(position.get("open_pnl_dollars"))
    broker_open_points = safe_float(position.get("open_pnl_points"))

    if broker_open_pnl is not None or broker_open_points is not None:
        if broker_open_pnl is None and broker_open_points is not None and size not in (None, 0):
            position["open_pnl_dollars"] = round(broker_open_points * point_value * size, 2)
        elif broker_open_pnl is not None:
            position["open_pnl_dollars"] = round(broker_open_pnl, 2)
        if broker_open_points is None and broker_open_pnl is not None and size not in (None, 0):
            position["open_pnl_points"] = round(broker_open_pnl / (point_value * size), 2)
        elif broker_open_points is not None:
            position["open_pnl_points"] = round(broker_open_points, 2)
        position["stop_price"] = stop_price
        position["target_price"] = target_price
        return

    if current_price is None or entry_price is None or side not in {"LONG", "SHORT"}:
        position["stop_price"] = stop_price
        position["target_price"] = target_price
        return
    if side == "LONG":
        points = current_price - entry_price
    else:
        points = entry_price - current_price
    position["open_pnl_points"] = round(points, 2)
    position["open_pnl_dollars"] = round(points * point_value * size, 2) if size is not None else None
    position["stop_price"] = stop_price
    position["target_price"] = target_price


def finalize_dashboard(dashboard: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(NY_TZ)
    dashboard["generated_at"] = now.isoformat()
    heartbeat = parse_iso(dashboard["bot"].get("last_heartbeat_time"))
    ready = parse_iso(dashboard["bot"].get("last_ready_time"))
    bar_time = parse_iso(dashboard["bot"].get("last_bar_time"))
    if heartbeat is not None:
        age = max(0.0, (now - heartbeat).total_seconds())
        dashboard["bot"]["heartbeat_age_seconds"] = round(age, 1)
        dashboard["bot"]["status"] = "online" if age <= 90 else "stale"
        if age > 90:
            dashboard["bot"]["warnings"].append("Heartbeat is stale")
    elif ready is not None and (now - ready).total_seconds() <= 90:
        dashboard["bot"]["heartbeat_age_seconds"] = None
        dashboard["bot"]["status"] = "online"
        dashboard["bot"]["warnings"].append("Awaiting first heartbeat after startup")
    elif bar_time is not None and (now - bar_time).total_seconds() <= 90:
        dashboard["bot"]["heartbeat_age_seconds"] = None
        dashboard["bot"]["status"] = "online"
        dashboard["bot"]["warnings"].append("Awaiting first heartbeat; recent market data is flowing")
    elif dashboard["bot"].get("last_bar_time"):
        dashboard["bot"]["status"] = "stale"
    else:
        dashboard["bot"]["status"] = "offline"

    bot_status = str(dashboard["bot"].get("status") or "").strip().lower()
    if bot_status in {"stale", "offline"}:
        for strategy in dashboard["strategies"].values():
            strategy["status"] = bot_status
        if bot_status == "stale":
            dashboard["bot"]["warnings"].append("Strategy statuses marked stale because bot heartbeat is stale")
        else:
            dashboard["bot"]["warnings"].append("Strategy statuses marked offline because bot is offline")

    enrich_position(dashboard["bot"].get("current_position"), dashboard["bot"])

    if not dashboard["bot"]["price_history"]:
        dashboard["bot"]["warnings"].append("No price history found in log")
    if not dashboard["events"]:
        dashboard["bot"]["warnings"].append("No filterless strategy events found in log")

    current_position = dashboard["bot"].get("current_position")
    for strategy_id in FILTERLESS_LIVE_DISABLED_STRATEGIES:
        strategy = dashboard["strategies"].get(strategy_id)
        if strategy is None:
            continue
        if isinstance(current_position, dict) and current_position.get("strategy_id") == strategy_id:
            continue
        message = disabled_strategy_message(strategy_id)
        strategy["status"] = "disabled"
        strategy["updated_at"] = strategy.get("updated_at") or now.isoformat()
        strategy["latest_activity"] = message
        strategy["latest_activity_time"] = strategy.get("latest_activity_time") or strategy["updated_at"]
        strategy["latest_activity_type"] = "SYSTEM"
        strategy["latest_activity_severity"] = "info"
        strategy["last_reason"] = message
        strategy["last_block_reason"] = message

    dashboard["strategies"] = [dashboard["strategies"][strategy_id] for strategy_id in STRATEGY_ORDER]
    dashboard["events"] = list(reversed(dashboard["events"]))
    dashboard["trades"] = list(reversed(dashboard["trades"]))
    # Hydrate daily-journal summaries (most recent first, capped at 14 days)
    try:
        dashboard["daily_journals"] = load_recent_daily_journals(max_entries=14)
    except Exception as exc:
        logging.debug("load_recent_daily_journals failed: %s", exc)
        dashboard["daily_journals"] = []
    return dashboard


_JOURNAL_DIR = Path(__file__).resolve().parent.parent / "ai_loop_data" / "journals"


def load_recent_daily_journals(max_entries: int = 14) -> List[Dict[str, Any]]:
    """Return the most recent EOD journal summaries from ai_loop_data/journals/.

    Ignores backtest_*.json. Sorted newest-first. Each entry is a slim dict
    with the fields the dashboard needs (date, summary, breakdown_by_layer,
    pattern_flags, price_context.range_pts/trend_pts/trend_dir).
    """
    if not _JOURNAL_DIR.exists():
        return []
    candidates: List[Tuple[str, Path]] = []
    for child in _JOURNAL_DIR.glob("*.json"):
        if child.name.startswith("backtest_"):
            continue
        # Expect 'YYYY-MM-DD.json' filenames
        stem = child.stem
        if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
            candidates.append((stem, child))
    candidates.sort(reverse=True)
    out: List[Dict[str, Any]] = []
    for date_str, path in candidates[:max_entries]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary") or {}
        price = payload.get("price_context") or {}
        out.append({
            "date": payload.get("date") or date_str,
            "summary": {
                "total_pnl": summary.get("total_pnl"),
                "n_trades": summary.get("n_trades"),
                "n_wins": summary.get("n_wins"),
                "n_losses": summary.get("n_losses"),
                "win_rate": summary.get("win_rate"),
                "max_drawdown": summary.get("max_drawdown"),
                "n_signals_fired": summary.get("n_signals_fired"),
                "n_kalshi_blocks": summary.get("n_kalshi_blocks"),
                "n_signals_blocked_strategy": summary.get("n_signals_blocked_strategy"),
            },
            "breakdown_by_layer": payload.get("breakdown_by_layer") or {},
            "pattern_flags": payload.get("pattern_flags") or [],
            "price_context": {
                "range_pts": price.get("range_pts"),
                "trend_pts": price.get("trend_pts"),
                "trend_dir": price.get("trend_dir"),
                "open": price.get("open"),
                "close": price.get("close"),
                "high": price.get("high"),
                "low": price.get("low"),
            },
        })
    return out


def record_event(dashboard: Dict[str, Any], max_events: int, when: Optional[datetime], event_type: str, message: str,
                 severity: str = "info", strategy_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
    item = {
        "time": iso_or_none(when),
        "event_type": event_type,
        "severity": severity.lower(),
        "strategy_id": strategy_id,
        "strategy_label": STRATEGY_LABELS.get(strategy_id) if strategy_id else None,
        "message": message,
        "details": details or {},
    }
    if should_skip_duplicate_event(dashboard["events"], item):
        return
    append_limited(dashboard["events"], item, max_events)


def build_dashboard_state(
    log_path: Path,
    state_path: Path,
    trade_factors_path: Path,
    max_events: int,
    max_trades: int,
    max_price_points: int,
    kalshi_snapshot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    dashboard = build_empty_state(log_path, state_path, trade_factors_path)
    kalshi_provider = build_kalshi_provider()
    trade_factor_index = load_trade_factor_index(trade_factors_path)
    trade_close_index = load_trade_close_index(trade_factors_path)
    persisted_state = load_bot_state(state_path)
    persisted_ts = parse_iso((persisted_state or {}).get("timestamp")) if isinstance(persisted_state, dict) else None
    trading_day_start_dt = parse_iso((persisted_state or {}).get("trading_day_start")) if isinstance(persisted_state, dict) else None
    if trading_day_start_dt is None:
        trading_day_start_dt = session_day_start(persisted_ts)
    dashboard["bot"]["trading_day_start"] = iso_or_none(trading_day_start_dt)
    daily_tracker: Dict[str, Any] = {
        "start": trading_day_start_dt,
        "seen": set(),
        "net_pnl": 0.0,
    }

    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                prefix_match = LOG_PREFIX_RE.match(line)
                if not prefix_match:
                    continue
                logged_at = parse_logged_at(prefix_match.group("logged_at"))
                level = prefix_match.group("level").lower()
                message = prefix_match.group("message")

                structured_match = STRUCTURED_EVENT_RE.match(message)
                if structured_match:
                    event_type = structured_match.group("event_type")
                    event_message, details = parse_pipe_details(structured_match.group("body"))
                    strategy_id = canonical_strategy_id(details.get("strategy"))
                    strategy = dashboard["strategies"].get(strategy_id) if strategy_id else None

                    if event_type == "STRATEGY_SIGNAL" and strategy is not None:
                        raw_status = str(details.get("status") or "CANDIDATE").strip().upper()
                        strategy["status"] = {
                            "CANDIDATE": "candidate",
                            "QUEUED": "queued",
                            "EXECUTED": "in_trade",
                            "BLOCKED": "blocked",
                        }.get(raw_status, raw_status.lower())
                        strategy["updated_at"] = iso_or_none(logged_at)
                        strategy["last_signal_time"] = iso_or_none(logged_at)
                        strategy["last_signal_side"] = details.get("side")
                        strategy["last_signal_price"] = safe_float(details.get("price"))
                        # tp_dist/sl_dist on BLOCKED signals are reporting
                        # placeholders (0.00/0.00) because gate filters fire
                        # before bracket resolution — see julie001.py blocked-
                        # signal logger and aetherflow_strategy resolution
                        # order. Preserve the last *real* bracket from a
                        # CANDIDATE/QUEUED/EXECUTED signal so the dashboard
                        # card shows the true designed brackets, not zeros.
                        incoming_tp = safe_float(details.get("tp_dist"))
                        incoming_sl = safe_float(details.get("sl_dist"))
                        is_blocked_zero_bracket = (
                            raw_status == "BLOCKED"
                            and (incoming_tp in (None, 0, 0.0))
                            and (incoming_sl in (None, 0, 0.0))
                        )
                        if not is_blocked_zero_bracket:
                            strategy["tp_dist"] = incoming_tp
                            strategy["sl_dist"] = incoming_sl
                        strategy["priority"] = details.get("priority")
                        strategy["last_reason"] = event_message
                        strategy["sub_strategy"] = details.get("sub_strategy")
                        strategy["combo_key"] = details.get("combo_key") or details.get("sub_strategy")
                        strategy["rule_id"] = details.get("rule_id")
                        if details.get("early_exit_enabled") is not None:
                            strategy["early_exit_enabled"] = str(details.get("early_exit_enabled")).lower() == "true"
                        if details.get("gate_prob") is not None:
                            strategy["gate_prob"] = safe_float(details.get("gate_prob"))
                        if details.get("gate_threshold") is not None:
                            strategy["gate_threshold"] = safe_float(details.get("gate_threshold"))
                        strategy["entry_mode"] = details.get("entry_mode") or strategy.get("entry_mode")
                        strategy["vol_regime"] = details.get("vol_regime") or strategy.get("vol_regime")
                        activity_message, blocked = format_strategy_signal_activity(event_message, details)
                        set_strategy_activity(
                            strategy,
                            logged_at,
                            event_type,
                            activity_message,
                            severity=level,
                            blocked=blocked,
                        )
                        record_event(dashboard, max_events, logged_at, event_type, activity_message, level, strategy_id, details)
                        continue

                    if event_type == "FILTER_CHECK" and strategy is not None:
                        continue

                    if event_type == "TRADE_PLACED" and strategy is not None:
                        order_id = details.get("order_id")
                        factor = trade_factor_index.get(str(order_id or "").strip(), {})
                        position = {
                            "strategy_id": strategy_id,
                            "strategy_label": STRATEGY_LABELS[strategy_id],
                            "side": details.get("side"),
                            "entry_price": safe_float(details.get("entry")) or safe_float(details.get("price")),
                            "tp_price": safe_float(details.get("tp")),
                            "sl_price": safe_float(details.get("sl")),
                            "size": factor.get("size"),
                            "order_id": order_id,
                            "opened_at": iso_or_none(logged_at),
                            "entry_mode": factor.get("entry_mode"),
                            "base_session": factor.get("base_session"),
                            "current_session": factor.get("current_session"),
                            "sub_strategy": factor.get("sub_strategy"),
                            "combo_key": factor.get("combo_key"),
                            "rule_id": factor.get("rule_id"),
                            "early_exit_enabled": factor.get("early_exit_enabled"),
                            "gate_prob": factor.get("gate_prob"),
                            "gate_threshold": factor.get("gate_threshold"),
                            "vol_regime": factor.get("vol_regime"),
                        }
                        dashboard["bot"]["current_position"] = position
                        strategy["status"] = "in_trade"
                        strategy["updated_at"] = iso_or_none(logged_at)
                        strategy["last_signal_side"] = details.get("side")
                        strategy["last_signal_price"] = position.get("entry_price")
                        strategy["sub_strategy"] = factor.get("sub_strategy")
                        strategy["combo_key"] = factor.get("combo_key")
                        strategy["rule_id"] = factor.get("rule_id")
                        strategy["early_exit_enabled"] = factor.get("early_exit_enabled")
                        strategy["gate_prob"] = factor.get("gate_prob")
                        strategy["gate_threshold"] = factor.get("gate_threshold")
                        strategy["entry_mode"] = factor.get("entry_mode")
                        strategy["base_session"] = factor.get("base_session")
                        strategy["current_session"] = factor.get("current_session")
                        strategy["vol_regime"] = factor.get("vol_regime")
                        activity_message = format_trade_placed_activity(details)
                        set_strategy_activity(strategy, logged_at, event_type, activity_message, severity=level)
                        record_event(dashboard, max_events, logged_at, event_type, activity_message, level, strategy_id, details)
                        continue

                    if event_type == "BREAKEVEN_ADJUST":
                        if strategy_id is None:
                            current_position = dashboard["bot"].get("current_position")
                            if isinstance(current_position, dict):
                                strategy_id = current_position.get("strategy_id")
                        strategy = dashboard["strategies"].get(strategy_id) if strategy_id else None
                        if strategy is not None:
                            set_strategy_activity(strategy, logged_at, event_type, event_message, severity=level)
                        record_event(dashboard, max_events, logged_at, event_type, event_message, level, strategy_id, details)
                        continue

                    if event_type == "SYSTEM_SENTIMENT_EVENT":
                        severity = level if level in {"warning", "error"} else "info"
                        if strategy_id is not None:
                            strategy = dashboard["strategies"].get(strategy_id)
                            if strategy is not None:
                                strategy["updated_at"] = iso_or_none(logged_at)
                                strategy["last_reason"] = event_message
                                if details.get("trigger_side"):
                                    strategy["status"] = "candidate"
                                    strategy["last_signal_time"] = iso_or_none(logged_at)
                                    strategy["last_signal_side"] = details.get("trigger_side")
                                    strategy["priority"] = "SENTIMENT"
                                set_strategy_activity(strategy, logged_at, event_type, event_message, severity=severity)
                            record_event(dashboard, max_events, logged_at, event_type, event_message, severity, strategy_id, details)
                        else:
                            record_event(dashboard, max_events, logged_at, event_type, event_message, severity, None, details)
                        continue

                    if event_type in {"TRADE_SIGNAL", "STRATEGY_EXEC", "SYSTEM"}:
                        severity = level if level in {"warning", "error"} else "info"
                        compact_message = (
                            format_system_activity(strategy_id, event_message, details)
                            if event_type == "SYSTEM" and strategy_id is not None
                            else event_message
                        )
                        if READY_RE.search(event_message):
                            dashboard["bot"]["status"] = "ready"
                            dashboard["bot"]["last_ready_time"] = iso_or_none(logged_at)
                        if strategy_id is not None:
                            strategy = dashboard["strategies"].get(strategy_id)
                            if strategy is not None:
                                set_strategy_activity(strategy, logged_at, event_type, compact_message, severity=severity)
                            record_event(dashboard, max_events, logged_at, event_type, compact_message, severity, strategy_id, details)
                        elif event_type == "SYSTEM" and READY_RE.search(event_message):
                            record_event(dashboard, max_events, logged_at, event_type, event_message, severity, None, details)
                        continue

                heartbeat_match = HEARTBEAT_RE.match(message)
                if heartbeat_match:
                    dashboard["bot"]["last_heartbeat_time"] = iso_or_none(logged_at)
                    dashboard["bot"]["session_connection_ok"] = "✅" in heartbeat_match.group("session_status")
                    dashboard["bot"]["price"] = safe_float(heartbeat_match.group("price"))
                    continue

                bar_match = BAR_RE.match(message)
                if bar_match:
                    bar_time = datetime.strptime(bar_match.group("bar_time"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY_TZ)
                    price = safe_float(bar_match.group("price"))
                    dashboard["bot"]["last_bar_time"] = iso_or_none(bar_time)
                    dashboard["bot"]["price"] = price
                    append_limited(
                        dashboard["bot"]["price_history"],
                        {"time": bar_time.isoformat(), "price": price},
                        max_price_points,
                    )
                    continue

                position_sync_match = POSITION_SYNC_RE.match(message)
                if position_sync_match:
                    dashboard["bot"]["last_position_sync_time"] = iso_or_none(logged_at)
                    status = (position_sync_match.group("status") or "").strip()
                    side = (position_sync_match.group("side") or "").strip().upper()
                    if status:
                        dashboard["bot"]["position_sync_status"] = status
                    elif side:
                        size = safe_float(position_sync_match.group("size"))
                        avg_price = safe_float(position_sync_match.group("avg_price"))
                        open_pnl = safe_float(position_sync_match.group("open_pnl"))
                        status = f"{side} {int(size) if size is not None else '?'}"
                        if avg_price is not None:
                            status = f"{status} @ {avg_price:.2f}"
                        dashboard["bot"]["position_sync_status"] = status
                        position = dashboard["bot"].get("current_position")
                        if not isinstance(position, dict):
                            position = {
                                "strategy_id": "unknown",
                                "strategy_label": "Open Position",
                            }
                            dashboard["bot"]["current_position"] = position
                        position["source"] = "projectx_api"
                        position["side"] = side
                        if size is not None:
                            position["size"] = int(size)
                        if avg_price is not None:
                            position["avg_price"] = avg_price
                            previous_entry = safe_float(position.get("entry_price"))
                            if previous_entry is not None and abs(previous_entry - avg_price) > 1e-9:
                                position["signal_entry_price"] = previous_entry
                            position["entry_price"] = avg_price
                        if open_pnl is not None:
                            position["open_pnl_dollars"] = round(open_pnl, 2)
                            if size not in (None, 0):
                                point_value = safe_float(position.get("point_value"))
                                if point_value is None or point_value <= 0.0:
                                    point_value = 5.0
                                position["open_pnl_points"] = round(open_pnl / (point_value * float(size)), 2)
                    if status.upper() == "FLAT":
                        current_position = dashboard["bot"].get("current_position")
                        if isinstance(current_position, dict):
                            strategy_id = current_position.get("strategy_id")
                            if strategy_id in dashboard["strategies"]:
                                dashboard["strategies"][strategy_id]["status"] = "idle"
                                dashboard["strategies"][strategy_id]["updated_at"] = iso_or_none(logged_at)
                        dashboard["bot"]["current_position"] = None
                    continue

                trade_closed_match = TRADE_CLOSED_RE.match(message)
                if trade_closed_match:
                    strategy_id = canonical_strategy_id(trade_closed_match.group("label"))
                    if strategy_id is None:
                        continue
                    close_details = parse_tail_details(trade_closed_match.group("tail"))
                    pnl_points = safe_float(trade_closed_match.group("pnl_points"))
                    pnl_dollars_gross = safe_float(trade_closed_match.group("pnl_dollars"))
                    current_position = dashboard["bot"].get("current_position")
                    inferred_size = None
                    if isinstance(current_position, dict) and current_position.get("strategy_id") == strategy_id:
                        inferred_size = int(max(1, safe_float(current_position.get("size")) or 1))
                    parsed_size = int(max(1, safe_float(close_details.get("size")) or 1)) if close_details.get("size") not in (None, "") else inferred_size
                    trade = {
                        "time": iso_or_none(logged_at),
                        "strategy_id": strategy_id,
                        "strategy_label": STRATEGY_LABELS[strategy_id],
                        "side": trade_closed_match.group("side"),
                        "size": parsed_size,
                        "entry_price": safe_float(trade_closed_match.group("entry")),
                        "exit_price": safe_float(trade_closed_match.group("exit")),
                        "pnl_points": pnl_points,
                        "pnl_dollars": pnl_dollars_gross,
                        "result": "win" if (pnl_dollars_gross or 0.0) > 0 else "loss" if (pnl_dollars_gross or 0.0) < 0 else "flat",
                        "order_id": close_details.get("order_id") or None,
                        "entry_order_id": close_details.get("entry_order_id") or None,
                    }
                    trade = enrich_trade_from_close_index(trade, trade_close_index)
                    trade = normalize_trade_realized_pnl(trade)
                    pnl_dollars = safe_float(trade.get("pnl_dollars"))
                    pnl_dollars_text = f"{pnl_dollars:+.2f}" if pnl_dollars is not None else "--"
                    pnl_points_text = f"{pnl_points:+.2f}" if pnl_points is not None else "--"
                    activity_message = f"Trade closed | ${pnl_dollars_text} | {pnl_points_text} pts"
                    upsert_trade(dashboard, trade, max_trades)
                    track_daily_realized_pnl(daily_tracker, trade)
                    apply_trade_summary(
                        dashboard,
                        trade,
                        updated_at=iso_or_none(logged_at),
                        touch_runtime_state=True,
                    )
                    strategy = dashboard["strategies"].get(strategy_id)
                    if strategy is not None:
                        set_strategy_activity(
                            strategy,
                            logged_at,
                            "TRADE_CLOSED",
                            activity_message,
                            severity="success" if (pnl_dollars or 0.0) > 0 else "warning",
                        )
                    record_event(
                        dashboard,
                        max_events,
                        logged_at,
                        "TRADE_CLOSED",
                        activity_message,
                        "success" if (pnl_dollars or 0.0) > 0 else "warning",
                        strategy_id,
                        {
                            "pnl_dollars": pnl_dollars,
                            "pnl_points": pnl_points,
                            "pnl_dollars_gross": trade.get("pnl_dollars_gross"),
                            "pnl_fee_dollars": trade.get("pnl_fee_dollars"),
                        },
                    )
                    continue

                manifold_abstain_match = MANIFOLD_ABSTAIN_RE.match(message)
                if manifold_abstain_match:
                    continue

                if "DynamicEngine3Strategy initialized" in message:
                    strategy = dashboard["strategies"]["dynamic_engine3"]
                    strategy["status"] = "ready"
                    strategy["updated_at"] = iso_or_none(logged_at)
                    set_strategy_activity(
                        strategy,
                        logged_at,
                        "SYSTEM",
                        format_system_activity("dynamic_engine3", message),
                        severity=level,
                    )
                    continue

                if "RegimeAdaptiveStrategy initialized" in message:
                    strategy = dashboard["strategies"].get("regime_adaptive")
                    if strategy is not None:
                        strategy["status"] = "ready"
                        strategy["updated_at"] = iso_or_none(logged_at)
                        set_strategy_activity(
                            strategy,
                            logged_at,
                            "SYSTEM",
                            format_system_activity("regime_adaptive", message),
                            severity=level,
                        )
                    continue

                if "MLPhysics(dist) active:" in message or "MLPhysics: legacy SessionManager models detached" in message:
                    strategy = dashboard["strategies"]["ml_physics"]
                    strategy["status"] = "ready"
                    strategy["updated_at"] = iso_or_none(logged_at)
                    set_strategy_activity(
                        strategy,
                        logged_at,
                        "SYSTEM",
                        format_system_activity("ml_physics", message),
                        severity=level,
                    )
                    continue

                if "ManifoldStrategy model loaded" in message or "ManifoldStrategy thresholds loaded" in message:
                    strategy = dashboard["strategies"].get("manifold")
                    if strategy is not None:
                        strategy["status"] = "ready"
                        strategy["updated_at"] = iso_or_none(logged_at)
                        set_strategy_activity(strategy, logged_at, "SYSTEM", message, severity=level)
                    continue

                if (
                    "AetherFlowStrategy model loaded" in message
                    or "AetherFlowStrategy thresholds loaded" in message
                    or "AetherFlowStrategy initialized for live execution" in message
                ):
                    strategy = dashboard["strategies"].get("aetherflow")
                    if strategy is not None:
                        strategy["status"] = "ready"
                        strategy["updated_at"] = iso_or_none(logged_at)
                        set_strategy_activity(
                            strategy,
                            logged_at,
                            "SYSTEM",
                            format_system_activity("aetherflow", message),
                            severity=level,
                        )
                    continue

                de3_breakeven_match = DE3_BREAKEVEN_ARMED_RE.match(message)
                if de3_breakeven_match:
                    strategy_id = canonical_strategy_id(de3_breakeven_match.group("label"))
                    strategy = dashboard["strategies"].get(strategy_id) if strategy_id else None
                    if strategy is not None:
                        activity_message = (
                            f"Stop moved to breakeven | {de3_breakeven_match.group('side')} -> "
                            f"{float(de3_breakeven_match.group('new_sl')):.2f}"
                        )
                        set_strategy_activity(strategy, logged_at, "BREAKEVEN_ADJUST", activity_message, severity=level)
                        record_event(
                            dashboard,
                            max_events,
                            logged_at,
                            "BREAKEVEN_ADJUST",
                            activity_message,
                            level,
                            strategy_id,
                            {"new_sl": safe_float(de3_breakeven_match.group("new_sl"))},
                        )
                    continue

                if READY_RE.search(message):
                    dashboard["bot"]["last_ready_time"] = iso_or_none(logged_at)
                    record_event(dashboard, max_events, logged_at, "SYSTEM", message, level)
    else:
        dashboard["bot"]["warnings"].append(f"Log file not found: {log_path}")

    apply_bot_state_snapshot(
        dashboard,
        persisted_state,
        max_trades=max_trades,
        trade_close_index=trade_close_index,
        daily_tracker=daily_tracker,
    )
    kalshi_snapshot: Dict[str, Any] = {}
    if isinstance(kalshi_snapshot_path, Path) and kalshi_snapshot_path.exists():
        try:
            parsed = json.loads(kalshi_snapshot_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                kalshi_snapshot = parsed
        except (OSError, json.JSONDecodeError):
            kalshi_snapshot = {}
    update_state(
        dashboard,
        kalshi_snapshot=kalshi_snapshot,
        kalshi_provider=kalshi_provider,
    )
    update_state(dashboard, kalshi_snapshot=kalshi_snapshot)
    dashboard["bot"]["risk"]["daily_pnl"] = compute_session_realized_pnl(
        dashboard["trades"],
        trading_day_start_dt,
    )
    return finalize_dashboard(dashboard)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    last_error: Optional[BaseException] = None
    for _ in range(8):
        try:
            tmp_path.write_text(text, encoding="utf-8")
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.15)
        except OSError as exc:
            last_error = exc
            break
    try:
        path.write_text(text, encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return
    except OSError as exc:
        if last_error is not None:
            raise last_error from exc
        raise


def path_signature(paths: Iterable[Path]) -> tuple[tuple[str, Optional[int], Optional[int]], ...]:
    signature = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


def run_once(args: argparse.Namespace) -> None:
    dashboard = build_dashboard_state(
        log_path=args.log_path,
        state_path=args.state_path,
        trade_factors_path=args.trade_factors_path,
        max_events=args.max_events,
        max_trades=args.max_trades,
        max_price_points=args.max_price_points,
        kalshi_snapshot_path=args.kalshi_snapshot_path,
    )
    write_json_atomic(args.output, dashboard)
    if args.output.parent.name == "public":
        dist_output = args.output.parent.parent / "dist" / args.output.name
        if dist_output.parent.exists():
            write_json_atomic(dist_output, dashboard)
    print(f"Wrote filterless dashboard state to {args.output}")


def run_follow(args: argparse.Namespace) -> None:
    watched = [args.log_path, args.state_path, args.trade_factors_path, args.kalshi_snapshot_path]
    last_seen: Optional[tuple[tuple[str, Optional[int], Optional[int]], ...]] = None
    while True:
        current = path_signature(watched)
        if current != last_seen:
            run_once(args)
            last_seen = current
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build live dashboard JSON for filterless strategies.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--trade-factors-path", type=Path, default=DEFAULT_TRADE_FACTORS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--kalshi-snapshot-path", type=Path, default=DEFAULT_KALSHI_SNAPSHOT_PATH)
    parser.add_argument("--max-events", type=int, default=400)
    parser.add_argument("--max-trades", type=int, default=500)
    parser.add_argument("--max-price-points", type=int, default=240)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.follow:
        bridge_lock = acquire_singleton_lock(BRIDGE_LOCK_PATH, name="filterless_dashboard_bridge")
        if bridge_lock is None:
            existing = ""
            try:
                existing = BRIDGE_LOCK_PATH.read_text(encoding="utf-8").strip()
            except OSError:
                existing = ""
            print(
                "Another filterless dashboard bridge instance is already running. "
                f"Lock: {BRIDGE_LOCK_PATH}"
            )
            if existing:
                print(existing)
            raise SystemExit(0)
        run_follow(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()

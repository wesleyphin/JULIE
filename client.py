"""
ProjectX Gateway API Client

Client for interacting with the ProjectX Gateway API for live futures trading.
Handles authentication, market data retrieval, order placement, and position management.
"""
import asyncio
import datetime
import inspect
import logging
import math
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:
    import websockets
except Exception:
    websockets = None

from config import CONFIG, refresh_target_symbol, determine_current_contract_symbol
from event_logger import event_logger

try:
    from signalrcore_async.hub_connection_builder import HubConnectionBuilder
except Exception:
    HubConnectionBuilder = None

try:
    from signalrcore_async.hub.base_hub_connection import WebSocketsConnection
except Exception:
    WebSocketsConnection = None

try:
    from signalrcore_async.hub.reconnection import ConnectionStateChecker
except Exception:
    ConnectionStateChecker = None


def _patch_websockets_connect_compat() -> None:
    """
    Bridge signalrcore_async's older `extra_headers=` call into websockets>=15,
    which renamed the keyword to `additional_headers=`.
    """
    if websockets is None:
        return
    connect = getattr(websockets, "connect", None)
    if connect is None or getattr(connect, "_projectx_extra_headers_compat", False):
        return
    try:
        signature = inspect.signature(connect)
    except Exception:
        return
    if "extra_headers" in signature.parameters or "additional_headers" not in signature.parameters:
        return

    def _compat_connect(*args, extra_headers=None, **kwargs):
        if extra_headers is not None and "additional_headers" not in kwargs:
            kwargs["additional_headers"] = extra_headers
        return connect(*args, **kwargs)

    _compat_connect._projectx_extra_headers_compat = True
    websockets.connect = _compat_connect
    logging.info("Applied websockets.connect compatibility shim for ProjectX user stream")


_patch_websockets_connect_compat()


def _patch_signalrcore_async_loop_compat() -> None:
    """
    signalrcore_async schedules websocket sends from background threads.

    On Python 3.13 / websockets>=15, `asyncio.create_task()` from those threads
    raises `RuntimeError: no running event loop`. Route those sends back onto the
    live websocket loop instead so keep-alive pings don't crash the stream thread.
    """
    if WebSocketsConnection is None:
        return
    if getattr(WebSocketsConnection, "_projectx_loop_compat", False):
        return

    original_run = WebSocketsConnection.run

    async def _compat_run(self, *args, **kwargs):
        self._projectx_event_loop = asyncio.get_running_loop()
        return await original_run(self, *args, **kwargs)

    def _compat_send(self, data):
        ws = getattr(self, "_ws", None)
        if ws is None:
            logging.debug("ProjectX user stream send skipped: websocket not initialized")
            return None

        target_loop = getattr(self, "_projectx_event_loop", None)
        if target_loop is None:
            try:
                target_loop = asyncio.get_running_loop()
            except RuntimeError:
                logging.warning("ProjectX user stream send skipped: no event loop is available")
                return None

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is target_loop:
            return running_loop.create_task(ws.send(data))

        try:
            return asyncio.run_coroutine_threadsafe(ws.send(data), target_loop)
        except RuntimeError as exc:
            logging.warning("ProjectX user stream send skipped on closed loop: %s", exc)
            return None

    WebSocketsConnection.run = _compat_run
    WebSocketsConnection.send = _compat_send
    WebSocketsConnection._projectx_loop_compat = True
    logging.info("Applied signalrcore_async event-loop compatibility shim for ProjectX user stream")

    if ConnectionStateChecker is not None and not getattr(ConnectionStateChecker, "_projectx_loop_compat", False):
        original_checker_run = ConnectionStateChecker.run

        def _compat_checker_run(self):
            try:
                return original_checker_run(self)
            except Exception as exc:
                logging.warning("ProjectX user stream keep-alive thread stopped after error: %s", exc)
                self.running = False

        ConnectionStateChecker.run = _compat_checker_run
        ConnectionStateChecker._projectx_loop_compat = True


_patch_signalrcore_async_loop_compat()


class ProjectXClient:
    """
    Client for ProjectX Gateway API (Live)
    Based on API Documentation:
    - REST API: https://gateway-api.s2f.projectx.com
    - Auth: JWT tokens via /api/Auth/loginKey (valid 24 hours)
    - Rate Limits:
        - /api/History/retrieveBars: 50 requests / 30 seconds
        - All other endpoints: 200 requests / 60 seconds
    """
    # Class-level (shared) rate limiting - all instances share these
    _shared_bar_timestamps = []
    _shared_general_timestamps = []
    _shared_last_bar_fetch = None
    _shared_lock = None  # Will be initialized on first use

    # Class-level rate limit config
    SHARED_BAR_RATE_LIMIT = 50
    SHARED_BAR_RATE_WINDOW = 30
    SHARED_GENERAL_RATE_LIMIT = 200
    SHARED_GENERAL_RATE_WINDOW = 60
    SHARED_MIN_FETCH_INTERVAL = 0.8  # Minimum 800ms between any bar fetches across all instances

    @classmethod
    def _get_lock(cls):
        """Thread-safe lock initialization"""
        if cls._shared_lock is None:
            import threading
            cls._shared_lock = threading.Lock()
        return cls._shared_lock

    @classmethod
    def _shared_check_bar_rate_limit(cls) -> bool:
        """Check shared rate limit for bar fetches across all client instances"""
        with cls._get_lock():
            now = time.time()
            # Clean old timestamps
            cls._shared_bar_timestamps = [
                t for t in cls._shared_bar_timestamps
                if now - t < cls.SHARED_BAR_RATE_WINDOW
            ]
            # Check if we're approaching limit (leave buffer of 10)
            if len(cls._shared_bar_timestamps) >= cls.SHARED_BAR_RATE_LIMIT - 10:
                logging.warning(f"Shared bar rate limit ({len(cls._shared_bar_timestamps)}/{cls.SHARED_BAR_RATE_LIMIT}). Using cache.")
                return False
            # Enforce minimum interval between ANY bar fetches
            if cls._shared_last_bar_fetch is not None:
                elapsed = now - cls._shared_last_bar_fetch
                if elapsed < cls.SHARED_MIN_FETCH_INTERVAL:
                    wait_time = cls.SHARED_MIN_FETCH_INTERVAL - elapsed
                    time.sleep(wait_time)
            return True

    @classmethod
    def _shared_track_bar_fetch(cls):
        """Track a bar fetch request in shared rate limiter"""
        with cls._get_lock():
            now = time.time()
            cls._shared_bar_timestamps.append(now)
            cls._shared_last_bar_fetch = now

    def __init__(self, contract_root: Optional[str] = None, target_symbol: Optional[str] = None):
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None
        self.base_url = CONFIG['REST_BASE_URL']
        self.et = ZoneInfo('America/New_York')

        # Contract configuration (allows per-instance override)
        self.contract_root = contract_root or CONFIG.get('CONTRACT_ROOT', 'MES')
        self.target_symbol = target_symbol or CONFIG.get('TARGET_SYMBOL')

        # Account and contract info (fetched after login)
        self.account_id = CONFIG.get('ACCOUNT_ID')
        self.contract_id = CONFIG.get('CONTRACT_ID')

        # Rate limiting for /History/retrieveBars: 50 requests / 30 seconds
        self.bar_fetch_timestamps = []
        self.last_bar_fetch_time = None
        self.cached_df = pd.DataFrame()
        self.last_bar_timestamp = None

        # Rate limit config
        self.BAR_RATE_LIMIT = 50
        self.BAR_RATE_WINDOW = 30
        self.MIN_FETCH_INTERVAL = 0.1

        # General rate limiting: 200 requests / 60 seconds
        self.general_request_timestamps = []
        self.GENERAL_RATE_LIMIT = 200
        self.GENERAL_RATE_WINDOW = 60

        # Shadow Position State (avoids unnecessary API calls)
        self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}

        # Exit order tracking (avoids search_orders calls)
        self._active_stop_order_id = None
        self._active_target_order_id = None
        self._order_cache = {}
        self._order_cache_ts = 0.0
        self._last_order_details = None
        self._last_close_order_details = None
        self._account_cache = None
        self._account_cache_ts = 0.0

        # ProjectX user-hub state
        self._user_hub_url = str(CONFIG.get("RTC_USER_HUB", "") or "").strip()
        self._user_stream = None
        self._user_stream_connected = False
        self._user_stream_last_error = None
        self._user_stream_loop = None
        self._user_stream_starting = False
        self._user_stream_subscribed = False
        self._user_stream_position = None
        self._user_stream_position_ts = None
        self._user_stream_account = None
        self._user_stream_account_ts = None
        self._user_stream_trades: List[Dict] = []
        self._user_stream_trade_cache = max(
            32,
            self._coerce_int(CONFIG.get("PROJECTX_USER_STREAM_TRADE_CACHE", 256), 256) or 256,
        )
        self._user_stream_position_max_age = max(
            1.0,
            self._coerce_float(CONFIG.get("PROJECTX_USER_STREAM_MAX_POSITION_AGE_SEC", 15.0), 15.0) or 15.0,
        )
        self._user_stream_account_max_age = max(
            5.0,
            self._coerce_float(CONFIG.get("PROJECTX_USER_STREAM_MAX_ACCOUNT_AGE_SEC", 300.0), 300.0) or 300.0,
        )
        self._aiohttp_fallback_warned = False
        self._auth_recovery_lock = threading.Lock()
        self._auth_recovery_in_progress = False
        self._auth_recovery_last_attempt_ts = 0.0
        self._auth_recovery_cooldown_sec = max(
            1.0,
            self._coerce_float(CONFIG.get("PROJECTX_AUTH_RECOVERY_COOLDOWN_SEC", 5.0), 5.0) or 5.0,
        )
        self._auth_failure_reason = None
        self._session_conflict_policy = str(
            CONFIG.get("PROJECTX_SESSION_CONFLICT_POLICY", "yield") or "yield"
        ).strip().lower()
        self._external_session_retry_sec = max(
            0.0,
            self._coerce_float(CONFIG.get("PROJECTX_EXTERNAL_SESSION_RETRY_SEC", 0.0), 0.0) or 0.0,
        )
        self._yielding_to_external_session = False
        self._yielding_to_external_session_since = 0.0
        self._yielding_to_external_session_reason = ""
        self._order_submission_lockout_until_ts = 0.0
        self._order_submission_lockout_reason = ""
        self._order_submission_lockout_log_ts = 0.0
        self._order_submission_lockout_cooldown_sec = max(
            0.0,
            self._coerce_float(
                CONFIG.get("PROJECTX_ORDER_LOCKOUT_COOLDOWN_SEC", 60.0),
                60.0,
            )
            or 60.0,
        )
        self.session.hooks.setdefault("response", []).append(self._requests_response_auth_hook)

    def _warn_async_http_fallback_once(self) -> None:
        if self._aiohttp_fallback_warned:
            return
        self._aiohttp_fallback_warned = True
        logging.warning("aiohttp not installed; falling back to sync REST calls for async background tasks")

    def _error_indicates_order_lockout(self, message: Optional[str]) -> bool:
        text = str(message or "").strip().lower()
        if not text:
            return False
        return ("locked out" in text) or ("lockout expires" in text)

    def _clear_order_submission_lockout(self) -> None:
        self._order_submission_lockout_until_ts = 0.0
        self._order_submission_lockout_reason = ""
        self._order_submission_lockout_log_ts = 0.0

    def _arm_order_submission_lockout(self, reason: Optional[str]) -> None:
        cooldown_sec = float(self._order_submission_lockout_cooldown_sec)
        if cooldown_sec <= 0.0:
            return
        now = time.time()
        reason_text = str(reason or "").strip() or "broker lockout"
        new_until_ts = now + cooldown_sec
        should_log = new_until_ts > self._order_submission_lockout_until_ts + 1e-9
        self._order_submission_lockout_until_ts = max(
            self._order_submission_lockout_until_ts,
            new_until_ts,
        )
        self._order_submission_lockout_reason = reason_text
        if should_log:
            logging.warning(
                "Broker order lockout detected; pausing new order submissions for %.0fs: %s",
                cooldown_sec,
                reason_text,
            )
            self._order_submission_lockout_log_ts = now

    def _order_submission_lockout_active(self) -> bool:
        now = time.time()
        until_ts = float(self._order_submission_lockout_until_ts or 0.0)
        if until_ts <= now:
            if until_ts > 0.0:
                self._clear_order_submission_lockout()
            return False
        if (now - float(self._order_submission_lockout_log_ts or 0.0)) >= 30.0:
            remaining_sec = max(0.0, until_ts - now)
            logging.warning(
                "Skipping order submission for another %.0fs due to broker lockout: %s",
                remaining_sec,
                self._order_submission_lockout_reason or "broker lockout",
            )
            self._order_submission_lockout_log_ts = now
        return True

    def _check_general_rate_limit(self) -> bool:
        """Check if we're within general rate limits"""
        now = time.time()
        self.general_request_timestamps = [
            t for t in self.general_request_timestamps
            if now - t < self.GENERAL_RATE_WINDOW
        ]
        if len(self.general_request_timestamps) >= self.GENERAL_RATE_LIMIT - 10:
            logging.warning(f"Approaching general rate limit ({len(self.general_request_timestamps)}/{self.GENERAL_RATE_LIMIT})")
            return False
        return True

    def _track_general_request(self):
        """Track a general API request for rate limiting"""
        self.general_request_timestamps.append(time.time())

    def _stale_position(self) -> Dict:
        """Return last known position marked as stale."""
        pos = self._local_position.copy()
        pos["stale"] = True
        return pos

    def _utc_isoformat(self, value: Optional[datetime.datetime]) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.et)
        value = value.astimezone(datetime.timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _coerce_float(value, default: Optional[float] = None) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return float(parsed)

    @staticmethod
    def _coerce_int(value, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _extract_execution_price(self, payload: Dict, fallback: Optional[float] = None) -> Optional[float]:
        for key in ("averagePrice", "avgPrice", "fillPrice", "price"):
            parsed = self._coerce_float(payload.get(key), None)
            if parsed is not None:
                return parsed
        return fallback

    def _normalize_position(self, raw_position: Optional[Dict]) -> Dict:
        if not isinstance(raw_position, dict):
            return {'side': None, 'size': 0, 'avg_price': 0.0}
        size_raw = raw_position.get('size', raw_position.get('positionSize', 0))
        avg_raw = raw_position.get('averagePrice', raw_position.get('avgPrice', 0.0))
        size = self._coerce_int(size_raw, 0) or 0
        avg_price = self._coerce_float(avg_raw, 0.0) or 0.0
        position_type = self._coerce_int(
            raw_position.get('type', raw_position.get('positionType')),
            None,
        )
        if size > 0:
            side = 'SHORT' if position_type == 2 else 'LONG'
        elif size < 0:
            side = 'SHORT'
        else:
            side = None
        position = {
            'side': side,
            'size': abs(size),
            'avg_price': avg_price,
        }
        open_pnl = None
        for key in (
            'openPnl',
            'openPNL',
            'profitAndLoss',
            'unrealizedProfitAndLoss',
            'unrealizedPnl',
            'unrealizedPnL',
            'openProfitAndLoss',
            'pnl',
        ):
            open_pnl = self._coerce_float(raw_position.get(key), None)
            if open_pnl is not None:
                break
        if open_pnl is not None:
            position['open_pnl'] = float(open_pnl)
        position['raw'] = raw_position
        return position

    def _filter_positions_for_contract(self, positions: List[Dict]) -> List[Dict]:
        if self.contract_id is None:
            return []
        return [
            pos for pos in positions
            if isinstance(pos, dict) and pos.get('contractId') == self.contract_id
        ]

    def _extract_trade_rows(self, payload) -> List[Dict]:
        if isinstance(payload, dict):
            rows = payload.get('trades')
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
            rows = payload.get('data')
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    @staticmethod
    def _signalr_payload(arguments) -> Optional[Any]:
        if isinstance(arguments, list):
            if len(arguments) == 0:
                return None
            if len(arguments) == 1:
                return arguments[0]
        return arguments

    @staticmethod
    def _normalize_order_row(row: Optional[Dict]) -> Dict:
        if not isinstance(row, dict):
            return {}
        normalized = dict(row)
        if normalized.get("orderId") is None and normalized.get("id") is not None:
            normalized["orderId"] = normalized.get("id")
        return normalized

    @staticmethod
    def _order_has_open_status(row: Optional[Dict]) -> bool:
        if not isinstance(row, dict):
            return False
        status = row.get("status")
        status_int = ProjectXClient._coerce_int(status, None)
        if status_int is not None:
            return status_int in {1, 6}
        status_text = str(status or "").strip().lower()
        return status_text in {"working", "pending", "accepted", "active", "open"}

    def _normalize_account_row(self, row: Optional[Dict]) -> Dict:
        if not isinstance(row, dict):
            return {}
        normalized = dict(row)
        normalized["id"] = self._coerce_int(normalized.get("id"), None)
        balance = self._coerce_float(normalized.get("balance"), None)
        if balance is not None:
            normalized["balance"] = float(balance)
        return normalized

    @staticmethod
    def _trade_row_key(row: Optional[Dict]) -> Tuple:
        if not isinstance(row, dict):
            return tuple()
        return (
            row.get("id"),
            row.get("orderId"),
            row.get("creationTimestamp"),
            row.get("price"),
            row.get("size"),
            row.get("contractId"),
        )

    def _is_recent_timestamp(
        self,
        value: Optional[datetime.datetime],
        max_age_sec: float,
    ) -> bool:
        if value is None:
            return False
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        return (now_utc - value).total_seconds() <= float(max_age_sec)

    def _get_stream_position(self) -> Optional[Dict]:
        if not isinstance(self._user_stream_position, dict):
            return None
        if not self._is_recent_timestamp(self._user_stream_position_ts, self._user_stream_position_max_age):
            return None
        return dict(self._user_stream_position)

    def _get_stream_account(self) -> Optional[Dict]:
        if not isinstance(self._user_stream_account, dict):
            return None
        if not self._is_recent_timestamp(self._user_stream_account_ts, self._user_stream_account_max_age):
            return None
        return dict(self._user_stream_account)

    def _get_stream_trades(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> List[Dict]:
        if not self._user_stream_trades:
            return []
        start_utc = None
        end_utc = None
        if isinstance(start_time, datetime.datetime):
            start_utc = start_time if start_time.tzinfo else start_time.replace(tzinfo=self.et)
            start_utc = start_utc.astimezone(datetime.timezone.utc)
        if isinstance(end_time, datetime.datetime):
            end_utc = end_time if end_time.tzinfo else end_time.replace(tzinfo=self.et)
            end_utc = end_utc.astimezone(datetime.timezone.utc)

        rows: List[Dict] = []
        for row in self._user_stream_trades:
            if row.get("contractId") != self.contract_id:
                continue
            row_ts = self._parse_trade_timestamp(row)
            if row_ts is None:
                continue
            if start_utc is not None and row_ts < start_utc:
                continue
            if end_utc is not None and row_ts > end_utc:
                continue
            rows.append(dict(row))
        rows.sort(
            key=lambda row: self._parse_trade_timestamp(row) or datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            )
        )
        return rows

    def _merge_trade_rows(self, primary_rows: List[Dict], secondary_rows: List[Dict]) -> List[Dict]:
        merged: Dict[Tuple, Dict] = {}
        for row in list(primary_rows or []) + list(secondary_rows or []):
            if not isinstance(row, dict):
                continue
            merged[self._trade_row_key(row)] = row
        rows = list(merged.values())
        rows.sort(
            key=lambda row: self._parse_trade_timestamp(row) or datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            )
        )
        return rows

    def _parse_trade_timestamp(self, row: Dict) -> Optional[datetime.datetime]:
        if not isinstance(row, dict):
            return None
        for key in (
            "creationTimestamp",
            "timestamp",
            "createdAt",
            "updatedAt",
            "tradeTime",
            "executionTime",
        ):
            raw = row.get(key)
            if not raw:
                continue
            try:
                text = str(raw).strip()
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                parsed = datetime.datetime.fromisoformat(text)
            except Exception:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            return parsed.astimezone(datetime.timezone.utc)
        return None

    def _record_user_position(self, row: Optional[Dict]) -> None:
        if not isinstance(row, dict):
            return
        if row.get("contractId") != self.contract_id:
            return
        position = self._normalize_position(row)
        self._user_stream_position = position
        self._user_stream_position_ts = datetime.datetime.now(datetime.timezone.utc)
        self._local_position = position.copy()

    def _record_user_account(self, row: Optional[Dict]) -> None:
        normalized = self._normalize_account_row(row)
        if not normalized:
            return
        account_id = self._coerce_int(self.account_id, None)
        if account_id is not None and normalized.get("id") not in (None, account_id):
            return
        self._user_stream_account = normalized
        self._user_stream_account_ts = datetime.datetime.now(datetime.timezone.utc)
        self._account_cache = dict(normalized)
        self._account_cache_ts = time.time()

    def _record_user_trade(self, row: Optional[Dict]) -> None:
        if not isinstance(row, dict):
            return
        if row.get("contractId") != self.contract_id:
            return
        key = self._trade_row_key(row)
        deduped = [
            existing for existing in self._user_stream_trades
            if self._trade_row_key(existing) != key
        ]
        deduped.append(dict(row))
        deduped.sort(
            key=lambda item: self._parse_trade_timestamp(item) or datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            )
        )
        if len(deduped) > self._user_stream_trade_cache:
            deduped = deduped[-self._user_stream_trade_cache:]
        self._user_stream_trades = deduped

    def _record_user_order(self, row: Optional[Dict]) -> None:
        normalized = self._normalize_order_row(row)
        if not normalized:
            return

        account_id = self._coerce_int(self.account_id, None)
        row_account_id = self._coerce_int(normalized.get("accountId"), None)
        if account_id is not None and row_account_id not in (None, account_id):
            return

        order_id = self._coerce_int(normalized.get("orderId"), None)
        if order_id is None:
            return

        contract_id = normalized.get("contractId")
        is_target_contract = contract_id == self.contract_id
        is_open = self._order_has_open_status(normalized)

        if is_target_contract and is_open:
            self._order_cache[order_id] = dict(normalized)
            self._order_cache_ts = time.time()
            return

        if order_id in self._order_cache:
            self._order_cache.pop(order_id, None)
            self._order_cache_ts = time.time()
        if self._active_stop_order_id == order_id:
            self._active_stop_order_id = None
        if self._active_target_order_id == order_id:
            self._active_target_order_id = None

    def _build_user_stream_connection_url(self) -> str:
        raw_url = str(self._user_hub_url or "").strip()
        if not raw_url:
            return ""
        parsed = urlparse(raw_url)
        scheme = parsed.scheme.lower()
        if scheme == "https":
            target_scheme = "wss"
        elif scheme == "http":
            target_scheme = "ws"
        else:
            target_scheme = scheme or "wss"

        query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if self.token:
            query_items["access_token"] = self.token
        return urlunparse(
            parsed._replace(
                scheme=target_scheme,
                query=urlencode(query_items),
            )
        )

    def _requests_response_auth_hook(self, response, *args, **kwargs):
        if response is None or getattr(response, "status_code", None) != 401:
            return response
        request = getattr(response, "request", None)
        request_url = str(getattr(request, "url", "") or "")
        if "/api/Auth/loginKey" in request_url:
            return response
        request_method = str(getattr(request, "method", "HTTP") or "HTTP").upper()
        context = f"{request_method} {request_url}".strip()
        self._handle_unauthorized_response(context)
        return response

    def _invalidate_auth_state(self, reason: str = "") -> None:
        self.token = None
        self.token_expiry = None
        self.session.headers.pop("Authorization", None)
        self._user_stream_connected = False
        self._user_stream_subscribed = False
        if reason:
            self._auth_failure_reason = str(reason)
            self._user_stream_last_error = str(reason)

    def _is_external_session_conflict(self, reason: str = "") -> bool:
        text = str(reason or "").strip().lower()
        if not text:
            return False
        return (
            "multiple sessions" in text
            or "one active device session" in text
            or "active device session" in text
        )

    def _clear_external_session_yield(self) -> None:
        self._yielding_to_external_session = False
        self._yielding_to_external_session_since = 0.0
        self._yielding_to_external_session_reason = ""

    def _external_session_retry_allowed(self) -> bool:
        if not self._yielding_to_external_session:
            return True
        if self._external_session_retry_sec <= 0.0:
            return False
        return (time.time() - self._yielding_to_external_session_since) >= self._external_session_retry_sec

    def _auth_temporarily_unavailable(self) -> bool:
        return self._yielding_to_external_session and not self._external_session_retry_allowed()

    def _yield_rest_auth_unavailable(self, reason: str = "") -> bool:
        if bool(CONFIG.get("PROJECTX_USER_STREAM_ENABLED", True)):
            return False
        if self._session_conflict_policy != "yield":
            return False
        reason_text = str(reason or "REST unauthorized response")
        self._yielding_to_external_session = True
        self._yielding_to_external_session_since = time.time()
        self._yielding_to_external_session_reason = reason_text
        self._invalidate_auth_state(reason_text)
        logging.warning(
            "ProjectX REST auth became unavailable (%s). Staying on cached/local state without re-authenticating.",
            reason_text,
        )
        return True

    def _yield_to_external_session(self, reason: str = "") -> bool:
        if self._session_conflict_policy != "yield":
            return False
        if not self._is_external_session_conflict(reason):
            return False
        reason_text = str(reason or "external TopstepX session takeover")
        self._yielding_to_external_session = True
        self._yielding_to_external_session_since = time.time()
        self._yielding_to_external_session_reason = reason_text
        self._invalidate_auth_state(reason_text)
        logging.warning(
            "ProjectX session conflict detected (%s). Yielding API session to the external TopstepX login.",
            reason_text,
        )
        scheduled = self._schedule_user_stream_task(
            lambda: self.stop_user_stream(),
            description="stop after external session conflict",
        )
        if not scheduled:
            self._clear_user_stream_runtime()
        return True

    async def _restart_user_stream_after_reauth(self, reason: str = "") -> None:
        if not bool(CONFIG.get("PROJECTX_USER_STREAM_ENABLED", True)):
            return
        if self.account_id is None:
            return
        try:
            await self.stop_user_stream()
        except Exception as exc:
            logging.warning("ProjectX user stream stop during auth recovery failed: %s", exc)
        started = await self.start_user_stream()
        if started:
            logging.info("📡 ProjectX user stream restarted after auth recovery")
        else:
            logging.warning("ProjectX user stream restart after auth recovery did not succeed")

    def recover_auth_sync(self, reason: str = "", *, restart_user_stream: bool = True) -> bool:
        reason_text = str(reason or "unauthorized response")
        if self._yield_to_external_session(reason_text):
            return False
        if self._yield_rest_auth_unavailable(reason_text):
            return False
        if self._auth_temporarily_unavailable():
            return False
        if self._yielding_to_external_session and self._external_session_retry_allowed():
            logging.warning("ProjectX external-session cooldown elapsed; attempting API re-authentication again.")
            self._clear_external_session_yield()
        now = time.time()
        with self._auth_recovery_lock:
            if self._auth_recovery_in_progress:
                return self.token is not None
            if (now - self._auth_recovery_last_attempt_ts) < self._auth_recovery_cooldown_sec:
                return self.token is not None
            self._auth_recovery_in_progress = True
            self._auth_recovery_last_attempt_ts = now

        try:
            had_user_stream = (
                bool(CONFIG.get("PROJECTX_USER_STREAM_ENABLED", True))
                and self.account_id is not None
                and (
                    self._user_stream is not None
                    or self._user_stream_connected
                    or self._user_stream_subscribed
                    or self._user_stream_loop is not None
                )
            )
            logging.warning(
                "ProjectX auth invalidated (%s). Re-authenticating via loginKey.",
                reason_text,
            )
            self._invalidate_auth_state(reason_text)
            self.login()
            if restart_user_stream and had_user_stream and self._user_stream_loop is not None:
                scheduled = self._schedule_user_stream_task(
                    lambda: self._restart_user_stream_after_reauth(reason_text),
                    description="restart after auth recovery",
                )
                if not scheduled:
                    logging.warning("ProjectX user stream restart scheduling failed after auth recovery")
            return self.token is not None
        except Exception as exc:
            logging.error("ProjectX auth recovery failed after %s: %s", reason_text, exc)
            return False
        finally:
            with self._auth_recovery_lock:
                self._auth_recovery_in_progress = False

    def _handle_unauthorized_response(self, context: str, *, restart_user_stream: bool = True) -> bool:
        return self.recover_auth_sync(context, restart_user_stream=restart_user_stream)

    async def _handle_unauthorized_response_async(
        self,
        context: str,
        *,
        restart_user_stream: bool = True,
    ) -> bool:
        return await asyncio.to_thread(
            self.recover_auth_sync,
            context,
            restart_user_stream=restart_user_stream,
        )

    def _clear_user_stream_runtime(self) -> None:
        self._user_stream = None
        self._user_stream_connected = False
        self._user_stream_starting = False
        self._user_stream_subscribed = False
        self._user_stream_loop = None

    def _schedule_user_stream_task(self, coroutine_factory, *, description: str) -> bool:
        target_loop = self._user_stream_loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        try:
            if running_loop is not None and (target_loop is None or running_loop is target_loop):
                task = running_loop.create_task(coroutine_factory())

                def _log_task_exception(done_task: asyncio.Task) -> None:
                    try:
                        done_task.result()
                    except Exception as exc:
                        self._user_stream_last_error = str(exc)
                        logging.warning("ProjectX user stream %s failed: %s", description, exc)

                task.add_done_callback(_log_task_exception)
                return True

            if target_loop is not None:
                future = asyncio.run_coroutine_threadsafe(coroutine_factory(), target_loop)

                def _log_future_exception(done_future) -> None:
                    try:
                        done_future.result()
                    except Exception as exc:
                        self._user_stream_last_error = str(exc)
                        logging.warning("ProjectX user stream %s failed: %s", description, exc)

                future.add_done_callback(_log_future_exception)
                return True
        except Exception as exc:
            self._user_stream_last_error = str(exc)
            logging.warning("ProjectX user stream %s scheduling failed: %s", description, exc)
            return False

        logging.warning("ProjectX user stream %s scheduling failed: no event loop available", description)
        return False

    def _on_user_stream_open(self) -> None:
        self._user_stream_connected = True
        self._user_stream_last_error = None
        logging.info("✅ ProjectX user stream connected")
        if not self._user_stream_starting:
            self._schedule_user_stream_task(
                self._subscribe_user_stream,
                description="resubscribe",
            )

    def _on_user_stream_close(self) -> None:
        self._user_stream_connected = False
        self._user_stream_subscribed = False
        logging.warning("⚠️ ProjectX user stream disconnected")

    def _on_gateway_user_account(self, arguments) -> None:
        self._record_user_account(self._signalr_payload(arguments))

    def _on_gateway_user_order(self, arguments) -> None:
        self._record_user_order(self._signalr_payload(arguments))

    def _on_gateway_user_position(self, arguments) -> None:
        self._record_user_position(self._signalr_payload(arguments))

    def _on_gateway_user_trade(self, arguments) -> None:
        self._record_user_trade(self._signalr_payload(arguments))

    def _on_gateway_logout(self, arguments) -> None:
        payload = self._signalr_payload(arguments)
        if isinstance(payload, dict):
            payload_text = payload.get("message") or payload.get("reason") or str(payload)
        elif payload not in (None, "", []):
            payload_text = str(payload)
        else:
            payload_text = "no details"
        logging.warning("⚠️ ProjectX user stream logout received: %s", payload_text)
        self._handle_unauthorized_response(
            f"GatewayLogout ({payload_text})",
            restart_user_stream=True,
        )

    async def _subscribe_user_stream(self) -> None:
        if self._user_stream is None:
            return
        try:
            await self._user_stream.invoke("SubscribeAccounts", [])
            if self.account_id is not None:
                await self._user_stream.invoke("SubscribeOrders", [int(self.account_id)])
                await self._user_stream.invoke("SubscribePositions", [int(self.account_id)])
                await self._user_stream.invoke("SubscribeTrades", [int(self.account_id)])
            self._user_stream_subscribed = True
            logging.info("📡 ProjectX user stream subscriptions active")
        except Exception as exc:
            self._user_stream_subscribed = False
            self._user_stream_last_error = str(exc)
            logging.warning(f"ProjectX user stream subscribe failed: {exc}")

    async def start_user_stream(self) -> bool:
        if HubConnectionBuilder is None:
            logging.warning("ProjectX user stream unavailable: signalrcore_async not installed")
            return False
        user_stream_url = self._build_user_stream_connection_url()
        if not user_stream_url or not self.token or self.account_id is None:
            return False
        if self._user_stream_connected and self._user_stream is not None:
            return True

        try:
            self._user_stream_loop = asyncio.get_running_loop()
            self._user_stream_starting = True
            self._user_stream_subscribed = False
            self._user_stream = (
                HubConnectionBuilder()
                .with_url(
                    user_stream_url,
                    options={
                        "access_token_factory": lambda: self.token,
                        "headers": {"Authorization": f"Bearer {self.token}"},
                        "skip_negotiation": True,
                    },
                )
                .with_automatic_reconnect(
                    {
                        "type": "raw",
                        "keep_alive_interval": 10,
                        "reconnect_interval": 5,
                        "max_attempts": 5,
                    }
                )
                .build()
            )
            self._user_stream.on_open(self._on_user_stream_open)
            self._user_stream.on_close(self._on_user_stream_close)
            self._user_stream.on("GatewayUserAccount", self._on_gateway_user_account)
            self._user_stream.on("GatewayUserOrder", self._on_gateway_user_order)
            self._user_stream.on("GatewayUserPosition", self._on_gateway_user_position)
            self._user_stream.on("GatewayUserTrade", self._on_gateway_user_trade)
            self._user_stream.on("GatewayLogout", self._on_gateway_logout)
            await self._user_stream.start()
            self._user_stream_starting = False
            await self._subscribe_user_stream()
            if not self._user_stream_subscribed:
                logging.warning("ProjectX user stream subscriptions failed; falling back to REST polling")
                self._clear_user_stream_runtime()
                return False
            return True
        except Exception as exc:
            self._user_stream_last_error = str(exc)
            self._clear_user_stream_runtime()
            logging.warning(f"ProjectX user stream start failed: {exc}")
            return False

    async def stop_user_stream(self) -> None:
        if self._user_stream is None:
            return
        try:
            await self._user_stream.stop()
        except Exception as exc:
            logging.warning(f"ProjectX user stream stop failed: {exc}")
        finally:
            self._clear_user_stream_runtime()

    def login(self):
        """
        Authenticate via API Key
        Endpoint: POST /api/Auth/loginKey
        Returns JWT token valid for 24 hours
        """
        url = f"{self.base_url}/api/Auth/loginKey"
        payload = {
            "userName": CONFIG['USERNAME'],
            "apiKey": CONFIG['API_KEY']
        }
        try:
            logging.info(f"Authenticating to {self.base_url}...")
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Check for API error response
            if data.get('errorCode') and data.get('errorCode') != 0:
                error_msg = data.get('errorMessage', 'Unknown Error')
                raise ValueError(f"Login Failed. ErrorCode: {data.get('errorCode')} | Msg: {error_msg}")

            self.token = data.get('token')
            if not self.token:
                raise ValueError("Login response missing 'token' field")

            # Token valid for 24 hours
            self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=24)

            # Set auth header for all subsequent requests
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self._clear_external_session_yield()
            logging.info("Authentication successful (JWT token acquired, valid 24h)")
            self._track_general_request()
        except Exception as e:
            logging.error(f"Login Failed: {e}")
            raise

    def validate_session(self) -> bool:
        """
        Validate and refresh session token if needed
        Endpoint: POST /api/Auth/validate
        """
        if self._auth_temporarily_unavailable():
            return False
        if not self._check_general_rate_limit():
            return self.token is not None

        url = f"{self.base_url}/api/Auth/validate"
        try:
            resp = self.session.post(url)
            self._track_general_request()
            if resp.status_code == 200:
                data = resp.json()
                if 'newToken' in data:
                    self.token = data['newToken']
                    self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
                    self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                    logging.info("Session token refreshed")
                return True
            elif resp.status_code == 401:
                logging.warning("Session validation failed: 401")
                return self._handle_unauthorized_response(
                    "POST /api/Auth/validate",
                    restart_user_stream=True,
                )
            else:
                logging.warning(f"Session validation failed: {resp.status_code}")
                return False
        except Exception as e:
            logging.error(f"Session validation error: {e}")
            return False

    def fetch_accounts(self) -> Optional[int]:
        """
        Retrieve active accounts and PROMPT USER for selection using beautiful UI.
        """
        # If a specific ID is hardcoded in CONFIG, use it automatically (good for automation)
        if CONFIG.get('ACCOUNT_ID'):
            self.account_id = CONFIG['ACCOUNT_ID']
            logging.info(f"Using Hardcoded Account ID from Config: {self.account_id}")
            return self.account_id

        try:
            # Try to use the beautiful account selector UI
            from account_selector import select_account_interactive

            selected = select_account_interactive(self.session)

            if selected is None:
                logging.warning("Account selection cancelled")
                return None

            # Handle single account selection (julie001 doesn't support multi-account)
            if isinstance(selected, list):
                # User selected "Monitor All" but julie001 can only trade one account
                print("\n⚠️  Note: Main trading bot can only trade ONE account at a time.")
                print("    Using the first account from your selection.\n")
                self.account_id = selected[0] if selected else None
            else:
                self.account_id = selected

            if self.account_id:
                logging.info(f"User selected account ID: {self.account_id}")
                return self.account_id
            else:
                logging.warning("No account selected")
                return None

        except ImportError:
            # Fallback to simple text-based selection if account_selector not available
            logging.warning("Beautiful UI not available, using simple selection")
            return self._fetch_accounts_simple()
        except Exception as e:
            logging.error(f"Error in account selection: {e}")
            return self._fetch_accounts_simple()

    def _fetch_accounts_simple(self) -> Optional[int]:
        """Fallback simple text-based account selection"""
        url = f"{self.base_url}/api/Account/search"
        payload = {"onlyActiveAccounts": True}

        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()

            if 'accounts' in data and len(data['accounts']) > 0:
                print("\n" + "="*40)
                print("SELECT AN ACCOUNT TO TRADE")
                print("="*40)
                accounts = data['accounts']

                # Print options nicely
                for idx, acc in enumerate(accounts):
                    print(f"  [{idx + 1}] Name: {acc.get('name')}")
                    print(f"      ID: {acc.get('id')}")
                    print("-" * 30)

                # Loop until valid input is received
                while True:
                    try:
                        selection = input(f"Enter number (1-{len(accounts)}): ")
                        choice_idx = int(selection) - 1
                        if 0 <= choice_idx < len(accounts):
                            selected_acc = accounts[choice_idx]
                            self.account_id = selected_acc.get('id')
                            print(f"Selected: {selected_acc.get('name')} (ID: {self.account_id})")
                            logging.info(f"User selected account ID: {self.account_id}")
                            return self.account_id
                        else:
                            print(f"Invalid number. Please enter 1-{len(accounts)}.")
                    except ValueError:
                        print("Please enter a valid number.")
            else:
                logging.warning("No active accounts found")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch accounts: {e}")
            return None

    def get_account_info(self, force_refresh: bool = False) -> Optional[Dict]:
        stream_account = self._get_stream_account()
        if stream_account is not None:
            return stream_account

        if (
            not force_refresh
            and isinstance(self._account_cache, dict)
            and (time.time() - self._account_cache_ts) <= 30.0
        ):
            return dict(self._account_cache)
        if self._auth_temporarily_unavailable():
            return dict(self._account_cache) if isinstance(self._account_cache, dict) else stream_account

        if not self._check_general_rate_limit():
            return dict(self._account_cache) if isinstance(self._account_cache, dict) else stream_account
        if self.account_id is None:
            return None

        url = f"{self.base_url}/api/Account/search"
        payload = {"onlyActiveAccounts": True}

        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            if resp.status_code != 200:
                if resp.status_code == 401:
                    logging.warning("Account search returned 401; using cached account snapshot while auth recovery runs")
                    return dict(self._account_cache) if isinstance(self._account_cache, dict) else None
                logging.warning(f"Account search failed: {resp.status_code} - {resp.text}")
                return dict(self._account_cache) if isinstance(self._account_cache, dict) else None
            data = resp.json()
            accounts = data.get("accounts", [])
            for account in accounts:
                normalized = self._normalize_account_row(account)
                if normalized.get("id") == int(self.account_id):
                    self._account_cache = dict(normalized)
                    self._account_cache_ts = time.time()
                    return dict(normalized)
            return None
        except Exception as exc:
            logging.error(f"Account search error: {exc}")
            return dict(self._account_cache) if isinstance(self._account_cache, dict) else None

    def fetch_contracts(self) -> Optional[str]:
        """
        Get available contracts using Search to find MES futures specifically.
        Endpoint: POST /api/Contract/search
        """
        refresh_target_symbol()

        if not self._check_general_rate_limit():
            return self.contract_id

        url = f"{self.base_url}/api/Contract/search"
        # Search using the root symbol (e.g., "MES") to find all contracts
        payload = {
            "live": False,  # Set to False to find Topstep tradable contracts
            "searchText": self.contract_root
        }

        try:
            logging.info(f"Searching for contracts with symbol: {payload['searchText']}...")
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            resp.raise_for_status()
            data = resp.json()

            if 'contracts' in data and len(data['contracts']) > 0:
                # TARGET_SYMBOL is short form like "MES.Z25" for matching
                target = self.target_symbol or determine_current_contract_symbol(self.contract_root)
                for contract in data['contracts']:
                    contract_id = contract.get('id', '')
                    contract_name = contract.get('name', '')
                    logging.info(f"  Found: {contract_name} ({contract_id})")

                    # Match contract IDs like "CON.F.US.MES.Z25" that end with ".MES.Z25"
                    if contract_id.endswith(f".{target}"):
                        self.contract_id = contract_id
                        logging.info(f"Selected Contract ID: {self.contract_id}")
                        return self.contract_id

                # Fallback: Just take the first one if exact matching logic above misses
                self.contract_id = data['contracts'][0].get('id')
                logging.warning(f"Exact match not confirmed, using first result: {self.contract_id}")
                return self.contract_id
            else:
                logging.warning("No contracts found in search results.")
                return None
        except Exception as e:
            logging.error(f"Failed to fetch contracts: {e}")
            return None

    def retrieve_bars_range(
        self,
        contract_id: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        *,
        minutes_per_bar: int = 1,
        limit: int = 20000,
        live: bool = False,
        request_timeout_sec: float = 30.0,
        raise_on_timeout: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical bars for an explicit contract/time window.

        This is the reusable, non-cached companion to ``get_market_data`` used
        by maintenance tools that need deterministic backfills instead of
        "latest lookback" snapshots.
        """
        if self._auth_temporarily_unavailable():
            return pd.DataFrame()
        if not ProjectXClient._shared_check_bar_rate_limit():
            return pd.DataFrame()
        if self.account_id is None:
            logging.error("No account ID set. Call fetch_accounts() first.")
            return pd.DataFrame()
        contract_id_text = str(contract_id or "").strip()
        if not contract_id_text:
            logging.error("retrieve_bars_range requires a contract_id")
            return pd.DataFrame()

        start_dt = start_time
        end_dt = end_time
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            start_dt = start_dt.astimezone(datetime.timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            end_dt = end_dt.astimezone(datetime.timezone.utc)
        if end_dt <= start_dt:
            logging.warning(
                "retrieve_bars_range received an invalid window for %s: %s -> %s",
                contract_id_text,
                start_dt,
                end_dt,
            )
            return pd.DataFrame()

        url = f"{self.base_url}/api/History/retrieveBars"
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id_text,
            "live": bool(live),
            "limit": max(1, min(int(limit or 20000), 20000)),
            "startTime": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": 2,
            "unitNumber": max(1, int(minutes_per_bar or 1)),
        }

        try:
            resp = self.session.post(
                url,
                json=payload,
                timeout=max(1.0, float(request_timeout_sec or 30.0)),
            )
            ProjectXClient._shared_track_bar_fetch()

            if resp.status_code == 429:
                logging.warning(
                    "Rate limited (429) retrieving %s bars for %s. Returning empty window.",
                    minutes_per_bar,
                    contract_id_text,
                )
                time.sleep(5)
                return pd.DataFrame()
            if resp.status_code == 401:
                logging.warning(
                    "History retrieveBars returned 401 for %s; returning empty window while auth recovery runs",
                    contract_id_text,
                )
                return pd.DataFrame()

            resp.raise_for_status()
            now = time.time()
            self.bar_fetch_timestamps.append(now)
            self.last_bar_fetch_time = now
            data = resp.json()
            bars = data.get("bars") or []
            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars)
            df = df.rename(
                columns={
                    "t": "ts",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])
            if df.empty:
                return pd.DataFrame()
            df["ts"] = df["ts"].dt.tz_convert(self.et)
            df = df.set_index("ts").sort_index()
            return df
        except requests.exceptions.Timeout as exc:
            logging.warning(
                "Range bar fetch timed out for %s (%s -> %s)",
                contract_id_text,
                start_dt,
                end_dt,
            )
            if raise_on_timeout:
                raise
            return pd.DataFrame()
        except requests.exceptions.HTTPError as exc:
            if hasattr(exc, "response") and exc.response is not None:
                logging.error("Range bar fetch HTTP error for %s: %s", contract_id_text, exc)
                logging.error("Server response: %s", exc.response.text)
            else:
                logging.error("Range bar fetch error for %s: %s", contract_id_text, exc)
            return pd.DataFrame()
        except Exception as exc:
            logging.error("Range bar fetch error for %s: %s", contract_id_text, exc)
            return pd.DataFrame()

    def get_market_data(self, lookback_minutes: int = 20000, force_fetch: bool = False) -> pd.DataFrame:
        """
        Fetch historical bars with rate limiting.
        UPDATED: limit increased to 20,000 for deep history (~14 days of 1m data).
        Endpoint: POST /api/History/retrieveBars
        Rate Limit: 50 requests / 30 seconds
        """
        if self._auth_temporarily_unavailable():
            return self.cached_df
        # SHARED rate limit check first (coordinates across MES/MNQ clients)
        if not ProjectXClient._shared_check_bar_rate_limit():
            return self.cached_df

        now = time.time()

        # Instance-level tracking (for per-client diagnostics)
        self.bar_fetch_timestamps = [
            t for t in self.bar_fetch_timestamps
            if now - t < self.BAR_RATE_WINDOW
        ]

        # Instance-level minimum interval (skip if force_fetch)
        if self.last_bar_fetch_time is not None:
            if now - self.last_bar_fetch_time < self.MIN_FETCH_INTERVAL and not force_fetch:
                return self.cached_df

        if self.contract_id is None:
            logging.error("No contract ID set. Call fetch_contracts() first.")
            return self.cached_df

        # Calculate start time based on the massive lookback
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(minutes=lookback_minutes)

        url = f"{self.base_url}/api/History/retrieveBars"
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "live": False,
            "limit": 20000,  # UPDATED TO 20,000 for deep history
            "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "unit": 2,
            "unitNumber": 1
        }

        # Add cache-busting headers
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }

        try:
            resp = self.session.post(url, json=payload, headers=headers)

            # Track request in shared limiter immediately after making the call
            ProjectXClient._shared_track_bar_fetch()

            if resp.status_code == 429:
                logging.warning(f"Rate limited (429) for {self.contract_root}. Backing off 5s...")
                time.sleep(5)
                return self.cached_df
            if resp.status_code == 401:
                logging.warning("History retrieveBars returned 401; using cached data while auth recovery runs")
                return self.cached_df

            resp.raise_for_status()
            self.bar_fetch_timestamps.append(now)
            self.last_bar_fetch_time = now
            data = resp.json()

            # DEBUG: Print last bar timestamp from raw response
            if 'bars' in data and data['bars']:
                newest_raw_bar = data['bars'][0]
                logging.debug(f"API raw: newest bar t={newest_raw_bar.get('t')}, c={newest_raw_bar.get('c')}")

            if 'bars' in data and data['bars']:
                df = pd.DataFrame(data['bars'])
                df = df.rename(columns={
                    't': 'ts', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                })
                # FASTEST - Direct fast-path access
                df['ts'] = pd.to_datetime(df['ts'], utc=True, errors="coerce")
                df = df.dropna(subset=['ts'])
                if df.empty:
                    logging.warning("API returned bars but timestamps could not be parsed.")
                    return self.cached_df if not self.cached_df.empty else pd.DataFrame()
                df['ts'] = df['ts'].dt.tz_convert(self.et)
                df = df.set_index('ts')

                # API returns data in REVERSE chronological order (newest first)
                df = df.iloc[::-1]  # Reverse to get oldest->newest (chronological)

                logging.debug(f"Final df: first={df.index[0]}, last={df.index[-1]}, len={len(df)}")

                self.cached_df = df
                if not df.empty:
                    new_bar_ts = df.index[-1]
                    if self.last_bar_timestamp is None or new_bar_ts > self.last_bar_timestamp:
                        self.last_bar_timestamp = new_bar_ts
                return df
            else:
                logging.warning(f"API returned no bars for {self.contract_id} (timeframe: {start_time} to {end_time})")
                return self.cached_df if not self.cached_df.empty else pd.DataFrame()

        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response'):
                logging.error(f"HTTP Error: {e}")
                logging.error(f"Server Response: {e.response.text}")
                if e.response.status_code == 429:
                    logging.warning("Rate limited. Backing off...")
                    time.sleep(5)
            else:
                logging.error(f"Data fetch error: {e}")
            return self.cached_df
        except Exception as e:
            logging.error(f"Data fetch error: {e}")
            return self.cached_df if not self.cached_df.empty else pd.DataFrame()

    def fetch_custom_bars(self, lookback_bars: int, minutes_per_bar: int) -> pd.DataFrame:
            """
            Fetch historical bars with custom timeframe (for HTF analysis).
            minutes_per_bar: 60 for 1H, 240 for 4H.
            """
            if self._auth_temporarily_unavailable():
                return pd.DataFrame()
            # SHARED rate limit check first (coordinates across MES/MNQ clients)
            if not ProjectXClient._shared_check_bar_rate_limit():
                return pd.DataFrame()

            end_time = datetime.datetime.now(datetime.timezone.utc)
            # Calculate start time based on bars needed * minutes per bar
            total_mins = lookback_bars * minutes_per_bar
            start_time = end_time - datetime.timedelta(minutes=total_mins + 1000) # Buffer

            url = f"{self.base_url}/api/History/retrieveBars"
            payload = {
                "accountId": self.account_id,
                "contractId": self.contract_id,
                "live": False,
                "limit": lookback_bars + 50, # Request slightly more to be safe
                "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "unit": 2,              # 2 = Minutes
                "unitNumber": minutes_per_bar  # 60 or 240
            }

            try:
                resp = self.session.post(url, json=payload)
                # Track in shared limiter immediately
                ProjectXClient._shared_track_bar_fetch()
                self._track_general_request()

                if resp.status_code == 200:
                    data = resp.json()
                    if 'bars' in data and data['bars']:
                        df = pd.DataFrame(data['bars'])
                        df = df.rename(columns={'t': 'ts', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                        df['ts'] = pd.to_datetime(df['ts'], utc=True, errors="coerce")
                        df = df.dropna(subset=['ts'])
                        if df.empty:
                            return pd.DataFrame()
                        df['ts'] = df['ts'].dt.tz_convert(self.et)
                        df = df.set_index('ts').sort_index()
                        return df
                if resp.status_code == 401:
                    logging.warning("HTF retrieveBars returned 401; returning empty dataframe while auth recovery runs")
                    return pd.DataFrame()
                return pd.DataFrame()
            except Exception as e:
                logging.error(f"HTF Data fetch error: {e}")
                return pd.DataFrame()

    def get_rate_limit_status(self) -> str:
        """Returns current rate limit usage (shared across all clients)"""
        now = time.time()
        # Use shared timestamps for accurate cross-client tracking
        shared_bar_recent = len([t for t in ProjectXClient._shared_bar_timestamps if now - t < ProjectXClient.SHARED_BAR_RATE_WINDOW])
        general_recent = len([t for t in self.general_request_timestamps if now - t < self.GENERAL_RATE_WINDOW])
        return f"Bars: {shared_bar_recent}/{ProjectXClient.SHARED_BAR_RATE_LIMIT} (30s) | General: {general_recent}/{self.GENERAL_RATE_LIMIT} (60s)"

    def place_order(self, signal: Dict, current_price: float):
        """
        Place a market order with brackets using SIGNED RELATIVE TICKS.

        CRITICAL FIX:
        - Long TP is positive (+), Long SL is negative (-)
        - Short TP is negative (-), Short SL is positive (+)
        This satisfies the engine requirement: "Ticks should be less than zero when going short."
        """
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot place order")
            return
        if self._order_submission_lockout_active():
            return

        if self.account_id is None:
            logging.error("No account ID set. Call fetch_accounts() first.")
            return

        if self.contract_id is None:
            logging.error("No contract ID set. Call fetch_contracts() first.")
            return

        url = f"{self.base_url}/api/Order/place"
        self._last_order_details = None

        # 1. Determine Side (0=Buy, 1=Sell)
        is_long = (signal['side'] == "LONG")
        side_code = 0 if is_long else 1

        # 2. Calculate Ticks Distance (Absolute)
        # MES tick size is 0.25
        # sl_dist and tp_dist are in POINTS, convert to ticks
        sl_raw = signal.get('sl_dist')
        tp_raw = signal.get('tp_dist')
        if sl_raw is None or tp_raw is None:
            missing = []
            if sl_raw is None:
                missing.append("sl_dist")
            if tp_raw is None:
                missing.append("tp_dist")
            msg = (
                f"⚠️ Strategy {signal.get('strategy', 'Unknown')} missing {', '.join(missing)}; "
                "skipping trade"
            )
            logging.warning(msg)
            event_logger.log_error("MISSING_SLTP", msg)
            return None
        try:
            sl_points = float(sl_raw)
            tp_points = float(tp_raw)
        except (TypeError, ValueError) as exc:
            msg = (
                f"⚠️ Strategy {signal.get('strategy', 'Unknown')} invalid sl/tp values "
                f"(sl={sl_raw}, tp={tp_raw}); skipping trade"
            )
            logging.warning(msg)
            event_logger.log_error("INVALID_SLTP", msg, exception=exc)
            return None
        if not math.isfinite(sl_points) or not math.isfinite(tp_points) or sl_points <= 0 or tp_points <= 0:
            msg = (
                f"⚠️ Strategy {signal.get('strategy', 'Unknown')} non-positive sl/tp values "
                f"(sl={sl_points}, tp={tp_points}); skipping trade"
            )
            logging.warning(msg)
            event_logger.log_error("INVALID_SLTP", msg)
            return None

        # === TICK CONVERSION ===
        # sl_dist/tp_dist are in POINTS; convert to ticks and round up so we don't undershoot
        tick_size = 0.25
        abs_sl_ticks = max(1, int(math.ceil(abs(sl_points) / tick_size)))
        abs_tp_ticks = max(1, int(math.ceil(abs(tp_points) / tick_size)))
        logging.debug(f"📏 SL ({sl_points}pts): {abs_sl_ticks} ticks @ {tick_size}")
        logging.debug(f"🎯 TP ({tp_points}pts): {abs_tp_ticks} ticks @ {tick_size}")


        # 3. Apply Directional Signs based on Side
        if is_long:
            # LONG: Profit is UP (+), Stop is DOWN (-)
            final_tp_ticks = abs_tp_ticks
            final_sl_ticks = -abs_sl_ticks
        else:
            # SHORT: Profit is DOWN (-), Stop is UP (+)
            final_tp_ticks = -abs_tp_ticks
            final_sl_ticks = abs_sl_ticks

        actual_tp_points = abs(final_tp_ticks) * tick_size
        actual_sl_points = abs(final_sl_ticks) * tick_size

        # 4. Generate Unique Client Order ID
        unique_order_id = str(uuid.uuid4())

        # 5. Get order size from signal (allows volatility filter to reduce size)
        order_size = int(signal.get('size', 5))

        # 6. Construct Payload
        # Using 'ticks' with the correct signs calculated above
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "clOrdId": unique_order_id,
            "type": 2,  # Market Order
            "side": side_code,
            "size": order_size,  # Use size from signal (volatility-adjusted)
            "stopLossBracket": {
                "type": 4,      # Stop Market
                "ticks": final_sl_ticks
            },
            "takeProfitBracket": {
                "type": 1,      # Limit
                "ticks": final_tp_ticks
            }
        }

        try:
            # Log exact details for verification (use final tick-derived distances)
            tp_price = current_price + actual_tp_points if is_long else current_price - actual_tp_points
            sl_price = current_price - actual_sl_points if is_long else current_price + actual_sl_points

            # Enhanced event logging: Order about to be placed
            event_logger.log_trade_signal_generated(
                strategy=signal.get('strategy', 'Unknown'),
                side=signal['side'],
                price=current_price,
                tp_dist=actual_tp_points,
                sl_dist=actual_sl_points
            )

            logging.info(f"SENDING ORDER: {signal['side']} @ ~{current_price:.2f}")
            logging.info(f"   Size: {order_size} contracts")
            logging.info(f"   TP: {actual_tp_points:.2f}pts ({final_tp_ticks} ticks, req {tp_points:.2f})")
            logging.info(f"   SL: {actual_sl_points:.2f}pts ({final_sl_ticks} ticks, req {sl_points:.2f})")

            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 429:
                logging.error("Rate limited on order placement!")
                event_logger.log_error("RATE_LIMIT", "Order placement rate limited")
                return None

            if resp.status_code != 200:
                if self._error_indicates_order_lockout(resp.text):
                    self._arm_order_submission_lockout(resp.text)
                logging.error(f"HTTP Error {resp.status_code}: {resp.text[:500] if resp.text else 'Empty response'}")
                return None

            # Only parse JSON after confirming 200 status
            try:
                resp_data = resp.json()
            except Exception as json_err:
                logging.error(f"Failed to parse order response: {json_err}")
                return None

            # Check for business logic success
            if resp_data.get('success') is False:
                err_msg = resp_data.get('errorMessage', 'Unknown Rejection')
                if self._error_indicates_order_lockout(err_msg):
                    self._arm_order_submission_lockout(err_msg)
                logging.error(f"Order Rejected by Engine: {err_msg}")

                # Enhanced event logging: Order rejected
                event_logger.log_trade_order_rejected(
                    side=signal['side'],
                    price=current_price,
                    error_msg=err_msg,
                    strategy=signal.get('strategy', 'Unknown')
                )
                return None

            self._clear_order_submission_lockout()
            logging.info(f"Order Placed Successfully [{unique_order_id[:8]}]")

            entry_price = current_price
            for key in ("averagePrice", "avgPrice", "fillPrice", "price"):
                if key not in resp_data:
                    continue
                try:
                    entry_price = float(resp_data[key])
                    break
                except (TypeError, ValueError):
                    continue

            tp_price = entry_price + actual_tp_points if is_long else entry_price - actual_tp_points
            sl_price = entry_price - actual_sl_points if is_long else entry_price + actual_sl_points

            # Enhanced event logging: Order placed successfully
            event_logger.log_trade_order_placed(
                order_id=unique_order_id,
                side=signal['side'],
                price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                strategy=signal.get('strategy', 'Unknown')
            )

            broker_order_id = self._coerce_int(
                resp_data.get("orderId", resp_data.get("id")),
                None,
            )
            stop_order_id = self._coerce_int(
                resp_data.get("stopLossOrderId", resp_data.get("stopOrderId")),
                None,
            )
            target_order_id = self._coerce_int(
                resp_data.get("takeProfitOrderId", resp_data.get("targetOrderId")),
                None,
            )
            if stop_order_id is None or target_order_id is None:
                detected_ids = self._identify_bracket_order_ids(
                    side=signal['side'],
                    size=order_size,
                    stop_price=sl_price,
                    target_price=tp_price,
                    prefer_stop_order_id=stop_order_id,
                    prefer_target_order_id=target_order_id,
                )
                stop_order_id = stop_order_id or detected_ids.get("stop_order_id")
                target_order_id = target_order_id or detected_ids.get("target_order_id")

            prior_local_position = (
                dict(self._local_position)
                if isinstance(self._local_position, dict)
                else {'side': None, 'size': 0, 'avg_price': 0.0}
            )
            prior_side = str(prior_local_position.get("side") or "").strip().upper()
            prior_size = max(0, self._coerce_int(prior_local_position.get("size"), 0) or 0)
            prior_avg_price = self._coerce_float(prior_local_position.get("avg_price"), None)
            if prior_side == signal['side'] and prior_size > 0 and prior_avg_price is not None:
                blended_size = prior_size + order_size
                blended_avg = (
                    ((prior_avg_price * prior_size) + (entry_price * order_size))
                    / float(blended_size)
                )
                self._local_position = {
                    'side': signal['side'],
                    'size': int(blended_size),
                    'avg_price': float(blended_avg),
                }
            else:
                self._local_position = {
                    'side': signal['side'],
                    'size': order_size,
                    'avg_price': entry_price,
                }
            self._last_order_details = {
                "order_id": unique_order_id,
                "client_order_id": unique_order_id,
                "broker_order_id": broker_order_id,
                "side": signal['side'],
                "entry_price": entry_price,
                "tp_points": actual_tp_points,
                "sl_points": actual_sl_points,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "tp_ticks": final_tp_ticks,
                "sl_ticks": final_sl_ticks,
                "size": order_size,
                "stop_order_id": stop_order_id,
                "target_order_id": target_order_id,
            }

            if stop_order_id is not None:
                self._active_stop_order_id = stop_order_id
                logging.debug(f"Captured stop order ID: {self._active_stop_order_id}")
            elif 'orderId' in resp_data:
                logging.debug(f"Main order ID: {resp_data['orderId']}")
            if target_order_id is not None:
                self._active_target_order_id = target_order_id

            return resp_data

        except Exception as e:
            logging.error(f"Order exception: {e}")
            return None

    def get_position(
        self,
        *,
        prefer_stream: bool = True,
        require_open_pnl: bool = False,
    ) -> Dict:
        """
        Get current position. Tries Search (POST) first, then GET fallback.
        UPDATED: Treats 404 as 'Flat Position' to stop log errors when no trades are open.
        FIX: Returns cached local state if rate limited, preventing 'Fake Flat' signals.
        """
        stream_position = self._get_stream_position() if prefer_stream else None
        if stream_position is not None:
            stream_side = str(stream_position.get("side") or "").strip().upper()
            stream_open_pnl = self._coerce_float(stream_position.get("open_pnl"), None)
            if not require_open_pnl or stream_side not in {"LONG", "SHORT"} or stream_open_pnl is not None:
                self._local_position = stream_position.copy()
                return stream_position
        if self._auth_temporarily_unavailable():
            return stream_position if stream_position is not None else self._stale_position()

        # --- FIX START ---
        if not self._check_general_rate_limit():
            logging.warning(f"⚠️ Rate limit hit in get_position - trusting cached state: {self._local_position}")
            return self._local_position  # Return last known state instead of Flat
        # --- FIX END ---

        if self.account_id is None:
            return {'side': None, 'size': 0, 'avg_price': 0.0}

        payload = {"accountId": self.account_id}
        best_position = None

        try:
            search_endpoints = [
                (f"{self.base_url}/api/Position/searchOpen", "post"),
                (f"{self.base_url}/api/Position/search", "post"),
                (f"{self.base_url}/api/Position", "get"),
            ]
            for url, method in search_endpoints:
                if method == "post":
                    resp = self.session.post(url, json=payload)
                else:
                    resp = self.session.get(url, params=payload)
                self._track_general_request()

                if resp.status_code == 200:
                    data = resp.json()
                    positions = data.get('positions', data) if isinstance(data, dict) else data
                    filtered = self._filter_positions_for_contract(
                        positions if isinstance(positions, list) else []
                    )
                    if filtered:
                        normalized = self._normalize_position(filtered[0])
                        if best_position is None:
                            best_position = normalized
                        normalized_side = str(normalized.get("side") or "").strip().upper()
                        normalized_open_pnl = self._coerce_float(normalized.get("open_pnl"), None)
                        if not require_open_pnl or normalized_side not in {"LONG", "SHORT"} or normalized_open_pnl is not None:
                            self._local_position = normalized.copy()
                            return normalized
                    continue

                if resp.status_code in (400, 404):
                    continue
                if resp.status_code == 401:
                    logging.warning("Position check returned 401; using stream/cached position while auth recovery runs")
                    return stream_position if stream_position is not None else self._stale_position()

                logging.warning(f"Position check failed: {resp.status_code} - {resp.text}")
                return stream_position if stream_position is not None else self._stale_position()

            if best_position is not None:
                self._local_position = best_position.copy()
                return best_position
            if stream_position is not None:
                self._local_position = stream_position.copy()
                return stream_position
            self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
            return {'side': None, 'size': 0, 'avg_price': 0.0}

        except Exception as e:
            logging.error(f"Position check error: {e}")
            return stream_position if stream_position is not None else self._stale_position()

    def close_position(self, position: Dict) -> bool:
        """
        Close an existing position and explicitly clean up any remaining exit orders.
        """
        if position['side'] is None or position['size'] == 0:
            return True  # Nothing to close

        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot close position")
            return False

        self._last_close_order_details = None

        try:
            close_method = None
            close_price = None
            close_order_id = None

            contract_close_url = f"{self.base_url}/api/Position/closeContract"
            contract_payload = {
                "accountId": self.account_id,
                "contractId": self.contract_id,
            }
            logging.info(
                "CLOSING POSITION via closeContract: %s %s contracts @ ~%.2f",
                position['side'],
                position['size'],
                position['avg_price'],
            )
            resp = self.session.post(contract_close_url, json=contract_payload)
            self._track_general_request()

            if resp.status_code == 429:
                logging.error("Rate limited on position close!")
                event_logger.log_error("RATE_LIMIT", "Position close rate limited")
                return False

            resp_data = None
            if resp.status_code == 200:
                try:
                    resp_data = resp.json()
                except Exception as json_err:
                    logging.error(f"Failed to parse closeContract response: {json_err}")
                if isinstance(resp_data, dict) and resp_data.get('success', False):
                    close_method = "closeContract"
                else:
                    logging.warning(f"closeContract rejected: {resp_data}")
            else:
                logging.warning(
                    "closeContract failed: %s - %s",
                    resp.status_code,
                    resp.text[:500] if resp.text else 'Empty response',
                )

            if close_method is None:
                url = f"{self.base_url}/api/Order/place"
                if position['side'] == 'LONG':
                    side_code = 1
                    action = "SELL"
                else:
                    side_code = 0
                    action = "BUY"
                payload = {
                    "accountId": self.account_id,
                    "contractId": self.contract_id,
                    "clOrdId": str(uuid.uuid4()),
                    "type": 2,
                    "side": side_code,
                    "size": position['size'],
                }

                logging.info(
                    "Falling back to market close: %s %s contracts to close %s @ ~%.2f",
                    action,
                    position['size'],
                    position['side'],
                    position['avg_price'],
                )
                resp = self.session.post(url, json=payload)
                self._track_general_request()

                if resp.status_code == 429:
                    logging.error("Rate limited on position close fallback!")
                    event_logger.log_error("RATE_LIMIT", "Position close rate limited")
                    return False

                if resp.status_code != 200:
                    logging.error(f"Position close HTTP Error {resp.status_code}: {resp.text[:500] if resp.text else 'Empty response'}")
                    event_logger.log_error("POSITION_CLOSE_FAILED", f"HTTP {resp.status_code}")
                    return False

                try:
                    resp_data = resp.json()
                except Exception as json_err:
                    logging.error(f"Failed to parse close response: {json_err}")
                    return False

                if not resp_data.get('success', False):
                    logging.error(f"Position close rejected: {resp_data}")
                    event_logger.log_error("POSITION_CLOSE_FAILED", f"Failed to close position: {resp_data}")
                    return False

                close_method = "market_order"
                close_price = self._extract_execution_price(resp_data, None)
                close_order_id = self._coerce_int(
                    resp_data.get('orderId', resp_data.get('id')),
                    None,
                )

            self._last_close_order_details = {
                "order_id": close_order_id,
                "side": position['side'],
                "size": int(position['size']),
                "exit_price": close_price,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "method": close_method,
            }
            logging.info(
                "Position close submitted via %s; awaiting confirmed close reconciliation",
                close_method,
            )

            time.sleep(0.35)
            post_close_position = self.get_position()
            if not post_close_position.get("stale") and (
                post_close_position.get("side") is None or post_close_position.get("size", 0) == 0
            ):
                cancelled = self.cancel_open_exit_orders(
                    side=None,
                    reason=f"{close_method} cleanup",
                )
                if cancelled:
                    logging.info(f"Cancelled {cancelled} orphan exit order(s) after {close_method}")
                self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
                self._active_stop_order_id = None
                self._active_target_order_id = None
            else:
                logging.warning(
                    "Position still not confirmed flat after %s; skipping exit-order cleanup",
                    close_method,
                )

            return True
        except Exception as e:
            logging.error(f"Position close exception: {e}")
            event_logger.log_error("POSITION_CLOSE_EXCEPTION", f"Exception closing position: {e}", exception=e)
            return False

    def close_trade_leg(self, trade: Dict) -> bool:
        """
        Close a specific tracked same-side leg without flattening the entire position.
        The caller is responsible for ensuring the trade size is <= current broker position size.
        """
        if not isinstance(trade, dict):
            return False
        side_name = str(trade.get("side") or "").strip().upper()
        trade_size = max(1, self._coerce_int(trade.get("size"), 1) or 1)
        if side_name not in {"LONG", "SHORT"}:
            return False
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot close trade leg")
            return False

        self._last_close_order_details = None
        for order_key in ("stop_order_id", "target_order_id"):
            order_id = self._coerce_int(trade.get(order_key), None)
            if order_id is None:
                continue
            try:
                self.cancel_order(order_id)
            except Exception:
                pass
            if self._active_stop_order_id == order_id:
                self._active_stop_order_id = None
            if self._active_target_order_id == order_id:
                self._active_target_order_id = None
        self._order_cache = {}
        self._order_cache_ts = 0.0
        time.sleep(0.2)

        url = f"{self.base_url}/api/Order/place"
        side_code = 1 if side_name == "LONG" else 0
        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "clOrdId": str(uuid.uuid4()),
            "type": 2,
            "side": side_code,
            "size": trade_size,
        }

        try:
            logging.info(
                "Closing tracked trade leg: %s %s contracts (%s)",
                side_name,
                trade_size,
                trade.get("strategy", "Unknown"),
            )
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            if resp.status_code == 429:
                logging.error("Rate limited on trade-leg close")
                event_logger.log_error("RATE_LIMIT", "Trade-leg close rate limited")
                return False
            if resp.status_code != 200:
                logging.error(
                    "Trade-leg close HTTP Error %s: %s",
                    resp.status_code,
                    resp.text[:500] if resp.text else "Empty response",
                )
                event_logger.log_error("TRADE_LEG_CLOSE_FAILED", f"HTTP {resp.status_code}")
                return False
            try:
                resp_data = resp.json()
            except Exception as json_err:
                logging.error("Failed to parse trade-leg close response: %s", json_err)
                return False
            if not resp_data.get("success", False):
                logging.error("Trade-leg close rejected: %s", resp_data)
                event_logger.log_error("TRADE_LEG_CLOSE_FAILED", f"Rejected: {resp_data}")
                return False

            close_price = self._extract_execution_price(resp_data, None)
            close_order_id = self._coerce_int(
                resp_data.get("orderId", resp_data.get("id")),
                None,
            )
            self._last_close_order_details = {
                "order_id": close_order_id,
                "side": side_name,
                "size": int(trade_size),
                "exit_price": close_price,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "method": "partial_market_order",
            }
            time.sleep(0.35)
            refreshed_position = self.get_position()
            if isinstance(refreshed_position, dict) and not refreshed_position.get("stale"):
                self._local_position = refreshed_position.copy()
                if refreshed_position.get("side") is None or refreshed_position.get("size", 0) == 0:
                    self._active_stop_order_id = None
                    self._active_target_order_id = None
            return True
        except Exception as e:
            logging.error(f"Trade-leg close exception: {e}")
            event_logger.log_error("TRADE_LEG_CLOSE_EXCEPTION", f"Exception closing trade leg: {e}", exception=e)
            return False

    def close_and_reverse(self, new_signal: Dict, current_price: float, opposite_signal_count: int) -> Tuple[bool, int]:
        """
        Close the current position and enter the new side immediately.
        The confirmation policy is handled upstream in julie001.py; the
        `opposite_signal_count` argument is kept only for API compatibility.
        """
        # Always sync shadow position with broker before deciding
        position = self.get_position()
        self._local_position = position.copy()
        if position.get("stale"):
            logging.warning("Position state stale; skipping order placement.")
            return False, opposite_signal_count

        # If no position, just place the order and reset count
        if position['side'] is None:
            resp = self.place_order(new_signal, current_price)
            return (resp is not None), 0

        # Same-side live adds diverge from the single-active-trade backtest/tracker model.
        # Ignore them defensively even if an upstream caller forgets to short-circuit.
        if position['side'] == new_signal['side']:
            logging.info(
                "Ignoring same-side live signal: already %s %s contracts, signal=%s",
                position['side'],
                position['size'],
                new_signal.get('strategy', 'Unknown'),
            )
            return False, 0

        logging.info(
            "Confirmed opposite live signal - closing %s %s contracts and reversing to %s",
            position['side'],
            position['size'],
            new_signal['side'],
        )

        event_logger.log_close_and_reverse(
            old_side=position['side'],
            new_side=new_signal['side'],
            price=current_price,
            strategy=new_signal.get('strategy', 'Unknown')
        )

        close_success = self.close_position(position)
        if not close_success:
            logging.error("Failed to close existing position, aborting new order")
            return False, opposite_signal_count

        time.sleep(0.5)

        resp = self.place_order(new_signal, current_price)
        return (resp is not None), 0

    def _update_order_cache(self, orders: List[Dict]) -> None:
        self._order_cache = {
            o.get("orderId"): o
            for o in orders
            if o.get("orderId") is not None
        }
        self._order_cache_ts = time.time()

    def _get_cached_orders(self, max_age_sec: float = 2.0, force_refresh: bool = False) -> List[Dict]:
        if force_refresh or not self._order_cache or (time.time() - self._order_cache_ts) > max_age_sec:
            orders = self.search_orders()
            self._update_order_cache(orders)
        return list(self._order_cache.values())

    def _extract_order_stop_price(self, order: Dict) -> Optional[float]:
        for key in ("stopPrice", "triggerPrice", "price"):
            value = order.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _extract_order_limit_price(self, order: Dict) -> Optional[float]:
        for key in ("limitPrice", "price", "triggerPrice"):
            value = order.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _select_stop_order(
        self,
        orders: List[Dict],
        expected_side: Optional[int],
        expected_price: Optional[float],
        expected_size: Optional[int],
        prefer_order_id: Optional[int],
        price_tolerance: float = 0.5,
    ) -> Optional[Dict]:
        candidates = [
            o for o in orders
            if o.get("type") in [4, 5]
            and self._order_has_open_status(o)
        ]
        if expected_side is not None:
            candidates = [o for o in candidates if o.get("side") == expected_side]

        if prefer_order_id is not None:
            for o in candidates:
                if o.get("orderId") == prefer_order_id:
                    return o

        if expected_size is not None:
            size_matches = [o for o in candidates if o.get("size") == expected_size]
            if size_matches:
                candidates = size_matches

        if expected_price is not None:
            scored = []
            for o in candidates:
                stop_price = self._extract_order_stop_price(o)
                if stop_price is None:
                    continue
                scored.append((abs(stop_price - expected_price), o))
            if scored:
                scored.sort(key=lambda item: item[0])
                if scored[0][0] <= price_tolerance:
                    return scored[0][1]
                return None

        if not candidates:
            return None
        candidates.sort(key=lambda o: o.get("orderId", 0), reverse=True)
        return candidates[0]

    def _select_limit_order(
        self,
        orders: List[Dict],
        expected_side: Optional[int],
        expected_price: Optional[float],
        expected_size: Optional[int],
        prefer_order_id: Optional[int],
        price_tolerance: float = 0.5,
    ) -> Optional[Dict]:
        candidates = [
            o for o in orders
            if o.get("type") == 1
            and self._order_has_open_status(o)
        ]
        if expected_side is not None:
            candidates = [o for o in candidates if o.get("side") == expected_side]

        if prefer_order_id is not None:
            for o in candidates:
                if o.get("orderId") == prefer_order_id:
                    return o

        if expected_size is not None:
            size_matches = [o for o in candidates if o.get("size") == expected_size]
            if size_matches:
                candidates = size_matches

        if expected_price is not None:
            scored = []
            for o in candidates:
                limit_price = self._extract_order_limit_price(o)
                if limit_price is None:
                    continue
                scored.append((abs(limit_price - expected_price), o))
            if scored:
                scored.sort(key=lambda item: item[0])
                if scored[0][0] <= price_tolerance:
                    return scored[0][1]
                return None

        if not candidates:
            return None
        candidates.sort(key=lambda o: o.get("orderId", 0), reverse=True)
        return candidates[0]

    def _identify_bracket_order_ids(
        self,
        *,
        side: str,
        size: int,
        stop_price: Optional[float],
        target_price: Optional[float],
        prefer_stop_order_id: Optional[int] = None,
        prefer_target_order_id: Optional[int] = None,
        max_attempts: int = 3,
        settle_delay_sec: float = 0.2,
    ) -> Dict[str, Optional[int]]:
        side_name = str(side or "").strip().upper()
        expected_side = 1 if side_name == "LONG" else 0 if side_name == "SHORT" else None
        expected_size = max(1, int(size)) if size is not None else None
        detected_stop_id = self._coerce_int(prefer_stop_order_id, None)
        detected_target_id = self._coerce_int(prefer_target_order_id, None)
        stop_price_val = self._coerce_float(stop_price, None)
        target_price_val = self._coerce_float(target_price, None)

        for attempt in range(max(1, int(max_attempts))):
            orders = self._get_cached_orders(max_age_sec=0.0, force_refresh=True)
            if detected_stop_id is None:
                stop_order = self._select_stop_order(
                    orders,
                    expected_side=expected_side,
                    expected_price=stop_price_val,
                    expected_size=expected_size,
                    prefer_order_id=prefer_stop_order_id,
                    price_tolerance=1.0,
                )
                detected_stop_id = (
                    self._coerce_int(stop_order.get("orderId"), None)
                    if isinstance(stop_order, dict)
                    else None
                )
            if detected_target_id is None:
                target_order = self._select_limit_order(
                    orders,
                    expected_side=expected_side,
                    expected_price=target_price_val,
                    expected_size=expected_size,
                    prefer_order_id=prefer_target_order_id,
                    price_tolerance=1.0,
                )
                detected_target_id = (
                    self._coerce_int(target_order.get("orderId"), None)
                    if isinstance(target_order, dict)
                    else None
                )
            if detected_stop_id is not None or detected_target_id is not None:
                if detected_stop_id is not None and detected_target_id is not None:
                    break
            if attempt + 1 < max(1, int(max_attempts)):
                time.sleep(max(0.0, float(settle_delay_sec)))

        return {
            "stop_order_id": detected_stop_id,
            "target_order_id": detected_target_id,
        }

    def _list_open_stop_orders(
        self,
        side: Optional[str] = None,
        *,
        expected_size: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[Dict]:
        expected_side = None
        side_name = str(side or "").strip().upper()
        if side_name == "LONG":
            expected_side = 1
        elif side_name == "SHORT":
            expected_side = 0

        orders = self._get_cached_orders(max_age_sec=0.0 if force_refresh else 2.0, force_refresh=force_refresh)
        candidates: List[Dict] = []
        for order in orders:
            if not isinstance(order, dict) or not self._order_has_open_status(order):
                continue
            order_type = self._coerce_int(order.get("type"), None)
            if order_type not in {4, 5}:
                continue
            order_side = self._coerce_int(order.get("side"), None)
            if expected_side is not None and order_side not in (None, expected_side):
                continue
            order_size = self._coerce_int(order.get("size"), None)
            if expected_size is not None and order_size not in (None, expected_size):
                continue
            candidates.append(self._normalize_order_row(order))
        return candidates

    def _pick_preferred_stop_order(
        self,
        orders: List[Dict],
        *,
        expected_price: Optional[float],
        prefer_order_id: Optional[int],
    ) -> Optional[Dict]:
        if not orders:
            return None

        expected_price_val = self._coerce_float(expected_price, None)
        scored: List[Tuple[float, int, int, Dict]] = []
        for order in orders:
            order_id = self._coerce_int(order.get("orderId"), None)
            stop_price = self._extract_order_stop_price(order)
            if expected_price_val is not None and stop_price is not None:
                price_distance = abs(float(stop_price) - float(expected_price_val))
            elif expected_price_val is None:
                price_distance = 0.0
            else:
                price_distance = 1e9
            prefer_penalty = 0 if prefer_order_id is not None and order_id == prefer_order_id else 1
            freshness_penalty = -(order_id or 0)
            scored.append((float(price_distance), int(prefer_penalty), int(freshness_penalty), order))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return scored[0][3]

    def _cancel_order_rows(
        self,
        orders: List[Dict],
        *,
        reason: str = "",
        settle_delay_sec: float = 0.35,
    ) -> int:
        cancelled = 0
        if reason and orders:
            logging.info("Cancelling %s order(s) for %s", len(orders), reason)
        any_cancelled = False
        for order in orders:
            order_id = self._coerce_int(order.get("orderId"), None)
            if order_id is None:
                continue
            if self.cancel_order(order_id):
                cancelled += 1
                any_cancelled = True
                if self._active_stop_order_id == order_id:
                    self._active_stop_order_id = None
                if self._active_target_order_id == order_id:
                    self._active_target_order_id = None
        if any_cancelled:
            self._order_cache = {}
            self._order_cache_ts = 0.0
            time.sleep(max(0.0, float(settle_delay_sec)))
        return cancelled

    def _cancel_open_stop_orders(
        self,
        side: Optional[str] = None,
        *,
        expected_size: Optional[int] = None,
        exclude_order_id: Optional[int] = None,
        reason: str = "",
        max_attempts: int = 3,
        settle_delay_sec: float = 0.35,
    ) -> int:
        cancelled = 0
        for _ in range(max(1, int(max_attempts))):
            candidates = self._list_open_stop_orders(
                side,
                expected_size=expected_size,
                force_refresh=True,
            )
            if exclude_order_id is not None:
                candidates = [
                    order for order in candidates
                    if self._coerce_int(order.get("orderId"), None) != exclude_order_id
                ]
            if not candidates:
                break
            cancelled_now = self._cancel_order_rows(
                candidates,
                reason=reason,
                settle_delay_sec=settle_delay_sec,
            )
            if cancelled_now <= 0:
                break
            cancelled += int(cancelled_now)
        return cancelled

    def _cleanup_duplicate_stop_orders(
        self,
        side: str,
        *,
        expected_size: Optional[int],
        expected_price: Optional[float],
        prefer_order_id: Optional[int],
        settle_delay_sec: float = 0.35,
    ) -> Optional[int]:
        stop_orders = self._list_open_stop_orders(side, expected_size=expected_size, force_refresh=True)
        if not stop_orders:
            return None

        preferred = self._pick_preferred_stop_order(
            stop_orders,
            expected_price=expected_price,
            prefer_order_id=prefer_order_id,
        )
        keep_order_id = self._coerce_int(preferred.get("orderId"), None) if isinstance(preferred, dict) else None
        duplicate_orders = [
            order for order in stop_orders
            if self._coerce_int(order.get("orderId"), None) != keep_order_id
        ]
        if duplicate_orders:
            logging.warning(
                "Detected %s open stop orders for %s; keeping %s and cancelling %s duplicate(s).",
                len(stop_orders),
                str(side or "").strip().upper(),
                keep_order_id,
                len(duplicate_orders),
            )
            self._cancel_order_rows(
                duplicate_orders,
                reason="duplicate stop cleanup",
                settle_delay_sec=settle_delay_sec,
            )
            stop_orders = self._list_open_stop_orders(side, expected_size=expected_size, force_refresh=True)
            preferred = self._pick_preferred_stop_order(
                stop_orders,
                expected_price=expected_price,
                prefer_order_id=keep_order_id,
            )
            keep_order_id = self._coerce_int(preferred.get("orderId"), None) if isinstance(preferred, dict) else None

        if keep_order_id is not None:
            self._active_stop_order_id = keep_order_id
        return keep_order_id

    def search_orders(self) -> List[Dict]:
        """
        Search for open orders (bracket orders) for the account.
        Endpoint: POST /api/Order/searchOpen (preferred) or /api/Order/search
        Returns: List of order dicts with orderId, type, side, price, etc.
        """
        if self._auth_temporarily_unavailable():
            return list(self._order_cache.values())
        if not self._check_general_rate_limit():
            return []

        if self.account_id is None:
            return []
        payload = {"accountId": self.account_id}

        try:
            search_endpoints = [
                f"{self.base_url}/api/Order/searchOpen",
                f"{self.base_url}/api/Order/search",
            ]
            for url in search_endpoints:
                resp = self.session.post(url, json=payload)
                self._track_general_request()
                if resp.status_code == 200:
                    data = resp.json()
                    orders = data.get('orders', [])
                    filtered = [
                        self._normalize_order_row(o)
                        for o in orders
                        if isinstance(o, dict) and o.get('contractId') == self.contract_id
                    ]
                    self._update_order_cache(filtered)
                    return filtered
                if resp.status_code in (400, 404):
                    continue
                if resp.status_code == 401:
                    logging.warning("Order search returned 401; auth recovery is in progress")
                    return []
                logging.warning(f"Order search failed: {resp.status_code} - {resp.text}")
                return []
            return []
        except Exception as e:
            logging.error(f"Order search error: {e}")
            return []

    def cancel_open_exit_orders(
        self,
        side: Optional[str] = None,
        *,
        reason: str = "",
        max_attempts: int = 3,
        settle_delay_sec: float = 0.35,
    ) -> int:
        expected_side = None
        side_name = str(side or "").strip().upper()
        if side_name == "LONG":
            expected_side = 1
        elif side_name == "SHORT":
            expected_side = 0

        cancelled = 0
        for _ in range(max(1, int(max_attempts))):
            orders = self._get_cached_orders(max_age_sec=0.0, force_refresh=True)
            candidates: List[Dict] = []
            for order in orders:
                if not isinstance(order, dict) or not self._order_has_open_status(order):
                    continue
                order_type = self._coerce_int(order.get("type"), None)
                if order_type not in {1, 4, 5}:
                    continue
                order_side = self._coerce_int(order.get("side"), None)
                if expected_side is not None and order_side not in (None, expected_side):
                    continue
                candidates.append(order)

            if not candidates:
                break

            if reason:
                logging.info("Cancelling %s open exit order(s) for %s", len(candidates), reason)

            any_cancelled = False
            for order in candidates:
                order_id = self._coerce_int(order.get("orderId"), None)
                if order_id is None:
                    continue
                if self.cancel_order(order_id):
                    cancelled += 1
                    any_cancelled = True
                    if self._active_stop_order_id == order_id:
                        self._active_stop_order_id = None
                    if self._active_target_order_id == order_id:
                        self._active_target_order_id = None

            if not any_cancelled:
                break

            self._order_cache = {}
            self._order_cache_ts = 0.0
            time.sleep(max(0.0, float(settle_delay_sec)))

        return cancelled

    def get_live_bracket_state(
        self,
        side: str,
        size: Optional[int] = None,
        reference_price: Optional[float] = None,
        expected_stop_price: Optional[float] = None,
        expected_target_price: Optional[float] = None,
        prefer_stop_order_id: Optional[int] = None,
        prefer_target_order_id: Optional[int] = None,
        max_cache_age_sec: float = 15.0,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        side_name = str(side or "").strip().upper()
        if side_name not in {"LONG", "SHORT"}:
            return {}

        try:
            expected_size = self._coerce_int(size, None) if size is not None else None
            exit_side = 1 if side_name == "LONG" else 0
            ref_price = self._coerce_float(reference_price, None)
            stop_price_val = self._coerce_float(expected_stop_price, None)
            target_price_val = self._coerce_float(expected_target_price, None)
            orders = self._get_cached_orders(
                max_age_sec=0.0 if force_refresh else max_cache_age_sec,
                force_refresh=force_refresh,
            )
        except Exception:
            return {}

        stop_order = self._select_stop_order(
            orders,
            expected_side=exit_side,
            expected_price=stop_price_val,
            expected_size=expected_size,
            prefer_order_id=prefer_stop_order_id,
            price_tolerance=4.0,
        )
        if stop_order is None and (prefer_stop_order_id is not None or stop_price_val is not None):
            stop_order = self._select_stop_order(
                orders,
                expected_side=exit_side,
                expected_price=None,
                expected_size=expected_size,
                prefer_order_id=prefer_stop_order_id,
            )

        target_order = self._select_limit_order(
            orders,
            expected_side=exit_side,
            expected_price=target_price_val,
            expected_size=expected_size,
            prefer_order_id=prefer_target_order_id,
            price_tolerance=4.0,
        )
        if target_order is None and (prefer_target_order_id is not None or target_price_val is not None):
            target_order = self._select_limit_order(
                orders,
                expected_side=exit_side,
                expected_price=None,
                expected_size=expected_size,
                prefer_order_id=prefer_target_order_id,
            )

        snapshot: Dict[str, Any] = {}
        if isinstance(stop_order, dict):
            stop_order = self._normalize_order_row(stop_order)
            stop_order_id = self._coerce_int(stop_order.get("orderId"), None)
            stop_price = self._extract_order_stop_price(stop_order)
            if stop_order_id is not None:
                snapshot["stop_order_id"] = stop_order_id
            if stop_price is not None:
                snapshot["stop_price"] = float(stop_price)
                snapshot["sl_price"] = float(stop_price)

        if isinstance(target_order, dict):
            target_order = self._normalize_order_row(target_order)
            target_order_id = self._coerce_int(target_order.get("orderId"), None)
            target_price = self._extract_order_limit_price(target_order)
            if (
                ref_price is not None
                and target_price is not None
                and (
                    (side_name == "LONG" and target_price <= ref_price)
                    or (side_name == "SHORT" and target_price >= ref_price)
                )
                and target_price_val is None
                and prefer_target_order_id is None
            ):
                target_order = None
            else:
                if target_order_id is not None:
                    snapshot["target_order_id"] = target_order_id
                if target_price is not None:
                    snapshot["target_price"] = float(target_price)
                    snapshot["tp_price"] = float(target_price)

        return snapshot

    def get_live_bracket_snapshot(
        self,
        side: str,
        size: Optional[int] = None,
        reference_price: Optional[float] = None,
        max_cache_age_sec: float = 15.0,
    ) -> Dict[str, float]:
        state = self.get_live_bracket_state(
            side,
            size=size,
            reference_price=reference_price,
            max_cache_age_sec=max_cache_age_sec,
        )
        snapshot: Dict[str, float] = {}
        stop_price = self._coerce_float(state.get("stop_price", state.get("sl_price")), None)
        if stop_price is not None:
            snapshot["stop_price"] = float(stop_price)
            snapshot["sl_price"] = float(stop_price)
        target_price = self._coerce_float(state.get("target_price", state.get("tp_price")), None)
        if target_price is not None:
            snapshot["target_price"] = float(target_price)
            snapshot["tp_price"] = float(target_price)
        return snapshot

    def search_trades(
        self,
        start_time: datetime.datetime,
        end_time: Optional[datetime.datetime] = None,
    ) -> List[Dict]:
        """
        Search recent trades for the account/contract.
        Endpoint: POST /api/Trade/search
        """
        if self._auth_temporarily_unavailable():
            return []
        if not self._check_general_rate_limit():
            return []
        if self.account_id is None:
            return []

        url = f"{self.base_url}/api/Trade/search"
        payload = {
            "accountId": self.account_id,
            "startTimestamp": self._utc_isoformat(start_time),
        }
        end_iso = self._utc_isoformat(end_time)
        if end_iso:
            payload["endTimestamp"] = end_iso

        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            if resp.status_code != 200:
                if resp.status_code in (400, 404):
                    return []
                if resp.status_code == 401:
                    logging.warning("Trade search returned 401; auth recovery is in progress")
                    return []
                logging.warning(f"Trade search failed: {resp.status_code} - {resp.text}")
                return []
            data = resp.json()
            rows = self._extract_trade_rows(data)
            filtered = [
                row for row in rows
                if row.get("contractId") == self.contract_id
                and not bool(row.get("voided", False))
            ]
            filtered.sort(
                key=lambda row: self._parse_trade_timestamp(row) or datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                )
            )
            return filtered
        except Exception as e:
            logging.error(f"Trade search error: {e}")
            return []

    def reconstruct_closed_trades(
        self,
        start_time: datetime.datetime,
        end_time: Optional[datetime.datetime] = None,
        *,
        include_stream_trades: bool = True,
    ) -> List[Dict]:
        """
        Reconstruct round-trip closed trades from ProjectX raw trade rows.

        ProjectX entry fills typically have ``profitAndLoss=None`` while close fills
        carry realized ``profitAndLoss``. We pair those close rows back to the
        preceding opposite-side entry inventory so missed live closes can be
        backfilled later.
        """
        merged_rows: List[Dict] = []
        if include_stream_trades:
            merged_rows = self._get_stream_trades(start_time, end_time)
        rest_rows = self.search_trades(start_time, end_time)
        merged_rows = self._merge_trade_rows(merged_rows, rest_rows)
        if not merged_rows:
            return []

        point_value = self._coerce_float(
            ((CONFIG.get("RISK_MANAGEMENT") or {}).get("POINT_VALUE")),
            5.0,
        ) or 5.0

        open_inventory: Dict[int, deque] = {
            0: deque(),  # buy-side entries => long inventory
            1: deque(),  # sell-side entries => short inventory
        }
        grouped_closes: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        close_order: List[Tuple[Any, ...]] = []

        for row in merged_rows:
            if not isinstance(row, dict):
                continue
            if row.get("contractId") != self.contract_id:
                continue
            if bool(row.get("voided", False)):
                continue

            side_code = self._coerce_int(row.get("side"), None)
            fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
            fill_price = self._coerce_float(row.get("price"), None)
            row_ts = self._parse_trade_timestamp(row)
            if side_code not in (0, 1) or fill_qty <= 0 or row_ts is None:
                continue

            pnl_value = self._coerce_float(row.get("profitAndLoss"), None)
            if pnl_value is None:
                open_inventory[side_code].append(
                    {
                        "row": dict(row),
                        "remaining_qty": int(fill_qty),
                        "timestamp": row_ts,
                        "price": fill_price,
                        "order_id": self._coerce_int(row.get("orderId"), None),
                    }
                )
                continue

            entry_side_code = 0 if side_code == 1 else 1
            matched_entry_legs: List[Dict[str, Any]] = []
            remaining_qty = int(fill_qty)
            while remaining_qty > 0 and open_inventory[entry_side_code]:
                entry_leg = open_inventory[entry_side_code][0]
                leg_qty = min(
                    remaining_qty,
                    max(0, int(entry_leg.get("remaining_qty", 0) or 0)),
                )
                if leg_qty <= 0:
                    open_inventory[entry_side_code].popleft()
                    continue
                matched_entry_legs.append(
                    {
                        "qty": int(leg_qty),
                        "timestamp": entry_leg.get("timestamp"),
                        "price": entry_leg.get("price"),
                        "order_id": entry_leg.get("order_id"),
                        "row": dict(entry_leg.get("row") or {}),
                    }
                )
                entry_leg["remaining_qty"] = int(entry_leg.get("remaining_qty", 0) or 0) - int(leg_qty)
                remaining_qty -= int(leg_qty)
                if int(entry_leg.get("remaining_qty", 0) or 0) <= 0:
                    open_inventory[entry_side_code].popleft()

            if not matched_entry_legs:
                continue

            close_order_id = self._coerce_int(row.get("orderId"), None)
            group_key = (
                "order",
                close_order_id,
            ) if close_order_id is not None else (
                "close",
                row_ts.isoformat(),
                side_code,
                fill_qty,
                fill_price,
            )
            group = grouped_closes.get(group_key)
            if group is None:
                group = {
                    "side": "LONG" if entry_side_code == 0 else "SHORT",
                    "entry_legs": [],
                    "exit_rows": [],
                    "exit_qty": 0,
                    "entry_qty": 0,
                    "weighted_exit": 0.0,
                    "pnl_dollars": 0.0,
                    "exit_time": None,
                    "close_order_id": close_order_id,
                }
                grouped_closes[group_key] = group
                close_order.append(group_key)

            group["exit_rows"].append(dict(row))
            group["entry_legs"].extend(matched_entry_legs)
            group["exit_qty"] += int(fill_qty)
            group["entry_qty"] += sum(int(leg.get("qty", 0) or 0) for leg in matched_entry_legs)
            if fill_price is not None:
                group["weighted_exit"] += float(fill_price) * float(fill_qty)
            group["pnl_dollars"] += float(pnl_value)
            if group["exit_time"] is None or row_ts > group["exit_time"]:
                group["exit_time"] = row_ts

        reconstructed: List[Dict[str, Any]] = []
        for group_key in close_order:
            group = grouped_closes.get(group_key) or {}
            entry_legs = list(group.get("entry_legs") or [])
            exit_qty = max(0, int(group.get("exit_qty", 0) or 0))
            entry_qty = max(0, int(group.get("entry_qty", 0) or 0))
            matched_qty = min(exit_qty, entry_qty)
            if matched_qty <= 0:
                continue

            entry_weighted = 0.0
            entry_time: Optional[datetime.datetime] = None
            entry_order_ids: List[int] = []
            for leg in entry_legs:
                leg_qty = max(0, int(leg.get("qty", 0) or 0))
                leg_price = self._coerce_float(leg.get("price"), None)
                if leg_qty > 0 and leg_price is not None:
                    entry_weighted += float(leg_price) * float(leg_qty)
                leg_ts = leg.get("timestamp")
                if isinstance(leg_ts, datetime.datetime) and (
                    entry_time is None or leg_ts < entry_time
                ):
                    entry_time = leg_ts
                leg_order_id = self._coerce_int(leg.get("order_id"), None)
                if leg_order_id is not None and leg_order_id not in entry_order_ids:
                    entry_order_ids.append(int(leg_order_id))

            entry_price = None
            if entry_weighted > 0.0 and matched_qty > 0:
                entry_price = entry_weighted / float(matched_qty)

            exit_price = None
            if float(group.get("weighted_exit", 0.0) or 0.0) > 0.0 and exit_qty > 0:
                exit_price = float(group.get("weighted_exit", 0.0) or 0.0) / float(exit_qty)

            pnl_dollars = float(group.get("pnl_dollars", 0.0) or 0.0)
            pnl_points = pnl_dollars / float(point_value * matched_qty) if point_value > 0.0 and matched_qty > 0 else None
            if pnl_points is None and entry_price is not None and exit_price is not None:
                if str(group.get("side") or "").upper() == "LONG":
                    pnl_points = float(exit_price - entry_price)
                else:
                    pnl_points = float(entry_price - exit_price)

            reconstructed.append(
                {
                    "source": "projectx_trade_history",
                    "strategy": None,
                    "sub_strategy": None,
                    "combo_key": None,
                    "side": str(group.get("side") or ""),
                    "size": int(matched_qty),
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "entry_time": entry_time,
                    "entry_order_id": entry_order_ids[0] if len(entry_order_ids) == 1 else None,
                    "entry_order_ids": entry_order_ids,
                    "entry_order_ids_ambiguous": len(entry_order_ids) > 1,
                    "exit_price": float(exit_price) if exit_price is not None else None,
                    "exit_time": group.get("exit_time"),
                    "order_id": group.get("close_order_id"),
                    "pnl_dollars": float(pnl_dollars),
                    "pnl_points": float(pnl_points) if pnl_points is not None else None,
                    "raw_close_rows": list(group.get("exit_rows") or []),
                    "raw_entry_rows": [dict(leg.get("row") or {}) for leg in entry_legs],
                }
            )

        reconstructed.sort(
            key=lambda row: row.get("exit_time") or datetime.datetime.min.replace(
                tzinfo=datetime.timezone.utc
            )
        )
        return reconstructed

    def get_trade_fill_summary(
        self,
        order_id: int,
        *,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        min_qty: Optional[int] = None,
        allow_rest_lookup: bool = True,
    ) -> Optional[Dict]:
        target_order_id = self._coerce_int(order_id, None)
        if target_order_id is None:
            return None

        now_et = datetime.datetime.now(self.et)
        if not isinstance(start_time, datetime.datetime):
            start_time = now_et - datetime.timedelta(hours=12)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=self.et)

        if not isinstance(end_time, datetime.datetime):
            end_time = now_et
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=self.et)

        required_qty = None
        if min_qty is not None:
            required_qty = max(1, int(min_qty))

        def _matching_rows(rows: List[Dict]) -> List[Dict]:
            matched: List[Dict] = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                if row.get("contractId") != self.contract_id:
                    continue
                if bool(row.get("voided", False)):
                    continue
                if self._coerce_int(row.get("orderId"), None) != target_order_id:
                    continue
                fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
                if fill_qty <= 0:
                    continue
                matched.append(dict(row))
            matched.sort(
                key=lambda row: self._parse_trade_timestamp(row) or datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                )
            )
            return matched

        matched_rows = _matching_rows(self._get_stream_trades(start_time, end_time))
        matched_qty = sum(abs(self._coerce_int(row.get("size"), 0) or 0) for row in matched_rows)
        need_rest_lookup = bool(allow_rest_lookup) and not matched_rows
        if required_qty is not None and matched_qty < required_qty:
            need_rest_lookup = bool(allow_rest_lookup)
        if need_rest_lookup:
            rest_rows = _matching_rows(self.search_trades(start_time, end_time))
            matched_rows = self._merge_trade_rows(matched_rows, rest_rows)
        if not matched_rows:
            return None

        filled_qty = 0
        weighted_price = 0.0
        latest_fill_time: Optional[datetime.datetime] = None
        for row in matched_rows:
            fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
            fill_price = self._coerce_float(row.get("price"), None)
            if fill_qty > 0:
                filled_qty += fill_qty
                if fill_price is not None:
                    weighted_price += float(fill_price) * float(fill_qty)
            row_ts = self._parse_trade_timestamp(row)
            if row_ts is not None:
                latest_fill_time = row_ts

        avg_price = None
        if filled_qty > 0 and weighted_price > 0.0:
            avg_price = weighted_price / float(filled_qty)

        return {
            "order_id": target_order_id,
            "filled_qty": int(filled_qty),
            "avg_price": float(avg_price) if avg_price is not None else None,
            "latest_fill_time": latest_fill_time,
            "matched_rows": len(matched_rows),
            "complete": required_qty is None or filled_qty >= required_qty,
        }

    def reconcile_trade_close(
        self,
        active_trade: Dict,
        *,
        exit_time: Optional[datetime.datetime] = None,
        fallback_exit_price: Optional[float] = None,
        close_order_id: Optional[int] = None,
        point_value: float = 5.0,
    ) -> Optional[Dict]:
        if not isinstance(active_trade, dict):
            return None
        trade_size = self._coerce_int(active_trade.get("size"), 0) or 0
        if trade_size <= 0:
            return None

        risk_cfg = CONFIG.get("RISK") or {}
        base_round_turn_fee = max(
            0.0,
            float(self._coerce_float(risk_cfg.get("FEES_PER_SIDE"), 0.37) or 0.37) * 2.0,
        )
        topstep_round_turn_commission = max(
            0.0,
            float(
                self._coerce_float(
                    risk_cfg.get("TOPSTEP_COMMISSION_ROUND_TURN_PER_CONTRACT"),
                    0.50,
                )
                or 0.50
            ),
        )
        fallback_round_turn_fee = base_round_turn_fee + topstep_round_turn_commission

        entry_time = active_trade.get("entry_time")
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.datetime.fromisoformat(entry_time)
            except Exception:
                entry_time = None
        if not isinstance(entry_time, datetime.datetime):
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            entry_time = now_utc - datetime.timedelta(minutes=30)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=self.et)

        effective_exit_time = exit_time or datetime.datetime.now(self.et)
        if effective_exit_time.tzinfo is None:
            effective_exit_time = effective_exit_time.replace(tzinfo=self.et)

        lookback_start = entry_time - datetime.timedelta(minutes=2)
        lookahead_end = effective_exit_time + datetime.timedelta(minutes=2)
        recent_trades = self._get_stream_trades(lookback_start, lookahead_end)
        need_rest_lookup = not recent_trades
        if not need_rest_lookup and close_order_id is not None:
            need_rest_lookup = not any(
                self._coerce_int(row.get("orderId"), None) == int(close_order_id)
                for row in recent_trades
            )
        if need_rest_lookup:
            rest_trades = self.search_trades(lookback_start, lookahead_end)
            recent_trades = self._merge_trade_rows(recent_trades, rest_trades)
        if not recent_trades:
            return None

        close_side_code = 1 if str(active_trade.get("side", "")).upper() == "LONG" else 0
        entry_side_code = 0 if close_side_code == 1 else 1
        entry_order_id = self._coerce_int(active_trade.get("entry_order_id"), None)

        def _trade_side_code(row: Dict) -> Optional[int]:
            return self._coerce_int(row.get("side"), None)

        def _trade_order_id(row: Dict) -> Optional[int]:
            return self._coerce_int(row.get("orderId"), None)

        def _build_candidates(rows: List[Dict]) -> List[Dict]:
            built: List[Dict] = []
            for row in rows:
                if _trade_side_code(row) != close_side_code:
                    continue
                row_ts = self._parse_trade_timestamp(row)
                if row_ts is None:
                    continue
                if row_ts < entry_time.astimezone(datetime.timezone.utc):
                    continue
                if entry_order_id is not None and _trade_order_id(row) == entry_order_id:
                    continue
                built.append(row)
            return built

        def _build_entry_candidates(rows: List[Dict]) -> List[Dict]:
            built: List[Dict] = []
            for row in rows:
                if _trade_side_code(row) != entry_side_code:
                    continue
                row_ts = self._parse_trade_timestamp(row)
                if row_ts is None:
                    continue
                if row_ts < lookback_start.astimezone(datetime.timezone.utc):
                    continue
                built.append(row)
            return built

        if entry_order_id is not None and not any(
            _trade_order_id(row) == entry_order_id for row in recent_trades
        ):
            rest_trades = self.search_trades(lookback_start, lookahead_end)
            recent_trades = self._merge_trade_rows(recent_trades, rest_trades)

        candidates = _build_candidates(recent_trades)
        candidate_qty = sum(abs(self._coerce_int(row.get("size"), 0) or 0) for row in candidates)
        if close_order_id is None and candidate_qty < trade_size and not need_rest_lookup:
            rest_trades = self.search_trades(lookback_start, lookahead_end)
            recent_trades = self._merge_trade_rows(recent_trades, rest_trades)
            candidates = _build_candidates(recent_trades)
        if not candidates:
            return None

        entry_candidates = _build_entry_candidates(recent_trades)
        matching_entry_rows: List[Dict] = []
        if entry_order_id is not None:
            matching_entry_rows = [
                row for row in entry_candidates
                if _trade_order_id(row) == entry_order_id
            ]
        if not matching_entry_rows:
            remaining = trade_size
            for row in sorted(
                entry_candidates,
                key=lambda item: self._parse_trade_timestamp(item) or datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                ),
            ):
                matching_entry_rows.append(row)
                fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
                remaining -= max(1, fill_qty)
                if remaining <= 0:
                    break

        matching_rows: List[Dict] = []
        source = "trade_search_recent"
        if close_order_id is not None:
            matching_rows = [
                row for row in candidates
                if _trade_order_id(row) == int(close_order_id)
            ]
            if matching_rows:
                source = "trade_search_exact_order"
        if not matching_rows:
            remaining = trade_size
            for row in sorted(
                candidates,
                key=lambda item: self._parse_trade_timestamp(item) or datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                ),
                reverse=True,
            ):
                matching_rows.append(row)
                fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
                remaining -= max(1, fill_qty)
                if remaining <= 0:
                    break
            matching_rows.sort(
                key=lambda item: self._parse_trade_timestamp(item) or datetime.datetime.min.replace(
                    tzinfo=datetime.timezone.utc
                )
            )

        total_qty = 0
        weighted_price = 0.0
        pnl_dollars_gross = 0.0
        pnl_found = False
        reported_fee_dollars = 0.0
        fee_found = False
        exit_ts: Optional[datetime.datetime] = None
        for row in matching_rows:
            fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
            fill_price = self._coerce_float(row.get("price"), None)
            if fill_qty > 0 and fill_price is not None:
                total_qty += fill_qty
                weighted_price += fill_price * fill_qty
            pnl_val = self._coerce_float(row.get("profitAndLoss"), None)
            if pnl_val is not None:
                pnl_dollars_gross += pnl_val
                pnl_found = True
            row_commissions = self._coerce_float(row.get("commissions"), None)
            row_fees = self._coerce_float(row.get("fees"), None)
            if row_commissions is not None or row_fees is not None:
                reported_fee_dollars += float(row_commissions or 0.0) + float(row_fees or 0.0)
                fee_found = True
            row_ts = self._parse_trade_timestamp(row)
            if row_ts is not None:
                exit_ts = row_ts

        if total_qty <= 0:
            total_qty = trade_size
        if total_qty <= 0:
            return None

        exit_price = None
        if weighted_price > 0.0:
            exit_price = weighted_price / float(total_qty)
        elif fallback_exit_price is not None:
            exit_price = float(fallback_exit_price)

        pnl_points = None
        if pnl_found:
            denom = float(point_value) * float(trade_size)
            if denom > 0.0:
                pnl_points = pnl_dollars_gross / denom
        if pnl_points is None and exit_price is not None:
            entry_price = self._coerce_float(active_trade.get("entry_price"), 0.0) or 0.0
            if str(active_trade.get("side", "")).upper() == "LONG":
                pnl_points = float(exit_price - entry_price)
            else:
                pnl_points = float(entry_price - exit_price)
            pnl_dollars_gross = float(pnl_points * float(point_value) * float(trade_size))

        if pnl_points is None or exit_price is None:
            return None

        pnl_fee_dollars = float(reported_fee_dollars) if fee_found else float(fallback_round_turn_fee * float(trade_size))
        pnl_dollars_net = float(pnl_dollars_gross - pnl_fee_dollars)

        entry_total_qty = 0
        entry_weighted_price = 0.0
        entry_ts: Optional[datetime.datetime] = None
        for row in matching_entry_rows:
            fill_qty = abs(self._coerce_int(row.get("size"), 0) or 0)
            fill_price = self._coerce_float(row.get("price"), None)
            if fill_qty > 0 and fill_price is not None:
                entry_total_qty += fill_qty
                entry_weighted_price += fill_price * fill_qty
            row_ts = self._parse_trade_timestamp(row)
            if row_ts is not None and entry_ts is None:
                entry_ts = row_ts

        entry_price = None
        if entry_weighted_price > 0.0 and entry_total_qty > 0:
            entry_price = entry_weighted_price / float(entry_total_qty)
        if entry_price is None:
            entry_price = self._coerce_float(
                active_trade.get("broker_entry_price"),
                self._coerce_float(active_trade.get("entry_price"), None),
            )

        return {
            "source": source,
            "entry_price": float(entry_price) if entry_price is not None else None,
            "entry_time": entry_ts,
            "exit_price": float(exit_price),
            "pnl_points": float(pnl_points),
            "pnl_dollars_gross": float(pnl_dollars_gross),
            "pnl_fee_dollars": float(pnl_fee_dollars),
            "pnl_dollars_net": float(pnl_dollars_net),
            "pnl_dollars": float(pnl_dollars_net),
            "exit_time": exit_ts,
            "entry_order_id": entry_order_id if entry_order_id is not None else self._coerce_int(
                matching_entry_rows[0].get("orderId"),
                None,
            ) if matching_entry_rows else None,
            "order_id": close_order_id if close_order_id is not None else self._coerce_int(
                matching_rows[-1].get("orderId"),
                None,
            ),
            "entry_matched_rows": len(matching_entry_rows),
            "matched_rows": len(matching_rows),
        }

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order by ID.
        Endpoint: POST /api/Order/cancel
        """
        if not self._check_general_rate_limit():
            return False

        url = f"{self.base_url}/api/Order/cancel"
        payload = {
            "accountId": self.account_id,  # Include accountId for API
            "orderId": order_id
        }

        try:
            resp = self.session.post(url, json=payload)
            self._track_general_request()
            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"Order {order_id} cancelled")
                    if self._active_stop_order_id == order_id:
                        self._active_stop_order_id = None
                    if self._active_target_order_id == order_id:
                        self._active_target_order_id = None
                    return True
                else:
                    err = data.get('errorMessage', 'Unknown error')
                    logging.warning(f"Order cancel rejected: {err}")
                    return False
            logging.warning(f"Order cancel failed: {resp.status_code} - {resp.text}")
            return False
        except Exception as e:
            logging.error(f"Order cancel error: {e}")
            return False

    def modify_order(self, order_id: int, stop_price: float = None, limit_price: float = None, size: int = None) -> bool:
        """
        Modify an existing order using the /api/Order/modify endpoint.

        Args:
            order_id: The order ID to modify
            stop_price: New stop price (for stop orders)
            limit_price: New limit price (for limit orders)
            size: New size (optional)

        Returns:
            True if modification successful
        """
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot modify order")
            return False

        url = f"{self.base_url}/api/Order/modify"

        payload = {
            "accountId": self.account_id,
            "orderId": order_id
        }

        # Only include fields that are being modified
        if size is not None:
            payload["size"] = size
        if limit_price is not None:
            payload["limitPrice"] = limit_price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        try:
            logging.info(f"MODIFYING ORDER {order_id}: stopPrice={stop_price}, limitPrice={limit_price}")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"Order {order_id} modified successfully")
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"Order modification rejected: {err}")
                    return False
            else:
                logging.error(f"Order modification failed: {resp.status_code} - {resp.text}")
                return False

        except Exception as e:
            logging.error(f"Order modification exception: {e}")
            return False

    def modify_stop_to_breakeven(
        self,
        stop_price: float,
        side: str,
        known_size: int = None,
        stop_order_id: int = None,
        current_stop_price: float = None
    ) -> bool:
        """
        Aggressively modify stop to break-even.
        Updates:
        1. Removed 'Skipping' logic - Forces update to ensure safety.
        2. Improved Search - Logs exactly what it finds.
        3. Robust Fallback - If modify fails, immediately cancels and places new stop.
        """
        # 1. Determine position size
        position_size = known_size if known_size is not None else 1
        if position_size == 0:
            logging.warning("No position size provided. Aborting stop modification.")
            return False

        # 2. Use the stop price directly
        be_price = round(stop_price * 4) / 4  # Tick alignment

        expected_side = 1 if side == 'LONG' else 0
        expected_size = position_size if known_size is not None else None

        # 3. Try direct modify if we have an ID
        target_stop_id = stop_order_id or self._active_stop_order_id
        if target_stop_id:
            logging.info(f"🔒 MOVING STOP: {side} -> {be_price:.2f} (Order ID: {target_stop_id})")
            if self.modify_order(target_stop_id, stop_price=be_price):
                logging.info(f"✅ STOP UPDATED to {be_price:.2f}")
                self._active_stop_order_id = self._coerce_int(target_stop_id, None)
                return True
            logging.warning(f"⚠️ Modify failed for {target_stop_id}. Attempting Cancel/Replace...")

        # 4. Find the best matching stop order for this trade
        orders = self._get_cached_orders(force_refresh=True)
        candidate = self._select_stop_order(
            orders,
            expected_side=expected_side,
            expected_price=current_stop_price,
            expected_size=expected_size,
            prefer_order_id=target_stop_id,
        )

        if candidate:
            candidate_id = candidate.get("orderId")
            current_stop_val = self._extract_order_stop_price(candidate) or 0
            logging.info(f"Matched stop order {candidate_id} @ {current_stop_val}")

            # Try modify against matched stop order
            if candidate_id and self.modify_order(candidate_id, stop_price=be_price):
                logging.info(f"✅ STOP UPDATED to {be_price:.2f}")
                self._active_stop_order_id = self._coerce_int(candidate_id, None)
                return True

            if candidate_id:
                logging.warning(f"⚠️ Modify failed for {candidate_id}. Attempting Cancel/Replace...")
                self.cancel_order(candidate_id)
                self._active_stop_order_id = None
                time.sleep(0.5)
                remaining = self._get_cached_orders(force_refresh=True)
                if any(o.get("orderId") == candidate_id for o in remaining):
                    logging.warning("Cancel failed; skipping new stop to avoid duplicates.")
                    return False
        else:
            logging.warning(
                "No matching stop order found; placing a replacement stop without touching other same-side brackets."
            )

        # 5. FALLBACK: Place New Stop Order
        logging.info(f"🔄 PLACING NEW STOP at {be_price:.2f}...")
        if not self._place_breakeven_stop(be_price, side, position_size):
            return False
        return True

    def _place_breakeven_stop(self, be_price: float, side: str, size: int) -> bool:
        """
        Internal method to place a new stop order at break-even price.

        Args:
            be_price: Break-even price for the stop
            side: 'LONG' or 'SHORT' (determines stop side)
            size: Position size

        Returns:
            True if stop successfully placed
        """
        if not self._check_general_rate_limit():
            logging.error("Rate limit reached, cannot place break-even stop")
            return False

        url = f"{self.base_url}/api/Order/place"

        # For break-even stop:
        # - If LONG position, we need a SELL stop (side=1)
        # - If SHORT position, we need a BUY stop (side=0)
        side_code = 1 if side == 'LONG' else 0

        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "clOrdId": str(uuid.uuid4()),  # Unique order ID for break-even stop
            "type": 4,  # Stop Market
            "side": side_code,
            "size": size,
            "stopPrice": be_price
        }

        try:
            logging.info(f"Placing break-even stop: {be_price:.2f} for {size} contracts ({side} position)")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"BREAK-EVEN STOP PLACED at {be_price:.2f}")
                    # Capture the new stop order ID
                    if 'orderId' in data:
                        self._active_stop_order_id = data['orderId']
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"Break-even stop rejected: {err}")
                    return False
            else:
                logging.error(f"Break-even stop failed: {resp.status_code} - {resp.text}")
                return False

        except Exception as e:
            logging.error(f"Break-even stop exception: {e}")
            return False

    def _try_bracket_modification(self, entry_price: float, side: str, size: int) -> bool:
        """Alternative method: try to modify stop via position bracket endpoint."""
        be_price = round(entry_price * 4) / 4

        # Try direct stop placement without cancelling first
        url = f"{self.base_url}/api/Order/place"
        side_code = 1 if side == 'LONG' else 0

        payload = {
            "accountId": self.account_id,
            "contractId": self.contract_id,
            "clOrdId": str(uuid.uuid4()),  # Unique order ID
            "type": 4,  # Stop Market
            "side": side_code,
            "size": size,
            "stopPrice": be_price
        }

        try:
            logging.info(f"BREAK-EVEN (alt): Placing stop at {be_price:.2f} for {size} contracts")
            resp = self.session.post(url, json=payload)
            self._track_general_request()

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success', False):
                    logging.info(f"Break-even stop placed at {be_price:.2f}")
                    return True
                else:
                    err = data.get('errorMessage', str(data))
                    logging.error(f"Break-even alt rejected: {err}")
            else:
                logging.error(f"Break-even alt failed: {resp.status_code}")
            return False
        except Exception as e:
            logging.error(f"Break-even alt exception: {e}")
            return False

    # ==========================================
    # ASYNC METHODS FOR ASYNCIO UPGRADE
    # ==========================================

    async def async_get_market_data(self, lookback_minutes: int = 20000, force_fetch: bool = False) -> pd.DataFrame:
        """Async wrapper for get_market_data() to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self.get_market_data,
            lookback_minutes=lookback_minutes,
            force_fetch=force_fetch,
        )

    async def async_close_and_reverse(
        self,
        new_signal: Dict,
        current_price: float,
        opposite_signal_count: int,
    ) -> Tuple[bool, int]:
        """Async wrapper for close_and_reverse() to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self.close_and_reverse,
            new_signal,
            current_price,
            opposite_signal_count,
        )

    async def async_place_order(self, signal: Dict, current_price: float):
        """Async wrapper for place_order() to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self.place_order,
            signal,
            current_price,
        )

    async def async_close_trade_leg(self, trade: Dict) -> bool:
        """Async wrapper for close_trade_leg() to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self.close_trade_leg,
            trade,
        )

    async def async_get_position(
        self,
        *,
        prefer_stream: bool = True,
        require_open_pnl: bool = False,
    ) -> Dict:
        """
        Async version of get_position() for use in independent async tasks.
        FIX: Returns cached local state if rate limited.

        Returns:
            Position dict with 'side', 'size', 'avg_price'
        """
        try:
            import aiohttp
        except Exception:
            self._warn_async_http_fallback_once()
            return await asyncio.to_thread(
                self.get_position,
                prefer_stream=prefer_stream,
                require_open_pnl=require_open_pnl,
            )

        stream_position = self._get_stream_position() if prefer_stream else None
        if stream_position is not None:
            stream_side = str(stream_position.get("side") or "").strip().upper()
            stream_open_pnl = self._coerce_float(stream_position.get("open_pnl"), None)
            if not require_open_pnl or stream_side not in {"LONG", "SHORT"} or stream_open_pnl is not None:
                self._local_position = stream_position.copy()
                return stream_position
        if self._auth_temporarily_unavailable():
            return stream_position if stream_position is not None else self._stale_position()

        # --- FIX START ---
        if not self._check_general_rate_limit():
            # In async, we might not want to log warnings every tick, but safety first
            return self._local_position
        # --- FIX END ---

        if self.account_id is None:
            return {'side': None, 'size': 0, 'avg_price': 0.0}

        payload = {"accountId": self.account_id}
        headers = {"Authorization": f"Bearer {self.token}"}
        best_position = None

        try:
            async with aiohttp.ClientSession() as session:
                search_endpoints = [
                    (f"{self.base_url}/api/Position/searchOpen", "post"),
                    (f"{self.base_url}/api/Position/search", "post"),
                    (f"{self.base_url}/api/Position", "get"),
                ]
                for url, method in search_endpoints:
                    if method == "post":
                        async with session.post(url, json=payload, headers=headers) as resp:
                            self._track_general_request()
                            if resp.status == 200:
                                data = await resp.json()
                                positions = data.get('positions', data) if isinstance(data, dict) else data
                                filtered = self._filter_positions_for_contract(
                                    positions if isinstance(positions, list) else []
                                )
                                if filtered:
                                    normalized = self._normalize_position(filtered[0])
                                    if best_position is None:
                                        best_position = normalized
                                    normalized_side = str(normalized.get("side") or "").strip().upper()
                                    normalized_open_pnl = self._coerce_float(normalized.get("open_pnl"), None)
                                    if not require_open_pnl or normalized_side not in {"LONG", "SHORT"} or normalized_open_pnl is not None:
                                        self._local_position = normalized.copy()
                                        return normalized
                                continue
                            if resp.status in (400, 404):
                                continue
                            if resp.status == 401:
                                logging.warning("Async position check returned 401; re-authenticating via API key")
                                await self._handle_unauthorized_response_async(
                                    "POST /api/Position/searchOpen|search (async)",
                                    restart_user_stream=True,
                                )
                                return stream_position if stream_position is not None else self._stale_position()
                            logging.warning(f"Async position check failed: {resp.status}")
                            return stream_position if stream_position is not None else self._stale_position()
                    else:
                        async with session.get(url, params=payload, headers=headers) as resp:
                            self._track_general_request()
                            if resp.status == 200:
                                data = await resp.json()
                                positions = data.get('positions', data) if isinstance(data, dict) else data
                                filtered = self._filter_positions_for_contract(
                                    positions if isinstance(positions, list) else []
                                )
                                if filtered:
                                    normalized = self._normalize_position(filtered[0])
                                    if best_position is None:
                                        best_position = normalized
                                    normalized_side = str(normalized.get("side") or "").strip().upper()
                                    normalized_open_pnl = self._coerce_float(normalized.get("open_pnl"), None)
                                    if not require_open_pnl or normalized_side not in {"LONG", "SHORT"} or normalized_open_pnl is not None:
                                        self._local_position = normalized.copy()
                                        return normalized
                                continue
                            if resp.status in (400, 404):
                                continue
                            if resp.status == 401:
                                logging.warning("Async position GET returned 401; re-authenticating via API key")
                                await self._handle_unauthorized_response_async(
                                    "GET /api/Position (async)",
                                    restart_user_stream=True,
                                )
                                return stream_position if stream_position is not None else self._stale_position()
                            logging.warning(f"Async position check failed: {resp.status}")
                            return stream_position if stream_position is not None else self._stale_position()
                if best_position is not None:
                    self._local_position = best_position.copy()
                    return best_position
                if stream_position is not None:
                    self._local_position = stream_position.copy()
                    return stream_position
                self._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
                return {'side': None, 'size': 0, 'avg_price': 0.0}

        except Exception as e:
            logging.error(f"Async position check error: {e}")
            return stream_position if stream_position is not None else self._stale_position()

    async def async_validate_session(self) -> bool:
        """
        Async version of validate_session() for heartbeat task.

        Returns:
            True if session is valid, False otherwise
        """
        try:
            import aiohttp
        except Exception:
            self._warn_async_http_fallback_once()
            return await asyncio.to_thread(self.validate_session)

        if self._auth_temporarily_unavailable():
            return False
        if not self._check_general_rate_limit():
            return self.token is not None

        url = f"{self.base_url}/api/Auth/validate"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as resp:
                    self._track_general_request()

                    if resp.status == 200:
                        data = await resp.json()
                        if 'newToken' in data:
                            self.token = data['newToken']
                            self.token_expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
                            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                            logging.info("Session token refreshed (async)")
                        return True
                    if resp.status == 401:
                        logging.warning("Async session validation failed: 401")
                        return await self._handle_unauthorized_response_async(
                            "POST /api/Auth/validate (async)",
                            restart_user_stream=True,
                        )
                    else:
                        logging.warning(f"Async session validation failed: {resp.status}")
                        return False

        except Exception as e:
            logging.error(f"Async session validation error: {e}")
            return False

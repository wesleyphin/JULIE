"""
Best-effort OANDA practice mirror for plotting JULIE executions in TradingView.

The mirror is intentionally decoupled from Topstep execution. OANDA requests run
on a tiny background executor so demo-chart updates never block the main order
path.
"""

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Dict, Optional

import requests

from event_logger import event_logger


class OandaMirrorBridge:
    """Mirrors Topstep fills to a tiny OANDA practice position for chart ink."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        self.enabled = self._coerce_bool(cfg.get("enabled"), default=False)
        self.base_url = str(
            cfg.get("base_url") or "https://api-fxpractice.oanda.com"
        ).rstrip("/")
        self.account_id = str(cfg.get("account_id") or "").strip()
        self.api_key = str(cfg.get("api_key") or "").strip()
        self.timeout_sec = max(
            0.5,
            self._coerce_float(cfg.get("timeout_sec"), 3.0) or 3.0,
        )
        self.entry_units = max(1, self._coerce_int(cfg.get("entry_units"), 1) or 1)
        raw_symbol_map = cfg.get("symbol_map") or {}
        self.symbol_map = {
            str(root).strip().upper(): str(instrument).strip()
            for root, instrument in raw_symbol_map.items()
            if str(root).strip() and str(instrument).strip()
        }
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="oanda-mirror",
        )
        self._thread_local = threading.local()
        self._disabled_warning_emitted = False
        self._instrument_tradeable_cache: Dict[str, bool] = {}

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_json(response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "JULIE-OANDA-Mirror/1.0",
                }
            )
            self._thread_local.session = session
        return session

    def _ready(self) -> bool:
        if not self.enabled:
            return False
        if self.account_id and self.api_key:
            return True
        if not self._disabled_warning_emitted:
            self._disabled_warning_emitted = True
            logging.warning(
                "OANDA mirror enabled but credentials are incomplete. "
                "Add OANDA_API_KEY and OANDA_ACCOUNT_ID to config_secrets.py."
            )
        return False

    def _resolve_instrument(self, contract_root: Optional[str]) -> Optional[str]:
        normalized = str(contract_root or "").strip().upper()
        if not normalized:
            logging.warning("OANDA mirror skipped: empty contract root")
            event_logger.log_error(
                "OANDA_MIRROR_CONFIG",
                "OANDA mirror skipped because contract_root was empty.",
            )
            return None
        if normalized in self.symbol_map:
            return self.symbol_map[normalized]
        for root, instrument in self.symbol_map.items():
            if (
                normalized == root
                or normalized.startswith(f"{root}.")
                or normalized.endswith(f".{root}")
                or f".{root}." in normalized
            ):
                return instrument
        logging.warning(
            "OANDA mirror skipped: no symbol mapping configured for %s",
            contract_root,
        )
        event_logger.log_error(
            "OANDA_MIRROR_MAPPING",
            f"No OANDA symbol mapping configured for contract root '{contract_root}'.",
        )
        return None

    def _instrument_is_tradeable(self, instrument: str) -> Optional[bool]:
        cached = self._instrument_tradeable_cache.get(instrument)
        if cached is not None:
            return cached

        url = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
        try:
            response = self._get_session().get(
                url,
                params={"instruments": instrument},
                timeout=self.timeout_sec,
            )
            data = self._safe_json(response)
            if response.status_code == 200:
                instruments = data.get("instruments") or []
                tradeable = any(
                    isinstance(item, dict) and item.get("name") == instrument
                    for item in instruments
                )
                self._instrument_tradeable_cache[instrument] = tradeable
                if not tradeable:
                    message = (
                        f"OANDA mirror instrument {instrument} is not tradeable on account "
                        f"{self.account_id}."
                    )
                    logging.warning(message)
                    event_logger.log_error("OANDA_MIRROR_INSTRUMENT", message)
                return tradeable

            if response.status_code in {400, 404}:
                message = data.get("errorMessage") or f"HTTP {response.status_code}"
                logging.warning(
                    "OANDA mirror instrument check failed for %s: %s",
                    instrument,
                    message,
                )
                event_logger.log_error(
                    "OANDA_MIRROR_INSTRUMENT",
                    f"{instrument} is not tradeable on account {self.account_id}: {message}",
                )
                self._instrument_tradeable_cache[instrument] = False
                return False
        except requests.RequestException as exc:
            logging.warning(
                "OANDA mirror instrument validation request failed for %s: %s",
                instrument,
                exc,
            )
            event_logger.log_error(
                "OANDA_MIRROR_VALIDATION",
                f"Could not validate OANDA instrument {instrument}: {exc}",
            )
            return None
        except Exception as exc:
            logging.warning(
                "OANDA mirror instrument validation error for %s: %s",
                instrument,
                exc,
            )
            event_logger.log_error(
                "OANDA_MIRROR_VALIDATION",
                f"Unexpected validation error for OANDA instrument {instrument}: {exc}",
            )
            return None

        return None

    def _submit(self, fn, *args, **kwargs) -> Optional[concurrent.futures.Future]:
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except RuntimeError as exc:
            logging.warning("OANDA mirror scheduling failed: %s", exc)
            return None

        def _log_future_failure(done_future: concurrent.futures.Future) -> None:
            try:
                done_future.result()
            except Exception as exc:
                logging.warning("OANDA mirror background task crashed: %s", exc)

        future.add_done_callback(_log_future_failure)
        return future

    def submit_entry(
        self,
        *,
        contract_root: Optional[str],
        side: str,
        source_order_id: Optional[str] = None,
    ) -> Optional[concurrent.futures.Future]:
        if not self._ready():
            return None
        instrument = self._resolve_instrument(contract_root)
        if instrument is None:
            return None
        return self._submit(
            self._mirror_entry_sync,
            instrument,
            side,
            source_order_id=source_order_id,
        )

    def submit_flatten(
        self,
        *,
        contract_root: Optional[str],
        side: Optional[str],
        reason: str = "",
    ) -> Optional[concurrent.futures.Future]:
        if not self._ready():
            return None
        instrument = self._resolve_instrument(contract_root)
        if instrument is None:
            return None
        return self._submit(
            self._mirror_flatten_sync,
            instrument,
            side,
            reason=reason,
        )

    async def async_mirror_entry(
        self,
        *,
        contract_root: Optional[str],
        side: str,
        source_order_id: Optional[str] = None,
    ) -> bool:
        future = self.submit_entry(
            contract_root=contract_root,
            side=side,
            source_order_id=source_order_id,
        )
        if future is None:
            return False
        return await asyncio.wrap_future(future)

    async def async_mirror_flatten(
        self,
        *,
        contract_root: Optional[str],
        side: Optional[str],
        reason: str = "",
    ) -> bool:
        future = self.submit_flatten(
            contract_root=contract_root,
            side=side,
            reason=reason,
        )
        if future is None:
            return False
        return await asyncio.wrap_future(future)

    def _mirror_entry_sync(
        self,
        instrument: str,
        side: str,
        *,
        source_order_id: Optional[str] = None,
    ) -> bool:
        side_name = str(side or "").strip().upper()
        if side_name not in {"LONG", "SHORT"}:
            logging.warning("OANDA mirror skipped: unsupported side %s", side)
            event_logger.log_error(
                "OANDA_MIRROR_SIDE",
                f"OANDA mirror skipped unsupported side '{side}'.",
            )
            return False

        units = self.entry_units if side_name == "LONG" else -self.entry_units
        tradeable = self._instrument_is_tradeable(instrument)
        if tradeable is False:
            return False
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        payload = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if source_order_id:
            payload["order"]["clientExtensions"] = {
                "tag": "topstep-mirror",
                "comment": str(source_order_id)[:64],
            }

        try:
            response = self._get_session().post(
                url,
                json=payload,
                timeout=self.timeout_sec,
            )
            data = self._safe_json(response)
            if response.status_code not in {200, 201}:
                logging.warning(
                    "OANDA mirror entry failed for %s %s: HTTP %s %s",
                    side_name,
                    instrument,
                    response.status_code,
                    (data.get("errorMessage") or response.text[:250] or "").strip(),
                )
                event_logger.log_error(
                    "OANDA_MIRROR_ENTRY",
                    f"OANDA entry failed for {side_name} {instrument}: HTTP {response.status_code} "
                    f"{(data.get('errorMessage') or response.text[:250] or '').strip()}",
                )
                return False
            reject = data.get("orderRejectTransaction")
            if isinstance(reject, dict):
                reason = reject.get("rejectReason") or data.get("errorMessage") or reject
                logging.warning(
                    "OANDA mirror entry rejected for %s %s: %s",
                    side_name,
                    instrument,
                    reason,
                )
                event_logger.log_error(
                    "OANDA_MIRROR_ENTRY",
                    f"OANDA entry rejected for {side_name} {instrument}: {reason}",
                )
                return False
            fill_txn = data.get("orderFillTransaction") or {}
            logging.info(
                "OANDA mirror entry placed: %s %s unit(s) on %s | tx=%s",
                side_name,
                abs(units),
                instrument,
                fill_txn.get("id") or data.get("lastTransactionID"),
            )
            event_logger.log_system_event(
                "OANDA_MIRROR",
                f"OANDA mirror entry placed on {instrument}",
                {
                    "side": side_name,
                    "units": abs(units),
                    "instrument": instrument,
                },
            )
            return True
        except requests.RequestException as exc:
            logging.warning("OANDA mirror entry request failed for %s: %s", instrument, exc)
            event_logger.log_error(
                "OANDA_MIRROR_ENTRY",
                f"OANDA entry request failed for {instrument}: {exc}",
            )
            return False
        except Exception as exc:
            logging.warning("OANDA mirror entry error for %s: %s", instrument, exc)
            event_logger.log_error(
                "OANDA_MIRROR_ENTRY",
                f"Unexpected OANDA entry error for {instrument}: {exc}",
            )
            return False

    def _mirror_flatten_sync(
        self,
        instrument: str,
        side: Optional[str],
        *,
        reason: str = "",
    ) -> bool:
        side_name = str(side or "").strip().upper()
        if side_name == "LONG":
            payload = {"longUnits": "ALL"}
        elif side_name == "SHORT":
            payload = {"shortUnits": "ALL"}
        else:
            payload = {"longUnits": "ALL", "shortUnits": "ALL"}

        url = (
            f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"
        )

        try:
            response = self._get_session().put(
                url,
                json=payload,
                timeout=self.timeout_sec,
            )
            data = self._safe_json(response)
            if response.status_code == 404:
                logging.info(
                    "OANDA mirror flatten skipped for %s: already flat or unavailable",
                    instrument,
                )
                return True
            if response.status_code != 200:
                logging.warning(
                    "OANDA mirror flatten failed for %s: HTTP %s %s",
                    instrument,
                    response.status_code,
                    (data.get("errorMessage") or response.text[:250] or "").strip(),
                )
                event_logger.log_error(
                    "OANDA_MIRROR_FLATTEN",
                    f"OANDA flatten failed for {instrument}: HTTP {response.status_code} "
                    f"{(data.get('errorMessage') or response.text[:250] or '').strip()}",
                )
                return False

            reject = data.get("longOrderRejectTransaction") or data.get(
                "shortOrderRejectTransaction"
            )
            if isinstance(reject, dict):
                message = (
                    reject.get("rejectReason")
                    or data.get("errorMessage")
                    or str(reject)
                )
                if "does not exist" in message.lower():
                    logging.info(
                        "OANDA mirror flatten skipped for %s: %s",
                        instrument,
                        message,
                    )
                    return True
                logging.warning(
                    "OANDA mirror flatten rejected for %s: %s",
                    instrument,
                    message,
                )
                event_logger.log_error(
                    "OANDA_MIRROR_FLATTEN",
                    f"OANDA flatten rejected for {instrument}: {message}",
                )
                return False

            logging.info(
                "OANDA mirror flattened %s on %s | reason=%s | tx=%s",
                side_name or "position",
                instrument,
                reason or "close",
                data.get("lastTransactionID"),
            )
            event_logger.log_system_event(
                "OANDA_MIRROR",
                f"OANDA mirror flattened {instrument}",
                {
                    "side": side_name or "position",
                    "instrument": instrument,
                    "reason": reason or "close",
                },
            )
            return True
        except requests.RequestException as exc:
            logging.warning("OANDA mirror flatten request failed for %s: %s", instrument, exc)
            event_logger.log_error(
                "OANDA_MIRROR_FLATTEN",
                f"OANDA flatten request failed for {instrument}: {exc}",
            )
            return False
        except Exception as exc:
            logging.warning("OANDA mirror flatten error for %s: %s", instrument, exc)
            event_logger.log_error(
                "OANDA_MIRROR_FLATTEN",
                f"Unexpected OANDA flatten error for {instrument}: {exc}",
            )
            return False

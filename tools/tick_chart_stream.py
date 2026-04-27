"""Tick chart stream — subscribes to ProjectX SignalR ContractTrades and
aggregates ticks into 15-second OHLC bars for the live dashboard chart.

Runs in a background daemon thread spun up by julie001 at startup. Failures
inside the thread NEVER crash the bot — the chart degrades gracefully.

Usage from julie001.py:
    from tools.tick_chart_stream import start_tick_chart_stream, get_ohlc_bars
    start_tick_chart_stream(jwt_token=client.token, contract_id=client.contract_id)
    # then in build_persisted_state:
    "price_history_ohlc": get_ohlc_bars(),
"""

from __future__ import annotations

import datetime
import logging
import threading
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
BUCKET_SECONDS = 15
MAX_BARS = 240  # rolling 1-hour window of completed bars


class OHLC15sAggregator:
    """Thread-safe 15-second OHLCV aggregator.
    Receives individual ticks; flushes bars on bucket-boundary cross.
    Maintains rolling buffer of last MAX_BARS completed bars + current
    in-progress bar (always returned at the tail of get_bars())."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._completed: List[Dict[str, Any]] = []
        self._cur_start: Optional[datetime.datetime] = None
        self._cur_o: Optional[float] = None
        self._cur_h: Optional[float] = None
        self._cur_l: Optional[float] = None
        self._cur_c: Optional[float] = None
        self._cur_v: float = 0.0

    @staticmethod
    def _bucket_start(ts: datetime.datetime) -> datetime.datetime:
        sec = (ts.second // BUCKET_SECONDS) * BUCKET_SECONDS
        return ts.replace(second=sec, microsecond=0)

    def add_tick(self, price: float, volume: float, ts: datetime.datetime) -> None:
        if price is None or not isinstance(price, (int, float)):
            return
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=NY_TZ)
        bucket = self._bucket_start(ts)
        with self._lock:
            if self._cur_start is None:
                self._cur_start = bucket
                self._cur_o = price
                self._cur_h = price
                self._cur_l = price
                self._cur_c = price
                self._cur_v = float(volume or 0.0)
                return
            if bucket != self._cur_start:
                # Flush current bucket; may need to insert empty buckets if
                # there were gaps (ticks paused). For chart simplicity we
                # only emit the bucket we collected and start a new one.
                self._completed.append({
                    "t": self._cur_start.isoformat(),
                    "o": float(self._cur_o or 0.0),
                    "h": float(self._cur_h or 0.0),
                    "l": float(self._cur_l or 0.0),
                    "c": float(self._cur_c or 0.0),
                    "v": float(self._cur_v),
                })
                if len(self._completed) > MAX_BARS:
                    self._completed = self._completed[-MAX_BARS:]
                self._cur_start = bucket
                self._cur_o = price
                self._cur_h = price
                self._cur_l = price
                self._cur_c = price
                self._cur_v = float(volume or 0.0)
                return
            # Same bucket — extend
            if price > (self._cur_h or price):
                self._cur_h = price
            if price < (self._cur_l or price):
                self._cur_l = price
            self._cur_c = price
            self._cur_v += float(volume or 0.0)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = list(self._completed)
            if self._cur_start is not None:
                out.append({
                    "t": self._cur_start.isoformat(),
                    "o": float(self._cur_o or 0.0),
                    "h": float(self._cur_h or 0.0),
                    "l": float(self._cur_l or 0.0),
                    "c": float(self._cur_c or 0.0),
                    "v": float(self._cur_v),
                })
            return out


_AGGREGATOR: Optional[OHLC15sAggregator] = None
_THREAD: Optional[threading.Thread] = None
_STARTED = False


def get_ohlc_bars() -> Optional[List[Dict[str, Any]]]:
    """Return current OHLC bar snapshot, or None if stream not started yet."""
    if _AGGREGATOR is None:
        return None
    try:
        return _AGGREGATOR.snapshot()
    except Exception:
        return None


def get_status() -> Dict[str, Any]:
    """Return debug status (for surfacing in dashboard pipeline panel)."""
    return {
        "started": _STARTED,
        "alive": (_THREAD is not None and _THREAD.is_alive()),
        "bars": len(_AGGREGATOR._completed) if _AGGREGATOR is not None else 0,
    }


def _parse_tick(payload: Any) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a single tick from ProjectX GatewayTrade payload.
    ProjectX tick events typically come as either a single dict or a list
    of dicts; field names are not 100% standardized across providers."""
    if payload is None:
        return None
    if isinstance(payload, list):
        # Some servers send [{...}, {...}]
        return None  # caller should iterate
    if not isinstance(payload, dict):
        return None
    price = payload.get("price")
    if price is None:
        price = payload.get("p")
    if price is None:
        price = payload.get("Price")
    volume = payload.get("volume")
    if volume is None:
        volume = payload.get("v")
    if volume is None:
        volume = payload.get("Volume")
    ts_str = payload.get("timestamp") or payload.get("t") or payload.get("Timestamp")
    if price is None or ts_str is None:
        return None
    try:
        import pandas as pd
        ts = pd.to_datetime(ts_str, utc=True).tz_convert(NY_TZ).to_pydatetime()
        return {"price": float(price), "volume": float(volume or 0.0), "ts": ts}
    except Exception:
        return None


def _tick_loop(jwt_token: str, contract_id: str, hub_url: str) -> None:
    """Run the SignalR tick subscription forever in this thread.
    All exceptions are swallowed and logged; the thread will retry the
    connection automatically via signalrcore_async's reconnect policy."""
    global _AGGREGATOR
    try:
        import asyncio
        from signalrcore_async.hub_connection_builder import HubConnectionBuilder
    except Exception as e:
        logging.warning("[TICK_STREAM] signalrcore_async import failed: %s", e)
        return

    if _AGGREGATOR is None:
        _AGGREGATOR = OHLC15sAggregator()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def on_trade(args: Any) -> None:
        try:
            # SignalR delivers args as a list; the tick payload is args[0]
            payload = args[0] if isinstance(args, (list, tuple)) and args else args
            ticks: List[Any] = payload if isinstance(payload, list) else [payload]
            for raw in ticks:
                tick = _parse_tick(raw)
                if tick is not None:
                    _AGGREGATOR.add_tick(tick["price"], tick["volume"], tick["ts"])
        except Exception as e:
            logging.debug("[TICK_STREAM] tick parse error: %s", e)

    async def run() -> None:
        try:
            connection = (
                HubConnectionBuilder()
                .with_url(
                    hub_url,
                    options={
                        "access_token_factory": lambda: jwt_token,
                        "headers": {"Authorization": f"Bearer {jwt_token}"},
                    },
                )
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 10,
                })
                .build()
            )
            connection.on("GatewayTrade", on_trade)
            connection.on("ContractTrade", on_trade)  # fallback name
            await connection.start()
            logging.info("[TICK_STREAM] connected to market hub for %s", contract_id)
            try:
                await connection.invoke("SubscribeContractTrades", [contract_id])
            except Exception as e:
                logging.warning("[TICK_STREAM] SubscribeContractTrades failed: %s", e)
                return
            while True:
                await asyncio.sleep(60)
        except Exception as e:
            logging.warning("[TICK_STREAM] connection error: %s", e)

    try:
        loop.run_until_complete(run())
    except Exception as e:
        logging.warning("[TICK_STREAM] loop error: %s", e)
    finally:
        try:
            loop.close()
        except Exception:
            pass


def start_tick_chart_stream(jwt_token: str, contract_id: str, hub_url: Optional[str] = None) -> None:
    """Spin up the tick stream in a daemon thread. Idempotent — safe to call
    multiple times; only the first call actually starts the thread."""
    global _THREAD, _STARTED, _AGGREGATOR
    if _STARTED:
        return
    if not jwt_token or not contract_id:
        logging.warning("[TICK_STREAM] missing token or contract_id; skip")
        return
    if hub_url is None:
        try:
            from config import CONFIG
            hub_url = CONFIG.get("RTC_MARKET_HUB", "https://rtc.topstepx.com/hubs/market")
        except Exception:
            hub_url = "https://rtc.topstepx.com/hubs/market"
    _AGGREGATOR = OHLC15sAggregator()
    _THREAD = threading.Thread(
        target=_tick_loop,
        args=(jwt_token, contract_id, hub_url),
        daemon=True,
        name="tick_chart_stream",
    )
    _THREAD.start()
    _STARTED = True
    logging.info("[TICK_STREAM] daemon thread started for contract=%s hub=%s", contract_id, hub_url)

"""
Async WebSocket Market Data Stream

Provides real-time market data streaming using SignalR/WebSockets.
Replaces the polling REST API with instant price updates.
"""
import asyncio
import logging
import pandas as pd
import datetime
from typing import Optional, Callable, Dict, Any
from zoneinfo import ZoneInfo
from signalrcore_async.hub_connection_builder import HubConnectionBuilder

from config import CONFIG


class AsyncMarketStream:
    """
    Real-time market data streaming using SignalR WebSockets.

    Benefits over polling:
    - Instant price updates (no 2-second delay)
    - Reduced API load and rate limit pressure
    - Independent from strategy logic execution
    """

    def __init__(self, contract_id: str, on_bar_update: Optional[Callable] = None):
        """
        Initialize WebSocket stream.

        Args:
            contract_id: Contract ID to subscribe to (e.g., "CON.F.US.MES.Z25")
            on_bar_update: Callback function called when new bar received
        """
        self.contract_id = contract_id
        self.on_bar_update = on_bar_update
        self.hub_url = CONFIG['RTC_MARKET_HUB']
        self.et = ZoneInfo('America/New_York')

        # Connection state
        self.connection = None
        self.is_connected = False
        self.reconnect_delay = 5  # seconds

        # Latest bar cache
        self.latest_bar: Optional[Dict[str, Any]] = None
        self.latest_bar_time: Optional[datetime.datetime] = None

        logging.info(f"🔌 AsyncMarketStream initialized for {contract_id}")

    async def start(self, token: str):
        """
        Start WebSocket connection and subscribe to market data.

        Args:
            token: JWT authentication token
        """
        try:
            # Build SignalR connection
            self.connection = (
                HubConnectionBuilder()
                .with_url(
                    self.hub_url,
                    options={
                        "access_token_factory": lambda: token,
                        "headers": {"Authorization": f"Bearer {token}"}
                    }
                )
                .with_automatic_reconnect({
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 5
                })
                .build()
            )

            # Register event handlers
            self.connection.on_open(self._on_connect)
            self.connection.on_close(self._on_disconnect)
            self.connection.on_error(self._on_error)

            # Register market data handler
            self.connection.on("BarUpdate", self._on_bar_received)

            # Start connection
            await self.connection.start()

            logging.info("✅ WebSocket connected to market hub")

        except Exception as e:
            logging.error(f"❌ WebSocket connection failed: {e}")
            raise

    async def _on_connect(self):
        """Called when WebSocket connects."""
        self.is_connected = True
        logging.info(f"🔌 Connected to market stream for {self.contract_id}")

        # Subscribe to contract bars
        try:
            await self.connection.invoke("SubscribeToBars", self.contract_id)
            logging.info(f"📊 Subscribed to bars for {self.contract_id}")
        except Exception as e:
            logging.error(f"❌ Failed to subscribe to bars: {e}")

    def _on_disconnect(self):
        """Called when WebSocket disconnects."""
        self.is_connected = False
        logging.warning("⚠️ Disconnected from market stream")

    def _on_error(self, error):
        """Called when WebSocket error occurs."""
        logging.error(f"❌ WebSocket error: {error}")

    async def _on_bar_received(self, bar_data: Dict[str, Any]):
        """
        Called when new bar received from WebSocket.

        Args:
            bar_data: Raw bar data from SignalR
                Expected format: {"t": "2025-12-25T10:30:00Z", "o": 5950.25, "h": 5952.0, "l": 5949.5, "c": 5951.75, "v": 1234}
        """
        try:
            # Parse bar
            bar = {
                'ts': pd.to_datetime(bar_data['t'], format='ISO8601').tz_convert(self.et),
                'open': bar_data['o'],
                'high': bar_data['h'],
                'low': bar_data['l'],
                'close': bar_data['c'],
                'volume': bar_data['v']
            }

            self.latest_bar = bar
            self.latest_bar_time = bar['ts']

            # Log only every 10th bar to avoid spam
            if not hasattr(self, '_bar_count'):
                self._bar_count = 0
            self._bar_count += 1

            if self._bar_count % 10 == 0:
                logging.debug(f"📊 Bar received: {bar['ts']} | Close: {bar['close']:.2f}")

            # Call user callback if provided
            if self.on_bar_update:
                await self.on_bar_update(bar)

        except Exception as e:
            logging.error(f"❌ Error processing bar: {e}")

    async def stop(self):
        """Stop WebSocket connection."""
        if self.connection:
            try:
                await self.connection.stop()
                logging.info("🔌 WebSocket connection closed")
            except Exception as e:
                logging.error(f"❌ Error closing WebSocket: {e}")

    def get_latest_bar(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent bar received.

        Returns:
            Latest bar dict or None if no bars received yet
        """
        return self.latest_bar

    def is_stream_active(self) -> bool:
        """
        Check if stream is actively receiving data.

        Returns:
            True if connected and recently received data
        """
        if not self.is_connected:
            return False

        if self.latest_bar_time is None:
            return False

        # Consider stream active if we received data in last 60 seconds
        now = datetime.datetime.now(self.et)
        age = (now - self.latest_bar_time).total_seconds()
        return age < 60


class AsyncMarketDataManager:
    """
    Manages multiple market data streams (MES, MNQ, etc.) concurrently.

    Aggregates bars from all streams and maintains a unified DataFrame.
    """

    def __init__(self):
        self.streams: Dict[str, AsyncMarketStream] = {}
        self.master_dfs: Dict[str, pd.DataFrame] = {}
        self.et = ZoneInfo('America/New_York')

    async def add_stream(self, name: str, contract_id: str, token: str):
        """
        Add and start a new market data stream.

        Args:
            name: Stream identifier (e.g., "MES", "MNQ")
            contract_id: Contract ID to stream
            token: JWT token for auth
        """
        async def on_bar_update(bar: Dict[str, Any]):
            """Callback to update master DataFrame when bar received."""
            await self._update_master_df(name, bar)

        stream = AsyncMarketStream(contract_id, on_bar_update)
        await stream.start(token)

        self.streams[name] = stream
        self.master_dfs[name] = pd.DataFrame()

        logging.info(f"✅ Started stream: {name} ({contract_id})")

    async def _update_master_df(self, name: str, bar: Dict[str, Any]):
        """
        Update master DataFrame with new bar.

        Args:
            name: Stream name
            bar: New bar data
        """
        try:
            # Convert bar to DataFrame row
            bar_df = pd.DataFrame([bar])
            bar_df = bar_df.set_index('ts')

            # Append to master (or create if empty)
            if self.master_dfs[name].empty:
                self.master_dfs[name] = bar_df
            else:
                # Concat and remove duplicates (keep last)
                self.master_dfs[name] = pd.concat([self.master_dfs[name], bar_df])
                self.master_dfs[name] = self.master_dfs[name][~self.master_dfs[name].index.duplicated(keep='last')]

                # Keep only last 20,000 bars (matching REST API limit)
                if len(self.master_dfs[name]) > 20000:
                    self.master_dfs[name] = self.master_dfs[name].iloc[-20000:]

        except Exception as e:
            logging.error(f"❌ Error updating master DF for {name}: {e}")

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """
        Get current master DataFrame for a stream.

        Args:
            name: Stream name

        Returns:
            DataFrame with full bar history
        """
        return self.master_dfs.get(name, pd.DataFrame())

    async def stop_all(self):
        """Stop all market data streams."""
        for name, stream in self.streams.items():
            await stream.stop()
            logging.info(f"🔌 Stopped stream: {name}")

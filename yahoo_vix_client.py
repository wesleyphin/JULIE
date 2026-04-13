import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None

class YahooVIXClient:
    """
    A virtual client that mimics ProjectXClient but fetches VIX data from Yahoo Finance.
    Used because Topstep/Rithmic often excludes CBOE data.
    """
    _dependency_warned = False

    def __init__(self, contract_root="^VIX", target_symbol="^VIX"):
        self.contract_root = contract_root
        self.target_symbol = self._normalize_symbol(target_symbol)
        self.account_id = "VIRTUAL_YAHOO_ACC"
        self.contract_id = "VIRTUAL_VIX_ID"

    @staticmethod
    def _normalize_symbol(value):
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item or "").strip()
                if text:
                    return text
            return "^VIX"
        text = str(value or "").strip()
        return text or "^VIX"

    @classmethod
    def _warn_dependency_once(cls) -> None:
        if cls._dependency_warned:
            return
        cls._dependency_warned = True
        logging.warning("YahooVIXClient unavailable: optional dependency 'yfinance' is not installed.")

    def login(self):
        if yf is None:
            self._warn_dependency_once()
            logging.info("YahooVIXClient: continuing without live Yahoo VIX data.")
            return True
        logging.info("YahooVIXClient: Virtual login successful.")
        return True

    def fetch_contracts(self):
        logging.info(f"YahooVIXClient: Virtual contract '{self.target_symbol}' selected.")
        return self.contract_id

    def validate_session(self):
        pass # No session to maintain for Yahoo

    def get_market_data(self, lookback_minutes=300, force_fetch=False):
        """
        Fetches 1-minute VIX data from Yahoo Finance and normalizes it
        to match the ProjectXClient DataFrame format.
        """
        if yf is None:
            self._warn_dependency_once()
            return pd.DataFrame()
        try:
            # Yahoo requires a period string (e.g., "5d") or start/end dates.
            # 1m data is valid for 7 days max in yfinance.
            # We request '5d' to ensure we have enough history for the 20-period SMA.
            df = yf.download(
                tickers=self.target_symbol,
                period="5d",
                interval="1m",
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                return pd.DataFrame()

            # FIX: Handle MultiIndex columns (e.g. ('Open', '^VIX')) if yfinance returns them
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten by taking the first level (Price Type)
                df.columns = df.columns.get_level_values(0)

            # Normalize Columns: yfinance returns 'Open', 'High', 'Low', 'Close', 'Volume'
            # We need lowercase: 'open', 'high', 'low', 'close', 'volume'
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            # Ensure columns are lowercase (yfinance sometimes changes case)
            df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

            # Normalize Index: Ensure it is a DatetimeIndex with timezone info
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Yahoo often returns data with 'America/New_York' tz or UTC.
            # We need to ensure it's compatible with your bot's timestamps.
            # If naive, localize to UTC. If aware, convert to UTC.
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            # Filter to requested lookback to match expected behavior (optional but clean)
            cutoff_time = datetime.now(df.index.tz) - timedelta(minutes=lookback_minutes + 100)
            df = df[df.index >= cutoff_time]

            return df

        except Exception as e:
            logging.error(f"YahooVIXClient Fetch Error: {e}")
            return pd.DataFrame()

    async def async_get_market_data(self, lookback_minutes=300, force_fetch=False):
        """Async wrapper for get_market_data() to avoid blocking the event loop."""
        return await asyncio.to_thread(
            self.get_market_data,
            lookback_minutes=lookback_minutes,
            force_fetch=force_fetch,
        )

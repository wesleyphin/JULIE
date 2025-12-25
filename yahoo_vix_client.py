import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

class YahooVIXClient:
    """
    A virtual client that mimics ProjectXClient but fetches VIX data from Yahoo Finance.
    Used because Topstep/Rithmic often excludes CBOE data.
    """
    def __init__(self, contract_root="^VIX", target_symbol="^VIX"):
        self.contract_root = contract_root
        self.target_symbol = target_symbol
        self.account_id = "VIRTUAL_YAHOO_ACC"
        self.contract_id = "VIRTUAL_VIX_ID"

    def login(self):
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
            df.columns = [c.lower() for c in df.columns]

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

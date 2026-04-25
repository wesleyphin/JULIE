from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd


NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
LOCAL_KALSHI_DAILY_DIR = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
UPSTREAM_KALSHI_DAILY_DIR = (
    ROOT / ".tmp_upstream_julie" / "data" / "kalshi" / "kxinxu_2025_daily"
)


def resolve_daily_dirs() -> List[Path]:
    dirs: List[Path] = []
    for candidate in [LOCAL_KALSHI_DAILY_DIR, UPSTREAM_KALSHI_DAILY_DIR]:
        if candidate.exists() and any(candidate.glob("*.parquet")):
            dirs.append(candidate)
    return dirs


class HistoricalKalshiProvider:
    _SETTLEMENT_HOURS = [10, 11, 12, 13, 14, 15, 16]

    def __init__(self, daily_dirs: List[Path]):
        self.daily_dirs = [Path(d) for d in daily_dirs]
        self.enabled = True
        self.is_healthy = True
        self.basis_offset = 0.0
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._sentiment_history: List[tuple[pd.Timestamp, float]] = []
        self._context_time: Optional[pd.Timestamp] = None

    def set_context_time(self, ts_et: pd.Timestamp) -> None:
        self._context_time = pd.Timestamp(ts_et).tz_convert(NY)

    def es_to_spx(self, es_price: float) -> float:
        return float(es_price) - float(self.basis_offset)

    def spx_to_es(self, spx_price: float) -> float:
        return float(spx_price) + float(self.basis_offset)

    def active_settlement_hour_et(
        self,
        ref_time: Optional[datetime] = None,
        rollover_minute: int = 5,
    ) -> Optional[int]:
        if ref_time is None:
            if self._context_time is None:
                return None
            now = self._context_time.to_pydatetime()
        else:
            now = ref_time if ref_time.tzinfo is not None else ref_time.replace(tzinfo=NY)
            now = now.astimezone(NY)
        for hour in self._SETTLEMENT_HOURS:
            if hour > now.hour or (hour == now.hour and now.minute < int(rollover_minute)):
                return hour
        return None

    def _load_day(self, date_str: str) -> Optional[pd.DataFrame]:
        if date_str in self._cache:
            return self._cache[date_str]
        for daily_dir in self.daily_dirs:
            path = daily_dir / f"{date_str}.parquet"
            if path.exists():
                df = pd.read_parquet(path).copy()
                self._cache[date_str] = df
                return df
        self._cache[date_str] = None
        return None

    def _markets_for_context(self) -> List[Dict[str, Any]]:
        if self._context_time is None:
            return []
        date_str = self._context_time.date().isoformat()
        df = self._load_day(date_str)
        if df is None or df.empty:
            return []
        settlement_hour = self.active_settlement_hour_et(
            self._context_time.to_pydatetime(),
            rollover_minute=5,
        )
        if settlement_hour is None:
            return []
        sub = df[
            (df["event_date"] == date_str)
            & (df["settlement_hour_et"].astype(int) == int(settlement_hour))
        ].copy()
        if sub.empty:
            return []
        markets: List[Dict[str, Any]] = []
        for _, row in sub.iterrows():
            hi = float(row.get("high") or 0.0)
            lo = float(row.get("low") or 0.0)
            prob = (hi + lo) / 200.0
            markets.append(
                {
                    "strike": float(row["strike"]),
                    "probability": float(prob),
                    "status": str(row.get("status", "") or ""),
                    "open_interest": int(row.get("open_interest") or 0),
                    "daily_volume": int(row.get("daily_volume") or 0),
                    "strike_es": float(row["strike"]),
                }
            )
        markets.sort(key=lambda r: r["strike"])
        return markets

    def get_relative_markets_for_ui(
        self,
        es_prices: Optional[List[float]] = None,
        window_size: int = 30,
    ) -> List[Dict[str, Any]]:
        markets = self._markets_for_context()
        if not markets:
            return []
        if len(markets) <= int(window_size):
            return markets
        ref_prices = [self.es_to_spx(float(p)) for p in (es_prices or []) if p is not None]
        if not ref_prices:
            midpoint = len(markets) // 2
            start = max(0, midpoint - (window_size // 2))
            end = min(len(markets), start + window_size)
            return markets[max(0, end - window_size):end]
        nearest = [
            min(range(len(markets)), key=lambda idx: abs(float(markets[idx]["strike"]) - ref))
            for ref in ref_prices
        ]
        lo = min(nearest)
        hi = max(nearest)
        span = (hi - lo) + 1
        if span >= window_size:
            midpoint = (lo + hi) // 2
            start = max(0, midpoint - (window_size // 2))
            end = min(len(markets), start + window_size)
            return markets[max(0, end - window_size):end]
        padding = window_size - span
        start = max(0, lo - (padding // 2))
        end = min(len(markets), hi + 1 + (padding - (padding // 2)))
        if (end - start) < window_size:
            if start == 0:
                end = min(len(markets), window_size)
            else:
                start = max(0, end - window_size)
        return markets[start:end]

    def get_probability(self, strike_price: float) -> Optional[float]:
        markets = self._markets_for_context()
        if not markets:
            return None
        exact = [m for m in markets if abs(float(m["strike"]) - float(strike_price)) < 0.01]
        if exact:
            return float(exact[0]["probability"])
        below = [m for m in markets if float(m["strike"]) <= float(strike_price)]
        above = [m for m in markets if float(m["strike"]) > float(strike_price)]
        if below and above:
            lo = below[-1]
            hi = above[0]
            lo_strike = float(lo["strike"])
            hi_strike = float(hi["strike"])
            span = max(0.5, hi_strike - lo_strike)
            weight = (float(strike_price) - lo_strike) / span
            return float(lo["probability"]) + (
                weight * (float(hi["probability"]) - float(lo["probability"]))
            )
        if below:
            return float(below[-1]["probability"])
        if above:
            return float(above[0]["probability"])
        return None

    def get_sentiment(self, strike_price: float) -> Dict[str, float]:
        markets = self._markets_for_context()
        if not markets:
            return {}
        nearest = min(markets, key=lambda m: abs(float(m["strike"]) - float(strike_price)))
        return {
            "distance_es": float(strike_price) - float(nearest["strike"]),
            "probability": float(nearest["probability"]),
            "open_interest": float(nearest.get("open_interest") or 0.0),
            "daily_volume": float(nearest.get("daily_volume") or 0.0),
        }

    def get_sentiment_momentum(
        self,
        strike_price: float,
        lookback: int = 3,
    ) -> Optional[float]:
        prob = self.get_probability(strike_price)
        if prob is None or self._context_time is None:
            return None
        self._sentiment_history.append((self._context_time, float(prob)))
        if len(self._sentiment_history) <= int(lookback):
            return 0.0
        prev = self._sentiment_history[-int(lookback) - 1][1]
        return float(prob) - float(prev)

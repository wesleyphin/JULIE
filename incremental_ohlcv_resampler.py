from __future__ import annotations

from typing import Optional

import pandas as pd


class IncrementalOHLCVResampler:
    """
    Incremental OHLCV resampler for append-only 1m streams.

    Falls back to full resample when input history diverges (non-append edits,
    truncation, or index/order changes).
    """

    def __init__(self, timeframe_minutes: int) -> None:
        self.timeframe_minutes = int(timeframe_minutes)
        self.rule = f"{self.timeframe_minutes}min"
        self._resampled: pd.DataFrame = pd.DataFrame()
        self._source_len = 0
        self._source_last_ts: Optional[pd.Timestamp] = None
        self._source_has_volume = False
        self._resampled_col_locs: Optional[dict[str, int]] = None

    @staticmethod
    def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for col in df.columns:
            c = str(col).lower()
            if c in {"open", "high", "low", "close", "volume"} and col != c:
                col_map[col] = c
        if not col_map:
            return df
        return df.rename(columns=col_map)

    def _full_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample(self.rule, closed="left", label="left").agg(agg)
        out = out.dropna(subset=["open", "high", "low", "close"])
        if "volume" in out.columns:
            out["volume"] = out["volume"].fillna(0.0)
        return out

    def _fallback_rebuild(self, df: pd.DataFrame) -> pd.DataFrame:
        self._resampled = self._full_resample(df)
        self._source_len = int(len(df))
        self._source_last_ts = df.index[-1] if len(df) else None
        self._source_has_volume = "volume" in df.columns
        self._resampled_col_locs = {name: int(idx) for idx, name in enumerate(self._resampled.columns)}
        return self._resampled

    def reset(self) -> None:
        self._resampled = pd.DataFrame()
        self._source_len = 0
        self._source_last_ts = None
        self._source_has_volume = False
        self._resampled_col_locs = None

    def _append_or_update_row(
        self,
        ts: pd.Timestamp,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        has_volume: bool,
    ) -> None:
        bin_start = pd.Timestamp(ts).floor(self.rule)
        if self._resampled.empty:
            data = {"open": [o], "high": [h], "low": [l], "close": [c]}
            if has_volume:
                data["volume"] = [v]
            self._resampled = pd.DataFrame(data, index=[bin_start])
            self._resampled_col_locs = {name: int(idx) for idx, name in enumerate(self._resampled.columns)}
            return

        last_idx = self._resampled.index[-1]
        if bin_start != last_idx:
            data = {"open": [o], "high": [h], "low": [l], "close": [c]}
            if has_volume:
                data["volume"] = [v]
            self._resampled = pd.concat([self._resampled, pd.DataFrame(data, index=[bin_start])], copy=False)
            self._resampled_col_locs = {name: int(idx) for idx, name in enumerate(self._resampled.columns)}
            return

        locs = self._resampled_col_locs
        if not locs:
            locs = {name: int(idx) for idx, name in enumerate(self._resampled.columns)}
            self._resampled_col_locs = locs

        i = -1
        high_loc = locs.get("high")
        low_loc = locs.get("low")
        close_loc = locs.get("close")
        if high_loc is not None:
            self._resampled.iat[i, high_loc] = max(float(self._resampled.iat[i, high_loc]), h)
        if low_loc is not None:
            self._resampled.iat[i, low_loc] = min(float(self._resampled.iat[i, low_loc]), l)
        if close_loc is not None:
            self._resampled.iat[i, close_loc] = c
        if has_volume:
            vol_loc = locs.get("volume")
            if vol_loc is not None:
                self._resampled.iat[i, vol_loc] = float(self._resampled.iat[i, vol_loc]) + v

    def update(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            self.reset()
            return pd.DataFrame()

        df = self._to_datetime_index(df)
        df = self._normalize_columns(df)

        required = {"open", "high", "low", "close"}
        columns = df.columns
        if any(col not in columns for col in required):
            return pd.DataFrame()
        has_volume = "volume" in columns

        cur_len = int(len(df))
        cur_ts = df.index[-1]

        if self._source_len == 0 or self._resampled.empty:
            return self._fallback_rebuild(df)

        # Same source snapshot: return cached bars as-is.
        if cur_len == self._source_len and self._source_last_ts == cur_ts:
            return self._resampled

        # History truncation or rewind.
        if cur_len < self._source_len:
            return self._fallback_rebuild(df)

        # Sliding-window append path:
        # source length is unchanged, but the last observed timestamp shifts forward by one bar
        # (common when callers provide a fixed-size rolling history window).
        if (
            cur_len == self._source_len
            and cur_len >= 2
            and self._source_last_ts is not None
            and df.index[-2] == self._source_last_ts
            and cur_ts > self._source_last_ts
        ):
            o = float(df["open"].iloc[-1])
            h = float(df["high"].iloc[-1])
            l = float(df["low"].iloc[-1])
            c = float(df["close"].iloc[-1])
            v = float(df["volume"].iloc[-1]) if has_volume else 0.0
            self._append_or_update_row(cur_ts, o, h, l, c, v, has_volume)
            self._source_last_ts = cur_ts
            self._source_has_volume = has_volume
            return self._resampled

        # Ensure previous terminal bar still aligns.
        prev_pos = self._source_len - 1
        if prev_pos < 0 or prev_pos >= cur_len:
            return self._fallback_rebuild(df)
        if df.index[prev_pos] != self._source_last_ts:
            return self._fallback_rebuild(df)

        # Fast append-by-one path (dominant in live/backtest loops).
        if cur_len == (self._source_len + 1):
            o = float(df["open"].iloc[-1])
            h = float(df["high"].iloc[-1])
            l = float(df["low"].iloc[-1])
            c = float(df["close"].iloc[-1])
            v = float(df["volume"].iloc[-1]) if has_volume else 0.0
            self._append_or_update_row(cur_ts, o, h, l, c, v, has_volume)
            self._source_len = cur_len
            self._source_last_ts = cur_ts
            self._source_has_volume = has_volume
            return self._resampled

        new_rows = df.iloc[self._source_len :]
        if new_rows.empty:
            # Same length but changed content/order; safest fallback.
            return self._fallback_rebuild(df)

        # Append/update bins from newly appended rows.
        col_names = ["open", "high", "low", "close"]
        if has_volume:
            col_names.append("volume")
        tuples = new_rows[col_names].itertuples(index=True, name=None)
        for row in tuples:
            ts = row[0]
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
            v = float(row[5]) if has_volume else 0.0
            self._append_or_update_row(ts, o, h, l, c, v, has_volume)

        self._source_len = cur_len
        self._source_last_ts = cur_ts
        self._source_has_volume = has_volume
        return self._resampled

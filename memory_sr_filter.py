import pandas as pd
import logging


class MemorySRFilter:
    """
    Scans 1-minute history to build a 'Memory' of Support and Resistance zones.
    Prevents Shorting into Support and Longing into Resistance.
    """

    def __init__(self, lookback_bars: int = 300, zone_width: float = 2.0, touch_threshold: int = 2):
        self.lookback = lookback_bars  # How far back to scan (300 mins = 5 hours)
        self.zone_width = zone_width  # Price range to consider a "cluster" (e.g., 2.0 pts)
        self.touch_threshold = touch_threshold  # Min touches to validate a zone
        self.supports = []  # List of valid support levels
        self.resistances = []  # List of valid resistance levels

    def _find_fractals(self, df: pd.DataFrame):
        """Identify fractal highs and lows in the dataframe."""
        highs = df['high'].values
        lows = df['low'].values

        fractal_highs = []
        fractal_lows = []

        # 5-bar fractal (2 left, 2 right)
        for i in range(2, len(df) - 2):
            # Bullish Fractal (Low)
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                fractal_lows.append(lows[i])

            # Bearish Fractal (High)
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                fractal_highs.append(highs[i])

        return fractal_highs, fractal_lows

    def _cluster_levels(self, levels):
        """Group nearby levels into 'Zones' based on zone_width."""
        if not levels:
            return []

        levels.sort()
        zones = []
        current_cluster = [levels[0]]

        for i in range(1, len(levels)):
            if levels[i] - current_cluster[0] <= self.zone_width:
                current_cluster.append(levels[i])
            else:
                if len(current_cluster) >= self.touch_threshold:
                    avg_price = sum(current_cluster) / len(current_cluster)
                    zones.append(avg_price)
                current_cluster = [levels[i]]

        if len(current_cluster) >= self.touch_threshold:
            avg_price = sum(current_cluster) / len(current_cluster)
            zones.append(avg_price)

        return zones

    def update(self, df: pd.DataFrame):
        """
        Re-scan history to update S/R map.
        Call this on every new bar.
        """
        if len(df) < 50:
            return

        window = df.iloc[-self.lookback:]
        current_price = df['close'].iloc[-1]

        f_highs, f_lows = self._find_fractals(window)

        raw_supports = self._cluster_levels(f_lows)
        raw_resistances = self._cluster_levels(f_highs)

        valid_supports = []
        for support in raw_supports:
            if current_price > (support - 4.0):
                valid_supports.append(support)

        valid_resistances = []
        for resistance in raw_resistances:
            if current_price < (resistance + 4.0):
                valid_resistances.append(resistance)

        self.supports = valid_supports
        self.resistances = valid_resistances

    def should_block_trade(self, side: str, current_price: float):
        """
        Check if trade is entering a Memory Zone.
        """
        block_buffer = 3.0

        if side == 'SHORT':
            for support in self.supports:
                dist = current_price - support
                if 0 < dist < block_buffer:
                    return True, f"Blocked SHORT: Memory Support Zone nearby @ {support:.2f} (Dist: {dist:.2f})"

        if side == 'LONG':
            for resistance in self.resistances:
                dist = resistance - current_price
                if 0 < dist < block_buffer:
                    return True, f"Blocked LONG: Memory Resistance Zone nearby @ {resistance:.2f} (Dist: {dist:.2f})"

        return False, None

from typing import Dict, Optional

import numpy as np
import pandas as pd


def build_volume_profile(
    df: pd.DataFrame,
    lookback: int = 120,
    tick_size: float = 0.25,
    value_area_pct: float = 0.70,
) -> Optional[Dict]:
    """
    Build a simple volume profile using the typical price per bar.

    Returns dict with: poc, vah, val, total_vol, profile (price->volume)
    """
    if df is None or len(df) < max(2, lookback):
        return None
    if tick_size <= 0:
        tick_size = 0.25

    window = df.iloc[-lookback:]
    if window.empty:
        return None

    typical_price = (window["high"] + window["low"] + window["close"]) / 3.0
    volume = window["volume"] if "volume" in window.columns else pd.Series(1.0, index=window.index)

    bins = (typical_price / tick_size).round() * tick_size
    profile: Dict[float, float] = {}
    for price, vol in zip(bins, volume):
        if not np.isfinite(price) or not np.isfinite(vol):
            continue
        price = float(price)
        profile[price] = profile.get(price, 0.0) + float(vol)

    if not profile:
        return None

    sorted_items = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)
    total_vol = float(sum(v for _, v in sorted_items))
    if total_vol <= 0:
        return None

    poc = float(sorted_items[0][0])
    target_vol = total_vol * float(value_area_pct)

    cumulative = 0.0
    value_prices = []
    for price, vol in sorted_items:
        cumulative += vol
        value_prices.append(float(price))
        if cumulative >= target_vol:
            break

    vah = float(max(value_prices))
    val = float(min(value_prices))

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "total_vol": total_vol,
        "profile": profile,
    }

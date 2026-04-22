"""Build RL training episodes from historical replay logs + parquet bars.

Each episode = one real historical trade. For the env to replay it, we need:
  - Trade metadata (strategy, side, size, entry, SL, TP, timings)
  - A DataFrame of bars covering entry time + lookahead window
  - Regime + session labels at entry
  - (Optional) Kalshi aligned probabilities at entry/SL/TP

We scan the canonical full_live_replay directories for closed_trades.json,
then pull OHLCV bars from es_master_outrights.parquet for each trade's
window (entry - 30 bars of context, +60 bars lookahead for replay).

Output: a pickle of a list[Episode] written to rl/episodes.pkl
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.trade_env import Episode, LOOKBACK_BARS, REGIME_LABELS, SESSION_LABELS

NY = ZoneInfo("America/New_York")
SCAN_ROOT = ROOT / "backtest_reports" / "full_live_replay"
ES_PARQUET = ROOT / "es_master_outrights.parquet"
OUT_EPISODES = ROOT / "rl" / "episodes.pkl"

# Same allowlist as the other overlay trainers for consistency
REPLAY_ALLOWLIST = {
    "2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06",
    "2025_07", "2025_08", "2025_09", "2025_10", "2025_11", "2025_12",
    "outrageous_feb", "outrageous_apr", "outrageous_jul", "outrageous_aug",
    "outrageous_oct",
}

# Bars to include in each episode's DataFrame: 30 context before entry,
# 60 lookahead after entry (enough for 50-bar episode horizon + buffer).
CONTEXT_BARS_BEFORE = 60   # extra buffer so LOOKBACK_BARS can always be
                            # filled even at episode start
LOOKAHEAD_BARS = 90
REGIME_WINDOW = 120
EFF_LOW = 0.05
EFF_HIGH = 0.12


def load_bars_parquet() -> pd.DataFrame:
    """Load ES parquet and resolve front-month PER DAY to avoid cross-contract
    price jumps (different outrights have different absolute prices; dedup
    per minute can flip between them on zero-volume minutes, causing ~100-pt
    discontinuities in adjacent bars)."""
    print(f"[parquet] loading {ES_PARQUET}")
    df = pd.read_parquet(ES_PARQUET)
    df = df[df.index.year >= 2024].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY)
    else:
        df.index = df.index.tz_convert(NY)
    if "symbol" in df.columns and "volume" in df.columns:
        # Per-day: pick the symbol with highest TOTAL daily volume (front month).
        df["_date"] = df.index.date
        day_sym_vol = df.groupby(["_date", "symbol"])["volume"].sum().reset_index()
        day_sym_vol = day_sym_vol.sort_values(["_date", "volume"], ascending=[True, False])
        front_per_day = day_sym_vol.drop_duplicates("_date", keep="first").set_index("_date")["symbol"]
        df["_front"] = df["_date"].map(front_per_day)
        df = df[df["symbol"] == df["_front"]].drop(columns=["_date", "_front"])
    df = df[["open", "high", "low", "close", "volume"]].copy()
    # De-dup within day (shouldn't be needed after front-month filter, but
    # handles any stray duplicate timestamps).
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"  {len(df):,} bars {df.index.min()} → {df.index.max()}")
    return df


def session_for_et_hour(h: int) -> str:
    if 18 <= h or h < 3: return "ASIA"
    if 3 <= h < 7: return "LONDON"
    if 7 <= h < 9: return "NY_PRE"
    if 9 <= h < 12: return "NY_AM"
    if 12 <= h < 16: return "NY_PM"
    return "POST"


def regime_at_bar(closes: np.ndarray, pos: int) -> str:
    if pos < REGIME_WINDOW:
        return "warmup"
    win = closes[pos - REGIME_WINDOW + 1: pos + 1]
    rets = np.diff(win) / win[:-1]
    if len(rets) == 0:
        return "warmup"
    mean = rets.mean()
    var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
    vol_bp = math.sqrt(var) * 10_000.0
    abs_sum = float(np.abs(rets).sum())
    eff = float(abs(rets.sum()) / abs_sum) if abs_sum > 0 else 0.0
    if vol_bp > 3.5 and eff < EFF_LOW:
        return "whipsaw"
    if eff > EFF_HIGH:
        return "calm_trend"
    return "neutral"


def atr14_at_bar(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, pos: int) -> float:
    if pos < 14:
        return 1.0
    trs = []
    for k in range(pos - 13, pos + 1):
        trs.append(max(highs[k] - lows[k],
                       abs(highs[k] - closes[k - 1]),
                       abs(lows[k] - closes[k - 1])))
    return float(np.mean(trs)) if trs else 1.0


def parse_sl_tp_dist(sub_strategy: str):
    """Extract (sl_dist, tp_dist) in points from strings like
    '5min_09-12_Long_Rev_T2_SL10_TP25' → (10.0, 25.0). Returns (None, None)
    if not parseable."""
    import re
    m = re.search(r"_SL(?P<sl>\d+(?:\.\d+)?)_TP(?P<tp>\d+(?:\.\d+)?)", sub_strategy or "")
    if not m:
        return None, None
    return float(m.group("sl")), float(m.group("tp"))


def build_episode_for_trade(
    trade: dict, all_bars: pd.DataFrame, closes_arr, highs_arr, lows_arr,
) -> Episode | None:
    """Build one Episode from one historical trade. Returns None if data is
    insufficient (e.g. bar window doesn't cover entry time)."""
    try:
        et_raw = datetime.fromisoformat(trade["entry_time"])
        if et_raw.tzinfo is None:
            et_raw = et_raw.replace(tzinfo=NY)
        et = et_raw.astimezone(NY)
    except Exception:
        return None
    # Find bar index for entry time
    pos = all_bars.index.searchsorted(et, side="left")
    if pos >= len(all_bars) or pos < CONTEXT_BARS_BEFORE:
        return None
    # Slice window [pos - CONTEXT_BARS_BEFORE, pos + LOOKAHEAD_BARS]
    i0 = pos - CONTEXT_BARS_BEFORE
    i1 = min(len(all_bars), pos + LOOKAHEAD_BARS + 1)
    ep_bars = all_bars.iloc[i0:i1].copy()

    # Compute features at the entry bar (ABSOLUTE index in the full parquet)
    regime = regime_at_bar(closes_arr, pos)
    atr14 = atr14_at_bar(highs_arr, lows_arr, closes_arr, pos)
    session = session_for_et_hour(et.hour)

    # SL/TP: parse from sub_strategy for DE3 trades, else derive from actual exit
    sl_dist, tp_dist = parse_sl_tp_dist(trade.get("sub_strategy", ""))
    entry_price = float(trade.get("entry_price", 0) or 0)
    side = str(trade.get("side", "")).upper()
    if sl_dist is None or tp_dist is None:
        # Approximation: use actual SL from trade if available, else skip
        sl_price = trade.get("effective_stop_price")
        tp_price = trade.get("tp_price")
        if sl_price is None or tp_price is None:
            return None
        sl_dist = abs(float(sl_price) - entry_price)
        tp_dist = abs(float(tp_price) - entry_price)
    if sl_dist <= 0 or tp_dist <= 0 or entry_price <= 0:
        return None

    if side == "LONG":
        orig_sl = entry_price - sl_dist
        orig_tp = entry_price + tp_dist
    elif side == "SHORT":
        orig_sl = entry_price + sl_dist
        orig_tp = entry_price - tp_dist
    else:
        return None

    # Align entry_time to an actual bar index in ep_bars (searchsorted snap)
    try:
        entry_bar_time = ep_bars.index[ep_bars.index.get_indexer([et], method="pad")[0]]
    except Exception:
        entry_bar_time = et

    return Episode(
        trade_id=trade.get("entry_order_id") or trade.get("trade_id"),
        strategy=str(trade.get("strategy", "Unknown")),
        sub_strategy=str(trade.get("sub_strategy", "")),
        side=side,
        size=int(trade.get("size", 1) or 1),
        entry_price=entry_price,
        entry_time=entry_bar_time,
        original_sl_price=float(orig_sl),
        original_tp_price=float(orig_tp),
        bars=ep_bars,
        regime_label=regime,
        session_label=session,
        kalshi_probs={},  # populated later if available
        atr14=float(atr14),
    )


def build_all_episodes(limit: int | None = None) -> list[Episode]:
    bars = load_bars_parquet()
    closes = bars["close"].to_numpy(dtype=np.float64)
    highs = bars["high"].to_numpy(dtype=np.float64)
    lows = bars["low"].to_numpy(dtype=np.float64)

    episodes: list[Episode] = []
    counts = {"ok": 0, "no_bars": 0, "no_sltp": 0, "parse_err": 0, "other": 0}
    for name in sorted(REPLAY_ALLOWLIST):
        ct = SCAN_ROOT / name / "closed_trades.json"
        if not ct.exists():
            continue
        try:
            trades = json.loads(ct.read_text(encoding="utf-8"))
        except Exception:
            counts["parse_err"] += 1
            continue
        for t in trades:
            ep = build_episode_for_trade(t, bars, closes, highs, lows)
            if ep is None:
                counts["no_bars"] += 1
                continue
            episodes.append(ep)
            counts["ok"] += 1
            if limit and counts["ok"] >= limit:
                print(f"  [limit] stopping at {limit}")
                return episodes
    print(f"\n[episodes] built {counts['ok']}   "
          f"(skipped: bars/sltp unavailable={counts['no_bars']}, "
          f"parse_err={counts['parse_err']})")
    return episodes


def main(limit: int | None = None):
    episodes = build_all_episodes(limit=limit)
    if not episodes:
        print("ERROR: no episodes built")
        return
    # Sort by entry_time for temporal splitting later
    episodes.sort(key=lambda e: pd.Timestamp(e.entry_time))
    print(f"[episodes] date range: {episodes[0].entry_time} → {episodes[-1].entry_time}")
    # Distribution checks
    from collections import Counter
    print(f"[strategy]: {dict(Counter(e.strategy for e in episodes))}")
    print(f"[side]:     {dict(Counter(e.side for e in episodes))}")
    print(f"[regime]:   {dict(Counter(e.regime_label for e in episodes))}")
    print(f"[session]:  {dict(Counter(e.session_label for e in episodes))}")
    with OUT_EPISODES.open("wb") as fh:
        pickle.dump(episodes, fh, protocol=pickle.HIGHEST_PROTOCOL)
    # Size on disk
    sz = OUT_EPISODES.stat().st_size / (1024 * 1024)
    print(f"\n[write] {OUT_EPISODES}  ({sz:.1f} MB, {len(episodes)} episodes)")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)

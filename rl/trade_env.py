"""Gymnasium environment for post-entry trade management.

One episode = one historical trade. Each step = one market bar. The agent
observes the bar tape + trade state and picks a management action.

─────────────────────────────────────────────────────────────────────────
State space (Box, 172-dim):

  Bar tape (last 30 bars × 5 OHLCV, normalized):
    For each bar: (close-entry)/entry, (high-close)/atr, (low-close)/atr,
                  (close-open)/atr, volume_zscore
    → 150 scalars

  Trade state (7):
    unrealized_pnl_pts / sl_dist       — in ATR-ish units, ~[-2, +5]
    bars_held / 50                     — normalized to [0,1]
    mfe_pts / sl_dist                  — max favorable excursion so far
    mae_pts / sl_dist                  — max adverse excursion so far
    side                                — +1 LONG, -1 SHORT
    current_sl_dist_pts / original_sl  — how much SL has moved (1=initial)
    current_tp_dist_pts / original_tp  — how much TP has moved

  Regime one-hot (4):   whipsaw, calm_trend, neutral, warmup
  Session one-hot (6):  ASIA, LONDON, NY_PRE, NY_AM, NY_PM, POST
  Kalshi features (3):  aligned_prob_at_current, at_sl, at_tp  (0.5 if n/a)
  Peak/trough (2):      running_peak_pnl/sl_dist, running_trough_pnl/sl_dist

  Total: 150 + 7 + 4 + 6 + 3 + 2 = 172

─────────────────────────────────────────────────────────────────────────
Action space (Discrete, 7):

  0  HOLD                  — do nothing, let the trade run
  1  MOVE_SL_TO_BE         — move stop to entry (+ 1 tick favorable)
  2  TIGHTEN_SL_25PCT      — pull SL 25% closer to current price
  3  TIGHTEN_SL_50PCT      — pull SL 50% closer
  4  TAKE_PARTIAL_50PCT    — close half the position at current price
  5  TAKE_PARTIAL_FULL     — close all at current price (manual close)
  6  REVERSE               — close + open opposite side, same size

─────────────────────────────────────────────────────────────────────────
Reward:

  per-step: -0.01 × bars_held   (small pressure against hold-forever)
  terminal: realized_pnl_dollars / 50   (scales ~$500 PnL → +10 reward)

─────────────────────────────────────────────────────────────────────────
Termination conditions:

  - Agent takes TAKE_PARTIAL_FULL or REVERSE
  - Market price crosses current SL (stopped out)
  - Market price crosses current TP (take profit hit)
  - Max bars held (50) exceeded
  - Episode bar data exhausted

─────────────────────────────────────────────────────────────────────────
Episode input contract (Episode dataclass):

  trade_id, strategy, sub_strategy, side, size, entry_price, entry_time,
  original_sl_price, original_tp_price, bars (pandas DataFrame of OHLCV
  indexed by datetime, covering entry_time through a lookahead window),
  regime_label, session_label, kalshi_probs (optional dict: prob_at_entry,
  prob_at_sl, prob_at_tp).

The env starts at the bar immediately after entry_time. At each step it
exposes the observation, accepts an action, updates SL/TP based on the
action, then advances to the next bar. If that bar's high/low crosses
the (possibly updated) SL or TP, the episode terminates with realized PnL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("gymnasium is required; pip install gymnasium") from e


# Fixed size constants (must match state-building logic exactly)
LOOKBACK_BARS = 30
BAR_FEATURES_PER_BAR = 5
BAR_TAPE_DIM = LOOKBACK_BARS * BAR_FEATURES_PER_BAR  # 150
TRADE_STATE_DIM = 7
REGIME_DIM = 4
SESSION_DIM = 6
KALSHI_DIM = 3
PEAK_TROUGH_DIM = 2
OBS_DIM = BAR_TAPE_DIM + TRADE_STATE_DIM + REGIME_DIM + SESSION_DIM + KALSHI_DIM + PEAK_TROUGH_DIM
assert OBS_DIM == 172, f"OBS_DIM mismatch: got {OBS_DIM}, expected 172"

MAX_BARS_HELD = 50

REGIME_LABELS = ("whipsaw", "calm_trend", "neutral", "warmup")
SESSION_LABELS = ("ASIA", "LONDON", "NY_PRE", "NY_AM", "NY_PM", "POST")

# Action enum
ACT_HOLD = 0
ACT_MOVE_SL_TO_BE = 1
ACT_TIGHTEN_SL_25 = 2
ACT_TIGHTEN_SL_50 = 3
ACT_TAKE_PARTIAL_50 = 4
ACT_TAKE_PARTIAL_FULL = 5
ACT_REVERSE = 6
ACTION_NAMES = {
    ACT_HOLD: "HOLD",
    ACT_MOVE_SL_TO_BE: "MOVE_SL_TO_BE",
    ACT_TIGHTEN_SL_25: "TIGHTEN_SL_25PCT",
    ACT_TIGHTEN_SL_50: "TIGHTEN_SL_50PCT",
    ACT_TAKE_PARTIAL_50: "TAKE_PARTIAL_50PCT",
    ACT_TAKE_PARTIAL_FULL: "TAKE_PARTIAL_FULL",
    ACT_REVERSE: "REVERSE",
}

# Assumptions (match live bot)
POINT_VALUE = 5.0           # MES $/pt
TICK_SIZE = 0.25
FEE_PER_CONTRACT_ROUNDTRIP = 0.74  # approximation
REVERSE_SIZE_FACTOR = 1.0   # reversal trades same size as original

# Execution-realism model (v2 follow-up after initial 802c7ef training).
#
# The v1 env let market-order closes fill at the CURRENT bar's close with
# 1 tick of slippage. That's too optimistic for two reasons:
#   (a) Live latency: a market order placed during bar T fills on bar T+1
#       open at the earliest, not bar T close. The agent effectively got
#       to see the future.
#   (b) Bar-close fills capture close-vs-open drift that the agent didn't
#       actually earn; in a bull year (2025 MES), any early-close action
#       is positive-expectation for LONGs and the 77%-LONG training set
#       makes Random baseline print absurd PnL.
#
# Fix:
#   INSTANT_FILL_BAR_OFFSET = 1    # market orders fill on bar T+1 open
#   INSTANT_SLIPPAGE_PTS = 0.5     # 2 ticks adverse (more realistic for
#                                    MES during slow hours; still optimistic
#                                    during news-driven bars)
#   PER_BAR_HOLDING_DRAG_USD = 0.0 # disable (already captured by the
#                                    holding_time_penalty reward shaping)
#
# Stop/TP fills continue to assume resting orders hit their documented
# price (instant_fill=False path, no slippage).
INSTANT_FILL_BAR_OFFSET = 1
INSTANT_SLIPPAGE_PTS = 0.5        # 2 ticks adverse
PER_BAR_HOLDING_DRAG_USD = 0.0


@dataclass
class Episode:
    """One historical trade to replay as an RL episode."""
    trade_id: Any
    strategy: str
    sub_strategy: str
    side: str                     # "LONG" | "SHORT"
    size: int
    entry_price: float
    entry_time: Any               # datetime-like
    original_sl_price: float
    original_tp_price: float
    bars: Any                     # pd.DataFrame indexed by ts, has OHLCV
    regime_label: str             # one of REGIME_LABELS
    session_label: str            # one of SESSION_LABELS
    kalshi_probs: Dict[str, float] = field(default_factory=dict)  # keys:
        # 'at_entry', 'at_sl', 'at_tp' (optional; default 0.5)
    atr14: float = 1.0            # ATR at entry (for normalization)

    def __post_init__(self):
        assert self.side in ("LONG", "SHORT"), f"bad side {self.side}"
        assert self.entry_price > 0
        assert self.atr14 > 0


def _norm_bar_tape(bars_df, i_start: int, i_end: int, entry_price: float, atr: float) -> np.ndarray:
    """Extract a fixed-width, normalized bar tape ending at i_end (inclusive).

    If not enough history exists before i_end, pad with zeros at the front.
    Returns a (LOOKBACK_BARS, BAR_FEATURES_PER_BAR) array flattened to 1-D.
    """
    out = np.zeros((LOOKBACK_BARS, BAR_FEATURES_PER_BAR), dtype=np.float32)
    take_start = max(i_start, i_end - LOOKBACK_BARS + 1)
    n_take = i_end - take_start + 1
    if n_take <= 0:
        return out.reshape(-1)
    sub = bars_df.iloc[take_start : i_end + 1]
    closes = sub["close"].to_numpy(dtype=np.float32)
    opens = sub["open"].to_numpy(dtype=np.float32)
    highs = sub["high"].to_numpy(dtype=np.float32)
    lows = sub["low"].to_numpy(dtype=np.float32)
    vols = sub["volume"].to_numpy(dtype=np.float32) if "volume" in sub.columns else np.zeros(n_take, dtype=np.float32)
    # Volume z-score within window (cheap)
    if vols.std() > 0:
        vols_z = (vols - vols.mean()) / (vols.std() + 1e-6)
    else:
        vols_z = np.zeros_like(vols)
    atr = max(atr, 0.25)  # never divide by zero
    # Pack features per bar
    block = np.stack(
        [
            (closes - entry_price) / entry_price,        # pct from entry
            (highs - closes) / atr,                       # upper wick, ATR units
            (lows - closes) / atr,                        # lower wick (negative), ATR units
            (closes - opens) / atr,                       # body
            vols_z,
        ],
        axis=1,
    )
    # Place at the END of the buffer (oldest bars at front, newest last)
    out[-n_take:] = block
    return out.reshape(-1)


def _onehot(idx: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v


class TradeManagementEnv(gym.Env):
    """Gymnasium env: one episode = one historical trade.

    Step: one bar elapses; agent picks an action; env updates SL/TP/position
    based on the action; env checks for stop/take-profit fill; env computes
    reward and advances.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: List[Episode],
        *,
        max_bars: int = MAX_BARS_HELD,
        holding_time_penalty: float = 0.01,
        reward_scale: float = 50.0,   # divide terminal $ by this
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert episodes, "at least one episode required"
        self.episodes = list(episodes)
        self.max_bars = max_bars
        self.holding_time_penalty = holding_time_penalty
        self.reward_scale = reward_scale

        self.action_space = spaces.Discrete(7)
        # Observations are clamped to a sane range; actual values rarely
        # exceed ±20 on normalized features.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._current_idx: Optional[int] = None
        self._ep: Optional[Episode] = None
        # Per-step mutable state (reset on reset())
        self._cur_bar_idx = 0           # index within ep.bars (offset from entry_bar)
        self._entry_bar_idx = 0         # index of the entry bar inside ep.bars
        self._bars_held = 0
        self._remaining_size = 0
        self._current_sl = 0.0
        self._current_tp = 0.0
        self._running_peak_pnl_pts = 0.0
        self._running_trough_pnl_pts = 0.0
        self._mfe_pts = 0.0
        self._mae_pts = 0.0
        self._realized_pnl_dollars = 0.0  # accumulated (partial closes add to it)
        self._done = False

    # -------- gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "episode_idx" in options:
            self._current_idx = int(options["episode_idx"])
        else:
            self._current_idx = int(self._rng.integers(0, len(self.episodes)))
        ep = self.episodes[self._current_idx]
        self._ep = ep
        # Locate the entry bar index in the DataFrame
        try:
            self._entry_bar_idx = int(ep.bars.index.get_loc(ep.entry_time))
        except Exception:
            # Fallback — find nearest bar at-or-after entry_time
            arr = ep.bars.index.to_numpy()
            mask = arr >= np.datetime64(ep.entry_time)
            if not mask.any():
                self._entry_bar_idx = len(ep.bars) - 1
            else:
                self._entry_bar_idx = int(np.argmax(mask))
        # Start at the bar AFTER entry. An entry executes during the signal
        # bar (typically filled at next_bar_open by the backtest). Starting
        # the agent at entry_bar would let it "close" at entry_bar_close which
        # captures close-vs-open drift the agent didn't earn.
        self._cur_bar_idx = self._entry_bar_idx + 1
        if self._cur_bar_idx >= len(ep.bars):
            # Entry at the last available bar — episode is degenerate, end it
            self._cur_bar_idx = self._entry_bar_idx
            self._done = True
        self._bars_held = 0
        self._remaining_size = int(ep.size)
        self._current_sl = float(ep.original_sl_price)
        self._current_tp = float(ep.original_tp_price)
        self._running_peak_pnl_pts = 0.0
        self._running_trough_pnl_pts = 0.0
        self._mfe_pts = 0.0
        self._mae_pts = 0.0
        self._realized_pnl_dollars = 0.0
        self._done = False
        obs = self._build_obs()
        info = {"episode_idx": self._current_idx, "trade_id": ep.trade_id}
        return obs, info

    def step(self, action: int):
        assert not self._done, "episode over; call reset()"
        ep = self._ep
        action = int(action)
        reward = 0.0

        # === apply the action BEFORE advancing the bar ===
        # cur_price is what the AGENT SEES (current bar close). fill_price
        # is what the agent ACTUALLY GETS for market orders (next bar's
        # open, per the execution-latency model).
        cur_price = self._current_price()
        fill_price = self._instant_fill_price()
        if action == ACT_MOVE_SL_TO_BE:
            self._move_sl_to_breakeven()
        elif action == ACT_TIGHTEN_SL_25:
            self._tighten_sl(0.25, cur_price)
        elif action == ACT_TIGHTEN_SL_50:
            self._tighten_sl(0.50, cur_price)
        elif action == ACT_TAKE_PARTIAL_50:
            close_sz = max(1, int(self._remaining_size * 0.5))
            pnl = self._realize_pnl(fill_price, close_sz)
            self._realized_pnl_dollars += pnl
            self._remaining_size -= close_sz
        elif action == ACT_TAKE_PARTIAL_FULL:
            pnl = self._realize_pnl(fill_price, self._remaining_size)
            self._realized_pnl_dollars += pnl
            self._remaining_size = 0
            self._done = True
        elif action == ACT_REVERSE:
            pnl = self._realize_pnl(fill_price, self._remaining_size)
            self._realized_pnl_dollars += pnl
            self._remaining_size = int(ep.size)
            new_side = "SHORT" if ep.side == "LONG" else "LONG"
            orig_sl_dist = abs(ep.original_sl_price - ep.entry_price)
            orig_tp_dist = abs(ep.original_tp_price - ep.entry_price)
            new_sl = fill_price + orig_sl_dist if new_side == "LONG" else fill_price - orig_sl_dist
            new_tp = fill_price - orig_tp_dist if new_side == "SHORT" else fill_price + orig_tp_dist
            ep = self._ep = _clone_episode_with(ep, side=new_side, entry_price=fill_price,
                                                 original_sl_price=new_sl, original_tp_price=new_tp)
            self._current_sl = new_sl
            self._current_tp = new_tp
            self._running_peak_pnl_pts = 0.0
            self._running_trough_pnl_pts = 0.0
            self._mfe_pts = 0.0
            self._mae_pts = 0.0
        # (else: HOLD, no-op)

        # === now advance the bar and check for stop/TP fills ===
        if not self._done:
            self._cur_bar_idx += 1
            self._bars_held += 1
            reward -= self.holding_time_penalty
            if self._cur_bar_idx >= len(ep.bars):
                # Bar data exhausted; close at last known price
                pnl = self._realize_pnl(cur_price, self._remaining_size)
                self._realized_pnl_dollars += pnl
                self._remaining_size = 0
                self._done = True
            else:
                bar = ep.bars.iloc[self._cur_bar_idx]
                high = float(bar["high"])
                low = float(bar["low"])
                # Check SL / TP fills in this bar (conservative: SL fills first).
                # These are resting orders → instant_fill=False (no slippage on
                # top of the documented price).
                if ep.side == "LONG":
                    if low <= self._current_sl:
                        fill = self._current_sl
                        pnl = self._realize_pnl(fill, self._remaining_size, instant_fill=False)
                        self._realized_pnl_dollars += pnl
                        self._remaining_size = 0
                        self._done = True
                    elif high >= self._current_tp:
                        fill = self._current_tp
                        pnl = self._realize_pnl(fill, self._remaining_size, instant_fill=False)
                        self._realized_pnl_dollars += pnl
                        self._remaining_size = 0
                        self._done = True
                else:  # SHORT
                    if high >= self._current_sl:
                        fill = self._current_sl
                        pnl = self._realize_pnl(fill, self._remaining_size, instant_fill=False)
                        self._realized_pnl_dollars += pnl
                        self._remaining_size = 0
                        self._done = True
                    elif low <= self._current_tp:
                        fill = self._current_tp
                        pnl = self._realize_pnl(fill, self._remaining_size, instant_fill=False)
                        self._realized_pnl_dollars += pnl
                        self._remaining_size = 0
                        self._done = True
                # Update running peak/trough / MFE / MAE
                self._update_excursions(high, low)
            # Cap bars
            if self._bars_held >= self.max_bars and not self._done:
                pnl = self._realize_pnl(self._current_price(), self._remaining_size)
                self._realized_pnl_dollars += pnl
                self._remaining_size = 0
                self._done = True

        # Terminal reward = realized $ / scale
        if self._done:
            reward += self._realized_pnl_dollars / self.reward_scale
        obs = self._build_obs()
        info = {
            "bars_held": self._bars_held,
            "remaining_size": self._remaining_size,
            "current_sl": self._current_sl,
            "current_tp": self._current_tp,
            "realized_pnl_dollars": self._realized_pnl_dollars,
            "action_name": ACTION_NAMES.get(action, f"UNK_{action}"),
        }
        terminated = self._done
        truncated = False
        return obs, float(reward), terminated, truncated, info

    # -------- internals --------
    def _current_price(self) -> float:
        ep = self._ep
        if self._cur_bar_idx >= len(ep.bars):
            return float(ep.bars.iloc[-1]["close"])
        return float(ep.bars.iloc[self._cur_bar_idx]["close"])

    def _instant_fill_price(self) -> float:
        """Price a market-order close actually fills at, given the env's
        INSTANT_FILL_BAR_OFFSET latency. Default offset=1 means the order
        is routed at the current bar's close but fills at the NEXT bar's
        open — capturing the 1-bar execution latency present in the live
        bot's ProjectX routing."""
        ep = self._ep
        target_idx = self._cur_bar_idx + INSTANT_FILL_BAR_OFFSET
        if target_idx >= len(ep.bars):
            # Not enough future data — fall back to close of last bar
            return float(ep.bars.iloc[-1]["close"])
        return float(ep.bars.iloc[target_idx]["open"])

    def _move_sl_to_breakeven(self):
        ep = self._ep
        # Move SL to entry + 1 tick in favorable direction
        if ep.side == "LONG":
            new_sl = ep.entry_price + TICK_SIZE
            # Only ratchet — never loosen
            if new_sl > self._current_sl:
                self._current_sl = new_sl
        else:
            new_sl = ep.entry_price - TICK_SIZE
            if new_sl < self._current_sl:
                self._current_sl = new_sl

    def _tighten_sl(self, fraction: float, cur_price: float):
        ep = self._ep
        if ep.side == "LONG":
            # SL below current; tighten = move up toward current
            dist = cur_price - self._current_sl
            if dist <= 0: return
            new_sl = self._current_sl + dist * fraction
            if new_sl > self._current_sl:
                self._current_sl = round(new_sl / TICK_SIZE) * TICK_SIZE
        else:
            dist = self._current_sl - cur_price
            if dist <= 0: return
            new_sl = self._current_sl - dist * fraction
            if new_sl < self._current_sl:
                self._current_sl = round(new_sl / TICK_SIZE) * TICK_SIZE

    def _realize_pnl(self, fill_price: float, size: int, *, instant_fill: bool = True) -> float:
        """Realize PnL for a partial or full close.

        instant_fill=True applies INSTANT_SLIPPAGE_PTS of adverse slippage
        (market-order fill, crosses the spread). instant_fill=False is used
        when the fill came from a resting SL/TP hitting its documented
        price (the stop/take orders rest in the book at known levels).
        """
        if size <= 0:
            return 0.0
        ep = self._ep
        if ep.side == "LONG":
            pts = fill_price - ep.entry_price
            if instant_fill:
                pts -= INSTANT_SLIPPAGE_PTS   # sold into the bid
        else:
            pts = ep.entry_price - fill_price
            if instant_fill:
                pts -= INSTANT_SLIPPAGE_PTS   # bought on the ask
        dollars = pts * POINT_VALUE * size
        dollars -= FEE_PER_CONTRACT_ROUNDTRIP * size
        return float(dollars)

    def _update_excursions(self, high: float, low: float):
        ep = self._ep
        if ep.side == "LONG":
            fav_pts = high - ep.entry_price
            adv_pts = ep.entry_price - low
        else:
            fav_pts = ep.entry_price - low
            adv_pts = high - ep.entry_price
        if fav_pts > self._mfe_pts: self._mfe_pts = fav_pts
        if adv_pts > self._mae_pts: self._mae_pts = adv_pts
        # Running peak/trough PnL (cumulative, not max)
        cur_pts = self._open_pnl_pts()
        if cur_pts > self._running_peak_pnl_pts: self._running_peak_pnl_pts = cur_pts
        if cur_pts < self._running_trough_pnl_pts: self._running_trough_pnl_pts = cur_pts

    def _open_pnl_pts(self) -> float:
        ep = self._ep
        cur = self._current_price()
        if ep.side == "LONG":
            return cur - ep.entry_price
        return ep.entry_price - cur

    def _build_obs(self) -> np.ndarray:
        ep = self._ep
        # Bar tape (ending at current bar)
        tape = _norm_bar_tape(
            ep.bars, 0, self._cur_bar_idx, ep.entry_price, ep.atr14
        )
        # Trade state
        orig_sl_dist = abs(ep.original_sl_price - ep.entry_price)
        orig_tp_dist = abs(ep.original_tp_price - ep.entry_price)
        cur_sl_dist = abs(self._current_sl - ep.entry_price)
        cur_tp_dist = abs(self._current_tp - ep.entry_price)
        sl_dist_norm = max(orig_sl_dist, 0.25)
        unrealized_pnl_pts = self._open_pnl_pts()
        ts = np.array([
            unrealized_pnl_pts / sl_dist_norm,
            self._bars_held / float(self.max_bars),
            self._mfe_pts / sl_dist_norm,
            self._mae_pts / sl_dist_norm,
            1.0 if ep.side == "LONG" else -1.0,
            cur_sl_dist / sl_dist_norm if orig_sl_dist > 0 else 1.0,
            cur_tp_dist / max(orig_tp_dist, 0.25),
        ], dtype=np.float32)
        # Regime
        reg_idx = REGIME_LABELS.index(ep.regime_label) if ep.regime_label in REGIME_LABELS else REGIME_LABELS.index("neutral")
        reg = _onehot(reg_idx, REGIME_DIM)
        # Session
        sess_idx = SESSION_LABELS.index(ep.session_label) if ep.session_label in SESSION_LABELS else SESSION_LABELS.index("NY_AM")
        sess = _onehot(sess_idx, SESSION_DIM)
        # Kalshi
        kp = ep.kalshi_probs or {}
        kals = np.array([
            float(kp.get("at_entry", 0.5)),
            float(kp.get("at_sl", 0.5)),
            float(kp.get("at_tp", 0.5)),
        ], dtype=np.float32)
        # Peak/trough
        pt = np.array([
            self._running_peak_pnl_pts / sl_dist_norm,
            self._running_trough_pnl_pts / sl_dist_norm,
        ], dtype=np.float32)
        obs = np.concatenate([tape, ts, reg, sess, kals, pt]).astype(np.float32)
        # Clip for numerical stability
        np.clip(obs, -10.0, 10.0, out=obs)
        return obs


def _clone_episode_with(ep: Episode, **overrides) -> Episode:
    """Return a copy of ep with specific fields overridden (for REVERSE action).

    This lets the env reuse the same bars_df window while flipping side /
    entry / SL / TP.
    """
    fields = dict(
        trade_id=ep.trade_id, strategy=ep.strategy, sub_strategy=ep.sub_strategy,
        side=ep.side, size=ep.size, entry_price=ep.entry_price,
        entry_time=ep.entry_time, original_sl_price=ep.original_sl_price,
        original_tp_price=ep.original_tp_price, bars=ep.bars,
        regime_label=ep.regime_label, session_label=ep.session_label,
        kalshi_probs=ep.kalshi_probs, atr14=ep.atr14,
    )
    fields.update(overrides)
    return Episode(**fields)

"""Rule-based baseline policies for the trade-management env.

Purpose: establish a reward floor the PPO agent must beat. The baselines
are:

  AlwaysHoldPolicy   — never touch SL/TP; let trade ride to original
                       SL or TP. This is the absolute baseline (no
                       management).
  DE3LikePolicy      — mirrors the live DE3 management stack:
                         - at 40% MFE → move SL to BE
                         - at 85% MFE → tighten SL to lock 85% of TP
                       (No partial TP. No reverse.)
  RandomPolicy       — sanity check; random actions across the 7-action
                       space. PPO should demolish this.

Each exposes .act(obs) → int action.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from rl.trade_env import (
    ACT_HOLD, ACT_MOVE_SL_TO_BE, ACT_TIGHTEN_SL_25, ACT_TIGHTEN_SL_50,
    ACT_TAKE_PARTIAL_50, ACT_TAKE_PARTIAL_FULL, ACT_REVERSE,
    BAR_TAPE_DIM,
)


# Trade-state offsets within the 172-dim observation (see trade_env.py layout)
# [0..150)        : bar tape
# [150..157)      : trade state (7 scalars)
# [157..161)      : regime (4)
# [161..167)      : session (6)
# [167..170)      : kalshi (3)
# [170..172)      : running peak/trough (2)
TS_OFFSET = BAR_TAPE_DIM   # 150
TS_UNREALIZED_PNL_RATIO = TS_OFFSET + 0   # unrealized_pnl_pts / sl_dist
TS_BARS_HELD_RATIO = TS_OFFSET + 1        # bars_held / max_bars
TS_MFE_RATIO = TS_OFFSET + 2              # mfe_pts / sl_dist
TS_MAE_RATIO = TS_OFFSET + 3              # mae_pts / sl_dist
TS_SIDE = TS_OFFSET + 4                    # +1 LONG, -1 SHORT
TS_CUR_SL_RATIO = TS_OFFSET + 5            # cur_sl_dist / orig_sl_dist
TS_CUR_TP_RATIO = TS_OFFSET + 6            # cur_tp_dist / orig_tp_dist


class BasePolicy:
    def reset(self):
        """Reset any per-episode state."""
        pass

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError


class AlwaysHoldPolicy(BasePolicy):
    def act(self, obs: np.ndarray) -> int:
        return ACT_HOLD


class RandomPolicy(BasePolicy):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, 7))


class DE3LikePolicy(BasePolicy):
    """Mirrors current DE3 management stack:
        - at 40% MFE ratio → MOVE_SL_TO_BE   (break-even lock)
        - at 85% MFE ratio → TIGHTEN_SL_50   (lock most of TP)
    'MFE ratio' is computed as mfe_pts / tp_dist_pts; since the observation
    normalizes MFE by SL distance, we convert using the original SL/TP ratio
    stored elsewhere. For simplicity we use mfe_pts / sl_dist (which is what
    the obs exposes) and tune thresholds empirically.

    A real bar typically has SL:TP ratios like 10:25 (0.4) or 4:6 (0.66),
    so an MFE of 40% of TP corresponds to ~1.0 (SL=TP) up to ~1.0× sl_dist.
    We use 1.0× sl_dist as the BE trigger, and 2.0× sl_dist as the lock trigger.
    That approximates 40% / 85% of TP for typical DE3 brackets.
    """
    BE_MFE_SL_RATIO = 1.0    # break-even once favorable move >= 1× sl_dist
    LOCK_MFE_SL_RATIO = 2.0  # tighten once favorable move >= 2× sl_dist

    def __init__(self):
        self._be_done = False
        self._lock_done = False

    def reset(self):
        self._be_done = False
        self._lock_done = False

    def act(self, obs: np.ndarray) -> int:
        mfe_ratio = float(obs[TS_MFE_RATIO])
        if not self._be_done and mfe_ratio >= self.BE_MFE_SL_RATIO:
            self._be_done = True
            return ACT_MOVE_SL_TO_BE
        if not self._lock_done and mfe_ratio >= self.LOCK_MFE_SL_RATIO:
            self._lock_done = True
            return ACT_TIGHTEN_SL_50
        return ACT_HOLD


def evaluate_policy(env, policy: BasePolicy, n_episodes: int, *, seed: int = 0) -> dict:
    """Run policy over env for n_episodes (picked via options={'episode_idx'})
    and report aggregate stats."""
    pnls = []
    rewards = []
    bars_held = []
    action_counts = {i: 0 for i in range(7)}
    terminal_reasons = {"stop_or_tp": 0, "manual_close": 0, "reverse": 0, "max_bars": 0, "data_end": 0}
    for i in range(n_episodes):
        policy.reset()
        obs, info = env.reset(options={"episode_idx": i % len(env.episodes)})
        total_r = 0.0
        last_info = info
        while True:
            a = policy.act(obs)
            action_counts[int(a)] += 1
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            last_info = info
            if term or trunc:
                break
        pnls.append(last_info["realized_pnl_dollars"])
        rewards.append(total_r)
        bars_held.append(last_info["bars_held"])
    pnls = np.array(pnls)
    return {
        "n": len(pnls),
        "total_pnl": float(pnls.sum()),
        "mean_pnl": float(pnls.mean()),
        "median_pnl": float(np.median(pnls)),
        "win_rate": float((pnls > 0).mean()),
        "mean_reward": float(np.mean(rewards)),
        "mean_bars_held": float(np.mean(bars_held)),
        "action_counts": action_counts,
    }


if __name__ == "__main__":
    import pickle, sys
    from rl.trade_env import TradeManagementEnv

    ep_pkl = __import__("pathlib").Path(__file__).resolve().parents[1] / "rl" / "episodes.pkl"
    episodes = pickle.load(open(ep_pkl, "rb"))
    print(f"Loaded {len(episodes)} episodes from {ep_pkl}\n")

    env = TradeManagementEnv(episodes)

    print(f"{'policy':<20}{'n':>6}{'total_PnL':>12}{'mean_$/tr':>11}"
          f"{'median':>10}{'win_rt':>8}{'bars':>7}{'actions':>10}")
    print("-" * 92)
    for name, pol in [
        ("AlwaysHold", AlwaysHoldPolicy()),
        ("DE3Like", DE3LikePolicy()),
        ("Random", RandomPolicy(seed=42)),
    ]:
        stats = evaluate_policy(env, pol, n_episodes=len(episodes))
        ac = stats["action_counts"]
        nz = sum(v for k, v in ac.items() if k != ACT_HOLD)
        print(f"{name:<20}{stats['n']:>6}{stats['total_pnl']:>+12,.0f}"
              f"{stats['mean_pnl']:>+11.2f}{stats['median_pnl']:>+10.2f}"
              f"{stats['win_rate']*100:>7.1f}%{stats['mean_bars_held']:>7.1f}"
              f"  {nz:>5} non-HOLD")

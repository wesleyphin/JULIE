"""Side-by-side comparison harness for PPO vs rule baselines.

Run after training to summarize the trained policy's behavior against
AlwaysHold and DE3Like on the full episode set (not just the training
split). Prints a tight table suitable for commit messages and README.

Usage:
  python3 rl/compare_policies.py [--model PATH]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.trade_env import TradeManagementEnv, ACTION_NAMES
from rl.baseline_policy import (
    AlwaysHoldPolicy, DE3LikePolicy, RandomPolicy, evaluate_policy,
)


def evaluate_sb3(model, episodes):
    env = TradeManagementEnv(episodes, seed=123)
    pnls, bars_held = [], []
    action_counts = {i: 0 for i in range(7)}
    for i in range(len(episodes)):
        obs, info = env.reset(options={"episode_idx": i})
        last_info = info
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, r, term, trunc, info = env.step(int(action))
            last_info = info
            if term or trunc:
                break
        pnls.append(last_info["realized_pnl_dollars"])
        bars_held.append(last_info["bars_held"])
    pnls = np.array(pnls)
    return {
        "n": len(pnls), "total_pnl": float(pnls.sum()),
        "mean_pnl": float(pnls.mean()), "median_pnl": float(np.median(pnls)),
        "win_rate": float((pnls > 0).mean()),
        "mean_bars_held": float(np.mean(bars_held)),
        "action_counts": action_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "signal_gate_2025" / "model_rl_management.zip"))
    ap.add_argument("--subset", type=int, default=0, help="Evaluate first N episodes only (0=all)")
    args = ap.parse_args()

    episodes = pickle.load(open(ROOT / "rl" / "episodes.pkl", "rb"))
    if args.subset:
        episodes = episodes[: args.subset]
    print(f"Evaluating on {len(episodes)} episodes...")
    env = TradeManagementEnv(episodes)

    results = {}
    for name, pol in [
        ("AlwaysHold", AlwaysHoldPolicy()),
        ("DE3Like", DE3LikePolicy()),
        ("Random", RandomPolicy(seed=42)),
    ]:
        results[name] = evaluate_policy(env, pol, n_episodes=len(episodes))

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    results["PPO"] = evaluate_sb3(model, episodes)

    print()
    print(f"{'policy':<14}{'n':>5}{'total_PnL':>12}{'mean_$':>10}"
          f"{'median':>9}{'WR':>7}{'bars':>6}  top actions")
    print("-" * 90)
    for name, s in results.items():
        ac = s["action_counts"]
        top = sorted(ac.items(), key=lambda kv: -kv[1])[:3]
        top_str = ", ".join(f"{ACTION_NAMES[k][:6]}={v}" for k, v in top if v > 0)
        print(f"{name:<14}{s['n']:>5}{s['total_pnl']:>+12,.0f}"
              f"{s['mean_pnl']:>+10.2f}{s['median_pnl']:>+9.2f}"
              f"{s['win_rate']*100:>6.1f}%{s['mean_bars_held']:>6.1f}  {top_str}")

    # Deltas
    r = results
    print()
    print(f"PPO Δ vs DE3Like:     ${r['PPO']['total_pnl'] - r['DE3Like']['total_pnl']:+,.0f}   "
          f"(${r['PPO']['mean_pnl'] - r['DE3Like']['mean_pnl']:+.2f}/trade)")
    print(f"PPO Δ vs AlwaysHold:  ${r['PPO']['total_pnl'] - r['AlwaysHold']['total_pnl']:+,.0f}")
    print(f"PPO Δ vs Random:      ${r['PPO']['total_pnl'] - r['Random']['total_pnl']:+,.0f}")


if __name__ == "__main__":
    main()

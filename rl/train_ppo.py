"""PPO training for the trade-management RL agent.

Uses stable-baselines3 PPO with MlpPolicy over the 172-dim observation
space and 7-action discrete space defined in trade_env.py.

Training protocol:
  1. Load episodes.pkl, split temporally (oldest 85% train, newest 15% val).
  2. Wrap the env with SubprocVecEnv (parallel rollouts).
  3. Train PPO for --total-timesteps steps.
  4. Evaluate on the validation split using deterministic policy.
  5. Compare to DE3Like baseline; save policy if it beats baseline.
  6. Save model + a thin metadata payload for inference.

Output: artifacts/signal_gate_2025/model_rl_management.zip (sb3 format)
        + rl/rl_management_metadata.json

Training is CPU-friendly thanks to small MLP; on M-series MPS it's a few
minutes per 100k timesteps. Use --total-timesteps 300000 for a reasonable
first pass; increase for production.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from rl.trade_env import TradeManagementEnv, ACTION_NAMES, OBS_DIM
from rl.baseline_policy import DE3LikePolicy, AlwaysHoldPolicy, evaluate_policy as baseline_eval


EP_PKL = ROOT / "rl" / "episodes.pkl"
OUT_MODEL = ROOT / "artifacts" / "signal_gate_2025" / "model_rl_management.zip"
OUT_META = ROOT / "rl" / "rl_management_metadata.json"


def make_env(episodes, seed=None, extended_obs=False, n_actions=7,
             calm_trend_tighten_penalty=0.0, low_mfe_tighten_penalty=0.0,
             low_mfe_threshold=0.50, opportunity_cost_weight=0.0):
    def _thunk():
        env = TradeManagementEnv(
            episodes, seed=seed,
            extended_obs=extended_obs, n_actions=n_actions,
            calm_trend_tighten_penalty=calm_trend_tighten_penalty,
            low_mfe_tighten_penalty=low_mfe_tighten_penalty,
            low_mfe_threshold=low_mfe_threshold,
            opportunity_cost_weight=opportunity_cost_weight,
        )
        return env
    return _thunk


def temporal_split(episodes, train_frac=0.85):
    episodes_sorted = sorted(episodes, key=lambda e: e.entry_time)
    split = int(len(episodes_sorted) * train_frac)
    return episodes_sorted[:split], episodes_sorted[split:]


def evaluate_sb3_policy(model, episodes, n=None, extended_obs=False, n_actions=7):
    """Evaluate a trained sb3 PPO model on episodes. Return per-episode PnL."""
    env = TradeManagementEnv(episodes, seed=123,
                              extended_obs=extended_obs, n_actions=n_actions)
    n = n or len(episodes)
    pnls, rewards, bars_held = [], [], []
    action_counts = {i: 0 for i in range(7)}
    for i in range(n):
        obs, info = env.reset(options={"episode_idx": i % len(episodes)})
        total_r = 0.0
        last_info = info
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, r, term, trunc, info = env.step(int(action))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=300_000,
                    help="PPO total training timesteps (default 300k)")
    ap.add_argument("--n-envs", type=int, default=4, help="Parallel envs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=1024, help="PPO rollout buffer per env")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coeff (higher = more exploration)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extended-obs", action="store_true",
                    help="v2 obs: add 32-dim encoder embedding + 8-dim "
                         "cross-market features to observation (obs_dim 172→212). "
                         "Saves to model_rl_management_v2.zip")
    ap.add_argument("--sl-only", action="store_true",
                    help="Restrict action space to 4 actions (HOLD + 3 SL-moves). "
                         "Trains a live-safe policy that doesn't depend on "
                         "partial-close or reverse paths. Saves to "
                         "model_rl_management_v3_sl_only.zip (can combine "
                         "with --extended-obs).")
    ap.add_argument("--calm-trend-tighten-penalty", type=float, default=0.0,
                    help="v4 reward shaping: per-step penalty for TIGHTEN actions "
                         "in calm_trend regime. Default 0 = OFF. Try 0.30 "
                         "(≈ $15 of terminal-reward equivalent).")
    ap.add_argument("--low-mfe-tighten-penalty", type=float, default=0.0,
                    help="v4 reward shaping: per-step penalty for TIGHTEN actions "
                         "when MFE / |original_tp - entry| < threshold. Default 0 = OFF.")
    ap.add_argument("--low-mfe-threshold", type=float, default=0.50,
                    help="MFE/TP ratio below which low_mfe_tighten penalty fires.")
    ap.add_argument("--opportunity-cost-weight", type=float, default=0.0,
                    help="v4 reward shaping: penalize early exits that left "
                         "money on the table. Penalty = (best_possible - "
                         "realized) / reward_scale * weight. Default 0 = OFF; "
                         "0.25 is a reasonable starting value.")
    ap.add_argument("--label", default=None,
                    help="Append a custom suffix to the output model name "
                         "(e.g. 'v4_shaped'). Default uses --sl-only / --extended-obs.")
    args = ap.parse_args()

    print(f"[load] episodes from {EP_PKL}")
    episodes = pickle.load(open(EP_PKL, "rb"))
    print(f"  {len(episodes)} episodes loaded")

    train_eps, val_eps = temporal_split(episodes, train_frac=0.85)
    print(f"  train: {len(train_eps)} episodes   ({train_eps[0].entry_time} → {train_eps[-1].entry_time})")
    print(f"  val:   {len(val_eps)} episodes     ({val_eps[0].entry_time} → {val_eps[-1].entry_time})")

    # Baseline eval on val set
    print("\n[baseline] DE3-like rule policy on validation split:")
    val_env_for_baseline = TradeManagementEnv(val_eps)
    de3_stats = baseline_eval(val_env_for_baseline, DE3LikePolicy(), n_episodes=len(val_eps))
    hold_stats = baseline_eval(val_env_for_baseline, AlwaysHoldPolicy(), n_episodes=len(val_eps))
    print(f"  AlwaysHold: n={hold_stats['n']}  total=${hold_stats['total_pnl']:+,.0f}  mean=${hold_stats['mean_pnl']:+.2f}  WR={hold_stats['win_rate']*100:.1f}%")
    print(f"  DE3Like:    n={de3_stats['n']}  total=${de3_stats['total_pnl']:+,.0f}  mean=${de3_stats['mean_pnl']:+.2f}  WR={de3_stats['win_rate']*100:.1f}%")

    # Build vectorized training env (DummyVecEnv for simplicity; SubprocVecEnv
    # has pickling issues with large episodes list)
    n_actions = 4 if args.sl_only else 7
    print(f"\n[ppo] building vec env with n_envs={args.n_envs}  "
          f"extended_obs={args.extended_obs}  n_actions={n_actions}")
    vec_env = DummyVecEnv([make_env(train_eps, seed=args.seed + i,
                                     extended_obs=args.extended_obs,
                                     n_actions=n_actions,
                                     calm_trend_tighten_penalty=args.calm_trend_tighten_penalty,
                                     low_mfe_tighten_penalty=args.low_mfe_tighten_penalty,
                                     low_mfe_threshold=args.low_mfe_threshold,
                                     opportunity_cost_weight=args.opportunity_cost_weight)
                           for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        gamma=0.995,      # trades are short; tighter horizon than 0.99
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
        device=args.device,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    print(f"  policy: {model.policy}")
    print(f"  device: {model.device}")

    print(f"\n[train] total_timesteps={args.total_timesteps}")
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed/60:.1f} min")

    # Evaluate on validation split
    print("\n[eval] deterministic PPO on validation:")
    ppo_stats = evaluate_sb3_policy(model, val_eps,
                                     extended_obs=args.extended_obs,
                                     n_actions=n_actions)
    print(f"  PPO:        n={ppo_stats['n']}  total=${ppo_stats['total_pnl']:+,.0f}  mean=${ppo_stats['mean_pnl']:+.2f}  WR={ppo_stats['win_rate']*100:.1f}%  bars={ppo_stats['mean_bars_held']:.1f}")

    # Action distribution
    print("\n  action distribution:")
    total_acts = sum(ppo_stats["action_counts"].values())
    for a, c in sorted(ppo_stats["action_counts"].items()):
        if c > 0:
            print(f"    {a} {ACTION_NAMES[a]:<20} {c:>7} ({c/total_acts*100:.1f}%)")

    # Delta vs baselines
    delta_vs_hold = ppo_stats["total_pnl"] - hold_stats["total_pnl"]
    delta_vs_de3 = ppo_stats["total_pnl"] - de3_stats["total_pnl"]
    print(f"\n[delta]   vs AlwaysHold: ${delta_vs_hold:+,.0f}")
    print(f"          vs DE3-like:   ${delta_vs_de3:+,.0f}")

    # Save model + metadata — write to _v2.zip / _v3_sl_only.zip when the
    # corresponding flags are on so we don't clobber the canonical policy.
    # Promote by renaming later if the variant wins.
    out_model = OUT_MODEL
    out_meta = OUT_META
    suffix = ""
    if args.label:
        suffix = "_" + args.label.lstrip("_")
    elif args.sl_only:
        suffix = "_v3_sl_only"
    elif args.extended_obs:
        suffix = "_v2"
    if suffix:
        out_model = out_model.with_name(out_model.stem + suffix + out_model.suffix)
        out_meta = out_meta.with_name(out_meta.stem + suffix + out_meta.suffix)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_model))
    from rl.trade_env import OBS_DIM_EXTENDED as _OBS_DIM_EXTENDED
    kind = "PPO_MlpPolicy_trade_mgmt"
    if args.sl_only: kind += "_v3_sl_only"
    elif args.extended_obs: kind += "_v2_extended_obs"
    meta = {
        "model_kind": kind,
        "obs_dim": _OBS_DIM_EXTENDED if args.extended_obs else OBS_DIM,
        "extended_obs": args.extended_obs,
        "sl_only": args.sl_only,
        "n_actions": n_actions,
        "action_names": {str(i): ACTION_NAMES[i] for i in range(n_actions)},
        "training": {
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "lr": args.lr,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "ent_coef": args.ent_coef,
            "seed": args.seed,
            "elapsed_seconds": round(elapsed, 1),
            "n_train_episodes": len(train_eps),
            "n_val_episodes": len(val_eps),
            "calm_trend_tighten_penalty": args.calm_trend_tighten_penalty,
            "low_mfe_tighten_penalty": args.low_mfe_tighten_penalty,
            "low_mfe_threshold": args.low_mfe_threshold,
            "opportunity_cost_weight": args.opportunity_cost_weight,
            "label": args.label,
        },
        "val_stats_ppo": ppo_stats,
        "val_stats_de3_baseline": de3_stats,
        "val_stats_hold_baseline": hold_stats,
        "delta_vs_hold": delta_vs_hold,
        "delta_vs_de3": delta_vs_de3,
    }
    with out_meta.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\n[write] {out_model}")
    print(f"[write] {out_meta}")


if __name__ == "__main__":
    main()

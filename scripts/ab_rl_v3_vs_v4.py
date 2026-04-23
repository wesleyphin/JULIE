#!/usr/bin/env python3
"""A/B compare RL v3 (canonical) vs v4 (regime+MFE+opportunity-cost shaped).

Both models are evaluated on the SAME held-out validation split
(Sep-Dec 2025, ~640 episodes). Promotion criterion:
    v4 must beat v3 on BOTH mean PnL/trade AND win rate.

If v4 wins, prints a "PROMOTE" recommendation. If it ties or loses,
prints "KEEP v3" and explains why.

The actual promotion (renaming v4.zip → canonical) is left to a manual
step so the user can review before flipping the live model.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from stable_baselines3 import PPO

from rl.trade_env import TradeManagementEnv, ACTION_NAMES

ROOT = Path("/Users/wes/Downloads/JULIE001")
ART = ROOT / "artifacts" / "signal_gate_2025"
EP_PKL = ROOT / "rl" / "episodes.pkl"

V3_PATH = ART / "model_rl_management_v3_sl_only.zip"
V4_PATH = ART / "model_rl_management_v4_calm_shaped.zip"


def temporal_split(episodes, train_frac=0.85):
    es = sorted(episodes, key=lambda e: e.entry_time)
    split = int(len(es) * train_frac)
    return es[:split], es[split:]


def evaluate(model, episodes, label):
    """Run model deterministically over every episode. Same eval logic as
    train_ppo.evaluate_sb3_policy but using shaping-OFF env so the metric
    is pure realized PnL (not biased by the v4 shaping penalties)."""
    env = TradeManagementEnv(
        episodes, seed=123, extended_obs=True, n_actions=4,
        # Shaping OFF for evaluation — we want raw realized $.
        calm_trend_tighten_penalty=0.0,
        low_mfe_tighten_penalty=0.0,
        opportunity_cost_weight=0.0,
    )
    pnls, bars = [], []
    action_counts = {i: 0 for i in range(4)}
    for i in range(len(episodes)):
        obs, _ = env.reset(options={"episode_idx": i})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, _, done, _, info = env.step(int(action))
        pnls.append(info["realized_pnl_dollars"])
        bars.append(info["bars_held"])
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    return {
        "label": label,
        "n": n,
        "total_pnl": round(sum(pnls), 2),
        "mean_pnl": round(sum(pnls) / n, 2),
        "win_rate": round(wins / n, 4),
        "wins": wins, "losses": losses,
        "mean_bars_held": round(sum(bars) / n, 1),
        "action_counts": action_counts,
    }


def main():
    print(f"[load] {EP_PKL}")
    eps = pickle.load(open(EP_PKL, "rb"))
    _, val_eps = temporal_split(eps)
    print(f"  val split: {len(val_eps)} episodes "
          f"({val_eps[0].entry_time} → {val_eps[-1].entry_time})")

    if not V3_PATH.exists():
        raise SystemExit(f"v3 not found at {V3_PATH}")
    if not V4_PATH.exists():
        raise SystemExit(
            f"v4 not found at {V4_PATH} — train it first:\n"
            f"  python3 -u -m rl.train_ppo --sl-only --extended-obs "
            f"--calm-trend-tighten-penalty 0.30 --low-mfe-tighten-penalty 0.20 "
            f"--opportunity-cost-weight 0.25 --label v4_calm_shaped"
        )

    print(f"\n[load] v3 = {V3_PATH.name}")
    v3 = PPO.load(str(V3_PATH))
    print(f"[load] v4 = {V4_PATH.name}")
    v4 = PPO.load(str(V4_PATH))

    print(f"\n[eval] running v3 over {len(val_eps)} val episodes...")
    v3_stats = evaluate(v3, val_eps, "v3_sl_only")
    print(f"[eval] running v4 over {len(val_eps)} val episodes...")
    v4_stats = evaluate(v4, val_eps, "v4_calm_shaped")

    # Pretty-print
    print(f"\n{'='*72}")
    print(f"{'metric':<25} {'v3 (canonical)':>18} {'v4 (shaped)':>18}  Δ")
    print("-"*72)
    for k, fmt in [("total_pnl", "${:+,.2f}"), ("mean_pnl", "${:+.2f}"),
                   ("win_rate", "{:.1%}"), ("mean_bars_held", "{:.1f}")]:
        a, b = v3_stats[k], v4_stats[k]
        d = b - a
        print(f"{k:<25} {fmt.format(a):>18} {fmt.format(b):>18}  {fmt.format(d) if k!='win_rate' else f'{d*100:+.2f}pp':>10}")

    print(f"\n  v3 actions:")
    for a, c in sorted(v3_stats["action_counts"].items()):
        if c: print(f"    {a} {ACTION_NAMES[a]:<22} {c:>6}")
    print(f"  v4 actions:")
    for a, c in sorted(v4_stats["action_counts"].items()):
        if c: print(f"    {a} {ACTION_NAMES[a]:<22} {c:>6}")

    # Promotion decision
    mean_pnl_better = v4_stats["mean_pnl"] > v3_stats["mean_pnl"]
    wr_better = v4_stats["win_rate"] > v3_stats["win_rate"]
    print(f"\n{'='*72}")
    if mean_pnl_better and wr_better:
        delta_total = v4_stats["total_pnl"] - v3_stats["total_pnl"]
        print(f"PROMOTE → v4 wins on both mean PnL ({v4_stats['mean_pnl']:.2f} > "
              f"{v3_stats['mean_pnl']:.2f}) AND WR ({v4_stats['win_rate']*100:.1f}% > "
              f"{v3_stats['win_rate']*100:.1f}%).  Δ total = ${delta_total:+,.2f}")
        print(f"\n  To promote:  cp {V4_PATH} {ART}/model_rl_management.zip")
    elif mean_pnl_better and not wr_better:
        print(f"AMBIGUOUS — v4 mean PnL up but WR down. "
              f"Keeping v3 (criterion requires both).")
    elif not mean_pnl_better and wr_better:
        print(f"AMBIGUOUS — v4 WR up but mean PnL down. "
              f"Keeping v3 (criterion requires both).")
    else:
        print(f"KEEP v3 — v4 lost on both metrics. Live executor gates "
              f"(julie001._apply_rl_management_action) still protect the bot.")
    print("="*72)

    # Save A/B json
    out = ROOT / "backtest_reports" / "rl_v3_vs_v4_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"v3": v3_stats, "v4": v4_stats}, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()

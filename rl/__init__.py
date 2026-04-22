"""Reinforcement-learning trade management stack (Path 3 of the smartness roadmap).

Module layout:
  trade_env.py        — gymnasium environment for post-entry trade management
  build_episodes.py   — extract training episodes from replay logs + parquet
  baseline_policy.py  — rule-based baseline policy (reward floor)
  train_ppo.py        — PPO training loop with rolling-origin validation
  inference.py        — live-inference wrapper (loads policy → action per bar)
"""

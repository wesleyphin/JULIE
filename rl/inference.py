"""Live-inference wrapper for the PPO trade-management policy.

Loads the trained SB3 PPO zip lazily (first use). Exposes:

  init_rl_policy() -> bool             — load the policy; True on success
  score_rl_management(**kwargs) -> Optional[tuple[int, str]]
      Build the 172-dim observation from the bot's live trade state + bar
      history + regime/session/Kalshi inputs and return (action_int,
      action_name). Returns None if policy unavailable.
  is_rl_management_live_active() -> bool
      Reads JULIE_ML_RL_MGMT_ACTIVE env var (default '0' = shadow only).

The caller is responsible for:
  - Building the observation inputs (bar tape, trade state, labels).
  - Interpreting / executing the action (the caller decides whether to
    actually apply it, even in "live" mode).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rl.trade_env import (
    LOOKBACK_BARS, BAR_FEATURES_PER_BAR, BAR_TAPE_DIM, TRADE_STATE_DIM,
    REGIME_DIM, SESSION_DIM, KALSHI_DIM, PEAK_TROUGH_DIM, OBS_DIM,
    REGIME_LABELS, SESSION_LABELS, ACTION_NAMES, _norm_bar_tape, _onehot,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "signal_gate_2025" / "model_rl_management.zip"

_POLICY = None
_POLICY_LOAD_FAILED = False


def init_rl_policy() -> bool:
    """Eager-load the PPO policy from disk. Returns True on success."""
    global _POLICY, _POLICY_LOAD_FAILED
    if _POLICY is not None:
        return True
    if _POLICY_LOAD_FAILED:
        return False
    if not MODEL_PATH.exists():
        logging.info("rl.inference: no model at %s — policy disabled", MODEL_PATH)
        _POLICY_LOAD_FAILED = True
        return False
    try:
        from stable_baselines3 import PPO
        _POLICY = PPO.load(str(MODEL_PATH), device="cpu")  # cpu inference is fast for MLP
        logging.info("rl.inference: PPO policy loaded from %s", MODEL_PATH)
        return True
    except Exception as exc:
        logging.warning("rl.inference: failed to load policy: %s", exc)
        _POLICY_LOAD_FAILED = True
        return False


def _build_observation(
    *,
    bars_df,                        # pandas DataFrame ending at the current bar
    entry_price: float,
    side: str,                      # "LONG" | "SHORT"
    atr14: float,
    bars_held: int,
    mfe_pts: float,
    mae_pts: float,
    current_sl_price: float,
    current_tp_price: float,
    original_sl_price: float,
    original_tp_price: float,
    running_peak_pnl_pts: float,
    running_trough_pnl_pts: float,
    regime_label: str,
    session_label: str,
    kalshi_probs: Optional[Dict[str, float]] = None,
    max_bars: int = 50,
) -> np.ndarray:
    """Assemble a 172-dim observation exactly matching the training env's
    layout. The bar tape is computed from the TAIL of bars_df (the most
    recent LOOKBACK_BARS bars, padded with zeros if fewer are available)."""
    kalshi_probs = kalshi_probs or {}
    # Reuse the env's normalizer (expects entire bars_df + start/end idx)
    end_idx = len(bars_df) - 1
    start_idx = 0  # use whole frame; normalizer takes last N itself
    tape = _norm_bar_tape(bars_df, start_idx, end_idx, float(entry_price), float(atr14))

    # Trade state
    orig_sl_dist = max(0.25, abs(original_sl_price - entry_price))
    orig_tp_dist = max(0.25, abs(original_tp_price - entry_price))
    cur_price = float(bars_df.iloc[-1]["close"])
    if side == "LONG":
        unrealized_pts = cur_price - entry_price
    else:
        unrealized_pts = entry_price - cur_price
    cur_sl_dist = abs(current_sl_price - entry_price)
    cur_tp_dist = abs(current_tp_price - entry_price)
    ts = np.array([
        unrealized_pts / orig_sl_dist,
        bars_held / float(max_bars),
        mfe_pts / orig_sl_dist,
        mae_pts / orig_sl_dist,
        1.0 if side == "LONG" else -1.0,
        cur_sl_dist / orig_sl_dist,
        cur_tp_dist / orig_tp_dist,
    ], dtype=np.float32)

    reg_idx = REGIME_LABELS.index(regime_label) if regime_label in REGIME_LABELS else REGIME_LABELS.index("neutral")
    sess_idx = SESSION_LABELS.index(session_label) if session_label in SESSION_LABELS else SESSION_LABELS.index("NY_AM")
    reg = _onehot(reg_idx, REGIME_DIM)
    sess = _onehot(sess_idx, SESSION_DIM)
    kals = np.array([
        float(kalshi_probs.get("at_entry", 0.5)),
        float(kalshi_probs.get("at_sl", 0.5)),
        float(kalshi_probs.get("at_tp", 0.5)),
    ], dtype=np.float32)
    pt = np.array([
        running_peak_pnl_pts / orig_sl_dist,
        running_trough_pnl_pts / orig_sl_dist,
    ], dtype=np.float32)
    obs = np.concatenate([tape, ts, reg, sess, kals, pt]).astype(np.float32)
    np.clip(obs, -10.0, 10.0, out=obs)
    assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape} != {OBS_DIM}"
    return obs


def score_rl_management(
    *, deterministic: bool = True, **obs_kwargs,
) -> Optional[Tuple[int, str]]:
    """Query the PPO policy for an action given the current trade state.

    Accepts the same keyword arguments as _build_observation(). Returns
    (action_int, action_name) or None if the policy isn't loaded.
    """
    if _POLICY is None and not init_rl_policy():
        return None
    try:
        obs = _build_observation(**obs_kwargs)
        action, _ = _POLICY.predict(obs, deterministic=deterministic)
        a = int(action)
        return a, ACTION_NAMES.get(a, f"UNK_{a}")
    except Exception as exc:
        logging.debug("score_rl_management failed: %s", exc)
        return None


def is_rl_management_live_active() -> bool:
    return os.environ.get("JULIE_ML_RL_MGMT_ACTIVE", "0").strip() == "1"

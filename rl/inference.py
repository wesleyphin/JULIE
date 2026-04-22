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
    OBS_DIM_EXTENDED, ENCODER_DIM, CROSS_MARKET_DIM,
    REGIME_LABELS, SESSION_LABELS, ACTION_NAMES, _norm_bar_tape, _onehot,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "signal_gate_2025" / "model_rl_management.zip"

_POLICY = None
_POLICY_LOAD_FAILED = False
_POLICY_OBS_DIM = OBS_DIM  # Updated at load time based on the actual model

# Per-trade cache for the v2 augmentation. Encoder embedding + cross-market
# features are snapshotted at trade entry and reused for every subsequent
# step — they don't change as the trade progresses. Keyed by trade_id so
# concurrent trades each get their own snapshot. Small dict, freed on exit.
_V2_TRADE_CACHE: Dict[Any, Dict[str, np.ndarray]] = {}


def clear_v2_trade_cache(trade_id: Any = None):
    """Drop cached v2 features for a trade. Call this when the trade closes.
    Pass trade_id=None to wipe the whole cache."""
    global _V2_TRADE_CACHE
    if trade_id is None:
        _V2_TRADE_CACHE = {}
    else:
        _V2_TRADE_CACHE.pop(trade_id, None)


def init_rl_policy() -> bool:
    """Eager-load the PPO policy from disk. Returns True on success.

    Supports both v1 (obs_dim=172) and v2 (obs_dim=212 with encoder + cross-
    market features). Dim is sniffed from the loaded model's observation
    space and stashed in _POLICY_OBS_DIM so the build-obs path can append
    augmentation when the loaded policy expects it.
    """
    global _POLICY, _POLICY_LOAD_FAILED, _POLICY_OBS_DIM
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
        _POLICY_OBS_DIM = int(_POLICY.observation_space.shape[0])
        kind = "v2 (extended obs)" if _POLICY_OBS_DIM == OBS_DIM_EXTENDED else "v1"
        logging.info("rl.inference: PPO policy loaded from %s — %s obs_dim=%d",
                     MODEL_PATH, kind, _POLICY_OBS_DIM)
        return True
    except Exception as exc:
        logging.warning("rl.inference: failed to load policy: %s", exc)
        _POLICY_LOAD_FAILED = True
        return False


def _compute_v2_augmentation(
    bars_df, entry_time, entry_bar_idx: int,
    *, vix_override_df=None, mnq_override_df=None,
) -> np.ndarray:
    """Return a 40-dim vector: 32-dim encoder embedding + 8-dim cross-market
    features, in the same layout and rescaling the training env applies.
    Returns zeros on any failure so the policy degrades gracefully rather
    than refusing to score.

    vix_override_df / mnq_override_df: live-accumulator DataFrames from
    the bot (master_vix_df / master_mnq_df) that supplant the cached
    parquet when they contain fresher data."""
    enc = np.zeros(ENCODER_DIM, dtype=np.float32)
    cm_scaled = np.zeros(CROSS_MARKET_DIM, dtype=np.float32)
    # --- Encoder embedding at the entry bar ---
    try:
        import torch
        from rl.bar_encoder import BarEncoder, encode as _enc_fn
        cache = _compute_v2_augmentation.__dict__
        if "encoder" not in cache:
            ckpt_path = ROOT / "artifacts" / "signal_gate_2025" / "bar_encoder.pt"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                e = BarEncoder(seq_len=int(ckpt.get("seq_len", 60)),
                               embed_dim=int(ckpt.get("embed_dim", 32)))
                e.load_state_dict(ckpt["state_dict"])
                e.eval()
                cache["encoder"] = e
            else:
                cache["encoder"] = None
        e = cache["encoder"]
        if e is not None and 0 < entry_bar_idx < len(bars_df):
            enc = _enc_fn(e, bars_df, int(entry_bar_idx)).astype(np.float32)
    except Exception as exc:
        logging.debug("rl.inference: encoder failed: %s", exc)
    # --- Cross-market features at entry_time ---
    try:
        from rl.cross_market import get_cross_market_features, CROSS_MARKET_FEATURE_KEYS
        feats = get_cross_market_features(
            entry_time, mes_bars=bars_df,
            vix_override_df=vix_override_df,
            mnq_override_df=mnq_override_df,
        )
        cm = np.array([float(feats.get(k, 0.0)) for k in CROSS_MARKET_FEATURE_KEYS], dtype=np.float32)
        # Match TradeManagementEnv._build_obs rescaling exactly
        cm_scaled = np.array([
            cm[0] / 2.0,
            cm[1] / 5.0,
            (cm[2] - 0.5) * 2.0,
            cm[3] / 5.0,
            (cm[4] - 16.0) / 16.0,
            cm[5] / 3.0,
            cm[6] / 50.0,
            (cm[7] - 100.0) / 10.0,
        ], dtype=np.float32)
    except Exception as exc:
        logging.debug("rl.inference: cross-market failed: %s", exc)
    return np.concatenate([enc, cm_scaled]).astype(np.float32)


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
    # v2 obs inputs — only consumed when the loaded policy is extended-obs.
    # Ignored for v1 policies (so callers can pass them unconditionally).
    trade_id: Any = None,
    entry_time: Any = None,
    entry_bar_idx: Optional[int] = None,
    vix_override_df: Any = None,
    mnq_override_df: Any = None,
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
    parts = [tape, ts, reg, sess, kals, pt]
    if _POLICY_OBS_DIM == OBS_DIM_EXTENDED:
        # v2: append cached (or freshly computed) encoder + cross-market
        # augmentation. These are episode-level features fixed at entry,
        # so we cache per-trade and reuse across every step.
        aug = None
        if trade_id is not None and trade_id in _V2_TRADE_CACHE:
            aug = _V2_TRADE_CACHE[trade_id].get("aug")
        if aug is None and entry_time is not None and entry_bar_idx is not None:
            aug = _compute_v2_augmentation(
                bars_df, entry_time, int(entry_bar_idx),
                vix_override_df=vix_override_df,
                mnq_override_df=mnq_override_df,
            )
            if trade_id is not None:
                _V2_TRADE_CACHE[trade_id] = {"aug": aug}
        if aug is None:
            # Degrade gracefully if caller didn't pass v2 inputs — zeros
            # match the "no signal" regime rather than an assert crash.
            aug = np.zeros(ENCODER_DIM + CROSS_MARKET_DIM, dtype=np.float32)
        parts.append(aug)
    obs = np.concatenate(parts).astype(np.float32)
    np.clip(obs, -10.0, 10.0, out=obs)
    assert obs.shape == (_POLICY_OBS_DIM,), f"obs dim {obs.shape} != {_POLICY_OBS_DIM}"
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

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

PROFIT_FACTOR_CAP = 10.0


def gpu_backend_available() -> bool:
    return cp is not None


def _safe_profit_factor(gross_win: float, gross_loss: float) -> float:
    gross_win = float(gross_win)
    gross_loss = float(gross_loss)
    if gross_loss <= 0.0:
        if gross_win <= 0.0:
            return 0.0
        return float(PROFIT_FACTOR_CAP)
    pf = gross_win / gross_loss
    if not math.isfinite(pf) or pf < 0.0:
        return 0.0
    return float(min(pf, PROFIT_FACTOR_CAP))


def score_candidate_from_summary(summary: Dict[str, float], min_trades: int) -> float:
    trades = int(summary.get("trades", 0) or 0)
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    avg_pnl = float(summary.get("avg_pnl", 0.0) or 0.0)
    profit_factor = float(summary.get("profit_factor", 0.0) or 0.0)
    trade_weight = min(1.0, math.sqrt(trades / max(1, int(min_trades))))
    pf_adj = 0.0
    if math.isfinite(profit_factor):
        pf_adj = min(profit_factor, 3.0) / 3.0
    score = (avg_pnl * 0.7) + ((win_rate - 0.5) * 5.0) + (pf_adj * 0.3)
    return float(score * trade_weight)


def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        out = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), _finite_float(x, float(lo)))))


def _trade_confidence(trades: int, tau: float) -> float:
    t = max(0.0, float(trades))
    tau_v = max(1e-9, float(tau))
    return float(1.0 - math.exp(-t / tau_v))


def _robust_unit_from_center_scale(
    x: float,
    center: float,
    scale: float,
    lo: float = -2.0,
    hi: float = 2.0,
) -> float:
    scale_v = max(1e-9, float(scale))
    return _clip((float(x) - float(center)) / scale_v, float(lo), float(hi))


def _safe_block_metrics(blocks: Any) -> Dict[str, float]:
    rows = blocks if isinstance(blocks, list) else []
    if not rows:
        return {
            "block_count": 0,
            "profitable_block_ratio": 0.0,
            "worst_block_avg_pnl": 0.0,
            "worst_block_pf": 0.0,
            "block_avg_pnl_std": 0.0,
        }

    avg_vals = np.asarray([_finite_float((r or {}).get("avg_pnl", 0.0), 0.0) for r in rows], dtype=float)
    pf_vals = np.asarray([_finite_float((r or {}).get("profit_factor", 0.0), 0.0) for r in rows], dtype=float)
    if avg_vals.size <= 0:
        return {
            "block_count": 0,
            "profitable_block_ratio": 0.0,
            "worst_block_avg_pnl": 0.0,
            "worst_block_pf": 0.0,
            "block_avg_pnl_std": 0.0,
        }

    block_count = int(avg_vals.size)
    profitable_block_ratio = float(np.mean(avg_vals > 0.0))
    worst_block_avg_pnl = float(np.min(avg_vals))
    worst_block_pf = float(np.min(pf_vals)) if pf_vals.size > 0 else 0.0
    block_avg_pnl_std = float(np.std(avg_vals))
    return {
        "block_count": int(block_count),
        "profitable_block_ratio": float(profitable_block_ratio),
        "worst_block_avg_pnl": float(worst_block_avg_pnl),
        "worst_block_pf": float(worst_block_pf),
        "block_avg_pnl_std": float(block_avg_pnl_std),
    }


def compute_structural_rank_fields(row: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rank_cfg = cfg if isinstance(cfg, dict) else {}
    enabled = bool(rank_cfg.get("enabled", False))

    oos = row.get("OOS", {}) if isinstance(row.get("OOS", {}), dict) else {}
    block_stats = row.get("oos_blocks_stats", {}) if isinstance(row.get("oos_blocks_stats", {}), dict) else {}
    sl_dist = max(1e-9, abs(_finite_float(row.get("Best_SL", 0.0), 0.0)))

    oos_trades = int(max(0, _finite_float(oos.get("trades", 0), 0.0)))
    oos_avg_pnl = _finite_float(oos.get("avg_pnl", 0.0), 0.0)
    oos_profit_factor = _finite_float(oos.get("profit_factor", 0.0), 0.0)
    oos_profit_factor_capped = min(float(oos_profit_factor), 2.0)
    oos_win_rate = _finite_float(oos.get("win_rate", 0.0), 0.0)
    oos_drawdown_norm = max(0.0, _finite_float(oos.get("max_oos_drawdown_norm", 0.0), 0.0))
    oos_stop_like_share = max(0.0, _finite_float(oos.get("stop_like_share", 0.0), 0.0))
    oos_loss_share = max(0.0, _finite_float(oos.get("loss_share", 0.0), 0.0))
    oos_tail_p10 = _finite_float(oos.get("pnl_p10_worst", oos.get("pnl_p10", 0.0)), 0.0)
    oos_tail_p10_scaled = abs(float(oos_tail_p10)) / sl_dist
    oos_sharpe_like = _finite_float(oos.get("sharpe_like", 0.0), 0.0)

    block_metrics = _safe_block_metrics(block_stats.get("blocks"))
    block_count = int(block_metrics["block_count"])
    profitable_block_ratio = float(block_metrics["profitable_block_ratio"])
    worst_block_avg_pnl = float(block_metrics["worst_block_avg_pnl"])
    worst_block_pf = float(block_metrics["worst_block_pf"])
    block_avg_pnl_std = float(block_metrics["block_avg_pnl_std"])

    # Rolling mode may not populate per-block lists; use aggregate counters when available.
    if block_count <= 0:
        total_blocks = int(max(0, _finite_float(oos.get("count_total_blocks", 0), 0.0)))
        pos_blocks = int(max(0, _finite_float(oos.get("count_positive_blocks", 0), 0.0)))
        if total_blocks > 0:
            block_count = int(total_blocks)
            profitable_block_ratio = float(pos_blocks / max(1, total_blocks))
            worst_block_avg_pnl = float(oos_avg_pnl)
            worst_block_pf = float(oos_profit_factor)
            block_avg_pnl_std = float(_finite_float(oos.get("std_oos_avg_pnl", 0.0), 0.0))

    trade_conf_tau = max(1.0, _finite_float(rank_cfg.get("trade_conf_tau", 100), 100.0))
    trade_confidence = _trade_confidence(oos_trades, trade_conf_tau)

    avg_term = _clip(oos_avg_pnl / 2.0, -2.0, 2.0)
    pf_term = _clip((oos_profit_factor_capped - 1.0) / 0.5, -2.0, 2.0)
    wr_term = _clip((oos_win_rate - 0.50) / 0.10, -2.0, 2.0)
    pb_term = _clip((profitable_block_ratio - 0.50) / 0.25, -2.0, 2.0)
    wba_term = _clip(worst_block_avg_pnl / 1.5, -2.0, 2.0)
    wbpf_term = _clip((worst_block_pf - 1.0) / 0.30, -2.0, 2.0)
    dd_term = _clip(oos_drawdown_norm / 0.60, 0.0, 2.0)
    stop_term = _clip(oos_stop_like_share / 0.40, 0.0, 2.0)
    loss_term = _clip(oos_loss_share / 0.50, 0.0, 2.0)
    tail_term = _clip(oos_tail_p10_scaled / 1.0, 0.0, 2.0)
    bstd_term = _clip(block_avg_pnl_std / 1.25, 0.0, 2.0)
    sharpe_term = _clip(oos_sharpe_like / 2.0, -2.0, 2.0)

    w_cfg = rank_cfg.get("weights", {}) if isinstance(rank_cfg.get("weights", {}), dict) else {}
    w_avg = _finite_float(w_cfg.get("avg_pnl", 1.50), 1.50)
    w_pf = _finite_float(w_cfg.get("profit_factor", 1.00), 1.00)
    w_wr = _finite_float(w_cfg.get("win_rate", 0.50), 0.50)
    w_trade_conf = _finite_float(w_cfg.get("trade_confidence", 0.60), 0.60)
    w_pb = _finite_float(w_cfg.get("profitable_block_ratio", 1.00), 1.00)
    w_wba = _finite_float(w_cfg.get("worst_block_avg_pnl", 0.85), 0.85)
    w_wbpf = _finite_float(w_cfg.get("worst_block_pf", 0.35), 0.35)
    w_dd = _finite_float(w_cfg.get("drawdown_norm", -1.10), -1.10)
    w_stop = _finite_float(w_cfg.get("stop_like_share", -0.90), -0.90)
    w_loss = _finite_float(w_cfg.get("loss_share", -0.70), -0.70)
    w_tail = _finite_float(w_cfg.get("tail_p10", -0.80), -0.80)
    w_bstd = _finite_float(w_cfg.get("block_std", -0.80), -0.80)
    w_sharpe = _finite_float(w_cfg.get("sharpe_like", 0.20), 0.20)

    structural_score = (
        (w_avg * avg_term)
        + (w_pf * pf_term)
        + (w_wr * wr_term)
        + (w_trade_conf * trade_confidence)
        + (w_pb * pb_term)
        + (w_wba * wba_term)
        + (w_wbpf * wbpf_term)
        + (w_dd * dd_term)
        + (w_stop * stop_term)
        + (w_loss * loss_term)
        + (w_tail * tail_term)
        + (w_bstd * bstd_term)
        + (w_sharpe * sharpe_term)
    )

    # Keep penalties explicit for diagnostics.
    structural_penalty = (
        (abs(w_dd) * dd_term)
        + (abs(w_stop) * stop_term)
        + (abs(w_loss) * loss_term)
        + (abs(w_tail) * tail_term)
        + (abs(w_bstd) * bstd_term)
    )

    min_oos_trades = int(max(0, _finite_float(rank_cfg.get("min_oos_trades", 80), 80.0)))
    min_profitable_block_ratio = _finite_float(rank_cfg.get("min_profitable_block_ratio", 0.60), 0.60)
    min_worst_block_avg_pnl = _finite_float(rank_cfg.get("min_worst_block_avg_pnl", -0.25), -0.25)
    min_worst_block_pf = _finite_float(rank_cfg.get("min_worst_block_pf", 0.90), 0.90)
    max_oos_drawdown_norm = _finite_float(rank_cfg.get("max_oos_drawdown_norm", 0.80), 0.80)
    max_stop_like_share = _finite_float(rank_cfg.get("max_stop_like_share", 0.42), 0.42)
    max_loss_share = _finite_float(rank_cfg.get("max_loss_share", 0.55), 0.55)
    max_tail_p10_abs_sl_mult = _finite_float(rank_cfg.get("max_tail_p10_abs_sl_mult", 1.00), 1.00)

    fail_reasons: List[str] = []
    # Keep StructuralPass as a severe-risk diagnostic, not a shape-quality veto.
    if worst_block_avg_pnl < min_worst_block_avg_pnl:
        fail_reasons.append(f"worst_block_avg_pnl<{min_worst_block_avg_pnl:.2f}")
    if worst_block_pf < min_worst_block_pf:
        fail_reasons.append(f"worst_block_pf<{min_worst_block_pf:.2f}")
    if oos_drawdown_norm > max_oos_drawdown_norm:
        fail_reasons.append(f"oos_drawdown_norm>{max_oos_drawdown_norm:.2f}")

    advisory_reasons: List[str] = []
    if oos_trades < min_oos_trades:
        advisory_reasons.append(f"oos_trades<{min_oos_trades}")
    if profitable_block_ratio < min_profitable_block_ratio:
        advisory_reasons.append(f"profitable_block_ratio<{min_profitable_block_ratio:.2f}")
    if oos_stop_like_share > max_stop_like_share:
        advisory_reasons.append(f"stop_like_share>{max_stop_like_share:.2f}")
    if oos_loss_share > max_loss_share:
        advisory_reasons.append(f"loss_share>{max_loss_share:.2f}")
    if oos_tail_p10_scaled > max_tail_p10_abs_sl_mult:
        advisory_reasons.append(f"tail_p10_scaled>{max_tail_p10_abs_sl_mult:.2f}")

    structural_pass = True
    trust_reason = "enabled"
    if enabled:
        structural_pass = len(fail_reasons) == 0
        reason_parts: List[str] = []
        if fail_reasons:
            reason_parts.append("hard:" + ";".join(fail_reasons))
        if advisory_reasons:
            reason_parts.append("soft:" + ";".join(advisory_reasons))
        trust_reason = "pass" if not reason_parts else "|".join(reason_parts)
    else:
        structural_pass = True
        trust_reason = "robust_ranking_disabled"

    components = {
        "avg_term": float(avg_term),
        "pf_term": float(pf_term),
        "wr_term": float(wr_term),
        "trade_confidence": float(trade_confidence),
        "profitable_block_ratio_term": float(pb_term),
        "worst_block_avg_pnl_term": float(wba_term),
        "worst_block_pf_term": float(wbpf_term),
        "drawdown_term": float(dd_term),
        "stop_like_term": float(stop_term),
        "loss_term": float(loss_term),
        "tail_term": float(tail_term),
        "block_std_term": float(bstd_term),
        "sharpe_term": float(sharpe_term),
    }

    return {
        "StructuralScore": float(structural_score),
        "StructuralPass": bool(structural_pass),
        "ProfitableBlockRatio": float(profitable_block_ratio),
        "WorstBlockAvgPnL": float(worst_block_avg_pnl),
        "WorstBlockPF": float(worst_block_pf),
        "BlockAvgPnLStd": float(block_avg_pnl_std),
        "TailP10Scaled": float(oos_tail_p10_scaled),
        "StructuralPenalty": float(structural_penalty),
        "StructuralComponents": components,
        "StructuralTrustReason": str(trust_reason),
        # Lower-case aliases for runtime convenience.
        "structural_score": float(structural_score),
        "structural_pass": bool(structural_pass),
        "profitable_block_ratio": float(profitable_block_ratio),
        "worst_block_avg_pnl": float(worst_block_avg_pnl),
        "worst_block_pf": float(worst_block_pf),
        "block_avg_pnl_std": float(block_avg_pnl_std),
        "tail_p10_scaled": float(oos_tail_p10_scaled),
        "structural_penalty": float(structural_penalty),
    }


def _summarize_from_stats(
    trades: int,
    wins: int,
    sum_pnl: float,
    gross_win: float,
    gross_loss: float,
) -> Dict[str, float]:
    if trades <= 0:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
        }
    win_rate = wins / trades if trades else 0.0
    avg_pnl = sum_pnl / trades if trades else 0.0
    profit_factor = _safe_profit_factor(gross_win, gross_loss)
    return {
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "avg_pnl": float(avg_pnl),
        "profit_factor": float(profit_factor),
        "total_pnl": float(sum_pnl),
    }


def _extract_trade_arrays(entries: Sequence[Sequence[float]], trade_resolution: str) -> Tuple[np.ndarray, ...]:
    n = len(entries)
    if n <= 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_i

    entry_pos_1m = np.fromiter((int(t[0]) for t in entries), dtype=np.int64, count=n)
    entry_pos_tf = np.fromiter((int(t[1]) for t in entries), dtype=np.int64, count=n)
    entry_price = np.fromiter((float(t[2]) for t in entries), dtype=float, count=n)
    entry_time_ns = np.fromiter((int(t[3]) for t in entries), dtype=np.int64, count=n)

    if len(entries[0]) >= 6:
        end_pos_1m = np.fromiter((int(t[4]) for t in entries), dtype=np.int64, count=n)
        end_pos_tf = np.fromiter((int(t[5]) for t in entries), dtype=np.int64, count=n)
    else:
        end_pos_1m = entry_pos_1m.copy()
        end_pos_tf = entry_pos_tf.copy()

    if str(trade_resolution).lower() == "1m":
        entry_pos = entry_pos_1m
        end_pos = end_pos_1m
    else:
        entry_pos = entry_pos_tf
        end_pos = end_pos_tf
    return entry_pos, end_pos, entry_price, entry_time_ns


def _choose_chunk_size(num_trades: int, combo_count: int, max_window: int, *, use_gpu: bool) -> int:
    if num_trades <= 0:
        return 0
    target_cells = 48_000_000 if use_gpu else 14_000_000
    denom = max(1, int(combo_count) * max(1, int(max_window)))
    size = int(target_cells // denom)
    size = max(256, size)
    return min(int(num_trades), size)


def _build_cumulative_windows(
    high_arr,
    low_arr,
    entry_pos,
    end_pos,
    *,
    use_gpu: bool,
):
    xp = cp if use_gpu else np
    if entry_pos.shape[0] <= 0:
        return None, None, 0

    max_idx = int(min(high_arr.shape[0], low_arr.shape[0]) - 1)
    if max_idx < 0:
        return None, None, 0

    entry_pos = xp.clip(entry_pos, 0, max_idx)
    end_pos = xp.clip(end_pos, 0, max_idx)
    end_pos = xp.maximum(end_pos, entry_pos)

    lengths = (end_pos - entry_pos + 1).astype(xp.int64, copy=False)
    if use_gpu:
        max_window = int(cp.max(lengths).item()) if lengths.size > 0 else 0
    else:
        max_window = int(np.max(lengths)) if lengths.size > 0 else 0
    if max_window <= 0:
        return None, None, 0

    offsets = xp.arange(max_window, dtype=xp.int64)
    idx = entry_pos[:, None] + offsets[None, :]
    valid = offsets[None, :] < lengths[:, None]
    idx = xp.minimum(idx, max_idx)

    high_window = high_arr[idx]
    low_window = low_arr[idx]
    high_window = xp.where(valid, high_window, -xp.inf)
    low_window = xp.where(valid, low_window, xp.inf)

    # CuPy on some environments does not implement ufunc.accumulate for
    # maximum/minimum. Keep a backend-safe path for GPU and CPU.
    if use_gpu:
        cummax = xp.empty_like(high_window)
        cummin = xp.empty_like(low_window)
        cummax[:, 0] = high_window[:, 0]
        cummin[:, 0] = low_window[:, 0]
        for j in range(1, max_window):
            cummax[:, j] = xp.maximum(cummax[:, j - 1], high_window[:, j])
            cummin[:, j] = xp.minimum(cummin[:, j - 1], low_window[:, j])
    else:
        cummax = xp.maximum.accumulate(high_window, axis=1)
        cummin = xp.minimum.accumulate(low_window, axis=1)
    return cummax, cummin, max_window


def evaluate_grid_metrics(
    entries: Sequence[Sequence[float]],
    tp_pairs: Sequence[Tuple[float, float]],
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side: str,
    *,
    trade_resolution: str = "1m",
    assume_sl_first: bool = False,
    sl_tp_conflict: str = "ohlc",
    exit_at_horizon: str = "close",
    min_trades_for_score: int = 30,
    acceleration: str = "cpu",
) -> List[Dict[str, float]]:
    if not entries or not tp_pairs:
        return []

    side = str(side or "").upper()
    use_gpu = str(acceleration or "cpu").lower() == "gpu" and cp is not None
    combo_count = len(tp_pairs)
    sl_arr = np.array([float(sl) for sl, _ in tp_pairs], dtype=float)
    tp_arr = np.array([float(tp) for _, tp in tp_pairs], dtype=float)

    entry_pos_arr, end_pos_arr, entry_price_arr, _ = _extract_trade_arrays(entries, trade_resolution)
    max_pos = len(high) - 1
    if max_pos < 0:
        return []
    entry_pos_arr = np.minimum(entry_pos_arr, max_pos)
    end_pos_arr = np.minimum(end_pos_arr, max_pos)
    end_pos_arr = np.maximum(end_pos_arr, entry_pos_arr)

    num_trades = len(entries)
    if num_trades <= 0:
        return []

    window_span = int(np.max(end_pos_arr - entry_pos_arr + 1)) if num_trades else 0
    # CPU path uses combo-wise 2D scans (not a single 3D tensor), so we can
    # use a milder combo factor for chunk sizing.
    combo_for_chunk = combo_count if use_gpu else max(1, min(combo_count, 8))
    chunk_size = _choose_chunk_size(num_trades, combo_for_chunk, window_span, use_gpu=use_gpu)
    if chunk_size <= 0:
        return []

    if use_gpu:
        open_a = cp.asarray(open_)
        high_a = cp.asarray(high)
        low_a = cp.asarray(low)
        close_a = cp.asarray(close)
        sl_a = cp.asarray(sl_arr, dtype=cp.float64)
        tp_a = cp.asarray(tp_arr, dtype=cp.float64)
        sum_pnl = cp.zeros(combo_count, dtype=cp.float64)
        wins = cp.zeros(combo_count, dtype=cp.int64)
        gross_win = cp.zeros(combo_count, dtype=cp.float64)
        gross_loss = cp.zeros(combo_count, dtype=cp.float64)
    else:
        # Float32 is sufficient for tick-based comparisons and reduces memory bandwidth.
        open_a = np.asarray(open_, dtype=np.float32)
        high_a = np.asarray(high, dtype=np.float32)
        low_a = np.asarray(low, dtype=np.float32)
        close_a = np.asarray(close, dtype=np.float32)
        sl_a = np.asarray(sl_arr, dtype=np.float32)
        tp_a = np.asarray(tp_arr, dtype=np.float32)
        sum_pnl = np.zeros(combo_count, dtype=np.float64)
        wins = np.zeros(combo_count, dtype=np.int64)
        gross_win = np.zeros(combo_count, dtype=np.float64)
        gross_loss = np.zeros(combo_count, dtype=np.float64)

    xp = cp if use_gpu else np
    exit_close = str(exit_at_horizon).lower() == "close"
    conflict_mode = str(sl_tp_conflict or "").lower()

    for start in range(0, num_trades, chunk_size):
        stop = min(num_trades, start + chunk_size)
        epos_np = entry_pos_arr[start:stop]
        end_np = end_pos_arr[start:stop]
        eprice_np = entry_price_arr[start:stop]

        if use_gpu:
            epos = cp.asarray(epos_np, dtype=cp.int64)
            endp = cp.asarray(end_np, dtype=cp.int64)
            eprice = cp.asarray(eprice_np, dtype=cp.float64)
        else:
            epos = epos_np
            endp = end_np
            eprice = np.asarray(eprice_np, dtype=np.float32)

        cummax, cummin, window_len = _build_cumulative_windows(
            high_a,
            low_a,
            epos,
            endp,
            use_gpu=use_gpu,
        )
        if window_len <= 0:
            continue

        if exit_close:
            exit_price = close_a[endp]
            pnl_no_hit = (exit_price - eprice) if side == "LONG" else (eprice - exit_price)
        else:
            pnl_no_hit = xp.zeros(eprice.shape[0], dtype=xp.float64)

        if use_gpu:
            if side == "LONG":
                tp_levels = eprice[:, None] + tp_a[None, :]
                sl_levels = eprice[:, None] - sl_a[None, :]
                tp_hit = cummax[:, :, None] >= tp_levels[:, None, :]
                sl_hit = cummin[:, :, None] <= sl_levels[:, None, :]
            else:
                tp_levels = eprice[:, None] - tp_a[None, :]
                sl_levels = eprice[:, None] + sl_a[None, :]
                tp_hit = cummin[:, :, None] <= tp_levels[:, None, :]
                sl_hit = cummax[:, :, None] >= sl_levels[:, None, :]

            has_tp = xp.any(tp_hit, axis=1)
            has_sl = xp.any(sl_hit, axis=1)
            tp_idx = xp.argmax(tp_hit, axis=1)
            sl_idx = xp.argmax(sl_hit, axis=1)
            tp_idx = xp.where(has_tp, tp_idx, window_len)
            sl_idx = xp.where(has_sl, sl_idx, window_len)

            pnl = xp.broadcast_to(pnl_no_hit[:, None], (eprice.shape[0], combo_count)).astype(xp.float64, copy=True)
            tp_before = has_tp & (~has_sl | (tp_idx < sl_idx))
            sl_before = has_sl & (~has_tp | (sl_idx < tp_idx))
            equal = has_tp & has_sl & (tp_idx == sl_idx)

            pnl = xp.where(tp_before, tp_a[None, :], pnl)
            pnl = xp.where(sl_before, -sl_a[None, :], pnl)
            if conflict_mode == "ohlc":
                hit_pos = xp.minimum(epos[:, None] + tp_idx, endp[:, None])
                bar_open = open_a[hit_pos]
                bar_close = close_a[hit_pos]
                is_green = bar_close >= bar_open
                stop_first = is_green if side == "LONG" else ~is_green
                equal_stop = equal & stop_first
                equal_take = equal & (~stop_first)
                pnl = xp.where(equal_stop, -sl_a[None, :], pnl)
                pnl = xp.where(equal_take, tp_a[None, :], pnl)
            elif assume_sl_first:
                pnl = xp.where(equal, -sl_a[None, :], pnl)
            else:
                pnl = xp.where(equal, tp_a[None, :], pnl)

            sum_pnl += xp.sum(pnl, axis=0)
            wins += xp.sum(pnl > 0, axis=0, dtype=xp.int64)
            gross_win += xp.sum(xp.where(pnl > 0, pnl, 0.0), axis=0)
            gross_loss += xp.sum(xp.where(pnl < 0, -pnl, 0.0), axis=0)
        else:
            # CPU: evaluate each (SL,TP) combo over 2D windows to avoid building
            # a large 3D boolean tensor that is memory-bandwidth bound.
            for j in range(combo_count):
                if side == "LONG":
                    tp_hit = cummax >= (eprice + tp_a[j])[:, None]
                    sl_hit = cummin <= (eprice - sl_a[j])[:, None]
                else:
                    tp_hit = cummin <= (eprice - tp_a[j])[:, None]
                    sl_hit = cummax >= (eprice + sl_a[j])[:, None]

                has_tp = np.any(tp_hit, axis=1)
                has_sl = np.any(sl_hit, axis=1)
                tp_idx = np.where(has_tp, np.argmax(tp_hit, axis=1), window_len)
                sl_idx = np.where(has_sl, np.argmax(sl_hit, axis=1), window_len)

                pnl = np.asarray(pnl_no_hit, dtype=np.float64).copy()
                tp_before = has_tp & (~has_sl | (tp_idx < sl_idx))
                sl_before = has_sl & (~has_tp | (sl_idx < tp_idx))
                equal = has_tp & has_sl & (tp_idx == sl_idx)

                pnl[tp_before] = float(tp_a[j])
                pnl[sl_before] = -float(sl_a[j])
                if conflict_mode == "ohlc":
                    if np.any(equal):
                        hit_pos = np.minimum(epos + tp_idx, endp)
                        bar_open = open_a[hit_pos]
                        bar_close = close_a[hit_pos]
                        is_green = bar_close >= bar_open
                        stop_first = is_green if side == "LONG" else ~is_green
                        eq_stop = equal & stop_first
                        eq_take = equal & (~stop_first)
                        pnl[eq_stop] = -float(sl_a[j])
                        pnl[eq_take] = float(tp_a[j])
                elif assume_sl_first:
                    pnl[equal] = -float(sl_a[j])
                else:
                    pnl[equal] = float(tp_a[j])

                sum_pnl[j] += float(np.sum(pnl, dtype=np.float64))
                pos = pnl > 0.0
                neg = pnl < 0.0
                wins[j] += int(np.sum(pos))
                if np.any(pos):
                    gross_win[j] += float(np.sum(pnl[pos], dtype=np.float64))
                if np.any(neg):
                    gross_loss[j] += float(np.sum(-pnl[neg], dtype=np.float64))

    if use_gpu:
        wins_np = cp.asnumpy(wins)
        sum_pnl_np = cp.asnumpy(sum_pnl)
        gross_win_np = cp.asnumpy(gross_win)
        gross_loss_np = cp.asnumpy(gross_loss)
    else:
        wins_np = wins
        sum_pnl_np = sum_pnl
        gross_win_np = gross_win
        gross_loss_np = gross_loss

    rows: List[Dict[str, float]] = []
    for j in range(combo_count):
        summary = _summarize_from_stats(
            trades=num_trades,
            wins=int(wins_np[j]),
            sum_pnl=float(sum_pnl_np[j]),
            gross_win=float(gross_win_np[j]),
            gross_loss=float(gross_loss_np[j]),
        )
        summary["sl"] = float(sl_arr[j])
        summary["tp"] = float(tp_arr[j])
        summary["score"] = float(score_candidate_from_summary(summary, min_trades=min_trades_for_score))
        rows.append(summary)
    return rows


def evaluate_single_combo(
    entries: Sequence[Sequence[float]],
    sl_dist: float,
    tp_dist: float,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    side: str,
    *,
    trade_resolution: str = "1m",
    assume_sl_first: bool = False,
    sl_tp_conflict: str = "ohlc",
    exit_at_horizon: str = "close",
    acceleration: str = "cpu",
) -> Dict[str, object]:
    if not entries:
        return {
            "summary": _summarize_from_stats(0, 0, 0.0, 0.0, 0.0),
            "pnls": np.zeros(0, dtype=float),
            "entry_time_ns": np.zeros(0, dtype=np.int64),
        }

    side = str(side or "").upper()
    use_gpu = str(acceleration or "cpu").lower() == "gpu" and cp is not None
    sl_dist = float(sl_dist)
    tp_dist = float(tp_dist)
    entry_pos_arr, end_pos_arr, entry_price_arr, entry_time_ns = _extract_trade_arrays(entries, trade_resolution)

    max_pos = len(high) - 1
    entry_pos_arr = np.minimum(entry_pos_arr, max_pos)
    end_pos_arr = np.minimum(end_pos_arr, max_pos)
    end_pos_arr = np.maximum(end_pos_arr, entry_pos_arr)

    num_trades = len(entry_pos_arr)
    if num_trades <= 0:
        return {
            "summary": _summarize_from_stats(0, 0, 0.0, 0.0, 0.0),
            "pnls": np.zeros(0, dtype=float),
            "entry_time_ns": np.zeros(0, dtype=np.int64),
        }

    window_span = int(np.max(end_pos_arr - entry_pos_arr + 1)) if num_trades else 0
    chunk_size = _choose_chunk_size(num_trades, 1, window_span, use_gpu=use_gpu)
    if chunk_size <= 0:
        chunk_size = num_trades

    if use_gpu:
        open_a = cp.asarray(open_)
        high_a = cp.asarray(high)
        low_a = cp.asarray(low)
        close_a = cp.asarray(close)
    else:
        open_a = np.asarray(open_, dtype=float)
        high_a = np.asarray(high, dtype=float)
        low_a = np.asarray(low, dtype=float)
        close_a = np.asarray(close, dtype=float)

    xp = cp if use_gpu else np
    exit_close = str(exit_at_horizon).lower() == "close"
    conflict_mode = str(sl_tp_conflict or "").lower()
    pnls = np.zeros(num_trades, dtype=float)

    for start in range(0, num_trades, chunk_size):
        stop = min(num_trades, start + chunk_size)
        epos_np = entry_pos_arr[start:stop]
        end_np = end_pos_arr[start:stop]
        eprice_np = entry_price_arr[start:stop]

        if use_gpu:
            epos = cp.asarray(epos_np, dtype=cp.int64)
            endp = cp.asarray(end_np, dtype=cp.int64)
            eprice = cp.asarray(eprice_np, dtype=cp.float64)
        else:
            epos = epos_np
            endp = end_np
            eprice = eprice_np

        cummax, cummin, window_len = _build_cumulative_windows(
            high_a,
            low_a,
            epos,
            endp,
            use_gpu=use_gpu,
        )
        if window_len <= 0:
            continue

        if exit_close:
            exit_price = close_a[endp]
            pnl = (exit_price - eprice) if side == "LONG" else (eprice - exit_price)
        else:
            pnl = xp.zeros(eprice.shape[0], dtype=xp.float64)

        if side == "LONG":
            tp_hit = cummax >= (eprice + tp_dist)[:, None]
            sl_hit = cummin <= (eprice - sl_dist)[:, None]
        else:
            tp_hit = cummin <= (eprice - tp_dist)[:, None]
            sl_hit = cummax >= (eprice + sl_dist)[:, None]

        has_tp = xp.any(tp_hit, axis=1)
        has_sl = xp.any(sl_hit, axis=1)
        tp_idx = xp.argmax(tp_hit, axis=1)
        sl_idx = xp.argmax(sl_hit, axis=1)
        tp_idx = xp.where(has_tp, tp_idx, window_len)
        sl_idx = xp.where(has_sl, sl_idx, window_len)

        tp_before = has_tp & (~has_sl | (tp_idx < sl_idx))
        sl_before = has_sl & (~has_tp | (sl_idx < tp_idx))
        equal = has_tp & has_sl & (tp_idx == sl_idx)

        pnl = xp.where(tp_before, tp_dist, pnl)
        pnl = xp.where(sl_before, -sl_dist, pnl)
        if conflict_mode == "ohlc":
            hit_pos = xp.minimum(epos + tp_idx, endp)
            bar_open = open_a[hit_pos]
            bar_close = close_a[hit_pos]
            is_green = bar_close >= bar_open
            stop_first = is_green if side == "LONG" else ~is_green
            equal_stop = equal & stop_first
            equal_take = equal & (~stop_first)
            pnl = xp.where(equal_stop, -sl_dist, pnl)
            pnl = xp.where(equal_take, tp_dist, pnl)
        elif assume_sl_first:
            pnl = xp.where(equal, -sl_dist, pnl)
        else:
            pnl = xp.where(equal, tp_dist, pnl)

        if use_gpu:
            pnls[start:stop] = cp.asnumpy(pnl)
        else:
            pnls[start:stop] = np.asarray(pnl, dtype=float)

    sum_pnl = float(np.sum(pnls))
    wins = int(np.sum(pnls > 0))
    gross_win = float(np.sum(pnls[pnls > 0])) if pnls.size else 0.0
    gross_loss = float(-np.sum(pnls[pnls < 0])) if pnls.size else 0.0

    summary = _summarize_from_stats(
        trades=int(len(pnls)),
        wins=wins,
        sum_pnl=sum_pnl,
        gross_win=gross_win,
        gross_loss=gross_loss,
    )
    return {
        "summary": summary,
        "pnls": np.asarray(pnls, dtype=float),
        "entry_time_ns": np.asarray(entry_time_ns, dtype=np.int64),
    }


def compute_drawdown_stats(pnls: np.ndarray) -> Dict[str, float]:
    if pnls is None or len(pnls) <= 0:
        return {"max_drawdown": 0.0, "max_drawdown_norm": 0.0}
    equity = np.cumsum(np.asarray(pnls, dtype=float))
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    dd = peaks[1:] - equity
    max_dd = float(np.max(dd)) if len(dd) else 0.0
    denom = max(1.0, float(np.max(np.abs(equity)))) if len(equity) else 1.0
    return {
        "max_drawdown": float(max_dd),
        "max_drawdown_norm": float(max_dd / denom),
    }


def compute_oos_block_stats(
    entry_time_ns: np.ndarray,
    pnls: np.ndarray,
    *,
    block_freq: str = "Q",
    tz: str = "America/New_York",
) -> Dict[str, object]:
    if entry_time_ns is None or pnls is None or len(entry_time_ns) <= 0 or len(pnls) <= 0:
        dd = compute_drawdown_stats(np.zeros(0, dtype=float))
        return {
            "blocks": [],
            "mean_avg_pnl": 0.0,
            "std_avg_pnl": 0.0,
            "positive_blocks": 0,
            "max_drawdown": dd["max_drawdown"],
            "max_drawdown_norm": dd["max_drawdown_norm"],
            "sharpe_like": 0.0,
        }

    entry_ns = np.asarray(entry_time_ns, dtype=np.int64)
    pnl_arr = np.asarray(pnls, dtype=float)
    n = min(len(entry_ns), len(pnl_arr))
    if n <= 0:
        dd = compute_drawdown_stats(np.zeros(0, dtype=float))
        return {
            "blocks": [],
            "mean_avg_pnl": 0.0,
            "std_avg_pnl": 0.0,
            "positive_blocks": 0,
            "max_drawdown": dd["max_drawdown"],
            "max_drawdown_norm": dd["max_drawdown_norm"],
            "sharpe_like": 0.0,
        }

    entry_ns = entry_ns[:n]
    pnl_arr = pnl_arr[:n]
    if n > 1 and not bool(np.all(entry_ns[1:] >= entry_ns[:-1])):
        order = np.argsort(entry_ns, kind="stable")
        entry_ns = entry_ns[order]
        pnl_arr = pnl_arr[order]

    period_labels = (
        pd.to_datetime(entry_ns, utc=True)
        .tz_convert(tz)
        .tz_localize(None)
        .to_period(block_freq)
        .astype(str)
        .to_numpy(dtype=object)
    )

    if len(period_labels) <= 0:
        dd = compute_drawdown_stats(pnl_arr)
        pnl_std = float(np.std(pnl_arr)) if len(pnl_arr) else 0.0
        sharpe_like = float((np.mean(pnl_arr) / pnl_std) * math.sqrt(len(pnl_arr))) if pnl_std > 1e-12 else 0.0
        return {
            "blocks": [],
            "mean_avg_pnl": 0.0,
            "std_avg_pnl": 0.0,
            "positive_blocks": 0,
            "max_drawdown": float(dd["max_drawdown"]),
            "max_drawdown_norm": float(dd["max_drawdown_norm"]),
            "sharpe_like": sharpe_like,
        }

    boundaries = np.flatnonzero(period_labels[1:] != period_labels[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(pnl_arr)]))

    blocks: List[Dict[str, float]] = []
    block_avg_vals = np.empty(len(starts), dtype=float)
    positive_blocks = 0
    for i, (s, e) in enumerate(zip(starts, ends)):
        vals = pnl_arr[int(s) : int(e)]
        trades = int(len(vals))
        wins = int(np.sum(vals > 0))
        sum_pnl = float(np.sum(vals))
        avg_pnl = float(sum_pnl / trades) if trades else 0.0
        gross_win = float(np.sum(vals[vals > 0])) if trades else 0.0
        gross_loss = float(-np.sum(vals[vals < 0])) if trades else 0.0
        pf = _safe_profit_factor(gross_win, gross_loss)
        if sum_pnl > 0.0:
            positive_blocks += 1
        block_avg_vals[i] = avg_pnl
        blocks.append(
            {
                "block": str(period_labels[int(s)]),
                "trades": trades,
                "wins": wins,
                "win_rate": float(wins / trades) if trades else 0.0,
                "avg_pnl": avg_pnl,
                "total_pnl": sum_pnl,
                "profit_factor": float(pf),
            }
        )

    block_avg = np.asarray(block_avg_vals, dtype=float)
    mean_avg = float(np.mean(block_avg)) if len(block_avg) else 0.0
    std_avg = float(np.std(block_avg)) if len(block_avg) else 0.0
    dd = compute_drawdown_stats(pnl_arr)
    pnl_std = float(np.std(pnl_arr)) if len(pnl_arr) else 0.0
    if pnl_std > 1e-12:
        sharpe_like = float((np.mean(pnl_arr) / pnl_std) * math.sqrt(len(pnl_arr)))
    else:
        sharpe_like = 0.0
    return {
        "blocks": blocks,
        "mean_avg_pnl": mean_avg,
        "std_avg_pnl": std_avg,
        "positive_blocks": positive_blocks,
        "max_drawdown": float(dd["max_drawdown"]),
        "max_drawdown_norm": float(dd["max_drawdown_norm"]),
        "sharpe_like": sharpe_like,
    }


def stability_weighted_score(
    mean_oos_avg_pnl: float,
    std_oos_avg_pnl: float,
    max_oos_drawdown_norm: float,
    *,
    lambda_std: float,
    gamma_dd: float,
) -> float:
    return float(
        float(mean_oos_avg_pnl)
        - (float(lambda_std) * float(std_oos_avg_pnl))
        - (float(gamma_dd) * float(max_oos_drawdown_norm))
    )


def select_plateau_candidate(
    rows: Sequence[Dict[str, float]],
    sl_values: Sequence[float],
    tp_values: Sequence[float],
    *,
    min_neighbors: int = 4,
    neighbor_def: str = "adjacent_grid",
    min_plateau_score: float = 0.0,
) -> Optional[Dict[str, float]]:
    if not rows:
        return None
    if str(neighbor_def).lower() != "adjacent_grid":
        raise ValueError(f"Unsupported plateau neighbor_def: {neighbor_def}")

    sl_idx = {float(v): i for i, v in enumerate(sl_values)}
    tp_idx = {float(v): i for i, v in enumerate(tp_values)}

    grid: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in rows:
        s = float(row.get("sl", 0.0))
        t = float(row.get("tp", 0.0))
        if s not in sl_idx or t not in tp_idx:
            continue
        grid[(sl_idx[s], tp_idx[t])] = dict(row)

    if not grid:
        return None

    best: Optional[Dict[str, float]] = None
    offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]

    for (i, j), center_row in grid.items():
        cluster: List[Dict[str, float]] = []
        for di, dj in offsets:
            key = (i + di, j + dj)
            row = grid.get(key)
            if row is not None:
                cluster.append(row)

        neighbor_count = max(0, len(cluster) - 1)
        if neighbor_count < int(min_neighbors):
            continue

        scores = np.asarray([float(r.get("score", 0.0) or 0.0) for r in cluster], dtype=float)
        if scores.size <= 0:
            continue
        cluster_score = float(np.percentile(scores, 25))
        if cluster_score < float(min_plateau_score):
            continue

        candidate = dict(center_row)
        candidate["plateau_cluster_score"] = cluster_score
        candidate["plateau_neighbors"] = float(neighbor_count)
        candidate["selected_by"] = "plateau"

        if best is None:
            best = candidate
            continue

        if float(candidate["plateau_cluster_score"]) > float(best.get("plateau_cluster_score", -1e18)):
            best = candidate
            continue
        if float(candidate["plateau_cluster_score"]) < float(best.get("plateau_cluster_score", -1e18)):
            continue

        if float(candidate.get("score", 0.0)) > float(best.get("score", 0.0)):
            best = candidate
            continue
        if float(candidate.get("score", 0.0)) < float(best.get("score", 0.0)):
            continue

        if float(candidate.get("avg_pnl", 0.0)) > float(best.get("avg_pnl", 0.0)):
            best = candidate

    return best

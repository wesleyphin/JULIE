import argparse
import json
import logging
import math
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from aetherflow_base_cache import (
    DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    validate_full_manifold_base_features_path,
)
from aetherflow_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, build_feature_frame, ensure_feature_columns
from config import (
    CONFIG,
    append_artifact_suffix,
    get_experimental_training_window,
    resolve_artifact_suffix,
)
from train_manifold_strategy import (
    DEFAULT_OUTRIGHT_SYMBOL_REGEX,
    _build_model,
    _coerce_session_ids,
    _filter_range,
    _load_bars,
    _parse_allowed_sessions,
    _parse_date,
    _sample_weights,
    _session_name_from_id,
)


def _load_cached_base_features(path: Path) -> pd.DataFrame:
    required = set(BASE_FEATURE_COLUMNS)
    validate_full_manifold_base_features_path(path, required)
    data = pd.read_parquet(path, columns=sorted(required))
    missing = sorted(col for col in required if col not in data.columns)
    if missing:
        raise RuntimeError(f"Cached manifold features missing columns: {', '.join(missing)}")
    return data.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_cached_training_data(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data = ensure_feature_columns(data)
    required = set(FEATURE_COLUMNS) | {"label", "net_points", "candidate_side", "setup_family"}
    missing = sorted(col for col in required if col not in data.columns)
    if missing:
        raise RuntimeError(f"Cached AetherFlow training parquet missing columns: {', '.join(missing)}")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "net_points"])
    data["label"] = data["label"].astype(int)
    return data


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def _serialize_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_json_value(v) for v in value]
    return value


def _default_checkpoint_dir(out_dir: Path, features_path: Path) -> Path:
    return out_dir / f"{features_path.stem}_checkpoint"


def _load_checkpoint_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    return payload


def _save_checkpoint_state(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(_serialize_json_value(payload), indent=2))


def _utc_now_iso() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def _simulate_candidate_outcome(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    entry_pos: int,
    side: int,
    sl_points: float,
    tp_points: float,
    horizon_bars: int,
) -> float:
    n = int(len(open_arr))
    if entry_pos < 0 or (entry_pos + 1) >= n:
        return math.nan
    entry_idx = entry_pos + 1
    entry_price = float(open_arr[entry_idx])
    if not np.isfinite(entry_price):
        return math.nan
    horizon_end = min(n - 1, entry_idx + max(1, int(horizon_bars)) - 1)

    if int(side) > 0:
        stop_price = entry_price - float(sl_points)
        take_price = entry_price + float(tp_points)
    else:
        stop_price = entry_price + float(sl_points)
        take_price = entry_price - float(tp_points)

    for j in range(entry_idx, horizon_end + 1):
        hi = float(high_arr[j])
        lo = float(low_arr[j])
        if int(side) > 0:
            hit_stop = lo <= stop_price
            hit_take = hi >= take_price
        else:
            hit_stop = hi >= stop_price
            hit_take = lo <= take_price
        if hit_stop and hit_take:
            return -float(sl_points)
        if hit_stop:
            return -float(sl_points)
        if hit_take:
            return float(tp_points)

    exit_price = float(close_arr[horizon_end])
    if not np.isfinite(exit_price):
        return math.nan
    return float(int(side)) * (exit_price - entry_price)


def _label_candidates(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    fees_points: float,
) -> pd.DataFrame:
    candidates = features.loc[pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0].copy()
    if candidates.empty:
        return pd.DataFrame(columns=list(features.columns) + ["gross_points", "net_points", "label"])

    positions = bars.index.get_indexer(candidates.index)
    valid_mask = positions >= 0
    candidates = candidates.loc[valid_mask].copy()
    positions = positions[valid_mask]
    if candidates.empty:
        return pd.DataFrame(columns=list(features.columns) + ["gross_points", "net_points", "label"])

    open_arr = bars["open"].to_numpy(dtype=float)
    high_arr = bars["high"].to_numpy(dtype=float)
    low_arr = bars["low"].to_numpy(dtype=float)
    close_arr = bars["close"].to_numpy(dtype=float)
    side_arr = (
        pd.to_numeric(candidates.get("candidate_side"), errors="coerce")
        .fillna(0.0)
        .round()
        .astype(int)
        .to_numpy(dtype=np.int8)
    )
    atr_arr = pd.to_numeric(candidates.get("atr14"), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    sl_mult_arr = pd.to_numeric(candidates.get("setup_sl_mult"), errors="coerce").fillna(1.1).to_numpy(dtype=float)
    tp_mult_arr = pd.to_numeric(candidates.get("setup_tp_mult"), errors="coerce").fillna(2.0).to_numpy(dtype=float)
    horizon_arr = (
        pd.to_numeric(candidates.get("setup_horizon_bars"), errors="coerce")
        .fillna(16.0)
        .round()
        .astype(int)
        .to_numpy(dtype=np.int16)
    )
    sl_points_arr = np.clip(atr_arr * sl_mult_arr, 1.0, 8.0)
    tp_floor_arr = np.maximum(sl_points_arr * 1.2, 1.5)
    tp_points_arr = np.clip(atr_arr * tp_mult_arr, tp_floor_arr, 16.0)

    gross_points = np.full(len(candidates), np.nan, dtype=float)
    net_points = np.full(len(candidates), np.nan, dtype=float)
    labels = np.full(len(candidates), np.nan, dtype=float)

    for i in range(len(candidates)):
        gross = _simulate_candidate_outcome(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            entry_pos=int(positions[i]),
            side=int(side_arr[i]),
            sl_points=float(sl_points_arr[i]),
            tp_points=float(tp_points_arr[i]),
            horizon_bars=int(max(6, int(horizon_arr[i]))),
        )
        if not np.isfinite(gross):
            continue
        net = float(gross) - float(fees_points)
        gross_points[i] = float(gross)
        net_points[i] = float(net)
        labels[i] = 1.0 if net > 0.0 else 0.0

    candidates["gross_points"] = gross_points
    candidates["net_points"] = net_points
    candidates["label"] = labels
    candidates = candidates.dropna(subset=["gross_points", "net_points", "label"])
    candidates["label"] = candidates["label"].astype(int)
    return candidates


def _evaluate_threshold(
    prob_success: np.ndarray,
    net_points: np.ndarray,
    threshold: float,
    *,
    sides: Optional[np.ndarray] = None,
    session_ids: Optional[np.ndarray] = None,
    allowed_session_ids: Optional[set[int]] = None,
) -> Dict:
    trade_mask = np.asarray(prob_success >= float(threshold), dtype=bool)
    if session_ids is not None and allowed_session_ids:
        allow = np.isin(np.asarray(session_ids, dtype=np.int16), np.asarray(sorted(allowed_session_ids), dtype=np.int16))
        trade_mask = trade_mask & allow
    trades = int(np.sum(trade_mask))
    if trades <= 0:
        return {
            "trade_count": 0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "long_share": 0.0,
            "short_share": 0.0,
        }
    pnl = np.asarray(net_points, dtype=float)[trade_mask]
    side_arr = np.asarray(sides if sides is not None else np.zeros_like(prob_success), dtype=float)
    return {
        "trade_count": trades,
        "avg_pnl": float(np.mean(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "win_rate": float(np.mean(pnl > 0.0)),
        "long_share": float(np.mean(side_arr[trade_mask] > 0.0)),
        "short_share": float(np.mean(side_arr[trade_mask] < 0.0)),
    }


def _evaluate_threshold_folds(
    prob_success: np.ndarray,
    net_points: np.ndarray,
    threshold: float,
    *,
    folds: int,
    sides: Optional[np.ndarray] = None,
    session_ids: Optional[np.ndarray] = None,
    allowed_session_ids: Optional[set[int]] = None,
) -> list[Dict]:
    n = int(len(prob_success))
    folds = max(1, int(folds))
    if n <= 0 or folds <= 1:
        return []
    edges = np.linspace(0, n, folds + 1, dtype=int)
    out: list[Dict] = []
    for idx in range(folds):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end - start <= 0:
            continue
        sess_slice = None if session_ids is None else np.asarray(session_ids[start:end], dtype=np.int16)
        side_slice = None if sides is None else np.asarray(sides[start:end], dtype=float)
        out.append(
            _evaluate_threshold(
                prob_success[start:end],
                net_points[start:end],
                threshold,
                sides=side_slice,
                session_ids=sess_slice,
                allowed_session_ids=allowed_session_ids,
            )
        )
    return out


def _search_threshold(
    prob_success: np.ndarray,
    net_points: np.ndarray,
    *,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    min_trades: int,
    min_coverage: float,
    coverage_target: float,
    coverage_penalty: float,
    val_folds: int,
    min_fold_trades: int,
    min_positive_folds: int,
    objective_mean_weight: float,
    objective_worst_weight: float,
    objective_total_weight: float,
    objective_std_penalty: float,
    sides: Optional[np.ndarray] = None,
    session_ids: Optional[np.ndarray] = None,
    allowed_session_ids: Optional[set[int]] = None,
) -> Dict:
    best: Optional[Dict] = None
    best_any: Optional[Dict] = None
    total_rows = max(1, int(len(prob_success)))
    thr = float(thr_min)
    while thr <= float(thr_max) + 1e-12:
        stats = _evaluate_threshold(
            prob_success,
            net_points,
            threshold=thr,
            sides=sides,
            session_ids=session_ids,
            allowed_session_ids=allowed_session_ids,
        )
        trades = int(stats["trade_count"])
        coverage = float(trades) / float(total_rows)
        if trades >= int(min_trades) and coverage >= float(min_coverage):
            fold_stats = _evaluate_threshold_folds(
                prob_success,
                net_points,
                threshold=thr,
                folds=int(val_folds),
                sides=sides,
                session_ids=session_ids,
                allowed_session_ids=allowed_session_ids,
            )
            valid_folds = [row for row in fold_stats if int(row.get("trade_count", 0)) >= int(min_fold_trades)]
            positive_folds = int(sum(1 for row in valid_folds if float(row.get("total_pnl", 0.0)) > 0.0))
            if valid_folds and positive_folds < int(min_positive_folds):
                thr += float(thr_step)
                continue
            fold_avg = np.asarray([float(row.get("avg_pnl", 0.0)) for row in valid_folds], dtype=float)
            mean_fold_avg = float(np.mean(fold_avg)) if fold_avg.size else float(stats["avg_pnl"])
            worst_fold_avg = float(np.min(fold_avg)) if fold_avg.size else float(stats["avg_pnl"])
            std_fold_avg = float(np.std(fold_avg)) if fold_avg.size else 0.0
            total_component = float(stats["avg_pnl"]) * coverage * 100.0
            coverage_gap = abs(coverage - float(coverage_target)) if float(coverage_target) > 0.0 else 0.0
            score = (
                float(objective_mean_weight) * mean_fold_avg
                + float(objective_worst_weight) * worst_fold_avg
                + float(objective_total_weight) * total_component
                - float(objective_std_penalty) * std_fold_avg
                - float(coverage_penalty) * coverage_gap
            )
            candidate = {
                "threshold": float(thr),
                "score": float(score),
                "coverage": coverage,
                "mean_fold_avg_pnl": mean_fold_avg,
                "worst_fold_avg_pnl": worst_fold_avg,
                "fold_std_avg_pnl": std_fold_avg,
                "positive_folds": positive_folds,
                "evaluated_folds": int(len(valid_folds)),
                **stats,
            }
            if best_any is None or candidate["score"] > best_any["score"] + 1e-12 or (
                abs(candidate["score"] - best_any["score"]) <= 1e-12 and candidate["total_pnl"] > best_any["total_pnl"] + 1e-12
            ):
                best_any = dict(candidate)
            if best is None or candidate["score"] > best["score"] + 1e-12 or (
                abs(candidate["score"] - best["score"]) <= 1e-12 and candidate["total_pnl"] > best["total_pnl"] + 1e-12
            ):
                best = candidate
        thr += float(thr_step)

    if best is None and best_any is not None:
        best_any["fallback_used"] = True
        return best_any
    if best is None:
        raise RuntimeError("No valid AetherFlow threshold found.")
    return best


def _setup_mix(frame: pd.DataFrame) -> Dict[str, int]:
    if frame is None or frame.empty or "setup_family" not in frame.columns:
        return {}
    counts = frame["setup_family"].astype(str).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items() if str(k)}


def _build_labeled_candidates_with_checkpoint(
    *,
    base_features: pd.DataFrame,
    bars: pd.DataFrame,
    features_path: Path,
    checkpoint_dir: Path,
    chunk_size: int,
    overlap_rows: int,
    fees_points: float,
    resume: bool,
    force_rebuild: bool,
) -> tuple[pd.DataFrame, int]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = checkpoint_dir / "labeled_parts"
    state_path = checkpoint_dir / "state.json"

    if features_path.exists() and not force_rebuild and not resume:
        logging.info("Existing labeled features found, loading: %s", features_path)
        cached = _load_cached_training_data(features_path)
        return cached, int(len(base_features))

    if force_rebuild:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        parts_dir.mkdir(parents=True, exist_ok=True)
    else:
        parts_dir.mkdir(parents=True, exist_ok=True)

    if resume and features_path.exists() and state_path.exists():
        existing_state = _load_checkpoint_state(state_path)
        if bool(existing_state.get("complete")):
            logging.info("Checkpoint marked complete, loading labeled features: %s", features_path)
            cached = _load_cached_training_data(features_path)
            return cached, int(existing_state.get("feature_rows_total", len(base_features)))

    if not resume and state_path.exists():
        raise RuntimeError(
            f"Checkpoint state already exists at {state_path}. "
            "Use --resume-label-build to continue or --force-rebuild-features to restart."
        )

    total_rows = int(len(base_features))
    if total_rows <= 0:
        raise RuntimeError("No cached base features available for AetherFlow chunk build.")
    chunk_size = max(10_000, int(chunk_size))
    overlap_rows = max(80, int(overlap_rows))
    total_chunks = int(math.ceil(float(total_rows) / float(chunk_size)))
    created_at = _utc_now_iso()
    state = {
        "created_at": created_at,
        "updated_at": created_at,
        "complete": False,
        "feature_rows_total": total_rows,
        "chunk_size": int(chunk_size),
        "overlap_rows": int(overlap_rows),
        "total_chunks": total_chunks,
        "next_chunk": 0,
        "completed_chunks": 0,
        "candidate_rows_total": 0,
        "parts": [],
    }
    if resume and state_path.exists():
        state = _load_checkpoint_state(state_path)
        state.setdefault("feature_rows_total", total_rows)
        state.setdefault("chunk_size", int(chunk_size))
        state.setdefault("overlap_rows", int(overlap_rows))
        state.setdefault("total_chunks", total_chunks)
        state.setdefault("next_chunk", 0)
        state.setdefault("completed_chunks", 0)
        state.setdefault("candidate_rows_total", 0)
        state.setdefault("parts", [])
        logging.info(
            "Resuming AetherFlow label build: chunk %d/%d complete_chunks=%d candidate_rows=%d",
            int(state.get("next_chunk", 0)),
            int(state.get("total_chunks", total_chunks)),
            int(state.get("completed_chunks", 0)),
            int(state.get("candidate_rows_total", 0)),
        )
    else:
        _save_checkpoint_state(state_path, state)

    build_started = time.time()
    completed_parts = set(str(p) for p in state.get("parts", []))
    for chunk_idx in range(int(state.get("next_chunk", 0)), total_chunks):
        chunk_start = int(chunk_idx * chunk_size)
        chunk_end = min(total_rows, int(chunk_start + chunk_size))
        context_start = max(0, int(chunk_start - overlap_rows))
        part_name = f"part_{chunk_idx:05d}.parquet"
        part_path = parts_dir / part_name
        if part_path.exists() and part_name in completed_parts:
            state["next_chunk"] = int(chunk_idx + 1)
            continue

        chunk_started = time.time()
        base_slice = base_features.iloc[context_start:chunk_end]
        feature_slice = build_feature_frame(base_features=base_slice)
        core_offset = int(chunk_start - context_start)
        core_rows = int(chunk_end - chunk_start)
        feature_core = feature_slice.iloc[core_offset : core_offset + core_rows]
        labeled = _label_candidates(feature_core, bars, fees_points=float(fees_points))
        labeled = labeled.sort_index()
        labeled.to_parquet(part_path, index=True)

        elapsed = time.time() - build_started
        processed_rows = int(chunk_end)
        rows_per_sec = float(processed_rows) / max(1e-9, elapsed)
        eta_seconds = float(total_rows - processed_rows) / max(1e-9, rows_per_sec)
        chunk_elapsed = time.time() - chunk_started

        state["next_chunk"] = int(chunk_idx + 1)
        state["completed_chunks"] = int(chunk_idx + 1)
        state["candidate_rows_total"] = int(state.get("candidate_rows_total", 0)) + int(len(labeled))
        state["updated_at"] = _utc_now_iso()
        parts = [str(p) for p in state.get("parts", []) if str(p) != part_name]
        parts.append(part_name)
        state["parts"] = parts
        completed_parts.add(part_name)
        _save_checkpoint_state(state_path, state)

        logging.info(
            "AetherFlow label build chunk %d/%d rows=%d:%d candidate_rows=%d cumulative=%d progress=%.2f%% chunk_time=%s eta=%s",
            int(chunk_idx + 1),
            int(total_chunks),
            int(chunk_start),
            int(chunk_end),
            int(len(labeled)),
            int(state["candidate_rows_total"]),
            100.0 * float(processed_rows) / float(total_rows),
            _format_eta(chunk_elapsed),
            _format_eta(eta_seconds),
        )

    part_files = sorted(parts_dir.glob("part_*.parquet"))
    if not part_files:
        raise RuntimeError(f"No AetherFlow checkpoint parts were produced under {parts_dir}")
    logging.info("Concatenating %d AetherFlow labeled parts...", len(part_files))
    frames = [pd.read_parquet(part) for part in part_files]
    data = pd.concat(frames, axis=0).sort_index()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "net_points"])
    data["label"] = data["label"].astype(int)
    data.to_parquet(features_path, index=True)

    state["complete"] = True
    state["updated_at"] = _utc_now_iso()
    state["feature_rows_total"] = int(total_rows)
    state["candidate_rows_total"] = int(len(data))
    _save_checkpoint_state(state_path, state)
    logging.info(
        "AetherFlow label build complete: feature_rows=%d candidate_rows=%d coverage=%.4f elapsed=%s saved=%s",
        int(total_rows),
        int(len(data)),
        float(len(data)) / float(total_rows),
        _format_eta(time.time() - build_started),
        features_path,
    )
    return data, int(total_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AetherFlow transition strategy.")
    parser.add_argument("--input", default="es_master_outrights.parquet")
    parser.add_argument("--base-features", default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    parser.add_argument("--start", dest="start", default=None)
    parser.add_argument("--end", dest="end", default=None)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--model-file", default="model_aetherflow_v1.pkl")
    parser.add_argument("--thresholds-file", default="aetherflow_thresholds_v1.json")
    parser.add_argument("--metrics-file", default="aetherflow_metrics_v1.json")
    parser.add_argument("--features-parquet", default="aetherflow_features_v1.parquet")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--thr-min", type=float, default=0.53)
    parser.add_argument("--thr-max", type=float, default=0.82)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--min-val-trades", type=int, default=120)
    parser.add_argument("--min-val-coverage", type=float, default=0.02)
    parser.add_argument("--coverage-target", type=float, default=0.08)
    parser.add_argument("--coverage-penalty", type=float, default=0.20)
    parser.add_argument("--val-folds", type=int, default=4)
    parser.add_argument("--min-fold-trades", type=int, default=25)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--objective-mean-weight", type=float, default=0.55)
    parser.add_argument("--objective-worst-weight", type=float, default=0.25)
    parser.add_argument("--objective-total-weight", type=float, default=0.20)
    parser.add_argument("--objective-std-penalty", type=float, default=0.10)
    parser.add_argument("--model", choices=["hgb", "rf", "logreg"], default="hgb")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=175000)
    parser.add_argument("--chunk-overlap", type=int, default=160)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-label-build", action="store_true")
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--fees-points", type=float, default=None)
    parser.add_argument("--allowed-session-ids", default="1,2,3")
    parser.add_argument("--allow-spreads", action="store_true")
    parser.add_argument("--symbol-regex", default=DEFAULT_OUTRIGHT_SYMBOL_REGEX)
    parser.add_argument("--min-valid-price", type=float, default=100.0)
    parser.add_argument("--experimental-window", action="store_true")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--reuse-cached-base", action="store_true")
    parser.add_argument("--reuse-features-parquet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    exp_enabled = bool(args.experimental_window)
    start_arg = args.start
    end_arg = args.end
    if exp_enabled:
        exp_start, exp_end = get_experimental_training_window()
        start_arg = exp_start
        end_arg = exp_end
        logging.info("Experimental window enabled: %s -> %s", start_arg, end_arg)

    artifact_suffix = resolve_artifact_suffix(args.artifact_suffix, exp_enabled)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = append_artifact_suffix(str(args.model_file), artifact_suffix) if artifact_suffix else str(args.model_file)
    thresholds_file = append_artifact_suffix(str(args.thresholds_file), artifact_suffix) if artifact_suffix else str(args.thresholds_file)
    metrics_file = append_artifact_suffix(str(args.metrics_file), artifact_suffix) if artifact_suffix else str(args.metrics_file)
    features_file = append_artifact_suffix(str(args.features_parquet), artifact_suffix) if artifact_suffix else str(args.features_parquet)

    model_path = out_dir / model_file
    thresholds_path = out_dir / thresholds_file
    metrics_path = out_dir / metrics_file
    features_path = out_dir / features_file

    features: Optional[pd.DataFrame] = None
    feature_row_count = 0
    if args.reuse_features_parquet:
        if not features_path.exists():
            raise RuntimeError(f"Requested --reuse-features-parquet but file does not exist: {features_path}")
        logging.info("Loading cached AetherFlow labeled training data: %s", features_path)
        data = _load_cached_training_data(features_path)
        logging.info("Cached AetherFlow rows=%d range=%s -> %s", len(data), data.index.min(), data.index.max())
    else:
        bars = _load_bars(
            Path(args.input),
            allow_spreads=bool(args.allow_spreads),
            symbol_regex=str(args.symbol_regex or "").strip() or None,
            min_valid_price=float(args.min_valid_price),
        ).sort_index()
        start_ts = _parse_date(start_arg, is_end=False)
        end_ts = _parse_date(end_arg, is_end=True)
        bars = _filter_range(bars, start_ts, end_ts)
        if args.max_rows and int(args.max_rows) > 0:
            bars = bars.iloc[-int(args.max_rows) :]
        if bars.empty:
            raise RuntimeError("No bars available after filtering.")
        logging.info("Loaded bars: rows=%d range=%s -> %s", len(bars), bars.index.min(), bars.index.max())

        base_features: Optional[pd.DataFrame] = None
        base_path = Path(args.base_features)
        use_cached_base = base_path.exists()
        if use_cached_base:
            logging.info("Loading cached manifold base features: %s", base_path)
            base_features = _load_cached_base_features(base_path)
            base_features = base_features.loc[(base_features.index >= bars.index.min()) & (base_features.index <= bars.index.max())]
            logging.info("Cached manifold base rows=%d", len(base_features))

        if args.fees_points is not None:
            fees_points = float(args.fees_points)
        else:
            risk_cfg = CONFIG.get("RISK", {}) or {}
            point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
            fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
            fees_points = (fees_per_side * 2.0) / max(1e-9, point_value)
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _default_checkpoint_dir(out_dir, features_path)

        if base_features is not None and not base_features.empty:
            data, feature_row_count = _build_labeled_candidates_with_checkpoint(
                base_features=base_features,
                bars=bars,
                features_path=features_path,
                checkpoint_dir=checkpoint_dir,
                chunk_size=int(args.chunk_size),
                overlap_rows=int(args.chunk_overlap),
                fees_points=float(fees_points),
                resume=bool(args.resume_label_build),
                force_rebuild=bool(args.force_rebuild_features),
            )
            logging.info(
                "AetherFlow labeled candidates loaded from checkpointed build: rows=%d coverage=%.4f setups=%s",
                len(data),
                float(len(data)) / float(max(1, feature_row_count)),
                _setup_mix(data),
            )
        else:
            features = build_feature_frame(bars, base_features=base_features)
            if features.empty:
                raise RuntimeError("AetherFlow feature build returned empty frame.")
            feature_row_count = int(len(features))
            logging.info("AetherFlow feature frame built: rows=%d cols=%d", len(features), len(features.columns))

            data = _label_candidates(features, bars, fees_points=float(fees_points))
            if data.empty:
                raise RuntimeError("AetherFlow produced no labeled candidates.")
            data = data.sort_index()
            data.to_parquet(features_path, index=True)
            logging.info(
                "AetherFlow labeled candidates: rows=%d coverage=%.4f setups=%s",
                len(data),
                float(len(data)) / float(len(features)),
                _setup_mix(data),
            )

    n = len(data)
    train_end = int(n * float(args.train_frac))
    val_end = train_end + int(n * float(args.val_frac))
    train_end = max(100, min(train_end, n - 50))
    val_end = max(train_end + 25, min(val_end, n - 25))

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Train/val/test split is empty.")
    if train["label"].nunique() < 2 or val["label"].nunique() < 2:
        raise RuntimeError("AetherFlow train/val split does not have both classes.")

    logging.info(
        "Split sizes: train=%d val=%d test=%d | setup_mix train=%s val=%s test=%s",
        len(train),
        len(val),
        len(test),
        _setup_mix(train),
        _setup_mix(val),
        _setup_mix(test),
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["label"]
    X_val = val[FEATURE_COLUMNS]
    y_val = val["label"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["label"]
    val_session_ids = _coerce_session_ids(X_val["session_id"])
    test_session_ids = _coerce_session_ids(X_test["session_id"])
    allowed_session_ids = _parse_allowed_sessions(args.allowed_session_ids)
    if allowed_session_ids:
        logging.info("Session allowlist enabled: ids=%s names=%s", sorted(allowed_session_ids), [_session_name_from_id(s) for s in sorted(allowed_session_ids)])

    model_choice = str(args.model)
    fit_workers = max(1, int(args.workers))
    model = _build_model(model_choice, args.seed, workers=fit_workers)
    sample_weight = _sample_weights(y_train)
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except (PermissionError, OSError) as exc:
        if model_choice == "hgb":
            logging.warning("HGB fit failed (%s); falling back to RandomForest.", exc)
            model_choice = "rf"
            model = _build_model(model_choice, args.seed, workers=fit_workers)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            logging.warning("RF fit failed with workers=%d (%s); retrying with workers=1.", fit_workers, exc)
            fit_workers = 1
            model = _build_model(model_choice, args.seed, workers=fit_workers)
            model.fit(X_train, y_train, sample_weight=sample_weight)

    logging.info("AetherFlow model fitted (%s, workers=%d).", model_choice, fit_workers)

    prob_val = model.predict_proba(X_val)[:, 1]
    best_thr = _search_threshold(
        prob_val,
        val["net_points"].to_numpy(dtype=float),
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        min_trades=int(args.min_val_trades),
        min_coverage=float(args.min_val_coverage),
        coverage_target=float(args.coverage_target),
        coverage_penalty=float(args.coverage_penalty),
        val_folds=int(args.val_folds),
        min_fold_trades=int(args.min_fold_trades),
        min_positive_folds=int(args.min_positive_folds),
        objective_mean_weight=float(args.objective_mean_weight),
        objective_worst_weight=float(args.objective_worst_weight),
        objective_total_weight=float(args.objective_total_weight),
        objective_std_penalty=float(args.objective_std_penalty),
        sides=val["candidate_side"].to_numpy(dtype=float),
        session_ids=val_session_ids,
        allowed_session_ids=allowed_session_ids,
    )
    logging.info(
        "Best AetherFlow threshold: %.3f trades=%d coverage=%.4f win=%.2f%% avg_pnl=%.3f total_pnl=%.1f",
        float(best_thr["threshold"]),
        int(best_thr["trade_count"]),
        float(best_thr["coverage"]),
        100.0 * float(best_thr["win_rate"]),
        float(best_thr["avg_pnl"]),
        float(best_thr["total_pnl"]),
    )

    prob_test = model.predict_proba(X_test)[:, 1]
    test_eval = _evaluate_threshold(
        prob_test,
        test["net_points"].to_numpy(dtype=float),
        threshold=float(best_thr["threshold"]),
        sides=test["candidate_side"].to_numpy(dtype=float),
        session_ids=test_session_ids,
        allowed_session_ids=allowed_session_ids,
    )
    pred_test = (prob_test >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_test, prob_test))
    except Exception:
        auc = 0.0

    artifact_bundle = {
        "version": "aetherflow_v1",
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "trained_at": _utc_now_iso(),
        "threshold": float(best_thr["threshold"]),
    }
    with model_path.open("wb") as fh:
        pickle.dump(artifact_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    thresholds_payload = {
        "threshold": float(best_thr["threshold"]),
        "validation": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in best_thr.items()},
        "allowed_session_ids": sorted(int(s) for s in allowed_session_ids) if allowed_session_ids else [],
        "feature_columns": list(FEATURE_COLUMNS),
        "trained_at": _utc_now_iso(),
    }
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2))

    metrics_payload = {
        "version": "aetherflow_v1",
        "trained_at": _utc_now_iso(),
        "model_type": model_choice,
        "workers": int(fit_workers),
        "calibrated": False,
        "input": str(args.input),
        "base_features": str(args.base_features),
        "rows": int(len(data)),
        "feature_rows": int(feature_row_count or len(data)),
        "candidate_coverage": float(len(data)) / float(max(1, feature_row_count or len(data))),
        "setup_mix_all": _setup_mix(data),
        "setup_mix_train": _setup_mix(train),
        "setup_mix_val": _setup_mix(val),
        "setup_mix_test": _setup_mix(test),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(np.mean(y_train.to_numpy(dtype=float))),
        "val_positive_rate": float(np.mean(y_val.to_numpy(dtype=float))),
        "test_positive_rate": float(np.mean(y_test.to_numpy(dtype=float))),
        "threshold": float(best_thr["threshold"]),
        "validation": thresholds_payload["validation"],
        "test": {
            **test_eval,
            "accuracy": float(accuracy_score(y_test, pred_test)),
            "precision": float(precision_score(y_test, pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, pred_test, zero_division=0)),
            "f1": float(f1_score(y_test, pred_test, zero_division=0)),
            "roc_auc": float(auc),
        },
        "artifacts": {
            "model_file": str(model_file),
            "thresholds_file": str(thresholds_file),
            "metrics_file": str(metrics_file),
            "features_parquet": str(features_file),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    logging.info("Saved AetherFlow model: %s", model_path)
    logging.info("Saved AetherFlow thresholds: %s", thresholds_path)
    logging.info("Saved AetherFlow metrics: %s", metrics_path)
    logging.info("Saved AetherFlow labeled features: %s", features_path)


if __name__ == "__main__":
    main()

import argparse
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_features import build_feature_frame
from config import CONFIG
from train_aetherflow import (
    _default_checkpoint_dir,
    _format_eta,
    _label_candidates,
    _load_bars,
    _load_cached_base_features,
    _load_cached_training_data,
    _load_checkpoint_state,
    _parse_date,
    _save_checkpoint_state,
    _setup_mix,
    _utc_now_iso,
)
from train_manifold_strategy import DEFAULT_OUTRIGHT_SYMBOL_REGEX, _filter_range


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _build_family_dataset_with_checkpoint(
    *,
    family_name: str,
    preferred_setup_families: set[str],
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
        logging.info("Existing family dataset found, loading: %s", features_path)
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
            logging.info("Checkpoint marked complete, loading family dataset: %s", features_path)
            cached = _load_cached_training_data(features_path)
            return cached, int(existing_state.get("feature_rows_total", len(base_features)))

    if not resume and state_path.exists():
        raise RuntimeError(
            f"Checkpoint state already exists at {state_path}. "
            "Use --resume to continue or --force-rebuild to restart."
        )

    total_rows = int(len(base_features))
    if total_rows <= 0:
        raise RuntimeError("No cached base features available for family dataset build.")
    chunk_size = max(10_000, int(chunk_size))
    overlap_rows = max(80, int(overlap_rows))
    total_chunks = int(math.ceil(float(total_rows) / float(chunk_size)))
    created_at = _utc_now_iso()
    state = {
        "created_at": created_at,
        "updated_at": created_at,
        "complete": False,
        "family_name": str(family_name),
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
            "Resuming %s family build: chunk %d/%d complete_chunks=%d candidate_rows=%d",
            str(family_name),
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
        feature_slice = build_feature_frame(
            base_features=base_slice,
            preferred_setup_families=set(preferred_setup_families),
        )
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
            "%s family chunk %d/%d rows=%d:%d candidate_rows=%d cumulative=%d progress=%.2f%% chunk_time=%s eta=%s",
            str(family_name),
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
        raise RuntimeError(f"No family checkpoint parts were produced under {parts_dir}")
    logging.info("Concatenating %d %s family parts...", len(part_files), str(family_name))
    frames = [pd.read_parquet(part) for part in part_files]
    data = pd.concat(frames, axis=0).sort_index()
    data = data.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["label", "net_points"])
    data["label"] = data["label"].astype(int)
    data["dataset_family"] = str(family_name)
    data.to_parquet(features_path, index=True)

    state["complete"] = True
    state["updated_at"] = _utc_now_iso()
    state["feature_rows_total"] = int(total_rows)
    state["candidate_rows_total"] = int(len(data))
    _save_checkpoint_state(state_path, state)
    logging.info(
        "%s family build complete: feature_rows=%d candidate_rows=%d coverage=%.4f elapsed=%s saved=%s setups=%s",
        str(family_name),
        int(total_rows),
        int(len(data)),
        float(len(data)) / float(total_rows),
        _format_eta(time.time() - build_started),
        features_path,
        _setup_mix(data),
    )
    return data, int(total_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fresh family-native AetherFlow labeled dataset from the corrected full base.")
    parser.add_argument("--input", default="es_master_outrights.parquet")
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--chunk-size", type=int, default=175000)
    parser.add_argument("--chunk-overlap", type=int, default=160)
    parser.add_argument("--fees-points", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--allow-spreads", action="store_true")
    parser.add_argument("--symbol-regex", default=DEFAULT_OUTRIGHT_SYMBOL_REGEX)
    parser.add_argument("--min-valid-price", type=float, default=100.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    family_name = str(args.family or "").strip()
    if not family_name:
        raise RuntimeError("family is required")

    source_path = _resolve_path(str(args.input), "es_master_outrights.parquet")
    base_path = _resolve_path(str(args.base_features))
    output_path = _resolve_path(str(args.output_parquet))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        _resolve_path(str(args.checkpoint_dir))
        if str(args.checkpoint_dir or "").strip()
        else _default_checkpoint_dir(output_path.parent, output_path)
    )

    bars = _load_bars(
        source_path,
        allow_spreads=bool(args.allow_spreads),
        symbol_regex=str(args.symbol_regex or "").strip() or None,
        min_valid_price=float(args.min_valid_price),
    ).sort_index()
    start_ts = _parse_date(args.start, is_end=False)
    end_ts = _parse_date(args.end, is_end=True)
    bars = _filter_range(bars, start_ts, end_ts)
    if bars.empty:
        raise RuntimeError("No bars available after filtering.")
    logging.info("Loaded bars: rows=%d range=%s -> %s", len(bars), bars.index.min(), bars.index.max())

    base_features = _load_cached_base_features(base_path)
    base_features = base_features.loc[(base_features.index >= bars.index.min()) & (base_features.index <= bars.index.max())]
    if base_features.empty:
        raise RuntimeError("No cached base features in the requested range.")
    logging.info("Loaded base features: rows=%d range=%s -> %s", len(base_features), base_features.index.min(), base_features.index.max())

    if args.fees_points is not None:
        fees_points = float(args.fees_points)
    else:
        risk_cfg = CONFIG.get("RISK", {}) or {}
        point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
        fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
        fees_points = (fees_per_side * 2.0) / max(1e-9, point_value)

    data, feature_row_count = _build_family_dataset_with_checkpoint(
        family_name=str(family_name),
        preferred_setup_families={str(family_name)},
        base_features=base_features,
        bars=bars,
        features_path=output_path,
        checkpoint_dir=checkpoint_dir,
        chunk_size=int(args.chunk_size),
        overlap_rows=int(args.chunk_overlap),
        fees_points=float(fees_points),
        resume=bool(args.resume),
        force_rebuild=bool(args.force_rebuild),
    )
    print(
        f"family={family_name} rows={len(data)} feature_rows={feature_row_count} coverage={float(len(data))/float(max(1, feature_row_count)):.6f} output={output_path}"
    )


if __name__ == "__main__":
    main()

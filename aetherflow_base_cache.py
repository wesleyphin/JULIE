from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FULL_MANIFOLD_BASE_FEATURES = (
    "artifacts/aetherflow_corrected_full_2011_2026/"
    "manifold_base_outrights_2011_2026.parquet"
)


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return [str(name) for name in pq.ParquetFile(path).schema.names]
    except Exception as exc:
        raise RuntimeError(
            f"Unable to inspect parquet schema for {path}. "
            "Install/enable pyarrow so AetherFlow can validate its manifold base cache."
        ) from exc


def resolve_full_manifold_base_features_path(configured: str | Path | None = None) -> Path:
    root = Path(__file__).resolve().parent
    explicit = str(configured or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    canonical = Path(DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    if not canonical.is_absolute():
        canonical = root / canonical
    # Keep base-cache resolution deterministic so runtime and research do not
    # silently drift to a different parquet when a newer scratch cache appears.
    return canonical.resolve()


def validate_full_manifold_base_features_path(
    path: Path,
    required_columns: Iterable[str],
) -> list[str]:
    cols = _parquet_columns(Path(path))
    col_set = {str(col) for col in cols}

    if "label" in col_set or "future_points" in col_set:
        raise RuntimeError(
            f"{path} is a labeled manifold training parquet, not a full bar-by-bar manifold base cache. "
            "Recent AetherFlow research was invalid when this file was reused as base input. "
            "Build a proper cache with tools/build_manifold_base_cache.py and pass that path instead."
        )

    missing = sorted(str(col) for col in required_columns if str(col) not in col_set)
    if missing:
        raise RuntimeError(
            f"Manifold base cache {path} is missing required columns: {', '.join(missing)}"
        )
    return cols


def metadata_path_for_parquet(path: Path) -> Path:
    return Path(str(Path(path)) + ".meta.json")


def load_base_cache_metadata(path: Path) -> dict:
    meta_path = metadata_path_for_parquet(Path(path))
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_manifold_engine_state_snapshot(path: Path) -> dict:
    payload = load_base_cache_metadata(Path(path))
    state = payload.get("engine_state")
    if not isinstance(state, dict):
        return {}
    end_text = payload.get("engine_state_end")
    try:
        end_time = pd.Timestamp(end_text) if end_text is not None else None
    except Exception:
        end_time = None
    try:
        lookback_bars = int(payload.get("engine_lookback_bars", 0) or 0)
    except Exception:
        lookback_bars = 0
    return {
        "state": state,
        "end_time": end_time,
        "lookback_bars": int(lookback_bars),
        "continuation_mode": str(payload.get("continuation_mode", "") or ""),
        "metadata": payload,
    }

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_FULL_MANIFOLD_BASE_FEATURES = "manifold_base_outrights_full.parquet"


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return [str(name) for name in pq.ParquetFile(path).schema.names]
    except Exception as exc:
        raise RuntimeError(
            f"Unable to inspect parquet schema for {path}. "
            "Install/enable pyarrow so AetherFlow can validate its manifold base cache."
        ) from exc


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

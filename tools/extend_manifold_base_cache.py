import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import (
    DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    load_manifold_engine_state_snapshot,
    validate_full_manifold_base_features_path,
)
from config import CONFIG
from manifold_strategy_features import EXPORT_COLUMNS, build_training_feature_frame, build_training_feature_frame_with_state
from train_manifold_strategy import (
    DEFAULT_OUTRIGHT_SYMBOL_REGEX,
    _filter_range,
    _load_bars,
    _parse_date,
)


DEFAULT_EXISTING_BASE = DEFAULT_FULL_MANIFOLD_BASE_FEATURES


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _default_output_path(existing_path: Path, end_time: pd.Timestamp) -> Path:
    end_tag = pd.Timestamp(end_time).strftime("%Y%m%d_%H%M")
    suffix = "".join(existing_path.suffixes) or ".parquet"
    stem = existing_path.name[: -len(suffix)] if suffix else existing_path.stem
    return existing_path.with_name(f"{stem}_extended_{end_tag}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing manifold base cache using the latest bars so AetherFlow "
            "direct backtests can reuse the cache instead of rebuilding from scratch."
        )
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--existing-base", default=DEFAULT_EXISTING_BASE)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--allow-spreads", action="store_true")
    parser.add_argument("--symbol-regex", default=DEFAULT_OUTRIGHT_SYMBOL_REGEX)
    parser.add_argument("--min-valid-price", type=float, default=100.0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    source_path = _resolve_path(str(args.source))
    existing_path = _resolve_path(str(args.existing_base), DEFAULT_EXISTING_BASE)
    if not existing_path.exists():
        raise RuntimeError(f"Existing manifold base cache not found: {existing_path}")

    validate_full_manifold_base_features_path(existing_path, EXPORT_COLUMNS)
    existing = pd.read_parquet(existing_path)
    if existing.empty:
        raise RuntimeError(f"Existing manifold base cache is empty: {existing_path}")
    existing.index = pd.DatetimeIndex(existing.index)
    existing = existing.sort_index()

    start_text = str(args.start).strip() if args.start is not None else ""
    end_text = str(args.end).strip() if args.end is not None else ""
    start_ts = _parse_date(start_text, is_end=False) if start_text else None
    end_ts = _parse_date(end_text, is_end=True) if end_text else None
    if end_ts is None:
        raise RuntimeError("--end is required")
    if start_ts is not None and start_ts > end_ts:
        raise RuntimeError("Start must be before end")

    requested_start = start_ts if start_ts is not None else pd.Timestamp(existing.index.min())
    requested_end = pd.Timestamp(end_ts)
    if requested_end <= pd.Timestamp(existing.index.max()):
        logging.info(
            "Existing cache already covers requested end. existing_end=%s requested_end=%s",
            existing.index.max(),
            requested_end,
        )

    overlap_days = max(1, int(args.overlap_days or 1))
    # Rebuild only the trailing overlap plus the missing tail. Using the
    # requested output start here causes accidental full-history rebuilds when
    # callers omit --start on a long canonical cache.
    rebuild_start = max(
        pd.Timestamp(existing.index.min()),
        pd.Timestamp(existing.index.max()) - pd.Timedelta(days=overlap_days),
    )

    bars = _load_bars(
        source_path,
        allow_spreads=bool(args.allow_spreads),
        symbol_regex=str(args.symbol_regex or "").strip() or None,
        min_valid_price=float(args.min_valid_price),
    ).sort_index()
    bars = _filter_range(bars, rebuild_start, requested_end)
    if bars.empty:
        raise RuntimeError("No bars available in the requested rebuild range.")

    manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
    manifold_cfg["enabled"] = True
    existing_state_snapshot = load_manifold_engine_state_snapshot(existing_path)
    exact_state = existing_state_snapshot.get("state") if existing_state_snapshot.get("continuation_mode") == "engine_state_exact" else None
    exact_state_end = existing_state_snapshot.get("end_time")
    can_continue_exact = (
        isinstance(exact_state, dict)
        and exact_state
        and exact_state_end is not None
        and pd.Timestamp(exact_state_end) == pd.Timestamp(existing.index.max())
    )
    logging.info(
        "Extending manifold cache from %s using %s bars (%s -> %s)",
        existing_path,
        len(bars),
        bars.index.min(),
        bars.index.max(),
    )
    continuation_mode = "engine_state_exact" if can_continue_exact else "overlap_rebuild_fallback"
    if can_continue_exact:
        rebuilt, final_state, lookback_bars = build_training_feature_frame_with_state(
            bars,
            manifold_cfg=manifold_cfg,
            log_every=int(args.log_every or 0),
            initial_state=exact_state,
            start_after=pd.Timestamp(existing.index.max()),
        )
        combined = pd.concat([existing, rebuilt], axis=0).sort_index()
    else:
        rebuilt, final_state, lookback_bars = build_training_feature_frame_with_state(
            bars,
            manifold_cfg=manifold_cfg,
            log_every=int(args.log_every or 0),
        )
        if rebuilt.empty:
            raise RuntimeError("Rebuild returned no rows.")
        cutoff = pd.Timestamp(rebuilt.index.min())
        combined = pd.concat(
            [
                existing.loc[existing.index < cutoff],
                rebuilt,
            ],
            axis=0,
        ).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    combined = combined.loc[(combined.index >= requested_start) & (combined.index <= requested_end)]
    combined = combined.reindex(columns=EXPORT_COLUMNS).replace([pd.NA], 0.0).fillna(0.0)
    if combined.empty:
        raise RuntimeError("Combined cache is empty after applying the requested range.")

    output_path = (
        _resolve_path(str(args.output))
        if str(args.output or "").strip()
        else _default_output_path(existing_path, requested_end)
    )
    if output_path.exists() and not bool(args.overwrite):
        raise RuntimeError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    metadata_path = (
        _resolve_path(str(args.metadata_json))
        if str(args.metadata_json or "").strip()
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=True)
    metadata = {
        "feature_kind": "manifold_base_extended",
        "source_path": str(source_path),
        "existing_base_path": str(existing_path),
        "output_path": str(output_path),
        "requested_range": {
            "start": requested_start.isoformat(),
            "end": requested_end.isoformat(),
        },
        "rebuild_range": {
            "start": pd.Timestamp(bars.index.min()).isoformat(),
            "end": pd.Timestamp(bars.index.max()).isoformat(),
        },
        "rows": int(len(combined)),
        "columns": list(combined.columns),
        "range": {
            "start": pd.Timestamp(combined.index.min()).isoformat(),
            "end": pd.Timestamp(combined.index.max()).isoformat(),
        },
        "continuation_mode": continuation_mode,
        "engine_state_end": pd.Timestamp(combined.index.max()).isoformat(),
        "engine_lookback_bars": int(lookback_bars),
        "engine_state": final_state,
        "overlap_days": int(overlap_days),
        "built_at": pd.Timestamp.now("UTC").isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"extended_base_cache={output_path}")
    print(f"metadata_json={metadata_path}")
    print(f"rows={len(combined)}")
    print(f"range={combined.index.min()} -> {combined.index.max()}")


if __name__ == "__main__":
    main()

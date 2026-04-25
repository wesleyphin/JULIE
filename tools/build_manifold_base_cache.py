import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES
from config import CONFIG
from manifold_strategy_features import build_training_feature_frame_with_state
from train_manifold_strategy import (
    DEFAULT_OUTRIGHT_SYMBOL_REGEX,
    _filter_range,
    _load_bars,
    _parse_date,
)


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a full bar-by-bar manifold base cache for AetherFlow research."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    parser.add_argument("--metadata-json", default=None)
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
    output_path = _resolve_path(str(args.output))
    metadata_path = (
        _resolve_path(str(args.metadata_json))
        if args.metadata_json
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )

    if output_path.exists() and not bool(args.overwrite):
        raise RuntimeError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    bars = _load_bars(
        source_path,
        allow_spreads=bool(args.allow_spreads),
        symbol_regex=str(args.symbol_regex or "").strip() or None,
        min_valid_price=float(args.min_valid_price),
    ).sort_index()
    start_text = str(args.start).strip() if args.start is not None else ""
    end_text = str(args.end).strip() if args.end is not None else ""
    start_ts = _parse_date(start_text, is_end=False) if start_text else None
    end_ts = _parse_date(end_text, is_end=True) if end_text else None
    bars = _filter_range(bars, start_ts, end_ts)
    if bars.empty:
        raise RuntimeError("No bars available after applying the requested range.")

    manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
    manifold_cfg["enabled"] = True

    print(
        f"Building manifold base cache from {source_path} rows={len(bars)} "
        f"range={bars.index.min()} -> {bars.index.max()}"
    )
    features, final_state, lookback_bars = build_training_feature_frame_with_state(
        bars,
        manifold_cfg=manifold_cfg,
        log_every=int(args.log_every or 0),
    )
    if features.empty:
        raise RuntimeError("Feature build returned no rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=True)

    metadata = {
        "feature_kind": "manifold_base_full",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "rows": int(len(features)),
        "columns": list(features.columns),
        "range": {
            "start": features.index.min().isoformat(),
            "end": features.index.max().isoformat(),
        },
        "continuation_mode": "engine_state_exact",
        "engine_state_end": pd.Timestamp(features.index.max()).isoformat(),
        "engine_lookback_bars": int(lookback_bars),
        "engine_state": final_state,
        "built_at": pd.Timestamp.now("UTC").isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"saved_base_cache={output_path}")
    print(f"saved_metadata={metadata_path}")


if __name__ == "__main__":
    main()

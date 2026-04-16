import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import backtest_mes_et as bt
import data_cache
from config import (
    CONFIG,
    append_artifact_suffix,
    get_experimental_training_window,
    resolve_artifact_suffix,
)


def _load_csv(csv_path: Path, cache_dir: Optional[Path], use_cache: bool) -> pd.DataFrame:
    return data_cache.load_bars(csv_path, cache_dir=cache_dir, use_cache=use_cache)


def _parse_date(value: Optional[str], *, is_end: bool = False) -> Optional[pd.Timestamp]:
    if not value:
        return None
    raw = str(value).strip()
    has_time = ("T" in raw) or (":" in raw)
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    if is_end and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _filter_range(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df.empty:
        return df
    if start is not None:
        df = df.loc[df.index >= start]
    if end is not None:
        df = df.loc[df.index <= end]
    return df


def _run_flip_confidence(df: pd.DataFrame, cfg: dict) -> tuple[set, dict]:
    local_cfg = dict(cfg)
    local_cfg["cache_file"] = None
    payload = bt.build_flip_confidence_from_df(df, local_cfg)
    allowlist = set(payload.get("allowlist") or [])
    return allowlist, payload or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flip-confidence allowlist from CSV history.")
    parser.add_argument("--csv", default="es_master.csv", help="Path to CSV history file.")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to config cache_file).")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--experimental-window",
        action="store_true",
        help="Train only on configured experimental window (2011-01-01 .. 2017-12-31).",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Suffix appended to output artifacts (e.g. _exp2011_2017).",
    )
    parser.add_argument(
        "--recent-start",
        default="2023-01-01",
        help="Recent window start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--recent-end",
        default="2025-12-31",
        help="Recent window end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--recent-mode",
        default="intersect",
        choices=("intersect", "union", "recent_only"),
        help="How to combine full vs recent allowlists.",
    )
    parser.add_argument(
        "--no-recent",
        action="store_true",
        help="Disable recency window.",
    )
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for parquet.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    exp_enabled = bool(args.experimental_window)
    train_start_raw = args.start
    train_end_raw = args.end
    if exp_enabled:
        exp_start, exp_end = get_experimental_training_window()
        train_start_raw = exp_start
        train_end_raw = exp_end
        logging.info("Experimental window enabled: %s -> %s", train_start_raw, train_end_raw)
    artifact_suffix = resolve_artifact_suffix(args.artifact_suffix, exp_enabled)
    if exp_enabled and not args.no_recent:
        args.recent_start = train_start_raw
        args.recent_end = train_end_raw
        logging.info("Experimental mode: recent window aligned to %s -> %s", args.recent_start, args.recent_end)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    df = _load_csv(csv_path, cache_dir, not args.no_cache)
    start = _parse_date(train_start_raw, is_end=False)
    end = _parse_date(train_end_raw, is_end=True)
    df = _filter_range(df, start, end)

    flip_cfg = CONFIG.get("BACKTEST_FLIP_CONFIDENCE", {}) or {}
    full_allow, full_payload = _run_flip_confidence(df, flip_cfg)

    recent_allow = None
    recent_payload = None
    if not args.no_recent:
        recent_start = _parse_date(args.recent_start, is_end=False)
        recent_end = _parse_date(args.recent_end, is_end=True)
        if recent_start is not None or recent_end is not None:
            recent_df = _filter_range(df, recent_start, recent_end)
            recent_allow, recent_payload = _run_flip_confidence(recent_df, flip_cfg)
            if args.recent_mode == "union":
                final_allow = full_allow | recent_allow
            elif args.recent_mode == "recent_only":
                final_allow = set(recent_allow)
            else:
                final_allow = full_allow & recent_allow
        else:
            final_allow = set(full_allow)
    else:
        final_allow = set(full_allow)

    out_name = args.out or flip_cfg.get("cache_file") or "backtest_reports/flip_confidence.json"
    if artifact_suffix:
        out_name = append_artifact_suffix(str(out_name), artifact_suffix)
    out_path = out_name
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": dt.datetime.now(bt.NY_TZ).isoformat(),
        "summary": {
            "full_keys_allowed": len(full_allow),
            "final_keys_allowed": len(final_allow),
        },
        "criteria": full_payload.get("criteria", flip_cfg),
        "key_fields": full_payload.get("key_fields", flip_cfg.get("key_fields")),
        "allowed_filters": full_payload.get("allowed_filters", flip_cfg.get("allowed_filters", [])),
        "allowlist": sorted(final_allow),
        "stats": full_payload.get("stats", {}),
    }
    if recent_payload is not None:
        payload["recent"] = {
            "mode": args.recent_mode,
            "summary": recent_payload.get("summary", {}),
            "allowlist": sorted(recent_allow or []),
            "stats": recent_payload.get("stats", {}),
        }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logging.info("Wrote flip-confidence allowlist: %s", out_path)
    logging.info("Allowlist size: %s", len(final_allow))


if __name__ == "__main__":
    main()

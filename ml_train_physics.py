from __future__ import annotations

import argparse
import json
import logging
import tempfile
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

from dist_bracket_ml.cli import main as dist_cli_main
from config import append_artifact_suffix, get_experimental_training_window, resolve_artifact_suffix

# Force immediate console output for long-running stages.
print = partial(print, flush=True)  # type: ignore[assignment]


def _resolve_data_path(args: argparse.Namespace) -> str:
    candidates = [
        getattr(args, "data", None),
        getattr(args, "csv", None),
        getattr(args, "parquet", None),
        getattr(args, "input", None),
    ]
    for value in candidates:
        if value:
            return str(value)
    raise SystemExit(
        "No input data path was provided. Use one of: --data, --csv, --parquet, --input"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility launcher for the legacy ml_physics trainer. "
            "This now routes to dist_bracket_ml train-all."
        )
    )
    parser.add_argument("--data", default=None, help="Input CSV/Parquet path")
    parser.add_argument("--csv", default=None, help="Input CSV path (legacy alias)")
    parser.add_argument("--parquet", default=None, help="Input Parquet path (legacy alias)")
    parser.add_argument("--input", default=None, help="Input file path (legacy alias)")
    parser.add_argument("--out", default=None, help="Output directory (new flag)")
    parser.add_argument("--out-dir", default=None, help="Output directory (legacy alias)")
    parser.add_argument("--session", default=None, help="Optional single-session training")
    parser.add_argument("--backend", choices=("lgbm", "xgb"), default=None, help="Model backend")
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", help="Use GPU for XGBoost backend")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Disable GPU for XGBoost backend")
    parser.set_defaults(use_gpu=None)
    parser.add_argument(
        "--workers",
        "--num-workers",
        dest="workers",
        type=int,
        default=6,
        help="Worker threads for training (default: 6)",
    )
    parser.add_argument("--wf-step-months", type=int, default=None, help="Walk-forward step in months")
    parser.add_argument("--no-hit-model", action="store_true", help="Disable hit-model training/grid search")
    parser.add_argument("--hit-sample-rows", type=int, default=None, help="Hit-model sampled rows cap")
    parser.add_argument("--config", default=None, help="Path to dist_bracket_ml config JSON")
    parser.add_argument("--start", "--train-start", dest="train_start", default=None, help="Train start date (YYYY-MM-DD)")
    parser.add_argument("--end", "--train-end", dest="train_end", default=None, help="Train end date (YYYY-MM-DD)")
    parser.add_argument(
        "--experimental-window",
        action="store_true",
        help="Use CONFIG experimental window for training bounds.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Suffix appended to output directory name (e.g. _exp2011_2017).",
    )
    parser.add_argument("--walk-forward", dest="walk_forward", action="store_true", default=True)
    parser.add_argument("--no-walk-forward", dest="walk_forward", action="store_false")
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Resume from an existing dist_bracket run directory.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from latest run under --out/--out-dir.",
    )
    parser.add_argument("--skip-gate", action="store_true", help="Skip gate dataset/model stages")
    parser.add_argument("--eval", action="store_true", help="Run eval at end")
    parser.add_argument("--eval-start", default=None, help="Optional eval start date (YYYY-MM-DD)")
    parser.add_argument("--eval-end", default=None, help="Optional eval end date (YYYY-MM-DD)")
    return parser


def _resolve_train_window(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    start = str(getattr(args, "train_start", "") or "").strip() or None
    end = str(getattr(args, "train_end", "") or "").strip() or None
    if bool(getattr(args, "experimental_window", False)):
        exp_start, exp_end = get_experimental_training_window()
        start = exp_start or start
        end = exp_end or end
    return start, end


def _write_config_override(
    *,
    base_config_path: Optional[str],
    train_start: Optional[str],
    train_end: Optional[str],
) -> Optional[str]:
    if not base_config_path and not train_start and not train_end:
        return None

    payload = {}
    if base_config_path:
        cfg_path = Path(base_config_path).expanduser().resolve()
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["start"] = train_start
    payload["end"] = train_end

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="ml_physics_cfg_",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.flush()
        return handle.name


def _translate_args(args: argparse.Namespace) -> tuple[List[str], Optional[str]]:
    forwarded: List[str] = ["train-all", "--data", _resolve_data_path(args)]
    temp_config_path: Optional[str] = None

    exp_enabled = bool(getattr(args, "experimental_window", False))
    artifact_suffix = resolve_artifact_suffix(getattr(args, "artifact_suffix", None), exp_enabled)

    out_dir = args.out or args.out_dir
    if artifact_suffix:
        if out_dir:
            out_dir = append_artifact_suffix(str(out_dir), artifact_suffix)
        elif exp_enabled:
            out_dir = append_artifact_suffix("dist_bracket_ml_runs", artifact_suffix)
    if out_dir:
        forwarded.extend(["--out", str(out_dir)])
    if args.session:
        forwarded.extend(["--session", str(args.session)])
    if args.backend:
        forwarded.extend(["--backend", str(args.backend)])
    if args.use_gpu is True:
        forwarded.append("--gpu")
    elif args.use_gpu is False:
        forwarded.append("--no-gpu")
    if args.workers is not None:
        forwarded.extend(["--workers", str(max(1, int(args.workers)))])
    if args.wf_step_months is not None:
        forwarded.extend(["--wf-step-months", str(max(1, int(args.wf_step_months)))])
    if bool(args.no_hit_model):
        forwarded.append("--no-hit-model")
    if args.hit_sample_rows is not None:
        forwarded.extend(["--hit-sample-rows", str(max(1000, int(args.hit_sample_rows)))])
    train_start, train_end = _resolve_train_window(args)
    temp_config_path = _write_config_override(
        base_config_path=getattr(args, "config", None),
        train_start=train_start,
        train_end=train_end,
    )
    if temp_config_path:
        forwarded.extend(["--config", str(temp_config_path)])
    elif args.config:
        forwarded.extend(["--config", str(args.config)])
    if not bool(args.walk_forward):
        forwarded.append("--no-walk-forward")
    if bool(args.skip_gate):
        forwarded.append("--skip-gate")
    if getattr(args, "resume_run", None):
        forwarded.extend(["--resume-run", str(args.resume_run)])
    if bool(getattr(args, "resume_latest", False)):
        forwarded.append("--resume-latest")
    if bool(args.eval):
        forwarded.append("--eval")
    if getattr(args, "eval_start", None):
        forwarded.extend(["--eval-start", str(args.eval_start)])
    if getattr(args, "eval_end", None):
        forwarded.extend(["--eval-end", str(args.eval_end)])
    return forwarded, temp_config_path


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logging.warning(
            "Ignoring legacy ml_physics args not used by dist_bracket_ml: %s",
            " ".join(unknown),
        )

    forwarded, temp_config_path = _translate_args(args)
    try:
        start, end = _resolve_train_window(args)
        is_resume = bool(getattr(args, "resume_run", None)) or bool(getattr(args, "resume_latest", False))
        if (not is_resume) and (not start or not end):
            logging.warning(
                "Training window is not fully bounded (start=%s end=%s). "
                "For strict OOS workflows, pass both --train-start and --train-end.",
                start,
                end,
            )
        if start or end:
            print(f"ml_train_physics: train window -> start={start} end={end}")
        print("ml_train_physics: routed to dist_bracket_ml ->", " ".join(forwarded))
        return int(dist_cli_main(forwarded))
    finally:
        if temp_config_path:
            try:
                Path(temp_config_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

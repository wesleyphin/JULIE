from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VENV = (ROOT / ".venv").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _require_project_venv() -> None:
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    if prefix != EXPECTED_VENV:
        raise RuntimeError(
            f"AetherFlow corrected pipeline must run inside the project venv. "
            f"expected_prefix={EXPECTED_VENV} actual_prefix={prefix} executable={executable}"
        )


def _append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def _run_stage(name: str, args: list[str], log_path: Path) -> None:
    command = [sys.executable, *args]
    _append_log(log_path, f"[{_timestamp()}] stage={name} start")
    _append_log(log_path, f"[{_timestamp()}] command={shlex.join(command)}")
    print(f"[pipeline] {name} start", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=_stage_env(_extract_worker_count(args)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        _append_log(log_path, line.rstrip("\n"))
    code = int(proc.wait())
    _append_log(log_path, f"[{_timestamp()}] stage={name} exit={code}")
    if code != 0:
        raise RuntimeError(f"Stage failed: {name} exit={code}")


def _extract_worker_count(args: list[str]) -> int | None:
    for idx, item in enumerate(args):
        if str(item) == "--workers" and (idx + 1) < len(args):
            try:
                return max(1, int(float(args[idx + 1])))
            except Exception:
                return None
    return None


def _stage_env(workers: int | None) -> dict[str, str]:
    env = dict(os.environ)
    if workers is None:
        return env
    worker_text = str(max(1, int(workers)))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = worker_text
    return env


def _wait_for_base(base_path: Path, poll_seconds: int) -> None:
    meta_path = Path(str(base_path) + ".meta.json")
    while not (base_path.exists() and meta_path.exists()):
        print(
            f"[pipeline] waiting_for_base base={base_path} meta={meta_path}",
            flush=True,
        )
        time.sleep(max(5, int(poll_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the corrected continuous-window AetherFlow research pipeline from the project venv."
    )
    parser.add_argument("--artifact-dir", default="artifacts/aetherflow_corrected_full_2011_2026")
    parser.add_argument(
        "--base-path",
        default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES,
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--train-start", default="2011-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument(
        "--variants-file",
        default="configs/aetherflow_corrected_full_history_candidates.json",
    )
    parser.add_argument(
        "--oos-windows-file",
        default="configs/aetherflow_corrected_continuous_oos_windows.json",
    )
    parser.add_argument(
        "--historical-windows-file",
        default="configs/aetherflow_corrected_continuous_historical_windows.json",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--mc-simulations", type=int, default=2000)
    args = parser.parse_args()

    _require_project_venv()

    artifact_dir = _resolve_path(str(args.artifact_dir))
    base_path = _resolve_path(str(args.base_path))
    source_path = _resolve_path(str(args.source))
    variants_path = _resolve_path(str(args.variants_file))
    oos_windows_path = _resolve_path(str(args.oos_windows_file))
    historical_windows_path = _resolve_path(str(args.historical_windows_file))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model_aetherflow_corrected_train_thru_2024.pkl"
    thresholds_path = artifact_dir / "aetherflow_thresholds_corrected_train_thru_2024.json"
    metrics_path = artifact_dir / "aetherflow_metrics_corrected_train_thru_2024.json"
    features_path = artifact_dir / "aetherflow_features_corrected_train_thru_2024.parquet"

    _wait_for_base(base_path, int(args.poll_seconds))

    _run_stage(
        "train_continuous_thru_2024",
        [
            "-u",
            "train_aetherflow.py",
            "--input",
            str(source_path),
            "--base-features",
            str(base_path),
            "--start",
            str(args.train_start),
            "--end",
            str(args.train_end),
            "--out-dir",
            str(artifact_dir),
            "--model-file",
            str(model_path.name),
            "--thresholds-file",
            str(thresholds_path.name),
            "--metrics-file",
            str(metrics_path.name),
            "--features-parquet",
            str(features_path.name),
            "--workers",
            str(int(args.workers)),
            "--resume-label-build",
        ],
        artifact_dir / "train_continuous_thru_2024.log",
    )

    _run_stage(
        "search_continuous_oos",
        [
            "-u",
            "tools/run_aetherflow_deploy_policy_search.py",
            "--source",
            str(source_path),
            "--base-features",
            str(base_path),
            "--model-file",
            str(model_path),
            "--thresholds-file",
            str(thresholds_path),
            "--variants-file",
            str(variants_path),
            "--windows-file",
            str(oos_windows_path),
            "--output-dir",
            "backtest_reports/aetherflow_corrected_continuous_oos_search",
            "--mc-simulations",
            str(int(args.mc_simulations)),
        ],
        artifact_dir / "search_continuous_oos.log",
    )

    _run_stage(
        "validate_historical_continuous",
        [
            "-u",
            "tools/run_aetherflow_deploy_policy_search.py",
            "--source",
            str(source_path),
            "--base-features",
            str(base_path),
            "--model-file",
            str(model_path),
            "--thresholds-file",
            str(thresholds_path),
            "--variants-file",
            str(variants_path),
            "--windows-file",
            str(historical_windows_path),
            "--output-dir",
            "backtest_reports/aetherflow_corrected_continuous_historical_validation",
            "--mc-simulations",
            str(int(args.mc_simulations)),
        ],
        artifact_dir / "validate_historical_continuous.log",
    )

    print("[pipeline] complete", flush=True)


if __name__ == "__main__":
    main()

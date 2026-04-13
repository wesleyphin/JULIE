from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WINDOWS = [
    {
        "name": "tune_2024",
        "start": "2024-01-01",
        "end": "2024-12-31",
    },
    {
        "name": "oos_2025",
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
    {
        "name": "full_2024_2025",
        "start": "2024-01-01",
        "end": "2025-12-31",
    },
]


def _resolve_path(raw: str) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _quote_ps(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _run_stage(
    *,
    name: str,
    command: List[str],
    artifact_root: Path,
    manifest: Dict[str, Any],
    dry_run: bool,
) -> None:
    logs_dir = artifact_root / "workflow_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    stage_meta = {
        "name": name,
        "command": command,
        "cwd": str(ROOT),
        "log_path": str(log_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if dry_run else "running",
    }
    manifest.setdefault("stages", []).append(stage_meta)
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    if dry_run:
        print(f"[dry-run] {name}: {_quote_ps(command)}")
        stage_meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        return

    print(f"[stage] {name}")
    print(f"[cmd] {_quote_ps(command)}")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="")
        return_code = int(process.wait())

    stage_meta["elapsed_sec"] = float(time.perf_counter() - started)
    stage_meta["return_code"] = return_code
    stage_meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    stage_meta["status"] = "ok" if return_code == 0 else "failed"
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    if return_code != 0:
        raise SystemExit(f"Stage failed: {name} (see {log_path})")


def _build_validation_command_specs(
    *,
    source_path: Path,
    base_bundle_path: Path,
    candidate_summary_path: Path,
    artifact_root: Path,
    symbol_mode: str,
    symbol_method: str,
) -> Dict[str, Any]:
    bundles = [
        {
            "name": "base_rebuilt",
            "bundle_path": str(base_bundle_path),
        }
    ]
    if candidate_summary_path.exists():
        try:
            summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        candidates = summary.get("candidates", {}) if isinstance(summary, dict) else {}
        if isinstance(candidates, dict):
            for name, row in candidates.items():
                if not isinstance(row, dict):
                    continue
                bundle_path = str(row.get("bundle_path", "") or "").strip()
                if not bundle_path:
                    continue
                bundles.append(
                    {
                        "name": str(name),
                        "bundle_path": bundle_path,
                    }
                )

    commands: List[Dict[str, Any]] = []
    ps_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location '{ROOT}'",
        "",
    ]
    for bundle in bundles:
        bundle_name = str(bundle["name"])
        bundle_path = str(bundle["bundle_path"])
        safe_name = bundle_name.replace(" ", "_")
        for window in DEFAULT_WINDOWS:
            report_dir = artifact_root / "backtest_reports" / safe_name
            cmd = [
                sys.executable,
                "-u",
                "tools/run_de3_backtest.py",
                "--source",
                str(source_path),
                "--start",
                str(window["start"]),
                "--end",
                str(window["end"]),
                "--symbol-mode",
                symbol_mode,
                "--symbol-method",
                symbol_method,
                "--bundle-path",
                bundle_path,
                "--sync-entry-model-from-bundle",
                "--report-dir",
                str(report_dir),
            ]
            commands.append(
                {
                    "bundle_name": bundle_name,
                    "window_name": str(window["name"]),
                    "start": str(window["start"]),
                    "end": str(window["end"]),
                    "command": cmd,
                }
            )
            ps_lines.append(_quote_ps(cmd))
        ps_lines.append("")

    commands_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "commands": commands,
    }
    _write_json(artifact_root / "recommended_validation_commands.json", commands_payload)
    (artifact_root / "recommended_validation_commands.ps1").write_text(
        "\n".join(ps_lines).strip() + "\n",
        encoding="utf-8",
    )
    return commands_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ground-up DE3 workflow runner. Builds a clean v2 DB and v4 bundle from "
            "es_master_outrights.parquet, exports a fresh current pool, retrains "
            "entry-policy candidates, and writes validation commands."
        )
    )
    parser.add_argument(
        "--source",
        default="es_master_outrights.parquet",
        help="Primary outright-only 1m parquet source.",
    )
    parser.add_argument(
        "--artifact-root",
        default="",
        help="Explicit output root. Defaults to artifacts/de3_ground_up_<timestamp>.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix used when artifact root is auto-generated.",
    )
    parser.add_argument("--workers", type=int, default=12, help="Worker count for DE3 v2 build.")
    parser.add_argument(
        "--acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="DE3 v2 scoring acceleration mode.",
    )
    parser.add_argument(
        "--symbol-mode",
        default="auto_by_day",
        help="Symbol mode used for current-pool export and validation commands.",
    )
    parser.add_argument(
        "--symbol-method",
        default="volume",
        help="Symbol method used for current-pool export and validation commands.",
    )
    parser.add_argument(
        "--current-pool-start",
        default="2011-01-01",
        help="Start date for fresh current-pool export.",
    )
    parser.add_argument(
        "--current-pool-end",
        default="2024-12-31",
        help="End date for fresh current-pool export.",
    )
    parser.add_argument(
        "--candidate-profiles",
        default="",
        help=(
            "Optional comma-separated subset for tools/train_de3_entry_policy_from_current_pool.py "
            "(example: current_pool_pf_coverage,current_pool_tail_guard_v2)."
        ),
    )
    parser.add_argument("--skip-v2", action="store_true", help="Assume the clean v2 DB already exists in artifact root.")
    parser.add_argument("--skip-v4", action="store_true", help="Assume the clean v4 bundle already exists in artifact root.")
    parser.add_argument("--skip-current-pool", action="store_true", help="Assume the current-pool export already exists.")
    parser.add_argument("--skip-entry-policy", action="store_true", help="Skip entry-policy candidate retraining.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/commands without executing stages.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = _resolve_path(str(args.source))
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    if str(args.artifact_root or "").strip():
        artifact_root = _resolve_path(str(args.artifact_root))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{str(args.tag).strip()}" if str(args.tag).strip() else ""
        artifact_root = ROOT / "artifacts" / f"de3_ground_up_{ts}{suffix}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    v2_db_path = artifact_root / "dynamic_engine3_strategies_v2.outrights.json"
    v2_checkpoint_path = artifact_root / "de3_v2_outrights.checkpoint.json"
    v4_bundle_path = artifact_root / "dynamic_engine3_v4_bundle.outrights.json"
    v4_reports_dir = artifact_root / "reports" / "base_bundle"
    current_pool_decisions_path = artifact_root / "reports" / "de3_current_pool_2011_2024.csv"
    current_pool_trade_path = artifact_root / "reports" / "de3_current_pool_2011_2024_trade_attribution.csv"
    entry_policy_dir = artifact_root / "entry_policy_candidates"
    candidate_summary_path = entry_policy_dir / "candidate_summary.json"

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "source_path": str(source_path),
        "settings": {
            "workers": int(args.workers),
            "acceleration": str(args.acceleration),
            "symbol_mode": str(args.symbol_mode),
            "symbol_method": str(args.symbol_method),
            "current_pool_start": str(args.current_pool_start),
            "current_pool_end": str(args.current_pool_end),
            "candidate_profiles": _parse_csv_list(str(args.candidate_profiles)),
            "dry_run": bool(args.dry_run),
        },
        "artifacts": {
            "v2_db_path": str(v2_db_path),
            "v2_checkpoint_path": str(v2_checkpoint_path),
            "v4_bundle_path": str(v4_bundle_path),
            "v4_reports_dir": str(v4_reports_dir),
            "current_pool_decisions_path": str(current_pool_decisions_path),
            "current_pool_trade_attribution_path": str(current_pool_trade_path),
            "entry_policy_dir": str(entry_policy_dir),
            "candidate_summary_path": str(candidate_summary_path),
        },
        "stages": [],
    }
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    if not bool(args.skip_v2):
        _run_stage(
            name="de3_v2_build",
            command=[
                sys.executable,
                "-u",
                "de3_v2_generator.py",
                "--source",
                str(source_path),
                "--out",
                str(v2_db_path),
                "--workers",
                str(max(1, int(args.workers))),
                "--acceleration",
                str(args.acceleration),
                "--checkpoint",
                "--no-resume",
                "--checkpoint-path",
                str(v2_checkpoint_path),
                "--cache-dir",
                "cache",
            ],
            artifact_root=artifact_root,
            manifest=manifest,
            dry_run=bool(args.dry_run),
        )

    if not bool(args.skip_v4):
        _run_stage(
            name="de3_v4_build",
            command=[
                sys.executable,
                "-u",
                "de3_v4_trainer.py",
                "--source-db",
                str(v2_db_path),
                "--source-parquet",
                str(source_path),
                "--out-bundle",
                str(v4_bundle_path),
                "--reports-dir",
                str(v4_reports_dir),
                "--full-build",
                "--train-start",
                "2011-01-01",
                "--train-end",
                "2023-12-31",
                "--tune-start",
                "2024-01-01",
                "--tune-end",
                "2024-12-31",
                "--oos-start",
                "2025-01-01",
                "--oos-end",
                "2025-12-31",
                "--future-start",
                "2026-01-01",
            ],
            artifact_root=artifact_root,
            manifest=manifest,
            dry_run=bool(args.dry_run),
        )

    if not bool(args.skip_current_pool):
        _run_stage(
            name="de3_current_pool_export",
            command=[
                sys.executable,
                "-u",
                "tools/export_de3_current_pool.py",
                "--source",
                str(source_path),
                "--start",
                str(args.current_pool_start),
                "--end",
                str(args.current_pool_end),
                "--symbol-mode",
                str(args.symbol_mode),
                "--symbol-method",
                str(args.symbol_method),
                "--bundle-path",
                str(v4_bundle_path),
                "--sync-entry-model-from-bundle",
                "--report-dir",
                str(artifact_root / "backtest_reports" / "current_pool"),
                "--decisions-out",
                str(current_pool_decisions_path),
            ],
            artifact_root=artifact_root,
            manifest=manifest,
            dry_run=bool(args.dry_run),
        )

    if not bool(args.skip_entry_policy):
        entry_cmd = [
            sys.executable,
            "-u",
            "tools/train_de3_entry_policy_from_current_pool.py",
            "--base-bundle",
            str(v4_bundle_path),
            "--decisions-csv",
            str(current_pool_decisions_path),
            "--trade-attribution-csv",
            str(current_pool_trade_path),
            "--output-dir",
            str(entry_policy_dir),
        ]
        candidate_profiles = _parse_csv_list(str(args.candidate_profiles))
        if candidate_profiles:
            entry_cmd.extend(["--only", ",".join(candidate_profiles)])
        _run_stage(
            name="de3_entry_policy_candidates",
            command=entry_cmd,
            artifact_root=artifact_root,
            manifest=manifest,
            dry_run=bool(args.dry_run),
        )

    commands_payload = _build_validation_command_specs(
        source_path=source_path,
        base_bundle_path=v4_bundle_path,
        candidate_summary_path=candidate_summary_path,
        artifact_root=artifact_root,
        symbol_mode=str(args.symbol_mode),
        symbol_method=str(args.symbol_method),
    )
    manifest["validation_commands_path"] = str(artifact_root / "recommended_validation_commands.json")
    manifest["validation_powershell_path"] = str(artifact_root / "recommended_validation_commands.ps1")
    manifest["validation_command_count"] = int(len(commands_payload.get("commands", [])))
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    print(f"artifact_root={artifact_root}")
    print(f"workflow_manifest={artifact_root / 'workflow_manifest.json'}")
    print(f"validation_commands={artifact_root / 'recommended_validation_commands.json'}")
    print(f"validation_powershell={artifact_root / 'recommended_validation_commands.ps1'}")


if __name__ == "__main__":
    main()

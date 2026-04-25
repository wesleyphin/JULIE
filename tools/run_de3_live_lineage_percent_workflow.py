from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from de3_v4_schema import build_family_id, safe_float


DEFAULT_WINDOWS = [
    {"name": "oos_2025", "start": "2025-01-01", "end": "2025-12-31"},
    {"name": "full_2024_2025", "start": "2024-01-01", "end": "2025-12-31"},
    {"name": "full_2011_2025", "start": "2011-01-01", "end": "2025-12-31"},
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


def _require_existing_paths(*, reason: str, paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{reason}: missing required path(s): {', '.join(missing)}")


def _run_stage(
    *,
    name: str,
    command: List[str],
    artifact_root: Path,
    manifest: Dict[str, Any],
    dry_run: bool,
    expected_paths: Optional[List[Path]] = None,
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

    missing = [str(path) for path in (expected_paths or []) if not path.exists()]
    if missing:
        stage_meta["status"] = "failed"
        stage_meta["missing_expected_paths"] = missing
        _write_json(artifact_root / "workflow_manifest.json", manifest)
        raise SystemExit(
            f"Stage {name} finished without expected outputs: {', '.join(missing)}"
        )


def _run_inline_stage(
    *,
    name: str,
    artifact_root: Path,
    manifest: Dict[str, Any],
    dry_run: bool,
    expected_paths: Optional[List[Path]],
    runner,
) -> None:
    logs_dir = artifact_root / "workflow_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    stage_meta = {
        "name": name,
        "command": ["<inline-python>"],
        "cwd": str(ROOT),
        "log_path": str(log_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if dry_run else "running",
    }
    manifest.setdefault("stages", []).append(stage_meta)
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    if dry_run:
        print(f"[dry-run] {name}: inline stage")
        stage_meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        return

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        runner(handle)
    stage_meta["elapsed_sec"] = float(time.perf_counter() - started)
    stage_meta["return_code"] = 0
    stage_meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    stage_meta["status"] = "ok"
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    missing = [str(path) for path in (expected_paths or []) if not path.exists()]
    if missing:
        stage_meta["status"] = "failed"
        stage_meta["missing_expected_paths"] = missing
        _write_json(artifact_root / "workflow_manifest.json", manifest)
        raise SystemExit(
            f"Stage {name} finished without expected outputs: {', '.join(missing)}"
        )


def _variant_rows_from_bundle(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in (
        "long_rev_variants",
        "short_rev_variants",
        "long_mom_variants",
        "short_mom_variants",
    ):
        payload = bundle.get(key, [])
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict):
                rows.append(dict(row))
    if rows:
        return rows
    lane_variant_quality = (
        bundle.get("lane_variant_quality", {})
        if isinstance(bundle.get("lane_variant_quality"), dict)
        else {}
    )
    for row in lane_variant_quality.values():
        if isinstance(row, dict):
            rows.append(dict(row))
    return rows


def _extract_control_family_targets(bundle: Dict[str, Any]) -> Dict[str, Any]:
    variant_rows = _variant_rows_from_bundle(bundle)
    family_counts: Counter[str] = Counter()
    family_meta: Dict[str, Dict[str, Any]] = {}
    for row in variant_rows:
        family_id = str(row.get("family_id", "") or "").strip()
        if not family_id:
            continue
        family_counts[family_id] += 1
        family_meta.setdefault(
            family_id,
            {
                "lane": str(row.get("lane", "") or ""),
                "timeframe": str(row.get("timeframe", "") or ""),
                "session": str(row.get("session", "") or ""),
                "strategy_type": str(row.get("strategy_type", "") or ""),
            },
        )
    family_bracket_selector = (
        bundle.get("family_bracket_selector", {})
        if isinstance(bundle.get("family_bracket_selector"), dict)
        else {}
    )
    families_payload = (
        family_bracket_selector.get("families", {})
        if isinstance(family_bracket_selector.get("families"), dict)
        else {}
    )
    for family_id in families_payload:
        family_id_text = str(family_id or "").strip()
        if family_id_text and family_id_text not in family_counts:
            family_counts[family_id_text] = 1
            family_meta.setdefault(family_id_text, {})
    lane_inventory = (
        bundle.get("lane_inventory", {})
        if isinstance(bundle.get("lane_inventory"), dict)
        else {}
    )
    return {
        "family_counts": dict(family_counts),
        "family_meta": family_meta,
        "lane_inventory_counts": {
            str(lane): len(values) if isinstance(values, list) else 0
            for lane, values in lane_inventory.items()
        },
        "family_count": int(len(family_counts)),
    }


def _source_family_id(row: Dict[str, Any]) -> str:
    return build_family_id(
        timeframe=row.get("TF", row.get("timeframe", "")),
        session=row.get("Session", row.get("session", "")),
        strategy_type=row.get("Type", row.get("strategy_type", "")),
        threshold=row.get("Thresh", row.get("thresh", 0.0)),
        family_tag=row.get("FamilyTag", row.get("family_tag", "")),
    )


def _strategy_sort_key(row: Dict[str, Any]) -> tuple:
    train = row.get("Train", {}) if isinstance(row.get("Train"), dict) else {}
    recent = row.get("Recent", {}) if isinstance(row.get("Recent"), dict) else {}
    oos = row.get("OOS", {}) if isinstance(row.get("OOS"), dict) else {}
    return (
        safe_float(row.get("Score"), -1e9),
        safe_float(train.get("score_train"), -1e9),
        safe_float(oos.get("profit_factor"), 0.0),
        safe_float(recent.get("profit_factor"), 0.0),
        safe_float(row.get("Trades"), 0.0),
        safe_float(row.get("Avg_PnL"), -1e9),
    )


def _filter_percent_source_db(
    *,
    control_bundle_path: Path,
    percent_source_db_path: Path,
    filtered_db_path: Path,
    audit_json_path: Path,
    audit_csv_path: Path,
) -> Dict[str, Any]:
    control_bundle = json.loads(control_bundle_path.read_text(encoding="utf-8"))
    control_targets = _extract_control_family_targets(control_bundle)
    percent_payload = json.loads(percent_source_db_path.read_text(encoding="utf-8"))
    strategies = percent_payload.get("strategies", [])
    if not isinstance(strategies, list):
        raise SystemExit(f"Invalid percent source DB payload: {percent_source_db_path}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in strategies:
        if not isinstance(row, dict):
            continue
        family_id = _source_family_id(row)
        if family_id:
            grouped[family_id].append(dict(row))

    selected_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    missing_families: List[str] = []
    extra_available_families = sorted(
        family_id
        for family_id in grouped.keys()
        if family_id not in control_targets["family_counts"]
    )
    control_family_counts = control_targets["family_counts"]
    control_family_meta = control_targets["family_meta"]
    for family_id in sorted(control_family_counts.keys()):
        target_count = max(1, int(control_family_counts.get(family_id, 1) or 1))
        candidates = sorted(grouped.get(family_id, []), key=_strategy_sort_key, reverse=True)
        selected = candidates[:target_count]
        if not selected:
            missing_families.append(family_id)
        selected_rows.extend(selected)
        meta = control_family_meta.get(family_id, {})
        exemplar = selected[0] if selected else {}
        audit_rows.append(
            {
                "family_id": family_id,
                "lane": str(meta.get("lane", "")),
                "timeframe": str(exemplar.get("TF", exemplar.get("timeframe", meta.get("timeframe", ""))) or ""),
                "session": str(exemplar.get("Session", exemplar.get("session", meta.get("session", ""))) or ""),
                "strategy_type": str(exemplar.get("Type", exemplar.get("strategy_type", meta.get("strategy_type", ""))) or ""),
                "control_variant_count": int(target_count),
                "percent_available_count": int(len(candidates)),
                "selected_count": int(len(selected)),
                "selected_strategy_ids": [
                    str(row.get("strategy_id", row.get("id", "")) or "")
                    for row in selected
                ],
                "missing": bool(not selected),
            }
        )

    filtered_payload = dict(percent_payload)
    filtered_payload["strategies"] = selected_rows
    filtered_payload["lineage_filter_audit"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_bundle_path": str(control_bundle_path),
        "percent_source_db_path": str(percent_source_db_path),
        "control_family_count": int(len(control_family_counts)),
        "selected_family_count": int(len({row["family_id"] for row in audit_rows if not row["missing"]})),
        "selected_strategy_count": int(len(selected_rows)),
        "missing_family_count": int(len(missing_families)),
        "missing_families": missing_families,
        "extra_available_family_count": int(len(extra_available_families)),
        "extra_available_families_sample": extra_available_families[:50],
        "control_lane_inventory_counts": control_targets["lane_inventory_counts"],
    }
    filtered_db_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_db_path.write_text(
        json.dumps(filtered_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    audit_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_bundle_path": str(control_bundle_path),
        "percent_source_db_path": str(percent_source_db_path),
        "filtered_db_path": str(filtered_db_path),
        "control_targets": control_targets,
        "selected_strategy_count": int(len(selected_rows)),
        "missing_families": missing_families,
        "extra_available_families_sample": extra_available_families[:50],
        "rows": audit_rows,
    }
    _write_json(audit_json_path, audit_payload)
    audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family_id",
                "lane",
                "timeframe",
                "session",
                "strategy_type",
                "control_variant_count",
                "percent_available_count",
                "selected_count",
                "selected_strategy_ids",
                "missing",
            ],
        )
        writer.writeheader()
        for row in audit_rows:
            flat = dict(row)
            flat["selected_strategy_ids"] = "|".join(row.get("selected_strategy_ids", []))
            writer.writerow(flat)
    return audit_payload


def _load_candidate_bundles(candidate_summary_path: Path) -> List[Dict[str, str]]:
    if not candidate_summary_path.exists():
        return []
    try:
        summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    candidates = summary.get("candidates", {}) if isinstance(summary, dict) else {}
    out: List[Dict[str, str]] = []
    if not isinstance(candidates, dict):
        return out
    for name, row in candidates.items():
        if not isinstance(row, dict):
            continue
        bundle_path = str(row.get("bundle_path", "") or "").strip()
        if not bundle_path:
            continue
        out.append({"name": str(name), "bundle_path": bundle_path})
    return out


def _build_validation_command_specs(
    *,
    source_path: Path,
    control_bundle_path: Path,
    base_bundle_path: Path,
    candidate_summary_path: Path,
    artifact_root: Path,
    symbol_mode: str,
    symbol_method: str,
) -> Dict[str, Any]:
    bundles = [
        {"name": "control_point_live", "bundle_path": str(control_bundle_path)},
        {"name": "lineage_percent_base", "bundle_path": str(base_bundle_path)},
    ]
    bundles.extend(_load_candidate_bundles(candidate_summary_path))

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
            report_dir = artifact_root / "backtest_reports" / "validation" / safe_name
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

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "commands": commands,
    }
    _write_json(artifact_root / "recommended_validation_commands.json", payload)
    (artifact_root / "recommended_validation_commands.ps1").write_text(
        "\n".join(ps_lines).strip() + "\n",
        encoding="utf-8",
    )
    return payload


def _build_decision_audit_command_specs(
    *,
    source_path: Path,
    control_bundle_path: Path,
    base_bundle_path: Path,
    candidate_summary_path: Path,
    artifact_root: Path,
    symbol_mode: str,
    symbol_method: str,
    decision_top_k: int,
) -> Dict[str, Any]:
    audit_root = artifact_root / "backtest_reports" / "decision_audit_2025"
    bundles = [{"name": "lineage_percent_base", "bundle_path": str(base_bundle_path)}]
    bundles.extend(_load_candidate_bundles(candidate_summary_path))

    commands: List[Dict[str, Any]] = []
    ps_lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location '{ROOT}'",
        "",
    ]
    control_decisions = audit_root / "control_point_live" / "de3_decisions_2025.csv"
    control_cmd = [
        sys.executable,
        "-u",
        "tools/run_de3_backtest.py",
        "--source",
        str(source_path),
        "--start",
        "2025-01-01",
        "--end",
        "2025-12-31",
        "--symbol-mode",
        symbol_mode,
        "--symbol-method",
        symbol_method,
        "--bundle-path",
        str(control_bundle_path),
        "--sync-entry-model-from-bundle",
        "--report-dir",
        str(audit_root / "control_point_live"),
        "--export-de3-decisions",
        "--de3-decisions-top-k",
        str(max(1, int(decision_top_k))),
        "--de3-decisions-out",
        str(control_decisions),
    ]
    commands.append({"name": "control_backtest_with_decisions", "command": control_cmd})
    ps_lines.append(_quote_ps(control_cmd))
    ps_lines.append("")

    for bundle in bundles:
        safe_name = str(bundle["name"]).replace(" ", "_")
        candidate_decisions = audit_root / safe_name / "de3_decisions_2025.csv"
        backtest_cmd = [
            sys.executable,
            "-u",
            "tools/run_de3_backtest.py",
            "--source",
            str(source_path),
            "--start",
            "2025-01-01",
            "--end",
            "2025-12-31",
            "--symbol-mode",
            symbol_mode,
            "--symbol-method",
            symbol_method,
            "--bundle-path",
            str(bundle["bundle_path"]),
            "--sync-entry-model-from-bundle",
            "--report-dir",
            str(audit_root / safe_name),
            "--export-de3-decisions",
            "--de3-decisions-top-k",
            str(max(1, int(decision_top_k))),
            "--de3-decisions-out",
            str(candidate_decisions),
        ]
        compare_cmd = [
            sys.executable,
            "-u",
            "tools/compare_de3_decision_exports.py",
            "--control-decisions",
            str(control_decisions),
            "--candidate-decisions",
            str(candidate_decisions),
            "--out-dir",
            str(audit_root / "compare" / safe_name),
        ]
        commands.append({"name": f"{safe_name}_backtest_with_decisions", "command": backtest_cmd})
        commands.append({"name": f"{safe_name}_compare_vs_control", "command": compare_cmd})
        ps_lines.append(_quote_ps(backtest_cmd))
        ps_lines.append(_quote_ps(compare_cmd))
        ps_lines.append("")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "commands": commands,
    }
    _write_json(artifact_root / "recommended_decision_audit_commands.json", payload)
    (artifact_root / "recommended_decision_audit_commands.ps1").write_text(
        "\n".join(ps_lines).strip() + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a percent-distance DE3 workflow that stays inside a point-mode live lineage. "
            "The workflow filters a percent source DB down to the control bundle's family set, "
            "rebuilds a v4 bundle from that filtered DB, retrains a scoped entry-policy set, "
            "and writes control-vs-candidate validation/audit command packs."
        )
    )
    parser.add_argument(
        "--control-bundle",
        default="artifacts/de3_v4_live/latest.json",
        help="Point-mode live control bundle. Use the exact April 8 winner path here when known.",
    )
    parser.add_argument(
        "--percent-source-db",
        default="artifacts/de3_ground_up_20260419_pct_full_v1/dynamic_engine3_strategies_v2.outrights.json",
        help="Percent-distance DE3 v2 DB used as the source pool for lineage filtering.",
    )
    parser.add_argument(
        "--source",
        default="es_master_outrights.parquet",
        help="Primary outright-only source parquet.",
    )
    parser.add_argument(
        "--artifact-root",
        default="",
        help="Explicit output root. Defaults to artifacts/de3_live_lineage_percent_<timestamp>.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix used when artifact root is auto-generated.",
    )
    parser.add_argument(
        "--symbol-mode",
        default="auto_by_day",
        help="Symbol mode used for current-pool export and validations.",
    )
    parser.add_argument(
        "--symbol-method",
        default="volume",
        help="Symbol method used for current-pool export and validations.",
    )
    parser.add_argument(
        "--current-pool-start",
        default="2011-01-01",
        help="Start date for current-pool export.",
    )
    parser.add_argument(
        "--current-pool-end",
        default="2024-12-31",
        help="End date for current-pool export.",
    )
    parser.add_argument(
        "--entry-profile-set",
        choices=("all", "current_pool", "shape", "rolling", "decision_direct"),
        default="current_pool",
        help="Named profile subset for entry/decision retraining.",
    )
    parser.add_argument(
        "--candidate-profiles",
        default="",
        help="Optional comma-separated candidate names to train.",
    )
    parser.add_argument(
        "--decision-audit-top-k",
        type=int,
        default=5,
        help="Top-K decision journal depth for the generated audit command pack.",
    )
    parser.add_argument("--skip-filter-db", action="store_true", help="Assume the filtered percent DB already exists.")
    parser.add_argument("--skip-v4", action="store_true", help="Assume the lineage percent bundle already exists.")
    parser.add_argument("--skip-current-pool", action="store_true", help="Assume current-pool exports already exist.")
    parser.add_argument("--skip-entry-policy", action="store_true", help="Skip entry-policy candidate retraining.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests and commands without executing stages.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    control_bundle_path = _resolve_path(str(args.control_bundle))
    percent_source_db_path = _resolve_path(str(args.percent_source_db))
    source_path = _resolve_path(str(args.source))
    _require_existing_paths(
        reason="workflow setup",
        paths=[control_bundle_path, percent_source_db_path, source_path],
    )

    if str(args.artifact_root or "").strip():
        artifact_root = _resolve_path(str(args.artifact_root))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{str(args.tag).strip()}" if str(args.tag).strip() else ""
        artifact_root = ROOT / "artifacts" / f"de3_live_lineage_percent_{ts}{suffix}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    filtered_db_path = artifact_root / "dynamic_engine3_strategies_v2.lineage_percent.json"
    lineage_audit_json = artifact_root / "lineage_family_audit.json"
    lineage_audit_csv = artifact_root / "lineage_family_audit.csv"
    v4_bundle_path = artifact_root / "dynamic_engine3_v4_bundle.lineage_percent.json"
    v4_reports_dir = artifact_root / "reports" / "lineage_bundle"
    current_pool_decisions_path = artifact_root / "reports" / "de3_current_pool_2011_2024.csv"
    current_pool_trade_path = artifact_root / "reports" / "de3_current_pool_2011_2024_trade_attribution.csv"
    entry_policy_dir = artifact_root / "entry_policy_candidates"
    candidate_summary_path = entry_policy_dir / "candidate_summary.json"

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "settings": {
            "control_bundle": str(control_bundle_path),
            "percent_source_db": str(percent_source_db_path),
            "source_path": str(source_path),
            "symbol_mode": str(args.symbol_mode),
            "symbol_method": str(args.symbol_method),
            "current_pool_start": str(args.current_pool_start),
            "current_pool_end": str(args.current_pool_end),
            "entry_profile_set": str(args.entry_profile_set),
            "candidate_profiles": _parse_csv_list(str(args.candidate_profiles)),
            "decision_audit_top_k": int(args.decision_audit_top_k),
            "skip_filter_db": bool(args.skip_filter_db),
            "skip_v4": bool(args.skip_v4),
            "skip_current_pool": bool(args.skip_current_pool),
            "skip_entry_policy": bool(args.skip_entry_policy),
            "dry_run": bool(args.dry_run),
        },
        "artifacts": {
            "filtered_db_path": str(filtered_db_path),
            "lineage_audit_json": str(lineage_audit_json),
            "lineage_audit_csv": str(lineage_audit_csv),
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

    if bool(args.skip_filter_db) and not bool(args.dry_run):
        _require_existing_paths(
            reason="--skip-filter-db was set",
            paths=[filtered_db_path, lineage_audit_json, lineage_audit_csv],
        )
    if not bool(args.skip_filter_db):
        def _filter_runner(handle) -> None:
            audit = _filter_percent_source_db(
                control_bundle_path=control_bundle_path,
                percent_source_db_path=percent_source_db_path,
                filtered_db_path=filtered_db_path,
                audit_json_path=lineage_audit_json,
                audit_csv_path=lineage_audit_csv,
            )
            handle.write(json.dumps(audit, indent=2, ensure_ascii=True))
            handle.write("\n")

        _run_inline_stage(
            name="filter_percent_source_db_to_live_lineage",
            artifact_root=artifact_root,
            manifest=manifest,
            dry_run=bool(args.dry_run),
            expected_paths=[filtered_db_path, lineage_audit_json, lineage_audit_csv],
            runner=_filter_runner,
        )

    if bool(args.skip_v4) and not bool(args.dry_run):
        _require_existing_paths(reason="--skip-v4 was set", paths=[v4_bundle_path])
    if not bool(args.skip_v4):
        _run_stage(
            name="de3_v4_build",
            command=[
                sys.executable,
                "-u",
                "de3_v4_trainer.py",
                "--source-db",
                str(filtered_db_path),
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
            expected_paths=[v4_bundle_path],
        )

    if bool(args.skip_current_pool) and not bool(args.dry_run):
        _require_existing_paths(
            reason="--skip-current-pool was set",
            paths=[current_pool_decisions_path, current_pool_trade_path],
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
            expected_paths=[current_pool_decisions_path, current_pool_trade_path],
        )

    if bool(args.skip_entry_policy) and not bool(args.dry_run):
        _require_existing_paths(
            reason="--skip-entry-policy was set",
            paths=[candidate_summary_path],
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
            "--profile-set",
            str(args.entry_profile_set),
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
            expected_paths=[candidate_summary_path],
        )

    if not bool(args.dry_run):
        _require_existing_paths(
            reason="workflow requires the lineage percent bundle before command packs can be written",
            paths=[v4_bundle_path],
        )

    validation_payload = _build_validation_command_specs(
        source_path=source_path,
        control_bundle_path=control_bundle_path,
        base_bundle_path=v4_bundle_path,
        candidate_summary_path=candidate_summary_path,
        artifact_root=artifact_root,
        symbol_mode=str(args.symbol_mode),
        symbol_method=str(args.symbol_method),
    )
    decision_audit_payload = _build_decision_audit_command_specs(
        source_path=source_path,
        control_bundle_path=control_bundle_path,
        base_bundle_path=v4_bundle_path,
        candidate_summary_path=candidate_summary_path,
        artifact_root=artifact_root,
        symbol_mode=str(args.symbol_mode),
        symbol_method=str(args.symbol_method),
        decision_top_k=max(1, int(args.decision_audit_top_k or 1)),
    )
    manifest["validation_commands_path"] = str(artifact_root / "recommended_validation_commands.json")
    manifest["validation_powershell_path"] = str(artifact_root / "recommended_validation_commands.ps1")
    manifest["validation_command_count"] = int(len(validation_payload.get("commands", [])))
    manifest["decision_audit_commands_path"] = str(artifact_root / "recommended_decision_audit_commands.json")
    manifest["decision_audit_powershell_path"] = str(artifact_root / "recommended_decision_audit_commands.ps1")
    manifest["decision_audit_command_count"] = int(len(decision_audit_payload.get("commands", [])))
    _write_json(artifact_root / "workflow_manifest.json", manifest)

    print(f"artifact_root={artifact_root}")
    print(f"workflow_manifest={artifact_root / 'workflow_manifest.json'}")
    print(f"validation_commands={artifact_root / 'recommended_validation_commands.json'}")
    print(f"decision_audit_commands={artifact_root / 'recommended_decision_audit_commands.json'}")


if __name__ == "__main__":
    main()

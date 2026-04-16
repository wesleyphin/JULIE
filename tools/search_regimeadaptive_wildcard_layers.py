import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_regimeadaptive_generalized_artifact import build_generalized_artifact
from tools.regimeadaptive_filterless_runner import NY_TZ
from tools.train_regimeadaptive_v2 import _json_safe


DEFAULT_SCENARIOS = [
    {
        "name": "ny_quarter_day_strict",
        "template_names": ["quarter_day_session", "day_session", "quarter_session"],
        "min_count": 2,
        "min_share": 1.0,
        "include_sessions": ["NY_AM", "NY_PM"],
        "active_only": True,
        "group_policy_priority": "fill_only",
    },
    {
        "name": "ny_session_only_strict",
        "template_names": ["session_only", "quarter_session"],
        "min_count": 2,
        "min_share": 1.0,
        "include_sessions": ["NY_AM", "NY_PM"],
        "active_only": True,
        "group_policy_priority": "fill_only",
    },
    {
        "name": "all_quarter_day_strict",
        "template_names": ["quarter_day_session", "day_session", "quarter_session"],
        "min_count": 2,
        "min_share": 1.0,
        "include_sessions": [],
        "active_only": True,
        "group_policy_priority": "fill_only",
    },
    {
        "name": "all_session_only_strict",
        "template_names": ["session_only", "quarter_session"],
        "min_count": 2,
        "min_share": 1.0,
        "include_sessions": [],
        "active_only": True,
        "group_policy_priority": "fill_only",
    },
    {
        "name": "all_quarter_day_relaxed",
        "template_names": ["quarter_day_session", "day_session", "quarter_session"],
        "min_count": 2,
        "min_share": 0.67,
        "include_sessions": [],
        "active_only": True,
        "group_policy_priority": "fill_only",
    },
    {
        "name": "ny_weekday_strict",
        "template_names": ["week_day_session", "quarter_day_session", "day_session"],
        "min_count": 2,
        "min_share": 1.0,
        "include_sessions": ["NY_AM", "NY_PM"],
        "active_only": True,
        "group_policy_priority": "fill_only",
    },
]


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _copy_gate_model_local(source_artifact: Path, output_artifact: Path) -> None:
    source_gate_model = source_artifact.parent / "regimeadaptive_gate_model.joblib"
    if not source_gate_model.is_file():
        return
    target_gate_model = output_artifact.parent / source_gate_model.name
    shutil.copyfile(source_gate_model, target_gate_model)
    payload = json.loads(output_artifact.read_text(encoding="utf-8"))
    signal_gate = payload.get("signal_gate", {}) if isinstance(payload.get("signal_gate", {}), dict) else {}
    if signal_gate:
        signal_gate["model_path"] = str(target_gate_model.name)
        payload["signal_gate"] = signal_gate
        output_artifact.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _run(command: list[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    print(f"RUN {printable}")
    subprocess.run(command, cwd=str(cwd), check=True)


def _latest_backtest_report(out_dir: Path) -> Path:
    reports = sorted(out_dir.glob("backtest_regimeadaptive_robust_*.json"))
    if not reports:
        raise SystemExit(f"No backtest report found in {out_dir}")
    return reports[-1]


def _report_summary(report_path: Path) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    result = payload.get("result", {}) if isinstance(payload.get("result", {}), dict) else {}
    return {
        "report_path": str(report_path),
        "equity": result.get("equity"),
        "trades": result.get("trades"),
        "max_drawdown": result.get("max_drawdown"),
        "profit_factor": result.get("profit_factor"),
        "daily_sharpe": result.get("daily_sharpe"),
        "sessions": result.get("sessions", {}),
    }


def _group_counts(artifact_path: Path) -> dict:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    group_signal_policies = payload.get("group_signal_policies", {}) if isinstance(payload.get("group_signal_policies", {}), dict) else {}
    session_counts = Counter()
    side_counts = Counter()
    for pattern, side_map in group_signal_policies.items():
        parts = [str(part or "").strip().upper() for part in str(pattern or "").split("_")]
        session_name = "_".join(parts[3:]) if len(parts) >= 4 else "UNK"
        session_counts[session_name] += 1
        if isinstance(side_map, dict):
            side_counts.update(side_map.keys())
    return {
        "group_pattern_count": int(len(group_signal_policies)),
        "group_session_counts": dict(session_counts),
        "group_side_counts": dict(side_counts),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search anchored RegimeAdaptive wildcard overlay artifacts and validate them on holdout plus fresh windows."
    )
    parser.add_argument("--source-artifact", default="artifacts/regimeadaptive_v19_liveplus_balanced_ny_candidate/latest.json")
    parser.add_argument("--backtest-source", default="es_master_outrights.parquet")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--holdout-end", default="2025-12-31")
    parser.add_argument("--fresh-start", default="2026-01-01")
    parser.add_argument("--fresh-end", default="2026-01-26")
    parser.add_argument("--gate-threshold-session-override", action="append", default=[])
    parser.add_argument("--search-root", default="artifacts/regimeadaptive_wildcard_overlay_search")
    parser.add_argument("--scenario", action="append", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source_artifact = _resolve_path(str(args.source_artifact), "artifacts/regimeadaptive_v19_liveplus_balanced_ny_candidate/latest.json")
    search_root = _resolve_path(str(args.search_root), "artifacts/regimeadaptive_wildcard_overlay_search")
    search_root.mkdir(parents=True, exist_ok=True)
    reports_root = search_root.parent / f"{search_root.name}_reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    selected_names = {str(name).strip() for name in list(args.scenario or []) if str(name).strip()}
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if not selected_names or str(scenario["name"]) in selected_names]
    if not scenarios:
        raise SystemExit("No scenarios selected.")

    python_exe = Path(sys.executable).resolve()
    rows = []
    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        artifact_dir = search_root / scenario_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_artifact = artifact_dir / "latest.json"

        build_generalized_artifact(
            source_artifact=source_artifact,
            output_artifact=output_artifact,
            template_names=list(scenario["template_names"]),
            min_count=int(scenario["min_count"]),
            min_share=float(scenario["min_share"]),
            active_only=bool(scenario["active_only"]),
            include_sessions=set(str(item).upper() for item in list(scenario["include_sessions"])),
            merge_existing_group_policies=False,
            group_policy_priority=str(scenario["group_policy_priority"]),
        )
        _copy_gate_model_local(source_artifact, output_artifact)

        scenario_reports_root = reports_root / scenario_name
        holdout_dir = scenario_reports_root / "holdout2025"
        fresh_dir = scenario_reports_root / "fresh2026"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        fresh_dir.mkdir(parents=True, exist_ok=True)

        holdout_cmd = [
            str(python_exe),
            "tools/backtest_regimeadaptive_robust.py",
            "--source",
            str(args.backtest_source),
            "--artifact",
            str(output_artifact),
            "--start",
            str(args.holdout_start),
            "--end",
            str(args.holdout_end),
            "--contracts",
            str(int(args.contracts)),
            "--out-dir",
            str(holdout_dir),
        ]
        fresh_cmd = [
            str(python_exe),
            "tools/backtest_regimeadaptive_robust.py",
            "--source",
            str(args.backtest_source),
            "--artifact",
            str(output_artifact),
            "--start",
            str(args.fresh_start),
            "--end",
            str(args.fresh_end),
            "--contracts",
            str(int(args.contracts)),
            "--out-dir",
            str(fresh_dir),
        ]
        for override in list(args.gate_threshold_session_override or []):
            holdout_cmd.extend(["--gate-threshold-session-override", str(override)])
            fresh_cmd.extend(["--gate-threshold-session-override", str(override)])
        _run(holdout_cmd, cwd=ROOT)
        _run(fresh_cmd, cwd=ROOT)

        row = {
            "scenario": scenario_name,
            "scenario_config": {key: _json_safe(value) for key, value in scenario.items()},
            "artifact_path": str(output_artifact),
            **_group_counts(output_artifact),
            "holdout_2025": _report_summary(_latest_backtest_report(holdout_dir)),
            "fresh_2026": _report_summary(_latest_backtest_report(fresh_dir)),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float((row.get("fresh_2026", {}) or {}).get("equity", float("-inf")) or float("-inf")),
            float((row.get("holdout_2025", {}) or {}).get("equity", float("-inf")) or float("-inf")),
        ),
        reverse=True,
    )

    output_payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "source_artifact": str(source_artifact),
        "search_root": str(search_root),
        "rows": [{key: _json_safe(value) for key, value in row.items()} for row in rows],
    }
    output_path = reports_root / "search_summary.json"
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"search_summary={output_path}")


if __name__ == "__main__":
    main()

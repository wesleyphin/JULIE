import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.regimeadaptive_filterless_runner import NY_TZ  # noqa: E402
from tools.train_regimeadaptive_v2 import _json_safe  # noqa: E402


DEFAULT_SCENARIOS: list[dict] = [
    {
        "name": "dense_813_base",
        "rule_specs": "8:21:0.0,13:34:0.0",
        "min_total_support": 120,
        "min_recent_support": 12,
        "min_positive_oos_splits": 1,
        "min_robust_edge": 0.0,
        "rank_metrics": "score,robust_edge",
        "top_n_grid": "36,48,60,72,84,96",
        "min_target_trades": 2500,
        "min_trade_day_coverage": 0.0,
        "trade_day_coverage_weight": 0.0,
    },
    {
        "name": "dense_813_2050_base",
        "rule_specs": "8:21:0.0,13:34:0.0,20:50:0.0",
        "min_total_support": 120,
        "min_recent_support": 12,
        "min_positive_oos_splits": 1,
        "min_robust_edge": 0.0,
        "rank_metrics": "score,robust_edge",
        "top_n_grid": "36,48,60,72,84,96",
        "min_target_trades": 2500,
        "min_trade_day_coverage": 0.0,
        "trade_day_coverage_weight": 0.0,
    },
    {
        "name": "dense_813_2050_cov",
        "rule_specs": "8:21:0.0,13:34:0.0,20:50:0.0",
        "min_total_support": 120,
        "min_recent_support": 12,
        "min_positive_oos_splits": 1,
        "min_robust_edge": 0.0,
        "rank_metrics": "score,robust_edge",
        "top_n_grid": "48,60,72,84,96,108",
        "min_target_trades": 2800,
        "min_trade_day_coverage": 0.22,
        "trade_day_coverage_weight": 4000.0,
    },
    {
        "name": "dense_813_2050_34100_cov",
        "rule_specs": "8:21:0.0,13:34:0.0,20:50:0.0,34:100:0.0",
        "min_total_support": 120,
        "min_recent_support": 12,
        "min_positive_oos_splits": 1,
        "min_robust_edge": 0.0,
        "rank_metrics": "score,robust_edge",
        "top_n_grid": "48,60,72,84,96,108",
        "min_target_trades": 2800,
        "min_trade_day_coverage": 0.22,
        "trade_day_coverage_weight": 4000.0,
    },
]


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _run(command: list[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    print(f"RUN {printable}")
    subprocess.run(command, cwd=str(cwd), check=True)


def _artifact_sessions(artifact_path: Path) -> list[str]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    signal_policies = payload.get("signal_policies", {}) if isinstance(payload.get("signal_policies", {}), dict) else {}
    sessions: list[str] = []
    for combo_key in signal_policies.keys():
        parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
        if len(parts) < 4:
            continue
        session_name = "_".join(parts[3:])
        if session_name and session_name not in sessions:
            sessions.append(session_name)
    return [name for name in ("ASIA", "LONDON", "NY_AM", "NY_PM") if name in sessions]


def _load_walkforward_report(report_path: Path) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "report_path": str(report_path),
        "artifact_path": str(payload.get("candidate_artifact_path") or ""),
        "stable_threshold": payload.get("stable_threshold"),
        "stable_session_thresholds": payload.get("stable_session_thresholds", {}),
        "oos_summary": payload.get("oos_summary", {}),
        "holdout_summary": payload.get("holdout_summary", {}),
    }


def _score_key(row: dict) -> tuple[float, float]:
    oos_summary = row.get("oos_summary", {}) if isinstance(row.get("oos_summary", {}), dict) else {}
    holdout_summary = row.get("holdout_summary", {}) if isinstance(row.get("holdout_summary", {}), dict) else {}
    return (
        float(oos_summary.get("score", float("-inf")) or float("-inf")),
        float(holdout_summary.get("equity", float("-inf")) or float("-inf")),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small dense RegimeAdaptive remodel sweep and score each dense artifact with walk-forward gating."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--baseline-artifact", default="artifacts/regimeadaptive_v19_live/latest.json")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--search-root", default="artifacts/regimeadaptive_dense_search")
    parser.add_argument("--scenario", action="append", default=[], help="Optional scenario name filter.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    search_root = _resolve_path(str(args.search_root), "artifacts/regimeadaptive_dense_search")
    search_root.mkdir(parents=True, exist_ok=True)

    selected_names = {str(name).strip() for name in list(args.scenario or []) if str(name).strip()}
    scenarios = [scenario for scenario in DEFAULT_SCENARIOS if not selected_names or str(scenario["name"]) in selected_names]
    if not scenarios:
        raise SystemExit("No scenarios selected.")

    python_exe = Path(sys.executable).resolve()
    rows: list[dict] = []
    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        scenario_root = search_root / scenario_name
        train_root = scenario_root / "train"
        walkforward_root = scenario_root / "walkforward"
        train_root.mkdir(parents=True, exist_ok=True)
        walkforward_root.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            str(python_exe),
            "tools/train_regimeadaptive_dense.py",
            "--source",
            str(args.source),
            "--baseline-artifact",
            str(args.baseline_artifact),
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--contracts",
            str(int(args.contracts)),
            "--valid-years",
            str(int(args.valid_years)),
            "--test-years",
            str(int(args.test_years)),
            "--rule-specs",
            str(scenario["rule_specs"]),
            "--min-total-support",
            str(int(scenario["min_total_support"])),
            "--min-recent-support",
            str(int(scenario["min_recent_support"])),
            "--min-positive-oos-splits",
            str(int(scenario["min_positive_oos_splits"])),
            "--min-robust-edge",
            str(float(scenario["min_robust_edge"])),
            "--rank-metrics",
            str(scenario["rank_metrics"]),
            "--top-n-grid",
            str(scenario["top_n_grid"]),
            "--min-target-trades",
            str(int(scenario["min_target_trades"])),
            "--min-trade-day-coverage",
            str(float(scenario["min_trade_day_coverage"])),
            "--trade-day-coverage-weight",
            str(float(scenario["trade_day_coverage_weight"])),
            "--artifact-root",
            str(train_root),
            "--write-latest",
        ]
        _run(train_cmd, cwd=ROOT)

        artifact_path = train_root / "regimeadaptive_dense_artifact.json"
        if not artifact_path.is_file():
            raise SystemExit(f"Expected train artifact missing: {artifact_path}")

        threshold_sessions = _artifact_sessions(artifact_path)
        walkforward_cmd = [
            str(python_exe),
            "tools/train_regimeadaptive_gate_walkforward.py",
            "--source",
            str(args.source),
            "--artifact",
            str(artifact_path),
            "--start",
            "2011-01-01",
            "--end",
            "2025-12-31",
            "--contracts",
            str(int(args.contracts)),
            "--model",
            "hgb",
            "--min-train-years",
            "5",
            "--valid-years",
            "1",
            "--test-years",
            "1",
            "--first-test-year",
            "2018",
            "--final-holdout-years",
            "1",
            "--threshold-sessions",
            ",".join(threshold_sessions),
            "--min-session-valid-signals",
            "60",
            "--min-session-threshold-fold-support",
            "2",
            "--min-valid-trades",
            "180",
            "--min-valid-trade-ratio",
            "0.35",
            "--require-positive-valid",
            "--require-valid-pf-above-one",
            "--trade-count-weight",
            "0.0",
            "--max-drawdown-penalty",
            "0.08",
            "--negative-year-penalty",
            "400",
            "--artifact-root",
            str(walkforward_root),
            "--write-latest",
        ]
        _run(walkforward_cmd, cwd=ROOT)

        report_path = walkforward_root / "walkforward_report.json"
        report_row = _load_walkforward_report(report_path)
        report_row["scenario"] = scenario_name
        report_row["train_artifact_path"] = str(artifact_path)
        report_row["threshold_sessions"] = threshold_sessions
        report_row["scenario_config"] = {key: _json_safe(value) for key, value in scenario.items()}
        rows.append(report_row)

    rows.sort(key=_score_key, reverse=True)
    output_payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "baseline_artifact": str(args.baseline_artifact),
        "search_root": str(search_root),
        "rows": [{key: _json_safe(value) for key, value in row.items()} for row in rows],
    }
    output_path = search_root / "search_summary.json"
    output_path.write_text(json.dumps(_json_safe(output_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"search_summary={output_path}")


if __name__ == "__main__":
    main()

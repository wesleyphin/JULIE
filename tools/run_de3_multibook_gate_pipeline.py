from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


BROAD_BUNDLE = ROOT / "artifacts/de3_v4_live/latest.json"
BALANCED_BUNDLE = ROOT / "artifacts/de3_decision_policy_pfcore_20260330/dynamic_engine3_v4_bundle.decision_direct_compare_sideaware_pfstrict_core_balanced2.json"
SHORTFLOOR_BUNDLE = ROOT / "artifacts/de3_decision_policy_pfcore_20260330/dynamic_engine3_v4_bundle.decision_direct_compare_sideaware_pfstrict_core_shortfloor.json"

EXPORT_WINDOWS = ("2011-01-01", "2024-12-31")
SHORTLIST_WINDOWS = [
    ("2022", "2022-01-01", "2022-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026_jan", "2026-01-01", "2026-01-26"),
]
FULL_WINDOWS = [
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026_jan", "2026-01-01", "2026-01-26"),
    ("2024_2025", "2024-01-01", "2025-12-31"),
    ("2022_2026jan", "2022-01-01", "2026-01-26"),
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _trade_attr_path(decisions_csv: Path) -> Path:
    return decisions_csv.with_name(f"{decisions_csv.stem}_trade_attribution.csv")


def _run_python(
    *,
    script: str,
    args: List[str],
    log_prefix: Path,
    skip_if_exists: bool = False,
    expected_paths: List[Path] | None = None,
) -> Dict[str, Any]:
    expected_paths = expected_paths or []
    if skip_if_exists and expected_paths and all(path.exists() and path.stat().st_size > 0 for path in expected_paths):
        return {
            "status": "skipped_existing",
            "command": [sys.executable, script, *args],
            "stdout_log": str(log_prefix.with_suffix(".stdout.log")),
            "stderr_log": str(log_prefix.with_suffix(".stderr.log")),
        }

    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_log = log_prefix.with_suffix(".stdout.log")
    stderr_log = log_prefix.with_suffix(".stderr.log")
    stdout_log.write_text("", encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")
    cmd = [sys.executable, script, *args]
    with stdout_log.open("w", encoding="utf-8") as out_handle, stderr_log.open("w", encoding="utf-8") as err_handle:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=out_handle,
            stderr=err_handle,
            text=True,
            check=False,
        )
    status = "ok" if result.returncode == 0 else "failed"
    if result.returncode != 0:
        tail = stderr_log.read_text(encoding="utf-8", errors="ignore")[-4000:]
        raise RuntimeError(f"{script} failed with exit code {result.returncode}\n{tail}")
    return {
        "status": status,
        "command": cmd,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def _export_book(
    *,
    name: str,
    bundle_path: Path,
    decisions_csv: Path,
    report_dir: Path,
    logs_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    trade_attr = _trade_attr_path(decisions_csv)
    result = _run_python(
        script="tools/run_de3_backtest.py",
        args=[
            "--start",
            EXPORT_WINDOWS[0],
            "--end",
            EXPORT_WINDOWS[1],
            "--bundle-path",
            str(bundle_path),
            "--sync-entry-model-from-bundle",
            "--export-de3-decisions",
            "--de3-decisions-out",
            str(decisions_csv),
            "--report-dir",
            str(report_dir),
        ],
        log_prefix=logs_dir / f"export_{name}_2011_2024",
        skip_if_exists=not force,
        expected_paths=[decisions_csv, trade_attr],
    )
    result.update(
        {
            "name": name,
            "bundle_path": str(bundle_path),
            "decisions_csv": str(decisions_csv),
            "trade_attribution_csv": str(trade_attr),
        }
    )
    return result


def _profit_factor(pnls: List[float]) -> float:
    gross_win = sum(v for v in pnls if v > 0.0)
    gross_loss = -sum(v for v in pnls if v < 0.0)
    if gross_loss <= 0.0:
        return float("inf") if gross_win > 0.0 else 0.0
    return float(gross_win / gross_loss)


def _sqn(pnls: List[float]) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((v - mean) ** 2 for v in pnls) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return float((mean / std) * math.sqrt(n))


def _daily_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    daily: Dict[str, float] = defaultdict(float)
    for trade in trades:
        ts_raw = trade.get("exit_time") or trade.get("entry_time")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            continue
        day_key = ts.date().isoformat()
        daily[day_key] += _safe_float(trade.get("pnl_net"), 0.0)
    values = list(daily.values())
    if len(values) < 2:
        return {"daily_sharpe": 0.0, "daily_sortino": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(var) if var > 0.0 else 0.0
    downside = [v for v in values if v < 0.0]
    if len(downside) >= 2:
        d_mean = sum(downside) / len(downside)
        d_var = sum((v - d_mean) ** 2 for v in downside) / (len(downside) - 1)
        d_std = math.sqrt(d_var) if d_var > 0.0 else 0.0
    else:
        d_std = 0.0
    sharpe = 0.0 if std <= 1e-12 else float((mean / std) * math.sqrt(252.0))
    sortino = 0.0 if d_std <= 1e-12 else float((mean / d_std) * math.sqrt(252.0))
    return {"daily_sharpe": sharpe, "daily_sortino": sortino}


def _load_metrics(report_path: Path) -> Dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    trades = payload.get("trade_log", []) if isinstance(payload.get("trade_log"), list) else []
    pnls = [_safe_float(trade.get("pnl_net"), 0.0) for trade in trades if isinstance(trade, dict)]
    side_counts = defaultdict(int)
    for trade in trades:
        side = str(trade.get("side", "") or "").strip().lower()
        if side:
            side_counts[side] += 1
    trade_count = len(trades)
    long_trades = int(side_counts.get("long", 0))
    short_trades = int(side_counts.get("short", 0))
    daily = _daily_metrics(trades)
    return {
        "report_path": str(report_path),
        "equity": float(_safe_float(summary.get("equity"), 0.0)),
        "trades": int(_safe_float(summary.get("trades"), trade_count)),
        "winrate": float(_safe_float(summary.get("winrate"), 0.0)),
        "max_drawdown": float(_safe_float(summary.get("max_drawdown"), 0.0)),
        "profit_factor": float(_profit_factor(pnls)),
        "sqn": float(_sqn(pnls)),
        "daily_sharpe": float(daily["daily_sharpe"]),
        "daily_sortino": float(daily["daily_sortino"]),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_share": float(long_trades / trade_count) if trade_count > 0 else 0.0,
        "short_share": float(short_trades / trade_count) if trade_count > 0 else 0.0,
    }


def _backtest_bundle(
    *,
    bundle_path: Path,
    bundle_name: str,
    label: str,
    start: str,
    end: str,
    output_dir: Path,
    logs_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    report_dir = output_dir / f"{bundle_name}_{label}"
    expected_report = None
    if report_dir.exists():
        reports = sorted(report_dir.glob("backtest_*.json"))
        reports = [path for path in reports if not path.name.endswith("_monte_carlo.json")]
        if reports:
            expected_report = reports[-1]
    result = _run_python(
        script="tools/run_de3_backtest.py",
        args=[
            "--start",
            start,
            "--end",
            end,
            "--bundle-path",
            str(bundle_path),
            "--sync-entry-model-from-bundle",
            "--report-dir",
            str(report_dir),
        ],
        log_prefix=logs_dir / f"backtest_{bundle_name}_{label}",
        skip_if_exists=not force and expected_report is not None,
        expected_paths=[expected_report] if expected_report is not None else [],
    )
    if expected_report is None or force:
        reports = sorted(report_dir.glob("backtest_*.json"))
        reports = [path for path in reports if not path.name.endswith("_monte_carlo.json")]
        if not reports:
            raise RuntimeError(f"No report generated in {report_dir}")
        expected_report = reports[-1]
    metrics = _load_metrics(expected_report)
    return {
        "bundle_name": bundle_name,
        "bundle_path": str(bundle_path),
        "window": label,
        "start": start,
        "end": end,
        "status": result["status"],
        "stdout_log": result["stdout_log"],
        "stderr_log": result["stderr_log"],
        "metrics": metrics,
    }


def _selection_score(candidate: Dict[str, Dict[str, Any]], baseline: Dict[str, Dict[str, Any]]) -> float:
    score = 0.0
    for label, weight in [("2022", 0.9), ("2025", 1.4), ("2026_jan", 1.2)]:
        c = candidate[label]
        b = baseline[label]
        score += weight * (c["equity"] - b["equity"])
        score += weight * 2400.0 * (c["profit_factor"] - b["profit_factor"])
        score += weight * 600.0 * (c["daily_sharpe"] - b["daily_sharpe"])
        score += weight * 220.0 * (c["daily_sortino"] - b["daily_sortino"])
        score += weight * 150.0 * (c["sqn"] - b["sqn"])
        score -= weight * 0.7 * (c["max_drawdown"] - b["max_drawdown"])
        score += weight * 400.0 * (b["long_share"] - c["long_share"])
    return float(score)


def _write_summary_md(
    *,
    path: Path,
    baseline_name: str,
    winner_name: str,
    results_by_bundle: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    lines: List[str] = []
    lines.append("# DE3 Multi-Book Gate Evaluation")
    lines.append("")
    lines.append(f"- baseline: `{baseline_name}`")
    lines.append(f"- selected winner: `{winner_name}`")
    lines.append("")
    for window in FULL_WINDOWS:
        label = window[0]
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| bundle | equity | pf | sharpe | sortino | sqn | max_dd | long_share | longs | shorts |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for bundle_name, bundle_results in results_by_bundle.items():
            metrics = bundle_results[label]
            lines.append(
                "| "
                + " | ".join(
                    [
                        bundle_name,
                        f"{metrics['equity']:.2f}",
                        f"{metrics['profit_factor']:.4f}",
                        f"{metrics['daily_sharpe']:.4f}",
                        f"{metrics['daily_sortino']:.4f}",
                        f"{metrics['sqn']:.4f}",
                        f"{metrics['max_drawdown']:.2f}",
                        f"{metrics['long_share']:.4f}",
                        str(metrics["long_trades"]),
                        str(metrics["short_trades"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DE3 multi-book gate export/train/validation pipeline.")
    parser.add_argument(
        "--artifact-root",
        default=f"artifacts/de3_multibook_gate_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Artifact root for gate bundles and summaries.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing export/backtest files when present.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=2,
        help="How many trained gate candidates to shortlist for the full validation pass.",
    )
    args = parser.parse_args()

    artifact_root = _resolve(args.artifact_root)
    logs_dir = artifact_root / "logs"
    eval_dir = artifact_root / "validation"
    report_dir = artifact_root / "reports"
    artifact_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    exports = {
        "broad": _export_book(
            name="broad",
            bundle_path=BROAD_BUNDLE,
            decisions_csv=_resolve("reports/de3_multibook_gate_broad_2011_2024.csv"),
            report_dir=_resolve("backtest_reports/de3_multibook_gate_exports/broad_2011_2024"),
            logs_dir=logs_dir,
            force=not args.skip_existing,
        ),
        "balanced2": _export_book(
            name="balanced2",
            bundle_path=BALANCED_BUNDLE,
            decisions_csv=_resolve("reports/de3_multibook_gate_balanced2_2011_2024.csv"),
            report_dir=_resolve("backtest_reports/de3_multibook_gate_exports/balanced2_2011_2024"),
            logs_dir=logs_dir,
            force=not args.skip_existing,
        ),
        "shortfloor": _export_book(
            name="shortfloor",
            bundle_path=SHORTFLOOR_BUNDLE,
            decisions_csv=_resolve("reports/de3_multibook_gate_shortfloor_2011_2024.csv"),
            report_dir=_resolve("backtest_reports/de3_multibook_gate_exports/shortfloor_2011_2024"),
            logs_dir=logs_dir,
            force=not args.skip_existing,
        ),
    }

    trainer_output = artifact_root / "trained_bundles"
    trainer_result = _run_python(
        script="tools/train_de3_multibook_gate.py",
        args=[
            "--base-bundle",
            str(BROAD_BUNDLE),
            "--base-decisions-csv",
            exports["broad"]["decisions_csv"],
            "--base-trade-attribution-csv",
            exports["broad"]["trade_attribution_csv"],
            "--balanced-bundle",
            str(BALANCED_BUNDLE),
            "--balanced-decisions-csv",
            exports["balanced2"]["decisions_csv"],
            "--balanced-trade-attribution-csv",
            exports["balanced2"]["trade_attribution_csv"],
            "--shortfloor-bundle",
            str(SHORTFLOOR_BUNDLE),
            "--shortfloor-decisions-csv",
            exports["shortfloor"]["decisions_csv"],
            "--shortfloor-trade-attribution-csv",
            exports["shortfloor"]["trade_attribution_csv"],
            "--output-dir",
            str(trainer_output),
        ],
        log_prefix=logs_dir / "train_multibook_gate",
        skip_if_exists=args.skip_existing,
        expected_paths=[trainer_output / "candidate_summary.json"],
    )
    candidate_summary_path = trainer_output / "candidate_summary.json"
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))

    baseline_shortlist: Dict[str, Dict[str, Any]] = {}
    for label, start, end in SHORTLIST_WINDOWS:
        baseline_shortlist[label] = _backtest_bundle(
            bundle_path=BROAD_BUNDLE,
            bundle_name="baseline",
            label=label,
            start=start,
            end=end,
            output_dir=eval_dir / "shortlist",
            logs_dir=logs_dir,
            force=not args.skip_existing,
        )["metrics"]

    candidate_scores: List[Dict[str, Any]] = []
    for row in candidate_summary:
        bundle_name = str(row["name"])
        bundle_path = _resolve(str(row["bundle_path"]))
        metrics_by_window: Dict[str, Dict[str, Any]] = {}
        for label, start, end in SHORTLIST_WINDOWS:
            metrics_by_window[label] = _backtest_bundle(
                bundle_path=bundle_path,
                bundle_name=bundle_name,
                label=label,
                start=start,
                end=end,
                output_dir=eval_dir / "shortlist",
                logs_dir=logs_dir,
                force=not args.skip_existing,
            )["metrics"]
        candidate_scores.append(
            {
                "name": bundle_name,
                "bundle_path": str(bundle_path),
                "metrics": metrics_by_window,
                "selection_score": _selection_score(metrics_by_window, baseline_shortlist),
            }
        )
    candidate_scores.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    shortlist = candidate_scores[: max(1, int(args.top_candidates))]

    final_bundles = {"baseline": str(BROAD_BUNDLE)}
    for row in shortlist:
        final_bundles[row["name"]] = str(row["bundle_path"])

    final_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for bundle_name, bundle_path_text in final_bundles.items():
        bundle_results: Dict[str, Dict[str, Any]] = {}
        for label, start, end in FULL_WINDOWS:
            bundle_results[label] = _backtest_bundle(
                bundle_path=_resolve(bundle_path_text),
                bundle_name=bundle_name,
                label=label,
                start=start,
                end=end,
                output_dir=eval_dir / "final",
                logs_dir=logs_dir,
                force=not args.skip_existing,
            )["metrics"]
        final_results[bundle_name] = bundle_results

    summary_payload = {
        "artifact_root": str(artifact_root),
        "exports": exports,
        "trainer_result": trainer_result,
        "candidate_summary": candidate_summary,
        "candidate_scores": candidate_scores,
        "shortlist": shortlist,
        "final_results": final_results,
    }
    summary_json = artifact_root / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    summary_md = artifact_root / "summary.md"
    winner_name = shortlist[0]["name"] if shortlist else "baseline"
    _write_summary_md(
        path=summary_md,
        baseline_name="baseline",
        winner_name=winner_name,
        results_by_bundle=final_results,
    )
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()

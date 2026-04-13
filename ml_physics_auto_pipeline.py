import argparse
import json
import logging
import subprocess
import sys
import shlex
from pathlib import Path
from typing import Dict, List, Tuple


def _run(cmd: List[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_report(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing LORO report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_loro(report: Dict, min_trades: int, min_holdouts: int, min_positive_ratio: float) -> Tuple[bool, List[str]]:
    def _safe_float(value):
        try:
            out = float(value)
        except Exception:
            return None
        if out != out:  # NaN guard without importing math
            return None
        return out

    def _is_positive_holdout(metrics: Dict) -> bool:
        wf = metrics.get("walk_forward") if isinstance(metrics, dict) else None
        if isinstance(wf, dict):
            objective = _safe_float(wf.get("objective_score"))
            if objective is not None:
                return objective > 0.0
            mean_ev = _safe_float(wf.get("mean_fold_ev"))
            worst_ev = _safe_float(wf.get("worst_fold_ev"))
            if mean_ev is not None and worst_ev is not None:
                # Match default trainer/session-manager objective weighting.
                return (0.65 * mean_ev + 0.35 * worst_ev) > 0.0
        avg_pnl = _safe_float(metrics.get("avg_pnl")) if isinstance(metrics, dict) else None
        return (avg_pnl or 0.0) > 0.0

    results = report.get("results", [])
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for item in results:
        session = item.get("session", "UNKNOWN")
        split = item.get("split", "ALL")
        metrics = item.get("metrics", {}) or {}
        grouped.setdefault((session, split), []).append(metrics)

    failures = []
    for (session, split), metrics_list in grouped.items():
        valid = []
        for m in metrics_list:
            trades = int(m.get("trade_count", 0) or 0)
            if trades >= min_trades:
                valid.append(m)
        if len(valid) < min_holdouts:
            failures.append(
                f"{session}/{split}: insufficient holdouts ({len(valid)}/{min_holdouts}) with >= {min_trades} trades"
            )
            continue
        positives = sum(1 for m in valid if _is_positive_holdout(m))
        ratio = positives / float(len(valid))
        if ratio < min_positive_ratio:
            failures.append(
                f"{session}/{split}: positive ratio {ratio:.2f} < {min_positive_ratio:.2f} "
                f"({positives}/{len(valid)})"
            )

    passed = len(failures) == 0
    return passed, failures


def _extract_extra_args(argv: List[str], flag: str, stop_flags: set) -> tuple[list[str], list[str]]:
    if flag not in argv:
        return [], argv
    idx = argv.index(flag)
    j = idx + 1
    extra: list[str] = []
    while j < len(argv):
        token = argv[j]
        if token in stop_flags:
            break
        extra.append(token)
        j += 1
    new_argv = argv[:idx] + argv[j:]
    if len(extra) == 1 and " " in extra[0]:
        extra = shlex.split(extra[0])
    return extra, new_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated MLPhysics LORO + training pipeline.")
    parser.add_argument("--manifest", help="Path to JSON manifest of regimes.")
    parser.add_argument("--base-csv", help="Optional base CSV for regime date ranges.")
    parser.add_argument("--regime", action="append", nargs=2, metavar=("NAME", "CSV"))
    parser.add_argument("--regime-range", action="append", nargs=3, metavar=("NAME", "START", "END"))
    parser.add_argument("--out-dir", default=".")
    default_train_csv = "es_master.csv"
    parser.add_argument("--train-csv", default=default_train_csv)
    parser.add_argument("--no-walk-forward", action="store_true")
    parser.add_argument("--no-regime-balance", action="store_true")
    parser.add_argument("--min-trades", type=int, default=60)
    parser.add_argument("--min-holdouts", type=int, default=2)
    parser.add_argument("--min-positive-ratio", type=float, default=0.60)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--loro-extra", nargs="*")
    parser.add_argument("--train-extra", nargs="*")
    stop_flags = {
        "--manifest",
        "--base-csv",
        "--regime",
        "--regime-range",
        "--out-dir",
        "--train-csv",
        "--no-walk-forward",
        "--no-regime-balance",
        "--min-trades",
        "--min-holdouts",
        "--min-positive-ratio",
        "--force-train",
        "--loro-extra",
        "--train-extra",
    }
    argv = sys.argv[1:]
    loro_extra, argv = _extract_extra_args(argv, "--loro-extra", stop_flags)
    train_extra, argv = _extract_extra_args(argv, "--train-extra", stop_flags)
    args = parser.parse_args(argv)
    if not loro_extra and args.loro_extra:
        loro_extra = list(args.loro_extra)
    if not train_extra and args.train_extra:
        train_extra = list(args.train_extra)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    manifest_source = None
    if args.manifest:
        try:
            payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            manifest_source = str(payload.get("source_csv", "")).strip() or None
        except Exception:
            manifest_source = None

    if manifest_source and not args.base_csv:
        args.base_csv = manifest_source
    train_csv = args.train_csv
    if manifest_source and train_csv == default_train_csv:
        train_csv = manifest_source

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loro_cmd = [sys.executable, "ml_train_physics_regime_loro.py", "--out-dir", str(out_dir)]
    if args.manifest:
        loro_cmd += ["--manifest", args.manifest]
    if args.base_csv:
        loro_cmd += ["--base-csv", args.base_csv]
    if args.regime:
        for name, csv_path in args.regime:
            loro_cmd += ["--regime", name, csv_path]
    if args.regime_range:
        for name, start, end in args.regime_range:
            loro_cmd += ["--regime-range", name, start, end]
    if args.no_regime_balance:
        loro_cmd.append("--no-regime-balance")
    if not args.no_walk_forward:
        loro_cmd.append("--walk-forward")
    if loro_extra:
        loro_cmd += loro_extra

    _run(loro_cmd)

    report_path = out_dir / "ml_physics_loro_report.json"
    report = _load_report(report_path)
    passed, failures = _summarize_loro(
        report,
        min_trades=args.min_trades,
        min_holdouts=args.min_holdouts,
        min_positive_ratio=args.min_positive_ratio,
    )

    if not passed:
        logging.warning("LORO guard failed:")
        for item in failures:
            logging.warning("  - %s", item)
        if not args.force_train:
            logging.error("Skipping training. Re-run with --force-train to ignore guard.")
            return

    train_cmd = [
        sys.executable,
        "ml_train_physics.py",
        "--csv",
        train_csv,
        "--out-dir",
        str(out_dir),
    ]
    if not args.no_walk_forward:
        train_cmd.append("--walk-forward")
    if train_extra:
        train_cmd += train_extra

    _run(train_cmd)
    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()

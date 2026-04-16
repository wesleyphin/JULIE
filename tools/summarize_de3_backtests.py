import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _profit_factor(pnls: list[float]) -> float:
    gross_win = sum(v for v in pnls if v > 0.0)
    gross_loss = -sum(v for v in pnls if v < 0.0)
    if gross_loss <= 0.0:
        return float("inf") if gross_win > 0.0 else 0.0
    return gross_win / gross_loss


def _sqn(pnls: list[float]) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((v - mean) ** 2 for v in pnls) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return (mean / std) * math.sqrt(n)


def _daily_sharpe(trades: list[dict[str, Any]]) -> float:
    daily_pnl: dict[str, float] = defaultdict(float)
    for trade in trades:
        ts_raw = trade.get("exit_time") or trade.get("entry_time")
        pnl = _safe_float(trade.get("pnl_net"), 0.0)
        if not ts_raw:
            continue
        try:
            dt = datetime.fromisoformat(str(ts_raw))
        except ValueError:
            continue
        day_key = dt.date().isoformat()
        daily_pnl[day_key] += pnl
    values = list(daily_pnl.values())
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return (mean / std) * math.sqrt(252.0)


def _window_label(start_iso: str, end_iso: str) -> str:
    start = start_iso[:10] if start_iso else "?"
    end = end_iso[:10] if end_iso else "?"
    return f"{start}_{end}"


def _bundle_label(path: Path, reports_root: Path) -> str:
    try:
        rel = path.parent.relative_to(reports_root)
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return path.parent.name


def _iter_reports(reports_root: Path) -> list[Path]:
    out: list[Path] = []
    for path in reports_root.rglob("backtest_*.json"):
        name_lower = path.name.lower()
        if name_lower.endswith("_monte_carlo.json"):
            continue
        out.append(path)
    return sorted(out)


def _compute_score(row: dict[str, Any]) -> float:
    # OOS-focused score: emphasize downside-adjusted returns + robustness.
    pf = _safe_float(row.get("profit_factor"), 0.0)
    sqn = _safe_float(row.get("sqn"), 0.0)
    sharpe = _safe_float(row.get("daily_sharpe"), 0.0)
    net = _safe_float(row.get("equity"), 0.0)
    mc_p05 = _safe_float(row.get("mc_net_pnl_p05"), 0.0)
    mc_dd95 = _safe_float(row.get("mc_max_dd_p95"), 0.0)
    max_dd = _safe_float(row.get("max_drawdown"), 0.0)
    return (
        net
        + (mc_p05 * 1.5)
        + (pf * 1200.0)
        + (sqn * 400.0)
        + (sharpe * 250.0)
        - (mc_dd95 * 0.9)
        - (max_dd * 0.35)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DE3 backtest reports with risk metrics.")
    parser.add_argument(
        "--reports-root",
        default="artifacts/de3_outrights_rebuild_20260319_024643/backtest_reports",
        help="Directory containing DE3 backtest report folders.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    reports_root = Path(args.reports_root).expanduser().resolve()
    if not reports_root.is_dir():
        raise SystemExit(f"reports root not found: {reports_root}")

    rows: list[dict[str, Any]] = []
    for report_path in _iter_reports(reports_root):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        trades = payload.get("trade_log", []) if isinstance(payload.get("trade_log"), list) else []
        monte = payload.get("monte_carlo", {}) if isinstance(payload.get("monte_carlo"), dict) else {}
        pnls = [_safe_float(t.get("pnl_net"), 0.0) for t in trades if isinstance(t, dict)]

        row = {
            "bundle": _bundle_label(report_path, reports_root),
            "window": _window_label(str(payload.get("range_start", "")), str(payload.get("range_end", ""))),
            "report_path": str(report_path),
            "equity": _safe_float(summary.get("equity"), 0.0),
            "trades": int(_safe_float(summary.get("trades"), 0.0)),
            "winrate": _safe_float(summary.get("winrate"), 0.0),
            "max_drawdown": _safe_float(summary.get("max_drawdown"), 0.0),
            "profit_factor": _profit_factor(pnls),
            "sqn": _sqn(pnls),
            "daily_sharpe": _daily_sharpe(trades),
            "mc_net_pnl_p05": _safe_float(monte.get("net_pnl_p05"), 0.0),
            "mc_net_pnl_p95": _safe_float(monte.get("net_pnl_p95"), 0.0),
            "mc_max_dd_p95": _safe_float(monte.get("max_drawdown_p95"), 0.0),
            "mc_max_dd_p99": _safe_float(monte.get("max_drawdown_p99"), 0.0),
        }
        row["score"] = _compute_score(row)
        rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)

    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "bundle",
            "window",
            "equity",
            "trades",
            "winrate",
            "profit_factor",
            "sqn",
            "daily_sharpe",
            "max_drawdown",
            "mc_net_pnl_p05",
            "mc_net_pnl_p95",
            "mc_max_dd_p95",
            "mc_max_dd_p99",
            "score",
            "report_path",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"csv={out_csv}")

    header = (
        f"{'bundle':30} {'window':23} {'eq':>10} {'trd':>6} "
        f"{'pf':>6} {'sqn':>6} {'shrp':>6} {'mdd':>10} {'mc_p05':>10} {'mc_dd95':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['bundle'][:30]:30} {row['window'][:23]:23} "
            f"{row['equity']:10.2f} {row['trades']:6d} "
            f"{row['profit_factor']:6.2f} {row['sqn']:6.2f} {row['daily_sharpe']:6.2f} "
            f"{row['max_drawdown']:10.2f} {row['mc_net_pnl_p05']:10.2f} {row['mc_max_dd_p95']:10.2f}"
        )


if __name__ == "__main__":
    main()

import argparse
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform.startswith("win"):
    import os

    _platform_machine = str(os.environ.get("PROCESSOR_ARCHITECTURE", "") or "").strip()
    if _platform_machine:
        platform.machine = lambda: _platform_machine  # type: ignore[assignment]

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _trade_pnl(trade: dict) -> float:
    pnl = _safe_float(trade.get("pnl_net"), float("nan"))
    if not math.isfinite(pnl):
        pnl = _safe_float(trade.get("pnl_dollars"), float("nan"))
    if not math.isfinite(pnl):
        pnl = _safe_float(trade.get("pnl"), 0.0)
    return float(pnl)


def _profit_factor(pnls: list[float]) -> float:
    gains = float(sum(value for value in pnls if value > 0.0))
    losses = float(-sum(value for value in pnls if value < 0.0))
    if losses <= 1e-12:
        return 999.0 if gains > 0.0 else 0.0
    return float(gains / losses)


def _daily_sharpe_sortino(day_pnls: np.ndarray) -> tuple[float, float]:
    if day_pnls.size <= 1:
        return 0.0, 0.0
    mean = float(np.mean(day_pnls))
    std = float(np.std(day_pnls, ddof=1))
    sharpe = (mean / std) * math.sqrt(252.0) if std > 1e-12 else 0.0
    downside = np.minimum(day_pnls, 0.0)
    downside_rms = float(math.sqrt(float(np.mean(np.square(downside)))))
    sortino = (mean / downside_rms) * math.sqrt(252.0) if downside_rms > 1e-12 else 0.0
    return float(sharpe), float(sortino)


def _max_drawdown_from_day_pnls(day_pnls: np.ndarray) -> float:
    if day_pnls.size == 0:
        return 0.0
    equity_curve = np.cumsum(day_pnls)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity_curve)))
    drawdowns = running_peak[1:] - equity_curve
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def _bootstrap_metric_summary(trade_log: list[dict], simulations: int, seed: int) -> dict:
    day_trade_pnls: dict[str, list[float]] = defaultdict(list)
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        day_key = bt._trade_timestamp_to_ny_day(trade.get("exit_time") or trade.get("entry_time"))
        if not day_key:
            continue
        day_trade_pnls[day_key].append(_trade_pnl(trade))

    if not day_trade_pnls:
        return {
            "enabled": True,
            "status": "empty",
            "simulations": int(simulations),
            "trade_days": 0,
        }

    rng = np.random.default_rng(int(seed))
    ordered_days = sorted(day_trade_pnls.keys())
    day_trade_lists = [day_trade_pnls[key] for key in ordered_days]
    day_nets = np.asarray([float(sum(pnls)) for pnls in day_trade_lists], dtype=float)
    day_count = int(len(day_trade_lists))

    net_pnls = np.empty(int(simulations), dtype=float)
    profit_factors = np.empty(int(simulations), dtype=float)
    sharpes = np.empty(int(simulations), dtype=float)
    sortinos = np.empty(int(simulations), dtype=float)
    max_drawdowns = np.empty(int(simulations), dtype=float)

    for idx in range(int(simulations)):
        sampled_idx = rng.integers(0, day_count, size=day_count, endpoint=False)
        sampled_day_nets = day_nets[sampled_idx]
        sampled_trade_pnls: list[float] = []
        for item in sampled_idx.tolist():
            sampled_trade_pnls.extend(day_trade_lists[int(item)])
        net_pnls[idx] = float(np.sum(sampled_day_nets))
        profit_factors[idx] = float(_profit_factor(sampled_trade_pnls))
        sharpe, sortino = _daily_sharpe_sortino(sampled_day_nets)
        sharpes[idx] = float(sharpe)
        sortinos[idx] = float(sortino)
        max_drawdowns[idx] = float(_max_drawdown_from_day_pnls(sampled_day_nets))

    def _metric(values: np.ndarray, *, threshold: float | None = None) -> dict:
        out = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
        if threshold is not None:
            out["probability_above_threshold"] = float(np.mean(values > float(threshold)))
        return out

    return {
        "enabled": True,
        "status": "ok",
        "simulations": int(simulations),
        "seed": int(seed),
        "trade_days": int(day_count),
        "net_pnl": _metric(net_pnls, threshold=0.0),
        "profit_factor": _metric(profit_factors, threshold=1.0),
        "daily_sharpe": _metric(sharpes, threshold=1.0),
        "daily_sortino": _metric(sortinos, threshold=1.0),
        "max_drawdown": _metric(max_drawdowns),
    }


def _count_by_field(trade_log: list[dict], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        key = str(trade.get(field) or "UNKNOWN").strip().upper()
        counts[key] += 1
    return dict(sorted(((key, int(value)) for key, value in counts.items()), key=lambda kv: (-kv[1], kv[0])))


def _sum_by_field(trade_log: list[dict], field: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        key = str(trade.get(field) or "UNKNOWN").strip().upper()
        totals[key] += _trade_pnl(trade)
    return dict(sorted(((key, round(float(value), 2)) for key, value in totals.items()), key=lambda kv: (-kv[1], kv[0])))


def _evaluate_report(report_path: Path, simulations: int, seed: int) -> dict:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    result = payload.get("result", {}) if isinstance(payload.get("result", {}), dict) else {}
    trade_log = payload.get("trade_log", [])
    if not isinstance(trade_log, list):
        trade_log = []

    risk = bt._compute_backtest_risk_metrics(trade_log)
    summary = {
        "equity": _safe_float(result.get("equity"), 0.0),
        "trades": int(result.get("trades", len(trade_log)) or len(trade_log)),
        "wins": int(result.get("wins", 0) or 0),
        "losses": int(result.get("losses", 0) or 0),
        "winrate": _safe_float(result.get("winrate"), 0.0),
        "max_drawdown": _safe_float(result.get("max_drawdown"), 0.0),
        "gross_profit": _safe_float(result.get("gross_profit"), risk.get("gross_profit", 0.0)),
        "gross_loss": _safe_float(result.get("gross_loss"), risk.get("gross_loss", 0.0)),
        "profit_factor": _safe_float(result.get("profit_factor"), risk.get("profit_factor", 0.0)),
        "avg_trade_net": _safe_float(result.get("avg_trade_net"), risk.get("avg_trade_net", 0.0)),
        "trade_sqn": _safe_float(result.get("trade_sqn"), risk.get("trade_sqn", 0.0)),
        "daily_sharpe": _safe_float(result.get("daily_sharpe"), risk.get("daily_sharpe", 0.0)),
        "daily_sortino": _safe_float(result.get("daily_sortino"), risk.get("daily_sortino", 0.0)),
    }
    session_counts = _count_by_field(trade_log, "session")
    pnl_by_session = _sum_by_field(trade_log, "session")
    mc_trade_order = bt._build_monte_carlo_summary(
        trade_log,
        result,
        simulations=max(1, int(simulations)),
        seed=int(seed),
        starting_balance=float(bt.BACKTEST_MONTE_CARLO_START_BALANCE),
    )
    mc_day_bootstrap = _bootstrap_metric_summary(
        trade_log,
        simulations=max(1, int(simulations)),
        seed=int(seed),
    )
    return {
        "report_path": str(report_path),
        "artifact_path": str(payload.get("artifact_path") or ""),
        "range_start": payload.get("range_start"),
        "range_end": payload.get("range_end"),
        "summary": summary,
        "session_counts": session_counts,
        "pnl_by_session": pnl_by_session,
        "side_counts": _count_by_field(trade_log, "side"),
        "ny_trade_count": int(session_counts.get("NY_AM", 0) + session_counts.get("NY_PM", 0)),
        "ny_pnl": round(float(pnl_by_session.get("NY_AM", 0.0) + pnl_by_session.get("NY_PM", 0.0)), 2),
        "gate_threshold": payload.get("summary", {}).get("gate_threshold")
        if isinstance(payload.get("summary", {}), dict)
        else None,
        "gate_session_thresholds": payload.get("summary", {}).get("gate_session_thresholds")
        if isinstance(payload.get("summary", {}), dict)
        else {},
        "monte_carlo_trade_order": mc_trade_order,
        "monte_carlo_trade_day_bootstrap": mc_day_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RegimeAdaptive backtest reports with Monte Carlo summaries.")
    parser.add_argument("--report", action="append", required=True, help="Repeatable report JSON path.")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    evaluations = [
        _evaluate_report(Path(report_text).expanduser().resolve(), int(args.simulations), int(args.seed))
        for report_text in args.report
    ]
    payload = {"reports": evaluations}
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    if str(args.output or "").strip():
        out_path = Path(str(args.output)).expanduser()
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")
EPSILON = 1e-9


@dataclass
class PropRuleConfig:
    account_size: float = 50_000.0
    max_daily_loss: float = 1_000.0
    max_total_loss: float = 2_000.0
    profit_target_eval: float = 3_000.0
    max_daily_profit_eval: float = 1_500.0
    express_payout_threshold: float = 150.0
    express_payouts_required: int = 5
    express_days_for_payout: int = 5
    live_resets_to_account_size: bool = True


@dataclass
class TradeRow:
    replay_label: str
    source_report: str
    trade_id: str
    strategy: str
    sub_strategy: str
    entry_time: datetime
    exit_time: datetime
    pnl_net: float
    pnl_dollars: float
    pnl_points: float
    size: float
    fee_paid: float
    mae_points: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an exact-path prop rule simulation on exported trade CSVs."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input trade CSV. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest_reports") / "prop_rule_sim_current_version",
        help="Directory for JSON and CSV outputs.",
    )
    parser.add_argument(
        "--account-size",
        type=float,
        default=50_000.0,
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=1_000.0,
    )
    parser.add_argument(
        "--max-total-loss",
        type=float,
        default=2_000.0,
    )
    parser.add_argument(
        "--profit-target-eval",
        type=float,
        default=3_000.0,
    )
    parser.add_argument(
        "--max-daily-profit-eval",
        type=float,
        default=1_500.0,
    )
    parser.add_argument(
        "--express-payout-threshold",
        type=float,
        default=150.0,
    )
    parser.add_argument(
        "--express-payouts-required",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--express-days-for-payout",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mode",
        choices=["realized_only", "mae_stress", "both"],
        default="both",
        help="Daily loss enforcement mode.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text.replace("$", "").replace(",", ""))


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def to_ny_day(dt: datetime) -> date:
    return dt.astimezone(NY_TZ).date()


def safe_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def load_trades(csv_path: Path) -> tuple[list[TradeRow], float]:
    trades: list[TradeRow] = []
    point_value_samples: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entry_raw = row.get("entry_time", "")
            exit_raw = row.get("exit_time", "") or entry_raw
            pnl_net = parse_float(row.get("pnl_net"))
            if math.isnan(pnl_net) or not entry_raw:
                continue

            pnl_dollars = parse_float(row.get("pnl_dollars"))
            pnl_points = parse_float(row.get("pnl_points"))
            size = parse_float(row.get("size"))
            fee_paid = parse_float(row.get("fee_paid"))
            mae_points = parse_float(row.get("mae_points"))

            trade = TradeRow(
                replay_label=row.get("replay_label", ""),
                source_report=row.get("source_report", ""),
                trade_id=str(row.get("trade_id", "") or row.get("id", "")),
                strategy=row.get("strategy", ""),
                sub_strategy=row.get("sub_strategy", ""),
                entry_time=parse_datetime(entry_raw),
                exit_time=parse_datetime(exit_raw),
                pnl_net=pnl_net,
                pnl_dollars=0.0 if math.isnan(pnl_dollars) else pnl_dollars,
                pnl_points=0.0 if math.isnan(pnl_points) else pnl_points,
                size=0.0 if math.isnan(size) else size,
                fee_paid=0.0 if math.isnan(fee_paid) else fee_paid,
                mae_points=None if math.isnan(mae_points) else mae_points,
            )
            trades.append(trade)

            if abs(trade.pnl_points) > EPSILON and abs(trade.size) > EPSILON:
                ratio = abs(trade.pnl_dollars) / (abs(trade.pnl_points) * abs(trade.size))
                if ratio > EPSILON:
                    point_value_samples.append(ratio)

    trades.sort(
        key=lambda trade: (
            trade.entry_time,
            trade.exit_time,
            trade.trade_id,
        )
    )

    inferred_point_value = statistics.median(point_value_samples) if point_value_samples else 5.0
    return trades, inferred_point_value


def estimate_trade_mae_loss(trade: TradeRow, point_value: float) -> float:
    mae_loss = 0.0
    if trade.mae_points is not None and abs(trade.size) > EPSILON:
        mae_loss = abs(trade.mae_points) * abs(trade.size) * point_value
        if trade.fee_paid > 0:
            mae_loss += trade.fee_paid
    realized_loss = max(0.0, -trade.pnl_net)
    return max(mae_loss, realized_loss)


def simulate_exact_path(
    trades: list[TradeRow],
    config: PropRuleConfig,
    mode: str,
    point_value: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not trades:
        raise ValueError("No trades available for simulation.")

    balance = config.account_size
    loss_limit_level = config.account_size - config.max_total_loss
    phase = "EVAL"
    current_day: date | None = None
    day_open_balance = balance
    day_phase_pnl = 0.0
    day_phase_trade_count = 0
    day_last_trade_time: datetime | None = None

    eval_progress = 0.0
    express_qualifying_days = 0
    express_payouts = 0
    eval_attempt_id = 1

    max_realized_daily_loss = 0.0
    max_intraday_daily_loss_proxy = 0.0
    lowest_balance_realized = balance
    lowest_balance_intraday_proxy = balance

    eval_passes = 0
    eval_failures_daily_loss = 0
    eval_failures_total_loss = 0
    express_entries = 0
    express_failures_daily_loss = 0
    express_failures_total_loss = 0
    express_payouts_total = 0
    live_entries = 0
    live_failures_daily_loss = 0
    live_failures_total_loss = 0
    live_days = 0
    live_trades = 0
    resets = 0

    first_eval_pass_time: str | None = None
    first_payout_time: str | None = None
    first_live_time: str | None = None

    events: list[dict[str, Any]] = []

    def record_event(event_type: str, event_time: datetime, **extra: Any) -> None:
        events.append(
            {
                "event_type": event_type,
                "event_time": event_time.isoformat(),
                "phase": phase,
                "eval_attempt_id": eval_attempt_id,
                **extra,
            }
        )

    def reset_to_eval(event_time: datetime, reason: str, trigger: str, trade: TradeRow, detail: dict[str, Any]) -> None:
        nonlocal balance
        nonlocal phase
        nonlocal day_open_balance
        nonlocal day_phase_pnl
        nonlocal day_phase_trade_count
        nonlocal eval_progress
        nonlocal express_qualifying_days
        nonlocal express_payouts
        nonlocal eval_attempt_id
        nonlocal resets
        nonlocal eval_failures_daily_loss
        nonlocal eval_failures_total_loss
        nonlocal express_failures_daily_loss
        nonlocal express_failures_total_loss
        nonlocal live_failures_daily_loss
        nonlocal live_failures_total_loss

        prior_phase = phase

        if prior_phase == "EVAL":
            if reason == "daily_loss":
                eval_failures_daily_loss += 1
            else:
                eval_failures_total_loss += 1
        elif prior_phase == "EXPRESS":
            if reason == "daily_loss":
                express_failures_daily_loss += 1
            else:
                express_failures_total_loss += 1
        else:
            if reason == "daily_loss":
                live_failures_daily_loss += 1
            else:
                live_failures_total_loss += 1

        record_event(
            "blow",
            event_time,
            prior_phase=prior_phase,
            reason=reason,
            trigger=trigger,
            trade_id=trade.trade_id,
            replay_label=trade.replay_label,
            balance_before_reset=round(balance, 2),
            **detail,
        )

        resets += 1
        balance = config.account_size
        phase = "EVAL"
        eval_progress = 0.0
        express_qualifying_days = 0
        express_payouts = 0
        eval_attempt_id += 1
        day_open_balance = balance
        day_phase_pnl = 0.0
        day_phase_trade_count = 0

        record_event(
            "eval_attempt_start",
            event_time,
            reason="reset_after_blow",
            balance=round(balance, 2),
        )

    def finalize_day() -> None:
        nonlocal balance
        nonlocal phase
        nonlocal day_phase_pnl
        nonlocal day_phase_trade_count
        nonlocal day_last_trade_time
        nonlocal eval_progress
        nonlocal express_qualifying_days
        nonlocal express_payouts
        nonlocal eval_passes
        nonlocal express_entries
        nonlocal express_payouts_total
        nonlocal live_entries
        nonlocal live_days
        nonlocal first_eval_pass_time
        nonlocal first_payout_time
        nonlocal first_live_time

        if day_phase_trade_count <= 0 or day_last_trade_time is None:
            return

        if phase == "EVAL":
            daily_contribution = day_phase_pnl
            if daily_contribution > config.max_daily_profit_eval:
                daily_contribution = config.max_daily_profit_eval

            eval_progress += daily_contribution
            record_event(
                "eval_day_close",
                day_last_trade_time,
                day_pnl=round(day_phase_pnl, 2),
                day_contribution=round(daily_contribution, 2),
                eval_progress=round(eval_progress, 2),
            )

            if eval_progress >= config.profit_target_eval - EPSILON:
                eval_passes += 1
                express_entries += 1
                record_event(
                    "eval_pass",
                    day_last_trade_time,
                    eval_progress=round(eval_progress, 2),
                )
                if first_eval_pass_time is None:
                    first_eval_pass_time = day_last_trade_time.isoformat()

                phase = "EXPRESS"
                balance = config.account_size
                eval_progress = 0.0
                express_qualifying_days = 0
                express_payouts = 0
                record_event(
                    "express_start",
                    day_last_trade_time,
                    balance=round(balance, 2),
                )

        elif phase == "EXPRESS":
            qualifies = day_phase_pnl >= config.express_payout_threshold - EPSILON
            if qualifies:
                express_qualifying_days += 1

            record_event(
                "express_day_close",
                day_last_trade_time,
                day_pnl=round(day_phase_pnl, 2),
                qualifying_day=qualifies,
                qualifying_day_count=express_qualifying_days,
                express_payouts=express_payouts,
            )

            if express_qualifying_days >= config.express_days_for_payout:
                express_payouts += 1
                express_payouts_total += 1
                express_qualifying_days = 0
                record_event(
                    "express_payout",
                    day_last_trade_time,
                    payout_number=express_payouts,
                )
                if first_payout_time is None:
                    first_payout_time = day_last_trade_time.isoformat()

                if express_payouts >= config.express_payouts_required:
                    live_entries += 1
                    record_event(
                        "live_reached",
                        day_last_trade_time,
                        payouts_completed=express_payouts,
                    )
                    if first_live_time is None:
                        first_live_time = day_last_trade_time.isoformat()

                    phase = "LIVE"
                    if config.live_resets_to_account_size:
                        balance = config.account_size
                        record_event(
                            "live_start",
                            day_last_trade_time,
                            balance=round(balance, 2),
                            reset_to_account_size=True,
                        )
                    else:
                        record_event(
                            "live_start",
                            day_last_trade_time,
                            balance=round(balance, 2),
                            reset_to_account_size=False,
                        )

        elif phase == "LIVE":
            live_days += 1
            record_event(
                "live_day_close",
                day_last_trade_time,
                day_pnl=round(day_phase_pnl, 2),
                balance=round(balance, 2),
            )

    record_event(
        "eval_attempt_start",
        trades[0].entry_time,
        reason="initial",
        balance=round(balance, 2),
    )

    for trade in trades:
        trade_day = to_ny_day(trade.entry_time)
        if current_day is None:
            current_day = trade_day
            day_open_balance = balance
        elif trade_day != current_day:
            finalize_day()
            current_day = trade_day
            day_open_balance = balance
            day_phase_pnl = 0.0
            day_phase_trade_count = 0
            day_last_trade_time = None

        if mode == "mae_stress":
            intraday_loss = estimate_trade_mae_loss(trade, point_value)
            intraday_floor = balance - intraday_loss
            daily_loss_proxy = day_open_balance - intraday_floor
            max_intraday_daily_loss_proxy = max(max_intraday_daily_loss_proxy, daily_loss_proxy)
            lowest_balance_intraday_proxy = min(lowest_balance_intraday_proxy, intraday_floor)

            if daily_loss_proxy >= config.max_daily_loss - EPSILON:
                reset_to_eval(
                    trade.entry_time,
                    reason="daily_loss",
                    trigger="intraday_mae_proxy",
                    trade=trade,
                    detail={
                        "intraday_floor": round(intraday_floor, 2),
                        "daily_loss_proxy": round(daily_loss_proxy, 2),
                    },
                )
                continue

            if intraday_floor < loss_limit_level - EPSILON:
                reset_to_eval(
                    trade.entry_time,
                    reason="max_total_loss",
                    trigger="intraday_mae_proxy",
                    trade=trade,
                    detail={
                        "intraday_floor": round(intraday_floor, 2),
                        "loss_limit_level": round(loss_limit_level, 2),
                    },
                )
                continue

        balance += trade.pnl_net
        day_phase_pnl += trade.pnl_net
        day_phase_trade_count += 1
        day_last_trade_time = trade.exit_time

        if phase == "LIVE":
            live_trades += 1

        lowest_balance_realized = min(lowest_balance_realized, balance)
        realized_daily_loss = day_open_balance - balance
        max_realized_daily_loss = max(max_realized_daily_loss, realized_daily_loss)

        if mode == "realized_only":
            max_intraday_daily_loss_proxy = max(max_intraday_daily_loss_proxy, realized_daily_loss)
            lowest_balance_intraday_proxy = min(lowest_balance_intraday_proxy, balance)

        if realized_daily_loss >= config.max_daily_loss - EPSILON:
            reset_to_eval(
                trade.exit_time,
                reason="daily_loss",
                trigger="realized_close",
                trade=trade,
                detail={
                    "ending_balance": round(balance, 2),
                    "daily_loss_realized": round(realized_daily_loss, 2),
                },
            )
            continue

        if balance < loss_limit_level - EPSILON:
            reset_to_eval(
                trade.exit_time,
                reason="max_total_loss",
                trigger="realized_close",
                trade=trade,
                detail={
                    "ending_balance": round(balance, 2),
                    "loss_limit_level": round(loss_limit_level, 2),
                },
            )
            continue

    finalize_day()

    summary = {
        "mode": mode,
        "trade_count": len(trades),
        "date_range": {
            "start": trades[0].entry_time.isoformat(),
            "end": trades[-1].exit_time.isoformat(),
        },
        "replay_label_counts": count_by_replay_label(trades),
        "inferred_point_value": round(point_value, 6),
        "config": asdict(config),
        "results": {
            "eval_attempts_started": eval_attempt_id,
            "eval_passes": eval_passes,
            "eval_failures_daily_loss": eval_failures_daily_loss,
            "eval_failures_total_loss": eval_failures_total_loss,
            "express_entries": express_entries,
            "express_failures_daily_loss": express_failures_daily_loss,
            "express_failures_total_loss": express_failures_total_loss,
            "express_payouts_total": express_payouts_total,
            "live_entries": live_entries,
            "live_failures_daily_loss": live_failures_daily_loss,
            "live_failures_total_loss": live_failures_total_loss,
            "live_days": live_days,
            "live_trades": live_trades,
            "resets": resets,
            "final_phase": phase,
            "final_balance": round(balance, 2),
            "max_realized_daily_loss": round(max_realized_daily_loss, 2),
            "max_intraday_daily_loss_proxy": round(max_intraday_daily_loss_proxy, 2),
            "lowest_balance_realized": round(lowest_balance_realized, 2),
            "lowest_balance_intraday_proxy": round(lowest_balance_intraday_proxy, 2),
            "first_eval_pass_time": first_eval_pass_time,
            "first_payout_time": first_payout_time,
            "first_live_time": first_live_time,
            "ever_reached_live": live_entries > 0,
        },
    }
    return summary, events


def count_by_replay_label(trades: list[TradeRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        counts[trade.replay_label] = counts.get(trade.replay_label, 0) + 1
    return counts


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_events_csv(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for event in events:
        for key in event.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def main() -> None:
    args = parse_args()
    config = PropRuleConfig(
        account_size=args.account_size,
        max_daily_loss=args.max_daily_loss,
        max_total_loss=args.max_total_loss,
        profit_target_eval=args.profit_target_eval,
        max_daily_profit_eval=args.max_daily_profit_eval,
        express_payout_threshold=args.express_payout_threshold,
        express_payouts_required=args.express_payouts_required,
        express_days_for_payout=args.express_days_for_payout,
    )

    modes = ["realized_only", "mae_stress"] if args.mode == "both" else [args.mode]
    timestamp = datetime.now(tz=NY_TZ).strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    overview: dict[str, Any] = {
        "created_at": datetime.now(tz=NY_TZ).isoformat(),
        "config": asdict(config),
        "runs": [],
    }

    for input_path_raw in args.inputs:
        input_path = Path(input_path_raw)
        trades, point_value = load_trades(input_path)
        file_slug = safe_slug(input_path.stem)

        for mode in modes:
            summary, events = simulate_exact_path(trades, config, mode, point_value)
            result_slug = f"{file_slug}_{mode}"
            summary_path = run_root / f"{result_slug}.json"
            events_path = run_root / f"{result_slug}_events.csv"

            write_json(summary_path, summary)
            write_events_csv(events_path, events)

            overview["runs"].append(
                {
                    "input_csv": str(input_path),
                    "mode": mode,
                    "summary_path": str(summary_path),
                    "events_path": str(events_path),
                    "results": summary["results"],
                }
            )

    write_json(run_root / "overview.json", overview)
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()

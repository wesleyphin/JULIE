#!/usr/bin/env python3
"""
Pull recent ProjectX / Topstep bars and replay them through the production
backtest engine as a dry-run sanity check.

This is intentionally a replay probe, not a broker-paper-trading loop:
- it uses live ProjectX historical bars
- it reuses the production backtest engine and current config
- it never places orders

Limitations are reported explicitly in the generated manifest so operators do
not confuse this with a full historical validation of live-only features.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_mes_et import (
    BACKTEST_SELECTABLE_STRATEGIES,
    NY_TZ,
    parse_user_datetime,
    run_backtest,
    save_backtest_report,
)
from backtest_symbol_context import attach_backtest_symbol_context
from bot_state import load_bot_state
from client import ProjectXClient
from config import CONFIG, determine_current_contract_symbol, refresh_target_symbol


DEFAULT_FILTERLESS_STRATEGIES = (
    "DynamicEngine3Strategy",
    "RegimeAdaptiveStrategy",
    "MLPhysicsStrategy",
    "AetherFlowStrategy",
)

STRATEGY_ALIASES = {
    "dynamicengine3": "DynamicEngine3Strategy",
    "dynamicengine3strategy": "DynamicEngine3Strategy",
    "regimeadaptive": "RegimeAdaptiveStrategy",
    "regimeadaptivestrategy": "RegimeAdaptiveStrategy",
    "mlphysics": "MLPhysicsStrategy",
    "mlphysicsstrategy": "MLPhysicsStrategy",
    "aetherflow": "AetherFlowStrategy",
    "aetherflowstrategy": "AetherFlowStrategy",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dry-run replay using fresh Topstep / ProjectX historical bars "
            "and the production backtest engine."
        )
    )
    parser.add_argument(
        "--contract-root",
        default=str(CONFIG.get("CONTRACT_ROOT", "MES") or "MES"),
        help="Primary contract root to replay (default: current live contract root).",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=20_000,
        help="How many recent 1-minute bars to pull from ProjectX (default: 20000).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help="Replay start in ET, e.g. '2026-04-10 09:30'. Default: first pulled bar.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="",
        help="Replay end in ET, e.g. '2026-04-17 16:00'. Default: last pulled bar.",
    )
    parser.add_argument(
        "--account-id",
        type=int,
        default=None,
        help="ProjectX account id. Default: config/env, then bot_state.json, then auto-discovery.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="backtest_reports/topstep_replay",
        help="Output directory for replay reports.",
    )
    parser.add_argument(
        "--bars-out",
        type=str,
        default="",
        help="Optional path to save pulled bars (.csv or .parquet).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_FILTERLESS_STRATEGIES),
        help=(
            "Comma-separated strategies to enable. "
            "Default matches the current filterless live roster."
        ),
    )
    parser.add_argument(
        "--with-filters",
        action="store_true",
        help="Use the backtest filter stack. Default is filterless replay (no external filters).",
    )
    parser.add_argument(
        "--with-mnq",
        action="store_true",
        help="Also pull MNQ bars and pass them into the replay for cross-market features.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser.parse_args()


def _discover_account_from_state() -> Optional[int]:
    state = load_bot_state(ROOT / "bot_state.json")
    if not isinstance(state, dict):
        return None
    live_drawdown = state.get("live_drawdown")
    if isinstance(live_drawdown, dict):
        value = live_drawdown.get("account_id")
        try:
            return int(value)
        except Exception:
            return None
    return None


def _search_active_accounts(client: ProjectXClient) -> list[dict]:
    url = f"{client.base_url}/api/Account/search"
    payload = {"onlyActiveAccounts": True}
    resp = client.session.post(url, json=payload)
    client._track_general_request()
    resp.raise_for_status()
    data = resp.json()
    accounts = data.get("accounts", [])
    if not isinstance(accounts, list):
        return []
    return [row for row in accounts if isinstance(row, dict)]


def _resolve_account_id(client: ProjectXClient, requested_account_id: Optional[int]) -> int:
    if requested_account_id is not None:
        client.account_id = int(requested_account_id)
        return int(client.account_id)

    cfg_account = CONFIG.get("ACCOUNT_ID")
    if cfg_account not in (None, ""):
        client.account_id = int(cfg_account)
        return int(client.account_id)

    state_account = _discover_account_from_state()
    if state_account is not None:
        client.account_id = int(state_account)
        return int(client.account_id)

    accounts = _search_active_accounts(client)
    if len(accounts) == 1:
        account_id = int(accounts[0].get("id"))
        client.account_id = account_id
        return account_id

    raise RuntimeError(
        "Could not resolve a single active ProjectX account. "
        "Pass --account-id or set JULIE_ACCOUNT_ID."
    )


def _strategy_selection(raw_value: str) -> Optional[set[str]]:
    values = set()
    for token in str(raw_value or "").split(","):
        clean = token.strip()
        if not clean:
            continue
        canonical = STRATEGY_ALIASES.get(clean.lower(), clean)
        values.add(canonical)
    unknown = sorted(name for name in values if name not in BACKTEST_SELECTABLE_STRATEGIES)
    if unknown:
        logging.warning(
            "Unknown replay strategies ignored: %s | valid options include: %s",
            ", ".join(unknown),
            ", ".join(BACKTEST_SELECTABLE_STRATEGIES),
        )
    values = {name for name in values if name in BACKTEST_SELECTABLE_STRATEGIES}
    return values or None


def _normalize_time_range(
    df,
    start_raw: str,
    end_raw: str,
) -> tuple[dt.datetime, dt.datetime]:
    if df.empty:
        raise ValueError("Pulled bar set is empty.")
    start_time = parse_user_datetime(start_raw, NY_TZ, is_end=False) if start_raw else df.index.min()
    end_time = parse_user_datetime(end_raw, NY_TZ, is_end=True) if end_raw else df.index.max()
    if start_time > end_time:
        raise ValueError("Start must be before end.")
    return start_time, end_time


def _prepare_symbol_df(df, symbol_label: str):
    work = df.copy()
    if "symbol" in work.columns:
        work = work.drop(columns=["symbol"], errors="ignore")
    attrs = getattr(df, "attrs", {}) or {}
    return attach_backtest_symbol_context(
        work,
        symbol_label,
        "single",
        source_key=attrs.get("source_cache_key"),
        source_label=attrs.get("source_label"),
        source_path=attrs.get("source_path"),
    )


def _save_bars_snapshot(df, path_value: str) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path)
        except Exception as exc:
            fallback = path.with_suffix(".csv")
            logging.warning("Parquet export failed (%s); falling back to CSV at %s", exc, fallback)
            df.to_csv(fallback)
            return fallback
        return path
    df.to_csv(path)
    return path


async def _pull_symbol_df(
    contract_root: str,
    account_id: Optional[int],
    lookback_minutes: int,
) -> tuple[ProjectXClient, object]:
    refresh_target_symbol()
    target_symbol = determine_current_contract_symbol(contract_root)
    client = ProjectXClient(contract_root=contract_root, target_symbol=target_symbol)
    client.login()
    resolved_account = _resolve_account_id(client, account_id)
    client.account_id = int(resolved_account)
    contract_id = client.fetch_contracts()
    if contract_id is None:
        raise RuntimeError(f"Could not resolve an active contract for {contract_root}.")
    df = await client.async_get_market_data(
        lookback_minutes=int(lookback_minutes),
        force_fetch=True,
    )
    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"ProjectX returned no bars for {contract_root}.")
    return client, df


def _write_manifest(
    report_path: Path,
    *,
    primary_client: ProjectXClient,
    primary_df,
    start_time: dt.datetime,
    end_time: dt.datetime,
    selected_strategies: Optional[Iterable[str]],
    with_filters: bool,
    mnq_used: bool,
    bars_snapshot_path: Optional[Path],
) -> Path:
    manifest = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "mode": "topstep_replay_probe",
        "data_source": "ProjectX historical bars",
        "contract_root": primary_client.contract_root,
        "target_symbol": primary_client.target_symbol,
        "contract_id": primary_client.contract_id,
        "account_id": primary_client.account_id,
        "lookback_bar_count": int(len(primary_df)),
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "strategies": sorted(selected_strategies) if selected_strategies is not None else None,
        "with_filters": bool(with_filters),
        "mnq_companion_used": bool(mnq_used),
        "bars_snapshot_path": str(bars_snapshot_path) if bars_snapshot_path is not None else None,
        "report_path": str(report_path),
        "validation_limits": {
            "historical_kalshi_curve_replay": False,
            "live_level_fill_optimizer_replay": False,
            "live_projectx_order_path": False,
            "oanda_mirror_replay": False,
        },
        "notes": [
            "Uses fresh Topstep / ProjectX bars with the production backtest engine.",
            "Good for signal-flow sanity checks and current-regime dry runs.",
            "Not a historical replay of live-only Kalshi ladder behavior or broker order routing.",
        ],
    }
    path = report_path.with_name(f"{report_path.stem}_topstep_manifest.json")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


async def _async_main(args: argparse.Namespace) -> int:
    primary_client, primary_df = await _pull_symbol_df(
        contract_root=str(args.contract_root or "MES").upper(),
        account_id=args.account_id,
        lookback_minutes=args.lookback_minutes,
    )
    logging.info(
        "Pulled %d bars for %s (%s) | %s -> %s",
        len(primary_df),
        primary_client.contract_root,
        primary_client.target_symbol,
        primary_df.index.min(),
        primary_df.index.max(),
    )

    mnq_df = None
    if args.with_mnq and str(primary_client.contract_root).upper() != "MNQ":
        _, mnq_df = await _pull_symbol_df(
            contract_root="MNQ",
            account_id=int(primary_client.account_id),
            lookback_minutes=args.lookback_minutes,
        )
        logging.info(
            "Pulled %d bars for MNQ (%s) | %s -> %s",
            len(mnq_df),
            determine_current_contract_symbol("MNQ"),
            mnq_df.index.min(),
            mnq_df.index.max(),
        )

    start_time, end_time = _normalize_time_range(primary_df, args.start, args.end)
    source_primary_df = primary_df[primary_df.index <= end_time]
    range_primary_df = source_primary_df[
        (source_primary_df.index >= start_time) & (source_primary_df.index <= end_time)
    ]
    if range_primary_df.empty:
        raise RuntimeError("No primary bars remain in the requested replay range.")

    primary_symbol_df = _prepare_symbol_df(source_primary_df, str(primary_client.target_symbol or args.contract_root))
    mnq_symbol_df = None
    if mnq_df is not None and not mnq_df.empty:
        source_mnq_df = mnq_df[mnq_df.index <= end_time]
        if not source_mnq_df.empty:
            mnq_symbol_df = _prepare_symbol_df(
                source_mnq_df,
                determine_current_contract_symbol("MNQ"),
            )

    bars_snapshot_path = _save_bars_snapshot(primary_df, str(args.bars_out or ""))

    selected_strategies = _strategy_selection(args.strategies)
    enabled_filters = None if args.with_filters else set()
    stats = run_backtest(
        primary_symbol_df,
        start_time,
        end_time,
        mnq_df=mnq_symbol_df,
        enabled_strategies=selected_strategies,
        enabled_filters=enabled_filters,
    )
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = save_backtest_report(
        stats,
        str(primary_client.target_symbol or args.contract_root),
        start_time,
        end_time,
        output_dir=report_dir,
    )
    manifest_path = _write_manifest(
        report_path,
        primary_client=primary_client,
        primary_df=primary_df,
        start_time=start_time,
        end_time=end_time,
        selected_strategies=selected_strategies,
        with_filters=bool(args.with_filters),
        mnq_used=mnq_symbol_df is not None,
        bars_snapshot_path=bars_snapshot_path,
    )

    print("")
    print("Topstep Replay Summary")
    print(f"Primary symbol: {primary_client.target_symbol}")
    print(f"Account ID: {primary_client.account_id}")
    print(f"Bars pulled: {len(primary_df)}")
    print(f"Replay range: {start_time} -> {end_time}")
    print(f"Strategies: {', '.join(sorted(selected_strategies)) if selected_strategies else 'ALL'}")
    print(f"Filters: {'enabled' if args.with_filters else 'disabled (filterless replay)'}")
    print(f"Trades: {stats.get('trades')}")
    print(f"Wins: {stats.get('wins')}  Losses: {stats.get('losses')}  Winrate: {float(stats.get('winrate', 0.0)):.2f}%")
    print(f"Net PnL: ${float(stats.get('equity', 0.0)):.2f}")
    print(f"Max drawdown: ${float(stats.get('max_drawdown', 0.0)):.2f}")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    if bars_snapshot_path is not None:
        print(f"Bars snapshot: {bars_snapshot_path}")
    print("")
    print("Limits: no historical Kalshi ladder replay, no live LevelFill replay, no broker order routing.")
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        logging.warning("Replay cancelled by user.")
        return 130
    except Exception as exc:
        logging.error("Topstep replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

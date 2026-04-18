#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import math
import os
import sys
import tempfile
import threading
import traceback
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_mes_et import _resolve_sl_tp_conflict
from bot_state import load_bot_state
from client import ProjectXClient as LiveProjectXClient
from config import CONFIG, determine_current_contract_symbol, refresh_target_symbol

NY_TZ = ZoneInfo("America/New_York")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the actual live JULIE loop against replayed ProjectX bars. "
            "No real orders are sent. OANDA mirror is disabled."
        )
    )
    parser.add_argument("--contract-root", default="MES")
    parser.add_argument("--lookback-minutes", type=int, default=20_000)
    parser.add_argument("--start", required=True, help="Replay start in ET, e.g. '2026-04-16 00:00'")
    parser.add_argument("--end", required=True, help="Replay end in ET, e.g. '2026-04-17 16:59'")
    parser.add_argument("--account-id", type=int, default=None)
    parser.add_argument("--initial-balance", type=float, default=50_000.0)
    parser.add_argument(
        "--report-dir",
        default="backtest_reports/full_live_replay",
        help="Directory for isolated replay outputs.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _parse_user_time(raw: str, is_end: bool) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Replay time is required.")
    value = pd.Timestamp(text)
    if value.tzinfo is None:
        value = value.tz_localize(NY_TZ)
    else:
        value = value.tz_convert(NY_TZ)
    if len(text) <= 10:
        if is_end:
            value = value.replace(hour=23, minute=59, second=0, microsecond=0)
        else:
            value = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return value.to_pydatetime()


def _discover_account_from_state() -> Optional[int]:
    state = load_bot_state(ROOT / "bot_state.json")
    if not isinstance(state, dict):
        return None
    live_drawdown = state.get("live_drawdown")
    if not isinstance(live_drawdown, dict):
        return None
    try:
        return int(live_drawdown.get("account_id"))
    except Exception:
        return None


def _resolve_account_id(client: LiveProjectXClient, requested_account_id: Optional[int]) -> int:
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
    raise RuntimeError("No ProjectX account id available. Pass --account-id or set JULIE_ACCOUNT_ID.")


async def _pull_projectx_bars(
    *,
    contract_root: str,
    lookback_minutes: int,
    account_id: Optional[int],
) -> tuple[LiveProjectXClient, pd.DataFrame]:
    refresh_target_symbol()
    target_symbol = determine_current_contract_symbol(contract_root)
    client = LiveProjectXClient(contract_root=contract_root, target_symbol=target_symbol)
    client.login()
    client.account_id = _resolve_account_id(client, account_id)
    if client.fetch_contracts() is None:
        raise RuntimeError(f"Could not resolve a live contract for {contract_root}.")
    df = await client.async_get_market_data(lookback_minutes=lookback_minutes, force_fetch=True)
    if df is None or df.empty:
        raise RuntimeError(f"ProjectX returned no bars for {contract_root}.")
    return client, df


def _resample_ohlcv(df: pd.DataFrame, minutes: int, lookback_bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rule = f"{int(minutes)}min"
    work = df[["open", "high", "low", "close", "volume"]].copy()
    agg = (
        work.resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    if lookback_bars > 0 and len(agg) > lookback_bars:
        agg = agg.iloc[-lookback_bars:]
    return agg


def _trade_fee_per_contract() -> float:
    risk_cfg = CONFIG.get("RISK") or {}
    fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    topstep_rt = float(
        risk_cfg.get("TOPSTEP_COMMISSION_ROUND_TURN_PER_CONTRACT", 0.50) or 0.50
    )
    return (fees_per_side * 2.0) + topstep_rt


def _compute_pnl_points(side: str, entry_price: float, exit_price: float) -> float:
    return float(exit_price - entry_price) if side == "LONG" else float(entry_price - exit_price)


def _point_value() -> float:
    risk_cfg = CONFIG.get("RISK") or {}
    return float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)


@dataclass
class ReplaySession:
    full_df: pd.DataFrame
    replay_start: dt.datetime
    replay_end: dt.datetime
    initial_balance: float
    warmup_df: pd.DataFrame = field(init=False)
    replay_df: pd.DataFrame = field(init=False)
    current_df: pd.DataFrame = field(init=False)
    pointer: int = field(default=0, init=False)
    next_order_id: int = field(default=500000, init=False)
    open_legs: list[dict] = field(default_factory=list, init=False)
    closed_trades: list[dict] = field(default_factory=list, init=False)
    filled_exit_orders: dict[int, dict] = field(default_factory=dict, init=False)
    balance: float = field(init=False)
    bars_delivered: int = field(default=0, init=False)
    manual_close_count: int = field(default=0, init=False)
    reverse_count: int = field(default=0, init=False)
    stop_update_count: int = field(default=0, init=False)
    passive_exit_counts: Counter = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        work = self.full_df.sort_index().copy()
        self.replay_start = pd.Timestamp(self.replay_start).tz_convert(NY_TZ).to_pydatetime()
        self.replay_end = pd.Timestamp(self.replay_end).tz_convert(NY_TZ).to_pydatetime()
        self.warmup_df = work.loc[work.index < self.replay_start].copy()
        self.replay_df = work.loc[(work.index >= self.replay_start) & (work.index <= self.replay_end)].copy()
        if self.replay_df.empty:
            raise ValueError("Replay range contains no bars.")
        if self.warmup_df.empty:
            # Keep at least one bar of state so startup code has a price anchor.
            self.warmup_df = self.replay_df.iloc[:1].copy()
            self.replay_df = self.replay_df.iloc[1:].copy()
        self.current_df = self.warmup_df.copy()
        self.balance = float(self.initial_balance)

    def _new_id(self) -> int:
        self.next_order_id += 1
        return int(self.next_order_id)

    def current_time(self) -> dt.datetime:
        ts = self.current_df.index[-1]
        return ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts

    def current_price(self) -> float:
        return float(self.current_df.iloc[-1]["close"])

    def startup_history(self, lookback_minutes: int) -> pd.DataFrame:
        if lookback_minutes > 0 and len(self.warmup_df) > lookback_minutes:
            return self.warmup_df.iloc[-lookback_minutes:].copy()
        return self.warmup_df.copy()

    def current_history(self, lookback_minutes: int) -> pd.DataFrame:
        if lookback_minutes > 0 and len(self.current_df) > lookback_minutes:
            return self.current_df.iloc[-lookback_minutes:].copy()
        return self.current_df.copy()

    def _fill_leg(
        self,
        leg: dict,
        *,
        exit_price: float,
        exit_time: dt.datetime,
        order_id: int,
        source: str,
    ) -> None:
        side = str(leg.get("side") or "").upper()
        size = int(leg.get("size") or 0)
        entry_price = float(leg.get("entry_price") or 0.0)
        pnl_points = _compute_pnl_points(side, entry_price, exit_price)
        gross = pnl_points * _point_value() * float(size)
        fees = _trade_fee_per_contract() * float(size)
        pnl_dollars = gross - fees
        self.balance += float(pnl_dollars)
        closed = {
            "strategy": leg.get("strategy"),
            "sub_strategy": leg.get("sub_strategy"),
            "combo_key": leg.get("combo_key"),
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "entry_time": leg.get("entry_time"),
            "exit_time": exit_time,
            "entry_order_id": leg.get("entry_order_id"),
            "order_id": int(order_id),
            "source": str(source),
            "pnl_points": float(pnl_points),
            "pnl_dollars": float(pnl_dollars),
            "raw_close_rows": [],
            "entry_order_ids": [leg.get("entry_order_id")],
        }
        self.closed_trades.append(closed)
        self.filled_exit_orders[int(order_id)] = {
            "complete": True,
            "avg_price": float(exit_price),
            "latest_fill_time": exit_time,
            "size": int(size),
            "order_id": int(order_id),
        }

    def _simulate_passive_fills(self, bar_time: dt.datetime, bar_row: pd.Series) -> None:
        if not self.open_legs:
            return
        bar_open = float(bar_row["open"])
        bar_high = float(bar_row["high"])
        bar_low = float(bar_row["low"])
        bar_close = float(bar_row["close"])
        survivors: list[dict] = []
        for leg in self.open_legs:
            side = str(leg.get("side") or "").upper()
            stop_price = float(leg.get("stop_price") or math.nan)
            take_price = float(leg.get("target_price") or math.nan)
            stop_order_id = int(leg.get("stop_order_id") or 0)
            target_order_id = int(leg.get("target_order_id") or 0)
            exit_price = math.nan
            exit_source = ""
            exit_order_id = 0
            if side == "LONG":
                if math.isfinite(stop_price) and bar_open <= stop_price:
                    exit_price, exit_source, exit_order_id = stop_price, "stop_gap", stop_order_id
                elif math.isfinite(take_price) and bar_open >= take_price:
                    exit_price, exit_source, exit_order_id = take_price, "take_gap", target_order_id
                else:
                    hit_stop = math.isfinite(stop_price) and bar_low <= stop_price
                    hit_take = math.isfinite(take_price) and bar_high >= take_price
                    if hit_stop and hit_take:
                        exit_price, exit_source = _resolve_sl_tp_conflict(
                            "LONG",
                            bar_open,
                            bar_close,
                            stop_price,
                            take_price,
                        )
                        exit_order_id = stop_order_id if exit_source == "stop" else target_order_id
                    elif hit_take:
                        exit_price, exit_source, exit_order_id = take_price, "take", target_order_id
                    elif hit_stop:
                        exit_price, exit_source, exit_order_id = stop_price, "stop", stop_order_id
            elif side == "SHORT":
                if math.isfinite(stop_price) and bar_open >= stop_price:
                    exit_price, exit_source, exit_order_id = stop_price, "stop_gap", stop_order_id
                elif math.isfinite(take_price) and bar_open <= take_price:
                    exit_price, exit_source, exit_order_id = take_price, "take_gap", target_order_id
                else:
                    hit_stop = math.isfinite(stop_price) and bar_high >= stop_price
                    hit_take = math.isfinite(take_price) and bar_low <= take_price
                    if hit_stop and hit_take:
                        exit_price, exit_source = _resolve_sl_tp_conflict(
                            "SHORT",
                            bar_open,
                            bar_close,
                            stop_price,
                            take_price,
                        )
                        exit_order_id = stop_order_id if exit_source == "stop" else target_order_id
                    elif hit_take:
                        exit_price, exit_source, exit_order_id = take_price, "take", target_order_id
                    elif hit_stop:
                        exit_price, exit_source, exit_order_id = stop_price, "stop", stop_order_id

            if math.isfinite(exit_price):
                self.passive_exit_counts[exit_source] += 1
                self._fill_leg(
                    leg,
                    exit_price=float(exit_price),
                    exit_time=bar_time,
                    order_id=int(exit_order_id),
                    source=str(exit_source),
                )
            else:
                survivors.append(leg)
        self.open_legs = survivors

    def advance_one_bar(self) -> pd.DataFrame:
        if self.pointer >= len(self.replay_df):
            if self.open_legs:
                self.close_all(
                    exit_price=self.current_price(),
                    exit_time=self.current_time(),
                    reason="end_of_replay",
                )
            raise KeyboardInterrupt
        row = self.replay_df.iloc[self.pointer]
        ts = self.replay_df.index[self.pointer]
        self.pointer += 1
        self.bars_delivered += 1
        frame = pd.DataFrame([row], index=[ts])
        self.current_df = pd.concat([self.current_df, frame])
        self.current_df = self.current_df[~self.current_df.index.duplicated(keep="last")]
        self._simulate_passive_fills(
            ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
            row,
        )
        return frame

    def aggregate_position(self) -> dict:
        if not self.open_legs:
            return {"side": None, "size": 0, "avg_price": 0.0, "open_pnl": 0.0}
        side = str(self.open_legs[0].get("side") or "").upper()
        total_size = sum(int(leg.get("size") or 0) for leg in self.open_legs)
        if total_size <= 0:
            return {"side": None, "size": 0, "avg_price": 0.0, "open_pnl": 0.0}
        weighted_entry = sum(
            float(leg.get("entry_price") or 0.0) * int(leg.get("size") or 0)
            for leg in self.open_legs
        ) / float(total_size)
        current_price = self.current_price()
        open_pnl = sum(
            _compute_pnl_points(side, float(leg.get("entry_price") or 0.0), current_price)
            * _point_value()
            * int(leg.get("size") or 0)
            for leg in self.open_legs
        )
        return {
            "side": side,
            "size": int(total_size),
            "avg_price": float(weighted_entry),
            "open_pnl": float(open_pnl),
        }

    def open_leg(self, signal: dict, current_price: float) -> dict:
        entry_price = float(current_price)
        side = str(signal.get("side") or "").upper()
        size = max(1, int(signal.get("size") or 1))
        tp_points = float(signal.get("tp_dist") or 0.0)
        sl_points = float(signal.get("sl_dist") or 0.0)
        entry_order_id = self._new_id()
        stop_order_id = self._new_id()
        target_order_id = self._new_id()
        target_price = entry_price + tp_points if side == "LONG" else entry_price - tp_points
        stop_price = entry_price - sl_points if side == "LONG" else entry_price + sl_points
        leg = {
            "strategy": signal.get("strategy"),
            "sub_strategy": signal.get("sub_strategy"),
            "combo_key": signal.get("combo_key"),
            "side": side,
            "size": size,
            "entry_price": float(entry_price),
            "entry_time": self.current_time(),
            "entry_order_id": int(entry_order_id),
            "stop_order_id": int(stop_order_id),
            "target_order_id": int(target_order_id),
            "stop_price": float(stop_price),
            "target_price": float(target_price),
        }
        self.open_legs.append(leg)
        return {
            "broker_order_id": int(entry_order_id),
            "order_id": int(entry_order_id),
            "entry_price": float(entry_price),
            "size": int(size),
            "tp_points": float(tp_points),
            "sl_points": float(sl_points),
            "tp_price": float(target_price),
            "sl_price": float(stop_price),
            "target_order_id": int(target_order_id),
            "stop_order_id": int(stop_order_id),
        }

    def close_all(self, *, exit_price: float, exit_time: dt.datetime, reason: str) -> Optional[int]:
        if not self.open_legs:
            return None
        close_order_id = self._new_id()
        for leg in list(self.open_legs):
            self._fill_leg(
                leg,
                exit_price=float(exit_price),
                exit_time=exit_time,
                order_id=int(close_order_id),
                source=str(reason),
            )
        self.open_legs = []
        self.manual_close_count += 1
        return int(close_order_id)

    def summary(self) -> dict:
        wins = sum(1 for row in self.closed_trades if float(row.get("pnl_dollars", 0.0)) > 0.0)
        losses = sum(1 for row in self.closed_trades if float(row.get("pnl_dollars", 0.0)) < 0.0)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for row in self.closed_trades:
            equity += float(row.get("pnl_dollars", 0.0) or 0.0)
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return {
            "bars_processed": int(self.bars_delivered),
            "closed_trades": int(len(self.closed_trades)),
            "wins": int(wins),
            "losses": int(losses),
            "winrate": float((wins / len(self.closed_trades)) * 100.0) if self.closed_trades else 0.0,
            "net_pnl": float(sum(float(row.get("pnl_dollars", 0.0) or 0.0) for row in self.closed_trades)),
            "max_drawdown": float(max_dd),
            "ending_balance": float(self.balance),
            "open_legs_remaining": int(len(self.open_legs)),
            "manual_closes": int(self.manual_close_count),
            "reversals": int(self.reverse_count),
            "stop_updates": int(self.stop_update_count),
            "passive_exit_counts": dict(self.passive_exit_counts),
            "strategy_counts": dict(Counter(str(row.get("strategy") or "Unknown") for row in self.closed_trades)),
            "exit_source_counts": dict(Counter(str(row.get("source") or "unknown") for row in self.closed_trades)),
        }


class ReplayProjectXClient:
    session: ReplaySession | None = None
    instances: list["ReplayProjectXClient"] = []
    primary: "ReplayProjectXClient" | None = None

    @classmethod
    def configure(cls, session: ReplaySession) -> None:
        cls.session = session
        cls.instances = []
        cls.primary = None

    def __init__(self, contract_root: Optional[str] = None, target_symbol: Optional[str] = None):
        if self.__class__.session is None:
            raise RuntimeError("ReplayProjectXClient session not configured.")
        self.session = self.__class__.session
        self.contract_root = contract_root or "MES"
        self.target_symbol = target_symbol or determine_current_contract_symbol(self.contract_root)
        self.account_id = CONFIG.get("ACCOUNT_ID")
        self.contract_id = f"CON.F.US.{self.contract_root}.{self.target_symbol.split('.', 1)[-1]}"
        self.cached_df = pd.DataFrame()
        self.last_bar_timestamp = None
        self.session_obj = None
        self.token = "replay-token"
        self.et = NY_TZ
        self._local_position = {"side": None, "size": 0, "avg_price": 0.0}
        self._active_stop_order_id = None
        self._active_target_order_id = None
        self._last_order_details = None
        self._last_close_order_details = None
        self._runtime_state_persist_ready = False
        self._startup_history_served = False
        self._user_stream = None
        self.__class__.instances.append(self)
        if self.contract_root == "MES" and self.__class__.primary is None:
            self.__class__.primary = self

    def login(self):
        return True

    def fetch_accounts(self) -> int:
        account_id = CONFIG.get("ACCOUNT_ID")
        if account_id in (None, ""):
            account_id = _discover_account_from_state() or 0
        self.account_id = int(account_id)
        return int(self.account_id)

    def fetch_contracts(self) -> str:
        return str(self.contract_id)

    async def start_user_stream(self) -> bool:
        return False

    async def stop_user_stream(self) -> None:
        return None

    def _auth_temporarily_unavailable(self) -> bool:
        return False

    async def async_validate_session(self) -> bool:
        return True

    def get_account_info(self, force_refresh: bool = False) -> dict:
        del force_refresh
        return {
            "id": int(self.account_id or 0),
            "balance": float(self.session.balance),
        }

    def _sync_local_position(self) -> dict:
        self._local_position = self.session.aggregate_position().copy()
        return self._local_position.copy()

    def get_market_data(self, lookback_minutes: int = 20000, force_fetch: bool = False) -> pd.DataFrame:
        del force_fetch
        if not self._startup_history_served and lookback_minutes >= 20_000:
            df = self.session.startup_history(lookback_minutes)
            self.cached_df = df.copy()
            self._startup_history_served = True
            if not df.empty:
                self.last_bar_timestamp = df.index[-1]
            return df.copy()
        if not self._startup_history_served:
            df = self.session.startup_history(lookback_minutes)
            self.cached_df = df.copy()
            if not df.empty:
                self.last_bar_timestamp = df.index[-1]
            return df.copy()
        recent = self.session.advance_one_bar()
        self.cached_df = self.session.current_history(max(lookback_minutes, 1000))
        self.last_bar_timestamp = self.cached_df.index[-1]
        self._sync_local_position()
        return recent.copy()

    async def async_get_market_data(self, lookback_minutes: int = 20000, force_fetch: bool = False) -> pd.DataFrame:
        return self.get_market_data(lookback_minutes=lookback_minutes, force_fetch=force_fetch)

    def fetch_custom_bars(self, lookback_bars: int, minutes_per_bar: int) -> pd.DataFrame:
        history = self.session.current_history(50_000)
        return _resample_ohlcv(history, int(minutes_per_bar), int(lookback_bars))

    def get_position(self, *, prefer_stream: bool = True, require_open_pnl: bool = False) -> dict:
        del prefer_stream, require_open_pnl
        return self._sync_local_position()

    async def async_get_position(
        self,
        *,
        prefer_stream: bool = True,
        require_open_pnl: bool = False,
    ) -> dict:
        return self.get_position(prefer_stream=prefer_stream, require_open_pnl=require_open_pnl)

    def place_order(self, signal: dict, current_price: float):
        order_details = self.session.open_leg(signal, current_price)
        self._last_order_details = dict(order_details)
        self._active_stop_order_id = order_details.get("stop_order_id")
        self._active_target_order_id = order_details.get("target_order_id")
        self._sync_local_position()
        return {"success": True, "order_id": order_details.get("order_id")}

    async def async_place_order(self, signal: dict, current_price: float):
        return self.place_order(signal, current_price)

    def close_position(self, position: dict) -> bool:
        del position
        close_order_id = self.session.close_all(
            exit_price=self.session.current_price(),
            exit_time=self.session.current_time(),
            reason="close_position",
        )
        self._last_close_order_details = {
            "order_id": int(close_order_id) if close_order_id is not None else None,
            "exit_price": float(self.session.current_price()),
        }
        self._sync_local_position()
        self._active_stop_order_id = None
        self._active_target_order_id = None
        return True

    def emergency_flatten_position(self, position: dict, reason: str = "") -> bool:
        del position
        close_order_id = self.session.close_all(
            exit_price=self.session.current_price(),
            exit_time=self.session.current_time(),
            reason=reason or "emergency_flatten",
        )
        self._last_close_order_details = {
            "order_id": int(close_order_id) if close_order_id is not None else None,
            "exit_price": float(self.session.current_price()),
        }
        self._sync_local_position()
        self._active_stop_order_id = None
        self._active_target_order_id = None
        return True

    async def async_emergency_flatten_position(self, position: dict, reason: str = "") -> bool:
        return self.emergency_flatten_position(position, reason=reason)

    async def async_close_and_reverse(
        self,
        new_signal: dict,
        current_price: float,
        opposite_signal_count: int,
    ):
        del opposite_signal_count
        if self.session.open_legs:
            self.session.reverse_count += 1
            close_order_id = self.session.close_all(
                exit_price=float(current_price),
                exit_time=self.session.current_time(),
                reason="reverse",
            )
            self._last_close_order_details = {
                "order_id": int(close_order_id) if close_order_id is not None else None,
                "exit_price": float(current_price),
            }
        self.place_order(new_signal, current_price)
        return True, 0

    def get_live_bracket_state(
        self,
        *,
        side: Optional[str] = None,
        size: Optional[int] = None,
        reference_price: Optional[float] = None,
        expected_stop_price: Optional[float] = None,
        expected_target_price: Optional[float] = None,
        prefer_stop_order_id: Optional[int] = None,
        prefer_target_order_id: Optional[int] = None,
        max_cache_age_sec: float = 0.0,
        force_refresh: bool = False,
    ) -> dict:
        del reference_price, expected_stop_price, expected_target_price, max_cache_age_sec, force_refresh
        legs = list(self.session.open_legs)
        if prefer_stop_order_id is not None:
            for leg in legs:
                if int(leg.get("stop_order_id") or 0) == int(prefer_stop_order_id):
                    return {
                        "stop_order_id": leg.get("stop_order_id"),
                        "target_order_id": leg.get("target_order_id"),
                        "stop_price": leg.get("stop_price"),
                        "target_price": leg.get("target_price"),
                    }
        if prefer_target_order_id is not None:
            for leg in legs:
                if int(leg.get("target_order_id") or 0) == int(prefer_target_order_id):
                    return {
                        "stop_order_id": leg.get("stop_order_id"),
                        "target_order_id": leg.get("target_order_id"),
                        "stop_price": leg.get("stop_price"),
                        "target_price": leg.get("target_price"),
                    }
        for leg in legs:
            if side and str(leg.get("side") or "").upper() != str(side or "").upper():
                continue
            if size and int(leg.get("size") or 0) != int(size):
                continue
            return {
                "stop_order_id": leg.get("stop_order_id"),
                "target_order_id": leg.get("target_order_id"),
                "stop_price": leg.get("stop_price"),
                "target_price": leg.get("target_price"),
            }
        return {}

    def get_live_bracket_snapshot(self, **kwargs) -> dict:
        return self.get_live_bracket_state(**kwargs)

    def modify_stop_to_breakeven(
        self,
        *,
        stop_price: float,
        side: str,
        known_size: Optional[int] = None,
        stop_order_id: Optional[int] = None,
        current_stop_price: Optional[float] = None,
    ) -> bool:
        del current_stop_price
        for leg in self.session.open_legs:
            if stop_order_id is not None and int(leg.get("stop_order_id") or 0) != int(stop_order_id):
                continue
            if str(leg.get("side") or "").upper() != str(side or "").upper():
                continue
            if known_size is not None and int(leg.get("size") or 0) != int(known_size):
                continue
            leg["stop_price"] = float(stop_price)
            self._active_stop_order_id = leg.get("stop_order_id")
            self.session.stop_update_count += 1
            return True
        return False

    def get_trade_fill_summary(
        self,
        order_id: int,
        *,
        start_time: Optional[dt.datetime] = None,
        end_time: Optional[dt.datetime] = None,
        min_qty: int = 0,
    ) -> Optional[dict]:
        del start_time, end_time, min_qty
        result = self.session.filled_exit_orders.get(int(order_id))
        return dict(result) if isinstance(result, dict) else None

    def cancel_order(self, order_id: int) -> bool:
        cancelled = False
        for leg in self.session.open_legs:
            if int(leg.get("stop_order_id") or 0) == int(order_id):
                leg["stop_order_id"] = None
                cancelled = True
            if int(leg.get("target_order_id") or 0) == int(order_id):
                leg["target_order_id"] = None
                cancelled = True
        return bool(cancelled)

    def cancel_open_exit_orders(self, side: Optional[str] = None, reason: str = "") -> int:
        del reason
        count = 0
        for leg in self.session.open_legs:
            if side and str(leg.get("side") or "").upper() != str(side).upper():
                continue
            if leg.get("stop_order_id") is not None:
                leg["stop_order_id"] = None
                count += 1
            if leg.get("target_order_id") is not None:
                leg["target_order_id"] = None
                count += 1
        return int(count)

    def reconstruct_closed_trades(
        self,
        start_time: dt.datetime,
        end_time: dt.datetime,
        include_stream_trades: bool = True,
    ) -> list[dict]:
        del include_stream_trades
        out = []
        for row in self.session.closed_trades:
            exit_time = row.get("exit_time")
            if not isinstance(exit_time, dt.datetime):
                continue
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=NY_TZ)
            if start_time <= exit_time <= end_time:
                out.append(dict(row))
        return out

    def reconcile_trade_close(
        self,
        active_trade: dict,
        *,
        exit_time: Optional[dt.datetime] = None,
        fallback_exit_price: Optional[float] = None,
        close_order_id: Optional[int] = None,
        point_value: float = 5.0,
    ) -> Optional[dict]:
        entry_order_id = active_trade.get("entry_order_id")
        for row in reversed(self.session.closed_trades):
            if close_order_id is not None and int(row.get("order_id") or 0) == int(close_order_id):
                if entry_order_id is None or row.get("entry_order_id") == entry_order_id:
                    return {
                        "entry_price": row.get("entry_price"),
                        "exit_price": row.get("exit_price"),
                        "pnl_points": row.get("pnl_points"),
                        "pnl_dollars": row.get("pnl_dollars"),
                        "exit_time": row.get("exit_time"),
                        "order_id": row.get("order_id"),
                        "entry_order_id": row.get("entry_order_id"),
                        "source": row.get("source"),
                    }
            if entry_order_id is not None and row.get("entry_order_id") == entry_order_id:
                return {
                    "entry_price": row.get("entry_price"),
                    "exit_price": row.get("exit_price"),
                    "pnl_points": row.get("pnl_points"),
                    "pnl_dollars": row.get("pnl_dollars"),
                    "exit_time": row.get("exit_time"),
                    "order_id": row.get("order_id"),
                    "entry_order_id": row.get("entry_order_id"),
                    "source": row.get("source"),
                }
        if fallback_exit_price is None:
            return None
        entry_price = float(active_trade.get("entry_price") or fallback_exit_price)
        side = str(active_trade.get("side") or "").upper()
        size = int(active_trade.get("size") or 1)
        pnl_points = _compute_pnl_points(side, entry_price, float(fallback_exit_price))
        pnl_dollars = (pnl_points * float(point_value) * float(size)) - (_trade_fee_per_contract() * float(size))
        return {
            "entry_price": float(entry_price),
            "exit_price": float(fallback_exit_price),
            "pnl_points": float(pnl_points),
            "pnl_dollars": float(pnl_dollars),
            "exit_time": exit_time or self.session.current_time(),
            "order_id": close_order_id,
            "entry_order_id": entry_order_id,
            "source": "fallback_price_snapshot",
        }


async def _noop_background(*args, **kwargs):
    del args, kwargs
    return None


def _install_typeerror_trace(run_dir: Path) -> tuple[Any, Any]:
    trace_path = run_dir / "exception_trace.log"
    seen_signatures: set[str] = set()

    def _trace(frame, event, arg):
        if event != "exception":
            return _trace
        exc_type, exc_value, exc_tb = arg
        if exc_type is not TypeError:
            return _trace
        message = str(exc_value or "")
        if "float() argument must be a string or a real number, not 'NoneType'" not in message:
            return _trace
        code = getattr(frame, "f_code", None)
        filename = str(getattr(code, "co_filename", "") or "")
        if str(ROOT) not in filename:
            return _trace
        signature = f"{filename}:{getattr(frame, 'f_lineno', 0)}:{message}"
        if signature in seen_signatures:
            return _trace
        seen_signatures.add(signature)
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        stack_lines = traceback.format_stack(frame)
        payload = [
            "=" * 80,
            f"TYPEERROR TRACE @ {filename}:{getattr(frame, 'f_lineno', 0)}",
            f"MESSAGE: {message}",
            "--- STACK ---",
            *[line.rstrip("\n") for line in stack_lines],
            "--- TRACEBACK ---",
            *[line.rstrip("\n") for line in tb_lines],
            "",
        ]
        with trace_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(payload))
            fh.write("\n")
        return _trace

    previous_trace = sys.gettrace()
    previous_threading_trace = threading.gettrace()
    sys.settrace(_trace)
    threading.settrace(_trace)
    return previous_trace, previous_threading_trace


def _restore_typeerror_trace(previous_trace: Any, previous_threading_trace: Any) -> None:
    sys.settrace(previous_trace)
    threading.settrace(previous_threading_trace)


def _install_replay_clock(session: ReplaySession, live_module: Any) -> Any:
    original_datetime_cls = live_module.datetime.datetime
    session_obj = session

    class ReplayDateTime(original_datetime_cls):
        @classmethod
        def now(cls, tz=None):
            current = session_obj.current_time()
            if tz is not None:
                return current.astimezone(tz)
            return current

        @classmethod
        def utcnow(cls):
            return session_obj.current_time().astimezone(dt.timezone.utc).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

    live_module.datetime.datetime = ReplayDateTime
    return original_datetime_cls


def _restore_replay_clock(live_module: Any, original_datetime_cls: Any) -> None:
    live_module.datetime.datetime = original_datetime_cls


def _prepare_simulation_env(run_dir: Path, account_id: int, contract_root: str) -> None:
    os.environ["JULIE_FILTERLESS_ONLY"] = "1"
    os.environ["JULIE_DISABLE_STRATEGY_FILTERS"] = "1"
    os.environ["JULIE_FILTERLESS_KEEP_GEMINI"] = "0"
    os.environ["JULIE_ACCOUNT_ID"] = str(account_id)
    os.chdir(run_dir)
    CONFIG["ACCOUNT_ID"] = int(account_id)
    CONFIG["CONTRACT_ROOT"] = str(contract_root).upper()
    CONFIG["TARGET_SYMBOL"] = determine_current_contract_symbol(contract_root)
    CONFIG["PROJECTX_USER_STREAM_ENABLED"] = False
    CONFIG["LIVE_MES_CSV_APPENDER_ENABLED"] = False
    CONFIG["LIVE_TRADE_FACTORS_LOGGER_ENABLED"] = False
    truth_cfg = CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) or {}
    if isinstance(truth_cfg, dict):
        truth_cfg["enabled"] = False
    disabled = {
        str(item).strip().lower()
        for item in (CONFIG.get("FILTERLESS_LIVE_DISABLED_STRATEGIES", []) or [])
        if str(item).strip()
    }
    disabled.add("ml_physics")
    CONFIG["FILTERLESS_LIVE_DISABLED_STRATEGIES"] = sorted(disabled)


def _collect_log_markers(log_path: Path) -> dict[str, int]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    markers = {
        "level_fill_pending": "level_fill_pending",
        "level_fill_fire": "LevelFill FIRE",
        "level_fill_at_level": "LevelFill AT LEVEL",
        "pivot_trail_updates": "[PivotTrail]",
        "kalshi_tp_trail": "Kalshi TP trail ratchet",
        "kalshi_tp_adjust": "Kalshi TP overlay:",
        "kalshi_entry_blocks": "Kalshi overlay blocked entry:",
        "de3_break_even_armed": "DE3 v4 break-even armed:",
        "trade_placed": "TRADE_PLACED",
        "trade_rejected": "TRADE_REJECTED",
    }
    return {key: text.count(token) for key, token in markers.items()}


_KALSHI_SETTLEMENT_HOURS_ET: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16)
_KALSHI_SIM_STRIKE_STEP: float = 5.0
_KALSHI_SIM_STRIKES_PER_SIDE: int = 15
_KALSHI_SIM_MIN_SIGMA_PTS: float = 1.25
_KALSHI_SIM_SIGMA_BARS_LOOKBACK: int = 60


class SimulatedKalshiProvider:
    """Deterministic Kalshi crowd-curve simulator backed by the replay tape.

    Rebuilds a KXINXU-style probability ladder each bar from the current MES
    price and recent realized volatility.  This gives the live overlay an
    informative curve during replay, so entry_view / tp_adjust / tp_trail
    branches exercise the same code paths as in production.
    """

    series: str = "KXINXU_SIM"

    def __init__(self, session: "ReplaySession"):
        self.session = session
        self.enabled = True
        self.is_healthy = True
        self.basis_offset = 0.0
        self._strike_step = float(_KALSHI_SIM_STRIKE_STEP)
        self._strikes_per_side = int(_KALSHI_SIM_STRIKES_PER_SIDE)
        self._min_sigma_pts = float(_KALSHI_SIM_MIN_SIGMA_PTS)
        self._sigma_bars_lookback = int(_KALSHI_SIM_SIGMA_BARS_LOOKBACK)
        self._normal = NormalDist()
        self._last_bar_key: Optional[int] = None
        self._last_bar_markets: list[dict] = []
        self._last_bar_sigma: float = 0.0
        self._sentiment_history: deque[tuple[int, float, float]] = deque(maxlen=240)

    def _bar_key(self) -> int:
        ts = self.session.current_time()
        return int(pd.Timestamp(ts).value)

    def _current_reference_price(self) -> float:
        return float(self.session.current_price())

    def _recent_sigma_points(self) -> float:
        df = self.session.current_df
        if df is None or df.empty:
            return self._min_sigma_pts
        tail = df.tail(max(2, self._sigma_bars_lookback))
        closes = tail["close"].astype(float).diff().dropna()
        if len(closes) < 5:
            return self._min_sigma_pts
        realized = float(closes.std(ddof=0))
        return max(self._min_sigma_pts, realized)

    def _minutes_to_next_settlement(self, ref_time: dt.datetime) -> float:
        et = ref_time.astimezone(NY_TZ)
        hour = et.hour
        for settlement_hour in _KALSHI_SETTLEMENT_HOURS_ET:
            if settlement_hour > hour:
                target = et.replace(hour=settlement_hour, minute=0, second=0, microsecond=0)
                return max(1.0, (target - et).total_seconds() / 60.0)
        tomorrow = et + dt.timedelta(days=1)
        target = tomorrow.replace(hour=_KALSHI_SETTLEMENT_HOURS_ET[0], minute=0, second=0, microsecond=0)
        return max(1.0, (target - et).total_seconds() / 60.0)

    def _build_markets_for_bar(self) -> list[dict]:
        key = self._bar_key()
        if key == self._last_bar_key:
            return self._last_bar_markets
        ref_price = self._current_reference_price()
        sigma_bar = self._recent_sigma_points()
        minutes_to_close = self._minutes_to_next_settlement(self.session.current_time())
        sigma_total = max(self._min_sigma_pts, sigma_bar * math.sqrt(max(1.0, minutes_to_close)))
        self._last_bar_sigma = sigma_total
        center_strike = round(ref_price / self._strike_step) * self._strike_step
        markets: list[dict] = []
        for offset in range(-self._strikes_per_side, self._strikes_per_side + 1):
            strike_es = float(center_strike + offset * self._strike_step)
            z = (ref_price - strike_es) / sigma_total
            probability = self._normal.cdf(z)
            probability = min(0.995, max(0.005, float(probability)))
            markets.append(
                {
                    "strike": float(strike_es),
                    "strike_spx": float(strike_es),
                    "strike_es": float(strike_es),
                    "probability": round(probability, 4),
                    "status": "active",
                    "reference_es": float(ref_price),
                }
            )
        markets.sort(key=lambda row: float(row["strike_es"]))
        self._last_bar_markets = markets
        self._last_bar_key = key
        return markets

    def _interpolated_probability(self, es_price: float) -> Optional[float]:
        markets = self._build_markets_for_bar()
        if not markets:
            return None
        below = [m for m in markets if m["strike_es"] <= es_price]
        above = [m for m in markets if m["strike_es"] > es_price]
        if below and above:
            lo = below[-1]
            hi = above[0]
            span = float(hi["strike_es"] - lo["strike_es"])
            if span <= 0.0:
                return float(lo["probability"])
            frac = (float(es_price) - float(lo["strike_es"])) / span
            return float(lo["probability"]) * (1.0 - frac) + float(hi["probability"]) * frac
        if below:
            return float(below[-1]["probability"])
        if above:
            return float(above[0]["probability"])
        return None

    def _implied_level_es(self) -> Optional[float]:
        markets = self._build_markets_for_bar()
        if len(markets) < 2:
            return None
        for idx in range(len(markets) - 1):
            p_lo = float(markets[idx]["probability"])
            p_hi = float(markets[idx + 1]["probability"])
            if p_lo >= 0.5 > p_hi:
                s_lo = float(markets[idx]["strike_es"])
                s_hi = float(markets[idx + 1]["strike_es"])
                if abs(p_hi - p_lo) < 1e-9:
                    return (s_lo + s_hi) / 2.0
                frac = (0.5 - p_lo) / (p_hi - p_lo)
                return s_lo + frac * (s_hi - s_lo)
        return None

    def es_to_spx(self, es_price: float) -> float:
        return float(es_price) - float(self.basis_offset)

    def spx_to_es(self, spx_price: float) -> float:
        return float(spx_price) + float(self.basis_offset)

    def active_settlement_hour_et(
        self,
        ref_time: Optional[dt.datetime] = None,
        rollover_minute: int = 5,
    ) -> Optional[int]:
        ref = ref_time or self.session.current_time()
        et = ref.astimezone(NY_TZ) if ref.tzinfo is not None else ref.replace(tzinfo=NY_TZ)
        for hour in _KALSHI_SETTLEMENT_HOURS_ET:
            if hour > et.hour or (hour == et.hour and et.minute < int(rollover_minute)):
                return int(hour)
        return None

    def get_relative_markets_for_ui(
        self,
        es_prices: Optional[list[float]] = None,
        window_size: int = 120,
    ) -> list[dict]:
        markets = self._build_markets_for_bar()
        if not markets:
            return []
        window_size = max(8, int(window_size))
        if len(markets) <= window_size:
            return [dict(row) for row in markets]
        reference = float(es_prices[0]) if es_prices else self._current_reference_price()
        nearest_idx = min(range(len(markets)), key=lambda i: abs(float(markets[i]["strike_es"]) - reference))
        half = window_size // 2
        start = max(0, nearest_idx - half)
        end = min(len(markets), start + window_size)
        if (end - start) < window_size:
            start = max(0, end - window_size)
        return [dict(row) for row in markets[start:end]]

    def get_probability(self, strike_price: float) -> Optional[float]:
        return self._interpolated_probability(float(strike_price))

    def get_probability_curve(self) -> dict:
        return {
            float(row["strike_es"]): float(row["probability"])
            for row in self._build_markets_for_bar()
        }

    def get_nearest_market_for_es_price(self, es_price: float) -> Optional[dict]:
        markets = self._build_markets_for_bar()
        if not markets:
            return None
        nearest = min(markets, key=lambda row: abs(float(row["strike_es"]) - float(es_price)))
        return dict(nearest)

    def get_target_probability(self, es_price: float, side: Optional[str] = None) -> dict:
        market = self.get_nearest_market_for_es_price(es_price)
        payload = {
            "probability": None,
            "market_probability": None,
            "outcome_side": None,
            "strike_spx": None,
            "strike_es": None,
            "reference_spx": None,
            "reference_es": None,
            "distance_spx": None,
            "distance_es": None,
            "status": None,
            "result": None,
        }
        if market is None:
            return payload
        raw_probability = float(market.get("probability") or 0.0)
        normalized_side = str(side or "").strip().upper()
        outcome_side = "below" if normalized_side == "SHORT" else "above"
        probability = 1.0 - raw_probability if outcome_side == "below" else raw_probability
        payload.update(
            {
                "probability": round(float(probability), 4),
                "market_probability": round(raw_probability, 4),
                "outcome_side": outcome_side,
                "strike_spx": market.get("strike_spx"),
                "strike_es": market.get("strike_es"),
                "reference_spx": float(es_price),
                "reference_es": float(es_price),
                "distance_spx": round(float(market["strike_es"]) - float(es_price), 2),
                "distance_es": round(float(market["strike_es"]) - float(es_price), 2),
                "status": market.get("status"),
                "result": None,
            }
        )
        return payload

    def get_sentiment(self, es_price: float) -> dict:
        probability = self._interpolated_probability(float(es_price))
        implied_level = self._implied_level_es()
        payload = {
            "probability": None,
            "classification": "unavailable",
            "implied_level": None,
            "distance": None,
            "implied_level_es": None,
            "distance_es": None,
            "implied_level_spx": None,
            "distance_spx": None,
            "healthy": True,
        }
        if probability is None:
            return payload
        if probability >= 0.70:
            classification = "strong_bull"
        elif probability >= 0.55:
            classification = "bull"
        elif probability >= 0.45:
            classification = "neutral"
        elif probability >= 0.30:
            classification = "bear"
        else:
            classification = "strong_bear"
        distance_es = (
            float(implied_level) - float(es_price) if implied_level is not None else None
        )
        payload.update(
            {
                "probability": round(float(probability), 4),
                "classification": classification,
                "implied_level": round(float(implied_level), 2) if implied_level is not None else None,
                "distance": round(float(distance_es), 2) if distance_es is not None else None,
                "implied_level_es": round(float(implied_level), 2) if implied_level is not None else None,
                "distance_es": round(float(distance_es), 2) if distance_es is not None else None,
                "implied_level_spx": round(float(implied_level), 2) if implied_level is not None else None,
                "distance_spx": round(float(distance_es), 2) if distance_es is not None else None,
                "healthy": True,
            }
        )
        self._record_sentiment_history(float(es_price), float(probability))
        return payload

    def _record_sentiment_history(self, es_price: float, probability: float) -> None:
        bar_key = self._bar_key()
        if self._sentiment_history and self._sentiment_history[-1][0] == bar_key:
            self._sentiment_history[-1] = (bar_key, float(es_price), float(probability))
            return
        self._sentiment_history.append((bar_key, float(es_price), float(probability)))

    def get_probability_gradient(self, es_price: float) -> Optional[float]:
        markets = self._build_markets_for_bar()
        if len(markets) < 3:
            return None
        below = [m for m in markets if float(m["strike_es"]) <= float(es_price)]
        above = [m for m in markets if float(m["strike_es"]) > float(es_price)]
        if not below or not above:
            return None
        lo = below[-1]
        hi = above[0]
        span = float(hi["strike_es"]) - float(lo["strike_es"])
        if span <= 0.0:
            return None
        gradient = (float(hi["probability"]) - float(lo["probability"])) / span
        return round(float(gradient), 6)

    def get_sentiment_momentum(self, es_price: float, lookback: int = 3) -> Optional[float]:
        probability = self._interpolated_probability(float(es_price))
        if probability is None:
            return None
        self._record_sentiment_history(float(es_price), float(probability))
        if len(self._sentiment_history) < int(lookback) + 1:
            return None
        prior_probability = float(self._sentiment_history[-(int(lookback) + 1)][2])
        return round(float(probability - prior_probability), 4)

    def get_implied_level(self) -> Optional[float]:
        spx_level = self._implied_level_es()
        if spx_level is None:
            return None
        return float(self.es_to_spx(spx_level))

    def clear_cache(self) -> None:
        self._last_bar_key = None
        self._last_bar_markets = []


def _install_simulated_kalshi_provider(
    *,
    session: "ReplaySession",
    live_module: Any,
) -> SimulatedKalshiProvider:
    provider = SimulatedKalshiProvider(session)
    live_module._KALSHI_PROVIDER = provider
    live_module._KALSHI_PROVIDER_INIT_DONE = True
    logging.info(
        "Replay SimulatedKalshiProvider installed (strike_step=%.2f strikes_per_side=%d)",
        provider._strike_step,
        provider._strikes_per_side,
    )
    return provider


# ---------------------------------------------------------------------------
# Historical Kalshi provider — pulls real archival Kalshi candlestick data
# from the Kalshi API for the replay window, so the overlay evaluates against
# the actual crowd-probability tape that existed alongside the ES bars.
# ---------------------------------------------------------------------------

_KALSHI_HISTORICAL_CACHE_DIR = ROOT / "backtest_reports" / "kalshi_historical_cache"
_KALSHI_HIST_STRIKE_BAND_SPX = 100.0
_KALSHI_HIST_CANDLE_INTERVAL = 1  # 1-minute candles


class _HistoricalKalshiEvent:
    __slots__ = ("event_ticker", "open_utc_ts", "close_utc_ts", "settlement_hour_et", "strikes")

    def __init__(
        self,
        event_ticker: str,
        open_utc_ts: int,
        close_utc_ts: int,
        settlement_hour_et: int,
        strikes: dict[float, list[tuple[int, float]]],
    ) -> None:
        self.event_ticker = event_ticker
        self.open_utc_ts = int(open_utc_ts)
        self.close_utc_ts = int(close_utc_ts)
        self.settlement_hour_et = int(settlement_hour_et)
        # strike_spx -> sorted list of (end_period_ts_utc_seconds, probability)
        self.strikes = strikes

    def curve_at(self, ts_utc: int) -> list[tuple[float, float]]:
        rows: list[tuple[float, float]] = []
        for strike, series in self.strikes.items():
            if not series:
                continue
            prob = series[0][1]
            for cts, cprob in series:
                if cts <= ts_utc:
                    prob = cprob
                else:
                    break
            rows.append((float(strike), float(prob)))
        rows.sort(key=lambda r: r[0])
        return rows


def _fetch_kalshi_historical_window(
    *,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    force_refresh: bool = False,
) -> dict[str, _HistoricalKalshiEvent]:
    """Pull real Kalshi event + candlestick data for every hourly event overlapping the window.

    Disk-cached per event in ``backtest_reports/kalshi_historical_cache/``.
    """

    try:
        from services.kalshi_provider import KalshiProvider
    except Exception as exc:
        logging.warning("Cannot import KalshiProvider: %s", exc)
        return {}

    kalshi_cfg = CONFIG.get("KALSHI", {}) if isinstance(CONFIG, dict) else {}
    provider = KalshiProvider(kalshi_cfg)
    if not getattr(provider, "enabled", False):
        logging.warning(
            "Real KalshiProvider is disabled (missing creds?); skipping historical fetch."
        )
        return {}

    _KALSHI_HISTORICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start_et = start_dt.astimezone(NY_TZ)
    end_et = end_dt.astimezone(NY_TZ)
    events: dict[str, _HistoricalKalshiEvent] = {}

    day_cursor = start_et.replace(hour=0, minute=0, second=0, microsecond=0)
    while day_cursor.date() <= end_et.date():
        for et_hour in _KALSHI_SETTLEMENT_HOURS_ET:
            event_ticker = (
                f"{provider.series}-"
                f"{day_cursor.strftime('%y%b%d').upper()}"
                f"H{et_hour * 100}"
            )
            open_et = day_cursor.replace(hour=max(0, et_hour - 1), minute=0, second=0, microsecond=0)
            close_et = day_cursor.replace(hour=et_hour, minute=0, second=0, microsecond=0)
            # Skip if event entirely outside replay window
            if close_et <= start_et or open_et >= end_et:
                continue

            cache_path = _KALSHI_HISTORICAL_CACHE_DIR / f"{event_ticker}.json"
            if cache_path.exists() and not force_refresh:
                try:
                    raw = json.loads(cache_path.read_text())
                    strikes = {
                        float(k): sorted((int(t), float(p)) for t, p in v)
                        for k, v in (raw.get("strikes") or {}).items()
                    }
                    events[event_ticker] = _HistoricalKalshiEvent(
                        event_ticker=raw["event_ticker"],
                        open_utc_ts=raw["open_utc_ts"],
                        close_utc_ts=raw["close_utc_ts"],
                        settlement_hour_et=raw["settlement_hour_et"],
                        strikes=strikes,
                    )
                    continue
                except Exception as exc:
                    logging.warning("Corrupt Kalshi cache %s: %s", cache_path, exc)

            open_utc_ts = int(open_et.astimezone(dt.timezone.utc).timestamp())
            close_utc_ts = int(close_et.astimezone(dt.timezone.utc).timestamp())

            resp = provider._get(f"/events/{event_ticker}", {"with_nested_markets": "true"})
            markets = ((resp or {}).get("event") or {}).get("markets") or []
            if not markets:
                logging.info("Kalshi event %s has no markets (skipping)", event_ticker)
                continue

            # Pick center strike from last_price transition band (0.02..0.98)
            transition = [
                m for m in markets
                if 0.02 < float(m.get("last_price_dollars") or 0.0) < 0.98
                and m.get("floor_strike") is not None
            ]
            if transition:
                center_strike = float(transition[len(transition) // 2]["floor_strike"])
            else:
                # Settled events have no transition band (all strikes resolved
                # YES at ~1.0 or NO at ~0.0). The center is the boundary between
                # the two: the highest YES-settled strike.
                with_strike = [
                    m for m in markets if m.get("floor_strike") is not None
                ]
                yes_settled = [
                    m for m in with_strike
                    if float(m.get("last_price_dollars") or 0.0) >= 0.5
                ]
                if yes_settled:
                    center_strike = max(
                        float(m["floor_strike"]) for m in yes_settled
                    )
                else:
                    center_strike = float(
                        min(
                            with_strike,
                            key=lambda m: abs(
                                float(m.get("last_price_dollars") or 0.0) - 0.5
                            ),
                        )["floor_strike"]
                    )

            strike_data: dict[float, list[tuple[int, float]]] = {}
            for market in markets:
                strike_raw = market.get("floor_strike")
                if strike_raw is None:
                    continue
                strike = float(strike_raw)
                if abs(strike - center_strike) > _KALSHI_HIST_STRIKE_BAND_SPX:
                    continue
                ticker = market.get("ticker")
                if not ticker:
                    continue
                cs_resp = provider._get(
                    f"/series/{provider.series}/markets/{ticker}/candlesticks",
                    {
                        "start_ts": open_utc_ts,
                        "end_ts": close_utc_ts,
                        "period_interval": _KALSHI_HIST_CANDLE_INTERVAL,
                    },
                )
                candles = (cs_resp or {}).get("candlesticks") or []
                series: list[tuple[int, float]] = []
                for candle in candles:
                    ts = candle.get("end_period_ts")
                    if ts is None:
                        continue
                    yes_bid = candle.get("yes_bid") or {}
                    yes_ask = candle.get("yes_ask") or {}
                    bid_val = yes_bid.get("close_dollars")
                    ask_val = yes_ask.get("close_dollars")
                    prob: Optional[float] = None
                    if bid_val is not None and ask_val is not None:
                        prob = (float(bid_val) + float(ask_val)) / 2.0
                    elif bid_val is not None:
                        prob = float(bid_val)
                    elif ask_val is not None:
                        prob = float(ask_val)
                    if prob is None:
                        continue
                    prob = min(0.995, max(0.005, prob))
                    series.append((int(ts), float(prob)))
                if not series:
                    last_price = market.get("last_price_dollars")
                    if last_price is not None:
                        series = [(open_utc_ts, min(0.995, max(0.005, float(last_price))))]
                strike_data[strike] = series

            evt = _HistoricalKalshiEvent(
                event_ticker=event_ticker,
                open_utc_ts=open_utc_ts,
                close_utc_ts=close_utc_ts,
                settlement_hour_et=et_hour,
                strikes=strike_data,
            )
            events[event_ticker] = evt
            cache_path.write_text(
                json.dumps(
                    {
                        "event_ticker": evt.event_ticker,
                        "open_utc_ts": evt.open_utc_ts,
                        "close_utc_ts": evt.close_utc_ts,
                        "settlement_hour_et": evt.settlement_hour_et,
                        "strikes": {
                            f"{k:.4f}": [[int(t), float(p)] for t, p in v]
                            for k, v in evt.strikes.items()
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logging.info(
                "Cached Kalshi event %s: %d strikes, %d candles",
                event_ticker,
                len(evt.strikes),
                sum(len(v) for v in evt.strikes.values()),
            )
        day_cursor = day_cursor + dt.timedelta(days=1)

    return events


class HistoricalKalshiProvider:
    """Replay-time Kalshi provider backed by pre-fetched historical candlesticks.

    Exposes the same interface as ``services.kalshi_provider.KalshiProvider``
    (and ``SimulatedKalshiProvider``), but all probabilities come from real
    Kalshi archival data aligned to ``ReplaySession.current_time()``.
    """

    def __init__(
        self,
        session: "ReplaySession",
        events: dict[str, _HistoricalKalshiEvent],
        *,
        basis_offset: float = 0.0,
        series: str = "KXINXU",
    ) -> None:
        self.session = session
        self._events = events
        self._events_sorted = sorted(events.values(), key=lambda e: e.open_utc_ts)
        self.basis_offset = float(basis_offset)
        self.series = series
        self.enabled = True
        self.is_healthy = True
        self._sentiment_history: deque[tuple[int, float, float]] = deque(maxlen=240)
        self._last_bar_key: Optional[int] = None
        self._last_bar_event: Optional[_HistoricalKalshiEvent] = None
        self._last_bar_markets: list[dict] = []
        # Per-event ES-SPX basis, locked at the first query inside the event.
        self._event_basis: dict[str, float] = {}

    # ---- helpers -----------------------------------------------------

    def _bar_key(self) -> int:
        return int(pd.Timestamp(self.session.current_time()).value)

    def _current_reference_price(self) -> float:
        return float(self.session.current_price())

    def _current_ts_utc_seconds(self) -> int:
        ts = self.session.current_time()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=NY_TZ)
        return int(ts.astimezone(dt.timezone.utc).timestamp())

    def _active_event(self) -> Optional[_HistoricalKalshiEvent]:
        ts = self._current_ts_utc_seconds()
        for evt in self._events_sorted:
            if evt.open_utc_ts <= ts < evt.close_utc_ts:
                return evt
        return None

    def _build_markets_for_bar(self) -> list[dict]:
        key = self._bar_key()
        if key == self._last_bar_key:
            return self._last_bar_markets
        evt = self._active_event()
        self._last_bar_event = evt
        if evt is None:
            self._last_bar_markets = []
            self._last_bar_key = key
            return self._last_bar_markets
        ts_utc = self._current_ts_utc_seconds()
        curve = evt.curve_at(ts_utc)
        basis = self._resolve_event_basis(evt, curve)
        markets: list[dict] = []
        for strike_spx, prob in curve:
            strike_es = float(strike_spx) + float(basis)
            markets.append(
                {
                    "strike": float(strike_spx),
                    "strike_spx": float(strike_spx),
                    "strike_es": float(strike_es),
                    "probability": round(float(prob), 4),
                    "status": "historical",
                }
            )
        self._last_bar_markets = markets
        self._last_bar_key = key
        return markets

    def _resolve_event_basis(
        self,
        evt: "_HistoricalKalshiEvent",
        curve: list[tuple[float, float]],
    ) -> float:
        """Lock the ES→SPX basis on first access within an event.

        Computed as ``current_ES - implied_SPX_50pct`` so the curve's
        at-the-money strike lines up with where ES is actually trading.
        """
        cached = self._event_basis.get(evt.event_ticker)
        if cached is not None:
            return cached
        if not curve:
            self._event_basis[evt.event_ticker] = float(self.basis_offset)
            return float(self.basis_offset)
        implied_spx = self._implied_level_spx_from_curve(curve)
        if implied_spx is None:
            self._event_basis[evt.event_ticker] = float(self.basis_offset)
            return float(self.basis_offset)
        es_now = self._current_reference_price()
        basis = float(es_now) - float(implied_spx)
        self._event_basis[evt.event_ticker] = basis
        logging.info(
            "Kalshi event %s basis locked: ES=%.2f SPX_50=%.2f basis=%.2f",
            evt.event_ticker,
            es_now,
            implied_spx,
            basis,
        )
        return basis

    @staticmethod
    def _implied_level_spx_from_curve(curve: list[tuple[float, float]]) -> Optional[float]:
        if len(curve) < 2:
            return None
        sorted_curve = sorted(curve, key=lambda r: r[0])
        for idx in range(len(sorted_curve) - 1):
            s_lo, p_lo = sorted_curve[idx]
            s_hi, p_hi = sorted_curve[idx + 1]
            # Kalshi "greater_or_equal" strikes: probability DECREASES as strike increases.
            if p_lo >= 0.5 > p_hi:
                if abs(p_hi - p_lo) < 1e-9:
                    return (s_lo + s_hi) / 2.0
                frac = (0.5 - p_lo) / (p_hi - p_lo)
                return s_lo + frac * (s_hi - s_lo)
        # No 0.5 crossing: curve is one-sided. If every strike is deep ITM or
        # deep OTM the center strike is outside the returned ladder — treat the
        # curve as uninformative so the caller leaves basis at the default.
        probs = [p for _, p in sorted_curve]
        if min(probs) > 0.95 or max(probs) < 0.05:
            return None
        return float(min(sorted_curve, key=lambda r: abs(r[1] - 0.5))[0])

    def _interpolated_probability(self, es_price: float) -> Optional[float]:
        markets = self._build_markets_for_bar()
        if not markets:
            return None
        below = [m for m in markets if float(m["strike_es"]) <= float(es_price)]
        above = [m for m in markets if float(m["strike_es"]) > float(es_price)]
        if below and above:
            lo = below[-1]
            hi = above[0]
            span = float(hi["strike_es"]) - float(lo["strike_es"])
            if span <= 0.0:
                return float(lo["probability"])
            frac = (float(es_price) - float(lo["strike_es"])) / span
            return float(lo["probability"]) * (1.0 - frac) + float(hi["probability"]) * frac
        if below:
            return float(below[-1]["probability"])
        if above:
            return float(above[0]["probability"])
        return None

    def _implied_level_es(self) -> Optional[float]:
        markets = self._build_markets_for_bar()
        if len(markets) < 2:
            return None
        for idx in range(len(markets) - 1):
            p_lo = float(markets[idx]["probability"])
            p_hi = float(markets[idx + 1]["probability"])
            if p_lo >= 0.5 > p_hi:
                s_lo = float(markets[idx]["strike_es"])
                s_hi = float(markets[idx + 1]["strike_es"])
                if abs(p_hi - p_lo) < 1e-9:
                    return (s_lo + s_hi) / 2.0
                frac = (0.5 - p_lo) / (p_hi - p_lo)
                return s_lo + frac * (s_hi - s_lo)
        return None

    # ---- public interface -------------------------------------------

    def _current_basis(self) -> float:
        evt = self._active_event()
        if evt is None:
            return float(self.basis_offset)
        cached = self._event_basis.get(evt.event_ticker)
        if cached is not None:
            return cached
        # Force a build so the basis gets locked.
        self._build_markets_for_bar()
        return self._event_basis.get(evt.event_ticker, float(self.basis_offset))

    def es_to_spx(self, es_price: float) -> float:
        return float(es_price) - self._current_basis()

    def spx_to_es(self, spx_price: float) -> float:
        return float(spx_price) + self._current_basis()

    def active_settlement_hour_et(
        self,
        ref_time: Optional[dt.datetime] = None,
        rollover_minute: int = 5,
    ) -> Optional[int]:
        # Use the currently-open event's settlement hour.  This matches the
        # semantic of the live provider: the hour you are actively trading
        # into.
        if ref_time is None:
            evt = self._active_event()
        else:
            ref = ref_time
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=NY_TZ)
            ts = int(ref.astimezone(dt.timezone.utc).timestamp())
            evt = next(
                (e for e in self._events_sorted if e.open_utc_ts <= ts < e.close_utc_ts),
                None,
            )
        return int(evt.settlement_hour_et) if evt is not None else None

    def get_relative_markets_for_ui(
        self,
        es_prices: Optional[list[float]] = None,
        window_size: int = 120,
    ) -> list[dict]:
        markets = self._build_markets_for_bar()
        if not markets:
            return []
        window_size = max(8, int(window_size))
        if len(markets) <= window_size:
            return [dict(row) for row in markets]
        reference = float(es_prices[0]) if es_prices else self._current_reference_price()
        nearest_idx = min(
            range(len(markets)),
            key=lambda i: abs(float(markets[i]["strike_es"]) - reference),
        )
        half = window_size // 2
        start = max(0, nearest_idx - half)
        end = min(len(markets), start + window_size)
        if (end - start) < window_size:
            start = max(0, end - window_size)
        return [dict(row) for row in markets[start:end]]

    def get_probability(self, strike_price: float) -> Optional[float]:
        return self._interpolated_probability(float(strike_price))

    def get_probability_curve(self) -> dict:
        return {
            float(row["strike_es"]): float(row["probability"])
            for row in self._build_markets_for_bar()
        }

    def get_nearest_market_for_es_price(self, es_price: float) -> Optional[dict]:
        markets = self._build_markets_for_bar()
        if not markets:
            return None
        nearest = min(markets, key=lambda row: abs(float(row["strike_es"]) - float(es_price)))
        return dict(nearest)

    def get_target_probability(self, es_price: float, side: Optional[str] = None) -> dict:
        market = self.get_nearest_market_for_es_price(es_price)
        payload = {
            "probability": None,
            "market_probability": None,
            "outcome_side": None,
            "strike_spx": None,
            "strike_es": None,
            "reference_spx": None,
            "reference_es": None,
            "distance_spx": None,
            "distance_es": None,
            "status": None,
            "result": None,
        }
        if market is None:
            return payload
        raw_probability = float(market.get("probability") or 0.0)
        normalized_side = str(side or "").strip().upper()
        outcome_side = "below" if normalized_side == "SHORT" else "above"
        probability = 1.0 - raw_probability if outcome_side == "below" else raw_probability
        payload.update(
            {
                "probability": round(float(probability), 4),
                "market_probability": round(raw_probability, 4),
                "outcome_side": outcome_side,
                "strike_spx": market.get("strike_spx"),
                "strike_es": market.get("strike_es"),
                "reference_spx": float(self.es_to_spx(es_price)),
                "reference_es": float(es_price),
                "distance_spx": round(float(market["strike_spx"]) - float(self.es_to_spx(es_price)), 2),
                "distance_es": round(float(market["strike_es"]) - float(es_price), 2),
                "status": market.get("status"),
                "result": None,
            }
        )
        return payload

    def _record_sentiment_history(self, es_price: float, probability: float) -> None:
        bar_key = self._bar_key()
        if self._sentiment_history and self._sentiment_history[-1][0] == bar_key:
            self._sentiment_history[-1] = (bar_key, float(es_price), float(probability))
            return
        self._sentiment_history.append((bar_key, float(es_price), float(probability)))

    def get_sentiment(self, es_price: float) -> dict:
        probability = self._interpolated_probability(float(es_price))
        implied_level = self._implied_level_es()
        payload = {
            "probability": None,
            "classification": "unavailable",
            "implied_level": None,
            "distance": None,
            "implied_level_es": None,
            "distance_es": None,
            "implied_level_spx": None,
            "distance_spx": None,
            "healthy": True,
        }
        if probability is None:
            return payload
        if probability >= 0.70:
            classification = "strong_bull"
        elif probability >= 0.55:
            classification = "bull"
        elif probability >= 0.45:
            classification = "neutral"
        elif probability >= 0.30:
            classification = "bear"
        else:
            classification = "strong_bear"
        distance_es = (
            float(implied_level) - float(es_price) if implied_level is not None else None
        )
        payload.update(
            {
                "probability": round(float(probability), 4),
                "classification": classification,
                "implied_level": round(float(implied_level), 2) if implied_level is not None else None,
                "distance": round(float(distance_es), 2) if distance_es is not None else None,
                "implied_level_es": round(float(implied_level), 2) if implied_level is not None else None,
                "distance_es": round(float(distance_es), 2) if distance_es is not None else None,
                "implied_level_spx": round(float(self.es_to_spx(implied_level)), 2) if implied_level is not None else None,
                "distance_spx": round(float(distance_es), 2) if distance_es is not None else None,
                "healthy": True,
            }
        )
        self._record_sentiment_history(float(es_price), float(probability))
        return payload

    def get_probability_gradient(self, es_price: float) -> Optional[float]:
        markets = self._build_markets_for_bar()
        if len(markets) < 3:
            return None
        below = [m for m in markets if float(m["strike_es"]) <= float(es_price)]
        above = [m for m in markets if float(m["strike_es"]) > float(es_price)]
        if not below or not above:
            return None
        lo = below[-1]
        hi = above[0]
        span = float(hi["strike_es"]) - float(lo["strike_es"])
        if span <= 0.0:
            return None
        gradient = (float(hi["probability"]) - float(lo["probability"])) / span
        return round(float(gradient), 6)

    def get_sentiment_momentum(self, es_price: float, lookback: int = 3) -> Optional[float]:
        probability = self._interpolated_probability(float(es_price))
        if probability is None:
            return None
        self._record_sentiment_history(float(es_price), float(probability))
        if len(self._sentiment_history) < int(lookback) + 1:
            return None
        prior_probability = float(self._sentiment_history[-(int(lookback) + 1)][2])
        return round(float(probability - prior_probability), 4)

    def get_implied_level(self) -> Optional[float]:
        es_level = self._implied_level_es()
        if es_level is None:
            return None
        return float(self.es_to_spx(es_level))

    def clear_cache(self) -> None:
        self._last_bar_key = None
        self._last_bar_markets = []
        self._last_bar_event = None


def _install_historical_kalshi_provider(
    *,
    session: "ReplaySession",
    live_module: Any,
    events: dict[str, _HistoricalKalshiEvent],
) -> HistoricalKalshiProvider:
    kalshi_cfg = CONFIG.get("KALSHI", {}) if isinstance(CONFIG, dict) else {}
    basis_offset = float(kalshi_cfg.get("basis_offset", 0.0) or 0.0)
    series = str(kalshi_cfg.get("series", "KXINXU") or "KXINXU")
    provider = HistoricalKalshiProvider(
        session, events, basis_offset=basis_offset, series=series
    )
    live_module._KALSHI_PROVIDER = provider
    live_module._KALSHI_PROVIDER_INIT_DONE = True
    total_candles = sum(sum(len(v) for v in e.strikes.values()) for e in events.values())
    logging.info(
        "Replay HistoricalKalshiProvider installed: events=%d total_candles=%d basis_offset=%.2f",
        len(events),
        total_candles,
        basis_offset,
    )
    return provider


def _install_kalshi_replay_provider(
    *,
    session: "ReplaySession",
    live_module: Any,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
) -> Any:
    """Prefer real pre-fetched Kalshi data; fall back to the synthesizer."""
    try:
        events = _fetch_kalshi_historical_window(start_dt=start_dt, end_dt=end_dt)
    except Exception as exc:
        logging.warning("Kalshi historical prefetch failed: %s — using simulator", exc)
        events = {}
    if events:
        return _install_historical_kalshi_provider(
            session=session, live_module=live_module, events=events
        )
    logging.info("No historical Kalshi events available — falling back to simulator")
    return _install_simulated_kalshi_provider(session=session, live_module=live_module)


def _install_aetherflow_replay_precompute(
    *,
    full_df: pd.DataFrame,
    replay_start: dt.datetime,
    replay_end: dt.datetime,
    live_module: Any,
) -> Any:
    original_loader = getattr(live_module, "_load_aetherflow_strategy_runtime", None)
    if not callable(original_loader):
        return None
    try:
        from aetherflow_strategy import AetherFlowStrategy as BaseAetherFlowStrategy
    except Exception as exc:
        logging.warning("Replay AetherFlow preload unavailable: %s", exc)
        return original_loader

    precomputed_df: Optional[pd.DataFrame] = None
    try:
        probe = BaseAetherFlowStrategy()
        candidate_df = probe.build_precomputed_backtest_df(full_df)
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
            precomputed_df = candidate_df.loc[
                (candidate_df.index >= replay_start) & (candidate_df.index <= replay_end)
            ].copy()
            logging.info(
                "Replay AetherFlow precompute built: rows=%d range=%s -> %s",
                len(precomputed_df),
                replay_start,
                replay_end,
            )
        else:
            logging.info("Replay AetherFlow precompute built: rows=0")
    except Exception as exc:
        logging.warning("Replay AetherFlow precompute failed: %s", exc)
        precomputed_df = None
    replay_start_ns = int(pd.Timestamp(replay_start).value)
    replay_end_ns = int(pd.Timestamp(replay_end).value)

    class ReplayAetherFlowStrategy(BaseAetherFlowStrategy):
        def __init__(self):
            super().__init__()
            if isinstance(precomputed_df, pd.DataFrame) and not precomputed_df.empty:
                self.set_precomputed_backtest_df(precomputed_df)
                logging.info("Replay AetherFlow precompute attached: rows=%d", len(precomputed_df))

        def on_bar(self, df, current_time=None) -> Optional[dict]:
            ts = current_time
            if ts is None and df is not None and not df.empty:
                try:
                    ts = pd.Timestamp(df.index[-1])
                except Exception:
                    ts = None
            if ts is not None:
                try:
                    ts_ns = int(pd.Timestamp(ts).value)
                except Exception:
                    ts_ns = None
                if ts_ns is not None and replay_start_ns <= ts_ns <= replay_end_ns:
                    cached = self._precomputed_lookup.get(ts_ns)
                    if cached is not None:
                        self.last_eval = dict(cached)
                        self._pending_runtime_event = None
                        return dict(cached)
                    self.last_eval = {"decision": "no_signal", "reason": "replay_precomputed_no_signal"}
                    self._pending_runtime_event = None
                    return None
            return super().on_bar(df, current_time=current_time)

    def _loader() -> Any:
        return ReplayAetherFlowStrategy

    live_module._load_aetherflow_strategy_runtime = _loader
    return original_loader


async def _run_replay(args: argparse.Namespace) -> tuple[Path, dict]:
    contract_root = str(args.contract_root or "MES").upper()
    start_time = _parse_user_time(args.start, is_end=False)
    end_time = _parse_user_time(args.end, is_end=True)
    if start_time > end_time:
        raise ValueError("Start must be before end.")

    live_client, bars_df = await _pull_projectx_bars(
        contract_root=contract_root,
        lookback_minutes=int(args.lookback_minutes),
        account_id=args.account_id,
    )
    account_id = int(live_client.account_id)
    full_df = bars_df.loc[bars_df.index <= end_time].copy()
    if full_df.empty:
        raise RuntimeError("Pulled ProjectX bars do not overlap the requested replay window.")

    report_root = Path(args.report_dir)
    if not report_root.is_absolute():
        report_root = ROOT / report_root
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(NY_TZ).strftime("%Y%m%d_%H%M%S")
    run_dir = report_root / f"live_loop_{contract_root}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    session = ReplaySession(
        full_df=full_df,
        replay_start=start_time,
        replay_end=end_time,
        initial_balance=float(args.initial_balance),
    )
    ReplayProjectXClient.configure(session)
    _prepare_simulation_env(run_dir, account_id, contract_root)

    import bot_state as bot_state_module
    import julie001 as live

    bot_state_module.STATE_PATH = run_dir / "bot_state.json"
    live.STATE_PATH = run_dir / "bot_state.json"
    live.ProjectXClient = ReplayProjectXClient
    live.heartbeat_task = _noop_background
    live.position_sync_task = _noop_background
    live.htf_structure_task = _noop_background
    live.kalshi_refresh_task = _noop_background
    live.sentiment_monitor_task = _noop_background
    _install_kalshi_replay_provider(
        session=session,
        live_module=live,
        start_dt=start_time,
        end_dt=end_time,
    )
    original_aetherflow_loader = _install_aetherflow_replay_precompute(
        full_df=full_df,
        replay_start=start_time,
        replay_end=end_time,
        live_module=live,
    )

    original_datetime_cls = _install_replay_clock(session, live)
    previous_trace, previous_threading_trace = _install_typeerror_trace(run_dir)
    original_sleep = asyncio.sleep

    async def _fast_sleep(delay: float = 0.0, result: Any = None):
        del delay
        await original_sleep(0)
        return result

    asyncio.sleep = _fast_sleep
    try:
        await live.run_bot()
    finally:
        asyncio.sleep = original_sleep
        if original_aetherflow_loader is not None:
            live._load_aetherflow_strategy_runtime = original_aetherflow_loader
        _restore_replay_clock(live, original_datetime_cls)
        _restore_typeerror_trace(previous_trace, previous_threading_trace)

    summary = session.summary()
    summary.update(
        {
            "mode": "full_live_loop_replay",
            "contract_root": contract_root,
            "target_symbol": determine_current_contract_symbol(contract_root),
            "account_id": account_id,
            "range_start": start_time.isoformat(),
            "range_end": end_time.isoformat(),
            "simulation_dir": str(run_dir),
            "log_markers": _collect_log_markers(run_dir / "topstep_live_bot.log"),
        }
    )
    summary_path = run_dir / "live_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    closed_path = run_dir / "closed_trades.json"
    closed_path.write_text(json.dumps(session.closed_trades, indent=2, default=str), encoding="utf-8")
    return run_dir, summary


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        run_dir, summary = asyncio.run(_run_replay(args))
    except KeyboardInterrupt:
        logging.warning("Replay interrupted.")
        return 130
    except Exception as exc:
        logging.error("Full live replay failed: %s", exc)
        return 1

    print("")
    print("Full Live Replay Summary")
    print(f"Simulation dir: {run_dir}")
    print(f"Bars processed: {summary.get('bars_processed')}")
    print(f"Closed trades: {summary.get('closed_trades')}")
    print(f"Wins: {summary.get('wins')}  Losses: {summary.get('losses')}  Winrate: {float(summary.get('winrate', 0.0)):.2f}%")
    print(f"Net PnL: ${float(summary.get('net_pnl', 0.0)):.2f}")
    print(f"Max drawdown: ${float(summary.get('max_drawdown', 0.0)):.2f}")
    print(f"Ending balance: ${float(summary.get('ending_balance', 0.0)):.2f}")
    print(f"Manual closes: {summary.get('manual_closes')} | Reversals: {summary.get('reversals')} | Stop updates: {summary.get('stop_updates')}")
    print(f"Passive exits: {summary.get('passive_exit_counts')}")
    print(f"Strategies: {summary.get('strategy_counts')}")
    print(f"Exit sources: {summary.get('exit_source_counts')}")
    print(f"Log markers: {summary.get('log_markers')}")
    print(f"Summary JSON: {run_dir / 'live_replay_summary.json'}")
    print(f"Closed trades JSON: {run_dir / 'closed_trades.json'}")
    print("")
    print("Note: OANDA mirror is excluded. Kalshi live-path code is exercised, but its historical crowd state is not reconstructed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

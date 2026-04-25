from __future__ import annotations

import datetime as dt
from typing import Any, Optional
from zoneinfo import ZoneInfo


NY_TZ = ZoneInfo("America/New_York")


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_side(value: Any) -> Optional[str]:
    side = str(value or "").strip().upper()
    if side in {"LONG", "BUY"}:
        return "LONG"
    if side in {"SHORT", "SELL"}:
        return "SHORT"
    return None


def _coerce_time(value: Any, *, default_tz: ZoneInfo) -> Optional[dt.datetime]:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz)
    return parsed.astimezone(default_tz)


def de3_trade_close_hit_stop(row: Optional[dict[str, Any]]) -> bool:
    if not isinstance(row, dict):
        return False
    order_id = str(row.get("order_id") or row.get("close_order_id") or "").strip()
    stop_order_id = str(row.get("stop_order_id") or "").strip()
    close_source = str(row.get("source") or row.get("close_source") or "").strip().lower()
    close_reason = str(row.get("de3_management_close_reason") or "").strip().lower()
    exit_reason = str(row.get("exit_reason") or "").strip().lower()
    if order_id and stop_order_id and order_id == stop_order_id:
        return True
    if exit_reason in {"stop", "stop_gap"}:
        return True
    if close_reason.endswith("_stop") or close_reason == "initial_stop_loss":
        return True
    return any(
        token in close_source
        for token in ("confirmed stop fill", "crossed stop", "stop fill", "stop_loss")
    )


def de3_flip_flop_guard_reason(
    signal_payload: Optional[dict[str, Any]],
    active_trades_payload: Optional[list[dict[str, Any]]],
    recent_closed_trades_payload: Optional[list[dict[str, Any]]],
    current_time: dt.datetime,
    *,
    cfg: Optional[dict[str, Any]] = None,
    default_tz: ZoneInfo = NY_TZ,
) -> Optional[str]:
    if not isinstance(signal_payload, dict):
        return None
    strategy_name = str(signal_payload.get("strategy") or "").strip()
    if not strategy_name.startswith("DynamicEngine3"):
        return None

    guard_cfg = cfg if isinstance(cfg, dict) else {}
    if not bool(guard_cfg.get("enabled", False)):
        return None

    side_name = _normalize_side(signal_payload.get("side"))
    if side_name not in {"LONG", "SHORT"}:
        return None

    active_trades = [
        trade for trade in (active_trades_payload or [])
        if isinstance(trade, dict)
        and str(trade.get("strategy") or "").strip().startswith("DynamicEngine3")
    ]

    vol_regime = str(
        signal_payload.get("vol_regime")
        or (active_trades[0].get("vol_regime") if active_trades else "")
        or ""
    ).strip().lower()
    allowed_regimes = {
        str(value).strip().lower()
        for value in (guard_cfg.get("apply_vol_regimes") or [])
        if str(value).strip()
    }
    if allowed_regimes and vol_regime not in allowed_regimes:
        return None

    current_time = _coerce_time(current_time, default_tz=default_tz)
    if current_time is None:
        current_time = dt.datetime.now(default_tz)

    if bool(guard_cfg.get("block_countertrend_reversal_in_trend_day", True)) and active_trades:
        trend_day_dir = str(
            signal_payload.get("trend_day_dir")
            or active_trades[0].get("trend_day_dir")
            or ""
        ).strip().lower()
        if trend_day_dir in {"up", "down"}:
            has_opposite_de3_trade = any(
                _normalize_side(trade.get("side")) not in {None, side_name}
                for trade in active_trades
            )
            if has_opposite_de3_trade:
                if (trend_day_dir == "up" and side_name == "SHORT") or (
                    trend_day_dir == "down" and side_name == "LONG"
                ):
                    return (
                        f"DE3 anti-flip blocked counter-trend reversal in {vol_regime or 'active'} vol "
                        f"(trend_day={trend_day_dir}, signal={side_name})"
                    )

    stop_reentry_cooldown_bars = max(
        0,
        _coerce_int(guard_cfg.get("stop_reentry_cooldown_bars"), 0),
    )
    if stop_reentry_cooldown_bars <= 0:
        return None

    same_sub_only = bool(guard_cfg.get("stop_reentry_same_sub_strategy_only", True))
    signal_sub_strategy = str(
        signal_payload.get("sub_strategy")
        or signal_payload.get("combo_key")
        or ""
    ).strip()
    bar_seconds = max(1, _coerce_int(guard_cfg.get("bar_seconds"), 60))

    for row in reversed(recent_closed_trades_payload or []):
        if not isinstance(row, dict):
            continue
        if not str(row.get("strategy") or "").strip().startswith("DynamicEngine3"):
            continue
        if _normalize_side(row.get("side")) != side_name:
            continue
        row_sub_strategy = str(
            row.get("sub_strategy")
            or row.get("combo_key")
            or ""
        ).strip()
        if same_sub_only and signal_sub_strategy and row_sub_strategy and row_sub_strategy != signal_sub_strategy:
            continue
        if not de3_trade_close_hit_stop(row):
            continue
        close_time = _coerce_time(
            row.get("time") or row.get("close_time") or row.get("exit_time"),
            default_tz=default_tz,
        )
        if close_time is None:
            continue
        delta_seconds = (current_time - close_time).total_seconds()
        if delta_seconds < 0:
            continue
        bars_since_stop = int(delta_seconds // float(bar_seconds))
        if bars_since_stop < stop_reentry_cooldown_bars:
            target_label = row_sub_strategy or signal_sub_strategy or "DE3 setup"
            return (
                f"DE3 stop cooldown active for {target_label} "
                f"({bars_since_stop + 1}/{stop_reentry_cooldown_bars} bars since stop)"
            )
        break

    return None

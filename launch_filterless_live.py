import asyncio
import atexit
import json
import os
import platform
import socket
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo


def _force_utf8_stdio() -> None:
    """
    Reconfigure stdio early so emoji-heavy logging does not crash under cp1252.

    This repo emits symbols like checkmarks and warning icons during import time,
    before runtime startup is complete. On some Windows machines, especially after
    migrating a workspace between PCs, redirected stdio can default to cp1252.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


def _stabilize_windows_platform_queries() -> None:
    """
    Avoid Python 3.13 Windows WMI probes during large library imports.

    On this machine, `platform.uname()` / `platform.system()` / `platform.machine()`
    can block inside `platform._wmi_query()`, which delays or stalls filterless bot
    startup before `run_bot()` begins.
    """
    if os.name != "nt":
        return

    arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or ""
    ).strip()
    if not arch:
        return

    normalized = {
        "AMD64": "x86_64",
        "X86": "x86",
        "ARM64": "arm64",
    }.get(arch.upper(), arch)

    winver = sys.getwindowsversion()
    release = "11" if winver.major >= 10 and winver.build >= 22000 else str(winver.major)
    version = f"{winver.major}.{winver.minor}.{winver.build}"
    uname_result = platform.uname_result(
        "Windows",
        socket.gethostname(),
        release,
        version,
        normalized,
    )

    platform.uname = lambda: uname_result
    platform.system = lambda: uname_result.system
    platform.machine = lambda: uname_result.machine
    platform.release = lambda: uname_result.release
    platform.version = lambda: uname_result.version
    platform.processor = lambda: normalized
    platform.win32_ver = lambda release="", version="", csd="", ptype="": (
        uname_result.release,
        uname_result.version,
        "",
        "",
    )


_force_utf8_stdio()
_stabilize_windows_platform_queries()

os.environ.setdefault("JULIE_FILTERLESS_ONLY", "1")
os.environ.setdefault("JULIE_DISABLE_STRATEGY_FILTERS", "1")

# --- iter-11 risk layer (regime-adaptive CB + LossFactorGuard) ---
# Validated on Oct 2025 (+$1,246 swing vs v4) and Apr 2026 Wk1+Wk2 (+$1,086).
# See commit 5d682cd. Any of these can be overridden via shell env.
os.environ.setdefault("JULIE_CB", "1")
os.environ.setdefault("JULIE_DLB", "1")
os.environ.setdefault("JULIE_DD_SCALE", "1")
os.environ.setdefault("JULIE_KALSHI_CONTINUATION_TP", "1")
os.environ.setdefault("JULIE_REGIME_CLASSIFIER", "1")
os.environ.setdefault("JULIE_REGIME_ADAPTIVE_CB", "1")
os.environ.setdefault("JULIE_LOSS_FACTOR_GUARD", "1")
os.environ.setdefault("JULIE_CB_MAX_DAILY_LOSS", "350")
os.environ.setdefault("JULIE_CB_MAX_CONSEC_LOSSES", "5")
# Filter A (trailing-DD CB) was validated DISABLED on 27 outrageous 2025 days:
# it never reduced DD violations (peak-to-trough damage already realized by the
# time CB trips) and cost ~$1,800 by killing late-day chop recoveries + the
# Apr 9 tariff-pause rally. Keep the code but default off (0 = disabled).
# Re-enable via shell export if you want to experiment with higher thresholds.
os.environ.setdefault("JULIE_CB_MAX_TRAILING_DD", "0")
# Filters A/C/D/E/F ARCHIVED in favor of filter G (ML signal gate):
# OOS on April 2026 (195 trades never seen in training):
#   Filter stack C+D+E+F:   baseline $+484 → ~$+675  (Δ ~+$191)
#   Filter G alone:         baseline $+484 → $+2,109 (Δ +$1,625)
# G is a separate joblib classifier at artifacts/signal_gate_2025/model.joblib
# that predicts P(pnl <= -$100) per trade and vetoes at P >= 0.35. Trained on
# 1669 iter-11 2025 trades. Revert by setting JULIE_SIGNAL_GATE_2025=0 and
# re-enabling the rules via the env vars below.
#
# Rules kept disabled as code paths (not deleted) so rollback is one env flip:
os.environ.setdefault("JULIE_LFG_TREND_BIAS_MIN_TIER", "99")   # filter C OFF (99 = never fires)
os.environ.setdefault("JULIE_REGIME_SIZE_CAP", "0")            # filter D OFF
os.environ.setdefault("JULIE_REGIME_SIZE_CAP_VALUE", "1")
os.environ.setdefault("JULIE_REGIME_GREEN_UNLOCK_PNL", "999999")  # filter E OFF (unreachable threshold)
os.environ.setdefault("JULIE_REGIME_GREEN_UNLOCK_SIZE", "3")
os.environ.setdefault("JULIE_LFG_CHART_VETO", "0")             # filter F OFF
#
# Filter G ON (primary veto layer for 2025+ regime):
os.environ.setdefault("JULIE_SIGNAL_GATE_2025", "1")

# --- ML overlay stack ACTIVATION DEFAULTS ---
#
# All five ML overlay models load automatically at bot startup regardless of
# these env vars. The env vars only control whether the ML's verdict
# OVERRIDES the rule's verdict when they disagree. Default in the shadow
# module is OFF (shadow / log-only). Here we flip all five to ON so the
# launcher activates ML steering out-of-the-box.
#
# Each is an `os.environ.setdefault`, so you can opt any individual layer
# back OUT via a shell export before running the launcher:
#     export JULIE_ML_KALSHI_TP_ACTIVE=0   # this layer stays shadow-only
#
# Layers (see README section 11):
#   model_lfo.joblib          — WAIT vs IMMEDIATE fill decision
#   model_pct_overlay.joblib  — breakout vs pivot bias on %-level touches
#   model_pivot_trail.joblib  — hold/skip SL ratchet on confirmed pivots
#   model_kalshi_gate.joblib  — entry-side Kalshi pass/block (clf + reg)
#   model_kalshi_tp_gate.joblib — TP-aligned Kalshi pass/block (reg on pnl$)
#
os.environ.setdefault("JULIE_ML_LFO_ACTIVE", "1")
os.environ.setdefault("JULIE_ML_PCT_ACTIVE", "1")
os.environ.setdefault("JULIE_ML_PIVOT_TRAIL_ACTIVE", "1")
os.environ.setdefault("JULIE_ML_KALSHI_ACTIVE", "1")
os.environ.setdefault("JULIE_ML_KALSHI_TP_ACTIVE", "1")
# Threshold for the Kalshi TP-gate regressor. Default: block when predicted
# pnl <= $0. Tune tighter (e.g. "10") to require predicted profit margin, or
# looser (e.g. "-10") to pass borderline trades. No retraining required.
os.environ.setdefault("JULIE_ML_KALSHI_TP_PNL_THR", "0")

# --- ML overlay: RL trade-management policy (Path 3) ---
# PPO policy over post-entry trade management. Canonical is now the
# v3 SL-only variant (4 actions: HOLD, MOVE_SL_TO_BE, TIGHTEN_SL_25,
# TIGHTEN_SL_50), trained with extended obs (encoder + cross-market).
# Every action the v3 policy can emit is wired in _apply_rl_management_action,
# so live mode is safe. Validation stats: mean=$30.15/trade, WR=98%,
# total +$128k over 4261 episodes (vs $10/trade, WR 50% under prior
# "v2 firing 7 actions but executor defers partial/reverse" setup).
#
# The full 7-action v2 policy is preserved at
#   artifacts/signal_gate_2025/model_rl_management_v2_for_future_partial_close.zip
# for later promotion once client.close_trade_leg_partial() and the
# bracket-resize logic are wired (the "Option B" path — partial-close
# captures a theoretical ~$296/trade ceiling but needs broker work).
#
# Set this to "0" to fall back to shadow-only ([SHADOW_RL] log lines);
# set to "1" (default) for live steering.
os.environ.setdefault("JULIE_ML_RL_MGMT_ACTIVE", "1")

# --- Kalshi CM-breakout gate v2 (direction-specific ML, AUC 0.77) ---
# Two GBT classifiers (LONG / SHORT) trained on 125,678 aligned
# MES+MNQ+VIX 1-min bars with a direct price-direction target
# ("does MES move ≥5 pts in 30 min in side direction?"). Rolling-origin
# AUC 0.776 LONG / 0.764 SHORT; p≥0.60 bucket hits 71–74% vs 31% base.
# When active, v2 replaces both the hand-tuned VIX/MNQ rule and the
# v1 CM gate for Kalshi override decisions. Guarded by
# entry_support_score ≥ 0.15 (never overrides catastrophic blocks) and
# only fires in non-background roles. Set to "0" to demote to
# shadow-only ([CM_GATE_V2] log lines still emit for observation).
os.environ.setdefault("JULIE_KALSHI_CM_GATE_V2_ACTIVE", "1")

os.environ.setdefault("JULIE_REGIME_CB_WHIPSAW", "250")
os.environ.setdefault("JULIE_REGIME_CB_NEUTRAL", "350")
os.environ.setdefault("JULIE_REGIME_CB_CALM", "500")
os.environ.setdefault("JULIE_REGIME_CONSEC_WHIPSAW", "4")
os.environ.setdefault("JULIE_REGIME_CONSEC_NEUTRAL", "5")
os.environ.setdefault("JULIE_REGIME_CONSEC_CALM", "7")
os.environ.setdefault("JULIE_LFG_MORNING_CASCADE", "99")
os.environ.setdefault("JULIE_LFG_LONG_STREAK", "5")
os.environ.setdefault("JULIE_LFG_SHORT_STREAK", "6")
os.environ.setdefault("JULIE_LFG_AFT_SHUTDOWN_PNL", "-250")

from bot_state import load_bot_state
from config import CONFIG
from config_secrets import SECRETS
from julie001 import run_bot
from process_singleton import acquire_singleton_lock
from services.kalshi_provider import KalshiProvider
from tools.filterless_dashboard_bridge import DEFAULT_KALSHI_SNAPSHOT_PATH, write_json_atomic


LIVE_LOCK_PATH = Path(__file__).resolve().parent / "logs" / "filterless_live.lock"
LIVE_LOCK = acquire_singleton_lock(LIVE_LOCK_PATH, name="filterless_live_bot")
if LIVE_LOCK is None:
    existing = ""
    try:
        existing = LIVE_LOCK_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    print(
        "Another filterless live bot instance is already running. "
        f"Lock: {LIVE_LOCK_PATH}"
    )
    if existing:
        print(existing)
    raise SystemExit(0)


NY_TZ = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
BRIDGE_SCRIPT = ROOT / "tools" / "filterless_dashboard_bridge.py"
BRIDGE_LOG_PATH = ROOT / "logs" / "dashboard_bridge.log"
BRIDGE_PROCESS: Optional[subprocess.Popen[Any]] = None
BRIDGE_LOG_HANDLE = None


def _load_live_position_from_state() -> Optional[Dict[str, Any]]:
    state = load_bot_state(ROOT / "bot_state.json")
    if not isinstance(state, dict):
        return None
    live_position = state.get("live_position")
    return live_position if isinstance(live_position, dict) else None


def _coerce_price_from_live_position(live_position: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(live_position, dict):
        return None
    for key in ("current_price", "avg_price", "entry_price"):
        value = live_position.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_target_price_from_live_position(live_position: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(live_position, dict):
        return None
    for key in ("target_price", "tp_price"):
        value = live_position.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_side_from_live_position(live_position: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(live_position, dict):
        return None
    side = str(live_position.get("side") or "").strip().upper()
    return side if side in {"LONG", "SHORT"} else None


def _build_kalshi_provider() -> Optional[KalshiProvider]:
    kalshi_cfg = CONFIG.get("KALSHI", {}) if isinstance(CONFIG, dict) else {}
    if not isinstance(kalshi_cfg, dict):
        return None
    provider_cfg = dict(kalshi_cfg)
    provider_cfg["key_id"] = str(SECRETS.get("KALSHI_KEY_ID", provider_cfg.get("key_id", "")) or "")
    provider_cfg["private_key_path"] = str(
        SECRETS.get("KALSHI_PRIVATE_KEY_PATH", provider_cfg.get("private_key_path", "")) or ""
    )
    provider = KalshiProvider(provider_cfg)
    return provider


def _kalshi_disabled_payload() -> Dict[str, Any]:
    return {
        "enabled": False,
        "healthy": False,
        "updated_at": datetime.now(NY_TZ).isoformat(),
        "basis_offset": 0.0,
        "probability_60m": None,
        "probability_reference_kind": None,
        "probability_reference_side": None,
        "probability_reference_es_price": None,
        "probability_contract_es_price": None,
        "probability_contract_spx_price": None,
        "probability_contract_probability": None,
        "probability_contract_outcome": None,
        "probability_contract_distance_es": None,
        "event_ticker": None,
        "es_reference_price": None,
        "spx_reference_price": None,
        "strikes": [],
    }


# KXINXU settlement hours are Eastern Time (per CFTC filing).
# Contracts settle at each hour 10 AM - 4 PM ET during market hours.
_KALSHI_SETTLEMENT_HOURS_ET = [10, 11, 12, 13, 14, 15, 16]
# Trade gating (3x sizing) only activates 12-4 PM ET where crowd has 70% accuracy.
# 10-11 AM excluded: backtest shows crowd is contrarian (39.5% accuracy).
_KALSHI_GATING_HOURS_ET = [12, 13, 14, 15, 16]


def _kalshi_trade_gating_status(provider: Optional[KalshiProvider] = None) -> tuple[bool, Optional[int]]:
    """Determine if trade gating should be active for the active settlement contract.

    Trade gating (3x sizing) activates for contracts settling 12-4 PM ET
    where backtest shows 70% crowd directional accuracy at 60%+ confidence.
    10-11 AM settlement contracts are observe-only because the crowd is
    contrarian (39.5% accuracy).
    """
    settlement_hour = None
    if provider is not None:
        try:
            settlement_hour = provider.active_settlement_hour_et()
        except Exception:
            settlement_hour = None
    if settlement_hour in _KALSHI_GATING_HOURS_ET:
        return True, settlement_hour
    return False, None


async def _kalshi_snapshot_loop(path: Path, interval_seconds: float = 10.0) -> None:
    provider = _build_kalshi_provider()
    last_et_date = None
    backfill_done = False

    while True:
        if provider is None or not getattr(provider, "enabled", False):
            write_json_atomic(path, _kalshi_disabled_payload())
            await asyncio.sleep(interval_seconds)
            continue

        now_et = datetime.now(NY_TZ)
        current_et_date = now_et.date()

        # Daily rollover: clear cache when the ET date changes
        if last_et_date is not None and current_et_date != last_et_date:
            provider.clear_cache()
            backfill_done = False
        last_et_date = current_et_date

        # Historical backfill on startup: fetch all of today's contracts
        # so the ML has data even for contracts whose trading windows
        # have already closed before the bot started.
        daily_contracts: list = []
        if not backfill_done:
            try:
                daily_contracts = provider.fetch_daily_contracts()
                backfill_done = True
            except Exception:
                pass

        live_position = _load_live_position_from_state()
        price = _coerce_price_from_live_position(live_position)
        target_price = _coerce_target_price_from_live_position(live_position)
        target_side = _coerce_side_from_live_position(live_position)
        sentiment = provider.get_sentiment(price) if price is not None else {}
        target_probability = (
            provider.get_target_probability(target_price, target_side)
            if target_price is not None
            else {}
        )
        target_probability_value = target_probability.get("probability")
        ui_reference_prices = [ref for ref in (price, target_price) if ref is not None]
        strikes = provider.get_relative_markets_for_ui(ui_reference_prices, window_size=30)

        active_settlement_hour = provider.active_settlement_hour_et() if provider is not None else None
        trade_gating_active, trade_gating_hour = _kalshi_trade_gating_status(provider)

        # When trade gating is active, Kalshi influences trade decisions
        # with 3x sizing; otherwise it is observe-only for the dashboard.
        if trade_gating_active:
            observer_only = False
            status_label = "Trade gating (3x)"
            status_reason = (
                f"Kalshi is actively gating trades with 3x sizing for the "
                f"{trade_gating_hour}:00 ET settlement window (70% accuracy at 60%+ confidence)."
            )
        elif active_settlement_hour in (10, 11):
            observer_only = True
            status_label = "Observe only (morning)"
            status_reason = (
                f"Kalshi data is live but not gating trades for the "
                f"{active_settlement_hour}:00 ET settlement window. "
                "Backtest shows crowd is contrarian in morning hours (39.5% accuracy). "
                "Trade gating activates at 12 PM ET."
            )
        else:
            observer_only = True
            status_label = "Observe only"
            status_reason = "Kalshi details are live on the dashboard without changing trade gating."

        payload: Dict[str, Any] = {
            "enabled": True,
            "requested": True,
            "configured": True,
            "observer_only": observer_only,
            "status_label": status_label,
            "status_reason": status_reason,
            "source": "kalshi_snapshot_loop",
            "healthy": bool(getattr(provider, "is_healthy", False)),
            "updated_at": datetime.now(NY_TZ).isoformat(),
            "basis_offset": float(getattr(provider, "basis_offset", 0.0) or 0.0),
            "probability_60m": (
                target_probability_value
                if target_probability_value is not None
                else sentiment.get("probability")
            ),
            "probability_reference_kind": (
                "open_position_target"
                if target_probability_value is not None
                else "current_price"
            ),
            "probability_reference_side": target_side if target_probability_value is not None else None,
            "probability_reference_es_price": (
                target_probability.get("reference_es")
                if target_probability_value is not None
                else (float(price) if price is not None else None)
            ),
            "probability_contract_es_price": target_probability.get("strike_es"),
            "probability_contract_spx_price": target_probability.get("strike_spx"),
            "probability_contract_probability": target_probability.get("market_probability"),
            "probability_contract_outcome": target_probability.get("outcome_side"),
            "probability_contract_distance_es": target_probability.get("distance_es"),
            "event_ticker": getattr(provider, "last_resolved_ticker", None) or provider._current_event_ticker(),  # noqa: SLF001
            "es_reference_price": float(price) if price is not None else None,
            "spx_reference_price": (float(price) - float(provider.basis_offset)) if price is not None else None,
            "trade_gating_active": trade_gating_active,
            "trade_gating_hour": trade_gating_hour,
            "strikes": strikes if isinstance(strikes, list) else [],
            "daily_contracts": daily_contracts if daily_contracts else None,
        }
        write_json_atomic(path, payload)
        await asyncio.sleep(interval_seconds)


def _start_bridge_process(path: Path) -> None:
    global BRIDGE_PROCESS, BRIDGE_LOG_HANDLE
    if BRIDGE_PROCESS is not None and BRIDGE_PROCESS.poll() is None:
        return

    BRIDGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_LOG_HANDLE = BRIDGE_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    cmd = [
        sys.executable,
        str(BRIDGE_SCRIPT),
        "--follow",
        "--kalshi-snapshot-path",
        str(path),
    ]
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        BRIDGE_PROCESS = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=BRIDGE_LOG_HANDLE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
    else:
        BRIDGE_PROCESS = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=BRIDGE_LOG_HANDLE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def _cleanup_bridge_process() -> None:
    global BRIDGE_PROCESS, BRIDGE_LOG_HANDLE
    process = BRIDGE_PROCESS
    if process is not None and process.poll() is None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
    if BRIDGE_LOG_HANDLE is not None:
        try:
            BRIDGE_LOG_HANDLE.flush()
            BRIDGE_LOG_HANDLE.close()
        except Exception:
            pass
    BRIDGE_PROCESS = None
    BRIDGE_LOG_HANDLE = None


async def _run_all() -> None:
    kalshi_snapshot_path = DEFAULT_KALSHI_SNAPSHOT_PATH
    _start_bridge_process(kalshi_snapshot_path)
    tasks = [
        asyncio.create_task(_kalshi_snapshot_loop(kalshi_snapshot_path)),
        asyncio.create_task(run_bot()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exc = task.exception()
        if exc is not None:
            raise exc


if __name__ == "__main__":
    atexit.register(_cleanup_bridge_process)
    asyncio.run(_run_all())

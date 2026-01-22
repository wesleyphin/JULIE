"""
Centralized Event Logger for JULIE Trading Bot
Provides structured logging with timestamps for all trading events
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
import json


class EventLogger:
    """Centralized event logger with structured logging and timestamps"""

    def __init__(self, logger_name: str = "JULIE"):
        self.logger = logging.getLogger(logger_name)

    def _format_timestamp(self) -> str:
        """Generate timestamp in Eastern Time"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def _log_event(self, level: str, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Log a structured event with timestamp

        Args:
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            event_type: Type of event (STRATEGY, REJECTION, TRADE, FILTER, etc.)
            message: Main message
            details: Optional dictionary of additional details
        """
        timestamp = self._format_timestamp()
        log_msg = f"[{timestamp}] [{event_type}] {message}"

        if details:
            # Format details as key=value pairs
            detail_str = " | ".join([f"{k}={v}" for k, v in details.items()])
            log_msg += f" | {detail_str}"

        if level == "INFO":
            self.logger.info(log_msg)
        elif level == "WARNING":
            self.logger.warning(log_msg)
        elif level == "ERROR":
            self.logger.error(log_msg)
        elif level == "DEBUG":
            self.logger.debug(log_msg)

    # ==================== STRATEGY EVENTS ====================

    def log_strategy_signal(self, strategy_name: str, side: str, tp_dist: float, sl_dist: float,
                           price: float, additional_info: Optional[Dict] = None):
        """Log when a strategy generates a signal"""
        details = {
            "strategy": strategy_name,
            "side": side,
            "price": f"{price:.2f}",
            "tp_dist": f"{tp_dist:.2f}",
            "sl_dist": f"{sl_dist:.2f}"
        }
        if additional_info:
            details.update(additional_info)

        self._log_event("INFO", "STRATEGY_SIGNAL",
                       f"📊 {strategy_name} generated {side} signal",
                       details)

    def log_strategy_no_signal(self, strategy_name: str, reason: Optional[str] = None):
        """Log when a strategy produces no signal"""
        details = {"strategy": strategy_name}
        if reason:
            details["reason"] = reason

        self._log_event("DEBUG", "STRATEGY_NO_SIGNAL",
                       f"Strategy {strategy_name} - No signal",
                       details)

    def log_strategy_execution(self, strategy_name: str, execution_type: str):
        """Log strategy execution type (FAST, STANDARD, QUEUED)"""
        self._log_event("INFO", "STRATEGY_EXEC",
                       f"⚡ {execution_type} execution for {strategy_name}",
                       {"strategy": strategy_name, "exec_type": execution_type})

    # ==================== REJECTION/FILTER EVENTS ====================

    def log_rejection_detected(self, rejection_type: str, direction: str, level: float,
                               current_price: float, additional_info: Optional[Dict] = None):
        """Log when a rejection is detected"""
        details = {
            "rejection_type": rejection_type,
            "direction": direction,
            "level": f"{level:.2f}",
            "current_price": f"{current_price:.2f}"
        }
        if additional_info:
            details.update(additional_info)

        self._log_event("INFO", "REJECTION_DETECTED",
                       f"🔄 {rejection_type} rejection detected - {direction} bias",
                       details)

    def log_rejection_block(self, filter_name: str, signal_side: str, reason: str,
                           additional_info: Optional[Dict] = None):
        """Log when a trade is blocked by rejection/filter"""
        details = {
            "filter": filter_name,
            "blocked_side": signal_side,
            "reason": reason
        }
        if additional_info:
            details.update(additional_info)

        self._log_event("WARNING", "REJECTION_BLOCK",
                       f"⛔ {filter_name} blocked {signal_side} trade",
                       details)

    def log_rejection_cleared(self, rejection_type: str, reason: str):
        """Log when a rejection bias is cleared"""
        self._log_event("INFO", "REJECTION_CLEARED",
                       f"✅ {rejection_type} rejection cleared - {reason}",
                       {"rejection_type": rejection_type, "reason": reason})

    def log_filter_check(self, filter_name: str, signal_side: str, passed: bool,
                        reason: Optional[str] = None, additional_info: Optional[Dict] = None,
                        strategy: Optional[str] = None, metrics: Optional[Dict] = None):
        """Log filter check results with forensic metrics"""
        details = {
            "filter": filter_name,
            "side": signal_side,
            "passed": passed
        }
        if strategy:
            details["strategy"] = strategy
        if reason:
            details["reason"] = reason
        if additional_info:
            details.update(additional_info)

        status = "✓ PASS" if passed else "✗ BLOCK"
        msg = f"{status} - {filter_name} for {signal_side}"

        # Add forensic metrics to message if provided
        if metrics:
            metric_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
            msg += f" | {metric_str}"
            details.update(metrics)

        self._log_event("INFO" if passed else "WARNING", "FILTER_CHECK",
                       msg,
                       details)

    # ==================== TRADE EVENTS ====================

    def log_trade_signal_generated(self, strategy: str, side: str, price: float,
                                   tp_dist: float, sl_dist: float):
        """Log when a valid trade signal passes all filters"""
        details = {
            "strategy": strategy,
            "side": side,
            "price": f"{price:.2f}",
            "tp_dist": f"{tp_dist:.2f}",
            "sl_dist": f"{sl_dist:.2f}"
        }
        self._log_event("INFO", "TRADE_SIGNAL",
                       f"🎯 Valid trade signal: {strategy} {side}",
                       details)

    def log_trade_order_placed(self, order_id: str, side: str, price: float,
                              tp_price: float, sl_price: float, strategy: str):
        """Log when an order is placed"""
        details = {
            "order_id": order_id,
            "strategy": strategy,
            "side": side,
            "entry": f"{price:.2f}",
            "tp": f"{tp_price:.2f}",
            "sl": f"{sl_price:.2f}"
        }
        self._log_event("INFO", "TRADE_PLACED",
                       f"🚀 Order placed: {side} @ {price:.2f}",
                       details)

    def log_trade_order_rejected(self, side: str, price: float, error_msg: str, strategy: str):
        """Log when an order is rejected by the broker"""
        details = {
            "strategy": strategy,
            "side": side,
            "price": f"{price:.2f}",
            "error": error_msg
        }
        self._log_event("ERROR", "TRADE_REJECTED",
                       f"❌ Order rejected: {side} @ {price:.2f}",
                       details)

    def log_trade_closed(self, side: str, entry_price: float, exit_price: float,
                        pnl: float, reason: str, strategy: Optional[str] = None):
        """Log when a trade is closed"""
        details = {
            "side": side,
            "entry": f"{entry_price:.2f}",
            "exit": f"{exit_price:.2f}",
            "pnl": f"{pnl:.2f}",
            "reason": reason
        }
        if strategy:
            details["strategy"] = strategy

        pnl_emoji = "💰" if pnl > 0 else "💸"
        self._log_event("INFO", "TRADE_CLOSED",
                       f"{pnl_emoji} Trade closed: {reason}",
                       details)

    def log_trade_modified(self, modification_type: str, old_value: float, new_value: float,
                          reason: str):
        """Log when a trade is modified (SL/TP adjustment, breakeven, etc.)"""
        details = {
            "modification": modification_type,
            "old_value": f"{old_value:.2f}",
            "new_value": f"{new_value:.2f}",
            "reason": reason
        }
        self._log_event("INFO", "TRADE_MODIFIED",
                       f"🔧 Trade modified: {modification_type}",
                       details)

    def log_close_and_reverse(self, old_side: str, new_side: str, price: float, strategy: str):
        """Log when closing a position and reversing"""
        details = {
            "old_side": old_side,
            "new_side": new_side,
            "price": f"{price:.2f}",
            "strategy": strategy
        }
        self._log_event("INFO", "TRADE_REVERSE",
                       f"🔄 Close & Reverse: {old_side} → {new_side}",
                       details)

    # ==================== POSITION MANAGEMENT EVENTS ====================

    def log_position_update(self, side: Optional[str], size: float, avg_price: float,
                           unrealized_pnl: Optional[float] = None):
        """Log position state updates"""
        details = {
            "side": side or "FLAT",
            "size": size,
            "avg_price": f"{avg_price:.2f}"
        }
        if unrealized_pnl is not None:
            details["unrealized_pnl"] = f"{unrealized_pnl:.2f}"

        self._log_event("INFO", "POSITION_UPDATE",
                       f"📍 Position: {side or 'FLAT'} {size} @ {avg_price:.2f}",
                       details)

    def log_breakeven_adjustment(self, old_sl: float, new_sl: float, current_price: float,
                                profit_points: float):
        """Log when stop is moved to breakeven"""
        details = {
            "old_sl": f"{old_sl:.2f}",
            "new_sl": f"{new_sl:.2f}",
            "current_price": f"{current_price:.2f}",
            "profit_points": f"{profit_points:.2f}"
        }
        self._log_event("INFO", "BREAKEVEN_ADJUST",
                       f"🛡️ Stop moved to breakeven",
                       details)

    def log_early_exit(self, reason: str, bars_held: int, current_price: float,
                      entry_price: float):
        """Log early exit triggers"""
        details = {
            "reason": reason,
            "bars_held": bars_held,
            "current_price": f"{current_price:.2f}",
            "entry_price": f"{entry_price:.2f}"
        }
        self._log_event("WARNING", "EARLY_EXIT",
                       f"⏰ Early exit triggered: {reason}",
                       details)

    # ==================== SYSTEM EVENTS ====================

    def log_system_event(self, event_type: str, message: str, details: Optional[Dict] = None):
        """Log general system events"""
        self._log_event("INFO", "SYSTEM", message, details)

    def log_error(self, error_type: str, message: str, exception: Optional[Exception] = None):
        """Log errors"""
        details = {"error_type": error_type}
        if exception:
            details["exception"] = str(exception)

        self._log_event("ERROR", "ERROR", message, details)

    def log_market_data(self, timestamp: str, price: float, additional_info: Optional[Dict] = None):
        """Log market data updates"""
        details = {"timestamp": timestamp, "price": f"{price:.2f}"}
        if additional_info:
            details.update(additional_info)

        self._log_event("DEBUG", "MARKET_DATA",
                       f"📈 Bar: {timestamp} | Price: {price:.2f}",
                       details)

    # ==================== 4. DEEP STRATEGY INTROSPECTION ====================
    # UI Destination: "LIVE EVENT LOG" (with high detail)

    def log_strategy_metric(self, strategy: str, metric_name: str, value: Any, threshold: Any, decision: str):
        """
        Log internal strategy logic.
        Ex: "ML Confidence: 0.82 (Threshold: 0.65) -> BUY"
        """
        icon = "🟢" if decision in ["BUY", "SELL", "GO", "PASS"] else "🔴"

        self._log_event("INFO", "STRATEGY_METRIC",
            f"{icon} {strategy} Logic: {metric_name} = {value} vs {threshold}",
            details={"decision": decision}
        )

    # ==================== 5. RISK & ACCOUNT HEALTH ====================
    # UI Destination: "MARKET CONTEXT" or "LIVE EVENT LOG"

    def log_risk_telemetry(self, current_loss: float, limit: float, daily_pnl: float):
        """
        Log live risk utilization.
        Ex: "Circuit Breaker: 45% used ($270/$600)"
        """
        usage_pct = (current_loss / limit) * 100 if limit > 0 else 0
        icon = "🛡️"
        if usage_pct > 80:
            icon = "🚨"
        elif usage_pct > 50:
            icon = "⚠️"

        # UI Keyword: "RISK"
        self._log_event("INFO", "RISK_TELEMETRY",
            f"{icon} RISK MONITOR: Daily Loss ${current_loss:.2f} / ${limit:.2f} ({usage_pct:.1f}%)",
            details={"Daily PnL": f"${daily_pnl:.2f}"}
        )

    # ==================== 6. SYSTEM HEALTH (LATENCY) ====================
    # UI Destination: "LIVE EVENT LOG" (Debug level mostly)

    def log_latency(self, component: str, duration_ms: float):
        """Log execution speed to catch lag"""
        # Only log if slow (>200ms) to reduce noise
        if duration_ms > 200:
            self._log_event("WARNING", "SYSTEM_LAG",
                f"🐢 SLOW EXECUTION: {component} took {duration_ms:.0f}ms",
                details={"threshold": "200ms"}
            )

    # ==================== 7. CONTINUATION STRATEGY EVENTS ====================
    # UI Destination: "CONTINUATION STRATEGY LOGS" section

    def log_continuation_bias_block(self, strategy: str, side: str, bias: str):
        """
        Log when a trade is blocked by bias and rescue is being attempted.
        Ex: "🛑 BIAS BLOCK: RegimeAdaptive (LONG) blocked by SHORT bias. Attempting Rescue..."
        """
        self._log_event("INFO", "CONTINUATION_BIAS_BLOCK",
            f"🛑 BIAS BLOCK: {strategy} ({side}) blocked by {bias} bias. Attempting Rescue...",
            details={
                "strategy": strategy,
                "blocked_side": side,
                "active_bias": bias
            }
        )

    def log_continuation_rescue_success(self, original_strategy: str, rescue_key: str, bias: str):
        """
        Log when a continuation strategy successfully rescues a blocked trade.
        Ex: "🚑 RESCUE SUCCESSFUL: Continuation_Q4_W45_D7_Asia confirms LONG bias!"
        """
        self._log_event("INFO", "CONTINUATION_RESCUE",
            f"🚑 RESCUE SUCCESSFUL: {rescue_key} confirms {bias} bias!",
            details={
                "original_strategy": original_strategy,
                "rescue_strategy": rescue_key,
                "bias": bias
            }
        )

    def log_continuation_rescue_failed(self, strategy: str, side: str):
        """
        Log when no matching continuation setup is found.
        Ex: "❌ RESCUE FAILED: No matching continuation setup found."
        """
        self._log_event("INFO", "CONTINUATION_NO_MATCH",
            f"❌ RESCUE FAILED: No matching continuation setup found.",
            details={
                "blocked_strategy": strategy,
                "blocked_side": side
            }
        )

    def log_continuation_rescue_blocked(self, original_strategy: str, rescue_key: str, bias: str, reason: str):
        """
        Log when a rescue flip occurs but the rescued signal is later blocked.
        Ex: "❌ RESCUE FAILED: Continuation_Q1_W4_D4_NY blocked (ChopFilter: IN_CHOP)"
        """
        self._log_event("INFO", "CONTINUATION_RESCUE",
            f"❌ RESCUE FAILED: {rescue_key} blocked ({reason})",
            details={
                "original_strategy": original_strategy,
                "rescue_strategy": rescue_key,
                "bias": bias,
                "reason": reason
            }
        )

    def log_continuation_window_active(self, key: str, session: str, quarter: int, week: int, day: int):
        """
        Log when the current time matches an active continuation window.
        Ex: "📅 CONTINUATION WINDOW: Q4_W45_D7_Asia is active"
        """
        self._log_event("INFO", "CONTINUATION_WINDOW",
            f"📅 CONTINUATION WINDOW: {key} is active",
            details={
                "key": key,
                "session": session,
                "quarter": quarter,
                "week": week,
                "day": day
            }
        )

    def log_continuation_signal(self, key: str, side: str, tp_dist: float, sl_dist: float):
        """
        Log when a continuation strategy generates a signal.
        Ex: "📈 CONTINUATION SIGNAL: Continuation_Q4_W45_D7_Asia LONG"
        """
        icon = "📈" if side == "LONG" else "📉"
        self._log_event("INFO", "CONTINUATION_SIGNAL",
            f"{icon} CONTINUATION SIGNAL: {key} {side}",
            details={
                "strategy": key,
                "side": side,
                "tp_dist": f"{tp_dist:.2f}",
                "sl_dist": f"{sl_dist:.2f}"
            }
        )


# Create a global instance for easy importing
event_logger = EventLogger()

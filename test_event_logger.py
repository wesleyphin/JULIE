"""
Test script for event_logger module
Verifies that all logging functions work correctly with timestamps
"""

import logging

# Set up logging to see output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

from event_logger import event_logger

print("=" * 80)
print("TESTING EVENT LOGGER - All events should have timestamps")
print("=" * 80)

# Test Strategy Events
print("\n--- Testing Strategy Events ---")
event_logger.log_strategy_signal(
    strategy_name="RegimeAdaptive",
    side="LONG",
    tp_dist=6.5,
    sl_dist=4.0,
    price=5850.25,
    additional_info={"session": "NY_AM", "quarter": 2}
)

event_logger.log_strategy_execution("ConfluenceStrategy", "FAST")

event_logger.log_strategy_no_signal("MLPhysics", "No valid session model")

# Test Rejection Events
print("\n--- Testing Rejection Events ---")
event_logger.log_rejection_detected(
    rejection_type="Midnight_ORB",
    direction="LONG",
    level=5848.50,
    current_price=5850.25,
    additional_info={"quarter": 1, "level_type": "LOW"}
)

event_logger.log_rejection_block(
    filter_name="RejectionFilter",
    signal_side="SELL",
    reason="Prev Day PM bias LONG blocks SELL",
    additional_info={"prev_day_pm_bias": "LONG"}
)

event_logger.log_rejection_cleared(
    rejection_type="Midnight_ORB",
    reason="Continuation breakout"
)

# Test Filter Events
print("\n--- Testing Filter Events ---")
event_logger.log_filter_check("HTF_FVG", "LONG", True)
event_logger.log_filter_check("ChopFilter", "SHORT", False, "ADX below threshold")
event_logger.log_filter_check("ExtensionFilter", "LONG", True)
event_logger.log_filter_check("VolatilityFilter", "SHORT", False, "Low volatility regime")

# Test Trade Events
print("\n--- Testing Trade Events ---")
event_logger.log_trade_signal_generated(
    strategy="RegimeAdaptive",
    side="LONG",
    price=5850.25,
    tp_dist=6.5,
    sl_dist=4.0
)

event_logger.log_trade_order_placed(
    order_id="abc123-def456",
    side="LONG",
    price=5850.25,
    tp_price=5856.75,
    sl_price=5846.25,
    strategy="RegimeAdaptive"
)

event_logger.log_trade_order_rejected(
    side="SHORT",
    price=5850.25,
    error_msg="Insufficient margin",
    strategy="ConfluenceStrategy"
)

event_logger.log_close_and_reverse(
    old_side="SHORT",
    new_side="LONG",
    price=5850.25,
    strategy="RegimeAdaptive"
)

event_logger.log_trade_closed(
    side="LONG",
    entry_price=5850.25,
    exit_price=5856.75,
    pnl=6.50,
    reason="Take Profit Hit",
    strategy="RegimeAdaptive"
)

event_logger.log_trade_modified(
    modification_type="StopLoss_to_Breakeven",
    old_value=5846.25,
    new_value=5850.50,
    reason="40% of TP reached"
)

# Test Position Management Events
print("\n--- Testing Position Management Events ---")
event_logger.log_position_update(
    side="LONG",
    size=1,
    avg_price=5850.25,
    unrealized_pnl=3.50
)

event_logger.log_breakeven_adjustment(
    old_sl=5846.25,
    new_sl=5850.50,
    current_price=5853.00,
    profit_points=2.75
)

event_logger.log_early_exit(
    reason="not green after 50 bars",
    bars_held=52,
    current_price=5848.00,
    entry_price=5850.25
)

# Test System Events
print("\n--- Testing System Events ---")
event_logger.log_system_event(
    event_type="INITIALIZATION",
    message="Bot started successfully",
    details={"strategies": 8, "filters": 6}
)

event_logger.log_error(
    error_type="API_ERROR",
    message="Rate limit exceeded",
    exception=Exception("429 Too Many Requests")
)

event_logger.log_market_data(
    timestamp="2025-12-13 10:30:00",
    price=5850.25,
    additional_info={"volume": 1250, "session": "NY_AM"}
)

print("\n" + "=" * 80)
print("TEST COMPLETE - All events logged with timestamps")
print("=" * 80)

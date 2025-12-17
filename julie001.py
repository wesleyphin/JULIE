import requests
import pandas as pd
import numpy as np
import datetime
import time
import logging
from zoneinfo import ZoneInfo
from datetime import timezone as dt_timezone
import uuid
from typing import Dict, Optional, List, Tuple

from config import CONFIG, refresh_target_symbol, determine_current_contract_symbol
from dynamic_sltp_params import dynamic_sltp_engine, get_sltp
from volatility_filter import volatility_filter, check_volatility, VolRegime
from regime_strategy import RegimeAdaptiveStrategy
from htf_fvg_filter import HTFFVGFilter
from rejection_filter import RejectionFilter
from chop_filter import ChopFilter
from extension_filter import ExtensionFilter
from trend_filter import TrendFilter
from dynamic_structure_blocker import DynamicStructureBlocker
from bank_level_quarter_filter import BankLevelQuarterFilter
from memory_sr_filter import MemorySRFilter
from orb_strategy import OrbStrategy
from intraday_dip_strategy import IntradayDipStrategy
from confluence_strategy import ConfluenceStrategy
from smt_strategy import SMTStrategy
from dynamic_chop import DynamicChopAnalyzer
from ict_model_strategy import ICTModelStrategy
from ml_physics_strategy import MLPhysicsStrategy
from dynamic_engine_strategy import DynamicEngineStrategy
from event_logger import event_logger
from circuit_breaker import CircuitBreaker
from news_filter import NewsFilter
from directional_loss_blocker import DirectionalLossBlocker
from impulse_filter import ImpulseFilter
from client import ProjectXClient
from risk_engine import OptimizedTPEngine
from gemini_optimizer import GeminiSessionOptimizer

# ==========================================
# RESAMPLER HELPER FUNCTION
# ==========================================
def resample_dataframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Resamples 1-minute OHLCV data into higher timeframes (5m, 15m, 60m).
    """
    if df.empty:
        return pd.DataFrame()

    # Define aggregation rules
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample using the timeframe string (e.g., '5min' for 5 minutes)
    tf_code = f"{timeframe_minutes}min"
    resampled_df = df.resample(tf_code).agg(agg_dict).dropna()

    return resampled_df


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("topstep_live_bot.log"), logging.StreamHandler()],
    force=True  # Override any pre-existing logging config
)

NY_TZ = ZoneInfo('America/New_York')

# ==========================================
# 2a. REJECTION FILTER (Trade Direction Filters)
# ==========================================
# Implementation moved to rejection_filter.py to keep this entrypoint focused on
# bot orchestration.

# ==========================================
# 2d. HTF FVG REJECTION
# ==========================================
try:
    from htf_fvg_filter import HTFFVGFilter
    logging.info("✅ HTFFVGFilter module loaded")
except ImportError as e:
    logging.error(f"❌ Failed to import htf_fvg_filter.py: {e}")
    # Dummy class to prevent crash if file missing
    class HTFFVGFilter:
        def check_signal_blocked(self, *args): return False, None



# OptimizedTPEngine moved to risk_engine.py



# ProjectXClient moved to client.py

# ==========================================
# 12. MAIN EXECUTION LOOP
# ==========================================
def run_bot():
    refresh_target_symbol()
    print("=" * 60)
    print("PROJECTX GATEWAY - MES FUTURES BOT (LIVE)")
    print("--- Julie Pro (Session Specialized) ---")
    print("--- DYNAMIC SL/TP ENGINE ENABLED ---")
    print(f"REST API: {CONFIG['REST_BASE_URL']}")
    print(f"Target Symbol: {CONFIG['TARGET_SYMBOL']}")
    print("=" * 60)
    
    client = ProjectXClient()
    
    # Step 1: Authenticate
    try:
        client.login()
    except Exception as e:
        print(f"CRITICAL: Failed to login. Check credentials. Error: {e}")
        return

    # Step 2: Fetch Account ID
    print("\n📋 Fetching account information...")
    account_id = client.fetch_accounts()
    if account_id is None:
        print("CRITICAL: Could not retrieve account ID")
        return
    
    # Step 3: Fetch Contract ID
    print("\n📋 Fetching available contracts...")
    contract_id = client.fetch_contracts()
    if contract_id is None:
        print("CRITICAL: Could not retrieve contract ID")
        return
    
    print(f"\n✅ Setup complete:")
    print(f"   Account ID: {client.account_id}")
    print(f"   Contract ID: {client.contract_id}")

    # Secondary client for MNQ data (SMT divergence inputs)
    mnq_target_symbol = determine_current_contract_symbol(
        "MNQ", tz_name=CONFIG.get("TIMEZONE", "US/Eastern")
    )
    mnq_client = ProjectXClient(contract_root="MNQ", target_symbol=mnq_target_symbol)

    try:
        mnq_client.login()
        mnq_client.account_id = client.account_id or mnq_client.fetch_accounts()
        mnq_client.fetch_contracts()
    except Exception as e:
        logging.error(f"❌ Failed to initialize MNQ data client: {e}")
        return

    # Initialize all strategies

    # Dynamic chop analyzer (tiered thresholds with LTF breakout override)
    chop_analyzer = DynamicChopAnalyzer(client)
    chop_analyzer.calibrate()
    last_chop_calibration = time.time()
    
    # HIGH PRIORITY - Execute immediately on signal
    fast_strategies = [
        RegimeAdaptiveStrategy(),
        IntradayDipStrategy(),
    ]
    
    # STANDARD PRIORITY - Normal execution
    ml_strategy = MLPhysicsStrategy()
    dynamic_engine_strat = DynamicEngineStrategy()
    smt_strategy = SMTStrategy()

    standard_strategies = [
        ConfluenceStrategy(),
        dynamic_engine_strat,
        smt_strategy,
    ]
    
    # Only add ML strategy if at least one model loaded successfully
    if ml_strategy.model_loaded:
        standard_strategies.append(ml_strategy)
    else:
        print(f"⚠️ MLPhysicsStrategy disabled - no session model files found")
    
    # LOW PRIORITY / LOOSE EXECUTION - Wait for next bar
    loose_strategies = [
        OrbStrategy(),
        ICTModelStrategy(),
    ]
    
    # Initialize filters
    rejection_filter = RejectionFilter()
    bank_filter = BankLevelQuarterFilter()
    chop_filter = ChopFilter(lookback=20)
    extension_filter = ExtensionFilter()
    trend_filter = TrendFilter()
    htf_fvg_filter = HTFFVGFilter() # Now uses Memory-Based Class
    structure_blocker = DynamicStructureBlocker(lookback=20)
    memory_sr = MemorySRFilter(lookback_bars=300, zone_width=2.0, touch_threshold=2)
    news_filter = NewsFilter()
    circuit_breaker = CircuitBreaker(max_daily_loss=600, max_consecutive_losses=7)
    directional_loss_blocker = DirectionalLossBlocker(consecutive_loss_limit=3, block_minutes=15)
    impulse_filter = ImpulseFilter(lookback=20, impulse_multiplier=2.5)

    # Initialize Gemini Session Optimizer
    optimizer = GeminiSessionOptimizer()
    last_processed_session = None

    print("\nActive Strategies:")
    print("  [FAST EXECUTION]")
    for strat in fast_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [STANDARD EXECUTION]")
    for strat in standard_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [LOOSE EXECUTION]")
    for strat in loose_strategies: print(f"    • {strat.__class__.__name__}")
    
    print("\nListening for market data (polling every 2 seconds)...")
    
    # === TRACKING VARIABLES ===
    last_htf_fetch_time = 0
    last_position_sync_time = 0
    POSITION_SYNC_INTERVAL = 30  # Sync every 30 seconds
    
    # Track pending signals for delayed execution
    pending_loose_signals = {}
    last_processed_bar = None
    opposite_signal_count = 0
    
    # Early Exit Tracking
    active_trade = None
    bar_count = 0
    
    # Token refresh
    last_token_check = time.time()
    TOKEN_CHECK_INTERVAL = 3600

    # Chop state tracking (only log when state changes)
    last_chop_reason = None

    # === STEP 1: INITIAL DATA LOAD (MAX HISTORY) ===
    logging.info("⏳ Startup: Fetching full 20,000 bar history (MES)...")
    # Fetch the maximum allowed history ONCE before the loop starts
    master_df = client.get_market_data(lookback_minutes=20000, force_fetch=True)

    logging.info("⏳ Startup: Fetching full 20,000 bar history (MNQ)...")
    master_mnq_df = mnq_client.get_market_data(lookback_minutes=20000, force_fetch=True)

    if master_df.empty:
        logging.warning("⚠️ Startup fetch returned empty data (MES). Bot will attempt to build history in loop.")
        master_df = pd.DataFrame()

    # --- 10/10 UPGRADE: DYNAMIC VOLATILITY CALIBRATION ---
    # Use the 20,000 bars (approx 2 weeks) to recalibrate the Volatility Map
    # This ensures "High Volatility" means "High relative to TODAY", not 2024.
    if not master_df.empty:
        try:
            volatility_filter.calibrate(master_df)
        except Exception as e:
            logging.error(f"❌ Calibration Failed: {e} (Continuing with static thresholds)")
    # --- END CALIBRATION ---

    if master_mnq_df.empty:
        logging.warning("⚠️ Startup fetch returned empty data (MNQ). Bot will attempt to build history in loop.")
        master_mnq_df = pd.DataFrame()

    # One-time backfill flag
    data_backfilled = False

    while True:
        try:
            # Periodic token validation
            if time.time() - last_token_check > TOKEN_CHECK_INTERVAL:
                client.validate_session()
                last_token_check = time.time()

            # Periodic chop threshold recalibration (default every 4 hours)
            if chop_analyzer.should_recalibrate(last_chop_calibration):
                chop_analyzer.calibrate()
                last_chop_calibration = time.time()

            # === GLOBAL RISK & NEWS FILTERS ===
            cb_blocked, cb_reason = circuit_breaker.should_block_trade()
            if cb_blocked:
                logging.info(f"🚫 Circuit Breaker Block: {cb_reason}")
                time.sleep(60)
                continue

            current_time = datetime.datetime.now(dt_timezone.utc)
            news_blocked, news_reason = news_filter.should_block_trade(current_time)
            if news_blocked:
                logging.info(f"🚫 NEWS WAIT: {news_reason}")
                time.sleep(10)
                continue

            # === SESSION DETECTION & GEMINI OPTIMIZATION ===
            current_time_et = datetime.datetime.now(NY_TZ)
            hour = current_time_et.hour

            if 18 <= hour or hour < 3:
                current_session_name = "ASIA"
            elif 3 <= hour < 8:
                current_session_name = "LONDON"
            elif 8 <= hour < 12:
                current_session_name = "NY_AM"
            elif 12 <= hour < 17:
                current_session_name = "NY_PM"
            else:
                current_session_name = "POST_MARKET"

            # --- OPTIMIZATION TRIGGER ---
            if current_session_name != last_processed_session:
                logging.info(f"🔄 SESSION HANDOVER: {last_processed_session} -> {current_session_name}")

                if CONFIG.get('GEMINI', {}).get('enabled', False):
                    print(f"\n🧠 OPTIMIZING FOR {current_session_name} SESSION...")

                    # 1. Fetch Events
                    try:
                        raw_events = news_filter.fetch_news()
                        events_str = str(raw_events)
                    except Exception as e:
                        events_str = "Events data unavailable."

                    # 2. Get Hardcoded Base Params for Session
                    session_cfg = CONFIG['SESSIONS'].get(current_session_name, {})
                    base_sl = session_cfg.get('SL', 4.0)
                    base_tp = session_cfg.get('TP', 8.0)

                    # 3. Call Gemini
                    opt_result = optimizer.optimize_new_session(
                        master_df,
                        current_session_name,
                        events_str,
                        base_sl,
                        base_tp
                    )

                    if opt_result:
                        sl_mult = float(opt_result.get('sl_multiplier', 1.0))
                        tp_mult = float(opt_result.get('tp_multiplier', 1.0))
                        reason = opt_result.get('reasoning', '')

                        # 4. Update Global Config
                        CONFIG['DYNAMIC_SL_MULTIPLIER'] = sl_mult
                        CONFIG['DYNAMIC_TP_MULTIPLIER'] = tp_mult

                        print(f"🎯 NEW MULTIPLIERS | SL: {sl_mult}x | TP: {tp_mult}x")
                        print(f"📝 Reasoning: {reason}")
                    else:
                        CONFIG['DYNAMIC_SL_MULTIPLIER'] = 1.0
                        CONFIG['DYNAMIC_TP_MULTIPLIER'] = 1.0

                last_processed_session = current_session_name

            # === STEP 2: INCREMENTAL UPDATE (SEQUENTIAL FETCH) ===
            # Fetch MES first, then MNQ immediately after to keep timestamps close
            recent_data = client.get_market_data(lookback_minutes=15, force_fetch=True)
            recent_mnq_data = mnq_client.get_market_data(lookback_minutes=15, force_fetch=True)

            if not recent_data.empty:
                # Append new data to our master history
                master_df = pd.concat([master_df, recent_data])

                # Remove duplicates based on timestamp (keep the newest version of the bar)
                master_df = master_df[~master_df.index.duplicated(keep='last')]

                # Optional: Keep memory safe (limit to 50k bars - deeper than API allows!)
                if len(master_df) > 50000:
                    master_df = master_df.iloc[-50000:]

            if not recent_mnq_data.empty:
                master_mnq_df = pd.concat([master_mnq_df, recent_mnq_data])
                master_mnq_df = master_mnq_df[~master_mnq_df.index.duplicated(keep='last')]
                if len(master_mnq_df) > 50000:
                    master_mnq_df = master_mnq_df.iloc[-50000:]

            # Make sure we have data before proceeding
            if master_df.empty or master_mnq_df.empty:
                # Early heartbeat - shows bot is alive even when no data available
                if not hasattr(client, '_empty_data_counter'):
                    client._empty_data_counter = 0
                client._empty_data_counter += 1
                if client._empty_data_counter % 30 == 0:
                    print(f"⏳ Waiting for data: {datetime.datetime.now().strftime('%H:%M:%S')} | No bars received (market may be closed or starting up)")
                    logging.info(f"No market data available - attempt #{client._empty_data_counter}")
                time.sleep(2)
                continue

            # Use master_df for all calculations now
            # This variable now holds 20k+ bars of history
            new_df = master_df

            # === LOCAL RESAMPLING ENGINE ===
            # Resample from our locally maintained deep history
            df_5m = resample_dataframe(new_df, 5)
            df_15m = resample_dataframe(new_df, 15)
            df_60m = resample_dataframe(new_df, 60)

            # === ONE-TIME BACKFILL ===
            if not data_backfilled:
                logging.info("🔄 Performing one-time backfill of filter state from history...")
                # Replay the history we just fetched
                # This restores Midnight ORB, Prev Session, etc. instantly
                rejection_filter.backfill(new_df)

                # Also backfill bank_filter (has same update() signature)
                for ts, row in new_df.sort_index().iterrows():
                    bank_filter.update(ts, row['high'], row['low'], row['close'])

                data_backfilled = True
                logging.info("✅ State restored from history.")

            # === DYNAMIC CHOP CHECK (Pass Local DFs) ===
            # We pass the locally generated df_60m so the analyzer can use it for breakout shift logic
            is_choppy, chop_reason = chop_analyzer.check_market_state(new_df, df_60m_current=df_60m)

            # Initialize Directional Restrictions
            allowed_chop_side = None  # None means ALL allowed (unless blocked)

            # Parse the new "Fade" reasons (only log when state changes)
            if "ALLOW_LONG_ONLY" in chop_reason:
                allowed_chop_side = "LONG"
                if chop_reason != last_chop_reason:
                    logging.info(f"⚠️ CHOP RESTRICTION: {chop_reason}")
                    last_chop_reason = chop_reason

            elif "ALLOW_SHORT_ONLY" in chop_reason:
                allowed_chop_side = "SHORT"
                if chop_reason != last_chop_reason:
                    logging.info(f"⚠️ CHOP RESTRICTION: {chop_reason}")
                    last_chop_reason = chop_reason

            elif is_choppy:
                if chop_reason != last_chop_reason:
                    logging.info(f"⛔ TRADE BLOCKED: {chop_reason}")
                    last_chop_reason = chop_reason
                time.sleep(2)
                continue
            else:
                # Clear chop state if no restriction active
                if last_chop_reason is not None:
                    logging.info("✅ CHOP RESTRICTION CLEARED")
                    last_chop_reason = None

            # Heartbeat
            current_price = new_df.iloc[-1]['close']
            current_time = new_df.index[-1]
            now_ts = time.time()
            
            if not hasattr(client, '_heartbeat_counter'):
                client._heartbeat_counter = 0
            client._heartbeat_counter += 1
            if client._heartbeat_counter % 30 == 0:
                print(f"💓 Heartbeat: {datetime.datetime.now().strftime('%H:%M:%S')} | Price: {current_price:.2f}")

            # ==========================================
            # HEARTBEAT POSITION SYNC
            # ==========================================
            if now_ts - last_position_sync_time > POSITION_SYNC_INTERVAL:
                try:
                    # 1. Force fetch from Broker
                    broker_pos = client.get_position()

                    # 2. Update Shadow State
                    client._local_position = broker_pos.copy()

                    # 3. Check for DRIFT (Critical Safety Check)
                    if active_trade is not None:
                        # Scenario: Bot thinks we are Long, Broker says Flat
                        if broker_pos['size'] == 0:
                            logging.warning(f"⚠️ STATE DRIFT: Bot has {active_trade['side']} trade, but Broker is FLAT.")
                            logging.warning("   -> Forcing local trade closure to prevent errors.")

                            # Log the ghost close so analytics stay clean
                            event_logger.log_trade_closed(
                                side=active_trade['side'],
                                entry_price=active_trade['entry_price'],
                                exit_price=current_price, # Best guess
                                pnl=0.0,
                                reason="State Drift / Broker Liquidation"
                            )

                            # Clean up local state
                            active_trade = None
                            opposite_signal_count = 0
                            client._active_stop_order_id = None

                        # Scenario: Bot thinks Long, Broker is Short (Rare, but bad)
                        elif broker_pos['side'] != active_trade['side']:
                            logging.critical("🚨 CRITICAL DRIFT: Bot/Broker side mismatch! Stopping bot for safety.")
                            break

                    last_position_sync_time = now_ts

                except Exception as e:
                    logging.error(f"❌ Heartbeat Sync Failed: {e}")

            # === UPDATE HTF FVG MEMORY (THROTTLED) ===
            # Only update memory once every 60 seconds to save API calls.
            # The filter will use this memory for checks every second.
            # === UPDATE HTF FVG MEMORY (THROTTLED) ===
            # Only update memory once every 60 seconds to save API calls.
            if now_ts - last_htf_fetch_time > 60:
                try:
                    # Use PRINT here so you definitely see it
                    print(f"\n⏳ [{datetime.datetime.now().strftime('%H:%M:%S')}] Updating HTF FVG Memory (1H/4H)...")
                    
                    # Fetch new data
                    df_1h_new = client.fetch_custom_bars(lookback_bars=240, minutes_per_bar=60)
                    df_4h_new = client.fetch_custom_bars(lookback_bars=200, minutes_per_bar=240)
                    
                    # Update Memory
                    if not df_1h_new.empty and not df_4h_new.empty:
                        htf_fvg_filter.check_signal_blocked('CHECK', current_price, df_1h_new, df_4h_new)
                        print(f"   ✅ Memory Updated. Active Structures: {len(htf_fvg_filter.memory)}")
                        last_htf_fetch_time = now_ts
                    else:
                        print("   ⚠️ HTF Data Fetch returned empty.")
                except Exception as e:
                    logging.error(f"Error updating HTF memory: {e}")
            
            # Initial Startup Fetch (if empty)
            if (not htf_fvg_filter.memory) and client._check_general_rate_limit() and (now_ts - last_htf_fetch_time > 60):
                 logging.info("🚀 Startup: Fetching initial HTF data...")
                 try:
                     df_1h_new = client.fetch_custom_bars(lookback_bars=240, minutes_per_bar=60)
                     df_4h_new = client.fetch_custom_bars(lookback_bars=200, minutes_per_bar=240)
                     htf_fvg_filter.check_signal_blocked('CHECK', current_price, df_1h_new, df_4h_new)
                     last_htf_fetch_time = now_ts
                 except Exception as e:
                     logging.error(f"Startup HTF fetch failed: {e}")
            
            # === TRAILING STOP / BREAK-EVEN CHECK (EVERY TICK) ===
            if active_trade is not None:
                be_config = CONFIG.get('BREAK_EVEN', {})
                if be_config.get('enabled', False):
                    tp_dist = active_trade.get('tp_dist', 6.0)
                    entry_price = active_trade['entry_price']
                    trigger_pct = be_config.get('trigger_pct', 0.40)
                    trail_pct = be_config.get('trail_pct', 0.25)  # Lock in 25% of profit above trigger

                    if active_trade['side'] == 'LONG':
                        current_profit = current_price - entry_price
                    else:
                        current_profit = entry_price - current_price

                    profit_threshold = tp_dist * trigger_pct

                    # Only act if we're above the initial trigger threshold
                    if current_profit >= profit_threshold:
                        buffer = be_config.get('buffer_ticks', 1) * 0.25

                        # Calculate trailing stop price based on current profit
                        # Start at break-even, then trail as profit increases
                        if not active_trade.get('break_even_triggered', False):
                            # First trigger: move to break-even
                            new_stop_price = entry_price + buffer if active_trade['side'] == 'LONG' else entry_price - buffer
                            logging.info(f"🔒 BREAK-EVEN TRIGGER: Profit {current_profit:.2f} >= {profit_threshold:.2f}")
                        else:
                            # Trailing: lock in trail_pct of profit above entry
                            # e.g., if profit is 4.0 pts and trail_pct is 0.5, lock in 2.0 pts
                            trail_amount = current_profit * trail_pct
                            if active_trade['side'] == 'LONG':
                                new_stop_price = entry_price + trail_amount
                            else:
                                new_stop_price = entry_price - trail_amount
                            # Round to nearest tick (0.25)
                            new_stop_price = round(new_stop_price * 4) / 4

                        # Get current stop level to avoid unnecessary modifications
                        current_stop = active_trade.get('current_stop_price', 0)

                        # Only modify if new stop is better than current stop
                        should_modify = False
                        if active_trade['side'] == 'LONG':
                            if new_stop_price > current_stop + 0.25:  # At least 1 tick better
                                should_modify = True
                        else:  # SHORT
                            if new_stop_price < current_stop - 0.25:
                                should_modify = True

                        if should_modify:
                            known_size = active_trade.get('size', 1)
                            stop_order_id = active_trade.get('stop_order_id')

                            if client.modify_stop_to_breakeven(new_stop_price, active_trade['side'], known_size, stop_order_id):
                                old_stop = active_trade.get('current_stop_price', entry_price - tp_dist if active_trade['side'] == 'LONG' else entry_price + tp_dist)
                                active_trade['break_even_triggered'] = True
                                active_trade['current_stop_price'] = new_stop_price
                                profit_pct = (current_profit / tp_dist) * 100
                                logging.info(f"✅ TRAILING STOP: Moved from {old_stop:.2f} to {new_stop_price:.2f} | Profit: {current_profit:.2f} ({profit_pct:.0f}% to TP)")

                                # Enhanced event logging: Breakeven/Trailing adjustment
                                event_logger.log_breakeven_adjustment(
                                    old_sl=old_stop,
                                    new_sl=new_stop_price,
                                    current_price=current_price,
                                    profit_points=current_profit
                                )

            currbar = new_df.iloc[-1]
            rejection_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            bank_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            chop_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            extension_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            structure_blocker.update(new_df)
            memory_sr.update(new_df)
            directional_loss_blocker.update_quarter(current_time)
            impulse_filter.update(new_df)


            # Only process signals on NEW bars
            is_new_bar = (last_processed_bar is None or current_time > last_processed_bar)
            
            if is_new_bar:
                # Sync local active trade with broker state to avoid getting stuck
                if active_trade is not None:
                    broker_pos = client.get_position()
                    if broker_pos.get('side') is None or broker_pos.get('size', 0) == 0:
                        logging.info("ℹ️ Broker reports flat while tracking active_trade; clearing local state so new signals can execute")
                        # Calculate PnL for directional loss tracking
                        trade_side = active_trade['side']
                        entry_price = active_trade['entry_price']
                        trade_size = active_trade.get('size', 5)
                        if trade_side == 'LONG':
                            pnl_points = current_price - entry_price
                        else:
                            pnl_points = entry_price - current_price
                        # Convert points to dollars: MES = $5 per point per contract
                        pnl_dollars = pnl_points * 5.0 * trade_size
                        directional_loss_blocker.record_trade_result(trade_side, pnl_points, current_time)
                        circuit_breaker.update_trade_result(pnl_dollars)
                        logging.info(f"📊 Trade closed: {trade_side} | Entry: {entry_price:.2f} | Exit: {current_price:.2f} | PnL: {pnl_points:.2f} pts (${pnl_dollars:.2f})")
                        active_trade = None
                        opposite_signal_count = 0
                        client._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}

                bar_count += 1
                logging.info(f"Bar: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Price: {current_price:.2f}")
                last_processed_bar = current_time
                
                # === UPDATE FILTERS ===
                curr_bar = new_df.iloc[-1]
                rejection_filter.update(current_time, curr_bar['high'], curr_bar['low'], curr_bar['close'])
                bank_filter.update(current_time, curr_bar['high'], curr_bar['low'], curr_bar['close'])
                chop_filter.update(curr_bar['high'], curr_bar['low'], curr_bar['close'], current_time)
                extension_filter.update(curr_bar['high'], curr_bar['low'], curr_bar['close'], current_time)
                structure_blocker.update(new_df)
                memory_sr.update(new_df)
                directional_loss_blocker.update_quarter(current_time)
                impulse_filter.update(new_df)

                # === EARLY EXIT CHECK ===
                if active_trade is not None:
                    active_trade['bars_held'] += 1
                    strategy_name = active_trade['strategy']
                    early_exit_config = CONFIG.get('EARLY_EXIT', {}).get(strategy_name, {})
                        
                    if active_trade['side'] == 'LONG':
                        is_green = current_price > active_trade['entry_price']
                    else:
                        is_green = current_price < active_trade['entry_price']
                        
                    was_green = active_trade.get('was_green')
                    if was_green is not None and is_green != was_green:
                        active_trade['profit_crosses'] = active_trade.get('profit_crosses', 0) + 1
                    active_trade['was_green'] = is_green
                    
                    if early_exit_config.get('enabled', False):
                        exit_time = early_exit_config.get('exit_if_not_green_by', 50)
                        exit_cross = early_exit_config.get('max_profit_crosses', 100)
                        should_exit = False
                        
                        if active_trade['bars_held'] >= exit_time and not is_green:
                            should_exit = True
                            exit_reason = f"not green after {active_trade['bars_held']} bars"
                        if active_trade.get('profit_crosses', 0) > exit_cross:
                            should_exit = True
                            exit_reason = f"choppy ({active_trade['profit_crosses']} crosses)"
                            
                        if should_exit:
                            logging.info(f"⏰ EARLY EXIT: {strategy_name} - {exit_reason}")

                            # Enhanced event logging: Early exit
                            event_logger.log_early_exit(
                                reason=exit_reason,
                                bars_held=active_trade['bars_held'],
                                current_price=current_price,
                                entry_price=active_trade['entry_price']
                            )

                            # Calculate PnL for directional loss tracking
                            trade_side = active_trade['side']
                            entry_price = active_trade['entry_price']
                            trade_size = active_trade.get('size', 5)
                            if trade_side == 'LONG':
                                pnl_points = current_price - entry_price
                            else:
                                pnl_points = entry_price - current_price
                            # Convert points to dollars: MES = $5 per point per contract
                            pnl_dollars = pnl_points * 5.0 * trade_size
                            directional_loss_blocker.record_trade_result(trade_side, pnl_points, current_time)
                            circuit_breaker.update_trade_result(pnl_dollars)
                            logging.info(f"📊 Early exit closed: {trade_side} | Entry: {entry_price:.2f} | Exit: {current_price:.2f} | PnL: {pnl_points:.2f} pts (${pnl_dollars:.2f})")

                            position = client.get_position()
                            if position['side'] is not None:
                                client.close_position(position)
                            active_trade = None

                # === STRATEGY EXECUTION ===
                strategy_results = {'checked': [], 'rejected': [], 'executed': None}
                
                # Run ML Analysis
                ml_signal = None
                if ml_strategy.model_loaded:
                    try:
                        ml_signal = ml_strategy.on_bar(new_df)
                        if ml_signal: strategy_results['checked'].append('MLPhysics')
                    except Exception as e:
                        logging.error(f"ML Strategy Error: {e}")
                
                # 2a. FAST STRATEGIES
                signal_executed = False
                for strat in fast_strategies:
                    strat_name = strat.__class__.__name__
                    try:
                        signal = strat.on_bar(new_df)
                        if signal:
                            strategy_results['checked'].append(strat_name)

                            # ==========================================
                            # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                            # ==========================================
                            # Apply the active session multipliers from CONFIG
                            # If Gemini is disabled or failed, these default to 1.0
                            sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                            tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                            if sl_mult != 1.0 or tp_mult != 1.0:
                                old_sl = signal.get('sl_dist', 4.0)
                                old_tp = signal.get('tp_dist', 6.0)

                                # Apply Multipliers
                                signal['sl_dist'] = old_sl * sl_mult
                                signal['tp_dist'] = old_tp * tp_mult

                                logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")
                            # ==========================================

                            # Enforce HTF range fade directional restriction
                            if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                                logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                                continue

                            # Enhanced event logging: Strategy signal generated
                            event_logger.log_strategy_signal(
                                strategy_name=signal.get('strategy', strat_name),
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist', 6.0),
                                sl_dist=signal.get('sl_dist', 4.0),
                                price=current_price,
                                additional_info={"execution_type": "FAST"}
                            )

                            # Filters - Rejection Filter
                            rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                            if rej_blocked:
                                event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                continue

                            # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                            dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                            if dir_blocked:
                                event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], False, dir_reason)
                                continue
                            else:
                                event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], True)

                            # Impulse Filter (Prevent catching falling knife / fading rocket ship)
                            impulse_blocked, impulse_reason = impulse_filter.should_block_trade(signal['side'])
                            if impulse_blocked:
                                event_logger.log_filter_check("ImpulseFilter", signal['side'], False, impulse_reason)
                                continue
                            else:
                                event_logger.log_filter_check("ImpulseFilter", signal['side'], True)

                            # HTF FVG (Memory Based) - CONTEXT AWARE
                            # Pass the strategy's target profit so we know how much room we need
                            tp_dist = signal.get('tp_dist', 15.0)

                            # === FIX: Relax FVG check if we are trading WITH the Range Fade ===
                            # If Chop says "Long Only" and we are going Long, we expect to break resistance.
                            # We reduce the effective TP distance passed to the filter, making it less strict.
                            effective_tp_dist = tp_dist
                            if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                                effective_tp_dist = tp_dist * 0.5  # Require 50% less room
                                logging.info(f"🔓 RELAXING FVG CHECK (Standard): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                            )

                            if fvg_blocked:
                                logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason)
                                continue
                            else:
                                event_logger.log_filter_check("HTF_FVG", signal['side'], True)

                            # Weak Level Blocker (EQH/EQL)
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if struct_blocked:
                                logging.info(f"🚫 {struct_reason}")
                                event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason)
                                continue
                            else:
                                event_logger.log_filter_check("StructureBlocker", signal['side'], True)

                            mem_blocked, mem_reason = memory_sr.should_block_trade(signal['side'], current_price)
                            if mem_blocked:
                                logging.info(f"🚫 {mem_reason}")
                                event_logger.log_filter_check("MemorySR", signal['side'], False, mem_reason)
                                continue
                            else:
                                event_logger.log_filter_check("MemorySR", signal['side'], True)

                            # Context for Chop Filter
                            trend_blocked_ctx, trend_reason_ctx = trend_filter.should_block_trade(new_df, signal['side'])
                            trend_state = ("Strong Bearish" if (trend_reason_ctx and "Bearish" in trend_reason_ctx)
                                           else ("Strong Bullish" if (trend_reason_ctx and "Bullish" in trend_reason_ctx)
                                                 else "NEUTRAL"))
                            vol_regime, _, _ = volatility_filter.get_regime(new_df)

                            # Chop
                            daily_bias = rejection_filter.prev_day_pm_bias
                            chop_blocked, chop_reason = chop_filter.should_block_trade(
                                signal['side'],
                                daily_bias,
                                current_price,
                                trend_state=trend_state,
                                vol_regime=vol_regime
                            )
                            if chop_blocked:
                                event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason)
                                continue
                            else:
                                event_logger.log_filter_check("ChopFilter", signal['side'], True)

                            # Extension
                            ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                            if ext_blocked:
                                event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason)
                                continue
                            else:
                                event_logger.log_filter_check("ExtensionFilter", signal['side'], True)

                            # Trend Filter
                            trend_blocked = trend_blocked_ctx
                            trend_reason = trend_reason_ctx
                            if trend_blocked:
                                event_logger.log_filter_check("TrendFilter", signal['side'], False, trend_reason)
                                continue
                            else:
                                event_logger.log_filter_check("TrendFilter", signal['side'], True)

                            # Volatility
                            should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)
                            if not should_trade:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed")
                                continue
                            else:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], True)

                            if vol_adj.get('adjustment_applied', False):
                                signal['sl_dist'] = vol_adj['sl_dist']
                                signal['tp_dist'] = vol_adj['tp_dist']
                                signal['size'] = vol_adj['size']  # Apply volatility-adjusted size
                                event_logger.log_trade_modified(
                                    "VolatilityAdjustment",
                                    signal.get('tp_dist', 6.0),
                                    vol_adj['tp_dist'],
                                    f"Volatility regime adjustment (size={vol_adj['size']})"
                                )

                            # Bank Filter (RegimeAdaptive Only)
                            bank_blocked, bank_reason = bank_filter.should_block_trade(signal['side'])
                            if bank_blocked:
                                event_logger.log_filter_check("BankFilter", signal['side'], False, bank_reason)
                                continue
                            else:
                                event_logger.log_filter_check("BankFilter", signal['side'], True)

                            # Execute
                            strategy_results['executed'] = strat_name
                            logging.info(f"✅ FAST EXEC: {signal['strategy']} signal")
                            event_logger.log_strategy_execution(signal.get('strategy', strat_name), "FAST")

                            # Track old trade result BEFORE close_and_reverse overwrites active_trade
                            if active_trade is not None:
                                old_side = active_trade['side']
                                old_entry = active_trade['entry_price']
                                old_size = active_trade.get('size', 5)
                                if old_side == 'LONG':
                                    old_pnl_points = current_price - old_entry
                                else:
                                    old_pnl_points = old_entry - current_price
                                # Convert points to dollars: MES = $5 per point per contract
                                old_pnl_dollars = old_pnl_points * 5.0 * old_size
                                directional_loss_blocker.record_trade_result(old_side, old_pnl_points, current_time)
                                circuit_breaker.update_trade_result(old_pnl_dollars)
                                logging.info(f"📊 Trade closed (reverse): {old_side} | Entry: {old_entry:.2f} | Exit: {current_price:.2f} | PnL: {old_pnl_points:.2f} pts (${old_pnl_dollars:.2f})")

                            success, opposite_signal_count = client.close_and_reverse(signal, current_price, opposite_signal_count)
                            if success:
                                sl_dist = signal.get('sl_dist', signal['tp_dist'])
                                initial_stop = current_price - sl_dist if signal['side'] == 'LONG' else current_price + sl_dist
                                active_trade = {
                                    'strategy': signal['strategy'],
                                    'side': signal['side'],
                                    'entry_price': current_price,
                                    'entry_bar': bar_count,
                                    'bars_held': 0,
                                    'tp_dist': signal['tp_dist'],
                                    'size': signal.get('size', 5),  # Use signal size (volatility-adjusted)
                                    'stop_order_id': client._active_stop_order_id,  # Cached stop ID
                                    'current_stop_price': initial_stop,  # Track for trailing stop
                                    'break_even_triggered': False
                                }

                            signal_executed = True
                            break
                    except Exception as e:
                        logging.error(f"Error in {strat_name}: {e}")
                
                # 2b. STANDARD STRATEGIES
                if not signal_executed:
                    for strat in standard_strategies:
                        strat_name = strat.__class__.__name__
                        signal = None
                        
                        if strat_name == "MLPhysicsStrategy":
                            signal = ml_signal
                        elif strat_name == "SMTStrategy":
                            signal = strat.on_bar(new_df, master_mnq_df)
                        else:
                            try:
                                signal = strat.on_bar(new_df)
                            except Exception as e:
                                logging.error(f"Error in {strat_name}: {e}")
                        
                        if signal:
                            strategy_results['checked'].append(strat_name)

                            # ==========================================
                            # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                            # ==========================================
                            # Apply the active session multipliers from CONFIG
                            # If Gemini is disabled or failed, these default to 1.0
                            sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                            tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                            if sl_mult != 1.0 or tp_mult != 1.0:
                                old_sl = signal.get('sl_dist', 4.0)
                                old_tp = signal.get('tp_dist', 6.0)

                                # Apply Multipliers
                                signal['sl_dist'] = old_sl * sl_mult
                                signal['tp_dist'] = old_tp * tp_mult

                                logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")
                            # ==========================================

                            # Enforce HTF range fade directional restriction
                            if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                                logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                                continue

                            # Enhanced event logging: Strategy signal generated
                            event_logger.log_strategy_signal(
                                strategy_name=signal.get('strategy', strat_name),
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist', 6.0),
                                sl_dist=signal.get('sl_dist', 4.0),
                                price=current_price,
                                additional_info={"execution_type": "STANDARD"}
                            )

                            # Rejection Filter
                            rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                            if rej_blocked:
                                event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                continue

                            # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                            dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                            if dir_blocked:
                                event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], False, dir_reason)
                                continue
                            else:
                                event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], True)

                            # Impulse Filter (Prevent catching falling knife / fading rocket ship)
                            impulse_blocked, impulse_reason = impulse_filter.should_block_trade(signal['side'])
                            if impulse_blocked:
                                event_logger.log_filter_check("ImpulseFilter", signal['side'], False, impulse_reason)
                                continue
                            else:
                                event_logger.log_filter_check("ImpulseFilter", signal['side'], True)

                            # HTF FVG (Memory Based) - CONTEXT AWARE
                            # Pass the strategy's target profit so we know how much room we need
                            tp_dist = signal.get('tp_dist', 15.0)

                            # === FIX: Relax FVG check if we are trading WITH the Range Fade ===
                            # If Chop says "Long Only" and we are going Long, we expect to break resistance.
                            # We reduce the effective TP distance passed to the filter, making it less strict.
                            effective_tp_dist = tp_dist
                            if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                                effective_tp_dist = tp_dist * 0.5  # Require 50% less room
                                logging.info(f"🔓 RELAXING FVG CHECK (Standard): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                            )

                            if fvg_blocked:
                                logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason)
                                continue
                            else:
                                event_logger.log_filter_check("HTF_FVG", signal['side'], True)

                            # Weak Level Blocker (EQH/EQL)
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if struct_blocked:
                                logging.info(f"🚫 {struct_reason}")
                                event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason)
                                continue
                            else:
                                event_logger.log_filter_check("StructureBlocker", signal['side'], True)

                            trend_blocked_ctx, trend_reason_ctx = trend_filter.should_block_trade(new_df, signal['side'])
                            trend_state = ("Strong Bearish" if (trend_reason_ctx and "Bearish" in trend_reason_ctx)
                                           else ("Strong Bullish" if (trend_reason_ctx and "Bullish" in trend_reason_ctx)
                                                 else "NEUTRAL"))
                            vol_regime, _, _ = volatility_filter.get_regime(new_df)

                            # Chop (Except DynamicEngine)
                            if signal['strategy'] not in ["DynamicEngine"]:
                                chop_blocked, chop_reason = chop_filter.should_block_trade(
                                    signal['side'],
                                    rejection_filter.prev_day_pm_bias,
                                    current_price,
                                    trend_state=trend_state,
                                    vol_regime=vol_regime
                                )
                                if chop_blocked:
                                    event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason)
                                    continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", signal['side'], True)

                            # Extension (Except DynamicEngine)
                            if signal['strategy'] not in ["DynamicEngine"]:
                                ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason)
                                    continue
                            else:
                                event_logger.log_filter_check("ExtensionFilter", signal['side'], True)

                            # Trend Filter
                            trend_blocked = trend_blocked_ctx
                            trend_reason = trend_reason_ctx
                            if trend_blocked:
                                event_logger.log_filter_check("TrendFilter", signal['side'], False, trend_reason)
                                continue
                            else:
                                event_logger.log_filter_check("TrendFilter", signal['side'], True)

                            # Volatility
                            should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)
                            if not should_trade:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed")
                                continue
                            else:
                                event_logger.log_filter_check("VolatilityFilter", signal['side'], True)

                            if vol_adj.get('adjustment_applied', False):
                                signal['sl_dist'] = vol_adj['sl_dist']
                                signal['tp_dist'] = vol_adj['tp_dist']
                                signal['size'] = vol_adj['size']  # Apply volatility-adjusted size
                                event_logger.log_trade_modified(
                                    "VolatilityAdjustment",
                                    signal.get('tp_dist', 6.0),
                                    vol_adj['tp_dist'],
                                    f"Volatility regime adjustment (size={vol_adj['size']})"
                                )

                            # Bank Filter (ML Only)
                            if strat_name == "MLPhysicsStrategy":
                                bank_blocked, bank_reason = bank_filter.should_block_trade(signal['side'])
                                if bank_blocked:
                                    event_logger.log_filter_check("BankFilter", signal['side'], False, bank_reason)
                                    continue
                                else:
                                    event_logger.log_filter_check("BankFilter", signal['side'], True)

                            strategy_results['executed'] = strat_name
                            logging.info(f"✅ STANDARD EXEC: {signal['strategy']} signal")
                            event_logger.log_strategy_execution(signal.get('strategy', strat_name), "STANDARD")

                            # Track old trade result BEFORE close_and_reverse overwrites active_trade
                            if active_trade is not None:
                                old_side = active_trade['side']
                                old_entry = active_trade['entry_price']
                                old_size = active_trade.get('size', 5)
                                if old_side == 'LONG':
                                    old_pnl_points = current_price - old_entry
                                else:
                                    old_pnl_points = old_entry - current_price
                                # Convert points to dollars: MES = $5 per point per contract
                                old_pnl_dollars = old_pnl_points * 5.0 * old_size
                                directional_loss_blocker.record_trade_result(old_side, old_pnl_points, current_time)
                                circuit_breaker.update_trade_result(old_pnl_dollars)
                                logging.info(f"📊 Trade closed (reverse): {old_side} | Entry: {old_entry:.2f} | Exit: {current_price:.2f} | PnL: {old_pnl_points:.2f} pts (${old_pnl_dollars:.2f})")

                            success, opposite_signal_count = client.close_and_reverse(signal, current_price, opposite_signal_count)
                            if success:
                                sl_dist = signal.get('sl_dist', signal['tp_dist'])
                                initial_stop = current_price - sl_dist if signal['side'] == 'LONG' else current_price + sl_dist
                                active_trade = {
                                    'strategy': signal['strategy'],
                                    'side': signal['side'],
                                    'entry_price': current_price,
                                    'entry_bar': bar_count,
                                    'bars_held': 0,
                                    'tp_dist': signal['tp_dist'],
                                    'size': signal.get('size', 5),  # Use signal size (volatility-adjusted)
                                    'stop_order_id': client._active_stop_order_id,  # Cached stop ID
                                    'current_stop_price': initial_stop,  # Track for trailing stop
                                    'break_even_triggered': False
                                }

                            signal_executed = True
                            break

# 2c. LOOSE STRATEGIES (Queued)
                if not signal_executed:
                    if is_new_bar:
                        # Process Pending
                        for s_name in list(pending_loose_signals.keys()):
                            pending = pending_loose_signals[s_name]
                            pending['bar_count'] += 1
                            if pending['bar_count'] >= 1:
                                sig = pending['signal']

                                # ==========================================
                                # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                                # ==========================================
                                # Apply the active session multipliers from CONFIG
                                # If Gemini is disabled or failed, these default to 1.0
                                sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                                tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                                if sl_mult != 1.0 or tp_mult != 1.0:
                                    old_sl = sig.get('sl_dist', 4.0)
                                    old_tp = sig.get('tp_dist', 6.0)

                                    # Apply Multipliers
                                    sig['sl_dist'] = old_sl * sl_mult
                                    sig['tp_dist'] = old_tp * tp_mult

                                    logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{sig['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{sig['tp_dist']:.2f} (x{tp_mult})")
                                # ==========================================

                                if allowed_chop_side is not None and sig['side'] != allowed_chop_side:
                                    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {sig['side']} vs Allowed {allowed_chop_side}")
                                    del pending_loose_signals[s_name]
                                    continue
                                # Re-check filters
                                rej_blocked, rej_reason = rejection_filter.should_block_trade(sig['side'])
                                if rej_blocked:
                                    event_logger.log_rejection_block("RejectionFilter", sig['side'], rej_reason or "Rejection bias")
                                    del pending_loose_signals[s_name]; continue

                                # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                                dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(sig['side'], current_time)
                                if dir_blocked:
                                    event_logger.log_filter_check("DirectionalLossBlocker", sig['side'], False, dir_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("DirectionalLossBlocker", sig['side'], True)

                                # Impulse Filter (Prevent catching falling knife / fading rocket ship)
                                impulse_blocked, impulse_reason = impulse_filter.should_block_trade(sig['side'])
                                if impulse_blocked:
                                    event_logger.log_filter_check("ImpulseFilter", sig['side'], False, impulse_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ImpulseFilter", sig['side'], True)

                                # HTF FVG (Memory Based) - CONTEXT AWARE
                                # Pass the strategy's target profit so we know how much room we need
                                tp_dist = sig.get('tp_dist', 15.0)

                                # === FIX: Relax FVG check if we are trading WITH the Range Fade ===
                                # If Chop says "Long Only" and we are going Long, we expect to break resistance.
                                # We reduce the effective TP distance passed to the filter, making it less strict.
                                effective_tp_dist = tp_dist
                                if allowed_chop_side is not None and sig['side'] == allowed_chop_side:
                                    effective_tp_dist = tp_dist * 0.5  # Require 50% less room
                                    logging.info(f"🔓 RELAXING FVG CHECK (Loose): Fading Range {sig['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                                fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                    sig['side'], current_price, None, None, tp_dist=effective_tp_dist
                                )

                                if fvg_blocked:
                                    logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                    event_logger.log_filter_check("HTF_FVG", sig['side'], False, fvg_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("HTF_FVG", sig['side'], True)

                                # === [FIX 1] UPDATED BLOCKER CHECK ===
                                struct_blocked, struct_reason = structure_blocker.should_block_trade(sig['side'], current_price)
                                if struct_blocked:
                                    logging.info(f"🚫 {struct_reason}")
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], False, struct_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], True)
                                mem_blocked, mem_reason = memory_sr.should_block_trade(sig['side'], current_price)
                                if mem_blocked:
                                    logging.info(f"🚫 {mem_reason}")
                                    event_logger.log_filter_check("MemorySR", sig['side'], False, mem_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("MemorySR", sig['side'], True)
                                # =====================================

                                trend_blocked_ctx, trend_reason_ctx = trend_filter.should_block_trade(new_df, sig['side'])
                                trend_state = ("Strong Bearish" if (trend_reason_ctx and "Bearish" in trend_reason_ctx)
                                               else ("Strong Bullish" if (trend_reason_ctx and "Bullish" in trend_reason_ctx)
                                                     else "NEUTRAL"))
                                vol_regime, _, _ = volatility_filter.get_regime(new_df)

                                chop_blocked, chop_reason = chop_filter.should_block_trade(
                                    sig['side'],
                                    rejection_filter.prev_day_pm_bias,
                                    current_price,
                                    trend_state=trend_state,
                                    vol_regime=vol_regime
                                )
                                if chop_blocked:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], False, chop_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], True)

                                ext_blocked, ext_reason = extension_filter.should_block_trade(sig['side'])
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], False, ext_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], True)

                                trend_blocked = trend_blocked_ctx
                                trend_reason = trend_reason_ctx
                                if trend_blocked:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], False, trend_reason)
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], True)

                                # Volatility
                                should_trade, vol_adj = check_volatility(new_df, sig.get('sl_dist', 4.0), sig.get('tp_dist', 6.0), base_size=5)
                                if not should_trade:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], False, "Volatility check failed")
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], True)

                                if vol_adj.get('adjustment_applied', False):
                                    sig['sl_dist'] = vol_adj['sl_dist']
                                    sig['tp_dist'] = vol_adj['tp_dist']
                                    sig['size'] = vol_adj['size']  # Apply volatility-adjusted size
                                    event_logger.log_trade_modified(
                                        "VolatilityAdjustment",
                                        sig.get('tp_dist', 6.0),
                                        vol_adj['tp_dist'],
                                        f"Volatility regime adjustment (size={vol_adj['size']})"
                                    )

                                logging.info(f"✅ LOOSE EXEC: {s_name}")
                                event_logger.log_strategy_execution(s_name, "LOOSE")

                                # Track old trade result BEFORE close_and_reverse overwrites active_trade
                                if active_trade is not None:
                                    old_side = active_trade['side']
                                    old_entry = active_trade['entry_price']
                                    old_size = active_trade.get('size', 5)
                                    if old_side == 'LONG':
                                        old_pnl_points = current_price - old_entry
                                    else:
                                        old_pnl_points = old_entry - current_price
                                    # Convert points to dollars: MES = $5 per point per contract
                                    old_pnl_dollars = old_pnl_points * 5.0 * old_size
                                    directional_loss_blocker.record_trade_result(old_side, old_pnl_points, current_time)
                                    circuit_breaker.update_trade_result(old_pnl_dollars)
                                    logging.info(f"📊 Trade closed (reverse): {old_side} | Entry: {old_entry:.2f} | Exit: {current_price:.2f} | PnL: {old_pnl_points:.2f} pts (${old_pnl_dollars:.2f})")

                                success, opposite_signal_count = client.close_and_reverse(sig, current_price, opposite_signal_count)
                                if success:
                                    sl_dist = sig.get('sl_dist', sig['tp_dist'])
                                    initial_stop = current_price - sl_dist if sig['side'] == 'LONG' else current_price + sl_dist
                                    active_trade = {
                                        'strategy': s_name,
                                        'side': sig['side'],
                                        'entry_price': current_price,
                                        'entry_bar': bar_count,
                                        'bars_held': 0,
                                        'tp_dist': sig['tp_dist'],
                                        'size': sig.get('size', 5),  # Use signal size (volatility-adjusted)
                                        'stop_order_id': client._active_stop_order_id,
                                        'current_stop_price': initial_stop,  # Track for trailing stop
                                        'break_even_triggered': False
                                    }

                                del pending_loose_signals[s_name]
                                signal_executed = True
                                break
                        
                        # Check New Loose Signals
                        if not signal_executed:
                            for strat in loose_strategies:
                                try:
                                    signal = strat.on_bar(new_df)
                                    s_name = strat.__class__.__name__
                                    if signal and s_name not in pending_loose_signals:
                                        # ==========================================
                                        # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                                        # ==========================================
                                        # Apply the active session multipliers from CONFIG
                                        # If Gemini is disabled or failed, these default to 1.0
                                        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                                        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                                        if sl_mult != 1.0 or tp_mult != 1.0:
                                            old_sl = signal.get('sl_dist', 4.0)
                                            old_tp = signal.get('tp_dist', 6.0)

                                            # Apply Multipliers
                                            signal['sl_dist'] = old_sl * sl_mult
                                            signal['tp_dist'] = old_tp * tp_mult

                                            logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")
                                        # ==========================================

                                        if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                                            logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                                            continue
                                        # Enhanced event logging: Strategy signal generated
                                        event_logger.log_strategy_signal(
                                            strategy_name=signal.get('strategy', s_name),
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist', 6.0),
                                            sl_dist=signal.get('sl_dist', 4.0),
                                            price=current_price,
                                            additional_info={"execution_type": "LOOSE"}
                                        )

                                        rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                                        if rej_blocked:
                                            event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                            continue

                                        # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                                        dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                                        if dir_blocked:
                                            event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], False, dir_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], True)

                                        tp_dist = signal.get('tp_dist', 15.0)

                                        effective_tp_dist = tp_dist
                                        if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                                            effective_tp_dist = tp_dist * 0.5
                                            logging.info(f"🔓 RELAXING FVG CHECK (Loose): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                                        fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                            signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                                        )
                                        if fvg_blocked:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], True)

                                        # === [FIX 2] UPDATED BLOCKER CHECK ===
                                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                                        if struct_blocked:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], True)
                                        mem_blocked, mem_reason = memory_sr.should_block_trade(signal['side'], current_price)
                                        if mem_blocked:
                                            event_logger.log_filter_check("MemorySR", signal['side'], False, mem_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("MemorySR", signal['side'], True)
                                        # =====================================

                                        trend_blocked_ctx, trend_reason_ctx = trend_filter.should_block_trade(new_df, signal['side'])
                                        trend_state = ("Strong Bearish" if (trend_reason_ctx and "Bearish" in trend_reason_ctx)
                                                       else ("Strong Bullish" if (trend_reason_ctx and "Bullish" in trend_reason_ctx)
                                                             else "NEUTRAL"))
                                        vol_regime, _, _ = volatility_filter.get_regime(new_df)

                                        chop_blocked, chop_reason = chop_filter.should_block_trade(
                                            signal['side'],
                                            rejection_filter.prev_day_pm_bias,
                                            current_price,
                                            trend_state=trend_state,
                                            vol_regime=vol_regime
                                        )
                                        if chop_blocked:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], True)

                                        ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                        if ext_blocked:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], True)

                                        trend_blocked = trend_blocked_ctx
                                        trend_reason = trend_reason_ctx
                                        if trend_blocked:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], False, trend_reason)
                                            continue
                                        else:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], True)

                                        # Volatility
                                        should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)
                                        if not should_trade:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed")
                                            continue
                                        else:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], True)

                                        if vol_adj.get('adjustment_applied', False):
                                            signal['sl_dist'] = vol_adj['sl_dist']
                                            signal['tp_dist'] = vol_adj['tp_dist']
                                            signal['size'] = vol_adj['size']  # Apply volatility-adjusted size
                                            event_logger.log_trade_modified(
                                                "VolatilityAdjustment",
                                                signal.get('tp_dist', 6.0),
                                                vol_adj['tp_dist'],
                                                f"Volatility regime adjustment (size={vol_adj['size']})"
                                            )

                                        logging.info(f"🕐 Queuing {s_name} signal")
                                        pending_loose_signals[s_name] = {'signal': signal, 'bar_count': 0}
                                except Exception as e:
                                    logging.error(f"Error in {s_name}: {e}")

            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nBot Stopped by User.")
            break
        except Exception as e:
            logging.error(f"Main Loop Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_bot()

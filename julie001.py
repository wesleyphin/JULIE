import requests
import pandas as pd
import numpy as np
import datetime
from datetime import date
import time
import logging
from zoneinfo import ZoneInfo
from datetime import timezone as dt_timezone
import uuid
from typing import Dict, Optional, List, Tuple
import random
import asyncio

from config import CONFIG, refresh_target_symbol, determine_current_contract_symbol
from dynamic_sltp_params import dynamic_sltp_engine, get_sltp
from volatility_filter import volatility_filter, check_volatility, VolRegime
from regime_strategy import RegimeAdaptiveStrategy
from htf_fvg_filter import HTFFVGFilter
from rejection_filter import RejectionFilter
from chop_filter import ChopFilter
from extension_filter import ExtensionFilter
from trend_filter import TrendFilter
from dynamic_structure_blocker import DynamicStructureBlocker, RegimeStructureBlocker, PenaltyBoxBlocker
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
import param_scaler
# --- NEW IMPORTS ---
from vixmeanreversion import VIXReversionStrategy
from yahoo_vix_client import YahooVIXClient
from legacy_filters import LegacyFilterSystem
from filter_arbitrator import FilterArbitrator
# --- NEW IMPORTS ---
from continuation_strategy import FractalSweepStrategy, STRATEGY_CONFIGS

# --- ASYNCIO IMPORTS ---
from async_market_stream import AsyncMarketDataManager
from async_tasks import heartbeat_task, position_sync_task, htf_structure_task

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

class ContinuationRescueManager:
    """
    Manages the FractalSweepStrategy (Continuation) lookups.
    Acts as a 'Second Opinion' when trades are blocked by filters.
    """
    def __init__(self):
        self.configs = STRATEGY_CONFIGS
        self.strategy_instances = {}
        # 1. TIMEZONE FIX: Strategies operate on NY Time
        self.ny_tz = ZoneInfo('America/New_York')

    def get_active_continuation_signal(self, df: pd.DataFrame, current_time, required_side: str):
        """
        Checks if the current time matches a known Continuation Strategy window.
        Returns a rescue signal if valid for the REQUIRED_SIDE.
        """
        if df.empty:
            return None

        # 2. Convert Bot Time (UTC) to Strategy Time (NY)
        if current_time.tzinfo is None:
             current_time = current_time.replace(tzinfo=dt_timezone.utc)

        ny_time = current_time.astimezone(self.ny_tz)

        # 3. Construct Key using NY TIME (e.g. Q4_W45_D7_Asia)
        quarter = (ny_time.month - 1) // 3 + 1
        week = ny_time.isocalendar().week
        day = ny_time.weekday() + 1 # 1=Monday, 7=Sunday
        h = ny_time.hour

        if 18 <= h or h < 3: session = "Asia"
        elif 3 <= h < 8: session = "London"
        elif 8 <= h < 17: session = "NY"
        else: session = "Other"

        candidate_key = f"Q{quarter}_W{week}_D{day}_{session}"

        # 4. Check Config & Instantiate
        if candidate_key not in self.configs:
            return None

        if candidate_key not in self.strategy_instances:
            try:
                self.strategy_instances[candidate_key] = FractalSweepStrategy(candidate_key)
            except ValueError:
                return None

        strat = self.strategy_instances[candidate_key]

        # 5. Generate Signal
        try:
            signals_df = strat.generate_signals(df)

            if not signals_df.empty:
                # Verify freshness
                last_sig_time = signals_df.index[-1]
                if last_sig_time.tzinfo is None:
                    last_sig_time = last_sig_time.replace(tzinfo=dt_timezone.utc)
                else:
                    last_sig_time = last_sig_time.astimezone(dt_timezone.utc)

                check_time = current_time.astimezone(dt_timezone.utc)

                if last_sig_time == check_time:
                    return {
                        'strategy': f"Continuation_{candidate_key}",
                        'side': required_side, # FORCE the direction we need (The Rescue Side)
                        'tp_dist': strat.target if hasattr(strat, 'target') else 6.0,
                        'sl_dist': strat.stop if hasattr(strat, 'stop') else 4.0,
                        'size': 5,
                        'rescued': True
                    }
        except Exception as e:
            logging.error(f"Continuation Strategy Error ({candidate_key}): {e}")
            return None

        return None

# ==========================================
# 12. MAIN EXECUTION LOOP (ASYNCIO UPGRADED)
# ==========================================
async def run_bot():
    """
    Main bot execution loop - now async with independent tasks.

    Benefits:
    - Independent Heartbeat task (validates session every 60s)
    - Independent Position Sync task (syncs broker position every 30s)
    - Non-blocking sleep for faster response times
    - Strategy calculations cannot block heartbeat or position sync
    """
    param_scaler.apply_scaling()  # Scale regime params to maintain R:R ratios

    refresh_target_symbol()
    print("=" * 60)
    print("PROJECTX GATEWAY - MES FUTURES BOT (LIVE)")
    print("--- Julie Pro (Session Specialized + AsyncIO) ---")
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

    # --- UPDATED: VIX Client (Using Yahoo Finance) ---
    # We use ^VIX (The Index) as it is the standard for mean reversion
    # and free via Yahoo, whereas Topstep Rithmic usually lacks CBOE data.
    logging.info("Initializing Virtual VIX Client (Yahoo Finance)...")
    vix_client = YahooVIXClient(target_symbol="^VIX")

    try:
        mnq_client.login()
        mnq_client.account_id = client.account_id or mnq_client.fetch_accounts()
        mnq_client.fetch_contracts()

        # Login VIX client (Virtual)
        vix_client.login()
        # No account ID needed for Yahoo, but we call methods for consistency
        vix_client.fetch_contracts()
    except Exception as e:
        logging.error(f"❌ Failed to initialize secondary clients: {e}")
        return

    # Initialize all strategies

    # Dynamic chop analyzer (tiered thresholds with LTF breakout override)
    chop_analyzer = DynamicChopAnalyzer(client)
    chop_analyzer.calibrate()  # Removed session_name argument
    last_chop_calibration = time.time()

    # --- NEW: Initialize VIX Strategy ---
    vix_strategy = VIXReversionStrategy()

    # HIGH PRIORITY - Execute immediately on signal
    # CHANGED: Dynamic Engine stays here. VIX added. Intraday Dip removed.
    dynamic_engine_strat = DynamicEngineStrategy()
    fast_strategies = [
        RegimeAdaptiveStrategy(),
        vix_strategy,          # Promoted to Fast
        dynamic_engine_strat,  # Kept in Fast (Not Demoted)
    ]

    # STANDARD PRIORITY - Normal execution
    ml_strategy = MLPhysicsStrategy()
    smt_strategy = SMTStrategy()

    standard_strategies = [
        IntradayDipStrategy(), # DEMOTED to Standard
        ConfluenceStrategy(),
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
    # 4-Tier Trend Filter (merged with Impulse logic)
    # Tier 1: Volume-supported impulse, Tier 2: Standard breakout, Tier 3: Extreme capitulation
    # Tier 4: Macro trend (50/200 EMA alignment) - bypassed by Range Fade logic
    trend_filter = TrendFilter()
    htf_fvg_filter = HTFFVGFilter() # Now uses Memory-Based Class
    structure_blocker = DynamicStructureBlocker(lookback=50)  # Macro trend + fade detection
    regime_blocker = RegimeStructureBlocker(lookback=20)      # Regime-based EQH/EQL tolerance
    penalty_blocker = PenaltyBoxBlocker(lookback=50, tolerance=5.0, penalty_bars=3)  # Fixed 5pt + 3-bar decay
    memory_sr = MemorySRFilter(lookback_bars=300, zone_width=2.0, touch_threshold=2)
    news_filter = NewsFilter()
    circuit_breaker = CircuitBreaker(max_daily_loss=600, max_consecutive_losses=7)
    directional_loss_blocker = DirectionalLossBlocker(consecutive_loss_limit=3, block_minutes=15)
    impulse_filter = ImpulseFilter(lookback=20, impulse_multiplier=2.5)

    # === DUAL-FILTER SYSTEM ===
    # Legacy (Dec 17th) filters for comparison + Arbitrator for override decisions
    legacy_filters = LegacyFilterSystem()
    filter_arbitrator = FilterArbitrator(confidence_threshold=0.6)

    # Initialize Gemini Session Optimizer
    optimizer = GeminiSessionOptimizer()

    # Initialize Rescue Manager
    continuation_manager = ContinuationRescueManager()

    last_processed_session = None
    last_processed_quarter = None  # Track quarter for quarterly optimization

    print("\nActive Strategies:")
    print("  [FAST EXECUTION]")
    for strat in fast_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [STANDARD EXECUTION]")
    for strat in standard_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [LOOSE EXECUTION]")
    for strat in loose_strategies: print(f"    • {strat.__class__.__name__}")

    print("\n🚀 AsyncIO Upgrade Active - Launching Independent Tasks...")
    print("  ✓ Heartbeat Task (validates session every 60s)")
    print("  ✓ Position Sync Task (syncs broker position every 30s)")
    print("\nListening for market data (faster polling with async)...")

    # === LAUNCH INDEPENDENT ASYNC TASKS ===
    # These tasks run independently and cannot be blocked by strategy calculations
    heartbeat = asyncio.create_task(heartbeat_task(client, interval=60))
    position_sync = asyncio.create_task(position_sync_task(client, interval=30))

    # NEW: Background HTF Updater
    # This keeps your FVG memory fresh without pausing the bot
    htf_updater = asyncio.create_task(htf_structure_task(client, htf_fvg_filter, interval=60))

    # === TRACKING VARIABLES ===
    # Position sync now handled by independent async task - removed manual tracking
    
    # Track pending signals for delayed execution
    pending_loose_signals = {}
    last_processed_bar = None
    opposite_signal_count = 0

    # Early Exit Tracking
    active_trade = None
    bar_count = 0

    # Token refresh now handled by independent heartbeat task

    # Chop state tracking (only log when state changes)
    last_chop_reason = None

    # === STEP 1: INITIAL DATA LOAD (MAX HISTORY) ===
    event_logger.log_system_event("STARTUP", "⏳ Startup: Fetching 20,000 bar history (MES)...", {"status": "IN_PROGRESS"})
    logging.info("⏳ Startup: Fetching full 20,000 bar history (MES)...")
    # Fetch the maximum allowed history ONCE before the loop starts
    master_df = client.get_market_data(lookback_minutes=20000, force_fetch=True)
    event_logger.log_system_event("STARTUP", f"✅ History Received: {len(master_df)} bars loaded (MES).", {"status": "COMPLETE"})

    event_logger.log_system_event("STARTUP", "⏳ Startup: Fetching 20,000 bar history (MNQ)...", {"status": "IN_PROGRESS"})
    logging.info("⏳ Startup: Fetching full 20,000 bar history (MNQ)...")
    master_mnq_df = mnq_client.get_market_data(lookback_minutes=20000, force_fetch=True)
    event_logger.log_system_event("STARTUP", f"✅ History Received: {len(master_mnq_df)} bars loaded (MNQ).", {"status": "COMPLETE"})

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

    # --- NEW: Initialize VIX master dataframe ---
    master_vix_df = pd.DataFrame()

    # One-time backfill flag
    data_backfilled = False

    while True:
        try:
            # Token validation now handled by independent heartbeat task

            # === DATA FRESHNESS CHECK (Safety Circuit Breaker) ===
            if not master_df.empty:
                last_bar_time = master_df.index[-1]
                if last_bar_time.tzinfo is None:
                    last_bar_time = last_bar_time.replace(tzinfo=dt_timezone.utc)

                seconds_since_last_update = (datetime.datetime.now(dt_timezone.utc) - last_bar_time).total_seconds()

                if seconds_since_last_update > 300:  # Increased from 60 for low volume periods
                    event_logger.log_error("DATA_STALE", f"🚨 DATA LAG: Last update was {seconds_since_last_update:.0f}s ago. Moving to DEFENSIVE mode.")
                    logging.warning(f"🚨 DATA LAG: Last update was {seconds_since_last_update:.0f}s ago. Attempting to fetch fresh data...")
                    await asyncio.sleep(5)
                    # continue  # Removed: Allow bot to proceed to data fetch even when stale

            # Periodic chop threshold recalibration (default every 4 hours)
            if chop_analyzer.should_recalibrate(last_chop_calibration):
                chop_analyzer.calibrate() # Removed session_name argument
                last_chop_calibration = time.time()

            # === GLOBAL RISK & NEWS FILTERS ===
            cb_blocked, cb_reason = circuit_breaker.should_block_trade()
            if cb_blocked:
                logging.info(f"🚫 Circuit Breaker Block: {cb_reason}")
                await asyncio.sleep(60)
                continue

            current_time = datetime.datetime.now(dt_timezone.utc)
            news_blocked, news_reason = news_filter.should_block_trade(current_time)
            if news_blocked:
                logging.info(f"🚫 NEWS WAIT: {news_reason}")
                # Enhanced logging with news filter details
                news_info = {
                    "Status": "BLACKOUT",
                    "Reason": "High-Impact Event"
                }
                # Extract time remaining from reason if available
                if "min" in news_reason:
                    # Try to extract the time remaining
                    import re
                    match = re.search(r'(\d+)\s*min', news_reason)
                    if match:
                        news_info["Wait"] = f"{match.group(1)}m"
                event_logger.log_filter_check("NewsFilter", "ALL", False, news_reason,
                                             additional_info=news_info, strategy="Global")
                await asyncio.sleep(10)
                continue

            # ==========================================
            # 🕒 UPDATED SESSION DETECTION (INTRADAY + MICRO-ZONES)
            # ==========================================
            current_time_et = datetime.datetime.now(NY_TZ)
            hour = current_time_et.hour
            minute = current_time_et.minute

            # 1. Determine Broad Parent Session (For Data Slicing & Config Lookup)
            # (Keeps your original logic intact for data fetching)
            if 18 <= hour or hour < 3:
                base_session = "ASIA"
            elif 3 <= hour < 8:
                base_session = "LONDON"
            elif 8 <= hour < 12:
                base_session = "NY_AM"
            elif 12 <= hour < 17:
                base_session = "NY_PM"
            else:
                base_session = "POST_MARKET"

            # 2. Determine Micro-Session (The "Trump Era" Logic)
            # (Adds granularity for the Optimizer, defaults to base_session)
            current_session_name = base_session

            if base_session == "NY_AM":
                # "Safe Window" is 09:30-10:15 (Standard NY_AM)
                # "Lunchtime Death" starts 10:30
                if hour == 10 and minute >= 30:
                    current_session_name = "NY_LUNCH"
                elif hour == 11:
                    current_session_name = "NY_LUNCH"

            elif base_session == "NY_PM":
                # "Lunchtime Death" ends 12:30
                if hour == 12 and minute < 30:
                    current_session_name = "NY_LUNCH"
                # "Close Trap" starts 15:00
                elif hour >= 15:
                    current_session_name = "NY_CLOSE"

            # --- OPTIMIZATION TRIGGER (Every Session Quarter) ---
            # Get current quarter (1-4) within the session
            current_quarter = bank_filter.get_quarter(hour, minute, base_session)

            # Trigger optimization on session change OR quarter change (4 sessions × 4 quarters = 16 per day)
            session_changed = current_session_name != last_processed_session
            quarter_changed = current_quarter != last_processed_quarter

            if session_changed or quarter_changed:
                if session_changed:
                    logging.info(f"🔄 SESSION HANDOVER: {last_processed_session} -> {current_session_name} Q{current_quarter} (Base: {base_session})")
                else:
                    logging.info(f"🔄 QUARTER CHANGE: {current_session_name} Q{last_processed_quarter} -> Q{current_quarter}")

                if CONFIG.get('GEMINI', {}).get('enabled', False):
                    print("\n" + "=" * 60)
                    print(f"🧠 GEMINI OPTIMIZATION - {current_session_name} Q{current_quarter}")
                    print("=" * 60)

                    # 1. Fetch Events & Holiday Context
                    try:
                        raw_events = news_filter.fetch_news()
                        events_str = str(raw_events)
                    except Exception as e:
                        events_str = "Events data unavailable."

                    try:
                        holiday_context = news_filter.get_holiday_context(current_time)
                        # Log holiday status
                        if holiday_context == "HOLIDAY_TODAY":
                            logging.info(f"🚨 HOLIDAY STATUS: {holiday_context} - Market closed/dead volume")
                        elif holiday_context.startswith("PRE_HOLIDAY"):
                            days = holiday_context.split("_")[-2]
                            logging.info(f"📅 HOLIDAY STATUS: Bank Holiday in {days} day(s) - Reducing targets")
                        elif holiday_context == "POST_HOLIDAY_RECOVERY":
                            logging.info(f"🔄 HOLIDAY STATUS: {holiday_context} - Volatility expanding")
                        else:
                            logging.info(f"✅ HOLIDAY STATUS: {holiday_context}")
                    except Exception as e:
                        logging.warning(f"Failed to get holiday context: {e}")
                        holiday_context = "NORMAL_LIQUIDITY"

                    # Get Seasonal Context
                    try:
                        seasonal_context = news_filter.get_seasonal_context(current_time)
                        # Log seasonal phase with specific emoji indicators
                        if seasonal_context == "PHASE_1_LAST_GASP":
                            logging.info(f"⚡ SEASONAL PHASE: LAST GASP (Dec 20-23) - High volume, violent trends")
                        elif seasonal_context == "PHASE_2_DEAD_ZONE":
                            logging.info(f"☠️  SEASONAL PHASE: DEAD ZONE (Dec 24-31) - 60% volume drop, broken structure")
                        elif seasonal_context == "PHASE_3_JAN2_REENTRY":
                            logging.info(f"🐻 SEASONAL PHASE: JAN 2 RE-ENTRY - Bearish bias, funds returning")
                        # NORMAL_SEASONAL doesn't need logging
                    except Exception as e:
                        logging.warning(f"Failed to get seasonal context: {e}")
                        seasonal_context = "NORMAL_SEASONAL"

                    # Log Micro-Session Specifics
                    if current_session_name == "NY_LUNCH":
                        logging.info(f"🧟 MICRO-SESSION: ZOMBIE ZONE (10:30-12:30) - Liquidity drops to 58%")
                    elif current_session_name == "NY_CLOSE":
                        logging.info(f"⚠️  MICRO-SESSION: CLOSE TRAP (15:00-16:00) - High volume, mean-reversion")

                    # 2. Get Hardcoded Base Params for Session
                    # CRITICAL: Use 'base_session' to look up CONFIG, not the new Micro-Session name
                    # because your config.py likely only has ASIA, LONDON, NY_AM, NY_PM.
                    session_cfg = CONFIG['SESSIONS'].get(base_session, {})
                    base_sl = session_cfg.get('SL', 4.0)
                    base_tp = session_cfg.get('TP', 8.0)

                    # --- NEW: Generate Structure Context String ---
                    structure_price = master_df.iloc[-1]['close'] if not master_df.empty else 0

                    # 2a. Get Memory S/R (Nearest 2 levels)
                    nearest_supports = sorted(
                        [s for s in memory_sr.supports if s < structure_price],
                        key=lambda x: structure_price - x
                    )[:2]
                    nearest_resistances = sorted(
                        [r for r in memory_sr.resistances if r > structure_price],
                        key=lambda x: x - structure_price
                    )[:2]

                    sr_str = f"Current Price: {structure_price:.2f}\n"
                    sr_str += f"Nearest Support (Memory): {nearest_supports}\n"
                    sr_str += f"Nearest Resistance (Memory): {nearest_resistances}\n"

                    # 2b. Get HTF FVGs (Active Memories)
                    active_fvgs = htf_fvg_filter.memory
                    fvg_str = "Active HTF FVGs:\n"
                    if active_fvgs:
                        for fvg in active_fvgs:
                            dist = 0
                            status = "Away"
                            if fvg['type'] == 'bullish':
                                if fvg['bottom'] <= structure_price <= fvg['top']:
                                    status = "INSIDE ZONE"
                                elif structure_price > fvg['top']:
                                    dist = structure_price - fvg['top']
                                    status = f"{dist:.2f} pts above"
                            else:
                                if fvg['bottom'] <= structure_price <= fvg['top']:
                                    status = "INSIDE ZONE"
                                elif structure_price < fvg['bottom']:
                                    dist = fvg['bottom'] - structure_price
                                    status = f"{dist:.2f} pts below"

                            fvg_str += f" - {fvg['tf']} {fvg['type'].upper()} ({fvg['bottom']:.2f}-{fvg['top']:.2f}): {status}\n"
                    else:
                        fvg_str += " - None nearby\n"

                    full_structure_context = sr_str + "\n" + fvg_str
                    # -----------------------------------------------

                    # 3. Call Gemini with structure context (including seasonal & micro-session)
                    opt_result = optimizer.optimize_new_session(
                        master_df,
                        current_session_name,
                        events_str,
                        base_sl,
                        base_tp,
                        structure_context=full_structure_context,
                        active_fvgs=active_fvgs,
                        holiday_context=holiday_context,
                        seasonal_context=seasonal_context,
                        base_session_name=base_session  # Pass parent session for data slicing
                    )

                    if opt_result:
                        sl_mult = float(opt_result.get('sl_multiplier', 1.0))
                        tp_mult = float(opt_result.get('tp_multiplier', 1.0))
                        # NEW: Extract Chop Multiplier
                        chop_mult = float(opt_result.get('chop_multiplier', 1.0))

                        reason = opt_result.get('reasoning', '')
                        trend_params = opt_result.get('trend_params', {})

                        # 4. Update Global Config & Filters
                        CONFIG['DYNAMIC_SL_MULTIPLIER'] = sl_mult
                        CONFIG['DYNAMIC_TP_MULTIPLIER'] = tp_mult

                        # NEW: Update DynamicChop Analyzer
                        chop_analyzer.update_gemini_params(chop_mult)

                        # Update Trend Filter with dynamic parameters from Gemini
                        if trend_params:
                            trend_filter.update_dynamic_params(trend_params)

                        # Enhanced logging with holiday context
                        logging.info(f"🎯 NEW MULTIPLIERS | SL: {sl_mult}x | TP: {tp_mult}x | CHOP: {chop_mult}x")
                        logging.info(f"🌊 TREND REGIME: {trend_params.get('regime', 'DEFAULT')}")

                        # Show holiday-specific adjustments if applicable
                        if holiday_context != "NORMAL_LIQUIDITY":
                            if holiday_context == "HOLIDAY_TODAY":
                                logging.info(f"⚠️  HOLIDAY ADJUSTMENTS: Extreme risk reduction (Market closed)")
                            elif holiday_context.startswith("PRE_HOLIDAY"):
                                logging.info(f"⚠️  HOLIDAY ADJUSTMENTS: Targets reduced ~40% (Pre-holiday illiquidity)")
                            elif holiday_context == "POST_HOLIDAY_RECOVERY":
                                logging.info(f"⚠️  HOLIDAY ADJUSTMENTS: Stops widened +12% (Post-holiday volatility)")

                        logging.info(f"📝 REASONING: {reason}")
                        print("=" * 60 + "\n")
                    else:
                        CONFIG['DYNAMIC_SL_MULTIPLIER'] = 1.0
                        CONFIG['DYNAMIC_TP_MULTIPLIER'] = 1.0
                        chop_analyzer.update_gemini_params(1.0)  # Reset on failure
                        logging.warning("⚠️  Gemini optimization failed - using default multipliers")
                        print("=" * 60 + "\n")

                last_processed_session = current_session_name
                last_processed_quarter = current_quarter

            # === STEP 2: INCREMENTAL UPDATE (SEQUENTIAL FETCH) ===
            # Fetch MES first, then MNQ, then VIX immediately after to keep timestamps close
            recent_data = client.get_market_data(lookback_minutes=15, force_fetch=True)
            recent_mnq_data = mnq_client.get_market_data(lookback_minutes=15, force_fetch=True)
            # --- NEW: Fetch VIX Data ---
            recent_vix_data = vix_client.get_market_data(lookback_minutes=15, force_fetch=True)

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

            # --- NEW: Handle VIX Data ---
            if not recent_vix_data.empty:
                master_vix_df = pd.concat([master_vix_df, recent_vix_data])
                master_vix_df = master_vix_df[~master_vix_df.index.duplicated(keep='last')]
                if len(master_vix_df) > 50000:
                    master_vix_df = master_vix_df.iloc[-50000:]

            # Make sure we have data before proceeding
            if master_df.empty or master_mnq_df.empty or master_vix_df.empty:
                # Early heartbeat - shows bot is alive even when no data available
                if not hasattr(client, '_empty_data_counter'):
                    client._empty_data_counter = 0
                client._empty_data_counter += 1
                if client._empty_data_counter % 30 == 0:
                    print(f"⏳ Waiting for data: {datetime.datetime.now().strftime('%H:%M:%S')} | No bars received (market may be closed or starting up)")
                    logging.info(f"No market data available - attempt #{client._empty_data_counter}")
                await asyncio.sleep(2)
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
                event_logger.log_system_event("STARTUP", "🔄 Restoring filter states from history...", {"type": "BACKFILL", "status": "IN_PROGRESS"})
                logging.info("🔄 Performing one-time backfill of filter state from history...")
                # Replay the history we just fetched
                # This restores Midnight ORB, Prev Session, etc. instantly
                rejection_filter.backfill(new_df)

                # Backfill extension_filter (prevents Mid-Day Amnesia bug)
                extension_filter.backfill(new_df)

                # Also backfill bank_filter (has same update() signature)
                for ts, row in new_df.sort_index().iterrows():
                    bank_filter.update(ts, row['high'], row['low'], row['close'])

                data_backfilled = True
                event_logger.log_system_event("STARTUP", "✅ State restored. Bot is ready.", {"status": "READY"})
                logging.info("✅ State restored from history.")

            # === UPDATE FILTERS (BEFORE CHOP CHECK - Prevents Stale Filters) ===
            # These must run before chop check so filters stay current even when choppy
            current_price = new_df.iloc[-1]['close']
            current_time = new_df.index[-1]
            currbar = new_df.iloc[-1]
            rejection_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            bank_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            chop_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            extension_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            structure_blocker.update(new_df)
            regime_blocker.update(new_df)
            penalty_blocker.update(new_df)
            memory_sr.update(new_df)
            directional_loss_blocker.update_quarter(current_time)
            impulse_filter.update(new_df)

            # === DYNAMIC CHOP CHECK (Pass Local DFs) ===
            # We pass the locally generated df_60m so the analyzer can use it for breakout shift logic
            is_choppy, chop_reason = chop_analyzer.check_market_state(new_df, df_60m_current=df_60m)

            # Initialize allowed_chop_side for this iteration (Fixes NameError)
            allowed_chop_side = None

            if is_choppy:
                # Check if this is a "Range Fade" permission instead of a hard block
                if "ALLOW_LONG_ONLY" in chop_reason:
                    allowed_chop_side = "LONG"
                    # Do NOT continue; allow the loop to proceed but enforce LONG only
                    if last_chop_reason != chop_reason:
                        logging.info(f"⚠️ CHOP RESTRICTION: {chop_reason}")
                        last_chop_reason = chop_reason

                elif "ALLOW_SHORT_ONLY" in chop_reason:
                    allowed_chop_side = "SHORT"
                    # Do NOT continue; allow the loop to proceed but enforce SHORT only
                    if last_chop_reason != chop_reason:
                        logging.info(f"⚠️ CHOP RESTRICTION: {chop_reason}")
                        last_chop_reason = chop_reason

                else:
                    # Hard Block (Standard Chop)
                    # Log every single time or throttle it
                    # logging.info(f"⛔ TRADE BLOCKED: {chop_reason}")
                    await asyncio.sleep(0.5)  # Faster check when choppy
                    continue
            else:
                # Clear chop state if no restriction active
                if last_chop_reason is not None:
                    logging.info("✅ CHOP RESTRICTION CLEARED")
                    last_chop_reason = None

            # ==========================================
            # HEARTBEAT & POSITION SYNC NOW HANDLED BY INDEPENDENT ASYNC TASKS
            # See: heartbeat_task() and position_sync_task() launched at startup
            # These tasks run independently and cannot be blocked by strategy logic
            # ==========================================
            now_ts = time.time()

            # === HTF FVG MEMORY NOW UPDATED BY BACKGROUND TASK ===
            # See: htf_structure_task() launched at startup
            # This task runs independently and cannot be blocked by strategy logic

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

            # Only process signals on NEW bars
            is_new_bar = (last_processed_bar is None or current_time > last_processed_bar)
            
            if is_new_bar:
                # Sync local active trade with broker state to avoid getting stuck
                if active_trade is not None:
                    broker_pos = client.get_position()

                    # SAFETY CHECK: Only clear if broker EXPLICITLY says Flat (side is None)
                    # and we are confident (size is 0).
                    # If broker_pos returns the cached state (from rate limit fix), this logic holds.
                    if broker_pos.get('side') is None or broker_pos.get('size', 0) == 0:
                        logging.info("ℹ️ Broker reports flat while tracking active_trade; clearing local state.")
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

                # === RISK TELEMETRY (PERIODIC HEARTBEAT) ===
                # Calculate current risk metrics
                current_dd = abs(min(circuit_breaker.daily_pnl, 0))  # Current daily loss (positive value)
                max_dd = circuit_breaker.max_daily_loss
                daily_pnl = circuit_breaker.daily_pnl

                # Log telemetry every 15 minutes OR if drawdown > 50%
                minute = current_time.minute
                usage_pct = (current_dd / max_dd * 100) if max_dd > 0 else 0
                if minute % 15 == 0 or usage_pct > 50:
                    event_logger.log_risk_telemetry(
                        current_loss=current_dd,
                        limit=max_dd,
                        daily_pnl=daily_pnl
                    )

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

                # =================================================================
                # 🎯 HARVEST ALL SIGNALS (Solves "Ghost Signal" Problem)
                # =================================================================
                # Collect ALL potential signals from ALL strategies BEFORE filtering
                # This enables opportunity cost analysis - see what was blocked
                candidate_signals = []  # List of (priority, strategy_instance, signal_dict, strat_name)

                # -----------------------------------------------------------------
                # HARVEST PHASE 1: FAST STRATEGIES (Priority 1)
                # -----------------------------------------------------------------
                current_fast = fast_strategies.copy()
                random.shuffle(current_fast)

                for strat in current_fast:
                    strat_name = strat.__class__.__name__
                    try:
                        # Handle specific arguments for VIX vs others
                        if strat_name == "VIXReversionStrategy":
                            signal = strat.on_bar(new_df, master_vix_df)
                        else:
                            signal = strat.on_bar(new_df)

                        if signal:
                            # ==========================================
                            # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                            # ==========================================
                            sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                            tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                            old_sl = signal.get('sl_dist', 4.0)
                            old_tp = signal.get('tp_dist', 6.0)

                            if 'sl_dist' not in signal or 'tp_dist' not in signal:
                                logging.warning(f"⚠️ {strat_name} missing sl_dist/tp_dist, using defaults")

                            signal['sl_dist'] = old_sl * sl_mult
                            signal['tp_dist'] = old_tp * tp_mult

                            # Enforce minimums
                            MIN_SL = 4.0
                            MIN_TP = 6.0
                            if signal['sl_dist'] < MIN_SL:
                                logging.warning(f"⚠️ SL too tight ({signal['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                                signal['sl_dist'] = MIN_SL
                            if signal['tp_dist'] < MIN_TP:
                                logging.warning(f"⚠️ TP too tight ({signal['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                                signal['tp_dist'] = MIN_TP

                            if sl_mult != 1.0 or tp_mult != 1.0:
                                logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")

                            # Enforce HTF range fade directional restriction
                            # if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                            #    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                            #    continue

                            # Add to candidate list (Priority 1 = FAST)
                            candidate_signals.append((1, strat, signal, strat_name))

                            # Log as candidate
                            event_logger.log_strategy_signal(
                                strategy_name=signal.get('strategy', strat_name),
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist', 6.0),
                                sl_dist=signal.get('sl_dist', 4.0),
                                price=current_price,
                                additional_info={"status": "CANDIDATE", "priority": "FAST"}
                            )
                            logging.info(f"📊 CANDIDATE (FAST): {strat_name} {signal['side']} @ {current_price:.2f}")

                    except Exception as e:
                        logging.error(f"Error in {strat_name}: {e}")

                # -----------------------------------------------------------------
                # HARVEST PHASE 2: STANDARD STRATEGIES (Priority 2)
                # -----------------------------------------------------------------
                # Shuffle standard strategies
                current_standard = standard_strategies.copy()
                if ml_strategy.model_loaded and ml_signal:
                    # Add ML signal as a candidate if it exists
                    current_standard.append(ml_strategy)
                random.shuffle(current_standard)

                for strat in current_standard:
                    strat_name = strat.__class__.__name__
                    signal = None

                    # (SMT needs master_mnq_df, ML needs ml_signal, others use new_df)
                    if strat_name == "MLPhysicsStrategy":
                        signal = ml_signal
                    elif strat_name == "SMTStrategy":
                        try:
                            signal = strat.on_bar(new_df, master_mnq_df)
                        except Exception as e:
                            logging.error(f"Error in {strat_name}: {e}")
                    else:
                        try:
                            signal = strat.on_bar(new_df)
                        except Exception as e:
                            logging.error(f"Error in {strat_name}: {e}")

                    if signal:
                        # ==========================================
                        # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                        # ==========================================
                        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                        old_sl = signal.get('sl_dist', 4.0)
                        old_tp = signal.get('tp_dist', 6.0)

                        if 'sl_dist' not in signal or 'tp_dist' not in signal:
                            logging.warning(f"⚠️ {strat_name} missing sl_dist/tp_dist, using defaults")

                        signal['sl_dist'] = old_sl * sl_mult
                        signal['tp_dist'] = old_tp * tp_mult

                        # Enforce minimums
                        MIN_SL = 4.0
                        MIN_TP = 6.0
                        if signal['sl_dist'] < MIN_SL:
                            logging.warning(f"⚠️ SL too tight ({signal['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                            signal['sl_dist'] = MIN_SL
                        if signal['tp_dist'] < MIN_TP:
                            logging.warning(f"⚠️ TP too tight ({signal['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                            signal['tp_dist'] = MIN_TP

                        if sl_mult != 1.0 or tp_mult != 1.0:
                            logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")

                        # Enforce HTF range fade directional restriction
                        # if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                        #    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                        #    continue

                        # Add to candidate list (Priority 2 = STANDARD)
                        candidate_signals.append((2, strat, signal, strat_name))

                        # Log as candidate
                        event_logger.log_strategy_signal(
                            strategy_name=signal.get('strategy', strat_name),
                            side=signal['side'],
                            tp_dist=signal.get('tp_dist', 6.0),
                            sl_dist=signal.get('sl_dist', 4.0),
                            price=current_price,
                            additional_info={"status": "CANDIDATE", "priority": "STANDARD"}
                        )
                        logging.info(f"📊 CANDIDATE (STANDARD): {strat_name} {signal['side']} @ {current_price:.2f}")

                # -----------------------------------------------------------------
                # SELECTION PHASE: Process candidates by priority until one passes
                # -----------------------------------------------------------------
                candidate_signals.sort(key=lambda x: x[0])

                signal_executed = False
                for priority, strat, sig, strat_name in candidate_signals:
                    signal = sig
                    priority_label = "FAST" if priority == 1 else "STANDARD"

                    # === 1. PREPARE THE RESCUE TICKET (OPPOSITE SIDE) ===
                    is_rescued = False

                    # Determine the OPPOSITE direction
                    original_side = signal['side']
                    rescue_side = 'SHORT' if original_side == 'LONG' else 'LONG'

                    # Fetch potential rescue signal for the OPPOSITE side
                    potential_rescue = continuation_manager.get_active_continuation_signal(
                        new_df, current_time, rescue_side
                    )

                    logging.info(f"🔍 EVALUATING {priority_label}: {strat_name} {original_side} | Rescue Available ({rescue_side}): {potential_rescue is not None}")

                    # === 2. TARGET FEASIBILITY (Physics Check - No Rescue) ===
                    is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                        entry_price=current_price, side=signal['side'], tp_distance=signal.get('tp_dist', 6.0), df_1m=new_df
                    )
                    if not is_feasible:
                        logging.info(f"⛔ Signal ignored ({priority_label}): {feasibility_reason}")
                        continue

                    # ==========================================
                    # LAYER 2: FILTER GAUNTLET (Safe Rescue Logic)
                    # ==========================================

                    # --- Helper Logic to Trigger Rescue ---
                    def try_rescue_trigger(block_reason, filter_name):
                        nonlocal signal, is_rescued, potential_rescue
                        # RESCUE LOGIC: Only flip if we have a ticket AND the block isn't a "Hard Stop"
                        if potential_rescue and not is_rescued:
                            logging.info(f"♻️ RESCUE FLIP: Blocked by {filter_name} ({block_reason}). Flipping to {potential_rescue['strategy']} ({potential_rescue['side']})")
                            signal = potential_rescue  # FLIP TO OPPOSITE SIGNAL
                            is_rescued = True
                            potential_rescue = None  # Ticket used
                            return True
                        else:
                            logging.info(f"⛔ BLOCKED by {filter_name}: {block_reason}")
                            return False
                    # ---------------------------------------

                    # 1. Rejection / Bias (SAFE TO RESCUE: Aligning with Bias)
                    rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                    range_bias_blocked = (allowed_chop_side is not None and signal['side'] != allowed_chop_side)

                    if rej_blocked or range_bias_blocked:
                        reason = rej_reason if rej_blocked else f"Opposite HTF Range Bias ({allowed_chop_side})"
                        if not try_rescue_trigger(reason, "Rejection/Bias"): continue

                    # 2. Directional Loss Blocker (SAFE TO RESCUE: Aligning with Performance)
                    dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                    if dir_blocked:
                        if not try_rescue_trigger(dir_reason, "DirectionalLoss"): continue

                    # 3. Impulse Filter (SAFE TO RESCUE: Aligning with Momentum)
                    impulse_blocked, impulse_reason = impulse_filter.should_block_trade(signal['side'])
                    if impulse_blocked:
                        if not try_rescue_trigger(impulse_reason, "ImpulseFilter"): continue

                    # 4. Regime Structure Blocker (EQH/EQL)
                    # 🛑 HARD STOP: DO NOT RESCUE 🛑
                    # If we are at EQH/EQL, a flip is dangerous because it could be a breakout.
                    regime_blocked, regime_reason = regime_blocker.should_block_trade(signal['side'], current_price)
                    if regime_blocked:
                        # Log and Die. No Rescue.
                        logging.info(f"⛔ HARD STOP by RegimeBlocker (EQH/EQL): {regime_reason} - No Rescue Allowed (Breakout Risk)")
                        continue

                    # 5. Independent System Checks (Trend/Macro)
                    # Note: If we just flipped to Rescue, we are now checking the NEW signal against these filters.

                    upgraded_blocked = False
                    upgraded_reasons = []

                    # HTF FVG (Memory Based)
                    fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(signal['side'], current_price, None, None, tp_dist=signal.get('tp_dist', 15.0))
                    if fvg_blocked: upgraded_reasons.append(f"FVG: {fvg_reason}")

                    # Macro Structure Trend (SAFE TO RESCUE: Aligning with Macro Trend)
                    struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                    if struct_blocked: upgraded_reasons.append(f"Structure: {struct_reason}")

                    bank_blocked, bank_reason = bank_filter.should_block_trade(signal['side'])
                    if bank_blocked: upgraded_reasons.append(f"Bank: {bank_reason}")

                    # Trend Check
                    upg_trend_blocked, upg_trend_reason = trend_filter.should_block_trade(new_df, signal['side'])
                    if upg_trend_blocked: upgraded_reasons.append(f"Trend: {upg_trend_reason}")

                    if upgraded_reasons: upgraded_blocked = True

                    # Legacy Check
                    legacy_blocked, legacy_reason = legacy_filters.check_trend(new_df, signal['side'])

                    # Arbitration
                    final_blocked = False
                    final_reason = ""
                    if legacy_blocked and upgraded_blocked:
                        final_blocked = True; final_reason = f"Unanimous: {legacy_reason} & {upgraded_reasons}"
                    elif not legacy_blocked and upgraded_blocked:
                        arb = filter_arbitrator.arbitrate(new_df, signal['side'], False, "", True, "|".join(upgraded_reasons), current_price, signal.get('tp_dist'), signal.get('sl_dist'))
                        if not arb.allow_trade: final_blocked = True; final_reason = arb.reason

                    if final_blocked:
                        # If we are ALREADY rescued, we have 'Diplomatic Immunity'
                        if is_rescued:
                            logging.info(f"🛡️ BYPASS Filters ({final_reason}): Rescued by {signal['strategy']}")
                        else:
                            # Attempt Rescue Trigger (Trend/Macro blocks are safe to flip)
                            if not try_rescue_trigger(final_reason, "FilterStack"): continue

                    # 6. Chop & Extension (Post-Arb)
                    vol_regime, _, _ = volatility_filter.get_regime(new_df)
                    chop_blocked, chop_reason = chop_filter.should_block_trade(signal['side'], rejection_filter.prev_day_pm_bias, current_price, "NEUTRAL", vol_regime)
                    if chop_blocked:
                        if is_rescued: logging.info(f"🛡️ BYPASS Chop: Rescued by {signal['strategy']}")
                        elif not try_rescue_trigger(chop_reason, "ChopFilter"): continue

                    ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                    if ext_blocked:
                        if is_rescued: logging.info(f"🛡️ BYPASS Extension: Rescued by {signal['strategy']}")
                        elif not try_rescue_trigger(ext_reason, "ExtensionFilter"): continue

                    # 7. Volatility Guardrail (Physics - Apply to Rescued too)
                    should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)
                    if not should_trade:
                        # If physics fail, we can't trade.
                        logging.info(f"⛔ BLOCKED by Volatility Guardrail")
                        continue

                    signal['sl_dist'] = vol_adj['sl_dist']
                    signal['tp_dist'] = vol_adj['tp_dist']
                    if vol_adj.get('adjustment_applied', False): signal['size'] = vol_adj['size']

                    # === EXECUTION ===
                    strategy_results['executed'] = strat_name
                    logging.info(f"✅ {priority_label} EXEC: {signal['strategy']} ({signal['side']})")

                    # ... [Remaining Execution Code same as before] ...
                    # Close and Reverse logic...
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
                        sl_dist = signal.get('sl_dist', 4.0)
                        initial_stop = current_price - sl_dist if signal['side'] == 'LONG' else current_price + sl_dist
                        active_trade = {
                            'strategy': signal['strategy'],
                            'side': signal['side'],
                            'entry_price': current_price,
                            'entry_bar': bar_count,
                            'bars_held': 0,
                            'tp_dist': signal['tp_dist'],
                            'sl_dist': sl_dist,
                            'size': signal.get('size', 5),
                            'stop_order_id': client._active_stop_order_id,
                            'current_stop_price': initial_stop,
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

                                # ALWAYS ensure sl_dist/tp_dist are set (fix for missing values)
                                old_sl = sig.get('sl_dist', 4.0)
                                old_tp = sig.get('tp_dist', 6.0)

                                # Warn if strategy didn't set these
                                if 'sl_dist' not in sig or 'tp_dist' not in sig:
                                    logging.warning(f"⚠️ {s_name} missing sl_dist/tp_dist, using defaults")

                                # Apply Multipliers
                                sig['sl_dist'] = old_sl * sl_mult
                                sig['tp_dist'] = old_tp * tp_mult

                                # Enforce minimums to prevent dangerously tight stops
                                MIN_SL = 4.0  # 16 ticks minimum
                                MIN_TP = 6.0  # 24 ticks minimum (1.5:1 RR)
                                if sig['sl_dist'] < MIN_SL:
                                    logging.warning(f"⚠️ SL too tight ({sig['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                                    sig['sl_dist'] = MIN_SL
                                if sig['tp_dist'] < MIN_TP:
                                    logging.warning(f"⚠️ TP too tight ({sig['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                                    sig['tp_dist'] = MIN_TP

                                if sl_mult != 1.0 or tp_mult != 1.0:
                                    logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{sig['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{sig['tp_dist']:.2f} (x{tp_mult})")
                                # ==========================================

                                # Enforce HTF range fade directional restriction
                                if allowed_chop_side is not None and sig['side'] != allowed_chop_side:
                                    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {sig['side']} vs Allowed {allowed_chop_side}")
                                    del pending_loose_signals[s_name]
                                    continue

                                # ==========================================
                                # LAYER 1: TARGET FEASIBILITY CHECK (Master Gate)
                                # ==========================================
                                # The market condition check (chop) already happened globally.
                                # Now check if the TARGET is realistic before wasting filter cycles.
                                is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                                    entry_price=current_price,
                                    side=sig['side'],
                                    tp_distance=sig.get('tp_dist', 6.0),
                                    df_1m=new_df
                                )
                                if not is_feasible:
                                    logging.info(f"⛔ Signal ignored (LOOSE): {feasibility_reason}")
                                    event_logger.log_filter_check("ChopFeasibility", sig['side'], False, feasibility_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFeasibility", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # ==========================================
                                # LAYER 2: SIGNAL QUALITY FILTERS
                                # ==========================================
                                # Re-check filters
                                rej_blocked, rej_reason = rejection_filter.should_block_trade(sig['side'])
                                if rej_blocked:
                                    event_logger.log_rejection_block("RejectionFilter", sig['side'], rej_reason or "Rejection bias")
                                    del pending_loose_signals[s_name]; continue

                                # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                                dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(sig['side'], current_time)
                                if dir_blocked:
                                    event_logger.log_filter_check("DirectionalLossBlocker", sig['side'], False, dir_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("DirectionalLossBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Impulse Filter (Prevent catching falling knife / fading rocket ship)
                                impulse_blocked, impulse_reason = impulse_filter.should_block_trade(sig['side'])
                                if impulse_blocked:
                                    event_logger.log_filter_check("ImpulseFilter", sig['side'], False, impulse_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ImpulseFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

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
                                    event_logger.log_filter_check("HTF_FVG", sig['side'], False, fvg_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("HTF_FVG", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # === [FIX 1] UPDATED BLOCKER CHECK ===
                                struct_blocked, struct_reason = structure_blocker.should_block_trade(sig['side'], current_price)
                                if struct_blocked:
                                    logging.info(f"🚫 {struct_reason}")
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], False, struct_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))
                                # Regime Structure Blocker (EQH/EQL with regime tolerance)
                                regime_blocked, regime_reason = regime_blocker.should_block_trade(sig['side'], current_price)
                                if regime_blocked:
                                    logging.info(f"🚫 {regime_reason}")
                                    event_logger.log_filter_check("RegimeBlocker", sig['side'], False, regime_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("RegimeBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))
                                # Penalty Box Blocker (Fixed 5.0pt tolerance + 3-bar decay)
                                penalty_blocked, penalty_reason = penalty_blocker.should_block_trade(sig['side'], current_price)
                                if penalty_blocked:
                                    logging.info(f"🚫 {penalty_reason}")
                                    event_logger.log_filter_check("PenaltyBoxBlocker", sig['side'], False, penalty_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("PenaltyBoxBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))
                                mem_blocked, mem_reason = memory_sr.should_block_trade(sig['side'], current_price)
                                if mem_blocked:
                                    logging.info(f"🚫 {mem_reason}")
                                    event_logger.log_filter_check("MemorySR", sig['side'], False, mem_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("MemorySR", sig['side'], True, strategy=sig.get('strategy', s_name))
                                # =====================================

                                # Determine if this is a Range Fade setup (used for filter bypasses)
                                is_range_fade = (allowed_chop_side is not None and sig['side'] == allowed_chop_side)

                                # === DUAL-FILTER TREND CHECK ===
                                legacy_trend_blocked, legacy_trend_reason = legacy_filters.check_trend(new_df, sig['side'])
                                upgraded_trend_blocked, upgraded_trend_reason = trend_filter.should_block_trade(new_df, sig['side'], is_range_fade=is_range_fade)

                                if legacy_trend_blocked != upgraded_trend_blocked:
                                    arb_result = filter_arbitrator.arbitrate(
                                        df=new_df, side=sig['side'],
                                        legacy_blocked=legacy_trend_blocked, legacy_reason=legacy_trend_reason or "",
                                        upgraded_blocked=upgraded_trend_blocked, upgraded_reason=upgraded_trend_reason or "",
                                        current_price=current_price,
                                        tp_dist=sig.get('tp_dist'), sl_dist=sig.get('sl_dist')
                                    )
                                    trend_blocked = not arb_result.allow_trade
                                    trend_reason = arb_result.reason
                                else:
                                    trend_blocked = upgraded_trend_blocked
                                    trend_reason = upgraded_trend_reason
                                    # Log when both agree (so we know dual-filter is running)
                                    if trend_blocked:
                                        logging.info(f"🛡️ DUAL-FILTER: Both BLOCK {sig['side']} | reason: {trend_reason}")
                                    else:
                                        logging.info(f"✅ DUAL-FILTER: Both ALLOW {sig['side']} trend check")

                                trend_state = ("Strong Bearish" if (trend_reason and "Bearish" in str(trend_reason))
                                               else ("Strong Bullish" if (trend_reason and "Bullish" in str(trend_reason))
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
                                    event_logger.log_filter_check("ChopFilter", sig['side'], False, chop_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                ext_blocked, ext_reason = extension_filter.should_block_trade(sig['side'])
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], False, ext_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Trend Filter (already checked above with is_range_fade)
                                if trend_blocked:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], False, trend_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Volatility & Guardrail Check
                                # We pass the Gemini-modified params (sig['sl_dist']) into the filter.
                                # The filter applies Guardrails + Rounding.
                                should_trade, vol_adj = check_volatility(new_df, sig.get('sl_dist', 4.0), sig.get('tp_dist', 6.0), base_size=5)

                                if not should_trade:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], False, "Volatility check failed", strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # === APPLY SANITIZED VALUES ===
                                # Always update to the rounded version (e.g. 4.52 -> 4.50)
                                # regardless of whether a 'regime' change happened.
                                sig['sl_dist'] = vol_adj['sl_dist']
                                sig['tp_dist'] = vol_adj['tp_dist']

                                # Only apply SIZE adjustment if the regime explicitly demands it (Low Vol)
                                if vol_adj.get('adjustment_applied', False):
                                    sig['size'] = vol_adj['size']
                                    event_logger.log_trade_modified(
                                        "VolatilityAdjustment",
                                        sig.get('tp_dist', 6.0),
                                        vol_adj['tp_dist'],
                                        f"Volatility/Guardrail adjustment (Regime: {vol_adj['regime']})"
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
                                    sl_dist = sig.get('sl_dist', 4.0)  # Standard default, NOT tp_dist
                                    initial_stop = current_price - sl_dist if sig['side'] == 'LONG' else current_price + sl_dist
                                    active_trade = {
                                        'strategy': s_name,
                                        'side': sig['side'],
                                        'entry_price': current_price,
                                        'entry_bar': bar_count,
                                        'bars_held': 0,
                                        'tp_dist': sig['tp_dist'],
                                        'sl_dist': sl_dist,  # Store SL for consistency
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

                                        # ALWAYS ensure sl_dist/tp_dist are set (fix for missing values)
                                        old_sl = signal.get('sl_dist', 4.0)
                                        old_tp = signal.get('tp_dist', 6.0)

                                        # Warn if strategy didn't set these
                                        if 'sl_dist' not in signal or 'tp_dist' not in signal:
                                            logging.warning(f"⚠️ {s_name} missing sl_dist/tp_dist, using defaults")

                                        # Apply Multipliers
                                        signal['sl_dist'] = old_sl * sl_mult
                                        signal['tp_dist'] = old_tp * tp_mult

                                        # Enforce minimums to prevent dangerously tight stops
                                        MIN_SL = 4.0  # 16 ticks minimum
                                        MIN_TP = 6.0  # 24 ticks minimum (1.5:1 RR)
                                        if signal['sl_dist'] < MIN_SL:
                                            logging.warning(f"⚠️ SL too tight ({signal['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                                            signal['sl_dist'] = MIN_SL
                                        if signal['tp_dist'] < MIN_TP:
                                            logging.warning(f"⚠️ TP too tight ({signal['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                                            signal['tp_dist'] = MIN_TP

                                        if sl_mult != 1.0 or tp_mult != 1.0:
                                            logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")
                                        # ==========================================

                                        # Enforce HTF range fade directional restriction
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
                                            event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], False, dir_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        tp_dist = signal.get('tp_dist', 15.0)

                                        effective_tp_dist = tp_dist
                                        if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                                            effective_tp_dist = tp_dist * 0.5
                                            logging.info(f"🔓 RELAXING FVG CHECK (Loose): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                                        fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                            signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                                        )
                                        if fvg_blocked:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # === [FIX 2] UPDATED BLOCKER CHECK ===
                                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                                        if struct_blocked:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        # Regime Structure Blocker (EQH/EQL with regime tolerance)
                                        regime_blocked, regime_reason = regime_blocker.should_block_trade(signal['side'], current_price)
                                        if regime_blocked:
                                            event_logger.log_filter_check("RegimeBlocker", signal['side'], False, regime_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("RegimeBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        # Penalty Box Blocker (Fixed 5.0pt tolerance + 3-bar decay)
                                        penalty_blocked, penalty_reason = penalty_blocker.should_block_trade(signal['side'], current_price)
                                        if penalty_blocked:
                                            event_logger.log_filter_check("PenaltyBoxBlocker", signal['side'], False, penalty_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("PenaltyBoxBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        mem_blocked, mem_reason = memory_sr.should_block_trade(signal['side'], current_price)
                                        if mem_blocked:
                                            event_logger.log_filter_check("MemorySR", signal['side'], False, mem_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("MemorySR", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        # =====================================

                                        # Determine if this is a Range Fade setup (used for filter bypasses)
                                        is_range_fade = (allowed_chop_side is not None and signal['side'] == allowed_chop_side)

                                        # === DUAL-FILTER TREND CHECK ===
                                        legacy_trend_blocked, legacy_trend_reason = legacy_filters.check_trend(new_df, signal['side'])
                                        upgraded_trend_blocked, upgraded_trend_reason = trend_filter.should_block_trade(new_df, signal['side'], is_range_fade=is_range_fade)

                                        if legacy_trend_blocked != upgraded_trend_blocked:
                                            arb_result = filter_arbitrator.arbitrate(
                                                df=new_df, side=signal['side'],
                                                legacy_blocked=legacy_trend_blocked, legacy_reason=legacy_trend_reason or "",
                                                upgraded_blocked=upgraded_trend_blocked, upgraded_reason=upgraded_trend_reason or "",
                                                current_price=current_price,
                                                tp_dist=signal.get('tp_dist'), sl_dist=signal.get('sl_dist')
                                            )
                                            trend_blocked = not arb_result.allow_trade
                                            trend_reason = arb_result.reason
                                        else:
                                            trend_blocked = upgraded_trend_blocked
                                            trend_reason = upgraded_trend_reason
                                            # Log when both agree (so we know dual-filter is running)
                                            if trend_blocked:
                                                logging.info(f"🛡️ DUAL-FILTER: Both BLOCK {signal['side']} | reason: {trend_reason}")
                                            else:
                                                logging.info(f"✅ DUAL-FILTER: Both ALLOW {signal['side']} trend check")

                                        trend_state = ("Strong Bearish" if (trend_reason and "Bearish" in str(trend_reason))
                                                       else ("Strong Bullish" if (trend_reason and "Bullish" in str(trend_reason))
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
                                            event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                        if ext_blocked:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # Trend Filter (already checked above with is_range_fade)
                                        if trend_blocked:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], False, trend_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # Volatility & Guardrail Check
                                        # We pass the Gemini-modified params (signal['sl_dist']) into the filter.
                                        # The filter applies Guardrails + Rounding.
                                        should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)

                                        if not should_trade:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed", strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # === APPLY SANITIZED VALUES ===
                                        # Always update to the rounded version (e.g. 4.52 -> 4.50)
                                        # regardless of whether a 'regime' change happened.
                                        signal['sl_dist'] = vol_adj['sl_dist']
                                        signal['tp_dist'] = vol_adj['tp_dist']

                                        # Only apply SIZE adjustment if the regime explicitly demands it (Low Vol)
                                        if vol_adj.get('adjustment_applied', False):
                                            signal['size'] = vol_adj['size']
                                            event_logger.log_trade_modified(
                                                "VolatilityAdjustment",
                                                signal.get('tp_dist', 6.0),
                                                vol_adj['tp_dist'],
                                                f"Volatility/Guardrail adjustment (Regime: {vol_adj['regime']})"
                                            )

                                        # Log as QUEUED for UI visibility
                                        event_logger.log_strategy_signal(
                                            strategy_name=signal.get('strategy', s_name),
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist', 0),
                                            sl_dist=signal.get('sl_dist', 0),
                                            price=current_price,
                                            additional_info={"status": "QUEUED", "priority": "LOOSE"}
                                        )
                                        logging.info(f"🕐 Queuing {s_name} signal")
                                        pending_loose_signals[s_name] = {'signal': signal, 'bar_count': 0}
                                except Exception as e:
                                    logging.error(f"Error in {s_name}: {e}")

            await asyncio.sleep(2.0)  # Slower polling to avoid Topstep rate limits

        except KeyboardInterrupt:
            print("\nBot Stopped by User.")
            break
        except Exception as e:
            logging.error(f"Main Loop Error: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    # Run the async bot with asyncio
    asyncio.run(run_bot())

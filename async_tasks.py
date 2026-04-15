"""
Independent Async Tasks for Bot

This module provides independent async tasks that run concurrently with the main
strategy loop. These tasks ensure critical operations (heartbeat, position sync)
are never blocked by heavy strategy calculations.
"""
import asyncio
import logging
import datetime
from typing import Any, Callable, Optional


async def heartbeat_task(client, interval: int = 60):
    """
    Independent heartbeat task that validates session and logs status + price.
    """
    heartbeat_count = 0

    while True:
        try:
            heartbeat_count += 1

            # 1. Validate session token
            is_valid = await client.async_validate_session()

            # 2. Get Last Known Price (from client memory)
            price_display = "0.00"
            if hasattr(client, 'cached_df') and not client.cached_df.empty:
                try:
                    # Get the close of the most recent bar
                    last_close = client.cached_df.iloc[-1]['close']
                    price_display = f"{last_close:.2f}"
                except Exception:
                    price_display = "N/A"

            # 3. Format Log Message
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            status = "✅" if is_valid else "❌"

            # UPDATED LOG FORMAT: Includes "Price: ..." so the UI can read it
            logging.info(f"💓 Heartbeat #{heartbeat_count}: {current_time} | Session: {status} | Price: {price_display}")

            if not is_valid:
                logging.warning("⚠️ Heartbeat: Session validation failed!")

        except Exception as e:
            logging.error(f"❌ Heartbeat task error: {e}")

        # Wait for next heartbeat
        await asyncio.sleep(interval)


async def position_sync_task(
    client,
    interval: int = 30,
    on_position_sync: Optional[Callable[[dict], Any]] = None,
):
    """
    Independent position sync task that fetches broker position every 'interval' seconds.
    """
    sync_count = 0
    last_logged_signature = None
    last_persisted_signature = None

    while True:
        try:
            sync_count += 1

            # Fetch position from broker
            broker_pos = await client.async_get_position(require_open_pnl=True)

            # Update shadow state
            client._local_position = broker_pos.copy()

            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            side = broker_pos.get('side') or 'FLAT'
            size = int(broker_pos.get('size', 0) or 0)
            avg_price = float(broker_pos.get('avg_price', 0.0) or 0.0)
            open_pnl = broker_pos.get('open_pnl')
            open_pnl_value = round(float(open_pnl), 2) if isinstance(open_pnl, (int, float)) else None
            state_signature = (side, size, round(avg_price, 4), open_pnl_value)

            persist_signature = (side, size, round(avg_price, 4))

            # Log immediately on first sync and any broker-side position change.
            should_log = sync_count == 1 or state_signature != last_logged_signature or sync_count % 10 == 0
            if should_log:
                if side == 'FLAT':
                    logging.info(f"🔄 Position Sync #{sync_count}: {current_time} | Status: {side}")
                else:
                    if open_pnl_value is not None:
                        logging.info(
                            f"🔄 Position Sync #{sync_count}: {current_time} | "
                            f"{side} {size} @ {avg_price:.2f} | OpenPnL: ${open_pnl_value:.2f}"
                        )
                    else:
                        logging.info(f"🔄 Position Sync #{sync_count}: {current_time} | {side} {size} @ {avg_price:.2f}")
                last_logged_signature = state_signature

            callback = on_position_sync or getattr(client, "_persist_runtime_state", None)
            should_persist = sync_count == 1 or persist_signature != last_persisted_signature
            if callable(callback) and should_persist:
                try:
                    callback_result = callback(broker_pos.copy())
                    if asyncio.iscoroutine(callback_result):
                        await callback_result
                    last_persisted_signature = persist_signature
                except Exception as exc:
                    logging.warning("Position sync persistence callback failed: %s", exc)

        except Exception as e:
            logging.error(f"❌ Position sync task error: {e}")

        # Wait for next sync
        await asyncio.sleep(interval)


async def market_data_monitor_task(market_manager, name: str, update_callback):
    """
    Monitor market data stream and trigger updates when new bars arrive.
    """
    last_bar_time = None

    while True:
        try:
            # Get latest bar from stream
            stream = market_manager.streams.get(name)
            if stream:
                latest_bar = stream.get_latest_bar()

                # Check if we have a new bar
                if latest_bar and latest_bar['ts'] != last_bar_time:
                    last_bar_time = latest_bar['ts']
                    # Trigger update callback with new bar
                    await update_callback(name, latest_bar)

        except Exception as e:
            logging.error(f"❌ Market data monitor error for {name}: {e}")

        # Check for new bars every 100ms
        await asyncio.sleep(0.1)


async def htf_structure_task(client, htf_filter, interval: int = 60):
    """
    Independent task to fetch HTF data and update FVG filter memory.
    """
    logging.info("🚀 HTF Structure Background Task Started")

    while True:
        try:
            # Run blocking fetch calls in a separate thread
            df_1h = await asyncio.to_thread(client.fetch_custom_bars, lookback_bars=240, minutes_per_bar=60)
            df_4h = await asyncio.to_thread(client.fetch_custom_bars, lookback_bars=200, minutes_per_bar=240)

            # Update the filter with new structures
            if not df_1h.empty and not df_4h.empty:
                htf_filter.update_structure_data(df_1h, df_4h)

        except Exception as e:
            logging.error(f"❌ HTF structure task error: {e}")

        # Wait for next update interval
        await asyncio.sleep(interval)


async def sentiment_monitor_task(sentiment_service):
    """
    Run the Truth Social sentiment polling loop without blocking the bot.
    """
    if sentiment_service is None:
        return
    while True:
        try:
            await sentiment_service.run_forever()
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.error("❌ Sentiment monitor task error: %s", exc)
            await asyncio.sleep(5)

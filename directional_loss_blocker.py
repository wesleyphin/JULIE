import datetime
import logging
from typing import Optional, Tuple

from event_logger import event_logger


class DirectionalLossBlocker:
    """
    Blocks trading in a specific direction after 3 consecutive losing trades.

    - 3 consecutive losing LONG trades -> blocks LONG for 15 minutes
    - 3 consecutive losing SHORT trades -> blocks SHORT for 15 minutes

    Consecutive losses reset when:
    - A winning trade occurs in that direction
    - The 15-minute block expires (resets the counter)
    """

    def __init__(self, consecutive_loss_limit: int = 3, block_minutes: int = 15):
        self.consecutive_loss_limit = consecutive_loss_limit
        self.block_minutes = block_minutes

        # Track consecutive losses per direction
        self.long_consecutive_losses = 0
        self.short_consecutive_losses = 0

        # Track when blocks expire
        self.long_blocked_until: Optional[datetime.datetime] = None
        self.short_blocked_until: Optional[datetime.datetime] = None

    def record_trade_result(self, side: str, pnl: float, current_time: Optional[datetime.datetime] = None):
        """
        Record the result of a closed trade.

        Args:
            side: 'LONG' or 'SHORT' - the direction of the closed trade
            pnl: The profit/loss of the trade (negative = loss)
            current_time: Current timestamp (optional, defaults to now)
        """
        if current_time is None:
            current_time = datetime.datetime.now()

        is_loss = pnl < 0

        if side == 'LONG':
            if is_loss:
                self.long_consecutive_losses += 1
                logging.info(f"📉 LONG loss recorded. Consecutive LONG losses: {self.long_consecutive_losses}/{self.consecutive_loss_limit}")

                if self.long_consecutive_losses >= self.consecutive_loss_limit:
                    self.long_blocked_until = current_time + datetime.timedelta(minutes=self.block_minutes)
                    logging.warning(f"🚫 LONG DIRECTION BLOCKED: {self.consecutive_loss_limit} consecutive losses. Blocked until {self.long_blocked_until.strftime('%H:%M:%S')}")

                    event_logger.log_filter_check(
                        "DirectionalLossBlocker",
                        "LONG",
                        False,
                        f"Triggered: {self.consecutive_loss_limit} consecutive LONG losses, blocked for {self.block_minutes} min"
                    )
            else:
                # Win resets consecutive losses
                if self.long_consecutive_losses > 0:
                    logging.info(f"✅ LONG win! Resetting consecutive LONG losses (was {self.long_consecutive_losses})")
                self.long_consecutive_losses = 0

        elif side == 'SHORT':
            if is_loss:
                self.short_consecutive_losses += 1
                logging.info(f"📉 SHORT loss recorded. Consecutive SHORT losses: {self.short_consecutive_losses}/{self.consecutive_loss_limit}")

                if self.short_consecutive_losses >= self.consecutive_loss_limit:
                    self.short_blocked_until = current_time + datetime.timedelta(minutes=self.block_minutes)
                    logging.warning(f"🚫 SHORT DIRECTION BLOCKED: {self.consecutive_loss_limit} consecutive losses. Blocked until {self.short_blocked_until.strftime('%H:%M:%S')}")

                    event_logger.log_filter_check(
                        "DirectionalLossBlocker",
                        "SHORT",
                        False,
                        f"Triggered: {self.consecutive_loss_limit} consecutive SHORT losses, blocked for {self.block_minutes} min"
                    )
            else:
                # Win resets consecutive losses
                if self.short_consecutive_losses > 0:
                    logging.info(f"✅ SHORT win! Resetting consecutive SHORT losses (was {self.short_consecutive_losses})")
                self.short_consecutive_losses = 0

    def should_block_trade(self, direction: str, current_time: Optional[datetime.datetime] = None) -> Tuple[bool, str]:
        """
        Check if a trade in the given direction should be blocked.

        Args:
            direction: 'LONG' or 'SHORT' (or 'BUY'/'SELL' which get mapped)
            current_time: Current timestamp (optional, defaults to now)

        Returns:
            Tuple of (is_blocked, reason_string)
        """
        if current_time is None:
            current_time = datetime.datetime.now()

        # Normalize direction names
        if direction == 'BUY':
            direction = 'LONG'
        elif direction == 'SELL':
            direction = 'SHORT'

        if direction == 'LONG' and self.long_blocked_until is not None:
            if current_time < self.long_blocked_until:
                remaining = (self.long_blocked_until - current_time).total_seconds() / 60
                reason = f"LONG blocked for {remaining:.1f} more minutes ({self.consecutive_loss_limit} consecutive losses)"
                logging.info(f"⛔️ DirectionalLossBlocker: {reason}")
                return True, reason
            else:
                # Block expired - reset
                logging.info(f"🔓 LONG block expired. Resetting consecutive loss counter.")
                self.long_blocked_until = None
                self.long_consecutive_losses = 0

        elif direction == 'SHORT' and self.short_blocked_until is not None:
            if current_time < self.short_blocked_until:
                remaining = (self.short_blocked_until - current_time).total_seconds() / 60
                reason = f"SHORT blocked for {remaining:.1f} more minutes ({self.consecutive_loss_limit} consecutive losses)"
                logging.info(f"⛔️ DirectionalLossBlocker: {reason}")
                return True, reason
            else:
                # Block expired - reset
                logging.info(f"🔓 SHORT block expired. Resetting consecutive loss counter.")
                self.short_blocked_until = None
                self.short_consecutive_losses = 0

        return False, ""

    def get_status(self) -> dict:
        """Get current status of the blocker for monitoring/UI."""
        now = datetime.datetime.now()

        long_remaining = None
        if self.long_blocked_until and now < self.long_blocked_until:
            long_remaining = (self.long_blocked_until - now).total_seconds() / 60

        short_remaining = None
        if self.short_blocked_until and now < self.short_blocked_until:
            short_remaining = (self.short_blocked_until - now).total_seconds() / 60

        return {
            'long_consecutive_losses': self.long_consecutive_losses,
            'short_consecutive_losses': self.short_consecutive_losses,
            'long_blocked': long_remaining is not None,
            'long_blocked_remaining_min': long_remaining,
            'short_blocked': short_remaining is not None,
            'short_blocked_remaining_min': short_remaining,
        }

    def reset_daily(self):
        """Reset all counters and blocks for a new trading day."""
        self.long_consecutive_losses = 0
        self.short_consecutive_losses = 0
        self.long_blocked_until = None
        self.short_blocked_until = None
        logging.info("🔄 DirectionalLossBlocker reset for new day")

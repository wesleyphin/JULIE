import datetime
import logging
from typing import Optional, Tuple

from event_logger import event_logger


class DirectionalLossBlocker:
    """
    Blocks trading in a specific direction after consecutive losing trades.

    Rules:
    - 3 consecutive losing trades in one direction -> blocks that direction for 15 minutes
    - 4 consecutive losing trades in one direction -> REVERSES bias for rest of quarter
      (only allows trades in the OPPOSITE direction)

    Consecutive losses reset when:
    - A winning trade occurs in that direction
    - The 15-minute block expires (resets the counter)
    - Quarter changes (resets bias reversal)
    """

    def __init__(self, consecutive_loss_limit: int = 3, block_minutes: int = 15, bias_reversal_limit: int = 4):
        self.consecutive_loss_limit = consecutive_loss_limit
        self.block_minutes = block_minutes
        self.bias_reversal_limit = bias_reversal_limit

        # Track consecutive losses per direction
        self.long_consecutive_losses = 0
        self.short_consecutive_losses = 0

        # Track when blocks expire (15-min block)
        self.long_blocked_until: Optional[datetime.datetime] = None
        self.short_blocked_until: Optional[datetime.datetime] = None

        # Track bias reversal (4 consecutive losses)
        # If reversed_bias = 'LONG', it means LONGs lost 4 times, so ONLY take SHORTs
        # If reversed_bias = 'SHORT', it means SHORTs lost 4 times, so ONLY take LONGs
        self.reversed_bias: Optional[str] = None
        self.reversal_session: Optional[str] = None
        self.reversal_quarter: int = 0

    def get_session(self, hour: int) -> str:
        """Determine session from hour (ET)."""
        if 18 <= hour <= 23 or 0 <= hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        return 'CLOSED'

    def get_quarter(self, hour: int, minute: int, session: str) -> int:
        """Determine which quarter (1-4) based on Daye's Quarterly Theory."""
        if session == 'ASIA':
            h_adj = hour if hour >= 18 else hour + 24
            mins_since_start = (h_adj - 18) * 60 + minute
            quarter_length = 135
        elif session == 'LONDON':
            mins_since_start = (hour - 3) * 60 + minute
            quarter_length = 75
        elif session == 'NY_AM':
            mins_since_start = (hour - 8) * 60 + minute
            quarter_length = 60
        elif session == 'NY_PM':
            mins_since_start = (hour - 12) * 60 + minute
            quarter_length = 75
        else:
            return 0

        quarter = min(4, (mins_since_start // quarter_length) + 1)
        return quarter

    def update_quarter(self, current_time) -> None:
        """
        Call this each bar to check if quarter changed and reset bias reversal.

        Args:
            current_time: Current timestamp (pandas Timestamp or datetime)
        """
        # Handle pandas Timestamp
        if hasattr(current_time, 'hour'):
            hour = current_time.hour
            minute = current_time.minute
        else:
            hour = current_time.hour
            minute = current_time.minute

        current_session = self.get_session(hour)
        current_quarter = self.get_quarter(hour, minute, current_session) if current_session != 'CLOSED' else 0

        # Check if quarter or session changed - reset bias reversal
        if self.reversed_bias is not None:
            if current_session != self.reversal_session or current_quarter != self.reversal_quarter:
                logging.info(f"🔄 QUARTER CHANGED: {self.reversal_session} Q{self.reversal_quarter} -> {current_session} Q{current_quarter}")
                logging.info(f"🔓 BIAS REVERSAL RESET: Was blocking {self.reversed_bias}s due to 4 consecutive losses")
                self.reversed_bias = None
                self.reversal_session = None
                self.reversal_quarter = 0

    def record_trade_result(self, side: str, pnl: float, current_time=None):
        """
        Record the result of a closed trade.

        Args:
            side: 'LONG' or 'SHORT' - the direction of the closed trade
            pnl: The profit/loss of the trade (negative = loss)
            current_time: Current timestamp (optional, defaults to now)
        """
        if current_time is None:
            current_time = datetime.datetime.now()

        # Get current session/quarter for bias reversal tracking
        if hasattr(current_time, 'hour'):
            hour = current_time.hour
            minute = current_time.minute
        else:
            hour = current_time.hour
            minute = current_time.minute

        current_session = self.get_session(hour)
        current_quarter = self.get_quarter(hour, minute, current_session) if current_session != 'CLOSED' else 0

        is_loss = pnl < 0

        if side == 'LONG':
            if is_loss:
                self.long_consecutive_losses += 1
                logging.info(f"📉 LONG loss recorded. Consecutive LONG losses: {self.long_consecutive_losses}")

                # Check for bias reversal (4 losses) - REVERSE to opposite direction
                if self.long_consecutive_losses >= self.bias_reversal_limit:
                    self.reversed_bias = 'LONG'
                    self.reversal_session = current_session
                    self.reversal_quarter = current_quarter
                    logging.warning(f"🔄 BIAS REVERSED TO SHORT: {self.bias_reversal_limit} consecutive LONG losses!")
                    logging.warning(f"🔄 ONLY SHORTS until end of {current_session} Q{current_quarter}")

                    event_logger.log_filter_check(
                        "DirectionalLossBlocker",
                        "LONG",
                        False,
                        f"BIAS REVERSED TO SHORT: {self.bias_reversal_limit} consecutive LONG losses"
                    )

                # Check for 15-min block (3 losses)
                elif self.long_consecutive_losses >= self.consecutive_loss_limit:
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
                logging.info(f"📉 SHORT loss recorded. Consecutive SHORT losses: {self.short_consecutive_losses}")

                # Check for bias reversal (4 losses) - REVERSE to opposite direction
                if self.short_consecutive_losses >= self.bias_reversal_limit:
                    self.reversed_bias = 'SHORT'
                    self.reversal_session = current_session
                    self.reversal_quarter = current_quarter
                    logging.warning(f"🔄 BIAS REVERSED TO LONG: {self.bias_reversal_limit} consecutive SHORT losses!")
                    logging.warning(f"🔄 ONLY LONGS until end of {current_session} Q{current_quarter}")

                    event_logger.log_filter_check(
                        "DirectionalLossBlocker",
                        "SHORT",
                        False,
                        f"BIAS REVERSED TO LONG: {self.bias_reversal_limit} consecutive SHORT losses"
                    )

                # Check for 15-min block (3 losses)
                elif self.short_consecutive_losses >= self.consecutive_loss_limit:
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

    def should_block_trade(self, direction: str, current_time=None) -> Tuple[bool, str]:
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

        # Check bias reversal FIRST (takes priority over 15-min block)
        # After 4 consecutive losses, bias is REVERSED - only opposite direction allowed
        if self.reversed_bias is not None:
            if direction == self.reversed_bias:
                opposite = 'SHORT' if direction == 'LONG' else 'LONG'
                reason = f"BIAS REVERSED to {opposite} only (4 consecutive {direction} losses) - until end of Q{self.reversal_quarter}"
                logging.info(f"🔄 DirectionalLossBlocker: {reason}")
                return True, reason

        # Check 15-minute time blocks
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
            'reversed_bias': self.reversed_bias,
            'reversal_quarter': self.reversal_quarter if self.reversed_bias else None,
        }

    def reset_daily(self):
        """Reset all counters and blocks for a new trading day."""
        self.long_consecutive_losses = 0
        self.short_consecutive_losses = 0
        self.long_blocked_until = None
        self.short_blocked_until = None
        self.reversed_bias = None
        self.reversal_session = None
        self.reversal_quarter = 0
        logging.info("🔄 DirectionalLossBlocker reset for new day")

    def get_state(self) -> dict:
        return {
            "long_consecutive_losses": self.long_consecutive_losses,
            "short_consecutive_losses": self.short_consecutive_losses,
            "long_blocked_until": self.long_blocked_until.isoformat() if self.long_blocked_until else None,
            "short_blocked_until": self.short_blocked_until.isoformat() if self.short_blocked_until else None,
            "reversed_bias": self.reversed_bias,
            "reversal_session": self.reversal_session,
            "reversal_quarter": self.reversal_quarter,
        }

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self.long_consecutive_losses = int(state.get("long_consecutive_losses", self.long_consecutive_losses))
        self.short_consecutive_losses = int(state.get("short_consecutive_losses", self.short_consecutive_losses))
        long_until = state.get("long_blocked_until")
        short_until = state.get("short_blocked_until")
        if long_until:
            try:
                self.long_blocked_until = datetime.datetime.fromisoformat(long_until)
            except Exception:
                pass
        if short_until:
            try:
                self.short_blocked_until = datetime.datetime.fromisoformat(short_until)
            except Exception:
                pass
        self.reversed_bias = state.get("reversed_bias", self.reversed_bias)
        self.reversal_session = state.get("reversal_session", self.reversal_session)
        self.reversal_quarter = state.get("reversal_quarter", self.reversal_quarter)

import datetime
import logging
from typing import Optional, Tuple


class BankLevelQuarterFilter:
    """Filter trades based on bank level rejection relative to prev PM, prev session, and midnight ORB."""

    BANK_GRID = 12.5  # $12.50 bank levels
    REJECTION_TOLERANCE = 0.5  # Points tolerance for rejection detection
    REJECTIONS_REQUIRED = 2  # Require at least 2 full candles to confirm / flip bias

    def __init__(self):
        self.reset()

    def reset(self):
        # === PREVIOUS PM SESSION (12:00-17:00 ET) ===
        self.prev_day_pm_high: Optional[float] = None
        self.prev_day_pm_low: Optional[float] = None
        self.current_pm_high: Optional[float] = None
        self.current_pm_low: Optional[float] = None

        # Bank levels relative to prev PM
        self.bank_below_prev_pm_low: Optional[float] = None
        self.bank_above_prev_pm_high: Optional[float] = None

        # === PREVIOUS SESSION ===
        self.prev_session_high: Optional[float] = None
        self.prev_session_low: Optional[float] = None
        self.curr_session_high: Optional[float] = None
        self.curr_session_low: Optional[float] = None
        self.last_session: Optional[str] = None

        # Bank levels relative to prev session
        self.bank_below_prev_session_low: Optional[float] = None
        self.bank_above_prev_session_high: Optional[float] = None

        # === MIDNIGHT ORB (00:00-00:15 ET) ===
        self.midnight_orb_high: Optional[float] = None
        self.midnight_orb_low: Optional[float] = None
        self.midnight_orb_set: bool = False

        # Bank levels relative to midnight ORB
        self.bank_below_orb_low: Optional[float] = None
        self.bank_above_orb_high: Optional[float] = None

        # === COMMON ===
        self.current_date: Optional[datetime.date] = None
        self.current_session: Optional[str] = None
        self.current_quarter: Optional[int] = None

        # Bias from bank level rejection (reset each quarter)
        self.bank_rejection_bias: Optional[str] = None
        self.rejection_source: Optional[str] = None  # Track which level triggered the bias

        # Pending state for 2-candle confirmation
        self.pending_dir: Optional[str] = None      # 'LONG' or 'SHORT'
        self.pending_source: Optional[str] = None   # description of level
        self.pending_count: int = 0

    # -------------------------------------------------
    # Utility helpers
    # -------------------------------------------------
    def _get_closest_bank_below(self, price: float) -> Optional[float]:
        """Get the $12.50 bank level closest below the price"""
        if price is None:
            return None
        return (price // self.BANK_GRID) * self.BANK_GRID

    def _get_closest_bank_above(self, price: float) -> Optional[float]:
        """Get the $12.50 bank level closest above the price"""
        if price is None:
            return None
        return ((price // self.BANK_GRID) + 1) * self.BANK_GRID

    def get_session(self, hour: int, minute: int = 0) -> str:
        """Determine session from hour (ET)"""
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
            if hour >= 18:
                mins_since_start = (hour - 18) * 60 + minute
            else:
                mins_since_start = (24 - 18 + hour) * 60 + minute
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

    def check_bank_rejection(self, high: float, low: float, close: float,
                             bank_level: Optional[float], direction: str) -> bool:
        """Check if bar shows rejection of a bank level."""
        if bank_level is None:
            return False

        if direction == 'BULLISH':
            # Wick through or tag slightly below the level, close back above
            if low < bank_level + self.REJECTION_TOLERANCE and close > bank_level:
                return True
        elif direction == 'BEARISH':
            # Wick through or tag slightly above the level, close back below
            if high > bank_level - self.REJECTION_TOLERANCE and close < bank_level:
                return True

        return False

    def _update_bank_levels(self):
        """Recalculate all bank levels based on current reference levels"""
        # Bank levels from prev PM
        self.bank_below_prev_pm_low = self._get_closest_bank_below(self.prev_day_pm_low)
        self.bank_above_prev_pm_high = self._get_closest_bank_above(self.prev_day_pm_high)

        # Bank levels from prev session
        self.bank_below_prev_session_low = self._get_closest_bank_below(self.prev_session_low)
        self.bank_above_prev_session_high = self._get_closest_bank_above(self.prev_session_high)

        # Bank levels from midnight ORB (only if set)
        if self.midnight_orb_set:
            self.bank_below_orb_low = self._get_closest_bank_below(self.midnight_orb_low)
            self.bank_above_orb_high = self._get_closest_bank_above(self.midnight_orb_high)

    # -------------------------------------------------
    # 2-candle confirmation logic for bank-level bias
    # -------------------------------------------------
    def _process_rejection(self, direction: str, source: str, current_quarter: int):
        """Apply 2-candle confirmation logic for bank-level bias."""
        if direction not in ('LONG', 'SHORT'):
            return

        # No existing bias: build initial bias
        if self.bank_rejection_bias is None:
            if self.pending_dir == direction:
                self.pending_count += 1
            else:
                self.pending_dir = direction
                self.pending_source = source
                self.pending_count = 1

            if self.pending_count >= self.REJECTIONS_REQUIRED:
                self.bank_rejection_bias = direction
                self.rejection_source = self.pending_source
                logging.info(
                    f"🎯 BANK BIAS CONFIRMED Q{current_quarter}: "
                    f"{direction} from {self.rejection_source} "
                    f"after {self.REJECTIONS_REQUIRED} rejection candles"
                )
                # Reset pending state
                self.pending_dir = None
                self.pending_source = None
                self.pending_count = 0
            return

        # There is an existing bias
        if direction == self.bank_rejection_bias:
            # Same-direction rejection reinforces current view, clear any flip attempts
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0
            return

        # Opposite rejection: candidate flip
        if self.pending_dir == direction:
            self.pending_count += 1
        else:
            self.pending_dir = direction
            self.pending_source = source
            self.pending_count = 1

        if self.pending_count >= self.REJECTIONS_REQUIRED:
            logging.info(
                f"🔁 BANK BIAS FLIP Q{current_quarter}: "
                f"{self.bank_rejection_bias} -> {direction} from {self.pending_source} "
                f"after {self.REJECTIONS_REQUIRED} opposite rejection candles"
            )
            self.bank_rejection_bias = direction
            self.rejection_source = self.pending_source
            # Reset pending state
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0

    # -------------------------------------------------
    # Main update loop
    # -------------------------------------------------
    def update(self, ts_et, high: float, low: float, close: float):
        """Update filter state with new bar data"""
        date = ts_et.date()
        hour = ts_et.hour
        minute = ts_et.minute

        current_session = self.get_session(hour, minute)
        current_quarter = self.get_quarter(hour, minute, current_session) if current_session != 'CLOSED' else 0

        # === NEW DAY ===
        if self.current_date != date:
            # Save previous day's PM session as prev_day_pm
            if self.current_pm_high is not None:
                self.prev_day_pm_high = self.current_pm_high
                self.prev_day_pm_low = self.current_pm_low
                logging.info(
                    f"🏦 NEW DAY: Prev PM levels set - "
                    f"High: {self.prev_day_pm_high:.2f}, Low: {self.prev_day_pm_low:.2f}"
                )

            self.current_pm_high = None
            self.current_pm_low = None
            self.current_date = date

            # Reset midnight ORB for new calendar day
            self.midnight_orb_high = None
            self.midnight_orb_low = None
            self.midnight_orb_set = False

            # Recompute PM-based bank levels now that prev_day_pm_* is set
            self._update_bank_levels()

            # New day: clear any partial confirmation streak
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0

        # === SESSION CHANGE ===
        if current_session != self.last_session and current_session != 'CLOSED':
            # Save outgoing session's range as prev_session
            if self.curr_session_high is not None:
                self.prev_session_high = self.curr_session_high
                self.prev_session_low = self.curr_session_low
                logging.info(
                    f"🏦 SESSION CHANGE: Prev {self.last_session} "
                    f"High: {self.prev_session_high:.2f}, Low: {self.prev_session_low:.2f}"
                )

            self.curr_session_high = None
            self.curr_session_low = None
            self.last_session = current_session

            # Update bank levels whenever the reference session rolls
            self._update_bank_levels()

            # Session change: clear any partial confirmation streak
            self.pending_dir = None
            self.pending_source = None
            self.pending_count = 0

        # === QUARTER CHANGE: Reset bias ===
        if current_session != 'CLOSED':
            if (self.current_session != current_session or
                    self.current_quarter != current_quarter):
                if self.bank_rejection_bias is not None:
                    logging.info(
                        f"🔄 QUARTER CHANGE: {current_session} Q{current_quarter} "
                        f"| Resetting bank level bias (was {self.bank_rejection_bias} "
                        f"from {self.rejection_source})"
                    )
                self.bank_rejection_bias = None
                self.rejection_source = None
                self.current_session = current_session
                self.current_quarter = current_quarter

                # Clear any partial confirmation streak
                self.pending_dir = None
                self.pending_source = None
                self.pending_count = 0

        # === UPDATE CURRENT SESSION HIGH/LOW ===
        if current_session != 'CLOSED':
            if self.curr_session_high is None:
                self.curr_session_high = high
                self.curr_session_low = low
            else:
                self.curr_session_high = max(self.curr_session_high, high)
                self.curr_session_low = min(self.curr_session_low, low)

        # === UPDATE PM SESSION (12:00-17:00 ET) ===
        if 12 <= hour < 17:
            if self.current_pm_high is None:
                self.current_pm_high = high
                self.current_pm_low = low
            else:
                self.current_pm_high = max(self.current_pm_high, high)
                self.current_pm_low = min(self.current_pm_low, low)

        # === BUILD MIDNIGHT ORB (00:00 - 00:15 ET) ===
        if hour == 0 and minute < 15:
            if self.midnight_orb_high is None:
                self.midnight_orb_high = high
                self.midnight_orb_low = low
            else:
                self.midnight_orb_high = max(self.midnight_orb_high, high)
                self.midnight_orb_low = min(self.midnight_orb_low, low)

        # === COMPLETE ORB AT 00:15 ===
        elif hour == 0 and minute >= 15 and not self.midnight_orb_set:
            self.midnight_orb_set = True
            if self.midnight_orb_high is not None and self.midnight_orb_low is not None:
                self.bank_below_orb_low = self._get_closest_bank_below(self.midnight_orb_low)
                self.bank_above_orb_high = self._get_closest_bank_above(self.midnight_orb_high)
                logging.info(
                    f"🏦 MIDNIGHT ORB SET: High: {self.midnight_orb_high:.2f}, "
                    f"Low: {self.midnight_orb_low:.2f}"
                )
                logging.info(
                    f"🏦 ORB Bank Levels: Below ${self.bank_below_orb_low:.2f} | "
                    f"Above ${self.bank_above_orb_high:.2f}"
                )

        # === CHECK FOR BANK LEVEL REJECTIONS (with 2-candle confirm) ===
        if current_session != 'CLOSED' and current_quarter > 0:
            # 1. Midnight ORB (if set)
            if self.midnight_orb_set:
                # Bullish: bank below ORB low
                if self.bank_below_orb_low is not None and self.check_bank_rejection(
                        high, low, close, self.bank_below_orb_low, 'BULLISH'):
                    self._process_rejection(
                        'LONG',
                        f"Midnight ORB (bank {self.bank_below_orb_low:.2f} below ORB low)",
                        current_quarter,
                    )
                    return  # Respect midnight ORB as top priority

                # Bearish: bank above ORB high
                if self.bank_above_orb_high is not None and self.check_bank_rejection(
                        high, low, close, self.bank_above_orb_high, 'BEARISH'):
                    self._process_rejection(
                        'SHORT',
                        f"Midnight ORB (bank {self.bank_above_orb_high:.2f} above ORB high)",
                        current_quarter,
                    )
                    return

            # 2. Prev session banks
            if self.prev_session_high is not None and self.prev_session_low is not None:
                if self.bank_below_prev_session_low is not None and self.check_bank_rejection(
                        high, low, close, self.bank_below_prev_session_low, 'BULLISH'):
                    self._process_rejection(
                        'LONG',
                        f"Prev Session (bank {self.bank_below_prev_session_low:.2f} below prev session low)",
                        current_quarter,
                    )
                    return

                if self.bank_above_prev_session_high is not None and self.check_bank_rejection(
                        high, low, close, self.bank_above_prev_session_high, 'BEARISH'):
                    self._process_rejection(
                        'SHORT',
                        f"Prev Session (bank {self.bank_above_prev_session_high:.2f} above prev session high)",
                        current_quarter,
                    )
                    return

            # 3. Prev PM banks
            if self.prev_day_pm_high is not None and self.prev_day_pm_low is not None:
                if self.bank_below_prev_pm_low is not None and self.check_bank_rejection(
                        high, low, close, self.bank_below_prev_pm_low, 'BULLISH'):
                    self._process_rejection(
                        'LONG',
                        f"Prev PM (bank {self.bank_below_prev_pm_low:.2f} below prev PM low)",
                        current_quarter,
                    )
                    return

                if self.bank_above_prev_pm_high is not None and self.check_bank_rejection(
                        high, low, close, self.bank_above_prev_pm_high, 'BEARISH'):
                    self._process_rejection(
                        'SHORT',
                        f"Prev PM (bank {self.bank_above_prev_pm_high:.2f} above prev PM high)",
                        current_quarter,
                    )
                    return

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def should_block_trade(self, side: str) -> Tuple[bool, str]:
        """Check if trade should be blocked based on bank level rejection bias."""
        if self.bank_rejection_bias is None:
            return False, ""

        if self.bank_rejection_bias == 'LONG' and side == 'SHORT':
            return True, f"Bank level bias: LONG only Q{self.current_quarter} ({self.rejection_source})"

        if self.bank_rejection_bias == 'SHORT' and side == 'LONG':
            return True, f"Bank level bias: SHORT only Q{self.current_quarter} ({self.rejection_source})"

        return False, ""

    def get_status(self) -> str:
        """Return current filter status for logging"""
        bias_str = f"{self.bank_rejection_bias} ({self.rejection_source})" if self.bank_rejection_bias else "None"
        return f"Bank Filter: {self.current_session} Q{self.current_quarter} | Bias: {bias_str}"

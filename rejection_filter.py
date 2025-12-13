import datetime
import logging
from typing import Optional, Tuple

import pytz

from event_logger import event_logger


class RejectionFilter:
    """Track rejection levels and filter opposing trades.

    Updated logic:
    - 1 candle CLOSE required to establish or flip bias (candle must fully close)
    - Continuation: If price bounces from low level then breaks high = reinforces LONG bias
    - Continuation: If price rejects high level then breaks low = reinforces SHORT bias
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Previous day PM (12:00-17:00 ET)
        self.prev_day_pm_high: Optional[float] = None
        self.prev_day_pm_low: Optional[float] = None
        self.current_date: Optional[datetime.date] = None
        self.current_pm_high: Optional[float] = None
        self.current_pm_low: Optional[float] = None

        # Session tracking
        self.prev_session_high: Optional[float] = None
        self.prev_session_low: Optional[float] = None
        self.curr_session_high: Optional[float] = None
        self.curr_session_low: Optional[float] = None
        self.last_session: Optional[str] = None

        # ORB levels
        self.midnight_orb_high: Optional[float] = None
        self.midnight_orb_low: Optional[float] = None
        self.midnight_orb_set: bool = False

        # Rejection biases (None, 'LONG', 'SHORT')
        self.prev_day_pm_bias: Optional[str] = None
        self.prev_session_bias: Optional[str] = None
        self.midnight_orb_bias: Optional[str] = None

        # Quarterly Tracking
        self.current_quarter: int = 0
        self.current_session_name: Optional[str] = None

        # Track last rejection for continuation logic
        self.last_rejection_level: Optional[str] = None  # 'HIGH' or 'LOW'
        self.last_rejection_source: Optional[str] = None  # Which level was rejected

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

    def check_rejection(self, high: float, low: float, close: float,
                        level_high: Optional[float], level_low: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
        """Check if candle CLOSED showing rejection of a level.

        Returns: (bias_direction, level_type) where level_type is 'HIGH' or 'LOW'
        """
        if level_high is None or level_low is None:
            return None, None

        # Bullish rejection: swept low, CLOSED back above = LONG bias
        if low < level_low and close > level_low:
            return 'LONG', 'LOW'

        # Bearish rejection: swept high, CLOSED back below = SHORT bias
        if high > level_high and close < level_high:
            return 'SHORT', 'HIGH'

        return None, None

    def check_continuation(self, high: float, low: float, close: float,
                          level_high: Optional[float], level_low: Optional[float],
                          current_bias: Optional[str]) -> Optional[str]:
        """Check for continuation - breakout after rejection reinforces bias.

        - If we had a LOW rejection (bounced from low) and now CLOSE above high = LONG continuation
        - If we had a HIGH rejection (rejected from high) and now CLOSE below low = SHORT continuation
        """
        if level_high is None or level_low is None:
            return None

        # LONG continuation: previously bounced from low, now closed above high
        if self.last_rejection_level == 'LOW' and close > level_high:
            logging.info(f"📈 CONTINUATION: Bounced from low, broke high -> reinforcing LONG bias")

            # Enhanced event logging: Continuation pattern detected
            event_logger.log_rejection_detected(
                rejection_type=f"{self.last_rejection_source}_CONTINUATION",
                direction='LONG',
                level=level_high,
                current_price=close,
                additional_info={"pattern": "Bounce_from_low_broke_high", "previous_rejection": "LOW"}
            )
            return 'LONG'

        # SHORT continuation: previously rejected from high, now closed below low
        if self.last_rejection_level == 'HIGH' and close < level_low:
            logging.info(f"📉 CONTINUATION: Rejected from high, broke low -> reinforcing SHORT bias")

            # Enhanced event logging: Continuation pattern detected
            event_logger.log_rejection_detected(
                rejection_type=f"{self.last_rejection_source}_CONTINUATION",
                direction='SHORT',
                level=level_low,
                current_price=close,
                additional_info={"pattern": "Reject_from_high_broke_low", "previous_rejection": "HIGH"}
            )
            return 'SHORT'

        return None

    def _process_rejection(self, label: str, rej: Optional[str], level_type: Optional[str],
                           current_bias: Optional[str], current_quarter: int,
                           high: float, low: float, close: float,
                           level_high: Optional[float], level_low: Optional[float]) -> Optional[str]:
        """Process rejection with 1-candle close confirmation and continuation logic."""

        # First check for continuation (breakout after rejection)
        continuation = self.check_continuation(high, low, close, level_high, level_low, current_bias)
        if continuation:
            if current_bias != continuation:
                logging.info(f"🎯 CONTINUATION CONFIRMED (Q{current_quarter}): {label} -> {continuation} bias (breakout after rejection)")
            self.last_rejection_level = None  # Reset after continuation
            return continuation

        # Then check for new rejection (1 candle close required)
        if rej is not None:
            # Track which level was rejected for continuation logic
            self.last_rejection_level = level_type
            self.last_rejection_source = label

            if current_bias is None:
                # No bias yet - establish it with 1 closed candle
                logging.info(f"🎯 REJECTION CONFIRMED (Q{current_quarter}): {label} -> {rej} bias (candle closed)")

                # Enhanced event logging: Rejection detected
                event_logger.log_rejection_detected(
                    rejection_type=label,
                    direction=rej,
                    level=level_high if level_type == 'HIGH' else level_low,
                    current_price=close,
                    additional_info={"quarter": current_quarter, "level_type": level_type}
                )
                return rej
            elif rej != current_bias:
                # Opposite rejection - flip bias with 1 closed candle
                logging.info(f"🔁 BIAS FLIP (Q{current_quarter}): {label} {current_bias} -> {rej} (candle closed)")

                # Enhanced event logging: Rejection detected (bias flip)
                event_logger.log_rejection_detected(
                    rejection_type=label,
                    direction=rej,
                    level=level_high if level_type == 'HIGH' else level_low,
                    current_price=close,
                    additional_info={"quarter": current_quarter, "level_type": level_type, "previous_bias": current_bias}
                )
                return rej
            else:
                # Same direction - reinforce
                return current_bias

        return current_bias

    def update(self, ts_et, high: float, low: float, close: float):
        """Update all tracked levels with new bar data (on candle CLOSE)."""
        date = ts_et.date()
        hour = ts_et.hour
        minute = ts_et.minute
        current_session = self.get_session(hour)
        current_quarter = self.get_quarter(hour, minute, current_session) if current_session != 'CLOSED' else 0

        # === NEW DAY ===
        if self.current_date != date:
            if self.current_pm_high is not None:
                self.prev_day_pm_high = self.current_pm_high
                self.prev_day_pm_low = self.current_pm_low
            self.current_pm_high = None
            self.current_pm_low = None
            self.current_date = date

            # Reset ORB and its bias each new calendar day
            self.midnight_orb_high = None
            self.midnight_orb_low = None
            self.midnight_orb_bias = None
            self.midnight_orb_set = False

            # Reset session tracking for new day
            self.prev_session_high = None
            self.prev_session_low = None
            self.curr_session_high = None
            self.curr_session_low = None
            self.last_session = None
            self.prev_session_bias = None
            self.current_session_name = None
            self.current_quarter = 0

            logging.info(f"📅 New day detected: {date}. Resetting ORB and session levels.")

        # === SESSION MANAGEMENT ===
        session = self.get_session(hour)
        quarter = self.get_quarter(hour, minute, session) if session != 'CLOSED' else 0

        # Track session changes
        if session != self.current_session_name:
            logging.info(f"🕒 Session change: {self.current_session_name} -> {session} (Q{quarter})")
            if self.curr_session_high is not None:
                self.prev_session_high = self.curr_session_high
                self.prev_session_low = self.curr_session_low
            self.curr_session_high = None
            self.curr_session_low = None
            self.prev_session_bias = None
            self.last_session = self.current_session_name
            self.current_session_name = session
            self.current_quarter = quarter

        # Track highs/lows for current session
        if self.curr_session_high is None or high > self.curr_session_high:
            self.curr_session_high = high
        if self.curr_session_low is None or low < self.curr_session_low:
            self.curr_session_low = low

        # Track current day PM highs/lows
        if 12 <= hour < 17:
            if self.current_pm_high is None or high > self.current_pm_high:
                self.current_pm_high = high
            if self.current_pm_low is None or low < self.current_pm_low:
                self.current_pm_low = low

        # === MIDNIGHT ORB (00:00-00:15 ET) ===
        if hour == 0 and 0 <= minute < 15:
            if self.midnight_orb_high is None or high > self.midnight_orb_high:
                self.midnight_orb_high = high
            if self.midnight_orb_low is None or low < self.midnight_orb_low:
                self.midnight_orb_low = low
            self.midnight_orb_set = True
        elif hour == 0 and minute == 15:
            # At 00:15 candle close, check if we need to derive bias
            if self.midnight_orb_set and self.midnight_orb_bias is None:
                bias, level_type = self.check_rejection(
                    high, low, close,
                    self.midnight_orb_high, self.midnight_orb_low
                )
                self.midnight_orb_bias = self._process_rejection(
                    'MIDNIGHT_ORB', bias, level_type, self.midnight_orb_bias,
                    current_quarter=quarter, high=high, low=low, close=close,
                    level_high=self.midnight_orb_high, level_low=self.midnight_orb_low
                )

        # === PREVIOUS DAY PM HIGH/LOW ===
        if self.prev_day_pm_high is not None and self.prev_day_pm_low is not None:
            prev_pm_bias, level_type = self.check_rejection(
                high, low, close,
                self.prev_day_pm_high, self.prev_day_pm_low
            )
            self.prev_day_pm_bias = self._process_rejection(
                'PREV_DAY_PM', prev_pm_bias, level_type, self.prev_day_pm_bias,
                current_quarter=quarter, high=high, low=low, close=close,
                level_high=self.prev_day_pm_high, level_low=self.prev_day_pm_low
            )

        # === PREVIOUS SESSION HIGH/LOW ===
        if self.prev_session_high is not None and self.prev_session_low is not None:
            prev_session_bias, level_type = self.check_rejection(
                high, low, close,
                self.prev_session_high, self.prev_session_low
            )
            self.prev_session_bias = self._process_rejection(
                'PREV_SESSION', prev_session_bias, level_type, self.prev_session_bias,
                current_quarter=quarter, high=high, low=low, close=close,
                level_high=self.prev_session_high, level_low=self.prev_session_low
            )

        # === CURRENT SESSION MID-QUARTER: derive bias when price closes beyond levels ===
        if self.curr_session_high is not None and self.curr_session_low is not None:
            session_bias, level_type = self.check_rejection(
                high, low, close,
                self.curr_session_high, self.curr_session_low
            )
            self.prev_session_bias = self._process_rejection(
                'CURR_SESSION', session_bias, level_type, self.prev_session_bias,
                current_quarter=quarter, high=high, low=low, close=close,
                level_high=self.curr_session_high, level_low=self.curr_session_low
            )

        # Update current quarter and session name
        self.current_quarter = quarter
        self.current_session_name = session

    def _bias_to_text(self, bias: Optional[str]) -> str:
        if bias is None:
            return "Neutral"
        return "Long" if bias == 'LONG' else "Short"

    def should_block_trade(self, direction: str) -> Tuple[bool, str]:
        """Check if a trade should be blocked based on current rejection biases."""
        blocks = []
        details = []

        def block_if_opposite(bias, label):
            if bias == 'LONG' and direction == 'SELL':
                blocks.append(True)
                details.append(f"{label} bias LONG blocks SELL")
            elif bias == 'SHORT' and direction == 'BUY':
                blocks.append(True)
                details.append(f"{label} bias SHORT blocks BUY")

        block_if_opposite(self.prev_day_pm_bias, 'Prev Day PM')
        block_if_opposite(self.prev_session_bias, 'Prev Session')
        block_if_opposite(self.midnight_orb_bias, 'Midnight ORB')

        if any(blocks):
            reason = "; ".join(details)
            logging.info(f"⛔️ Trade blocked by RejectionFilter: {reason}")

            # Enhanced event logging: Trade blocked by rejection
            bias_info = {}
            if self.prev_day_pm_bias:
                bias_info['prev_day_pm_bias'] = self.prev_day_pm_bias
            if self.prev_session_bias:
                bias_info['prev_session_bias'] = self.prev_session_bias
            if self.midnight_orb_bias:
                bias_info['midnight_orb_bias'] = self.midnight_orb_bias

            event_logger.log_rejection_block(
                filter_name="RejectionFilter",
                signal_side=direction,
                reason=reason,
                additional_info=bias_info
            )
            return True, reason

        return False, ""

    def backfill_from_df(self, df, tz):
        """Backfill filter state from historical DataFrame."""
        logging.info("Backfilling rejection filter from historical data...")
        for _, row in df.iterrows():
            ts = row['timestamp'].tz_localize(pytz.UTC).tz_convert(tz)
            self.update(ts, row['high'], row['low'], row['close'])

        logging.info(f"✅ Backfill Complete.")
        if self.midnight_orb_set:
             logging.info(f"   ORB Loaded: {self.midnight_orb_high} - {self.midnight_orb_low}")
        if self.prev_day_pm_high:
             logging.info(f"   Prev PM Loaded: {self.prev_day_pm_high} - {self.prev_day_pm_low}")

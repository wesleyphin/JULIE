import datetime
import logging
from datetime import timezone as dt_timezone

from zoneinfo import ZoneInfo


class NewsFilter:
    """
    Blocks trading during specific time windows (e.g., High Impact News, Exchange Close).
    """

    def __init__(self):
        self.et = ZoneInfo("America/New_York")
        # Daily Recurrent Blackouts (Hour, Minute, Duration_Minutes)
        # Example: CME Close/Reopen (16:55 - 18:05 ET)
        self.daily_blackouts = [
            (16, 55, 70),
        ]
        # Specific Event Blackouts (Year, Month, Day, Hour, Minute, Duration_Minutes)
        self.calendar_blackouts = [
            # Example: FOMC
            # (2025, 12, 18, 14, 0, 30),
        ]

    def should_block_trade(self, current_time: datetime.datetime) -> tuple[bool, str]:
        # Ensure time is ET
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt_timezone.utc).astimezone(self.et)
        else:
            current_time = current_time.astimezone(self.et)

        # 1. Check Daily Recurring Blackouts
        for hour, minute, duration in self.daily_blackouts:
            start = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            end = start + datetime.timedelta(minutes=duration)

            # Handle timeframe overlap if needed (simplified for intraday)
            if start <= current_time <= end:
                return True, f"Daily Blackout: {start.strftime('%H:%M')} - {end.strftime('%H:%M')}"

        # 2. Check Specific Calendar Events
        for year, month, day, h, m, duration in self.calendar_blackouts:
            event_start = datetime.datetime(year, month, day, h, m, tzinfo=self.et)
            # Block 5 mins before event
            block_start = event_start - datetime.timedelta(minutes=5)
            block_end = event_start + datetime.timedelta(minutes=duration)

            if block_start <= current_time <= block_end:
                return True, f"News Event Blackout until {block_end.strftime('%H:%M')}"

        return False, ""

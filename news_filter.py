import datetime
import logging
import requests
import pandas as pd
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo
from pandas.tseries.holiday import USFederalHolidayCalendar


class NewsFilter:
    """
    Blocks trading during High Impact News events.
    Fetches data dynamically from ForexFactory.
    Also stores recent past events for Gemini 3.0 Analysis.
    """

    def __init__(self):
        self.et = ZoneInfo("America/New_York")
        self.ff_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

        # 1. Daily Recurrent Blackouts (Hour, Minute, Duration_Minutes)
        self.daily_blackouts = [
            (16, 55, 70), # CME Close
        ]

        # 2. Event Containers
        self.calendar_blackouts = [] # Future/Active events (for blocking trades)
        self.recent_events = []      # All events in current month (for Gemini context)

        # 3. Holiday Calendar
        self.holiday_cal = USFederalHolidayCalendar()

        # Load the calendar immediately
        self.refresh_calendar()

    def refresh_calendar(self):
        """Fetches 'Red Folder' (High Impact) USD news from ForexFactory."""
        try:
            logging.info("📅 Fetching news calendar from ForexFactory...")
            resp = requests.get(self.ff_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            count = 0
            current_time = datetime.datetime.now(self.et)

            # Reset lists
            self.calendar_blackouts = []
            self.recent_events = []

            # ==========================================
            # TIERED EVENT HANDLING
            # ==========================================
            # TIER 1: Market-moving events (CPI, FOMC, NFP, Powell)
            TIER_1_KEYWORDS = ['CPI', 'Non-Farm', 'FOMC', 'Rate Decision', 'Powell', 'NFP', 'Nonfarm']
            TIER_1_DURATION = 60  # Block for 1 hour after
            TIER_1_BUFFER = 10    # Block 10 mins before

            # TIER 2: Important but less volatile
            TIER_2_KEYWORDS = ['GDP', 'PPI', 'Retail Sales', 'Unemployment Claims', 'ISM', 'PMI']
            TIER_2_DURATION = 30
            TIER_2_BUFFER = 5

            # DEFAULT (Tier 3): Other high-impact events
            DEFAULT_DURATION = 15
            DEFAULT_BUFFER = 3
            # ==========================================

            for event in data:
                # Filter for High Impact USD news only
                if event.get('country') == 'USD' and event.get('impact') == 'High':
                    # Parse timestamp (Format: "2024-12-18T14:00:00-04:00")
                    event_dt_str = event.get('date')
                    try:
                        # Parse ISO format
                        event_dt = datetime.datetime.fromisoformat(event_dt_str)
                        # Convert to ET to match bot's internal clock
                        event_dt_et = event_dt.astimezone(self.et)

                        title = event.get('title', '')

                        # Determine Tier based on event title
                        duration = DEFAULT_DURATION
                        pre_buffer = DEFAULT_BUFFER
                        tier = 3

                        if any(k.lower() in title.lower() for k in TIER_1_KEYWORDS):
                            duration = TIER_1_DURATION
                            pre_buffer = TIER_1_BUFFER
                            tier = 1
                        elif any(k.lower() in title.lower() for k in TIER_2_KEYWORDS):
                            duration = TIER_2_DURATION
                            pre_buffer = TIER_2_BUFFER
                            tier = 2

                        # Create generic event object
                        event_obj = {
                            'title': title,
                            'time': event_dt_et,
                            'date_str': event_dt_et.strftime('%Y-%m-%d %H:%M'),
                            'impact': 'High',
                            'tier': tier
                        }

                        # A. Populate Context List (For Gemini)
                        # We capture events that happened recently in the current month
                        if event_dt_et.month == current_time.month:
                            self.recent_events.append(event_obj)

                        # B. Populate Blackout List (For Circuit Breaker)
                        # Only add future events (or events from today)
                        if event_dt_et.date() >= current_time.date():
                            self.calendar_blackouts.append({
                                'time': event_dt_et,
                                'title': title,
                                'duration': duration,      # DYNAMIC DURATION
                                'pre_buffer': pre_buffer,  # DYNAMIC BUFFER
                                'tier': tier
                            })
                            count += 1
                            logging.debug(f"📌 Event: {title} | Tier {tier} | Block: -{pre_buffer}m/+{duration}m")

                    except Exception as parse_err:
                        logging.warning(f"Failed to parse event date: {event_dt_str} - {parse_err}")

            logging.info(f"✅ Calendar updated: {count} upcoming blocking events.")
            logging.info(f"📋 Gemini Context: {len(self.recent_events)} events stored for analysis.")

        except Exception as e:
            logging.error(f"❌ Failed to fetch news calendar: {e}")
            logging.warning("⚠️ Running with NO dynamic news filters! Be careful.")

    def fetch_news(self) -> list[str]:
        """
        Returns a formatted list of all High Impact events for the current month.
        Used by GeminiSessionOptimizer to build the 'Big Picture'.
        """
        # Sort by date
        sorted_events = sorted(self.recent_events, key=lambda x: x['time'])

        # Format for LLM consumption
        if not sorted_events:
            return ["No major high-impact events detected this month."]

        return [f"{e['date_str']} | {e['title']}" for e in sorted_events]

    def get_holiday_context(self, current_time: datetime.datetime) -> str:
        """
        Determines if a bank holiday is approaching or just passed.
        Returns status codes for GeminiSessionOptimizer to adjust risk parameters.

        Status Codes:
        - NORMAL_LIQUIDITY: No holidays nearby
        - HOLIDAY_TODAY: Market is closed or extremely low volume
        - PRE_HOLIDAY_1_DAYS, PRE_HOLIDAY_2_DAYS, PRE_HOLIDAY_3_DAYS: Approaching holiday
        - POST_HOLIDAY_RECOVERY: Day after holiday (volume returning)
        """
        # Ensure time is in ET timezone
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt_timezone.utc).astimezone(self.et)
        else:
            current_time = current_time.astimezone(self.et)

        today = current_time.date()

        # Look for holidays within 1 day before and 3 days after
        start_date = pd.Timestamp(today - datetime.timedelta(days=1))
        end_date = pd.Timestamp(today + datetime.timedelta(days=3))
        holidays = self.holiday_cal.holidays(start=start_date, end=end_date)

        if holidays.empty:
            return "NORMAL_LIQUIDITY"

        # Get the nearest holiday
        h_date = holidays[0].date()

        if h_date == today:
            return "HOLIDAY_TODAY"
        elif h_date > today:
            days_away = (h_date - today).days
            return f"PRE_HOLIDAY_{days_away}_DAYS"
        else:
            # Holiday was yesterday
            return "POST_HOLIDAY_RECOVERY"

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
            if start <= current_time <= end:
                return True, f"Daily Blackout: {start.strftime('%H:%M')} - {end.strftime('%H:%M')}"

        # 2. Check Dynamic Calendar Events
        for event in self.calendar_blackouts:
            event_time = event['time']
            block_start = event_time - datetime.timedelta(minutes=event['pre_buffer'])
            block_end = event_time + datetime.timedelta(minutes=event['duration'])

            if block_start <= current_time <= block_end:
                return True, f"NEWS: {event['title']} ({event_time.strftime('%H:%M')})"

        return False, ""

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

    def _get_holiday_name(self, date_obj):
        """Helper to identify which specific holiday it is."""
        # Thanksgiving is always 4th Thursday of Nov
        if date_obj.month == 11 and date_obj.weekday() == 3 and 22 <= date_obj.day <= 28:
            return "Thanksgiving"
        # Labor Day is 1st Monday of Sept
        if date_obj.month == 9 and date_obj.weekday() == 0 and 1 <= date_obj.day <= 7:
            return "Labor Day"
        # MLK is 3rd Monday of Jan
        if date_obj.month == 1 and date_obj.weekday() == 0 and 15 <= date_obj.day <= 21:
            return "MLK Day"
        # Presidents is 3rd Monday of Feb
        if date_obj.month == 2 and date_obj.weekday() == 0 and 15 <= date_obj.day <= 21:
            return "Presidents Day"
        # Memorial Day is Last Monday of May
        if date_obj.month == 5 and date_obj.weekday() == 0 and date_obj.day >= 25:
            return "Memorial Day"
        # Juneteenth
        if date_obj.month == 6 and date_obj.day == 19:
            return "Juneteenth"
        # Independence Day
        if date_obj.month == 7 and date_obj.day == 4:
            return "Independence Day"
        # Good Friday (approximate - varies by year)
        # For precise calculation, would need Easter algorithm
        return "Bank Holiday"

    def get_holiday_context(self, current_time: datetime.datetime) -> str:
        """
        Returns the MASTER GAMEPLAN context string for Gemini.
        Integrates: Bearish Tuesdays, Dead Zones, Volatility Explosions.

        Tactical Holiday Playbook:
        - Labor Day Tuesday: Volatility Explosion (1.74x, -39pt returns)
        - MLK Day Tuesday: Gap Fill Reversal (-9pt gap)
        - Presidents Day Tuesday: Bearish Drift (0.8x vol, -20pts)
        - Thanksgiving: Dead Zone (Wed 12pm cutoff, Friday full disable)
        - July 4th: Patriot Drift (Mean reversion only, 50% size)
        """
        # Ensure time is in ET timezone
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt_timezone.utc).astimezone(self.et)
        else:
            current_time = current_time.astimezone(self.et)

        today = current_time.date()

        # 1. DEC/JAN SEASONAL DANGER (Keep existing logic - check first for priority)
        seasonal_msg = self.get_seasonal_context(current_time)
        if seasonal_msg != "NORMAL_SEASONAL":
            # Seasonal takes precedence during Dec/Jan
            return seasonal_msg

        # 2. HOLIDAY PROXIMITY CHECK (Next 3 days or Last 2 days)

        # Check Post-Holiday (Look back 1 day - The Bearish Tuesdays)
        yesterday = today - datetime.timedelta(days=1)
        holidays_yesterday = self.holiday_cal.holidays(start=yesterday, end=yesterday)

        if not holidays_yesterday.empty:
            h_name = self._get_holiday_name(yesterday)

            # === THE BEARISH TUESDAYS ===
            if h_name == "MLK Day":
                return (f"POST_HOLIDAY_TUESDAY (MLK Day). "
                       f"GAMEPLAN: Volatility 1.4x. GAP FILL REVERSAL. "
                       f"Expect -9pt gap. If gap > 10pts, look for Longs (Catch Up).")

            if h_name == "Presidents Day":
                return (f"POST_HOLIDAY_TUESDAY (Presidents Day). "
                       f"GAMEPLAN: Volatility 0.8x (Low). TREND SHORT. "
                       f"Expect -20pt sell-off drift. Take profits early.")

            if h_name == "Juneteenth":
                return (f"POST_HOLIDAY_TUESDAY (Juneteenth). "
                       f"GAMEPLAN: Standard Day but respect Negative Gap bias (-8.5pts).")

            # === THE VOLATILITY EXPLOSION ===
            if h_name == "Labor Day":
                return (f"POST_HOLIDAY_EXPLOSION (Labor Day). "
                       f"GAMEPLAN: Volatility 1.74x (MASSIVE). Return -39pts. "
                       f"AGGRESSIVE SHORT BIAS. Disable Mean Reversion. Widen Stops.")

            # === THE NON-EVENT ===
            if h_name == "Memorial Day":
                return "POST_HOLIDAY (Memorial Day). GAMEPLAN: Neutral/Normal."

            if "Good Friday" in h_name or (today.month == 4 and today.weekday() == 0):
                return "POST_HOLIDAY (Easter Monday). GAMEPLAN: No Pre-Market Trade (Europe Closed). Normal Open."

        # Check Pre-Holiday / Near Holiday
        start_date = pd.Timestamp(today)
        end_date = pd.Timestamp(today + datetime.timedelta(days=3))
        holidays = self.holiday_cal.holidays(start=start_date, end=end_date)

        if not holidays.empty:
            h_date = holidays[0].date()
            days_away = (h_date - today).days
            h_name = self._get_holiday_name(h_date)

            # === THE DEAD ZONES ===
            if h_name == "Thanksgiving":
                if days_away == 1:  # Wednesday
                    return ("PRE_HOLIDAY (Thanksgiving Eve). "
                           "GAMEPLAN: HARD STOP @ 12:00 PM. Volume fading to 87%.")
                elif days_away == 0:  # Thursday (should be blocked by should_block_trade)
                    return "HOLIDAY_TODAY"

            if h_name == "Independence Day":
                if days_away == 1:  # July 3rd
                    return ("PRE_HOLIDAY (July 4th). "
                           "GAMEPLAN: 'Patriot Drift'. Mean Reversion Only. 50% Size. Bullish Drift Bias.")
                elif days_away == 0:
                    return "HOLIDAY_TODAY"

            # Generic pre-holiday warnings for other holidays
            if days_away == 0:
                return "HOLIDAY_TODAY"
            elif days_away <= 3:
                return f"PRE_HOLIDAY_{days_away}_DAYS"

        return "NORMAL_LIQUIDITY"

    def get_seasonal_context(self, current_time: datetime.datetime) -> str:
        """
        HARD-CODED 'SEASONAL DANGER CALENDAR' (Dec 20 - Jan 2).
        Returns specific directives for Gemini based on historical seasonal patterns.

        Phases:
        - PHASE_1_LAST_GASP: Dec 20-23 (High volume, violent trends)
        - PHASE_2_DEAD_ZONE: Dec 24-31 (60% volume drop, broken structure)
        - PHASE_3_JAN2_REENTRY: Jan 2 (Bearish bias, funds return)
        """
        # Ensure time is in ET timezone
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt_timezone.utc).astimezone(self.et)
        else:
            current_time = current_time.astimezone(self.et)

        month = current_time.month
        day = current_time.day

        # PHASE 1: The "Last Gasp" (Dec 20-23)
        if month == 12 and 20 <= day <= 23:
            return "PHASE_1_LAST_GASP"

        # PHASE 2: The "Dead Zone" (Dec 24-31)
        elif month == 12 and 24 <= day <= 31:
            return "PHASE_2_DEAD_ZONE"

        # PHASE 3: The "Jan 2 Re-Entry" (Jan 2 specific)
        elif month == 1 and day == 2:
            return "PHASE_3_JAN2_REENTRY"

        return "NORMAL_SEASONAL"

    def should_block_trade(self, current_time: datetime.datetime) -> tuple[bool, str]:
        # Ensure time is ET
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt_timezone.utc).astimezone(self.et)
        else:
            current_time = current_time.astimezone(self.et)

        # === 1. DEAD ZONE HARD BLOCKS (The Kill Switches) ===
        month = current_time.month
        day = current_time.day
        weekday = current_time.weekday()  # 0=Mon, 6=Sun
        hour = current_time.hour
        minute = current_time.minute

        # A. Thanksgiving Week (4th Thursday of November)
        if month == 11:
            # Calculate Thanksgiving date dynamically
            # Find first day of November
            first_nov = datetime.date(current_time.year, 11, 1)
            # Find first Thursday (weekday 3)
            offset = (3 - first_nov.weekday()) % 7
            thanksgiving_day = 1 + offset + 21  # Add 3 weeks to get 4th Thursday

            # BLACKOUT: Wednesday after 12:00 PM
            if day == (thanksgiving_day - 1) and hour >= 12:
                return True, "DEAD ZONE: Thanksgiving Eve (Post-12PM) - Volume fading to 87%"

            # BLACKOUT: Thursday (Thanksgiving Day)
            if day == thanksgiving_day:
                return True, "HOLIDAY: Thanksgiving Day - Market Closed"

            # BLACKOUT: Friday (The Trap) - FULL DISABLE
            if day == (thanksgiving_day + 1):
                return True, "DEAD ZONE: Thanksgiving Friday (Volume 41%) - FULL DISABLE"

        # B. Independence Day Week
        if month == 7:
            # July 4th
            if day == 4:
                return True, "HOLIDAY: Independence Day - Market Closed"

            # July 3rd Early Close (1 PM block)
            if day == 3 and hour >= 13:
                return True, "DEAD ZONE: July 3rd (Post-1PM) - Early Close for Holiday"

        # === 2. STANDARD DAILY BLACKOUTS ===
        # Check Daily Recurring Blackouts (CME Close, etc.)
        for hour_cfg, minute_cfg, duration in self.daily_blackouts:
            start = current_time.replace(hour=hour_cfg, minute=minute_cfg, second=0, microsecond=0)
            end = start + datetime.timedelta(minutes=duration)
            if start <= current_time <= end:
                return True, f"Daily Blackout: {start.strftime('%H:%M')} - {end.strftime('%H:%M')}"

        # === 3. DYNAMIC NEWS CALENDAR EVENTS ===
        for event in self.calendar_blackouts:
            event_time = event['time']
            block_start = event_time - datetime.timedelta(minutes=event['pre_buffer'])
            block_end = event_time + datetime.timedelta(minutes=event['duration'])

            if block_start <= current_time <= block_end:
                return True, f"NEWS: {event['title']} ({event_time.strftime('%H:%M')})"

        return False, ""

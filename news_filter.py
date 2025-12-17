import datetime
import logging
import requests
import pandas as pd
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo


class NewsFilter:
    """
    Blocks trading during High Impact News events.
    Fetches data dynamically from ForexFactory.
    """

    def __init__(self):
        self.et = ZoneInfo("America/New_York")
        self.ff_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

        # 1. Daily Recurrent Blackouts (Hour, Minute, Duration_Minutes)
        # CME Close: 16:55 - 18:05 ET
        self.daily_blackouts = [
            (16, 55, 70),
        ]

        # 2. Dynamic Event Blackouts (Populated on startup)
        self.calendar_blackouts = []

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

            for event in data:
                # Filter for High Impact USD news only
                if event.get('country') == 'USD' and event.get('impact') == 'High':
                    # Parse timestamp (Format: "2024-12-18T14:00:00-04:00")
                    # We treat the date string carefully to ensure timezone awareness
                    event_dt_str = event.get('date')
                    try:
                        # Parse ISO format
                        event_dt = datetime.datetime.fromisoformat(event_dt_str)
                        # Convert to ET to match bot's internal clock
                        event_dt_et = event_dt.astimezone(self.et)

                        # Only add future events (or events from today)
                        if event_dt_et.date() >= current_time.date():
                            # Store: (datetime object, title)
                            # We block 5 mins before and 30 mins after (35 min duration)
                            self.calendar_blackouts.append({
                                'time': event_dt_et,
                                'title': event.get('title'),
                                'duration': 35, # Default blackout duration
                                'pre_buffer': 5 # Minutes before event to stop
                            })
                            count += 1
                    except Exception as parse_err:
                        logging.warning(f"Failed to parse event date: {event_dt_str} - {parse_err}")

            logging.info(f"✅ Calendar updated: {count} high-impact USD events found this week.")

        except Exception as e:
            logging.error(f"❌ Failed to fetch news calendar: {e}")
            logging.warning("⚠️ Running with NO dynamic news filters! Be careful.")

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
            # Define window: 5 mins before to X mins after
            block_start = event_time - datetime.timedelta(minutes=event['pre_buffer'])
            block_end = event_time + datetime.timedelta(minutes=event['duration'])

            if block_start <= current_time <= block_end:
                return True, f"NEWS: {event['title']} ({event_time.strftime('%H:%M')})"

        return False, ""

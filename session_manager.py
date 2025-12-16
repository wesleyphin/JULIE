import datetime
import logging

import joblib
from zoneinfo import ZoneInfo

from config import CONFIG

NY_TZ = ZoneInfo('America/New_York')


class SessionManager:
    """Manages the 4 Neural Networks and switches them based on NY Time."""

    def __init__(self):
        self.brains = {}
        self.load_all_brains()

    def load_all_brains(self):
        logging.info("🧠 Initializing Neural Network Array...")
        for name, settings in CONFIG["SESSIONS"].items():
            path = settings["MODEL_FILE"]
            try:
                self.brains[name] = joblib.load(path)
                logging.info(f"  ✅ {name} Specialist Loaded (Thresh: {settings['THRESHOLD']})")
            except Exception as e:
                logging.error(f"  ❌ Failed to load {name} ({path}): {e}")

    def get_current_setup(self):
        """Returns the active Model, Threshold, SL, and TP for the current minute."""
        now_ny = datetime.datetime.now(NY_TZ)
        current_hour = now_ny.hour

        for name, settings in CONFIG["SESSIONS"].items():
            if current_hour in settings["HOURS"]:
                model = self.brains.get(name)
                return {
                    "name": name,
                    "model": model,
                    "threshold": settings["THRESHOLD"],
                    "sl": settings["SL"],
                    "tp": settings["TP"]
                }

        return None  # Market Closed or Gap Time

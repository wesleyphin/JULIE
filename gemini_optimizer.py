import pandas as pd
import requests
import json
import logging
import datetime
import numpy as np
from config import CONFIG


class GeminiSessionOptimizer:
    def __init__(self):
        self.config = CONFIG.get('GEMINI', {})
        self.api_key = self.config.get('api_key', '')
        self.model = "gemini-3-pro-preview"
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}
        self.csv_path = 'es_2023_2024_2025.csv'

    def _load_historical_data(self):
        """Loads and cleans the historical CSV."""
        try:
            df = pd.read_csv(self.csv_path, thousands=',')
            df.columns = [c.strip().lower() for c in df.columns]

            date_col = next((c for c in df.columns if 'date' in c), None)
            if not date_col:
                return None

            df['timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df.set_index('timestamp', inplace=True)

            # Numeric cleanup
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns and df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logging.error(f"Error loading historical CSV: {e}")
            return None

    def _get_volatility_profile(self, df):
        """Calculates 13-day volatility stats."""
        if df.empty:
            return {}
        df = df.copy()
        df['range'] = df['high'] - df['low']
        return {
            "avg_range": round(df['range'].mean(), 2),
            "max_range": round(df['range'].max(), 2),
            "std_dev": round(df['range'].std(), 2)
        }

    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp):
        """
        Main Optimization Routine.
        """
        logging.info(f"🧠 Gemini 3.0: Analyzing {session_name} Session (13-Day vs History)...")

        # --- 1. PROCESS CURRENT 13-DAY DATA ---
        if master_df.empty:
            return None

        last_time = master_df.index.max()
        start_time_13d = last_time - datetime.timedelta(days=13)
        current_window = master_df[master_df.index >= start_time_13d]

        curr_metrics = self._get_volatility_profile(current_window)

        # --- 2. PROCESS HISTORICAL DATA (Seasonality) ---
        hist_df = self._load_historical_data()
        hist_context = "Historical Data Unavailable"

        if hist_df is not None:
            mask = (
                (hist_df.index.month == last_time.month) &
                (hist_df.index.day >= start_time_13d.day) &
                (hist_df.index.day <= last_time.day)
            )
            hist_window = hist_df[mask]

            if not hist_window.empty:
                h_metrics = self._get_volatility_profile(hist_window)
                hist_context = (
                    f"HISTORICAL SEASONALITY (Same 13-day window in 2023-2025):\n"
                    f"- Historical Avg Volatility: {h_metrics.get('avg_range')} pts\n"
                    f"- Historical Max Spike: {h_metrics.get('max_range')} pts"
                )

        # --- 3. CONSTRUCT PROMPT ---
        system_instruction = (
            f"You are a Quantitative Risk Manager optimizing for the {session_name} session. "
            "Your Task: Compare Current vs Historical Volatility and analyze Major Events to adjust TP/SL Multipliers.\n"
            "Rules:\n"
            "- If High Impact Events (CPI/PPI/FOMC) are present: WIDEN SL/TP significantly (1.5x+).\n"
            "- If Current Volatility > Historical: WIDEN SL/TP (1.2x).\n"
            "- If Current Volatility < Historical: TIGHTEN SL/TP (0.8x-1.0x)."
        )

        user_prompt = (
            f"**SESSION HANDOVER: {session_name}**\n\n"
            f"=== FINANCIAL EVENTS (Live Feed) ===\n"
            f"{events_data}\n\n"
            f"=== CURRENT REGIME (Last 13 Days) ===\n"
            f"Avg 1-min Range: {curr_metrics.get('avg_range', 'N/A')} pts\n\n"
            f"=== HISTORICAL BASELINE ===\n"
            f"{hist_context}\n\n"
            f"=== DEFAULT PARAMETERS ===\n"
            f"Base SL: {base_sl}\n"
            f"Base TP: {base_tp}\n\n"
            "**OPTIMIZATION OUTPUT (JSON ONLY):**\n"
            "{\n"
            '  "session": "' + session_name + '",\n'
            '  "regime": "VOLATILE" or "QUIET" or "EVENT_RISK",\n'
            '  "sl_multiplier": number (e.g. 1.2),\n'
            '  "tp_multiplier": number (e.g. 1.5),\n'
            '  "reasoning": "Explain decision based on Event + Data comparison"\n'
            "}"
        )

        # --- 4. EXECUTE GEMINI CALL ---
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=45)
            if response.status_code == 200:
                return json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
            else:
                logging.error(f"Gemini API Error: {response.text}")
                return None
        except Exception as e:
            logging.error(f"Gemini Optimization Error: {e}")
            return None

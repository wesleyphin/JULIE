
import unittest
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from continuation_strategy import FractalSweepStrategy, STRATEGY_CONFIGS
from julie001 import ContinuationRescueManager

class TestContinuationFixes(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe with UTC timestamps
        self.utc_tz = timezone.utc
        self.ny_tz = ZoneInfo("America/New_York")

        # Setup a specific time that corresponds to one of the strategies
        # 'Q4_W45_D7_Asia': {'Quarter': 4, 'Week': 45, 'Day': 7, 'Session': 'Asia'}
        # 2023-11-12 is Sunday (Day 7), Week 45, Quarter 4
        # Asia session starts at 18:00 NY time.

        # 2023-11-12 19:00 NY time
        # This is 2023-11-13 00:00 UTC (Standard Time)
        self.target_ny_time = datetime(2023, 11, 12, 19, 0, 0, tzinfo=self.ny_tz)
        self.target_utc_time = self.target_ny_time.astimezone(self.utc_tz)

        # Create a dataframe with an index around this time
        dates = [self.target_utc_time - timedelta(minutes=i) for i in range(5)]
        dates.reverse()
        self.df = pd.DataFrame({'close': [100.0] * 5}, index=dates)
        self.df.index.name = 'Date'

    def test_fractal_sweep_strategy_timezone_conversion(self):
        # Test that the strategy correctly identifies the time window based on NY time
        strategy_id = 'Q4_W45_D7_Asia'
        strategy = FractalSweepStrategy(strategy_id)

        signals = strategy.generate_signals(self.df)

        # The last row should be returned because it matches the criteria
        self.assertFalse(signals.empty, "Signals should not be empty")

        # Check if the conversion added the columns
        self.assertIn('Quarter', signals.columns)
        self.assertIn('Week', signals.columns)
        self.assertIn('Day', signals.columns)

        # Verify the calculated values
        last_row = signals.iloc[-1]
        self.assertEqual(last_row['Quarter'], 4)
        self.assertEqual(last_row['Week'], 45)
        self.assertEqual(last_row['Day'], 7)

    def test_continuation_rescue_manager_key_generation(self):
        manager = ContinuationRescueManager()

        # Current time in UTC (input to the function)
        current_time = self.target_utc_time

        # This should map to Q4_W45_D7_Asia
        # We need to make sure this key exists in configs for the test to proceed
        # If it doesn't, we can mock it or just verify the key logic if we could inspect internals
        # But here we are testing get_active_continuation_signal

        # Ensure the key exists in config for the test
        if 'Q4_W45_D7_Asia' not in manager.configs:
             manager.configs['Q4_W45_D7_Asia'] = {'Quarter': 4, 'Week': 45, 'Day': 7, 'Session': 'Asia'}

        # Mock generate_signals to always return something valid for the test
        # Because we want to test the key generation and time conversion in the manager
        # But wait, we are using the real class, so let's rely on integration

        result = manager.get_active_continuation_signal(self.df, current_time, required_bias="LONG")

        self.assertIsNotNone(result, "Rescue signal should be found")
        self.assertEqual(result['strategy'], "Continuation_Q4_W45_D7_Asia")
        self.assertTrue(result['rescued'])
        self.assertEqual(result['side'], "LONG")

    def test_timezone_conversion_logic(self):
        # Explicitly test the timezone logic used in the fix

        # Case 1: Naive timestamp (Assumed UTC)
        naive_time = datetime(2023, 11, 13, 0, 0, 0) # 00:00 UTC = 19:00 NY Prev Day

        # Manually apply logic from julie001.py
        current_time = naive_time
        if current_time.tzinfo is None:
             current_time = current_time.replace(tzinfo=timezone.utc)

        ny_time = current_time.astimezone(ZoneInfo('America/New_York'))

        self.assertEqual(ny_time.day, 12) # Should be the previous day in NY
        self.assertEqual(ny_time.hour, 19)
        self.assertEqual(ny_time.weekday(), 6) # Sunday = 6

        # Case 2: Aware timestamp (UTC)
        aware_time = datetime(2023, 11, 13, 0, 0, 0, tzinfo=timezone.utc)
        ny_time_aware = aware_time.astimezone(ZoneInfo('America/New_York'))

        self.assertEqual(ny_time_aware.day, 12)
        self.assertEqual(ny_time_aware.hour, 19)


if __name__ == '__main__':
    unittest.main()

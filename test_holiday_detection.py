#!/usr/bin/env python3
"""
Test script to verify bank holiday detection functionality.
"""

import sys
import logging
import datetime
from datetime import timezone as dt_timezone
from news_filter import NewsFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_holiday_detection():
    """Test the get_holiday_context method."""
    print("=" * 60)
    print("Testing Bank Holiday Detection")
    print("=" * 60)

    try:
        # Initialize NewsFilter
        print("\n1. Initializing NewsFilter...")
        news_filter = NewsFilter()
        print("   ✅ NewsFilter initialized successfully")

        # Test holiday context with current time
        print("\n2. Getting Holiday Context...")
        current_time = datetime.datetime.now(dt_timezone.utc)
        holiday_status = news_filter.get_holiday_context(current_time)
        print(f"   Holiday Status: {holiday_status}")

        # Display the result
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Holiday Status Code: {holiday_status}")
        print("=" * 60)

        # Interpret status codes
        if holiday_status == "HOLIDAY_TODAY":
            print("\n⚠️  CRITICAL: Bank Holiday TODAY!")
            print("   Risk Engine: Market is dead. TP multiplier → 0.5x minimum")
            print("   Chop Engine: Extreme chop. Chop multiplier → 2.0x - 3.0x")
            print("   Trend Engine: Standard filters apply")
        elif holiday_status.startswith("PRE_HOLIDAY"):
            days = holiday_status.split("_")[-2]
            print(f"\n📅 Bank Holiday in {days} day(s)")
            print("   Risk Engine: Price action tightens ~40%. TP multiplier → 0.6x - 0.7x")
            print("   Chop Engine: Dead market chop. Chop multiplier → 1.5x - 2.0x")
            print("   Trend Engine: Increase t1_body/t1_vol to 3.0x - 4.0x (trap avoidance)")
        elif holiday_status == "POST_HOLIDAY_RECOVERY":
            print("\n🔄 Post-Holiday Recovery (Day After)")
            print("   Risk Engine: Volatility expands ~12%. SL multiplier → 1.2x")
            print("   Chop Engine: Moderate multiplier → 1.2x - 1.5x")
            print("   Trend Engine: Maintain elevated filters (2.5x+) until clear flow")
        elif holiday_status == "NORMAL_LIQUIDITY":
            print("\n✅ Normal Market Liquidity")
            print("   All engines: Apply standard rules based on ADX and structure")
        else:
            print(f"\n⚠️  Unknown status: {holiday_status}")

        print("\n✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_holiday_detection()
    sys.exit(0 if success else 1)

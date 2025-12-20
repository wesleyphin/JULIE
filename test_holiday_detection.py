#!/usr/bin/env python3
"""
Test script to verify bank holiday detection functionality.
"""

import sys
import logging
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

        # Test holiday context
        print("\n2. Getting Holiday Context...")
        holiday_context = news_filter.get_holiday_context()
        print(f"   Holiday Context: {holiday_context}")

        # Display the result
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Holiday Status: {holiday_context}")
        print("=" * 60)

        if "HOLIDAY TODAY" in holiday_context or "HOLIDAY TOMORROW" in holiday_context:
            print("\n⚠️  CRITICAL: Bank Holiday detected within 24 hours!")
            print("   Risk Engine will reduce TP multipliers to 0.5x - 0.7x")
            print("   Chop Engine will raise multiplier to 1.5x - 2.0x")
        elif "NEAR HOLIDAY" in holiday_context:
            print("\n📅 Bank Holiday detected within 2-3 days")
            print("   Risk Engine will moderately reduce TP to 0.8x - 0.9x")
            print("   Chop Engine will favor mean-reversion strategies")
        else:
            print("\n✅ No bank holidays detected in the next 3 days")
            print("   Normal trading parameters will be used")

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

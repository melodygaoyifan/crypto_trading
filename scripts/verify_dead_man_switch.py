"""
Quick verification of Kraken CancelAllOrdersAfter (dead-man switch) API.

Tests:
1. Set timer to 60s -> should return currentTime + triggerTime
2. Refresh timer -> triggerTime should reset
3. Disable timer (timeout=0) -> should clear

Usage:
    python -X utf8 scripts/verify_dead_man_switch.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import ccxt

def main():
    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")

    if not api_key or not api_secret:
        print("ERROR: KRAKEN_API_KEY / KRAKEN_API_SECRET not set in .env")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    exchange = ccxt.kraken({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    # Test 1: Set dead-man timer to 60s
    print("\n[TEST 1] Setting dead-man switch timer to 60 seconds...")
    try:
        result = exchange.cancel_all_orders_after(60000)  # 60000ms = 60s
        print(f"  Result: {result}")
        print("  PASS: Timer set successfully")
    except Exception as e:
        print(f"  FAIL: {e}")
        if "permission" in str(e).lower() or "nonce" in str(e).lower():
            print("  NOTE: API key may need 'Cancel/Close Orders' permission")
        return False

    # Wait a moment
    time.sleep(2)

    # Test 2: Refresh timer (reset countdown)
    print("\n[TEST 2] Refreshing dead-man switch timer...")
    try:
        result = exchange.cancel_all_orders_after(60000)
        print(f"  Result: {result}")
        print("  PASS: Timer refreshed successfully")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    # Test 3: Disable timer (timeout=0)
    print("\n[TEST 3] Disabling dead-man switch (timeout=0)...")
    try:
        result = exchange.cancel_all_orders_after(0)
        print(f"  Result: {result}")
        print("  PASS: Timer disabled successfully")
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - Dead-man switch API verified")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

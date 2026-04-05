"""
HMATS v6.4.1 - False Breakout Detector Tests
=============================================

Tests for the FalseBreakoutDetector module.
Verifies that veto only triggers for breakout/retest entry types.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.false_breakout_detector import (
    FalseBreakoutDetector,
    FalseBreakoutConfig,
    FalseBreakoutResult,
)


def generate_price_series(base_price, pattern="uptrend", length=20):
    """Generate synthetic price series."""
    prices = []
    price = base_price
    
    if pattern == "uptrend":
        for i in range(length):
            price += 0.5 + (i * 0.1)
            prices.append(price)
    elif pattern == "false_breakout_long":
        # Price breaks above, then reverses
        for i in range(length - 3):
            price += 0.3
            prices.append(price)
        # Break above
        breakout_high = price + 2.0
        prices.append(breakout_high)
        # Reversal
        prices.append(breakout_high - 1.5)
        prices.append(breakout_high - 3.0)  # Back below breakout
    elif pattern == "valid_breakout":
        # Price breaks above and holds
        for i in range(length - 3):
            price += 0.3
            prices.append(price)
        # Break above
        breakout_high = price + 2.0
        prices.append(breakout_high)
        # Hold above
        prices.append(breakout_high + 0.5)
        prices.append(breakout_high + 1.0)
    elif pattern == "sideways":
        for i in range(length):
            prices.append(base_price + (i % 3 - 1) * 0.5)
    
    return prices


def generate_volume_series(pattern="normal", length=20):
    """Generate synthetic volume series."""
    base_vol = 1000
    
    if pattern == "normal":
        return [base_vol + (i * 10) for i in range(length)]
    elif pattern == "declining":
        return [base_vol - (i * 30) for i in range(length)]
    elif pattern == "weak":
        return [base_vol * 0.3 for _ in range(length)]
    elif pattern == "strong":
        return [base_vol * 2 for _ in range(length)]
    
    return [base_vol] * length


def test_false_breakout_detected():
    """Test that false breakout is detected for breakout entry."""
    detector = FalseBreakoutDetector()
    
    # Create false breakout scenario
    prices = generate_price_series(100.0, pattern="false_breakout_long")
    volumes = generate_volume_series(pattern="weak")  # Weak volume
    breakout_level = 105.0  # Price breaks above, then falls below
    
    result = detector.detect(
        price_series=prices,
        atr=2.0,
        volume_series=volumes,
        rsi_or_momentum=35.0,  # Weak RSI for long = divergence
        breakout_level=breakout_level,
        direction=1,  # Long
        entry_type="breakout",
    )
    
    print(f"False Breakout Detection: is_false={result.is_false_breakout}, "
          f"confidence={result.confidence:.2f}, reasons={result.reasons}")
    
    # Should detect false breakout
    assert result.is_false_breakout == True, "Should detect false breakout"
    assert len(result.reasons) > 0, "Should have reasons"
    
    return True


def test_valid_breakout_not_vetoed():
    """Test that valid breakout is not vetoed."""
    detector = FalseBreakoutDetector()
    
    # Create valid breakout scenario
    prices = generate_price_series(100.0, pattern="valid_breakout")
    volumes = generate_volume_series(pattern="strong")  # Strong volume
    breakout_level = 105.0  # Price breaks and holds
    
    result = detector.detect(
        price_series=prices,
        atr=2.0,
        volume_series=volumes,
        rsi_or_momentum=65.0,  # Strong RSI for long
        breakout_level=breakout_level,
        direction=1,  # Long
        entry_type="breakout",
    )
    
    print(f"Valid Breakout: is_false={result.is_false_breakout}, "
          f"confidence={result.confidence:.2f}")
    
    # Should NOT detect false breakout
    assert result.is_false_breakout == False, "Should not veto valid breakout"
    
    return True


def test_momentum_entry_not_checked():
    """Test that momentum entries bypass false breakout check."""
    detector = FalseBreakoutDetector()
    
    # Even with bad data, momentum entry should not be vetoed
    prices = generate_price_series(100.0, pattern="false_breakout_long")
    volumes = generate_volume_series(pattern="weak")
    
    result = detector.detect(
        price_series=prices,
        atr=2.0,
        volume_series=volumes,
        rsi_or_momentum=30.0,
        breakout_level=105.0,
        direction=1,
        entry_type="momentum",  # Not a breakout entry
    )
    
    print(f"Momentum Entry: is_false={result.is_false_breakout}")
    
    # Should NOT detect false breakout for non-breakout entries
    assert result.is_false_breakout == False, "Momentum entry should bypass check"
    assert "entry_type=momentum not in veto list" in str(result.details)
    
    return True


def test_retest_entry_checked():
    """Test that retest entries are checked."""
    detector = FalseBreakoutDetector()
    
    # False breakout scenario for retest
    prices = generate_price_series(100.0, pattern="false_breakout_long")
    volumes = generate_volume_series(pattern="weak")
    
    result = detector.detect(
        price_series=prices,
        atr=2.0,
        volume_series=volumes,
        rsi_or_momentum=35.0,
        breakout_level=105.0,
        direction=1,
        entry_type="retest",  # Retest entry IS checked
    )
    
    print(f"Retest Entry: is_false={result.is_false_breakout}")
    
    # Retest entries should be checked
    assert result.is_false_breakout == True, "Retest entry should be checked"
    
    return True


def test_insufficient_data_safe():
    """Test that insufficient data returns safe default."""
    detector = FalseBreakoutDetector()
    
    # Only 2 prices - insufficient
    result = detector.detect(
        price_series=[100.0, 101.0],
        entry_type="breakout",
        direction=1,
    )
    
    print(f"Insufficient Data: is_false={result.is_false_breakout}")
    
    # Should NOT veto with insufficient data
    assert result.is_false_breakout == False, "Should be safe with insufficient data"
    assert "insufficient_price_data" in str(result.details)
    
    return True


def test_disabled_detector():
    """Test detector when disabled."""
    config = FalseBreakoutConfig(enabled=False)
    detector = FalseBreakoutDetector(config)
    
    # Even with false breakout scenario, should not veto
    prices = generate_price_series(100.0, pattern="false_breakout_long")
    
    result = detector.detect(
        price_series=prices,
        atr=2.0,
        volume_series=generate_volume_series(pattern="weak"),
        breakout_level=105.0,
        direction=1,
        entry_type="breakout",
    )
    
    print(f"Disabled Detector: is_false={result.is_false_breakout}")
    
    # Should NOT veto when disabled
    assert result.is_false_breakout == False
    
    return True


def test_criteria_accumulation():
    """Test that multiple criteria increase confidence."""
    detector = FalseBreakoutDetector()
    
    # Scenario with all negative criteria
    prices = generate_price_series(100.0, pattern="false_breakout_long")
    volumes = generate_volume_series(pattern="weak")
    
    result = detector.detect(
        price_series=prices,
        atr=1.0,  # Small ATR = larger normalized rejection
        volume_series=volumes,
        rsi_or_momentum=30.0,  # Weak RSI divergence
        breakout_level=105.0,
        direction=1,
        entry_type="breakout",
    )
    
    print(f"Multiple Criteria: reasons={result.reasons}, confidence={result.confidence:.2f}")
    
    # Should have multiple reasons
    assert len(result.reasons) >= 2, f"Expected 2+ reasons, got {len(result.reasons)}"
    assert result.confidence > 0.3, "Confidence should be higher with multiple criteria"
    
    return True


def run_all_tests():
    """Run all false breakout detector tests."""
    print("=" * 60)
    print("FALSE BREAKOUT DETECTOR TESTS")
    print("=" * 60)
    
    tests = [
        ("False Breakout Detected", test_false_breakout_detected),
        ("Valid Breakout Not Vetoed", test_valid_breakout_not_vetoed),
        ("Momentum Entry Bypass", test_momentum_entry_not_checked),
        ("Retest Entry Checked", test_retest_entry_checked),
        ("Insufficient Data Safe", test_insufficient_data_safe),
        ("Disabled Detector", test_disabled_detector),
        ("Criteria Accumulation", test_criteria_accumulation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"  ✓ {name}")
                passed += 1
            else:
                print(f"  ✗ {name}")
                failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

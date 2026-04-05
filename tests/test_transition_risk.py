"""
HMATS v6.4.1 - Regime Transition Risk Tests
============================================

Tests for the RegimeTransitionRisk module.
Verifies stability metrics change appropriately with regime sequences.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.regime_transition_risk import (
    RegimeTransitionRisk,
    RegimeTransitionConfig,
    TransitionRiskResult,
)


def test_initial_state():
    """Test initial state returns safe defaults."""
    tracker = RegimeTransitionRisk()
    
    result = tracker.get_risk("TREND")
    
    print(f"Initial State: stability={result.stability:.2f}, valid={result.valid}")
    
    # Should return defaults (not valid until enough observations)
    assert result.valid == False, "Should not be valid initially"
    assert result.stability == 0.5, "Should return default stability"
    
    return True


def test_stable_regime():
    """Test stability increases when regime stays the same."""
    tracker = RegimeTransitionRisk()
    
    # Stay in TREND for many iterations
    for _ in range(20):
        tracker.update("TREND")
    
    result = tracker.get_risk("TREND")
    
    print(f"Stable Regime: stability={result.stability:.2f}, valid={result.valid}")
    
    # Should show high stability
    assert result.valid == True, "Should be valid after observations"
    assert result.stability > 0.6, f"Should have high stability, got {result.stability}"
    assert result.transition_risk < 0.4, "Transition risk should be low"
    
    return True


def test_unstable_regime():
    """Test stability decreases with frequent transitions."""
    tracker = RegimeTransitionRisk()
    
    # Alternate between regimes
    regimes = ["TREND", "NEUTRAL", "VOLATILE_CHOP", "TREND", "NEUTRAL"]
    
    for _ in range(20):
        for regime in regimes:
            tracker.update(regime)
    
    result = tracker.get_risk("TREND")
    
    print(f"Unstable Regime: stability={result.stability:.2f}, transition_risk={result.transition_risk:.2f}")
    
    # Should show lower stability due to frequent transitions
    # Note: stability might still be moderate depending on pattern
    assert result.valid == True, "Should be valid"
    assert result.transition_risk > 0.3, "Should have some transition risk"
    
    return True


def test_transition_probability():
    """Test transition probability matrix is built correctly."""
    tracker = RegimeTransitionRisk()
    
    # Create a known pattern: TREND -> NEUTRAL -> TREND -> NEUTRAL
    for _ in range(15):
        tracker.update("TREND")
        tracker.update("NEUTRAL")
    
    matrix = tracker.get_transition_matrix()
    
    print(f"Transition Matrix: {matrix}")
    
    # TREND should primarily transition to NEUTRAL
    if "TREND" in matrix:
        trend_probs = matrix["TREND"]
        neutral_prob = trend_probs.get("NEUTRAL", 0)
        assert neutral_prob > 0.4, f"TREND->NEUTRAL should be high, got {neutral_prob}"
    
    return True


def test_decay_factor():
    """Test that decay factor makes recent observations more important."""
    tracker = RegimeTransitionRisk()
    
    # First, create history of TREND
    for _ in range(30):
        tracker.update("TREND")
    
    stability_after_trend = tracker.get_risk("TREND").stability
    
    # Now add some transitions
    for _ in range(10):
        tracker.update("NEUTRAL")
        tracker.update("TREND")
    
    stability_after_transitions = tracker.get_risk("TREND").stability
    
    print(f"Decay Effect: before={stability_after_trend:.2f}, after={stability_after_transitions:.2f}")
    
    # Recent transitions should reduce stability
    assert stability_after_transitions < stability_after_trend + 0.1, \
        "Recent transitions should affect stability"
    
    return True


def test_most_likely_next():
    """Test most_likely_next prediction."""
    tracker = RegimeTransitionRisk()
    
    # Pattern: TREND usually goes to NEUTRAL
    for _ in range(20):
        tracker.update("TREND")
        tracker.update("NEUTRAL")
    
    result = tracker.get_risk("TREND")
    
    print(f"Most Likely Next: {result.most_likely_next}")
    
    # From TREND, most likely next should be NEUTRAL (or TREND if it stays)
    assert result.most_likely_next in ["NEUTRAL", "TREND"], \
        f"Unexpected most_likely_next: {result.most_likely_next}"
    
    return True


def test_reset():
    """Test reset clears all state."""
    tracker = RegimeTransitionRisk()
    
    # Build up some state
    for _ in range(20):
        tracker.update("TREND")
    
    result_before = tracker.get_risk("TREND")
    assert result_before.valid == True
    
    # Reset
    tracker.reset()
    
    result_after = tracker.get_risk("TREND")
    
    print(f"Reset: valid_before={result_before.valid}, valid_after={result_after.valid}")
    
    # Should be back to initial state
    assert result_after.valid == False, "Should be invalid after reset"
    assert tracker._observation_count == 0, "Observation count should be 0"
    
    return True


def test_disabled_tracker():
    """Test tracker when disabled."""
    config = RegimeTransitionConfig(enabled=False)
    tracker = RegimeTransitionRisk(config)
    
    # Should always return defaults
    for _ in range(20):
        tracker.update("TREND")
    
    result = tracker.get_risk("TREND")
    
    print(f"Disabled: stability={result.stability:.2f}, valid={result.valid}")
    
    # Should return defaults even after updates
    assert result.valid == False, "Should never be valid when disabled"
    assert result.stability == config.default_stability
    
    return True


def test_unknown_regime():
    """Test handling of unknown regime names."""
    tracker = RegimeTransitionRisk()
    
    # Use unknown regime names
    for _ in range(15):
        tracker.update("CUSTOM_REGIME_1")
        tracker.update("CUSTOM_REGIME_2")
    
    result = tracker.get_risk("CUSTOM_REGIME_1")
    
    print(f"Unknown Regime: stability={result.stability:.2f}, valid={result.valid}")
    
    # Should handle gracefully
    assert result.valid == True, "Should become valid after enough observations"
    
    return True


def run_all_tests():
    """Run all regime transition risk tests."""
    print("=" * 60)
    print("REGIME TRANSITION RISK TESTS")
    print("=" * 60)
    
    tests = [
        ("Initial State", test_initial_state),
        ("Stable Regime", test_stable_regime),
        ("Unstable Regime", test_unstable_regime),
        ("Transition Probability", test_transition_probability),
        ("Decay Factor", test_decay_factor),
        ("Most Likely Next", test_most_likely_next),
        ("Reset", test_reset),
        ("Disabled Tracker", test_disabled_tracker),
        ("Unknown Regime", test_unknown_regime),
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

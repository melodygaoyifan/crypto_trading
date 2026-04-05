"""
HMATS v6.4.1 - Signal Quality Scorer Tests
==========================================

Tests for the SignalQualityScorer module.
Verifies score ranges and multiplier bounds across various market conditions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.signal_quality_scorer import (
    SignalQualityScorer,
    SignalQualityScorerConfig,
    SignalQualityResult,
)


def test_trending_high_quality():
    """Test high-quality trending market scenario."""
    scorer = SignalQualityScorer()
    
    result = scorer.score(
        regime="STRONG_TREND",
        regime_confidence=0.8,
        trend_direction=1.0,      # Strong uptrend
        signal_direction=1.0,      # Long signal (aligned)
        volatility_regime="NORMAL",
        volume_ratio=1.5,          # Above average volume
        transition_stability=0.8,  # Stable regime
        funding_rate=0.02,         # Moderate funding
        signal_side=1,
    )
    
    print(f"Trending High Quality: score={result.score:.1f}, "
          f"conviction={result.conviction_multiplier:.2f}, "
          f"urgency={result.execution_urgency:.2f}")
    
    # High quality scenario should score high
    assert 60 <= result.score <= 100, f"Expected high score, got {result.score}"
    assert result.conviction_multiplier >= 1.0, "Should have high conviction"
    assert result.execution_urgency >= 0.9, "Should have reasonable urgency"
    assert result.allow_add == True, "Should allow add-on"
    
    return True


def test_choppy_low_quality():
    """Test low-quality choppy market scenario."""
    scorer = SignalQualityScorer()
    
    result = scorer.score(
        regime="VOLATILE_CHOP",
        regime_confidence=0.4,
        trend_direction=0.1,       # Weak/no trend
        signal_direction=1.0,      # Long signal
        volatility_regime="HIGH",
        volume_ratio=0.6,          # Below average volume
        transition_stability=0.3,  # Unstable
        funding_rate=0.20,         # Extreme funding (crowded)
        signal_side=1,             # Going with crowd
    )
    
    print(f"Choppy Low Quality: score={result.score:.1f}, "
          f"conviction={result.conviction_multiplier:.2f}, "
          f"urgency={result.execution_urgency:.2f}")
    
    # Low quality should score low
    assert 20 <= result.score <= 60, f"Expected lower score, got {result.score}"
    assert result.conviction_multiplier <= 1.0, "Should have reduced conviction"
    
    return True


def test_counter_trend():
    """Test counter-trend signal."""
    scorer = SignalQualityScorer()
    
    result = scorer.score(
        regime="TREND",
        regime_confidence=0.7,
        trend_direction=1.0,       # Uptrend
        signal_direction=-1.0,     # Short signal (counter-trend)
        volatility_regime="NORMAL",
        volume_ratio=1.0,
        transition_stability=0.6,
        funding_rate=0.0,
        signal_side=-1,
    )
    
    print(f"Counter Trend: score={result.score:.1f}, "
          f"conviction={result.conviction_multiplier:.2f}")
    
    # Counter-trend should be penalized
    assert result.score < 70, "Counter-trend should have moderate score"
    breakdown = result.breakdown
    assert breakdown.get("trend_alignment", 0) < 15, "Trend alignment should be low"
    
    return True


def test_extreme_funding_counter():
    """Test counter-funding trade (should get bonus)."""
    scorer = SignalQualityScorer()
    
    # Extreme positive funding (crowded longs) + short signal
    result = scorer.score(
        regime="NEUTRAL",
        regime_confidence=0.6,
        trend_direction=0.0,
        signal_direction=-1.0,
        volatility_regime="NORMAL",
        volume_ratio=1.0,
        transition_stability=0.5,
        funding_rate=0.18,         # Extreme positive (longs pay)
        signal_side=-1,            # Going short (counter-crowding)
    )
    
    print(f"Counter Funding: score={result.score:.1f}, "
          f"conviction={result.conviction_multiplier:.2f}")
    
    # Counter-funding should get bonus (weight=5, counter multiplier=0.8 -> score=4)
    funding_score = result.breakdown.get("funding_bias", 0)
    assert funding_score >= 3, f"Should have funding bonus, got {funding_score}"
    
    return True


def test_multiplier_bounds():
    """Test that multipliers stay within bounds."""
    scorer = SignalQualityScorer()
    
    # Extreme high quality
    result_high = scorer.score(
        regime="STRONG_TREND",
        regime_confidence=1.0,
        trend_direction=1.0,
        signal_direction=1.0,
        volatility_regime="CALM",
        volume_ratio=2.0,
        transition_stability=1.0,
        funding_rate=-0.15,  # Counter-funding bonus
        signal_side=1,
    )
    
    # Extreme low quality
    result_low = scorer.score(
        regime="VOLATILE_CHOP",
        regime_confidence=0.2,
        trend_direction=1.0,
        signal_direction=-1.0,
        volatility_regime="EXTREME",
        volume_ratio=0.3,
        transition_stability=0.1,
        funding_rate=0.25,  # Extremely crowded
        signal_side=-1,
    )
    
    print(f"Bounds check - High: conv={result_high.conviction_multiplier:.2f}, "
          f"Low: conv={result_low.conviction_multiplier:.2f}")
    
    # Check bounds
    config = SignalQualityScorerConfig()
    assert config.conviction_multiplier_min <= result_low.conviction_multiplier <= config.conviction_multiplier_max
    assert config.conviction_multiplier_min <= result_high.conviction_multiplier <= config.conviction_multiplier_max
    assert config.urgency_min <= result_low.execution_urgency <= config.urgency_max
    assert config.urgency_min <= result_high.execution_urgency <= config.urgency_max
    
    return True


def test_disabled_scorer():
    """Test scorer when disabled."""
    config = SignalQualityScorerConfig(enabled=False)
    scorer = SignalQualityScorer(config)
    
    result = scorer.score(
        regime="STRONG_TREND",
        regime_confidence=0.9,
    )
    
    # Should return default values
    assert result.score == config.default_score_on_error
    assert result.conviction_multiplier == 1.0
    assert result.execution_urgency == 1.0
    
    print("Disabled scorer test passed")
    return True


def test_asset_side_override_lowers_addon_threshold():
    """Asset+side-specific add-on threshold should override the global default."""
    config = SignalQualityScorerConfig(
        min_score_for_add=101.0,
        min_score_for_add_by_asset_side={"BTC:LONG": 55.0},
    )
    scorer = SignalQualityScorer(config)

    btc_result = scorer.score(
        regime="STRONG_TREND",
        regime_confidence=0.7,
        trend_direction=1.0,
        signal_direction=1.0,
        volatility_regime="NORMAL",
        volume_ratio=1.0,
        transition_stability=0.7,
        funding_rate=0.0,
        signal_side=1,
        asset="BTC",
    )
    eth_result = scorer.score(
        regime="STRONG_TREND",
        regime_confidence=0.7,
        trend_direction=1.0,
        signal_direction=1.0,
        volatility_regime="NORMAL",
        volume_ratio=1.0,
        transition_stability=0.7,
        funding_rate=0.0,
        signal_side=1,
        asset="ETH",
    )

    assert btc_result.allow_add is True
    assert eth_result.allow_add is False
    assert btc_result.raw_inputs["min_score_for_add"] == 55.0
    assert eth_result.raw_inputs["min_score_for_add"] == 101.0


def run_all_tests():
    """Run all signal quality scorer tests."""
    print("=" * 60)
    print("SIGNAL QUALITY SCORER TESTS")
    print("=" * 60)
    
    tests = [
        ("Trending High Quality", test_trending_high_quality),
        ("Choppy Low Quality", test_choppy_low_quality),
        ("Counter Trend", test_counter_trend),
        ("Extreme Funding Counter", test_extreme_funding_counter),
        ("Multiplier Bounds", test_multiplier_bounds),
        ("Disabled Scorer", test_disabled_scorer),
        ("Asset Side Add Threshold", test_asset_side_override_lowers_addon_threshold),
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
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

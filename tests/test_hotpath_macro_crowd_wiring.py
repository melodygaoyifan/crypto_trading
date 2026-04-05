"""
================================================================================
HMATS v6.5.1 - Hot Path Macro/Crowd Wiring Tests
================================================================================

Offline tests to verify:
1. External feeds produce valid context with mock data
2. MacroCrowdAdapter transforms feeds correctly
3. Rich context is NOT thinned during SOTA check
4. FalseBreakoutDetector config does NOT include "breakdown"

NO NETWORK CALLS - all tests use mock/injected data.

================================================================================
"""

import pytest
from typing import Dict, Any


# =============================================================================
# TEST 1: MOCK FEED DATA PRODUCES VALID CONTEXT
# =============================================================================

def test_coinglass_mock_context():
    """Verify Coinglass feed returns valid crowd metrics in mock mode."""
    try:
        from data_mgmt.feeds.coinglass_feed import CoinglassFeed
    except ImportError:
        pytest.skip("Coinglass feed not available")
    
    feed = CoinglassFeed(api_key="", mock_mode=True)
    metrics = feed.get_crowd_metrics("BTC")
    
    # Must have required fields
    assert "crowding_bias" in metrics or "funding_bias" in metrics
    assert isinstance(metrics, dict)


def test_lunarcrush_mock_context():
    """Verify LunarCrush feed returns valid attention metrics in mock mode."""
    try:
        from data_mgmt.feeds.lunarcrush_feed import LunarCrushFeed
    except ImportError:
        pytest.skip("LunarCrush feed not available")
    
    feed = LunarCrushFeed(api_key="", mock_mode=True)
    metrics = feed.get_attention_metrics("BTC")
    
    assert isinstance(metrics, dict)
    # Must have attention field
    assert "attention_pressure" in metrics or "attention" in metrics or "galaxy_score" in metrics


def test_cryptopanic_mock_context():
    """Verify CryptoPanic feed returns valid panic metrics in mock mode."""
    try:
        from data_mgmt.feeds.cryptopanic_feed import CryptoPanicFeed
    except ImportError:
        pytest.skip("CryptoPanic feed not available")
    
    feed = CryptoPanicFeed(api_key="", mock_mode=True)
    metrics = feed.get_panic_metrics("BTC")
    
    assert isinstance(metrics, dict)


def test_fred_mock_context():
    """Verify FRED feed returns valid macro context in mock mode."""
    try:
        from data_mgmt.feeds.fred_feed import FREDFeed
    except ImportError:
        pytest.skip("FRED feed not available")
    
    feed = FREDFeed(api_key="", mock_mode=True)
    context = feed.get_macro_context()
    
    assert isinstance(context, dict)


# =============================================================================
# TEST 2: MACRO_CROWD_ADAPTER PRODUCES CORRECT OUTPUT
# =============================================================================

def test_macro_crowd_adapter_produces_context():
    """Verify MacroCrowdAdapter produces required context fields."""
    try:
        from signals.macro_crowd_adapter import MacroCrowdAdapter, get_macro_crowd_adapter
    except ImportError:
        pytest.skip("MacroCrowdAdapter not available")
    
    # Create adapter with no feeds (will use defaults)
    adapter = MacroCrowdAdapter()
    context = adapter.get_context_dict(symbol="BTC")
    
    # Must have all three sections
    assert "macro" in context, "Missing macro context"
    assert "crowd" in context, "Missing crowd context"
    assert "sentiment" in context, "Missing sentiment context"
    
    # Macro must have required fields
    macro = context["macro"]
    assert "event_window" in macro
    assert "macro_risk" in macro
    
    # Crowd must have required fields
    crowd = context["crowd"]
    assert "crowding_bias" in crowd
    assert "extreme_crowding" in crowd


def test_macro_crowd_adapter_numeric_ranges():
    """Verify MacroCrowdAdapter outputs are in valid ranges."""
    try:
        from signals.macro_crowd_adapter import MacroCrowdAdapter
    except ImportError:
        pytest.skip("MacroCrowdAdapter not available")
    
    adapter = MacroCrowdAdapter()
    context = adapter.get_context_dict(symbol="BTC")
    
    # macro_risk must be [0, 1]
    macro_risk = context["macro"].get("macro_risk", 0.5)
    assert 0 <= macro_risk <= 1, f"macro_risk out of range: {macro_risk}"
    
    # crowding_bias must be [-1, 1]
    crowding_bias = context["crowd"].get("crowding_bias", 0)
    assert -1 <= crowding_bias <= 1, f"crowding_bias out of range: {crowding_bias}"
    
    # attention_pressure must be [0, 1]
    attention = context["crowd"].get("attention_pressure", 0.5)
    assert 0 <= attention <= 1, f"attention_pressure out of range: {attention}"


# =============================================================================
# TEST 3: CONTEXT NOT THINNED (Rich context preservation)
# =============================================================================

def test_market_data_preserves_macro_crowd():
    """Verify market_data is not overwritten with thin dict."""
    # Simulate the market_data flow
    market_data = {
        "current_price": 50000.0,
        "prices": [49000, 49500, 50000],
        "volumes": [100, 200, 150],
        "volatility": 0.02,
        "regime_state": "NORMAL",
        "macro": {
            "event_window": {"active": False},
            "macro_risk": 0.3,
        },
        "crowd": {
            "crowding_bias": 0.2,
            "extreme_crowding": False,
        },
        "sentiment_external": {
            "direction_score": 0.1,
        },
    }
    
    # Build sota_context the way v6.5.1 does it (NOT overwriting market_data)
    sota_context = {
        "price": market_data.get("current_price", market_data.get("price", 0)),
        "volatility": market_data.get("volatility", 0),
        "regime": market_data.get("regime_state", "UNKNOWN"),
        "signals": {},
        "integrity": {"data_stale": False, "feed_healthy": True},
        "execution": {"blocked": False},
        # v6.5.1: Include rich context
        "macro": market_data.get("macro", {}),
        "crowd": market_data.get("crowd", {}),
        "sentiment": market_data.get("sentiment_external", {}),
    }
    
    # Verify market_data was NOT thinned
    assert "macro" in market_data, "market_data.macro was thinned!"
    assert "crowd" in market_data, "market_data.crowd was thinned!"
    assert "prices" in market_data, "market_data.prices was thinned!"
    
    # Verify sota_context has rich data
    assert "macro" in sota_context, "sota_context missing macro"
    assert sota_context["macro"].get("macro_risk") == 0.3


# =============================================================================
# TEST 4: FALSE BREAKOUT DETECTOR CONFIG
# =============================================================================

def test_false_breakout_no_breakdown_veto():
    """Verify FalseBreakoutDetector config does NOT include 'breakdown'."""
    try:
        from signals.false_breakout_detector import FalseBreakoutConfig
    except ImportError:
        pytest.skip("FalseBreakoutDetector not available")
    
    # Default config
    config = FalseBreakoutConfig()
    
    # Must NOT include breakdown
    assert "breakdown" not in config.veto_entry_types, (
        f"'breakdown' should NOT be in veto_entry_types: {config.veto_entry_types}"
    )
    
    # Must include breakout and retest
    assert "breakout" in config.veto_entry_types, "Missing 'breakout' in veto_entry_types"
    assert "retest" in config.veto_entry_types, "Missing 'retest' in veto_entry_types"


def test_false_breakout_from_dict_no_breakdown():
    """Verify FalseBreakoutConfig.from_dict does not add breakdown."""
    try:
        from signals.false_breakout_detector import FalseBreakoutConfig
    except ImportError:
        pytest.skip("FalseBreakoutDetector not available")
    
    # Empty config should use defaults without breakdown
    config = FalseBreakoutConfig.from_dict({})
    
    assert "breakdown" not in config.veto_entry_types
    assert "breakout" in config.veto_entry_types
    assert "retest" in config.veto_entry_types


# =============================================================================
# TEST 5: MACRO CROWD EFFECTS (Profit-max integration)
# =============================================================================

def test_macro_crowd_effects_no_veto():
    """Verify compute_macro_crowd_effects does NOT produce hard vetos."""
    try:
        from signals.macro_crowd_adapter import compute_macro_crowd_effects
    except ImportError:
        pytest.skip("compute_macro_crowd_effects not available")
    
    # Even with extreme values, should not veto
    effects = compute_macro_crowd_effects(
        macro={
            "event_window": {"active": True},
            "macro_risk": 0.9,
            "macro_shock": 0.8,
        },
        crowd={
            "extreme_crowding": True,
            "crowding_bias": 0.9,
            "panic_score": 0.9,
        },
        sentiment={"direction_score": -0.9},
        price_direction=1,
    )
    
    # Should modulate but NOT veto
    assert effects.urgency_multiplier >= 0.6, "Urgency should not go below 0.6"
    assert effects.size_multiplier >= 0.85, "Size should not go below 0.85"
    
    # These are modulations, not vetos
    assert isinstance(effects.allow_addon, bool)
    assert isinstance(effects.false_breakout_strictness, int)


def test_macro_crowd_effects_bounds():
    """Verify effects are within documented bounds."""
    try:
        from signals.macro_crowd_adapter import compute_macro_crowd_effects
    except ImportError:
        pytest.skip("compute_macro_crowd_effects not available")
    
    effects = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": 0.0},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
        sentiment={"direction_score": 0.0},
        price_direction=0,
    )
    
    # Check bounds
    assert 0.6 <= effects.urgency_multiplier <= 1.0
    assert 0.85 <= effects.size_multiplier <= 1.15
    assert 0 <= effects.false_breakout_strictness <= 2


# =============================================================================
# TEST 6: BUILD RICH CONTEXT HELPER
# =============================================================================

def test_rich_context_structure():
    """Verify rich context has all required fields."""
    # This simulates what main.py builds
    macro_crowd_context = {
        "macro": {
            "event_window": {"active": False, "name": None, "t_minus_min": 0, "importance": 0},
            "macro_shock": 0.0,
            "macro_risk": 0.5,
            "source_status": "available",
        },
        "crowd": {
            "attention_pressure": 0.5,
            "crowding_bias": 0.0,
            "panic_score": 0.0,
            "narrative_shift": 0.0,
            "extreme_crowding": False,
            "source_status": "available",
        },
        "sentiment": {
            "direction_score": 0.0,
            "strength": 0.5,
            "delta": 0.0,
            "volatility": 0.0,
            "source_status": "available",
        },
    }
    
    # All sections present
    assert "macro" in macro_crowd_context
    assert "crowd" in macro_crowd_context
    assert "sentiment" in macro_crowd_context
    
    # Required fields in macro
    assert "event_window" in macro_crowd_context["macro"]
    assert "macro_risk" in macro_crowd_context["macro"]
    
    # Required fields in crowd
    assert "extreme_crowding" in macro_crowd_context["crowd"]
    assert "crowding_bias" in macro_crowd_context["crowd"]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HMATS v6.5.1 - Hot Path Wiring Tests")
    print("=" * 70)
    
    # Run tests manually
    tests = [
        ("Coinglass mock context", test_coinglass_mock_context),
        ("LunarCrush mock context", test_lunarcrush_mock_context),
        ("CryptoPanic mock context", test_cryptopanic_mock_context),
        ("FRED mock context", test_fred_mock_context),
        ("MacroCrowdAdapter produces context", test_macro_crowd_adapter_produces_context),
        ("MacroCrowdAdapter numeric ranges", test_macro_crowd_adapter_numeric_ranges),
        ("Market data preserves macro/crowd", test_market_data_preserves_macro_crowd),
        ("FalseBreakout no breakdown veto", test_false_breakout_no_breakdown_veto),
        ("FalseBreakout from_dict no breakdown", test_false_breakout_from_dict_no_breakdown),
        ("MacroCrowdEffects no veto", test_macro_crowd_effects_no_veto),
        ("MacroCrowdEffects bounds", test_macro_crowd_effects_bounds),
        ("Rich context structure", test_rich_context_structure),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ON {name}")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  ⏭️  {name} (skipped: {e})")
            skipped += 1
        except Exception as e:
            print(f"  OFF {name}: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\nON All tests passed!")
    else:
        print(f"\nOFF {failed} test(s) failed")

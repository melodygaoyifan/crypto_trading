"""
================================================================================
HMATS v6.5 - Macro/Crowd Integration Test Suite
================================================================================

Comprehensive tests for:
- FRED event window encoding (TASK F)
- Trading Economics macro_shock mapping (TASK F)
- Coinglass crowding normalization (TASK F)
- LunarCrush attention quantization (TASK F)
- CryptoPanic panic score mapping (TASK F)
- Wiring test: context fields exist and numeric (TASK F)
- Safe defaults when API unavailable (TASK F)
- Profit-max integration tests

================================================================================
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# TEST UTILITIES
# =============================================================================

def assert_in_range(value: float, min_val: float, max_val: float, name: str):
    """Assert value is in range."""
    assert min_val <= value <= max_val, f"{name}={value} not in [{min_val}, {max_val}]"


def assert_context_field_numeric(ctx: Dict, path: str):
    """Assert a nested field exists and is numeric."""
    import numpy as np
    parts = path.split(".")
    current = ctx
    for part in parts:
        assert part in current, f"Missing field: {path}"
        current = current[part]
    # Accept numpy types as well
    is_numeric = isinstance(current, (int, float, bool, np.integer, np.floating, np.bool_))
    assert is_numeric, f"{path} is not numeric: {type(current)}"


# =============================================================================
# TEST 1: FRED Event Window Encoding
# =============================================================================

def test_fred_event_window():
    """Test FRED event window computation."""
    print("\n" + "=" * 70)
    print("TEST 1: FRED Event Window Encoding")
    print("=" * 70)
    
    from data_mgmt.feeds.fred_feed import FREDFeed, FREDRelease, MacroEventWindow
    
    feed = FREDFeed(mock_mode=True)
    
    # Test event window computation
    now = datetime.now(timezone.utc)
    
    # Create test releases
    releases = [
        FREDRelease(
            release_id=1,
            name="CPI Release",
            release_date=now + timedelta(minutes=60),  # 60 min in future
            importance=3,
        ),
        FREDRelease(
            release_id=2,
            name="GDP Estimate",
            release_date=now + timedelta(hours=24),  # 24 hours in future
            importance=2,
        ),
    ]
    
    # Test: Within window
    window = feed._compute_event_window(releases, now)
    assert window.active, "Event window should be active (CPI in 60 min)"
    assert window.name == "CPI Release"
    assert window.importance == 3
    assert 50 <= window.t_minus_min <= 70, f"t_minus_min should be ~60, got {window.t_minus_min}"
    
    # Test: Outside window
    releases_far = [
        FREDRelease(
            release_id=3,
            name="Far Event",
            release_date=now + timedelta(hours=24),
            importance=3,
        ),
    ]
    window_far = feed._compute_event_window(releases_far, now)
    assert not window_far.active, "Event window should not be active (24h away)"
    
    print("  ON Event window encoding correct")
    print("  ON t_minus_min computed correctly")
    print("  ON Importance preserved")


# =============================================================================
# TEST 2: Trading Economics Macro Shock
# =============================================================================

def test_trading_economics_macro_shock():
    """Test Trading Economics macro_shock calculation."""
    print("\n" + "=" * 70)
    print("TEST 2: Trading Economics Macro Shock Mapping")
    print("=" * 70)
    
    from data_mgmt.feeds.trading_economics_feed import (
        TradingEconomicsFeed, EconomicEvent, EventImportance, TradingEconomicsData
    )
    import math
    
    feed = TradingEconomicsFeed(mock_mode=True)
    now = datetime.now(timezone.utc)
    
    # Test shock calculation
    # shock_raw = (actual - forecast) / |forecast|
    # macro_shock = tanh(shock_raw * k)
    
    # Create event with known values
    event = EconomicEvent(
        event_id="test",
        name="CPI",
        country="US",
        category="Inflation",
        importance=EventImportance.HIGH,
        event_time=now - timedelta(hours=1),
        actual=3.5,
        forecast=3.0,
        previous=3.2,
    )
    event.compute_surprise()
    
    # Verify surprise calculation
    assert event.surprise is not None
    expected_surprise = 3.5 - 3.0
    assert abs(event.surprise - expected_surprise) < 0.001
    
    expected_surprise_pct = 0.5 / 3.0  # ~0.167
    assert abs(event.surprise_pct - expected_surprise_pct) < 0.001
    
    # Verify macro_shock via feed
    data = TradingEconomicsData(timestamp=now, staleness_sec=0.0)
    data.recent_events = [event]
    feed._compute_macro_shock(data, now)
    
    # Expected: tanh(0.167 * 2) ≈ 0.321
    expected_shock = math.tanh(expected_surprise_pct * 2.0)
    assert abs(data.macro_shock - expected_shock) < 0.01, f"Expected {expected_shock}, got {data.macro_shock}"
    
    assert_in_range(data.macro_shock, -1.0, 1.0, "macro_shock")
    
    print(f"  ON Surprise computed: {event.surprise:.3f}")
    print(f"  ON Surprise %: {event.surprise_pct:.3f}")
    print(f"  ON Macro shock: {data.macro_shock:.3f}")
    print("  ON Shock in [-1, 1] range")


# =============================================================================
# TEST 3: Coinglass Crowding Normalization
# =============================================================================

def test_coinglass_crowding():
    """Test Coinglass crowding metric normalization."""
    print("\n" + "=" * 70)
    print("TEST 3: Coinglass Crowding Normalization")
    print("=" * 70)
    
    from data_mgmt.feeds.coinglass_feed import CoinglassFeed
    
    feed = CoinglassFeed(mock_mode=True)
    
    # Fetch mock data
    data = asyncio.run(feed.fetch())
    
    for symbol in ["BTC", "ETH", "SOL"]:
        # Funding bias
        fb = data.funding_bias.get(symbol, 0.0)
        assert_in_range(fb, -1.0, 1.0, f"{symbol} funding_bias")
        
        # OI trend
        oi = data.oi_trend.get(symbol, 0.0)
        assert_in_range(oi, -1.0, 1.0, f"{symbol} oi_trend")
        
        # Liquidation imbalance
        li = data.liquidation_imbalance.get(symbol, 0.0)
        assert_in_range(li, -1.0, 1.0, f"{symbol} liquidation_imbalance")
        
        # Crowding score
        cs = data.crowding_score.get(symbol, 0.0)
        assert_in_range(cs, 0.0, 1.0, f"{symbol} crowding_score")
        
        print(f"  ON {symbol}: fb={fb:.2f}, oi={oi:.2f}, li={li:.2f}, crowd={cs:.2f}")
    
    # Test crowd metrics interface
    metrics = feed.get_crowd_metrics("BTC")
    assert "crowding_score" in metrics
    assert "extreme_crowding" in metrics

    print("  ON All Coinglass metrics normalized correctly")


# =============================================================================
# TEST 4: LunarCrush Attention Quantization
# =============================================================================

def test_lunarcrush_attention():
    """Test LunarCrush attention metric quantization."""
    print("\n" + "=" * 70)
    print("TEST 4: LunarCrush Attention Quantization")
    print("=" * 70)
    
    from data_mgmt.feeds.lunarcrush_feed import LunarCrushFeed
    
    feed = LunarCrushFeed(mock_mode=True)
    
    # Fetch mock data
    data = asyncio.run(feed.fetch())
    
    for symbol in ["BTC", "ETH", "SOL"]:
        # Attention pressure [0, 1]
        ap = data.attention_pressure.get(symbol, 0.5)
        assert_in_range(ap, 0.0, 1.0, f"{symbol} attention_pressure")
        
        # Crowding bias [-1, 1]
        cb = data.crowding_bias.get(symbol, 0.0)
        assert_in_range(cb, -1.0, 1.0, f"{symbol} crowding_bias")
        
        # Narrative shift [-1, 1]
        ns = data.narrative_shift.get(symbol, 0.0)
        assert_in_range(ns, -1.0, 1.0, f"{symbol} narrative_shift")
        
        # Extreme attention (bool)
        ea = data.extreme_attention.get(symbol, False)
        assert isinstance(ea, bool)
        
        print(f"  ON {symbol}: attn={ap:.2f}, crowd={cb:.2f}, narr={ns:.2f}, extreme={ea}")
    
    # Test attention metrics interface
    metrics = feed.get_attention_metrics("BTC")
    assert "attention_pressure" in metrics
    assert "extreme_attention" in metrics

    print("  ON All LunarCrush metrics quantized correctly")


# =============================================================================
# TEST 5: CryptoPanic Panic Score
# =============================================================================

def test_cryptopanic_panic():
    """Test CryptoPanic panic score mapping."""
    print("\n" + "=" * 70)
    print("TEST 5: CryptoPanic Panic Score Mapping")
    print("=" * 70)
    
    from data_mgmt.feeds.cryptopanic_feed import CryptoPanicFeed
    
    feed = CryptoPanicFeed(mock_mode=True)
    
    # Fetch mock data
    data = asyncio.run(feed.fetch())
    
    for currency in ["BTC", "ETH", "SOL"]:
        # Panic score [0, 1]
        ps = data.panic_score.get(currency, 0.0)
        assert_in_range(ps, 0.0, 1.0, f"{currency} panic_score")
        
        # News velocity (normalized)
        nv = data.news_velocity.get(currency, 0.0)
        assert isinstance(nv, (int, float))
        
        # Sentiment consensus [-1, 1]
        sc = data.sentiment_consensus.get(currency, 0.0)
        assert_in_range(sc, -1.0, 1.0, f"{currency} sentiment_consensus")
        
        # Narrative intensity [0, 1]
        ni = data.narrative_intensity.get(currency, 0.0)
        assert_in_range(ni, 0.0, 1.0, f"{currency} narrative_intensity")
        
        print(f"  ON {currency}: panic={ps:.2f}, velocity={nv:.2f}, consensus={sc:.2f}")
    
    # Global metrics
    assert_in_range(data.global_panic, 0.0, 1.0, "global_panic")
    
    print("  ON All CryptoPanic metrics mapped correctly")


# =============================================================================
# TEST 6: Context Fields Existence and Numeric
# =============================================================================

def test_context_fields():
    """Test that all context fields exist and are numeric."""
    print("\n" + "=" * 70)
    print("TEST 6: Context Fields Existence & Numeric")
    print("=" * 70)
    
    from signals.macro_crowd_adapter import MacroCrowdAdapter
    
    adapter = MacroCrowdAdapter()
    ctx = adapter.get_context_dict("BTC")
    
    # Required macro fields
    assert_context_field_numeric(ctx, "macro.event_window.active")
    assert_context_field_numeric(ctx, "macro.event_window.importance")
    assert_context_field_numeric(ctx, "macro.macro_shock")
    assert_context_field_numeric(ctx, "macro.macro_risk")
    print("  ON macro.* fields present and numeric")
    
    # Required crowd fields
    assert_context_field_numeric(ctx, "crowd.attention_pressure")
    assert_context_field_numeric(ctx, "crowd.crowding_bias")
    assert_context_field_numeric(ctx, "crowd.panic_score")
    assert_context_field_numeric(ctx, "crowd.narrative_shift")
    assert_context_field_numeric(ctx, "crowd.extreme_crowding")
    print("  ON crowd.* fields present and numeric")
    
    # Required sentiment fields
    assert_context_field_numeric(ctx, "sentiment.direction_score")
    assert_context_field_numeric(ctx, "sentiment.strength")
    assert_context_field_numeric(ctx, "sentiment.delta")
    assert_context_field_numeric(ctx, "sentiment.volatility")
    print("  ON sentiment.* fields present and numeric")
    
    # Verify value ranges (TASK C contract)
    assert_in_range(ctx["macro"]["macro_shock"], -1.0, 1.0, "macro.macro_shock")
    assert_in_range(ctx["macro"]["macro_risk"], 0.0, 1.0, "macro.macro_risk")
    assert_in_range(ctx["crowd"]["attention_pressure"], 0.0, 1.0, "crowd.attention_pressure")
    assert_in_range(ctx["crowd"]["crowding_bias"], -1.0, 1.0, "crowd.crowding_bias")
    assert_in_range(ctx["crowd"]["panic_score"], 0.0, 1.0, "crowd.panic_score")
    assert_in_range(ctx["sentiment"]["direction_score"], -1.0, 1.0, "sentiment.direction_score")
    print("  ON All value ranges conform to TASK C contract")


# =============================================================================
# TEST 7: Safe Defaults When API Unavailable
# =============================================================================

def test_safe_defaults():
    """Test safe defaults when feeds are unavailable."""
    print("\n" + "=" * 70)
    print("TEST 7: Safe Defaults (API Unavailable)")
    print("=" * 70)
    
    from signals.macro_crowd_adapter import (
        MacroCrowdAdapter, MacroContext, CrowdContext, SentimentContext
    )
    
    # Create adapter with no feeds
    adapter = MacroCrowdAdapter(
        fred_feed=None,
        coinglass_feed=None,
        lunarcrush_feed=None,
        cryptopanic_feed=None,
        sentiment_engine=None,
    )
    
    # Compute should succeed with defaults
    output = adapter.compute("BTC")
    
    # Verify defaults are safe (neutral values)
    assert output.macro.macro_shock == 0.0, "Default macro_shock should be 0"
    assert 0.4 <= output.macro.macro_risk <= 0.6, "Default macro_risk should be ~0.5"
    assert not output.macro.event_window["active"], "Default event_window should be inactive"
    
    assert output.crowd.attention_pressure == 0.5, "Default attention should be 0.5"
    assert output.crowd.crowding_bias == 0.0, "Default crowding_bias should be 0"
    assert not output.crowd.extreme_crowding, "Default extreme_crowding should be False"
    
    assert output.sentiment.direction_score == 0.0, "Default direction should be 0"
    assert output.sentiment.strength == 0.5, "Default strength should be 0.5"
    
    print("  ON macro defaults safe")
    print("  ON crowd defaults safe")
    print("  ON sentiment defaults safe")
    print("  ON System operates safely without API data")


# =============================================================================
# TEST 8: Profit-Max Integration (NO NEW VETOS)
# =============================================================================

def test_profit_max_no_new_vetos():
    """Test that macro/crowd never creates new vetos."""
    print("\n" + "=" * 70)
    print("TEST 8: Profit-Max Integration (NO NEW VETOS)")
    print("=" * 70)
    
    from signals.profit_max_adapter import (
        ProfitMaxAdapter, ProfitMaxConfig, VetoSource, reset_profit_max_adapter
    )
    
    reset_profit_max_adapter()
    
    config = ProfitMaxConfig(
        macro_crowd_enabled=True,
        false_breakout_enabled=False,  # Disable to isolate test
        loss_streak_enabled=False,
    )
    
    adapter = ProfitMaxAdapter(config)
    
    # Test 1: Extreme macro conditions should NOT veto
    result = adapter.evaluate(
        regime="TREND",
        direction=1,
        macro={
            "event_window": {"active": True, "name": "FOMC", "importance": 3},
            "macro_risk": 1.0,  # Maximum risk
            "macro_shock": -1.0,  # Maximum adverse shock
        },
        crowd={
            "extreme_crowding": True,
            "crowding_bias": 1.0,
            "panic_score": 1.0,
        },
        sentiment={
            "direction_score": -1.0,  # Maximum disagreement
            "strength": 1.0,
        },
    )
    
    assert not result.veto, "Macro/crowd should NEVER veto"
    assert result.veto_source == VetoSource.NONE
    print("  ON Extreme macro/crowd does NOT veto")
    
    # Test 2: Verify modulation is applied
    assert result.execution_urgency < 1.0, "Urgency should be reduced"
    assert result.allow_add == False, "Add-on should be blocked"
    print(f"  ON Urgency modulated: {result.execution_urgency:.2f}")
    print(f"  ON Add-on blocked: allow_add={result.allow_add}")
    
    # Test 3: Size multiplier affected
    assert result.size_multiplier < 1.0, "Size should be reduced for misaligned shock"
    print(f"  ON Size modulated: {result.size_multiplier:.2f}")
    
    # Test 4: Breakdown should document macro/crowd effects
    assert "macro_event" in result.breakdown or "crowd_extreme" in result.breakdown
    print("  ON Effects documented in breakdown")


# =============================================================================
# TEST 9: Macro Shock Alignment
# =============================================================================

def test_macro_shock_alignment():
    """Test macro shock size scaling based on alignment."""
    print("\n" + "=" * 70)
    print("TEST 9: Macro Shock Alignment Scaling")
    print("=" * 70)
    
    from signals.profit_max_adapter import ProfitMaxAdapter, ProfitMaxConfig, reset_profit_max_adapter
    
    reset_profit_max_adapter()
    
    config = ProfitMaxConfig(
        macro_crowd_enabled=True,
        macro_shock_threshold=0.4,
        macro_shock_aligned_size_mult=1.15,
        macro_shock_misaligned_size_mult=0.85,
    )
    
    adapter = ProfitMaxAdapter(config)
    
    # Test aligned: positive shock + long position
    result_aligned = adapter.evaluate(
        regime="TREND",
        direction=1,  # Long
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.5},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
    )
    
    # Test misaligned: positive shock + short position
    result_misaligned = adapter.evaluate(
        regime="TREND",
        direction=-1,  # Short
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.5},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
    )
    
    # Aligned should increase size
    assert result_aligned.size_multiplier > 1.0, f"Aligned size should increase, got {result_aligned.size_multiplier}"
    
    # Misaligned should decrease size
    assert result_misaligned.size_multiplier < 1.0, f"Misaligned size should decrease, got {result_misaligned.size_multiplier}"
    
    print(f"  ON Aligned (long + positive shock): size={result_aligned.size_multiplier:.2f}")
    print(f"  ON Misaligned (short + positive shock): size={result_misaligned.size_multiplier:.2f}")


# =============================================================================
# TEST 10: Wiring Test - Full Pipeline
# =============================================================================

def test_full_pipeline_wiring():
    """Test full pipeline wiring from feeds to profit_max."""
    print("\n" + "=" * 70)
    print("TEST 10: Full Pipeline Wiring")
    print("=" * 70)
    
    from signals.macro_crowd_adapter import MacroCrowdAdapter, compute_macro_crowd_effects
    from signals.profit_max_adapter import ProfitMaxAdapter, reset_profit_max_adapter
    
    reset_profit_max_adapter()
    
    # Create adapter (would use real feeds in production)
    macro_adapter = MacroCrowdAdapter()
    
    # Get context
    ctx = macro_adapter.get_context_dict("BTC")
    
    # Verify structure
    assert "macro" in ctx
    assert "crowd" in ctx
    assert "sentiment" in ctx
    
    # Compute effects
    effects = compute_macro_crowd_effects(
        macro=ctx["macro"],
        crowd=ctx["crowd"],
        sentiment=ctx["sentiment"],
        price_direction=1,
    )
    
    # Verify effects structure
    assert hasattr(effects, "urgency_multiplier")
    assert hasattr(effects, "size_multiplier")
    assert hasattr(effects, "allow_addon")
    assert hasattr(effects, "false_breakout_strictness")
    
    print(f"  ON Context: {list(ctx.keys())}")
    print(f"  ON Effects: urgency={effects.urgency_multiplier:.2f}, size={effects.size_multiplier:.2f}")
    
    # Now test through profit_max_adapter
    pma = ProfitMaxAdapter()
    result = pma.evaluate(
        regime="TREND",
        direction=1,
        macro=ctx["macro"],
        crowd=ctx["crowd"],
        sentiment=ctx["sentiment"],
    )
    
    assert not result.veto, "Pipeline should not create spurious vetos"
    print(f"  ON ProfitMaxAdapter result: veto={result.veto}, conv={result.conviction_multiplier:.2f}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all macro/crowd integration tests."""
    print("\n" + "=" * 70)
    print("HMATS v6.5 - MACRO/CROWD INTEGRATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("FRED Event Window", test_fred_event_window),
        ("Trading Economics Macro Shock", test_trading_economics_macro_shock),
        ("Coinglass Crowding", test_coinglass_crowding),
        ("LunarCrush Attention", test_lunarcrush_attention),
        ("CryptoPanic Panic", test_cryptopanic_panic),
        ("Context Fields", test_context_fields),
        ("Safe Defaults", test_safe_defaults),
        ("Profit-Max No Vetos", test_profit_max_no_new_vetos),
        ("Macro Shock Alignment", test_macro_shock_alignment),
        ("Full Pipeline Wiring", test_full_pipeline_wiring),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\nOFF FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\nON ALL TESTS PASSED!")
    else:
        print(f"\nOFF {failed} TEST(S) FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

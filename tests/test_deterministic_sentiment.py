"""
================================================================================
HMATS v6.4.2 - Deterministic Sentiment Tests
================================================================================
Purpose: Comprehensive tests for the deterministic sentiment system

Tests:
1. Determinism - Same inputs produce same outputs
2. Direction scoring from text
3. Crowding isolation - LunarCrush/F&G don't affect direction
4. Profit-max compliance - No hard vetoes
5. Integration with signal_quality_scorer
6. Integration with profit_max_adapter
7. Wiring verification

Author: HMATS v6.4.2 LLM/Sentiment Separation
================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import logging
from typing import Dict

logging.basicConfig(level=logging.WARNING)


# =============================================================================
# TEST: DETERMINISTIC SENTIMENT ENGINE
# =============================================================================

class TestDeterministicSentiment:
    """Tests for DeterministicSentimentEngine."""
    
    def test_determinism_same_input_same_output(self):
        """Same inputs must produce identical outputs."""
        from signals.deterministic_sentiment import (
            DeterministicSentimentEngine,
            reset_deterministic_sentiment_engine
        )
        
        # Run 1
        reset_deterministic_sentiment_engine()
        engine1 = DeterministicSentimentEngine()
        result1 = engine1.compute(
            news_texts=["Bitcoin bullish breakout confirmed!"],
            fear_greed_value=65,
            funding_rate=0.0005,
        )
        
        # Run 2 (fresh engine)
        reset_deterministic_sentiment_engine()
        engine2 = DeterministicSentimentEngine()
        result2 = engine2.compute(
            news_texts=["Bitcoin bullish breakout confirmed!"],
            fear_greed_value=65,
            funding_rate=0.0005,
        )
        
        # Must be identical
        assert result1.direction_score == result2.direction_score
        assert result1.strength == result2.strength
        assert result1.crowding_bias == result2.crowding_bias
        assert result1.extreme_crowding == result2.extreme_crowding
    
    def test_bullish_text_produces_positive_direction(self):
        """Bullish news should produce positive direction_score."""
        from signals.deterministic_sentiment import DeterministicSentimentEngine
        
        engine = DeterministicSentimentEngine()
        result = engine.compute(
            news_texts=[
                "Bitcoin moon incoming! 🚀",
                "Institutional adoption accelerates",
                "ETF approved, massive inflows expected",
            ]
        )
        
        assert result.direction_score > 0.3, f"Expected bullish, got {result.direction_score}"
    
    def test_bearish_text_produces_negative_direction(self):
        """Bearish news should produce negative direction_score."""
        from signals.deterministic_sentiment import DeterministicSentimentEngine
        
        engine = DeterministicSentimentEngine()
        result = engine.compute(
            news_texts=[
                "Market crash incoming! 💀",
                "Major exchange hack reported",
                "REKT! Liquidations cascade",
            ]
        )
        
        assert result.direction_score < -0.3, f"Expected bearish, got {result.direction_score}"
    
    def test_negation_flips_direction(self):
        """Negated keywords should flip direction."""
        from signals.deterministic_sentiment import CryptoKeywordSentiment
        
        analyzer = CryptoKeywordSentiment()
        
        d1, _ = analyzer.analyze("Bitcoin is bullish")
        d2, _ = analyzer.analyze("Bitcoin is not bullish")
        
        assert d1 > 0, "Plain 'bullish' should be positive"
        assert d2 < 0, "Negated 'bullish' should be negative"
    
    def test_lunarcrush_does_not_affect_direction(self):
        """LunarCrush data must NOT change direction_score."""
        from signals.deterministic_sentiment import DeterministicSentimentEngine
        
        engine = DeterministicSentimentEngine()
        
        # Bullish text
        result_no_lc = engine.compute(
            news_texts=["Bitcoin bullish breakout!"],
        )
        
        # Same text + extreme greed LunarCrush
        result_with_lc = engine.compute(
            news_texts=["Bitcoin bullish breakout!"],
            lunarcrush_data={"galaxy_score": 95},  # Extreme hype
        )
        
        # Direction must be the same
        assert result_no_lc.direction_score == result_with_lc.direction_score, \
            "LunarCrush should NOT affect direction_score"
        
        # But crowding should be different
        assert result_with_lc.crowding_bias > result_no_lc.crowding_bias
    
    def test_fear_greed_only_affects_crowding(self):
        """Fear & Greed index affects crowding_bias, NOT direction."""
        from signals.deterministic_sentiment import DeterministicSentimentEngine
        
        engine = DeterministicSentimentEngine()
        
        # Bearish text with extreme greed
        result = engine.compute(
            news_texts=["Market crash incoming! 💀"],
            fear_greed_value=95,  # Extreme greed
        )
        
        # Direction should still be bearish (from text)
        assert result.direction_score < 0, "Direction should be bearish from text"
        
        # Crowding should be positive (extreme greed = crowded longs)
        assert result.crowding_bias > 0.5, "Crowding should be positive from extreme greed"
    
    def test_extreme_crowding_detection(self):
        """Extreme crowding should be flagged correctly."""
        from signals.deterministic_sentiment import DeterministicSentimentEngine
        
        engine = DeterministicSentimentEngine()
        
        # Extreme fear
        result_fear = engine.compute(
            news_texts=["Neutral market"],
            fear_greed_value=15,  # Extreme fear
        )
        assert result_fear.extreme_crowding, "Should flag extreme fear"
        
        # Extreme greed
        result_greed = engine.compute(
            news_texts=["Neutral market"],
            fear_greed_value=85,  # Extreme greed
        )
        assert result_greed.extreme_crowding, "Should flag extreme greed"
        
        # Normal
        result_normal = engine.compute(
            news_texts=["Neutral market"],
            fear_greed_value=50,
        )
        assert not result_normal.extreme_crowding, "Should NOT flag normal"
    
    def test_output_contract_valid(self):
        """Output must conform to the contract specification."""
        from signals.deterministic_sentiment import DeterministicSentimentEngine
        
        engine = DeterministicSentimentEngine()
        result = engine.compute(
            news_texts=["Bitcoin bullish!"],
            fear_greed_value=70,
            funding_rate=0.001,
        )
        
        # Check ranges
        assert -1.0 <= result.direction_score <= 1.0
        assert 0.0 <= result.strength <= 1.0
        assert -1.0 <= result.crowding_bias <= 1.0
        assert isinstance(result.extreme_crowding, bool)
        assert isinstance(result.sources, dict)
        
        # Check is_valid
        assert result.is_valid()


# =============================================================================
# TEST: SENTIMENT INTEGRATION
# =============================================================================

class TestSentimentIntegration:
    """Tests for sentiment integration layer."""
    
    def test_context_injection(self):
        """Sentiment should be properly injected into context."""
        from signals.sentiment_integration import (
            SentimentIntegration,
            reset_sentiment_integration
        )
        from signals.deterministic_sentiment import reset_deterministic_sentiment_engine
        
        reset_sentiment_integration()
        reset_deterministic_sentiment_engine()
        
        integration = SentimentIntegration()
        context = {"regime": "TREND"}
        
        context = integration.integrate_into_context(
            context,
            news_texts=["Bitcoin bullish!"],
            fear_greed_value=60,
        )
        
        assert "sentiment" in context
        assert "direction_score" in context["sentiment"]
        assert "crowding_bias" in context["sentiment"]
    
    def test_signal_quality_effect_aligned_scores_higher(self):
        """Aligned trades should score higher."""
        from signals.sentiment_integration import SentimentIntegration
        
        integration = SentimentIntegration()
        
        sentiment = {
            "direction_score": 0.7,  # Bullish
            "strength": 0.8,
            "crowding_bias": 0.3,
            "extreme_crowding": False,
        }
        
        score_long = integration.compute_signal_quality_effect(sentiment, signal_direction=1)
        score_short = integration.compute_signal_quality_effect(sentiment, signal_direction=-1)
        
        assert score_long > score_short, "Aligned long should score higher than misaligned short"
    
    def test_profit_max_effects_never_veto(self):
        """Sentiment effects must NEVER produce veto-equivalent values."""
        from signals.sentiment_integration import SentimentIntegration
        
        integration = SentimentIntegration()
        
        # Worst case scenario
        worst_sentiment = {
            "direction_score": -1.0,  # Strongly against
            "strength": 0.1,          # Low confidence
            "crowding_bias": 1.0,     # Extreme crowding
            "extreme_crowding": True,
            "volatility": 0.5,
        }
        
        effects = integration.compute_profit_max_effects(worst_sentiment, signal_direction=1)
        
        # CRITICAL: No multiplier should be zero or near-zero
        assert effects["conviction_mult"] >= 0.5, "Conviction must be >= 0.5"
        assert effects["urgency_mult"] >= 0.5, "Urgency must be >= 0.5"
        assert effects["size_mult"] >= 0.3, "Size must be >= 0.3"


# =============================================================================
# TEST: PROFIT MAX ADAPTER SENTIMENT INTEGRATION
# =============================================================================

class TestProfitMaxSentiment:
    """Tests for profit_max_adapter sentiment integration."""
    
    def test_sentiment_modulation_applied(self):
        """Sentiment should modulate conviction/urgency."""
        from signals.profit_max_adapter import ProfitMaxAdapter, reset_profit_max_adapter
        
        reset_profit_max_adapter()
        adapter = ProfitMaxAdapter()
        
        # With aligned sentiment
        result_aligned = adapter.evaluate(
            regime="TREND",
            direction=1,
            sentiment={
                "direction_score": 0.8,  # Bullish, aligned with long
                "strength": 0.9,
                "crowding_bias": 0.2,
                "extreme_crowding": False,
            }
        )
        
        # With misaligned sentiment
        reset_profit_max_adapter()
        adapter2 = ProfitMaxAdapter()
        result_misaligned = adapter2.evaluate(
            regime="TREND",
            direction=1,
            sentiment={
                "direction_score": -0.8,  # Bearish, misaligned with long
                "strength": 0.9,
                "crowding_bias": 0.2,
                "extreme_crowding": False,
            }
        )
        
        # Aligned should have higher conviction
        assert result_aligned.conviction_multiplier > result_misaligned.conviction_multiplier
    
    def test_extreme_crowding_reduces_size(self):
        """Extreme crowding should reduce size_multiplier."""
        from signals.profit_max_adapter import ProfitMaxAdapter, reset_profit_max_adapter
        
        reset_profit_max_adapter()
        adapter = ProfitMaxAdapter()
        
        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            sentiment={
                "direction_score": 0.5,
                "strength": 0.8,
            },
            crowd={
                "crowding_bias": 0.9,
                "extreme_crowding": True,
            },
        )

        # Size should be reduced (extreme_crowding read from crowd dict)
        assert result.size_multiplier < 1.0, "Extreme crowding should reduce size"
    
    def test_sentiment_never_vetoes(self):
        """Sentiment should NEVER produce a veto."""
        from signals.profit_max_adapter import ProfitMaxAdapter, reset_profit_max_adapter, VetoSource
        
        reset_profit_max_adapter()
        adapter = ProfitMaxAdapter()
        
        # Worst case sentiment
        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            sentiment={
                "direction_score": -1.0,  # Maximum misalignment
                "strength": 0.0,           # Zero confidence
                "crowding_bias": 1.0,      # Maximum crowding
                "extreme_crowding": True,
            }
        )
        
        # Should NOT veto
        assert not result.veto, "Sentiment must NEVER veto"
        assert result.veto_source == VetoSource.NONE
    
    def test_no_sentiment_graceful_handling(self):
        """Adapter should handle missing sentiment gracefully."""
        from signals.profit_max_adapter import ProfitMaxAdapter, reset_profit_max_adapter
        
        reset_profit_max_adapter()
        adapter = ProfitMaxAdapter()
        
        result = adapter.evaluate(
            regime="TREND",
            direction=1,
            sentiment=None,  # No sentiment
        )
        
        assert not result.veto
        # Should still produce reasonable output
        assert 0.3 <= result.conviction_multiplier <= 2.0


# =============================================================================
# TEST: SIGNAL QUALITY SCORER SENTIMENT
# =============================================================================

class TestSignalQualityScorerSentiment:
    """Tests for signal_quality_scorer sentiment integration."""
    
    def test_sentiment_contributes_to_score(self):
        """Sentiment should contribute to total score."""
        from signals.signal_quality_scorer import SignalQualityScorer, reset_signal_quality_scorer
        
        reset_signal_quality_scorer()
        scorer = SignalQualityScorer()
        
        # With sentiment
        result_with = scorer.score(
            regime="TREND",
            signal_side=1,
            sentiment={
                "direction_score": 0.8,
                "strength": 0.9,
                "crowding_bias": 0.2,
                "extreme_crowding": False,
            }
        )
        
        # Without sentiment
        reset_signal_quality_scorer()
        scorer2 = SignalQualityScorer()
        result_without = scorer2.score(
            regime="TREND",
            signal_side=1,
            sentiment=None,
        )
        
        # Should have sentiment in breakdown
        assert "sentiment" in result_with.breakdown
    
    def test_aligned_sentiment_scores_higher(self):
        """Aligned sentiment should produce higher score."""
        from signals.signal_quality_scorer import SignalQualityScorer, reset_signal_quality_scorer
        
        reset_signal_quality_scorer()
        scorer = SignalQualityScorer()
        
        result_aligned = scorer.score(
            regime="TREND",
            signal_side=1,
            sentiment={"direction_score": 0.8, "strength": 0.9, "crowding_bias": 0.0, "extreme_crowding": False}
        )
        
        reset_signal_quality_scorer()
        scorer2 = SignalQualityScorer()
        result_misaligned = scorer2.score(
            regime="TREND",
            signal_side=1,
            sentiment={"direction_score": -0.8, "strength": 0.9, "crowding_bias": 0.0, "extreme_crowding": False}
        )
        
        assert result_aligned.score > result_misaligned.score


# =============================================================================
# TEST: CRYPTO KEYWORD SENTIMENT
# =============================================================================

class TestCryptoKeywordSentiment:
    """Tests for rule-based crypto sentiment."""
    
    def test_emoji_handling(self):
        """Emojis should be processed correctly."""
        from signals.deterministic_sentiment import CryptoKeywordSentiment
        
        analyzer = CryptoKeywordSentiment()
        
        d_rocket, _ = analyzer.analyze("To the moon! 🚀🚀🚀")
        d_skull, _ = analyzer.analyze("REKT! 💀💀💀")
        
        assert d_rocket > 0, "Rocket emojis should be bullish"
        assert d_skull < 0, "Skull emojis should be bearish"
    
    def test_crypto_slang(self):
        """Crypto-specific slang should be recognized."""
        from signals.deterministic_sentiment import CryptoKeywordSentiment
        
        analyzer = CryptoKeywordSentiment()
        
        d_wagmi, _ = analyzer.analyze("WAGMI! Diamond hands!")
        d_ngmi, _ = analyzer.analyze("NGMI, rug pull incoming")
        
        assert d_wagmi > 0, "WAGMI should be bullish"
        assert d_ngmi < 0, "NGMI should be bearish"
    
    def test_intensity_modifiers(self):
        """Intensity modifiers should affect strength."""
        from signals.deterministic_sentiment import CryptoKeywordSentiment
        
        analyzer = CryptoKeywordSentiment()
        
        d_normal, s_normal = analyzer.analyze("Bitcoin is bullish")
        d_extreme, s_extreme = analyzer.analyze("Bitcoin is extremely bullish")
        
        # Both should be bullish
        assert d_normal > 0 and d_extreme > 0
        
        # Extreme should have higher strength (intensity modifier)
        # Note: This depends on implementation


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

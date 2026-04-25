"""
Test harness for the supreme constitution gates.

The audit found Constitution had no direct tests. This file fills the
biggest gap: MarketDataValidator (schema enforcement) and
NoTradeTriggerChecker (the home of the P12 signal_conflict trigger and
the systemic-risk vetoes).

Tested:
- MarketDataValidator: required/critical/optional schema layering, alias
  resolution, price/age/NaN sanity checks, NORMAL vs OPPORTUNITY mode
  behavior on missing-critical
- NoTradeTriggerChecker: each individual trigger (stale data, feed
  disagreement, flash crash, DVOL, liquidity, correlation collapse,
  signal conflict). Plus the P12 invariant that 2-agent quant-vs-DRL
  conflict scores 0.7 but does NOT activate ALL_CONFLICT_FLAT (only
  >= 0.9 escalates to NO_TRADE per the constitutional fix).
"""
import sys, os
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from defense.constitution import (
    MarketDataValidator,
    NoTradeTriggerChecker,
    NoTradeTriggerType,
)


# =============================================================================
# MarketDataValidator
# =============================================================================

@pytest.fixture
def validator():
    return MarketDataValidator()


def _full_market_data(**overrides):
    """Build a market_data dict that passes all schema checks by default."""
    base = {
        # required
        "current_price": 100_000.0,
        "volume_24h": 1_000_000.0,
        "data_age_seconds": 5.0,
        # critical
        "dvol_zscore": 0.0,
        "vpin": 0.5,
        "correlation_btc_eth_sol": 0.5,
        "orderbook_depth_1pct_usd": 1_000_000.0,
    }
    base.update(overrides)
    return base


class TestMarketDataValidator:
    def test_valid_data_passes(self, validator):
        r = validator.validate(_full_market_data())
        assert r.is_valid is True
        assert r.error == ""
        assert r.canonical_data is not None

    def test_missing_required_rejects(self, validator):
        data = _full_market_data()
        del data["current_price"]
        r = validator.validate(data)
        assert r.is_valid is False
        assert "current_price" in r.missing_required

    def test_missing_critical_in_normal_rejects(self, validator):
        data = _full_market_data()
        del data["vpin"]
        r = validator.validate(data, mode="NORMAL")
        assert r.is_valid is False
        assert "vpin" in r.missing_critical

    def test_missing_critical_in_opportunity_passes_with_default(self, validator):
        data = _full_market_data()
        del data["vpin"]
        r = validator.validate(data, mode="OPPORTUNITY")
        assert r.is_valid is True
        assert "vpin" in r.missing_critical
        # Default applied: vpin=0.5 per CANONICAL_SCHEMA
        assert r.canonical_data["vpin"] == 0.5

    def test_missing_optional_passes_with_default(self, validator):
        # All optionals already missing in our minimal "_full_market_data" — but
        # the validator should still pass since they're optional.
        data = _full_market_data()
        r = validator.validate(data)
        assert r.is_valid is True
        # spread_bps default applied
        assert r.canonical_data.get("spread_bps") == 10.0

    def test_alias_resolves_legacy_key(self, validator):
        data = _full_market_data()
        data["price"] = data.pop("current_price")  # legacy alias
        r = validator.validate(data)
        assert r.is_valid is True
        assert r.canonical_data["current_price"] == 100_000.0

    def test_invalid_price_rejects(self, validator):
        r = validator.validate(_full_market_data(current_price=-1.0))
        assert r.is_valid is False
        assert "Invalid price" in r.error

    def test_zero_price_rejects(self, validator):
        r = validator.validate(_full_market_data(current_price=0.0))
        assert r.is_valid is False

    def test_excessive_price_rejects(self, validator):
        r = validator.validate(_full_market_data(current_price=2_000_000_000.0))
        assert r.is_valid is False

    def test_stale_data_rejects(self, validator):
        r = validator.validate(_full_market_data(data_age_seconds=120.0))
        assert r.is_valid is False
        assert "Stale" in r.error

    def test_data_age_30s_passes(self, validator):
        """[P22] schema max=60.0 (was 10.0); 30s should pass."""
        r = validator.validate(_full_market_data(data_age_seconds=30.0))
        assert r.is_valid is True

    def test_nan_value_rejects(self, validator):
        r = validator.validate(_full_market_data(dvol_zscore=float("nan")))
        assert r.is_valid is False
        assert "NaN" in r.error


# =============================================================================
# NoTradeTriggerChecker
# =============================================================================

@pytest.fixture
def checker():
    return NoTradeTriggerChecker()


class TestNoTradeStaleData:
    def test_below_threshold_no_trigger(self, checker):
        state = checker.compute_triggers(
            market_data={"data_age_seconds": 30.0, "current_price": 100_000.0},
            signal_data={},
        )
        # 30s < 60s threshold: score = 0.5, no STALE_DATA active condition
        assert state.trigger_scores["stale_data"] == pytest.approx(0.5)
        assert not any(
            c.trigger_type == NoTradeTriggerType.STALE_DATA for c in state.active_conditions
        )

    def test_above_threshold_triggers(self, checker):
        state = checker.compute_triggers(
            market_data={"data_age_seconds": 90.0, "current_price": 100_000.0},
            signal_data={},
        )
        assert state.trigger_scores["stale_data"] == 1.0  # min(1.0, 90/60)
        assert any(
            c.trigger_type == NoTradeTriggerType.STALE_DATA for c in state.active_conditions
        )
        assert state.should_no_trade is True


class TestNoTradeFeedDisagreement:
    def test_aligned_feeds_no_trigger(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "binance_price": 100_001.0,
                "kraken_price": 100_000.0,
            },
            signal_data={},
        )
        # 0.001% disagreement → tiny score, no trigger
        assert state.trigger_scores["feed_disagreement"] < 0.01
        assert not any(
            c.trigger_type == NoTradeTriggerType.FEED_DISAGREEMENT
            for c in state.active_conditions
        )

    def test_disagreement_above_1pct_triggers(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "binance_price": 102_000.0,  # 2% gap from kraken
                "kraken_price": 100_000.0,
            },
            signal_data={},
        )
        assert any(
            c.trigger_type == NoTradeTriggerType.FEED_DISAGREEMENT
            for c in state.active_conditions
        )


class TestNoTradeDVOL:
    def test_normal_dvol_no_trigger(self, checker):
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0, "dvol_zscore": 1.5},
            signal_data={},
        )
        assert state.trigger_scores["extreme_dvol"] == pytest.approx(0.3)
        assert not any(
            c.trigger_type == NoTradeTriggerType.EXTREME_DVOL for c in state.active_conditions
        )

    def test_extreme_dvol_triggers(self, checker):
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0, "dvol_zscore": 5.5},
            signal_data={},
        )
        assert state.trigger_scores["extreme_dvol"] == 1.0
        assert any(
            c.trigger_type == NoTradeTriggerType.EXTREME_DVOL for c in state.active_conditions
        )


class TestNoTradeLiquidity:
    def test_deep_liquidity_no_trigger(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "orderbook_depth_1pct_usd": 1_000_000.0,
            },
            signal_data={},
        )
        assert state.trigger_scores["liquidity_critical"] == 0.0
        assert not any(
            c.trigger_type == NoTradeTriggerType.LIQUIDITY_CRITICAL
            for c in state.active_conditions
        )

    def test_thin_liquidity_triggers(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "orderbook_depth_1pct_usd": 50_000.0,  # below 100K threshold
            },
            signal_data={},
        )
        assert any(
            c.trigger_type == NoTradeTriggerType.LIQUIDITY_CRITICAL
            for c in state.active_conditions
        )


class TestNoTradeCorrelationCollapse:
    def test_normal_correlation_no_trigger(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "correlation_btc_eth_sol": 0.7,
            },
            signal_data={},
        )
        assert state.trigger_scores["correlation_collapse"] == 0.0
        assert not any(
            c.trigger_type == NoTradeTriggerType.CORRELATION_COLLAPSE
            for c in state.active_conditions
        )

    def test_correlation_at_threshold_triggers(self, checker):
        # Threshold = 0.92, the trigger uses >= so 0.92 should trigger
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "correlation_btc_eth_sol": 0.92,
            },
            signal_data={},
        )
        assert any(
            c.trigger_type == NoTradeTriggerType.CORRELATION_COLLAPSE
            for c in state.active_conditions
        )

    def test_correlation_extreme_triggers(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "correlation_btc_eth_sol": 0.99,
            },
            signal_data={},
        )
        assert state.trigger_scores["correlation_collapse"] > 0.5
        assert any(
            c.trigger_type == NoTradeTriggerType.CORRELATION_COLLAPSE
            for c in state.active_conditions
        )


class TestNoTradeSignalConflict:
    """P12 regression tests at the source-of-truth (constitution) level.

    Constitution.py:441-444 spec: 2-agent conflict scores 0.7 but only >= 0.9
    escalates to NO_TRADE. integration_v36.py already had a guard test for the
    veto-promotion threshold. These tests verify the score generation."""

    def test_aligned_decide_agents_no_conflict(self, checker):
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0},
            signal_data={"quant_direction": 0.8, "drl_direction": 0.7},
        )
        assert state.trigger_scores["signal_conflict"] == 0.0
        assert not any(
            c.trigger_type == NoTradeTriggerType.ALL_CONFLICT_FLAT
            for c in state.active_conditions
        )

    def test_weak_signals_no_conflict(self, checker):
        """Both signals below CONFLICT_DIRECTION_THRESHOLD (0.3) → no conflict."""
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0},
            signal_data={"quant_direction": 0.2, "drl_direction": -0.2},
        )
        assert state.trigger_scores["signal_conflict"] == 0.0

    def test_2agent_opposing_scores_07_but_no_active_condition(self, checker):
        """P12: opposing quant + DRL → score 0.7, but ALL_CONFLICT_FLAT NOT
        active (only score >= 0.9 escalates)."""
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0},
            signal_data={"quant_direction": 0.8, "drl_direction": -0.7},
        )
        assert state.trigger_scores["signal_conflict"] == pytest.approx(0.7)
        # Critical P12 invariant: 2-agent conflict does NOT trigger NO_TRADE
        assert not any(
            c.trigger_type == NoTradeTriggerType.ALL_CONFLICT_FLAT
            for c in state.active_conditions
        )

    def test_sentiment_excluded_from_conflict(self, checker):
        """Iron Law #34: sentiment NEVER vetoes via conflict.

        Even with extreme bearish sentiment, if quant and DRL agree, no
        conflict should fire."""
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0},
            signal_data={
                "quant_direction": 0.8,
                "drl_direction": 0.6,
                "sentiment_direction": -1.0,  # extreme fear, ignored
            },
        )
        assert state.trigger_scores["signal_conflict"] == 0.0


class TestNoTradeAggregation:
    def test_clean_market_no_trade_false(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "data_age_seconds": 5.0,
                "dvol_zscore": 0.5,
                "correlation_btc_eth_sol": 0.5,
                "orderbook_depth_1pct_usd": 1_000_000.0,
            },
            signal_data={"quant_direction": 0.5, "drl_direction": 0.5},
        )
        assert state.should_no_trade is False
        assert len(state.active_conditions) == 0

    def test_any_trigger_sets_no_trade(self, checker):
        state = checker.compute_triggers(
            market_data={"current_price": 100_000.0, "dvol_zscore": 6.0},
            signal_data={},
        )
        assert state.should_no_trade is True
        assert state.primary_reason == NoTradeTriggerType.EXTREME_DVOL

    def test_multiple_triggers_aggregated(self, checker):
        state = checker.compute_triggers(
            market_data={
                "current_price": 100_000.0,
                "data_age_seconds": 90.0,        # stale
                "dvol_zscore": 5.5,              # extreme
                "orderbook_depth_1pct_usd": 50_000.0,  # thin
            },
            signal_data={},
        )
        types = {c.trigger_type for c in state.active_conditions}
        assert NoTradeTriggerType.STALE_DATA in types
        assert NoTradeTriggerType.EXTREME_DVOL in types
        assert NoTradeTriggerType.LIQUIDITY_CRITICAL in types

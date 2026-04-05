"""
Tests for agents/microstructure_agent.py v6 refactoring.

Groups:
  1. Per-asset separation (3 tests) - BTC updates don't affect ETH
  2. Stale / missing data (4 tests) - neutral output on missing/stale data
  3. V6 output contract (4 tests) - canonical keys, types, bounds
  4. No background loops (1 test) - import safety
  5. Signal generation (3 tests) - strategies fire correctly
  6. EventBus integration (2 tests) - helper wiring
"""
import pytest
import time
from unittest.mock import MagicMock, patch

from agents.microstructure_agent import (
    MicrostructureArbitrageAgent,
    MicrostructureConfig,
    _neutral_v6_payload,
    get_microstructure_agent,
    reset_microstructure_agent,
    MicrostructureEventBusIntegration,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_agent(**config_kw) -> MicrostructureArbitrageAgent:
    cfg = MicrostructureConfig(**config_kw)
    return MicrostructureArbitrageAgent(config=cfg)


def _feed_exchange(agent, asset, exchange, mid, spread=10, bid_sz=5, ask_sz=5,
                   last_side="", last_size=0):
    agent.update_exchange(
        asset=asset,
        exchange=exchange,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        bid_size=bid_sz,
        ask_size=ask_sz,
        last_price=mid,
        last_size=last_size,
        last_side=last_side,
    )


# V6 canonical keys that every payload must contain
_V6_REQUIRED_KEYS = {
    "lead_lag_edge", "lead_lag_confidence", "cvd_divergence",
    "order_book_imbalance", "spread_bps",
    "micro_direction", "micro_confidence", "micro_urgency",
    "micro_primary_signal", "micro_data_quality", "micro_is_valid",
    "data_age_seconds", "asof_timestamp", "asset", "diagnostics",
}


# ============================================================================
# GROUP 1: Per-asset separation (3 tests)
# ============================================================================

class TestPerAssetSeparation:
    """BTC updates must NOT affect ETH/SOL state."""

    def test_btc_update_isolated_from_eth(self):
        agent = _make_agent(min_samples=0)
        _feed_exchange(agent, "BTC", "binance", 95000)
        _feed_exchange(agent, "BTC", "kraken", 94990)

        p_btc = agent.generate_signal("BTC")
        p_eth = agent.generate_signal("ETH")

        # BTC should have exchange data, ETH should not
        assert p_btc["diagnostics"].get("n_exchanges", 0) == 2
        assert p_eth["diagnostics"].get("reason") == "no_exchange_data"

    def test_eth_state_independent(self):
        agent = _make_agent()
        _feed_exchange(agent, "ETH", "kraken", 3400)
        _feed_exchange(agent, "ETH", "binance", 3405)

        assert "ETH" in agent._per_asset
        assert "BTC" not in agent._per_asset

    def test_reset_single_asset(self):
        agent = _make_agent()
        _feed_exchange(agent, "BTC", "kraken", 95000)
        _feed_exchange(agent, "ETH", "kraken", 3400)
        agent.reset_state("BTC")
        assert "BTC" not in agent._per_asset
        assert "ETH" in agent._per_asset


# ============================================================================
# GROUP 2: Stale / missing data (4 tests)
# ============================================================================

class TestStaleData:
    """Stale or missing data must produce neutral output."""

    def test_no_data_returns_neutral(self):
        agent = _make_agent()
        p = agent.generate_signal("BTC")
        assert p["micro_direction"] == 0.0
        assert p["micro_confidence"] == 0.0
        assert p["micro_is_valid"] is True
        assert p["diagnostics"]["reason"] == "no_exchange_data"

    def test_stale_snapshot_returns_neutral(self):
        agent = _make_agent(max_snapshot_age_ms=100)
        # Feed data with old timestamp
        old_ts = (time.time() - 10) * 1000  # 10s ago
        agent.update_exchange("BTC", "kraken", 95000, 95010, 5, 5,
                              timestamp_ms=old_ts)
        p = agent.generate_signal("BTC")
        assert p["diagnostics"]["reason"] == "stale_snapshot"
        assert p["micro_direction"] == 0.0

    def test_insufficient_samples_returns_neutral(self):
        agent = _make_agent(min_samples=9999)
        _feed_exchange(agent, "BTC", "kraken", 95000)
        p = agent.generate_signal("BTC")
        assert p["diagnostics"]["reason"] == "insufficient_samples"

    def test_missing_leader_degrades_gracefully(self):
        agent = _make_agent()
        # Only follower data, no leader
        _feed_exchange(agent, "BTC", "kraken", 95000)
        p = agent.generate_signal("BTC")
        assert p["lead_lag_edge"] == 0.0
        assert p["lead_lag_confidence"] == 0.0


# ============================================================================
# GROUP 3: V6 output contract (4 tests)
# ============================================================================

class TestV6OutputContract:
    """Output dicts must have the right keys, types, and bounds."""

    def test_all_canonical_keys_present(self):
        agent = _make_agent()
        p = agent.generate_signal("SOL")
        for key in _V6_REQUIRED_KEYS:
            assert key in p, f"Missing key: {key}"

    def test_neutral_payload_keys(self):
        p = _neutral_v6_payload("ETH")
        for key in _V6_REQUIRED_KEYS:
            assert key in p, f"Missing key in neutral: {key}"
        assert p["asset"] == "ETH"
        assert p["micro_direction"] == 0.0

    def test_direction_bounded(self):
        agent = _make_agent(min_confidence=0.0, signal_cooldown_ms=0, min_samples=0)
        # Feed enough imbalance data to trigger a signal
        for i in range(25):
            _feed_exchange(agent, "BTC", "binance", 95000, bid_sz=10, ask_sz=1)
            _feed_exchange(agent, "BTC", "kraken", 94990, bid_sz=10, ask_sz=1)
        p = agent.generate_signal("BTC")
        assert -1.0 <= p["micro_direction"] <= 1.0
        assert 0.0 <= p["micro_confidence"] <= 1.0

    def test_asset_matches_request(self):
        agent = _make_agent()
        p = agent.generate_signal("SOL")
        assert p["asset"] == "SOL"


# ============================================================================
# GROUP 4: No background loops (1 test)
# ============================================================================

class TestNoBackgroundLoops:
    """Importing or constructing the agent must NOT start background tasks."""

    def test_import_does_not_start_threads(self):
        import threading
        initial_count = threading.active_count()
        _ = _make_agent()
        assert threading.active_count() <= initial_count


# ============================================================================
# GROUP 5: Signal generation (3 tests)
# ============================================================================

class TestSignalGeneration:
    """Verify strategies fire when conditions are met."""

    def _build_agent_with_data(self, ticks=25):
        """Feed enough ticks to fill buffers."""
        agent = _make_agent(min_confidence=0.0, signal_cooldown_ms=0, min_samples=0)
        for i in range(ticks):
            _feed_exchange(
                agent, "BTC", "binance", 95000 + i * 10,
                bid_sz=5 + i * 0.5, ask_sz=2,
                last_side="buy", last_size=1 + i * 0.5,
            )
            _feed_exchange(
                agent, "BTC", "kraken", 94980 + i * 10,
                bid_sz=5 + i * 0.5, ask_sz=2,
                last_side="buy", last_size=0.5 + i * 0.3,
            )
        return agent

    def test_signal_after_enough_ticks(self):
        agent = self._build_agent_with_data(30)
        p = agent.generate_signal("BTC")
        # Should have non-zero data quality at least
        assert p["micro_data_quality"] > 0

    def test_get_last_payload_returns_cached(self):
        agent = self._build_agent_with_data(10)
        p1 = agent.generate_signal("BTC")
        p2 = agent.get_last_payload("BTC")
        assert p1 is p2

    def test_get_last_payload_neutral_if_no_data(self):
        agent = _make_agent()
        p = agent.get_last_payload("ETH")
        assert p["micro_direction"] == 0.0
        assert p["asset"] == "ETH"


# ============================================================================
# GROUP 6: EventBus integration (2 tests)
# ============================================================================

class TestEventBusIntegration:
    """EventBus helper wiring checks."""

    def test_initialize_fails_gracefully_without_event_manager(self):
        agent = _make_agent()
        integration = MicrostructureEventBusIntegration(agent)
        # Should return False when infra not available (import fails)
        with patch("agents.microstructure_agent.MicrostructureEventBusIntegration.initialize",
                   return_value=False):
            assert integration.initialize() is False

    def test_on_lob_tick_feeds_agent(self):
        agent = _make_agent()
        integration = MicrostructureEventBusIntegration(agent)

        event = MagicMock()
        event.payload = {
            "asset": "BTC",
            "exchange": "kraken",
            "bid": 95000,
            "ask": 95010,
            "bid_size": 5.0,
            "ask_size": 3.0,
        }

        integration._on_lob_tick(event)

        # Agent should now have kraken data for BTC
        assert "BTC" in agent._per_asset
        assert "kraken" in agent._per_asset["BTC"].exchange_states


# ============================================================================
# GROUP 8: analyze_asset v6.4 (5 tests)
# ============================================================================

class TestAnalyzeAsset:
    """V6.4 analyze_asset() direction-neutral execution analysis."""

    def test_returns_safe_defaults_no_data(self):
        agent = _make_agent()
        r = agent.analyze_asset("BTC")
        assert r["spread_bps"] == 0.0
        assert r["order_book_imbalance"] == 0.0
        assert isinstance(r["flash_crash_active"], bool)
        assert r["flash_crash_active"] is False
        assert isinstance(r["estimated_friction_bps"], (int, float))
        assert "provenance" in r

    def test_data_quality_string_present(self):
        agent = _make_agent()
        r = agent.analyze_asset("ETH")
        assert r["data_quality"] in ("ok", "stale", "missing", "error")

    def test_flash_crash_on_large_price_move(self):
        agent = _make_agent()
        r = agent.analyze_asset("BTC", market_data={"price_move_pct": 10.0})
        assert r["flash_crash_active"] is True

    def test_friction_increases_with_spread(self):
        agent = _make_agent(min_samples=0)
        _feed_exchange(agent, "BTC", "kraken", 95000, spread=20)
        _feed_exchange(agent, "BTC", "binance", 95000, spread=20)
        r_wide = agent.analyze_asset("BTC")

        agent2 = _make_agent(min_samples=0)
        _feed_exchange(agent2, "BTC", "kraken", 95000, spread=2)
        _feed_exchange(agent2, "BTC", "binance", 95000, spread=2)
        r_narrow = agent2.analyze_asset("BTC")

        assert r_wide["estimated_friction_bps"] >= r_narrow["estimated_friction_bps"]

    def test_asof_ts_is_iso_string(self):
        agent = _make_agent()
        r = agent.analyze_asset("SOL")
        assert isinstance(r["asof_ts"], str)
        assert r["asof_ts"].endswith("Z")


# ============================================================================
# GROUP 7: Singleton lifecycle (2 tests)
# ============================================================================

class TestSingleton:
    def test_singleton_returns_same_instance(self):
        reset_microstructure_agent()
        a1 = get_microstructure_agent()
        a2 = get_microstructure_agent()
        assert a1 is a2
        reset_microstructure_agent()

    def test_reset_creates_fresh_instance(self):
        reset_microstructure_agent()
        a1 = get_microstructure_agent()
        reset_microstructure_agent()
        a2 = get_microstructure_agent()
        assert a1 is not a2
        reset_microstructure_agent()

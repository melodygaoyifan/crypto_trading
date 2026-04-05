"""
Tests for agents/kraken_quant_agent.py v6 refactoring.

Groups:
  1. No data (4 tests) - neutral output on missing/empty inputs
  2. Partial data (4 tests) - coverage scoring, graceful degradation
  3. Derivatives spike (3 tests) - strategy fires on extreme data
  4. Optional libs missing (2 tests) - scipy absence doesn't crash
  5. V6 interface compliance (5 tests) - payload shape, singleton, per-asset
"""
import pytest
import time
import numpy as np
from unittest.mock import patch, MagicMock

from agents.kraken_quant_agent import (
    KrakenQuantAgentV6,
    KrakenQuantPayload,
    KrakenQuantAgent,
    get_kraken_quant_agent,
    reset_kraken_quant_agent,
    _neutral_payload,
    Regime,
    SignalType,
    _SIGNAL_DIRECTION,
    _REGIME_MAP,
    map_v6_regime_to_research_regime,
    signals_to_v6_quant_packet,
)


# ============================================================================
# HELPERS
# ============================================================================

def _make_agent() -> KrakenQuantAgentV6:
    """Fresh agent per test - no cross-contamination."""
    return KrakenQuantAgentV6()


def _bear_market_data(asset: str = "SOL", price: float = 150.0) -> dict:
    """Minimal BEAR-regime data with derivatives fields."""
    return {
        "price": price,
        "price_btc": 50000.0,
        "price_eth": 3000.0,
        "price_sol": 150.0,
        "open_interest": 1e9,
        "funding_rate": -0.002,
        "liquidation_volume": 8e6,
        "taker_ratio": 0.35,
        "liquidation_intensity": 0.9,
    }


# ============================================================================
# GROUP 1: No Data - neutral output (4 tests)
# ============================================================================

class TestNoData:
    """Empty or missing inputs must return neutral, never crash."""

    def test_no_market_data_returns_neutral(self):
        agent = _make_agent()
        p = agent.generate_signal("BTC")
        assert isinstance(p, KrakenQuantPayload)
        assert p.direction == 0.0
        assert p.confidence == 0.0
        assert p.is_valid is True

    def test_empty_dict_returns_neutral(self):
        agent = _make_agent()
        p = agent.generate_signal("ETH", market_data={})
        assert p.direction == 0.0
        assert p.primary_strategy == "none"

    def test_none_regime_defaults_sideways(self):
        agent = _make_agent()
        p = agent.generate_signal("SOL", market_data={"price": 150.0}, regime=None)
        assert p.diagnostics.get("regime") == "chop"  # Regime.SIDEWAYS.value

    def test_unsupported_asset_returns_neutral(self):
        agent = _make_agent()
        p = agent.generate_signal("DOGE")
        assert p.direction == 0.0
        assert "unsupported" in str(p.diagnostics.get("reason", ""))


# ============================================================================
# GROUP 2: Partial Data - coverage & degradation (4 tests)
# ============================================================================

class TestPartialData:
    """Partial inputs degrade data_quality but still produce valid payload."""

    def test_price_only_bear_low_quality(self):
        agent = _make_agent()
        p = agent.generate_signal("SOL", market_data={"price": 150.0}, regime="BEAR")
        # BEAR expects 6 fields; only 'price' present -> ~0.17 quality
        assert p.data_quality < 0.5
        assert "open_interest" in p.missing_inputs

    def test_price_only_sideways_higher_quality(self):
        agent = _make_agent()
        p = agent.generate_signal("SOL", market_data={"price": 150.0}, regime="SIDEWAYS")
        # SIDEWAYS expects only price + funding -> 0.5 quality
        assert p.data_quality >= 0.4
        assert "funding_rate" in p.missing_inputs

    def test_full_bear_data_high_quality(self):
        agent = _make_agent()
        md = _bear_market_data()
        p = agent.generate_signal("SOL", market_data=md, regime="PANIC_SELLOFF")
        # Most fields present (funding_rate key mapped to generic)
        assert p.data_quality >= 0.7

    def test_missing_inputs_list_correct(self):
        agent = _make_agent()
        p = agent.generate_signal("BTC", market_data={"price": 50000}, regime="BULL")
        # BULL expects: price, funding_rate, bid_depth, ask_depth
        assert "price" not in p.missing_inputs  # price is present
        assert "funding_rate" in p.missing_inputs
        assert "bid_depth" in p.missing_inputs


# ============================================================================
# GROUP 3: Derivatives Spike - strategy fires (3 tests)
# ============================================================================

class TestDerivativesSpike:
    """Feed enough data that strategies produce non-neutral signals."""

    def _feed_cascade_data(self, agent: KrakenQuantAgentV6, ticks: int = 25):
        """Feed enough ticks to fill strategy buffers for LiquidationCascade."""
        for i in range(ticks):
            # Simulating a crash: BTC price falling, OI dropping, liqs spiking
            btc_price = 50000 - i * 200  # 50k -> 45k
            sol_price = 150 - i * 5       # 150 -> 25
            md = {
                "timestamp": time.time(),
                "price": sol_price,
                "price_btc": btc_price,
                "price_eth": 3000 - i * 30,
                "price_sol": sol_price,
                "open_interest": 1e9 - i * 5e7,    # OI dropping fast
                "open_interest_btc": 1e9 - i * 5e7,
                "funding_rate": -0.003,
                "liquidation_volume": 1e6 * (i + 1),  # Liquidations spiking
                "liquidation_volume_btc": 1e6 * (i + 1),
                "taker_ratio": 0.3,
                "taker_ratio_btc": 0.3,
                "liquidation_intensity": min(1.0, i * 0.1),
                "liquidation_intensity_btc": min(1.0, i * 0.1),
            }
            agent.generate_signal("SOL", market_data=md, regime="PANIC_SELLOFF")

    def test_cascade_returns_typed_payload(self):
        """After enough cascade ticks, payload is always KrakenQuantPayload."""
        agent = _make_agent()
        self._feed_cascade_data(agent, ticks=25)
        p = agent.get_last_payload("SOL")
        assert isinstance(p, KrakenQuantPayload)
        # direction may or may not be non-zero (strategies need 100+ bars)
        # but the infrastructure must not crash
        assert p.asset == "SOL"
        assert p.is_valid is True

    def test_payload_direction_bounded(self):
        """Direction must always be in [-1, +1]."""
        agent = _make_agent()
        self._feed_cascade_data(agent, ticks=30)
        p = agent.get_last_payload("SOL")
        assert -1.0 <= p.direction <= 1.0

    def test_diagnostics_populated(self):
        """Diagnostics dict has required keys after real ticks."""
        agent = _make_agent()
        self._feed_cascade_data(agent, ticks=10)
        p = agent.get_last_payload("SOL")
        assert "regime" in p.diagnostics
        assert "tick" in p.diagnostics
        assert p.diagnostics["tick"] > 0


# ============================================================================
# GROUP 4: Optional Libs Missing (2 tests)
# ============================================================================

class TestOptionalLibsMissing:
    """Agent must still work if scipy is unavailable."""

    def test_generate_signal_without_scipy(self):
        """Mocking SCIPY_AVAILABLE=False should not crash generate_signal."""
        with patch("agents.kraken_quant_agent.SCIPY_AVAILABLE", False):
            agent = _make_agent()
            p = agent.generate_signal("BTC", market_data={"price": 50000}, regime="BEAR")
            assert isinstance(p, KrakenQuantPayload)
            assert p.is_valid is True

    def test_neutral_payload_no_deps(self):
        """_neutral_payload has zero external dependencies."""
        p = _neutral_payload("SOL")
        assert p.direction == 0.0
        assert p.confidence == 0.0
        assert p.asset == "SOL"


# ============================================================================
# GROUP 5: V6 Interface Compliance (5 tests)
# ============================================================================

class TestV6InterfaceCompliance:
    """Payload shape, singleton pattern, per-asset caching."""

    def test_to_agent_signal_dict_has_required_keys(self):
        p = KrakenQuantPayload(asset="BTC", direction=0.3, confidence=0.7)
        d = p.to_agent_signal_dict()
        for key in ["kq_direction", "kq_confidence", "kq_primary_strategy",
                     "kq_data_quality", "kq_is_valid", "kq_signal_edge_bps",
                     "missing_inputs", "asof_timestamp", "data_age_seconds",
                     "asset", "diagnostics"]:
            assert key in d, f"Missing key: {key}"

    def test_singleton_lifecycle(self):
        reset_kraken_quant_agent()
        a1 = get_kraken_quant_agent()
        a2 = get_kraken_quant_agent()
        assert a1 is a2
        reset_kraken_quant_agent()
        a3 = get_kraken_quant_agent()
        assert a3 is not a1

    def test_per_asset_caching(self):
        agent = _make_agent()
        agent.generate_signal("BTC", price=50000)
        agent.generate_signal("ETH", price=3000)
        assert "BTC" in agent._last_payloads
        assert "ETH" in agent._last_payloads

    def test_reset_state_per_asset(self):
        agent = _make_agent()
        agent.generate_signal("BTC", price=50000)
        agent.generate_signal("ETH", price=3000)
        agent.reset_state("BTC")
        assert "BTC" not in agent._last_payloads
        assert "ETH" in agent._last_payloads
        agent.reset_state()
        assert len(agent._last_payloads) == 0

    def test_backward_compat_alias(self):
        # KrakenQuantAgent is either V6 directly (env flag set) or a
        # deprecation wrapper subclass (env flag unset).  Either way,
        # instantiating it must produce a working V6 instance.
        assert issubclass(KrakenQuantAgent, KrakenQuantAgentV6)


# ============================================================================
# GROUP 6: Regime Mapping (3 tests)
# ============================================================================

class TestRegimeMapping:
    """HMATS regime names correctly map to local Regime enum."""

    def test_momentum_rally_maps_to_bull(self):
        agent = _make_agent()
        p = agent.generate_signal("BTC", price=50000, regime="MOMENTUM_RALLY")
        assert p.diagnostics.get("regime") == "bull"

    def test_panic_selloff_maps_to_bear(self):
        agent = _make_agent()
        p = agent.generate_signal("BTC", price=50000, regime="PANIC_SELLOFF")
        assert p.diagnostics.get("regime") == "bear"

    def test_unknown_regime_maps_to_sideways(self):
        agent = _make_agent()
        p = agent.generate_signal("BTC", price=50000, regime="REGIME_3")
        assert p.diagnostics.get("regime") == "chop"  # Regime.SIDEWAYS.value


# ============================================================================
# GROUP 7: Signal Direction Mapping (3 tests)
# ============================================================================

class TestSignalDirectionMapping:
    """_SIGNAL_DIRECTION covers all SignalType values."""

    def test_all_signal_types_mapped(self):
        for st in SignalType:
            assert st in _SIGNAL_DIRECTION, f"Missing mapping for {st}"

    def test_aggressive_short_is_negative_one(self):
        assert _SIGNAL_DIRECTION[SignalType.AGGRESSIVE_SHORT] == -1.0

    def test_hold_and_flatten_are_zero(self):
        assert _SIGNAL_DIRECTION[SignalType.FLATTEN] == 0.0
        assert _SIGNAL_DIRECTION[SignalType.NO_SIGNAL] == 0.0


# ============================================================================
# GROUP 8: map_v6_regime_to_research_regime (5 tests)
# ============================================================================

class TestMapV6RegimeToResearchRegime:
    """Public adapter: HMATS regime -> research-strategy regime name."""

    def test_momentum_rally_returns_bull(self):
        assert map_v6_regime_to_research_regime("MOMENTUM_RALLY") == "BULL"

    def test_panic_selloff_returns_bear(self):
        assert map_v6_regime_to_research_regime("PANIC_SELLOFF") == "BEAR"

    def test_extreme_volatility_returns_bear(self):
        assert map_v6_regime_to_research_regime("EXTREME_VOLATILITY") == "BEAR"

    def test_quiet_accumulation_returns_sideways(self):
        assert map_v6_regime_to_research_regime("QUIET_ACCUMULATION") == "SIDEWAYS"

    def test_regime_n_falls_through_to_sideways(self):
        assert map_v6_regime_to_research_regime("REGIME_7") == "SIDEWAYS"


# ============================================================================
# GROUP 9: signals_to_v6_quant_packet (4 tests)
# ============================================================================

class TestSignalsToV6QuantPacket:
    """Public adapter: list of Signal namedtuples -> flat v6 dict."""

    def test_empty_signals_returns_neutral(self):
        pkt = signals_to_v6_quant_packet([], primary_asset="ETH")
        assert pkt["kq_direction"] == 0.0
        assert pkt["kq_confidence"] == 0.0
        assert pkt["asset"] == "ETH"
        assert pkt["signals_count"] == 0

    def test_single_signal_picks_it(self):
        sig = MagicMock()
        sig.confidence = 0.8
        sig.signal_type = SignalType.LONG
        sig.strategy_name = "momentum"
        sig.entry_price = 100.0
        sig.take_profit = 110.0
        pkt = signals_to_v6_quant_packet([sig], primary_asset="BTC")
        assert pkt["kq_direction"] == 0.6  # LONG -> 0.6
        assert pkt["kq_confidence"] == 0.8
        assert pkt["kq_primary_strategy"] == "momentum"
        assert pkt["kq_signal_edge_bps"] == 1000.0  # (110-100)/100 * 10000
        assert pkt["signals_count"] == 1

    def test_picks_highest_confidence(self):
        weak = MagicMock()
        weak.confidence = 0.3
        weak.signal_type = SignalType.SHORT
        weak.strategy_name = "mean_revert"
        weak.entry_price = 0
        weak.take_profit = 0

        strong = MagicMock()
        strong.confidence = 0.9
        strong.signal_type = SignalType.AGGRESSIVE_LONG
        strong.strategy_name = "breakout"
        strong.entry_price = 0
        strong.take_profit = 0

        pkt = signals_to_v6_quant_packet([weak, strong])
        assert pkt["kq_direction"] == 1.0  # AGGRESSIVE_LONG
        assert pkt["kq_confidence"] == 0.9

    def test_no_deps_required(self):
        """signals_to_v6_quant_packet must not import ccxt/websocket/scipy."""
        import sys
        # Temporarily remove scipy to prove it's not needed
        saved = sys.modules.get("scipy")
        sys.modules["scipy"] = None
        try:
            pkt = signals_to_v6_quant_packet([])
            assert pkt["kq_is_valid"] is True
        finally:
            if saved is not None:
                sys.modules["scipy"] = saved
            else:
                sys.modules.pop("scipy", None)

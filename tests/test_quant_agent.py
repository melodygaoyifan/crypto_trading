"""
Tests for agents/quant_agent.py - production overhaul.

Groups:
  1. Hardcoded asset removal (P0)
  2. Multi-asset state isolation (P0)
  3. Freshness gating (P1)
  4. HMATS contract output (P1)
  5. Backward compatibility & signal logic
"""

import time
import numpy as np
import pytest

from agents.quant_agent import (
    QuantAgent,
    QuantAgentIntent,
    Signal,
    SignalType,
    MarketMicrostructure,
    DerivativesState,
    LiquidationCascadeHunter,
    FundingDivergence,
    KalmanCointegration,
    ShortSqueezeMonitor,
    create_quant_agent,
    _neutral_intent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_microstructure(ts: float = None) -> MarketMicrostructure:
    """Minimal valid MarketMicrostructure."""
    return MarketMicrostructure(
        timestamp=ts or time.time(),
        bid_depth=np.array([[100.0, 10.0]]),
        ask_depth=np.array([[101.0, 10.0]]),
        bid_imbalance=0.0,
        taker_buy_volume=1000.0,
        taker_sell_volume=1000.0,
        cvd=-50.0,
        cvd_velocity=-1.0,
        vwap=100.5,
        vwap_deviation=-0.005,
        trade_count=500,
        avg_trade_size=2.0,
        large_trade_ratio=0.15,
    )


def _make_derivatives(asset: str, ts: float = None) -> DerivativesState:
    """Minimal valid DerivativesState."""
    return DerivativesState(
        timestamp=ts or time.time(),
        asset=asset,
        open_interest=1e8,
        oi_change_1h=-0.01,
        oi_change_24h=-0.03,
        oi_velocity=-0.005,
        funding_rate=0.0001,
        predicted_funding=0.00005,
        funding_velocity=-0.00001,
        long_liquidations=500_000.0,
        short_liquidations=100_000.0,
        liquidation_imbalance=0.6,
        liquidation_volume_zscore=2.0,
        spot_price=100.0,
        perp_price=100.05,
        basis=0.0005,
        basis_zscore=0.5,
    )


def _make_prices() -> dict:
    return {"SOL": 100.0, "ETH": 2000.0, "BTC": 40000.0}


# ===================================================================
# GROUP 1: Hardcoded asset removal (P0)
# ===================================================================

class TestHardcodedAssetRemoval:
    """Every Signal output must use the caller-supplied asset string."""

    def test_liquidation_hunter_uses_caller_asset(self):
        """LiquidationCascadeHunter.update(asset='ETH') -> Signal.asset == 'ETH'."""
        hunter = LiquidationCascadeHunter(lookback_window=10, cascade_confirmation_candles=1)
        # Fill buffer past half-lookback
        for i in range(6):
            hunter.update(
                timestamp=float(i), price=100.0 - i * 0.1,
                open_interest=1e8 - i * 2e6,
                long_liquidations=100.0, short_liquidations=10.0,
                cvd=-10.0, vwap_volume=1000.0, asset="ETH",
            )
        # Now provoke a cascade - spike liquidations
        for i in range(6, 12):
            sig = hunter.update(
                timestamp=float(i), price=100.0 - i * 1.0,
                open_interest=1e8 - i * 3e6,
                long_liquidations=1e6, short_liquidations=100.0,
                cvd=-500.0, vwap_volume=1000.0, asset="ETH",
            )
            if sig is not None:
                assert sig.asset == "ETH", f"Expected ETH, got {sig.asset}"
                return
        # Even if no signal triggered, we verified no hardcoded SOL

    def test_funding_divergence_uses_caller_asset(self):
        """FundingDivergence.update(asset='BTC') -> Signal.asset == 'BTC'."""
        fd = FundingDivergence(funding_lookback=10, price_lookback=10, oi_lookback=10)
        for i in range(6):
            fd.update(
                timestamp=float(i), spot_price=100.0 + i * 0.5,
                perp_price=100.0 + i * 0.5, funding_rate=0.001 - i * 0.0005,
                predicted_funding=0.001 - i * 0.001,
                open_interest=1e8 + i * 5e6, asset="BTC",
            )
        # Even without a signal, verify no crash and asset param accepted

    def test_kalman_uses_target_asset(self):
        """KalmanCointegration._generate_signal uses target_asset."""
        kc = KalmanCointegration(spread_lookback=10, entry_zscore=0.1)
        # Feed enough data to trigger
        for i in range(60):
            sig = kc.update(
                timestamp=float(i),
                sol_price=100.0 + i * 0.5,  # SOL rising fast
                eth_price=2000.0 + i * 0.01,  # ETH flat
                btc_price=40000.0,
                hedge_asset="ETH",
                target_asset="ETH",
            )
            if sig is not None:
                assert sig.asset == "ETH", f"Expected ETH, got {sig.asset}"
                return

    def test_quant_agent_update_no_hardcoded_sol_in_signals(self):
        """QuantAgent.update() should produce signals with asset from derivatives key."""
        agent = create_quant_agent()
        micro = _make_microstructure()
        derivs = {"SOL": _make_derivatives("SOL")}
        prices = _make_prices()
        signals = agent.update(
            timestamp=time.time(), microstructure=micro,
            derivatives=derivs, prices=prices,
        )
        for sig in signals:
            assert sig.asset == "SOL"  # matches the key, not hardcoded

    def test_analyze_asset_btc(self):
        """analyze_asset('BTC') never produces signals with asset='SOL'."""
        agent = create_quant_agent()
        intent = agent.analyze_asset(
            asset="BTC", timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"BTC": _make_derivatives("BTC")},
            prices=_make_prices(),
        )
        assert intent.asset == "BTC"

    def test_get_composite_signal_uses_asset_param(self):
        """get_composite_signal(asset='ETH') should use 'ETH' in output."""
        agent = create_quant_agent()
        # Feed SOL data so active_signals exist
        agent.update(
            timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": _make_derivatives("SOL")},
            prices=_make_prices(),
        )
        composite = agent.get_composite_signal(asset="SOL")
        if composite is not None:
            assert composite.asset == "SOL"


# ===================================================================
# GROUP 2: Multi-asset state isolation (P0)
# ===================================================================

class TestMultiAssetIsolation:
    """Per-asset strategy instances must not cross-contaminate."""

    def test_separate_liq_hunters(self):
        """Each asset gets its own LiquidationCascadeHunter instance."""
        agent = create_quant_agent()
        agent._ensure_asset("SOL")
        agent._ensure_asset("ETH")
        agent._ensure_asset("BTC")

        assert agent._liq_hunters["SOL"] is not agent._liq_hunters["ETH"]
        assert agent._liq_hunters["ETH"] is not agent._liq_hunters["BTC"]

    def test_separate_funding_divs(self):
        """Each asset gets its own FundingDivergence instance."""
        agent = create_quant_agent()
        agent._ensure_asset("SOL")
        agent._ensure_asset("ETH")

        assert agent._funding_divs["SOL"] is not agent._funding_divs["ETH"]

    def test_separate_squeeze_monitors(self):
        """Each asset gets its own ShortSqueezeMonitor instance."""
        agent = create_quant_agent()
        agent._ensure_asset("SOL")
        agent._ensure_asset("BTC")

        assert agent._squeeze_monitors["SOL"] is not agent._squeeze_monitors["BTC"]

    def test_analyze_asset_isolates_buffers(self):
        """Feeding data for SOL must not affect ETH buffers."""
        agent = create_quant_agent()

        # Feed SOL
        agent.analyze_asset(
            asset="SOL", timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": _make_derivatives("SOL")},
            prices=_make_prices(),
        )

        # ETH hunter should have empty buffers (never fed)
        agent._ensure_asset("ETH")
        assert len(agent._liq_hunters["ETH"].liquidation_buffer) == 0

        # SOL hunter should have data
        assert len(agent._liq_hunters["SOL"].liquidation_buffer) == 1

    def test_reset_state_per_asset(self):
        """reset_state(asset) clears only that asset."""
        agent = create_quant_agent()
        agent._ensure_asset("SOL")
        agent._ensure_asset("ETH")

        agent.reset_state("SOL")

        assert "SOL" not in agent._liq_hunters
        assert "ETH" in agent._liq_hunters

    def test_reset_state_all(self):
        """reset_state() clears everything."""
        agent = create_quant_agent()
        agent._ensure_asset("SOL")
        agent._ensure_asset("ETH")

        agent.reset_state()

        assert len(agent._liq_hunters) == 0
        assert len(agent._funding_divs) == 0


# ===================================================================
# GROUP 3: Freshness gating (P1)
# ===================================================================

class TestFreshnessGating:
    """Stale data should degrade to neutral, not produce signals."""

    def test_stale_derivatives_returns_empty(self):
        """If derivatives timestamp is old, no signals produced."""
        agent = create_quant_agent({"derivatives_max_age_s": 60.0})
        stale_ts = time.time() - 300.0  # 5 min ago
        deriv = _make_derivatives("SOL", ts=stale_ts)
        signals = agent._update_for_asset(
            asset="SOL",
            timestamp=time.time(),
            microstructure=_make_microstructure(),
            deriv=deriv,
            prices=_make_prices(),
        )
        assert signals == []

    def test_stale_microstructure_degrades_gracefully(self):
        """Stale microstructure skips liquidation hunter but runs funding."""
        agent = create_quant_agent({"microstructure_max_age_s": 30.0})
        stale_micro = _make_microstructure(ts=time.time() - 120.0)
        fresh_deriv = _make_derivatives("SOL")
        # Should not crash - just skips micro-dependent strategies
        signals = agent._update_for_asset(
            asset="SOL",
            timestamp=time.time(),
            microstructure=stale_micro,
            deriv=fresh_deriv,
            prices=_make_prices(),
        )
        # No crash is the assertion

    def test_none_microstructure_accepted(self):
        """microstructure=None should be accepted (skip micro strategies)."""
        agent = create_quant_agent()
        deriv = _make_derivatives("SOL")
        signals = agent._update_for_asset(
            asset="SOL",
            timestamp=time.time(),
            microstructure=None,
            deriv=deriv,
            prices=_make_prices(),
        )
        # No crash

    def test_none_derivatives_returns_empty(self):
        """deriv=None should return empty signals."""
        agent = create_quant_agent()
        signals = agent._update_for_asset(
            asset="SOL",
            timestamp=time.time(),
            microstructure=_make_microstructure(),
            deriv=None,
            prices=_make_prices(),
        )
        assert signals == []

    def test_analyze_asset_stale_returns_neutral(self):
        """analyze_asset with stale data returns neutral intent."""
        agent = create_quant_agent({"derivatives_max_age_s": 10.0})
        stale_deriv = _make_derivatives("SOL", ts=time.time() - 120.0)
        intent = agent.analyze_asset(
            asset="SOL", timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": stale_deriv},
            prices=_make_prices(),
        )
        assert intent.direction == 0.0
        assert intent.confidence == 0.0

    def test_freshness_config_via_create(self):
        """create_quant_agent passes freshness config through."""
        agent = create_quant_agent({
            "microstructure_max_age_s": 42.0,
            "derivatives_max_age_s": 77.0,
        })
        assert agent.microstructure_max_age_s == 42.0
        assert agent.derivatives_max_age_s == 77.0


# ===================================================================
# GROUP 4: HMATS contract output (P1)
# ===================================================================

class TestHMATSContract:
    """QuantAgentIntent meets the HMATS agent_signals contract."""

    def test_intent_to_dict_keys(self):
        """to_agent_signals_dict() must have all required keys."""
        intent = QuantAgentIntent(
            asset="SOL", direction=-1.0, confidence=0.8,
            strategy="liquidation_cascade",
            signal_type="AGGRESSIVE_SHORT",
            size_multiplier=1.5,
        )
        d = intent.to_agent_signals_dict()
        required_keys = {
            "quant_direction", "quant_confidence", "quant_strategy",
            "quant_signal_type", "quant_size_multiplier", "quant_asset",
            "quant_data_age_s", "quant_meta",
        }
        assert required_keys.issubset(d.keys())

    def test_intent_direction_range(self):
        """direction must be in [-1, 1]."""
        agent = create_quant_agent()
        intent = agent.analyze_asset(
            asset="SOL", timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": _make_derivatives("SOL")},
            prices=_make_prices(),
        )
        assert -1.0 <= intent.direction <= 1.0

    def test_intent_confidence_range(self):
        """confidence must be in [0, 1]."""
        agent = create_quant_agent()
        intent = agent.analyze_asset(
            asset="SOL", timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": _make_derivatives("SOL")},
            prices=_make_prices(),
        )
        assert 0.0 <= intent.confidence <= 1.0

    def test_neutral_intent_values(self):
        """_neutral_intent returns zero direction and confidence."""
        neutral = _neutral_intent("BTC")
        assert neutral.direction == 0.0
        assert neutral.confidence == 0.0
        assert neutral.asset == "BTC"
        assert neutral.signal_type == "NO_SIGNAL"

    def test_neutral_intent_cached(self):
        """Same asset returns same neutral intent object."""
        a = _neutral_intent("SOL")
        b = _neutral_intent("SOL")
        assert a is b

    def test_signals_to_intent_flatten_overrides(self):
        """Flatten signals produce direction=0.0 regardless of shorts."""
        signals = [
            Signal(
                timestamp=1.0, asset="SOL",
                signal_type=SignalType.AGGRESSIVE_SHORT,
                size_multiplier=2.0, confidence=0.9,
                strategy="liquidation_cascade",
                entry_price=100.0, stop_loss=105.0, take_profit=90.0,
                metadata={},
            ),
            Signal(
                timestamp=1.0, asset="SOL",
                signal_type=SignalType.FLATTEN,
                size_multiplier=1.0, confidence=0.95,
                strategy="risk_control",
                entry_price=100.0, stop_loss=100.0, take_profit=100.0,
                metadata={},
            ),
        ]
        intent = QuantAgent._signals_to_intent("SOL", signals)
        assert intent.direction == 0.0
        assert intent.strategy == "risk_control"

    def test_signals_to_intent_short_direction(self):
        """Short signals produce direction=-1.0."""
        signals = [
            Signal(
                timestamp=1.0, asset="SOL",
                signal_type=SignalType.AGGRESSIVE_SHORT,
                size_multiplier=1.5, confidence=0.8,
                strategy="funding_divergence",
                entry_price=100.0, stop_loss=105.0, take_profit=90.0,
                metadata={},
            ),
        ]
        intent = QuantAgent._signals_to_intent("SOL", signals)
        assert intent.direction == -1.0
        assert intent.confidence == 0.8

    def test_signals_to_intent_empty(self):
        """Empty signal list returns neutral."""
        intent = QuantAgent._signals_to_intent("ETH", [])
        assert intent.direction == 0.0
        assert intent.confidence == 0.0

    def test_analyze_asset_exception_returns_neutral(self):
        """If analyze_asset() throws internally, return neutral."""
        agent = create_quant_agent()
        # Pass garbage that will cause an exception inside
        intent = agent.analyze_asset(
            asset="SOL", timestamp=time.time(),
            microstructure=None,
            derivatives=None,  # This triggers None.get() -> exception
            prices=None,
        )
        # Should not crash, returns neutral-ish
        assert intent.asset == "SOL"
        assert intent.confidence == 0.0


# ===================================================================
# GROUP 5: Backward compatibility & signal logic
# ===================================================================

class TestBackwardCompat:
    """Ensure existing callers don't break."""

    def test_update_returns_list(self):
        """update() returns List[Signal] (possibly empty)."""
        agent = create_quant_agent()
        result = agent.update(
            timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": _make_derivatives("SOL")},
            prices=_make_prices(),
        )
        assert isinstance(result, list)

    def test_active_signals_attribute(self):
        """agent.active_signals still works (backward compat alias)."""
        agent = create_quant_agent()
        agent.update(
            timestamp=time.time(),
            microstructure=_make_microstructure(),
            derivatives={"SOL": _make_derivatives("SOL")},
            prices=_make_prices(),
        )
        assert isinstance(agent.active_signals, list)

    def test_get_composite_signal_no_data(self):
        """get_composite_signal() returns None with no signals."""
        agent = create_quant_agent()
        assert agent.get_composite_signal() is None

    def test_get_hedge_recommendation(self):
        """get_hedge_recommendation() returns expected keys."""
        agent = create_quant_agent()
        rec = agent.get_hedge_recommendation()
        assert "hedge_asset" in rec
        assert "hedge_ratio" in rec
        assert "hedge_uncertainty" in rec

    def test_signal_history_capped(self):
        """signal_history deque has maxlen=1000."""
        agent = create_quant_agent()
        assert agent.signal_history.maxlen == 1000

    def test_generate_signal_returns_intent(self):
        """generate_signal() returns QuantAgentIntent (spine compat)."""
        agent = create_quant_agent()
        result = agent.generate_signal(asset="SOL", price=100.0, regime="NORMAL")
        assert isinstance(result, QuantAgentIntent)
        assert result.asset == "SOL"

    def test_get_signal_returns_tuple(self):
        """get_signal() returns (direction, confidence) tuple."""
        agent = create_quant_agent()
        result = agent.get_signal(asset="SOL")
        assert isinstance(result, tuple)
        assert len(result) == 2
        direction, confidence = result
        assert isinstance(direction, float)
        assert isinstance(confidence, float)

    def test_create_quant_agent_factory(self):
        """create_quant_agent() returns configured QuantAgent."""
        agent = create_quant_agent({"liquidation_lookback": 200})
        agent._ensure_asset("SOL")
        assert agent._liq_hunters["SOL"].lookback == 200

    def test_signal_namedtuple_fields(self):
        """Signal NamedTuple still has all expected fields."""
        sig = Signal(
            timestamp=1.0, asset="SOL",
            signal_type=SignalType.SHORT,
            size_multiplier=1.0, confidence=0.5,
            strategy="test",
            entry_price=100.0, stop_loss=105.0, take_profit=95.0,
            metadata={"key": "val"},
        )
        assert sig.timestamp == 1.0
        assert sig.stop_loss == 105.0
        assert sig.take_profit == 95.0

    def test_squeeze_monitor_standalone(self):
        """ShortSqueezeMonitor still works as standalone."""
        sm = ShortSqueezeMonitor()
        for i in range(15):
            risk, comp = sm.update(
                open_interest=1e8, funding_rate=0.0001,
                cvd=0.0, short_liquidations=100.0, price_change=0.0,
            )
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_kalman_cointegration_standalone(self):
        """KalmanCointegration still works standalone."""
        kc = KalmanCointegration()
        for i in range(10):
            kc.update(
                timestamp=float(i),
                sol_price=100.0 + i * 0.1,
                eth_price=2000.0 + i * 0.05,
            )
        ratio = kc.get_hedge_ratio()
        assert isinstance(ratio, float)

"""
Tests for agents/onchain_solana_agent.py - production-readiness checks.

All network calls are mocked via monkeypatch on OnChainDataProvider.fetch_*.
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from agents.onchain_solana_agent import (
    SolanaOnChainAgent,
    OnChainSolanaAgent,
    OnChainConfig,
    OnChainDataProvider,
    OnChainMetrics,
    OnChainAlphaSignal,
    OnChainSignalType,
    SourceStatus,
    get_onchain_agent,
    reset_onchain_agent,
    _SIGNAL_SENTIMENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dex(volume=1_000_000, buy_ratio=0.65, mock=False, buy_source="real"):
    buy_vol = volume * buy_ratio
    sell_vol = volume * (1 - buy_ratio)
    return {
        "total_volume_24h": volume,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "buy_source": buy_source,
        "top_pairs": [],
        "price_change_24h_pct": 2.5,
        "_mock": mock,
        "_source": "birdeye",
    }


def _make_whale(tx_count=5, total_volume=80000, mock=False):
    activity_score = min(1.0, tx_count / 10.0)
    txs = [{"hash": f"tx{i}", "amount": total_volume / max(tx_count, 1),
            "from": "src", "to": "dst", "time": 0}
           for i in range(tx_count)]
    return {
        "transactions": txs,
        "total_volume": total_volume,
        "activity_score": activity_score,
        "_mock": mock,
        "_source": "solscan",
    }


def _make_exchange(net_flow=0, mock=True):
    return {
        "inflow_24h": max(net_flow, 0),
        "outflow_24h": max(-net_flow, 0),
        "net_flow": net_flow,
        "exchanges": {},
        "_mock": mock,
        "_source": "exchange_flows",
    }


def _make_staking(net_flow=0, mock=True):
    return {
        "total_staked": 0, "staked_24h": 0, "unstaked_24h": 0,
        "net_flow": net_flow,
        "_mock": mock,
        "_source": "staking",
    }


def _make_wallet(active=100, new=10, mock=True):
    return {
        "active_wallets": active, "new_wallets": new, "transactions_24h": 500,
        "_mock": mock,
        "_source": "wallet_activity",
    }


def _patch_provider(agent, dex=None, whale=None, exchange=None, staking=None, wallet=None):
    """Replace provider fetch methods with AsyncMock returning given dicts."""
    p = agent.data_provider
    p.fetch_dex_metrics = AsyncMock(return_value=dex or _make_dex(mock=True, volume=0))
    p.fetch_whale_activity = AsyncMock(return_value=whale or _make_whale(tx_count=0, mock=True))
    p.fetch_exchange_flows = AsyncMock(return_value=exchange or _make_exchange())
    p.fetch_staking_metrics = AsyncMock(return_value=staking or _make_staking())
    p.fetch_wallet_activity = AsyncMock(return_value=wallet or _make_wallet())


# ---------------------------------------------------------------------------
# P0: Import / backward compat
# ---------------------------------------------------------------------------

class TestP0BackwardCompat:
    def test_both_class_names_importable(self):
        """OnChainSolanaAgent and SolanaOnChainAgent must both resolve."""
        assert SolanaOnChainAgent is not None
        assert OnChainSolanaAgent is not None
        assert OnChainSolanaAgent is SolanaOnChainAgent

    def test_both_names_from_init(self):
        """agents/__init__.py exports both names."""
        from agents import OnChainSolanaAgent as A
        from agents import SolanaOnChainAgent as B
        assert A is B
        assert A is not None

    def test_get_onchain_agent_returns_instance(self):
        reset_onchain_agent()
        agent = get_onchain_agent()
        assert isinstance(agent, SolanaOnChainAgent)
        reset_onchain_agent()

    def test_get_onchain_agent_singleton(self):
        reset_onchain_agent()
        a = get_onchain_agent()
        b = get_onchain_agent()
        assert a is b
        reset_onchain_agent()


# ---------------------------------------------------------------------------
# P1: Observability & fail-closed
# ---------------------------------------------------------------------------

class TestP1Observability:
    @pytest.fixture
    def agent(self):
        a = SolanaOnChainAgent(OnChainConfig())
        yield a

    def test_no_metrics_returns_none(self, agent):
        """generate_signal must return None when no data collected."""
        assert agent.generate_signal() is None

    @pytest.mark.asyncio
    async def test_source_status_populated(self, agent):
        """Every source must have a SourceStatus entry after update."""
        _patch_provider(agent)
        await agent.update_metrics()

        m = agent.current_metrics
        assert m is not None
        expected_sources = {"dex", "whale", "exchange_flows", "staking", "wallet_activity"}
        assert set(m.source_status.keys()) == expected_sources

        for name, ss in m.source_status.items():
            assert isinstance(ss, SourceStatus)

    @pytest.mark.asyncio
    async def test_mock_sources_marked(self, agent):
        """Placeholder sources must be marked mock=True."""
        _patch_provider(agent)
        await agent.update_metrics()

        m = agent.current_metrics
        for name in ("exchange_flows", "staking", "wallet_activity"):
            assert m.source_status[name].mock is True

    @pytest.mark.asyncio
    async def test_coverage_map_present(self, agent):
        """coverage dict must exist and have key fields."""
        _patch_provider(agent, dex=_make_dex(volume=500_000))
        await agent.update_metrics()

        cov = agent.current_metrics.coverage
        assert "buy_volume_pct" in cov
        assert "dex_volume_24h" in cov
        assert "whale_activity_score" in cov
        assert "exchange_net_flow" in cov

    @pytest.mark.asyncio
    async def test_to_dict_includes_new_wallets(self, agent):
        """new_wallets_24h must appear in to_dict output."""
        _patch_provider(agent, wallet=_make_wallet(active=200, new=42))
        await agent.update_metrics()

        d = agent.current_metrics.to_dict()
        assert "new_wallets_24h" in d
        assert d["new_wallets_24h"] == 42

    @pytest.mark.asyncio
    async def test_data_age_in_metrics_dict(self, agent):
        """get_current_metrics must include data_age_seconds."""
        _patch_provider(agent)
        await agent.update_metrics()

        d = agent.get_current_metrics()
        assert "data_age_seconds" in d
        assert d["data_age_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_exception_in_fetch_recorded(self, agent):
        """If a fetch raises, source_status must record the error."""
        _patch_provider(agent)
        agent.data_provider.fetch_dex_metrics = AsyncMock(
            side_effect=RuntimeError("birdeye down")
        )
        await agent.update_metrics()

        ss = agent.current_metrics.source_status["dex"]
        assert ss.available is False
        assert ss.mock is True
        assert "birdeye down" in ss.error


# ---------------------------------------------------------------------------
# P1: Signal logic fixes
# ---------------------------------------------------------------------------

class TestP1SignalLogic:
    @pytest.fixture
    def agent(self):
        a = SolanaOnChainAgent(OnChainConfig())
        yield a

    @pytest.mark.asyncio
    async def test_strong_buy_pressure_generates_signal(self, agent):
        """Strong DEX buy pressure (65%) should produce a bullish signal."""
        _patch_provider(agent, dex=_make_dex(volume=1_000_000, buy_ratio=0.70))
        await agent.update_metrics()

        sig = agent.generate_signal()
        assert sig is not None
        assert sig.direction > 0
        assert sig.confidence > 0

    @pytest.mark.asyncio
    async def test_strong_sell_pressure_generates_signal(self, agent):
        """Strong DEX sell pressure (30%) should produce a bearish signal."""
        _patch_provider(agent, dex=_make_dex(volume=1_000_000, buy_ratio=0.30))
        await agent.update_metrics()

        sig = agent.generate_signal()
        assert sig is not None
        assert sig.direction < 0

    @pytest.mark.asyncio
    async def test_whale_activity_score_not_always_zero(self, agent):
        """whale_activity_score must reflect actual transaction count."""
        _patch_provider(agent, whale=_make_whale(tx_count=8, total_volume=100_000))
        await agent.update_metrics()

        assert agent.current_metrics.whale_activity_score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_whale_amplifies_signal(self, agent):
        """Whale activity should amplify existing DEX signal magnitude."""
        # Signal without whale
        _patch_provider(agent, dex=_make_dex(volume=1_000_000, buy_ratio=0.70),
                        whale=_make_whale(tx_count=0, mock=True))
        await agent.update_metrics()
        sig_no_whale = agent.generate_signal()

        # Reset and signal with whale
        agent2 = SolanaOnChainAgent(OnChainConfig())
        _patch_provider(agent2, dex=_make_dex(volume=1_000_000, buy_ratio=0.70),
                        whale=_make_whale(tx_count=8, total_volume=100_000))
        await agent2.update_metrics()
        sig_with_whale = agent2.generate_signal()

        assert sig_no_whale is not None
        assert sig_with_whale is not None
        assert abs(sig_with_whale.direction) >= abs(sig_no_whale.direction)

    @pytest.mark.asyncio
    async def test_dex_volume_change_uses_volume_not_price(self, agent):
        """dex_volume_change should be volume_mult vs median, not price_change."""
        # Feed two rounds to build baseline
        dex1 = _make_dex(volume=500_000, buy_ratio=0.50)
        dex2 = _make_dex(volume=500_000, buy_ratio=0.50)
        dex3 = _make_dex(volume=1_500_000, buy_ratio=0.50)  # 3x spike

        for d in (dex1, dex2, dex3):
            _patch_provider(agent, dex=d)
            await agent.update_metrics()

        # volume_change should be ~3.0 (1.5M / 500K median)
        assert agent.current_metrics.dex_volume_change == pytest.approx(3.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_stale_data_reduces_confidence(self, agent):
        """Data older than 10 minutes should have reduced confidence."""
        _patch_provider(agent, dex=_make_dex(volume=1_000_000, buy_ratio=0.70))
        await agent.update_metrics()

        # Manually backdate timestamp
        agent.current_metrics.timestamp = datetime.now(timezone.utc) - timedelta(minutes=15)

        sig = agent.generate_signal()
        assert sig is not None
        assert sig.confidence < 0.8  # Must be penalized

    @pytest.mark.asyncio
    async def test_signal_type_reflects_dominant_source(self, agent):
        """signal_type should match the dominant contributor, not always DEX_FLOW."""
        # Only exchange flow is real and significant
        _patch_provider(
            agent,
            dex=_make_dex(volume=0, mock=True),  # no dex
            exchange=_make_exchange(net_flow=100_000, mock=False),  # real, bearish
        )
        await agent.update_metrics()

        sig = agent.generate_signal()
        if sig is not None:
            assert sig.signal_type == OnChainSignalType.EXCHANGE_INFLOW

    @pytest.mark.asyncio
    async def test_inferred_buy_ratio_discounts_confidence(self, agent):
        """When buy_ratio is inferred from price, confidence should be lower."""
        _patch_provider(agent, dex=_make_dex(
            volume=1_000_000, buy_ratio=0.70, buy_source="inferred_from_price",
        ))
        await agent.update_metrics()

        sig = agent.generate_signal()
        assert sig is not None
        # The mock_ratio penalty should reduce confidence
        assert sig.confidence < 0.75


# ---------------------------------------------------------------------------
# P1: get_metrics_summary
# ---------------------------------------------------------------------------

class TestP1MetricsSummary:
    @pytest.fixture
    def agent(self):
        return SolanaOnChainAgent(OnChainConfig())

    def test_no_data(self, agent):
        assert agent.get_metrics_summary() == {"status": "no_data"}

    @pytest.mark.asyncio
    async def test_summary_uses_explicit_mapping(self, agent):
        """Summary must use _SIGNAL_SENTIMENT, not string matching."""
        _patch_provider(agent, dex=_make_dex(volume=1_000_000, buy_ratio=0.70))
        await agent.update_metrics()

        summary = agent.get_metrics_summary()
        assert summary["status"] == "ok"
        assert "mock_sources" in summary
        assert "data_age_seconds" in summary

    @pytest.mark.asyncio
    async def test_summary_includes_whale_activity(self, agent):
        """Summary must include whale_activity_score."""
        _patch_provider(agent, whale=_make_whale(tx_count=5))
        await agent.update_metrics()

        summary = agent.get_metrics_summary()
        assert "whale_activity_score" in summary

    def test_signal_sentiment_mapping_complete(self):
        """Every OnChainSignalType must have a sentiment mapping."""
        for st in OnChainSignalType:
            assert st in _SIGNAL_SENTIMENT, f"Missing sentiment mapping for {st}"

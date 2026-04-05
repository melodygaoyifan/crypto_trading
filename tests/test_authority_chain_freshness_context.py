import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


from core.authority_chain import AuthorityChain, AuthorityStage
from defense.trade_gate import TradeGate, TradeGateConfig


class _RiskManagerAllow:
    def check_trade_allowed(self, **kwargs):
        return {"approved": True}


class _LeverageGuardAllow:
    def validate_order(self, **kwargs):
        return True, "ok"


def _build_chain() -> tuple[AuthorityChain, TradeGate]:
    trade_gate = TradeGate(
        TradeGateConfig(
            max_data_age_seconds=60.0,
            max_snapshot_age_at_fetch_seconds=5.0,
            tick_processing_grace_seconds=120.0,
            max_cached_orderbook_age_for_tick_grace_seconds=120.0,
        )
    )
    stale_ts = time.time() - 61.0
    for source in ("price", "orderbook", "vpin"):
        trade_gate.stale_guard.update_timestamp(source, stale_ts)

    chain = AuthorityChain(
        trade_gate=trade_gate,
        risk_manager=_RiskManagerAllow(),
        leverage_guard=_LeverageGuardAllow(),
        strict_mode=False,
    )
    return chain, trade_gate


def test_authority_chain_forwards_cached_depth_context():
    chain, _ = _build_chain()

    result = chain.evaluate(
        asset="ETH",
        direction=-0.2,
        exposure_fraction=0.05,
        base_quantity=0.2,
        price=2000.0,
        account_equity=10_000.0,
        confidence=0.5,
        regime="QUIET_ACCUMULATION",
        regime_confidence=0.8,
        expected_alpha_bps=40.0,
        volume_ratio=1.0,
        price_direction="down",
        market_data_age_seconds=0.5,
        market_data_fetched_at_ts=time.time() - 68.0,
        orderbook_stale=True,
        orderbook_fallback_reason="cached_depth",
        orderbook_cache_age_seconds=40.0,
    )

    assert result.approved is True
    assert result.final_stage == AuthorityStage.EXECUTION


def test_authority_chain_rejects_static_floor_fallback():
    chain, _ = _build_chain()

    result = chain.evaluate(
        asset="ETH",
        direction=-0.2,
        exposure_fraction=0.05,
        base_quantity=0.2,
        price=2000.0,
        account_equity=10_000.0,
        confidence=0.5,
        regime="QUIET_ACCUMULATION",
        regime_confidence=0.8,
        expected_alpha_bps=40.0,
        volume_ratio=1.0,
        price_direction="down",
        market_data_age_seconds=0.5,
        market_data_fetched_at_ts=time.time() - 68.0,
        orderbook_stale=True,
        orderbook_fallback_reason="static_floor",
        orderbook_cache_age_seconds=None,
    )

    assert result.approved is False
    assert result.veto_stage == AuthorityStage.TRADE_GATE
    assert "STALE_DATA" in result.veto_reason


def test_authority_chain_forwards_alpha_gate_passed_to_trade_gate():
    chain, _ = _build_chain()

    result = chain.evaluate(
        asset="BTC",
        direction=0.2,
        exposure_fraction=0.03,
        base_quantity=0.0042,
        price=71_500.0,
        account_equity=10_000.0,
        confidence=0.8,
        regime="QUIET_ACCUMULATION",
        regime_confidence=0.95,
        expected_alpha_bps=9.0,
        alpha_gate_passed=True,
        volume_ratio=1.0,
        price_direction="up",
        market_data_age_seconds=0.5,
        market_data_fetched_at_ts=time.time() - 1.0,
        orderbook_stale=False,
        orderbook_fallback_reason="",
        orderbook_cache_age_seconds=None,
    )

    assert result.approved is True
    assert result.final_stage == AuthorityStage.EXECUTION

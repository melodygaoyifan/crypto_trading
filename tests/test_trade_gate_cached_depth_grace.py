import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


from defense.trade_gate import TradeGate, TradeGateConfig


def _stale_sources(gate: TradeGate, age_seconds: float = 61.0) -> None:
    stale_ts = time.time() - age_seconds
    for source in ("price", "orderbook", "vpin"):
        gate.stale_guard.update_timestamp(source, stale_ts)


def test_cached_depth_with_bounded_age_uses_tick_grace():
    gate = TradeGate(
        TradeGateConfig(
            max_data_age_seconds=60.0,
            max_snapshot_age_at_fetch_seconds=5.0,
            tick_processing_grace_seconds=120.0,
            max_cached_orderbook_age_for_tick_grace_seconds=120.0,
        )
    )
    _stale_sources(gate)

    is_fresh, details = gate._check_freshness_with_context(
        market_data_age_seconds=0.5,
        market_data_fetched_at_ts=time.time() - 68.0,
        orderbook_stale=True,
        orderbook_fallback_reason="cached_depth",
        orderbook_cache_age_seconds=42.0,
    )

    assert is_fresh is True
    assert details["freshness_mode"] == "tick_grace"
    assert details["orderbook_cached_depth_grace"] is True
    assert details["orderbook_fallback_reason"] == "cached_depth"
    assert details["orderbook_cache_age_seconds"] == 42.0


def test_static_floor_orderbook_does_not_get_tick_grace():
    gate = TradeGate(TradeGateConfig(max_data_age_seconds=60.0))
    _stale_sources(gate)

    is_fresh, details = gate._check_freshness_with_context(
        market_data_age_seconds=0.5,
        market_data_fetched_at_ts=time.time() - 68.0,
        orderbook_stale=True,
        orderbook_fallback_reason="static_floor",
        orderbook_cache_age_seconds=None,
    )

    assert is_fresh is False
    assert details["stale_sources"] == ["price", "orderbook", "vpin"]
    assert details["orderbook_fallback_reason"] == "static_floor"


def test_cached_depth_older_than_bound_is_rejected():
    gate = TradeGate(
        TradeGateConfig(
            max_data_age_seconds=60.0,
            max_cached_orderbook_age_for_tick_grace_seconds=90.0,
        )
    )
    _stale_sources(gate)

    is_fresh, details = gate._check_freshness_with_context(
        market_data_age_seconds=0.5,
        market_data_fetched_at_ts=time.time() - 68.0,
        orderbook_stale=True,
        orderbook_fallback_reason="cached_depth",
        orderbook_cache_age_seconds=95.0,
    )

    assert is_fresh is False
    assert details["stale_sources"] == ["price", "orderbook", "vpin"]
    assert details["orderbook_cache_age_seconds"] == 95.0

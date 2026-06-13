"""
HMATS v5.1 Phase 2 - execute_intent_v2 Coinbase fork routing gate.

Proves the fork is INERT by default (Kraken path untouched) and only engages
when flag ON + RoutingPolicy advanced to DUAL for that asset. Fail-closed.
"""
from types import SimpleNamespace

import core.execution_service as es
from exchange.routing import CutoverPhase, RoutingPolicy


def _ctx(flag):
    return SimpleNamespace(config=SimpleNamespace(coinbase_routing_enabled=flag))


def teardown_function(_):
    es._CB_ROUTING = None  # reset module cache between tests


def test_routed_false_when_flag_off_even_if_dual():
    es._CB_ROUTING = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE,
                                   coinbase_assets=["SOL"])
    assert es._coinbase_routed(_ctx(False), "SOL") is False


def test_routed_false_in_pre_phase2_default():
    es._CB_ROUTING = RoutingPolicy(phase=CutoverPhase.PRE_PHASE_2)
    assert es._coinbase_routed(_ctx(True), "SOL") is False


def test_routed_false_in_shadow():
    es._CB_ROUTING = RoutingPolicy(phase=CutoverPhase.SHADOW,
                                   coinbase_assets=["SOL"])
    assert es._coinbase_routed(_ctx(True), "SOL") is False  # shadow = no orders


def test_routed_true_only_for_dual_coinbase_asset():
    es._CB_ROUTING = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE,
                                   coinbase_assets=["SOL"],
                                   kraken_assets=["BTC", "ETH"])
    assert es._coinbase_routed(_ctx(True), "SOL") is True
    assert es._coinbase_routed(_ctx(True), "BTC") is False  # stays kraken


def test_routed_fail_closed_on_bad_ctx():
    assert es._coinbase_routed(object(), "SOL") is False

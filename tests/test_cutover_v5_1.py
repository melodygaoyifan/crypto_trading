"""
HMATS v5.1 Phase 2 - tests for cutover invariants + Coinbase funding scaffold.

Covers:
  - exchange.cutover Iron Law 1/5/8/9 invariant functions
  - assert_safe_to_advance combined gate (ROLLBACK bypass, DRL gate, transitions)
  - CoinbaseFundingFeed fail-closed behavior + 8h normalization
"""
import asyncio

import pytest

from exchange.cutover import (
    REQUIRED_OBS_DIM,
    CutoverInvariantResult,
    assert_safe_to_advance,
    cutover_invariants,
    validate_drl_active,
    validate_maker_first,
    validate_obs_dim,
)
from exchange.routing import CutoverPhase, RoutingPolicy
from data_mgmt.feeds.coinbase_funding_feed import CoinbaseFundingFeed


# ---------- individual invariants ----------------------------------------

@pytest.mark.parametrize("level,ok", [
    ("ACTIVE", True), ("active", True),
    ("SHADOW", False), ("DISABLED", False), ("", False), (None, False),
])
def test_validate_drl_active(level, ok):
    assert validate_drl_active(level)[0] is ok


@pytest.mark.parametrize("dim,ok", [(126, True), (125, False), (1008, False), (0, False)])
def test_validate_obs_dim(dim, ok):
    assert validate_obs_dim(dim)[0] is ok
    assert REQUIRED_OBS_DIM == 126


def test_validate_maker_first():
    assert validate_maker_first(True)[0] is True
    assert validate_maker_first(False)[0] is False


# ---------- aggregate sweep ----------------------------------------------

def test_cutover_invariants_all_hold():
    r = cutover_invariants("ACTIVE", obs_dim=126, post_only_default=True)
    assert isinstance(r, CutoverInvariantResult)
    assert r.ok is True
    assert r.violations == []
    assert set(r.checked) == {"obs_dim", "drl_active", "maker_first"}


def test_cutover_invariants_catches_each_violation():
    assert cutover_invariants("SHADOW", 126, True).ok is False
    assert cutover_invariants("ACTIVE", 125, True).ok is False
    assert cutover_invariants("ACTIVE", 126, False).ok is False
    multi = cutover_invariants("SHADOW", 999, False)
    assert multi.ok is False
    assert len(multi.violations) == 3


# ---------- combined safe-to-advance gate ---------------------------------

def test_rollback_always_safe_even_when_drl_demoted():
    p = RoutingPolicy()
    ok, reason = assert_safe_to_advance(p, CutoverPhase.ROLLBACK, "SHADOW")
    assert ok is True
    assert "ROLLBACK" in reason


def test_advance_blocked_when_drl_not_active():
    p = RoutingPolicy()
    ok, reason = assert_safe_to_advance(p, CutoverPhase.SHADOW, "SHADOW")
    assert ok is False
    assert "Iron Law" in reason


def test_advance_allowed_pre_to_shadow_when_active():
    p = RoutingPolicy()
    ok, reason = assert_safe_to_advance(p, CutoverPhase.SHADOW, "ACTIVE")
    assert ok is True


def test_advance_rejects_invalid_skip_transition():
    p = RoutingPolicy()  # PRE_PHASE_2
    # PRE_PHASE_2 -> COINBASE_PRIMARY is not a legal single step
    ok, reason = assert_safe_to_advance(p, CutoverPhase.COINBASE_PRIMARY, "ACTIVE")
    assert ok is False
    assert "invalid transition" in reason


def test_assert_safe_does_not_mutate_policy():
    p = RoutingPolicy()
    assert_safe_to_advance(p, CutoverPhase.SHADOW, "ACTIVE")
    assert p.phase == CutoverPhase.PRE_PHASE_2  # unchanged


# ---------- Coinbase funding feed (fail-closed scaffold) ------------------

def test_funding_feed_disabled_by_default_returns_none():
    feed = CoinbaseFundingFeed()  # no client -> disabled
    assert feed.is_enabled() is False
    assert asyncio.run(feed.fetch_funding_rate("BTC")) is None


def test_funding_feed_normalize_to_8h():
    # hourly rate 0.0001 -> 8h equivalent 0.0008
    assert CoinbaseFundingFeed._normalize_to_8h(0.0001, 1.0) == pytest.approx(0.0008)
    # already-8h rate is unchanged
    assert CoinbaseFundingFeed._normalize_to_8h(0.0005, 8.0) == pytest.approx(0.0005)
    # zero/garbage period -> returns rate unchanged (no div-by-zero)
    assert CoinbaseFundingFeed._normalize_to_8h(0.0003, 0.0) == pytest.approx(0.0003)


def test_funding_feed_unknown_asset_returns_none_when_enabled():
    # enabled with a dummy client, but unknown asset -> symbol map miss -> None
    feed = CoinbaseFundingFeed(ccxt_client=object(), enabled=True)
    assert feed.is_enabled() is True
    assert asyncio.run(feed.fetch_funding_rate("DOGE")) is None

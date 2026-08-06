"""[P155] Iron Law 8 was defined but never called — a P152-class defect.

`RoutingPolicy.advance_phase()` enforces "DRL must be ACTIVE to advance the
cutover", but production never calls advance_phase: `_coinbase_get_routing()`
assigns `rp.phase` straight from data/coinbase_routing_state.json. So the phase
could be advanced by editing a JSON file with DRL demoted to SHADOW and no
guard ever ran, while three docstrings claimed a "continuous check".

These tests pin the wiring AND the deliberate decision that it observes rather
than blocks: fail-closing here would route the asset back to Kraken, which is
structurally flat post-Phase-B, turning a DRL-authority problem into a silent
total trading stop.
"""

import logging
import types

import pytest

from core import execution_service as es
from exchange.routing import CutoverPhase, RoutingPolicy


@pytest.fixture(autouse=True)
def _reset_once_flag():
    es._CB_IRON_LAW_8_WARNED = False
    yield
    es._CB_IRON_LAW_8_WARNED = False


def _ctx(drl_level="ACTIVE", routing_enabled=True):
    return types.SimpleNamespace(
        drl_authority_level=drl_level,
        config=types.SimpleNamespace(coinbase_routing_enabled=routing_enabled),
    )


def _critical(caplog):
    return [r for r in caplog.records if r.levelno >= logging.CRITICAL]


# ---------------------------------------------------------------------------
# the check itself
# ---------------------------------------------------------------------------

def test_demoted_drl_mid_cutover_is_reported(caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(_ctx("SHADOW"), rp)
    msgs = [r.message for r in _critical(caplog)]
    assert any("IRON-LAW-8" in m for m in msgs), msgs
    assert any("DUAL_VENUE" in m for m in msgs)


def test_active_drl_mid_cutover_is_silent(caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(_ctx("ACTIVE"), rp)
    assert _critical(caplog) == []


def test_pre_phase_2_is_not_a_cutover_so_no_alarm(caplog):
    """Kraken-only is the default resting state, not a violation."""
    rp = RoutingPolicy(phase=CutoverPhase.PRE_PHASE_2)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(_ctx("SHADOW"), rp)
    assert _critical(caplog) == []


def test_missing_authority_attribute_fails_closed_to_reporting(caplog):
    rp = RoutingPolicy(phase=CutoverPhase.COINBASE_PRIMARY)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(types.SimpleNamespace(), rp)
    assert _critical(caplog), "an absent authority level is NOT ACTIVE"


def test_reported_once_per_process_not_every_tick(caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    with caplog.at_level(logging.CRITICAL):
        for _ in range(5):
            es._coinbase_check_iron_law_8(_ctx("SHADOW"), rp)
    assert len(_critical(caplog)) == 1


def test_check_never_raises_into_the_order_path():
    """A diagnostics failure must not break routing."""
    es._coinbase_check_iron_law_8(_ctx("SHADOW"), object())  # no .phase


# ---------------------------------------------------------------------------
# [P193] the caller passes two different shapes
# ---------------------------------------------------------------------------

def _runner_ctx(drl_level="ACTIVE", routing_enabled=True):
    """The HMATSProductionRunner shape.

    main.py:8562 calls `_coinbase_routed(self, asset)` for the P172 venue-fee
    resolution, so `ctx` is the runner, which names the authority
    `_drl_authority_level`. Every _ctx() above is the ExecutionContext shape;
    nothing covered this one, which is how P193 shipped.
    """
    return types.SimpleNamespace(
        _drl_authority_level=drl_level,
        config=types.SimpleNamespace(coinbase_routing_enabled=routing_enabled),
    )


def test_active_drl_on_the_runner_shape_is_silent(caplog):
    """The live false alarm: runner says ACTIVE, check read '' and cried wolf."""
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(_runner_ctx("ACTIVE"), rp)
    assert _critical(caplog) == [], (
        "DRL is ACTIVE on the runner but Iron Law 8 reported a violation — it "
        "is reading `drl_authority_level` and missing `_drl_authority_level`."
    )


def test_demoted_drl_on_the_runner_shape_is_still_reported(caplog):
    """Falsification: the fix must not make the runner path unconditionally quiet."""
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(_runner_ctx("SHADOW"), rp)
    assert any("IRON-LAW-8" in r.message for r in _critical(caplog))


def test_a_false_alarm_does_not_burn_the_one_shot_for_a_real_one(caplog):
    """The consequence that made P193 worth fixing.

    The latch fires once per process. A false positive on the runner path spent
    it during warmup, so a genuine demotion later in the same process was never
    reported at all.
    """
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(_runner_ctx("ACTIVE"), rp)   # must stay silent
        es._coinbase_check_iron_law_8(_ctx("SHADOW"), rp)          # real violation
    msgs = [r.message for r in _critical(caplog)]
    assert len(msgs) == 1, msgs
    # Asserting the COUNT alone would pass either way — pre-fix the single
    # alert is the phantom "authority is ''" and the real SHADOW demotion is
    # the one that got swallowed. Assert WHICH violation survived.
    assert "'SHADOW'" in msgs[0], (
        f"the surviving alert is not the real demotion: {msgs[0]!r}. A false "
        f"alarm consumed the once-per-process latch and the genuine violation "
        f"was never reported."
    )


def test_empty_string_authority_is_not_treated_as_a_level(caplog):
    """An empty ExecutionContext field must fall through to the runner name."""
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    ctx = types.SimpleNamespace(
        drl_authority_level="", _drl_authority_level="ACTIVE",
        config=types.SimpleNamespace(coinbase_routing_enabled=True),
    )
    with caplog.at_level(logging.CRITICAL):
        es._coinbase_check_iron_law_8(ctx, rp)
    assert _critical(caplog) == []


# ---------------------------------------------------------------------------
# it must OBSERVE, not block
# ---------------------------------------------------------------------------

def test_routing_decision_is_unchanged_by_a_violation(monkeypatch, caplog):
    """The whole point: a demoted DRL must not silently stop trading."""
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    with caplog.at_level(logging.CRITICAL):
        assert es._coinbase_routed(_ctx("SHADOW"), "SOL") is True
    assert _critical(caplog), "…but it must still be loudly reported"


def test_routed_still_fails_closed_when_the_flag_is_off(monkeypatch):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    assert es._coinbase_routed(_ctx("ACTIVE", routing_enabled=False), "SOL") is False


def test_unrouted_asset_still_goes_to_kraken(monkeypatch):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    assert es._coinbase_routed(_ctx("ACTIVE"), "BTC") is False


# ---------------------------------------------------------------------------
# [P155] fee-model mismatch reporting (Coinbase routed, Kraken fees charged)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_fee_flag():
    es._CB_FEE_MODEL_WARNED = False
    yield
    es._CB_FEE_MODEL_WARNED = False


def _warnings(caplog):
    return [r.message for r in caplog.records if r.levelno == logging.WARNING]


def test_routed_asset_reports_the_fee_mismatch(monkeypatch, caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    with caplog.at_level(logging.WARNING):
        es._coinbase_routed(_ctx("ACTIVE"), "SOL")
    msgs = _warnings(caplog)
    assert any("FEE-MODEL-MISMATCH" in m for m in msgs), msgs
    assert any("ZERO_EXPOSURE" in m for m in msgs)


def test_kraken_asset_does_not_report_a_coinbase_fee_mismatch(monkeypatch, caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    with caplog.at_level(logging.WARNING):
        es._coinbase_routed(_ctx("ACTIVE"), "BTC")
    assert not any("FEE-MODEL-MISMATCH" in m for m in _warnings(caplog))


def test_fee_mismatch_reported_once_not_every_tick(monkeypatch, caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            es._coinbase_routed(_ctx("ACTIVE"), "SOL")
    assert len([m for m in _warnings(caplog) if "FEE-MODEL-MISMATCH" in m]) == 1


def test_fee_warning_falls_back_when_friction_is_unreachable(caplog):
    """ctx has no .engine — the warning must still name a magnitude, not crash."""
    with caplog.at_level(logging.WARNING):
        es._coinbase_fee_model_warning(_ctx("ACTIVE"))
    assert any("26.0bps" in m for m in _warnings(caplog))


def test_fee_warning_reports_the_live_configured_fee(caplog):
    ctx = _ctx("ACTIVE")
    ctx.engine = types.SimpleNamespace(
        guarantees=types.SimpleNamespace(
            alpha_calculator=types.SimpleNamespace(
                FRICTION=types.SimpleNamespace(taker_fee_bps=10.0, maker_fee_bps=4.0))))
    with caplog.at_level(logging.WARNING):
        es._coinbase_fee_model_warning(ctx)
    msgs = _warnings(caplog)
    assert any("taker=10.0bps" in m for m in msgs), msgs
    assert any("over-charged ~7bps taker" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# [P155e] venue-aware fee constants + the default-OFF contract
# ---------------------------------------------------------------------------

def test_coinbase_fee_constants_are_cheaper_than_kraken():
    """If these ever invert, the venue-aware path would TIGHTEN the gate
    instead of loosening it, and the default-OFF rationale would be wrong."""
    assert es._COINBASE_TAKER_BPS < 26.0
    assert es._COINBASE_MAKER_BPS < 16.0


def test_venue_aware_fees_default_to_off(tmp_path):
    """Enabling this loosens a risk gate, so it must never default on. A config
    JSON written before the flag existed must also still load, and load OFF."""
    import json as _json
    from main import ProductionConfig

    assert ProductionConfig().coinbase_venue_aware_fees is False

    legacy = tmp_path / "legacy.json"
    legacy.write_text(_json.dumps({"coinbase_routing_enabled": True}))
    assert ProductionConfig.from_file(legacy).coinbase_venue_aware_fees is False

    enabled = tmp_path / "enabled.json"
    enabled.write_text(_json.dumps({"coinbase_venue_aware_fees": True}))
    assert ProductionConfig.from_file(enabled).coinbase_venue_aware_fees is True

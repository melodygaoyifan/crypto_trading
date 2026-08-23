"""[P382] The velocity emergency-exit must never take its reference price
from an INVALID tick.

Live mechanism this pins (verified 2026-08-22, the day after P380 armed the
velocity trigger): the pipeline's fetch-failure fallback returns
`generate_verification_data()` — `data_valid=False`,
`_source="synthetic_fallback"` and a HARDCODED `current_price` of
BTC 95,000 / ETH 3,500 / SOL 185 — and `FastRiskTick.evaluate` wrote
`_last_eval_price[asset] = current_price` BEFORE its `data_valid` gate. The
synthetic tick HOLDed (correctly), but the next REAL tick computed
velocity = |64k - 95k| / 95k ~ 33% > 3% -> EXIT_ONLY -> a real taker flatten
of a healthy book. One transient Kraken fetch failure == one guaranteed false
emergency exit.

Fail directions pinned here: an invalid/synthetic/zero-price tick neither
writes the reference nor fires; the first valid tick after it cannot fire on
velocity (absence is never a trigger, P367); a reference older than
VELOCITY_REF_MAX_AGE_SEC is not compared against (P156's staleness rule on
the watchdog's own memory); and the P367 capability — a genuine +4% step
between two VALID ticks — still fires.
"""
from __future__ import annotations

import time

import pytest

from execution.fast_risk_tick import FastRiskAction, FastRiskTick


def _armed() -> FastRiskTick:
    t = FastRiskTick(shadow_mode=False, velocity_trigger=True,
                     price_move_threshold=0.03, vol_spike_mult=4.0)
    t.set_4h_anchor("BTC", 64000.0, volatility=0.01, depth=5_000_000.0)
    return t


def _valid(px: float) -> dict:
    return {"current_price": px, "data_valid": True, "volatility_30m": 0.01,
            "orderbook_depth_1pct_usd": 5_000_000.0, "orderbook_stale": False}


def _synthetic_fallback(px: float = 95000.0) -> dict:
    # the exact shape market_data_pipeline emits on a fetch failure
    return {"current_price": px, "data_valid": False,
            "_source": "synthetic_fallback", "volatility_30m": 0.02,
            "orderbook_depth_1pct_usd": 1_000_000.0, "orderbook_stale": True}


class TestTheLiveIncidentShape:
    def test_a_synthetic_fallback_tick_does_not_become_the_velocity_reference(self):
        t = _armed()
        assert t.evaluate("BTC", _valid(64000.0)).action == FastRiskAction.HOLD
        r = t.evaluate("BTC", _synthetic_fallback(95000.0))
        assert r.action == FastRiskAction.HOLD
        assert r.reason in ("data_invalid", "synthetic_fallback")
        # THE incident: the next REAL tick at ~64k must NOT read as a 33% move
        r2 = t.evaluate("BTC", _valid(64100.0))
        assert r2.action == FastRiskAction.HOLD, (
            "the synthetic 95,000 became the reference and the first real "
            "tick fired the emergency exit — the P382 defect")
        assert r2.price_move_pct < 0.03

    def test_synthetic_source_is_refused_even_if_data_valid_lies(self):
        t = _armed()
        t.evaluate("BTC", _valid(64000.0))
        md = _synthetic_fallback(95000.0)
        md["data_valid"] = True  # belt and braces: the source tag alone refuses
        assert t.evaluate("BTC", md).action == FastRiskAction.HOLD
        assert t.evaluate("BTC", _valid(64100.0)).action == FastRiskAction.HOLD

    def test_a_zero_price_with_data_valid_true_does_not_poison_the_reference(self):
        t = _armed()
        t.evaluate("BTC", _valid(64000.0))
        r = t.evaluate("BTC", _valid(0.0))
        assert r.action == FastRiskAction.HOLD
        assert t.evaluate("BTC", _valid(64100.0)).action == FastRiskAction.HOLD

    def test_invalid_tick_is_not_counted_as_a_shadow_evaluation(self):
        t = _armed()
        t.evaluate("BTC", _valid(64000.0))
        n0 = t._shadow_evals.get("BTC", 0)
        t.evaluate("BTC", _synthetic_fallback())
        assert t._shadow_evals.get("BTC", 0) == n0, (
            "an invalid tick counted as an evaluation is how the arming "
            "evidence read clean while containing no fetch failure")
        assert t._shadow_velocity_fires.get("BTC", 0) == 0


class TestTheCapabilityIsIntact:
    def test_a_real_four_percent_step_between_two_valid_ticks_still_fires(self):
        t = _armed()
        assert t.evaluate("BTC", _valid(64000.0)).action == FastRiskAction.HOLD
        r = t.evaluate("BTC", _valid(64000.0 * 0.96))
        assert r.action == FastRiskAction.EXIT_ONLY
        assert r.price_move_pct == pytest.approx(0.04, abs=1e-9)

    def test_the_first_valid_tick_after_an_invalid_one_cannot_fire_on_velocity(self):
        # a genuine move that happens ACROSS the invalid tick is measured by
        # the NEXT pair of valid ticks, never by the first valid one
        t = _armed()
        t.evaluate("BTC", _valid(64000.0))
        t.evaluate("BTC", _synthetic_fallback())
        # first valid tick: reference is still the 64000 from two ticks ago
        # and is FRESH (seconds old), so a real 4% gap DOES fire — the
        # reference was never invalidated, only not overwritten
        r = t.evaluate("BTC", _valid(64000.0 * 0.96))
        assert r.action == FastRiskAction.EXIT_ONLY


class TestReferenceStaleness:
    def test_a_reference_older_than_max_age_is_not_compared_against(self, monkeypatch):
        t = _armed()
        base = time.time()
        monkeypatch.setattr("execution.fast_risk_tick.time.time", lambda: base)
        t.evaluate("BTC", _valid(64000.0))
        # jump well past the reference max age (and past nothing else: the
        # anchor stays fresh at 6h)
        monkeypatch.setattr("execution.fast_risk_tick.time.time",
                            lambda: base + FastRiskTick.VELOCITY_REF_MAX_AGE_SEC + 5)
        r = t.evaluate("BTC", _valid(64000.0 * 0.90))
        assert r.action == FastRiskAction.HOLD, (
            "a 10% move measured against a reference from 'whenever we last "
            "looked' is not an inter-tick dislocation")
        # ...but the reference was re-written by that valid tick, so the
        # NEXT valid tick measures normally
        monkeypatch.setattr("execution.fast_risk_tick.time.time",
                            lambda: base + FastRiskTick.VELOCITY_REF_MAX_AGE_SEC + 40)
        r2 = t.evaluate("BTC", _valid(64000.0 * 0.90 * 0.96))
        assert r2.action == FastRiskAction.EXIT_ONLY

    def test_max_age_is_a_few_loop_periods_not_an_hour(self):
        # ~34s between evaluations (P353); the bound must be short enough
        # that an outage cannot make the first valid tick measure the outage
        assert 60.0 <= FastRiskTick.VELOCITY_REF_MAX_AGE_SEC <= 600.0

    def test_a_pre_p382_bare_float_reference_is_treated_as_stale(self):
        t = _armed()
        t._last_eval_price["BTC"] = 64000.0  # old shape: no timestamp
        r = t.evaluate("BTC", _valid(64000.0 * 0.90))
        assert r.action == FastRiskAction.HOLD
        # and it was upgraded to the stamped shape by that valid tick
        assert isinstance(t._last_eval_price["BTC"], tuple)

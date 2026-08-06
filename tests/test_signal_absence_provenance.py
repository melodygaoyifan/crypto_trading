"""[P170] Absence must not be representable as health or as opportunity.

Two guards in this system read a key that their producer never wrote, and both
defaulted to the reassuring answer:

  1. `agent_signals.get("quant_data_quality", 1.0)` (integration_v36.py) — the
     consumer half of P126's staleness guard, written 2026-04-27. The pipeline
     sets `quant_data_quality` in market_data on every path (setdefault 0.0 at
     market_data_pipeline.py:664, 1.0 at :1314 on Best-of-N success), but
     main.py's `agent_signals` literal never copied it across. The key was
     therefore always absent, the default always won, and the guard has never
     excluded a stale quant signal in its entire life. It did not fail — it
     could not fail, which reads identically from the logs.

  2. `agent_signals.get("signal_edge_bps", 50.0)` at three deadlock call sites.
     Here the key IS always present on the live path, so the 50.0 never fired.
     That is what makes it a trap rather than an active bug: it is a fabricated
     constant on a path nothing exercises, waiting for the first caller that
     builds agent_signals differently.

Under the pipeline's own calibration (`signal_edge_bps = abs(quant_dir) * 65`,
market_data_pipeline.py:1318, whose comment records avg |quant_dir| ~= 0.3) a
typical edge is ~19.5bps. The fabricated 50.0 corresponds to |quant_dir| = 0.77.
In TrancheAwareDeadlockResolver that difference inverts the resolution at T1
after two stuck bars against 15bps friction — 2.33x forces, 0.91x aborts — so
the fabricated number would have bought its way out of the system's own
patience. These tests pin both halves shut.
"""

import pytest

from integration.integration_v36 import (
    EDGE_ABSENT_BPS,
    resolve_signal_edge_bps,
)

# Mirrors defense/production_reliability.py TrancheAwareDeadlockResolver
MIN_EDGE_FOR_FORCE = 1.5
EDGE_DECAY_PER_BAR = 0.15


def _would_force(edge_bps, friction_bps=15.0, stuck_bars=2):
    """Reproduces the resolver's T1 force test."""
    decay = max(0.1, 1.0 - EDGE_DECAY_PER_BAR * stuck_bars)
    decayed = edge_bps * decay
    ratio = decayed / friction_bps if friction_bps > 0 else 0.0
    return ratio >= MIN_EDGE_FOR_FORCE


class TestEdgeResolutionPrefersRealValues:
    def test_agent_signals_wins(self):
        edge, src = resolve_signal_edge_bps({"signal_edge_bps": 19.5}, {"signal_edge_bps": 42.0})
        assert (edge, src) == (19.5, "agent_signals")

    def test_falls_through_to_market_data(self):
        edge, src = resolve_signal_edge_bps({}, {"signal_edge_bps": 42.0})
        assert (edge, src) == (42.0, "market_data")

    def test_zero_is_a_real_value_not_an_absence(self):
        # 0.0 means "flat signal" and must be reported as measured, because a
        # flat signal is a legitimate observation.
        edge, src = resolve_signal_edge_bps({"signal_edge_bps": 0.0}, {})
        assert edge == 0.0
        assert src == "agent_signals"

    def test_string_numbers_are_accepted(self):
        assert resolve_signal_edge_bps({"signal_edge_bps": "19.5"}, {})[0] == 19.5

    def test_negative_edge_is_clamped_not_rejected(self):
        # A negative edge is nonsense but is still a signal that the producer
        # ran; clamp rather than pretend nobody spoke.
        edge, src = resolve_signal_edge_bps({"signal_edge_bps": -20.0}, {})
        assert edge == 0.0
        assert src == "agent_signals"


class TestAbsenceIsNamed:
    @pytest.mark.parametrize("agent,market", [({}, {}), ({}, None), (None, None), (None, {})])
    def test_missing_everywhere_is_absent(self, agent, market):
        edge, src = resolve_signal_edge_bps(agent, market)
        assert src == "ABSENT"
        assert edge == EDGE_ABSENT_BPS

    def test_absent_is_distinguishable_from_zero(self):
        # The whole point: "no edge" and "no data" must not be the same answer.
        zero_edge, zero_src = resolve_signal_edge_bps({"signal_edge_bps": 0.0}, {})
        absent_edge, absent_src = resolve_signal_edge_bps({}, {})
        assert zero_edge == absent_edge  # same number...
        assert zero_src != absent_src    # ...different provenance

    def test_none_value_is_treated_as_missing(self):
        assert resolve_signal_edge_bps({"signal_edge_bps": None}, {})[1] == "ABSENT"

    @pytest.mark.parametrize("bad", ["abc", object(), [], {}])
    def test_malformed_values_do_not_resolve(self, bad):
        assert resolve_signal_edge_bps({"signal_edge_bps": bad}, {})[1] == "ABSENT"

    def test_nan_does_not_resolve(self):
        assert resolve_signal_edge_bps({"signal_edge_bps": float("nan")}, {})[1] == "ABSENT"

    def test_malformed_agent_signals_falls_through_to_market_data(self):
        edge, src = resolve_signal_edge_bps({"signal_edge_bps": "abc"}, {"signal_edge_bps": 19.5})
        assert (edge, src) == (19.5, "market_data")

    def test_source_is_always_one_of_three(self):
        for a in ({}, {"signal_edge_bps": 1.0}, {"signal_edge_bps": "x"}):
            for m in ({}, {"signal_edge_bps": 2.0}):
                assert resolve_signal_edge_bps(a, m)[1] in (
                    "agent_signals", "market_data", "ABSENT"
                )


class TestAbsenceDeclinesToForce:
    """The behavioural point of the fix."""

    def test_absent_edge_cannot_force(self):
        edge, _ = resolve_signal_edge_bps({}, {})
        assert not _would_force(edge)

    def test_the_old_default_would_have_forced(self):
        # Documents what was at stake: the fabricated constant crosses the bar.
        assert _would_force(50.0)

    def test_a_typical_real_edge_does_not_force(self):
        # avg |quant_dir| ~= 0.3 -> 0.3 * 65 = 19.5bps.
        assert not _would_force(19.5)

    def test_the_fabricated_default_inverted_the_decision(self):
        # Same friction, same stuck bars, opposite outcome. This is the finding.
        assert _would_force(50.0) and not _would_force(19.5)

    def test_force_requires_a_genuinely_strong_signal(self):
        # Break-even: decayed edge must reach 1.5 * 15 = 22.5, so edge >= 32.14,
        # i.e. |quant_dir| >= 0.494 under the x65 calibration.
        assert not _would_force(32.0)
        assert _would_force(32.2)
        assert not _would_force(0.49 * 65)
        assert _would_force(0.50 * 65)

    def test_absence_is_never_more_permissive_than_any_real_edge(self):
        edge, _ = resolve_signal_edge_bps({}, {})
        for real in (0.0, 5.0, 19.5, 50.0, 200.0):
            assert edge <= real or real < 0


class TestQuantDataQualityFailsClosed:
    """P126's guard could not fire. Now absence is its own case."""

    def _fuse_dq(self, agent_signals):
        """Mirrors the resolution block in integration_v36.decide()."""
        raw = agent_signals.get("quant_data_quality", None)
        if raw is None:
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def test_absent_quality_is_not_healthy(self):
        # The old default was 1.0. That is the bug.
        assert self._fuse_dq({}) == 0.0
        assert self._fuse_dq({}) < 0.5  # -> quant excluded from fusion

    def test_present_healthy_value_passes(self):
        assert self._fuse_dq({"quant_data_quality": 1.0}) == 1.0

    def test_present_failure_value_is_honoured(self):
        assert self._fuse_dq({"quant_data_quality": 0.0}) == 0.0

    @pytest.mark.parametrize("bad", [None, "abc", object()])
    def test_unusable_quality_fails_closed(self, bad):
        assert self._fuse_dq({"quant_data_quality": bad}) < 0.5

    def test_producer_now_emits_the_key(self):
        # The other half of the fix: main.py's agent_signals literal must copy
        # quant_data_quality across, or the consumer is guarding nothing.
        import io
        src = io.open("main.py", encoding="utf-8").read()
        assert '"quant_data_quality": market_data.get("quant_data_quality"' in src, (
            "main.py no longer propagates quant_data_quality into agent_signals; "
            "P126's guard is dead again"
        )

    def test_producer_default_is_the_failing_value(self):
        # If market_data somehow lacks it, the propagated value must be 0.0
        # (unverified), never 1.0 (healthy).
        import io
        src = io.open("main.py", encoding="utf-8").read()
        assert '"quant_data_quality": market_data.get("quant_data_quality", 0.0)' in src


class TestNoFabricatedDefaultsRemain:
    def test_fifty_bps_default_is_gone_from_call_sites(self):
        import io
        src = io.open("integration/integration_v36.py", encoding="utf-8").read()
        code = [
            ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#") and "signal_edge_bps" in ln
        ]
        offenders = [ln for ln in code if "50.0" in ln]
        assert not offenders, f"fabricated edge default resurfaced: {offenders}"

    def test_quant_data_quality_default_of_one_is_gone(self):
        import io
        src = io.open("integration/integration_v36.py", encoding="utf-8").read()
        assert 'agent_signals.get("quant_data_quality", 1.0)' not in src

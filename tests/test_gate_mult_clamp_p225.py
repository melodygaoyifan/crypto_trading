"""[P225] Terminal clamp on the composed alpha-gate multiplier + per-asset phase.

Two defects from the smart-beta live audit (2026-08-07):

1. Six writers stack multiplicatively into agent_signals["_regime_alpha_gate_mult"]
   (RegimeAggressor assigns, then SmartBeta / AlphaBoost / EXTERNAL-COMPOSITE /
   EC-ORPHAN each `*=`), each bounding only its OWN factor. The single
   consumption point — constitution.check_alpha_gate — multiplied the EV
   threshold by the raw product with NO terminal bound, on the path that
   produces vetoes and (post-P206) sleeve flattens. The size-path twin has had
   a [0.2, 2.0] clamp at main.py's WIRE-REGIME-SIZE for months; the gate path,
   the one that reaches live orders, had nothing. Live logs already show the
   composed SIZE mult breaching its module-level floor (0.59 < 0.70 on
   2026-08-08), proving composition escapes per-writer bounds in practice.

2. SmartBeta read agent_signals['phase'] before either phase writer ran in the
   tick, so on every asset after the first it saw the PREVIOUS ASSET's phase —
   cross-asset contamination on the exact input that gates the TREND_STRONG
   long-side loosening P173 newly armed. Fixed with a per-asset phase store;
   same-asset one-tick-stale beats cross-asset fresh (the P206 lesson).

The behavioral tests drive the real check_alpha_gate; the source guards pin
the wiring that a unit test cannot reach without constructing the 20k-line
runner (the P152 lesson: a fix that exists but is not wired is invisible to
unit tests of the fix alone).
"""

import logging
import math
import re
from pathlib import Path

import pytest

from defense.constitution import (
    AlphaThresholdCalculator,
    REGIME_GATE_MULT_MIN,
    REGIME_GATE_MULT_MAX,
)

REPO = Path(__file__).resolve().parents[1]


def _gate(mult: float, monkeypatch=None):
    calc = AlphaThresholdCalculator()
    return calc.check_alpha_gate(
        signal_strength=0.5,
        regime_confidence=0.6,
        mode="NORMAL",
        min_alpha_bps=0.0,  # keep the EV leg dominant so the mult is visible
        direction=1.0,
        regime_alpha_gate_mult=mult,
    )


class TestTheClampBinds:
    def test_runaway_high_is_clamped_to_max(self):
        at_max = _gate(REGIME_GATE_MULT_MAX)
        runaway = _gate(REGIME_GATE_MULT_MAX * 3)
        assert runaway.ev_threshold_bps == pytest.approx(at_max.ev_threshold_bps)

    def test_runaway_low_is_clamped_to_min(self):
        at_min = _gate(REGIME_GATE_MULT_MIN)
        runaway = _gate(REGIME_GATE_MULT_MIN / 5)
        assert runaway.ev_threshold_bps == pytest.approx(at_min.ev_threshold_bps)

    def test_clamp_actually_changes_the_outcome(self):
        """The clamp must be reachable: an unclamped 6x would differ from 2x."""
        neutral = _gate(1.0)
        runaway = _gate(6.0)
        # If the clamp were absent, runaway would be 6x neutral; clamped it is MAX x.
        assert runaway.ev_threshold_bps == pytest.approx(
            neutral.ev_threshold_bps * REGIME_GATE_MULT_MAX
        )
        assert runaway.ev_threshold_bps < neutral.ev_threshold_bps * 6 * 0.99

    def test_clamp_warns_with_the_p222_tag(self, caplog):
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            _gate(9.9)
        assert any("[P225-GATE-MULT]" in r.message for r in caplog.records)


class TestInRangeBehaviorIsUnchanged:
    """The clamp is a runaway backstop, not a retune — every composition
    observed live (0.99, 1.14, 1.19, 1.20, 1.386 worst-case) must pass through
    untouched."""

    @pytest.mark.parametrize("mult", [0.85, 0.99, 1.14, 1.19, 1.20, 1.386, 1.5])
    def test_observed_live_compositions_pass_through_exactly(self, mult):
        neutral = _gate(1.0)
        scaled = _gate(mult)
        assert scaled.ev_threshold_bps == pytest.approx(
            neutral.ev_threshold_bps * mult
        )

    def test_neutral_one_is_untouched_and_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            _gate(1.0)
        assert not any("[P225-GATE-MULT]" in r.message for r in caplog.records)

    def test_bounds_are_wider_than_any_single_writers_own_range(self):
        # SmartBeta clamps its own factor to [0.85, 1.20]; RegimeAggressor's
        # config values run 0.85-1.20; AlphaBoost is bounded similarly. The
        # terminal clamp must never bite a single well-behaved writer.
        assert REGIME_GATE_MULT_MIN < 0.85
        assert REGIME_GATE_MULT_MAX > 1.20


class TestNonFiniteFailsToNeutral:
    """A poisoned multiplier must become 1.0 — not a silent trading stop
    (inf -> threshold=inf -> permanent veto) and not a disabled gate (nan)."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_equals_neutral(self, bad):
        neutral = _gate(1.0)
        poisoned = _gate(bad)
        assert math.isfinite(poisoned.ev_threshold_bps)
        assert poisoned.ev_threshold_bps == pytest.approx(neutral.ev_threshold_bps)

    def test_non_finite_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="defense.constitution"):
            _gate(float("nan"))
        assert any(
            "[P225-GATE-MULT]" in r.message and "non-finite" in r.message
            for r in caplog.records
        )


class TestPhasePerAssetWiring:
    """Source guards for the main.py half (P152 lesson: assert the wiring)."""

    @pytest.fixture(scope="class")
    def main_src(self):
        return (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")

    def test_per_asset_phase_store_exists(self, main_src):
        assert "self._last_phase_by_asset[asset] = _phase_result" in main_src, (
            "P225 regression: the T22 phase-detector block no longer stores "
            "the per-asset phase; agent_signals['phase'] is back to reading "
            "the previous asset's engine-global value."
        )

    def test_phase_read_prefers_the_per_asset_store(self, main_src):
        assert re.search(
            r"getattr\(self,\s*'_last_phase_by_asset',\s*\{\}\)\.get\(asset\)",
            main_src,
        ), (
            "P225 regression: the v3.2-B5 phase write no longer prefers this "
            "asset's stored phase over the engine-global slot."
        )

    def test_read_site_precedes_write_site(self, main_src):
        """The whole point: the reader runs earlier in the tick than the
        writers, so it MUST go through the per-asset store. If the read site
        ever moves after the T22 block this guard (and the fix) can be
        retired."""
        read_at = main_src.find("getattr(self, '_last_phase_by_asset', {}).get(asset)")
        write_at = main_src.find("self._last_phase_by_asset[asset] = _phase_result")
        assert 0 < read_at < write_at


class TestDeadShellsStayDead:
    """[P225] engine/ and shadow/ were empty __init__ shells (real code in
    archive/) COPYed into both images. Deleting the dirs without the COPY
    lines breaks the build (the P192 shape) — pin both halves together."""

    def test_engine_and_shadow_dirs_are_gone(self):
        assert not (REPO / "engine").exists()
        assert not (REPO / "shadow").exists()

    @pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.engine"])
    def test_no_dockerfile_copies_a_deleted_dir(self, dockerfile):
        src = (REPO / dockerfile).read_text(encoding="utf-8", errors="replace")
        for line in src.splitlines():
            if line.strip().startswith("COPY"):
                assert " engine/ " not in line and " shadow/ " not in line, (
                    f"{dockerfile} COPYs a directory deleted in P225 — "
                    f"the image build will fail at this line: {line!r}"
                )

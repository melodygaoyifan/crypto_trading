"""[P311] Self-normalising cascade liquidation thresholds.

A single dollar threshold across assets was wrong in BOTH directions, and
the derivflow ledger measured it over 11 days: median 24h liquidations are
BTC $79.6M / ETH $37.8M / SOL $6.4M, so $10M/h is ~3x BTC's normal hourly
rate (it fired on 7% of ticks) and ~38x SOL's maximum EVER observed
($2.09M/h — structurally unreachable).

The threshold is now a multiple of the asset's OWN recent hourly rate. One
multiplier serves all three because the SHAPE of the burst distribution is
comparable once normalised — peak/baseline was BTC 8.1x, ETH 12.0x, SOL 7.9x
while the LEVELS differ by 12.5x — and that similarity is the finding that
made a dimensionless constant defensible.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The measured baselines (median 24h / 24), for readability in the tests.
BASELINE_1H = {"BTC": 79_629_742 / 24, "ETH": 37_840_157 / 24,
               "SOL": 6_366_840 / 24}


@pytest.fixture(autouse=True)
def _clean():
    from risk import cascade_exhaustion_governor as M
    M.reset_cascade_exhaustion_governor()
    yield
    M.reset_cascade_exhaustion_governor()


def _gov(asset=None):
    from risk.cascade_exhaustion_governor import get_cascade_exhaustion_governor
    return get_cascade_exhaustion_governor(asset=asset)


class TestTheResolver:
    def test_absent_baseline_falls_back_to_the_absolute_pair(self):
        """Exactly today's behaviour when no caller supplies a baseline."""
        g = _gov()
        assert g.effective_liq_thresholds(None) == (
            g.config.cascade_detect_liq_threshold,
            g.config.cascade_accelerate_threshold)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"),
                                     None, "x", object()])
    def test_an_unusable_baseline_never_produces_a_zero_threshold(self, bad):
        """A zero threshold makes DETECT fire on EVERY tick. Absence must
        never become the permissive value (P2)."""
        g = _gov()
        d, a = g.effective_liq_thresholds(bad)
        assert d == g.config.cascade_detect_liq_threshold
        assert a == g.config.cascade_accelerate_threshold
        assert d > 0 and a > 0

    def test_the_small_asset_becomes_able_to_fire_at_all(self):
        """SOL's maximum ever observed hourly rate is $2.09M. Under the old
        global $10M it could not fire in any market condition."""
        g = _gov()
        detect, _ = g.effective_liq_thresholds(BASELINE_1H["SOL"])
        assert detect < 2_087_611, (
            f"SOL detect ${detect:,.0f}/h still exceeds the largest burst "
            f"ever observed ($2,087,611/h) — the detector remains unreachable")
        assert detect > BASELINE_1H["SOL"] * 2, (
            "the threshold must still be well above SOL's normal rate, or "
            "DETECT becomes routine instead of exceptional")

    def test_the_large_asset_stops_firing_routinely(self):
        """BTC's p95 hourly rate is $10.06M — the old global threshold sat
        right on it, so DETECT fired on ~7% of ticks."""
        g = _gov()
        detect, _ = g.effective_liq_thresholds(BASELINE_1H["BTC"])
        assert detect > 10_491_944, (
            f"BTC detect ${detect:,.0f}/h is at or below its measured p90/p95 "
            f"band — DETECT would stay routine")

    def test_accelerate_stays_above_detect_on_every_asset(self):
        g = _gov()
        for a, b in BASELINE_1H.items():
            d, acc = g.effective_liq_thresholds(b)
            assert acc > d, f"{a}: accelerate {acc} <= detect {d}"

    def test_the_multiple_is_the_one_the_measurement_chose(self):
        """5.0 was picked by a rule stated before reading the table: the
        smallest multiple putting every asset at or below ~2.5% of ticks
        while leaving the smallest asset able to fire. A different value is a
        different calibration and needs its own replay."""
        g = _gov()
        assert g.config.cascade_detect_liq_multiple == 5.0
        assert g.config.cascade_accelerate_liq_multiple == 12.5


class TestTheStateMachine:
    def _feed(self, g, rate, baseline):
        g.update_metrics(liquidation_volume_1h=rate,
                         liquidation_volume_4h=rate * 4,
                         price_change_1h_pct=0.0, price_change_4h_pct=0.0,
                         volume_spike_ratio=1.0, baseline_liq_1h=baseline)

    def test_a_burst_detects_on_the_small_asset(self):
        from risk.cascade_exhaustion_governor import CascadePhase
        g = _gov("SOL")
        self._feed(g, BASELINE_1H["SOL"] * 6, BASELINE_1H["SOL"])
        assert g._phase != CascadePhase.NONE, (
            "a 6x burst on SOL did not detect — the threshold is still "
            "expressed in units SOL cannot reach")

    def test_a_normal_hour_does_not_detect_on_the_large_asset(self):
        from risk.cascade_exhaustion_governor import CascadePhase
        g = _gov("BTC")
        self._feed(g, BASELINE_1H["BTC"] * 1.2, BASELINE_1H["BTC"])
        assert g._phase == CascadePhase.NONE, (
            "BTC detected on a rate 1.2x its own normal — DETECT is routine")

    def test_intensity_uses_the_same_normalised_denominator(self):
        """Otherwise a small-cap asset reads ~0 intensity during its own
        cascade, because the numerator is its dollars and the denominator is
        a market-wide constant."""
        g = _gov("SOL")
        self._feed(g, BASELINE_1H["SOL"] * 6, BASELINE_1H["SOL"])
        assert g._current_metrics.cascade_intensity > 0.5, (
            f"intensity {g._current_metrics.cascade_intensity:.3f} — the "
            f"denominator is not normalised to the asset")

    def test_thresholds_are_available_before_the_first_update(self):
        """P85: the transition block reads these attributes; a governor
        restored from state or freshly built must not AttributeError."""
        g = _gov("ETH")
        assert g._detect_thr > 0 and g._accel_thr > 0


class TestTheDecision:
    def _live(self):
        return json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(
                encoding="utf-8"))

    def test_the_window_is_armed_with_its_evidence_recorded(self):
        cfg = self._live()
        assert cfg.get("cascade_real_liquidation_window") is True
        note = cfg.get("_p311_cascade_note", "")
        for token in ("5x", "REVERT", "2.4%"):
            assert token in note, f"the arming note lost {token!r}"

    def test_the_caller_supplies_the_baseline_before_it_is_read(self):
        """The first draft defined `_liq_base_1h` AFTER the log line that
        reads it — a NameError inside a swallowing try, i.e. an instrument
        that silently stops reporting (P234 ordering class)."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        defn = src.index("_liq_base_1h = (_liq_vol_24h / 24.0)")
        for reader in ("effective_liq_thresholds(_liq_base_1h)",
                       "baseline_liq_1h=_liq_base_1h"):
            assert src.index(reader) > defn, f"{reader} reads it too early"

    def test_the_observe_log_reports_the_threshold_in_force(self):
        """P300: an instrument that prints a constant while the real
        threshold is computed elsewhere cannot be used to judge the arming."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        # anchor on the f-string, not the first mention: the
        # config dataclass documents the tag in a comment.
        i = src.index('f"[CASCADE-OBSERVE] ')
        blk = src[i:i + 1400]
        assert "effective_liq_thresholds" in blk
        assert "$10,000,000" not in blk, (
            "the log still prints the retired global constant")

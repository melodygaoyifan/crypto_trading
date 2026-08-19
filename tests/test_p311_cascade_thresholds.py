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
        """[P316] Expressed as a RATIO, which is the invariant this constant
        exists to hold. The old form pinned "< $2,087,611/h" — SOL's largest
        burst on 11 days of the derivflow estimator — and that literal rotted
        the moment the calibration moved to a 6-month basis, where SOL's
        largest observed spike is 51.9x its own baseline. What must stay true
        is that the multiple sits BELOW what the smallest asset has actually
        produced (so it is reachable) and well ABOVE 1x (so it is not
        routine)."""
        g = _gov()
        m = g.config.cascade_detect_liq_multiple
        assert m < 51.9, (
            f"detect multiple {m}x exceeds the largest spike SOL has produced "
            f"in six months (51.9x) — the detector is unreachable there")
        assert m > 5.0, (
            "at 5x the measured firing rate is 11-14% of bars, which is "
            "routine, not exceptional")
        detect, _ = g.effective_liq_thresholds(BASELINE_1H["SOL"])
        assert detect == BASELINE_1H["SOL"] * m

    def test_the_large_asset_stops_firing_routinely(self):
        """[P316] BTC's p99 spike is 26.8x its own baseline on six months of
        4H liquidation history; a multiple at or below its p95 (10.2x) is
        routine. Pinned in ratio terms for the same reason as above."""
        g = _gov()
        m = g.config.cascade_detect_liq_multiple
        assert m > 10.2, (
            f"detect multiple {m}x is at or below BTC's p95 spike (10.2x) — "
            f"DETECT would fire on >5% of bars")

    def test_accelerate_stays_above_detect_on_every_asset(self):
        g = _gov()
        for a, b in BASELINE_1H.items():
            d, acc = g.effective_liq_thresholds(b)
            assert acc > d, f"{a}: accelerate {acc} <= detect {d}"

    def test_the_multiple_is_the_one_the_measurement_chose(self):
        """[P316] 20.0, calibrated on 1,039 bars per asset rather than 85.
        At 20x the measured firing rates are BTC 2.1% / ETH 1.0% / SOL 1.5%;
        at the P311 value of 5.0 they are 13.7 / 10.7 / 11.1%. A different
        value is a different calibration and needs its own replay."""
        g = _gov()
        assert g.config.cascade_detect_liq_multiple == 20.0
        assert g.config.cascade_accelerate_liq_multiple == 50.0


class TestTheStateMachine:
    def _feed(self, g, rate, baseline):
        g.update_metrics(liquidation_volume_1h=rate,
                         liquidation_volume_4h=rate * 4,
                         price_change_1h_pct=0.0, price_change_4h_pct=0.0,
                         volume_spike_ratio=1.0, baseline_liq_1h=baseline)

    def test_a_burst_detects_on_the_small_asset(self):
        from risk.cascade_exhaustion_governor import CascadePhase
        g = _gov("SOL")
        # 25x: above the 20x gate and well inside what SOL has produced
        # (51.9x max over six months)
        self._feed(g, BASELINE_1H["SOL"] * 25, BASELINE_1H["SOL"])
        assert g._phase != CascadePhase.NONE, (
            "a 25x burst on SOL did not detect — the threshold is still "
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
        self._feed(g, BASELINE_1H["SOL"] * 60, BASELINE_1H["SOL"])
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

    def test_the_window_decision_is_recorded_either_way(self):
        """[P316 supersedes P311's ARMED assertion.] P311 armed this on 85
        observations per asset; six months of 4H liquidation history says
        that calibration understated the firing rate ~6x and that a spike at
        the armed level carries no forward information. It is disarmed, and
        what is pinned is that the decision — in whichever direction — is
        written down beside the flag."""
        cfg = self._live()
        armed = cfg.get("cascade_real_liquidation_window")
        note = (cfg.get("_p316_cascade_disarmed_note", "")
                or cfg.get("_p311_cascade_note", ""))
        assert note, "the window's state changed with no recorded reason"
        if armed:
            assert "PRECONDITION" not in note, (
                "armed while carrying the note that explains why it is off")

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

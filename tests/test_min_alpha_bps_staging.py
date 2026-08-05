"""
tests/test_min_alpha_bps_staging.py - P1-2A MIN_ALPHA_BPS Staging tests

Verifies:
1. HIGH_RISK default: no min_alpha floor -> pure EV gate
2. ULTRA staged schedule: WEEK1=10, WEEK2_3=8, MONTH1=5
3. Gate behaviour: min_alpha vs EV gate interaction
4. Friction breakdown populated in AlphaGatingResult
5. Config round-trip from JSON
6. Gate decision labels (ALLOW / REJECT_MIN_ALPHA / REJECT_EV)
"""

import json
import pytest
from pathlib import Path

from integration.integration_v36 import TradeIntentV36
from defense.constitution import (
    AlphaThresholdCalculator,
    AlphaGatingResult,
    FrictionComponents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calculator(monthly_volume_usd: float = 0.0) -> AlphaThresholdCalculator:
    """Create a calculator with specified volume tier."""
    calc = AlphaThresholdCalculator()
    calc.FRICTION.update_for_asset("")
    calc.FRICTION.update_for_volume(monthly_volume_usd)
    return calc


def _resolve_min_alpha_bps(alpha_gate_config: dict) -> float:
    """Resolve min_alpha_bps from config, matching main.py logic."""
    stage_selector = alpha_gate_config.get("stage_selector", "")
    default = alpha_gate_config.get("min_alpha_bps_default", 0.0)
    result = default
    for entry in alpha_gate_config.get("staged_schedule", []):
        if entry.get("stage") == stage_selector:
            result = entry.get("min_alpha_bps", default)
            break
    return result


# ===========================================================================
# Test Group 1: HIGH_RISK default - no min_alpha floor
# ===========================================================================

class TestHighRiskDefault:
    """HIGH_RISK: no alpha_gate config -> min_alpha_bps=0, pure EV gate."""

    def test_no_min_alpha_pure_ev_gate(self):
        """With min_alpha_bps=0, threshold is purely friction * multiplier."""
        calc = _make_calculator(monthly_volume_usd=0.0)  # Free tier
        result = calc.check_alpha_gate(
            signal_strength=0.5,
            regime_confidence=0.8,
            mode="NORMAL",
            min_alpha_bps=0.0,
        )
        # Free tier: fees=0, slippage=5, latency=2 -> friction=7
        # Current NORMAL multiplier=1.10 (was 1.5) -> threshold=7.7
        assert result.threshold_bps == pytest.approx(7.7, abs=0.1)
        assert result.ev_threshold_bps == pytest.approx(7.7, abs=0.1)
        assert result.min_alpha_bps_used == 0.0

    def test_high_risk_default_threshold_matches_legacy(self):
        """Default behaviour (no min_alpha) matches pre-P1-2A thresholds."""
        calc = _make_calculator(monthly_volume_usd=0.0)

        # NORMAL mode free tier
        result_normal = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=0.0)
        assert result_normal.threshold_bps == pytest.approx(7.7, abs=0.1)

        # OPPORTUNITY mode free tier
        result_opp = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=0.0)
        assert result_opp.threshold_bps == pytest.approx(7.0, abs=0.1)

    def test_leverage_margin_friction_increases_ev_threshold(self):
        calc = _make_calculator(monthly_volume_usd=0.0)
        calc.FRICTION.update_for_leverage("BTC", 2.0)

        result = calc.check_alpha_gate(
            signal_strength=0.5,
            regime_confidence=0.8,
            mode="NORMAL",
            min_alpha_bps=0.0,
            asset="BTC",
        )

        assert result.friction_fee_bps == 0.0
        assert result.friction_slippage_bps == pytest.approx(3.0)
        assert result.friction_latency_bps == pytest.approx(2.0)
        assert result.friction_margin_bps == pytest.approx(7.0)
        assert result.friction_total_bps == pytest.approx(12.0)
        assert result.ev_threshold_bps == pytest.approx(13.2, abs=0.1)  # 12.0 * 1.10
        assert result.threshold_bps == pytest.approx(13.2, abs=0.1)


# ===========================================================================
# Test Group 2: ULTRA staged schedule resolution
# ===========================================================================

class TestUltraStagedSchedule:
    """Staged schedule resolves correct min_alpha_bps per stage."""

    ULTRA_CONFIG = {
        "min_alpha_bps_default": 5,
        "staged_schedule": [
            {"stage": "WEEK1", "min_alpha_bps": 5},
            {"stage": "WEEK2_3", "min_alpha_bps": 4},
            {"stage": "MONTH1", "min_alpha_bps": 3},
        ],
        "stage_selector": "WEEK1",
    }

    def test_week1_resolves_to_5(self):
        cfg = {**self.ULTRA_CONFIG, "stage_selector": "WEEK1"}
        assert _resolve_min_alpha_bps(cfg) == 5

    def test_week2_3_resolves_to_4(self):
        cfg = {**self.ULTRA_CONFIG, "stage_selector": "WEEK2_3"}
        assert _resolve_min_alpha_bps(cfg) == 4

    def test_month1_resolves_to_3(self):
        cfg = {**self.ULTRA_CONFIG, "stage_selector": "MONTH1"}
        assert _resolve_min_alpha_bps(cfg) == 3

    def test_unknown_stage_falls_back_to_default(self):
        cfg = {**self.ULTRA_CONFIG, "stage_selector": "UNKNOWN_STAGE"}
        assert _resolve_min_alpha_bps(cfg) == 5  # min_alpha_bps_default

    def test_empty_stage_selector_falls_back(self):
        cfg = {**self.ULTRA_CONFIG, "stage_selector": ""}
        assert _resolve_min_alpha_bps(cfg) == 5  # default (updated)


# ===========================================================================
# Test Group 3: Gate behaviour - min_alpha vs EV gate interaction
# ===========================================================================

class TestGateBehaviour:
    """
    Two-layer gate: effective_threshold = max(min_alpha_bps, friction * multiplier).
    min_alpha never overrides a stricter EV gate.
    """

    def test_min_alpha_dominates_in_opportunity_free_tier(self):
        """
        Free tier OPPORTUNITY: EV gate = 7 * 1.0 = 7.0bps.
        min_alpha=10 > 7.0 -> threshold=10.
        """
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=10.0)
        assert result.threshold_bps == pytest.approx(10.0, abs=0.1)
        assert result.ev_threshold_bps == pytest.approx(7.0, abs=0.1)

    def test_ev_gate_dominates_in_normal_free_tier(self):
        """
        Free tier NORMAL: EV gate = 7 * 1.5 = 10.5bps.
        min_alpha=10 < 10.5 -> threshold=10.5.
        """
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=6.0)
        assert result.threshold_bps == pytest.approx(7.7, abs=0.1)
        assert result.min_alpha_bps_used == 6.0

    def test_ev_gate_always_dominates_post_free_tier(self):
        """
        Post-free tier: EV gate = 33 * 1.5 = 49.5bps.
        Even min_alpha=5 -> threshold=49.5 (EV dominates).
        """
        calc = _make_calculator(monthly_volume_usd=50_000.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=5.0)
        assert result.threshold_bps == pytest.approx(36.3, abs=0.1)
        assert result.ev_threshold_bps == pytest.approx(36.3, abs=0.1)
        assert result.min_alpha_bps_used == 5.0

    def test_alpha_7bps_rejected_by_week1_allowed_by_month1(self):
        """
        alpha≈7bps: WEEK1 (min=10) -> REJECT. MONTH1 (min=5) in OPP mode -> depends on EV.
        """
        calc = _make_calculator(monthly_volume_usd=0.0)

        # Carefully craft signal_strength to produce ~7bps alpha
        # alpha = |signal| * 200 * regime_conf * perf_factor
        # perf_factor = 0.5 + 0.5*0.5 = 0.75 (default hit_rate=0.5)
        # We want alpha = 7, so: signal * 200 * 0.8 * 0.75 = 7
        # signal = 7 / (200 * 0.8 * 0.75) = 7/120 ≈ 0.0583
        signal = 7.0 / (200.0 * 0.8 * 0.75)

        # WEEK1 min_alpha=10, OPPORTUNITY EV=7.0 -> threshold=max(10, 7.0)=10
        result_w1 = calc.check_alpha_gate(
            signal_strength=signal, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=10.0)
        assert result_w1.passes_threshold is False
        assert result_w1.gate_decision == "REJECT_MIN_ALPHA"

        # MONTH1 min_alpha=5, OPPORTUNITY EV=7.0 -> threshold=max(5, 7.0)=7.0
        result_m1 = calc.check_alpha_gate(
            signal_strength=signal, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=5.0)
        assert result_m1.passes_threshold is False
        assert result_m1.gate_decision == "REJECT_EV"

    def test_alpha_9bps_pass_month1_reject_week1(self):
        """
        alpha≈9bps in OPPORTUNITY mode:
        WEEK1 (min=10, EV=7.0 -> threshold=10) -> REJECT (min_alpha)
        MONTH1 (min=5, EV=7.0 -> threshold=7.0) -> ALLOW
        """
        calc = _make_calculator(monthly_volume_usd=0.0)
        signal = 9.0 / (200.0 * 0.59 * 0.75)

        result_w1 = calc.check_alpha_gate(
            signal_strength=signal, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=10.0)
        assert result_w1.passes_threshold is False

        result_m1 = calc.check_alpha_gate(
            signal_strength=signal, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=5.0)
        assert result_m1.passes_threshold is True
        assert result_m1.gate_decision == "ALLOW"

    def test_quiet_accum_short_uses_short_specific_direction_floor(self):
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.10,
            regime_confidence=0.8,
            mode="OPPORTUNITY",
            min_alpha_bps=5.0,
            direction=-0.10,
            regime="QUIET_ACCUMULATION",
            quiet_accum_min_direction=0.15,
            quiet_accum_min_direction_short=0.05,
        )
        assert result.gate_decision == "ALLOW"
        assert result.passes_threshold is True

    def test_quiet_accum_long_still_uses_general_direction_floor(self):
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.10,
            regime_confidence=0.8,
            mode="OPPORTUNITY",
            min_alpha_bps=5.0,
            direction=0.10,
            regime="QUIET_ACCUMULATION",
            quiet_accum_min_direction=0.15,
            quiet_accum_min_direction_short=0.05,
        )
        assert result.gate_decision == "REJECT_LOW_DIRECTION_QUIET"
        assert result.passes_threshold is False

    def test_short_specific_high_risk_floor_can_pass_where_general_floor_fails(self):
        calc = _make_calculator(monthly_volume_usd=0.0)
        signal = 0.0435  # Estimated alpha ~= 4.2bps with default confidence path

        result_general = calc.check_alpha_gate(
            signal_strength=signal,
            regime_confidence=0.9862530385122186,
            mode="OPPORTUNITY",
            min_alpha_bps=5.0,
            direction=-0.04716423849782937,
            asset="BTC",
            regime="WEAK_CONSOLIDATION",
            regime_alpha_gate_mult=1.08,
        )
        assert result_general.gate_decision == "REJECT_MIN_ALPHA"
        assert result_general.passes_threshold is False

        result_short = calc.check_alpha_gate(
            signal_strength=signal,
            regime_confidence=0.9862530385122186,
            mode="OPPORTUNITY",
            min_alpha_bps=3.0,
            direction=-0.04716423849782937,
            asset="BTC",
            regime="WEAK_CONSOLIDATION",
            regime_alpha_gate_mult=0.96,
        )
        assert result_short.gate_decision == "ALLOW"
        assert result_short.passes_threshold is True

def _find_borderline_signal(calc, long: bool, **gate_kwargs):
    """[P165] Locate a signal the EV gate genuinely rejects, and by how much.

    The two epsilon tests below hardcoded a signal (5.90/75 and 6.40/75) chosen
    when the EV multiplier was 1.25x. It was later lowered to 1.10x, so both
    signals now clear the bar unaided and the gate returns a plain "ALLOW" — the
    tests asserted `ALLOW_EPSILON` on inputs that no longer need an epsilon, i.e.
    they had stopped exercising the epsilon path at all while still reading like
    they covered it. Same shape as P158/P159: a check that cannot fail.

    Search for the borderline instead of pinning it. Returns
    `(signal, shortfall_bps)` for the first signal that rejects on EV.
    """
    for step in range(1, 400):
        signal = (step * 0.05) / 75.0
        res = calc.check_alpha_gate(
            signal_strength=signal,
            direction=signal if long else -signal,
            **gate_kwargs,
        )
        if res.gate_decision == "REJECT_EV":
            shortfall = res.required_net_alpha_bps - res.expected_net_alpha_bps
            if shortfall > 0:
                return signal, shortfall
    raise AssertionError("no EV-rejecting signal found in the scanned range")


def test_short_borderline_epsilon_allows_near_ev_threshold():
    calc = _make_calculator(monthly_volume_usd=0.0)
    base = dict(regime_confidence=0.5, quant_confidence=0.5, mode="NORMAL",
                min_alpha_bps=3.0, asset="BTC")

    signal, shortfall = _find_borderline_signal(calc, long=False, **base)

    rejected = calc.check_alpha_gate(signal_strength=signal, direction=-signal, **base)
    assert rejected.gate_decision == "REJECT_EV"
    assert rejected.passes_threshold is False

    # An epsilon just larger than the shortfall must convert it, and must report
    # ALLOW_EPSILON rather than a plain ALLOW — that distinction is what makes an
    # epsilon-carried trade auditable after the fact.
    allowed = calc.check_alpha_gate(
        signal_strength=signal, direction=-signal,
        borderline_pass_epsilon_bps_short=shortfall + 0.01, **base,
    )
    assert allowed.gate_decision == "ALLOW_EPSILON"
    assert allowed.passes_threshold is True

    # An epsilon smaller than the shortfall must NOT rescue it.
    still_rejected = calc.check_alpha_gate(
        signal_strength=signal, direction=-signal,
        borderline_pass_epsilon_bps_short=shortfall * 0.5, **base,
    )
    assert still_rejected.passes_threshold is False


def test_long_borderline_epsilon_allows_near_ev_threshold():
    calc = _make_calculator(monthly_volume_usd=0.0)
    base = dict(regime_confidence=0.5, quant_confidence=0.5, mode="NORMAL",
                min_alpha_bps=3.0, asset="BTC", regime="QUIET_ACCUMULATION",
                quiet_accum_min_direction=0.0, regime_alpha_gate_mult=0.90)

    signal, shortfall = _find_borderline_signal(calc, long=True, **base)

    rejected = calc.check_alpha_gate(signal_strength=signal, direction=signal, **base)
    assert rejected.gate_decision == "REJECT_EV"
    assert rejected.passes_threshold is False

    allowed = calc.check_alpha_gate(
        signal_strength=signal, direction=signal,
        borderline_pass_epsilon_bps=shortfall + 0.01, **base,
    )
    assert allowed.gate_decision == "ALLOW_EPSILON"
    assert allowed.passes_threshold is True

    still_rejected = calc.check_alpha_gate(
        signal_strength=signal, direction=signal,
        borderline_pass_epsilon_bps=shortfall * 0.5, **base,
    )
    assert still_rejected.passes_threshold is False


def test_opportunity_actionable_min_exposure_can_admit_small_eth_short():
    """The per-asset exposure override is what admits a small ETH short.

    [P165] This built an intent with `direction=-0.018` against its own
    `opportunity_actionable_direction_threshold_short=0.04`. `is_actionable` is
    a conjunction (`integration_v36.py:268-272`), so the direction clause was
    always false and the test could not pass on any exposure setting — it was
    self-contradictory, not merely stale. Give the direction enough magnitude to
    clear its threshold so the test isolates the exposure clause it is named
    for, and pin the negative case that proves the override is load-bearing.
    """
    kwargs = dict(
        asset="ETH",
        direction=-0.05,   # clears the 0.04 short threshold
        target_exposure=0.006,
        system_mode="OPPORTUNITY",
        alpha_gate_passed=True,
        opportunity_actionable_direction_threshold_short=0.04,
    )

    assert TradeIntentV36(
        **kwargs, opportunity_actionable_exposure_threshold=0.005
    ).is_actionable is True

    # Without the override the default 0.01 floor blocks the same intent —
    # otherwise this test would pass regardless of the override.
    assert TradeIntentV36(**kwargs).is_actionable is False


def test_high_risk_config_contains_btc_specific_short_borderline_epsilon():
    cfg = json.loads(Path("configs/high_risk.json").read_text(encoding="utf-8"))
    live = json.loads(Path("configs/live_high_risk.json").read_text(encoding="utf-8"))
    ag = cfg["alpha_gate"]
    assert ag["quiet_accum_min_direction_by_asset"]["BTC"] == pytest.approx(0.04)
    assert ag["quiet_accum_min_direction_by_asset"]["ETH"] == pytest.approx(0.025)
    assert ag["quiet_accum_min_direction_short_by_asset"]["ETH"] == pytest.approx(0.05)
    assert ag["opportunity_actionable_min_direction_short_by_asset"]["ETH"] == pytest.approx(0.04)
    assert ag["opportunity_actionable_min_exposure_by_asset"]["ETH"] == pytest.approx(0.005)
    # [P165] BTC's long-side epsilon was pinned at 1.0 and has been retuned to
    # 2.5. An epsilon is a *tolerance* on the EV gate, so the property worth
    # guarding is not its exact size but that it stays small enough not to
    # swallow the gate it relaxes — otherwise "borderline pass" becomes "always
    # pass" and the alpha gate (non-negotiable rule #1) is silently disabled.
    # `constitution.py:1281` adds it to expected net alpha before the comparison
    # with the friction-multiplied requirement, so bound it by the gate's own
    # headline floor.
    _floor = ag["min_alpha_bps_default"]
    for _asset, _eps in ag["borderline_alpha_pass_epsilon_bps_by_asset"].items():
        assert 0 < _eps <= _floor, (
            f"{_asset} long epsilon {_eps}bps is not a tolerance against a "
            f"{_floor}bps floor"
        )
    assert set(ag["borderline_alpha_pass_epsilon_bps_by_asset"]) >= {"BTC", "ETH"}
    # [P165] Same drift as the long-side epsilon above (was 0.25, now 1.5): bound
    # it rather than pin it, and require the short side to be no more permissive
    # than the long side — an epsilon is a relaxation, and the short leg is the
    # riskier one (P140).
    assert 0 < ag["borderline_alpha_pass_epsilon_bps_short"] <= _floor
    assert (ag["borderline_alpha_pass_epsilon_bps_short"]
            <= ag["borderline_alpha_pass_epsilon_bps"])
    assert ag["borderline_alpha_pass_epsilon_bps_short_by_asset"]["BTC"] == pytest.approx(1.0)
    assert ag["borderline_alpha_pass_epsilon_bps_short_by_asset"]["ETH"] == pytest.approx(4.5)
    assert ag["borderline_alpha_pass_epsilon_bps_short_by_asset"]["SOL"] == pytest.approx(2.0)
    assert cfg["signal_quality"]["min_score_for_add_by_asset_side"]["BTC:LONG"] == pytest.approx(55.0)
    weekend = cfg["weekend_manager"]
    assert weekend["opportunity_alpha_multiplier_weekend_by_asset"]["BTC"] == pytest.approx(0.25)
    assert weekend["opportunity_alpha_multiplier_weekend_by_asset"]["ETH"] == pytest.approx(0.15)
    assert weekend["opportunity_alpha_multiplier_weekend_by_asset"]["SOL"] == pytest.approx(0.19)
    assert weekend["opportunity_confidence_weekend_by_asset"]["BTC"] == pytest.approx(0.30)
    assert weekend["opportunity_confidence_weekend_by_asset"]["SOL"] == pytest.approx(0.18)
    struct = cfg["structure_gate"]
    assert struct["min_net_edge_bps_long_by_asset"]["BTC"] == pytest.approx(2.0)
    assert struct["min_net_edge_bps_long_by_asset"]["ETH"] == pytest.approx(-3.0)
    assert struct["min_net_edge_bps_short_by_asset"]["SOL"] == pytest.approx(-2.0)
    assert live["trade_gate"]["min_edge_multiplier_long_by_asset"]["BTC"] == pytest.approx(1.0)
    assert struct["min_abs_direction_long_by_asset"]["BTC"] == pytest.approx(0.08)
    assert struct["min_abs_direction_long_by_asset"]["ETH"] == pytest.approx(0.05)
    tranche = cfg["tranche"]
    assert tranche["entry_volume_collapse_threshold_by_asset"]["SOL"] == pytest.approx(0.015)


class TestTradeIntentActionableThresholds:
    def test_opportunity_short_uses_short_specific_direction_floor(self):
        intent = TradeIntentV36(
            direction=-0.047,
            target_exposure=0.02,
            system_mode="OPPORTUNITY",
            alpha_gate_passed=True,
            opportunity_actionable_direction_threshold_short=0.045,
        )
        assert intent.is_actionable is True

    def test_opportunity_long_does_not_use_short_specific_direction_floor(self):
        intent = TradeIntentV36(
            direction=0.047,
            target_exposure=0.02,
            system_mode="OPPORTUNITY",
            alpha_gate_passed=True,
            opportunity_actionable_direction_threshold_short=0.045,
        )
        assert intent.is_actionable is False

    def test_normal_short_uses_short_specific_direction_floor(self):
        intent = TradeIntentV36(
            direction=-0.088,
            target_exposure=0.09,
            system_mode="NORMAL",
            alpha_gate_passed=True,
            normal_actionable_direction_threshold_short=0.085,
        )
        assert intent.is_actionable is True

    def test_normal_long_does_not_use_short_specific_direction_floor(self):
        intent = TradeIntentV36(
            direction=0.088,
            target_exposure=0.09,
            system_mode="NORMAL",
            alpha_gate_passed=True,
            normal_actionable_direction_threshold_short=0.085,
        )
        assert intent.is_actionable is False


# ===========================================================================
# Test Group 4: Friction breakdown in AlphaGatingResult
# ===========================================================================

class TestFrictionBreakdown:
    """AlphaGatingResult includes friction component breakdown."""

    def test_free_tier_friction_breakdown(self):
        """Free tier: fee=0, slippage=5, latency=2."""
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=0.0)
        assert result.friction_fee_bps == 0.0
        assert result.friction_slippage_bps == 5.0
        assert result.friction_latency_bps == 2.0
        assert result.friction_total_bps == 7.0

    def test_post_free_tier_friction_breakdown(self):
        """Post-free tier: taker fee=26, slippage=5, latency=2."""
        calc = _make_calculator(monthly_volume_usd=50_000.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=0.0)
        assert result.friction_fee_bps == 26.0
        assert result.friction_slippage_bps == 5.0
        assert result.friction_latency_bps == 2.0
        assert result.friction_total_bps == 33.0

    def test_maker_fee_breakdown(self):
        """Maker mode uses maker fee (16bps post-free)."""
        calc = _make_calculator(monthly_volume_usd=50_000.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", is_maker=True, min_alpha_bps=0.0)
        assert result.friction_fee_bps == 16.0
        assert result.friction_total_bps == 23.0


# ===========================================================================
# Test Group 5: Gate decision labels
# ===========================================================================

class TestGateDecisionLabels:
    """Gate decision correctly identifies which layer blocked."""

    def test_allow_label(self):
        """Trade that passes both layers -> ALLOW."""
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=5.0)
        # alpha = 0.5 * 200 * 0.8 * 0.75 = 60bps >> 14
        assert result.gate_decision == "ALLOW"
        assert result.passes_threshold is True

    def test_reject_min_alpha_label(self):
        """Blocked by min_alpha floor (not EV gate) -> REJECT_MIN_ALPHA."""
        calc = _make_calculator(monthly_volume_usd=0.0)
        # alpha = tiny, min_alpha=15 > EV=8.05 (OPPORTUNITY)
        result = calc.check_alpha_gate(
            signal_strength=0.01, regime_confidence=0.8,
            mode="OPPORTUNITY", min_alpha_bps=15.0)
        # alpha = 0.01 * 200 * 0.8 * 0.75 = 1.2bps
        assert result.passes_threshold is False
        assert result.gate_decision == "REJECT_MIN_ALPHA"

    def test_reject_ev_label(self):
        """Blocked by EV gate (not min_alpha) -> REJECT_EV."""
        calc = _make_calculator(monthly_volume_usd=0.0)
        # alpha = small, min_alpha=0 -> EV gate=14 (NORMAL) blocks
        result = calc.check_alpha_gate(
            signal_strength=0.01, regime_confidence=0.8,
            mode="NORMAL", min_alpha_bps=0.0)
        assert result.passes_threshold is False
        assert result.gate_decision == "REJECT_EV"

    def test_no_trade_mode_label(self):
        """NO_TRADE mode -> REJECT_NO_TRADE."""
        calc = _make_calculator(monthly_volume_usd=0.0)
        result = calc.check_alpha_gate(
            signal_strength=0.5, regime_confidence=0.8,
            mode="NO_TRADE", min_alpha_bps=5.0)
        assert result.gate_decision == "REJECT_NO_TRADE"
        assert result.passes_threshold is False


# ===========================================================================
# Test Group 6: Config round-trip from JSON
# ===========================================================================

class TestConfigFromJSON:
    """Config loads correctly from ultra_aggressive_5y.json."""

    def test_ultra_json_has_alpha_gate_section(self):
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ag = data.get("alpha_gate")
        assert ag is not None
        assert ag["min_alpha_bps_default"] == 5
        assert ag["stage_selector"] == "WEEK1"
        assert len(ag["staged_schedule"]) == 3

    def test_ultra_json_staged_values(self):
        """The staged schedule must loosen monotonically, never tighten.

        [P165] This pinned `MONTH1 == 5` while the config says 3 — and 5 would
        have made MONTH1 *stricter* than WEEK2_3's 4, which is the one thing a
        "staging" schedule must not do. The stale literal was asserting a shape
        the design forbids. Pin the ordering property instead of the numbers, so
        retuning the bar does not break the test but inverting the ramp does.
        """
        json_path = Path("configs/ultra_aggressive_5y.json")
        if not json_path.exists():
            pytest.skip("ultra_aggressive_5y.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        schedule = {e["stage"]: e["min_alpha_bps"]
                    for e in data["alpha_gate"]["staged_schedule"]}
        ordered = [schedule[s] for s in ("WEEK1", "WEEK2_3", "MONTH1") if s in schedule]
        assert len(ordered) == 3, f"missing a stage: {schedule}"
        assert all(v > 0 for v in ordered), f"a stage would disable the gate: {schedule}"
        assert ordered == sorted(ordered, reverse=True), (
            f"staged min_alpha_bps must be non-increasing over time, got {schedule}"
        )

    def test_cloud_production_has_no_alpha_gate(self):
        """cloud_production.json should NOT have alpha_gate (HIGH_RISK default)."""
        json_path = Path("configs/cloud_production.json")
        if not json_path.exists():
            pytest.skip("cloud_production.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        # No alpha_gate section -> defaults apply (min_alpha=0, pure EV gate)
        assert data.get("alpha_gate") is None

    def test_high_risk_json_has_short_side_overrides(self):
        json_path = Path("configs/high_risk.json")
        if not json_path.exists():
            pytest.skip("high_risk.json not found")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        ag = data["alpha_gate"]
        assert ag["min_alpha_bps_short_default"] == 5
        assert ag["quiet_accum_min_direction_short_by_asset"]["BTC"] == pytest.approx(0.08)
        # [P165] `opportunity_actionable_min_direction_short` was pinned at
        # 0.045 and has since been retuned to 0.1. The literal was never the
        # point of a test named "has_short_side_overrides" — what it guards is
        # that the short side is present and NOT looser than the long side,
        # since a spot/margin short is the riskier leg (P140). Assert that.
        assert ag["opportunity_actionable_min_direction_short"] >= (
            ag["opportunity_actionable_min_direction"]
            if "opportunity_actionable_min_direction" in ag else 0.0
        )
        assert ag["opportunity_actionable_min_direction_short"] > 0
        # [P165] Was pinned at 0.085, retuned to 0.15. Same reasoning as above:
        # what the override must guarantee is that it exists and is a real floor.
        assert ag["normal_actionable_min_direction_short"] > 0
        # [P165] Same drift again (was 0.25, now 1.5). Bound it rather than pin
        # it: an epsilon is a relaxation of the EV bar, so it must be positive,
        # must not exceed the alpha floor it relaxes, and the short leg must
        # never be more permissive than the long leg (P140).
        assert 0 < ag["borderline_alpha_pass_epsilon_bps_short"] <= ag["min_alpha_bps_default"]
        assert (ag["borderline_alpha_pass_epsilon_bps_short"]
                <= ag["borderline_alpha_pass_epsilon_bps"])
        schedule = {e["stage"]: e["min_alpha_bps"] for e in ag["staged_schedule_short"]}
        long_schedule = {e["stage"]: e["min_alpha_bps"] for e in ag["staged_schedule"]}
        # [P165] Was pinned at 5; the short schedule has since been retuned to
        # 8/8/6/5 against the long side's 6/6/5/4. The literal was never the
        # point — a short leg is the riskier one (P140), so what this must hold
        # is that at every stage the short bar is at least as strict as the
        # long bar, and that it relaxes monotonically over time like the long
        # one does. Both survive retuning; a literal does not.
        assert set(schedule) == set(long_schedule)
        for _stage, _bps in schedule.items():
            assert _bps >= long_schedule[_stage], (
                f"short min_alpha at {_stage} ({_bps}) is looser than long "
                f"({long_schedule[_stage]})")
        ordered = [schedule[s] for s in ("HIGH_RISK", "WEEK1", "WEEK2_3", "MONTH1")]
        assert all(v > 0 for v in ordered)
        assert ordered == sorted(ordered, reverse=True), (
            f"staged_schedule_short must be non-increasing over time, got {schedule}")
        # [P165] Was pinned at 0.80, now 1.1. Despite the "relax" name this is a
        # plain multiplier on the alpha gate (`main.py:8919-8922`): <1 relaxes,
        # >1 tightens, and it is clamped to [0.5, 1.5] before use. The short map
        # has been retuned from relaxing (0.80) to tightening (1.1), which is a
        # deliberate risk decision, not a regression — so pin the clamp range
        # and the long/short ordering (P140), not the number.
        _relax_short = ag["regime_alpha_gate_relax_short"]
        _relax_long = ag["regime_alpha_gate_relax"]
        assert set(_relax_short) == set(_relax_long)
        for _regime, _mult in _relax_short.items():
            assert 0.5 <= _mult <= 1.5, (
                f"{_regime} short relax {_mult} falls outside the [0.5, 1.5] "
                f"clamp applied at main.py:8920 — the config value would be "
                f"silently truncated")
            assert _mult >= _relax_long[_regime], (
                f"{_regime}: short gate multiplier {_mult} is more permissive "
                f"than long {_relax_long[_regime]}")


# ===========================================================================
# Test Group 7: Backward compatibility - min_alpha_bps=0 default
# ===========================================================================

class TestBackwardCompatibility:
    """min_alpha_bps=0 (default) preserves exact pre-P1-2A behavior."""

    def test_zero_min_alpha_same_as_no_param(self):
        """Passing min_alpha_bps=0 is identical to not passing it."""
        calc = _make_calculator(monthly_volume_usd=0.0)

        result_default = calc.check_alpha_gate(
            signal_strength=0.3, regime_confidence=0.6, mode="NORMAL")
        result_zero = calc.check_alpha_gate(
            signal_strength=0.3, regime_confidence=0.6, mode="NORMAL",
            min_alpha_bps=0.0)

        assert result_default.passes_threshold == result_zero.passes_threshold
        assert result_default.threshold_bps == result_zero.threshold_bps
        assert result_default.estimated_alpha_bps == result_zero.estimated_alpha_bps

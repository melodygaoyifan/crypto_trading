"""[P166] The promotion gate must not clear a strategy that cannot pay its fees.

`determine_verdict` promoted on `min(|IC|) > 0.05 and sharpe > 0.5`. Three
defects, each independently sufficient to promote a strategy that is
mathematically certain to lose money:

  1. No cost term at all. IC is dimensionless, fees are in bps, and nothing in
     the function converted between them. At the ~107bps of 16h forward vol
     these assets actually show, IC 0.05 is worth about 4.5bps per round trip
     against 6bps of Coinbase taker fee before any spread — so the pass mark
     sat below break-even. Realized round-trip cost over the 85 closed trades
     in data/trade_attribution.jsonl was 31.1bps median.

  2. No significance term. SE(IC) ~= 1/sqrt(n-1); at the shipped min_samples=30
     an IC of 0.05 is 0.27 SE from zero. Five days of 4H bars reached that.

  3. `abs()` on the promote branch, while nothing downstream inverts a
     negative-IC strategy — `promotion_gate/promotion_plan.py` maps PROMOTE
     straight to PROMOTE_TO_FUSION. P143 measured model_alpha at IC -0.160.
     The old gate would have promoted it, and fusion would have traded it in
     the direction it predicts against.

Every check added here can only ever remove a PROMOTE, never add one, so the
gate cannot have become looser. These tests pin that property directly.
"""

import math

import pytest

from analytics.shadow_ic.compute_shadow_ic import (
    DEFAULT_COST_MARGIN,
    DEFAULT_MIN_IC_T_STAT,
    DEFAULT_ROUND_TRIP_COST_BPS,
    PromotionAssessment,
    Verdict,
    assess_promotion,
    assess_record,
    determine_verdict,
    expected_edge_bps,
    required_ic_for_costs,
    spearman_to_pearson,
)

# An IC that is large enough, measured on enough samples, on an asset volatile
# enough to be worth trading. Everything below perturbs one axis of this.
# [P253] n_per_h rescaled for the overlap correction (t now uses
# n_eff = n / h, the P231 arithmetic): the old flat 400 gave n_eff of
# 100/33/16 at h=4/12/24 — sample counts the corrected gate correctly calls
# insignificant. Inputs rescaled, assertions untouched (the P167 rule).
CLEAN = dict(
    ic_per_h={4: 0.12, 12: 0.13, 24: 0.14},
    n_per_h={4: 2000, 12: 4000, 24: 6000},
    sharpe=0.8,
    window_days=30,
    fwd_vol_bps_per_h={4: 400.0, 12: 400.0, 24: 400.0},
)


def _assess(**overrides):
    kwargs = dict(CLEAN)
    kwargs.update(overrides)
    return assess_promotion(**kwargs)


# ---------- the baseline still promotes -------------------------------------

def test_a_genuinely_good_strategy_still_promotes():
    a = _assess()
    assert a.verdict is Verdict.PROMOTE
    assert a.blockers == []


# ---------- defect 1: costs --------------------------------------------------

def test_edge_below_cost_blocks_promotion():
    """Same IC, low-volatility asset: the correlation is real but there is not
    enough movement in the underlying to pay for the round trip."""
    a = _assess(fwd_vol_bps_per_h={4: 40.0, 12: 40.0, 24: 40.0})

    assert a.verdict is not Verdict.PROMOTE
    assert any("edge" in b and "required" in b for b in a.blockers)


def test_the_legacy_promote_inputs_now_hold():
    """The exact inputs the old test called a clean promote."""
    v = determine_verdict({4: 0.06, 12: 0.07, 24: 0.08},
                          {4: 100, 12: 100, 24: 100},
                          sharpe=0.8, window_days=30,
                          fwd_vol_bps_per_h={4: 107.0, 12: 190.0, 24: 270.0})
    assert v is Verdict.HOLD


def test_ic_005_at_realistic_vol_does_not_cover_costs():
    """The headline number: the old bar, priced."""
    edge = expected_edge_bps(0.05, 107.0)
    assert edge < DEFAULT_ROUND_TRIP_COST_BPS, (
        f"IC 0.05 on 107bps forward vol is worth {edge:.2f}bps, which must be "
        f"below the {DEFAULT_ROUND_TRIP_COST_BPS}bps round-trip fee alone"
    )


def test_required_ic_rises_as_volatility_falls():
    quiet = required_ic_for_costs(50.0)
    loud = required_ic_for_costs(400.0)
    assert quiet > loud > 0.0


def test_required_ic_is_infinite_when_no_correlation_could_pay():
    """A vol so low that even a perfect signal loses to fees. Must be
    unreachable, not merely large — the caller compares against it."""
    assert required_ic_for_costs(0.01) == math.inf
    assert required_ic_for_costs(0.0) == math.inf
    assert required_ic_for_costs(-1.0) == math.inf


def test_required_ic_is_the_exact_inverse_of_expected_edge():
    for vol in (80.0, 150.0, 400.0, 1200.0):
        need = required_ic_for_costs(vol)
        assert math.isfinite(need)
        got = expected_edge_bps(need, vol)
        target = DEFAULT_ROUND_TRIP_COST_BPS * DEFAULT_COST_MARGIN
        assert got == pytest.approx(target, rel=1e-9), f"vol={vol}"


def test_spearman_to_pearson_is_monotone_and_anchored():
    assert spearman_to_pearson(0.0) == pytest.approx(0.0)
    assert spearman_to_pearson(1.0) == pytest.approx(1.0)
    assert spearman_to_pearson(-1.0) == pytest.approx(-1.0)
    assert spearman_to_pearson(0.05) > 0.05          # slight inflation
    assert spearman_to_pearson(0.3) > spearman_to_pearson(0.2)


def test_out_of_range_correlation_is_clamped_not_exploded():
    """Defensive: a caller passing a malformed IC must not produce NaN that
    then compares false against every threshold and reads as a pass."""
    for bad in (2.5, -2.5, 1.0000001):
        assert math.isfinite(spearman_to_pearson(bad))


# ---------- the fail-closed rule (P159/P164 lesson) --------------------------

def test_missing_volatility_blocks_promotion_rather_than_skipping_the_check():
    """A cost check that could not run is not a cost check that passed."""
    a = _assess(fwd_vol_bps_per_h={})

    assert a.verdict is not Verdict.PROMOTE
    assert any("unavailable" in b for b in a.blockers)


def test_partially_missing_volatility_still_blocks():
    a = _assess(fwd_vol_bps_per_h={4: 400.0, 12: 400.0})  # 24 absent
    assert a.verdict is not Verdict.PROMOTE
    assert any("h=24" in b and "unavailable" in b for b in a.blockers)


@pytest.mark.parametrize("bad_vol", [0.0, -5.0, float("nan"), float("inf")])
def test_degenerate_volatility_is_treated_as_missing(bad_vol):
    a = _assess(fwd_vol_bps_per_h={4: bad_vol, 12: 400.0, 24: 400.0})
    assert a.verdict is not Verdict.PROMOTE
    assert any("h=4" in b and "unavailable" in b for b in a.blockers)


def test_no_volatility_argument_at_all_cannot_promote():
    """The default path. Any caller that has not been taught to supply vol
    gets a refusal, not a free pass."""
    v = determine_verdict(CLEAN["ic_per_h"], CLEAN["n_per_h"],
                          sharpe=0.8, window_days=30)
    assert v is not Verdict.PROMOTE


# ---------- defect 2: significance ------------------------------------------

def test_ic_indistinguishable_from_zero_blocks_promotion():
    """Big enough IC, volatile enough asset, but only 30 observations."""
    a = _assess(n_per_h={4: 30, 12: 30, 24: 30})

    assert a.verdict is not Verdict.PROMOTE
    assert any("SE from zero" in b for b in a.blockers)


def test_min_samples_30_is_far_below_significance_for_a_005_ic():
    """The arithmetic that made the old min_samples meaningless."""
    t = 0.05 * math.sqrt(30 - 1)
    assert t < 0.3
    n_needed = (DEFAULT_MIN_IC_T_STAT / 0.05) ** 2 + 1
    assert n_needed > 1500


def test_blocker_reports_the_sample_size_that_would_clear():
    a = _assess(n_per_h={4: 40, 12: 400, 24: 400})
    msg = next(b for b in a.blockers if b.startswith("h=4:") and "SE from zero" in b)
    assert "n_required" in msg


def test_more_samples_alone_can_turn_a_hold_into_a_promote():
    assert _assess(n_per_h={4: 40, 12: 40, 24: 40}).verdict is not Verdict.PROMOTE
    # [P253] the promote leg needs overlap-corrected significance:
    # n_eff = n/h must clear (t_req/ic)^2 + 1 at every horizon
    assert _assess(n_per_h={4: 2000, 12: 4000, 24: 6000}).verdict is Verdict.PROMOTE


# ---------- defect 3: sign ---------------------------------------------------

def test_anti_predictive_strategy_is_never_promoted():
    """P143 measured model_alpha at IC -0.160. Nothing downstream flips it."""
    a = _assess(ic_per_h={4: -0.12, 12: -0.13, 24: -0.14})

    assert a.verdict is not Verdict.PROMOTE
    assert any("not positive" in b for b in a.blockers)


def test_mixed_sign_ic_is_not_promoted():
    a = _assess(ic_per_h={4: 0.12, 12: -0.13, 24: 0.14})
    assert a.verdict is not Verdict.PROMOTE


def test_kill_still_uses_absolute_ic():
    """A strongly negative IC is informative, not weak — it must not be KILLed
    as if it had no signal. Only the promote path cares about sign."""
    a = _assess(ic_per_h={4: -0.12, 12: -0.13, 24: -0.14})
    assert a.verdict is Verdict.HOLD


# ---------- unchanged semantics ---------------------------------------------

def test_kill_semantics_are_unchanged():
    a = _assess(ic_per_h={4: 0.01, 12: 0.02, 24: 0.03}, sharpe=1.0)
    assert a.verdict is Verdict.KILL


def test_insufficient_samples_semantics_are_unchanged():
    a = _assess(n_per_h={4: 5, 12: 5, 24: 5})
    assert a.verdict is Verdict.INSUFFICIENT_SAMPLES


def test_short_window_never_promotes_and_says_why():
    a = _assess(window_days=14)
    assert a.verdict is Verdict.HOLD
    assert any("too short to promote" in b for b in a.blockers)


def test_short_window_kill_unchanged():
    a = _assess(ic_per_h={4: 0.01, 12: 0.02, 24: 0.03}, window_days=14)
    assert a.verdict is Verdict.KILL


def test_sharpe_bar_still_applies():
    a = _assess(sharpe=0.3)
    assert a.verdict is not Verdict.PROMOTE
    assert any("sharpe" in b for b in a.blockers)


def test_empty_input_is_insufficient_not_promote():
    a = assess_promotion({}, {}, sharpe=5.0, window_days=90)
    assert a.verdict is Verdict.INSUFFICIENT_SAMPLES


# ---------- the "only ever tightens" property -------------------------------

def _legacy_verdict(ic_per_h, n_per_h, sharpe, window_days,
                    min_samples=30, promote_ic=0.05, kill_ic=0.05,
                    promote_sharpe=0.5):
    """The gate exactly as it shipped before P166."""
    if all(n < min_samples for n in n_per_h.values()):
        return Verdict.INSUFFICIENT_SAMPLES
    valid = [h for h, n in n_per_h.items() if n >= min_samples]
    if not valid:
        return Verdict.INSUFFICIENT_SAMPLES
    ics = [abs(ic_per_h[h]) for h in valid]
    if window_days <= 14:
        return Verdict.KILL if max(ics) < kill_ic else Verdict.HOLD
    if min(ics) > promote_ic and sharpe > promote_sharpe:
        return Verdict.PROMOTE
    if max(ics) < kill_ic:
        return Verdict.KILL
    return Verdict.HOLD


CASES = [
    ({4: 0.06, 12: 0.07, 24: 0.08}, {4: 100, 12: 100, 24: 100}, 0.8, 30),
    ({4: 0.12, 12: 0.13, 24: 0.14}, {4: 400, 12: 400, 24: 400}, 0.8, 30),
    ({4: -0.20, 12: -0.21, 24: -0.22}, {4: 500, 12: 500, 24: 500}, 1.5, 60),
    ({4: 0.01, 12: 0.02, 24: 0.03}, {4: 100, 12: 100, 24: 100}, 1.0, 30),
    ({4: 0.06, 12: 0.04, 24: 0.07}, {4: 100, 12: 100, 24: 100}, 0.8, 30),
    ({4: 0.10, 12: 0.10, 24: 0.10}, {4: 5, 12: 5, 24: 5}, 2.0, 14),
    ({4: 0.30, 12: 0.30, 24: 0.30}, {4: 1000, 12: 1000, 24: 1000}, 3.0, 90),
]


@pytest.mark.parametrize("ic,n,sharpe,window", CASES)
def test_new_gate_never_promotes_where_the_old_one_would_not(ic, n, sharpe, window):
    """The safety property. Adding conditions to a conjunction can only shrink
    the accepted set; this pins it against future edits to the branch order."""
    new = assess_promotion(ic, n, sharpe, window,
                           fwd_vol_bps_per_h={h: 400.0 for h in ic}).verdict
    if new is Verdict.PROMOTE:
        assert _legacy_verdict(ic, n, sharpe, window) is Verdict.PROMOTE


@pytest.mark.parametrize("ic,n,sharpe,window", CASES)
def test_kill_and_insufficient_verdicts_are_bit_identical_to_the_old_gate(
        ic, n, sharpe, window):
    old = _legacy_verdict(ic, n, sharpe, window)
    if old in (Verdict.KILL, Verdict.INSUFFICIENT_SAMPLES):
        new = assess_promotion(ic, n, sharpe, window,
                               fwd_vol_bps_per_h={h: 400.0 for h in ic}).verdict
        assert new is old


# ---------- observability ----------------------------------------------------

def test_hold_always_explains_itself():
    """A HOLD with no blocker is unactionable — the operator cannot tell
    whether to wait for samples or archive the strategy."""
    for ic, n, sharpe, window in CASES:
        a = assess_promotion(ic, n, sharpe, window,
                             fwd_vol_bps_per_h={h: 400.0 for h in ic})
        if a.verdict is not Verdict.PROMOTE:
            assert a.blockers, f"{a.verdict} with no reason: {ic} {n}"


def test_assessment_reports_the_numbers_behind_the_verdict():
    a = _assess()
    for h in (4, 12, 24):
        d = a.per_horizon[h]
        assert d["edge_bps"] > d["required_bps"]
        assert d["t_stat"] >= DEFAULT_MIN_IC_T_STAT
        assert d["fwd_vol_bps"] == 400.0
        assert d["required_ic"] < abs(d["ic"])
    assert a.round_trip_cost_bps == DEFAULT_ROUND_TRIP_COST_BPS
    assert a.cost_margin == DEFAULT_COST_MARGIN


def test_assessment_serializes_for_the_json_report():
    d = _assess().to_dict()
    assert d["verdict"] == "PROMOTE"
    assert d["blockers"] == []
    assert set(d["per_horizon"]) == {"4", "12", "24"}   # JSON keys are strings
    import json
    json.loads(json.dumps(d))


def test_summary_and_report_cannot_disagree():
    """Both call sites route through assess_record, so the console verdict and
    the JSON verdict are the same object by construction."""
    record = {
        "ic_per_horizon": CLEAN["ic_per_h"],
        "n_per_horizon": CLEAN["n_per_h"],
        "fwd_vol_bps_per_horizon": CLEAN["fwd_vol_bps_per_h"],
        "annualized_sharpe": 0.8,
    }
    assert assess_record(record, 30).verdict is Verdict.PROMOTE
    assert assess_record(record, 30).to_dict()["verdict"] == "PROMOTE"


def test_assess_record_tolerates_a_record_with_no_volatility_key():
    """Reports written before P166 have no fwd_vol_bps_per_horizon. They must
    degrade to "cannot verify", not to "verified"."""
    record = {
        "ic_per_horizon": CLEAN["ic_per_h"],
        "n_per_horizon": CLEAN["n_per_h"],
        "annualized_sharpe": 0.8,
    }
    a = assess_record(record, 30)
    assert a.verdict is not Verdict.PROMOTE
    assert any("unavailable" in b for b in a.blockers)


def test_compute_per_strategy_ic_emits_forward_volatility():
    """The producer side of the contract: if this key stops being written,
    every promotion silently fails closed and nobody would know why."""
    import inspect

    from analytics.shadow_ic import compute_shadow_ic

    src = inspect.getsource(compute_shadow_ic.compute_per_strategy_ic)
    assert '"fwd_vol_bps_per_horizon"' in src


def test_promotion_assessment_defaults_do_not_imply_a_pass():
    """A bare PromotionAssessment must not look like a clean promote."""
    a = PromotionAssessment(verdict=Verdict.HOLD)
    assert a.verdict is not Verdict.PROMOTE
    assert a.round_trip_cost_bps == 0.0

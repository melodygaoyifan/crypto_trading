"""[P352] The two armed entry filters, debugged rather than disarmed — plus the
gambler gate's input.

Operator, asked whether to disarm the filters: "instead of disarm, can we debug
the code, to make those integrate". Three defects were found by measurement,
and they are one shape: **a gate acting on a number that describes something
other than what it claims to describe.**

DEFECT A — the ledger claim's magnitude was the ACCOUNT'S CONTRACT SIZE.
`MAFilterEchoStrategy` sets `conf = min(1.0, abs(direction))`, and `direction`
is the sleeve TARGET. Until P274 `target_for` returned +/-1, so that expression
WAS a sign indicator; P274 made the target equity-scaled and the claim silently
became a magnitude. Measured live on the server's own ledgers, x = direction x
confidence:

    ma_filtered     BTC {1, 2}   ETH {7, 8}   SOL {3}
    whale_filtered  BTC {2}      ETH {7, 8}   SOL {3}

and these are the ONLY 2 of 30 shadow families whose |x| ever exceeded 1.0.
`compute_shadow_ic` scores x by RANK and POOLS both families across assets, and
P332 pre-committed the pooled read as the one that GOVERNS promote/kill — so
ETH's claims outranked SOL's outranked BTC's for no reason but sizing. The
scorer's own docstring guards exactly this on the RETURN side ("a
high-volatility asset would otherwise occupy the extreme ranks") and nobody
guarded the SIGNAL side. BTC's magnitude also moved 1 -> 2 as equity grew, so
the series was tracking the account balance.

DEFECT B — the whale veto fired on a "net pressure" computed from ONE trade.
`net_pressure` is a RATIO over the whales seen in the last hour, so it says
which way they leaned and cannot say how many there were. Measured over every
retained log (118 bucket-hours, 1,233 detections):

    n whales   buckets   emits a direction   |p| == 1.0 (saturated)
    1               27          100%                 100%
    2               23           87%                  83%
    3-4             18           89%                  61%
    5-9             16           69%                  25%
    10+             34           76%                  18%

and 73% of all directional whale readings in attribution carry confidence
exactly 1.000. `whale_count` has been computed, stashed, and written to this
very ledger since P349 — and the DECISION never received it (P144:
computed-but-unenforced, on an ARMED live veto).

DEFECT C — the gambler gate judged the retired decider. `strategy_agreement` is
abs(sum(signs))/4 over the Best-of-N TA indicators; since P298 the direction
comes from the regimebook seat, which consults none of them. On the gate's
first firing in 50,676 ticks it blocked an entry the alpha gate had passed at
66bps against 53 (P346).

CHECKED AND DELIBERATELY NOT BUILT: model_alpha's confidence when it disagrees
spans 0.341..0.619 (p50 0.465) and its data_quality is 1.0 on all 295
disagreements, so there is no dq defect to fix and a confidence floor would be
a NEW hypothesis with no evidence behind it. And 33 of 309 whale disagreements
carry data_quality -1.0, which is `DQ_NOT_REPORTED` — an attribution-extractor
artifact about a MISSING KEY, not a statement about the reading — so gating on
it would be gating on the wrong subject.
"""

import inspect
import re

import pytest

import main
from main import (
    HMATSProductionRunner,
    ProductionConfig,
    sleeve_agent_filter_decision,
    sleeve_ma_filter_decision,
    whale_direction_from_pressure,
)
from defense.strategy_shadow_v5_1 import MAFilterEchoStrategy
from risk.gambler_entry_exit import GamblerEntryChecker
from tests._guard_pins import assert_guard_live, assert_text_pin


# ==========================================================================
# DEFECT A — the ledger claim is a SIGN
# ==========================================================================
def _echo(ledger_dir, **extra):
    obs = {"_maf_ledger_dir": ledger_dir, "_maf_ma_dir": -1.0,
           "_maf_raw_target": ledger_dir, "_maf_sleeve_dir": 1.0,
           "_maf_pos": 0, "_maf_action": "", "_maf_reason": "x",
           "_maf_enforce": True}
    obs.update(extra)
    return MAFilterEchoStrategy().evaluate("ETH", obs)


@pytest.mark.parametrize("target,expect_dir", [
    (8, 1.0), (7, 1.0), (3, 1.0), (2, 1.0), (1, 1.0),
    (0, 0.0),
    (-1, -1.0), (-2, -1.0), (-3, -1.0), (-7, -1.0), (-8, -1.0),
])
def test_the_ledger_claim_is_a_sign_not_a_contract_count(target, expect_dir):
    sig = _echo(target)
    assert sig.direction == pytest.approx(expect_dir)


def test_a_directional_claim_carries_full_confidence_and_a_flat_one_carries_none():
    """The P236-followup contract, which the code stopped honouring at P274."""
    assert _echo(7).confidence == pytest.approx(1.0)
    assert _echo(-2).confidence == pytest.approx(1.0)
    assert _echo(0).confidence == pytest.approx(0.0), (
        "a confidence must never saturate on a zero direction (P224)"
    )


@pytest.mark.parametrize("magnitude", [1, 2, 3, 7, 8])
def test_the_scored_signal_no_longer_encodes_the_assets_contract_size(magnitude):
    """The pooling defect, stated as the property that fixes it.

    `compute_shadow_ic` scores x = direction x confidence and ranks it. Every
    directional claim must therefore produce the SAME |x| whatever the asset's
    contract size, or the pooled Spearman measures the sizing, not the filter.
    The magnitudes here are the ones measured live on the server.
    """
    sig = _echo(magnitude)
    x = sig.direction * sig.confidence
    assert abs(x) == pytest.approx(1.0), (
        f"|x| = {abs(x)} for a {magnitude}-contract claim; the pooled read "
        f"would rank this above a smaller asset's identical claim"
    )


def test_the_contract_count_is_preserved_rather_than_lost():
    sig = _echo(-8)
    assert sig.diagnostics["ledger_target"] == -8


def test_the_new_diagnostics_keys_are_on_the_whitelist():
    """P350's warn-once latch exists precisely so this cannot go unnoticed."""
    sig = _echo(2, _maf_whale_count=5, _maf_whale_min=2,
                _maf_whale_evidence_ok=True)
    assert sig.diagnostics["whale_count"] == 5
    assert sig.diagnostics["whale_min_count"] == 2
    assert sig.diagnostics["whale_evidence_ok"] is True


def test_a_ledger_row_for_an_asset_with_no_whale_data_omits_the_whale_keys():
    """P349's rule: an unconditional key writes null into every ma_filter row
    and reads as 'measured zero whales' rather than 'not applicable'."""
    sig = _echo(2)
    for k in ("whale_count", "whale_min_count", "whale_evidence_ok"):
        assert k not in sig.diagnostics


# ==========================================================================
# DEFECT B — the whale veto needs a sample size
# ==========================================================================
def test_with_one_observation_the_ratio_is_saturated_by_construction():
    """Why the floor is 2 and not a fitted number.

    A single whale puts all volume on one side, so net_pressure is exactly
    +/-1.0 and the +/-0.3 deadband tests nothing. Measured: 100% of n=1
    bucket-hours emit a direction, 100% of them saturated.
    """
    assert whale_direction_from_pressure(1.0) == 1.0
    assert whale_direction_from_pressure(-1.0) == -1.0


@pytest.mark.parametrize("pos,raw,agent_dir", [
    (0, 1, -1.0),      # entry-from-flat against the advisor
    (0, -1, 1.0),
    (1, -1, 1.0),      # a flip the advisor contradicts
    (-1, 1, -1.0),
])
def test_insufficient_evidence_fails_open_instead_of_vetoing(pos, raw, agent_dir):
    led, act, why = sleeve_agent_filter_decision(
        pos, raw, agent_dir, agent_tag="whale", evidence_ok=False)
    assert act == "", "an unevidenced opinion must not produce a veto"
    assert led == raw, "the ledger claim is the unfiltered target"
    assert why == "whale_low_evidence"


@pytest.mark.parametrize("pos", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("raw", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("agent_dir", [-1.0, 0.0, 1.0])
def test_the_evidence_gate_can_only_remove_vetoes_never_create_one(
        pos, raw, agent_dir):
    """Load-bearing safety property: this change must be a strict subset."""
    armed = sleeve_agent_filter_decision(pos, raw, agent_dir,
                                         agent_tag="whale", evidence_ok=True)
    unarmed = sleeve_agent_filter_decision(pos, raw, agent_dir,
                                           agent_tag="whale", evidence_ok=False)
    if unarmed[1]:
        assert unarmed[1] == armed[1], (
            "low evidence produced an action the evidenced path did not"
        )


@pytest.mark.parametrize("pos", [-1, 0, 1])
@pytest.mark.parametrize("raw", [-1, 0, 1])
@pytest.mark.parametrize("agent_dir", [-1.0, 0.0, 1.0])
def test_the_default_is_byte_identical_to_the_pre_p352_behaviour(
        pos, raw, agent_dir):
    assert (sleeve_agent_filter_decision(pos, raw, agent_dir, agent_tag="whale")
            == sleeve_agent_filter_decision(pos, raw, agent_dir,
                                            agent_tag="whale",
                                            evidence_ok=True))


@pytest.mark.parametrize("pos", [-1, 0, 1])
@pytest.mark.parametrize("raw", [-1, 0, 1])
@pytest.mark.parametrize("ma_dir", [-1.0, 0.0, 1.0])
def test_the_ma_filter_wrapper_is_untouched(pos, raw, ma_dir):
    """model_alpha has no measured evidence defect, so its filter must not
    quietly acquire a gate — see the module docstring."""
    led, act, why = sleeve_ma_filter_decision(pos, raw, ma_dir)
    assert "low_evidence" not in why


def test_the_min_whales_knob_is_declared_and_parsed_with_the_forced_floor():
    assert ProductionConfig().coinbase_whale_filter_min_whales == 2
    src = inspect.getsource(ProductionConfig.from_file)
    assert_text_pin(
        src, 'data.get("coinbase_whale_filter_min_whales", 2)',
        why="a declared-but-unparsed field is a switch that controls nothing "
            "(P313) — and this one gates a live veto",
    )


def test_the_live_profile_leaves_the_floor_at_its_default():
    """P239: absent = ctor default = the arithmetically forced floor. Setting
    it to 3 or 5 removes 47% / 63% of would-be vetoes and is a risk
    preference, which is the operator's to state, not this change's."""
    import json
    with open("configs/live_high_risk.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert "coinbase_whale_filter_min_whales" not in cfg


def test_an_unknown_whale_count_is_none_and_never_zero():
    """A producer gap must not be the thing that disarms an armed veto (P2)."""
    src = inspect.getsource(main)
    assert_text_pin(
        src, "self._last_whale_counts[asset] = None",
        near="_wh_cnt_raw = market_data.get('whale_count')",
        why="an absent whale_count must stash UNKNOWN, not 0 — 0 would fail "
            "the minimum and silently switch the filter off",
    )
    assert "self._last_whale_counts[asset] = 0\n" not in src


def test_unknown_evidence_keeps_the_veto_armed():
    src = inspect.getsource(HMATSProductionRunner.run_live)
    assert_guard_live(
        src, "_wf_unknown",
        why="the unknown branch must be a live guard, not commented out",
    )
    i = src.index("_wf_unknown = (")
    window = src[i:i + 1200]
    assert "_wf_ev = True" in window, (
        "an unreadable whale_count must leave the veto ARMED — the opposite "
        "makes a producer gap into a silent loosening"
    )


# ==========================================================================
# DEFECT C — the gambler gate's agreement must be about the decider
# ==========================================================================
def _check(agreement):
    return GamblerEntryChecker().check_entry_allowed(
        signal_confidence=0.95,
        strategy_agreement=agreement,
        composite_score=0.95,
        tranche_level=1,
    )


def test_a_not_applicable_agreement_is_skipped_rather_than_failed():
    assert _check(None).allowed is True


def test_a_real_agreement_below_the_bar_still_blocks():
    """The clause is not disabled in general — only when it is about someone
    else. This is the test that stops the fix becoming a blanket loosening."""
    res = _check(0.0)
    assert res.allowed is False
    assert "agreement" in res.reason


def test_the_reason_string_never_formats_a_missing_agreement():
    """A None reaching the f-string would raise inside the gate's own
    try/except and read as 'entry check skipped' (P193's shape)."""
    res = GamblerEntryChecker().check_entry_allowed(
        signal_confidence=0.0, strategy_agreement=None,
        composite_score=0.0, tranche_level=1)
    assert res.allowed is False
    assert "None" not in res.reason


def test_the_other_two_clauses_still_bind_when_agreement_is_not_applicable():
    low_conf = GamblerEntryChecker().check_entry_allowed(
        signal_confidence=0.0, strategy_agreement=None,
        composite_score=0.95, tranche_level=1)
    low_score = GamblerEntryChecker().check_entry_allowed(
        signal_confidence=0.95, strategy_agreement=None,
        composite_score=0.0, tranche_level=1)
    assert low_conf.allowed is False and "confidence" in low_conf.reason
    assert low_score.allowed is False and "score" in low_score.reason


def test_the_pipeline_stamps_which_strategy_the_agreement_describes():
    import data_mgmt.market_data_pipeline as mdp
    src = inspect.getsource(mdp)
    assert 'raw["strategy_agreement_producer"] = best_name' in src, (
        "without the stamp the consumer cannot tell whether the measure is "
        "about the decider that produced the trade"
    )
    # every writer of strategy_agreement must also name its producer
    writers = len(re.findall(r'"strategy_agreement"\s*:', src)) + len(
        re.findall(r'\["strategy_agreement"\]\s*=', src))
    stamps = len(re.findall(r'"strategy_agreement_producer"', src))
    assert stamps >= writers, (
        f"{writers} site(s) publish strategy_agreement but only {stamps} "
        f"name its producer — an unstamped one falls back to being applied"
    )


def test_the_gate_only_skips_on_a_real_mismatch():
    src = inspect.getsource(HMATSProductionRunner)
    assert_guard_live(
        src,
        "_g_agree_prod and _g_decider and _g_agree_prod != _g_decider",
        why="an ABSENT stamp must keep today's behaviour — a missing field "
            "must never be the thing that skips a check",
    )

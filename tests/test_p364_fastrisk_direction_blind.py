"""[P364] The inter-tick watchdog's price trigger is DIRECTION-BLIND, measured
live over a 12-hour rally — and the window was PROFITABLE, which is why this
is a recorded decision and not a fix.

Asked "any remaining tasks", a live sweep turned up 79 ERROR/CRITICAL in 12h
where an earlier 6h window had zero. Traced:

    ETH  2384.45 -> 2545.22   (+6.7%)
    SOL     90.69 ->   94.94   (+4.7%)

    FastRiskTick EXIT_ONLY, 12h:  32 real flattens
                                  31 re-fires on an already-flat asset
                                   8 SKIPPED_STALE (venue 500s, P141 refusal)

`price_move_pct = abs(current_price - anchor_price) / anchor_price` — so a
+6.7% rally fires the emergency exit exactly as hard as a −6.7% crash. The
book was LONG. The watchdog repeatedly flattened a winning position, the P232
cooldown blocked re-entry for 2 ticks, and the 4H book re-entered after it.

**AND THE WINDOW MADE MONEY**, which is the half that stops this being an
incident report: sleeve PnL −211.42 -> −167.76 (**+$43.66**), equity to a new
high with drawdown 0.00%. The book captured part of the rally despite 32
flattens. So the honest statement is *a live control is direction-blind and
that cost fees in a window it still won*, NOT *the watchdog is losing money*
— and the difference between those two sentences is the measurement.

WHY NOTHING IS CHANGED HERE
    Making the trigger direction-aware would make an emergency exit fire
    LESS. That is a LOOSENING of a live risk control, which is P141's
    boundary — the same reason P356's filter disarm needed an explicit
    instruction. It is also genuinely arguable on the merits: a 3% move
    inside one 4H bar means the market is disordered, and "reduce risk in
    disorder regardless of which way it went" is a defensible design for a
    30-second watchdog, not obviously a bug.

    What is NOT arguable is that the choice should be visible. The comment at
    the trigger says only "Trigger 1: Price move > 3%" — direction-blindness
    is nowhere stated, so a reader cannot tell a decision from an oversight.
    This file pins the premise with its numbers so the decision can be taken
    on evidence rather than re-derived.
"""

import inspect
import pathlib

import pytest

import main
from execution.fast_risk_tick import FastRiskAction, FastRiskTick

REPO = pathlib.Path(main.__file__).parent


def test_the_price_trigger_is_direction_blind():
    """The premise everything else rests on. If this ever becomes signed, the
    P364 measurement no longer describes the deployed control and the whole
    entry must be re-derived rather than inherited."""
    src = inspect.getsource(FastRiskTick)
    assert "abs(current_price - anchor_price)" in src, (
        "the price-move trigger is no longer absolute — it may now be "
        "direction-aware, which is a LOOSENING of an emergency exit and needs "
        "an operator decision (P141) plus a re-measurement of P364"
    )


def test_the_threshold_the_measurement_used():
    """3% over a 4H anchor. ETH moved 6.7% and SOL 4.7% in the measured
    window, i.e. both comfortably past it — the flattens were the control
    doing what it says, not a stale anchor (P156's failure mode, excluded)."""
    assert FastRiskTick.PRICE_MOVE_THRESHOLD == pytest.approx(0.03)


def test_the_anchor_staleness_bound_still_excludes_the_P156_artifact():
    """P156: an unboundedly old anchor makes ordinary movement look extreme
    forever. The measured flattens are only meaningful because that bound
    exists and is shorter than the observed burst."""
    assert FastRiskTick.ANCHOR_MAX_AGE_SEC == pytest.approx(21600.0)


def test_exit_only_is_reachable_on_a_FAVOURABLE_move():
    """The behaviour itself, exercised rather than read (P234): a long
    position and a price move UP by more than the threshold still produces
    EXIT_ONLY. This is the sentence the trigger's comment does not say."""
    frt = FastRiskTick()
    frt.set_4h_anchor("ETH", price=2000.0, volatility=0.01, depth=1_000_000.0)
    res = frt.evaluate("ETH", {
        "current_price": 2000.0 * 1.05,      # +5%, i.e. IN OUR FAVOUR
        "data_valid": True,
        "volatility_30m": 0.01,
        "orderbook_depth_1pct_usd": 1_000_000.0,
        "orderbook_stale": False,
    }, has_position=True)
    assert res.action.name == "EXIT_ONLY", (
        f"expected EXIT_ONLY on a +5% move, got {res.action}"
    )
    assert "price_move" in res.reason


def test_an_adverse_move_of_the_same_size_is_treated_identically():
    """The other half: symmetric by construction. Pinning both directions is
    what makes 'direction-blind' a measured property rather than a claim."""
    frt = FastRiskTick()
    out = {}
    for tag, mult in (("up", 1.05), ("down", 0.95)):
        frt.set_4h_anchor("ETH", price=2000.0, volatility=0.01, depth=1_000_000.0)
        out[tag] = frt.evaluate("ETH", {
            "current_price": 2000.0 * mult,
            "data_valid": True,
            "volatility_30m": 0.01,
            "orderbook_depth_1pct_usd": 1_000_000.0,
            "orderbook_stale": False,
        }, has_position=True).action.name
    assert out["up"] == out["down"] == "EXIT_ONLY", out


def test_the_trigger_states_what_it_does_about_direction():
    """[P364] The one thing changed here: the comment said only 'Price move >
    3%', so a reader could not tell a deliberate symmetry from an oversight.
    A live control whose behaviour surprises its own reader is the P177/P202
    shape — the decision must be visible at the site where it is taken."""
    src = inspect.getsource(FastRiskTick)
    i = src.index("Trigger 1: Price move")
    window = src[i:i + 900]
    # The property is that the site states it is SYMMETRIC, not merely that
    # the word "direction" appears somewhere nearby — the looser form stayed
    # green under a probe that deleted the headline claim, because the rest
    # of the block still discussed direction (P238: distrust the probe first,
    # then find that the guard was loose too).
    assert "DIRECTION-BLIND" in window, (
        "the price trigger does not state that it is direction-blind — that "
        "is the difference between a decision and an accident, and it is the "
        "one thing P364 changed"
    )


# ==========================================================================
# [P366] ROOT CAUSE: the trigger measures DRIFT, not VELOCITY
# ==========================================================================
def test_the_trigger_compares_against_the_4H_ANCHOR_not_the_last_tick():
    """[P366] The operator declined P364's symptom-level options ("make it
    signed", "raise the threshold") and asked for research. This is what it
    found, and it reframes P364's direction-blindness as a SYMPTOM.

    `price_move_pct` compares the current price to a reference set once per
    4H tick — so it measures CUMULATIVE DRIFT over up to four hours, while
    the control is a 30-second INTER-TICK watchdog whose job is a
    dislocation BETWEEN ticks. Those are different quantities.

    Measured over ~13,800 live samples per asset (persistent log, 24h):

        asset   one-step (~34s) >= 3%     drift-from-4H-anchor >= 3%
        BTC                          4     2732  (19.7% of samples)
        ETH                          4     2574  (18.6%)
        SOL                          4     2585  (18.7%)

    So it fires on roughly ONE EVALUATION IN FIVE, while genuine inter-tick
    dislocations happen 4 times in 13,838 (0.03%) — a ~650x gap. Median
    one-step move is 0.011-0.017%; median drift is 0.62-0.82% and p95 drift
    is ~10%, i.e. ordinary trending routinely clears a 3% drift bar.

    This also explains P364's finding without needing a separate cause:
    cumulative drift in a trend is monotone, so an ABSOLUTE drift measure
    necessarily fires on rallies. Direction-blindness is downstream of
    measuring the wrong quantity.

    Nothing is changed. Switching to velocity is a real behaviour change to a
    live risk control (it would fire ~650x less), and while that is arguably
    the control doing its own job instead of duplicating the 4H tick's, it is
    the operator's call (P141). Pinned so the decision rests on the numbers."""
    src = inspect.getsource(FastRiskTick)
    i = src.index("price_move_pct = abs(")
    line = src[i:src.index("\n", i)]
    assert "anchor_price" in line, (
        "the trigger no longer references the 4H anchor — if it now measures "
        "an inter-tick move, P366's measurement describes a control that no "
        "longer exists and must be re-derived, not inherited"
    )


def test_the_anchor_is_set_once_per_4H_tick_which_is_why_it_is_drift():
    """The other half of the premise: the reference is refreshed on the 4H
    decision path, so by the end of a bar it is up to four hours old. If it
    were refreshed every evaluation the same code WOULD measure velocity."""
    src = inspect.getsource(FastRiskTick.set_4h_anchor)
    assert "_last_4h_prices" in src
    doc = (FastRiskTick.set_4h_anchor.__doc__ or "")
    assert "4H" in doc or "4h" in doc, (
        "set_4h_anchor no longer documents its cadence — the drift-vs-velocity "
        "distinction rests on it"
    )


def test_a_slow_drift_and_a_sudden_dislocation_are_INDISTINGUISHABLE():
    """The defect stated as behaviour rather than as prose: a 4% move that
    accrued smoothly over four hours and a 4% gap in one tick produce the
    identical action, because only the endpoint is compared."""
    frt = FastRiskTick()
    frt.set_4h_anchor("ETH", price=2000.0, volatility=0.01, depth=1_000_000.0)
    md = {"current_price": 2080.0, "data_valid": True, "volatility_30m": 0.01,
          "orderbook_depth_1pct_usd": 1_000_000.0, "orderbook_stale": False}
    smooth = frt.evaluate("ETH", md, has_position=True)

    frt2 = FastRiskTick()
    frt2.set_4h_anchor("ETH", price=2000.0, volatility=0.01, depth=1_000_000.0)
    # the same endpoint, reached in one step
    sudden = frt2.evaluate("ETH", md, has_position=True)

    assert smooth.action == sudden.action == FastRiskAction.EXIT_ONLY
    assert smooth.price_move_pct == pytest.approx(sudden.price_move_pct), (
        "the control cannot tell a four-hour drift from a one-tick gap — "
        "that is the root cause, and it is why the trigger fires on ~19% of "
        "evaluations while real dislocations are 0.03% (P366)"
    )

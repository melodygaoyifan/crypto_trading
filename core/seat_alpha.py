"""
================================================================================
HMATS [P320] - the asserted alpha, calibrated to the SEAT'S OWN holding horizon
================================================================================

WHY THIS EXISTS

    P318 established that the alpha gate compares two quantities in different
    units: `estimated_alpha_bps` is a per-TICK constant while `friction` is a
    per-ROUND-TRIP cost. The regimebook seat asserted a flat

        _rb_edge = 30.0 * abs(_rb_dir)

    inherited in shape from the trend seat's fast signal (P231 records
    `base_edge_bps = 40` as "a constant chosen to clear the gate"). The regime
    book holds ~40 bars. So the gate was pricing a multi-week position with a
    per-tick number, and correcting only the FEE side (P315) would have
    rejected trades whose realized edge is 2.5x-27x their cost.

    This module replaces the constant with a MEASUREMENT: what the seat has
    actually earned, gross, per round trip, in its WORST era.

THE MEASUREMENT (training/funding_legs_lab, 6y, honest per-contract fees,
gross bps per round trip = 2 x gross per unit turnover)

        asset   pre_design   design   validation   MIN (asserted)
        BTC            2.3     68.5         24.1              2.3
        ETH          251.7     88.1         52.1             52.1
        SOL          427.6    221.7        -20.8            -20.8

WHY THE MINIMUM, AND WHAT IT DECIDES

    The gate is a safety control, so it must assume the worst era repeats
    (P167: overcharging costs opportunity, undercharging spends money). Using
    the mean instead would be a deliberate loosening and needs its own entry.

    Read off the table, this is not a tuning knob — it is a verdict:

      * SOL is NEGATIVE in the most recent era. The seat asserts a negative
        edge, which can never clear friction, so SOL stops trading on its own
        arithmetic. That independently confirms the risk-column reading
        (hold beats the SOL book on return, Sharpe AND drawdown).
      * BTC's worst era is 2.3bps against ~28bps of round-trip friction, and
        even its validation era (24.1) sits below cost. The BTC book is
        profitable in ONE era of three — era-fragility of exactly the kind
        P243/P244 treat as disqualifying.
      * ETH clears in every era (min 52.1) and is the only seat whose measured
        edge survives its own worst window. It is also the certified,
        un-fitted config (P247) that beats hold on Sharpe (0.72 vs 0.58) with
        41% less drawdown.

FAIL DIRECTIONS (all resolve toward NOT trading)

    * An unknown asset asserts 0.0 -> can never clear friction.
    * A negative calibration is passed through NEGATIVE, not clamped to 0:
      "this seat loses money per round trip" is information the gate should
      act on, and clamping would silently upgrade it to merely-unprofitable.
    * The value is per ROUND TRIP, matching what `check_alpha_gate` compares
      it against. Scaling it by |direction| would re-introduce the per-tick
      shape this module exists to remove, so it deliberately does NOT.

PRE-COMMITTED REVISION RULE

    Re-derive only from the lab, only across ALL eras, and only by taking the
    minimum. Raising an entry requires a new P-entry stating the window. This
    table may never be edited to make a desired trade pass.
================================================================================
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("HMATS.SeatAlpha")

# Gross bps per ROUND TRIP, era-minimum. Provenance in the module docstring.
REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP: Dict[str, float] = {
    "BTC": 2.3,
    "ETH": 52.1,
    "SOL": -20.8,
}

# Full per-era measurement, kept so a reader can see the dispersion the
# minimum is drawn from rather than having to trust the single number.
REGIMEBOOK_ALPHA_BY_ERA: Dict[str, Dict[str, float]] = {
    "BTC": {"pre_design": 2.3, "design": 68.5, "validation": 24.1},
    "ETH": {"pre_design": 251.7, "design": 88.1, "validation": 52.1},
    "SOL": {"pre_design": 427.6, "design": 221.7, "validation": -20.8},
}

_MEASURED_ON = "2026-08-19"
_MEASURED_BY = "training/funding_legs_lab.py (FEE_MODEL=per_contract, 6y)"


def regimebook_alpha_bps(asset: str) -> Tuple[float, str]:
    """Calibrated per-ROUND-TRIP gross edge for the regimebook seat.

    Returns ``(bps, provenance)``. Unknown asset -> 0.0, which cannot clear
    friction: a seat with no measurement must not be able to trade on one.
    """
    a = str(asset or "").upper().strip()
    if a not in REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP:
        return 0.0, f"no_calibration_for:{a}"
    v = REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP[a]
    eras = REGIMEBOOK_ALPHA_BY_ERA.get(a, {})
    # `key=eras.get` is an overloaded signature and mypy rejects it; the
    # lambda is the same lookup with a single concrete type (P287g: fix the
    # finding at source, never by re-baselining).
    worst = min(eras, key=lambda k: eras[k]) if eras else "?"
    return v, f"era_min({worst})@{_MEASURED_ON}"


def calibrated_seat_alpha(asset: str, seat: str,
                          fallback_bps: float) -> Tuple[float, str]:
    """Dispatch by seat. Only `regimebook` is calibrated so far.

    Any other seat keeps its existing asserted constant (`fallback_bps`) —
    calibrating one seat must not silently re-price another whose horizon was
    never measured (the P315 lesson: a units fix applied to the wrong side is
    worse than no fix).
    """
    if str(seat or "").lower() == "regimebook":
        return regimebook_alpha_bps(asset)
    return float(fallback_bps), f"uncalibrated_seat:{seat}"


def resolve_seat_edge(asset: str, seat: str, direction: float,
                      base_bps: float, calibrated_enabled: bool,
                      honest_fees_enabled: bool) -> float:
    """The whole arming decision as ONE pure call, so the seat block stays
    short enough for the P256/P265 window guards to see its dict writes.

    Returns the edge in bps. A FLAT book (direction == 0) asserts nothing and
    always gets `base_bps * |direction|` == 0 — a calibrated round-trip value
    must never be asserted for a position the seat is not taking.

    THE INTERLOCK LIVES HERE, and both flags are required:
      * calibrated alpha WITHOUT honest fees raises ETH 22.5 -> 52.1 while the
        fee stays ~3x understated — a pure loosening;
      * honest fees WITHOUT calibrated alpha rejects trades whose realized
        edge is 2.5x-27x their cost (P318).
    They are halves of one correction and may only move together.
    """
    fallback = float(base_bps) * abs(float(direction or 0.0))
    if not direction or not (calibrated_enabled and honest_fees_enabled):
        return fallback
    try:
        bps, prov = calibrated_seat_alpha(asset, seat, fallback)
    except Exception as e:  # noqa: silent-swallow — logged, keeps the constant
        logger.warning("[P320] calibration failed for %s/%s (%s: %s) — "
                       "keeping the asserted constant", asset, seat,
                       type(e).__name__, e)
        return fallback
    logger.info("[P320-ALPHA] %s: seat alpha %+.1fbps/round-trip (%s) "
                "replaces the asserted %.1f", asset, bps, prov, fallback)
    return float(bps)

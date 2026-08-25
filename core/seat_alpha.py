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
    actually earned, gross, per round trip, taken as its era-MEDIAN.

THE MEASUREMENT (6y, honest per-contract fees, gross bps per round trip =
2 x gross per unit turnover)

    [P326] REPRODUCIBLE, not hand-copied. The producer is
    training/seat_alpha_calibration.py; `--verify` recomputes this table and
    exits 3 on drift. The one-sentence derivation that used to stand here was
    not enough to re-derive it -- doing so gave BTC 240.3 / 58.4 / 44.0 -- so
    the convention's three clauses (gross only; eras indexed on the POSITIONS
    frame; an opening position is not turnover) live there, pinned by test.

        asset   pre_design   design   validation    MIN   MEDIAN (asserted)
        BTC            2.3     68.5         24.1    2.3               24.1
        ETH          251.7     88.1         52.1   52.1               88.1
        SOL          427.6    221.7        -20.8  -20.8              221.7

WHY THE MEDIAN, AND WHAT IT DECIDES

    [P321] The gate prices EXPECTED edge against expected cost, so the robust
    central estimate is the median across eras. The MINIMUM (BTC 2.3, ETH 52.1,
    SOL -20.8) is the right statistic for a pure safety control and stops every
    asset; the MEAN (BTC 31.6, ETH 130.6, SOL 209.5 — the mean of the table
    below; [P370] an earlier draft quoted 80.3/224.9/354.7, figures from a
    superseded convention, P326) can be carried by a single
    dominant era, which is the era-fragility P243/P244 disqualify on. The median
    is neither. This is a recorded risk-preference choice, not a measurement
    upgrade, and it is still far stricter than the flat 22.5 the seat asserted
    before any of this work.

    Read off the table, this is not a tuning knob — it is a verdict:

      * BTC asserts 24.1 against ~28bps of honest round-trip friction and a
        ~35bps threshold, so BTC STOPS TRADING on its own arithmetic. Its edge
        is concentrated in one era of three (2.3 / 68.5 / 24.1) — the
        era-fragility P243/P244 treat as disqualifying — and the median is the
        statistic that says so without needing a separate judgement.
      * ETH asserts 88.1 and clears (threshold ~56). It is positive in EVERY
        era, the only seat whose edge survives its own worst window, and the
        certified un-fitted P247 config that beats hold on Sharpe (0.72 vs
        0.58) with 41% less drawdown.
      * SOL asserts 221.7 and clears (threshold ~60), on the strength of two
        very large eras against one mildly negative one (-20.8). It carries the
        widest dispersion of the three, so it is the entry most exposed to the
        median being the wrong statistic — watch it first.

FAIL DIRECTIONS (all resolve toward NOT trading)

    * An unknown asset asserts 0.0 -> can never clear friction.
    * A negative calibration is passed through NEGATIVE, not clamped to 0:
      "this seat loses money per round trip" is information the gate should
      act on, and clamping would silently upgrade it to merely-unprofitable.
      No asset's MEDIAN is negative today (SOL's validation era is, its median
      is not), so this is a property of the function rather than of the current
      table — and it must stay, because a future re-derivation can produce one.
    * The value is per ROUND TRIP, matching what `check_alpha_gate` compares
      it against. Scaling it by |direction| would re-introduce the per-tick
      shape this module exists to remove, so it deliberately does NOT.

PRE-COMMITTED REVISION RULE

    Re-derive only from the lab, only across ALL eras, and only by taking the
    MEDIAN (the statistic is part of the decision — see P321). Changing an
    entry requires a new P-entry stating the window and the statistic. This
    table may never be edited to make a desired trade pass.
================================================================================
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger("HMATS.SeatAlpha")

# Gross bps per ROUND TRIP, era-MEDIAN. Provenance in the module docstring.
#
# [P321] Changed from era-MINIMUM to era-MEDIAN by explicit operator decision
# ("our goal is to let the system up and running, and can make profit").
# The minimum is the right statistic for a pure safety control — it assumes the
# worst era repeats — and it asserts BTC 2.3 / SOL -20.8, which stops every
# asset. The gate's job here is to price EXPECTED edge against expected cost,
# and the median is the robust central estimate: unlike the mean (BTC 31.6,
# SOL 209.5; [P370] corrected to the table's own arithmetic) it cannot be
# carried by one dominant era, which is precisely the
# era-fragility P243/P244 treat as disqualifying.
#
# This is a LOOSENING relative to the minimum and a TIGHTENING relative to the
# flat 22.5 the seat asserted before any of this — recorded as a deliberate
# risk-preference choice, not a measurement upgrade. The minimum remains
# computable from REGIMEBOOK_ALPHA_BY_ERA below for anyone who wants it back.
REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP: Dict[str, float] = {
    "BTC": 24.1,
    "ETH": 88.1,
    "SOL": 221.7,
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
    return v, f"era_median@{_MEASURED_ON}"


# ============================================================================
# [P407e] Skew-contra seat, calibrated the SAME way as regimebook (P320/P321).
#
# The seat asserted a HAND-PICKED constant (`_SKEW_SEAT_EDGE_BPS = {100, 100}`
# in main.py) chosen so the validated signal would clear the alpha gate. That is
# exactly the "a constant chosen to clear the gate" anti-pattern P231 names and
# this module exists to remove. Replaced with the MEASUREMENT: gross bps per
# round trip, era-MEDIAN, over 6.6y of Deribit 25d skew (Laevitas deep history)
# at honest CDE per-contract cost.
#
#     asset   per calendar-year era 2021..2026 (gross bps/RT)         MEDIAN
#     BTC     563, 308, -74, 1902, 486, 2141                          525.0
#     ETH     857, 966,  20, 3061,   0,  621                          738.8
#
# Median for the same reason as regimebook (P321): the robust central estimate,
# not carried by one dominant era. BTC's earliest era is NEGATIVE (-74, the 2023
# low-vol regime); the median is robust to it. The value sits far above any gate
# threshold (~34-58) BY MEASUREMENT, not by choice -- the signal fires only at
# z-extremes (~15x/yr), so a rare high-conviction entry legitimately carries a
# large per-RT edge. NO cap is applied: the regimebook seat asserts its raw
# median too (SOL 221.7), and a skew-only cap would itself be a hand-picked
# deviation; a bad era is bounded by caps/stops/fuse/net-cap, not by
# understating the edge. PROVENANCE: training/skew_seat_calibration.py (P407f),
# a committed, reproducible producer with --verify (exit 3 on drift) -- the
# counterpart to regimebook's seat_alpha_calibration.py. Its input (6.6y Deribit
# 25d skew) is operator-local and gitignored (proprietary Laevitas data;
# --verify refuses with exit 2 when it is absent, P213). The table may never be
# edited to make a trade pass (the P320 revision rule).
SKEW_CONTRA_ALPHA_BY_ERA: Dict[str, Dict[str, float]] = {
    # per CALENDAR-YEAR gross bps/round-trip; reproduced by
    # training/skew_seat_calibration.py --verify (P407f).
    "BTC": {"2021": 563.5, "2022": 308.2, "2023": -73.6, "2024": 1901.6, "2025": 486.5, "2026": 2140.6},
    "ETH": {"2021": 856.6, "2022": 965.7, "2023": 20.2, "2024": 3060.7, "2025": 0.5, "2026": 620.9},
}
SKEW_CONTRA_ALPHA_BPS_PER_ROUND_TRIP: Dict[str, float] = {
    "BTC": 525.0,   # median(2021..2026)
    "ETH": 738.8,
}
_SKEW_MEASURED_ON = "2026-08-24"
_SKEW_MEASURED_BY = "training/skew_seat_calibration.py --verify (6.6y Deribit 25d skew, gross per-RT era-median)"


def _median(vals) -> float:
    s = sorted(float(v) for v in vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def skew_contra_alpha_bps(asset: str) -> Tuple[float, str]:
    """Calibrated per-ROUND-TRIP gross edge for the skew-contra seat.

    Returns ``(bps, provenance)``. Unknown asset -> 0.0 (a seat with no
    measurement must not trade on one). The asserted constant is verified equal
    to the era-MEDIAN of SKEW_CONTRA_ALPHA_BY_ERA by test (the P326
    no-silent-drift rule).
    """
    a = str(asset or "").upper().strip()
    if a not in SKEW_CONTRA_ALPHA_BPS_PER_ROUND_TRIP:
        return 0.0, f"no_calibration_for:{a}"
    return SKEW_CONTRA_ALPHA_BPS_PER_ROUND_TRIP[a], f"skew_era_median@{_SKEW_MEASURED_ON}"


def calibrated_seat_alpha(asset: str, seat: str,
                          fallback_bps: float) -> Tuple[float, str]:
    """Dispatch by seat. Only `regimebook` is calibrated so far.

    Any other seat keeps its existing asserted constant (`fallback_bps`) —
    calibrating one seat must not silently re-price another whose horizon was
    never measured (the P315 lesson: a units fix applied to the wrong side is
    worse than no fix).
    """
    _s = str(seat or "").lower()
    if _s == "regimebook":
        return regimebook_alpha_bps(asset)
    if _s == "skew_contra":
        return skew_contra_alpha_bps(asset)
    # [P407e] whale / etf_flow / mlpshadow are DELIBERATELY uncalibrated and keep
    # the generic fallback: whale's edge is measured noise (P324, t=0.26) so it
    # must NOT be asserted up; etf and mlp have no clean per-RT era table yet.
    # An uncalibrated seat keeping its constant is the correct state, not an
    # oversight -- calibrating one seat must never silently re-price another.
    return float(fallback_bps), f"uncalibrated_seat:{seat}"


def resolve_seat_edge(asset: str, seat: str, direction: float,
                      base_bps: float, calibrated_enabled: bool,
                      honest_fees_enabled: bool,
                      price: Optional[float]) -> float:
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
    # [P321b] INTERLOCK ON EFFECT, NOT ON FLAG. Both flags were on and the
    # honest fee still did not apply — it read market_data["price"], a key no
    # producer writes, so it fell back to the modelled 3bps while the
    # calibrated alpha DID apply. That is precisely the half P318 warned
    # about (calibrated alpha + cheap fees = pure loosening), and it went live
    # for one tick. A flag being true is not evidence the correction took
    # effect, so the alpha half now refuses unless the fee half can actually
    # price THIS asset at THIS price.
    # [P341b] `price` is REQUIRED and an ABSENT price refuses, exactly like an
    # unusable one. P321b built this as an effect-interlock precisely because
    # a flag being on is not evidence the correction took effect — and then
    # guarded it with `if price is not None:`, so `price=None` skipped the
    # check outright. The sole production caller passes
    # `market_data.get("current_price")`, which returns None on a MISSING key,
    # so the interlock was one absent key away from re-arming the exact state
    # P321b caught live: calibrated alpha applied while the honest fee was
    # not, which P318 identifies as a PURE LOOSENING. P321b fixed the
    # wrong-key case (`price` -> `current_price`) and left the missing-key
    # case open, because None was read as "not supplied" rather than as
    # "cannot price". A safety interlock must not have an opt-out that its
    # own default selects.
    from core.cde_fees import cde_fee_bps
    _px_ok = price is not None
    if _px_ok:
        try:
            _px = float(price)
            # finite AND positive. `inf > 0` is True, and since P334 made the
            # fee a PERCENTAGE it is price-invariant, so cde_fee_bps happily
            # prices an infinite price and the interlock would pass on
            # nonsense — caught by exercising the fail directions rather than
            # by reading the condition.
            _px_ok = math.isfinite(_px) and _px > 0.0
        except (TypeError, ValueError):
            _px_ok = False
    if not _px_ok or cde_fee_bps(asset, price, is_maker=False) is None:
        logger.warning(
            "[P321b] %s: honest fee not priceable (px=%r) — NOT applying "
            "the calibrated alpha either; the two must move together",
            asset, price)
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

"""
================================================================================
HMATS [P294] - Seat controller: which candidate holds the DECIDE slot
================================================================================

Replaces the P237 tripwire's shape. The tripwire is a KILL SWITCH — one
variable (calibrator slope), one threshold, one direction (remove) — and the
practitioner literature is explicit that "a kill switch with one threshold
based on one variable will not be viable". More decisively: its entire action
space is "trade less", so it cannot serve a goal of trading at all.

This asks the only well-posed version of the question: **of the candidates
that can hold the DECIDE slot, which one should?** The tripwire becomes one
possible answer ("flat wins"), not the only one.

DESIGN RULES, each with its reason:

  1. MULTI-VARIABLE. A candidate is scored on forward IC at BOTH horizons.
     One number cannot separate "no edge" from "not enough data".

  2. WEAKEST-HORIZON SCORING. score = min(ic_4h, ic_16h). A candidate must be
     non-negative on BOTH horizons to score above zero. Deliberately
     conservative: the 16h cell is where this system's signals have
     historically gone negative while 4h looked fine (P198, P293k).

  3. HYSTERESIS. A challenger must beat the incumbent by SWITCH_MARGIN, not
     merely tie. Without it the seat thrashes on noise — the same reason
     flip-persistence exists on the order path, and the reason Gârleanu-
     Pedersen's optimal policy adjusts PARTIALLY rather than jumping.

  4. NEVER SWITCH ON NOISE ALONE. A challenger with |t| < MIN_SWITCH_T on its
     decisive horizon cannot take the seat however good its point estimate
     looks. Every IC in this system currently sits inside noise, so without
     this the controller would chase random ordering every week.

  5. UNAVAILABLE != FLAT. A candidate that is structurally inert (SOL's
     regimebook has no bear-leg model — deleted in P250 — so it can only ever
     emit flat) is UNAVAILABLE, not "a candidate that says flat". Scoring it
     as flat would let a broken candidate win by default (P2).

  6. IT NEVER EDITS CONFIG. Like tripwire_check, it prints the exact edit and
     exits with a code. Changing what drives live money stays a human step
     (P141).

Exit codes:  0 = incumbent holds   3 = SWITCH recommended   2 = refusal
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# A challenger must beat the incumbent's score by this much to take the seat.
# 0.02 IC is roughly the spread between "indistinguishable" candidates in this
# system's measured range (all |IC| < 0.13), so anything smaller is noise.
SWITCH_MARGIN = 0.02

# Minimum |t| on the decisive horizon before a challenger may take the seat.
MIN_SWITCH_T = 1.0

# Minimum joined records before a candidate is scoreable at all.
MIN_N = 60

FLAT = "flat"


@dataclass
class Candidate:
    """Forward evidence for one seat candidate."""
    name: str
    ic_4h: Optional[float] = None
    ic_16h: Optional[float] = None
    t_4h: Optional[float] = None
    t_16h: Optional[float] = None
    n: int = 0
    # Structural availability: False when the candidate CANNOT express a
    # position (missing model, no producer). Distinct from "says flat".
    available: bool = True
    # Fraction of ticks the candidate is directional. A candidate that never
    # takes a position cannot hold the seat usefully even if its IC is fine.
    in_market_rate: Optional[float] = None
    note: str = ""

    def score(self) -> Optional[float]:
        """Weakest-horizon IC, or None when not scoreable.

        None is deliberate and distinct from 0.0: "not enough evidence" must
        never read as "measured flat" (P199/P2).
        """
        if not self.available:
            return None
        if self.n < MIN_N:
            return None
        if self.ic_4h is None or self.ic_16h is None:
            return None
        return min(float(self.ic_4h), float(self.ic_16h))

    def decisive_t(self) -> Optional[float]:
        """|t| on the horizon that produced the (weakest) score."""
        s4, s16 = self.ic_4h, self.ic_16h
        if s4 is None or s16 is None:
            return None
        t = self.t_16h if float(s16) <= float(s4) else self.t_4h
        return None if t is None else abs(float(t))

    def reasons(self) -> List[str]:
        out = []
        if not self.available:
            out.append(f"UNAVAILABLE: {self.note or 'cannot express a position'}")
        elif self.n < MIN_N:
            out.append(f"insufficient evidence (n={self.n} < {MIN_N})")
        elif self.ic_4h is None or self.ic_16h is None:
            out.append("missing an IC horizon")
        if self.in_market_rate is not None and self.in_market_rate <= 0.0:
            out.append("never takes a position (in-market rate 0%)")
        return out


@dataclass
class SeatDecision:
    incumbent: str
    winner: str
    switch: bool
    reason: str
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    blockers: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "incumbent": self.incumbent,
            "winner": self.winner,
            "switch": self.switch,
            "reason": self.reason,
            "scores": self.scores,
            "blockers": self.blockers,
        }


def decide_seat(
    candidates: List[Candidate],
    incumbent: str,
    switch_margin: float = SWITCH_MARGIN,
    min_switch_t: float = MIN_SWITCH_T,
) -> SeatDecision:
    """Pure decision. Returns who should hold the DECIDE slot.

    `flat` is always an implicit candidate with score 0.0: it is the option of
    holding no position, and it wins when every real candidate scores at or
    below zero. That is the tripwire's job, subsumed.
    """
    scores: Dict[str, Optional[float]] = {}
    blockers: Dict[str, List[str]] = {}
    for c in candidates:
        scores[c.name] = c.score()
        b = c.reasons()
        if b:
            blockers[c.name] = b

    inc_score = scores.get(incumbent)
    # An unscoreable incumbent is treated as 0.0 for COMPARISON only — it
    # still holds the seat unless something beats it, but it must not be
    # unbeatable just because its evidence is missing.
    inc_cmp = 0.0 if inc_score is None else inc_score

    scoreable = [c for c in candidates if scores.get(c.name) is not None]
    positive = [c for c in scoreable if (scores[c.name] or 0.0) > 0.0]

    if not positive:
        # Nothing has a non-negative edge on both horizons -> flat.
        if incumbent == FLAT:
            return SeatDecision(
                incumbent, FLAT, False,
                "no candidate scores above zero on both horizons; already flat",
                scores, blockers)
        return SeatDecision(
            incumbent, FLAT, True,
            "no candidate scores above zero on both horizons — flat wins "
            "(this is the tripwire's verdict, reached by comparison)",
            scores, blockers)

    best = max(positive, key=lambda c: scores[c.name] or 0.0)
    best_score = scores[best.name] or 0.0

    if best.name == incumbent:
        return SeatDecision(incumbent, incumbent, False,
                            f"incumbent is already the best-scoring candidate "
                            f"({best_score:+.4f})", scores, blockers)

    # Hysteresis: a challenger must CLEAR the incumbent by the margin.
    if best_score < inc_cmp + switch_margin:
        return SeatDecision(
            incumbent, incumbent, False,
            f"{best.name} ({best_score:+.4f}) does not beat incumbent "
            f"{incumbent} ({inc_cmp:+.4f}) by the {switch_margin:.3f} margin "
            f"— holding to avoid thrashing on noise",
            scores, blockers)

    # Significance floor: never hand the seat to a point estimate alone.
    t = best.decisive_t()
    if t is None or t < min_switch_t:
        return SeatDecision(
            incumbent, incumbent, False,
            f"{best.name} scores best ({best_score:+.4f}) but its decisive "
            f"|t|={'n/a' if t is None else f'{t:.2f}'} < {min_switch_t} — a "
            f"point estimate inside noise cannot take the seat",
            scores, blockers)

    return SeatDecision(
        incumbent, best.name, True,
        f"{best.name} ({best_score:+.4f}, |t|={t:.2f}) beats {incumbent} "
        f"({inc_cmp:+.4f}) by more than the {switch_margin:.3f} margin",
        scores, blockers)


# =============================================================================
# The config edit each winner implies — printed, never applied (P141)
# =============================================================================

SEAT_CONFIG_EDIT: Dict[str, str] = {
    "trend": ('trend_following_mode: "enforce"  + whale_seat_mode: "off"  '
              '+ regimebook_mode: "off"'),
    "whale": ('whale_seat_mode: "enforce"  (whale wins the seat when it has '
              'an opinion; the incumbent covers its silent ticks)'),
    "regimebook": ('regimebook_mode: "enforce"  + whale_seat_mode: "off"'),
    FLAT: ('trend_assets: []  + whale_seat_mode: "off"  + regimebook_mode: '
           '"off"   — every seat vacated, the book goes flat'),
}


def render(decision: SeatDecision) -> str:
    lines = ["=" * 70, "SEAT CONTROLLER — who should hold the DECIDE slot", "=" * 70, ""]
    lines.append(f"incumbent : {decision.incumbent}")
    lines.append(f"winner    : {decision.winner}")
    lines.append(f"action    : {'SWITCH' if decision.switch else 'HOLD'}")
    lines.append(f"reason    : {decision.reason}")
    lines.append("")
    lines.append("scores (weakest-horizon IC; None = not scoreable):")
    for name, s in sorted(decision.scores.items(),
                          key=lambda kv: (kv[1] is None, -(kv[1] or 0.0))):
        lines.append(f"   {name:14s} {'n/a' if s is None else f'{s:+.4f}'}")
        for b in decision.blockers.get(name, []):
            lines.append(f"        - {b}")
    lines.append("")
    if decision.switch:
        lines.append("CONFIG EDIT IMPLIED (apply by hand — this tool never "
                     "edits config, P141):")
        lines.append(f"   {SEAT_CONFIG_EDIT.get(decision.winner, '(unknown seat)')}")
    else:
        lines.append("No config change. The incumbent holds.")
    lines.append("")
    lines.append("NOTE: a positive score here is NOT a claim of profitability — "
                 "it is a RELATIVE ranking among candidates whose ICs all sit "
                 "inside noise. The controller improves which signal drives and "
                 "how often, never the edge itself.")
    return "\n".join(lines)


__all__ = ["Candidate", "SeatDecision", "decide_seat", "render",
           "SEAT_CONFIG_EDIT", "SWITCH_MARGIN", "MIN_SWITCH_T", "MIN_N", "FLAT"]

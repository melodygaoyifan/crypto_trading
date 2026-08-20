"""[P327] One enforced contract for every constant that is a MEASUREMENT.

THE ROOT CAUSE, stated as the three incidents it produced:

  * P326 — the seat-alpha table could not be re-derived. Its whole derivation
    was one docstring sentence; implementing exactly that sentence gave BTC
    240.3 / 58.4 / 44.0 against the shipped 2.3 / 68.5 / 24.1, and the
    convention turned out to have three clauses. An entire comparison was
    invalidated until it was recovered by enumeration.
  * P315 — CDE fees were modelled as a percentage for the life of the system
    while the venue charges a flat fee per contract. Nothing recorded WHEN the
    model was last checked against a fill, so nobody knew it was a year old.
  * P316 — the cascade threshold was calibrated from an 11-day noisy proxy
    while six months of the direct series sat on disk. Nothing recorded WHAT
    DATA a constant was derived from, so nobody could ask "is there better?".

Each author invented their own provenance convention, so the tree had three:
a pair of module stamps, a per-call `provenance` string, and — for the spread
table that prices every trade — nothing at all.

WHAT THIS FIXES, per field:

    producer       the COMMAND that re-derives it. Not a module name, a
                   runnable command — P326's gap was precisely that a name is
                   not a derivation.
    source         the data it was derived FROM, so "is there a better source"
                   is answerable without archaeology (the P316 gap).
    measured_on    when, so staleness is visible (the P315 gap).
    staleness_days a DECISION about how fast the world moves under it, with
                   the reason recorded beside it.
    revision_rule  which direction may be changed on what evidence, so a
                   loosening cannot be slipped in as a refresh (P167).

This registry is DECLARATIVE and imports nothing at module scope beyond the
stdlib: it is read by a guard and by scripts/calibration_check.py, and must
never become a second copy of the values themselves (P172/P310). `resolve()`
imports on demand and returns the LIVE value, so a renamed or deleted constant
fails loudly instead of leaving a registry entry describing nothing.
"""
from __future__ import annotations

import datetime as _dt
import importlib
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# The stamps a covered module must carry. Any module that declares one is
# claiming to hold a measurement and MUST appear here — enforced in both
# directions by tests/test_p327_calibration_registry.py, which is the half
# that stops the convention being used without registering (P310).
STAMP_MEASURED_ON = "_MEASURED_ON"
STAMP_MEASURED_BY = "_MEASURED_BY"


@dataclass(frozen=True)
class Calibration:
    symbol: str            # dotted path, e.g. "core.seat_alpha.FOO"
    measured_on: str       # ISO date
    producer: str          # the command that re-derives it
    source: str            # the data it was derived from
    staleness_days: int    # a decision; see staleness_reason
    staleness_reason: str
    revision_rule: str

    @property
    def module(self) -> str:
        return self.symbol.rsplit(".", 1)[0]

    @property
    def attr(self) -> str:
        return self.symbol.rsplit(".", 1)[1]

    def age_days(self, today: Optional[_dt.date] = None) -> int:
        d = _dt.date.fromisoformat(self.measured_on)
        return ((today or _dt.date.today()) - d).days

    def is_stale(self, today: Optional[_dt.date] = None) -> bool:
        return self.age_days(today) > self.staleness_days


REGISTRY: Tuple[Calibration, ...] = (
    Calibration(
        symbol="core.seat_alpha.REGIMEBOOK_ALPHA_BY_ERA",
        measured_on="2026-08-19",
        producer=("python -X utf8 training/seat_alpha_calibration.py "
                  "--verify"),
        source=("training/training_data 4H closes + daily funding, 6y, "
                "honest per-contract fees (funding_legs_lab)"),
        staleness_days=180,
        staleness_reason=(
            "Derived from six years of history, so a month of new bars moves "
            "it little — but the VALIDATION era is open-ended, so new data "
            "lands entirely in the era the gate leans on. Half a year is the "
            "point at which that era has grown enough to matter."),
        revision_rule=(
            "Re-run the producer; it exits 3 on drift. Do NOT edit the "
            "constants to match a new number without deciding whether the "
            "DATA moved or the CONVENTION did (P326)."),
    ),
    Calibration(
        symbol="core.cde_fees.CDE_FEE_BPS",
        measured_on="2026-08-20",
        producer=("python -X utf8 scripts/coinbase_probe_stop_support.py "
                  "  # venue PREVIEW quotes; fills via "
                  "scripts/fill_quality_review.py"),
        source=("data/fill_quality.jsonl (6 fills) + read-only CDE preview "
                "quotes 2026-08-20 — the pair is what refuted the "
                "flat-per-contract model (P334): fills alone spanned only "
                "0.35% of price and could not tell flat from percentage"),
        staleness_days=90,
        staleness_reason=(
            "A venue can change its fee schedule quietly and the only way we "
            "learn is from fills. P315 found this model wrong for the life of "
            "the system because nothing said when it was last checked."),
        revision_rule=(
            "May be RAISED on any evidence. May be LOWERED only on >=20 "
            "filled legs for that asset plus a new P-entry (P315's "
            "pre-committed rule) — overcharging costs opportunity, "
            "undercharging spends money (P167)."),
    ),
    Calibration(
        symbol="defense.constitution.CDE_SPREAD_BPS_MEASURED",
        measured_on="2026-08-16",
        producer=("python -X utf8 scripts/coinbase_probe_stop_support.py "
                  "  # read-only CDE order-book probe (P289)"),
        source=("live CDE order-book snapshots, 6 samples/contract, taken on "
                "a WEEKEND book (deliberately the worst case)"),
        staleness_days=90,
        staleness_reason=(
            "Spreads track volume and listing depth, which drift on a scale "
            "of months. Measured on a weekend book so the standing figure is "
            "conservative; a refresh should only ever tighten it."),
        revision_rule=(
            "Full spread, rounded UP (P167). The P290 fill-quality ledger is "
            "the instrument that eventually replaces the probe: lower only on "
            ">=20 filled legs per asset plus a new P-entry."),
    ),
)


def resolve(cal: Calibration) -> Any:
    """The LIVE value behind an entry. Raises if the symbol no longer exists —
    a registry entry describing nothing is worse than no entry, because it
    reads as coverage (P174)."""
    mod = importlib.import_module(cal.module)
    return getattr(mod, cal.attr)


def stale_entries(today: Optional[_dt.date] = None) -> List[Calibration]:
    return [c for c in REGISTRY if c.is_stale(today)]


def by_symbol(symbol: str) -> Optional[Calibration]:
    for c in REGISTRY:
        if c.symbol == symbol:
            return c
    return None

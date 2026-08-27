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
    # [P420] The module-level stamp names this entry is paired with. A module
    # may carry MORE THAN ONE measured table (core.seat_alpha holds both the
    # regimebook and the skew tables, stamped `_MEASURED_ON` and
    # `_SKEW_MEASURED_ON`); the guard matches every `_*MEASURED_ON` name and
    # each must be registered under its own entry — the pre-P420 scanner saw
    # only the bare name, so the skew table was stamped-but-unregistered.
    stamp_measured_on: str = STAMP_MEASURED_ON
    stamp_measured_by: str = STAMP_MEASURED_BY

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
                "honest per-contract fees (funding_legs_lab); BTC/ETH/SOL "
                "2026-08-19, XRP + BNB 2026-08-26 (P412b/P412c, same "
                "producer, their own 6y _4H_ohlcv.parquet)"),
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
        # [P420] The skew seat's calibrated per-RT edge (P407e/P407f). Was
        # stamped in core.seat_alpha as `_SKEW_MEASURED_ON` and registered
        # NOWHERE — the registry guard matched only the bare stamp name, so
        # the constant that decides whether the LIVE BTC/ETH decider clears
        # the alpha gate had no staleness clock.
        symbol="core.seat_alpha.SKEW_CONTRA_ALPHA_BY_ERA",
        measured_on="2026-08-24",
        producer=("python -X utf8 training/skew_seat_calibration.py "
                  "--verify"),
        source=("training/training_data/laevitas_skew/skew_{btc,eth}_{25d,10d}"
                ".json + gex_{btc,eth}.json (6.6y Deribit 25d+10d skew and "
                "daily spot via the logged-in Laevitas DASHBOARD backend, "
                "P407) — operator-local, gitignored, not the apiv2 by-tenor "
                "series the seat reads live (P420 series caveat)"),
        staleness_days=180,
        staleness_reason=(
            "Six calendar-year eras; new data lands entirely in the open 2026 "
            "era the gate leans on. And the live series is a close cousin of "
            "this one (decisions agree ~46/59), so the archive the seat banks "
            "(data/laevitas_apiv2_skew_*.jsonl) should re-run the producer on "
            "the RUNTIME series once it is ~6 months deep."),
        revision_rule=(
            "Re-derive ONLY from the producer (exit 3 on drift, exit 2 when "
            "the operator-local data is absent). Never edited to make a trade "
            "pass (the P320 rule); a runtime-series recalibration is a new "
            "P-entry, not a refresh."),
        stamp_measured_on="_SKEW_MEASURED_ON",
        stamp_measured_by="_SKEW_MEASURED_BY",
    ),
    Calibration(
        symbol="core.cde_fees.CDE_FEE_BPS",
        measured_on="2026-08-26",
        producer=("python -X utf8 scripts/coinbase_probe_stop_support.py "
                  "  # venue PREVIEW quotes; fills via "
                  "scripts/fill_quality_review.py"),
        source=("data/fill_quality.jsonl (6 fills) + read-only CDE preview "
                "quotes 2026-08-20 (BTC/ETH/SOL) + XRP preview 2026-08-26 "
                "(P412, 9.03bps) — the pair is what refuted the "
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
        measured_on="2026-08-27",   # [P420] breadth XRP/BNB added, trio re-read as control
        producer=("python -X utf8 scripts/coinbase_probe_stop_support.py "
                  "  # read-only CDE order-book probe (P289)"),
        source=("live CDE order-book snapshots, 6 samples/contract: home trio "
                "on a WEEKEND book 2026-08-16 (deliberately the worst case); "
                "[P420] XRP + BNB via a read-only in-container get_product_book "
                "probe 2026-08-27 with the trio re-read as a control"),
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

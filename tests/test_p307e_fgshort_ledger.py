"""[P307e] The funding-gated-short variant's forward ledger.

P307d isolated the one structural difference between the live trend seat and
the book — trend is always in the market, the book refuses to short — and
measured, with the long leg held constant, that the UNCONDITIONAL short flips
sign across eras on BTC/ETH and is negative in both on SOL, while a short
gated on causal funding z > 1.0 is positive in both eras on all three.

That earns a forward exam and nothing more: the out-of-selection increments
are +0.007/+0.050/+0.009 and the threshold was designed on BTC. So the rule
ships as an observation-only ledger (Iron Law 7), and these tests pin the
properties that make its evidence readable.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return io.open(REPO / rel, encoding="utf-8").read()


class TestTheRule:
    def _f(self):
        from defense.regime_book_shadow import fgshort_target
        return fgshort_target

    def test_it_never_overrides_an_opinion_the_book_already_has(self):
        f = self._f()
        for bt in (1.0, -1.0, 0.5, -0.25):
            for fz in (None, -3.0, 0.0, 3.0):
                assert f(bt, "hold", fz) == (bt, "book"), (
                    "the variant must measure the INCREMENT over the book, "
                    "so a non-flat cell has to pass through untouched")

    def test_it_fills_a_chosen_flat_cell_above_the_gate(self):
        f = self._f()
        assert f(0.0, "trend_flat", 3.0) == (-1.0, "funding_gated_short")

    def test_it_leaves_a_chosen_flat_cell_alone_below_the_gate(self):
        f = self._f()
        assert f(0.0, "trend_flat", 1.0)[0] == 0.0, "the gate is strict >"
        assert f(0.0, "trend_flat", 0.9)[0] == 0.0

    def test_absence_is_never_converted_into_a_short(self):
        """The bug the BTC parity control caught on the first run: `warmup`
        and `flat_no_funding_history` are 0.0 meaning "no opinion is
        POSSIBLE", not "flat is the position". Filling them turns an absence
        into a live short — the P2 collapse."""
        f = self._f()
        for leg in ("warmup", "flat_no_funding_history"):
            tgt, why = f(0.0, leg, 99.0)
            assert tgt == 0.0, f"{leg} was filled with a short"
            assert "absent" in why or "no_funding" in why
        # and a genuinely missing z, on a cell the book DID evaluate
        assert f(0.0, "trend_flat", None) == (0.0, "flat_no_funding_history")

    def test_btc_is_a_parity_control_by_construction(self):
        """BTC already shorts every non-bull cell at z > 1.0 (bear leg) and
        from z > 0.5 (peace leg), so the variant CANNOT differ from the
        deployed book there. Any divergence in the BTC ledger is a bug, not a
        signal — which is what makes ETH and SOL carry the whole claim."""
        from defense.regime_book_shadow import book_target
        f = self._f()
        for regime in ("bull", "bear", "peace", "warmup"):
            for fz in (None, -3.0, -0.6, -0.4, 0.0, 0.4, 0.6, 0.9,
                       1.0, 1.1, 3.0):
                bt, leg = book_target("BTC", regime, fz)
                assert f(bt, leg, fz)[0] == pytest.approx(bt), (
                    f"BTC diverges at regime={regime} fz={fz}: book {bt} "
                    f"-> variant {f(bt, leg, fz)[0]}")

    @pytest.mark.parametrize("asset", ["ETH", "SOL"])
    def test_eth_and_sol_actually_gain_short_cells(self, asset):
        """Anti-vacuity (P174): if the variant changed nothing anywhere, the
        ledger would accrue a perfect copy of the book and the exam would be
        theatre."""
        from defense.regime_book_shadow import book_target
        f = self._f()
        filled = 0
        for regime in ("bear", "peace"):
            for fz in (1.1, 3.0):
                bt, leg = book_target(asset, regime, fz)
                if f(bt, leg, fz)[0] != bt:
                    filled += 1
        assert filled > 0, f"{asset}: the variant fills no cell at all"


class TestTheLedger:
    def test_it_is_registered_as_a_pooled_family_at_the_scorer(self):
        """P293g: a per-asset 16h exam needs ~330 days to certify; pooling
        three assets running ONE rule cuts it to ~123. Without this the
        candidate is on a clock that cannot fire."""
        from analytics.shadow_ic.compute_shadow_ic import POOLABLE_FAMILIES
        assert "regimebook_fgshort" in POOLABLE_FAMILIES

    def test_the_file_prefix_is_already_covered_by_the_glob(self):
        """The scorer globs `{prefix}_*.jsonl`, so `regimebook` matches
        `regimebook_fgshort_BTC.jsonl` — the same way `regimebook_adj` is
        picked up. Pinned so a future narrowing of the prefix list is caught
        (P264: registered-but-unreadable is the failure that mattered)."""
        from analytics.shadow_ic.compute_shadow_ic import load_shadow_ledgers
        prefixes = load_shadow_ledgers.__defaults__[0]
        assert "regimebook" in prefixes

    def test_the_write_is_fail_soft_and_cannot_lose_the_raw_record(self):
        src = _src("defense/regime_book_shadow.py")
        i = src.index('strategy="regimebook_fgshort"')
        blk = src[max(0, i - 900):i + 900]
        assert "except Exception" in blk
        assert "logger.warning" in blk
        # the raw record must already be on disk before the overlay runs
        assert src.index('path = self._dir / f"regimebook_{asset}.jsonl"') < i

    def test_the_record_carries_its_threshold_and_a_scoreable_confidence(self):
        """P236: the scorer multiplies direction x confidence, so a flat row
        must contribute zero rather than a saturated claim (P224). And the
        threshold travels with the row so a later change is auditable."""
        src = _src("defense/regime_book_shadow.py")
        i = src.index('strategy="regimebook_fgshort"')
        blk = src[i:i + 500]
        assert "confidence=abs(float(_fg))" in blk
        assert "fgshort_z=FGSHORT_FUNDING_Z" in blk

    def test_it_places_no_orders_and_takes_no_seat(self):
        """Iron Law 7. The seat reads `_last_records`, which only the RAW
        book writes; the variant must not touch it."""
        src = _src("defense/regime_book_shadow.py")
        i = src.index('strategy="regimebook_fgshort"')
        blk = src[i - 400:i + 900]
        assert "_last_records" not in blk, (
            "the variant is writing the seat stash — it is observation-only")

    def test_the_threshold_matches_the_measurement_that_justified_it(self):
        from defense.regime_book_shadow import FGSHORT_FUNDING_Z
        assert FGSHORT_FUNDING_Z == 1.0, (
            "P307d measured z > 1.0; a different threshold is a different "
            "candidate and needs its own lab run")

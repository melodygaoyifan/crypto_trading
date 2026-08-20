"""[P324] Guards for the advisor-disagreement instrument.

The load-bearing ones are the ABSENCE test (a silent advisor is not a
disagreeing advisor — P2) and the ANTI-GOALPOST test (the contrast statistic
is reported and must never drive the verdict, or the pre-commitment is
theatre).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import analytics.ic.agent_disagreement_review as adr  # noqa: E402

MOD = REPO / "analytics" / "ic" / "agent_disagreement_review.py"


def _src() -> str:
    return io.open(MOD, encoding="utf-8").read()


class TestSingleSource:
    """[P172] A second copy of fetch_closes is how two reviewers start
    disagreeing about which bar a signal belongs to — and this one carries the
    P265 in-progress-candle drop that the sibling had to be taught."""

    def test_fetchers_are_imported_not_restated(self):
        src = _src()
        assert "from analytics.ic.agent_ic_review import" in src
        for name in ("fetch_closes", "load_signal_records", "HORIZON_BARS",
                     "KRAKEN_PAIRS"):
            assert name in src
        assert "def fetch_closes" not in src, (
            "fetch_closes is reimplemented here; import it instead")
        assert "urllib" not in src, (
            "this module fetches prices itself — the point is that it does not")

    def test_the_horizon_set_is_the_shared_one(self):
        from analytics.ic.agent_ic_review import HORIZON_BARS
        assert adr.HORIZON_BARS is HORIZON_BARS


class TestAbsenceIsNotDisagreement:
    """P2, the repo's most common defect: a silent advisor must not be counted
    as disagreeing. If it were, every dark agent would look like a veto with
    perfect coverage, and the filter's own fail-OPEN semantics would be
    contradicted by its own evidence.

    Tested by CALLING classify_bucket — a source pin proves the code was
    written, not that it runs (P234/P307b).
    """

    def test_a_missing_advisor_is_silent_not_disagreeing(self):
        assert adr.classify_bucket(1.0, None)[0] == "advisor_silent"

    def test_a_zero_direction_advisor_is_silent_not_disagreeing(self):
        assert adr.classify_bucket(1.0, {"direction": 0.0})[0]             == "advisor_silent"
        assert adr.classify_bucket(-1.0, {"direction": 0.0})[0]             == "advisor_silent"

    def test_an_unreadable_advisor_is_silent_and_never_disagreement(self):
        """Unparseable is UNKNOWN. Resolving it to 'silent' makes the veto NOT
        fire — the same fail-OPEN direction the live filter takes on a dark
        agent — and it must never land in the disagree bucket."""
        for junk in ("abc", None, [1], {}):
            b = adr.classify_bucket(1.0, {"direction": junk})[0]
            assert b == "advisor_silent", junk

    def test_real_agreement_and_disagreement_still_classify(self):
        assert adr.classify_bucket(1.0, {"direction": 0.4})[0] == "agree"
        assert adr.classify_bucket(-1.0, {"direction": -0.4})[0] == "agree"
        assert adr.classify_bucket(1.0, {"direction": -0.4})[0] == "disagree"
        assert adr.classify_bucket(-1.0, {"direction": 0.4})[0] == "disagree"

    def test_a_flat_decider_is_skipped_entirely(self):
        """No entry means nothing for a veto to act on."""
        for dd in (0.0, 1e-12, -1e-12):
            assert adr.classify_bucket(dd, {"direction": 1.0})[0] is None

    def test_an_unreadable_decider_is_skipped_not_bucketed(self):
        assert adr.classify_bucket("nonsense", {"direction": 1.0})[0] is None

    def test_the_decider_direction_is_returned_for_the_sign(self):
        """main() signs the forward return with it; losing it would measure
        the |return|, not the decider's realized edge."""
        assert adr.classify_bucket(-0.7, {"direction": 1.0})[1]             == pytest.approx(-0.7)


    def test_the_note_reports_unreadable_records_so_the_counter_can_rise(self):
        """A dropped record nobody counts is a silent failure, and a counter
        that cannot rise is worse than none (P174)."""
        assert adr.classify_bucket("junk", {"direction": 1.0})[3]             == "bad_decider"
        assert adr.classify_bucket(1.0, {"direction": "junk"})[3]             == "bad_advisor"
        assert adr.classify_bucket(1.0, {"direction": 0.4})[3] == ""
        assert adr.classify_bucket(0.0, {"direction": 0.4})[3] == ""

    def test_main_counts_both_notes(self):
        src = _src()
        i = src.index("def main() -> int:")
        body = src[i:]
        assert "bad_decider += 1" in body and "bad_advisor += 1" in body
        assert "unparseable directions" in body


class TestOverlapCorrection:
    """[P231] An h-bar return sampled every bar overlaps h times. Without the
    correction the t is inflated by ~sqrt(h) — the artifact that made P236's
    headline read t=-3.42, and P230's model_alpha 16h read t=4.4."""

    def test_n_eff_divides_by_the_horizon(self):
        vals = [1.0, -1.0] * 40
        one = adr._stats(vals, 1)
        four = adr._stats(vals, 4)
        assert one["n"] == four["n"] == 80
        assert one["n_eff"] == pytest.approx(80.0)
        assert four["n_eff"] == pytest.approx(20.0)

    def test_the_correction_only_ever_shrinks_significance(self):
        vals = [5.0 + (i % 7) for i in range(120)]
        assert abs(adr._stats(vals, 4)["t"]) < abs(adr._stats(vals, 1)["t"])

    def test_degenerate_inputs_do_not_fabricate_a_t(self):
        assert adr._stats([], 1)["t"] is None
        assert adr._stats([3.0], 1)["t"] is None
        assert adr._stats([2.0] * 50, 1)["t"] is None  # zero variance


class TestTheContrastIsReportedNotDeciding:
    """ANTI-GOALPOST. The contrast was added AFTER seeing the pre-committed
    rule fail. It is the better lens for what a filter exploits, and it agreed
    with the rule — but if it ever entered the verdict, the pre-commitment
    would be decoration."""

    def test_the_verdict_never_reads_the_contrast(self):
        src = _src()
        i = src.index("# ---- the pre-committed verdict")
        # scope to the COMPUTATION: the JSON payload below it reports the
        # contrast on purpose, which is not the same as deciding on it.
        j = src.index("    out = {", i)
        verdict_block = src[i:j]
        assert "contrast" not in verdict_block.lower(), (
            "the verdict block reads the post-hoc contrast; the pre-committed "
            "rule must stand on its own")

    def test_the_contrast_is_labelled_post_hoc_where_it_is_emitted(self):
        src = _src()
        assert "NOT the pre-committed" in src
        assert "REPORTED, NOT PRE-COMMITTED" in src

    def test_the_contrast_is_a_two_sample_statistic(self):
        """It must compare the buckets, not test one level against zero."""
        good = adr._contrast([10.0 + (i % 3) for i in range(80)],
                             [-10.0 + (i % 3) for i in range(80)], 1)
        assert good["delta_bps"] == pytest.approx(20.0, abs=1.0)
        assert good["t"] is not None and good["t"] > 5
        none = adr._contrast([1.0], [2.0], 1)
        assert none["t"] is None


class TestTheVerdictRule:
    """All three pre-committed conditions must bind, and each alone must be
    able to block — else the rule is looser than it reads. Tested by CALLING
    decide_verdict (P234/P307b)."""

    H = (1,)

    def _pooled(self, dis_mean, dis_t, agree_mean, n=200):
        return {
            ("disagree", 1): {"n": n, "mean_bps": dis_mean, "t": dis_t},
            ("agree", 1): {"n": n, "mean_bps": agree_mean, "t": 0.1},
        }

    def test_all_three_conditions_together_earn_it(self):
        v, reasons, earned = adr.decide_verdict(
            self._pooled(-40.0, -3.0, +10.0), self.H)
        assert v == "EARNED" and earned == [1] and not reasons

    def test_a_positive_disagree_bucket_blocks(self):
        v, reasons, _ = adr.decide_verdict(
            self._pooled(+40.0, +3.0, +60.0), self.H)
        assert v == "NOT_EARNED"
        assert any("not negative" in r for r in reasons)

    def test_an_insignificant_t_blocks(self):
        """The live case: whale h1 -9.1bps at t=-1.79, agree better — and
        still NOT EARNED."""
        v, reasons, _ = adr.decide_verdict(
            self._pooled(-9.1, -1.79, -0.9), self.H)
        assert v == "NOT_EARNED"
        assert any("1.79" in r for r in reasons)

    def test_agree_no_better_than_disagree_blocks(self):
        v, reasons, _ = adr.decide_verdict(
            self._pooled(-40.0, -3.0, -50.0), self.H)
        assert v == "NOT_EARNED"
        assert any("agree not better" in r for r in reasons)

    def test_a_missing_t_cannot_pass(self):
        v, _, _ = adr.decide_verdict(
            {("disagree", 1): {"n": 200, "mean_bps": -40.0, "t": None},
             ("agree", 1): {"n": 200, "mean_bps": +10.0, "t": 0.1}}, self.H)
        assert v == "NOT_EARNED"

    def test_too_few_disagreements_cannot_produce_a_verdict(self):
        """Thin evidence must read as 'cannot judge', never as 'earned'."""
        v, reasons, _ = adr.decide_verdict(
            self._pooled(-99.0, -9.0, +50.0, n=adr.MIN_DISAGREE_N - 1),
            self.H)
        assert v == "NOT_EARNED"
        assert any("cannot judge" in r for r in reasons)
        assert adr.MIN_DISAGREE_N >= 30

    def test_an_empty_pooled_dict_earns_nothing(self):
        assert adr.decide_verdict({}, self.H)[0] == "NOT_EARNED"

    def test_exit_codes_are_distinct(self):
        """[P199/P213] 'refused' must never read as 'not earned'."""
        src = _src()
        assert "return 3 if verdict" in src
        assert "raise SystemExit(2)" in src

    def test_a_silent_advisor_refuses_rather_than_reporting_unearned(self):
        src = _src()
        i = src.index("seen_advisor == 0")
        blk = src[i:i + 400]
        assert "_refuse(" in blk
        assert "a wiring gap rather than a verdict" in blk


class TestTheEraSplitSurvives:
    """[P320c] The decider's identity changed inside the window. A pooled
    number that cannot be decomposed is a claim about a retired seat."""

    def test_seat_eras_cover_the_known_changes(self):
        names = [n for n, _, _ in adr.SEAT_ERAS]
        assert names == ["trend_seat", "whale_seat", "book_seat"]
        assert adr._era("2026-07-01") == "trend_seat"
        assert adr._era("2026-08-17") == "whale_seat"
        assert adr._era("2026-08-19") == "book_seat"

    def test_the_output_states_the_composition_limit(self):
        src = _src()
        assert "identity changed mid-window" in src


class TestEndToEnd:
    """A refusal must be a real process exit, not an inferred one (P185)."""

    def test_missing_log_dir_exits_two(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(MOD),
             "--log-dir", str(tmp_path / "nope")],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO),
            timeout=180)
        assert r.returncode == 2, r.stdout[-400:]


class TestTheSeamsAreNotDecorative:
    """[P170/P312] A seam nothing calls is decoration. main() must route
    through both extracted functions, or the behavioural tests above are
    testing code the tool does not run."""

    def test_main_calls_both_pure_functions(self):
        src = _src()
        i = src.index("def main() -> int:")
        body = src[i:]
        assert "classify_bucket(" in body
        assert "decide_verdict(pooled, HORIZON_BARS)" in body

    def test_main_does_not_reimplement_the_bucketing(self):
        src = _src()
        i = src.index("def main() -> int:")
        body = src[i:]
        assert 'bucket = "disagree"' not in body, (
            "main() classifies inline again; one implementation only (P172)")


class TestTheBookDeciderMode:
    """[P337] Measuring the filters against the decider they actually filter.

    The filters were written off as "structurally forward-only because they
    judge AGENT OUTPUTS, which were never stored". Half right, and it cost
    three weeks of assumed waiting: the ADVISOR is stored (130 days of
    attribution), and the DECIDER is the regimebook, whose direction is
    deterministic from price and funding — so the pair reconstructs backwards.
    """

    def test_the_book_series_is_imported_from_the_lab_not_restated(self):
        """[P172] A second copy of book_target would measure a book that is
        not the deployed one."""
        src = _src()
        assert "import training.funding_legs_lab as lab" in src
        assert "lab.build_positions" in src
        assert "def book_target" not in src

    def test_a_record_before_the_book_history_is_SKIPPED_not_flattened(self):
        """P2: absence of a book direction is not a flat book — counting it as
        flat would put pre-history ticks in the agree/disagree buckets."""
        src = _src()
        i = src.index("bi = bisect_right(bts, rec[")
        blk = src[max(0, i - 400):i + 260]
        assert "if bi < 0:" in blk and blk.split("if bi < 0:")[1].lstrip().startswith("continue")
        assert "never defaulted to flat" in blk

    def test_the_era_caption_does_not_claim_a_decider_change_in_book_mode(self):
        """Under --decider book the decider is CONSTANT, so the seat-history
        caption would describe a different measurement than the one that ran."""
        import sys as _s
        from pathlib import Path as _P
        _s.path.insert(0, str(_P(__file__).resolve().parent))
        from _guard_pins import assert_guard_live
        src = _src()
        assert "PER DATE BAND" in src
        # the caption must be gated on the mode, and gated LIVE: a substring
        # pin would survive `if False and book is not None` (P234/P307b), and
        # the precise detector rightly flagged the first draft of this line.
        assert_guard_live(src, "book is not None", near="PER DATE BAND",
                          why="the date-band caption must be mode-gated")

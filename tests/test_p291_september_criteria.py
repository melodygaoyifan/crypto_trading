"""[P291] The September criteria that existed only in prose, now pinned.

Two gaps closed here:
  1. The P288/P289 trend-rule challengers (donchian/emaens) had forward
     ledgers accruing with NO pre-committed promotion criterion and no
     instrument — the P230 shape ("a bar without an instrument becomes
     whoever re-greps by hand").
  2. The sizing ladder had no pre-commitment at all: nothing said what
     happens to `coinbase_target_fraction_by_asset` when something certifies.

The load-bearing pins, and why each exists:
  * The criterion must be able to fire AND to refuse — a condition that
     cannot fail is not a condition (P174), and pinning that code merely
     EXISTS is what let the P285 checker ship with a structurally
     unevaluated kill-screen (P287).
  * `ic_per_horizon` is the scorer's SUCCESS key; `ic_per_h` is its ERROR
     key. Reading the wrong one is the exact P287 finding.
  * Refusals are asserted at real EXIT-CODE level through a subprocess —
     asserting a return value inside the process misses the P185
     pipeline-exit trap.
  * The live fractions are pinned at 0.15 (P237's decided-value pattern) so
     a silent raise fails loudly rather than arriving as a config diff
     nobody re-derived.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.challenger_seat_check import (  # noqa: E402
    CRASH_DODGE_CAVEAT, EARLIEST, LAB_DETHRONED, MIN_DIRECTIONAL,
    MIN_SPAN_DAYS, _ic_from_stats, _min_ic, decide,
)

SCRIPT = REPO / "scripts" / "challenger_seat_check.py"
DOC = REPO / "docs" / "SEPTEMBER_DECISION_TREE.md"
LIVE_PROFILE = REPO / "configs" / "live_high_risk.json"

# A cell the P288 lab actually dethroned, and the all-conditions-good inputs.
GOOD_CELL = ("donchian", "SOL")
GOOD = dict(
    today=date(2026, 9, 20),
    span_days=31.0,
    n_directional=40,
    verdict="PROMOTE",
    challenger_min_ic=0.12,
    incumbent_min_ic=0.04,
)


def _fires(cell=GOOD_CELL, **over):
    kw = dict(GOOD)
    kw.update(over)
    ok, _ = decide(cell, **kw)
    return ok


class TestCriterionFiresAndRefuses:
    """Both directions — the P287 lesson: pin that a condition CAN fire."""

    def test_all_conditions_good_is_eligible(self):
        assert _fires() is True

    @pytest.mark.parametrize("field,bad", [
        ("today", date(2026, 9, 15)),          # one day before the gate
        ("span_days", MIN_SPAN_DAYS - 0.1),
        ("n_directional", MIN_DIRECTIONAL - 1),
        ("verdict", "HOLD"),
        ("verdict", "KILL"),
        ("verdict", "INSUFFICIENT_SAMPLES"),
        ("challenger_min_ic", 0.04),           # ties the incumbent's 0.04
        ("challenger_min_ic", 0.01),           # below the incumbent
    ])
    def test_each_condition_can_block(self, field, bad):
        assert _fires(**{field: bad}) is False

    @pytest.mark.parametrize("field", [
        "verdict", "challenger_min_ic", "incumbent_min_ic"])
    def test_missing_data_never_passes(self, field):
        # P199: None means "could not be evaluated", which is not "passed".
        assert _fires(**{field: None}) is False

    def test_boundaries_are_inclusive_where_stated(self):
        assert _fires(today=EARLIEST) is True
        assert _fires(span_days=MIN_SPAN_DAYS) is True
        assert _fires(n_directional=MIN_DIRECTIONAL) is True

    def test_beats_incumbent_is_strict(self):
        # A tie is not a win: swapping the seat on equal evidence is churn.
        assert _fires(challenger_min_ic=0.05, incumbent_min_ic=0.05) is False
        assert _fires(challenger_min_ic=0.0501, incumbent_min_ic=0.05) is True

    def test_every_condition_is_reported_with_a_detail(self):
        _, conds = decide(GOOD_CELL, **GOOD)
        assert len(conds) == 6
        for name, _ok, detail in conds:
            assert name and isinstance(detail, str) and detail


class TestLabPrecondition:
    """The dethroning map is per-CELL, not per-asset — the sharpest edge."""

    def test_the_map_is_exactly_the_three_dethroned_cells(self):
        assert set(LAB_DETHRONED) == {
            ("donchian", "ETH"), ("donchian", "SOL"), ("emaens", "SOL")}

    def test_emaens_eth_cannot_be_eligible_though_eth_was_dethroned(self):
        # P288 per-cell: emaens design -0.231 vs the incumbent's +0.088.
        # An asset-level "ETH+SOL" reading would wrongly admit this cell.
        assert _fires(cell=("emaens", "ETH")) is False

    def test_no_btc_cell_can_be_eligible(self):
        # SMA200 STOOD on BTC (donchian +0.582, emaens +0.467 vs +0.594).
        for strat in ("donchian", "emaens"):
            assert _fires(cell=(strat, "BTC")) is False

    def test_a_forward_pass_on_a_non_dethroned_cell_still_fails(self):
        # The whole point: forward evidence alone is not a seat claim.
        ok, conds = decide(("emaens", "ETH"), **GOOD)
        assert ok is False
        blocked = [n for n, o, _ in conds if not o]
        assert blocked == ["lab_dethroned"], blocked

    def test_per_asset_independence(self):
        # donchian/SOL eligible while donchian/BTC is not, same inputs.
        assert _fires(cell=("donchian", "SOL")) is True
        assert _fires(cell=("donchian", "BTC")) is False


class TestScorerKeyIsTheSuccessShape:
    """[P287] `ic_per_horizon` is the success key; `ic_per_h` is the error key."""

    def test_reads_ic_per_horizon(self):
        st = {"ic_per_horizon": {4: 0.11, 12: 0.09, 24: 0.07}}
        assert _ic_from_stats(st, 4) == pytest.approx(0.11)
        assert _min_ic(st) == pytest.approx(0.07)

    def test_string_keys_accepted(self):
        st = {"ic_per_horizon": {"4": 0.11, "12": 0.09, "24": 0.07}}
        assert _min_ic(st) == pytest.approx(0.07)

    def test_missing_horizon_makes_min_ic_none_not_zero(self):
        # A partially-scored record must not silently read as IC 0.0 — that
        # would compare a challenger against the incumbent on a fiction.
        st = {"ic_per_horizon": {4: 0.11}}
        assert _min_ic(st) is None

    def test_source_reads_the_success_key_first(self):
        src = SCRIPT.read_text(encoding="utf-8")
        body = src.split('"""', 2)[-1]  # strip the module docstring
        i_success = body.index('st.get("ic_per_horizon")')
        i_error = body.index('st.get("ic_per_h")')
        assert i_success < i_error, (
            "the ERROR-shape key must only ever be a fallback; reading it "
            "first is the P287 defect")

    def test_error_record_shape_does_not_masquerade_as_a_score(self):
        # The scorer's ohlcv_missing record carries ic_per_h = {} — empty,
        # so it must yield None (unevaluated), never a number.
        assert _min_ic({"n": 0, "ic_per_h": {}, "error": "ohlcv_missing"}) is None


class TestRefusalExitCodes:
    """[P185] Assert at real exit-code level, through a subprocess."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(SCRIPT), *args],
            capture_output=True, text=True, cwd=str(REPO))

    def test_missing_pulled_dir_refuses_with_2(self, tmp_path):
        r = self._run("--ledger-dir", str(tmp_path / "nope"))
        assert r.returncode == 2, r.stdout + r.stderr
        assert "CANNOT BE EVALUATED" in r.stderr

    def test_empty_dir_refuses_with_2_and_says_no_ledger_is_not_a_verdict(
            self, tmp_path):
        d = tmp_path / "strategy_shadow_pulled"
        d.mkdir()
        r = self._run("--ledger-dir", str(d))
        assert r.returncode == 2, r.stdout + r.stderr
        assert "not" in r.stderr.lower() and "eligible" in r.stderr.lower()

    def test_refusal_is_distinct_from_not_yet(self, tmp_path):
        # 2 (cannot evaluate) and 0 (not yet) must never collapse — that
        # conflation is the P199 defect this instrument is built against.
        d = tmp_path / "pulled"
        d.mkdir()
        (d / "donchian_SOL.jsonl").write_text(
            json.dumps({"ts": 1.0, "strategy": "donchian", "asset": "SOL",
                        "direction": 0.0, "confidence": 0.0}) + "\n",
            encoding="utf-8")
        r = self._run("--ledger-dir", str(d))
        # Either a refusal (no OHLCV locally) or "not yet" — never 3.
        assert r.returncode in (0, 2), r.stdout + r.stderr
        assert r.returncode != 3


class TestNeverActs:
    """P141: the checker states a verdict; a human edits config."""

    def test_script_never_writes_config(self):
        src = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("json.dump(", "write_text(", ".write(", "open("):
            # `open(` appears in no write mode here; assert no write verbs.
            assert f"config{forbidden}" not in src
        assert "live_high_risk" not in src.split('"""', 2)[-1], (
            "the checker must not touch the live profile, even by path")

    def test_eligible_verdict_is_a_preference_decision_not_a_fire(self):
        src = SCRIPT.read_text(encoding="utf-8")
        assert "ELIGIBLE" in src
        assert "NOT A FIRE INSTRUCTION" in src
        assert "RISK-PREFERENCE" in src.upper()

    def test_the_measured_tradeoff_travels_with_the_verdict(self):
        # The caveat must carry the NUMBERS — a verdict citing a memory is
        # how a pre-commitment quietly becomes a vibe.
        for token in ("-0.60", "-0.51", "-0.28", "-0.35", "2x"):
            assert token in CRASH_DODGE_CAVEAT, token


class TestDocPreCommitments:
    def test_doc_has_the_challenger_section_and_its_criteria(self):
        doc = DOC.read_text(encoding="utf-8")
        assert "challenger_seat_check.py" in doc
        assert "2026-09-16" in doc
        assert "lab precondition" in doc.lower()
        # the per-cell precision the asset-level framing loses
        assert "emaens/ETH" in doc and "NOT eligible" in doc

    def test_doc_has_the_sizing_ladder_and_its_governing_sentence(self):
        doc = DOC.read_text(encoding="utf-8")
        assert "size follows certification, never precedes it" in doc
        assert "coinbase_target_fraction_by_asset" in doc
        # the arithmetic that makes the ladder non-obvious must be shown
        assert "0.50" in doc and "zero headroom" in doc

    def test_doc_states_every_september_outcome_for_fractions(self):
        doc = DOC.read_text(encoding="utf-8")
        sizing = doc.split("## The sizing ladder")[1].split("\n## ")[0]
        for outcome in ("nothing certifies", "still live", "removes **one**",
                        "removes **two**", "all three"):
            assert outcome in sizing, outcome


class TestLiveFractionsUnchanged:
    """[P237] Pin the DECIDED value: a silent raise must fail loudly."""

    def test_fractions_are_still_015(self):
        prof = json.loads(LIVE_PROFILE.read_text(encoding="utf-8-sig"))
        fr = prof["coinbase_target_fraction_by_asset"]
        assert fr == {"BTC": 0.15, "ETH": 0.15, "SOL": 0.15}, (
            "the sizing ladder (P291) pre-commits that fractions only move "
            "AFTER a certification, per the doc's table — if this changed, "
            "the change needs its own recorded P-entry")

    def test_nominal_max_net_stays_under_the_p208_cap(self):
        prof = json.loads(LIVE_PROFILE.read_text(encoding="utf-8-sig"))
        fr = prof["coinbase_target_fraction_by_asset"]
        cap = prof["risk"]["max_net_exposure"]
        assert sum(fr.values()) <= cap, (
            f"nominal max net {sum(fr.values())} exceeds the P208 cap {cap}")
        # and keeps real headroom, not just "not blocked" (the gate is `>`)
        assert sum(fr.values()) <= cap - 0.04

    def test_no_fraction_exceeds_its_per_asset_cap(self):
        prof = json.loads(LIVE_PROFILE.read_text(encoding="utf-8-sig"))
        fr = prof["coinbase_target_fraction_by_asset"]
        caps = prof["post_leverage_caps"]
        for asset, f in fr.items():
            assert f <= caps[asset], asset
            assert f <= 0.25, f"{asset}: P274 ctor clamp"

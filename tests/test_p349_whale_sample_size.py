"""[P349] The whale deadband screens BALANCED flow and cannot screen THIN flow.

`net_pressure = (buy_vol - sell_vol) / (buy_vol + sell_vol)` over the whales
detected in the last hour, so it is a RATIO. One whale gives exactly +/-1.0.
Two on the same side give +/-1.0. Both clear the +/-0.3 deadband at full
conviction. Measured on the live logs: median 3 whales per (hour, asset) and
53 of 123 buckets hold <= 2.

The field that WOULD express "enough big money", `whale_count`, is computed by
the detector and published into both signal dicts, and is read by nothing that
makes a decision (P144/P170: computed but unenforced).

This change is OBSERVATION ONLY -- the count is recorded in the whale_filter
ledger so the question becomes measurable. Gating the direction on it would
make the armed filter fire LESS, which is a loosening and the operator's call
(P141), and P348 says the filter has no measured basis to be armed anyway.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from defense.strategy_shadow_v5_1 import MAFilterEchoStrategy  # noqa: E402

MAIN = REPO / "main.py"


def _main_src() -> str:
    return io.open(MAIN, encoding="utf-8").read()


def _whale_dir():
    """Import the pure function without importing main.py (heavy)."""
    import ast
    import textwrap
    src = _main_src()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and \
                node.name == "whale_direction_from_pressure":
            seg = ast.get_source_segment(src, node)
            ns: dict = {}
            exec(textwrap.dedent(seg), ns)  # noqa: S102 - the function itself
            return ns["whale_direction_from_pressure"]
    raise AssertionError("whale_direction_from_pressure not found")


class TestTheDeadbandCannotSeeSampleSize:
    """The finding, pinned as a property rather than as an anecdote."""

    def test_a_single_whale_emits_FULL_conviction(self):
        """With one whale the ratio is exactly +/-1.0, so the deadband is
        cleared by the thinnest possible sample."""
        f = _whale_dir()
        assert f(1.0) == 1.0
        assert f(-1.0) == -1.0

    def test_the_deadband_screens_balance_not_thinness(self):
        f = _whale_dir()
        # balanced flow -> screened, which is what the deadband is FOR
        assert f(0.1) == 0.0
        assert f(-0.29) == 0.0
        # one-sided flow -> passes, whether it is 1 whale or 400
        assert f(0.9) == 1.0

    def test_the_direction_function_takes_no_count_argument(self):
        """It cannot gate on sample size because it is never given one."""
        import inspect
        assert list(inspect.signature(_whale_dir()).parameters) == [
            "net_pressure"]


class TestTheLedgerRecordsTheSampleSize:

    def _row(self, name, md):
        return MAFilterEchoStrategy(strategy_name=name).evaluate("ETH", md)

    BASE = {"_maf_ma_dir": -1.0, "_maf_ledger_dir": 0.0, "_maf_reason": "x"}

    def test_whale_rows_carry_count_and_pressure(self):
        d = self._row("whale_filtered",
                      dict(self.BASE, _maf_whale_count=2,
                           _maf_whale_pressure=1.0)).diagnostics
        assert d["whale_count"] == 2
        assert d["whale_pressure"] == 1.0

    def test_ma_rows_do_not_gain_a_NULL_count(self):
        """A null would read as 'measured zero whales' rather than 'not
        applicable' -- the missing-vs-neutral collapse (P2), one level down."""
        d = self._row("ma_filtered", dict(self.BASE)).diagnostics
        assert "whale_count" not in d
        assert "whale_pressure" not in d

    def test_the_diagnostics_dict_is_a_whitelist(self):
        """Anti-vacuity: this is WHY the keys had to be added at the producer.
        If it ever stops being a whitelist the conditional add is redundant,
        and this test says so instead of passing silently."""
        d = self._row("whale_filtered",
                      dict(self.BASE, _maf_not_a_real_key=123)).diagnostics
        assert "not_a_real_key" not in d and 123 not in d.values()


class TestObservationOnly:
    """Nothing may DECIDE on the stashed count -- that would change live
    behaviour on an armed filter (P141)."""

    # A line DECIDES on the count only if it compares the stash's VALUE. The
    # `hasattr` existence guard is not that, and a detector that cannot tell
    # the two apart is one an author silences rather than obeys -- the
    # over-broad-detector mistake this repo has now made three times (P307
    # prose, P330's `--`, and the first draft of this very test).
    _CMP = r"(==|!=|<=|>=|<|>)"

    def _decision_lines(self, src):
        out = []
        for m in re.finditer(r"_last_whale_counts", src):
            start = src.rfind("\n", 0, m.start()) + 1
            end = src.find("\n", m.start())
            line = src[start:end if end != -1 else len(src)]
            if "hasattr(" in line:
                continue
            if re.search(self._CMP, line):
                out.append(line.strip())
        return out

    def test_the_count_stash_is_never_read_by_a_decision(self):
        src = _main_src()
        assert src.count("_last_whale_counts") >= 3, "the stash vanished"
        bad = self._decision_lines(src)
        assert not bad, (
            "a decision is being made on the whale sample size; that is a live "
            "behaviour change on an armed filter (P141), not the "
            "observation-only wiring this entry shipped: " + repr(bad))

    def test_that_guard_would_actually_catch_a_real_gate(self):
        """Anti-vacuity (P174): a detector that cannot fire reports clean, and
        clean is what a healthy tree also reports."""
        synthetic = (
            'x = 1\n'
            '        if int(getattr(self, "_last_whale_counts", {}).get(a, 0)) < 3:\n'
            '            _wf_dir = 0.0\n')
        assert self._decision_lines(synthetic), (
            "the observation-only guard cannot detect a real gate")

    def test_the_guard_does_not_fire_on_the_hasattr_init(self):
        benign = 'y = 2\n        if not hasattr(self, "_last_whale_counts"):\n'
        assert not self._decision_lines(benign)

    def test_both_the_reset_and_the_set_half_exist(self):
        """[P155-L5/P294] A stash with no per-tick reset carries the previous
        tick's value into a ledger row that claims to describe this one."""
        src = _main_src().replace("\r\n", "\n")
        # Pinned by LOCATION, not by substring: the SET half's exception
        # fallback also assigns 0, so `"... = 0" in src` stays true with the
        # reset deleted -- a count is not a location (P293b), and the
        # falsification probe caught exactly that in this test's first draft.
        anchor = "self._last_whale_directions[asset] = 0.0"
        assert src.count(anchor) == 1, "the sibling reset moved; re-anchor"
        block = src[src.index(anchor):src.index(anchor) + 1500]
        assert "self._last_whale_counts[asset] = 0" in block, (
            "the count stash has no per-tick reset beside the direction reset "
            "it describes; without it a ledger row carries the previous "
            "tick's sample size (P155-L5/P294)")
        assert "_last_whale_counts[asset] = int(" in src, "no set half"

    def test_the_count_comes_from_the_producer_key(self):
        src = _main_src()
        assert "market_data.get('whale_count', 0)" in src


class TestThePremiseHolds:
    """If whale_count ever gains a real consumer this entry's framing is
    stale and the roster of 'computed but unenforced' shrinks."""

    def test_whale_count_still_has_no_decision_consumer(self):
        import subprocess
        out = subprocess.run(
            ["git", "grep", "-n", "whale_count", "--", "*.py"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
        hits = [l for l in out.stdout.splitlines()
                if "/tests/" not in l and not l.startswith("tests/")]
        # every surviving hit must be a write, a log, a diag record or the
        # detector's own bookkeeping -- never a comparison
        for line in hits:
            body = line.split(":", 2)[-1]
            if re.search(r"if .*whale_count|whale_count\s*[<>]=?|>=?\s*whale_count",
                         body):
                assert "whale_dir" not in body, body

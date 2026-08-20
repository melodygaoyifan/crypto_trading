"""[P350] Four lessons from P349, each converted into a mechanism.

P349 recorded four things as prose and fixed each one instance-at-a-time:
  1. a ledger whitelist that DROPPED an observation key silently
  2. a pin whose needle a sibling occurrence also satisfied (vacuous)
  3. a detector that was over-broad and would have been loosened to fit
  4. a commit that nearly swept a parallel session's uncommitted hunks

A lesson recorded in prose is the thing this repo exists to convert into a
mechanism (P280/P328), so each now has one and each has a falsification probe.
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests._guard_pins import (  # noqa: E402
    assert_detector_is_precise, assert_text_pin)
from tools.isolate_commit import (  # noqa: E402
    describe_dropped, foreign_markers, select_hunks, split_hunks)


# --------------------------------------------------------------------------
# 1. The pin that a sibling occurrence satisfies
# --------------------------------------------------------------------------
class TestAssertTextPinRefusesAmbiguity:
    """The P349 trap: `"x = 0" in src` stayed green with the site deleted,
    because the exception fallback assigns the same text. A count is not a
    location (P293b)."""

    SRC = (
        "def reset(self):\n"
        "    self._counts[a] = 0\n"
        "\n"
        "def setter(self):\n"
        "    try:\n"
        "        self._counts[a] = int(md['n'])\n"
        "    except ValueError:\n"
        "        self._counts[a] = 0\n")

    def test_an_ambiguous_needle_is_an_ERROR_not_a_pass(self):
        with pytest.raises(AssertionError, match="appears 2 times"):
            assert_text_pin(self.SRC, "self._counts[a] = 0")

    def test_near_scopes_it_to_the_site_you_meant(self):
        assert_text_pin(self.SRC, "self._counts[a] = 0", near="def reset")

    def test_it_fails_when_that_site_is_deleted(self):
        """The whole point: with the reset gone, the sibling must NOT rescue
        the pin — provided the sibling is outside the window, which is the
        caller's judgement (see the helper's WHAT IT DOES NOT DO)."""
        far = self.SRC.replace("def setter", "#" * 600 + "\ndef setter")
        assert_text_pin(far, "self._counts[a] = 0", near="def reset")
        broken = far.replace("    self._counts[a] = 0\n", "", 1)
        with pytest.raises(AssertionError):
            assert_text_pin(broken, "self._counts[a] = 0", near="def reset")

    def test_the_window_cannot_exclude_an_ADJACENT_sibling(self):
        """The limit, pinned rather than hidden. A scoped pin whose sibling
        sits inside the window still passes with the real site deleted, so
        the ambiguity REFUSAL — not `near` — is the load-bearing half."""
        broken = self.SRC.replace("    self._counts[a] = 0\n", "", 1)
        assert_text_pin(broken, "self._counts[a] = 0", near="def reset")

    def test_a_unique_needle_needs_no_anchor(self):
        assert_text_pin(self.SRC, "int(md['n'])")

    def test_an_ambiguous_ANCHOR_is_refused(self):
        """P238/P337: an anchor that occurs twice scopes the pin to a window
        you did not read."""
        with pytest.raises(AssertionError, match="anchor"):
            assert_text_pin(self.SRC, "int(md['n'])", near="self._counts[a]")

    def test_a_missing_needle_still_fails(self):
        with pytest.raises(AssertionError, match="does not appear"):
            assert_text_pin(self.SRC, "nope_not_here")


# --------------------------------------------------------------------------
# 2. The detector that would have been loosened to fit
# --------------------------------------------------------------------------
class TestAssertDetectorIsPrecise:
    """P307 (prose), P330 (`--`), P349 (`hasattr`) were all over-broad
    detectors, and the tempting fix is always to relax until your own case
    passes (P248)."""

    def test_it_catches_an_OVER_broad_detector(self):
        over = lambda s: "count" in s  # noqa: E731 - fires on the init too
        with pytest.raises(AssertionError, match="FIRED on"):
            assert_detector_is_precise(
                over,
                must_catch=["if counts[a] < 3:"],
                must_not_catch=['if not hasattr(self, "counts"):'])

    def test_it_catches_an_UNDER_broad_detector(self):
        """Recall is the direction that goes quiet (P174)."""
        under = lambda s: False  # noqa: E731
        with pytest.raises(AssertionError, match="FAILED TO CATCH"):
            assert_detector_is_precise(
                under, must_catch=["if counts[a] < 3:"], must_not_catch=["x"])

    def test_a_precise_detector_passes_both_ways(self):
        good = lambda s: "counts" in s and "<" in s  # noqa: E731
        assert_detector_is_precise(
            good,
            must_catch=["if counts[a] < 3:"],
            must_not_catch=['if not hasattr(self, "counts"):'])

    @pytest.mark.parametrize("catch,notcatch", [([], ["x"]), (["x"], [])])
    def test_both_directions_are_REQUIRED(self, catch, notcatch):
        """Declaring only one half is how the other half rots."""
        with pytest.raises(AssertionError):
            assert_detector_is_precise(lambda s: True, catch, notcatch)


# --------------------------------------------------------------------------
# 3. The whitelist that dropped a key silently
# --------------------------------------------------------------------------
class TestTheLedgerReportsWhatItDrops:

    def _echo(self):
        from defense.strategy_shadow_v5_1 import MAFilterEchoStrategy
        return MAFilterEchoStrategy(strategy_name="whale_filtered")

    BASE = {"_maf_ma_dir": -1.0, "_maf_ledger_dir": 0.0, "_maf_reason": "x"}

    def test_an_unknown_observation_key_is_reported(self, caplog):
        e = self._echo()
        with caplog.at_level(logging.WARNING):
            e.evaluate("ETH", dict(self.BASE, _maf_brand_new=7))
        assert any("_maf_brand_new" in r.message and "DROPPED" in r.message
                   for r in caplog.records), (
            "a key the caller sends and the whitelist does not name vanished "
            "silently — that is the defect P349 hit")

    def test_it_warns_ONCE_per_key(self, caplog):
        e = self._echo()
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                e.evaluate("ETH", dict(self.BASE, _maf_brand_new=7))
        hits = [r for r in caplog.records if "_maf_brand_new" in r.message]
        assert len(hits) == 1, f"per-tick wallpaper: {len(hits)} warnings"

    def test_known_keys_are_silent(self, caplog):
        e = self._echo()
        with caplog.at_level(logging.WARNING):
            e.evaluate("ETH", dict(self.BASE, _maf_whale_count=2,
                                   _maf_whale_pressure=1.0))
        assert not [r for r in caplog.records if "DROPPED" in r.message]

    def test_a_dropped_key_never_breaks_the_tick(self):
        """A ledger must not be able to raise into a live path (Iron Law 7)."""
        sig = self._echo().evaluate("ETH", dict(self.BASE, _maf_weird=object()))
        assert sig is not None


# --------------------------------------------------------------------------
# 4. The commit that nearly swept a parallel session's work
# --------------------------------------------------------------------------
class TestIsolateCommit:

    DIFF = (
        "diff --git a/main.py b/main.py\n"
        "--- a/main.py\n"
        "+++ b/main.py\n"
        "@@ -1,3 +1,4 @@\n"
        " ctx\n"
        "+    # [P350] mine\n"
        " ctx\n"
        "@@ -10,3 +11,4 @@\n"
        " ctx\n"
        "+    # [P341b] theirs\n"
        " ctx\n")

    def test_hunks_are_split_and_selected_by_marker(self):
        header, hunks = split_hunks(self.DIFF)
        assert len(hunks) == 2 and header[0].startswith("diff --git")
        mine, theirs = select_hunks(hunks, "[P350]")
        assert len(mine) == 1 and len(theirs) == 1
        assert "[P350]" in mine[0] and "[P341b]" in theirs[0]

    def test_a_marker_only_in_CONTEXT_is_not_yours(self):
        """Selection keys on ADDED lines: someone else's committed marker
        sitting in the context of your hunk must not claim it."""
        diff = self.DIFF.replace("+    # [P341b] theirs", "     # [P341b] theirs")
        _, hunks = split_hunks(diff)
        mine, _ = select_hunks(hunks, "[P341b]")
        assert mine == []

    def test_foreign_markers_are_detected(self):
        staged = "x [P350] y [P341b] z"
        assert foreign_markers(staged, "[P350]", baseline="") == ["[P341b]"]

    def test_markers_already_in_HEAD_are_not_foreign(self):
        staged = "x [P350] y [P226] z"
        assert foreign_markers(staged, "[P350]", baseline="old [P226]") == []

    def test_the_tool_refuses_when_no_hunk_carries_the_marker(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(REPO / "tools" / "isolate_commit.py"),
             "--marker", "[P99999]", "--dry-run", "CLAUDE.md"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO),
            timeout=600)
        # rc 1 = nothing changed vs HEAD, rc 2 = changed but none is mine.
        # Either way it must never silently stage.
        assert r.returncode in (1, 2), r.stdout
        assert "staged" not in r.stdout

    # ---- [P352b] the blind spot that cost a red CI --------------------
    def test_an_unmarked_dropped_hunk_is_shown_not_counted(self):
        """The incident: two hunks of mine carried no marker (a function
        SIGNATURE line and a config PARSE line, whose explaining comments sat
        in neighbouring hunks), so they were dropped and the tool printed
        success — both of its checks pass on a PARTIAL commit. `3 hunks do not
        carry your marker` was a COUNT; a count is not a location (P293b/P349).
        """
        _, hunks = split_hunks(self.DIFF)
        _, theirs = select_hunks(hunks, "[P350]")
        # theirs carries [P341b] -> attributable, nothing to show
        assert describe_dropped(theirs, "[P350]") == []

        unmarked = self.DIFF.replace("+    # [P341b] theirs",
                                     "+    def f(self, x, evidence_ok=True):")
        _, hunks2 = split_hunks(unmarked)
        _, theirs2 = select_hunks(hunks2, "[P350]")
        shown = describe_dropped(theirs2, "[P350]")
        assert len(shown) == 1
        assert "evidence_ok=True" in shown[0], (
            "the dropped hunk must be rendered with its added lines — the "
            "author cannot recognise their own code from a count"
        )
        assert shown[0].lstrip().startswith("@@"), "no location given"

    def test_a_dropped_hunk_with_someone_elses_marker_stays_quiet(self):
        """Refusing on every dropped hunk would fire on every shared-tree run
        and become wallpaper (P202). Only the AMBIGUOUS ones stop you."""
        _, hunks = split_hunks(self.DIFF)
        _, theirs = select_hunks(hunks, "[P350]")
        assert describe_dropped(theirs, "[P350]") == []

    def test_the_tool_refuses_on_an_unmarked_dropped_hunk(self):
        import subprocess, tempfile, os, shutil
        NL = chr(10)
        LINE = "a = 1" + NL
        repo = tempfile.mkdtemp()
        try:
            def git(*a):
                return subprocess.run(["git", *a], cwd=repo, capture_output=True,
                                      text=True, encoding="utf-8", timeout=600)
            git("init", "-q")
            git("config", "user.email", "t@t")
            git("config", "user.name", "t")
            f = os.path.join(repo, "f.py")
            io.open(f, "w", encoding="utf-8").write(LINE * 20)
            git("add", "f.py")
            git("commit", "-qm", "base")
            lines = [LINE] * 20
            lines[2] = "b = 2  # [P350] mine" + NL
            lines[15] = "c = 3" + NL          # unmarked -> ambiguous
            io.open(f, "w", encoding="utf-8").write("".join(lines))
            r = subprocess.run(
                [sys.executable, "-X", "utf8",
                 str(REPO / "tools" / "isolate_commit.py"),
                 "--marker", "[P350]", "f.py"],
                cwd=repo, capture_output=True, text=True, encoding="utf-8",
                timeout=600)
            assert r.returncode == 2, r.stdout
            assert "c = 3" in r.stdout, "the ambiguous hunk was not shown"
            assert "staged" not in r.stdout

            ok = subprocess.run(
                [sys.executable, "-X", "utf8",
                 str(REPO / "tools" / "isolate_commit.py"),
                 "--marker", "[P350]", "--accept-unmarked", "f.py"],
                cwd=repo, capture_output=True, text=True, encoding="utf-8",
                timeout=600)
            assert ok.returncode == 0, ok.stdout
            assert "staged" in ok.stdout, (
                "the escape hatch must still work once the author has read them"
            )
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_it_never_commits_by_itself(self):
        """A tool that commits is one that eventually commits the wrong thing
        unattended; this one stages and verifies only."""
        src = io.open(REPO / "tools" / "isolate_commit.py", encoding="utf-8").read()
        body = src.split('"""', 2)[-1]
        assert '"commit"' not in body and "'commit'" not in body

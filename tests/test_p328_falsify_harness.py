"""[P328] The falsification harness must catch a VACUOUS probe.

The scenario that motivated it, reproduced synthetically: a probe that edits the
file (matching its anchor exactly once) without changing any behaviour. Under
the old ad-hoc scripts that printed "-> 37 passed", which is indistinguishable
from a healthy line and is what I read past in P327.

These tests build a throwaway module + guard under tmp_path and probe THOSE, so
nothing here depends on a production file staying the way it is today.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.falsify import Probe, run_probe, run_probes  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A tiny module + a guard over it, living inside the repo so pytest and
    relative paths behave, but removed afterwards."""
    pkg = REPO / "_falsify_sandbox"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    mod = pkg / "subject.py"
    mod.write_text(textwrap.dedent('''
        def classify(x):
            """Return 'big' above the threshold, else 'small'."""
            if x > 10:
                return "big"
            return "small"
    ''').lstrip(), encoding="utf-8")
    test = pkg / "test_guard.py"
    test.write_text(textwrap.dedent('''
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from _falsify_sandbox.subject import classify

        def test_threshold_holds():
            assert classify(11) == "big"
            assert classify(9) == "small"
    ''').lstrip(), encoding="utf-8")
    yield {"module": "_falsify_sandbox/subject.py",
           "test": "_falsify_sandbox/test_guard.py",
           "dir": pkg}
    shutil.rmtree(pkg, ignore_errors=True)


class TestItCatchesTheProbeThatFooledMe:

    def test_a_comment_only_probe_is_reported_VACUOUS(self, sandbox):
        """THE P327 CASE. The anchor matches exactly once and the file
        genuinely changes — and the guard stays green, because a comment
        changes nothing. The old harness printed that as '-> N passed'."""
        p = Probe(name="comment-only (vacuous)",
                  path=sandbox["module"],
                  old='    if x > 10:',
                  new='    # a comment that changes nothing\n    if x > 10:',
                  expect_red=[sandbox["test"]])
        assert run_probe(p) is False
        assert p.result == "VACUOUS"
        assert "stayed GREEN" in p.detail

    def test_a_real_defect_is_reported_OK(self, sandbox):
        p = Probe(name="threshold inverted (real)",
                  path=sandbox["module"],
                  old='    if x > 10:',
                  new='    if x > 1000:',
                  expect_red=[sandbox["test"]])
        assert run_probe(p) is True
        assert p.result == "OK"


class TestTheOtherWaysAProbeLies:

    def test_a_missing_anchor_is_INVALID_not_a_pass(self, sandbox):
        p = Probe(name="anchor absent", path=sandbox["module"],
                  old='if x > 999999:', new='if False:',
                  expect_red=[sandbox["test"]])
        assert run_probe(p) is False
        assert p.result == "INVALID"
        assert "matched 0 times" in p.detail

    def test_an_ambiguous_anchor_is_INVALID(self, sandbox):
        """[P238] More than one match means the probe edited somewhere the
        author did not read."""
        path = REPO / sandbox["module"]
        src = io.open(path, encoding="utf-8").read()
        io.open(path, "w", encoding="utf-8", newline="").write(
            src + '\n\ndef other(x):\n    return "small"\n')
        p = Probe(name="ambiguous", path=sandbox["module"],
                  old='return "small"', new='return "BIG"',
                  expect_red=[sandbox["test"]])
        assert run_probe(p) is False
        assert p.result == "INVALID"
        assert "matched 2 times" in p.detail

    def test_a_no_op_replacement_is_INVALID(self, sandbox):
        p = Probe(name="no-op", path=sandbox["module"],
                  old='    if x > 10:', new='    if x > 10:',
                  expect_red=[sandbox["test"]])
        assert run_probe(p) is False
        assert p.result == "INVALID"
        assert "no-op" in p.detail

    def test_an_already_red_suite_is_INVALID(self, sandbox):
        """Probing a suite that was already failing proves nothing about the
        probe — the red is someone else's."""
        path = REPO / sandbox["module"]
        io.open(path, "w", encoding="utf-8", newline="").write(
            "def classify(x):\n    return 'WRONG'\n")
        p = Probe(name="against a red suite", path=sandbox["module"],
                  old="return 'WRONG'", new="return 'ALSO WRONG'",
                  expect_red=[sandbox["test"]])
        assert run_probe(p) is False
        assert p.result == "INVALID"
        assert "ALREADY RED" in p.detail


class TestRestoration:
    """[P265] Probe reversal must be a surgical restore of the PRE-PROBE text,
    never a tree checkout — which reverts to HEAD rather than to the state the
    probe started from, and silently discards uncommitted work."""

    def test_the_file_is_restored_byte_identically_after_a_real_probe(
            self, sandbox):
        path = REPO / sandbox["module"]
        before = io.open(path, encoding="utf-8").read()
        run_probe(Probe(name="real", path=sandbox["module"],
                        old='    if x > 10:', new='    if x > 1000:',
                        expect_red=[sandbox["test"]]))
        assert io.open(path, encoding="utf-8").read() == before

    def test_it_is_restored_even_after_a_vacuous_probe(self, sandbox):
        path = REPO / sandbox["module"]
        before = io.open(path, encoding="utf-8").read()
        run_probe(Probe(name="vacuous", path=sandbox["module"],
                        old='    if x > 10:',
                        new='    # nothing\n    if x > 10:',
                        expect_red=[sandbox["test"]]))
        assert io.open(path, encoding="utf-8").read() == before

    def test_the_harness_never_shells_out_to_git(self):
        src = io.open(REPO / "tools" / "falsify.py", encoding="utf-8").read()
        code = src.split('"""', 2)[-1]
        assert "checkout" not in code
        assert "git" not in code.replace("git grep", "")


class TestTheAggregateVerdict:

    def test_one_vacuous_probe_fails_the_whole_run(self, sandbox, capsys):
        ok = run_probes([
            Probe(name="real", path=sandbox["module"], old='    if x > 10:',
                  new='    if x > 1000:', expect_red=[sandbox["test"]]),
            Probe(name="vacuous", path=sandbox["module"], old='    if x > 10:',
                  new='    # nothing\n    if x > 10:',
                  expect_red=[sandbox["test"]]),
        ])
        assert ok is False
        out = capsys.readouterr().out
        assert "VACUOUS" in out
        assert "DID NOT FALSIFY" in out

    def test_all_real_probes_pass_the_run(self, sandbox, capsys):
        ok = run_probes([
            Probe(name="real", path=sandbox["module"], old='    if x > 10:',
                  new='    if x > 1000:', expect_red=[sandbox["test"]]),
        ])
        assert ok is True
        assert "ALL PROBES FALSIFIED" in capsys.readouterr().out


class TestTheDeployPreflightScansTheDeployedCommit:
    """[P328] The second half of the same mistake: a verification that measured
    the wrong SUBJECT.

    Step 1 of the deploy pulls whatever origin/main holds, so scanning the
    local checkout answers a different question — and in a shared checkout it
    answers it about another session's uncommitted edits. That happened: a
    clean commit was refused because a parallel session had four unrelated
    findings in files it was still editing. The working tree cannot tell you
    what you committed (P311b).
    """

    SCRIPT = REPO / "scripts" / "hetzner_deploy.sh"

    def _src(self):
        return io.open(self.SCRIPT, encoding="utf-8").read()

    def test_the_scan_targets_a_worktree_at_the_deployed_sha(self):
        src = self._src()
        assert 'git worktree add --detach -q "${SCAN_TMP}" "${DEPLOY_SHA}"' in src
        assert 'cd "${SCAN_TREE}" &&' in src

    def test_the_worktree_is_removed_afterwards(self):
        """A leaked worktree accumulates a full checkout per deploy."""
        assert 'git worktree remove --force "${SCAN_TREE}"' in self._src()

    def test_the_fallback_announces_that_it_changed_subject(self):
        """A silently different subject is exactly what this fixes, so the
        degraded path must say which tree it scanned."""
        src = self._src()
        assert "build a worktree at the deployed sha." in src
        assert "Falling back to the WORKING TREE" in src

    def test_the_gate_still_refuses_on_a_nonzero_scan(self):
        """The fix must not become a way to deploy past a real regression.

        The first version of this test asserted only that "SCAN_RC" appeared
        somewhere in the file, and the falsification harness reported it
        VACUOUS: replacing the guard with `if [ 0 -ne 0 ]` left the symbol
        present elsewhere and the test green. Pin the exact conditional, and
        the refusal that must follow it.
        """
        src = self._src()
        guard = "if [ ${SCAN_RC} -ne 0 ]; then"
        assert guard in src
        after = src[src.index(guard):src.index(guard) + 500]
        assert "Refusing to deploy." in after
        assert "exit 1" in after

    def test_a_worktree_gives_the_scanners_a_real_git(self):
        """The load-bearing claim behind using a worktree rather than a file
        copy: the authority audit REFUSES to run without a git-grep engine
        rather than emit false findings (P158), so a bare copy of the tree
        cannot be scanned at all."""
        import subprocess as sp
        import tempfile
        wt = tempfile.mkdtemp()
        made = sp.run(["git", "worktree", "add", "--detach", "-q", wt, "HEAD"],
                      cwd=str(REPO), capture_output=True, text=True,
                      encoding="utf-8")
        if made.returncode != 0:
            pytest.skip(f"git worktree unavailable: {made.stderr[:120]}")
        try:
            r = sp.run(["git", "grep", "-P", "-l", r"\bdef\s+main\b"],
                       cwd=wt, capture_output=True, text=True,
                       encoding="utf-8", timeout=180)
            assert r.returncode in (0, 1), r.stderr[:200]
            assert r.stdout.strip(), "git grep found nothing in the worktree"
        finally:
            sp.run(["git", "worktree", "remove", "--force", wt],
                   cwd=str(REPO), capture_output=True)
            sp.run(["git", "worktree", "prune"], cwd=str(REPO),
                   capture_output=True)

    def test_an_interrupted_deploy_still_removes_the_worktree(self):
        """Observed while validating this very change: a deploy whose stdout
        was piped through `head` took SIGPIPE partway through and left a full
        checkout behind, which then appears in `git worktree list` forever.
        The explicit remove is the happy path; the trap is the backstop."""
        # [P329] the hand-rolled version of this check became
        # tests/_guard_pins.assert_live_line, so the next author who pins a
        # line in a shell script gets the non-commented requirement for free.
        sys.path.insert(0, str(REPO / "tests"))
        from _guard_pins import assert_live_line
        src = self._src()
        assert_live_line(src, "trap 'git worktree remove --force",
                         why="the interrupt backstop must be live")
        assert_live_line(src, "EXIT INT TERM HUP PIPE",
                         why="the trap must cover the signals a pipe sends")


class TestAssertLiveLine:
    """[P329] The generic half of the guard-pin trap: in a shell script, YAML
    or Dockerfile the cheapest defeat is a `#`, and `assert "<line>" in src`
    stays green over dead code. `assert_guard_live` cannot help — it needs a
    Python condition."""

    def _f(self):
        sys.path.insert(0, str(REPO / "tests"))
        from _guard_pins import assert_live_line
        return assert_live_line

    def test_a_live_line_passes(self):
        self._f()("    trap 'cleanup' EXIT\n", "trap 'cleanup'")

    def test_a_line_commented_at_the_start_fails(self):
        with pytest.raises(AssertionError, match="COMMENTED"):
            self._f()("    # trap 'cleanup' EXIT\n", "trap 'cleanup'")

    def test_a_line_commented_INLINE_fails(self):
        """The case the falsification harness caught in the first version:
        `true # trap ...` does not START with a comment, yet the text is dead."""
        with pytest.raises(AssertionError, match="COMMENTED"):
            self._f()("    true # trap 'cleanup' EXIT\n", "trap 'cleanup'")

    def test_an_absent_line_fails_with_a_different_message(self):
        """Absent and commented are different faults; collapsing them would
        send the author looking for the wrong thing."""
        with pytest.raises(AssertionError, match="does not appear at all"):
            self._f()("echo hi\n", "trap 'cleanup'")

    def test_a_long_flag_is_not_treated_as_a_comment(self):
        """`--` was a default marker in the first version and cut the very line
        it protected: shell long flags look exactly like a SQL comment, so the
        pinned text vanished and every guard read as dead."""
        self._f()("    git worktree remove --force /tmp/x\n",
                  "git worktree remove --force")

    def test_markers_stay_overridable_for_languages_that_need_them(self):
        with pytest.raises(AssertionError, match="COMMENTED"):
            self._f()("    -- DROP TABLE t\n", "DROP TABLE t",
                      markers=("--",))

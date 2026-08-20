"""[P344] Two process failures of mine, converted into mechanisms.

  1. I polled a repo slug that does not exist and my loop printed "none-yet"
     twenty times at it -- "I could not ask" rendered identically to "no
     answer yet", inside a retry loop, which spent the API budget.
  2. I created two worktrees for a mypy attribution and leaked both.

The load-bearing tests here are NOT "the slug parses". They are:
  * UNREADABLE is not retryable, and main() proves it by call count.
  * the newest run per workflow wins (a green-then-red sha must read RED).
  * the worktree context manager cleans up on an EXCEPTION path.
  * line numbers are stripped before diffing mypy findings.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import ci_status as ci  # noqa: E402
from tools.mypy_attribution import normalize  # noqa: E402
from tools.scratch_worktree import (  # noqa: E402
    WorktreeError, list_worktrees, remove, scratch_worktree)

DEPLOY = REPO / "scripts" / "hetzner_deploy.sh"


def _run(name, status, conclusion):
    return {"name": name, "status": status, "conclusion": conclusion}


class TestTheSlugIsDerivedNeverTyped:

    @pytest.mark.parametrize("url", [
        "git@github.com:melodygaoyifan/crypto_trading.git",
        "https://github.com/melodygaoyifan/crypto_trading.git",
        "https://github.com/melodygaoyifan/crypto_trading",
        "ssh://git@github.com/melodygaoyifan/crypto_trading.git",
        "  https://github.com/melodygaoyifan/crypto_trading/  \n",
    ])
    def test_every_remote_form_yields_the_same_slug(self, url):
        assert ci.slug_from_remote(url) == "melodygaoyifan/crypto_trading"

    @pytest.mark.parametrize("bad", ["", "not-a-url", "https://github.com/",
                                     "https://github.com/owner", "a/b/c"])
    def test_an_underivable_remote_REFUSES_rather_than_guessing(self, bad):
        """A guessed slug reports 'no runs yet' forever, which is how a typo
        reads as a healthy pending build. That is the incident."""
        with pytest.raises(ci.Unreadable):
            ci.slug_from_remote(bad)

    def test_the_real_remote_resolves(self):
        assert "/" in ci.current_slug(cwd=str(REPO))


class TestClassify:

    def test_green(self):
        code, _ = ci.classify([_run("codebase-invariants", "completed", "success"),
                               _run("test-suite", "completed", "success")])
        assert code == ci.GREEN

    def test_red(self):
        code, detail = ci.classify([
            _run("codebase-invariants", "completed", "failure"),
            _run("test-suite", "completed", "success")])
        assert code == ci.RED and "failure" in detail

    def test_pending(self):
        code, _ = ci.classify([_run("codebase-invariants", "in_progress", None),
                               _run("test-suite", "completed", "success")])
        assert code == ci.PENDING

    def test_missing_is_not_the_same_verdict_as_pending(self):
        """A workflow with no run at all is a different state from one that is
        running, and a poller may want to treat them differently."""
        code, detail = ci.classify([_run("test-suite", "completed", "success")])
        assert code == ci.MISSING and "codebase-invariants" in detail

    def test_the_NEWEST_run_per_workflow_wins(self):
        """[P287] The API returns newest-first. The old bug was an
        unconditional overwrite, so the OLDEST won: a sha whose first run was
        green and whose re-run went red read as GREEN and deployed."""
        runs = [  # newest first: the re-run failed
            _run("codebase-invariants", "completed", "failure"),
            _run("test-suite", "completed", "success"),
            _run("codebase-invariants", "completed", "success"),
        ]
        assert ci.classify(runs)[0] == ci.RED

    def test_unrelated_workflows_are_ignored(self):
        runs = [_run("auto-deploy", "completed", "failure"),
                _run("codebase-invariants", "completed", "success"),
                _run("test-suite", "completed", "success")]
        assert ci.classify(runs)[0] == ci.GREEN


class TestUnreadableIsNeverRetried:
    """THE incident, pinned. Retrying a question the API refused is what turns
    a typo into twenty minutes and a spent budget."""

    def test_unreadable_is_not_in_the_retryable_set(self):
        assert ci.UNREADABLE not in ci.RETRYABLE
        assert ci.PENDING in ci.RETRYABLE and ci.MISSING in ci.RETRYABLE

    def test_every_verdict_code_is_distinct(self):
        codes = [ci.GREEN, ci.RED, ci.UNREADABLE, ci.PENDING, ci.MISSING]
        assert len(set(codes)) == len(codes)

    def test_main_calls_the_api_exactly_ONCE_when_it_cannot_ask(self, monkeypatch, capsys):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise ci.Unreadable("GitHub has no repository 'x/y' (404)")

        monkeypatch.setattr(ci, "status", boom)
        rc = ci.main(["--sha", "deadbeef", "--slug", "x/y",
                      "--wait-seconds", "600", "--interval", "0"])
        assert rc == ci.UNREADABLE
        assert len(calls) == 1, (
            "an unreadable answer was retried -- that is the defect this "
            "module exists to prevent")
        assert "UNREADABLE" in capsys.readouterr().out

    def test_main_DOES_retry_a_pending_build_up_to_the_cap(self, monkeypatch):
        calls = []

        def pending(*a, **k):
            calls.append(1)
            return ci.PENDING, "test-suite=in_progress"

        monkeypatch.setattr(ci, "status", pending)
        rc = ci.main(["--sha", "d", "--slug", "x/y", "--wait-seconds", "600",
                      "--interval", "0", "--max-requests", "3"])
        assert rc == ci.PENDING
        assert len(calls) == 3, "the hard request cap did not bind"

    def test_main_returns_immediately_on_a_terminal_verdict(self, monkeypatch):
        calls = []

        def green(*a, **k):
            calls.append(1)
            return ci.GREEN, "ok"

        monkeypatch.setattr(ci, "status", green)
        assert ci.main(["--sha", "d", "--slug", "x/y", "--wait-seconds", "600",
                        "--interval", "0"]) == ci.GREEN
        assert len(calls) == 1


class TestTheDeployUsesTheOneImplementation:
    """[P172] The correct check existed here and nowhere else, which is why
    every ad-hoc copy was worse."""

    def _src(self):
        return io.open(DEPLOY, encoding="utf-8").read()

    def test_the_deploy_calls_the_tool(self):
        assert "tools/ci_status.py" in self._src()

    def test_the_deploy_no_longer_carries_its_own_parser(self):
        """A second implementation is how the two drift."""
        src = self._src()
        assert "workflow_runs" not in src, (
            "the deploy is parsing the API again; there must be exactly one "
            "implementation of this check")

    def test_a_missing_tool_REFUSES_rather_than_skipping(self):
        """P159/P187: a gate that cannot run must never read as one that
        passed. This is the whole reason the tool call is safe."""
        src = self._src()
        i = src.index("CI_TOOL=")
        block = src[i:i + 700]
        assert "if [ ! -f" in block and "exit 1" in block

    def test_a_nonzero_verdict_blocks_the_deploy(self):
        src = self._src()
        i = src.index("CI_RC=$?")
        block = src[i:i + 900]
        assert 'if [ "${CI_RC}" -ne 0 ]; then' in block
        assert "exit 1" in block


class TestScratchWorktreeCannotLeak:

    def test_it_is_removed_on_the_happy_path(self):
        before = len(list_worktrees())
        with scratch_worktree("HEAD") as wt:
            assert wt.exists()
            assert len(list_worktrees()) == before + 1
        assert len(list_worktrees()) == before

    def test_it_is_removed_when_the_body_RAISES(self):
        """The path that leaked both of mine: an exception (or a Ctrl-C) in
        the middle of an attribution run."""
        before = len(list_worktrees())
        with pytest.raises(RuntimeError):
            with scratch_worktree("HEAD"):
                raise RuntimeError("boom")
        assert len(list_worktrees()) == before

    def test_remove_refuses_the_repository_itself(self):
        """A cleanup helper that can be pointed anywhere is a worse problem
        than the leak it fixes."""
        with pytest.raises(WorktreeError):
            remove(REPO)
        assert (REPO / ".git").exists()

    def test_a_bad_ref_raises_rather_than_yielding_an_empty_tree(self):
        with pytest.raises(WorktreeError):
            with scratch_worktree("no-such-ref-p344"):
                pass


class TestAttributionComparesFindingsNotPositions:

    def test_line_numbers_are_stripped(self):
        """A file that gained insertions reports every downstream finding as
        'new' otherwise -- which is how a one-error delta reads as thirty and
        sends you re-baselining (P171/P175)."""
        a = normalize('core/x.py:10: error: Incompatible types  [assignment]')
        b = normalize('core/x.py:99: error: Incompatible types  [assignment]')
        assert a == b is not None

    def test_a_different_message_is_a_different_finding(self):
        a = normalize('core/x.py:10: error: Incompatible types  [assignment]')
        b = normalize('core/x.py:10: error: Name "q" is not defined  [name-defined]')
        assert a != b

    def test_a_different_file_is_a_different_finding(self):
        a = normalize('core/x.py:10: error: boom  [misc]')
        b = normalize('core/y.py:10: error: boom  [misc]')
        assert a != b

    def test_non_error_lines_are_dropped(self):
        assert normalize("Found 3 errors in 2 files") is None
        assert normalize('core/x.py:1: note: see here') is None

    def test_it_imports_CRITICAL_DIRS_rather_than_restating_them(self):
        """P172: the diagnostic must not drift from the gate it diagnoses."""
        src = io.open(REPO / "tools" / "mypy_attribution.py",
                      encoding="utf-8").read()
        assert "from tools.lint_mypy_baseline import CRITICAL_DIRS" in src
        for d in ("risk", "defense", "signals"):
            assert '"' + d + '"' not in src.split('"""', 2)[-1]

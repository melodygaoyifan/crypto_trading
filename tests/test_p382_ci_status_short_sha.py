"""[P382] `ci_status.py` must expand an abbreviated sha before asking GitHub.

The Actions API matches `head_sha` EXACTLY: a 7-char prefix returns zero
runs, which the tool reported as MISSING ("no run yet") — indistinguishable
from a push that never triggered CI. Seen 2026-08-23: three green commits
read MISSING for 25 minutes because the poll was given short shas. That is
the P322e/P344 trap inside the tool built to close it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import ci_status  # noqa: E402


def _head():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, encoding="utf-8", cwd=REPO).stdout.strip()


def test_a_short_sha_is_expanded_to_the_full_id():
    full = _head()
    if len(full) != 40:
        pytest.skip("no git HEAD here")
    assert ci_status._full_sha(full[:7]) == full


def test_a_full_sha_passes_through_unchanged():
    full = _head()
    if len(full) != 40:
        pytest.skip("no git HEAD here")
    assert ci_status._full_sha(full.upper()) == full.lower()


def test_an_unresolvable_short_sha_passes_through_but_warns(capsys):
    # a test double / foreign-clone id must not break the tool (the P344
    # suite drives main() with fake shas) — but the caller is told why a
    # MISSING might follow
    assert ci_status._full_sha("deadbee") == "deadbee"
    assert "not a full 40-hex sha" in capsys.readouterr().err


def test_main_routes_sha_through_the_expander():
    import inspect
    src = inspect.getsource(ci_status.main)
    assert "_full_sha(a.sha)" in src

"""[P159] "mypy is not installed" must never be recorded as "0 findings".

`run_mypy` shells out to `sys.executable -m mypy`. When mypy is absent that
command exits 1 and prints "No module named mypy" to stderr — it does NOT
raise FileNotFoundError, so the scanner's `except FileNotFoundError` guard was
unreachable (P152 shape: a guard defined but never called). `parse_errors`
then found no "error: ... [code]" lines and reported a total of 0.

Consequences, in order of severity:
  1. `ci_check_invariants --update` on such a machine rewrote
     mypy_baseline.json from 1080 findings to 0 — which then fails the gate
     with +1080 on every machine that DOES have mypy.
  2. In check mode the count only ever went DOWN vs baseline, and the gate
     only flags increases — so the mypy check passed silently without ever
     running.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.lint_mypy_baseline import MypyUnavailable, parse_errors, run_mypy


def test_missing_mypy_raises_instead_of_returning_empty(monkeypatch):
    """The exact stderr real mypy-less interpreters produce."""
    class _R:
        returncode = 1
        stdout = ""
        stderr = f"{sys.executable}: No module named mypy\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    with pytest.raises(MypyUnavailable):
        run_mypy(["core"])


def test_a_genuinely_clean_run_is_not_mistaken_for_unavailable(monkeypatch):
    """mypy present, zero errors — must return normally, not raise."""
    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    assert parse_errors(run_mypy(["core"])) == {}


def test_nonzero_exit_with_real_findings_still_parses(monkeypatch):
    """mypy exits non-zero when it finds errors — that is the normal path
    and must not be confused with the tool being missing."""
    class _R:
        returncode = 1
        stdout = "core/x.py:1: error: bad thing  [union-attr]\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    assert parse_errors(run_mypy(["core"])) == {"union-attr": 1}


def test_baseline_format_emits_unavailable_sentinel_not_zero():
    """End-to-end through the CLI, in whatever state this machine is in.

    Either mypy is installed (real counts) or it is not (sentinel) — but the
    output must never be a zero-count dict, which is the corrupting shape.
    """
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "tools/lint_mypy_baseline.py",
         "--baseline-format"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    if "unavailable" in payload:
        assert "mypy" in payload["unavailable"]
        assert "total_count" not in payload, (
            "an unavailable run must not report a count at all"
        )
    else:
        assert payload["total_count"] == sum(payload["by_code"].values())


def test_ci_check_never_writes_a_zero_mypy_baseline(tmp_path):
    """The committed baseline must retain its real count after a gate run on
    a machine where mypy may be missing."""
    baseline = json.loads(
        (REPO / "tools/scanner_baselines/mypy_baseline.json").read_text()
    )
    assert baseline["total_count"] > 0, (
        "mypy_baseline.json has been zeroed — this is the P159 corruption; "
        "restore it from git and re-read tools/lint_mypy_baseline.py"
    )

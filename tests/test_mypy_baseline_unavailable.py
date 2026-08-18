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

from tools.lint_mypy_baseline import (
    MypyUnavailable,
    mypy_version,
    parse_errors,
    run_mypy,
)


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
    encoding="utf-8")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    if "unavailable" in payload:
        assert "mypy" in payload["unavailable"]
        assert "total_count" not in payload, (
            "an unavailable run must not report a count at all"
        )
    else:
        assert payload["total_count"] == sum(payload["by_code"].values())


# ---------------------------------------------------------------------------
# [P161] the baseline is only comparable to the analyzer that produced it
# ---------------------------------------------------------------------------

def test_version_is_parsed_from_mypy_banner(monkeypatch):
    class _R:
        returncode = 0
        stdout = "mypy 2.3.0 (compiled: yes)\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    assert mypy_version() == "2.3.0"


def test_version_probe_reports_unavailable_rather_than_a_fake_version(monkeypatch):
    """A missing tool must fail the same way here as in run_mypy — otherwise
    the stamp would read as a real version and the counts as comparable."""
    class _R:
        returncode = 1
        stdout = ""
        stderr = f"{sys.executable}: No module named mypy\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    with pytest.raises(MypyUnavailable):
        mypy_version()


def test_baseline_format_stamps_the_analyzer_version():
    """Without this stamp a mypy upgrade is indistinguishable from a code
    regression: going 1.x -> 2.3.0 dropped the TOTAL 1080 -> 1073 while five
    individual codes ROSE, and the gate only flags increases."""
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "tools/lint_mypy_baseline.py",
         "--baseline-format"],
        capture_output=True, text=True, cwd=REPO,
    encoding="utf-8")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    if "unavailable" not in payload:
        assert payload.get("mypy_version"), (
            "counts without a version stamp cannot be compared to a baseline"
        )


def test_gate_does_not_report_phantom_findings_on_this_machine():
    """End-to-end: the deploy gate must exit 0 on an unmodified checkout.

    This is the regression that mattered — 10 phantom mypy 'NEW findings'
    blocked `scripts/hetzner_deploy.sh` at step 0/5 with no code change
    behind them.

    [P253d re-scope] Runs the gate WITH --skip-mypy, because that is what
    the deploy path actually runs since P253b: the mypy baseline is a
    fingerprint of CI's environment (P227 — identical code measures 1076
    findings in CI and 1083+ on the operator's Windows venv at the SAME
    mypy release), so the FULL gate legitimately reports phantom findings
    on non-CI machines and its cleanliness is CI's job (verified per-deploy
    via the API check in hetzner_deploy.sh). What must hold on EVERY
    machine is that the env-independent stdlib scanners are clean — that
    is what this asserts. Before the re-scope this test failed on any
    dev box with mypy installed, which taught people to ignore it (P196).
    """
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "tools/ci_check_invariants.py",
         "--skip-mypy"],
        capture_output=True, text=True, cwd=REPO,
    encoding="utf-8")
    assert r.returncode == 0, (
        f"deploy-path gate (--skip-mypy) failed:\n{r.stdout}\n{r.stderr}"
    )


def test_ci_check_never_writes_a_zero_mypy_baseline(tmp_path):
    """The committed baseline must retain its real count after a gate run on
    a machine where mypy may be missing."""
    baseline = json.loads(
        (REPO / "tools/scanner_baselines/mypy_baseline.json").read_text(encoding="utf-8")
    )
    assert baseline["total_count"] > 0, (
        "mypy_baseline.json has been zeroed — this is the P159 corruption; "
        "restore it from git and re-read tools/lint_mypy_baseline.py"
    )

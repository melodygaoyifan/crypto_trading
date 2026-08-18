"""[P175] The mypy gate was skipping every run. Keep it from going dark again.

P161 (2026-08-04) made the mypy baseline analyzer-version-aware: per-code counts
are a fingerprint of the mypy release, so diffing across versions reports
phantom regressions. The guard carries the old baseline forward and prints a
SKIPPED warning instead. That is the right behaviour — but the committed
baseline was written 2026-06-13 and carried no version stamp, so from the moment
P161 landed, `ci_check_invariants.py` verified **no type errors at all** while
still printing `OK — no new findings vs baseline`.

The warning was there. It scrolled past on every run, above a green result line.
That is the same family as P171/P174: the check did not run, and nothing in the
exit code distinguished that from the check passing.

Before re-baselining, the delta had to be attributed. Baseline said 1080 errors;
mypy 2.3.0 said 1076, with five per-code counts RISING (arg-type +2, float +2,
index +2, operator +1, var-annotated +3) even as the total fell. Re-baselining
blind would have accepted those ten as the new floor. So HEAD was exported to a
scratch tree with `git archive`, mypy run against both trees, and the two error
sets compared with line numbers stripped:

    HEAD errors: 1139   CURRENT errors: 1139
    errors present NOW but not at HEAD:  (none)
    errors present at HEAD but fixed NOW: (none)

The working tree introduces zero new type errors, so the entire delta is the
analyzer version. Only then was the baseline re-stamped at mypy 2.3.0 / 1076.

Then — per P174, do not assume a restored check works, construct the failure —
a deliberate `x: int = "not an int"` was dropped into core/ and the gate was run:

    + mypy.by_code.assignment: count INCREASED 453 -> 454 (+1)
    + mypy.total_count: count INCREASED 1076 -> 1077 (+1)

The probe was removed and the gate returned to 0. These tests keep both halves
true: the stamp must exist and match, and the scanner must still count errors.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "tools" / "scanner_baselines" / "mypy_baseline.json"
SCANNER = REPO_ROOT / "tools" / "lint_mypy_baseline.py"


def _installed_mypy_version():
    try:
        import mypy.version  # type: ignore
        return mypy.version.__version__
    except Exception:
        return None


class TestTheBaselineCarriesItsAnalyzerVersion:
    def test_baseline_exists_and_is_stamped(self):
        b = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert b.get("mypy_version"), (
            "the mypy baseline has no `mypy_version` key. P161's guard treats "
            "an unstamped baseline as a version mismatch and SKIPS the mypy "
            "check on every run, while ci_check still prints OK. That is how "
            "this gate spent its life green and inert. Re-baseline with "
            "`python -X utf8 tools/lint_mypy_baseline.py --baseline-format`."
        )

    def test_baseline_has_the_shape_the_gate_diffs(self):
        b = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert isinstance(b.get("by_code"), dict) and b["by_code"]
        assert isinstance(b.get("total_count"), int)
        assert b["total_count"] == sum(b["by_code"].values()), (
            "total_count must equal the sum of by_code, or one of the two is "
            "measuring something the other is not"
        )

    def test_the_stamp_matches_the_installed_analyzer(self):
        installed = _installed_mypy_version()
        if installed is None:
            pytest.skip("mypy not installed in this interpreter")
        # .get, not [] — an unstamped baseline is the exact state this file
        # exists to catch, and it should report that, not raise KeyError.
        stamped = json.loads(BASELINE.read_text(encoding="utf-8")).get(
            "mypy_version", "<unstamped>")
        assert stamped == installed, (
            f"baseline was produced by mypy {stamped}, this environment has "
            f"{installed}. RIGHT NOW ci_check is skipping the mypy gate and "
            f"still exiting 0. Re-baseline deliberately: confirm the working "
            f"tree adds no new errors vs HEAD first (git archive HEAD to a "
            f"scratch tree, run mypy on both, diff with line numbers stripped), "
            f"then regenerate. Do not re-baseline to make this test pass."
        )


class TestTheScannerStillCountsErrors:
    """Falsifiability. A restored check that cannot fail is P174 again."""

    def _scan(self, path):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(SCANNER),
             "--baseline-format", "--paths", str(path)],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        encoding="utf-8")
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    def test_a_known_type_error_is_counted(self, tmp_path):
        if _installed_mypy_version() is None:
            pytest.skip("mypy not installed in this interpreter")
        (tmp_path / "bad.py").write_text(textwrap.dedent("""
            def probe() -> int:
                x: int = "definitely not an int"
                return x
        """), encoding="utf-8")
        res = self._scan(tmp_path)
        assert res["total_count"] >= 1, (
            "the scanner reported no errors on a file that plainly has one — "
            "the gate is counting nothing and will pass forever"
        )
        assert "assignment" in res["by_code"]

    def test_clean_code_scores_zero(self, tmp_path):
        if _installed_mypy_version() is None:
            pytest.skip("mypy not installed in this interpreter")
        (tmp_path / "good.py").write_text(textwrap.dedent("""
            def probe() -> int:
                x: int = 1
                return x
        """), encoding="utf-8")
        assert self._scan(tmp_path)["total_count"] == 0, (
            "a clean file scores non-zero, so the count is not measuring what "
            "the gate thinks it measures"
        )

    def test_unavailable_mypy_is_not_reported_as_clean(self):
        # [P159] The distinction this whole file rests on. Reading the source
        # rather than uninstalling mypy to check.
        src = SCANNER.read_text(encoding="utf-8")
        assert "class MypyUnavailable" in src
        assert "unavailable" in src, (
            "the scanner must signal 'could not run' distinctly from 'found "
            "nothing'; conflating them once rewrote the baseline to 0"
        )


class TestCiCheckDoesNotPrintOkWhileSkipping:
    def test_the_skip_warning_and_the_ok_line_are_both_reachable(self):
        # The failure mode was cosmetic-looking and load-bearing: a warning
        # block above a green summary. Keep both strings present so the
        # warning cannot be deleted while the OK line stays.
        src = (REPO_ROOT / "tools" / "ci_check_invariants.py").read_text(
            encoding="utf-8")
        assert "mypy check SKIPPED" in src
        assert "did NOT verify type errors" in src, (
            "the skip path must say in words that nothing was verified"
        )

    def test_update_still_bypasses_the_version_carry_forward(self):
        # Sharp edge worth keeping visible: `--update` deliberately disables
        # the mismatch guard so an operator CAN re-stamp. That also means
        # `--update` silently re-baselines mypy along with everything else —
        # which is why this session wrote the baseline file directly instead.
        src = (REPO_ROOT / "tools" / "ci_check_invariants.py").read_text(
            encoding="utf-8")
        assert "and not args.update" in src

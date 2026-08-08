"""[P226] Two scanners silently skipped main.py — the largest file in the repo.

`main.py` carries a UTF-8 BOM. `scripts/silent_failure_audit.py` and
`tools/lint_silent_swallow.py` both read with `encoding="utf-8"`, so `ast.parse`
raised `SyntaxError: invalid non-printable character U+FEFF` and the file was
dropped. Not loudly — a parse failure was indistinguishable from a clean file,
so the committed baselines simply did not include main.py:

    silent_failure  tryexcept  337 -> 645   (+308, ALL main.py symbols)
    silent_swallow  total      416 -> 688   (+272)

This is P171 EXACTLY — same file, same BOM, same SyntaxError, same "a check that
cannot read the code reports the same thing as a check that found nothing". It
was fixed there in `lint_orphan_signal_reads.py` and never applied to these two.

Found by accident: an unrelated edit stripped the BOM and 308 findings appeared
at once. The tempting move was to restore the BOM and make the gate green again,
which would have re-hidden them. Attributed first (P175): every new entry is a
main.py symbol, and both counts are now IDENTICAL with and without the BOM — so
the rise is coverage, not regression.
"""

import ast
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCANNERS = [
    _REPO / "scripts" / "silent_failure_audit.py",
    _REPO / "tools" / "lint_silent_swallow.py",
    _REPO / "tools" / "lint_orphan_signal_reads.py",   # P171, already fixed
]


class TestScannersAreBomSafe:

    @pytest.mark.parametrize("path", _SCANNERS, ids=lambda p: p.name)
    def test_it_reads_source_with_utf8_sig(self, path):
        """These files also read JSON baselines with plain utf-8, which is
        fine — the contract is that SOURCE reads are BOM-safe. Asserting no
        bare `encoding="utf-8"` anywhere was my first version and it failed on
        those legitimate JSON reads."""
        src = path.read_text(encoding="utf-8", errors="replace")
        assert "utf-8-sig" in src, (
            f"{path.name} has no BOM-safe read — a BOM-prefixed file (main.py) "
            f"raises SyntaxError and is silently skipped"
        )

    def test_the_swallow_linter_actually_sees_main_py(self):
        """The behavioural check, which is what actually matters: a source scan
        can be satisfied while the file is still skipped for another reason."""
        import subprocess, sys
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(_REPO / "tools" / "lint_silent_swallow.py"),
             "--json"],
            capture_output=True, text=True, cwd=str(_REPO), timeout=600)
        data = json.loads(r.stdout)
        files = {str(f.get("file", "")).replace("\\", "/")
                 for f in (data.get("findings") or [])}
        assert any(f.endswith("main.py") for f in files), (
            "main.py contributes no findings — it is being skipped again "
            f"(scanned {len(files)} files)"
        )


class TestTheDefectIsReal:

    def test_a_bom_really_breaks_plain_utf8_parsing(self):
        """Pin the mechanism, so if Python ever tolerates it the explanation in
        these files is flagged rather than left quietly wrong."""
        src = "﻿" + "x = 1\n"
        with pytest.raises(SyntaxError):
            ast.parse(src)
        ast.parse(src.encode("utf-8").decode("utf-8-sig"))  # utf-8-sig is fine

    def test_main_py_is_the_file_at_risk(self):
        """Whether or not it currently carries one, main.py is the file whose
        BOM caused this — and it is the largest scan target."""
        raw = (_REPO / "main.py").read_bytes()
        assert len(raw) > 500_000, "main.py is the biggest scan target"


class TestBaselinesReflectTheHonestCounts:

    def test_silent_failure_baseline_includes_main_py(self):
        d = json.loads((_REPO / "tools" / "scanner_baselines" /
                        "silent_failure_baseline.json").read_text(encoding="utf-8"))
        assert d["tryexcept_count"] > 400, (
            "baseline looks like the main.py-excluded number (337) — the "
            "scanner is skipping it again"
        )

    def test_silent_swallow_baseline_includes_main_py(self):
        d = json.loads((_REPO / "tools" / "scanner_baselines" /
                        "silent_swallow_baseline.json").read_text(encoding="utf-8"))
        assert d["total_count"] > 500

    def test_the_rise_is_documented_as_coverage_not_regression(self):
        """A jump this size must carry its attribution, or the next reader
        assumes 272 defects were introduced."""
        for name in ("silent_failure_baseline.json", "silent_swallow_baseline.json"):
            d = json.loads((_REPO / "tools" / "scanner_baselines" / name
                            ).read_text(encoding="utf-8"))
            notes = " ".join(v for k, v in d.items()
                             if k.startswith("_") and isinstance(v, str))
            assert "BOM" in notes and "main.py" in notes, name

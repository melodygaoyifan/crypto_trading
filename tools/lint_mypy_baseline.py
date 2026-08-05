#!/usr/bin/env python3
"""
lint_mypy_baseline.py — count-only mypy baseline scanner
==========================================================

[P113 (4/6) 2026-04-27] Wraps mypy with the same baseline-diff
semantics as the existing 5 CI scanners. Future PRs that ADD new mypy
errors fail; PRs that fix existing errors lower the count and become
the new baseline (after operator runs --update).

Doesn't try to fix the 982 existing errors at once — that's a multi-
day refactor. Instead, freezes the count so it can only DECREASE.

Strategy: count by error category (mypy [...code]), so adding a
single None-deref bug fails CI even if you simultaneously fix 2
annotation-gap errors. Each category is independently floor-locked.

Usage:
    python tools/lint_mypy_baseline.py                    # human output
    python tools/lint_mypy_baseline.py --baseline-format  # CI consumption
    python tools/lint_mypy_baseline.py --paths risk/ core/  # subset
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]

# Critical directories — full mypy scan covers these.
# Smaller scope = faster CI run + sharper signal.
CRITICAL_DIRS = ["risk", "core", "defense", "analytics", "signals", "execution"]


class MypyUnavailable(RuntimeError):
    """[P159] mypy is not importable by this interpreter.

    Distinct from "mypy ran and found nothing". Conflating the two is what
    let `ci_check_invariants --update` rewrite the baseline from 1080 findings
    to 0 on a machine without mypy — which would then fail the gate with
    +1080 on every machine that has it.
    """

    def __init__(self, interpreter: str):
        super().__init__(
            f"mypy is not installed in {interpreter}. "
            f"Run: pip install mypy  (declared in requirements-train.txt)"
        )


def run_mypy(paths: List[str]) -> str:
    """Run mypy in non-strict mode (catches real bugs without
    drowning in annotation noise). --ignore-missing-imports avoids
    third-party stub gaps."""
    cmd = [
        sys.executable, "-X", "utf8", "-m", "mypy",
        "--ignore-missing-imports",
        "--no-error-summary",
        "--no-color-output",
        *paths,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    except FileNotFoundError:
        # [P159] Unreachable in practice: the executable is sys.executable,
        # which by definition exists. Kept only for the pathological case.
        print("ERROR: mypy not installed. Run: pip install mypy", file=sys.stderr)
        sys.exit(2)
    out = r.stdout + r.stderr
    # [P159] `python -m mypy` with mypy absent exits 1 and prints
    # "No module named mypy" to stderr — it does NOT raise FileNotFoundError,
    # so the guard above never fired. parse_errors() then found no
    # "error: ... [code]" lines and reported ZERO findings, which is
    # indistinguishable from a clean tree. Under `ci_check_invariants --update`
    # that silently rewrote the baseline from 1080 findings to 0, which would
    # then make the gate fail with +1080 on any machine that DOES have mypy.
    # A missing tool is a broken check, never a passing one.
    if r.returncode != 0 and "No module named mypy" in out:
        raise MypyUnavailable(sys.executable)
    return out


def parse_errors(output: str) -> Dict[str, int]:
    """Group errors by [error-code] tag. Mypy lines look like:
    file.py:123: error: message  [union-attr]
    """
    by_code: Counter = Counter()
    pattern = re.compile(r"error:.*?\[([a-z-]+)\]")
    for line in output.splitlines():
        m = pattern.search(line)
        if m:
            by_code[m.group(1)] += 1
    return dict(by_code)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="Override default critical-dirs scope")
    ap.add_argument("--baseline-format", action="store_true",
                    help="Emit count-only JSON for ci_check_invariants")
    args = ap.parse_args()

    paths = args.paths if args.paths else CRITICAL_DIRS
    try:
        output = run_mypy(paths)
    except MypyUnavailable as err:
        # [P159] Report unavailability as DATA, not as zero findings. The
        # consumer (ci_check_invariants) carries the previous baseline forward
        # and prints a SKIPPED banner rather than silently passing a check
        # that never ran.
        print(f"[lint_mypy_baseline] UNAVAILABLE: {err}", file=sys.stderr)
        if args.baseline_format:
            print(json.dumps({"unavailable": str(err)}, indent=2, sort_keys=True))
            return 0
        return 2

    by_code = parse_errors(output)
    total = sum(by_code.values())

    if args.baseline_format:
        # Sorted for deterministic output
        result = {
            "by_code": dict(sorted(by_code.items())),
            "total_count": total,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if total == 0:
        print("[lint_mypy_baseline] OK — 0 errors.")
        return 0

    print(f"[lint_mypy_baseline] {total} errors across {len(by_code)} codes:")
    for code, count in sorted(by_code.items(), key=lambda x: -x[1]):
        print(f"  [{code}]: {count}")
    print(
        "\nMypy baseline locks counts; new errors block CI. "
        "To rebaseline (e.g. after legitimate code change):\n"
        "  python tools/ci_check_invariants.py --update"
    )
    return 0  # Diagnostic-only; ci_check_invariants does the gate


if __name__ == "__main__":
    sys.exit(main())

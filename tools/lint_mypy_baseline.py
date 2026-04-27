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
        return r.stdout + r.stderr
    except FileNotFoundError:
        print("ERROR: mypy not installed. Run: pip install mypy", file=sys.stderr)
        sys.exit(2)


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
    output = run_mypy(paths)
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

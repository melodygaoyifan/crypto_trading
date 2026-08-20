"""[P344] Which commit added this mypy finding? -- the recipe, as a tool.

WHY. When the gate goes red the tempting move is `--update`, which bakes the
finding in as the new floor (P171/P175). The honest move is to measure two
trees and diff the ERROR SETS, and I have now hand-rolled that twice: P338b
(one new error, from an import that pulled a lab into CRITICAL_DIRS for the
first time) and again this session. Both times the worktrees leaked.

Two things this encodes that the hand-rolled version kept getting wrong:

  * LINE NUMBERS ARE STRIPPED before comparing. A file that gained insertions
    reports every downstream finding as "new" otherwise, which is how a
    one-error delta reads as thirty and sends you re-baselining.
  * The comparison is between two COMMITS, not between a commit and the
    working tree -- in a shared checkout the working tree contains someone
    else's edits, and the answer would be about them (P311b/P328).

CRITICAL_DIRS is imported from the baseline scanner rather than restated, so
this cannot drift from the gate it is diagnosing (P172).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.lint_mypy_baseline import CRITICAL_DIRS  # noqa: E402
from tools.scratch_worktree import scratch_worktree  # noqa: E402

# "path:line: error: message  [code]" -> drop the line number only
_LINE = re.compile(r"^(?P<path>[^:]+):\d+:(?P<rest>.*)$")


def normalize(line: str) -> Optional[str]:
    """A finding, identified by file + message + code but NOT by position."""
    if ": error:" not in line:
        return None
    m = _LINE.match(line.strip())
    if not m:
        return line.strip()
    return m.group("path").replace("\\", "/") + ":" + m.group("rest").strip()


def run_mypy(tree: Path, dirs: List[str], cache: Path) -> Set[str]:
    present = [d for d in dirs if (tree / d).exists()]
    if not present:
        return set()
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "mypy", "--ignore-missing-imports",
         "--no-error-summary", "--cache-dir", str(cache), *present],
        capture_output=True, text=True, encoding="utf-8", cwd=str(tree),
        timeout=3600)
    out: Set[str] = set()
    for line in (r.stdout or "").splitlines():
        n = normalize(line)
        if n:
            out.add(n)
    return out


def compare(base_ref: str, head_ref: str,
            dirs: Optional[List[str]] = None) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    dirs = dirs or list(CRITICAL_DIRS)
    with scratch_worktree(base_ref, prefix="hmats-attr-base-") as base:
        base_set = run_mypy(base, dirs, base / ".mypy_cache_attr")
        with scratch_worktree(head_ref, prefix="hmats-attr-head-") as head:
            head_set = run_mypy(head, dirs, head / ".mypy_cache_attr")
    return (head_set - base_set, base_set - head_set,
            {"base": len(base_set), "head": len(head_set)})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Diff mypy findings between two commits, line-agnostic.")
    ap.add_argument("base", help="the last known-good commit")
    ap.add_argument("head", nargs="?", default="HEAD")
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="default: CRITICAL_DIRS, imported from the gate")
    a = ap.parse_args(argv)

    added, fixed, counts = compare(a.base, a.head, a.dirs)
    print("base " + a.base + ": " + str(counts["base"]) + " findings")
    print("head " + a.head + ": " + str(counts["head"]) + " findings")
    print("")
    print("ADDED by " + a.head + " (" + str(len(added)) + "):")
    for f in sorted(added):
        print("  + " + f)
    print("FIXED by " + a.head + " (" + str(len(fixed)) + "):")
    for f in sorted(fixed):
        print("  - " + f)
    if not added:
        print("")
        print("No finding was introduced. A raw count delta against the "
              "committed baseline is then the P227 environment fingerprint, "
              "NOT code drift -- do not re-baseline it.")
    return 1 if added else 0


if __name__ == "__main__":
    sys.exit(main())

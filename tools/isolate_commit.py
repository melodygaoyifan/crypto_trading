"""[P350] Stage only YOUR hunks from a file a parallel session is also editing.

THE PROBLEM, eleven P-number collisions deep in this repo. Two sessions share
one working tree. You edit main.py; so do they. Then:

  * `git add main.py && git commit`  commits the whole INDEX, including
    whatever they staged (P314, and P311b in the other direction).
  * `git commit -- main.py`          commits the WORKING TREE for that path,
    which contains their uncommitted hunks (P349 nearly did this; P304 and
    P314 actually did, in both directions).

There is no incantation of add/commit that means "my changes to this file".
This does mean that: it takes the diff against HEAD, keeps only the hunks
whose ADDED lines carry your marker, and applies those to the INDEX with
`git apply --cached` — which never touches the working tree, so their work
stays exactly where it is.

THE VERIFICATION IS THE POINT. Staging the right bytes is easy to believe and
easy to get wrong, so it re-reads the staged blob and refuses when your marker
is missing or when a FOREIGN marker appeared. "I checked the diff looked fine"
is what the previous eleven collisions all had.

    python -X utf8 tools/isolate_commit.py --marker "[P350]" main.py
    git commit -m "..."          # commits the index, not the working tree

Deliberately NOT a committer: it stages and verifies, and you write the commit
yourself. A tool that commits is one that will eventually commit the wrong
thing unattended.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

MARKER_RE = re.compile(r"\[P\d{1,4}[a-z]?\]")


def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], capture_output=True, timeout=600)
    if binary:
        return r
    return subprocess.CompletedProcess(
        r.args, r.returncode,
        r.stdout.decode("utf-8", "replace"),
        r.stderr.decode("utf-8", "replace"))


def split_hunks(diff: str) -> Tuple[List[str], List[str]]:
    """(header_lines, hunks) for a single-file unified diff."""
    lines = diff.splitlines(keepends=True)
    head: List[str] = []
    hunks: List[str] = []
    cur: List[str] = []
    for ln in lines:
        if ln.startswith("@@"):
            if cur:
                hunks.append("".join(cur))
            cur = [ln]
        elif cur:
            cur.append(ln)
        else:
            head.append(ln)
    if cur:
        hunks.append("".join(cur))
    return head, hunks


def select_hunks(hunks: Sequence[str], marker: str) -> Tuple[List[str], List[str]]:
    """(mine, theirs) by whether an ADDED line carries the marker."""
    mine, theirs = [], []
    for h in hunks:
        added = [ln for ln in h.splitlines() if ln.startswith("+")]
        (mine if any(marker in ln for ln in added) else theirs).append(h)
    return mine, theirs


def foreign_markers(text: str, marker: str, baseline: str) -> List[str]:
    """Markers present in `text` that are neither yours nor already in HEAD."""
    base = set(MARKER_RE.findall(baseline))
    found = set(MARKER_RE.findall(text))
    return sorted(m for m in found - base if m != marker)


def isolate(path: str, marker: str, apply: bool = True) -> int:
    head = _git("show", f"HEAD:{path}").stdout
    diff = _git("diff", "HEAD", "--", path).stdout
    if not diff.strip():
        print(f"{path}: no changes vs HEAD — nothing to isolate")
        return 1
    header, hunks = split_hunks(diff)
    mine, theirs = select_hunks(hunks, marker)
    print(f"{path}: {len(hunks)} hunk(s) — {len(mine)} carry {marker}, "
          f"{len(theirs)} do not")
    if not mine:
        print(f"REFUSING: no hunk carries {marker}. Mark your changes, or you "
              f"cannot tell them from anyone else's.")
        return 2
    if not apply:
        return 0

    patch = "".join(header) + "".join(mine)
    # index := HEAD for this path, then apply only my hunks to the index.
    # --cached leaves the WORKING TREE untouched, which is what keeps the
    # other session's uncommitted hunks safe.
    r = _git("reset", "-q", "HEAD", "--", path)
    if r.returncode != 0:
        print("REFUSING: could not reset the index for", path, r.stderr[:200])
        return 2
    proc = subprocess.run(["git", "apply", "--cached", "--unidiff-zero", "-"],
                          input=patch.encode("utf-8"), capture_output=True,
                          timeout=600)
    if proc.returncode != 0:
        print("REFUSING: could not apply the isolated patch to the index:",
              proc.stderr.decode("utf-8", "replace")[:400])
        return 2

    staged = _git("show", f":{path}").stdout
    if marker not in staged:
        print(f"REFUSING: {marker} is absent from the STAGED blob — the patch "
              f"applied but did not carry your change.")
        return 2
    alien = foreign_markers(staged, marker, head)
    if alien:
        print(f"REFUSING: staged blob carries foreign marker(s) {alien} that "
              f"are not in HEAD — someone else's uncommitted work would be "
              f"swept into your commit (P314).")
        return 2
    print(f"staged {path}: {marker} present, no foreign markers. "
          f"Commit the INDEX (`git commit`), not a pathspec.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--marker", required=True,
                    help='e.g. "[P350]" — must appear on your added lines')
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    for p in a.paths:
        rc = max(rc, isolate(p, a.marker, apply=not a.dry_run))
    return rc


if __name__ == "__main__":
    sys.exit(main())

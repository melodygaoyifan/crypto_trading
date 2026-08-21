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

[P352b] AND ITS VERIFICATION HAD A BLIND SPOT THAT COST A RED CI. Both clauses
— "my marker is present" and "no foreign marker appeared" — are true of a
PARTIAL commit. Two hunks of mine carried no marker (a function SIGNATURE line
and a config PARSE line; the explaining comment sat in a neighbouring hunk),
so they were dropped silently, the tool printed success, and HEAD went red with
142 failures because every caller of the new kwarg hit a TypeError.

The tool cannot know whose an UNMARKED hunk is — so it must not decide, it must
SHOW. It now prints every dropped hunk with its location and added lines, and
REFUSES when a dropped hunk carries no marker of any kind, which is exactly the
ambiguous case. `--accept-unmarked` is the explicit escape for when you have
read them and they are theirs.

"3 hunks do not carry your marker" was a COUNT. A count is not a location
(P293b/P349), and that is the whole of this fix.

[P357] AND THE SENTENCE THAT USED TO SIT HERE — "a dropped hunk that carries
somebody else's marker is attributable and passes quietly" — WAS FALSE, and it
cost a second red CI. A hunk that EDITS a line stamped by an earlier entry of
YOUR OWN campaign carries that earlier marker in its added text and is
indistinguishable, to this tool, from another session's work. Five of six
dispatch-site rewrites here modified lines tagged `# [P356]`, so `--marker
"[P357]"` dropped them silently and the commit went out with one sixth of the
change in it. P352b closed the unmarked case and REASONED the marked case away;
foreign-marked drops are now listed as a NOTE (not a refusal — see
describe_foreign_dropped for why). Stamp such a line with BOTH markers.
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


def campaign_markers(marker: str) -> "re.Pattern[str]":
    """[P357c] `[P357]` must also mean `[P357b]`, `[P357c]`, …

    A sub-entry suffix makes the marker a DIFFERENT LITERAL, so a substring
    match on the base number silently drops every sub-entry hunk — which is
    what happened here on the third commit of this entry: the `_bg_handed_off`
    initialiser, the hand-off branch and the heartbeat segment all carried
    `# [P357b]`, none contained the substring `[P357]`, and all three were
    classified as somebody else's work and left behind. Suffixed sub-entries
    are this repo's normal way of recording a follow-up (P322b–f, P329b–d,
    P355b, P357b), so the base number IS the campaign.

    Deliberately exact on the NUMBER: `[P357]` must not match `[P35]` or
    `[P3570]`, or isolating one entry would sweep a neighbour's.
    """
    base = marker.strip().strip("[]").rstrip("abcdefghijklmnopqrstuvwxyz")
    return re.compile(r"\[" + re.escape(base) + r"[a-z]?\]")


def select_hunks(hunks: Sequence[str], marker: str) -> Tuple[List[str], List[str]]:
    """(mine, theirs) by whether an ADDED line carries the marker's campaign."""
    pat = campaign_markers(marker)
    mine, theirs = [], []
    for h in hunks:
        added = [ln for ln in h.splitlines() if ln.startswith("+")]
        (mine if any(pat.search(ln) for ln in added) else theirs).append(h)
    return mine, theirs


def _render_hunk(h: str) -> str:
    lines = h.splitlines()
    added = [ln for ln in lines if ln.startswith("+")]
    loc = lines[0] if lines else "@@ ?"
    shown = [("      " + ln) for ln in added[:6]]
    if len(added) > 6:
        shown.append("      ... %d more added line(s)" % (len(added) - 6))
    return "  " + loc + "\n" + "\n".join(shown)


def describe_dropped(theirs: Sequence[str], marker: str) -> List[str]:
    """[P352b] Render the dropped hunks that carry NO marker of any kind.

    Those are the ambiguous ones: an unmarked dropped hunk may be yours, which
    is the case that produced a partial commit and a red CI.
    """
    out: List[str] = []
    for h in theirs:
        added = [ln for ln in h.splitlines() if ln.startswith("+")]
        if any(MARKER_RE.search(ln) for ln in added):
            continue          # carries SOME marker — see describe_foreign
        out.append(_render_hunk(h))
    return out


def describe_foreign_dropped(theirs: Sequence[str],
                             marker: str) -> List[Tuple[str, str]]:
    """[P357] Render the dropped hunks that carry a marker other than yours.

    P352b's docstring asserted these "need no attention" because they are
    attributable to another session. **That is false in the common case and it
    cost a red CI**: a hunk that EDITS a line already stamped with an earlier
    marker of your own campaign carries that earlier marker in its added text.
    Five of six dispatch-site rewrites here rewrote lines tagged `# [P356]`,
    so `--marker "[P357]"` classified them as somebody else's and dropped them
    **silently** — the same silent partial commit P352b was built to end, one
    case over, invisible because the refusal only covered UNMARKED hunks.

    Reported as a WARNING rather than a refusal, deliberately. In a shared
    tree most foreign hunks really are foreign, and refusing on every one
    would make the tool unusable for the situation it exists for — a guard
    that fires on the normal case gets bypassed (P202/P303). Naming the
    markers is enough: the author recognises their own campaign's number.
    """
    pat = campaign_markers(marker)
    out: List[Tuple[str, str]] = []
    for h in theirs:
        added = [ln for ln in h.splitlines() if ln.startswith("+")]
        # [P357c] "not mine" means outside the CAMPAIGN, not merely a
        # different literal — otherwise a sub-entry marker of your own reads
        # as foreign in the report as well as in the selection.
        found = sorted({m for ln in added for m in MARKER_RE.findall(ln)
                        if not pat.fullmatch(m)})
        if found:
            out.append((", ".join(found), _render_hunk(h)))
    return out


def foreign_markers(text: str, marker: str, baseline: str) -> List[str]:
    """Markers present in `text` that are neither yours nor already in HEAD."""
    pat = campaign_markers(marker)   # [P357c] the campaign, not one literal
    base = set(MARKER_RE.findall(baseline))
    found = set(MARKER_RE.findall(text))
    return sorted(m for m in found - base if not pat.fullmatch(m))


def isolate(path: str, marker: str, apply: bool = True,
            accept_unmarked: bool = False) -> int:
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

    # [P352b] A count is not a location. Show what is being LEFT OUT, because
    # a dropped hunk of your own is invisible to every check below — all of
    # them pass on a partial commit.
    unmarked = describe_dropped(theirs, marker)
    if unmarked and not accept_unmarked:
        print(f"REFUSING: {len(unmarked)} dropped hunk(s) in {path} carry NO "
              f"marker at all, so this tool cannot tell whether they are "
              f"yours. A hunk of yours dropped here is silent — every check "
              f"below still passes (P352b). Read them, then either add "
              f"{marker} to the ones that are yours or pass "
              f"--accept-unmarked:")
        for d in unmarked:
            print(d)
        return 2

    # [P357] A dropped hunk carrying ANOTHER marker is not automatically
    # somebody else's — it is usually YOUR edit to a line your own earlier
    # entry stamped. Warn, do not refuse (see describe_foreign_dropped).
    foreign = describe_foreign_dropped(theirs, marker)
    if foreign:
        print(f"NOTE: {len(foreign)} dropped hunk(s) carry a marker that is "
              f"not {marker}. That usually means another session — but a hunk "
              f"of YOURS that edits a line stamped by an earlier entry of "
              f"your own campaign looks identical, and is dropped silently "
              f"(P357). Confirm none of these is yours:")
        for mk, body in foreign:
            print(f"  [carries {mk}]")
            print(body)
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
    ap.add_argument("--accept-unmarked", action="store_true",
                    help="proceed even though some dropped hunks carry no "
                         "marker — only after reading the ones it lists")
    a = ap.parse_args(argv)
    rc = 0
    for p in a.paths:
        rc = max(rc, isolate(p, a.marker, apply=not a.dry_run,
                             accept_unmarked=a.accept_unmarked))
    return rc


if __name__ == "__main__":
    sys.exit(main())

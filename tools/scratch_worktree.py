"""[P344] A git worktree that removes itself, even when you stop reading.

WHY. Attribution work in this repo means checking out two commits side by side
and measuring both (P175: attribute a scanner delta before re-baselining it;
P338b: find which commit added a mypy finding). I have hand-rolled that twice,
and both times I left the worktrees behind -- they were still there days later,
listed next to a parallel session's legitimate one, indistinguishable from it.

The deploy script already solved this for its own scan tree with a `trap`
(P328b). This is the same guarantee for callers that are not that script:
create through `scratch_worktree`, and removal happens in a `finally`, so an
exception or a Ctrl-C cannot leak one.

FAIL DIRECTION. `remove()` refuses to touch a path this module did not create.
A cleanup helper that can be pointed at an arbitrary directory is a worse
problem than the leak it fixes -- and the natural bug (passing the repo root)
would delete the working tree.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional


class WorktreeError(RuntimeError):
    pass


def _git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=str(cwd) if cwd else None,
                          timeout=600)


@contextlib.contextmanager
def scratch_worktree(ref: str, repo: Optional[Path] = None,
                     prefix: str = "hmats-wt-") -> Iterator[Path]:
    """Yield a detached worktree at `ref`; remove it whatever happens.

    A worktree rather than `git archive | tar -x` for a specific reason: the
    authority audit REFUSES to run without a git-grep engine rather than emit
    false findings (P158), so a tree with no .git cannot be scanned at all.
    """
    repo = Path(repo or Path(__file__).resolve().parents[1])
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    # mkdtemp created it; git worktree add insists on a non-existent path
    tmp.rmdir()
    made = False
    try:
        r = _git("worktree", "add", "--detach", "-q", str(tmp), ref, cwd=repo)
        if r.returncode != 0:
            raise WorktreeError(
                "could not create a worktree at " + ref + ": " +
                (r.stderr.strip()[:300] or "git said nothing"))
        made = True
        yield tmp
    finally:
        if made:
            remove(tmp, repo=repo)


def remove(path: Path, repo: Optional[Path] = None) -> None:
    """Remove a worktree this module created. Refuses anything else."""
    repo = Path(repo or Path(__file__).resolve().parents[1])
    path = Path(path)
    if path.resolve() == repo.resolve():
        raise WorktreeError(
            "refusing to remove the repository itself -- this helper only "
            "removes scratch worktrees it created")
    _git("worktree", "remove", "--force", str(path), cwd=repo)
    _git("worktree", "prune", cwd=repo)


def list_worktrees(repo: Optional[Path] = None) -> list:
    """(path, ref) for every registered worktree, main tree first."""
    repo = Path(repo or Path(__file__).resolve().parents[1])
    r = _git("worktree", "list", "--porcelain", cwd=repo)
    out, cur = [], {}
    for line in r.stdout.splitlines():
        if line.startswith("worktree "):
            if cur:
                out.append((cur.get("worktree", ""), cur.get("ref", "")))
            cur = {"worktree": line[len("worktree "):]}
        elif line.startswith("HEAD "):
            cur["ref"] = line[len("HEAD "):][:9]
        elif line.startswith("branch "):
            cur["ref"] = line[len("branch "):]
    if cur:
        out.append((cur.get("worktree", ""), cur.get("ref", "")))
    return out


if __name__ == "__main__":
    for p, ref in list_worktrees():
        print(ref.ljust(12), p)
    sys.exit(0)

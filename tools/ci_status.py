"""[P344] The single implementation of "is CI green for this commit?".

WHY THIS EXISTS. The correct version of this check already lived inside
`scripts/hetzner_deploy.sh` -- it derives the repo slug from the git remote,
keeps only the NEWEST run per workflow (P287), and refuses on anything that is
not green (P233/P253b). But it lived in bash, reachable only by running a
deploy, so every ad-hoc "did CI pass?" got hand-rolled again, worse. I did that
twice in one session, and the second time I hardcoded a repo slug that does not
exist.

THE DEFECT WORTH THE MODULE IS NOT THE TYPO. My hand-rolled poller printed

    [1] none-yet
    ... nineteen more times ...

for a repository GitHub had never heard of, and then for a rate limit it had
exhausted asking. "I could not ask the question" was rendered identically to
"the question has no answer yet" -- the P159/P199 conflation, inside a retry
loop, which is what turns a typo into twenty minutes and a spent API budget.

So the two properties this module exists to hold:

  1. A slug is DERIVED, never typed. `slug_from_remote` is the only source.
  2. UNREADABLE and NOT-YET-STARTED get different exit codes, and a poller
     MUST NOT retry an unreadable answer. Retrying a question the API refused
     is exactly how the budget is burned: a 403 will not change inside the
     polling window, and a 404 will never change at all.

EXIT CODES (distinct on purpose -- collapsing any two recreates the defect):
    0  GREEN      every required workflow completed successfully
    1  RED        at least one completed with a non-success conclusion
    2  UNREADABLE repo unknown, rate limited, network down, unparseable
                  -> do NOT retry; this is a fact about US, not about CI
    3  PENDING    runs exist and at least one is queued/in_progress
    4  MISSING    no run yet for one or more required workflows
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

API = "https://api.github.com"
REQUIRED_WORKFLOWS = ("codebase-invariants", "test-suite")

GREEN, RED, UNREADABLE, PENDING, MISSING = 0, 1, 2, 3, 4

# Exit codes for which asking again might legitimately change the answer.
# UNREADABLE is deliberately NOT here -- see the module docstring.
RETRYABLE = (PENDING, MISSING)


class Unreadable(Exception):
    """We could not ask the question. Never means "the answer is no"."""


def slug_from_remote(url: str) -> str:
    """owner/repo from any remote form. The ONLY way a slug is produced."""
    s = url.strip()
    s = re.sub(r"^git@github\.com:", "", s)
    s = re.sub(r"^ssh://git@github\.com/", "", s)
    s = re.sub(r"^https?://[^/]*github\.com/", "", s)
    s = re.sub(r"\.git$", "", s)
    s = s.strip("/")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", s):
        raise Unreadable(
            "cannot derive an owner/repo slug from remote " + repr(url) +
            "; refusing to guess one (a guessed slug reports 'no runs yet' "
            "forever, which is how a typo reads as a healthy pending build)")
    return s


def current_slug(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True,
            text=True, encoding="utf-8", cwd=cwd, timeout=30)
    except OSError as e:
        raise Unreadable("git is not runnable: " + type(e).__name__ + str(e))
    if out.returncode != 0:
        raise Unreadable("no origin remote: " + out.stderr.strip()[:120])
    return slug_from_remote(out.stdout)


def classify(runs: Iterable[dict],
             required: Sequence[str] = REQUIRED_WORKFLOWS) -> Tuple[int, str]:
    """Pure. Runs arrive NEWEST-FIRST; keep the FIRST match per workflow.

    [P287] The old deploy bug here was an unconditional overwrite, which let
    the OLDEST run win: a sha whose first run was green and whose re-run went
    red read as GREEN and deployed. Backwards fail direction on a safety gate.
    """
    need = set(required)
    got: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for r in runs:
        n = r.get("name")
        if n in need and n not in got:
            got[n] = (r.get("status"), r.get("conclusion"))
    missing = sorted(need - set(got))
    if missing:
        return MISSING, "no run yet for: " + ", ".join(missing)
    pending = sorted(k for k, (s, _) in got.items() if s != "completed")
    if pending:
        return PENDING, " ".join(k + "=" + str(got[k][0]) for k in sorted(got))
    bad = sorted(k for k, (_, c) in got.items() if c != "success")
    if bad:
        return RED, " ".join(k + "=" + str(got[k][1]) for k in sorted(got))
    return GREEN, " ".join(sorted(got))


def fetch_runs(slug: str, sha: str, token: Optional[str] = None,
               timeout: float = 30.0) -> List[dict]:
    """Raise Unreadable for every "could not ask" case, with the reason."""
    url = API + "/repos/" + slug + "/actions/runs?head_sha=" + sha + "&per_page=50"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hmats-ci-status",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = str(json.loads(
                e.read().decode("utf-8", "replace")).get("message", ""))[:160]
        except Exception:  # noqa: silent-swallow -- detail is best-effort
            pass
        if e.code == 404:
            raise Unreadable(
                "GitHub has no repository " + repr(slug) + " (404). The slug "
                "is derived from `git remote get-url origin` -- check the "
                "remote. Do NOT retry: a 404 will never become a run.")
        if e.code in (403, 429):
            reset = e.headers.get("X-RateLimit-Reset")
            remaining = e.headers.get("X-RateLimit-Remaining")
            when = ""
            if reset and str(reset).isdigit():
                when = " resets " + time.strftime(
                    "%H:%M:%SZ", time.gmtime(int(reset)))
            raise Unreadable(
                "GitHub refused the request (" + str(e.code) + ", remaining=" +
                str(remaining) + when + "): " + detail + ". Polling again "
                "spends budget without changing the answer -- wait for the "
                "reset or pass --token.")
        raise Unreadable("HTTP " + str(e.code) + " from GitHub: " + detail)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise Unreadable("network error: " + type(e).__name__ + ": " + str(e))
    try:
        payload = json.loads(body)
    except ValueError as e:
        raise Unreadable("unparseable response: " + str(e))
    if not isinstance(payload, dict) or "workflow_runs" not in payload:
        # a 200 carrying {"message": ...} is still us being unable to ask
        msg = ""
        if isinstance(payload, dict):
            msg = str(payload.get("message", ""))[:160]
        raise Unreadable("unexpected response shape: " + (msg or str(type(payload))))
    runs = payload["workflow_runs"]
    if not isinstance(runs, list):
        raise Unreadable("workflow_runs is not a list")
    return runs


def status(sha: str, slug: Optional[str] = None, token: Optional[str] = None,
           required: Sequence[str] = REQUIRED_WORKFLOWS) -> Tuple[int, str]:
    slug = slug or current_slug()
    return classify(fetch_runs(slug, sha, token=token), required)


def _head_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, encoding="utf-8", timeout=30)
    if out.returncode != 0:
        raise Unreadable("cannot resolve HEAD")
    return out.stdout.strip()


VERDICT_NAME = {GREEN: "GREEN", RED: "RED", PENDING: "PENDING",
                MISSING: "MISSING"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Report whether CI is green for a commit.")
    ap.add_argument("--sha", help="commit to check (default: HEAD)")
    ap.add_argument("--slug", help="owner/repo (default: derived from origin)")
    ap.add_argument("--token", help="GitHub token (raises the rate limit)")
    ap.add_argument("--wait-seconds", type=float, default=0.0,
                    help="poll until a terminal verdict, up to this long")
    ap.add_argument("--interval", type=float, default=45.0)
    ap.add_argument("--max-requests", type=int, default=20,
                    help="hard cap on API calls, whatever --wait-seconds says")
    a = ap.parse_args(argv)

    try:
        sha = a.sha or _head_sha()
        slug = a.slug or current_slug()
    except Unreadable as e:
        print("UNREADABLE: " + str(e))
        return UNREADABLE

    deadline = time.monotonic() + max(0.0, a.wait_seconds)
    calls = 0
    while True:
        try:
            code, detail = status(sha, slug=slug, token=a.token)
        except Unreadable as e:
            # NOT retried, deliberately: this is a fact about us, not about CI.
            print("UNREADABLE: " + str(e))
            return UNREADABLE
        calls += 1
        print(VERDICT_NAME[code] + " " + slug + "@" + sha[:9] + " -- " + detail)
        if code not in RETRYABLE:
            return code
        if calls >= a.max_requests or time.monotonic() + a.interval > deadline:
            return code
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

"""[P311] Assert a GUARD is live, not merely present in the source.

THE TRAP THIS CLOSES, three sightings deep
    `assert "<condition>" in src` survives `if False and <condition>:` — the
    substring is still there, so the pin proves the code was WRITTEN, not
    that it RUNS. P234 (gate hysteresis), P251 (the stale-snapshot guard) and
    P307 (the GMM shape guard, whose first falsification probe stayed green)
    were all this.

    The durable fix is a callable predicate, and P307 did that where the
    guard was worth extracting. But extracting a predicate out of five more
    live modules — the GCI flow path, the feed-degradation counter, the
    whale seat — is a large change to working code for guards that are
    currently correct. This helper closes the same hole from the test side:
    it requires the condition to be the ENTIRE condition of its statement,
    so any prefix (`False and`, `not`, an extra clause) fails.

WHAT IT DOES NOT DO
    It is a source assertion and it stays one. It cannot tell you the branch
    is reachable, only that nobody has neutered it in place. When a guard is
    load-bearing enough to deserve more, extract the predicate and call it —
    `data_mgmt.market_data_pipeline.gmm_shape_mismatch` is the pattern.
"""
from __future__ import annotations

import re

_KEYWORDS = ("if", "elif", "while")


def assert_guard_live(src: str, condition: str, why: str = "",
                      near: str = "") -> None:
    """Fail unless `condition` is the whole condition of an if/elif/while.

    `condition` may be given with or without its leading keyword and
    trailing colon, so existing pins can be moved over verbatim.

    [P337] `near` restricts the search to a window around a unique nearby
    string. Without it the pin passes as long as SOME statement in the file
    uses the condition — so a file with two `if book is not None:` sites is
    pinned by whichever one the author did not mean, and neutering the other
    stays green. The falsification harness caught exactly that; anchor
    uniqueness is never something to assume (P238), and it is no more safe
    inside a shared helper than in a one-off test.
    """
    if near:
        i = src.find(near)
        if i < 0:
            raise AssertionError(
                "anchor %r is absent, so the guard could not be located. %s"
                % (near, why))
        if src.find(near, i + 1) >= 0:
            raise AssertionError(
                "anchor %r occurs more than once, so it cannot identify a "
                "single guard site. %s" % (near, why))
        src = src[max(0, i - 400):i + 400]
    cond = condition.strip().rstrip(":").strip()
    for kw in _KEYWORDS:
        if cond.startswith(kw + " "):
            cond = cond[len(kw) + 1:].strip()
            break
    pat = re.compile(
        r"^[ \t]*(?:%s)[ \t]+%s[ \t]*:" % ("|".join(_KEYWORDS), re.escape(cond)),
        re.M)
    if pat.search(src):
        return
    loose = cond in src
    raise AssertionError(
        ("guard %r is present but NOT the whole condition of its statement "
         "— something was prefixed to it (`if False and ...`), which a plain "
         "substring pin cannot see. %s"
         if loose else
         "guard %r is absent from the source. %s") % (cond, why))


# [P330] The same trap in a file the AST cannot help with.
#
# `assert_guard_live` needs a Python condition. Shell scripts, YAML and
# Dockerfiles have guards too, and there the equivalent defeat is simply
# COMMENTING THE LINE OUT — the substring survives, so `assert "<line>" in src`
# stays green over dead code. Observed in P328b: a pin on the deploy script's
# cleanup trap passed with the trap commented out, and the falsification
# harness reported it VACUOUS.
# Deliberately just these two. `--` was in the first version and cut the very
# line it was meant to protect: shell long flags (`--force`, `--detach`) are
# preceded by whitespace and look exactly like a SQL comment, so the pinned
# text vanished and every guard read as dead. Caught by the falsification
# harness within minutes — the same over-broad-detector mistake P330 had just
# fixed in the condition-pin scanner. A caller that genuinely needs `--` or
# `REM` passes `markers=`.
_COMMENT_PREFIXES = ("#", "//")


def _code_part(line: str, markers=_COMMENT_PREFIXES) -> str:
    """The line with any trailing comment removed.

    A leading-marker check alone is not enough: `true # trap ...` does not
    START with a comment, yet the pinned text is dead. The falsification
    harness caught exactly that in the first version of this helper.

    A marker counts when it opens the line or follows whitespace, which is the
    shell/Python convention. A marker inside a quoted string is therefore cut
    too — deliberately, because for a PIN the safe error is "this looks
    commented" (fails loudly) rather than "this looks live" (passes over dead
    code).
    """
    out = line
    for marker in markers:
        idx = 0
        while True:
            i = out.find(marker, idx)
            if i < 0:
                break
            if i == 0 or out[i - 1].isspace():
                out = out[:i]
                break
            idx = i + 1
    return out


def assert_live_line(src: str, text: str, why: str = "",
                     markers=_COMMENT_PREFIXES) -> None:
    """Fail unless `text` appears on a line that is NOT commented out.

    Deliberately narrow: it checks the line carrying `text` does not START
    with a comment marker. It cannot see a block comment, a heredoc or an
    `if false` wrapper — for a Python condition use `assert_guard_live`, and
    for anything load-bearing extract the predicate and CALL it. What it does
    close is the cheapest and most common defeat, which is a `#`.
    """
    hits = [ln for ln in src.splitlines() if text in ln]
    if not hits:
        raise AssertionError(
            f"{text!r} does not appear at all. {why}".strip())
    live = [ln for ln in hits if text in _code_part(ln, markers)]
    if not live:
        raise AssertionError(
            f"{text!r} appears only on COMMENTED lines — the pin would pass "
            f"over dead code (P328b/P330). {why}".strip())


# ---------------------------------------------------------------------------
# [P350] Two more shapes, both from traps this repo has now hit repeatedly.
# ---------------------------------------------------------------------------

def assert_text_pin(src: str, needle: str, why: str = "",
                    near: str = "", window: int = 400) -> None:
    """Fail unless `needle` appears, and REFUSE when it is ambiguous.

    THE TRAP (P349, and P293b one level up): a pin asserts a substring that a
    SIBLING occurrence also satisfies, so the pin stays green when the site it
    was written for is deleted. Mine asserted
    `"self._last_whale_counts[asset] = 0" in src` while the set-half's
    exception fallback assigns the identical text — the reset could be removed
    and the test never noticed. **A count is not a location.**

    So an ambiguous needle is an ERROR rather than a pass: either the pin is
    about a specific site, in which case give `near` and it is checked inside
    that window, or it is genuinely about "this text exists anywhere", in
    which case say so with `near=""` AND a needle that occurs once.

    `near` must itself be unique, for the same reason (P238/P337).

    WHAT IT DOES NOT DO, stated because the first version of this docstring
    implied otherwise and its own test caught it: `near` anchors the search,
    it does not EXCLUDE a sibling that happens to sit inside the window. If
    the site you meant is deleted and an identical string exists 200 chars
    later, the scoped pin still passes. The load-bearing half is the
    ambiguity REFUSAL above — that is what would have caught P349, whose pin
    had no anchor at all — and the window is a convenience whose tightness is
    the caller's judgement. Keep it small, and anchor on something adjacent.
    """
    n = src.count(needle)
    if n == 0:
        raise AssertionError(f"{needle!r} does not appear at all. {why}".strip())
    if not near:
        if n > 1:
            raise AssertionError(
                f"{needle!r} appears {n} times, so this pin cannot tell which "
                f"occurrence survived — a sibling satisfies it and the site "
                f"you meant can be deleted silently (P349). Pass near=<unique "
                f"anchor> to scope it. {why}".strip())
        return
    a = src.count(near)
    if a != 1:
        raise AssertionError(
            f"anchor {near!r} occurs {a} times; an ambiguous anchor scopes the "
            f"pin to a window you did not read (P238). {why}".strip())
    i = src.index(near)
    if needle not in src[i:i + window]:
        raise AssertionError(
            f"{needle!r} does not appear within {window} chars of {near!r} — "
            f"it exists elsewhere in the file, which is exactly the ambiguity "
            f"this check exists to catch. {why}".strip())


def assert_detector_is_precise(detector, must_catch, must_not_catch,
                               why: str = "") -> None:
    """Force a hand-rolled detector to declare BOTH directions.

    THE TRAP, three sightings: P307 (a condition-pin scanner matching English
    prose containing "not"), P330 (`--` treated as a comment marker, cutting
    the shell long flags it was meant to protect), P349 (an observation-only
    guard flagging a `hasattr` initialisation as a decision). Every one was an
    over-broad detector, and the tempting fix is always to loosen it until the
    author's own case passes — which is how a real guard dies (P248).

    Recall is the direction that goes quiet: a detector that stops catching
    anything reports clean, and clean is what a healthy tree also reports
    (P174). So both lists are required and neither may be empty.

    `detector` takes one string and returns something truthy when it fires.
    """
    if not must_catch:
        raise AssertionError(
            "must_catch is empty: a detector with no positive example cannot "
            "be shown to fire at all (P174). " + why)
    if not must_not_catch:
        raise AssertionError(
            "must_not_catch is empty: precision is the half that sends the "
            "next author to loosen the detector (P248). " + why)
    missed = [s for s in must_catch if not detector(s)]
    if missed:
        raise AssertionError(
            f"detector FAILED TO CATCH {len(missed)} case(s) it must: "
            f"{missed!r}. A detector that cannot fire reports clean. "
            f"{why}".strip())
    false_pos = [s for s in must_not_catch if detector(s)]
    if false_pos:
        raise AssertionError(
            f"detector FIRED on {len(false_pos)} case(s) it must not: "
            f"{false_pos!r}. Fix its PRECISION — do not relax it until your "
            f"own case passes (P248/P330). {why}".strip())

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


def assert_guard_live(src: str, condition: str, why: str = "") -> None:
    """Fail unless `condition` is the whole condition of an if/elif/while.

    `condition` may be given with or without its leading keyword and
    trailing colon, so existing pins can be moved over verbatim.
    """
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

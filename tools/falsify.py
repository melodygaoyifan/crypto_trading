"""[P328] A falsification-probe harness that fails when a PROBE is vacuous.

This repo falsifies its guards by hand: edit the production file to reintroduce
the defect, run the tests, confirm they go red, restore. The discipline works —
it has caught real vacuous guards (P174, P238, P293d, P319) — but the harness
was an ad-hoc script per P-entry, and its verdict was a line of output a human
had to NOTICE:

    PROBE a stamped module is unregistered  -> 37 passed

That line means THE PROBE FAILED TO PROBE, and it looks exactly like a line
that means the guard is fine. In P327 it was mine: the probe added a COMMENT,
which matched its anchor exactly once and changed the file without changing any
behaviour, so the guard was never exercised and the report read as success.

The fix is that a probe declares what it expects to break, and a probe that
breaks nothing is an ERROR rather than a line of output:

    from tools.falsify import Probe, run_probes
    ok = run_probes([
        Probe(name="silent advisor buckets as disagreement",
              path="analytics/ic/agent_disagreement_review.py",
              old='if advisor_sig is None or bad or abs(ad) < 1e-9:',
              new='if False:',
              expect_red=["tests/test_p324_disagreement_instrument.py"]),
    ])

WHAT IT ENFORCES, and each clause is one way a probe lies:

  * the anchor matches EXACTLY once — zero means the probe never applied
    (an "ANCHOR MISS" that is easy to skim past); more than one means it
    edited somewhere you did not read (P238 anchor-uniqueness). [P357] When
    the text you want genuinely repeats -- two call sites, two classes with
    the same __init__, a handler and the one below it -- pass `near=` and the
    probe is resolved WITHIN that window instead of refused, which is what
    `assert_text_pin` already does for pins (P350). `near` must itself be
    unique, for the same reason.
  * the edit CHANGES THE FILE — a no-op replacement is a probe that tests
    nothing.
  * the named tests were GREEN BEFORE — probing an already-red suite proves
    nothing about the probe.
  * the named tests go RED after. This is the clause that would have caught
    the P327 comment probe.
  * the file is restored BYTE-IDENTICALLY, verified by comparison rather than
    by having written the original back (P265: probe reversal must never be a
    tree checkout, which reverts to HEAD instead of to the pre-probe state).

It restores in a `finally`, so an interrupted run cannot leave a deliberate
defect in the tree.
"""
from __future__ import annotations

import io
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Probe:
    name: str
    path: str
    old: str
    new: str
    expect_red: Sequence[str]           # pytest targets that MUST fail
    near: str = ""                      # [P357] disambiguates a repeated `old`
    near_window: int = 600              # chars either side of `near` to search
    skip_green_check: bool = False      # only when the pre-check is expensive
    result: Optional[str] = field(default=None, init=False)
    detail: str = field(default="", init=False)


def _pytest(targets: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", *targets,
         "-q", "-p", "no:randomly"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO),
        timeout=1800)


def _summary(r: subprocess.CompletedProcess) -> str:
    lines = [ln for ln in (r.stdout or "").splitlines()
             if "passed" in ln or "failed" in ln or "error" in ln]
    return lines[-1].strip() if lines else f"(rc={r.returncode})"


def run_probe(p: Probe) -> bool:
    """True when the probe genuinely falsified its guards."""
    target = REPO / p.path
    try:
        original = io.open(target, encoding="utf-8").read()
    except OSError as e:
        p.result, p.detail = "INVALID", f"cannot read {p.path}: {e}"
        return False

    n = original.count(p.old)
    if n != 1 and p.near:
        # [P357] SCOPE it instead of sending the author away to hand-expand
        # the anchor. Sibling ambiguity has now cost four INVALID rounds in
        # three entries, every time because two call sites, two classes with
        # identical __init__ signatures, or a handler and the handler below it
        # share the text. The harness caught all four — the waste was the
        # round trip, not a missed defect — so `near` closes the loop the way
        # assert_text_pin's `near` does for pins (P350): one concept, both
        # tools, rather than a fifth reminder to be careful.
        if original.count(p.near) != 1:
            p.result = "INVALID"
            p.detail = (f"near={p.near!r} occurs {original.count(p.near)} "
                        f"times; an ambiguous ANCHOR for an ambiguous needle "
                        f"scopes the probe to a window you did not read "
                        f"(P238).")
            return False
        # [P357] BIDIRECTIONAL. The first cut searched forward only, and its
        # own demonstration failed on that: the distinguishing text
        # (`logging.getLogger('BinanceTaker')`) sits AFTER the line being
        # probed. "near" means nearby, not after — and a helper that silently
        # means the latter sends the author looking for the wrong thing.
        # (`assert_text_pin` is forward-only and documents it; that is the
        # right trade for a PIN, where the anchor is chosen to precede.)
        # [P357] TIGHT by default. The first cut used 4000 chars, and its own
        # test exposed that as no scoping at all for anything but a large
        # file: two siblings a few lines apart both land inside the window and
        # the probe is refused for "2 within the window", which is the same
        # dead end `near` exists to remove. P350 made this exact correction to
        # `assert_text_pin` (1500 -> 400) for the same reason. A window that is
        # too small only ever REFUSES, which is the safe direction; too large
        # silently fails to discriminate.
        W = p.near_window
        centre = original.index(p.near)
        lo = max(0, centre - W)
        hi = min(len(original), centre + len(p.near) + W)
        window = original[lo:hi]
        if window.count(p.old) != 1:
            p.result = "INVALID"
            p.detail = (f"anchor matched {n} times overall and "
                        f"{window.count(p.old)} within {W} chars either side "
                        f"(widen or narrow with near_window=) "
                        f"of near={p.near!r} — needs exactly 1 there.")
            return False
        at = lo + window.index(p.old)
        mutated = original[:at] + p.new + original[at + len(p.old):]
        n = 1
    elif n != 1:
        p.result = "INVALID"
        p.detail = (f"anchor matched {n} times (need exactly 1). Zero means "
                    f"the probe never applied; more than one means it edited "
                    f"somewhere you did not read."
                    + (" Pass near=<unique nearby text> to scope it (P357)."
                       if n > 1 else ""))
        return False
    else:
        mutated = original.replace(p.old, p.new)
    if mutated == original:
        p.result, p.detail = "INVALID", "the replacement is a no-op"
        return False

    if not p.skip_green_check:
        before = _pytest(p.expect_red)
        if before.returncode != 0:
            p.result = "INVALID"
            p.detail = (f"the targets were ALREADY RED before the probe "
                        f"({_summary(before)}) — a probe against a red suite "
                        f"proves nothing")
            return False

    try:
        io.open(target, "w", encoding="utf-8", newline="").write(mutated)
        after = _pytest(p.expect_red)
        if after.returncode == 0:
            p.result = "VACUOUS"
            p.detail = (f"the guards stayed GREEN with the defect present "
                        f"({_summary(after)}). The probe changed the file but "
                        f"not the behaviour, or the guard does not cover this "
                        f"defect. Distrust the PROBE first (P238).")
            return False
        p.result, p.detail = "OK", _summary(after)
        return True
    finally:
        io.open(target, "w", encoding="utf-8", newline="").write(original)
        restored = io.open(target, encoding="utf-8").read()
        if restored != original:
            p.result = "NOT_RESTORED"
            p.detail = (f"{p.path} did NOT restore byte-identically — the tree "
                        f"may still carry the deliberate defect")


def run_probes(probes: List[Probe], verbose: bool = True) -> bool:
    """Run all probes. Returns True only if EVERY probe genuinely falsified."""
    ok = True
    for p in probes:
        good = run_probe(p)
        ok = ok and good and p.result == "OK"
        if verbose:
            print(f"[{p.result:<13}] {p.name}\n               {p.detail}")
    if verbose:
        print("\nALL PROBES FALSIFIED THEIR GUARDS" if ok else
              "\nONE OR MORE PROBES DID NOT FALSIFY — see above. A probe that "
              "breaks nothing is not evidence that a guard works.")
    return ok

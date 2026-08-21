"""[P363] Two classes I had been closing one instance at a time.

The operator quoted back two of my own sentences. Both describe a class, and
in both cases I had handled the instance and left the class open — which is
P171/P226's finding ("a mitigation applied to one instance of a class is not
applied to the class"), for the sixth and seventh recorded time.

--------------------------------------------------------------------------
1. THE PIPELINE-EXIT TRAP — and the honest result is that the repo was fine
--------------------------------------------------------------------------
`cmd | tail` followed by `$?` reports TAIL's status. I hit it interactively:
`echo "exit=$?"` printed 0 for a tool that had correctly returned 2, which is
P185's trap firing in my own shell.

Audited before assuming: **8 of 9 shell scripts already set `pipefail`**, and
the ninth (`hmats_monitor.sh`) captures output rather than branching on `$?`
and already appends `|| true` where it matters. So there was no live defect —
the deploy gate is safe and the trap was mine alone. That negative result is
worth as much as a finding would have been, because "I hit this, therefore
the codebase has it" is exactly the inference that wastes an afternoon.

What was open was the CLASS: nothing required the next script to set it. Now
something does.

--------------------------------------------------------------------------
2. A HELPER IS ONLY REAL ONCE SOMETHING OUTSIDE ITS OWN TEST USES IT
--------------------------------------------------------------------------
P345 recorded the rule (`infra/classified_retry.py` had been written for a
real class and had ZERO callers, so it prevented none of the five incidents
it was designed for). P359 and P360 each then hand-rolled their own
"caller outside its own test" check — **twice, in two days, differently.**

One roster-driven guard replaces both: every public helper in the shared test
modules must be used somewhere that is not its own test file. The roster is
DERIVED from the modules, not restated, so a new helper is covered the moment
it is written rather than when someone remembers to add a check (P310/P172).
"""

import ast
import pathlib
import re

import main

REPO = pathlib.Path(main.__file__).parent
TESTS = pathlib.Path(__file__).parent
HELPER_MODULES = ("_guard_pins.py", "_cli_harness.py", "_source_scan.py")


# ==========================================================================
# 1. Every shell script sets pipefail
# ==========================================================================
def test_every_shell_script_sets_pipefail():
    """Without it a pipeline reports its LAST stage's status, so `cmd | grep`
    reads grep's code and a failing cmd looks fine (P185). The deploy path
    already had it — this stops the next script from not."""
    # [P363] Must match a `set` COMMAND, not the bare word: the first cut
    # scanned for the substring and went VACUOUS on its own probe, because
    # the comment explaining the fix in hmats_monitor.sh says "pipefail"
    # three times. A scanner that matches its own explanation is P177, and
    # this repo has now hit it five times.
    _SET_PIPEFAIL = re.compile(r"^\s*set\s+-\S*o\S*\s+pipefail", re.M)
    missing = []
    for f in sorted((REPO / "scripts").glob("*.sh")):
        if not _SET_PIPEFAIL.search(
                f.read_text(encoding="utf-8", errors="replace")):
            missing.append(f.name)
    assert not missing, (
        f"shell scripts without `set -o pipefail`: {missing}. A pipeline's "
        f"status is its last stage's, so any `$?` or `if cmd | grep` after "
        f"one silently reads the filter (P185/P363)."
    )


def test_the_deploy_path_specifically_still_has_it():
    """The one where it is load-bearing: hetzner_deploy.sh gates on CI and on
    the scanner suite, and both are `if ...; then` over commands whose status
    must be their own (P253b/P344)."""
    src = (REPO / "scripts" / "hetzner_deploy.sh").read_text(encoding="utf-8")
    assert re.search(r"^set -[a-z]*o[a-z]* pipefail|^set -o pipefail", src,
                     re.M), "the deploy script lost pipefail"


def test_the_guard_would_catch_a_new_script(tmp_path, monkeypatch):
    """Anti-vacuity (P174): a scan that cannot fire reports clean, and clean
    is what a healthy tree also reports."""
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "good.sh").write_text("set -euo pipefail\n", encoding="utf-8")
    (d / "bad.sh").write_text("set -e\n", encoding="utf-8")
    (d / "comment_only.sh").write_text(
        "# we should set -o pipefail here one day\nset -e\n", encoding="utf-8")
    _SET_PIPEFAIL = re.compile(r"^\s*set\s+-\S*o\S*\s+pipefail", re.M)
    missing = [f.name for f in sorted(d.glob("*.sh"))
               if not _SET_PIPEFAIL.search(f.read_text(encoding="utf-8"))]
    assert missing == ["bad.sh", "comment_only.sh"], (
        "a script that only MENTIONS pipefail in a comment must still be "
        "flagged — that is the P177 false negative"
    )


# ==========================================================================
# 2. Every shared test helper has a caller outside its own test
# ==========================================================================
def _public_helpers():
    """Derived from the modules, never restated — a hardcoded mirror is the
    drift this guard exists to catch (P310/P172)."""
    out = {}
    for name in HELPER_MODULES:
        p = TESTS / name
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not n.name.startswith("_"):
                out[n.name] = name
    return out


def _files_using(symbol):
    return [q.name for q in sorted(TESTS.glob("test_*.py"))
            if re.search(rf"\b{re.escape(symbol)}\b",
                         q.read_text(encoding="utf-8", errors="replace"))]


# A helper whose ONLY referencing file is also the file that tests it. Each
# entry must name the real adoption site, and the guard checks that file
# actually references the symbol — an exemption that names nothing is how a
# roster becomes a parking spot (the P361 rule, same shape).
_ADOPTED_IN_ITS_OWN_TEST_FILE = {
    "assert_live_line": (
        "test_p328_falsify_harness.py",
        "pins the deploy script's cleanup trap — the real site P330 built it "
        "for; that pin and the helper's own tests happen to share one file"),
}


def test_every_shared_helper_is_actually_used():
    """[P363] P345's rule, mechanised. `infra/classified_retry.py` was written
    for a real class and had ZERO callers, so it prevented none of the five
    incidents it was designed for — a helper nothing calls is decoration
    (P170). P359 and P360 each hand-rolled this check for their own helper,
    twice in two days; this replaces both and covers the ones nobody wrote a
    check for.

    The rule is >= 2 referencing FILES, because a helper's own test file
    always references it — so one file means nothing has adopted it. My first
    cut instead tried to identify 'its own test' by NAME and would have given
    a false pass; `assert_live_line` is the case that exposed it."""
    helpers = _public_helpers()
    assert helpers, "no public helpers found — the scan is broken, not clean"
    unused = {}
    for sym, module in sorted(helpers.items()):
        files = _files_using(sym)
        if len(files) >= 2 or sym in _ADOPTED_IN_ITS_OWN_TEST_FILE:
            continue
        unused[sym] = (module, files)
    assert not unused, (
        f"shared test helpers with no adopter: {unused}. A helper nothing "
        f"calls is decoration (P170/P345) — adopt it at a real site, delete "
        f"it, or add an entry to _ADOPTED_IN_ITS_OWN_TEST_FILE naming where."
    )


def test_no_adoption_exemption_is_a_parking_spot():
    """The other direction (P310/P361): an exemption must name a file that
    really references the symbol, or it is coverage that is not."""
    for sym, (fname, reason) in _ADOPTED_IN_ITS_OWN_TEST_FILE.items():
        f = TESTS / fname
        assert f.exists(), f"{sym}: exemption names a missing file {fname}"
        assert re.search(rf"\b{re.escape(sym)}\b",
                         f.read_text(encoding="utf-8", errors="replace")), (
            f"{sym}: {fname} does not reference it — the exemption is stale"
        )
        assert len(reason) > 30, f"{sym}: exemption gives no reason"


def test_the_roster_is_derived_not_restated():
    """If the helper list were hardcoded here, a new helper would be
    uncovered until someone remembered — the exact failure this replaces."""
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    i = src.index("def _public_helpers")
    body = src[i:i + 900]
    assert "ast.parse" in body, "the roster is not read from the modules"
    for known in ("assert_text_pin", "run_cli"):
        assert f'"{known}"' not in body, (
            f"{known} is hardcoded into the roster — it must be discovered"
        )

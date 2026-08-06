"""[P177] A risk controller that is imported, logged as loaded, and never called.

main.py imported seven symbols from risk/short_position_controller.py and
analytics/sota_metrics_calculator.py, used none of them, set a flag nothing
read, and logged on every boot:

    [OK]V6 SOTA modules loaded (short risk + metrics)

`get_short_controller()` had no call site anywhere in the repo, so
`assess_risk`, `check_stop_loss` and `get_position_size_multiplier` have never
run in production. The log line was literally true and materially false: an
operator reading it concludes short-side risk is governed.

Live short risk is defense/short_control.py, invoked at main.py:10577. This
file pins three things:

  1. the misleading import block does not come back,
  2. the controller does not silently acquire a production call site,
  3. the path that IS live stays live — so if somebody deletes short_control's
     call site, this fails rather than leaving the system with two dead short
     controllers and nothing guarding shorts.

Point 3 is the one that matters. Points 1-2 alone would be satisfied by a
system with no short risk control at all.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_scan import code_only, read_source  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN = REPO_ROOT / "main.py"
DEAD = REPO_ROOT / "risk" / "short_position_controller.py"
LIVE = REPO_ROOT / "defense" / "short_control.py"

# Directories that are not the running system.
_EXCLUDED = ("/.git/", "/archive/", "/legacy/", "/tests/", "/scripts/",
             "/tools/", "/docs/")


def _production_py():
    for p in REPO_ROOT.rglob("*.py"):
        s = "/" + str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        if any(x in s for x in _EXCLUDED):
            continue
        yield p


_read = read_source


def _code(p):
    """Source with `#` comments stripped, string literals preserved.

    These tests search for call sites and for a log string. The P177 fix
    replaced the offending code with a comment block that *quotes* the log
    line and names `get_short_controller()`, so the next reader understands
    why they are gone — which made the first draft of these regexes match
    their own documentation and fail. A scanner that cannot tell code from
    prose about code is not measuring what it claims to.

    Literals are kept (`strip_docstrings=False`) because the banner under test
    IS a literal (`logger.info("...V6 SOTA modules loaded...")`); blanking
    strings would make that test unable to fail. P179 needed the opposite
    setting for the same reason in reverse — see tests/_source_scan.py.
    """
    return code_only(p, strip_docstrings=False)


class TestTheMisleadingLoadBannerIsGone:
    def test_main_does_not_claim_short_risk_is_loaded(self):
        src = _code(MAIN)
        assert "V6 SOTA modules loaded" not in src, (
            "main.py logs that short risk modules are loaded. Loaded is not "
            "wired: unless get_short_controller() now has a call site, this "
            "line tells the operator a risk actuator is running when it is not."
        )

    def test_main_does_not_import_what_it_never_uses(self):
        """Import-and-never-use is how the banner became believable."""
        tree = ast.parse(_read(MAIN))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                "short_position_controller" in node.module
                or "sota_metrics_calculator" in node.module
            ):
                imported |= {(a.asname or a.name) for a in node.names}
        if not imported:
            return
        src = _code(MAIN)
        for name in sorted(imported):
            # >1 because the import statement itself is one occurrence.
            assert len(re.findall(rf"\b{re.escape(name)}\b", src)) > 1, (
                f"main.py imports {name} and never uses it. Either wire it in "
                f"deliberately or drop the import; an unused import of a risk "
                f"module is what P177 was."
            )


class TestTheControllerStillHasNoProductionCaller:
    def test_get_short_controller_is_uncalled(self):
        callers = []
        for p in _production_py():
            if p == DEAD:
                continue
            src = _code(p)
            if re.search(r"\bget_short_controller\s*\(", src) or re.search(
                r"\bShortPositionController\s*\(", src
            ):
                callers.append(str(p.relative_to(REPO_ROOT)))
        assert not callers, (
            f"risk/short_position_controller.py now has production callers: "
            f"{callers}. That is a live-system change, not a cleanup — its "
            f"stop-loss, daily-loss halt and squeeze sizing paths have never "
            f"executed against a real fill. Reconcile it with "
            f"defense/short_control.py (two controllers clamping the same "
            f"exposure is worse than one), then update this test to record "
            f"the decision."
        )

    def test_the_module_says_so_at_the_top(self):
        head = _read(DEAD)[:2000]
        assert "NOT WIRED" in head, (
            "the NOT WIRED banner is the only thing standing between the next "
            "reader and the assumption that this file guards live shorts"
        )


class TestTheShortRiskThatIsActuallyLiveStaysLive:
    """The half that would still pass if shorts were left ungoverned."""

    def test_short_control_is_constructed_and_evaluated_in_main(self):
        src = _code(MAIN)
        assert "from defense.short_control import" in src
        assert re.search(r"_short_control\s*=\s*ShortControl\(", src), (
            "main.py no longer constructs ShortControl"
        )
        assert re.search(r"_short_control\.evaluate\(", src), (
            "ShortControl is constructed but never evaluated. Both short "
            "controllers are now dead and nothing governs short exposure."
        )

    def test_evaluate_is_reached_on_short_intents(self):
        """Guarded by direction < 0 — pin it, or the call is unreachable."""
        src = _code(MAIN)
        m = re.search(r"if self\._short_control is not None and "
                      r"intent\.direction\s*<\s*0", src)
        assert m, (
            "the ShortControl.evaluate() guard changed shape. If the "
            "direction condition inverted or was dropped, short risk either "
            "never runs or runs on longs."
        )

    def test_the_two_controllers_have_not_both_been_wired(self):
        src = _code(MAIN)
        live = bool(re.search(r"_short_control\.evaluate\(", src))
        dead = bool(re.search(r"\bget_short_controller\s*\(", src))
        assert not (live and dead), (
            "main.py now drives BOTH short controllers. They clamp exposure "
            "independently and neither knows about the other."
        )

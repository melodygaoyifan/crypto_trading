"""[P357] Two limitations I recorded as prose and left without a mechanism —
a background loop whose death is unobservable, and a falsification anchor that
cannot be scoped.

The operator quoted both paragraphs back. That move has now produced P344, P350
and this entry, always from the same reading: **a lesson recorded in prose is
the thing this file exists to convert into a mechanism (P280/P328).**

--------------------------------------------------------------------------
1. "ITS SILENCE PROVES NOTHING EITHER WAY."
--------------------------------------------------------------------------
P356 fixed the discarded-Task reference and then had to report the fix as
LATENT rather than live, because it could not tell a healthy background loop
from a dead one: `OnChainFeed.start()` logs only on a *caught* fetch error, so
an empty log is exactly what a running feed and a collected one both produce.

The half nobody had named is worse than the weak reference, and it is a
property of asyncio rather than of this repo: **a Task that raises STORES the
exception and reports it to nobody.** `task.exception()` has to be called for
it to surface. So the failure mode written in the comment above the defect —
"OnChainFeed ... stay dormant -> dead signals for months" — was not merely
unnoticed, it was *unnoticeable*: no traceback, no log line, no exit code.
Holding a reference keeps the task from being collected and does nothing at
all about the task dying.

`_spawn_bg` closes both halves at once, and the SECOND is what makes silence
mean something:
  * the reference is held (P356's fix, now in one place instead of six), and
  * a done-callback retrieves the exception and LOGS it, at a severity that
    matches what happened — ERROR with the traceback for a raise, WARNING for
    a clean return (a loop that is supposed to be endless returning is itself
    a fault), INFO for a cancellation at shutdown.

`background_task_names()` is the POSITIVE half, and it is the one that
answers the original question. A death report answers "did it stop?"; the
heartbeat segment answers "is it running right now?", which had no answer at
all. After this, an empty `bg:` field is evidence.

--------------------------------------------------------------------------
2. SIBLING ANCHOR AMBIGUITY, FOUR TIMES IN THREE ENTRIES
--------------------------------------------------------------------------
P355's pin matched the terminal handler as well as the transient one it meant;
two P356 probes matched both `BinanceTakerMonitor` and `LeadLagAlphaEngine`,
whose `__init__` signatures are identical. Every time the harness caught it
(the refusal is correct and must stay), and every time the author's only
remedy was to hand-expand the anchor into a multi-line block until it happened
to be unique -- which is fragile in the opposite direction, since a wider
anchor breaks on any edit to the lines it swept up.

`assert_text_pin` solved exactly this for PINS in P350 with `near=`. The probe
harness did not have it, so the same concept was missing from the tool where
the same mistake kept happening. `Probe.near` scopes the search to a window
around a unique anchor instead of refusing -- and `near` must itself be
unique, or an ambiguous anchor would scope the probe to a window the author
never read (P238 one level up).

**Its own demonstration then failed twice, on two separate defects of mine,
and both are now pinned below.** The first cut searched FORWARD only, and the
distinguishing text in the real case sits *after* the probed line -- "near"
that means "after" is a helper that silently does something other than what it
says (P350's lesson about a docstring overstating a guarantee). And the second
probe came back VACUOUS: the guard it targeted covered only one of the two
sibling classes, so the defect was genuinely undetectable. **The harness
reporting a gap in the GUARD rather than in the fix is the whole point of
running probes at all**, and that gap (P171/P226: a mitigation applied to one
instance of a class) is closed in the P356 file itself.
"""

import ast
import asyncio
import inspect
import logging
import pathlib
import textwrap

import main
from tools.falsify import Probe, run_probe

REPO = pathlib.Path(main.__file__).parent


class _Runner:
    """The three methods under test, bound to a bare object — constructing the
    real runner needs a config, an exchange and an event loop, and none of
    that is part of what is being pinned (P234: exercise the behaviour)."""

    def __init__(self):
        self._bg_tasks = set()

    _spawn_bg = main.HMATSProductionRunner._spawn_bg
    _on_bg_task_done = main.HMATSProductionRunner._on_bg_task_done
    background_task_names = main.HMATSProductionRunner.background_task_names


def _inside_helper(lineno: int) -> bool:
    src, start = inspect.getsourcelines(main.HMATSProductionRunner._spawn_bg)
    return start <= lineno < start + len(src)


# ==========================================================================
# 1. A background task's DEATH is now observable
# ==========================================================================
def test_a_task_that_raises_is_reported_as_an_error_with_its_cause(caplog):
    """THE defect. Without the callback this is completely silent: asyncio
    stores the exception on the Task and nothing ever retrieves it."""

    async def _boom():
        raise RuntimeError("feed died")

    async def _go():
        r = _Runner()
        t = r._spawn_bg(_boom(), name="onchain_feed")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return r, t

    with caplog.at_level(logging.INFO):
        runner, task = asyncio.run(_go())

    errs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert errs, (
        "a background task raised and NOTHING was logged — this is the exact "
        "silence P356 could not distinguish from health"
    )
    blob = " ".join(r.getMessage() for r in errs)
    assert "onchain_feed" in blob, "the report does not name WHICH task died"
    assert "RuntimeError" in blob and "feed died" in blob, (
        "the report does not carry the cause"
    )
    assert any(r.exc_info for r in errs), "no traceback attached"
    assert task not in runner._bg_tasks, "the dead task is still held"


def test_the_consequence_is_stated_not_just_the_event(caplog):
    """P240: an alert must say what it means for the operator. 'task ended' is
    not actionable; 'the signals it feeds are now stale' is."""

    async def _boom():
        raise ValueError("x")

    async def _go():
        _Runner()._spawn_bg(_boom(), name="lead_lag")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with caplog.at_level(logging.INFO):
        asyncio.run(_go())
    blob = " ".join(r.getMessage() for r in caplog.records)
    assert "stale" in blob.lower(), (
        "the message reports the event without its consequence"
    )


def test_a_clean_return_is_a_warning_because_the_loop_should_not_end(caplog):
    """All three dispatched coroutines are endless loops. Returning normally
    is not success — it is a different fault, and collapsing it into the
    healthy case would make it invisible the way the raise was."""

    async def _ends():
        return None

    async def _go():
        _Runner()._spawn_bg(_ends(), name="sol_onchain")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with caplog.at_level(logging.INFO):
        asyncio.run(_go())
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns, "a supposedly-endless loop returned and nothing said so"
    assert "sol_onchain" in " ".join(r.getMessage() for r in warns)


def test_a_cancellation_at_shutdown_is_routine_not_an_alarm(caplog):
    """The opposite failure: if shutdown produced ERRORs the report becomes
    wallpaper and the real death stops being read (P202)."""

    async def _forever():
        await asyncio.sleep(3600)

    async def _go():
        t = _Runner()._spawn_bg(_forever(), name="onchain_feed")
        await asyncio.sleep(0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)

    with caplog.at_level(logging.INFO):
        asyncio.run(_go())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "a routine shutdown cancellation alarmed"
    )


def test_retrieving_the_exception_can_never_take_the_loop_down():
    """The callback runs inside the event loop. A raise there would convert an
    observability fix into an outage — the P85 shape, in the guard."""
    src = inspect.getsource(main.HMATSProductionRunner._on_bg_task_done)
    tree = ast.parse(textwrap.dedent(src))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "exception"]
    assert calls, "the callback does not retrieve the exception at all"
    guarded = [t for t in ast.walk(tree) if isinstance(t, ast.Try)
               and any(c in list(ast.walk(t)) for c in calls)]
    assert guarded, "task.exception() is not wrapped — it can raise"


def test_the_live_names_are_reported_so_silence_is_evidence():
    """The POSITIVE half. Without this, 'no bg log lines' still cannot tell a
    healthy loop from one that died before the callback was ever attached."""

    async def _forever():
        await asyncio.sleep(3600)

    async def _go():
        r = _Runner()
        r._spawn_bg(_forever(), name="onchain_feed")
        r._spawn_bg(_forever(), name="lead_lag")
        await asyncio.sleep(0)
        names = r.background_task_names()
        for t in list(r._bg_tasks):
            t.cancel()
        return names

    assert asyncio.run(_go()) == ["lead_lag", "onchain_feed"]


def test_a_finished_task_is_not_reported_as_alive():
    """Otherwise the heartbeat would assert liveness for a dead loop — worse
    than saying nothing, because it reads as a measurement (P2/P223)."""

    async def _ends():
        return None

    async def _go():
        r = _Runner()
        r._spawn_bg(_ends(), name="gone")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return r.background_task_names()

    assert asyncio.run(_go()) == []


def test_the_done_check_covers_the_window_before_the_callback_runs():
    """[P357] The test above passes on the DISCARD alone, so it left the
    `done()` check unpinned — a falsification probe replacing it with `True`
    stayed GREEN. Found by the harness, not by reading (P238).

    The check is not redundant: `add_done_callback` schedules via `call_soon`,
    so between a task completing and its callback running the task is done AND
    still in the set. Reported as alive there, the heartbeat would assert
    liveness for a loop that has already stopped. This constructs that window
    directly instead of hoping to land in it."""

    async def _ends():
        return None

    async def _go():
        r = _Runner()
        t = asyncio.ensure_future(_ends())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert t.done()
        r._bg_tasks.add(t)          # the pre-callback window, by construction
        return r.background_task_names()

    assert asyncio.run(_go()) == [], (
        "a completed task still in the holder was reported as running"
    )


def test_the_heartbeat_actually_carries_the_liveness_segment():
    """A mechanism nothing calls is decoration (P170). The heartbeat is where
    the operator already looks."""
    src = inspect.getsource(main.HMATSProductionRunner.run_live)
    assert "background_task_names()" in src, (
        "the liveness list is computed nowhere on the live path"
    )
    i = src.index("background_task_names()")
    assert "NONE RUNNING" in src[max(0, i - 800):i + 800], (
        "an empty list must render as a loud statement, not as an empty "
        "string that reads like a healthy field"
    )


def test_every_dispatch_site_goes_through_the_helper():
    """The P356 finding was that six sites each had to remember; one helper is
    the fix, and a seventh site written the old way would be silent again."""
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8",
                                                  errors="replace"))
    raw = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Attribute)
           and n.func.attr in ("create_task", "ensure_future")
           and not _inside_helper(n.lineno)]
    assert not raw, (
        f"raw create_task/ensure_future outside the helper at {raw} — those "
        f"tasks die silently"
    )


# ==========================================================================
# 2. Probe.near — scoping a genuinely repeated anchor
# ==========================================================================
# Two sibling classes with identical __init__ signatures, separated by a
# method body — the geometry of the real case (lead_lag_engine.py). The
# separation is load-bearing: a fixture where both siblings fit inside any
# sane window cannot tell scoping from no scoping, which is precisely how the
# first 4000-char default looked correct.
_FILLER = "".join("        self.f%d = %d\n" % (i, i) for i in range(40))

REPEATED = (
    "class A:\n"
    "    def __init__(self, x=None):\n"
    "        self.log = mklog('AAA')\n"
    "\n"
    "    def work(self):\n"
    + _FILLER +
    "\n"
    "MIDDLE_MARKER = 1\n"
    "\n"
    "class B:\n"
    "    def __init__(self, x=None):\n"
    "        self.log = mklog('BBB')\n"
)

OLD = "    def __init__(self, x=None):"
NEW = "    def __init__(self, x=[]):"


def _rc(code, out="1 failed"):
    return type("R", (), {"returncode": code, "stdout": out, "stderr": ""})()


def _stateful(path, marker):
    """GREEN until the probe's mutation lands, RED after — the shape a real
    guard has. A fake that is red from the start trips the harness's own
    "ALREADY RED" refusal, which is correct behaviour and would make these
    tests measure that clause instead of the one they name."""

    def _run(_targets):
        src = path.read_text(encoding="utf-8")
        return _rc(1) if marker in src else _rc(0, "ok")

    return _run


def test_without_near_a_repeated_anchor_is_still_refused(tmp_path,
                                                         monkeypatch):
    """The refusal is CORRECT and must survive — it is what caught all four
    ambiguities. `near` adds a way to proceed, it does not relax the rule."""
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text(REPEATED, encoding="utf-8")
    p = Probe(name="t", path="s.py", old=OLD, new=NEW,
              expect_red=["missing.py"])
    assert run_probe(p) is False
    assert p.result == "INVALID"
    assert "near=" in p.detail, (
        "the refusal does not tell the author the remedy — that is what sent "
        "them to hand-expand the anchor four times"
    )


def test_near_resolves_the_intended_occurrence(tmp_path, monkeypatch):
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text(REPEATED, encoding="utf-8")
    seen = {}

    def _capture(_targets):
        seen["src"] = (tmp_path / "s.py").read_text(encoding="utf-8")
        return _rc(1) if "x=[]" in seen["src"] else _rc(0, "ok")

    monkeypatch.setattr(fz, "_pytest", _capture)
    p = Probe(name="t", path="s.py", old=OLD, new=NEW,
              near="mklog('BBB')", expect_red=["missing.py"])
    assert run_probe(p) is True, p.detail
    mutated = seen["src"]
    before, after = mutated.split("class B:")
    assert after.count("x=[]") == 1, "the wrong sibling was edited"
    assert "x=[]" not in before, (
        "class A was edited — the whole point of `near` is that it was not"
    )


def test_near_looks_BOTH_directions(tmp_path, monkeypatch):
    """My first cut searched forward only, and its own demonstration failed on
    it: the distinguishing text sat AFTER the probed line. A helper whose name
    says 'nearby' must not silently mean 'after' (P350)."""
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text(REPEATED, encoding="utf-8")
    monkeypatch.setattr(fz, "_pytest", _stateful(tmp_path / "s.py", "x=[]"))
    p = Probe(name="t", path="s.py", old=OLD, new=NEW,
              near="mklog('BBB')",   # sits AFTER the line being probed
              expect_red=["missing.py"])
    assert run_probe(p) is True, (
        f"a `near` anchor placed after the target was refused: {p.detail}"
    )


def test_an_ambiguous_near_is_itself_refused(tmp_path, monkeypatch):
    """P238 one level up: scoping by an ambiguous anchor points the probe at a
    window the author never read, which is the failure `near` exists to end."""
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text(REPEATED, encoding="utf-8")
    p = Probe(name="t", path="s.py", old=OLD, new=NEW,
              near="def __init__", expect_red=["missing.py"])
    assert run_probe(p) is False
    assert p.result == "INVALID" and "near=" in p.detail


def test_a_near_that_does_not_scope_is_refused(tmp_path, monkeypatch):
    """A unique anchor with both occurrences inside its window resolves
    nothing — refusing beats silently picking the closer one. Also exercises
    `near_window`, which exists because the right width is a property of the
    file: too small only ever refuses (safe), too large silently stops
    discriminating (which is how the first 4000-char cut looked correct)."""
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text(REPEATED, encoding="utf-8")
    p = Probe(name="t", path="s.py", old=OLD, new=NEW,
              near="MIDDLE_MARKER", near_window=5000,
              expect_red=["missing.py"])
    assert run_probe(p) is False
    assert p.result == "INVALID"
    assert "either side" in p.detail


def test_near_changes_nothing_when_the_anchor_is_already_unique(tmp_path,
                                                                monkeypatch):
    """Behaviour-neutral by default: every existing probe passes no `near`."""
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text("UNIQUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(fz, "_pytest",
                        _stateful(tmp_path / "s.py", "UNIQUE = 2"))
    p = Probe(name="t", path="s.py", old="UNIQUE = 1", new="UNIQUE = 2",
              expect_red=["missing.py"])
    assert run_probe(p) is True, p.detail


def test_a_no_op_replacement_is_still_caught_under_near(tmp_path, monkeypatch):
    """The other clauses must keep applying inside the new branch — a scoped
    probe that changes nothing is as vacuous as an unscoped one."""
    import tools.falsify as fz
    monkeypatch.setattr(fz, "REPO", tmp_path)
    (tmp_path / "s.py").write_text(REPEATED, encoding="utf-8")
    p = Probe(name="t", path="s.py", old=OLD, new=OLD,
              near="mklog('BBB')", expect_red=["missing.py"])
    assert run_probe(p) is False
    assert p.result == "INVALID" and "no-op" in p.detail


# ==========================================================================
# 3. isolate_commit — the SECOND silent-drop case (P357)
# ==========================================================================
from tools.isolate_commit import (                      # noqa: E402
    describe_dropped, describe_foreign_dropped, select_hunks)

_HUNK_MINE = (
    "@@ -1,3 +1,3 @@\n"
    " ctx\n"
    "-old\n"
    "+new  # [P357]\n"
)
_HUNK_UNMARKED = (
    "@@ -10,3 +10,3 @@\n"
    " ctx\n"
    "-a\n"
    "+b\n"
)
# The case that cost a red CI: MY edit to a line stamped by MY OWN earlier
# entry. Indistinguishable, to a marker match, from another session's hunk.
_HUNK_MY_OLD_MARKER = (
    "@@ -20,3 +20,3 @@\n"
    " ctx\n"
    "-        self._bg_tasks.add(asyncio.create_task(x()))  # [P356]\n"
    "+        self._spawn_bg(x(), name=\"x\")  # [P356]\n"
)


def test_a_hunk_carrying_an_earlier_marker_of_your_own_is_surfaced():
    """[P357] P352b refused on UNMARKED drops and reasoned the marked ones
    away as 'attributable'. They are not: five of six dispatch-site rewrites
    edited lines tagged `# [P356]` and were dropped silently, so the commit
    went out with one sixth of the change and CI went red."""
    hunks = [_HUNK_MINE, _HUNK_MY_OLD_MARKER]
    mine, theirs = select_hunks(hunks, "[P357]")
    assert len(mine) == 1 and len(theirs) == 1
    assert describe_dropped(theirs, "[P357]") == [], (
        "it carries a marker, so it is not the UNMARKED case"
    )
    foreign = describe_foreign_dropped(theirs, "[P357]")
    assert len(foreign) == 1, "the drop is still silent"
    markers, body = foreign[0]
    assert "[P356]" in markers, "the report does not name the marker"
    assert "_spawn_bg" in body, (
        "the report does not show the added line, so the author cannot "
        "recognise it as their own — a count is not a location (P293b)"
    )


def test_the_unmarked_case_is_unchanged():
    """P352b's refusal must survive: it is the stronger of the two, and this
    change must not trade one silent drop for another."""
    mine, theirs = select_hunks([_HUNK_MINE, _HUNK_UNMARKED], "[P357]")
    assert len(describe_dropped(theirs, "[P357]")) == 1
    assert describe_foreign_dropped(theirs, "[P357]") == [], (
        "an unmarked hunk must not also be reported as foreign — it would be "
        "listed twice with two different remedies"
    )


def test_a_dual_marked_line_is_recognised_as_yours():
    """The remedy the NOTE points at: stamp the line with BOTH markers. If
    that did not work the advice would be wrong, which is worse than silence."""
    dual = _HUNK_MY_OLD_MARKER.replace('# [P356]\n', '# [P356] [P357]\n')
    mine, theirs = select_hunks([dual], "[P357]")
    assert len(mine) == 1 and not theirs


def test_foreign_drops_warn_rather_than_refuse():
    """A guard that fires on the NORMAL case gets bypassed (P202/P303). In a
    shared tree most foreign hunks really are foreign, so refusing on every
    one would make the tool unusable for the situation it exists for."""
    src = pathlib.Path(
        REPO / "tools" / "isolate_commit.py").read_text(encoding="utf-8")
    i = src.index("foreign = describe_foreign_dropped(")
    after = src[i:i + 900]
    assert "return 2" not in after.split("for mk, body in foreign:")[0], (
        "the foreign-marker report refuses instead of warning"
    )

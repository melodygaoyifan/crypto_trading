"""[P356] The entry filters disarmed, and a codebase sweep for async-lifecycle
defects — one real, three clean.

PART 1 — THE DISARM.
The operator asked why the market fluctuates with no trade. Decomposed over 6
days / 336 asset-ticks: 75% is the certified book choosing FLAT by design
(ETH trend_flat 39 vs trend_hold 31; SOL trend_flat 65 vs hold 8; BTC carries
the funding legs and did 17 trades). Of the alpha gate's 340 refusals, 70 are
`est=0` — nothing to gate — and 172 of the remaining 270 are at `est=10bps`,
the trend seat's 2-of-3 vote, which P295 showed is untradeable by construction.

The one NON-design constraint: **of ETH's 31 actionable ticks, 23 were
blocked** — 16 MA_VETO, 6 HOLD, 1 WHALE_VETO. Shown that, the operator
instructed disarming both filters. Nothing measured has ever distinguished
either from noise in either direction (P324 NOT EARNED at the pre-committed
bar; P337, against the decider they actually filter, found model_alpha's
disagreements marked entries that did BETTER; P348: no obtainable sample makes
it significant). Both ledgers keep accruing with enforcement off (P340), so
the only thing given up is the blocking.

PART 2 — THE SWEEP. Four mechanically-findable classes, none swept before:

  unawaited coroutines .......... 0 real (32 candidates, all name collisions
                                  with sync methods, or coroutines put in a
                                  list and star-unpacked into gather)
  blocking I/O in async def ..... 0 in the engine (2 in an operator-run CLI)
  discarded task references ..... 1 REAL, latent — fixed here
  mutable default arguments ..... 2 on the live path — fixed here
  persisted-state schema drift .. 0 (23 keys written, 23 read back)

The real one: `asyncio.create_task(...)` whose Task is discarded. asyncio's
own docs say "Save a reference to the result of this function, to avoid a task
disappearing mid-execution" — the loop keeps only a WEAK reference. Of the
three dispatched background loops it matters most for `OnChainFeed.start()`,
which runs its polling loop INLINE, so the Task IS the loop; the other two
spawn tracked inner tasks and return. **P37 fixed exactly this class INSIDE
lead_lag_engine** — its comment says so — **and it was never applied to the
call sites** (P171/P226, again).

Reported as LATENT, not live: the loop logs only on error, so its silence is
not evidence either way, and a task suspended on `asyncio.sleep` is generally
kept alive by that future's callback chain. The fix is free and the failure
mode is documented, which is enough.
"""

import ast
import inspect
import json
import pathlib

import pytest

import main
import market.lead_lag_engine as lle

REPO = pathlib.Path(main.__file__).parent


def _live():
    return json.loads((REPO / "configs" / "live_high_risk.json").read_text(
        encoding="utf-8-sig"))


# ==========================================================================
# PART 1 — the disarm
# ==========================================================================
@pytest.mark.parametrize("flag", [
    "coinbase_ma_filter_enforce",
    "coinbase_whale_filter_enforce",
])
def test_both_entry_filters_are_disarmed(flag):
    """Pinned at the DECIDED value, so a silent RE-ARM fails too — either
    direction is a live-money change (P237/P270)."""
    assert _live().get(flag) is False, (
        f"{flag} is not at its decided value False"
    )


def test_the_disarm_carries_its_reason_in_the_profile():
    """P141: a live flip is a decision, and a bare `false` loses why."""
    d = _live()
    notes = [k for k in d if k.startswith("_p356")]
    assert notes, "no P356 note in the live profile"
    body = " ".join(str(d[k]) for k in notes)
    for must in ("23", "ETH", "REVERT"):
        assert must in body, f"the note does not state {must!r}"


def test_the_ledgers_keep_accruing_so_nothing_is_given_up():
    """The harness records the counterfactual whether or not enforcement is
    on (P340) — that is why disarming costs no evidence. Pinned at the
    driver: the shadow observe() must NOT be gated on the enforce flag."""
    src = inspect.getsource(main.HMATSProductionRunner.run_live)
    i = src.index("_maf_enf = bool(getattr(")
    window = src[i:i + 1200]
    assert "_ma_filter_shadow.observe(" in window
    j = window.index("_ma_filter_shadow.observe(")
    guard = window[:j]
    assert "if _maf_enf" not in guard, (
        "the ledger write is gated on enforcement — disarming would then stop "
        "the evidence, which is the opposite of the intent"
    )


def test_the_decision_function_itself_is_untouched():
    """Disarming is a CONFIG change. The filter logic must still be there and
    still correct, so re-arming is a one-line revert rather than a rebuild."""
    led, act, why = main.sleeve_ma_filter_decision(0, 1, -1.0)
    assert (led, act, why) == (0, "block_entry", "ma_disagrees_entry")


# ==========================================================================
# PART 2 — the async-lifecycle sweep
# ==========================================================================
def test_no_background_task_reference_is_discarded():
    """asyncio keeps only a WEAK reference, so a discarded handle lets a task
    be garbage-collected mid-execution."""
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8",
                                                  errors="replace"))
    discarded = []
    for node in ast.walk(tree):
        for _f, value in ast.iter_fields(node):
            for item in (value if isinstance(value, list) else [value]):
                if not isinstance(item, ast.Expr):
                    continue
                c = item.value
                if not isinstance(c, ast.Call):
                    continue
                fn = c.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(
                    fn, "id", "")
                if name in ("create_task", "ensure_future"):
                    discarded.append(c.lineno)
    assert not discarded, (
        f"create_task/ensure_future results discarded at lines {discarded} — "
        f"asyncio's docs: 'Save a reference to the result of this function, to "
        f"avoid a task disappearing mid-execution'"
    )


def test_the_runner_holds_the_references():
    src = inspect.getsource(main.HMATSProductionRunner)
    assert "self._bg_tasks: set = set()" in src, "no holder is initialised"
    assert src.count("self._bg_tasks.add(asyncio.create_task(") == 6, (
        "expected six dispatch sites (three background loops x paper/live)"
    )


def test_the_dangerous_shape_is_the_one_that_motivated_it():
    """OnChainFeed.start() runs its loop INLINE, so the Task IS the loop —
    unlike the other two, which spawn tracked inner tasks and return. If that
    ever changes, the reasoning in P356 needs re-deriving."""
    import data_mgmt.feeds.onchain_feed as ocf
    src = inspect.getsource(ocf.OnChainFeed.start)
    assert "while self._running" in src and "asyncio.sleep" in src, (
        "OnChainFeed.start() no longer runs its poll loop inline"
    )
    lle_src = inspect.getsource(lle.LeadLagAlphaEngine.start)
    assert "self._binance_task = asyncio.create_task" in lle_src, (
        "lead_lag no longer stores its inner tasks — P37's fix, which is why "
        "its outer task is safe to complete"
    )


@pytest.mark.parametrize("cls", [lle.BinanceTakerMonitor, lle.LeadLagAlphaEngine])
def test_no_shared_mutable_default_asset_list(cls):
    a, b = cls(), cls()
    assert a.assets == ["BTC", "ETH", "SOL"]
    assert a.assets is not b.assets, (
        "both instances share ONE list object — a mutable default is created "
        "once at def time, and this one is stored on self"
    )


def test_an_explicit_asset_list_is_still_honoured_and_copied():
    given = ["BTC"]
    m = lle.BinanceTakerMonitor(given)
    assert m.assets == ["BTC"]
    assert m.assets is not given, (
        "the caller's list is stored by reference; a later mutation by the "
        "caller would change the monitor's universe underneath it"
    )


def test_no_mutable_default_argument_anywhere_on_the_live_path():
    """The class, not the two instances (P171/P226)."""
    bad = []
    for rel in ("market/lead_lag_engine.py", "main.py",
                "data_mgmt/market_data_pipeline.py",
                "exchange/coinbase_sleeve.py", "core/execution_service.py"):
        p = REPO / rel
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = list(fn.args.defaults) + [
                d for d in fn.args.kw_defaults if d]
            for d in defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    bad.append(f"{rel}:{fn.lineno} {fn.name}")
    assert not bad, f"mutable default arguments: {bad}"


def test_no_blocking_io_inside_an_async_def_in_the_engine():
    """The sweep's clean result, pinned so it stays clean. Scripts are
    excluded: an operator-run CLI blocking its own loop harms nothing."""
    BLOCKING = {"sleep", "urlopen", "check_output"}
    bad = []
    for rel in ("main.py", "data_mgmt/market_data_pipeline.py",
                "exchange/coinbase_sleeve.py", "exchange/coinbase_adapter.py",
                "core/execution_service.py", "market/lead_lag_engine.py"):
        p = REPO / rel
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                if not isinstance(f, ast.Attribute) or f.attr not in BLOCKING:
                    continue
                mod = getattr(f.value, "id", "")
                if mod in ("time", "subprocess") or f.attr == "urlopen":
                    bad.append(f"{rel}:{n.lineno} {mod}.{f.attr} in {fn.name}")
    assert not bad, f"blocking I/O on an async path: {bad}"

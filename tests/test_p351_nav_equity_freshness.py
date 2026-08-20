"""[P351] The LIVE drawdown halt could never fire — the snapshot always read a
4H-stale equity, because the only per-tick refresh runs later in the same tick.

`_update_drawdown_snapshot` calls `account_sync.get_equity()`, which
FAIL-CLOSES once the cached state is older than
`core.account_sync.MAX_EQUITY_AGE_SECONDS` (120s). The ONLY per-tick
`refresh()` sits at the top of `_process_4h_tick_inner`, i.e. inside the
per-asset loop that `run_live` enters AFTER taking this snapshot. So the
freshest equity available at snapshot time is the PREVIOUS tick's — a full 4H
bar, ~120x the freshness bound — and the read raised on every tick.

Measured live 2026-08-20, 4 ticks of 4:

    [NAV] equity fetch failed (FAIL-CLOSED: Account equity unavailable.
          Status=VALID, Age=14326.6s, Equity=0.398431766369)
    [NAV] equity is the notional fallback ($20,812.98), not an exchange reading

Note `Status=VALID`: the state was not broken, it was old. Consequences, all
silent: `_current_drawdown_pct` was never written, so the LIVE 25% halt
(P201), the regime-leverage de-risking ladder and the DRL observation's
`drawdown` dim read their getattr default 0.0 forever; `_peak_equity` was
never set (the persisted value on the server is 0.0, which is the
fingerprint); and `[NAV-LIVE]` reported `initial_capital + sleeve` as though
it were an exchange reading.

P163 fixed the missing WRITER of `_current_drawdown_pct`. The writer then
became unreachable for a different reason — and the snapshot's own comment
predicted it would self-heal "until account_sync has refreshed", which it
cannot, because the refresh is always downstream of the read.
"""

import asyncio
import inspect
import re
import time

import pytest

import main
from main import HMATSProductionRunner
from core.account_sync import (
    AccountSyncManager,
    AccountState,
    EquityStatus,
    MAX_EQUITY_AGE_SECONDS,
)
from tests._guard_pins import assert_text_pin

# One 4H bar. The trading loop sleeps to the next 4H boundary, so "one tick
# ago" means at least this long ago.
FOUR_HOURS = 4 * 60 * 60.0


# --------------------------------------------------------------------------
# the premise
# --------------------------------------------------------------------------
def test_one_tick_old_equity_can_never_satisfy_the_freshness_bound():
    """The whole finding rests on this inequality; pin it rather than assume.

    If MAX_EQUITY_AGE_SECONDS is ever raised above a 4H bar the defect
    dissolves — and so does the reason for the pre-snapshot refresh. This
    test is the thing that says so.
    """
    assert MAX_EQUITY_AGE_SECONDS < FOUR_HOURS, (
        f"MAX_EQUITY_AGE_SECONDS={MAX_EQUITY_AGE_SECONDS}s is no longer below "
        f"one 4H tick ({FOUR_HOURS}s). The P351 analysis assumed a refresh one "
        f"tick earlier is always stale; re-derive it before trusting the fix."
    )


def _sync_with_age(age_seconds: float, equity: float = 10_000.0):
    """A REAL AccountSyncManager whose state is `age_seconds` old and VALID."""
    sync = AccountSyncManager(exchange_client=None, exchange_name="kraken",
                              dry_run=True)
    sync._state = AccountState(
        equity=equity,
        timestamp=time.time() - age_seconds,
        exchange="kraken",
        status=EquityStatus.VALID,
    )
    return sync


def test_a_one_tick_old_reading_fail_closes_at_the_real_class():
    """Measured at the producer, not at a fake: VALID status, stale age."""
    sync = _sync_with_age(FOUR_HOURS - 60.0)
    with pytest.raises(RuntimeError) as err:
        sync.get_equity()
    msg = str(err.value)
    assert "Status=VALID" in msg, (
        "the live message said Status=VALID while refusing — that is what "
        "made this read as a venue fault rather than as staleness"
    )
    assert "Age=" in msg


def test_a_freshly_refreshed_reading_is_served():
    """The other direction: the fix is only worth having if fresh works."""
    sync = _sync_with_age(1.0, equity=1234.5)
    assert sync.get_equity() == pytest.approx(1234.5)


# --------------------------------------------------------------------------
# the consequence, at the snapshot
# --------------------------------------------------------------------------
class _Cfg:
    initial_capital = 10_000.0
    coinbase_routing_enabled = False
    mode = None


class _Pipeline:
    current_drawdown_pct = None


class _Runner:
    """Carries only what the snapshot and the refresh helper touch."""

    def __init__(self, sync):
        self.config = _Cfg()
        self.account_sync = sync
        self._market_pipeline = _Pipeline()


def _snapshot(runner):
    return HMATSProductionRunner._update_drawdown_snapshot(runner)


def test_a_stale_read_leaves_the_whole_ladder_disarmed():
    """The bug, reproduced: nothing is written, so every consumer reads 0.0."""
    runner = _Runner(_sync_with_age(FOUR_HOURS))
    equity, dd = _snapshot(runner)

    assert not hasattr(runner, "_current_drawdown_pct"), (
        "a stale read must not be mistaken for a measured 0% drawdown — but "
        "the consumers' getattr default makes those two indistinguishable, "
        "which is why this went unseen"
    )
    assert not hasattr(runner, "_peak_equity")
    assert equity == pytest.approx(10_000.0), "the fabricated notional fallback"
    assert dd == pytest.approx(0.0)


def test_a_fresh_read_arms_it():
    runner = _Runner(_sync_with_age(1.0, equity=8_000.0))
    _snapshot(runner)
    assert runner._current_drawdown_pct == pytest.approx(0.0)
    assert runner._peak_equity == pytest.approx(8_000.0)

    runner.account_sync = _sync_with_age(1.0, equity=6_000.0)
    _, dd_after = _snapshot(runner)
    assert dd_after == pytest.approx(0.25), "the 25% halt is now reachable"


# --------------------------------------------------------------------------
# the helper's fail direction
# --------------------------------------------------------------------------
class _RaisingSync:
    async def refresh(self):
        raise RuntimeError("kraken 502")


class _OkSync:
    def __init__(self):
        self.calls = 0

    async def refresh(self):
        self.calls += 1
        return True, ""


def _refresh(runner):
    return asyncio.run(HMATSProductionRunner._refresh_equity_for_nav(runner))


def test_the_helper_refreshes_and_reports_it():
    sync = _OkSync()
    runner = _Runner(sync)
    assert _refresh(runner) is True
    assert sync.calls == 1


def test_a_failing_refresh_never_raises_into_the_trading_loop():
    """Conservative direction: the snapshot then behaves exactly as before."""
    runner = _Runner(_RaisingSync())
    assert _refresh(runner) is False


def test_no_account_sync_is_not_an_error():
    runner = _Runner(None)
    assert _refresh(runner) is False


# --------------------------------------------------------------------------
# wiring — the half that was actually missing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("loop_name", ["run_live", "run_paper"])
def test_both_loops_refresh_before_they_snapshot(loop_name):
    src = inspect.getsource(getattr(HMATSProductionRunner, loop_name))
    refresh_at = src.find("await self._refresh_equity_for_nav()")
    snap_at = src.find("self._update_drawdown_snapshot()")
    assert refresh_at != -1, (
        f"{loop_name} takes a drawdown snapshot without refreshing the equity "
        f"first — the reading is a full 4H bar stale and FAIL-CLOSES, so the "
        f"halt is inert on every tick (P351)"
    )
    assert snap_at != -1
    assert refresh_at < snap_at, (
        f"{loop_name} refreshes AFTER the snapshot, which is the original bug "
        f"one line down instead of one loop down"
    )


def test_live_still_snapshots_before_it_decides():
    """P163's ordering must survive P351's insertion."""
    src = inspect.getsource(HMATSProductionRunner.run_live)
    assert src.index("self._update_drawdown_snapshot()") < src.index(
        "self.process_4h_tick("), (
        "run_live must refresh drawdown before it decides trades, otherwise "
        "the de-risking ladder gates on the previous tick's state"
    )


def test_the_refresh_sits_immediately_above_the_live_snapshot():
    """A guarded or relocated refresh would re-open the gap.

    Anchored, because the identical call appears in run_paper too — a bare
    substring pin could not tell the two sites apart (P349).
    """
    src = inspect.getsource(HMATSProductionRunner.run_live)
    assert_text_pin(
        src,
        "await self._refresh_equity_for_nav()\n"
        "                    _live_equity, _live_dd = "
        "self._update_drawdown_snapshot()",
        why="the live refresh must sit immediately above the snapshot it feeds",
    )


def test_single_writer_of_current_drawdown_pct_survives():
    """Carried from P163: two writers is how paper and live diverged."""
    src = inspect.getsource(main)
    writers = re.findall(r"self\._current_drawdown_pct\s*=", src)
    assert len(writers) == 1, (
        f"expected exactly one writer (_update_drawdown_snapshot), found "
        f"{len(writers)}"
    )

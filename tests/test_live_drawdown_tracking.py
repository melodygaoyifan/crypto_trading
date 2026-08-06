"""[P163] The drawdown breakers must actually be armed in LIVE.

`_current_drawdown_pct` had exactly one writer, and it sat inline inside
`run_paper`. `run_live` never assigned it. Every consumer reads it defensively:

    main.py:11797  _rl_dd = getattr(self, '_current_drawdown_pct', 0.0)   # regime leverage
    main.py:19448  dd     = getattr(self, '_current_drawdown_pct', 0.0)   # DRL observation

so in the one mode that risks real money, the de-risking ladder, the drawdown
halt, and the DRL's own state vector all read a permanent 0.0 — the system
believed it was at its all-time high no matter how far equity had fallen.

The failure is invisible by construction: a defaulted `getattr` makes "never
written" and "zero drawdown" the same reading. Nothing could observe the gap,
which is why it survived. These tests pin both halves of the fix — that the
computation is correct, and that LIVE actually calls it.
"""

import inspect
import re

import pytest

import main
from main import HMATSProductionRunner


# --- a minimal stand-in; instantiating the real runner boots the whole system --
class _Cfg:
    initial_capital = 10_000.0


class _Sync:
    def __init__(self, equity):
        self._equity = equity

    def get_equity(self):
        if isinstance(self._equity, Exception):
            raise self._equity
        return self._equity


class _Pipeline:
    current_drawdown_pct = None


class _Runner:
    """Carries only what _update_drawdown_snapshot touches."""

    def __init__(self, equity=10_000.0, pipeline=None):
        self.config = _Cfg()
        self.account_sync = _Sync(equity)
        self._market_pipeline = _Pipeline() if pipeline is None else pipeline


def _snapshot(runner):
    return HMATSProductionRunner._update_drawdown_snapshot(runner)


# --- computation -----------------------------------------------------------
def test_no_drawdown_at_peak():
    runner = _Runner(equity=10_000.0)
    equity, dd = _snapshot(runner)
    assert equity == pytest.approx(10_000.0)
    assert dd == pytest.approx(0.0)


def test_drawdown_is_positive_fraction_below_peak():
    runner = _Runner(equity=10_000.0)
    _snapshot(runner)                      # establish peak
    runner.account_sync = _Sync(7_500.0)   # -25%
    equity, dd = _snapshot(runner)
    assert equity == pytest.approx(7_500.0)
    assert dd == pytest.approx(0.25)
    assert runner._current_drawdown_pct == pytest.approx(0.25)


def test_peak_ratchets_and_does_not_fall_back():
    """A recovery must not erase the peak — otherwise drawdown resets to 0."""
    runner = _Runner(equity=10_000.0)
    _snapshot(runner)
    runner.account_sync = _Sync(6_000.0)
    _snapshot(runner)
    runner.account_sync = _Sync(8_000.0)   # partial recovery
    _, dd = _snapshot(runner)
    assert runner._peak_equity == pytest.approx(10_000.0)
    assert dd == pytest.approx(0.20)


def test_pipeline_is_kept_in_sync():
    runner = _Runner(equity=10_000.0)
    _snapshot(runner)
    runner.account_sync = _Sync(9_000.0)
    _snapshot(runner)
    assert runner._market_pipeline.current_drawdown_pct == pytest.approx(0.10)


def test_missing_pipeline_does_not_raise():
    """LIVE calls this before some components are guaranteed present."""
    runner = _Runner(equity=10_000.0, pipeline=None)
    runner._market_pipeline = None
    _, dd = _snapshot(runner)
    assert dd == pytest.approx(0.0)


def test_equity_fetch_failure_holds_last_known_drawdown():
    """A bad read must not fabricate a 0% drawdown and silently disarm risk.

    Falling back to `initial_capital` would report "no drawdown" whenever the
    exchange read fails — de-risking would switch off exactly when the venue is
    unhealthy. A stale drawdown is conservative; a fabricated zero is not.
    """
    runner = _Runner(equity=10_000.0)
    _snapshot(runner)
    runner.account_sync = _Sync(6_000.0)
    _, dd_before = _snapshot(runner)
    assert dd_before == pytest.approx(0.40)

    runner.account_sync = _Sync(RuntimeError("exchange timeout"))
    _, dd_after = _snapshot(runner)

    assert dd_after == pytest.approx(0.40), "risk ladder must stay armed"
    assert runner._current_drawdown_pct == pytest.approx(0.40)
    assert runner._peak_equity == pytest.approx(10_000.0)


def test_equity_fetch_failure_on_first_ever_call_is_survivable():
    """No prior value to hold: must not crash, and must not invent a peak."""
    runner = _Runner(equity=RuntimeError("cold start timeout"))
    equity, dd = _snapshot(runner)
    assert dd == pytest.approx(0.0)
    assert equity == pytest.approx(10_000.0)


def test_no_account_sync_falls_back_to_initial_capital():
    runner = _Runner()
    runner.account_sync = None
    equity, dd = _snapshot(runner)
    assert equity == pytest.approx(10_000.0)
    assert dd == pytest.approx(0.0)


# --- wiring: the half that was actually missing -----------------------------
def _source_of(fn):
    return inspect.getsource(fn)


@pytest.mark.parametrize("method_name", ["run_live", "run_paper"])
def test_both_trading_loops_write_the_drawdown(method_name):
    """The regression itself: LIVE had no writer for ~7 weeks of real money."""
    src = _source_of(getattr(HMATSProductionRunner, method_name))
    assert "_update_drawdown_snapshot()" in src, (
        f"{method_name} does not refresh _current_drawdown_pct — every "
        f"drawdown-scaled risk control in that mode reads a stale 0.0"
    )


def test_live_updates_drawdown_before_processing_assets():
    """Order matters: a post-trade update gates next tick, not this one."""
    src = _source_of(HMATSProductionRunner.run_live)
    dd_at = src.index("_update_drawdown_snapshot()")
    tick_at = src.index("self.process_4h_tick(")
    assert dd_at < tick_at, (
        "run_live must refresh drawdown before it decides trades, otherwise "
        "the de-risking ladder gates on the previous tick's state"
    )


def test_single_writer_of_current_drawdown_pct():
    """Guard against a second inline writer reappearing and drifting."""
    src = inspect.getsource(main)
    writers = re.findall(r"self\._current_drawdown_pct\s*=", src)
    assert len(writers) == 1, (
        f"expected exactly one writer (_update_drawdown_snapshot), found "
        f"{len(writers)} — duplicated writers are how paper and live diverged"
    )

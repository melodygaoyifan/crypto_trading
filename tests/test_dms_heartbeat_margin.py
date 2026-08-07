"""[P195] The DMS heartbeat must hold its rate when refreshes are slow.

`DeadManSwitchMonitor.run()` used to end every cycle with
`self._stop_event.wait(self._interval_sec)` unconditionally, i.e. FIXED DELAY.
The period was therefore `refresh_duration + interval`, which charges the REST
retry budget on top of the heartbeat instead of inside it:

    healthy   ~0.3s refresh + 24s  =  ~24s   -> 2.5x  margin vs the 60s timer
    failing   ~31s  refresh + 24s  =  ~55s   -> 1.09x margin (5 seconds)

31s is 3 attempts x 10s read timeout + 2 x 0.5s backoff. The 24s interval is
deliberately 40% of the timeout (main.py: `min(30, max(5, int(t*0.4)))`) to buy a
2.5x margin — and the old loop gave that margin away precisely when the API was
unhealthy and the timer mattered most. Confirmed against the 2026-08-06 Kraken
outage, where the failures are spaced exactly 55s apart.

The danger is not the full outage (nothing survives 13 minutes at a 60s timer);
it is the MARGINAL case, where Kraken is slow but answers on attempt 3 and the
refresh "succeeds" after the server timer already lapsed, silently.

These tests drive the real `run()` loop with a fake clock/event, so they measure
the scheduler rather than a copy of its logic.
"""

import threading

import pytest

from execution import dead_man_switch as dms_mod
from execution.dead_man_switch import (
    DeadManSwitchMonitor,
    MIN_HEARTBEAT_GAP_SEC,
)


class _FakeClock:
    """Monotonic clock advanced only by the code under test."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, secs):
        self.now += secs


class _RecordingEvent:
    """Stands in for threading.Event: records each wait() and advances the clock.

    Stops the loop after `max_cycles` so run() terminates deterministically.
    """

    def __init__(self, clock, max_cycles):
        self.clock = clock
        self.waits = []
        self._cycles = 0
        self._max = max_cycles
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.clock.advance(timeout or 0.0)
        self._cycles += 1
        if self._cycles >= self._max:
            self._set = True
        return self._set


class _SlowSwitch:
    """A DeadManSwitch whose refresh burns `cost` seconds of the fake clock."""

    def __init__(self, clock, cost, timeout_sec=60, succeeds=True):
        self.clock = clock
        self.cost = cost
        self._timeout_sec = timeout_sec
        self.succeeds = succeeds
        self.refresh_starts = []

    def refresh(self):
        self.refresh_starts.append(self.clock.now)
        self.clock.advance(self.cost)
        return self.succeeds


def _run(monkeypatch, cost, interval=24, timeout=60, cycles=4, succeeds=True):
    clock = _FakeClock()
    monkeypatch.setattr(dms_mod.time, "monotonic", clock)
    sw = _SlowSwitch(clock, cost, timeout_sec=timeout, succeeds=succeeds)
    mon = DeadManSwitchMonitor(dms=sw, interval_sec=interval, max_failures=3)
    mon._stop_event = _RecordingEvent(clock, max_cycles=cycles)
    mon.run()
    return sw, mon._stop_event


def test_cheap_refresh_waits_essentially_the_full_interval(monkeypatch):
    """Healthy case must be unchanged: ~24s cadence."""
    _, ev = _run(monkeypatch, cost=0.3, interval=24)
    assert all(w == pytest.approx(23.7, abs=0.01) for w in ev.waits), ev.waits


def test_slow_refresh_does_not_add_its_cost_on_top_of_the_interval(monkeypatch):
    """The P195 defect: period must stay ~interval, not interval + elapsed."""
    sw, _ = _run(monkeypatch, cost=31.0, interval=24, cycles=4)
    gaps = [b - a for a, b in zip(sw.refresh_starts, sw.refresh_starts[1:])]
    assert gaps, "loop did not run enough cycles to measure a gap"
    for g in gaps:
        assert g < 40.0, (
            f"heartbeat period was {g:.1f}s. The refresh cost is being added on "
            f"top of the interval (fixed-delay), which is exactly the P195 bug: "
            f"a 31s refresh at a 24s interval yields the observed 55s spacing."
        )


def test_gap_between_refreshes_never_reaches_the_server_timeout(monkeypatch):
    """The invariant that actually matters, asserted directly.

    If successive refreshes can be spaced >= timeout_sec, the Kraken timer can
    expire between two 'successful' heartbeats and every order is cancelled with
    nothing in the log saying the timer lapsed.
    """
    for cost in (0.3, 10.0, 21.0, 31.0, 45.0):
        sw, _ = _run(monkeypatch, cost=cost, interval=24, timeout=60, cycles=5)
        gaps = [b - a for a, b in zip(sw.refresh_starts, sw.refresh_starts[1:])]
        worst = max(gaps) if gaps else 0.0
        assert worst < 60.0, (
            f"with a {cost}s refresh the worst gap between refreshes was "
            f"{worst:.1f}s, at or beyond the 60s server timer"
        )


def test_a_refresh_longer_than_the_interval_still_leaves_a_floor(monkeypatch):
    """Fixed-rate must not become a tight loop against a struggling API."""
    _, ev = _run(monkeypatch, cost=45.0, interval=24)
    assert all(w == pytest.approx(MIN_HEARTBEAT_GAP_SEC) for w in ev.waits), ev.waits
    assert MIN_HEARTBEAT_GAP_SEC > 0


def test_the_floor_never_exceeds_the_configured_interval(monkeypatch):
    """A floor above the cadence would slow a HEALTHY monitor below its config.

    It would also starve callers using a sub-second interval — which is exactly
    what tests/test_dead_man_switch_monitor.py does (interval_sec=0.05), and how
    a flat 5s floor was caught during P195.
    """
    _, ev = _run(monkeypatch, cost=1.0, interval=0.05, cycles=3)
    assert all(w <= 0.05 + 1e-9 for w in ev.waits), (
        f"waits {ev.waits} exceed the configured 0.05s interval — the floor is "
        f"not capped at the interval"
    )


def test_failing_refresh_is_scheduled_at_the_same_rate_as_a_succeeding_one(monkeypatch):
    """A failure must not slow the retry cadence — that is when it matters most."""
    sw_ok, _ = _run(monkeypatch, cost=31.0, cycles=4, succeeds=True)
    sw_bad, _ = _run(monkeypatch, cost=31.0, cycles=4, succeeds=False)
    gaps_ok = [b - a for a, b in zip(sw_ok.refresh_starts, sw_ok.refresh_starts[1:])]
    gaps_bad = [b - a for a, b in zip(sw_bad.refresh_starts, sw_bad.refresh_starts[1:])]
    assert gaps_ok == gaps_bad, (gaps_ok, gaps_bad)


def test_the_monitor_still_escalates_after_three_failures(monkeypatch, caplog):
    """Guard the behaviour the scheduling change must not disturb."""
    import logging
    with caplog.at_level(logging.CRITICAL):
        _run(monkeypatch, cost=1.0, cycles=4, succeeds=False)
    assert any("consecutive" in r.message for r in caplog.records
               if r.levelno >= logging.CRITICAL), [r.message for r in caplog.records]


# ---------------------------------------------------------------------------
# [P195] the removed LIVE escalation
# ---------------------------------------------------------------------------

class TestTheDeadEscalationStaysRemoved:
    """`main.py` carried a LIVE "3 consecutive DMS failures -> emergency flatten"
    block that was dead three independent ways and could never have run:

      1. refresh() catches internally and returns False rather than raising, and
         the return value was discarded — so the `except` never fired and the
         counter stayed 0 forever;
      2. it called an underscore-prefixed emergency-flatten method that does not
         exist on the class (the real one is `trigger_emergency_flatten`);
      3. that AttributeError was swallowed by its own `except`.

    It was deleted rather than repaired, deliberately: the 2026-08-06 incident was
    a 13-minute Kraken PRIVATE-endpoint outage with the public API healthy, and
    that must not liquidate the book. Fail-closing there converts an API problem
    into a forced exit at whatever price the outage leaves — the failure shape
    P141 exists to prevent.
    """

    @staticmethod
    def _main_src():
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        return (root / "main.py").read_text(encoding="utf-8", errors="replace")

    def test_no_call_to_a_nonexistent_emergency_flatten(self):
        src = self._main_src()
        assert "self._emergency_flatten(" not in src, (
            "main.py calls self._emergency_flatten(), which is not defined on the "
            "class — it would AttributeError. The real method is "
            "trigger_emergency_flatten. If flatten-on-DMS-failure is being "
            "reintroduced deliberately, wire the counter and the return value too, "
            "and reconsider whether an API outage should liquidate the book."
        )

    def test_the_real_flatten_method_was_not_collaterally_deleted(self):
        assert "def trigger_emergency_flatten(" in self._main_src()

    def test_the_refresh_and_its_alert_are_still_there(self):
        """Removing the escalation must not remove the heartbeat itself."""
        src = self._main_src()
        assert "self.dead_man_switch.refresh()" in src
        assert "Dead-man switch refresh failed" in src

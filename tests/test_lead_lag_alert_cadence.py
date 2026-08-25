"""[P408] Connect-failure alerts are once-per-streak, not once-per-attempt.

The reconnect loop attempts every ~30s. The old `>= threshold:
logger.error` made a sustained venue outage page the operator EVERY 30
SECONDS for its whole life (2026-08-25: 5 ERRORs in 3 minutes for a 14-min
Deribit WS blip that self-healed). P329b's rule: ERROR exactly once when a
streak crosses the sustained threshold, hourly re-alert while it persists,
WARNING otherwise, and a one-line recovery announcement (P265f).
"""
from __future__ import annotations

import inspect

from market.lead_lag_engine import (
    SUSTAINED_REALERT_SEC,
    connect_failure_severity,
)
import market.lead_lag_engine as lle


class TestSeverityTruthTable:
    def test_pre_threshold_is_warning(self):
        for n in (1, 3, 7):
            sev, ts = connect_failure_severity(n, 8, now=1000.0,
                                               last_error_ts=0.0)
            assert sev == "warning" and ts == 0.0, n

    def test_crossing_the_threshold_errors_exactly_once(self):
        sev, ts = connect_failure_severity(8, 8, now=1000.0, last_error_ts=0.0)
        assert sev == "error" and ts == 1000.0
        # the very next attempt (30s later) must NOT error again
        sev2, ts2 = connect_failure_severity(9, 8, now=1030.0,
                                             last_error_ts=ts)
        assert sev2 == "warning" and ts2 == 1000.0

    def test_a_30s_cadence_outage_stays_quiet_between_hourly_realerts(self):
        ts = 0.0
        errors = 0
        for i in range(8, 8 + 250):            # ~2h05m of 30s attempts
            now = 1000.0 + (i - 8) * 30.0
            sev, ts = connect_failure_severity(i, 8, now, ts)
            errors += (sev == "error")
        # crossing (t=1000) + re-alerts at t>=4600 (attempt 128) and
        # t>=8200 (attempt 248) — never one per attempt (the old shape
        # produced 250 ERRORs over this window)
        assert errors == 3, errors

    def test_hourly_realert_fires_after_the_interval(self):
        sev, ts = connect_failure_severity(8, 8, 1000.0, 0.0)
        assert sev == "error"
        sev2, ts2 = connect_failure_severity(100, 8,
                                             1000.0 + SUSTAINED_REALERT_SEC,
                                             ts)
        assert sev2 == "error" and ts2 == 1000.0 + SUSTAINED_REALERT_SEC

    def test_binance_threshold_is_lower_than_deribits(self):
        # taker flow is a live input; DVOL is advisory (P303)
        src = inspect.getsource(lle)
        assert "connect_failure_severity(\n                self._connect_failures, 4," in src
        assert "connect_failure_severity(\n                self._connect_failures, 8," in src


class TestWiring:
    def test_both_monitors_use_the_helper_and_the_bare_shape_is_gone(self):
        src = inspect.getsource(lle)
        assert src.count("connect_failure_severity(") >= 3  # def + 2 call sites
        assert "if self._connect_failures >= 8:\n                self.logger.error" not in src
        assert "if self._connect_failures >= 4:\n                self.logger.error" not in src, (
            "the per-attempt ERROR shape paged the operator every 30s for the "
            "life of an outage — P408")

    def test_success_resets_the_sustained_error_clock(self):
        src = inspect.getsource(lle)
        assert src.count("self._sustained_error_ts = 0.0") == 2, (
            "without the reset, the NEXT streak inherits the previous one's "
            "clock and its crossing ERROR is silently swallowed")

    def test_recovery_of_a_sustained_streak_is_announced(self):
        src = inspect.getsource(lle)
        assert "Binance WS RECOVERED after" in src
        assert "Deribit WS RECOVERED after" in src
        # short blips stay quiet: the announce is gated on the threshold
        assert 'getattr(self, "_connect_failures", 0) >= 4' in src
        assert 'getattr(self, "_connect_failures", 0) >= 8' in src

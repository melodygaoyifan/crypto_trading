"""[P160] A broken dashboard export must not fail silently.

`_export_dashboard_state` is the ONLY writer of dashboard_state.json — the file
the API serves and an operator reads to decide whether the engine is alive. Its
blanket `except Exception` logged at DEBUG, which production log levels drop.

So a single bad attribute access froze the dashboard at its last good values
with nothing said anywhere: identical to the P155/P156 shape of state that
reads as live but stopped updating. That is exactly how it went unnoticed that
the two tests in test_dashboard_state_incremental_export.py were failing on a
missing file — the cause (`SimpleNamespace` has no `mode`) was swallowed.
"""

import logging
from types import SimpleNamespace

import pytest

from main import HMATSProductionRunner, RunMode


def _runner(tmp_path, *, broken=False):
    r = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {"initial_capital": 10_000.0, "risk_profile": "HIGH_RISK"}
    if not broken:
        cfg["mode"] = RunMode.PAPER  # omitting this is the real failure mode
    r.config = SimpleNamespace(**cfg)
    r.account_sync = None
    r._paper_positions = {}
    r._position_entry_times = {}
    r._dashboard_asset_snapshot = {}
    r._dashboard_asset_runtime = {}
    r._DASHBOARD_STATE_FILE = tmp_path / "dashboard_state.json"
    r._STEP15_STATUS_FILE = tmp_path / "step15_status.json"
    r._PAPER_RUN_PID_FILE = tmp_path / "paper_run.pid"
    r._KILL_SWITCH_TEST_FILE = tmp_path / "kill_switch_test.json"
    r._drl_models_ready = 0
    r._drl_ensembles = {}
    r._drl_bootstrap_applied = False
    return r


def test_export_failure_is_logged_at_error_not_debug(tmp_path, caplog):
    r = _runner(tmp_path, broken=True)
    with caplog.at_level(logging.ERROR):
        r._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    errs = [x for x in caplog.records if "State export FAILED" in x.message]
    assert errs, "a silently-frozen dashboard is indistinguishable from a live one"
    assert errs[0].levelno >= logging.ERROR
    assert "STALE" in errs[0].message, "must say what the consequence is"
    assert "AttributeError" in errs[0].message, "must name the exception type"


def test_export_failure_never_propagates(tmp_path):
    """Diagnostics must not kill the tick — the swallow itself was correct."""
    r = _runner(tmp_path, broken=True)
    r._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})  # must not raise


def test_repeated_failures_are_rate_limited(tmp_path, caplog):
    """Loud enough to notice, quiet enough not to bury a 4H log."""
    r = _runner(tmp_path, broken=True)
    with caplog.at_level(logging.ERROR):
        for _ in range(120):
            r._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    errs = [x for x in caplog.records if "State export FAILED" in x.message]
    # 1st, 10th, 100th
    assert len(errs) == 3, f"expected 1/10/100 cadence, got {len(errs)}"
    assert "consecutive failures=100" in errs[-1].message


def test_recovery_is_announced_and_counter_resets(tmp_path, caplog):
    r = _runner(tmp_path, broken=True)
    r._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})
    assert r._dashboard_export_fail_count == 1

    r.config.mode = RunMode.PAPER  # repair
    with caplog.at_level(logging.WARNING):
        r._export_dashboard_state(1, 2, {"SOL": {"price": 100.0}})

    assert r._dashboard_export_fail_count == 0
    assert any("recovered after 1" in x.message for x in caplog.records)
    assert r._DASHBOARD_STATE_FILE.exists()


def test_healthy_export_logs_nothing_and_writes_the_file(tmp_path, caplog):
    r = _runner(tmp_path)
    with caplog.at_level(logging.WARNING):
        r._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    assert r._DASHBOARD_STATE_FILE.exists()
    assert not [x for x in caplog.records if "DASHBOARD" in x.message], (
        "the happy path must stay silent"
    )

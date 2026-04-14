"""Tests for core/health_validator.py — startup + per-tick health checks."""
import sys, os, pytest, json
from unittest.mock import MagicMock, patch
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.health_validator import (
    StartupHealthValidator, PerTickInvariantChecker,
    HealthCheck, HealthReport,
)


# =============================================================================
# HEALTH CHECK DATACLASS
# =============================================================================

class TestHealthCheck:
    def test_str_format(self):
        c = HealthCheck("S1", "DRL state", "PASS", "file=ACTIVE, runtime=ACTIVE")
        assert "[HEALTH_S1]" in str(c)
        assert "PASS" in str(c)

    def test_report_counts(self):
        r = HealthReport(checks=[
            HealthCheck("S1", "a", "PASS"),
            HealthCheck("S2", "b", "WARN"),
            HealthCheck("S3", "c", "CRITICAL"),
            HealthCheck("S4", "d", "PASS"),
        ])
        assert r.passed == 2
        assert r.warnings == 1
        assert r.critical == 1
        assert r.healthy is False

    def test_healthy_when_no_critical(self):
        r = HealthReport(checks=[
            HealthCheck("S1", "a", "PASS"),
            HealthCheck("S2", "b", "WARN"),
        ])
        assert r.healthy is True


# =============================================================================
# STARTUP HEALTH VALIDATOR
# =============================================================================

class TestStartupS1DRLState:
    def test_pass_when_consistent(self, tmp_path):
        runner = MagicMock()
        runner._drl_authority_level = "ACTIVE"
        state_file = Path("data/drl_promotion_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            if state.get("authority_level") == "ACTIVE":
                v = StartupHealthValidator()
                c = v._s1_drl_state_consistency(runner)
                assert c.status == "PASS"

    def test_critical_when_mismatch(self):
        runner = MagicMock()
        runner._drl_authority_level = "ACTIVE"
        v = StartupHealthValidator()
        # Monkey-patch to simulate mismatch
        with patch("builtins.open", side_effect=lambda *a, **k: __import__("io").StringIO(
            '{"authority_level": "EXIT_ONLY"}'
        )):
            with patch("pathlib.Path.exists", return_value=True):
                c = v._s1_drl_state_consistency(runner)
                assert c.status == "CRITICAL"


class TestStartupS2DRLModels:
    def test_critical_when_active_no_models(self):
        runner = MagicMock()
        runner._drl_authority_level = "ACTIVE"
        runner._drl_models_ready = 0
        v = StartupHealthValidator()
        c = v._s2_drl_models_loaded(runner)
        assert c.status == "CRITICAL"

    def test_pass_when_models_loaded(self):
        runner = MagicMock()
        runner._drl_authority_level = "ACTIVE"
        runner._drl_models_ready = 3
        v = StartupHealthValidator()
        c = v._s2_drl_models_loaded(runner)
        assert c.status == "PASS"


class TestStartupS3FastRisk:
    def test_pass_when_empty(self):
        runner = MagicMock()
        runner.fast_risk_tick._last_4h_prices = {}
        v = StartupHealthValidator()
        c = v._s3_fast_risk_no_stale_anchor(runner)
        assert c.status == "PASS"

    def test_warn_when_populated(self):
        runner = MagicMock()
        runner.fast_risk_tick._last_4h_prices = {"BTC": 70000.0}
        v = StartupHealthValidator()
        c = v._s3_fast_risk_no_stale_anchor(runner)
        assert c.status == "WARN"


class TestStartupS5AlphaGate:
    def test_pass_when_reasonable(self):
        runner = MagicMock()
        runner._min_alpha_bps = 5.0
        runner._min_alpha_bps_short = 3.0
        v = StartupHealthValidator()
        c = v._s5_alpha_gate_threshold(runner)
        assert c.status == "PASS"

    def test_warn_when_high(self):
        runner = MagicMock()
        runner._min_alpha_bps = 20.0
        runner._min_alpha_bps_short = 15.0
        v = StartupHealthValidator()
        c = v._s5_alpha_gate_threshold(runner)
        assert c.status == "WARN"


class TestStartupS12SmartBeta:
    def test_critical_when_direction_override(self):
        runner = MagicMock()
        runner._smart_beta.config.allow_direction_override = True
        runner._smart_beta.config.allow_veto = False
        runner._smart_beta.config.enabled = True
        runner._smart_beta.config.mode = "bounded_influence"
        v = StartupHealthValidator()
        c = v._s12_smart_beta_config(runner)
        assert c.status == "CRITICAL"

    def test_pass_when_safe(self):
        runner = MagicMock()
        runner._smart_beta.config.allow_direction_override = False
        runner._smart_beta.config.allow_veto = False
        runner._smart_beta.config.enabled = True
        runner._smart_beta.config.mode = "observe_only"
        v = StartupHealthValidator()
        c = v._s12_smart_beta_config(runner)
        assert c.status == "PASS"


# =============================================================================
# PER-TICK INVARIANT CHECKER
# =============================================================================

class TestLayer3WatchdogChecks:
    def test_w2_drl_state_initial(self):
        """First call should return PASS with initial level."""
        from scripts.live_watchdog import check_drl_state_file
        import scripts.live_watchdog as wd
        wd._last_known_drl_level = None  # reset
        status, detail = check_drl_state_file()
        assert status in ("PASS", "SKIP")

    def test_w2_drl_state_unchanged(self):
        """Same level on next call should PASS."""
        from scripts.live_watchdog import check_drl_state_file
        import scripts.live_watchdog as wd
        wd._last_known_drl_level = None
        check_drl_state_file()  # first call sets level
        status, detail = check_drl_state_file()  # second call
        assert status == "PASS"
        assert "unchanged" in detail

    def test_w3_log_growing(self):
        """Log file should not be frozen."""
        from scripts.live_watchdog import check_log_growing
        status, detail = check_log_growing()
        assert status in ("PASS", "SKIP")

    def test_w4_data_rate(self):
        """Should find LIVE_DATA entries in recent logs."""
        from scripts.live_watchdog import check_data_rate
        status, detail = check_data_rate()
        assert status in ("PASS", "SKIP", "WARN")


class TestTickT1AlphaEstimate:
    def test_pass_when_hold(self):
        checker = PerTickInvariantChecker()
        agent_signals = {"primary_strategy": "hold"}
        market_data = {"signal_edge_bps": 0}
        c = checker._t1_alpha_estimate_nonzero("BTC", None, market_data, agent_signals)
        assert c.status == "PASS"

    def test_warn_when_momentum_zero_edge(self):
        checker = PerTickInvariantChecker()
        agent_signals = {"primary_strategy": "momentum"}
        market_data = {"signal_edge_bps": 0}
        c = checker._t1_alpha_estimate_nonzero("BTC", None, market_data, agent_signals)
        assert c.status == "WARN"


class TestTickT2DRLStability:
    def test_pass_on_first_tick(self):
        checker = PerTickInvariantChecker()
        runner = MagicMock()
        runner._drl_authority_level = "ACTIVE"
        c = checker._t2_drl_authority_stable(runner)
        assert c.status == "PASS"

    def test_critical_on_unexpected_change(self):
        checker = PerTickInvariantChecker()
        runner = MagicMock()
        runner._drl_authority_level = "ACTIVE"
        checker._t2_drl_authority_stable(runner)  # first tick
        runner._drl_authority_level = "EXIT_ONLY"
        c = checker._t2_drl_authority_stable(runner)  # second tick
        assert c.status == "CRITICAL"


class TestTickT5AgentOutputs:
    def test_pass_when_all_present(self):
        checker = PerTickInvariantChecker()
        signals = {
            "quant_direction": 0.5,
            "quant_confidence": 0.8,
            "drl_direction": -0.3,
            "sentiment_zscore": -1.5,
        }
        c = checker._t5_agent_output_consumed("BTC", signals)
        assert c.status == "PASS"

    def test_warn_when_missing(self):
        checker = PerTickInvariantChecker()
        signals = {"quant_direction": 0.5}  # missing others
        c = checker._t5_agent_output_consumed("BTC", signals)
        assert c.status == "WARN"
        assert "missing" in c.detail

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import step15_gate_report as report_mod


def test_step15_gate_report_prefers_runtime_step15_status(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    status_path = data_dir / "step15_status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-12T04:02:46.738166Z",
                "run": {"started_at": "2026-03-12T03:55:32.093590Z"},
                "fills": {
                    "count": 4,
                    "fills_by_asset": {"SOL": 2, "ETH": 2},
                },
                "positions": {
                    "assets_with_positions": ["SOL", "ETH"],
                },
                "alpha_gate": {
                    "passed": 6,
                    "total": 6,
                    "pass_rate": 1.0,
                },
                "ood": {
                    "scored_decisions": 6,
                    "soft_degrade_count": 0,
                    "soft_degrade_ratio": 0.0,
                },
                "observation": {
                    "single_obs_dim": 126,
                    "effective_stacked_dim": 1008,
                },
                "wavelet": {
                    "meets_nonzero_and_changing_evidence": True,
                    "observed_assets": ["BTC", "ETH", "SOL"],
                },
                "kill_switch": {
                    "manual_test_passed": True,
                    "last_manual_test_at": "2026-03-12T03:36:29.741728Z",
                    "last_manual_test_source": "FORCE_FLAT",
                },
                "realized_pnl": {
                    "total": {"total_pnl_usd": 23.5},
                    "by_side": {"short": {"count": 2, "total_pnl_usd": 23.5}},
                },
                "execution_advisory": {
                    "guard_enabled": True,
                    "min_session_fills": 3,
                    "min_asset_fills": 1,
                    "max_latest_slippage_bps": 12.0,
                    "max_toxicity_rate": 0.60,
                },
                "profit_calibration": {
                    "enabled": True,
                    "mode": "SHADOW",
                    "min_closed_trades_per_side": 2,
                    "by_side": {
                        "long": {"recommendation": "KEEP_LONG_PATH_OPEN"},
                        "short": {"recommendation": "RELAX_SHORT_STRUCTURE_CANDIDATE"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(report_mod, "MANUAL_GATES_DIR", data_dir / "manual_gates")
    monkeypatch.setattr(report_mod, "STEP16_MANUAL_CHECKS", data_dir / "manual_gates" / "step16_manual_checks.json")
    monkeypatch.setattr(report_mod, "STEP15_STATUS", status_path)
    monkeypatch.setattr(report_mod, "iter_log_events", lambda: [])
    monkeypatch.setattr(
        report_mod,
        "gather_anchor",
        lambda events: datetime(2026, 3, 12, 4, 5, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(report_mod, "load_shadow_ledger", lambda start: [])
    monkeypatch.setattr(report_mod, "load_live_experiences", lambda start: [])
    monkeypatch.setattr(report_mod, "load_diagnostics", lambda start: [])
    monkeypatch.setattr(report_mod, "load_feature_manifest", lambda: {})
    monkeypatch.setattr(report_mod, "load_manual_step16_checks", lambda: {})
    monkeypatch.setattr(report_mod, "latest_start_banner", lambda: (None, 0))

    result = report_mod.build_report(hours=48)
    checks = {item["name"]: item for item in result["checks"]}

    assert checks["DRL Shadow Obs Shape"]["status"] == "PASS"
    assert "runtime obs=126" in checks["DRL Shadow Obs Shape"]["detail"]
    assert checks["Wavelet Features Nonzero"]["status"] == "PASS"
    assert checks["Kill Switch Manual Test"]["status"] == "PASS"
    assert "step15_status manual_test_passed=true" in checks["Kill Switch Manual Test"]["detail"]
    assert checks["Execution Advisory Runtime"]["status"] == "PASS"
    assert "guard_enabled=True" in checks["Execution Advisory Runtime"]["detail"]
    assert checks["Profit Calibration Shadow"]["status"] == "PASS"
    assert "short=RELAX_SHORT_STRUCTURE_CANDIDATE" in checks["Profit Calibration Shadow"]["detail"]
    assert checks["Simulated Trades 24h"]["detail"] == "fills_24h=4 by_asset={'SOL': 2, 'ETH': 2}"
    assert result["metrics"]["runtime_step15_status_loaded"] is True
    assert result["metrics"]["runtime_realized_pnl_total_usd"] == 23.5
    assert result["metrics"]["runtime_realized_pnl_by_side"]["short"]["total_pnl_usd"] == 23.5
    assert result["metrics"]["runtime_execution_advisory"]["guard_enabled"] is True
    assert result["metrics"]["runtime_profit_calibration"]["by_side"]["short"]["recommendation"] == "RELAX_SHORT_STRUCTURE_CANDIDATE"


def test_step15_gate_report_prefers_runtime_rolling_fill_views(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    status_path = data_dir / "step15_status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-15T03:40:00Z",
                "run": {"started_at": "2026-03-12T04:44:27.883247Z"},
                "fills": {
                    "count": 1,
                    "fills_by_asset": {"SOL": 1},
                    "rolling_24h": {
                        "count": 5,
                        "fills_by_asset": {"SOL": 2, "ETH": 2, "BTC": 1},
                    },
                    "rolling_48h": {
                        "count": 11,
                        "fills_by_asset": {"SOL": 4, "ETH": 4, "BTC": 3},
                    },
                },
                "positions": {"assets_with_positions": ["SOL", "ETH"]},
                "alpha_gate": {"passed": 8, "total": 10, "pass_rate": 0.8},
                "ood": {"scored_decisions": 10, "soft_degrade_count": 0, "soft_degrade_ratio": 0.0},
                "observation": {"single_obs_dim": 126, "effective_stacked_dim": 1008},
                "wavelet": {"meets_nonzero_and_changing_evidence": True, "observed_assets": ["BTC", "ETH", "SOL"]},
                "kill_switch": {"manual_test_passed": True, "last_manual_test_at": "2026-03-12T03:36:29.741728Z", "last_manual_test_source": "FORCE_FLAT"},
                "execution_advisory": {"guard_enabled": True, "min_session_fills": 3, "min_asset_fills": 1, "max_latest_slippage_bps": 12.0, "max_toxicity_rate": 0.6},
                "profit_calibration": {"enabled": True, "mode": "SHADOW", "min_closed_trades_per_side": 2, "by_side": {"long": {"recommendation": "INSUFFICIENT_SAMPLE"}, "short": {"recommendation": "INSUFFICIENT_SAMPLE"}}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(report_mod, "MANUAL_GATES_DIR", data_dir / "manual_gates")
    monkeypatch.setattr(report_mod, "STEP16_MANUAL_CHECKS", data_dir / "manual_gates" / "step16_manual_checks.json")
    monkeypatch.setattr(report_mod, "STEP15_STATUS", status_path)
    monkeypatch.setattr(report_mod, "iter_log_events", lambda: [])
    monkeypatch.setattr(report_mod, "gather_anchor", lambda events: datetime(2026, 3, 15, 3, 42, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(report_mod, "load_shadow_ledger", lambda start: [])
    monkeypatch.setattr(report_mod, "load_live_experiences", lambda start: [])
    monkeypatch.setattr(report_mod, "load_diagnostics", lambda start: [])
    monkeypatch.setattr(report_mod, "load_feature_manifest", lambda: {})
    monkeypatch.setattr(report_mod, "load_manual_step16_checks", lambda: {})
    monkeypatch.setattr(report_mod, "latest_start_banner", lambda: (None, 0))

    result = report_mod.build_report(hours=48)
    checks = {item["name"]: item for item in result["checks"]}

    assert checks["Simulated Trades 24h"]["status"] == "PASS"
    assert checks["Simulated Trades 24h"]["detail"] == "fills_24h=5 by_asset={'SOL': 2, 'ETH': 2, 'BTC': 1}"
    assert checks["48h Trades / Assets"]["status"] == "PASS"
    assert checks["48h Trades / Assets"]["detail"] == "fills_48h=11 by_asset={'SOL': 4, 'ETH': 4, 'BTC': 3}"
    assert result["metrics"]["reported_fills_24h"] == 5
    assert result["metrics"]["reported_fills_48h"] == 11


def test_step15_gate_report_uses_48h_observation_count_for_ood_ratio(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(report_mod, "MANUAL_GATES_DIR", data_dir / "manual_gates")
    monkeypatch.setattr(report_mod, "STEP16_MANUAL_CHECKS", data_dir / "manual_gates" / "step16_manual_checks.json")
    monkeypatch.setattr(report_mod, "STEP15_STATUS", data_dir / "step15_status.json")
    monkeypatch.setattr(
        report_mod,
        "iter_log_events",
        lambda: [
            (
                datetime(2026, 3, 14, 8, 0, 0, tzinfo=timezone.utc),
                "logs/paper_run_stderr.log",
                "2026-03-14 03:00:00 [OOD] soft degrade engaged",
            ),
        ],
    )
    monkeypatch.setattr(report_mod, "gather_anchor", lambda events: datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(report_mod, "load_shadow_ledger", lambda start: [])
    monkeypatch.setattr(
        report_mod,
        "load_live_experiences",
        lambda start: [
            {
                "type": "observation",
                "timestamp": "2026-03-14T06:00:00Z",
                "_dt": datetime(2026, 3, 14, 6, 0, 0, tzinfo=timezone.utc),
            },
            {
                "type": "observation",
                "timestamp": "2026-03-13T02:00:00Z",
                "_dt": datetime(2026, 3, 13, 2, 0, 0, tzinfo=timezone.utc),
            },
        ],
    )
    monkeypatch.setattr(report_mod, "load_diagnostics", lambda start: [])
    monkeypatch.setattr(report_mod, "load_feature_manifest", lambda: {})
    monkeypatch.setattr(report_mod, "load_manual_step16_checks", lambda: {})
    monkeypatch.setattr(report_mod, "latest_start_banner", lambda: (None, 0))

    result = report_mod.build_report(hours=48)
    checks = {item["name"]: item for item in result["checks"]}

    assert checks["OOD Soft Degrade Ratio"]["detail"] == "soft_degrade=1 decisions=2 ratio=50.0%"


def test_step15_gate_report_uses_runtime_dashboard_for_lstm_and_ood(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    status_path = data_dir / "step15_status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-14T00:02:43.613346Z",
                "run": {"started_at": "2026-03-12T04:44:27.883247Z"},
                "alpha_gate": {"passed": 12, "total": 15, "pass_rate": 0.8},
                "fills": {"count": 0, "fills_by_asset": {}},
                "ood": {"scored_decisions": 36, "soft_degrade_count": 0, "soft_degrade_ratio": 0.0},
                "observation": {"single_obs_dim": 126, "effective_stacked_dim": 1008},
                "wavelet": {"meets_nonzero_and_changing_evidence": True, "observed_assets": ["BTC", "ETH", "SOL"]},
                "kill_switch": {"manual_test_passed": True, "last_manual_test_at": "2026-03-12T03:36:29.741728Z", "last_manual_test_source": "FORCE_FLAT"},
                "execution_advisory": {"guard_enabled": True, "min_session_fills": 3, "min_asset_fills": 1, "max_latest_slippage_bps": 12.0, "max_toxicity_rate": 0.6},
                "profit_calibration": {"enabled": True, "mode": "SHADOW", "min_closed_trades_per_side": 2, "by_side": {"long": {"recommendation": "INSUFFICIENT_SAMPLE"}, "short": {"recommendation": "INSUFFICIENT_SAMPLE"}}},
            }
        ),
        encoding="utf-8",
    )
    dashboard_path = data_dir / "dashboard_state.json"
    dashboard_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-14T00:02:43.613346Z",
                "drl": {
                    "models_ready": 3,
                    "ready_assets": ["BTC", "ETH", "SOL"],
                },
                "assets": {
                    "BTC": {"ood_distance": 9.3, "ood_threshold": 18.2, "ood_hard_switch": False, "ood_shadow_alert": False},
                    "ETH": {"ood_distance": 9.1, "ood_threshold": 18.5, "ood_hard_switch": False, "ood_shadow_alert": False},
                    "SOL": {"ood_distance": 10.6, "ood_threshold": 17.5, "ood_hard_switch": False, "ood_shadow_alert": False},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(report_mod, "MANUAL_GATES_DIR", data_dir / "manual_gates")
    monkeypatch.setattr(report_mod, "STEP16_MANUAL_CHECKS", data_dir / "manual_gates" / "step16_manual_checks.json")
    monkeypatch.setattr(report_mod, "STEP15_STATUS", status_path)
    monkeypatch.setattr(report_mod, "DASHBOARD_STATE", dashboard_path)
    monkeypatch.setattr(report_mod, "iter_log_events", lambda: [])
    monkeypatch.setattr(report_mod, "gather_anchor", lambda events: datetime(2026, 3, 14, 0, 57, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(report_mod, "load_shadow_ledger", lambda start: [])
    monkeypatch.setattr(report_mod, "load_live_experiences", lambda start: [])
    monkeypatch.setattr(report_mod, "load_diagnostics", lambda start: [])
    monkeypatch.setattr(report_mod, "load_feature_manifest", lambda: {})
    monkeypatch.setattr(report_mod, "load_manual_step16_checks", lambda: {})
    monkeypatch.setattr(report_mod, "latest_start_banner", lambda: (None, 0))

    result = report_mod.build_report(hours=48)
    checks = {item["name"]: item for item in result["checks"]}

    assert checks["LSTM FiLM Extractor Loads"]["status"] == "PASS"
    assert "runtime=BTC,ETH,SOL" in checks["LSTM FiLM Extractor Loads"]["detail"]
    assert checks["OOD Detector Running"]["status"] == "WARN"
    assert "loaded=BTC,ETH,SOL" in checks["OOD Detector Running"]["detail"]
    assert "runtime=BTC,ETH,SOL" in checks["OOD Detector Running"]["detail"]


def test_step15_gate_report_accepts_double_timezone_updated_at(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    status_path = data_dir / "step15_status.json"
    status_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-14T00:02:43.613346+00:00Z",
                "run": {"started_at": "2026-03-12T04:44:27.883247Z"},
                "fills": {"count": 0, "fills_by_asset": {}},
                "alpha_gate": {"passed": 1, "total": 1, "pass_rate": 1.0},
                "ood": {"scored_decisions": 1, "soft_degrade_count": 0, "soft_degrade_ratio": 0.0},
                "observation": {"single_obs_dim": 126, "effective_stacked_dim": 1008},
                "wavelet": {"meets_nonzero_and_changing_evidence": True, "observed_assets": ["BTC", "ETH", "SOL"]},
                "kill_switch": {"manual_test_passed": True, "last_manual_test_at": "2026-03-12T03:36:29.741728Z", "last_manual_test_source": "FORCE_FLAT"},
                "execution_advisory": {"guard_enabled": True, "min_session_fills": 3, "min_asset_fills": 1, "max_latest_slippage_bps": 12.0, "max_toxicity_rate": 0.6},
                "profit_calibration": {"enabled": True, "mode": "SHADOW", "min_closed_trades_per_side": 2, "by_side": {"long": {"recommendation": "INSUFFICIENT_SAMPLE"}, "short": {"recommendation": "INSUFFICIENT_SAMPLE"}}},
            }
        ),
        encoding="utf-8",
    )
    dashboard_path = data_dir / "dashboard_state.json"
    dashboard_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-14T00:02:43.613346+00:00Z",
                "drl": {"models_ready": 3, "ready_assets": ["BTC", "ETH", "SOL"]},
                "assets": {
                    "BTC": {"drl_models_ready": 3, "ood_distance": 9.3, "ood_threshold": 18.2},
                    "ETH": {"drl_models_ready": 3, "ood_distance": 9.1, "ood_threshold": 18.5},
                    "SOL": {"drl_models_ready": 3, "ood_distance": 10.6, "ood_threshold": 17.5},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(report_mod, "ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(report_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(report_mod, "MANUAL_GATES_DIR", data_dir / "manual_gates")
    monkeypatch.setattr(report_mod, "STEP16_MANUAL_CHECKS", data_dir / "manual_gates" / "step16_manual_checks.json")
    monkeypatch.setattr(report_mod, "STEP15_STATUS", status_path)
    monkeypatch.setattr(report_mod, "DASHBOARD_STATE", dashboard_path)
    monkeypatch.setattr(report_mod, "iter_log_events", lambda: [])
    monkeypatch.setattr(report_mod, "gather_anchor", lambda events: datetime(2026, 3, 14, 0, 57, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(report_mod, "load_shadow_ledger", lambda start: [])
    monkeypatch.setattr(report_mod, "load_live_experiences", lambda start: [])
    monkeypatch.setattr(report_mod, "load_diagnostics", lambda start: [])
    monkeypatch.setattr(report_mod, "load_feature_manifest", lambda: {})
    monkeypatch.setattr(report_mod, "load_manual_step16_checks", lambda: {})
    monkeypatch.setattr(report_mod, "latest_start_banner", lambda: (None, 0))

    result = report_mod.build_report(hours=48)
    checks = {item["name"]: item for item in result["checks"]}

    assert checks["LSTM FiLM Extractor Loads"]["status"] == "PASS"
    assert "runtime=BTC,ETH,SOL" in checks["LSTM FiLM Extractor Loads"]["detail"]

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from main import HMATSProductionRunner, RunMode
from core.runtime_authority import ALLOWED_RUNTIME_AUTHORITY_STATES


def _make_runner(tmp_path):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(
        initial_capital=10_000.0,
        risk_profile="HIGH_RISK",
        assets=["BTC", "ETH", "SOL"],
        # [P160] Required by _export_dashboard_state (ProductionConfig.mode,
        # main.py:1361), which is the entry point these tests drive. Without
        # it the AttributeError was swallowed at DEBUG and every assertion
        # failed on a missing file instead of naming the cause.
        mode=RunMode.PAPER,
    )
    runner.account_sync = None
    runner.dead_man_switch = None
    runner._paper_positions = {}
    runner._position_entry_times = {}
    runner._dashboard_asset_runtime = {}
    runner._dashboard_asset_snapshot = {}
    runner._step15_alpha_gate_stats = {"total": 0, "passed": 0, "by_asset": {}}
    runner._step15_ood_stats = {
        "decisions": 0,
        "soft_degrade": 0,
        "hard_switch": 0,
        "shadow_alert": 0,
        "by_asset": {},
    }
    runner._tick_crash_count = 0
    runner._consecutive_crashes = {}
    runner._DASHBOARD_STATE_FILE = tmp_path / "dashboard_state.json"
    runner._STEP15_STATUS_FILE = tmp_path / "step15_status.json"
    runner._PAPER_RUN_PID_FILE = tmp_path / "paper_run.pid"
    runner._KILL_SWITCH_TEST_FILE = tmp_path / "kill_switch_test.json"
    runner._step15_wavelet_stats = {"ticks": 0, "by_asset": {}}
    runner.p0_integrator = None
    runner._drl_models_ready = 0
    runner._drl_ensembles = {}
    runner._drl_bootstrap_applied = False
    runner.learned_exec_policy = None
    return runner


def test_step15_status_export_tracks_runtime_acceptance_metrics(tmp_path):
    runner = _make_runner(tmp_path)
    runner.dead_man_switch = object()
    runner._profit_calibration_shadow_enabled = True
    runner._profit_calibration_shadow_min_closed_trades_per_side = 2
    runner._profit_calibration_shadow_tighten_long_avg_bps_below = -10.0
    runner._profit_calibration_shadow_tighten_long_total_pnl_usd_below = -20.0
    runner._profit_calibration_shadow_keep_long_avg_bps_above = 10.0
    runner._profit_calibration_shadow_relax_short_avg_bps_above = 15.0
    runner._profit_calibration_shadow_relax_short_total_pnl_usd_above = 20.0
    runner._execution_advisory_guard_enabled = True
    runner._execution_advisory_min_session_fills = 3
    runner._execution_advisory_min_asset_fills = 1
    runner._execution_advisory_require_verified_fill_telemetry = True
    runner._execution_advisory_max_latest_slippage_bps = 12.0
    runner._execution_advisory_max_toxicity_rate = 0.60
    runner._execution_advisory_min_side_closed_trades = 1
    runner._execution_advisory_min_side_total_realized_pnl_usd = -25.0
    runner._execution_advisory_min_side_avg_realized_pnl_bps = -15.0
    runner._record_realized_pnl_breakdown(
        asset="SOL",
        side="short",
        strategy="momentum",
        pnl_usd=42.5,
        pnl_bps=110.0,
        exit_type="FULL",
    )
    runner._paper_positions = {
        "SOL": {
            "direction": -1.0,
            "exposure": 0.1,
            "entry_price": 85.0,
            "notional": 100.0,
            "entry_tick": 1,
            "tranche": 4,
            "entry_time": "2026-03-12T02:03:07+00:00",
        }
    }
    runner._dashboard_asset_runtime = {
        "SOL": {"session_fill_count": 1},
        "ETH": {"session_fill_count": 1},
        "BTC": {
            "session_fill_count": 0,
            "btc_short_shadow_relax_candidate": True,
            "btc_short_shadow_relax_applied": False,
            "btc_short_shadow_relax_reason": "BTC short shadow-relax candidate gap=32.0bps net_edge=2.8bps vpin=0.42 scale=0.35 enabled=False",
        },
    }
    runner._ea_tracker = SimpleNamespace(
        summary_dict=lambda: {
            "tracked_trades": 3,
            "avg_retention_pct": 68.0,
            "avg_peak_unrealized_bps": 140.0,
            "avg_realized_bps": 95.0,
            "avg_peak_to_exit_bps": 45.0,
            "total_pnl_usd": 84.0,
            "by_trigger": {"T5_PROFIT_DRAWDOWN": {"count": 2, "avg_retention_pct": 72.0}},
            "by_asset": {"SOL": {"count": 2, "avg_retention_pct": 70.0, "total_pnl_usd": 60.0}},
        }
    )
    runner._step15_alpha_gate_stats = {
        "total": 3,
        "passed": 2,
        "by_asset": {
            "SOL": {"total": 1, "passed": 1},
            "BTC": {"total": 1, "passed": 0},
            "ETH": {"total": 1, "passed": 1},
        },
    }
    runner._step15_ood_stats = {
        "decisions": 6,
        "soft_degrade": 1,
        "hard_switch": 0,
        "shadow_alert": 2,
        "by_asset": {
            "SOL": {"decisions": 2, "soft_degrade": 0, "hard_switch": 0, "shadow_alert": 1},
            "BTC": {"decisions": 2, "soft_degrade": 0, "hard_switch": 0, "shadow_alert": 1},
            "ETH": {"decisions": 2, "soft_degrade": 1, "hard_switch": 0, "shadow_alert": 0},
        },
    }
    runner._tick_crash_count = 1
    runner._consecutive_crashes = {"ETH": 1}

    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps(
            {
                "pid": 24416,
                "started_at": "2026-03-12T02:07:14.258956Z",
                "command": "main.py --mode paper --risk-profile HIGH_RISK",
            }
        ),
        encoding="utf-8",
    )

    runner._export_dashboard_state(
        3,
        9,
        {
            "SOL": {"price": 86.07, "alpha_gate_passed": True, "actionable": False, "veto_reason": "EXPOSURE_DELTA_BELOW_THRESHOLD"},
            "BTC": {"price": 69841.2, "alpha_gate_passed": False, "actionable": False, "veto_reason": "alpha gate"},
            "ETH": {"price": 2045.19, "alpha_gate_passed": True, "actionable": False, "veto_reason": "[STRUCTURE] SHORT blocked -no fractal low break"},
        },
    )

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))

    assert status["round"] == 3
    assert status["tick_count"] == 9
    assert status["run"]["started_at"] == "2026-03-12T02:07:14.258956Z"
    assert status["fills"]["count"] == 2
    assert status["fills"]["fills_by_asset"] == {"ETH": 1, "SOL": 1}
    assert status["fills"]["meets_5plus_threshold"] is False
    assert status["positions"]["count"] == 1
    assert status["positions"]["assets_with_positions"] == ["SOL"]
    assert status["alpha_gate"]["passed"] == 2
    assert status["alpha_gate"]["total"] == 3
    assert status["alpha_gate"]["pass_rate"] == pytest.approx(2 / 3)
    assert status["alpha_gate"]["meets_40pct_threshold"] is True
    assert status["ood"]["soft_degrade_count"] == 1
    assert status["ood"]["soft_degrade_ratio"] == pytest.approx(1 / 6)
    assert status["ood"]["shadow_alert_count"] == 2
    assert status["realized_pnl"]["closed_trade_count"] == 1
    assert status["realized_pnl"]["by_side"]["short"]["total_pnl_usd"] == pytest.approx(42.5)
    assert status["realized_pnl"]["by_side"]["short"]["avg_pnl_bps"] == pytest.approx(110.0)
    assert status["execution_advisory"]["guard_enabled"] is True
    assert status["execution_advisory"]["require_verified_fill_telemetry"] is True
    assert status["profit_calibration"]["mode"] == "SHADOW"
    assert status["profit_calibration"]["by_side"]["short"]["recommendation"] == "INSUFFICIENT_SAMPLE"
    assert status["exit_alpha_review"]["tracked_trades"] == 3
    assert status["exit_alpha_review"]["avg_retention_pct"] == pytest.approx(68.0)
    allowed = set(ALLOWED_RUNTIME_AUTHORITY_STATES)
    assert set(status["authority_states"]["allowed_states"]) == allowed
    assert set(status["assets"]["SOL"]["authority_states"].values()).issubset(allowed)
    assert status["kill_switch"]["dead_man_running"] is True
    assert status["kill_switch"]["manual_test_passed"] is False
    assert status["assets"]["ETH"]["veto_reason"] == "[STRUCTURE] SHORT blocked -no fractal low break"
    assert status["assets"]["BTC"]["btc_short_shadow_relax_candidate"] is True
    assert status["assets"]["BTC"]["btc_short_shadow_relax_applied"] is False
    assert "enabled=False" in status["assets"]["BTC"]["btc_short_shadow_relax_reason"]


def test_step15_status_export_includes_learned_execution_runtime_truth(tmp_path):
    runner = _make_runner(tmp_path)
    runner.learned_exec_policy = SimpleNamespace(
        config=SimpleNamespace(mode="advisory"),
        get_status=lambda: {
            "mode": "advisory",
            "effective_mode": "ADVISORY_HEURISTIC",
            "weights_loaded": False,
            "heuristic_only": True,
            "fused": False,
            "fuse_reason": "",
            "last_advice_source": "fallback",
            "last_advice_reason": "missing_embedding",
        },
    )
    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps({"pid": 1, "started_at": "2026-03-12T02:07:14.258956Z"}),
        encoding="utf-8",
    )

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))
    dashboard = json.loads(runner._DASHBOARD_STATE_FILE.read_text(encoding="utf-8"))

    assert status["learned_execution_policy"]["available"] is True
    assert status["learned_execution_policy"]["configured_mode"] == "ADVISORY"
    assert status["learned_execution_policy"]["effective_mode"] == "ADVISORY_HEURISTIC"
    assert status["learned_execution_policy"]["weights_loaded"] is False
    assert status["learned_execution_policy"]["heuristic_only"] is True
    assert status["learned_execution_policy"]["last_advice_source"] == "fallback"
    assert dashboard["learned_execution_policy"]["effective_mode"] == "ADVISORY_HEURISTIC"


def test_step15_status_export_reads_kill_switch_record(tmp_path):
    runner = _make_runner(tmp_path)
    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps({"pid": 1, "started_at": "2026-03-12T02:07:14.258956Z"}),
        encoding="utf-8",
    )
    runner._KILL_SWITCH_TEST_FILE.write_text(
        json.dumps(
            {
                "triggered_at": "2026-03-12T05:00:00Z",
                "source": "FORCE_FLAT",
                "reason": "manual acceptance test",
                "mode": "PAPER",
            }
        ),
        encoding="utf-8",
    )

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))

    assert status["kill_switch"]["manual_test_passed"] is True
    assert status["kill_switch"]["last_manual_test_at"] == "2026-03-12T05:00:00Z"
    assert status["kill_switch"]["last_manual_test_source"] == "FORCE_FLAT"
    assert status["kill_switch"]["last_manual_test_reason"] == "manual acceptance test"


def test_step15_status_export_accepts_legacy_double_timezone_kill_record(tmp_path):
    runner = _make_runner(tmp_path)
    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps({"pid": 1, "started_at": "2026-03-12T02:07:14.258956Z"}),
        encoding="utf-8",
    )
    runner._KILL_SWITCH_TEST_FILE.write_text(
        json.dumps(
            {
                "triggered_at": "2026-03-12T05:00:00+00:00Z",
                "source": "FORCE_FLAT",
                "reason": "manual acceptance test",
                "mode": "PAPER",
            }
        ),
        encoding="utf-8",
    )

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))

    assert status["kill_switch"]["manual_test_passed"] is True
    assert status["kill_switch"]["last_manual_test_at"] == "2026-03-12T05:00:00+00:00Z"


def test_step15_status_export_includes_observation_and_wavelet_evidence(tmp_path):
    runner = _make_runner(tmp_path)
    runner._obs_builder = SimpleNamespace(feature_cols=[f"f{i}" for i in range(122)])
    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps({"pid": 1, "started_at": "2026-03-12T02:07:14.258956Z"}),
        encoding="utf-8",
    )
    runner._step15_wavelet_stats = {
        "ticks": 2,
        "by_asset": {
            "SOL": {
                "ticks": 2,
                "any_nonzero_ticks": 2,
                "all_nonzero_ticks": 2,
                "changed_ticks": 1,
                "features": {
                    feature: {"nonzero_count": 2, "min": 0.1, "max": 0.2, "last": 0.2}
                    for feature in (
                        "rsi_14_denoised",
                        "macd_12_26_denoised",
                        "bb_width_20_denoised",
                        "atr_14_denoised",
                        "vol_ratio_s_denoised",
                    )
                },
            }
        },
    }

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}})

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))

    assert status["observation"]["single_obs_dim"] == 126
    assert status["observation"]["effective_stacked_dim"] == 1008
    assert status["observation"]["stacking_mode"] == "internal_runtime_buffer"
    assert status["wavelet"]["feature_count"] == 5
    assert status["wavelet"]["meets_nonzero_and_changing_evidence"] is True
    assert status["wavelet"]["by_asset"]["SOL"]["meets_nonzero_evidence"] is True
    assert status["wavelet"]["by_asset"]["SOL"]["meets_changing_evidence"] is True


def test_step15_status_export_uses_shadow_ledger_session_fill_helper(tmp_path):
    class _ShadowLedger:
        def get_session_fill_summary(self):
            return {
                "count": 2,
                "counts_by_asset": {"SOL": 1, "ETH": 1},
                "last_fill_at": "2026-03-12T06:00:00Z",
                "last_fill_at_by_asset": {
                    "SOL": "2026-03-12T05:30:00Z",
                    "ETH": "2026-03-12T06:00:00Z",
                },
            }

    runner = _make_runner(tmp_path)
    runner.p0_integrator = SimpleNamespace(shadow_ledger=_ShadowLedger())
    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps({"pid": 1, "started_at": "2026-03-12T02:07:14.258956Z"}),
        encoding="utf-8",
    )

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}, "ETH": {"price": 200.0}})

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))

    assert status["fills"]["count"] == 2
    assert status["fills"]["fills_by_asset"] == {"ETH": 1, "SOL": 1}
    assert status["fills"]["last_fill_at"] == "2026-03-12T06:00:00Z"
    assert status["fills"]["last_fill_at_by_asset"]["SOL"] == "2026-03-12T05:30:00Z"


def test_step15_status_export_includes_session_and_rolling_fill_views(tmp_path):
    now = datetime.now(timezone.utc)
    ledger_dir = tmp_path / "shadow_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / f"ledger_{now.strftime('%Y%m%d')}.jsonl"
    rows = [
        {
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "entry_type": "FILL",
            "asset": "SOL",
            "data": {"realized_pnl": 0.0},
        },
        {
            "timestamp": (now - timedelta(hours=23)).isoformat(),
            "entry_type": "FILL",
            "asset": "ETH",
            "data": {"realized_pnl": 0.0},
        },
        {
            "timestamp": (now - timedelta(hours=30)).isoformat(),
            "entry_type": "FILL",
            "asset": "BTC",
            "data": {"realized_pnl": 0.0},
        },
        {
            "timestamp": (now - timedelta(hours=52)).isoformat(),
            "entry_type": "FILL",
            "asset": "SOL",
            "data": {"realized_pnl": 0.0},
        },
    ]
    ledger_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    class _ShadowLedger:
        output_dir = ledger_dir

        @staticmethod
        def get_session_fill_summary():
            return {
                "count": 1,
                "counts_by_asset": {"SOL": 1},
                "last_fill_at": rows[0]["timestamp"],
                "last_fill_at_by_asset": {"SOL": rows[0]["timestamp"]},
            }

        @staticmethod
        def flush():
            return None

    runner = _make_runner(tmp_path)
    runner.p0_integrator = SimpleNamespace(shadow_ledger=_ShadowLedger())
    runner._PAPER_RUN_PID_FILE.write_text(
        json.dumps({"pid": 1, "started_at": "2026-03-12T02:07:14.258956Z"}),
        encoding="utf-8",
    )

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0}, "ETH": {"price": 200.0}, "BTC": {"price": 300.0}})

    status = json.loads(runner._STEP15_STATUS_FILE.read_text(encoding="utf-8"))

    assert status["fills"]["scope"] == "session"
    assert status["fills"]["count"] == 1
    assert status["fills"]["session"]["fills_by_asset"] == {"SOL": 1}
    assert status["fills"]["rolling_24h"]["count"] == 2
    assert status["fills"]["rolling_24h"]["fills_by_asset"] == {"ETH": 1, "SOL": 1}
    assert status["fills"]["rolling_24h"]["meets_5plus_threshold"] is False
    assert status["fills"]["rolling_48h"]["count"] == 3
    assert status["fills"]["rolling_48h"]["fills_by_asset"] == {"BTC": 1, "ETH": 1, "SOL": 1}
    assert status["fills"]["rolling_48h"]["meets_10plus_threshold"] is False
    assert status["positions"]["three_assets_trading"] is True
    assert status["positions"]["three_assets_trading_session"] is False
    assert status["positions"]["three_assets_trading_48h"] is True

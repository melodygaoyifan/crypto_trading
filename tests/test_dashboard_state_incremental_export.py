import json
from types import SimpleNamespace

from main import HMATSProductionRunner
from core.runtime_authority import ALLOWED_RUNTIME_AUTHORITY_STATES


def _make_runner(tmp_path):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(initial_capital=10_000.0, risk_profile="HIGH_RISK")
    runner.account_sync = None
    runner._paper_positions = {}
    runner._position_entry_times = {}
    runner._dashboard_asset_snapshot = {}
    runner._dashboard_asset_runtime = {}
    runner._DASHBOARD_STATE_FILE = tmp_path / "dashboard_state.json"
    runner._STEP15_STATUS_FILE = tmp_path / "step15_status.json"
    runner._PAPER_RUN_PID_FILE = tmp_path / "paper_run.pid"
    runner._KILL_SWITCH_TEST_FILE = tmp_path / "kill_switch_test.json"
    runner._drl_models_ready = 0
    runner._drl_ensembles = {}
    runner._drl_bootstrap_applied = False
    return runner


def test_incremental_dashboard_export_preserves_previous_assets(tmp_path):
    runner = _make_runner(tmp_path)

    runner._export_dashboard_state(1, 1, {"SOL": {"price": 100.0, "llm_tradeable_alpha": True}})
    runner._export_dashboard_state(1, 2, {"BTC": {"price": 200.0}})

    state = json.loads(runner._DASHBOARD_STATE_FILE.read_text(encoding="utf-8"))

    assert state["tick_count"] == 2
    assert set(state["assets"]) == {"SOL", "BTC"}
    assert state["assets"]["SOL"]["price"] == 100.0
    assert state["assets"]["BTC"]["price"] == 200.0
    assert state["sentiment"]["assets"]["SOL"]["tradeable_alpha"] is True
    allowed = set(ALLOWED_RUNTIME_AUTHORITY_STATES)
    assert set(state["authority_states"]["allowed_states"]) == allowed
    assert set(state["authority_states"]["by_asset"]["SOL"].values()).issubset(allowed)
    assert set(state["authority_states"]["by_asset"]["BTC"].values()).issubset(allowed)


def test_incremental_dashboard_export_overwrites_current_asset_snapshot(tmp_path):
    runner = _make_runner(tmp_path)
    runner._record_realized_pnl_breakdown(
        asset="ETH",
        side="long",
        strategy="momentum",
        pnl_usd=-12.5,
        pnl_bps=-25.0,
        exit_type="FULL",
    )

    runner._export_dashboard_state(1, 1, {"ETH": {"price": 2000.0, "actionable": False}})
    runner._export_dashboard_state(1, 2, {"ETH": {"price": 2025.0, "actionable": True}})

    state = json.loads(runner._DASHBOARD_STATE_FILE.read_text(encoding="utf-8"))

    assert set(state["assets"]) == {"ETH"}
    assert state["assets"]["ETH"]["price"] == 2025.0
    assert state["assets"]["ETH"]["actionable"] is True
    assert state["realized_pnl"]["by_side"]["long"]["total_pnl_usd"] == -12.5
    assert set(state["assets"]["ETH"]["authority_states"].values()).issubset(
        set(ALLOWED_RUNTIME_AUTHORITY_STATES)
    )

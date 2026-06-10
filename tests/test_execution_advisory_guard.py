from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from execution.composite_toxicity import CompositeToxicityResult
from execution.learned_execution_policy import ExecutionAction, ExecutionAdvice
from main import HMATSProductionRunner, RunMode

# [TEST-CLEANUP 2026-06-09] All 4 tests in this file drive
# HMATSProductionRunner._execute_intent which was retired per P22 cutover
# to execute_intent_v2. Same skip rationale as test_realized_pnl_exit_path.py.
pytestmark = pytest.mark.skip(
    reason="[TEST-CLEANUP] HMATSProductionRunner._execute_intent removed per "
           "P22 cutover; tests not yet ported to execute_intent_v2 driver."
)


class FakeAccountSync:
    def __init__(self, equity: float = 10_000.0):
        self.dry_run = True
        self._base_equity = equity
        self._dry_run_pnl = 0.0

    async def refresh(self):
        return None

    def get_equity(self) -> float:
        return self._base_equity + self._dry_run_pnl

    def update_dry_run_pnl(self, pnl_usd: float):
        self._dry_run_pnl += float(pnl_usd)


class FakeRiskManager:
    def __init__(self):
        self.realized_pnls = []
        self.balance_updates = []

    def record_realized_pnl(self, pnl_usd: float):
        self.realized_pnls.append(float(pnl_usd))

    def update_balance(self, equity: float):
        self.balance_updates.append(float(equity))


class FakeExecutionResult:
    def __init__(self, *, size: float, filled_price: float):
        self.success = True
        self.status = "FILLED"
        self.filled_size = float(size)
        self.requested_size = float(size)
        self.filled_price = float(filled_price)
        self.order_id = "guard_test_order"

    def to_dict(self):
        return {
            "success": self.success,
            "status": self.status,
            "filled_size": self.filled_size,
            "requested_size": self.requested_size,
            "filled_price": self.filled_price,
            "order_id": self.order_id,
        }


class FakeExecutionManager:
    def __init__(self, *, filled_price: float):
        self.filled_price = float(filled_price)
        self.calls = []

    def execute_order(self, **kwargs):
        self.calls.append(dict(kwargs))
        return FakeExecutionResult(size=kwargs["size"], filled_price=self.filled_price)

    def cancel_stop_loss(self, symbol: str) -> bool:
        return True


class FakeP0Integrator:
    def __init__(self):
        self.fills = []

    def record_fill(self, **kwargs):
        self.fills.append(dict(kwargs))
        return True


class FakeAntiChurn:
    def __init__(self):
        self._fills_today = 0
        self._fills_date = ""
        self.fills = []

    def record_fill(self, asset: str, tick_count: int):
        self.fills.append((asset, tick_count))


class FakeWarmupTracker:
    threshold = 2

    def get_tick_count(self, asset: str) -> int:
        return 0


class FakeFillSlopeMonitor:
    def __init__(
        self,
        *,
        slippage_bps: float = 2.0,
        toxicity_rate: float = 0.1,
        fill_count: int = 2,
        telemetry_verified: bool = True,
    ):
        self._slippage_bps = float(slippage_bps)
        self._toxicity_rate = float(toxicity_rate)
        self._fill_count = int(fill_count)
        self._telemetry_verified = bool(telemetry_verified)

    def get_execution_guidance(self, asset: str):
        return {
            "force_passive": False,
            "extra_improve_bps": 0.0,
            "size_reduce_pct": 0.0,
            "toxicity_rate": self._toxicity_rate,
            "reason": "",
        }

    def get_asset_fill_metrics(self, asset: str):
        return {
            "fill_count": self._fill_count,
            "telemetry_verified": self._telemetry_verified,
            "toxicity_rate": self._toxicity_rate,
            "latest_slippage_bps": self._slippage_bps,
            "last_toxic_fill_ms": None,
            "adverse_selection_proxy": 0.1,
            "latest_is_toxic": False,
        }


class FakeCompositeToxicity:
    def compute(self, **kwargs):
        return CompositeToxicityResult(
            score=0.72,
            should_veto=False,
            should_warn=True,
            dominant_dimension="adverse_selection",
        )


class FakeLEP:
    def __init__(self):
        self.config = SimpleNamespace(mode="advisory")

    def get_advice(self, lob, market_embedding, exec_state):
        return ExecutionAdvice(
            recommended_action=ExecutionAction.WAIT,
            action_probs={"wait": 0.9},
            slice_size_pct=0.25,
            wait_time_sec=15.0,
            limit_offset_bps=4.0,
            aggressiveness=0.2,
            confidence=0.9,
        )


def _make_runner(session_fill_count: int) -> HMATSProductionRunner:
    now = datetime.now(timezone.utc)

    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(
        mode=RunMode.PAPER,
        initial_capital=10_000.0,
        post_leverage_caps={"SOL": 0.25},
        assets=["SOL", "ETH", "BTC"],
        execution_config={
            "advisory_activation": {
                "enabled": True,
                "min_session_fills": 3,
                "min_asset_fills": 1,
                "max_latest_slippage_bps": 12.0,
                "max_toxicity_rate": 0.60,
                "min_side_closed_trades": 1,
                "min_side_total_realized_pnl_usd": -25.0,
                "min_side_avg_realized_pnl_bps": -15.0,
            }
        },
    )
    runner.account_sync = FakeAccountSync()
    runner.risk_manager = FakeRiskManager()
    runner.execution_manager = FakeExecutionManager(filled_price=90.0)
    runner.p0_integrator = FakeP0Integrator()
    runner._anti_churn = FakeAntiChurn()
    runner._warmup_tracker = FakeWarmupTracker()
    runner._fill_slope_monitor = FakeFillSlopeMonitor()
    runner._composite_toxicity = FakeCompositeToxicity()
    runner.learned_exec_policy = FakeLEP()
    runner._paper_positions = {
        "SOL": {
            "exposure": 0.10,
            "direction": -1.0,
            "tranche": 1,
            "entry_price": 100.0,
            "notional": 1000.0,
            "entry_time": (now - timedelta(hours=8)).isoformat(),
            "strategy": "momentum",
            "entry_alpha_est": 12.0,
            "mode_at_entry": "OPPORTUNITY",
            "phase_at_entry": "UNKNOWN",
            "crack_weight": 0.10,
            "original_entry_exposure": 0.10,
            "cumulative_funding_pnl": 4.0,
            "entry_fee_usd": 10.0,
            "cumulative_entry_fees_usd": 10.0,
            "entry_tick": 1,
        }
    }
    runner._position_entry_times = {
        "SOL": {"entry_time": now - timedelta(hours=8), "mode": "OPPORTUNITY"}
    }
    runner._recent_fill_state = {}
    runner._asset_trade_pnls = {"SOL": []}
    runner._realized_pnl_breakdown = runner._new_realized_pnl_breakdown()
    runner._rebuild_cooldown = {}
    runner._exit_trigger_tag = {}
    runner._orphaned_stops = set()
    runner._dashboard_asset_runtime = {
        "SOL": {"session_fill_count": session_fill_count},
        "ETH": {"session_fill_count": max(0, 3 - session_fill_count)},
    }
    runner._dashboard_asset_snapshot = {}
    runner._tick_count = 5
    runner._REBUILD_COOLDOWN_TICKS = 1
    runner._AC1_MIN_HOLD_TICKS = 0
    runner._AC5_MAX_FILLS_PER_DAY = 999
    runner._ac5_fills_today = 0
    runner._ac5_fills_date = now.strftime("%Y-%m-%d")
    runner._ac2_fill_ticks = {"SOL": []}
    runner._AC2_WINDOW_TICKS = 6
    runner._AC2_MAX_FILLS_PER_ASSET_PER_DAY = 99
    runner._AC2_MAX_FILLS_GLOBAL_PER_DAY = 999
    runner._ac0_post_restart_tick = False
    runner._ac4_extended_cooldown = 0
    runner._current_drawdown_pct = 0.0
    runner._fee_blending_enabled = False
    runner._drl_models_ready = 0
    runner._drl_ensembles = {}
    runner._drl_bootstrap_applied = False
    runner._drl_authority_level = "SHADOW"
    runner._last_aging_check = None
    runner._execution_advisory_guard_enabled = True
    runner._execution_advisory_min_session_fills = 3
    runner._execution_advisory_min_asset_fills = 1
    runner._execution_advisory_require_verified_fill_telemetry = True
    runner._execution_advisory_max_latest_slippage_bps = 12.0
    runner._execution_advisory_max_toxicity_rate = 0.60
    runner._execution_advisory_min_side_closed_trades = 1
    runner._execution_advisory_min_side_total_realized_pnl_usd = -25.0
    runner._execution_advisory_min_side_avg_realized_pnl_bps = -15.0

    runner._get_drl_weight = lambda: 0.0
    runner._maybe_block_micro_rebalance = lambda **kwargs: None
    runner._persist_tranche_state = lambda: None
    runner._margin_tracker = SimpleNamespace(record_trade=lambda **kwargs: None)
    runner.engine = SimpleNamespace(_last_timing_score=None)

    optional_none_attrs = [
        "dead_man_switch",
        "execution_guard",
        "authority_chain",
        "leverage_guard",
        "_dynamic_limits_result",
        "unified_sizer",
        "_hplv_filter",
        "_alpha_tilt",
        "_impact_model",
        "pa_executor",
        "orderbook_analyzer",
        "level2_analyzer",
        "integrity_shield",
        "rl_exec_agent",
        "fill_rate_kpi",
        "_sol_exec_guard",
        "impact_cal_table",
        "exec_quality_logger",
        "audit_manager",
        "thesis_budget_governor",
        "feedback_loop",
        "gambler_exit",
        "exit_alpha",
        "_adaptive_stop",
        "opportunity_budget",
        "existence_fuse",
        "_trade_attributor",
        "_sq_tracker",
        "_experience_buffer",
        "_ea_tracker",
        "_confidence_scorer",
        "_meta_decision",
        "_pnl_attribution",
        "sota_integration",
        "_strategy_aging",
        "_failure_memory",
        "portfolio_brain",
    ]
    for attr in optional_none_attrs:
        setattr(runner, attr, None)

    return runner


def _make_intent():
    return SimpleNamespace(
        direction=-1.0,
        target_exposure=0.05,
        execution_mode=SimpleNamespace(value="AGGRESSIVE"),
        execution_urgency=1.3,
        prefer_limit_order=False,
        limit_offset_bps=0.0,
        system_mode="OPPORTUNITY",
        regime_leverage=1.0,
        quant_confidence=0.5,
        alpha_estimated_bps=15.0,
        quant_strategy_id="momentum",
        veto_active=False,
        veto_reason="",
        tranche_target=1,
        is_addon=False,
        force_execution=False,
    )


def _make_market_data():
    return {
        "current_price": 90.0,
        "regime_state": "QUIET_ACCUMULATION",
        "regime_confidence": 0.9,
        "spread_bps": 1.0,
        "vpin": 0.3,
        "vpin_source": "synthetic",
        "orderbook_depth_1pct_usd": 100000.0,
        "order_book_imbalance": 0.0,
        "atr_ratio": 1.0,
        "bid_depth_usd": 100000.0,
        "ask_depth_usd": 100000.0,
    }


@pytest.mark.asyncio
async def test_execution_advisory_guard_blocks_application_until_fill_samples_exist():
    runner = _make_runner(session_fill_count=0)
    runner._dashboard_asset_runtime["ETH"]["session_fill_count"] = 0
    intent = _make_intent()

    result = await runner._execute_intent("SOL", intent, _make_market_data(), agent_signals={})

    assert result["status"] == "FILLED"
    assert runner._dashboard_asset_runtime["SOL"]["execution_advisory_guard_open"] is False
    assert "SESSION_FILLS_0/3" in runner._dashboard_asset_runtime["SOL"]["execution_advisory_guard_reason"]
    assert runner._dashboard_asset_runtime["SOL"]["composite_toxicity_applied"] is False
    assert runner._dashboard_asset_runtime["SOL"]["learned_exec_applied"] is False
    assert runner._dashboard_asset_runtime["SOL"]["learned_exec_weights_loaded"] is False
    assert runner._dashboard_asset_runtime["SOL"]["learned_exec_heuristic_only"] is True
    assert runner.execution_manager.calls[0]["order_type"] == "MARKET"


@pytest.mark.asyncio
async def test_execution_advisory_guard_allows_application_after_fill_and_pnl_evidence():
    runner = _make_runner(session_fill_count=2)
    runner._record_realized_pnl_breakdown(
        asset="SOL",
        side="short",
        strategy="momentum",
        pnl_usd=12.0,
        pnl_bps=18.0,
        exit_type="FULL",
    )
    intent = _make_intent()

    result = await runner._execute_intent("SOL", intent, _make_market_data(), agent_signals={})

    assert result["status"] == "FILLED"
    assert runner._dashboard_asset_runtime["SOL"]["execution_advisory_guard_open"] is True
    assert runner._dashboard_asset_runtime["SOL"]["composite_toxicity_applied"] is True
    assert runner._dashboard_asset_runtime["SOL"]["learned_exec_applied"] is True
    assert runner._dashboard_asset_runtime["SOL"]["learned_exec_weights_loaded"] is False
    assert runner._dashboard_asset_runtime["SOL"]["learned_exec_heuristic_only"] is True
    assert runner._dashboard_asset_runtime["SOL"]["execution_advisory_guard_side"] == "short"
    assert runner.execution_manager.calls[0]["order_type"] == "LIMIT"
    assert intent.execution_urgency < 1.3
    assert intent.prefer_limit_order is True


@pytest.mark.asyncio
async def test_execution_advisory_guard_blocks_when_fill_telemetry_is_unverified():
    runner = _make_runner(session_fill_count=2)
    runner._fill_slope_monitor = FakeFillSlopeMonitor(fill_count=0, telemetry_verified=False)
    runner._record_realized_pnl_breakdown(
        asset="SOL",
        side="short",
        strategy="momentum",
        pnl_usd=12.0,
        pnl_bps=18.0,
        exit_type="FULL",
    )
    intent = _make_intent()

    result = await runner._execute_intent("SOL", intent, _make_market_data(), agent_signals={})

    assert result["status"] == "FILLED"
    assert runner._dashboard_asset_runtime["SOL"]["execution_advisory_guard_open"] is False
    assert runner._dashboard_asset_runtime["SOL"]["execution_advisory_evidence_quality_open"] is False
    assert "EVIDENCE_QUALITY_UNVERIFIED_FILL_TELEMETRY_0/2" in runner._dashboard_asset_runtime["SOL"]["execution_advisory_guard_reason"]
    assert runner._dashboard_asset_runtime["SOL"]["execution_advisory_fill_telemetry_verified"] is False


@pytest.mark.asyncio
async def test_addon_block_is_enforced_before_order_execution():
    runner = _make_runner(session_fill_count=2)
    intent = _make_intent()
    intent.target_exposure = 0.15
    intent.allow_add = False

    result = await runner._execute_intent("SOL", intent, _make_market_data(), agent_signals={})

    assert result["status"] == "skipped"
    assert result["reason"] == "ADDON_BLOCKED"
    assert runner.execution_manager.calls == []

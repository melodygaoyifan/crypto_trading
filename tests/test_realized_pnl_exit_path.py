from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import main as main_mod
from main import HMATSProductionRunner, RunMode


class FakeAccountSync:
    def __init__(self, equity: float = 10_000.0):
        self.dry_run = True
        self._base_equity = equity
        self._dry_run_pnl = 0.0
        self.pnl_updates = []

    async def refresh(self):
        return None

    def get_equity(self) -> float:
        return self._base_equity + self._dry_run_pnl

    def update_dry_run_pnl(self, pnl_usd: float):
        self._dry_run_pnl += float(pnl_usd)
        self.pnl_updates.append(float(pnl_usd))


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
        self.order_id = "test_order_1"

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
        self.fills = []
        self._fills_today = 0
        self._fills_date = ""

    def record_fill(self, asset: str, tick_count: int):
        self.fills.append((asset, tick_count))


class FakeWarmupTracker:
    threshold = 2

    def get_tick_count(self, asset: str) -> int:
        return 0


def _make_partial_exit_runner():
    now = datetime.now(timezone.utc)

    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(
        mode=RunMode.PAPER,
        initial_capital=10_000.0,
        post_leverage_caps={"SOL": 0.25},
    )
    runner.account_sync = FakeAccountSync()
    runner.risk_manager = FakeRiskManager()
    runner.execution_manager = FakeExecutionManager(filled_price=90.0)
    runner.p0_integrator = FakeP0Integrator()
    runner._anti_churn = FakeAntiChurn()
    runner._warmup_tracker = FakeWarmupTracker()
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
    runner._rebuild_cooldown = {}
    runner._exit_trigger_tag = {}
    runner._orphaned_stops = set()
    runner._dashboard_asset_runtime = {}
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
        "_fill_slope_monitor",
        "pa_executor",
        "_composite_toxicity",
        "learned_exec_policy",
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


def _make_fee_enabled_runner():
    runner = _make_partial_exit_runner()
    runner._fee_blending_enabled = True
    runner.engine = SimpleNamespace(
        _last_timing_score=None,
        guarantees=SimpleNamespace(
            alpha_calculator=SimpleNamespace(
                FRICTION=SimpleNamespace(
                    MARGIN_FEE_BPS={"SOL": 2.0, "BTC": 2.0, "ETH": 2.0}
                )
            )
        ),
    )
    return runner


def test_build_realized_outcome_allocates_entry_fee_and_funding():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)

    realized = runner._build_realized_outcome(
        position={
            "notional": 1000.0,
            "cumulative_entry_fees_usd": 10.0,
        },
        gross_pnl_usd=50.0,
        closed_notional_usd=500.0,
        exit_fee_usd=1.0,
        funding_pnl_usd=2.0,
    )

    assert realized["close_fraction"] == pytest.approx(0.5)
    assert realized["entry_fee_allocated_usd"] == pytest.approx(5.0)
    assert realized["remaining_entry_fee_usd"] == pytest.approx(5.0)
    assert realized["gross_pnl_bps"] == pytest.approx(1000.0)
    assert realized["net_pnl_usd"] == pytest.approx(46.0)
    assert realized["net_pnl_bps"] == pytest.approx(920.0)


def test_split_trade_fee_usd_allocates_close_and_open_notional():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)

    split = runner._split_trade_fee_usd(
        trade_fee_usd=2.6,
        total_notional_usd=1300.0,
        close_notional_usd=1000.0,
    )

    assert split["close_fraction"] == pytest.approx(1000.0 / 1300.0)
    assert split["close_fee_usd"] == pytest.approx(2.0)
    assert split["open_fee_usd"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_execute_intent_short_entry_charges_margin_open_fee_and_records_total_fee(monkeypatch):
    runner = _make_fee_enabled_runner()
    runner._paper_positions = {}
    runner._position_entry_times = {}
    runner._asset_trade_pnls = {"SOL": []}

    captured = {}

    def _fake_on_trade_fill(**kwargs):
        captured.update(kwargs)
        return {
            "fee_usd": 1.6,
            "fee_effective": kwargs["fee_std"],
            "fee_std": kwargs["fee_std"],
            "weight": 1.0,
            "in_free_tier": False,
            "in_blend_zone": False,
        }

    monkeypatch.setattr(main_mod, "on_trade_fill", _fake_on_trade_fill)

    intent = SimpleNamespace(
        direction=-1.0,
        target_exposure=0.10,
        execution_mode=SimpleNamespace(value="PASSIVE_PREFERRED"),
        execution_urgency=0.5,
        prefer_limit_order=True,
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
    market_data = {
        "current_price": 100.0,
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

    result = await runner._execute_intent("SOL", intent, market_data, agent_signals={})

    assert result["status"] == "FILLED"
    assert captured["is_spot"] is False
    assert runner._paper_positions["SOL"]["entry_trade_fee_usd"] == pytest.approx(1.6)
    assert runner._paper_positions["SOL"]["entry_margin_opening_fee_usd"] == pytest.approx(0.2)
    assert runner._paper_positions["SOL"]["entry_fee_usd"] == pytest.approx(1.8)
    assert runner.account_sync.pnl_updates == [pytest.approx(-1.8)]
    assert result["paper_fee_cost_pnl_delta"] == pytest.approx(-1.8)

    assert len(runner.p0_integrator.fills) == 1
    shadow_fill = runner.p0_integrator.fills[0]
    assert shadow_fill["fee"] == pytest.approx(1.8)
    assert shadow_fill["extra"]["trade_fee_usd"] == pytest.approx(1.6)
    assert shadow_fill["extra"]["margin_opening_fee_usd"] == pytest.approx(0.2)
    assert runner._paper_positions["SOL"]["notional"] == pytest.approx(1000.0)
    expected_exposure = runner._paper_positions["SOL"]["notional"] / runner.account_sync._base_equity
    assert runner._paper_positions["SOL"]["exposure"] == pytest.approx(expected_exposure)
    assert runner._paper_positions["SOL"]["original_entry_exposure"] == pytest.approx(expected_exposure)


@pytest.mark.asyncio
async def test_execute_intent_partial_exit_updates_realized_pnl_and_remaining_position():
    runner = _make_partial_exit_runner()
    intent = SimpleNamespace(
        direction=-1.0,
        target_exposure=0.05,
        execution_mode=SimpleNamespace(value="PASSIVE_PREFERRED"),
        execution_urgency=0.5,
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
    market_data = {
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

    result = await runner._execute_intent("SOL", intent, market_data, agent_signals={})

    assert result["status"] == "FILLED"
    assert result["shadow_fill_recorded"] is True
    assert result["shadow_fill_count_delta"] == 1

    remaining = runner._paper_positions["SOL"]
    assert remaining["entry_price"] == pytest.approx(100.0)
    assert remaining["notional"] == pytest.approx(495.0)
    assert remaining["exposure"] == pytest.approx(495.0 / 10000.0)
    assert remaining["cumulative_funding_pnl"] == pytest.approx(1.98)
    assert remaining["entry_fee_usd"] == pytest.approx(4.95)
    assert remaining["cumulative_entry_fees_usd"] == pytest.approx(4.95)

    assert runner.account_sync.pnl_updates == [pytest.approx(47.47)]
    assert runner.account_sync.get_equity() == pytest.approx(10047.47)
    assert runner.risk_manager.realized_pnls == [pytest.approx(47.47)]
    assert runner.risk_manager.balance_updates[-1] == pytest.approx(10047.47)
    assert runner._asset_trade_pnls["SOL"] == [pytest.approx(47.47)]
    assert runner._get_realized_pnl_summary()["by_side"]["short"]["total_pnl_usd"] == pytest.approx(47.47)
    assert runner._anti_churn.fills == [("SOL", 5)]

    assert len(runner.p0_integrator.fills) == 1
    shadow_fill = runner.p0_integrator.fills[0]
    assert shadow_fill["asset"] == "SOL"
    assert shadow_fill["side"] == "BUY"
    assert shadow_fill["realized_pnl"] == pytest.approx(47.47)


@pytest.mark.asyncio
async def test_execute_intent_full_exit_with_fee_model_does_not_double_count_exit_fee(monkeypatch):
    runner = _make_fee_enabled_runner()

    captured = {}

    def _fake_on_trade_fill(**kwargs):
        captured.update(kwargs)
        return {
            "fee_usd": 1.0,
            "fee_effective": kwargs["fee_std"],
            "fee_std": kwargs["fee_std"],
            "weight": 1.0,
            "in_free_tier": False,
            "in_blend_zone": False,
        }

    monkeypatch.setattr(main_mod, "on_trade_fill", _fake_on_trade_fill)

    intent = SimpleNamespace(
        direction=0.0,
        target_exposure=0.0,
        execution_mode=SimpleNamespace(value="PASSIVE_PREFERRED"),
        execution_urgency=0.5,
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
        force_execution=True,
    )
    market_data = {
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

    result = await runner._execute_intent("SOL", intent, market_data, agent_signals={})

    assert result["status"] == "FILLED"
    assert captured["is_spot"] is False
    assert runner.account_sync.pnl_updates == [pytest.approx(93.0)]
    assert runner.account_sync.get_equity() == pytest.approx(10093.0)
    assert runner.risk_manager.realized_pnls == [pytest.approx(93.0)]
    assert result["paper_fee_cost_pnl_delta"] == pytest.approx(0.0)

    assert len(runner.p0_integrator.fills) == 1
    shadow_fill = runner.p0_integrator.fills[0]
    assert shadow_fill["fee"] == pytest.approx(1.0)
    assert shadow_fill["extra"]["trade_fee_usd"] == pytest.approx(1.0)
    assert shadow_fill["extra"]["margin_opening_fee_usd"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_execute_intent_full_exit_uses_existing_position_size_when_direction_is_flat():
    runner = _make_partial_exit_runner()
    intent = SimpleNamespace(
        direction=0.0,
        target_exposure=0.0,
        execution_mode=SimpleNamespace(value="PASSIVE_PREFERRED"),
        execution_urgency=0.5,
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
        force_execution=True,
    )
    market_data = {
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

    result = await runner._execute_intent("SOL", intent, market_data, agent_signals={})

    assert result["status"] == "FILLED"
    assert result["shadow_fill_recorded"] is True
    assert result["p0_details"]["base_quantity"] == pytest.approx(1000.0 / 90.0)
    assert runner.execution_manager.calls[0]["side"] == "BUY"
    assert "SOL" not in runner._paper_positions
    assert runner.account_sync.pnl_updates == [pytest.approx(94.0)]
    assert runner.account_sync.get_equity() == pytest.approx(10094.0)
    assert runner.risk_manager.realized_pnls == [pytest.approx(94.0)]
    assert runner._asset_trade_pnls["SOL"] == [pytest.approx(94.0)]
    assert runner._get_realized_pnl_summary()["by_side"]["short"]["count"] == 1

    assert len(runner.p0_integrator.fills) == 1
    shadow_fill = runner.p0_integrator.fills[0]
    assert shadow_fill["asset"] == "SOL"
    assert shadow_fill["side"] == "BUY"
    assert shadow_fill["realized_pnl"] == pytest.approx(94.0)

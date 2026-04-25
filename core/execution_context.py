"""
================================================================================
EXECUTION CONTEXT - Dependency Container for _execute_intent Extraction
================================================================================
Version: 1.0.0
Purpose: Bundle all ~93 self.* references used by _execute_intent into a single
         container, enabling extraction from the God Object without losing access
         to shared mutable state.

DESIGN:
    - Holds REFERENCES to mutable objects (dicts, lists, components), not copies
    - Components that are None = gracefully degraded (same as main.py behavior)
    - build_from_runner() is the only constructor — called once per tick or once at init
    - Methods on this class are PURE ACCESSORS, no business logic

USAGE:
    ctx = ExecutionContext.build_from_runner(self)
    result = await execute_intent(ctx, asset, intent, market_data)

MIGRATION PATH:
    1. Create ExecutionContext (this file)
    2. Extract _execute_intent to core/execution_service.py using ctx
    3. Shadow mode: run both paths, compare results for 48h
    4. Remove old _execute_intent, keep only ctx-based version
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """
    Dependency container for the execution pipeline.

    Groups all self.* references from _execute_intent by functional role.
    All fields hold live references — mutations propagate back to the runner.
    """

    # ── Config ──
    config: Any = None                  # ProductionConfig
    run_mode: Any = None                # RunMode enum value

    # ── Core Components (read-only references) ──
    engine: Any = None                  # HMATSv36Engine
    execution_manager: Any = None       # ExecutionManager (Kraken order routing)
    account_sync: Any = None            # AccountSyncManager (equity provider)
    risk_manager: Any = None            # RiskManager
    authority_chain: Any = None         # AuthorityChain (one-veto-kill)
    leverage_guard: Any = None          # LeverageGuard
    unified_sizer: Any = None           # UnifiedPositionSizer
    p0_integrator: Any = None           # P0SafetyIntegrator
    dead_man_switch: Any = None         # DeadManSwitch (Kraken CancelAllOrdersAfter)

    # ── Execution Components ──
    execution_guard: Any = None         # ExecutionGuard (pre-execution safety)
    pa_executor: Any = None             # PassiveAggressiveExecutor
    rl_exec_agent: Any = None           # RLExecutionAgent
    learned_exec_policy: Any = None     # LearnedExecutionPolicy (heuristic)
    orderbook_analyzer: Any = None      # Level2OrderBookAnalyzer
    level2_analyzer: Any = None         # Same or alias
    impact_cal_table: Any = None        # MarketImpactCalibrationTable
    integrity_shield: Any = None        # KrakenIntegrityShield

    # ── Risk / Exit Components ──
    adaptive_stop: Any = None           # AdaptiveStopManager
    adaptive_stop_regime_mult: float = 1.0
    exit_alpha: Any = None              # ExitAlphaManager
    gambler_exit: Any = None            # GamblerExitChecker
    stop_authority: Any = None          # StopLossAuthorityManager
    existence_fuse: Any = None          # StrategyExistenceFuse
    thesis_budget_governor: Any = None  # ThesisBudgetGovernor
    opportunity_budget: Any = None      # OpportunityBudgetGovernor
    composite_toxicity: Any = None      # CompositeToxicityFilter
    hplv_filter: Any = None             # HighPositionLowVolumeFilter
    sol_exec_guard: Any = None          # SOLExecutionGuard
    impact_model: Any = None            # AdaptiveMarketImpactModel

    # ── Analytics / Feedback Components ──
    audit_manager: Any = None           # PersistenceAndAlertingManager
    feedback_loop: Any = None           # FeedbackLoop
    pnl_attribution: Any = None         # PnLAttributionManager
    strategy_aging: Any = None          # StrategyAgingManager
    failure_memory: Any = None          # FailureAwareMetaMemory
    confidence_scorer: Any = None       # ConfidenceScorer
    trade_attributor: Any = None        # TradeAttributor
    strategic_coordinator: Any = None   # [FIX 2026-04-24] For v521 adaptive weight feedback
    sq_tracker: Any = None              # SignalQualityTracker
    ea_tracker: Any = None              # ExitAlphaTracker
    exec_quality_logger: Any = None     # ExecutionQualityLogger
    fill_rate_kpi: Any = None           # FillRateKPI
    fill_slope_monitor: Any = None      # FillSlopeMonitor
    meta_decision: Any = None           # MetaDecision
    experience_buffer: Any = None       # LiveExperienceBuffer
    alpha_tilt: Any = None              # AssetAlphaTilt
    margin_tracker: Any = None          # MarginCostTracker
    sota_integration: Any = None        # SOTAIntegration

    # ── DRL State ──
    drl_authority_level: str = "DISABLED"
    drl_models_ready: int = 0
    promotion_gate: Any = None          # DRLPromotionGate

    # ── Mutable State (live references — mutations propagate) ──
    paper_positions: Dict[str, Dict] = field(default_factory=dict)
    position_entry_times: Dict[str, Dict] = field(default_factory=dict)
    rebuild_cooldown: Dict[str, Any] = field(default_factory=dict)
    exit_trigger_tag: Dict[str, str] = field(default_factory=dict)
    dashboard_asset_runtime: Dict[str, Dict] = field(default_factory=dict)
    asset_trade_pnls: Dict[str, Any] = field(default_factory=dict)
    orphaned_stops: Set[str] = field(default_factory=set)
    recent_fill_state: Dict[str, Dict] = field(default_factory=dict)
    confidence_signal_times: Dict[str, float] = field(default_factory=dict)

    # ── Runner back-ref ──
    # Only write-back channel for scalars that can't be routed through a
    # dedicated component (see `_last_aging_check` in execute_intent_v2).
    # Populated by build_from_runner; stays None if ctx is built elsewhere.
    _runner_ref: Any = None

    # ── Scalar State ──
    tick_count: int = 0
    # last_aging_check retired 2026-04-24 — the canonical scalar lives on the
    # runner (`_last_aging_check`) and is read/written via `ctx._runner_ref`
    # inside the strategy-aging branch of execute_intent_v2. sync_scalars_back
    # was never called so the old ctx scalar drifted every tick.
    drawdown_tracker: Any = None
    current_drawdown_pct: float = 0.0
    dynamic_limits_result: Any = None
    warmup_tracker: Any = None
    anti_churn: Any = None

    # ── Anti-churn config (backward compat scalars) ──
    REBUILD_COOLDOWN_TICKS: int = 2
    AC1_MIN_HOLD_TICKS: int = 1
    AC2_MAX_FILLS_PER_ASSET: int = 2
    AC2_MAX_FILLS_GLOBAL: int = 6
    AC2_WINDOW_TICKS: int = 6
    AC5_MAX_FILLS_PER_DAY: int = 8
    ac2_fill_ticks: Any = None
    # ac5_fills_today / ac5_fills_date removed 2026-04-24: they shadowed
    # runner._ac5_fills_today which was never incremented (canonical counter
    # lives on anti_churn._fills_today). See core/execution_service.py AC-5
    # gate and core/anti_churn.py check_fill_budget() for the live path.

    # ── Bound Method References ──
    # These are methods from HMATSProductionRunner that the execution service
    # needs to call. Stored as callables to avoid circular imports.
    fn_is_active_paper_position: Optional[Callable] = None
    fn_normalize_runtime_position_state: Optional[Callable] = None
    fn_get_effective_position_state: Optional[Callable] = None
    fn_normalize_existing_hold_intent: Optional[Callable] = None
    fn_maybe_block_micro_rebalance: Optional[Callable] = None
    fn_build_realized_outcome: Optional[Callable] = None
    fn_build_execution_fee_result: Optional[Callable] = None
    fn_compute_margin_opening_fee_usd: Optional[Callable] = None
    fn_split_trade_fee_usd: Optional[Callable] = None
    fn_get_position_entry_fee_usd: Optional[Callable] = None
    fn_normalize_kraken_pair: Optional[Callable] = None
    fn_get_drl_weight: Optional[Callable] = None
    fn_get_fast_market_execution_decision: Optional[Callable] = None
    fn_get_execution_advisory_guard: Optional[Callable] = None
    fn_get_learned_exec_runtime_snapshot: Optional[Callable] = None
    fn_get_ac0_entry_block_reason: Optional[Callable] = None
    fn_resolve_execution_trade_side: Optional[Callable] = None
    fn_record_realized_pnl_breakdown: Optional[Callable] = None
    fn_persist_tranche_state: Optional[Callable] = None
    fn_save_paper_positions: Optional[Callable] = None
    fn_sync_drl_authority: Optional[Callable] = None
    fn_prepare_market_data: Optional[Callable] = None

    @classmethod
    def build_from_runner(cls, runner: Any) -> "ExecutionContext":
        """
        Build ExecutionContext from a HMATSProductionRunner instance.

        This is the ONLY constructor. Captures live references to mutable state
        so mutations in the execution service propagate back to the runner.
        """
        ctx = cls()

        # Live runner ref — only for writing back scalar mutations that can't
        # be piped through a dedicated component (e.g. `_last_aging_check`).
        # Prefer `fn_*` callbacks or direct component mutation where possible;
        # this is a small escape hatch for true god-object scalars.
        ctx._runner_ref = runner

        # Config
        ctx.config = runner.config
        ctx.run_mode = runner.config.mode

        # Core components
        ctx.engine = getattr(runner, 'engine', None)
        ctx.execution_manager = getattr(runner, 'execution_manager', None)
        ctx.account_sync = getattr(runner, 'account_sync', None)
        ctx.risk_manager = getattr(runner, 'risk_manager', None)
        ctx.authority_chain = getattr(runner, 'authority_chain', None)
        ctx.leverage_guard = getattr(runner, 'leverage_guard', None)
        ctx.unified_sizer = getattr(runner, 'unified_sizer', None)
        ctx.p0_integrator = getattr(runner, 'p0_integrator', None)
        # [FIX 2026-04-24] Wire strategic_coordinator so execution_service can
        # feed realized PnL into v521 AdaptiveWeightManager (record_trade_completed).
        # Without this, 915-line Sharpe/Calmar/WinRate adaptive weight system is
        # a no-op (no trade data -> all strategies stay at neutral weight 1.0).
        ctx.strategic_coordinator = getattr(runner, 'strategic_coordinator', None)
        ctx.dead_man_switch = getattr(runner, 'dead_man_switch', None)

        # Execution components
        ctx.execution_guard = getattr(runner, 'execution_guard', None)
        ctx.pa_executor = getattr(runner, 'pa_executor', None)
        ctx.rl_exec_agent = getattr(runner, 'rl_exec_agent', None)
        ctx.learned_exec_policy = getattr(runner, 'learned_exec_policy', None)
        ctx.orderbook_analyzer = getattr(runner, 'orderbook_analyzer', None)
        ctx.level2_analyzer = getattr(runner, 'level2_analyzer', None)
        ctx.impact_cal_table = getattr(runner, 'impact_cal_table', None)
        ctx.integrity_shield = getattr(runner, 'integrity_shield', None)

        # Risk / Exit components
        ctx.adaptive_stop = getattr(runner, '_adaptive_stop', None)
        ctx.adaptive_stop_regime_mult = getattr(runner, '_adaptive_stop_regime_mult', 1.0)
        ctx.exit_alpha = getattr(runner, 'exit_alpha', None)
        ctx.gambler_exit = getattr(runner, 'gambler_exit', None)
        ctx.stop_authority = getattr(runner, 'stop_authority', None)
        ctx.existence_fuse = getattr(runner, 'existence_fuse', None)
        ctx.thesis_budget_governor = getattr(runner, 'thesis_budget_governor', None)
        ctx.opportunity_budget = getattr(runner, 'opportunity_budget', None)
        ctx.composite_toxicity = getattr(runner, '_composite_toxicity', None)
        ctx.hplv_filter = getattr(runner, '_hplv_filter', None)
        ctx.sol_exec_guard = getattr(runner, '_sol_exec_guard', None)
        ctx.impact_model = getattr(runner, '_impact_model', None)

        # Analytics / Feedback
        ctx.audit_manager = getattr(runner, 'audit_manager', None)
        ctx.feedback_loop = getattr(runner, 'feedback_loop', None)
        ctx.pnl_attribution = getattr(runner, '_pnl_attribution', None)
        ctx.strategy_aging = getattr(runner, '_strategy_aging', None)
        ctx.failure_memory = getattr(runner, '_failure_memory', None)
        ctx.confidence_scorer = getattr(runner, '_confidence_scorer', None)
        ctx.trade_attributor = getattr(runner, '_trade_attributor', None)
        ctx.sq_tracker = getattr(runner, '_sq_tracker', None)
        ctx.ea_tracker = getattr(runner, '_ea_tracker', None)
        ctx.exec_quality_logger = getattr(runner, 'exec_quality_logger', None)
        ctx.fill_rate_kpi = getattr(runner, 'fill_rate_kpi', None)
        ctx.fill_slope_monitor = getattr(runner, '_fill_slope_monitor', None)
        ctx.meta_decision = getattr(runner, '_meta_decision', None)
        ctx.experience_buffer = getattr(runner, '_experience_buffer', None)
        ctx.alpha_tilt = getattr(runner, '_alpha_tilt', None)
        ctx.margin_tracker = getattr(runner, '_margin_tracker', None)
        ctx.sota_integration = getattr(runner, 'sota_integration', None)

        # DRL state
        ctx.drl_authority_level = getattr(runner, '_drl_authority_level', "DISABLED")
        ctx.drl_models_ready = getattr(runner, '_drl_models_ready', 0)
        ctx.promotion_gate = getattr(runner, '_promotion_gate', None)

        # Mutable state — LIVE REFERENCES (mutations propagate back)
        ctx.paper_positions = runner._paper_positions
        ctx.position_entry_times = runner._position_entry_times
        ctx.rebuild_cooldown = runner._rebuild_cooldown
        ctx.exit_trigger_tag = getattr(runner, '_exit_trigger_tag', {})
        ctx.dashboard_asset_runtime = runner._dashboard_asset_runtime
        ctx.asset_trade_pnls = runner._asset_trade_pnls
        ctx.orphaned_stops = runner._orphaned_stops
        ctx.recent_fill_state = getattr(runner, '_recent_fill_state', {})
        ctx.confidence_signal_times = getattr(runner, '_confidence_signal_times', {})

        # Scalar state
        ctx.tick_count = runner._tick_count
        # last_aging_check retired — read/write directly against ctx._runner_ref.
        ctx.drawdown_tracker = getattr(runner, '_drawdown_tracker', None)
        ctx.current_drawdown_pct = getattr(runner, '_current_drawdown_pct', 0.0)
        ctx.dynamic_limits_result = getattr(runner, '_dynamic_limits_result', None)
        ctx.warmup_tracker = runner._warmup_tracker
        ctx.anti_churn = runner._anti_churn

        # Anti-churn config
        ctx.REBUILD_COOLDOWN_TICKS = runner._REBUILD_COOLDOWN_TICKS
        ctx.AC1_MIN_HOLD_TICKS = runner._AC1_MIN_HOLD_TICKS
        ctx.AC2_MAX_FILLS_PER_ASSET = runner._AC2_MAX_FILLS_PER_ASSET_PER_DAY
        ctx.AC2_MAX_FILLS_GLOBAL = runner._AC2_MAX_FILLS_GLOBAL_PER_DAY
        ctx.AC2_WINDOW_TICKS = runner._AC2_WINDOW_TICKS
        ctx.AC5_MAX_FILLS_PER_DAY = runner._AC5_MAX_FILLS_PER_DAY
        ctx.ac2_fill_ticks = runner._ac2_fill_ticks
        # ac5_* ctx scalars retired — read ctx.anti_churn._fills_today directly.

        # Bound method references
        ctx.fn_is_active_paper_position = runner._is_active_paper_position
        ctx.fn_normalize_runtime_position_state = runner._normalize_runtime_position_state
        ctx.fn_get_effective_position_state = runner._get_effective_position_state
        ctx.fn_normalize_existing_hold_intent = runner._normalize_existing_hold_intent
        ctx.fn_maybe_block_micro_rebalance = runner._maybe_block_micro_rebalance
        ctx.fn_build_realized_outcome = runner._build_realized_outcome
        ctx.fn_build_execution_fee_result = runner._build_execution_fee_result
        ctx.fn_compute_margin_opening_fee_usd = runner._compute_margin_opening_fee_usd
        ctx.fn_split_trade_fee_usd = runner._split_trade_fee_usd
        ctx.fn_get_position_entry_fee_usd = getattr(runner, '_get_position_entry_fee_usd', None)
        ctx.fn_normalize_kraken_pair = getattr(runner, '_normalize_kraken_pair', None)
        ctx.fn_get_drl_weight = getattr(runner, '_get_drl_weight', None)
        ctx.fn_get_fast_market_execution_decision = runner._get_fast_market_execution_decision
        ctx.fn_get_execution_advisory_guard = runner._get_execution_advisory_guard
        ctx.fn_get_learned_exec_runtime_snapshot = runner._get_learned_exec_runtime_snapshot
        ctx.fn_get_ac0_entry_block_reason = runner._get_ac0_entry_block_reason
        ctx.fn_resolve_execution_trade_side = runner._resolve_execution_trade_side
        ctx.fn_record_realized_pnl_breakdown = runner._record_realized_pnl_breakdown
        ctx.fn_persist_tranche_state = runner._persist_tranche_state
        ctx.fn_save_paper_positions = runner._save_paper_positions
        ctx.fn_sync_drl_authority = getattr(runner, '_sync_drl_authority', None)
        ctx.fn_prepare_market_data = getattr(runner, '_prepare_market_data', None)

        return ctx

    # sync_scalars_back() removed 2026-04-24: it was never called, and
    # attempting to add callers retroactively would silently back-write
    # whatever default value sat on the dataclass for fields that the
    # execution path never mutated, corrupting runner state. Scalar
    # writes now happen through explicit callbacks (fn_sync_drl_authority)
    # or via ctx._runner_ref (_last_aging_check).

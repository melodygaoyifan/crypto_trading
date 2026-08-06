"""
_DEPRECATED (T29): Alternative canonical main loop, NOT used.
Actual tick processing lives in main.py._process_4h_tick_inner().
WARNING: Contains hardcoded 10% drawdown threshold that conflicts with ULTRA profile.

================================================================================
HMATS v5.1.0 - RUNTIME SPINE
================================================================================
Version: 5.1.0-HARDENED
Purpose: Single canonical runtime loop

CANONICAL FLOW (LOCKED):
    1. Data Provider (Kraken REST + WebSocket)
    2. Regime Detection (Market state classification)
    3. Agent Signals (Quant, DRL, Risk)
    4. Authority Fusion (Signal combination with weights)
    5. Execution (Kraken-only, with timing)
    6. Risk Governor (Runtime veto authority)
    7. Feedback Loop (PnL attribution, strategy aging)

DESIGN PRINCIPLES:
    - No parallel orphan pipelines
    - Single source of truth for system state
    - All decisions flow through this spine
    - Risk governor has veto authority over all execution

================================================================================
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# SPINE ENUMS
# =============================================================================

class SystemMode(Enum):
    """System operating mode - mode hierarchy is LOCKED."""
    NO_TRADE = "NO_TRADE"        # Highest priority - no trading allowed
    OPPORTUNITY = "OPPORTUNITY"  # Medium priority - aggressive on opportunities
    NORMAL = "NORMAL"            # Lowest priority - standard operation


class RegimePhase(Enum):
    """Regime phase within a trend."""
    UNKNOWN = "UNKNOWN"
    IGNITION = "IGNITION"
    EXPANSION = "EXPANSION"
    SATURATION = "SATURATION"
    EXHAUSTION = "EXHAUSTION"


class TickStage(Enum):
    """Stages of tick processing."""
    DATA_FETCH = auto()
    REGIME_DETECT = auto()
    SIGNAL_GENERATE = auto()
    SIGNAL_FUSE = auto()
    RISK_CHECK = auto()
    EXECUTION = auto()
    FEEDBACK = auto()
    COMPLETE = auto()


# =============================================================================
# SPINE DATA STRUCTURES
# =============================================================================

@dataclass
class SpineConfig:
    """Configuration for runtime spine."""
    # Timing
    master_tick_seconds: int = 14400  # 4H = 14400 seconds
    execution_loop_ms: int = 200       # 200ms execution modifiers
    
    # Assets (v3.3-HR: SOL primary)
    assets: List[str] = field(default_factory=lambda: ["SOL", "BTC", "ETH"])
    primary_asset: str = "SOL"
    
    # Risk (v3.3-HR philosophy - NON-NEGOTIABLE)
    hard_drawdown_halt: float = 0.20   # 20% halt [UNLEASH UL-5]
    correlation_crisis: float = 0.98   # [UNLEASH UL-4]
    
    # Mode multipliers
    opportunity_multiplier: float = 0.70
    normal_multiplier: float = 0.50
    
    # Exchange
    exchange: str = "kraken"
    
    # Logging
    enable_proof_logs: bool = True
    
    @classmethod
    def default(cls) -> 'SpineConfig':
        """Get default configuration."""
        return cls()


@dataclass
class MarketSnapshot:
    """Snapshot of market data at a point in time."""
    timestamp: datetime
    asset: str
    current_price: float = 0.0
    volume_24h: float = 0.0
    data_age_seconds: float = 0.0
    
    # OHLCV
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    
    # Orderbook
    bid_depth_1pct: float = 0.0
    ask_depth_1pct: float = 0.0
    spread_bps: float = 0.0
    
    # Derived
    vpin: float = 0.5
    
    def is_valid(self) -> bool:
        """Check if data is valid for trading."""
        return (
            self.current_price > 0 and
            self.data_age_seconds < 10.0 and
            self.volume_24h > 0
        )


@dataclass
class RegimeState:
    """Current regime state."""
    timestamp: datetime
    regime: str = "UNKNOWN"
    phase: RegimePhase = RegimePhase.UNKNOWN
    direction: float = 0.0       # -1 to 1
    confidence: float = 0.0      # 0 to 1
    volatility: float = 0.0      # ATR-based
    persistence_bars: int = 0    # How long in current regime
    
    # Cross-asset
    correlation_btc_eth_sol: float = 0.0
    beta_decoupling: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentSignal:
    """Signal from an agent."""
    agent_id: str
    timestamp: datetime
    asset: str
    direction: float = 0.0       # -1 to 1
    confidence: float = 0.0      # 0 to 1
    urgency: float = 0.5         # 0 to 1
    strategy_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedSignal:
    """Result of signal fusion."""
    timestamp: datetime
    asset: str
    direction: float = 0.0
    confidence: float = 0.0
    urgency: float = 0.5
    target_exposure: float = 0.0
    execution_mode: str = "PASSIVE_PREFERRED"
    
    # Component breakdown
    quant_weight: float = 0.0
    drl_weight: float = 0.0
    quant_signal: float = 0.0
    drl_signal: float = 0.0
    
    # Authority
    authority_source: str = ""
    deadlock_resolution: str = "NONE"


@dataclass
class ExecutionIntent:
    """Intent to execute a trade."""
    timestamp: datetime
    asset: str
    direction: float
    target_exposure: float
    
    # Sizing
    size_usd: float = 0.0
    tranche_level: int = 0
    
    # Execution parameters
    execution_mode: str = "PASSIVE_PREFERRED"
    urgency: float = 0.5
    max_slippage_bps: float = 10.0
    
    # Risk limits
    max_gross: float = 1.50
    max_asset: float = 0.80
    
    # SOL dominance
    sol_dominance_active: bool = False
    sol_dominance_ttl: int = 0
    
    # Veto tracking
    veto_active: bool = False
    veto_reason: str = ""


@dataclass
class ExecutionResult:
    """Result of execution attempt."""
    timestamp: datetime
    asset: str
    success: bool = False
    
    # Order details
    order_id: str = ""
    side: str = ""
    requested_size: float = 0.0
    filled_size: float = 0.0
    requested_price: float = 0.0
    filled_price: float = 0.0
    slippage_bps: float = 0.0
    
    # Status
    status: str = ""
    error: str = ""
    
    # Position state
    new_exposure: float = 0.0
    pnl_realized: float = 0.0


# =============================================================================
# H1 FIX: EXECUTION ACCEPTANCE (竞态修复)
# =============================================================================

@dataclass
class ExecutionAcceptance:
    """
    H1 FIX: 显式的执行层接受/拒绝状态
    
    解决问题: 上层生成intent，但执行层处于WAIT/PASSIVE时的竞态
    
    行为约束 (Hard Invariant):
    - 当 accepted=False 时，上层不得新建仓位或加仓
    - 拒绝必须被显式记录，不允许 silent skip
    """
    accepted: bool
    reason: str                    # e.g. "EXEC_WAIT", "EXTREME_SPREAD", "FALLBACK_ONLY", "FUSED"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 详细信息
    rejection_source: str = ""     # 哪个组件拒绝的
    spread_bps: float = 0.0        # 当时的 spread
    execution_mode: str = ""       # 当时的执行模式
    
    def to_log_string(self) -> str:
        """生成可审计的日志字符串"""
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return (
            f"[INTENT_{status}_BY_EXECUTION] "
            f"reason={self.reason} source={self.rejection_source} "
            f"spread={self.spread_bps:.1f}bps mode={self.execution_mode}"
        )


@dataclass
class TickResult:
    """Complete result of a tick cycle."""
    tick_id: str
    timestamp: datetime
    stage: TickStage = TickStage.COMPLETE
    
    # Mode
    system_mode: SystemMode = SystemMode.NORMAL
    
    # Data
    market_data: Dict[str, MarketSnapshot] = field(default_factory=dict)
    
    # Regime
    regime_state: Optional[RegimeState] = None
    
    # Signals
    agent_signals: Dict[str, AgentSignal] = field(default_factory=dict)
    fused_signal: Optional[FusedSignal] = None
    
    # Execution
    execution_intent: Optional[ExecutionIntent] = None
    execution_acceptance: Optional[ExecutionAcceptance] = None  # H1 FIX
    execution_result: Optional[ExecutionResult] = None
    
    # Risk
    risk_veto: bool = False
    risk_veto_reason: str = ""
    
    # Feedback
    pnl_attribution: Dict[str, float] = field(default_factory=dict)
    
    # M3 FIX: Decision summary
    decision_summary: str = ""
    blocked_by: str = ""
    secondary_caps: List[str] = field(default_factory=list)
    
    # Proof log
    proof_log: str = ""
    
    def to_proof_log(self) -> str:
        """Generate proof log line."""
        mode = self.system_mode.value
        regime = self.regime_state.regime if self.regime_state else "UNKNOWN"
        phase = self.regime_state.phase.value if self.regime_state else "UNKNOWN"
        direction = self.fused_signal.direction if self.fused_signal else 0.0
        confidence = self.fused_signal.confidence if self.fused_signal else 0.0
        exposure = self.fused_signal.target_exposure if self.fused_signal else 0.0
        veto = "VETO" if self.risk_veto else "OK"
        
        return (
            f"[PROOF] {self.timestamp.isoformat()} | "
            f"mode={mode} | regime={regime}/{phase} | "
            f"dir={direction:.2f} conf={confidence:.2f} exp={exposure:.2f} | "
            f"risk={veto}"
        )


# =============================================================================
# RUNTIME SPINE
# =============================================================================

class RuntimeSpine:
    """
    Canonical runtime spine for HMATS v5.1.0.
    
    This is the SINGLE source of truth for system flow:
        data -> regime -> agents -> fusion -> execution -> risk -> feedback
    
    All decisions flow through this spine. No parallel pipelines.
    """
    
    def __init__(self, config: Optional[SpineConfig] = None):
        """Initialize runtime spine.

        [P50 2026-04-25] This class is DEPRECATED. The actual tick loop is in
        main.py._process_4h_tick_inner. RuntimeSpine had a hardcoded 10% DD
        halt that conflicts with ULTRA profile (35%). If accidentally
        instantiated by a test or new code, this would shadow main.py's spine
        with stale invariants. Loud-fail at instantiation; remove this guard
        if/when RuntimeSpine is genuinely revived.
        """
        import os as _os
        if _os.environ.get("HMATS_ALLOW_RUNTIME_SPINE") != "1":
            raise RuntimeError(
                "[RUNTIME_SPINE_DEPRECATED] RuntimeSpine is deprecated; live "
                "tick processing is in main.py._process_4h_tick_inner. Set "
                "HMATS_ALLOW_RUNTIME_SPINE=1 to override (e.g., for testing)."
            )
        self.config = config or SpineConfig.default()
        
        # State
        self._initialized = False
        self._running = False
        self._tick_count = 0
        self._current_mode = SystemMode.NORMAL
        self._current_regime: Optional[RegimeState] = None
        
        # Position state
        self._positions: Dict[str, float] = {}  # asset -> exposure
        self._portfolio_value: float = 100000.0
        self._current_drawdown: float = 0.0
        self._peak_value: float = 100000.0
        
        # SOL dominance tracking
        self._sol_dominance_active = False
        self._sol_dominance_ttl = 0
        
        # Components (lazy loaded)
        self._data_provider = None
        self._regime_detector = None
        self._quant_agent = None
        self._drl_agent = None
        self._fusion_engine = None
        self._execution_manager = None
        self._risk_governor = None
        self._feedback_loop = None
        
        # Event bus integration
        self._event_bus = None
        
        # Proof logs
        self._proof_logs: List[str] = []
        
        logger.info(f"RuntimeSpine initialized with config: {self.config}")
    
    def initialize(self, portfolio_value: float = 100000.0) -> bool:
        """
        Initialize all components in deterministic order.
        
        Order is critical for proper system startup:
        1. Event bus (communication backbone)
        2. Data provider (market data)
        3. Risk governor (must be ready before any decisions)
        4. Regime detector
        5. Agents (quant, drl)
        6. Fusion engine
        7. Execution manager
        8. Feedback loop
        """
        logger.info("[SPINE] Beginning initialization sequence...")
        
        self._portfolio_value = portfolio_value
        self._peak_value = portfolio_value
        
        try:
            # 1. Event bus
            logger.info("[SPINE] Step 1/8: Initializing event bus...")
            self._init_event_bus()
            
            # 2. Data provider
            logger.info("[SPINE] Step 2/8: Initializing data provider...")
            self._init_data_provider()
            
            # 3. Risk governor (MUST be ready before any decisions)
            logger.info("[SPINE] Step 3/8: Initializing risk governor...")
            self._init_risk_governor()
            
            # 4. Regime detector
            logger.info("[SPINE] Step 4/8: Initializing regime detector...")
            self._init_regime_detector()
            
            # 5. Agents
            logger.info("[SPINE] Step 5/8: Initializing agents...")
            self._init_agents()
            
            # 6. Fusion engine
            logger.info("[SPINE] Step 6/8: Initializing fusion engine...")
            self._init_fusion_engine()
            
            # 7. Execution manager
            logger.info("[SPINE] Step 7/8: Initializing execution manager...")
            self._init_execution_manager()
            
            # 8. Feedback loop
            logger.info("[SPINE] Step 8/8: Initializing feedback loop...")
            self._init_feedback_loop()
            
            self._initialized = True
            logger.info("[SPINE] ✓ Initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"[SPINE] Initialization failed: {e}")
            return False
    
    def _init_event_bus(self):
        """Initialize event bus."""
        try:
            from infra.event_bus import get_event_bus, EventBus
            self._event_bus = get_event_bus()
        except ImportError:
            logger.warning("[SPINE] Event bus not available, using stub")
            self._event_bus = None
    
    def _init_data_provider(self):
        """Initialize data provider."""
        try:
            from infra.data_provider import DataProvider
            self._data_provider = DataProvider()
        except ImportError:
            logger.warning("[SPINE] DataProvider not available, using stub")
            self._data_provider = None
    
    def _init_risk_governor(self):
        """Initialize risk governor."""
        try:
            from core.risk_governor import get_risk_governor, GovernorConfig
            config = GovernorConfig(
                hard_drawdown_halt=self.config.hard_drawdown_halt,
                correlation_crisis=self.config.correlation_crisis,
            )
            self._risk_governor = get_risk_governor(config)
        except ImportError:
            logger.warning("[SPINE] Risk governor not available, will create inline")
            self._risk_governor = None
    
    def _init_regime_detector(self):
        """Initialize regime detector."""
        try:
            from market.regime_detector import RegimeDetector
            self._regime_detector = RegimeDetector()
        except ImportError:
            logger.warning("[SPINE] RegimeDetector not available, using stub")
            self._regime_detector = None
    
    def _init_agents(self):
        """Initialize trading agents."""
        # Quant agent
        try:
            from agents.quant_agent import QuantAgent
            self._quant_agent = QuantAgent()
        except ImportError:
            logger.warning("[SPINE] QuantAgent not available, using stub")
            self._quant_agent = None
        
        # DRL agent
        try:
            from agents.drl_agent import DRLAgent
            self._drl_agent = DRLAgent()
        except ImportError:
            logger.warning("[SPINE] DRLAgent not available, using stub")
            self._drl_agent = None
    
    def _init_fusion_engine(self):
        """Initialize signal fusion engine."""
        try:
            from signals.authority_fusion import AuthorityFusionEngine
            self._fusion_engine = AuthorityFusionEngine()
        except ImportError:
            logger.warning("[SPINE] AuthorityFusionEngine not available, using stub")
            self._fusion_engine = None
    
    def _init_execution_manager(self):
        """Initialize execution manager."""
        try:
            from execution.execution_manager import ExecutionManager
            # Use dry_run by default for safety
            self._execution_manager = ExecutionManager(exchange=None, dry_run=True)
        except ImportError:
            logger.warning("[SPINE] ExecutionManager not available, using stub")
            self._execution_manager = None
    
    def _init_feedback_loop(self):
        """Initialize feedback loop."""
        try:
            from core.feedback_loop import get_feedback_loop
            self._feedback_loop = get_feedback_loop()
        except ImportError:
            logger.warning("[SPINE] FeedbackLoop not available, using stub")
            self._feedback_loop = None
    
    # =========================================================================
    # TICK PROCESSING
    # =========================================================================
    
    async def process_tick(self, tick_data: Optional[Dict] = None) -> TickResult:
        """
        Process a single tick through the canonical spine.
        
        Flow:
            1. Fetch/validate market data
            2. Detect regime
            3. Generate agent signals
            4. Fuse signals
            5. Risk governor check
            6. Execute (if approved)
            7. Feedback/attribution
        
        Args:
            tick_data: Optional pre-fetched tick data
            
        Returns:
            Complete TickResult
        """
        self._tick_count += 1
        tick_id = f"T{self._tick_count:08d}"
        timestamp = datetime.now(timezone.utc)
        
        result = TickResult(
            tick_id=tick_id,
            timestamp=timestamp,
        )
        
        try:
            # Stage 1: Data
            result.stage = TickStage.DATA_FETCH
            market_data = await self._fetch_market_data(tick_data)
            result.market_data = market_data
            
            # Check data validity
            if not self._validate_market_data(market_data):
                result.system_mode = SystemMode.NO_TRADE
                result.risk_veto = True
                result.risk_veto_reason = "Invalid market data"
                result.proof_log = result.to_proof_log()
                return result
            
            # Stage 2: Regime
            result.stage = TickStage.REGIME_DETECT
            regime_state = self._detect_regime(market_data)
            result.regime_state = regime_state
            self._current_regime = regime_state
            
            # Stage 3: System Mode
            result.system_mode = self._determine_mode(market_data, regime_state)
            self._current_mode = result.system_mode
            
            # Early exit on NO_TRADE
            if result.system_mode == SystemMode.NO_TRADE:
                result.proof_log = result.to_proof_log()
                self._add_proof_log(result.proof_log)
                return result
            
            # Stage 4: Agent Signals
            result.stage = TickStage.SIGNAL_GENERATE
            agent_signals = self._generate_agent_signals(market_data, regime_state)
            result.agent_signals = agent_signals
            
            # Stage 5: Fusion
            result.stage = TickStage.SIGNAL_FUSE
            fused_signal = self._fuse_signals(agent_signals, regime_state, result.system_mode)
            result.fused_signal = fused_signal
            
            # Stage 6: Risk Governor Check
            result.stage = TickStage.RISK_CHECK
            risk_decision = self._check_risk_governor(fused_signal, market_data)
            result.risk_veto = risk_decision.vetoed
            result.risk_veto_reason = risk_decision.reason if risk_decision.vetoed else ""
            
            # Stage 7: Execution
            result.stage = TickStage.EXECUTION
            if not result.risk_veto and fused_signal and abs(fused_signal.direction) > 0.1:
                execution_intent = self._create_execution_intent(fused_signal, result.system_mode)
                result.execution_intent = execution_intent
                
                # H1 FIX: Check execution acceptance BEFORE executing
                result.execution_acceptance = self._check_execution_acceptance(
                    execution_intent, market_data, fused_signal
                )
                
                # H1 FIX: Log acceptance/rejection explicitly
                logger.info(result.execution_acceptance.to_log_string())
                
                # H1 FIX: Only execute if accepted
                if result.execution_acceptance.accepted:
                    execution_result = await self._execute(execution_intent)
                    result.execution_result = execution_result
                else:
                    # H1 FIX: Explicit rejection - no silent skip
                    logger.warning(
                        f"[INTENT_REJECTED_BY_EXECUTION] "
                        f"intent_dir={execution_intent.direction:.2f} "
                        f"reason={result.execution_acceptance.reason} "
                        f"asset={execution_intent.asset}"
                    )
                    result.blocked_by = f"EXECUTION:{result.execution_acceptance.reason}"
            
            # Stage 8: Feedback
            result.stage = TickStage.FEEDBACK
            if result.execution_result:
                self._process_feedback(result)
            
            # M3 FIX: Generate decision summary
            result.decision_summary = self._generate_decision_summary(result)
            logger.info(result.decision_summary)
            
            # Complete
            result.stage = TickStage.COMPLETE
            result.proof_log = result.to_proof_log()
            self._add_proof_log(result.proof_log)
            
            return result
            
        except Exception as e:
            logger.error(f"[SPINE] Tick processing error at stage {result.stage}: {e}")
            result.risk_veto = True
            result.risk_veto_reason = f"Processing error: {e}"
            result.proof_log = result.to_proof_log()
            return result
    
    async def _fetch_market_data(
        self, 
        tick_data: Optional[Dict]
    ) -> Dict[str, MarketSnapshot]:
        """Fetch market data for all assets."""
        snapshots = {}
        
        for asset in self.config.assets:
            if tick_data and asset in tick_data:
                # Use provided data
                data = tick_data[asset]
                snapshots[asset] = MarketSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    asset=asset,
                    current_price=data.get("current_price", data.get("close", 0)),
                    volume_24h=data.get("volume_24h", data.get("volume", 0)),
                    data_age_seconds=data.get("data_age_seconds", 1.0),
                    open=data.get("open", 0),
                    high=data.get("high", 0),
                    low=data.get("low", 0),
                    close=data.get("close", data.get("current_price", 0)),
                    bid_depth_1pct=data.get("bid_depth_1pct", 100000),
                    ask_depth_1pct=data.get("ask_depth_1pct", 100000),
                    spread_bps=data.get("spread_bps", 5.0),
                    vpin=data.get("vpin", 0.35),
                )
            elif self._data_provider:
                # Fetch from provider
                try:
                    data = await self._data_provider.get_snapshot(asset)
                    snapshots[asset] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {asset}: {e}")
                    snapshots[asset] = self._generate_stub_snapshot(asset)
            else:
                # Use stub
                snapshots[asset] = self._generate_stub_snapshot(asset)
        
        return snapshots
    
    def _generate_stub_snapshot(self, asset: str) -> MarketSnapshot:
        """Generate stub market snapshot for testing."""
        base_prices = {"SOL": 185.0, "BTC": 95000.0, "ETH": 3500.0}
        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            current_price=base_prices.get(asset, 100.0),
            volume_24h=50000000.0,
            data_age_seconds=1.0,
            vpin=0.5,
        )
    
    def _validate_market_data(self, market_data: Dict[str, MarketSnapshot]) -> bool:
        """Validate market data quality."""
        if not market_data:
            return False
        
        for asset, snapshot in market_data.items():
            if not snapshot.is_valid():
                logger.warning(f"Invalid data for {asset}")
                return False
        
        return True
    
    def _detect_regime(self, market_data: Dict[str, MarketSnapshot]) -> RegimeState:
        """Detect current market regime."""
        if self._regime_detector:
            try:
                # Get primary asset data
                primary = market_data.get(self.config.primary_asset)
                if primary:
                    result = self._regime_detector.detect({
                        "price": primary.current_price,
                        "volume": primary.volume_24h,
                    })
                    return RegimeState(
                        timestamp=datetime.now(timezone.utc),
                        regime=getattr(result, 'regime', 'UNKNOWN'),
                        phase=getattr(result, 'phase', RegimePhase.UNKNOWN),
                        direction=getattr(result, 'direction', 0.0),
                        confidence=getattr(result, 'confidence', 0.5),
                        volatility=getattr(result, 'volatility', 0.1),
                    )
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")
        
        # Return default
        return RegimeState(
            timestamp=datetime.now(timezone.utc),
            regime="TRENDING",
            phase=RegimePhase.EXPANSION,
            direction=0.5,
            confidence=0.6,
        )
    
    def _determine_mode(
        self, 
        market_data: Dict[str, MarketSnapshot],
        regime_state: RegimeState
    ) -> SystemMode:
        """
        Determine system operating mode.
        
        Mode Hierarchy (LOCKED):
            NO_TRADE > OPPORTUNITY > NORMAL
        """
        # Check NO_TRADE conditions
        no_trade_reasons = []
        
        # 1. Drawdown check (NON-NEGOTIABLE)
        if self._current_drawdown >= self.config.hard_drawdown_halt:
            no_trade_reasons.append(f"DD={self._current_drawdown:.1%} >= {self.config.hard_drawdown_halt:.1%}")
        
        # 2. Correlation crisis
        if regime_state.correlation_btc_eth_sol >= self.config.correlation_crisis:
            no_trade_reasons.append(f"CORR={regime_state.correlation_btc_eth_sol:.2f}")
        
        # 3. Data staleness
        for asset, snapshot in market_data.items():
            if snapshot.data_age_seconds > 5.0:
                no_trade_reasons.append(f"{asset}_STALE={snapshot.data_age_seconds:.1f}s")
        
        if no_trade_reasons:
            logger.info(f"[SPINE] NO_TRADE: {', '.join(no_trade_reasons)}")
            return SystemMode.NO_TRADE
        
        # Check OPPORTUNITY conditions
        opportunity_reasons = []
        
        # High confidence regime
        if regime_state.confidence > 0.75:
            opportunity_reasons.append(f"CONF={regime_state.confidence:.2f}")
        
        # Strong direction
        if abs(regime_state.direction) > 0.6:
            opportunity_reasons.append(f"DIR={regime_state.direction:.2f}")
        
        # IGNITION or EXPANSION phase
        if regime_state.phase in [RegimePhase.IGNITION, RegimePhase.EXPANSION]:
            opportunity_reasons.append(f"PHASE={regime_state.phase.value}")
        
        if len(opportunity_reasons) >= 2:
            logger.info(f"[SPINE] OPPORTUNITY: {', '.join(opportunity_reasons)}")
            return SystemMode.OPPORTUNITY
        
        return SystemMode.NORMAL
    
    def _generate_agent_signals(
        self,
        market_data: Dict[str, MarketSnapshot],
        regime_state: RegimeState
    ) -> Dict[str, AgentSignal]:
        """Generate signals from all agents."""
        signals = {}
        timestamp = datetime.now(timezone.utc)
        
        for asset in self.config.assets:
            snapshot = market_data.get(asset)
            if not snapshot:
                continue
            
            # Quant signal
            if self._quant_agent:
                try:
                    quant_result = self._quant_agent.generate_signal(
                        asset=asset,
                        price=snapshot.current_price,
                        regime=regime_state.regime,
                    )
                    signals[f"quant_{asset}"] = AgentSignal(
                        agent_id="quant",
                        timestamp=timestamp,
                        asset=asset,
                        direction=getattr(quant_result, 'direction', 0.0),
                        confidence=getattr(quant_result, 'confidence', 0.5),
                    )
                except Exception as e:
                    logger.debug(f"Quant signal error for {asset}: {e}")
                    signals[f"quant_{asset}"] = AgentSignal(
                        agent_id="quant",
                        timestamp=timestamp,
                        asset=asset,
                        direction=0.3,  # Slight bullish bias for stub
                        confidence=0.5,
                    )
            else:
                # Stub signal
                signals[f"quant_{asset}"] = AgentSignal(
                    agent_id="quant",
                    timestamp=timestamp,
                    asset=asset,
                    direction=0.3 if regime_state.direction > 0 else -0.2,
                    confidence=0.55,
                )
            
            # DRL signal - authority level determines forwarding (v6.7)
            if self._drl_agent:
                try:
                    # [P178] Three of eight arguments. position_state,
                    # market_data and agent_signals are omitted, so every
                    # state-bearing input to build_state() is empty and the
                    # DRL would score a zeroed vector. As of P178
                    # generate_signal refuses that case and returns
                    # is_valid=False / direction=0.0, so the branches below
                    # cannot forward a fabricated signal.
                    #
                    # Left uncorrected on purpose: supplying the three dicts
                    # here would put a never-exercised DRL path into a file
                    # this module's own docstring marks NOT USED, on a live
                    # trading system. Production DRL is the TQC ensemble in
                    # main.py (`_drl_ensembles`, :3697/:7795); this DRLAgent
                    # wrapper is itself legacy (agents/drl_agent.py:172).
                    # Fixing the caller is only worth doing as part of
                    # deleting or reviving this spine deliberately.
                    drl_result = self._drl_agent.generate_signal(
                        asset=asset,
                        price=snapshot.current_price,
                        regime=regime_state.regime,
                    )
                    drl_dir = getattr(drl_result, 'direction', 0.0)
                    drl_conf = getattr(drl_result, 'confidence', 0.5)
                    current_pos = self._positions.get(asset, 0.0)
                    drl_authority = getattr(self, '_drl_authority_level', 'DISABLED')

                    if drl_authority == "ACTIVE":
                        # ACTIVE: forward all signals (entry + exit)
                        if abs(drl_dir) > 0.1:
                            signals[f"drl_{asset}"] = AgentSignal(
                                agent_id="drl_active",
                                timestamp=timestamp,
                                asset=asset,
                                direction=drl_dir,
                                confidence=drl_conf,
                            )
                    elif drl_authority == "EXIT_ONLY":
                        # EXIT_ONLY: only forward exit signals
                        if current_pos > 0 and drl_dir < -0.3:
                            signals[f"drl_{asset}"] = AgentSignal(
                                agent_id="drl_exit",
                                timestamp=timestamp,
                                asset=asset,
                                direction=drl_dir,
                                confidence=drl_conf,
                            )
                        elif current_pos < 0 and drl_dir > 0.3:
                            signals[f"drl_{asset}"] = AgentSignal(
                                agent_id="drl_exit",
                                timestamp=timestamp,
                                asset=asset,
                                direction=drl_dir,
                                confidence=drl_conf,
                            )
                    # SHADOW/DISABLED: don't forward
                except Exception as e:
                    logger.debug(f"DRL signal error for {asset}: {e}")
        
        return signals
    
    def _fuse_signals(
        self,
        agent_signals: Dict[str, AgentSignal],
        regime_state: RegimeState,
        mode: SystemMode
    ) -> Optional[FusedSignal]:
        """Fuse signals from multiple agents."""
        timestamp = datetime.now(timezone.utc)
        
        # Focus on primary asset
        primary = self.config.primary_asset
        quant_key = f"quant_{primary}"
        drl_key = f"drl_{primary}"
        
        quant_signal = agent_signals.get(quant_key)
        drl_signal = agent_signals.get(drl_key)
        
        if not quant_signal:
            return None
        
        # Determine weights based on mode
        if mode == SystemMode.OPPORTUNITY:
            quant_weight = 0.7
            drl_weight = 0.3
        else:
            quant_weight = 0.8
            drl_weight = 0.2
        
        # Calculate fused direction
        fused_direction = quant_signal.direction * quant_weight
        fused_confidence = quant_signal.confidence * quant_weight
        
        if drl_signal:
            fused_direction += drl_signal.direction * drl_weight
            fused_confidence += drl_signal.confidence * drl_weight
        
        # Normalize
        total_weight = quant_weight + (drl_weight if drl_signal else 0)
        fused_direction /= total_weight
        fused_confidence /= total_weight
        
        # Calculate target exposure
        if mode == SystemMode.OPPORTUNITY:
            multiplier = self.config.opportunity_multiplier
        else:
            multiplier = self.config.normal_multiplier
        
        target_exposure = abs(fused_direction) * fused_confidence * multiplier
        
        # Determine execution mode
        if abs(fused_direction) > 0.7 and fused_confidence > 0.7:
            exec_mode = "AGGRESSIVE"
        elif abs(fused_direction) > 0.5:
            exec_mode = "BALANCED"
        else:
            exec_mode = "PASSIVE_PREFERRED"
        
        return FusedSignal(
            timestamp=timestamp,
            asset=primary,
            direction=fused_direction,
            confidence=fused_confidence,
            urgency=0.5 + 0.3 * abs(fused_direction),
            target_exposure=target_exposure,
            execution_mode=exec_mode,
            quant_weight=quant_weight,
            drl_weight=drl_weight if drl_signal else 0,
            quant_signal=quant_signal.direction,
            drl_signal=drl_signal.direction if drl_signal else 0,
            authority_source="quant" if not drl_signal else "fusion",
        )
    
    def _check_risk_governor(
        self,
        fused_signal: Optional[FusedSignal],
        market_data: Dict[str, MarketSnapshot]
    ):
        """
        Check with risk governor for trade approval.
        
        Risk governor has VETO AUTHORITY over all execution.
        """
        if not fused_signal:
            return _RiskDecision(vetoed=False, reason="")
        
        # Check drawdown
        if self._current_drawdown >= self.config.hard_drawdown_halt:
            return _RiskDecision(
                vetoed=True, 
                reason=f"HARD_DD: {self._current_drawdown:.1%}"
            )
        
        # Check exposure limits
        current_exposure = sum(abs(v) for v in self._positions.values())
        proposed_exposure = current_exposure + fused_signal.target_exposure
        
        if proposed_exposure > 1.5:  # 150% max gross
            return _RiskDecision(
                vetoed=True,
                reason=f"EXPOSURE: {proposed_exposure:.1%} > 150%"
            )
        
        # Check asset-specific limits
        asset = fused_signal.asset
        current_asset_exposure = abs(self._positions.get(asset, 0))
        proposed_asset = current_asset_exposure + fused_signal.target_exposure
        
        if proposed_asset > 0.8:  # 80% max per asset
            return _RiskDecision(
                vetoed=True,
                reason=f"{asset}_LIMIT: {proposed_asset:.1%} > 80%"
            )
        
        # If risk governor module available, use it
        if self._risk_governor:
            try:
                decision = self._risk_governor.check(
                    asset=asset,
                    direction=fused_signal.direction,
                    size=fused_signal.target_exposure,
                )
                return decision
            except Exception as e:
                logger.warning(f"Risk governor error: {e}")
        
        return _RiskDecision(vetoed=False, reason="")
    
    def _create_execution_intent(
        self,
        fused_signal: FusedSignal,
        mode: SystemMode
    ) -> ExecutionIntent:
        """Create execution intent from fused signal."""
        # Check SOL dominance
        sol_dominance = self._check_sol_dominance(fused_signal, mode)
        
        return ExecutionIntent(
            timestamp=datetime.now(timezone.utc),
            asset=fused_signal.asset,
            direction=fused_signal.direction,
            target_exposure=fused_signal.target_exposure,
            size_usd=fused_signal.target_exposure * self._portfolio_value,
            execution_mode=fused_signal.execution_mode,
            urgency=fused_signal.urgency,
            sol_dominance_active=sol_dominance.get("active", False),
            sol_dominance_ttl=sol_dominance.get("ttl", 0),
        )
    
    # =========================================================================
    # H1 FIX: EXECUTION ACCEPTANCE CHECK
    # =========================================================================
    
    def _check_execution_acceptance(
        self,
        intent: ExecutionIntent,
        market_data: Dict[str, MarketSnapshot],
        fused_signal: FusedSignal,
    ) -> ExecutionAcceptance:
        """
        H1 FIX: 检查执行层是否接受 intent
        
        这是一致性修复，解决上层建仓与执行层WAIT的竞态问题
        
        返回 ExecutionAcceptance，明确记录接受/拒绝状态
        """
        timestamp = datetime.now(timezone.utc)
        asset_data = market_data.get(intent.asset)
        
        # 获取当前 spread
        spread_bps = asset_data.spread_bps if asset_data else 0.0
        
        # 检查1: 极端 spread (与 learned_execution_policy 一致)
        if spread_bps > 100:
            return ExecutionAcceptance(
                accepted=False,
                reason="EXTREME_SPREAD",
                timestamp=timestamp,
                rejection_source="execution_acceptance_check",
                spread_bps=spread_bps,
                execution_mode=intent.execution_mode,
            )
        
        # 检查2: 执行模式是否为 PASSIVE_ONLY 且方向激进
        if intent.execution_mode == "PASSIVE_ONLY" and abs(intent.direction) > 0.8:
            return ExecutionAcceptance(
                accepted=False,
                reason="PASSIVE_ONLY_MODE",
                timestamp=timestamp,
                rejection_source="execution_mode_mismatch",
                spread_bps=spread_bps,
                execution_mode=intent.execution_mode,
            )
        
        # 检查3: LearnedExecutionPolicy 是否熔断
        try:
            from execution.learned_execution_policy import get_learned_execution_policy
            learned_policy = get_learned_execution_policy()
            if learned_policy._fused:
                return ExecutionAcceptance(
                    accepted=False,
                    reason="EXEC_POLICY_FUSED",
                    timestamp=timestamp,
                    rejection_source="learned_execution_policy",
                    spread_bps=spread_bps,
                    execution_mode=intent.execution_mode,
                )
        except ImportError:
            pass  # LearnedExecutionPolicy not available
        
        # 检查4: 数据有效性
        if asset_data and asset_data.data_age_seconds > 5.0:
            return ExecutionAcceptance(
                accepted=False,
                reason="STALE_DATA",
                timestamp=timestamp,
                rejection_source="data_staleness_check",
                spread_bps=spread_bps,
                execution_mode=intent.execution_mode,
            )
        
        # 全部通过，接受 intent
        return ExecutionAcceptance(
            accepted=True,
            reason="OK",
            timestamp=timestamp,
            rejection_source="",
            spread_bps=spread_bps,
            execution_mode=intent.execution_mode,
        )
    
    # =========================================================================
    # M3 FIX: DECISION SUMMARY
    # =========================================================================
    
    def _generate_decision_summary(self, result: 'TickResult') -> str:
        """
        M3 FIX: 生成统一的决策摘要
        
        目的: 30分钟内完成事后复盘
        
        Updated: 增加 sentiment_mode 追踪
        """
        # 确定 intent 方向
        if result.fused_signal:
            intent_dir = "LONG" if result.fused_signal.direction > 0 else "SHORT"
            intent_strength = abs(result.fused_signal.direction)
        else:
            intent_dir = "NONE"
            intent_strength = 0.0
        
        # 确定状态
        if result.execution_result and result.execution_result.success:
            status = "EXECUTED"
            blocked_by = ""
        elif result.risk_veto:
            status = "REJECTED"
            blocked_by = f"RISK_GOVERNOR:{result.risk_veto_reason}"
        elif result.execution_acceptance and not result.execution_acceptance.accepted:
            status = "REJECTED"
            blocked_by = f"EXECUTION:{result.execution_acceptance.reason}"
        elif result.system_mode == SystemMode.NO_TRADE:
            status = "REJECTED"
            blocked_by = "NO_TRADE_MODE"
        elif not result.fused_signal or abs(result.fused_signal.direction) <= 0.1:
            status = "NO_SIGNAL"
            blocked_by = ""
        else:
            status = "SKIPPED"
            blocked_by = "UNKNOWN"
        
        # 收集 secondary caps (从 fused_signal 或其他来源)
        secondary_caps = list(result.secondary_caps) if result.secondary_caps else []
        
        # 检查是否有额外的限制
        if result.fused_signal:
            if result.fused_signal.execution_mode == "PASSIVE_PREFERRED":
                secondary_caps.append("PASSIVE_MODE")
        
        # P3 FIX: 添加 sentiment mode 到 secondary_caps
        try:
            from engine.compute.vllm_inference_wrapper import is_sentiment_mock
            if is_sentiment_mock():
                secondary_caps.append("SENTIMENT=MOCK")
        except ImportError:
            secondary_caps.append("SENTIMENT=UNAVAILABLE")
        
        # 更新 result 字段
        result.blocked_by = blocked_by
        
        # 生成摘要
        caps_str = ",".join(secondary_caps) if secondary_caps else "none"
        
        return (
            f"[DECISION_SUMMARY] "
            f"tick={result.tick_id} "
            f"intent={intent_dir}({intent_strength:.2f}) "
            f"status={status} "
            f"blocked_by={blocked_by or 'none'} "
            f"secondary_caps=[{caps_str}]"
        )
    
    def _check_sol_dominance(self, fused_signal: FusedSignal, mode: SystemMode) -> Dict:
        """Check v3.3-HR SOL dominance conditions."""
        if fused_signal.asset != "SOL":
            return {"active": False, "ttl": 0}
        
        # Decrement TTL
        if self._sol_dominance_ttl > 0:
            self._sol_dominance_ttl -= 1
        
        # Auto-disable on TTL expiry
        if self._sol_dominance_active and self._sol_dominance_ttl <= 0:
            logger.info("[SPINE] SOL dominance AUTO-DISABLED: TTL expired")
            self._sol_dominance_active = False
            return {"active": False, "ttl": 0}
        
        # Check activation
        if not self._sol_dominance_active:
            phase = self._current_regime.phase if self._current_regime else RegimePhase.UNKNOWN
            if (
                mode == SystemMode.OPPORTUNITY and
                phase in [RegimePhase.IGNITION, RegimePhase.EXPANSION] and
                abs(fused_signal.direction) > 0.7
            ):
                self._sol_dominance_active = True
                self._sol_dominance_ttl = 2  # 2 bars TTL
                logger.warning(
                    f"[SPINE] SOL DOMINANCE ACTIVATED: phase={phase.value}, "
                    f"signal={fused_signal.direction:.2f}"
                )
                return {"active": True, "ttl": 2}
        
        return {
            "active": self._sol_dominance_active,
            "ttl": self._sol_dominance_ttl
        }
    
    async def _execute(self, intent: ExecutionIntent) -> ExecutionResult:
        """Execute the trade intent."""
        timestamp = datetime.now(timezone.utc)
        
        if self._execution_manager:
            try:
                # Map to execution manager interface
                from execution.execution_manager import OrderSide, OrderType
                
                side = OrderSide.BUY if intent.direction > 0 else OrderSide.SELL
                
                result = self._execution_manager.execute_order(
                    symbol=f"{intent.asset}/USD",
                    side=side,
                    size=intent.size_usd / 100,  # Simplified sizing
                )
                
                # Update positions
                if result.success:
                    current = self._positions.get(intent.asset, 0.0)
                    delta = intent.target_exposure if intent.direction > 0 else -intent.target_exposure
                    self._positions[intent.asset] = current + delta
                
                return ExecutionResult(
                    timestamp=timestamp,
                    asset=intent.asset,
                    success=result.success,
                    order_id=result.order_id or "",
                    side="BUY" if intent.direction > 0 else "SELL",
                    requested_size=intent.size_usd,
                    filled_size=result.filled_size * result.filled_price if result.success else 0,
                    filled_price=result.filled_price,
                    slippage_bps=result.slippage * 10000 if result.slippage else 0,
                    status="FILLED" if result.success else "REJECTED",
                    error=result.error_message,
                    new_exposure=self._positions.get(intent.asset, 0.0),
                )
                
            except Exception as e:
                logger.error(f"Execution error: {e}")
                return ExecutionResult(
                    timestamp=timestamp,
                    asset=intent.asset,
                    success=False,
                    error=str(e),
                )
        
        # Stub execution
        current = self._positions.get(intent.asset, 0.0)
        delta = intent.target_exposure if intent.direction > 0 else -intent.target_exposure
        self._positions[intent.asset] = current + delta
        
        return ExecutionResult(
            timestamp=timestamp,
            asset=intent.asset,
            success=True,
            order_id=f"STUB_{self._tick_count}",
            side="BUY" if intent.direction > 0 else "SELL",
            requested_size=intent.size_usd,
            filled_size=intent.size_usd,
            status="FILLED_STUB",
            new_exposure=self._positions[intent.asset],
        )
    
    def _process_feedback(self, result: TickResult):
        """Process execution feedback."""
        if self._feedback_loop:
            try:
                self._feedback_loop.record(result)
            except Exception as e:
                logger.warning(f"Feedback recording error: {e}")
        
        # Update portfolio value (simplified)
        if result.execution_result and result.execution_result.pnl_realized:
            self._portfolio_value += result.execution_result.pnl_realized
            self._peak_value = max(self._peak_value, self._portfolio_value)
            self._current_drawdown = (
                (self._peak_value - self._portfolio_value) / self._peak_value
            )
    
    def _add_proof_log(self, log_line: str):
        """Add proof log entry."""
        self._proof_logs.append(log_line)
        logger.info(log_line)
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    async def run(self, max_ticks: Optional[int] = None):
        """Run the spine continuously."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Spine initialization failed")
        
        self._running = True
        logger.info("[SPINE] Starting main loop...")
        
        try:
            tick_count = 0
            while self._running:
                if max_ticks and tick_count >= max_ticks:
                    break
                
                result = await self.process_tick()
                tick_count += 1
                
                # Log progress
                if tick_count % 10 == 0:
                    logger.info(f"[SPINE] Processed {tick_count} ticks")
                
                # Sleep until next tick (4H in production)
                await asyncio.sleep(0.1)  # Fast for testing
                
        except asyncio.CancelledError:
            logger.info("[SPINE] Run cancelled")
        finally:
            self._running = False
    
    def stop(self):
        """Stop the spine gracefully."""
        logger.info("[SPINE] Stop requested")
        self._running = False
    
    def get_state(self) -> Dict:
        """Get current spine state."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "tick_count": self._tick_count,
            "mode": self._current_mode.value,
            "positions": self._positions.copy(),
            "portfolio_value": self._portfolio_value,
            "drawdown": self._current_drawdown,
            "sol_dominance": {
                "active": self._sol_dominance_active,
                "ttl": self._sol_dominance_ttl,
            },
        }
    
    def get_proof_logs(self) -> List[str]:
        """Get all proof logs."""
        return self._proof_logs.copy()


# =============================================================================
# HELPER CLASSES
# =============================================================================

@dataclass
class _RiskDecision:
    """Internal risk decision."""
    vetoed: bool = False
    reason: str = ""


# =============================================================================
# SINGLETON
# =============================================================================

_runtime_spine: Optional[RuntimeSpine] = None


def get_runtime_spine(config: Optional[SpineConfig] = None) -> RuntimeSpine:
    """Get runtime spine singleton."""
    global _runtime_spine
    if _runtime_spine is None:
        _runtime_spine = RuntimeSpine(config)
    elif config is not None and _runtime_spine.config != config:
        _runtime_spine = RuntimeSpine(config)
    return _runtime_spine


def reset_runtime_spine():
    """Reset runtime spine singleton."""
    global _runtime_spine
    _runtime_spine = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        spine = RuntimeSpine()
        spine.initialize()
        
        # Process a few ticks
        for i in range(5):
            result = await spine.process_tick()
            print(f"\nTick {i+1}: {result.system_mode.value}")
            print(f"  Direction: {result.fused_signal.direction if result.fused_signal else 0:.2f}")
            print(f"  Veto: {result.risk_veto}")
        
        print("\n" + "="*60)
        print("Final State:")
        print(spine.get_state())
    
    asyncio.run(test())

"""
HMATS Defense Module - 防御系统
===============================
Version: 5.1.0-HARDENED

SOTA对照: "策略规则强制执行 (Constitution Guarantees)"

核心组件:
- Constitution: 策略宪法强制执行
- ProductionReliability: 生产可靠性 (call-chain, vetoes)
- HumanOverride: 人工干预协议 (时间限制)
- ShadowLedger: 影子账本审计
- Watchdog: 系统看门狗
- StartupReconciler: 启动协调器
- KrakenIntegrityShield: Kraken完整性保护
- DataDriftDetector: 数据漂移检测
- VRAMGuard: VRAM保护
- KrakenRateLimitManager: Kraken速率限制
- TradeGate: Final gate before execution (v5.1.0-HARDENED)
"""

from .constitution import (
    MarketDataValidator,
    MarketDataValidationResult,
    NoTradeTriggerType,
    NoTradeTriggerChecker,
    NoTradeState,
    OpportunityTriggerGroup,
    OpportunityTriggerChecker,
    OpportunityState,
    AlphaThresholdCalculator,
    AlphaGatingResult,
    TrancheScheduler,
    TrancheDecision,
    TrancheLevel,
    ConstitutionRule,
    StrategyConstitutionEnforcer,
    ConstitutionGuaranteesModule,
    CANONICAL_SCHEMA,
)

# Alias for common usage
ConstitutionEnforcer = StrategyConstitutionEnforcer

from defense.production_reliability import (
    CallChainProof,
    canonical_entrypoint,
    VetoType,
    RiskVetoClassifier,
)

from .human_override import (
    OverrideType,
    HumanOverride,
    HumanOverrideProtocol,
    get_override_protocol,
)

# v5.1.0-HARDENED: TradeGate exports (glue-layer fix)
from .trade_gate import (
    TradeGate,
    TradeGateConfig,
    GateDecision,
    RejectReason,
    GateResult,
    TradeProposal,
    get_trade_gate,
)

__all__ = [
    # Constitution
    "MarketDataValidator",
    "MarketDataValidationResult",
    "NoTradeTriggerType",
    "NoTradeTriggerChecker",
    "NoTradeState",
    "OpportunityTriggerGroup",
    "OpportunityTriggerChecker",
    "OpportunityState",
    "AlphaThresholdCalculator",
    "AlphaGatingResult",
    "TrancheScheduler",
    "TrancheDecision",
    "TrancheLevel",
    "ConstitutionRule",
    "StrategyConstitutionEnforcer",
    "ConstitutionEnforcer",  # Alias
    "ConstitutionGuaranteesModule",
    "CANONICAL_SCHEMA",
    # Production Reliability
    "CallChainProof",
    "canonical_entrypoint",
    "VetoType",
    "RiskVetoClassifier",
    # Human Override
    "OverrideType",
    "HumanOverride",
    "HumanOverrideProtocol",
    "get_override_protocol",
    # Trade Gate
    "TradeGate",
    "TradeGateConfig",
    "GateDecision",
    "RejectReason",
    "GateResult",
    "TradeProposal",
    "get_trade_gate",
]

# Additional exports for compatibility
try:
    from .startup_reconciler import StartupReconciler, StartupReconcileConfig
    __all__.extend(["StartupReconciler", "StartupReconcileConfig"])
except ImportError:
    StartupReconciler = None
    StartupReconcileConfig = None

try:
    from .shadow_ledger_jsonl import ShadowLedgerJSONL as ShadowLedger
    __all__.append("ShadowLedger")
except ImportError:
    ShadowLedger = None

try:
    from .kraken_integrity_shield import KrakenIntegrityShield
    __all__.append("KrakenIntegrityShield")
except ImportError:
    KrakenIntegrityShield = None

try:
    from .drift_detector import DataDriftDetector
    __all__.append("DataDriftDetector")
except ImportError:
    DataDriftDetector = None

try:
    from .vram_guard import VRAMGuard
    __all__.append("VRAMGuard")
except ImportError:
    VRAMGuard = None

try:
    from .rate_limit_manager import KrakenRateLimitManager
    __all__.append("KrakenRateLimitManager")
except ImportError:
    KrakenRateLimitManager = None

try:
    from .p0_safety_integrator import P0SafetyIntegrator, get_p0_safety
    __all__.extend(["P0SafetyIntegrator", "get_p0_safety"])
except ImportError:
    P0SafetyIntegrator = None
    get_p0_safety = None

# SOL Defense Module (Jito Tips, Network Health)
try:
    from .sol_defense import (
        SOLDefenseModule,
        SOLDefenseConfig,
        SOLRiskLevel,
        NetworkStatus,
    )
    __all__.extend(["SOLDefenseModule", "SOLDefenseConfig", "SOLRiskLevel", "NetworkStatus"])
except ImportError:
    SOLDefenseModule = None
    SOLDefenseConfig = None

# Governor Integration (v6.1.3 - Long-Term Survival)
try:
    from .governor_integration import (
        GovernorIntegration,
        GovernorIntegrationConfig,
        GovernorCheckResult,
        GovernorVetoSource,
        get_governor_integration,
        reset_governor_integration,
        check_governors,
        update_governor_market_data,
    )
    __all__.extend([
        "GovernorIntegration",
        "GovernorIntegrationConfig", 
        "GovernorCheckResult",
        "GovernorVetoSource",
        "get_governor_integration",
        "reset_governor_integration",
        "check_governors",
        "update_governor_market_data",
    ])
except ImportError:
    GovernorIntegration = None

# V6.2.3 Backfill: BitBeast Rules Layer
try:
    from .bitbeast_rules import (
        BitBeastBehaviorGuard,
        VolumeFilter,
        StructureBreakoutFilter,
        VolumeAnalysis,
        StructureAnalysis,
    )
    __all__.extend([
        "BitBeastBehaviorGuard",
        "VolumeFilter",
        "StructureBreakoutFilter",
        "VolumeAnalysis",
        "StructureAnalysis",
    ])
except ImportError as e:
    BitBeastBehaviorGuard = None

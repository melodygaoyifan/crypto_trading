"""
HMATS Market Module - 市场状态识别 + 市场分析
=============================================

核心组件:
- RegimeDetector: 市场制度检测 (BEAR/BULL/SIDEWAYS)
- PhaseDetector: 阶段检测 (IGNITION->EXHAUSTION)
- CorrelationMonitor: 多窗口相关性监控 (v4.0)
- StructureAnalyzer: 结构分析
- AlphaThreshold: Alpha阈值管理 (moved to defense/constitution.py)
- HighRiskConfig: 高风险配置
- MultiAssetIntelligence: 多资产智能
- LeadLagEngine: 领先滞后引擎
"""

from .phase_detector import (
    RegimePhase,
    PhaseIndicators,
    RegimePhaseResult,
    RegimePhaseDetector,
    PHASE_OPPORTUNITY_SCORES,
)

# Alias for backward compatibility
PhaseDetector = RegimePhaseDetector

__all__ = [
    "RegimePhase",
    "RegimePhaseDetector",
    "PhaseDetector",  # Alias
    "PHASE_OPPORTUNITY_SCORES",
    "PhaseIndicators",
    "RegimePhaseResult",
]


# v3.3-STABILIZED 模块迁移 (2026-01-25)
from .intraday_correlation_monitor import (
    IntradayCorrelationMonitor,
    FastCorrelationChecker,
    CorrelationState,
    CorrelationConfig,
    CorrelationLevel,
    get_correlation_monitor,
    get_fast_correlation_checker
)

# Regime detector
try:
    from .regime_detector import (
        RegimeDetector,
        MarketRegime,
        RegimeConfig,
        get_regime_detector,
    )
    __all__.extend(["RegimeDetector", "MarketRegime", "RegimeConfig", "get_regime_detector"])
except ImportError:
    pass

# v6.2.3 Backfill: Enhanced Regime Navigator
try:
    from .regime_navigator import (
        EnhancedMarketRegimeNavigator,
        VolOfVolTensor,
        LiquidityRegime,
        VolatilityRegime,
    )
    __all__.extend([
        "EnhancedMarketRegimeNavigator", 
        "VolOfVolTensor", 
        "LiquidityRegime",
        "VolatilityRegime",
    ])
except ImportError as e:
    print(f"[WARN]️ regime_navigator import warning: {e}")

# v6.2.3 Backfill: Structure Analyzer
try:
    from .structure_analyzer import (
        StructureBreakAnalyzer,
        StructureBreakResult,
        BreakDirection,
        FractalType,
    )
    __all__.extend([
        "StructureBreakAnalyzer",
        "StructureBreakResult",
        "BreakDirection",
        "FractalType",
    ])
except ImportError as e:
    print(f"[WARN]️ structure_analyzer import warning: {e}")

# AlphaThresholdCalculator: production version in defense/constitution.py (UNLEASH-tuned multipliers)
# Legacy market/alpha_threshold.py moved to archive - do not restore (stale multipliers 3.0/1.5/2.0/2.5)

try:
    from .high_risk_config import (
        HighRiskConfig,
        OverrideMode,
        get_hr_config,
    )
    __all__.extend(["HighRiskConfig", "OverrideMode", "get_hr_config"])
except ImportError:
    pass

try:
    from .multi_asset_intelligence import MultiAssetHeavyModel
    __all__.append("MultiAssetHeavyModel")
except ImportError:
    pass

try:
    from .lead_lag_engine import LeadLagAlphaEngine
    __all__.append("LeadLagAlphaEngine")
except ImportError:
    pass

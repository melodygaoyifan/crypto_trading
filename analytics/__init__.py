"""
HMATS Analytics Module - 分析与归因系统
=======================================

SOTA对照: "PnL归因分析，持续优化策略表现"
"""

from .pnl_attribution import (
    EntryAttribution,
    ExecutionAttribution,
    ExitAttribution,
    TradeAttribution,
    PnLAttributionManager,
    get_attribution_manager,
)

from .strategy_aging import (
    StrategyAgingConfig,
    StrategyEffectiveness,
    StrategyAgingManager,
    get_aging_manager,
)

from .failure_memory import (
    OpportunityOutcome,
    FailureAwareMetaMemory,
)

from .confidence_scorer import (
    StrategyConfidenceScorer,
)

__all__ = [
    "EntryAttribution",
    "PnLAttributionManager",
    "PnLAttributionEngine",  # v5.1.0-HARDENED: Alias for main.py
    "get_attribution_manager",
    "StrategyAgingManager",
    "get_aging_manager",
    "FailureAwareMetaMemory",
    "StrategyConfidenceScorer",
]

# v5.1.0-HARDENED: Alias for main.py compatibility
PnLAttributionEngine = PnLAttributionManager

# P3: Trade Explainer
from .trade_explainer import (
    LLMExplainer,
    RuleBasedExplainer,
    TradeContext,
    Explanation,
    ExplainerResult,
    ExplanationType,
    get_trade_explainer,
    get_explanation_logger,
)

# P3: Monitoring Dashboard
from .monitoring_dashboard import (
    MetricCollector,
    ConsoleDashboard,
    JSONExporter,
    DashboardSnapshot,
    PnLMetrics,
    PositionMetrics,
    RiskMetrics,
    SystemMetrics,
    get_metric_collector,
    get_console_dashboard,
    get_json_exporter,
)

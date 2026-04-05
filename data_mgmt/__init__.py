"""
HMATS Data Management - 数据管理
================================

数据存储、状态管理、日志系统。

核心组件:
- ZeroDiskRAMManager: 零磁盘内存管理
- ShadowLedgerManager: 影子账本
- GlobalContextInformer: 全局上下文
- StateVectorAugmentor: 状态向量增强
- ProofLogSystem: 证明日志系统
- DataHealthMonitor: 数据健康监控
"""

__all__ = []

# P3: Data Validation (removes random fallbacks)
from .data_validation import (
    DataValidator,
    DataFallback,
    ValidationResult,
    FallbackStrategy,
    DataQuality,
    OnchainFeatures,
    OnchainDataProcessor,
    SignalWeightAdjuster,
    get_data_validator,
    get_onchain_processor,
)

__all__.extend([
    "DataValidator",
    "DataFallback", 
    "ValidationResult",
    "FallbackStrategy",
    "DataQuality",
    "get_data_validator",
])

# Data Health Monitoring
try:
    from .data_health import (
        DataSourceType,
        HealthStatus,
        DegradationLevel,
        SourceHealth,
        DataHealthSnapshot,
    )
    
    # Alias for common usage
    DataHealthMonitor = DataHealthSnapshot
    
    __all__.extend([
        "DataSourceType",
        "HealthStatus",
        "DegradationLevel",
        "SourceHealth",
        "DataHealthSnapshot",
        "DataHealthMonitor",
    ])
except ImportError:
    pass

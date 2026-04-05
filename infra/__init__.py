"""
HMATS Infrastructure Module - 基础设施
======================================
Version: 5.1.0-HARDENED

核心组件:
- DataProvider: 市场数据提供
- KrakenClient: Kraken API客户端
- EventBus: 事件总线
- Persistence: 持久化和报警
"""

# =============================================================================
# v5.1.0-HARDENED: Core infrastructure exports (glue-layer fix)
# =============================================================================

# EventBus - system-wide event routing
from .event_bus import (
    EventBus,
    Event,
    EventType,
    EventPriority,
    SignalEvent,
    OrderEvent,
    get_event_bus,
    init_event_bus,
    publish_signal,
    publish_order,
    publish_risk_alert,
    publish_position_event,
    publish_system_event,
)

# DataProvider - market data fetching
from .data_provider import (
    MarketDataProvider,
    MarketDataConfig,
    MarketDataResult,
    AssetMarketData,
    DataSource,
    get_market_data_provider,
    fetch_market_data,
    validate_market_data_for_trading,
)

# v5.1.0-HARDENED: DataProvider alias for backward compat
DataProvider = MarketDataProvider

__all__ = [
    # EventBus
    "EventBus",
    "Event", 
    "EventType",
    "EventPriority",
    "SignalEvent",
    "OrderEvent",
    "get_event_bus",
    "init_event_bus",
    "publish_signal",
    "publish_order",
    "publish_risk_alert",
    # DataProvider
    "DataProvider",
    "MarketDataProvider",
    "MarketDataConfig",
    "MarketDataResult",
    "get_market_data_provider",
    "fetch_market_data",
    # Persistence
    "HotRestartPersistence",
    "HotRestartManager",
    "get_persistence",
    # Kraken
    "KrakenLink",
    "LocalOrderbook",
    "ErrorClassifier",
]


# v3.3-STABILIZED 模块迁移 (2026-01-25)
from .hot_restart_persistence import (
    HotRestartPersistence,
    get_persistence
)

# Additional exports
from .kraken_link import (
    LocalOrderbook,
    ErrorClassifier
)
from .data_schema import (
    CanonicalMarketData,
    MarketDataValidator as SchemaValidator,
    OHLCVData
)

# P1: EventBus Integration
from .event_integration import (
    EventIntegrationManager,
    MarketDataEmitter,
    RegimeEmitter,
    SignalEmitter,
    OrderEmitter,
    RiskEmitter,
    TickLogBuilder,
    TickEvent,
    get_event_integration,
    reset_event_integration,
)

# =============================================================================
# v5.1.0-HARDENED: KrakenLink export (glue-layer fix)
# =============================================================================
try:
    from .kraken_link import KrakenLink
except ImportError:
    KrakenLink = None

# =============================================================================
# v5.1.0-HARDENED: HotRestartManager alias (glue-layer fix)
# =============================================================================
# main.py references HotRestartManager but module exports HotRestartPersistence
HotRestartManager = HotRestartPersistence

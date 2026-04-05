"""
================================================================================
HMATS v6.5 - Production-Grade Multi-Modal Data Feeds
================================================================================

所有数据采集必须通过此模块进行，禁止在 Agent 内直接调用外部 API。

v6.5 包含:
- onchain_feed: 链上数据采集
- sentiment_feed: 舆情数据采集 (基础)
- macro_feed: 宏观数据采集 (基础/legacy)
- lob_feed: 订单簿数据采集

P0 数据源 (必须):
- fred_feed: FRED API (宏观事件窗口, 金融条件)
- coinglass_feed: 衍生品数据 (资金费率, OI, 清算)
- lunarcrush_feed: 社交注意力 (拥挤检测)
- cryptopanic_feed: 新闻恐慌指数

P1 数据源 (可选):
- trading_economics_feed: 经济日历 (macro_shock计算)

每个 feed 输出统一 schema:
- timestamp
- staleness_sec
- confidence
- data
================================================================================
"""

from .onchain_feed import (
    OnChainFeed,
    OnChainTick,
    get_onchain_feed,
)

from .sentiment_feed import (
    SentimentFeed,
    SentimentTick,
    get_sentiment_feed,
)

from .macro_feed import (
    MacroFeed,
    MacroTick,
    get_macro_feed,
)

from .lob_feed import (
    LOBFeed,
    LOBTick,
    get_lob_feed,
)

# v6.5 P0 Feeds
from .fred_feed import (
    FREDFeed,
    FREDMacroData,
    get_fred_feed,
    reset_fred_feed,
)

from .coinglass_feed import (
    CoinglassFeed,
    CoinglassCrowdData,
    get_coinglass_feed,
    reset_coinglass_feed,
)

from .lunarcrush_feed import (
    LunarCrushFeed,
    LunarCrushAttentionData,
    get_lunarcrush_feed,
    reset_lunarcrush_feed,
)

from .cryptopanic_feed import (
    CryptoPanicFeed,
    CryptoPanicData,
    get_cryptopanic_feed,
    reset_cryptopanic_feed,
)

# v6.5 P1 Feeds (Optional)
from .trading_economics_feed import (
    TradingEconomicsFeed,
    TradingEconomicsData,
    get_trading_economics_feed,
    reset_trading_economics_feed,
)

__all__ = [
    # OnChain
    'OnChainFeed',
    'OnChainTick',
    'get_onchain_feed',
    # Sentiment
    'SentimentFeed',
    'SentimentTick',
    'get_sentiment_feed',
    # Macro (legacy)
    'MacroFeed',
    'MacroTick',
    'get_macro_feed',
    # LOB
    'LOBFeed',
    'LOBTick',
    'get_lob_feed',
    # v6.5 P0 Feeds
    'FREDFeed',
    'FREDMacroData',
    'get_fred_feed',
    'reset_fred_feed',
    'CoinglassFeed',
    'CoinglassCrowdData',
    'get_coinglass_feed',
    'reset_coinglass_feed',
    'LunarCrushFeed',
    'LunarCrushAttentionData',
    'get_lunarcrush_feed',
    'reset_lunarcrush_feed',
    'CryptoPanicFeed',
    'CryptoPanicData',
    'get_cryptopanic_feed',
    'reset_cryptopanic_feed',
    # v6.5 P1 Feeds
    'TradingEconomicsFeed',
    'TradingEconomicsData',
    'get_trading_economics_feed',
    'reset_trading_economics_feed',
]

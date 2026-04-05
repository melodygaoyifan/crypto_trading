"""
================================================================================
HMATS v5.2.1 P0 - 舆情数据采集器
Sentiment Data Feed
================================================================================

负责从舆情数据源采集数据，输出标准化 SentimentTick。

功能:
1. Fear & Greed 指数
2. Twitter/X 情绪分析
3. Reddit 情绪分析
4. 新闻情绪分析
5. 资金费率

生产替换说明:
- 当前使用 mock/placeholder API
- 生产环境替换 _fetch_from_source() 中的实现
- 支持 Alternative.me / LunarCrush / Santiment 等

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session, fetch_with_retry
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SentimentDataSource(Enum):
    """舆情数据源"""
    ALTERNATIVE_ME = "alternative_me"
    LUNAR_CRUSH = "lunar_crush"
    SANTIMENT = "santiment"
    CUSTOM_LLM = "custom_llm"
    MOCK = "mock"


@dataclass
class FearGreedData:
    """Fear & Greed 数据"""
    value: int  # 0-100
    label: str  # extreme_fear, fear, neutral, greed, extreme_greed
    timestamp: datetime


@dataclass
class SocialSentiment:
    """社交媒体情绪"""
    platform: str  # twitter, reddit
    token: str
    sentiment_score: float  # -1 to 1
    volume: int  # 提及量
    change_24h: float  # 情绪变化
    timestamp: datetime


@dataclass
class NewsSentiment:
    """新闻情绪"""
    source: str
    headline: str
    sentiment_score: float  # -1 to 1
    relevance: float  # 0-1
    tokens_mentioned: List[str]
    timestamp: datetime


@dataclass
class FundingRate:
    """资金费率"""
    exchange: str
    symbol: str
    rate: float  # 资金费率
    next_funding_time: datetime
    timestamp: datetime


@dataclass
class SentimentTick:
    """
    标准化舆情数据输出
    
    所有舆情数据必须通过此格式输出
    """
    timestamp: datetime
    staleness_sec: float
    confidence: float  # 0-1, 数据可信度
    
    # Fear & Greed
    fear_greed: Optional[FearGreedData] = None
    
    # 社交情绪
    social_sentiments: List[SocialSentiment] = field(default_factory=list)
    
    # 新闻情绪
    news_sentiments: List[NewsSentiment] = field(default_factory=list)
    
    # 资金费率
    funding_rates: List[FundingRate] = field(default_factory=list)
    
    # 聚合指标
    overall_sentiment: float = 0.0  # -1 to 1
    sentiment_trend: str = "neutral"  # improving, stable, deteriorating
    extreme_fear: bool = False
    extreme_greed: bool = False
    
    # 元数据
    source: str = "unknown"
    tokens_covered: List[str] = field(default_factory=list)
    
    def is_stale(self, max_staleness_sec: float = 300) -> bool:
        """检查数据是否过期"""
        return self.staleness_sec > max_staleness_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "staleness_sec": self.staleness_sec,
            "confidence": self.confidence,
            "fear_greed_value": self.fear_greed.value if self.fear_greed else None,
            "fear_greed_label": self.fear_greed.label if self.fear_greed else None,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_trend": self.sentiment_trend,
            "extreme_fear": self.extreme_fear,
            "extreme_greed": self.extreme_greed,
            "source": self.source,
            "tokens_covered": self.tokens_covered,
            "social_count": len(self.social_sentiments),
            "news_count": len(self.news_sentiments),
            "funding_count": len(self.funding_rates),
        }


# =============================================================================
# SENTIMENT FEED
# =============================================================================

class SentimentFeed:
    """
    舆情数据采集器
    
    生产级数据管道，所有舆情数据采集必须通过此类。
    禁止在 Agent 内直接调用外部 API。
    """
    
    def __init__(
        self,
        source: SentimentDataSource = SentimentDataSource.MOCK,
        tokens: List[str] = None,
        poll_interval_sec: float = 60,
        event_bus_callback: Optional[Callable] = None,
    ):
        self.source = source
        self.tokens = tokens or ["SOL", "BTC", "ETH"]
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        
        # 内部状态
        self._last_tick: Optional[SentimentTick] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        self._total_fetches = 0
        
        # 历史情绪用于趋势计算
        self._sentiment_history: List[float] = []
        
        # 配置
        self._staleness_threshold_sec = 300
        self._extreme_fear_threshold = 20
        self._extreme_greed_threshold = 80
        
        logger.info(f"SentimentFeed initialized: source={source.value}, tokens={self.tokens}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """启动数据采集"""
        if self._running:
            return
        
        self._running = True
        logger.info("SentimentFeed started")
        
        while self._running:
            try:
                tick = await self.fetch()
                if tick and self._event_bus_callback:
                    self._event_bus_callback("SENTIMENT_TICK", tick.to_dict())
            except Exception as e:
                logger.error(f"SentimentFeed fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """停止数据采集"""
        self._running = False
        logger.info("SentimentFeed stopped")
    
    async def fetch(self) -> Optional[SentimentTick]:
        """
        获取最新舆情数据
        
        Returns:
            SentimentTick: 标准化舆情数据
        """
        self._total_fetches += 1
        
        try:
            # 从数据源获取原始数据
            raw_data = await self._fetch_from_source()
            
            # 转换为标准格式
            tick = self._parse_raw_data(raw_data)
            
            # 计算 staleness
            tick.staleness_sec = (datetime.now() - tick.timestamp).total_seconds()
            
            # 更新历史并计算趋势
            self._update_sentiment_history(tick.overall_sentiment)
            tick.sentiment_trend = self._compute_trend()
            
            # 缓存
            self._last_tick = tick
            self._last_fetch_time = datetime.now()
            
            return tick
            
        except Exception as e:
            logger.error(f"SentimentFeed fetch failed: {e}")
            self._fetch_errors += 1
            return None
    
    def get_latest(self) -> Optional[SentimentTick]:
        """获取最新缓存的数据"""
        if self._last_tick:
            self._last_tick.staleness_sec = (
                datetime.now() - self._last_tick.timestamp
            ).total_seconds()
        return self._last_tick
    
    def get_health(self) -> Dict[str, Any]:
        """获取数据健康状态"""
        tick = self.get_latest()
        
        return {
            "source": self.source.value,
            "last_fetch_time": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
            "staleness_sec": tick.staleness_sec if tick else float('inf'),
            "is_stale": tick.is_stale(self._staleness_threshold_sec) if tick else True,
            "confidence": tick.confidence if tick else 0.0,
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "error_rate": self._fetch_errors / max(1, self._total_fetches),
            "tokens_covered": self.tokens,
        }
    
    # =========================================================================
    # DATA SOURCE (生产环境需替换)
    # =========================================================================
    
    async def _fetch_from_source(self) -> Dict[str, Any]:
        """
        从数据源获取原始数据
        
        生产替换说明:
        - ALTERNATIVE_ME: 替换为 alternative.me/crypto/fear-and-greed-index/ API
        - LUNAR_CRUSH: 替换为 LunarCrush API
        - SANTIMENT: 替换为 Santiment API
        """
        if self.source == SentimentDataSource.MOCK:
            return await self._fetch_mock()
        
        if self.source == SentimentDataSource.ALTERNATIVE_ME:
            return await self._fetch_alternative_me()
        
        if self.source == SentimentDataSource.LUNAR_CRUSH:
            return await self._fetch_lunar_crush()
        
        # Fallback to mock
        logger.warning(f"Unknown source {self.source}, using mock")
        return await self._fetch_mock()
    
    async def _fetch_alternative_me(self) -> Dict[str, Any]:
        """Fetch from Alternative.me Fear & Greed API (free, no key required)."""
        try:
            async with create_session() as session:
                fg_url = "https://api.alternative.me/fng/?limit=1"
                data = await fetch_with_retry(
                    session, fg_url,
                    max_retries=3, operation="alternative_me_fg"
                )
                if data:
                    fg_data = data.get("data", [{}])[0]
                    fg_value = int(fg_data.get("value", 50))
                    fg_label = fg_data.get("value_classification", "neutral").lower().replace(" ", "_")

                    return {
                        "fear_greed": {
                            "value": fg_value,
                            "label": fg_label,
                            "timestamp": datetime.now().isoformat()
                        },
                        "social_sentiments": [],
                        "news_sentiments": [],
                        "funding_rates": [],
                        "_source": "alternative_me",
                        "_mock": False
                    }
                else:
                    logger.warning("Alternative.me API returned no data after retries")
        except Exception as e:
            logger.error(f"Alternative.me API error: {e}")

        # Fallback to mock
        return await self._fetch_mock()
    
    async def _fetch_lunar_crush(self) -> Dict[str, Any]:
        """Fetch from LunarCrush API (requires API key)."""
        api_key = getattr(self.config, 'lunar_crush_api_key', '')
        if not api_key:
            logger.debug("LunarCrush API key not configured, using mock")
            return await self._fetch_mock()
        
        try:
            async with create_session() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                
                social_sentiments = []
                for token in self.tokens:
                    url = f"https://lunarcrush.com/api3/coins/{token.lower()}/social"
                    async with session.get(url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            coin_data = data.get("data", {})
                            social_sentiments.append({
                                "platform": "twitter",
                                "token": token,
                                "sentiment_score": (coin_data.get("sentiment", 50) - 50) / 50,
                                "volume": coin_data.get("social_volume", 0),
                                "change_24h": coin_data.get("sentiment_change_24h", 0) / 100,
                                "timestamp": datetime.now().isoformat()
                            })
                
                return {
                    "fear_greed": {"value": 50, "label": "neutral", "timestamp": datetime.now().isoformat()},
                    "social_sentiments": social_sentiments,
                    "news_sentiments": [],
                    "funding_rates": [],
                    "_source": "lunar_crush",
                    "_mock": False
                }
        except Exception as e:
            logger.error(f"LunarCrush API error: {e}")
        
        return await self._fetch_mock()
    
    async def _fetch_mock(self) -> Dict[str, Any]:
        """
        Neutral-safe fallback when real sentiment feeds are unavailable.

        [PATCH-7a] Replaced random.randint/uniform with neutral constants.
        Root cause: random mock data injected noise into agent signals,
        causing non-deterministic sentiment-driven decisions on feed failure.
        """
        await asyncio.sleep(0.1)

        now = datetime.now()

        # Neutral Fear & Greed = 50 (neither fear nor greed)
        fg_value = 50
        fg_label = "neutral"

        # Neutral social sentiments (zero signal)
        social_sentiments = []
        for token in self.tokens:
            for platform in ["twitter", "reddit"]:
                social_sentiments.append({
                    "platform": platform,
                    "token": token,
                    "sentiment_score": 0.0,
                    "volume": 0,
                    "change_24h": 0.0,
                    "timestamp": now.isoformat(),
                })

        # No mock news — empty list is the honest fallback
        news_sentiments = []

        # Neutral funding rates (zero = no directional bias)
        funding_rates = []
        for token in self.tokens:
            funding_rates.append({
                "exchange": "mock",
                "symbol": f"{token}USDT",
                "rate": 0.0,
                "next_funding_time": (now + timedelta(hours=8)).isoformat(),
                "timestamp": now.isoformat(),
            })

        return {
            "timestamp": now.isoformat(),
            "fear_greed": {
                "value": fg_value,
                "label": fg_label,
                "timestamp": now.isoformat(),
            },
            "social_sentiments": social_sentiments,
            "news_sentiments": news_sentiments,
            "funding_rates": funding_rates,
            "confidence": 0.0,  # Zero confidence: downstream should ignore mock data
        }
    
    def _parse_raw_data(self, raw: Dict[str, Any]) -> SentimentTick:
        """解析原始数据为标准格式"""
        timestamp = datetime.fromisoformat(raw.get("timestamp", datetime.now().isoformat()))
        
        # 解析 Fear & Greed
        fear_greed = None
        if "fear_greed" in raw:
            fg = raw["fear_greed"]
            fear_greed = FearGreedData(
                value=fg["value"],
                label=fg["label"],
                timestamp=datetime.fromisoformat(fg["timestamp"]),
            )
        
        # 解析社交情绪
        social_sentiments = []
        for s in raw.get("social_sentiments", []):
            social_sentiments.append(SocialSentiment(
                platform=s["platform"],
                token=s["token"],
                sentiment_score=s["sentiment_score"],
                volume=s["volume"],
                change_24h=s["change_24h"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
            ))
        
        # 解析新闻情绪
        news_sentiments = []
        for n in raw.get("news_sentiments", []):
            news_sentiments.append(NewsSentiment(
                source=n["source"],
                headline=n["headline"],
                sentiment_score=n["sentiment_score"],
                relevance=n["relevance"],
                tokens_mentioned=n["tokens_mentioned"],
                timestamp=datetime.fromisoformat(n["timestamp"]),
            ))
        
        # 解析资金费率
        funding_rates = []
        for f in raw.get("funding_rates", []):
            funding_rates.append(FundingRate(
                exchange=f["exchange"],
                symbol=f["symbol"],
                rate=f["rate"],
                next_funding_time=datetime.fromisoformat(f["next_funding_time"]),
                timestamp=datetime.fromisoformat(f["timestamp"]),
            ))
        
        # 计算整体情绪
        overall_sentiment = self._compute_overall_sentiment(
            fear_greed, social_sentiments, news_sentiments, funding_rates
        )
        
        return SentimentTick(
            timestamp=timestamp,
            staleness_sec=0.0,
            confidence=raw.get("confidence", 0.8),
            fear_greed=fear_greed,
            social_sentiments=social_sentiments,
            news_sentiments=news_sentiments,
            funding_rates=funding_rates,
            overall_sentiment=overall_sentiment,
            extreme_fear=fear_greed.value < self._extreme_fear_threshold if fear_greed else False,
            extreme_greed=fear_greed.value > self._extreme_greed_threshold if fear_greed else False,
            source=self.source.value,
            tokens_covered=self.tokens,
        )
    
    def _compute_overall_sentiment(
        self,
        fear_greed: Optional[FearGreedData],
        social: List[SocialSentiment],
        news: List[NewsSentiment],
        funding: List[FundingRate],
    ) -> float:
        """计算整体情绪分数"""
        scores = []
        weights = []
        
        # Fear & Greed (权重 0.4)
        if fear_greed:
            fg_normalized = (fear_greed.value - 50) / 50  # -1 to 1
            scores.append(fg_normalized)
            weights.append(0.4)
        
        # 社交情绪 (权重 0.3)
        if social:
            social_avg = sum(s.sentiment_score for s in social) / len(social)
            scores.append(social_avg)
            weights.append(0.3)
        
        # 新闻情绪 (权重 0.2)
        if news:
            news_weighted = sum(n.sentiment_score * n.relevance for n in news)
            news_total_relevance = sum(n.relevance for n in news)
            if news_total_relevance > 0:
                scores.append(news_weighted / news_total_relevance)
                weights.append(0.2)
        
        # 资金费率 (权重 0.1)
        if funding:
            # 高正费率 = 多头过度 = 看空信号
            avg_funding = sum(f.rate for f in funding) / len(funding)
            funding_sentiment = -avg_funding * 100  # 反向并放大
            funding_sentiment = max(-1, min(1, funding_sentiment))
            scores.append(funding_sentiment)
            weights.append(0.1)
        
        if not scores:
            return 0.0
        
        # 加权平均
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(s * w for s, w in zip(scores, weights)) / total_weight
        return 0.0
    
    def _update_sentiment_history(self, sentiment: float):
        """更新情绪历史"""
        self._sentiment_history.append(sentiment)
        if len(self._sentiment_history) > 24:  # 保留最近24个数据点
            self._sentiment_history = self._sentiment_history[-24:]
    
    def _compute_trend(self) -> str:
        """计算情绪趋势"""
        if len(self._sentiment_history) < 3:
            return "neutral"
        
        recent = self._sentiment_history[-3:]
        change = recent[-1] - recent[0]
        
        if change > 0.1:
            return "improving"
        elif change < -0.1:
            return "deteriorating"
        return "stable"


# =============================================================================
# SINGLETON
# =============================================================================

_sentiment_feed_instance: Optional[SentimentFeed] = None


def get_sentiment_feed(
    source: SentimentDataSource = SentimentDataSource.MOCK,
    tokens: List[str] = None,
    event_bus_callback: Optional[Callable] = None,
) -> SentimentFeed:
    """获取或创建 SentimentFeed 单例"""
    global _sentiment_feed_instance
    if _sentiment_feed_instance is None:
        _sentiment_feed_instance = SentimentFeed(
            source=source,
            tokens=tokens,
            event_bus_callback=event_bus_callback,
        )
    return _sentiment_feed_instance


def reset_sentiment_feed():
    """重置单例"""
    global _sentiment_feed_instance
    if _sentiment_feed_instance:
        _sentiment_feed_instance.stop()
    _sentiment_feed_instance = None

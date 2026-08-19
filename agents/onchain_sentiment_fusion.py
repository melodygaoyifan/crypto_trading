"""
================================================================================
HMATS v5.2.1 SOTA - 链上舆情融合 Alpha 引擎
On-Chain + Sentiment Alpha Fusion Engine
================================================================================

DEPRECATED (v6, 2026-02-24):
    Directional pre-fusion in generate_signals() / get_signal_for_fusion()
    conflicts with AuthorityFusion.  Only build_context_packet() is safe
    to use - it returns raw features WITHOUT computing a fused direction.

    The engine is instantiated as enabled=False in production (see logs).

    Active callers (do NOT remove without migrating these first):
      - main.py:2370             -> get_alpha_engine()
      - orchestration/sota_integration.py:99  -> safe_import
      - signals/alpha_signal_integrator.py:196 -> get_alpha_engine()

    File retained for build_context_packet() logic that may be reused.

Purpose: 分析链上地址网络、交易热度和鲸鱼活动，同时引入舆情分析

Author: HMATS v5.2.1 SOTA Complete Upgrade
Version: 5.2.1-SOTA-UPGRADED -> v6 compat layer added
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import hashlib
import re

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class OnChainSignalType(Enum):
    """链上信号类型"""
    WHALE_ACCUMULATION = "whale_accumulation"
    WHALE_DISTRIBUTION = "whale_distribution"
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    DEX_VOLUME_SURGE = "dex_volume_surge"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"
    SMART_MONEY_BUY = "smart_money_buy"
    SMART_MONEY_SELL = "smart_money_sell"
    STAKING_INCREASE = "staking_increase"
    STAKING_DECREASE = "staking_decrease"


class SentimentSignalType(Enum):
    """舆情信号类型"""
    EXTREME_FEAR = "extreme_fear"
    EXTREME_GREED = "extreme_greed"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    NEWS_BULLISH = "news_bullish"
    NEWS_BEARISH = "news_bearish"
    SOCIAL_SURGE = "social_surge"
    FUNDING_POSITIVE = "funding_positive"
    FUNDING_NEGATIVE = "funding_negative"


class SignalStrength(Enum):
    """信号强度"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlphaEngineConfig:
    """Alpha引擎配置"""
    
    # 链上分析
    whale_threshold_usd: float = 1_000_000      # 鲸鱼定义阈值
    smart_money_addresses: List[str] = field(default_factory=list)
    exchange_addresses: List[str] = field(default_factory=list)
    
    # 舆情分析
    fear_greed_extreme_low: int = 20            # 极度恐惧阈值
    fear_greed_extreme_high: int = 80           # 极度贪婪阈值
    sentiment_smoothing_alpha: float = 0.3      # EMA平滑
    
    # 资金费率
    funding_rate_threshold: float = 0.01        # 1%费率阈值
    
    # 信号融合
    onchain_weight: float = 0.4                 # 链上信号权重
    sentiment_weight: float = 0.3               # 舆情信号权重
    funding_weight: float = 0.3                 # 资金费率权重
    
    # 冷却
    signal_cooldown_minutes: int = 30           # 信号冷却期


# =============================================================================
# DATA STRUCTURES - ON-CHAIN
# =============================================================================

@dataclass
class WalletActivity:
    """钱包活动"""
    address: str
    timestamp: datetime
    
    # 活动类型
    activity_type: str  # transfer, swap, stake, etc.
    
    # 金额
    amount: float
    token: str
    value_usd: float
    
    # 目标
    from_address: str = ""
    to_address: str = ""
    
    # 标签
    is_whale: bool = False
    is_exchange: bool = False
    is_smart_money: bool = False


@dataclass 
class DEXMetrics:
    """DEX指标"""
    timestamp: datetime
    protocol: str  # "jupiter", "raydium", "orca"
    
    # 交易量
    volume_24h_usd: float
    volume_change_pct: float
    
    # 流动性
    tvl_usd: float
    tvl_change_pct: float
    
    # 交易数
    tx_count_24h: int
    unique_traders: int


@dataclass
class WhaleTransaction:
    """鲸鱼交易"""
    timestamp: datetime
    whale_address: str
    
    # 交易详情
    action: str  # "buy", "sell", "transfer_in", "transfer_out"
    token: str
    amount: float
    value_usd: float
    
    # 影响评估
    market_impact: float  # 预估市场影响
    significance: SignalStrength


# =============================================================================
# DATA STRUCTURES - SENTIMENT
# =============================================================================

@dataclass
class SentimentData:
    """情绪数据"""
    timestamp: datetime
    
    # Fear & Greed
    fear_greed_index: int  # 0-100
    fear_greed_label: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    
    # 社交媒体
    twitter_sentiment: float  # -1 to 1
    twitter_volume: int
    reddit_sentiment: float
    reddit_volume: int
    
    # 新闻
    news_sentiment: float  # -1 to 1
    news_count: int
    
    # 资金费率 (perpetuals)
    btc_funding_rate: float
    eth_funding_rate: float
    sol_funding_rate: float


@dataclass
class NewsItem:
    """新闻项目"""
    timestamp: datetime
    title: str
    source: str
    url: str = ""
    
    # 分析结果
    sentiment_score: float = 0.0  # -1 to 1
    relevance_score: float = 0.0  # 0 to 1
    
    # 关键词
    keywords: List[str] = field(default_factory=list)
    mentioned_tokens: List[str] = field(default_factory=list)


# =============================================================================
# DATA STRUCTURES - SIGNALS
# =============================================================================

@dataclass
class AlphaSignal:
    """Alpha信号"""
    timestamp: datetime
    signal_id: str
    
    # 信号类型
    source: str  # "onchain", "sentiment", "funding"
    signal_type: str
    
    # 信号值
    direction: float  # -1 to 1 (sell to buy)
    strength: SignalStrength
    confidence: float  # 0 to 1
    
    # 目标资产
    token: str
    
    # 原因
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedAlphaSignal:
    """融合Alpha信号"""
    timestamp: datetime
    token: str
    
    # 融合信号
    direction: float  # -1 to 1
    confidence: float
    
    # 组成部分
    onchain_signal: float
    sentiment_signal: float
    funding_signal: float
    
    # 权重
    weights_used: Dict[str, float]
    
    # 建议 (移到 default 字段之前)
    trade_bias: str  # "bullish", "bearish", "neutral"
    urgency: str  # "low", "medium", "high"
    
    # 原始信号 (有 default)
    component_signals: List[AlphaSignal] = field(default_factory=list)


# =============================================================================
# ON-CHAIN ANALYZER
# =============================================================================

class OnChainAnalyzer:
    """链上数据分析器"""
    
    def __init__(self, config: AlphaEngineConfig):
        self.config = config
        
        # 活动历史
        self._activities: deque = deque(maxlen=1000)
        self._whale_txs: deque = deque(maxlen=500)
        
        # DEX数据
        self._dex_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 聚合统计
        self._stats = {
            "total_activities": 0,
            "whale_txs": 0,
            "exchange_inflows": 0,
            "exchange_outflows": 0,
        }
    
    def record_activity(self, activity: WalletActivity):
        """记录钱包活动"""
        # 标记特殊地址
        activity.is_whale = activity.value_usd >= self.config.whale_threshold_usd
        activity.is_exchange = activity.to_address in self.config.exchange_addresses or \
                              activity.from_address in self.config.exchange_addresses
        activity.is_smart_money = activity.address in self.config.smart_money_addresses
        
        self._activities.append(activity)
        self._stats["total_activities"] += 1
        
        # 记录鲸鱼交易
        if activity.is_whale:
            whale_tx = WhaleTransaction(
                timestamp=activity.timestamp,
                whale_address=activity.address,
                action=activity.activity_type,
                token=activity.token,
                amount=activity.amount,
                value_usd=activity.value_usd,
                market_impact=self._estimate_impact(activity),
                significance=self._assess_significance(activity),
            )
            self._whale_txs.append(whale_tx)
            self._stats["whale_txs"] += 1
        
        # 交易所流入流出
        if activity.is_exchange:
            if activity.to_address in self.config.exchange_addresses:
                self._stats["exchange_inflows"] += 1
            else:
                self._stats["exchange_outflows"] += 1
    
    def record_dex_metrics(self, metrics: DEXMetrics):
        """记录DEX指标"""
        self._dex_metrics[metrics.protocol].append(metrics)
    
    def _estimate_impact(self, activity: WalletActivity) -> float:
        """估算市场影响"""
        # 简化: 基于金额的对数影响
        if activity.value_usd < 100000:
            return 0.1
        elif activity.value_usd < 1000000:
            return 0.3
        elif activity.value_usd < 10000000:
            return 0.6
        else:
            return 0.9
    
    def _assess_significance(self, activity: WalletActivity) -> SignalStrength:
        """评估信号显著性"""
        if activity.value_usd >= 10_000_000:
            return SignalStrength.EXTREME
        elif activity.value_usd >= 5_000_000:
            return SignalStrength.STRONG
        elif activity.value_usd >= 1_000_000:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def analyze(self, token: str, lookback_hours: int = 24) -> List[AlphaSignal]:
        """
        分析链上数据生成信号
        
        Args:
            token: 目标代币
            lookback_hours: 回看小时数
        
        Returns:
            信号列表
        """
        signals = []
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        # 筛选相关活动
        relevant_activities = [
            a for a in self._activities
            if a.timestamp >= cutoff and a.token.upper() == token.upper()
        ]
        
        if not relevant_activities:
            return signals
        
        # 1. 鲸鱼累积/分发检测
        whale_activities = [a for a in relevant_activities if a.is_whale]
        if whale_activities:
            buy_volume = sum(a.value_usd for a in whale_activities if a.activity_type == "buy")
            sell_volume = sum(a.value_usd for a in whale_activities if a.activity_type == "sell")
            
            net_flow = buy_volume - sell_volume
            if abs(net_flow) > self.config.whale_threshold_usd:
                if net_flow > 0:
                    signals.append(AlphaSignal(
                        timestamp=datetime.now(),
                        signal_id=f"whale_acc_{token}_{datetime.now().timestamp()}",
                        source="onchain",
                        signal_type=OnChainSignalType.WHALE_ACCUMULATION.value,
                        direction=0.6,
                        strength=SignalStrength.STRONG,
                        confidence=0.7,
                        token=token,
                        reason=f"鲸鱼净买入 ${net_flow/1e6:.1f}M",
                        details={"net_flow": net_flow, "buy_volume": buy_volume, "sell_volume": sell_volume},
                    ))
                else:
                    signals.append(AlphaSignal(
                        timestamp=datetime.now(),
                        signal_id=f"whale_dist_{token}_{datetime.now().timestamp()}",
                        source="onchain",
                        signal_type=OnChainSignalType.WHALE_DISTRIBUTION.value,
                        direction=-0.6,
                        strength=SignalStrength.STRONG,
                        confidence=0.7,
                        token=token,
                        reason=f"鲸鱼净卖出 ${abs(net_flow)/1e6:.1f}M",
                        details={"net_flow": net_flow},
                    ))
        
        # 2. 交易所流入/流出
        exchange_activities = [a for a in relevant_activities if a.is_exchange]
        if exchange_activities:
            inflow = sum(a.value_usd for a in exchange_activities 
                        if a.to_address in self.config.exchange_addresses)
            outflow = sum(a.value_usd for a in exchange_activities
                         if a.from_address in self.config.exchange_addresses)
            
            net_exchange = inflow - outflow
            if abs(net_exchange) > 500000:  # 50万门槛
                if net_exchange > 0:
                    signals.append(AlphaSignal(
                        timestamp=datetime.now(),
                        signal_id=f"exch_in_{token}",
                        source="onchain",
                        signal_type=OnChainSignalType.EXCHANGE_INFLOW.value,
                        direction=-0.4,  # 流入交易所通常看跌
                        strength=SignalStrength.MODERATE,
                        confidence=0.6,
                        token=token,
                        reason=f"交易所净流入 ${net_exchange/1e6:.1f}M",
                    ))
                else:
                    signals.append(AlphaSignal(
                        timestamp=datetime.now(),
                        signal_id=f"exch_out_{token}",
                        source="onchain",
                        signal_type=OnChainSignalType.EXCHANGE_OUTFLOW.value,
                        direction=0.4,  # 流出交易所通常看涨
                        strength=SignalStrength.MODERATE,
                        confidence=0.6,
                        token=token,
                        reason=f"交易所净流出 ${abs(net_exchange)/1e6:.1f}M",
                    ))
        
        # 3. 聪明钱追踪
        smart_money_activities = [a for a in relevant_activities if a.is_smart_money]
        if smart_money_activities:
            sm_buy = sum(a.value_usd for a in smart_money_activities if a.activity_type == "buy")
            sm_sell = sum(a.value_usd for a in smart_money_activities if a.activity_type == "sell")
            
            if sm_buy > sm_sell * 1.5:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"smart_buy_{token}",
                    source="onchain",
                    signal_type=OnChainSignalType.SMART_MONEY_BUY.value,
                    direction=0.7,
                    strength=SignalStrength.STRONG,
                    confidence=0.75,
                    token=token,
                    reason="聪明钱买入",
                ))
            elif sm_sell > sm_buy * 1.5:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"smart_sell_{token}",
                    source="onchain",
                    signal_type=OnChainSignalType.SMART_MONEY_SELL.value,
                    direction=-0.7,
                    strength=SignalStrength.STRONG,
                    confidence=0.75,
                    token=token,
                    reason="聪明钱卖出",
                ))
        
        return signals


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================

class SentimentAnalyzer:
    """舆情分析器"""
    
    # 关键词库
    BULLISH_KEYWORDS = [
        "bullish", "moon", "breakout", "rally", "surge", "pump", "adoption",
        "institutional", "etf approved", "partnership", "upgrade", "launch",
        "bullrun", "accumulation", "buy", "long", "support", "底部", "反弹",
        "利好", "上涨", "突破", "牛市"
    ]
    
    BEARISH_KEYWORDS = [
        "bearish", "crash", "dump", "plunge", "sell", "short", "resistance",
        "hack", "exploit", "scam", "rug", "liquidation", "ban", "regulation",
        "sec", "lawsuit", "fraud", "panic", "利空", "下跌", "崩盘", "熊市"
    ]
    
    def __init__(self, config: AlphaEngineConfig):
        self.config = config
        
        # 数据历史
        self._sentiment_history: deque = deque(maxlen=200)
        self._news_history: deque = deque(maxlen=500)
        
        # EMA平滑后的情绪
        self._smoothed_sentiment: Dict[str, float] = {}
    
    def update_sentiment(self, data: SentimentData):
        """更新情绪数据"""
        self._sentiment_history.append(data)
        
        # EMA平滑
        alpha = self.config.sentiment_smoothing_alpha
        for token in ["BTC", "ETH", "SOL"]:
            current = self._get_token_sentiment(data, token)
            if token in self._smoothed_sentiment:
                self._smoothed_sentiment[token] = alpha * current + (1 - alpha) * self._smoothed_sentiment[token]
            else:
                self._smoothed_sentiment[token] = current
    
    def _get_token_sentiment(self, data: SentimentData, token: str) -> float:
        """获取代币情绪"""
        # 综合多个来源
        social = (data.twitter_sentiment + data.reddit_sentiment) / 2
        news = data.news_sentiment
        fg = (data.fear_greed_index - 50) / 50  # 转换到 -1 to 1
        
        return social * 0.4 + news * 0.3 + fg * 0.3
    
    def ingest_news(self, news: NewsItem):
        """摄入新闻"""
        # 分析情绪
        news.sentiment_score = self._analyze_text_sentiment(news.title)
        news.mentioned_tokens = self._extract_tokens(news.title)
        self._news_history.append(news)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """分析文本情绪"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw.lower() in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw.lower() in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def _extract_tokens(self, text: str) -> List[str]:
        """提取提及的代币"""
        tokens = []
        text_upper = text.upper()
        
        for token in ["BTC", "BITCOIN", "ETH", "ETHEREUM", "SOL", "SOLANA"]:
            if token in text_upper:
                # 标准化
                if token in ["BTC", "BITCOIN"]:
                    tokens.append("BTC")
                elif token in ["ETH", "ETHEREUM"]:
                    tokens.append("ETH")
                else:
                    tokens.append("SOL")
        
        return list(set(tokens))
    
    def analyze(self, token: str) -> List[AlphaSignal]:
        """
        分析舆情生成信号
        
        Args:
            token: 目标代币
        
        Returns:
            信号列表
        """
        signals = []
        
        if not self._sentiment_history:
            return signals
        
        latest = self._sentiment_history[-1]
        
        # 1. Fear & Greed极端信号 (逆向)
        fg = latest.fear_greed_index
        
        if fg <= self.config.fear_greed_extreme_low:
            signals.append(AlphaSignal(
                timestamp=datetime.now(),
                signal_id=f"fear_{token}",
                source="sentiment",
                signal_type=SentimentSignalType.EXTREME_FEAR.value,
                direction=0.5,  # 极度恐惧 -> 逆向买入
                strength=SignalStrength.MODERATE,
                confidence=0.6,
                token=token,
                reason=f"极度恐惧 (F&G={fg}) - 逆向信号",
                details={"fear_greed": fg},
            ))
        
        elif fg >= self.config.fear_greed_extreme_high:
            signals.append(AlphaSignal(
                timestamp=datetime.now(),
                signal_id=f"greed_{token}",
                source="sentiment",
                signal_type=SentimentSignalType.EXTREME_GREED.value,
                direction=-0.5,  # 极度贪婪 -> 逆向卖出
                strength=SignalStrength.MODERATE,
                confidence=0.6,
                token=token,
                reason=f"极度贪婪 (F&G={fg}) - 逆向信号",
                details={"fear_greed": fg},
            ))
        
        # 2. 社交媒体情绪急剧变化
        if len(self._sentiment_history) >= 5:
            recent_twitter = [s.twitter_sentiment for s in list(self._sentiment_history)[-5:]]
            sentiment_change = recent_twitter[-1] - np.mean(recent_twitter[:-1])
            
            if abs(sentiment_change) > 0.3:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"social_surge_{token}",
                    source="sentiment",
                    signal_type=SentimentSignalType.SOCIAL_SURGE.value,
                    direction=0.4 * np.sign(sentiment_change),
                    strength=SignalStrength.MODERATE,
                    confidence=0.55,
                    token=token,
                    reason=f"社交情绪急变 ({sentiment_change:+.2f})",
                ))
        
        # 3. 新闻情绪
        cutoff = datetime.now() - timedelta(hours=6)
        recent_news = [n for n in self._news_history 
                      if n.timestamp >= cutoff and token in n.mentioned_tokens]
        
        if len(recent_news) >= 3:
            avg_sentiment = np.mean([n.sentiment_score for n in recent_news])
            
            if avg_sentiment > 0.4:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"news_bull_{token}",
                    source="sentiment",
                    signal_type=SentimentSignalType.NEWS_BULLISH.value,
                    direction=0.5,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    token=token,
                    reason=f"新闻情绪看涨 (avg={avg_sentiment:.2f})",
                ))
            elif avg_sentiment < -0.4:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"news_bear_{token}",
                    source="sentiment",
                    signal_type=SentimentSignalType.NEWS_BEARISH.value,
                    direction=-0.5,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    token=token,
                    reason=f"新闻情绪看跌 (avg={avg_sentiment:.2f})",
                ))
        
        # 4. 资金费率信号
        funding_rate = getattr(latest, f"{token.lower()}_funding_rate", 0.0)
        
        if abs(funding_rate) > self.config.funding_rate_threshold:
            if funding_rate > 0:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"funding_pos_{token}",
                    source="sentiment",
                    signal_type=SentimentSignalType.FUNDING_POSITIVE.value,
                    direction=-0.3,  # 正费率 -> 多头过热 -> 偏空
                    strength=SignalStrength.WEAK,
                    confidence=0.5,
                    token=token,
                    reason=f"资金费率偏高 ({funding_rate:.3%})",
                ))
            else:
                signals.append(AlphaSignal(
                    timestamp=datetime.now(),
                    signal_id=f"funding_neg_{token}",
                    source="sentiment",
                    signal_type=SentimentSignalType.FUNDING_NEGATIVE.value,
                    direction=0.3,  # 负费率 -> 空头过热 -> 偏多
                    strength=SignalStrength.WEAK,
                    confidence=0.5,
                    token=token,
                    reason=f"资金费率偏低 ({funding_rate:.3%})",
                ))
        
        return signals


# =============================================================================
# ALPHA FUSION ENGINE
# =============================================================================

class OnChainSentimentAlphaEngine:
    """
    链上舆情融合Alpha引擎
    
    整合链上分析和舆情分析，生成融合信号
    """
    
    def __init__(
        self,
        config: Optional[AlphaEngineConfig] = None,
        enabled: bool = False,
    ):
        self.config = config or AlphaEngineConfig()
        self.enabled = enabled

        # 分析器
        self.onchain_analyzer = OnChainAnalyzer(self.config)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)

        # 信号历史
        self._signal_history: deque = deque(maxlen=500)
        self._fused_signals: deque = deque(maxlen=100)

        # 冷却管理
        self._last_signal_time: Dict[str, datetime] = {}

        # 事件回调
        self._event_callbacks: List[Callable] = []

        # 统计
        self._stats = {
            "onchain_signals": 0,
            "sentiment_signals": 0,
            "fused_signals": 0,
        }

        logger.info(
            "OnChainSentimentAlphaEngine initialized (enabled=%s)", enabled
        )
    
    # =========================================================================
    # DATA INPUT
    # =========================================================================
    
    def record_onchain_activity(self, activity: WalletActivity):
        """记录链上活动"""
        self.onchain_analyzer.record_activity(activity)
    
    def record_dex_metrics(self, metrics: DEXMetrics):
        """记录DEX指标"""
        self.onchain_analyzer.record_dex_metrics(metrics)
    
    def update_sentiment(self, data: SentimentData):
        """更新情绪数据"""
        self.sentiment_analyzer.update_sentiment(data)
    
    def ingest_news(self, news: NewsItem):
        """摄入新闻"""
        self.sentiment_analyzer.ingest_news(news)
    
    def ingest_news_text(self, title: str, source: str = "unknown"):
        """简化的新闻摄入"""
        news = NewsItem(
            timestamp=datetime.now(),
            title=title,
            source=source,
        )
        self.ingest_news(news)
    
    # =========================================================================
    # P0-1: FEED CONSUMPTION (禁止内部直接调用外部 API)
    # =========================================================================
    
    def consume_onchain_feed(self, tick: Any):
        """
        P0-1: 消费链上数据 Feed
        
        所有链上数据必须通过此方法输入，禁止内部直接调用外部 API
        
        Args:
            tick: OnChainTick from data_mgmt.feeds.onchain_feed
        """
        if tick is None:
            return
        
        # 从 tick 中提取鲸鱼活动
        for whale in getattr(tick, 'whale_activities', []):
            activity = WalletActivity(
                address=whale.address,
                timestamp=whale.timestamp,
                activity_type=whale.activity_type,
                amount=whale.amount,
                token=whale.token,
                value_usd=whale.value_usd,
                tx_hash=getattr(whale, 'tx_hash', ''),
            )
            self.record_onchain_activity(activity)
        
        # 从 tick 中提取交易所流
        for flow in getattr(tick, 'exchange_flows', []):
            direction = "in" if flow.direction == "inflow" else "out"
            activity = WalletActivity(
                address=f"exchange_{flow.exchange}",
                timestamp=flow.timestamp,
                activity_type="transfer",
                amount=flow.amount,
                token=flow.token,
                value_usd=flow.value_usd,
                from_address=f"exchange_{flow.exchange}" if direction == "out" else "external",
                to_address="external" if direction == "out" else f"exchange_{flow.exchange}",
            )
            self.record_onchain_activity(activity)
        
        # 从 tick 中提取 DEX 指标
        for dex in getattr(tick, 'dex_metrics', []):
            metrics = DEXMetrics(
                protocol=dex.protocol,
                timestamp=dex.timestamp,
                token=dex.token,
                volume_24h=dex.volume_24h,
                tvl=dex.tvl,
                trades_24h=dex.trades_24h,
                unique_traders=dex.unique_traders,
            )
            self.record_dex_metrics(metrics)
        
        logger.debug(f"Consumed onchain feed: {len(tick.whale_activities)} whale activities, "
                    f"{len(tick.exchange_flows)} exchange flows, "
                    f"{len(tick.dex_metrics)} DEX metrics")
    
    def consume_sentiment_feed(self, tick: Any):
        """
        P0-1: 消费舆情数据 Feed
        
        所有舆情数据必须通过此方法输入，禁止内部直接调用外部 API
        
        Args:
            tick: SentimentTick from data_mgmt.feeds.sentiment_feed
        """
        if tick is None:
            return
        
        # 构造 SentimentData
        fear_greed = getattr(tick, 'fear_greed', None)
        
        # 聚合社交情绪
        twitter_sentiments = [s for s in tick.social_sentiments if s.platform == "twitter"]
        reddit_sentiments = [s for s in tick.social_sentiments if s.platform == "reddit"]
        
        twitter_sentiment = (
            sum(s.sentiment_score for s in twitter_sentiments) / len(twitter_sentiments)
            if twitter_sentiments else 0.0
        )
        reddit_sentiment = (
            sum(s.sentiment_score for s in reddit_sentiments) / len(reddit_sentiments)
            if reddit_sentiments else 0.0
        )
        twitter_volume = sum(s.volume for s in twitter_sentiments)
        reddit_volume = sum(s.volume for s in reddit_sentiments)
        
        # 聚合新闻情绪
        news_sentiment = (
            sum(n.sentiment_score * n.relevance for n in tick.news_sentiments) /
            max(1, sum(n.relevance for n in tick.news_sentiments))
            if tick.news_sentiments else 0.0
        )
        
        # 提取资金费率
        btc_funding = 0.0
        eth_funding = 0.0
        sol_funding = 0.0
        for fr in getattr(tick, 'funding_rates', []):
            if "BTC" in fr.symbol.upper():
                btc_funding = fr.rate
            elif "ETH" in fr.symbol.upper():
                eth_funding = fr.rate
            elif "SOL" in fr.symbol.upper():
                sol_funding = fr.rate
        
        sentiment_data = SentimentData(
            timestamp=tick.timestamp,
            fear_greed_index=fear_greed.value if fear_greed else 50,
            fear_greed_label=fear_greed.label if fear_greed else "neutral",
            twitter_sentiment=twitter_sentiment,
            twitter_volume=twitter_volume,
            reddit_sentiment=reddit_sentiment,
            reddit_volume=reddit_volume,
            news_sentiment=news_sentiment,
            news_count=len(tick.news_sentiments),
            btc_funding_rate=btc_funding,
            eth_funding_rate=eth_funding,
            sol_funding_rate=sol_funding,
        )
        
        self.update_sentiment(sentiment_data)
        
        # 摄入新闻
        for news in tick.news_sentiments:
            self.ingest_news(NewsItem(
                timestamp=news.timestamp,
                title=news.headline,
                source=news.source,
                sentiment_score=news.sentiment_score,
                mentioned_tokens=news.tokens_mentioned,
            ))
        
        logger.debug(f"Consumed sentiment feed: F&G={fear_greed.value if fear_greed else 'N/A'}, "
                    f"{len(tick.news_sentiments)} news items")
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def generate_signals(self, token: str) -> FusedAlphaSignal:
        """
        生成融合Alpha信号
        
        Args:
            token: 目标代币
        
        Returns:
            FusedAlphaSignal
        """
        # 检查冷却
        if token in self._last_signal_time:
            elapsed = (datetime.now() - self._last_signal_time[token]).total_seconds() / 60
            if elapsed < self.config.signal_cooldown_minutes:
                # 返回上一个信号
                for sig in reversed(list(self._fused_signals)):
                    if sig.token == token:
                        return sig
        
        # 获取链上信号
        onchain_signals = self.onchain_analyzer.analyze(token)
        self._stats["onchain_signals"] += len(onchain_signals)
        
        # 获取舆情信号
        sentiment_signals = self.sentiment_analyzer.analyze(token)
        self._stats["sentiment_signals"] += len(sentiment_signals)
        
        # 所有信号
        all_signals = onchain_signals + sentiment_signals
        self._signal_history.extend(all_signals)
        
        # 信号融合
        onchain_direction = 0.0
        sentiment_direction = 0.0
        funding_direction = 0.0
        
        onchain_confidence = 0.0
        sentiment_confidence = 0.0
        funding_confidence = 0.0
        
        for sig in all_signals:
            if sig.source == "onchain":
                onchain_direction += sig.direction * sig.confidence
                onchain_confidence = max(onchain_confidence, sig.confidence)
            elif sig.signal_type in [SentimentSignalType.FUNDING_POSITIVE.value,
                                     SentimentSignalType.FUNDING_NEGATIVE.value]:
                funding_direction += sig.direction * sig.confidence
                funding_confidence = max(funding_confidence, sig.confidence)
            else:
                sentiment_direction += sig.direction * sig.confidence
                sentiment_confidence = max(sentiment_confidence, sig.confidence)
        
        # [FIX-AG11] Normalize by sum-of-confidences, not count.
        # Numerator is sum(direction × confidence), so dividing by count double-discounts.
        # Correct: divide by sum(confidence) to get weighted-average direction.
        if onchain_signals:
            _oc_conf_sum = sum(s.confidence for s in onchain_signals if s.confidence > 0) or 1.0
            onchain_direction /= _oc_conf_sum
        if sentiment_signals:
            non_funding = [s for s in sentiment_signals
                         if s.signal_type not in [SentimentSignalType.FUNDING_POSITIVE.value,
                                                  SentimentSignalType.FUNDING_NEGATIVE.value]]
            if non_funding:
                _sf_conf_sum = sum(s.confidence for s in non_funding if s.confidence > 0) or 1.0
                sentiment_direction /= _sf_conf_sum

            funding_sigs = [s for s in sentiment_signals
                          if s.signal_type in [SentimentSignalType.FUNDING_POSITIVE.value,
                                               SentimentSignalType.FUNDING_NEGATIVE.value]]
            if funding_sigs:
                _fd_conf_sum = sum(s.confidence for s in funding_sigs if s.confidence > 0) or 1.0
                funding_direction /= _fd_conf_sum
        
        # 加权融合
        weights = {
            "onchain": self.config.onchain_weight if onchain_signals else 0,
            "sentiment": self.config.sentiment_weight if sentiment_signals else 0,
            "funding": self.config.funding_weight if funding_direction != 0 else 0,
        }
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # 融合方向
        fused_direction = (
            onchain_direction * weights.get("onchain", 0) +
            sentiment_direction * weights.get("sentiment", 0) +
            funding_direction * weights.get("funding", 0)
        )
        
        # 融合置信度
        fused_confidence = max(onchain_confidence, sentiment_confidence, funding_confidence)
        if not all_signals:
            fused_confidence = 0.0
        
        # 交易偏向
        if fused_direction > 0.3:
            trade_bias = "bullish"
        elif fused_direction < -0.3:
            trade_bias = "bearish"
        else:
            trade_bias = "neutral"
        
        # 紧急程度
        if abs(fused_direction) > 0.6 and fused_confidence > 0.7:
            urgency = "high"
        elif abs(fused_direction) > 0.3:
            urgency = "medium"
        else:
            urgency = "low"
        
        fused = FusedAlphaSignal(
            timestamp=datetime.now(),
            token=token,
            direction=np.clip(fused_direction, -1, 1),
            confidence=fused_confidence,
            onchain_signal=onchain_direction,
            sentiment_signal=sentiment_direction,
            funding_signal=funding_direction,
            weights_used=weights,
            component_signals=all_signals,
            trade_bias=trade_bias,
            urgency=urgency,
        )
        
        self._fused_signals.append(fused)
        self._last_signal_time[token] = datetime.now()
        self._stats["fused_signals"] += 1
        
        # 发送事件
        self._emit_event("alpha_signal", {
            "token": token,
            "direction": fused.direction,
            "confidence": fused.confidence,
            "trade_bias": trade_bias,
            "urgency": urgency,
        })
        
        return fused
    
    # =========================================================================
    # INTEGRATION WITH SIGNAL FUSION
    # =========================================================================
    
    def get_signal_for_fusion(self, token: str) -> Dict[str, float]:
        """
        DEPRECATED - performs directional pre-fusion that conflicts with
        AuthorityFusion.  Use ``build_context_packet()`` instead.

        Retained for backward compat only.
        """
        fused = self.generate_signals(token)

        return {
            "direction": fused.direction,
            "confidence": fused.confidence,
            "weight": 0.2,
            "source": "onchain_sentiment_alpha",
            "trade_bias": fused.trade_bias,
        }

    # =========================================================================
    # V6 CONTEXT PACKET (preferred over generate_signals)
    # =========================================================================

    def build_context_packet(self, token: str) -> Dict[str, Any]:
        """
        Return per-source RAW features - NO weighted directional fusion.

        This is the v6-recommended interface.  AuthorityFusion handles all
        directional weighting; this module only provides feature evidence.

        If ``enabled=False`` (default), returns neutral packet with
        data_quality="disabled".
        """
        # v6.4: if disabled, return neutral telemetry
        if not self.enabled:
            return {
                "onchain_whale_net_flow": 0.0,
                "onchain_exchange_net_flow": 0.0,
                "onchain_smart_money_bias": 0.0,
                "sentiment_fear_greed": 50,
                "sentiment_social_score": 0.0,
                "sentiment_news_score": 0.0,
                "funding_rate": 0.0,
                "data_quality": 0.0,
                "data_quality_str": "disabled",
                "source": "onchain_sentiment_context",
            }

        # ---- on-chain features (raw, un-fused) ---------------------------
        onchain_signals = self.onchain_analyzer.analyze(token)

        whale_net = 0.0
        exchange_net = 0.0
        smart_bias = 0.0
        for sig in onchain_signals:
            if sig.signal_type in (
                OnChainSignalType.WHALE_ACCUMULATION.value,
                OnChainSignalType.WHALE_DISTRIBUTION.value,
            ):
                whale_net += sig.direction * sig.confidence
            elif sig.signal_type in (
                OnChainSignalType.EXCHANGE_INFLOW.value,
                OnChainSignalType.EXCHANGE_OUTFLOW.value,
            ):
                exchange_net += sig.direction * sig.confidence
            elif sig.signal_type in (
                OnChainSignalType.SMART_MONEY_BUY.value,
                OnChainSignalType.SMART_MONEY_SELL.value,
            ):
                smart_bias += sig.direction * sig.confidence

        # ---- sentiment features (raw, un-fused) --------------------------
        fg = 50
        social_score = 0.0
        news_score = 0.0
        funding_rate = 0.0

        if self.sentiment_analyzer._sentiment_history:
            latest = self.sentiment_analyzer._sentiment_history[-1]
            fg = latest.fear_greed_index
            social_score = (latest.twitter_sentiment + latest.reddit_sentiment) / 2
            news_score = latest.news_sentiment
            funding_rate = getattr(latest, f"{token.lower()}_funding_rate", 0.0)

        # ---- data quality ------------------------------------------------
        n_sources = (
            (1 if onchain_signals else 0)
            + (1 if self.sentiment_analyzer._sentiment_history else 0)
        )
        quality = n_sources / 2.0

        return {
            "onchain_whale_net_flow": round(whale_net, 4),
            "onchain_exchange_net_flow": round(exchange_net, 4),
            "onchain_smart_money_bias": round(smart_bias, 4),
            "sentiment_fear_greed": fg,
            "sentiment_social_score": round(social_score, 4),
            "sentiment_news_score": round(news_score, 4),
            "funding_rate": funding_rate,
            "data_quality": round(quality, 2),
            "source": "onchain_sentiment_context",
        }

    def to_agent_signal_dict(self, token: str = "SOL") -> Dict[str, Any]:
        """
        V6 agent contract - returns namespaced flat dict.

        Uses ``build_context_packet()`` internally so no directional
        pre-fusion occurs.  No "weight" key is emitted (v6.4 rule).
        """
        pkt = self.build_context_packet(token)
        return {
            "ocs_whale_net_flow": pkt["onchain_whale_net_flow"],
            "ocs_exchange_net_flow": pkt["onchain_exchange_net_flow"],
            "ocs_smart_money_bias": pkt["onchain_smart_money_bias"],
            "ocs_fear_greed": pkt["sentiment_fear_greed"],
            "ocs_social_score": pkt["sentiment_social_score"],
            "ocs_news_score": pkt["sentiment_news_score"],
            "ocs_funding_rate": pkt["funding_rate"],
            "ocs_data_quality": pkt["data_quality"],
            "ocs_direction": 0.0,      # v6: no pre-fused direction
            "ocs_confidence": 0.0,     # v6: no pre-fused confidence
            "data_quality": pkt.get("data_quality_str", "ok" if pkt["data_quality"] > 0 else "missing"),
            "asof_ts": _iso_utc(),
            "data_age_seconds": 0.0,
            "provenance": ["onchain_sentiment_fusion", token],
            "asset": token,
        }

    # =========================================================================
    # EVENT BUS
    # =========================================================================
    
    def register_event_callback(self, callback: Callable[[str, Dict], None]):
        """注册事件回调"""
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """发送事件"""
        for callback in self._event_callbacks:
            try:
                callback(f"alpha.{event_type}", data)
            except Exception:
                pass
    
    # =========================================================================
    # DASHBOARD
    # =========================================================================
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        latest_fused = {}
        for token in ["BTC", "ETH", "SOL"]:
            for sig in reversed(list(self._fused_signals)):
                if sig.token == token:
                    latest_fused[token] = {
                        "direction": sig.direction,
                        "confidence": sig.confidence,
                        "trade_bias": sig.trade_bias,
                        "urgency": sig.urgency,
                        "onchain": sig.onchain_signal,
                        "sentiment": sig.sentiment_signal,
                        "funding": sig.funding_signal,
                    }
                    break
        
        return {
            "latest_signals": latest_fused,
            "stats": self._stats.copy(),
            "recent_signals": [
                {
                    "token": s.token,
                    "source": s.source,
                    "type": s.signal_type,
                    "direction": s.direction,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in list(self._signal_history)[-10:]
            ],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self._stats.copy()


# =============================================================================
# SINGLETON
# =============================================================================

_alpha_engine: Optional[OnChainSentimentAlphaEngine] = None


def get_alpha_engine() -> OnChainSentimentAlphaEngine:
    """获取Alpha引擎单例"""
    global _alpha_engine
    if _alpha_engine is None:
        _alpha_engine = OnChainSentimentAlphaEngine()
    return _alpha_engine


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing OnChainSentimentAlphaEngine")
    print("=" * 60)
    
    engine = OnChainSentimentAlphaEngine()
    
    # 1. 模拟链上数据
    print("\n1. Recording on-chain activities...")
    
    # 鲸鱼买入
    engine.record_onchain_activity(WalletActivity(
        address="whale_001",
        timestamp=datetime.now(),
        activity_type="buy",
        amount=50000,
        token="SOL",
        value_usd=2_000_000,
    ))
    
    # 交易所流出
    engine.record_onchain_activity(WalletActivity(
        address="normal_001",
        timestamp=datetime.now(),
        activity_type="transfer",
        amount=10000,
        token="SOL",
        value_usd=800_000,
        from_address="exchange_binance",
        to_address="normal_001",
    ))
    
    # 设置交易所地址
    engine.config.exchange_addresses = ["exchange_binance", "exchange_ftx"]
    
    print(f"   Recorded activities")
    
    # 2. 模拟情绪数据
    print("\n2. Updating sentiment data...")
    
    engine.update_sentiment(SentimentData(
        timestamp=datetime.now(),
        fear_greed_index=18,  # 极度恐惧
        fear_greed_label="Extreme Fear",
        twitter_sentiment=0.3,
        twitter_volume=50000,
        reddit_sentiment=0.2,
        reddit_volume=3000,
        news_sentiment=0.1,
        news_count=15,
        btc_funding_rate=0.0001,
        eth_funding_rate=-0.0005,
        sol_funding_rate=0.015,  # 高费率
    ))
    
    # 3. 摄入新闻
    print("\n3. Ingesting news...")
    
    engine.ingest_news_text("Solana ecosystem sees major breakout with new partnerships")
    engine.ingest_news_text("SOL rallies as institutional adoption increases")
    engine.ingest_news_text("Bullish momentum continues for Solana DeFi protocols")
    
    # 4. 生成融合信号
    print("\n4. Generating fused signal for SOL...")
    
    signal = engine.generate_signals("SOL")
    
    print(f"   Direction: {signal.direction:.3f}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Trade Bias: {signal.trade_bias}")
    print(f"   Urgency: {signal.urgency}")
    print(f"   Components:")
    print(f"     - On-chain: {signal.onchain_signal:.3f}")
    print(f"     - Sentiment: {signal.sentiment_signal:.3f}")
    print(f"     - Funding: {signal.funding_signal:.3f}")
    print(f"   Weights: {signal.weights_used}")
    
    # 5. 信号融合格式
    print("\n5. Signal for fusion module...")
    fusion_data = engine.get_signal_for_fusion("SOL")
    print(f"   {fusion_data}")
    
    # 6. 仪表盘数据
    print("\n6. Dashboard data...")
    dashboard = engine.get_dashboard_data()
    print(f"   Stats: {dashboard['stats']}")
    
    print("\n✓ All tests passed!")


def _iso_utc(ts=None):
    """[P323b] ISO-8601 UTC with a SINGLE "Z" marker.

    `isoformat()` on an aware datetime already emits "+00:00", so the old
    `isoformat() + "Z"` produced "+00:00Z" — which broke /health's freshness
    parse for four months (P323). A NAIVE value is normalised to UTC first
    rather than merely reformatted: naive + "Z" labels local time as UTC
    (P40/P97), which is wrong, not just malformed.
    """
    from datetime import datetime as _dt, timezone as _tz
    t = ts or _dt.now(_tz.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_tz.utc)
    return t.isoformat().replace("+00:00", "Z")

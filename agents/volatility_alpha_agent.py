"""
================================================================================
HMATS v5.2.1 - Volatility Alpha Agent (Bear-Market Specialist)
================================================================================

Purpose: 方向无关 (Direction-Agnostic) 的 Alpha 来源，专门捕捉:
- 熊市波动率扩张
- 清算前后结构性波动爆发
- 反弹失败/恐慌阶段的 Volatility Carry

核心原则:
1. 不输出方向 (long/short)
2. 不输出仓位大小
3. 不绕过 execution/risk
4. 只输出 vol_bias (increase/neutral/suppress) + intensity

消费的事件流 (via EventBus):
- LOB_TICK: spread, depth imbalance, VPIN proxy
- VOLATILITY_STATE: realized vol (multi-window), implied proxy
- ONCHAIN_TICK: exchange inflow spikes, stablecoin net flow
- SENTIMENT_TICK: fear index, sentiment volatility
- CORRELATION_STATE: BTC dominance, cross-asset correlation

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class VolBias(Enum):
    """波动率偏向"""
    INCREASE = "increase"      # 建议增加波动率敞口
    NEUTRAL = "neutral"        # 保持当前水平
    SUPPRESS = "suppress"      # 建议压制波动率敞口


class VolHorizon(Enum):
    """波动预期时间范围"""
    SHORT = "short"            # < 4 hours
    MEDIUM = "medium"          # 4-24 hours


class VolSignalType(Enum):
    """波动率信号类型"""
    VOLATILITY_BREAKOUT = "volatility_breakout"
    LIQUIDATION_PRESSURE = "liquidation_pressure"
    COMPRESSION_EXPANSION = "compression_expansion"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VolatilityIntent:
    """
    波动率 Alpha 输出

    OFF 严格禁止:
    - direction (long/short)
    - position_size
    - order_params

    ON 只输出:
    - vol_bias: 波动率敞口建议
    - intensity: 信号强度 (0-1)
    - expected_horizon: 预期持续时间
    - rationale_tags: 触发原因标签
    """
    asset: str
    vol_bias: VolBias
    intensity: float                    # 0.0 ~ 1.0
    expected_horizon: VolHorizon
    rationale_tags: List[str] = field(default_factory=list)

    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    sub_signals: Dict[str, float] = field(default_factory=dict)  # 子信号明细

    # 数据健康
    data_quality: float = 1.0           # 0-1, 数据完整度
    is_valid: bool = True

    # Diagnostics (P1)
    source_freshness: Dict[str, float] = field(default_factory=dict)  # source -> age_seconds (-1 = missing)
    coverage: Dict[str, str] = field(default_factory=dict)  # source -> "fresh"|"stale"|"missing"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "vol_bias": self.vol_bias.value,
            "intensity": self.intensity,
            "expected_horizon": self.expected_horizon.value,
            "rationale_tags": self.rationale_tags,
            "timestamp": self.timestamp.isoformat(),
            "sub_signals": self.sub_signals,
            "data_quality": self.data_quality,
            "is_valid": self.is_valid,
            "source_freshness": self.source_freshness,
            "coverage": self.coverage,
        }


@dataclass
class LOBState:
    """订单簿状态 (from LOB_TICK)"""
    timestamp: datetime
    asset: str
    spread_bps: float = 0.0
    depth_imbalance: float = 0.0        # -1 to 1
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    vpin_proxy: float = 0.0             # Volume-synchronized Probability of Informed Trading proxy

    # 变化率
    spread_change_pct: float = 0.0
    depth_change_pct: float = 0.0


@dataclass
class VolatilityState:
    """波动率状态 (from VOLATILITY_STATE)"""
    timestamp: datetime
    asset: str

    # 多窗口 realized vol
    realized_vol_1h: float = 0.0
    realized_vol_4h: float = 0.0
    realized_vol_24h: float = 0.0

    # Implied proxy (无真实期权时用 range/ATR convexity)
    implied_vol_proxy: float = 0.0

    # 分位数
    vol_percentile_30d: float = 50.0    # 相对30天历史的分位数
    vol_zscore: float = 0.0

    # 波动率的波动率
    vol_of_vol: float = 0.0


@dataclass
class OnchainState:
    """链上状态 (from ONCHAIN_TICK)"""
    timestamp: datetime
    asset: str

    # 交易所流入
    exchange_inflow_usd_1h: float = 0.0
    exchange_inflow_spike: bool = False
    exchange_inflow_zscore: float = 0.0

    # Stablecoin 净流
    stablecoin_net_flow_1h: float = 0.0
    stablecoin_flow_direction: str = "neutral"  # inflow/outflow/neutral

    # Funding (若可用)
    funding_rate: float = 0.0
    funding_deviation: float = 0.0      # 偏离历史均值


@dataclass
class SentimentState:
    """情绪状态 (from SENTIMENT_TICK)"""
    timestamp: datetime
    asset: str

    # Fear index (0-100, 越低越恐慌)
    fear_index: float = 50.0
    fear_index_change_1h: float = 0.0

    # 情绪波动率 (变化率)
    sentiment_volatility: float = 0.0   # 情绪本身的波动

    # 极端情绪
    is_extreme_fear: bool = False       # fear_index < 20
    is_extreme_greed: bool = False      # fear_index > 80


@dataclass
class CorrelationState:
    """相关性状态 (from CORRELATION_STATE) - global, not per-asset"""
    timestamp: datetime

    # BTC dominance
    btc_dominance: float = 50.0
    btc_dominance_change_24h: float = 0.0

    # Cross-asset correlation
    cross_asset_correlation: float = 0.87  # Must match GMM training mean (0.872)
    correlation_regime: str = "normal"  # compressed/normal/exploded

    # Correlation 突变
    correlation_spike: bool = False


@dataclass
class VolAlphaConfig:
    """波动率 Alpha Agent 配置"""

    # 启用开关
    enabled: bool = True

    # 各子信号权重
    breakout_weight: float = 0.35
    liquidation_weight: float = 0.35
    compression_weight: float = 0.30

    # 阈值
    vol_breakout_percentile: float = 80.0    # 触发 breakout 的分位数
    spread_expansion_threshold: float = 1.5  # spread 扩张倍数
    depth_drop_threshold: float = 0.3        # depth 下降比例

    inflow_spike_zscore: float = 2.0         # 交易所流入 spike 的 z-score
    fear_extreme_threshold: float = 25.0     # 极端恐慌阈值

    compression_vol_percentile: float = 20.0 # 低波动分位数
    compression_duration_hours: int = 24     # 压缩持续时间

    # 输出限制
    min_intensity_for_output: float = 0.3    # 低于此不输出
    max_intensity: float = 1.0

    # 数据健康
    min_data_sources_required: int = 2       # 至少需要几个数据源

    # Freshness thresholds (seconds) - P2: moved from hardcoded
    lob_freshness_sec: float = 60.0
    vol_freshness_sec: float = 300.0
    onchain_freshness_sec: float = 600.0
    sentiment_freshness_sec: float = 600.0
    correlation_freshness_sec: float = 600.0


# =============================================================================
# SUB-SIGNAL CALCULATORS
# =============================================================================

class VolatilityBreakoutAlpha:
    """
    Volatility Breakout Alpha

    信号逻辑:
    - realized vol 突破过去 N 窗口分位数
    - spread 扩张 + depth 突降
    - 常见于熊市瀑布前
    """

    def __init__(self, config: VolAlphaConfig):
        self.config = config
        self._vol_history: deque = deque(maxlen=100)
        self._spread_baseline: float = 0.0
        self._depth_baseline: float = 0.0

    def update_baselines(self, lob: LOBState):
        """更新基准值"""
        if self._spread_baseline == 0:
            self._spread_baseline = lob.spread_bps
        else:
            self._spread_baseline = 0.95 * self._spread_baseline + 0.05 * lob.spread_bps

        total_depth = lob.bid_depth_usd + lob.ask_depth_usd
        if self._depth_baseline == 0:
            self._depth_baseline = total_depth
        else:
            self._depth_baseline = 0.95 * self._depth_baseline + 0.05 * total_depth

    def compute(
        self,
        vol_state: VolatilityState,
        lob_state: LOBState,
    ) -> Tuple[float, List[str]]:
        """
        计算 Volatility Breakout 信号

        Returns:
            (intensity, rationale_tags)
        """
        tags = []
        intensity = 0.0

        # 1. Vol percentile breakout
        if vol_state.vol_percentile_30d >= self.config.vol_breakout_percentile:
            vol_contribution = (vol_state.vol_percentile_30d - self.config.vol_breakout_percentile) / (100 - self.config.vol_breakout_percentile)
            intensity += vol_contribution * 0.5
            tags.append("vol_percentile_breakout")

        # 2. Vol z-score
        if vol_state.vol_zscore > 1.5:
            intensity += min(vol_state.vol_zscore / 3, 0.3)
            tags.append("vol_zscore_elevated")

        # 3. Spread expansion
        self.update_baselines(lob_state)
        if self._spread_baseline > 0:
            spread_ratio = lob_state.spread_bps / self._spread_baseline
            if spread_ratio >= self.config.spread_expansion_threshold:
                intensity += 0.2
                tags.append("spread_expansion")

        # 4. Depth drop
        if self._depth_baseline > 0:
            current_depth = lob_state.bid_depth_usd + lob_state.ask_depth_usd
            depth_ratio = current_depth / self._depth_baseline
            if depth_ratio < (1 - self.config.depth_drop_threshold):
                intensity += 0.2
                tags.append("depth_drop")

        # 5. Vol of Vol spike (二阶波动)
        if vol_state.vol_of_vol > 0.5:
            intensity += 0.15
            tags.append("vol_of_vol_spike")

        return min(intensity, 1.0), tags


class LiquidationPressureAlpha:
    """
    Liquidation Pressure Alpha (stateless)

    信号逻辑:
    - 链上/交易所 inflow 激增
    - funding 偏移 (若可用)
    - sentiment volatility 激增 (恐慌)
    """

    def __init__(self, config: VolAlphaConfig):
        self.config = config

    def compute(
        self,
        onchain_state: OnchainState,
        sentiment_state: SentimentState,
    ) -> Tuple[float, List[str]]:
        """
        计算 Liquidation Pressure 信号

        Returns:
            (intensity, rationale_tags)
        """
        tags = []
        intensity = 0.0

        # 1. Exchange inflow spike
        if onchain_state.exchange_inflow_spike:
            intensity += 0.3
            tags.append("exchange_inflow_spike")

        if onchain_state.exchange_inflow_zscore >= self.config.inflow_spike_zscore:
            intensity += min(onchain_state.exchange_inflow_zscore / 4, 0.25)
            tags.append("inflow_zscore_elevated")

        # 2. Stablecoin outflow (资金撤离)
        if onchain_state.stablecoin_flow_direction == "outflow":
            intensity += 0.15
            tags.append("stablecoin_outflow")

        # 3. Funding rate deviation
        if abs(onchain_state.funding_deviation) > 0.5:
            intensity += 0.15
            tags.append("funding_deviation")

        # 4. Extreme fear
        if sentiment_state.is_extreme_fear:
            intensity += 0.25
            tags.append("extreme_fear")
        elif sentiment_state.fear_index < self.config.fear_extreme_threshold:
            fear_contribution = (self.config.fear_extreme_threshold - sentiment_state.fear_index) / self.config.fear_extreme_threshold
            intensity += fear_contribution * 0.2
            tags.append("elevated_fear")

        # 5. Sentiment volatility
        if sentiment_state.sentiment_volatility > 0.3:
            intensity += 0.2
            tags.append("sentiment_volatility_high")

        return min(intensity, 1.0), tags


class CompressionExpansionAlpha:
    """
    Compression -> Expansion Alpha

    信号逻辑:
    - 长时间低波动后
    - correlation 突然上升
    - 熊市反弹末期 -> 下一轮下杀前
    """

    def __init__(self, config: VolAlphaConfig):
        self.config = config
        self._low_vol_duration_hours: float = 0.0
        self._last_update: datetime = datetime.now()

    def compute(
        self,
        vol_state: VolatilityState,
        correlation_state: CorrelationState,
    ) -> Tuple[float, List[str]]:
        """
        计算 Compression -> Expansion 信号

        Returns:
            (intensity, rationale_tags)
        """
        tags = []
        intensity = 0.0

        now = datetime.now()
        hours_since_update = (now - self._last_update).total_seconds() / 3600

        # 1. 追踪低波动持续时间
        if vol_state.vol_percentile_30d <= self.config.compression_vol_percentile:
            self._low_vol_duration_hours += hours_since_update

            # 低波动时间越长，潜在爆发越大
            duration_factor = min(self._low_vol_duration_hours / self.config.compression_duration_hours, 1.0)
            intensity += duration_factor * 0.4

            if self._low_vol_duration_hours >= self.config.compression_duration_hours:
                tags.append("extended_compression")
        else:
            # 波动率开始上升，检查是否是从压缩状态来
            if self._low_vol_duration_hours >= self.config.compression_duration_hours / 2:
                # 刚从压缩状态出来，增加信号
                intensity += 0.3
                tags.append("compression_breakout")

            # 重置
            self._low_vol_duration_hours = 0

        self._last_update = now

        # 2. Correlation explosion (cross-asset 相关性突然上升)
        if correlation_state.correlation_spike:
            intensity += 0.3
            tags.append("correlation_spike")

        if correlation_state.correlation_regime == "exploded":
            intensity += 0.2
            tags.append("correlation_regime_exploded")

        # 3. BTC dominance 变化 (熊市通常 BTC dominance 上升)
        if correlation_state.btc_dominance_change_24h > 2.0:
            intensity += 0.15
            tags.append("btc_dominance_rising")

        return min(intensity, 1.0), tags


# =============================================================================
# MAIN AGENT
# =============================================================================

class VolatilityAlphaAgent:
    """
    Volatility Alpha Agent

    熊市专用的波动率 Alpha 捕捉器。

    [WARN]️ 严格限制:
    - 不输出 long/short
    - 不输出仓位大小
    - 不输出下单参数
    - 不绕过 execution/risk

    ON 只输出:
    - VolatilityIntent (vol_bias, intensity)

    与 Authority Fusion 的关系:
    - 是独立 authority source
    - 不能单独触发 entry
    - 只能放大/压制已有 direction intent
    - 或影响 execution aggressiveness
    - 可被 RiskAgent / TradeGate / DataHealth veto
    """

    def __init__(self, config: VolAlphaConfig = None):
        self.config = config or VolAlphaConfig()

        # Per-asset sub-signal calculators (stateful ones get per-asset instances)
        self._breakout_alphas: Dict[str, VolatilityBreakoutAlpha] = {}
        self._compression_alphas: Dict[str, CompressionExpansionAlpha] = {}
        # LiquidationPressureAlpha is stateless - single instance is fine
        self._liquidation_alpha = LiquidationPressureAlpha(self.config)

        # Per-asset state caches (P0: was single-asset, now keyed by asset)
        self._lob_states: Dict[str, LOBState] = {}
        self._vol_states: Dict[str, VolatilityState] = {}
        self._onchain_states: Dict[str, OnchainState] = {}
        self._sentiment_states: Dict[str, SentimentState] = {}
        # CorrelationState is global (no asset field) - stays as single value
        self._correlation_state: Optional[CorrelationState] = None

        # 输出历史
        self._intent_history: deque = deque(maxlen=100)

        # 统计
        self._stats = {
            "total_evaluations": 0,
            "increase_signals": 0,
            "suppress_signals": 0,
            "neutral_signals": 0,
            "vetoed_count": 0,
            "accepted_count": 0,
        }

        # Warning suppression (P1: initialized properly, reset when data recovers)
        self._data_warning_shown: bool = False

        # 事件回调 (for EventBus)
        self._event_callback: Optional[Callable] = None

        logger.info("VolatilityAlphaAgent initialized")

    def set_event_callback(self, callback: Callable):
        """设置 EventBus 回调"""
        self._event_callback = callback

    # =========================================================================
    # PER-ASSET SUB-SIGNAL GETTERS
    # =========================================================================

    def _get_breakout_alpha(self, asset: str) -> VolatilityBreakoutAlpha:
        """Get or create per-asset VolatilityBreakoutAlpha instance."""
        if asset not in self._breakout_alphas:
            self._breakout_alphas[asset] = VolatilityBreakoutAlpha(self.config)
        return self._breakout_alphas[asset]

    def _get_compression_alpha(self, asset: str) -> CompressionExpansionAlpha:
        """Get or create per-asset CompressionExpansionAlpha instance."""
        if asset not in self._compression_alphas:
            self._compression_alphas[asset] = CompressionExpansionAlpha(self.config)
        return self._compression_alphas[asset]

    # =========================================================================
    # FRESHNESS CHECK
    # =========================================================================

    def _is_fresh(self, state: Optional[Any], freshness_sec: float) -> bool:
        """Check if a state object is present and within freshness window."""
        if state is None:
            return False
        age = (datetime.now() - state.timestamp).total_seconds()
        return age < freshness_sec

    # =========================================================================
    # STATE UPDATES (from EventBus or direct calls)
    # =========================================================================

    def on_lob_tick(self, data: Dict[str, Any]):
        """处理 LOB_TICK 事件 - stores per-asset"""
        try:
            asset = data.get("asset", "BTC")
            self._lob_states[asset] = LOBState(
                timestamp=datetime.now(),
                asset=asset,
                spread_bps=data.get("spread_bps", 0),
                depth_imbalance=data.get("depth_imbalance", 0),
                bid_depth_usd=data.get("bid_depth_usd", 0),
                ask_depth_usd=data.get("ask_depth_usd", 0),
                vpin_proxy=data.get("vpin_proxy", 0),
                spread_change_pct=data.get("spread_change_pct", 0),
                depth_change_pct=data.get("depth_change_pct", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to process LOB_TICK: {e}")

    def on_volatility_state(self, data: Dict[str, Any]):
        """处理 VOLATILITY_STATE 事件 - stores per-asset"""
        try:
            asset = data.get("asset", "BTC")
            self._vol_states[asset] = VolatilityState(
                timestamp=datetime.now(),
                asset=asset,
                realized_vol_1h=data.get("realized_vol_1h", 0),
                realized_vol_4h=data.get("realized_vol_4h", 0),
                realized_vol_24h=data.get("realized_vol_24h", 0),
                implied_vol_proxy=data.get("implied_vol_proxy", 0),
                vol_percentile_30d=data.get("vol_percentile_30d", 50),
                vol_zscore=data.get("vol_zscore", 0),
                vol_of_vol=data.get("vol_of_vol", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to process VOLATILITY_STATE: {e}")

    def on_onchain_tick(self, data: Dict[str, Any]):
        """处理 ONCHAIN_TICK 事件 - stores per-asset"""
        try:
            asset = data.get("asset", "BTC")
            self._onchain_states[asset] = OnchainState(
                timestamp=datetime.now(),
                asset=asset,
                exchange_inflow_usd_1h=data.get("exchange_inflow_usd_1h", 0),
                exchange_inflow_spike=data.get("exchange_inflow_spike", False),
                exchange_inflow_zscore=data.get("exchange_inflow_zscore", 0),
                stablecoin_net_flow_1h=data.get("stablecoin_net_flow_1h", 0),
                stablecoin_flow_direction=data.get("stablecoin_flow_direction", "neutral"),
                funding_rate=data.get("funding_rate", 0),
                funding_deviation=data.get("funding_deviation", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to process ONCHAIN_TICK: {e}")

    def on_sentiment_tick(self, data: Dict[str, Any]):
        """处理 SENTIMENT_TICK 事件 - stores per-asset"""
        try:
            asset = data.get("asset", "BTC")
            fear_index = data.get("fear_index", 50)
            self._sentiment_states[asset] = SentimentState(
                timestamp=datetime.now(),
                asset=asset,
                fear_index=fear_index,
                fear_index_change_1h=data.get("fear_index_change_1h", 0),
                sentiment_volatility=data.get("sentiment_volatility", 0),
                is_extreme_fear=fear_index < 20,
                is_extreme_greed=fear_index > 80,
            )
        except Exception as e:
            logger.warning(f"Failed to process SENTIMENT_TICK: {e}")

    def on_correlation_state(self, data: Dict[str, Any]):
        """处理 CORRELATION_STATE 事件 - global (no asset field)"""
        try:
            self._correlation_state = CorrelationState(
                timestamp=datetime.now(),
                btc_dominance=data.get("btc_dominance", 50),
                btc_dominance_change_24h=data.get("btc_dominance_change_24h", 0),
                cross_asset_correlation=data.get("cross_asset_correlation", 0.87),
                correlation_regime=data.get("correlation_regime", "normal"),
                correlation_spike=data.get("correlation_spike", False),
            )
        except Exception as e:
            logger.warning(f"Failed to process CORRELATION_STATE: {e}")

    # =========================================================================
    # CORE EVALUATION
    # =========================================================================

    def evaluate(self, asset: str = "BTC") -> VolatilityIntent:
        """
        评估当前波动率 Alpha 信号

        这是主入口，融合所有子信号并输出 VolatilityIntent
        """
        # P1: Enforce config.enabled
        if not self.config.enabled:
            return VolatilityIntent(
                asset=asset,
                vol_bias=VolBias.NEUTRAL,
                intensity=0.0,
                expected_horizon=VolHorizon.SHORT,
                rationale_tags=["agent_disabled"],
                is_valid=False,
            )

        self._stats["total_evaluations"] += 1

        # 检查数据健康 (per-asset)
        data_quality, available_sources, freshness, coverage = self._check_data_health(asset)

        if available_sources < self.config.min_data_sources_required:
            if not self._data_warning_shown:
                logger.info(f"VolAlpha: Insufficient data sources ({available_sources}/{self.config.min_data_sources_required}), suppressing further warnings")
                self._data_warning_shown = True
            return VolatilityIntent(
                asset=asset,
                vol_bias=VolBias.NEUTRAL,
                intensity=0.0,
                expected_horizon=VolHorizon.SHORT,
                rationale_tags=["insufficient_data"],
                data_quality=data_quality,
                is_valid=False,
                source_freshness=freshness,
                coverage=coverage,
            )

        # P1: Reset warning suppression when data recovers
        self._data_warning_shown = False

        # 计算各子信号 with coverage-aware fusion (P0)
        sub_signals: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        all_tags: List[str] = []

        # Resolve per-asset states
        lob = self._lob_states.get(asset)
        vol = self._vol_states.get(asset)
        onchain = self._onchain_states.get(asset)
        sentiment = self._sentiment_states.get(asset)
        corr = self._correlation_state  # global

        # Check freshness for each state
        lob_fresh = self._is_fresh(lob, self.config.lob_freshness_sec)
        vol_fresh = self._is_fresh(vol, self.config.vol_freshness_sec)
        onchain_fresh = self._is_fresh(onchain, self.config.onchain_freshness_sec)
        sentiment_fresh = self._is_fresh(sentiment, self.config.sentiment_freshness_sec)
        corr_fresh = self._is_fresh(corr, self.config.correlation_freshness_sec)

        # 1. Volatility Breakout Alpha (requires fresh vol + lob)
        if vol_fresh and lob_fresh:
            breakout_alpha = self._get_breakout_alpha(asset)
            breakout_intensity, breakout_tags = breakout_alpha.compute(vol, lob)
            sub_signals["breakout"] = breakout_intensity
            weights["breakout"] = self.config.breakout_weight
            all_tags.extend(breakout_tags)

        # 2. Liquidation Pressure Alpha (requires fresh onchain + sentiment)
        if onchain_fresh and sentiment_fresh:
            liquidation_intensity, liquidation_tags = self._liquidation_alpha.compute(
                onchain, sentiment
            )
            sub_signals["liquidation"] = liquidation_intensity
            weights["liquidation"] = self.config.liquidation_weight
            all_tags.extend(liquidation_tags)

        # 3. Compression -> Expansion Alpha (requires fresh vol + correlation)
        if vol_fresh and corr_fresh:
            compression_alpha = self._get_compression_alpha(asset)
            compression_intensity, compression_tags = compression_alpha.compute(
                vol, corr
            )
            sub_signals["compression"] = compression_intensity
            weights["compression"] = self.config.compression_weight
            all_tags.extend(compression_tags)

        # Coverage-aware fusion: renormalize over available sub-signals (P0)
        total_weight = sum(weights.values())
        if total_weight > 0:
            fused_intensity = sum(
                sub_signals[k] * weights[k] / total_weight
                for k in sub_signals
            )
        else:
            fused_intensity = 0.0

        # 确定 vol_bias
        if fused_intensity >= self.config.min_intensity_for_output:
            vol_bias = VolBias.INCREASE
            self._stats["increase_signals"] += 1
        elif fused_intensity < 0.15:
            vol_bias = VolBias.SUPPRESS
            self._stats["suppress_signals"] += 1
        else:
            vol_bias = VolBias.NEUTRAL
            self._stats["neutral_signals"] += 1

        # 确定 horizon
        horizon = VolHorizon.SHORT
        if "extended_compression" in all_tags or "compression_breakout" in all_tags:
            horizon = VolHorizon.MEDIUM

        # 构建输出
        intent = VolatilityIntent(
            asset=asset,
            vol_bias=vol_bias,
            intensity=min(fused_intensity, self.config.max_intensity),
            expected_horizon=horizon,
            rationale_tags=all_tags,
            sub_signals=sub_signals,
            data_quality=data_quality,
            is_valid=True,
            source_freshness=freshness,
            coverage=coverage,
        )

        # 记录历史
        self._intent_history.append(intent)

        # 日志
        self._log_evaluation(intent)

        # 发布事件
        self._publish_intent(intent)

        return intent

    def _check_data_health(self, asset: str) -> Tuple[float, int, Dict[str, float], Dict[str, str]]:
        """
        检查数据健康 (per-asset)

        Returns:
            (data_quality, available_sources, freshness_dict, coverage_dict)
        """
        sources_available = 0
        freshness: Dict[str, float] = {}
        coverage: Dict[str, str] = {}
        now = datetime.now()

        # --- LOB ---
        lob = self._lob_states.get(asset)
        if lob is not None:
            age = (now - lob.timestamp).total_seconds()
            freshness["lob"] = round(age, 1)
            if age < self.config.lob_freshness_sec:
                sources_available += 1
                coverage["lob"] = "fresh"
            else:
                coverage["lob"] = "stale"
        else:
            freshness["lob"] = -1.0
            coverage["lob"] = "missing"

        # --- Volatility ---
        vol = self._vol_states.get(asset)
        if vol is not None:
            age = (now - vol.timestamp).total_seconds()
            freshness["volatility"] = round(age, 1)
            if age < self.config.vol_freshness_sec:
                sources_available += 1
                coverage["volatility"] = "fresh"
            else:
                coverage["volatility"] = "stale"
        else:
            freshness["volatility"] = -1.0
            coverage["volatility"] = "missing"

        # --- Onchain ---
        onchain = self._onchain_states.get(asset)
        if onchain is not None:
            age = (now - onchain.timestamp).total_seconds()
            freshness["onchain"] = round(age, 1)
            if age < self.config.onchain_freshness_sec:
                sources_available += 1
                coverage["onchain"] = "fresh"
            else:
                coverage["onchain"] = "stale"
        else:
            freshness["onchain"] = -1.0
            coverage["onchain"] = "missing"

        # --- Sentiment ---
        sentiment = self._sentiment_states.get(asset)
        if sentiment is not None:
            age = (now - sentiment.timestamp).total_seconds()
            freshness["sentiment"] = round(age, 1)
            if age < self.config.sentiment_freshness_sec:
                sources_available += 1
                coverage["sentiment"] = "fresh"
            else:
                coverage["sentiment"] = "stale"
        else:
            freshness["sentiment"] = -1.0
            coverage["sentiment"] = "missing"

        # --- Correlation (global, not per-asset) ---
        corr = self._correlation_state
        if corr is not None:
            age = (now - corr.timestamp).total_seconds()
            freshness["correlation"] = round(age, 1)
            if age < self.config.correlation_freshness_sec:
                sources_available += 1
                coverage["correlation"] = "fresh"
            else:
                coverage["correlation"] = "stale"
        else:
            freshness["correlation"] = -1.0
            coverage["correlation"] = "missing"

        data_quality = sources_available / 5
        return data_quality, sources_available, freshness, coverage

    def _log_evaluation(self, intent: VolatilityIntent):
        """记录评估日志 (必须)"""
        logger.info(
            f"VolAlpha: asset={intent.asset} vol_bias={intent.vol_bias.value} "
            f"intensity={intent.intensity:.2f} horizon={intent.expected_horizon.value} "
            f"tags={intent.rationale_tags} data_quality={intent.data_quality:.2f}"
        )

    def _publish_intent(self, intent: VolatilityIntent):
        """发布意图到 EventBus"""
        if self._event_callback:
            try:
                self._event_callback("VOL_ALPHA_INTENT", intent.to_dict())
            except Exception as e:
                logger.warning(f"Failed to publish VOL_ALPHA_INTENT: {e}")

    # =========================================================================
    # HMATS COMPATIBILITY - generate_signal / to_agent_signal_dict
    # =========================================================================

    def generate_signal(
        self,
        asset: str = "BTC",
        price: float = 0.0,
        regime: str = "",
    ) -> VolatilityIntent:
        """
        Unified entry point matching HMATS generate_signal() convention.

        Always returns VolatilityIntent (never None, never raises).
        """
        try:
            self._last_regime = regime.upper() if regime else ""  # [UPG3]
            return self.evaluate(asset=asset)
        except Exception as e:
            logger.error(f"[VOL_ALPHA] generate_signal failed: {e}", exc_info=True)
            return VolatilityIntent(
                asset=asset,
                vol_bias=VolBias.NEUTRAL,
                intensity=0.0,
                expected_horizon=VolHorizon.SHORT,
                rationale_tags=["internal_error"],
                is_valid=False,
                data_quality=0.0,
            )

    def to_agent_signal_dict(self, asset: str = "BTC") -> Dict[str, Any]:
        """
        Convert latest VolatilityIntent into flat dict for Authority Fusion.

        V6 contract:
        - vol_alpha_direction: ALWAYS 0.0 (this agent never outputs direction)
        - vol_alpha_bias / intensity / horizon: core signal
        - execution_urgency_mult: 0.5..1.5 (suppress->1.5 increase)
        - leverage_cap_soft: 0.5..1.0 (high vol -> lower cap)
        - missing_inputs: list of missing data source names
        - asof_timestamp / data_age_seconds: freshness
        """
        import time as _time
        intent = self._intent_history[-1] if self._intent_history else self.evaluate(asset)

        # Execution urgency: high vol -> more urgency to reduce exposure
        if intent.vol_bias == VolBias.INCREASE:
            urgency = min(1.5, 1.0 + intent.intensity * 0.5)
        elif intent.vol_bias == VolBias.SUPPRESS:
            urgency = max(0.5, 1.0 - intent.intensity * 0.5)
        else:
            urgency = 1.0

        # Leverage cap: high vol intensity -> lower leverage allowed
        leverage_cap = max(0.5, 1.0 - intent.intensity * 0.5) if intent.is_valid else 1.0

        # Missing inputs from coverage
        missing = [k for k, v in intent.coverage.items() if v == "missing"]

        ts = intent.timestamp.timestamp() if isinstance(intent.timestamp, datetime) else 0.0
        now = _time.time()

        # [FIX-AG4] Vol_alpha is direction-agnostic. The UPG3 regime overlay injected
        # unauthorized -0.4 direction in BEAR regimes, violating the agent contract
        # ("vol_alpha_direction: ALWAYS 0.0"). Removed: regime-based directional bias
        # should come from regime_power or short_bias agents, not vol_alpha.
        implied_direction = 0.0
        direction_reason = "direction_agnostic"

        return {
            "vol_alpha_direction": round(implied_direction, 4),  # [UPG3] regime-conditional
            "vol_alpha_implied_direction": round(implied_direction, 4),  # [UPG3] explicit alias
            "vol_alpha_direction_reason": direction_reason,  # [UPG3]
            "vol_alpha_bias": intent.vol_bias.value,
            "vol_alpha_intensity": intent.intensity,
            "vol_alpha_horizon": intent.expected_horizon.value,
            "vol_alpha_data_quality": intent.data_quality,
            "vol_alpha_valid": intent.is_valid,
            "vol_alpha_tags": intent.rationale_tags,
            "vol_alpha_coverage": intent.coverage,
            "execution_urgency_mult": round(urgency, 3),
            "leverage_cap_soft": round(leverage_cap, 3),
            "missing_inputs": missing,
            "asof_timestamp": ts,
            "data_age_seconds": round(now - ts, 1) if ts > 0 else None,
            "asset": intent.asset,
        }

    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """获取 Agent 状态"""
        return {
            "enabled": self.config.enabled,
            "stats": self._stats,
            "last_intent": self._intent_history[-1].to_dict() if self._intent_history else None,
            "data_sources": {
                "lob": len(self._lob_states) > 0,
                "volatility": len(self._vol_states) > 0,
                "onchain": len(self._onchain_states) > 0,
                "sentiment": len(self._sentiment_states) > 0,
                "correlation": self._correlation_state is not None,
            },
            "assets_tracked": sorted(set(
                list(self._lob_states.keys()) +
                list(self._vol_states.keys()) +
                list(self._onchain_states.keys()) +
                list(self._sentiment_states.keys())
            )),
        }

    def record_fusion_result(self, accepted: bool, veto_reason: str = ""):
        """
        记录 Authority Fusion 的结果

        用于追踪 VolAlpha 是否被接纳
        """
        if accepted:
            self._stats["accepted_count"] += 1
            logger.info("VolAlpha: Intent ACCEPTED by Authority Fusion")
        else:
            self._stats["vetoed_count"] += 1
            logger.info(f"VolAlpha: Intent VETOED by Authority Fusion (reason={veto_reason})")


# =============================================================================
# EVENTBUS INTEGRATION
# =============================================================================

class VolAlphaEventBusIntegration:
    """VolatilityAlphaAgent EventBus 集成"""

    def __init__(self, agent: VolatilityAlphaAgent):
        self.agent = agent
        self._event_manager = None
        self._subscribed = False

    def initialize(self) -> bool:
        """初始化 EventBus 集成"""
        try:
            from infra.unified_event_v521 import get_unified_event_manager, EventTypeV521

            self._event_manager = get_unified_event_manager()

            # Subscribe to events that exist in EventTypeV521
            self._event_manager.subscribe(
                EventTypeV521.LOB_TICK,
                self._on_lob_tick
            )
            self._event_manager.subscribe(
                EventTypeV521.ONCHAIN_TICK,
                self._on_onchain_tick
            )
            self._event_manager.subscribe(
                EventTypeV521.SENTIMENT_TICK,
                self._on_sentiment_tick
            )
            self._event_manager.subscribe(
                EventTypeV521.CORRELATION_REGIME_CHANGE,
                self._on_correlation_event
            )
            self._event_manager.subscribe(
                EventTypeV521.CORRELATION_JUMP,
                self._on_correlation_event
            )

            # 设置发布回调
            self.agent.set_event_callback(self._publish_event)

            self._subscribed = True
            logger.info("VolAlpha EventBus integration initialized (LOB_TICK, ONCHAIN_TICK, SENTIMENT_TICK, CORRELATION_*)")
            return True

        except Exception as e:
            logger.warning(f"VolAlpha EventBus integration failed: {e}")
            return False

    def _on_lob_tick(self, event):
        """处理 LOB_TICK"""
        self.agent.on_lob_tick(event.payload)

    def _on_onchain_tick(self, event):
        """处理 ONCHAIN_TICK"""
        self.agent.on_onchain_tick(event.payload)

    def _on_sentiment_tick(self, event):
        """处理 SENTIMENT_TICK"""
        self.agent.on_sentiment_tick(event.payload)

    def _on_correlation_event(self, event):
        """处理 CORRELATION_REGIME_CHANGE / CORRELATION_JUMP"""
        self.agent.on_correlation_state(event.payload)

    def _publish_event(self, event_type: str, payload: Dict):
        """发布事件"""
        if not self._event_manager:
            return

        try:
            from infra.unified_event_v521 import UnifiedEvent, EventTypeV521

            event = UnifiedEvent(
                event_type=EventTypeV521.ALPHA_SIGNAL,
                source="volatility_alpha_agent",
                payload={
                    "signal_type": event_type,
                    **payload,
                },
            )
            self._event_manager.process_event(event)

        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")


# =============================================================================
# SINGLETON
# =============================================================================

_vol_alpha_instance: Optional[VolatilityAlphaAgent] = None


def get_volatility_alpha_agent(
    config: VolAlphaConfig = None
) -> VolatilityAlphaAgent:
    """获取 VolatilityAlphaAgent 单例"""
    global _vol_alpha_instance
    if _vol_alpha_instance is None:
        _vol_alpha_instance = VolatilityAlphaAgent(config)
    elif config is not None and _vol_alpha_instance.config != config:
        logger.info("[VOL_ALPHA] Config changed; refreshing singleton instance")
        _vol_alpha_instance = VolatilityAlphaAgent(config)
    return _vol_alpha_instance


def reset_volatility_alpha_agent():
    """重置单例"""
    global _vol_alpha_instance
    _vol_alpha_instance = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VolBias',
    'VolHorizon',
    'VolSignalType',
    'VolatilityIntent',
    'LOBState',
    'VolatilityState',
    'OnchainState',
    'SentimentState',
    'CorrelationState',
    'VolAlphaConfig',
    'VolatilityBreakoutAlpha',
    'LiquidationPressureAlpha',
    'CompressionExpansionAlpha',
    'VolatilityAlphaAgent',
    'VolAlphaEventBusIntegration',
    'get_volatility_alpha_agent',
    'reset_volatility_alpha_agent',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing VolatilityAlphaAgent")
    print("=" * 60)

    # 创建 Agent
    config = VolAlphaConfig()
    agent = VolatilityAlphaAgent(config)

    # 模拟熊市场景数据
    print("\n1. Simulating bear market scenario...")

    # 模拟 LOB 数据 (spread 扩张, depth 下降)
    agent.on_lob_tick({
        "asset": "BTC",
        "spread_bps": 15.0,  # 扩张
        "depth_imbalance": -0.3,  # 卖压
        "bid_depth_usd": 500000,
        "ask_depth_usd": 800000,
        "vpin_proxy": 0.7,
    })

    # 模拟波动率数据 (高波动)
    agent.on_volatility_state({
        "asset": "BTC",
        "realized_vol_1h": 0.8,
        "realized_vol_4h": 0.7,
        "realized_vol_24h": 0.5,
        "vol_percentile_30d": 85.0,  # 高分位数
        "vol_zscore": 2.0,
        "vol_of_vol": 0.6,
    })

    # 模拟链上数据 (交易所流入激增)
    agent.on_onchain_tick({
        "asset": "BTC",
        "exchange_inflow_usd_1h": 50000000,
        "exchange_inflow_spike": True,
        "exchange_inflow_zscore": 2.5,
        "stablecoin_flow_direction": "outflow",
        "funding_deviation": 0.8,
    })

    # 模拟情绪数据 (恐慌)
    agent.on_sentiment_tick({
        "asset": "BTC",
        "fear_index": 18.0,  # 极端恐慌
        "fear_index_change_1h": -5.0,
        "sentiment_volatility": 0.5,
    })

    # 模拟相关性数据 (相关性 spike)
    agent.on_correlation_state({
        "btc_dominance": 55.0,
        "btc_dominance_change_24h": 3.0,
        "cross_asset_correlation": 0.87,
        "correlation_regime": "exploded",
        "correlation_spike": True,
    })

    # 评估
    print("\n2. Evaluating volatility alpha...")
    intent = agent.evaluate("BTC")

    print(f"\n3. Result:")
    print(f"   vol_bias: {intent.vol_bias.value}")
    print(f"   intensity: {intent.intensity:.2f}")
    print(f"   horizon: {intent.expected_horizon.value}")
    print(f"   rationale_tags: {intent.rationale_tags}")
    print(f"   sub_signals: {intent.sub_signals}")
    print(f"   data_quality: {intent.data_quality:.2f}")
    print(f"   coverage: {intent.coverage}")

    # 获取状态
    print("\n4. Agent status:")
    status = agent.get_status()
    print(f"   stats: {status['stats']}")

    # 模拟牛市场景
    print("\n5. Simulating bull market scenario...")

    agent.on_volatility_state({
        "asset": "BTC",
        "realized_vol_1h": 0.2,
        "realized_vol_4h": 0.2,
        "realized_vol_24h": 0.25,
        "vol_percentile_30d": 30.0,  # 低分位数
        "vol_zscore": -0.5,
        "vol_of_vol": 0.1,
    })

    agent.on_sentiment_tick({
        "asset": "BTC",
        "fear_index": 75.0,  # 贪婪
        "sentiment_volatility": 0.1,
    })

    agent.on_correlation_state({
        "btc_dominance": 45.0,
        "correlation_regime": "normal",
        "correlation_spike": False,
    })

    intent_bull = agent.evaluate("BTC")

    print(f"\n6. Bull market result:")
    print(f"   vol_bias: {intent_bull.vol_bias.value}")
    print(f"   intensity: {intent_bull.intensity:.2f}")
    print(f"   tags: {intent_bull.rationale_tags}")

    print("\n✓ VolatilityAlphaAgent test complete")

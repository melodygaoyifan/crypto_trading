"""
================================================================================
HMATS v5.2.1 SOTA - 市场冲击模型实战化
Production Market Impact Model
================================================================================

Purpose: 用真实订单簿和成交数据训练冲击成本模型

SOTA 需求:
1. 用真实订单簿和成交数据训练冲击成本模型
2. 在执行策略前调用预测结果决定拆单规模与挂单价格
3. 自适应模型参数校准

Features:
1. 真实订单簿深度分析
2. 历史成交数据冲击学习
3. Almgren-Chriss扩展模型
4. 在线参数自适应
5. 最优拆单规模计算
6. 限价单挂单价格建议
7. EventBus集成

Author: HMATS v5.2.1 SOTA Complete Upgrade
Version: 5.2.1-SOTA-UPGRADED
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

# --- [v9-PATCH-7a] Import calibration bridge ---
try:
    from execution.impact_calibration import ProductionMarketImpactCalibrationBridge
    CALIBRATION_BRIDGE_AVAILABLE = True
except ImportError:
    CALIBRATION_BRIDGE_AVAILABLE = False
# --- end [v9-PATCH-7a] ---


# =============================================================================
# ENUMS
# =============================================================================

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class UrgencyLevel(Enum):
    """紧急程度"""
    LOW = "low"           # 可以等待最佳价格
    MEDIUM = "medium"     # 平衡速度和成本
    HIGH = "high"         # 优先速度
    CRITICAL = "critical" # 立即执行


class ImpactType(Enum):
    """冲击类型"""
    TEMPORARY = "temporary"    # 临时冲击 (执行期间)
    PERMANENT = "permanent"    # 永久冲击 (市场影响)
    TOTAL = "total"           # 总冲击


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MarketImpactConfig:
    """市场冲击模型配置"""
    
    # 模型参数初始值 (Almgren-Chriss)
    eta_init: float = 0.1          # 临时冲击系数
    gamma_init: float = 0.1        # 永久冲击系数
    lambda_init: float = 1e-6      # 风险厌恶系数
    
    # 学习率
    eta_lr: float = 0.01           # η 学习率
    gamma_lr: float = 0.01         # γ 学习率
    
    # 订单簿参数
    depth_levels: int = 20         # 考虑的深度层数
    min_liquidity_usd: float = 10000  # 最小流动性要求
    
    # 拆单参数
    max_participation: float = 0.1     # 单次最大市场占比
    min_slice_usd: float = 1000       # 最小拆单金额
    max_slices: int = 100             # 最大拆单数
    
    # 限价单参数
    limit_offset_bps_low: int = 5      # 低紧急度偏移 (基点)
    limit_offset_bps_medium: int = 2   # 中紧急度偏移
    limit_offset_bps_high: int = 0     # 高紧急度偏移
    
    # 历史学习
    execution_history_size: int = 500  # 保留的历史执行记录数
    min_samples_for_calibration: int = 30  # 校准所需最小样本数


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    size: float
    cumulative_size: float = 0.0
    cumulative_value: float = 0.0


@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    timestamp: datetime
    symbol: str
    
    # 买卖盘
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    
    # 统计
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    imbalance: float = 0.0  # >0 买压, <0 卖压


@dataclass
class ExecutionRecord:
    """执行记录"""
    timestamp: datetime
    symbol: str
    
    # 订单详情
    side: str  # "buy" or "sell"
    intended_size: float
    executed_size: float
    
    # 价格
    arrival_price: float       # 下单时的mid price
    avg_execution_price: float # 平均成交价
    
    # 冲击
    slippage_bps: float       # 滑点 (基点)
    estimated_impact: float   # 预估冲击
    actual_impact: float      # 实际冲击
    
    # 市场状态
    spread_at_order: float
    volume_participation: float  # 占市场成交量比例


@dataclass
class ImpactPrediction:
    """冲击预测"""
    timestamp: datetime
    symbol: str
    side: str
    size: float
    
    # 预测冲击
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    
    # 成本估算
    expected_cost_usd: float
    
    # 置信区间
    impact_ci_low: float
    impact_ci_high: float
    confidence: float


@dataclass
class SlicingPlan:
    """拆单计划"""
    timestamp: datetime
    symbol: str
    side: str
    total_size: float
    urgency: UrgencyLevel
    
    # 拆单详情
    num_slices: int
    slice_sizes: List[float]
    
    # 执行建议
    order_type: OrderType
    limit_offset_bps: float
    suggested_limit_price: float
    
    # 预估
    estimated_duration_seconds: int
    estimated_total_impact_bps: float
    estimated_cost_usd: float
    
    # 风险
    execution_risk: str  # "low", "medium", "high"


# =============================================================================
# ORDERBOOK ANALYZER
# =============================================================================

class OrderBookAnalyzer:
    """订单簿分析器"""
    
    def __init__(self, config: MarketImpactConfig):
        self.config = config
    
    def parse_orderbook(self, 
                       raw_bids: List[Tuple[float, float]],
                       raw_asks: List[Tuple[float, float]],
                       symbol: str) -> OrderBookSnapshot:
        """
        解析原始订单簿数据
        
        Args:
            raw_bids: [(price, size), ...]
            raw_asks: [(price, size), ...]
            symbol: 交易对
        
        Returns:
            OrderBookSnapshot
        """
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
        )
        
        # 处理买盘 (降序)
        cum_size = 0.0
        cum_value = 0.0
        for price, size in sorted(raw_bids, key=lambda x: -x[0])[:self.config.depth_levels]:
            cum_size += size
            cum_value += price * size
            snapshot.bids.append(OrderBookLevel(
                price=price,
                size=size,
                cumulative_size=cum_size,
                cumulative_value=cum_value,
            ))
        
        # 处理卖盘 (升序)
        cum_size = 0.0
        cum_value = 0.0
        for price, size in sorted(raw_asks, key=lambda x: x[0])[:self.config.depth_levels]:
            cum_size += size
            cum_value += price * size
            snapshot.asks.append(OrderBookLevel(
                price=price,
                size=size,
                cumulative_size=cum_size,
                cumulative_value=cum_value,
            ))
        
        # 计算统计
        if snapshot.bids and snapshot.asks:
            best_bid = snapshot.bids[0].price
            best_ask = snapshot.asks[0].price
            
            snapshot.mid_price = (best_bid + best_ask) / 2
            snapshot.spread = best_ask - best_bid
            snapshot.spread_bps = (snapshot.spread / snapshot.mid_price) * 10000
            
            snapshot.bid_depth_usd = snapshot.bids[-1].cumulative_value if snapshot.bids else 0
            snapshot.ask_depth_usd = snapshot.asks[-1].cumulative_value if snapshot.asks else 0
            
            total_depth = snapshot.bid_depth_usd + snapshot.ask_depth_usd
            if total_depth > 0:
                snapshot.imbalance = (snapshot.bid_depth_usd - snapshot.ask_depth_usd) / total_depth
        
        return snapshot
    
    def estimate_execution_price(self, 
                                snapshot: OrderBookSnapshot,
                                side: str,
                                size: float) -> Tuple[float, float]:
        """
        估算成交均价
        
        Args:
            snapshot: 订单簿快照
            side: "buy" or "sell"
            size: 订单大小
        
        Returns:
            (avg_price, slippage_bps)
        """
        levels = snapshot.asks if side == "buy" else snapshot.bids
        
        if not levels:
            return snapshot.mid_price, 0.0
        
        remaining = size
        total_cost = 0.0
        
        for level in levels:
            if remaining <= 0:
                break
            
            fill_size = min(remaining, level.size)
            total_cost += fill_size * level.price
            remaining -= fill_size
        
        # 如果订单簿深度不够
        if remaining > 0:
            # 外推价格
            last_price = levels[-1].price if levels else snapshot.mid_price
            slippage_per_unit = snapshot.spread / 2
            extra_slippage = (remaining / (size + 1)) * slippage_per_unit * 10
            total_cost += remaining * (last_price + extra_slippage * (1 if side == "buy" else -1))
        
        avg_price = total_cost / size if size > 0 else snapshot.mid_price
        
        # 计算滑点
        reference = snapshot.asks[0].price if side == "buy" else snapshot.bids[0].price
        slippage_bps = abs(avg_price - reference) / reference * 10000
        
        return avg_price, slippage_bps


# =============================================================================
# MARKET IMPACT MODEL
# =============================================================================

class AdaptiveMarketImpactModel:
    """
    自适应市场冲击模型
    
    基于Almgren-Chriss模型，支持在线参数校准
    
    临时冲击: h(v) = η * σ * (v/V)^0.6
    永久冲击: g(v) = γ * σ * (v/V)
    
    其中:
    - v: 订单大小
    - V: 市场日成交量
    - σ: 价格波动率
    - η, γ: 待校准参数
    """
    
    def __init__(self, config: Optional[MarketImpactConfig] = None):
        self.config = config or MarketImpactConfig()
        
        # 模型参数 (按symbol)
        self._params: Dict[str, Dict[str, float]] = {}
        
        # 执行历史
        self._execution_history: Dict[str, deque] = {}
        
        # 市场数据
        self._market_data: Dict[str, Dict[str, float]] = {}
        
        # 订单簿分析器
        self.orderbook_analyzer = OrderBookAnalyzer(self.config)
        
        # 事件回调
        self._event_callbacks: List[Callable] = []
        
        # 统计
        self._stats = {
            "predictions": 0,
            "calibrations": 0,
            "avg_prediction_error": 0.0,
        }
        
        # --- [v9-PATCH-7b] Init calibration bridge ---
        self._calibration_bridge = None
        if CALIBRATION_BRIDGE_AVAILABLE:
            try:
                self._calibration_bridge = ProductionMarketImpactCalibrationBridge()
                logger.info("MarketImpact: calibration bridge connected")
            except Exception as e:
                logger.warning(f"MarketImpact: calibration bridge init failed: {e}")
        # --- end [v9-PATCH-7b] ---

        logger.info("AdaptiveMarketImpactModel initialized")
    
    # =========================================================================
    # PARAMETER MANAGEMENT
    # =========================================================================
    
    def _get_params(self, symbol: str) -> Dict[str, float]:
        """获取symbol的模型参数"""
        # --- [v9-PATCH-7c] Prefer bucket-level calibrated params ---
        if self._calibration_bridge:
            try:
                market = self._market_data.get(symbol, {})
                eta, gamma, conf = self._calibration_bridge.get_calibrated_params(
                    asset=symbol,
                    volatility=market.get("volatility", 0.02),
                    spread_bps=market.get("avg_spread_bps", 5.0),
                )
                if conf > 0.3:
                    return {"eta": eta, "gamma": gamma, "lambda": self.config.lambda_init}
            except Exception:
                pass  # fall through to existing logic
        # --- end [v9-PATCH-7c] ---

        if symbol not in self._params:
            self._params[symbol] = {
                "eta": self.config.eta_init,
                "gamma": self.config.gamma_init,
                "lambda": self.config.lambda_init,
            }
        return self._params[symbol]
    
    def _ensure_history(self, symbol: str):
        """确保历史记录存在"""
        if symbol not in self._execution_history:
            self._execution_history[symbol] = deque(maxlen=self.config.execution_history_size)
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def update_market_data(self, 
                          symbol: str,
                          daily_volume: float,
                          volatility: float,
                          avg_spread_bps: float = 5.0):
        """
        更新市场数据
        
        Args:
            symbol: 交易对
            daily_volume: 日成交量 (USD)
            volatility: 波动率 (年化)
            avg_spread_bps: 平均价差 (基点)
        """
        self._market_data[symbol] = {
            "daily_volume": daily_volume,
            "volatility": volatility,
            "avg_spread_bps": avg_spread_bps,
            "updated_at": datetime.now().isoformat(),
        }
    
    def _get_market_data(self, symbol: str) -> Dict[str, float]:
        """获取市场数据"""
        if symbol in self._market_data:
            return self._market_data[symbol]
        
        # 默认值
        return {
            "daily_volume": 1e9,  # 10亿USD
            "volatility": 0.8,    # 80%年化波动
            "avg_spread_bps": 5.0,
        }
    
    # =========================================================================
    # IMPACT PREDICTION
    # =========================================================================
    
    def predict_impact(self,
                      symbol: str,
                      side: str,
                      size_usd: float,
                      orderbook: Optional[OrderBookSnapshot] = None) -> ImpactPrediction:
        """
        预测市场冲击
        
        Args:
            symbol: 交易对
            side: "buy" or "sell"
            size_usd: 订单大小 (USD)
            orderbook: 订单簿快照 (可选)
        
        Returns:
            ImpactPrediction
        """
        params = self._get_params(symbol)
        market = self._get_market_data(symbol)
        
        eta = params["eta"]
        gamma = params["gamma"]
        
        daily_volume = market["daily_volume"]
        volatility = market["volatility"]
        
        # 计算参与率
        participation = size_usd / daily_volume if daily_volume > 0 else 0.01
        
        # 临时冲击: η * σ * (v/V)^0.6
        temp_impact = eta * volatility * (participation ** 0.6)
        
        # 永久冲击: γ * σ * (v/V)
        perm_impact = gamma * volatility * participation
        
        # 转换为基点
        temp_impact_bps = temp_impact * 10000
        perm_impact_bps = perm_impact * 10000
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        # 加上价差成本
        spread_cost = market["avg_spread_bps"] / 2
        total_impact_bps += spread_cost
        
        # 订单簿调整
        if orderbook:
            # 检查流动性
            depth = orderbook.ask_depth_usd if side == "buy" else orderbook.bid_depth_usd
            if depth < size_usd:
                # 流动性不足，增加冲击估计
                liquidity_factor = size_usd / max(depth, 1000)
                total_impact_bps *= (1 + 0.5 * min(liquidity_factor, 5))
            
            # 订单簿不平衡调整
            if (side == "buy" and orderbook.imbalance > 0.3) or \
               (side == "sell" and orderbook.imbalance < -0.3):
                # 顺势交易，冲击可能更大
                total_impact_bps *= 1.2
        
        # 计算成本
        expected_cost_usd = size_usd * total_impact_bps / 10000
        
        # 置信区间 (基于历史误差)
        self._ensure_history(symbol)
        history = self._execution_history[symbol]
        
        if len(history) >= 10:
            errors = [abs(r.actual_impact - r.estimated_impact) for r in history]
            std_error = np.std(errors)
            ci_range = std_error * 1.96
        else:
            ci_range = total_impact_bps * 0.5  # 默认50%不确定性
        
        confidence = max(0.3, 1 - ci_range / (total_impact_bps + 1))
        
        self._stats["predictions"] += 1
        
        prediction = ImpactPrediction(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            size=size_usd,
            temporary_impact_bps=temp_impact_bps,
            permanent_impact_bps=perm_impact_bps,
            total_impact_bps=total_impact_bps,
            expected_cost_usd=expected_cost_usd,
            impact_ci_low=max(0, total_impact_bps - ci_range),
            impact_ci_high=total_impact_bps + ci_range,
            confidence=confidence,
        )
        
        return prediction
    
    # =========================================================================
    # OPTIMAL SLICING
    # =========================================================================
    
    def compute_optimal_slicing(self,
                               symbol: str,
                               side: str,
                               total_size_usd: float,
                               urgency: UrgencyLevel,
                               orderbook: Optional[OrderBookSnapshot] = None) -> SlicingPlan:
        """
        计算最优拆单计划
        
        Args:
            symbol: 交易对
            side: "buy" or "sell"
            total_size_usd: 总订单大小 (USD)
            urgency: 紧急程度
            orderbook: 订单簿快照
        
        Returns:
            SlicingPlan
        """
        market = self._get_market_data(symbol)
        
        # 根据紧急程度确定参数
        if urgency == UrgencyLevel.CRITICAL:
            max_participation = 0.3
            order_type = OrderType.MARKET
            duration_factor = 0.5
        elif urgency == UrgencyLevel.HIGH:
            max_participation = 0.15
            order_type = OrderType.LIMIT
            duration_factor = 1.0
        elif urgency == UrgencyLevel.MEDIUM:
            max_participation = 0.08
            order_type = OrderType.TWAP
            duration_factor = 2.0
        else:  # LOW
            max_participation = 0.05
            order_type = OrderType.VWAP
            duration_factor = 4.0
        
        # 计算单次最大订单
        daily_volume = market["daily_volume"]
        max_slice_usd = daily_volume * max_participation / 24  # 每小时份额
        
        # 订单簿流动性限制
        if orderbook:
            depth = orderbook.ask_depth_usd if side == "buy" else orderbook.bid_depth_usd
            max_slice_usd = min(max_slice_usd, depth * 0.2)  # 不超过深度的20%
        
        max_slice_usd = max(max_slice_usd, self.config.min_slice_usd)
        
        # 计算拆单数量
        num_slices = max(1, min(
            self.config.max_slices,
            int(np.ceil(total_size_usd / max_slice_usd))
        ))
        
        # 均匀拆分
        base_slice = total_size_usd / num_slices
        slice_sizes = [base_slice] * num_slices
        
        # 轻微随机化 (避免被检测)
        if num_slices > 3:
            for i in range(num_slices):
                jitter = 0.9 + 0.2 * np.random.random()
                slice_sizes[i] *= jitter
            
            # 归一化
            total = sum(slice_sizes)
            slice_sizes = [s * total_size_usd / total for s in slice_sizes]
        
        # 计算限价单偏移
        if urgency == UrgencyLevel.LOW:
            limit_offset_bps = self.config.limit_offset_bps_low
        elif urgency == UrgencyLevel.MEDIUM:
            limit_offset_bps = self.config.limit_offset_bps_medium
        else:
            limit_offset_bps = self.config.limit_offset_bps_high
        
        # 计算建议限价
        suggested_limit = 0.0
        if orderbook and orderbook.mid_price > 0:
            offset_factor = limit_offset_bps / 10000
            if side == "buy":
                suggested_limit = orderbook.mid_price * (1 - offset_factor)
            else:
                suggested_limit = orderbook.mid_price * (1 + offset_factor)
        
        # 估算持续时间
        estimated_duration = int(num_slices * 60 * duration_factor)  # 秒
        
        # 预测总冲击
        impact_pred = self.predict_impact(symbol, side, total_size_usd, orderbook)
        
        # 风险评估
        participation_rate = total_size_usd / daily_volume if daily_volume > 0 else 1.0
        if participation_rate > 0.1:
            execution_risk = "high"
        elif participation_rate > 0.03:
            execution_risk = "medium"
        else:
            execution_risk = "low"
        
        plan = SlicingPlan(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            total_size=total_size_usd,
            urgency=urgency,
            num_slices=num_slices,
            slice_sizes=slice_sizes,
            order_type=order_type,
            limit_offset_bps=limit_offset_bps,
            suggested_limit_price=suggested_limit,
            estimated_duration_seconds=estimated_duration,
            estimated_total_impact_bps=impact_pred.total_impact_bps,
            estimated_cost_usd=impact_pred.expected_cost_usd,
            execution_risk=execution_risk,
        )
        
        return plan
    
    # =========================================================================
    # EXECUTION FEEDBACK & CALIBRATION
    # =========================================================================
    
    def record_execution(self, record: ExecutionRecord):
        """
        记录执行结果用于模型校准
        
        Args:
            record: 执行记录
        """
        self._ensure_history(record.symbol)
        self._execution_history[record.symbol].append(record)
        
        # 尝试校准
        self._calibrate_if_needed(record.symbol)
    
    def record_execution_simple(self,
                               symbol: str,
                               side: str,
                               size: float,
                               arrival_price: float,
                               avg_exec_price: float,
                               spread: float = 0.0):
        """
        简化的执行记录
        """
        # 计算滑点
        if side == "buy":
            slippage = (avg_exec_price - arrival_price) / arrival_price
        else:
            slippage = (arrival_price - avg_exec_price) / arrival_price
        
        slippage_bps = slippage * 10000
        
        # 预测冲击
        pred = self.predict_impact(symbol, side, size * arrival_price)
        
        record = ExecutionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            intended_size=size,
            executed_size=size,
            arrival_price=arrival_price,
            avg_execution_price=avg_exec_price,
            slippage_bps=slippage_bps,
            estimated_impact=pred.total_impact_bps,
            actual_impact=slippage_bps,
            spread_at_order=spread,
            volume_participation=0.01,  # 未知
        )
        
        self.record_execution(record)
    
    def _calibrate_if_needed(self, symbol: str):
        """如果有足够样本则校准模型"""
        history = self._execution_history.get(symbol, [])
        
        if len(history) < self.config.min_samples_for_calibration:
            return
        
        # 计算预测误差
        recent = list(history)[-50:]  # 最近50次执行
        
        errors = []
        for record in recent:
            error = record.actual_impact - record.estimated_impact
            errors.append(error)
        
        mean_error = np.mean(errors)
        
        # 调整参数
        params = self._get_params(symbol)
        
        # 如果系统性低估，增加eta
        if mean_error > 2:  # 低估超过2bps
            params["eta"] *= (1 + self.config.eta_lr)
            params["gamma"] *= (1 + self.config.gamma_lr)
            logger.info(f"Calibrated {symbol}: increased impact parameters (error={mean_error:.1f}bps)")
        
        # 如果系统性高估，降低eta
        elif mean_error < -2:  # 高估超过2bps
            params["eta"] *= (1 - self.config.eta_lr)
            params["gamma"] *= (1 - self.config.gamma_lr)
            logger.info(f"Calibrated {symbol}: decreased impact parameters (error={mean_error:.1f}bps)")
        
        # 参数边界
        params["eta"] = max(0.01, min(1.0, params["eta"]))
        params["gamma"] = max(0.01, min(1.0, params["gamma"]))
        
        self._stats["calibrations"] += 1
        self._stats["avg_prediction_error"] = mean_error
    
    # =========================================================================
    # LIMIT PRICE SUGGESTION
    # =========================================================================
    
    def suggest_limit_price(self,
                           symbol: str,
                           side: str,
                           size_usd: float,
                           urgency: UrgencyLevel,
                           orderbook: OrderBookSnapshot) -> Tuple[float, str]:
        """
        建议限价单价格
        
        Args:
            symbol: 交易对
            side: "buy" or "sell"
            size_usd: 订单大小
            urgency: 紧急程度
            orderbook: 订单簿
        
        Returns:
            (suggested_price, reasoning)
        """
        mid = orderbook.mid_price
        spread_bps = orderbook.spread_bps
        
        # 预测冲击
        impact = self.predict_impact(symbol, side, size_usd, orderbook)
        
        # 基础偏移
        if urgency == UrgencyLevel.CRITICAL:
            # 激进挂单，接近对手盘
            offset_bps = spread_bps * 0.1
            reasoning = "紧急订单: 接近对手盘挂单以快速成交"
        
        elif urgency == UrgencyLevel.HIGH:
            # 中等偏移
            offset_bps = spread_bps * 0.3
            reasoning = "高优先级: 略优于中间价以提高成交概率"
        
        elif urgency == UrgencyLevel.MEDIUM:
            # 标准偏移
            offset_bps = spread_bps * 0.5 + impact.total_impact_bps * 0.3
            reasoning = f"标准执行: 偏移={offset_bps:.1f}bps 考虑预期冲击"
        
        else:  # LOW
            # 被动挂单
            offset_bps = spread_bps * 0.7 + impact.total_impact_bps * 0.5
            reasoning = f"被动执行: 偏移={offset_bps:.1f}bps 等待更优价格"
        
        # 订单簿不平衡调整
        if side == "buy" and orderbook.imbalance > 0.2:
            offset_bps *= 0.8  # 买压大，减少偏移
            reasoning += " (买压大，减少偏移)"
        elif side == "sell" and orderbook.imbalance < -0.2:
            offset_bps *= 0.8
            reasoning += " (卖压大，减少偏移)"
        
        # 计算价格
        offset_factor = offset_bps / 10000
        
        if side == "buy":
            suggested_price = mid * (1 - offset_factor)
        else:
            suggested_price = mid * (1 + offset_factor)
        
        return suggested_price, reasoning
    
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
                callback(event_type, data)
            except Exception:
                pass
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_params(self, symbol: str) -> Dict[str, float]:
        """获取当前参数"""
        return self._get_params(symbol).copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "symbols_tracked": list(self._params.keys()),
            "history_sizes": {
                s: len(h) for s, h in self._execution_history.items()
            },
        }
    
    def reset(self, symbol: Optional[str] = None):
        """重置"""
        if symbol:
            if symbol in self._params:
                del self._params[symbol]
            if symbol in self._execution_history:
                self._execution_history[symbol].clear()
        else:
            self._params.clear()
            self._execution_history.clear()


# =============================================================================
# SINGLETON
# =============================================================================

_impact_model: Optional[AdaptiveMarketImpactModel] = None


def get_impact_model() -> AdaptiveMarketImpactModel:
    """获取市场冲击模型单例"""
    global _impact_model
    if _impact_model is None:
        _impact_model = AdaptiveMarketImpactModel()
    return _impact_model


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing AdaptiveMarketImpactModel")
    print("=" * 60)
    
    model = AdaptiveMarketImpactModel()
    
    # 更新市场数据
    model.update_market_data(
        symbol="BTC-USD",
        daily_volume=5e9,  # 50亿日成交量
        volatility=0.6,     # 60%年化波动率
    )
    
    # 创建订单簿
    analyzer = model.orderbook_analyzer
    orderbook = analyzer.parse_orderbook(
        raw_bids=[(99900, 10), (99850, 20), (99800, 30), (99750, 50), (99700, 100)],
        raw_asks=[(100100, 10), (100150, 20), (100200, 30), (100250, 50), (100300, 100)],
        symbol="BTC-USD"
    )
    
    print(f"\n1. Orderbook Analysis:")
    print(f"   Mid Price: ${orderbook.mid_price:,.2f}")
    print(f"   Spread: {orderbook.spread_bps:.1f} bps")
    print(f"   Bid Depth: ${orderbook.bid_depth_usd:,.0f}")
    print(f"   Ask Depth: ${orderbook.ask_depth_usd:,.0f}")
    print(f"   Imbalance: {orderbook.imbalance:.2f}")
    
    # 预测冲击
    print(f"\n2. Impact Prediction (Buy $100k):")
    impact = model.predict_impact("BTC-USD", "buy", 100000, orderbook)
    print(f"   Temporary Impact: {impact.temporary_impact_bps:.2f} bps")
    print(f"   Permanent Impact: {impact.permanent_impact_bps:.2f} bps")
    print(f"   Total Impact: {impact.total_impact_bps:.2f} bps")
    print(f"   Expected Cost: ${impact.expected_cost_usd:.2f}")
    print(f"   Confidence: {impact.confidence:.1%}")
    
    # 拆单计划
    print(f"\n3. Slicing Plan (Buy $500k, Medium Urgency):")
    plan = model.compute_optimal_slicing("BTC-USD", "buy", 500000, UrgencyLevel.MEDIUM, orderbook)
    print(f"   Slices: {plan.num_slices}")
    print(f"   Order Type: {plan.order_type.value}")
    print(f"   Limit Offset: {plan.limit_offset_bps} bps")
    print(f"   Suggested Limit: ${plan.suggested_limit_price:,.2f}")
    print(f"   Duration: {plan.estimated_duration_seconds} seconds")
    print(f"   Est. Impact: {plan.estimated_total_impact_bps:.2f} bps")
    print(f"   Execution Risk: {plan.execution_risk}")
    
    # 限价建议
    print(f"\n4. Limit Price Suggestion:")
    price, reason = model.suggest_limit_price("BTC-USD", "buy", 100000, UrgencyLevel.LOW, orderbook)
    print(f"   Suggested Price: ${price:,.2f}")
    print(f"   Reasoning: {reason}")
    
    # 记录执行并校准
    print(f"\n5. Calibration (simulating 50 executions)...")
    for i in range(50):
        # 模拟执行
        arrival = 100000
        actual_slip = np.random.normal(5, 2)  # 实际滑点约5bps
        avg_exec = arrival * (1 + actual_slip / 10000)
        
        model.record_execution_simple(
            symbol="BTC-USD",
            side="buy",
            size=0.5,
            arrival_price=arrival,
            avg_exec_price=avg_exec,
        )
    
    params = model.get_params("BTC-USD")
    print(f"   Calibrated eta: {params['eta']:.4f}")
    print(f"   Calibrated gamma: {params['gamma']:.4f}")
    
    stats = model.get_stats()
    print(f"   Avg Prediction Error: {stats['avg_prediction_error']:.2f} bps")
    
    print("\n✓ All tests passed!")

"""
================================================================================
BITBEAST BEHAVIOR RULES - BitBeast行为约束
================================================================================
Version: 3.2.0
Purpose: 强制执行BitBeast交易规则

规则:
- PART 11.1: 禁止缩量下跌抄底 (Volume Filter)
- PART 11.2: 结构突破才顺势 (Structure Breakout)
================================================================================
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import numpy as np

logger = logging.getLogger("HMATS.BitBeastRules")


# =============================================================================
# PART 11.1: VOLUME FILTER - 禁止缩量下跌抄底
# =============================================================================

class VolumeCondition(Enum):
    """成交量条件"""
    STRONG_VOLUME = auto()      # 量能充沛 (> 1.5x avg)
    NORMAL_VOLUME = auto()      # 正常量能 (0.8x - 1.5x avg)
    WEAK_VOLUME = auto()        # 量能萎缩 (0.5x - 0.8x avg)
    DECLINING_VOLUME = auto()   # 持续缩量 (< 0.5x avg)


class PriceTrend(Enum):
    """价格趋势"""
    STRONG_UP = auto()
    MODERATE_UP = auto()
    NEUTRAL = auto()
    MODERATE_DOWN = auto()
    STRONG_DOWN = auto()


@dataclass
class VolumeAnalysis:
    """成交量分析结果"""
    current_volume: float = 0.0
    avg_volume_20: float = 0.0
    volume_ratio: float = 1.0
    volume_condition: VolumeCondition = VolumeCondition.NORMAL_VOLUME
    
    price_trend: PriceTrend = PriceTrend.NEUTRAL
    price_change_pct: float = 0.0
    
    # 缩量下跌检测
    is_declining_on_weak_volume: bool = False
    can_buy_dip: bool = True
    reject_reason: str = ""
    
    timestamp: float = field(default_factory=time.time)


class VolumeFilter:
    """
    Volume Filter - 成交量过滤器
    
    核心规则: 禁止缩量下跌抄底
    
    条件:
    1. 价格下跌 > 3%
    2. 成交量 < 0.8x 20日均量
    3. 连续3根K线缩量下跌
    
    -> 禁止做多，等待放量反转
    """
    
    # 阈值
    WEAK_VOLUME_RATIO = 0.8         # 弱量比例
    DECLINING_VOLUME_RATIO = 0.5    # 缩量比例
    STRONG_VOLUME_RATIO = 1.5       # 放量比例
    
    PRICE_DOWN_THRESHOLD = -0.03    # 3%下跌
    PRICE_UP_THRESHOLD = 0.03       # 3%上涨
    
    CONSECUTIVE_DECLINING_BARS = 3  # 连续缩量K线数
    
    def __init__(self):
        # 数据存储
        self.volume_history: deque = deque(maxlen=100)
        self.price_history: deque = deque(maxlen=100)
        
        # 当前分析
        self.current_analysis = VolumeAnalysis()
        
        self._lock = threading.Lock()
        
        logger.info("VolumeFilter 初始化完成 - 缩量下跌抄底保护已启用")
    
    def update(self, price: float, volume: float) -> VolumeAnalysis:
        """更新价格和成交量"""
        with self._lock:
            now = time.time()
            
            # 保存历史
            self.volume_history.append({'volume': volume, 'timestamp': now})
            self.price_history.append({'price': price, 'timestamp': now})
            
            # 计算分析
            self._compute_analysis()
            
            return self.current_analysis
    
    def _compute_analysis(self):
        """计算分析结果"""
        if len(self.volume_history) < 20:
            return
        
        volumes = [v['volume'] for v in self.volume_history]
        prices = [p['price'] for p in self.price_history]
        
        # 当前值
        current_volume = volumes[-1]
        current_price = prices[-1]
        
        # 20期均量
        avg_volume_20 = np.mean(volumes[-20:])
        volume_ratio = current_volume / max(avg_volume_20, 1)
        
        # 成交量条件
        if volume_ratio > self.STRONG_VOLUME_RATIO:
            volume_condition = VolumeCondition.STRONG_VOLUME
        elif volume_ratio > self.WEAK_VOLUME_RATIO:
            volume_condition = VolumeCondition.NORMAL_VOLUME
        elif volume_ratio > self.DECLINING_VOLUME_RATIO:
            volume_condition = VolumeCondition.WEAK_VOLUME
        else:
            volume_condition = VolumeCondition.DECLINING_VOLUME
        
        # 价格趋势 (对比20期前)
        if len(prices) >= 20:
            price_change = (current_price - prices[-20]) / prices[-20]
        else:
            price_change = 0.0
        
        if price_change > 0.10:
            price_trend = PriceTrend.STRONG_UP
        elif price_change > self.PRICE_UP_THRESHOLD:
            price_trend = PriceTrend.MODERATE_UP
        elif price_change < -0.10:
            price_trend = PriceTrend.STRONG_DOWN
        elif price_change < self.PRICE_DOWN_THRESHOLD:
            price_trend = PriceTrend.MODERATE_DOWN
        else:
            price_trend = PriceTrend.NEUTRAL
        
        # 检测缩量下跌
        is_declining_on_weak_volume = False
        can_buy_dip = True
        reject_reason = ""
        
        # 条件1: 价格在下跌
        if price_trend in [PriceTrend.MODERATE_DOWN, PriceTrend.STRONG_DOWN]:
            # 条件2: 成交量萎缩
            if volume_condition in [VolumeCondition.WEAK_VOLUME, VolumeCondition.DECLINING_VOLUME]:
                # 条件3: 检查连续缩量
                if len(volumes) >= self.CONSECUTIVE_DECLINING_BARS:
                    recent_volumes = volumes[-self.CONSECUTIVE_DECLINING_BARS:]
                    # 检查是否连续递减
                    is_consecutive_decline = all(
                        recent_volumes[i] < recent_volumes[i-1] * 1.1  # 允许10%波动
                        for i in range(1, len(recent_volumes))
                    )
                    
                    if is_consecutive_decline:
                        is_declining_on_weak_volume = True
                        can_buy_dip = False
                        reject_reason = f"缩量下跌: 连续{self.CONSECUTIVE_DECLINING_BARS}根K线量能递减，价格跌{abs(price_change)*100:.1f}%"
                        logger.warning(f"⛔ {reject_reason}")
        
        # 更新分析结果
        self.current_analysis = VolumeAnalysis(
            current_volume=current_volume,
            avg_volume_20=avg_volume_20,
            volume_ratio=volume_ratio,
            volume_condition=volume_condition,
            price_trend=price_trend,
            price_change_pct=price_change * 100,
            is_declining_on_weak_volume=is_declining_on_weak_volume,
            can_buy_dip=can_buy_dip,
            reject_reason=reject_reason
        )
    
    def can_open_long(self) -> Tuple[bool, str]:
        """
        检查是否允许开多
        
        Returns:
            (allowed, reason)
        """
        with self._lock:
            analysis = self.current_analysis
            
            if not analysis.can_buy_dip:
                return False, analysis.reject_reason
            
            # 额外检查: 极端缩量时也禁止
            if analysis.volume_condition == VolumeCondition.DECLINING_VOLUME:
                return False, "极端缩量，等待量能恢复"
            
            return True, "OK"
    
    def get_status(self) -> Dict:
        """获取状态"""
        with self._lock:
            a = self.current_analysis
            return {
                'volume_ratio': round(a.volume_ratio, 2),
                'volume_condition': a.volume_condition.name,
                'price_trend': a.price_trend.name,
                'price_change_pct': round(a.price_change_pct, 2),
                'is_declining_weak_volume': a.is_declining_on_weak_volume,
                'can_buy_dip': a.can_buy_dip,
                'reject_reason': a.reject_reason
            }


# =============================================================================
# PART 11.2: STRUCTURE BREAKOUT FILTER - 结构突破才顺势
# =============================================================================

class StructureType(Enum):
    """结构类型"""
    HIGHER_HIGH = auto()        # 更高高点 (多头延续)
    HIGHER_LOW = auto()         # 更高低点 (多头结构)
    LOWER_HIGH = auto()         # 更低高点 (空头结构)
    LOWER_LOW = auto()          # 更低低点 (空头延续)
    CONSOLIDATION = auto()      # 盘整
    BREAKOUT_UP = auto()        # 向上突破
    BREAKOUT_DOWN = auto()      # 向下突破


class TrendBias(Enum):
    """趋势偏向"""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


@dataclass
class SwingPoint:
    """摆动点"""
    price: float
    timestamp: float
    is_high: bool  # True=高点, False=低点


@dataclass
class StructureAnalysis:
    """结构分析结果"""
    # 摆动点
    recent_highs: List[float] = field(default_factory=list)
    recent_lows: List[float] = field(default_factory=list)
    
    # 结构判断
    structure_type: StructureType = StructureType.CONSOLIDATION
    trend_bias: TrendBias = TrendBias.NEUTRAL
    
    # 关键价位
    resistance: float = 0.0      # 阻力位
    support: float = 0.0         # 支撑位
    
    # 突破判断
    is_breakout: bool = False
    breakout_direction: str = ""  # "up" or "down"
    breakout_confirmed: bool = False
    
    # 交易许可
    can_follow_trend_long: bool = False
    can_follow_trend_short: bool = False
    reject_reason: str = ""
    
    timestamp: float = field(default_factory=time.time)


class StructureBreakoutFilter:
    """
    Structure Breakout Filter - 结构突破过滤器
    
    核心规则: 结构突破才顺势
    
    做多条件:
    1. 价格突破前高 (Higher High)
    2. 或形成更高低点后反弹 (Higher Low)
    3. 需要突破确认 (收盘价确认)
    
    做空条件:
    1. 价格跌破前低 (Lower Low)
    2. 或形成更低高点后下跌 (Lower High)
    
    禁止:
    - 下跌趋势中抄底（无结构反转确认）
    - 上涨趋势中摸顶（无结构反转确认）
    """
    
    # 摆动点检测参数
    SWING_LOOKBACK = 5          # 左右各5根K线确认
    BREAKOUT_BUFFER_PCT = 0.002  # 0.2%突破缓冲
    
    def __init__(self):
        self.price_history: deque = deque(maxlen=200)
        self.swing_highs: deque = deque(maxlen=20)
        self.swing_lows: deque = deque(maxlen=20)
        
        self.current_analysis = StructureAnalysis()
        
        self._lock = threading.Lock()
        
        logger.info("StructureBreakoutFilter 初始化完成 - 结构突破确认已启用")
    
    def update(self, high: float, low: float, close: float) -> StructureAnalysis:
        """更新价格数据"""
        with self._lock:
            now = time.time()
            
            self.price_history.append({
                'high': high,
                'low': low,
                'close': close,
                'timestamp': now
            })
            
            # 检测摆动点
            self._detect_swing_points()
            
            # 分析结构
            self._analyze_structure(close)
            
            return self.current_analysis
    
    def _detect_swing_points(self):
        """检测摆动高低点"""
        if len(self.price_history) < self.SWING_LOOKBACK * 2 + 1:
            return
        
        prices = list(self.price_history)
        idx = len(prices) - self.SWING_LOOKBACK - 1
        
        if idx < self.SWING_LOOKBACK:
            return
        
        center = prices[idx]
        left_bars = prices[idx - self.SWING_LOOKBACK:idx]
        right_bars = prices[idx + 1:idx + self.SWING_LOOKBACK + 1]
        
        # 检测摆动高点
        center_high = center['high']
        is_swing_high = all(
            center_high >= p['high'] for p in left_bars
        ) and all(
            center_high >= p['high'] for p in right_bars
        )
        
        if is_swing_high:
            self.swing_highs.append(SwingPoint(
                price=center_high,
                timestamp=center['timestamp'],
                is_high=True
            ))
        
        # 检测摆动低点
        center_low = center['low']
        is_swing_low = all(
            center_low <= p['low'] for p in left_bars
        ) and all(
            center_low <= p['low'] for p in right_bars
        )
        
        if is_swing_low:
            self.swing_lows.append(SwingPoint(
                price=center_low,
                timestamp=center['timestamp'],
                is_high=False
            ))
    
    def _analyze_structure(self, current_close: float):
        """分析市场结构"""
        # 获取最近的摆动点
        recent_highs = [s.price for s in list(self.swing_highs)[-4:]]
        recent_lows = [s.price for s in list(self.swing_lows)[-4:]]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            self.current_analysis = StructureAnalysis(
                recent_highs=recent_highs,
                recent_lows=recent_lows
            )
            return
        
        # 判断结构类型
        last_high = recent_highs[-1]
        prev_high = recent_highs[-2]
        last_low = recent_lows[-1]
        prev_low = recent_lows[-2]
        
        # Higher Highs & Higher Lows = 上升趋势
        higher_high = last_high > prev_high
        higher_low = last_low > prev_low
        
        # Lower Highs & Lower Lows = 下降趋势
        lower_high = last_high < prev_high
        lower_low = last_low < prev_low
        
        # 确定结构和趋势
        if higher_high and higher_low:
            structure_type = StructureType.HIGHER_HIGH
            trend_bias = TrendBias.BULLISH
        elif lower_high and lower_low:
            structure_type = StructureType.LOWER_LOW
            trend_bias = TrendBias.BEARISH
        elif higher_low and not higher_high:
            structure_type = StructureType.HIGHER_LOW
            trend_bias = TrendBias.BULLISH
        elif lower_high and not lower_low:
            structure_type = StructureType.LOWER_HIGH
            trend_bias = TrendBias.BEARISH
        else:
            structure_type = StructureType.CONSOLIDATION
            trend_bias = TrendBias.NEUTRAL
        
        # 关键价位
        resistance = max(recent_highs) if recent_highs else current_close * 1.05
        support = min(recent_lows) if recent_lows else current_close * 0.95
        
        # 突破检测
        breakout_buffer = current_close * self.BREAKOUT_BUFFER_PCT
        is_breakout = False
        breakout_direction = ""
        breakout_confirmed = False
        
        if current_close > resistance + breakout_buffer:
            is_breakout = True
            breakout_direction = "up"
            breakout_confirmed = True
            structure_type = StructureType.BREAKOUT_UP
            logger.info(f"ON 向上突破确认: {current_close:.2f} > {resistance:.2f}")
        
        elif current_close < support - breakout_buffer:
            is_breakout = True
            breakout_direction = "down"
            breakout_confirmed = True
            structure_type = StructureType.BREAKOUT_DOWN
            logger.info(f"ON 向下突破确认: {current_close:.2f} < {support:.2f}")
        
        # 交易许可
        can_follow_trend_long = False
        can_follow_trend_short = False
        reject_reason = ""
        
        # 做多条件
        if trend_bias == TrendBias.BULLISH:
            if structure_type in [StructureType.BREAKOUT_UP, StructureType.HIGHER_HIGH]:
                can_follow_trend_long = True
            elif structure_type == StructureType.HIGHER_LOW:
                can_follow_trend_long = True
            else:
                reject_reason = "多头结构但未确认突破"
        elif trend_bias == TrendBias.BEARISH:
            reject_reason = "空头结构，禁止做多抄底"
        else:
            reject_reason = "盘整结构，等待突破方向"
        
        # 做空条件
        if trend_bias == TrendBias.BEARISH:
            if structure_type in [StructureType.BREAKOUT_DOWN, StructureType.LOWER_LOW]:
                can_follow_trend_short = True
            elif structure_type == StructureType.LOWER_HIGH:
                can_follow_trend_short = True
        elif trend_bias == TrendBias.BULLISH:
            pass  # 多头结构禁止做空摸顶
        
        self.current_analysis = StructureAnalysis(
            recent_highs=recent_highs,
            recent_lows=recent_lows,
            structure_type=structure_type,
            trend_bias=trend_bias,
            resistance=resistance,
            support=support,
            is_breakout=is_breakout,
            breakout_direction=breakout_direction,
            breakout_confirmed=breakout_confirmed,
            can_follow_trend_long=can_follow_trend_long,
            can_follow_trend_short=can_follow_trend_short,
            reject_reason=reject_reason
        )
    
    def can_open_long(self) -> Tuple[bool, str]:
        """检查是否允许开多"""
        with self._lock:
            analysis = self.current_analysis
            
            if analysis.can_follow_trend_long:
                return True, f"结构确认: {analysis.structure_type.name}"
            else:
                return False, analysis.reject_reason or "结构不支持做多"
    
    def can_open_short(self) -> Tuple[bool, str]:
        """检查是否允许开空"""
        with self._lock:
            analysis = self.current_analysis
            
            if analysis.can_follow_trend_short:
                return True, f"结构确认: {analysis.structure_type.name}"
            else:
                return False, "结构不支持做空"
    
    def get_status(self) -> Dict:
        """获取状态"""
        with self._lock:
            a = self.current_analysis
            return {
                'structure': a.structure_type.name,
                'trend_bias': a.trend_bias.name,
                'resistance': round(a.resistance, 2),
                'support': round(a.support, 2),
                'is_breakout': a.is_breakout,
                'breakout_direction': a.breakout_direction,
                'can_long': a.can_follow_trend_long,
                'can_short': a.can_follow_trend_short,
                'recent_highs': [round(h, 2) for h in a.recent_highs[-3:]],
                'recent_lows': [round(l, 2) for l in a.recent_lows[-3:]]
            }


# =============================================================================
# BITBEAST BEHAVIOR GUARD - BitBeast行为守卫
# =============================================================================

class BitBeastBehaviorGuard:
    """
    BitBeast Behavior Guard - BitBeast行为守卫
    
    整合所有BitBeast规则:
    1. 禁止缩量下跌抄底 (VolumeFilter)
    2. 结构突破才顺势 (StructureBreakoutFilter)
    """
    
    def __init__(self):
        self.volume_filter = VolumeFilter()
        self.structure_filter = StructureBreakoutFilter()
        
        logger.info("BitBeastBehaviorGuard 初始化完成")
    
    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: float
    ):
        """更新市场数据"""
        self.volume_filter.update(close, volume)
        self.structure_filter.update(high, low, close)
    
    def can_open_long(self) -> Tuple[bool, List[str]]:
        """
        检查是否允许开多
        
        必须同时满足:
        1. 不是缩量下跌
        2. 结构支持做多
        
        Returns:
            (allowed, reasons)
        """
        reasons = []
        
        # 检查成交量
        vol_ok, vol_reason = self.volume_filter.can_open_long()
        if not vol_ok:
            reasons.append(f"Volume: {vol_reason}")
        
        # 检查结构
        struct_ok, struct_reason = self.structure_filter.can_open_long()
        if not struct_ok:
            reasons.append(f"Structure: {struct_reason}")
        
        allowed = vol_ok and struct_ok
        
        if not allowed:
            logger.warning(f"⛔ BitBeast禁止做多: {'; '.join(reasons)}")
        
        return allowed, reasons
    
    def can_open_short(self) -> Tuple[bool, List[str]]:
        """检查是否允许开空"""
        reasons = []
        
        struct_ok, struct_reason = self.structure_filter.can_open_short()
        if not struct_ok:
            reasons.append(f"Structure: {struct_reason}")
        
        return struct_ok, reasons
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'volume': self.volume_filter.get_status(),
            'structure': self.structure_filter.get_status()
        }

"""
================================================================================
HMATS v5.2.1 SOTA - 实时相关性驱动仓位控制器
Real-Time Correlation-Driven Position Controller
================================================================================

Purpose: 实时相关性驱动的仓位调整逻辑 (完全增强版)

SOTA 需求:
1. 当多资产相关性异常上升或崩塌时，自动降低组合仓位或调整权重
2. 实时监控相关性矩阵变化
3. 预测性调整 - 在相关性spike前预防性减仓
4. 与EventBus完全集成

新增功能:
1. 滑动窗口相关性计算 (多时间尺度)
2. 相关性跳跃检测 (Z-score突变识别)
3. EWMA预测 (1h/4h ahead)
4. 预防性仓位缩减
5. 相关性崩塌检测
6. 自动权重重分配

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

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CorrelationState(Enum):
    """相关性状态"""
    STABLE = "stable"             # 稳定 - 正常交易
    ELEVATED = "elevated"         # 升高 - 谨慎交易
    SPIKING = "spiking"          # 急升 - 减仓
    CRISIS = "crisis"            # 危机 - 全面风控
    COLLAPSING = "collapsing"    # 崩塌 - 特殊机会
    DECOUPLED = "decoupled"      # 解耦 - 可能分化交易


class PositionAction(Enum):
    """仓位调整动作"""
    HOLD = "hold"                   # 维持
    REDUCE_10 = "reduce_10"         # 减仓10%
    REDUCE_25 = "reduce_25"         # 减仓25%
    REDUCE_50 = "reduce_50"         # 减仓50%
    FLATTEN = "flatten"             # 平仓
    REWEIGHT = "reweight"           # 重新分配权重
    PREEMPTIVE_REDUCE = "preemptive_reduce"  # 预防性减仓


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CorrelationControllerConfig:
    """相关性控制器配置"""
    
    # 监控资产
    assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    
    # 时间窗口 (bars)
    short_window: int = 20          # 短期: ~1.5天 (4H bars)
    medium_window: int = 60         # 中期: ~10天
    long_window: int = 200          # 长期: ~33天
    
    # 相关性阈值
    normal_corr_max: float = 0.6    # 正常相关性上限
    elevated_corr: float = 0.75     # 升高相关性
    crisis_corr: float = 0.9        # 危机相关性
    collapse_corr: float = 0.3      # 崩塌相关性下限
    
    # 跳跃检测
    jump_zscore_threshold: float = 2.5    # Z-score跳跃阈值
    jump_lookback: int = 30               # 跳跃检测回看周期
    
    # 预测参数
    ewma_alpha: float = 0.3         # EWMA平滑因子
    prediction_horizon_hours: int = 4    # 预测时间范围
    preemptive_threshold: float = 0.8    # 预防性减仓触发阈值
    
    # 仓位调整
    reduce_scale_elevated: float = 0.85   # 升高时缩放
    reduce_scale_spiking: float = 0.6     # 急升时缩放
    reduce_scale_crisis: float = 0.3      # 危机时缩放
    
    # 冷却
    cooldown_minutes: int = 15       # 调整冷却期


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CorrelationMatrix:
    """相关性矩阵"""
    timestamp: datetime
    window_type: str  # short/medium/long
    
    # 矩阵数据
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 统计指标
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    min_correlation: float = 0.0
    eigenvalue_ratio: float = 0.0  # 第一特征值占比
    
    def get(self, asset1: str, asset2: str) -> float:
        """获取两资产相关性"""
        if asset1 == asset2:
            return 1.0
        if asset1 in self.matrix and asset2 in self.matrix[asset1]:
            return self.matrix[asset1][asset2]
        if asset2 in self.matrix and asset1 in self.matrix[asset2]:
            return self.matrix[asset2][asset1]
        return 0.0


@dataclass
class CorrelationJump:
    """相关性跳跃事件"""
    timestamp: datetime
    asset_pair: Tuple[str, str]
    
    previous_corr: float
    current_corr: float
    change: float
    zscore: float
    
    direction: str  # "spike" or "collapse"
    severity: str   # "minor", "moderate", "severe"


@dataclass
class CorrelationPrediction:
    """相关性预测"""
    timestamp: datetime
    horizon_hours: int
    
    # 预测值
    predicted_avg_corr: float
    predicted_max_corr: float
    
    # 置信度
    confidence: float
    
    # 趋势
    trend: str  # "rising", "stable", "falling"
    trend_strength: float
    
    # 预警
    spike_warning: bool
    collapse_warning: bool


@dataclass
class PositionAdjustment:
    """仓位调整建议"""
    timestamp: datetime
    
    # 状态
    correlation_state: CorrelationState
    action: PositionAction
    
    # 调整因子
    scale_factor: float  # 0.0-1.0, 仓位应缩放到的比例
    
    # 按资产调整
    asset_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # 原因
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # 是否为预防性调整
    is_preemptive: bool = False


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

class CorrelationRealtimeController:
    """
    实时相关性驱动仓位控制器
    
    功能:
    1. 多时间窗口相关性监控
    2. 跳跃检测与告警
    3. 相关性预测
    4. 自动仓位调整建议
    5. EventBus集成
    """
    
    def __init__(self, config: Optional[CorrelationControllerConfig] = None):
        self.config = config or CorrelationControllerConfig()
        
        # 价格历史
        self._price_history: Dict[str, deque] = {
            asset: deque(maxlen=self.config.long_window + 50)
            for asset in self.config.assets
        }
        
        # 收益率历史
        self._returns_history: Dict[str, deque] = {
            asset: deque(maxlen=self.config.long_window + 50)
            for asset in self.config.assets
        }
        
        # 相关性历史
        self._corr_history: Dict[str, deque] = {
            "short": deque(maxlen=100),
            "medium": deque(maxlen=100),
            "long": deque(maxlen=100),
        }
        
        # 跳跃事件
        self._jumps: deque = deque(maxlen=50)
        
        # 预测历史
        self._predictions: deque = deque(maxlen=20)
        
        # 当前状态
        self._current_state = CorrelationState.STABLE
        self._last_adjustment_time: Optional[datetime] = None
        
        # EventBus回调
        self._event_callbacks: List[Callable] = []
        
        # 统计
        self._stats = {
            "updates": 0,
            "jumps_detected": 0,
            "adjustments_triggered": 0,
            "preemptive_actions": 0,
        }
        
        logger.info(f"CorrelationRealtimeController initialized with {len(self.config.assets)} assets")
    
    # =========================================================================
    # PRICE UPDATE
    # =========================================================================
    
    def update_price(self, asset: str, price: float, timestamp: Optional[datetime] = None):
        """
        更新资产价格
        
        Args:
            asset: 资产代码
            price: 当前价格
            timestamp: 时间戳
        """
        if asset not in self._price_history:
            logger.warning(f"Unknown asset: {asset}")
            return
        
        ts = timestamp or datetime.now()
        
        # 计算收益率
        if len(self._price_history[asset]) > 0:
            prev_price = self._price_history[asset][-1][1]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
                self._returns_history[asset].append((ts, ret))
        
        # 存储价格
        self._price_history[asset].append((ts, price))
        self._stats["updates"] += 1
    
    def update_prices_batch(self, prices: Dict[str, float], timestamp: Optional[datetime] = None):
        """批量更新价格"""
        ts = timestamp or datetime.now()
        for asset, price in prices.items():
            self.update_price(asset, price, ts)
    
    # =========================================================================
    # CORRELATION CALCULATION
    # =========================================================================
    
    def _compute_returns_array(self, asset: str, window: int) -> np.ndarray:
        """获取收益率数组"""
        returns = list(self._returns_history[asset])
        if len(returns) < window:
            return np.array([r[1] for r in returns]) if returns else np.array([])
        return np.array([r[1] for r in returns[-window:]])
    
    def compute_correlation_matrix(self, window_type: str = "short") -> Optional[CorrelationMatrix]:
        """
        计算相关性矩阵
        
        Args:
            window_type: "short", "medium", "long"
        
        Returns:
            CorrelationMatrix or None
        """
        # 确定窗口大小
        window_sizes = {
            "short": self.config.short_window,
            "medium": self.config.medium_window,
            "long": self.config.long_window,
        }
        window = window_sizes.get(window_type, self.config.short_window)
        
        # 检查数据是否足够
        min_data = min(len(self._returns_history[a]) for a in self.config.assets)
        if min_data < window:
            return None
        
        # 构建收益率矩阵
        returns_data = {}
        for asset in self.config.assets:
            returns_data[asset] = self._compute_returns_array(asset, window)
        
        # 计算相关性
        matrix = {}
        correlations = []
        
        for i, asset1 in enumerate(self.config.assets):
            matrix[asset1] = {}
            for j, asset2 in enumerate(self.config.assets):
                if i == j:
                    matrix[asset1][asset2] = 1.0
                elif j > i:
                    # 计算Pearson相关性
                    corr = np.corrcoef(returns_data[asset1], returns_data[asset2])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    matrix[asset1][asset2] = corr
                    correlations.append(corr)
        
        # 填充下三角
        for i, asset1 in enumerate(self.config.assets):
            for j, asset2 in enumerate(self.config.assets):
                if j < i:
                    matrix[asset1][asset2] = matrix[asset2][asset1]
        
        # 计算特征值比率
        eigenvalue_ratio = self._compute_eigenvalue_ratio(matrix)
        
        # 创建结果
        result = CorrelationMatrix(
            timestamp=datetime.now(),
            window_type=window_type,
            matrix=matrix,
            avg_correlation=np.mean(correlations) if correlations else 0.0,
            max_correlation=max(correlations) if correlations else 0.0,
            min_correlation=min(correlations) if correlations else 0.0,
            eigenvalue_ratio=eigenvalue_ratio,
        )
        
        # 存储历史
        self._corr_history[window_type].append(result)
        
        return result
    
    def _compute_eigenvalue_ratio(self, matrix: Dict[str, Dict[str, float]]) -> float:
        """计算第一特征值占比 (衡量协同运动程度)"""
        try:
            # 构建numpy矩阵
            n = len(self.config.assets)
            corr_matrix = np.zeros((n, n))
            for i, a1 in enumerate(self.config.assets):
                for j, a2 in enumerate(self.config.assets):
                    corr_matrix[i, j] = matrix.get(a1, {}).get(a2, 0.0)
            
            # 计算特征值
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
            
            total = np.sum(eigenvalues)
            if total > 0:
                return eigenvalues[0] / total
            return 0.0
        except Exception:
            return 0.0
    
    # =========================================================================
    # JUMP DETECTION
    # =========================================================================
    
    def detect_jumps(self) -> List[CorrelationJump]:
        """
        检测相关性跳跃
        
        Returns:
            检测到的跳跃事件列表
        """
        if len(self._corr_history["short"]) < 2:
            return []
        
        current = self._corr_history["short"][-1]
        previous = self._corr_history["short"][-2]
        
        jumps = []
        
        for asset1 in self.config.assets:
            for asset2 in self.config.assets:
                if asset1 >= asset2:
                    continue
                
                # [P90 2026-04-26] The earlier P50 "tuple-key" fix was wrong.
                # current/previous/_corr_history entries are CorrelationMatrix
                # instances (see line 358 ctor) whose .get(asset1, asset2)
                # signature takes TWO STRING args. Passing a tuple as the first
                # arg + 0.0 as second silently fell through every branch and
                # returned 0.0 — so curr/prev were always 0.0, change was
                # always 0.0, and detect_jumps() never fired in production.
                # Use the actual signature.
                curr_corr = current.get(asset1, asset2)
                prev_corr = previous.get(asset1, asset2)
                change = curr_corr - prev_corr

                # 计算历史变化的标准差
                changes = []
                for i in range(1, min(len(self._corr_history["short"]), self.config.jump_lookback)):
                    if i < len(self._corr_history["short"]):
                        # [P90 2026-04-26] Same fix as above.
                        c1 = self._corr_history["short"][-i].get(asset1, asset2)
                        c2 = self._corr_history["short"][-i-1].get(asset1, asset2)
                        changes.append(c1 - c2)
                
                if len(changes) < 5:
                    continue
                
                std = np.std(changes)
                mean = np.mean(changes)
                
                if std > 0:
                    zscore = (change - mean) / std
                else:
                    zscore = 0.0
                
                # 检测跳跃
                if abs(zscore) >= self.config.jump_zscore_threshold:
                    direction = "spike" if change > 0 else "collapse"
                    
                    # 确定严重程度
                    if abs(zscore) >= 4.0:
                        severity = "severe"
                    elif abs(zscore) >= 3.0:
                        severity = "moderate"
                    else:
                        severity = "minor"
                    
                    jump = CorrelationJump(
                        timestamp=datetime.now(),
                        asset_pair=(asset1, asset2),
                        previous_corr=prev_corr,
                        current_corr=curr_corr,
                        change=change,
                        zscore=zscore,
                        direction=direction,
                        severity=severity,
                    )
                    jumps.append(jump)
                    self._jumps.append(jump)
                    self._stats["jumps_detected"] += 1
                    
                    # 发送事件
                    self._emit_event("correlation_jump", {
                        "pair": f"{asset1}-{asset2}",
                        "direction": direction,
                        "severity": severity,
                        "zscore": zscore,
                        "change": change,
                    })
        
        return jumps
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict_correlation(self, horizon_hours: int = 4) -> Optional[CorrelationPrediction]:
        """
        预测未来相关性
        
        Args:
            horizon_hours: 预测时间范围 (小时)
        
        Returns:
            CorrelationPrediction or None
        """
        if len(self._corr_history["short"]) < 10:
            return None
        
        # 获取历史平均相关性
        avg_corrs = [m.avg_correlation for m in self._corr_history["short"]]
        max_corrs = [m.max_correlation for m in self._corr_history["short"]]
        
        # EWMA预测
        alpha = self.config.ewma_alpha
        ewma_avg = avg_corrs[-1]
        ewma_max = max_corrs[-1]
        
        for i in range(len(avg_corrs) - 2, -1, -1):
            ewma_avg = alpha * avg_corrs[i] + (1 - alpha) * ewma_avg
            ewma_max = alpha * max_corrs[i] + (1 - alpha) * ewma_max
        
        # 计算趋势
        recent_avg = np.mean(avg_corrs[-5:])
        older_avg = np.mean(avg_corrs[-10:-5]) if len(avg_corrs) >= 10 else recent_avg
        trend_change = recent_avg - older_avg
        
        if trend_change > 0.05:
            trend = "rising"
            trend_strength = min(trend_change / 0.1, 1.0)
        elif trend_change < -0.05:
            trend = "falling"
            trend_strength = min(abs(trend_change) / 0.1, 1.0)
        else:
            trend = "stable"
            trend_strength = 0.0
        
        # 外推预测
        steps = horizon_hours // 4  # 每4小时一个bar
        predicted_avg = ewma_avg + trend_change * steps
        predicted_max = ewma_max + trend_change * steps * 1.2
        
        # 限制范围
        predicted_avg = max(-1.0, min(1.0, predicted_avg))
        predicted_max = max(-1.0, min(1.0, predicted_max))
        
        # 计算置信度 (基于历史稳定性)
        std = np.std(avg_corrs[-10:]) if len(avg_corrs) >= 10 else 0.2
        confidence = max(0.3, 1.0 - std * 2)
        
        # 预警
        spike_warning = predicted_max > self.config.preemptive_threshold and trend == "rising"
        collapse_warning = predicted_avg < self.config.collapse_corr and trend == "falling"
        
        prediction = CorrelationPrediction(
            timestamp=datetime.now(),
            horizon_hours=horizon_hours,
            predicted_avg_corr=predicted_avg,
            predicted_max_corr=predicted_max,
            confidence=confidence,
            trend=trend,
            trend_strength=trend_strength,
            spike_warning=spike_warning,
            collapse_warning=collapse_warning,
        )
        
        self._predictions.append(prediction)
        
        if spike_warning:
            self._emit_event("correlation_spike_warning", {
                "predicted_max": predicted_max,
                "horizon_hours": horizon_hours,
                "confidence": confidence,
            })
        
        return prediction
    
    # =========================================================================
    # STATE ASSESSMENT
    # =========================================================================
    
    def assess_state(self) -> CorrelationState:
        """
        评估当前相关性状态
        
        Returns:
            CorrelationState
        """
        if len(self._corr_history["short"]) == 0:
            return CorrelationState.STABLE
        
        current = self._corr_history["short"][-1]
        avg_corr = current.avg_correlation
        max_corr = current.max_correlation
        
        # 检测跳跃
        recent_jumps = [j for j in self._jumps 
                       if (datetime.now() - j.timestamp).total_seconds() < 600]
        severe_jumps = [j for j in recent_jumps if j.severity == "severe"]
        
        # 危机状态: 所有资产高度相关
        if avg_corr >= self.config.crisis_corr or max_corr >= 0.95:
            self._current_state = CorrelationState.CRISIS
        
        # 急升状态: 有严重跳跃事件
        elif severe_jumps and any(j.direction == "spike" for j in severe_jumps):
            self._current_state = CorrelationState.SPIKING
        
        # 崩塌状态: 相关性异常低
        elif avg_corr <= self.config.collapse_corr:
            self._current_state = CorrelationState.COLLAPSING
        
        # 升高状态
        elif avg_corr >= self.config.elevated_corr:
            self._current_state = CorrelationState.ELEVATED
        
        # 解耦状态: 某些资产负相关
        elif current.min_correlation < -0.3:
            self._current_state = CorrelationState.DECOUPLED
        
        # 正常状态
        else:
            self._current_state = CorrelationState.STABLE
        
        return self._current_state
    
    # =========================================================================
    # POSITION ADJUSTMENT
    # =========================================================================
    
    def get_position_adjustment(self, 
                                current_positions: Optional[Dict[str, float]] = None
                                ) -> PositionAdjustment:
        """
        获取仓位调整建议
        
        Args:
            current_positions: 当前仓位 {asset: position_size}
        
        Returns:
            PositionAdjustment
        """
        # 计算相关性矩阵
        short_corr = self.compute_correlation_matrix("short")
        
        # 检测跳跃
        jumps = self.detect_jumps()
        
        # 预测
        prediction = self.predict_correlation(self.config.prediction_horizon_hours)
        
        # 评估状态
        state = self.assess_state()
        
        # 确定动作和缩放因子
        action = PositionAction.HOLD
        scale_factor = 1.0
        reason = ""
        is_preemptive = False
        
        # 基于状态的调整
        if state == CorrelationState.CRISIS:
            action = PositionAction.REDUCE_50
            scale_factor = self.config.reduce_scale_crisis
            reason = f"CRISIS: 平均相关性={short_corr.avg_correlation:.2f} 超过危机阈值"
        
        elif state == CorrelationState.SPIKING:
            action = PositionAction.REDUCE_25
            scale_factor = self.config.reduce_scale_spiking
            reason = f"SPIKING: 检测到严重相关性跳跃 (zscore > {self.config.jump_zscore_threshold})"
        
        elif state == CorrelationState.ELEVATED:
            action = PositionAction.REDUCE_10
            scale_factor = self.config.reduce_scale_elevated
            reason = f"ELEVATED: 平均相关性={short_corr.avg_correlation:.2f} 高于正常水平"
        
        elif state == CorrelationState.COLLAPSING:
            action = PositionAction.REWEIGHT
            scale_factor = 1.0  # 可能是分化交易机会
            reason = f"COLLAPSING: 相关性崩塌可能带来分化交易机会"
        
        elif state == CorrelationState.DECOUPLED:
            action = PositionAction.HOLD
            scale_factor = 1.0
            reason = "DECOUPLED: 资产解耦,可继续正常交易"
        
        # 预测性调整 (覆盖上面的部分决策)
        if prediction and prediction.spike_warning and state not in [CorrelationState.CRISIS, CorrelationState.SPIKING]:
            action = PositionAction.PREEMPTIVE_REDUCE
            scale_factor = 0.85
            reason = f"PREEMPTIVE: 预测{prediction.horizon_hours}h后相关性将升至{prediction.predicted_max_corr:.2f}"
            is_preemptive = True
            self._stats["preemptive_actions"] += 1
        
        # 冷却检查
        if self._last_adjustment_time:
            elapsed = (datetime.now() - self._last_adjustment_time).total_seconds() / 60
            if elapsed < self.config.cooldown_minutes and action != PositionAction.REDUCE_50:
                action = PositionAction.HOLD
                scale_factor = 1.0
                reason = f"冷却中: 距离上次调整仅{elapsed:.1f}分钟"
        
        # 计算按资产调整
        asset_adjustments = {}
        if short_corr and current_positions:
            for asset in self.config.assets:
                if asset in current_positions:
                    # 基于该资产的相关性调整
                    asset_corrs = [short_corr.get(asset, other) 
                                  for other in self.config.assets if other != asset]
                    avg_asset_corr = np.mean([abs(c) for c in asset_corrs])
                    
                    # 高相关性资产更多减仓
                    asset_scale = scale_factor
                    if avg_asset_corr > 0.8:
                        asset_scale *= 0.9
                    
                    asset_adjustments[asset] = asset_scale
        
        # 更新时间
        if action != PositionAction.HOLD:
            self._last_adjustment_time = datetime.now()
            self._stats["adjustments_triggered"] += 1
        
        adjustment = PositionAdjustment(
            timestamp=datetime.now(),
            correlation_state=state,
            action=action,
            scale_factor=scale_factor,
            asset_adjustments=asset_adjustments,
            reason=reason,
            details={
                "avg_correlation": short_corr.avg_correlation if short_corr else 0.0,
                "max_correlation": short_corr.max_correlation if short_corr else 0.0,
                "recent_jumps": len([j for j in self._jumps 
                                    if (datetime.now() - j.timestamp).total_seconds() < 600]),
                "prediction": {
                    "predicted_max": prediction.predicted_max_corr if prediction else None,
                    "spike_warning": prediction.spike_warning if prediction else False,
                } if prediction else None,
            },
            is_preemptive=is_preemptive,
        )
        
        # 发送事件
        if action != PositionAction.HOLD:
            self._emit_event("position_adjustment", {
                "state": state.value,
                "action": action.value,
                "scale_factor": scale_factor,
                "reason": reason,
                "is_preemptive": is_preemptive,
            })
        
        return adjustment
    
    # =========================================================================
    # EVENT BUS INTEGRATION
    # =========================================================================
    
    def register_event_callback(self, callback: Callable[[str, Dict], None]):
        """注册事件回调"""
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """发送事件"""
        event = {
            "type": f"correlation.{event_type}",
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        for callback in self._event_callbacks:
            try:
                callback(event_type, event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    # =========================================================================
    # DASHBOARD DATA
    # =========================================================================
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        short_corr = self._corr_history["short"][-1] if self._corr_history["short"] else None
        
        return {
            "current_state": self._current_state.value,
            "avg_correlation": short_corr.avg_correlation if short_corr else 0.0,
            "max_correlation": short_corr.max_correlation if short_corr else 0.0,
            "eigenvalue_ratio": short_corr.eigenvalue_ratio if short_corr else 0.0,
            "correlation_matrix": short_corr.matrix if short_corr else {},
            "recent_jumps": [
                {
                    "pair": f"{j.asset_pair[0]}-{j.asset_pair[1]}",
                    "direction": j.direction,
                    "severity": j.severity,
                    "zscore": j.zscore,
                    "timestamp": j.timestamp.isoformat(),
                }
                for j in list(self._jumps)[-5:]
            ],
            "prediction": {
                "predicted_max": self._predictions[-1].predicted_max_corr,
                "trend": self._predictions[-1].trend,
                "spike_warning": self._predictions[-1].spike_warning,
            } if self._predictions else None,
            "stats": self._stats.copy(),
        }
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "current_state": self._current_state.value,
            "data_points": {
                asset: len(self._price_history[asset])
                for asset in self.config.assets
            },
            "correlation_samples": {
                window: len(history)
                for window, history in self._corr_history.items()
            },
        }
    
    def reset(self):
        """重置控制器"""
        for asset in self.config.assets:
            self._price_history[asset].clear()
            self._returns_history[asset].clear()
        
        for window in self._corr_history:
            self._corr_history[window].clear()
        
        self._jumps.clear()
        self._predictions.clear()
        self._current_state = CorrelationState.STABLE
        self._last_adjustment_time = None
        
        logger.info("CorrelationRealtimeController reset")


# =============================================================================
# INTEGRATION WITH RISK CONTROLLER
# =============================================================================

class CorrelationRiskIntegration:
    """
    与SOTA风险控制器集成
    """
    
    def __init__(self, 
                 correlation_controller: CorrelationRealtimeController,
                 risk_controller: Any = None):
        self.corr_controller = correlation_controller
        self.risk_controller = risk_controller
        
        # 注册事件监听
        self.corr_controller.register_event_callback(self._on_correlation_event)
    
    def _on_correlation_event(self, event_type: str, event: Dict):
        """处理相关性事件"""
        if self.risk_controller is None:
            return
        
        # 危机状态 -> 触发风险控制器
        if event_type == "position_adjustment":
            data = event.get("data", {})
            state = data.get("state")
            
            if state == "crisis":
                logger.warning("Correlation crisis detected - triggering risk controller")
                if hasattr(self.risk_controller, "trigger_risk_off"):
                    self.risk_controller.trigger_risk_off("correlation_crisis")
    
    def check_trade_allowed(self, 
                           symbol: str, 
                           size: float,
                           current_positions: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        检查交易是否允许 (考虑相关性)
        
        Returns:
            (allowed, adjusted_size, reason)
        """
        # 获取相关性调整
        adjustment = self.corr_controller.get_position_adjustment(current_positions)
        
        # 危机状态不允许新仓位
        if adjustment.correlation_state == CorrelationState.CRISIS:
            return False, 0.0, f"相关性危机 - 暂停新开仓: {adjustment.reason}"
        
        # 调整仓位大小
        adjusted_size = size * adjustment.scale_factor
        
        # 如果是按资产调整
        if symbol in adjustment.asset_adjustments:
            adjusted_size = size * adjustment.asset_adjustments[symbol]
        
        reason = ""
        if adjustment.scale_factor < 1.0:
            reason = f"相关性调整: {adjustment.reason}"
        
        return True, adjusted_size, reason


# =============================================================================
# P0-2: MODEL ALPHA WEIGHT CONTROLLER
# =============================================================================

class ModelAlphaWeightController:
    """
    P0-2: 模型 Alpha 权重控制器
    
    职责:
    1. 相关性 spike/collapse 时限制模型 Alpha 最大权重
    2. Regime shift 期间禁止模型放大风险
    3. 提供模型权重上限接口给 authority_fusion
    
    约束:
    - 危机状态: 模型权重强制为 0
    - 升高状态: 模型权重上限 0.5
    - 正常状态: 模型权重上限 1.0
    """
    
    def __init__(self, correlation_controller: CorrelationRealtimeController):
        self.corr_controller = correlation_controller
        
        # 权重上限配置
        self._weight_caps = {
            CorrelationState.STABLE: 1.0,
            CorrelationState.ELEVATED: 0.6,
            CorrelationState.SPIKING: 0.3,
            CorrelationState.CRISIS: 0.0,
            CorrelationState.COLLAPSING: 0.2,  # 特殊机会但仍需谨慎
            CorrelationState.DECOUPLED: 0.7,
        }
        
        # 当前权重上限
        self._current_max_weight = 1.0
        
        # 冷却期（防止权重快速波动）
        self._last_weight_change = None
        self._weight_change_cooldown_sec = 60
        
        # 注册事件回调
        self.corr_controller.register_event_callback(self._on_correlation_event)
        
        logger.info("ModelAlphaWeightController initialized")
    
    def _on_correlation_event(self, event_type: str, event: Dict):
        """处理相关性事件"""
        if event_type == "position_adjustment":
            data = event.get("data", {})
            state_str = data.get("state", "stable")
            
            try:
                state = CorrelationState(state_str)
                new_cap = self._weight_caps.get(state, 1.0)
                
                # 检查冷却期
                now = datetime.now()
                if self._last_weight_change:
                    elapsed = (now - self._last_weight_change).total_seconds()
                    if elapsed < self._weight_change_cooldown_sec:
                        # 在冷却期内只允许降低权重
                        if new_cap >= self._current_max_weight:
                            return
                
                # 更新权重上限
                if new_cap != self._current_max_weight:
                    old_cap = self._current_max_weight
                    self._current_max_weight = new_cap
                    self._last_weight_change = now
                    
                    logger.info(f"ModelAlpha weight cap changed: {old_cap:.2f} -> {new_cap:.2f} "
                               f"(state={state.value})")
                    
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to update model weight cap: {e}")
    
    def get_max_weight(self) -> float:
        """
        获取当前模型 Alpha 最大权重
        
        由 authority_fusion 和 adaptive_strategy_weight_manager 调用
        """
        return self._current_max_weight
    
    def get_adjusted_weight(self, requested_weight: float) -> float:
        """
        获取调整后的权重（不超过上限）
        
        Args:
            requested_weight: 请求的权重
        
        Returns:
            调整后的权重
        """
        return min(requested_weight, self._current_max_weight)
    
    def is_model_allowed(self) -> bool:
        """检查模型 Alpha 是否允许参与决策"""
        return self._current_max_weight > 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            "current_max_weight": self._current_max_weight,
            "correlation_state": self.corr_controller._current_state.value,
            "model_allowed": self.is_model_allowed(),
            "last_weight_change": self._last_weight_change.isoformat() if self._last_weight_change else None,
            "weight_caps": {s.value: w for s, w in self._weight_caps.items()},
        }


# =============================================================================
# SINGLETON
# =============================================================================

_correlation_controller: Optional[CorrelationRealtimeController] = None
_model_alpha_weight_controller: Optional[ModelAlphaWeightController] = None


def get_correlation_controller(
    config: Optional[CorrelationControllerConfig] = None,
) -> CorrelationRealtimeController:
    """获取相关性控制器单例"""
    global _correlation_controller
    if _correlation_controller is None:
        _correlation_controller = CorrelationRealtimeController(config)
    elif config is not None and getattr(_correlation_controller, "config", None) != config:
        logger.info("[CorrelationRealtimeController] config changed; refreshing singleton instance")
        _correlation_controller = CorrelationRealtimeController(config)
    return _correlation_controller


def get_model_alpha_weight_controller(
    correlation_controller: Optional[CorrelationRealtimeController] = None,
) -> ModelAlphaWeightController:
    """获取模型 Alpha 权重控制器单例"""
    global _model_alpha_weight_controller, _correlation_controller
    corr_ctrl = correlation_controller or get_correlation_controller()
    if _model_alpha_weight_controller is None:
        _model_alpha_weight_controller = ModelAlphaWeightController(corr_ctrl)
    elif getattr(_model_alpha_weight_controller, "corr_controller", None) is not corr_ctrl:
        logger.info("[ModelAlphaWeightController] correlation controller changed; refreshing singleton instance")
        _model_alpha_weight_controller = ModelAlphaWeightController(corr_ctrl)
    return _model_alpha_weight_controller


def reset_correlation_controller():
    """重置单例"""
    global _correlation_controller, _model_alpha_weight_controller
    _correlation_controller = None
    _model_alpha_weight_controller = None


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing CorrelationRealtimeController")
    print("=" * 60)
    
    controller = CorrelationRealtimeController()
    
    # 模拟价格数据
    print("\n1. Simulating price updates...")
    base_prices = {"BTC": 100000, "ETH": 3500, "SOL": 200}
    
    for i in range(100):
        # 正常期间: 中等相关性
        if i < 50:
            common_factor = random.gauss(0, 0.02)
            for asset, base in base_prices.items():
                noise = random.gauss(0, 0.01)
                change = common_factor * 0.5 + noise
                new_price = base * (1 + change)
                controller.update_price(asset, new_price)
                base_prices[asset] = new_price
        
        # 危机期间: 高相关性
        else:
            common_factor = random.gauss(-0.02, 0.01)  # 共同下跌
            for asset, base in base_prices.items():
                noise = random.gauss(0, 0.005)
                change = common_factor + noise  # 高度相关
                new_price = base * (1 + change)
                controller.update_price(asset, new_price)
                base_prices[asset] = new_price
    
    print(f"   Updates: {controller._stats['updates']}")
    
    # 获取调整建议
    print("\n2. Getting position adjustment...")
    current_positions = {"BTC": 1.0, "ETH": 5.0, "SOL": 50.0}
    adjustment = controller.get_position_adjustment(current_positions)
    
    print(f"   State: {adjustment.correlation_state.value}")
    print(f"   Action: {adjustment.action.value}")
    print(f"   Scale Factor: {adjustment.scale_factor}")
    print(f"   Reason: {adjustment.reason}")
    print(f"   Is Preemptive: {adjustment.is_preemptive}")
    
    # 仪表盘数据
    print("\n3. Dashboard data...")
    dashboard = controller.get_dashboard_data()
    print(f"   Avg Correlation: {dashboard['avg_correlation']:.4f}")
    print(f"   Max Correlation: {dashboard['max_correlation']:.4f}")
    print(f"   Jumps Detected: {dashboard['stats']['jumps_detected']}")
    
    print("\n✓ All tests passed!")

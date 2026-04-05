"""
================================================================================
HMATS v5.2.1 - 增强型相关性动态风险控制器
Enhanced Correlation Dynamic Risk Controller
================================================================================
Purpose: 实时相关性驱动的仓位调整逻辑 (增强版)

SOTA 对照:
1. 当多资产相关性异常上升或崩塌时，自动降低组合仓位或调整权重
2. 实时监控 BTC-ETH-SOL 相关性矩阵
3. 检测相关性 regime 变化并触发风险响应
4. **新增** 滚动相关性预测 (EWMA + 趋势外推)
5. **新增** 相关性跳跃检测 (z-score 突变)
6. **新增** EventBus 事件发布
7. **新增** 与 SOTA Risk Controller 集成

Features:
1. 滚动相关性矩阵计算 (多窗口)
2. 相关性异常检测 (spike/collapse/jump)
3. 动态仓位缩放因子 + 预测性调整
4. 组合级风险聚合
5. EventBus 集成 + 告警发布
6. 自动与 sota_risk_controller 联动

Author: HMATS v5.2.1 SOTA Upgrade
Version: 5.2.1
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class CorrelationRegime(Enum):
    """相关性regime状态"""
    NORMAL = "normal"               # 正常相关性
    ELEVATED = "elevated"           # 相关性升高
    CRISIS = "crisis"               # 相关性危机 (全部趋近1)
    DECOUPLED = "decoupled"         # 相关性解耦
    BREAKDOWN = "breakdown"         # 相关性崩塌
    TRANSITION = "transition"       # 转换中
    JUMPING = "jumping"             # 跳跃中 (新增)


class RiskAction(Enum):
    """风险响应动作"""
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    EXIT_CORRELATED = "exit_correlated"
    HEDGE_REQUIRED = "hedge_required"
    HALT_NEW_POSITIONS = "halt_new_positions"
    EMERGENCY_FLATTEN = "emergency_flatten"
    PREEMPTIVE_REDUCE = "preemptive_reduce"  # 新增: 预测性减仓


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CorrelationSnapshot:
    """相关性矩阵快照"""
    timestamp: datetime
    matrix: Dict[str, Dict[str, float]]  # symbol -> symbol -> correlation
    eigenvalues: np.ndarray = None       # 主成分特征值
    explained_variance: float = 0.0      # 第一主成分解释方差比
    condition_number: float = 1.0        # 条件数 (衡量稳定性)
    
    def get_correlation(self, sym1: str, sym2: str) -> float:
        """获取两资产相关性"""
        if sym1 == sym2:
            return 1.0
        if sym1 in self.matrix and sym2 in self.matrix[sym1]:
            return self.matrix[sym1][sym2]
        if sym2 in self.matrix and sym1 in self.matrix[sym2]:
            return self.matrix[sym2][sym1]
        return 0.0
    
    def average_correlation(self) -> float:
        """计算平均相关性 (非对角线)"""
        values = []
        for sym1, row in self.matrix.items():
            for sym2, corr in row.items():
                if sym1 != sym2:
                    values.append(abs(corr))
        return np.mean(values) if values else 0.0


@dataclass
class CorrelationPrediction:
    """相关性预测结果 (v5.2.1 新增)"""
    timestamp: datetime
    predicted_avg_corr_1h: float      # 1小时后预测
    predicted_avg_corr_4h: float      # 4小时后预测
    prediction_confidence: float       # 预测置信度
    trend_direction: str              # "rising", "falling", "stable"
    jump_probability: float           # 跳跃概率 (0-1)


@dataclass
class CorrelationDynamicsConfigV521:
    """相关性动态控制配置 (v5.2.1 增强)"""
    # 计算参数
    lookback_bars: int = 100           # 相关性计算窗口
    short_lookback_bars: int = 20      # 短期窗口
    long_lookback_bars: int = 200      # 长期窗口
    update_frequency_bars: int = 1     # 更新频率
    min_samples: int = 30              # 最小样本数
    
    # 相关性阈值
    normal_btc_eth: float = 0.80       # BTC-ETH 正常相关性
    normal_btc_sol: float = 0.70       # BTC-SOL 正常相关性
    normal_eth_sol: float = 0.75       # ETH-SOL 正常相关性
    
    # 异常检测
    spike_threshold: float = 0.15      # 相关性上升阈值
    collapse_threshold: float = 0.25   # 相关性下降阈值
    crisis_threshold: float = 0.95     # 危机阈值 (所有相关性 > 0.95)
    decoupling_threshold: float = 0.30 # 解耦阈值 (相关性 < 0.3)
    
    # 跳跃检测 (v5.2.1 新增)
    jump_zscore_threshold: float = 3.0  # z-score 阈值
    jump_min_change: float = 0.10       # 最小变化幅度
    
    # EWMA 预测 (v5.2.1 新增)
    ewma_span: int = 20                 # EWMA span
    prediction_horizon_bars: int = 6    # 预测前瞻 (4H bar = 24h)
    
    # 仓位调整
    max_correlated_exposure: float = 0.60  # 高相关资产最大敞口
    crisis_position_scale: float = 0.30    # 危机时仓位缩放
    elevated_position_scale: float = 0.70  # 升高时仓位缩放
    preemptive_scale_factor: float = 0.85  # 预测性减仓 (v5.2.1)
    
    # 风险限制
    max_portfolio_correlation: float = 0.85  # 组合最大平均相关性
    eigenvalue_concentration_limit: float = 0.80  # 第一主成分限制
    
    # EventBus 集成 (v5.2.1 新增)
    emit_events: bool = True
    alert_on_regime_change: bool = True
    alert_on_jump: bool = True


@dataclass
class CorrelationRiskAssessmentV521:
    """相关性风险评估结果 (v5.2.1 增强)"""
    timestamp: datetime
    regime: CorrelationRegime
    action: RiskAction
    
    # 相关性指标
    avg_correlation: float
    btc_eth_corr: float
    btc_sol_corr: float
    eth_sol_corr: float
    
    # 变化指标
    correlation_change_1h: float
    correlation_change_24h: float
    
    # 跳跃检测 (v5.2.1 新增)
    jump_detected: bool = False
    jump_zscore: float = 0.0
    
    # 预测 (v5.2.1 新增)
    prediction: Optional[CorrelationPrediction] = None
    
    # 风险指标
    concentration_risk: float          # 集中度风险 (0-1)
    position_scale_factor: float       # 仓位缩放因子
    
    # 各资产调整
    asset_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # 解释
    reasoning: str = ""
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "action": self.action.value,
            "avg_correlation": self.avg_correlation,
            "btc_eth_corr": self.btc_eth_corr,
            "btc_sol_corr": self.btc_sol_corr,
            "eth_sol_corr": self.eth_sol_corr,
            "position_scale_factor": self.position_scale_factor,
            "concentration_risk": self.concentration_risk,
            "jump_detected": self.jump_detected,
            "jump_zscore": self.jump_zscore,
            "reasoning": self.reasoning,
            "alerts": self.alerts
        }


# =============================================================================
# MULTI-WINDOW CORRELATION CALCULATOR (v5.2.1 新增)
# =============================================================================

class MultiWindowCorrelationCalculator:
    """多窗口相关性计算器"""
    
    def __init__(self, symbols: List[str], 
                 short_window: int = 20,
                 medium_window: int = 100,
                 long_window: int = 200):
        self.symbols = symbols
        self.windows = {
            "short": short_window,
            "medium": medium_window,
            "long": long_window
        }
        
        # 价格历史 (returns)
        max_window = max(self.windows.values())
        self.returns_history: Dict[str, deque] = {
            sym: deque(maxlen=max_window + 10) for sym in symbols
        }
        self.price_history: Dict[str, deque] = {
            sym: deque(maxlen=max_window + 10) for sym in symbols
        }
    
    def update_price(self, symbol: str, price: float):
        """更新价格并计算收益率"""
        if symbol not in self.price_history:
            return
        
        self.price_history[symbol].append(price)
        
        # 计算收益率
        if len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
                self.returns_history[symbol].append(ret)
    
    def compute_correlation_matrix(self, window: str = "medium") -> Optional[CorrelationSnapshot]:
        """计算指定窗口的相关性矩阵"""
        lookback = self.windows.get(window, self.windows["medium"])
        
        # 检查数据充足性
        min_len = min(len(self.returns_history[sym]) for sym in self.symbols)
        if min_len < 30:
            return None
        
        # 构建收益率矩阵
        n_samples = min(min_len, lookback)
        returns_matrix = np.zeros((n_samples, len(self.symbols)))
        
        for i, sym in enumerate(self.symbols):
            returns_list = list(self.returns_history[sym])[-n_samples:]
            returns_matrix[:, i] = returns_list
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(returns_matrix.T)
        
        # 处理 NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 构建字典格式
        matrix_dict = {}
        for i, sym1 in enumerate(self.symbols):
            matrix_dict[sym1] = {}
            for j, sym2 in enumerate(self.symbols):
                matrix_dict[sym1][sym2] = float(corr_matrix[i, j])
        
        # 计算特征值 (PCA)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
        
        # 第一主成分解释方差
        total_var = np.sum(eigenvalues)
        explained_var = eigenvalues[0] / total_var if total_var > 0 else 0
        
        # 条件数
        condition = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else float('inf')
        
        return CorrelationSnapshot(
            timestamp=datetime.utcnow(),
            matrix=matrix_dict,
            eigenvalues=eigenvalues,
            explained_variance=explained_var,
            condition_number=condition
        )
    
    def get_multi_window_correlations(self) -> Dict[str, Optional[CorrelationSnapshot]]:
        """获取所有窗口的相关性矩阵"""
        return {
            window: self.compute_correlation_matrix(window)
            for window in self.windows.keys()
        }


# =============================================================================
# CORRELATION JUMP DETECTOR (v5.2.1 新增)
# =============================================================================

class CorrelationJumpDetector:
    """相关性跳跃检测器"""
    
    def __init__(self, lookback: int = 50, zscore_threshold: float = 3.0):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.correlation_history: deque = deque(maxlen=lookback)
        self.change_history: deque = deque(maxlen=lookback)
    
    def update(self, avg_correlation: float) -> Tuple[bool, float]:
        """
        更新并检测跳跃
        
        Returns:
            (jump_detected, zscore)
        """
        # 计算变化
        if len(self.correlation_history) > 0:
            prev_corr = self.correlation_history[-1]
            change = avg_correlation - prev_corr
            self.change_history.append(change)
        else:
            change = 0.0
        
        self.correlation_history.append(avg_correlation)
        
        # 计算 z-score
        if len(self.change_history) < 10:
            return False, 0.0
        
        changes = np.array(self.change_history)
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        if std_change < 0.001:
            return False, 0.0
        
        zscore = (change - mean_change) / std_change
        jump_detected = abs(zscore) > self.zscore_threshold
        
        return jump_detected, zscore
    
    def get_jump_probability(self) -> float:
        """估计未来跳跃概率"""
        if len(self.change_history) < 20:
            return 0.1  # 默认低概率
        
        changes = np.array(self.change_history)
        
        # 计算尾部分布
        std_change = np.std(changes)
        recent_volatility = np.std(list(changes)[-10:])
        
        if std_change < 0.001:
            return 0.1
        
        # 波动率上升 = 跳跃概率上升
        volatility_ratio = recent_volatility / std_change
        
        # 映射到概率
        if volatility_ratio < 1.0:
            return 0.1
        elif volatility_ratio < 1.5:
            return 0.2 + (volatility_ratio - 1.0) * 0.2
        elif volatility_ratio < 2.0:
            return 0.3 + (volatility_ratio - 1.5) * 0.3
        else:
            return min(0.8, 0.45 + (volatility_ratio - 2.0) * 0.1)


# =============================================================================
# CORRELATION PREDICTOR (v5.2.1 新增)
# =============================================================================

class CorrelationPredictor:
    """相关性预测器 (EWMA + 趋势外推)"""
    
    def __init__(self, ewma_span: int = 20, lookback: int = 100):
        self.ewma_span = ewma_span
        self.lookback = lookback
        self.correlation_history: deque = deque(maxlen=lookback)
        self.timestamps: deque = deque(maxlen=lookback)
    
    def update(self, avg_correlation: float, timestamp: Optional[datetime] = None):
        """更新历史数据"""
        self.correlation_history.append(avg_correlation)
        self.timestamps.append(timestamp or datetime.utcnow())
    
    def predict(self, horizon_bars: int = 6) -> Optional[CorrelationPrediction]:
        """
        预测未来相关性
        
        Args:
            horizon_bars: 预测前瞻期数
        """
        if len(self.correlation_history) < 30:
            return None
        
        corrs = np.array(self.correlation_history)
        
        # EWMA 平滑
        alpha = 2 / (self.ewma_span + 1)
        ewma = [corrs[0]]
        for i in range(1, len(corrs)):
            ewma.append(alpha * corrs[i] + (1 - alpha) * ewma[-1])
        ewma = np.array(ewma)
        
        # 趋势计算 (线性回归)
        x = np.arange(len(ewma))
        slope, intercept = np.polyfit(x[-20:], ewma[-20:], 1)
        
        # 预测
        current = ewma[-1]
        predicted_1h = current + slope * (horizon_bars / 6)  # 假设6 bars = 1h
        predicted_4h = current + slope * horizon_bars
        
        # 限制范围
        predicted_1h = np.clip(predicted_1h, -1.0, 1.0)
        predicted_4h = np.clip(predicted_4h, -1.0, 1.0)
        
        # 趋势方向
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "rising"
        else:
            trend = "falling"
        
        # 置信度 (基于残差)
        residuals = ewma[-20:] - (slope * x[-20:] + intercept)
        residual_std = np.std(residuals)
        confidence = max(0.3, min(0.9, 1.0 - residual_std * 5))
        
        return CorrelationPrediction(
            timestamp=datetime.utcnow(),
            predicted_avg_corr_1h=predicted_1h,
            predicted_avg_corr_4h=predicted_4h,
            prediction_confidence=confidence,
            trend_direction=trend,
            jump_probability=0.0  # 由 JumpDetector 填充
        )


# =============================================================================
# ENHANCED CORRELATION DYNAMIC CONTROLLER (v5.2.1)
# =============================================================================

class CorrelationDynamicControllerV521:
    """
    增强型相关性动态风险控制器 (v5.2.1)
    
    新增功能:
    1. 多窗口相关性分析
    2. 跳跃检测
    3. EWMA 趋势预测
    4. 预测性仓位调整
    5. EventBus 集成
    """
    
    def __init__(self, 
                 config: Optional[CorrelationDynamicsConfigV521] = None,
                 symbols: List[str] = None):
        self.config = config or CorrelationDynamicsConfigV521()
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        
        # 多窗口计算器
        self.calculator = MultiWindowCorrelationCalculator(
            self.symbols,
            short_window=self.config.short_lookback_bars,
            medium_window=self.config.lookback_bars,
            long_window=self.config.long_lookback_bars
        )
        
        # 跳跃检测器
        self.jump_detector = CorrelationJumpDetector(
            lookback=50,
            zscore_threshold=self.config.jump_zscore_threshold
        )
        
        # 预测器
        self.predictor = CorrelationPredictor(
            ewma_span=self.config.ewma_span,
            lookback=200
        )
        
        # 状态
        self.current_regime = CorrelationRegime.NORMAL
        self.current_snapshot: Optional[CorrelationSnapshot] = None
        self.current_assessment: Optional[CorrelationRiskAssessmentV521] = None
        
        # 历史
        self.snapshot_history: deque = deque(maxlen=100)
        self.assessment_history: deque = deque(maxlen=50)
        
        # EventBus (延迟导入)
        self._event_bus = None
        
        logger.info(f"CorrelationDynamicControllerV521 initialized with symbols: {self.symbols}")
    
    def _get_event_bus(self):
        """延迟获取 EventBus"""
        if self._event_bus is None:
            try:
                from infra.event_bus import get_event_bus
                self._event_bus = get_event_bus()
            except ImportError:
                logger.warning("EventBus not available")
        return self._event_bus
    
    def _emit_event(self, event_type: str, payload: Dict):
        """发送事件"""
        if not self.config.emit_events:
            return
        
        bus = self._get_event_bus()
        if bus:
            try:
                from infra.event_bus import Event, EventType, EventPriority
                event = Event(
                    event_type=EventType.RISK_ALERT,
                    source="correlation_controller",
                    priority=EventPriority.HIGH,
                    payload={
                        "sub_type": event_type,
                        **payload
                    }
                )
                bus.publish(event)
            except Exception as e:
                logger.error(f"Failed to emit event: {e}")
    
    def update_price(self, symbol: str, price: float):
        """更新价格"""
        self.calculator.update_price(symbol, price)
    
    def compute_assessment(self) -> Optional[CorrelationRiskAssessmentV521]:
        """计算完整风险评估"""
        # 获取多窗口相关性
        multi_window = self.calculator.get_multi_window_correlations()
        
        # 使用中期窗口作为主要参考
        snapshot = multi_window.get("medium")
        if snapshot is None:
            return None
        
        self.current_snapshot = snapshot
        self.snapshot_history.append(snapshot)
        
        # 提取相关性
        btc_eth = snapshot.get_correlation("BTC", "ETH")
        btc_sol = snapshot.get_correlation("BTC", "SOL")
        eth_sol = snapshot.get_correlation("ETH", "SOL")
        avg_corr = snapshot.average_correlation()
        
        # 跳跃检测
        jump_detected, jump_zscore = self.jump_detector.update(avg_corr)
        jump_prob = self.jump_detector.get_jump_probability()
        
        # 更新预测器
        self.predictor.update(avg_corr)
        prediction = self.predictor.predict(self.config.prediction_horizon_bars)
        if prediction:
            prediction.jump_probability = jump_prob
        
        # 计算变化
        change_1h, change_24h = self._compute_changes()
        
        # 判断 regime
        regime = self._determine_regime(
            avg_corr, btc_eth, btc_sol, eth_sol,
            change_1h, jump_detected
        )
        
        # 确定动作
        action, reasoning = self._determine_action(
            regime, avg_corr, change_1h, jump_detected, jump_zscore, prediction
        )
        
        # 计算仓位缩放
        position_scale = self._compute_position_scale(
            regime, avg_corr, prediction
        )
        
        # 计算集中度风险
        concentration_risk = snapshot.explained_variance
        
        # 各资产调整
        asset_adjustments = self._compute_asset_adjustments(
            regime, btc_eth, btc_sol, eth_sol
        )
        
        # 生成告警
        alerts = self._generate_alerts(
            regime, avg_corr, change_1h, concentration_risk,
            jump_detected, jump_zscore
        )
        
        # 创建评估结果
        assessment = CorrelationRiskAssessmentV521(
            timestamp=datetime.utcnow(),
            regime=regime,
            action=action,
            avg_correlation=avg_corr,
            btc_eth_corr=btc_eth,
            btc_sol_corr=btc_sol,
            eth_sol_corr=eth_sol,
            correlation_change_1h=change_1h,
            correlation_change_24h=change_24h,
            jump_detected=jump_detected,
            jump_zscore=jump_zscore,
            prediction=prediction,
            concentration_risk=concentration_risk,
            position_scale_factor=position_scale,
            asset_adjustments=asset_adjustments,
            reasoning=reasoning,
            alerts=alerts
        )
        
        # 更新状态
        prev_regime = self.current_regime
        self.current_regime = regime
        self.current_assessment = assessment
        self.assessment_history.append(assessment)
        
        # 发送事件
        if regime != prev_regime and self.config.alert_on_regime_change:
            self._emit_event("REGIME_CHANGE", {
                "prev_regime": prev_regime.value,
                "new_regime": regime.value,
                "avg_correlation": avg_corr
            })
        
        if jump_detected and self.config.alert_on_jump:
            self._emit_event("CORRELATION_JUMP", {
                "zscore": jump_zscore,
                "avg_correlation": avg_corr,
                "direction": "up" if jump_zscore > 0 else "down"
            })
        
        return assessment
    
    def _compute_changes(self) -> Tuple[float, float]:
        """计算相关性变化"""
        if len(self.snapshot_history) < 2:
            return 0.0, 0.0
        
        current_avg = self.snapshot_history[-1].average_correlation()
        
        # 1小时变化 (约6个4H bars = 1天，假设1bar=4h，1h约0.25 bar)
        # 简化: 取前一个快照
        if len(self.snapshot_history) >= 2:
            prev_avg = self.snapshot_history[-2].average_correlation()
            change_1h = current_avg - prev_avg
        else:
            change_1h = 0.0
        
        # 24小时变化
        if len(self.snapshot_history) >= 6:
            old_avg = self.snapshot_history[-6].average_correlation()
            change_24h = current_avg - old_avg
        else:
            change_24h = 0.0
        
        return change_1h, change_24h
    
    def _determine_regime(self, avg_corr: float, btc_eth: float, 
                          btc_sol: float, eth_sol: float,
                          change_1h: float, jump_detected: bool) -> CorrelationRegime:
        """判断相关性 regime"""
        # 跳跃中
        if jump_detected:
            return CorrelationRegime.JUMPING
        
        # 危机: 所有相关性 > 0.95
        if btc_eth > self.config.crisis_threshold and \
           btc_sol > self.config.crisis_threshold and \
           eth_sol > self.config.crisis_threshold:
            return CorrelationRegime.CRISIS
        
        # 崩塌: 相关性快速下降
        if change_1h < -self.config.collapse_threshold:
            return CorrelationRegime.BREAKDOWN
        
        # 解耦: 相关性很低
        if avg_corr < self.config.decoupling_threshold:
            return CorrelationRegime.DECOUPLED
        
        # 升高: 相关性上升
        if avg_corr > self.config.max_portfolio_correlation or \
           change_1h > self.config.spike_threshold:
            return CorrelationRegime.ELEVATED
        
        return CorrelationRegime.NORMAL
    
    def _determine_action(self, regime: CorrelationRegime, avg_corr: float,
                          change_1h: float, jump_detected: bool, 
                          jump_zscore: float,
                          prediction: Optional[CorrelationPrediction]) -> Tuple[RiskAction, str]:
        """确定风险响应动作"""
        # 跳跃
        if regime == CorrelationRegime.JUMPING:
            if jump_zscore > 0:
                return (
                    RiskAction.HALT_NEW_POSITIONS,
                    f"JUMPING UP: 相关性跳跃上升 (z={jump_zscore:.1f})，暂停新仓位"
                )
            else:
                return (
                    RiskAction.REDUCE_SIZE,
                    f"JUMPING DOWN: 相关性跳跃下降 (z={jump_zscore:.1f})，减少仓位"
                )
        
        # 危机
        if regime == CorrelationRegime.CRISIS:
            return (
                RiskAction.EMERGENCY_FLATTEN,
                f"CRISIS: 相关性危机 (avg={avg_corr:.2f})，紧急减仓"
            )
        
        # 崩塌
        if regime == CorrelationRegime.BREAKDOWN:
            return (
                RiskAction.REDUCE_SIZE,
                f"BREAKDOWN: 相关性结构崩塌，减少仓位"
            )
        
        # 升高
        if regime == CorrelationRegime.ELEVATED:
            if change_1h > 0.1:
                return (
                    RiskAction.HALT_NEW_POSITIONS,
                    f"ELEVATED + RAPID: 相关性快速上升 (+{change_1h:.2f}/1h)，暂停新仓位"
                )
            return (
                RiskAction.REDUCE_SIZE,
                f"ELEVATED: 相关性升高 (avg={avg_corr:.2f})，减少仓位"
            )
        
        # 预测性调整 (v5.2.1 新增)
        if prediction and prediction.prediction_confidence > 0.6:
            if prediction.trend_direction == "rising" and \
               prediction.predicted_avg_corr_4h > self.config.max_portfolio_correlation:
                return (
                    RiskAction.PREEMPTIVE_REDUCE,
                    f"PREEMPTIVE: 预测相关性将升高到 {prediction.predicted_avg_corr_4h:.2f}，预防性减仓"
                )
        
        # 解耦
        if regime == CorrelationRegime.DECOUPLED:
            return (
                RiskAction.NONE,
                f"DECOUPLED: 资产解耦，分散化有效"
            )
        
        return (RiskAction.NONE, "NORMAL: 相关性正常")
    
    def _compute_position_scale(self, regime: CorrelationRegime, 
                                avg_corr: float,
                                prediction: Optional[CorrelationPrediction]) -> float:
        """计算仓位缩放因子"""
        if regime == CorrelationRegime.CRISIS:
            return self.config.crisis_position_scale
        
        if regime == CorrelationRegime.JUMPING:
            return 0.50
        
        if regime == CorrelationRegime.ELEVATED:
            return self.config.elevated_position_scale
        
        if regime == CorrelationRegime.BREAKDOWN:
            return 0.50
        
        # 预测性调整
        if prediction and prediction.trend_direction == "rising":
            if prediction.predicted_avg_corr_4h > self.config.max_portfolio_correlation:
                return self.config.preemptive_scale_factor
        
        # 正常: 根据相关性线性调整
        if avg_corr > self.config.max_portfolio_correlation:
            excess = avg_corr - self.config.max_portfolio_correlation
            scale = max(0.5, 1.0 - excess * 2)
            return scale
        
        return 1.0
    
    def _compute_asset_adjustments(self, regime: CorrelationRegime,
                                   btc_eth: float, btc_sol: float,
                                   eth_sol: float) -> Dict[str, float]:
        """计算各资产仓位调整因子"""
        adjustments = {"BTC": 1.0, "ETH": 1.0, "SOL": 1.0}
        
        if regime == CorrelationRegime.CRISIS:
            # 危机: 保留 BTC，大幅减少其他
            adjustments = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        
        elif regime == CorrelationRegime.ELEVATED:
            # 升高: 减少高相关资产
            if btc_eth > 0.9:
                adjustments["ETH"] = 0.7
            if btc_sol > 0.85:
                adjustments["SOL"] = 0.6
        
        elif regime == CorrelationRegime.BREAKDOWN:
            # 崩塌: 增加分散化资产
            if btc_sol < 0.4:
                adjustments["SOL"] = 1.2  # SOL 解耦，可增加
        
        return adjustments
    
    def _generate_alerts(self, regime: CorrelationRegime, avg_corr: float,
                         change_1h: float, concentration: float,
                         jump_detected: bool, jump_zscore: float) -> List[str]:
        """生成告警消息"""
        alerts = []
        
        if regime == CorrelationRegime.CRISIS:
            alerts.append(f"🚨 CRITICAL: 相关性危机! 平均相关性={avg_corr:.2f}")
        
        if regime == CorrelationRegime.BREAKDOWN:
            alerts.append(f"[WARN]️ WARNING: 相关性结构崩塌!")
        
        if jump_detected:
            direction = "上升" if jump_zscore > 0 else "下降"
            alerts.append(f"⚡ JUMP: 相关性跳跃{direction} (z-score={jump_zscore:.1f})")
        
        if abs(change_1h) > 0.15:
            direction = "上升" if change_1h > 0 else "下降"
            alerts.append(f"📊 相关性快速{direction}: {change_1h:+.2f}/1h")
        
        if concentration > self.config.eigenvalue_concentration_limit:
            alerts.append(f"🎯 集中度风险: 第一主成分={concentration:.1%}")
        
        return alerts
    
    def get_position_adjustment(self, symbol: str, 
                                base_size: float) -> Tuple[float, str]:
        """获取仓位调整"""
        if self.current_assessment is None:
            return base_size, "无相关性数据"
        
        assessment = self.current_assessment
        
        # 组合级缩放
        scale = assessment.position_scale_factor
        
        # 资产级调整
        asset_adj = assessment.asset_adjustments.get(symbol, 1.0)
        
        final_scale = scale * asset_adj
        adjusted_size = base_size * final_scale
        
        reason = f"相关性调整: regime={assessment.regime.value}, " \
                 f"scale={scale:.2f}, asset_adj={asset_adj:.2f}"
        
        return adjusted_size, reason
    
    def can_add_position(self, symbol: str, 
                         existing_positions: Dict[str, float]) -> Tuple[bool, str]:
        """检查是否可以新增仓位"""
        if self.current_assessment is None:
            return True, "无相关性数据，允许交易"
        
        assessment = self.current_assessment
        
        # 危机/跳跃时禁止新仓位
        if assessment.regime in [CorrelationRegime.CRISIS, CorrelationRegime.JUMPING]:
            return False, f"{assessment.regime.value} 状态，禁止新仓位"
        
        # 暂停新仓位动作
        if assessment.action == RiskAction.HALT_NEW_POSITIONS:
            return False, "风险控制暂停新仓位"
        
        # 检查相关敞口
        total_correlated = 0.0
        for pos_symbol, size in existing_positions.items():
            if pos_symbol != symbol and self.current_snapshot:
                corr = self.current_snapshot.get_correlation(symbol, pos_symbol)
                if corr > 0.7:
                    total_correlated += abs(size)
        
        if total_correlated > self.config.max_correlated_exposure:
            return False, f"高相关敞口已达 {total_correlated:.1%}，超过限制"
        
        return True, "相关性检查通过"
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        if self.current_assessment is None:
            return {"status": "initializing", "data_available": False}
        
        assessment = self.current_assessment
        
        data = {
            "status": "active",
            "data_available": True,
            "regime": assessment.regime.value,
            "action": assessment.action.value,
            "avg_correlation": round(assessment.avg_correlation, 3),
            "correlations": {
                "BTC-ETH": round(assessment.btc_eth_corr, 3),
                "BTC-SOL": round(assessment.btc_sol_corr, 3),
                "ETH-SOL": round(assessment.eth_sol_corr, 3),
            },
            "position_scale": round(assessment.position_scale_factor, 2),
            "concentration_risk": round(assessment.concentration_risk, 3),
            "change_1h": round(assessment.correlation_change_1h, 3),
            "change_24h": round(assessment.correlation_change_24h, 3),
            "jump_detected": assessment.jump_detected,
            "jump_zscore": round(assessment.jump_zscore, 2),
            "alerts": assessment.alerts,
            "reasoning": assessment.reasoning
        }
        
        # 添加预测数据
        if assessment.prediction:
            data["prediction"] = {
                "avg_corr_1h": round(assessment.prediction.predicted_avg_corr_1h, 3),
                "avg_corr_4h": round(assessment.prediction.predicted_avg_corr_4h, 3),
                "confidence": round(assessment.prediction.prediction_confidence, 2),
                "trend": assessment.prediction.trend_direction,
                "jump_probability": round(assessment.prediction.jump_probability, 2)
            }
        
        return data
    
    def get_status(self) -> Dict:
        """获取控制器状态"""
        return {
            "symbols": self.symbols,
            "current_regime": self.current_regime.value,
            "snapshot_count": len(self.snapshot_history),
            "assessment_count": len(self.assessment_history),
            "has_assessment": self.current_assessment is not None,
            "position_scale": self.current_assessment.position_scale_factor if self.current_assessment else 1.0
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_correlation_controller_v521: Optional[CorrelationDynamicControllerV521] = None


def get_correlation_controller_v521(
    config: CorrelationDynamicsConfigV521 = None,
    symbols: List[str] = None
) -> CorrelationDynamicControllerV521:
    """获取或创建相关性控制器单例 (v5.2.1)"""
    global _correlation_controller_v521
    if _correlation_controller_v521 is None:
        _correlation_controller_v521 = CorrelationDynamicControllerV521(config, symbols)
    return _correlation_controller_v521


def reset_correlation_controller_v521():
    """重置单例"""
    global _correlation_controller_v521
    _correlation_controller_v521 = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CorrelationRegime',
    'RiskAction',
    'CorrelationSnapshot',
    'CorrelationPrediction',
    'CorrelationDynamicsConfigV521',
    'CorrelationRiskAssessmentV521',
    'MultiWindowCorrelationCalculator',
    'CorrelationJumpDetector',
    'CorrelationPredictor',
    'CorrelationDynamicControllerV521',
    'get_correlation_controller_v521',
    'reset_correlation_controller_v521'
]


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    controller = CorrelationDynamicControllerV521()
    
    # 模拟价格数据
    base_prices = {"BTC": 100000, "ETH": 3000, "SOL": 150}
    
    for i in range(200):
        # 生成相关的价格变动
        common_factor = random.gauss(0, 0.01)
        
        # 在某些点制造跳跃
        if i == 100:
            common_factor = 0.05  # 大幅共振
        
        for sym, base in base_prices.items():
            idio_factor = random.gauss(0, 0.005)
            change = common_factor + idio_factor
            new_price = base * (1 + change)
            controller.update_price(sym, new_price)
            base_prices[sym] = new_price
        
        # 每10个点评估一次
        if i > 50 and i % 10 == 0:
            assessment = controller.compute_assessment()
            if assessment:
                print(f"\n[{i}] Regime: {assessment.regime.value}")
                print(f"    Avg Corr: {assessment.avg_correlation:.3f}")
                print(f"    Scale: {assessment.position_scale_factor:.2f}")
                print(f"    Jump: {assessment.jump_detected} (z={assessment.jump_zscore:.1f})")
                if assessment.prediction:
                    print(f"    Predicted 4h: {assessment.prediction.predicted_avg_corr_4h:.3f}")
                if assessment.alerts:
                    for alert in assessment.alerts:
                        print(f"    {alert}")
    
    # 最终状态
    print(f"\nFinal Status: {controller.get_status()}")
    print(f"Dashboard Data: {controller.get_dashboard_data()}")

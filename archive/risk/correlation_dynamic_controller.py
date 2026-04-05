"""
================================================================================
HMATS v5.2.1 - 相关性动态风险控制器
Correlation Dynamic Risk Controller
================================================================================
Purpose: 实时相关性驱动的仓位调整逻辑

SOTA 对照:
- 当多资产相关性异常上升或崩塌时，自动降低组合仓位或调整权重
- 实时监控 BTC-ETH-SOL 相关性矩阵
- 检测相关性regime变化并触发风险响应

Features:
1. 滚动相关性矩阵计算
2. 相关性异常检测 (spike/collapse)
3. 动态仓位缩放因子
4. 组合级风险聚合
5. EventBus 集成

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


class RiskAction(Enum):
    """风险响应动作"""
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    EXIT_CORRELATED = "exit_correlated"
    HEDGE_REQUIRED = "hedge_required"
    HALT_NEW_POSITIONS = "halt_new_positions"
    EMERGENCY_FLATTEN = "emergency_flatten"


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
class CorrelationDynamicsConfig:
    """相关性动态控制配置"""
    # 计算参数
    lookback_bars: int = 100           # 相关性计算窗口
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
    
    # 仓位调整
    max_correlated_exposure: float = 0.60  # 高相关资产最大敞口
    crisis_position_scale: float = 0.30    # 危机时仓位缩放
    elevated_position_scale: float = 0.70  # 升高时仓位缩放
    
    # 风险限制
    max_portfolio_correlation: float = 0.85  # 组合最大平均相关性
    eigenvalue_concentration_limit: float = 0.80  # 第一主成分限制


@dataclass
class CorrelationRiskAssessment:
    """相关性风险评估结果"""
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
    
    # 风险指标
    concentration_risk: float          # 集中度风险 (0-1)
    position_scale_factor: float       # 仓位缩放因子
    
    # 各资产调整
    asset_adjustments: Dict[str, float]  # symbol -> adjustment factor
    
    # 解释
    reasoning: str
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
            "reasoning": self.reasoning,
            "alerts": self.alerts
        }


# =============================================================================
# CORRELATION MATRIX CALCULATOR
# =============================================================================

class CorrelationMatrixCalculator:
    """实时相关性矩阵计算器"""
    
    def __init__(self, symbols: List[str], lookback: int = 100):
        self.symbols = symbols
        self.lookback = lookback
        
        # 价格历史 (returns)
        self.returns_history: Dict[str, deque] = {
            sym: deque(maxlen=lookback) for sym in symbols
        }
        self.price_history: Dict[str, deque] = {
            sym: deque(maxlen=lookback + 1) for sym in symbols
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
    
    def compute_correlation_matrix(self) -> Optional[CorrelationSnapshot]:
        """计算相关性矩阵"""
        # 检查数据充足性
        min_len = min(len(self.returns_history[sym]) for sym in self.symbols)
        if min_len < 30:
            return None
        
        # 构建收益率矩阵
        n_samples = min(min_len, self.lookback)
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


# =============================================================================
# CORRELATION DYNAMIC CONTROLLER
# =============================================================================

class CorrelationDynamicController:
    """
    相关性动态风险控制器
    
    核心功能:
    1. 实时监控多资产相关性矩阵
    2. 检测相关性异常 (spike/collapse/crisis)
    3. 动态调整仓位缩放因子
    4. 通过 EventBus 发布风险事件
    """
    
    def __init__(self, config: CorrelationDynamicsConfig = None,
                 symbols: List[str] = None):
        self.config = config or CorrelationDynamicsConfig()
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        
        # 相关性计算器
        self.calculator = CorrelationMatrixCalculator(
            self.symbols, 
            self.config.lookback_bars
        )
        
        # 状态
        self.current_snapshot: Optional[CorrelationSnapshot] = None
        self.current_regime: CorrelationRegime = CorrelationRegime.NORMAL
        self.snapshot_history: deque = deque(maxlen=500)
        
        # 风险评估
        self.current_assessment: Optional[CorrelationRiskAssessment] = None
        self.assessment_history: deque = deque(maxlen=100)
        
        # 回调
        self.event_callbacks: List[Callable] = []
        
        # 正常相关性基准
        self.normal_correlations = {
            ("BTC", "ETH"): self.config.normal_btc_eth,
            ("BTC", "SOL"): self.config.normal_btc_sol,
            ("ETH", "SOL"): self.config.normal_eth_sol,
        }
        
        logger.info(f"CorrelationDynamicController initialized for {self.symbols}")
    
    def register_event_callback(self, callback: Callable):
        """注册事件回调"""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """发布事件"""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "CorrelationDynamicController",
            "data": data
        }
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def update_price(self, symbol: str, price: float):
        """更新价格数据"""
        self.calculator.update_price(symbol, price)
    
    def update_prices(self, prices: Dict[str, float]):
        """批量更新价格"""
        for symbol, price in prices.items():
            self.update_price(symbol, price)
    
    def compute_assessment(self) -> Optional[CorrelationRiskAssessment]:
        """计算相关性风险评估"""
        # 计算相关性矩阵
        snapshot = self.calculator.compute_correlation_matrix()
        if snapshot is None:
            return None
        
        self.current_snapshot = snapshot
        self.snapshot_history.append(snapshot)
        
        # 提取关键相关性
        btc_eth = snapshot.get_correlation("BTC", "ETH")
        btc_sol = snapshot.get_correlation("BTC", "SOL")
        eth_sol = snapshot.get_correlation("ETH", "SOL")
        avg_corr = snapshot.average_correlation()
        
        # 计算相关性变化
        change_1h = self._compute_correlation_change(hours=1)
        change_24h = self._compute_correlation_change(hours=24)
        
        # 检测regime
        regime = self._detect_regime(btc_eth, btc_sol, eth_sol, avg_corr)
        
        # 确定风险动作
        action, reasoning = self._determine_action(regime, avg_corr, change_1h)
        
        # 计算仓位缩放因子
        scale_factor = self._compute_position_scale(regime, avg_corr)
        
        # 计算各资产调整
        asset_adjustments = self._compute_asset_adjustments(
            regime, btc_eth, btc_sol, eth_sol
        )
        
        # 集中度风险
        concentration = snapshot.explained_variance
        
        # 生成告警
        alerts = self._generate_alerts(regime, avg_corr, change_1h, concentration)
        
        assessment = CorrelationRiskAssessment(
            timestamp=datetime.utcnow(),
            regime=regime,
            action=action,
            avg_correlation=avg_corr,
            btc_eth_corr=btc_eth,
            btc_sol_corr=btc_sol,
            eth_sol_corr=eth_sol,
            correlation_change_1h=change_1h,
            correlation_change_24h=change_24h,
            concentration_risk=concentration,
            position_scale_factor=scale_factor,
            asset_adjustments=asset_adjustments,
            reasoning=reasoning,
            alerts=alerts
        )
        
        # 检测regime变化
        if self.current_assessment and regime != self.current_regime:
            self._emit_event("CORRELATION_REGIME_CHANGE", {
                "old_regime": self.current_regime.value,
                "new_regime": regime.value,
                "avg_correlation": avg_corr,
                "action": action.value
            })
        
        self.current_regime = regime
        self.current_assessment = assessment
        self.assessment_history.append(assessment)
        
        return assessment
    
    def _compute_correlation_change(self, hours: int) -> float:
        """计算相关性变化"""
        if len(self.snapshot_history) < 2:
            return 0.0
        
        # 估算bars数 (假设4H timeframe)
        bars_per_hour = 0.25
        target_bars = int(hours * bars_per_hour)
        
        # 查找历史快照
        current = self.snapshot_history[-1]
        
        idx = max(0, len(self.snapshot_history) - 1 - target_bars)
        historical = self.snapshot_history[idx]
        
        current_avg = current.average_correlation()
        historical_avg = historical.average_correlation()
        
        return current_avg - historical_avg
    
    def _detect_regime(self, btc_eth: float, btc_sol: float, 
                       eth_sol: float, avg_corr: float) -> CorrelationRegime:
        """检测相关性regime"""
        
        # 危机检测: 所有相关性都非常高
        if (btc_eth > self.config.crisis_threshold and 
            btc_sol > self.config.crisis_threshold and
            eth_sol > self.config.crisis_threshold):
            return CorrelationRegime.CRISIS
        
        # 解耦检测: 任一主要相关性大幅下降
        normal_btc_eth = self.config.normal_btc_eth
        normal_btc_sol = self.config.normal_btc_sol
        normal_eth_sol = self.config.normal_eth_sol
        
        if (btc_eth < normal_btc_eth - self.config.collapse_threshold or
            btc_sol < normal_btc_sol - self.config.collapse_threshold or
            eth_sol < normal_eth_sol - self.config.collapse_threshold):
            return CorrelationRegime.BREAKDOWN
        
        if avg_corr < self.config.decoupling_threshold:
            return CorrelationRegime.DECOUPLED
        
        # 升高检测
        if (btc_eth > normal_btc_eth + self.config.spike_threshold or
            btc_sol > normal_btc_sol + self.config.spike_threshold or
            eth_sol > normal_eth_sol + self.config.spike_threshold):
            return CorrelationRegime.ELEVATED
        
        return CorrelationRegime.NORMAL
    
    def _determine_action(self, regime: CorrelationRegime, 
                          avg_corr: float, change_1h: float) -> Tuple[RiskAction, str]:
        """确定风险响应动作"""
        
        if regime == CorrelationRegime.CRISIS:
            return (
                RiskAction.EMERGENCY_FLATTEN,
                f"CRISIS: 相关性危机 (avg={avg_corr:.2f}), 所有资产同向移动，需紧急减仓"
            )
        
        if regime == CorrelationRegime.BREAKDOWN:
            return (
                RiskAction.HEDGE_REQUIRED,
                f"BREAKDOWN: 相关性崩塌，市场结构变化，需要对冲"
            )
        
        if regime == CorrelationRegime.ELEVATED:
            if change_1h > 0.10:  # 快速上升
                return (
                    RiskAction.HALT_NEW_POSITIONS,
                    f"ELEVATED + RAPID: 相关性快速上升 (+{change_1h:.2f}/1h)，暂停新仓位"
                )
            return (
                RiskAction.REDUCE_SIZE,
                f"ELEVATED: 相关性升高 (avg={avg_corr:.2f})，减少仓位"
            )
        
        if regime == CorrelationRegime.DECOUPLED:
            return (
                RiskAction.NONE,
                f"DECOUPLED: 资产解耦，分散化有效"
            )
        
        return (RiskAction.NONE, "NORMAL: 相关性正常")
    
    def _compute_position_scale(self, regime: CorrelationRegime, 
                                avg_corr: float) -> float:
        """计算仓位缩放因子"""
        if regime == CorrelationRegime.CRISIS:
            return self.config.crisis_position_scale
        
        if regime == CorrelationRegime.ELEVATED:
            return self.config.elevated_position_scale
        
        if regime == CorrelationRegime.BREAKDOWN:
            return 0.50  # 结构变化时保守
        
        # 正常regime: 根据相关性线性调整
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
            # 危机时: 保留BTC,大幅减少其他
            adjustments = {"BTC": 0.5, "ETH": 0.3, "SOL": 0.2}
        
        elif regime == CorrelationRegime.ELEVATED:
            # 升高时: 减少高相关资产
            if btc_eth > 0.9:
                adjustments["ETH"] = 0.7
            if btc_sol > 0.85:
                adjustments["SOL"] = 0.6
        
        elif regime == CorrelationRegime.BREAKDOWN:
            # 崩塌时: 增加分散化资产
            if btc_sol < 0.4:
                adjustments["SOL"] = 1.2  # SOL 解耦，可增加
        
        return adjustments
    
    def _generate_alerts(self, regime: CorrelationRegime, avg_corr: float,
                         change_1h: float, concentration: float) -> List[str]:
        """生成告警消息"""
        alerts = []
        
        if regime == CorrelationRegime.CRISIS:
            alerts.append(f"🚨 CRITICAL: 相关性危机! 平均相关性={avg_corr:.2f}")
        
        if regime == CorrelationRegime.BREAKDOWN:
            alerts.append(f"[WARN]️ WARNING: 相关性结构崩塌!")
        
        if abs(change_1h) > 0.15:
            direction = "上升" if change_1h > 0 else "下降"
            alerts.append(f"📊 相关性快速{direction}: {change_1h:+.2f}/1h")
        
        if concentration > self.config.eigenvalue_concentration_limit:
            alerts.append(f"🎯 集中度风险: 第一主成分={concentration:.1%}")
        
        return alerts
    
    def get_position_adjustment(self, symbol: str, 
                                base_size: float) -> Tuple[float, str]:
        """
        获取仓位调整
        
        Returns:
            (adjusted_size, reason)
        """
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
    
    def can_add_position(self, symbol: str, existing_positions: Dict[str, float]) -> Tuple[bool, str]:
        """
        检查是否可以新增仓位
        
        Args:
            symbol: 要新增的资产
            existing_positions: 现有仓位 {symbol: size}
        """
        if self.current_assessment is None:
            return True, "无相关性数据，允许交易"
        
        assessment = self.current_assessment
        
        # 危机时禁止新仓位
        if assessment.regime == CorrelationRegime.CRISIS:
            return False, "相关性危机，禁止新仓位"
        
        # 检查相关敞口
        total_correlated = 0.0
        for pos_symbol, size in existing_positions.items():
            if pos_symbol != symbol and self.current_snapshot:
                corr = self.current_snapshot.get_correlation(symbol, pos_symbol)
                if corr > 0.7:  # 高相关
                    total_correlated += abs(size)
        
        # 检查是否超过限制
        if total_correlated > self.config.max_correlated_exposure:
            return False, f"高相关敞口已达 {total_correlated:.1%}，超过限制"
        
        return True, "相关性检查通过"
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        if self.current_assessment is None:
            return {"status": "initializing", "data_available": False}
        
        assessment = self.current_assessment
        
        return {
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
            "alerts": assessment.alerts,
            "reasoning": assessment.reasoning
        }
    
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

_correlation_controller: Optional[CorrelationDynamicController] = None


def get_correlation_controller(config: CorrelationDynamicsConfig = None,
                                symbols: List[str] = None) -> CorrelationDynamicController:
    """获取或创建相关性控制器单例"""
    global _correlation_controller
    if _correlation_controller is None:
        _correlation_controller = CorrelationDynamicController(config, symbols)
    return _correlation_controller


def reset_correlation_controller():
    """重置单例"""
    global _correlation_controller
    _correlation_controller = None


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    controller = CorrelationDynamicController()
    
    # 模拟价格数据
    base_prices = {"BTC": 100000, "ETH": 3000, "SOL": 150}
    
    for i in range(200):
        # 生成相关的价格变动
        common_factor = random.gauss(0, 0.01)
        
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
                if assessment.alerts:
                    for alert in assessment.alerts:
                        print(f"    {alert}")
    
    # 最终状态
    print(f"\nFinal Status: {controller.get_status()}")
    print(f"Dashboard Data: {controller.get_dashboard_data()}")

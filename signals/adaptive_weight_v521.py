"""
================================================================================
HMATS v5.4.0 - 策略权重自适应调整 (CANONICAL)
Adaptive Strategy Weight Adjustment
================================================================================
ON CANONICAL MODULE (v5.4.0)

This is THE canonical weight management system for production.

Usage:
    from signals.adaptive_weight_v521 import get_adaptive_weight_manager_v521
    
    weight_manager = get_adaptive_weight_manager_v521()
    weight = weight_manager.get_weight("momentum")

Deprecated alternatives (FROZEN, blocked in PROD):
    - signals/adaptive_weights.py -> get_adaptive_weights()
    - signals/adaptive_strategy_weight_manager.py -> get_weight_manager()
================================================================================
Purpose: 引入 Sharpe、Calmar 和短期胜率作为动态权重依据，实现实时自调

SOTA 对照:
1. 增强 signals/authority_fusion.py 和 analytics/strategy_aging.py
2. 引入 Sharpe、Calmar 和短期胜率作为动态权重依据
3. 实现实时自调

Features:
1. 滚动 Sharpe Ratio 计算
2. 滚动 Calmar Ratio 计算 (收益/最大回撤)
3. 短期胜率追踪 (7天/30天)
4. 综合权重计算
5. 动态权重调整 (基于近期表现)
6. EventBus 集成

Author: HMATS v5.2.1 SOTA Upgrade
Version: 5.4.0-CANONICAL
================================================================================
"""

# =============================================================================
# CANONICAL MODULE FLAG (v5.4.0)
# =============================================================================
IS_CANONICAL = True
CANONICAL_VERSION = "5.4.0"

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AdaptiveWeightConfigV521:
    """自适应权重配置"""
    # Sharpe 计算
    sharpe_window_days: int = 30        # Sharpe 计算窗口
    sharpe_annualization: float = 365   # 年化因子
    risk_free_rate: float = 0.04        # 无风险利率
    
    # Calmar 计算
    calmar_window_days: int = 90        # Calmar 计算窗口
    min_drawdown_for_calmar: float = 0.01  # 最小回撤阈值
    
    # 胜率计算
    short_term_window_days: int = 7     # 短期胜率窗口
    medium_term_window_days: int = 30   # 中期胜率窗口
    min_trades_for_winrate: int = 5     # 最小交易数
    
    # 权重计算
    sharpe_weight: float = 0.40         # Sharpe 在综合权重中的占比
    calmar_weight: float = 0.25         # Calmar 占比
    winrate_short_weight: float = 0.20  # 短期胜率占比
    winrate_medium_weight: float = 0.15 # 中期胜率占比
    
    # 权重边界
    min_strategy_weight: float = 0.2    # 最小权重 (不会完全禁用)
    max_strategy_weight: float = 2.0    # 最大权重 (不会过度放大)
    
    # 调整速率
    weight_smoothing_alpha: float = 0.3  # EMA 平滑系数
    
    # 评估频率
    evaluation_interval_hours: int = 4   # 每4小时评估一次


@dataclass
class StrategyPerformanceRecord:
    """策略表现记录"""
    timestamp: datetime
    strategy_name: str
    
    # 交易结果
    pnl_usd: float
    pnl_pct: float
    is_winner: bool
    
    # 市场状态
    regime: str
    volatility: float


@dataclass
class StrategyMetrics:
    """策略指标"""
    strategy_name: str
    timestamp: datetime
    
    # 核心指标
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    winrate_short: float = 0.5      # 短期胜率
    winrate_medium: float = 0.5     # 中期胜率
    
    # 辅助指标
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 1.0
    
    # 数据质量
    trade_count_short: int = 0
    trade_count_medium: int = 0
    sufficient_data: bool = False
    
    # 计算的权重
    computed_weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy_name,
            "sharpe": round(self.sharpe_ratio, 2),
            "calmar": round(self.calmar_ratio, 2),
            "winrate_short": round(self.winrate_short, 2),
            "winrate_medium": round(self.winrate_medium, 2),
            "computed_weight": round(self.computed_weight, 3),
            "sufficient_data": self.sufficient_data
        }


@dataclass
class WeightAdjustmentResult:
    """权重调整结果"""
    timestamp: datetime
    
    # 各策略权重
    strategy_weights: Dict[str, float]
    
    # 指标详情
    strategy_metrics: Dict[str, StrategyMetrics]
    
    # 调整说明
    adjustments_made: List[str]
    
    # 系统级指标
    portfolio_sharpe: float = 0.0
    portfolio_calmar: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_weights": {k: round(v, 3) for k, v in self.strategy_weights.items()},
            "portfolio_sharpe": round(self.portfolio_sharpe, 2),
            "portfolio_calmar": round(self.portfolio_calmar, 2),
            "adjustments": self.adjustments_made
        }


# =============================================================================
# PERFORMANCE CALCULATORS
# =============================================================================

class SharpeCalculator:
    """滚动 Sharpe Ratio 计算器"""
    
    def __init__(self, window_days: int = 30, risk_free_rate: float = 0.04):
        self.window_days = window_days
        self.risk_free_rate = risk_free_rate
        self.daily_rfr = risk_free_rate / 365
    
    def calculate(self, returns: List[float]) -> float:
        """
        计算 Sharpe Ratio
        
        Args:
            returns: 日收益率列表
        """
        if len(returns) < 5:
            return 0.0
        
        returns_arr = np.array(returns[-self.window_days:])
        
        # 超额收益
        excess_returns = returns_arr - self.daily_rfr
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return < 0.0001:
            return 0.0
        
        # 年化 Sharpe
        sharpe = mean_return / std_return * np.sqrt(365)
        
        # 限制范围
        return np.clip(sharpe, -5.0, 5.0)


class CalmarCalculator:
    """滚动 Calmar Ratio 计算器"""
    
    def __init__(self, window_days: int = 90, min_drawdown: float = 0.01):
        self.window_days = window_days
        self.min_drawdown = min_drawdown
    
    def calculate(self, equity_curve: List[float]) -> Tuple[float, float]:
        """
        计算 Calmar Ratio
        
        Args:
            equity_curve: 权益曲线 (累计)
            
        Returns:
            (calmar_ratio, max_drawdown)
        """
        if len(equity_curve) < 10:
            return 0.0, 0.0
        
        curve = np.array(equity_curve[-self.window_days:])
        
        # 计算最大回撤
        peak = np.maximum.accumulate(curve)
        drawdown = (peak - curve) / peak
        max_dd = np.max(drawdown)
        
        # 计算年化收益
        total_return = (curve[-1] / curve[0]) - 1
        days = len(curve)
        annual_return = total_return * (365 / days)
        
        # Calmar = 年化收益 / 最大回撤
        if max_dd < self.min_drawdown:
            calmar = annual_return / self.min_drawdown
        else:
            calmar = annual_return / max_dd
        
        # 限制范围
        calmar = np.clip(calmar, -10.0, 10.0)
        
        return calmar, max_dd


class WinRateCalculator:
    """胜率计算器"""
    
    def __init__(self, short_window: int = 7, medium_window: int = 30):
        self.short_window = short_window
        self.medium_window = medium_window
    
    def calculate(self, trade_results: List[bool], 
                  timestamps: List[datetime]) -> Tuple[float, float, int, int]:
        """
        计算短期和中期胜率
        
        Args:
            trade_results: 交易结果 (True=盈利, False=亏损)
            timestamps: 交易时间戳
            
        Returns:
            (short_winrate, medium_winrate, short_count, medium_count)
        """
        if not trade_results:
            return 0.5, 0.5, 0, 0
        
        now = datetime.now(timezone.utc)
        
        # 短期
        short_cutoff = now - timedelta(days=self.short_window)
        short_trades = [r for r, t in zip(trade_results, timestamps) if t >= short_cutoff]
        
        # 中期
        medium_cutoff = now - timedelta(days=self.medium_window)
        medium_trades = [r for r, t in zip(trade_results, timestamps) if t >= medium_cutoff]
        
        # 计算胜率
        short_winrate = sum(short_trades) / len(short_trades) if short_trades else 0.5
        medium_winrate = sum(medium_trades) / len(medium_trades) if medium_trades else 0.5
        
        return short_winrate, medium_winrate, len(short_trades), len(medium_trades)


# =============================================================================
# ADAPTIVE WEIGHT MANAGER (v5.2.1)
# =============================================================================

class AdaptiveWeightManagerV521:
    """
    自适应策略权重管理器 (v5.2.1)
    
    核心功能:
    1. 追踪各策略的表现记录
    2. 计算 Sharpe、Calmar、胜率
    3. 综合计算权重调整
    4. 平滑权重变化
    """
    
    # 默认策略列表 (from v3.2)
    DEFAULT_STRATEGIES = [
        # Bear
        "mean_reversion", "volatility_breakout", "rsi_divergence",
        # Bull
        "trend_following", "momentum", "obv_confirmation",
        # Volatile
        "range_breakout", "vwap_deviation", "volume_profile",
        # Universal
        "orderbook_imbalance", "funding_rate", "liquidation_cascade",
        # Alpha agents
        "sentiment", "onchain", "microstructure"
    ]
    
    def __init__(self, 
                 config: Optional[AdaptiveWeightConfigV521] = None,
                 strategies: List[str] = None):
        self.config = config or AdaptiveWeightConfigV521()
        self.strategies = strategies or self.DEFAULT_STRATEGIES
        
        # 计算器
        self.sharpe_calc = SharpeCalculator(
            window_days=self.config.sharpe_window_days,
            risk_free_rate=self.config.risk_free_rate
        )
        self.calmar_calc = CalmarCalculator(
            window_days=self.config.calmar_window_days,
            min_drawdown=self.config.min_drawdown_for_calmar
        )
        self.winrate_calc = WinRateCalculator(
            short_window=self.config.short_term_window_days,
            medium_window=self.config.medium_term_window_days
        )
        
        # 数据存储 (每策略)
        self._returns: Dict[str, deque] = {
            s: deque(maxlen=365) for s in self.strategies
        }
        self._equity_curves: Dict[str, deque] = {
            s: deque(maxlen=365) for s in self.strategies
        }
        self._trade_results: Dict[str, List[bool]] = {
            s: [] for s in self.strategies
        }
        self._trade_timestamps: Dict[str, List[datetime]] = {
            s: [] for s in self.strategies
        }
        
        # 当前权重
        self._current_weights: Dict[str, float] = {
            s: 1.0 for s in self.strategies
        }
        
        # 当前指标
        self._current_metrics: Dict[str, StrategyMetrics] = {}
        
        # 历史
        self._adjustment_history: deque = deque(maxlen=100)
        
        # 上次评估时间
        self._last_evaluation: Optional[datetime] = None
        
        # EventBus
        self._event_bus = None
        
        logger.info(f"AdaptiveWeightManagerV521 initialized with {len(self.strategies)} strategies")
    
    def _get_event_bus(self):
        """延迟获取 EventBus"""
        if self._event_bus is None:
            try:
                from infra.event_bus import get_event_bus
                self._event_bus = get_event_bus()
            except ImportError:
                pass
        return self._event_bus
    
    def record_trade(self, 
                     strategy_name: str,
                     pnl_pct: float,
                     is_winner: bool,
                     timestamp: Optional[datetime] = None):
        """
        记录策略交易结果
        
        Args:
            strategy_name: 策略名称
            pnl_pct: 收益率 (小数)
            is_winner: 是否盈利
            timestamp: 交易时间
        """
        ts = timestamp or datetime.now(timezone.utc)
        
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return
        
        # 记录收益率
        self._returns[strategy_name].append(pnl_pct)
        
        # 更新权益曲线
        if not self._equity_curves[strategy_name]:
            self._equity_curves[strategy_name].append(1.0)
        prev_equity = self._equity_curves[strategy_name][-1]
        new_equity = prev_equity * (1 + pnl_pct)
        self._equity_curves[strategy_name].append(new_equity)
        
        # 记录胜负
        self._trade_results[strategy_name].append(is_winner)
        self._trade_timestamps[strategy_name].append(ts)
        
        # 清理旧数据
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.calmar_window_days + 10)
        while (self._trade_timestamps[strategy_name] and 
               self._trade_timestamps[strategy_name][0] < cutoff):
            self._trade_timestamps[strategy_name].pop(0)
            self._trade_results[strategy_name].pop(0)
        
        logger.debug(f"Trade recorded: {strategy_name} pnl={pnl_pct:.2%} win={is_winner}")
    
    def update_daily_return(self, strategy_name: str, daily_return: float):
        """记录日收益率 (用于 Sharpe 计算)"""
        if strategy_name in self._returns:
            self._returns[strategy_name].append(daily_return)
    
    def compute_strategy_metrics(self, strategy_name: str) -> StrategyMetrics:
        """计算单个策略的所有指标"""
        metrics = StrategyMetrics(
            strategy_name=strategy_name,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Sharpe
        returns = list(self._returns.get(strategy_name, []))
        if len(returns) >= 10:
            metrics.sharpe_ratio = self.sharpe_calc.calculate(returns)
        
        # Calmar
        equity = list(self._equity_curves.get(strategy_name, []))
        if len(equity) >= 10:
            calmar, max_dd = self.calmar_calc.calculate(equity)
            metrics.calmar_ratio = calmar
            metrics.max_drawdown_pct = max_dd
        
        # 胜率
        results = self._trade_results.get(strategy_name, [])
        timestamps = self._trade_timestamps.get(strategy_name, [])
        
        if results:
            short_wr, medium_wr, short_cnt, medium_cnt = self.winrate_calc.calculate(
                results, timestamps
            )
            metrics.winrate_short = short_wr
            metrics.winrate_medium = medium_wr
            metrics.trade_count_short = short_cnt
            metrics.trade_count_medium = medium_cnt
        
        # 数据充足性
        metrics.sufficient_data = (
            len(returns) >= 10 and 
            metrics.trade_count_medium >= self.config.min_trades_for_winrate
        )
        
        # 辅助指标
        if returns:
            metrics.total_pnl_pct = sum(returns)
        
        if results:
            wins = [r for r, is_win in zip(returns[-len(results):], results) if is_win]
            losses = [r for r, is_win in zip(returns[-len(results):], results) if not is_win]
            
            if wins:
                metrics.avg_win_pct = np.mean(wins)
            if losses:
                metrics.avg_loss_pct = np.mean(losses)
            if losses and sum(abs(l) for l in losses) > 0:
                metrics.profit_factor = sum(wins) / abs(sum(losses))
        
        return metrics
    
    def compute_weight(self, metrics: StrategyMetrics) -> float:
        """
        根据指标计算策略权重
        
        综合公式:
        weight = w1 * normalize(sharpe) + w2 * normalize(calmar) + 
                 w3 * short_winrate + w4 * medium_winrate
        """
        if not metrics.sufficient_data:
            return 1.0  # 数据不足，保持中性
        
        # 标准化 Sharpe (假设正常范围 -2 到 2)
        sharpe_score = (metrics.sharpe_ratio + 2) / 4  # 映射到 0-1
        sharpe_score = np.clip(sharpe_score, 0, 1)
        
        # 标准化 Calmar (假设正常范围 -2 到 2)
        calmar_score = (metrics.calmar_ratio + 2) / 4
        calmar_score = np.clip(calmar_score, 0, 1)
        
        # 胜率已经是 0-1
        short_wr_score = metrics.winrate_short
        medium_wr_score = metrics.winrate_medium
        
        # 综合得分
        score = (
            self.config.sharpe_weight * sharpe_score +
            self.config.calmar_weight * calmar_score +
            self.config.winrate_short_weight * short_wr_score +
            self.config.winrate_medium_weight * medium_wr_score
        )
        
        # 映射到权重 (0.5 = 中性权重 1.0)
        # score 0.0 -> weight 0.2
        # score 0.5 -> weight 1.0
        # score 1.0 -> weight 2.0
        weight = self.config.min_strategy_weight + \
                 (self.config.max_strategy_weight - self.config.min_strategy_weight) * score
        
        return weight
    
    def evaluate_all_strategies(self) -> WeightAdjustmentResult:
        """评估所有策略并调整权重"""
        adjustments_made = []
        new_metrics = {}
        new_weights = {}
        
        for strategy in self.strategies:
            # 计算指标
            metrics = self.compute_strategy_metrics(strategy)
            new_metrics[strategy] = metrics
            
            # 计算新权重
            raw_weight = self.compute_weight(metrics)
            
            # 平滑调整 (EMA)
            old_weight = self._current_weights.get(strategy, 1.0)
            smoothed_weight = (
                self.config.weight_smoothing_alpha * raw_weight +
                (1 - self.config.weight_smoothing_alpha) * old_weight
            )
            
            new_weights[strategy] = smoothed_weight
            metrics.computed_weight = smoothed_weight
            
            # 记录显著调整
            if abs(smoothed_weight - old_weight) > 0.1:
                direction = "↑" if smoothed_weight > old_weight else "↓"
                adjustments_made.append(
                    f"{strategy}: {old_weight:.2f} -> {smoothed_weight:.2f} {direction} "
                    f"(Sharpe={metrics.sharpe_ratio:.2f}, WR={metrics.winrate_short:.0%})"
                )
        
        # 更新状态
        self._current_weights = new_weights
        self._current_metrics = new_metrics
        self._last_evaluation = datetime.now(timezone.utc)
        
        # 计算组合级指标
        portfolio_sharpe = self._compute_portfolio_sharpe()
        portfolio_calmar = self._compute_portfolio_calmar()
        
        result = WeightAdjustmentResult(
            timestamp=datetime.now(timezone.utc),
            strategy_weights=new_weights,
            strategy_metrics=new_metrics,
            adjustments_made=adjustments_made,
            portfolio_sharpe=portfolio_sharpe,
            portfolio_calmar=portfolio_calmar
        )
        
        self._adjustment_history.append(result)
        
        # 发送事件
        if adjustments_made:
            bus = self._get_event_bus()
            if bus:
                try:
                    from infra.event_bus import Event, EventType, EventPriority
                    event = Event(
                        event_type=EventType.MODE_CHANGED,
                        source="adaptive_weight_manager",
                        priority=EventPriority.NORMAL,
                        payload={
                            "sub_type": "WEIGHT_ADJUSTMENT",
                            "adjustments": adjustments_made,
                            "portfolio_sharpe": portfolio_sharpe
                        }
                    )
                    bus.publish(event)
                except Exception as e:
                    logger.error(f"Failed to emit event: {e}")
        
        logger.info(f"Weight evaluation complete: {len(adjustments_made)} adjustments")
        for adj in adjustments_made:
            logger.info(f"  {adj}")
        
        return result
    
    def _compute_portfolio_sharpe(self) -> float:
        """计算组合 Sharpe"""
        all_returns = []
        for strategy in self.strategies:
            returns = list(self._returns.get(strategy, []))
            if returns:
                weight = self._current_weights.get(strategy, 1.0) / len(self.strategies)
                weighted_returns = [r * weight for r in returns[-30:]]
                all_returns.append(weighted_returns)
        
        if not all_returns:
            return 0.0
        
        # 对齐长度
        min_len = min(len(r) for r in all_returns)
        if min_len < 5:
            return 0.0
        
        # 组合收益
        portfolio_returns = np.sum([r[-min_len:] for r in all_returns], axis=0)
        
        return self.sharpe_calc.calculate(list(portfolio_returns))
    
    def _compute_portfolio_calmar(self) -> float:
        """计算组合 Calmar"""
        all_equity = []
        for strategy in self.strategies:
            equity = list(self._equity_curves.get(strategy, []))
            if len(equity) > 10:
                weight = self._current_weights.get(strategy, 1.0) / len(self.strategies)
                weighted_equity = [e * weight for e in equity[-90:]]
                all_equity.append(weighted_equity)
        
        if not all_equity:
            return 0.0
        
        # 对齐
        min_len = min(len(e) for e in all_equity)
        if min_len < 10:
            return 0.0
        
        # 组合权益
        portfolio_equity = np.sum([e[-min_len:] for e in all_equity], axis=0)
        
        calmar, _ = self.calmar_calc.calculate(list(portfolio_equity))
        return calmar
    
    def get_weight(self, strategy_name: str) -> float:
        """获取策略权重"""
        return self._current_weights.get(strategy_name, 1.0)
    
    def get_all_weights(self) -> Dict[str, float]:
        """获取所有策略权重"""
        # 检查是否需要重新评估
        if self._last_evaluation is None:
            self.evaluate_all_strategies()
        else:
            hours_since = (datetime.now(timezone.utc) - self._last_evaluation).total_seconds() / 3600
            if hours_since >= self.config.evaluation_interval_hours:
                self.evaluate_all_strategies()
        
        return dict(self._current_weights)
    
    def get_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        return self._current_metrics.get(strategy_name)
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        # 确保有最新数据
        self.get_all_weights()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_sharpe": round(self._compute_portfolio_sharpe(), 2),
            "portfolio_calmar": round(self._compute_portfolio_calmar(), 2),
            "strategies": {
                name: {
                    "weight": round(self._current_weights.get(name, 1.0), 3),
                    "sharpe": round(m.sharpe_ratio, 2) if m else 0,
                    "calmar": round(m.calmar_ratio, 2) if m else 0,
                    "winrate_short": round(m.winrate_short, 2) if m else 0.5,
                    "sufficient_data": m.sufficient_data if m else False
                }
                for name, m in self._current_metrics.items()
            },
            "recent_adjustments": [
                adj.to_dict() for adj in list(self._adjustment_history)[-5:]
            ]
        }
    
    def reset(self):
        """重置所有数据"""
        for s in self.strategies:
            self._returns[s].clear()
            self._equity_curves[s].clear()
            self._trade_results[s] = []
            self._trade_timestamps[s] = []
            self._current_weights[s] = 1.0
        
        self._current_metrics.clear()
        self._adjustment_history.clear()
        self._last_evaluation = None


# =============================================================================
# AUTHORITY FUSION INTEGRATION
# =============================================================================

class AuthorityFusionWeightAdapter:
    """
    将自适应权重适配到 AuthorityFusion
    
    这个适配器封装了 AdaptiveWeightManager，
    提供与现有 authority_fusion.py 兼容的接口
    """
    
    def __init__(self, weight_manager: AdaptiveWeightManagerV521):
        self.weight_manager = weight_manager
    
    def get_agent_weight_modifier(self, agent_name: str) -> float:
        """
        获取代理权重修改器
        
        用于 AuthorityFusionEngine 中调整代理信号
        """
        # 映射代理名到策略名
        agent_to_strategies = {
            "quant": ["trend_following", "momentum", "mean_reversion"],
            "sentiment": ["sentiment"],
            "onchain": ["onchain"],
            "drl": ["range_breakout", "vwap_deviation"],
            "macro": ["liquidation_cascade", "funding_rate"]
        }
        
        strategies = agent_to_strategies.get(agent_name, [])
        if not strategies:
            return 1.0
        
        # 取相关策略的平均权重
        weights = [self.weight_manager.get_weight(s) for s in strategies]
        return np.mean(weights) if weights else 1.0

    def get_adjusted_weights(
        self, base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply adaptive multipliers to base strategy weights.

        Used by StrategicCoordinator.get_adjusted_weights().
        For each strategy in *base_weights*, the adaptive weight from the
        manager is used as a multiplier (1.0 = no change).
        """
        all_adaptive = self.weight_manager.get_all_weights()
        adjusted: Dict[str, float] = {}
        for strategy, base_w in base_weights.items():
            multiplier = all_adaptive.get(strategy, 1.0)
            adjusted[strategy] = base_w * multiplier
        return adjusted

    def weighted_fusion(
        self,
        signals: Dict[str, Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Fuse strategy signals using adaptive weights.

        Args:
            signals: {strategy_name: (direction, confidence)}

        Returns:
            (fused_direction, fused_confidence)
        """
        if not signals:
            return 0.0, 0.0

        all_adaptive = self.weight_manager.get_all_weights()
        total_weight = 0.0
        weighted_dir = 0.0

        for strategy, (direction, confidence) in signals.items():
            w = all_adaptive.get(strategy, 1.0) * confidence
            weighted_dir += direction * w
            total_weight += w

        if total_weight == 0:
            return 0.0, 0.0

        fused_dir = weighted_dir / total_weight
        fused_conf = total_weight / len(signals)
        return fused_dir, fused_conf


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_adaptive_weight_manager_v521: Optional[AdaptiveWeightManagerV521] = None


def get_adaptive_weight_manager_v521(
    config: AdaptiveWeightConfigV521 = None
) -> AdaptiveWeightManagerV521:
    """获取单例"""
    global _adaptive_weight_manager_v521
    if _adaptive_weight_manager_v521 is None:
        _adaptive_weight_manager_v521 = AdaptiveWeightManagerV521(config)
    elif config is not None and getattr(_adaptive_weight_manager_v521, "config", None) != config:
        logger.info("[ADAPTIVE_WEIGHT_V521] Config changed; refreshing singleton instance")
        _adaptive_weight_manager_v521 = AdaptiveWeightManagerV521(config)
    return _adaptive_weight_manager_v521


def reset_adaptive_weight_manager_v521():
    """重置单例"""
    global _adaptive_weight_manager_v521
    _adaptive_weight_manager_v521 = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AdaptiveWeightConfigV521',
    'StrategyPerformanceRecord',
    'StrategyMetrics',
    'WeightAdjustmentResult',
    'SharpeCalculator',
    'CalmarCalculator',
    'WinRateCalculator',
    'AdaptiveWeightManagerV521',
    'AuthorityFusionWeightAdapter',
    'get_adaptive_weight_manager_v521',
    'reset_adaptive_weight_manager_v521'
]


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    manager = AdaptiveWeightManagerV521()
    
    # 模拟交易数据
    print("\n=== Simulating Trades ===")
    
    for day in range(60):
        for strategy in manager.strategies[:5]:  # 测试前5个策略
            # 模拟每天1-3笔交易
            for _ in range(random.randint(1, 3)):
                # 不同策略有不同的胜率
                strategy_skill = {
                    "trend_following": 0.6,
                    "momentum": 0.55,
                    "mean_reversion": 0.45,
                    "volatility_breakout": 0.5,
                    "rsi_divergence": 0.52
                }.get(strategy, 0.5)
                
                is_winner = random.random() < strategy_skill
                
                if is_winner:
                    pnl = random.uniform(0.005, 0.03)  # 0.5% - 3%
                else:
                    pnl = random.uniform(-0.02, -0.005)  # -2% - -0.5%
                
                manager.record_trade(
                    strategy,
                    pnl,
                    is_winner,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=60-day)
                )
    
    # 评估
    print("\n=== Evaluating Strategies ===")
    result = manager.evaluate_all_strategies()
    
    print(f"\nPortfolio Sharpe: {result.portfolio_sharpe:.2f}")
    print(f"Portfolio Calmar: {result.portfolio_calmar:.2f}")
    
    print("\n=== Strategy Weights ===")
    for strategy, weight in sorted(result.strategy_weights.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
        metrics = result.strategy_metrics.get(strategy)
        if metrics:
            print(f"{strategy}:")
            print(f"  Weight: {weight:.3f}")
            print(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"  Calmar: {metrics.calmar_ratio:.2f}")
            print(f"  WinRate (7d): {metrics.winrate_short:.0%}")
            print(f"  WinRate (30d): {metrics.winrate_medium:.0%}")
    
    print("\n=== Dashboard Data ===")
    dashboard = manager.get_dashboard_data()
    import json
    print(json.dumps(dashboard, indent=2, default=str))

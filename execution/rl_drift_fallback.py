"""
================================================================================
HMATS v5.2.1 - RL 执行漂移检测与安全回退
RL Execution Drift Detection & Safe Fallback
================================================================================
Purpose: 检测 RL 模型信号与环境分布差异，自动切换到规则化执行

SOTA 对照:
- 在 RL 执行代理中接入 drift detector
- 若模型信号与环境分布差异过大，则自动切换到规则化的执行模式

Features:
1. 多维度漂移检测 (KL散度、PSI、KS检验)
2. 特征分布监控
3. 预测质量追踪
4. 自动回退机制
5. 渐进恢复策略
6. EventBus 集成

Author: HMATS v5.2.1 SOTA Upgrade
Version: 5.2.1
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class DriftStatus(Enum):
    """漂移状态"""
    NORMAL = "normal"           # 正常
    WARNING = "warning"         # 警告
    CRITICAL = "critical"       # 严重
    FALLBACK_ACTIVE = "fallback_active"  # 回退激活


class DriftType(Enum):
    """漂移类型"""
    FEATURE_DRIFT = "feature"       # 特征分布漂移
    CONCEPT_DRIFT = "concept"       # 概念漂移 (输入-输出关系变化)
    PREDICTION_DRIFT = "prediction" # 预测分布漂移
    PERFORMANCE_DRIFT = "performance"  # 性能漂移


class ExecutionMode(Enum):
    """执行模式"""
    RL_AGENT = "rl_agent"           # RL 代理执行
    RULE_BASED = "rule_based"       # 规则化执行
    HYBRID = "hybrid"               # 混合模式


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DriftMetrics:
    """漂移指标"""
    timestamp: datetime
    
    # 特征漂移
    feature_psi: Dict[str, float] = field(default_factory=dict)
    feature_ks: Dict[str, float] = field(default_factory=dict)
    
    # 预测漂移
    prediction_kl_divergence: float = 0.0
    prediction_mean_shift: float = 0.0
    prediction_std_change: float = 0.0
    
    # 性能漂移
    performance_degradation: float = 0.0
    recent_accuracy: float = 0.0
    
    # 综合评分
    overall_drift_score: float = 0.0
    drift_status: DriftStatus = DriftStatus.NORMAL
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "feature_psi": self.feature_psi,
            "prediction_kl": self.prediction_kl_divergence,
            "prediction_mean_shift": self.prediction_mean_shift,
            "performance_degradation": self.performance_degradation,
            "overall_score": self.overall_drift_score,
            "status": self.drift_status.value
        }


@dataclass
class DriftDetectorConfig:
    """漂移检测器配置"""
    # 窗口大小
    reference_window: int = 500     # 参考窗口
    current_window: int = 50        # 当前窗口
    min_samples: int = 30           # 最小样本数
    
    # 阈值
    psi_warning_threshold: float = 0.1      # PSI 警告阈值
    psi_critical_threshold: float = 0.25    # PSI 严重阈值
    kl_warning_threshold: float = 0.1       # KL 散度警告
    kl_critical_threshold: float = 0.3      # KL 散度严重
    ks_warning_threshold: float = 0.1       # KS 统计量警告
    
    # 性能阈值
    performance_drop_warning: float = 0.1   # 性能下降 10% 警告
    performance_drop_critical: float = 0.2  # 性能下降 20% 严重
    
    # 回退配置
    auto_fallback_enabled: bool = True
    fallback_cooldown_minutes: int = 30     # 回退冷却时间
    recovery_check_interval: int = 10       # 恢复检查间隔 (样本数)
    
    # 权重 (各维度在综合评分中的权重)
    feature_drift_weight: float = 0.3
    prediction_drift_weight: float = 0.4
    performance_drift_weight: float = 0.3


@dataclass
class FallbackDecision:
    """回退决策"""
    timestamp: datetime
    should_fallback: bool
    reason: str
    drift_metrics: DriftMetrics
    recommended_mode: ExecutionMode
    recovery_estimate_minutes: int = 0


# =============================================================================
# DRIFT DETECTION ALGORITHMS
# =============================================================================

class DriftDetectionAlgorithms:
    """漂移检测算法集合"""
    
    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, 
                    n_bins: int = 10) -> float:
        """
        计算 Population Stability Index (PSI)
        
        PSI < 0.1: 无显著变化
        0.1 <= PSI < 0.25: 中等变化
        PSI >= 0.25: 显著变化
        """
        # 计算分位点
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # 计算分布
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]
        
        # 避免零值
        ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
        cur_pct = (cur_counts + 1) / (len(current) + n_bins)
        
        # 计算 PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    @staticmethod
    def compute_kl_divergence(reference: np.ndarray, current: np.ndarray,
                              n_bins: int = 50) -> float:
        """计算 KL 散度"""
        # 确定 bin 范围
        all_data = np.concatenate([reference, current])
        bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)
        
        # 计算直方图
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        
        # 避免零值
        ref_hist = ref_hist + 1e-10
        cur_hist = cur_hist + 1e-10
        
        # 归一化
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()
        
        # KL 散度
        kl = np.sum(cur_hist * np.log(cur_hist / ref_hist))
        
        return float(kl)
    
    @staticmethod
    def compute_ks_statistic(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        计算 Kolmogorov-Smirnov 统计量
        
        Returns:
            (ks_statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)
    
    @staticmethod
    def compute_mean_shift(reference: np.ndarray, current: np.ndarray) -> float:
        """计算均值偏移 (标准化)"""
        ref_mean = np.mean(reference)
        ref_std = np.std(reference) + 1e-10
        cur_mean = np.mean(current)
        
        return (cur_mean - ref_mean) / ref_std


# =============================================================================
# RL DRIFT DETECTOR
# =============================================================================

class RLDriftDetector:
    """
    RL 模型漂移检测器
    
    核心功能:
    1. 监控特征分布漂移
    2. 监控预测分布漂移
    3. 监控性能漂移
    4. 决定是否触发回退
    """
    
    def __init__(self, config: DriftDetectorConfig = None,
                 feature_names: List[str] = None):
        self.config = config or DriftDetectorConfig()
        self.feature_names = feature_names or []
        
        # 参考分布 (训练时的数据分布)
        self.reference_features: Dict[str, deque] = {
            name: deque(maxlen=self.config.reference_window)
            for name in self.feature_names
        }
        self.reference_predictions: deque = deque(maxlen=self.config.reference_window)
        self.reference_outcomes: deque = deque(maxlen=self.config.reference_window)
        
        # 当前分布
        self.current_features: Dict[str, deque] = {
            name: deque(maxlen=self.config.current_window)
            for name in self.feature_names
        }
        self.current_predictions: deque = deque(maxlen=self.config.current_window)
        self.current_outcomes: deque = deque(maxlen=self.config.current_window)
        
        # 状态
        self.drift_status: DriftStatus = DriftStatus.NORMAL
        self.current_mode: ExecutionMode = ExecutionMode.RL_AGENT
        self.last_fallback_time: Optional[datetime] = None
        self.recovery_attempts: int = 0
        
        # 历史
        self.metrics_history: deque = deque(maxlen=500)
        self.fallback_history: deque = deque(maxlen=100)
        
        # 事件回调
        self.event_callbacks: List[Callable] = []
        
        logger.info("RLDriftDetector initialized")
    
    def register_event_callback(self, callback: Callable):
        """注册事件回调"""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """发布事件"""
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "RLDriftDetector",
            "data": data
        }
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def set_reference_distribution(self, features: Dict[str, np.ndarray],
                                   predictions: np.ndarray,
                                   outcomes: np.ndarray = None):
        """设置参考分布 (通常用训练数据)"""
        for name, values in features.items():
            if name in self.reference_features:
                self.reference_features[name] = deque(values, maxlen=self.config.reference_window)
        
        self.reference_predictions = deque(predictions, maxlen=self.config.reference_window)
        
        if outcomes is not None:
            self.reference_outcomes = deque(outcomes, maxlen=self.config.reference_window)
        
        logger.info("Reference distribution set")
    
    def update(self, features: Dict[str, float], prediction: float,
               outcome: float = None):
        """
        更新当前数据
        
        Args:
            features: 当前特征值
            prediction: RL 模型预测
            outcome: 实际结果 (可延迟提供)
        """
        # 更新特征
        for name, value in features.items():
            if name in self.current_features:
                self.current_features[name].append(value)
        
        # 更新预测
        self.current_predictions.append(prediction)
        
        # 更新结果
        if outcome is not None:
            self.current_outcomes.append(outcome)
    
    def detect_drift(self) -> DriftMetrics:
        """执行漂移检测"""
        metrics = DriftMetrics(timestamp=datetime.now(timezone.utc))
        
        # 检查样本数
        if len(self.current_predictions) < self.config.min_samples:
            metrics.drift_status = self.drift_status
            return metrics
        
        # 特征漂移检测
        feature_scores = []
        for name in self.feature_names:
            if (len(self.reference_features.get(name, [])) >= self.config.min_samples and
                len(self.current_features.get(name, [])) >= self.config.min_samples):
                
                ref = np.array(self.reference_features[name])
                cur = np.array(self.current_features[name])
                
                psi = DriftDetectionAlgorithms.compute_psi(ref, cur)
                ks, _ = DriftDetectionAlgorithms.compute_ks_statistic(ref, cur)
                
                metrics.feature_psi[name] = psi
                metrics.feature_ks[name] = ks
                
                # 评分 (0-1, 越高越严重)
                psi_score = min(1.0, psi / self.config.psi_critical_threshold)
                feature_scores.append(psi_score)
        
        avg_feature_score = np.mean(feature_scores) if feature_scores else 0.0
        
        # 预测漂移检测
        if (len(self.reference_predictions) >= self.config.min_samples and
            len(self.current_predictions) >= self.config.min_samples):
            
            ref_pred = np.array(self.reference_predictions)
            cur_pred = np.array(self.current_predictions)
            
            metrics.prediction_kl_divergence = DriftDetectionAlgorithms.compute_kl_divergence(
                ref_pred, cur_pred
            )
            metrics.prediction_mean_shift = DriftDetectionAlgorithms.compute_mean_shift(
                ref_pred, cur_pred
            )
            metrics.prediction_std_change = (
                np.std(cur_pred) / (np.std(ref_pred) + 1e-10) - 1.0
            )
        
        pred_score = min(1.0, metrics.prediction_kl_divergence / self.config.kl_critical_threshold)
        
        # 性能漂移检测
        perf_score = 0.0
        if len(self.current_outcomes) >= self.config.min_samples:
            cur_accuracy = self._compute_accuracy(
                list(self.current_predictions)[-len(self.current_outcomes):],
                list(self.current_outcomes)
            )
            
            ref_accuracy = self._compute_accuracy(
                list(self.reference_predictions)[:len(self.reference_outcomes)],
                list(self.reference_outcomes)
            ) if len(self.reference_outcomes) >= self.config.min_samples else 0.5
            
            metrics.recent_accuracy = cur_accuracy
            metrics.performance_degradation = max(0, ref_accuracy - cur_accuracy)
            
            perf_score = min(1.0, metrics.performance_degradation / self.config.performance_drop_critical)
        
        # 综合评分
        metrics.overall_drift_score = (
            avg_feature_score * self.config.feature_drift_weight +
            pred_score * self.config.prediction_drift_weight +
            perf_score * self.config.performance_drift_weight
        )
        
        # 确定状态
        if metrics.overall_drift_score >= 0.7:
            metrics.drift_status = DriftStatus.CRITICAL
        elif metrics.overall_drift_score >= 0.4:
            metrics.drift_status = DriftStatus.WARNING
        else:
            metrics.drift_status = DriftStatus.NORMAL
        
        # 记录历史
        self.metrics_history.append(metrics)
        
        # 检查是否需要状态变更
        old_status = self.drift_status
        self.drift_status = metrics.drift_status
        
        if old_status != self.drift_status:
            self._emit_event("DRIFT_STATUS_CHANGED", {
                "old_status": old_status.value,
                "new_status": self.drift_status.value,
                "score": metrics.overall_drift_score
            })
        
        return metrics
    
    def _compute_accuracy(self, predictions: List[float], 
                          outcomes: List[float]) -> float:
        """计算预测准确性"""
        if not predictions or not outcomes:
            return 0.5
        
        n = min(len(predictions), len(outcomes))
        correct = sum(
            1 for p, o in zip(predictions[:n], outcomes[:n])
            if (p > 0) == (o > 0)  # 方向一致
        )
        return correct / n
    
    def should_fallback(self) -> FallbackDecision:
        """判断是否应该回退"""
        if not self.config.auto_fallback_enabled:
            return FallbackDecision(
                timestamp=datetime.now(timezone.utc),
                should_fallback=False,
                reason="Auto fallback disabled",
                drift_metrics=self.metrics_history[-1] if self.metrics_history else DriftMetrics(datetime.now(timezone.utc)),
                recommended_mode=ExecutionMode.RL_AGENT
            )
        
        # 检测漂移
        metrics = self.detect_drift()
        
        # 检查冷却期
        if self.last_fallback_time:
            cooldown = timedelta(minutes=self.config.fallback_cooldown_minutes)
            if datetime.now(timezone.utc) - self.last_fallback_time < cooldown:
                if self.current_mode == ExecutionMode.RULE_BASED:
                    return FallbackDecision(
                        timestamp=datetime.now(timezone.utc),
                        should_fallback=True,
                        reason="Still in cooldown period",
                        drift_metrics=metrics,
                        recommended_mode=ExecutionMode.RULE_BASED,
                        recovery_estimate_minutes=int(
                            (cooldown - (datetime.now(timezone.utc) - self.last_fallback_time)).total_seconds() / 60
                        )
                    )
        
        # 决定是否回退
        should_fb = False
        reason = ""
        recommended_mode = ExecutionMode.RL_AGENT
        
        if metrics.drift_status == DriftStatus.CRITICAL:
            should_fb = True
            reason = f"Critical drift detected (score={metrics.overall_drift_score:.2f})"
            recommended_mode = ExecutionMode.RULE_BASED
        
        elif metrics.drift_status == DriftStatus.WARNING and self.current_mode == ExecutionMode.RL_AGENT:
            # 警告状态下使用混合模式
            reason = f"Warning drift detected (score={metrics.overall_drift_score:.2f})"
            recommended_mode = ExecutionMode.HYBRID
        
        elif metrics.drift_status == DriftStatus.NORMAL and self.current_mode != ExecutionMode.RL_AGENT:
            # 恢复检查
            reason = "Drift normalized, attempting recovery"
            recommended_mode = ExecutionMode.RL_AGENT
        
        decision = FallbackDecision(
            timestamp=datetime.now(timezone.utc),
            should_fallback=should_fb,
            reason=reason,
            drift_metrics=metrics,
            recommended_mode=recommended_mode
        )
        
        # 记录
        if should_fb or recommended_mode != self.current_mode:
            self.fallback_history.append(decision)
        
        return decision
    
    def activate_fallback(self, reason: str = "Manual"):
        """激活回退"""
        old_mode = self.current_mode
        self.current_mode = ExecutionMode.RULE_BASED
        self.drift_status = DriftStatus.FALLBACK_ACTIVE
        self.last_fallback_time = datetime.now(timezone.utc)
        self.recovery_attempts = 0
        
        logger.warning(f"Fallback activated: {reason}")
        
        self._emit_event("FALLBACK_ACTIVATED", {
            "old_mode": old_mode.value,
            "new_mode": self.current_mode.value,
            "reason": reason
        })
    
    def attempt_recovery(self) -> bool:
        """尝试恢复到 RL 模式"""
        if self.current_mode != ExecutionMode.RULE_BASED:
            return True
        
        metrics = self.detect_drift()
        
        if metrics.drift_status == DriftStatus.NORMAL:
            self.recovery_attempts += 1
            
            # 需要连续多次正常才恢复
            if self.recovery_attempts >= 3:
                self.current_mode = ExecutionMode.RL_AGENT
                self.drift_status = DriftStatus.NORMAL
                
                logger.info("Recovery successful, switching back to RL mode")
                
                self._emit_event("RECOVERY_SUCCESSFUL", {
                    "attempts": self.recovery_attempts,
                    "score": metrics.overall_drift_score
                })
                
                return True
        else:
            self.recovery_attempts = 0
        
        return False
    
    def get_current_mode(self) -> ExecutionMode:
        """获取当前执行模式"""
        return self.current_mode
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        recent_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "status": self.drift_status.value,
            "mode": self.current_mode.value,
            "drift_score": recent_metrics.overall_drift_score if recent_metrics else 0.0,
            "recent_accuracy": recent_metrics.recent_accuracy if recent_metrics else 0.0,
            "feature_psi": recent_metrics.feature_psi if recent_metrics else {},
            "kl_divergence": recent_metrics.prediction_kl_divergence if recent_metrics else 0.0,
            "performance_degradation": recent_metrics.performance_degradation if recent_metrics else 0.0,
            "fallback_count": len(self.fallback_history),
            "last_fallback": self.last_fallback_time.isoformat() if self.last_fallback_time else None
        }
    
    def get_status(self) -> Dict:
        """获取检测器状态"""
        return {
            "drift_status": self.drift_status.value,
            "current_mode": self.current_mode.value,
            "samples_reference": len(self.reference_predictions),
            "samples_current": len(self.current_predictions),
            "recovery_attempts": self.recovery_attempts,
            "metrics_history_size": len(self.metrics_history)
        }


# =============================================================================
# SAFE FALLBACK EXECUTOR
# =============================================================================

class SafeFallbackExecutor:
    """
    安全回退执行器
    
    当 RL 模型漂移时，使用规则化策略执行
    """
    
    def __init__(self, drift_detector: RLDriftDetector = None):
        self.drift_detector = drift_detector or RLDriftDetector()
        
        # 规则化策略配置
        self.rule_config = {
            "twap_default_slices": 5,
            "max_order_size_pct": 0.02,  # 最大单笔 2%
            "spread_threshold_bps": 10,   # 点差阈值
            "urgency_timeout_seconds": 60
        }
    
    def execute(self, symbol: str, side: str, size: float, 
                orderbook: Dict = None, urgency: str = "medium") -> Dict:
        """
        统一执行接口
        
        自动根据漂移状态选择执行模式
        """
        # 检查漂移状态
        decision = self.drift_detector.should_fallback()
        mode = decision.recommended_mode
        
        # 如果需要回退，激活回退
        if decision.should_fallback and self.drift_detector.current_mode != ExecutionMode.RULE_BASED:
            self.drift_detector.activate_fallback(decision.reason)
            mode = ExecutionMode.RULE_BASED
        
        # 执行
        if mode == ExecutionMode.RL_AGENT:
            return self._execute_rl(symbol, side, size, orderbook, urgency)
        elif mode == ExecutionMode.RULE_BASED:
            return self._execute_rules(symbol, side, size, orderbook, urgency)
        else:  # HYBRID
            return self._execute_hybrid(symbol, side, size, orderbook, urgency)
    
    def _execute_rl(self, symbol: str, side: str, size: float,
                    orderbook: Dict, urgency: str) -> Dict:
        """RL 代理执行"""
        return {
            "mode": "rl_agent",
            "symbol": symbol,
            "side": side,
            "size": size,
            "strategy": "rl_optimized",
            "note": "Using RL agent for optimal execution"
        }
    
    def _execute_rules(self, symbol: str, side: str, size: float,
                       orderbook: Dict, urgency: str) -> Dict:
        """规则化执行"""
        # 简单 TWAP 拆单
        slices = self.rule_config["twap_default_slices"]
        slice_size = size / slices
        
        return {
            "mode": "rule_based",
            "symbol": symbol,
            "side": side,
            "size": size,
            "strategy": "twap",
            "slices": slices,
            "slice_size": slice_size,
            "interval_seconds": 60 / slices,
            "note": "Fallback to rule-based TWAP execution"
        }
    
    def _execute_hybrid(self, symbol: str, side: str, size: float,
                        orderbook: Dict, urgency: str) -> Dict:
        """混合执行"""
        # RL 建议 + 规则约束
        return {
            "mode": "hybrid",
            "symbol": symbol,
            "side": side,
            "size": size,
            "strategy": "constrained_rl",
            "max_size_per_slice": size * self.rule_config["max_order_size_pct"],
            "note": "Hybrid mode: RL suggestions with rule constraints"
        }
    
    def update_feedback(self, features: Dict[str, float], prediction: float,
                        outcome: float = None):
        """更新反馈数据"""
        self.drift_detector.update(features, prediction, outcome)
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "drift_detector": self.drift_detector.get_status(),
            "dashboard": self.drift_detector.get_dashboard_data()
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_drift_detector: Optional[RLDriftDetector] = None
_fallback_executor: Optional[SafeFallbackExecutor] = None


def get_drift_detector(config: DriftDetectorConfig = None) -> RLDriftDetector:
    """获取漂移检测器单例"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = RLDriftDetector(config)
    elif config is not None and _drift_detector.config != config:
        logger.info("[RL_DRIFT] Config changed; refreshing singleton instance")
        _drift_detector = RLDriftDetector(config)
    return _drift_detector


def get_fallback_executor() -> SafeFallbackExecutor:
    """获取回退执行器单例"""
    global _fallback_executor
    drift_detector = get_drift_detector()
    if _fallback_executor is None:
        _fallback_executor = SafeFallbackExecutor(drift_detector)
    elif _fallback_executor.drift_detector is not drift_detector:
        logger.info("[RL_FALLBACK] Drift detector changed; refreshing fallback executor")
        _fallback_executor = SafeFallbackExecutor(drift_detector)
    return _fallback_executor


def reset_drift_detection():
    """重置单例"""
    global _drift_detector, _fallback_executor
    _drift_detector = None
    _fallback_executor = None


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    # 创建检测器
    feature_names = ["spread", "depth", "volatility", "ofi"]
    detector = RLDriftDetector(feature_names=feature_names)
    
    # 设置参考分布 (模拟训练数据)
    n_ref = 500
    reference_features = {
        "spread": np.random.normal(5, 1, n_ref),
        "depth": np.random.normal(100000, 20000, n_ref),
        "volatility": np.random.normal(0.02, 0.005, n_ref),
        "ofi": np.random.normal(0, 1000, n_ref)
    }
    reference_predictions = np.random.normal(0, 0.5, n_ref)
    reference_outcomes = reference_predictions + np.random.normal(0, 0.1, n_ref)
    
    detector.set_reference_distribution(
        reference_features, reference_predictions, reference_outcomes
    )
    
    print("=== Phase 1: Normal Operation ===")
    for i in range(50):
        features = {
            "spread": random.gauss(5, 1),
            "depth": random.gauss(100000, 20000),
            "volatility": random.gauss(0.02, 0.005),
            "ofi": random.gauss(0, 1000)
        }
        pred = random.gauss(0, 0.5)
        outcome = pred + random.gauss(0, 0.1)
        
        detector.update(features, pred, outcome)
    
    decision = detector.should_fallback()
    print(f"Status: {decision.drift_metrics.drift_status.value}")
    print(f"Score: {decision.drift_metrics.overall_drift_score:.3f}")
    print(f"Mode: {decision.recommended_mode.value}")
    
    print("\n=== Phase 2: Introducing Drift ===")
    for i in range(50):
        # 引入漂移: spread 均值和方差都变化
        features = {
            "spread": random.gauss(15, 5),  # 漂移!
            "depth": random.gauss(50000, 30000),  # 漂移!
            "volatility": random.gauss(0.05, 0.02),  # 漂移!
            "ofi": random.gauss(0, 1000)
        }
        pred = random.gauss(0.5, 1.0)  # 预测分布也变了
        outcome = -pred  # 性能下降
        
        detector.update(features, pred, outcome)
    
    decision = detector.should_fallback()
    print(f"Status: {decision.drift_metrics.drift_status.value}")
    print(f"Score: {decision.drift_metrics.overall_drift_score:.3f}")
    print(f"Should fallback: {decision.should_fallback}")
    print(f"Reason: {decision.reason}")
    
    # 测试回退执行器
    print("\n=== Testing SafeFallbackExecutor ===")
    executor = SafeFallbackExecutor(detector)
    
    result = executor.execute("SOL/USD", "buy", 100.0, urgency="medium")
    print(f"Execution result: {result}")
    
    print(f"\nDashboard data: {detector.get_dashboard_data()}")

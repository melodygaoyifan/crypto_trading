"""
================================================================================
HMATS v5.2.1 - RL 执行漂移检测与安全回退
RL Execution Drift Detection and Safe Fallback
================================================================================
Purpose: 在 RL 执行代理中接入 drift detector，若模型信号与环境分布差异过大，
         则自动切换到规则化的执行模式

SOTA 对照:
1. 接入 drift detector 检测数据分布漂移
2. 若模型信号与环境分布差异过大，自动切换到规则化执行模式
3. 支持多种漂移检测方法

Features:
1. 多维度漂移检测 (KL散度, PSI, KS检验)
2. 自动安全回退机制
3. 回退策略库 (TWAP, VWAP, 被动跟随)
4. 渐进式恢复
5. EventBus 集成

Author: HMATS v5.2.1 SOTA Upgrade
Version: 5.2.1
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class DriftType(Enum):
    """漂移类型"""
    NONE = "none"
    COVARIATE = "covariate"      # 特征分布漂移
    CONCEPT = "concept"          # 概念漂移 (输入-输出关系变化)
    SUDDEN = "sudden"            # 突然漂移
    GRADUAL = "gradual"          # 渐进漂移
    RECURRING = "recurring"      # 周期性漂移


class FallbackStrategy(Enum):
    """回退策略"""
    TWAP = "twap"               # 时间加权平均
    VWAP = "vwap"               # 成交量加权平均
    PASSIVE = "passive"         # 被动跟随
    ICEBERG = "iceberg"         # 冰山单
    AGGRESSIVE = "aggressive"   # 激进执行 (紧急时)


class DriftSeverity(Enum):
    """漂移严重程度"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DriftDetectorConfigV521:
    """漂移检测器配置"""
    # 检测窗口
    reference_window_size: int = 500    # 参考分布窗口
    test_window_size: int = 50          # 测试分布窗口
    
    # 漂移阈值
    kl_threshold: float = 0.5           # KL散度阈值
    psi_threshold: float = 0.25         # PSI阈值 (> 0.25 = 显著漂移)
    ks_threshold: float = 0.1           # KS检验阈值
    
    # 多维检测
    feature_drift_ratio_threshold: float = 0.3  # 多少比例特征漂移算整体漂移
    
    # 严重程度映射
    low_drift_threshold: float = 0.15
    medium_drift_threshold: float = 0.25
    high_drift_threshold: float = 0.4
    critical_drift_threshold: float = 0.6
    
    # 回退配置
    auto_fallback_on_medium: bool = True
    fallback_cooldown_minutes: int = 30
    recovery_observation_period: int = 100  # 观察多少样本后尝试恢复


@dataclass
class DriftDetectionResult:
    """漂移检测结果"""
    timestamp: datetime
    
    # 整体评估
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    overall_score: float          # 0-1, 综合漂移分数
    
    # 各方法得分
    kl_divergence: float
    psi_score: float
    ks_statistic: float
    
    # 特征级别
    drifted_features: List[str]
    feature_drift_scores: Dict[str, float]
    
    # 建议
    recommended_action: str
    fallback_strategy: Optional[FallbackStrategy]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type.value,
            "severity": self.severity.name,
            "overall_score": round(self.overall_score, 3),
            "kl_divergence": round(self.kl_divergence, 3),
            "psi_score": round(self.psi_score, 3),
            "ks_statistic": round(self.ks_statistic, 3),
            "drifted_features": self.drifted_features,
            "recommended_action": self.recommended_action,
            "fallback_strategy": self.fallback_strategy.value if self.fallback_strategy else None
        }


# =============================================================================
# DRIFT DETECTION METHODS
# =============================================================================

class KLDivergenceDetector:
    """KL散度检测器"""
    
    @staticmethod
    def compute(reference: np.ndarray, test: np.ndarray, bins: int = 50) -> float:
        """计算 KL 散度"""
        # 创建直方图
        min_val = min(reference.min(), test.min())
        max_val = max(reference.max(), test.max())
        
        ref_hist, bin_edges = np.histogram(reference, bins=bins, range=(min_val, max_val), density=True)
        test_hist, _ = np.histogram(test, bins=bins, range=(min_val, max_val), density=True)
        
        # 避免除零
        ref_hist = ref_hist + 1e-10
        test_hist = test_hist + 1e-10
        
        # 归一化
        ref_hist = ref_hist / ref_hist.sum()
        test_hist = test_hist / test_hist.sum()
        
        # KL散度
        kl = np.sum(test_hist * np.log(test_hist / ref_hist))
        
        return max(0, kl)


class PSIDetector:
    """Population Stability Index 检测器"""
    
    @staticmethod
    def compute(reference: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
        """计算 PSI"""
        # 分箱
        min_val = min(reference.min(), test.min())
        max_val = max(reference.max(), test.max())
        
        ref_hist, bin_edges = np.histogram(reference, bins=bins, range=(min_val, max_val))
        test_hist, _ = np.histogram(test, bins=bins, range=(min_val, max_val))
        
        # 转换为比例
        ref_pct = ref_hist / len(reference) + 1e-10
        test_pct = test_hist / len(test) + 1e-10
        
        # PSI = Σ (test% - ref%) * ln(test% / ref%)
        psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        
        return max(0, psi)


class KSTestDetector:
    """Kolmogorov-Smirnov 检验检测器"""
    
    @staticmethod
    def compute(reference: np.ndarray, test: np.ndarray) -> Tuple[float, float]:
        """计算 KS 统计量和 p-value"""
        statistic, pvalue = stats.ks_2samp(reference, test)
        return statistic, pvalue


# =============================================================================
# MULTI-DIMENSIONAL DRIFT DETECTOR
# =============================================================================

class MultiDimensionalDriftDetector:
    """多维漂移检测器"""
    
    def __init__(self, feature_names: List[str], config: DriftDetectorConfigV521 = None):
        self.feature_names = feature_names
        self.config = config or DriftDetectorConfigV521()
        
        # 参考分布 (每个特征)
        self.reference_data: Dict[str, deque] = {
            f: deque(maxlen=self.config.reference_window_size)
            for f in feature_names
        }
        
        # 测试分布
        self.test_data: Dict[str, deque] = {
            f: deque(maxlen=self.config.test_window_size)
            for f in feature_names
        }
        
        # 检测器
        self.kl_detector = KLDivergenceDetector()
        self.psi_detector = PSIDetector()
        self.ks_detector = KSTestDetector()
        
        # 历史
        self.detection_history: deque = deque(maxlen=100)
    
    def update(self, features: Dict[str, float]):
        """更新特征数据"""
        for name, value in features.items():
            if name in self.feature_names:
                # 先添加到参考分布，达到一定量后也添加到测试分布
                self.reference_data[name].append(value)
                if len(self.reference_data[name]) > self.config.reference_window_size * 0.5:
                    self.test_data[name].append(value)
    
    def detect(self) -> DriftDetectionResult:
        """执行漂移检测"""
        feature_scores = {}
        drifted_features = []
        
        kl_scores = []
        psi_scores = []
        ks_scores = []
        
        for feature in self.feature_names:
            ref_data = np.array(self.reference_data[feature])
            test_data = np.array(self.test_data[feature])
            
            if len(ref_data) < 100 or len(test_data) < 20:
                continue
            
            # KL散度
            kl = self.kl_detector.compute(ref_data, test_data)
            kl_scores.append(kl)
            
            # PSI
            psi = self.psi_detector.compute(ref_data, test_data)
            psi_scores.append(psi)
            
            # KS检验
            ks_stat, ks_pval = self.ks_detector.compute(ref_data, test_data)
            ks_scores.append(ks_stat)
            
            # 综合评分
            feature_score = (
                0.4 * min(kl / self.config.kl_threshold, 1.0) +
                0.4 * min(psi / self.config.psi_threshold, 1.0) +
                0.2 * min(ks_stat / self.config.ks_threshold, 1.0)
            )
            
            feature_scores[feature] = feature_score
            
            if feature_score > 0.5:
                drifted_features.append(feature)
        
        # 整体评分
        if not kl_scores:
            overall_score = 0.0
        else:
            overall_kl = np.mean(kl_scores)
            overall_psi = np.mean(psi_scores)
            overall_ks = np.mean(ks_scores)
            
            overall_score = (
                0.4 * min(overall_kl / self.config.kl_threshold, 1.0) +
                0.4 * min(overall_psi / self.config.psi_threshold, 1.0) +
                0.2 * min(overall_ks / self.config.ks_threshold, 1.0)
            )
        
        # 判断漂移
        drift_ratio = len(drifted_features) / len(self.feature_names) if self.feature_names else 0
        drift_detected = drift_ratio > self.config.feature_drift_ratio_threshold
        
        # 严重程度
        severity = self._determine_severity(overall_score)
        
        # 漂移类型
        drift_type = self._determine_drift_type(overall_score, drifted_features)
        
        # 建议动作
        action, fallback = self._recommend_action(severity, drift_type)
        
        result = DriftDetectionResult(
            timestamp=datetime.utcnow(),
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            overall_score=overall_score,
            kl_divergence=np.mean(kl_scores) if kl_scores else 0,
            psi_score=np.mean(psi_scores) if psi_scores else 0,
            ks_statistic=np.mean(ks_scores) if ks_scores else 0,
            drifted_features=drifted_features,
            feature_drift_scores=feature_scores,
            recommended_action=action,
            fallback_strategy=fallback
        )
        
        self.detection_history.append(result)
        
        return result
    
    def _determine_severity(self, score: float) -> DriftSeverity:
        """判断严重程度"""
        if score < self.config.low_drift_threshold:
            return DriftSeverity.NONE
        elif score < self.config.medium_drift_threshold:
            return DriftSeverity.LOW
        elif score < self.config.high_drift_threshold:
            return DriftSeverity.MEDIUM
        elif score < self.config.critical_drift_threshold:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _determine_drift_type(self, score: float, drifted: List[str]) -> DriftType:
        """判断漂移类型"""
        if score < self.config.low_drift_threshold:
            return DriftType.NONE
        
        # 查看历史趋势
        recent_scores = [r.overall_score for r in list(self.detection_history)[-10:]]
        
        if len(recent_scores) < 3:
            return DriftType.COVARIATE
        
        # 检测突然漂移
        if recent_scores[-1] - recent_scores[-2] > 0.2:
            return DriftType.SUDDEN
        
        # 检测渐进漂移
        if all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            return DriftType.GRADUAL
        
        # 检测周期性
        if len(recent_scores) > 5:
            diffs = np.diff(recent_scores)
            if np.sum(np.abs(np.diff(np.sign(diffs)))) > 4:
                return DriftType.RECURRING
        
        return DriftType.COVARIATE
    
    def _recommend_action(self, severity: DriftSeverity, 
                          drift_type: DriftType) -> Tuple[str, Optional[FallbackStrategy]]:
        """推荐动作"""
        if severity == DriftSeverity.NONE:
            return "continue", None
        
        if severity == DriftSeverity.LOW:
            return "monitor", None
        
        if severity == DriftSeverity.MEDIUM:
            if self.config.auto_fallback_on_medium:
                return "fallback", FallbackStrategy.VWAP
            return "alert", None
        
        if severity == DriftSeverity.HIGH:
            return "fallback", FallbackStrategy.TWAP
        
        if severity == DriftSeverity.CRITICAL:
            if drift_type == DriftType.SUDDEN:
                return "emergency_fallback", FallbackStrategy.PASSIVE
            return "halt", FallbackStrategy.PASSIVE
        
        return "continue", None


# =============================================================================
# RL EXECUTION FALLBACK MANAGER
# =============================================================================

class RLExecutionFallbackManager:
    """
    RL 执行回退管理器
    
    管理:
    1. 漂移检测
    2. 自动回退
    3. 渐进恢复
    4. 状态追踪
    """
    
    def __init__(self, 
                 feature_names: List[str],
                 config: DriftDetectorConfigV521 = None):
        self.config = config or DriftDetectorConfigV521()
        
        # 漂移检测器
        self.drift_detector = MultiDimensionalDriftDetector(feature_names, self.config)
        
        # 状态
        self.is_in_fallback = False
        self.current_fallback_strategy: Optional[FallbackStrategy] = None
        self.fallback_start_time: Optional[datetime] = None
        self.recovery_observations = 0
        
        # 历史
        self.fallback_history: List[Dict] = []
        
        # 回退策略实现
        self._fallback_executors: Dict[FallbackStrategy, Callable] = {}
        
        # 原始 RL 模型
        self._rl_model = None
        
        # EventBus
        self._event_bus = None
        
        logger.info("RLExecutionFallbackManager initialized")
    
    def _get_event_bus(self):
        """延迟获取 EventBus"""
        if self._event_bus is None:
            try:
                from infra.event_bus import get_event_bus
                self._event_bus = get_event_bus()
            except ImportError:
                pass
        return self._event_bus
    
    def set_rl_model(self, model: Any):
        """设置 RL 模型"""
        self._rl_model = model
    
    def register_fallback_executor(self, strategy: FallbackStrategy, executor: Callable):
        """注册回退策略执行器"""
        self._fallback_executors[strategy] = executor
    
    def update_features(self, features: Dict[str, float]):
        """更新特征数据"""
        self.drift_detector.update(features)
        
        # 如果在回退中，计数恢复观察
        if self.is_in_fallback:
            self.recovery_observations += 1
    
    def check_and_decide(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        检查漂移并决定执行方式
        
        Returns:
            {
                "use_rl": bool,
                "fallback_strategy": Optional[FallbackStrategy],
                "drift_result": DriftDetectionResult,
                "reason": str
            }
        """
        # 更新特征
        self.update_features(features)
        
        # 检测漂移
        drift_result = self.drift_detector.detect()
        
        # 当前在回退中
        if self.is_in_fallback:
            # 检查是否可以恢复
            if self._can_recover(drift_result):
                self._exit_fallback()
                return {
                    "use_rl": True,
                    "fallback_strategy": None,
                    "drift_result": drift_result,
                    "reason": "Recovery: drift normalized"
                }
            else:
                return {
                    "use_rl": False,
                    "fallback_strategy": self.current_fallback_strategy,
                    "drift_result": drift_result,
                    "reason": f"In fallback: {drift_result.severity.name} drift"
                }
        
        # 正常运行，检查是否需要回退
        if drift_result.fallback_strategy:
            self._enter_fallback(drift_result.fallback_strategy, drift_result)
            
            # 发送事件
            self._emit_fallback_event(drift_result)
            
            return {
                "use_rl": False,
                "fallback_strategy": drift_result.fallback_strategy,
                "drift_result": drift_result,
                "reason": f"Fallback triggered: {drift_result.drift_type.value} drift, "
                          f"severity={drift_result.severity.name}"
            }
        
        # 正常使用 RL
        return {
            "use_rl": True,
            "fallback_strategy": None,
            "drift_result": drift_result,
            "reason": "Normal operation"
        }
    
    def _can_recover(self, drift_result: DriftDetectionResult) -> bool:
        """判断是否可以从回退中恢复"""
        # 检查冷却时间
        if self.fallback_start_time:
            elapsed = (datetime.utcnow() - self.fallback_start_time).total_seconds() / 60
            if elapsed < self.config.fallback_cooldown_minutes:
                return False
        
        # 检查观察期
        if self.recovery_observations < self.config.recovery_observation_period:
            return False
        
        # 检查漂移是否已恢复
        if drift_result.severity.value >= DriftSeverity.MEDIUM.value:
            return False
        
        return True
    
    def _enter_fallback(self, strategy: FallbackStrategy, drift_result: DriftDetectionResult):
        """进入回退模式"""
        self.is_in_fallback = True
        self.current_fallback_strategy = strategy
        self.fallback_start_time = datetime.utcnow()
        self.recovery_observations = 0
        
        self.fallback_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": strategy.value,
            "reason": drift_result.recommended_action,
            "severity": drift_result.severity.name,
            "drift_score": drift_result.overall_score
        })
        
        logger.warning(f"Entering fallback mode: {strategy.value} "
                       f"(severity={drift_result.severity.name}, "
                       f"score={drift_result.overall_score:.3f})")
    
    def _exit_fallback(self):
        """退出回退模式"""
        logger.info(f"Exiting fallback mode after "
                    f"{self.recovery_observations} observations")
        
        self.is_in_fallback = False
        self.current_fallback_strategy = None
        self.fallback_start_time = None
        self.recovery_observations = 0
    
    def _emit_fallback_event(self, drift_result: DriftDetectionResult):
        """发送回退事件"""
        bus = self._get_event_bus()
        if bus:
            try:
                from infra.event_bus import Event, EventType, EventPriority
                event = Event(
                    event_type=EventType.RISK_ALERT,
                    source="rl_fallback_manager",
                    priority=EventPriority.HIGH,
                    payload={
                        "sub_type": "RL_FALLBACK_TRIGGERED",
                        "fallback_strategy": self.current_fallback_strategy.value,
                        "drift_severity": drift_result.severity.name,
                        "drift_score": drift_result.overall_score,
                        "drifted_features": drift_result.drifted_features
                    }
                )
                bus.publish(event)
            except Exception as e:
                logger.error(f"Failed to emit event: {e}")
    
    def execute(self, 
                symbol: str,
                side: str,
                quantity: float,
                features: Dict[str, float],
                **kwargs) -> Dict[str, Any]:
        """
        执行交易 (自动选择 RL 或回退)
        
        Args:
            symbol: 交易对
            side: "buy" or "sell"
            quantity: 数量
            features: 当前特征
            **kwargs: 其他参数
            
        Returns:
            执行结果
        """
        # 检查并决定
        decision = self.check_and_decide(features)
        
        if decision["use_rl"]:
            # 使用 RL 模型
            if self._rl_model:
                return self._execute_with_rl(symbol, side, quantity, features, **kwargs)
            else:
                logger.warning("RL model not set, using default fallback")
                decision["fallback_strategy"] = FallbackStrategy.TWAP
        
        # 使用回退策略
        strategy = decision["fallback_strategy"] or FallbackStrategy.TWAP
        executor = self._fallback_executors.get(strategy)
        
        if executor:
            return executor(symbol, side, quantity, **kwargs)
        else:
            return self._default_fallback_execute(symbol, side, quantity, strategy, **kwargs)
    
    def _execute_with_rl(self, symbol: str, side: str, quantity: float,
                         features: Dict[str, float], **kwargs) -> Dict:
        """使用 RL 执行"""
        # 这里应该调用实际的 RL 模型
        # 示例返回
        return {
            "method": "rl",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "simulated"
        }
    
    def _default_fallback_execute(self, symbol: str, side: str, quantity: float,
                                   strategy: FallbackStrategy, **kwargs) -> Dict:
        """默认回退执行"""
        logger.info(f"Executing fallback: {strategy.value} for {side} {quantity} {symbol}")
        
        return {
            "method": "fallback",
            "strategy": strategy.value,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "simulated"
        }
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "is_in_fallback": self.is_in_fallback,
            "current_strategy": self.current_fallback_strategy.value if self.current_fallback_strategy else None,
            "fallback_start_time": self.fallback_start_time.isoformat() if self.fallback_start_time else None,
            "recovery_observations": self.recovery_observations,
            "fallback_count": len(self.fallback_history),
            "recent_fallbacks": self.fallback_history[-5:] if self.fallback_history else []
        }
    
    def get_drift_stats(self) -> Dict:
        """获取漂移统计"""
        history = list(self.drift_detector.detection_history)
        
        if not history:
            return {"samples": 0}
        
        scores = [r.overall_score for r in history]
        severities = [r.severity.name for r in history]
        
        return {
            "samples": len(history),
            "mean_drift_score": np.mean(scores),
            "max_drift_score": np.max(scores),
            "recent_severity": severities[-1] if severities else "NONE",
            "severity_distribution": {
                s.name: severities.count(s.name) for s in DriftSeverity
            }
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_rl_fallback_manager: Optional[RLExecutionFallbackManager] = None


def get_rl_fallback_manager(feature_names: List[str] = None) -> RLExecutionFallbackManager:
    """获取单例"""
    global _rl_fallback_manager
    if _rl_fallback_manager is None:
        features = feature_names or [
            "price_change", "volume_ratio", "spread", "volatility",
            "order_imbalance", "momentum", "rsi", "correlation"
        ]
        _rl_fallback_manager = RLExecutionFallbackManager(features)
    return _rl_fallback_manager


def reset_rl_fallback_manager():
    """重置单例"""
    global _rl_fallback_manager
    _rl_fallback_manager = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DriftType',
    'FallbackStrategy',
    'DriftSeverity',
    'DriftDetectorConfigV521',
    'DriftDetectionResult',
    'KLDivergenceDetector',
    'PSIDetector',
    'KSTestDetector',
    'MultiDimensionalDriftDetector',
    'RLExecutionFallbackManager',
    'get_rl_fallback_manager',
    'reset_rl_fallback_manager'
]


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    # 初始化
    feature_names = ["price_change", "volume_ratio", "spread", "volatility"]
    manager = RLExecutionFallbackManager(feature_names)
    
    print("=== Simulating Normal Operation ===")
    
    # 模拟正常数据
    for i in range(200):
        features = {
            "price_change": random.gauss(0, 0.01),
            "volume_ratio": random.uniform(0.8, 1.2),
            "spread": random.uniform(0.0005, 0.001),
            "volatility": random.uniform(0.02, 0.04)
        }
        manager.update_features(features)
    
    # 检查
    decision = manager.check_and_decide(features)
    print(f"Decision: use_rl={decision['use_rl']}, reason={decision['reason']}")
    
    print("\n=== Simulating Drift ===")
    
    # 模拟漂移
    for i in range(100):
        features = {
            "price_change": random.gauss(0.02, 0.02),  # 均值偏移
            "volume_ratio": random.uniform(0.5, 0.7),   # 分布变化
            "spread": random.uniform(0.002, 0.005),     # 显著变化
            "volatility": random.uniform(0.05, 0.08)    # 波动率上升
        }
        manager.update_features(features)
        
        if i % 20 == 0:
            decision = manager.check_and_decide(features)
            print(f"[{i}] Decision: use_rl={decision['use_rl']}, "
                  f"severity={decision['drift_result'].severity.name}, "
                  f"score={decision['drift_result'].overall_score:.3f}")
    
    print("\n=== Status ===")
    print(manager.get_status())
    
    print("\n=== Drift Stats ===")
    print(manager.get_drift_stats())

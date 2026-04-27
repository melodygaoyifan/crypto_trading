"""
================================================================================
HMATS v5.2.1 P0 - 数据健康度与降级机制
Data Health Monitoring and Degradation
================================================================================

负责监控所有数据源的健康状态，并在数据异常时触发降级机制。

功能:
1. Staleness 检测 - 数据过期监控
2. Missing Rate 检测 - 数据缺失率监控
3. Schema Validation - 数据格式验证
4. Spike/Anomaly Flag - 数据异常标记
5. 降级决策输出

输出: DataHealthSnapshot - 统一健康状态对象

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class DataSourceType(Enum):
    """数据源类型"""
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    LOB = "lob"
    PRICE = "price"
    # [FIX-4] P0 feed independent monitoring
    COINGLASS = "coinglass"
    FRED = "fred"
    LUNARCRUSH = "lunarcrush"
    CRYPTOPANIC = "cryptopanic"
    KRAKEN_FUTURES = "kraken_futures"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    MISSING = "missing"
    ANOMALY = "anomaly"
    CRITICAL = "critical"


class DegradationLevel(Enum):
    """降级级别"""
    NONE = "none"              # 正常运行
    CAUTION = "caution"        # 谨慎运行
    REDUCED = "reduced"        # 降低暴露
    EXIT_ONLY = "exit_only"    # 仅允许平仓
    NO_TRADE = "no_trade"      # 禁止交易


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SourceHealth:
    """单个数据源健康状态"""
    source_type: DataSourceType
    status: HealthStatus
    
    # 时效性
    last_update: Optional[datetime]
    staleness_sec: float
    is_stale: bool
    
    # 可用性
    available: bool
    missing_rate: float  # 最近N次采集的缺失率
    
    # 置信度
    confidence: float
    
    # 异常
    has_anomaly: bool
    anomaly_details: str = ""
    
    # 错误
    error_count: int = 0
    last_error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "status": self.status.value,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "staleness_sec": self.staleness_sec,
            "is_stale": self.is_stale,
            "available": self.available,
            "missing_rate": self.missing_rate,
            "confidence": self.confidence,
            "has_anomaly": self.has_anomaly,
            "anomaly_details": self.anomaly_details,
            "error_count": self.error_count,
        }


@dataclass
class DataHealthSnapshot:
    """
    数据健康快照
    
    汇总所有数据源的健康状态，用于决策降级
    """
    timestamp: datetime
    
    # 各数据源健康状态
    source_health: Dict[DataSourceType, SourceHealth] = field(default_factory=dict)
    
    # 整体状态
    overall_status: HealthStatus = HealthStatus.HEALTHY
    degradation_level: DegradationLevel = DegradationLevel.NONE
    degradation_reason: str = ""
    
    # 统计
    healthy_sources: int = 0
    degraded_sources: int = 0
    missing_sources: int = 0
    
    # 关键源状态
    critical_sources_ok: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "degradation_level": self.degradation_level.value,
            "degradation_reason": self.degradation_reason,
            "healthy_sources": self.healthy_sources,
            "degraded_sources": self.degraded_sources,
            "missing_sources": self.missing_sources,
            "critical_sources_ok": self.critical_sources_ok,
            "sources": {
                k.value: v.to_dict() for k, v in self.source_health.items()
            },
        }
    
    def is_trade_allowed(self) -> bool:
        """是否允许交易"""
        return self.degradation_level not in [
            DegradationLevel.EXIT_ONLY,
            DegradationLevel.NO_TRADE,
        ]
    
    def is_entry_allowed(self) -> bool:
        """是否允许新开仓"""
        return self.degradation_level in [
            DegradationLevel.NONE,
            DegradationLevel.CAUTION,
        ]


# =============================================================================
# DATA HEALTH MONITOR
# =============================================================================

class DataHealthMonitor:
    """
    数据健康监控器
    
    监控所有数据源的健康状态，输出统一的 DataHealthSnapshot
    """
    
    def __init__(self):
        # 数据源配置
        self._source_configs: Dict[DataSourceType, Dict[str, Any]] = {
            DataSourceType.ONCHAIN: {
                "max_staleness_sec": 300,
                "critical": False,
                "weight": 0.15,
            },
            DataSourceType.SENTIMENT: {
                "max_staleness_sec": 300,
                "critical": False,
                "weight": 0.15,
            },
            DataSourceType.MACRO: {
                "max_staleness_sec": 600,
                "critical": False,
                "weight": 0.1,
            },
            DataSourceType.LOB: {
                "max_staleness_sec": 10,
                "critical": True,  # 订单簿是关键数据
                "weight": 0.3,
            },
            DataSourceType.PRICE: {
                "max_staleness_sec": 5,
                "critical": True,  # 价格是关键数据
                "weight": 0.3,
            },
        }
        
        # 历史记录（用于计算缺失率）
        self._history: Dict[DataSourceType, deque] = {
            st: deque(maxlen=100) for st in DataSourceType
        }
        
        # 当前状态
        self._current_health: Dict[DataSourceType, SourceHealth] = {}
        
        # 异常检测阈值
        self._anomaly_thresholds = {
            "spike_multiplier": 3.0,  # 超过历史均值3倍视为异常
            "min_confidence": 0.5,    # 置信度低于0.5视为异常
        }
        
        # 降级规则
        self._degradation_rules = {
            "exit_only_threshold": 2,   # 2个关键源异常 -> exit_only
            "no_trade_threshold": 1,    # 1个关键源缺失 -> no_trade
            "reduced_threshold": 0.5,   # 50%源异常 -> reduced
            "caution_threshold": 0.3,   # 30%源异常 -> caution
        }
        
        logger.info("DataHealthMonitor initialized")
    
    def update_source(
        self,
        source_type: DataSourceType,
        available: bool,
        staleness_sec: float = 0.0,
        confidence: float = 1.0,
        data: Optional[Dict] = None,
    ) -> SourceHealth:
        """
        更新数据源状态
        
        Args:
            source_type: 数据源类型
            available: 数据是否可用
            staleness_sec: 数据延迟（秒）
            confidence: 数据置信度
            data: 原始数据（用于异常检测）
        
        Returns:
            SourceHealth: 更新后的健康状态
        """
        config = self._source_configs.get(source_type, {})
        max_staleness = config.get("max_staleness_sec", 60)
        
        # 记录历史
        self._history[source_type].append({
            "timestamp": datetime.now(timezone.utc),
            "available": available,
            "staleness_sec": staleness_sec,
            "confidence": confidence,
        })
        
        # 计算缺失率
        history = list(self._history[source_type])
        missing_rate = sum(1 for h in history if not h["available"]) / len(history) if history else 0
        
        # 检测异常
        has_anomaly, anomaly_details = self._detect_anomaly(source_type, data)
        
        # 确定状态
        is_stale = staleness_sec > max_staleness
        
        if not available:
            status = HealthStatus.MISSING
        elif has_anomaly:
            status = HealthStatus.ANOMALY
        elif is_stale:
            status = HealthStatus.STALE
        elif confidence < 0.8 or missing_rate > 0.2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        health = SourceHealth(
            source_type=source_type,
            status=status,
            last_update=datetime.now(timezone.utc) if available else None,
            staleness_sec=staleness_sec,
            is_stale=is_stale,
            available=available,
            missing_rate=missing_rate,
            confidence=confidence,
            has_anomaly=has_anomaly,
            anomaly_details=anomaly_details,
        )
        
        self._current_health[source_type] = health
        
        return health
    
    def get_snapshot(self) -> DataHealthSnapshot:
        """
        获取数据健康快照
        
        Returns:
            DataHealthSnapshot: 当前健康状态快照
        """
        now = datetime.now(timezone.utc)
        
        # 统计
        healthy_count = 0
        degraded_count = 0
        missing_count = 0
        critical_issues = []
        
        for source_type, health in self._current_health.items():
            config = self._source_configs.get(source_type, {})
            is_critical = config.get("critical", False)
            
            if health.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif health.status in [HealthStatus.DEGRADED, HealthStatus.STALE, HealthStatus.ANOMALY]:
                degraded_count += 1
                if is_critical:
                    critical_issues.append(source_type)
            else:  # MISSING or CRITICAL
                missing_count += 1
                if is_critical:
                    critical_issues.append(source_type)
        
        # 检查未更新的源
        for source_type in DataSourceType:
            if source_type not in self._current_health:
                missing_count += 1
                config = self._source_configs.get(source_type, {})
                if config.get("critical", False):
                    critical_issues.append(source_type)
        
        # 确定降级级别
        total_sources = len(DataSourceType)
        degradation_level, degradation_reason = self._determine_degradation(
            healthy_count, degraded_count, missing_count,
            critical_issues, total_sources
        )
        
        # 确定整体状态
        if degradation_level == DegradationLevel.NO_TRADE:
            overall_status = HealthStatus.CRITICAL
        elif degradation_level == DegradationLevel.EXIT_ONLY:
            overall_status = HealthStatus.CRITICAL
        elif degradation_level == DegradationLevel.REDUCED:
            overall_status = HealthStatus.DEGRADED
        elif degradation_level == DegradationLevel.CAUTION:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return DataHealthSnapshot(
            timestamp=now,
            source_health=dict(self._current_health),
            overall_status=overall_status,
            degradation_level=degradation_level,
            degradation_reason=degradation_reason,
            healthy_sources=healthy_count,
            degraded_sources=degraded_count,
            missing_sources=missing_count,
            critical_sources_ok=len(critical_issues) == 0,
        )
    
    def _detect_anomaly(
        self,
        source_type: DataSourceType,
        data: Optional[Dict],
    ) -> tuple:
        """检测数据异常"""
        if data is None:
            return False, ""
        
        anomalies = []
        
        # 检查置信度
        confidence = data.get("confidence", 1.0)
        if confidence < self._anomaly_thresholds["min_confidence"]:
            anomalies.append(f"low_confidence:{confidence:.2f}")
        
        # 基于历史数据的 spike 检测
        history = list(self._history[source_type])
        if len(history) >= 5:
            # 检查价格 spike
            if "price" in data:
                prices = [h.get("data", {}).get("price") for h in history if h.get("data", {}).get("price")]
                if len(prices) >= 3:
                    import numpy as np
                    mean_price = np.mean(prices[-10:])
                    std_price = np.std(prices[-10:]) + 1e-10
                    current_price = data["price"]
                    zscore = abs(current_price - mean_price) / std_price
                    if zscore > self._anomaly_thresholds.get("max_price_zscore", 5.0):
                        anomalies.append(f"price_spike:zscore={zscore:.1f}")
            
            # 检查 volume spike
            if "volume" in data:
                volumes = [h.get("data", {}).get("volume") for h in history if h.get("data", {}).get("volume")]
                if len(volumes) >= 3:
                    import numpy as np
                    mean_vol = np.mean(volumes[-10:])
                    std_vol = np.std(volumes[-10:]) + 1e-10
                    current_vol = data["volume"]
                    if mean_vol > 0:
                        vol_ratio = current_vol / mean_vol
                        if vol_ratio > self._anomaly_thresholds.get("max_volume_ratio", 10.0):
                            anomalies.append(f"volume_spike:ratio={vol_ratio:.1f}")
        
        return len(anomalies) > 0, "; ".join(anomalies)
    
    def _determine_degradation(
        self,
        healthy: int,
        degraded: int,
        missing: int,
        critical_issues: List[DataSourceType],
        total: int,
    ) -> tuple:
        """确定降级级别"""
        rules = self._degradation_rules
        
        # 关键源缺失 -> NO_TRADE
        critical_missing = sum(
            1 for s in critical_issues
            if s in self._current_health and self._current_health[s].status == HealthStatus.MISSING
        )
        if critical_missing >= rules["no_trade_threshold"]:
            return (
                DegradationLevel.NO_TRADE,
                f"Critical source missing: {[s.value for s in critical_issues]}"
            )
        
        # 关键源异常 -> EXIT_ONLY
        if len(critical_issues) >= rules["exit_only_threshold"]:
            return (
                DegradationLevel.EXIT_ONLY,
                f"Critical source issues: {[s.value for s in critical_issues]}"
            )
        
        # 多数源异常 -> REDUCED
        abnormal_rate = (degraded + missing) / total if total > 0 else 0
        if abnormal_rate >= rules["reduced_threshold"]:
            return (
                DegradationLevel.REDUCED,
                f"High abnormal rate: {abnormal_rate:.1%}"
            )
        
        # 部分源异常 -> CAUTION
        if abnormal_rate >= rules["caution_threshold"]:
            return (
                DegradationLevel.CAUTION,
                f"Elevated abnormal rate: {abnormal_rate:.1%}"
            )
        
        return DegradationLevel.NONE, ""
    
    def get_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        snapshot = self.get_snapshot()
        return {
            "overall_status": snapshot.overall_status.value,
            "degradation_level": snapshot.degradation_level.value,
            "degradation_reason": snapshot.degradation_reason,
            "trade_allowed": snapshot.is_trade_allowed(),
            "entry_allowed": snapshot.is_entry_allowed(),
            "sources": {
                st.value: {
                    "status": self._current_health[st].status.value if st in self._current_health else "missing",
                    "staleness_sec": self._current_health[st].staleness_sec if st in self._current_health else float('inf'),
                }
                for st in DataSourceType
            },
        }


# =============================================================================
# INTEGRATION WITH FEEDS
# =============================================================================

class DataHealthIntegration:
    """
    数据健康与 Feeds 集成
    
    自动从各个 feed 收集健康状态
    """
    
    def __init__(self):
        self._monitor = DataHealthMonitor()
        self._feeds_registered = False
    
    def register_feeds(
        self,
        onchain_feed=None,
        sentiment_feed=None,
        macro_feed=None,
        lob_feed=None,
    ):
        """注册数据 feeds"""
        self._onchain_feed = onchain_feed
        self._sentiment_feed = sentiment_feed
        self._macro_feed = macro_feed
        self._lob_feed = lob_feed
        self._feeds_registered = True
    
    def update_from_feeds(self) -> DataHealthSnapshot:
        """从所有 feeds 更新健康状态"""
        
        # OnChain
        if self._onchain_feed:
            health = self._onchain_feed.get_health()
            self._monitor.update_source(
                DataSourceType.ONCHAIN,
                available=not health.get("is_stale", True),
                staleness_sec=health.get("staleness_sec", float('inf')),
                confidence=health.get("confidence", 0.0),
            )
        
        # Sentiment
        if self._sentiment_feed:
            health = self._sentiment_feed.get_health()
            self._monitor.update_source(
                DataSourceType.SENTIMENT,
                available=not health.get("is_stale", True),
                staleness_sec=health.get("staleness_sec", float('inf')),
                confidence=health.get("confidence", 0.0),
            )
        
        # Macro
        if self._macro_feed:
            health = self._macro_feed.get_health()
            self._monitor.update_source(
                DataSourceType.MACRO,
                available=not health.get("is_stale", True),
                staleness_sec=health.get("staleness_sec", float('inf')),
                confidence=health.get("confidence", 0.0),
            )
        
        # LOB
        if self._lob_feed:
            health = self._lob_feed.get_health()
            self._monitor.update_source(
                DataSourceType.LOB,
                available=not health.get("is_stale", True),
                staleness_sec=health.get("staleness_sec", float('inf')),
                confidence=health.get("confidence", 0.0),
            )
        
        return self._monitor.get_snapshot()
    
    def get_snapshot(self) -> DataHealthSnapshot:
        """获取当前健康快照"""
        return self._monitor.get_snapshot()
    
    def get_monitor(self) -> DataHealthMonitor:
        """获取底层监控器"""
        return self._monitor


# =============================================================================
# SINGLETON
# =============================================================================

_data_health_instance: Optional[DataHealthIntegration] = None


def get_data_health() -> DataHealthIntegration:
    """获取数据健康单例"""
    global _data_health_instance
    if _data_health_instance is None:
        _data_health_instance = DataHealthIntegration()
    return _data_health_instance


def reset_data_health():
    """重置单例"""
    global _data_health_instance
    _data_health_instance = None

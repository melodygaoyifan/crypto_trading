"""
================================================================================
HMATS v5.2.1 P1 - 冲击模型校准
Impact Calibration
================================================================================

Purpose: 离线 + 在线校准市场冲击模型参数

P1-B 核心要求:
1. 按 bucket 统计并输出 ImpactCalibrationTable
2. production_market_impact.py 必须优先使用校准参数
3. 支持历史数据离线校准
4. 支持实时执行数据在线微调

校准维度:
- asset
- UTC session (asia_morning, europe_morning, us_morning, etc.)
- volatility bucket (low, medium, high, extreme)
- spread bucket (tight, normal, wide, very_wide)

================================================================================
"""

import logging
import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImpactCalibrationEntry:
    """
    单个 bucket 的冲击模型校准参数
    """
    # Bucket 标识
    asset: str
    session: str                       # UTC session
    volatility_bucket: str
    spread_bucket: str
    
    # Almgren-Chriss 参数
    eta: float                         # 临时冲击系数
    gamma: float                       # 永久冲击系数
    
    # 置信度
    sample_count: int
    confidence: float                  # 基于样本数和一致性
    
    # 统计
    avg_realized_impact_bps: float
    std_realized_impact_bps: float
    avg_prediction_error_bps: float
    
    # 时间戳
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_bucket_key(self) -> str:
        """获取 bucket 键"""
        return f"{self.asset}:{self.session}:{self.volatility_bucket}:{self.spread_bucket}"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["last_updated"] = self.last_updated.isoformat()
        return d
    
    @staticmethod
    def from_dict(d: Dict) -> 'ImpactCalibrationEntry':
        d = d.copy()
        d["last_updated"] = datetime.fromisoformat(d["last_updated"])
        return ImpactCalibrationEntry(**d)


@dataclass
class CalibrationSample:
    """校准样本"""
    timestamp: datetime
    asset: str
    side: str
    size_usd: float
    expected_impact_bps: float
    realized_impact_bps: float
    
    # 环境
    volatility: float
    spread_bps: float
    session: str
    
    # 分桶
    volatility_bucket: str = ""
    spread_bucket: str = ""
    
    def __post_init__(self):
        """自动计算分桶"""
        if not self.volatility_bucket:
            if self.volatility < 0.3:
                self.volatility_bucket = "low"
            elif self.volatility < 0.6:
                self.volatility_bucket = "medium"
            elif self.volatility < 1.0:
                self.volatility_bucket = "high"
            else:
                self.volatility_bucket = "extreme"
        
        if not self.spread_bucket:
            if self.spread_bps < 5:
                self.spread_bucket = "tight"
            elif self.spread_bps < 15:
                self.spread_bucket = "normal"
            elif self.spread_bps < 30:
                self.spread_bucket = "wide"
            else:
                self.spread_bucket = "very_wide"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ImpactCalibrationConfig:
    """冲击模型校准配置"""
    
    # 默认参数
    default_eta: float = 0.1           # 默认临时冲击系数
    default_gamma: float = 0.1         # 默认永久冲击系数
    
    # 校准条件
    min_samples_for_calibration: int = 20    # 最少样本数
    min_samples_for_high_confidence: int = 100  # 高置信度样本数
    
    # 学习率
    offline_learning_rate: float = 0.5       # 离线校准学习率
    online_learning_rate: float = 0.05       # 在线微调学习率
    
    # 参数边界
    eta_min: float = 0.01
    eta_max: float = 1.0
    gamma_min: float = 0.01
    gamma_max: float = 1.0
    
    # 存储
    calibration_file: str = "data/impact_calibration.json"
    
    # 自动校准
    auto_calibrate_interval_minutes: int = 60


# =============================================================================
# IMPACT CALIBRATION TABLE
# =============================================================================

class ImpactCalibrationTable:
    """
    冲击模型校准表
    
    存储所有 bucket 的校准参数
    供 production_market_impact 查询使用
    """
    
    def __init__(self, config: ImpactCalibrationConfig = None):
        self.config = config or ImpactCalibrationConfig()
        
        # 校准条目
        self._entries: Dict[str, ImpactCalibrationEntry] = {}
        
        # 样本缓存 (用于在线校准)
        self._samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )
        
        # 加载已有校准
        self._load_calibration()
        
        logger.info(f"ImpactCalibrationTable initialized with {len(self._entries)} entries")
    
    def _load_calibration(self):
        """加载校准文件"""
        if not os.path.exists(self.config.calibration_file):
            return
        
        try:
            with open(self.config.calibration_file, "r") as f:
                data = json.load(f)
            
            for bucket_key, entry_dict in data.get("entries", {}).items():
                self._entries[bucket_key] = ImpactCalibrationEntry.from_dict(entry_dict)
            
            logger.info(f"Loaded {len(self._entries)} calibration entries")
            
        except Exception as e:
            logger.warning(f"Failed to load calibration file: {e}")
    
    def save(self):
        """保存校准文件"""
        try:
            os.makedirs(os.path.dirname(self.config.calibration_file), exist_ok=True)
            
            data = {
                "version": "5.2.1",
                "updated_at": datetime.now().isoformat(),
                "entries": {
                    k: v.to_dict() for k, v in self._entries.items()
                }
            }
            
            # [P37 2026-04-24] Was open(w)+json.dump → corrupt on crash.
            from core.state_persistence import save_state
            if save_state(self.config.calibration_file, data):
                logger.info(f"Saved {len(self._entries)} calibration entries")
            else:
                logger.error(f"Failed to save calibration file")

        except Exception as e:
            logger.error(f"Failed to save calibration file: {e}")
    
    # =========================================================================
    # LOOKUP API (供 production_market_impact 使用)
    # =========================================================================
    
    def get_params(
        self,
        asset: str,
        session: str,
        volatility_bucket: str,
        spread_bucket: str,
    ) -> Tuple[float, float, float]:
        """
        获取校准参数
        
        Returns:
            (eta, gamma, confidence)
        """
        bucket_key = f"{asset}:{session}:{volatility_bucket}:{spread_bucket}"
        
        if bucket_key in self._entries:
            entry = self._entries[bucket_key]
            return entry.eta, entry.gamma, entry.confidence
        
        # 尝试降级查找 (忽略 spread)
        fallback_key = f"{asset}:{session}:{volatility_bucket}:*"
        for key, entry in self._entries.items():
            if key.startswith(f"{asset}:{session}:{volatility_bucket}:"):
                return entry.eta, entry.gamma, entry.confidence * 0.8
        
        # 尝试更宽松的降级 (仅 asset)
        for key, entry in self._entries.items():
            if key.startswith(f"{asset}:"):
                return entry.eta, entry.gamma, entry.confidence * 0.5
        
        # 返回默认值
        return self.config.default_eta, self.config.default_gamma, 0.0
    
    def get_entry(self, bucket_key: str) -> Optional[ImpactCalibrationEntry]:
        """获取校准条目"""
        return self._entries.get(bucket_key)
    
    def get_all_entries(self) -> Dict[str, ImpactCalibrationEntry]:
        """获取所有条目"""
        return self._entries.copy()
    
    # =========================================================================
    # CALIBRATION (离线)
    # =========================================================================
    
    def calibrate_offline(self, samples: List[CalibrationSample]):
        """
        离线校准
        
        使用历史执行数据批量校准所有 bucket
        """
        if not samples:
            logger.warning("No samples for offline calibration")
            return
        
        # 按 bucket 分组
        bucket_samples: Dict[str, List[CalibrationSample]] = defaultdict(list)
        
        for sample in samples:
            bucket_key = f"{sample.asset}:{sample.session}:{sample.volatility_bucket}:{sample.spread_bucket}"
            bucket_samples[bucket_key].append(sample)
        
        # 校准每个 bucket
        calibrated_count = 0
        
        for bucket_key, bucket_data in bucket_samples.items():
            if len(bucket_data) < self.config.min_samples_for_calibration:
                continue
            
            entry = self._calibrate_bucket(bucket_key, bucket_data)
            if entry:
                self._entries[bucket_key] = entry
                calibrated_count += 1
        
        logger.info(f"Offline calibration complete: {calibrated_count} buckets calibrated")
        
        # 保存
        self.save()
    
    def _calibrate_bucket(
        self,
        bucket_key: str,
        samples: List[CalibrationSample],
    ) -> Optional[ImpactCalibrationEntry]:
        """校准单个 bucket"""
        if not samples:
            return None
        
        # 解析 bucket key
        parts = bucket_key.split(":")
        if len(parts) != 4:
            return None
        
        asset, session, vol_bucket, spread_bucket = parts
        
        # 计算统计
        realized_impacts = [s.realized_impact_bps for s in samples]
        expected_impacts = [s.expected_impact_bps for s in samples]
        prediction_errors = [r - e for r, e in zip(realized_impacts, expected_impacts)]
        
        avg_realized = np.mean(realized_impacts)
        std_realized = np.std(realized_impacts)
        avg_error = np.mean(prediction_errors)
        
        # 校准 eta 和 gamma
        # 基于误差调整参数
        current_entry = self._entries.get(bucket_key)
        
        if current_entry:
            eta = current_entry.eta
            gamma = current_entry.gamma
        else:
            eta = self.config.default_eta
            gamma = self.config.default_gamma
        
        # 如果系统性低估 (avg_error > 0)，增加参数
        # 如果系统性高估 (avg_error < 0)，减少参数
        adjustment = avg_error / 100  # 转换为调整比例
        adjustment = np.clip(adjustment, -0.3, 0.3)  # 限制调整幅度
        
        eta *= (1 + adjustment * self.config.offline_learning_rate)
        gamma *= (1 + adjustment * self.config.offline_learning_rate)
        
        # 约束参数范围
        eta = np.clip(eta, self.config.eta_min, self.config.eta_max)
        gamma = np.clip(gamma, self.config.gamma_min, self.config.gamma_max)
        
        # 计算置信度
        sample_count = len(samples)
        if sample_count >= self.config.min_samples_for_high_confidence:
            confidence = 0.9
        elif sample_count >= self.config.min_samples_for_calibration * 2:
            confidence = 0.7
        else:
            confidence = 0.5
        
        # 一致性降低置信度
        cv = std_realized / abs(avg_realized) if avg_realized != 0 else 1.0
        if cv > 1.0:
            confidence *= 0.7
        elif cv > 0.5:
            confidence *= 0.85
        
        return ImpactCalibrationEntry(
            asset=asset,
            session=session,
            volatility_bucket=vol_bucket,
            spread_bucket=spread_bucket,
            eta=eta,
            gamma=gamma,
            sample_count=sample_count,
            confidence=confidence,
            avg_realized_impact_bps=avg_realized,
            std_realized_impact_bps=std_realized,
            avg_prediction_error_bps=avg_error,
            last_updated=datetime.now(),
        )
    
    # =========================================================================
    # CALIBRATION (在线)
    # =========================================================================
    
    def add_sample(self, sample: CalibrationSample):
        """
        添加在线样本
        
        用于实时微调校准参数
        """
        bucket_key = f"{sample.asset}:{sample.session}:{sample.volatility_bucket}:{sample.spread_bucket}"
        self._samples[bucket_key].append(sample)
    
    def calibrate_online(self, bucket_key: str = None):
        """
        在线微调校准
        
        使用最近的样本微调参数
        """
        if bucket_key:
            keys_to_calibrate = [bucket_key]
        else:
            keys_to_calibrate = list(self._samples.keys())
        
        for key in keys_to_calibrate:
            samples = list(self._samples.get(key, []))
            
            if len(samples) < self.config.min_samples_for_calibration:
                continue
            
            # 只用最近的样本
            recent_samples = samples[-50:]
            
            # 计算误差
            errors = [s.realized_impact_bps - s.expected_impact_bps for s in recent_samples]
            avg_error = np.mean(errors)
            
            # 微调
            current = self._entries.get(key)
            if not current:
                continue
            
            adjustment = avg_error / 100
            adjustment = np.clip(adjustment, -0.1, 0.1)  # 在线微调幅度更小
            
            current.eta *= (1 + adjustment * self.config.online_learning_rate)
            current.gamma *= (1 + adjustment * self.config.online_learning_rate)
            
            # 约束
            current.eta = np.clip(current.eta, self.config.eta_min, self.config.eta_max)
            current.gamma = np.clip(current.gamma, self.config.gamma_min, self.config.gamma_max)
            
            # 更新统计
            current.avg_prediction_error_bps = avg_error
            current.last_updated = datetime.now()
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """获取校准报告"""
        if not self._entries:
            return {"status": "no_calibration_data"}
        
        entries = list(self._entries.values())
        
        return {
            "total_buckets": len(entries),
            "total_samples": sum(e.sample_count for e in entries),
            "assets": list(set(e.asset for e in entries)),
            "avg_eta": np.mean([e.eta for e in entries]),
            "avg_gamma": np.mean([e.gamma for e in entries]),
            "avg_confidence": np.mean([e.confidence for e in entries]),
            "avg_prediction_error": np.mean([e.avg_prediction_error_bps for e in entries]),
            "by_asset": self._get_by_asset_report(),
        }
    
    def _get_by_asset_report(self) -> Dict[str, Dict]:
        """按资产分组报告"""
        by_asset: Dict[str, List[ImpactCalibrationEntry]] = defaultdict(list)
        
        for entry in self._entries.values():
            by_asset[entry.asset].append(entry)
        
        result = {}
        for asset, entries in by_asset.items():
            result[asset] = {
                "bucket_count": len(entries),
                "avg_eta": np.mean([e.eta for e in entries]),
                "avg_gamma": np.mean([e.gamma for e in entries]),
                "avg_error": np.mean([e.avg_prediction_error_bps for e in entries]),
            }
        
        return result


# =============================================================================
# SINGLETON
# =============================================================================

_calibration_table_instance: Optional[ImpactCalibrationTable] = None


def get_impact_calibration_table(
    config: ImpactCalibrationConfig = None
) -> ImpactCalibrationTable:
    """获取冲击校准表单例"""
    global _calibration_table_instance
    if _calibration_table_instance is None:
        _calibration_table_instance = ImpactCalibrationTable(config)
    elif config is not None and _calibration_table_instance.config != config:
        logger.info("[IMPACT_CAL] Config changed; refreshing singleton instance")
        _calibration_table_instance = ImpactCalibrationTable(config)
    return _calibration_table_instance


def reset_impact_calibration_table():
    """重置单例"""
    global _calibration_table_instance
    _calibration_table_instance = None


# =============================================================================
# INTEGRATION WITH PRODUCTION MARKET IMPACT
# =============================================================================

class ProductionMarketImpactCalibrationBridge:
    """
    production_market_impact 校准桥接
    
    提供简化的接口供 production_market_impact 使用
    """
    
    def __init__(self, calibration_table: ImpactCalibrationTable = None):
        self._table = calibration_table or get_impact_calibration_table()
    
    def get_calibrated_params(
        self,
        asset: str,
        volatility: float,
        spread_bps: float,
    ) -> Tuple[float, float, float]:
        """
        获取校准参数
        
        自动计算 session 和 bucket
        
        Returns:
            (eta, gamma, confidence)
        """
        # 计算 session
        hour = datetime.now().hour
        if hour < 4:
            session = "asia_morning"
        elif hour < 8:
            session = "asia_afternoon"
        elif hour < 12:
            session = "europe_morning"
        elif hour < 16:
            session = "europe_afternoon"
        elif hour < 20:
            session = "us_morning"
        else:
            session = "us_afternoon"
        
        # 计算 volatility bucket
        if volatility < 0.3:
            vol_bucket = "low"
        elif volatility < 0.6:
            vol_bucket = "medium"
        elif volatility < 1.0:
            vol_bucket = "high"
        else:
            vol_bucket = "extreme"
        
        # 计算 spread bucket
        if spread_bps < 5:
            spread_bucket = "tight"
        elif spread_bps < 15:
            spread_bucket = "normal"
        elif spread_bps < 30:
            spread_bucket = "wide"
        else:
            spread_bucket = "very_wide"
        
        return self._table.get_params(asset, session, vol_bucket, spread_bucket)
    
    def record_execution(
        self,
        asset: str,
        side: str,
        size_usd: float,
        expected_impact_bps: float,
        realized_impact_bps: float,
        volatility: float,
        spread_bps: float,
    ):
        """
        记录执行结果用于在线校准
        """
        # 计算 session
        hour = datetime.now().hour
        if hour < 4:
            session = "asia_morning"
        elif hour < 8:
            session = "asia_afternoon"
        elif hour < 12:
            session = "europe_morning"
        elif hour < 16:
            session = "europe_afternoon"
        elif hour < 20:
            session = "us_morning"
        else:
            session = "us_afternoon"
        
        sample = CalibrationSample(
            timestamp=datetime.now(),
            asset=asset,
            side=side,
            size_usd=size_usd,
            expected_impact_bps=expected_impact_bps,
            realized_impact_bps=realized_impact_bps,
            volatility=volatility,
            spread_bps=spread_bps,
            session=session,
        )
        
        self._table.add_sample(sample)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ImpactCalibrationEntry',
    'CalibrationSample',
    'ImpactCalibrationConfig',
    'ImpactCalibrationTable',
    'get_impact_calibration_table',
    'reset_impact_calibration_table',
    'ProductionMarketImpactCalibrationBridge',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing ImpactCalibrationTable")
    print("=" * 60)
    
    # 创建校准表
    config = ImpactCalibrationConfig(
        calibration_file="/tmp/test_calibration.json",
        min_samples_for_calibration=10,
    )
    table = ImpactCalibrationTable(config)
    
    # 生成模拟样本
    print("\n1. Generating calibration samples...")
    
    samples = []
    for i in range(200):
        asset = "BTC" if i % 2 == 0 else "ETH"
        vol = 0.2 + 0.6 * np.random.random()
        spread = 3 + 20 * np.random.random()
        
        # 模拟预期 vs 实际
        expected = 5 + 3 * np.random.random()
        # 实际通常比预期高一点
        realized = expected * (1.1 + 0.2 * np.random.random())
        
        sample = CalibrationSample(
            timestamp=datetime.now(),
            asset=asset,
            side="buy" if i % 3 else "sell",
            size_usd=10000 + 90000 * np.random.random(),
            expected_impact_bps=expected,
            realized_impact_bps=realized,
            volatility=vol,
            spread_bps=spread,
            session="us_morning" if i % 4 < 2 else "europe_morning",
        )
        samples.append(sample)
    
    print(f"   Generated {len(samples)} samples")
    
    # 离线校准
    print("\n2. Running offline calibration...")
    table.calibrate_offline(samples)
    print(f"   Calibrated {len(table.get_all_entries())} buckets")
    
    # 查询参数
    print("\n3. Querying calibrated params...")
    eta, gamma, conf = table.get_params("BTC", "us_morning", "medium", "normal")
    print(f"   BTC/us_morning/medium/normal: eta={eta:.4f}, gamma={gamma:.4f}, conf={conf:.2f}")
    
    eta, gamma, conf = table.get_params("ETH", "europe_morning", "high", "wide")
    print(f"   ETH/europe_morning/high/wide: eta={eta:.4f}, gamma={gamma:.4f}, conf={conf:.2f}")
    
    # 测试桥接
    print("\n4. Testing calibration bridge...")
    bridge = ProductionMarketImpactCalibrationBridge(table)
    eta, gamma, conf = bridge.get_calibrated_params("BTC", volatility=0.5, spread_bps=10)
    print(f"   Bridge query: eta={eta:.4f}, gamma={gamma:.4f}, conf={conf:.2f}")
    
    # 校准报告
    print("\n5. Calibration report:")
    report = table.get_calibration_report()
    print(f"   Total buckets: {report['total_buckets']}")
    print(f"   Assets: {report['assets']}")
    print(f"   Avg eta: {report['avg_eta']:.4f}")
    print(f"   Avg gamma: {report['avg_gamma']:.4f}")
    print(f"   Avg prediction error: {report['avg_prediction_error']:.2f} bps")
    
    print("\n✓ ImpactCalibrationTable test complete")

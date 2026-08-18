"""
================================================================================
HMATS v5.2.1 P1 - 执行质量日志
Execution Quality Logger
================================================================================

Purpose: 详细记录每一笔执行的质量指标，支持离线分析和在线监控

P1-B 核心要求:
1. 每笔执行必须记录完整指标
2. 支持按 bucket 统计 (asset, session, vol, spread)
3. 区分 RL advisory vs baseline 执行
4. 输出可用于冲击模型校准的数据

================================================================================
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionSource(Enum):
    """执行来源"""
    BASELINE = "baseline"              # 纯规则/启发式
    RL_ADVISORY = "rl_advisory"        # 使用 RL 建议
    RL_SHADOW = "rl_shadow"            # RL 影子模式 (记录但未使用)
    FALLBACK = "fallback"              # 回退到 baseline
    MANUAL = "manual"                  # 手动干预


class UTCSession(Enum):
    """UTC 交易时段"""
    ASIA_MORNING = "asia_morning"      # 00:00-04:00 UTC
    ASIA_AFTERNOON = "asia_afternoon"  # 04:00-08:00 UTC
    EUROPE_MORNING = "europe_morning"  # 08:00-12:00 UTC
    EUROPE_AFTERNOON = "europe_afternoon"  # 12:00-16:00 UTC
    US_MORNING = "us_morning"          # 16:00-20:00 UTC
    US_AFTERNOON = "us_afternoon"      # 20:00-24:00 UTC


class VolatilityBucket(Enum):
    """波动率分桶"""
    LOW = "low"                        # < 0.3 (年化 ~30%)
    MEDIUM = "medium"                  # 0.3 - 0.6
    HIGH = "high"                      # 0.6 - 1.0
    EXTREME = "extreme"                # > 1.0


class SpreadBucket(Enum):
    """点差分桶"""
    TIGHT = "tight"                    # < 5 bps
    NORMAL = "normal"                  # 5 - 15 bps
    WIDE = "wide"                      # 15 - 30 bps
    VERY_WIDE = "very_wide"            # > 30 bps


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExecutionQualityRecord:
    """
    执行质量记录
    
    每一笔执行必须产生一条这样的记录
    """
    # 基础信息
    timestamp: datetime
    execution_id: str                  # 唯一标识
    asset: str
    side: str                          # "buy" or "sell"
    
    # 订单参数
    intended_quantity: float           # 计划数量
    executed_quantity: float           # 实际执行数量
    arrival_price: float               # 下单时价格
    avg_execution_price: float         # 成交均价
    
    # 执行方式
    num_slices: int                    # 拆单数
    order_type: str                    # "market", "limit", "twap", "vwap"
    
    # 核心质量指标
    expected_slippage_bps: float       # 预期滑点
    realized_slippage_bps: float       # 实际滑点
    fill_ratio: float                  # 成交比例 (executed / intended)
    time_to_fill_sec: float            # 完成时间 (秒)
    cancel_count: int                  # 撤单次数
    
    # RL 相关
    execution_source: ExecutionSource  # 执行来源
    rl_advice_used: bool               # 是否使用 RL 建议
    rl_advice_confidence: float = 0.0  # RL 置信度 (若使用)
    rl_slice_size_pct: float = 0.0     # RL 建议的拆单比例
    rl_wait_time_sec: float = 0.0      # RL 建议的等待时间
    rl_limit_offset_bps: float = 0.0   # RL 建议的限价偏移
    
    # 市场环境
    volatility: float = 0.0            # 执行时波动率
    spread_bps: float = 0.0            # 执行时点差
    lob_imbalance: float = 0.0         # 订单簿失衡
    
    # 分桶信息 (用于统计)
    utc_session: str = ""
    volatility_bucket: str = ""
    spread_bucket: str = ""
    
    # 冲击分析
    impact_temporary_bps: float = 0.0  # 临时冲击
    impact_permanent_bps: float = 0.0  # 永久冲击
    
    # 备注
    notes: str = ""
    
    def __post_init__(self):
        """自动计算分桶"""
        if not self.utc_session:
            self.utc_session = self._compute_session().value
        if not self.volatility_bucket:
            self.volatility_bucket = self._compute_vol_bucket().value
        if not self.spread_bucket:
            self.spread_bucket = self._compute_spread_bucket().value
    
    def _compute_session(self) -> UTCSession:
        """计算 UTC 时段"""
        hour = self.timestamp.hour
        if hour < 4:
            return UTCSession.ASIA_MORNING
        elif hour < 8:
            return UTCSession.ASIA_AFTERNOON
        elif hour < 12:
            return UTCSession.EUROPE_MORNING
        elif hour < 16:
            return UTCSession.EUROPE_AFTERNOON
        elif hour < 20:
            return UTCSession.US_MORNING
        else:
            return UTCSession.US_AFTERNOON
    
    def _compute_vol_bucket(self) -> VolatilityBucket:
        """计算波动率分桶"""
        if self.volatility < 0.3:
            return VolatilityBucket.LOW
        elif self.volatility < 0.6:
            return VolatilityBucket.MEDIUM
        elif self.volatility < 1.0:
            return VolatilityBucket.HIGH
        else:
            return VolatilityBucket.EXTREME
    
    def _compute_spread_bucket(self) -> SpreadBucket:
        """计算点差分桶"""
        if self.spread_bps < 5:
            return SpreadBucket.TIGHT
        elif self.spread_bps < 15:
            return SpreadBucket.NORMAL
        elif self.spread_bps < 30:
            return SpreadBucket.WIDE
        else:
            return SpreadBucket.VERY_WIDE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["execution_source"] = self.execution_source.value
        return d
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'ExecutionQualityRecord':
        """从字典创建"""
        d = d.copy()
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["execution_source"] = ExecutionSource(d["execution_source"])
        return ExecutionQualityRecord(**d)


@dataclass
class BucketStatistics:
    """分桶统计"""
    bucket_key: str                    # e.g., "BTC:us_morning:medium:normal"
    sample_count: int = 0
    
    # 滑点统计
    slippage_mean: float = 0.0
    slippage_std: float = 0.0
    slippage_p50: float = 0.0
    slippage_p90: float = 0.0
    slippage_p99: float = 0.0
    
    # 预期 vs 实际
    prediction_error_mean: float = 0.0
    prediction_error_std: float = 0.0
    
    # 成交质量
    fill_ratio_mean: float = 0.0
    time_to_fill_mean: float = 0.0
    cancel_rate: float = 0.0
    
    # RL 对比
    rl_usage_rate: float = 0.0         # RL 使用率
    rl_vs_baseline_slippage: float = 0.0  # RL 相对 baseline 的滑点改进
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExecutionQualityLoggerConfig:
    """执行质量日志配置"""
    
    # 存储
    log_dir: str = "data/execution_logs"
    max_in_memory_records: int = 10000
    flush_interval_records: int = 100   # 每 N 条刷新到磁盘
    
    # 统计计算
    bucket_stats_window: int = 1000     # 计算统计用的窗口大小
    
    # 告警阈值
    slippage_warning_bps: float = 10.0  # 单笔滑点超过此值告警
    fill_ratio_warning: float = 0.8     # 成交率低于此值告警
    
    # EventBus
    publish_every_record: bool = True
    publish_bucket_stats: bool = True


# =============================================================================
# EXECUTION QUALITY LOGGER
# =============================================================================

class ExecutionQualityLogger:
    """
    执行质量日志记录器
    
    职责:
    1. 记录每一笔执行的详细信息
    2. 按多维度分桶统计
    3. 支持 RL vs baseline 对比
    4. 为冲击模型校准提供数据
    """
    
    def __init__(self, config: ExecutionQualityLoggerConfig = None):
        self.config = config or ExecutionQualityLoggerConfig()
        
        # 内存存储
        self._records: deque = deque(maxlen=self.config.max_in_memory_records)
        
        # 分桶存储 (用于快速统计)
        self._bucket_records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.bucket_stats_window)
        )
        
        # 统计缓存
        self._bucket_stats_cache: Dict[str, BucketStatistics] = {}
        
        # RL vs Baseline 记录
        self._rl_records: deque = deque(maxlen=5000)
        self._baseline_records: deque = deque(maxlen=5000)
        
        # 计数器
        self._total_records = 0
        self._records_since_flush = 0
        
        # 事件回调
        self._event_callback: Optional[Callable] = None
        
        # 确保目录存在
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logger.info(f"ExecutionQualityLogger initialized: log_dir={self.config.log_dir}")
    
    def set_event_callback(self, callback: Callable):
        """设置 EventBus 回调"""
        self._event_callback = callback
    
    # =========================================================================
    # CORE LOGGING
    # =========================================================================
    
    def log(self, record: ExecutionQualityRecord):
        """
        记录一笔执行
        
        这是主入口，每一笔执行必须调用此方法
        """
        self._total_records += 1
        self._records_since_flush += 1
        
        # 存入内存
        self._records.append(record)
        
        # 分桶存储
        bucket_key = self._compute_bucket_key(record)
        self._bucket_records[bucket_key].append(record)
        
        # RL vs Baseline 分类
        if record.rl_advice_used:
            self._rl_records.append(record)
        else:
            self._baseline_records.append(record)
        
        # 告警检查
        self._check_alerts(record)
        
        # 发布事件
        if self.config.publish_every_record and self._event_callback:
            self._publish_record_event(record)
        
        # 定期刷新
        if self._records_since_flush >= self.config.flush_interval_records:
            self._flush_to_disk()
            self._records_since_flush = 0
        
        # 更新统计
        self._invalidate_stats_cache(bucket_key)
    
    def log_simple(
        self,
        asset: str,
        side: str,
        intended_qty: float,
        executed_qty: float,
        arrival_price: float,
        avg_exec_price: float,
        expected_slippage_bps: float,
        time_to_fill_sec: float,
        rl_advice_used: bool = False,
        num_slices: int = 1,
        cancel_count: int = 0,
        volatility: float = 0.0,
        spread_bps: float = 0.0,
        rl_confidence: float = 0.0,
        notes: str = "",
    ):
        """
        简化的记录接口
        
        自动计算滑点和其他派生字段
        """
        # 计算实际滑点
        if side == "buy":
            realized_slippage_bps = (avg_exec_price - arrival_price) / arrival_price * 10000
        else:
            realized_slippage_bps = (arrival_price - avg_exec_price) / arrival_price * 10000
        
        # 成交比例
        fill_ratio = executed_qty / intended_qty if intended_qty > 0 else 0
        
        # 执行来源
        if rl_advice_used:
            source = ExecutionSource.RL_ADVISORY
        else:
            source = ExecutionSource.BASELINE
        
        # P68: tz-aware so _compute_session().hour reads UTC, not local time.
        # Naive datetime.now() returns the host's local hour and silently
        # corrupts session bucketing on non-UTC containers.
        _now_utc = datetime.now(timezone.utc)
        record = ExecutionQualityRecord(
            timestamp=_now_utc,
            execution_id=f"{asset}_{_now_utc.timestamp():.0f}",
            asset=asset,
            side=side,
            intended_quantity=intended_qty,
            executed_quantity=executed_qty,
            arrival_price=arrival_price,
            avg_execution_price=avg_exec_price,
            num_slices=num_slices,
            order_type="limit",
            expected_slippage_bps=expected_slippage_bps,
            realized_slippage_bps=realized_slippage_bps,
            fill_ratio=fill_ratio,
            time_to_fill_sec=time_to_fill_sec,
            cancel_count=cancel_count,
            execution_source=source,
            rl_advice_used=rl_advice_used,
            rl_advice_confidence=rl_confidence,
            volatility=volatility,
            spread_bps=spread_bps,
            notes=notes,
        )
        
        self.log(record)
        return record
    
    def _compute_bucket_key(self, record: ExecutionQualityRecord) -> str:
        """计算分桶键"""
        return f"{record.asset}:{record.utc_session}:{record.volatility_bucket}:{record.spread_bucket}"
    
    def _check_alerts(self, record: ExecutionQualityRecord):
        """检查告警条件"""
        # 滑点告警
        if record.realized_slippage_bps > self.config.slippage_warning_bps:
            logger.warning(
                f"High slippage: {record.asset} {record.side} "
                f"{record.realized_slippage_bps:.1f} bps > {self.config.slippage_warning_bps} bps"
            )
        
        # 成交率告警
        if record.fill_ratio < self.config.fill_ratio_warning:
            logger.warning(
                f"Low fill ratio: {record.asset} {record.side} "
                f"{record.fill_ratio:.1%} < {self.config.fill_ratio_warning:.1%}"
            )
    
    def _publish_record_event(self, record: ExecutionQualityRecord):
        """发布记录事件"""
        if not self._event_callback:
            return
        
        try:
            self._event_callback("EXECUTION_QUALITY_LOG", {
                "record": record.to_dict(),
                "total_records": self._total_records,
            })
        except Exception as e:
            logger.warning(f"Failed to publish execution quality event: {e}")
    
    def _invalidate_stats_cache(self, bucket_key: str):
        """使统计缓存失效"""
        if bucket_key in self._bucket_stats_cache:
            del self._bucket_stats_cache[bucket_key]
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_bucket_stats(self, bucket_key: str) -> BucketStatistics:
        """
        获取分桶统计
        
        使用缓存提高性能
        """
        if bucket_key in self._bucket_stats_cache:
            return self._bucket_stats_cache[bucket_key]
        
        records = list(self._bucket_records.get(bucket_key, []))
        if not records:
            return BucketStatistics(bucket_key=bucket_key)
        
        # 计算滑点统计
        slippages = [r.realized_slippage_bps for r in records]
        prediction_errors = [r.realized_slippage_bps - r.expected_slippage_bps for r in records]
        
        # RL 相关
        rl_records = [r for r in records if r.rl_advice_used]
        baseline_records = [r for r in records if not r.rl_advice_used]
        
        rl_usage_rate = len(rl_records) / len(records) if records else 0
        
        # RL vs Baseline 对比
        rl_vs_baseline = 0.0
        if rl_records and baseline_records:
            rl_avg = np.mean([r.realized_slippage_bps for r in rl_records])
            baseline_avg = np.mean([r.realized_slippage_bps for r in baseline_records])
            rl_vs_baseline = baseline_avg - rl_avg  # 正数表示 RL 更好
        
        stats = BucketStatistics(
            bucket_key=bucket_key,
            sample_count=len(records),
            slippage_mean=np.mean(slippages),
            slippage_std=np.std(slippages),
            slippage_p50=np.percentile(slippages, 50),
            slippage_p90=np.percentile(slippages, 90),
            slippage_p99=np.percentile(slippages, 99),
            prediction_error_mean=np.mean(prediction_errors),
            prediction_error_std=np.std(prediction_errors),
            fill_ratio_mean=np.mean([r.fill_ratio for r in records]),
            time_to_fill_mean=np.mean([r.time_to_fill_sec for r in records]),
            cancel_rate=np.mean([1 if r.cancel_count > 0 else 0 for r in records]),
            rl_usage_rate=rl_usage_rate,
            rl_vs_baseline_slippage=rl_vs_baseline,
        )
        
        self._bucket_stats_cache[bucket_key] = stats
        return stats
    
    def get_all_bucket_stats(self) -> Dict[str, BucketStatistics]:
        """获取所有分桶统计"""
        result = {}
        for bucket_key in self._bucket_records.keys():
            result[bucket_key] = self.get_bucket_stats(bucket_key)
        return result
    
    def get_rl_comparison(self) -> Dict[str, Any]:
        """
        获取 RL vs Baseline 对比
        
        这是 P1 的核心输出之一
        """
        rl_list = list(self._rl_records)
        baseline_list = list(self._baseline_records)
        
        if not rl_list and not baseline_list:
            return {"status": "no_data"}
        
        result = {
            "rl_count": len(rl_list),
            "baseline_count": len(baseline_list),
            "rl_ratio": len(rl_list) / (len(rl_list) + len(baseline_list)) if (rl_list or baseline_list) else 0,
        }
        
        # RL 统计
        if rl_list:
            result["rl_slippage_mean"] = np.mean([r.realized_slippage_bps for r in rl_list])
            result["rl_slippage_p50"] = np.percentile([r.realized_slippage_bps for r in rl_list], 50)
            result["rl_fill_ratio"] = np.mean([r.fill_ratio for r in rl_list])
            result["rl_time_to_fill"] = np.mean([r.time_to_fill_sec for r in rl_list])
        
        # Baseline 统计
        if baseline_list:
            result["baseline_slippage_mean"] = np.mean([r.realized_slippage_bps for r in baseline_list])
            result["baseline_slippage_p50"] = np.percentile([r.realized_slippage_bps for r in baseline_list], 50)
            result["baseline_fill_ratio"] = np.mean([r.fill_ratio for r in baseline_list])
            result["baseline_time_to_fill"] = np.mean([r.time_to_fill_sec for r in baseline_list])
        
        # 对比
        if rl_list and baseline_list:
            result["slippage_improvement_bps"] = result["baseline_slippage_mean"] - result["rl_slippage_mean"]
            result["slippage_improvement_pct"] = (
                result["slippage_improvement_bps"] / result["baseline_slippage_mean"] * 100
                if result["baseline_slippage_mean"] != 0 else 0
            )
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """获取整体摘要"""
        records = list(self._records)
        if not records:
            return {"status": "no_data", "total_records": 0}
        
        return {
            "total_records": self._total_records,
            "in_memory_records": len(records),
            "unique_assets": len(set(r.asset for r in records)),
            "unique_buckets": len(self._bucket_records),
            "overall_slippage_mean": np.mean([r.realized_slippage_bps for r in records]),
            "overall_fill_ratio": np.mean([r.fill_ratio for r in records]),
            "rl_comparison": self.get_rl_comparison(),
        }
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _flush_to_disk(self):
        """刷新到磁盘"""
        if not self._records:
            return
        
        try:
            # 按日期分文件
            date_str = datetime.now().strftime("%Y%m%d")
            filepath = os.path.join(self.config.log_dir, f"execution_quality_{date_str}.jsonl")
            
            # 追加写入
            records_to_write = list(self._records)[-self.config.flush_interval_records:]
            with open(filepath, "a", encoding="utf-8") as f:
                for record in records_to_write:
                    f.write(json.dumps(record.to_dict()) + "\n")
            
            logger.debug(f"Flushed {len(records_to_write)} records to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to flush execution quality logs: {e}")
    
    def load_historical(self, days: int = 7) -> List[ExecutionQualityRecord]:
        """加载历史记录"""
        records = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            filepath = os.path.join(self.config.log_dir, f"execution_quality_{date_str}.jsonl")
            
            if not os.path.exists(filepath):
                continue
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            records.append(ExecutionQualityRecord.from_dict(d))
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        return records

    def has_persisted_records(self) -> bool:
        """Return True when at least one non-empty execution-quality file exists."""
        try:
            for name in os.listdir(self.config.log_dir):
                if not name.startswith("execution_quality_") or not name.endswith(".jsonl"):
                    continue
                path = os.path.join(self.config.log_dir, name)
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    return True
        except Exception:
            return False
        return False

    def backfill_from_shadow_ledger(self, ledger_dir: str = "data/shadow_ledger") -> int:
        """Backfill coarse execution-quality records from historical shadow fills."""
        if not os.path.isdir(ledger_dir):
            return 0

        count = 0
        for name in sorted(os.listdir(ledger_dir)):
            if not name.startswith("ledger_") or not name.endswith(".jsonl"):
                continue
            path = os.path.join(ledger_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if row.get("entry_type") != "FILL":
                            continue
                        data = row.get("data", {})
                        price = float(data.get("price", 0.0) or 0.0)
                        qty = abs(float(data.get("size", 0.0) or 0.0))
                        if price <= 0 or qty <= 0:
                            continue
                        ts_raw = str(row.get("timestamp", ""))
                        try:
                            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                        except Exception:
                            # P68: tz-aware fallback — match the success branch
                            # (which produces an aware datetime via "+00:00").
                            ts = datetime.now(timezone.utc)
                        record = ExecutionQualityRecord(
                            timestamp=ts,
                            execution_id=str(
                                data.get("fill_id")
                                or data.get("order_id")
                                or f"{row.get('asset', 'UNK')}_{count}"
                            ),
                            asset=str(row.get("asset", "UNK")),
                            side=str(data.get("side", "buy")).lower(),
                            intended_quantity=qty,
                            executed_quantity=qty,
                            arrival_price=price,
                            avg_execution_price=price,
                            num_slices=1,
                            order_type="backfill",
                            expected_slippage_bps=0.0,
                            realized_slippage_bps=0.0,
                            fill_ratio=1.0,
                            time_to_fill_sec=0.0,
                            cancel_count=0,
                            execution_source=ExecutionSource.BASELINE,
                            rl_advice_used=False,
                            volatility=0.0,
                            spread_bps=0.0,
                            notes="shadow_ledger_backfill_partial",
                        )
                        self.log(record)
                        count += 1
            except Exception as e:
                logger.warning(f"Failed to backfill execution quality from {path}: {e}")
        return count

    def ensure_backfilled(self, ledger_dir: str = "data/shadow_ledger") -> int:
        """Populate execution-quality logs from shadow-ledger fills when empty."""
        if self.has_persisted_records():
            return 0
        count = self.backfill_from_shadow_ledger(ledger_dir=ledger_dir)
        if count > 0:
            logger.info(f"ExecutionQualityLogger backfilled {count} records from {ledger_dir}")
        return count

    def export_for_calibration(self) -> List[Dict]:
        """
        导出用于冲击模型校准的数据
        
        格式适配 impact_calibration.py
        """
        records = list(self._records)
        
        return [
            {
                "asset": r.asset,
                "side": r.side,
                "size_usd": r.intended_quantity * r.arrival_price,
                "expected_impact_bps": r.expected_slippage_bps,
                "realized_impact_bps": r.realized_slippage_bps,
                "volatility": r.volatility,
                "spread_bps": r.spread_bps,
                "utc_session": r.utc_session,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records
        ]


# =============================================================================
# SINGLETON
# =============================================================================

_logger_instance: Optional[ExecutionQualityLogger] = None


def get_execution_quality_logger(
    config: ExecutionQualityLoggerConfig = None
) -> ExecutionQualityLogger:
    """获取执行质量日志单例"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ExecutionQualityLogger(config)
    elif config is not None and _logger_instance.config != config:
        logger.info("[EXEC_QUALITY] Config changed; refreshing singleton instance")
        _logger_instance = ExecutionQualityLogger(config)
    return _logger_instance


def reset_execution_quality_logger():
    """重置单例"""
    global _logger_instance
    _logger_instance = None


# =============================================================================
# EVENTBUS INTEGRATION
# =============================================================================

class ExecutionQualityEventBusIntegration:
    """执行质量 EventBus 集成"""
    
    def __init__(self, quality_logger: ExecutionQualityLogger):
        self.logger = quality_logger
        self._event_manager = None
    
    def _lazy_init(self):
        """延迟初始化"""
        try:
            from infra.unified_event_v521 import get_unified_event_manager
            self._event_manager = get_unified_event_manager()
            
            # 设置发布回调
            self.logger.set_event_callback(self._publish_event)
            
            return True
        except Exception as e:
            logger.warning(f"EventBus integration init failed: {e}")
            return False
    
    def _publish_event(self, event_type: str, payload: Dict):
        """发布事件"""
        if not self._event_manager:
            if not self._lazy_init():
                return
        
        try:
            from infra.unified_event_v521 import UnifiedEvent, EventTypeV521
            
            event = UnifiedEvent(
                event_type=EventTypeV521.EXECUTION_COMPLETE,
                source="execution_quality_logger",
                payload=payload,
            )
            self._event_manager.process_event(event)
            
        except Exception as e:
            logger.warning(f"Failed to publish event: {e}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ExecutionQualityRecord',
    'ExecutionSource',
    'UTCSession',
    'VolatilityBucket',
    'SpreadBucket',
    'BucketStatistics',
    'ExecutionQualityLoggerConfig',
    'ExecutionQualityLogger',
    'get_execution_quality_logger',
    'reset_execution_quality_logger',
    'ExecutionQualityEventBusIntegration',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing ExecutionQualityLogger")
    print("=" * 60)
    
    # 创建 logger
    config = ExecutionQualityLoggerConfig(
        log_dir="/tmp/execution_logs_test",
        flush_interval_records=10,
    )
    quality_logger = ExecutionQualityLogger(config)
    
    # 模拟记录
    print("\n1. Logging simulated executions...")
    
    for i in range(50):
        # 交替 RL 和 baseline
        rl_used = i % 3 == 0
        
        # 模拟滑点 (RL 稍好)
        base_slippage = 5.0
        if rl_used:
            realized = base_slippage * (0.9 + 0.1 * np.random.random())
        else:
            realized = base_slippage * (1.0 + 0.15 * np.random.random())
        
        quality_logger.log_simple(
            asset="BTC" if i % 2 == 0 else "ETH",
            side="buy" if i % 4 < 2 else "sell",
            intended_qty=1.0,
            executed_qty=0.95 + 0.05 * np.random.random(),
            arrival_price=100000,
            avg_exec_price=100000 * (1 + realized / 10000),
            expected_slippage_bps=base_slippage,
            time_to_fill_sec=3 + 2 * np.random.random(),
            rl_advice_used=rl_used,
            volatility=0.3 + 0.4 * np.random.random(),
            spread_bps=5 + 10 * np.random.random(),
        )
    
    print(f"   Logged {quality_logger._total_records} records")
    
    # 获取摘要
    print("\n2. Summary:")
    summary = quality_logger.get_summary()
    print(f"   Total Records: {summary['total_records']}")
    print(f"   Unique Assets: {summary['unique_assets']}")
    print(f"   Overall Slippage: {summary['overall_slippage_mean']:.2f} bps")
    print(f"   Overall Fill Ratio: {summary['overall_fill_ratio']:.1%}")
    
    # RL 对比
    print("\n3. RL vs Baseline Comparison:")
    comparison = quality_logger.get_rl_comparison()
    print(f"   RL Count: {comparison.get('rl_count', 0)}")
    print(f"   Baseline Count: {comparison.get('baseline_count', 0)}")
    if 'slippage_improvement_bps' in comparison:
        print(f"   RL Slippage: {comparison['rl_slippage_mean']:.2f} bps")
        print(f"   Baseline Slippage: {comparison['baseline_slippage_mean']:.2f} bps")
        print(f"   Improvement: {comparison['slippage_improvement_bps']:.2f} bps ({comparison['slippage_improvement_pct']:.1f}%)")
    
    # 分桶统计
    print("\n4. Bucket Statistics:")
    all_stats = quality_logger.get_all_bucket_stats()
    for bucket_key, stats in list(all_stats.items())[:3]:
        print(f"   {bucket_key}: n={stats.sample_count}, slip={stats.slippage_mean:.2f}bps")
    
    # 导出校准数据
    print("\n5. Export for calibration:")
    calibration_data = quality_logger.export_for_calibration()
    print(f"   Exported {len(calibration_data)} records")
    
    print("\n✓ ExecutionQualityLogger test complete")

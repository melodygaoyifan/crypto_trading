"""
================================================================================
HMATS v5.2.1 - 统一事件驱动架构
Unified Event-Driven Architecture
================================================================================
Purpose: 确保所有策略、风险和执行模块均通过 EventBus 传递事件，
         统一 backtest、paper、live 逻辑

SOTA 对照:
1. 确保所有策略、风险和执行模块均通过 EventBus 传递事件
2. 统一 backtest、paper、live 逻辑
3. 实现 event_replay.py 完整回放

Features:
1. 事件适配器 (将各模块接入 EventBus)
2. 统一运行模式 (backtest/paper/live)
3. 完整事件回放支持
4. 事件订阅管理
5. 事件过滤与路由

Author: HMATS v5.2.1 SOTA Upgrade
Version: 5.2.1
================================================================================
"""

import logging
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# RUN MODES
# =============================================================================

class RunMode(Enum):
    """运行模式"""
    BACKTEST = "backtest"   # 历史回测
    PAPER = "paper"         # 模拟交易
    LIVE = "live"           # 实盘交易
    REPLAY = "replay"       # 事件回放


# =============================================================================
# EVENT TYPES (扩展)
# =============================================================================

class EventTypeV521(Enum):
    """v5.2.1 扩展事件类型"""
    # 原有类型
    SIGNAL_GENERATED = auto()
    SIGNAL_UPDATED = auto()
    FUSION_COMPLETED = auto()
    ORDER_REQUESTED = auto()
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    RISK_ALERT = auto()
    STOP_TRIGGERED = auto()
    DRAWDOWN_WARNING = auto()
    MODE_CHANGED = auto()
    REGIME_CHANGED = auto()
    HEARTBEAT = auto()
    ERROR = auto()
    PRICE_UPDATE = auto()
    
    # v5.2.1 新增
    CORRELATION_REGIME_CHANGE = auto()
    CORRELATION_JUMP = auto()
    MARKET_IMPACT_PREDICTED = auto()
    EXECUTION_PLAN_CREATED = auto()
    EXECUTION_COMPLETED = auto()
    WEIGHT_ADJUSTED = auto()
    DRIFT_DETECTED = auto()
    RL_FALLBACK_TRIGGERED = auto()
    ALPHA_SIGNAL = auto()           # 链上/情绪 Alpha
    EXPLANATION_GENERATED = auto()   # 决策解释
    DASHBOARD_REFRESH = auto()       # 仪表盘刷新
    
    # P0 多模态数据管道事件
    ONCHAIN_TICK = auto()           # 链上数据更新
    SENTIMENT_TICK = auto()         # 舆情数据更新
    MACRO_TICK = auto()             # 宏观数据更新
    LOB_TICK = auto()               # 订单簿数据更新
    DATA_HEALTH_CHANGE = auto()     # 数据健康状态变化
    DEGRADATION_TRIGGERED = auto()  # 降级触发
    
    # P0 模型 Alpha 事件
    MODEL_INTENT = auto()           # 模型 Alpha 意图
    MODEL_ACCEPTED = auto()         # 模型意图被接受
    MODEL_VETOED = auto()           # 模型意图被否决
    MODEL_WEIGHT_ADJUSTED = auto()  # 模型权重调整


# =============================================================================
# UNIFIED EVENT
# =============================================================================

@dataclass
class UnifiedEvent:
    """统一事件格式"""
    event_type: EventTypeV521
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # 因果链追踪
    event_id: str = ""
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    # 运行模式标记
    run_mode: RunMode = RunMode.LIVE
    
    # 回放支持
    original_timestamp: Optional[datetime] = None  # 原始时间 (回放时)
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "payload": self.payload,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "run_mode": self.run_mode.value,
            "original_timestamp": self.original_timestamp.isoformat() if self.original_timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UnifiedEvent':
        return cls(
            event_type=EventTypeV521[data["event_type"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", ""),
            payload=data.get("payload", {}),
            event_id=data.get("event_id", ""),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            run_mode=RunMode(data.get("run_mode", "live")),
            original_timestamp=datetime.fromisoformat(data["original_timestamp"]) if data.get("original_timestamp") else None
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# =============================================================================
# MODULE ADAPTERS
# =============================================================================

class ModuleEventAdapter:
    """
    模块事件适配器
    
    将各种模块的输出转换为统一事件格式
    """
    
    def __init__(self, source_name: str, run_mode: RunMode = RunMode.LIVE):
        self.source_name = source_name
        self.run_mode = run_mode
        self._event_bus = None
        self._event_counter = 0
    
    def _get_event_bus(self):
        """延迟获取 EventBus"""
        if self._event_bus is None:
            try:
                from infra.event_bus import get_event_bus
                self._event_bus = get_event_bus()
            except ImportError:
                logger.warning("EventBus not available")
        return self._event_bus
    
    def _generate_event_id(self) -> str:
        """生成事件 ID"""
        self._event_counter += 1
        return f"{self.source_name}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{self._event_counter}"
    
    def emit(self, 
             event_type: EventTypeV521,
             payload: Dict[str, Any],
             correlation_id: Optional[str] = None,
             causation_id: Optional[str] = None) -> Optional[UnifiedEvent]:
        """发送事件"""
        event = UnifiedEvent(
            event_type=event_type,
            source=self.source_name,
            payload=payload,
            event_id=self._generate_event_id(),
            correlation_id=correlation_id,
            causation_id=causation_id,
            run_mode=self.run_mode
        )
        
        bus = self._get_event_bus()
        if bus:
            try:
                # 转换为原 EventBus 格式
                from infra.event_bus import Event, EventType, EventPriority
                
                # 映射事件类型
                eb_type = self._map_event_type(event_type)
                
                eb_event = Event(
                    event_type=eb_type,
                    source=self.source_name,
                    payload={
                        **payload,
                        "unified_event_type": event_type.name,
                        "run_mode": self.run_mode.value
                    },
                    event_id=event.event_id,
                    correlation_id=correlation_id,
                    causation_id=causation_id
                )
                bus.publish(eb_event)
                
                logger.debug(f"Event emitted: {event_type.name} from {self.source_name}")
                
            except Exception as e:
                logger.error(f"Failed to emit event: {e}")
        
        return event
    
    def _map_event_type(self, event_type: EventTypeV521):
        """映射到原 EventBus 事件类型"""
        try:
            from infra.event_bus import EventType
            
            # 尝试直接映射
            mapping = {
                EventTypeV521.SIGNAL_GENERATED: EventType.SIGNAL_GENERATED,
                EventTypeV521.FUSION_COMPLETED: EventType.FUSION_COMPLETED,
                EventTypeV521.ORDER_SUBMITTED: EventType.ORDER_SUBMITTED,
                EventTypeV521.ORDER_FILLED: EventType.ORDER_FILLED,
                EventTypeV521.POSITION_OPENED: EventType.POSITION_OPENED,
                EventTypeV521.POSITION_CLOSED: EventType.POSITION_CLOSED,
                EventTypeV521.RISK_ALERT: EventType.RISK_ALERT,
                EventTypeV521.STOP_TRIGGERED: EventType.STOP_TRIGGERED,
                EventTypeV521.MODE_CHANGED: EventType.MODE_CHANGED,
                EventTypeV521.REGIME_CHANGED: EventType.REGIME_CHANGED,
                EventTypeV521.HEARTBEAT: EventType.HEARTBEAT,
                EventTypeV521.ERROR: EventType.ERROR,
                EventTypeV521.PRICE_UPDATE: EventType.PRICE_UPDATE,
            }
            
            return mapping.get(event_type, EventType.SIGNAL_GENERATED)
            
        except ImportError:
            return None


# =============================================================================
# STRATEGY EVENT ADAPTER
# =============================================================================

class StrategyEventAdapter(ModuleEventAdapter):
    """策略模块事件适配器"""
    
    def emit_signal(self,
                    strategy_name: str,
                    symbol: str,
                    direction: float,
                    confidence: float,
                    metadata: Dict = None) -> Optional[UnifiedEvent]:
        """发送策略信号事件"""
        payload = {
            "strategy": strategy_name,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            **(metadata or {})
        }
        return self.emit(EventTypeV521.SIGNAL_GENERATED, payload)
    
    def emit_alpha_signal(self,
                          alpha_source: str,
                          symbol: str,
                          signal_type: str,
                          strength: float,
                          metadata: Dict = None) -> Optional[UnifiedEvent]:
        """发送 Alpha 信号事件 (链上/情绪)"""
        payload = {
            "alpha_source": alpha_source,
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength,
            **(metadata or {})
        }
        return self.emit(EventTypeV521.ALPHA_SIGNAL, payload)


# =============================================================================
# RISK EVENT ADAPTER
# =============================================================================

class RiskEventAdapter(ModuleEventAdapter):
    """风险模块事件适配器"""
    
    def emit_correlation_regime_change(self,
                                        prev_regime: str,
                                        new_regime: str,
                                        avg_correlation: float,
                                        action: str) -> Optional[UnifiedEvent]:
        """发送相关性 regime 变化事件"""
        payload = {
            "prev_regime": prev_regime,
            "new_regime": new_regime,
            "avg_correlation": avg_correlation,
            "action": action
        }
        return self.emit(EventTypeV521.CORRELATION_REGIME_CHANGE, payload)
    
    def emit_correlation_jump(self,
                              direction: str,
                              zscore: float,
                              avg_correlation: float) -> Optional[UnifiedEvent]:
        """发送相关性跳跃事件"""
        payload = {
            "direction": direction,
            "zscore": zscore,
            "avg_correlation": avg_correlation
        }
        return self.emit(EventTypeV521.CORRELATION_JUMP, payload)
    
    def emit_drawdown_warning(self,
                               current_drawdown: float,
                               threshold: float,
                               action: str) -> Optional[UnifiedEvent]:
        """发送回撤警告事件"""
        payload = {
            "current_drawdown": current_drawdown,
            "threshold": threshold,
            "action": action
        }
        return self.emit(EventTypeV521.DRAWDOWN_WARNING, payload)
    
    def emit_drift_detected(self,
                            model_name: str,
                            drift_score: float,
                            fallback_triggered: bool) -> Optional[UnifiedEvent]:
        """发送漂移检测事件"""
        payload = {
            "model_name": model_name,
            "drift_score": drift_score,
            "fallback_triggered": fallback_triggered
        }
        return self.emit(EventTypeV521.DRIFT_DETECTED, payload)


# =============================================================================
# EXECUTION EVENT ADAPTER
# =============================================================================

class ExecutionEventAdapter(ModuleEventAdapter):
    """执行模块事件适配器"""
    
    def emit_impact_prediction(self,
                               symbol: str,
                               side: str,
                               quantity_usd: float,
                               estimated_impact_bps: float,
                               recommended_algo: str) -> Optional[UnifiedEvent]:
        """发送冲击预测事件"""
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity_usd": quantity_usd,
            "estimated_impact_bps": estimated_impact_bps,
            "recommended_algo": recommended_algo
        }
        return self.emit(EventTypeV521.MARKET_IMPACT_PREDICTED, payload)
    
    def emit_execution_plan(self,
                            symbol: str,
                            side: str,
                            total_quantity: float,
                            algorithm: str,
                            num_slices: int,
                            duration_minutes: int) -> Optional[UnifiedEvent]:
        """发送执行计划事件"""
        payload = {
            "symbol": symbol,
            "side": side,
            "total_quantity": total_quantity,
            "algorithm": algorithm,
            "num_slices": num_slices,
            "duration_minutes": duration_minutes
        }
        return self.emit(EventTypeV521.EXECUTION_PLAN_CREATED, payload)
    
    def emit_execution_completed(self,
                                  order_id: str,
                                  symbol: str,
                                  side: str,
                                  filled_quantity: float,
                                  avg_price: float,
                                  realized_impact_bps: float) -> Optional[UnifiedEvent]:
        """发送执行完成事件"""
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "filled_quantity": filled_quantity,
            "avg_price": avg_price,
            "realized_impact_bps": realized_impact_bps
        }
        return self.emit(EventTypeV521.EXECUTION_COMPLETED, payload)
    
    def emit_rl_fallback(self,
                         reason: str,
                         drift_score: float,
                         fallback_strategy: str) -> Optional[UnifiedEvent]:
        """发送 RL 回退事件"""
        payload = {
            "reason": reason,
            "drift_score": drift_score,
            "fallback_strategy": fallback_strategy
        }
        return self.emit(EventTypeV521.RL_FALLBACK_TRIGGERED, payload)


# =============================================================================
# UNIFIED EVENT MANAGER
# =============================================================================

class UnifiedEventManager:
    """
    统一事件管理器
    
    负责:
    1. 管理所有事件适配器
    2. 协调 backtest/paper/live 模式
    3. 事件过滤和路由
    4. 事件日志记录
    """
    
    def __init__(self, run_mode: RunMode = RunMode.LIVE):
        self.run_mode = run_mode
        
        # 适配器注册
        self._adapters: Dict[str, ModuleEventAdapter] = {}
        
        # 事件订阅
        self._subscriptions: Dict[EventTypeV521, List[Callable]] = defaultdict(list)
        
        # 事件日志
        self._event_log: List[UnifiedEvent] = []
        self._log_enabled = True
        
        # 统计
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "by_type": defaultdict(int)
        }
        
        logger.info(f"UnifiedEventManager initialized in {run_mode.value} mode")
    
    def register_adapter(self, name: str, adapter_type: str) -> ModuleEventAdapter:
        """注册事件适配器"""
        adapter_classes = {
            "strategy": StrategyEventAdapter,
            "risk": RiskEventAdapter,
            "execution": ExecutionEventAdapter,
            "default": ModuleEventAdapter
        }
        
        adapter_class = adapter_classes.get(adapter_type, ModuleEventAdapter)
        adapter = adapter_class(name, self.run_mode)
        
        self._adapters[name] = adapter
        logger.info(f"Registered adapter: {name} ({adapter_type})")
        
        return adapter
    
    def get_adapter(self, name: str) -> Optional[ModuleEventAdapter]:
        """获取适配器"""
        return self._adapters.get(name)
    
    def subscribe(self, event_type: EventTypeV521, handler: Callable):
        """订阅事件"""
        self._subscriptions[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventTypeV521, handler: Callable):
        """取消订阅"""
        if handler in self._subscriptions[event_type]:
            self._subscriptions[event_type].remove(handler)
    
    def process_event(self, event: UnifiedEvent):
        """处理事件"""
        self._stats["events_published"] += 1
        self._stats["by_type"][event.event_type.name] += 1
        
        # 记录日志
        if self._log_enabled:
            self._event_log.append(event)
        
        # 分发到订阅者
        handlers = self._subscriptions.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
                self._stats["events_delivered"] += 1
            except Exception as e:
                logger.error(f"Handler error for {event.event_type}: {e}")
    
    def set_mode(self, mode: RunMode):
        """切换运行模式"""
        old_mode = self.run_mode
        self.run_mode = mode
        
        # 更新所有适配器的模式
        for adapter in self._adapters.values():
            adapter.run_mode = mode
        
        logger.info(f"Run mode changed: {old_mode.value} -> {mode.value}")
    
    def get_event_log(self, 
                      event_types: List[EventTypeV521] = None,
                      start_time: datetime = None,
                      end_time: datetime = None) -> List[UnifiedEvent]:
        """获取事件日志"""
        events = self._event_log
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def save_event_log(self, filepath: str):
        """保存事件日志"""
        with open(filepath, 'w', encoding="utf-8") as f:
            for event in self._event_log:
                f.write(event.to_json() + '\n')
        logger.info(f"Event log saved: {filepath} ({len(self._event_log)} events)")
    
    def load_event_log(self, filepath: str) -> int:
        """加载事件日志 (用于回放)"""
        count = 0
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event = UnifiedEvent.from_dict(data)
                    event.original_timestamp = event.timestamp
                    self._event_log.append(event)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse event: {e}")
        
        logger.info(f"Loaded {count} events from {filepath}")
        return count
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            "run_mode": self.run_mode.value,
            "adapters_registered": len(self._adapters),
            "events_published": self._stats["events_published"],
            "events_delivered": self._stats["events_delivered"],
            "events_logged": len(self._event_log),
            "by_type": dict(self._stats["by_type"])
        }


# =============================================================================
# EVENT REPLAY ENGINE (v5.2.1 增强)
# =============================================================================

class EventReplayEngineV521:
    """
    事件回放引擎 (v5.2.1)
    
    支持:
    1. 完整事件回放
    2. 速度控制
    3. 故障注入
    4. 断点支持
    """
    
    def __init__(self, event_manager: UnifiedEventManager):
        self.event_manager = event_manager
        self._events: List[UnifiedEvent] = []
        self._current_index = 0
        self._is_playing = False
        self._speed = 1.0
        
        # 故障注入
        self._fault_injection_enabled = False
        self._fault_types: List[str] = []
        self._fault_probability = 0.1
    
    def load_events(self, events: List[UnifiedEvent]):
        """加载事件"""
        self._events = sorted(events, key=lambda e: e.original_timestamp or e.timestamp)
        self._current_index = 0
        logger.info(f"Loaded {len(self._events)} events for replay")
    
    def load_from_file(self, filepath: str):
        """从文件加载"""
        self.event_manager.load_event_log(filepath)
        self._events = self.event_manager._event_log[:]
        self._current_index = 0
    
    def set_speed(self, speed: float):
        """设置回放速度"""
        self._speed = max(0.1, min(100.0, speed))
    
    def enable_fault_injection(self, fault_types: List[str], probability: float = 0.1):
        """启用故障注入"""
        self._fault_injection_enabled = True
        self._fault_types = fault_types
        self._fault_probability = probability
    
    async def replay(self, 
                     start_time: datetime = None,
                     end_time: datetime = None,
                     speed: float = None):
        """异步回放事件"""
        if speed:
            self._speed = speed
        
        self._is_playing = True
        self.event_manager.set_mode(RunMode.REPLAY)
        
        # 过滤事件
        events = self._events
        if start_time:
            events = [e for e in events if (e.original_timestamp or e.timestamp) >= start_time]
        if end_time:
            events = [e for e in events if (e.original_timestamp or e.timestamp) <= end_time]
        
        logger.info(f"Starting replay: {len(events)} events at {self._speed}x speed")
        
        prev_time = None
        
        for i, event in enumerate(events):
            if not self._is_playing:
                break
            
            event_time = event.original_timestamp or event.timestamp
            
            # 计算延迟
            if prev_time and self._speed > 0:
                delay = (event_time - prev_time).total_seconds() / self._speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 5.0))
            
            # 故障注入
            if self._fault_injection_enabled and self._should_inject_fault():
                self._inject_fault(event)
            
            # 更新时间戳为当前时间
            event.timestamp = datetime.now(timezone.utc)
            
            # 处理事件
            self.event_manager.process_event(event)
            
            prev_time = event_time
            self._current_index = i + 1
        
        self._is_playing = False
        logger.info(f"Replay complete: processed {self._current_index} events")
    
    def _should_inject_fault(self) -> bool:
        """判断是否注入故障"""
        import random
        return random.random() < self._fault_probability
    
    def _inject_fault(self, event: UnifiedEvent):
        """注入故障"""
        import random
        
        if not self._fault_types:
            return
        
        fault_type = random.choice(self._fault_types)
        
        if fault_type == "delay":
            # 延迟
            import time
            time.sleep(random.uniform(0.1, 0.5))
            
        elif fault_type == "corrupt":
            # 数据损坏
            if "confidence" in event.payload:
                event.payload["confidence"] = random.uniform(0, 1)
                
        elif fault_type == "drop":
            # 丢弃 (不处理)
            pass
        
        logger.warning(f"Fault injected: {fault_type} on event {event.event_id}")
    
    def stop(self):
        """停止回放"""
        self._is_playing = False
    
    def get_progress(self) -> Dict:
        """获取回放进度"""
        return {
            "total_events": len(self._events),
            "processed": self._current_index,
            "progress_pct": self._current_index / len(self._events) * 100 if self._events else 0,
            "is_playing": self._is_playing,
            "speed": self._speed
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_unified_event_manager: Optional[UnifiedEventManager] = None


def get_unified_event_manager(run_mode: RunMode = None) -> UnifiedEventManager:
    """获取单例"""
    global _unified_event_manager
    if _unified_event_manager is None:
        _unified_event_manager = UnifiedEventManager(run_mode or RunMode.LIVE)
    return _unified_event_manager


def reset_unified_event_manager():
    """重置单例"""
    global _unified_event_manager
    _unified_event_manager = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def init_event_driven_system(run_mode: RunMode = RunMode.LIVE) -> UnifiedEventManager:
    """初始化事件驱动系统"""
    reset_unified_event_manager()
    manager = get_unified_event_manager(run_mode)
    
    # 注册标准适配器
    manager.register_adapter("strategy_quant", "strategy")
    manager.register_adapter("strategy_sentiment", "strategy")
    manager.register_adapter("strategy_onchain", "strategy")
    manager.register_adapter("risk_correlation", "risk")
    manager.register_adapter("risk_drawdown", "risk")
    manager.register_adapter("execution", "execution")
    
    logger.info(f"Event-driven system initialized in {run_mode.value} mode")
    
    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RunMode',
    'EventTypeV521',
    'UnifiedEvent',
    'ModuleEventAdapter',
    'StrategyEventAdapter',
    'RiskEventAdapter',
    'ExecutionEventAdapter',
    'UnifiedEventManager',
    'EventReplayEngineV521',
    'get_unified_event_manager',
    'reset_unified_event_manager',
    'init_event_driven_system'
]


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化系统
    manager = init_event_driven_system(RunMode.PAPER)
    
    # 订阅事件
    def handle_signal(event: UnifiedEvent):
        print(f"Signal received: {event.payload}")
    
    manager.subscribe(EventTypeV521.SIGNAL_GENERATED, handle_signal)
    
    # 获取适配器并发送事件
    strategy_adapter = manager.get_adapter("strategy_quant")
    if strategy_adapter and isinstance(strategy_adapter, StrategyEventAdapter):
        strategy_adapter.emit_signal(
            strategy_name="trend_following",
            symbol="BTC",
            direction=0.8,
            confidence=0.75
        )
    
    risk_adapter = manager.get_adapter("risk_correlation")
    if risk_adapter and isinstance(risk_adapter, RiskEventAdapter):
        risk_adapter.emit_correlation_regime_change(
            prev_regime="normal",
            new_regime="elevated",
            avg_correlation=0.88,
            action="reduce_size"
        )
    
    # 查看统计
    print("\n=== Event Stats ===")
    print(manager.get_stats())
    
    # 保存日志
    manager.save_event_log("/tmp/test_events.jsonl")
    
    # 测试回放
    print("\n=== Replay Test ===")
    replay_engine = EventReplayEngineV521(manager)
    replay_engine.load_from_file("/tmp/test_events.jsonl")
    replay_engine.set_speed(10.0)
    
    import asyncio
    asyncio.run(replay_engine.replay())
    
    print(f"Replay progress: {replay_engine.get_progress()}")

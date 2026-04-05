"""
================================================================================
HMATS v5.2.1 SOTA - LOB 回放仿真引擎
LOB Replay Engine
================================================================================

Purpose: 使用历史 LOB 数据进行执行仿真

核心功能:
1. 历史 LOB 回放
2. Fill probability 模型
3. Queue dynamics 模拟
4. Spread widening (熊市场景)

用于:
- 执行策略训练
- 执行策略评估
- 熊市执行压力测试

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LOBReplayConfig:
    """LOB 回放配置"""
    
    # 时间分辨率
    tick_interval_ms: int = 100       # 100ms tick
    
    # 订单簿深度
    max_depth_levels: int = 20
    
    # Fill 模型参数
    base_fill_prob: float = 0.8       # 基础成交概率
    queue_priority_decay: float = 0.1 # 队列优先级衰减
    
    # 熊市参数
    bear_spread_multiplier: float = 2.5  # 熊市 spread 放大
    bear_depth_reduction: float = 0.4    # 熊市 depth 收缩
    panic_fill_prob: float = 0.3         # 恐慌时的成交概率


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LOBLevel:
    """订单簿层级"""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class LOBSnapshot:
    """订单簿快照"""
    timestamp: datetime
    asset: str
    
    # Bid/Ask
    bids: List[LOBLevel] = field(default_factory=list)
    asks: List[LOBLevel] = field(default_factory=list)
    
    # 聚合指标
    mid_price: float = 0.0
    spread_bps: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    imbalance: float = 0.0
    
    def compute_metrics(self):
        """计算聚合指标"""
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            
            self.mid_price = (best_bid + best_ask) / 2
            self.spread_bps = (best_ask - best_bid) / self.mid_price * 10000 if self.mid_price > 0 else 0
            
            self.bid_depth_usd = sum(l.price * l.quantity for l in self.bids)
            self.ask_depth_usd = sum(l.price * l.quantity for l in self.asks)
            
            total_depth = self.bid_depth_usd + self.ask_depth_usd
            if total_depth > 0:
                self.imbalance = (self.bid_depth_usd - self.ask_depth_usd) / total_depth


@dataclass
class Order:
    """订单"""
    order_id: str
    side: str                  # "buy" or "sell"
    price: float               # 0 for market order
    quantity: float
    order_type: str = "limit"  # "limit", "market"
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 状态
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"    # pending, partial, filled, cancelled


@dataclass 
class Fill:
    """成交记录"""
    order_id: str
    price: float
    quantity: float
    timestamp: datetime
    side: str


@dataclass
class MarketCondition:
    """市场状况"""
    regime: str = "normal"     # normal, stressed, panic
    volatility: float = 0.0
    momentum: float = 0.0
    fear_index: float = 50.0


# =============================================================================
# FILL PROBABILITY MODEL
# =============================================================================

class FillProbabilityModel:
    """
    成交概率模型
    
    模拟限价单的成交概率，考虑:
    - 价格距离
    - 队列位置
    - 市场状况 (熊市偏向)
    """
    
    def __init__(self, config: LOBReplayConfig):
        self.config = config
    
    def compute_fill_probability(
        self,
        order: Order,
        lob: LOBSnapshot,
        queue_position: int,
        market_condition: MarketCondition,
    ) -> float:
        """
        计算成交概率
        
        Args:
            order: 订单
            lob: 当前 LOB
            queue_position: 队列位置 (0 = 最前)
            market_condition: 市场状况
        
        Returns:
            fill probability [0, 1]
        """
        # 市价单
        if order.order_type == "market":
            if market_condition.regime == "panic":
                return self.config.panic_fill_prob
            return 0.95
        
        # 限价单
        prob = self.config.base_fill_prob
        
        # 1. 价格距离因素
        if order.side == "buy":
            best_ask = lob.asks[0].price if lob.asks else lob.mid_price * 1.001
            price_edge = (best_ask - order.price) / best_ask * 10000  # bps
        else:
            best_bid = lob.bids[0].price if lob.bids else lob.mid_price * 0.999
            price_edge = (order.price - best_bid) / best_bid * 10000
        
        # 价格越aggressive (edge越小)，成交概率越高
        prob *= np.exp(-price_edge / 10)
        
        # 2. 队列位置因素
        prob *= np.exp(-queue_position * self.config.queue_priority_decay)
        
        # 3. 市场状况因素
        if market_condition.regime == "stressed":
            prob *= 0.7
        elif market_condition.regime == "panic":
            prob *= self.config.panic_fill_prob / self.config.base_fill_prob
        
        # 4. 做空方向在熊市更容易成交
        if order.side == "sell" and market_condition.momentum < 0:
            prob *= 1.2
        
        return float(np.clip(prob, 0, 1))


# =============================================================================
# QUEUE POSITION MODEL
# =============================================================================

class QueuePositionModel:
    """
    队列位置模型
    
    追踪订单在队列中的位置
    """
    
    def __init__(self):
        # 队列: {(price, side): [order_id, ...]}
        self._queues: Dict[Tuple[float, str], List[str]] = {}
    
    def add_order(self, order: Order) -> int:
        """添加订单到队列，返回队列位置"""
        key = (order.price, order.side)
        
        if key not in self._queues:
            self._queues[key] = []
        
        position = len(self._queues[key])
        self._queues[key].append(order.order_id)
        
        return position
    
    def remove_order(self, order: Order):
        """移除订单"""
        key = (order.price, order.side)
        
        if key in self._queues and order.order_id in self._queues[key]:
            self._queues[key].remove(order.order_id)
    
    def get_position(self, order: Order) -> int:
        """获取队列位置"""
        key = (order.price, order.side)
        
        if key not in self._queues:
            return 0
        
        try:
            return self._queues[key].index(order.order_id)
        except ValueError:
            return 0
    
    def update_from_trade(self, price: float, side: str, quantity: float):
        """
        根据成交更新队列
        
        当有成交时，队列前面的订单被消耗
        """
        key = (price, side)
        
        if key not in self._queues:
            return
        
        # 简化：假设成交消耗队列前部
        consumed = min(int(quantity / 0.01), len(self._queues[key]))
        self._queues[key] = self._queues[key][consumed:]


# =============================================================================
# LOB REPLAY ENGINE
# =============================================================================

class LOBReplayEngine:
    """
    LOB 回放引擎
    
    核心仿真引擎
    """
    
    def __init__(self, config: LOBReplayConfig = None):
        self.config = config or LOBReplayConfig()
        
        # 模型
        self._fill_model = FillProbabilityModel(self.config)
        self._queue_model = QueuePositionModel()
        
        # 历史数据
        self._lob_history: List[LOBSnapshot] = []
        self._current_idx: int = 0
        
        # 活跃订单
        self._active_orders: Dict[str, Order] = {}
        
        # 成交记录
        self._fills: List[Fill] = []
        
        # 市场状况
        self._market_condition = MarketCondition()
        
        # 统计
        self._stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_volume": 0.0,
            "avg_fill_price": 0.0,
        }
        
        logger.info("LOBReplayEngine initialized")
    
    def load_history(self, snapshots: List[LOBSnapshot]):
        """加载历史 LOB 数据"""
        self._lob_history = snapshots
        self._current_idx = 0
        logger.info(f"Loaded {len(snapshots)} LOB snapshots")
    
    def set_market_condition(self, condition: MarketCondition):
        """设置市场状况"""
        self._market_condition = condition
    
    def get_current_lob(self) -> Optional[LOBSnapshot]:
        """获取当前 LOB"""
        if not self._lob_history or self._current_idx >= len(self._lob_history):
            return None
        return self._lob_history[self._current_idx]
    
    def submit_order(self, order: Order) -> str:
        """
        提交订单
        
        Returns:
            order_id
        """
        self._active_orders[order.order_id] = order
        self._stats["orders_submitted"] += 1
        
        # 添加到队列
        if order.order_type == "limit":
            position = self._queue_model.add_order(order)
            logger.debug(f"Order {order.order_id} at queue position {position}")
        
        # 立即检查市价单
        if order.order_type == "market":
            self._process_market_order(order)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self._active_orders:
            return False
        
        order = self._active_orders[order_id]
        order.status = "cancelled"
        
        self._queue_model.remove_order(order)
        del self._active_orders[order_id]
        
        self._stats["orders_cancelled"] += 1
        
        return True
    
    def step(self) -> List[Fill]:
        """
        推进一步
        
        Returns:
            本步的成交列表
        """
        if self._current_idx >= len(self._lob_history):
            return []
        
        lob = self._lob_history[self._current_idx]
        fills = []
        
        # 处理所有活跃限价单
        for order_id, order in list(self._active_orders.items()):
            if order.order_type == "limit" and order.status == "pending":
                fill = self._try_fill_limit_order(order, lob)
                if fill:
                    fills.append(fill)
        
        self._current_idx += 1
        
        return fills
    
    def _process_market_order(self, order: Order):
        """处理市价单"""
        lob = self.get_current_lob()
        if not lob:
            return
        
        # 市价单立即成交
        if order.side == "buy" and lob.asks:
            fill_price = self._compute_market_fill_price(order, lob.asks)
        elif order.side == "sell" and lob.bids:
            fill_price = self._compute_market_fill_price(order, lob.bids)
        else:
            return
        
        # 考虑熊市 slippage
        if self._market_condition.regime == "panic":
            slippage_bps = 20 if order.side == "buy" else -20
            fill_price *= (1 + slippage_bps / 10000)
        
        fill = Fill(
            order_id=order.order_id,
            price=fill_price,
            quantity=order.quantity,
            timestamp=datetime.now(),
            side=order.side,
        )
        
        self._fills.append(fill)
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.status = "filled"
        
        del self._active_orders[order.order_id]
        
        self._stats["orders_filled"] += 1
        self._stats["total_volume"] += order.quantity
    
    def _compute_market_fill_price(
        self,
        order: Order,
        levels: List[LOBLevel],
    ) -> float:
        """计算市价单成交价 (volume-weighted)"""
        remaining = order.quantity
        total_cost = 0.0
        
        for level in levels:
            take = min(remaining, level.quantity)
            total_cost += take * level.price
            remaining -= take
            
            if remaining <= 0:
                break
        
        if order.quantity > 0:
            return total_cost / order.quantity
        return levels[0].price if levels else 0
    
    def _try_fill_limit_order(
        self,
        order: Order,
        lob: LOBSnapshot,
    ) -> Optional[Fill]:
        """尝试成交限价单"""
        # 检查价格是否可成交
        if order.side == "buy":
            if not lob.asks or order.price < lob.asks[0].price:
                return None
        else:
            if not lob.bids or order.price > lob.bids[0].price:
                return None
        
        # 计算成交概率
        queue_position = self._queue_model.get_position(order)
        fill_prob = self._fill_model.compute_fill_probability(
            order, lob, queue_position, self._market_condition
        )
        
        # 随机判断是否成交
        if np.random.random() > fill_prob:
            return None
        
        # 成交
        fill = Fill(
            order_id=order.order_id,
            price=order.price,
            quantity=order.quantity - order.filled_quantity,
            timestamp=datetime.now(),
            side=order.side,
        )
        
        self._fills.append(fill)
        order.filled_quantity = order.quantity
        order.avg_fill_price = order.price
        order.status = "filled"
        
        self._queue_model.remove_order(order)
        del self._active_orders[order.order_id]
        
        self._stats["orders_filled"] += 1
        self._stats["total_volume"] += fill.quantity
        
        return fill
    
    def run_simulation(self, orders: List[Order]) -> Dict[str, Any]:
        """
        运行完整仿真
        
        Args:
            orders: 按时间排序的订单列表
        
        Returns:
            仿真结果
        """
        all_fills = []
        order_idx = 0
        
        while self._current_idx < len(self._lob_history):
            current_time = self._lob_history[self._current_idx].timestamp
            
            # 提交到达的订单
            while order_idx < len(orders) and orders[order_idx].timestamp <= current_time:
                self.submit_order(orders[order_idx])
                order_idx += 1
            
            # 推进一步
            fills = self.step()
            all_fills.extend(fills)
        
        # 计算统计
        if all_fills:
            prices = [f.price for f in all_fills]
            self._stats["avg_fill_price"] = np.mean(prices)
        
        return {
            "fills": all_fills,
            "stats": self._stats,
            "fill_rate": self._stats["orders_filled"] / max(self._stats["orders_submitted"], 1),
        }
    
    def reset(self):
        """重置引擎"""
        self._current_idx = 0
        self._active_orders = {}
        self._fills = []
        self._queue_model = QueuePositionModel()
        self._stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_volume": 0.0,
            "avg_fill_price": 0.0,
        }


# =============================================================================
# EXECUTION SIM ENVIRONMENT
# =============================================================================

class ExecutionSimEnv:
    """
    执行仿真环境
    
    提供类似 Gym 的接口，用于训练执行策略
    """
    
    def __init__(self, config: LOBReplayConfig = None):
        self.config = config or LOBReplayConfig()
        self._engine = LOBReplayEngine(self.config)
        
        # Episode 状态
        self._target_quantity: float = 0.0
        self._executed_quantity: float = 0.0
        self._arrival_price: float = 0.0
        self._total_cost: float = 0.0
        
        # 观察空间: LOB 特征 + 执行状态
        self.observation_dim = 20
        self.action_dim = 4  # [wait, market, limit_passive, limit_aggressive]
    
    def reset(
        self,
        lob_history: List[LOBSnapshot],
        target_quantity: float,
        side: str,
    ) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始观察
        """
        self._engine.reset()
        self._engine.load_history(lob_history)
        
        self._target_quantity = target_quantity
        self._side = side
        self._executed_quantity = 0.0
        
        initial_lob = self._engine.get_current_lob()
        if initial_lob:
            self._arrival_price = initial_lob.mid_price
        
        self._total_cost = 0.0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 0=wait, 1=market, 2=limit_passive, 3=limit_aggressive
        
        Returns:
            (observation, reward, done, info)
        """
        lob = self._engine.get_current_lob()
        if lob is None:
            return self._get_observation(), 0.0, True, {}
        
        # 执行动作
        fills = []
        remaining = self._target_quantity - self._executed_quantity
        
        if action == 0:  # Wait
            pass
        
        elif action == 1:  # Market order
            order = Order(
                order_id=f"sim_{datetime.now().timestamp()}",
                side=self._side,
                price=0,
                quantity=remaining * 0.5,  # 一次只执行一半
                order_type="market",
            )
            self._engine.submit_order(order)
        
        elif action == 2:  # Limit passive
            if self._side == "buy":
                price = lob.bids[0].price if lob.bids else lob.mid_price * 0.999
            else:
                price = lob.asks[0].price if lob.asks else lob.mid_price * 1.001
            
            order = Order(
                order_id=f"sim_{datetime.now().timestamp()}",
                side=self._side,
                price=price,
                quantity=remaining * 0.3,
                order_type="limit",
            )
            self._engine.submit_order(order)
        
        elif action == 3:  # Limit aggressive
            if self._side == "buy":
                price = lob.asks[0].price if lob.asks else lob.mid_price * 1.001
            else:
                price = lob.bids[0].price if lob.bids else lob.mid_price * 0.999
            
            order = Order(
                order_id=f"sim_{datetime.now().timestamp()}",
                side=self._side,
                price=price,
                quantity=remaining * 0.5,
                order_type="limit",
            )
            self._engine.submit_order(order)
        
        # 推进引擎
        fills = self._engine.step()
        
        # 更新状态
        for fill in fills:
            self._executed_quantity += fill.quantity
            self._total_cost += fill.price * fill.quantity
        
        # 计算奖励 (负的实现短缺)
        progress = self._executed_quantity / self._target_quantity
        
        if fills:
            avg_fill = self._total_cost / self._executed_quantity
            slippage_bps = (avg_fill - self._arrival_price) / self._arrival_price * 10000
            if self._side == "sell":
                slippage_bps = -slippage_bps
            
            # 奖励 = 进度奖励 - slippage 惩罚
            reward = progress * 0.1 - slippage_bps * 0.01
        else:
            reward = -0.01  # 等待惩罚
        
        # 检查是否完成
        done = (
            progress >= 0.99 or
            self._engine._current_idx >= len(self._engine._lob_history)
        )
        
        info = {
            "progress": progress,
            "executed_quantity": self._executed_quantity,
            "avg_fill_price": self._total_cost / max(self._executed_quantity, 1e-8),
            "slippage_bps": (self._total_cost / max(self._executed_quantity, 1e-8) - self._arrival_price) / self._arrival_price * 10000 if self._executed_quantity > 0 else 0,
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        lob = self._engine.get_current_lob()
        
        if lob is None:
            return np.zeros(self.observation_dim)
        
        # LOB 特征
        obs = [
            lob.spread_bps / 100,
            lob.imbalance,
            np.log1p(lob.bid_depth_usd) / 15,
            np.log1p(lob.ask_depth_usd) / 15,
        ]
        
        # 前 5 档 bid/ask
        for i in range(5):
            if i < len(lob.bids):
                obs.append((lob.mid_price - lob.bids[i].price) / lob.mid_price * 1000)
            else:
                obs.append(0)
        
        for i in range(5):
            if i < len(lob.asks):
                obs.append((lob.asks[i].price - lob.mid_price) / lob.mid_price * 1000)
            else:
                obs.append(0)
        
        # 执行状态
        progress = self._executed_quantity / max(self._target_quantity, 1e-8)
        obs.extend([
            progress,
            1 - progress,  # remaining
            self._side == "buy",
            self._side == "sell",
        ])
        
        # Padding
        while len(obs) < self.observation_dim:
            obs.append(0)
        
        return np.array(obs[:self.observation_dim], dtype=np.float32)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LOBReplayConfig',
    'LOBLevel',
    'LOBSnapshot',
    'Order',
    'Fill',
    'MarketCondition',
    'FillProbabilityModel',
    'QueuePositionModel',
    'LOBReplayEngine',
    'ExecutionSimEnv',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing LOB Replay Engine")
    print("=" * 60)
    
    # 生成模拟 LOB 数据
    print("\n1. Generating mock LOB history...")
    
    np.random.seed(42)
    lob_history = []
    
    base_price = 50000.0
    for t in range(100):
        # 熊市：价格下跌，spread 扩大
        price_drift = -0.001 * t
        current_price = base_price * (1 + price_drift)
        
        # 随着时间推移 spread 扩大 (熊市)
        spread_bps = 5 + t * 0.2
        spread = current_price * spread_bps / 10000
        
        # 生成 bid/ask
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = current_price - spread / 2 - i * spread * 0.5
            ask_price = current_price + spread / 2 + i * spread * 0.5
            
            bids.append(LOBLevel(price=bid_price, quantity=np.random.uniform(0.1, 1.0)))
            asks.append(LOBLevel(price=ask_price, quantity=np.random.uniform(0.1, 1.0)))
        
        snapshot = LOBSnapshot(
            timestamp=datetime.now() + timedelta(seconds=t * 10),
            asset="BTC",
            bids=bids,
            asks=asks,
        )
        snapshot.compute_metrics()
        lob_history.append(snapshot)
    
    print(f"   Generated {len(lob_history)} snapshots")
    print(f"   Initial price: {lob_history[0].mid_price:.2f}")
    print(f"   Final price: {lob_history[-1].mid_price:.2f}")
    print(f"   Initial spread: {lob_history[0].spread_bps:.1f} bps")
    print(f"   Final spread: {lob_history[-1].spread_bps:.1f} bps")
    
    # 测试引擎
    print("\n2. Testing LOBReplayEngine...")
    
    config = LOBReplayConfig()
    engine = LOBReplayEngine(config)
    engine.load_history(lob_history)
    
    # 设置熊市状况
    engine.set_market_condition(MarketCondition(
        regime="stressed",
        momentum=-0.02,
        fear_index=25,
    ))
    
    # 提交卖单 (做空)
    orders = []
    for i in range(5):
        order = Order(
            order_id=f"order_{i}",
            side="sell",
            price=lob_history[i * 10].bids[0].price,  # 被动卖
            quantity=0.1,
            order_type="limit",
            timestamp=lob_history[i * 10].timestamp,
        )
        orders.append(order)
    
    result = engine.run_simulation(orders)
    
    print(f"\n3. Simulation Result:")
    print(f"   Orders submitted: {result['stats']['orders_submitted']}")
    print(f"   Orders filled: {result['stats']['orders_filled']}")
    print(f"   Fill rate: {result['fill_rate']:.1%}")
    print(f"   Avg fill price: {result['stats']['avg_fill_price']:.2f}")
    
    # 测试仿真环境
    print("\n4. Testing ExecutionSimEnv...")
    
    env = ExecutionSimEnv(config)
    obs = env.reset(lob_history, target_quantity=0.5, side="sell")
    
    print(f"   Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(50):
        action = np.random.randint(0, 4)  # Random policy
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"   Steps: {step + 1}")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Progress: {info['progress']:.1%}")
    print(f"   Slippage: {info['slippage_bps']:.1f} bps")
    
    print("\n✓ LOB Replay Engine test complete")

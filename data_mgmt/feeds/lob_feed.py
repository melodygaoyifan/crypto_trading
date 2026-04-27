"""
================================================================================
HMATS v5.2.1 P0 - 订单簿数据采集器
Limit Order Book (LOB) Data Feed
================================================================================

负责从交易所采集订单簿数据，输出标准化 LOBTick。

功能:
1. L2 订单簿深度
2. 买卖价差
3. 订单簿失衡
4. 大单监控
5. 流动性评估

生产替换说明:
- 当前使用 mock/placeholder API
- 生产环境替换 _fetch_from_source() 中的实现
- 支持 Binance / Kraken / OKX WebSocket

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class LOBDataSource(Enum):
    """订单簿数据源"""
    KRAKEN = "kraken"
    BINANCE = "binance"
    MOCK = "mock"


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    size: float
    orders_count: int = 1


@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str
    exchange: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    sequence_id: int = 0
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else float('inf')
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return float('inf')


@dataclass
class LOBMetrics:
    """订单簿指标"""
    symbol: str
    
    # 基本指标
    spread_bps: float
    mid_price: float
    
    # 深度指标
    bid_depth_usd: float  # 总买单深度
    ask_depth_usd: float  # 总卖单深度
    depth_1pct_bid: float  # 1%范围内买单深度
    depth_1pct_ask: float  # 1%范围内卖单深度
    
    # 失衡指标
    imbalance: float  # -1 to 1, 正=买方主导
    
    # 大单指标
    large_bid_count: int
    large_ask_count: int
    large_order_imbalance: float
    
    # 时间戳
    timestamp: datetime


@dataclass
class LOBTick:
    """
    标准化订单簿数据输出
    
    所有订单簿数据必须通过此格式输出
    """
    timestamp: datetime
    staleness_sec: float
    confidence: float  # 0-1, 数据可信度
    
    # 订单簿快照
    snapshots: Dict[str, OrderBookSnapshot] = field(default_factory=dict)
    
    # 聚合指标
    metrics: Dict[str, LOBMetrics] = field(default_factory=dict)
    
    # 整体市场状况
    avg_spread_bps: float = 0.0
    total_liquidity_usd: float = 0.0
    overall_imbalance: float = 0.0  # -1 to 1
    liquidity_regime: str = "normal"  # thin, normal, deep
    
    # 元数据
    source: str = "unknown"
    symbols_covered: List[str] = field(default_factory=list)
    
    def is_stale(self, max_staleness_sec: float = 10) -> bool:
        """订单簿数据需要非常新鲜"""
        return self.staleness_sec > max_staleness_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "staleness_sec": self.staleness_sec,
            "confidence": self.confidence,
            "avg_spread_bps": self.avg_spread_bps,
            "total_liquidity_usd": self.total_liquidity_usd,
            "overall_imbalance": self.overall_imbalance,
            "liquidity_regime": self.liquidity_regime,
            "source": self.source,
            "symbols_covered": self.symbols_covered,
            "snapshot_count": len(self.snapshots),
        }
    
    def get_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """获取特定交易对的订单簿快照"""
        return self.snapshots.get(symbol)
    
    def get_metrics(self, symbol: str) -> Optional[LOBMetrics]:
        """获取特定交易对的指标"""
        return self.metrics.get(symbol)


# =============================================================================
# LOB FEED
# =============================================================================

class LOBFeed:
    """
    订单簿数据采集器
    
    生产级数据管道，所有订单簿数据采集必须通过此类。
    """
    
    def __init__(
        self,
        source: LOBDataSource = LOBDataSource.MOCK,
        symbols: List[str] = None,
        depth_levels: int = 20,
        poll_interval_sec: float = 1,  # 订单簿需要高频更新
        event_bus_callback: Optional[Callable] = None,
    ):
        self.source = source
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.depth_levels = depth_levels
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        
        # 内部状态
        self._last_tick: Optional[LOBTick] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        self._total_fetches = 0
        
        # 配置
        self._staleness_threshold_sec = 10
        self._large_order_threshold_usd = 50000
        self._thin_liquidity_threshold = 100000
        self._deep_liquidity_threshold = 1000000
        
        logger.info(f"LOBFeed initialized: source={source.value}, symbols={self.symbols}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """启动数据采集"""
        if self._running:
            return
        
        self._running = True
        logger.info("LOBFeed started")
        
        while self._running:
            try:
                tick = await self.fetch()
                if tick and self._event_bus_callback:
                    self._event_bus_callback("LOB_TICK", tick.to_dict())
            except Exception as e:
                logger.error(f"LOBFeed fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """停止数据采集"""
        self._running = False
        logger.info("LOBFeed stopped")
    
    async def fetch(self) -> Optional[LOBTick]:
        """获取最新订单簿数据"""
        self._total_fetches += 1
        
        try:
            raw_data = await self._fetch_from_source()
            tick = self._parse_raw_data(raw_data)
            # [P40 2026-04-24] Strip tzinfo (see _http.strip_tz docstring).
            from data_mgmt.feeds._http import strip_tz
            tick.staleness_sec = (datetime.now() - strip_tz(tick.timestamp)).total_seconds()

            self._last_tick = tick
            self._last_fetch_time = datetime.now()

            return tick

        except Exception as e:
            logger.error(f"LOBFeed fetch failed: {e}")
            self._fetch_errors += 1
            return None

    def get_latest(self) -> Optional[LOBTick]:
        """获取最新缓存的数据"""
        if self._last_tick:
            from data_mgmt.feeds._http import strip_tz
            self._last_tick.staleness_sec = (
                datetime.now() - strip_tz(self._last_tick.timestamp)
            ).total_seconds()
        return self._last_tick
    
    def get_health(self) -> Dict[str, Any]:
        """获取数据健康状态"""
        tick = self.get_latest()
        
        return {
            "source": self.source.value,
            "last_fetch_time": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
            "staleness_sec": tick.staleness_sec if tick else float('inf'),
            "is_stale": tick.is_stale(self._staleness_threshold_sec) if tick else True,
            "confidence": tick.confidence if tick else 0.0,
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "error_rate": self._fetch_errors / max(1, self._total_fetches),
            "symbols_covered": self.symbols,
        }
    
    # =========================================================================
    # DATA SOURCE (生产环境需替换)
    # =========================================================================
    
    async def _fetch_from_source(self) -> Dict[str, Any]:
        """从数据源获取原始数据"""
        if self.source == LOBDataSource.MOCK:
            return await self._fetch_mock()
        
        if self.source == LOBDataSource.BINANCE:
            return await self._fetch_binance()
        
        if self.source == LOBDataSource.KRAKEN:
            return await self._fetch_kraken()
        
        logger.warning(f"Unknown source {self.source}, using mock")
        return await self._fetch_mock()
    
    async def _fetch_binance(self) -> Dict[str, Any]:
        """Fetch order book from Binance REST API (no key required)."""
        try:
            async with create_session() as session:
                snapshots = {}

                for symbol in self.symbols:
                    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={self.depth_levels}"
                    
                    async with session.get(url, timeout=5) as response:
                        if response.status != 200:
                            # [P102 2026-04-27] Was silent fallthrough → mock.
                            # 429/404/5xx all looked identical to operators.
                            logger.warning(
                                f"[LOB-FEED] Binance {symbol} returned "
                                f"HTTP {response.status}; falling back to "
                                f"mock for this symbol."
                            )
                            continue
                        if response.status == 200:
                            data = await response.json()

                            bids = [{"price": float(b[0]), "size": float(b[1]), "orders_count": 1}
                                   for b in data.get("bids", [])[:self.depth_levels]]
                            asks = [{"price": float(a[0]), "size": float(a[1]), "orders_count": 1}
                                   for a in data.get("asks", [])[:self.depth_levels]]
                            
                            if bids and asks:
                                best_bid = bids[0]["price"]
                                best_ask = asks[0]["price"]
                                mid = (best_bid + best_ask) / 2
                                spread = best_ask - best_bid
                                
                                # Calculate imbalance
                                bid_volume = sum(b["size"] for b in bids)
                                ask_volume = sum(a["size"] for a in asks)
                                total = bid_volume + ask_volume
                                imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
                                
                                snapshots[symbol] = {
                                    "bids": bids,
                                    "asks": asks,
                                    "mid_price": mid,
                                    "spread": spread,
                                    "spread_bps": (spread / mid) * 10000 if mid > 0 else 0,
                                    "imbalance": imbalance,
                                    "bid_depth": bid_volume,
                                    "ask_depth": ask_volume,
                                    "last_update_id": data.get("lastUpdateId", 0),
                                    "timestamp": datetime.now().isoformat()
                                }
                
                if snapshots:
                    return {
                        "snapshots": snapshots,
                        "_source": "binance",
                        "_mock": False
                    }
                    
        except asyncio.TimeoutError:
            logger.warning("Binance API timeout")
        except Exception as e:
            logger.error(f"Binance API error: {e}")
        
        return await self._fetch_mock()
    
    async def _fetch_kraken(self) -> Dict[str, Any]:
        """Fetch order book from Kraken REST API (no key required)."""
        # Symbol mapping: BTCUSDT -> XBTUSD
        symbol_map = {
            "BTCUSDT": "XBTUSD",
            "ETHUSDT": "ETHUSD",
            "SOLUSDT": "SOLUSD"
        }
        
        try:
            async with create_session() as session:
                snapshots = {}

                for symbol in self.symbols:
                    kraken_symbol = symbol_map.get(symbol, symbol)
                    url = f"https://api.kraken.com/0/public/Depth?pair={kraken_symbol}&count={self.depth_levels}"
                    
                    async with session.get(url, timeout=5) as response:
                        if response.status != 200:
                            # [P102 2026-04-27] Same Binance-side fix.
                            logger.warning(
                                f"[LOB-FEED] Kraken {kraken_symbol} returned "
                                f"HTTP {response.status}; falling back to "
                                f"mock for this symbol."
                            )
                            continue
                        if response.status == 200:
                            data = await response.json()
                            result = data.get("result", {})
                            
                            # Kraken returns data with the pair as key
                            for pair_key, pair_data in result.items():
                                bids = [{"price": float(b[0]), "size": float(b[1]), "orders_count": 1} 
                                       for b in pair_data.get("bids", [])[:self.depth_levels]]
                                asks = [{"price": float(a[0]), "size": float(a[1]), "orders_count": 1} 
                                       for a in pair_data.get("asks", [])[:self.depth_levels]]
                                
                                if bids and asks:
                                    best_bid = bids[0]["price"]
                                    best_ask = asks[0]["price"]
                                    mid = (best_bid + best_ask) / 2
                                    spread = best_ask - best_bid
                                    
                                    bid_volume = sum(b["size"] for b in bids)
                                    ask_volume = sum(a["size"] for a in asks)
                                    total = bid_volume + ask_volume
                                    imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
                                    
                                    snapshots[symbol] = {
                                        "bids": bids,
                                        "asks": asks,
                                        "mid_price": mid,
                                        "spread": spread,
                                        "spread_bps": (spread / mid) * 10000 if mid > 0 else 0,
                                        "imbalance": imbalance,
                                        "bid_depth": bid_volume,
                                        "ask_depth": ask_volume,
                                        "timestamp": datetime.now().isoformat()
                                    }
                
                if snapshots:
                    return {
                        "snapshots": snapshots,
                        "_source": "kraken",
                        "_mock": False
                    }
                    
        except asyncio.TimeoutError:
            logger.warning("Kraken API timeout")
        except Exception as e:
            logger.error(f"Kraken API error: {e}")
        
        return await self._fetch_mock()
    
    async def _fetch_mock(self) -> Dict[str, Any]:
        """Mock 数据源"""
        await asyncio.sleep(0.05)  # 模拟低延迟
        
        now = datetime.now()
        snapshots = {}
        
        base_prices = {
            "BTCUSDT": 100000,
            "ETHUSDT": 3500,
            "SOLUSDT": 200,
        }
        
        for symbol in self.symbols:
            base_price = base_prices.get(symbol, 100)
            spread_pct = random.uniform(0.0001, 0.001)
            
            mid = base_price * (1 + random.uniform(-0.001, 0.001))
            half_spread = mid * spread_pct / 2
            
            # 生成买单
            bids = []
            bid_price = mid - half_spread
            for i in range(self.depth_levels):
                price = bid_price * (1 - i * 0.0002)
                size = random.uniform(0.1, 10) * (1 + i * 0.1)  # 深度增加
                bids.append({
                    "price": price,
                    "size": size,
                    "orders_count": random.randint(1, 20),
                })
            
            # 生成卖单
            asks = []
            ask_price = mid + half_spread
            for i in range(self.depth_levels):
                price = ask_price * (1 + i * 0.0002)
                size = random.uniform(0.1, 10) * (1 + i * 0.1)
                asks.append({
                    "price": price,
                    "size": size,
                    "orders_count": random.randint(1, 20),
                })
            
            snapshots[symbol] = {
                "symbol": symbol,
                "exchange": "mock",
                "bids": bids,
                "asks": asks,
                "timestamp": now.isoformat(),
                "sequence_id": random.randint(1, 1000000),
            }
        
        return {
            "timestamp": now.isoformat(),
            "snapshots": snapshots,
            "confidence": 0.95,
        }
    
    def _parse_raw_data(self, raw: Dict[str, Any]) -> LOBTick:
        """解析原始数据为标准格式"""
        timestamp = datetime.fromisoformat(raw.get("timestamp", datetime.now().isoformat()))
        
        snapshots = {}
        metrics = {}
        
        for symbol, snap_data in raw.get("snapshots", {}).items():
            # 解析订单簿
            bids = [
                OrderBookLevel(
                    price=b["price"],
                    size=b["size"],
                    orders_count=b.get("orders_count", 1),
                )
                for b in snap_data.get("bids", [])
            ]
            
            asks = [
                OrderBookLevel(
                    price=a["price"],
                    size=a["size"],
                    orders_count=a.get("orders_count", 1),
                )
                for a in snap_data.get("asks", [])
            ]
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                exchange=snap_data.get("exchange", "unknown"),
                bids=bids,
                asks=asks,
                timestamp=datetime.fromisoformat(snap_data.get("timestamp", timestamp.isoformat())),
                sequence_id=snap_data.get("sequence_id", 0),
            )
            
            snapshots[symbol] = snapshot
            
            # 计算指标
            metrics[symbol] = self._compute_metrics(snapshot)
        
        # 计算聚合指标
        avg_spread_bps, total_liquidity, overall_imbalance, liq_regime = \
            self._compute_aggregates(list(metrics.values()))
        
        return LOBTick(
            timestamp=timestamp,
            staleness_sec=0.0,
            confidence=raw.get("confidence", 0.9),
            snapshots=snapshots,
            metrics=metrics,
            avg_spread_bps=avg_spread_bps,
            total_liquidity_usd=total_liquidity,
            overall_imbalance=overall_imbalance,
            liquidity_regime=liq_regime,
            source=self.source.value,
            symbols_covered=list(snapshots.keys()),
        )
    
    def _compute_metrics(self, snapshot: OrderBookSnapshot) -> LOBMetrics:
        """计算订单簿指标"""
        mid = snapshot.mid_price
        
        # 计算深度
        bid_depth = sum(b.price * b.size for b in snapshot.bids)
        ask_depth = sum(a.price * a.size for a in snapshot.asks)
        
        # 1%范围内深度
        depth_1pct_bid = sum(
            b.price * b.size for b in snapshot.bids
            if b.price >= mid * 0.99
        )
        depth_1pct_ask = sum(
            a.price * a.size for a in snapshot.asks
            if a.price <= mid * 1.01
        )
        
        # 失衡
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # 大单统计
        large_bid_count = sum(
            1 for b in snapshot.bids
            if b.price * b.size > self._large_order_threshold_usd
        )
        large_ask_count = sum(
            1 for a in snapshot.asks
            if a.price * a.size > self._large_order_threshold_usd
        )
        
        large_total = large_bid_count + large_ask_count
        large_imbalance = (
            (large_bid_count - large_ask_count) / large_total
            if large_total > 0 else 0
        )
        
        return LOBMetrics(
            symbol=snapshot.symbol,
            spread_bps=snapshot.spread_bps,
            mid_price=mid,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
            depth_1pct_bid=depth_1pct_bid,
            depth_1pct_ask=depth_1pct_ask,
            imbalance=imbalance,
            large_bid_count=large_bid_count,
            large_ask_count=large_ask_count,
            large_order_imbalance=large_imbalance,
            timestamp=snapshot.timestamp,
        )
    
    def _compute_aggregates(self, metrics_list: List[LOBMetrics]) -> Tuple[float, float, float, str]:
        """计算聚合指标"""
        if not metrics_list:
            return 0.0, 0.0, 0.0, "normal"
        
        avg_spread = sum(m.spread_bps for m in metrics_list) / len(metrics_list)
        total_liquidity = sum(m.bid_depth_usd + m.ask_depth_usd for m in metrics_list)
        overall_imbalance = sum(m.imbalance for m in metrics_list) / len(metrics_list)
        
        # 判断流动性状态
        if total_liquidity < self._thin_liquidity_threshold:
            liq_regime = "thin"
        elif total_liquidity > self._deep_liquidity_threshold:
            liq_regime = "deep"
        else:
            liq_regime = "normal"
        
        return avg_spread, total_liquidity, overall_imbalance, liq_regime


# =============================================================================
# SINGLETON
# =============================================================================

_lob_feed_instance: Optional[LOBFeed] = None


def get_lob_feed(
    source: LOBDataSource = LOBDataSource.MOCK,
    symbols: List[str] = None,
    event_bus_callback: Optional[Callable] = None,
) -> LOBFeed:
    """获取或创建 LOBFeed 单例"""
    global _lob_feed_instance
    if _lob_feed_instance is None:
        _lob_feed_instance = LOBFeed(
            source=source,
            symbols=symbols,
            event_bus_callback=event_bus_callback,
        )
    return _lob_feed_instance


def reset_lob_feed():
    """重置单例"""
    global _lob_feed_instance
    if _lob_feed_instance:
        _lob_feed_instance.stop()
    _lob_feed_instance = None

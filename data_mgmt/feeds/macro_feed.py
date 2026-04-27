"""
================================================================================
HMATS v5.2.1 P0 - 宏观数据采集器
Macro Data Feed
================================================================================

负责从宏观数据源采集数据，输出标准化 MacroTick。

功能:
1. 利率数据 (Fed Funds, SOFR)
2. 美元指数 (DXY)
3. VIX 波动率指数
4. 股指数据 (S&P 500, Nasdaq)
5. 大宗商品 (黄金, 石油)
6. 经济日历事件

生产替换说明:
- 当前使用 mock/placeholder API
- 生产环境替换 _fetch_from_source() 中的实现
- 支持 FRED / Alpha Vantage / Investing.com 等

================================================================================
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
import random

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MacroDataSource(Enum):
    """宏观数据源"""
    FRED = "fred"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    INVESTING = "investing"
    MOCK = "mock"


@dataclass
class InterestRateData:
    """利率数据"""
    rate_type: str  # fed_funds, sofr, 10y_treasury
    value: float
    change_1d: float
    timestamp: datetime


@dataclass
class IndexData:
    """指数数据"""
    index_name: str  # dxy, vix, spx, ndx
    value: float
    change_1d: float
    change_1d_pct: float
    timestamp: datetime


@dataclass
class CommodityData:
    """大宗商品数据"""
    commodity: str  # gold, oil, silver
    price: float
    change_1d: float
    change_1d_pct: float
    timestamp: datetime


@dataclass
class EconomicEvent:
    """经济日历事件"""
    event_name: str
    country: str
    importance: str  # high, medium, low
    actual: Optional[float]
    forecast: Optional[float]
    previous: Optional[float]
    event_time: datetime


@dataclass
class MacroTick:
    """
    标准化宏观数据输出
    
    所有宏观数据必须通过此格式输出
    """
    timestamp: datetime
    staleness_sec: float
    confidence: float  # 0-1, 数据可信度
    
    # 利率数据
    interest_rates: List[InterestRateData] = field(default_factory=list)
    
    # 指数数据
    indices: List[IndexData] = field(default_factory=list)
    
    # 大宗商品数据
    commodities: List[CommodityData] = field(default_factory=list)
    
    # 经济事件
    upcoming_events: List[EconomicEvent] = field(default_factory=list)
    recent_events: List[EconomicEvent] = field(default_factory=list)
    
    # 聚合指标
    risk_on_off_score: float = 0.0  # -1 (risk-off) to 1 (risk-on)
    usd_strength: float = 0.0  # -1 (weak) to 1 (strong)
    volatility_regime: str = "normal"  # low, normal, high, extreme
    
    # 元数据
    source: str = "unknown"
    
    def is_stale(self, max_staleness_sec: float = 600) -> bool:
        """检查数据是否过期（宏观数据允许更长staleness）"""
        return self.staleness_sec > max_staleness_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "staleness_sec": self.staleness_sec,
            "confidence": self.confidence,
            "risk_on_off_score": self.risk_on_off_score,
            "usd_strength": self.usd_strength,
            "volatility_regime": self.volatility_regime,
            "source": self.source,
            "rate_count": len(self.interest_rates),
            "index_count": len(self.indices),
            "commodity_count": len(self.commodities),
            "upcoming_event_count": len(self.upcoming_events),
        }
    
    def has_high_impact_event_soon(self, hours: int = 24) -> bool:
        """检查是否有高影响力事件即将发生"""
        # [P102 2026-04-27] cutoff was naive datetime.now(); event_time
        # may be aware (parsed from ISO with tz marker). Subtraction/
        # comparison would TypeError silently inside any caller's try
        # block. Force aware UTC for cutoff; defend the per-event
        # comparison against naive event_time too.
        cutoff = datetime.now(timezone.utc) + timedelta(hours=hours)
        for event in self.upcoming_events:
            if event.importance != "high":
                continue
            _et = event.event_time
            try:
                if _et is not None and _et.tzinfo is None:
                    _et = _et.replace(tzinfo=timezone.utc)
                if _et is not None and _et <= cutoff:
                    return True
            except Exception:  # noqa: silent-swallow
                # Malformed event_time on a single record shouldn't
                # poison the whole "any high-impact soon?" query.
                continue
        return False


# =============================================================================
# MACRO FEED
# =============================================================================

class MacroFeed:
    """
    宏观数据采集器
    
    生产级数据管道，所有宏观数据采集必须通过此类。
    """
    
    def __init__(
        self,
        source: MacroDataSource = MacroDataSource.MOCK,
        poll_interval_sec: float = 300,  # 宏观数据更新频率较低
        event_bus_callback: Optional[Callable] = None,
    ):
        self.source = source
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        
        # 内部状态
        self._last_tick: Optional[MacroTick] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        self._total_fetches = 0
        
        # 配置
        self._staleness_threshold_sec = 600
        self._vix_high_threshold = 25
        self._vix_extreme_threshold = 35
        
        logger.info(f"MacroFeed initialized: source={source.value}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """启动数据采集"""
        if self._running:
            return
        
        self._running = True
        logger.info("MacroFeed started")
        
        while self._running:
            try:
                tick = await self.fetch()
                if tick and self._event_bus_callback:
                    self._event_bus_callback("MACRO_TICK", tick.to_dict())
            except Exception as e:
                logger.error(f"MacroFeed fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """停止数据采集"""
        self._running = False
        logger.info("MacroFeed stopped")
    
    async def fetch(self) -> Optional[MacroTick]:
        """获取最新宏观数据"""
        self._total_fetches += 1
        
        try:
            raw_data = await self._fetch_from_source()
            tick = self._parse_raw_data(raw_data)
            # [P40 2026-04-24] Strip tzinfo to match naive datetime.now() —
            # tick.timestamp may be aware/naive depending on the source ISO format.
            from data_mgmt.feeds._http import strip_tz
            tick.staleness_sec = (datetime.now() - strip_tz(tick.timestamp)).total_seconds()

            self._last_tick = tick
            self._last_fetch_time = datetime.now()

            return tick

        except Exception as e:
            logger.error(f"MacroFeed fetch failed: {e}")
            self._fetch_errors += 1
            return None

    def get_latest(self) -> Optional[MacroTick]:
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
        }
    
    # =========================================================================
    # DATA SOURCE (生产环境需替换)
    # =========================================================================
    
    async def _fetch_from_source(self) -> Dict[str, Any]:
        """从数据源获取原始数据"""
        if self.source == MacroDataSource.MOCK:
            return await self._fetch_mock()
        
        logger.warning(f"Unknown source {self.source}, using mock")
        return await self._fetch_mock()
    
    async def _fetch_mock(self) -> Dict[str, Any]:
        """Mock 数据源.

        [P113 (3/6) 2026-04-27] WARN signal for mock fallback visibility.
        P101 prevention.
        """
        if not getattr(self, '_mock_warned', False):
            logger.warning(
                "[MACRO] _fetch_mock invoked — macro indicators using mock "
                "data. VIX, rates, indices are SYNTHETIC this fetch."
            )
            self._mock_warned = True
        await asyncio.sleep(0.1)

        now = datetime.now()
        
        # 模拟利率数据
        interest_rates = [
            {
                "rate_type": "fed_funds",
                "value": 5.25 + random.uniform(-0.05, 0.05),
                "change_1d": random.uniform(-0.01, 0.01),
                "timestamp": now.isoformat(),
            },
            {
                "rate_type": "10y_treasury",
                "value": 4.5 + random.uniform(-0.1, 0.1),
                "change_1d": random.uniform(-0.05, 0.05),
                "timestamp": now.isoformat(),
            },
        ]
        
        # 模拟指数数据
        indices = [
            {
                "index_name": "dxy",
                "value": 103 + random.uniform(-1, 1),
                "change_1d": random.uniform(-0.5, 0.5),
                "change_1d_pct": random.uniform(-0.5, 0.5),
                "timestamp": now.isoformat(),
            },
            {
                "index_name": "vix",
                "value": 18 + random.uniform(-5, 10),
                "change_1d": random.uniform(-2, 2),
                "change_1d_pct": random.uniform(-10, 10),
                "timestamp": now.isoformat(),
            },
            {
                "index_name": "spx",
                "value": 5800 + random.uniform(-100, 100),
                "change_1d": random.uniform(-50, 50),
                "change_1d_pct": random.uniform(-1, 1),
                "timestamp": now.isoformat(),
            },
        ]
        
        # 模拟大宗商品
        commodities = [
            {
                "commodity": "gold",
                "price": 2700 + random.uniform(-50, 50),
                "change_1d": random.uniform(-20, 20),
                "change_1d_pct": random.uniform(-1, 1),
                "timestamp": now.isoformat(),
            },
            {
                "commodity": "oil",
                "price": 75 + random.uniform(-3, 3),
                "change_1d": random.uniform(-2, 2),
                "change_1d_pct": random.uniform(-2, 2),
                "timestamp": now.isoformat(),
            },
        ]
        
        # 模拟经济事件
        upcoming_events = []
        for i in range(random.randint(0, 3)):
            event_time = now + timedelta(hours=random.randint(1, 72))
            upcoming_events.append({
                "event_name": random.choice(["FOMC Meeting", "CPI Release", "NFP", "GDP"]),
                "country": "US",
                "importance": random.choice(["high", "medium", "low"]),
                "actual": None,
                "forecast": random.uniform(-0.5, 0.5) if random.random() > 0.5 else None,
                "previous": random.uniform(-0.5, 0.5),
                "event_time": event_time.isoformat(),
            })
        
        return {
            "timestamp": now.isoformat(),
            "interest_rates": interest_rates,
            "indices": indices,
            "commodities": commodities,
            "upcoming_events": upcoming_events,
            "recent_events": [],
            "confidence": 0.9,
        }
    
    def _parse_raw_data(self, raw: Dict[str, Any]) -> MacroTick:
        """解析原始数据为标准格式"""
        timestamp = datetime.fromisoformat(raw.get("timestamp", datetime.now().isoformat()))
        
        # 解析利率数据
        interest_rates = []
        for r in raw.get("interest_rates", []):
            interest_rates.append(InterestRateData(
                rate_type=r["rate_type"],
                value=r["value"],
                change_1d=r["change_1d"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            ))
        
        # 解析指数数据
        indices = []
        for i in raw.get("indices", []):
            indices.append(IndexData(
                index_name=i["index_name"],
                value=i["value"],
                change_1d=i["change_1d"],
                change_1d_pct=i["change_1d_pct"],
                timestamp=datetime.fromisoformat(i["timestamp"]),
            ))
        
        # 解析大宗商品
        commodities = []
        for c in raw.get("commodities", []):
            commodities.append(CommodityData(
                commodity=c["commodity"],
                price=c["price"],
                change_1d=c["change_1d"],
                change_1d_pct=c["change_1d_pct"],
                timestamp=datetime.fromisoformat(c["timestamp"]),
            ))
        
        # 解析经济事件
        upcoming_events = []
        for e in raw.get("upcoming_events", []):
            upcoming_events.append(EconomicEvent(
                event_name=e["event_name"],
                country=e["country"],
                importance=e["importance"],
                actual=e.get("actual"),
                forecast=e.get("forecast"),
                previous=e.get("previous"),
                event_time=datetime.fromisoformat(e["event_time"]),
            ))
        
        recent_events = []
        for e in raw.get("recent_events", []):
            recent_events.append(EconomicEvent(
                event_name=e["event_name"],
                country=e["country"],
                importance=e["importance"],
                actual=e.get("actual"),
                forecast=e.get("forecast"),
                previous=e.get("previous"),
                event_time=datetime.fromisoformat(e["event_time"]),
            ))
        
        # 计算聚合指标
        risk_on_off, usd_strength, vol_regime = self._compute_aggregates(
            indices, commodities
        )
        
        return MacroTick(
            timestamp=timestamp,
            staleness_sec=0.0,
            confidence=raw.get("confidence", 0.8),
            interest_rates=interest_rates,
            indices=indices,
            commodities=commodities,
            upcoming_events=upcoming_events,
            recent_events=recent_events,
            risk_on_off_score=risk_on_off,
            usd_strength=usd_strength,
            volatility_regime=vol_regime,
            source=self.source.value,
        )
    
    def _compute_aggregates(
        self,
        indices: List[IndexData],
        commodities: List[CommodityData],
    ) -> tuple:
        """计算聚合指标"""
        risk_on_off = 0.0
        usd_strength = 0.0
        vol_regime = "normal"
        
        for idx in indices:
            if idx.index_name == "spx":
                # SPX上涨 = risk-on
                risk_on_off += idx.change_1d_pct * 0.3
            elif idx.index_name == "vix":
                # VIX
                if idx.value > self._vix_extreme_threshold:
                    vol_regime = "extreme"
                    risk_on_off -= 0.5
                elif idx.value > self._vix_high_threshold:
                    vol_regime = "high"
                    risk_on_off -= 0.3
                elif idx.value < 15:
                    vol_regime = "low"
            elif idx.index_name == "dxy":
                # DXY强弱
                usd_strength = idx.change_1d_pct * 0.2
        
        # 黄金上涨 = risk-off
        for comm in commodities:
            if comm.commodity == "gold":
                risk_on_off -= comm.change_1d_pct * 0.1
        
        risk_on_off = max(-1, min(1, risk_on_off))
        usd_strength = max(-1, min(1, usd_strength))
        
        return risk_on_off, usd_strength, vol_regime


# =============================================================================
# SINGLETON
# =============================================================================

_macro_feed_instance: Optional[MacroFeed] = None


def get_macro_feed(
    source: MacroDataSource = MacroDataSource.MOCK,
    event_bus_callback: Optional[Callable] = None,
) -> MacroFeed:
    """获取或创建 MacroFeed 单例"""
    global _macro_feed_instance
    if _macro_feed_instance is None:
        _macro_feed_instance = MacroFeed(
            source=source,
            event_bus_callback=event_bus_callback,
        )
    return _macro_feed_instance


def reset_macro_feed():
    """重置单例"""
    global _macro_feed_instance
    if _macro_feed_instance:
        _macro_feed_instance.stop()
    _macro_feed_instance = None

"""
================================================================================
HMATS v6.5 - Trading Economics Data Feed (P1 Optional)
================================================================================

Trading Economics API integration for structured economic calendar.

Provides:
1. Structured economic calendar with importance levels
2. Actual vs forecast data for macro shock calculation
3. Event importance classification (high/medium/low)

Purpose (STRICT):
- Encode macro event windows
- Compute macro_shock when actual vs forecast available
- Decide WHEN it is safe to SCALE UP (when shock aligned with trend)
- NEVER used as a hard gate
- NEVER overrides trade direction

CRITICAL CONSTRAINTS:
- This is a P1 (OPTIONAL) data source
- Enhances FRED event windows with structured importance
- Provides macro_shock calculation when data available
- Falls back gracefully when unavailable

API Docs: https://tradingeconomics.com/api

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
import os
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Key economic events to track
KEY_EVENTS = {
    "CPI": {"keywords": ["cpi", "consumer price"], "importance": 3},
    "PCE": {"keywords": ["pce", "personal consumption"], "importance": 3},
    "NFP": {"keywords": ["nonfarm", "non-farm", "payroll"], "importance": 3},
    "FOMC": {"keywords": ["fomc", "fed rate", "federal funds"], "importance": 3},
    "GDP": {"keywords": ["gdp", "gross domestic"], "importance": 2},
    "UNEMPLOYMENT": {"keywords": ["unemployment", "jobless"], "importance": 2},
    "RETAIL": {"keywords": ["retail sales"], "importance": 2},
    "HOUSING": {"keywords": ["housing", "home sales"], "importance": 2},
    "PMI": {"keywords": ["pmi", "purchasing manager"], "importance": 2},
    "ISM": {"keywords": ["ism"], "importance": 2},
}

# Importance mapping
IMPORTANCE_WEIGHT = {
    1: 0.33,  # low
    2: 0.66,  # medium  
    3: 1.0,   # high
}

# Macro shock sensitivity
MACRO_SHOCK_K = 2.0  # tanh sensitivity


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EventImportance(Enum):
    """Event importance level."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class EconomicEvent:
    """Single economic event from calendar."""
    event_id: str
    name: str
    country: str
    category: str
    importance: EventImportance
    
    # Timing
    event_time: datetime
    
    # Values
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    
    # Derived
    surprise: Optional[float] = None  # actual - forecast
    surprise_pct: Optional[float] = None  # (actual - forecast) / |forecast|
    
    def compute_surprise(self):
        """Compute surprise metrics if actual and forecast available."""
        if self.actual is not None and self.forecast is not None:
            self.surprise = self.actual - self.forecast
            if abs(self.forecast) > 1e-6:
                self.surprise_pct = self.surprise / abs(self.forecast)
            else:
                self.surprise_pct = self.surprise if self.surprise != 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "country": self.country,
            "importance": self.importance.value,
            "event_time": self.event_time.isoformat(),
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "surprise": self.surprise,
            "surprise_pct": self.surprise_pct,
        }


@dataclass
class TradingEconomicsData:
    """Aggregated Trading Economics data."""
    timestamp: datetime
    staleness_sec: float
    
    # Upcoming events
    upcoming_events: List[EconomicEvent] = field(default_factory=list)
    
    # Recent events (with actual values)
    recent_events: List[EconomicEvent] = field(default_factory=list)
    
    # Nearest high-importance event
    nearest_high_importance: Optional[EconomicEvent] = None
    
    # Computed macro shock (from most recent high-importance event)
    macro_shock: float = 0.0  # [-1, 1]
    
    # Event window status
    event_window_active: bool = False
    event_window_name: Optional[str] = None
    event_window_importance: int = 0
    t_minus_min: int = 0  # Minutes until event
    t_plus_min: int = 0   # Minutes since event
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "macro_shock": self.macro_shock,
            "event_window_active": self.event_window_active,
            "event_window_name": self.event_window_name,
            "event_window_importance": self.event_window_importance,
            "t_minus_min": self.t_minus_min,
            "t_plus_min": self.t_plus_min,
            "upcoming_count": len(self.upcoming_events),
            "recent_count": len(self.recent_events),
        }


# =============================================================================
# TRADING ECONOMICS FEED
# =============================================================================

class TradingEconomicsFeed:
    """
    Trading Economics API feed for economic calendar.
    
    Provides structured event data with importance classification
    and macro shock calculation.
    
    CRITICAL: This is a P1 (optional) enhancement layer.
    Falls back gracefully when unavailable.
    """
    
    BASE_URL = "https://api.tradingeconomics.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_sec: float = 900,  # 15 minutes
        event_bus_callback: Optional[Callable] = None,
        mock_mode: bool = False,
        event_window_minutes: int = 180,  # 3 hours
    ):
        self.api_key = api_key or os.environ.get("TRADING_ECONOMICS_API_KEY", "")
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        self._mock_mode = mock_mode
        self._event_window_minutes = event_window_minutes
        
        # Internal state
        self._last_data: Optional[TradingEconomicsData] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        
        # Cache for recent shocks (to persist across fetches)
        self._shock_cache: Dict[str, float] = {}
        self._shock_cache_expiry: Dict[str, datetime] = {}
        self._shock_cache_duration_hours = 4  # Keep shocks for 4 hours
        
        if not self.api_key and not mock_mode:
            logger.warning("[TE_FEED] No API key. Set TRADING_ECONOMICS_API_KEY env var.")
        
        logger.info(f"[TE_FEED] Initialized: mock={mock_mode}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """Start polling loop."""
        if self._running:
            return
        
        self._running = True
        logger.info("[TE_FEED] Started")
        
        while self._running:
            try:
                data = await self.fetch()
                if data and self._event_bus_callback:
                    self._event_bus_callback("TE_DATA", data.to_dict())
            except Exception as e:
                logger.error(f"[TE_FEED] Fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """Stop the feed."""
        self._running = False
        logger.info("[TE_FEED] Stopped")
    
    async def fetch(self) -> Optional[TradingEconomicsData]:
        """Fetch latest Trading Economics data."""
        try:
            if self._mock_mode:
                return await self._fetch_mock()
            
            if not self.api_key:
                logger.debug("[TE_FEED] No API key, using mock")
                return await self._fetch_mock()
            
            return await self._fetch_real()
            
        except Exception as e:
            logger.error(f"[TE_FEED] Fetch failed: {e}")
            self._fetch_errors += 1
            return self._last_data
    
    def get_latest(self) -> Optional[TradingEconomicsData]:
        """Get cached data."""
        return self._last_data
    
    def get_macro_context(self) -> Dict[str, Any]:
        """
        Get macro context for integration with profit_max_adapter.
        
        Returns standardized macro context that can enhance FRED data.
        """
        if not self._last_data:
            return {
                "event_window": {
                    "active": False,
                    "name": None,
                    "t_minus_min": 0,
                    "t_plus_min": 0,
                    "importance": 0,
                },
                "macro_shock": 0.0,
                "source": "trading_economics",
                "available": False,
            }
        
        data = self._last_data
        
        return {
            "event_window": {
                "active": data.event_window_active,
                "name": data.event_window_name,
                "t_minus_min": data.t_minus_min,
                "t_plus_min": data.t_plus_min,
                "importance": data.event_window_importance,
            },
            "macro_shock": data.macro_shock,
            "source": "trading_economics",
            "available": True,
        }
    
    # =========================================================================
    # REAL API FETCH
    # =========================================================================
    
    async def _fetch_real(self) -> Optional[TradingEconomicsData]:
        """Fetch from real Trading Economics API."""
        now = datetime.now(timezone.utc)
        data = TradingEconomicsData(timestamp=now, staleness_sec=0.0)
        
        async with create_session() as session:
            # Fetch calendar for next 7 days
            upcoming = await self._fetch_calendar(session, days_ahead=7)
            data.upcoming_events = upcoming
            
            # Fetch recent events (last 24 hours)
            recent = await self._fetch_calendar(session, days_back=1)
            data.recent_events = recent
            
            # Compute derived values
            self._compute_event_window(data, now)
            self._compute_macro_shock(data, now)
        
        self._last_data = data
        self._last_fetch_time = now
        
        return data
    
    async def _fetch_calendar(
        self,
        session: aiohttp.ClientSession,
        days_ahead: int = 0,
        days_back: int = 0,
    ) -> List[EconomicEvent]:
        """Fetch economic calendar."""
        # Build date range
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days_back)
        end_date = now + timedelta(days=days_ahead)
        
        url = f"{self.BASE_URL}/calendar/country/united states"
        params = {
            "c": self.api_key,
            "d1": start_date.strftime("%Y-%m-%d"),
            "d2": end_date.strftime("%Y-%m-%d"),
        }
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"[TE_FEED] Calendar fetch failed: {resp.status}")
                    return []
                
                items = await resp.json()
                
                events = []
                for item in items:
                    event = self._parse_event(item)
                    if event:
                        events.append(event)
                
                return events
                
        except asyncio.TimeoutError:
            logger.error("[TE_FEED] Calendar fetch timed out (10s)")
            return []
        except Exception as e:
            logger.error(f"[TE_FEED] Calendar fetch error: {e}")
            return []
    
    def _parse_event(self, item: Dict) -> Optional[EconomicEvent]:
        """Parse a single event from API response."""
        try:
            name = item.get("Event", "")
            
            # Determine importance
            importance = self._classify_importance(name, item)
            
            # Parse event time
            event_time_str = item.get("Date", "")
            if event_time_str:
                event_time = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
            else:
                return None
            
            event = EconomicEvent(
                event_id=str(item.get("CalendarId", "")),
                name=name,
                country=item.get("Country", "US"),
                category=item.get("Category", ""),
                importance=importance,
                event_time=event_time,
                actual=item.get("Actual"),
                forecast=item.get("Forecast"),
                previous=item.get("Previous"),
            )
            
            event.compute_surprise()
            
            return event
            
        except Exception as e:
            logger.debug(f"[TE_FEED] Event parse error: {e}")
            return None
    
    def _classify_importance(self, name: str, item: Dict) -> EventImportance:
        """Classify event importance."""
        name_lower = name.lower()
        
        # Check against known important events
        for event_type, config in KEY_EVENTS.items():
            if any(kw in name_lower for kw in config["keywords"]):
                return EventImportance(config["importance"])
        
        # Use API importance if available
        api_importance = item.get("Importance")
        if api_importance:
            if api_importance >= 3:
                return EventImportance.HIGH
            elif api_importance >= 2:
                return EventImportance.MEDIUM
        
        return EventImportance.LOW
    
    # =========================================================================
    # COMPUTATIONS
    # =========================================================================
    
    def _compute_event_window(self, data: TradingEconomicsData, now: datetime):
        """Compute event window status."""
        window_min = self._event_window_minutes
        nearest_event: Optional[EconomicEvent] = None
        nearest_delta_min = float('inf')
        
        # Check all events
        all_events = data.upcoming_events + data.recent_events
        
        for event in all_events:
            # Only consider medium/high importance
            if event.importance.value < 2:
                continue
            
            delta_sec = (event.event_time - now).total_seconds()
            delta_min = delta_sec / 60
            
            # Check if within window
            if -window_min <= delta_min <= window_min:
                if abs(delta_min) < abs(nearest_delta_min):
                    nearest_delta_min = delta_min
                    nearest_event = event
        
        if nearest_event:
            data.event_window_active = True
            data.event_window_name = nearest_event.name
            data.event_window_importance = nearest_event.importance.value
            data.t_minus_min = int(nearest_delta_min) if nearest_delta_min > 0 else 0
            data.t_plus_min = int(-nearest_delta_min) if nearest_delta_min < 0 else 0
            data.nearest_high_importance = nearest_event
        else:
            data.event_window_active = False
    
    def _compute_macro_shock(self, data: TradingEconomicsData, now: datetime):
        """
        Compute macro_shock from recent high-importance events.
        
        Formula (TASK D1):
        shock_raw = (actual - forecast) / max(|forecast|, eps)
        macro_shock = tanh(shock_raw * k) clipped to [-1, 1]
        """
        # Clean expired cache entries
        self._clean_shock_cache(now)
        
        # Find most recent high-importance event with actual value
        best_shock = 0.0
        best_event_name = None
        
        for event in data.recent_events:
            if event.importance.value < 3:
                continue
            
            if event.surprise_pct is not None:
                # Compute shock
                shock_raw = event.surprise_pct
                shock = math.tanh(shock_raw * MACRO_SHOCK_K)
                shock = max(-1.0, min(1.0, shock))
                
                # Cache it
                self._shock_cache[event.event_id] = shock
                self._shock_cache_expiry[event.event_id] = now + timedelta(
                    hours=self._shock_cache_duration_hours
                )
                
                # Track best (most recent high-importance)
                if abs(shock) > abs(best_shock):
                    best_shock = shock
                    best_event_name = event.name
        
        # If no recent shocks, check cache
        if best_shock == 0.0 and self._shock_cache:
            for event_id, shock in self._shock_cache.items():
                if abs(shock) > abs(best_shock):
                    best_shock = shock
        
        data.macro_shock = best_shock
        
        if best_shock != 0.0:
            logger.debug(f"[TE_FEED] Macro shock: {best_shock:.3f} from {best_event_name}")
    
    def _clean_shock_cache(self, now: datetime):
        """Remove expired shock cache entries."""
        expired = [
            eid for eid, expiry in self._shock_cache_expiry.items()
            if expiry < now
        ]
        for eid in expired:
            del self._shock_cache[eid]
            del self._shock_cache_expiry[eid]
    
    # =========================================================================
    # MOCK DATA
    # =========================================================================
    
    async def _fetch_mock(self) -> TradingEconomicsData:
        """Generate mock data for testing.

        [P113 (3/6) 2026-04-27] WARN signal for mock fallback visibility.
        P101 prevention.
        """
        if not getattr(self, '_mock_warned', False):
            logger.warning(
                "[TRADING_ECON] _fetch_mock invoked — calendar events are "
                "SYNTHETIC. has_high_impact_event_soon() will report mock "
                "events. Verify TRADING_ECONOMICS_API_KEY configured."
            )
            self._mock_warned = True
        await asyncio.sleep(0.1)

        import random
        now = datetime.now(timezone.utc)

        data = TradingEconomicsData(timestamp=now, staleness_sec=0.0)
        
        # Generate some mock upcoming events
        event_types = ["CPI Release", "NFP Report", "FOMC Minutes", "GDP Estimate"]
        
        for i in range(random.randint(1, 4)):
            hours_ahead = random.randint(1, 168)  # 1 hour to 7 days
            event_time = now + timedelta(hours=hours_ahead)
            
            importance = EventImportance.HIGH if random.random() < 0.3 else EventImportance.MEDIUM
            
            event = EconomicEvent(
                event_id=f"mock_{i}",
                name=random.choice(event_types),
                country="US",
                category="Economic",
                importance=importance,
                event_time=event_time,
                forecast=random.uniform(-0.5, 0.5) if random.random() > 0.3 else None,
                previous=random.uniform(-0.5, 0.5),
            )
            
            data.upcoming_events.append(event)
        
        # Maybe generate a recent event with actual value
        if random.random() < 0.3:
            hours_ago = random.randint(1, 12)
            event_time = now - timedelta(hours=hours_ago)
            
            actual = random.uniform(-0.3, 0.3)
            forecast = actual + random.uniform(-0.15, 0.15)
            
            event = EconomicEvent(
                event_id="mock_recent",
                name="CPI Release",
                country="US",
                category="Inflation",
                importance=EventImportance.HIGH,
                event_time=event_time,
                actual=actual,
                forecast=forecast,
                previous=random.uniform(-0.3, 0.3),
            )
            event.compute_surprise()
            
            data.recent_events.append(event)
        
        # Compute derived values
        self._compute_event_window(data, now)
        self._compute_macro_shock(data, now)
        
        self._last_data = data
        self._last_fetch_time = now
        
        return data


# =============================================================================
# SINGLETON
# =============================================================================

_te_feed_instance: Optional[TradingEconomicsFeed] = None


def get_trading_economics_feed(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> TradingEconomicsFeed:
    """Get or create TradingEconomicsFeed singleton."""
    global _te_feed_instance
    if _te_feed_instance is None:
        _te_feed_instance = TradingEconomicsFeed(
            api_key=api_key,
            mock_mode=mock_mode,
        )
    return _te_feed_instance


def reset_trading_economics_feed():
    """Reset singleton."""
    global _te_feed_instance
    if _te_feed_instance:
        _te_feed_instance.stop()
    _te_feed_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 70)
        print("Trading Economics Feed Test")
        print("=" * 70)
        
        feed = TradingEconomicsFeed(mock_mode=True)
        data = await feed.fetch()
        
        print(f"\nEvent Window Active: {data.event_window_active}")
        print(f"Event Window Name: {data.event_window_name}")
        print(f"T-Minus (min): {data.t_minus_min}")
        print(f"Macro Shock: {data.macro_shock:.3f}")
        
        print(f"\nUpcoming Events: {len(data.upcoming_events)}")
        for event in data.upcoming_events[:3]:
            print(f"  - {event.name}: importance={event.importance.value}")
        
        print("\nMacro Context:")
        ctx = feed.get_macro_context()
        for k, v in ctx.items():
            print(f"  {k}: {v}")
        
        print("\nON Test passed!")
    
    asyncio.run(test())

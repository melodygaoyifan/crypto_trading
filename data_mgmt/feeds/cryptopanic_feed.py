"""
================================================================================
HMATS v6.5 - CryptoPanic Data Feed
================================================================================

CryptoPanic API integration for news aggregation and panic detection.

Provides:
1. News aggregation with community votes
2. Panic score calculation
3. Narrative intensity tracking
4. News velocity metrics

Purpose (STRICT):
- Detect panic acceleration / exhaustion
- Measure crowd agreement vs fragmentation
- NOT used for trade direction

API Docs: https://cryptopanic.com/developers/api/

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta, timezone
import os
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

SUPPORTED_CURRENCIES = ["BTC", "ETH", "SOL"]

# Panic thresholds
PANIC_HIGH_THRESHOLD = 0.7
PANIC_EXTREME_THRESHOLD = 0.85

# News velocity window (hours)
VELOCITY_WINDOW_HOURS = 4


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NewsItem:
    """Single news item from CryptoPanic."""
    id: str
    title: str
    source: str
    url: str
    published_at: datetime
    currencies: List[str]
    
    # Vote counts
    votes_positive: int
    votes_negative: int
    votes_important: int
    votes_liked: int
    votes_disliked: int
    votes_lol: int
    votes_toxic: int
    votes_saved: int
    
    # Computed sentiment
    sentiment_score: float  # -1 to 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "sentiment_score": self.sentiment_score,
            "published_at": self.published_at.isoformat(),
        }


@dataclass
class CryptoPanicData:
    """Aggregated CryptoPanic data."""
    timestamp: datetime
    staleness_sec: float
    
    # Recent news items
    recent_news: List[NewsItem] = field(default_factory=list)
    
    # Per-currency metrics
    panic_score: Dict[str, float] = field(default_factory=dict)  # [0, 1]
    news_velocity: Dict[str, float] = field(default_factory=dict)  # normalized
    sentiment_consensus: Dict[str, float] = field(default_factory=dict)  # -1 to 1
    narrative_intensity: Dict[str, float] = field(default_factory=dict)  # [0, 1]
    
    # Global metrics
    global_panic: float = 0.0
    global_news_velocity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "panic_score": self.panic_score,
            "news_velocity": self.news_velocity,
            "sentiment_consensus": self.sentiment_consensus,
            "narrative_intensity": self.narrative_intensity,
            "global_panic": self.global_panic,
            "news_count": len(self.recent_news),
        }


# =============================================================================
# CRYPTOPANIC FEED
# =============================================================================

class CryptoPanicFeed:
    """
    CryptoPanic API feed for news panic detection.
    
    Measures panic level and crowd fragmentation.
    """

    BASE_URL = "https://cryptopanic.com/api/growth/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_sec: float = 300,  # 5 minutes
        event_bus_callback: Optional[Callable] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key or os.environ.get("CRYPTOPANIC_API_KEY", "")
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        self._mock_mode = mock_mode
        
        # Internal state
        self._last_data: Optional[CryptoPanicData] = None
        self._last_fetch_time: Optional[datetime] = None
        self._backoff_until: Optional[datetime] = None
        self._last_status_code: Optional[int] = None
        self._running = False
        self._fetch_errors = 0
        
        # History for velocity calculation
        self._news_count_history: deque = deque(maxlen=24)  # 2 hours at 5min
        self._panic_history: Dict[str, deque] = {
            c: deque(maxlen=24) for c in SUPPORTED_CURRENCIES
        }
        
        if not self.api_key and not mock_mode:
            logger.warning("[CRYPTOPANIC] No API key. Set CRYPTOPANIC_API_KEY env var.")
        
        logger.info(f"[CRYPTOPANIC] Initialized: mock={mock_mode}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """Start polling loop."""
        if self._running:
            return
        
        self._running = True
        logger.info("[CRYPTOPANIC] Started")
        
        while self._running:
            try:
                data = await self.fetch()
                if data and self._event_bus_callback:
                    self._event_bus_callback("CRYPTOPANIC_DATA", data.to_dict())
            except Exception as e:
                logger.error(f"[CRYPTOPANIC] Fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """Stop the feed."""
        self._running = False
        logger.info("[CRYPTOPANIC] Stopped")
    
    async def fetch(self) -> Optional[CryptoPanicData]:
        """Fetch latest CryptoPanic data."""
        try:
            now = datetime.now(timezone.utc)
            if self._mock_mode:
                return await self._fetch_mock()

            if not self.api_key:
                logger.warning("[CRYPTOPANIC] No API key, using mock")
                return await self._fetch_mock()

            if (
                self._backoff_until is not None
                and now < self._backoff_until
            ):
                logger.debug(
                    "[CRYPTOPANIC] Backoff active until %s, using cached data",
                    self._backoff_until.isoformat(),
                )
                return self._last_data

            if (
                self._last_fetch_time is not None
                and (now - self._last_fetch_time).total_seconds() < self.poll_interval_sec
            ):
                return self._last_data

            return await self._fetch_real()

        except Exception as e:
            logger.error(f"[CRYPTOPANIC] Fetch failed: {e}")
            self._fetch_errors += 1
            return self._last_data
    
    def get_latest(self) -> Optional[CryptoPanicData]:
        """Get cached data."""
        return self._last_data
    
    def get_panic_metrics(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get panic metrics for integration.
        """
        if not self._last_data:
            return self._get_default_panic_metrics()
        
        data = self._last_data
        
        return {
            "panic_score": data.panic_score.get(symbol, 0.0),
            "news_velocity": data.news_velocity.get(symbol, 0.0),
            "sentiment_consensus": data.sentiment_consensus.get(symbol, 0.0),
            "narrative_intensity": data.narrative_intensity.get(symbol, 0.5),
            "extreme_panic": data.panic_score.get(symbol, 0.0) >= PANIC_EXTREME_THRESHOLD,
        }
    
    def _get_default_panic_metrics(self) -> Dict[str, Any]:
        """Default metrics when no data available."""
        return {
            "panic_score": 0.0,
            "news_velocity": 0.0,
            "sentiment_consensus": 0.0,
            "narrative_intensity": 0.5,
            "extreme_panic": False,
        }
    
    # =========================================================================
    # REAL API
    # =========================================================================
    
    async def _fetch_real(self) -> CryptoPanicData:
        """Fetch real data from CryptoPanic."""
        now = datetime.now(timezone.utc)
        
        data = CryptoPanicData(
            timestamp=now,
            staleness_sec=0.0,
        )
        
        async with create_session() as session:
            # Fetch posts for each currency
            for currency in SUPPORTED_CURRENCIES:
                try:
                    news = await self._fetch_posts(session, currency)
                    data.recent_news.extend(news)
                    if self._last_status_code == 429:
                        break
                except Exception as e:
                    logger.warning(f"[CRYPTOPANIC] {currency}: {e}")
        
        # Compute metrics
        self._compute_metrics(data)
        
        self._last_data = data
        self._last_fetch_time = now
        
        return data
    
    async def _fetch_posts(
        self,
        session: aiohttp.ClientSession,
        currency: str,
    ) -> List[NewsItem]:
        """Fetch posts for a currency."""
        url = f"{self.BASE_URL}/posts/"
        params = {
            "auth_token": self.api_key,
            "currencies": currency,
            "filter": "all",
            "public": "true",
        }
        
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            self._last_status_code = resp.status
            if resp.status != 200:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    retry_after_sec = 900
                    if retry_after:
                        try:
                            retry_after_sec = max(int(float(retry_after)), 60)
                        except (TypeError, ValueError):
                            retry_after_sec = 900
                    self._backoff_until = datetime.now(timezone.utc) + timedelta(seconds=retry_after_sec)
                logger.warning(
                    f"[CRYPTOPANIC] {currency}: HTTP {resp.status} from API"
                )
                return []

            self._backoff_until = None
            result = await resp.json()
            posts = result.get("results", [])
            if not posts:
                logger.debug(f"[CRYPTOPANIC] {currency}: API returned 0 posts")
            
            news_items = []
            for post in posts[:20]:  # Limit to recent 20
                votes = post.get("votes", {})
                
                # Calculate sentiment from votes
                positive = votes.get("positive", 0)
                negative = votes.get("negative", 0)
                total_votes = positive + negative
                
                if total_votes > 0:
                    sentiment_score = (positive - negative) / total_votes
                else:
                    sentiment_score = 0.0
                
                # Parse published date
                pub_str = post.get("published_at", "")
                try:
                    pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pub_date = datetime.now(timezone.utc)
                
                news_items.append(NewsItem(
                    id=str(post.get("id", "")),
                    title=post.get("title", ""),
                    source=post.get("source", {}).get("title", "unknown"),
                    url=post.get("url", ""),
                    published_at=pub_date,
                    currencies=[c.get("code", "") for c in post.get("currencies", [])],
                    votes_positive=positive,
                    votes_negative=negative,
                    votes_important=votes.get("important", 0),
                    votes_liked=votes.get("liked", 0),
                    votes_disliked=votes.get("disliked", 0),
                    votes_lol=votes.get("lol", 0),
                    votes_toxic=votes.get("toxic", 0),
                    votes_saved=votes.get("saved", 0),
                    sentiment_score=sentiment_score,
                ))
            
            return news_items
    
    # =========================================================================
    # METRICS COMPUTATION
    # =========================================================================
    
    def _compute_metrics(self, data: CryptoPanicData):
        """Compute panic and narrative metrics."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=VELOCITY_WINDOW_HOURS)
        
        for currency in SUPPORTED_CURRENCIES:
            # Filter news for this currency
            currency_news = [
                n for n in data.recent_news
                if currency in n.currencies and n.published_at >= cutoff
            ]
            
            if not currency_news:
                data.panic_score[currency] = 0.0
                data.news_velocity[currency] = 0.0
                data.sentiment_consensus[currency] = 0.0
                data.narrative_intensity[currency] = 0.5
                continue
            
            # === Panic Score ===
            # Based on negative sentiment ratio and toxic votes
            total_positive = sum(n.votes_positive for n in currency_news)
            total_negative = sum(n.votes_negative for n in currency_news)
            total_toxic = sum(n.votes_toxic for n in currency_news)
            total_votes = total_positive + total_negative
            
            if total_votes > 0:
                # Negative ratio contributes to panic
                negative_ratio = total_negative / total_votes
                # Toxic votes indicate fear/uncertainty
                toxic_ratio = total_toxic / max(total_votes, 1)
                
                panic_score = negative_ratio * 0.7 + min(toxic_ratio * 3, 0.3)
                panic_score = np.clip(panic_score, 0.0, 1.0)
            else:
                panic_score = 0.0
            
            data.panic_score[currency] = panic_score
            self._panic_history[currency].append(panic_score)
            
            # === News Velocity ===
            # Number of news items per hour
            hours = VELOCITY_WINDOW_HOURS
            velocity = len(currency_news) / hours
            # Normalize: 5 news/hour = 0.5, 20+ = 1.0
            normalized_velocity = np.clip(velocity / 20, 0.0, 1.0)
            data.news_velocity[currency] = normalized_velocity
            
            # === Sentiment Consensus ===
            # Average sentiment score
            avg_sentiment = np.mean([n.sentiment_score for n in currency_news])
            data.sentiment_consensus[currency] = avg_sentiment
            
            # === Narrative Intensity ===
            # Combination of velocity and engagement
            total_engagement = sum(
                n.votes_positive + n.votes_negative + n.votes_important
                for n in currency_news
            )
            engagement_per_news = total_engagement / max(len(currency_news), 1)
            # Normalize: 50 engagements = 0.5, 200+ = 1.0
            normalized_engagement = np.clip(engagement_per_news / 200, 0.0, 1.0)
            
            narrative_intensity = (normalized_velocity + normalized_engagement) / 2
            data.narrative_intensity[currency] = narrative_intensity
        
        # Update global metrics
        self._news_count_history.append(len(data.recent_news))
        
        if data.panic_score:
            data.global_panic = np.mean(list(data.panic_score.values()))
        
        if data.news_velocity:
            data.global_news_velocity = np.mean(list(data.news_velocity.values()))
    
    # =========================================================================
    # MOCK DATA
    # =========================================================================
    
    async def _fetch_mock(self) -> CryptoPanicData:
        """Generate mock data.

        [P113 (3/6) 2026-04-27] WARN-level signal so production operator
        can detect when this fallback is taken. Silent mock fallback was
        the P101 bug class. Test test_mock_fallback_signaling.py enforces
        this contract.
        """
        if not getattr(self, '_mock_warned', False):
            logger.warning(
                "[CRYPTOPANIC] _fetch_mock invoked — production using "
                "mock data. Verify API key configured + endpoint reachable."
            )
            self._mock_warned = True
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        data = CryptoPanicData(timestamp=now, staleness_sec=0.0)
        
        import random
        
        # Generate mock news
        for currency in SUPPORTED_CURRENCIES:
            for i in range(random.randint(3, 10)):
                sentiment = random.uniform(-0.5, 0.5)
                pos_votes = random.randint(5, 50)
                neg_votes = random.randint(5, 50)
                
                data.recent_news.append(NewsItem(
                    id=f"mock_{currency}_{i}",
                    title=f"Mock {currency} News {i}",
                    source="MockSource",
                    url="https://example.com",
                    published_at=now - timedelta(minutes=random.randint(1, 240)),
                    currencies=[currency],
                    votes_positive=pos_votes,
                    votes_negative=neg_votes,
                    votes_important=random.randint(0, 20),
                    votes_liked=random.randint(0, 30),
                    votes_disliked=random.randint(0, 10),
                    votes_lol=random.randint(0, 5),
                    votes_toxic=random.randint(0, 5),
                    votes_saved=random.randint(0, 10),
                    sentiment_score=sentiment,
                ))
        
        self._compute_metrics(data)
        self._last_data = data
        self._last_fetch_time = now
        
        return data


# =============================================================================
# SINGLETON
# =============================================================================

_cryptopanic_feed_instance: Optional[CryptoPanicFeed] = None


def get_cryptopanic_feed(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> CryptoPanicFeed:
    """Get or create CryptoPanicFeed singleton."""
    global _cryptopanic_feed_instance
    if _cryptopanic_feed_instance is None:
        _cryptopanic_feed_instance = CryptoPanicFeed(
            api_key=api_key,
            mock_mode=mock_mode,
        )
    return _cryptopanic_feed_instance


def reset_cryptopanic_feed():
    """Reset singleton."""
    global _cryptopanic_feed_instance
    if _cryptopanic_feed_instance:
        _cryptopanic_feed_instance.stop()
    _cryptopanic_feed_instance = None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=" * 70)
        print("CryptoPanic Feed Test")
        print("=" * 70)
        
        feed = CryptoPanicFeed(mock_mode=True)
        data = await feed.fetch()
        
        print(f"\nPanic Score: {data.panic_score}")
        print(f"News Velocity: {data.news_velocity}")
        print(f"Sentiment Consensus: {data.sentiment_consensus}")
        print(f"Narrative Intensity: {data.narrative_intensity}")
        print(f"Global Panic: {data.global_panic:.3f}")
        
        print("\nBTC Panic Metrics:")
        metrics = feed.get_panic_metrics("BTC")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        print("\nON Test passed!")
    
    asyncio.run(test())

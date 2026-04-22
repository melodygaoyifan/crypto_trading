"""CryptoCompare News feed — aggregates crypto news headlines.

Endpoint: https://data-api.cryptocompare.com/news/v1/article/list
Auth: CRYPTOCOMPARE_API_KEY (existing key, no extra cost)
Rate limit: shared with CC account (Growth plan: plenty for 4H cadence)

Purpose:
  Second news source alongside CryptoPanic. When CryptoPanic is rate-limited
  or down, CC News provides redundancy. Unified output: list of NewsItem.

Usage:
  feed = get_cc_news_feed()
  headlines = await feed.fetch_headlines(asset="BTC", limit=20)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from data_mgmt.feeds._http import create_session

logger = logging.getLogger("CCNews")

BASE_URL = "https://data-api.cryptocompare.com/news/v1/article/list"
MIN_FETCH_INTERVAL = 300.0  # 5 min cache (news changes slowly vs 4H cadence)


@dataclass
class CCNewsItem:
    id: int
    title: str
    body: str
    url: str
    source: str
    published_at: datetime
    categories: List[str] = field(default_factory=list)
    sentiment: str = "NEUTRAL"  # CC's sentiment tag: POSITIVE/NEUTRAL/NEGATIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body[:500],
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "categories": self.categories,
            "sentiment": self.sentiment,
        }


class CCNewsFeed:
    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("CRYPTOCOMPARE_API_KEY", "")
        self._mock_mode = not bool(self._api_key)
        self._cache: Dict[str, tuple[float, List[CCNewsItem]]] = {}
        self._last_error = ""
        if self._mock_mode:
            logger.warning("[CC_NEWS] MOCK mode (no key)")
        else:
            logger.info(f"[CC_NEWS] LIVE (key=...{self._api_key[-4:]})")

    async def fetch_headlines(self, asset: str = "BTC", limit: int = 20) -> List[CCNewsItem]:
        """Fetch recent news for an asset. Uses 5-min cache per asset."""
        cache_key = asset.upper()
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < MIN_FETCH_INTERVAL:
            return cached[1]

        if self._mock_mode:
            return []

        params = {
            "lang": "EN",
            "api_key": self._api_key,
            "categories": asset.upper(),
            "limit": str(limit),
        }
        try:
            async with create_session() as session:
                async with session.get(BASE_URL, params=params, timeout=10) as resp:
                    if resp.status != 200:
                        self._last_error = f"HTTP {resp.status}"
                        logger.warning(f"[CC_NEWS] {asset}: {self._last_error}")
                        return []
                    data = await resp.json()

            items: List[CCNewsItem] = []
            for row in data.get("Data", [])[:limit]:
                try:
                    items.append(CCNewsItem(
                        id=int(row.get("ID", 0) or 0),
                        title=str(row.get("TITLE", "") or ""),
                        body=str(row.get("BODY", "") or "")[:1000],
                        url=str(row.get("URL", "") or ""),
                        source=str(row.get("SOURCE_DATA", {}).get("NAME") if isinstance(row.get("SOURCE_DATA"), dict) else row.get("SOURCE_ID", "") or ""),
                        published_at=datetime.fromtimestamp(
                            int(row.get("PUBLISHED_ON", 0) or 0), tz=timezone.utc,
                        ),
                        categories=[c.get("NAME", "") for c in (row.get("CATEGORY_DATA") or []) if isinstance(c, dict)],
                        sentiment=str(row.get("SENTIMENT", "NEUTRAL") or "NEUTRAL").upper(),
                    ))
                except Exception as e:
                    logger.debug(f"[CC_NEWS] row parse skipped: {e}")
                    continue

            self._cache[cache_key] = (now, items)
            return items
        except Exception as e:
            self._last_error = str(e)
            logger.warning(f"[CC_NEWS] {asset}: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": not self._mock_mode,
            "mock": self._mock_mode,
            "last_error": self._last_error,
            "cached_assets": list(self._cache.keys()),
        }


_instance: Optional[CCNewsFeed] = None


def get_cc_news_feed(api_key: str = "") -> CCNewsFeed:
    global _instance
    if _instance is None:
        _instance = CCNewsFeed(api_key=api_key)
    return _instance

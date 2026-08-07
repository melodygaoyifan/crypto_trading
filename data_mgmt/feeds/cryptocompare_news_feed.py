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
    # [P216] Bump when the serialized shape changes (P153 pattern).
    _STATE_VERSION = "ccnews_cache_v1"
    # Fallback wait when the server sends a 429 with no Retry-After header —
    # which is what it actually does here (`Retry-After=None` in the live logs).
    _DEFAULT_BACKOFF_SEC = 900.0

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("CRYPTOCOMPARE_API_KEY", "")
        self._mock_mode = not bool(self._api_key)
        self._cache: Dict[str, tuple[float, List[CCNewsItem]]] = {}
        self._last_error = ""
        # [P216] Epoch after which requests may resume. Zero = not backed off.
        self._backoff_until: float = 0.0
        if self._mock_mode:
            logger.warning("[CC_NEWS] MOCK mode (no key)")
        else:
            self._restore_state()
            logger.info(f"[CC_NEWS] LIVE (key=...{self._api_key[-4:]})")

    # ----- [P216] persistence: the backoff must survive a restart -----------
    def _data_dir(self) -> str:
        return os.environ.get("HMATS_DATA_DIR", "data")

    def _state_path(self) -> str:
        return os.path.join(self._data_dir(), "ccnews_state.json")

    def _restore_state(self) -> None:
        """Restore the backoff before the first fetch. An in-RAM backoff is not
        a rate limit: it re-arms on every restart, and restart-heavy failure
        modes are exactly when you most need it to hold (P154, same lesson)."""
        try:
            import json as _json
            p = self._state_path()
            if not os.path.exists(p):
                return
            with open(p, "r", encoding="utf-8") as fh:
                st = _json.load(fh)
            if st.get("state_version") != self._STATE_VERSION:
                logger.info("[CC_NEWS] state version mismatch — discarding")
                return
            self._backoff_until = float(st.get("backoff_until") or 0.0)
            _left = self._backoff_until - time.time()
            if _left > 0:
                logger.warning(
                    f"[CC_NEWS] restored 429 backoff: {_left / 60.0:.1f} min "
                    f"remaining — not calling the API until it expires")
        except Exception as e:  # noqa: silent-swallow — a bad state file must not break startup
            logger.warning(f"[CC_NEWS] state restore failed: {type(e).__name__}: {e}")

    def _persist_state(self) -> None:
        """Atomic write. Never raises."""
        if self._mock_mode:
            return
        try:
            import json as _json
            p = self._state_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                _json.dump({"state_version": self._STATE_VERSION,
                            "backoff_until": self._backoff_until,
                            "saved_ts": time.time()}, fh)
            os.replace(tmp, p)
        except Exception as e:  # noqa: silent-swallow — telemetry must not break the tick
            logger.debug(f"[CC_NEWS] state persist failed: {type(e).__name__}: {e}")

    def backoff_remaining_sec(self) -> float:
        return max(0.0, self._backoff_until - time.time())

    async def fetch_headlines(self, asset: str = "BTC", limit: int = 20) -> List[CCNewsItem]:
        """Fetch recent news for an asset. Uses 5-min cache per asset."""
        cache_key = asset.upper()
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < MIN_FETCH_INTERVAL:
            return cached[1]

        if self._mock_mode:
            return []

        # [P216] Honour an active 429 backoff. Without this the feed re-hit the
        # API on the very next asset, seconds after being told to slow down:
        # 3 assets x every tick, forever, which is what kept it rate-limited.
        # Serve the last good cache while backed off rather than [] — stale
        # headlines beat no headlines, and `[]` makes llm_sentiment fall back to
        # f&g, which main.py:8529 deliberately treats as untradeable.
        _left = self._backoff_until - now
        if _left > 0:
            if cached:
                logger.debug(
                    f"[CC_NEWS] {asset}: backoff {_left / 60.0:.1f}min — "
                    f"serving cached ({len(cached[1])} items)")
                return cached[1]
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
                        if resp.status == 429:
                            from data_mgmt.feeds._http import parse_retry_after
                            _retry = parse_retry_after(resp.headers.get("Retry-After"))
                            # [P216] ACT on Retry-After instead of only printing
                            # it. It was parsed into the log line and thrown
                            # away — a computed-but-unenforced value (P144
                            # shape) — so the next call went straight back out.
                            _wait = float(_retry) if _retry else self._DEFAULT_BACKOFF_SEC
                            self._backoff_until = time.time() + _wait
                            self._persist_state()
                            self._last_error = f"HTTP 429 (Retry-After={_retry}s)"
                            logger.warning(
                                f"[CC_NEWS] {asset} rate-limited (429), "
                                f"Retry-After={_retry}s — backing off {_wait / 60.0:.1f}min "
                                f"(all assets); falling back to heuristic"
                            )
                        else:
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

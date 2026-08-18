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
# [P220] 5 min -> 12 h. At a 4H decision cadence a 5-minute cache expired
# before every tick, so this fetched 6x/day = ~180 calls/month against an
# account capped at 100/MONTH. 12h => 2 calls/day => ~60/month, which fits
# inside the shared budget alongside the on-chain feed. Headline sentiment
# is a slow variable; a 12h-old corpus is materially the same signal, and
# the alternative is no corpus at all.
MIN_FETCH_INTERVAL = 43200.0  # 12 h

# [P219] Categories requested in a single combined call. Must cover every asset
# the engine trades, or that asset falls back to an empty list and llm_sentiment
# drops to the f&g path that main.py:8529 treats as untradeable.
TRACKED_CATEGORIES = ("BTC", "ETH", "SOL")


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

            # [P293b] Restore the CACHE (timestamp + items), not just the
            # backoff. The 12h throttle is `now - cached[0] < MIN_FETCH_
            # INTERVAL`, and `self._cache` was RAM-only — so every process
            # start missed the cache and spent a call against an account
            # capped at 100/MONTH. Measured: 14 process starts across the
            # retained logs, and the shared budget was exhausted by Aug 10,
            # nine days into the month. Restoring the timestamp alone would
            # leave `cached` falsy and the throttle unable to bind, so the
            # items come back too.
            _cache = st.get("cache") or {}
            if isinstance(_cache, dict):
                for _cat, _entry in _cache.items():
                    try:
                        _ts = float(_entry.get("ts") or 0.0)
                        if _ts <= 0.0:
                            continue
                        _items = []
                        for _r in (_entry.get("items") or []):
                            _pub = _r.get("published_at")
                            _dt = (datetime.fromisoformat(_pub) if _pub
                                   else datetime.now(timezone.utc))
                            if _dt.tzinfo is None:
                                _dt = _dt.replace(tzinfo=timezone.utc)
                            _items.append(CCNewsItem(
                                id=int(_r.get("id", 0) or 0),
                                title=str(_r.get("title", "")),
                                body=str(_r.get("body", "")),
                                url=str(_r.get("url", "")),
                                source=str(_r.get("source", "")),
                                published_at=_dt,
                                categories=list(_r.get("categories") or []),
                                sentiment=str(_r.get("sentiment", "NEUTRAL")),
                            ))
                        self._cache[str(_cat)] = (_ts, _items)
                    except (TypeError, ValueError, AttributeError):  # noqa: silent-swallow
                        continue  # one corrupt category must not lose the rest
                if self._cache:
                    _newest = max(v[0] for v in self._cache.values())
                    logger.info(
                        f"[CC_NEWS] restored cache: {len(self._cache)} "
                        f"category(ies), age "
                        f"{(time.time() - _newest) / 3600.0:.1f}h of "
                        f"{MIN_FETCH_INTERVAL / 3600.0:.0f}h — this restart "
                        f"will NOT re-spend account quota")
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
                # [P293b] Persist the cache with the backoff so the 12h
                # throttle survives a restart (see _restore_state). Bodies
                # are truncated to keep the state file small; only the
                # timestamp is load-bearing for quota protection.
                _cache_out = {}
                for _cat, _entry in (self._cache or {}).items():
                    try:
                        _ts, _items = _entry
                    except (TypeError, ValueError):  # noqa: silent-swallow
                        continue
                    _cache_out[str(_cat)] = {
                        "ts": float(_ts),
                        "items": [{
                            "id": getattr(i, "id", 0),
                            "title": getattr(i, "title", "")[:300],
                            "body": getattr(i, "body", "")[:500],
                            "url": getattr(i, "url", ""),
                            "source": getattr(i, "source", ""),
                            "published_at": (
                                i.published_at.isoformat()
                                if getattr(i, "published_at", None) else None),
                            "categories": list(getattr(i, "categories", []) or []),
                            "sentiment": getattr(i, "sentiment", "NEUTRAL"),
                        } for i in (_items or [])[:20]],
                    }
                _json.dump({"state_version": self._STATE_VERSION,
                            "backoff_until": self._backoff_until,
                            "cache": _cache_out,
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
        # [P287] This gate runs BEFORE the quota reservation. It used to run
        # after try_consume(), so every cache-miss during an active backoff
        # reserved 1 call from the shared 90/month budget and then returned
        # WITHOUT making the HTTP request — reserved, never spent,
        # permanently counted against cc_news in the P220 ledger.
        # Reserve-then-call means: reserve only when a call will be made.
        _left = self._backoff_until - now
        if _left > 0:
            if cached:
                logger.debug(
                    f"[CC_NEWS] {asset}: backoff {_left / 60.0:.1f}min — "
                    f"serving cached ({len(cached[1])} items)")
                return cached[1]
            return []

        # [P220] Reserve against the SHARED account budget before spending a
        # call. One call now serves all tracked assets (P219), so this is 1.
        from data_mgmt.feeds._cc_quota import get_cc_quota
        if not get_cc_quota().try_consume(1, caller="cc_news"):
            return cached[1] if cached else []

        params = {
            "lang": "EN",
            "api_key": self._api_key,
            # [P219] Ask for ALL tracked assets in ONE call and populate every
            # asset's cache from the result. This was one request PER ASSET,
            # i.e. 3x the calls for the same headlines, against an account
            # capped at 100 calls/MONTH (measured: 283 used). Filtering happens
            # client-side below, which costs nothing.
            "categories": ",".join(TRACKED_CATEGORIES),
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

            # [P219] Fan the single response out to every tracked asset's cache
            # so the other two assets this tick are served without another call.
            for _cat in TRACKED_CATEGORIES:
                _hit = [
                    it for it in items
                    # A row with NO categories is general market news and is
                    # relevant to all of them — dropping it would quietly shrink
                    # the corpus and push headline_count back toward 0, which is
                    # the condition this whole change exists to lift.
                    if (not it.categories) or (_cat in {c.upper() for c in it.categories})
                ]
                self._cache[_cat] = (now, _hit)
            logger.info(
                f"[CC_NEWS] 1 call served {len(TRACKED_CATEGORIES)} assets: "
                + ", ".join(f"{c}={len(self._cache[c][1])}" for c in TRACKED_CATEGORIES)
            )
            # [P293b] Persist on SUCCESS, not only on 429. Without this the
            # restored cache would always be empty and the 12h throttle could
            # never bind across a restart.
            self._persist_state()
            return self._cache.get(cache_key, (now, items))[1]
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

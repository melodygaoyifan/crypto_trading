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
import json
import os
import time
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


# [P293b] Pure helpers for the monthly-quota case — unit-testable without a
# live session, so the distinction between "rate limited for a moment" and
# "out of budget until the month rolls" can be pinned by test.

_MONTHLY_QUOTA_MARKERS = (
    "monthly quota",
    "monthly limit",
    "quota exceeded",
)


def _is_monthly_quota_error(body: str) -> bool:
    """True when a 429 body says the MONTHLY allowance is spent.

    Deliberately conservative: an unrecognised 429 falls through to the
    ordinary short backoff, because mistaking a transient rate limit for a
    month-long outage would silence the feed for weeks. The failure
    direction is "retry sooner than necessary", never "go dark by mistake".
    """
    if not body:
        return False
    low = body.lower()
    return any(m in low for m in _MONTHLY_QUOTA_MARKERS)


# [P299] How long to wait before re-probing while a MONTHLY quota is
# exhausted. See quota_backoff_until() for why this is not "until the 1st".
QUOTA_REPROBE_SEC = 24 * 3600


def _next_month_start_utc(now: datetime) -> datetime:
    """00:00 UTC on the 1st of the following month."""
    year, month = now.year, now.month
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


def quota_backoff_until(now: datetime) -> datetime:
    """[P299] When to re-probe after a MONTHLY-quota 429.

    P293b backed off to 00:00 on the 1st of the next month. That encodes a
    GUESS about the vendor's billing cycle, and if the plan's quota resets on
    the SUBSCRIPTION ANNIVERSARY instead — normal for a paid monthly plan —
    the guess fails in the worst direction and keeps failing:

        Sep 1  we re-probe -> still 429 -> back off to Oct 1
        Sep 12 quota actually resets, but we are locked out
        Oct 1  we finally re-probe

    i.e. ~19 extra dark days, every month, forever, with the feed reporting
    headlines=0 the whole time. P293b's own stated fail direction was "retry
    sooner than necessary, never go dark for weeks by mistake"; hard-coding
    the calendar inverted it.

    So: do not encode a reset date at all. Re-probe DAILY. One request a day
    is ~96x cheaper than the per-tick hammering P293b actually fixed (that
    was the real defect: ~2,900 requests/month at a 15-minute retry), it
    cannot outlive a genuine month-long exhaustion by more than a day, and it
    is immune to whatever the vendor's cycle turns out to be. The month
    boundary is still used as a CEILING, because the quota cannot fail to
    have reset by then.
    """
    return min(now + timedelta(seconds=QUOTA_REPROBE_SEC),
               _next_month_start_utc(now))


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

    ⚠️ [P155, 2026-08-04] BILLING: the plan tier is baked into BASE_URL. This
    is the **Growth** (paid) endpoint, NOT the free tier that .env.example and
    the 2026-02 E2E training guide (now docs/archive/
    HMATS_E2E_TRAINING_GUIDE_2026-02.md) described for months. That mismatch is why
    a ~540 req/month workload produced a $400 bill and read as runaway traffic
    (see P154): the cost is the subscription, not the request count. Changing
    this string changes what you are billed — do not treat it as cosmetic.
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
        # [P319b] HTTP statuses already reported, so a rejected key does not
        # log ~18 identical lines a day. Declared here rather than created
        # lazily: an inferred `set[Never]` fails the type gate on every
        # .add(), which is the second CI round-trip this pattern has cost
        # (P317b was the first).
        self._http_warned: set = set()
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

        # [P154] Restore throttle + 429 backoff + news cache from disk BEFORE the
        # first tick. Without this every process start re-armed a fresh fetch and
        # forgot any active rate-limit backoff — see the class docstring on
        # _restore_state for why that is the actual quota driver.
        self._restore_state()

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
            # [P322c] Disabled means SILENT and EMPTY — checked before the
            # mock branch, or turning the feed off would turn the fabricator
            # on for any instance that was built without a key.
            if _cryptopanic_disabled:
                return None
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
    # [P154] PERSISTENCE — rate-limit state must survive restarts
    # =========================================================================
    # `_last_fetch_time` (300s throttle), `_backoff_until` (429 circuit breaker)
    # and `_last_data` (the 1h cache the LLM-sentiment agent gates its refresh on)
    # were in-memory only. run_live() runs a full tick immediately on entering its
    # loop (main.py, before any sleep), so EVERY process start burned a fresh
    # 3-request fetch AND forgot an active 429 backoff — i.e. a crash-restart loop
    # hammered an API that had just said "wait 15 minutes". Same in-memory-state-
    # is-not-a-control-across-restarts class as P150/P148/P140-B2.
    #
    # Bump _STATE_VERSION whenever the serialized SHAPE changes so stale files are
    # discarded on restore rather than half-parsed (P153 pattern).
    _STATE_VERSION = "cp_cache_v1"

    # Cache older than this is dropped on restore. Consumers already gate on
    # staleness (the agent refetches above 1h), so this is just a sanity bound.
    _MAX_RESTORED_CACHE_AGE_SEC = 86400.0

    def _data_dir(self) -> str:
        return os.environ.get("HMATS_DATA_DIR", "data")

    def _state_path(self) -> str:
        return os.path.join(self._data_dir(), "cryptopanic_state.json")

    @staticmethod
    def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
        """Parse an ISO timestamp, forcing tz-aware UTC (P40/P97 discipline —
        a naive datetime here would raise on every comparison in fetch())."""
        if not raw:
            return None
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _news_to_json(item: NewsItem) -> Dict[str, Any]:
        """Full round-trippable form. NewsItem.to_dict() is the lossy display
        shape (5 fields) and cannot rebuild the item — don't reuse it here."""
        return {
            "id": item.id,
            "title": item.title,
            "source": item.source,
            "url": item.url,
            "published_at": item.published_at.isoformat(),
            "currencies": list(item.currencies),
            "votes_positive": item.votes_positive,
            "votes_negative": item.votes_negative,
            "votes_important": item.votes_important,
            "votes_liked": item.votes_liked,
            "votes_disliked": item.votes_disliked,
            "votes_lol": item.votes_lol,
            "votes_toxic": item.votes_toxic,
            "votes_saved": item.votes_saved,
            "sentiment_score": item.sentiment_score,
        }

    @classmethod
    def _news_from_json(cls, raw: Dict[str, Any]) -> NewsItem:
        return NewsItem(
            id=str(raw.get("id", "")),
            title=str(raw.get("title", "")),
            source=str(raw.get("source", "unknown")),
            url=str(raw.get("url", "")),
            published_at=cls._parse_dt(raw.get("published_at")) or datetime.now(timezone.utc),
            currencies=list(raw.get("currencies", [])),
            votes_positive=int(raw.get("votes_positive", 0)),
            votes_negative=int(raw.get("votes_negative", 0)),
            votes_important=int(raw.get("votes_important", 0)),
            votes_liked=int(raw.get("votes_liked", 0)),
            votes_disliked=int(raw.get("votes_disliked", 0)),
            votes_lol=int(raw.get("votes_lol", 0)),
            votes_toxic=int(raw.get("votes_toxic", 0)),
            votes_saved=int(raw.get("votes_saved", 0)),
            sentiment_score=float(raw.get("sentiment_score", 0.0)),
        )

    def _restore_state(self) -> None:
        """[P154] Rehydrate throttle/backoff/cache from disk. Best-effort: any
        problem falls back to a cold start (current behaviour), never raises."""
        if self._mock_mode:
            return  # mock never touches the real cache file
        try:
            p = self._state_path()
            if not os.path.exists(p):
                return
            with open(p, "r", encoding="utf-8") as fh:
                st = json.load(fh)

            if st.get("state_version") != self._STATE_VERSION:
                logger.warning(
                    f"[CRYPTOPANIC] state_version={st.get('state_version')!r} != "
                    f"{self._STATE_VERSION!r} -> discarding stale cache file"
                )
                return

            # Backoff first — this is the field whose loss actually costs money.
            # An already-expired value is harmless (fetch() compares against now).
            # [P319] CAP A RESTORED BACKOFF AT THE RE-PROBE INTERVAL.
            # P299 replaced the month-long monthly-quota lockout with a daily
            # re-probe — but only for a backoff computed FROM NOW ON. The
            # value already on disk was `2026-09-01T00:00:00Z`, restored
            # verbatim on every boot, so the live feed stayed dark until
            # September exactly as before and the fix was INERT where it
            # mattered. That is P261b's lesson: a migration case is a
            # first-boot case with pre-existing state, and "the new code
            # computes it correctly" says nothing about the value already
            # persisted.
            #
            # Capping only ever SHORTENS a wait, never lengthens one, so a
            # legitimate short Retry-After backoff is untouched; the fail
            # direction stays "retry sooner than necessary", which is what
            # P293b said it wanted and P299 said again.
            self._combined_probe_done = bool(st.get("combined_probe_done", False))
            _restored_backoff = self._parse_dt(st.get("backoff_until"))
            if _restored_backoff is not None:
                _cap = datetime.now(timezone.utc) + timedelta(
                    seconds=QUOTA_REPROBE_SEC)
                if _restored_backoff > _cap:
                    logger.info(
                        "[CRYPTOPANIC] restored backoff %s is longer than the "
                        "%dh re-probe interval — capping to %s. A stored "
                        "month-long lockout predates P299 and would make its "
                        "daily re-probe unreachable (P319).",
                        _restored_backoff.isoformat(),
                        int(QUOTA_REPROBE_SEC // 3600), _cap.isoformat())
                    _restored_backoff = _cap
            self._backoff_until = _restored_backoff
            self._last_fetch_time = self._parse_dt(st.get("last_fetch_time"))

            payload = st.get("data")
            age_s = None
            if payload:
                ts = self._parse_dt(payload.get("timestamp"))
                if ts is not None:
                    age_s = (datetime.now(timezone.utc) - ts).total_seconds()
                    if age_s <= self._MAX_RESTORED_CACHE_AGE_SEC:
                        self._last_data = CryptoPanicData(
                            timestamp=ts,
                            staleness_sec=float(payload.get("staleness_sec", 0.0)),
                            recent_news=[
                                self._news_from_json(n)
                                for n in payload.get("recent_news", [])
                            ],
                            panic_score=dict(payload.get("panic_score", {})),
                            news_velocity=dict(payload.get("news_velocity", {})),
                            sentiment_consensus=dict(payload.get("sentiment_consensus", {})),
                            narrative_intensity=dict(payload.get("narrative_intensity", {})),
                            global_panic=float(payload.get("global_panic", 0.0)),
                            global_news_velocity=float(payload.get("global_news_velocity", 0.0)),
                        )
                    else:
                        logger.info(
                            f"[CRYPTOPANIC] cached news {age_s / 3600:.1f}h old "
                            f"(> {self._MAX_RESTORED_CACHE_AGE_SEC / 3600:.0f}h) -> dropped"
                        )

            logger.info(
                "[CRYPTOPANIC] restored state: news=%d age=%s backoff=%s",
                len(self._last_data.recent_news) if self._last_data else 0,
                f"{age_s:.0f}s" if age_s is not None else "n/a",
                self._backoff_until.isoformat() if self._backoff_until else "none",
            )
        except Exception as e:  # noqa: silent-swallow — bad/old state file; cold start + log
            logger.warning(f"[CRYPTOPANIC] state restore failed: {type(e).__name__}: {e}")

    def _persist_state(self) -> None:
        """[P154] Atomically write throttle/backoff/cache. Never raises."""
        if self._mock_mode:
            return
        try:
            p = self._state_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            payload = None
            if self._last_data is not None:
                payload = {
                    "timestamp": self._last_data.timestamp.isoformat(),
                    "staleness_sec": self._last_data.staleness_sec,
                    "recent_news": [
                        self._news_to_json(n) for n in self._last_data.recent_news
                    ],
                    "panic_score": self._last_data.panic_score,
                    "news_velocity": self._last_data.news_velocity,
                    "sentiment_consensus": self._last_data.sentiment_consensus,
                    "narrative_intensity": self._last_data.narrative_intensity,
                    "global_panic": self._last_data.global_panic,
                    "global_news_velocity": self._last_data.global_news_velocity,
                }
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "state_version": self._STATE_VERSION,
                    "last_fetch_time": (
                        self._last_fetch_time.isoformat() if self._last_fetch_time else None
                    ),
                    # [P319] one-shot marker for the combined-currency probe
                    "combined_probe_done": bool(
                        getattr(self, "_combined_probe_done", False)),
                    "backoff_until": (
                        self._backoff_until.isoformat() if self._backoff_until else None
                    ),
                    "data": payload,
                    "saved_ts": time.time(),
                }, fh, default=float)
            os.replace(tmp, p)
        except Exception as e:  # noqa: silent-swallow — persistence is best-effort, never break the tick
            logger.debug(f"[CRYPTOPANIC] state persist failed: {type(e).__name__}: {e}")

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
        
        _cp_rate_limited = False
        _cp_errors = 0
        # [P287] Track WHICH currencies failed (transport error or non-200),
        # so a partial outage can carry their cached items instead of
        # zeroing them. A 200 with zero posts is genuine "no news", NOT a
        # failure.
        _cp_failed: set = set()
        async with create_session() as session:
            # Fetch posts for each currency
            for _cp_idx, currency in enumerate(SUPPORTED_CURRENCIES):
                try:
                    news = await self._fetch_posts(session, currency)
                    data.recent_news.extend(news)
                    if self._last_status_code is not None and self._last_status_code != 200:
                        _cp_failed.add(currency)
                    if self._last_status_code == 429:
                        _cp_rate_limited = True
                        # Currencies not yet attempted are failed too.
                        _cp_failed.update(SUPPORTED_CURRENCIES[_cp_idx + 1:])
                        break
                except Exception as e:
                    _cp_errors += 1
                    _cp_failed.add(currency)
                    logger.warning(f"[CRYPTOPANIC] {currency}: {e}")

        # [P319] ONE-SHOT: does `currencies=BTC,ETH,SOL` mean OR or AND?
        #
        # We issue THREE requests per fetch, one per currency. If the filter is
        # an OR, a single combined request returns the same corpus for a THIRD
        # of the quota — and on a plan whose real limit we cannot read (the
        # vendor's plans page is JS-rendered and the API answers 429), a 3x
        # reduction may be the difference between fitting and not.
        #
        # But if the filter is an AND, combining would silently return only
        # posts tagged with all three — a coverage collapse that looks like a
        # quiet news day (the P2 shape). Neither the API (429) nor the docs
        # (JS-rendered) can settle it today, so the change is NOT made on a
        # guess. This measures it exactly once, when the quota is healthy,
        # LOGS the answer, and acts on nothing — converting "someone must
        # remember to test this" into a mechanism (P230/P280).
        if (not _cp_failed and not _cp_errors
                and not getattr(self, "_combined_probe_done", False)
                and data.recent_news):
            try:
                # [P420] `session` here is the `async with create_session()`
                # block ABOVE, already CLOSED by the time this runs — so the
                # probe raised RuntimeError on every healthy fetch, the
                # one-shot measurement could never complete, and its "will
                # retry on a later healthy fetch" warning fired forever. The
                # probe now opens its own session (the guard is unchanged).
                async with create_session() as _probe_session:
                    _combo = await self._fetch_posts(
                        _probe_session, ",".join(SUPPORTED_CURRENCIES))
                _seen = set()
                for _it in _combo or []:
                    for _c in (getattr(_it, "currencies", None) or []):
                        if _c in SUPPORTED_CURRENCIES:
                            _seen.add(_c)
                self._combined_probe_done = True
                logger.warning(
                    "[CRYPTOPANIC][P319] combined-currency probe: "
                    "`currencies=%s` returned %d post(s) covering %s. "
                    "If that spans ALL of %s the filter is an OR and this "
                    "feed can drop from 3 requests per fetch to 1 (a third of "
                    "the quota); if it is a strict subset the filter is an AND "
                    "and the per-currency loop must stay. Measured once ever; "
                    "nothing was changed on the strength of it.",
                    ",".join(SUPPORTED_CURRENCIES), len(_combo or []),
                    sorted(_seen) or "NOTHING",
                    SUPPORTED_CURRENCIES)
            except Exception as _pe:  # noqa: silent-swallow — logged below; a measurement must never break the feed it measures
                logger.warning(
                    "[CRYPTOPANIC][P319] combined-currency probe failed "
                    "(%s: %s) — will retry on a later healthy fetch",
                    type(_pe).__name__, str(_pe)[:120])

        # [P265] The tick that ARMS the backoff must not destroy the cache the
        # backoff exists to serve. The old code ran _compute_metrics on the
        # empty/partial corpus (panic 0.0, velocity 0.0, fresh timestamp),
        # assigned it over the restored good cache AND persisted it — so every
        # fetch() during the backoff served the empty dataset, a restart
        # restored it, llm_sentiment went dark until the next fully-successful
        # fetch, and a partial outage re-stamped the failed assets' panic as
        # 0.0/fresh. On a rate-limited or fruitless fetch, serve the existing
        # cache (P154's design, actually honored) and persist NOTHING.
        if (_cp_rate_limited or not data.recent_news) and self._last_data is not None:
            logger.warning(
                f"[CRYPTOPANIC] fetch yielded no news "
                f"(rate_limited={_cp_rate_limited}, errors={_cp_errors}) — "
                f"serving the cached corpus instead of overwriting it with "
                f"an empty dataset")
            # Backoff state (set inside _fetch_posts) still persists.
            self._persist_state()
            return self._last_data

        # [P287] Per-currency carry. The P265 guard above covers only the
        # full-outage / rate-limited case; a PARTIAL failure (BTC ok, ETH
        # timeout) still passed it and _compute_metrics then stamped the
        # failed currencies' panic/velocity as 0.0/fresh, cached and
        # PERSISTED it — the exact scenario the P265 comment names. Carry
        # the previous corpus's items for the failed currencies into this
        # fetch (dedup by id). The carried items keep their own
        # published_at, so the failed currency's metrics age honestly out
        # of the 4h windows instead of snapping to a fabricated zero.
        if _cp_failed and self._last_data is not None:
            _have_ids = {n.id for n in data.recent_news}
            _carried_n = 0
            for _old in self._last_data.recent_news:
                if _old.id in _have_ids:
                    continue
                if any(c in _cp_failed for c in _old.currencies):
                    data.recent_news.append(_old)
                    _have_ids.add(_old.id)
                    _carried_n += 1
            logger.warning(
                f"[CRYPTOPANIC] partial fetch — {sorted(_cp_failed)} failed; "
                f"carried {_carried_n} cached items for them (items keep "
                f"their own timestamps, metrics decay honestly — NOT "
                f"re-stamped as fresh zeros)")

        # Compute metrics
        self._compute_metrics(data)

        self._last_data = data
        self._last_fetch_time = now

        # [P154] Checkpoint the throttle + news cache so a restart reuses them
        # instead of re-spending 3 requests on the immediate startup tick.
        self._persist_state()

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

                    # [P293b] A MONTHLY QUOTA is not a rate limit, and
                    # treating it as one is why this looked like a recurring
                    # incident. CryptoPanic returns 429 with NO Retry-After
                    # and a body reading "API monthly quota exceeded", so the
                    # 900s default above had the engine re-asking every 15
                    # minutes for the REST OF THE MONTH — ~2,900 pointless
                    # requests, each logging a warning that reads like a
                    # transient fault (the P202 alert-fatigue shape).
                    #
                    # Quota exhaustion resolves on the calendar, not on a
                    # timer: back off to the 1st of next month UTC, and say
                    # once, plainly, that this is a BUDGET state requiring a
                    # plan decision — the same language the P220 CryptoCompare
                    # guard uses, for the same reason.
                    _quota_exhausted = False
                    try:
                        _body = (await resp.text())[:400]
                    except Exception:  # noqa: silent-swallow
                        _body = ""
                    if _body and _is_monthly_quota_error(_body):
                        _quota_exhausted = True
                        _now = datetime.now(timezone.utc)
                        self._backoff_until = quota_backoff_until(_now)
                        self._persist_state()
                        if not getattr(self, "_quota_warned_month", None) == (
                            _now.year, _now.month
                        ):
                            self._quota_warned_month = (_now.year, _now.month)
                            logger.error(
                                "[CRYPTOPANIC] MONTHLY QUOTA EXHAUSTED for "
                                "%04d-%02d — the API says: %s. Backing off "
                                "until %s — a DAILY re-probe, deliberately "
                                "not 'until the 1st': the vendor's reset date "
                                "is not something we know, and guessing it "
                                "wrong locks the feed out for weeks (P299). "
                                "This is a BUDGET/PLAN decision, "
                                "NOT a market condition and NOT a transient "
                                "fault: llm_sentiment will report "
                                "headlines=0 until the month rolls or the "
                                "plan is upgraded. Serving the cached corpus "
                                "meanwhile.",
                                _now.year, _now.month,
                                _body.strip()[:160],
                                self._backoff_until.isoformat(),
                            )
                        return []

                    self._backoff_until = datetime.now(timezone.utc) + timedelta(seconds=retry_after_sec)
                    # [P154] Persist the backoff IMMEDIATELY. _fetch_real's
                    # checkpoint is not enough: if the process dies between the
                    # 429 and the end of the fetch, the restart forgets it and
                    # hammers the API that just rate-limited us.
                    self._persist_state()
                # [P319b] A CREDENTIAL failure is permanent, not transient.
                # The old branch set NO backoff and logged on every fetch, so
                # an invalid key (401) meant ~18 identical warnings a day
                # forever with the request re-issued each time — the P202
                # shape, and precisely what CANCELLING OR CHANGING THE PLAN
                # would produce. A hardcoded credential that the API rejects
                # cannot start working before someone edits the environment,
                # so it gets the same daily re-probe as an exhausted quota.
                #
                # 401 ONLY. A 403 is ambiguous here — P293b measured
                # Cloudflare returning `403 error code: 1010` for an
                # unidentified client, which a User-Agent fixes and which must
                # keep retrying. Treating 403 as permanent would turn a
                # transient block into a self-inflicted outage.
                if resp.status == 401:
                    self._backoff_until = quota_backoff_until(
                        datetime.now(timezone.utc))
                    self._persist_state()
                if resp.status not in self._http_warned:
                    self._http_warned.add(resp.status)
                    logger.warning(
                        "[CRYPTOPANIC] HTTP %d from API — %s. Logged once per "
                        "status per process; the corpus falls back to the "
                        "cached items and to the free RSS blend (P314), which "
                        "currently carries llm_sentiment on its own.",
                        resp.status,
                        ("the key is REJECTED: check CRYPTOPANIC_API_KEY and "
                         "the plan is still active. Re-probing daily, because "
                         "a bad credential cannot fix itself (P319b)"
                         if resp.status == 401 else
                         "retrying on the normal cadence"))
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
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError, AttributeError):
                    # [P287] An unparseable/absent timestamp must not read
                    # as "just published" — stamping NOW put undated items
                    # inside every recency window (panic score, velocity),
                    # i.e. absence read as maximal freshness (the P100
                    # timestamp-lies shape). Stamp epoch instead: the item
                    # stays in the corpus but never enters a recency window.
                    logger.debug(
                        f"[CRYPTOPANIC] unparseable published_at="
                        f"{pub_str!r} — stamped epoch (excluded from "
                        f"recency windows)")
                    pub_date = datetime(1970, 1, 1, tzinfo=timezone.utc)
                
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

# [P322c] MODULE-LEVEL, because the singleton has callers main.py does not
# own. P322 gated main.py's construction site and the feed kept initialising
# and would have kept fetching: `agents/sentiment_llm_agent.py` calls
# `get_cryptopanic_feed()` DIRECTLY at two sites, so gating one handle left
# the real consumers untouched — the P152/P2 shape ("a guard defined and never
# reached"), in the fix meant to close it. Verified by reading the live log
# after the deploy rather than trusting the config (P295b).
#
# The switch lives beside the singleton so that EVERY route through it obeys
# the same decision, whoever imported it.
_cryptopanic_disabled: bool = False


def set_feed_enabled(enabled: bool) -> None:
    """Turn the feed on/off process-wide. Called by main.py from config."""
    global _cryptopanic_disabled
    _cryptopanic_disabled = not bool(enabled)


def is_feed_enabled() -> bool:
    return not _cryptopanic_disabled


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

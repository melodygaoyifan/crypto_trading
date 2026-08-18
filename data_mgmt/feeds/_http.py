"""
Shared aiohttp session factory with ThreadedResolver and retry helper.

On Windows, aiodns (c-ares) DNS resolution fails with
"Could not contact DNS servers". ThreadedResolver uses
socket.getaddrinfo() in a thread pool, which works reliably.
"""

import asyncio
import logging
import aiohttp
from aiohttp.resolver import ThreadedResolver

logger = logging.getLogger(__name__)


# [P293b] Cloudflare-fronted vendors (CryptoPanic among them) reject clients
# by signature. Probed live from the server: a request with the default
# stdlib/urllib User-Agent returns HTTP 403 "error code: 1010" — Cloudflare's
# banned-client-signature code — while the SAME url with any ordinary UA
# reaches the API. That 403 is a different failure from a rate limit and
# would keep a feed dark even with quota available, so every session this
# factory builds now identifies itself.
DEFAULT_USER_AGENT = "hmats/6.8 (automated trading system; contact: operator)"


def create_session(**kwargs) -> aiohttp.ClientSession:
    """Create aiohttp.ClientSession with ThreadedResolver for Windows DNS fix."""
    connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
    kwargs.setdefault("timeout", aiohttp.ClientTimeout(total=10))
    # [P293b] Caller-supplied headers win; only fill the UA if absent.
    _headers = dict(kwargs.pop("headers", None) or {})
    if not any(k.lower() == "user-agent" for k in _headers):
        _headers["User-Agent"] = DEFAULT_USER_AGENT
    return aiohttp.ClientSession(connector=connector, headers=_headers, **kwargs)


def cache_age_seconds(last_fetch, now=None):
    """[P293f] Age of a cache stamp in seconds, or None if never fetched.

    Shared so the fetch-throttle pattern has ONE definition instead of the
    sixth hand-rolled copy (P172). Accepts a datetime (aware or naive) or a
    float epoch, because the feeds in this package use both.

    None means "never fetched" and is deliberately distinct from a large
    number: those have different causes and different fixes, and collapsing
    them is how several feeds' permanently-empty caches stayed invisible.
    """
    import datetime as _dt
    if last_fetch is None:
        return None
    try:
        if isinstance(last_fetch, (int, float)):
            if last_fetch <= 0:
                return None
            base = _dt.datetime.fromtimestamp(last_fetch, tz=_dt.timezone.utc)
        else:
            base = last_fetch
            if base.tzinfo is None:   # [P40/P97] naive/aware defence
                base = base.replace(tzinfo=_dt.timezone.utc)
        ref = now or _dt.datetime.now(_dt.timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=_dt.timezone.utc)
        return (ref - base).total_seconds()
    except (TypeError, ValueError, OverflowError, OSError):  # noqa: silent-swallow
        # [P293f] An unreadable stamp is "age unknown" -> None, which every
        # caller treats as "never fetched" and therefore FETCHES. The fail
        # direction is deliberately toward doing the work, never toward
        # silently serving a stale cache forever.
        return None


def strip_tz(dt):
    """Strip tzinfo from a datetime — used to keep timestamps comparable
    with naive `datetime.now()`.

    [P40 2026-04-24] Added so feeds can defensively normalize timestamps
    that `datetime.fromisoformat()` may have parsed as either aware or
    naive depending on the persisted ISO string's tz marker. Mirrors the
    pattern in `defense/strategy_existence_fuse.py` and `drl/promotion_gate.py`.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def parse_retry_after(raw: str | None) -> float | None:
    """Parse a Retry-After header value (seconds OR HTTP-date) → seconds.

    Returns None if `raw` is empty or unparseable. Negative wait times are
    clamped to 0. Caller should additionally cap at a sensible ceiling
    (e.g. 30s) so a malicious server can't request "wait 99999s".

    Extracted from `fetch_with_retry`'s 429 path so direct-call clients
    (Coinglass, CryptoCompare, etc.) can honor Retry-After without
    routing through the generic retry helper.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        _dt = parsedate_to_datetime(raw)
        if _dt is not None:
            if _dt.tzinfo is None:
                _dt = _dt.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (_dt - datetime.now(timezone.utc)).total_seconds(),
            )
    except Exception:
        return None
    return None


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    *,
    max_retries: int = 3,
    backoff_base: float = 0.3,
    operation: str = "fetch",
) -> dict | None:
    """
    GET JSON from url with retry + exponential backoff.

    Returns parsed JSON dict on success, None on exhausted retries.
    Only retries on network/timeout errors, not 4xx.
    """
    last_error = None
    for attempt in range(max_retries):
        retry_after: float | None = None
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                # [P22 2026-04-24] 429 was previously bucketed with all 4xx and dropped
                # without retry. CryptoPanic / Coinglass / Anthropic burn quota silently
                # under load. Now: parse Retry-After (seconds or HTTP-date), fall back to
                # exponential backoff, and continue the retry loop.
                if resp.status == 429:
                    raw = resp.headers.get("Retry-After", "").strip()
                    if raw:
                        try:
                            retry_after = float(raw)
                        except ValueError:
                            try:
                                from email.utils import parsedate_to_datetime
                                from datetime import datetime, timezone
                                _dt = parsedate_to_datetime(raw)
                                if _dt is not None:
                                    if _dt.tzinfo is None:
                                        _dt = _dt.replace(tzinfo=timezone.utc)
                                    retry_after = max(
                                        0.0,
                                        (_dt - datetime.now(timezone.utc)).total_seconds(),
                                    )
                            except Exception:
                                retry_after = None
                    last_error = f"HTTP 429 (retry_after={retry_after})"
                elif 400 <= resp.status < 500:
                    logger.warning(f"[HTTP] {operation} got {resp.status}, not retrying")
                    return None
                else:
                    # 5xx - retry
                    last_error = f"HTTP {resp.status}"

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            if retry_after is not None:
                # Server told us how long to wait; cap at 30s to avoid death-stalls.
                sleep = min(float(retry_after), 30.0)
            else:
                sleep = backoff_base * (2 ** attempt)
            logger.debug(f"[HTTP] {operation} attempt {attempt+1} failed, retry in {sleep:.1f}s")
            await asyncio.sleep(sleep)

    logger.warning(f"[HTTP] {operation} failed after {max_retries} retries: {last_error}")
    return None

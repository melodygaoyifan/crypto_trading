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


def create_session(**kwargs) -> aiohttp.ClientSession:
    """Create aiohttp.ClientSession with ThreadedResolver for Windows DNS fix."""
    connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
    kwargs.setdefault("timeout", aiohttp.ClientTimeout(total=10))
    return aiohttp.ClientSession(connector=connector, **kwargs)


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
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                if 400 <= resp.status < 500:
                    logger.warning(f"[HTTP] {operation} got {resp.status}, not retrying")
                    return None
                # 5xx - retry
                last_error = f"HTTP {resp.status}"

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            sleep = backoff_base * (2 ** attempt)
            logger.debug(f"[HTTP] {operation} attempt {attempt+1} failed, retry in {sleep:.1f}s")
            await asyncio.sleep(sleep)

    logger.warning(f"[HTTP] {operation} failed after {max_retries} retries: {last_error}")
    return None

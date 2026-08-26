"""
================================================================================
HMATS [P293c] - RSS news feed (free, keyless, unquota'd)
================================================================================

The cheaper CryptoPanic option: there isn't a cheaper *vendor* worth buying —
there is a free one already publishing the only thing this system actually
consumes.

WHAT THE LLM LAYER ACTUALLY NEEDS. `SentimentLLMAgent` sends **headline text**
to Haiku and derives sentiment itself. It does not consume CryptoPanic's vote
tallies for direction. So the expensive part of CryptoPanic (its curated
sentiment metadata) is not what the tradeable signal rests on — the headlines
are, and headlines are free.

MEASURED (live probe 2026-08-17, no key, no account):
    cointelegraph.com/rss      30 items
    decrypt.co/feed            34 items
    bitcoinmagazine.com/feed   10 items
    theblock.co/rss.xml        20 items
                              ~94 headlines per poll, unlimited, $0

(coindesk returns 308 to its feed URL and is deliberately omitted rather than
followed blindly — an unverified redirect target is not a source.)

WHAT THIS DOES NOT REPLACE. CryptoPanic also supplies `panic_score`,
`news_velocity` and `narrative_intensity` from its vote data. RSS has no
equivalent, and this feed does NOT fabricate one — those metrics stay at
their current values (all 0.0 today) rather than being invented from a
headline count. Absence stays absence (P2).

RELEVANCE FILTERING IS WORD-BOUNDED ON PURPOSE. A naive `"sol" in title`
matches "solution", "sold", "console", "solar" — which would flood SOL with
unrelated news and quietly corrupt the one input this exists to supply.
================================================================================
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

logger = logging.getLogger("HMATS.RSSNews")


# Verified live. Adding a source means PROBING it first — a feed that 404s or
# redirects returns zero items and is indistinguishable from a quiet news day.
RSS_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    # [P314] Added because the original four are a CORPUS, not a firehose.
    # Measured 2026-08-19: 95 items across them, median age 15-29h, and only
    # TWO inside the agent's 4h freshness window — per asset BTC 0, ETH 1,
    # SOL 0. That is why `[LLM_SENTIMENT]` read `headlines=0 tradeable=False`
    # on BTC and SOL while `[RSS] 94 headline(s) relevant {BTC: 23}` sat four
    # lines above it in the same log. I had recorded that as "a data fact"
    # and it was not: the window was fine, the ARRIVAL RATE was too low.
    #
    # Each entry below was probed for http status, item count, dated-item
    # count and in-window yield BEFORE being added (the P218 rule), and the
    # numbers are the reason it is here:
    #     gnews_*      100-102 items, 4-6 inside 4h  <- the decisive ones
    #     bitcoincom   10 items,  4 inside 4h, median  5.1h
    #     cryptoslate  10 items,  3 inside 4h, median  7.1h
    #     ambcrypto    16 items,  2 inside 4h, median  7.2h
    #     coindesk     25 items,  2 inside 4h, median 12.7h
    # coindesk is back: P293c omitted it because the bare domain 308-redirects
    # and "an unverified redirect target is not a source"; the arc
    # outboundfeeds URL answers 200 directly and is now verified.
    ("gnews_btc",
     "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en"),
    ("gnews_eth",
     "https://news.google.com/rss/search?q=ethereum&hl=en-US&gl=US&ceid=US:en"),
    ("gnews_sol",
     "https://news.google.com/rss/search?q=solana&hl=en-US&gl=US&ceid=US:en"),
    # [P413] Breadth-asset scoped queries, probed 2026-08-26 BEFORE adding
    # (the P218/P314 rule). Plain `q=XRP` read 13 inside 4h; plain `q=BNB`
    # was REJECTABLE (100 items, 1 inside 4h, median ~105 DAYS -- a bare
    # ticker returns an undated corpus). The `when:1d` operator restricts
    # results to the last day and flips both above the admission bar:
    #     gnews_xrp  "XRP OR ripple when:1d"  100 items, 32 inside 4h, median 6.9h
    #     gnews_bnb  "BNB when:1d"             50 items,  5 inside 4h, median 9.4h
    # The home-trio queries are deliberately NOT retrofitted with when:1d --
    # they are working evidence streams (P299: no discontinuity for tidiness).
    ("gnews_xrp",
     "https://news.google.com/rss/search?q=XRP%20OR%20ripple%20when%3A1d&hl=en-US&gl=US&ceid=US:en"),
    ("gnews_bnb",
     "https://news.google.com/rss/search?q=BNB%20when%3A1d&hl=en-US&gl=US&ceid=US:en"),
    ("bitcoincom", "https://news.bitcoin.com/feed/"),
    ("cryptoslate", "https://cryptoslate.com/feed/"),
    ("ambcrypto", "https://ambcrypto.com/feed/"),
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
)

# [P314] PROBED AND REJECTED, recorded so they are not re-added on a hunch:
#     newsbtc   https://www.newsbtc.com/feed/   median age 174.9h, 0 inside 4h
#     utoday    https://u.today/rss             97 items but median 64.4h,
#                                               only 1 inside 4h
# Both answer HTTP 200 with well-formed XML, so they LOOK healthy — a source
# whose items are days old adds corpus and no freshness, which is the exact
# failure this change exists to fix.
REJECTED_SOURCES: Dict[str, str] = {
    "newsbtc": "median item age 174.9h, 0 inside the 4h window (P314)",
    "utoday": "97 items but median age 64.4h, 1 inside the 4h window (P314)",
}

# [P314] Sources whose URL is an ASSET QUERY. Their relevance is established
# by the request, not by the words in the title: a Google-News "bitcoin"
# result headlined "Crypto market rallies as ETF inflows resume" is about BTC
# even though the word-bounded matcher cannot see it. Attributing those by
# title alone would silently discard most of what makes these feeds worth
# adding. Title matching still runs on top, so one item can serve several
# assets.
SOURCE_ASSET_SCOPE: Dict[str, str] = {
    "gnews_btc": "BTC",
    "gnews_eth": "ETH",
    "gnews_sol": "SOL",
    "gnews_xrp": "XRP",   # [P413]
    "gnews_bnb": "BNB",   # [P413]
}

# Word-bounded so "solution"/"sold"/"console" cannot match SOL.
_ASSET_PATTERNS: Dict[str, re.Pattern] = {
    "BTC": re.compile(r"\b(bitcoin|btc|xbt)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|eth|ether)\b", re.I),
    "SOL": re.compile(r"\b(solana|sol)\b", re.I),
    # [P413] word-bounded: "Airbnb" cannot match BNB (no boundary before
    # the substring) and "ripples"/"rippled" cannot match ripple. Deliberately
    # NO bare "binance" term for BNB -- exchange news is not coin news.
    "XRP": re.compile(r"\b(xrp|ripple)\b", re.I),
    "BNB": re.compile(r"\b(bnb)\b", re.I),
}

DEFAULT_TTL_SEC = 1800.0   # 30 min; RSS is free, but hammering is still rude


@dataclass
class RSSNewsItem:
    """Minimal shape matching what the headline blender consumes."""
    title: str
    published_at: Optional[datetime]
    source: str
    link: str = ""

    def matches(self, asset: str) -> bool:
        a = asset.upper().replace("/USD", "")
        # [P314] An asset-scoped source answers for its own asset: the QUERY
        # is the relevance claim. Word matching still applies so a scoped
        # item mentioning another asset is attributed to that one too.
        if SOURCE_ASSET_SCOPE.get(self.source) == a:
            return True
        pat = _ASSET_PATTERNS.get(a)
        return bool(pat and pat.search(self.title or ""))


def _parse_date(raw: Optional[str]) -> Optional[datetime]:
    """RFC-822 (RSS) or ISO-8601 (Atom) -> aware UTC, else None.

    None is deliberate: an unparseable date must NOT become `now`, which
    would let an old article pass a freshness window (the P287 defect).
    """
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):  # noqa: silent-swallow
        pass  # [P293c] not RFC-822; fall through to the ISO attempt below
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):  # noqa: silent-swallow
        # [P293c] Unparseable in BOTH formats -> None, deliberately. The
        # caller treats None as "age unknown" and counts it separately;
        # stamping NOW here would let an old article pass a freshness
        # window (the P287 defect).
        return None


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_feed(xml_text: str, source: str) -> List[RSSNewsItem]:
    """Parse RSS 2.0 <item> or Atom <entry>. Never raises."""
    out: List[RSSNewsItem] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        logger.warning("[RSS] %s: unparseable XML (%s)", source, e)
        return out

    for node in root.iter():
        if _strip_ns(node.tag) not in ("item", "entry"):
            continue
        title = ""
        link = ""
        date_raw = None
        for child in node:
            t = _strip_ns(child.tag)
            if t == "title" and child.text:
                title = child.text.strip()
            elif t == "link":
                link = (child.text or child.attrib.get("href") or "").strip()
            elif t in ("pubDate", "published", "updated", "date"):
                if date_raw is None and child.text:
                    date_raw = child.text
        if title:
            out.append(RSSNewsItem(
                title=title,
                published_at=_parse_date(date_raw),
                source=source,
                link=link,
            ))
    return out


class RSSNewsFeed:
    """Multi-source RSS reader. Read-only, keyless, fail-soft."""

    def __init__(self, ttl_sec: float = DEFAULT_TTL_SEC,
                 sources: Tuple[Tuple[str, str], ...] = RSS_SOURCES):
        self._ttl = float(ttl_sec)
        self._sources = tuple(sources)
        self._items: List[RSSNewsItem] = []
        self._last_fetch: float = 0.0
        self._fetch_errors = 0
        # [P293f] url -> {etag, last_modified} for conditional GETs
        self._validators: Dict[str, Dict[str, Optional[str]]] = {}
        self._not_modified = 0
        logger.info(
            "[RSS] Initialized: %d source(s), ttl=%.0fs (no key, no quota)",
            len(self._sources), self._ttl,
        )

    def cache_age_sec(self) -> Optional[float]:
        return None if self._last_fetch <= 0 else (time.time() - self._last_fetch)

    def get_items(self, asset: Optional[str] = None) -> List[RSSNewsItem]:
        if asset is None:
            return list(self._items)
        return [i for i in self._items if i.matches(asset)]

    async def fetch_if_stale(self) -> List[RSSNewsItem]:
        age = self.cache_age_sec()
        if age is not None and age < self._ttl:
            return self._items
        return await self.fetch()

    async def fetch(self) -> List[RSSNewsItem]:
        """Fetch every source concurrently. Never raises."""
        from data_mgmt.feeds._http import create_session
        import aiohttp

        collected: List[RSSNewsItem] = []
        try:
            async with create_session() as session:
                async def _one(name: str, url: str) -> List[RSSNewsItem]:
                    try:
                        # [P293f] CONDITIONAL GET. Probed 2026-08-17: the RSS
                        # sources are the ONLY dependency here that exposes
                        # validators (cointelegraph returns `last-modified`
                        # and `s-maxage=300`); the JSON APIs return no ETag
                        # and no Last-Modified, so for those a client-side
                        # TTL is the only lever. Where a validator DOES
                        # exist, sending it back is the textbook-correct
                        # move: an unchanged feed answers 304 with no body.
                        _hdrs: Dict[str, str] = {}
                        _val = self._validators.get(url) or {}
                        _etag = _val.get("etag")
                        if _etag:
                            _hdrs["If-None-Match"] = str(_etag)
                        _lastmod = _val.get("last_modified")
                        if _lastmod:
                            _hdrs["If-Modified-Since"] = str(_lastmod)
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=15),
                            allow_redirects=True, headers=_hdrs or None,
                        ) as resp:
                            if resp.status == 304:
                                # Unchanged — reuse what we already parsed for
                                # this source rather than dropping it, or a
                                # 304 would read as "this source went silent".
                                self._not_modified += 1
                                return [i for i in self._items
                                        if i.source == name]
                            if resp.status != 200:
                                logger.warning("[RSS] %s -> HTTP %s", name, resp.status)
                                return []
                            _et = resp.headers.get("ETag")
                            _lm = resp.headers.get("Last-Modified")
                            if _et or _lm:
                                self._validators[url] = {
                                    "etag": _et, "last_modified": _lm}
                            text = await resp.text()
                        return parse_feed(text, name)
                    except Exception as e:
                        logger.warning("[RSS] %s failed: %s: %s",
                                       name, type(e).__name__, e)
                        return []

                results = await asyncio.gather(
                    *[_one(n, u) for n, u in self._sources],
                    return_exceptions=True,
                )
            for r in results:
                if isinstance(r, list):
                    collected.extend(r)
        except Exception as e:
            self._fetch_errors += 1
            logger.warning("[RSS] fetch failed: %s: %s", type(e).__name__, e)
            return self._items

        if not collected:
            # Do NOT stamp a failed sweep as fresh — that is the P265f
            # "fresh-stamped zeros" defect, and here it would read as
            # "no news happened".
            self._fetch_errors += 1
            logger.warning("[RSS] all sources returned nothing — keeping cache "
                           "(%d item(s))", len(self._items))
            return self._items

        # Dedup by normalised title; keep the earliest-seen (they are
        # equivalent for the LLM, and stable ordering keeps logs readable).
        seen = set()
        deduped: List[RSSNewsItem] = []
        for it in collected:
            key = (it.title or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        self._items = deduped
        self._last_fetch = time.time()
        _per_asset = {a: len(self.get_items(a)) for a in ("BTC", "ETH", "SOL")}
        logger.info(
            "[RSS] %d headline(s) from %d source(s); relevant: %s",
            len(deduped), len(self._sources), _per_asset,
        )
        return self._items

    def get_health(self) -> Dict[str, Any]:
        return {
            "source": "rss_public",
            "sources": [n for n, _ in self._sources],
            "items": len(self._items),
            "cache_age_sec": self.cache_age_sec(),
            "fetch_errors": self._fetch_errors,
            "not_modified_304": self._not_modified,
            "validators_held": len(self._validators),
        }


_rss_feed_instance: Optional[RSSNewsFeed] = None


def get_rss_news_feed(**kwargs) -> RSSNewsFeed:
    global _rss_feed_instance
    if _rss_feed_instance is None:
        _rss_feed_instance = RSSNewsFeed(**kwargs)
    return _rss_feed_instance


def reset_rss_news_feed():
    global _rss_feed_instance
    _rss_feed_instance = None


if __name__ == "__main__":
    async def _main():
        logging.basicConfig(level=logging.INFO)
        f = RSSNewsFeed()
        await f.fetch()
        for a in ("BTC", "ETH", "SOL"):
            rows = f.get_items(a)
            print(f"\n{a}: {len(rows)} relevant")
            for r in rows[:3]:
                age = ("?" if r.published_at is None else
                       f"{(datetime.now(timezone.utc) - r.published_at).total_seconds() / 3600:.1f}h")
                print(f"  [{r.source} {age}] {r.title[:80]}")
        print("\nhealth:", f.get_health())

    asyncio.run(_main())

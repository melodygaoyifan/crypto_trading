"""
[FRINGE-PANIC] Fringe Panic Detector - Multi-Layer Social Panic Signal
=======================================================================
Authority Level: ADVISE (no veto, no trigger)

Captures "fringe community panic surge" as a 12-48H leading indicator
for crypto crashes by monitoring:

  Layer 1: X API (tweet counts per keyword group, z-score spikes)
  Layer 2: Google Trends (search volume spikes for panic keywords)
  Layer 3: Reddit (post volume in panic subreddits)
  Layer 4: CryptoPanic (already exists, passed in as input)

Output:
  fringe_panic_score: 0.0-1.0 composite score
  panic_level: CALM / ELEVATED / HIGH / EXTREME
  short_signal: NEUTRAL / FAVORABLE / SQUEEZE_RISK
  leverage_adjustment: -0.3 to +0.3

Data Sources: X API v2, Google Trends (pytrends), Reddit JSON API
Frequency: Every 4H tick
Fail-safe: All API errors -> neutral output (score=0, level=CALM)
"""

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Safe import of shared HTTP helper
try:
    from data_mgmt.feeds._http import create_session
except ImportError:
    import aiohttp

    def create_session(**kwargs):
        return aiohttp.ClientSession(**kwargs)

# Optional: pytrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

# Import keyword config
try:
    from signals.fringe_panic_keywords import (
        PANIC_KEYWORD_GROUPS,
        CHINA_KEYWORDS_ZH,
        TRENDS_KEYWORDS_BATCH1,
        TRENDS_KEYWORDS_BATCH2,
        PANIC_SUBREDDITS,
    )
except ImportError:
    PANIC_KEYWORD_GROUPS = {}
    CHINA_KEYWORDS_ZH = {}
    TRENDS_KEYWORDS_BATCH1 = []
    TRENDS_KEYWORDS_BATCH2 = []
    PANIC_SUBREDDITS = []


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class GroupTweetStats:
    """Stats for a single keyword group."""
    group_name: str
    tweet_count_4h: int = 0
    rolling_mean: float = 0.0
    rolling_std: float = 1.0
    zscore: float = 0.0
    spike_detected: bool = False
    last_update: float = 0.0


# ─────────────────────────────────────────────────────────────
# LAYER 1: X API Monitor
# ─────────────────────────────────────────────────────────────

class XFringePanicMonitor:
    """
    [FRINGE-PANIC] X API tweet count monitor.

    Uses Recent Search API (Basic tier compatible).
    Falls back to defaults when no bearer token is set.
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        keyword_groups: Optional[Dict] = None,
        china_keywords: Optional[Dict] = None,
    ):
        self._bearer_token = bearer_token or os.getenv("X_API_BEARER_TOKEN", "")
        self._groups = keyword_groups or PANIC_KEYWORD_GROUPS
        self._china_kw = china_keywords or CHINA_KEYWORDS_ZH
        self._mock_mode = not bool(self._bearer_token)

        # History per group (42 bars = 7 days at 4H)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=42))
        self._stats: Dict[str, GroupTweetStats] = {}
        self._errors: int = 0

        if self._mock_mode:
            logger.info("[FRINGE-PANIC] X Monitor: no bearer token -> mock mode")
        else:
            logger.info("[FRINGE-PANIC] X Monitor: initialized")

    async def poll_search(self) -> Dict[str, GroupTweetStats]:
        """Poll X API Recent Search for each keyword group."""
        if self._mock_mode:
            return self._get_default_stats()

        try:
            return await self._do_poll()
        except Exception as e:
            self._errors += 1
            logger.debug(f"[FRINGE-PANIC] X poll error: {e}")
            return self._stats if self._stats else self._get_default_stats()

    async def _do_poll(self) -> Dict[str, GroupTweetStats]:
        """Execute search queries against X API."""
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "Content-Type": "application/json",
        }

        from datetime import datetime, timezone, timedelta
        start_time = (datetime.now(timezone.utc) - timedelta(hours=4)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        stats = {}
        async with create_session() as session:
            for group_name, group_config in self._groups.items():
                x_query = group_config.get("x_query", "")
                if not x_query:
                    continue
                try:
                    count = await self._search_count(
                        session, headers, x_query, start_time
                    )
                    stats[group_name] = self._update_group_stats(
                        group_name, count, group_config
                    )
                except Exception as e:
                    logger.debug(f"[FRINGE-PANIC] X search {group_name}: {e}")
                    stats[group_name] = GroupTweetStats(group_name=group_name)

                await asyncio.sleep(1.0)  # Rate limit

            # Chinese keywords -> merge into china_crisis
            if self._china_kw and "china_crisis" in stats:
                zh_query = self._china_kw.get("x_query", "")
                if zh_query:
                    try:
                        zh_count = await self._search_count(
                            session, headers, zh_query, start_time
                        )
                        stats["china_crisis"].tweet_count_4h += zh_count
                        self._recalc_zscore(stats["china_crisis"], "china_crisis")
                    except Exception as e:
                        logger.debug(f"[FRINGE-PANIC] X Chinese search: {e}")

        self._stats = stats
        return stats

    async def _search_count(
        self, session, headers: Dict, query: str, start_time: str,
    ) -> int:
        """Get tweet count via counts/recent or fallback to search/recent."""
        import aiohttp

        # Try counts endpoint first (more efficient)
        counts_url = "https://api.x.com/2/tweets/counts/recent"
        params = {"query": query, "start_time": start_time, "granularity": "hour"}

        try:
            async with session.get(
                counts_url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("meta", {}).get("total_tweet_count", 0)
        except Exception:
            pass

        # Fallback: search/recent
        search_url = "https://api.x.com/2/tweets/search/recent"
        params = {
            "query": query,
            "start_time": start_time,
            "max_results": 100,
        }
        try:
            async with session.get(
                search_url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("meta", {}).get("result_count", 0)
                return 0
        except Exception:
            return 0

    def _update_group_stats(
        self, group_name: str, count_4h: int, group_config: Dict,
    ) -> GroupTweetStats:
        """Update group stats and compute z-score."""
        stats = GroupTweetStats(group_name=group_name)
        stats.tweet_count_4h = count_4h
        stats.last_update = time.time()

        self._history[group_name].append((time.time(), count_4h))

        counts = [c for _, c in self._history[group_name]]
        if len(counts) >= 6:
            stats.rolling_mean = float(np.mean(counts))
            stats.rolling_std = float(max(np.std(counts), 1.0))
            stats.zscore = (count_4h - stats.rolling_mean) / stats.rolling_std
        else:
            baseline_4h = group_config.get("baseline_daily_tweets", 500) / 6
            stats.rolling_mean = baseline_4h
            stats.rolling_std = max(baseline_4h * 0.3, 1.0)
            stats.zscore = (count_4h - baseline_4h) / stats.rolling_std

        stats.spike_detected = stats.zscore > 2.0

        if stats.spike_detected:
            logger.warning(
                f"[FRINGE-PANIC] X SPIKE: {group_name} "
                f"count_4h={count_4h} zscore={stats.zscore:.1f}"
            )
        return stats

    def _recalc_zscore(self, stats: GroupTweetStats, group_name: str):
        """Recalculate z-score after merging Chinese counts."""
        counts = [c for _, c in self._history[group_name]]
        if counts:
            mean = float(np.mean(counts))
            std = float(max(np.std(counts), 1.0))
            stats.zscore = (stats.tweet_count_4h - mean) / std
            stats.spike_detected = stats.zscore > 2.0

    def _get_default_stats(self) -> Dict[str, GroupTweetStats]:
        """Neutral defaults when API unavailable."""
        return {
            name: GroupTweetStats(group_name=name) for name in self._groups
        }

    def get_composite_score(self) -> float:
        """0.0-1.0 weighted composite from X data."""
        if not self._stats:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for group_name, stats in self._stats.items():
            w = self._groups.get(group_name, {}).get("weight", 0.1)
            score = 1.0 / (1.0 + np.exp(-stats.zscore))  # sigmoid
            weighted_sum += score * w
            weight_total += w

        if weight_total > 0:
            return float(np.clip(weighted_sum / weight_total, 0.0, 1.0))
        return 0.0

    def get_active_spikes(self) -> List[str]:
        """Groups currently in spike."""
        return [n for n, s in self._stats.items() if s.spike_detected]


# ─────────────────────────────────────────────────────────────
# LAYER 2: Google Trends
# ─────────────────────────────────────────────────────────────

class GoogleTrendsPanicMonitor:
    """
    [FRINGE-PANIC] Google Trends search volume spikes.

    Uses pytrends (optional dependency). Returns 0.0 if not installed.
    """

    def __init__(self):
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=42))
        self._scores: Dict[str, float] = {}
        self._errors: int = 0

        if not PYTRENDS_AVAILABLE:
            logger.info("[FRINGE-PANIC] Google Trends: pytrends not installed -> disabled")
        else:
            logger.info("[FRINGE-PANIC] Google Trends: initialized")

    async def poll(self) -> Dict[str, float]:
        """Query Google Trends. Runs synchronous pytrends in executor."""
        if not PYTRENDS_AVAILABLE:
            return {}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._fetch_trends)
            self._scores = result
            return result
        except Exception as e:
            self._errors += 1
            logger.debug(f"[FRINGE-PANIC] Google Trends error: {e}")
            return self._scores

    def _fetch_trends(self) -> Dict[str, float]:
        """Synchronous Google Trends fetch."""
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        scores = {}

        for batch_name, keywords in [
            ("en1", TRENDS_KEYWORDS_BATCH1),
            ("en2", TRENDS_KEYWORDS_BATCH2),
        ]:
            if not keywords:
                continue
            try:
                pytrends.build_payload(keywords, timeframe='now 7-d', geo='')
                interest = pytrends.interest_over_time()

                if interest.empty:
                    continue

                for kw in keywords:
                    if kw in interest.columns:
                        values = interest[kw].values
                        if len(values) == 0:
                            continue
                        current = float(values[-1])
                        mean = float(np.mean(values))
                        std = float(max(np.std(values), 1.0))
                        zscore = (current - mean) / std
                        scores[kw] = zscore

                        if zscore > 2.0:
                            logger.warning(
                                f"[FRINGE-PANIC] Trends SPIKE: '{kw}' "
                                f"zscore={zscore:.1f} current={current:.0f}"
                            )

                time.sleep(2)  # Rate limit friendly
            except Exception as e:
                logger.debug(f"[FRINGE-PANIC] Trends batch '{batch_name}': {e}")

        return scores

    def get_composite_score(self) -> float:
        """0.0-1.0 from top-3 z-scores."""
        if not self._scores:
            return 0.0
        zscores = list(self._scores.values())
        top3 = sorted(zscores, reverse=True)[:3]
        avg_z = float(np.mean(top3))
        return float(np.clip(1.0 / (1.0 + np.exp(-avg_z)), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# LAYER 3: Reddit Panic Subreddits
# ─────────────────────────────────────────────────────────────

class RedditPanicMonitor:
    """
    [FRINGE-PANIC] Reddit panic subreddit post volume.

    Uses Reddit JSON API (free, no key needed, just User-Agent).
    """

    def __init__(self):
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=42))
        self._scores: Dict[str, float] = {}
        self._errors: int = 0
        logger.info("[FRINGE-PANIC] Reddit Monitor: initialized")

    async def poll(self) -> Dict[str, float]:
        """Check post volume in panic subreddits."""
        import aiohttp

        headers = {"User-Agent": "HMATS-FringePanic/1.0"}
        scores = {}

        try:
            async with create_session() as session:
                for sub in PANIC_SUBREDDITS:
                    try:
                        url = f"https://www.reddit.com/r/{sub}/new.json?limit=100"
                        async with session.get(
                            url, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=15),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                posts = data.get("data", {}).get("children", [])

                                cutoff = time.time() - 4 * 3600
                                recent_count = sum(
                                    1 for p in posts
                                    if p.get("data", {}).get("created_utc", 0) > cutoff
                                )

                                self._history[sub].append((time.time(), recent_count))

                                counts = [c for _, c in self._history[sub]]
                                if len(counts) >= 6:
                                    mean = float(np.mean(counts))
                                    std = float(max(np.std(counts), 1.0))
                                    zscore = (recent_count - mean) / std
                                else:
                                    zscore = 0.0

                                scores[sub] = zscore

                                if zscore > 2.0:
                                    logger.warning(
                                        f"[FRINGE-PANIC] Reddit SPIKE: r/{sub} "
                                        f"posts_4h={recent_count} zscore={zscore:.1f}"
                                    )

                            await asyncio.sleep(2.0)  # Reddit rate limit
                    except Exception as e:
                        logger.debug(f"[FRINGE-PANIC] Reddit r/{sub}: {e}")
        except Exception as e:
            self._errors += 1
            logger.debug(f"[FRINGE-PANIC] Reddit poll error: {e}")

        self._scores = scores
        return scores

    def get_composite_score(self) -> float:
        """0.0-1.0 from top-3 subreddit z-scores."""
        if not self._scores:
            return 0.0
        zscores = list(self._scores.values())
        top3 = sorted(zscores, reverse=True)[:3]
        avg_z = float(np.mean(top3))
        return float(np.clip(1.0 / (1.0 + np.exp(-avg_z)), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# FUSION: Combine all 4 layers
# ─────────────────────────────────────────────────────────────

class FringePanicFusion:
    """
    [FRINGE-PANIC] Fuses X + Google Trends + Reddit + CryptoPanic.

    Weights:
      X API:          0.35 (most real-time, richest narrative)
      Google Trends:  0.30 (broadest coverage)
      Reddit:         0.15 (direct panic community activity)
      CryptoPanic:    0.20 (already exists, crypto-specific)
    """

    WEIGHTS = {
        "x_api": 0.35,
        "google_trends": 0.30,
        "reddit": 0.15,
        "crypto_panic": 0.20,
    }

    def __init__(
        self,
        x_monitor: Optional[XFringePanicMonitor] = None,
        trends_monitor: Optional[GoogleTrendsPanicMonitor] = None,
        reddit_monitor: Optional[RedditPanicMonitor] = None,
    ):
        self._x = x_monitor or XFringePanicMonitor()
        self._trends = trends_monitor or GoogleTrendsPanicMonitor()
        self._reddit = reddit_monitor or RedditPanicMonitor()

        self._last_result: Optional[Dict[str, Any]] = None
        self._last_poll_time: float = 0
        self._errors: int = 0

        logger.info("[FRINGE-PANIC] Fusion: initialized")

    async def poll_all(self):
        """Poll all layers in parallel. Called every 4H tick."""
        try:
            await asyncio.gather(
                self._x.poll_search(),
                self._trends.poll(),
                self._reddit.poll(),
                return_exceptions=True,
            )
            self._last_poll_time = time.time()
        except Exception as e:
            self._errors += 1
            logger.debug(f"[FRINGE-PANIC] poll_all error: {e}")

    def compute_fringe_panic_score(
        self,
        crypto_panic_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute composite fringe panic assessment.

        For short-biased system:
          CALM/ELEVATED: no effect
          HIGH: panic spreading -> short tailwind (leverage +0.1~0.2)
          EXTREME: near capitulation -> short gold but squeeze risk rising
        """
        try:
            return self._compute(crypto_panic_score)
        except Exception as e:
            self._errors += 1
            logger.debug(f"[FRINGE-PANIC] compute error: {e}")
            return self._neutral_result()

    def _compute(self, crypto_panic_score: float) -> Dict[str, Any]:
        x_score = self._x.get_composite_score()
        trends_score = self._trends.get_composite_score()
        reddit_score = self._reddit.get_composite_score()

        composite = (
            x_score * self.WEIGHTS["x_api"]
            + trends_score * self.WEIGHTS["google_trends"]
            + reddit_score * self.WEIGHTS["reddit"]
            + crypto_panic_score * self.WEIGHTS["crypto_panic"]
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # Level classification
        if composite < 0.3:
            level = "CALM"
        elif composite < 0.55:
            level = "ELEVATED"
        elif composite < 0.75:
            level = "HIGH"
        else:
            level = "EXTREME"

        # Short-bias interpretation
        if level == "CALM":
            short_signal = "NEUTRAL"
            leverage_adj = 0.0
        elif level == "ELEVATED":
            short_signal = "NEUTRAL"
            leverage_adj = 0.05
        elif level == "HIGH":
            short_signal = "FAVORABLE"
            leverage_adj = 0.15
        else:  # EXTREME
            active_spikes = self._x.get_active_spikes()
            if len(active_spikes) >= 4:
                short_signal = "SQUEEZE_RISK"
                leverage_adj = 0.05
            else:
                short_signal = "FAVORABLE"
                leverage_adj = 0.25

        result = {
            "fringe_panic_score": round(composite, 3),
            "panic_level": level,
            "active_spike_groups": self._x.get_active_spikes(),
            "x_score": round(x_score, 3),
            "trends_score": round(trends_score, 3),
            "reddit_score": round(reddit_score, 3),
            "crypto_panic_score": round(crypto_panic_score, 3),
            "short_signal": short_signal,
            "leverage_adjustment": round(leverage_adj, 3),
            "source": "fringe_panic",
        }

        self._last_result = result
        return result

    @staticmethod
    def _neutral_result() -> Dict[str, Any]:
        return {
            "fringe_panic_score": 0.0,
            "panic_level": "CALM",
            "active_spike_groups": [],
            "x_score": 0.0,
            "trends_score": 0.0,
            "reddit_score": 0.0,
            "crypto_panic_score": 0.0,
            "short_signal": "NEUTRAL",
            "leverage_adjustment": 0.0,
            "source": "fringe_panic",
        }

    def get_status(self) -> Dict[str, Any]:
        """Diagnostics."""
        return {
            "errors": self._errors,
            "x_errors": self._x._errors,
            "trends_errors": self._trends._errors,
            "reddit_errors": self._reddit._errors,
            "x_mock_mode": self._x._mock_mode,
            "trends_available": PYTRENDS_AVAILABLE,
            "last_poll_time": self._last_poll_time,
            "last_result": self._last_result,
        }


# ─────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────

_instance: Optional[FringePanicFusion] = None


def get_fringe_panic_detector(
    x_monitor: Optional[XFringePanicMonitor] = None,
    trends_monitor: Optional[GoogleTrendsPanicMonitor] = None,
    reddit_monitor: Optional[RedditPanicMonitor] = None,
) -> FringePanicFusion:
    """Get or create singleton."""
    global _instance
    if _instance is None:
        _instance = FringePanicFusion(
            x_monitor=x_monitor,
            trends_monitor=trends_monitor,
            reddit_monitor=reddit_monitor,
        )
    elif (
        (x_monitor is not None and getattr(_instance, "_x", None) is not x_monitor)
        or (trends_monitor is not None and getattr(_instance, "_trends", None) is not trends_monitor)
        or (reddit_monitor is not None and getattr(_instance, "_reddit", None) is not reddit_monitor)
    ):
        logger.info("[FRINGE-PANIC] constructor inputs changed; refreshing singleton instance")
        _instance = FringePanicFusion(
            x_monitor=x_monitor,
            trends_monitor=trends_monitor,
            reddit_monitor=reddit_monitor,
        )
    return _instance


def reset_fringe_panic_detector():
    """Reset singleton."""
    global _instance
    _instance = None

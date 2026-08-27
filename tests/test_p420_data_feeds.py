"""[P420] Feed-layer absence honesty — fork-3 tasks 6-11.

  6. CC News caches XRP/BNB (they were tradeable and served the UNFILTERED
     corpus at one quota call each); an asset outside the roster returns []
     without reserving quota.
  7. The GCI "N consecutive DAYS of outflow" streak advances once per
     completed day (it advanced per HOURLY update) and survives a restart
     without double-counting.
  8. CoinGlass: a first-boot total outage returns None instead of a
     fresh-stamped all-zero snapshot.
  9. LunarCrush: a fetch in which no symbol returned metrics keeps the
     previous cache and does not stamp _last_fetch_time.
 10. F&G: a fetch failure serves the previous REAL tick (original
     timestamp) or a tick FLAGGED is_mock — never mock-50 under the real
     source's name.
 11. CryptoPanic: the P319 combined-currency probe runs on a LIVE session.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


class _Session:
    """aiohttp-shaped stand-in: `closed` flips when the CM exits; `get`
    raises so every real endpoint reads as an outage."""
    def __init__(self):
        self.closed = False

    def get(self, *a, **k):
        raise ConnectionError("outage")


@contextlib.asynccontextmanager
async def _session_cm():
    s = _Session()
    try:
        yield s
    finally:
        s.closed = True


# ---------------------------------------------------------------------------
# 6 — CC News
# ---------------------------------------------------------------------------
class TestCCNewsRoster:
    def test_roster_covers_every_tradeable_asset(self):
        from data_mgmt.feeds.cryptocompare_news_feed import TRACKED_CATEGORIES
        assert {"BTC", "ETH", "SOL", "XRP", "BNB"} <= set(TRACKED_CATEGORIES)

    def test_fanout_still_iterates_the_roster_with_one_call(self):
        import data_mgmt.feeds.cryptocompare_news_feed as m
        src = inspect.getsource(m)
        assert "for _cat in TRACKED_CATEGORIES:" in src
        assert '",".join(TRACKED_CATEGORIES)' in src

    def test_an_untracked_asset_returns_empty_without_touching_the_quota(
            self, monkeypatch, caplog):
        import data_mgmt.feeds.cryptocompare_news_feed as m
        import data_mgmt.feeds._cc_quota as q

        class _Quota:
            def try_consume(self, *a, **k):
                raise AssertionError("quota reserved for an untracked asset")
        monkeypatch.setattr(q, "get_cc_quota", lambda: _Quota())
        f = m.CCNewsFeed(api_key="k")
        with caplog.at_level(logging.WARNING):
            out = asyncio.run(f.fetch_headlines("DOGE"))
            out2 = asyncio.run(f.fetch_headlines("DOGE"))
        assert out == [] and out2 == []
        msgs = [r.getMessage() for r in caplog.records
                if "not in TRACKED_CATEGORIES" in r.getMessage()]
        assert len(msgs) == 1, "warn ONCE per asset, not per tick"

    def test_the_untracked_gate_precedes_the_quota_reservation(self):
        import data_mgmt.feeds.cryptocompare_news_feed as m
        src = inspect.getsource(m.CCNewsFeed.fetch_headlines)
        assert src.index("cache_key not in TRACKED_CATEGORIES") < \
            src.index("try_consume(")


# ---------------------------------------------------------------------------
# 7 — GCI ETF streak
# ---------------------------------------------------------------------------
class TestEtfStreakByDay:
    def test_five_updates_in_one_day_count_once(self):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        for _ in range(5):
            t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-26")
        assert t.get_streak("BTC") == -1

    def test_a_new_day_advances_and_a_flip_resets(self):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-25")
        t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-26")
        t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-27")
        assert t.get_streak("BTC") == -3
        t._calculate_streak_for_asset("BTC", +50.0, day_iso="2026-08-28")
        assert t.get_streak("BTC") == 1

    def test_restart_inside_a_counted_day_does_not_double_count(self, tmp_path):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-25")
        t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-26")
        assert (tmp_path / "etf_streak_state.json").exists()
        t2 = ETFFlowTracker()                       # restart
        t2._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-26")
        assert t2.get_streak("BTC") == -2, (
            "the restart re-counted the day the previous process already "
            "counted")
        t2._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-27")
        assert t2.get_streak("BTC") == -3

    def test_streaks_are_per_asset(self):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-26")
        assert t.get_streak("ETH") == 0

    def test_legacy_no_day_path_is_unchanged_and_touches_no_disk(self, tmp_path):
        from data_mgmt.global_context_informer import ETFFlowTracker
        t = ETFFlowTracker()
        for _ in range(3):
            t._calculate_streak_for_asset("BTC", -100.0)
        assert t.get_streak("BTC") == -3
        assert not (tmp_path / "etf_streak_state.json").exists()

    def test_the_live_aggregate_path_passes_the_completed_day(self):
        from data_mgmt import global_context_informer as g
        src = inspect.getsource(g.ETFFlowTracker.fetch_all_flows_async)
        assert 'day_iso=getattr(_agg, "flow_day", None)' in src
        src2 = inspect.getsource(g.ETFFlowTracker._fetch_aggregate_flow_coinglass)
        assert "flow_day=" in src2

    def test_a_corrupt_state_file_is_a_cold_streak(self, tmp_path, caplog):
        from data_mgmt.global_context_informer import ETFFlowTracker
        (tmp_path / "etf_streak_state.json").write_text("{not json",
                                                        encoding="utf-8")
        t = ETFFlowTracker()
        with caplog.at_level(logging.WARNING):
            t._calculate_streak_for_asset("BTC", -100.0, day_iso="2026-08-26")
        assert t.get_streak("BTC") == -1
        assert any("unreadable" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 8 — CoinGlass first-boot outage
# ---------------------------------------------------------------------------
class TestCoinglassColdOutage:
    def test_total_outage_with_no_cache_returns_absent(self, monkeypatch, caplog):
        import data_mgmt.feeds.coinglass_feed as m
        monkeypatch.setattr(m, "create_session", _session_cm)
        f = m.CoinglassFeed(api_key="k")
        assert f._last_data is None
        with caplog.at_level(logging.WARNING):
            out = asyncio.run(f._fetch_real())
        assert out is None, "a fresh all-zero snapshot was fabricated"
        assert f._last_data is None
        assert f._last_fetch_time is None, (
            "a failed fetch stamped the throttle clock")
        assert any("cold cache" in r.getMessage() for r in caplog.records)

    def test_total_outage_with_a_cache_still_serves_it(self, monkeypatch):
        import data_mgmt.feeds.coinglass_feed as m
        monkeypatch.setattr(m, "create_session", _session_cm)
        f = m.CoinglassFeed(api_key="k")
        prev = m.CoinglassCrowdData(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            staleness_sec=0.0)
        f._last_data = prev
        out = asyncio.run(f._fetch_real())
        assert out is prev and out.staleness_sec > 3000


# ---------------------------------------------------------------------------
# 9 — LunarCrush non-200
# ---------------------------------------------------------------------------
class TestLunarCrushNon200:
    def _feed(self, monkeypatch):
        import data_mgmt.feeds.lunarcrush_feed as m
        monkeypatch.setattr(m, "create_session", _session_cm)
        f = m.LunarCrushFeed(api_key="k")

        async def _none(session, headers, symbol):
            f._last_status_by_symbol[symbol] = 503
            return None
        monkeypatch.setattr(f, "_fetch_coin_metrics", _none)
        return m, f

    def test_no_metrics_keeps_the_previous_cache(self, monkeypatch, caplog):
        m, f = self._feed(monkeypatch)
        prev = m.LunarCrushAttentionData(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            staleness_sec=0.0)
        f._last_data = prev
        f._last_fetch_time = prev.timestamp
        with caplog.at_level(logging.WARNING):
            out = asyncio.run(f._fetch_real())
        assert out is prev
        assert f._last_fetch_time == prev.timestamp, (
            "a failed fetch was stamped fresh")
        assert any("NO symbol" in r.getMessage() and "503" in r.getMessage()
                   for r in caplog.records)

    def test_no_metrics_and_no_cache_is_absent(self, monkeypatch):
        m, f = self._feed(monkeypatch)
        assert asyncio.run(f._fetch_real()) is None
        assert f._last_fetch_time is None

    def test_the_non_200_branch_logs_the_status(self):
        import data_mgmt.feeds.lunarcrush_feed as m
        src = inspect.getsource(m.LunarCrushFeed._fetch_coin_metrics)
        i = src.index("if resp.status != 200:")
        assert "logger.warning" in src[i:i + 200]


# ---------------------------------------------------------------------------
# 10 — F&G fallback
# ---------------------------------------------------------------------------
class TestFearGreedFallback:
    def _feed(self, monkeypatch):
        from data_mgmt.feeds import sentiment_feed as m
        f = m.SentimentFeed(source=m.SentimentDataSource.ALTERNATIVE_ME)

        async def _fail_to_mock():
            return await f._fetch_mock()
        monkeypatch.setattr(f, "_fetch_from_source", _fail_to_mock)
        return m, f

    def test_with_no_prior_tick_the_mock_is_flagged(self, monkeypatch):
        m, f = self._feed(monkeypatch)
        t = asyncio.run(f.fetch())
        assert t is not None
        assert t.is_mock is True and t.source == "mock"
        assert t.confidence == 0.0
        assert t.source != m.SentimentDataSource.ALTERNATIVE_ME.value
        assert f._last_tick is None, "a mock entered the cache"

    def test_with_a_prior_real_tick_it_is_served_with_its_own_timestamp(
            self, monkeypatch):
        m, f = self._feed(monkeypatch)
        # the real alternative_me path stamps NAIVE local `datetime.now()`
        # (sentiment_feed._fetch_alternative_me) and `fetch()` measures
        # staleness against naive local now — mirror that, not UTC-aware
        _t3 = (datetime.now() - timedelta(hours=3)).isoformat()
        real_raw = {
            "timestamp": _t3,
            "fear_greed": {"value": 31, "label": "fear", "timestamp": _t3},
            "social_sentiments": [], "news_sentiments": [],
            "funding_rates": [], "_source": "alternative_me", "_mock": False,
        }
        real = f._parse_raw_data(real_raw)
        assert real.is_mock is False and real.source == "alternative_me"
        f._last_tick = real
        t = asyncio.run(f.fetch())
        assert t is real
        assert t.fear_greed.value == 31
        assert t.staleness_sec > 3 * 3600 - 60, "staleness must show its age"
        assert f._last_tick is real

    def test_a_real_fetch_is_untouched(self):
        from data_mgmt.feeds import sentiment_feed as m
        f = m.SentimentFeed(source=m.SentimentDataSource.ALTERNATIVE_ME)
        t = f._parse_raw_data({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fear_greed": {"value": 70, "label": "greed",
                           "timestamp": datetime.now(timezone.utc).isoformat()},
            "social_sentiments": [], "news_sentiments": [],
            "funding_rates": [], "_source": "alternative_me", "_mock": False})
        assert t.is_mock is False and t.source == "alternative_me"

    def test_explicit_mock_source_still_serves_mock_ticks(self):
        """a feed CONFIGURED as MOCK is not a fallback; it keeps working"""
        from data_mgmt.feeds import sentiment_feed as m
        f = m.SentimentFeed(source=m.SentimentDataSource.MOCK)
        t = asyncio.run(f.fetch())
        assert t is not None and t.is_mock is True and t.source == "mock"
        assert f._last_tick is t


# ---------------------------------------------------------------------------
# 11 — CryptoPanic probe session
# ---------------------------------------------------------------------------
class TestCryptoPanicProbeSession:
    def test_the_probe_runs_on_an_open_session(self, monkeypatch, caplog):
        import data_mgmt.feeds.cryptopanic_feed as m
        monkeypatch.setattr(m, "create_session", _session_cm)
        f = m.CryptoPanicFeed(api_key="k")
        calls = []

        async def _posts(session, currency):
            assert not session.closed, (
                "the probe ran on a CLOSED session (RuntimeError live)")
            calls.append(currency)
            return [m.NewsItem(
                id=f"{currency}-1", title="t", source="s", url="u",
                published_at=datetime.now(timezone.utc),
                currencies=[c for c in currency.split(",")],
                votes_positive=1, votes_negative=0, votes_important=0,
                votes_liked=0, votes_disliked=0, votes_lol=0, votes_toxic=0,
                votes_saved=0, sentiment_score=0.1)]
        monkeypatch.setattr(f, "_fetch_posts", _posts)
        f._last_status_code = 200
        with caplog.at_level(logging.WARNING):
            out = asyncio.run(f._fetch_real())
        assert out is not None
        assert f._combined_probe_done is True, "the probe did not complete"
        assert any("," in c for c in calls), "the combined request never ran"
        assert not any("probe failed" in r.getMessage()
                       for r in caplog.records)

    def test_source_pin_probe_opens_its_own_session(self):
        import data_mgmt.feeds.cryptopanic_feed as m
        src = inspect.getsource(m.CryptoPanicFeed._fetch_real)
        g = src.index("if (not _cp_failed and not _cp_errors")
        blk = src[g:g + 1200]
        assert "async with create_session() as _probe_session" in blk
        assert "_probe_session, " in blk

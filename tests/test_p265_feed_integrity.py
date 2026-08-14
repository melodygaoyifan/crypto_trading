"""[P265] Feed-integrity fixes.

  * CoinGlass: an all-endpoints-failed fetch built a fresh-stamped
    CoinglassCrowdData full of computed zeros and OVERWROTE the cached good
    data (defeating the P100 staleness machinery — the timestamp lied); the
    failure-warning latch never re-armed after recovery; change_24h_pct
    actually carried the 4-HOUR OI change (h4OIChangePercent) and
    change_1h_pct carried a VOLUME change (h1VolChangePercent) — probe-
    verified against live values 2026-08-14; and the OI sum double-counted
    the exchangeName="All" aggregate row (~2x).
  * CryptoPanic: the tick that ARMED the 429 backoff also computed metrics
    over the empty corpus, overwrote the cache AND persisted it — destroying
    exactly the cache the backoff was designed to serve (P154 defeated one
    layer up).
  * WhaleDetector: one global size deque blended BTC/ETH/SOL notionals
    (SOL detection suppressed, BTC inflated); signals were stamped at
    DETECTION time and overlapping fetch_trades windows re-detected the same
    trade every tick — days-old flow read as "pressure in the last hour".
  * TrendDecisionLayer: cached closes had no age bound — under enforce, a
    TA-cache outage kept asserting a frozen trend forever (P156 unapplied to
    the book's only driver).
  * main.py read coinglass_feed._last_data directly (skipping the
    staleness-aware get_latest) and injected without any age gate.
  * data_age was the constant 0.5 (Kraken's ticker has no timestamp; 35,974
    identical log lines) — now the fetch-latency approximation plus
    frozen-content detection, flagged via data_age_measured.
"""

import asyncio
import re
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# CoinGlass
# ---------------------------------------------------------------------------

class _FailSession:
    def get(self, *a, **k):
        raise ConnectionError("api down")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _cg_feed(monkeypatch, last_data=None):
    import data_mgmt.feeds.coinglass_feed as cgmod
    feed = cgmod.CoinglassFeed(api_key="test-key")
    feed._mock_mode = False
    if last_data is not None:
        feed._last_data = last_data
    monkeypatch.setattr(cgmod, "create_session", lambda: _FailSession())
    return cgmod, feed


def _good_cg_data(cgmod):
    d = cgmod.CoinglassCrowdData(
        timestamp=datetime.now(timezone.utc), staleness_sec=0.0)
    d.funding_bias["BTC"] = -0.42
    d.liquidation_imbalance["BTC"] = -0.53
    return d


class TestCoinglassOutageKeepsCache:
    def test_total_outage_does_not_overwrite_the_cache(self, monkeypatch):
        cgmod, feed = _cg_feed(monkeypatch)
        good = _good_cg_data(cgmod)
        feed._last_data = good
        out = asyncio.run(feed._fetch_real())
        assert out is good, (
            "a zero-content fetch replaced the cache — fabricated fresh "
            "zeros defeat every staleness-aware consumer (P265)")
        assert feed._last_data is good
        assert good.funding_bias["BTC"] == pytest.approx(-0.42)
        assert out.staleness_sec > 0.0 or (
            datetime.now(timezone.utc) - good.timestamp).total_seconds() < 2

    def test_warning_latch_rearms_on_recovery(self, monkeypatch):
        import data_mgmt.feeds.coinglass_feed as cgmod
        feed = cgmod.CoinglassFeed(api_key="test-key")
        feed._dns_warning_shown = True

        class _OkSession(_FailSession):
            pass
        # Simulate recovery by injecting content directly through the
        # post-session logic: run _fetch_real with a failing session but a
        # pre-populated data object is not reachable — so exercise the latch
        # logic via the module-level branch contract instead: source pin.
        src = (REPO / "data_mgmt" / "feeds" / "coinglass_feed.py").read_text(
            encoding="utf-8")
        assert "_dns_warning_shown = False" in src, (
            "the failure-warning latch never re-arms — one bad 5-minute "
            "cycle silences the feed's failure telemetry for the process "
            "lifetime")
        assert "feed recovered" in src


class TestCoinglassFieldSemantics:
    def _src(self):
        return (REPO / "data_mgmt" / "feeds" / "coinglass_feed.py").read_text(
            encoding="utf-8")

    def test_24h_field_reads_the_24h_key(self):
        src = self._src()
        assert 'first.get("h24Change"' in src, (
            "change_24h_pct no longer reads h24Change — it carried the "
            "4-HOUR change before, compressing every 24h-calibrated "
            "consumer (P265, probe-verified)")
        assert re.search(r"h4_change\s*=\s*float\(first", src) is None

    def test_1h_field_reads_oi_not_volume(self):
        src = self._src()
        assert 'first.get("h1OIChangePercent"' in src
        assert 'first.get("h1VolChangePercent"' not in src, (
            "change_1h_pct reads a VOLUME change again")

    def test_the_all_row_is_not_double_counted(self):
        src = self._src()
        assert '"all"' in src.lower() and "_all_row" in src, (
            "the exchangeName='All' aggregate row is summed together with "
            "the per-exchange rows again — total OI ~2x")


# ---------------------------------------------------------------------------
# CryptoPanic
# ---------------------------------------------------------------------------

class TestCryptoPanic429KeepsCache:
    def test_rate_limited_fetch_serves_the_cache(self, monkeypatch, tmp_path):
        import data_mgmt.feeds.cryptopanic_feed as cpmod
        feed = cpmod.CryptoPanicFeed(api_key="k")
        monkeypatch.setattr(feed, "_state_path",
                            lambda: str(tmp_path / "cp.json"),
                            raising=False)
        good = cpmod.CryptoPanicData(
            timestamp=datetime.now(timezone.utc), staleness_sec=0.0)
        good.panic_score["BTC"] = 0.7
        feed._last_data = good

        async def _limited(session, currency):
            feed._last_status_code = 429
            return []
        monkeypatch.setattr(feed, "_fetch_posts", _limited)
        monkeypatch.setattr(cpmod, "create_session", lambda: _FailSession())
        out = asyncio.run(feed._fetch_real())
        assert out is good, (
            "the 429 tick overwrote the cache with an empty dataset — the "
            "backoff now serves nothing and a restart persists the "
            "emptiness (P265)")
        assert feed._last_data is good

    def test_a_successful_fetch_still_updates(self, monkeypatch, tmp_path):
        import data_mgmt.feeds.cryptopanic_feed as cpmod
        feed = cpmod.CryptoPanicFeed(api_key="k")
        monkeypatch.setattr(feed, "_state_path",
                            lambda: str(tmp_path / "cp.json"),
                            raising=False)
        old = cpmod.CryptoPanicData(
            timestamp=datetime.now(timezone.utc), staleness_sec=0.0)
        feed._last_data = old
        item = cpmod.NewsItem(
            id="n1", title="BTC breaks resistance", source="s", url="u",
            published_at=datetime.now(timezone.utc), currencies=["BTC"],
            votes_positive=1, votes_negative=0, votes_important=0,
            votes_liked=0, votes_disliked=0, votes_lol=0, votes_toxic=0,
            votes_saved=0, sentiment_score=0.2)

        async def _ok(session, currency):
            feed._last_status_code = 200
            return [item]
        monkeypatch.setattr(feed, "_fetch_posts", _ok)
        monkeypatch.setattr(cpmod, "create_session", lambda: _FailSession())
        out = asyncio.run(feed._fetch_real())
        assert out is not old
        assert feed._last_data is out
        assert out.recent_news


# ---------------------------------------------------------------------------
# WhaleDetector
# ---------------------------------------------------------------------------

class TestWhalePerAssetBaselines:
    def _fresh(self):
        from agents.whale_detector import WhaleDetector
        return WhaleDetector()

    def test_sol_baseline_is_not_contaminated_by_btc(self):
        wd = self._fresh()
        # BTC: 100 x $500k trades (huge tickets)
        for i in range(100):
            wd.detect("BTC", 500_000.0, "BUY", trade_id=f"b{i}")
        # SOL: 100 x $2k trades — its own world
        for i in range(100):
            wd.detect("SOL", 2_000.0, "SELL", trade_id=f"s{i}")
        # A $30k SOL trade is 15x SOL's average -> RELATIVE_SIZE must fire.
        sig = wd.detect("SOL", 30_000.0, "BUY", trade_id="s-big")
        dims = [d[0] for d in sig.details]
        assert "RELATIVE_SIZE" in dims, (
            "SOL's 15x-average trade did not trigger RELATIVE_SIZE — its "
            "baseline is still blended with BTC's $500k tickets (P265)")

    def test_duplicate_trade_ids_are_not_redetected(self):
        wd = self._fresh()
        first = wd.detect("BTC", 500_000.0, "BUY",
                          orderbook_depth_usd=1_000_000.0, trade_id="T1",
                          trade_ts=time.time())
        assert first.is_whale
        second = wd.detect("BTC", 500_000.0, "BUY",
                           orderbook_depth_usd=1_000_000.0, trade_id="T1",
                           trade_ts=time.time())
        assert not second.is_whale, (
            "the same venue trade id was re-detected — overlapping "
            "fetch_trades windows re-stamp old whales as fresh flow (P265)")
        assert len(wd._whale_signals["BTC"]) == 1

    def test_pressure_uses_the_trades_own_time(self):
        wd = self._fresh()
        old_ts = time.time() - 7 * 24 * 3600  # a week ago
        wd.detect("BTC", 500_000.0, "BUY",
                  orderbook_depth_usd=1_000_000.0, trade_id="OLD",
                  trade_ts=old_ts)
        p = wd.get_pressure("BTC")
        assert p.whale_count == 0, (
            "a week-old trade counted as pressure in the last hour — "
            "signals are stamped at detection time again (P265)")

    def test_fresh_trades_still_register(self):
        wd = self._fresh()
        wd.detect("BTC", 500_000.0, "SELL",
                  orderbook_depth_usd=1_000_000.0, trade_id="NEW",
                  trade_ts=time.time() - 60)
        p = wd.get_pressure("BTC")
        assert p.whale_count == 1
        assert p.net_pressure < 0


# ---------------------------------------------------------------------------
# TrendDecisionLayer staleness
# ---------------------------------------------------------------------------

class TestTrendClosesStaleness:
    def _layer(self):
        from core.trend_decision_layer import TrendDecisionLayer
        layer = TrendDecisionLayer(mode="enforce")
        layer._strat = types.SimpleNamespace(
            compute=lambda closes: {"signal": 0.9},
            min_history=lambda: 10)
        layer._closes["BTC"] = [100.0 + i for i in range(300)]
        return layer

    def test_stale_closes_refuse_to_assert_a_trend(self):
        layer = self._layer()
        layer._closes_cached_at["BTC"] = time.time() - 10 * 3600  # 10h old
        agent_signals: dict = {}
        market_data: dict = {"quant_direction": 0.0}
        out = layer.process("BTC", None, agent_signals, market_data)
        assert out is None, (
            "a 10h-old frozen close series still asserted a trend whose "
            "40bps constant clears the gate (P265/P156)")
        assert "quant_direction" not in agent_signals or \
            not agent_signals.get("quant_direction")

    def test_one_missed_refresh_is_tolerated(self):
        layer = self._layer()
        layer._closes_cached_at["BTC"] = time.time() - 5 * 3600  # 5h < 8h
        out = layer.process("BTC", None, {}, {"quant_direction": 0.0})
        assert out is not None


# ---------------------------------------------------------------------------
# main.py injection + data_age (source pins; falsification-checked)
# ---------------------------------------------------------------------------

class TestInjectionAndAgePins:
    def test_coinglass_injection_goes_through_get_latest(self):
        start = MAIN_SRC.index("[COINGLASS] cached data is")
        seg = MAIN_SRC[max(0, start - 3000):start + 500]
        assert "get_latest()" in seg
        assert "staleness_sec" in seg

    def test_coinglass_direct_last_data_read_is_gone(self):
        assert re.search(
            r"cg_data\s*=\s*self\.coinglass_feed\._last_data", MAIN_SRC) is None, (
            "main.py reads coinglass_feed._last_data directly again — the "
            "staleness-aware accessor is bypassed and stale values inject "
            "as current")

    def test_data_age_is_no_longer_the_constant(self):
        src = (REPO / "data_mgmt" / "market_data_pipeline.py").read_text(
            encoding="utf-8")
        assert re.search(r"data_age\s*=\s*0\.5\s*$", src, re.M) is None, (
            "the bare data_age = 0.5 constant is back (35,974 identical log "
            "lines of age=0.50s)")
        assert "data_age_measured" in src
        assert "_ticker_fingerprint" in src

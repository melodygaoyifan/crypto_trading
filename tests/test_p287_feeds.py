"""[P287] Feeds batch — the P265f corners that survived, pinned.

Covers the P287 feeds fixes:
  1. CoinGlass partial-fetch carries the failed FAMILY forward with honest
     age instead of fabricating fresh zeros (and carried entries never
     re-enter the trend history deques).
  2. CoinGlass clamps extreme funding prints instead of zeroing/dropping
     them (a squeeze-level print must not read as "no crowding").
  3. CryptoPanic partial fetch carries the FAILED currencies' cached items
     (own timestamps) instead of re-stamping their panic as 0.0/fresh.
  4. CryptoPanic stamps unparseable published_at as EPOCH, never NOW.
  5. CC News: the 429 backoff gate runs BEFORE the shared-quota
     reservation (no quota burned for calls that are never made).
  6. BinanceFlowFeed: per-(asset, cause) warn latch, not one global bool.
  7. flow_features: latest_fv2_vector serves the last COMPLETE 4H bucket
     (the in-progress resample bucket never becomes a "full bar").
  8. market_data_pipeline: OFI history is fresh-orderbook-gated (source
     pin — the site lives deep inside the async fetch path).
  9. market_data_pipeline emits the top-of-book keys micro's follower
     ingest actually reads (reader/writer pinned together, P2).
 10. kraken_quant annualizes at the real 4H cadence.
"""

import asyncio
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_mgmt.feeds import coinglass_feed as cg_mod  # noqa: E402
from data_mgmt.feeds.coinglass_feed import (  # noqa: E402
    CoinglassFeed, FundingData, LiquidationData, OpenInterestData,
    SUPPORTED_SYMBOLS,
)
from data_mgmt.feeds import cryptopanic_feed as cp_mod  # noqa: E402
from data_mgmt.feeds.cryptopanic_feed import (  # noqa: E402
    CryptoPanicData, CryptoPanicFeed, NewsItem,
)
from data_mgmt.flow_features import (  # noqa: E402
    FV2_COLUMNS, _agg_4h, cross_asset_features, flow_features_4h,
    latest_fv2_vector,
)

UTC = timezone.utc


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _NullAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _json_resp(payload):
    class _Resp(_NullAsyncCM):
        status = 200

        async def json(self):
            return payload
    return _Resp()


def _json_sess(payload):
    class _Sess:
        def get(self, *a, **k):
            return _json_resp(payload)
    return _Sess()


# ---------------------------------------------------------------------------
# 1+2. CoinGlass
# ---------------------------------------------------------------------------

def _mk_cg_feed(monkeypatch):
    feed = CoinglassFeed(api_key="test-key", mock_mode=False)
    monkeypatch.setattr(cg_mod, "create_session", lambda: _NullAsyncCM())
    return feed


def _patch_cg_endpoints(monkeypatch, feed, ts, *, funding_ok=True, liq_ok=True):
    async def oi_and_funding(session, headers, symbol):
        oi = OpenInterestData(symbol, 1e9, 1e4, 5.0, 0.1, ts)
        fr = (FundingData(symbol, 0.0002, None, 8, "aggregate", ts)
              if funding_ok else None)
        return oi, fr

    async def funding_detailed(session, headers):
        return {}

    async def liq(session, data):
        if not liq_ok:
            raise RuntimeError("v3 down")
        for s in SUPPORTED_SYMBOLS:
            data.liquidations[s] = LiquidationData(s, 30e6, 10e6, 40e6, 5e6, ts)

    monkeypatch.setattr(feed, "_fetch_oi_and_funding", oi_and_funding)
    monkeypatch.setattr(feed, "_fetch_funding_detailed", funding_detailed)
    monkeypatch.setattr(feed, "_fetch_liquidation_v3", liq)


class TestCoinglassPartialFetchCarry:
    def test_failed_family_is_carried_with_original_timestamp_not_fresh_zero(
            self, monkeypatch):
        feed = _mk_cg_feed(monkeypatch)
        ts1 = datetime.now(UTC) - timedelta(minutes=10)
        _patch_cg_endpoints(monkeypatch, feed, ts1, liq_ok=True)
        d1 = _run(feed.fetch())
        assert d1.liquidation_imbalance["BTC"] == pytest.approx(0.5)

        # Second fetch: liquidation family down, OI+funding fine.
        _patch_cg_endpoints(monkeypatch, feed, datetime.now(UTC), liq_ok=False)
        d2 = _run(feed.fetch())
        # The defect this pins: pre-P287 the failed family computed to a
        # FRESH-stamped 0.0 ("endpoint down" == "calm market").
        assert d2.liquidation_imbalance["BTC"] == pytest.approx(0.5), (
            "liquidation family failed but its metric was zeroed — the "
            "carry-forward is not working")
        assert d2.liquidations["BTC"].timestamp == ts1, (
            "carried entry must keep its ORIGINAL timestamp (honest age)")
        assert d2.family_age_sec.get("liquidations:BTC", 0.0) >= 500, (
            "carried family must expose its age to consumers")
        # Fresh families carry no age marker.
        assert "funding:BTC" not in d2.family_age_sec

    def test_carried_entries_do_not_reenter_history_deques(self, monkeypatch):
        feed = _mk_cg_feed(monkeypatch)
        ts1 = datetime.now(UTC) - timedelta(minutes=10)
        _patch_cg_endpoints(monkeypatch, feed, ts1, funding_ok=True)
        _run(feed.fetch())
        assert len(feed._funding_history["BTC"]) == 1

        # funding family down for 3 consecutive fetches
        for _ in range(3):
            _patch_cg_endpoints(monkeypatch, feed, datetime.now(UTC),
                                funding_ok=False)
            d = _run(feed.fetch())
            # metric still computed from the carried datum, not zeroed
            assert d.funding_bias["BTC"] == pytest.approx(
                np.clip(0.0002 / 0.0003, -1, 1))
        assert len(feed._funding_history["BTC"]) == 1, (
            "a carried (stale) funding value re-entered the trend history — "
            "repeated stale appends collapse the rolling stats")

    def test_crowd_metrics_expose_family_ages(self, monkeypatch):
        feed = _mk_cg_feed(monkeypatch)
        ts1 = datetime.now(UTC) - timedelta(minutes=10)
        _patch_cg_endpoints(monkeypatch, feed, ts1)
        _run(feed.fetch())
        _patch_cg_endpoints(monkeypatch, feed, datetime.now(UTC), liq_ok=False)
        _run(feed.fetch())
        m = feed.get_crowd_metrics("BTC")
        assert m["liquidation_age_sec"] >= 500
        assert m["funding_age_sec"] == 0.0


class TestCoinglassExtremeFundingClamped:
    def test_aggregate_extreme_print_is_clamped_not_zeroed(self):
        feed = CoinglassFeed(api_key="k", mock_mode=False)
        payload = {"code": "0", "data": [{
            "exchangeName": "All", "openInterest": 1e9,
            "h24Change": 1.0, "h1OIChangePercent": 0.1,
            "avgFundingRateBySymbol": 0.02,   # squeeze-level, beyond ±1%
        }]}
        oi, fr = _run(feed._fetch_oi_and_funding(_json_sess(payload), {}, "BTC"))
        assert fr is not None
        assert fr.rate == pytest.approx(0.01), (
            f"extreme funding must CLAMP to the bound, got {fr.rate} — "
            f"zeroing it reads as 'no crowding' exactly when it matters")

    def test_non_finite_stays_no_data_not_neutral(self):
        feed = CoinglassFeed(api_key="k", mock_mode=False)
        payload = {"code": "0", "data": [{
            "exchangeName": "All", "openInterest": 1e9,
            "h24Change": 1.0, "h1OIChangePercent": 0.1,
            "avgFundingRateBySymbol": float("nan"),
        }]}
        oi, fr = _run(feed._fetch_oi_and_funding(_json_sess(payload), {}, "BTC"))
        assert fr is None, "non-finite funding must be NO DATA, not 0.0"

    def test_detailed_extreme_rates_clamped_into_average_not_dropped(self):
        feed = CoinglassFeed(api_key="k", mock_mode=False)
        payload = {"code": "0", "data": [{
            "symbol": "BTC",
            "uMarginList": [
                {"rate": 0.05, "status": 1},     # extreme -> clamps to 0.01
                {"rate": 0.0002, "status": 1},
            ],
        }]}
        fr_map = _run(feed._fetch_funding_detailed(_json_sess(payload), {}))
        assert "BTC" in fr_map
        assert fr_map["BTC"].rate == pytest.approx((0.01 + 0.0002) / 2), (
            "the extreme exchange must be CLAMPED into the average, not "
            "silently removed from it")


# ---------------------------------------------------------------------------
# 3+4. CryptoPanic
# ---------------------------------------------------------------------------

def _news(id_, currencies, minutes_ago, neg=0, pos=0):
    return NewsItem(
        id=id_, title=f"t{id_}", source="s", url="u",
        published_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        currencies=currencies,
        votes_positive=pos, votes_negative=neg, votes_important=0,
        votes_liked=0, votes_disliked=0, votes_lol=0, votes_toxic=0,
        votes_saved=0, sentiment_score=0.0,
    )


class TestCryptoPanicPartialFetchCarry:
    def _mk_feed(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        feed = CryptoPanicFeed(api_key="k", mock_mode=False)
        monkeypatch.setattr(cp_mod, "create_session", lambda: _NullAsyncCM())
        return feed

    def test_failed_currency_items_are_carried_and_deduped(
            self, monkeypatch, tmp_path):
        feed = self._mk_feed(monkeypatch, tmp_path)
        eth_old = _news("eth1", ["ETH"], minutes_ago=30, neg=5)
        shared = _news("both1", ["BTC", "ETH"], minutes_ago=20)
        feed._last_data = CryptoPanicData(
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            staleness_sec=0.0, recent_news=[eth_old, shared])

        async def fetch_posts(session, currency):
            if currency == "ETH":
                raise RuntimeError("timeout")
            feed._last_status_code = 200
            if currency == "BTC":
                return [_news("btc1", ["BTC"], minutes_ago=5),
                        _news("both1", ["BTC", "ETH"], minutes_ago=20)]
            return []
        monkeypatch.setattr(feed, "_fetch_posts", fetch_posts)

        result = _run(feed._fetch_real())
        ids = [n.id for n in result.recent_news]
        assert "eth1" in ids, (
            "ETH's fetch failed — its cached item must be carried, else "
            "ETH's panic/velocity re-stamp as 0.0/fresh (the P265 comment's "
            "own named scenario)")
        assert ids.count("both1") == 1, "carry must dedup by id"
        assert "btc1" in ids

    def test_successful_zero_post_currency_is_not_treated_as_failed(
            self, monkeypatch, tmp_path):
        feed = self._mk_feed(monkeypatch, tmp_path)
        sol_old = _news("sol1", ["SOL"], minutes_ago=30)
        feed._last_data = CryptoPanicData(
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            staleness_sec=0.0, recent_news=[sol_old])

        async def fetch_posts(session, currency):
            feed._last_status_code = 200
            if currency == "BTC":
                return [_news("btc1", ["BTC"], minutes_ago=5)]
            return []   # ETH/SOL: HTTP 200, genuinely no posts
        monkeypatch.setattr(feed, "_fetch_posts", fetch_posts)

        result = _run(feed._fetch_real())
        assert "sol1" not in [n.id for n in result.recent_news], (
            "a 200-with-zero-posts is genuine 'no news', NOT a failure — "
            "carrying stale items for it would fabricate persistence of a "
            "dead story")


class TestCryptoPanicUnparseableTimestamp:
    def test_unparseable_published_at_stamps_epoch_never_now(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        feed = CryptoPanicFeed(api_key="k", mock_mode=False)
        payload = {"results": [{
            "id": 1, "title": "undated", "source": {"title": "s"},
            "url": "u", "published_at": "not-a-date",
            "currencies": [{"code": "BTC"}], "votes": {},
        }, {
            "id": 2, "title": "nulldate", "source": {"title": "s"},
            "url": "u", "published_at": None,
            "currencies": [{"code": "BTC"}], "votes": {},
        }]}

        class _Resp(_NullAsyncCM):
            status = 200

            async def json(self):
                return payload

        class _Sess:
            def get(self, *a, **k):
                return _Resp()
        items = _run(feed._fetch_posts(_Sess(), "BTC"))
        assert len(items) == 2
        for it in items:
            assert it.published_at.year == 1970, (
                f"undated item stamped {it.published_at} — absence of a "
                f"timestamp must never read as maximal freshness")


# ---------------------------------------------------------------------------
# 5. CC News — backoff precedes quota reservation
# ---------------------------------------------------------------------------

class TestCCNewsBackoffBeforeQuota:
    def test_no_quota_reserved_while_backed_off(self, monkeypatch, tmp_path):
        import time as _time
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        from data_mgmt.feeds.cryptocompare_news_feed import CCNewsFeed
        from data_mgmt.feeds import _cc_quota as quota_mod
        feed = CCNewsFeed(api_key="k")
        feed._backoff_until = _time.time() + 900

        calls = {"n": 0}

        class _FakeQuota:
            def try_consume(self, n, caller=""):
                calls["n"] += 1
                return True
        monkeypatch.setattr(quota_mod, "get_cc_quota", lambda: _FakeQuota())

        out = _run(feed.fetch_headlines("BTC"))
        assert out == []
        assert calls["n"] == 0, (
            "a cache-miss during an active backoff reserved quota for an "
            "HTTP call that is never made — reserve only when calling")

    def test_expired_backoff_still_reserves_before_calling(
            self, monkeypatch, tmp_path):
        """The reorder must not delete the reservation on the real call path."""
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        from data_mgmt.feeds.cryptocompare_news_feed import CCNewsFeed
        from data_mgmt.feeds import _cc_quota as quota_mod
        feed = CCNewsFeed(api_key="k")
        feed._backoff_until = 0.0

        calls = {"n": 0}

        class _FakeQuota:
            def try_consume(self, n, caller=""):
                calls["n"] += 1
                return False   # refuse, so no HTTP attempt follows
        monkeypatch.setattr(quota_mod, "get_cc_quota", lambda: _FakeQuota())
        out = _run(feed.fetch_headlines("BTC"))
        assert out == []
        assert calls["n"] == 1, "the call path must still reserve first"


# ---------------------------------------------------------------------------
# 6. BinanceFlowFeed per-(asset, cause) latch
# ---------------------------------------------------------------------------

class TestFlowFeedPerAssetLatch:
    def test_each_assets_failure_warns_independently(self, monkeypatch, caplog):
        from data_mgmt.feeds.binance_flow_feed import BinanceFlowFeed
        feed = BinanceFlowFeed()
        monkeypatch.setattr(feed, "_fetch_1h", lambda sym: None)
        with caplog.at_level("WARNING"):
            feed.latest("BTC")
            feed.latest("SOL")
        unavailable = [r for r in caplog.records
                       if "fv2 vector UNAVAILABLE" in r.getMessage()]
        assert len(unavailable) == 2, (
            "BTC's first failure consumed the one warning for SOL — the "
            "single-latch shape P202/P229 retired")

    def test_repeat_failure_same_asset_warns_once(self, monkeypatch, caplog):
        from data_mgmt.feeds.binance_flow_feed import BinanceFlowFeed
        feed = BinanceFlowFeed(cache_ttl_sec=0.0)   # defeat the cache
        monkeypatch.setattr(feed, "_fetch_1h", lambda sym: None)
        with caplog.at_level("WARNING"):
            feed.latest("BTC")
            feed.latest("BTC")
        unavailable = [r for r in caplog.records
                       if "fv2 vector UNAVAILABLE" in r.getMessage()]
        assert len(unavailable) == 1, "same (asset, cause) must latch"


# ---------------------------------------------------------------------------
# 7. fv2 — last COMPLETE 4H bucket
# ---------------------------------------------------------------------------

def _raw_1h(n, seed=13, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    vol = rng.lognormal(10, 1, n)
    taker = vol * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="1h"),
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": vol, "quote_volume": vol * close,
        "count": rng.integers(1000, 50000, n).astype(float),
        "taker_buy_base": taker, "taker_buy_quote": taker * close,
    })


def _training_rows(raw_self, raw_ref, asset="SOL"):
    f = flow_features_4h(raw_self)
    g_self = _agg_4h(raw_self)[["timestamp", "close"]]
    g_ref = _agg_4h(raw_ref)[["timestamp", "close"]]
    x = cross_asset_features(asset, {asset: g_self, "BTC": g_ref})
    return f.merge(x, on="timestamp", how="left")


class TestFv2LastCompleteBucket:
    def test_mid_bucket_call_serves_previous_complete_bucket(self):
        # 4802 hourly rows: the trailing 4H bucket holds only 2 hours.
        raw_self = _raw_1h(4802, seed=13)
        raw_ref = _raw_1h(4802, seed=14)
        merged = _training_rows(raw_self, raw_ref)
        vec = latest_fv2_vector(raw_self, raw_ref, "SOL")
        assert vec is not None
        complete_row = merged.iloc[-2][FV2_COLUMNS]
        partial_row = merged.iloc[-1][FV2_COLUMNS]
        for c in FV2_COLUMNS:
            assert vec[c] == complete_row[c], (
                f"{c}: mid-bucket call must serve the last COMPLETE bucket")
        # the two buckets are 4h apart -> the hour seasonality MUST differ,
        # proving the assertion above is not vacuous
        assert partial_row["fv2_hour_sin"] != complete_row["fv2_hour_sin"]

    def test_aligned_call_still_serves_last_row(self):
        raw_self = _raw_1h(4800, seed=13)   # divisible by 4: complete bucket
        raw_ref = _raw_1h(4800, seed=14)
        merged = _training_rows(raw_self, raw_ref)
        vec = latest_fv2_vector(raw_self, raw_ref, "SOL")
        assert vec is not None
        for c in FV2_COLUMNS:
            assert vec[c] == merged.iloc[-1][FV2_COLUMNS][c]

    def test_bar_count_column_never_becomes_a_feature(self):
        raw = _raw_1h(4800)
        assert "bar_1h_count" not in flow_features_4h(raw).columns
        assert "bar_1h_count" not in FV2_COLUMNS


# ---------------------------------------------------------------------------
# 8+9. market_data_pipeline (source pins — the sites live deep inside the
# async fetch path; each pin was falsification-probed red)
# ---------------------------------------------------------------------------

_PIPELINE_SRC = io.open(
    REPO / "data_mgmt" / "market_data_pipeline.py", encoding="utf-8").read()


class TestOfiFreshOnlyGate:
    def test_ofi_append_is_gated_on_fresh_orderbook(self):
        assert ("if asset in self._ofi_history and not orderbook_stale:"
                in _PIPELINE_SRC), (
            "the OFI history append lost its fresh-orderbook gate — outage "
            "ticks re-enter the rolling sigma and recovery produces a "
            "spurious |z| spike into the |z|>3 SOL toxicity veto")

    def test_there_is_exactly_one_ofi_append_site(self):
        # A second, ungated append elsewhere would defeat the gate silently.
        assert _PIPELINE_SRC.count(
            "self._ofi_history[asset].append(") == 1


class TestTopOfBookProducer:
    def test_pipeline_emits_the_keys_micros_follower_ingest_reads(self):
        # Writer side: genuinely-quoted values only, written with LITERAL
        # keys (a **splat would be a new dynamic write site in the P174
        # orphan scanner — not re-baselineable).
        assert '_ret["bid"] = _bid_quoted' in _PIPELINE_SRC
        assert '_ret["ask"] = _ask_quoted' in _PIPELINE_SRC
        assert "if _bid_quoted > 0 and _ask_quoted > 0:" in _PIPELINE_SRC, (
            "the top-of-book emission must be gated on GENUINE quotes — a "
            "fabricated bid==ask==last claims a zero-spread book (P2)")
        # Reader side (agents/microstructure_agent.py _ingest_market_data):
        # pin the reader/writer PAIR so neither can drift alone.
        agent_src = io.open(
            REPO / "agents" / "microstructure_agent.py", encoding="utf-8"
        ).read()
        assert 'md.get("bid")' in agent_src and 'md.get("ask")' in agent_src, (
            "micro's follower ingest no longer reads market_data['bid'/'ask']"
            " — the P287 producer now feeds nothing; retire or repoint it")


# ---------------------------------------------------------------------------
# 10. kraken_quant cadence
# ---------------------------------------------------------------------------

class TestKrakenQuantCadence:
    def test_information_ratio_annualizes_at_4h_cadence(self):
        from agents.kraken_quant_agent import StrategyState, Regime
        st = StrategyState(1, "t", Regime.SIDEWAYS)
        rng = np.random.default_rng(7)
        pnl = rng.normal(0.001, 0.01, 40)
        for p in pnl:
            st.pnl_history.append(p)
        expected = (np.mean(pnl) * np.sqrt(365 * 6)) / (np.std(pnl) + 1e-10)
        assert st.information_ratio == pytest.approx(expected), (
            "IR must annualize at 6 bars/day (4H cadence); sqrt(365*24) "
            "overstated it 2x into the softmax weights")

    def test_no_hourly_annualization_call_remains(self):
        src = io.open(
            REPO / "agents" / "kraken_quant_agent.py", encoding="utf-8"
        ).read()
        # Scan for the CALL forms, not the words — comments legitimately
        # name the retired constants (the P177 trap).
        assert "np.sqrt(365 * 24)" not in src
        assert "np.sqrt(24)" not in src
        # Newline-anchored: the IR comment also spells "BARS_PER_DAY_4H = 6",
        # so a bare substring check would stay green if the CONSTANT changed
        # while the comment survived (the P177 trap, hit by this very test's
        # first version during its own falsification probe).
        assert "\nBARS_PER_DAY_4H = 6\n" in src

    def test_vol_adjustment_uses_daily_vol_from_4h_returns(self):
        src = io.open(
            REPO / "agents" / "kraken_quant_agent.py", encoding="utf-8"
        ).read()
        assert "np.std(returns) * np.sqrt(BARS_PER_DAY_4H)" in src, (
            "daily vol from 4H log returns is std*sqrt(6); sqrt(24) tripped "
            "the 0.5x reduction at half its declared threshold")

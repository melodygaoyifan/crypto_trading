"""[P154] CryptoPanic rate-limit state must survive a process restart.

`_last_fetch_time` (300s throttle), `_backoff_until` (429 circuit breaker) and
`_last_data` (the 1h cache the LLM-sentiment agent gates its refresh on) were
in-memory only. run_live() runs a full tick immediately on entering its loop
(before any sleep), so every process start burned a fresh 3-request fetch and
forgot an active 429 backoff — a crash-restart loop hammered an API that had
just said "wait 15 minutes". Same bug class as P150/P148/P140-B2.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from data_mgmt.feeds.cryptopanic_feed import (
    CryptoPanicData,
    CryptoPanicFeed,
    NewsItem,
)


def _feed(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    kw.setdefault("api_key", "test-key")
    return CryptoPanicFeed(**kw)


def _news(title="BTC rips", currency="BTC", age_min=5):
    return NewsItem(
        id="1", title=title, source="src", url="https://example.com",
        published_at=datetime.now(timezone.utc) - timedelta(minutes=age_min),
        currencies=[currency],
        votes_positive=3, votes_negative=1, votes_important=2, votes_liked=4,
        votes_disliked=0, votes_lol=0, votes_toxic=1, votes_saved=2,
        sentiment_score=0.5,
    )


def _seed(feed, news=None, fetched_sec_ago=10.0):
    """Put the feed in the state a completed _fetch_real would leave it in."""
    now = datetime.now(timezone.utc)
    data = CryptoPanicData(timestamp=now - timedelta(seconds=fetched_sec_ago), staleness_sec=0.0)
    data.recent_news = news if news is not None else [_news()]
    data.panic_score = {"BTC": 0.25}
    data.news_velocity = {"BTC": 0.5}
    data.sentiment_consensus = {"BTC": 0.5}
    data.narrative_intensity = {"BTC": 0.4}
    data.global_panic = 0.25
    data.global_news_velocity = 0.5
    feed._last_data = data
    feed._last_fetch_time = now - timedelta(seconds=fetched_sec_ago)
    feed._persist_state()
    return data


# ---------------------------------------------------------------------------
# The money bug: a 429 backoff must not be forgotten by a restart
# ---------------------------------------------------------------------------

def test_backoff_survives_restart(tmp_path, monkeypatch):
    a = _feed(tmp_path, monkeypatch)
    a._backoff_until = datetime.now(timezone.utc) + timedelta(seconds=900)
    a._persist_state()

    b = _feed(tmp_path, monkeypatch)
    assert b._backoff_until is not None
    assert b._backoff_until.tzinfo is not None  # tz-aware or fetch() raises on compare
    assert b._backoff_until > datetime.now(timezone.utc)


def test_restored_backoff_blocks_the_startup_fetch(tmp_path, monkeypatch):
    """The end-to-end claim: restart during an active 429 backoff spends 0 requests."""
    a = _feed(tmp_path, monkeypatch)
    _seed(a, fetched_sec_ago=7200.0)  # cache stale enough that the throttle won't save us
    a._backoff_until = datetime.now(timezone.utc) + timedelta(seconds=900)
    a._persist_state()

    b = _feed(tmp_path, monkeypatch)

    async def _boom():
        raise AssertionError("_fetch_real called while a restored backoff was active")

    monkeypatch.setattr(b, "_fetch_real", _boom)
    out = asyncio.run(b.fetch())
    assert out is b._last_data  # served from the restored cache


def test_expired_backoff_does_not_block(tmp_path, monkeypatch):
    a = _feed(tmp_path, monkeypatch)
    a._backoff_until = datetime.now(timezone.utc) - timedelta(seconds=60)
    a._persist_state()

    b = _feed(tmp_path, monkeypatch)
    assert b._backoff_until < datetime.now(timezone.utc)  # restored but harmless


# ---------------------------------------------------------------------------
# Cache reuse: the startup tick should not re-spend 3 requests
# ---------------------------------------------------------------------------

def test_news_cache_survives_restart(tmp_path, monkeypatch):
    a = _feed(tmp_path, monkeypatch)
    _seed(a, news=[_news(title="ETH merge", currency="ETH")])

    b = _feed(tmp_path, monkeypatch)
    assert b._last_data is not None
    assert [n.title for n in b._last_data.recent_news] == ["ETH merge"]
    assert b._last_data.recent_news[0].currencies == ["ETH"]
    assert b._last_data.recent_news[0].votes_toxic == 1  # full round-trip, not to_dict()
    assert b._last_data.recent_news[0].published_at.tzinfo is not None
    assert b._last_data.panic_score == {"BTC": 0.25}
    assert b.get_panic_metrics("BTC")["panic_score"] == pytest.approx(0.25)


def test_throttle_survives_restart(tmp_path, monkeypatch):
    """Restart 10s after a fetch: the 300s throttle still holds, so no request."""
    a = _feed(tmp_path, monkeypatch)
    _seed(a, fetched_sec_ago=10.0)

    b = _feed(tmp_path, monkeypatch)

    async def _boom():
        raise AssertionError("_fetch_real called inside the restored 300s throttle")

    monkeypatch.setattr(b, "_fetch_real", _boom)
    assert asyncio.run(b.fetch()) is b._last_data


def test_throttle_expiry_still_allows_fetch(tmp_path, monkeypatch):
    """Persistence must not wedge the feed shut — past the window it fetches."""
    a = _feed(tmp_path, monkeypatch)
    _seed(a, fetched_sec_ago=600.0)  # > poll_interval_sec (300)

    b = _feed(tmp_path, monkeypatch)
    called = []

    async def _ok():
        called.append(1)
        return b._last_data

    monkeypatch.setattr(b, "_fetch_real", _ok)
    asyncio.run(b.fetch())
    assert called == [1]


def test_stale_cache_dropped_but_backoff_kept(tmp_path, monkeypatch):
    a = _feed(tmp_path, monkeypatch)
    _seed(a, fetched_sec_ago=48 * 3600.0)  # older than _MAX_RESTORED_CACHE_AGE_SEC
    a._backoff_until = datetime.now(timezone.utc) + timedelta(seconds=900)
    a._persist_state()

    b = _feed(tmp_path, monkeypatch)
    assert b._last_data is None            # ancient news is not served
    assert b._backoff_until is not None    # the rate-limit control still holds


# ---------------------------------------------------------------------------
# Degradation: a bad state file must never break startup
# ---------------------------------------------------------------------------

def test_corrupt_state_file_degrades_to_cold_start(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    p = os.path.join(str(tmp_path), "cryptopanic_state.json")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("{not json")

    f = _feed(tmp_path, monkeypatch)  # must not raise
    assert f._last_data is None
    assert f._backoff_until is None


def test_version_mismatch_discards_state(tmp_path, monkeypatch):
    a = _feed(tmp_path, monkeypatch)
    _seed(a)
    with open(a._state_path(), "r", encoding="utf-8") as fh:
        st = json.load(fh)
    st["state_version"] = "cp_cache_v0_old_shape"
    with open(a._state_path(), "w", encoding="utf-8") as fh:
        json.dump(st, fh)

    b = _feed(tmp_path, monkeypatch)
    assert b._last_data is None
    assert b._last_fetch_time is None


def test_missing_state_file_is_a_clean_cold_start(tmp_path, monkeypatch):
    f = _feed(tmp_path, monkeypatch)
    assert f._last_data is None
    assert f._last_fetch_time is None
    assert f._backoff_until is None


def test_mock_mode_never_touches_the_cache_file(tmp_path, monkeypatch):
    """Mock data must not poison the real cache, nor read it."""
    a = _feed(tmp_path, monkeypatch)
    _seed(a, news=[_news(title="real headline")])

    m = _feed(tmp_path, monkeypatch, mock_mode=True)
    assert m._last_data is None  # did not read

    m._last_data = CryptoPanicData(timestamp=datetime.now(timezone.utc), staleness_sec=0.0)
    m._last_data.recent_news = [_news(title="MOCK headline")]
    m._persist_state()  # did not write

    with open(a._state_path(), "r", encoding="utf-8") as fh:
        st = json.load(fh)
    assert st["data"]["recent_news"][0]["title"] == "real headline"


def test_numpy_metrics_round_trip(tmp_path, monkeypatch):
    """_compute_metrics writes numpy scalars (np.clip / np.mean). If json.dump
    choked on them, _persist_state's except would swallow it at DEBUG and the
    whole fix would silently no-op — so exercise the real computed shape."""
    a = _feed(tmp_path, monkeypatch)
    data = CryptoPanicData(timestamp=datetime.now(timezone.utc), staleness_sec=0.0)
    data.recent_news = [_news(), _news(title="BTC dumps", age_min=30)]
    a._compute_metrics(data)  # populates dicts with numpy scalars
    a._last_data = data
    a._last_fetch_time = datetime.now(timezone.utc)
    a._persist_state()

    assert os.path.exists(a._state_path())  # did not silently fail
    b = _feed(tmp_path, monkeypatch)
    assert b._last_data is not None
    assert b._last_data.panic_score["BTC"] == pytest.approx(float(data.panic_score["BTC"]))
    assert b._last_data.global_panic == pytest.approx(float(data.global_panic))


def test_persist_is_atomic_and_leaves_no_tmp(tmp_path, monkeypatch):
    a = _feed(tmp_path, monkeypatch)
    _seed(a)
    assert os.path.exists(a._state_path())
    assert not os.path.exists(a._state_path() + ".tmp")

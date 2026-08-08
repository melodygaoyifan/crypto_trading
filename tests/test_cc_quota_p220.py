"""[P220] Two feeds, one CryptoCompare account, one 100-calls/MONTH quota.

Measured from the account's own `/stats/rate/limit`:

    calls_made  hour 3 · day 43 · month 283 · total 39,223
    max_calls   hour 100 · day 100 · month 100

The binding limit is the MONTH — about 3 calls/day. Demand before this change:

    cc_news      1 call/fetch  (3 before P219), 5 min TTL  -> ~180/month
    cc_onchain   2 calls/fetch (BTC + ETH),    15 min TTL  -> ~360/month
                                                     total  ~540/month

i.e. ~5.4x the allowance. The structural fault was not the TTLs though — it was
that **neither feed knew the other existed**. Both key off the same
`CRYPTOCOMPARE_API_KEY`, each had its own independent backoff, so one could
exhaust the month while the other kept calling, and then both would sit in
separate 15-minute backoffs while the real constraint (the month) was already
blown. A per-feed rate limiter cannot express a per-ACCOUNT budget.

Two changes, and the second is the one that matters:
  1. TTLs raised to fit the cadence inside the budget (news 12h, on-chain 24h).
  2. A SHARED, PERSISTED monthly budget both feeds reserve against BEFORE
     calling — so the constraint is enforced rather than discovered via 429s.

Arithmetic after: news 2 calls/day + on-chain 2 calls/day = ~120/month of
DEMAND against a 90 budget, so the guard still binds near month-end. That is
deliberate and honest: it degrades to cached data with a loud warning instead of
silently hammering an API that has already said no.
"""

import json
import time
from pathlib import Path

import pytest

from data_mgmt.feeds._cc_quota import (
    DEFAULT_MONTHLY_BUDGET,
    CryptoCompareQuota,
    _period_key,
)

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HMATS_CC_MONTHLY_BUDGET", raising=False)
    import data_mgmt.feeds._cc_quota as q
    monkeypatch.setattr(q, "_QUOTA", None)
    yield


class TestBudgetIsShared:

    def test_both_feeds_draw_from_the_same_pool(self):
        """The whole point: cc_news spending must reduce what cc_onchain can
        spend. Two independent limiters cannot express this."""
        q = CryptoCompareQuota(monthly_budget=5)
        assert q.try_consume(3, caller="cc_news")
        assert q.try_consume(2, caller="cc_onchain")
        assert not q.try_consume(1, caller="cc_news"), (
            "second feed still had budget after the first exhausted the account"
        )

    def test_the_singleton_is_process_wide(self):
        from data_mgmt.feeds._cc_quota import get_cc_quota
        assert get_cc_quota() is get_cc_quota()

    def test_spend_is_attributed_per_caller(self):
        """Without this you cannot tell which feed ate the month."""
        q = CryptoCompareQuota(monthly_budget=10)
        q.try_consume(2, caller="cc_news")
        q.try_consume(4, caller="cc_onchain")
        assert q.status()["by_caller"] == {"cc_news": 2, "cc_onchain": 4}


class TestReserveBeforeCalling:

    def test_a_refused_reservation_costs_nothing(self):
        q = CryptoCompareQuota(monthly_budget=2)
        q.try_consume(2, caller="a")
        before = q.status()["used"]
        assert not q.try_consume(1, caller="b")
        assert q.status()["used"] == before

    def test_all_or_nothing(self):
        """A partial on-chain fetch would spend quota for a partial picture."""
        q = CryptoCompareQuota(monthly_budget=3)
        q.try_consume(2, caller="x")
        assert not q.try_consume(2, caller="cc_onchain")
        assert q.status()["remaining"] == 1

    def test_zero_is_free(self):
        q = CryptoCompareQuota(monthly_budget=0)
        assert q.try_consume(0, caller="x")


class TestPersistence:

    def test_budget_survives_a_restart(self):
        """An in-RAM budget re-arms on every restart, and restart-heavy failure
        modes are exactly when it matters most (P154)."""
        q = CryptoCompareQuota(monthly_budget=10)
        q.try_consume(7, caller="cc_news")
        assert CryptoCompareQuota(monthly_budget=10).status()["used"] == 7

    def test_a_new_month_resets(self, monkeypatch):
        q = CryptoCompareQuota(monthly_budget=10)
        q.try_consume(9, caller="cc_news")
        import data_mgmt.feeds._cc_quota as m
        monkeypatch.setattr(m, "_period_key", lambda ts=None: "2099-01")
        fresh = CryptoCompareQuota(monthly_budget=10)
        assert fresh.status()["used"] == 0

    def test_corrupt_state_does_not_break_startup(self, tmp_path):
        (tmp_path / "cc_quota_state.json").write_text("{not json", encoding="utf-8")
        assert CryptoCompareQuota(monthly_budget=10).status()["used"] == 0

    def test_version_mismatch_discards(self, tmp_path):
        (tmp_path / "cc_quota_state.json").write_text(
            json.dumps({"state_version": "old", "period": _period_key(),
                        "used": 99}), encoding="utf-8")
        assert CryptoCompareQuota(monthly_budget=10).status()["used"] == 0


class TestExhaustionIsLoud:

    def test_it_warns_once_naming_the_spend(self, caplog):
        import logging
        q = CryptoCompareQuota(monthly_budget=1)
        q.try_consume(1, caller="cc_news")
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                q.try_consume(1, caller="cc_onchain")
        hits = [r for r in caplog.records if "CC_QUOTA" in r.message]
        assert len(hits) == 1, f"expected one warning, got {len(hits)}"
        assert "cc_news" in hits[0].message, "must say who spent it"
        assert "not a market condition" in hits[0].message, (
            "silent exhaustion is indistinguishable from 'no news' — the "
            "conflation this codebase keeps producing"
        )


class TestBudgetSizing:

    def test_default_is_below_the_real_cap(self):
        """The provider counts calls this process cannot see — ad-hoc probes,
        another process on the same key — and /stats/rate/limit itself costs a
        call. A budget equal to the cap always discovers the difference as a
        429."""
        assert DEFAULT_MONTHLY_BUDGET < 100

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HMATS_CC_MONTHLY_BUDGET", "42")
        assert CryptoCompareQuota().status()["budget"] == 42

    def test_a_bad_override_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("HMATS_CC_MONTHLY_BUDGET", "not-a-number")
        assert CryptoCompareQuota().status()["budget"] == DEFAULT_MONTHLY_BUDGET


class TestFeedsAreWired:

    def _src(self, name):
        return (_REPO / "data_mgmt" / "feeds" / name).read_text(
            encoding="utf-8", errors="replace")

    def test_news_reserves_before_the_request(self):
        s = self._src("cryptocompare_news_feed.py")
        assert 'try_consume(1, caller="cc_news")' in s
        assert s.index("try_consume") < s.index("session.get("), (
            "reserving after the request defeats the purpose"
        )

    def test_onchain_reserves_for_every_asset(self):
        s = self._src("cryptocompare_onchain.py")
        assert 'try_consume(len(SUPPORTED_ASSETS)' in s, (
            "one fetch costs one call per asset — reserving 1 would undercount"
        )
        assert s.index("try_consume") < s.index("session.get(")

    def test_refusal_returns_cached_data_not_an_error(self):
        """Degrading to stale data is correct; raising would turn a budget
        decision into a broken tick."""
        n = self._src("cryptocompare_news_feed.py")
        i = n.index('try_consume(1, caller="cc_news")')
        assert "return cached[1] if cached else []" in n[i:i + 200]
        o = self._src("cryptocompare_onchain.py")
        j = o.index("try_consume(len(SUPPORTED_ASSETS)")
        assert "return self._data" in o[j:j + 220]

    def test_ttls_fit_the_budget(self):
        """The arithmetic this change exists for, pinned so a future TTL edit
        has to re-derive it."""
        from data_mgmt.feeds.cryptocompare_news_feed import (
            MIN_FETCH_INTERVAL as NEWS_TTL)
        from data_mgmt.feeds.cryptocompare_onchain import (
            MIN_FETCH_INTERVAL as OC_TTL, SUPPORTED_ASSETS)
        per_month = lambda ttl, cost: (30 * 24 * 3600 / ttl) * cost  # noqa: E731
        demand = per_month(NEWS_TTL, 1) + per_month(OC_TTL, len(SUPPORTED_ASSETS))
        assert demand < 200, (
            f"~{demand:.0f} calls/month demanded against a 100 cap — before "
            f"P220 this was ~540; if a TTL is lowered, re-do this arithmetic"
        )

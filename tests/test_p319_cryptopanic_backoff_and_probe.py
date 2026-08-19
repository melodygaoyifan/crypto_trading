"""
[P319] "Is the CryptoPanic API arranged properly?" — no, and one of the two
reasons was a fix of mine that was inert on the live system.

WHAT WAS ESTABLISHED (probed from the trading server, 2026-08-19):
  * we are on `growth/v2`; `developer/v2` 404s and `v1` 403s, so the plan tier
    is baked into BASE_URL and cannot be changed by URL alone (P154);
  * the key answers 429 `API monthly quota exceeded` on every path, so the
    exhaustion is real and vendor-confirmed, not inferred;
  * the vendor's plans page AND its API docs are both JS-rendered, so neither
    the plan's true monthly limit nor the semantics of a comma-separated
    `currencies` filter can be read from them. Those stay open, and the plan
    limit is a dashboard question no amount of probing answers.

TWO DEFECTS, one fixed and one MEASURED rather than guessed:

1. THE PERSISTED BACKOFF MADE P299 INERT. P299 replaced the month-long
   monthly-quota lockout with a daily re-probe — but only for a backoff
   computed from then on. The value already on disk was
   `2026-09-01T00:00:00Z`, restored verbatim on every boot, so the live feed
   stayed dark until September exactly as before. P261b's lesson: a migration
   case is a first-boot case with pre-existing state, and "the new code
   computes it correctly" says nothing about the value already persisted.

2. WE ISSUE THREE REQUESTS WHERE ONE MIGHT DO. One per currency. If
   `currencies=BTC,ETH,SOL` is an OR, a single request returns the same
   corpus for a third of the quota. If it is an AND, combining silently
   returns only posts tagged with all three — a coverage collapse that reads
   as a quiet news day (P2). Unknowable today, so it is MEASURED once, when
   healthy, and acted on by nobody.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_mgmt.feeds.cryptopanic_feed import (  # noqa: E402
    QUOTA_REPROBE_SEC, SUPPORTED_CURRENCIES, quota_backoff_until)


class TestARestoredBackoffCannotOutliveTheReProbeInterval:

    def _feed(self, tmp_path, monkeypatch, stored_backoff: str):
        import json
        import data_mgmt.feeds.cryptopanic_feed as mod
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        (tmp_path / "cryptopanic_state.json").write_text(json.dumps({
            "state_version": "cp_cache_v1",
            "last_fetch_time": datetime.now(timezone.utc).isoformat(),
            "backoff_until": stored_backoff,
            "data": None,
        }), encoding="utf-8")
        return mod.CryptoPanicFeed(api_key="k")

    def test_the_real_stored_value_is_capped(self, tmp_path, monkeypatch):
        """THE BUG, with the exact value that was on the live volume."""
        f = self._feed(tmp_path, monkeypatch, "2026-09-01T00:00:00+00:00")
        assert f._backoff_until is not None
        remaining = (f._backoff_until - datetime.now(timezone.utc)).total_seconds()
        assert remaining <= QUOTA_REPROBE_SEC + 60, (
            f"a stored month-long lockout survived restore ({remaining/3600:.1f}h) "
            f"— P299's daily re-probe is unreachable behind it")

    def test_a_short_backoff_is_left_alone(self, tmp_path, monkeypatch):
        """Capping must only ever SHORTEN. A legitimate Retry-After backoff is
        the case the cap must not touch."""
        soon = datetime.now(timezone.utc) + timedelta(seconds=900)
        f = self._feed(tmp_path, monkeypatch, soon.isoformat())
        assert abs((f._backoff_until - soon).total_seconds()) < 2

    def test_absent_backoff_stays_absent(self, tmp_path, monkeypatch):
        """Missing must not become 'capped to now+1d' — absence is not a
        backoff (P2)."""
        import json
        import data_mgmt.feeds.cryptopanic_feed as mod
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        (tmp_path / "cryptopanic_state.json").write_text(json.dumps({
            "state_version": "cp_cache_v1", "backoff_until": None,
            "last_fetch_time": None, "data": None}), encoding="utf-8")
        assert mod.CryptoPanicFeed(api_key="k")._backoff_until is None

    def test_a_freshly_computed_quota_backoff_is_already_daily(self):
        """The P299 half, unchanged — the cap is for stored values, not a
        replacement for computing them correctly."""
        now = datetime(2026, 8, 19, 4, 0, tzinfo=timezone.utc)
        assert (quota_backoff_until(now) - now) <= timedelta(hours=25)


class TestTheCombinedCurrencyProbeMeasuresWithoutActing:

    def test_the_feed_still_fetches_per_currency(self):
        """The 3->1 change is NOT made on a guess. If the filter turns out to
        be an AND, combining would silently collapse coverage into 'posts
        tagged with all three' — indistinguishable from a quiet news day."""
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        assert "for _cp_idx, currency in enumerate(SUPPORTED_CURRENCIES)" in src

    def test_the_one_shot_marker_round_trips_through_disk(
            self, tmp_path, monkeypatch):
        """BEHAVIOURAL, because the source-text version was vacuous: with the
        PERSIST key renamed the string still appeared in the RESTORE, so the
        pin stayed green while the marker no longer survived a restart — and
        the probe would then re-run on every process, which is exactly the
        cost it exists to avoid."""
        import data_mgmt.feeds.cryptopanic_feed as mod
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        f = mod.CryptoPanicFeed(api_key="k")
        f._combined_probe_done = True
        f._persist_state()
        assert mod.CryptoPanicFeed(api_key="k")._combined_probe_done is True, (
            "the marker did not survive a restart — the probe would re-run "
            "every process")

    def test_an_unmeasured_feed_starts_ready_to_probe(self):
        """The mirror: a fresh install must still take the measurement once."""
        import data_mgmt.feeds.cryptopanic_feed as mod
        assert getattr(mod.CryptoPanicFeed(api_key="k"),
                       "_combined_probe_done", False) is False

    def test_the_probe_only_runs_on_a_healthy_cycle(self):
        """Spending a request while the quota is exhausted is the behaviour
        P293b/P299 spent two entries removing."""
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        # Anchor on the GUARD, not on the phrase — the phrase's first
        # occurrence is the persist-marker comment, which proved nothing.
        g = src.index("if (not _cp_failed and not _cp_errors")
        block = src[g:g + 400]
        assert "_combined_probe_done" in block
        assert "_fetch_posts(" in src[g:g + 900], (
            "the guard must actually gate the probe request")

    def test_the_probe_states_what_each_outcome_would_mean(self):
        """A measurement whose reading nobody can interpret is not a
        measurement (P240)."""
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        i = src.index("combined-currency probe:")
        block = src[i:i + 900]
        assert "OR" in block and "AND" in block
        assert "nothing was changed" in block.lower()

    def test_the_probe_cannot_break_the_feed_it_measures(self):
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        i = src.index("combined-currency probe")
        assert "except Exception" in src[i:i + 2200]

    def test_the_currency_list_it_probes_is_the_one_we_fetch(self):
        """If the two ever diverge the probe answers a question about a
        different request than the one it would replace."""
        assert SUPPORTED_CURRENCIES == ["BTC", "ETH", "SOL"]
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        assert '",".join(SUPPORTED_CURRENCIES)' in src


class TestTheOpenQuestionIsRecordedAsOperatorSide:

    def test_the_plan_tier_is_documented_as_url_baked(self):
        """Probed: developer/v2 404s and v1 403s, so the tier cannot be
        changed by editing a URL — downgrading breaks the feed (P154)."""
        src = (REPO / "data_mgmt" / "feeds"
               / "cryptopanic_feed.py").read_text(encoding="utf-8-sig")
        assert "growth/v2" in src
        assert "BILLING" in src, "the tier-in-URL hazard must stay stated"

"""
[P293e/f/g] Sentiment interpretation exam, API fetch efficiency, and the
promotion-clock arithmetic.

  P293e  three F&G readings recorded side by side, judged by one gate
  P293f  fetch throttling + conditional GETs (the "are we fetching
         efficiently" audit)
  P293g  the gate states its sample requirement in DAYS — the finding that
         explains why ~14 candidates never promote
"""

import math
import re
from pathlib import Path

import pytest

# [P311] Guard pins go through assert_guard_live: a plain substring
# assertion survives `if False and <condition>`, which is how P234,
# P251 and P307 each shipped a neutered guard that still read as pinned.
from tests._guard_pins import assert_guard_live  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="ignore")


# =============================================================================
# P293e — the sentiment exam
# =============================================================================

class TestSentimentVariantExam:
    def test_the_two_live_readings_disagree_on_greed(self):
        """The whole reason this exam exists: at greed the momentum reading
        is BULLISH and the contrarian reading is BEARISH. If they ever agree
        everywhere, there is nothing to measure."""
        from defense.sentiment_variant_shadow import (
            momentum_direction, contrarian_direction)
        # F&G = 80 (greed) -> linear z = (80-50)/50*3 = +1.8
        assert momentum_direction(1.8) == 1.0
        assert contrarian_direction(80.0) == -1.0

    def test_contrarian_can_never_be_bullish(self):
        """The audit's structural finding: every branch of the deterministic
        engine's F&G mapping is <= 0, so 'sentiment turned bullish' is not an
        output it can produce."""
        from defense.sentiment_variant_shadow import contrarian_signal
        for fg in range(0, 101):
            assert contrarian_signal(float(fg)) <= 0.0, fg

    def test_extreme_fear_is_neutral_not_bullish(self):
        """`<25 -> 0.0` with the comment "don't chase shorts" — deliberately
        NOT a buy signal, which is what makes it different from -momentum."""
        from defense.sentiment_variant_shadow import contrarian_signal
        assert contrarian_signal(10.0) == 0.0
        assert contrarian_signal(24.9) == 0.0
        assert contrarian_signal(30.0) == -0.2

    def test_contrarian_is_not_merely_negated_momentum(self):
        """A pure negation would be statistically empty (its IC is exactly
        the negative). The recorded claim must be the engine's ASYMMETRIC
        mapping, so at least one F&G value must break the mirror."""
        from defense.sentiment_variant_shadow import (
            momentum_direction, contrarian_direction)
        broken = []
        for fg in range(0, 101):
            z = (fg - 50) / 50.0 * 3.0
            if contrarian_direction(float(fg)) != -momentum_direction(z):
                broken.append(fg)
        assert broken, "contrarian is a pure negation — the exam adds nothing"
        assert any(f < 25 for f in broken), (
            "the extreme-fear asymmetry is what makes it a distinct claim"
        )

    def test_thresholds_match_the_deterministic_engine(self):
        """Copied thresholds must stay equal to the engine's, or the exam
        judges a strategy nobody runs."""
        eng = _src(REPO / "signals" / "deterministic_sentiment.py")
        m = re.search(r"# 3\. Fear & Greed.*?signals\.append\(\(\"fear_greed\"",
                      eng, re.S)
        assert m, "F&G branch not found in the engine"
        blk = m.group(0)
        for frag in ("> 75", "-0.6", "> 55", "-0.3", "< 25", "< 45", "-0.2"):
            assert frag in blk, f"engine no longer contains {frag!r}"

    def test_flat_rows_carry_zero_confidence(self):
        """The scorer multiplies direction x confidence (P236/P224) — a flat
        row must contribute zero, never a saturated claim."""
        import tempfile
        from defense.sentiment_variant_shadow import SentimentVariantShadow
        with tempfile.TemporaryDirectory() as d:
            s = SentimentVariantShadow(data_dir=d)
            rows = s.record_tick("BTC", 50.0, 0.0, 0.0)
            assert rows
            for r in rows:
                assert r["confidence"] == abs(r["direction"])

    def test_absent_input_is_skipped_not_written_flat(self):
        """An absent historical z is not the claim 'historical says flat' —
        conflating them lets a starved variant look like a confident neutral
        (P2)."""
        import tempfile
        from defense.sentiment_variant_shadow import SentimentVariantShadow
        with tempfile.TemporaryDirectory() as d:
            s = SentimentVariantShadow(data_dir=d)
            names = [r["strategy"] for r in
                     s.record_tick("BTC", 31.0, -1.14, None)]
        assert "sent_momentum_hist" not in names
        assert "sent_momentum_linear" in names and "sent_contrarian" in names

    def test_prefix_registered_at_both_scorer_sites(self):
        s = _src(REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py")
        assert re.search(
            r'prefixes: Tuple\[str, \.\.\.\] = \([^)]*"sentvariant"', s)
        cli = next(l for l in s.splitlines() if "microstructure,cascade" in l)
        assert "sentvariant" in cli
        # and ma_filter must STILL be last (the P236 end-anchor, P248 lesson)
        assert re.search(r'default="[^"]*,ma_filter"', s)

    def test_wired_observation_only(self):
        """Must not touch any signal — it is an exam, not a strategy."""
        main = _src(REPO / "main.py")
        m = re.search(r"get_sentiment_variant_shadow\(\)\.record_tick\(.*?\)",
                      main, re.S)
        assert m, "not wired"
        blk = m.group(0)
        assert "agent_signals" not in blk and "market_data[" not in blk


# =============================================================================
# P293f — fetch efficiency
# =============================================================================

class TestFetchEfficiency:
    @pytest.mark.parametrize("mod,cls_hint", [
        ("data_mgmt.feeds.coinglass_feed", "Coinglass"),
        ("data_mgmt.feeds.sentiment_feed", "Sentiment"),
        ("data_mgmt.feeds.fred_feed", "FRED"),
        ("data_mgmt.feeds.lunarcrush_feed", "LunarCrush"),
        ("data_mgmt.feeds.deribit_feed", "Deribit"),
        ("data_mgmt.feeds.exchange_netflow_feed", "ExchangeNetflow"),
    ])
    def test_feed_has_a_throttled_entry_point(self, mod, cls_hint):
        """The tick loop calls these once per ASSET. Without a self-throttle
        that is 3x the requests for data the first call already contained."""
        import importlib
        m = importlib.import_module(mod)
        # Select by CAPABILITY, not by name: the name heuristic picked
        # dataclasses like FREDObservation. The feed is the class that owns
        # a `fetch`.
        feeds = [v for v in vars(m).values()
                 if isinstance(v, type) and callable(getattr(v, "fetch", None))]
        assert feeds, f"no feed class with fetch() in {mod}"
        for klass in feeds:
            assert hasattr(klass, "fetch_if_stale"), (
                f"{klass.__name__} has fetch() but no throttled entry point — "
                f"the tick loop calls it once per ASSET"
            )

    @pytest.mark.parametrize("call", [
        "self.coinglass_feed.fetch_if_stale()",
        "self.sentiment_feed.fetch_if_stale()",
        "self.fred_feed.fetch_if_stale()",
        "self.lunarcrush_feed.fetch_if_stale()",
    ])
    def test_tick_uses_the_throttled_form(self, call):
        assert call in _src(REPO / "main.py"), f"{call} not wired"

    def test_unthrottled_coinglass_fetch_is_gone_from_the_tick(self):
        """The measured 3x waste: fetch() had no throttle and _fetch_real
        already loops all three symbols internally."""
        main = _src(REPO / "main.py")
        assert "_feed_tasks.append(self.coinglass_feed.fetch())" not in main

    def test_shared_cache_age_helper_handles_both_stamp_types(self):
        """Feeds in this package stamp with datetime AND float epoch."""
        import time as _t
        from datetime import datetime, timedelta, timezone
        from data_mgmt.feeds._http import cache_age_seconds
        assert cache_age_seconds(None) is None
        assert cache_age_seconds(0) is None, "epoch 0 means never fetched"
        aware = datetime.now(timezone.utc) - timedelta(seconds=30)
        assert 25 <= cache_age_seconds(aware) <= 60
        naive = datetime.now() - timedelta(seconds=30)   # [P40/P97]
        assert cache_age_seconds(naive) is not None
        assert 25 <= cache_age_seconds(_t.time() - 30) <= 60

    def test_rss_sends_conditional_validators(self):
        """RSS is the ONLY dependency exposing validators (probed): a 304
        costs no bandwidth. The JSON APIs return no ETag/Last-Modified, so
        for those a client-side TTL is the only available lever."""
        s = _src(REPO / "data_mgmt" / "feeds" / "rss_news_feed.py")
        assert "If-None-Match" in s and "If-Modified-Since" in s
        assert "resp.status == 304" in s

    def test_304_reuses_items_rather_than_dropping_the_source(self):
        """A 304 means UNCHANGED. Returning [] would read as 'this source
        went silent' and shrink the corpus."""
        s = _src(REPO / "data_mgmt" / "feeds" / "rss_news_feed.py")
        m = re.search(r"if resp\.status == 304:.*?return \[.*?\]", s, re.S)
        assert m, "304 branch not found"
        assert "i.source == name" in m.group(0)

    def test_every_session_identifies_itself(self):
        """Cloudflare 1010 blocks unidentified clients (P293b)."""
        from data_mgmt.feeds._http import DEFAULT_USER_AGENT
        assert "hmats" in DEFAULT_USER_AGENT.lower()


# =============================================================================
# P293g — the promotion clock arithmetic
# =============================================================================

class TestPromotionClockIsHonest:
    def test_gate_states_the_requirement_in_days(self):
        """The sample figure was already computed and nobody converted it —
        which is how ~14 candidates came to sit on 30-day clocks against
        requirements that are ~a year at 16h."""
        from analytics.shadow_ic.compute_shadow_ic import assess_promotion
        a = assess_promotion(
            ic_per_h={4: 0.09}, n_per_h={4: 180}, sharpe=1.0, window_days=30,
            fwd_vol_bps_per_h={4: 107.0})
        joined = " ".join(a.blockers)
        assert "d of 4H bars vs" in joined, (
            "the gate must state days required vs days held"
        )
        assert "d held" in joined

    def test_the_statistical_bar_dominates_at_16h(self):
        """The finding itself, as arithmetic: at 16h a 30-day window can only
        certify IC >= ~0.30, while the ECONOMIC bar asks ~0.13 — so
        significance, not economics, is what blocks, and no realistic signal
        can clear it in 30 days."""
        BARS_PER_DAY, REQ_T, H = 6, 2.0, 4
        n_eff_30d = (30 * BARS_PER_DAY) / H
        ic_needed_30d = REQ_T / math.sqrt(n_eff_30d - 1)
        assert ic_needed_30d > 0.29, ic_needed_30d
        # and an economically-adequate IC needs roughly a year
        ic_econ = 0.134
        days = ((REQ_T / ic_econ) ** 2 + 1) * H / BARS_PER_DAY
        assert days > 140, days

    def test_four_hour_horizon_is_reachable(self):
        """Not everything is unreachable — at 4h the ECONOMIC bar binds and
        ~a month suffices. The defect is horizon-specific, which is exactly
        why the gate should say so per horizon."""
        BARS_PER_DAY, REQ_T, H = 6, 2.0, 1
        ic_econ_4h = 0.26
        days = ((REQ_T / ic_econ_4h) ** 2 + 1) * H / BARS_PER_DAY
        assert days < 20, days

    def test_bars_per_day_constant_matches_the_cadence(self):
        from analytics.shadow_ic.compute_shadow_ic import BARS_PER_DAY_4H
        assert BARS_PER_DAY_4H == 6


# =============================================================================
# P293k — is GATE-CLOSED a code artifact? (asked directly; answer: no)
# =============================================================================

class TestTripwireVerdictIsNotAnArtifact:
    """Two candidate artifacts were checked and one real defect found."""

    def test_thresholds_match_the_current_friction_regime(self):
        """DEFECT FOUND: the snapshot was stamped 2026-08-08 and overtaken
        twice by CODE (P289 spreads, P291b venue-true hold), leaving the
        tripwire comparing against thresholds 1.5-1.9x too high.

        The deeper defect is that the staleness guard is TIME-based (30d)
        while these move when friction CODE changes — so it read "verified
        10d ago" while the values were already wrong.
        """
        from analytics.calibration.slope_calibrator import DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["BTC"] == pytest.approx(19.1)
        assert DEFAULT_THRESHOLDS["ETH"] == pytest.approx(26.7)
        assert DEFAULT_THRESHOLDS["SOL"] == pytest.approx(29.0)

    def test_the_threshold_error_did_not_change_the_verdict(self):
        """Immaterial to the 2026-08-17 reading — published max alpha was
        0.0/1.95/1.25, closed against BOTH the stale and corrected bars.
        Recorded so nobody re-opens the question thinking it was decisive."""
        stale = {"BTC": 28.93, "ETH": 42.77, "SOL": 55.34}
        from analytics.calibration.slope_calibrator import DEFAULT_THRESHOLDS
        measured = {"BTC": 0.0, "ETH": 1.95, "SOL": 1.25}
        for a, alpha in measured.items():
            assert (alpha >= stale[a]) == (alpha >= DEFAULT_THRESHOLDS[a])
            assert alpha < DEFAULT_THRESHOLDS[a], "still GATE-CLOSED"

    def test_both_alignments_are_reported(self):
        """The calibrator and agent_ic_review share `closes[i+h]/closes[i]`,
        so they were never INDEPENDENT confirmation of the negative result —
        a shared convention means a shared bias. Reporting the entry-aligned
        series alongside makes the difference visible instead of assumed."""
        src = _src(REPO / "analytics" / "calibration" / "slope_calibrator.py")
        assert "pairs_entry" in src
        assert "entry_aligned_slope_ols" in src
        assert "entry_aligned_t" in src

    def test_the_verdict_still_keys_on_the_established_basis(self):
        """Switching the basis would invalidate every prior weekly report the
        tripwire has counted. Both are reported; only one decides."""
        src = _src(REPO / "analytics" / "calibration" / "slope_calibrator.py")
        m = re.search(r'max_alpha = r\["slope_published"\]', src)
        assert m, "verdict must read the established series"
        assert 'r_entry["slope_published"]' not in src

    def test_shrinkage_floor_only_bites_on_negative_slopes(self):
        """Confirms GATE-CLOSED is driven by the MEASUREMENT, not by the
        shrinkage: a strong positive slope must survive to the verdict."""
        from analytics.calibration.slope_calibrator import shrunk_slope
        strong = [(1.0 if k % 2 else -1.0,
                   (60.0 if k % 2 else -60.0)) for k in range(400)]
        r = shrunk_slope(strong, overlap=1)
        assert r["slope_ols"] > 50, r
        assert r["slope_published"] > 0, (
            "a real positive slope must reach the verdict — if this fails, "
            "the gate would be closed by construction"
        )


# =============================================================================
# P293h — a deposit must not be reported as profit
# =============================================================================

class TestExternalFlowExcludedFromPnl:
    """MEASURED on the live ledger 2026-08-17: after a $7,074 deposit the
    sleeve reported +$6,850 (+171%) while actual trading PnL over the same
    65 days was -$225 (-6.3%). A 177-percentage-point misreport, in the
    flattering direction, on the one number that answers "is this making
    money"."""

    def _sleeve(self, notional=0.0, prev=None, prev_ts=None):
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = CoinbaseSleeve.__new__(CoinbaseSleeve)
        s._last_positions = {}
        s._last_equity_for_flow = prev
        s._last_equity_for_flow_ts = prev_ts
        s._external_flow_usd = 0.0
        s._position_notional_usd = lambda: notional
        return s

    def test_the_live_deposit_is_detected(self):
        s = self._sleeve(prev=3540.0)
        assert s._detect_external_flow(10614.0) == pytest.approx(7074.0)

    def test_ordinary_pnl_is_not_reclassified(self):
        """The failure direction must be 'keep counting it as PnL' — a
        silent adjustment could hide a real loss."""
        s = self._sleeve(prev=3561.0)
        for eq in (3540.0, 3480.0, 3600.0, 3400.0):
            assert s._detect_external_flow(eq) == 0.0, eq

    def test_large_book_raises_the_threshold(self):
        """A bigger position can legitimately move more, so the limit scales
        with notional rather than being a fixed dollar figure."""
        s = self._sleeve(notional=2000.0, prev=5000.0)
        assert s._detect_external_flow(5400.0) == 0.0      # 400 < 1000 limit
        assert s._detect_external_flow(12074.0) != 0.0     # unmistakable

    def test_first_reading_is_never_a_flow(self):
        s = self._sleeve(prev=None)
        assert s._detect_external_flow(10000.0) == 0.0

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_unusable_equity_is_never_a_flow(self, bad):
        s = self._sleeve(prev=3500.0)
        assert s._detect_external_flow(bad) == 0.0

    def test_withdrawal_direction_also_handled(self):
        """P274 documented withdrawals reading as drawdown; the same
        detector must catch them so PnL is not understated either."""
        s = self._sleeve(prev=10000.0)
        assert s._detect_external_flow(3000.0) == pytest.approx(-7000.0)

    def test_pnl_subtracts_flows_and_rebases_the_percentage(self):
        src = _src(REPO / "exchange" / "coinbase_sleeve.py")
        assert "eq - start - _flows" in src, "flows must be excluded from PnL"
        assert "(start + _flows)" in src, (
            "percentage must be against invested capital, else a deposit "
            "makes the ratio incomparable over time"
        )
        assert '"external_flow_usd": round(_flows, 2)' in src, (
            "the ledger must record the flow so the adjustment is auditable"
        )

    def test_flow_accumulator_survives_restart(self):
        """Without persistence a restart forgets the transfer and re-books
        it as profit (the P154 class)."""
        src = _src(REPO / "exchange" / "coinbase_sleeve.py")
        assert '"external_flow_usd": getattr(' in src, "not persisted"
        assert 'st.get("external_flow_usd")' in src, "not restored"


class TestExternalFlowSurvivesDowntime:
    """[P294] The reference equity must persist, or the ONE sequence that
    actually happens is the one the detector cannot see.

    The wire is a manual operator step (P274) and deploys are frequent, so
    "deposit, then restart" is the normal order. With `_last_equity_for_flow`
    in RAM only, a fresh process always took the `prev is None` early return
    and the deposit was booked as profit forever — which is exactly what
    happened to the 2026-08-16 transfer this whole mechanism was written for.
    """

    def _sleeve_with_state(self, tmp_path, **state):
        """`_state_path` is a METHOD returning a str, not an attribute — the
        fixture has to match the real shape or it tests a different object."""
        import json
        from pathlib import Path
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = CoinbaseSleeve.__new__(CoinbaseSleeve)
        path = Path(tmp_path) / "sleeve_state.json"
        s._state_path = lambda _p=str(path): _p
        s._BASE_VERSION = "portfolio_total_v3"
        s._sleeve_start_equity = None
        s._external_flow_usd = 0.0
        s._last_equity_for_flow = None
        s._last_equity_for_flow_ts = None
        s._halted = False
        s._halt_reason = ""
        if state:
            path.write_text(json.dumps(state), encoding="utf-8")
        return s

    @staticmethod
    def _read_state(s):
        import json
        from pathlib import Path
        return json.loads(Path(s._state_path()).read_text(encoding="utf-8"))

    def test_reference_round_trips_through_the_state_file(self, tmp_path):
        import json
        s = self._sleeve_with_state(tmp_path)
        s._sleeve_start_equity = 4000.0
        s._last_equity_for_flow = 3540.0
        s._last_equity_for_flow_ts = 1_787_000_000.0
        s._external_flow_usd = 0.0
        s._persist_state()
        written = self._read_state(s)
        assert written["last_equity_for_flow"] == pytest.approx(3540.0)
        assert written["last_equity_for_flow_ts"] == pytest.approx(1_787_000_000.0)

        fresh = self._sleeve_with_state(tmp_path, **written)
        fresh._restore_state()
        assert fresh._last_equity_for_flow == pytest.approx(3540.0), (
            "a fresh process must inherit the reference, or a transfer during "
            "downtime is invisible"
        )

    def test_a_deposit_during_downtime_is_detected_on_restart(self, tmp_path):
        """The end-to-end case: persist, die, deposit lands, restart."""
        s = self._sleeve_with_state(tmp_path)
        s._sleeve_start_equity = 4000.0
        s._last_equity_for_flow = 3540.0
        s._last_equity_for_flow_ts = 1_787_000_000.0
        s._persist_state()

        fresh = self._sleeve_with_state(tmp_path, **self._read_state(s))
        fresh._restore_state()
        fresh._last_positions = {}
        fresh._position_notional_usd = lambda: 0.0
        # 4h later the engine comes back up and the venue reports the deposit
        flow = fresh._detect_external_flow(
            10614.0, now=1_787_000_000.0 + 4 * 3600)
        assert flow == pytest.approx(7074.0)

    def test_a_missing_reference_re_anchors_instead_of_inventing_a_flow(
            self, tmp_path):
        """An unreadable/absent stamp means 'no reference'. It must NEVER be
        defaulted to a number — a wrong reference invents a transfer."""
        s = self._sleeve_with_state(
            tmp_path, sleeve_start_equity=4000.0, external_flow_usd=0.0)
        s._restore_state()
        assert s._last_equity_for_flow is None
        s._last_positions = {}
        s._position_notional_usd = lambda: 0.0
        assert s._detect_external_flow(10614.0) == 0.0

    def test_a_long_outage_widens_the_bound_rather_than_narrowing_it(self):
        """A multi-day gap can accumulate more honest mark-to-market than one
        4H tick. Widening is the CONSERVATIVE direction: the failure it
        guards against is reclassifying real PnL as a transfer, which would
        hide a loss."""
        from exchange.coinbase_sleeve import CoinbaseSleeve
        s = CoinbaseSleeve.__new__(CoinbaseSleeve)
        s._last_positions = {}
        s._position_notional_usd = lambda: 4000.0
        s._last_equity_for_flow = 10000.0
        base = 1_787_000_000.0
        # one tick: limit = 4000 * 0.5 * 1 = 2000 -> a 2500 step is a flow
        s._last_equity_for_flow_ts = base
        assert s._detect_external_flow(12500.0, now=base + 4 * 3600) != 0.0
        # two days later: limit = 4000 * 0.5 * 12 = 24000 -> same step is PnL
        assert s._detect_external_flow(12500.0, now=base + 48 * 3600) == 0.0

    def test_the_widening_is_capped(self):
        """Uncapped, a long enough outage would make the bound so wide that
        nothing is ever detected again."""
        from exchange.coinbase_sleeve import CoinbaseSleeve
        assert CoinbaseSleeve.FLOW_DETECT_MAX_PERIODS <= 24.0, (
            "a cap above ~4 days makes the detector unable to fire after any "
            "extended outage"
        )

    def test_the_reference_is_refreshed_on_ordinary_ticks(self):
        """Persisting only when a flow fires leaves a stale stamp, and the
        elapsed-gap tolerance is computed FROM that stamp."""
        src = _src(REPO / "exchange" / "coinbase_sleeve.py")
        assert_guard_live(src, "if not _flow:",
                          "the reference must be persisted on ordinary ticks too")
        _unused_msg = (
            "_persist_state()" in src and
            "the reference must be persisted on ordinary ticks too"
        )

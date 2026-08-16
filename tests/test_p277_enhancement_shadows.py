"""P277 — the "full enhancement" shadow families, pinned.

Five new forward-ledgered candidates (stablecoinflow proxy, oidiv twins,
calbasis, xsmom, eventfilter) + the venue-aware funding activation. All
observation-only; every P166 exam decides, nothing here trades.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests._source_scan import read_source
from defense.enhancement_shadows import (
    EnhancementShadows, oidiv_signals, in_event_window,
    FOMC_DECISION_DAYS_2026, XS_ASSETS)

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "main.py"


def _enh(tmp_path):
    s = object.__new__(EnhancementShadows)
    s._dir = tmp_path
    s._cg_key = ""
    s._stable_cache = None
    s._basis_hist = {}
    s._warned = {}
    return s


class TestOidivTwins:
    def test_confirm_follows_price_when_oi_rises(self):
        c, f = oidiv_signals(+2.0, +3.0)
        assert (c, f) == (1.0, 0.0)
        c, f = oidiv_signals(-2.0, +3.0)
        assert (c, f) == (-1.0, 0.0)

    def test_fade_opposes_price_when_oi_falls(self):
        c, f = oidiv_signals(+2.0, -3.0)
        assert (c, f) == (0.0, -1.0)
        c, f = oidiv_signals(-2.0, -3.0)
        assert (c, f) == (0.0, 1.0)

    def test_disjoint_conditions_never_both_fire(self):
        for p in (-2.0, 0.0, 2.0):
            for oi in (-5.0, -0.5, 0.0, 0.5, 5.0):
                c, f = oidiv_signals(p, oi)
                assert not (c != 0.0 and f != 0.0), (
                    "both twins fired on one input — they must be disjoint "
                    "hypotheses (the P219 twin pattern), not overlapping")

    def test_missing_inputs_are_flat_never_fabricated(self):
        assert oidiv_signals(None, 5.0) == (0.0, 0.0)
        assert oidiv_signals(2.0, None) == (0.0, 0.0)


class TestEventWindow:
    def test_fomc_decision_afternoon_blocks(self):
        t = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)
        assert in_event_window(t) == "fomc"
        # morning of the same day is OUTSIDE the window
        t2 = datetime(2026, 9, 16, 8, 0, tzinfo=timezone.utc)
        assert in_event_window(t2) is None

    def test_cpi_release_window(self):
        t = datetime(2026, 10, 13, 12, 30, tzinfo=timezone.utc)
        assert in_event_window(t) == "cpi"

    def test_sunday_thin_window(self):
        # Sunday 2026-08-16 is a Sunday
        t = datetime(2026, 8, 16, 23, 0, tzinfo=timezone.utc)
        assert t.weekday() == 6
        assert in_event_window(t) == "sunday_thin"
        assert in_event_window(
            datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)) is None

    def test_ordinary_hour_is_open(self):
        t = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)  # Thursday
        assert in_event_window(t) is None

    def test_calendar_carries_the_verify_warning(self):
        src = read_source(REPO / "defense" / "enhancement_shadows.py")
        assert "OPERATOR-VERIFY" in src, (
            "the static FOMC/CPI dates lost their verify-before-enforce "
            "warning — a wrong date is harmless in shadow and a mistimed "
            "entry block under enforcement")


class TestXsmom:
    def test_top_and_bottom_k_at_full_universe(self):
        closes = {}
        for i, a in enumerate(XS_ASSETS):
            # asset i has 30d return of i percent
            base = 100.0
            closes[a] = [base] * 200 + [base * (1 + i / 100.0)]
        s = _enh.__wrapped__ if hasattr(_enh, "__wrapped__") else None
        e = EnhancementShadows.__new__(EnhancementShadows)
        out = e.xsmom_directions(closes)
        longs = [a for a, d in out.items() if d == 1.0]
        shorts = [a for a, d in out.items() if d == -1.0]
        assert set(longs) == set(XS_ASSETS[-2:])
        assert set(shorts) == set(XS_ASSETS[:2])

    def test_thin_universe_is_all_flat(self):
        # 3 correlated majors are NOT a cross-section (the taxonomy audit's
        # explicit finding) — below 6 rankable assets everything is flat
        e = EnhancementShadows.__new__(EnhancementShadows)
        closes = {a: [100.0] * 201 for a in ("BTC", "ETH", "SOL")}
        out = e.xsmom_directions(closes)
        assert all(d == 0.0 for d in out.values())


class TestStablecoinFlow:
    def test_proxy_basis_is_labeled_on_every_row(self, tmp_path):
        # the ledger judges a PROXY (mcap delta), not the cited paper's
        # exchange-inflow variable — every row must say so
        s = _enh(tmp_path)
        now = time.time()
        midnight = (int(now) // 86400) * 86400
        # varying deltas (constant deltas correctly read zero_variance)
        rows = [((midnight - (40 - i) * 86400) * 1000.0,
                 {"USDT": 100000.0 + i * 10 + (i % 5) * 37})
                for i in range(40)]
        s._stable_cache = (now, rows)
        d, z, reason = s.stablecoin_direction(now_ts=now)
        assert reason in ("mint_inflow", "burn_outflow", "neutral")
        summary = s.tick()
        rec = json.loads((tmp_path / "stablecoinflow_BTC.jsonl")
                         .read_text().splitlines()[-1])
        assert rec["basis"] == "mcap_delta_proxy"
        assert rec["confidence"] == abs(rec["direction"])

    def test_stale_series_is_flat(self, tmp_path):
        s = _enh(tmp_path)
        now = time.time()
        old = [((now - (50 - i) * 86400) * 1000.0, {"USDT": 1e5})
               for i in range(40)]
        s._stable_cache = (now, old)
        d, z, reason = s.stablecoin_direction(now_ts=now + 10 * 86400)
        assert d == 0.0 and reason.startswith("stale")


class TestCalbasis:
    def test_warmup_and_missing_quotes_are_flat(self, tmp_path):
        s = _enh(tmp_path)
        d, sl, r = s.calbasis_direction("BTC", None, None, 63045.0, 1500.0)
        assert d == 0.0 and r == "no_quotes"
        d, sl, r = s.calbasis_direction("BTC", 63360.0, 40.0, 63045.0, 1587.0)
        assert d == 0.0 and r == "warmup"

    def test_signal_is_change_based_not_level_based(self, tmp_path):
        # crypto sits in near-permanent mild contango: a level-based signal
        # would be a constant short. Feed a CONSTANT slope history — the
        # signal must stay flat (zero variance), not emit the level's sign.
        s = _enh(tmp_path)
        # constant history -> zero variance -> flat
        for _ in range(25):
            d, sl, r = s.calbasis_direction("BTC", 63360.0, 40.0,
                                            63045.0, 1587.0)
        assert d == 0.0 and r in ("zero_variance", "neutral")
        # the discriminating case (the first probe of this test stayed
        # GREEN because constant history never reached the z-branch —
        # P238: a probe that doesn't fail indicts the test's inputs):
        # a VARYING history whose LEVEL is always positive contango but
        # whose latest reading sits within the noise band. Level-based
        # logic emits a permanent short here; change-based stays flat.
        s2 = _enh(tmp_path)
        import math
        for i in range(30):
            near = 63360.0 * (1 + 0.0004 * math.sin(i))
            s2.calbasis_direction("BTC", near, 40.0, 63045.0, 1587.0)
        # final reading at the oscillation's MID-BAND: within the noise of
        # its own history (|z| small), but its LEVEL is still contango —
        # only a level-based implementation emits a direction here
        d, sl, r = s2.calbasis_direction("BTC", 63360.0, 40.0,
                                         63045.0, 1587.0)
        assert sl is not None and sl != 0, "fixture degenerate"
        assert d == 0.0 and r == "neutral", (
            f"steady mild contango produced dir={d} ({r}) — the signal "
            f"has become level-based (permanent-short bias trap)")


class TestWiringAndScorer:
    def test_prefixes_registered_at_both_scorer_sites(self):
        src = read_source(REPO / "analytics" / "shadow_ic" /
                          "compute_shadow_ic.py")
        for p in ("stablecoinflow", "oidiv", "calbasis", "xsmom",
                  "eventfilter"):
            assert src.count(p) >= 2, (
                f"{p} missing from one of the two scorer default sites "
                f"(P192 two-site rule)")

    def test_main_inits_and_ticks(self):
        src = read_source(MAIN)
        assert "EnhancementShadows" in src
        assert "_enhancement_shadows.tick(" in src, (
            "constructed but never ticked — ledgers with no writer "
            "(P199 class)")
        assert "_cde_quote_map" in src

    def test_venue_aware_funding_is_on_in_live_profile(self):
        # [P277] the funding-hold-cost activation: the gate's
        # FrictionComponents.update_funding_rate charges whatever rate
        # market_data carries — OFF meant the sleeve's hold cost was
        # priced on KRAKEN's funding
        live = json.loads((REPO / "configs" / "live_high_risk.json")
                          .read_text(encoding="utf-8"))
        assert live.get("coinbase_venue_aware_funding") is True, (
            "coinbase_venue_aware_funding flipped off — if deliberate, "
            "update this pin + the note; the sleeve's gate hold-cost "
            "reverts to Kraken's funding rate")

    def test_no_enforcement_path_exists_yet(self):
        # Iron Law 7: these are ledgers. The eventfilter's future enforce
        # flag must not exist until its own P-entry ships it.
        src = read_source(MAIN)
        assert "event_window_filter_enforce" not in src

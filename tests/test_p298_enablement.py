"""
================================================================================
HMATS [P298] - the enablement batch, and the units bug that would have flattened
================================================================================

Operator: "make a plan on enabling all items, i don't want to wait, implement
right away." Seven flags flipped in one deploy; four deliberately NOT flipped
because measurement contradicts them; one blocked by a units bug found while
checking what it actually arms.
================================================================================
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIVE = REPO / "configs" / "live_high_risk.json"
MAIN = REPO / "main.py"


def _live():
    return json.loads(LIVE.read_text(encoding="utf-8"))


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestGroupAEnabled:
    """Each of these is real data replacing a fabricated constant on a channel
    that cannot reach an order, or a strictly tightening control."""

    @pytest.mark.parametrize("flag", [
        "options_use_deribit",
        "exchange_netflow_to_flow_agent",
        # [P356] the two entry filters moved OFF this roster and onto the
        # disarmed one below — by operator instruction, on the no-trade
        # decomposition. They are still pinned, just to the other value.
        "macro_gci_live",
        # [P417] fusion_conviction_to_sleeve left this roster: turned OFF by
        # the conviction_channel_lab historical verdict; its decided-value
        # pin (False) lives in test_p293d_whale_options.py.
    ])
    def test_flag_is_on_with_its_evidence_note(self, flag):
        d = _live()
        assert d.get(flag) is True, f"{flag} is not enabled"
        notes = [k for k in d if k.startswith("_") and flag.split("_")[0] in k]
        assert notes, f"{flag} enabled with no annotation (P141: a live flip is a decision)"

    @pytest.mark.parametrize("flag", [
        "coinbase_whale_filter_enforce",
        "coinbase_ma_filter_enforce",
    ])
    def test_the_entry_filters_are_disarmed_with_their_reason(self, flag):
        """[P356] Armed by P298, disarmed by explicit operator instruction
        after the no-trade decomposition put 23 of ETH's 31 actionable ticks
        on them. Pinned at the DECIDED value so a silent re-arm fails too."""
        d = _live()
        assert d.get(flag) is False, (
            f"{flag} is not at its decided value False — re-arming is a "
            f"live-money change and needs its own record"
        )
        assert any(k.startswith("_p356") for k in d), (
            "the disarm must carry its reason in the profile (P141)"
        )

    def test_sentiment_uses_the_historical_distribution(self):
        assert _live().get("sentiment_zscore_mode") == "historical"

    def test_regimebook_holds_the_seat(self):
        assert _live().get("regimebook_mode") == "enforce"


class TestGroupDStaysOff:
    """Not flipped, because measurement says otherwise. Pinned so a later
    'enable everything' pass has to argue with the evidence rather than
    quietly include them."""

    def test_withdrawn_model_is_not_seated(self):
        """P285c: mlp_small was withdrawn by its own 10-seed probe (median
        -0.088, 5/10 nonpositive) and its checker refuses permanently."""
        assert "mlpshadow_mode" not in _live()

    def test_dead_strategies_stay_off(self):
        """P199: 6 of 9 v5.1 strategies scored KILL with n_directional = 0 -
        they emit nothing at all."""
        assert _live().get("v5_1_strategies_live") is False

    def test_inverted_gate_stays_in_shadow(self):
        """P198/P270: forward evidence is INVERTED - the ticks this gate would
        block made +9.2bps/4h while the ones it keeps made -2.5."""
        assert _live().get("trend_regime_gate") == "shadow"

    def test_gate_that_would_flatten_the_book_stays_off(self):
        """P287 measured enforcement would flatten the standing BTC long
        (alpha ~30bps vs friction x1.5 ~34.5bps)."""
        assert _live().get("dynamic_alpha_gate_enforce") in (None, False)


class TestDvolUnitsBug:
    """THE FINDING. `dvol_to_market_data` reads like 'arm an emergency stop at
    z>=5'. It is not: Deribit publishes DVOL as an INDEX LEVEL (BTC 34.5, ETH
    46.0 measured live), the constitution ALIASES dvol -> dvol_zscore, and
    EXTREME_DVOL fires at >= 5.0. 34.5 >= 5.0 on every tick, forever - and
    EXTREME_DVOL is not in the sleeve's HOLD set, so it falls through to
    veto_flat. Enabling it would have flattened the book permanently.

    P219/P169 class: a raw quantity fed into a field whose consumer expects a
    standardized one. The flag stays absent until a real z is published.
    """

    def test_the_alias_that_makes_it_a_units_bug_is_still_there(self):
        src = _src(REPO / "defense" / "constitution.py")
        assert '"dvol": "dvol_zscore"' in src, (
            "the alias moved - re-derive whether publishing raw dvol is still "
            "a units bug before enabling the flag"
        )

    def test_the_threshold_is_a_zscore_threshold(self):
        src = _src(REPO / "defense" / "constitution.py")
        m = re.search(r"DVOL_ZSCORE_EXTREME\s*=\s*([0-9.]+)", src)
        assert m, "threshold constant moved"
        assert float(m.group(1)) <= 10.0, (
            "a threshold this small can only be a z-score, never a DVOL index "
            "level (which runs 20-100)"
        )

    def test_extreme_dvol_flattens_rather_than_holds(self):
        """Why the bug is severe rather than noisy."""
        src = _src(MAIN)
        m = re.search(r"_SLEEVE_HOLD_NO_TRADE_TRIGGERS\s*=\s*\(([^)]*)\)", src)
        assert m, "hold roster moved"
        assert "EXTREME_DVOL" not in m.group(1), (
            "if EXTREME_DVOL became a HOLD trigger the severity changes - "
            "re-derive this test"
        )

    def test_the_flag_is_enabled_only_because_a_real_zscore_is_published(self):
        """[P306] The units bug is FIXED, so the flag is on - but the reason
        this pin existed has not gone away, it has moved. What must never
        happen is the flag being on while the raw INDEX LEVEL is published,
        so both halves are asserted together and either one reverting is red.
        """
        src = _src(MAIN)
        assert 'market_data["dvol"] = float(_dvz)' in src, (
            "the z-score publication is gone; with the flag on, whatever is "
            "published lands in a field that fires EXTREME_DVOL at 5.0"
        )
        assert 'market_data["dvol"] = float(_drb_m.dvol)' not in src, (
            "the raw Deribit index level is being published again - BTC ~34 "
            "reads as z=34 and permanently flattens the book"
        )
        if "dvol_to_market_data" in _live():
            assert _live()["dvol_to_market_data"] is True


class TestSeatPrecedence:
    """With regimebook enforced, the whale seat would otherwise override it -
    it runs last and wins by default (P293d placed it there when it was the
    only enforced seat). That default is backwards on evidence: the book is
    certified over 6 years / 3 eras and beats a same-cell random control by
    ~286 points (P297); whale's own instrument puts it at 16h t=0.26."""

    def test_whale_defers_to_a_directional_book(self):
        src = _src(MAIN)
        assert 'asset not in getattr(self, "_rb_seat_took", set())' in src, (
            "whale no longer defers - the weaker signal would override the "
            "certified one"
        )

    def test_only_a_directional_book_claims_the_seat(self):
        """A FLAT book is not the book claiming the seat; whale keeps those
        bars, which is the P293j intent."""
        src = _src(MAIN)
        # anchor on the CLAIM site, not the first mention - the reset at the
        # top of the tick comes earlier in the file and matched instead.
        i = src.index('self._rb_seat_took = getattr(self, "_rb_seat_took", set())')
        window = src[max(0, i - 400): i]
        assert "if _rb_dir:" in window, (
            "the marker must be set only for a DIRECTIONAL book target"
        )

    def test_the_marker_is_cleared_every_tick(self):
        """A stale marker would mute whale on a tick the book never ran
        (P155-L5)."""
        src = _src(MAIN)
        assert 'getattr(self, "_rb_seat_took", set()).discard(asset)' in src

    def test_the_reset_precedes_both_seats(self):
        src = _src(MAIN)
        reset = src.index('getattr(self, "_rb_seat_took", set()).discard(asset)')
        book = src.index("self._rb_seat_took = getattr(self, \"_rb_seat_took\", set())")
        whale = src.index('asset not in getattr(self, "_rb_seat_took", set())')
        assert reset < book < whale, (
            "order must be reset -> book claims -> whale defers"
        )


class TestFredNaiveAwareBug:
    """[P299] Enabling macro_gci_live surfaced that P293's headline fix never
    actually delivered data.

    P293's finding was "FRED has a valid key and mock=False but NO CALLER" and
    it wired fetch_if_stale() into the tick. The producer then existed and
    THREW on every call: FRED dates parse NAIVE
    (`datetime.fromisoformat("2026-08-18")`) and `_compute_event_window`
    subtracts them from an AWARE `datetime.now(timezone.utc)` ->
    "can't subtract offset-naive and offset-aware datetimes". fetch()'s
    handler caught it, so the macro context stayed on neutral defaults - the
    SAME observable state as having no producer at all.

    Measured live 2026-08-18 08:33:55 on the first tick after the flip.
    """

    def test_bare_dates_are_made_utc_aware_at_the_parse_boundary(self):
        from datetime import datetime, timezone
        from data_mgmt.feeds.fred_feed import _aware_utc
        naive = datetime.fromisoformat("2026-08-18")
        assert naive.tzinfo is None, "premise: FRED dates parse naive"
        assert _aware_utc(naive).tzinfo is timezone.utc
        # and it must not re-stamp something already aware
        aware = datetime.now(timezone.utc)
        assert _aware_utc(aware) is aware

    def test_the_subtraction_that_killed_the_fetch_now_works(self):
        from datetime import datetime, timezone
        from data_mgmt.feeds.fred_feed import _aware_utc
        release = datetime.fromisoformat("2026-08-18")
        now = datetime.now(timezone.utc)
        # pre-fix this raised TypeError and took the whole fetch with it
        (_aware_utc(release) - _aware_utc(now)).total_seconds()

    def test_every_fred_date_parse_goes_through_the_helper(self):
        """A single unwrapped fromisoformat re-arms the same failure."""
        import re
        import sys
        sys.path.insert(0, str(REPO / "tests"))
        # [P177/P184] Strip DOCSTRINGS as well as comments: the helper's own
        # docstring quotes `datetime.fromisoformat("2026-08-18")` to explain
        # the bug, and a comments-only scan matched that prose - a scanner
        # firing on its own explanation. The repo already has the right tool.
        from _source_scan import code_only
        code = code_only(REPO / "data_mgmt" / "feeds" / "fred_feed.py",
                         strip_docstrings=True)
        # The lookbehind already EXCLUDES wrapped calls, so this counts the
        # unwrapped ones and must be zero. (The first cut asserted
        # len(bare) == len(wrapped), which is 0 == 4 - a test that fails on
        # correct code and would have "passed" only while the bug existed.)
        unwrapped = re.findall(r"(?<!_aware_utc\()datetime\.fromisoformat\(", code)
        wrapped = re.findall(r"_aware_utc\(datetime\.fromisoformat\(", code)
        assert wrapped, "premise: fred_feed parses dates with fromisoformat"
        assert not unwrapped, (
            f"{len(unwrapped)} fromisoformat call(s) are not wrapped in "
            f"_aware_utc - a naive date will reach the comparison again"
        )

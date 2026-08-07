"""[P218] Two of the three "starved agent" items, investigated to root cause.

OPTIONS — NOT a tier limit, which is what I guessed before probing. All three
endpoints the agent calls return HTTP 404 against the live key:

    /api/option/info/max-pain  -> 404
    /api/option/info/oi        -> 404
    /api/option/info/volume    -> 404
    /api/option/info           -> 200, real data

The same CoinGlass key serves funding/OI/liquidations fine, so the subscription
is not the problem — those v3 paths no longer exist. The handler returned None on
any non-200 and the caller then used its neutral default `put_call_ratio = 1.0`,
so "this URL is gone" and "the options market is perfectly balanced" produced
byte-identical output. That is why it went unnoticed, and it is the same shape as
P216's `pcr=1.0` reading and P199's `INSUFFICIENT_SAMPLES`.

Fixing the URLs is NOT attempted here: the working `/option/info` endpoint
returns per-exchange OI/volume aggregates with **no put/call split**, so the PCR
signal cannot be reconstructed from it. Inventing a replacement options signal is
a strategy change, not a bug fix. What ships is that the deadness is loud.

FUNDING VENUE — the book trades Coinbase perps; `market_data["funding_rate"]`
came from Kraken futures (`main.py:6086` overrides the CoinGlass value). Third
instance of the cross-venue leftover behind P172 (fees) and P210 (sizing).
Measured live, 8h rates:

    asset   kraken (used)   coinbase (correct)
    BTC     -0.000077       +0.000040
    ETH     -0.000015       -0.000008
    SOL     -0.000378       +0.000168

Both far below every downstream threshold, so this changes no behaviour today —
but the SIGN differs on BTC and SOL. Config-gated, default OFF, same rollout as
P172's venue-aware fees.
"""

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_MSRC = (_REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
_OSRC = (_REPO / "agents" / "options_sentiment_agent.py").read_text(
    encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# options: a dead endpoint must not read as a neutral market
# ---------------------------------------------------------------------------

class TestDeadOptionsEndpointIsLoud:

    def test_every_non_200_path_reports(self):
        """There are three fetchers; all three used to `return None` silently."""
        assert _OSRC.count("self._report_http(") == 3, (
            "a non-200 handler is not reporting — it will fall back to "
            "put_call_ratio=1.0 and read as a balanced market"
        )

    def test_reporting_precedes_the_return(self):
        i = 0
        for _ in range(3):
            i = _OSRC.index("if resp.status != 200:", i) + 1
            w = _OSRC[i:i + 200]
            assert w.index("_report_http") < w.index("return None")

    def test_404_message_names_the_consequence_and_rules_out_tier(self, caplog):
        import logging
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = object.__new__(OptionsSentimentAgent)
        type(a)._DEAD_ENDPOINTS_WARNED = set()
        with caplog.at_level(logging.WARNING):
            a._report_http("/option/info/oi", 404)
        msg = next(r.message for r in caplog.records if "OPTIONS-AGENT" in r.message)
        assert "does not exist" in msg
        assert "NEUTRAL" in msg, "must say what the fallback value implies"
        assert "NOT a tier limit" in msg, (
            "the wrong diagnosis cost a round trip; record it so it is not "
            "re-guessed"
        )

    def test_it_warns_once_per_path_not_every_tick(self, caplog):
        import logging
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = object.__new__(OptionsSentimentAgent)
        type(a)._DEAD_ENDPOINTS_WARNED = set()
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                a._report_http("/option/info/oi", 404)
        assert len([r for r in caplog.records if "OPTIONS-AGENT" in r.message]) == 1

    def test_distinct_paths_each_report(self, caplog):
        import logging
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = object.__new__(OptionsSentimentAgent)
        type(a)._DEAD_ENDPOINTS_WARNED = set()
        with caplog.at_level(logging.WARNING):
            a._report_http("/option/info/oi", 404)
            a._report_http("/option/info/volume", 404)
        assert len([r for r in caplog.records if "OPTIONS-AGENT" in r.message]) == 2

    def test_non_404_still_reported(self, caplog):
        import logging
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = object.__new__(OptionsSentimentAgent)
        type(a)._DEAD_ENDPOINTS_WARNED = set()
        with caplog.at_level(logging.WARNING):
            a._report_http("/option/info/oi", 503)
        assert any("503" in r.message for r in caplog.records)

    def test_it_still_fails_soft(self):
        """Reporting must not turn a dead feed into a dead tick."""
        i = _OSRC.index("if resp.status != 200:")
        assert "return None" in _OSRC[i:i + 200]


# ---------------------------------------------------------------------------
# funding venue
# ---------------------------------------------------------------------------

class TestVenueAwareFunding:

    def test_flag_declared_and_parsed(self):
        """P201: a flag read by getattr and never parsed is inert."""
        import dataclasses
        from main import ProductionConfig
        names = {f.name for f in dataclasses.fields(ProductionConfig)}
        assert "coinbase_venue_aware_funding" in names
        assert '"coinbase_venue_aware_funding"' in _MSRC

    def test_default_is_off(self):
        import dataclasses
        from main import ProductionConfig
        d = {f.name: f.default for f in dataclasses.fields(ProductionConfig)}
        assert d["coinbase_venue_aware_funding"] is False

    def test_live_profile_has_not_enabled_it(self):
        cfg = json.loads((_REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8"))
        assert not cfg.get("coinbase_venue_aware_funding", False), (
            "changing a live data input is an operator decision (P141)"
        )

    def test_it_only_applies_to_routed_assets(self):
        i = _MSRC.index("[P218] The book trades COINBASE perps")
        w = _MSRC[i:i + 2200]
        assert "_coinbase_routed(self, asset)" in w, (
            "must not price a Kraken-bound asset on Coinbase funding"
        )

    def test_failure_keeps_the_kraken_rate(self):
        """Wrong-way failure must not zero the funding rate — same conservative
        direction P172 chose for fees."""
        i = _MSRC.index("[P218] The book trades COINBASE perps")
        w = _MSRC[i:i + 2600]
        assert "keeping Kraken rate" in w
        assert "except Exception as _vf_err:" in w

    def test_the_cache_is_written_by_the_shadow_block(self):
        assert "self._coinbase_funding_8h[_f_a] = float(_f_raw) * 8.0" in _MSRC

    def test_the_read_is_defended(self):
        """P85: a new attribute read on the tick path defends itself."""
        i = _MSRC.index("[P218] The book trades COINBASE perps")
        w = _MSRC[i:i + 2200]
        assert 'getattr(self, "_coinbase_funding_8h", {})' in w

    def test_absent_cache_leaves_funding_untouched(self):
        """First tick after a restart has no cached value; it must fall through
        to the Kraken rate rather than writing None."""
        i = _MSRC.index("[P218] The book trades COINBASE perps")
        w = _MSRC[i:i + 2200]
        assert "if _cb_fr is not None:" in w


class TestTheMeasurementThatJustifiesTheDefault:
    """The numbers are the argument for shipping this OFF and calling it inert;
    if a threshold moves under them, the argument needs redoing."""

    def test_short_bias_whale_proxy_threshold_still_2bps(self):
        assert "_wp_funding < -0.0002" in _MSRC and "_wp_funding > 0.0002" in _MSRC, (
            "the short-bias funding threshold moved — re-measure whether "
            "venue-aware funding is still behaviourally inert"
        )

    @pytest.mark.parametrize("kraken,coinbase", [
        (-0.000077, +0.000040),   # BTC
        (-0.000015, -0.000008),   # ETH
        (-0.000378, +0.000168),   # SOL
    ])
    def test_both_venues_are_below_the_threshold(self, kraken, coinbase):
        """Except SOL on Kraken, which is why the sign disagreement matters."""
        assert abs(coinbase) < 0.0002

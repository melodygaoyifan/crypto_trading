"""[P219] A flow signal from data we already pay for, shadowed not promoted.

The `flow` agent emits 0.00 every tick because its whale proxy needs
CryptoCompare's `large_transaction_count`, and that account is hard-capped at
**100 API calls/month** (measured: 283 used, 39,223 lifetime, "please upgrade
your account"). The upgrade is blocked.

CoinGlass is already paid for and already fetched every tick, and carries a
directional signal — live at build time:

    liquidation_imbalance   BTC -0.53   ETH -0.50   SOL +0.06
    liquidations 24h        BTC $83M    ETH $64M    SOL $7.4M

SIGN, verified against the live API rather than assumed. `coinglass_feed.py:650`
computes `(long - short) / total`, "positive = more longs liquidated". A 1h probe
appeared to contradict the live value; re-probing at the feed's own **24h** range
resolved it (long $8.5M vs short $31.7M -> -0.576, matching market_data). Getting
this backwards would have inverted the signal, so it is pinned below.

TWO OPPOSITE STRATEGIES ON PURPOSE. Forced liquidation is directionally
ambiguous — squeeze (momentum) vs exhaustion (reversion) — so both are emitted
and the P166 cost-aware IC gate decides. They are exact negations, so at most one
can be right and "both are noise" is a perfectly possible, informative outcome.
That is the P147 lesson (never promote an unvalidated signal into live fusion)
and P198's (an in-sample split is a hypothesis to shadow, not a gate to enforce).

Iron Law 7: observation-only. Nothing here reaches agent_signals or fusion.
"""

from pathlib import Path

import pytest

from strategies.derivatives_flow_v1 import (
    MIN_ABS_IMBALANCE,
    MIN_LIQUIDATION_USD,
    LiquidationExhaustionStrategy,
    LiquidationSqueezeStrategy,
    build_derivatives_flow_strategies,
)

_REPO = Path(__file__).resolve().parents[1]
_MSRC = (_REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")

# Live BTC values, 2026-08-07
_LIVE_BTC = {
    "liquidation_imbalance": -0.5256,
    "total_liquidations_24h": 83_315_141.0,
    "long_liquidations_24h": 8_545_076.0,
    "short_liquidations_24h": 31_726_583.0,
    "open_interest": 99_188_542_270.0,
}


class TestSignConvention:

    def test_the_feed_formula_is_still_long_minus_short(self):
        """The whole signal hangs off this. If the feed's formula flips, both
        strategies invert and the shadow evidence becomes actively misleading."""
        src = (_REPO / "data_mgmt" / "feeds" / "coinglass_feed.py").read_text(
            encoding="utf-8", errors="replace")
        assert ("(liq.long_liquidations_24h - liq.short_liquidations_24h) / total"
                in src), "liquidation_imbalance formula changed — re-derive the sign"

    def test_shorts_liquidated_reads_as_upward_pressure_for_squeeze(self):
        """Negative imbalance = shorts liquidated = forced BUYING = up."""
        sig = LiquidationSqueezeStrategy().evaluate("BTC", _LIVE_BTC)
        assert sig.direction > 0, "squeeze got the sign backwards"

    def test_exhaustion_is_the_exact_opposite(self):
        sq = LiquidationSqueezeStrategy().evaluate("BTC", _LIVE_BTC)
        ex = LiquidationExhaustionStrategy().evaluate("BTC", _LIVE_BTC)
        assert sq.direction == pytest.approx(-ex.direction)
        assert sq.direction != 0.0

    def test_longs_liquidated_flips_both(self):
        md = dict(_LIVE_BTC, liquidation_imbalance=+0.60)
        assert LiquidationSqueezeStrategy().evaluate("BTC", md).direction < 0
        assert LiquidationExhaustionStrategy().evaluate("BTC", md).direction > 0


class TestGating:

    def test_a_tiny_liquidation_print_emits_nothing(self):
        """$50k of liquidations on a $99B OI market is background hum. Emitting
        a direction off it would pollute the IC measurement this exists for."""
        md = dict(_LIVE_BTC, total_liquidations_24h=50_000.0)
        s = LiquidationSqueezeStrategy().evaluate("BTC", md)
        assert s.direction == 0.0 and "quiet" in s.reason

    def test_a_balanced_book_emits_nothing(self):
        md = dict(_LIVE_BTC, liquidation_imbalance=0.02)
        s = LiquidationSqueezeStrategy().evaluate("BTC", md)
        assert s.direction == 0.0 and "balanced" in s.reason

    def test_missing_data_is_named_not_guessed(self):
        s = LiquidationSqueezeStrategy().evaluate("BTC", {})
        assert s.direction == 0.0 and s.reason == "no_liquidation_data"

    def test_thresholds_are_exposed_for_review(self):
        assert MIN_LIQUIDATION_USD > 0 and 0 < MIN_ABS_IMBALANCE < 1

    def test_non_numeric_input_does_not_raise(self):
        """market_data carries strings and None on degraded ticks."""
        md = dict(_LIVE_BTC, liquidation_imbalance="n/a", open_interest=None)
        assert LiquidationSqueezeStrategy().evaluate("BTC", md).direction == 0.0

    def test_direction_is_bounded(self):
        md = dict(_LIVE_BTC, liquidation_imbalance=-5.0)
        assert abs(LiquidationSqueezeStrategy().evaluate("BTC", md).direction) <= 1.0


class TestConfidence:

    def test_confidence_scales_with_size_relative_to_oi(self):
        """Absolute USD is not comparable across assets — $83M is enormous for
        SOL and ordinary for BTC."""
        small = LiquidationSqueezeStrategy().evaluate(
            "BTC", dict(_LIVE_BTC, total_liquidations_24h=2_000_000.0))
        big = LiquidationSqueezeStrategy().evaluate(
            "BTC", dict(_LIVE_BTC, total_liquidations_24h=900_000_000.0))
        assert big.confidence > small.confidence

    def test_confidence_is_bounded(self):
        s = LiquidationSqueezeStrategy().evaluate(
            "BTC", dict(_LIVE_BTC, total_liquidations_24h=9e12))
        assert 0.0 <= s.confidence <= 1.0

    def test_missing_oi_does_not_fabricate_a_denominator(self):
        s = LiquidationSqueezeStrategy().evaluate(
            "BTC", dict(_LIVE_BTC, open_interest=0.0))
        assert 0.0 <= s.confidence <= 1.0


class TestWiring:

    def test_two_strategies_are_built(self):
        names = {type(s).__name__ for s in build_derivatives_flow_strategies()}
        assert names == {"LiquidationSqueezeStrategy",
                         "LiquidationExhaustionStrategy"}

    def test_the_harness_uses_its_own_ledger_prefix(self):
        from defense.strategy_shadow_v5_1 import build_derivflow_shadow_harness
        assert build_derivflow_shadow_harness()._log_prefix == "derivflow"

    def test_main_initialises_and_observes(self):
        assert "build_derivflow_shadow_harness" in _MSRC
        assert "self._derivflow_shadow.observe(asset, market_data)" in _MSRC

    def test_the_init_error_handler_names_the_right_harness(self):
        """I first inserted this block between the funding harness's try and its
        except, which replaced that handler with a silent `pass` and labelled
        derivflow failures as 'FundingShadowHarness init failed'."""
        i = _MSRC.index("DerivFlowShadowHarness init failed")
        w = _MSRC[i - 400:i]
        assert "except Exception as _df_err:" in w
        assert "FundingShadowHarness: ACTIVE" not in _MSRC[i - 200:i]

    def test_the_funding_handler_survived_intact(self):
        assert "FundingShadowHarness init failed" in _MSRC
        i = _MSRC.index("FundingShadowHarness init failed")
        assert "except Exception:\n            pass" not in _MSRC[i - 300:i]

    def test_it_adds_no_api_calls(self):
        """It reads market_data the CoinGlass feed already populated."""
        i = _MSRC.index("self._derivflow_shadow.observe(asset, market_data)")
        assert "market_data" in _MSRC[i:i + 80]

    def test_the_ic_gate_scores_the_new_ledger(self):
        src = (_REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py").read_text(
            encoding="utf-8", errors="replace")
        assert src.count("derivflow") >= 2, (
            "the ledger is written but the IC gate does not read it — the "
            "evidence would accumulate and never be scored"
        )

    def test_it_stays_out_of_fusion(self):
        """Iron Law 7 — and P147: promoting an unvalidated signal is the exact
        mistake this design avoids."""
        assert "derivflow" not in _MSRC.split("v5_1_strategies_live")[-1][:2000]


class TestCCNewsSingleCall:
    """One call for all three assets instead of three, against a 100/month cap."""

    def test_the_request_asks_for_all_tracked_categories(self):
        src = (_REPO / "data_mgmt" / "feeds" / "cryptocompare_news_feed.py"
               ).read_text(encoding="utf-8", errors="replace")
        assert '",".join(TRACKED_CATEGORIES)' in src
        assert '"categories": asset.upper(),' not in src

    def test_every_tracked_asset_cache_is_populated_from_one_response(self):
        src = (_REPO / "data_mgmt" / "feeds" / "cryptocompare_news_feed.py"
               ).read_text(encoding="utf-8", errors="replace")
        assert "for _cat in TRACKED_CATEGORIES:" in src
        assert "self._cache[_cat] = (now, _hit)" in src

    def test_uncategorised_rows_are_kept_for_every_asset(self):
        """Dropping them would shrink the corpus and push headline_count back
        toward 0 — the very condition this change exists to lift."""
        src = (_REPO / "data_mgmt" / "feeds" / "cryptocompare_news_feed.py"
               ).read_text(encoding="utf-8", errors="replace")
        assert "if (not it.categories) or" in src

    def test_tracked_categories_cover_the_traded_assets(self):
        from data_mgmt.feeds.cryptocompare_news_feed import TRACKED_CATEGORIES
        assert {"BTC", "ETH", "SOL"} <= set(TRACKED_CATEGORIES)


class TestOIScopeMismatch:
    """[P219-fix] Caught by reading the FIRST live ledger record, not by a test:
    every confidence came back 1.0.

    `market_data["open_interest"]` is written by CoinGlass (GLOBAL, ~$52.9B for
    ETH) at main.py:6038 and then OVERWRITTEN by the Kraken futures block
    (ONE VENUE, ~$51M) at :6156. The liquidation figures next to it stay global.
    So normalising global liquidations by that key divides a global number by a
    single-venue one — a ~1000x scope error that pinned confidence at its cap and
    would have made the IC gate's confidence weighting meaningless.
    """

    _ETH = {
        "liquidation_imbalance": -0.5146,
        "total_liquidations_24h": 63_232_011.0,
        "long_liquidations_24h": 1.0,
        "short_liquidations_24h": 1.0,
        "coinglass_open_interest_usd": 52_939_514_538.0,   # global
        "open_interest": 51_456_072.0,                     # Kraken only
    }

    def test_global_oi_is_preferred(self):
        s = LiquidationSqueezeStrategy().evaluate("ETH", self._ETH)
        assert s.confidence < 0.5, (
            f"confidence {s.confidence} — still normalising by the single-venue OI"
        )

    def test_the_venue_scoped_oi_would_saturate(self):
        """Pins that this was a real defect, not a cosmetic key rename."""
        md = {k: v for k, v in self._ETH.items()
              if k != "coinglass_open_interest_usd"}
        assert LiquidationSqueezeStrategy().evaluate("ETH", md).confidence == 1.0

    def test_direction_is_unaffected_by_the_denominator(self):
        md = {k: v for k, v in self._ETH.items()
              if k != "coinglass_open_interest_usd"}
        a = LiquidationSqueezeStrategy().evaluate("ETH", self._ETH).direction
        b = LiquidationSqueezeStrategy().evaluate("ETH", md).direction
        assert a == b

    def test_main_publishes_the_venue_scoped_key(self):
        assert 'market_data["coinglass_open_interest_usd"]' in _MSRC

    def test_it_is_published_before_kraken_overwrites(self):
        i = _MSRC.index('market_data["coinglass_open_interest_usd"]')
        j = _MSRC.index('market_data["open_interest"] = _kf_ticker.open_interest_usd')
        assert i < j

    def test_fallback_when_the_global_key_is_absent(self):
        """A degraded tick without CoinGlass must still produce a bounded
        confidence rather than dividing by zero."""
        md = {k: v for k, v in self._ETH.items()
              if k != "coinglass_open_interest_usd"}
        assert 0.0 <= LiquidationSqueezeStrategy().evaluate("ETH", md).confidence <= 1.0

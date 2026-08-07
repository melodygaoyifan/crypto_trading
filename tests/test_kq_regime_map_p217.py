"""[P217] STEADY_UPTREND spent four months routed to the mean-reversion bucket.

The 12 kraken_quant strategies are bucketed by REGIME, not by asset — trading
BTC/ETH/SOL does not make more of them reachable; only the regime does. So which
strategies can ever fire is decided entirely by `_map_regime`.

Measured over 2,545 per-tick diagnostics (2026-04-08 -> 2026-08-07):

    QUIET_ACCUMULATION  1489  58.5%  -> SIDEWAYS   (correct)
    WEAK_CONSOLIDATION   859  33.8%  -> SIDEWAYS   (correct)
    STEADY_UPTREND        93   3.7%  -> SIDEWAYS   *** WRONG: a trending-up
                                                    regime handed to the
                                                    mean-reversion bucket ***
    EXTREME_VOLATILITY    41   1.6%  -> BEAR       (correct)
    NEUTRAL_DRIFT         27   1.1%  -> SIDEWAYS   (right, but by accident)
    VOLATILE_CHOP         17   0.7%  -> SIDEWAYS   (correct)

`STEADY_UPTREND` and `NEUTRAL_DRIFT` were simply absent from `_REGIME_MAP`, and
`_map_regime` returned its SIDEWAYS default in silence. MOMENTUM_RALLY — the only
BULL name that WAS mapped — has never occurred once in four months, so the four
BULL strategies had 93 ticks of their own regime and never saw a single one.

The root shape: a GMM regime name is DATA (cluster names come out of the model),
`_REGIME_MAP` is a hardcoded mirror of it, and the two drift silently the moment
a model is retrained with different cluster names. Same family as P215 (the
diagnostic's hardcoded strategy-name list) and P2 generally. The durable fix is
not the two new entries — it is that an unmapped name is now LOUD.
"""

from pathlib import Path

import pytest

from agents.kraken_quant_agent import (
    Regime,
    _REGIME_MAP,
    _UNMAPPED_REGIMES_WARNED,
)

_REPO = Path(__file__).resolve().parents[1]


def _mapper():
    import agents.kraken_quant_agent as m
    for name in dir(m):
        obj = getattr(m, name)
        if hasattr(obj, "_map_regime"):
            return obj._map_regime
    raise AssertionError("no _map_regime found")


# Names the deployed per-asset GMMs emit (models/regime_classifier/*/gmm_config
# .json, read off the live container). This is the contract the map must cover.
_GMM_EMITTED = {
    "STEADY_UPTREND", "NEUTRAL_DRIFT", "WEAK_CONSOLIDATION", "MOMENTUM_RALLY",
    "QUIET_ACCUMULATION", "VOLATILE_CHOP", "EXTREME_VOLATILITY", "PANIC_SELLOFF",
}


class TestEveryEmittedRegimeIsMapped:

    @pytest.mark.parametrize("name", sorted(_GMM_EMITTED))
    def test_name_is_explicitly_mapped(self, name):
        """Not "resolves to something" — that is always true because of the
        default. It must be an ENTRY, i.e. a decision someone made."""
        assert name in _REGIME_MAP, (
            f"{name} is emitted by the deployed GMMs but absent from "
            f"_REGIME_MAP, so it silently defaults to SIDEWAYS and the BEAR/"
            f"BULL buckets cannot fire in it"
        )

    def test_steady_uptrend_is_bull_not_sideways(self):
        """The defect itself. A trending-up regime routed to mean-reversion."""
        assert _REGIME_MAP["STEADY_UPTREND"] is Regime.BULL

    def test_the_bull_bucket_is_reachable_by_something_that_occurs(self):
        """MOMENTUM_RALLY was the only mapped BULL name and has occurred ZERO
        times in four months. A bucket reachable only by a regime that never
        happens is unreachable."""
        bull = {k for k, v in _REGIME_MAP.items() if v is Regime.BULL}
        assert bull & _GMM_EMITTED - {"MOMENTUM_RALLY"}, (
            "the only BULL-mapped name the GMMs emit is MOMENTUM_RALLY, which "
            "has never fired — the BULL strategies remain unreachable"
        )

    def test_bear_mapping_unchanged(self):
        assert _REGIME_MAP["EXTREME_VOLATILITY"] is Regime.BEAR
        assert _REGIME_MAP["PANIC_SELLOFF"] is Regime.BEAR

    def test_sideways_names_still_sideways(self):
        for n in ("QUIET_ACCUMULATION", "WEAK_CONSOLIDATION", "VOLATILE_CHOP",
                  "NEUTRAL_DRIFT"):
            assert _REGIME_MAP[n] is Regime.SIDEWAYS


class TestUnmappedIsLoud:

    def test_an_unknown_name_warns_once(self, caplog):
        import logging
        _UNMAPPED_REGIMES_WARNED.discard("BRAND_NEW_CLUSTER")
        m = _mapper()
        with caplog.at_level(logging.WARNING):
            for _ in range(4):
                assert m("BRAND_NEW_CLUSTER") is Regime.SIDEWAYS
        hits = [r for r in caplog.records if "KQ_REGIME" in r.message]
        assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
        assert "BRAND_NEW_CLUSTER" in hits[0].message

    def test_the_warning_names_the_consequence_and_the_fix(self, caplog):
        import logging
        _UNMAPPED_REGIMES_WARNED.discard("ANOTHER_ONE")
        with caplog.at_level(logging.WARNING):
            _mapper()("ANOTHER_ONE")
        msg = next(r.message for r in caplog.records if "KQ_REGIME" in r.message)
        assert "cannot fire" in msg
        assert "_REGIME_MAP" in msg

    def test_a_mapped_name_is_silent(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            _mapper()("QUIET_ACCUMULATION")
        assert not [r for r in caplog.records if "KQ_REGIME" in r.message]

    def test_regime_N_shape_stays_quiet(self, caplog):
        """`REGIME_0..7` is the expected unnamed-cluster shape, not drift."""
        import logging
        with caplog.at_level(logging.WARNING):
            assert _mapper()("REGIME_3") is Regime.SIDEWAYS
        assert not [r for r in caplog.records if "KQ_REGIME" in r.message]

    def test_none_is_still_sideways(self):
        assert _mapper()(None) is Regime.SIDEWAYS


class TestBucketingIsByRegimeNotAsset:
    """Recording the thing that is easy to assume otherwise: all three assets
    share ONE set of 12 strategies, selected by regime. Trading BTC+ETH+SOL does
    not widen the reachable set."""

    def test_the_map_has_no_asset_dimension(self):
        for k in _REGIME_MAP:
            assert k not in ("BTC", "ETH", "SOL")

    def test_three_buckets_only(self):
        assert {v for v in _REGIME_MAP.values()} <= {
            Regime.BULL, Regime.BEAR, Regime.SIDEWAYS}

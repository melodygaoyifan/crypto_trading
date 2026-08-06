"""[P172] One tick, one asset, two friction numbers.

`coinbase_venue_aware_fees` has been ON in `configs/live_high_risk.json` since
2026-08-04 (P165, by explicit operator instruction). It made the alpha gate
price a Coinbase-routed asset at the Coinbase schedule (0/3bps) instead of
Kraken's blended tier (16/26bps) — the correct number for where the order
actually goes.

But only the alpha gate. The `_fee_context` dict built ~60 lines later in the
same method hardcoded the Kraken tier and labelled itself
`"kraken_plus_fee_blender"`, and three consumers read it:

  * `main.py:12793`  — a second pre-trade veto, `alpha < friction * 1.5`
  * `main.py:18997`  — the paper exit-fee accrual
  * `main.py:15827`  — `friction_fee_bps` telemetry, which *overrides* the
                       alpha gate's own value, so the dashboard reported
                       Kraken pricing for a decision made on Coinbase pricing

So the gate said 3bps and everything downstream said 26bps, on the same tick,
for the same asset. These tests pin the resolver that both blocks now share,
and the asymmetry that makes its failure mode safe: every unknown resolves to
Kraken, because over-charging friction only costs opportunity while
under-charging spends money.
"""

import io
from pathlib import Path

import pytest

from core.execution_service import (
    _COINBASE_MAKER_BPS,
    _COINBASE_TAKER_BPS,
    resolve_venue_fee_bps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

KRAKEN_MAKER = 16.0
KRAKEN_TAKER = 26.0


def _resolve(enabled, routed, maker=KRAKEN_MAKER, taker=KRAKEN_TAKER):
    return resolve_venue_fee_bps(
        kraken_maker_bps=maker,
        kraken_taker_bps=taker,
        venue_aware_enabled=enabled,
        is_coinbase_routed=routed,
    )


class TestCoinbaseRoutedIsPricedOnCoinbase:
    def test_enabled_and_routed_uses_the_venue_schedule(self):
        maker, taker, venue, source = _resolve(True, True)
        assert (maker, taker) == (_COINBASE_MAKER_BPS, _COINBASE_TAKER_BPS)
        assert venue == "coinbase"
        assert source == "coinbase_venue_schedule"

    def test_the_correction_is_worth_about_23bps_of_taker_friction(self):
        _m, taker, _v, _s = _resolve(True, True)
        assert KRAKEN_TAKER - taker == pytest.approx(23.0)

    def test_kraken_tier_is_ignored_entirely_when_routed(self):
        # The blended Kraken numbers must not leak into a Coinbase price, even
        # at an unusual tier.
        maker, taker, _v, _s = _resolve(True, True, maker=1.0, taker=2.0)
        assert (maker, taker) == (_COINBASE_MAKER_BPS, _COINBASE_TAKER_BPS)


class TestEveryUnknownFallsBackToKraken:
    """Over-charging blocks a trade; under-charging takes one. Prefer blocked."""

    @pytest.mark.parametrize("enabled,routed", [
        (False, False),   # flag off
        (False, True),    # flag off, routed anyway — flag wins
        (True, False),    # flag on, RoutingPolicy says Kraken
    ])
    def test_falls_back_to_the_kraken_tier(self, enabled, routed):
        maker, taker, venue, source = _resolve(enabled, routed)
        assert (maker, taker) == (KRAKEN_MAKER, KRAKEN_TAKER)
        assert venue == "kraken"
        assert source == "kraken_plus_fee_blender"

    def test_the_flag_alone_never_reprices(self):
        # Turning the flag on must not discount an asset that still trades on
        # Kraken; the routing lookup is the authority.
        assert _resolve(True, False)[1] == KRAKEN_TAKER

    def test_fallback_is_never_cheaper_than_the_kraken_tier(self):
        for enabled, routed in ((False, False), (False, True), (True, False)):
            assert _resolve(enabled, routed)[1] >= KRAKEN_TAKER


class TestProvenanceIsNotDecoration:
    """`fee_source` names where the number came from, or it is worthless."""

    def test_blender_label_only_on_blender_numbers(self):
        for enabled, routed in ((True, True), (True, False), (False, True), (False, False)):
            maker, taker, _venue, source = _resolve(enabled, routed)
            if source == "kraken_plus_fee_blender":
                assert (maker, taker) == (KRAKEN_MAKER, KRAKEN_TAKER)

    def test_coinbase_label_only_on_coinbase_numbers(self):
        for enabled, routed in ((True, True), (True, False), (False, True), (False, False)):
            maker, taker, _venue, source = _resolve(enabled, routed)
            if source == "coinbase_venue_schedule":
                assert (maker, taker) == (_COINBASE_MAKER_BPS, _COINBASE_TAKER_BPS)

    def test_venue_and_source_never_disagree(self):
        pairs = {("coinbase", "coinbase_venue_schedule"),
                 ("kraken", "kraken_plus_fee_blender")}
        for enabled, routed in ((True, True), (True, False), (False, True), (False, False)):
            _m, _t, venue, source = _resolve(enabled, routed)
            assert (venue, source) in pairs

    def test_source_is_always_populated(self):
        for enabled, routed in ((True, True), (True, False), (False, True), (False, False)):
            assert _resolve(enabled, routed)[3]


class TestResolverIsPure:
    def test_repeated_calls_agree(self):
        assert _resolve(True, True) == _resolve(True, True)

    def test_maker_is_never_above_taker(self):
        for enabled, routed in ((True, True), (True, False)):
            maker, taker, _v, _s = _resolve(enabled, routed)
            assert maker <= taker

    def test_returns_floats_not_ints(self):
        maker, taker, _v, _s = _resolve(True, True)
        assert isinstance(maker, float) and isinstance(taker, float)


class TestBothCallSitesShareTheResolver:
    """The point of the fix: one venue decision per tick, not two."""

    def _main(self):
        return io.open(REPO_ROOT / "main.py", encoding="utf-8").read()

    def test_fee_context_no_longer_hardcodes_the_blender_label(self):
        src = self._main()
        assert '"fee_source": "kraken_plus_fee_blender",' not in src, (
            "fee_context is stamping the blender label on numbers again without "
            "checking whether they came from the blender"
        )

    def test_fee_context_carries_the_venue(self):
        assert '"venue": _fee_venue,' in self._main()

    def test_alpha_gate_and_fee_context_share_one_resolution(self):
        src = self._main()
        assert "_venue_fee_resolved = (maker_fee_bps, taker_fee_bps" in src
        assert "_maker_fee_bps, _taker_fee_bps, _fee_venue, _fee_source = _venue_fee_resolved" in src

    def test_resolution_is_a_per_tick_local_not_instance_state(self):
        # `asset` is a parameter of _process_4h_tick_inner, so a `self.` field
        # would carry one asset's venue pricing into the next asset's tick.
        src = self._main()
        assert "self._venue_fee_bps" not in src
        assert "_venue_fee_resolved = None" in src

    def test_local_is_initialised_before_the_guarded_block(self):
        src = self._main()
        init = src.index("_venue_fee_resolved = None")
        first_use = src.index("_venue_fee_resolved = (maker_fee_bps")
        guard = src.index("if self._fee_blending_enabled and hasattr(self.engine, 'guarantees'):")
        assert init < guard < first_use, (
            "the fee_context builder can be reached with _venue_fee_resolved "
            "undefined"
        )

    def test_unresolved_tick_keeps_the_conservative_tier(self):
        src = self._main()
        assert '_fee_venue, _fee_source = "kraken", "kraken_plus_fee_blender"' in src


class TestTelemetryExposesTheVenue:
    def test_friction_export_carries_the_venue(self):
        src = io.open(REPO_ROOT / "main.py", encoding="utf-8").read()
        assert '"friction_venue"' in src

    def test_missing_context_is_unknown_not_kraken(self):
        # A tick that built no fee_context must not be recorded as a Kraken-
        # priced decision; that is the missing-vs-neutral collapse this repo
        # keeps rediscovering (P170, P171).
        src = io.open(REPO_ROOT / "main.py", encoding="utf-8").read()
        assert '.get("venue", "UNKNOWN")' in src

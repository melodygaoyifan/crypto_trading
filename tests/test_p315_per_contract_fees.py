"""
[P315] CDE fees are PER CONTRACT, and the bps model understated them ~3x.

Found by reading the money path against live evidence rather than the code's
own constants. `core/paper_fee_service.VENUE_FEE_STD` prices Coinbase at
maker 0.0 / taker 3.0 bps; the venue's own reported fees in
`data/fill_quality.jsonl` say BTC maker 9.37 / taker 9.87 and ETH maker 13.76,
because the charge is a flat ~$0.60 (BTC) / ~$0.26 (ETH) PER CONTRACT.

Two consequences, both measured:

  LIVE   the alpha gate cleared BTC at alpha=22bps vs threshold=18bps. Adding
         the omitted ~6.9bps/leg (charged round-trip, P167, x1.1) puts that
         threshold near 33bps — the trade is net NEGATIVE and should be
         rejected. Every entry was taken on ~3x-understated cost.

  LABS   funding_legs_lab charged the same 3bps, so the P297 six-year
         certification was priced the same wrong way. Re-run with the honest
         per-bar fee: book 243.9% -> 144.5%, trend_only 151.7% -> 120.5%,
         both now BELOW buy-and-hold's 186.3% (which does not trade and is
         fee-immune). The funding-leg increment collapses 92.2 -> 24.0.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.cde_fees import (  # noqa: E402
    CDE_FEE_ASSUMED,
    CDE_FEE_BPS,
    _contract_sizes,
    cde_fee_bps,
)


# =============================================================================
# 1. The model must reproduce the VENUE's own reported fees
# =============================================================================

class TestReproducesMeasuredFills:

    # (asset, contracts, fill price, fee_usd, is_maker) straight from
    # data/fill_quality.jsonl — the venue's numbers, not ours (P169).
    MEASURED = [
        ("BTC", 2, 64105.0, 1.202, True),
        ("BTC", 2, 64330.0, 1.269, False),
        ("BTC", 2, 64305.0, 1.205, True),
        ("BTC", 2, 64435.0, 1.207, True),
        ("BTC", 2, 64345.0, 1.205, True),
        ("ETH", 6, 1916.5, 1.582, True),
    ]

    @pytest.mark.parametrize("asset,ct,px,fee,is_maker", MEASURED)
    def test_within_a_bp_of_the_realized_fill(self, asset, ct, px, fee, is_maker):
        """The whole point: the model must land on what we were CHARGED."""
        cs = _contract_sizes()[asset]
        realized_bps = fee / (ct * cs * px) * 1e4
        got = cde_fee_bps(asset, px, is_maker=is_maker)
        assert got is not None
        assert abs(got[0] - realized_bps) < 1.0, (
            f"{asset}: model {got[0]:.2f}bps vs realized {realized_bps:.2f}bps")

    def test_the_old_model_was_wrong_by_at_least_3x(self):
        """Anti-regression on the FINDING itself. VENUE_FEE_STD says taker
        3.0bps; if a future edit makes the honest number agree with that, the
        per-contract model has been silently defeated."""
        from core.paper_fee_service import VENUE_FEE_STD
        modeled = VENUE_FEE_STD["coinbase"]["taker"] * 1e4      # 3.0
        honest = cde_fee_bps("BTC", 64435.0, is_maker=False)[0]
        assert honest > 3 * modeled, (
            f"honest BTC taker fee {honest:.2f}bps is no longer >3x the "
            f"modeled {modeled:.1f}bps — check the measurement, not the test")

    def test_maker_is_not_free(self):
        """The model charged ZERO for a maker fill that really costs ~9.4bps.
        A per-contract fee is paid on both sides."""
        assert cde_fee_bps("BTC", 64435.0, is_maker=True)[0] > 5.0


# =============================================================================
# 2. The structural property: cost moves INVERSELY with price
# =============================================================================

class TestPerContractStructure:

    def test_the_fee_is_PRICE_INVARIANT(self):
        """[P334] THE CORRECTION, reversing this class original claim.

        P315 read the fee as flat dollars per contract from 5 fills spanning
        64.1k-64.3k -- no price range at all, so flat and percentage were
        indistinguishable. With ~8% now available the flat model FAILS: BTC
        price +7.7%, fee +5.5% (flat predicts +0.0%, percentage +7.7%); bps
        are stable per asset across that move while dollars per contract are
        not. The test that lived here asserted bps DOUBLE when price halves,
        which is the property this refutes.
        """
        for a, px in (("BTC", 64330.0), ("ETH", 1916.5), ("SOL", 84.44)):
            lo = cde_fee_bps(a, px, is_maker=False)[0]
            hi = cde_fee_bps(a, px * 3.0, is_maker=False)[0]
            assert lo == pytest.approx(hi), a

    def test_the_old_3bps_model_is_still_far_too_cheap(self):
        """What SURVIVES: paper_fee_service prices Coinbase at 0/3bps and the
        measured rate is ~9.4-9.9bps on BTC, ~13.8 on ETH. The MAGNITUDE of
        P315 finding stands; its STRUCTURE did not."""
        assert cde_fee_bps("BTC", 64330.0, is_maker=False)[0] > 9.0
        assert cde_fee_bps("BTC", 64330.0, is_maker=True)[0] > 5.0

    def test_eth_is_dearer_than_btc_in_bps(self):
        """Still true, now for the right reason: the per-asset RATE differs
        (ETH ~13.8bps vs BTC ~9.4bps). P315 attributed this to ETH smaller
        nano notional, which was the wrong explanation for a real reading."""
        eth = cde_fee_bps("ETH", 1916.5, is_maker=True)[0]
        btc = cde_fee_bps("BTC", 64435.0, is_maker=True)[0]
        assert eth > btc
        assert CDE_FEE_BPS["ETH"]["maker"] > CDE_FEE_BPS["BTC"]["maker"]


# =============================================================================
# 3. Fail directions — every unknown resolves EXPENSIVE (P167)
# =============================================================================

class TestFailDirections:

    def test_unusable_price_returns_none_not_zero(self):
        """Absence is not zero (P2). The caller must keep its existing figure
        rather than be handed a fabricated free trade."""
        for px in (0.0, -1.0, float("nan"), None, "abc"):
            assert cde_fee_bps("BTC", px, is_maker=False) is None

    def test_unknown_asset_falls_back_to_the_most_expensive_measured(self):
        got = cde_fee_bps("DOGE", 0.25, is_maker=False, contract_size=5000.0)
        assert got is not None
        worst = max(max(v["maker"], v["taker"])
                    for v in CDE_FEE_BPS.values())
        assert got[0] == pytest.approx(worst)
        assert "assumed" in got[1]

    def test_sol_is_flagged_as_assumed_not_measured(self):
        """No SOL fill with a reported fee exists. It must price at the
        expensive end AND say so, or an assumption reads as a measurement."""
        assert "SOL" in CDE_FEE_ASSUMED
        assert cde_fee_bps("SOL", 73.0, is_maker=True)[1] == \
            "assumed_no_measured_fill"
        assert cde_fee_bps("BTC", 64435.0, is_maker=True)[1] == \
            "measured_fill_quality"

    def test_taker_is_never_cheaper_than_maker(self):
        for a, sched in CDE_FEE_BPS.items():
            assert sched["taker"] >= sched["maker"], a


# =============================================================================
# 4. Wiring — the gate may only ever RAISE the cost, and stays default-OFF
# =============================================================================

class TestGateWiring:

    def _src(self):
        return io.open(REPO / "main.py", encoding="utf-8").read()

    def test_the_gate_reads_the_price_key_a_producer_actually_writes(self):
        """[P321b] It read market_data["price"], which no producer writes —
        the price is under "current_price". So the honest fee silently fell
        back to the modelled 3bps on EVERY tick while the calibrated alpha
        applied: `[P315-FEE] BTC: per-contract fee not priceable (px=0.0)`
        in production. The P2 reader/writer mismatch, in the fix for a
        mispricing."""
        src = self._src()
        i = src.index("[P315-FEE]")
        blk = src[max(0, i - 2500):i]
        assert 'market_data.get("current_price"' in blk, (
            "the fee block is not reading current_price — it will fall back "
            "to the modelled fee on every tick")

    def test_the_gate_applies_it_with_max_so_it_can_only_tighten(self):
        """A fee correction that could LOWER the charge would re-open the
        undercharging this exists to end (P167)."""
        src = self._src()
        i = src.index("[P315-FEE]")
        blk = src[max(0, i - 2000):i]
        assert "maker_fee_bps = max(maker_fee_bps, _mk[0])" in blk
        assert "taker_fee_bps = max(taker_fee_bps, _tk[0])" in blk

    def test_config_trio_and_default_off(self):
        """Declared + parsed + consumed (P201). Default OFF: arming it can
        stop an asset trading, which is an operator decision (P141) — the
        P291 precedent for a cost correction that changes what trades."""
        import dataclasses
        import main
        names = {f.name for f in dataclasses.fields(main.ProductionConfig)}
        assert "coinbase_per_contract_fees" in names
        assert main.ProductionConfig().coinbase_per_contract_fees is False
        src = self._src()
        assert 'data.get("coinbase_per_contract_fees", False)' in src

    def test_armed_now_that_the_alpha_side_is_calibrated(self):
        """[P318] P315 recommended arming this. That recommendation is
        WITHDRAWN and this test is the guard.

        The gate is wrong on BOTH sides. Measured per position change at
        honest fees, the book's realized gross is 40-450 bps/leg against
        ~16 bps/leg of cost (2.5x-27x). The gate asserts a FLAT 30bps of
        edge per trade against 27.7bps round-trip friction (~1.08x) — a
        constant calibrated for the TREND seat's fast signal (P231),
        applied to a book that holds ~40 bars.

        So arming the honest FEE while the alpha side still asserts 30bps
        would reject trades whose realized edge is many times their cost.
        The fee correction is right; shipping it alone is not.

        Arming requires the alpha side calibrated to the seat's own holding
        horizon, with its own evidence and P-entry — at which point this
        test becomes a DECIDED-value pin (the P237/P270 pattern)."""
        live = json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        # [P321] ARMED — and the precondition this guard demanded is met: the
        # alpha side IS calibrated (P320) and the two are interlocked in code,
        # so the fee can no longer ship alone. Now a DECIDED-value pin, so a
        # silent revert of EITHER half fails here (P237/P270).
        assert live.get("coinbase_per_contract_fees") is True
        assert live.get("seat_alpha_calibrated") is True, (
            "the per-contract fee is armed while the calibrated alpha is NOT — "
            "that combination REJECTS trades measured at 2.5x-27x edge/cost "
            "(P318). Both halves move together or neither does.")

    def test_the_alpha_side_is_now_calibrated_so_arming_is_re_opened(self):
        """[P320] This guard was written by P318 to fire exactly here: "if the
        seat's asserted edge ever becomes horizon-calibrated, the objection no
        longer applies and the arming decision must be RE-OPENED rather than
        left blocked by a stale comment." It fired, and this is the update.

        The alpha side IS now calibrated (core/seat_alpha.py: measured gross
        bps per round trip, era-minimum), and the two corrections are
        INTERLOCKED so neither can ship alone. So P318's objection is
        resolved — what remains is a live decision, not a defect:

            with both armed, BTC -32.9, ETH -3.9, SOL -80.3 vs threshold
            -> the book goes ~flat, which is the honest answer on this
               evidence and is an operator call (P141).
        """
        import dataclasses
        import main
        names = {f.name for f in dataclasses.fields(main.ProductionConfig)}
        assert "seat_alpha_calibrated" in names, (
            "the calibrated-alpha flag vanished; P318's objection to arming "
            "the fee alone is live again")
        from core.seat_alpha import resolve_seat_edge
        # interlocked: neither half alone changes the asserted edge
        assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True, False) == 30.0
        assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, False, True) == 30.0


# =============================================================================
# 5. The lab must price reality — and must refuse to fall back silently
# =============================================================================

class TestLabPricing:

    def test_lab_prices_a_CONSTANT_rate_not_a_price_dependent_one(self):
        """[P334] Reversed. This asserted the series RISES as price falls,
        which is the flat-fee property the correction refutes. The lab now
        charges a constant per-asset rate, and the P315 re-pricing that
        charged ~63bps/leg at BTC $10k must be re-run before it is cited."""
        import pandas as pd
        import training.funding_legs_lab as L
        c = pd.Series([10000.0, 30000.0, 69000.0])
        got = list(L.per_leg_cost_series("BTC", c))
        assert got[0] == pytest.approx(got[1]) == pytest.approx(got[2])
        assert 5e-4 < got[0] < 3e-3

    def test_lab_refuses_rather_than_silently_using_3bps(self):
        """If no MEASURED rate exists for the asset, the lab must STOP.

        [P334] The refusal now also covers the unknown-asset fallback: that
        path always returns a number (the worst measured rate), so a
        None-only check would be unreachable — a dead guard reads exactly
        like one that never fires (P174). Pricing six years of one asset on
        another asset's fee is the same fabrication as the 3bps constant."""
        import pandas as pd
        import training.funding_legs_lab as L
        with pytest.raises(SystemExit):
            L.per_leg_cost_series("DOGE", pd.Series([0.25, 0.26]))

    def test_legacy_mode_still_reproduces_the_old_constant(self):
        """Kept so the P297 run can be reproduced exactly — a verdict whose
        inputs cannot be re-created is not auditable."""
        import pandas as pd
        import training.funding_legs_lab as L
        try:
            L.FEE_MODEL = "legacy_3bps"
            s = L.per_leg_cost_series("BTC", pd.Series([64000.0, 10000.0]))
            assert s.nunique() == 1, "legacy mode must be price-INdependent"
            assert s.iloc[0] == pytest.approx((2.0 / 2.0 + 3.0) / 1e4)
        finally:
            L.FEE_MODEL = "per_contract"

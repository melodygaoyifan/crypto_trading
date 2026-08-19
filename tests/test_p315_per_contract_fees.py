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
    CDE_FEE_PER_CONTRACT_USD,
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

    def test_bps_cost_doubles_when_price_halves(self):
        """This is why no bps constant can price six years: a flat fee on a
        fixed-size contract is a rising % as price falls. At BTC $10k a 0.01
        nano is ~$100 of notional, so $0.60 is ~60bps."""
        hi = cde_fee_bps("BTC", 64000.0, is_maker=False)[0]
        lo = cde_fee_bps("BTC", 32000.0, is_maker=False)[0]
        assert lo == pytest.approx(hi * 2, rel=1e-6)
        cheap = cde_fee_bps("BTC", 10000.0, is_maker=False)[0]
        assert cheap > 50.0, "a $10k BTC nano contract must price above 50bps"

    def test_smaller_notional_contract_is_more_expensive_in_bps(self):
        """ETH nano ($192 notional) costs more in bps than BTC nano ($644),
        even though its per-contract fee is LOWER in dollars."""
        btc = cde_fee_bps("BTC", 64435.0, is_maker=True)[0]
        eth = cde_fee_bps("ETH", 1916.5, is_maker=True)[0]
        assert CDE_FEE_PER_CONTRACT_USD["ETH"]["maker"] < \
            CDE_FEE_PER_CONTRACT_USD["BTC"]["maker"]
        assert eth > btc


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
                    for v in CDE_FEE_PER_CONTRACT_USD.values())
        assert got[0] == pytest.approx(worst / (5000.0 * 0.25) * 1e4)
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
        for a, sched in CDE_FEE_PER_CONTRACT_USD.items():
            assert sched["taker"] >= sched["maker"], a


# =============================================================================
# 4. Wiring — the gate may only ever RAISE the cost, and stays default-OFF
# =============================================================================

class TestGateWiring:

    def _src(self):
        return io.open(REPO / "main.py", encoding="utf-8").read()

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

    def test_lab_uses_a_price_dependent_series_not_a_constant(self):
        import training.funding_legs_lab as L
        assert hasattr(L, "per_leg_cost_series")
        assert L.FEE_MODEL == "per_contract", (
            "the lab has been switched back to the 3bps constant that "
            "mis-priced the P297 certification")

    def test_lab_refuses_rather_than_silently_using_3bps(self):
        """If the per-contract fee cannot be resolved, the lab must STOP.
        Falling back to the constant is the defect it exists to correct."""
        import pandas as pd
        import training.funding_legs_lab as L
        from core import cde_fees
        saved = dict(cde_fees.CDE_FEE_PER_CONTRACT_USD)
        try:
            cde_fees.CDE_FEE_PER_CONTRACT_USD.clear()
            with pytest.raises(SystemExit):
                L.per_leg_cost_series("BTC", pd.Series([64000.0, 64100.0]))
        finally:
            cde_fees.CDE_FEE_PER_CONTRACT_USD.update(saved)

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

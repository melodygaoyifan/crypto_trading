"""[P1-1] Venue fee schedule: real structure, Phase-0 provenance, live wiring."""
import io
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "cost" / "venue_fee_schedule_coinbase.json"
PARAMS = REPO / "reports" / "phase0_params.json"

from core import cde_fees  # noqa: E402
from core.cde_fees import (VenueFeeSchedule, cde_fee_bps,  # noqa: E402
                           load_venue_fee_schedule, venue_fee_bps)

# Current-tier venue fills, copied from the Coinbase ledger (Phase 0
# data/phase0/fills_raw.parquet): (asset, ct, price, liquidity, commission).
OCT_FILLS = [
    ("BTC", 2, 83245.0, "MAKER", 1.802),
    ("BTC", 1, 83375.0, "TAKER", 0.944),
    ("BTC", 1, 81600.0, "MAKER", 0.885),
    ("BTC", 1, 80630.0, "TAKER", 0.916),
    ("BTC", 1, 81315.0, "TAKER", 0.923),
]


def _sched():
    s = load_venue_fee_schedule(str(CFG), reload=True)
    assert s is not None, "configs/venue_fee_schedule_coinbase.json must load"
    return s


class TestProvenance:
    def test_config_is_what_the_producer_writes_from_phase0(self):
        if not PARAMS.exists():
            pytest.skip("reports/phase0_params.json absent")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evfs", REPO / "scripts" / "export_venue_fee_schedule.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        want = m.build(json.load(io.open(PARAMS, encoding="utf-8")))
        got = json.load(io.open(CFG, encoding="utf-8"))
        for k in ("rate_bps_maker", "rate_bps_taker", "fixed_per_contract_usd",
                  "min_fee_per_contract_usd", "contract_size"):
            assert got[k] == want[k], k

    def test_producer_refuses_without_phase0(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evfs2", REPO / "scripts" / "export_venue_fee_schedule.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        assert m.main(["--params", str(tmp_path / "nope.json"),
                       "--out", str(tmp_path / "o.json")]) == 2
        assert not (tmp_path / "o.json").exists()


class TestStructure:
    @pytest.mark.parametrize("a,ct,px,liq,comm", OCT_FILLS)
    def test_reproduces_current_tier_fills_to_a_cent(self, a, ct, px, liq, comm):
        s = _sched()
        assert s.fee_usd(a, ct, px, liq == "MAKER") == pytest.approx(comm, abs=0.01)

    def test_fixed_term_is_ADDITIVE_not_a_floor(self):
        """Phase 0: commission = pct + fixed/ct. max(pct, fixed) under-prices
        a BTC nano fill by ~12% — the spec's max() form does not fit this venue."""
        s = _sched()
        pct = 1 * 0.01 * 81315.0 * s.rate_bps_taker / 1e4
        assert s.fee_usd("BTC", 1, 81315.0, False) == pytest.approx(
            pct + s.fixed_per_contract_usd)
        assert s.fee_usd("BTC", 1, 81315.0, False) > max(pct, s.fixed_per_contract_usd)

    def test_bps_independent_of_contract_count(self):
        s = _sched()
        assert s.fee_bps("ETH", 1, 2400.0, False) == pytest.approx(
            s.fee_bps("ETH", 7, 2400.0, False))

    def test_maker_not_more_expensive_than_taker(self):
        s = _sched()
        assert s.rate_bps_maker <= s.rate_bps_taker

    def test_zero_notional_is_inf_not_free(self):
        assert math.isinf(_sched().fee_bps("BTC", 0, 81000.0, False))


class TestLivePricing:
    def test_venue_fee_bps_uses_schedule(self):
        bps, prov = venue_fee_bps("BTC", 81315.0, is_maker=False)
        assert prov == "venue_fee_schedule_phase0"
        assert bps == pytest.approx(10.0 + 0.11 / 813.15 * 1e4, abs=1e-6)

    def test_unpriceable_price_is_none(self):
        for px in (0.0, -1.0, float("nan"), None, "x"):
            assert venue_fee_bps("BTC", px, is_maker=False) is None

    def test_uncovered_asset_falls_back_to_table(self):
        got = venue_fee_bps("XRP", 1.3, is_maker=True, contract_size=500.0)
        assert got == cde_fee_bps("XRP", 1.3, is_maker=True, contract_size=500.0)

    def test_missing_schedule_falls_back_to_table(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_VENUE_FEE_SCHEDULE", str(tmp_path / "missing.json"))
        cde_fees._SCHEDULE_CACHE.clear()
        try:
            got = venue_fee_bps("BTC", 81000.0, is_maker=False)
            assert got == cde_fee_bps("BTC", 81000.0, is_maker=False)
        finally:
            cde_fees._SCHEDULE_CACHE.clear()

    def test_training_lab_rate_probe_is_unchanged(self):
        """training/funding_legs_lab.py reads cde_fee_bps(a, 1.0, cs=1.0) as a
        constant RATE; training/ is off-limits, so that function must not
        acquire the fixed term (~1,100bps/leg on that probe)."""
        bps, _ = cde_fee_bps("BTC", 1.0, is_maker=False, contract_size=1.0)
        assert bps == cde_fees.CDE_FEE_BPS["BTC"]["taker"]


class TestWiring:
    def _main(self):
        return io.open(REPO / "main.py", encoding="utf-8").read()

    def test_gate_prices_with_the_schedule(self):
        assert "from core.cde_fees import venue_fee_bps as _cfb" in self._main()

    def test_seat_alpha_interlock_uses_the_same_function(self):
        src = io.open(REPO / "core" / "seat_alpha.py", encoding="utf-8").read()
        assert "venue_fee_bps(asset, price, is_maker=False) is None" in src

    def test_blender_is_off_the_per_tick_hot_path(self):
        src = self._main()
        assert "_fc.blender.apply" not in src
        assert "fee_calc.blender.apply" not in src
        assert "if self._fee_blending_enabled and hasattr(self.engine, 'guarantees'):" not in src

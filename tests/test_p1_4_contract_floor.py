"""[P1-4] Contract-granularity sizing: a sub-1-contract budget is not rounded up."""
import json
from pathlib import Path

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

REPO = Path(__file__).resolve().parents[1]


def _sleeve(mode, cur=0.0, budget=(0.01, 10_000.0, 648.0)):
    s = CoinbaseSleeve(adapter=None, assets=("XRP", "ETH"), contract_floor_mode=mode)
    s._sizing_inputs = lambda asset: budget          # fr, eq, one_ct  -> 0.15 ct
    s.signed_contracts = lambda asset: cur
    return s


class TestFloor:
    def test_shadow_keeps_one_contract_and_logs(self, caplog):
        s = _sleeve("shadow")
        with caplog.at_level("INFO"):
            assert s.target_for("XRP", 1.0) == 1
        assert any("WOULD_SUPPRESS" in r.message for r in caplog.records)

    def test_shadow_logs_once_per_4h(self, caplog):
        s = _sleeve("shadow")
        with caplog.at_level("INFO"):
            for _ in range(5):
                s.target_for("XRP", 1.0)
        assert sum("[P1-4-FLOOR]" in r.message for r in caplog.records) == 1

    def test_enforce_flat_stays_flat(self):
        assert _sleeve("enforce", cur=0.0).target_for("XRP", 1.0) == 0

    def test_enforce_holds_same_side_position(self):
        assert _sleeve("enforce", cur=1.0).target_for("XRP", 1.0) == 1

    def test_enforce_closes_opposite_never_opens(self):
        assert _sleeve("enforce", cur=1.0).target_for("XRP", -1.0) == 0

    def test_off_is_historical(self):
        assert _sleeve("off").target_for("XRP", 1.0) == 1

    def test_unknown_mode_is_shadow(self):
        assert _sleeve("nonsense")._contract_floor_mode == "shadow"

    def test_above_one_contract_untouched(self):
        s = _sleeve("enforce", budget=(0.15, 10_000.0, 241.5))   # 6.2 ct
        assert s.target_for("ETH", 1.0) == 6

    def test_flat_signal_still_zero(self):
        assert _sleeve("enforce").target_for("XRP", 0.0) == 0

    def test_object_new_sleeve_defaults_to_historical(self):
        """P85: fixtures/scripts build via object.__new__ — a missing attr
        must read as shadow (1 contract), never raise."""
        s = object.__new__(CoinbaseSleeve)
        s._sizing_inputs = lambda a: (0.01, 10_000.0, 648.0)
        s.signed_contracts = lambda a: 0.0
        assert s.target_for("XRP", 1.0) == 1


class TestFeeAfterRounding:
    def test_one_contract_eth_fee_bps_from_schedule(self):
        """Smallest-notional trio contract (Phase 0 product_specs: ETH 0.1 x
        ~$2,415): 1-contract fee_bps comes out of the venue schedule with the
        fixed term, and is the same per contract at any count."""
        from core.cde_fees import load_venue_fee_schedule
        s = load_venue_fee_schedule(str(REPO / "configs" / "cost" /
                                        "venue_fee_schedule_coinbase.json"), reload=True)
        bps = s.fee_bps("ETH", 1, 2415.5, False)
        assert bps == pytest.approx(10.0 + 0.11 / 241.55 * 1e4, abs=1e-6)
        assert bps == pytest.approx(s.fee_bps("ETH", 4, 2415.5, False))


class TestWiring:
    def test_config_default_shadow_and_passed_to_sleeve(self):
        import main
        assert main.ProductionConfig().contract_floor_sizing_mode == "shadow"
        src = (REPO / "main.py").read_text(encoding="utf-8")
        assert 'contract_floor_mode=str(getattr(\n                    self.config, "contract_floor_sizing_mode", "shadow")' in src

    def test_live_profile_does_not_arm_enforce(self):
        live = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        assert live.get("contract_floor_sizing_mode", "shadow") != "enforce"

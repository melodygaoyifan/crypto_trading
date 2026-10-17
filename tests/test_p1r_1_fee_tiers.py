"""[P1R-1] Tiered venue fee schedule: tier from the rolling 30d volume of the
sleeve's own fill ledger, per-asset fixed components, fail toward the
dearest tier, and the fraction fallbacks no longer carry the refuted 0/3bps."""
import json
from pathlib import Path

import pytest

from core import cde_fees
from core.cde_fees import (load_venue_fee_schedule,
                           rolling_30d_volume_usd_from_ledger, venue_fee_bps)

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "cost" / "venue_fee_schedule_coinbase.json"


def _sched():
    s = load_venue_fee_schedule(str(CFG), reload=True)
    assert s is not None and s.tiers, "the shipped schedule must carry a tier table"
    return s


class TestTiers:
    def test_tiers_ascend_and_get_cheaper(self):
        t = _sched().tiers
        assert [x[0] for x in t] == sorted(x[0] for x in t)
        assert t[0][0] == 0.0
        assert all(t[i][2] >= t[i + 1][2] for i in range(len(t) - 1))
        assert all(m <= k for _, m, k in t)

    def test_tier_zero_is_what_the_venue_reports_now(self):
        ft = json.load(open(REPO / "data" / "phase0" / "fee_tier.json", encoding="utf-8")) \
            if (REPO / "data" / "phase0" / "fee_tier.json").exists() else None
        if ft is None:
            pytest.skip("data/phase0/fee_tier.json absent")
        t0 = _sched().tiers[0]
        assert t0[2] == pytest.approx(float(ft["fee_tier"]["taker_fee_rate"]) * 1e4)
        assert t0[1] == pytest.approx(float(ft["fee_tier"]["maker_fee_rate"]) * 1e4)

    @pytest.mark.parametrize("vol", [None, float("nan"), -1.0])
    def test_unknown_volume_prices_the_dearest_tier(self, vol):
        s = _sched()
        assert s.rate_bps(False, vol) == max(t[2] for t in s.tiers)
        assert s.rate_bps(True, vol) == max(t[1] for t in s.tiers)

    def test_volume_selects_the_tier(self):
        s = _sched()
        for lo, mk, tk in s.tiers:
            assert s.rate_bps(False, lo) == tk
            assert s.rate_bps(True, lo + 1.0) == mk

    def test_aug_fill_reproduced_at_its_tier(self):
        """An Aug-2026 maker fill (rolling 30d vol ~67k): pct 6.5bps + fixed."""
        s = _sched()
        fee = s.fee_usd("BTC", 1, 64000.0, True, 67000.0)
        assert fee == pytest.approx(640.0 * 6.5e-4 + s.fixed_usd_per_contract("BTC"))

    def test_fixed_components_sum(self):
        s = _sched()
        c = s.fixed_components_by_asset["BTC"]
        assert set(c) == {"venue", "clearing", "regulatory"}
        assert s.fixed_usd_per_contract("BTC") == pytest.approx(sum(c.values()))


class TestLedgerVolume:
    def _ledger(self, tmp_path, rows):
        p = tmp_path / "fill_quality.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return str(p)

    def test_only_filled_rows_inside_30d_count(self, tmp_path):
        now = 2_000_000_000.0
        p = self._ledger(tmp_path, [
            {"ts": now - 86400, "asset": "BTC", "status": "filled", "contracts": 2, "fill_avg_price": 80000},
            {"ts": now - 86400, "asset": "ETH", "status": "unresolved", "contracts": 6, "fill_avg_price": None},
            {"ts": now - 40 * 86400, "asset": "BTC", "status": "filled", "contracts": 5, "fill_avg_price": 80000},
            "not json",
        ])
        v = rolling_30d_volume_usd_from_ledger(p, now=now)
        assert v == pytest.approx(2 * 0.01 * 80000)

    def test_missing_ledger_is_none_not_zero(self, tmp_path):
        assert rolling_30d_volume_usd_from_ledger(str(tmp_path / "nope.jsonl"), now=1.0) is None

    def test_live_bps_uses_the_ledger_tier(self, monkeypatch):
        s = _sched()
        monkeypatch.setattr(cde_fees, "rolling_30d_volume_usd_from_ledger", lambda *a, **k: 1e9)
        q = venue_fee_bps("BTC", 80000.0, is_maker=False)
        assert q[0] == pytest.approx(s.tiers[-1][2] + s.fixed_usd_per_contract("BTC") / 800.0 * 1e4)
        monkeypatch.setattr(cde_fees, "rolling_30d_volume_usd_from_ledger", lambda *a, **k: None)
        q2 = venue_fee_bps("BTC", 80000.0, is_maker=False)
        assert q2[0] > q[0]                      # unreadable volume -> dearer, never cheaper


class TestFallbackModels:
    def test_paper_fee_std_is_frozen_because_training_reads_it(self):
        """BLOCKED by iron rule 1: training/train_drl_full.py uses
        VENUE_FEE_STD as the DRL trainer's fee source, so it stays at the
        refuted 0/3bps until training/ may change. Pinned so a future edit
        is a deliberate training decision, not a side effect."""
        from core.paper_fee_service import VENUE_FEE_STD
        assert VENUE_FEE_STD["coinbase"] == {"maker": 0.0000, "taker": 0.0003}
        src = open("training/train_drl_full.py", encoding="utf-8").read()
        assert "from core.paper_fee_service import VENUE_FEE_STD" in src

    def test_execution_service_constants_off_the_refuted_model(self):
        from core.execution_service import _COINBASE_MAKER_BPS, _COINBASE_TAKER_BPS
        assert _COINBASE_MAKER_BPS > 0.0 and _COINBASE_TAKER_BPS > 3.0


class TestProducer:
    def test_refuses_without_fills(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evfs3", REPO / "scripts" / "export_venue_fee_schedule.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        params = REPO / "reports" / "phase0_params.json"
        if not params.exists():
            pytest.skip("reports/phase0_params.json absent")
        assert m.main(["--params", str(params), "--fills", str(tmp_path / "nope.parquet"),
                       "--out", str(tmp_path / "o.json"),
                       "--friction-out", str(tmp_path / "f.json")]) == 2
        assert not (tmp_path / "o.json").exists()

    def test_config_tiers_are_what_the_producer_derives(self):
        fills = REPO / "data" / "phase0" / "fills_raw.parquet"
        ft = REPO / "data" / "phase0" / "fee_tier.json"
        if not (fills.exists() and ft.exists()):
            pytest.skip("Phase 0 fills absent (operator-local)")
        pd = pytest.importorskip("pandas")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evfs4", REPO / "scripts" / "export_venue_fee_schedule.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        want = m.build_tiers(pd.read_parquet(fills), json.load(open(ft, encoding="utf-8"))["fee_tier"])
        got = json.load(open(CFG, encoding="utf-8"))
        assert [(t["min_30d_volume_usd"], t["maker_bps"], t["taker_bps"]) for t in got["tiers"]] == \
               [(t["min_30d_volume_usd"], t["maker_bps"], t["taker_bps"]) for t in want["tiers"]]
        assert got["fixed_components_by_asset"] == want["fixed_components_by_asset"]

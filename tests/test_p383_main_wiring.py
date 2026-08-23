"""[P383] The six deferred P382 items, main.py / integration side.

 1. Integrity shield: an UNFED shield is INERT, not healthy. `is_fed()` is
    False until an orderbook update arrives; the P0 block is gated on it;
    the diag reports `fed`; the WIRE-SHIELD key is None (unknown), not True.
 2. Deadlock friction: the patience manager / deadlock resolver compare the
    seat's ROUND-TRIP edge against the alpha gate's own round-trip
    friction (`friction_total_bps`), never the one-leg fee-less
    `estimated_friction_bps`, and never a fabricated 15.0 when a real
    number exists.
 3. BullTransitionDetector: main.py evaluates ONE DETECTOR PER ASSET and
    persists/restores the per-asset map.
 4. Dashboard export: `equity` is COMBINED (same denomination as
    `peak_equity`), with parts + validity + the sleeve book exported beside it.
"""
from __future__ import annotations

import inspect
import re
import types

import pytest

import main as m
from integration.integration_v36 import _deadlock_friction_bps


# ---------------------------------------------------------------------------
# 1. integrity shield
# ---------------------------------------------------------------------------
class TestIntegrityShieldIsInertUntilFed:
    def test_a_fresh_shield_is_not_fed_but_reads_healthy(self):
        # the vacuity that made the P0 check pass forever
        from defense.kraken_integrity_shield import KrakenIntegrityShield
        s = KrakenIntegrityShield(symbols=["XBT/USD"])
        assert s.is_fed() is False
        assert s.is_healthy() is True, (
            "is_healthy() on zero data is the reason the check was vacuous; "
            "if this ever flips, the is_fed() gate becomes redundant — revisit")

    def test_p0_block_is_gated_on_fed(self):
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        i = src.index("KRAKEN INTEGRITY SHIELD")
        j = src.index("TASK 3: ENHANCED REGIME NAVIGATOR", i)
        blk = src[i:j]   # the whole P0 shield block (it grew with the P384 feed)
        assert "if self.integrity_shield and _shield_fed:" in blk
        assert "integrity check is INERT" in blk

    def test_diag_reports_fed_and_never_a_constant_healthy(self):
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        i = src.index("_diag_record('integrity_shield'")
        blk = src[i:i + 900]
        assert "'fed':" in blk
        assert "is_fed" in blk

    def test_wire_shield_key_is_unknown_not_true_when_unfed(self):
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        i = src.index("[WIRE-SHIELD] Kraken Integrity Shield")
        blk = src[i:i + 2500]   # widened: the P384 primary-shield note sits above the code
        assert "else None" in blk
        assert "if _shield_healthy is False:" in blk, (
            "`if not _shield_healthy` would log UNHEALTHY on the None (unfed) "
            "reading")


# ---------------------------------------------------------------------------
# 2. deadlock friction
# ---------------------------------------------------------------------------
class TestDeadlockFrictionIsTheGatesRoundTrip:
    def test_gate_friction_wins_when_present(self):
        ar = types.SimpleNamespace(friction_total_bps=41.0)
        assert _deadlock_friction_bps(ar, {"estimated_friction_bps": 4.0}) == 41.0

    def test_without_a_gate_result_the_one_leg_figure_is_doubled(self):
        assert _deadlock_friction_bps(None, {"estimated_friction_bps": 4.0}) == 8.0
        ar = types.SimpleNamespace(friction_total_bps=0.0)
        assert _deadlock_friction_bps(ar, {"estimated_friction_bps": 4.0}) == 8.0

    def test_no_number_at_all_falls_back_to_the_old_constant(self):
        assert _deadlock_friction_bps(None, {}) == 15.0
        assert _deadlock_friction_bps(None, {"estimated_friction_bps": "x"}) == 15.0

    def test_every_deadlock_site_uses_the_helper(self):
        from pathlib import Path
        src = Path(m.__file__).resolve().parent.joinpath(
            "integration", "integration_v36.py").read_text(encoding="utf-8")
        assert 'market_data.get("estimated_friction_bps", 15.0)' not in src, (
            "a deadlock site still reads the one-leg fee-less friction")
        assert src.count("_deadlock_friction_bps(self._last_alpha_result, market_data)") >= 3


# ---------------------------------------------------------------------------
# 3. bull transition per asset
# ---------------------------------------------------------------------------
class TestBullDetectorPerAsset:
    def test_main_evaluates_a_per_asset_detector(self):
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        assert "self._bull_detector_for(asset).evaluate(" in src
        assert "self._bull_detector.evaluate(" not in src

    def test_per_asset_instances_are_distinct(self):
        pytest.importorskip("risk.bull_transition_detector")
        from risk import bull_transition_detector as bt
        if not hasattr(bt, "all_bull_transition_states"):
            pytest.skip("per-asset registry not landed yet")
        runner = types.SimpleNamespace(_bull_detector=None)
        a = m.HMATSProductionRunner._bull_detector_for(runner, "BTC")
        b = m.HMATSProductionRunner._bull_detector_for(runner, "ETH")
        assert a is not None and b is not None and a is not b
        assert m.HMATSProductionRunner._bull_detector_for(runner, "BTC") is a

    def test_persist_and_restore_use_the_registry(self):
        src = inspect.getsource(m.HMATSProductionRunner._save_paper_positions)
        assert "self._bull_transition_states_payload()" in src
        rsrc = inspect.getsource(m.HMATSProductionRunner._load_paper_positions)
        assert "restore_bull_transition_states(bt_data)" in rsrc


# ---------------------------------------------------------------------------
# 4. dashboard export
# ---------------------------------------------------------------------------
class TestDashboardExportIsCombined:
    def test_export_emits_parts_validity_and_sleeve_book(self):
        src = inspect.getsource(m.HMATSProductionRunner._export_dashboard_state)
        for key in ('"kraken_equity"', '"sleeve_equity"', '"equity_valid"',
                    '"equity_basis"', '"sleeve_positions"', '"sleeve_reconcile_ok"'):
            assert key in src, key
        # equity is the SUM when both halves are readable
        assert "equity = _kr_equity + _sl_equity" in src
        # an unreadable sleeve half falls back to the last known COMBINED
        assert '_equity_basis = "combined"        # last known combined (P261)' in src
        # the old Kraken-only assignment is gone
        assert "equity = self.account_sync.get_equity()" not in src


class TestDashboardExportBehaviour:
    """Drive the real `_export_dashboard_state` (the fixture pattern from
    tests/test_dashboard_state_incremental_export.py) with a fake Kraken
    account and a fake sleeve, and read the JSON back."""

    def _runner(self, tmp_path, kraken=0.40, sleeve=10_900.0, sleeve_ok=True,
                held=None):
        from main import HMATSProductionRunner, RunMode
        r = HMATSProductionRunner.__new__(HMATSProductionRunner)
        r.config = types.SimpleNamespace(initial_capital=10_000.0,
                                         risk_profile="HIGH_RISK",
                                         mode=RunMode.LIVE)
        r.account_sync = (types.SimpleNamespace(get_equity=lambda: kraken,
                                                dry_run=False)
                          if kraken is not None else None)
        r._paper_positions = {}
        r._position_entry_times = {}
        r._dashboard_asset_snapshot = {}
        r._dashboard_asset_runtime = {}
        r._DASHBOARD_STATE_FILE = tmp_path / "dashboard_state.json"
        r._STEP15_STATUS_FILE = tmp_path / "step15_status.json"
        r._PAPER_RUN_PID_FILE = tmp_path / "paper_run.pid"
        r._KILL_SWITCH_TEST_FILE = tmp_path / "kill_switch_test.json"
        r._drl_models_ready = 0
        r._drl_ensembles = {}
        r._drl_bootstrap_applied = False
        r._peak_equity = 10_946.75
        if held is not None:
            r._p0_last_combined_equity = held
        if sleeve is not None:
            r._coinbase_sleeve = types.SimpleNamespace(
                _reconcile_ok=sleeve_ok,
                sleeve_equity_usd=lambda: sleeve,
                sleeve_equity_age_sec=lambda: 30.0,
                _last_positions={"ETH": {"signed_contracts": 4.0,
                                         "entry_vwap": 2376.0,
                                         "current_price": 2380.0}})
        return r

    def _read(self, r):
        import json
        r._export_dashboard_state(1, 1, {"ETH": {"price": 2380.0}})
        return json.loads(r._DASHBOARD_STATE_FILE.read_text(encoding="utf-8"))

    def test_equity_is_the_combined_sum_with_parts(self, tmp_path):
        st = self._read(self._runner(tmp_path))
        assert st["equity"] == pytest.approx(10_900.40)
        assert st["kraken_equity"] == pytest.approx(0.40)
        assert st["sleeve_equity"] == pytest.approx(10_900.0)
        assert st["equity_basis"] == "combined" and st["equity_valid"] is True
        assert st["sleeve_positions"] == [{"asset": "ETH", "venue": "coinbase",
                                          "signed_contracts": 4.0,
                                          "entry_vwap": 2376.0,
                                          "current_price": 2380.0}]
        assert st["sleeve_reconcile_ok"] is True
        # the old defect: equity ~0.40 beside a ~10.9k peak
        assert st["equity"] > 0.5 * st["peak_equity"]

    def test_unreadable_sleeve_half_serves_the_last_known_combined(self, tmp_path):
        r = self._runner(tmp_path, sleeve=None, held=10_865.13)
        st = self._read(r)
        assert st["equity"] == pytest.approx(10_865.13)
        assert st["equity_basis"] == "combined"
        assert st["sleeve_equity"] is None and st["equity_valid"] is False
        assert st["sleeve_positions"] == []

    def test_first_ever_boot_is_labelled_kraken_only(self, tmp_path):
        st = self._read(self._runner(tmp_path, sleeve=None, held=None))
        assert st["equity_basis"] == "kraken_only"
        assert st["equity"] == pytest.approx(0.40)

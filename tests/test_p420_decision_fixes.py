"""[P420] Decision-path fixes from the 2026-08-27 read-through (fork 1 of 5:
main.py / defense/constitution.py / core/seat_alpha.py).

  1. breadth CDE spreads MEASURED (XRP 4.0 / BNB 8.0) and the unmeasured
     coinbase fallback made EXPENSIVE (10.0, P167) with a once-per-asset log;
  2. `_compute_crack_weight` alignment filtered to the home trio + clamped
     (5-asset stash x /3.0 cleared the 0.50 bar on alignment alone);
  3. `dvol_zscore` finally carries the honest DVOL z (the [P0-FIX] funding
     proxy moved to `funding_abs_zscore`; the P306 block publishes both keys
     when fresh) -- EXTREME_DVOL is REACHABLE for the first time;
  4. the 30s watchdog hands the stop follow-up the SNAPSHOT on a refused
     exit (intent only on OK/NOOP, the 4H driver's rule);
  5. no-order driver branches sweep stale resting ENTRY orders
     (getattr-defended, fork-2 contract);
  6. a calibrated seat's REFUSAL fallback can never assert more than its
     measurement (BTC 24.1, not 30);
  7. external capital flows are netted out of EVERY equity anchor
     (peak / daily / p0-held / SOTA peak), persisted ref, migration-safe;
  8. [CONFIDENCE_GATE] / [AUTO_RECOVERY_LATCH] are ENTRY-QUALITY, and their
     write sites read the SLEEVE book;
  9. whale confidence is 0.0 when the P352 evidence gate fails (n=1 must not
     vote at full confidence through the P417 weight), both dicts;
 10. `_cde_quote_map` prefixes derived from SYMBOL_MAP (+XRP/BNB);
 11. the skew seat fetches only for a decide asset;
 12. the trend_assets exclusion log is INFO for breadth on the default trio;
 13. the realtime correlation controller is fed only the assets it knows.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import main as m  # noqa: E402
from tests._source_scan import code_only  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _src(rel):
    return code_only(REPO / rel, strip_docstrings=True)


# ---------------------------------------------------------------------------
# 1. breadth spreads + expensive unmeasured fallback
# ---------------------------------------------------------------------------
class TestBreadthSpreads:
    def _fc(self):
        from defense.constitution import FrictionComponents
        return FrictionComponents()

    @pytest.mark.parametrize("asset,expected", [
        ("XRP", 4.0), ("BNB", 8.0),
        ("BTC", 2.0), ("ETH", 5.5), ("SOL", 4.0),   # home trio unchanged
    ])
    def test_measured_cde_spreads(self, asset, expected):
        f = self._fc()
        f.set_spread_venue(asset, "coinbase")
        f.update_for_asset(asset)
        assert f.slippage_bps == expected

    def test_rounded_UP_from_the_probe_medians(self):
        # XRP median 2.81 / BNB 5.66 (max 7.08): the table may only sit ABOVE
        from defense.constitution import CDE_SPREAD_BPS_MEASURED
        assert CDE_SPREAD_BPS_MEASURED["XRP"] >= 2.81
        assert CDE_SPREAD_BPS_MEASURED["BNB"] >= 7.08

    def test_unmeasured_asset_on_coinbase_is_priced_EXPENSIVE(self, caplog):
        from defense.constitution import CDE_SPREAD_BPS_UNMEASURED_FALLBACK
        assert CDE_SPREAD_BPS_UNMEASURED_FALLBACK == 10.0
        f = self._fc()
        f.set_spread_venue("ADA", "coinbase")
        with caplog.at_level(logging.WARNING):
            f.update_for_asset("ADA")
            f.update_for_asset("ADA")
        assert f.slippage_bps == 10.0
        warns = [r for r in caplog.records if "[P420][SPREAD] ADA" in r.getMessage()]
        assert len(warns) == 1, "the unmeasured-fallback warning must fire ONCE per asset"

    def test_unmeasured_fallback_is_above_every_measured_entry(self):
        from defense.constitution import (CDE_SPREAD_BPS_MEASURED,
                                          CDE_SPREAD_BPS_UNMEASURED_FALLBACK)
        assert CDE_SPREAD_BPS_UNMEASURED_FALLBACK > max(CDE_SPREAD_BPS_MEASURED.values())

    def test_kraken_path_is_untouched(self):
        f = self._fc()
        f.update_for_asset("ADA")           # no venue memory -> Kraken table
        assert f.slippage_bps == 5.0
        f.update_for_asset("SOL")
        assert f.slippage_bps == 10.0       # the Kraken-era SOL constant

    def test_stamp_moved_with_the_measurement(self):
        from defense import constitution as c
        assert c._MEASURED_ON == "2026-08-27"
        assert "P420" in c._MEASURED_BY


# ---------------------------------------------------------------------------
# 2. crack weight alignment
# ---------------------------------------------------------------------------
class TestCrackAlignment:
    FIVE = {"BTC": 0.9, "ETH": 0.8, "SOL": 0.7, "XRP": 1.0, "BNB": 1.0}

    def test_five_aligned_assets_clamp_to_one(self):
        assert m.crack_alignment_strength(self.FIVE) == pytest.approx(1.0)

    def test_the_clamp_holds_even_if_the_roster_is_widened(self):
        """The clamp is independent of the trio filter: a future roster
        widening must still cap the component at its documented weight."""
        wide = ("BTC", "ETH", "SOL", "XRP", "BNB")
        assert m.crack_alignment_strength(self.FIVE, home=wide) == pytest.approx(1.0)

    def test_breadth_alone_does_not_count(self):
        assert m.crack_alignment_strength({"XRP": 1.0, "BNB": 1.0, "BTC": 0.0}) == 0.0

    def test_two_of_three_is_two_thirds(self):
        assert m.crack_alignment_strength({"BTC": 0.9, "ETH": 0.8, "SOL": 0.0,
                                           "XRP": 1.0}) == pytest.approx(2 / 3)

    def test_disagreement_is_zero(self):
        assert m.crack_alignment_strength({"BTC": 0.9, "ETH": -0.8, "SOL": 0.7}) == 0.0

    def test_the_real_method_cannot_activate_on_alignment_alone(self):
        """The live defect: 0.40 x 4/3 = 0.533 > 0.50 with no kinetic/volume."""
        fake = types.SimpleNamespace(_last_quant_directions=dict(self.FIVE))
        md = {"structure_break_pct": 0.0, "volume_ratio": 1.0, "bar_progress": 1.0}
        w = m.HMATSProductionRunner._compute_crack_weight(fake, "BTC", md)
        assert w == pytest.approx(0.40)
        assert w <= 0.50


# ---------------------------------------------------------------------------
# 3. dvol_zscore
# ---------------------------------------------------------------------------
class TestDvolZscore:
    def test_fresh_armed_publishes_both_keys(self):
        assert m.dvol_publish_values(True, -1.4) == {"dvol": -1.4, "dvol_zscore": -1.4}

    @pytest.mark.parametrize("armed,z", [
        (True, None), (False, 2.0), (True, float("nan")), (True, "x"),
    ])
    def test_stale_or_disarmed_publishes_nothing(self, armed, z):
        assert m.dvol_publish_values(armed, z) == {}

    def test_p0_fix_no_longer_writes_dvol_zscore(self):
        src = _src("main.py")
        i = src.index("_fr_asset = asset.replace(")
        blk = src[i:i + 1600]                      # the [P0-FIX] funding block
        assert 'market_data["dvol_zscore"]' not in blk, (
            "the funding |z| proxy landed in dvol_zscore again -- it overwrites "
            "the key EXTREME_DVOL / VOLATILITY_EXPANSION / PATCH-4 read")
        assert 'market_data["funding_abs_zscore"]' in blk
        assert m.FUNDING_ABS_ZSCORE_KEY == "funding_abs_zscore"
        # the ONLY dvol_zscore writer left in main.py is the P306 publish
        writes = [mm.start() for mm in re.finditer(
            r'market_data\["dvol_zscore"\]\s*=', src)]
        assert len(writes) == 1
        assert "if _dv_pub:" in src[writes[0] - 900:writes[0]]

    def test_p306_block_publishes_through_the_pure_function(self):
        src = _src("main.py")
        i = src.index("get_dvol_history()")
        blk = src[i:i + 2800]
        assert "_dv_pub = dvol_publish_values(_dv_armed, _dvz)" in blk
        assert 'market_data["dvol"] = float(_dvz)' in blk
        assert 'market_data["dvol_zscore"] = float(_dvz)' in blk
        # both writes sit under the pure-function gate
        i2 = blk.index("if _dv_pub:")
        assert i2 < blk.index('market_data["dvol_zscore"] = float(_dvz)')

    def test_extreme_dvol_is_reachable_from_the_published_key(self):
        """End to end through the real constitution consumer."""
        from defense.constitution import NoTradeTriggerChecker, NoTradeTriggerType
        chk = NoTradeTriggerChecker()
        md = {"dvol_zscore": 0.0, "current_price": 100.0}
        md.update(m.dvol_publish_values(True, 5.5))
        res = chk.compute_triggers(md, {})
        active = {c.trigger_type for c in res.active_conditions}
        assert NoTradeTriggerType.EXTREME_DVOL in active
        md2 = {"dvol_zscore": 0.0, "current_price": 100.0}
        md2.update(m.dvol_publish_values(True, None))   # not fresh -> 0.0 stands
        res2 = chk.compute_triggers(md2, {})
        assert NoTradeTriggerType.EXTREME_DVOL not in {
            c.trigger_type for c in res2.active_conditions}


# ---------------------------------------------------------------------------
# 4. watchdog follow-up target
# ---------------------------------------------------------------------------
class _WatchdogSleeve:
    def __init__(self, status, contracts=3):
        self.status = status
        self._contracts = contracts
        self._reconcile_ok = True
        self.followups = []

    def reconcile_positions(self):
        return {}

    def signed_contracts(self, asset):
        return self._contracts

    async def execute_target(self, asset, target, urgent=False):
        return {"status": self.status}

    def request_stop_followup(self, asset, intended):
        self.followups.append(float(intended))


class TestWatchdogFollowupTarget:
    @pytest.mark.parametrize("status,expected", [
        ("OK", 0.0), ("NOOP", 0.0),
        ("BLOCKED", 3.0), ("FAILED", 3.0), ("ERROR", 3.0),
        ("SKIPPED_STALE", 3.0), ("NOT_READY", 3.0), (None, 3.0),
    ])
    def test_truth_table(self, status, expected):
        assert m.watchdog_stop_followup_target(status, 0.0, 3.0) == expected

    def test_reduce_keeps_intent_only_on_ok(self):
        assert m.watchdog_stop_followup_target("OK", 1.0, 3.0) == 1.0
        assert m.watchdog_stop_followup_target("FAILED", 1.0, 3.0) == 3.0

    @pytest.mark.parametrize("status,expected", [
        ("OK", 0.0), ("FAILED", 3.0), ("BLOCKED", 3.0)])
    def test_the_real_helper_hands_the_followup_the_right_target(self, status, expected):
        s = _WatchdogSleeve(status, contracts=3)
        _run(m.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert s.followups == [expected], (
            "a refused exit must re-arm the stop on the SNAPSHOT, not on 0 "
            "(the stop was swept before the attempt)")

    def test_reduce_path_too(self):
        s = _WatchdogSleeve("FAILED", contracts=4)
        _run(m.sleeve_fast_risk_action(s, "ETH", "REDUCE_50", True))
        assert s.followups == [4.0]
        s2 = _WatchdogSleeve("OK", contracts=4)
        _run(m.sleeve_fast_risk_action(s2, "ETH", "REDUCE_50", True))
        assert s2.followups == [2.0]


# ---------------------------------------------------------------------------
# 5. stale-entry sweep on no-order branches
# ---------------------------------------------------------------------------
class TestSweepStaleEntries:
    def test_absent_method_is_skipped_silently(self):
        assert _run(m.sleeve_sweep_stale_entries(types.SimpleNamespace(), "BTC")) is None
        assert _run(m.sleeve_sweep_stale_entries(None, "BTC")) is None

    def test_async_contract_is_awaited(self):
        calls = []

        async def sweep(asset):
            calls.append(asset)
            return 2
        s = types.SimpleNamespace(sweep_stale_entries=sweep)
        assert _run(m.sleeve_sweep_stale_entries(s, "ETH")) == 2
        assert calls == ["ETH"]

    def test_sync_implementation_also_works(self):
        s = types.SimpleNamespace(sweep_stale_entries=lambda a: 1)
        assert _run(m.sleeve_sweep_stale_entries(s, "ETH")) == 1

    def test_a_raising_sweep_is_logged_and_never_propagates(self, caplog):
        async def boom(asset):
            raise RuntimeError("venue 502")
        s = types.SimpleNamespace(sweep_stale_entries=boom)
        with caplog.at_level(logging.WARNING):
            assert _run(m.sleeve_sweep_stale_entries(s, "SOL")) is None
        assert any("[P420][SWEEP] SOL" in r.getMessage() for r in caplog.records)

    def test_every_no_order_branch_sweeps_before_the_stop_reconcile(self):
        import inspect
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        # HOLD + MA block_entry + whale block_entry + cooldown = 4 branches
        hits = [mm.start() for mm in re.finditer(
            r"await sleeve_sweep_stale_entries\(_sl, _m_a\)", src)]
        assert len(hits) >= 4, f"only {len(hits)} no-order branches sweep"
        for h in hits:
            after = src[h:h + 400]
            assert "ensure_protective_stop(" in after, (
                "the sweep must sit immediately BEFORE the stop reconcile")


# ---------------------------------------------------------------------------
# 6. refusal fallback capped at the measurement
# ---------------------------------------------------------------------------
class TestSeatAlphaRefusalCap:
    def test_btc_refusal_asserts_the_measurement_not_30(self):
        from core.seat_alpha import resolve_seat_edge
        # price=None -> the P321b interlock REFUSES (still refuses: != 24.1
        # would only be true if the calibrated path had applied... it IS
        # 26.0 [P420b] either way for BTC, so pin the refusal via a bad price too)
        assert resolve_seat_edge("BTC", "regimebook", 1.0, 30.0, True, True,
                                 None) == pytest.approx(26.0)
        assert resolve_seat_edge("BTC", "regimebook", 1.0, 30.0, True, True,
                                 float("nan")) == pytest.approx(26.0)

    def test_eth_refusal_keeps_30_because_its_measurement_is_larger(self):
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True, True,
                                 None) == pytest.approx(30.0)

    def test_unknown_asset_refusal_unchanged(self):
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("DOGE", "regimebook", 1.0, 30.0, True, True,
                                 None) == pytest.approx(30.0)

    def test_the_cap_scales_with_direction_and_flat_asserts_nothing(self):
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("BTC", "regimebook", 0.5, 30.0, True, True,
                                 None) == pytest.approx(13.0)  # [P420b] 26.0 x 0.5
        assert resolve_seat_edge("BTC", "regimebook", 0.0, 30.0, True, True,
                                 None) == 0.0

    def test_flags_off_path_is_untouched(self):
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("BTC", "regimebook", 1.0, 30.0, False, False,
                                 69280.0) == pytest.approx(30.0)

    def test_refusal_still_refuses_the_calibrated_value(self):
        """The magnitude changed; the REFUSAL did not (P321b)."""
        from core.seat_alpha import resolve_seat_edge
        assert resolve_seat_edge("ETH", "regimebook", 1.0, 30.0, True, True,
                                 None) != pytest.approx(88.1)

    def test_calibration_cap_lookup(self):
        from core.seat_alpha import calibration_cap_bps
        assert calibration_cap_bps("BTC", "regimebook") == pytest.approx(26.0)
        assert calibration_cap_bps("BTC", "skew_contra") == pytest.approx(680.0)
        assert calibration_cap_bps("BTC", "whale") is None
        assert calibration_cap_bps("DOGE", "regimebook") is None


# ---------------------------------------------------------------------------
# 7. external-flow re-anchoring
# ---------------------------------------------------------------------------
class _FlowSleeve:
    def __init__(self, flow, ok=True):
        self._external_flow_usd = flow
        self._reconcile_ok = ok


def _runner(flow_ref, sleeve, peak=10_000.0, daily=10_000.0, held=10_000.0,
            sota_peak=10_000.0):
    r = types.SimpleNamespace()
    r._coinbase_sleeve = sleeve
    r._p0_flow_ref = flow_ref
    r._peak_equity = peak
    r._daily_pnl_anchor = daily
    r._p0_last_combined_equity = held
    r.p0_integrator = types.SimpleNamespace(
        risk_controller=types.SimpleNamespace(peak_equity=sota_peak))
    return r


class TestFlowReanchor:
    @pytest.mark.parametrize("cur,ref,expected", [
        (0.0, 0.0, 0.0), (7000.0, 0.0, 7000.0), (4000.0, 7000.0, -3000.0),
        (0.005, 0.0, 0.0),                       # float drift is not a flow
        (None, 0.0, None), ("x", 0.0, None), (float("nan"), 0.0, None),
    ])
    def test_delta_truth_table(self, cur, ref, expected):
        assert m.flow_reanchor_delta(cur, ref) == expected

    def test_a_withdrawal_does_not_move_the_drawdown(self):
        # $10k peak, $3k withdrawn: equity is now $7k with NO trading loss
        r = _runner(0.0, _FlowSleeve(-3000.0))
        d = m.HMATSProductionRunner._reanchor_on_external_flow(r)
        assert d == -3000.0
        equity_now = 7000.0
        dd = (r._peak_equity - equity_now) / r._peak_equity
        assert dd == pytest.approx(0.0)
        assert r._peak_equity == 7000.0
        assert r.p0_integrator.risk_controller.peak_equity == 7000.0
        assert r._p0_last_combined_equity == 7000.0
        assert r._p0_flow_ref == -3000.0

    def test_a_deposit_does_not_move_the_daily_pnl(self):
        r = _runner(0.0, _FlowSleeve(7000.0))
        m.HMATSProductionRunner._reanchor_on_external_flow(r)
        equity_now = 17000.0
        assert equity_now - r._daily_pnl_anchor == pytest.approx(0.0)
        assert r._peak_equity == 17000.0

    def test_idempotent_second_call_shifts_nothing(self):
        r = _runner(0.0, _FlowSleeve(7000.0))
        m.HMATSProductionRunner._reanchor_on_external_flow(r)
        assert m.HMATSProductionRunner._reanchor_on_external_flow(r) == 0.0
        assert r._peak_equity == 17000.0

    def test_unreadable_sleeve_never_shifts(self):
        r = _runner(0.0, _FlowSleeve(-3000.0, ok=False))
        assert m.HMATSProductionRunner._reanchor_on_external_flow(r) == 0.0
        assert r._peak_equity == 10_000.0
        assert r._p0_flow_ref == 0.0

    def test_absent_ref_seeds_without_a_shift(self):
        """Migration: a pre-P420 state file has no ref; the first tick must
        NOT treat the whole cumulative history as a fresh flow."""
        r = _runner(None, _FlowSleeve(7074.28))
        assert m.HMATSProductionRunner._reanchor_on_external_flow(r) == 0.0
        assert r._p0_flow_ref == 7074.28
        assert r._peak_equity == 10_000.0

    def test_no_sleeve_is_a_noop(self):
        r = _runner(0.0, None)
        assert m.HMATSProductionRunner._reanchor_on_external_flow(r) == 0.0

    def test_sota_peak_attribute_contract(self):
        from risk.sota_risk_controller import SOTARiskController
        rc = SOTARiskController()
        assert hasattr(rc, "peak_equity")   # the documented direct write target

    def test_ref_is_persisted_and_restored(self):
        src = _src("main.py")
        assert '"p0_flow_ref": getattr(self, "_p0_flow_ref", None)' in src
        assert 'data.get("p0_flow_ref")' in src

    def test_reanchor_runs_before_the_peak_arithmetic_and_after_flow_detection(self):
        import inspect
        dd = inspect.getsource(m.HMATSProductionRunner._update_drawdown_snapshot)
        assert dd.index("_p418_rea(_sleeve)") < dd.index(
            "self._peak_equity = max(self._peak_equity, current_equity)")
        live = inspect.getsource(m.HMATSProductionRunner.run_live)
        i = live.index("log_pnl_point()")
        assert "_p418_rea(self._coinbase_sleeve)" in live[i:i + 800]


# ---------------------------------------------------------------------------
# 8. entry-only gates -> entry quality; sleeve-aware position read
# ---------------------------------------------------------------------------
class TestEntryOnlyGates:
    @pytest.mark.parametrize("tag", ["[CONFIDENCE_GATE]", "[AUTO_RECOVERY_LATCH]"])
    def test_classified_entry_quality_not_flatten(self, tag):
        assert tag in m._SLEEVE_ENTRY_QUALITY_VETOES
        assert tag not in m._SLEEVE_FLATTEN_INTENDED_VETOES
        assert tag not in m._SLEEVE_HOLD_VETOES

    @pytest.mark.parametrize("tag", ["[CONFIDENCE_GATE]", "[AUTO_RECOVERY_LATCH]"])
    def test_translator_resolves_against_the_book(self, tag):
        it = types.SimpleNamespace(direction=0.0, target_exposure=0.0,
                                   veto_active=True, veto_reason=f"{tag} x")
        d, _ = m.sleeve_direction_from_intent(it, 0.9)
        assert d is m.SLEEVE_ENTRY_BLOCKED
        # held, agreeing -> HOLD; flat -> HOLD; flip -> close the leg (0.0)
        assert m.sleeve_entry_blocked_resolve(1, 0.9, "x")[0] is m.SLEEVE_HOLD
        assert m.sleeve_entry_blocked_resolve(0, 0.9, "x")[0] is m.SLEEVE_HOLD
        assert m.sleeve_entry_blocked_resolve(1, -0.9, "x")[0] == 0.0

    def test_sleeve_position_held_truth_table(self):
        ok = types.SimpleNamespace(_reconcile_ok=True, signed_contracts=lambda a: 2)
        flat = types.SimpleNamespace(_reconcile_ok=True, signed_contracts=lambda a: 0)
        stale = types.SimpleNamespace(_reconcile_ok=False, signed_contracts=lambda a: 2)
        assert m.sleeve_position_held(ok, "ETH", {}) is True
        assert m.sleeve_position_held(flat, "ETH", {"current_exposure": 0.5}) is False
        assert m.sleeve_position_held(stale, "ETH", {}) is None
        assert m.sleeve_position_held(None, "ETH", {"current_exposure": -0.3}) is True
        assert m.sleeve_position_held(None, "ETH", {"current_exposure": 0}) is False
        assert m.sleeve_position_held(None, "ETH", {"current_exposure": "x"}) is None

    def test_both_write_sites_read_the_sleeve_book(self):
        src = _src("main.py")
        i = src.index("_cg_result = self._confidence_gate.check(")
        blk = src[i:i + 1500]
        assert "sleeve_position_held(" in blk
        assert "self._paper_positions.get(asset)" not in blk
        j = src.index("self._auto_recovery_gate.clear_halt(_now_ts)")
        blk2 = src[j:j + 1800]
        assert "sleeve_position_held(" in blk2
        assert '_paper_positions.get(asset, {}).get("exposure", 0)' not in blk2


# ---------------------------------------------------------------------------
# 9. whale confidence after the evidence gate
# ---------------------------------------------------------------------------
class TestWhaleConfidenceEvidence:
    @pytest.mark.parametrize("conf,cnt,mn,d,expected", [
        (1.0, 1, 2, 1.0, 0.0),      # n=1 saturated -> muted
        (1.0, 2, 2, 1.0, 1.0),      # meets the minimum
        (0.6, 5, 2, -1.0, 0.6),
        (1.0, None, 2, 1.0, 1.0),   # unknown keeps the reading (P2)
        (1.0, 0, 2, 1.0, 1.0),      # 0 beside a direction = count absent
        (1.0, "x", 2, 1.0, 1.0),
        (0.0, 1, 2, 0.0, 0.0),      # flat contributes nothing either way
    ])
    def test_truth_table(self, conf, cnt, mn, d, expected):
        assert m.whale_confidence_after_evidence(conf, cnt, mn, d) == expected

    def test_written_to_both_dicts_and_the_stash(self):
        src = _src("main.py")
        i = src.index("_wh_conf_gated = whale_confidence_after_evidence(")
        blk = src[i:i + 1200]
        assert "agent_signals['whale_confidence'] = _wh_conf_gated" in blk
        assert "market_data['whale_confidence'] = _wh_conf_gated" in blk
        assert "self._last_whale_confidences[asset] = float(_wh_conf_gated)" in blk


# ---------------------------------------------------------------------------
# 10. cde quote map prefixes
# ---------------------------------------------------------------------------
class TestCdePrefixMap:
    def test_breadth_prefixes_present(self):
        pm = m.cde_prefix_map()
        assert pm["XP"] == "XRP" and pm["BN"] == "BNB"
        assert pm["BI"] == "BTC" and pm["ET"] == "ETH"
        assert pm["SL"] == "SOL" and pm["SO"] == "SOL"

    def test_derived_from_symbol_map(self):
        from exchange.symbol_mapping import SYMBOL_MAP
        pm = m.cde_prefix_map()
        for asset, pid in SYMBOL_MAP["coinbase"]["perp"].items():
            assert pm[pid[:2]] == asset

    def test_the_quote_map_uses_it(self):
        src = _src("main.py")
        assert "base_map = cde_prefix_map()" in src
        assert '{"BI": "BTC", "ET": "ETH", "SO": "SOL", "SL": "SOL"}' not in src


# ---------------------------------------------------------------------------
# 11. skew seat fetch only for decide assets
# ---------------------------------------------------------------------------
class TestSkewSeatFetchGuard:
    @pytest.mark.parametrize("asset,decide,expected", [
        ("BTC", ["BTC", "ETH"], True), ("ETH", ["BTC", "ETH"], True),
        ("SOL", ["BTC", "ETH"], False), ("XRP", ["BTC", "ETH"], False),
        ("BNB", ["BTC", "ETH"], False), ("BTC", [], False), ("BTC", None, False),
    ])
    def test_truth_table(self, asset, decide, expected):
        assert m.skew_seat_should_fetch(asset, decide) is expected

    def test_the_seat_block_is_guarded(self):
        """A fake signal counting calls, driven through the guard exactly as
        the seat block composes it."""
        calls = []
        sig = types.SimpleNamespace(seat_direction=lambda a: calls.append(a) or (1.0, True))
        decide = ["BTC", "ETH"]
        for asset in ("BTC", "ETH", "SOL", "XRP", "BNB"):
            _ss = (sig.seat_direction(asset)
                   if m.skew_seat_should_fetch(asset, decide) else None)
            assert (_ss is not None) == (asset in decide)
        assert calls == ["BTC", "ETH"]
        src = _src("main.py")
        i = src.index('_skew_decide = getattr(self.config, "skew_seat_assets", None) or []')
        blk = src[i:i + 400]
        assert "if skew_seat_should_fetch(asset, _skew_decide) else None" in blk


# ---------------------------------------------------------------------------
# 12. trend exclusion log level
# ---------------------------------------------------------------------------
class TestTrendExclusionLog:
    @pytest.mark.parametrize("asset,explicit,expected", [
        ("XRP", False, "info"), ("BNB", False, "info"),
        ("SOL", False, "warning"),          # a home asset off the roster
        ("XRP", True, "warning"),           # key explicitly set = tripwire semantics
    ])
    def test_level(self, asset, explicit, expected):
        assert m.trend_exclusion_log_level(asset, explicit) == expected

    def test_the_default_roster_is_untouched_and_the_wording_moved(self):
        src = _src("main.py")
        i = src.index('_trend_assets = getattr(self.config, "trend_assets", None) or [')
        blk = src[i:i + 1600]
        assert '"BTC", "ETH", "SOL"]' in blk[:200]
        assert "EXCLUDED by trend_assets" not in blk
        assert "not in trend_assets (default = " in blk


# ---------------------------------------------------------------------------
# 13. correlation controller fed only known assets
# ---------------------------------------------------------------------------
class TestCorrControllerKnownPrices:
    def test_filters_to_the_controller_roster(self, caplog):
        from risk.correlation_realtime_controller import CorrelationRealtimeController
        ctrl = CorrelationRealtimeController()
        prices = {"BTC": 1.0, "ETH": 2.0, "SOL": 3.0, "XRP": 4.0, "BNB": 5.0}
        known = m.corr_controller_known_prices(ctrl, prices)
        assert set(known) == {"BTC", "ETH", "SOL"}
        with caplog.at_level(logging.WARNING):
            ctrl.update_prices_batch(known)
        assert not any("Unknown asset" in r.getMessage() for r in caplog.records)

    def test_a_controller_without_the_attribute_gets_the_dict_unchanged(self):
        prices = {"BTC": 1.0, "XRP": 4.0}
        assert m.corr_controller_known_prices(object(), prices) == prices

    def test_every_batch_call_site_is_filtered(self):
        src = _src("main.py")
        raw = len(re.findall(r"update_prices_batch\(", src))
        filtered = len(re.findall(
            r"update_prices_batch\(\s*corr_controller_known_prices\(", src))
        assert raw >= 4 and filtered == raw, (
            f"{raw - filtered} update_prices_batch call(s) feed the controller "
            f"an unfiltered dict")

"""[P253] Tests for the 2026-08-09 full-read-through fix batch.

Each class pins one fix from the P253 read-through:
  1. A tick crash / Kraken disconnect must HOLD the sleeve, never flatten it.
  2. The protective-stop reconcile must not trust an intent the venue refused.
  3. execute_target refuses stale snapshots and priceless orders.
  4. reconcile_positions derives sign safely and refuses unknown sides.
  5. pre_tick_update refreshes the stale-data guard only on fresh data,
     and the daily-loss producer arithmetic exists.
  6. SOTARiskController state survives a restart, one-directionally.
  7. The existence fuse persists a full 28d window (not 8 days).
  8. The tripwire never counts "not enough data" as GATE-CLOSED.
  9. The shadow-IC gate's t-stat is overlap-corrected (P231 parity).
 10. Routing failure with the flag ON fails SAFE (routed), not open (Kraken).
 11. Source pins for the offline-tooling fixes (funding shift, export guard).
"""

import asyncio
import inspect
import json
import sys
import io
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Intent:
    """Minimal stand-in carrying only the fields the translator reads."""

    def __init__(self, direction=0.0, target_exposure=0.0, veto_active=False,
                 veto_reason=""):
        self.direction = direction
        self.target_exposure = target_exposure
        self.veto_active = veto_active
        self.veto_reason = veto_reason


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ===========================================================================
# 1. crash / disconnect -> sleeve HOLD
# ===========================================================================

class TestCrashIntentHoldsTheSleeve:
    def test_markers_are_in_the_hold_veto_set(self):
        import main as hm
        assert "TICK_CRASH_HOLD" in hm._SLEEVE_HOLD_VETOES
        assert "EXCHANGE_DISCONNECTED_HOLD" in hm._SLEEVE_HOLD_VETOES

    @pytest.mark.parametrize("reason", [
        "[TICK_FATAL] TICK_CRASH_HOLD - tick crashed; sleeve holds",
        "[TICK] EXCHANGE_DISCONNECTED_HOLD - kraken exec manager not ready; "
        "sleeve holds",
    ])
    def test_crash_shaped_intent_translates_to_HOLD_not_flatten(self, reason):
        import main as hm
        intent = _Intent(veto_active=True, veto_reason=reason)
        tgt, why = hm.sleeve_direction_from_intent(intent, fallback_dir=0.9)
        assert tgt is hm.SLEEVE_HOLD, (
            f"a crash/disconnect intent flattened the sleeve ({why}) — the "
            f"exact P253 finding: state-unknown read as position-unwanted")

    def test_a_BARE_empty_intent_still_reads_as_flatten(self):
        """Pins WHY the return sites must never emit a bare TradeIntentV36():
        the translator itself deliberately keeps zero_target_exposure ->
        flatten (that branch is load-bearing for the fuse's close encoding,
        P206 rule 4). The fix lives at the RETURN SITES."""
        import main as hm
        intent = _Intent()  # veto_active=False, target_exposure=0.0
        tgt, why = hm.sleeve_direction_from_intent(intent, 0.9)
        assert tgt == 0.0 and why == "zero_target_exposure"

    def test_the_wrapper_has_no_bare_empty_intent_returns_left(self):
        """process_4h_tick (the crash/disconnect wrapper) must return MARKED
        intents only. A bare `return TradeIntentV36()` reappearing here is
        the regression."""
        import main as hm
        src = inspect.getsource(hm.HMATSProductionRunner.process_4h_tick)
        assert "return TradeIntentV36()" not in src, (
            "a bare empty-intent return is back in process_4h_tick — it reads "
            "to the sleeve as FLATTEN (P253 finding 1)")
        assert "TICK_CRASH_HOLD" in src
        assert "EXCHANGE_DISCONNECTED_HOLD" in src


# ===========================================================================
# 2. stop reconcile must not trust a refused intent
# ===========================================================================

class TestStopReconcileTrust:
    @pytest.mark.parametrize("status", ["OK", "NOOP"])
    def test_acted_statuses_pass_the_intent_through(self, status):
        import main as hm
        assert hm.stop_reconcile_intended_target(status, 0) == 0
        assert hm.stop_reconcile_intended_target(status, 1) == 1

    @pytest.mark.parametrize("status", [
        "BLOCKED", "FAILED", "ERROR", "SKIPPED_STALE", "FLIP_DEFERRED",
        "NOT_READY", None, "",
    ])
    def test_refused_statuses_fall_back_to_the_snapshot(self, status):
        import main as hm
        assert hm.stop_reconcile_intended_target(status, 0) is None, (
            f"manage status {status!r} let intended_target=0 through — "
            f"ensure_protective_stop would CANCEL the stop of a position the "
            f"venue just refused to close (the inverse-P207 failure)")

    def test_run_live_routes_through_the_helper(self):
        import main as hm
        src = inspect.getsource(hm.HMATSProductionRunner.run_live)
        assert "stop_reconcile_intended_target(" in src, (
            "run_live's ensure_protective_stop call no longer routes its "
            "intended_target through stop_reconcile_intended_target")


# ===========================================================================
# 3. execute_target guards
# ===========================================================================

def _bare_sleeve():
    from exchange.coinbase_sleeve import CoinbaseSleeve

    class _Client:
        def get_product(self, product_id):
            return {}  # no price fields -> mid resolves 0.0

    class _Adapter:
        _client = _Client()

        def is_connected(self):
            return True

        def to_venue_symbol(self, asset, kind):
            return "BIP-20DEC30-CDE"

        def _contract_size(self, pid):
            return 0.01

    s = object.__new__(CoinbaseSleeve)
    s._adapter = _Adapter()
    s._last_positions = {}
    s._reconcile_ok = True
    s.reconcile_positions = lambda: {}  # type: ignore[assignment]
    s.can_trade = lambda asset, delta: (True, "ok")  # type: ignore[assignment]

    async def _no_cancel(pid, asset):
        return 0

    s._cancel_resting_orders = _no_cancel  # type: ignore[assignment]
    return s


class TestExecuteTargetGuards:
    def test_stale_snapshot_is_refused(self):
        s = _bare_sleeve()
        s._reconcile_ok = False
        res = _run(s.execute_target("BTC", 0))
        assert res["status"] == "SKIPPED_STALE", (
            "execute_target acted on a failed reconcile's last-known "
            "snapshot — sizing delta off stale state can overshoot into an "
            "OPPOSITE position on a venue with no reduce_only")

    def test_priceless_order_is_refused(self):
        s = _bare_sleeve()
        res = _run(s.execute_target("BTC", 1))
        assert res["status"] == "ERROR"
        assert res["reason"].startswith("no_price"), (
            f"expected a no_price refusal, got {res!r} — a SELL limit priced "
            f"off mid=0.0 is 'sell at any price'")


# ===========================================================================
# 4. reconcile sign safety
# ===========================================================================

def _sleeve_with_positions(positions):
    from exchange.coinbase_sleeve import CoinbaseSleeve

    class _Client:
        def list_futures_positions(self):
            return {"positions": positions}

    class _Adapter:
        _client = _Client()

        def is_connected(self):
            return True

    s = object.__new__(CoinbaseSleeve)
    s._adapter = _Adapter()
    s._pid_to_asset = {"BIP-20DEC30-CDE": "BTC"}
    s._last_positions = {}
    s._reconcile_ok = False
    return s


class TestReconcileSignSafety:
    def test_long_is_positive(self):
        s = _sleeve_with_positions([
            {"product_id": "BIP-20DEC30-CDE", "side": "LONG",
             "number_of_contracts": 2}])
        out = s.reconcile_positions()
        assert out["BTC"]["signed_contracts"] == 2
        assert s._reconcile_ok

    def test_sdk_enum_long_is_positive_not_a_phantom_short(self):
        s = _sleeve_with_positions([
            {"product_id": "BIP-20DEC30-CDE",
             "side": "FUTURES_POSITION_SIDE_LONG", "number_of_contracts": 1}])
        out = s.reconcile_positions()
        assert out["BTC"]["signed_contracts"] == 1, (
            "an SDK enum rename manufactured a phantom SHORT — the exact "
            "P253 finding")

    def test_short_with_signed_net_size_does_not_double_negate(self):
        s = _sleeve_with_positions([
            {"product_id": "BIP-20DEC30-CDE", "side": "SHORT",
             "net_size": -2}])
        out = s.reconcile_positions()
        assert out["BTC"]["signed_contracts"] == -2, (
            "SHORT + signed net_size read as a LONG (sign applied twice)")

    def test_unknown_side_refuses_instead_of_guessing(self):
        s = _sleeve_with_positions([
            {"product_id": "BIP-20DEC30-CDE", "side": "SIDEWAYS?",
             "number_of_contracts": 1}])
        s._last_positions = {"BTC": {"signed_contracts": 1.0}}
        out = s.reconcile_positions()
        assert not s._reconcile_ok, (
            "an unrecognized side string was guessed into a sign instead of "
            "failing the reconcile")
        # last-known snapshot returned, not a fabricated position
        assert out == {"BTC": {"signed_contracts": 1.0}}


# ===========================================================================
# 5. stale-data guard honesty + daily-loss producer
# ===========================================================================

class TestPreTickUpdateFreshness:
    def _integrator(self, calls):
        from defense.p0_safety_integrator import P0SafetyIntegrator

        class _Guard:
            def update_timestamp(self, source):
                calls.append(source)

        p = object.__new__(P0SafetyIntegrator)
        p._tick_count = 0
        p.risk_controller = None
        p.human_override = None
        p.stale_guard = _Guard()
        return p

    def test_fresh_data_stamps_the_guard(self):
        calls = []
        self._integrator(calls).pre_tick_update(
            equity=1.0, prices={}, data_fresh=True)
        assert calls == ["kraken_ws", "kraken_rest"]

    def test_stale_data_lets_the_guard_age(self):
        calls = []
        self._integrator(calls).pre_tick_update(
            equity=1.0, prices={}, data_fresh=False)
        assert calls == [], (
            "the stale-data guard was stamped on a tick with invalid data — "
            "back to a check that can never fail (P253 finding)")

    def test_default_preserves_legacy_callers(self):
        calls = []
        self._integrator(calls).pre_tick_update(equity=1.0, prices={})
        assert calls == ["kraken_ws", "kraken_rest"]

    def test_the_producer_site_exists_and_the_orphan_read_is_gone(self):
        # comment-stripped scan (the P177 trap: the fix's own comment quotes
        # the removed string, so a raw-source scan matches its explanation)
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert 'market_data.get("realized_pnl_today"' not in src, (
            "the orphan read is back — no code anywhere writes that key, so "
            "the daily-loss kill switch reads a permanent 0.0")
        assert "_daily_pnl_anchor" in src


# ===========================================================================
# 6. SOTA risk controller persistence
# ===========================================================================

class TestSotaRiskPersistence:
    def test_peak_survives_a_restart(self):
        from risk.sota_risk_controller import SOTARiskController
        c1 = SOTARiskController()
        c1.update_equity(100_000.0)
        c1.update_equity(90_000.0)
        assert c1.peak_equity == 100_000.0
        c2 = SOTARiskController()
        c2.update_equity(90_000.0)  # the restart re-anchor this fix removes
        c2.from_dict(c1.to_dict())
        assert c2.peak_equity == 100_000.0, (
            "restart re-anchored the peak — accumulated drawdown erased")

    def test_kill_switch_restores_as_active(self):
        from risk.sota_risk_controller import SOTARiskController
        c1 = SOTARiskController()
        c1.kill_switch_active = True
        c2 = SOTARiskController()
        c2.from_dict(c1.to_dict())
        assert c2.kill_switch_active, (
            "a restart cleared the kill switch — the control disarmed itself")

    def test_from_dict_is_one_directional(self):
        from risk.sota_risk_controller import SOTARiskController
        c = SOTARiskController()
        c.kill_switch_active = True
        c.peak_equity = 100_000.0
        c.from_dict({"kill_switch_active": False, "peak_equity": 1.0})
        assert c.kill_switch_active, "a payload CLEARED a live kill switch"
        assert c.peak_equity == 100_000.0, "a payload LOWERED the peak"

    def test_malformed_payload_is_harmless(self):
        from risk.sota_risk_controller import SOTARiskController
        c = SOTARiskController()
        c.from_dict({"peak_equity": "banana", "risk_state": "NOT_A_STATE",
                     "kill_switch_active": True,
                     "kill_switch_reason": "NOT_A_REASON",
                     "kill_switch_time": "not-a-time"})
        assert c.kill_switch_active  # armed, with best-effort metadata
        c2 = SOTARiskController()
        c2.from_dict("not a dict")  # type: ignore[arg-type]
        assert not c2.kill_switch_active


# ===========================================================================
# 7. fuse window persistence
# ===========================================================================

class TestFuseWindowPersistence:
    def test_todict_keeps_a_full_28d_window(self):
        # comment-stripped: the fix's own comment names the old [-50:] cap
        from tests._source_scan import code_only
        src = code_only(REPO / "defense" / "strategy_existence_fuse.py")
        assert "[-400:]" in src, (
            "to_dict's record cap shrank — at 6 records/day, anything under "
            "~168 truncates the 28d window across restarts (P253 finding: "
            "the old 50-record cap made '28 days' mean 8)")
        assert "[-50:]" not in src


# ===========================================================================
# 8. tripwire: no-data is not GATE-CLOSED
# ===========================================================================

class TestTripwireNoData:
    def _write_report(self, d, day, vs):
        """vs=None -> all horizons INSUFFICIENT (no vs_threshold key)."""
        rep = {"generated": f"{day}T06:20:00+00:00", "assets": {
            a: {"4": ({"vs_threshold": vs} if vs is not None
                      else {"status": "INSUFFICIENT"})}
            for a in ("BTC", "ETH", "SOL")}}
        (d / f"slope_{day}.json").write_text(json.dumps(rep), encoding="utf-8")

    def _run_main(self, reports_dir, monkeypatch, capsys):
        from analytics.calibration import tripwire_check as tw
        monkeypatch.setattr(sys, "argv", [
            "tripwire_check", "--reports-dir", str(reports_dir),
            "--today", "2026-09-08"])
        rc = tw.main()
        return rc, capsys.readouterr().out

    def test_insufficient_reports_do_not_fire_the_tripwire(
            self, tmp_path, monkeypatch, capsys):
        for i, day in enumerate(
                ["2026-08-11", "2026-08-18", "2026-08-25", "2026-09-01"]):
            self._write_report(tmp_path, day, vs=None)
        rc, out = self._run_main(tmp_path, monkeypatch, capsys)
        assert rc == 0, (
            "four all-INSUFFICIENT reports FIRED the tripwire — 'not enough "
            "data' was read as 'gate closed', which deactivates a live asset "
            "on an outage (the P199 refusal principle, violated)")
        assert "no-data" in out

    def test_real_gate_closed_reports_still_fire(
            self, tmp_path, monkeypatch, capsys):
        for day in ["2026-08-11", "2026-08-18", "2026-08-25", "2026-09-01"]:
            self._write_report(tmp_path, day, vs="BELOW-THRESHOLD")
        rc, out = self._run_main(tmp_path, monkeypatch, capsys)
        # [P299] The PRESCRIPTION is retired, the DETECTION is not: the
        # streak must still be counted and reported (that evidence feeds
        # the seat decision), but the checker no longer exits 3 or tells
        # anyone to edit trend_assets.
        assert rc == 0, "the tripwire no longer FIRES — it reports (P299)"
        assert "FIRED" in out, "the streak must still be DETECTED"
        assert "SUPERSEDED" in out
        assert "trend_assets" not in out or "do NOT edit" in out


# ===========================================================================
# 9. shadow-IC overlap correction
# ===========================================================================

class TestShadowIcOverlapCorrection:
    def _assess(self, n):
        from analytics.shadow_ic.compute_shadow_ic import assess_promotion
        return assess_promotion(
            ic_per_h={4: 0.10}, n_per_h={4: n}, sharpe=1.0, window_days=30,
            fwd_vol_bps_per_h={4: 500.0},  # big vol so the cost bar passes
        )

    def test_t_stat_uses_effective_samples(self):
        # ic=0.10, n=500, h=4: naive t = 2.23 (passes 2.0), corrected
        # t = 0.10*sqrt(124) = 1.11 (fails). The naive arithmetic let this
        # promote-grade significance claim through on 4x-overlapped samples.
        a = self._assess(500)
        d = a.per_horizon[4]
        assert d["n_eff"] == 125
        assert d["t_stat"] == pytest.approx(0.10 * (124 ** 0.5), rel=1e-6)
        assert any("n_eff" in b for b in a.blockers), (
            "overlap-corrected significance did not block — the shadow gate "
            "is still ~sqrt(h) looser than agent_ic_review on the same "
            "doctrine (P253 finding)")

    def test_enough_effective_samples_clears_the_significance_bar(self):
        # n=1700, h=4 -> n_eff=425 -> t = 0.10*sqrt(424) = 2.06 >= 2.0
        a = self._assess(1700)
        assert not any("SE from zero" in b for b in a.blockers)


# ===========================================================================
# 10. routing fail-safe
# ===========================================================================

class TestRoutingFailSafe:
    def _ctx(self, flag):
        class _Cfg:
            coinbase_routing_enabled = flag

        class _Ctx:
            config = _Cfg()

        return _Ctx()

    def test_unreadable_routing_with_flag_on_blocks_kraken(self, monkeypatch):
        import core.execution_service as es
        monkeypatch.setattr(es, "_coinbase_get_routing", lambda: None)
        monkeypatch.setattr(es, "_CB_ROUTED_FAILSAFE_WARNED", False)
        assert es._coinbase_routed(self._ctx(True), "BTC") is True, (
            "flag ON + unreadable routing state resumed KRAKEN SPOT entries "
            "— reopening the venue P152 closed, on a file-read error")

    def test_flag_off_is_still_not_routed(self, monkeypatch):
        import core.execution_service as es
        monkeypatch.setattr(es, "_coinbase_get_routing", lambda: None)
        monkeypatch.setattr(es, "_CB_ROUTED_FAILSAFE_WARNED", False)
        assert es._coinbase_routed(self._ctx(False), "BTC") is False

    def test_the_duplicate_sleeve_factory_stays_deleted(self):
        import core.execution_service as es
        assert not hasattr(es, "_coinbase_get_sleeve"), (
            "_coinbase_get_sleeve is back — a second, unconfigured "
            "CoinbaseSleeve (no stop, no caps) beside the real one is a "
            "P139-shaped duplicate book")


# ===========================================================================
# 11. offline tooling source pins
# ===========================================================================

class TestOfflineToolingPins:
    def test_rebuild_pipeline_funding_is_causally_shifted(self):
        src = (REPO / "training" / "scripts" / "rebuild_pipeline.py").read_text(
            encoding="utf-8")
        assert 'funding["funding_close"].shift(1)' in src.replace("\n", "").replace(
            "            ", " ") or 'funding["funding_close"].shift(1)' in src, (
            "the P247-F1 funding look-ahead is back in rebuild_pipeline — "
            "z-scoring the UNSHIFTED day-close on day-open-stamped rows hands "
            "every 00:00-12:00 bar up to 16h of future funding")

    def test_sol_export_refuses_without_force(self):
        src = (REPO / "training" / "scripts" /
               "export_regime_book_models.py").read_text(encoding="utf-8")
        assert "--force-retired" in src and "REFUSING" in src, (
            "the retired SOL bear export lost its refusal gate — re-running "
            "it silently resurrects a leg retired on evidence (P250-F1b)")

    def test_fastrisk_gate_is_sleeve_exists_not_cached_nonzero(self):
        import main as hm
        src = inspect.getsource(
            hm.HMATSProductionRunner._handle_fast_risk_action)
        assert "_frs_sleeve is not None:" in src
        assert "_frs_sleeve is not None and int(" not in src, (
            "the FastRiskTick sleeve branch is gated on the CACHED contract "
            "count again — a stale-zero cache silently no-ops an EXIT_ONLY")

    def test_run_live_saves_state_at_loop_level(self):
        import main as hm
        src = inspect.getsource(hm.HMATSProductionRunner.run_live)
        assert src.count("_save_paper_positions(force=True)") >= 2, (
            "run_live's loop-level state save is gone — with the Coinbase "
            "adapter down, governor/fuse state stops being persisted (the "
            "P209 failure re-armed)")


# ===========================================================================
# 12. [P253b] deploy gate: CI-green on the deployed sha + explicit mypy skip
# ===========================================================================

class TestDeployGateP253b:
    def test_skip_mypy_and_require_all_gates_are_mutually_exclusive(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "tools/ci_check_invariants.py",
             "--skip-mypy", "--require-all-gates"],
            capture_output=True, text=True, cwd=REPO, encoding="utf-8")
        assert r.returncode == 2, (
            "--skip-mypy + --require-all-gates must refuse: one demands the "
            "gate runs, the other refuses to run it")

    def test_deploy_script_verifies_ci_for_the_deployed_sha(self):
        # comment-stripped (the P177 trap: the step-0 comment explains WHY
        # --require-all-gates was removed, by naming it)
        src = "\n".join(
            line.split("#", 1)[0]
            for line in (REPO / "scripts" / "hetzner_deploy.sh").read_text(
                encoding="utf-8").splitlines())
        # 0a: the deployed sha is origin/main and its CI conclusions are read
        assert "git ls-remote origin refs/heads/main" in src
        # [P344] The API query moved into tools/ci_status.py (ONE
        # implementation). The PROPERTY is unchanged and is asserted
        # where it now lives: the deploy passes the DEPLOYED sha to the
        # tool, and the tool queries that sha.
        assert '--sha "${DEPLOY_SHA}"' in src
        tool = io.open(REPO / "tools" / "ci_status.py",
                       encoding="utf-8").read()
        assert "actions/runs?head_sha=" in tool
        assert "HMATS_DEPLOY_SKIP_CI_CHECK" in src, (
            "the emergency override is gone — an API outage would then "
            "permanently block deploys")
        # 0b: local scanners run WITHOUT local mypy enforcement — the mypy
        # baseline is CI's environment fingerprint (P227), so a local
        # --require-all-gates blocks every deploy from the operator machine
        assert "--skip-mypy" in src
        assert "--require-all-gates" not in src, (
            "--require-all-gates is back in the deploy script — the P253b "
            "finding: it blocked every deploy from the operator machine on "
            "phantom environment-fingerprint findings")

    def test_skipped_mypy_is_bannered_not_silent(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "tools/ci_check_invariants.py",
             "--skip-mypy"],
            capture_output=True, text=True, cwd=REPO, encoding="utf-8")
        assert "SKIPPED BY FLAG" in r.stderr, (
            "the explicit mypy skip must announce itself — a silent skip is "
            "the exact P187 hole this flag exists to avoid recreating")


# ===========================================================================
# 13. [P253d] the armed / completed ledger items
# ===========================================================================

class TestP253dArmedItems:
    def test_correlation_key_now_has_a_real_producer(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "data_mgmt" / "market_data_pipeline.py")
        assert 'raw["correlation_btc_eth_sol"] = _xcorr' in src, (
            "correlation_btc_eth_sol lost its real producer — back to a "
            "write-only 0.87 constant that no consumer can ever see move "
            "(P253c ledger item 1)")

    def test_hard_flatten_threshold_is_unreachable_by_measurement(self):
        # The arming decision leaned on this: the 20-bar mean pairwise
        # correlation never reached 0.98 in 8y of data (p95 = 0.93). If the
        # crisis threshold is ever LOWERED toward the measured range, the
        # arming must be re-evaluated — this pins the number the decision
        # assumed.
        import json
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert live.get("correlation_crisis", 0.98) >= 0.95, (
            "correlation_crisis was lowered below 0.95 — the P253d arming "
            "was justified by 0.98 being unreachable (0.000% of 13,013 "
            "bars); re-measure before accepting this")

    def test_dd_snapshot_reads_sleeve_equity_live(self):
        import main as hm
        src = inspect.getsource(hm.HMATSProductionRunner._update_drawdown_snapshot)
        assert "sleeve_equity_usd()" in src, (
            "the DD halt is back on the cached _last_equity_usd — one 4H "
            "bar of sleeve drawdown invisible to the halt (P253c ledger)")

    def test_kraken_spot_map_matches_the_canonical_symbol(self):
        from exchange.symbol_mapping import to_venue_symbol
        from core.execution_service import _CANONICAL_SPOT_SYMBOL
        for asset in ("BTC", "ETH", "SOL"):
            assert (to_venue_symbol(asset, "kraken", "spot")
                    == _CANONICAL_SPOT_SYMBOL.get(asset, f"{asset}/USD")), (
                f"the two sources of truth for the Kraken {asset} spot "
                f"symbol disagree again — a Kraken revival would trade the "
                f"wrong quote pair")

    def test_derivatives_router_is_guarded_for_routed_assets(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert "_deriv_asset_routed" in src, (
            "the derivatives router lost its P152-class guard — it runs "
            "BEFORE execute_intent_v2, so re-enabling it would place Kraken "
            "derivative orders beside the Coinbase sleeve")

    def test_cc_onchain_backoff_persists_and_binds_without_cache(self, tmp_path, monkeypatch):
        import time as _time
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        from data_mgmt.feeds.cryptocompare_onchain import CryptoCompareOnChainFeed
        f1 = CryptoCompareOnChainFeed(api_key="k")
        f1._rate_limited_until = _time.time() + 900
        f1._persist_backoff()
        # a fresh instance (= a restart) must restore the ACTIVE backoff
        f2 = CryptoCompareOnChainFeed(api_key="k")
        assert f2._rate_limited_until > _time.time(), (
            "restart cleared an active CryptoCompare backoff — the limiter "
            "re-arms on restart (the P154 non-control)")

    def test_gambler_regimes_are_real_vocabulary(self):
        import json
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        regimes = set(live.get("gambler", {}).get("allowed_regimes", []))
        live_vocab = {"QUIET_ACCUMULATION", "WEAK_CONSOLIDATION",
                      "STEADY_UPTREND", "MOMENTUM_RALLY", "NEUTRAL_DRIFT",
                      "VOLATILE_CHOP", "EXTREME_VOLATILITY", "PANIC_SELLOFF"}
        assert regimes and regimes <= live_vocab, (
            f"gambler.allowed_regimes {regimes} contains names outside the "
            f"live GMM vocabulary — the feature is dormant by vocabulary "
            f"drift again (P253c ledger item 9)")

    def test_config_schema_is_wired_warn_only(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert "validate_config_consistency" in src, (
            "configs/config_schema.py went back to zero production "
            "consumers — the live config is never schema-validated")
        assert "[CONFIG-SCHEMA]" in src

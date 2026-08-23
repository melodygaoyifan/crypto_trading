"""[P382] main.py fixes from the 2026-08-22 read-through, pinned.

 1. Sleeve veto classification — four strings that reached the sleeve as
    FLATTEN and should not have:
      CORRELATION_COLLAPSE  (NO_TRADE subtype; live checker fires on
                             corr >= 0.92 ALONE — fired 2026-08-19 x2)
      VOLUME_CONTRACTING    (trade_gate entry-quality, position-blind;
                             fired 2026-08-22 08:27 on SOL)
      [PATCH-4] SOFT block  (reads the empty Kraken book for "has position")
      [INTEGRITY]           (Kraken data-integrity abort = state unknown)
    plus MAX_HOLD's early return, which handed the sleeve a bare intent
    (=> zero_target_exposure => FLATTEN the Coinbase book on a Kraken exit).
 2. `alert_manager.send_alert` was called with the wrong signature at two
    CRITICAL sites (message passed as alert_type, no title/message) ->
    TypeError swallowed -> neither alert has ever reached the channel.
 3. The `[COINBASE-SHADOW]` parity `get_product` loop was unguarded inside
    the try whose handler wraps the ENTIRE Coinbase block, order path
    included — one transient 5xx skipped manage/stop/fuse for a 4H tick.
 4. LIVE's cascade-governor restore fed the per-asset map into the
    single-instance `from_dict` (vacuous restore logged as success).
 5. The watchdog exit/reduce path now asks the sleeve for a stop follow-up,
    and the 30s loop runs pending follow-ups (see
    tests/test_p382_stop_sizing_followup.py for the sleeve side).
 6. `_enh_gated_dirs` is reset at loop level (was a high-water mark on
    ticks where the driver block did not run).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import re
from pathlib import Path

import pytest

import main as m
from main import (SLEEVE_HOLD, SLEEVE_ENTRY_BLOCKED,
                  sleeve_direction_from_intent, sleeve_fast_risk_action)

REPO = Path(__file__).resolve().parents[1]


class _Intent:
    def __init__(self, direction=0.9, target_exposure=0.3, veto_active=True,
                 veto_reason=""):
        self.direction = direction
        self.target_exposure = target_exposure
        self.veto_active = veto_active
        self.veto_reason = veto_reason


# ---------------------------------------------------------------------------
# 1. classification
# ---------------------------------------------------------------------------
class TestReclassifiedVetoes:
    def test_correlation_collapse_no_trade_holds(self):
        d, r = sleeve_direction_from_intent(
            _Intent(veto_reason="[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE"), 0.9)
        assert d is SLEEVE_HOLD and "hold_veto" in r

    @pytest.mark.parametrize("reason", [
        "[TRADE_GATE] VOLUME_CONTRACTING",
        "[P0_SAFETY] Trade gate reject: VOLUME_CONTRACTING",
    ])
    def test_volume_contracting_is_entry_quality(self, reason):
        d, r = sleeve_direction_from_intent(_Intent(veto_reason=reason), 0.9)
        assert d is SLEEVE_ENTRY_BLOCKED, (d, r)

    def test_volume_contracting_from_flat_resolves_to_hold_not_flatten(self):
        # the resolver keeps a flat book flat and an aligned position held
        t, why = m.sleeve_entry_blocked_resolve(0, 1.0, "x")
        assert t is SLEEVE_HOLD
        t2, _ = m.sleeve_entry_blocked_resolve(2, 1.0, "x")
        assert t2 is SLEEVE_HOLD

    def test_patch4_soft_block_holds_but_hard_still_flattens(self):
        d, r = sleeve_direction_from_intent(
            _Intent(veto_reason="[PATCH-4] SOFT block(NORMAL): ['vpin']"), 0.9)
        assert d is SLEEVE_HOLD, (d, r)
        d2, r2 = sleeve_direction_from_intent(
            _Intent(veto_reason="[PATCH-4] HARD: corr>=0.98"), 0.9)
        assert d2 == 0.0 and r2.startswith("veto_flat")

    def test_integrity_abort_holds(self):
        d, r = sleeve_direction_from_intent(
            _Intent(veto_reason="[INTEGRITY] Data integrity check failed - stale or corrupt data"),
            0.9)
        assert d is SLEEVE_HOLD

    def test_correlation_crisis_and_sol_defense_still_flatten(self):
        for reason in ("CORRELATION_CRISIS: BTC-ETH 0.99",
                       "[SOL DEFENSE] FORCE FLAT - congestion",
                       "[SOL DEFENSE] CRITICAL - outage"):
            d, r = sleeve_direction_from_intent(_Intent(veto_reason=reason), 0.9)
            assert d == 0.0 and r.startswith("veto_flat"), (reason, d, r)

    def test_max_hold_exit_returns_a_hold_veto_not_a_bare_intent(self):
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        i = src.index("[MAX_HOLD_TIMEOUT]")
        blk = src[i:i + 6000]
        assert 'veto_reason="MAX_HOLD_EXIT_HOLD' in blk, (
            "the MAX_HOLD early return no longer carries the HOLD veto — a "
            "bare TradeIntentV36() reads as zero_target_exposure and FLATTENS "
            "the Coinbase book")
        assert "MAX_HOLD_EXIT_HOLD" in m._SLEEVE_HOLD_VETOES
        d, _ = sleeve_direction_from_intent(
            _Intent(veto_reason="MAX_HOLD_EXIT_HOLD - Kraken max-hold exit"), 0.9)
        assert d is SLEEVE_HOLD


# ---------------------------------------------------------------------------
# 2. send_alert signature
# ---------------------------------------------------------------------------
def _send_alert_calls():
    src = (REPO / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "send_alert"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "alert_manager"):
            calls.append(node)
    return calls


class TestSendAlertBindsToTheSignature:
    def test_every_call_site_binds(self):
        from infra.alert_manager import AlertManager
        sig = inspect.signature(AlertManager.send_alert)
        calls = _send_alert_calls()
        assert len(calls) >= 4, "the scan found too few call sites (P174)"
        for c in calls:
            # positional count + keyword names must bind; the first positional
            # must be an AlertType attribute, not a string literal
            kw = {k.arg: None for k in c.keywords if k.arg}
            try:
                sig.bind(None, *([None] * len(c.args)), **kw)
            except TypeError as e:
                pytest.fail(f"main.py:{c.lineno} send_alert(...) does not "
                            f"bind: {e}")
            assert c.args, f"main.py:{c.lineno}: no alert_type positional"
            first = c.args[0]
            assert isinstance(first, ast.Attribute) and \
                isinstance(first.value, ast.Name) and \
                first.value.id == "AlertType", (
                    f"main.py:{c.lineno}: first positional must be an "
                    f"AlertType member, got {ast.dump(first)[:60]}")
            sev = [k for k in c.keywords if k.arg == "severity"]
            for k in sev:
                assert not isinstance(k.value, ast.Constant), (
                    f"main.py:{c.lineno}: severity must be an AlertSeverity "
                    f"member, not a string")

    def test_the_two_fixed_sites_are_present(self):
        src = (REPO / "main.py").read_text(encoding="utf-8")
        assert "Dead-man switch refresh failed - orders at risk" in src
        assert "data feeds failed" in src


# ---------------------------------------------------------------------------
# 3. parity loop guarded
# ---------------------------------------------------------------------------
class TestParityLoopIsGuarded:
    def test_the_get_product_loop_body_is_a_try(self):
        src = (REPO / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        found = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == "_cb_a"):
                found.append(node)
        assert found, "the [COINBASE-SHADOW] parity loop was not found"
        for f in found:
            assert f.body and isinstance(f.body[0], ast.Try), (
                f"main.py:{f.lineno}: the parity get_product loop is not "
                f"wrapped in its own try — a diagnostic read failure would "
                f"skip the sleeve driver for the whole tick")


# ---------------------------------------------------------------------------
# 4. cascade restore shape
# ---------------------------------------------------------------------------
class TestCascadeRestoreShape:
    def test_load_paper_positions_uses_the_per_asset_reader(self):
        src = inspect.getsource(m.HMATSProductionRunner._load_paper_positions)
        assert "restore_governor_states(cascade_data)" in src
        assert "get_cascade_exhaustion_governor().from_dict(cascade_data)" not in src, (
            "the single-instance from_dict on the per-asset map is the "
            "vacuous restore")

    def test_the_per_asset_reader_accepts_the_writer_shape(self):
        from risk.cascade_exhaustion_governor import (
            all_governor_states, restore_governor_states,
            get_cascade_exhaustion_governor)
        get_cascade_exhaustion_governor(asset="BTC")   # ensure a per-asset instance exists
        snap = all_governor_states()
        assert "BTC" in snap, list(snap)
        assert isinstance(snap, dict)
        # round trip: whatever the writer emits, the reader restores >= 1
        n = restore_governor_states(snap)
        assert n >= 1


# ---------------------------------------------------------------------------
# 5. watchdog follow-up hook
# ---------------------------------------------------------------------------
class _FakeSleeve:
    def __init__(self, cur, status="OK"):
        self._cur = cur
        self._reconcile_ok = True
        self._status = status
        self.followups = []
        self.targets = []

    def reconcile_positions(self): return {}
    def signed_contracts(self, a): return self._cur

    async def execute_target(self, asset, target, urgent=False):
        self.targets.append(target)
        return {"status": self._status}

    def request_stop_followup(self, asset, intended):
        self.followups.append((asset, intended))


class TestWatchdogRequestsStopFollowup:
    def test_exit_only_requests_followup_with_intent_zero(self):
        s = _FakeSleeve(cur=3)
        st, _ = asyncio.run(sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "EXITED"
        assert s.followups == [("ETH", 0.0)]

    def test_reduce_requests_followup_with_the_reduced_target(self):
        s = _FakeSleeve(cur=4)
        st, _ = asyncio.run(sleeve_fast_risk_action(s, "ETH", "REDUCE_50", True))
        assert st == "REDUCED"
        assert s.followups == [("ETH", 2.0)]

    def test_a_sleeve_without_the_hook_still_works(self):
        s = _FakeSleeve(cur=3)
        s.request_stop_followup = None   # pre-P382 sleeve shape: no hook
        st, _ = asyncio.run(sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "EXITED"

    def test_the_30s_loop_runs_pending_followups(self):
        src = (REPO / "main.py").read_text(encoding="utf-8")
        # both loop copies (run_paper/run_live) carry the call
        assert src.count("followup_protective_stop") >= 2
        i = src.index("frt_result = self.fast_risk_tick.evaluate(")
        blk = src[max(0, i - 3000):i]
        assert "followup_protective_stop" in blk, (
            "the stop follow-up must run in the 30s loop BEFORE the "
            "evaluate, or a pending oversized stop waits for the 4H tick")


# ---------------------------------------------------------------------------
# 6. _enh_gated_dirs reset
# ---------------------------------------------------------------------------
class TestEnhGatedDirsResetPerLoop:
    def test_reset_precedes_the_coinbase_block(self):
        src = inspect.getsource(m.HMATSProductionRunner.run_live)
        i_reset = src.index("self._enh_gated_dirs = {}")
        # the first per-asset WRITE of the claim dict (inside the driver block)
        i_block = src.index("self._enh_gated_dirs[_m_a]")
        assert i_reset < i_block, (
            "the eventfilter claim dict must be reset at loop level, before "
            "the block that may or may not write it")

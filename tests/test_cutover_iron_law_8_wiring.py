"""[P202] The cutover runtime guard: what it checks now, and why the old one went.

HISTORY. [P155] found Iron Law 8 ("DRL authority must be ACTIVE during cutover")
was defined but never enforced — `advance_phase()` enforces it and nothing in
production calls it, because `_coinbase_get_routing()` assigns `rp.phase`
straight from data/coinbase_routing_state.json. P155 wired a log-only check.
[P193] then fixed that check reading the wrong attribute on one of the two ctx
shapes, which made it cry wolf and burn its one-shot latch.

[P202] RETIRES the DRL clause entirely. It is literally true and substantively
meaningless: DRL cannot influence a single live order. The sleeve trades
`_last_quant_directions` (written main.py:6480 / :7834) while `drl_direction`
is written at :7902 — after both — and no path connects them. Re-promoting DRL
to ACTIVE would satisfy the law and change nothing, so the CRITICAL fired every
process start, paged Discord, and told the operator to fix a non-cause. An
alert nobody can act on is one everybody learns to ignore, which is the same
reasoning that retired the always-red auto-deploy in P196.

WHAT REPLACES IT: the condition that actually protects a routed asset — that it
is trading a live perp with no venue-resting protective stop. Post-Phase-B the
sleeve carries 100% of the directional risk and bypasses the core risk stack
(P201), so the stop is the only thing that survives the process dying. Unlike
the DRL clause this is self-extinguishing: fixing the gap silences the alert.
"""

import logging
import types

import pytest

from core import execution_service as es
from exchange.routing import CutoverPhase, RoutingPolicy


@pytest.fixture(autouse=True)
def _reset_latches():
    es._CB_GUARD_WARNED.clear()
    es._CB_FEE_MODEL_WARNED = False
    yield
    es._CB_GUARD_WARNED.clear()
    es._CB_FEE_MODEL_WARNED = False


def _ctx(stop_pct=0.0, stop_assets=None, routing_enabled=True):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            coinbase_routing_enabled=routing_enabled,
            coinbase_protective_stop_pct=stop_pct,
            coinbase_protective_stop_assets=list(stop_assets or []),
        )
    )


def _critical(caplog):
    return [r.message for r in caplog.records if r.levelno >= logging.CRITICAL]


# ---------------------------------------------------------------------------
# the guard that replaced the DRL clause
# ---------------------------------------------------------------------------

class TestUnprotectedRoutedAssetIsReported:

    def test_routed_with_no_stop_configured_at_all_is_critical(self, caplog):
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
        with caplog.at_level(logging.CRITICAL):
            es._coinbase_check_cutover_guards(_ctx(stop_pct=0.0), rp, "BTC")
        msgs = _critical(caplog)
        assert any("CUTOVER-GUARDS" in m and "BTC" in m for m in msgs), msgs

    def test_routed_but_outside_the_stop_allowlist_is_critical(self, caplog):
        """The live P197 rollout state: SOL protected, BTC/ETH are not."""
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
        with caplog.at_level(logging.CRITICAL):
            es._coinbase_check_cutover_guards(
                _ctx(stop_pct=0.10, stop_assets=["SOL"]), rp, "BTC")
        assert any("BTC" in m for m in _critical(caplog))

    def test_a_protected_asset_is_silent(self, caplog):
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
        with caplog.at_level(logging.CRITICAL):
            es._coinbase_check_cutover_guards(
                _ctx(stop_pct=0.10, stop_assets=["SOL"]), rp, "SOL")
        assert _critical(caplog) == []

    def test_an_empty_allowlist_means_every_asset_is_covered(self, caplog):
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
        with caplog.at_level(logging.CRITICAL):
            for a in ("BTC", "ETH", "SOL"):
                es._coinbase_check_cutover_guards(
                    _ctx(stop_pct=0.10, stop_assets=[]), rp, a)
        assert _critical(caplog) == []

    def test_it_is_self_extinguishing(self, caplog):
        """The property the DRL clause lacked: fixing the gap stops the alert."""
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
        with caplog.at_level(logging.CRITICAL):
            es._coinbase_check_cutover_guards(_ctx(stop_pct=0.0), rp, "BTC")
        assert _critical(caplog), "unprotected asset was not reported"
        caplog.clear()
        es._CB_GUARD_WARNED.clear()
        with caplog.at_level(logging.CRITICAL):
            es._coinbase_check_cutover_guards(
                _ctx(stop_pct=0.10, stop_assets=[]), rp, "BTC")
        assert _critical(caplog) == [], (
            "still alerting after the stop was configured — the alert cannot be "
            "resolved by acting on it, which is what made the DRL clause noise"
        )

    def test_pre_phase_2_is_not_a_cutover_so_nothing_is_reported(self, caplog):
        rp = RoutingPolicy(phase=CutoverPhase.PRE_PHASE_2)
        with caplog.at_level(logging.CRITICAL):
            es._coinbase_check_cutover_guards(_ctx(stop_pct=0.0), rp, "BTC")
        assert _critical(caplog) == []

    def test_latched_per_asset_not_globally(self, caplog):
        """The old latch was one global bool, so the first asset consumed the
        single shot and the others were never reported at all."""
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
        with caplog.at_level(logging.CRITICAL):
            for a in ("BTC", "ETH"):
                for _ in range(3):
                    es._coinbase_check_cutover_guards(_ctx(stop_pct=0.0), rp, a)
        msgs = _critical(caplog)
        assert len(msgs) == 2, msgs
        assert any("BTC" in m for m in msgs) and any("ETH" in m for m in msgs)

    def test_the_message_names_the_fix(self):
        """An alert that cannot be acted on is one that gets ignored."""
        import inspect
        src = inspect.getsource(es._coinbase_check_cutover_guards)
        assert "coinbase_protective_stop_assets" in src
        assert "FIX:" in src

    def test_it_never_raises_into_the_order_path(self):
        es._coinbase_check_cutover_guards(_ctx(), object(), "BTC")   # no .phase
        es._coinbase_check_cutover_guards(object(), RoutingPolicy(
            phase=CutoverPhase.DUAL_VENUE), "BTC")                   # no .config


# ---------------------------------------------------------------------------
# the retired clause must stay retired
# ---------------------------------------------------------------------------

class TestTheDrlClauseStaysRetired:

    def test_no_runtime_caller_of_validate_drl_active(self):
        """`validate_drl_active` remains in exchange/cutover.py for
        advance_phase(), which production does not call. Re-wiring it to the
        live path re-creates a CRITICAL nobody can act on — do that only after
        re-establishing that DRL actually reaches an order."""
        import re
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1]
               / "core" / "execution_service.py").read_text(encoding="utf-8")
        # Match an IMPORT or a CALL, not a prose mention — the docstring
        # deliberately names the function to say where it still lives. A bare
        # substring guard would fire on that, which is the P192 mistake
        # (the `_emergency_flatten` source guard tripping on its own comment).
        offenders = re.findall(
            r"^\s*(?:from\s+\S+\s+import\s+[^\n]*\bvalidate_drl_active\b"
            r"|import\s+[^\n]*\bvalidate_drl_active\b"
            r"|[^#\n]*\bvalidate_drl_active\s*\()", src, re.M)
        assert not offenders, (
            f"the DRL-ACTIVE clause is back on the live path: {offenders}. It is "
            f"vacuous while the sleeve cannot see drl_direction (P202)"
        )

    def test_routing_is_unchanged_by_an_unprotected_asset(self, monkeypatch, caplog):
        """Observes, never blocks — blocking would route back to a flat Kraken."""
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
        monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
        with caplog.at_level(logging.CRITICAL):
            assert es._coinbase_routed(_ctx(stop_pct=0.0), "SOL") is True
        assert _critical(caplog), "…but it must still be reported"

    def test_the_guard_only_runs_for_assets_actually_routed(self, monkeypatch, caplog):
        """The old clause ran unconditionally and reported a global condition,
        so it fired even for assets going to Kraken."""
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
        monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
        with caplog.at_level(logging.CRITICAL):
            assert es._coinbase_routed(_ctx(stop_pct=0.0), "BTC") is False
        assert _critical(caplog) == [], "reported a Kraken-routed asset"

    def test_routed_still_fails_closed_when_the_flag_is_off(self, monkeypatch):
        rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
        monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
        assert es._coinbase_routed(_ctx(routing_enabled=False), "SOL") is False


# ---------------------------------------------------------------------------
# [P155] fee-model mismatch reporting — unchanged, kept from the original file
# ---------------------------------------------------------------------------

def _warnings(caplog):
    return [r.message for r in caplog.records if r.levelno == logging.WARNING]


def test_routed_asset_reports_the_fee_mismatch(monkeypatch, caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    with caplog.at_level(logging.WARNING):
        es._coinbase_routed(_ctx(stop_pct=0.10, stop_assets=[]), "SOL")
    msgs = _warnings(caplog)
    assert any("FEE-MODEL-MISMATCH" in m for m in msgs), msgs


def test_fee_mismatch_reported_once_not_every_tick(monkeypatch, caplog):
    rp = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE, coinbase_assets=["SOL"])
    monkeypatch.setattr(es, "_coinbase_get_routing", lambda: rp)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            es._coinbase_routed(_ctx(stop_pct=0.10, stop_assets=[]), "SOL")
    assert len([m for m in _warnings(caplog) if "FEE-MODEL-MISMATCH" in m]) == 1

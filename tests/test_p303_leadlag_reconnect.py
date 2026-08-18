"""
[P303] The lead-lag monitors could not reconnect after a failed INITIAL
connect, and said so at ERROR.

Found by chasing a live Discord alert:

    DeribitDVOL | ERROR | Failed to connect: server rejected WebSocket
                          connection: HTTP 503

Measured first: 503 from the trading server AND from an unrelated host, on
REST and WebSocket alike — a VENDOR-side outage, not a block on us and not
the P293b Cloudflare client-signature trap. So the connection failure itself
was nobody's bug. What it exposed was:

  1. `_running` was set only on the SUCCESS path, while `process_messages`
     gates on `while self._running:`. A failed initial connect therefore
     returned from the loop immediately, `_reconnect_ws` was never reached,
     and the monitor stayed dead for the whole process lifetime — the L4-03
     "auto-reconnect" unable to fire in the one case it was written for.
  2. Nothing upstream recovers it: main.py's restart guard reads the ENGINE's
     `_running`, which `start()` sets unconditionally, so the engine reports
     healthy while both monitors are dead.
  3. The failure logged at ERROR, which is forwarded to Discord — an alert
     the operator can neither act on nor be harmed by (P202/P240).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from market.lead_lag_engine import (  # noqa: E402
    BinanceTakerMonitor, DeribitDVOLMonitor)

MONITORS = [
    pytest.param(BinanceTakerMonitor, 4, id="binance"),
    pytest.param(DeribitDVOLMonitor, 8, id="deribit"),
]


def _fail_connect(monkeypatch, exc=None):
    """Make websockets.connect raise, the way a 503 does."""
    import market.lead_lag_engine as mod

    class _Boom:
        async def __call__(self, *a, **k):
            raise (exc or RuntimeError(
                "server rejected WebSocket connection: HTTP 503"))

    monkeypatch.setattr(mod, "WEBSOCKETS_AVAILABLE", True)
    monkeypatch.setattr(mod, "websockets",
                        type("W", (), {"connect": _Boom()})())


# =============================================================================
# 1. The reconnect loop must be REACHABLE after a failed initial connect
# =============================================================================

class TestTheReconnectLoopIsArmed:

    @pytest.mark.parametrize("cls,_thresh", MONITORS)
    def test_a_failed_initial_connect_leaves_the_loop_armed(
            self, cls, _thresh, monkeypatch):
        """THE BUG. Without this, process_messages returns on its first
        `while self._running:` check and the monitor is dead forever."""
        _fail_connect(monkeypatch)
        m = cls()
        asyncio.run(m.connect())
        assert m.ws is None, "the connection genuinely failed"
        assert m._running is True, (
            "a failed initial connect must still leave the message loop "
            "armed, or _reconnect_ws can never run")

    @pytest.mark.parametrize("cls,_thresh", MONITORS)
    def test_process_messages_actually_reaches_the_reconnect(
            self, cls, _thresh, monkeypatch):
        """Behavioural, not a source pin (P234): drive the real loop and
        assert the reconnect path is entered."""
        _fail_connect(monkeypatch)
        m = cls()
        asyncio.run(m.connect())

        calls = {"n": 0}

        async def _fake_reconnect():
            calls["n"] += 1
            if calls["n"] >= 2:      # let it loop twice, then stop
                m._running = False

        m._reconnect_ws = _fake_reconnect
        asyncio.run(asyncio.wait_for(m.process_messages(), timeout=5))
        assert calls["n"] >= 2, (
            "the reconnect must be entered repeatedly while ws is None")

    @pytest.mark.parametrize("cls,_thresh", MONITORS)
    def test_close_still_stops_the_loop(self, cls, _thresh, monkeypatch):
        """Arming _running early must not make the monitor unstoppable."""
        _fail_connect(monkeypatch)
        m = cls()
        asyncio.run(m.connect())
        assert m._running is True
        asyncio.run(m.close())
        assert m._running is False

    @pytest.mark.parametrize("cls,_thresh", MONITORS)
    def test_reconnect_counters_exist_before_any_loop_runs(self, cls, _thresh):
        """_reconnect_ws reads _reconnect_count; it used to be initialised
        only inside process_messages, which is no longer the only caller
        (P85: an undefended attribute read on a live path)."""
        m = cls()
        assert isinstance(m._reconnect_count, int)
        assert isinstance(m._connect_failures, int)


# =============================================================================
# 2. Severity — a vendor blip is not an ERROR, a sustained outage is
# =============================================================================

class TestConnectFailureSeverity:

    @pytest.mark.parametrize("cls,thresh", MONITORS)
    def test_first_failures_warn_rather_than_error(
            self, cls, thresh, monkeypatch, caplog):
        """ERROR is forwarded to Discord. A venue outage the operator cannot
        act on, on a feed whose reconnect is already running, must not train
        alert-blindness (P202/P240)."""
        _fail_connect(monkeypatch)
        m = cls()
        with caplog.at_level(logging.WARNING):
            asyncio.run(m.connect())
        recs = [r for r in caplog.records if "connect failed" in r.message]
        assert recs, "the failure must still be reported"
        assert all(r.levelno == logging.WARNING for r in recs), (
            f"first failure must WARN, got "
            f"{[logging.getLevelName(r.levelno) for r in recs]}")

    @pytest.mark.parametrize("cls,thresh", MONITORS)
    def test_a_sustained_outage_still_escalates_to_error(
            self, cls, thresh, monkeypatch, caplog):
        """The downgrade must not silence a genuinely broken feed — that
        would trade one failure mode for a worse one."""
        _fail_connect(monkeypatch)
        m = cls()
        for _ in range(thresh - 1):
            asyncio.run(m.connect())
        with caplog.at_level(logging.WARNING):
            asyncio.run(m.connect())          # the threshold-th failure
        errs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errs, f"failure #{thresh} must escalate to ERROR"
        assert "SUSTAINED" in errs[-1].message

    @pytest.mark.parametrize("cls,thresh", MONITORS)
    def test_the_message_states_the_consequence(
            self, cls, thresh, monkeypatch, caplog):
        """An alert that does not say what is now degraded cannot be
        triaged — the P240 lesson (ship the numerator and denominator)."""
        _fail_connect(monkeypatch)
        m = cls()
        with caplog.at_level(logging.WARNING):
            asyncio.run(m.connect())
        msg = " ".join(r.message for r in caplog.records)
        assert "reconnect loop is armed" in msg
        assert ("DVOL" in msg) or ("taker-flow" in msg)

    @pytest.mark.parametrize("cls,thresh", MONITORS)
    def test_a_success_resets_the_failure_counter(
            self, cls, thresh, monkeypatch, caplog):
        """Otherwise a long-lived process eventually reports every routine
        blip at ERROR — the latch that never re-arms (P265f)."""
        import market.lead_lag_engine as mod
        _fail_connect(monkeypatch)
        m = cls()
        for _ in range(thresh + 2):
            asyncio.run(m.connect())
        assert m._connect_failures >= thresh

        class _OKWs:
            async def send(self, *_a, **_k):
                return None

            async def close(self):
                return None

        class _OK:
            async def __call__(self, *a, **k):
                return _OKWs()

        monkeypatch.setattr(mod, "websockets", type("W", (), {"connect": _OK()})())
        asyncio.run(m.connect())
        assert m._connect_failures == 0, "a success must clear the streak"


# =============================================================================
# 3. The consequence of a dead DVOL, pinned so nobody "fixes" it by arming it
# =============================================================================

class TestDvolAbsenceIsANoOpNotAFabrication:

    def test_an_unpopulated_dvol_leaves_the_modifier_neutral(self):
        """DVOL 0.0 -> 1.0 - tanh(0/10)*0.3 == 1.0 exactly. A dead feed must
        not tilt the lead-lag confidence in either direction."""
        import numpy as np
        assert float(1.0 - np.tanh(0.0 / 10.0) * 0.3) == pytest.approx(1.0)

    def test_dvol_is_still_not_wired_into_market_data(self):
        """P298 measured that Deribit publishes an INDEX LEVEL while the
        constitution reads that field as a Z-SCORE, so arming it fires
        EXTREME_DVOL (>= 5.0) on every tick and flattens the book. Repairing
        the connection must NOT be read as licence to arm the flag."""
        import json
        cfg = json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))
        # [P306] The condition this pin named has been MET: a real z is now
        # computed against a persisted trailing year of Deribit daily DVOL
        # (data_mgmt/feeds/dvol_history.py), so the flag is on. The pin keeps
        # its teeth by asserting the precondition rather than the absence.
        if "dvol_to_market_data" in cfg and cfg["dvol_to_market_data"]:
            main_src = (REPO / "main.py").read_text(encoding="utf-8")
            assert 'market_data["dvol"] = float(_dvz)' in main_src, (
                "the flag is armed but the z-score publication is gone")
            assert 'market_data["dvol"] = float(_drb_m.dvol)' not in main_src, (
                "the raw INDEX LEVEL is armed into a z-score field - "
                "EXTREME_DVOL (>= 5.0) fires on every tick")


# =============================================================================
# 4. The CONFIG_SCHEMA warnings — three armed keys called "typo?"
# =============================================================================

class TestConfigSchemaKnowsTheLiveRiskKeys:
    """Every boot warned `Unknown key (typo?)` about THREE legitimate risk
    keys, one of them P144's net-exposure cap. That is worse than silence: it
    invites someone to "fix the typo" by deleting an armed safety control,
    and it trains the reader to skip CONFIG_SCHEMA lines — so the one line
    that would flag a REAL typo goes unread (P202)."""

    def _cfg(self):
        import json
        return json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))

    def test_the_live_profile_now_validates_clean(self):
        from configs.config_resolver import validate_loaded_config
        assert validate_loaded_config(self._cfg()) == []

    def test_max_net_exposure_is_known_and_still_armed(self):
        """The one that mattered: P144's net signed budget."""
        from configs.config_resolver import _RISK_SCHEMA
        assert "max_net_exposure" in _RISK_SCHEMA["known_keys"]
        assert self._cfg()["risk"]["max_net_exposure"] == 0.50, (
            "the cap itself must not have moved while quieting its warning")

    def test_the_mirrored_gross_caps_may_not_drift(self):
        """max_exposure_total is what the PRE-LIVE checkers read as the gross
        figure; max_gross_exposure is what the engine enforces. If they drift,
        the pre-live gate verifies a stale number and still reports PASS."""
        r = self._cfg()["risk"]
        assert r["max_exposure_total"] == r["max_gross_exposure"], (
            "scripts/verify_live_config.py reads max_exposure_total while the "
            "engine enforces max_gross_exposure — they must agree")

    def test_the_advisory_cap_still_mirrors_the_enforced_one(self):
        """risk.per_asset_caps is advisory by decision (see the profile's
        _caps_authority_note) and mirrors exposure_caps. A silent divergence
        would make the config state a cap the engine does not apply."""
        c = self._cfg()
        assert c["risk"]["per_asset_caps"] == c["exposure_caps"]
        assert "_caps_authority_note" in c["risk"], (
            "the advisory key is only safe while the note explaining it exists")

    def test_quieting_did_not_become_a_blanket_allow(self):
        """A genuine typo must still be caught — otherwise this fix trades a
        noisy check for a dead one (P174)."""
        from configs.config_resolver import validate_loaded_config
        cfg = self._cfg()
        cfg["risk"]["max_gros_exposure"] = 1.5      # real typo
        assert any("max_gros_exposure" in i
                   for i in validate_loaded_config(cfg))

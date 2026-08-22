"""[P378] A live 403 exposed a severity gap on the only venue that trades.

THE INCIDENT (operator pasted it from production):

    2026-08-22 06:12:20 coinbase.RESTClient  ERROR  HTTP Error: 403 Forbidden
      {"error":"PERMISSION_DENIED","error_details":"User does not have access
       to portfolio"}
    2026-08-22 06:12:20 exchange.coinbase_sleeve WARNING [COINBASE_SLEEVE]
      reconcile failed: HTTPError: 403 ...

It happened ONCE, recovered on the next cycle, and the sleeve behaved
correctly throughout: `_reconcile_ok` went False, `manage_to_signal` returned
SKIPPED_STALE, and nothing was traded off a stale snapshot (P141). Today's
event was not a defect and this file does not claim otherwise.

WHAT IT EXPOSED. The handler is a blanket `except Exception` -> WARNING, so a
403 (your key lost access; never self-heals) and a 502 (venue blip; clears on
its own) are reported identically. Had that 403 PERSISTED, then with the book
holding ETH 3ct + SOL 2ct:

  * every 4H manage returns SKIPPED_STALE, logged at INFO (main.py:~24126);
  * P329's escalation only fires when FastRiskTick actually TRIGGERS, so a
    quiet market escalates nothing at all;
  * the operator's only signal is a WARNING line among thousands.

A sleeve holding positions it cannot see, exit or flip, reporting at INFO, is
the state P141/P195 exist to prevent — and the severity was too LOW to act on,
which is P202/P240's rule pointed the other way.

THE LOAD-BEARING DECISION, and it is the opposite of what "adopt the
classifier" naively implies: P345's contract says PERMANENT suppresses for the
PROCESS. Wiring that suppression here would have turned today's one-cycle blip
into a permanently blind sleeve. So the class is used for SEVERITY and for the
message only; the retry cadence is untouched. That is pinned first below,
because it is the property most likely to be "tidied" into a bug later.
"""

import logging
import time

import pytest

from exchange.coinbase_sleeve import (
    RECONCILE_BLIND_ALERT_SEC,
    RECONCILE_BLIND_REALERT_SEC,
    CoinbaseSleeve,
    _http_status,
)


# ==========================================================================
# Fixtures
# ==========================================================================
class _Resp:
    def __init__(self, code):
        self.status_code = code


class _Boom(Exception):
    """An SDK-shaped failure: carries `.response.status_code` like requests'."""

    def __init__(self, msg, code=None):
        super().__init__(msg)
        if code is not None:
            self.response = _Resp(code)


class _Client:
    """Counts venue reads so 'was the next attempt suppressed?' is a fact."""

    def __init__(self, owner):
        self.owner = owner

    def list_futures_positions(self, *a, **k):
        self.owner.calls += 1
        if self.owner.raises is not None:
            raise self.owner.raises
        return {"positions": []}


class _Adapter:
    def __init__(self, raises=None):
        self.raises = raises
        self.calls = 0
        self._client = _Client(self)

    def is_connected(self):
        return True


def _sleeve(adapter, held=None):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = adapter
    s.assets = ("BTC", "ETH", "SOL")
    s._pid_to_asset = {}
    s._last_positions = held or {}
    s._reconcile_ok = True
    s._reconcile_blind_since = None
    s._reconcile_last_alert_at = None
    return s


def _fail(sleeve, monkeypatch, exc, at):
    """Drive one failed reconcile at wall-clock `at`."""
    monkeypatch.setattr(time, "time", lambda: at)
    sleeve._adapter.raises = exc
    try:
        return sleeve.reconcile_positions()
    except Exception:
        return None


HELD = {"ETH": {"signed_contracts": 3.0}, "SOL": {"signed_contracts": 2.0}}


# ==========================================================================
# 1. THE PROPERTY THAT MUST NOT BE "TIDIED" AWAY
# ==========================================================================
class TestReconcileIsNeverSuppressed:
    """A failure to READ is not evidence the next read fails — and this
    reader guards live positions, so suppression is the dangerous direction
    (P329, which cost 23 minutes of a disarmed watchdog)."""

    def test_a_403_does_not_stop_the_next_attempt(self, monkeypatch):
        """The exact incident. PERMANENT class, and we still retry."""
        a = _Adapter()
        s = _sleeve(a, HELD)
        exc = _Boom("403 Client Error: Forbidden PERMISSION_DENIED", 403)
        for i in range(5):
            _fail(s, monkeypatch, exc, 1000.0 + i)
        assert a.calls == 5, (
            "the venue was called fewer times than we asked — something is "
            "suppressing reconcile after a PERMANENT-class failure. That "
            "converts a transient 403 into a permanently blind sleeve."
        )

    def test_the_policys_suppression_is_deliberately_not_consulted(self):
        """P345 would have us back off for the process on a 403. Here that is
        wrong, so the handler must not read the suppression fields at all."""
        import inspect

        src = inspect.getsource(CoinbaseSleeve.reconcile_positions)
        for banned in ("retry_not_before", ".suppresses", "retry_after_sec"):
            assert banned not in src, (
                f"reconcile_positions consults {banned!r}. Suppression here "
                f"means a blind sleeve holding unmanageable positions."
            )

    def test_last_known_snapshot_is_still_returned(self, monkeypatch):
        """Unchanged behaviour: callers get last-known and decide for
        themselves (manage_to_signal refuses; that is P141's job, not ours)."""
        s = _sleeve(_Adapter(), HELD)
        out = _fail(s, monkeypatch, _Boom("502 Server Error", 502), 1000.0)
        assert out == HELD
        assert s._reconcile_ok is False


# ==========================================================================
# 2. Severity tracks what the operator can DO
# ==========================================================================
class TestSeverity:
    def test_one_blip_stays_a_warning(self, monkeypatch, caplog):
        """Today's 403 was a single cycle. If that alone screamed, the alert
        would be wallpaper within a week (P202/P303) — and the whole value of
        this change is that a REAL one is legible."""
        s = _sleeve(_Adapter(), HELD)
        with caplog.at_level(logging.DEBUG):
            _fail(s, monkeypatch, _Boom("403 Forbidden", 403), 1000.0)
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warns and "class=PERMANENT" in warns[0].message
        assert "status=403" in warns[0].message

    def test_sustained_blindness_while_holding_is_CRITICAL(self, monkeypatch, caplog):
        """The state that matters: positions we cannot see, exit or flip."""
        s = _sleeve(_Adapter(), HELD)
        exc = _Boom("403 Forbidden PERMISSION_DENIED", 403)
        _fail(s, monkeypatch, exc, 1000.0)
        with caplog.at_level(logging.DEBUG):
            _fail(s, monkeypatch, exc, 1000.0 + RECONCILE_BLIND_ALERT_SEC + 1)
        crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(crit) == 1, [r.message for r in caplog.records]
        m = crit[0].message
        assert "ETH" in m and "SOL" in m, "must name what is exposed"
        assert "SKIPPED_STALE" in m, "must name the consequence, not just the fact"
        assert "does NOT self-heal" in m, "a 403 needs an operator, and must say so"

    def test_sustained_blindness_while_FLAT_is_only_an_error(self, monkeypatch, caplog):
        """Nothing is at risk on a flat book — it is opportunity cost, not
        exposure. Crying CRITICAL here is how the CRITICAL above gets ignored."""
        s = _sleeve(_Adapter(), {})
        exc = _Boom("403 Forbidden", 403)
        _fail(s, monkeypatch, exc, 1000.0)
        with caplog.at_level(logging.DEBUG):
            _fail(s, monkeypatch, exc, 1000.0 + RECONCILE_BLIND_ALERT_SEC + 1)
        assert not [r for r in caplog.records if r.levelno == logging.CRITICAL]
        errs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errs) == 1 and "FLAT" in errs[0].message

    def test_a_transient_5xx_is_labelled_as_such(self, monkeypatch, caplog):
        """The distinction the blanket handler could not make: a 502 may clear
        on its own, a 403 may not."""
        s = _sleeve(_Adapter(), HELD)
        exc = _Boom("502 Server Error: Bad Gateway", 502)
        _fail(s, monkeypatch, exc, 1000.0)
        with caplog.at_level(logging.DEBUG):
            _fail(s, monkeypatch, exc, 1000.0 + RECONCILE_BLIND_ALERT_SEC + 1)
        crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(crit) == 1
        assert "class=TRANSIENT" in crit[0].message
        assert "may clear on its own" in crit[0].message

    def test_realerts_are_rate_limited(self, monkeypatch, caplog):
        """reconcile runs every ~30s; one line per failure would bury the
        first one (P329b: ERROR exactly once per sustained streak)."""
        s = _sleeve(_Adapter(), HELD)
        exc = _Boom("403 Forbidden", 403)
        _fail(s, monkeypatch, exc, 1000.0)
        t = 1000.0 + RECONCILE_BLIND_ALERT_SEC + 1
        with caplog.at_level(logging.DEBUG):
            for i in range(40):                       # ~20 min of 30s cycles
                _fail(s, monkeypatch, exc, t + i * 30.0)
        crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(crit) == 1, f"{len(crit)} CRITICALs in 20 min of retries"

    def test_it_realerts_eventually(self, monkeypatch, caplog):
        """...but a still-blind sleeve must not go quiet forever either."""
        s = _sleeve(_Adapter(), HELD)
        exc = _Boom("403 Forbidden", 403)
        _fail(s, monkeypatch, exc, 1000.0)
        t = 1000.0 + RECONCILE_BLIND_ALERT_SEC + 1
        _fail(s, monkeypatch, exc, t)
        with caplog.at_level(logging.DEBUG):
            _fail(s, monkeypatch, exc, t + RECONCILE_BLIND_REALERT_SEC + 1)
        assert [r for r in caplog.records if r.levelno == logging.CRITICAL]


# ==========================================================================
# 3. Recovery resets — or a long-lived process drifts (P265f/P329)
# ==========================================================================
class TestRecovery:
    def test_recovery_is_announced_and_state_is_reset(self, monkeypatch, caplog):
        s = _sleeve(_Adapter(), HELD)
        exc = _Boom("403 Forbidden", 403)
        _fail(s, monkeypatch, exc, 1000.0)
        _fail(s, monkeypatch, exc, 1000.0 + RECONCILE_BLIND_ALERT_SEC + 1)
        assert s._reconcile_blind_since is not None

        s._adapter.raises = None
        monkeypatch.setattr(time, "time", lambda: 3000.0)
        with caplog.at_level(logging.DEBUG):
            s.reconcile_positions()
        assert s._reconcile_blind_since is None
        assert s._reconcile_last_alert_at is None
        assert any("RECOVERED" in r.message for r in caplog.records)

    def test_a_healthy_reconcile_is_silent(self, monkeypatch, caplog):
        """The happy path must not gain a per-tick line."""
        s = _sleeve(_Adapter(), {})
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        with caplog.at_level(logging.DEBUG):
            s.reconcile_positions()
        assert not [r for r in caplog.records
                    if "RECOVERED" in r.message or "reconcile failed" in r.message]


# ==========================================================================
# 4. The status extractor
# ==========================================================================
class TestHttpStatus:
    def test_prefers_the_response_object(self):
        assert _http_status(_Boom("nothing numeric here", 403)) == 403

    def test_falls_back_to_the_message(self):
        """The incident line carries the code only as text."""
        assert _http_status(Exception("403 Client Error: Forbidden")) == 403
        assert _http_status(Exception("502 Server Error: Bad Gateway")) == 502

    def test_unknown_is_None_not_a_guess(self):
        """None classifies TRANSIENT, which is the safe direction: it warns
        rather than asserting a credentials problem nobody measured (P2)."""
        assert _http_status(Exception("Connection reset by peer")) is None

    @pytest.mark.parametrize("msg", [
        "failed at 2026-08-22 06:12:20",     # a timestamp
        "timeout after 1500ms",              # a duration CONTAINING 500
        "request id 7403abc failed",         # an id CONTAINING 403
    ])
    def test_a_number_that_merely_contains_a_status_is_not_one(self, msg):
        """Word boundaries matter, and the discriminating cases are the ones
        where the digits are EMBEDDED: without \b, `1500ms` reads as a 500
        and `7403abc` as a 403 — the latter would assert a credentials
        problem, at CRITICAL, from a log id. My first version of this test
        used only a timestamp, which contains no 4xx/5xx digits at all and so
        could not fail either way (found by its own falsification probe)."""
        assert _http_status(Exception(msg)) is None


# ==========================================================================
# 5. Anti-vacuity (P174): the guard must be able to fire
# ==========================================================================
def test_the_escalation_is_reachable_at_the_shipped_threshold():
    """A threshold nothing can cross is a check that cannot fail. reconcile is
    attempted at least every ~30s by the FastRiskTick loop, so 15 min is ~30
    attempts — reachable long before the next 4H tick."""
    assert 0 < RECONCILE_BLIND_ALERT_SEC <= 4 * 3600, (
        "an alert threshold at or beyond the 4H tick could be crossed by a "
        "single missed tick, which is not evidence of anything"
    )
    assert RECONCILE_BLIND_REALERT_SEC > RECONCILE_BLIND_ALERT_SEC


@pytest.mark.parametrize("code,expect", [(403, "PERMANENT"), (401, "PERMANENT"),
                                         (502, "TRANSIENT"), (500, "TRANSIENT")])
def test_classification_matches_the_shared_policy(code, expect):
    """Read from P345's module rather than restated here (P172), so the two
    cannot drift into disagreeing about what a 403 means."""
    from infra.failure_policy import classify_external_failure

    assert classify_external_failure(
        status=code, message=f"{code} error").failure_class.name == expect

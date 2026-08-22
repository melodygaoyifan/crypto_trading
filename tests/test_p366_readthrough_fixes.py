"""
[P366] Three defects from the 2026-08-21 read-through.

1. `sleeve_fast_risk_action` reported the literal "EXITED" for every outcome
   of `execute_target`, which returns SEVEN statuses of which only OK is an
   exit. Measured live 2026-08-21: ETH held 3 contracts and logged 29
   `EXIT_ONLY -> EXITED (execute_target(0) -> SKIPPED_STALE)` lines during a
   venue 502 storm, alongside 16 genuine `-> OK` ones. Four consumers branch
   on that status:
     - on_venue_readable()  cleared the P329 unreadable-streak on a tick where
                            the venue was NOT readable
     - on_reduce_executed() re-anchored the depth baseline on a flatten that
                            did not happen
     - the P232 re-entry cooldown armed for a non-event
     - on_exit_failed()     was UNREACHABLE, because its `elif` needs a status
                            the wrapper never produced — so P110's backoff and
                            P329's escalation never ran on a genuinely
                            un-actioned emergency exit
   The sibling 4H path already reads `_m_res.get("status")` properly; the
   correct pattern existed next door.

2. `_wavelet_buffers` is the only rolling buffer with no state file, and P354
   had just slowed its feed 424x (34s -> the 4H decision tick) to restore
   P164's causal-wavelet parity. That made the warmup 32 HOURS before
   denoising starts and ~42 DAYS to fill 256. Measured live: ofi 42/42 and
   depth 120/120 restored from disk, wavelet holding 4 samples after 4
   decision ticks. Both consumers are SHADOW, so no live order changes.

3. The sleeve's "cannot verify the resting-order book is clear" refusal had no
   sustained-failure escalation: 28 consecutive refusals on one ETH order
   rendered identically to a single blip, at WARNING (not forwarded). P329
   built exactly this split for the sibling watchdog.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import deque
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests._guard_pins import assert_text_pin  # noqa: E402


# ---------------------------------------------------------------------------
# Fix 1 - the status collapse
# ---------------------------------------------------------------------------

class _FakeSleeve:
    """Minimal sleeve for driving the real helper (P206: unit-testable
    without the runner or an event loop of its own)."""

    def __init__(self, venue_status: str, contracts: int = 3):
        self.venue_status = venue_status
        self._contracts = contracts
        self._reconcile_ok = True
        self.targets: list = []

    def reconcile_positions(self):
        return {}

    def signed_contracts(self, asset):
        return self._contracts

    async def execute_target(self, asset, target, urgent=False):
        self.targets.append(target)
        return {"status": self.venue_status, "asset": asset}


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestTheClassifierTruthTable:
    """`sleeve_exit_status` is pure so the whole table is testable."""

    @pytest.mark.parametrize("venue,expected", [
        ("OK", "EXITED"),               # the ONLY status that is an exit
        ("NOOP", "FLAT"),               # delta == 0 <=> already flat
        ("SKIPPED_STALE", "SKIPPED_STALE"),
        ("NOT_READY", "SKIPPED_STALE"),  # also "could not reach the venue"
        ("BLOCKED", "EXIT_BLOCKED"),    # OUR policy refused, venue readable
        ("FAILED", "EXIT_FAILED"),      # venue rejected the order
        ("ERROR", "EXIT_FAILED"),
    ])
    def test_each_venue_status_maps_to_its_own_outcome(self, venue, expected):
        import main
        assert main.sleeve_exit_status(venue, "EXITED") == expected

    def test_the_success_path_is_unchanged(self):
        """The fix must not move the one case that was already right."""
        import main
        assert main.sleeve_exit_status("OK", "EXITED") == "EXITED"
        assert main.sleeve_exit_status("OK", "REDUCED") == "REDUCED"

    @pytest.mark.parametrize("venue", ["WEIRD_NEW_STATUS", None, "", "  "])
    def test_an_unrecognised_status_is_never_a_silent_success(self, venue):
        """The fail direction. A new venue status must fail loudly rather than
        read as a completed emergency exit — that IS the defect being fixed."""
        import main
        got = main.sleeve_exit_status(venue, "EXITED")
        assert got == "EXIT_FAILED"
        assert got not in ("EXITED", "REDUCED", "FLAT")

    def test_case_and_whitespace_do_not_manufacture_an_exit(self):
        import main
        assert main.sleeve_exit_status(" ok ", "EXITED") == "EXITED"


class TestTheLiveIncident:
    """Drive the REAL helper, not the classifier (P234: a pin on the table
    proves the table is right, not that anything consults it)."""

    def test_a_refused_exit_no_longer_reports_EXITED(self):
        """The regression: ETH, 3 contracts, execute_target refuses on a stale
        snapshot (P141, correct) — 29 times on 2026-08-21."""
        import main
        s = _FakeSleeve("SKIPPED_STALE", contracts=3)
        st, why = _run(main.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "SKIPPED_STALE", (
            "a refused emergency exit reported as EXITED cleared the P329 "
            "streak, refreshed the depth baseline and armed the re-entry "
            "cooldown, while on_exit_failed stayed unreachable")
        assert "SKIPPED_STALE" in why

    def test_a_real_exit_still_reports_EXITED(self):
        import main
        s = _FakeSleeve("OK", contracts=3)
        st, _ = _run(main.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "EXITED"
        assert s.targets == [0], "an EXIT_ONLY must still target flat"

    def test_a_policy_blocked_exit_is_distinguishable(self):
        import main
        s = _FakeSleeve("BLOCKED", contracts=3)
        st, _ = _run(main.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "EXIT_BLOCKED"

    def test_a_venue_rejected_exit_is_structural(self):
        import main
        s = _FakeSleeve("FAILED", contracts=3)
        st, _ = _run(main.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "EXIT_FAILED"

    def test_the_reduce_path_has_the_same_fix(self):
        """A BLOCKED reduce reported as REDUCED refreshed the depth baseline,
        masking the very depth collapse the trigger fired on."""
        import main
        s = _FakeSleeve("BLOCKED", contracts=4)
        st, _ = _run(main.sleeve_fast_risk_action(s, "ETH", "REDUCE_50", True))
        assert st == "EXIT_BLOCKED"
        s_ok = _FakeSleeve("OK", contracts=4)
        st_ok, _ = _run(main.sleeve_fast_risk_action(s_ok, "ETH", "REDUCE_50", True))
        assert st_ok == "REDUCED"
        assert s_ok.targets == [2]

    def test_the_helpers_own_staleness_check_is_untouched(self):
        import main
        s = _FakeSleeve("OK", contracts=3)
        s._reconcile_ok = False
        st, _ = _run(main.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "SKIPPED_STALE"

    def test_flat_still_short_circuits_before_any_order(self):
        import main
        s = _FakeSleeve("OK", contracts=0)
        st, _ = _run(main.sleeve_fast_risk_action(s, "ETH", "EXIT_ONLY", True))
        assert st == "FLAT"
        assert s.targets == [], "a flat asset must not reach execute_target"


class TestTheCallerActsOnTheNewStatuses:
    """The classifier is worthless if the caller's branches ignore it."""

    def _caller_src(self) -> str:
        import inspect
        import main
        return inspect.getsource(
            main.HMATSProductionRunner._handle_fast_risk_action)

    def test_a_structural_failure_does_not_clear_the_unreadable_streak(self):
        src = self._caller_src()
        assert_text_pin(
            src, '("SKIPPED_STALE", "ERROR", "EXIT_FAILED",', why=(
                "EXIT_FAILED subsumes the old helper-level ERROR plus a "
                "venue-rejected order; an exception on the way to the venue "
                "is not evidence the venue was readable"))

    def test_a_structural_failure_reaches_the_P110_failure_detector(self):
        src = self._caller_src()
        assert_text_pin(
            src, '("ERROR", "SKIPPED_STALE", "EXIT_FAILED")', why=(
                "before P366 this elif could not be reached when "
                "execute_target refused internally, so P110's backoff and "
                "P329's escalation never ran on an un-actioned exit"))

    def test_only_a_real_exit_arms_the_reentry_cooldown(self):
        src = self._caller_src()
        assert_text_pin(
            src, 'if _frs_st == "EXITED":', why=(
                "the P232 cooldown must arm on a flatten that HAPPENED; with "
                "the old collapse a refused exit armed it for a non-event"))

    def test_only_a_real_action_refreshes_the_depth_baseline(self):
        src = self._caller_src()
        assert_text_pin(
            src, '("EXITED", "REDUCED", "REDUCE_NOOP")', why=(
                "refreshing the baseline on a refused reduce masks the depth "
                "collapse the trigger fired on"))

    def test_a_policy_block_is_in_neither_failure_mechanism(self):
        """EXIT_BLOCKED is deliberately neither transient nor structural:
        labelling it 'could not READ the venue' would make a false CRITICAL
        (P155), and the 30-min structural backoff would re-create the exact
        P329 disarming during a venue outage. Its repetition is escalated at
        the source instead (fix 3)."""
        src = self._caller_src()
        i = src.index("on_exit_failed(")
        window = src[max(0, i - 1500):i + 500]
        assert "EXIT_BLOCKED" not in window


# ---------------------------------------------------------------------------
# Fix 2 - the wavelet buffer had no state file
# ---------------------------------------------------------------------------

class _BufHost:
    """The two persistence methods under test, on a bare host — no network,
    no pipeline construction."""

    _BUFFER_MAX_AGE_SEC = 7 * 24 * 3600.0

    def __init__(self, assets=("BTC", "ETH", "SOL"), feats=("rsi_14", "atr_14")):
        self._wavelet_buffers = {
            a: {f: deque(maxlen=256) for f in feats} for a in assets}


def _bind(host):
    """Bind the real pipeline methods to the bare host."""
    from data_mgmt.market_data_pipeline import MarketDataPipeline
    host._wavelet_flat_view = MarketDataPipeline._wavelet_flat_view.__get__(host)
    host._restore_rolling_buffer = (
        MarketDataPipeline._restore_rolling_buffer.__get__(host))
    host._persist_rolling_buffers = (
        MarketDataPipeline._persist_rolling_buffers.__get__(host))
    return host


@pytest.fixture(autouse=True)
def _isolated_warmup_dir(tmp_path, monkeypatch):
    """P294: CONSTRUCT the state, never inherit it from the machine."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    import strategies._warmup_state as ws
    import importlib
    importlib.reload(ws)
    yield
    importlib.reload(ws)


class TestTheWaveletBufferSurvivesARestart:

    def test_the_flat_view_exposes_the_same_deque_objects(self):
        """Identity matters: the restore helper appends IN PLACE, so a view
        that copied would restore into throwaway deques and report success."""
        h = _bind(_BufHost())
        view = h._wavelet_flat_view()
        assert view["BTC::rsi_14"] is h._wavelet_buffers["BTC"]["rsi_14"]

    def test_the_composite_key_keeps_assets_and_features_apart(self):
        h = _bind(_BufHost())
        view = h._wavelet_flat_view()
        assert len(view) == 3 * 2
        assert "BTC::rsi_14" in view and "ETH::rsi_14" in view

    def test_a_decision_tick_append_survives_a_restart(self):
        """The regression, end to end through the real save/load."""
        h = _bind(_BufHost())
        for i in range(12):
            h._wavelet_buffers["BTC"]["rsi_14"].append(float(i))
        h._wavelet_dirty = True
        h._persist_rolling_buffers()

        fresh = _bind(_BufHost())
        assert len(fresh._wavelet_buffers["BTC"]["rsi_14"]) == 0
        fresh._restore_rolling_buffer("wavelet_buffers",
                                      fresh._wavelet_flat_view())
        assert list(fresh._wavelet_buffers["BTC"]["rsi_14"]) == [
            float(i) for i in range(12)]

    def test_restoring_reaches_the_denoise_threshold(self):
        """The point of the fix: >= 8 samples is what makes the denoise run at
        all, and at 6 decision ticks/day a cold start needs 32 hours."""
        h = _bind(_BufHost())
        for i in range(8):
            h._wavelet_buffers["ETH"]["atr_14"].append(float(i))
        h._wavelet_dirty = True
        h._persist_rolling_buffers()
        fresh = _bind(_BufHost())
        fresh._restore_rolling_buffer("wavelet_buffers",
                                      fresh._wavelet_flat_view())
        assert len(fresh._wavelet_buffers["ETH"]["atr_14"]) >= 8

    def test_it_is_written_only_when_a_decision_tick_appended(self):
        """The persist call runs on every ~34s pass; this view only changes
        6x/day. Writing it every pass is pure I/O."""
        import strategies._warmup_state as ws
        h = _bind(_BufHost())
        h._wavelet_buffers["BTC"]["rsi_14"].append(1.0)
        h._persist_rolling_buffers()          # not dirty
        assert ws.load("wavelet_buffers") in ({}, None)
        h._wavelet_dirty = True
        h._persist_rolling_buffers()
        assert ws.load("wavelet_buffers")

    def test_a_failed_save_keeps_the_dirty_flag_so_it_retries(self, monkeypatch):
        """_wsave returns False WITHOUT raising; clearing on that would drop a
        decision tick's append for good."""
        import strategies._warmup_state as ws
        h = _bind(_BufHost())
        h._wavelet_buffers["BTC"]["rsi_14"].append(1.0)
        h._wavelet_dirty = True
        monkeypatch.setattr(ws, "save", lambda *a, **k: False)
        h._persist_rolling_buffers()
        assert h._wavelet_dirty is True

    def test_the_cold_start_message_no_longer_asserts_a_warmup_length(self, caplog):
        """It said '~20h' for every buffer — wrong by ~50x for this one.

        Asserted on the EMITTED line, not the source: the source now carries a
        comment quoting the old wording to explain the fix, and a substring
        scan would match its own explanation (P177, which this test hit on its
        first run). A behavioural check cannot be fooled that way (P234).
        """
        h = _bind(_BufHost())
        with caplog.at_level(logging.INFO):
            h._restore_rolling_buffer("wavelet_buffers", h._wavelet_flat_view())
        msgs = [r.getMessage() for r in caplog.records if "[BUFFER]" in r.getMessage()]
        assert msgs, "a cold start must announce itself, never return silently"
        assert not any("20h" in m for m in msgs), (
            "the message stated a warmup length it cannot know — it is "
            "per-buffer and this one is ~50x longer")

    def test_the_wavelet_buffer_is_in_the_persist_roster(self):
        import inspect
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        src = inspect.getsource(MarketDataPipeline._persist_rolling_buffers)
        assert "wavelet_buffers" in src

    def test_the_constructor_restores_it(self):
        import inspect
        from data_mgmt.market_data_pipeline import MarketDataPipeline
        src = inspect.getsource(MarketDataPipeline.__init__)
        assert_text_pin(
            src, '_restore_rolling_buffer("wavelet_buffers"', why=(
                "P354 slowed this deque 424x onto the one rolling buffer "
                "with no state file"))


# ---------------------------------------------------------------------------
# Fix 3 - sustained cancel-refusal escalation
# ---------------------------------------------------------------------------

class _RefuseHost:
    pass


def _refuse_host():
    from exchange.coinbase_sleeve import CoinbaseSleeve
    h = _RefuseHost()
    h._CANCEL_REFUSE_SUSTAINED = CoinbaseSleeve._CANCEL_REFUSE_SUSTAINED
    h._note_cancel_refusal = CoinbaseSleeve._note_cancel_refusal.__get__(h)
    h._note_cancel_ok = CoinbaseSleeve._note_cancel_ok.__get__(h)
    return h


class TestSustainedCancelRefusalIsLoud:

    def test_an_isolated_refusal_does_not_escalate(self, caplog):
        h = _refuse_host()
        with caplog.at_level(logging.WARNING):
            h._note_cancel_refusal("ETH", "cancel unconfirmed")
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    def test_a_sustained_refusal_escalates_once(self, caplog):
        """28 consecutive refusals on one ETH order rendered identically to a
        single blip on 2026-08-21."""
        h = _refuse_host()
        with caplog.at_level(logging.WARNING):
            for _ in range(h._CANCEL_REFUSE_SUSTAINED * 3):
                h._note_cancel_refusal("ETH", "cancel unconfirmed")
        errs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errs) == 1, (
            "escalate ONCE per streak — an ERROR every 35s is wallpaper and "
            "would bury the one line that matters (P202)")
        assert "SUSTAINED" in errs[0].getMessage()

    def test_the_escalation_names_the_consequence(self, caplog):
        h = _refuse_host()
        with caplog.at_level(logging.WARNING):
            for _ in range(h._CANCEL_REFUSE_SUSTAINED):
                h._note_cancel_refusal("ETH", "cancel unconfirmed")
        msg = [r for r in caplog.records if r.levelname == "ERROR"][0].getMessage()
        assert "emergency exits" in msg.lower() or "flatten" in msg.lower(), (
            "P240: an alert must say what it MEANS, not just that it happened")

    def test_a_verified_book_resets_the_streak(self, caplog):
        """Without a reset, isolated blips accumulate into a permanent
        SUSTAINED that no longer describes the present (P303/P265f)."""
        h = _refuse_host()
        for _ in range(h._CANCEL_REFUSE_SUSTAINED - 1):
            h._note_cancel_refusal("ETH", "x")
        h._note_cancel_ok("ETH")
        with caplog.at_level(logging.WARNING):
            h._note_cancel_refusal("ETH", "x")
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    def test_the_streak_is_per_asset(self):
        h = _refuse_host()
        for _ in range(h._CANCEL_REFUSE_SUSTAINED - 1):
            h._note_cancel_refusal("ETH", "x")
        h._note_cancel_refusal("SOL", "x")
        assert h._cancel_refuse_streak["SOL"] == 1
        assert h._cancel_refuse_streak["ETH"] == h._CANCEL_REFUSE_SUSTAINED - 1

    def test_the_counter_never_raises_into_the_order_path(self):
        """P85: fixtures and operator scripts build sleeves via object.__new__,
        and this sits on the live order path."""
        h = _refuse_host()
        h._cancel_refuse_streak = "not a dict"
        h._note_cancel_refusal("ETH", "x")   # must not raise
        h._note_cancel_ok("ETH")             # must not raise

    def test_both_refusal_paths_feed_the_counter(self):
        """The listing failure and the unconfirmed cancel both mean 'this
        asset cannot trade this tick'."""
        import inspect
        from exchange.coinbase_sleeve import CoinbaseSleeve
        src = inspect.getsource(CoinbaseSleeve._cancel_resting_orders)
        assert src.count("_note_cancel_refusal(") == 2
        assert "_note_cancel_ok(" in src

    def test_the_refusal_itself_is_unchanged(self):
        """The guard is correct and stays: an order we cannot see plus a new
        one is two live orders for one delta (P265/P287)."""
        import inspect
        from exchange.coinbase_sleeve import CoinbaseSleeve
        src = inspect.getsource(CoinbaseSleeve._cancel_resting_orders)
        assert src.count("return None") == 2

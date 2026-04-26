"""
[AP-14] HotRestartPersistence governor-state round-trip + restore tests.

Verifies the OC-5 fix: gambler / failure_memory / confidence_scorer / mode_ttl
state survives a crash-restart cycle so post-crash GamblerSizing cannot
burst-bet past its cooldown / weekly limit, and FailureMemory cannot forget
recent loss streaks.
"""
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infra.hot_restart_persistence import (
    HotRestartPersistence,
    PersistenceEvent,
    SystemState,
)


@pytest.fixture
def persistence(tmp_path):
    """Fresh SQLite-backed persistence per test."""
    db = tmp_path / "ap14_state.db"
    p = HotRestartPersistence(str(db))
    yield p
    p.clear()


@pytest.fixture
def fresh_state():
    return SystemState(system_mode="NORMAL", opportunity_trigger=None)


# ---------------------------------------------------------------------------
# SystemState round-trip — new AP-14 fields survive to_dict / from_dict
# ---------------------------------------------------------------------------

class TestSystemStateRoundTrip:
    def test_default_governor_fields_are_empty_dicts(self, fresh_state):
        assert fresh_state.gambler_state == {}
        assert fresh_state.failure_memory_state == {}
        assert fresh_state.confidence_scorer_state == {}
        assert fresh_state.mode_ttl_state == {}

    def test_to_dict_includes_governor_fields(self, fresh_state):
        d = fresh_state.to_dict()
        assert "gambler_state" in d
        assert "failure_memory_state" in d
        assert "confidence_scorer_state" in d
        assert "mode_ttl_state" in d

    def test_from_dict_restores_governor_fields(self, fresh_state):
        fresh_state.gambler_state = {"bet_timestamps": [1.0, 2.0, 3.0]}
        fresh_state.failure_memory_state = {"consecutive_failures": 4}
        fresh_state.confidence_scorer_state = {"confidence": {"momentum": {"confidence": 0.6}}}
        fresh_state.mode_ttl_state = {"BTC": {"entered_bar": 12, "trigger": "DRL_STRONG"}}

        d = fresh_state.to_dict()
        restored = SystemState.from_dict(d)

        assert restored.gambler_state == {"bet_timestamps": [1.0, 2.0, 3.0]}
        assert restored.failure_memory_state == {"consecutive_failures": 4}
        assert restored.confidence_scorer_state == {"confidence": {"momentum": {"confidence": 0.6}}}
        assert restored.mode_ttl_state == {"BTC": {"entered_bar": 12, "trigger": "DRL_STRONG"}}

    def test_from_dict_handles_old_db_without_governor_fields(self):
        """Forward-compat: DBs persisted before AP-14 lack the new keys.
        from_dict() must default them to {} rather than raising KeyError."""
        old_data = {
            "system_mode": "NORMAL",
            "opportunity_trigger": None,
            "positions": {},
            "intents": {},
            "stops": {},
            "metrics": {
                "account_balance": 0.0,
                "peak_equity": 0.0,
                "current_drawdown": 0.0,
                "drawdown_status": "NORMAL",
                "daily_pnl": 0.0,
                "daily_trades": 0,
                "last_trade_time": None,
            },
            "thesis_budget_state": {},
            # gambler_state / failure_memory_state / etc. intentionally absent
            "version": "3.3.0",
            "persisted_at": "",
            "persist_event": "",
        }
        state = SystemState.from_dict(old_data)
        assert state.gambler_state == {}
        assert state.failure_memory_state == {}
        assert state.confidence_scorer_state == {}
        assert state.mode_ttl_state == {}


# ---------------------------------------------------------------------------
# Gambler — crash-restart cooldown / weekly-limit survival
# ---------------------------------------------------------------------------

class TestGamblerCrashRestart:
    def test_gambler_state_survives_persist_load_restore(self, persistence, fresh_state):
        """The OC-5 invariant. Without persistence, post-crash _bet_timestamps
        resets to [] and the next OPPORTUNITY window can immediately fire."""
        from orchestration.gambler_sizing import GamblerSizing, GamblerSizingConfig

        # Simulate a runner with 2 recent bets (within 7-day window).
        gambler1 = GamblerSizing(GamblerSizingConfig(enabled=True))
        now = time.time()
        gambler1._bet_timestamps = [now - 3600, now - 1800]  # 1h ago, 30min ago

        persistence.set_gambler_sizing(gambler1)
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)

        # Simulate process restart: fresh persistence + fresh gambler.
        gambler2 = GamblerSizing(GamblerSizingConfig(enabled=True))
        assert gambler2._bet_timestamps == []  # baseline — would burst-bet without restore

        persistence2 = HotRestartPersistence(str(persistence.db_path))
        persistence2.set_gambler_sizing(gambler2)
        loaded = persistence2.load()
        assert loaded is not None
        results = persistence2.restore_governors(loaded)

        assert results["gambler"] is True
        assert len(gambler2._bet_timestamps) == 2
        # Cooldown protection now active again on the restarted process.
        assert gambler2.get_weekly_count() == 2

    def test_gambler_persist_with_no_registered_engine_is_noop(self, persistence, fresh_state):
        """If main.py never wires a gambler, persist() must not blow up."""
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)
        loaded = persistence.load()
        assert loaded.gambler_state == {}

    def test_gambler_old_bets_pruned_on_restore(self, persistence, fresh_state):
        """Bets older than 7 days must not survive. Otherwise an ancient
        weekly-limit-hit could permanently lock out the gambler."""
        from orchestration.gambler_sizing import GamblerSizing, GamblerSizingConfig

        gambler1 = GamblerSizing(GamblerSizingConfig(enabled=True))
        now = time.time()
        # 2 ancient (>7d), 1 recent
        gambler1._bet_timestamps = [now - 30 * 86400, now - 10 * 86400, now - 3600]

        persistence.set_gambler_sizing(gambler1)
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)

        gambler2 = GamblerSizing(GamblerSizingConfig(enabled=True))
        persistence2 = HotRestartPersistence(str(persistence.db_path))
        persistence2.set_gambler_sizing(gambler2)
        persistence2.restore_governors(persistence2.load())

        # to_dict() prunes; from_dict() also prunes — only the recent bet survives.
        assert len(gambler2._bet_timestamps) == 1


# ---------------------------------------------------------------------------
# Failure memory — caution-mode + consecutive-failure streak survival
# ---------------------------------------------------------------------------

class TestFailureMemoryCrashRestart:
    def test_failure_memory_state_survives_restart(self, persistence, fresh_state):
        from analytics.failure_memory import FailureAwareMetaMemory

        memory1 = FailureAwareMetaMemory()
        memory1._consecutive_failures = 4
        memory1._consecutive_false_breakouts = 2
        memory1._density_boost = 0.3
        memory1._tranche_delay = 1
        memory1._in_caution_mode = True

        persistence.set_failure_memory(memory1)
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)

        memory2 = FailureAwareMetaMemory()
        assert memory2._consecutive_failures == 0  # baseline

        persistence2 = HotRestartPersistence(str(persistence.db_path))
        persistence2.set_failure_memory(memory2)
        results = persistence2.restore_governors(persistence2.load())

        assert results["failure_memory"] is True
        assert memory2._consecutive_failures == 4
        assert memory2._consecutive_false_breakouts == 2
        assert memory2._density_boost == pytest.approx(0.3)
        assert memory2._tranche_delay == 1
        assert memory2._in_caution_mode is True


# ---------------------------------------------------------------------------
# Confidence scorer — per-strategy confidence survival
# ---------------------------------------------------------------------------

class TestConfidenceScorerCrashRestart:
    def test_confidence_state_survives_restart(self, persistence, fresh_state):
        from analytics.confidence_scorer import StrategyConfidenceScorer

        scorer1 = StrategyConfidenceScorer()
        # Pick an existing strategy slot and mutate its confidence.
        any_strat = next(iter(scorer1._confidence.keys()))
        scorer1._confidence[any_strat].confidence = 0.42

        persistence.set_confidence_scorer(scorer1)
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)

        scorer2 = StrategyConfidenceScorer()
        assert scorer2._confidence[any_strat].confidence != 0.42  # baseline

        persistence2 = HotRestartPersistence(str(persistence.db_path))
        persistence2.set_confidence_scorer(scorer2)
        results = persistence2.restore_governors(persistence2.load())

        assert results["confidence_scorer"] is True
        assert scorer2._confidence[any_strat].confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Mode TTL — runner._mode_ttl dict reference semantics
# ---------------------------------------------------------------------------

class TestModeTtlCrashRestart:
    def test_mode_ttl_dict_survives_restart(self, persistence, fresh_state):
        # Simulate runner._mode_ttl: dict keyed by asset.
        runner_ttl_1 = {
            "BTC": {"entered_bar": 1234, "trigger": "DRL_STRONG", "confidence": 0.78},
            "ETH": None,  # cleared TTL — should NOT persist
            "SOL": {"entered_bar": 1240, "trigger": "FUNDING_FLIP", "confidence": 0.65},
        }
        persistence.set_mode_ttl_dict(runner_ttl_1)
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)

        # Restart: fresh runner with empty TTL dict.
        runner_ttl_2 = {}
        persistence2 = HotRestartPersistence(str(persistence.db_path))
        persistence2.set_mode_ttl_dict(runner_ttl_2)
        results = persistence2.restore_governors(persistence2.load())

        assert results["mode_ttl"] is True
        # Restored in place — runner_ttl_2 is the same dict reference the
        # runner holds, so the runner sees the restored entries.
        assert "BTC" in runner_ttl_2
        assert runner_ttl_2["BTC"]["trigger"] == "DRL_STRONG"
        assert "SOL" in runner_ttl_2
        # None entries were filtered on persist — not restored.
        assert "ETH" not in runner_ttl_2

    def test_mode_ttl_restore_clears_stale_entries(self, persistence, fresh_state):
        """If the restarted runner re-init'd a stale TTL entry, restore() must
        wipe it before applying the persisted state — else the runner's
        self-recovery shim could collide with the restored TTL."""
        persistence.set_mode_ttl_dict({"BTC": {"entered_bar": 100, "trigger": "OLD"}})
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)

        # Restart runner with a stale entry already populated.
        runner_ttl_2 = {"ETH": {"entered_bar": 999, "trigger": "STALE_RESTART"}}
        persistence2 = HotRestartPersistence(str(persistence.db_path))
        persistence2.set_mode_ttl_dict(runner_ttl_2)
        persistence2.restore_governors(persistence2.load())

        # Stale ETH entry must be gone; persisted BTC entry present.
        assert "ETH" not in runner_ttl_2
        assert runner_ttl_2["BTC"]["trigger"] == "OLD"


# ---------------------------------------------------------------------------
# Restore-when-not-registered — no-op contract
# ---------------------------------------------------------------------------

class TestRestoreWithoutRegistration:
    def test_restore_governors_skips_unregistered_components(self, persistence, fresh_state):
        """A subset of governors may be wired in. restore_governors() must
        only touch the registered ones and return {} for the rest."""
        persistence.persist(fresh_state, PersistenceEvent.HEARTBEAT)
        results = persistence.restore_governors(persistence.load())
        # Nothing registered — empty results, no exception.
        assert results == {}

"""Tests for the three pattern-codification helpers introduced in P73:

  - infra.preserve_fresh_state.PreserveFreshState  (P69 pattern)
  - infra.shared_state_registry.SharedRegistry     (P68 pattern)
  - infra.classified_retry.classified_retry_run    (P68 pattern)
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import pytest

from infra.preserve_fresh_state import (
    PreserveFreshState, StaleStateError, StateStatus,
)
from infra.shared_state_registry import SharedRegistry, SharedHolder
from infra.classified_retry import (
    classified_retry_run, RetryAction, _ActionKind,
)


# ---------------------------------------------------------------------------
# PreserveFreshState
# ---------------------------------------------------------------------------

class TestPreserveFreshState:

    def test_initial_state_is_unavailable(self):
        s = PreserveFreshState[float](max_age_seconds=60.0, label="test")
        assert s.get_status() == StateStatus.UNINITIALIZED
        assert not s.is_valid()
        with pytest.raises(StaleStateError):
            s.get()
        val, ok = s.get_safe()
        assert val is None
        assert ok is False

    def test_successful_update_sets_valid(self):
        s = PreserveFreshState[float](max_age_seconds=60.0, label="test")
        ok, err = s.try_update(lambda: 42.0)
        assert ok is True
        assert err is None
        assert s.is_valid()
        assert s.get() == 42.0
        assert s.get_status() == StateStatus.VALID
        assert s.failure_count() == 0

    def test_p69_invariant_failure_within_window_keeps_valid(self):
        """Core P69 invariant: transient failure within max-age preserves
        the cached value."""
        s = PreserveFreshState[float](max_age_seconds=60.0, label="test")
        s.try_update(lambda: 100.0)  # establish VALID

        def boom():
            raise RuntimeError("transient network blip")

        ok, err = s.try_update(boom)
        assert ok is False
        assert "transient_keep_valid" in (err or "")
        # Critical: value still readable, status still VALID.
        assert s.is_valid()
        assert s.get() == 100.0
        assert s.failure_count() == 1

    def test_failure_outside_window_flips_unavailable(self):
        s = PreserveFreshState[float](max_age_seconds=0.05, label="test")
        s.try_update(lambda: 100.0)
        time.sleep(0.1)  # exceed max_age

        ok, err = s.try_update(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert ok is False
        assert err is not None
        assert "transient_keep_valid" not in err
        assert not s.is_valid()
        assert s.get_status() == StateStatus.UNAVAILABLE
        with pytest.raises(StaleStateError):
            s.get()

    def test_first_call_failure_flips_unavailable(self):
        """No prior VALID state → any failure → UNAVAILABLE."""
        s = PreserveFreshState[float](max_age_seconds=60.0, label="test")
        ok, err = s.try_update(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert ok is False
        assert "transient_keep_valid" not in (err or "")
        assert s.get_status() == StateStatus.UNAVAILABLE

    def test_success_clears_failure_count(self):
        s = PreserveFreshState[float](max_age_seconds=60.0, label="test")
        s.try_update(lambda: 1.0)
        s.try_update(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert s.failure_count() == 1
        s.try_update(lambda: 2.0)
        assert s.failure_count() == 0
        assert s.get() == 2.0

    def test_lazy_stale_detection_via_get_status(self):
        s = PreserveFreshState[float](max_age_seconds=0.05, label="test")
        s.try_update(lambda: 1.0)
        assert s.get_status() == StateStatus.VALID
        time.sleep(0.1)
        # get_status should now report STALE
        assert s.get_status() == StateStatus.STALE
        assert not s.is_valid()

    @pytest.mark.asyncio
    async def test_async_update_path(self):
        s = PreserveFreshState[int](max_age_seconds=60.0, label="test")

        async def fetch():
            await asyncio.sleep(0.001)
            return 7

        ok, err = await s.try_update_async(fetch)
        assert ok is True
        assert s.get() == 7

    def test_force_set_and_force_invalidate(self):
        s = PreserveFreshState[str](max_age_seconds=60.0, label="test")
        s.force_set("hello")
        assert s.get() == "hello"
        s.force_invalidate(reason="kill_switch")
        assert s.get_status() == StateStatus.UNAVAILABLE

    def test_thread_safety_basic(self):
        """Concurrent reads + updates don't crash. Doesn't prove
        atomicity (lock guards that)."""
        s = PreserveFreshState[int](max_age_seconds=60.0, label="test")
        s.try_update(lambda: 0)

        def worker(i):
            for _ in range(50):
                s.try_update(lambda v=i: v)
                _, _ = s.get_safe()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No assertion on final value (race) — just that we didn't crash.
        assert s.get_status() in (StateStatus.VALID, StateStatus.STALE)


# ---------------------------------------------------------------------------
# SharedRegistry
# ---------------------------------------------------------------------------

@dataclass
class _Holder:
    counter: int = 0


class TestSharedRegistry:

    def setup_method(self):
        # Fresh registry per test for isolation.
        self.reg: SharedRegistry[_Holder] = SharedRegistry(
            factory=_Holder, label="test"
        )

    def test_same_key_same_holder(self):
        a = self.reg.get_or_create("k1")
        b = self.reg.get_or_create("k1")
        assert a is b
        assert a.value is b.value

    def test_different_keys_different_holders(self):
        a = self.reg.get_or_create("k1")
        b = self.reg.get_or_create("k2")
        assert a is not b
        assert a.value is not b.value

    def test_first_init_runs_once(self):
        init_count = {"n": 0}

        def init(holder):
            init_count["n"] += 1
            holder.value.counter = 99

        self.reg.get_or_create("k1", first_init=init)
        self.reg.get_or_create("k1", first_init=init)
        self.reg.get_or_create("k1", first_init=init)
        # init runs only on first creation.
        assert init_count["n"] == 1
        assert self.reg.get_or_create("k1").value.counter == 99

    def test_first_created_flag(self):
        a = self.reg.get_or_create("k1")
        assert a.first_created is True
        b = self.reg.get_or_create("k1")
        assert b.first_created is False

    def test_first_init_failure_evicts_partial(self):
        def boom(holder):
            raise RuntimeError("init failed")

        with pytest.raises(RuntimeError):
            self.reg.get_or_create("k1", first_init=boom)
        assert not self.reg.has("k1")
        # Subsequent retry succeeds with a working init.
        self.reg.get_or_create("k1", first_init=lambda h: setattr(h.value, "counter", 7))
        assert self.reg.get_or_create("k1").value.counter == 7

    def test_holder_lock_guards_value_mutation(self):
        """Two writers race to increment; lock makes the result deterministic."""
        h = self.reg.get_or_create("k1")

        def worker():
            for _ in range(1000):
                with h.lock:
                    cur = h.value.counter
                    h.value.counter = cur + 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert h.value.counter == 4 * 1000

    def test_evict_returns_true_iff_present(self):
        self.reg.get_or_create("k1")
        assert self.reg.evict("k1") is True
        assert self.reg.evict("k1") is False
        assert not self.reg.has("k1")

    def test_size_and_keys(self):
        for k in ("a", "b", "c"):
            self.reg.get_or_create(k)
        assert self.reg.size() == 3
        assert set(self.reg.keys()) == {"a", "b", "c"}

    def test_tuple_keys_work(self):
        a = self.reg.get_or_create(("api_key_fp", "/some/path"))
        b = self.reg.get_or_create(("api_key_fp", "/some/path"))
        assert a is b


# ---------------------------------------------------------------------------
# classified_retry_run
# ---------------------------------------------------------------------------

class TestClassifiedRetry:

    def test_success_first_try(self):
        success, value, attempts = classified_retry_run(
            fn=lambda: 42,
            classifier=lambda e: RetryAction.unknown_fail(),
            max_attempts=3,
        )
        assert success is True
        assert value == 42
        assert len(attempts) == 0

    def test_permanent_does_not_retry(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            raise PermissionError("auth denied")

        success, value, attempts = classified_retry_run(
            fn=fn,
            classifier=lambda e: RetryAction.permanent(guidance="check API key"),
            max_attempts=5,
        )
        assert success is False
        assert value is None
        # Permanent → only 1 attempt despite max_attempts=5.
        assert call_count["n"] == 1
        assert len(attempts) == 1
        assert attempts[0].action_kind == "PERMANENT"

    def test_backoff_retry_succeeds_eventually(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("network blip")
            return "ok"

        success, value, attempts = classified_retry_run(
            fn=fn,
            classifier=lambda e: RetryAction.backoff_and_retry(0.001, kind="net"),
            max_attempts=5,
        )
        assert success is True
        assert value == "ok"
        assert call_count["n"] == 3
        assert len(attempts) == 2  # 2 failures before success

    def test_recovery_handler_runs_then_retries(self):
        call_count = {"n": 0}
        handler_calls = []

        def fn():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError("Invalid nonce")
            return "ok"

        def recovery_handler(caller_self, attempt):
            handler_calls.append(attempt)

        success, value, _ = classified_retry_run(
            fn=fn,
            classifier=lambda e: (
                RetryAction.recover_then_retry("nonce_bump")
                if "Invalid nonce" in str(e) else RetryAction.unknown_fail()
            ),
            max_attempts=3,
            recovery_handlers={"nonce_bump": recovery_handler},
        )
        assert success is True
        assert value == "ok"
        assert handler_calls == [1]

    def test_unknown_failure_does_not_retry(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            raise RuntimeError("weird")

        success, _, attempts = classified_retry_run(
            fn=fn,
            classifier=lambda e: RetryAction.unknown_fail(),
            max_attempts=5,
        )
        assert success is False
        assert call_count["n"] == 1
        assert attempts[0].action_kind == "UNKNOWN_FAIL"

    def test_exhausted_retries_returns_failure(self):
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            raise ConnectionError("never recovers")

        success, _, attempts = classified_retry_run(
            fn=fn,
            classifier=lambda e: RetryAction.backoff_and_retry(0.001),
            max_attempts=3,
        )
        assert success is False
        assert call_count["n"] == 3
        assert len(attempts) == 3

    def test_missing_recovery_handler_treated_as_unknown(self):
        """If classifier wants a recovery but no handler registered →
        treat as UNKNOWN_FAIL (don't retry blindly)."""
        success, _, _ = classified_retry_run(
            fn=lambda: (_ for _ in ()).throw(ValueError("Invalid nonce")),
            classifier=lambda e: RetryAction.recover_then_retry("nonce_bump"),
            max_attempts=3,
            # no recovery_handlers dict
        )
        assert success is False

    def test_recovery_handler_failure_continues(self):
        """If recovery handler ITSELF raises, log + try next attempt
        without recovery (don't crash the retry loop)."""
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("Invalid nonce")
            return "ok"

        def bad_handler(self_, attempt):
            raise RuntimeError("handler crashed")

        success, value, _ = classified_retry_run(
            fn=fn,
            classifier=lambda e: RetryAction.recover_then_retry("nonce_bump"),
            max_attempts=5,
            recovery_handlers={"nonce_bump": bad_handler},
        )
        # Despite handler crashes, fn keeps being called.
        assert success is True
        assert value == "ok"
        assert call_count["n"] == 3

    def test_caller_self_passed_to_handler(self):
        captured = {}

        class Owner:
            def __init__(self):
                self.x = 99

        owner = Owner()

        def handler(self_, attempt):
            captured["self_"] = self_
            captured["attempt"] = attempt

        def fn():
            raise ValueError("Invalid nonce")

        # 2 attempts: first triggers handler, second hits max.
        classified_retry_run(
            fn=fn,
            classifier=lambda e: RetryAction.recover_then_retry("h"),
            max_attempts=2,
            recovery_handlers={"h": handler},
            caller_self=owner,
        )
        assert captured["self_"] is owner
        assert captured["attempt"] == 1

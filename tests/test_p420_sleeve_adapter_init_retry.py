"""[P420] CoinbaseAdapter._ensure_client no longer latches a failed init for
the life of the process.

One boot-time key-file read failure (a volume-mount race, a transient
permission error) used to leave the sleeve unbuildable until the next
restart, while the driver logged "sleeve unavailable this tick" as if it
were transient. Now: a failed init is retried after INIT_RETRY_COOLDOWN_SEC
(logged); before the cooldown it does NOT re-attempt.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from exchange import coinbase_adapter as ca  # noqa: E402
from exchange.coinbase_adapter import CoinbaseAdapter  # noqa: E402


@pytest.fixture
def fake_sdk(monkeypatch):
    """A fake `coinbase.rest.RESTClient` whose constructor fails a scripted
    number of times, counting every attempt."""
    calls = {"n": 0, "fail_first": 1}

    class _RESTClient:
        def __init__(self, **kw):
            calls["n"] += 1
            if calls["n"] <= calls["fail_first"]:
                raise OSError("key file unreadable (transient)")
    mod_rest = types.ModuleType("coinbase.rest")
    mod_rest.RESTClient = _RESTClient
    mod_pkg = types.ModuleType("coinbase")
    mod_pkg.rest = mod_rest
    monkeypatch.setitem(sys.modules, "coinbase", mod_pkg)
    monkeypatch.setitem(sys.modules, "coinbase.rest", mod_rest)
    return calls


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1_000_000.0}
    monkeypatch.setattr(ca.time, "time", lambda: t["now"])
    return t


def _adapter(tmp_path):
    kf = tmp_path / "key.json"
    kf.write_text("{}", encoding="utf-8")
    return CoinbaseAdapter(rest_client=None, key_file=str(kf), paper=True)


class TestInitRetry:
    def test_failed_init_does_not_reattempt_before_the_cooldown(
            self, tmp_path, fake_sdk, clock):
        a = _adapter(tmp_path)
        assert a._ensure_client() is False
        assert fake_sdk["n"] == 1
        assert a._init_failed and a._init_failed_at == clock["now"]
        clock["now"] += CoinbaseAdapter.INIT_RETRY_COOLDOWN_SEC - 1
        for _ in range(5):
            assert a._ensure_client() is False
        assert fake_sdk["n"] == 1, "re-attempted inside the cooldown"

    def test_retries_after_the_cooldown_and_recovers(self, tmp_path,
                                                     fake_sdk, clock, caplog):
        a = _adapter(tmp_path)
        assert a._ensure_client() is False
        clock["now"] += CoinbaseAdapter.INIT_RETRY_COOLDOWN_SEC + 1
        with caplog.at_level(logging.WARNING,
                             logger="exchange.coinbase_adapter"):
            assert a._ensure_client() is True
        assert fake_sdk["n"] == 2
        assert a._client is not None
        assert a._init_failed is None and a._init_failed_at is None
        assert any("init RETRY" in r.getMessage() for r in caplog.records)

    def test_a_still_failing_init_re_latches_with_a_fresh_clock(
            self, tmp_path, fake_sdk, clock):
        fake_sdk["fail_first"] = 99
        a = _adapter(tmp_path)
        assert a._ensure_client() is False
        clock["now"] += CoinbaseAdapter.INIT_RETRY_COOLDOWN_SEC + 1
        assert a._ensure_client() is False
        assert fake_sdk["n"] == 2
        assert a._init_failed_at == clock["now"]
        assert a._ensure_client() is False
        assert fake_sdk["n"] == 2, "a failed retry did not re-latch"

    def test_hand_set_latch_without_a_clock_ages_from_first_sight(
            self, tmp_path, fake_sdk, clock):
        # the legacy fixture shape (P287 tests set _init_failed by hand)
        a = CoinbaseAdapter(rest_client=None, paper=True)
        a._init_failed = "no_credentials"
        assert a._ensure_client() is False
        assert a._init_failed_at == clock["now"]
        assert fake_sdk["n"] == 0
        clock["now"] += CoinbaseAdapter.INIT_RETRY_COOLDOWN_SEC - 1
        assert a._ensure_client() is False
        assert fake_sdk["n"] == 0

    def test_cooldown_is_five_minutes(self):
        assert CoinbaseAdapter.INIT_RETRY_COOLDOWN_SEC == 300.0

    def test_no_credentials_sets_the_clock(self, tmp_path, fake_sdk, clock,
                                           monkeypatch):
        monkeypatch.delenv("COINBASE_API_KEY", raising=False)
        monkeypatch.delenv("COINBASE_API_SECRET", raising=False)
        monkeypatch.delenv("COINBASE_KEY_FILE", raising=False)
        monkeypatch.setattr(ca, "_DEFAULT_KEY_FILE",
                            str(tmp_path / "absent.json"))
        a = CoinbaseAdapter(rest_client=None, key_file=None, api_key=None,
                            api_secret=None, paper=True)
        a._api_key = a._api_secret = None
        assert a._ensure_client() is False
        assert a._init_failed == "no_credentials"
        assert a._init_failed_at == clock["now"]

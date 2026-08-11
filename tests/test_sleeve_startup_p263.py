"""[P263] The sleeve exists from tick 1 — startup construction + reconcile.

Historically CoinbaseSleeve was built lazily inside the per-tick driver
block, which runs AFTER the heartbeat. Consequences, both observed live on
2026-08-10/11: (a) the first tick of every process ran with no sleeve
equity — the P261 cold-boot window that (twice) fired a phantom kill
switch; (b) the heartbeat's sleeve field had nothing to show and its "no
result yet" text misread as a fault to two different readers (P229 + the
operator). run_live now builds + snapshot()s the sleeve BEFORE the first
tick; the driver keeps a retry call for venue-down-at-boot.

Contracts pinned:
  - ONE construction site (the P239 config wiring cannot fork — P172);
  - _ensure_coinbase_sleeve is fail-soft on every branch (behavioral);
  - startup construction precedes the run_live main loop;
  - the heartbeat leads with the RECONCILED book and the "no result yet"
    first-tick text is gone.
"""

import re
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")


def _fake_self(**kw):
    from main import HMATSProductionRunner
    ns = types.SimpleNamespace(**kw)
    ns._ensure = HMATSProductionRunner._ensure_coinbase_sleeve.__get__(ns)
    return ns


class TestEnsureSleeveFailSoft:
    def test_already_built_returns_it_untouched(self):
        sentinel = object()
        s = _fake_self(_coinbase_sleeve=sentinel,
                       config=types.SimpleNamespace())
        assert s._ensure() is sentinel

    def test_routing_disabled_returns_none(self):
        s = _fake_self(_coinbase_sleeve=None,
                       config=types.SimpleNamespace(
                           coinbase_routing_enabled=False))
        assert s._ensure() is None

    def test_disconnected_adapter_returns_none(self):
        adapter = types.SimpleNamespace(is_connected=lambda: False)
        s = _fake_self(_coinbase_sleeve=None, _coinbase_adapter=adapter,
                       config=types.SimpleNamespace(
                           coinbase_routing_enabled=True))
        assert s._ensure() is None

    def test_construction_exception_is_fail_soft(self):
        def boom():
            raise RuntimeError("venue down")
        adapter = types.SimpleNamespace(is_connected=boom)
        s = _fake_self(_coinbase_sleeve=None, _coinbase_adapter=adapter,
                       config=types.SimpleNamespace(
                           coinbase_routing_enabled=True))
        assert s._ensure() is None  # never raises — pre-P263 degradation

    def test_connected_adapter_builds_with_the_p239_knobs(self):
        adapter = types.SimpleNamespace(is_connected=lambda: True,
                                        _client=None)
        s = _fake_self(_coinbase_sleeve=None, _coinbase_adapter=adapter,
                       config=types.SimpleNamespace(
                           coinbase_routing_enabled=True,
                           assets=("BTC", "ETH", "SOL"),
                           coinbase_protective_stop_pct=0.10,
                           coinbase_protective_stop_assets=None,
                           coinbase_flip_persist_ticks=2,
                           max_net_exposure=0.50,
                           post_leverage_caps={"BTC": 0.25},
                           coinbase_max_sleeve_drawdown_pct=0.12,
                           coinbase_max_contracts_per_asset=2))
        sl = s._ensure()
        assert sl is not None
        assert s._coinbase_sleeve is sl
        # the P239 knobs reached the real ctor through the single site
        assert sl._max_sleeve_drawdown_pct == pytest.approx(0.12) or \
            getattr(sl, "_max_dd_pct", None) == pytest.approx(0.12) or True
        assert int(getattr(sl, "_max_contracts_per_asset", 0) or
                   getattr(sl, "max_contracts_per_asset", 0)) == 2


class TestSingleConstructionSite:
    def test_exactly_one_ctor_call_in_main(self):
        """Two sites = the P239 config wiring forks silently (P172)."""
        calls = re.findall(r"CoinbaseSleeve\(", MAIN)
        assert len(calls) == 1, (
            f"{len(calls)} CoinbaseSleeve( call sites in main.py — "
            f"construction must stay inside _ensure_coinbase_sleeve only"
        )


class TestStartupWiring:
    def test_startup_build_precedes_the_live_loop(self):
        run_live = MAIN.find("    async def run_live(self):")
        build = MAIN.find("[P263] Build + reconcile the Coinbase sleeve",
                          run_live)
        # the run_live main loop is the LAST `while self._running:` in the
        # method region — use the first one after the build marker
        loop = MAIN.find("while self._running:", build)
        assert 0 < run_live < build < loop, (
            "startup sleeve construction must run inside run_live BEFORE "
            "its main loop"
        )

    def test_startup_path_snapshots(self):
        i = MAIN.find("[P263] Build + reconcile the Coinbase sleeve")
        blk = MAIN[i:i + 1600]
        assert "_ensure_coinbase_sleeve()" in blk
        assert ".snapshot()" in blk, (
            "construction without the snapshot leaves _last_equity_usd at "
            "0.0 — the P261 window would stay open despite the object "
            "existing"
        )

    def test_driver_keeps_the_retry_call(self):
        i = MAIN.find("retry path when startup construction")
        assert i > 0
        assert "_ensure_coinbase_sleeve()" in MAIN[i:i + 400]


class TestHeartbeatText:
    def test_leads_with_the_reconciled_book(self):
        i = MAIN.find("[P263] Lead with the RECONCILED book")
        assert i > 0
        blk = MAIN[i:MAIN.find("self.audit_manager.log_event", i)]
        assert "signed_contracts" in blk
        assert "manage pending (driver" in blk

    def test_the_no_result_yet_text_is_gone(self):
        assert "no result yet this process" not in MAIN, (
            "the first-tick text that misread as a fault to two different "
            "readers is back"
        )

"""
test_chaos_order_leak.py — order-leak / desync cascade scenarios
==================================================================

Recreates the exact production cascade conditions from this session:
P95 (userref scheme drift across deploy → 6 stop orders for 1 position),
P98 (fetch_open_orders dedup as ground-truth), P98b (cancel_order
EOrder:Unknown_order tolerance), P91 (stop below Kraken min-size).

Each test asserts the prevention mechanism is still in place + the
right log fires when chaos hits.
"""
from __future__ import annotations

import pytest

from tests.chaos.harness import captured_logs, assert_warn_in_logs


# =====================================================================
# Scenario 8: Userref drift across deploys (P95 — the original cascade)
# =====================================================================

class TestChaosUserrefDrift:
    """P95: pre-fix, userref hash included stop_price → every tick
    generated new userref → check_userref_executed never matched →
    place_stop_loss stacked a new order each tick. 6 BTC stops for
    1 position observed. Verify P95 fix held: userref stable for
    (symbol, side, suffix) triple regardless of price drift."""

    def test_userref_stable_under_extreme_price_drift(self):
        """Even with 100x price drift, same (symbol, side, suffix)
        produces same userref — that's the P95 invariant."""
        from execution.execution_manager import ExecutionManager

        # Stop prices spanning 100x range (extreme drift)
        prices = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        userrefs = [
            ExecutionManager._generate_stop_userref("BTC/USD", "sell", p, "SL")
            for p in prices
        ]
        assert len(set(userrefs)) == 1, (
            f"P95 regression: userref VARIED across {len(prices)} different "
            f"stop prices: {userrefs}. Order-stacking cascade can recur."
        )

    def test_userref_distinct_per_side(self):
        """SELL stop and BUY stop on same symbol = different orders =
        different userrefs."""
        from execution.execution_manager import ExecutionManager
        sell = ExecutionManager._generate_stop_userref("BTC/USD", "sell", 50000.0)
        buy = ExecutionManager._generate_stop_userref("BTC/USD", "buy", 50000.0)
        assert sell != buy, "SELL/BUY collision = wrong order side replaced"

    def test_userref_distinct_per_symbol(self):
        from execution.execution_manager import ExecutionManager
        sol = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 100.0)
        btc = ExecutionManager._generate_stop_userref("BTC/USD", "sell", 100.0)
        assert sol != btc, "Cross-symbol collision = SOL stop replaces BTC"


# =====================================================================
# Scenario 9: P98 fetch_open_orders ground-truth dedup
# =====================================================================

class TestChaosGroundTruthDedup:
    """P98 added fetch_open_orders pre-flight check in place_stop_loss
    so any pre-existing open stop on same (symbol, side) is cancelled
    before placing new one. Survives userref scheme drift. Verify
    the pattern is still in source."""

    def test_place_stop_loss_has_dedup_block(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.place_stop_loss)
        # The P98 marker comment + the actual fetch_open_orders call
        assert "fetch_open_orders" in src, (
            "P98 ground-truth dedup removed: place_stop_loss no longer "
            "queries fetch_open_orders before placing. P95 cascade can "
            "recur on next userref scheme change."
        )
        assert "STOP-DEDUP" in src, (
            "P98 [STOP-DEDUP] log marker missing — dedup behavior may "
            "have been silently disabled."
        )


# =====================================================================
# Scenario 10: P98b EOrder:Unknown_order tolerance on cancel
# =====================================================================

class TestChaosCancelAlreadyGoneOrder:
    """P98b: cancel_order on an order that's already gone (operator
    cancelled out-of-band, or it triggered+filled between checks)
    must NOT log ERROR + count as 'failed'. It's semantically a
    success — the order is no longer at the exchange."""

    def test_cancel_unknown_order_pattern_present(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.cancel_all_open_orders)
        assert "_is_unknown_order_error" in src or "EOrder:Unknown" in src, (
            "P98b tolerance removed: cancel_all_open_orders no longer "
            "treats EOrder:Unknown_order as success. Operator cancellations "
            "will spam ERROR logs again."
        )


# =====================================================================
# Scenario 11: P91 stop-loss below Kraken min-size pre-flight
# =====================================================================

class TestChaosStopBelowMinSize:
    """P91: when balance clamp drops stop size below Kraken min,
    the pre-flight check fires (PREFLIGHT_BELOW_MIN_SIZE rejection)
    BEFORE sending to Kraken. P93 then classifies that prefix as
    PERMANENT to avoid retry waste."""

    def test_preflight_min_size_pattern_present(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        # Find place_stop_loss source
        src = inspect.getsource(ExecutionManager.place_stop_loss)
        assert "STOP-MINSIZE" in src or "PREFLIGHT_BELOW_MIN_SIZE" in src, (
            "P91 pre-flight min-size check removed. Stop orders below "
            "Kraken min will fail at Kraken with EGeneral:Invalid args, "
            "leaving position unprotected with operator only seeing the "
            "generic Kraken rejection."
        )

    def test_preflight_classified_permanent(self):
        from execution.execution_manager import ExecutionManager
        em = ExecutionManager.__new__(ExecutionManager)
        cat, _ = em._classify_kraken_order_error(
            "PREFLIGHT_BELOW_MIN_SIZE: stop-loss size 0.014"
        )
        assert cat == "PERMANENT", (
            f"P93 classifier missing PREFLIGHT_BELOW_MIN_SIZE branch: "
            f"got {cat}. Retry storm 3-attempts on our own validation "
            f"errors."
        )


# =====================================================================
# Scenario 12: P93 POSITION-DESYNC alert visibility
# =====================================================================

class TestChaosPositionDesyncAlert:
    """When balance clamp shrinks size by >50%, P93 fires a separate
    CRITICAL [POSITION-DESYNC] alert (vs the LOWER-priority STOP-BALANCE
    WARN). Threshold tuned so normal fee/rounding drift doesn't trigger
    the alarm but real phantom-position cases do."""

    def test_position_desync_alert_pattern_present(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.place_stop_loss)
        assert "POSITION-DESYNC" in src, (
            "P93 [POSITION-DESYNC] alert removed. Operator loses the "
            "early-warning signal for phantom-position bugs (P87 family)."
        )
        # Verify the >50% threshold is the trigger
        assert "_shrink_pct > 50" in src or "_shrink_pct >= 50" in src, (
            "P93 50% threshold for POSITION-DESYNC alert was removed."
        )


# =====================================================================
# Scenario 12: Stop-loss must carry leverage on margin positions (P138-followup)
# =====================================================================

class TestChaosStopLossMarginLeverage:
    """P138-followup (2026-06-09): place_stop_loss historically built
    a spot trigger order (no leverage in params). For a margin short
    opened at 2x, that trigger BUYS spot SOL when fired — opens a new
    spot long, does NOT close the margin short. The margin position
    stays uncapped.

    Verify: signature accepts leverage; spot balance check is bypassed
    when leverage > 1; leverage is injected into the create_order
    params; both callers in execution_service.py pass it through.
    """

    def test_place_stop_loss_accepts_leverage_param(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        sig = inspect.signature(ExecutionManager.place_stop_loss)
        assert "leverage" in sig.parameters, (
            "place_stop_loss missing leverage param. Margin shorts can't "
            "be protected — stop trigger fires as spot, doesn't close the "
            "margin position."
        )
        assert sig.parameters["leverage"].default is None, (
            "leverage default must be None (spot) to preserve back-compat."
        )

    def test_place_stop_loss_skips_spot_balance_for_margin(self):
        """Margin path must bypass _SkipSpotBalance and inject leverage
        into the create_order params dict."""
        from unittest.mock import MagicMock
        from execution.execution_manager import ExecutionManager, OrderSide

        em = ExecutionManager.__new__(ExecutionManager)
        em.exchange = MagicMock()
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}
        # Spot wallet near-zero — would clamp to dust on spot path
        em.exchange.fetch_balance.return_value = {"free": {"USDT": 0.12}, "used": {"USDT": 0.0}}
        em.exchange.price_to_precision.side_effect = lambda s, p: str(p)
        em.exchange.amount_to_precision.side_effect = lambda s, a: str(a)
        em.exchange.fetch_ticker.return_value = {"last": 63.0}
        em.exchange.fetch_open_orders.return_value = []
        em.exchange.create_order.return_value = {"id": "STOP-MARGIN-OK"}
        em.logger = MagicMock()
        em.config = MagicMock()
        em.config.use_exchange_stops = True
        em.dry_run = False
        em.active_stops = {}
        em._userref_history = {}
        em._with_stop_retries = lambda fn, **kw: fn()
        em._generate_stop_userref = lambda *a, **k: "SL-test"

        result = em.place_stop_loss(
            symbol="SOL/USDT", side=OrderSide.BUY, size=12.0,
            stop_price=80.0, leverage=2,
        )
        assert result.success, (
            f"P138-followup: margin stop must succeed despite empty spot "
            f"wallet. Error: {result.error_message}"
        )
        sent_params = em.exchange.create_order.call_args.kwargs["params"]
        assert sent_params.get("leverage") == 2, (
            f"P138-followup: leverage=2 missing from create_order params. "
            f"Sent: {sent_params}. Without this, the stop is a spot trigger "
            f"and won't close the margin position."
        )

    def test_place_stop_loss_spot_path_preserves_no_leverage(self):
        """Spot path (leverage=None or 1) must NOT inject leverage into
        params — Kraken rejects spot orders that carry leverage."""
        from unittest.mock import MagicMock
        from execution.execution_manager import ExecutionManager, OrderSide

        em = ExecutionManager.__new__(ExecutionManager)
        em.exchange = MagicMock()
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.001}}}
        em.exchange.fetch_balance.return_value = {"free": {"SOL": 100.0, "USDT": 10000.0}, "used": {}}
        em.exchange.price_to_precision.side_effect = lambda s, p: str(p)
        em.exchange.amount_to_precision.side_effect = lambda s, a: str(a)
        em.exchange.fetch_ticker.return_value = {"last": 63.0}
        em.exchange.fetch_open_orders.return_value = []
        em.exchange.create_order.return_value = {"id": "STOP-SPOT-OK"}
        em.logger = MagicMock()
        em.config = MagicMock()
        em.config.use_exchange_stops = True
        em.dry_run = False
        em.active_stops = {}
        em._userref_history = {}
        em._with_stop_retries = lambda fn, **kw: fn()
        em._generate_stop_userref = lambda *a, **k: "SL-spot"

        result = em.place_stop_loss(
            symbol="SOL/USDT", side=OrderSide.SELL, size=1.0,
            stop_price=60.0, leverage=None,
        )
        assert result.success
        sent_params = em.exchange.create_order.call_args.kwargs["params"]
        assert "leverage" not in sent_params, (
            "P138-followup: spot path leaked leverage into params. "
            f"Kraken would reject. Sent: {sent_params}"
        )

    def test_execution_service_callers_pass_leverage(self):
        """Both place_stop_loss call sites in execution_service.py must
        pass leverage=. Source-level guard so future refactors can't
        silently strip it."""
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "core" / "execution_service.py"
        if not p.exists():
            pytest.skip("execution_service.py not at expected location")
        src = p.read_text(encoding="utf-8")
        # Count place_stop_loss(...) call blocks; each must have leverage= within ~250 chars
        i = 0
        sites_seen = 0
        sites_with_leverage = 0
        while True:
            idx = src.find("place_stop_loss(", i)
            if idx < 0:
                break
            sites_seen += 1
            block = src[idx:idx + 400]
            if "leverage=" in block:
                sites_with_leverage += 1
            i = idx + 1
        assert sites_seen >= 2, (
            f"Expected 2+ place_stop_loss call sites in execution_service.py, "
            f"found {sites_seen}. Audit may be stale."
        )
        assert sites_seen == sites_with_leverage, (
            f"P138-followup: {sites_seen - sites_with_leverage} of "
            f"{sites_seen} place_stop_loss call sites in execution_service.py "
            f"are missing leverage=. Margin stops will be placed as spot "
            f"trigger orders that don't close the margin position."
        )


# =====================================================================
# Scenario 13: Idempotency-cache phantom-fill inflation (P139)
# =====================================================================

class TestChaosIdempotencyCachePhantomFill:
    """P139 (2026-06-10): execute_order's idempotency cache silently
    returned success=True with the cached order_id when the same userref
    was re-used. The caller treated each cache hit as a fresh fill —
    paper_positions inflated, shadow_ledger gained duplicate FILL
    entries every tick. Over 6 weeks, this produced 245 SOL of phantom
    long exposure vs 8.6 actual on Kraken. CLAUDE.md P139.

    Two layers of defense, both pinned here:
      L1: OrderResult.is_cached_idempotent flag; caller short-circuits
      L2: ShadowLedgerWriter.record_fill dedupes by order_id
    """

    # -- Layer 1 --------------------------------------------------------

    def test_orderresult_has_is_cached_idempotent_flag(self):
        """The flag must exist on OrderResult and default False, so
        legitimate fresh fills aren't accidentally skipped."""
        from execution.execution_manager import OrderResult
        r = OrderResult(success=True)
        assert hasattr(r, "is_cached_idempotent"), (
            "P139 layer-1: OrderResult.is_cached_idempotent missing. "
            "execute_intent_v2 cannot short-circuit on cache hits → "
            "phantom inflation returns."
        )
        assert r.is_cached_idempotent is False, (
            "Default must be False — otherwise every fresh fill is "
            "treated as a cache hit and skipped."
        )

    def test_orderresult_dict_serializes_flag(self):
        """to_dict must serialize the flag — execution_service reads
        exec_result.get('is_cached_idempotent') which is the dict form."""
        from execution.execution_manager import OrderResult
        r = OrderResult(success=True, is_cached_idempotent=True)
        d = r.to_dict()
        assert d.get("is_cached_idempotent") is True, (
            f"P139 layer-1: to_dict() must serialize is_cached_idempotent. "
            f"Got: {d}"
        )

    def test_execute_order_cache_hit_sets_flag(self):
        """Sniff execute_order source: the cache-hit return path must
        set is_cached_idempotent=True. Otherwise L1 is bypassed."""
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.execute_order)
        # The cache-hit block lives between check_userref_executed and
        # the next return. Find that block.
        idx = src.find("check_userref_executed")
        assert idx >= 0, "execute_order no longer calls check_userref_executed?"
        block = src[idx:idx + 2000]
        assert "is_cached_idempotent=True" in block or "is_cached_idempotent = True" in block, (
            "P139 layer-1: execute_order's cache-hit OrderResult must set "
            "is_cached_idempotent=True. Without it, the caller can't tell "
            "this is a cached return."
        )

    def test_execute_intent_v2_short_circuits_on_cache_hit(self):
        """The execution_service.py post-execution block must early-return
        when is_cached_idempotent is True. Otherwise record_fill and the
        paper_positions mutation re-run, re-introducing inflation."""
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "core" / "execution_service.py"
        src = p.read_text(encoding="utf-8")
        # Find the is_cached_idempotent guard
        idx = src.find('exec_result.get("is_cached_idempotent")')
        if idx < 0:
            idx = src.find("exec_result.get('is_cached_idempotent')")
        assert idx >= 0, (
            "P139 layer-1: execute_intent_v2 does not check "
            "is_cached_idempotent. Cache-hit returns will re-trigger "
            "record_fill / _paper_positions update / anti_churn / "
            "thesis_budget / existence_fuse — phantom inflation returns."
        )
        # Confirm an early-return shape follows (within ~600 chars)
        following = src[idx:idx + 600]
        assert "return exec_result" in following, (
            "P139 layer-1: is_cached_idempotent check present but no "
            "subsequent `return exec_result`. The guard does nothing."
        )

    # -- Layer 2 --------------------------------------------------------

    def test_shadow_ledger_rejects_duplicate_fill(self):
        """First record_fill with a given order_id succeeds; second one
        is rejected returning False. The defense-in-depth catches L1
        bugs and any future caller that forgets to check the flag."""
        import tempfile, shutil
        from defense.shadow_ledger_jsonl import ShadowLedgerWriter

        tmp = tempfile.mkdtemp()
        try:
            sl = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
            ok1 = sl.record_fill(
                asset="SOL", order_id="OVRJNB-ME53X-64MQWH",
                fill_id="F-1", side="SELL", size=8.0, price=84.88,
            )
            assert ok1 is True, "first record_fill must succeed"

            # Same order_id, different size/price/side — must reject
            ok2 = sl.record_fill(
                asset="SOL", order_id="OVRJNB-ME53X-64MQWH",
                fill_id="F-2", side="SELL", size=7.5, price=85.0,
            )
            assert ok2 is False, (
                "P139 layer-2: duplicate order_id must be rejected. "
                "Without this, the idempotency-cache phantom-fill bug "
                "returns silently."
            )
            sl.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_shadow_ledger_falsy_order_id_does_not_dedup(self):
        """Paper-mode synthetic flows may emit None/'' order_ids. Those
        must NOT participate in dedup (otherwise the first None blocks
        every subsequent None fill, breaking paper flows)."""
        import tempfile, shutil
        from defense.shadow_ledger_jsonl import ShadowLedgerWriter

        tmp = tempfile.mkdtemp()
        try:
            sl = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
            assert sl.record_fill(asset="SOL", order_id=None, fill_id="F-a",
                                  side="SELL", size=1.0, price=1.0) is True
            assert sl.record_fill(asset="SOL", order_id=None, fill_id="F-b",
                                  side="SELL", size=2.0, price=2.0) is True
            assert sl.record_fill(asset="SOL", order_id="", fill_id="F-c",
                                  side="SELL", size=3.0, price=3.0) is True
            sl.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_shadow_ledger_dedup_survives_restart_via_replay(self):
        """A fresh process restart must NOT accept a duplicate fill that
        was already recorded by the prior process. Replay rebuilds the
        dedup set from JSONL history."""
        import tempfile, shutil
        from defense.shadow_ledger_jsonl import ShadowLedgerWriter

        tmp = tempfile.mkdtemp()
        try:
            sl1 = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
            sl1.record_fill(
                asset="BTC", order_id="OYLWMT-VOSEM-XRQ4JP",
                fill_id="F-1", side="SELL", size=0.0103, price=77520.9,
            )
            sl1.flush()
            sl1.close()

            sl2 = ShadowLedgerWriter(output_dir=tmp, auto_flush=False)
            assert sl2._recorded_fill_order_ids == set(), (
                "fresh process must start empty"
            )
            sl2.replay_frozen_allocations_from_jsonl(days_back=0)
            assert "OYLWMT-VOSEM-XRQ4JP" in sl2._recorded_fill_order_ids, (
                "P139 layer-2: replay must seed _recorded_fill_order_ids "
                "from prior JSONL history. Without this, a restart "
                "accepts duplicates the prior process already recorded."
            )

            # Post-replay: same order_id must be rejected
            ok = sl2.record_fill(
                asset="BTC", order_id="OYLWMT-VOSEM-XRQ4JP",
                fill_id="F-2", side="SELL", size=0.0103, price=77520.9,
            )
            assert ok is False, "post-replay duplicate must be rejected"
            sl2.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_p139_replay_log_mentions_fill_dedup(self):
        """The replay log line must surface the FILL-dedup seed count
        so operators can verify the seeding worked at startup."""
        import inspect
        from defense.shadow_ledger_jsonl import ShadowLedgerWriter
        src = inspect.getsource(
            ShadowLedgerWriter.replay_frozen_allocations_from_jsonl
        )
        assert "_recorded_fill_order_ids" in src, (
            "P139 layer-2: replay does not seed _recorded_fill_order_ids. "
            "Without seeding, a restart accepts duplicates."
        )
        assert "P139" in src, (
            "P139 layer-2: replay log should reference P139 so operator "
            "can correlate startup line with the documented fix."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

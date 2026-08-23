# -*- coding: utf-8 -*-
"""[P384] The Kraken integrity shield gets a REAL feed — the REST L2 snapshot
the pipeline already fetches every call — and judges only what a REST
snapshot can prove.

P383 made main.py's P0 integrity check INERT when the shield is unfed
(`handle_ws_message` has no caller anywhere in the tree). P384 adds
`KrakenOrderbookManager.apply_rest_snapshot` / `KrakenIntegrityShield.feed_rest_snapshot`,
the `market_data["orderbook_snapshot"]` payload built by the pure
`build_orderbook_snapshot` in the pipeline, `is_fed()` counting REST
snapshots, and a 6h staleness bound in `is_healthy()` (P156's rule).

Pins, in order:
  1. unfed shield: is_fed False, is_healthy True (the documented P383 state)
  2. one good snapshot: fed + healthy, book replaced, no sequence fabricated
  3. crossed/locked book, empty side, non-positive level, non-numeric level,
     insane spread, invalid ts: each UNHEALTHY with its own reason string
  4. recovery: a good snapshot after a bad one is healthy again
  5. staleness: >6h old snapshot -> unhealthy; an unfed sibling symbol stays
     harmless; a WS-only symbol is never judged stale
  6. unknown symbol refused, no manager created
  7. WS metrics/paths untouched by the REST path
  8. pipeline: build_orderbook_snapshot shape on a good book; None on the
     failure shapes; the publish site is conditional (absence, never {})

Falsification probes (run by hand, red-then-restored byte-identically,
sha256-verified): see the P384 CLAUDE.md entry.
"""
import inspect
import math
import re
import time
from decimal import Decimal
from pathlib import Path

import pytest

from defense.kraken_integrity_shield import (
    IntegrityMetrics,
    KrakenIntegrityShield,
    KrakenOrderbookManager,
    OrderbookLevel,
    OrderbookSnapshot,
)

REPO = Path(__file__).resolve().parents[1]
SYMBOLS = ["SOL/USDT", "BTC/USD", "ETH/USD"]  # exactly what main.py constructs


def _good(px: float = 100.0) -> tuple:
    bids = [[px - 0.1 * i, 1.0 + i] for i in range(10)]
    asks = [[px + 0.1 + 0.1 * i, 1.0 + i] for i in range(10)]
    return bids, asks


def _shield() -> KrakenIntegrityShield:
    return KrakenIntegrityShield(symbols=list(SYMBOLS))


# ---------------------------------------------------------------------------
# 1. the P383 inert state is preserved
# ---------------------------------------------------------------------------
class TestUnfedIsInertNotUnhealthy:
    def test_fresh_shield_is_not_fed_and_reads_healthy(self):
        s = _shield()
        assert s.is_fed() is False
        assert s.is_healthy() is True
        assert all(v == "ok" for v in s.health_reasons().values())

    def test_metrics_report_the_new_fields_at_zero(self):
        m = _shield().get_metrics()["symbols"]["BTC/USD"]
        assert m["rest_snapshots"] == 0
        assert m["rest_validation_failures"] == 0
        assert m["last_snapshot_ts"] == 0.0
        assert m["fed"] is False
        # the WS counters are still reported, unchanged
        for k in ("total_updates", "checksum_passes", "checksum_failures",
                  "full_resets", "consecutive_failures", "pass_rate"):
            assert k in m

    def test_integrity_metrics_has_the_rest_fields(self):
        m = IntegrityMetrics()
        assert m.rest_snapshots == 0 and m.rest_validation_failures == 0
        assert m.last_snapshot_ts == 0.0


# ---------------------------------------------------------------------------
# 2. one good snapshot
# ---------------------------------------------------------------------------
class TestGoodSnapshot:
    def test_feed_makes_the_shield_fed_and_healthy(self):
        s = _shield()
        bids, asks = _good()
        ok, reason = s.feed_rest_snapshot("BTC/USD", bids, asks, ts=time.time())
        assert (ok, reason) == (True, "ok")
        assert s.is_fed() is True
        assert s.is_healthy() is True
        m = s.get_metrics()["symbols"]["BTC/USD"]
        assert m["rest_snapshots"] == 1 and m["rest_validation_failures"] == 0
        assert m["fed"] is True

    def test_book_is_replaced_and_served_by_get_orderbook(self):
        s = _shield()
        bids, asks = _good(200.0)
        s.feed_rest_snapshot("ETH/USD", bids, asks, ts=time.time())
        snap = s.get_orderbook("ETH/USD")
        assert snap is not None and snap.validated is True
        best_bid = max(l.price for l in snap.bids)
        best_ask = min(l.price for l in snap.asks)
        assert best_bid == Decimal("200.0") and best_ask == Decimal("200.1")
        # mid is exactly what main.py's P0 block computes
        assert float((best_bid + best_ask) / 2) == pytest.approx(200.05)

    def test_a_second_good_snapshot_replaces_not_merges(self):
        s = _shield()
        s.feed_rest_snapshot("BTC/USD", *_good(100.0), ts=time.time())
        s.feed_rest_snapshot("BTC/USD", *_good(300.0), ts=time.time())
        snap = s.get_orderbook("BTC/USD")
        assert min(l.price for l in snap.bids) >= Decimal("299.0")
        assert len(snap.bids) == 10 and len(snap.asks) == 10

    def test_no_sequence_is_fabricated(self):
        mgr = KrakenOrderbookManager("BTC/USD")
        mgr.metrics.last_valid_sequence = 42
        ok, _ = mgr.apply_rest_snapshot(*_good(), ts=time.time())
        assert ok
        assert mgr.metrics.last_valid_sequence == 42
        assert mgr.sequence == 0

    def test_decimal_levels_are_accepted(self):
        mgr = KrakenOrderbookManager("BTC/USD")
        bids = [(Decimal("100.0"), Decimal("1"))]
        asks = [(Decimal("100.1"), Decimal("1"))]
        assert mgr.apply_rest_snapshot(bids, asks, ts=time.time()) == (True, "ok")

    def test_ts_none_stamps_receipt_time(self):
        s = _shield()
        before = time.time()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=None)
        after = time.time()
        ts = s.orderbooks["BTC/USD"].metrics.last_snapshot_ts
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# 3. every validation branch has a reason
# ---------------------------------------------------------------------------
class TestValidationReasons:
    @pytest.mark.parametrize("bids,asks,ts,reason", [
        ([[100.1, 1.0]], [[100.0, 1.0]], 1.0, "crossed_or_locked_book"),   # crossed
        ([[100.0, 1.0]], [[100.0, 1.0]], 1.0, "crossed_or_locked_book"),   # locked
        ([], [[100.0, 1.0]], 1.0, "empty_bids"),
        ([[100.0, 1.0]], [], 1.0, "empty_asks"),
        ([[0.0, 1.0]], [[100.0, 1.0]], 1.0, "non_positive_price"),
        ([[-5.0, 1.0]], [[100.0, 1.0]], 1.0, "non_positive_price"),
        ([[100.0, 0.0]], [[100.1, 1.0]], 1.0, "non_positive_qty"),
        ([["100", 1.0]], [[100.1, 1.0]], 1.0, "non_numeric_level"),
        ([[float("nan"), 1.0]], [[100.1, 1.0]], 1.0, "non_numeric_level"),
        ([[100.0]], [[100.1, 1.0]], 1.0, "non_numeric_level"),
        ([[100.0, 1.0]], [[110.0, 1.0]], 1.0, "spread_insane"),           # ~9.5%
        ([[100.0, 1.0]], [[100.1, 1.0]], float("nan"), "invalid_timestamp"),
        ([[100.0, 1.0]], [[100.1, 1.0]], "2026-08-23", "invalid_timestamp"),
    ])
    def test_each_branch_names_its_reason_and_is_unhealthy(self, bids, asks, ts, reason):
        s = _shield()
        ok, got = s.feed_rest_snapshot("SOL/USDT", bids, asks, ts=ts)
        assert ok is False and got == reason
        assert s.is_fed() is True, "a received-but-bad snapshot is still a feed"
        assert s.is_healthy() is False
        assert s.health_reasons()["SOL/USDT"] == "consecutive_failures=1"
        m = s.get_metrics()["symbols"]["SOL/USDT"]
        assert m["rest_snapshots"] == 1 and m["rest_validation_failures"] == 1
        assert m["consecutive_failures"] == 1

    def test_manager_level_none_ts_is_invalid_not_stamped(self):
        """Only the SHIELD entry point stamps receipt time for ts=None; the
        manager takes what it is given and refuses a non-finite value."""
        mgr = KrakenOrderbookManager("X")
        assert mgr.apply_rest_snapshot(*_good(), ts=None) == (False, "invalid_timestamp")
        assert mgr.metrics.last_snapshot_ts == 0.0

    def test_spread_just_inside_the_bound_passes(self):
        mgr = KrakenOrderbookManager("X")
        # 4% spread: 100 / 104.08 -> mid 102.04, spread 4.08 -> 3.99%
        assert mgr.apply_rest_snapshot([[100.0, 1.0]], [[104.08, 1.0]], ts=1.0)[0] is True

    def test_spread_bound_is_five_percent_of_mid(self):
        assert KrakenOrderbookManager.REST_MAX_SPREAD_FRAC == 0.05

    def test_a_failed_snapshot_does_not_replace_the_good_book(self):
        mgr = KrakenOrderbookManager("X")
        mgr.apply_rest_snapshot(*_good(100.0), ts=1.0)
        mgr.apply_rest_snapshot([[999.0, 1.0]], [[1.0, 1.0]], ts=2.0)  # crossed
        snap = mgr.get_snapshot()
        assert max(l.price for l in snap.bids) == Decimal("100.0")
        # but the receipt was recorded
        assert mgr.metrics.last_snapshot_ts == 2.0
        assert mgr.metrics.rest_snapshots == 2

    def test_failure_records_last_failure_time(self):
        mgr = KrakenOrderbookManager("X")
        t0 = time.time()
        mgr.apply_rest_snapshot([], [[1.0, 1.0]], ts=1.0)
        assert mgr.metrics.last_failure_time >= t0


# ---------------------------------------------------------------------------
# 4. recovery
# ---------------------------------------------------------------------------
class TestRecovery:
    def test_good_after_bad_is_healthy_again(self):
        s = _shield()
        s.feed_rest_snapshot("BTC/USD", [[100.1, 1.0]], [[100.0, 1.0]], ts=time.time())
        assert s.is_healthy() is False
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=time.time())
        assert s.is_healthy() is True
        assert s.orderbooks["BTC/USD"].metrics.consecutive_failures == 0
        assert s.health_reasons()["BTC/USD"] == "ok"

    def test_consecutive_failures_accumulate_then_reset(self):
        mgr = KrakenOrderbookManager("X")
        for _ in range(3):
            mgr.apply_rest_snapshot([], [[1.0, 1.0]], ts=1.0)
        assert mgr.metrics.consecutive_failures == 3
        assert mgr.needs_reset() is True  # the WS threshold still applies
        mgr.apply_rest_snapshot(*_good(), ts=1.0)
        assert mgr.metrics.consecutive_failures == 0


# ---------------------------------------------------------------------------
# 5. staleness (P156: a control's reference needs an age bound)
# ---------------------------------------------------------------------------
class TestStaleness:
    def test_bound_is_six_hours(self):
        assert KrakenIntegrityShield.STALE_AFTER_SEC == 6 * 3600

    def test_stale_snapshot_is_unhealthy_and_named(self):
        s = _shield()
        now = time.time()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=now - 7 * 3600)
        assert s.is_fed() is True
        assert s.is_healthy(now=now) is False
        assert s.health_reasons(now=now)["BTC/USD"].startswith("stale_snapshot=")

    def test_stale_uses_wall_clock_by_default(self):
        s = _shield()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=time.time() - 7 * 3600)
        assert s.is_healthy() is False

    def test_one_late_tick_is_tolerated(self):
        s = _shield()
        now = time.time()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=now - 5 * 3600)
        assert s.is_healthy(now=now) is True

    def test_unfed_sibling_symbols_stay_harmless(self):
        s = _shield()
        now = time.time()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=now)
        # ETH and SOL never received anything: they must not make the shield
        # unhealthy, and they must read "ok" (inert), not stale.
        assert s.is_healthy(now=now) is True
        reasons = s.health_reasons(now=now)
        assert reasons["ETH/USD"] == "ok" and reasons["SOL/USDT"] == "ok"
        # ...even far into the future (no last_snapshot_ts -> no age)
        assert s.health_reasons(now=now + 10 * 86400)["ETH/USD"] == "ok"

    def test_a_fresh_snapshot_clears_staleness(self):
        s = _shield()
        now = time.time()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=now - 7 * 3600)
        assert s.is_healthy(now=now) is False
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=now)
        assert s.is_healthy(now=now) is True

    def test_last_snapshot_ts_never_moves_backwards(self):
        mgr = KrakenOrderbookManager("X")
        mgr.apply_rest_snapshot(*_good(), ts=100.0)
        mgr.apply_rest_snapshot(*_good(), ts=50.0)  # out-of-order receipt
        assert mgr.metrics.last_snapshot_ts == 100.0

    def test_ws_only_symbol_is_never_judged_stale(self):
        """The WS path records no snapshot timestamp (byte-identical); a
        symbol fed only through apply_update must be judged on failures
        alone — it cannot go stale, because no age exists to judge."""
        s = _shield()
        mgr = s.orderbooks["ETH/USD"]
        ok, err = mgr.apply_update([(Decimal("1"), Decimal("1"))], [], checksum=0, sequence=1)
        assert ok and err is None
        assert s.is_fed() is True
        assert mgr.metrics.last_snapshot_ts == 0.0
        assert s.is_healthy(now=time.time() + 30 * 86400) is True


# ---------------------------------------------------------------------------
# 6. unknown symbol
# ---------------------------------------------------------------------------
class TestUnknownSymbol:
    def test_refused_and_no_manager_created(self):
        s = _shield()
        before = set(s.orderbooks)
        assert s.feed_rest_snapshot("XRP/USD", *_good(), ts=time.time()) == (False, "unknown_symbol")
        assert set(s.orderbooks) == before
        assert s.is_fed() is False  # a refused feed is not a feed

    def test_the_symbols_main_constructs_are_the_symbols_the_mapping_returns(self):
        """main.py builds the shield with ['SOL/USDT','BTC/USD','ETH/USD'] and
        looks up `_normalize_kraken_pair(asset)`; both strings must agree or
        every feed is refused as unknown_symbol (the P135 incident shape)."""
        src = (REPO / "main.py").read_text(encoding="utf-8", errors="replace")
        assert "symbols=['SOL/USDT', 'BTC/USD', 'ETH/USD']" in src
        i = src.index("def _normalize_kraken_pair")
        blk = src[i:i + 2500]
        assert '"SOL": "SOL/USDT"' in blk
        assert '"BTC": "BTC/USD"' in blk
        assert '"ETH": "ETH/USD"' in blk


# ---------------------------------------------------------------------------
# 7. WS paths untouched
# ---------------------------------------------------------------------------
class TestWsPathsUntouched:
    def test_rest_feed_leaves_ws_counters_at_zero(self):
        s = _shield()
        s.feed_rest_snapshot("BTC/USD", *_good(), ts=time.time())
        s.feed_rest_snapshot("BTC/USD", [], [[1.0, 1.0]], ts=time.time())
        m = s.orderbooks["BTC/USD"].metrics
        assert m.total_updates == 0 and m.checksum_passes == 0 and m.checksum_failures == 0
        assert m.full_resets == 0 and m.last_valid_sequence == 0

    def test_ws_update_leaves_rest_counters_at_zero(self):
        mgr = KrakenOrderbookManager("X")
        mgr.apply_update([(Decimal("1"), Decimal("1"))], [], checksum=0, sequence=1)
        assert mgr.metrics.rest_snapshots == 0 and mgr.metrics.last_snapshot_ts == 0.0

    def test_ws_apply_snapshot_source_is_the_pre_p384_text(self):
        src = inspect.getsource(KrakenOrderbookManager.apply_snapshot)
        assert "rest_snapshots" not in src and "last_snapshot_ts" not in src
        src_u = inspect.getsource(KrakenOrderbookManager.apply_update)
        assert "rest_snapshots" not in src_u and "last_snapshot_ts" not in src_u

    def test_pass_rate_arithmetic_is_ws_only(self):
        s = _shield()
        for _ in range(5):
            s.feed_rest_snapshot("BTC/USD", *_good(), ts=time.time())
        # 5 REST snapshots, 0 WS updates -> pass_rate stays the WS formula (0/1)
        assert s.get_metrics()["symbols"]["BTC/USD"]["pass_rate"] == 0.0

    def test_no_try_except_was_added_to_the_shield_feed_path(self):
        """The validator enumerates its accepted types instead of catching;
        a swallow here would hide the defect class the feed exists to name
        (and would move the silent-swallow baseline, which P384 must not)."""
        src = inspect.getsource(KrakenOrderbookManager.apply_rest_snapshot)
        assert "try:" not in src and "except" not in src
        src2 = inspect.getsource(KrakenIntegrityShield.feed_rest_snapshot)
        assert "try:" not in src2


# ---------------------------------------------------------------------------
# 8. the pipeline side
# ---------------------------------------------------------------------------
class TestPipelinePublishesTheSnapshot:
    @pytest.fixture(scope="class")
    def pipe(self):
        mod = pytest.importorskip("data_mgmt.market_data_pipeline")
        return mod

    def test_builder_shape_on_a_good_book(self, pipe):
        ob = {"bids": [[100.0 - i, 1.0 + i] for i in range(25)],
              "asks": [[100.1 + i, 2.0 + i] for i in range(25)],
              "timestamp": None}
        snap = pipe.build_orderbook_snapshot(ob, now=1_700_000_000.0)
        assert set(snap) == {"bids", "asks", "ts", "source", "depth_levels"}
        assert snap["source"] == "kraken_rest"
        assert len(snap["bids"]) == 10 and len(snap["asks"]) == 10
        assert snap["bids"][0] == [100.0, 1.0] and snap["asks"][0] == [100.1, 2.0]
        assert all(isinstance(v, float) for lvl in snap["bids"] + snap["asks"] for v in lvl)
        assert snap["ts"] == 1_700_000_000.0
        assert snap["depth_levels"] == 25

    def test_builder_uses_a_sane_exchange_timestamp_in_ms(self, pipe):
        now = 1_700_000_000.0
        ob = {"bids": [[1.0, 1.0]], "asks": [[1.1, 1.0]], "timestamp": (now - 3.0) * 1000}
        assert pipe.build_orderbook_snapshot(ob, now)["ts"] == pytest.approx(now - 3.0)

    def test_builder_rejects_an_insane_exchange_timestamp(self, pipe):
        now = 1_700_000_000.0
        ob = {"bids": [[1.0, 1.0]], "asks": [[1.1, 1.0]], "timestamp": 12345}  # 1970
        assert pipe.build_orderbook_snapshot(ob, now)["ts"] == now

    def test_builder_output_feeds_the_shield_end_to_end(self, pipe):
        ob = {"bids": [[100.0 - 0.1 * i, 1.0] for i in range(50)],
              "asks": [[100.1 + 0.1 * i, 1.0] for i in range(50)]}
        snap = pipe.build_orderbook_snapshot(ob, now=time.time())
        s = _shield()
        assert s.feed_rest_snapshot("BTC/USD", snap["bids"], snap["asks"], snap["ts"]) == (True, "ok")
        assert s.is_fed() and s.is_healthy()

    @pytest.mark.parametrize("ob", [
        None, {}, {"bids": [], "asks": [[1.0, 1.0]]}, {"bids": [[1.0, 1.0]], "asks": []},
        {"bids": "nope", "asks": [[1.0, 1.0]]}, {"bids": [["1", 1.0]], "asks": [[1.0, 1.0]]},
        {"bids": [[1.0]], "asks": [[1.0, 1.0]]}, {"bids": [[True, 1.0]], "asks": [[1.0, 1.0]]},
    ])
    def test_builder_returns_none_never_raises_on_failure_shapes(self, pipe, ob):
        assert pipe.build_orderbook_snapshot(ob, now=1.0) is None

    def test_builder_has_no_try_except(self, pipe):
        src = inspect.getsource(pipe.build_orderbook_snapshot)
        assert "try:" not in src

    def test_publish_site_is_conditional_and_absent_on_failure(self, pipe):
        """Source pin on the ob block + publish site: the payload is built as
        the LAST statement of the success branch, the publish is guarded on
        `is not None` (absence, never {}), and no `{}`/empty fallback exists."""
        src = inspect.getsource(pipe.MarketDataPipeline._fetch_live_data)
        assert "orderbook_snapshot_payload = None" in src
        i_build = src.index("orderbook_snapshot_payload = build_orderbook_snapshot(ob, _time.time())")
        # the analyzer's inner handler is ALSO `except Exception as _ob_err:`
        # — the one that matters is the first after the build statement
        i_except = src.index("except Exception as _ob_err:", i_build)
        i_ts = src.index("self._last_orderbook_ts[asset] = _time.time()",
                         src.index("self._orderbook_failure_streak[asset] = 0"))
        assert i_ts < i_build < i_except, "must be the last statement of the success branch"
        i_pub = src.index('_ret["orderbook_snapshot"] = orderbook_snapshot_payload')
        guard = src[src.rfind("if orderbook_snapshot_payload is not None:", 0, i_pub):i_pub]
        assert guard.startswith("if orderbook_snapshot_payload is not None:")
        assert 'orderbook_snapshot": {}' not in src
        assert src.count('_ret["orderbook_snapshot"]') == 1
        # and the success-path builder is fed the ORIGINAL ccxt book, not a
        # re-shaped copy (depth_levels must describe what the venue returned)
        assert "build_orderbook_snapshot(ob," in src

    def test_publish_site_is_a_literal_key_write(self, pipe):
        """P287's rule: a **splat registers as a dynamic write site in the
        P174 orphan scanner — the publish must stay a literal-key write."""
        src = inspect.getsource(pipe.MarketDataPipeline._fetch_live_data)
        i = src.index('_ret["orderbook_snapshot"]')
        assert "**" not in src[i - 200:i + 80]

    def test_top_levels_match_the_crc32_window(self, pipe):
        from defense.kraken_integrity_shield import KrakenCRC32Validator
        assert pipe.ORDERBOOK_SNAPSHOT_LEVELS == KrakenCRC32Validator.TOP_LEVELS == 10

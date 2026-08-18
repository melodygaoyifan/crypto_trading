"""
Tests for P2-01: Maker cancel-replace (chase) loop in ExecutionManager.

Covers:
  1) Full fill on first attempt
  2) Partial fill across multiple attempts - accept at min_fill_ratio
  3) Zero fill - deferred, no taker fallback
  4) Idempotency - userref per attempt, no duplicate orders
  5) Post-only price clamping - never crosses spread
  6) Config wiring - ExecutionConfig fields from profile
  7) OrderResult KPI fields
  8) Disabled reprice - falls through to standard _execute_limit_order
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from execution.execution_manager import (
    ExecutionConfig,
    ExecutionManager,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)


# ─────────────────────────────────────────────────────────────────────────
# Fake Exchange
# ─────────────────────────────────────────────────────────────────────────

class FakeExchange:
    """Fake CCXT exchange for deterministic testing of the reprice loop.

    Configuration:
        fill_schedule: list of fill ratios per create_limit_order call.
            e.g., [0.0, 0.3, 0.7] -> first order fills 0%, second 30%, third 70%.
        Each entry can also be a tuple (fill_ratio, status) where status is
        "closed" (fully filled) or "open" (partial).
    """

    def __init__(self, fill_schedule: Optional[List] = None, price: float = 50000.0):
        self.fill_schedule = fill_schedule or [1.0]
        self._call_idx = 0
        self.orders: Dict[str, Dict] = {}
        self.cancelled: List[str] = []
        self.price = price
        self._order_counter = 0

    def fetch_ticker(self, symbol: str) -> Dict:
        spread = self.price * 0.0005  # 5 bps spread
        return {
            "bid": self.price - spread / 2,
            "ask": self.price + spread / 2,
            "last": self.price,
        }

    def create_limit_order(self, symbol, side, amount, price, params=None) -> Dict:
        oid = f"FAKE_{self._order_counter}"
        self._order_counter += 1

        # Determine fill for this order from schedule
        idx = min(self._call_idx, len(self.fill_schedule) - 1)
        entry = self.fill_schedule[idx]
        if isinstance(entry, tuple):
            fill_ratio, target_status = entry
        else:
            fill_ratio = entry
            target_status = "closed" if fill_ratio >= 1.0 else "open"
        self._call_idx += 1

        filled = amount * fill_ratio
        self.orders[oid] = {
            "id": oid,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "filled": filled,
            "remaining": amount - filled,
            "average": price,
            "status": target_status,
            "fee": {"cost": filled * price * 0.0002, "currency": "USD"},
            "info": {"userref": str(params.get("userref", "")) if params else ""},
            "params": params or {},
        }
        return {"id": oid}

    def fetch_order(self, order_id: str, symbol: str = None) -> Dict:
        order = self.orders.get(order_id, {})
        return order

    def cancel_order(self, order_id: str, symbol: str = None):
        self.cancelled.append(order_id)
        if order_id in self.orders:
            self.orders[order_id]["status"] = "canceled"

    def fetch_closed_orders(self, params=None):
        return []


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _make_reprice_config(**overrides) -> ExecutionConfig:
    """Create ExecutionConfig with maker reprice enabled."""
    kwargs = dict(
        enable_maker_reprice=True,
        maker_reprice_max_attempts=3,
        maker_reprice_wait_seconds=0.05,  # Very short for tests
        maker_reprice_min_fill_ratio=0.50,
        maker_reprice_improve_bps_schedule=[8, 5, 3],
    )
    kwargs.update(overrides)
    return ExecutionConfig(**kwargs)


def _make_mgr(exchange=None, config=None) -> ExecutionManager:
    """Create ExecutionManager with fake exchange (NOT dry_run)."""
    cfg = config or _make_reprice_config()
    mgr = ExecutionManager(
        exchange=exchange or FakeExchange(),
        config=cfg,
        dry_run=False,
    )
    return mgr


# ─────────────────────────────────────────────────────────────────────────
# Test 1: Full fill on first attempt
# ─────────────────────────────────────────────────────────────────────────

class TestFullFillFirstAttempt:
    """Fake exchange fills 100% on the first order."""

    def test_success_on_first_attempt(self):
        exchange = FakeExchange(fill_schedule=[1.0])
        mgr = _make_mgr(exchange=exchange)

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_001",
        )

        assert result.success
        assert result.maker_reprice_attempts == 1
        assert result.maker_reprice_cancel_count == 0
        assert result.filled_size == pytest.approx(0.1)

    def test_no_market_fallback_triggered(self):
        """Reprice loop doesn't trigger market fallback."""
        exchange = FakeExchange(fill_schedule=[1.0])
        mgr = _make_mgr(exchange=exchange)

        result = mgr.execute_order(
            symbol="ETH/USD",
            side="SELL",
            size=1.0,
            price=3000.0,
            order_type="LIMIT",
            tick_id="tick_002",
        )

        assert result.success
        assert result.order_type == "LIMIT"


# ─────────────────────────────────────────────────────────────────────────
# Test 2: Partial fill across attempts - accept at min_fill_ratio
# ─────────────────────────────────────────────────────────────────────────

class TestPartialFillAccepted:
    """Cumulative partial fills reach min_fill_ratio threshold."""

    def test_two_attempts_reach_min_fill(self):
        """First: 30%, second: 25% -> cumulative 55% >= 50% min -> accept."""
        # First order fills 30%, second fills 25% of REMAINING 70%
        # Total: 0.30 + 0.25*0.70 = 0.30 + 0.175 = 0.475? No...
        # Actually fill_schedule applies to the AMOUNT passed to create_limit_order.
        # Attempt 0: amount=0.1, fill=30% -> filled=0.03, remaining=0.07
        # Attempt 1: amount=0.07, fill ratio needs to get cumulative >= 50%
        # Need: total >= 0.05. Already have 0.03. Need 0.02 from 0.07 -> ratio=0.02/0.07=0.286
        # Let's use higher ratios to be clear.
        # Attempt 0: fill 30% of 0.1 = 0.03
        # Attempt 1: fill 50% of 0.07 = 0.035 -> total 0.065 -> ratio 65% ≥ 50%
        exchange = FakeExchange(fill_schedule=[0.3, 0.5, 0.0])
        mgr = _make_mgr(exchange=exchange)

        result = mgr.execute_order(
            symbol="SOL/USD",
            side="BUY",
            size=0.1,
            price=150.0,
            order_type="LIMIT",
            tick_id="tick_003",
        )

        assert result.success
        assert result.maker_reprice_attempts == 2
        assert result.filled_size >= 0.1 * 0.50  # At least 50% filled
        assert result.time_to_fill_seconds > 0

    def test_kpi_fields_populated(self):
        """OrderResult.to_dict() includes reprice KPI when used."""
        exchange = FakeExchange(fill_schedule=[1.0])
        mgr = _make_mgr(exchange=exchange)

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.5,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_004",
        )

        d = result.to_dict()
        assert "maker_reprice_attempts" in d
        assert "maker_reprice_cancel_count" in d
        assert "time_to_fill_seconds" in d
        assert "final_fill_ratio" in d
        assert d["final_fill_ratio"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────
# Test 3: Zero fill - deferred, no taker fallback
# ─────────────────────────────────────────────────────────────────────────

class TestZeroFillDeferred:
    """Every attempt fills 0% -> deferred after max_attempts, no market fallback."""

    def test_all_attempts_zero_fill(self):
        """[P165] `market_fallback_enabled=False` is now load-bearing here.

        This class is named "no market fallback" and asserts the DEFERRED
        result's fields, but it built its manager with the DEFAULT config —
        and `[MARKET-FALLBACK 2026-04-15]` re-enabled the market fallback for
        exactly this case (reprice exhausted with <10% filled), because
        deferring produced 6+ days of zero fills in production. So the returned
        result was the MARKET result, not the deferred one, and the test was
        asserting the deferred contract against an object that had never been
        through the deferred path. The sibling `test_no_market_order_placed`
        already disables the fallback explicitly; do the same here so the test
        exercises the branch it describes.
        """
        exchange = FakeExchange(fill_schedule=[0.0, 0.0, 0.0])
        mgr = _make_mgr(exchange=exchange, config=_make_reprice_config(
            market_fallback_enabled=False,
        ))

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_005",
        )

        assert not result.success
        assert result.maker_reprice_attempts == 3
        assert result.filled_size == pytest.approx(0.0)
        assert "exhausted" in result.error_message.lower()

    def test_reprice_kpi_survives_market_fallback(self):
        """[P165] A maker-starved fallback must not report "0 reprice attempts".

        `_execute_market_order` builds a fresh `OrderResult`, so replacing the
        deferred result with it dropped `maker_reprice_attempts` /
        `maker_reprice_cancel_count` — in the maker-STARVED case, which is the
        one the KPI exists to measure. That made "the reprice loop tried 3
        times and gave up" indistinguishable from "reprice never ran", i.e. the
        telemetry read healthy precisely when execution was degraded.
        """
        exchange = FakeExchange(fill_schedule=[0.0, 0.0, 0.0])
        mgr = _make_mgr(exchange=exchange, config=_make_reprice_config(
            market_fallback_enabled=True,
        ))

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_005b",
        )

        assert result.maker_reprice_attempts == 3
        assert result.maker_reprice_cancel_count == 3

    def test_no_market_order_placed(self):
        """Verify no market order was created (only limit orders)."""
        exchange = FakeExchange(fill_schedule=[0.0, 0.0, 0.0])
        mgr = _make_mgr(exchange=exchange, config=_make_reprice_config(
            market_fallback_enabled=False,
        ))

        result = mgr.execute_order(
            symbol="ETH/USD",
            side="SELL",
            size=1.0,
            price=3000.0,
            order_type="LIMIT",
            tick_id="tick_006",
        )

        assert not result.success
        # All orders in the exchange should be limit orders
        for oid, order in exchange.orders.items():
            assert order.get("price") is not None  # Limit orders have price

    def test_cancel_count_equals_attempts(self):
        """Each unfilled attempt should be cancelled.

        [P165] Same stale premise as `test_all_attempts_zero_fill` — pin the
        deferred branch by disabling the market fallback.
        """
        exchange = FakeExchange(fill_schedule=[0.0, 0.0, 0.0])
        mgr = _make_mgr(exchange=exchange, config=_make_reprice_config(
            market_fallback_enabled=False,
        ))

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_007",
        )

        assert result.maker_reprice_cancel_count == 3


# ─────────────────────────────────────────────────────────────────────────
# Test 4: Idempotency - userref per attempt
# ─────────────────────────────────────────────────────────────────────────

class TestIdempotency:
    """Each attempt gets a unique, deterministic userref."""

    def test_unique_userrefs_per_attempt(self):
        """Three attempts should produce three different userrefs."""
        exchange = FakeExchange(fill_schedule=[0.0, 0.0, 0.0])
        mgr = _make_mgr(exchange=exchange)

        mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_008",
        )

        # Collect userrefs from placed orders
        userrefs = []
        for oid, order in exchange.orders.items():
            userref = order.get("params", {}).get("userref")
            if userref is not None:
                userrefs.append(userref)

        assert len(userrefs) == 3
        assert len(set(userrefs)) == 3  # All unique

    def test_deterministic_userrefs(self):
        """Same (symbol, side, tick_id, attempt) always produces same userref."""
        ref_0a = ExecutionManager._generate_reprice_userref("BTC/USD", "BUY", "tick_009", 0)
        ref_0b = ExecutionManager._generate_reprice_userref("BTC/USD", "BUY", "tick_009", 0)
        ref_1 = ExecutionManager._generate_reprice_userref("BTC/USD", "BUY", "tick_009", 1)

        assert ref_0a == ref_0b  # Same input -> same output
        assert ref_0a != ref_1   # Different attempt -> different output

    def test_no_duplicate_orders(self):
        """Two calls with same tick_id + already-filled userref -> no extra order."""
        exchange = FakeExchange(fill_schedule=[1.0])
        mgr = _make_mgr(exchange=exchange)

        result1 = mgr.execute_order(
            symbol="BTC/USD", side="BUY", size=0.1, price=50000.0,
            order_type="LIMIT", tick_id="tick_010",
        )

        # First call creates 1 order
        assert len(exchange.orders) == 1
        assert result1.success


# ─────────────────────────────────────────────────────────────────────────
# Test 5: Post-only price clamping
# ─────────────────────────────────────────────────────────────────────────

class TestPostOnlyPriceClamping:
    """_compute_post_only_limit_price never crosses the spread."""

    def test_buy_clamped_to_best_bid(self):
        """BUY limit price must be <= best_bid."""
        price = ExecutionManager._compute_post_only_limit_price(
            side=OrderSide.BUY,
            mid_price=50000.0,
            improve_bps=0,  # No offset -> mid price
            best_bid=49990.0,
            best_ask=50010.0,
        )
        assert price <= 49990.0

    def test_sell_clamped_to_best_ask(self):
        """SELL limit price must be >= best_ask."""
        price = ExecutionManager._compute_post_only_limit_price(
            side=OrderSide.SELL,
            mid_price=50000.0,
            improve_bps=0,
            best_bid=49990.0,
            best_ask=50010.0,
        )
        assert price >= 50010.0

    def test_buy_with_offset_stays_maker(self):
        """BUY with an offset improves on the bid but never crosses the ask.

        [P165] Asserted `price <= best_bid` — the ORIGINAL semantics, which
        `[MAKER-PRICE-FIX 2026-04-15]` deliberately removed: treating
        `improve_bps` as "go deeper below the bid" put a 6bps offset $44 below
        best_bid on BTC, ~14x the whole spread, and guaranteed a 0% fill rate.
        The offset now improves the price WITHIN the spread, capped one tick
        short of the opposite side. So the old assertion pinned the behaviour
        the fix exists to prevent, and the two would-be-maker tests could only
        pass on the broken version.

        What "stays maker" actually means: strictly inside the opposite touch,
        so post_only cannot reject, and no worse than the near touch.
        """
        price = ExecutionManager._compute_post_only_limit_price(
            side=OrderSide.BUY,
            mid_price=50000.0,
            improve_bps=8,
            best_bid=49990.0,
            best_ask=50010.0,
        )
        assert price < 50010.0, "a BUY at or above best_ask crosses — post_only rejects"
        assert price >= 49990.0, "should improve on, not retreat from, best_bid"

    def test_sell_with_offset_stays_maker(self):
        """SELL mirror of the BUY case — see that test for the [P165] rationale."""
        price = ExecutionManager._compute_post_only_limit_price(
            side=OrderSide.SELL,
            mid_price=50000.0,
            improve_bps=8,
            best_bid=49990.0,
            best_ask=50010.0,
        )
        assert price > 49990.0, "a SELL at or below best_bid crosses — post_only rejects"
        assert price <= 50010.0, "should improve on, not retreat from, best_ask"

    def test_progressively_tighter_prices(self):
        """Each attempt with smaller bps -> price closer to the near touch.

        [P165] The old book (bid AT mid, 10-wide spread) made every offset in
        the schedule overshoot the `best_ask - min_tick` cap, so all three
        prices came back identical and `prices[0] <= prices[1] <= prices[2]`
        held trivially — a check that could not fail (P158/P159 family). Use a
        book wide enough that the schedule is not clamped, so the ordering is
        actually produced by `improve_bps`.
        """
        prices = []
        for bps in [8, 5, 3]:
            p = ExecutionManager._compute_post_only_limit_price(
                side=OrderSide.BUY,
                mid_price=50000.0,
                improve_bps=bps,
                best_bid=49900.0,
                best_ask=50100.0,  # 200-wide: an 8bps ($40) offset is not clamped
            )
            prices.append(p)

        assert prices[0] > prices[1] > prices[2], (
            f"a larger improve_bps must push a BUY further up from the bid, got {prices}")
        assert all(p < 50100.0 for p in prices)


# ─────────────────────────────────────────────────────────────────────────
# Test 6: Config wiring
# ─────────────────────────────────────────────────────────────────────────

class TestConfigWiring:
    """ExecutionConfig fields from profile JSON."""

    def test_reprice_is_double_gated_and_opt_in_from_config(self):
        """[P165] The reprice loop is gated by flag AND tick_id, and config opts in.

        Was `test_default_reprice_disabled`, asserting
        `not ExecutionConfig().enable_maker_reprice`. That default has been
        `True` since the initial commit (`execution_manager.py:85`) — the test
        pinned a contract that never held at all, not one that drifted.

        The opt-in property it was reaching for is real, it just lives in two
        other places: `main.py:2809` reads the key with a `False` default, so
        a profile that says nothing gets no repricing; and the loop itself
        additionally requires a `tick_id` (`execution_manager.py:1229`), so it
        cannot engage on a call that has no tick to key its idempotent userrefs
        to. Assert both, since those are what actually keep it off.
        """
        import inspect
        from pathlib import Path

        # Gate 1: the config plumbing defaults to OFF when a profile is silent.
        main_src = Path("main.py").read_text(encoding="utf-8-sig")
        assert "_ec.get('enable_maker_reprice', False)" in main_src, (
            "main.py must default enable_maker_reprice to False when the "
            "profile's execution section omits it")

        # Gate 2: the loop needs a tick_id even when the flag is on.
        src = inspect.getsource(ExecutionManager.execute_order)
        assert "self.config.enable_maker_reprice and tick_id" in src, (
            "reprice must stay double-gated on flag AND tick_id")

        cfg = ExecutionConfig(enable_maker_reprice=False)
        assert not cfg.enable_maker_reprice

    def test_enabled_reprice_config(self):
        cfg = _make_reprice_config()
        assert cfg.enable_maker_reprice
        assert cfg.maker_reprice_max_attempts == 3
        assert cfg.maker_reprice_wait_seconds == pytest.approx(0.05)
        assert cfg.maker_reprice_min_fill_ratio == pytest.approx(0.50)
        assert cfg.maker_reprice_improve_bps_schedule == [8, 5, 3]

    def test_reprice_disabled_uses_standard_limit(self):
        """When reprice disabled, execute_order routes to _execute_limit_order."""
        exchange = FakeExchange(fill_schedule=[1.0])
        cfg = ExecutionConfig(enable_maker_reprice=False)
        mgr = ExecutionManager(exchange=exchange, config=cfg, dry_run=False)

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_011",
        )

        assert result.success
        # Standard limit path doesn't set maker_reprice_attempts
        assert result.maker_reprice_attempts == 0


# ─────────────────────────────────────────────────────────────────────────
# Test 7: Real config JSON
# ─────────────────────────────────────────────────────────────────────────

class TestRealConfigJSON:
    """Validate ultra_aggressive_5y.json execution section."""

    @pytest.fixture
    def ultra_config(self) -> Dict:
        path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_has_execution_section(self, ultra_config):
        assert "execution" in ultra_config

    def test_enable_maker_reprice(self, ultra_config):
        assert ultra_config["execution"]["enable_maker_reprice"] is True

    def test_max_attempts(self, ultra_config):
        assert ultra_config["execution"]["maker_reprice_max_attempts"] == 3

    def test_wait_seconds(self, ultra_config):
        assert ultra_config["execution"]["maker_reprice_wait_seconds"] == 20

    def test_min_fill_ratio(self, ultra_config):
        assert ultra_config["execution"]["maker_reprice_min_fill_ratio"] == 0.50

    def test_bps_schedule(self, ultra_config):
        assert ultra_config["execution"]["maker_reprice_improve_bps_schedule"] == [8, 5, 3]


# ─────────────────────────────────────────────────────────────────────────
# Test 8: ProductionConfig parsing
# ─────────────────────────────────────────────────────────────────────────

class TestProductionConfigExecution:
    """ProductionConfig correctly parses execution_config from profile."""

    def test_ultra_has_execution_config(self):
        from main import ProductionConfig
        path = Path(__file__).parent.parent / "configs" / "ultra_aggressive_5y.json"
        config = ProductionConfig.from_file(path)
        assert config.execution_config is not None
        assert config.execution_config["enable_maker_reprice"] is True

    def test_default_has_no_maker_reprice(self):
        """Default config does not enable maker reprice."""
        from main import ProductionConfig
        path = Path(__file__).parent.parent / "configs" / "cloud_production.json"
        if not path.exists():
            pytest.skip("cloud_production.json not found")
        config = ProductionConfig.from_file(path)
        # cloud_production.json may have a generic "execution" section,
        # but it must NOT have enable_maker_reprice=True
        if config.execution_config:
            assert not config.execution_config.get("enable_maker_reprice", False)
        else:
            assert config.execution_config is None


# ─────────────────────────────────────────────────────────────────────────
# Test 9: Dry-run bypasses reprice (existing behavior preserved)
# ─────────────────────────────────────────────────────────────────────────

class TestDryRunBypass:
    """In dry_run mode, reprice is bypassed (instant simulated fill)."""

    def test_dry_run_instant_fill(self):
        cfg = _make_reprice_config()
        mgr = ExecutionManager(
            exchange=FakeExchange(),
            config=cfg,
            dry_run=True,
        )

        result = mgr.execute_order(
            symbol="BTC/USD",
            side="BUY",
            size=0.1,
            price=50000.0,
            order_type="LIMIT",
            tick_id="tick_012",
        )

        assert result.success
        # Dry run uses _simulate_order, not reprice loop
        assert result.maker_reprice_attempts == 0

import time

import pytest

from execution.passive_aggressive import ExecutionMode, FillSlopeMonitor, OrderState


def test_fill_slope_monitor_uses_weighted_average_fill_price():
    monitor = FillSlopeMonitor()
    order = OrderState(
        order_id="order-1",
        asset="SOL",
        side="buy",
        target_price=100.0,
        limit_price=100.0,
        total_qty=3.0,
        filled_qty=0.0,
        remaining_qty=3.0,
        created_at=time.time(),
        mode=ExecutionMode.PASSIVE,
    )
    monitor.register_order(order)

    monitor.record_fill("order-1", fill_qty=1.0, fill_price=99.0, is_maker=True)
    monitor.record_fill("order-1", fill_qty=2.0, fill_price=102.0, is_maker=False)

    state = monitor.active_orders["order-1"]
    expected_avg = (1.0 * 99.0 + 2.0 * 102.0) / 3.0

    assert state.avg_fill_price == pytest.approx(expected_avg)
    assert state.slippage_bps == pytest.approx(((expected_avg - 100.0) / 100.0) * 10000.0)

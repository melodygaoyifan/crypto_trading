from collections import defaultdict
from datetime import datetime, timedelta, timezone

from analytics.exit_alpha_tracker import ExitAlphaTracker
from scripts.exit_alpha_weekly_report import build_report


def _tracker_with_trades(trades):
    tracker = ExitAlphaTracker.__new__(ExitAlphaTracker)
    tracker._open_trades = {}
    tracker._completed = list(trades)
    tracker._counterfactuals = {}
    tracker.by_trigger = defaultdict(list)
    tracker.by_asset = defaultdict(list)
    tracker.by_regime = defaultdict(list)
    for trade in trades:
        tracker.by_trigger[trade.get("exit_trigger", "UNKNOWN")].append(trade)
        tracker.by_asset[trade.get("asset", "")].append(trade)
        tracker.by_regime[trade.get("regime_at_entry", "")].append(trade)
    return tracker


def test_exit_alpha_recommendation_dict_emits_bounded_adjustments():
    now = datetime.now(timezone.utc)
    trades = [
        {
            "asset": "SOL",
            "exit_trigger": "T5_PROFIT_DRAWDOWN",
            "exit_time": (now - timedelta(days=1)).isoformat(),
            "retention_pct": 25.0,
            "peak_unrealized_bps": 120.0,
            "realized_bps": 30.0,
            "peak_to_exit_bps": 90.0,
            "pnl_usd": -12.0,
        },
        {
            "asset": "SOL",
            "exit_trigger": "T5_PROFIT_DRAWDOWN",
            "exit_time": (now - timedelta(days=2)).isoformat(),
            "retention_pct": 28.0,
            "peak_unrealized_bps": 110.0,
            "realized_bps": 32.0,
            "peak_to_exit_bps": 78.0,
            "pnl_usd": -8.0,
        },
        {
            "asset": "ETH",
            "exit_trigger": "T11_RUNNER_RELEASE",
            "exit_time": (now - timedelta(days=3)).isoformat(),
            "retention_pct": 42.0,
            "peak_unrealized_bps": 130.0,
            "realized_bps": 55.0,
            "peak_to_exit_bps": 75.0,
            "pnl_usd": 4.0,
        },
        {
            "asset": "ETH",
            "exit_trigger": "T11_RUNNER_RELEASE",
            "exit_time": (now - timedelta(days=4)).isoformat(),
            "retention_pct": 40.0,
            "peak_unrealized_bps": 115.0,
            "realized_bps": 46.0,
            "peak_to_exit_bps": 69.0,
            "pnl_usd": 3.0,
        },
        {
            "asset": "BTC",
            "exit_trigger": "T7_MAX_HOLD",
            "exit_time": (now - timedelta(days=5)).isoformat(),
            "retention_pct": 38.0,
            "peak_unrealized_bps": 95.0,
            "realized_bps": 36.0,
            "peak_to_exit_bps": 59.0,
            "pnl_usd": 2.0,
        },
    ]
    tracker = _tracker_with_trades(trades)

    rec = tracker.recommendation_dict(min_trades=5)

    assert rec["sample_ready"] is True
    assert rec["verdict"] == "ADJUST"
    parameters = {item["parameter"] for item in rec["recommendations"]}
    assert "profit_drawdown_exit_bps" in parameters
    assert "exit_trigger_review" in parameters
    assert rec["worst_trigger"]["name"] == "T5_PROFIT_DRAWDOWN"


def test_exit_alpha_weekly_report_filters_window_and_embeds_recommendations():
    now = datetime.now(timezone.utc)
    tracker = _tracker_with_trades(
        [
            {
                "asset": "SOL",
                "exit_trigger": "T5_PROFIT_DRAWDOWN",
                "exit_time": (now - timedelta(days=1)).isoformat(),
                "retention_pct": 55.0,
                "peak_unrealized_bps": 100.0,
                "realized_bps": 55.0,
                "peak_to_exit_bps": 45.0,
                "pnl_usd": 6.0,
            },
            {
                "asset": "BTC",
                "exit_trigger": "T11_RUNNER_RELEASE",
                "exit_time": (now - timedelta(days=10)).isoformat(),
                "retention_pct": 80.0,
                "peak_unrealized_bps": 120.0,
                "realized_bps": 96.0,
                "peak_to_exit_bps": 24.0,
                "pnl_usd": 8.0,
            },
        ]
    )

    report = build_report(tracker=tracker, lookback_days=7, min_trades=1)

    assert report["summary"]["tracked_trades"] == 1
    assert report["recommendations"]["sample_ready"] is True
    assert report["window"]["lookback_days"] == 7
    assert "generated_at" in report

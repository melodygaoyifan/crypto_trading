from main import HMATSProductionRunner


def _seed_runner() -> HMATSProductionRunner:
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._profit_calibration_shadow_enabled = True
    runner._profit_calibration_shadow_min_closed_trades_per_side = 2
    runner._profit_calibration_shadow_tighten_long_avg_bps_below = -10.0
    runner._profit_calibration_shadow_tighten_long_total_pnl_usd_below = -20.0
    runner._profit_calibration_shadow_keep_long_avg_bps_above = 10.0
    runner._profit_calibration_shadow_relax_short_avg_bps_above = 15.0
    runner._profit_calibration_shadow_relax_short_total_pnl_usd_above = 20.0
    runner._realized_pnl_breakdown = runner._new_realized_pnl_breakdown()
    return runner


def test_profit_calibration_shadow_emits_side_specific_recommendations():
    runner = _seed_runner()
    runner._realized_pnl_breakdown["closed_trade_count"] = 4
    runner._realized_pnl_breakdown["by_side"] = {
        "long": {"count": 2, "total_pnl_usd": -24.0, "avg_pnl_bps": -12.0},
        "short": {"count": 2, "total_pnl_usd": 28.0, "avg_pnl_bps": 18.0},
    }

    summary = runner._get_profit_calibration_shadow()

    assert summary["enabled"] is True
    assert summary["mode"] == "SHADOW"
    assert summary["by_side"]["long"]["recommendation"] == "TIGHTEN_QUIET_LONG"
    assert summary["by_side"]["short"]["recommendation"] == "RELAX_SHORT_STRUCTURE_CANDIDATE"


def test_profit_calibration_shadow_requires_minimum_sample_per_side():
    runner = _seed_runner()
    runner._realized_pnl_breakdown["closed_trade_count"] = 2
    runner._realized_pnl_breakdown["by_side"] = {
        "long": {"count": 1, "total_pnl_usd": 15.0, "avg_pnl_bps": 11.0},
        "short": {"count": 1, "total_pnl_usd": 18.0, "avg_pnl_bps": 17.0},
    }

    summary = runner._get_profit_calibration_shadow()

    assert summary["by_side"]["long"]["recommendation"] == "INSUFFICIENT_SAMPLE"
    assert summary["by_side"]["short"]["recommendation"] == "INSUFFICIENT_SAMPLE"

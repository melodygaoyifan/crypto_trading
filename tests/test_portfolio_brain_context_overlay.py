import pytest

from portfolio.portfolio_brain_offensive import PortfolioBrainOffensive


def test_portfolio_brain_live_context_overlays_raw_weights():
    brain = PortfolioBrainOffensive(assets=["BTC", "ETH", "SOL"], shadow_mode=False)
    brain.current_weights = {"BTC": 1.0 / 3.0, "ETH": 1.0 / 3.0, "SOL": 1.0 / 3.0}

    brain.update_live_context(
        "SOL",
        expected_net_alpha_bps=18.0,
        toxicity_score=0.10,
        side_avg_realized_pnl_bps=22.0,
        side_total_realized_pnl_usd=55.0,
    )
    brain.update_live_context(
        "BTC",
        expected_net_alpha_bps=2.0,
        toxicity_score=0.85,
        side_avg_realized_pnl_bps=-12.0,
        side_total_realized_pnl_usd=-40.0,
    )

    weights = brain.get_weights()
    snap = brain.snapshot()

    assert weights["SOL"] > (1.0 / 3.0)
    assert weights["BTC"] < 1.0 / 3.0
    assert snap["raw_weights"]["SOL"] == pytest.approx(1.0 / 3.0)
    assert snap["context_multipliers"]["SOL"] > snap["context_multipliers"]["BTC"]


def test_portfolio_brain_cold_start_prior_uses_live_context():
    brain = PortfolioBrainOffensive(
        assets=["BTC", "ETH", "SOL"],
        shadow_mode=False,
        cold_start_prior_enabled=True,
        cold_start_prior_floor=0.15,
        cold_start_prior_max_multiplier=1.40,
    )
    brain.update_live_context(
        "SOL",
        expected_net_alpha_bps=16.0,
        toxicity_score=0.10,
        side_avg_realized_pnl_bps=18.0,
        side_total_realized_pnl_usd=40.0,
    )
    brain.update_live_context(
        "BTC",
        expected_net_alpha_bps=1.0,
        toxicity_score=0.90,
        side_avg_realized_pnl_bps=-10.0,
        side_total_realized_pnl_usd=-30.0,
    )

    brain._rebalance()

    assert brain.current_weights["SOL"] > (1.0 / 3.0)
    assert brain.current_weights["BTC"] < (1.0 / 3.0)
    assert brain.snapshot()["cold_start_prior_active"] is True

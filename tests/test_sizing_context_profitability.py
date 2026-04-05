from orchestration.aggressive_allocator import AggressiveAllocator
from portfolio.portfolio_brain_offensive import PortfolioBrainOffensive


def test_aggressive_allocator_rewards_crowding_alignment_and_exit_retention():
    supportive = AggressiveAllocator.compute_score(
        0.6,
        0.8,
        0.8,
        expected_net_alpha_bps=12.0,
        toxicity_score=0.15,
        side_avg_realized_pnl_bps=15.0,
        side_total_realized_pnl_usd=30.0,
        crowding_alignment_score=0.8,
        panic_score=0.1,
        exit_avg_retention_pct=78.0,
        exit_avg_peak_to_exit_bps=22.0,
    )
    hostile = AggressiveAllocator.compute_score(
        0.6,
        0.8,
        0.8,
        expected_net_alpha_bps=12.0,
        toxicity_score=0.15,
        side_avg_realized_pnl_bps=15.0,
        side_total_realized_pnl_usd=30.0,
        crowding_alignment_score=-0.8,
        panic_score=0.8,
        exit_avg_retention_pct=28.0,
        exit_avg_peak_to_exit_bps=95.0,
    )

    assert supportive > hostile


def test_portfolio_brain_context_penalizes_crowding_headwinds_and_rewards_exit_quality():
    brain = PortfolioBrainOffensive(assets=["BTC", "ETH", "SOL"], shadow_mode=False)
    brain.current_weights = {"BTC": 1.0 / 3.0, "ETH": 1.0 / 3.0, "SOL": 1.0 / 3.0}

    brain.update_live_context(
        "SOL",
        expected_net_alpha_bps=14.0,
        toxicity_score=0.10,
        side_avg_realized_pnl_bps=18.0,
        side_total_realized_pnl_usd=50.0,
        crowding_alignment_score=0.8,
        panic_score=0.1,
        extreme_crowding=False,
        exit_avg_retention_pct=80.0,
        exit_avg_peak_to_exit_bps=20.0,
    )
    brain.update_live_context(
        "BTC",
        expected_net_alpha_bps=14.0,
        toxicity_score=0.10,
        side_avg_realized_pnl_bps=18.0,
        side_total_realized_pnl_usd=50.0,
        crowding_alignment_score=-0.8,
        panic_score=0.8,
        extreme_crowding=True,
        exit_avg_retention_pct=25.0,
        exit_avg_peak_to_exit_bps=90.0,
    )

    weights = brain.get_weights()
    snap = brain.snapshot()

    assert weights["SOL"] > weights["BTC"]
    assert snap["context_multipliers"]["SOL"] > snap["context_multipliers"]["BTC"]

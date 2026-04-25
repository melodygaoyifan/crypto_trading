"""
Smoke tests for training/exit_drl/generate_expert_trajectories.py.

These are cheap deterministic tests — NO access to parquet files, no long runs.
Goal: catch shape/oracle/reward regressions on a synthetic bar sequence.
"""
import numpy as np
import pandas as pd
import pytest

from training.exit_drl.generate_expert_trajectories import (
    ACTION_HOLD, ACTION_PARTIAL_EXIT, ACTION_RELEASE_RUNNER, ACTION_EXIT_ALL,
    ACTION_DIM, STATE_DIM, MARKET_COLS, FLOW_COLS,
    LOOKAHEAD_BARS, MIN_BARS_HELD, MAX_BARS_HELD,
    build_state, compute_reward, generate_trajectories, oracle_exit_action,
)


def _synthetic_df(n_bars: int, price_series: np.ndarray) -> pd.DataFrame:
    """Build a parquet-like dataframe for trajectory generation."""
    assert len(price_series) == n_bars
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='4h'),
        'open': price_series,
        'high': price_series * 1.005,
        'low': price_series * 0.995,
        'close': price_series,
        'volume': np.ones(n_bars) * 1000.0,
        'atr_14': price_series * 0.02,  # 2% ATR
        'regime': np.zeros(n_bars),
        'regime_confidence': np.full(n_bars, 0.5),
    })
    for i in range(8):
        df[f'regime_proba_{i}'] = 0.125
    for col in MARKET_COLS + FLOW_COLS:
        df[col] = 0.0
    return df


def test_state_dimensions():
    df = _synthetic_df(60, np.linspace(100.0, 110.0, 60))
    state = build_state(idx=10, entry_idx=5, direction=+1, df=df,
                        close=df['close'].to_numpy(dtype=np.float64))
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    assert not np.any(np.isnan(state))
    assert not np.any(np.isinf(state))


def test_oracle_forces_exit_at_max_hold():
    prices = np.ones(MAX_BARS_HELD + 10) * 100.0
    action = oracle_exit_action(
        entry_idx=0, current_idx=MAX_BARS_HELD,
        direction=+1, prices=prices, atr_pct_at_entry=0.02,
    )
    assert action == ACTION_EXIT_ALL


def test_oracle_holds_before_min_bars_held():
    prices = np.ones(20) * 100.0
    action = oracle_exit_action(
        entry_idx=0, current_idx=MIN_BARS_HELD - 1,
        direction=+1, prices=prices, atr_pct_at_entry=0.02,
    )
    assert action == ACTION_HOLD


def test_oracle_stops_on_adverse_move():
    # Long entry, price drops hard → unrealized PnL below stop ATR → EXIT_ALL
    prices = np.concatenate([[100.0], np.full(10, 95.0)])
    action = oracle_exit_action(
        entry_idx=0, current_idx=5,
        direction=+1, prices=prices, atr_pct_at_entry=0.02,
    )
    # -5% loss vs 1.5*2% = 3% threshold → EXIT_ALL
    assert action == ACTION_EXIT_ALL


def test_oracle_partial_exit_near_peak():
    # Long entry at 100; price rises to 103 at peak, then starts falling
    prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 102.0, 101.0])
    # At idx=6 (peak), oracle should fire PARTIAL_EXIT (unrealized >= 20bps AND at peak)
    action = oracle_exit_action(
        entry_idx=0, current_idx=6,
        direction=+1, prices=prices, atr_pct_at_entry=0.02,
    )
    assert action == ACTION_PARTIAL_EXIT


def test_reward_is_sparse():
    # HOLD should return 0.0 regardless of PnL
    assert compute_reward(prev_pnl=0.01, curr_pnl=0.02, action=ACTION_HOLD) == 0.0
    # EXIT_ALL returns full PnL (signed)
    assert compute_reward(prev_pnl=0.01, curr_pnl=0.05, action=ACTION_EXIT_ALL) == pytest.approx(0.05)
    assert compute_reward(prev_pnl=-0.01, curr_pnl=-0.03, action=ACTION_EXIT_ALL) == pytest.approx(-0.03)
    # PARTIAL_EXIT caps at 25% of positive PnL
    assert compute_reward(prev_pnl=0.01, curr_pnl=0.04, action=ACTION_PARTIAL_EXIT) == pytest.approx(0.01)
    # PARTIAL_EXIT returns 0 on loss (don't reward locking in losses mid-peak)
    assert compute_reward(prev_pnl=-0.01, curr_pnl=-0.02, action=ACTION_PARTIAL_EXIT) == 0.0


def test_full_trajectory_generation_shapes():
    # Generate trajectories on a small rising-then-falling synthetic asset
    n = 250
    prices = 100.0 + 5.0 * np.sin(np.linspace(0, 6 * np.pi, n))
    df = _synthetic_df(n, prices)
    out = generate_trajectories(df, 'SMOKE')

    assert out['observations'].ndim == 2
    assert out['observations'].shape[1] == STATE_DIM
    assert out['actions'].ndim == 1
    assert out['rewards'].shape == out['actions'].shape
    assert out['terminals'].shape == out['actions'].shape
    assert out['next_observations'].shape == out['observations'].shape

    # Every episode terminates (no infinite trajectories)
    # -> last action of every episode is EXIT_ALL or RELEASE_RUNNER
    terminal_actions = out['actions'][out['terminals']]
    assert len(terminal_actions) > 0
    assert set(terminal_actions.tolist()).issubset({ACTION_EXIT_ALL, ACTION_RELEASE_RUNNER})

    # Action distribution is in valid range
    dist = np.bincount(out['actions'], minlength=ACTION_DIM)
    assert dist.sum() == len(out['actions'])
    # HOLD should still be the majority on any reasonable trajectory
    assert dist[ACTION_HOLD] > dist[ACTION_EXIT_ALL]


def test_trajectory_generation_handles_both_directions():
    n = 200
    prices = 100.0 + np.linspace(0, 10, n)  # monotonic rise
    df = _synthetic_df(n, prices)
    out = generate_trajectories(df, 'SMOKE')
    # Long trajectories should produce at least some PARTIAL_EXIT (price keeps rising)
    # Short trajectories should hit the stop (price moves against)
    assert out['actions'].tolist().count(ACTION_EXIT_ALL) > 0

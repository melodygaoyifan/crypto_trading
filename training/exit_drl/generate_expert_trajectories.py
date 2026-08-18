#!/usr/bin/env python3
"""
training/exit_drl/generate_expert_trajectories.py

Generate oracle-labeled expert trajectories for Exit DRL training.

Per HMATS_EXIT_DRL_TRAINING.md Stage 1:
- Per-asset (no cross-asset mixing).
- Long + short generated separately.
- Oracle uses future-peeking (lookahead window) to label optimal exit actions.
- Output: D4RL-style npz with observations, actions, rewards, terminals, next_observations.

Action space (v1 — ESCALATE dropped pending oracle rule):
  0 HOLD
  1 PARTIAL_EXIT      (lock in 25%)
  2 RELEASE_RUNNER    (release 50%)
  3 EXIT_ALL          (close fully)

State: 40-d (remapped from the spec's 48-d to fit the actual parquet schema).
  Position  (8): direction, bars_held_norm, unrealized_pnl, max_unrealized,
                 drawdown_from_peak, tranche_proxy, entry_price_rel, is_runner
  Regime    (10): regime_proba_0..7, regime_confidence, regime_scalar
  Market    (14): ret_4h, ret_12h, ret_24h, ret_48h, ret_168h,
                  vol_24h, vol_ratio_s, atr_14_denoised,
                  rsi_14, macd_12_26, bb_pos_20, bb_width_20,
                  vol_zscore, vol_spike
  Flow      (8): funding_rate_zscore, oi_change_5d, liq_imbalance,
                 taker_ratio_zscore, tradecount_zscore, taker_vol_momentum,
                 panic, capitulation
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# ────────────────────────────────────────────────────────────────
# Action space (v1 — 4 actions; ESCALATE deferred)
# ────────────────────────────────────────────────────────────────
ACTION_HOLD = 0
ACTION_PARTIAL_EXIT = 1
ACTION_RELEASE_RUNNER = 2
ACTION_EXIT_ALL = 3
ACTION_DIM = 4

# ────────────────────────────────────────────────────────────────
# State layout — 40-d. Keep in sync with agents/exit_drl_agent.py.
# ────────────────────────────────────────────────────────────────
STATE_DIM = 40
POSITION_DIM = 8
REGIME_DIM = 10
MARKET_DIM = 14
FLOW_DIM = 8
assert POSITION_DIM + REGIME_DIM + MARKET_DIM + FLOW_DIM == STATE_DIM

MARKET_COLS = [
    'ret_4h', 'ret_12h', 'ret_24h', 'ret_48h', 'ret_168h',
    'vol_24h', 'vol_ratio_s', 'atr_14_denoised',
    'rsi_14', 'macd_12_26', 'bb_pos_20', 'bb_width_20',
    'vol_zscore', 'vol_spike',
]
FLOW_COLS = [
    'funding_rate_zscore', 'oi_change_5d', 'liq_imbalance',
    'taker_ratio_zscore', 'tradecount_zscore', 'taker_vol_momentum',
    'panic', 'capitulation',
]
assert len(MARKET_COLS) == MARKET_DIM
assert len(FLOW_COLS) == FLOW_DIM

# ────────────────────────────────────────────────────────────────
# Trajectory parameters (from spec Stage 1)
# ────────────────────────────────────────────────────────────────
LOOKAHEAD_BARS = 48     # 8 days of 4H bars
MIN_BARS_HELD = 2       # oracle always HOLDs for first 2 bars
MAX_BARS_HELD = 48      # force EXIT_ALL at this horizon
ENTRY_STRIDE = 6        # one entry sample every 6 bars (≈1 day)

# Oracle thresholds
PARTIAL_PROFIT_MIN_BPS = 20        # 0.20% — minimum profit to consider PARTIAL_EXIT
PEAK_TOLERANCE_BPS = 20            # within 20bps of future peak → PARTIAL_EXIT
POST_PEAK_DRAWDOWN_PCT = 0.30      # give back 30% of peak → RELEASE_RUNNER
STOP_ATR_MULT = 1.5                # opposite move > 1.5 ATR → EXIT_ALL


def oracle_exit_action(
    entry_idx: int,
    current_idx: int,
    direction: int,
    prices: np.ndarray,
    atr_pct_at_entry: float,
) -> int:
    """
    Oracle decides the best exit action at `current_idx` using future info.

    Rule precedence (first match wins):
      1. bars_held >= MAX_BARS_HELD              → EXIT_ALL
      2. bars_held < MIN_BARS_HELD                → HOLD
      3. unrealized_pnl < -STOP_ATR_MULT*atr     → EXIT_ALL (stop out)
      4. current bar is at/near future peak
         AND unrealized_pnl >= PARTIAL_PROFIT_MIN_BPS → PARTIAL_EXIT
      5. past a peak with >30% drawdown          → RELEASE_RUNNER
      6. otherwise                                → HOLD
    """
    bars_held = current_idx - entry_idx
    if bars_held >= MAX_BARS_HELD:
        return ACTION_EXIT_ALL
    if bars_held < MIN_BARS_HELD:
        return ACTION_HOLD

    entry_price = prices[entry_idx]
    if entry_price <= 0:
        return ACTION_HOLD
    current_price = prices[current_idx]
    unrealized_pnl = direction * (current_price - entry_price) / entry_price

    # Stop out
    if unrealized_pnl < -STOP_ATR_MULT * max(atr_pct_at_entry, 1e-4):
        return ACTION_EXIT_ALL

    # Lookahead window from current_idx to entry_idx + LOOKAHEAD_BARS
    end_idx = min(entry_idx + LOOKAHEAD_BARS, len(prices) - 1)
    if current_idx >= end_idx:
        return ACTION_EXIT_ALL
    remaining = prices[current_idx:end_idx + 1]
    future_pnls = direction * (remaining - entry_price) / entry_price
    peak_future = float(future_pnls.max())
    peak_offset = int(future_pnls.argmax())

    # Historical peak up to now (from entry)
    past_pnls = direction * (prices[entry_idx:current_idx + 1] - entry_price) / entry_price
    past_peak = float(past_pnls.max())

    # Near-peak partial exit
    near_peak = (peak_future - unrealized_pnl) < (PEAK_TOLERANCE_BPS / 10000.0)
    if unrealized_pnl >= (PARTIAL_PROFIT_MIN_BPS / 10000.0) and near_peak:
        return ACTION_PARTIAL_EXIT

    # Post-peak drawdown release
    if past_peak > 0 and peak_offset == 0:  # peak is behind us
        drawdown_ratio = (past_peak - unrealized_pnl) / past_peak if past_peak > 0 else 0.0
        if drawdown_ratio > POST_PEAK_DRAWDOWN_PCT and past_peak > (PARTIAL_PROFIT_MIN_BPS / 10000.0):
            return ACTION_RELEASE_RUNNER

    return ACTION_HOLD


def build_state(
    idx: int,
    entry_idx: int,
    direction: int,
    df: pd.DataFrame,
    close: np.ndarray,
) -> np.ndarray:
    """Build 40-d state vector at bar `idx` given entry at `entry_idx`."""
    bars_held = idx - entry_idx
    entry_price = close[entry_idx]
    current_price = close[idx]

    unrealized_pnl = direction * (current_price - entry_price) / entry_price
    segment_pnls = direction * (close[entry_idx:idx + 1] - entry_price) / entry_price
    max_unrealized = float(segment_pnls.max()) if len(segment_pnls) else 0.0
    drawdown_from_peak = max(0.0, max_unrealized - unrealized_pnl)

    position = np.array([
        float(direction),
        min(bars_held / MAX_BARS_HELD, 1.0),
        unrealized_pnl,
        max_unrealized,
        drawdown_from_peak,
        min(bars_held / 12.0, 1.0),                     # tranche proxy
        current_price / entry_price - 1.0,
        1.0 if bars_held > 12 else 0.0,                  # is_runner proxy
    ], dtype=np.float32)

    row = df.iloc[idx]
    regime_scalar = float(row.get('regime', 0.0) or 0.0) / 7.0  # normalize to [0,1]
    regime = np.array([
        float(row.get(f'regime_proba_{i}', 0.0) or 0.0) for i in range(8)
    ] + [
        float(row.get('regime_confidence', 0.0) or 0.0),
        regime_scalar,
    ], dtype=np.float32)

    market = np.array([
        float(row.get(c, 0.0) or 0.0) for c in MARKET_COLS
    ], dtype=np.float32)

    flow = np.array([
        float(row.get(c, 0.0) or 0.0) for c in FLOW_COLS
    ], dtype=np.float32)

    state = np.concatenate([position, regime, market, flow])
    assert state.shape == (STATE_DIM,), f"State dim mismatch: {state.shape}"
    # Defensive NaN scrub — training data should be clean but be paranoid.
    return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


def compute_reward(
    prev_pnl: float,
    curr_pnl: float,
    action: int,
) -> float:
    """
    Sparse reward (spec §Stage 4 pitfall #1): only pay out on exit actions.
    - HOLD                 → 0.0 (no dense shaping — avoids premature exit bias)
    - PARTIAL_EXIT         → 25% of current PnL if positive, else 0
    - RELEASE_RUNNER       → 50% of current PnL
    - EXIT_ALL             → full current PnL

    Note: all rewards are signed (a losing EXIT_ALL returns negative reward,
    which is correct — the oracle was forced to stop out).
    """
    if action == ACTION_HOLD:
        return 0.0
    if action == ACTION_PARTIAL_EXIT:
        return 0.25 * max(curr_pnl, 0.0)
    if action == ACTION_RELEASE_RUNNER:
        return 0.50 * curr_pnl
    if action == ACTION_EXIT_ALL:
        return curr_pnl
    return 0.0


def generate_trajectories(df: pd.DataFrame, asset: str) -> Dict[str, np.ndarray]:
    """Generate all trajectories for one asset."""
    observations, actions, rewards, terminals, next_observations = [], [], [], [], []
    close = df['close'].to_numpy(dtype=np.float64)
    # atr_14 is raw-ATR; convert to ATR-as-fraction-of-price for stop thresholds
    atr_14 = df['atr_14'].to_numpy(dtype=np.float64)
    atr_pct_series = np.where(close > 0, atr_14 / close, 0.02)

    n_bars = len(df)
    print(f"  [{asset}] bars={n_bars}, lookahead={LOOKAHEAD_BARS}, stride={ENTRY_STRIDE}")

    traj_count = 0
    skipped_short = 0
    for entry_idx in range(24, n_bars - LOOKAHEAD_BARS, ENTRY_STRIDE):
        atr_at_entry = float(atr_pct_series[entry_idx]) if atr_pct_series[entry_idx] > 0 else 0.02
        for direction in (+1, -1):
            last_obs = None
            for bar_offset in range(MIN_BARS_HELD, LOOKAHEAD_BARS):
                current_idx = entry_idx + bar_offset
                if current_idx >= n_bars - 1:
                    break

                state = build_state(current_idx, entry_idx, direction, df, close)
                action = oracle_exit_action(
                    entry_idx, current_idx, direction, close, atr_at_entry,
                )

                entry_price = close[entry_idx]
                prev_price = close[current_idx - 1]
                curr_price = close[current_idx]
                prev_pnl = direction * (prev_price - entry_price) / entry_price
                curr_pnl = direction * (curr_price - entry_price) / entry_price

                is_terminal = action in (ACTION_EXIT_ALL, ACTION_RELEASE_RUNNER)
                reward = compute_reward(prev_pnl, curr_pnl, action)

                next_idx = min(current_idx + 1, n_bars - 1)
                next_state = build_state(next_idx, entry_idx, direction, df, close)

                observations.append(state)
                actions.append(action)
                rewards.append(reward)
                terminals.append(is_terminal)
                next_observations.append(next_state)
                last_obs = current_idx

                if is_terminal:
                    traj_count += 1
                    break
            else:
                # Didn't terminate within window — force-close at horizon
                if last_obs is not None:
                    traj_count += 1
                    # Overwrite last transition's terminal flag + reward
                    terminals[-1] = True
                    actions[-1] = ACTION_EXIT_ALL
                    entry_price = close[entry_idx]
                    final_pnl = direction * (close[last_obs] - entry_price) / entry_price
                    rewards[-1] = compute_reward(0.0, final_pnl, ACTION_EXIT_ALL)
                else:
                    skipped_short += 1

    # [P188] Annotated so the values are each checked against the declared
    # return type. Without it mypy joins the five differently-dtyped asarray
    # results to `object` and reports the return as incompatible. The error was
    # pre-existing and invisible: this module only entered mypy's import graph
    # once train_exit_sac started importing STATE_DIM/ACTION_DIM from it.
    out: Dict[str, np.ndarray] = {
        'observations': np.asarray(observations, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.int64),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'terminals': np.asarray(terminals, dtype=bool),
        'next_observations': np.asarray(next_observations, dtype=np.float32),
    }
    dist = np.bincount(out['actions'], minlength=ACTION_DIM)
    print(
        f"  [{asset}] trajectories={traj_count}, transitions={len(observations)}, "
        f"skipped_short={skipped_short}, "
        f"actions=[HOLD={dist[0]}, PARTIAL={dist[1]}, RELEASE={dist[2]}, EXIT={dist[3]}]"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True, choices=['BTC', 'ETH', 'SOL'])
    parser.add_argument(
        '--data-path',
        default='training/training_data/drl_training',
        help='directory containing {asset}_4H_full.parquet',
    )
    parser.add_argument(
        '--output-path',
        default='training/exit_drl/data',
        help='output directory for npz file',
    )
    parser.add_argument(
        '--max-bars',
        type=int,
        default=None,
        help='optional cap on dataframe rows (smoke tests)',
    )
    args = parser.parse_args()

    data_file = Path(args.data_path) / f'{args.asset}_4H_full.parquet'
    print(f"Loading {data_file}")
    df = pd.read_parquet(data_file)
    if args.max_bars:
        df = df.iloc[:args.max_bars].reset_index(drop=True)

    dataset = generate_trajectories(df, args.asset)

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.asset}_exit_trajectories.npz'
    np.savez_compressed(output_file, **dataset)

    # Sidecar: state spec so trainer + runtime agent can sanity-check it.
    spec = {
        'asset': args.asset,
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'action_names': ['HOLD', 'PARTIAL_EXIT', 'RELEASE_RUNNER', 'EXIT_ALL'],
        'market_cols': MARKET_COLS,
        'flow_cols': FLOW_COLS,
        'oracle_thresholds': {
            'partial_profit_min_bps': PARTIAL_PROFIT_MIN_BPS,
            'peak_tolerance_bps': PEAK_TOLERANCE_BPS,
            'post_peak_drawdown_pct': POST_PEAK_DRAWDOWN_PCT,
            'stop_atr_mult': STOP_ATR_MULT,
        },
        'horizon_bars': LOOKAHEAD_BARS,
    }
    (output_dir / f'{args.asset}_state_spec.json').write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print(f"Saved {output_file} ({dataset['observations'].shape[0]} transitions)")


if __name__ == '__main__':
    main()

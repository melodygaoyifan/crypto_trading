#!/usr/bin/env python3
"""
================================================================================
HMATS v7 - Ultimate Rebuild DRL Training (train_drl_full.py)
================================================================================

Key constraints (from 5 rounds of failure):
  - DummyVecEnv ONLY (SubprocVecEnv deadlocks on Windows, no speedup on WSL2)
  - ent_coef=0.1 FIXED (auto causes gradient explosion 0.2->10^23->NaN)
  - checkpoint_freq=500K (lost 500K+ steps on deadlocks)
  - Sequential per-asset (parallel training deadlocked all 3 at 1.47M steps)
  - gap=42, window_size=10, n_folds=3, val_ratio=0.15

Data pipeline improvements:
  - P2: RobustScaler normalization (fit on train, transform both, saved per fold)
  - P3: Rare regime reward weighting (inverse sqrt frequency, clipped [0.5, 2.0])
  - P4: K-Fold purging (--purge-window N adds N bars to gap between train/val)

Usage:
    python train_drl_full.py --asset BTC --data training/training_data/drl_training/BTC_4H_full.parquet
    python train_drl_full.py --asset ETH --data training/training_data/drl_training/ETH_4H_full.parquet
    python train_drl_full.py --asset SOL --data training/training_data/drl_training/SOL_4H_full.parquet --timesteps 3000000 --lr 2e-5
    python train_drl_full.py --asset BTC --resume  # Resume from last checkpoint

================================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from sb3_contrib import TQC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("DRL_Full")

# ── Path setup: add project root for cross-package imports ──
_TRAINING_DIR = Path(__file__).resolve().parent          # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root
sys.path.insert(0, str(PROJECT_ROOT))
# training/ must be higher priority than ROOT so training/models/
# (feature_extractors) is found before ROOT/models/ (runtime models)
sys.path.insert(0, str(_TRAINING_DIR))

from core.regime_smoother import RegimeSmoother  # noqa: E402 - single source of truth


# =============================================================================
# ULTIMATE Preset (proven stable on RTX 5090)
# =============================================================================

ULTIMATE_PRESET = {
    "policy": "MlpPolicy",
    "learning_rate": 1.5255e-05,     # Config 1 Optuna winner (was 3e-5)
    "buffer_size": 500_000,          # Stage 9.7 fix: n_stack=8 memory (was 2M)
    "learning_starts": 30_000,       # Phase A: 63.6% importance, confirmed at 30K
    "batch_size": 256,               # pre_rebuild proven: 256 -> peak +33.69
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 4,             # pre_rebuild proven: 4 -> 2x more gradient updates per step
    "ent_coef": 0.1,                 # FIXED - NEVER "auto" (gradient explosion)
    "target_update_interval": 1,
    "top_quantiles_to_drop_per_net": 2,
    "policy_kwargs": {
        "net_arch": [384, 384, 256],
        "n_critics": 2,
        "n_quantiles": 24,           # Config 1 Optuna winner (was 32)
    },
}

ASSET_OVERRIDES = {
    "BTC": {"total_timesteps": 2_500_000, "learning_rate": 3e-5},
    "ETH": {"total_timesteps": 2_500_000, "learning_rate": 3e-5},
    "SOL": {"total_timesteps": 3_000_000, "learning_rate": 2e-5, "sol_vol_multiplier": 0.6},
}

BEAR_REGIMES = frozenset(["PANIC_SELLOFF", "VOLATILE_CHOP", "EXTREME_VOLATILITY"])
BULL_REGIMES = frozenset(["MOMENTUM_RALLY", "QUIET_ACCUMULATION"])
NEUTRAL_REGIMES = frozenset(["WEAK_CONSOLIDATION"])

# =============================================================================
# Stage 3: EnhancedRewardCalculator (Sortino-based)
# =============================================================================

from collections import deque


def _compute_turnover_cost(
    position_change: float,
    trade_cost_bps: float,
    turnover_cost_mult: float = 1.0,
) -> float:
    """Stage 9.2: Continuous proportional turnover cost for reward shaping.

    Two layers:
      Layer 1 - Real cost mapping: maps env.step() friction bps -> reward signal.
      Layer 2 - Churn penalty: additional penalty proportional to |Δposition|.

    Returns negative reward penalty (larger change = more negative).
    """
    if abs(position_change) < 0.005:
        return 0.0

    # Layer 1: real cost -> reward signal (10 bps ≈ -0.01 reward)
    cost_signal = -trade_cost_bps * 0.001

    # Layer 2: churn penalty (|Δ|=0.1 -> -0.001, |Δ|=2.0 -> -0.02)
    churn_penalty = -abs(position_change) * 0.01

    return (cost_signal + churn_penalty) * turnover_cost_mult


class EnhancedRewardCalculator:
    """Sortino/Sharpe reward shaping with Optuna-searchable weights.

    Key differences vs classic:
      1. Asymmetric: losses penalized downside_weight× (Sortino core idea)
      2. Rolling Sortino bonus: after warmup, small risk-adjusted signal
      3. Escalating drawdown: quadratic penalty (not linear)
      4. Proportional turnover cost (v9: replaces threshold penalty)
      5. 4-component weighted combination (pnl/quality/risk/sortino)

    Sharpe mode (Stage 7):
      - Replaces sortino component with rolling Sharpe ratio
      - Weights: 0.4*PnL + 0.2*Sortino + 0.2*Sharpe + 0.1*Quality - 0.1*Risk
    """

    def __init__(
        self,
        window: int = 48,
        downside_weight: float = 2.0,
        dd_threshold: float = 0.05,
        dd_penalty_mult: float = 5.0,
        turnover_penalty: float = 0.02,
        pnl_weight: float = 0.5,
        sortino_weight: float = 0.3,
        quality_weight: float = 0.1,
        risk_weight: float = 0.1,
        mode: str = "sortino",
    ):
        self.window = window
        self.downside_weight = downside_weight
        self.dd_threshold = dd_threshold
        self.dd_penalty_mult = dd_penalty_mult
        self.turnover_penalty = turnover_penalty
        self.mode = mode

        if mode == "sharpe":
            # Sharpe mode: 0.4 PnL + 0.2 Sortino + 0.2 Sharpe + 0.1 Quality - 0.1 Risk
            self.pnl_weight = 0.4
            self.sortino_weight = 0.2
            self.sharpe_weight = 0.2
            self.quality_weight = 0.1
            self.risk_weight = 0.1
        else:
            self.pnl_weight = pnl_weight
            self.sortino_weight = sortino_weight
            self.sharpe_weight = 0.0
            self.quality_weight = quality_weight
            self.risk_weight = risk_weight

        self._returns: deque = deque(maxlen=window)
        self._sharpe_window = 20  # 20 steps = 80H ~ 3.3 days

    def update_params(self, **kwargs):
        """Update parameters from Optuna trial."""
        for k, v in kwargs.items():
            if k == 'sortino_window':
                self.window = v
                self._returns = deque(self._returns, maxlen=v)
            elif hasattr(self, k):
                setattr(self, k, v)

    def reset(self):
        self._returns.clear()

    def compute(
        self,
        pnl_pct: float,
        drawdown: float,
        position: float,
        position_change: float,
        regime: str,
        regime_weight: float,
        bear_mult: float,
        bull_mult: float,
        sol_vol_mult: float,
        trade_cost_bps: float = 0.0,
        turnover_cost_mult: float = 1.0,
        turnover_min_change: float = 0.01,
    ) -> float:
        self._returns.append(pnl_pct)

        # --- PNL component ---
        pnl_r = pnl_pct * 100 * sol_vol_mult
        if pnl_pct < 0:
            pnl_r *= self.downside_weight
        pnl_r *= regime_weight
        if regime in BEAR_REGIMES and position < 0:
            pnl_r *= bear_mult
        elif regime in BULL_REGIMES and position > 0:
            pnl_r *= bull_mult

        # --- Quality component (direction alignment) ---
        quality_r = 0.0
        position_bias = POSITION_BIAS.get(regime, 0.0)
        if position * position_bias > 0:
            quality_r += abs(position_bias) * 0.5
        if abs(position_change) < 0.1:
            if regime in BULL_REGIMES and position < 0:
                quality_r -= 0.03
            elif regime in BEAR_REGIMES and position > 0:
                quality_r -= 0.05

        # --- Risk component (drawdown + proportional turnover) ---
        risk_r = 0.0
        if drawdown > self.dd_threshold:
            risk_r -= (drawdown * self.dd_penalty_mult) ** 2
        # Stage 9.2: proportional turnover cost (replaces threshold penalty)
        if abs(position_change) >= turnover_min_change:
            risk_r += _compute_turnover_cost(
                position_change=abs(position_change),
                trade_cost_bps=trade_cost_bps,
                turnover_cost_mult=turnover_cost_mult,
            )

        # --- Sortino component (rolling risk-adjusted bonus) ---
        sortino_r = 0.0
        if len(self._returns) >= 20:
            returns = np.array(self._returns)
            mean_ret = np.mean(returns)
            neg = returns[returns < 0]
            if len(neg) > 2:
                ds = np.std(neg)
                if ds > 1e-8:
                    sortino = mean_ret / ds
                    sortino_r = float(np.clip(sortino * 0.05, -0.2, 0.2))

        # --- Sharpe component (Stage 7) ---
        sharpe_r = 0.0
        if self.sharpe_weight > 0 and len(self._returns) >= 10:
            recent = list(self._returns)[-self._sharpe_window:]
            _mean = np.mean(recent)
            _std = np.std(recent) + 1e-8
            sharpe_r = float(np.clip(_mean / _std, -0.3, 0.3))

        # Weighted combination
        reward = (self.pnl_weight * pnl_r +
                  self.quality_weight * quality_r +
                  self.risk_weight * risk_r +
                  self.sortino_weight * sortino_r +
                  self.sharpe_weight * sharpe_r)

        return reward

# REGIME_NAMES loaded dynamically from GMM config (cluster ordering changes on retrain)
_REGIME_NAMES_DEFAULT = [
    "VOLATILE_CHOP", "EXTREME_VOLATILITY", "MOMENTUM_RALLY",
    "PANIC_SELLOFF", "WEAK_CONSOLIDATION", "QUIET_ACCUMULATION",
]


def _load_regime_names(asset: str = None) -> list:
    """Load regime_names from per-asset or shared GMM config."""
    # Per-asset paths first, then shared fallback
    search_paths = []
    if asset:
        search_paths.append(_TRAINING_DIR / "training_data" / "gmm_models" / asset / "gmm_config.json")
        search_paths.append(PROJECT_ROOT / "models" / "regime_classifier" / asset / "gmm_config.json")
    search_paths.append(_TRAINING_DIR / "training_data" / "gmm_models" / "gmm_config.json")
    search_paths.append(PROJECT_ROOT / "models" / "regime_classifier" / "gmm_config.json")

    for path in search_paths:
        if path.exists():
            try:
                with open(path) as f:
                    config = json.load(f)
                names = config.get("regime_names")
                if names and len(names) >= 3:
                    logger.info(f"  Loaded regime names from {path}: k={len(names)}")
                    return names
            except Exception:
                pass
    logger.warning("  Could not load regime names from GMM config, using defaults")
    return list(_REGIME_NAMES_DEFAULT)


# Default shared regime names (overridden per-asset in main())
REGIME_NAMES = _load_regime_names()

POSITION_BIAS = {
    "MOMENTUM_RALLY":       +0.08,
    "WEAK_CONSOLIDATION":    0.00,
    "EXTREME_VOLATILITY":   -0.05,
    "VOLATILE_CHOP":        -0.05,
    "QUIET_ACCUMULATION":    0.00,
    "PANIC_SELLOFF":        -0.40,
}

# Fold weights (newest data = highest weight)
FOLD_WEIGHTS = [0.45, 0.35, 0.20]


# =============================================================================
# Stage 9: Real Friction Cost Model
# =============================================================================

# Kraken Pro fee structure
KRAKEN_PRO_FEES = {
    'within_free_tier': {
        'maker_fee_bps': 0.0,
        'taker_fee_bps': 0.0,
    },
    'beyond_free_tier': {
        'maker_fee_bps': 16.0,   # 0.16%
        'taker_fee_bps': 26.0,   # 0.26%
    },
}

# Per-asset baseline slippage (bps) - from historical data medians
ASSET_SLIPPAGE_BPS = {
    'BTC': 3.0,   # high liquidity
    'ETH': 5.0,   # medium liquidity
    'SOL': 10.0,  # lower liquidity, Asia sessions can reach 15-20 bps
}


# =============================================================================
# Trading Environment (direction-regime aware)
# =============================================================================

class TradingEnvFull(gym.Env):
    """
    v7: Direction-regime aware trading environment.
    v9: + real friction costs in NAV, proportional turnover cost in reward.
    State: features + [position_ratio, position_direction, pnl_ratio, drawdown]
    Action: continuous [-1, 1]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        initial_balance: float = 100000,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
        window_size: int = 10,
        bear_reward_multiplier: float = 1.2,
        bull_reward_multiplier: float = 1.1,
        sol_vol_multiplier: float = 1.0,
        reward_mode: str = "classic",
        # Reward hyperparams (Optuna-searchable)
        dd_threshold: float = 0.0866,       # Config 1 Optuna winner (was 0.10)
        dd_penalty_mult: float = 2.8243,     # Config 1 Optuna winner (was 2.0)
        reward_clip: float = 20.0,           # Config 1 Optuna winner (was 50)
        regime_weight_min: float = 0.5962,   # Config 1 Optuna winner (was 0.5)
        regime_weight_max: float = 1.5697,   # Config 1 Optuna winner (was 2.0)
        # Sortino/Sharpe-specific
        sortino_window: int = 48,
        turnover_penalty: float = 0.02,
        pnl_weight: float = 0.5,
        sortino_weight: float = 0.3,
        quality_weight: float = 0.1,
        risk_weight: float = 0.1,
        # Stage 7: State augmentation (training only)
        augment_enabled: bool = False,
        # Stage 9: Friction
        asset: str = "BTC",
        friction_enabled: bool = True,
        slippage_bps_override: Optional[float] = None,
        turnover_cost_mult: float = 1.0,
        turnover_min_change: float = 0.01,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.bear_reward_multiplier = bear_reward_multiplier
        self.bull_reward_multiplier = bull_reward_multiplier
        self.sol_vol_multiplier = sol_vol_multiplier
        self.reward_mode = reward_mode
        self._augment_enabled = augment_enabled

        # Stage 9: Friction parameters
        self._friction_enabled = friction_enabled
        self._asset = asset
        self._slippage_bps = slippage_bps_override if slippage_bps_override is not None else ASSET_SLIPPAGE_BPS.get(asset, 10.0)
        self._turnover_cost_mult = turnover_cost_mult
        self._turnover_min_change = turnover_min_change
        # Friction tracking (reset in reset())
        self.cumulative_trade_cost = 0.0
        self._last_trade_cost_bps = 0.0
        self._last_trade_cost_usd = 0.0
        self._trade_count = 0

        # Reward hyperparams (Optuna-searchable)
        self.dd_threshold = dd_threshold
        self.dd_penalty_mult = dd_penalty_mult
        self.reward_clip = reward_clip
        self.regime_weight_min = regime_weight_min
        self.regime_weight_max = regime_weight_max

        # P3: Rare regime reward weighting (inverse sqrt frequency)
        self.regime_weights = self._compute_regime_weights()

        # Stage 3/7: Enhanced reward calculator (sortino or sharpe mode)
        self._sortino_calc = EnhancedRewardCalculator(
            window=sortino_window,
            dd_threshold=dd_threshold,
            dd_penalty_mult=dd_penalty_mult * 2.5,  # enhanced uses steeper penalty
            turnover_penalty=turnover_penalty,
            pnl_weight=pnl_weight,
            sortino_weight=sortino_weight,
            quality_weight=quality_weight,
            risk_weight=risk_weight,
            mode=reward_mode,
        ) if reward_mode in ("sortino", "sharpe") else None

        n_features = len(feature_cols)
        # +4: position_ratio, position_direction, pnl_ratio, drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features + 4,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.total_pnl = 0.0
        # Stage 9: reset friction tracking
        self.cumulative_trade_cost = 0.0
        self._last_trade_cost_bps = 0.0
        self._last_trade_cost_usd = 0.0
        self._trade_count = 0
        if self._sortino_calc:
            self._sortino_calc.reset()
        return self._get_observation(), {}

    def _compute_trade_cost_bps(self, trade_notional_usd: float) -> float:
        """Stage 9.1: Compute total friction cost (bps) for a single trade.

        Training assumes within free tier (fee=0), only slippage + impact.
        """
        if not self._friction_enabled:
            return 0.0

        fee_bps = 0.0  # within free tier ($10K/mo)
        slippage_bps = self._slippage_bps

        # Large-trade market impact (simplified linear model)
        impact_bps = 0.0
        if trade_notional_usd > 2000:
            impact_bps = (trade_notional_usd - 2000) / 1000 * 1.0
            impact_bps = min(impact_bps, 10.0)

        return fee_bps + slippage_bps + impact_bps

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)

        position_ratio = self.position / self.max_position if self.max_position > 0 else 0
        position_direction = float(np.sign(self.position))
        pnl_ratio = self.total_pnl / self.initial_balance
        drawdown = (
            (self.peak_balance - self.balance) / self.peak_balance
            if self.peak_balance > 0 else 0
        )

        obs = np.concatenate([
            features,
            np.array([position_ratio, position_direction, pnl_ratio, drawdown],
                      dtype=np.float32)
        ]).astype(np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs, -10.0, 10.0)  # Bound outliers (gap_risk can reach 2369 post-scale)

        # Stage 7: State augmentation (training envs only, never eval)
        if self._augment_enabled:
            obs_clean = obs.copy()
            # Gaussian noise (2%) on features only
            noise = np.random.normal(0, 0.02, size=obs.shape).astype(np.float32)
            obs = obs + noise
            # Feature dropout (5%)
            dropout_mask = np.random.binomial(1, 0.95, size=obs.shape).astype(np.float32)
            obs = obs * dropout_mask
            # Preserve exact env state (last 4) and no-scale features (regime_proba + has_external_data)
            obs[-4:] = obs_clean[-4:]    # position_ratio, position_direction, pnl_ratio, drawdown
            obs[-12:-4] = obs_clean[-12:-4]  # regime_proba_0..7
            obs[-13] = obs_clean[-13]    # has_external_data (binary)

        return obs

    def _compute_regime_weights(self) -> Dict[str, float]:
        """P3: Inverse sqrt frequency weighting for rare regimes."""
        if "regime" not in self.df.columns:
            return {}
        regime_col = self.df["regime"]
        n = len(regime_col)
        n_regimes = regime_col.nunique()
        if n_regimes == 0:
            return {}
        weights = {}
        for regime_val, count in regime_col.value_counts().items():
            name = self._regime_to_name(regime_val)
            # Inverse sqrt frequency: rare regimes get higher weight
            w = (n / (count * n_regimes)) ** 0.5
            weights[name] = float(np.clip(w, self.regime_weight_min, self.regime_weight_max))
        return weights

    @staticmethod
    def _regime_to_name(regime_val) -> str:
        """Convert regime int/string to name. Handles variable k."""
        if isinstance(regime_val, (int, np.integer)):
            if 0 <= regime_val < len(REGIME_NAMES):
                return REGIME_NAMES[regime_val]
            return f"REGIME_{regime_val}"
        return str(regime_val)

    def _get_regime(self) -> str:
        if "regime" in self.df.columns:
            regime_val = self.df.iloc[self.current_step]["regime"]
            return self._regime_to_name(regime_val)
        return "WEAK_CONSOLIDATION"

    def step(self, action: np.ndarray):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        current_price = self.df.iloc[self.current_step]["close"]

        target_position = action_val * self.max_position
        position_change = target_position - self.position

        old_position = self.position
        self.position = target_position

        # === Stage 9.1: Real friction cost (deducted from NAV) ===
        self._last_trade_cost_bps = 0.0
        self._last_trade_cost_usd = 0.0
        abs_change = abs(position_change)
        if self._friction_enabled and abs_change > self._turnover_min_change:
            trade_notional = abs_change * self.initial_balance * self.max_position
            cost_bps = self._compute_trade_cost_bps(trade_notional)
            cost_usd = trade_notional * cost_bps / 10000
            self._last_trade_cost_bps = cost_bps
            self._last_trade_cost_usd = cost_usd
            self.cumulative_trade_cost += cost_usd
            self._trade_count += 1
            trade_cost = cost_usd
        elif abs_change > 0.001:
            # Legacy flat cost for sub-threshold changes (backwards compat)
            trade_cost = abs_change * current_price * self.transaction_cost
        else:
            trade_cost = 0.0

        if old_position == 0 and self.position != 0:
            self.entry_price = current_price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False

        if not done:
            new_price = self.df.iloc[self.current_step]["close"]

            if self.position != 0:
                price_change = (new_price - current_price) / current_price
                pnl = self.position * self.initial_balance * price_change
            else:
                pnl = 0

            self.balance += pnl - trade_cost
            self.total_pnl += pnl - trade_cost
            self.peak_balance = max(self.peak_balance, self.balance)

            # === Direction-Regime Aware Reward ===
            regime = self._get_regime()
            regime_weight = self.regime_weights.get(regime, 1.0)
            pnl_pct = pnl / self.initial_balance
            drawdown = (self.peak_balance - self.balance) / self.peak_balance

            if self._sortino_calc:
                # Stage 3: Sortino-based reward
                reward = self._sortino_calc.compute(
                    pnl_pct=pnl_pct,
                    drawdown=drawdown,
                    position=self.position,
                    position_change=position_change,
                    regime=regime,
                    regime_weight=regime_weight,
                    bear_mult=self.bear_reward_multiplier,
                    bull_mult=self.bull_reward_multiplier,
                    sol_vol_mult=self.sol_vol_multiplier,
                    trade_cost_bps=self._last_trade_cost_bps,
                    turnover_cost_mult=self._turnover_cost_mult,
                    turnover_min_change=self._turnover_min_change,
                )
            else:
                # Classic reward
                position_bias = POSITION_BIAS.get(regime, 0.0)

                reward = pnl_pct * 100
                reward *= self.sol_vol_multiplier

                # P3: Rare regime weighting (inverse sqrt frequency)
                reward *= regime_weight

                if regime in BEAR_REGIMES and self.position < 0:
                    reward *= self.bear_reward_multiplier
                elif regime in BULL_REGIMES and self.position > 0:
                    reward *= self.bull_reward_multiplier

                if self.position * position_bias > 0:
                    reward += abs(position_bias) * 0.5

                position_held = abs(position_change) < 0.1
                if position_held:
                    if regime in BULL_REGIMES and self.position < 0:
                        reward -= 0.03
                    elif regime in BEAR_REGIMES and self.position > 0:
                        reward -= 0.05

                if drawdown > self.dd_threshold:
                    reward -= drawdown * self.dd_penalty_mult

                # Stage 9.2: Proportional turnover cost (classic mode)
                if self._friction_enabled and abs(position_change) >= self._turnover_min_change:
                    turnover_cost = _compute_turnover_cost(
                        position_change=abs(position_change),
                        trade_cost_bps=self._last_trade_cost_bps,
                        turnover_cost_mult=self._turnover_cost_mult,
                    )
                    reward += turnover_cost
        else:
            reward = 0

        # Clip reward to prevent gradient explosion on extreme moves
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        info = {
            "balance": self.balance,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "regime": self._get_regime(),
            "trade_cost_bps": self._last_trade_cost_bps,
            "trade_cost_usd": self._last_trade_cost_usd,
            "cumulative_trade_cost": self.cumulative_trade_cost,
            "trade_count": self._trade_count,
        }

        return self._get_observation(), reward, done, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step {self.current_step}: Balance={self.balance:.2f}, "
                  f"Position={self.position:.2f}, PnL={self.total_pnl:.2f}")

    def set_reward_params(self, **kwargs):
        """Allow Optuna to override reward hyperparameters at runtime."""
        if 'dd_threshold' in kwargs:
            self.dd_threshold = kwargs['dd_threshold']
        if 'dd_penalty_mult' in kwargs:
            self.dd_penalty_mult = kwargs['dd_penalty_mult']
        if 'reward_clip' in kwargs:
            self.reward_clip = kwargs['reward_clip']
        if 'regime_weight_range' in kwargs:
            self.regime_weight_min, self.regime_weight_max = kwargs['regime_weight_range']
            self.regime_weights = self._compute_regime_weights()
        # Forward sortino-specific params
        if self._sortino_calc:
            sortino_keys = {'sortino_window', 'turnover_penalty', 'dd_threshold',
                            'dd_penalty_mult', 'pnl_weight', 'sortino_weight',
                            'quality_weight', 'risk_weight'}
            sortino_kwargs = {k: v for k, v in kwargs.items() if k in sortino_keys}
            if sortino_kwargs:
                self._sortino_calc.update_params(**sortino_kwargs)


# =============================================================================
# Gradient Monitor Callback
# =============================================================================

class GradientMonitorCallback(BaseCallback):
    """Monitor training health, auto-stop on gradient explosion."""

    def __init__(self, max_actor_loss: float = 1e6,
                 max_consecutive_explosions: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.max_actor_loss = max_actor_loss
        self.max_consecutive_explosions = max_consecutive_explosions
        self.explosion_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % 50_000 == 0 and self.num_timesteps > 0:
            try:
                import psutil
                mem_gb = psutil.Process().memory_info().rss / 1024**3
                logger.info(f"[MONITOR] Step {self.num_timesteps:,} | Memory: {mem_gb:.1f} GB")
                if mem_gb > 32:
                    logger.warning(f"[MONITOR] Memory {mem_gb:.1f} GB > 32GB!")
            except ImportError:
                pass

        if self.num_timesteps % 1000 == 0:
            actor_loss = self.logger.name_to_value.get("train/actor_loss", 0)
            ent_coef = self.logger.name_to_value.get("train/ent_coef", None)

            if ent_coef is not None and abs(ent_coef - 0.1) > 0.01:
                logger.error(f"[MONITOR] ent_coef CHANGED: {ent_coef:.4f} (expected 0.1)")

            if actor_loss and abs(actor_loss) > self.max_actor_loss:
                self.explosion_count += 1
                logger.warning(
                    f"[MONITOR] actor_loss={actor_loss:.2f} > {self.max_actor_loss:.0f} "
                    f"({self.explosion_count}/{self.max_consecutive_explosions})"
                )
                if self.explosion_count >= self.max_consecutive_explosions:
                    logger.error("[MONITOR] STOPPING: Gradient explosions!")
                    return False
            else:
                self.explosion_count = 0

        return True


# =============================================================================
# FPS Logger Callback
# =============================================================================

class FPSLoggerCallback(BaseCallback):
    """Log FPS periodically to detect /mnt/c/ IO bottlenecks."""

    def __init__(self, log_freq: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_time = None
        self._last_steps = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > 0:
            now = time.time()
            if self._last_time is not None:
                elapsed = now - self._last_time
                steps = self.num_timesteps - self._last_steps
                fps = steps / elapsed if elapsed > 0 else 0
                logger.info(f"[FPS] Step {self.num_timesteps:,} | FPS: {fps:.0f}")
                if fps < 50:
                    logger.warning(
                        "[FPS] Below 50 - possible /mnt/c/ IO bottleneck. "
                        "Consider copying data to WSL2 local filesystem."
                    )
            self._last_time = now
            self._last_steps = self.num_timesteps
        return True


class LearningStartsResetCallback(BaseCallback):
    """Reset EvalCallback best_mean_reward after learning_starts completes.

    During learning_starts, no gradient updates occur - the policy is random.
    Any "best" eval reward from this phase is meaningless and can cause
    premature early stopping once real training begins (ETH fold_3 bug).

    This callback monitors num_timesteps and, once learning_starts is passed,
    resets best_mean_reward to -inf and the patience counter to 0.
    """

    def __init__(self, eval_callback: EvalCallback, verbose: int = 1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self._reset_done = False

    def _on_step(self) -> bool:
        if not self._reset_done and self.num_timesteps >= self.model.learning_starts:
            self._reset_done = True

            # Reset best reward on eval callback
            old_best = self.eval_callback.best_mean_reward
            self.eval_callback.best_mean_reward = -np.inf

            # Reset patience counter on stop-training callback
            # EvalCallback stores callback_after_eval as self.callback (via EventCallback)
            stop_cb = self.eval_callback.callback
            if stop_cb is not None and hasattr(stop_cb, 'no_improvement_evals'):
                stop_cb.no_improvement_evals = 0

            if self.verbose > 0:
                logger.info(
                    f"[TRAIN] learning_starts={self.model.learning_starts} passed "
                    f"at step {self.num_timesteps}. "
                    f"Reset best_mean_reward: {old_best:.2f} -> -inf. "
                    f"Patience counter reset to 0."
                )

        return True


# =============================================================================
# Trainer
# =============================================================================

class FullDRLTrainer:
    """3-fold TQC trainer with direction-regime aware rewards."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        asset: str = "BTC",
        output_dir: str = "models/retrained",
        n_folds: int = 3,
        total_timesteps: int = 2_500_000,
        eval_freq: int = 5_000,
        learning_rate: Optional[float] = None,
        bear_reward_multiplier: float = 1.2,
        bull_reward_multiplier: float = 1.1,
        sol_vol_multiplier: float = 1.0,
        resume: bool = False,
        device: str = "auto",
        progress_bar: bool = True,
        purge_window: int = 0,
        no_scale_features: Optional[set] = None,
        reward_mode: str = "classic",
        extractor: Optional[str] = None,
        augment: bool = False,
        # Stage 8: Optuna-derived or manual overrides
        buffer_size: Optional[int] = None,
        learning_starts: Optional[int] = None,
        n_quantiles: Optional[int] = None,
        dd_threshold: float = 0.0866,       # Config 1 Optuna winner (was 0.10)
        dd_penalty_mult: float = 2.8243,     # Config 1 Optuna winner (was 2.0)
        reward_clip: float = 20.0,           # Config 1 Optuna winner (was 50)
        regime_weight_min: float = 0.5962,   # Config 1 Optuna winner (was 0.5)
        regime_weight_max: float = 1.5697,   # Config 1 Optuna winner (was 2.0)
        tag: Optional[str] = None,
        # Stage 9: Friction
        friction_enabled: bool = True,
        slippage_bps: Optional[float] = None,
        turnover_cost_mult: float = 1.0,
        turnover_min_change: float = 0.01,
        # VecEnv
        vec_env_type: str = "dummy",
        n_envs: int = 1,
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.no_scale_features = no_scale_features or set()
        self.reward_mode = reward_mode
        self.extractor = extractor
        self.augment = augment
        self.asset = asset

        # Output directory with optional tag suffix
        base_dir = Path(output_dir) / asset
        if tag:
            base_dir = base_dir / tag
        self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_folds = n_folds
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.resume = resume
        self.progress_bar = progress_bar
        self.purge_window = purge_window

        self.bear_reward_multiplier = bear_reward_multiplier
        self.bull_reward_multiplier = bull_reward_multiplier
        self.sol_vol_multiplier = sol_vol_multiplier

        # TQC model overrides (None = use ULTIMATE_PRESET defaults)
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.n_quantiles = n_quantiles

        # Reward env params
        self.dd_threshold = dd_threshold
        self.dd_penalty_mult = dd_penalty_mult
        self.reward_clip = reward_clip
        self.regime_weight_min = regime_weight_min
        self.regime_weight_max = regime_weight_max

        # Stage 9: Friction
        self.friction_enabled = friction_enabled
        self.slippage_bps = slippage_bps
        self.turnover_cost_mult = turnover_cost_mult
        self.turnover_min_change = turnover_min_change

        # VecEnv
        self.vec_env_type = vec_env_type
        self.n_envs = n_envs

        overrides = ASSET_OVERRIDES.get(asset, {})
        self.learning_rate = learning_rate or overrides.get(
            "learning_rate", ULTIMATE_PRESET["learning_rate"]
        )

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.results = {}

    def _create_env(self, df: pd.DataFrame,
                    augment_enabled: bool = False) -> gym.Env:
        env = TradingEnvFull(
            df=df,
            feature_cols=self.feature_cols,
            bear_reward_multiplier=self.bear_reward_multiplier,
            bull_reward_multiplier=self.bull_reward_multiplier,
            sol_vol_multiplier=self.sol_vol_multiplier,
            reward_mode=self.reward_mode,
            augment_enabled=augment_enabled,
            dd_threshold=self.dd_threshold,
            dd_penalty_mult=self.dd_penalty_mult,
            reward_clip=self.reward_clip,
            regime_weight_min=self.regime_weight_min,
            regime_weight_max=self.regime_weight_max,
            # Stage 9: Friction
            asset=self.asset,
            friction_enabled=self.friction_enabled,
            slippage_bps_override=self.slippage_bps,
            turnover_cost_mult=self.turnover_cost_mult,
            turnover_min_change=self.turnover_min_change,
        )
        return Monitor(env)

    def _scale_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, save_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """P2: Fit RobustScaler on train features, transform both. Save scaler.
        Features in self.no_scale_features (regime_proba_*, has_external_data) are
        passed through unchanged - they are already [0,1] bounded.
        """
        scale_cols = [c for c in self.feature_cols if c not in self.no_scale_features]
        pass_cols = [c for c in self.feature_cols if c in self.no_scale_features]

        if pass_cols:
            logger.info(f"  RobustScaler: scaling {len(scale_cols)} features, "
                        f"passing through {len(pass_cols)} (regime_proba/binary)")

        scaler = RobustScaler()
        scaler.fit(train_df[scale_cols])

        # Handle features with zero IQR (sparse binary features) - avoid division by zero
        scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)

        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        train_scaled[scale_cols] = scaler.transform(train_df[scale_cols])
        val_scaled[scale_cols] = scaler.transform(val_df[scale_cols])

        # Replace any remaining inf/NaN from edge cases, then clip outliers
        for df in [train_scaled, val_scaled]:
            df[scale_cols] = df[scale_cols].replace(
                [np.inf, -np.inf], 0
            ).fillna(0)
            # Post-scale clip: gap_risk can reach 2369 scaled - bound all features
            df[scale_cols] = df[scale_cols].clip(-10.0, 10.0)

        # Save scaler for runtime inference (records which cols are scaled)
        scaler_meta = {
            "scaler": scaler,
            "scale_cols": scale_cols,
            "pass_cols": pass_cols,
        }
        scaler_path = save_dir / "feature_scaler.pkl"
        joblib.dump(scaler_meta, scaler_path)
        logger.info(f"  RobustScaler saved: {scaler_path}")

        return train_scaled, val_scaled

    def _get_fold_splits(self, purge_window: int = 0) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """3-fold time-series splits with optional purging.

        Args:
            purge_window: Additional bars to remove from train end (P4 purging).
                         Total gap between train and val = gap + purge_window.
        """
        n = len(self.data)
        val_size = int(n * 0.15)
        gap = 42       # 42 4H bars = 1 week
        window_size = 10

        folds = []
        for fold_idx in range(self.n_folds):
            val_end = n - fold_idx * val_size
            val_start = val_end - val_size
            train_end = val_start - gap - purge_window

            if train_end < 5000:
                logger.warning(
                    f"Fold {fold_idx + 1}: train_end={train_end} < 5000, skipping"
                )
                continue

            eval_steps = val_size - window_size
            if eval_steps < 100:
                logger.warning(
                    f"Fold {fold_idx + 1}: eval_steps={eval_steps} < 100, skipping"
                )
                continue

            train_df = self.data.iloc[:train_end].copy()
            val_df = self.data.iloc[val_start:val_end].copy()

            total_gap = gap + purge_window
            folds.append((train_df, val_df))
            logger.info(
                f"  Fold {fold_idx + 1}: train=[0:{train_end}] ({len(train_df)} rows), "
                f"val=[{val_start}:{val_end}] ({len(val_df)} rows), "
                f"eval_steps={eval_steps}, gap={gap}+purge={purge_window}={total_gap}"
            )

        return folds

    def run(self) -> Dict:
        logger.info("=" * 70)
        logger.info(f"HMATS v7 - Ultimate Rebuild DRL Training: {self.asset}")
        logger.info("=" * 70)
        logger.info(f"  Data rows:  {len(self.data)}")
        logger.info(f"  Folds:      {self.n_folds}")
        logger.info(f"  Timesteps:  {self.total_timesteps:,}")
        logger.info(f"  LR:         {self.learning_rate}")
        logger.info(f"  ent_coef:   {ULTIMATE_PRESET['ent_coef']} (FIXED)")
        logger.info(f"  net_arch:   {ULTIMATE_PRESET['policy_kwargs']['net_arch']}")
        _eff_nq = self.n_quantiles or ULTIMATE_PRESET['policy_kwargs']['n_quantiles']
        _eff_buf = self.buffer_size or ULTIMATE_PRESET['buffer_size']
        _eff_ls = self.learning_starts or ULTIMATE_PRESET['learning_starts']
        logger.info(f"  quantiles:  {_eff_nq}")
        logger.info(f"  buffer:     {_eff_buf:,}")
        logger.info(f"  learn_start:{_eff_ls:,}")
        logger.info(f"  Features:   {len(self.feature_cols)}")
        logger.info(f"  Obs dim:    {len(self.feature_cols) + 4} "
                     f"(+4: pos_ratio, pos_dir, pnl, dd)")
        logger.info(f"  Device:     {self.device}")
        logger.info(f"  Resume:     {self.resume}")
        logger.info(f"  Bear mult:  {self.bear_reward_multiplier}")
        logger.info(f"  Bull mult:  {self.bull_reward_multiplier}")
        logger.info(f"  SOL vol:    {self.sol_vol_multiplier}")
        logger.info(f"  Reward:     {self.reward_mode}")
        logger.info(f"  VecEnv:     {self.vec_env_type} (n_envs={self.n_envs})")
        logger.info(f"  Extractor:  {self.extractor or 'default (MLP flat)'}")
        logger.info(f"  Augment:    {self.augment} (train only)")
        logger.info(f"  dd_thr:     {self.dd_threshold}")
        logger.info(f"  dd_mult:    {self.dd_penalty_mult}")
        logger.info(f"  rwd_clip:   {self.reward_clip}")
        _slip = self.slippage_bps if self.slippage_bps is not None else ASSET_SLIPPAGE_BPS.get(self.asset, 10.0)
        logger.info(f"  Friction:   {'ON' if self.friction_enabled else 'OFF'} "
                     f"(slip={_slip} bps, to_mult={self.turnover_cost_mult}, "
                     f"min_chg={self.turnover_min_change})")
        logger.info(f"  rw_range:   [{self.regime_weight_min}, {self.regime_weight_max}]")
        logger.info(f"  Checkpoint: every 500K steps")
        logger.info("=" * 70)

        if self.device == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            logger.info(f"  GPU Memory: {props.total_memory / 1024**3:.1f} GB")

        folds = self._get_fold_splits(purge_window=self.purge_window)
        if not folds:
            logger.error("No valid folds! Check data size.")
            return {}

        # Resolve extractor config once
        _use_extractor = self.extractor is not None
        if _use_extractor:
            from models.feature_extractors import (
                get_extractor_class, get_extractor_kwargs,
                SEQUENCE_ARCHS, DEFAULT_N_STACK,
            )
            _n_stack = DEFAULT_N_STACK  # 8

        for fold_idx, (train_df, val_df) in enumerate(folds):
            fold_name = f"fold_{fold_idx + 1}"
            # Stage 7: include reward_mode + augment in run_label to avoid collisions
            _label_parts = [self.extractor.upper() if _use_extractor else "ULTIMATE"]
            if self.reward_mode != "classic":
                _label_parts.append(self.reward_mode.upper())
            if self.augment:
                _label_parts.append("AUG")
            run_label = "_".join(_label_parts)
            log_dir = self.output_dir / fold_name / "logs" / run_label

            logger.info(f"\n{'=' * 60}")
            logger.info(f"  {self.asset} - {fold_name} [{run_label}]")
            logger.info(f"  Train: {len(train_df)} rows | Val: {len(val_df)} rows")
            logger.info("=" * 60)

            # Skip completed
            final_model_path = log_dir / "final_model.zip"
            if final_model_path.exists() and not self.resume:
                logger.info(f"  Already done, skipping ({final_model_path})")
                # Load results if they exist
                self._load_fold_result(fold_name, log_dir)
                continue

            log_dir.mkdir(parents=True, exist_ok=True)

            # Pre-declare for finally cleanup
            model = None
            best_model = None
            train_env = None
            eval_env = None

            try:  # === Crash-safe fold wrapper: any exception -> log + continue ===

                # P2: RobustScaler - fit on train, transform both
                train_df, val_df = self._scale_features(train_df, val_df, log_dir)

                # Stage 9.3: Fit OOD detector on scaled training features (Rule #31: train only)
                try:
                    import importlib.util
                    _ood_spec = importlib.util.spec_from_file_location(
                        "ood_detector", str(PROJECT_ROOT / "drl" / "ood_detector.py"))
                    _ood_mod = importlib.util.module_from_spec(_ood_spec)
                    _ood_spec.loader.exec_module(_ood_mod)
                    _ood = _ood_mod.MahalanobisOODDetector()
                    _ood.fit(train_df[self.feature_cols].values)
                    _ood_path = str(log_dir / "ood_detector.npz")
                    _ood.save(_ood_path)
                except Exception as e:
                    logger.warning(f"  OOD detector fit failed (non-fatal): {e}")

                # Create environments
                if self.vec_env_type == "subproc" and self.n_envs > 1:
                    from stable_baselines3.common.vec_env import SubprocVecEnv
                    train_env = SubprocVecEnv([
                        lambda df=train_df: self._create_env(df, augment_enabled=self.augment)
                        for _ in range(self.n_envs)
                    ])
                    logger.info(f"  SubprocVecEnv: {self.n_envs} workers")
                else:
                    train_env = DummyVecEnv([lambda df=train_df: self._create_env(df, augment_enabled=self.augment)])
                # Eval always DummyVecEnv (single env, deterministic)
                eval_env = DummyVecEnv([lambda df=val_df: self._create_env(df, augment_enabled=False)])

                obs_dim = len(self.feature_cols) + 4

                # Stage 6: wrap with VecFrameStack for sequence extractors
                if _use_extractor and self.extractor in SEQUENCE_ARCHS:
                    train_env = VecFrameStack(train_env, n_stack=_n_stack)
                    eval_env = VecFrameStack(eval_env, n_stack=_n_stack)
                    logger.info(f"  VecFrameStack: n_stack={_n_stack}, "
                                f"stacked obs={obs_dim * _n_stack}")

                # Smoke test: 10 steps to catch NaN/shape issues before training
                if fold_idx == 0:
                    try:
                        _obs = train_env.reset()
                        _n_envs = self.n_envs
                        if _use_extractor and self.extractor in SEQUENCE_ARCHS:
                            expected_shape = (_n_envs, obs_dim * _n_stack)
                        else:
                            expected_shape = (_n_envs, obs_dim)
                        assert _obs.shape == expected_shape, \
                            f"Obs shape mismatch: {_obs.shape} vs {expected_shape}"
                        for _step in range(10):
                            _act = np.random.uniform(-1, 1, size=(_n_envs, 1)).astype(np.float32)
                            _obs, _rew, _done, _info = train_env.step(_act)
                            assert not np.isnan(_obs).any(), f"NaN in obs at smoke step {_step}"
                            assert not np.isnan(_rew).any(), f"NaN in reward at smoke step {_step}"
                        train_env.reset()  # Reset after smoke test
                        logger.info(f"  Smoke test: 10 steps OK, obs={_obs.shape}, no NaN")
                    except Exception as e:
                        logger.error(f"  Smoke test FAILED: {e} - training will proceed but may fail")
                        train_env.reset()

                # Check for checkpoint resume
                checkpoint_dir = log_dir / "checkpoints"
                resume_path = None
                if self.resume and checkpoint_dir.exists():
                    checkpoints = sorted(checkpoint_dir.glob(f"{run_label}_*_steps.zip"))
                    if checkpoints:
                        resume_path = checkpoints[-1]
                        logger.info(f"  Resuming from: {resume_path.name}")

                if resume_path:
                    model = TQC.load(
                        str(resume_path),
                        env=train_env,
                        device=self.device,
                        tensorboard_log=str(log_dir),
                    )
                else:
                    preset = dict(ULTIMATE_PRESET)
                    preset["learning_rate"] = self.learning_rate
                    # Stage 8: override TQC params from CLI
                    if self.buffer_size is not None:
                        preset["buffer_size"] = self.buffer_size
                    if self.learning_starts is not None:
                        preset["learning_starts"] = self.learning_starts
                    policy_kwargs = preset.pop("policy_kwargs")
                    policy = preset.pop("policy")

                    # Stage 8: override n_quantiles from CLI
                    _n_quantiles = self.n_quantiles or policy_kwargs.get("n_quantiles", 32)

                    # Stage 6: override policy_kwargs with custom extractor
                    if _use_extractor:
                        extractor_cls = get_extractor_class(self.extractor)
                        extractor_kwargs = get_extractor_kwargs(
                            self.extractor, single_obs_dim=obs_dim
                        )
                        policy_kwargs = {
                            "features_extractor_class": extractor_cls,
                            "features_extractor_kwargs": extractor_kwargs,
                            "net_arch": [256, 256],
                            "n_critics": 2,
                            "n_quantiles": _n_quantiles,
                        }
                    else:
                        policy_kwargs["n_quantiles"] = _n_quantiles

                    model = TQC(
                        policy,
                        env=train_env,
                        verbose=1,
                        tensorboard_log=str(log_dir),
                        device=self.device,
                        policy_kwargs=policy_kwargs,
                        **preset,
                    )

                    if _use_extractor:
                        total_params = sum(
                            p.numel() for p in model.policy.parameters()
                        )
                        logger.info(f"  Parameters: {total_params:,} total")

                # Early stopping
                stop_train_callback = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=20,
                    min_evals=50,
                    verbose=1,
                )

                # Callbacks
                _eval_cb = EvalCallback(
                    eval_env,
                    best_model_save_path=str(log_dir / "best_model"),
                    log_path=str(log_dir / "eval"),
                    eval_freq=self.eval_freq,
                    deterministic=True,
                    render=False,
                    callback_after_eval=stop_train_callback,
                )
                callbacks = [
                    GradientMonitorCallback(max_actor_loss=1e6, max_consecutive_explosions=3),
                    FPSLoggerCallback(log_freq=100_000),
                    _eval_cb,
                    LearningStartsResetCallback(eval_callback=_eval_cb),
                    CheckpointCallback(
                        save_freq=500_000,
                        save_path=str(checkpoint_dir),
                        name_prefix=run_label,
                    ),
                ]

                # Train
                start_time = time.time()
                _train_ok = False
                try:
                    model.learn(
                        total_timesteps=self.total_timesteps,
                        callback=callbacks,
                        progress_bar=self.progress_bar,
                        reset_num_timesteps=not bool(resume_path),
                    )
                    _train_ok = True
                except Exception as e:
                    logger.error(f"  Training failed: {e}")
                    import traceback
                    traceback.print_exc()

                train_time = time.time() - start_time

                if _train_ok:
                    # Save final model (may be overfitted - best_model is preferred)
                    model.save(str(log_dir / "final_model"))
                    logger.info(f"  Final model saved: {log_dir / 'final_model.zip'}")

                    # Stage 9: Save friction stats from training env
                    if self.friction_enabled:
                        try:
                            inner_env = train_env.envs[0].env
                            friction_stats = {
                                "cumulative_trade_cost": inner_env.cumulative_trade_cost,
                                "trade_count": inner_env._trade_count,
                                "avg_cost_per_trade_bps": inner_env._last_trade_cost_bps,
                                "slippage_bps": inner_env._slippage_bps,
                                "friction_enabled": True,
                                "turnover_cost_mult": inner_env._turnover_cost_mult,
                            }
                            friction_path = log_dir / "training_friction.json"
                            with open(friction_path, "w") as f:
                                json.dump(friction_stats, f, indent=2)
                            logger.info(f"  Friction stats: cost=${friction_stats['cumulative_trade_cost']:.2f}, "
                                        f"trades={friction_stats['trade_count']}")
                        except Exception as e:
                            logger.debug(f"  Friction stats save failed: {e}")

                    # Evaluate with best_model (saved by EvalCallback at peak eval reward)
                    best_path = log_dir / "best_model" / "best_model.zip"
                    if best_path.exists():
                        logger.info(f"  Evaluating with best_model (not overfitted final_model)")
                        best_model = TQC.load(str(best_path), env=eval_env, device=self.device)
                        mean_reward, std_reward = self._evaluate(best_model, eval_env)
                    else:
                        logger.warning(f"  No best_model found, evaluating with final_model")
                        mean_reward, std_reward = self._evaluate(model, eval_env)

                    self.results[fold_name] = {
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                        "train_time_minutes": train_time / 60,
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                    }

                    logger.info(
                        f"  Done: reward={mean_reward:.2f} +/- {std_reward:.2f} "
                        f"({train_time / 60:.1f} min)"
                    )
                else:
                    logger.warning(f"  {fold_name} training failed - skipping save/eval")

            except Exception as e:
                logger.error(f"  FOLD {fold_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Memory cleanup - ALWAYS runs, even if any part of fold failed
                for _obj_name in ('train_env', 'eval_env'):
                    try:
                        _obj = locals().get(_obj_name)
                        if _obj is not None:
                            _obj.close()
                    except Exception:
                        pass
                for _obj_name in ('model', 'best_model'):
                    try:
                        if locals().get(_obj_name) is not None:
                            del locals()[_obj_name]
                    except Exception:
                        pass
                model = None
                best_model = None
                train_env = None
                eval_env = None
                import gc as _gc
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                import psutil as _ps
                rss_gb = _ps.Process().memory_info().rss / 1e9
                logger.info(f"[MEM] Post-fold RSS: {rss_gb:.1f} GB")
                if rss_gb > 7.0:
                    logger.error(f"[MEM] RSS {rss_gb:.1f} GB > 7.0 GB - possible leak, continuing anyway")
                elif rss_gb > 5.0:
                    logger.warning(f"[MEM] Post-fold RSS {rss_gb:.1f} GB elevated but within safe range")

        # Summary
        summary = self._summarize()
        self._save_results(summary)
        return summary

    def _load_fold_result(self, fold_name: str, log_dir: Path):
        """Try to load existing eval results for a completed fold."""
        eval_path = log_dir / "eval" / "evaluations.npz"
        if eval_path.exists():
            try:
                data = np.load(eval_path)
                results = data.get("results", [])
                if len(results) > 0:
                    # Use best eval reward (not last - early stopping means last != best)
                    means = [float(np.mean(r)) for r in results]
                    best_idx = int(np.argmax(means))
                    best_rewards = results[best_idx]
                    self.results[fold_name] = {
                        "mean_reward": float(np.mean(best_rewards)),
                        "std_reward": float(np.std(best_rewards)),
                        "train_time_minutes": 0,
                        "train_rows": 0,
                        "val_rows": 0,
                    }
            except Exception:
                pass

    def _evaluate(self, model, env, n_episodes: int = 10) -> Tuple[float, float]:
        rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total += reward[0]
            rewards.append(total)
        return float(np.mean(rewards)), float(np.std(rewards))

    def _summarize(self) -> Dict:
        if not self.results:
            return {"asset": self.asset, "folds": {}, "best_fold": None}

        n = len(self.results)
        weights = {}
        w_values = FOLD_WEIGHTS[:n]
        w_sum = sum(w_values)
        for i, fold_name in enumerate(sorted(self.results.keys())):
            weights[fold_name] = w_values[i] / w_sum

        weighted_score = sum(
            weights.get(k, 1.0 / n) * v["mean_reward"]
            for k, v in self.results.items()
        )

        best_fold = max(self.results, key=lambda k: self.results[k]["mean_reward"])

        return {
            "asset": self.asset,
            "folds": self.results,
            "weights": weights,
            "weighted_score": weighted_score,
            "best_fold": best_fold,
            "best_reward": self.results[best_fold]["mean_reward"],
        }

    def _save_results(self, summary: Dict):
        path = self.output_dir / "results.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("\n" + "=" * 70)
        logger.info(f"Training Complete - {self.asset}")
        logger.info("=" * 70)
        for fold, metrics in self.results.items():
            w = summary.get("weights", {}).get(fold, 0)
            logger.info(
                f"  {fold}: reward={metrics['mean_reward']:.2f} "
                f"+/- {metrics['std_reward']:.2f} (weight={w:.2f})"
            )
        logger.info(f"\n  Weighted score: {summary.get('weighted_score', 0):.2f}")
        logger.info(f"  Best fold: {summary.get('best_fold')} "
                     f"({summary.get('best_reward', 0):.2f})")
        logger.info(f"  Results: {path}")


# =============================================================================
# Optuna Hyperparameter Tuning
# =============================================================================

class OptunaPruneCallback(BaseCallback):
    """Report eval rewards to Optuna for pruning. Used as callback_after_eval.

    Reports at training step count (not eval_count) so that
    MedianPruner(n_warmup_steps=50_000, interval_steps=10_000) works directly.
    """

    def __init__(self, trial, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self._eval_count = 0
        self.is_pruned = False
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        import optuna
        self._eval_count += 1
        reward = self.parent.last_mean_reward if self.parent else 0
        self.best_mean_reward = max(self.best_mean_reward, reward)
        # Report at training step count for MedianPruner compatibility
        step = self.parent.num_timesteps if self.parent else self._eval_count
        self.trial.report(reward, step)
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True


def run_optuna_tuning(
    asset: str,
    data: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = "models/retrained",
    n_trials: int = 50,
    trial_timesteps: int = 200_000,
    eval_freq: int = 10_000,
    device: str = "auto",
    bear_mult: float = 1.2,
    bull_mult: float = 1.1,
    sol_vol_mult: float = 1.0,
    no_scale_features: Optional[set] = None,
    reward_mode: str = "classic",
    extractor: str = "lstm_film_a",
    tier3: bool = False,
):
    """
    Stage 8: Optuna hyperparameter search for TQC + FiLM PosA extractor.

    Tier 1 - Core hyperparams (always searched):
      lr, buffer_size, learning_starts, n_quantiles

    Tier 2 - Classic reward shaping (always searched):
      dd_threshold, dd_penalty_mult, reward_clip, regime_weight_range

    Tier 3 - Architecture (optional, --optuna-tier3):
      net_arch, top_quantiles_to_drop, n_critics

    Fixed (iron rules - NEVER searched):
      ent_coef=0.1, batch=256, grad_steps=4, gamma=0.99, tau=0.005
      extractor=lstm_film_a (FiLM PosA), VecFrameStack(n_stack=8)
      reward_mode=classic (forced), augment=False (forced)

    Objective: mean final NAV% = (balance - 100K) / 100K * 100.
    Scale-invariant across different dd_penalty_mult values.

    Storage: sqlite for crash recovery (load_if_exists=True).
    """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 8: Force classic reward, resolve extractor
    reward_mode = "classic"
    from models.feature_extractors import (
        get_extractor_class, get_extractor_kwargs,
        SEQUENCE_ARCHS, DEFAULT_N_STACK,
    )
    _use_extractor = extractor is not None
    _n_stack = DEFAULT_N_STACK  # 8
    obs_dim = len(feature_cols) + 4  # 126

    # Use fold_1 split (latest data, same as production evaluation)
    n = len(data)
    val_size = int(n * 0.15)
    gap = 42
    window_size = 10
    val_end = n
    val_start = val_end - val_size
    train_end = val_start - gap

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[val_start:val_end].copy()
    eval_steps = val_size - window_size

    # P2: RobustScaler for Optuna trials (skip no_scale features)
    _no_scale = no_scale_features or set()
    scale_cols = [c for c in feature_cols if c not in _no_scale]
    scaler = RobustScaler()
    scaler.fit(train_df[scale_cols])
    scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    train_df[scale_cols] = scaler.transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    for df_ in [train_df, val_df]:
        df_[scale_cols] = df_[scale_cols].replace([np.inf, -np.inf], 0).fillna(0)
        df_[scale_cols] = df_[scale_cols].clip(-10.0, 10.0)

    results_dir = Path(output_dir) / asset / "optuna"
    results_dir.mkdir(parents=True, exist_ok=True)

    # SQLite storage for crash recovery
    tier_tag = "_t3" if tier3 else ""
    db_path = Path(output_dir) / asset / f"optuna_hmats_v8{tier_tag}.db"
    storage = f"sqlite:///{db_path}"
    study_name = f"hmats_v8_{asset.lower()}{tier_tag}"

    logger.info("=" * 70)
    logger.info(f"Stage 8: Optuna Hyperparameter Tuning - {asset}")
    logger.info("=" * 70)
    logger.info(f"  Train: {len(train_df)} rows | Val: {len(val_df)} rows")
    logger.info(f"  Eval steps: {eval_steps}")
    logger.info(f"  Trials: {n_trials} | Steps/trial: {trial_timesteps:,}")
    logger.info(f"  Eval freq: {eval_freq}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Objective: NAV% = (final_balance - initial) / initial * 100")
    logger.info(f"  Extractor: {extractor}  n_stack: {_n_stack}")
    logger.info(f"  Reward mode: {reward_mode} (forced)")
    logger.info(f"  Study: {study_name}")
    logger.info(f"  Storage: {storage}")
    logger.info(f"  FIXED - Iron rules:")
    logger.info(f"    ent_coef=0.1  batch=256  grad_steps=4  gamma=0.99  tau=0.005")
    if not tier3:
        logger.info(f"    net_arch=[256,256]  n_critics=2  top_drop=2")
    else:
        logger.info(f"    (Tier 3 active - net_arch, n_critics, top_drop SEARCHED)")
    logger.info(f"  SEARCHED - Tier 1:")
    logger.info(f"    lr=[1e-5, 1e-4]  buffer=[500K,1M,2M]  "
                f"learning_starts=[10K,30K,50K]  n_quantiles=[20,40,step=4]")
    logger.info(f"  SEARCHED - Tier 2 (classic):")
    logger.info(f"    dd_threshold  dd_penalty_mult  reward_clip  regime_weight_range")
    if tier3:
        logger.info(f"  SEARCHED - Tier 3 (architecture):")
        logger.info(f"    net_arch=[256x2, 384x3, 512x3]  n_critics=[2,3]  top_drop=[1-4]")
    logger.info("=" * 70)

    trial_times = []

    def _suggest_tier1(trial):
        """Tier 1: Core hyperparams that directly affect learning dynamics."""
        return {
            'lr': trial.suggest_float('lr', 1e-5, 1e-4, log=True),
            'buffer_size': trial.suggest_categorical('buffer_size',
                                                     [500_000, 1_000_000, 2_000_000]),
            'learning_starts': trial.suggest_categorical('learning_starts',
                                                         [10_000, 30_000, 50_000]),
            'n_quantiles': trial.suggest_int('n_quantiles', 20, 40, step=4),
        }

    def _suggest_tier2_classic(trial):
        """Tier 2: Reward shaping params (classic mode). Continuous floats for TPE."""
        return {
            'dd_threshold': trial.suggest_float('dd_threshold', 0.05, 0.15),
            'dd_penalty_mult': trial.suggest_float('dd_penalty_mult', 1.0, 5.0),
            'reward_clip': trial.suggest_categorical('reward_clip', [20, 50, 100]),
            'regime_weight_min': trial.suggest_float('regime_weight_min', 0.3, 0.8),
            'regime_weight_max': trial.suggest_float('regime_weight_max', 1.5, 3.0),
        }

    def _suggest_tier2_sortino(trial):
        """Tier 2: Reward shaping params (sortino mode - extends classic)."""
        params = _suggest_tier2_classic(trial)
        params.update({
            'pnl_weight': trial.suggest_float('pnl_weight', 0.3, 0.7, step=0.1),
            'sortino_weight': trial.suggest_float('sortino_weight', 0.1, 0.5, step=0.1),
            'quality_weight': trial.suggest_float('quality_weight', 0.05, 0.2, step=0.05),
            'risk_weight': trial.suggest_float('risk_weight', 0.05, 0.2, step=0.05),
            'sortino_window': trial.suggest_categorical('sortino_window', [24, 48, 96]),
            # Stage 9: turnover_cost_mult replaces old turnover_penalty
            'turnover_cost_mult': trial.suggest_float('turnover_cost_mult', 0.5, 3.0),
            'turnover_min_change': trial.suggest_float('turnover_min_change', 0.005, 0.05),
        })
        return params

    # === Tier 3: Architecture (optional) ===
    NET_ARCH_MAP = {
        '256_256': [256, 256],
        '384_384_256': [384, 384, 256],
        '512_512_256': [512, 512, 256],
    }

    def _suggest_tier3(trial):
        """Tier 3: Architecture params (only when --optuna-tier3)."""
        return {
            'net_arch_choice': trial.suggest_categorical(
                'net_arch', list(NET_ARCH_MAP.keys())),
            'top_quantiles_to_drop': trial.suggest_int(
                'top_quantiles_to_drop', 1, 4),
            'n_critics': trial.suggest_categorical('n_critics', [2, 3]),
        }

    def _eval_nav(model, eval_env, n_episodes: int = 5) -> float:
        """Run eval episodes and return mean NAV%.

        NAV% = (mean_final_balance - initial_balance) / initial_balance * 100.
        Scale-invariant across different dd_penalty_mult values.
        """
        initial = eval_env.get_attr("initial_balance")[0]
        balances = []
        for _ in range(n_episodes):
            obs = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = eval_env.step(action)
                done = dones[0]
            # Get final balance from env info
            final_balance = infos[0].get("balance", initial)
            balances.append(final_balance)
        mean_balance = np.mean(balances)
        return (mean_balance - initial) / initial * 100

    def objective(trial):
        t_start = time.time()

        # === Tier 1: Core hyperparams ===
        t1 = _suggest_tier1(trial)

        # === Tier 2: Reward shaping (always classic) ===
        t2 = _suggest_tier2_classic(trial)

        # === Tier 3: Architecture (optional) ===
        t3 = _suggest_tier3(trial) if tier3 else {}

        # Log trial params
        tier3_str = ""
        if tier3:
            tier3_str = (
                f", arch={t3['net_arch_choice']}, "
                f"critics={t3['n_critics']}, drop={t3['top_quantiles_to_drop']}"
            )
        logger.info(
            f"\n  Trial {trial.number}: "
            f"lr={t1['lr']:.2e}, buf={t1['buffer_size']//1000}K, "
            f"starts={t1['learning_starts']//1000}K, quant={t1['n_quantiles']}, "
            f"dd_thr={t2['dd_threshold']:.3f}, dd_mult={t2['dd_penalty_mult']:.1f}, "
            f"clip={t2['reward_clip']}, "
            f"rw=[{t2['regime_weight_min']:.1f},{t2['regime_weight_max']:.1f}]"
            f"{tier3_str}"
        )

        # === Create environments with reward params ===
        env_kwargs = dict(
            feature_cols=feature_cols,
            bear_reward_multiplier=bear_mult,
            bull_reward_multiplier=bull_mult,
            sol_vol_multiplier=sol_vol_mult,
            reward_mode="classic",
            dd_threshold=t2['dd_threshold'],
            dd_penalty_mult=t2['dd_penalty_mult'],
            reward_clip=t2['reward_clip'],
            regime_weight_min=t2['regime_weight_min'],
            regime_weight_max=t2['regime_weight_max'],
            # Stage 9: Friction enabled in Optuna trials too
            asset=asset,
            friction_enabled=True,
        )

        model = None
        train_env = None
        eval_env = None

        try:
            train_env = DummyVecEnv([lambda: Monitor(TradingEnvFull(
                df=train_df, **env_kwargs))])
            eval_env = DummyVecEnv([lambda: Monitor(TradingEnvFull(
                df=val_df, **env_kwargs))])

            # Wrap with VecFrameStack for sequence extractors
            if _use_extractor and extractor in SEQUENCE_ARCHS:
                from stable_baselines3.common.vec_env import VecFrameStack
                train_env = VecFrameStack(train_env, n_stack=_n_stack)
                eval_env = VecFrameStack(eval_env, n_stack=_n_stack)
                logger.info(f"    VecFrameStack: n_stack={_n_stack}, "
                            f"stacked obs={obs_dim * _n_stack}")
            # === Build policy_kwargs ===
            # Architecture params: from Tier 3 or defaults
            if tier3:
                net_arch = NET_ARCH_MAP[t3['net_arch_choice']]
                n_critics = t3['n_critics']
                top_drop = t3['top_quantiles_to_drop']
            else:
                net_arch = [256, 256]
                n_critics = 2
                top_drop = 2

            policy_kwargs = {
                "net_arch": net_arch,
                "n_critics": n_critics,
                "n_quantiles": t1['n_quantiles'],
            }

            # Add extractor if using custom architecture
            if _use_extractor:
                ext_cls = get_extractor_class(extractor)
                ext_kwargs = get_extractor_kwargs(extractor, obs_dim)
                policy_kwargs["features_extractor_class"] = ext_cls
                policy_kwargs["features_extractor_kwargs"] = ext_kwargs
                logger.info(f"    Extractor: {ext_cls.__name__}")

            # === Iron rules: FIXED params ===
            model = TQC(
                "MlpPolicy",
                train_env,
                learning_rate=t1['lr'],
                buffer_size=t1['buffer_size'],
                learning_starts=t1['learning_starts'],
                batch_size=256,             # FIXED
                tau=0.005,                  # FIXED
                gamma=0.99,                 # FIXED
                train_freq=1,
                gradient_steps=4,           # FIXED
                ent_coef=0.1,               # FIXED - NEVER searched
                target_update_interval=1,
                top_quantiles_to_drop_per_net=top_drop,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
            )

            # Verify param counts
            total_params = sum(p.numel() for p in model.policy.parameters())
            # TQC actor-critic: extractor lives on actor, not policy root
            _fe = getattr(model.actor, 'features_extractor', None)
            ext_params = sum(
                p.numel() for p in _fe.parameters()
            ) if _fe is not None else 0
            assert ext_params < 500_000, (
                f"Extractor {ext_params:,} > 500K - iron rule #28"
            )
            logger.info(f"    Params: ext={ext_params:,} total={total_params:,}")

            # Eval + prune callback
            trial_dir = results_dir / f"trial_{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            prune_cb = OptunaPruneCallback(trial)

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(trial_dir),
                log_path=str(trial_dir),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                callback_after_eval=prune_cb,
            )

            model.learn(
                total_timesteps=trial_timesteps,
                callback=[
                    eval_callback,
                    GradientMonitorCallback(max_actor_loss=1e6,
                                            max_consecutive_explosions=3),
                ],
                progress_bar=False,
            )

            # Check if pruned
            if prune_cb.is_pruned:
                elapsed = time.time() - t_start
                trial_times.append(elapsed)
                step_pruned = prune_cb._eval_count * eval_freq
                logger.info(
                    f"  Trial {trial.number} PRUNED at ~{step_pruned:,} steps "
                    f"({elapsed:.0f}s)"
                )
                raise optuna.TrialPruned()

            # Get best raw reward (for pruner compatibility)
            best_reward = prune_cb.best_mean_reward
            eval_path = trial_dir / "evaluations.npz"
            if eval_path.exists():
                evals = np.load(eval_path)
                results_arr = evals.get("results", [])
                if len(results_arr) > 0:
                    best_reward = float(max(np.mean(r) for r in results_arr))

            # === NAV-based objective (scale-invariant) ===
            nav_pct = _eval_nav(model, eval_env, n_episodes=5)

            # Store metrics as user attributes
            trial.set_user_attr("best_raw_reward", best_reward)
            trial.set_user_attr("nav_pct", nav_pct)
            trial.set_user_attr("ext_params", ext_params)
            trial.set_user_attr("total_params", total_params)
            trial.set_user_attr("elapsed_s", time.time() - t_start)

            elapsed = time.time() - t_start
            trial_times.append(elapsed)
            logger.info(
                f"  Trial {trial.number} DONE: NAV={nav_pct:+.2f}% "
                f"(raw_reward={best_reward:.2f}, {elapsed:.0f}s)"
            )
            return nav_pct

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"  Trial {trial.number} FAILED: {e}")
            import traceback
            traceback.print_exc()
            trial_times.append(time.time() - t_start)
            return -float("inf")
        finally:
            # Free GPU memory between trials (prevents OOM after ~3 trials)
            if model is not None:
                del model
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    # === Create study with crash-safe SQLite storage ===
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=10,    # First 10 random (build baseline)
            multivariate=True,      # Model param correlations
            seed=42,
        ),
        pruner=MedianPruner(
            n_startup_trials=10,    # First 10 trials always complete
            n_warmup_steps=50_000,  # Don't prune before 50K training steps
            interval_steps=10_000,  # Check every 10K training steps
        ),
        load_if_exists=True,        # Crash recovery
    )

    # If resuming, report how many trials already exist
    existing = len(study.trials)
    if existing > 0:
        logger.info(f"\n  Resuming: {existing} trials already in DB")
        remaining = max(0, n_trials - existing)
        logger.info(f"  Running {remaining} more trials")
    else:
        remaining = n_trials

    logger.info(f"\nStarting {remaining} trials...")
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    # === Results ===
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials
              if t.state == optuna.trial.TrialState.FAIL]

    if not completed:
        logger.error("No completed trials! All were pruned or failed.")
        return {"asset": asset, "error": "no_completed_trials"}

    best = study.best_params

    # Parameter importance (requires completed trials)
    try:
        importances = optuna.importance.get_param_importances(study)
        importance_list = [(k, v) for k, v in importances.items()]
    except Exception:
        importance_list = []

    # Build fixed_params dict (includes Tier 3 if active)
    fixed_params = {
        "ent_coef": 0.1, "batch_size": 256, "gradient_steps": 4,
        "gamma": 0.99, "tau": 0.005, "weight_decay": 0,
        "extractor": extractor, "n_stack": _n_stack,
        "reward_mode": "classic", "augment": False,
    }
    if not tier3:
        fixed_params.update({
            "net_arch": [256, 256], "n_critics": 2,
            "top_quantiles_to_drop": 2,
        })

    results = {
        "asset": asset,
        "stage": 8,
        "objective": "nav_pct",
        "reward_mode": "classic",
        "best_params": best,
        "best_value": float(study.best_value),
        "best_nav_pct": float(study.best_value),
        "best_raw_reward": study.best_trial.user_attrs.get("best_raw_reward"),
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_failed": len(failed),
        "total_time_minutes": sum(trial_times) / 60 if trial_times else 0,
        "trial_timesteps": trial_timesteps,
        "eval_freq": eval_freq,
        "tier3_enabled": tier3,
        "fixed_params": fixed_params,
        "param_importances": importance_list,
        # All trial details
        "all_trials": [{
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
            "nav_pct": t.user_attrs.get("nav_pct"),
            "raw_reward": t.user_attrs.get("best_raw_reward"),
        } for t in study.trials],
        # Top 5 for easy comparison
        "top_5": [{
            "number": t.number,
            "value": t.value,
            "nav_pct": t.user_attrs.get("nav_pct"),
            "raw_reward": t.user_attrs.get("best_raw_reward"),
            "params": t.params,
        } for t in sorted(completed, key=lambda t: t.value or -1e9, reverse=True)[:5]],
    }

    results_path = Path(output_dir) / asset / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # === Print summary ===
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Stage 8 Complete - Optuna Tuning - {asset}")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Trials: {len(completed)} completed, {len(pruned)} pruned, "
                f"{len(failed)} failed")
    if trial_times:
        logger.info(f"  Total time: {sum(trial_times) / 60:.1f} min")
    logger.info(f"  Best NAV: {results['best_nav_pct']:+.2f}%")
    logger.info(f"  Best raw reward: {results['best_raw_reward']}")
    logger.info(f"  Best params:")
    for k, v in best.items():
        if isinstance(v, float):
            logger.info(f"    {k}: {v:.6f}")
        else:
            logger.info(f"    {k}: {v}")

    if importance_list:
        logger.info(f"\n  Parameter importance:")
        for name, imp in importance_list[:10]:
            bar = "█" * int(imp * 40)
            logger.info(f"    {name:25s} {imp:.3f} {bar}")

    # Boundary check: warn if best params are at search space edges
    logger.info(f"\n  Boundary check:")
    edge_warnings = []
    best_trial = study.best_trial
    for param_name, param_value in best.items():
        dist = best_trial.distributions.get(param_name)
        if dist is None:
            continue
        if hasattr(dist, 'low') and hasattr(dist, 'high'):
            rng = dist.high - dist.low
            if rng > 0:
                if param_value <= dist.low + rng * 0.01:
                    edge_warnings.append(
                        f"{param_name}={param_value} at lower bound")
                elif param_value >= dist.high - rng * 0.01:
                    edge_warnings.append(
                        f"{param_name}={param_value} at upper bound")
        elif hasattr(dist, 'choices'):
            choices = list(dist.choices)
            if len(choices) >= 2 and param_value == choices[0]:
                edge_warnings.append(
                    f"{param_name}={param_value} at lowest choice")
            elif len(choices) >= 2 and param_value == choices[-1]:
                edge_warnings.append(
                    f"{param_name}={param_value} at highest choice")
    if edge_warnings:
        for w in edge_warnings:
            logger.warning(f"    [WARN] {w}")
    else:
        logger.info(f"    ✓ No params at search space edges")

    # At least 5 completed trials check
    if len(completed) < 5:
        logger.warning(f"    [WARN] Only {len(completed)} completed (target: ≥5)")
    else:
        logger.info(f"    ✓ {len(completed)} completed trials (≥5 target)")

    logger.info(f"\n  Top 5 trials (by NAV%):")
    for entry in results["top_5"]:
        p = entry['params']
        nav = entry.get('nav_pct', entry['value'])
        raw = entry.get('raw_reward', '?')
        logger.info(
            f"    #{entry['number']}: NAV={nav:+.2f}% (raw={raw}) - "
            f"lr={p.get('lr', '?'):.2e}, "
            f"buf={p.get('buffer_size', '?')}, "
            f"starts={p.get('learning_starts', '?')}, "
            f"quant={p.get('n_quantiles', '?')}"
        )

    logger.info(f"\n  Results saved: {results_path}")
    logger.info(f"  DB: {db_path}")
    logger.info(f"\n  Next steps:")
    logger.info(f"    Phase B: Top 3 params × 2.5M steps × 3 folds")
    if not tier3:
        logger.info(f"    Phase C (optional): --optuna-tier3 for architecture search")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HMATS v7 Ultimate Rebuild DRL Training"
    )

    parser.add_argument("--asset", type=str, required=True,
                        choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--data", type=str, default=None,
                        help="Parquet data file (default: training/training_data/drl_training/{ASSET}_4H_full.parquet)")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps (default: asset-specific)")
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models" / "retrained"))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--bear-mult", type=float, default=1.2)
    parser.add_argument("--bull-mult", type=float, default=1.1)
    parser.add_argument("--smooth-regimes", type=int, default=2,
                        help="RegimeSmoother persistence (0=disable)")
    parser.add_argument("--purge-window", type=int, default=0,
                        help="P4: Extra bars to purge from train end before gap (default: 0)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cuda:0, cpu")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--vec-env", type=str, default="dummy",
                        choices=["dummy", "subproc"],
                        help="VecEnv type: dummy (safe) or subproc (faster, may deadlock on Windows)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel envs (only used with --vec-env subproc)")
    parser.add_argument("--reward-mode", type=str, default="classic",
                        choices=["classic", "sortino", "sharpe"],
                        help="Reward function: classic (PnL), sortino (risk-adjusted), or sharpe (rolling Sharpe)")
    parser.add_argument("--extractor", type=str, default=None,
                        choices=["lstm", "lstm_film", "lstm_film_lite",
                                 "lstm_film_a", "lstm_film_b"],
                        help="Stage 6: custom feature extractor (default: flat MLP)")
    parser.add_argument("--augment", action="store_true",
                        help="Stage 7: state augmentation (2%% noise + 5%% dropout) on train env only")

    # Stage 8: reward / TQC hyperparams (Optuna-derived or manual)
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer size (default: ULTIMATE 2M)")
    parser.add_argument("--learning-starts", type=int, default=None,
                        help="Steps before learning starts (default: 30K)")
    parser.add_argument("--n-quantiles", type=int, default=None,
                        help="TQC n_quantiles (default: 32)")
    parser.add_argument("--dd-threshold", type=float, default=0.10,
                        help="Drawdown threshold for penalty (default: 0.10)")
    parser.add_argument("--dd-penalty-mult", type=float, default=2.0,
                        help="Drawdown penalty multiplier (default: 2.0)")
    parser.add_argument("--reward-clip", type=float, default=50.0,
                        help="Reward clipping bound (default: 50)")
    parser.add_argument("--regime-weight-min", type=float, default=0.5,
                        help="Min regime weight (default: 0.5)")
    parser.add_argument("--regime-weight-max", type=float, default=2.0,
                        help="Max regime weight (default: 2.0)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag suffix for output directory (e.g. config1_optuna)")

    # Stage 9: Friction args
    parser.add_argument("--no-friction", action="store_true",
                        help="Disable real friction costs (baseline comparison)")
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="Override per-asset slippage (bps). Default: BTC=3, ETH=5, SOL=10")
    parser.add_argument("--turnover-mult", type=float, default=1.0,
                        help="Turnover cost multiplier (default: 1.0)")
    parser.add_argument("--turnover-min-change", type=float, default=0.01,
                        help="Min position change to trigger friction (default: 0.01)")

    # Optuna tuning args
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (e.g. config/optuna_winner.json). "
                             "CLI args override config values.")

    parser.add_argument("--optuna", action="store_true",
                        help="Run Optuna hyperparameter tuning instead of training")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--optuna-timesteps", type=int, default=200_000,
                        help="Timesteps per Optuna trial (default: 200K)")
    parser.add_argument("--optuna-eval-freq", type=int, default=10_000,
                        help="Eval frequency for Optuna trials (default: 10K)")
    parser.add_argument("--optuna-tier3", action="store_true",
                        help="Stage 8: enable Tier 3 architecture search (net_arch, n_critics, top_drop)")

    args = parser.parse_args()

    # Load config JSON if provided (CLI args override config values)
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / args.config
        with open(config_path) as f:
            cfg = json.load(f)
        logger.info(f"  Loaded config: {config_path}")

        # Map config keys -> argparse defaults (only if CLI didn't explicitly set them)
        _config_map = {
            'lr': 'lr',
            'buffer_size': 'buffer_size',
            'learning_starts': 'learning_starts',
            'n_quantiles': 'n_quantiles',
            'dd_threshold': 'dd_threshold',
            'dd_penalty_mult': 'dd_penalty_mult',
            'reward_clip': 'reward_clip',
            'regime_weight_min': 'regime_weight_min',
            'regime_weight_max': 'regime_weight_max',
            'reward_mode': 'reward_mode',
            'extractor': 'extractor',
            'ent_coef': None,  # always fixed 0.1, ignore
            'batch_size': None,  # always 256, ignore
            'grad_steps': None,  # always 4, ignore
        }
        # Detect which args were explicitly passed on CLI
        _cli_explicit = {a.dest for a in parser._actions
                         if a.dest in sys.argv or any(
                             o in sys.argv for o in (a.option_strings or [])
                         )}
        for cfg_key, arg_key in _config_map.items():
            if arg_key is None or cfg_key not in cfg:
                continue
            # Only apply config if CLI didn't explicitly set this arg
            if arg_key not in _cli_explicit:
                setattr(args, arg_key, cfg[cfg_key])
                logger.info(f"    config: {arg_key} = {cfg[cfg_key]}")

        # Force buffer_size=500K max for memory safety (n_stack=8 amplification)
        if args.buffer_size and args.buffer_size > 500_000:
            logger.warning(f"  [MEM] buffer_size {args.buffer_size:,} > 500K, capping to 500K "
                          f"(n_stack=8 memory: {args.buffer_size * 8 * 126 * 4 * 2 / 1e9:.1f} GB)")
            args.buffer_size = 500_000

    # Resolve data path
    data_path = args.data or str(_TRAINING_DIR / "training_data" / "drl_training" / f"{args.asset}_4H_full.parquet")
    logger.info(f"Loading data: {data_path}")

    data = pd.read_parquet(data_path)
    logger.info(f"  Loaded: {len(data)} rows, {len(data.columns)} columns")

    # Apply RegimeSmoother
    if args.smooth_regimes > 0 and "regime" in data.columns:
        smoother = RegimeSmoother(min_persistence=args.smooth_regimes)
        before = sum(1 for i in range(1, len(data))
                     if data["regime"].iloc[i] != data["regime"].iloc[i - 1])
        data = smoother.smooth_column(data, "regime")
        after = sum(1 for i in range(1, len(data))
                    if data["regime"].iloc[i] != data["regime"].iloc[i - 1])
        reduction = (1 - after / max(before, 1)) * 100
        logger.info(
            f"  RegimeSmoother(persistence={args.smooth_regimes}): "
            f"flips {before} -> {after} ({reduction:.0f}% reduction)"
        )
        if reduction == 0:
            logger.info("  RegimeSmoother: data already smoothed upstream, no effect")

    # Feature columns: load from manifest (preferred) or auto-detect fallback
    manifest_path = PROJECT_ROOT / "configs" / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        feature_cols = [c for c in manifest["all_features"] if c in data.columns]
        no_scale = set(manifest.get("no_scale_features", []))
        logger.info(f"  Features: {len(feature_cols)} (from manifest, "
                    f"{len(no_scale)} no-scale)")
    else:
        # Fallback: auto-detect float cols, exclude OHLCV + regime metadata
        feature_cols = [
            col for col in data.columns
            if data[col].dtype in ["float64", "float32"]
            and col not in [
                "timestamp", "open", "high", "low", "close", "volume",
                "regime", "regime_confidence",
            ]
        ]
        no_scale = set()
        logger.warning("  No feature_manifest.json found - using auto-detect fallback")
        logger.info(f"  Features: {len(feature_cols)} (auto-detected)")
    logger.info(f"  Obs dim: {len(feature_cols) + 4} (features + 4 env state)")

    # Save feature_cols for deployment
    output_dir = Path(args.output) / args.asset
    output_dir.mkdir(parents=True, exist_ok=True)
    fc_path = output_dir / "feature_cols.json"
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"  Saved feature_cols: {fc_path}")

    # Override global REGIME_NAMES with per-asset names
    global REGIME_NAMES
    REGIME_NAMES = _load_regime_names(asset=args.asset)
    logger.info(f"  Regime names ({args.asset}): k={len(REGIME_NAMES)} {REGIME_NAMES}")

    # Asset-specific defaults
    overrides = ASSET_OVERRIDES.get(args.asset, {})
    sol_vol = overrides.get("sol_vol_multiplier", 1.0)

    if device := args.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Optuna mode ===
    if args.optuna:
        optuna_ext = args.extractor or "lstm_film_a"
        logger.info("\n  [MODE] Stage 8 - Optuna Hyperparameter Tuning")
        logger.info(f"  [CHECK] ent_coef = 0.1 (FIXED, not searched)")
        logger.info(f"  [CHECK] batch=256, grad_steps=4, gamma=0.99, tau=0.005 (FIXED)")
        logger.info(f"  [CHECK] reward_mode = classic (forced)")
        logger.info(f"  [CHECK] extractor = {optuna_ext}")
        logger.info(f"  [CHECK] Tier 3 = {args.optuna_tier3}")
        run_optuna_tuning(
            asset=args.asset,
            data=data,
            feature_cols=feature_cols,
            output_dir=args.output,
            n_trials=args.optuna_trials,
            trial_timesteps=args.optuna_timesteps,
            eval_freq=args.optuna_eval_freq,
            device=device,
            bear_mult=args.bear_mult,
            bull_mult=args.bull_mult,
            sol_vol_mult=sol_vol,
            no_scale_features=no_scale,
            reward_mode="classic",
            extractor=optuna_ext,
            tier3=args.optuna_tier3,
        )
        return

    # === Training mode ===
    timesteps = args.timesteps or overrides.get("total_timesteps", 2_500_000)

    logger.info(f"\n  [CHECK] VecEnv = {args.vec_env} (n_envs={args.n_envs})")
    logger.info(f"  [CHECK] ent_coef = {ULTIMATE_PRESET['ent_coef']} (FIXED)")
    logger.info(f"  [CHECK] checkpoint_freq = 500K")
    logger.info(f"  [CHECK] gap = 42, window = 10, val_ratio = 0.15")
    logger.info(f"  [CHECK] RobustScaler: ENABLED (P2)")
    logger.info(f"  [CHECK] Regime weighting: ENABLED (P3)")
    logger.info(f"  [CHECK] Reward mode: {args.reward_mode}")
    logger.info(f"  [CHECK] Purge window: {args.purge_window} (P4)")
    if args.tag:
        logger.info(f"  [CHECK] Tag: {args.tag}")
    if args.buffer_size or args.learning_starts or args.n_quantiles:
        logger.info(f"  [CHECK] TQC overrides: buffer={args.buffer_size}, "
                     f"learning_starts={args.learning_starts}, n_quantiles={args.n_quantiles}")
    if args.dd_threshold != 0.0866 or args.dd_penalty_mult != 2.8243 or args.reward_clip != 20.0:
        logger.info(f"  [CHECK] Reward overrides: dd_thr={args.dd_threshold}, "
                     f"dd_mult={args.dd_penalty_mult}, clip={args.reward_clip}")
    if args.regime_weight_min != 0.5 or args.regime_weight_max != 2.0:
        logger.info(f"  [CHECK] Regime weight: [{args.regime_weight_min}, {args.regime_weight_max}]")

    trainer = FullDRLTrainer(
        data=data,
        feature_cols=feature_cols,
        asset=args.asset,
        output_dir=args.output,
        n_folds=args.folds,
        total_timesteps=timesteps,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        bear_reward_multiplier=args.bear_mult,
        bull_reward_multiplier=args.bull_mult,
        sol_vol_multiplier=sol_vol,
        resume=args.resume,
        device=device,
        progress_bar=not args.no_progress_bar,
        purge_window=args.purge_window,
        no_scale_features=no_scale,
        reward_mode=args.reward_mode,
        extractor=args.extractor,
        augment=args.augment,
        # Stage 8: Optuna-derived overrides
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        n_quantiles=args.n_quantiles,
        dd_threshold=args.dd_threshold,
        dd_penalty_mult=args.dd_penalty_mult,
        reward_clip=args.reward_clip,
        regime_weight_min=args.regime_weight_min,
        regime_weight_max=args.regime_weight_max,
        tag=args.tag,
        # Stage 9: Friction
        friction_enabled=not args.no_friction,
        slippage_bps=args.slippage_bps,
        turnover_cost_mult=args.turnover_mult,
        turnover_min_change=args.turnover_min_change,
        # VecEnv
        vec_env_type=args.vec_env,
        n_envs=args.n_envs,
    )

    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # Let argparse --help / errors exit normally
    except Exception as _fatal:
        logging.getLogger(__name__).error(f"FATAL: {_fatal}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

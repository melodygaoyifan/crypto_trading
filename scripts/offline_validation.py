"""
E2E Step 12: Offline Validation (Stage 13)

Friction-aware closed-loop backtest of TQC (LSTM_FILM_A) on OOS data.
Runs per-asset, per-fold walk-forward validation with all metrics from the
E2E guide.

Metrics per fold:
  - OOS Sharpe (annualized)
  - Max drawdown %
  - Trade count
  - Action entropy (3-bin: long/neutral/short)
  - vs Random: Sharpe difference
  - Cumulative friction cost % of NAV
  - Avg turnover per bar

StatisticalPromotionGate:
  - min_sharpe: 0.7 (v9 adjusted)
  - min_bear_sharpe: 0.5
  - max_drawdown: 15%
  - max_bear_drawdown: 20%
  - min_sample_size: 30 trades
  - min_win_rate: 0.52

Usage:
    python -X utf8 -u scripts/offline_validation.py
    python -X utf8 -u scripts/offline_validation.py --asset BTC
"""

import importlib
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Project root on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# Register training.models for cloudpickle deserialization
def _register_modules():
    if "models" not in sys.modules or sys.modules["models"] is None:
        try:
            sys.modules["models"] = importlib.import_module("training.models")
            sys.modules["models.film_extractor"] = importlib.import_module("training.models.film_extractor")
            sys.modules["models.feature_extractors"] = importlib.import_module("training.models.feature_extractors")
        except Exception:
            pass

_register_modules()

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("OfflineValidation")

# ============================================================================
# Config
# ============================================================================

ASSETS = ["BTC", "ETH", "SOL"]
N_FOLDS = 3
VAL_RATIO = 0.15
GAP_BARS = 42
N_STACK = 8
INITIAL_BALANCE = 100_000.0
ANNUALIZE_FACTOR = np.sqrt(6 * 365.25)  # 4H bars -> annual (6 bars/day × 365.25 days)

# WalkForwardValidator thresholds (E2E guide v9)
WFV_MIN_SHARPE = 0.3
WFV_MAX_DD = 0.15
WFV_MIN_TRADES = 10
WFV_MAX_ACTION_ENTROPY = 0.7   # Normalized entropy [0,1]
WFV_SHARPE_VS_RANDOM = 0.2
WFV_MAX_FRICTION_PCT = 0.15    # Cumulative friction < 15% of NAV
WFV_MAX_AVG_TURNOVER = 0.3     # Avg |Δposition| per bar

# StatisticalPromotionGate thresholds (E2E guide v9)
SPG_MIN_SHARPE = 0.7
SPG_MIN_BEAR_SHARPE = 0.5
SPG_MAX_DD = 0.15
SPG_MAX_BEAR_DD = 0.20
SPG_MIN_TRADES = 30
SPG_MIN_WIN_RATE = 0.52

BEAR_REGIMES = frozenset(["PANIC_SELLOFF", "VOLATILE_CHOP", "EXTREME_VOLATILITY"])
BULL_REGIMES = frozenset(["MOMENTUM_RALLY", "QUIET_ACCUMULATION"])


# ============================================================================
# Data helpers
# ============================================================================

def load_feature_manifest() -> dict:
    with open("configs/feature_manifest.json") as f:
        return json.load(f)


def load_parquet(asset: str) -> pd.DataFrame:
    path = f"training/training_data/drl_training/{asset}_4H_full.parquet"
    return pd.read_parquet(path)


def get_fold_splits(df: pd.DataFrame, n_folds: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Anchored walk-forward splits (same as training)."""
    n = len(df)
    val_size = int(n * VAL_RATIO)
    splits = []
    for fold_idx in range(n_folds):
        val_end = n - fold_idx * val_size
        val_start = val_end - val_size
        train_end = val_start - GAP_BARS
        if train_end < 100:
            continue
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        splits.append((train_df, val_df))
    return splits


def scale_features(
    val_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: str,
) -> pd.DataFrame:
    """Scale val features using fold scaler."""
    scaler_obj = joblib.load(scaler_path)
    if isinstance(scaler_obj, dict):
        scaler = scaler_obj["scaler"]
        scale_cols = scaler_obj.get("scale_cols", [])
    else:
        scaler = scaler_obj
        scale_cols = feature_cols  # fallback

    valid_scale = [c for c in scale_cols if c in val_df.columns]
    val_scaled = val_df.copy()
    if valid_scale:
        val_scaled[valid_scale] = scaler.transform(val_scaled[valid_scale])
        for c in valid_scale:
            val_scaled[c] = val_scaled[c].replace([np.inf, -np.inf], 0).fillna(0).clip(-10.0, 10.0)
    return val_scaled


# ============================================================================
# Environment + Model loading
# ============================================================================

def make_eval_env(val_scaled: pd.DataFrame, feature_cols: List[str], asset: str):
    """Create friction-aware eval env wrapped with DummyVecEnv + VecFrameStack."""
    from training.train_drl_full import TradingEnvFull
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    env = TradingEnvFull(
        df=val_scaled,
        feature_cols=feature_cols,
        initial_balance=INITIAL_BALANCE,
        max_position=1.0,
        reward_mode="classic",
        bear_reward_multiplier=1.2,
        bull_reward_multiplier=1.1,
        sol_vol_multiplier=0.6 if asset == "SOL" else 1.0,
        asset=asset,
        friction_enabled=True,
        augment_enabled=False,  # No augmentation for eval
    )
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    return vec_env


def load_tqc_model(asset: str, fold: int):
    """Load TQC model for a specific fold."""
    from sb3_contrib import TQC
    model_path = f"models/retrained/{asset}/fold_{fold}/logs/LSTM_FILM_A/best_model/best_model"
    return TQC.load(model_path, device="cuda")


# ============================================================================
# Episode runner
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Detailed metrics from one episode."""
    # Core
    final_balance: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    max_drawdown: float = 0.0
    # Per-step
    per_step_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    regimes: List[str] = field(default_factory=list)
    # Trades
    trade_count: int = 0
    cumulative_friction_cost: float = 0.0
    # Sharpe
    sharpe: float = 0.0
    sortino: float = 0.0
    # Bear/bull splits
    bear_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    bull_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    bear_sharpe: float = 0.0
    bear_max_dd: float = 0.0
    # Action stats
    action_entropy: float = 0.0
    avg_turnover: float = 0.0
    friction_pct_nav: float = 0.0
    win_rate: float = 0.0
    n_steps: int = 0


def run_episode(model, env, random_mode: bool = False) -> EpisodeMetrics:
    """Run one deterministic episode, collect detailed metrics."""
    obs = env.reset()
    done = False

    balances = []
    positions = []
    regimes = []
    actions_raw = []
    step_returns = []
    prev_balance = INITIAL_BALANCE
    prev_position = 0.0
    turnovers = []

    while not done:
        if random_mode:
            action = np.random.uniform(-1, 1, size=(1, 1)).astype(np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]

        bal = info.get("balance", INITIAL_BALANCE)
        pos = info.get("position", 0.0)
        regime = info.get("regime", "UNKNOWN")

        # Per-step return (balance-based)
        step_ret = (bal - prev_balance) / max(abs(prev_balance), 1.0)
        step_returns.append(step_ret)
        balances.append(bal)
        positions.append(pos)
        regimes.append(regime)
        actions_raw.append(float(action[0][0]) if hasattr(action[0], '__len__') else float(action[0]))
        turnovers.append(abs(pos - prev_position))

        prev_balance = bal
        prev_position = pos

    # --- Compute metrics ---
    m = EpisodeMetrics()
    m.n_steps = len(step_returns)
    if m.n_steps == 0:
        return m

    balances = np.array(balances)
    step_returns = np.array(step_returns)
    positions = np.array(positions)
    actions_raw = np.array(actions_raw)
    turnovers = np.array(turnovers)

    m.per_step_returns = step_returns
    m.positions = positions
    m.regimes = regimes
    m.final_balance = float(balances[-1])
    m.total_pnl = float(balances[-1] - INITIAL_BALANCE)
    m.return_pct = (m.final_balance / INITIAL_BALANCE - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(balances)
    dd = (peak - balances) / np.maximum(peak, 1.0)
    m.max_drawdown = float(np.max(dd))

    # Sharpe (annualized)
    mean_ret = float(np.mean(step_returns))
    std_ret = float(np.std(step_returns))
    m.sharpe = (mean_ret / max(std_ret, 1e-10)) * ANNUALIZE_FACTOR

    # Sortino (annualized)
    neg_returns = step_returns[step_returns < 0]
    down_std = float(np.std(neg_returns)) if len(neg_returns) > 2 else 1e-10
    m.sortino = (mean_ret / max(down_std, 1e-10)) * ANNUALIZE_FACTOR

    # Trades (position changes > threshold)
    m.trade_count = int(info.get("trade_count", 0))
    m.cumulative_friction_cost = float(info.get("cumulative_trade_cost", 0.0))
    m.friction_pct_nav = m.cumulative_friction_cost / INITIAL_BALANCE

    # Win rate (per-step)
    m.win_rate = float(np.mean(step_returns > 0)) if len(step_returns) > 0 else 0.0

    # Avg turnover per bar
    m.avg_turnover = float(np.mean(turnovers))

    # Action entropy (3-bin: short/neutral/long)
    bins = np.array([-1.0, -0.1, 0.1, 1.01])
    hist, _ = np.histogram(actions_raw, bins=bins)
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(3)  # max for 3 bins
    m.action_entropy = entropy / max_entropy  # Normalized [0, 1]

    # Bear/Bull split
    regimes_arr = np.array(regimes)
    bear_mask = np.isin(regimes_arr, list(BEAR_REGIMES))
    bull_mask = np.isin(regimes_arr, list(BULL_REGIMES))

    m.bear_returns = step_returns[bear_mask]
    m.bull_returns = step_returns[bull_mask]

    if len(m.bear_returns) > 2:
        bear_mean = float(np.mean(m.bear_returns))
        bear_std = float(np.std(m.bear_returns))
        m.bear_sharpe = (bear_mean / max(bear_std, 1e-10)) * ANNUALIZE_FACTOR

        # Bear max DD
        bear_cum = np.cumsum(m.bear_returns)
        bear_peak = np.maximum.accumulate(bear_cum)
        bear_dd = bear_peak - bear_cum
        m.bear_max_dd = float(np.max(bear_dd)) / INITIAL_BALANCE if len(bear_dd) > 0 else 0.0

    return m


# ============================================================================
# Gate checks
# ============================================================================

@dataclass
class FoldGateResult:
    fold: int
    asset: str
    metrics: EpisodeMetrics
    random_sharpe: float = 0.0
    # WalkForwardValidator
    wfv_sharpe_pass: bool = False
    wfv_dd_pass: bool = False
    wfv_trades_pass: bool = False
    wfv_entropy_pass: bool = False
    wfv_vs_random_pass: bool = False
    wfv_friction_pass: bool = False
    wfv_turnover_pass: bool = False
    wfv_all_pass: bool = False
    # StatisticalPromotionGate
    spg_sharpe_pass: bool = False
    spg_bear_sharpe_pass: bool = False
    spg_dd_pass: bool = False
    spg_bear_dd_pass: bool = False
    spg_trades_pass: bool = False
    spg_win_rate_pass: bool = False
    spg_all_pass: bool = False


def check_gates(m: EpisodeMetrics, random_sharpe: float, fold: int, asset: str) -> FoldGateResult:
    g = FoldGateResult(fold=fold, asset=asset, metrics=m, random_sharpe=random_sharpe)

    # WalkForwardValidator
    g.wfv_sharpe_pass = m.sharpe >= WFV_MIN_SHARPE
    g.wfv_dd_pass = m.max_drawdown <= WFV_MAX_DD
    g.wfv_trades_pass = m.trade_count >= WFV_MIN_TRADES
    g.wfv_entropy_pass = m.action_entropy <= WFV_MAX_ACTION_ENTROPY
    g.wfv_vs_random_pass = (m.sharpe - random_sharpe) > WFV_SHARPE_VS_RANDOM
    g.wfv_friction_pass = m.friction_pct_nav < WFV_MAX_FRICTION_PCT
    g.wfv_turnover_pass = m.avg_turnover < WFV_MAX_AVG_TURNOVER
    g.wfv_all_pass = all([
        g.wfv_sharpe_pass, g.wfv_dd_pass, g.wfv_trades_pass,
        g.wfv_entropy_pass, g.wfv_vs_random_pass, g.wfv_friction_pass,
        g.wfv_turnover_pass,
    ])

    # StatisticalPromotionGate
    g.spg_sharpe_pass = m.sharpe >= SPG_MIN_SHARPE
    g.spg_bear_sharpe_pass = m.bear_sharpe >= SPG_MIN_BEAR_SHARPE or len(m.bear_returns) < 5
    g.spg_dd_pass = m.max_drawdown <= SPG_MAX_DD
    g.spg_bear_dd_pass = m.bear_max_dd <= SPG_MAX_BEAR_DD or len(m.bear_returns) < 5
    g.spg_trades_pass = m.trade_count >= SPG_MIN_TRADES
    g.spg_win_rate_pass = m.win_rate >= SPG_MIN_WIN_RATE
    g.spg_all_pass = all([
        g.spg_sharpe_pass, g.spg_bear_sharpe_pass, g.spg_dd_pass,
        g.spg_bear_dd_pass, g.spg_trades_pass, g.spg_win_rate_pass,
    ])

    return g


# ============================================================================
# Per-asset validation
# ============================================================================

def validate_asset(asset: str, manifest: dict):
    logger.info(f"\n{'='*70}")
    logger.info(f"  OFFLINE VALIDATION - {asset}")
    logger.info(f"{'='*70}")

    df = load_parquet(asset)
    feature_cols = [c for c in manifest["all_features"] if c in df.columns]
    logger.info(f"  Data: {len(df)} rows, {len(feature_cols)} features")

    splits = get_fold_splits(df, N_FOLDS)
    fold_results: List[FoldGateResult] = []

    for fold_idx, (train_df, val_df) in enumerate(splits):
        fold_num = fold_idx + 1
        logger.info(f"\n  --- Fold {fold_num} ---")
        logger.info(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")

        # Scale features
        scaler_path = f"models/retrained/{asset}/fold_{fold_num}/logs/LSTM_FILM_A/feature_scaler.pkl"
        if not os.path.exists(scaler_path):
            logger.warning(f"  Scaler not found: {scaler_path}, skipping")
            continue

        val_scaled = scale_features(val_df, feature_cols, scaler_path)

        # Create eval env
        eval_env = make_eval_env(val_scaled, feature_cols, asset)

        # Load model
        try:
            model = load_tqc_model(asset, fold_num)
            logger.info(f"  Model loaded: fold_{fold_num}")
        except Exception as e:
            logger.error(f"  Model load failed: {e}")
            eval_env.close()
            continue

        # Run TQC episode
        logger.info(f"  Running TQC episode ({len(val_df)} bars)...")
        m = run_episode(model, eval_env)

        # Run random baseline
        logger.info(f"  Running random baseline...")
        eval_env_rand = make_eval_env(val_scaled, feature_cols, asset)
        rand_m = run_episode(None, eval_env_rand, random_mode=True)
        eval_env_rand.close()

        # Gate checks
        g = check_gates(m, rand_m.sharpe, fold_num, asset)
        fold_results.append(g)

        # Print fold results
        logger.info(f"\n  Fold {fold_num} Results:")
        logger.info(f"    Return:       {m.return_pct:+.2f}%  (${m.total_pnl:+,.0f})")
        logger.info(f"    Sharpe:       {m.sharpe:.4f}  (threshold: {WFV_MIN_SHARPE})")
        logger.info(f"    Sortino:      {m.sortino:.4f}")
        logger.info(f"    MaxDD:        {m.max_drawdown:.2%}  (threshold: {WFV_MAX_DD:.0%})")
        logger.info(f"    Trades:       {m.trade_count}  (threshold: {WFV_MIN_TRADES})")
        logger.info(f"    Win Rate:     {m.win_rate:.2%}  (threshold: {SPG_MIN_WIN_RATE:.0%})")
        logger.info(f"    Friction:     ${m.cumulative_friction_cost:,.2f}  ({m.friction_pct_nav:.2%} of NAV)")
        logger.info(f"    Avg Turnover: {m.avg_turnover:.4f}/bar  (threshold: {WFV_MAX_AVG_TURNOVER})")
        logger.info(f"    Action Ent:   {m.action_entropy:.4f}  (threshold: {WFV_MAX_ACTION_ENTROPY})")
        logger.info(f"    Random Sharpe:{rand_m.sharpe:.4f}  (diff: {m.sharpe - rand_m.sharpe:+.4f}, need >{WFV_SHARPE_VS_RANDOM})")
        logger.info(f"    Bear Sharpe:  {m.bear_sharpe:.4f}  ({len(m.bear_returns)} bars)")
        logger.info(f"    Bear MaxDD:   {m.bear_max_dd:.2%}  ({SPG_MAX_BEAR_DD:.0%} threshold)")

        # Gate summary
        wfv_checks = [
            ("Sharpe≥0.3", g.wfv_sharpe_pass),
            ("DD≤15%", g.wfv_dd_pass),
            ("Trades≥10", g.wfv_trades_pass),
            ("Entropy≤0.7", g.wfv_entropy_pass),
            ("vsRandom+0.2", g.wfv_vs_random_pass),
            ("Friction<15%", g.wfv_friction_pass),
            ("Turnover<0.3", g.wfv_turnover_pass),
        ]
        spg_checks = [
            ("Sharpe≥0.7", g.spg_sharpe_pass),
            ("BearSharpe≥0.5", g.spg_bear_sharpe_pass),
            ("DD≤15%", g.spg_dd_pass),
            ("BearDD≤20%", g.spg_bear_dd_pass),
            ("Trades≥30", g.spg_trades_pass),
            ("WinRate≥52%", g.spg_win_rate_pass),
        ]

        wfv_str = " | ".join(f"{name}:{'PASS' if ok else 'FAIL'}" for name, ok in wfv_checks)
        spg_str = " | ".join(f"{name}:{'PASS' if ok else 'FAIL'}" for name, ok in spg_checks)
        logger.info(f"    WFV: [{wfv_str}] -> {'PASS' if g.wfv_all_pass else 'FAIL'}")
        logger.info(f"    SPG: [{spg_str}] -> {'PASS' if g.spg_all_pass else 'FAIL'}")

        eval_env.close()

    return fold_results


# ============================================================================
# Summary + Decision
# ============================================================================

def print_summary(all_results: Dict[str, List[FoldGateResult]]):
    logger.info(f"\n{'='*70}")
    logger.info(f"  OFFLINE VALIDATION SUMMARY")
    logger.info(f"{'='*70}")

    # Per-asset summary table
    logger.info(f"\n  {'Asset':>5} | {'Fold':>4} | {'Return':>8} | {'Sharpe':>7} | {'MaxDD':>7} | "
                f"{'Trades':>6} | {'WinRate':>7} | {'WFV':>4} | {'SPG':>4}")
    logger.info(f"  {'-'*5}-+-{'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*4}-+-{'-'*4}")

    for asset, results in all_results.items():
        for g in results:
            m = g.metrics
            logger.info(
                f"  {asset:>5} | {g.fold:>4} | {m.return_pct:>+7.1f}% | {m.sharpe:>7.3f} | "
                f"{m.max_drawdown:>6.1%} | {m.trade_count:>6} | {m.win_rate:>6.1%} | "
                f"{'PASS' if g.wfv_all_pass else 'FAIL':>4} | {'PASS' if g.spg_all_pass else 'FAIL':>4}"
            )

    # Decision logic
    logger.info(f"\n  --- DEPLOYMENT DECISION ---")
    for asset, results in all_results.items():
        if not results:
            logger.info(f"  {asset}: NO DATA")
            continue

        sharpes = [g.metrics.sharpe for g in results]
        max_dds = [g.metrics.max_drawdown for g in results]
        wfv_passes = sum(1 for g in results if g.wfv_all_pass)
        spg_passes = sum(1 for g in results if g.spg_all_pass)
        mean_sharpe = np.mean(sharpes)
        worst_dd = max(max_dds)

        # E2E guide decision
        if mean_sharpe >= 0.5:
            sharpe_verdict = "EXCELLENT (≥0.5)"
        elif mean_sharpe >= 0.3:
            sharpe_verdict = "ACCEPTABLE (0.3-0.5)"
        else:
            sharpe_verdict = "REJECT (<0.3) -> retrain"

        if worst_dd > 0.20:
            dd_verdict = "REJECT (>20% DD) -> do not deploy"
        elif worst_dd > 0.15:
            dd_verdict = "WARNING (>15% DD)"
        else:
            dd_verdict = "OK (≤15%)"

        deploy = mean_sharpe >= 0.3 and worst_dd <= 0.20
        mode = "ACTIVE" if spg_passes >= 2 else "SHADOW"

        logger.info(f"\n  {asset}:")
        logger.info(f"    Mean Sharpe: {mean_sharpe:.4f} -> {sharpe_verdict}")
        logger.info(f"    Worst DD:    {worst_dd:.2%} -> {dd_verdict}")
        logger.info(f"    WFV passes:  {wfv_passes}/{len(results)} folds")
        logger.info(f"    SPG passes:  {spg_passes}/{len(results)} folds")
        logger.info(f"    >>> {'DEPLOY' if deploy else 'DO NOT DEPLOY'} - DRL mode: {mode}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="E2E Step 12: Offline Validation")
    parser.add_argument("--asset", choices=ASSETS, default=None, help="Single asset (default: all)")
    args = parser.parse_args()

    manifest = load_feature_manifest()
    assets = [args.asset] if args.asset else ASSETS

    all_results: Dict[str, List[FoldGateResult]] = {}
    for asset in assets:
        try:
            results = validate_asset(asset, manifest)
            all_results[asset] = results
        except Exception as e:
            logger.error(f"  {asset} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[asset] = []

    print_summary(all_results)


if __name__ == "__main__":
    main()

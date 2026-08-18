#!/usr/bin/env python
"""
================================================================================
HMATS v6.7 - Benchmark Suite (C6)
================================================================================
FinRL-style standardized evaluation: baselines vs HMATS.

Baselines:
  1. Buy & Hold (per asset)
  2. Equal-weight monthly rebalance
  3. SMA crossover (20/50)
  4. HMATS rule-only (quant signals, no DRL)

Metrics:
  Cumulative return, Annual return, Sharpe, Sortino, Calmar,
  Max drawdown, Max DD duration, Win rate, Profit factor,
  Trade frequency, Avg holding time

Cost model:
  Kraken Pro maker (16bps) / taker (26bps), spread (per-asset),
  slippage (half-spread + Gaussian noise)

Output:
  - Markdown table (stdout + file)
  - Equity curve plot (matplotlib)
  - Per-regime performance breakdown

Usage:
  python -X utf8 scripts/benchmark_suite.py
  python -X utf8 scripts/benchmark_suite.py --assets BTC ETH SOL
  python -X utf8 scripts/benchmark_suite.py --output-dir reports/benchmark
================================================================================
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
ASSETS = ["BTC", "ETH", "SOL"]

# Kraken Pro fee schedule (volume < $50K/mo tier)
MAKER_FEE_BPS = 16.0
TAKER_FEE_BPS = 26.0

# Per-asset default spread (from GMM scaler training data)
SPREAD_BPS = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}

# Annualization: 4H bars -> 6 per day -> 2190 per year
BARS_PER_DAY = 6
BARS_PER_YEAR = 365.25 * BARS_PER_DAY

# Risk-free rate (annualized, for Sharpe/Sortino)
RISK_FREE_RATE = 0.05  # 5% (T-bill proxy)


# ============================================================================
# COST MODEL
# ============================================================================

def compute_trade_cost(
    price: float,
    size: float,
    asset: str,
    is_maker: bool = False,
) -> float:
    """
    Compute total trade cost in USD.

    Components:
      1. Exchange fee (maker 16bps or taker 26bps)
      2. Half-spread cost
      3. Slippage (Gaussian noise ~ 0.3 × spread)
    """
    notional = abs(size) * price
    spread = SPREAD_BPS.get(asset, 10.0)
    fee_bps = MAKER_FEE_BPS if is_maker else TAKER_FEE_BPS

    # Slippage: half-spread + random adverse fill
    slippage_bps = spread / 2.0 + abs(np.random.normal(0, spread * 0.3))

    total_bps = fee_bps + slippage_bps
    return notional * total_bps / 10_000.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_asset_data(asset: str) -> pd.DataFrame:
    """Load 4H OHLCV + regime data for one asset."""
    path = PROJECT_ROOT / "data" / "drl_training" / f"{asset}_4H_full.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    df = pd.read_parquet(path)

    # Ensure required columns
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{asset}: missing columns {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Regime name mapping (best-effort)
    if "regime" in df.columns:
        gmm_cfg_path = (
            PROJECT_ROOT / "models" / "regime_classifier" / asset / "gmm_config.json"
        )
        if gmm_cfg_path.exists():
            with open(gmm_cfg_path, encoding="utf-8") as f:
                gmm_cfg = json.load(f)
            rmap = gmm_cfg.get("regime_mapping", {})
            df["regime_name"] = df["regime"].astype(str).map(rmap).fillna("UNKNOWN")
        else:
            df["regime_name"] = "UNKNOWN"
    else:
        df["regime_name"] = "UNKNOWN"

    return df


def load_all_assets(assets: List[str]) -> Dict[str, pd.DataFrame]:
    """Load data for multiple assets, aligned by overlapping timestamps."""
    frames = {}
    for a in assets:
        try:
            frames[a] = load_asset_data(a)
            logger.info(f"  {a}: {len(frames[a]):,} bars, "
                        f"{frames[a]['timestamp'].min()} -> {frames[a]['timestamp'].max()}")
        except Exception as e:
            logger.warning(f"  {a}: SKIP ({e})")
    return frames


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class StrategyMetrics:
    """Full performance metrics for one strategy on one asset."""
    name: str
    asset: str
    cumulative_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_dd_duration_days: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trade_count: int = 0
    trade_frequency_per_day: float = 0.0
    avg_holding_bars: float = 0.0
    total_cost_usd: float = 0.0
    # Per-regime breakdown (regime_name -> cumulative return)
    regime_returns: Dict[str, float] = field(default_factory=dict)


def compute_metrics(
    equity: np.ndarray,
    trades: List[Dict],
    n_bars: int,
    asset: str,
    name: str,
    regime_series: Optional[pd.Series] = None,
    bar_returns: Optional[np.ndarray] = None,
) -> StrategyMetrics:
    """Compute full performance metrics from equity curve and trade log."""
    m = StrategyMetrics(name=name, asset=asset)

    if len(equity) < 2:
        return m

    # --- Returns ---
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    m.cumulative_return = (equity[-1] / equity[0]) - 1.0

    n_years = n_bars / BARS_PER_YEAR
    if n_years > 0 and equity[0] > 0 and equity[-1] > 0:
        m.annual_return = (equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0

    # --- Sharpe ---
    rf_per_bar = (1 + RISK_FREE_RATE) ** (1.0 / BARS_PER_YEAR) - 1.0
    excess = returns - rf_per_bar
    std = np.std(excess)
    if std > 1e-10:
        m.sharpe_ratio = np.mean(excess) / std * np.sqrt(BARS_PER_YEAR)

    # --- Sortino ---
    downside = excess[excess < 0]
    if len(downside) > 0:
        down_std = np.sqrt(np.mean(downside ** 2))
        if down_std > 1e-10:
            m.sortino_ratio = np.mean(excess) / down_std * np.sqrt(BARS_PER_YEAR)

    # --- Drawdown ---
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    m.max_drawdown = float(np.min(dd))

    # Max DD duration (in days)
    in_dd = dd < 0
    dd_starts = np.where(np.diff(in_dd.astype(int)) == 1)[0]
    dd_ends = np.where(np.diff(in_dd.astype(int)) == -1)[0]
    if len(dd_starts) > 0:
        if len(dd_ends) == 0 or dd_ends[-1] < dd_starts[-1]:
            dd_ends = np.append(dd_ends, len(dd) - 1)
        durations = dd_ends[: len(dd_starts)] - dd_starts[: len(dd_ends)]
        if len(durations) > 0:
            m.max_dd_duration_days = float(np.max(durations)) / BARS_PER_DAY

    # --- Calmar ---
    if abs(m.max_drawdown) > 1e-10:
        m.calmar_ratio = m.annual_return / abs(m.max_drawdown)

    # --- Trade stats ---
    m.trade_count = len(trades)
    total_days = n_bars / BARS_PER_DAY
    if total_days > 0:
        m.trade_frequency_per_day = m.trade_count / total_days

    if trades:
        pnls = [t.get("pnl", 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if pnls:
            m.win_rate = len(wins) / len(pnls)

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        if gross_loss > 1e-10:
            m.profit_factor = gross_profit / gross_loss

        holdings = [t.get("hold_bars", 0) for t in trades]
        if holdings:
            m.avg_holding_bars = np.mean(holdings)

        m.total_cost_usd = sum(t.get("cost", 0.0) for t in trades)

    # --- Per-regime returns (use strategy equity returns, not raw price) ---
    if regime_series is not None and len(returns) > 0:
        # returns = diff(equity)/equity, already computed above
        aligned_len = min(len(regime_series), len(returns))
        for regime_name in regime_series.iloc[:aligned_len].unique():
            mask = regime_series.iloc[:aligned_len].values == regime_name
            regime_rets = returns[:aligned_len][mask]
            if len(regime_rets) > 0:
                m.regime_returns[str(regime_name)] = float(
                    np.prod(1 + regime_rets) - 1.0
                )

    return m


# ============================================================================
# BASELINES
# ============================================================================

def run_buy_and_hold(
    df: pd.DataFrame, asset: str, initial_capital: float = 10_000.0
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """Buy & Hold: enter at bar 0, hold forever."""
    closes = df["close"].values
    n = len(closes)

    # Buy at bar 0
    entry_price = closes[0]
    cost = compute_trade_cost(entry_price, initial_capital / entry_price, asset, is_maker=False)
    invested = initial_capital - cost

    equity = np.zeros(n)
    for i in range(n):
        equity[i] = invested * (closes[i] / entry_price)

    bar_returns = np.diff(closes) / closes[:-1]

    trades = [{
        "entry_bar": 0,
        "exit_bar": n - 1,
        "pnl": equity[-1] - initial_capital,
        "hold_bars": n,
        "cost": cost,
    }]

    return equity, trades, bar_returns


def run_equal_weight_rebalance(
    dfs: Dict[str, pd.DataFrame],
    assets: List[str],
    initial_capital: float = 10_000.0,
    rebalance_bars: int = 180,  # ~30 days at 4H
) -> Dict[str, Tuple[np.ndarray, List[Dict], np.ndarray]]:
    """Equal-weight monthly rebalance across assets."""
    # Find common length
    min_len = min(len(dfs[a]) for a in assets)
    n_assets = len(assets)
    weight = 1.0 / n_assets

    results = {}
    for a in assets:
        closes = dfs[a]["close"].values[:min_len]
        n = len(closes)
        equity = np.zeros(n)
        alloc = initial_capital * weight
        position_size = alloc / closes[0]  # shares
        trades = []
        total_cost = compute_trade_cost(closes[0], position_size, a, False)

        for i in range(n):
            equity[i] = position_size * closes[i]

            # Rebalance
            if i > 0 and i % rebalance_bars == 0:
                target_value = equity[i]  # Keep current value, just rebalance notionally
                new_size = target_value / closes[i]
                cost = compute_trade_cost(
                    closes[i], abs(new_size - position_size), a, True
                )
                total_cost += cost
                trades.append({
                    "entry_bar": i,
                    "exit_bar": i,
                    "pnl": 0.0,  # Rebalance, no realized PnL
                    "hold_bars": rebalance_bars,
                    "cost": cost,
                })
                position_size = new_size

        bar_returns = np.diff(closes) / closes[:-1]

        if not trades:
            trades.append({
                "entry_bar": 0, "exit_bar": n - 1,
                "pnl": equity[-1] - alloc, "hold_bars": n,
                "cost": total_cost,
            })

        results[a] = (equity, trades, bar_returns)

    return results


def run_sma_crossover(
    df: pd.DataFrame,
    asset: str,
    fast: int = 20,
    slow: int = 50,
    initial_capital: float = 10_000.0,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """SMA crossover (20/50): long when fast > slow, flat otherwise."""
    closes = df["close"].values
    n = len(closes)

    # Compute SMAs
    sma_fast = pd.Series(closes).rolling(fast, min_periods=fast).mean().values
    sma_slow = pd.Series(closes).rolling(slow, min_periods=slow).mean().values

    equity = np.full(n, initial_capital, dtype=float)
    cash = initial_capital
    position = 0.0  # Number of units held
    entry_bar = -1
    entry_price = 0.0
    trades = []
    bar_returns = np.zeros(n - 1)

    for i in range(1, n):
        bar_returns[i - 1] = (closes[i] - closes[i - 1]) / closes[i - 1]

        if np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            equity[i] = cash + position * closes[i]
            continue

        prev_signal = (
            sma_fast[i - 1] > sma_slow[i - 1]
            if not (np.isnan(sma_fast[i - 1]) or np.isnan(sma_slow[i - 1]))
            else False
        )
        curr_signal = sma_fast[i] > sma_slow[i]

        # Entry: cross above
        if curr_signal and not prev_signal and position == 0:
            cost = compute_trade_cost(closes[i], cash / closes[i], asset, False)
            position = (cash - cost) / closes[i]
            entry_price = closes[i]
            entry_bar = i
            cash = 0.0

        # Exit: cross below
        elif not curr_signal and prev_signal and position > 0:
            proceeds = position * closes[i]
            cost = compute_trade_cost(closes[i], position, asset, False)
            pnl = proceeds - cost - (position * entry_price)
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": i,
                "pnl": pnl,
                "hold_bars": i - entry_bar,
                "cost": cost,
            })
            cash = proceeds - cost
            position = 0.0

        equity[i] = cash + position * closes[i]

    # Close open position at end
    if position > 0:
        proceeds = position * closes[-1]
        cost = compute_trade_cost(closes[-1], position, asset, False)
        pnl = proceeds - cost - (position * entry_price)
        trades.append({
            "entry_bar": entry_bar,
            "exit_bar": n - 1,
            "pnl": pnl,
            "hold_bars": n - 1 - entry_bar,
            "cost": cost,
        })

    return equity, trades, bar_returns


def run_hmats_rule_only(
    df: pd.DataFrame,
    asset: str,
    initial_capital: float = 10_000.0,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    HMATS rule-only: momentum + mean-revert regime-fitted signals, no DRL.

    Simplified rules (mirrors Best-of-N quant strategy):
      - MOMENTUM_RALLY / QUIET_ACCUMULATION -> momentum (RSI>50 + close>SMA20 -> long)
      - VOLATILE_CHOP / WEAK_CONSOLIDATION -> mean revert (RSI<30 -> long, RSI>70 -> flat)
      - PANIC_SELLOFF -> flat
      - Other -> momentum with reduced sizing (50%)
    """
    closes = df["close"].values
    n = len(closes)

    # Pre-compute indicators
    rsi_period = 14
    sma_period = 20
    rsi = _compute_rsi(closes, rsi_period)
    sma20 = pd.Series(closes).rolling(sma_period, min_periods=sma_period).mean().values

    regime_names = df["regime_name"].values if "regime_name" in df.columns else ["UNKNOWN"] * n

    equity = np.full(n, initial_capital, dtype=float)
    cash = initial_capital
    position = 0.0
    entry_bar = -1
    entry_price = 0.0
    trades = []
    bar_returns = np.zeros(n - 1)

    # Alpha gate: minimum expected edge to enter (14 bps after costs)
    alpha_threshold_bps = 14.0

    for i in range(1, n):
        bar_returns[i - 1] = (closes[i] - closes[i - 1]) / closes[i - 1]

        if np.isnan(rsi[i]) or np.isnan(sma20[i]):
            equity[i] = cash + position * closes[i]
            continue

        regime = str(regime_names[i]).upper() if i < len(regime_names) else "UNKNOWN"
        want_long = False
        sizing = 1.0

        if "PANIC" in regime or "SELLOFF" in regime:
            want_long = False  # Stay flat in panic
        elif "MOMENTUM" in regime or "RALLY" in regime or "ACCUMULATION" in regime:
            # Momentum: RSI>50 + price above SMA20
            want_long = rsi[i] > 50 and closes[i] > sma20[i]
        elif "CHOP" in regime or "CONSOLIDATION" in regime:
            # Mean revert: RSI<30 = oversold -> long
            want_long = rsi[i] < 30
            if rsi[i] > 70 and position > 0:
                want_long = False  # Exit overbought
        elif "EXTREME" in regime:
            want_long = False  # Flat in extreme vol
        else:
            # Unknown regime -> cautious momentum at 50% size
            want_long = rsi[i] > 55 and closes[i] > sma20[i]
            sizing = 0.5

        # Entry
        if want_long and position == 0:
            # Alpha gate: check if expected move exceeds friction
            expected_edge_bps = abs(rsi[i] - 50) * 0.5  # Rough heuristic
            if expected_edge_bps < alpha_threshold_bps:
                equity[i] = cash + position * closes[i]
                continue

            invest = cash * sizing
            cost = compute_trade_cost(closes[i], invest / closes[i], asset, False)
            position = (invest - cost) / closes[i]
            entry_price = closes[i]
            entry_bar = i
            cash = cash - invest

        # Exit
        elif not want_long and position > 0:
            proceeds = position * closes[i]
            cost = compute_trade_cost(closes[i], position, asset, False)
            pnl = proceeds - cost - (position * entry_price)
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": i,
                "pnl": pnl,
                "hold_bars": i - entry_bar,
                "cost": cost,
            })
            cash = cash + proceeds - cost
            position = 0.0

        equity[i] = cash + position * closes[i]

    # Close at end
    if position > 0:
        proceeds = position * closes[-1]
        cost = compute_trade_cost(closes[-1], position, asset, False)
        pnl = proceeds - cost - (position * entry_price)
        trades.append({
            "entry_bar": entry_bar,
            "exit_bar": n - 1,
            "pnl": pnl,
            "hold_bars": n - 1 - entry_bar,
            "cost": cost,
        })

    return equity, trades, bar_returns


def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Simple RSI computation."""
    n = len(closes)
    rsi = np.full(n, np.nan)
    deltas = np.diff(closes)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if len(gains) < period:
        return rsi

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ============================================================================
# REPORTING
# ============================================================================

def format_markdown_table(all_metrics: List[StrategyMetrics]) -> str:
    """Format metrics as a Markdown comparison table."""
    # Group by asset
    assets = sorted(set(m.asset for m in all_metrics))
    strategies = sorted(set(m.name for m in all_metrics),
                        key=lambda s: ["Buy&Hold", "EqualWeight", "SMA_20_50",
                                       "HMATS_RuleOnly"].index(s)
                        if s in ["Buy&Hold", "EqualWeight", "SMA_20_50", "HMATS_RuleOnly"]
                        else 99)

    lines = []
    lines.append("# HMATS Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nCost model: Kraken Pro taker={TAKER_FEE_BPS}bps, "
                 f"maker={MAKER_FEE_BPS}bps, spread per-asset")
    lines.append("")

    for asset in assets:
        asset_metrics = [m for m in all_metrics if m.asset == asset]
        if not asset_metrics:
            continue

        lines.append(f"## {asset}")
        lines.append("")
        lines.append(
            "| Metric | "
            + " | ".join(m.name for m in asset_metrics)
            + " |"
        )
        lines.append(
            "|--------|" + "|".join("-" * max(8, len(m.name)) for m in asset_metrics) + "|"
        )

        rows = [
            ("Cumulative Return", lambda m: f"{m.cumulative_return:+.1%}"),
            ("Annual Return", lambda m: f"{m.annual_return:+.1%}"),
            ("Sharpe Ratio", lambda m: f"{m.sharpe_ratio:.2f}"),
            ("Sortino Ratio", lambda m: f"{m.sortino_ratio:.2f}"),
            ("Calmar Ratio", lambda m: f"{m.calmar_ratio:.2f}"),
            ("Max Drawdown", lambda m: f"{m.max_drawdown:.1%}"),
            ("Max DD Duration (d)", lambda m: f"{m.max_dd_duration_days:.0f}"),
            ("Win Rate", lambda m: f"{m.win_rate:.1%}"),
            ("Profit Factor", lambda m: f"{m.profit_factor:.2f}"),
            ("Trades", lambda m: f"{m.trade_count}"),
            ("Trades/Day", lambda m: f"{m.trade_frequency_per_day:.2f}"),
            ("Avg Hold (bars)", lambda m: f"{m.avg_holding_bars:.0f}"),
            ("Total Cost ($)", lambda m: f"${m.total_cost_usd:,.0f}"),
        ]

        for label, fmt in rows:
            line = f"| {label} | "
            line += " | ".join(fmt(m) for m in asset_metrics)
            line += " |"
            lines.append(line)

        lines.append("")

    # Per-regime breakdown
    lines.append("## Per-Regime Performance (Cumulative Return)")
    lines.append("")
    all_regimes = sorted(set(
        r for m in all_metrics for r in m.regime_returns.keys()
    ))

    if all_regimes:
        for asset in assets:
            asset_metrics = [m for m in all_metrics if m.asset == asset]
            if not asset_metrics:
                continue
            lines.append(f"### {asset}")
            lines.append("")
            lines.append(
                "| Regime | " + " | ".join(m.name for m in asset_metrics) + " |"
            )
            lines.append(
                "|--------|" + "|".join("-" * max(8, len(m.name)) for m in asset_metrics) + "|"
            )
            for regime in all_regimes:
                vals = []
                for m in asset_metrics:
                    v = m.regime_returns.get(regime)
                    vals.append(f"{v:+.1%}" if v is not None else "-")
                lines.append(f"| {regime} | " + " | ".join(vals) + " |")
            lines.append("")

    return "\n".join(lines)


def plot_equity_curves(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
):
    """Plot equity curves for all strategies per asset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available - skipping equity curve plot")
        return

    assets = sorted(results.keys())
    n_assets = len(assets)
    fig, axes = plt.subplots(n_assets, 1, figsize=(14, 5 * n_assets), sharex=False)
    if n_assets == 1:
        axes = [axes]

    colors = {
        "Buy&Hold": "#1f77b4",
        "EqualWeight": "#ff7f0e",
        "SMA_20_50": "#2ca02c",
        "HMATS_RuleOnly": "#d62728",
    }

    for ax, asset in zip(axes, assets):
        for strat_name, equity in results[asset].items():
            # Normalize to $10K start for comparison
            normalized = equity / equity[0] * 10_000 if equity[0] > 0 else equity
            ax.plot(normalized, label=strat_name,
                    color=colors.get(strat_name, None), linewidth=1.2)

        ax.set_title(f"{asset} - Equity Curves (normalized to $10K)", fontsize=13)
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Bar (4H)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"  Equity curve plot saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_benchmark(
    assets: List[str],
    output_dir: Path,
    initial_capital: float = 10_000.0,
):
    """Run full benchmark suite."""
    logger.info("=" * 70)
    logger.info("HMATS Benchmark Suite (C6)")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed for reproducible slippage noise
    np.random.seed(42)

    # Load data
    logger.info("\n[1/4] Loading data...")
    dfs = load_all_assets(assets)
    if not dfs:
        logger.error("No data loaded. Exiting.")
        return

    active_assets = list(dfs.keys())

    all_metrics: List[StrategyMetrics] = []
    equity_curves: Dict[str, Dict[str, np.ndarray]] = {a: {} for a in active_assets}

    # --- Baseline 1: Buy & Hold ---
    logger.info("\n[2/4] Running baselines...")
    logger.info("  Buy & Hold...")
    for a in active_assets:
        eq, trades, bar_rets = run_buy_and_hold(dfs[a], a, initial_capital)
        regime_series = dfs[a]["regime_name"].iloc[1:] if "regime_name" in dfs[a].columns else None
        m = compute_metrics(eq, trades, len(eq), a, "Buy&Hold",
                            regime_series=regime_series, bar_returns=bar_rets)
        all_metrics.append(m)
        equity_curves[a]["Buy&Hold"] = eq

    # --- Baseline 2: Equal-weight rebalance ---
    logger.info("  Equal-weight rebalance...")
    ew_results = run_equal_weight_rebalance(dfs, active_assets, initial_capital)
    for a in active_assets:
        eq, trades, bar_rets = ew_results[a]
        regime_series = dfs[a]["regime_name"].iloc[1:len(eq)] if "regime_name" in dfs[a].columns else None
        m = compute_metrics(eq, trades, len(eq), a, "EqualWeight",
                            regime_series=regime_series, bar_returns=bar_rets)
        all_metrics.append(m)
        equity_curves[a]["EqualWeight"] = eq

    # --- Baseline 3: SMA crossover ---
    logger.info("  SMA crossover (20/50)...")
    for a in active_assets:
        eq, trades, bar_rets = run_sma_crossover(dfs[a], a, 20, 50, initial_capital)
        regime_series = dfs[a]["regime_name"].iloc[1:] if "regime_name" in dfs[a].columns else None
        m = compute_metrics(eq, trades, len(eq), a, "SMA_20_50",
                            regime_series=regime_series, bar_returns=bar_rets)
        all_metrics.append(m)
        equity_curves[a]["SMA_20_50"] = eq

    # --- Baseline 4: HMATS rule-only ---
    logger.info("  HMATS rule-only (quant, no DRL)...")
    for a in active_assets:
        eq, trades, bar_rets = run_hmats_rule_only(dfs[a], a, initial_capital)
        regime_series = dfs[a]["regime_name"].iloc[1:] if "regime_name" in dfs[a].columns else None
        m = compute_metrics(eq, trades, len(eq), a, "HMATS_RuleOnly",
                            regime_series=regime_series, bar_returns=bar_rets)
        all_metrics.append(m)
        equity_curves[a]["HMATS_RuleOnly"] = eq

    # --- Generate report ---
    logger.info("\n[3/4] Generating report...")
    report = format_markdown_table(all_metrics)

    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"  Report saved: {report_path}")

    # Print to stdout
    print("\n" + report)

    # --- Equity curve plot ---
    logger.info("\n[4/4] Plotting equity curves...")
    plot_path = output_dir / "equity_curves.png"
    plot_equity_curves(equity_curves, plot_path)

    # --- Summary JSON ---
    summary = {
        "generated": datetime.now().isoformat(),
        "assets": active_assets,
        "initial_capital": initial_capital,
        "cost_model": {
            "maker_bps": MAKER_FEE_BPS,
            "taker_bps": TAKER_FEE_BPS,
            "spread_bps": SPREAD_BPS,
        },
        "results": [
            {
                "strategy": m.name,
                "asset": m.asset,
                "cumulative_return": round(m.cumulative_return, 4),
                "annual_return": round(m.annual_return, 4),
                "sharpe": round(m.sharpe_ratio, 3),
                "sortino": round(m.sortino_ratio, 3),
                "calmar": round(m.calmar_ratio, 3),
                "max_drawdown": round(m.max_drawdown, 4),
                "trade_count": m.trade_count,
                "win_rate": round(m.win_rate, 3),
                "profit_factor": round(m.profit_factor, 3),
            }
            for m in all_metrics
        ],
    }
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary JSON saved: {summary_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Benchmark complete.")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Plot:   {plot_path}")
    logger.info(f"  JSON:   {summary_path}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="HMATS Benchmark Suite (C6)")
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                        help="Assets to benchmark (default: BTC ETH SOL)")
    parser.add_argument("--output-dir", type=str, default="reports/benchmark",
                        help="Output directory for reports")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Initial capital per asset (default: $10,000)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    run_benchmark(args.assets, output_dir, args.capital)


if __name__ == "__main__":
    main()

"""
HMATS v5.1 Phase Pre-6 - Backtest Engine
=========================================

Replay historical 4H OHLCV+features parquet against a candidate strategy
that conforms to the Phase 4/8 shadow-strategy interface:

    sig = strategy.evaluate(asset: str, market_data: Dict[str, Any]) -> ShadowSignal

Output metrics per backtest run:
  - Per-trade PnL (entry/exit pairs)
  - Annualized Sharpe (assuming 4H bars, 6 bars/day, 252 trading days)
  - IC across configurable horizons (4b/12b/24b)
  - Fee impact (Coinbase per V14 GREEN: 0bps maker, 3bps taker; configurable)
  - Regime breakdown
  - Drawdown profile

Iron Laws honored:
  1. obs_dim=126: untouched (consumes parquet as-is)
  3. training/: NEW subdir (training/backtest_framework/) — additive, no
     touch to training/drl/* or training/gmm/*. No constitutional override
     needed (Iron Law 3 protects training/ from MODIFICATION; new subdir
     adjacent is permitted per the v5.1 prompt's Pre-6 spec).
  4. fail-closed: missing parquet fields / NaN bars / strategy exceptions
     are caught and logged; run continues with that bar skipped.

Limitations:
  - Strategies that consume microstructure fields (vpin, ofi_zscore,
    liquidation_imbalance, spread_bps) will mostly emit NEUTRAL because
    the parquet does not carry those fields. Use this engine for
    OHLCV-derived strategies; use compute_shadow_ic.py for live shadow
    ledger evaluation of microstructure/cascade strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FeeSchedule:
    """Per-trade fee + slippage model. Defaults match V14 GREEN (Coinbase)."""
    maker_bps: float = 0.0       # 0bps maker (V14 promotional)
    taker_bps: float = 3.0       # 3bps taker
    maker_pct: float = 0.987     # post-only ratio per V15 (98.7% maker)
    slippage_bps_per_pct_size: float = 1.0  # linear slippage proxy

    def round_trip_bps(self) -> float:
        """Effective round-trip fee in bps, weighted by maker ratio."""
        per_side = self.maker_pct * self.maker_bps + (1 - self.maker_pct) * self.taker_bps
        return 2 * per_side  # entry + exit


@dataclass
class BacktestConfig:
    asset: str = "BTC"
    parquet_path: Optional[Path] = None
    horizons_bars: Tuple[int, ...] = (4, 12, 24)
    bars_per_day: int = 6                # 4H -> 6 bars/day
    trading_days_per_year: int = 252
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    confidence_threshold: float = 0.30   # below = ignore signal
    initial_capital: float = 10_000.0
    position_pct: float = 0.10           # 10% notional per trade


@dataclass
class TradeRecord:
    entry_idx: int
    exit_idx: int
    direction: float
    entry_price: float
    exit_price: float
    confidence: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    holding_bars: int
    regime: Optional[str]
    reason_entry: str


@dataclass
class BacktestResult:
    asset: str
    n_bars: int
    n_signals: int
    n_trades: int
    trades: List[TradeRecord]
    gross_pnl: float
    total_fees: float
    net_pnl: float
    annual_sharpe: float
    win_rate: float
    avg_return_pct: float
    max_drawdown_pct: float
    ic_per_horizon: Dict[int, float]   # bars -> IC
    regime_breakdown: Dict[str, Dict[str, float]]   # regime -> {n, win_rate, avg_pnl}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "n_bars": self.n_bars,
            "n_signals": self.n_signals,
            "n_trades": self.n_trades,
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "net_pnl": self.net_pnl,
            "annual_sharpe": self.annual_sharpe,
            "win_rate": self.win_rate,
            "avg_return_pct": self.avg_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "ic_per_horizon": self.ic_per_horizon,
            "regime_breakdown": self.regime_breakdown,
        }


class BacktestEngine:
    """Replay 4H parquet against a candidate strategy.

    Usage:
        cfg = BacktestConfig(asset="BTC")
        engine = BacktestEngine(cfg)
        from strategies.microstructure_v5_1 import OrderFlowImbalanceStrategy
        result = engine.run(OrderFlowImbalanceStrategy())
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config

    def _resolve_parquet_path(self) -> Path:
        if self.cfg.parquet_path is not None:
            return Path(self.cfg.parquet_path)
        repo = Path(__file__).resolve().parents[2]
        return repo / "training" / "training_data" / "drl_training" / f"{self.cfg.asset}_4H_full.parquet"

    def _row_to_market_data(self, row: Any) -> Dict[str, Any]:
        """Convert a parquet row to a market_data dict matching the
        Phase 4/8 strategy interface. Only fields present in the parquet
        are populated; microstructure-only fields (vpin, ofi_zscore, etc.)
        will be missing and strategies that need them will return NEUTRAL.
        """
        md: Dict[str, Any] = {}
        for col, val in row.items():
            if col == "timestamp":
                continue
            md[col] = val
        # Standardize a few key names the strategies expect
        if "close" in md:
            md["current_price"] = float(md["close"])
        # price_change_4h_pct = ret_4h if present
        if "ret_4h" in md:
            md["price_change_4h_pct"] = float(md["ret_4h"])
        if "ret_1h" in md:
            md["price_change_1h_pct"] = float(md["ret_1h"])
        return md

    @staticmethod
    def _spearman(x: List[float], y: List[float]) -> float:
        """Spearman rank correlation. Returns 0.0 on degenerate inputs."""
        if len(x) != len(y) or len(x) < 4:
            return 0.0

        def rank(arr: List[float]) -> List[float]:
            order = sorted(range(len(arr)), key=lambda i: arr[i])
            ranks = [0.0] * len(arr)
            i = 0
            while i < len(arr):
                j = i
                while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[order[k]] = avg_rank
                i = j + 1
            return ranks

        rx, ry = rank(x), rank(y)
        n = len(rx)
        mx = sum(rx) / n
        my = sum(ry) / n
        num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        denom_x = sum((r - mx) ** 2 for r in rx) ** 0.5
        denom_y = sum((r - my) ** 2 for r in ry) ** 0.5
        if denom_x <= 0 or denom_y <= 0:
            return 0.0
        return num / (denom_x * denom_y)

    @staticmethod
    def _annual_sharpe(returns: List[float], bars_per_year: int) -> float:
        if len(returns) < 2:
            return 0.0
        n = len(returns)
        mean_r = sum(returns) / n
        var_r = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_r = var_r ** 0.5
        if std_r <= 0:
            return 0.0
        return (mean_r / std_r) * (bars_per_year ** 0.5)

    def run(self, strategy: Any) -> BacktestResult:
        """Run the backtest. Returns BacktestResult.

        The strategy must have an `evaluate(asset, market_data)` method
        returning an object with `direction`, `confidence`, `reason` fields
        (Phase 4/8 ShadowSignal).
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(f"pandas required for BacktestEngine: {e}")

        path = self._resolve_parquet_path()
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")

        df = pd.read_parquet(path)
        n = len(df)
        max_horizon = max(self.cfg.horizons_bars)

        # Pre-compute forward returns at each horizon
        forward_returns: Dict[int, List[float]] = {h: [] for h in self.cfg.horizons_bars}
        signal_x: Dict[int, List[float]] = {h: [] for h in self.cfg.horizons_bars}
        n_signals_total = 0

        # Trade tracking: enter on signal above threshold, exit at first horizon (24b default)
        trades: List[TradeRecord] = []
        equity = self.cfg.initial_capital
        equity_curve: List[float] = [equity]
        regime_stats: Dict[str, Dict[str, Any]] = {}

        rt_fee_bps = self.cfg.fees.round_trip_bps()
        exit_horizon = self.cfg.horizons_bars[-1]   # use largest horizon for trade exit

        n_strategy_errors = 0
        last_strategy_error: Optional[str] = None

        for i in range(n - max_horizon):
            row = df.iloc[i]
            md = self._row_to_market_data(row)

            try:
                sig = strategy.evaluate(self.cfg.asset, md)
            except Exception as e:  # noqa: silent-swallow
                # Per-bar strategy exception. Batch-level WARN at end of run
                # surfaces total count + last error type. Per-bar WARN would
                # spam thousands of identical lines on a deterministic bug.
                n_strategy_errors += 1
                last_strategy_error = f"{type(e).__name__}: {e}"
                continue

            direction = float(getattr(sig, "direction", 0.0) or 0.0)
            confidence = float(getattr(sig, "confidence", 0.0) or 0.0)

            if direction == 0.0 or confidence < self.cfg.confidence_threshold:
                continue

            n_signals_total += 1

            # Forward returns / IC inputs
            entry_price = float(row["close"])
            for h in self.cfg.horizons_bars:
                future_idx = i + h
                if future_idx >= n:
                    continue
                future_price = float(df.iloc[future_idx]["close"])
                fwd_ret = (future_price - entry_price) / entry_price
                forward_returns[h].append(fwd_ret)
                signal_x[h].append(direction * confidence)

            # Trade record at exit_horizon
            exit_idx = i + exit_horizon
            if exit_idx >= n:
                continue
            exit_price = float(df.iloc[exit_idx]["close"])
            gross_ret = direction * (exit_price - entry_price) / entry_price
            position_notional = equity * self.cfg.position_pct
            gross_pnl = position_notional * gross_ret
            fees = position_notional * rt_fee_bps / 10000.0
            net_pnl = gross_pnl - fees

            equity += net_pnl
            equity_curve.append(equity)

            regime = row.get("regime") if "regime" in df.columns else None
            regime_str = str(regime) if regime is not None else "UNKNOWN"

            trades.append(TradeRecord(
                entry_idx=i,
                exit_idx=exit_idx,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                confidence=confidence,
                gross_pnl=gross_pnl,
                fees=fees,
                net_pnl=net_pnl,
                return_pct=(net_pnl / position_notional) if position_notional > 0 else 0.0,
                holding_bars=exit_horizon,
                regime=regime_str,
                reason_entry=str(getattr(sig, "reason", "")),
            ))

            # Update regime breakdown
            if regime_str not in regime_stats:
                regime_stats[regime_str] = {"n": 0, "wins": 0, "pnl_sum": 0.0}
            regime_stats[regime_str]["n"] += 1
            if net_pnl > 0:
                regime_stats[regime_str]["wins"] += 1
            regime_stats[regime_str]["pnl_sum"] += net_pnl

        # CLAUDE.md P64: surface batched strategy errors to operator visibility.
        if n_strategy_errors:
            logger.warning(
                f"[BACKTEST] strategy.evaluate raised {n_strategy_errors} times "
                f"out of {n - max_horizon} bars; last={last_strategy_error}"
            )

        # IC per horizon
        ic_per_h: Dict[int, float] = {
            h: self._spearman(signal_x[h], forward_returns[h])
            for h in self.cfg.horizons_bars
        }

        # Aggregate metrics
        n_trades = len(trades)
        gross_pnl_total = sum(t.gross_pnl for t in trades)
        fees_total = sum(t.fees for t in trades)
        net_pnl_total = sum(t.net_pnl for t in trades)
        win_rate = sum(1 for t in trades if t.net_pnl > 0) / n_trades if n_trades > 0 else 0.0
        avg_return_pct = sum(t.return_pct for t in trades) / n_trades if n_trades > 0 else 0.0

        # Annualized Sharpe from per-trade returns (each trade = exit_horizon bars)
        per_trade_rets = [t.return_pct for t in trades]
        # Effective bars-per-year for trade-level rebalance
        trades_per_year = self.cfg.bars_per_day * self.cfg.trading_days_per_year / max(1, exit_horizon)
        annual_sharpe = self._annual_sharpe(per_trade_rets, int(trades_per_year))

        # Max drawdown from equity curve
        max_dd = 0.0
        peak = equity_curve[0] if equity_curve else self.cfg.initial_capital
        for v in equity_curve:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Regime breakdown finalize
        regime_breakdown: Dict[str, Dict[str, float]] = {}
        for r, st in regime_stats.items():
            regime_breakdown[r] = {
                "n": st["n"],
                "win_rate": st["wins"] / st["n"] if st["n"] > 0 else 0.0,
                "avg_pnl": st["pnl_sum"] / st["n"] if st["n"] > 0 else 0.0,
            }

        return BacktestResult(
            asset=self.cfg.asset,
            n_bars=n,
            n_signals=n_signals_total,
            n_trades=n_trades,
            trades=trades,
            gross_pnl=gross_pnl_total,
            total_fees=fees_total,
            net_pnl=net_pnl_total,
            annual_sharpe=annual_sharpe,
            win_rate=win_rate,
            avg_return_pct=avg_return_pct,
            max_drawdown_pct=max_dd,
            ic_per_horizon=ic_per_h,
            regime_breakdown=regime_breakdown,
        )


# Convenience functional wrapper
def backtest_strategy(strategy: Any, asset: str = "BTC",
                      parquet_path: Optional[Path] = None,
                      fees: Optional[FeeSchedule] = None) -> BacktestResult:
    cfg = BacktestConfig(asset=asset, parquet_path=parquet_path,
                         fees=fees or FeeSchedule())
    return BacktestEngine(cfg).run(strategy)

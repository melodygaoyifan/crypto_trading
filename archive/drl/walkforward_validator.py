"""
================================================================================
HMATS v6 - Walk-Forward Validator
================================================================================
Out-of-sample validation for DRL/strategy promotion decisions.

Implements anchored walk-forward splits with per-fold metric calculation
to prevent overfitting before model promotion.

Pipeline position: training -> validation gate (before promotion_gate.py).
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""

    # Number of folds
    n_folds: int = 3

    # Minimum training samples per fold
    min_train_samples: int = 5000

    # Validation ratio (fraction of each fold used for OOS test)
    val_ratio: float = 0.15

    # Gap between train and val (bars) to prevent leakage
    gap_bars: int = 42

    # Metric thresholds for promotion
    min_sharpe: float = 0.5
    min_win_rate: float = 0.45
    max_drawdown_pct: float = 25.0

    # Require at least this many folds to pass
    min_passing_folds: int = 2


# =============================================================================
# METRIC CALCULATOR
# =============================================================================

class MetricCalculator:
    """
    Calculate strategy performance metrics from a returns series.

    Stateless - all methods are pure functions.
    """

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray, risk_free: float = 0.0, annualize: int = 252
    ) -> float:
        """Annualised Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / annualize
        std = np.std(excess, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(annualize))

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray, risk_free: float = 0.0, annualize: int = 252
    ) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / annualize
        downside = excess[excess < 0]
        if len(downside) < 1:
            return float("inf") if np.mean(excess) > 0 else 0.0
        dd = np.std(downside, ddof=1)
        if dd < 1e-10:
            return 0.0
        return float(np.mean(excess) / dd * np.sqrt(annualize))

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Maximum drawdown as a positive percentage."""
        if len(returns) < 1:
            return 0.0
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns) * 100.0)

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Fraction of positive returns."""
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns > 0))

    @staticmethod
    def calmar_ratio(returns: np.ndarray, annualize: int = 252) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if len(returns) < 2:
            return 0.0
        ann_return = float(np.mean(returns) * annualize)
        mdd = MetricCalculator.max_drawdown(returns)
        if mdd < 0.01:
            return 0.0
        return ann_return / (mdd / 100.0)


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

@dataclass
class FoldResult:
    """Result of a single walk-forward fold."""
    fold_idx: int
    train_size: int
    val_size: int
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float
    calmar: float
    passed: bool
    fail_reasons: List[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregate result of walk-forward validation."""
    folds: List[FoldResult]
    passed: bool
    mean_sharpe: float
    mean_drawdown: float
    passing_folds: int
    total_folds: int
    fail_reasons: List[str] = field(default_factory=list)


class WalkForwardValidator:
    """
    Anchored walk-forward cross-validator for strategy returns.

    Usage:
        validator = WalkForwardValidator()
        result = validator.validate(returns_array)
        if result.passed:
            promote_model()
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.calc = MetricCalculator()

    def validate(
        self,
        returns: np.ndarray,
        regime_labels: Optional[np.ndarray] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on a returns series.

        Args:
            returns: 1-D array of per-bar returns.
            regime_labels: Optional regime labels aligned with returns.

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        n = len(returns)
        cfg = self.config
        folds: List[FoldResult] = []
        fail_reasons: List[str] = []

        if n < cfg.min_train_samples + cfg.gap_bars + 100:
            return WalkForwardResult(
                folds=[],
                passed=False,
                mean_sharpe=0.0,
                mean_drawdown=0.0,
                passing_folds=0,
                total_folds=0,
                fail_reasons=[f"insufficient data: {n} bars"],
            )

        fold_size = n // cfg.n_folds

        for i in range(cfg.n_folds):
            # Anchored: train always starts at 0
            train_end = (i + 1) * fold_size - cfg.gap_bars
            val_start = train_end + cfg.gap_bars
            val_end = min(val_start + int(fold_size * cfg.val_ratio), n)

            if train_end < cfg.min_train_samples or val_end - val_start < 50:
                continue

            val_returns = returns[val_start:val_end]

            sharpe = self.calc.sharpe_ratio(val_returns)
            sortino = self.calc.sortino_ratio(val_returns)
            mdd = self.calc.max_drawdown(val_returns)
            wr = self.calc.win_rate(val_returns)
            calmar = self.calc.calmar_ratio(val_returns)

            fold_fails = []
            if sharpe < cfg.min_sharpe:
                fold_fails.append(f"sharpe {sharpe:.2f} < {cfg.min_sharpe}")
            if wr < cfg.min_win_rate:
                fold_fails.append(f"win_rate {wr:.2%} < {cfg.min_win_rate:.2%}")
            if mdd > cfg.max_drawdown_pct:
                fold_fails.append(f"drawdown {mdd:.1f}% > {cfg.max_drawdown_pct}%")

            folds.append(
                FoldResult(
                    fold_idx=i,
                    train_size=train_end,
                    val_size=val_end - val_start,
                    sharpe=sharpe,
                    sortino=sortino,
                    max_drawdown_pct=mdd,
                    win_rate=wr,
                    calmar=calmar,
                    passed=len(fold_fails) == 0,
                    fail_reasons=fold_fails,
                )
            )

        passing = sum(1 for f in folds if f.passed)
        mean_sharpe = float(np.mean([f.sharpe for f in folds])) if folds else 0.0
        mean_dd = float(np.mean([f.max_drawdown_pct for f in folds])) if folds else 0.0

        if passing < cfg.min_passing_folds:
            fail_reasons.append(
                f"only {passing}/{len(folds)} folds passed "
                f"(need {cfg.min_passing_folds})"
            )

        return WalkForwardResult(
            folds=folds,
            passed=passing >= cfg.min_passing_folds,
            mean_sharpe=mean_sharpe,
            mean_drawdown=mean_dd,
            passing_folds=passing,
            total_folds=len(folds),
            fail_reasons=fail_reasons,
        )

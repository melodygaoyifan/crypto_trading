"""
================================================================================
HMATS v6 - Statistical Promotion Gate
================================================================================
Statistical tests for strategy promotion decisions.  Evaluates a candidate
strategy's returns against minimum thresholds for Sharpe, win rate, drawdown,
and regime coverage before allowing promotion to paper/live.

Pipeline position: training -> promotion gate (after walk-forward validation).
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Calculated performance metrics for a strategy."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    total_return_pct: float = 0.0
    regime_coverage: Dict[str, int] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    """Result of the statistical promotion gate evaluation."""
    approved: bool
    strategy_id: str = ""
    tests_passed: List[str] = field(default_factory=list)
    tests_failed: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    metrics: Optional[PerformanceMetrics] = None


# =============================================================================
# STATISTICAL PROMOTION GATE
# =============================================================================

class StatisticalPromotionGate:
    """
    Evaluates strategy returns and decides whether to promote.

    Default thresholds are conservative - designed to reject underperforming
    strategies rather than approve borderline ones (fail-closed).

    Args:
        min_samples: Minimum number of return observations.
        min_sharpe: Minimum annualised Sharpe ratio.
        min_win_rate: Minimum fraction of positive returns.
        max_drawdown_pct: Maximum allowable drawdown (%).
        min_regimes: Minimum distinct regimes that must be represented.
        min_regime_pct: Minimum fraction of samples in any required regime.
    """

    def __init__(
        self,
        min_samples: int = 100,
        min_sharpe: float = 0.5,
        min_win_rate: float = 0.45,
        max_drawdown_pct: float = 25.0,
        min_regimes: int = 3,
        min_regime_pct: float = 0.05,
    ):
        self.min_samples = min_samples
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.max_drawdown_pct = max_drawdown_pct
        self.min_regimes = min_regimes
        self.min_regime_pct = min_regime_pct

    def evaluate(
        self,
        strategy_id: str,
        returns: np.ndarray,
        regime_labels: Optional[List[str]] = None,
    ) -> PromotionDecision:
        """
        Evaluate a strategy for promotion.

        Args:
            strategy_id: Unique identifier for the strategy.
            returns: 1-D array of per-bar returns.
            regime_labels: Optional list of regime labels (same length as returns).

        Returns:
            PromotionDecision with pass/fail details and rejection reasons.
        """
        passed: List[str] = []
        failed: List[str] = []
        reasons: List[str] = []

        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)

        # --- Sample size ---
        if n >= self.min_samples:
            passed.append("sample_size")
        else:
            failed.append("sample_size")
            reasons.append(
                f"insufficient samples: {n} < {self.min_samples}"
            )

        # --- Sharpe ratio ---
        sharpe = self._sharpe(returns)
        if sharpe >= self.min_sharpe:
            passed.append("sharpe_ratio")
        else:
            failed.append("sharpe_ratio")
            reasons.append(f"sharpe {sharpe:.3f} < {self.min_sharpe}")

        # --- Win rate ---
        win_rate = float(np.mean(returns > 0)) if n > 0 else 0.0
        if win_rate >= self.min_win_rate:
            passed.append("win_rate")
        else:
            failed.append("win_rate")
            reasons.append(f"win_rate {win_rate:.3f} < {self.min_win_rate}")

        # --- Max drawdown ---
        mdd = self._max_drawdown(returns)
        if mdd <= self.max_drawdown_pct:
            passed.append("max_drawdown")
        else:
            failed.append("max_drawdown")
            reasons.append(f"drawdown {mdd:.1f}% > {self.max_drawdown_pct}%")

        # --- Regime coverage ---
        regime_coverage: Dict[str, int] = {}
        if regime_labels is not None and len(regime_labels) == n:
            for label in regime_labels:
                regime_coverage[label] = regime_coverage.get(label, 0) + 1

            unique_regimes = len(regime_coverage)
            if unique_regimes >= self.min_regimes:
                passed.append("regime_coverage")
            else:
                failed.append("regime_coverage")
                reasons.append(
                    f"only {unique_regimes} regimes "
                    f"(need {self.min_regimes}): {list(regime_coverage.keys())}"
                )

            # Check minimum representation per regime
            for regime, count in regime_coverage.items():
                pct = count / n
                if pct < self.min_regime_pct:
                    if "regime_balance" not in failed:
                        failed.append("regime_balance")
                    reasons.append(
                        f"regime '{regime}' has {pct:.1%} "
                        f"(< {self.min_regime_pct:.1%})"
                    )

        # --- Build metrics ---
        metrics = PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=self._sortino(returns),
            max_drawdown_pct=mdd,
            win_rate=win_rate,
            profit_factor=self._profit_factor(returns),
            n_trades=n,
            total_return_pct=float(np.sum(returns) * 100) if n > 0 else 0.0,
            regime_coverage=regime_coverage,
        )

        approved = len(failed) == 0

        return PromotionDecision(
            approved=approved,
            strategy_id=strategy_id,
            tests_passed=passed,
            tests_failed=failed,
            rejection_reasons=reasons,
            metrics=metrics,
        )

    # -----------------------------------------------------------------
    # METRIC HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: np.ndarray, annualize: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        std = np.std(returns, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(annualize))

    @staticmethod
    def _sortino(returns: np.ndarray, annualize: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) < 1:
            return 0.0
        dd = np.std(downside, ddof=1)
        if dd < 1e-10:
            return 0.0
        return float(np.mean(returns) / dd * np.sqrt(annualize))

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if len(returns) < 1:
            return 0.0
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        dd = (running_max - cumulative) / running_max
        return float(np.max(dd) * 100.0)

    @staticmethod
    def _profit_factor(returns: np.ndarray) -> float:
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        total_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
        total_gain = float(np.sum(gains)) if len(gains) > 0 else 0.0
        if total_loss < 1e-10:
            return float("inf") if total_gain > 0 else 0.0
        return total_gain / total_loss

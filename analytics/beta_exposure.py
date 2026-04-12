"""
Beta Exposure Analytics — non-invasive observer for alpha/beta decomposition.

Does NOT modify any trading decisions. Only measures and reports.

Implements:
  - Rolling beta (strategy vs benchmark)
  - Rolling correlation
  - Upside/downside capture ratios
  - Residual alpha attribution
  - Regime-bucketed beta report

Usage:
    from analytics.beta_exposure import BetaExposureAnalyzer
    analyzer = BetaExposureAnalyzer()
    report = analyzer.full_report(strategy_returns, benchmark_returns, regimes)
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BetaReport:
    """Complete beta/alpha decomposition report."""
    # Overall
    overall_beta: float = 0.0
    overall_correlation: float = 0.0
    overall_alpha_annualized: float = 0.0
    # Upside/Downside
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    upside_sample_count: int = 0
    downside_sample_count: int = 0
    # Rolling (latest values)
    rolling_beta_latest: float = 0.0
    rolling_corr_latest: float = 0.0
    rolling_window: int = 0
    # Regime breakdown
    regime_betas: Dict[str, float] = field(default_factory=dict)
    regime_alphas: Dict[str, float] = field(default_factory=dict)
    regime_counts: Dict[str, int] = field(default_factory=dict)
    # Benchmark info
    benchmark_name: str = ""
    benchmark_return: float = 0.0
    strategy_return: float = 0.0
    # Risk
    beta_risk_level: str = "UNKNOWN"  # LOW / MEDIUM / HIGH / EXTREME


class BetaExposureAnalyzer:
    """Non-invasive beta/alpha decomposition analyzer."""

    def __init__(self, annualization_factor: float = 2190.0):
        """
        Args:
            annualization_factor: bars per year. 4H bars = 6*365 = 2190.
        """
        self.ann_factor = annualization_factor

    def compute_rolling_beta(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        window: int = 126,
    ) -> np.ndarray:
        """Rolling beta = cov(strat, bench) / var(bench) over window."""
        n = len(strategy_returns)
        betas = np.full(n, np.nan)
        for i in range(window, n):
            s = strategy_returns[i - window:i]
            b = benchmark_returns[i - window:i]
            var_b = np.var(b)
            if var_b > 1e-15:
                betas[i] = np.cov(s, b)[0, 1] / var_b
            else:
                betas[i] = 0.0
        return betas

    def compute_rolling_correlation(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        window: int = 126,
    ) -> np.ndarray:
        """Rolling Pearson correlation over window."""
        n = len(strategy_returns)
        corrs = np.full(n, np.nan)
        for i in range(window, n):
            s = strategy_returns[i - window:i]
            b = benchmark_returns[i - window:i]
            std_s = np.std(s)
            std_b = np.std(b)
            if std_s > 1e-15 and std_b > 1e-15:
                corrs[i] = np.corrcoef(s, b)[0, 1]
            else:
                corrs[i] = 0.0
        return corrs

    def compute_upside_downside_capture(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Tuple[float, float, int, int]:
        """
        Upside capture = mean(strat | bench > 0) / mean(bench | bench > 0)
        Downside capture = mean(strat | bench < 0) / mean(bench | bench < 0)
        """
        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0
        up_count = int(np.sum(up_mask))
        down_count = int(np.sum(down_mask))

        if up_count > 0:
            up_capture = float(np.mean(strategy_returns[up_mask]) / np.mean(benchmark_returns[up_mask]))
        else:
            up_capture = 0.0

        if down_count > 0:
            down_capture = float(np.mean(strategy_returns[down_mask]) / np.mean(benchmark_returns[down_mask]))
        else:
            down_capture = 0.0

        return up_capture, down_capture, up_count, down_count

    def compute_residual_alpha(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        beta: Optional[float] = None,
    ) -> np.ndarray:
        """Residual alpha = strategy_return - beta * benchmark_return per bar."""
        if beta is None:
            var_b = np.var(benchmark_returns)
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / var_b if var_b > 1e-15 else 0.0
        return strategy_returns - beta * benchmark_returns

    def compute_regime_bucket_report(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        regimes: np.ndarray,
    ) -> Dict[str, Dict]:
        """Per-regime beta, alpha, count."""
        unique_regimes = sorted(set(str(r) for r in regimes))
        result = {}
        for regime in unique_regimes:
            mask = np.array([str(r) == regime for r in regimes])
            count = int(np.sum(mask))
            if count < 5:
                result[regime] = {"beta": 0.0, "alpha_bps": 0.0, "count": count, "note": "insufficient"}
                continue
            s = strategy_returns[mask]
            b = benchmark_returns[mask]
            var_b = np.var(b)
            beta = float(np.cov(s, b)[0, 1] / var_b) if var_b > 1e-15 else 0.0
            residual = s - beta * b
            alpha_bps = float(np.mean(residual) * 10000)
            result[regime] = {"beta": round(beta, 3), "alpha_bps": round(alpha_bps, 2), "count": count}
        return result

    def full_report(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        regimes: Optional[np.ndarray] = None,
        benchmark_name: str = "BTC Buy&Hold",
        window: int = 126,
    ) -> BetaReport:
        """Generate complete beta/alpha decomposition report."""
        report = BetaReport()
        report.benchmark_name = benchmark_name
        report.rolling_window = window

        n = min(len(strategy_returns), len(benchmark_returns))
        if n < 10:
            report.beta_risk_level = "INSUFFICIENT_DATA"
            return report

        s = strategy_returns[:n]
        b = benchmark_returns[:n]

        # Overall beta
        var_b = np.var(b)
        report.overall_beta = float(np.cov(s, b)[0, 1] / var_b) if var_b > 1e-15 else 0.0
        # Overall correlation
        std_s, std_b = np.std(s), np.std(b)
        report.overall_correlation = float(np.corrcoef(s, b)[0, 1]) if std_s > 1e-15 and std_b > 1e-15 else 0.0
        # Overall alpha (annualized)
        residual = s - report.overall_beta * b
        report.overall_alpha_annualized = float(np.mean(residual) * self.ann_factor * 100)  # in %

        # Returns
        report.strategy_return = float((np.prod(1 + s) - 1) * 100)
        report.benchmark_return = float((np.prod(1 + b) - 1) * 100)

        # Upside/Downside capture
        up, down, up_n, down_n = self.compute_upside_downside_capture(s, b)
        report.upside_capture = round(up, 3)
        report.downside_capture = round(down, 3)
        report.upside_sample_count = up_n
        report.downside_sample_count = down_n

        # Rolling
        if n >= window:
            r_beta = self.compute_rolling_beta(s, b, window)
            r_corr = self.compute_rolling_correlation(s, b, window)
            valid_beta = r_beta[np.isfinite(r_beta)]
            valid_corr = r_corr[np.isfinite(r_corr)]
            report.rolling_beta_latest = float(valid_beta[-1]) if len(valid_beta) > 0 else 0.0
            report.rolling_corr_latest = float(valid_corr[-1]) if len(valid_corr) > 0 else 0.0

        # Regime buckets
        if regimes is not None and len(regimes) >= n:
            report.regime_betas = {}
            report.regime_alphas = {}
            report.regime_counts = {}
            regime_data = self.compute_regime_bucket_report(s, b, regimes[:n])
            for regime, data in regime_data.items():
                report.regime_betas[regime] = data["beta"]
                report.regime_alphas[regime] = data["alpha_bps"]
                report.regime_counts[regime] = data["count"]

        # Risk classification
        abs_beta = abs(report.overall_beta)
        abs_corr = abs(report.overall_correlation)
        if abs_beta < 0.3 and abs_corr < 0.3:
            report.beta_risk_level = "LOW"
        elif abs_beta < 0.6 and abs_corr < 0.6:
            report.beta_risk_level = "MEDIUM"
        elif abs_beta < 1.0:
            report.beta_risk_level = "HIGH"
        else:
            report.beta_risk_level = "EXTREME"

        return report

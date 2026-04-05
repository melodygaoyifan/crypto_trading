"""
================================================================================
HMATS v6.0.0 - Monte Carlo Validation
================================================================================
Purpose: Stress test trading strategy using Monte Carlo simulation

Method:
1. Randomly shuffle trade sequence
2. Simulate equity curve
3. Repeat 1000+ times
4. Calculate confidence intervals for key metrics

SOTA Compliance:
- Provides statistical confidence in strategy performance
- Identifies tail risks through simulation
- Validates robustness across random sequences
================================================================================
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SimulationResult:
    """Result of a single simulation."""
    simulation_id: int
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    went_bankrupt: bool = False  # Equity < 50% initial


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo validation result."""
    
    # Configuration
    num_simulations: int = 0
    initial_capital: float = 0.0
    num_trades: int = 0
    
    # Return Distribution
    return_mean: float = 0.0
    return_std: float = 0.0
    return_5th_percentile: float = 0.0
    return_25th_percentile: float = 0.0
    return_median: float = 0.0
    return_75th_percentile: float = 0.0
    return_95th_percentile: float = 0.0
    
    # Drawdown Distribution
    drawdown_mean: float = 0.0
    drawdown_std: float = 0.0
    drawdown_5th_percentile: float = 0.0
    drawdown_95th_percentile: float = 0.0
    drawdown_worst: float = 0.0
    
    # Sharpe Distribution
    sharpe_mean: float = 0.0
    sharpe_std: float = 0.0
    sharpe_5th_percentile: float = 0.0
    sharpe_95th_percentile: float = 0.0
    
    # Risk Metrics
    bankruptcy_probability: float = 0.0  # % of simulations with equity < 50%
    loss_probability: float = 0.0        # % of simulations with negative return
    
    # Confidence Intervals (95%)
    return_ci_lower: float = 0.0
    return_ci_upper: float = 0.0
    drawdown_ci_lower: float = 0.0
    drawdown_ci_upper: float = 0.0
    
    # Raw results (for further analysis)
    all_results: List[SimulationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'configuration': {
                'num_simulations': self.num_simulations,
                'initial_capital': self.initial_capital,
                'num_trades': self.num_trades,
            },
            'return_distribution': {
                'mean': self.return_mean,
                'std': self.return_std,
                '5th_percentile': self.return_5th_percentile,
                '25th_percentile': self.return_25th_percentile,
                'median': self.return_median,
                '75th_percentile': self.return_75th_percentile,
                '95th_percentile': self.return_95th_percentile,
            },
            'drawdown_distribution': {
                'mean': self.drawdown_mean,
                'std': self.drawdown_std,
                '5th_percentile': self.drawdown_5th_percentile,
                '95th_percentile': self.drawdown_95th_percentile,
                'worst': self.drawdown_worst,
            },
            'sharpe_distribution': {
                'mean': self.sharpe_mean,
                'std': self.sharpe_std,
                '5th_percentile': self.sharpe_5th_percentile,
                '95th_percentile': self.sharpe_95th_percentile,
            },
            'risk_metrics': {
                'bankruptcy_probability': self.bankruptcy_probability,
                'loss_probability': self.loss_probability,
            },
            'confidence_intervals_95': {
                'return': (self.return_ci_lower, self.return_ci_upper),
                'drawdown': (self.drawdown_ci_lower, self.drawdown_ci_upper),
            },
        }


@dataclass
class TradeRecord:
    """Simplified trade record for simulation."""
    pnl: float          # Dollar P&L
    pnl_pct: float      # Percentage P&L
    duration_days: float
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TradeRecord':
        return cls(
            pnl=d.get('pnl', 0),
            pnl_pct=d.get('pnl_pct', 0),
            duration_days=d.get('duration_days', 1),
        )


# =============================================================================
# MONTE CARLO VALIDATOR
# =============================================================================

class MonteCarloValidator:
    """
    Monte Carlo simulation for strategy validation.
    
    Usage:
        validator = MonteCarloValidator()
        result = validator.validate(
            trades=trade_records,
            initial_capital=100000,
            num_simulations=1000,
        )
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        bankruptcy_threshold: float = 0.50,  # 50% of initial capital
        parallel: bool = True,
    ):
        self.risk_free_rate = risk_free_rate
        self.bankruptcy_threshold = bankruptcy_threshold
        self.parallel = parallel
    
    def validate(
        self,
        trades: List[TradeRecord],
        initial_capital: float = 100_000,
        num_simulations: int = 1000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo validation.
        
        Args:
            trades: List of historical trades
            initial_capital: Starting capital
            num_simulations: Number of simulations to run
            seed: Random seed for reproducibility
        
        Returns:
            MonteCarloResult with statistical analysis
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        logger.info(
            f"Starting Monte Carlo validation: {num_simulations} simulations, "
            f"{len(trades)} trades, ${initial_capital:,.0f} capital"
        )
        
        # Run simulations
        if self.parallel and num_simulations > 100:
            results = self._run_parallel(trades, initial_capital, num_simulations)
        else:
            results = self._run_sequential(trades, initial_capital, num_simulations)
        
        # Analyze results
        mc_result = self._analyze_results(
            results, initial_capital, len(trades), num_simulations
        )
        
        logger.info(
            f"Monte Carlo complete: "
            f"Return {mc_result.return_mean:.1f}% ± {mc_result.return_std:.1f}%, "
            f"Bankruptcy prob {mc_result.bankruptcy_probability:.2%}"
        )
        
        return mc_result
    
    def _run_sequential(
        self,
        trades: List[TradeRecord],
        initial_capital: float,
        num_simulations: int,
    ) -> List[SimulationResult]:
        """Run simulations sequentially."""
        results = []
        for i in range(num_simulations):
            result = self._simulate_once(i, trades, initial_capital)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"  Completed {i + 1}/{num_simulations} simulations")
        
        return results
    
    def _run_parallel(
        self,
        trades: List[TradeRecord],
        initial_capital: float,
        num_simulations: int,
    ) -> List[SimulationResult]:
        """Run simulations in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._simulate_once, i, trades, initial_capital)
                for i in range(num_simulations)
            ]
            
            for i, future in enumerate(futures):
                results.append(future.result())
                
                if (i + 1) % 250 == 0:
                    logger.debug(f"  Completed {i + 1}/{num_simulations} simulations")
        
        return results
    
    def _simulate_once(
        self,
        sim_id: int,
        trades: List[TradeRecord],
        initial_capital: float,
    ) -> SimulationResult:
        """
        Run a single simulation.
        
        Randomly shuffles trade order and simulates equity curve.
        """
        # Shuffle trades
        shuffled = trades.copy()
        random.shuffle(shuffled)
        
        # Simulate equity curve
        equity = initial_capital
        peak_equity = initial_capital
        max_drawdown = 0.0
        returns = []
        went_bankrupt = False
        
        for trade in shuffled:
            # Apply trade P&L
            equity += trade.pnl
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, current_dd)
            
            # Track returns
            returns.append(trade.pnl_pct)
            
            # Check bankruptcy
            if equity < initial_capital * self.bankruptcy_threshold:
                went_bankrupt = True
                break
        
        # Calculate metrics
        total_return = (equity - initial_capital) / initial_capital
        
        # Sharpe (simplified)
        if returns and np.std(returns) > 0:
            sharpe = (np.mean(returns) * 252 - self.risk_free_rate) / (np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0
        
        # Win rate
        winners = sum(1 for r in returns if r > 0)
        win_rate = winners / len(returns) if returns else 0
        
        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return SimulationResult(
            simulation_id=sim_id,
            final_equity=equity,
            total_return_pct=total_return * 100,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            win_rate_pct=win_rate * 100,
            profit_factor=profit_factor,
            went_bankrupt=went_bankrupt,
        )
    
    def _analyze_results(
        self,
        results: List[SimulationResult],
        initial_capital: float,
        num_trades: int,
        num_simulations: int,
    ) -> MonteCarloResult:
        """Analyze simulation results and calculate statistics."""
        
        returns = [r.total_return_pct for r in results]
        drawdowns = [r.max_drawdown_pct for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        
        # Return distribution
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        return_percentiles = np.percentile(returns, [5, 25, 50, 75, 95])
        
        # Drawdown distribution
        drawdown_mean = np.mean(drawdowns)
        drawdown_std = np.std(drawdowns)
        drawdown_percentiles = np.percentile(drawdowns, [5, 95])
        drawdown_worst = np.max(drawdowns)
        
        # Sharpe distribution
        sharpe_mean = np.mean(sharpes)
        sharpe_std = np.std(sharpes)
        sharpe_percentiles = np.percentile(sharpes, [5, 95])
        
        # Risk metrics
        bankruptcy_count = sum(1 for r in results if r.went_bankrupt)
        loss_count = sum(1 for r in results if r.total_return_pct < 0)
        
        # 95% confidence intervals
        ci_multiplier = 1.96  # 95% CI
        return_ci_lower = return_mean - ci_multiplier * return_std / np.sqrt(num_simulations)
        return_ci_upper = return_mean + ci_multiplier * return_std / np.sqrt(num_simulations)
        drawdown_ci_lower = drawdown_mean - ci_multiplier * drawdown_std / np.sqrt(num_simulations)
        drawdown_ci_upper = drawdown_mean + ci_multiplier * drawdown_std / np.sqrt(num_simulations)
        
        return MonteCarloResult(
            num_simulations=num_simulations,
            initial_capital=initial_capital,
            num_trades=num_trades,
            
            return_mean=return_mean,
            return_std=return_std,
            return_5th_percentile=return_percentiles[0],
            return_25th_percentile=return_percentiles[1],
            return_median=return_percentiles[2],
            return_75th_percentile=return_percentiles[3],
            return_95th_percentile=return_percentiles[4],
            
            drawdown_mean=drawdown_mean,
            drawdown_std=drawdown_std,
            drawdown_5th_percentile=drawdown_percentiles[0],
            drawdown_95th_percentile=drawdown_percentiles[1],
            drawdown_worst=drawdown_worst,
            
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            sharpe_5th_percentile=sharpe_percentiles[0],
            sharpe_95th_percentile=sharpe_percentiles[1],
            
            bankruptcy_probability=bankruptcy_count / num_simulations,
            loss_probability=loss_count / num_simulations,
            
            return_ci_lower=return_ci_lower,
            return_ci_upper=return_ci_upper,
            drawdown_ci_lower=drawdown_ci_lower,
            drawdown_ci_upper=drawdown_ci_upper,
            
            all_results=results,
        )
    
    def generate_report(self, result: MonteCarloResult) -> str:
        """Generate human-readable Monte Carlo report."""
        lines = []
        lines.append("=" * 70)
        lines.append("MONTE CARLO VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append("CONFIGURATION:")
        lines.append(f"  Simulations: {result.num_simulations:,}")
        lines.append(f"  Trades: {result.num_trades:,}")
        lines.append(f"  Initial Capital: ${result.initial_capital:,.0f}")
        lines.append("")
        
        lines.append("RETURN DISTRIBUTION:")
        lines.append(f"  Mean: {result.return_mean:.2f}%")
        lines.append(f"  Std Dev: {result.return_std:.2f}%")
        lines.append(f"  5th Percentile: {result.return_5th_percentile:.2f}%")
        lines.append(f"  Median: {result.return_median:.2f}%")
        lines.append(f"  95th Percentile: {result.return_95th_percentile:.2f}%")
        lines.append(f"  95% CI: [{result.return_ci_lower:.2f}%, {result.return_ci_upper:.2f}%]")
        lines.append("")
        
        lines.append("DRAWDOWN DISTRIBUTION:")
        lines.append(f"  Mean: {result.drawdown_mean:.2f}%")
        lines.append(f"  Std Dev: {result.drawdown_std:.2f}%")
        lines.append(f"  95th Percentile: {result.drawdown_95th_percentile:.2f}%")
        lines.append(f"  Worst Case: {result.drawdown_worst:.2f}%")
        lines.append("")
        
        lines.append("SHARPE RATIO DISTRIBUTION:")
        lines.append(f"  Mean: {result.sharpe_mean:.2f}")
        lines.append(f"  Std Dev: {result.sharpe_std:.2f}")
        lines.append(f"  5th Percentile: {result.sharpe_5th_percentile:.2f}")
        lines.append(f"  95th Percentile: {result.sharpe_95th_percentile:.2f}")
        lines.append("")
        
        lines.append("RISK METRICS:")
        lines.append(f"  Bankruptcy Probability: {result.bankruptcy_probability:.2%}")
        lines.append(f"  Loss Probability: {result.loss_probability:.2%}")
        lines.append("")
        
        # Assessment
        lines.append("ASSESSMENT:")
        if result.bankruptcy_probability < 0.01:
            lines.append("  ✓ Bankruptcy risk is very low (<1%)")
        elif result.bankruptcy_probability < 0.05:
            lines.append("  [WARN] Bankruptcy risk is moderate (1-5%)")
        else:
            lines.append("  ✗ Bankruptcy risk is HIGH (>5%)")
        
        if result.return_5th_percentile > 0:
            lines.append("  ✓ 95% of scenarios are profitable")
        elif result.return_25th_percentile > 0:
            lines.append("  [WARN] 75% of scenarios are profitable")
        else:
            lines.append("  ✗ Less than 75% of scenarios are profitable")
        
        if result.drawdown_95th_percentile < 15:
            lines.append("  ✓ 95% of scenarios have drawdown < 15%")
        elif result.drawdown_95th_percentile < 20:
            lines.append("  [WARN] 95% of scenarios have drawdown < 20%")
        else:
            lines.append("  ✗ Drawdown risk is HIGH")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_monte_carlo(
    trades: List[Dict],
    initial_capital: float = 100_000,
    num_simulations: int = 1000,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo validation.
    
    Args:
        trades: List of trade dicts with 'pnl', 'pnl_pct', 'duration_days'
        initial_capital: Starting capital
        num_simulations: Number of simulations
        seed: Random seed
    
    Returns:
        MonteCarloResult
    """
    trade_records = [TradeRecord.from_dict(t) for t in trades]
    validator = MonteCarloValidator()
    return validator.validate(
        trades=trade_records,
        initial_capital=initial_capital,
        num_simulations=num_simulations,
        seed=seed,
    )

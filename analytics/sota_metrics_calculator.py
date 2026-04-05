"""
================================================================================
HMATS v6.0.0 - SOTA Metrics Calculator
================================================================================
Purpose: Calculate all SOTA-compliant performance metrics for Dry Run validation

Metrics Calculated:
1. Returns: Annualized Return, Cumulative Return
2. Risk: Max Drawdown, Volatility, VaR, CVaR
3. Risk-Adjusted: Sharpe, Sortino, Calmar Ratios
4. Trading: Win Rate, Profit/Loss Ratio, Profit Factor
5. Distribution: Skewness, Kurtosis

SOTA Compliance:
- All metrics align with industry standards
- Proper handling of short strategy characteristics
- Support for funding rate cost adjustments
================================================================================
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Trade:
    """Single trade record."""
    trade_id: str
    asset: str
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    fees: float = 0.0
    funding_cost: float = 0.0
    
    @property
    def duration_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def net_pnl(self) -> float:
        return self.pnl - self.fees - self.funding_cost


@dataclass
class SOTAMetrics:
    """Complete SOTA-compliant metrics."""
    
    # Identification
    calculation_time: str = ""
    data_start: str = ""
    data_end: str = ""
    total_days: int = 0
    
    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)
    
    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility_annual_pct: float = 0.0
    downside_volatility_pct: float = 0.0
    var_95_pct: float = 0.0
    cvar_95_pct: float = 0.0
    
    # Risk-Adjusted Returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading Performance
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Distribution
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Short-Specific
    short_trades: int = 0
    short_win_rate_pct: float = 0.0
    total_funding_cost: float = 0.0
    funding_cost_pct: float = 0.0
    
    # Targets Comparison
    targets_met: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'identification': {
                'calculation_time': self.calculation_time,
                'data_start': self.data_start,
                'data_end': self.data_end,
                'total_days': self.total_days,
            },
            'returns': {
                'total_return_pct': self.total_return_pct,
                'annualized_return_pct': self.annualized_return_pct,
            },
            'risk': {
                'max_drawdown_pct': self.max_drawdown_pct,
                'max_drawdown_duration_days': self.max_drawdown_duration_days,
                'volatility_annual_pct': self.volatility_annual_pct,
                'downside_volatility_pct': self.downside_volatility_pct,
                'var_95_pct': self.var_95_pct,
                'cvar_95_pct': self.cvar_95_pct,
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
            },
            'trading': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate_pct': self.win_rate_pct,
                'profit_loss_ratio': self.profit_loss_ratio,
                'profit_factor': self.profit_factor,
            },
            'distribution': {
                'skewness': self.skewness,
                'kurtosis': self.kurtosis,
            },
            'short_specific': {
                'short_trades': self.short_trades,
                'short_win_rate_pct': self.short_win_rate_pct,
                'total_funding_cost': self.total_funding_cost,
                'funding_cost_pct': self.funding_cost_pct,
            },
            'targets_met': self.targets_met,
        }


@dataclass
class SOTATargets:
    """SOTA target thresholds."""
    annualized_return_pct: float = 30.0
    max_drawdown_pct: float = 10.0
    sharpe_ratio: float = 1.5
    sortino_ratio: float = 2.0
    calmar_ratio: float = 2.0
    win_rate_pct: float = 50.0
    profit_loss_ratio: float = 2.0
    profit_factor: float = 1.8


# =============================================================================
# CALCULATOR
# =============================================================================

class SOTAMetricsCalculator:
    """
    Calculate SOTA-compliant performance metrics.
    
    Usage:
        calculator = SOTAMetricsCalculator()
        metrics = calculator.calculate(
            trades=trade_list,
            equity_curve=equity_series,
            initial_capital=100000,
        )
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,  # 5% annual
        targets: Optional[SOTATargets] = None,
    ):
        self.risk_free_rate = risk_free_rate
        self.targets = targets or SOTATargets()
    
    def calculate(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        initial_capital: float,
    ) -> SOTAMetrics:
        """
        Calculate all SOTA metrics.
        
        Args:
            trades: List of completed trades
            equity_curve: Daily equity values (index=date)
            initial_capital: Starting capital
        
        Returns:
            SOTAMetrics with all calculated values
        """
        metrics = SOTAMetrics()
        
        # Identification
        metrics.calculation_time = datetime.now(timezone.utc).isoformat()
        if len(equity_curve) > 0:
            metrics.data_start = str(equity_curve.index[0])
            metrics.data_end = str(equity_curve.index[-1])
            metrics.total_days = len(equity_curve)
        
        # Returns
        returns_metrics = self._calculate_returns(equity_curve, initial_capital)
        metrics.total_return_pct = returns_metrics['total_return_pct']
        metrics.annualized_return_pct = returns_metrics['annualized_return_pct']
        metrics.monthly_returns = returns_metrics['monthly_returns']
        
        # Risk
        risk_metrics = self._calculate_risk(equity_curve)
        metrics.max_drawdown_pct = risk_metrics['max_drawdown_pct']
        metrics.max_drawdown_duration_days = risk_metrics['max_drawdown_duration_days']
        metrics.volatility_annual_pct = risk_metrics['volatility_annual_pct']
        metrics.downside_volatility_pct = risk_metrics['downside_volatility_pct']
        metrics.var_95_pct = risk_metrics['var_95_pct']
        metrics.cvar_95_pct = risk_metrics['cvar_95_pct']
        
        # Risk-Adjusted
        metrics.sharpe_ratio = self._calculate_sharpe(
            metrics.annualized_return_pct,
            metrics.volatility_annual_pct,
        )
        metrics.sortino_ratio = self._calculate_sortino(
            metrics.annualized_return_pct,
            metrics.downside_volatility_pct,
        )
        metrics.calmar_ratio = self._calculate_calmar(
            metrics.annualized_return_pct,
            metrics.max_drawdown_pct,
        )
        
        # Trading Performance
        trading_metrics = self._calculate_trading_metrics(trades)
        metrics.total_trades = trading_metrics['total_trades']
        metrics.winning_trades = trading_metrics['winning_trades']
        metrics.losing_trades = trading_metrics['losing_trades']
        metrics.win_rate_pct = trading_metrics['win_rate_pct']
        metrics.avg_win_pct = trading_metrics['avg_win_pct']
        metrics.avg_loss_pct = trading_metrics['avg_loss_pct']
        metrics.profit_loss_ratio = trading_metrics['profit_loss_ratio']
        metrics.profit_factor = trading_metrics['profit_factor']
        
        # Distribution
        distribution_metrics = self._calculate_distribution(equity_curve)
        metrics.skewness = distribution_metrics['skewness']
        metrics.kurtosis = distribution_metrics['kurtosis']
        
        # Short-Specific
        short_metrics = self._calculate_short_metrics(trades, initial_capital)
        metrics.short_trades = short_metrics['short_trades']
        metrics.short_win_rate_pct = short_metrics['short_win_rate_pct']
        metrics.total_funding_cost = short_metrics['total_funding_cost']
        metrics.funding_cost_pct = short_metrics['funding_cost_pct']
        
        # Compare to targets
        metrics.targets_met = self._check_targets(metrics)
        
        return metrics
    
    # =========================================================================
    # RETURN CALCULATIONS
    # =========================================================================
    
    def _calculate_returns(
        self,
        equity_curve: pd.Series,
        initial_capital: float,
    ) -> Dict:
        """Calculate return metrics."""
        if len(equity_curve) < 2:
            return {
                'total_return_pct': 0.0,
                'annualized_return_pct': 0.0,
                'monthly_returns': [],
            }
        
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Annualized return
        days = len(equity_curve)
        if days > 0:
            annualized = (1 + total_return) ** (365 / days) - 1
        else:
            annualized = 0.0
        
        # Monthly returns
        monthly_returns = []
        if hasattr(equity_curve.index, 'to_period'):
            monthly = equity_curve.resample('M').last()
            monthly_ret = monthly.pct_change().dropna()
            monthly_returns = monthly_ret.tolist()
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annualized * 100,
            'monthly_returns': monthly_returns,
        }
    
    # =========================================================================
    # RISK CALCULATIONS
    # =========================================================================
    
    def _calculate_risk(self, equity_curve: pd.Series) -> Dict:
        """Calculate risk metrics."""
        if len(equity_curve) < 2:
            return {
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration_days': 0,
                'volatility_annual_pct': 0.0,
                'downside_volatility_pct': 0.0,
                'var_95_pct': 0.0,
                'cvar_95_pct': 0.0,
            }
        
        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        # Max Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Max Drawdown Duration
        dd_duration = self._calculate_dd_duration(equity_curve)
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Downside Volatility (only negative returns)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_vol = negative_returns.std() * np.sqrt(252)
        else:
            downside_vol = 0.0
        
        # VaR 95%
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        
        # CVaR 95% (Expected Shortfall)
        returns_below_var = daily_returns[daily_returns <= var_95]
        cvar_95 = returns_below_var.mean() if len(returns_below_var) > 0 else var_95
        
        return {
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_duration_days': dd_duration,
            'volatility_annual_pct': volatility * 100,
            'downside_volatility_pct': downside_vol * 100,
            'var_95_pct': abs(var_95) * 100,
            'cvar_95_pct': abs(cvar_95) * 100,
        }
    
    def _calculate_dd_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        rolling_max = equity_curve.expanding().max()
        in_drawdown = equity_curve < rolling_max
        
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    # =========================================================================
    # RISK-ADJUSTED CALCULATIONS
    # =========================================================================
    
    def _calculate_sharpe(
        self,
        annualized_return: float,
        volatility: float,
    ) -> float:
        """Calculate Sharpe Ratio."""
        if volatility == 0:
            return 0.0
        
        excess_return = annualized_return - (self.risk_free_rate * 100)
        return excess_return / volatility
    
    def _calculate_sortino(
        self,
        annualized_return: float,
        downside_volatility: float,
    ) -> float:
        """Calculate Sortino Ratio."""
        if downside_volatility == 0:
            return 0.0
        
        excess_return = annualized_return - (self.risk_free_rate * 100)
        return excess_return / downside_volatility
    
    def _calculate_calmar(
        self,
        annualized_return: float,
        max_drawdown: float,
    ) -> float:
        """Calculate Calmar Ratio."""
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    # =========================================================================
    # TRADING PERFORMANCE
    # =========================================================================
    
    def _calculate_trading_metrics(self, trades: List[Trade]) -> Dict:
        """Calculate trading performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'profit_loss_ratio': 0.0,
                'profit_factor': 0.0,
            }
        
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        total = len(trades)
        win_count = len(winners)
        loss_count = len(losers)
        
        win_rate = win_count / total if total > 0 else 0
        
        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl_pct for t in losers])) if losers else 0
        
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        total_profit = sum(t.pnl for t in winners)
        total_loss = abs(sum(t.pnl for t in losers))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'total_trades': total,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate_pct': win_rate * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_loss_ratio': pl_ratio,
            'profit_factor': profit_factor,
        }
    
    # =========================================================================
    # DISTRIBUTION
    # =========================================================================
    
    def _calculate_distribution(self, equity_curve: pd.Series) -> Dict:
        """Calculate return distribution metrics."""
        if len(equity_curve) < 3:
            return {'skewness': 0.0, 'kurtosis': 0.0}
        
        daily_returns = equity_curve.pct_change().dropna()
        
        from scipy import stats
        try:
            skewness = stats.skew(daily_returns)
            kurtosis = stats.kurtosis(daily_returns)
        except Exception:
            skewness = 0.0
            kurtosis = 0.0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
    
    # =========================================================================
    # SHORT-SPECIFIC
    # =========================================================================
    
    def _calculate_short_metrics(
        self,
        trades: List[Trade],
        initial_capital: float,
    ) -> Dict:
        """Calculate short-specific metrics."""
        short_trades = [t for t in trades if t.direction == "SHORT"]
        
        if not short_trades:
            return {
                'short_trades': 0,
                'short_win_rate_pct': 0.0,
                'total_funding_cost': 0.0,
                'funding_cost_pct': 0.0,
            }
        
        short_winners = [t for t in short_trades if t.is_winner]
        short_win_rate = len(short_winners) / len(short_trades) if short_trades else 0
        
        total_funding = sum(t.funding_cost for t in short_trades)
        funding_pct = total_funding / initial_capital if initial_capital > 0 else 0
        
        return {
            'short_trades': len(short_trades),
            'short_win_rate_pct': short_win_rate * 100,
            'total_funding_cost': total_funding,
            'funding_cost_pct': funding_pct * 100,
        }
    
    # =========================================================================
    # TARGETS COMPARISON
    # =========================================================================
    
    def _check_targets(self, metrics: SOTAMetrics) -> Dict[str, bool]:
        """Check if metrics meet SOTA targets."""
        return {
            'annualized_return': metrics.annualized_return_pct >= self.targets.annualized_return_pct,
            'max_drawdown': metrics.max_drawdown_pct <= self.targets.max_drawdown_pct,
            'sharpe_ratio': metrics.sharpe_ratio >= self.targets.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio >= self.targets.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio >= self.targets.calmar_ratio,
            'win_rate': metrics.win_rate_pct >= self.targets.win_rate_pct,
            'profit_loss_ratio': metrics.profit_loss_ratio >= self.targets.profit_loss_ratio,
            'profit_factor': metrics.profit_factor >= self.targets.profit_factor,
        }
    
    def generate_summary(self, metrics: SOTAMetrics) -> str:
        """Generate human-readable summary."""
        targets_met = sum(metrics.targets_met.values())
        total_targets = len(metrics.targets_met)
        
        summary = []
        summary.append("=" * 60)
        summary.append("SOTA METRICS SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Period: {metrics.data_start} to {metrics.data_end}")
        summary.append(f"Total Days: {metrics.total_days}")
        summary.append("")
        
        summary.append("RETURNS:")
        summary.append(f"  Total Return: {metrics.total_return_pct:.2f}%")
        summary.append(f"  Annualized: {metrics.annualized_return_pct:.2f}%")
        summary.append("")
        
        summary.append("RISK:")
        summary.append(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        summary.append(f"  Volatility: {metrics.volatility_annual_pct:.2f}%")
        summary.append(f"  VaR 95%: {metrics.var_95_pct:.2f}%")
        summary.append("")
        
        summary.append("RISK-ADJUSTED:")
        summary.append(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
        summary.append(f"  Sortino: {metrics.sortino_ratio:.2f}")
        summary.append(f"  Calmar: {metrics.calmar_ratio:.2f}")
        summary.append("")
        
        summary.append("TRADING:")
        summary.append(f"  Total Trades: {metrics.total_trades}")
        summary.append(f"  Win Rate: {metrics.win_rate_pct:.1f}%")
        summary.append(f"  P/L Ratio: {metrics.profit_loss_ratio:.2f}")
        summary.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
        summary.append("")
        
        summary.append("SHORT-SPECIFIC:")
        summary.append(f"  Short Trades: {metrics.short_trades}")
        summary.append(f"  Short Win Rate: {metrics.short_win_rate_pct:.1f}%")
        summary.append(f"  Funding Cost: ${metrics.total_funding_cost:,.2f}")
        summary.append("")
        
        summary.append(f"TARGETS MET: {targets_met}/{total_targets}")
        for target, met in metrics.targets_met.items():
            status = "✓ PASS" if met else "✗ FAIL"
            summary.append(f"  {target}: {status}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_sota_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    initial_capital: float = 10_000,  # [CFG-4] was 100K, actual $10K
    risk_free_rate: float = 0.05,
) -> SOTAMetrics:
    """Convenience function to calculate SOTA metrics."""
    calculator = SOTAMetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate(trades, equity_curve, initial_capital)

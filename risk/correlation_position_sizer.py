"""
================================================================================
HMATS v5.2.0 - Correlation-Aware Position Sizer
================================================================================
Purpose: Shrink positions when portfolio contains highly correlated assets

Features:
1. Real-time correlation matrix tracking
2. Portfolio-level risk aggregation
3. Automatic position reduction for correlated holdings
4. Diversification scoring

Author: HMATS v5.2 Upgrade
Version: 5.2.0
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class DiversificationLevel(Enum):
    """Portfolio diversification levels."""
    EXCELLENT = "excellent"      # Low correlation, well diversified
    GOOD = "good"                # Moderate correlation
    MODERATE = "moderate"        # Some concentration risk
    POOR = "poor"                # High correlation, concentrated
    CRITICAL = "critical"        # Extremely correlated, reduce exposure


@dataclass
class CorrelationConfig:
    """Configuration for correlation-aware sizing."""
    # Correlation thresholds
    high_corr_threshold: float = 0.75       # Above this = highly correlated
    moderate_corr_threshold: float = 0.50   # Above this = moderately correlated
    
    # Position adjustment factors
    high_corr_reduction: float = 0.50       # Reduce by 50% for high correlation
    moderate_corr_reduction: float = 0.75   # Reduce by 25% for moderate
    
    # Portfolio limits
    max_correlated_exposure: float = 0.40   # Max 40% in correlated assets
    max_single_asset: float = 0.30          # Max 30% in single asset
    
    # Calculation parameters
    correlation_lookback: int = 100         # Bars for correlation calc
    min_samples: int = 30                   # Minimum samples needed
    update_interval_seconds: int = 300      # Update correlation every 5 min


@dataclass
class CorrelationMatrix:
    """Current correlation state."""
    timestamp: datetime
    matrix: Dict[str, Dict[str, float]]     # symbol -> symbol -> correlation
    eigenvalues: Optional[np.ndarray] = None
    diversification_score: float = 1.0
    
    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        if sym1 == sym2:
            return 1.0
        if sym1 in self.matrix and sym2 in self.matrix[sym1]:
            return self.matrix[sym1][sym2]
        if sym2 in self.matrix and sym1 in self.matrix[sym2]:
            return self.matrix[sym2][sym1]
        return 0.0


@dataclass
class PositionSizeResult:
    """Result of correlation-aware position sizing."""
    symbol: str
    base_size: float
    adjusted_size: float
    reduction_factor: float
    reason: str
    correlation_impact: Dict[str, float]    # Which assets caused reduction
    diversification_level: DiversificationLevel
    portfolio_corr_exposure: float


class CorrelationAwarePositionSizer:
    """
    Position sizer that accounts for portfolio correlation.
    
    Automatically reduces position sizes when:
    1. New position is highly correlated with existing holdings
    2. Portfolio becomes too concentrated in correlated assets
    3. Diversification score drops below threshold
    """
    
    def __init__(self, config: CorrelationConfig = None, symbols: List[str] = None):
        self.config = config or CorrelationConfig()
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        
        # Price history for correlation calculation
        self.returns_history: Dict[str, deque] = {
            sym: deque(maxlen=self.config.correlation_lookback)
            for sym in self.symbols
        }
        self.last_prices: Dict[str, float] = {}
        
        # Correlation state
        self.correlation_matrix: Optional[CorrelationMatrix] = None
        self.last_update: Optional[datetime] = None
        
        # Current positions for portfolio-level analysis
        self.current_positions: Dict[str, float] = {}  # symbol -> notional value
        self.total_equity: float = 0.0
        
        logger.info(f"[CORR-SIZER] Initialized for {self.symbols} "
                   f"with high_corr_threshold={self.config.high_corr_threshold}")
    
    # =========================================================================
    # PRICE & RETURN TRACKING
    # =========================================================================
    
    def update_price(self, symbol: str, price: float):
        """Update price and calculate return."""
        if symbol not in self.symbols:
            return
        
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            if last_price > 0:
                ret = np.log(price / last_price)
                self.returns_history[symbol].append(ret)
        
        self.last_prices[symbol] = price
        
        # Check if we should update correlation matrix
        now = datetime.now(timezone.utc)
        if (self.last_update is None or 
            (now - self.last_update).total_seconds() > self.config.update_interval_seconds):
            self._update_correlation_matrix()
    
    def update_positions(self, positions: Dict[str, float], equity: float):
        """Update current position notionals and equity."""
        self.current_positions = positions.copy()
        self.total_equity = equity
    
    # =========================================================================
    # CORRELATION CALCULATION
    # =========================================================================
    
    def _update_correlation_matrix(self):
        """Recalculate correlation matrix from return history."""
        # Check if we have enough data
        min_len = min(len(self.returns_history[sym]) for sym in self.symbols)
        if min_len < self.config.min_samples:
            logger.debug(f"[CORR-SIZER] Not enough data: {min_len} < {self.config.min_samples}")
            return
        
        # Build returns matrix
        n = min_len
        returns_matrix = np.zeros((n, len(self.symbols)))
        
        for i, sym in enumerate(self.symbols):
            returns_list = list(self.returns_history[sym])[-n:]
            returns_matrix[:, i] = returns_list
        
        # Calculate correlation matrix
        corr_np = np.corrcoef(returns_matrix.T)
        
        # Convert to dict format
        matrix_dict = {}
        for i, sym1 in enumerate(self.symbols):
            matrix_dict[sym1] = {}
            for j, sym2 in enumerate(self.symbols):
                matrix_dict[sym1][sym2] = float(corr_np[i, j])
        
        # Calculate eigenvalues for diversification score
        eigenvalues = np.linalg.eigvalsh(corr_np)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Diversification score: ratio of smallest to largest eigenvalue
        # Higher = more diversified, Lower = more concentrated
        if eigenvalues[0] > 0:
            div_score = eigenvalues[-1] / eigenvalues[0]
        else:
            div_score = 0.0
        
        self.correlation_matrix = CorrelationMatrix(
            timestamp=datetime.now(timezone.utc),
            matrix=matrix_dict,
            eigenvalues=eigenvalues,
            diversification_score=div_score
        )
        self.last_update = datetime.now(timezone.utc)
        
        logger.info(f"[CORR-SIZER] Updated correlation matrix. "
                   f"Diversification score: {div_score:.3f}")
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def calculate_adjusted_size(
        self,
        symbol: str,
        base_size: float,
        base_notional: float = None
    ) -> PositionSizeResult:
        """
        Calculate correlation-adjusted position size.
        
        Args:
            symbol: Asset to size
            base_size: Base position size (before correlation adjustment)
            base_notional: Notional value of position (optional)
            
        Returns:
            PositionSizeResult with adjusted size and reasoning
        """
        if base_notional is None:
            base_notional = base_size * self.last_prices.get(symbol, 1.0)
        
        reduction_factor = 1.0
        correlation_impact = {}
        reasons = []
        
        # Check if correlation matrix is available
        if self.correlation_matrix is None:
            return PositionSizeResult(
                symbol=symbol,
                base_size=base_size,
                adjusted_size=base_size,
                reduction_factor=1.0,
                reason="No correlation data available",
                correlation_impact={},
                diversification_level=DiversificationLevel.GOOD,
                portfolio_corr_exposure=0.0
            )
        
        # 1. Check correlation with existing positions
        correlated_exposure = 0.0
        
        for existing_sym, existing_notional in self.current_positions.items():
            if existing_sym == symbol or existing_notional <= 0:
                continue
            
            corr = self.correlation_matrix.get_correlation(symbol, existing_sym)
            correlation_impact[existing_sym] = corr
            
            if corr >= self.config.high_corr_threshold:
                # High correlation - apply reduction
                position_weight = existing_notional / max(self.total_equity, 1)
                impact = position_weight * corr
                correlated_exposure += impact
                
                reasons.append(f"High corr with {existing_sym}: {corr:.2f}")
                reduction_factor = min(reduction_factor, self.config.high_corr_reduction)
                
            elif corr >= self.config.moderate_corr_threshold:
                # Moderate correlation
                position_weight = existing_notional / max(self.total_equity, 1)
                impact = position_weight * corr * 0.5  # Half weight
                correlated_exposure += impact
                
                reasons.append(f"Moderate corr with {existing_sym}: {corr:.2f}")
                reduction_factor = min(reduction_factor, self.config.moderate_corr_reduction)
        
        # 2. Check portfolio-level correlated exposure limit
        if correlated_exposure > self.config.max_correlated_exposure:
            excess_ratio = self.config.max_correlated_exposure / max(correlated_exposure, 0.01)
            reduction_factor = min(reduction_factor, excess_ratio)
            reasons.append(f"Portfolio corr exposure {correlated_exposure:.1%} > limit")
        
        # 3. Check single-asset concentration
        if self.total_equity > 0:
            new_weight = base_notional / self.total_equity
            if new_weight > self.config.max_single_asset:
                cap_factor = self.config.max_single_asset / new_weight
                reduction_factor = min(reduction_factor, cap_factor)
                reasons.append(f"Single asset limit: {new_weight:.1%} > {self.config.max_single_asset:.1%}")
        
        # 4. Determine diversification level
        div_level = self._get_diversification_level(correlated_exposure)
        
        # Apply reduction
        adjusted_size = base_size * reduction_factor
        
        return PositionSizeResult(
            symbol=symbol,
            base_size=base_size,
            adjusted_size=adjusted_size,
            reduction_factor=reduction_factor,
            reason="; ".join(reasons) if reasons else "No correlation adjustment needed",
            correlation_impact=correlation_impact,
            diversification_level=div_level,
            portfolio_corr_exposure=correlated_exposure
        )
    
    def _get_diversification_level(self, corr_exposure: float) -> DiversificationLevel:
        """Determine diversification level from correlated exposure."""
        if corr_exposure < 0.15:
            return DiversificationLevel.EXCELLENT
        elif corr_exposure < 0.30:
            return DiversificationLevel.GOOD
        elif corr_exposure < 0.50:
            return DiversificationLevel.MODERATE
        elif corr_exposure < 0.70:
            return DiversificationLevel.POOR
        else:
            return DiversificationLevel.CRITICAL
    
    # =========================================================================
    # PORTFOLIO ANALYSIS
    # =========================================================================
    
    def get_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics."""
        if self.correlation_matrix is None:
            return {
                "status": "no_data",
                "diversification_score": None,
                "correlation_matrix": None
            }
        
        # Calculate portfolio variance contribution
        variance_contrib = {}
        total_notional = sum(self.current_positions.values())
        
        if total_notional > 0:
            for sym in self.current_positions:
                weight = self.current_positions[sym] / total_notional
                # Marginal contribution to risk
                contrib = 0.0
                for other_sym in self.current_positions:
                    other_weight = self.current_positions[other_sym] / total_notional
                    corr = self.correlation_matrix.get_correlation(sym, other_sym)
                    contrib += weight * other_weight * corr
                variance_contrib[sym] = contrib
        
        return {
            "status": "active",
            "timestamp": self.correlation_matrix.timestamp.isoformat(),
            "diversification_score": self.correlation_matrix.diversification_score,
            "diversification_level": self._get_diversification_level(
                sum(variance_contrib.values())
            ).value,
            "correlation_matrix": self.correlation_matrix.matrix,
            "variance_contributions": variance_contrib,
            "position_weights": {
                sym: val / max(total_notional, 1)
                for sym, val in self.current_positions.items()
            }
        }
    
    def should_reduce_exposure(self) -> Tuple[bool, str]:
        """Check if portfolio correlation warrants exposure reduction."""
        if self.correlation_matrix is None:
            return False, "No correlation data"
        
        metrics = self.get_portfolio_risk_metrics()
        div_level = metrics.get("diversification_level")
        
        if div_level == DiversificationLevel.CRITICAL.value:
            return True, "Critical concentration - reduce all positions"
        elif div_level == DiversificationLevel.POOR.value:
            return True, "Poor diversification - consider reducing correlated positions"
        
        return False, "Portfolio diversification acceptable"


# =============================================================================
# SINGLETON
# =============================================================================

_correlation_sizer: Optional[CorrelationAwarePositionSizer] = None


def get_correlation_sizer(config: CorrelationConfig = None) -> CorrelationAwarePositionSizer:
    """Get or create the global correlation-aware position sizer."""
    global _correlation_sizer
    if _correlation_sizer is None:
        _correlation_sizer = CorrelationAwarePositionSizer(config)
    elif config is not None and _correlation_sizer.config != config:
        logger.info("[CORRELATION_SIZER] Config changed; refreshing singleton instance")
        _correlation_sizer = CorrelationAwarePositionSizer(config)
    return _correlation_sizer


def reset_correlation_sizer():
    """Reset the global correlation sizer."""
    global _correlation_sizer
    _correlation_sizer = None

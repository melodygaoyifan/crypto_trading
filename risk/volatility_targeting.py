#!/usr/bin/env python3
"""
================================================================================
HMATS v5.0.5 - Enhanced Volatility Targeting Position Sizer
================================================================================
Purpose: Flexible per-asset volatility targeting with customizable parameters

v5.0.5 Enhancements:
1. Per-asset target daily volatility (e.g., SOL=2%, BTC=1.5%)
2. Per-asset maximum weight caps
3. Correlation-adjusted portfolio volatility
4. Volatility regime-aware scaling
5. Full backward compatibility with v4.5/v5.0

Configuration:
    # Per-asset targeting
    asset_config = {
        "SOL/USD": AssetVolConfig(target_daily_vol=0.02, max_weight=0.30),
        "BTC/USD": AssetVolConfig(target_daily_vol=0.015, max_weight=0.40),
        "ETH/USD": AssetVolConfig(target_daily_vol=0.018, max_weight=0.35),
    }

Formula:
    Position_Weight = min(
        (Asset_Target_Vol / Asset_Current_Vol) * Base_Weight,
        Asset_Max_Weight
    )

Author: HMATS v5.0.5 Patch
Version: 5.0.5
================================================================================
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AssetVolConfig:
    """
    v5.0.5: Per-asset volatility targeting configuration.
    
    Allows different volatility targets and caps for each asset.
    """
    target_daily_vol: float = 0.02      # Target daily volatility (2% default)
    target_annual_vol: float = None     # Auto-calculated if not set
    max_weight: float = 0.30            # Maximum portfolio weight
    min_weight: float = 0.01            # Minimum position size
    
    # Scaling parameters
    vol_scale_factor: float = 1.0       # Multiplier on base vol target
    use_regime_scaling: bool = True     # Adjust target based on vol regime
    
    # Risk limits
    max_daily_var_pct: float = 3.0      # Maximum daily VaR (99%)
    
    def __post_init__(self):
        if self.target_annual_vol is None:
            self.target_annual_vol = self.target_daily_vol * np.sqrt(252)


@dataclass
class VolatilityMetrics:
    """Volatility metrics for an asset."""
    symbol: str
    
    # Volatility measures
    daily_vol: float = 0.0         # Daily return volatility
    annual_vol: float = 0.0        # Annualized volatility
    vol_of_vol: float = 0.0        # Volatility of volatility
    
    # v5.0.5 additions
    vol_percentile: float = 50.0   # Percentile vs history
    vol_zscore: float = 0.0        # Z-score vs recent history
    vol_trend: float = 0.0         # Trend in volatility
    
    # Rolling window data
    lookback_days: int = 30
    n_observations: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_high_vol(self) -> bool:
        """Check if volatility is elevated (>80% annualized)."""
        return self.annual_vol > 0.8
    
    @property
    def is_low_vol(self) -> bool:
        """Check if volatility is low (<30% annualized)."""
        return self.annual_vol < 0.3
    
    @property
    def vol_regime(self) -> str:
        """Get volatility regime classification."""
        if self.vol_percentile < 25:
            return "low"
        elif self.vol_percentile > 75:
            return "high"
        elif self.vol_percentile > 90:
            return "extreme"
        return "normal"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "daily_vol": round(self.daily_vol, 6),
            "annual_vol": round(self.annual_vol, 4),
            "vol_of_vol": round(self.vol_of_vol, 6),
            "vol_percentile": round(self.vol_percentile, 1),
            "vol_zscore": round(self.vol_zscore, 2),
            "vol_regime": self.vol_regime,
            "is_high_vol": self.is_high_vol,
            "n_observations": self.n_observations
        }


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation."""
    symbol: str
    
    # Size metrics
    base_weight: float = 0.0       # Base allocation weight [0,1]
    adjusted_weight: float = 0.0   # Vol-adjusted weight [0,1]
    position_usd: float = 0.0      # Position in USD
    
    # Risk metrics
    expected_daily_vol_usd: float = 0.0
    expected_daily_vol_pct: float = 0.0  # v5.0.5
    max_loss_at_stop: float = 0.0
    
    # Adjustments applied
    vol_adjustment_factor: float = 1.0
    cap_applied: bool = False
    floor_applied: bool = False    # v5.0.5
    
    # v5.0.5: Target comparison
    target_daily_vol: float = 0.0
    actual_vs_target_ratio: float = 1.0
    
    reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "base_weight": round(self.base_weight, 4),
            "adjusted_weight": round(self.adjusted_weight, 4),
            "position_usd": round(self.position_usd, 2),
            "expected_daily_vol_usd": round(self.expected_daily_vol_usd, 2),
            "expected_daily_vol_pct": round(self.expected_daily_vol_pct, 4),
            "vol_adjustment_factor": round(self.vol_adjustment_factor, 4),
            "target_daily_vol": round(self.target_daily_vol, 4),
            "actual_vs_target": round(self.actual_vs_target_ratio, 2),
            "cap_applied": self.cap_applied,
            "floor_applied": self.floor_applied,
            "reason": self.reason
        }


@dataclass
class PortfolioRiskState:
    """Current portfolio risk state."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Portfolio volatility
    target_daily_vol: float = 0.02  # 2% daily target
    current_daily_vol: float = 0.0
    vol_utilization: float = 0.0    # current / target
    
    # Exposure
    total_exposure_usd: float = 0.0
    gross_exposure_pct: float = 0.0
    
    # Per-asset
    asset_allocations: Dict[str, float] = field(default_factory=dict)
    
    # v5.0.5 additions
    correlation_adjusted_vol: float = 0.0
    diversification_ratio: float = 1.0
    marginal_contributions: Dict[str, float] = field(default_factory=dict)
    
    def is_over_risk(self) -> bool:
        """Check if portfolio risk exceeds target."""
        return self.vol_utilization > 1.2  # Allow 20% buffer
    
    def to_dict(self) -> Dict:
        return {
            "target_daily_vol": self.target_daily_vol,
            "current_daily_vol": round(self.current_daily_vol, 4),
            "vol_utilization": round(self.vol_utilization, 2),
            "total_exposure_usd": round(self.total_exposure_usd, 2),
            "gross_exposure_pct": round(self.gross_exposure_pct, 2),
            "diversification_ratio": round(self.diversification_ratio, 2),
            "asset_allocations": self.asset_allocations
        }


# =============================================================================
# ENHANCED VOLATILITY CALCULATOR (v5.0.5)
# =============================================================================

class EnhancedVolatilityCalculator:
    """
    v5.0.5: Enhanced volatility calculator with percentile ranking and trends.
    """
    
    def __init__(
        self,
        lookback_days: int = 30,
        min_observations: int = 10,
        percentile_window: int = 252
    ):
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.percentile_window = percentile_window
        
        # Return buffers per symbol
        self._return_buffers: Dict[str, deque] = {}
        self._vol_buffers: Dict[str, deque] = {}
        self._historical_vol: Dict[str, deque] = {}  # For percentile
        
        self._lock = threading.Lock()
    
    def update(
        self,
        symbol: str,
        current_price: float,
        previous_price: float
    ) -> VolatilityMetrics:
        """
        Update volatility with new price observation.
        """
        # Calculate return
        if previous_price > 0:
            ret = (current_price - previous_price) / previous_price
        else:
            ret = 0.0
        
        with self._lock:
            # Initialize buffers
            if symbol not in self._return_buffers:
                self._return_buffers[symbol] = deque(maxlen=self.lookback_days)
                self._vol_buffers[symbol] = deque(maxlen=self.lookback_days)
                self._historical_vol[symbol] = deque(maxlen=self.percentile_window)
            
            self._return_buffers[symbol].append(ret)
            
            returns = np.array(self._return_buffers[symbol])
            n_obs = len(returns)
            
            if n_obs >= self.min_observations:
                daily_vol = np.std(returns)
                annual_vol = daily_vol * np.sqrt(252)
                
                # Update vol buffer
                self._vol_buffers[symbol].append(daily_vol)
                self._historical_vol[symbol].append(daily_vol)
                
                # Vol of vol
                vol_of_vol = np.std(self._vol_buffers[symbol]) if len(self._vol_buffers[symbol]) >= 5 else 0.0
                
                # v5.0.5: Calculate percentile and z-score
                hist = list(self._historical_vol[symbol])
                if len(hist) >= 20:
                    vol_percentile = self._calculate_percentile(daily_vol, hist)
                    vol_zscore = (daily_vol - np.mean(hist)) / (np.std(hist) + 1e-10)
                else:
                    vol_percentile = 50.0
                    vol_zscore = 0.0
                
                # v5.0.5: Vol trend (recent vs older)
                if len(self._vol_buffers[symbol]) >= 10:
                    recent = list(self._vol_buffers[symbol])[-5:]
                    older = list(self._vol_buffers[symbol])[-10:-5]
                    vol_trend = (np.mean(recent) - np.mean(older)) / (np.mean(older) + 1e-10)
                else:
                    vol_trend = 0.0
                
            else:
                daily_vol = 0.03
                annual_vol = daily_vol * np.sqrt(252)
                vol_of_vol = 0.0
                vol_percentile = 50.0
                vol_zscore = 0.0
                vol_trend = 0.0
        
        return VolatilityMetrics(
            symbol=symbol,
            daily_vol=daily_vol,
            annual_vol=annual_vol,
            vol_of_vol=vol_of_vol,
            vol_percentile=vol_percentile,
            vol_zscore=vol_zscore,
            vol_trend=vol_trend,
            lookback_days=self.lookback_days,
            n_observations=n_obs,
            last_updated=datetime.now(timezone.utc)
        )
    
    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile rank."""
        sorted_hist = sorted(history)
        rank = sum(1 for v in sorted_hist if v < value)
        return (rank / len(sorted_hist)) * 100
    
    def get_volatility(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Get current volatility metrics."""
        with self._lock:
            if symbol not in self._return_buffers:
                return None
            
            returns = np.array(self._return_buffers[symbol])
            if len(returns) < self.min_observations:
                return None
            
            daily_vol = np.std(returns)
            return VolatilityMetrics(
                symbol=symbol,
                daily_vol=daily_vol,
                annual_vol=daily_vol * np.sqrt(252),
                n_observations=len(returns)
            )
    
    def bulk_update(self, symbol: str, prices: List[float]) -> VolatilityMetrics:
        """Bulk update with price history."""
        metrics = None
        for i in range(1, len(prices)):
            metrics = self.update(symbol, prices[i], prices[i-1])
        return metrics


# =============================================================================
# ENHANCED VOLATILITY TARGETING SIZER (v5.0.5)
# =============================================================================

class EnhancedVolatilityTargetingSizer:
    """
    v5.0.5: Enhanced volatility targeting with per-asset configuration.
    
    Features:
    - Per-asset target volatility
    - Per-asset weight caps
    - Correlation-adjusted portfolio vol
    - Regime-aware scaling
    """
    
    # Default per-asset configs
    DEFAULT_CONFIGS = {
        "SOL/USD": AssetVolConfig(target_daily_vol=0.02, max_weight=0.30),
        "BTC/USD": AssetVolConfig(target_daily_vol=0.015, max_weight=0.40),
        "ETH/USD": AssetVolConfig(target_daily_vol=0.018, max_weight=0.35),
    }
    
    def __init__(
        self,
        portfolio_target_daily_vol: float = 0.02,
        asset_configs: Optional[Dict[str, AssetVolConfig]] = None,
        default_target_daily_vol: float = 0.02,
        default_max_weight: float = 0.25,
        default_min_weight: float = 0.01,
        use_correlation_adjustment: bool = True,
        scaling_factor: float = 1.0
    ):
        """
        Initialize enhanced volatility targeting sizer.
        
        Args:
            portfolio_target_daily_vol: Overall portfolio target
            asset_configs: Per-asset configurations
            default_target_daily_vol: Default for unlisted assets
            default_max_weight: Default max weight
            default_min_weight: Default min weight
            use_correlation_adjustment: Adjust for correlations
            scaling_factor: Global scaling multiplier
        """
        self.portfolio_target_daily_vol = portfolio_target_daily_vol
        self.default_target_daily_vol = default_target_daily_vol
        self.default_max_weight = default_max_weight
        self.default_min_weight = default_min_weight
        self.use_correlation_adjustment = use_correlation_adjustment
        self.scaling_factor = scaling_factor
        
        # Per-asset configs
        self._asset_configs: Dict[str, AssetVolConfig] = {}
        if asset_configs:
            self._asset_configs.update(asset_configs)
        
        # Vol calculator
        self._vol_calculator = EnhancedVolatilityCalculator()
        self._asset_metrics: Dict[str, VolatilityMetrics] = {}
        
        # Correlation matrix (updated periodically)
        self._return_history: Dict[str, deque] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._corr_symbols: List[str] = []
        
        self._lock = threading.Lock()
        
        logger.info(
            f"EnhancedVolatilityTargetingSizer initialized: "
            f"portfolio_target={portfolio_target_daily_vol:.1%}, "
            f"correlation_adj={use_correlation_adjustment}"
        )
    
    def set_asset_config(
        self,
        symbol: str,
        config: AssetVolConfig
    ) -> None:
        """Set or update per-asset configuration."""
        with self._lock:
            self._asset_configs[symbol] = config
            logger.info(f"Asset config set for {symbol}: target_vol={config.target_daily_vol:.1%}")
    
    def get_asset_config(self, symbol: str) -> AssetVolConfig:
        """Get asset config, creating default if needed."""
        with self._lock:
            if symbol not in self._asset_configs:
                self._asset_configs[symbol] = AssetVolConfig(
                    target_daily_vol=self.default_target_daily_vol,
                    max_weight=self.default_max_weight,
                    min_weight=self.default_min_weight
                )
            return self._asset_configs[symbol]
    
    def update_volatility(
        self,
        symbol: str,
        current_price: float,
        previous_price: float
    ) -> VolatilityMetrics:
        """Update volatility for symbol."""
        metrics = self._vol_calculator.update(symbol, current_price, previous_price)
        
        with self._lock:
            self._asset_metrics[symbol] = metrics
            
            # Track returns for correlation
            if previous_price > 0:
                ret = (current_price - previous_price) / previous_price
                if symbol not in self._return_history:
                    self._return_history[symbol] = deque(maxlen=60)
                self._return_history[symbol].append(ret)
        
        return metrics
    
    def bulk_update_volatility(
        self,
        symbol: str,
        prices: List[float]
    ) -> VolatilityMetrics:
        """Bulk update volatility from price history."""
        metrics = self._vol_calculator.bulk_update(symbol, prices)
        
        with self._lock:
            self._asset_metrics[symbol] = metrics
            
            # Bulk add returns
            if symbol not in self._return_history:
                self._return_history[symbol] = deque(maxlen=60)
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0
                self._return_history[symbol].append(ret)
        
        return metrics
    
    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix from return history."""
        with self._lock:
            # Get symbols with enough data
            valid_symbols = [
                s for s, h in self._return_history.items() 
                if len(h) >= 20
            ]
            
            if len(valid_symbols) < 2:
                self._correlation_matrix = None
                return
            
            # Build return matrix
            min_len = min(len(self._return_history[s]) for s in valid_symbols)
            returns = np.array([
                list(self._return_history[s])[-min_len:]
                for s in valid_symbols
            ])
            
            # Calculate correlation
            self._correlation_matrix = np.corrcoef(returns)
            self._corr_symbols = valid_symbols
    
    def calculate_position(
        self,
        symbol: str,
        capital: float,
        base_weight: float = None,
        current_price: float = 0.0
    ) -> PositionSizeRecommendation:
        """
        Calculate vol-adjusted position size with per-asset targets.
        
        Args:
            symbol: Asset symbol
            capital: Total portfolio capital
            base_weight: Base allocation weight (uses equal weight if None)
            current_price: Current asset price
            
        Returns:
            PositionSizeRecommendation
        """
        # Get asset config
        config = self.get_asset_config(symbol)
        
        # Get metrics
        with self._lock:
            metrics = self._asset_metrics.get(symbol)
        
        if base_weight is None:
            base_weight = config.max_weight / 2  # Start at half max
        
        if metrics is None or metrics.n_observations < 10:
            return PositionSizeRecommendation(
                symbol=symbol,
                base_weight=base_weight,
                adjusted_weight=min(base_weight * 0.5, config.min_weight),
                position_usd=capital * base_weight * 0.5,
                vol_adjustment_factor=0.5,
                target_daily_vol=config.target_daily_vol,
                reason="Insufficient volatility data - using conservative sizing"
            )
        
        # v5.0.5: Use per-asset target
        asset_target_vol = config.target_daily_vol
        asset_current_vol = metrics.daily_vol
        
        # Apply regime scaling if enabled
        if config.use_regime_scaling:
            regime_scale = self._get_regime_scale(metrics)
            asset_target_vol *= regime_scale
        
        # Calculate vol adjustment
        if asset_current_vol > 0.001:
            vol_adjustment = asset_target_vol / asset_current_vol
        else:
            vol_adjustment = 1.0
        
        # Apply global scaling
        vol_adjustment *= self.scaling_factor * config.vol_scale_factor
        
        # Calculate adjusted weight
        adjusted_weight = base_weight * vol_adjustment
        
        # Apply caps and floors
        cap_applied = False
        floor_applied = False
        
        if adjusted_weight > config.max_weight:
            adjusted_weight = config.max_weight
            cap_applied = True
        
        if adjusted_weight < config.min_weight:
            adjusted_weight = config.min_weight
            floor_applied = True
        
        # Calculate position metrics
        position_usd = capital * adjusted_weight
        expected_daily_vol_usd = position_usd * asset_current_vol
        expected_daily_vol_pct = adjusted_weight * asset_current_vol
        
        # Actual vs target ratio
        actual_vs_target = asset_current_vol / asset_target_vol if asset_target_vol > 0 else 1.0
        
        # Build reason
        reasons = []
        if metrics.is_high_vol:
            reasons.append(f"High vol ({metrics.annual_vol:.0%}ann) -> reduced")
        elif metrics.is_low_vol:
            reasons.append(f"Low vol ({metrics.annual_vol:.0%}ann) -> increased")
        
        if cap_applied:
            reasons.append(f"Capped at {config.max_weight:.0%}")
        if floor_applied:
            reasons.append(f"Floor at {config.min_weight:.0%}")
        
        reasons.append(f"Target {asset_target_vol:.1%}d vs actual {asset_current_vol:.1%}d")
        
        return PositionSizeRecommendation(
            symbol=symbol,
            base_weight=base_weight,
            adjusted_weight=adjusted_weight,
            position_usd=position_usd,
            expected_daily_vol_usd=expected_daily_vol_usd,
            expected_daily_vol_pct=expected_daily_vol_pct,
            vol_adjustment_factor=vol_adjustment,
            cap_applied=cap_applied,
            floor_applied=floor_applied,
            target_daily_vol=asset_target_vol,
            actual_vs_target_ratio=actual_vs_target,
            reason="; ".join(reasons)
        )
    
    def _get_regime_scale(self, metrics: VolatilityMetrics) -> float:
        """Get regime-based scaling factor."""
        # In high vol, reduce target (tighter sizing)
        # In low vol, increase target (larger sizing ok)
        if metrics.vol_percentile > 90:
            return 0.7  # Extreme vol: reduce target
        elif metrics.vol_percentile > 75:
            return 0.85
        elif metrics.vol_percentile < 25:
            return 1.15
        return 1.0
    
    def calculate_portfolio_allocations(
        self,
        symbols: List[str],
        capital: float,
        base_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, PositionSizeRecommendation], PortfolioRiskState]:
        """
        Calculate vol-adjusted allocations for portfolio.
        
        Args:
            symbols: List of symbols
            capital: Total capital
            base_weights: Base weights per symbol (equal weight if None)
            
        Returns:
            (recommendations dict, portfolio risk state)
        """
        # Update correlation matrix
        if self.use_correlation_adjustment:
            self._update_correlation_matrix()
        
        # Calculate per-asset recommendations
        recommendations = {}
        
        for symbol in symbols:
            base = base_weights.get(symbol, 1.0/len(symbols)) if base_weights else 1.0/len(symbols)
            rec = self.calculate_position(symbol, capital, base)
            recommendations[symbol] = rec
        
        # Calculate portfolio volatility
        if self.use_correlation_adjustment and self._correlation_matrix is not None:
            portfolio_vol = self._calculate_portfolio_vol_with_correlation(recommendations)
        else:
            portfolio_vol = self._calculate_portfolio_vol_simple(recommendations)
        
        # Calculate diversification ratio
        sum_individual = sum(
            rec.adjusted_weight * self._asset_metrics.get(symbol, VolatilityMetrics(symbol=symbol)).daily_vol
            for symbol, rec in recommendations.items()
        )
        div_ratio = sum_individual / portfolio_vol if portfolio_vol > 0 else 1.0
        
        # Calculate marginal contributions
        marginal_contributions = {}
        for symbol, rec in recommendations.items():
            metrics = self._asset_metrics.get(symbol)
            if metrics:
                # Simplified marginal contribution
                mc = rec.adjusted_weight * metrics.daily_vol / (portfolio_vol + 1e-10)
                marginal_contributions[symbol] = mc
        
        # Build state
        state = PortfolioRiskState(
            target_daily_vol=self.portfolio_target_daily_vol,
            current_daily_vol=portfolio_vol,
            vol_utilization=portfolio_vol / self.portfolio_target_daily_vol if self.portfolio_target_daily_vol > 0 else 0,
            total_exposure_usd=sum(r.position_usd for r in recommendations.values()),
            gross_exposure_pct=sum(r.adjusted_weight for r in recommendations.values()),
            asset_allocations={s: r.adjusted_weight for s, r in recommendations.items()},
            correlation_adjusted_vol=portfolio_vol,
            diversification_ratio=div_ratio,
            marginal_contributions=marginal_contributions
        )
        
        return recommendations, state
    
    def _calculate_portfolio_vol_simple(
        self,
        recommendations: Dict[str, PositionSizeRecommendation]
    ) -> float:
        """Calculate portfolio vol ignoring correlations."""
        total_var = 0.0
        for symbol, rec in recommendations.items():
            metrics = self._asset_metrics.get(symbol, VolatilityMetrics(symbol=symbol))
            total_var += (rec.adjusted_weight * metrics.daily_vol) ** 2
        return np.sqrt(total_var)
    
    def _calculate_portfolio_vol_with_correlation(
        self,
        recommendations: Dict[str, PositionSizeRecommendation]
    ) -> float:
        """Calculate portfolio vol with correlation adjustment."""
        if self._correlation_matrix is None or len(self._corr_symbols) < 2:
            return self._calculate_portfolio_vol_simple(recommendations)
        
        # Build weight and vol vectors aligned with correlation matrix
        weights = []
        vols = []
        
        for symbol in self._corr_symbols:
            if symbol in recommendations:
                rec = recommendations[symbol]
                weights.append(rec.adjusted_weight)
                metrics = self._asset_metrics.get(symbol, VolatilityMetrics(symbol=symbol))
                vols.append(metrics.daily_vol)
            else:
                weights.append(0)
                vols.append(0)
        
        w = np.array(weights)
        v = np.array(vols)
        
        # Portfolio variance: w' * Σ * w where Σ = diag(v) * corr * diag(v)
        cov = np.outer(v, v) * self._correlation_matrix
        portfolio_var = w @ cov @ w
        
        return np.sqrt(max(0, portfolio_var))
    
    def get_risk_budget_remaining(
        self,
        current_positions: Dict[str, float]
    ) -> float:
        """Calculate remaining risk budget."""
        total_risk = 0.0
        for symbol, position_usd in current_positions.items():
            metrics = self._asset_metrics.get(symbol)
            if metrics:
                total_risk += (position_usd * metrics.daily_vol) ** 2
        
        portfolio_vol = np.sqrt(total_risk)
        utilization = portfolio_vol / self.portfolio_target_daily_vol if self.portfolio_target_daily_vol > 0 else 0
        
        return max(0, 1 - utilization)
    
    def get_all_volatility_metrics(self) -> Dict[str, Dict]:
        """Get all tracked volatility metrics."""
        with self._lock:
            return {
                symbol: metrics.to_dict()
                for symbol, metrics in self._asset_metrics.items()
            }
    
    def get_all_asset_configs(self) -> Dict[str, Dict]:
        """Get all asset configurations."""
        with self._lock:
            return {
                symbol: {
                    "target_daily_vol": config.target_daily_vol,
                    "target_annual_vol": config.target_annual_vol,
                    "max_weight": config.max_weight,
                    "min_weight": config.min_weight
                }
                for symbol, config in self._asset_configs.items()
            }


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class VolatilityTargetingSizer(EnhancedVolatilityTargetingSizer):
    """
    Backward compatible volatility sizer.
    
    Provides v4.5 interface using v5.0.5 implementation.
    """
    
    def __init__(
        self,
        target_daily_vol: float = 0.02,
        target_annual_vol: float = None,  # Calculated from daily
        max_position_weight: float = 0.25,
        min_position_weight: float = 0.01,
        scaling_factor: float = 1.0
    ):
        # Calculate annual from daily if not provided
        if target_annual_vol is None:
            target_annual_vol = target_daily_vol * np.sqrt(252)
        
        super().__init__(
            portfolio_target_daily_vol=target_daily_vol,
            default_target_daily_vol=target_daily_vol,
            default_max_weight=max_position_weight,
            default_min_weight=min_position_weight,
            scaling_factor=scaling_factor
        )
        
        # v4.5 compatible attributes
        self.target_daily_vol = target_daily_vol
        self.target_annual_vol = target_annual_vol
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight


# For direct import compatibility
VolatilityCalculator = EnhancedVolatilityCalculator


# =============================================================================
# SINGLETON
# =============================================================================

_sizer: Optional[EnhancedVolatilityTargetingSizer] = None


def get_volatility_sizer(
    target_daily_vol: float = 0.02,
    asset_configs: Optional[Dict[str, AssetVolConfig]] = None
) -> EnhancedVolatilityTargetingSizer:
    """Get singleton sizer instance."""
    global _sizer
    if _sizer is None:
        _sizer = EnhancedVolatilityTargetingSizer(
            portfolio_target_daily_vol=target_daily_vol,
            asset_configs=asset_configs
        )
    elif (
        _sizer.portfolio_target_daily_vol != target_daily_vol
        or _sizer._asset_configs != (asset_configs or {})
    ):
        logger.info("[VOL_TARGET] Constructor inputs changed; refreshing singleton instance")
        _sizer = EnhancedVolatilityTargetingSizer(
            portfolio_target_daily_vol=target_daily_vol,
            asset_configs=asset_configs
        )
    return _sizer


def reset_volatility_sizer():
    """Reset singleton sizer instance (for testing)."""
    global _sizer
    _sizer = None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("HMATS v5.0.5 - Enhanced Volatility Targeting Test")
    print("=" * 70)
    
    # Create sizer with per-asset configs
    asset_configs = {
        "SOL/USD": AssetVolConfig(target_daily_vol=0.02, max_weight=0.30),
        "BTC/USD": AssetVolConfig(target_daily_vol=0.015, max_weight=0.40),
        "ETH/USD": AssetVolConfig(target_daily_vol=0.025, max_weight=0.35),
    }
    
    sizer = EnhancedVolatilityTargetingSizer(
        portfolio_target_daily_vol=0.02,
        asset_configs=asset_configs,
        use_correlation_adjustment=True
    )
    
    # Simulate price data
    np.random.seed(42)
    n_days = 60
    
    # SOL: High volatility (5% daily)
    sol_returns = np.random.randn(n_days) * 0.05
    sol_prices = 100 * np.cumprod(1 + sol_returns)
    
    # BTC: Medium volatility (3% daily)
    btc_returns = np.random.randn(n_days) * 0.03
    btc_prices = 40000 * np.cumprod(1 + btc_returns)
    
    # ETH: Lower volatility (2% daily)
    eth_returns = np.random.randn(n_days) * 0.02
    eth_prices = 2500 * np.cumprod(1 + eth_returns)
    
    # Bulk update
    sizer.bulk_update_volatility("SOL/USD", list(sol_prices))
    sizer.bulk_update_volatility("BTC/USD", list(btc_prices))
    sizer.bulk_update_volatility("ETH/USD", list(eth_prices))
    
    # Print volatility metrics
    print("\n1. Per-Asset Volatility Metrics:")
    for symbol, metrics in sizer.get_all_volatility_metrics().items():
        print(f"   {symbol}:")
        print(f"      Daily vol: {metrics['daily_vol']:.2%}")
        print(f"      Annual vol: {metrics['annual_vol']:.0%}")
        print(f"      Vol percentile: {metrics['vol_percentile']:.0f}%")
        print(f"      Regime: {metrics['vol_regime']}")
    
    # Print asset configs
    print("\n2. Per-Asset Target Configs:")
    for symbol, config in sizer.get_all_asset_configs().items():
        print(f"   {symbol}: target={config['target_daily_vol']:.1%}d, max={config['max_weight']:.0%}")
    
    # Calculate portfolio allocation
    print("\n3. Portfolio Allocation ($100k):")
    capital = 100_000
    
    recommendations, state = sizer.calculate_portfolio_allocations(
        symbols=["SOL/USD", "BTC/USD", "ETH/USD"],
        capital=capital,
        base_weights={"SOL/USD": 0.25, "BTC/USD": 0.50, "ETH/USD": 0.25}
    )
    
    for symbol, rec in recommendations.items():
        print(f"\n   {symbol}:")
        print(f"      Base weight: {rec.base_weight:.1%}")
        print(f"      Adjusted weight: {rec.adjusted_weight:.1%}")
        print(f"      Position: ${rec.position_usd:,.0f}")
        print(f"      Vol adjustment: {rec.vol_adjustment_factor:.2f}x")
        print(f"      Target vs Actual: {rec.actual_vs_target_ratio:.2f}x")
        print(f"      Reason: {rec.reason}")
    
    print(f"\n4. Portfolio Risk State:")
    print(f"   Target daily vol: {state.target_daily_vol:.1%}")
    print(f"   Current daily vol: {state.current_daily_vol:.2%}")
    print(f"   Vol utilization: {state.vol_utilization:.0%}")
    print(f"   Diversification ratio: {state.diversification_ratio:.2f}")
    print(f"   Total exposure: ${state.total_exposure_usd:,.0f}")
    print(f"   Gross exposure: {state.gross_exposure_pct:.0%}")
    
    # Test backward compatibility
    print("\n5. Backward Compatibility Test:")
    legacy_sizer = VolatilityTargetingSizer(target_daily_vol=0.02)
    legacy_sizer.bulk_update_volatility("TEST/USD", list(sol_prices))
    rec = legacy_sizer.calculate_position("TEST/USD", 50000, 0.2)
    print(f"   Legacy sizer result: weight={rec.adjusted_weight:.1%}, pos=${rec.position_usd:,.0f}")
    
    print("\n" + "=" * 70)
    print("v5.0.5 Volatility Targeting Test Complete!")
    print("=" * 70)

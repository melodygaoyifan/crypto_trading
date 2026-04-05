"""
================================================================================
HMATS v5.1 - STRATEGY ALLOCATOR
================================================================================
Version: 5.1.0
Purpose: Orchestrate strategy allocation based on regime and performance metrics

This module provides the StrategyAllocator class which manages strategy
weighting based on Information Ratio, volatility targeting, and drawdown
management.

NOTE: The primary implementation is in kraken_quant_agent.py. This module
provides a standalone export for direct imports.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regime classification."""
    BEAR = "BEAR"
    BULL = "BULL"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


@dataclass
class StrategyState:
    """State tracking for a single strategy."""
    name: str
    regime: Regime
    information_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.5
    avg_pnl: float = 0.0
    trade_count: int = 0
    enabled: bool = True
    
    # Rolling performance
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_performance(self, pnl: float):
        """Update rolling performance metrics."""
        self.pnl_history.append(pnl)
        if len(self.pnl_history) > 0:
            self.avg_pnl = np.mean(self.pnl_history)
            std = np.std(self.pnl_history)
            if std > 0:
                self.information_ratio = self.avg_pnl / std
                self.sharpe_ratio = self.information_ratio * np.sqrt(252)
        self.trade_count += 1


@dataclass
class AllocationResult:
    """Result of strategy allocation."""
    regime: Regime
    weights: Dict[str, float]
    vol_adjustment: float
    total_allocation: float
    is_killed: bool
    reason: Optional[str] = None


class StrategyAllocator:
    """
    Orchestrates all strategies based on regime.
    
    ALLOCATION LOGIC:
    -----------------
    
    1. Regime Selection: Upstream RegimeAgent provides current regime
    
    2. Information Ratio (IR) Softmax Weighting:
       
       w_i = exp(IR_i / τ) / Σ exp(IR_j / τ)
       
       where:
           IR_i = Information Ratio of strategy i (48h rolling)
           τ = Temperature (lower = more aggressive toward best)
    
    3. Volatility Target:
       If 1H realized vol > 150% of normal -> reduce all sizes by 50%
    
    4. Global Drawdown Kill-Switch:
       If portfolio drawdown > threshold -> flatten all
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        vol_reduction_threshold: float = 1.5,  # 150% of normal vol
        max_drawdown: float = 0.15,  # 15% kill switch
        initial_capital: float = 10000.0,  # [T19] was 100K; match actual $10K account
    ):
        self.temperature = temperature
        self.vol_reduction_threshold = vol_reduction_threshold
        self.max_drawdown = max_drawdown
        
        # Strategy registry
        self.strategies: Dict[str, StrategyState] = {}
        
        # Portfolio state
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.current_drawdown = 0.0
        
        # Volatility tracking
        self.vol_buffer: deque = deque(maxlen=100)
        self.normal_vol = 0.02  # 2% daily vol baseline
        
        # Kill switch state
        self.killed = False
        self.kill_reason: Optional[str] = None
        
        logger.info(f"StrategyAllocator initialized: temp={temperature}, max_dd={max_drawdown}")
    
    def register_strategy(
        self,
        name: str,
        regime: Regime,
        initial_ir: float = 0.0
    ):
        """Register a strategy for allocation."""
        self.strategies[name] = StrategyState(
            name=name,
            regime=regime,
            information_ratio=initial_ir
        )
        logger.debug(f"Registered strategy: {name} for regime {regime.value}")
    
    def get_strategies_for_regime(self, regime: Regime) -> List[StrategyState]:
        """Get all strategies applicable to a regime."""
        return [s for s in self.strategies.values() if s.regime == regime and s.enabled]
    
    def calculate_ir_weights(self, regime: Regime) -> Dict[str, float]:
        """
        Calculate Information Ratio Softmax weights.
        
        w_i = exp(IR_i / τ) / Σ exp(IR_j / τ)
        """
        strategies = self.get_strategies_for_regime(regime)
        
        if not strategies:
            return {}
        
        irs = np.array([s.information_ratio for s in strategies])
        
        # Softmax with temperature
        exp_irs = np.exp(irs / self.temperature)
        weights = exp_irs / (np.sum(exp_irs) + 1e-10)
        
        return {s.name: float(w) for s, w in zip(strategies, weights)}
    
    def calculate_vol_adjustment(self, current_vol: float) -> float:
        """
        Calculate position size adjustment based on volatility.
        
        Returns multiplier [0.5, 1.0]
        """
        vol_ratio = current_vol / (self.normal_vol + 1e-10)
        
        if vol_ratio > self.vol_reduction_threshold:
            return 0.5
        else:
            return 1.0
    
    def update_portfolio_state(self, pnl: float) -> bool:
        """
        Update portfolio value and check drawdown.
        
        Returns True if kill switch triggered.
        """
        self.portfolio_value += pnl
        
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        if self.current_drawdown > self.max_drawdown:
            self.killed = True
            self.kill_reason = f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}"
            logger.warning(f"KILL SWITCH TRIGGERED: {self.kill_reason}")
            return True
        
        return False
    
    def allocate(
        self,
        regime: Regime,
        current_vol: float,
        pnl_since_last: float = 0.0
    ) -> AllocationResult:
        """
        Main allocation method.
        
        Args:
            regime: Current market regime
            current_vol: Current realized volatility
            pnl_since_last: PnL since last allocation
            
        Returns:
            AllocationResult with weights and adjustments
        """
        # Update portfolio state
        kill_triggered = self.update_portfolio_state(pnl_since_last)
        
        if self.killed:
            return AllocationResult(
                regime=regime,
                weights={},
                vol_adjustment=0.0,
                total_allocation=0.0,
                is_killed=True,
                reason=self.kill_reason
            )
        
        # Calculate weights
        weights = self.calculate_ir_weights(regime)
        
        # Apply volatility adjustment
        vol_adj = self.calculate_vol_adjustment(current_vol)
        
        # Calculate total allocation
        total = sum(weights.values()) * vol_adj
        
        return AllocationResult(
            regime=regime,
            weights=weights,
            vol_adjustment=vol_adj,
            total_allocation=total,
            is_killed=False
        )
    
    def reset_kill_switch(self):
        """Reset the kill switch (use with caution)."""
        self.killed = False
        self.kill_reason = None
        self.peak_value = self.portfolio_value
        logger.info("Kill switch reset")
    
    def get_status(self) -> Dict:
        """Get allocator status summary."""
        return {
            "portfolio_value": self.portfolio_value,
            "peak_value": self.peak_value,
            "current_drawdown": self.current_drawdown,
            "killed": self.killed,
            "kill_reason": self.kill_reason,
            "num_strategies": len(self.strategies),
            "temperature": self.temperature,
            "vol_threshold": self.vol_reduction_threshold,
            "max_drawdown": self.max_drawdown
        }


# Singleton instance
_allocator_instance: Optional[StrategyAllocator] = None


def get_strategy_allocator(
    temperature: float = 0.5,
    max_drawdown: float = 0.15
) -> StrategyAllocator:
    """Get or create the global StrategyAllocator instance."""
    global _allocator_instance
    if _allocator_instance is None:
        _allocator_instance = StrategyAllocator(
            temperature=temperature,
            max_drawdown=max_drawdown
        )
    elif (
        _allocator_instance.temperature != temperature
        or _allocator_instance.max_drawdown != max_drawdown
    ):
        logger.info("[STRATEGY_ALLOCATOR] Constructor inputs changed; refreshing singleton instance")
        _allocator_instance = StrategyAllocator(
            temperature=temperature,
            max_drawdown=max_drawdown
        )
    return _allocator_instance


def reset_strategy_allocator():
    """Reset the global allocator instance."""
    global _allocator_instance
    _allocator_instance = None


# Exports
__all__ = [
    "Regime",
    "StrategyState",
    "AllocationResult",
    "StrategyAllocator",
    "get_strategy_allocator",
    "reset_strategy_allocator",
]

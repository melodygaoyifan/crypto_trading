"""
================================================================================
OVERRIDE 2: FAST CORRELATION SENTINEL (Exit-Only)
================================================================================
Version: 3.3-HR
Purpose: Faster correlation detection during OPPORTUNITY for quicker exits

RULE:
    Add a fast correlation sentinel:
    - 30-minute rolling window (vs 4H standard)
    - Only active during OPPORTUNITY mode
    - EXIT-ONLY: Cannot veto entries
    - Trigger fast exit when short-term correlation > 0.90

WHY THIS EXISTS:
    The standard 4H correlation window is too slow to detect rapid correlation
    spikes during volatile moves. We want to STAY IN the trade (aggressive entry)
    but EXIT FAST when correlation collapses (defensive exit).

CRITICAL DISTINCTION:
    - Standard correlation (4H): Used for entry decisions, NO_TRADE triggers
    - Fast correlation (30min): Used ONLY for exit acceleration in OPPORTUNITY
    - Fast correlation does NOT block new entries
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from collections import deque

from market.high_risk_config import get_hr_config, OverrideMode

logger = logging.getLogger(__name__)


@dataclass
class FastCorrelationSnapshot:
    """Fast correlation state snapshot."""
    
    # Correlations
    btc_eth: float = 0.0
    btc_sol: float = 0.0
    eth_sol: float = 0.0
    average: float = 0.0
    
    # Exit signal
    exit_triggered: bool = False
    exit_reason: str = ""
    
    # Comparison to standard
    standard_correlation: float = 0.0
    correlation_divergence: float = 0.0  # fast - standard
    
    # Window info
    window_minutes: int = 30
    data_points: int = 0
    
    # Metadata
    is_active: bool = False  # Only active in OPPORTUNITY
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "average": self.average,
            "exit_triggered": self.exit_triggered,
            "exit_reason": self.exit_reason,
            "standard_correlation": self.standard_correlation,
            "divergence": self.correlation_divergence,
            "is_active": self.is_active,
            "data_points": self.data_points,
        }


@dataclass
class PriceReturn:
    """Single return observation."""
    btc: float
    eth: float
    sol: float
    timestamp: datetime


class FastCorrelationSentinel:
    """
    Fast correlation detector for exit acceleration.
    
    CRITICAL: This is EXIT-ONLY.
    - Does NOT veto entries
    - Does NOT trigger NO_TRADE
    - ONLY signals that current positions should be reduced
    
    USAGE:
        sentinel = FastCorrelationSentinel()
        
        # Feed price data (1-minute intervals)
        sentinel.add_prices(btc_price, eth_price, sol_price)
        
        # Check for exit signal (only meaningful in OPPORTUNITY)
        snapshot = sentinel.get_snapshot(
            system_mode="OPPORTUNITY",
            standard_correlation=0.75,  # From 4H window
        )
        
        if snapshot.exit_triggered:
            # Accelerate exit, but do NOT block new entries
            pass
    """
    
    def __init__(self):
        self.config = get_hr_config()
        self.cfg = self.config.fast_correlation_config
        
        # Price return buffer
        # 30 minutes at 1-minute intervals = 30 data points
        self.window_size = self.cfg["window_minutes"]
        self.returns_buffer: deque = deque(maxlen=self.window_size)
        
        # Last prices for return calculation
        self._last_prices: Optional[Tuple[float, float, float]] = None
        
        # Current snapshot
        self.snapshot = FastCorrelationSnapshot()
        
        logger.info(f"FastCorrelationSentinel initialized (window={self.window_size}min)")
    
    def add_prices(
        self,
        btc_price: float,
        eth_price: float,
        sol_price: float,
        timestamp: Optional[datetime] = None,
    ):
        """Add new price observation and calculate returns."""
        ts = timestamp or datetime.utcnow()
        
        if self._last_prices is None:
            self._last_prices = (btc_price, eth_price, sol_price)
            return
        
        # Calculate log returns
        last_btc, last_eth, last_sol = self._last_prices
        
        btc_ret = np.log(btc_price / last_btc) if last_btc > 0 else 0.0
        eth_ret = np.log(eth_price / last_eth) if last_eth > 0 else 0.0
        sol_ret = np.log(sol_price / last_sol) if last_sol > 0 else 0.0
        
        self.returns_buffer.append(PriceReturn(
            btc=btc_ret,
            eth=eth_ret,
            sol=sol_ret,
            timestamp=ts,
        ))
        
        self._last_prices = (btc_price, eth_price, sol_price)
    
    def get_snapshot(
        self,
        system_mode: str,
        standard_correlation: float = 0.0,
    ) -> FastCorrelationSnapshot:
        """
        Get current fast correlation snapshot.
        
        Args:
            system_mode: Current system mode (OPPORTUNITY, NORMAL, NO_TRADE)
            standard_correlation: Correlation from 4H window for comparison
        
        Returns:
            FastCorrelationSnapshot with exit signal if triggered
        """
        
        # Check if override is enabled
        if self.config.fast_correlation_override != OverrideMode.ENABLED:
            return FastCorrelationSnapshot(is_active=False)
        
        # Only active in OPPORTUNITY mode
        is_active = (
            self.cfg["only_in_opportunity"] and 
            system_mode == "OPPORTUNITY"
        )
        
        self.snapshot = FastCorrelationSnapshot(
            is_active=is_active,
            standard_correlation=standard_correlation,
            window_minutes=self.window_size,
        )
        
        if not is_active:
            return self.snapshot
        
        # Need minimum data
        if len(self.returns_buffer) < self.window_size // 2:
            self.snapshot.data_points = len(self.returns_buffer)
            return self.snapshot
        
        # Calculate correlations
        recent = list(self.returns_buffer)
        
        btc_rets = np.array([r.btc for r in recent])
        eth_rets = np.array([r.eth for r in recent])
        sol_rets = np.array([r.sol for r in recent])
        
        btc_eth = self._safe_corr(btc_rets, eth_rets)
        btc_sol = self._safe_corr(btc_rets, sol_rets)
        eth_sol = self._safe_corr(eth_rets, sol_rets)
        
        avg_corr = (btc_eth + btc_sol + eth_sol) / 3.0
        
        self.snapshot.btc_eth = btc_eth
        self.snapshot.btc_sol = btc_sol
        self.snapshot.eth_sol = eth_sol
        self.snapshot.average = avg_corr
        self.snapshot.data_points = len(recent)
        self.snapshot.correlation_divergence = avg_corr - standard_correlation
        
        # Check exit trigger
        exit_threshold = self.cfg["exit_threshold"]
        
        if avg_corr >= exit_threshold:
            self.snapshot.exit_triggered = True
            self.snapshot.exit_reason = (
                f"Fast correlation {avg_corr:.3f} >= {exit_threshold} "
                f"(30min window, standard={standard_correlation:.3f})"
            )
            
            logger.warning(
                f"FAST CORRELATION EXIT SIGNAL: {avg_corr:.3f} "
                f"(divergence={self.snapshot.correlation_divergence:+.3f})"
            )
        
        return self.snapshot
    
    def _safe_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation with safety checks."""
        if len(x) < 2:
            return 0.0
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        try:
            corr = np.corrcoef(x, y)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    def should_accelerate_exit(
        self,
        system_mode: str,
        standard_correlation: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Simple check for exit acceleration.
        
        CRITICAL: This is EXIT-ONLY. It does NOT block entries.
        
        Returns:
            (should_exit, reason)
        """
        snapshot = self.get_snapshot(system_mode, standard_correlation)
        
        if not snapshot.is_active:
            return False, "Fast sentinel inactive (not OPPORTUNITY)"
        
        if snapshot.exit_triggered:
            return True, snapshot.exit_reason
        
        return False, ""
    
    def get_correlation_trend(self) -> str:
        """Get short-term correlation trend direction."""
        if len(self.returns_buffer) < 10:
            return "INSUFFICIENT_DATA"
        
        # Compare first half to second half of window
        recent = list(self.returns_buffer)
        mid = len(recent) // 2
        
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        def avg_corr(data: List[PriceReturn]) -> float:
            btc = np.array([r.btc for r in data])
            sol = np.array([r.sol for r in data])
            return self._safe_corr(btc, sol)
        
        first_corr = avg_corr(first_half)
        second_corr = avg_corr(second_half)
        
        diff = second_corr - first_corr
        
        if diff > 0.05:
            return "RISING"
        elif diff < -0.05:
            return "FALLING"
        else:
            return "STABLE"
    
    def reset(self):
        """Reset sentinel state."""
        self.returns_buffer.clear()
        self._last_prices = None
        self.snapshot = FastCorrelationSnapshot()


# Singleton
_fast_sentinel: Optional[FastCorrelationSentinel] = None


def get_fast_correlation_sentinel() -> FastCorrelationSentinel:
    """Get singleton instance."""
    global _fast_sentinel
    if _fast_sentinel is None:
        _fast_sentinel = FastCorrelationSentinel()
    return _fast_sentinel


def reset_fast_correlation_sentinel():
    """Reset singleton."""
    global _fast_sentinel
    _fast_sentinel = None

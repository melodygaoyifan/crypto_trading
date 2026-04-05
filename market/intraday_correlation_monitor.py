"""
================================================================================
HMATS v4.0 - INTRADAY CORRELATION MONITOR
================================================================================
Version: 4.0.0
Purpose: Real-time intraday correlation monitoring for cross-asset risk

RESTORED FROM: v3.3-STABILIZED (intraday_correlation_monitor.py)

KEY FEATURES:
    1. Rolling correlation calculation (5m, 15m, 1h windows)
    2. Correlation spike detection
    3. Regime correlation vs intraday divergence
    4. Cross-asset pair monitoring (BTC-ETH, BTC-SOL, ETH-SOL)
    5. Correlation crisis early warning

CRITICAL THRESHOLDS:
    - Warning: >= 0.88
    - Danger: >= 0.92
    - Crisis: >= 0.95 (triggers NO_TRADE)
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class CorrelationLevel(Enum):
    """Correlation risk level."""
    NORMAL = "NORMAL"           # < 0.80
    ELEVATED = "ELEVATED"       # 0.80-0.87
    WARNING = "WARNING"         # 0.88-0.91
    DANGER = "DANGER"           # 0.92-0.94
    CRISIS = "CRISIS"           # >= 0.95


@dataclass
class CorrelationPair:
    """Correlation data for an asset pair."""
    pair: str  # e.g., "BTC-ETH"
    asset_a: str
    asset_b: str
    
    # Rolling correlations
    correlation_5m: float = 0.0
    correlation_15m: float = 0.0
    correlation_1h: float = 0.0
    correlation_4h: float = 0.0
    
    # Levels
    level: CorrelationLevel = CorrelationLevel.NORMAL
    
    # Change detection
    correlation_change_1h: float = 0.0
    spike_detected: bool = False
    
    # Timestamps
    last_update: Optional[datetime] = None


@dataclass
class CorrelationState:
    """Aggregated correlation state."""
    max_correlation: float = 0.0
    worst_pair: str = ""
    overall_level: CorrelationLevel = CorrelationLevel.NORMAL
    pairs: Dict[str, CorrelationPair] = field(default_factory=dict)
    crisis_triggered: bool = False
    warning_triggered: bool = False
    
    # Trend
    correlation_trending_up: bool = False
    hours_at_elevated: float = 0.0


@dataclass
class CorrelationConfig:
    """Configuration for correlation monitor."""
    
    # Windows (in minutes)
    window_5m: int = 5
    window_15m: int = 15
    window_1h: int = 60
    window_4h: int = 240
    
    # Thresholds
    threshold_warning: float = 0.88
    threshold_danger: float = 0.92
    threshold_crisis: float = 0.95
    
    # Spike detection
    spike_change_threshold: float = 0.10  # 10% change in 1h
    
    # Time-based
    max_hours_at_elevated: float = 4.0  # Warn if elevated for 4h+


class IntradayCorrelationMonitor:
    """
    Monitors intraday correlation for cross-asset risk.
    
    From v3.3-STABILIZED: Essential for detecting correlation regime shifts.
    """
    
    ASSET_PAIRS = [
        ("BTC", "ETH"),
        ("BTC", "SOL"),
        ("ETH", "SOL"),
    ]
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self.state = CorrelationState()
        
        # Price histories (deques with timestamps)
        self._price_history: Dict[str, deque] = {
            "BTC": deque(maxlen=1000),
            "ETH": deque(maxlen=1000),
            "SOL": deque(maxlen=1000),
        }
        
        # Initialize pairs
        for asset_a, asset_b in self.ASSET_PAIRS:
            pair_name = f"{asset_a}-{asset_b}"
            self.state.pairs[pair_name] = CorrelationPair(
                pair=pair_name,
                asset_a=asset_a,
                asset_b=asset_b,
            )
        
        self._elevated_since: Optional[datetime] = None
        
        logger.info("[v4.0] IntradayCorrelationMonitor initialized")
    
    def update_prices(
        self,
        btc_price: float,
        eth_price: float,
        sol_price: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Update price histories.
        
        Should be called every tick (e.g., every 200ms or 1s).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self._price_history["BTC"].append((timestamp, btc_price))
        self._price_history["ETH"].append((timestamp, eth_price))
        self._price_history["SOL"].append((timestamp, sol_price))
    
    def compute_correlations(self) -> CorrelationState:
        """
        Compute all correlation metrics.
        
        Returns updated CorrelationState.
        """
        now = datetime.now(timezone.utc)
        
        for pair_name, pair in self.state.pairs.items():
            # Get returns for each window
            returns_a_5m = self._get_returns(pair.asset_a, self.config.window_5m)
            returns_b_5m = self._get_returns(pair.asset_b, self.config.window_5m)
            
            returns_a_15m = self._get_returns(pair.asset_a, self.config.window_15m)
            returns_b_15m = self._get_returns(pair.asset_b, self.config.window_15m)
            
            returns_a_1h = self._get_returns(pair.asset_a, self.config.window_1h)
            returns_b_1h = self._get_returns(pair.asset_b, self.config.window_1h)
            
            returns_a_4h = self._get_returns(pair.asset_a, self.config.window_4h)
            returns_b_4h = self._get_returns(pair.asset_b, self.config.window_4h)
            
            # Compute correlations
            prev_1h = pair.correlation_1h
            
            pair.correlation_5m = self._compute_correlation(returns_a_5m, returns_b_5m)
            pair.correlation_15m = self._compute_correlation(returns_a_15m, returns_b_15m)
            pair.correlation_1h = self._compute_correlation(returns_a_1h, returns_b_1h)
            pair.correlation_4h = self._compute_correlation(returns_a_4h, returns_b_4h)
            
            # Change detection
            pair.correlation_change_1h = pair.correlation_1h - prev_1h
            pair.spike_detected = abs(pair.correlation_change_1h) >= self.config.spike_change_threshold
            
            # Level classification
            pair.level = self._classify_level(pair.correlation_1h)
            pair.last_update = now
        
        # Aggregate state
        self._aggregate_state()
        
        return self.state
    
    def _get_returns(self, asset: str, window_minutes: int) -> np.ndarray:
        """Get log returns for asset over window."""
        
        history = self._price_history.get(asset, deque())
        if len(history) < 10:
            return np.array([])
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        
        # Filter by time
        prices = [(ts, p) for ts, p in history if ts >= cutoff]
        
        if len(prices) < 10:
            return np.array([])
        
        price_array = np.array([p for _, p in prices])
        
        # Log returns
        returns = np.diff(np.log(price_array + 1e-10))
        
        return returns
    
    def _compute_correlation(self, returns_a: np.ndarray, returns_b: np.ndarray) -> float:
        """Compute Pearson correlation between two return series."""
        
        if len(returns_a) < 10 or len(returns_b) < 10:
            return 0.0
        
        # Ensure same length
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]
        
        # Handle edge cases
        std_a = np.std(returns_a)
        std_b = np.std(returns_b)
        
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(returns_a, returns_b)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)
    
    def _classify_level(self, correlation: float) -> CorrelationLevel:
        """Classify correlation level."""
        
        abs_corr = abs(correlation)
        
        if abs_corr >= self.config.threshold_crisis:
            return CorrelationLevel.CRISIS
        elif abs_corr >= self.config.threshold_danger:
            return CorrelationLevel.DANGER
        elif abs_corr >= self.config.threshold_warning:
            return CorrelationLevel.WARNING
        elif abs_corr >= 0.80:
            return CorrelationLevel.ELEVATED
        else:
            return CorrelationLevel.NORMAL
    
    def _aggregate_state(self):
        """Aggregate pair states into overall state."""
        
        now = datetime.now(timezone.utc)
        
        # Find worst pair
        max_corr = 0.0
        worst_pair = ""
        worst_level = CorrelationLevel.NORMAL
        
        for pair_name, pair in self.state.pairs.items():
            if abs(pair.correlation_1h) > max_corr:
                max_corr = abs(pair.correlation_1h)
                worst_pair = pair_name
                worst_level = pair.level
        
        self.state.max_correlation = max_corr
        self.state.worst_pair = worst_pair
        self.state.overall_level = worst_level
        
        # Crisis/warning triggers
        self.state.crisis_triggered = worst_level == CorrelationLevel.CRISIS
        self.state.warning_triggered = worst_level in [
            CorrelationLevel.WARNING, 
            CorrelationLevel.DANGER, 
            CorrelationLevel.CRISIS
        ]
        
        # Track elevated duration
        if worst_level != CorrelationLevel.NORMAL:
            if self._elevated_since is None:
                self._elevated_since = now
            self.state.hours_at_elevated = (now - self._elevated_since).total_seconds() / 3600
        else:
            self._elevated_since = None
            self.state.hours_at_elevated = 0.0
        
        # Trend detection
        trend_count = sum(1 for p in self.state.pairs.values() if p.correlation_change_1h > 0)
        self.state.correlation_trending_up = trend_count >= 2
    
    def should_trigger_no_trade(self) -> Tuple[bool, str]:
        """
        Check if NO_TRADE should be triggered due to correlation.
        
        Returns:
            (should_trigger, reason)
        """
        
        if self.state.crisis_triggered:
            return (
                True,
                f"Correlation crisis: {self.state.worst_pair} at {self.state.max_correlation:.2f}"
            )
        
        # Extended elevated warning
        if (self.state.overall_level in [CorrelationLevel.WARNING, CorrelationLevel.DANGER]
            and self.state.hours_at_elevated >= self.config.max_hours_at_elevated):
            return (
                True,
                f"Extended elevated correlation: {self.state.hours_at_elevated:.1f}h at {self.state.overall_level.value}"
            )
        
        return (False, "")
    
    def get_exposure_adjustment(self) -> float:
        """
        Get exposure adjustment factor based on correlation.
        
        Returns multiplier (0.0 to 1.0).
        """
        
        level = self.state.overall_level
        
        adjustments = {
            CorrelationLevel.NORMAL: 1.0,
            CorrelationLevel.ELEVATED: 0.9,
            CorrelationLevel.WARNING: 0.7,
            CorrelationLevel.DANGER: 0.5,
            CorrelationLevel.CRISIS: 0.0,  # Block
        }
        
        return adjustments.get(level, 1.0)
    
    def get_proof_log(self) -> str:
        """Generate proof log for correlation state."""
        
        pair_strs = []
        for name, pair in self.state.pairs.items():
            pair_strs.append(f"{name}={pair.correlation_1h:.2f}")
        
        return (
            f"[CORRELATION] level={self.state.overall_level.value} | "
            f"max={self.state.max_correlation:.2f} ({self.state.worst_pair}) | "
            f"{', '.join(pair_strs)} | "
            f"hours_elevated={self.state.hours_at_elevated:.1f} | "
            f"trending_up={self.state.correlation_trending_up}"
        )
    
    def get_detailed_report(self) -> Dict:
        """Get detailed correlation report."""
        
        return {
            "overall": {
                "level": self.state.overall_level.value,
                "max_correlation": self.state.max_correlation,
                "worst_pair": self.state.worst_pair,
                "crisis": self.state.crisis_triggered,
                "hours_elevated": self.state.hours_at_elevated,
            },
            "pairs": {
                name: {
                    "5m": pair.correlation_5m,
                    "15m": pair.correlation_15m,
                    "1h": pair.correlation_1h,
                    "4h": pair.correlation_4h,
                    "level": pair.level.value,
                    "spike": pair.spike_detected,
                }
                for name, pair in self.state.pairs.items()
            }
        }


# =============================================================================
# FAST CORRELATION CHECKER (from override_fast_correlation.py)
# =============================================================================

class FastCorrelationChecker:
    """
    Fast correlation check for execution decisions.
    
    From v3.3-STABILIZED override_fast_correlation.py
    Uses 30-second rolling window for real-time checks.
    """
    
    FAST_WINDOW_SECONDS = 30
    EXIT_THRESHOLD = 0.90
    
    def __init__(self):
        self._recent_prices: Dict[str, List[Tuple[datetime, float]]] = {
            "BTC": [],
            "ETH": [],
            "SOL": [],
        }
    
    def update(self, asset: str, price: float, timestamp: Optional[datetime] = None):
        """Update fast price cache."""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self._recent_prices[asset].append((timestamp, price))
        
        # Clean old entries
        cutoff = timestamp - timedelta(seconds=self.FAST_WINDOW_SECONDS)
        self._recent_prices[asset] = [
            (ts, p) for ts, p in self._recent_prices[asset] if ts >= cutoff
        ]
    
    def check_fast_correlation(self, asset_a: str, asset_b: str) -> Tuple[bool, float]:
        """
        Check 30-second correlation for immediate exit decision.
        
        Returns:
            (should_exit, correlation)
        """
        
        prices_a = self._recent_prices.get(asset_a, [])
        prices_b = self._recent_prices.get(asset_b, [])
        
        if len(prices_a) < 10 or len(prices_b) < 10:
            return (False, 0.0)
        
        # Compute returns
        returns_a = np.diff([p for _, p in prices_a])
        returns_b = np.diff([p for _, p in prices_b])
        
        min_len = min(len(returns_a), len(returns_b))
        if min_len < 5:
            return (False, 0.0)
        
        correlation = np.corrcoef(returns_a[-min_len:], returns_b[-min_len:])[0, 1]
        
        if np.isnan(correlation):
            return (False, 0.0)
        
        should_exit = abs(correlation) >= self.EXIT_THRESHOLD
        
        return (should_exit, float(correlation))


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_correlation_monitor: Optional[IntradayCorrelationMonitor] = None
_fast_checker: Optional[FastCorrelationChecker] = None


def get_correlation_monitor() -> IntradayCorrelationMonitor:
    """Get singleton correlation monitor."""
    global _correlation_monitor
    if _correlation_monitor is None:
        _correlation_monitor = IntradayCorrelationMonitor()
    return _correlation_monitor


def get_fast_correlation_checker() -> FastCorrelationChecker:
    """Get singleton fast correlation checker."""
    global _fast_checker
    if _fast_checker is None:
        _fast_checker = FastCorrelationChecker()
    return _fast_checker

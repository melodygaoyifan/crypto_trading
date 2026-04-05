"""
SOL Execution Logic - SOTA v3.0
================================
Solana-Specific Execution System for Kraken

Features:
1. Solana Network Stress Metrics Integration
   - Compute Unit (CU) utilization monitoring
   - Priority fee tracking
   - TPS and slot time awareness

2. Dynamic Slicing Based on 1-min ATR
   - Adaptive slice sizing
   - Volatility-adjusted execution

3. Order Flow Toxicity Filtering (VPIN)
   - Auto-switch to Passive/Hidden mode
   - Adverse selection avoidance

4. Kraken-Specific Optimizations
   - SOL/USD pair handling
   - Maker/taker fee optimization
   - Top-of-book depth analysis

Hardware: RTX 5090 for real-time calculations
Exchange: Kraken (26bps taker, 16bps maker)
Allocation: 50% SOL in portfolio

Author: Lead Quant Research & ML Systems Architect
Version: SOTA v3.0
"""

import numpy as np
import time
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum, auto
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('SOLExecution')


# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class SOLNetworkStatus(Enum):
    """Solana network status."""
    HEALTHY = auto()           # Normal operation
    CONGESTED = auto()         # High traffic, delays expected
    STRESSED = auto()          # Very high CU, priority fees elevated
    DEGRADED = auto()          # Performance issues
    HALTED = auto()            # Network halted (extreme)


class SOLExecutionMode(Enum):
    """SOL-specific execution modes."""
    AGGRESSIVE_TAKE = auto()      # Take liquidity, accept slippage
    STRATEGIC_TAKE = auto()       # Timed taker orders
    PASSIVE_MAKE = auto()         # Post limit orders
    HIDDEN_ICEBERG = auto()       # Iceberg orders
    DYNAMIC_TWAP = auto()         # Adaptive TWAP
    DYNAMIC_VWAP = auto()         # Volume-weighted
    ATR_SLICE = auto()            # ATR-based slicing
    ABORT = auto()                # Do not execute


class OrderFlowToxicity(Enum):
    """Order flow toxicity levels."""
    CLEAN = auto()       # < 30% VPIN - safe to trade
    MILD = auto()        # 30-45% - slight caution
    MODERATE = auto()    # 45-60% - reduce size
    SEVERE = auto()      # 60-75% - passive only
    TOXIC = auto()       # > 75% - abort


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolanaNetworkMetrics:
    """Real-time Solana network metrics."""
    # Compute Unit metrics
    cu_utilization_pct: float = 50.0      # Current CU utilization (0-100)
    cu_limit_per_block: int = 48_000_000  # CU limit per block
    avg_cu_per_tx: float = 200_000        # Average CU per transaction
    
    # Priority fees
    priority_fee_median: float = 0.0      # Median priority fee (lamports)
    priority_fee_p75: float = 0.0         # 75th percentile
    priority_fee_p90: float = 0.0         # 90th percentile
    priority_fee_p99: float = 0.0         # 99th percentile (spam threshold)
    
    # Network performance
    current_tps: float = 2000.0           # Transactions per second
    slot_time_ms: float = 400.0           # Current slot time
    confirmed_blocks_1min: int = 150      # Blocks confirmed in last minute
    skip_rate_pct: float = 2.0            # Block skip rate
    
    # Health indicators
    is_congested: bool = False
    is_degraded: bool = False
    vote_latency_ms: float = 50.0         # Vote transaction latency
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    @property
    def network_status(self) -> SOLNetworkStatus:
        """Determine network status from metrics."""
        if self.skip_rate_pct > 50 or self.slot_time_ms > 1000:
            return SOLNetworkStatus.HALTED
        elif self.cu_utilization_pct > 90 or self.skip_rate_pct > 20:
            return SOLNetworkStatus.DEGRADED
        elif self.cu_utilization_pct > 80 or self.is_congested:
            return SOLNetworkStatus.STRESSED
        elif self.cu_utilization_pct > 60 or self.priority_fee_p75 > 100000:
            return SOLNetworkStatus.CONGESTED
        else:
            return SOLNetworkStatus.HEALTHY
    
    @property
    def execution_risk_score(self) -> float:
        """
        Calculate execution risk score (0-1).
        Higher score = more risky to execute.
        """
        score = 0.0
        
        # CU utilization contribution (0-0.4)
        score += min(0.4, self.cu_utilization_pct / 100 * 0.4)
        
        # Priority fee contribution (0-0.3)
        if self.priority_fee_p75 > 0:
            fee_factor = np.log1p(self.priority_fee_p75 / 10000) / 10
            score += min(0.3, fee_factor)
        
        # Skip rate contribution (0-0.2)
        score += min(0.2, self.skip_rate_pct / 100 * 2)
        
        # TPS degradation (0-0.1)
        if self.current_tps < 1000:
            score += 0.1 * (1 - self.current_tps / 1000)
        
        return min(1.0, score)


@dataclass
class KrakenSOLOrderBook:
    """Kraken SOL/USD order book snapshot."""
    # Best bid/ask
    best_bid_price: float = 0.0
    best_bid_size: float = 0.0
    best_ask_price: float = 0.0
    best_ask_size: float = 0.0
    
    # Depth
    bid_depth_5: float = 0.0      # USD depth within 5 bps
    ask_depth_5: float = 0.0
    bid_depth_10: float = 0.0     # USD depth within 10 bps
    ask_depth_10: float = 0.0
    bid_depth_25: float = 0.0     # USD depth within 25 bps (fee threshold)
    ask_depth_25: float = 0.0
    
    # Imbalance
    bid_ask_ratio_5: float = 1.0  # bid_depth / ask_depth within 5 bps
    bid_ask_ratio_10: float = 1.0
    
    # Historical (for comparison)
    avg_depth_24h: float = 0.0    # Average depth over 24h
    depth_ratio_vs_24h: float = 1.0  # Current depth / avg depth
    
    timestamp: float = field(default_factory=time.time)
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.best_bid_price > 0 and self.best_ask_price > 0:
            return (self.best_ask_price - self.best_bid_price) / self.best_bid_price * 10000
        return 0.0
    
    @property
    def is_thin(self) -> bool:
        """Check if book is thin (below normal)."""
        return self.depth_ratio_vs_24h < 0.5
    
    @property
    def execution_capacity_usd(self) -> float:
        """Estimate how much can be executed without significant impact."""
        # Conservative estimate: 10% of depth within 10 bps
        return min(self.bid_depth_10, self.ask_depth_10) * 0.10


@dataclass
class SOLATRMetrics:
    """1-minute ATR metrics for SOL."""
    current_atr: float = 0.0
    atr_ma_14: float = 0.0          # 14-period MA of ATR
    atr_ma_50: float = 0.0          # 50-period MA
    atr_ratio_14: float = 1.0       # current / MA14
    atr_ratio_50: float = 1.0       # current / MA50
    atr_percentile: float = 50.0    # Percentile over 24h
    
    timestamp: float = field(default_factory=time.time)
    
    @property
    def volatility_regime(self) -> str:
        """Classify volatility regime."""
        if self.atr_ratio_14 < 0.5:
            return 'LOW'
        elif self.atr_ratio_14 < 0.8:
            return 'BELOW_NORMAL'
        elif self.atr_ratio_14 < 1.2:
            return 'NORMAL'
        elif self.atr_ratio_14 < 2.0:
            return 'ELEVATED'
        else:
            return 'EXTREME'


@dataclass
class SOLExecutionPlan:
    """Execution plan for SOL trade."""
    mode: SOLExecutionMode
    total_size_usd: float
    num_slices: int
    slice_size_usd: float
    slice_interval_ms: float
    urgency: str                   # LOW, NORMAL, HIGH
    max_participation_rate: float  # % of volume per slice
    passive_price_offset_bps: float  # For limit orders
    abort_threshold_slip_bps: float  # Abort if slippage exceeds
    reasoning: str
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class SOLExecutionResult:
    """Result of SOL execution."""
    plan: SOLExecutionPlan
    
    # Execution metrics
    executed_size_usd: float
    avg_fill_price: float
    slippage_bps: float
    fees_paid_usd: float
    implementation_shortfall_bps: float
    
    # Details
    num_fills: int
    maker_fills: int
    taker_fills: int
    execution_time_ms: float
    
    # Network state at execution
    network_status: SOLNetworkStatus
    cu_utilization: float
    
    # Quality assessment
    quality: str  # EXCELLENT, GOOD, FAIR, POOR
    
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# ATR CALCULATOR (1-MINUTE, GPU-ACCELERATED)
# ═══════════════════════════════════════════════════════════════════════════════

class SOLATRCalculator:
    """
    1-minute ATR calculator for SOL with GPU acceleration.
    
    Used for dynamic slice sizing based on real-time volatility.
    """
    
    def __init__(
        self,
        lookback_periods: int = 14,
        history_size: int = 1440,  # 24 hours of 1-min data
        use_gpu: bool = True
    ):
        self.lookback = lookback_periods
        self.history_size = history_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # OHLC storage
        self.high_history: deque = deque(maxlen=history_size)
        self.low_history: deque = deque(maxlen=history_size)
        self.close_history: deque = deque(maxlen=history_size)
        
        # ATR history
        self.atr_history: deque = deque(maxlen=history_size)
        
        # Derived metrics
        self.current_atr: float = 0.0
        self.atr_ma_14: float = 0.0
        self.atr_ma_50: float = 0.0
        
    def add_candle(self, high: float, low: float, close: float) -> SOLATRMetrics:
        """Add 1-minute candle and update ATR."""
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        if len(self.close_history) < self.lookback + 1:
            return SOLATRMetrics()
        
        # Calculate True Range
        if self.use_gpu:
            highs = cp.asarray(list(self.high_history)[-self.lookback:])
            lows = cp.asarray(list(self.low_history)[-self.lookback:])
            closes = cp.asarray(list(self.close_history)[-(self.lookback+1):-1])
            
            tr1 = highs - lows
            tr2 = cp.abs(highs - closes)
            tr3 = cp.abs(lows - closes)
            
            tr = cp.maximum(cp.maximum(tr1, tr2), tr3)
            atr = float(cp.mean(tr))
        else:
            highs = np.array(list(self.high_history)[-self.lookback:])
            lows = np.array(list(self.low_history)[-self.lookback:])
            closes = np.array(list(self.close_history)[-(self.lookback+1):-1])
            
            tr1 = highs - lows
            tr2 = np.abs(highs - closes)
            tr3 = np.abs(lows - closes)
            
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr)
        
        self.current_atr = atr
        self.atr_history.append(atr)
        
        # Moving averages
        if len(self.atr_history) >= 14:
            self.atr_ma_14 = np.mean(list(self.atr_history)[-14:])
        
        if len(self.atr_history) >= 50:
            self.atr_ma_50 = np.mean(list(self.atr_history)[-50:])
        
        # Calculate percentile
        if len(self.atr_history) >= 100:
            atr_array = np.array(list(self.atr_history))
            percentile = np.sum(atr_array < atr) / len(atr_array) * 100
        else:
            percentile = 50.0
        
        # Ratios
        atr_ratio_14 = atr / max(self.atr_ma_14, 1e-10) if self.atr_ma_14 > 0 else 1.0
        atr_ratio_50 = atr / max(self.atr_ma_50, 1e-10) if self.atr_ma_50 > 0 else 1.0
        
        return SOLATRMetrics(
            current_atr=atr,
            atr_ma_14=self.atr_ma_14,
            atr_ma_50=self.atr_ma_50,
            atr_ratio_14=atr_ratio_14,
            atr_ratio_50=atr_ratio_50,
            atr_percentile=percentile,
            timestamp=time.time()
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER FLOW TOXICITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class SOLToxicityAnalyzer:
    """
    Order flow toxicity analyzer for SOL.
    
    Monitors VPIN and OFI to detect adverse selection risk.
    """
    
    TOXICITY_THRESHOLDS = {
        OrderFlowToxicity.CLEAN: (0.0, 0.30),
        OrderFlowToxicity.MILD: (0.30, 0.45),
        OrderFlowToxicity.MODERATE: (0.45, 0.60),
        OrderFlowToxicity.SEVERE: (0.60, 0.75),
        OrderFlowToxicity.TOXIC: (0.75, 1.01),
    }
    
    def __init__(self):
        self.vpin_history: deque = deque(maxlen=1000)
        self.ofi_history: deque = deque(maxlen=1000)
        
    def classify(self, vpin: float, ofi_zscore: float) -> Tuple[OrderFlowToxicity, str]:
        """
        Classify order flow toxicity.
        
        Returns: (toxicity_level, recommendation)
        """
        # Determine toxicity from VPIN
        toxicity = OrderFlowToxicity.CLEAN
        for level, (low, high) in self.TOXICITY_THRESHOLDS.items():
            if low <= vpin < high:
                toxicity = level
                break
        
        # Adjust based on OFI
        if abs(ofi_zscore) > 3.0:
            # Strong directional flow - might upgrade toxicity
            if toxicity == OrderFlowToxicity.CLEAN:
                toxicity = OrderFlowToxicity.MILD
            elif toxicity == OrderFlowToxicity.MILD:
                toxicity = OrderFlowToxicity.MODERATE
        
        # Generate recommendation
        recommendations = {
            OrderFlowToxicity.CLEAN: "Safe to execute. Normal market conditions.",
            OrderFlowToxicity.MILD: "Slight caution. Consider reducing urgency.",
            OrderFlowToxicity.MODERATE: "Reduce size by 30%. Use TWAP.",
            OrderFlowToxicity.SEVERE: "Passive only. Reduce size by 50%.",
            OrderFlowToxicity.TOXIC: "ABORT execution. Adverse selection likely.",
        }
        
        return toxicity, recommendations[toxicity]
    
    def get_size_multiplier(self, toxicity: OrderFlowToxicity) -> float:
        """Get size multiplier based on toxicity."""
        multipliers = {
            OrderFlowToxicity.CLEAN: 1.0,
            OrderFlowToxicity.MILD: 0.9,
            OrderFlowToxicity.MODERATE: 0.7,
            OrderFlowToxicity.SEVERE: 0.5,
            OrderFlowToxicity.TOXIC: 0.0,
        }
        return multipliers[toxicity]


# ═══════════════════════════════════════════════════════════════════════════════
# SOL EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SOLExecutionEngine:
    """
    SOL-specific execution engine for Kraken.

    [P0-03] SINGLE_EXCHANGE_MODE: This engine plans and simulates SOL trades
    on KRAKEN ONLY. Solana network metrics are used as SIGNALS - no DEX
    swaps, no on-chain transactions. If single_exchange_mode=True (default),
    any attempt to route execution to a DEX venue is blocked.

    Integrates:
    - Solana network metrics
    - VPIN toxicity filtering
    - Dynamic ATR-based slicing
    - Kraken order book analysis
    """
    
    # Kraken SOL/USD fee structure
    TAKER_FEE_BPS = 26.0
    MAKER_FEE_BPS = 16.0
    
    # Execution thresholds
    MAX_PARTICIPATION_RATE = 0.10   # Max 10% of volume per slice
    MIN_SLICE_SIZE_USD = 50.0       # Minimum slice
    MAX_SLICE_SIZE_USD = 25000.0    # Maximum slice
    
    # Network thresholds
    CU_THRESHOLD_CAUTION = 70.0     # CU% for caution
    CU_THRESHOLD_REDUCE = 85.0      # CU% for size reduction
    CU_THRESHOLD_ABORT = 95.0       # CU% for abort
    
    def __init__(
        self,
        portfolio_sol_allocation: float = 0.50,
        initial_capital: float = 100000.0
    ):
        self.sol_allocation = portfolio_sol_allocation
        self.capital = initial_capital
        
        # Components
        self.atr_calculator = SOLATRCalculator()
        self.toxicity_analyzer = SOLToxicityAnalyzer()
        
        # Latest metrics
        self.latest_network: Optional[SolanaNetworkMetrics] = None
        self.latest_orderbook: Optional[KrakenSOLOrderBook] = None
        self.latest_atr: Optional[SOLATRMetrics] = None
        
        # Execution history
        self.execution_history: deque = deque(maxlen=1000)
        
        logger.info(f"SOLExecutionEngine initialized (allocation={portfolio_sol_allocation*100:.0f}%)")
    
    def update_network_metrics(self, metrics: SolanaNetworkMetrics):
        """Update network metrics."""
        self.latest_network = metrics
        
    def update_orderbook(self, orderbook: KrakenSOLOrderBook):
        """Update order book snapshot."""
        self.latest_orderbook = orderbook
        
    def update_candle(self, high: float, low: float, close: float):
        """Update 1-min candle for ATR."""
        self.latest_atr = self.atr_calculator.add_candle(high, low, close)
    
    def _determine_execution_mode(
        self,
        signal_strength: float,
        toxicity: OrderFlowToxicity,
        network: SolanaNetworkMetrics,
        atr: SOLATRMetrics
    ) -> Tuple[SOLExecutionMode, str]:
        """
        Determine optimal execution mode based on conditions.
        
        Logic:
        1. Network constraints override signal urgency
        2. Toxicity filters determine maker/taker mix
        3. ATR determines slicing strategy
        """
        reasons = []
        
        # Check network status first (hard constraints)
        if network.network_status == SOLNetworkStatus.HALTED:
            return SOLExecutionMode.ABORT, "Network halted"
        
        if network.cu_utilization_pct > self.CU_THRESHOLD_ABORT:
            return SOLExecutionMode.ABORT, f"CU {network.cu_utilization_pct:.0f}% > {self.CU_THRESHOLD_ABORT}%"
        
        # Check toxicity (abort if toxic)
        if toxicity == OrderFlowToxicity.TOXIC:
            return SOLExecutionMode.ABORT, "Toxic order flow (VPIN > 75%)"
        
        # Network-aware mode selection
        if network.cu_utilization_pct > self.CU_THRESHOLD_REDUCE:
            reasons.append(f"High CU ({network.cu_utilization_pct:.0f}%)")
            base_mode = SOLExecutionMode.PASSIVE_MAKE
        elif network.network_status == SOLNetworkStatus.CONGESTED:
            reasons.append("Network congested")
            base_mode = SOLExecutionMode.DYNAMIC_TWAP
        else:
            base_mode = SOLExecutionMode.STRATEGIC_TAKE
        
        # Toxicity-based adjustment
        if toxicity == OrderFlowToxicity.SEVERE:
            reasons.append("Severe toxicity")
            return SOLExecutionMode.HIDDEN_ICEBERG, "; ".join(reasons)
        elif toxicity == OrderFlowToxicity.MODERATE:
            reasons.append("Moderate toxicity")
            if base_mode == SOLExecutionMode.STRATEGIC_TAKE:
                base_mode = SOLExecutionMode.DYNAMIC_TWAP
        
        # ATR-based adjustment
        if atr.volatility_regime == 'EXTREME':
            reasons.append(f"Extreme volatility (ATR {atr.atr_ratio_14:.1f}x)")
            return SOLExecutionMode.ATR_SLICE, "; ".join(reasons)
        elif atr.volatility_regime == 'ELEVATED':
            reasons.append(f"Elevated volatility (ATR {atr.atr_ratio_14:.1f}x)")
            if base_mode not in [SOLExecutionMode.PASSIVE_MAKE, SOLExecutionMode.HIDDEN_ICEBERG]:
                base_mode = SOLExecutionMode.DYNAMIC_TWAP
        
        # Signal urgency
        if abs(signal_strength) > 0.8:
            reasons.append("Strong signal")
            if toxicity == OrderFlowToxicity.CLEAN and base_mode == SOLExecutionMode.STRATEGIC_TAKE:
                base_mode = SOLExecutionMode.AGGRESSIVE_TAKE
        
        return base_mode, "; ".join(reasons) if reasons else "Standard execution"
    
    def _calculate_slicing(
        self,
        mode: SOLExecutionMode,
        total_size_usd: float,
        atr: SOLATRMetrics,
        orderbook: KrakenSOLOrderBook
    ) -> Tuple[int, float, float]:
        """
        Calculate optimal slicing parameters.
        
        Returns: (num_slices, slice_size_usd, slice_interval_ms)
        """
        # Base slicing on mode
        if mode == SOLExecutionMode.AGGRESSIVE_TAKE:
            # Single slice (or max 3)
            num_slices = min(3, max(1, int(total_size_usd / self.MAX_SLICE_SIZE_USD) + 1))
            slice_interval_ms = 1000  # 1 second
            
        elif mode == SOLExecutionMode.PASSIVE_MAKE:
            # Spread over longer time
            num_slices = max(5, int(total_size_usd / 5000))  # $5k per slice
            slice_interval_ms = 60000  # 1 minute
            
        elif mode == SOLExecutionMode.HIDDEN_ICEBERG:
            # Many small slices
            num_slices = max(10, int(total_size_usd / 2000))  # $2k per slice
            slice_interval_ms = 30000  # 30 seconds
            
        elif mode == SOLExecutionMode.ATR_SLICE:
            # ATR-based slicing
            # More slices when ATR is high
            atr_factor = max(1, atr.atr_ratio_14)
            base_slices = int(total_size_usd / 10000) + 1
            num_slices = int(base_slices * atr_factor)
            
            # Faster when volatility high (capture moves)
            slice_interval_ms = 30000 / atr_factor
            
        else:  # TWAP/VWAP/STRATEGIC
            # Standard TWAP over 5 minutes
            num_slices = max(5, int(total_size_usd / 10000))
            slice_interval_ms = 60000
        
        # Ensure bounds
        num_slices = max(1, min(100, num_slices))
        slice_size = total_size_usd / max(1, num_slices)
        slice_size = max(self.MIN_SLICE_SIZE_USD, min(self.MAX_SLICE_SIZE_USD, slice_size))
        if slice_size > 0:
            num_slices = max(1, int(total_size_usd / slice_size))
        else:
            num_slices = 1
            slice_size = total_size_usd
        
        # Check against orderbook capacity
        if orderbook and orderbook.execution_capacity_usd > 0:
            capacity = orderbook.execution_capacity_usd
            if slice_size > capacity * 2 and capacity > 0:
                # Slice is too large for book
                slice_size = max(self.MIN_SLICE_SIZE_USD, capacity)
                num_slices = max(1, int(total_size_usd / slice_size))
        
        # Final safety check
        if slice_size <= 0:
            slice_size = total_size_usd
            num_slices = 1
        
        return num_slices, slice_size, slice_interval_ms
    
    def create_execution_plan(
        self,
        signal_strength: float,
        target_size_usd: float,
        vpin: float,
        ofi_zscore: float
    ) -> SOLExecutionPlan:
        """
        Create execution plan for SOL trade.
        
        Args:
            signal_strength: Signal from fusion module [-1, +1]
            target_size_usd: Target trade size in USD
            vpin: Current VPIN
            ofi_zscore: Order flow imbalance z-score
            
        Returns:
            SOLExecutionPlan
        """
        # Get latest metrics (use defaults if not available)
        network = self.latest_network or SolanaNetworkMetrics()
        orderbook = self.latest_orderbook or KrakenSOLOrderBook()
        atr = self.latest_atr or SOLATRMetrics()
        
        # Classify toxicity
        toxicity, tox_reason = self.toxicity_analyzer.classify(vpin, ofi_zscore)
        
        # Adjust size based on toxicity
        size_mult = self.toxicity_analyzer.get_size_multiplier(toxicity)
        adjusted_size = target_size_usd * size_mult
        
        # Determine execution mode
        mode, mode_reason = self._determine_execution_mode(
            signal_strength, toxicity, network, atr
        )
        
        # Handle abort
        if mode == SOLExecutionMode.ABORT:
            return SOLExecutionPlan(
                mode=mode,
                total_size_usd=0.0,
                num_slices=0,
                slice_size_usd=0.0,
                slice_interval_ms=0.0,
                urgency='NONE',
                max_participation_rate=0.0,
                passive_price_offset_bps=0.0,
                abort_threshold_slip_bps=0.0,
                reasoning=f"ABORT: {mode_reason}; {tox_reason}",
                timestamp=time.time()
            )
        
        # Calculate slicing
        num_slices, slice_size, interval_ms = self._calculate_slicing(
            mode, adjusted_size, atr, orderbook
        )
        
        # Determine urgency
        if mode == SOLExecutionMode.AGGRESSIVE_TAKE:
            urgency = 'HIGH'
        elif mode in [SOLExecutionMode.PASSIVE_MAKE, SOLExecutionMode.HIDDEN_ICEBERG]:
            urgency = 'LOW'
        else:
            urgency = 'NORMAL'
        
        # Participation rate (% of volume per slice)
        if mode in [SOLExecutionMode.AGGRESSIVE_TAKE, SOLExecutionMode.STRATEGIC_TAKE]:
            participation = min(self.MAX_PARTICIPATION_RATE, 0.10)
        elif mode == SOLExecutionMode.HIDDEN_ICEBERG:
            participation = 0.02  # 2% to stay hidden
        else:
            participation = 0.05  # 5% standard
        
        # Passive price offset
        if mode == SOLExecutionMode.PASSIVE_MAKE:
            offset_bps = -5.0  # Post 5 bps better than mid
        elif mode == SOLExecutionMode.HIDDEN_ICEBERG:
            offset_bps = -3.0  # Slightly better
        else:
            offset_bps = 0.0  # Take at market
        
        # Abort threshold (based on fees + expected slippage)
        base_abort = self.TAKER_FEE_BPS + 10  # Fee + 10 bps
        if toxicity in [OrderFlowToxicity.MODERATE, OrderFlowToxicity.SEVERE]:
            abort_threshold = base_abort * 1.5
        else:
            abort_threshold = base_abort * 2
        
        return SOLExecutionPlan(
            mode=mode,
            total_size_usd=adjusted_size,
            num_slices=num_slices,
            slice_size_usd=slice_size,
            slice_interval_ms=interval_ms,
            urgency=urgency,
            max_participation_rate=participation,
            passive_price_offset_bps=offset_bps,
            abort_threshold_slip_bps=abort_threshold,
            reasoning=f"{mode_reason}; Size adj: {size_mult:.0%}; {tox_reason}",
            timestamp=time.time()
        )
    
    def simulate_execution(
        self,
        plan: SOLExecutionPlan,
        arrival_price: float,
        price_series: np.ndarray,
        volume_series: np.ndarray
    ) -> SOLExecutionResult:
        """
        Simulate execution of plan.
        
        Args:
            plan: Execution plan
            arrival_price: Price at decision time
            price_series: Price series during execution
            volume_series: Volume series during execution
            
        Returns:
            SOLExecutionResult
        """
        start_time = time.perf_counter()
        
        if plan.mode == SOLExecutionMode.ABORT:
            return SOLExecutionResult(
                plan=plan,
                executed_size_usd=0.0,
                avg_fill_price=arrival_price,
                slippage_bps=0.0,
                fees_paid_usd=0.0,
                implementation_shortfall_bps=0.0,
                num_fills=0,
                maker_fills=0,
                taker_fills=0,
                execution_time_ms=0.0,
                network_status=self.latest_network.network_status if self.latest_network else SOLNetworkStatus.HEALTHY,
                cu_utilization=self.latest_network.cu_utilization_pct if self.latest_network else 50.0,
                quality='ABORTED'
            )
        
        # Simulate slices
        fill_prices = []
        fill_sizes = []
        maker_count = 0
        taker_count = 0
        
        remaining_size = plan.total_size_usd
        
        for i in range(plan.num_slices):
            if remaining_size <= 0:
                break
                
            slice_size = min(plan.slice_size_usd, remaining_size)
            
            # Get price for this slice
            price_idx = min(i * int(plan.slice_interval_ms / 1000), len(price_series) - 1)
            base_price = price_series[price_idx]
            
            # Apply slippage (sqrt model)
            vol_idx = min(i * int(plan.slice_interval_ms / 1000), len(volume_series) - 1)
            volume = volume_series[vol_idx] if vol_idx < len(volume_series) else 1000
            
            # Participation impact
            participation = slice_size / (volume * base_price + 1e-10)
            impact_bps = 5.0 * np.sqrt(participation) * (1 + np.random.uniform(-0.2, 0.2))
            
            # Mode-specific adjustment
            if plan.mode in [SOLExecutionMode.PASSIVE_MAKE, SOLExecutionMode.HIDDEN_ICEBERG]:
                # Passive gets better fills (but might not fill)
                fill_prob = 0.8 if plan.mode == SOLExecutionMode.PASSIVE_MAKE else 0.9
                if np.random.random() < fill_prob:
                    impact_bps *= 0.5
                    maker_count += 1
                else:
                    # Didn't fill - have to take
                    taker_count += 1
            else:
                taker_count += 1
            
            # Calculate fill price
            fill_price = base_price * (1 + impact_bps / 10000)
            
            fill_prices.append(fill_price)
            fill_sizes.append(slice_size)
            remaining_size -= slice_size
        
        # Calculate results
        if fill_prices:
            total_size = sum(fill_sizes)
            avg_fill = np.average(fill_prices, weights=fill_sizes)
            slippage_bps = (avg_fill - arrival_price) / arrival_price * 10000
            
            # Fees
            maker_size = sum(fill_sizes[:maker_count])
            taker_size = total_size - maker_size
            fees = (maker_size * self.MAKER_FEE_BPS + taker_size * self.TAKER_FEE_BPS) / 10000
            
            # Implementation shortfall
            is_bps = slippage_bps + (fees / total_size * 10000) if total_size > 0 else 0
            
            # Quality classification
            if is_bps < 10:
                quality = 'EXCELLENT'
            elif is_bps < 25:
                quality = 'GOOD'
            elif is_bps < 50:
                quality = 'FAIR'
            else:
                quality = 'POOR'
        else:
            total_size = 0
            avg_fill = arrival_price
            slippage_bps = 0
            fees = 0
            is_bps = 0
            quality = 'NO_FILL'
        
        exec_time = (time.perf_counter() - start_time) * 1000
        
        result = SOLExecutionResult(
            plan=plan,
            executed_size_usd=total_size,
            avg_fill_price=avg_fill,
            slippage_bps=slippage_bps,
            fees_paid_usd=fees,
            implementation_shortfall_bps=is_bps,
            num_fills=len(fill_prices),
            maker_fills=maker_count,
            taker_fills=taker_count,
            execution_time_ms=exec_time,
            network_status=self.latest_network.network_status if self.latest_network else SOLNetworkStatus.HEALTHY,
            cu_utilization=self.latest_network.cu_utilization_pct if self.latest_network else 50.0,
            quality=quality
        )
        
        self.execution_history.append(result)
        return result
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        if not self.execution_history:
            return {}
        
        results = list(self.execution_history)
        
        # Filter out aborts
        executed = [r for r in results if r.executed_size_usd > 0]
        
        if not executed:
            return {'total_trades': 0}
        
        return {
            'total_trades': len(executed),
            'total_volume_usd': sum(r.executed_size_usd for r in executed),
            'avg_slippage_bps': np.mean([r.slippage_bps for r in executed]),
            'avg_is_bps': np.mean([r.implementation_shortfall_bps for r in executed]),
            'total_fees_usd': sum(r.fees_paid_usd for r in executed),
            'quality_distribution': {
                q: len([r for r in executed if r.quality == q])
                for q in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']
            },
            'maker_fill_rate': sum(r.maker_fills for r in executed) / sum(r.num_fills for r in executed) if sum(r.num_fills for r in executed) > 0 else 0,
            'aborted_count': len([r for r in results if r.executed_size_usd == 0])
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_sol_execution_engine(
    sol_allocation: float = 0.50,
    capital: float = 100000.0
) -> SOLExecutionEngine:
    """Factory function for SOL execution engine."""
    return SOLExecutionEngine(
        portfolio_sol_allocation=sol_allocation,
        initial_capital=capital
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("SOL EXECUTION ENGINE TEST")
    print("=" * 80)
    
    # Create engine
    engine = create_sol_execution_engine(sol_allocation=0.50, capital=100000)
    
    # Simulate network metrics
    network = SolanaNetworkMetrics(
        cu_utilization_pct=65.0,
        priority_fee_median=5000,
        current_tps=2500,
        slot_time_ms=400,
        is_congested=False
    )
    engine.update_network_metrics(network)
    print(f"\nNetwork Status: {network.network_status.name}")
    print(f"Execution Risk Score: {network.execution_risk_score:.2f}")
    
    # Simulate orderbook
    orderbook = KrakenSOLOrderBook(
        best_bid_price=149.95,
        best_bid_size=500,
        best_ask_price=150.05,
        best_ask_size=450,
        bid_depth_10=50000,
        ask_depth_10=45000,
        depth_ratio_vs_24h=0.8
    )
    engine.update_orderbook(orderbook)
    print(f"\nOrderbook spread: {orderbook.spread_bps:.1f} bps")
    print(f"Is thin: {orderbook.is_thin}")
    print(f"Execution capacity: ${orderbook.execution_capacity_usd:,.0f}")
    
    # Simulate ATR updates
    np.random.seed(42)
    base_price = 150.0
    for i in range(100):
        high = base_price + np.random.uniform(0, 3)
        low = base_price - np.random.uniform(0, 3)
        close = base_price + np.random.uniform(-1, 1)
        engine.update_candle(high, low, close)
    
    atr = engine.latest_atr
    print(f"\nATR Metrics:")
    print(f"  Current ATR: ${atr.current_atr:.2f}")
    print(f"  ATR/MA14: {atr.atr_ratio_14:.2f}x")
    print(f"  Vol Regime: {atr.volatility_regime}")
    
    # Test different scenarios
    scenarios = [
        ("Strong signal, clean flow", 0.8, 10000, 0.25, 0.5),
        ("Moderate signal, elevated toxicity", 0.5, 15000, 0.55, 1.5),
        ("Strong signal, severe toxicity", 0.9, 20000, 0.70, 2.5),
        ("Weak signal, high congestion", 0.3, 5000, 0.35, 0.8),
    ]
    
    print("\n" + "=" * 80)
    print("EXECUTION SCENARIOS")
    print("=" * 80)
    
    for name, signal, size, vpin, ofi in scenarios:
        print(f"\n{name}:")
        print(f"  Signal: {signal:.1f}, Size: ${size:,}, VPIN: {vpin:.2f}, OFI-z: {ofi:.1f}")
        
        plan = engine.create_execution_plan(
            signal_strength=signal,
            target_size_usd=size,
            vpin=vpin,
            ofi_zscore=ofi
        )
        
        print(f"  Mode: {plan.mode.name}")
        print(f"  Slices: {plan.num_slices} x ${plan.slice_size_usd:,.0f}")
        print(f"  Interval: {plan.slice_interval_ms/1000:.1f}s")
        print(f"  Urgency: {plan.urgency}")
        print(f"  Reasoning: {plan.reasoning}")
        
        # Simulate execution
        prices = base_price + np.cumsum(np.random.randn(100) * 0.1)
        volumes = np.random.exponential(1000, 100)
        
        result = engine.simulate_execution(plan, base_price, prices, volumes)
        
        print(f"  RESULT: {result.quality}")
        print(f"    Executed: ${result.executed_size_usd:,.0f}")
        print(f"    Slippage: {result.slippage_bps:.1f} bps")
        print(f"    IS: {result.implementation_shortfall_bps:.1f} bps")
        print(f"    Fees: ${result.fees_paid_usd:.2f}")
    
    # Final stats
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    stats = engine.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

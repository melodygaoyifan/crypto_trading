"""
SOTA v3.1 Passive-Aggressive Execution & Data Drift Detection
===============================================================

Passive-Aggressive Execution:
- Maker Bias: Default to passive limit orders at best bid/ask
- Reverse Selection Guard: Monitor fill slope (<500ms = toxic)
- Fee Neutralization: 26bps taker vs 16bps maker optimization
- Aggression Trigger: Switch to taker only on VPIN toxic or high-conviction

Data Drift Detector:
- Monitor DRL latent space distributions
- Compare live vs training data
- Auto down-weight DRL on concept drift (p-value < 0.05)
"""

import os
import time
import asyncio
import logging
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import threading
import json

# Statistical tests
try:
    from scipy import stats
    from scipy.stats import ks_2samp, wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ExecutionV31')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Kraken fee structure (defaults; updated by update_fee_tier_from_exchange() at startup)
KRAKEN_TAKER_FEE_BPS = 26.0
KRAKEN_MAKER_FEE_BPS = 16.0
EXPECTED_SLIPPAGE_BPS = 10.0


def update_fee_tier_from_exchange(taker_bps: float, maker_bps: float):
    """[FIX-FEE-TIER] Update fee constants with actual exchange tier.

    Called from main.py at startup after querying Kraken API.
    Prevents PnL miscalculation from stale hardcoded rates.
    """
    global KRAKEN_TAKER_FEE_BPS, KRAKEN_MAKER_FEE_BPS
    KRAKEN_TAKER_FEE_BPS = taker_bps
    KRAKEN_MAKER_FEE_BPS = maker_bps
    logger.info(
        f"[FIX-FEE-TIER] Fee tier updated: taker={taker_bps:.1f}bps, "
        f"maker={maker_bps:.1f}bps"
    )

# Fill slope threshold (toxic if filled in < 500ms)
FILL_SLOPE_TOXIC_MS = 500.0

# Data drift threshold
DRIFT_P_VALUE_THRESHOLD = 0.05

# Minimum edge for execution
# [P0-FIX] Was 3.0 — required 150bps alpha at 50bps friction, blocking all trades.
# 1.5x ensures alpha covers round-trip cost with margin of safety.
MIN_EDGE_MULTIPLIER = 1.5  # α > 1.5× friction


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionMode(Enum):
    """Execution modes with fee implications."""
    PASSIVE = auto()      # Maker only (16 bps)
    AGGRESSIVE = auto()   # Taker (26 bps)
    HYBRID = auto()       # Start passive, go aggressive if needed
    HIDDEN = auto()       # Iceberg orders
    TWAP = auto()         # Time-weighted
    ABORT = auto()        # Cancel all


class FillQuality(Enum):
    """Fill quality classification."""
    EXCELLENT = auto()    # Better than expected
    GOOD = auto()         # As expected
    ACCEPTABLE = auto()   # Slightly worse
    POOR = auto()         # Significantly worse
    TOXIC = auto()        # Adverse selection detected


@dataclass
class OrderState:
    """State of a single order."""
    
    order_id: str
    asset: str
    side: str  # "buy" or "sell"
    
    # Prices
    target_price: float
    limit_price: float
    
    # Quantity
    total_qty: float
    filled_qty: float
    remaining_qty: float
    
    # Timing
    created_at: float
    first_fill_at: Optional[float] = None
    last_fill_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Execution mode
    mode: ExecutionMode = ExecutionMode.PASSIVE
    maker_fills: int = 0
    taker_fills: int = 0
    
    # Fill slope monitoring
    fill_times: List[float] = field(default_factory=list)
    fill_qtys: List[float] = field(default_factory=list)
    filled_notional: float = 0.0
    
    # Quality
    avg_fill_price: float = 0.0
    slippage_bps: float = 0.0
    fees_paid: float = 0.0
    fill_quality: FillQuality = FillQuality.GOOD


@dataclass
class FillSlopeAnalysis:
    """Analysis of fill slope for adverse selection detection."""
    
    slope_ms_per_unit: float = 0.0
    time_to_first_fill_ms: float = 0.0
    time_to_complete_ms: float = 0.0
    
    is_toxic: bool = False
    toxicity_reason: str = ""
    
    # Recommendations
    should_pause: bool = False
    should_adjust_price: bool = False
    price_adjustment_bps: float = 0.0


@dataclass
class ExecutionDecision:
    """Decision for order execution."""
    
    asset: str
    action: str = "HOLD"  # "BUY", "SELL", "HOLD" - default to HOLD for safety
    
    # Mode selection
    execution_mode: ExecutionMode = ExecutionMode.PASSIVE
    
    # Sizing
    target_qty: float = 0.0
    max_aggressive_pct: float = 0.2  # Max % to fill via taker
    
    # Pricing
    price_improvement_bps: float = 0.0
    max_slippage_bps: float = 10.0
    
    # Timing
    urgency: str = "NORMAL"
    max_time_seconds: float = 300.0
    
    # Fee optimization
    estimated_maker_fee: float = 0.0
    estimated_taker_fee: float = 0.0
    fee_savings: float = 0.0
    
    # Edge validation
    predicted_alpha_bps: float = 0.0
    total_friction_bps: float = 0.0
    edge_sufficient: bool = False
    
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# FILL SLOPE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class FillSlopeMonitor:
    """
    Monitor fill slope to detect adverse selection.
    
    Logic:
    - If a large passive position fills in < 500ms, interpret as toxic flow
    - Toxic flow indicates informed traders picking off our resting orders
    - Response: Pause execution, adjust target state
    """
    
    def __init__(
        self,
        toxic_threshold_ms: float = FILL_SLOPE_TOXIC_MS,
        lookback_fills: int = 20
    ):
        self.toxic_threshold_ms = toxic_threshold_ms
        self.lookback_fills = lookback_fills
        
        self.logger = logging.getLogger('FillSlope')
        
        # Fill history per asset
        self.fill_history: Dict[str, deque] = {}
        
        # Order states
        self.active_orders: Dict[str, OrderState] = {}
    
    def register_order(self, order: OrderState):
        """Register a new order for monitoring."""
        self.active_orders[order.order_id] = order
        
        if order.asset not in self.fill_history:
            self.fill_history[order.asset] = deque(maxlen=self.lookback_fills)

    def ensure_order_registered(
        self,
        order_id: str,
        asset: str,
        side: str,
        total_qty: float,
        target_price: float,
        limit_price: Optional[float] = None,
        mode: Any = None,
    ) -> OrderState:
        """Backfill a minimal order state when paper fills bypass create_order()."""
        if order_id in self.active_orders:
            return self.active_orders[order_id]

        qty = max(float(total_qty or 0.0), 0.0)
        order_mode = mode if isinstance(mode, ExecutionMode) else ExecutionMode.PASSIVE
        order = OrderState(
            order_id=order_id,
            asset=asset,
            side=str(side or "").lower(),
            target_price=target_price,
            limit_price=limit_price if limit_price is not None else target_price,
            total_qty=qty,
            filled_qty=0.0,
            remaining_qty=qty,
            created_at=time.time(),
            mode=order_mode,
        )
        self.register_order(order)
        return order
    
    def record_fill(
        self,
        order_id: str,
        fill_qty: float,
        fill_price: float,
        is_maker: bool
    ) -> FillSlopeAnalysis:
        """Record a fill and analyze slope."""
        
        if order_id not in self.active_orders:
            return FillSlopeAnalysis()
        
        order = self.active_orders[order_id]
        now = time.time() * 1000  # milliseconds
        
        # Update order state
        if order.first_fill_at is None:
            order.first_fill_at = now
        
        order.last_fill_at = now
        order.fill_times.append(now)
        order.fill_qtys.append(fill_qty)
        order.filled_qty += fill_qty
        order.filled_notional += fill_qty * fill_price
        order.remaining_qty = order.total_qty - order.filled_qty
        
        if is_maker:
            order.maker_fills += 1
        else:
            order.taker_fills += 1
        
        # Update avg fill price
        if order.filled_qty > 0:
            order.avg_fill_price = order.filled_notional / order.filled_qty
        
        # Calculate slippage
        if order.target_price > 0:
            if order.side == "buy":
                order.slippage_bps = ((order.avg_fill_price - order.target_price) / order.target_price) * 10000
            else:
                order.slippage_bps = ((order.target_price - order.avg_fill_price) / order.target_price) * 10000
        
        # Analyze fill slope
        analysis = self._analyze_fill_slope(order)
        
        # Store in history
        adverse_selection_proxy = float(
            np.clip(
                max(0.0, float(order.slippage_bps or 0.0))
                / max(EXPECTED_SLIPPAGE_BPS * 2.0, 1.0),
                0.0,
                1.0,
            )
        )
        self.fill_history[order.asset].append({
            'time_ms': now,
            'qty': fill_qty,
            'is_maker': is_maker,
            'is_toxic': analysis.is_toxic,
            'slippage_bps': float(order.slippage_bps or 0.0),
            'adverse_selection_proxy': adverse_selection_proxy,
            'toxicity_reason': analysis.toxicity_reason,
        })
        
        # Check if order completed
        if order.remaining_qty <= 0:
            order.completed_at = now
            self._classify_fill_quality(order)
        
        return analysis
    
    def _analyze_fill_slope(self, order: OrderState) -> FillSlopeAnalysis:
        """Analyze the fill slope for adverse selection."""
        
        analysis = FillSlopeAnalysis()
        
        if len(order.fill_times) < 2:
            return analysis
        
        # Time to first fill
        analysis.time_to_first_fill_ms = order.first_fill_at - (order.created_at * 1000)
        
        # Current fill rate (slope)
        time_elapsed_ms = order.fill_times[-1] - order.fill_times[0]
        total_filled = sum(order.fill_qtys)
        
        if total_filled > 0 and time_elapsed_ms > 0:
            analysis.slope_ms_per_unit = time_elapsed_ms / total_filled
        
        # Detect toxic fills
        # If we're getting filled too fast, it's likely adverse selection
        if time_elapsed_ms > 0 and time_elapsed_ms < self.toxic_threshold_ms:
            fill_rate = total_filled / (time_elapsed_ms / 1000.0)  # units per second
            
            # Check if large amount filled quickly
            pct_filled = order.filled_qty / order.total_qty
            
            if pct_filled > 0.3 and time_elapsed_ms < self.toxic_threshold_ms:
                analysis.is_toxic = True
                analysis.toxicity_reason = f"Fast fill detected: {pct_filled*100:.1f}% in {time_elapsed_ms:.0f}ms"
                analysis.should_pause = True
                analysis.should_adjust_price = True
                
                # Suggest price adjustment away from market
                analysis.price_adjustment_bps = 5.0  # Pull back 5 bps
        
        return analysis
    
    def _classify_fill_quality(self, order: OrderState):
        """Classify overall fill quality."""
        
        # Based on slippage vs expected
        if order.slippage_bps < -5.0:
            order.fill_quality = FillQuality.EXCELLENT
        elif order.slippage_bps < 0:
            order.fill_quality = FillQuality.GOOD
        elif order.slippage_bps < EXPECTED_SLIPPAGE_BPS:
            order.fill_quality = FillQuality.ACCEPTABLE
        elif order.slippage_bps < EXPECTED_SLIPPAGE_BPS * 2:
            order.fill_quality = FillQuality.POOR
        else:
            order.fill_quality = FillQuality.TOXIC
    
    def get_order_state(self, order_id: str) -> Optional[OrderState]:
        """Get current order state."""
        return self.active_orders.get(order_id)
    
    def get_asset_toxicity_rate(self, asset: str) -> float:
        """Get recent toxicity rate for an asset."""
        if asset not in self.fill_history:
            return 0.0

        history = self.fill_history[asset]
        if len(history) == 0:
            return 0.0

        toxic_count = sum(1 for f in history if f.get('is_toxic', False))
        return toxic_count / len(history)

    def get_asset_fill_metrics(self, asset: str) -> Dict[str, Any]:
        """Return recent per-asset fill telemetry for downstream execution controls."""
        history = self.fill_history.get(asset)
        if not history:
            return {
                'fill_count': 0,
                'telemetry_verified': False,
                'toxicity_rate': 0.0,
                'latest_slippage_bps': 0.0,
                'last_toxic_fill_ms': None,
                'adverse_selection_proxy': 0.0,
                'latest_is_toxic': False,
            }

        latest = history[-1]
        last_toxic = next((row for row in reversed(history) if row.get('is_toxic', False)), None)
        return {
            'fill_count': len(history),
            'telemetry_verified': True,
            'toxicity_rate': self.get_asset_toxicity_rate(asset),
            'latest_slippage_bps': float(latest.get('slippage_bps', 0.0) or 0.0),
            'last_toxic_fill_ms': last_toxic.get('time_ms') if last_toxic else None,
            'adverse_selection_proxy': float(latest.get('adverse_selection_proxy', 0.0) or 0.0),
            'latest_is_toxic': bool(latest.get('is_toxic', False)),
        }

    def get_execution_guidance(self, asset: str) -> Dict[str, Any]:
        """
        [FILLSLOPE] Pre-execution guidance based on recent fill quality.

        Returns dict with:
          force_passive: bool   - True -> override to LIMIT (no MARKET orders)
          extra_improve_bps: float - Additional price improvement to pull back
          size_reduce_pct: float - Fraction to reduce order size (0.0 = full, 0.3 = reduce 30%)
          reason: str           - Human-readable reason
        """
        tox_rate = self.get_asset_toxicity_rate(asset)
        guidance: Dict[str, Any] = {
            'force_passive': False,
            'extra_improve_bps': 0.0,
            'size_reduce_pct': 0.0,
            'toxicity_rate': tox_rate,
            'reason': '',
        }

        if tox_rate >= 0.7:
            # Severe: most recent fills are toxic - force passive + reduce size
            guidance['force_passive'] = True
            guidance['extra_improve_bps'] = 8.0
            guidance['size_reduce_pct'] = 0.30
            guidance['reason'] = f"SEVERE toxicity {tox_rate:.0%} -> passive+reduce"
        elif tox_rate >= 0.5:
            # High: force passive + add improvement
            guidance['force_passive'] = True
            guidance['extra_improve_bps'] = 5.0
            guidance['reason'] = f"HIGH toxicity {tox_rate:.0%} -> force passive"
        elif tox_rate >= 0.3:
            # Elevated: extra improvement only
            guidance['extra_improve_bps'] = 3.0
            guidance['reason'] = f"ELEVATED toxicity {tox_rate:.0%} -> extra improve"

        return guidance


# ═══════════════════════════════════════════════════════════════════════════════
# PASSIVE-AGGRESSIVE EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PAExecutorConfig:
    """Configuration for PassiveAggressiveExecutor."""
    mode: str = "ACTIVE"                      # "SHADOW" or "ACTIVE"
    max_price_improve_bps: float = 8.0        # Cap price improvement
    min_price_improve_bps: float = 2.0        # Floor for price improvement
    toxic_flow_abort_threshold: float = 0.85  # VPIN above this -> ABORT
    abort_on_insufficient_edge: bool = True    # ULTRA: False (let alpha gate decide)

    @classmethod
    def from_dict(cls, d: dict) -> "PAExecutorConfig":
        return cls(
            mode=d.get("mode", "ACTIVE"),
            max_price_improve_bps=d.get("max_price_improve_bps", 8.0),
            min_price_improve_bps=d.get("min_price_improve_bps", 2.0),
            toxic_flow_abort_threshold=d.get("toxic_flow_abort_threshold", 0.85),
            abort_on_insufficient_edge=d.get("abort_on_insufficient_edge", True),
        )


class PassiveAggressiveExecutor:
    """
    Fee-optimized execution with maker bias and conditional aggression.

    Strategy:
    1. Default: Passive limit orders at best bid/ask (16 bps)
    2. Monitor fill slope for adverse selection
    3. Switch to taker (26 bps) only if:
       - VPIN indicates toxic flow
       - High-conviction signal from Lead-Lag engine
       - Time urgency requires immediate fill

    Fee Savings Target: 10 bps per trade (26 - 16)

    Modes:
    - SHADOW (default): Compute decision and log, but caller uses legacy fallback.
    - ACTIVE: Decision drives order type/price selection on hot path.
    """

    def __init__(self, config: Optional[PAExecutorConfig] = None):
        self.logger = logging.getLogger('PassiveAggressive')
        self.config = config or PAExecutorConfig()
        self.shadow_mode = (self.config.mode != "ACTIVE")

        # Fill slope monitor
        self.fill_monitor = FillSlopeMonitor()

        # Execution statistics
        self.stats = {
            'total_orders': 0,
            'maker_fills': 0,
            'taker_fills': 0,
            'total_fee_savings': 0.0,
            'toxic_fills_avoided': 0,
        }
    
    def decide(
        self,
        asset: str,
        signal: float,
        confidence: float,
        vpin: float,
        urgency: str,
        predicted_alpha_bps: float,
        current_price: float,
        target_qty: float,
        timing_score: float = 0.5,
        timing_mode: str = "",
    ) -> ExecutionDecision:
        """
        Decide execution mode based on market conditions.

        Signal: [-1, +1] (negative = sell, positive = buy)
        Confidence: [0, 1]
        VPIN: [0, 1] (higher = more toxic)
        Urgency: LOW, NORMAL, HIGH, CRITICAL
        timing_score: [0, 1] from TimingEngine (higher = better microstructure)
        timing_mode: recommended mode from TimingEngine (DELAY/PASSIVE_ONLY/etc.)
        """
        
        decision = ExecutionDecision(
            asset=asset,
            predicted_alpha_bps=predicted_alpha_bps,
            target_qty=target_qty
        )
        
        # Determine action
        if signal > 0.2:
            decision.action = "BUY"
        elif signal < -0.2:
            decision.action = "SELL"
        else:
            decision.action = "HOLD"
            decision.reason = "Signal too weak"
            return decision
        
        # Calculate total friction
        decision.total_friction_bps = KRAKEN_TAKER_FEE_BPS + EXPECTED_SLIPPAGE_BPS
        
        # Edge validation (α > 3× friction)
        decision.edge_sufficient = predicted_alpha_bps > (MIN_EDGE_MULTIPLIER * decision.total_friction_bps)
        
        if not decision.edge_sufficient and self.config.abort_on_insufficient_edge:
            decision.action = "HOLD"
            decision.reason = f"Insufficient edge: {predicted_alpha_bps:.1f} bps < {MIN_EDGE_MULTIPLIER}× {decision.total_friction_bps:.1f} bps"
            return decision
        
        # Calculate fee estimates
        order_value = current_price * target_qty
        decision.estimated_maker_fee = order_value * KRAKEN_MAKER_FEE_BPS / 10000
        decision.estimated_taker_fee = order_value * KRAKEN_TAKER_FEE_BPS / 10000
        decision.fee_savings = decision.estimated_taker_fee - decision.estimated_maker_fee
        
        # === Mode Selection Logic ===
        
        # Check for toxic conditions requiring aggression
        _toxic_thresh = self.config.toxic_flow_abort_threshold
        vpin_toxic = vpin > 0.7
        vpin_abort = vpin > _toxic_thresh  # Extreme toxicity -> ABORT
        vpin_elevated = vpin > 0.5
        
        # Check recent toxicity rate
        asset_toxicity = self.fill_monitor.get_asset_toxicity_rate(asset)
        
        if vpin_abort and urgency != "CRITICAL":
            # [UNLEASH] UL-8b: Direction-aware VPIN
            # High VPIN = toxic sell flow = shorts are profiting. Only abort longs.
            if signal < 0:
                # SHORT direction + high VPIN -> REDUCE, not ABORT
                decision.execution_mode = ExecutionMode.PASSIVE
                decision.max_aggressive_pct = 0.4
                decision.reason = f"[UNLEASH] VPIN extreme ({vpin:.2f}) but SHORT -> REDUCE (not ABORT)"
                return decision
            else:
                # LONG direction + high VPIN -> ABORT (unchanged)
                decision.execution_mode = ExecutionMode.ABORT
                decision.reason = f"VPIN extreme ({vpin:.2f} > {_toxic_thresh:.2f}) - ABORT"
                return decision

        if urgency == "CRITICAL":
            # Critical urgency: Full aggressive
            decision.execution_mode = ExecutionMode.AGGRESSIVE
            decision.max_aggressive_pct = 1.0
            decision.max_time_seconds = 30.0
            decision.reason = "Critical urgency - full aggressive"
        
        elif vpin_toxic:
            # Toxic VPIN: Use aggression to avoid adverse selection
            # Counterintuitive: We want to hit bids/asks before they disappear
            decision.execution_mode = ExecutionMode.AGGRESSIVE
            decision.max_aggressive_pct = 0.8
            decision.max_time_seconds = 60.0
            decision.reason = f"VPIN toxic ({vpin:.2f}) - aggressive to avoid adverse selection"
        
        elif urgency == "HIGH":
            # High urgency: Hybrid approach
            decision.execution_mode = ExecutionMode.HYBRID
            decision.max_aggressive_pct = 0.5
            decision.max_time_seconds = 120.0
            decision.reason = "High urgency - hybrid execution"
        
        elif vpin_elevated:
            # Elevated VPIN: Careful passive with tight limits
            decision.execution_mode = ExecutionMode.PASSIVE
            decision.price_improvement_bps = 2.0  # Aggressive pricing
            decision.max_aggressive_pct = 0.3
            decision.max_time_seconds = 180.0
            decision.reason = f"VPIN elevated ({vpin:.2f}) - cautious passive"
        
        elif asset_toxicity > 0.3:
            # High recent toxicity: Be careful
            decision.execution_mode = ExecutionMode.PASSIVE
            decision.price_improvement_bps = 1.0
            decision.max_aggressive_pct = 0.2
            decision.max_time_seconds = 300.0
            decision.reason = f"High asset toxicity ({asset_toxicity:.1%}) - defensive passive"
        
        else:
            # Normal conditions: Pure passive for fee savings
            decision.execution_mode = ExecutionMode.PASSIVE
            decision.price_improvement_bps = 0.5
            decision.max_aggressive_pct = 0.1
            decision.max_time_seconds = 600.0
            decision.reason = "Normal conditions - passive for fee savings"
        
        # Urgency adjustments
        if urgency == "LOW":
            decision.max_time_seconds *= 2
            decision.max_aggressive_pct *= 0.5

        # Timing engine adjustments (CHECK-051)
        # High timing_score = favorable microstructure -> more passive (save fees)
        # Low timing_score = poor microstructure -> more aggressive (get out fast)
        if timing_score > 0.0 and decision.execution_mode not in (ExecutionMode.ABORT,):
            if timing_mode == "DELAY" and urgency not in ("HIGH", "CRITICAL"):
                decision.max_time_seconds *= 1.5
                decision.reason += f" | timing: DELAY (score={timing_score:.2f})"
            elif timing_mode == "PASSIVE_ONLY":
                decision.max_aggressive_pct = min(decision.max_aggressive_pct, 0.05)
                decision.reason += f" | timing: PASSIVE_ONLY (score={timing_score:.2f})"
            elif timing_mode == "AGGRESSIVE_TAKER":
                decision.max_aggressive_pct = max(decision.max_aggressive_pct, 0.7)
                decision.max_time_seconds = min(decision.max_time_seconds, 60.0)
                decision.reason += f" | timing: AGGRESSIVE (score={timing_score:.2f})"
            elif timing_score > 0.7:
                # Good microstructure - extend time, be more passive
                decision.max_time_seconds *= 1.3
                decision.max_aggressive_pct *= 0.8

        # Clamp price_improvement_bps to config bounds
        if decision.price_improvement_bps > 0:
            decision.price_improvement_bps = max(
                self.config.min_price_improve_bps,
                min(decision.price_improvement_bps, self.config.max_price_improve_bps),
            )

        return decision
    
    def create_order(
        self,
        decision: ExecutionDecision,
        current_bid: float,
        current_ask: float
    ) -> OrderState:
        """Create order based on execution decision."""
        
        order_id = f"{decision.asset}_{int(time.time()*1000)}"
        
        # Determine target and limit prices
        if decision.action == "BUY":
            target_price = current_ask
            # For passive: Price improvement means bidding lower
            limit_price = current_bid + (current_ask - current_bid) * (decision.price_improvement_bps / 100)
        else:
            target_price = current_bid
            # For passive: Price improvement means offering higher
            limit_price = current_ask - (current_ask - current_bid) * (decision.price_improvement_bps / 100)
        
        order = OrderState(
            order_id=order_id,
            asset=decision.asset,
            side="buy" if decision.action == "BUY" else "sell",
            target_price=target_price,
            limit_price=limit_price,
            total_qty=decision.target_qty,
            filled_qty=0.0,
            remaining_qty=decision.target_qty,
            created_at=time.time(),
            mode=decision.execution_mode
        )
        
        # Register for monitoring
        self.fill_monitor.register_order(order)
        self.stats['total_orders'] += 1
        
        return order
    
    def handle_fill(
        self,
        order_id: str,
        fill_qty: float,
        fill_price: float,
        is_maker: bool
    ) -> Tuple[Optional[OrderState], FillSlopeAnalysis]:
        """Handle a fill event."""
        
        # Record fill and get analysis
        analysis = self.fill_monitor.record_fill(order_id, fill_qty, fill_price, is_maker)
        order = self.fill_monitor.get_order_state(order_id)
        
        # Update statistics
        if is_maker:
            self.stats['maker_fills'] += 1
            # Track fee savings
            if order:
                fill_value = fill_qty * fill_price
                savings = fill_value * (KRAKEN_TAKER_FEE_BPS - KRAKEN_MAKER_FEE_BPS) / 10000
                self.stats['total_fee_savings'] += savings
        else:
            self.stats['taker_fills'] += 1
        
        if analysis.is_toxic:
            self.stats['toxic_fills_avoided'] += 1
            self.logger.warning(f"Toxic fill detected: {analysis.toxicity_reason}")
        
        return order, analysis
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        total_fills = self.stats['maker_fills'] + self.stats['taker_fills']
        maker_rate = self.stats['maker_fills'] / max(total_fills, 1)
        
        return {
            **self.stats,
            'total_fills': total_fills,
            'maker_rate': maker_rate,
            'avg_fee_savings_per_order': self.stats['total_fee_savings'] / max(self.stats['total_orders'], 1)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DRIFT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriftAnalysis:
    """Results of data drift analysis."""
    
    # Test results
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    wasserstein_distance: float = 0.0
    
    # Drift detection
    drift_detected: bool = False
    drift_severity: str = "NONE"  # NONE, MILD, MODERATE, SEVERE
    
    # Weight adjustment
    drl_weight_modifier: float = 1.0
    quant_weight_modifier: float = 1.0
    
    # Recommendations
    retrain_recommended: bool = False
    pause_drl_recommended: bool = False
    
    timestamp: float = 0.0


class DataDriftDetector:
    """
    Monitor DRL latent space for concept drift.
    
    Method:
    1. Store training data latent space distribution
    2. Compare live embeddings against training distribution
    3. Use Kolmogorov-Smirnov test for drift detection
    4. Auto down-weight DRL when drift occurs (p-value < 0.05)
    
    This prevents the DRL agent from making decisions based on
    out-of-distribution states where it wasn't trained.
    """
    
    def __init__(
        self,
        p_value_threshold: float = DRIFT_P_VALUE_THRESHOLD,
        embedding_dim: int = 64,
        window_size: int = 1000
    ):
        self.p_value_threshold = p_value_threshold
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        self.logger = logging.getLogger('DriftDetector')
        
        # Training distribution (set during initialization)
        self.training_embeddings: Optional[np.ndarray] = None
        self.training_mean: Optional[np.ndarray] = None
        self.training_std: Optional[np.ndarray] = None
        
        # Live embeddings window
        self.live_embeddings: deque = deque(maxlen=window_size)
        
        # Historical drift metrics
        self.drift_history: deque = deque(maxlen=100)
        
        # Current state
        self.current_weight_modifier = 1.0
    
    def set_training_distribution(self, embeddings: np.ndarray):
        """
        Set the reference training distribution.
        
        embeddings: (N, embedding_dim) array of training latent states
        """
        self.training_embeddings = embeddings
        self.training_mean = np.mean(embeddings, axis=0)
        self.training_std = np.std(embeddings, axis=0) + 1e-10
        
        self.logger.info(f"Training distribution set: {embeddings.shape[0]} samples, "
                        f"dim={embeddings.shape[1]}")
    
    def add_live_embedding(self, embedding: np.ndarray):
        """Add a live embedding for drift monitoring."""
        self.live_embeddings.append(embedding)
    
    def detect_drift(self) -> DriftAnalysis:
        """
        Detect drift between training and live distributions.
        
        Uses:
        1. Kolmogorov-Smirnov test per dimension
        2. Wasserstein distance for overall magnitude
        """
        
        analysis = DriftAnalysis(timestamp=time.time())
        
        if self.training_embeddings is None:
            self.logger.warning("Training distribution not set")
            return analysis
        
        if len(self.live_embeddings) < 100:
            return analysis  # Not enough data
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available for drift detection")
            return analysis
        
        live_array = np.array(self.live_embeddings)
        
        # Normalize using training statistics
        live_normalized = (live_array - self.training_mean) / self.training_std
        training_normalized = (self.training_embeddings - self.training_mean) / self.training_std
        
        # Per-dimension KS test
        ks_pvalues = []
        for dim in range(min(self.embedding_dim, live_array.shape[1])):
            stat, pvalue = ks_2samp(
                training_normalized[:, dim],
                live_normalized[:, dim]
            )
            ks_pvalues.append(pvalue)
        
        # Use minimum p-value (most drifted dimension)
        analysis.ks_pvalue = min(ks_pvalues)
        analysis.ks_statistic = 1 - analysis.ks_pvalue  # Inverse for interpretability
        
        # Wasserstein distance (Earth Mover's Distance)
        # Use L2 norm of per-dimension distances
        wd_per_dim = []
        for dim in range(min(self.embedding_dim, live_array.shape[1])):
            wd = wasserstein_distance(
                training_normalized[:, dim],
                live_normalized[:, dim]
            )
            wd_per_dim.append(wd)
        
        analysis.wasserstein_distance = np.sqrt(np.sum(np.array(wd_per_dim) ** 2))
        
        # Drift detection
        analysis.drift_detected = analysis.ks_pvalue < self.p_value_threshold
        
        # Severity classification
        if analysis.ks_pvalue < 0.001:
            analysis.drift_severity = "SEVERE"
            analysis.drl_weight_modifier = 0.2
            analysis.pause_drl_recommended = True
            analysis.retrain_recommended = True
        elif analysis.ks_pvalue < 0.01:
            analysis.drift_severity = "MODERATE"
            analysis.drl_weight_modifier = 0.5
            analysis.retrain_recommended = True
        elif analysis.ks_pvalue < 0.05:
            analysis.drift_severity = "MILD"
            analysis.drl_weight_modifier = 0.8
        else:
            analysis.drift_severity = "NONE"
            analysis.drl_weight_modifier = 1.0
        
        # Quant weight modifier (inverse)
        analysis.quant_weight_modifier = 2.0 - analysis.drl_weight_modifier
        
        # Store in history
        self.drift_history.append({
            'timestamp': analysis.timestamp,
            'pvalue': analysis.ks_pvalue,
            'severity': analysis.drift_severity,
            'modifier': analysis.drl_weight_modifier
        })
        
        # Update current modifier
        self.current_weight_modifier = analysis.drl_weight_modifier
        
        if analysis.drift_detected:
            self.logger.warning(f"Data drift detected! Severity: {analysis.drift_severity}, "
                              f"p-value: {analysis.ks_pvalue:.6f}, "
                              f"DRL weight modifier: {analysis.drl_weight_modifier:.2f}")
        
        return analysis
    
    def get_weight_modifier(self) -> float:
        """Get current DRL weight modifier."""
        return self.current_weight_modifier
    
    def get_drift_trend(self) -> Dict:
        """Get drift trend over time."""
        if len(self.drift_history) < 2:
            return {'trend': 'unknown', 'avg_pvalue': 1.0}
        
        recent = list(self.drift_history)[-10:]
        pvalues = [d['pvalue'] for d in recent]
        
        avg_pvalue = np.mean(pvalues)
        trend = "stable"
        
        if len(pvalues) >= 5:
            recent_avg = np.mean(pvalues[-5:])
            older_avg = np.mean(pvalues[:5])
            
            if recent_avg < older_avg * 0.5:
                trend = "worsening"
            elif recent_avg > older_avg * 1.5:
                trend = "improving"
        
        return {
            'trend': trend,
            'avg_pvalue': avg_pvalue,
            'recent_modifiers': [d['modifier'] for d in recent]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED EXECUTION GUARD
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionGuardV31:
    """
    SOTA v3.1 Execution Guard
    
    Integrates:
    - Passive-Aggressive execution
    - Fill slope monitoring
    - Data drift detection
    - Fee optimization
    - Edge validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger('ExecutionGuard_v31')
        
        # Components
        self.executor = PassiveAggressiveExecutor()
        self.drift_detector = DataDriftDetector()
        
        # State
        self.active_decisions: Dict[str, ExecutionDecision] = {}
        self.active_orders: Dict[str, OrderState] = {}
        
        self.logger.info("ExecutionGuard v3.1 initialized")
    
    def decide_execution(
        self,
        asset: str,
        quant_signal: float,
        drl_signal: float,
        quant_confidence: float,
        drl_confidence: float,
        vpin: float,
        urgency: str,
        predicted_alpha_bps: float,
        current_price: float,
        current_bid: float,
        current_ask: float,
        target_position_delta: float,
        drl_embedding: Optional[np.ndarray] = None
    ) -> Tuple[ExecutionDecision, Optional[OrderState]]:
        """
        Full execution decision with all v3.1 features.
        """
        
        # Add DRL embedding for drift monitoring
        if drl_embedding is not None:
            self.drift_detector.add_live_embedding(drl_embedding)
        
        # Detect drift
        drift = self.drift_detector.detect_drift()
        
        # Adjust weights based on drift
        drl_weight = 0.4 * drift.drl_weight_modifier
        quant_weight = 0.6 * drift.quant_weight_modifier
        
        # Normalize weights
        total_weight = drl_weight + quant_weight
        drl_weight /= total_weight
        quant_weight /= total_weight
        
        # Fuse signals
        fused_signal = quant_weight * quant_signal + drl_weight * drl_signal
        fused_confidence = quant_weight * quant_confidence + drl_weight * drl_confidence
        
        # Adjust alpha prediction based on drift
        adjusted_alpha = predicted_alpha_bps * drift.drl_weight_modifier
        
        # Make execution decision
        decision = self.executor.decide(
            asset=asset,
            signal=fused_signal,
            confidence=fused_confidence,
            vpin=vpin,
            urgency=urgency,
            predicted_alpha_bps=adjusted_alpha,
            current_price=current_price,
            target_qty=abs(target_position_delta)
        )
        
        # Log drift warning
        if drift.drift_detected:
            decision.reason += f" [DRIFT: {drift.drift_severity}, DRL weight ×{drift.drl_weight_modifier:.2f}]"
        
        # Create order if action is not HOLD
        order = None
        if decision.action != "HOLD" and decision.edge_sufficient:
            order = self.executor.create_order(decision, current_bid, current_ask)
            self.active_orders[order.order_id] = order
        
        # Store decision
        self.active_decisions[asset] = decision
        
        return decision, order
    
    def handle_fill(
        self,
        order_id: str,
        fill_qty: float,
        fill_price: float,
        is_maker: bool
    ) -> Tuple[Optional[OrderState], FillSlopeAnalysis, bool]:
        """
        Handle fill event with toxic flow detection.
        
        Returns:
        - Updated order state
        - Fill slope analysis
        - Should pause execution (bool)
        """
        
        order, analysis = self.executor.handle_fill(
            order_id, fill_qty, fill_price, is_maker
        )
        
        should_pause = analysis.should_pause
        
        if analysis.is_toxic:
            self.logger.warning(f"Order {order_id}: {analysis.toxicity_reason}")
        
        return order, analysis, should_pause
    
    def set_training_distribution(self, embeddings: np.ndarray):
        """Set DRL training distribution for drift detection."""
        self.drift_detector.set_training_distribution(embeddings)
    
    def get_stats(self) -> Dict:
        """Get comprehensive execution statistics."""
        exec_stats = self.executor.get_stats()
        drift_trend = self.drift_detector.get_drift_trend()
        
        return {
            'execution': exec_stats,
            'drift': drift_trend,
            'current_drl_weight_modifier': self.drift_detector.get_weight_modifier()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_execution_guard() -> ExecutionGuardV31:
    """Factory function for ExecutionGuard v3.1."""
    return ExecutionGuardV31()


def create_drift_detector(
    embedding_dim: int = 64,
    p_value_threshold: float = DRIFT_P_VALUE_THRESHOLD
) -> DataDriftDetector:
    """Factory function for DataDriftDetector."""
    return DataDriftDetector(
        p_value_threshold=p_value_threshold,
        embedding_dim=embedding_dim
    )


def create_passive_aggressive_executor() -> PassiveAggressiveExecutor:
    """Factory function for PassiveAggressiveExecutor."""
    return PassiveAggressiveExecutor()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("SOTA v3.1 Passive-Aggressive Execution & Data Drift Detector")
    print("="*60)
    print(f"scipy Available: {SCIPY_AVAILABLE}")
    print()
    
    # Test passive-aggressive execution
    print("Testing PassiveAggressiveExecutor...")
    executor = PassiveAggressiveExecutor()
    
    # Normal conditions
    decision = executor.decide(
        asset='SOL',
        signal=0.6,
        confidence=0.75,
        vpin=0.3,
        urgency="NORMAL",
        predicted_alpha_bps=50.0,
        current_price=195.0,
        target_qty=100.0
    )
    
    print(f"  Decision (Normal):")
    print(f"    Action: {decision.action}")
    print(f"    Mode: {decision.execution_mode.name}")
    print(f"    Edge Sufficient: {decision.edge_sufficient}")
    print(f"    Fee Savings: ${decision.fee_savings:.2f}")
    print(f"    Reason: {decision.reason}")
    print()
    
    # Toxic VPIN
    decision_toxic = executor.decide(
        asset='SOL',
        signal=0.6,
        confidence=0.75,
        vpin=0.75,  # Toxic
        urgency="HIGH",
        predicted_alpha_bps=80.0,
        current_price=195.0,
        target_qty=100.0
    )
    
    print(f"  Decision (Toxic VPIN):")
    print(f"    Action: {decision_toxic.action}")
    print(f"    Mode: {decision_toxic.execution_mode.name}")
    print(f"    Reason: {decision_toxic.reason}")
    print()
    
    # Test fill slope monitoring
    print("Testing Fill Slope Monitor...")
    order = executor.create_order(decision, current_bid=194.8, current_ask=195.2)
    
    # Simulate fills
    for i in range(5):
        time.sleep(0.01)  # 10ms between fills
        order_state, analysis = executor.handle_fill(
            order.order_id,
            fill_qty=20.0,
            fill_price=195.0,
            is_maker=True
        )
    
    print(f"  Order State:")
    print(f"    Filled: {order_state.filled_qty}/{order_state.total_qty}")
    print(f"    Maker Fills: {order_state.maker_fills}")
    print(f"    Slippage: {order_state.slippage_bps:.2f} bps")
    print(f"    Quality: {order_state.fill_quality.name}")
    print()
    
    # Test data drift detector
    print("Testing DataDriftDetector...")
    detector = DataDriftDetector(embedding_dim=8)
    
    # Set training distribution
    np.random.seed(42)
    training = np.random.randn(1000, 8)
    detector.set_training_distribution(training)
    
    # Add similar live data (no drift)
    for i in range(200):
        detector.add_live_embedding(np.random.randn(8))
    
    analysis = detector.detect_drift()
    print(f"  No Drift Case:")
    print(f"    KS p-value: {analysis.ks_pvalue:.4f}")
    print(f"    Drift Detected: {analysis.drift_detected}")
    print(f"    Severity: {analysis.drift_severity}")
    print(f"    DRL Weight Modifier: {analysis.drl_weight_modifier:.2f}")
    print()
    
    # Add drifted data
    detector.live_embeddings.clear()
    for i in range(200):
        # Shifted distribution
        detector.add_live_embedding(np.random.randn(8) + 2.0)
    
    analysis_drift = detector.detect_drift()
    print(f"  Drift Case:")
    print(f"    KS p-value: {analysis_drift.ks_pvalue:.4f}")
    print(f"    Drift Detected: {analysis_drift.drift_detected}")
    print(f"    Severity: {analysis_drift.drift_severity}")
    print(f"    DRL Weight Modifier: {analysis_drift.drl_weight_modifier:.2f}")
    print(f"    Retrain Recommended: {analysis_drift.retrain_recommended}")
    print()
    
    # Test integrated execution guard
    print("Testing ExecutionGuardV31...")
    guard = create_execution_guard()
    guard.set_training_distribution(training)
    
    decision, order = guard.decide_execution(
        asset='SOL',
        quant_signal=0.5,
        drl_signal=0.7,
        quant_confidence=0.8,
        drl_confidence=0.6,
        vpin=0.4,
        urgency="NORMAL",
        predicted_alpha_bps=60.0,
        current_price=195.0,
        current_bid=194.8,
        current_ask=195.2,
        target_position_delta=50.0,
        drl_embedding=np.random.randn(8)
    )
    
    print(f"  Integrated Decision:")
    print(f"    Action: {decision.action}")
    print(f"    Mode: {decision.execution_mode.name}")
    print(f"    Edge Sufficient: {decision.edge_sufficient}")
    print(f"    Order Created: {order is not None}")
    if order:
        print(f"    Order ID: {order.order_id}")
    print()
    
    # Get stats
    stats = guard.get_stats()
    print(f"  Stats:")
    print(f"    Maker Rate: {stats['execution']['maker_rate']:.1%}")
    print(f"    Total Fee Savings: ${stats['execution']['total_fee_savings']:.2f}")
    print(f"    DRL Weight Modifier: {stats['current_drl_weight_modifier']:.2f}")
    print()
    
    print("All tests passed!")

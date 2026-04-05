"""
================================================================================
HMATS v5.2.0 - Level 2 Order Book Analyzer
================================================================================
Purpose: Analyze order book depth for intelligent order placement

Features:
1. Real-time order book depth analysis
2. Liquidity detection at price levels
3. Optimal execution price calculation
4. Market impact prediction from book state
5. Iceberg/hidden order detection heuristics

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


class LiquidityLevel(Enum):
    """Order book liquidity classification."""
    EXCELLENT = "excellent"      # Deep book, tight spread
    GOOD = "good"               # Adequate liquidity
    MODERATE = "moderate"       # Some liquidity concerns
    THIN = "thin"               # Shallow book
    DANGEROUS = "dangerous"     # Very illiquid, high impact risk


class OrderPlacementStrategy(Enum):
    """Recommended order placement strategy."""
    AGGRESSIVE_TAKER = "aggressive_taker"    # Market order or cross spread
    PASSIVE_MAKER = "passive_maker"          # Post limit at best bid/ask
    DEEP_PASSIVE = "deep_passive"            # Post deeper in book
    TWAP_SLICE = "twap_slice"               # Split over time
    ICEBERG = "iceberg"                      # Hide size
    WAIT = "wait"                            # Book too thin, wait


@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    size: float
    count: int = 1                # Number of orders at this level
    
    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: datetime
    
    bids: List[OrderBookLevel]    # Sorted best (highest) to worst
    asks: List[OrderBookLevel]    # Sorted best (lowest) to worst
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread_bps(self) -> float:
        if self.best_bid and self.best_ask and self.mid_price:
            return (self.best_ask - self.best_bid) / self.mid_price * 10000
        return float('inf')


@dataclass
class DepthAnalysis:
    """Results of order book depth analysis."""
    symbol: str
    timestamp: datetime
    
    # Basic metrics
    mid_price: float
    spread_bps: float
    
    # Depth metrics
    bid_depth_1pct: float        # Total bid size within 1% of mid
    ask_depth_1pct: float
    bid_depth_5pct: float        # Total bid size within 5% of mid
    ask_depth_5pct: float
    
    # Imbalance
    book_imbalance: float        # Positive = more bids, negative = more asks
    
    # Liquidity classification
    liquidity_level: LiquidityLevel
    
    # Impact estimates
    buy_impact_100k: float       # Estimated price impact buying $100k
    sell_impact_100k: float
    
    # Recommendations
    recommended_strategy: OrderPlacementStrategy
    max_size_low_impact: float   # Max size with < 10bps impact


@dataclass 
class ExecutionRecommendation:
    """Specific execution recommendation."""
    strategy: OrderPlacementStrategy
    
    # Price targets
    target_price: Optional[float]
    limit_price: Optional[float]
    
    # Size management
    recommended_size: float
    max_single_order: float
    slice_count: int
    
    # Timing
    urgency: str                 # "immediate", "patient", "wait"
    estimated_fill_time_ms: int
    
    # Impact
    expected_impact_bps: float
    confidence: float
    
    # Reasoning
    reasoning: str


class Level2OrderBookAnalyzer:
    """
    Analyzes Level 2 order book data for intelligent order placement.
    
    Features:
    - Real-time depth analysis
    - Impact estimation
    - Optimal execution recommendations
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        
        # Current snapshots
        self.snapshots: Dict[str, OrderBookSnapshot] = {}
        
        # Historical metrics for trend analysis
        self.depth_history: Dict[str, deque] = {
            sym: deque(maxlen=100) for sym in self.symbols
        }
        self.spread_history: Dict[str, deque] = {
            sym: deque(maxlen=100) for sym in self.symbols
        }
        
        # Configuration
        self.impact_threshold_bps = 10.0      # "Low impact" threshold
        self.excellent_spread_bps = 5.0
        self.good_spread_bps = 15.0
        self.thin_depth_threshold = 50000     # USD
        
        logger.info(f"[L2] Level2OrderBookAnalyzer initialized for {self.symbols}")
    
    # =========================================================================
    # DATA UPDATE
    # =========================================================================
    
    def update_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],     # [(price, size), ...]
        asks: List[Tuple[float, float]],
        timestamp: datetime = None
    ):
        """
        Update order book snapshot.
        
        Args:
            symbol: Trading pair
            bids: List of (price, size) tuples, sorted best to worst
            asks: List of (price, size) tuples, sorted best to worst
            timestamp: Snapshot time
        """
        bid_levels = [OrderBookLevel(price=p, size=s) for p, s in bids]
        ask_levels = [OrderBookLevel(price=p, size=s) for p, s in asks]
        
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            timestamp=timestamp or datetime.now(timezone.utc),
            bids=bid_levels,
            asks=ask_levels
        )
        
        self.snapshots[symbol] = snapshot
        
        # Update history
        if snapshot.mid_price:
            self.spread_history[symbol].append(snapshot.spread_bps)
            depth_1pct = self._calculate_depth(snapshot, 0.01)
            self.depth_history[symbol].append(depth_1pct[0] + depth_1pct[1])
    
    # =========================================================================
    # DEPTH ANALYSIS
    # =========================================================================
    
    def analyze_depth(self, symbol: str) -> Optional[DepthAnalysis]:
        """
        Perform comprehensive depth analysis.
        
        Returns:
            DepthAnalysis object with metrics and recommendations
        """
        snapshot = self.snapshots.get(symbol)
        if not snapshot or not snapshot.mid_price:
            return None
        
        mid = snapshot.mid_price
        
        # Calculate depth at different levels
        bid_1pct, ask_1pct = self._calculate_depth(snapshot, 0.01)
        bid_5pct, ask_5pct = self._calculate_depth(snapshot, 0.05)
        
        # Calculate imbalance
        total_near = bid_1pct + ask_1pct
        if total_near > 0:
            imbalance = (bid_1pct - ask_1pct) / total_near
        else:
            imbalance = 0.0
        
        # Classify liquidity
        liquidity = self._classify_liquidity(snapshot, bid_1pct, ask_1pct)
        
        # Estimate impact
        buy_impact = self._estimate_impact(snapshot, 100000, "buy")
        sell_impact = self._estimate_impact(snapshot, 100000, "sell")
        
        # Recommend strategy
        strategy = self._recommend_strategy(snapshot, liquidity, imbalance)
        
        # Calculate max size for low impact
        max_size = self._find_max_low_impact_size(snapshot)
        
        return DepthAnalysis(
            symbol=symbol,
            timestamp=snapshot.timestamp,
            mid_price=mid,
            spread_bps=snapshot.spread_bps,
            bid_depth_1pct=bid_1pct,
            ask_depth_1pct=ask_1pct,
            bid_depth_5pct=bid_5pct,
            ask_depth_5pct=ask_5pct,
            book_imbalance=imbalance,
            liquidity_level=liquidity,
            buy_impact_100k=buy_impact,
            sell_impact_100k=sell_impact,
            recommended_strategy=strategy,
            max_size_low_impact=max_size
        )
    
    def _calculate_depth(
        self,
        snapshot: OrderBookSnapshot,
        pct_from_mid: float
    ) -> Tuple[float, float]:
        """Calculate total depth (USD) within percentage of mid price."""
        mid = snapshot.mid_price
        if not mid:
            return 0.0, 0.0
        
        bid_depth = 0.0
        for level in snapshot.bids:
            if level.price >= mid * (1 - pct_from_mid):
                bid_depth += level.notional
            else:
                break
        
        ask_depth = 0.0
        for level in snapshot.asks:
            if level.price <= mid * (1 + pct_from_mid):
                ask_depth += level.notional
            else:
                break
        
        return bid_depth, ask_depth
    
    def _classify_liquidity(
        self,
        snapshot: OrderBookSnapshot,
        bid_1pct: float,
        ask_1pct: float
    ) -> LiquidityLevel:
        """Classify order book liquidity level."""
        spread = snapshot.spread_bps
        total_depth = bid_1pct + ask_1pct
        
        if spread < self.excellent_spread_bps and total_depth > 500000:
            return LiquidityLevel.EXCELLENT
        elif spread < self.good_spread_bps and total_depth > 200000:
            return LiquidityLevel.GOOD
        elif spread < 30 and total_depth > 100000:
            return LiquidityLevel.MODERATE
        elif spread < 50 and total_depth > self.thin_depth_threshold:
            return LiquidityLevel.THIN
        else:
            return LiquidityLevel.DANGEROUS
    
    # =========================================================================
    # IMPACT ESTIMATION
    # =========================================================================
    
    def _estimate_impact(
        self,
        snapshot: OrderBookSnapshot,
        notional: float,
        side: str
    ) -> float:
        """
        Estimate price impact of executing given notional.
        
        Returns:
            Estimated impact in basis points
        """
        mid = snapshot.mid_price
        if not mid:
            return float('inf')
        
        levels = snapshot.asks if side == "buy" else snapshot.bids
        if not levels:
            return float('inf')
        
        remaining = notional
        total_cost = 0.0
        total_size = 0.0
        
        for level in levels:
            level_notional = level.notional
            if remaining <= level_notional:
                # Partial fill at this level
                fill_size = remaining / level.price
                total_cost += remaining
                total_size += fill_size
                break
            else:
                # Full level consumed
                total_cost += level_notional
                total_size += level.size
                remaining -= level_notional
        
        if total_size == 0:
            return float('inf')
        
        avg_price = total_cost / total_size
        impact_bps = abs(avg_price - mid) / mid * 10000
        
        return impact_bps
    
    def _find_max_low_impact_size(self, snapshot: OrderBookSnapshot) -> float:
        """Find maximum order size with impact below threshold."""
        # Binary search for max size
        low, high = 1000, 1000000
        
        while high - low > 1000:
            mid_size = (low + high) / 2
            impact = max(
                self._estimate_impact(snapshot, mid_size, "buy"),
                self._estimate_impact(snapshot, mid_size, "sell")
            )
            
            if impact <= self.impact_threshold_bps:
                low = mid_size
            else:
                high = mid_size
        
        return low
    
    # =========================================================================
    # STRATEGY RECOMMENDATION
    # =========================================================================
    
    def _recommend_strategy(
        self,
        snapshot: OrderBookSnapshot,
        liquidity: LiquidityLevel,
        imbalance: float
    ) -> OrderPlacementStrategy:
        """Recommend order placement strategy based on book state."""
        
        if liquidity == LiquidityLevel.DANGEROUS:
            return OrderPlacementStrategy.WAIT
        
        if liquidity == LiquidityLevel.THIN:
            return OrderPlacementStrategy.TWAP_SLICE
        
        spread = snapshot.spread_bps
        
        # Excellent liquidity - can be aggressive
        if liquidity == LiquidityLevel.EXCELLENT:
            if spread < 3:
                return OrderPlacementStrategy.AGGRESSIVE_TAKER
            else:
                return OrderPlacementStrategy.PASSIVE_MAKER
        
        # Good liquidity - prefer passive
        if liquidity == LiquidityLevel.GOOD:
            return OrderPlacementStrategy.PASSIVE_MAKER
        
        # Moderate liquidity - be careful
        if abs(imbalance) > 0.3:
            # Heavy imbalance - use iceberg to hide
            return OrderPlacementStrategy.ICEBERG
        
        return OrderPlacementStrategy.PASSIVE_MAKER
    
    def get_execution_recommendation(
        self,
        symbol: str,
        side: str,
        size: float,
        notional: float = None,
        urgency: str = "normal"
    ) -> Optional[ExecutionRecommendation]:
        """
        Get specific execution recommendation for an order.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            size: Order size in base asset
            notional: Order notional in USD (optional, calculated if not provided)
            urgency: "immediate", "normal", or "patient"
            
        Returns:
            ExecutionRecommendation with detailed guidance
        """
        analysis = self.analyze_depth(symbol)
        if not analysis:
            return None
        
        snapshot = self.snapshots[symbol]
        if notional is None:
            notional = size * analysis.mid_price
        
        # Calculate expected impact
        expected_impact = self._estimate_impact(snapshot, notional, side)
        
        # Determine strategy based on analysis and urgency
        if urgency == "immediate":
            strategy = OrderPlacementStrategy.AGGRESSIVE_TAKER
        elif analysis.liquidity_level == LiquidityLevel.DANGEROUS:
            strategy = OrderPlacementStrategy.WAIT
        elif notional > analysis.max_size_low_impact * 2:
            strategy = OrderPlacementStrategy.TWAP_SLICE
        else:
            strategy = analysis.recommended_strategy
        
        # Calculate prices
        if side == "buy":
            target_price = snapshot.best_ask
            limit_price = snapshot.best_ask * 1.001 if urgency == "immediate" else snapshot.best_bid
        else:
            target_price = snapshot.best_bid
            limit_price = snapshot.best_bid * 0.999 if urgency == "immediate" else snapshot.best_ask
        
        # Calculate sizing
        max_single = min(size, analysis.max_size_low_impact / analysis.mid_price)
        slice_count = max(1, int(np.ceil(size / max_single)))
        
        # Estimate fill time
        if strategy == OrderPlacementStrategy.AGGRESSIVE_TAKER:
            fill_time = 100
        elif strategy == OrderPlacementStrategy.PASSIVE_MAKER:
            fill_time = 5000  # 5 seconds typical
        elif strategy == OrderPlacementStrategy.TWAP_SLICE:
            fill_time = slice_count * 60000  # 1 min per slice
        else:
            fill_time = 30000
        
        # Build reasoning
        reasons = []
        reasons.append(f"Liquidity: {analysis.liquidity_level.value}")
        reasons.append(f"Spread: {analysis.spread_bps:.1f}bps")
        reasons.append(f"Book imbalance: {analysis.book_imbalance:+.2f}")
        reasons.append(f"Expected impact: {expected_impact:.1f}bps")
        
        if notional > analysis.max_size_low_impact:
            reasons.append(f"Size exceeds low-impact threshold (${analysis.max_size_low_impact:,.0f})")
        
        return ExecutionRecommendation(
            strategy=strategy,
            target_price=target_price,
            limit_price=limit_price,
            recommended_size=min(size, max_single),
            max_single_order=max_single,
            slice_count=slice_count,
            urgency=urgency,
            estimated_fill_time_ms=fill_time,
            expected_impact_bps=expected_impact,
            confidence=0.8 if analysis.liquidity_level in [LiquidityLevel.EXCELLENT, LiquidityLevel.GOOD] else 0.5,
            reasoning="; ".join(reasons)
        )
    
    # =========================================================================
    # BOOK QUALITY METRICS
    # =========================================================================
    
    def get_book_quality_score(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive book quality metrics."""
        analysis = self.analyze_depth(symbol)
        if not analysis:
            return {"status": "no_data"}
        
        # Calculate rolling averages
        avg_spread = np.mean(list(self.spread_history[symbol])) if self.spread_history[symbol] else None
        avg_depth = np.mean(list(self.depth_history[symbol])) if self.depth_history[symbol] else None
        
        # Quality score (0-100)
        spread_score = max(0, 100 - analysis.spread_bps * 2)
        depth_score = min(100, (analysis.bid_depth_1pct + analysis.ask_depth_1pct) / 5000)
        imbalance_penalty = abs(analysis.book_imbalance) * 20
        
        quality_score = (spread_score + depth_score) / 2 - imbalance_penalty
        quality_score = max(0, min(100, quality_score))
        
        return {
            "symbol": symbol,
            "timestamp": analysis.timestamp.isoformat(),
            "quality_score": quality_score,
            "liquidity_level": analysis.liquidity_level.value,
            "spread_bps": analysis.spread_bps,
            "avg_spread_bps": avg_spread,
            "depth_1pct_usd": analysis.bid_depth_1pct + analysis.ask_depth_1pct,
            "avg_depth_usd": avg_depth,
            "book_imbalance": analysis.book_imbalance,
            "max_low_impact_size": analysis.max_size_low_impact,
            "recommended_strategy": analysis.recommended_strategy.value
        }


# =============================================================================
# SINGLETON
# =============================================================================

_l2_analyzer: Optional[Level2OrderBookAnalyzer] = None


def get_l2_analyzer(symbols: List[str] = None) -> Level2OrderBookAnalyzer:
    """Get or create the global L2 analyzer."""
    global _l2_analyzer
    if _l2_analyzer is None:
        _l2_analyzer = Level2OrderBookAnalyzer(symbols)
    elif list(_l2_analyzer.symbols) != list(symbols or ["BTC", "ETH", "SOL"]):
        logger.info("[L2] Symbols changed; refreshing singleton instance")
        _l2_analyzer = Level2OrderBookAnalyzer(symbols)
    return _l2_analyzer


def reset_l2_analyzer():
    """Reset the global L2 analyzer."""
    global _l2_analyzer
    _l2_analyzer = None


# Alias for backwards compatibility with sota_integration safe_import
Level2Analyzer = Level2OrderBookAnalyzer

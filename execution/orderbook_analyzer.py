"""
================================================================================
HMATS v5.1.0 - Order Book Depth Analyzer
================================================================================
Purpose: Improve order placement with Level 2 order book inspection

SOTA Requirement:
- Improve order placement with order book depth (Level 2 data) inspection.

Features:
1. Real-time order book analysis
2. Liquidity depth assessment
3. Optimal execution price calculation
4. Slippage prediction
5. Book imbalance signals

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    size: float
    order_count: int = 1
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted by price descending
    asks: List[OrderBookLevel]  # Sorted by price ascending
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0


class BookImbalance(Enum):
    """Order book imbalance classification."""
    STRONG_BID = "strong_bid"      # Heavy buy pressure
    SLIGHT_BID = "slight_bid"
    BALANCED = "balanced"
    SLIGHT_ASK = "slight_ask"
    STRONG_ASK = "strong_ask"      # Heavy sell pressure


@dataclass
class DepthAnalysis:
    """Result of order book depth analysis."""
    symbol: str
    timestamp: datetime
    
    # Basic metrics
    mid_price: float
    spread_bps: float
    
    # Depth metrics
    bid_depth_1pct: float      # Total bid size within 1% of mid
    ask_depth_1pct: float
    bid_depth_5pct: float      # Total bid size within 5% of mid
    ask_depth_5pct: float
    
    # Imbalance
    imbalance_ratio: float     # (bid_depth - ask_depth) / total
    imbalance: BookImbalance
    
    # Execution estimates
    estimated_slippage_buy: float   # bps for given size
    estimated_slippage_sell: float
    optimal_buy_price: float
    optimal_sell_price: float
    
    # Liquidity quality
    liquidity_score: float     # 0-1, higher is better
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
            "bid_depth_1pct": self.bid_depth_1pct,
            "ask_depth_1pct": self.ask_depth_1pct,
            "imbalance_ratio": self.imbalance_ratio,
            "imbalance": self.imbalance.value,
            "estimated_slippage_buy": self.estimated_slippage_buy,
            "estimated_slippage_sell": self.estimated_slippage_sell,
            "liquidity_score": self.liquidity_score
        }


@dataclass
class ExecutionRecommendation:
    """Recommendation for order execution."""
    should_execute: bool
    recommended_price: float
    recommended_size: float
    execution_mode: str  # "market", "limit_aggressive", "limit_passive", "split"
    expected_slippage_bps: float
    reasoning: str
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# ORDER BOOK DEPTH ANALYZER
# =============================================================================

class OrderBookDepthAnalyzer:
    """
    Analyzes order book depth to optimize execution.
    
    Key functions:
    1. Calculate liquidity at various price levels
    2. Estimate slippage for given order sizes
    3. Detect order book imbalances
    4. Recommend optimal execution strategy
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Analysis parameters
        self.depth_levels = [0.01, 0.02, 0.05, 0.10]  # 1%, 2%, 5%, 10%
        self.imbalance_threshold = 0.3  # 30% imbalance
        self.min_liquidity_score = 0.3
        
        # Order book cache
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.analysis_cache: Dict[str, DepthAnalysis] = {}
        
        # Historical tracking
        self.spread_history: Dict[str, deque] = {}
        self.imbalance_history: Dict[str, deque] = {}
        
        # Stats
        self.stats = {
            "analyses_performed": 0,
            "avg_spread_bps": {},
            "avg_liquidity_score": {}
        }
        
        logger.info("[OrderBookAnalyzer] Initialized")
    
    # =========================================================================
    # ORDER BOOK UPDATES
    # =========================================================================
    
    def update_order_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],  # [(price, size), ...]
        asks: List[Tuple[float, float]],
        timestamp: datetime = None
    ):
        """
        Update order book snapshot.
        
        Args:
            symbol: Trading pair
            bids: List of (price, size) tuples, best first
            asks: List of (price, size) tuples, best first
            timestamp: Snapshot timestamp
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        
        # Convert to OrderBookLevel
        bid_levels = [OrderBookLevel(price=p, size=s) for p, s in bids]
        ask_levels = [OrderBookLevel(price=p, size=s) for p, s in asks]
        
        # Sort
        bid_levels.sort(key=lambda x: x.price, reverse=True)
        ask_levels.sort(key=lambda x: x.price)
        
        self.order_books[symbol] = OrderBookSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            bids=bid_levels,
            asks=ask_levels
        )
        
        # Initialize history if needed
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=100)
            self.imbalance_history[symbol] = deque(maxlen=100)
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def analyze(self, symbol: str, order_size_usd: float = 10000) -> Optional[DepthAnalysis]:
        """
        Perform comprehensive order book analysis.
        
        Args:
            symbol: Symbol to analyze
            order_size_usd: Reference size for slippage estimation
            
        Returns:
            DepthAnalysis or None if no data
        """
        book = self.order_books.get(symbol)
        if not book or not book.bids or not book.asks:
            return None
        
        mid = book.mid_price
        
        # Calculate depth at various levels
        bid_depth_1pct = self._calculate_depth(book.bids, mid, 0.01)
        ask_depth_1pct = self._calculate_depth(book.asks, mid, 0.01)
        bid_depth_5pct = self._calculate_depth(book.bids, mid, 0.05)
        ask_depth_5pct = self._calculate_depth(book.asks, mid, 0.05)
        
        # Calculate imbalance
        total_depth = bid_depth_1pct + ask_depth_1pct
        if total_depth > 0:
            imbalance_ratio = (bid_depth_1pct - ask_depth_1pct) / total_depth
        else:
            imbalance_ratio = 0.0
        
        imbalance = self._classify_imbalance(imbalance_ratio)
        
        # Estimate slippage
        slippage_buy = self._estimate_slippage(book.asks, order_size_usd, mid)
        slippage_sell = self._estimate_slippage(book.bids, order_size_usd, mid, is_sell=True)
        
        # Calculate optimal prices
        optimal_buy = self._calculate_optimal_price(book.asks, order_size_usd, is_buy=True)
        optimal_sell = self._calculate_optimal_price(book.bids, order_size_usd, is_buy=False)
        
        # Liquidity score
        liquidity_score = self._calculate_liquidity_score(
            book, bid_depth_1pct, ask_depth_1pct, order_size_usd
        )
        
        analysis = DepthAnalysis(
            symbol=symbol,
            timestamp=book.timestamp,
            mid_price=mid,
            spread_bps=book.spread_bps,
            bid_depth_1pct=bid_depth_1pct,
            ask_depth_1pct=ask_depth_1pct,
            bid_depth_5pct=bid_depth_5pct,
            ask_depth_5pct=ask_depth_5pct,
            imbalance_ratio=imbalance_ratio,
            imbalance=imbalance,
            estimated_slippage_buy=slippage_buy,
            estimated_slippage_sell=slippage_sell,
            optimal_buy_price=optimal_buy,
            optimal_sell_price=optimal_sell,
            liquidity_score=liquidity_score
        )
        
        # Update cache and history
        self.analysis_cache[symbol] = analysis
        self.spread_history[symbol].append(book.spread_bps)
        self.imbalance_history[symbol].append(imbalance_ratio)
        
        # Update stats
        self.stats["analyses_performed"] += 1
        
        return analysis
    
    def _calculate_depth(
        self, 
        levels: List[OrderBookLevel], 
        mid_price: float, 
        pct_range: float
    ) -> float:
        """Calculate total depth (USD) within price range."""
        total = 0.0
        for level in levels:
            if abs(level.price - mid_price) / mid_price <= pct_range:
                total += level.value
            else:
                break  # Assuming sorted
        return total
    
    def _classify_imbalance(self, ratio: float) -> BookImbalance:
        """Classify imbalance ratio."""
        if ratio > 0.5:
            return BookImbalance.STRONG_BID
        elif ratio > 0.2:
            return BookImbalance.SLIGHT_BID
        elif ratio < -0.5:
            return BookImbalance.STRONG_ASK
        elif ratio < -0.2:
            return BookImbalance.SLIGHT_ASK
        else:
            return BookImbalance.BALANCED
    
    def _estimate_slippage(
        self, 
        levels: List[OrderBookLevel], 
        size_usd: float,
        mid_price: float,
        is_sell: bool = False
    ) -> float:
        """Estimate slippage in bps for given order size."""
        if not levels or mid_price <= 0:
            return 100.0  # Default high slippage
        
        remaining = size_usd
        total_cost = 0.0
        
        for level in levels:
            level_value = level.value
            if remaining <= level_value:
                total_cost += remaining
                remaining = 0
                break
            else:
                total_cost += level_value
                remaining -= level_value
        
        if remaining > 0:
            # Not enough liquidity
            return 200.0  # Very high slippage
        
        avg_price = total_cost / (size_usd / levels[0].price) if levels[0].price > 0 else mid_price
        slippage_pct = abs(avg_price - mid_price) / mid_price
        
        return slippage_pct * 10000  # Convert to bps
    
    def _calculate_optimal_price(
        self,
        levels: List[OrderBookLevel],
        size_usd: float,
        is_buy: bool
    ) -> float:
        """Calculate optimal limit price for given size."""
        if not levels:
            return 0.0
        
        # Find price level where we can fill most of the order
        cumulative = 0.0
        for level in levels:
            cumulative += level.value
            if cumulative >= size_usd * 0.8:  # 80% fill target
                return level.price
        
        # If not enough liquidity, return worst level
        return levels[-1].price if levels else 0.0
    
    def _calculate_liquidity_score(
        self,
        book: OrderBookSnapshot,
        bid_depth: float,
        ask_depth: float,
        reference_size: float
    ) -> float:
        """Calculate liquidity quality score (0-1)."""
        score = 0.0
        
        # Spread component (lower is better)
        if book.spread_bps < 5:
            score += 0.3
        elif book.spread_bps < 15:
            score += 0.2
        elif book.spread_bps < 30:
            score += 0.1
        
        # Depth component (higher is better)
        depth_ratio = min(bid_depth, ask_depth) / reference_size if reference_size > 0 else 0
        score += min(0.4, depth_ratio * 0.1)
        
        # Balance component (more balanced is better)
        if bid_depth > 0 and ask_depth > 0:
            balance = min(bid_depth, ask_depth) / max(bid_depth, ask_depth)
            score += balance * 0.3
        
        return min(1.0, score)
    
    # =========================================================================
    # EXECUTION RECOMMENDATIONS
    # =========================================================================
    
    def get_execution_recommendation(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        size_usd: float,
        urgency: str = "medium"  # "low", "medium", "high"
    ) -> ExecutionRecommendation:
        """
        Get recommendation for order execution.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            size_usd: Order size in USD
            urgency: Execution urgency
            
        Returns:
            ExecutionRecommendation
        """
        analysis = self.analyze(symbol, size_usd)
        
        if analysis is None:
            return ExecutionRecommendation(
                should_execute=False,
                recommended_price=0.0,
                recommended_size=0.0,
                execution_mode="wait",
                expected_slippage_bps=0.0,
                reasoning="No order book data available",
                warnings=["Missing order book data"]
            )
        
        warnings = []
        
        # Check liquidity
        if analysis.liquidity_score < self.min_liquidity_score:
            warnings.append(f"Low liquidity score: {analysis.liquidity_score:.2f}")
        
        # Check spread
        if analysis.spread_bps > 30:
            warnings.append(f"Wide spread: {analysis.spread_bps:.1f} bps")
        
        # Determine execution mode based on conditions
        is_buy = side.lower() == "buy"
        slippage = analysis.estimated_slippage_buy if is_buy else analysis.estimated_slippage_sell
        
        # Check for adverse imbalance
        if is_buy and analysis.imbalance in [BookImbalance.STRONG_ASK, BookImbalance.SLIGHT_ASK]:
            warnings.append("Order book imbalance unfavorable for buy")
        elif not is_buy and analysis.imbalance in [BookImbalance.STRONG_BID, BookImbalance.SLIGHT_BID]:
            warnings.append("Order book imbalance unfavorable for sell")
        
        # Determine execution mode
        if urgency == "high" or slippage < 5:
            execution_mode = "market"
            price = analysis.optimal_buy_price if is_buy else analysis.optimal_sell_price
        elif slippage < 20:
            execution_mode = "limit_aggressive"
            book = self.order_books.get(symbol)
            if book:
                # Place slightly better than best
                if is_buy:
                    price = book.best_ask - (book.spread * 0.1)
                else:
                    price = book.best_bid + (book.spread * 0.1)
            else:
                price = analysis.optimal_buy_price if is_buy else analysis.optimal_sell_price
        elif size_usd > 50000:
            execution_mode = "split"
            price = analysis.optimal_buy_price if is_buy else analysis.optimal_sell_price
            warnings.append("Consider splitting large order")
        else:
            execution_mode = "limit_passive"
            book = self.order_books.get(symbol)
            if book:
                price = book.best_bid if is_buy else book.best_ask
            else:
                price = analysis.mid_price
        
        # Determine if we should execute
        should_execute = True
        if analysis.liquidity_score < 0.2:
            should_execute = False
            warnings.append("Insufficient liquidity")
        if slippage > 100:
            should_execute = False
            warnings.append("Slippage too high")
        
        reasoning = (
            f"Spread={analysis.spread_bps:.1f}bps, "
            f"Liquidity={analysis.liquidity_score:.2f}, "
            f"Imbalance={analysis.imbalance.value}, "
            f"Est.Slippage={slippage:.1f}bps"
        )
        
        return ExecutionRecommendation(
            should_execute=should_execute,
            recommended_price=price,
            recommended_size=size_usd,
            execution_mode=execution_mode,
            expected_slippage_bps=slippage,
            reasoning=reasoning,
            warnings=warnings
        )
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_current_analysis(self, symbol: str) -> Optional[Dict]:
        """Get cached analysis for symbol."""
        analysis = self.analysis_cache.get(symbol)
        return analysis.to_dict() if analysis else None
    
    def get_spread_stats(self, symbol: str) -> Dict:
        """Get spread statistics for symbol."""
        history = self.spread_history.get(symbol, [])
        if not history:
            return {"avg": 0, "min": 0, "max": 0, "current": 0}
        
        return {
            "avg": np.mean(history),
            "min": np.min(history),
            "max": np.max(history),
            "current": history[-1] if history else 0
        }
    
    def get_imbalance_signal(self, symbol: str) -> Dict:
        """Get imbalance signal for trading."""
        analysis = self.analysis_cache.get(symbol)
        if not analysis:
            return {"signal": "neutral", "strength": 0}
        
        # Strong bid imbalance suggests price may rise
        if analysis.imbalance == BookImbalance.STRONG_BID:
            return {"signal": "bullish", "strength": 0.8}
        elif analysis.imbalance == BookImbalance.SLIGHT_BID:
            return {"signal": "bullish", "strength": 0.4}
        elif analysis.imbalance == BookImbalance.STRONG_ASK:
            return {"signal": "bearish", "strength": 0.8}
        elif analysis.imbalance == BookImbalance.SLIGHT_ASK:
            return {"signal": "bearish", "strength": 0.4}
        else:
            return {"signal": "neutral", "strength": 0}


# =============================================================================
# SINGLETON
# =============================================================================

_analyzer: Optional[OrderBookDepthAnalyzer] = None


def get_orderbook_analyzer(config: Dict = None) -> OrderBookDepthAnalyzer:
    """Get or create the global order book analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OrderBookDepthAnalyzer(config)
    elif config is not None and (_analyzer.config or {}) != (config or {}):
        logger.info("[ORDERBOOK] Config changed; refreshing singleton instance")
        _analyzer = OrderBookDepthAnalyzer(config)
    return _analyzer


def reset_orderbook_analyzer():
    """Reset the global analyzer."""
    global _analyzer
    _analyzer = None

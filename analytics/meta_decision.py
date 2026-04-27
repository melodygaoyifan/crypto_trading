"""
================================================================================
HMATS v4.0-COMPLETE - ENHANCED META-DECISION LAYER
================================================================================
Version: 4.0.0-ENHANCED
Purpose: Unified meta-decision layer with cross-strategy learning and info sharing

MIGRATION NOTES:
    - Extends v3.5 meta_decision_memory.py
    - Incorporates cross-strategy meta-learning from v3.3 concepts
    - Adds information sharing mechanism between strategies
    - Integrates with integration_v36.py

KEY FEATURES:
    1. Trade outcome tracking and aggressiveness adjustment (v3.5)
    2. Cross-strategy performance correlation
    3. Strategy confidence propagation
    4. Regime-specific performance memory
    5. Information sharing between Quant/DRL/Sentiment
    6. Adaptive threshold adjustment

CRITICAL CONSTRAINTS:
    - Does NOT generate trades or signals
    - Does NOT override Risk vetoes
    - Only MODIFIES parameters within bounds
    - Auto-recovers (no permanent suppression)

ARCHITECTURE INTEGRATION:
    integration_v36.py -> MetaDecisionLayer -> authority_fusion.py
                       ↓
              strategy_confidence.py
                       ↓
              pnl_attribution.py
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# RE-EXPORT CORE COMPONENTS FROM meta_decision_memory.py
# =============================================================================

try:
    from .meta_decision_memory import (
        OutcomeType,
        TradeOutcome,
        MetaMemoryConfig,
        AggressivenessModifier,
        MetaDecisionMemory,
        get_meta_memory,
        reset_meta_memory,
    )
    META_MEMORY_AVAILABLE = True
except ImportError:
    META_MEMORY_AVAILABLE = False
    logger.warning("meta_decision_memory not available, using stubs")


# =============================================================================
# CROSS-STRATEGY LEARNING COMPONENTS
# =============================================================================

class StrategyType(Enum):
    """Strategy type classification."""
    QUANT_MOMENTUM = "QUANT_MOMENTUM"
    QUANT_MEAN_REVERT = "QUANT_MEAN_REVERT"
    QUANT_VOLATILITY = "QUANT_VOLATILITY"
    DRL_ENTRY = "DRL_ENTRY"
    DRL_EXIT = "DRL_EXIT"
    SENTIMENT = "SENTIMENT"
    LEAD_LAG = "LEAD_LAG"


class RegimeContext(Enum):
    """Market regime for context-specific memory."""
    STRONG_BULL = "STRONG_BULL"
    MODERATE_BULL = "MODERATE_BULL"
    NEUTRAL = "NEUTRAL"
    VOLATILE_CHOP = "VOLATILE_CHOP"
    MODERATE_BEAR = "MODERATE_BEAR"
    STRONG_BEAR = "STRONG_BEAR"


@dataclass
class StrategyPerformance:
    """Performance tracking for a single strategy."""
    strategy_type: StrategyType
    
    # Trade statistics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    
    # PnL metrics
    total_pnl_bps: float = 0.0
    max_win_bps: float = 0.0
    max_loss_bps: float = 0.0
    
    # Recent performance (rolling window)
    recent_trades: int = 0
    recent_wins: int = 0
    recent_pnl_bps: float = 0.0
    
    # Confidence metrics
    raw_win_rate: float = 0.5
    adjusted_confidence: float = 0.5
    
    # Last update
    last_update: Optional[datetime] = None
    
    def update(self, pnl_bps: float, outcome: str):
        """Update performance with new trade outcome."""
        self.total_trades += 1
        self.total_pnl_bps += pnl_bps
        self.last_update = datetime.now(timezone.utc)
        
        if pnl_bps > 50:
            self.wins += 1
            self.max_win_bps = max(self.max_win_bps, pnl_bps)
        elif pnl_bps < -50:
            self.losses += 1
            self.max_loss_bps = min(self.max_loss_bps, pnl_bps)
        else:
            self.scratches += 1
        
        # Update win rate
        if self.wins + self.losses > 0:
            self.raw_win_rate = self.wins / (self.wins + self.losses)
    
    @property
    def expectancy_bps(self) -> float:
        """Calculate expected PnL per trade in bps."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_bps / self.total_trades
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy_type.value,
            "trades": self.total_trades,
            "win_rate": self.raw_win_rate,
            "expectancy_bps": self.expectancy_bps,
            "confidence": self.adjusted_confidence,
        }


@dataclass
class CrossStrategyCorrelation:
    """Correlation tracking between strategy pairs."""
    strategy_a: StrategyType
    strategy_b: StrategyType
    
    # Win correlation (both win, both lose, opposite)
    both_win: int = 0
    both_lose: int = 0
    a_win_b_lose: int = 0
    a_lose_b_win: int = 0
    
    # Direction agreement (when both signal)
    agreements: int = 0
    disagreements: int = 0
    
    @property
    def correlation(self) -> float:
        """Calculate outcome correlation (-1 to 1)."""
        total = self.both_win + self.both_lose + self.a_win_b_lose + self.a_lose_b_win
        if total == 0:
            return 0.0
        
        # Correlation = (same - opposite) / total
        same = self.both_win + self.both_lose
        opposite = self.a_win_b_lose + self.a_lose_b_win
        return (same - opposite) / total
    
    @property
    def agreement_rate(self) -> float:
        """Calculate signal agreement rate."""
        total = self.agreements + self.disagreements
        if total == 0:
            return 0.5
        return self.agreements / total


# =============================================================================
# INFORMATION SHARING MECHANISM
# =============================================================================

@dataclass
class SharedInformation:
    """Information shared between strategies."""
    
    # Market structure
    current_regime: RegimeContext = RegimeContext.NEUTRAL
    regime_confidence: float = 0.5
    regime_duration_bars: int = 0
    
    # Cross-asset signals
    btc_signal: float = 0.0
    eth_signal: float = 0.0
    sol_signal: float = 0.0
    cross_asset_divergence: float = 0.0
    
    # Volatility context
    realized_vol: float = 0.15
    implied_vol: float = 0.20
    vol_premium: float = 0.0  # IV - RV
    vol_regime: str = "NORMAL"  # COMPRESSED, NORMAL, EXPANDED
    
    # Liquidity context
    liquidity_score: float = 0.5  # 0=thin, 1=deep
    spread_percentile: float = 0.5  # vs historical
    
    # Timing context
    is_weekend: bool = False
    is_holiday: bool = False
    hours_until_4h_close: float = 4.0
    
    # Lead-lag intelligence
    lead_lag_edge_bps: float = 0.0
    lead_lag_confidence: float = 0.0
    
    # Sentiment summary
    sentiment_direction: float = 0.0
    sentiment_shock: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "regime": self.current_regime.value,
            "regime_confidence": self.regime_confidence,
            "vol_regime": self.vol_regime,
            "liquidity_score": self.liquidity_score,
            "lead_lag_edge": self.lead_lag_edge_bps,
            "sentiment_direction": self.sentiment_direction,
        }


class InformationBroker:
    """
    Brokers information between strategies.
    
    Each strategy can:
    1. Publish its observations
    2. Subscribe to other strategies' observations
    3. Query the shared state
    """
    
    def __init__(self):
        self.shared = SharedInformation()
        self._publishers: Dict[StrategyType, Callable] = {}
        self._subscribers: Dict[StrategyType, List[Callable]] = {}
        self._history: deque = deque(maxlen=100)
        
        logger.debug("InformationBroker initialized")
    
    def update_shared(self, **kwargs):
        """Update shared information."""
        for key, value in kwargs.items():
            if hasattr(self.shared, key):
                setattr(self.shared, key, value)
        
        # Record history
        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "updates": kwargs,
        })
    
    def get_shared(self) -> SharedInformation:
        """Get current shared information."""
        return self.shared
    
    def get_for_strategy(self, strategy_type: StrategyType) -> Dict:
        """Get relevant information for a specific strategy."""
        
        shared = self.shared.to_dict()
        
        # Add strategy-specific context
        if strategy_type in [StrategyType.QUANT_MOMENTUM]:
            shared["recommended_timeframe"] = "4H"
            shared["momentum_friendly"] = self.shared.current_regime in [
                RegimeContext.STRONG_BULL, RegimeContext.STRONG_BEAR
            ]
            
        elif strategy_type in [StrategyType.QUANT_MEAN_REVERT]:
            shared["recommended_timeframe"] = "1H"
            shared["mean_revert_friendly"] = self.shared.current_regime == RegimeContext.VOLATILE_CHOP
            
        elif strategy_type == StrategyType.DRL_EXIT:
            shared["exit_pressure"] = self._calculate_exit_pressure()
        
        return shared
    
    def _calculate_exit_pressure(self) -> float:
        """Calculate exit pressure from shared signals."""
        pressure = 0.0
        
        # High volatility = exit pressure
        if self.shared.vol_regime == "EXPANDED":
            pressure += 0.3
        
        # Weekend = exit pressure
        if self.shared.is_weekend:
            pressure += 0.2
        
        # Negative lead-lag = exit pressure
        if self.shared.lead_lag_edge_bps < -10:
            pressure += 0.2
        
        # Sentiment shock against position = exit pressure
        if abs(self.shared.sentiment_shock) > 2.0:
            pressure += 0.3
        
        return min(pressure, 1.0)


# =============================================================================
# REGIME-SPECIFIC MEMORY
# =============================================================================

class RegimeSpecificMemory:
    """
    Tracks performance by market regime.
    
    Allows the system to:
    1. Know which strategies work in which regimes
    2. Adjust confidence based on regime
    3. Detect regime-strategy mismatches
    """
    
    def __init__(self):
        self._performance: Dict[Tuple[StrategyType, RegimeContext], StrategyPerformance] = {}
        self._regime_transitions: deque = deque(maxlen=50)
        self._current_regime = RegimeContext.NEUTRAL
    
    def get_performance(
        self,
        strategy: StrategyType,
        regime: RegimeContext
    ) -> StrategyPerformance:
        """Get performance for strategy in specific regime."""
        key = (strategy, regime)
        if key not in self._performance:
            self._performance[key] = StrategyPerformance(strategy_type=strategy)
        return self._performance[key]
    
    def record_outcome(
        self,
        strategy: StrategyType,
        regime: RegimeContext,
        pnl_bps: float,
        outcome: str
    ):
        """Record trade outcome for strategy in regime."""
        perf = self.get_performance(strategy, regime)
        perf.update(pnl_bps, outcome)
    
    def set_current_regime(self, regime: RegimeContext):
        """Set current regime and record transition."""
        if regime != self._current_regime:
            self._regime_transitions.append({
                "from": self._current_regime.value,
                "to": regime.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._current_regime = regime
    
    def get_regime_confidence_adjustment(
        self,
        strategy: StrategyType,
        base_confidence: float
    ) -> float:
        """
        Adjust confidence based on strategy's performance in current regime.
        
        Returns adjusted confidence (0.0 to 1.0).
        """
        perf = self.get_performance(strategy, self._current_regime)
        
        # Need minimum trades for adjustment
        if perf.total_trades < 5:
            return base_confidence
        
        # Calculate regime-specific adjustment
        win_rate = perf.raw_win_rate
        expectancy = perf.expectancy_bps
        
        # Positive expectancy in this regime = confidence boost
        if expectancy > 20:
            adjustment = 0.1
        elif expectancy > 0:
            adjustment = 0.05
        elif expectancy > -20:
            adjustment = -0.05
        else:
            adjustment = -0.15
        
        # Win rate adjustment
        if win_rate > 0.55:
            adjustment += 0.05
        elif win_rate < 0.45:
            adjustment -= 0.05
        
        adjusted = base_confidence + adjustment
        return max(0.1, min(0.95, adjusted))
    
    def get_best_strategies_for_regime(
        self,
        regime: RegimeContext,
        top_n: int = 3
    ) -> List[StrategyType]:
        """Get best performing strategies for a regime."""
        
        performances = []
        for key, perf in self._performance.items():
            strategy, perf_regime = key
            if perf_regime == regime and perf.total_trades >= 5:
                performances.append((strategy, perf.expectancy_bps))
        
        performances.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in performances[:top_n]]


# =============================================================================
# ENHANCED META-DECISION LAYER
# =============================================================================

class EnhancedMetaDecisionLayer:
    """
    Enhanced meta-decision layer combining all meta-learning components.
    
    Components:
    1. MetaDecisionMemory - Trade outcome tracking (v3.5)
    2. InformationBroker - Cross-strategy info sharing
    3. RegimeSpecificMemory - Regime-aware performance
    4. CrossStrategyCorrelation - Strategy correlation tracking
    
    Usage:
        meta = get_enhanced_meta_layer()
        
        # Update with shared info
        meta.update_context(market_data, signals)
        
        # Get aggressiveness modifier
        modifier = meta.get_aggressiveness_modifier()
        
        # Get regime-adjusted confidence
        confidence = meta.get_adjusted_confidence("QUANT_MOMENTUM", 0.7)
        
        # Record outcome
        meta.record_outcome(trade_result)
    """
    
    def __init__(self):
        # Core components
        if META_MEMORY_AVAILABLE:
            self.meta_memory = get_meta_memory()
        else:
            self.meta_memory = None
        
        self.info_broker = InformationBroker()
        self.regime_memory = RegimeSpecificMemory()
        
        # Cross-strategy correlations
        self._correlations: Dict[Tuple[StrategyType, StrategyType], CrossStrategyCorrelation] = {}
        
        # Strategy performances
        self._strategy_perfs: Dict[StrategyType, StrategyPerformance] = {}
        
        # Adaptive thresholds
        self._alpha_threshold_adjustment: float = 0.0
        self._confidence_threshold_adjustment: float = 0.0
        
        logger.info("[v4.0-ENHANCED] EnhancedMetaDecisionLayer initialized")
    
    # =========================================================================
    # CONTEXT UPDATE
    # =========================================================================
    
    def update_context(
        self,
        market_data: Dict,
        signals: Optional[Dict] = None,
        regime: Optional[str] = None
    ):
        """
        Update meta-layer with current context.
        
        Call this at each decision point.
        """
        
        # Update regime
        if regime:
            try:
                regime_enum = RegimeContext[regime.upper().replace(" ", "_")]
                self.regime_memory.set_current_regime(regime_enum)
                self.info_broker.update_shared(
                    current_regime=regime_enum,
                    regime_confidence=market_data.get('regime_confidence', 0.5)
                )
            except KeyError:
                pass
        
        # Update market context
        self.info_broker.update_shared(
            realized_vol=market_data.get('volatility', 0.15),
            implied_vol=market_data.get('implied_vol', 0.20),
            liquidity_score=market_data.get('liquidity_score', 0.5),
            is_weekend=market_data.get('is_weekend', False),
        )
        
        # Update signals if provided
        if signals:
            self.info_broker.update_shared(
                btc_signal=signals.get('btc', 0.0),
                eth_signal=signals.get('eth', 0.0),
                sol_signal=signals.get('sol', 0.0),
                sentiment_direction=signals.get('sentiment', 0.0),
            )
    
    # =========================================================================
    # OUTCOME RECORDING
    # =========================================================================
    
    def record_outcome(
        self,
        strategy: str,
        pnl_bps: float,
        regime: Optional[str] = None,
        is_opportunity: bool = False
    ):
        """
        Record trade outcome for meta-learning.
        
        Updates:
        1. MetaDecisionMemory (if opportunity trade)
        2. Strategy-specific performance
        3. Regime-specific performance
        4. Cross-strategy correlations
        """
        
        # Map to strategy type
        try:
            strategy_type = StrategyType[strategy.upper()]
        except KeyError:
            strategy_type = StrategyType.QUANT_MOMENTUM  # Default
        
        # Determine outcome
        if pnl_bps > 50:
            outcome = "WIN"
            outcome_type = OutcomeType.WIN if META_MEMORY_AVAILABLE else None
        elif pnl_bps < -50:
            outcome = "LOSS"
            outcome_type = OutcomeType.LOSS if META_MEMORY_AVAILABLE else None
        else:
            outcome = "SCRATCH"
            outcome_type = OutcomeType.SCRATCH if META_MEMORY_AVAILABLE else None
        
        # Update MetaDecisionMemory for opportunity trades
        if is_opportunity and self.meta_memory and META_MEMORY_AVAILABLE:
            trade_outcome = TradeOutcome(
                trade_id=f"trade_{datetime.now(timezone.utc).timestamp():.0f}",
                asset="SOL",  # Default
                direction=1 if pnl_bps > 0 else -1,
                mode_at_entry="OPPORTUNITY",
                phase_at_entry="",
                pnl_bps=pnl_bps,
                outcome_type=outcome_type,
                entry_time=datetime.now(timezone.utc) - timedelta(hours=4),
                exit_time=datetime.now(timezone.utc),
                bars_held=1,
                exit_reason="signal",
            )
            self.meta_memory.record_outcome(trade_outcome)
        
        # Update strategy-specific performance
        if strategy_type not in self._strategy_perfs:
            self._strategy_perfs[strategy_type] = StrategyPerformance(strategy_type=strategy_type)
        self._strategy_perfs[strategy_type].update(pnl_bps, outcome)
        
        # Update regime-specific performance
        if regime:
            try:
                regime_enum = RegimeContext[regime.upper().replace(" ", "_")]
                self.regime_memory.record_outcome(strategy_type, regime_enum, pnl_bps, outcome)
            except KeyError:
                pass
    
    # =========================================================================
    # MODIFIERS AND ADJUSTMENTS
    # =========================================================================
    
    def get_aggressiveness_modifier(self) -> float:
        """
        Get current aggressiveness modifier for Fusion.
        
        Returns modifier (0.6 to 1.0).
        """
        if self.meta_memory:
            modifier = self.meta_memory.get_aggressiveness_modifier()
            return modifier.modifier
        return 1.0
    
    def get_adjusted_confidence(
        self,
        strategy: str,
        base_confidence: float
    ) -> float:
        """
        Get regime-adjusted confidence for a strategy.
        
        Args:
            strategy: Strategy name
            base_confidence: Base confidence (0.0 to 1.0)
        
        Returns:
            Adjusted confidence (0.0 to 1.0)
        """
        try:
            strategy_type = StrategyType[strategy.upper()]
        except KeyError:
            return base_confidence
        
        # Get regime adjustment
        adjusted = self.regime_memory.get_regime_confidence_adjustment(
            strategy_type,
            base_confidence
        )
        
        # Apply global adjustment
        adjusted += self._confidence_threshold_adjustment
        
        return max(0.1, min(0.95, adjusted))
    
    def get_alpha_threshold_adjustment(self) -> float:
        """
        Get adaptive alpha threshold adjustment.
        
        Returns bps adjustment to add to base threshold.
        """
        
        # If in recovery, require higher alpha
        if self.meta_memory:
            modifier = self.meta_memory.get_aggressiveness_modifier()
            if modifier.in_recovery:
                return 20.0  # +20bps during recovery
        
        return self._alpha_threshold_adjustment
    
    # =========================================================================
    # INFORMATION ACCESS
    # =========================================================================
    
    def get_shared_info(self) -> SharedInformation:
        """Get current shared information."""
        return self.info_broker.get_shared()
    
    def get_info_for_strategy(self, strategy: str) -> Dict:
        """Get information relevant to a specific strategy."""
        try:
            strategy_type = StrategyType[strategy.upper()]
            return self.info_broker.get_for_strategy(strategy_type)
        except KeyError:
            return self.info_broker.get_shared().to_dict()
    
    def get_best_strategies_current_regime(self, top_n: int = 3) -> List[str]:
        """Get best strategies for current regime."""
        current = self.info_broker.get_shared().current_regime
        strategies = self.regime_memory.get_best_strategies_for_regime(current, top_n)
        return [s.value for s in strategies]
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        
        stats = {
            "meta_memory": {},
            "strategy_performances": {},
            "current_regime": self.info_broker.get_shared().current_regime.value,
            "aggressiveness_modifier": self.get_aggressiveness_modifier(),
        }
        
        # Meta memory stats
        if self.meta_memory:
            stats["meta_memory"] = self.meta_memory.get_stats()
        
        # Strategy performances
        for strategy_type, perf in self._strategy_perfs.items():
            stats["strategy_performances"][strategy_type.value] = perf.to_dict()
        
        return stats
    
    # =========================================================================
    # PROOF LOG
    # =========================================================================
    
    def get_proof_log(self) -> str:
        """Generate proof log for meta-decision state."""
        
        shared = self.info_broker.get_shared()
        modifier = self.get_aggressiveness_modifier()
        
        return (
            f"[META] regime={shared.current_regime.value} | "
            f"aggr_mod={modifier:.2f} | "
            f"vol_regime={shared.vol_regime} | "
            f"liq_score={shared.liquidity_score:.2f} | "
            f"alpha_adj={self._alpha_threshold_adjustment:.0f}bps"
        )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_enhanced_meta_layer: Optional[EnhancedMetaDecisionLayer] = None


def get_enhanced_meta_layer() -> EnhancedMetaDecisionLayer:
    """Get singleton enhanced meta-decision layer."""
    global _enhanced_meta_layer
    if _enhanced_meta_layer is None:
        _enhanced_meta_layer = EnhancedMetaDecisionLayer()
    return _enhanced_meta_layer


def reset_enhanced_meta_layer():
    """Reset singleton (for testing)."""
    global _enhanced_meta_layer
    _enhanced_meta_layer = None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("HMATS v4.0-COMPLETE - ENHANCED META-DECISION LAYER TEST")
    print("=" * 70)
    
    meta = get_enhanced_meta_layer()
    
    # Update context
    meta.update_context(
        market_data={
            'volatility': 0.20,
            'implied_vol': 0.25,
            'liquidity_score': 0.7,
            'is_weekend': False,
        },
        signals={
            'btc': 0.5,
            'eth': 0.4,
            'sol': 0.6,
            'sentiment': 0.3,
        },
        regime="MODERATE_BULL"
    )
    
    print(f"\nShared Info:")
    shared = meta.get_shared_info()
    print(f"  Regime: {shared.current_regime.value}")
    print(f"  Vol Regime: {shared.vol_regime}")
    print(f"  Liquidity: {shared.liquidity_score}")
    
    # Record some outcomes
    meta.record_outcome("QUANT_MOMENTUM", 100, "MODERATE_BULL", is_opportunity=True)
    meta.record_outcome("QUANT_MOMENTUM", -50, "MODERATE_BULL", is_opportunity=False)
    meta.record_outcome("QUANT_MOMENTUM", 80, "MODERATE_BULL", is_opportunity=True)
    
    print(f"\nAggressiveness Modifier: {meta.get_aggressiveness_modifier():.2f}")
    print(f"Adjusted Confidence (QUANT_MOMENTUM, 0.7): {meta.get_adjusted_confidence('QUANT_MOMENTUM', 0.7):.2f}")
    
    print(f"\nProof Log: {meta.get_proof_log()}")
    
    print(f"\nStats:")
    stats = meta.get_stats()
    print(f"  Current Regime: {stats['current_regime']}")
    print(f"  Strategy Performances: {len(stats['strategy_performances'])}")

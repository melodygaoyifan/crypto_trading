"""
================================================================================
HMATS v3.4 - STRATEGY AGING MANAGEMENT
================================================================================
Version: 3.4.0
Type: META-LAYER (not an agent, does not generate signals)
Purpose: Track rolling effectiveness of Quant strategies and prevent drift

FUNCTION:
    - Track performance of each of the 12 Quant strategies
    - Calculate rolling effectiveness scores
    - Apply aging/decay factors on top of regime weights
    - Detect and flag strategy degradation
    - Prevent long-term strategy drift

CRITICAL CONSTRAINTS:
    - Does NOT generate trades or signals
    - Does NOT disable strategies (only adjusts weights)
    - Provides weight modifiers to Quant Agent
    - Decay is gradual and bounded

WHY THIS EXISTS:
    Market conditions evolve. A strategy that worked well 6 months ago may
    have degraded. Without aging management, dead strategies maintain their
    weights forever. This layer tracks effectiveness and gradually reduces
    weights for underperforming strategies while allowing recovery.

THE 12 STRATEGIES (from v3.2):
    Bear: mean_reversion, volatility_breakout, rsi_divergence
    Bull: trend_following, momentum, obv_confirmation
    Volatile: range_breakout, vwap_deviation, volume_profile
    Universal: orderbook_imbalance, funding_rate, liquidation_cascade
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# Strategy definitions from v3.2
STRATEGY_GROUPS = {
    "bear": ["mean_reversion", "volatility_breakout", "rsi_divergence"],
    "bull": ["trend_following", "momentum", "obv_confirmation"],
    "volatile": ["range_breakout", "vwap_deviation", "volume_profile"],
    # [P317] `regimebook` is what main.py:10294 writes into
    # `primary_strategy` whenever the regimebook seat is enforced
    # (P298 armed it). This vocabulary is the v3.2 set and never
    # learned the seat names, so record_signal took its unknown-name
    # branch and returned "" — the strategy actually DRIVING the book
    # was the one strategy this tracker recorded nothing about, three
    # times per tick. "universal" because a regime book is
    # regime-agnostic by construction: it picks its leg FROM the regime.
    #
    # ONLY `regimebook` is added. `trend` already arrives as
    # `trend_following`, and the whale seat does not overwrite
    # `primary_strategy` at all — a vocabulary entry no producer emits
    # is the inverse defect (P310), so nothing is added speculatively.
    "universal": ["orderbook_imbalance", "funding_rate",
                  "liquidation_cascade", "regimebook"],
}

ALL_STRATEGIES = [s for group in STRATEGY_GROUPS.values() for s in group]
STRATEGY_ALIASES = {
    "mean_revert": "mean_reversion",
    "volume_breakout": "volatility_breakout",
    "vrp": "momentum",
}


@dataclass
class StrategySignalRecord:
    """Record of a strategy signal and its outcome."""
    
    strategy_name: str
    timestamp: datetime
    
    # Signal
    direction: float  # -1 to 1
    confidence: float
    regime_at_signal: str
    
    # Outcome (filled after trade completes)
    outcome_pnl_bps: Optional[float] = None
    was_correct_direction: Optional[bool] = None
    bars_to_outcome: Optional[int] = None
    
    @property
    def has_outcome(self) -> bool:
        return self.outcome_pnl_bps is not None


@dataclass
class StrategyAgingConfig:
    """Configuration for strategy aging."""
    
    # Tracking window
    rolling_window_days: int = 60       # 60-day rolling window
    min_signals_for_assessment: int = 20  # Minimum signals to evaluate
    
    # Effectiveness calculation
    direction_accuracy_weight: float = 0.4
    pnl_contribution_weight: float = 0.4
    signal_quality_weight: float = 0.2
    
    # Decay/boost factors
    max_decay_factor: float = 0.5       # Maximum 50% reduction
    max_boost_factor: float = 1.2       # Maximum 20% boost
    decay_rate_per_10pct: float = 0.1   # 10% decay for each 10% underperformance
    
    # Thresholds
    underperformance_threshold: float = 0.3  # Below 30% effectiveness = decay
    strong_performance_threshold: float = 0.7  # Above 70% = boost
    
    # Recovery
    recovery_rate: float = 0.05  # 5% recovery per evaluation period
    
    # Evaluation frequency
    evaluation_period_hours: int = 24   # Re-evaluate daily


@dataclass
class StrategyEffectiveness:
    """Effectiveness assessment for a strategy."""
    
    strategy_name: str
    
    # Scores (0 to 1)
    direction_accuracy: float = 0.5
    pnl_contribution: float = 0.5
    signal_quality: float = 0.5
    overall_effectiveness: float = 0.5
    
    # Derived modifier
    weight_modifier: float = 1.0  # Applied to regime weights
    
    # Status
    is_degraded: bool = False
    is_strong: bool = False
    
    # Data quality
    signal_count: int = 0
    sufficient_data: bool = False
    
    # Metadata
    last_evaluated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy_name,
            "effectiveness": self.overall_effectiveness,
            "modifier": self.weight_modifier,
            "degraded": self.is_degraded,
            "strong": self.is_strong,
            "signals": self.signal_count,
        }


@dataclass
class StrategyAgingReport:
    """Complete aging report for all strategies."""
    
    # Individual assessments
    strategies: Dict[str, StrategyEffectiveness] = field(default_factory=dict)
    
    # Group summaries
    group_health: Dict[str, float] = field(default_factory=dict)
    
    # Alerts
    degraded_strategies: List[str] = field(default_factory=list)
    strong_strategies: List[str] = field(default_factory=list)
    
    # Overall
    system_health: float = 0.5
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "system_health": self.system_health,
            "group_health": self.group_health,
            "degraded": self.degraded_strategies,
            "strong": self.strong_strategies,
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
        }


class StrategyAgingManager:
    """
    Meta-layer for tracking and aging Quant strategies.
    
    USAGE:
        aging = StrategyAgingManager()
        
        # Record strategy signals
        aging.record_signal(strategy_name, direction, confidence, regime)
        
        # Record outcomes when trades complete
        aging.record_outcome(strategy_name, signal_timestamp, pnl_bps, was_correct)
        
        # Get weight modifiers for Quant Agent
        modifiers = aging.get_weight_modifiers()
        
        # Apply in Quant Agent
        effective_weight = base_weight * modifiers[strategy_name]
    
    CRITICAL: This does NOT disable strategies. It only adjusts weights.
    Even a degraded strategy keeps 50% of its base weight.
    """
    
    def __init__(self, config: Optional[StrategyAgingConfig] = None):
        self.config = config or StrategyAgingConfig()
        
        # Signal history by strategy
        self._signals: Dict[str, List[StrategySignalRecord]] = defaultdict(list)
        
        # Current effectiveness by strategy
        self._effectiveness: Dict[str, StrategyEffectiveness] = {}
        
        # Initialize all strategies
        for strategy in ALL_STRATEGIES:
            self._effectiveness[strategy] = StrategyEffectiveness(
                strategy_name=strategy,
                sufficient_data=False,
            )
        
        # Last evaluation time
        self._last_evaluation: Optional[datetime] = None
        
        logger.info(f"StrategyAgingManager initialized with {len(ALL_STRATEGIES)} strategies")

    @staticmethod
    def _normalize_strategy_name(strategy_name: str) -> str:
        """Map cross-module strategy aliases to canonical aging keys."""
        if not strategy_name:
            return ""
        normalized = strategy_name.strip().lower()
        return STRATEGY_ALIASES.get(normalized, normalized)
    
    def record_signal(
        self,
        strategy_name: str,
        direction: float,
        confidence: float,
        regime: str,
    ) -> str:
        """
        Record a strategy signal for tracking.
        
        Returns signal_id for later outcome recording.
        """

        raw_strategy_name = strategy_name
        strategy_name = self._normalize_strategy_name(strategy_name)

        if strategy_name not in ALL_STRATEGIES:
            # [P317] Latched per NAME. This fires once per asset per
            # tick, so an unrecognised decider produced ~18 identical
            # lines a day forever — noise that says nothing new after the
            # first, which is how a log stops being read (P202). Per NAME
            # and not globally, or the first unknown silences every later
            # one (the P193 latch bug).
            if not hasattr(self, "_unknown_warned"):
                self._unknown_warned = set()
            if raw_strategy_name not in self._unknown_warned:
                self._unknown_warned.add(raw_strategy_name)
                logger.warning(
                    "[STRATEGY_AGING] unknown strategy %r — NOTHING is "
                    "being recorded for it, so it can never appear in the "
                    "weekly degraded-strategy report. This vocabulary is "
                    "the v3.2 set (ALL_STRATEGIES); add the name there if "
                    "it is a real decider (P317). Once per name.",
                    raw_strategy_name)
            # [P317] This early exit is UNLOGGED on the repeat path BY
            # DESIGN: the warning above is latched per name, so the scanner
            # correctly counts a return with no log. Logging it every time is
            # exactly the condition this change removed (~18 identical lines
            # a day, P202). Counted in the swallow baseline with attribution
            # rather than suppressed — the noqa tag only applies to 
            # handlers, not to this finding kind.
            return ""
        
        record = StrategySignalRecord(
            strategy_name=strategy_name,
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            regime_at_signal=regime,
        )
        
        self._signals[strategy_name].append(record)
        
        # Trim old signals
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.rolling_window_days)
        self._signals[strategy_name] = [
            s for s in self._signals[strategy_name]
            if s.timestamp > cutoff
        ]
        
        return f"{strategy_name}_{record.timestamp.isoformat()}"
    
    def record_outcome(
        self,
        strategy_name: str,
        signal_timestamp: datetime,
        pnl_bps: float,
        was_correct_direction: bool,
        bars_to_outcome: int = 0,
    ):
        """
        Record outcome for a previous signal.
        
        Called when a trade influenced by this strategy completes.
        """

        strategy_name = self._normalize_strategy_name(strategy_name)
        if strategy_name not in self._signals:
            return
        
        # Find matching signal (within 1 hour tolerance)
        for record in self._signals[strategy_name]:
            if abs((record.timestamp - signal_timestamp).total_seconds()) < 3600:
                if not record.has_outcome:
                    record.outcome_pnl_bps = pnl_bps
                    record.was_correct_direction = was_correct_direction
                    record.bars_to_outcome = bars_to_outcome
                    break
        
        # Check if we should re-evaluate
        self._maybe_evaluate()
    
    def _maybe_evaluate(self):
        """Check if evaluation is due."""
        
        if self._last_evaluation is None:
            self._evaluate_all()
            return
        
        hours_since = (datetime.now(timezone.utc) - self._last_evaluation).total_seconds() / 3600
        
        if hours_since >= self.config.evaluation_period_hours:
            self._evaluate_all()
    
    def _evaluate_all(self):
        """Evaluate effectiveness of all strategies."""
        
        for strategy_name in ALL_STRATEGIES:
            self._evaluate_strategy(strategy_name)
        
        self._last_evaluation = datetime.now(timezone.utc)
        
        # Log summary
        degraded = [s for s, e in self._effectiveness.items() if e.is_degraded]
        strong = [s for s, e in self._effectiveness.items() if e.is_strong]
        
        if degraded:
            logger.warning(f"StrategyAging: Degraded strategies: {degraded}")
        if strong:
            logger.info(f"StrategyAging: Strong strategies: {strong}")
    
    def _evaluate_strategy(self, strategy_name: str):
        """Evaluate a single strategy."""
        
        signals = self._signals.get(strategy_name, [])
        completed = [s for s in signals if s.has_outcome]
        
        effectiveness = StrategyEffectiveness(
            strategy_name=strategy_name,
            signal_count=len(completed),
        )
        
        # Check minimum data requirement
        if len(completed) < self.config.min_signals_for_assessment:
            effectiveness.sufficient_data = False
            effectiveness.weight_modifier = 1.0  # No change without data
            self._effectiveness[strategy_name] = effectiveness
            return
        
        effectiveness.sufficient_data = True
        
        # Calculate direction accuracy
        correct = sum(1 for s in completed if s.was_correct_direction)
        effectiveness.direction_accuracy = correct / len(completed)
        
        # Calculate PnL contribution (normalized)
        total_pnl = sum(s.outcome_pnl_bps for s in completed)
        avg_pnl = total_pnl / len(completed)
        # Normalize: -100bps = 0.0, 0bps = 0.5, +100bps = 1.0
        effectiveness.pnl_contribution = max(0, min(1, (avg_pnl + 100) / 200))
        
        # Calculate signal quality (confidence vs outcome correlation)
        high_conf_signals = [s for s in completed if s.confidence > 0.6]
        if high_conf_signals:
            high_conf_correct = sum(1 for s in high_conf_signals if s.was_correct_direction)
            effectiveness.signal_quality = high_conf_correct / len(high_conf_signals)
        else:
            effectiveness.signal_quality = 0.5
        
        # Calculate overall effectiveness
        effectiveness.overall_effectiveness = (
            self.config.direction_accuracy_weight * effectiveness.direction_accuracy +
            self.config.pnl_contribution_weight * effectiveness.pnl_contribution +
            self.config.signal_quality_weight * effectiveness.signal_quality
        )
        
        # Determine weight modifier
        current_modifier = self._effectiveness.get(strategy_name, StrategyEffectiveness(strategy_name=strategy_name)).weight_modifier
        
        if effectiveness.overall_effectiveness < self.config.underperformance_threshold:
            # Degraded - apply decay
            effectiveness.is_degraded = True
            underperformance = self.config.underperformance_threshold - effectiveness.overall_effectiveness
            decay = underperformance / 0.1 * self.config.decay_rate_per_10pct
            decay = min(decay, 1 - self.config.max_decay_factor)
            
            new_modifier = max(
                self.config.max_decay_factor,
                current_modifier - decay
            )
            effectiveness.weight_modifier = new_modifier
            
        elif effectiveness.overall_effectiveness > self.config.strong_performance_threshold:
            # Strong - apply boost
            effectiveness.is_strong = True
            overperformance = effectiveness.overall_effectiveness - self.config.strong_performance_threshold
            boost = overperformance / 0.1 * 0.05  # 5% boost per 10% over threshold
            
            new_modifier = min(
                self.config.max_boost_factor,
                current_modifier + boost
            )
            effectiveness.weight_modifier = new_modifier
            
        else:
            # Normal range - gradual recovery toward 1.0
            if current_modifier < 1.0:
                effectiveness.weight_modifier = min(1.0, current_modifier + self.config.recovery_rate)
            else:
                effectiveness.weight_modifier = max(1.0, current_modifier - self.config.recovery_rate)
        
        self._effectiveness[strategy_name] = effectiveness
    
    def get_weight_modifiers(self) -> Dict[str, float]:
        """
        Get weight modifiers for all strategies.
        
        WHERE CONSUMED — [P317] CORRECTED; this was a claim, not a fact:
            The only caller outside this module is
            `core/execution_service.py:~3292`, a WEEKLY (168h) block that LOGS
            a CRITICAL when a modifier falls below 0.7 and an INFO above 1.1.
            Nothing multiplies a weight by it. The old text ("Quant Agent:
            Applies to strategy weights / effective_weight =
            base_regime_weight * aging_modifier") described an intended wiring
            that does not exist in the tree — the P177 shape, where a
            docstring asserts a connection and readers reason from it.
            Re-verify with `grep -rn "get_weight_modifiers" --include=*.py .`
            before trusting any statement about what consumes this.

        CRITICAL: Modifier is bounded [0.5, 1.2].
        Strategies are NEVER disabled, only reduced.
        """
        
        self._maybe_evaluate()
        
        return {
            name: eff.weight_modifier
            for name, eff in self._effectiveness.items()
        }
    
    def get_strategy_effectiveness(self, strategy_name: str) -> StrategyEffectiveness:
        """Get effectiveness for a specific strategy."""
        strategy_name = self._normalize_strategy_name(strategy_name)
        return self._effectiveness.get(
            strategy_name,
            StrategyEffectiveness(strategy_name=strategy_name)
        )
    
    def get_report(self) -> StrategyAgingReport:
        """Get full aging report."""
        
        self._maybe_evaluate()
        
        report = StrategyAgingReport(
            strategies=dict(self._effectiveness),
        )
        
        # Group health
        for group_name, strategies in STRATEGY_GROUPS.items():
            group_eff = [
                self._effectiveness[s].overall_effectiveness
                for s in strategies
                if self._effectiveness[s].sufficient_data
            ]
            if group_eff:
                report.group_health[group_name] = sum(group_eff) / len(group_eff)
            else:
                report.group_health[group_name] = 0.5
        
        # Degraded/Strong lists
        report.degraded_strategies = [
            s for s, e in self._effectiveness.items() if e.is_degraded
        ]
        report.strong_strategies = [
            s for s, e in self._effectiveness.items() if e.is_strong
        ]
        
        # System health (average of groups)
        if report.group_health:
            report.system_health = sum(report.group_health.values()) / len(report.group_health)
        
        return report
    
    def force_reset_strategy(self, strategy_name: str):
        """Reset a specific strategy to neutral."""
        strategy_name = self._normalize_strategy_name(strategy_name)
        if strategy_name in self._effectiveness:
            self._effectiveness[strategy_name] = StrategyEffectiveness(
                strategy_name=strategy_name,
                weight_modifier=1.0,
            )
            logger.info(f"StrategyAging: Force reset {strategy_name}")
    
    def reset(self):
        """Full reset."""
        self._signals.clear()
        for strategy in ALL_STRATEGIES:
            self._effectiveness[strategy] = StrategyEffectiveness(
                strategy_name=strategy,
            )
        self._last_evaluation = None


# Singleton
_aging_manager: Optional[StrategyAgingManager] = None

def get_aging_manager() -> StrategyAgingManager:
    global _aging_manager
    if _aging_manager is None:
        _aging_manager = StrategyAgingManager()
    return _aging_manager

def reset_aging_manager():
    global _aging_manager
    _aging_manager = None

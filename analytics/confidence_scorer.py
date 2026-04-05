"""
================================================================================
HMATS v3.5 - STRATEGY CONFIDENCE SCORING
================================================================================
Version: 3.5.0
Upgrade From: v3.4 Strategy Aging
Purpose: Convert aging decay to confidence scoring for Quant layer

CHANGES FROM v3.4:
    v3.4: Binary aging (degraded/normal/strong) with weight modifiers
    v3.5: Continuous confidence scoring with realized vs expected PnL tracking

NEW FEATURES:
    1. Track realized PnL vs expected PnL per strategy
    2. Track hit rate in correct regime (regime-aware accuracy)
    3. Confidence score ∈ [0,1] as continuous multiplier
    4. Confidence as multiplier, NOT hard gate (no strategy disabled)

WHY THIS AVOIDS OVERFITTING:
    - Rolling window (60 days) prevents recent bias
    - Regime-aware tracking prevents blaming strategy for wrong regime
    - Minimum sample size before confidence adjustment
    - Smooth decay, not cliff edges

WHY THIS PREVENTS SILENT STRATEGY DECAY:
    - Tracks expected vs realized PnL (not just wins)
    - A strategy that "wins" but underperforms expectation is flagged
    - Continuous monitoring, not periodic checks
================================================================================
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# Strategy definitions from v3.4
STRATEGY_GROUPS = {
    "bear": ["mean_reversion", "volatility_breakout", "rsi_divergence"],
    "bull": ["trend_following", "momentum", "obv_confirmation"],
    "volatile": ["range_breakout", "vwap_deviation", "volume_profile"],
    "universal": ["orderbook_imbalance", "funding_rate", "liquidation_cascade"],
}

ALL_STRATEGIES = [s for group in STRATEGY_GROUPS.values() for s in group]

# Regime to strategy group mapping
REGIME_STRATEGY_MAP = {
    "BULL_TREND": "bull",
    "BEAR_TREND": "bear",
    "VOLATILE": "volatile",
    "SIDEWAYS": "volatile",
}


@dataclass
class StrategySignalRecord:
    """Record of a strategy signal with outcome."""
    
    strategy_name: str
    timestamp: datetime
    
    # Signal
    direction: float  # -1 to 1
    confidence: float  # Expected edge
    expected_pnl_bps: float  # What we expected to make
    
    # Context
    regime_at_signal: str
    was_correct_regime: bool  # Was this strategy in its favored regime?
    
    # Outcome
    realized_pnl_bps: Optional[float] = None
    direction_correct: Optional[bool] = None
    
    @property
    def has_outcome(self) -> bool:
        return self.realized_pnl_bps is not None
    
    @property
    def pnl_vs_expected(self) -> Optional[float]:
        """Realized PnL vs expected (positive = outperformed)."""
        if self.realized_pnl_bps is None:
            return None
        return self.realized_pnl_bps - self.expected_pnl_bps


@dataclass
class StrategyConfidenceConfig:
    """Configuration for strategy confidence scoring."""
    
    # Window
    rolling_window_days: int = 60
    min_signals_for_confidence: int = 15
    
    # Confidence calculation weights
    direction_accuracy_weight: float = 0.35
    pnl_vs_expected_weight: float = 0.35
    regime_accuracy_weight: float = 0.30
    
    # Confidence bounds
    min_confidence: float = 0.3   # Never below 30%
    max_confidence: float = 1.0   # Normal = 100%
    neutral_confidence: float = 0.7  # Default when insufficient data
    
    # Decay/boost rates
    decay_sensitivity: float = 0.1  # How much underperformance affects confidence
    boost_sensitivity: float = 0.05  # How much overperformance affects confidence
    
    # Smoothing
    confidence_smoothing: float = 0.3  # EMA smoothing factor


@dataclass 
class StrategyConfidence:
    """Confidence assessment for a single strategy."""
    
    strategy_name: str
    
    # Core confidence score
    confidence: float = 0.7  # ∈ [0, 1]
    
    # Component scores
    direction_accuracy: float = 0.5
    pnl_vs_expected_score: float = 0.5
    regime_accuracy: float = 0.5
    
    # Statistics
    total_signals: int = 0
    signals_in_correct_regime: int = 0
    avg_realized_pnl: float = 0.0
    avg_expected_pnl: float = 0.0
    
    # Status
    has_sufficient_data: bool = False
    is_underperforming: bool = False
    is_outperforming: bool = False
    
    # Trend
    confidence_trend: str = "STABLE"  # IMPROVING, STABLE, DECLINING
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy_name,
            "confidence": self.confidence,
            "direction_accuracy": self.direction_accuracy,
            "pnl_vs_expected": self.pnl_vs_expected_score,
            "regime_accuracy": self.regime_accuracy,
            "sufficient_data": self.has_sufficient_data,
            "underperforming": self.is_underperforming,
            "outperforming": self.is_outperforming,
            "trend": self.confidence_trend,
        }


class StrategyConfidenceScorer:
    """
    Strategy confidence scoring system for v3.5.
    
    UPGRADES FROM v3.4 Strategy Aging:
        1. Continuous confidence ∈ [0,1] instead of discrete states
        2. Tracks realized vs expected PnL
        3. Regime-aware accuracy (was strategy in correct regime?)
        4. Confidence as multiplier, not gate
    
    USAGE:
        scorer = StrategyConfidenceScorer()
        
        # Record strategy signal
        scorer.record_signal(
            strategy_name="momentum",
            direction=0.8,
            confidence=0.6,
            expected_pnl_bps=50,
            regime="BULL_TREND",
        )
        
        # Record outcome
        scorer.record_outcome(
            strategy_name="momentum",
            signal_timestamp=...,
            realized_pnl_bps=75,
            direction_correct=True,
        )
        
        # Get confidence for Quant layer
        confidence = scorer.get_strategy_confidence("momentum")
        effective_weight = base_weight * confidence.confidence
    """
    
    def __init__(self, config: Optional[StrategyConfidenceConfig] = None):
        self.config = config or StrategyConfidenceConfig()
        
        # Signal records by strategy
        self._signals: Dict[str, List[StrategySignalRecord]] = defaultdict(list)
        
        # Current confidence by strategy
        self._confidence: Dict[str, StrategyConfidence] = {}
        
        # Previous confidence for trend detection
        self._prev_confidence: Dict[str, float] = {}
        
        # Initialize all strategies
        for strategy in ALL_STRATEGIES:
            self._confidence[strategy] = StrategyConfidence(
                strategy_name=strategy,
                confidence=self.config.neutral_confidence,
            )
        
        logger.info(f"StrategyConfidenceScorer v3.5 initialized with {len(ALL_STRATEGIES)} strategies")
    
    def record_signal(
        self,
        strategy_name: str,
        direction: float,
        confidence: float,
        expected_pnl_bps: float,
        regime: str,
    ) -> Optional[datetime]:
        """
        Record a strategy signal.
        
        Returns timestamp for later outcome recording.
        """
        
        if strategy_name not in ALL_STRATEGIES:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return None
        
        # Determine if strategy is in its correct regime
        strategy_group = None
        for group, strategies in STRATEGY_GROUPS.items():
            if strategy_name in strategies:
                strategy_group = group
                break
        
        correct_regime = REGIME_STRATEGY_MAP.get(regime) == strategy_group
        
        # Universal strategies are always "correct"
        if strategy_group == "universal":
            correct_regime = True
        
        timestamp = datetime.now(timezone.utc)
        
        record = StrategySignalRecord(
            strategy_name=strategy_name,
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            expected_pnl_bps=expected_pnl_bps,
            regime_at_signal=regime,
            was_correct_regime=correct_regime,
        )
        
        self._signals[strategy_name].append(record)
        
        # Trim old records
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.rolling_window_days)
        self._signals[strategy_name] = [
            s for s in self._signals[strategy_name]
            if s.timestamp > cutoff
        ]
        
        return timestamp
    
    def record_outcome(
        self,
        strategy_name: str,
        signal_timestamp: datetime,
        realized_pnl_bps: float,
        direction_correct: bool,
    ):
        """Record outcome for a previous signal."""
        
        if strategy_name not in self._signals:
            return
        
        # Find matching signal (within 1 hour tolerance)
        for record in self._signals[strategy_name]:
            if abs((record.timestamp - signal_timestamp).total_seconds()) < 3600:
                if not record.has_outcome:
                    record.realized_pnl_bps = realized_pnl_bps
                    record.direction_correct = direction_correct
                    
                    # Recalculate confidence
                    self._update_confidence(strategy_name)
                    break
    
    def _update_confidence(self, strategy_name: str):
        """Recalculate confidence for a strategy."""
        
        signals = self._signals.get(strategy_name, [])
        completed = [s for s in signals if s.has_outcome]
        
        conf = StrategyConfidence(strategy_name=strategy_name)
        conf.total_signals = len(completed)
        
        # Check minimum data
        if len(completed) < self.config.min_signals_for_confidence:
            conf.has_sufficient_data = False
            conf.confidence = self.config.neutral_confidence
            self._confidence[strategy_name] = conf
            return
        
        conf.has_sufficient_data = True
        
        # 1. Direction accuracy
        correct_dir = sum(1 for s in completed if s.direction_correct)
        conf.direction_accuracy = correct_dir / len(completed)
        
        # 2. PnL vs Expected score
        pnl_diffs = [s.pnl_vs_expected for s in completed if s.pnl_vs_expected is not None]
        if pnl_diffs:
            avg_diff = statistics.mean(pnl_diffs)
            # Normalize: -100bps diff = 0.0, 0 = 0.5, +100bps = 1.0
            conf.pnl_vs_expected_score = max(0, min(1, (avg_diff + 100) / 200))
            
            conf.avg_realized_pnl = statistics.mean([s.realized_pnl_bps for s in completed])
            conf.avg_expected_pnl = statistics.mean([s.expected_pnl_bps for s in completed])
        
        # 3. Regime accuracy (how well does it perform in correct regime?)
        correct_regime_signals = [s for s in completed if s.was_correct_regime]
        if correct_regime_signals:
            conf.signals_in_correct_regime = len(correct_regime_signals)
            regime_wins = sum(1 for s in correct_regime_signals if s.direction_correct)
            conf.regime_accuracy = regime_wins / len(correct_regime_signals)
        else:
            conf.regime_accuracy = 0.5
        
        # Calculate composite confidence
        raw_confidence = (
            self.config.direction_accuracy_weight * conf.direction_accuracy +
            self.config.pnl_vs_expected_weight * conf.pnl_vs_expected_score +
            self.config.regime_accuracy_weight * conf.regime_accuracy
        )
        
        # Apply smoothing with previous confidence
        prev_conf = self._confidence.get(strategy_name)
        if prev_conf and prev_conf.has_sufficient_data:
            smoothed = (
                self.config.confidence_smoothing * raw_confidence +
                (1 - self.config.confidence_smoothing) * prev_conf.confidence
            )
        else:
            smoothed = raw_confidence
        
        # Bound confidence
        conf.confidence = max(
            self.config.min_confidence,
            min(self.config.max_confidence, smoothed)
        )
        
        # Determine status
        conf.is_underperforming = conf.confidence < 0.5
        conf.is_outperforming = conf.confidence > 0.8
        
        # Determine trend
        prev_val = self._prev_confidence.get(strategy_name, conf.confidence)
        diff = conf.confidence - prev_val
        if diff > 0.05:
            conf.confidence_trend = "IMPROVING"
        elif diff < -0.05:
            conf.confidence_trend = "DECLINING"
        else:
            conf.confidence_trend = "STABLE"
        
        self._prev_confidence[strategy_name] = conf.confidence
        self._confidence[strategy_name] = conf
    
    def get_strategy_confidence(self, strategy_name: str) -> StrategyConfidence:
        """
        Get confidence for a specific strategy.
        
        WHERE CONSUMED:
            - Quant Agent: Multiplies base weight by confidence
        
        HOW APPLIED:
            effective_weight = base_regime_weight * confidence.confidence
        """
        return self._confidence.get(
            strategy_name,
            StrategyConfidence(strategy_name=strategy_name, confidence=self.config.neutral_confidence)
        )
    
    def get_all_confidences(self) -> Dict[str, float]:
        """Get confidence values for all strategies."""
        return {
            name: conf.confidence
            for name, conf in self._confidence.items()
        }
    
    def get_confidence_report(self) -> Dict:
        """Get comprehensive confidence report."""
        
        report = {
            "strategies": {},
            "groups": {},
            "underperforming": [],
            "outperforming": [],
            "declining": [],
        }
        
        for name, conf in self._confidence.items():
            report["strategies"][name] = conf.to_dict()
            
            if conf.is_underperforming:
                report["underperforming"].append(name)
            if conf.is_outperforming:
                report["outperforming"].append(name)
            if conf.confidence_trend == "DECLINING":
                report["declining"].append(name)
        
        # Group averages
        for group_name, strategies in STRATEGY_GROUPS.items():
            group_confs = [self._confidence[s].confidence for s in strategies]
            report["groups"][group_name] = statistics.mean(group_confs)
        
        return report
    
    def get_alpha_leak_signals(self) -> List[str]:
        """
        Identify strategies that may be leaking alpha.
        
        A strategy leaks alpha if:
        - Direction accuracy is good (>60%)
        - But PnL vs expected is poor (<40%)
        
        This suggests the signal is right but execution/exit is losing value.
        """
        
        leaking = []
        
        for name, conf in self._confidence.items():
            if not conf.has_sufficient_data:
                continue
            
            # Good direction but poor PnL capture
            if conf.direction_accuracy > 0.6 and conf.pnl_vs_expected_score < 0.4:
                leaking.append(name)
        
        return leaking
    
    def reset(self):
        """Full reset."""
        self._signals.clear()
        self._prev_confidence.clear()
        for strategy in ALL_STRATEGIES:
            self._confidence[strategy] = StrategyConfidence(
                strategy_name=strategy,
                confidence=self.config.neutral_confidence,
            )

    def to_dict(self) -> Dict:
        """Serialize confidence state for restart persistence."""
        return {
            "confidence": {
                name: conf.to_dict() for name, conf in self._confidence.items()
            },
            "prev_confidence": dict(self._prev_confidence),
        }

    def from_dict(self, data: Dict):
        """Restore confidence state from persistence."""
        conf_data = data.get("confidence", {})
        for name, cd in conf_data.items():
            if name in self._confidence:
                sc = self._confidence[name]
                sc.confidence = cd.get("confidence", sc.confidence)
                sc.direction_accuracy = cd.get("direction_accuracy", sc.direction_accuracy)
                sc.pnl_vs_expected_score = cd.get("pnl_vs_expected", sc.pnl_vs_expected_score)
                sc.regime_accuracy = cd.get("regime_accuracy", sc.regime_accuracy)
                sc.has_sufficient_data = cd.get("sufficient_data", False)
                sc.is_underperforming = cd.get("underperforming", False)
                sc.is_outperforming = cd.get("outperforming", False)
                sc.confidence_trend = cd.get("trend", "STABLE")
        self._prev_confidence = data.get("prev_confidence", {})
        logger.info(
            f"[CONFIDENCE] State restored: {len(conf_data)} strategies"
        )


# Singleton
_confidence_scorer: Optional[StrategyConfidenceScorer] = None

def get_confidence_scorer() -> StrategyConfidenceScorer:
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = StrategyConfidenceScorer()
    return _confidence_scorer

def reset_confidence_scorer():
    global _confidence_scorer
    _confidence_scorer = None

"""
================================================================================
HMATS v6.4.1 - Regime Transition Risk
================================================================================
Purpose: Track regime transition probabilities and provide stability metrics

This module maintains a rolling transition matrix that estimates:
- P(stay in current regime) = stability
- P(transition to different regime) = transition_risk

Used to:
- Tighten chasing/add-on conditions when regime is unstable
- Adjust execution aggressiveness
- NOT a global trade stop (profit-max philosophy)

DESIGN:
- Online Bayesian-ish update with exponential decay
- Safe defaults when insufficient data
- No global veto - only modulation

Author: HMATS P1 Upgrade
Version: 6.4.1
================================================================================
"""

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, Optional, Any, List
from collections import defaultdict
import time

logger = logging.getLogger("RegimeTransitionRisk")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RegimeTransitionConfig:
    """Configuration for regime transition tracking."""
    
    # Master switch
    enabled: bool = True
    
    # Update parameters
    decay_factor: float = 0.95  # Exponential decay for old observations
    min_observations_for_valid: int = 10  # Minimum obs before trusted output
    
    # Safe defaults
    default_stability: float = 0.5
    default_transition_risk: float = 0.5
    
    # Known regime names (for matrix initialization)
    known_regimes: List[str] = field(default_factory=lambda: [
        "STRONG_TREND", "TREND", "NEUTRAL", "VOLATILE_CHOP",
        "MEAN_REVERT", "TRANSITION", "UNKNOWN"
    ])
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegimeTransitionConfig':
        """Load from config dict with safe defaults."""
        rt = data.get("regime_transition", {})
        defaults = cls()
        kwargs = {
            f.name: rt.get(f.name, getattr(defaults, f.name))
            for f in dataclass_fields(cls)
        }
        return cls(**kwargs)


# =============================================================================
# TRACKING RESULT
# =============================================================================

@dataclass
class TransitionRiskResult:
    """Result of transition risk calculation."""
    
    stability: float  # 0-1, P(stay in current regime)
    transition_risk: float  # 0-1, 1 - stability
    current_regime: str
    
    # Additional info
    valid: bool  # True if enough data
    observations: int
    most_likely_next: str  # Most probable next regime
    transition_probs: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "stability": self.stability,
            "transition_risk": self.transition_risk,
            "current_regime": self.current_regime,
            "valid": self.valid,
            "observations": self.observations,
            "most_likely_next": self.most_likely_next,
            "transition_probs": self.transition_probs,
        }


# =============================================================================
# TRANSITION TRACKER IMPLEMENTATION
# =============================================================================

class RegimeTransitionRisk:
    """
    Tracks regime transition probabilities using a rolling online estimate.
    
    Uses exponential decay to weight recent transitions more heavily.
    Provides stability/transition_risk metrics for downstream modulation.
    
    PROFIT-MAX PHILOSOPHY:
    - This does NOT veto trades
    - It provides modulation signals for sizing/urgency
    - Low stability -> tighter add-on rules, more passive execution
    """
    
    def __init__(self, config: RegimeTransitionConfig = None):
        """
        Initialize tracker.
        
        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or RegimeTransitionConfig()
        
        # Transition count matrix (from_regime -> to_regime -> count)
        # Using exponentially decayed counts
        self._transition_counts: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # Initialize known regimes
        for regime in self.config.known_regimes:
            self._transition_counts[regime]  # Create entry
        
        # Total observations per regime
        self._total_from: Dict[str, float] = defaultdict(float)
        
        # Raw observation count
        self._observation_count = 0
        
        # Last seen regime
        self._last_regime: Optional[str] = None
        self._last_update_ts: float = 0.0
        
        logger.info(
            f"[REGIME_TRANSITION] Initialized: enabled={self.config.enabled}, "
            f"decay_factor={self.config.decay_factor}"
        )
    
    def update(self, current_regime: str) -> TransitionRiskResult:
        """
        Update with new regime observation and return current risk metrics.
        
        Args:
            current_regime: Currently detected regime
            
        Returns:
            TransitionRiskResult with stability and transition_risk
        """
        if not self.config.enabled:
            return self._default_result(current_regime)
        
        current_regime = current_regime.upper()
        
        # Apply decay to all counts
        self._apply_decay()
        
        # Record transition if we have a previous regime
        if self._last_regime is not None:
            self._transition_counts[self._last_regime][current_regime] += 1.0
            self._total_from[self._last_regime] += 1.0
            self._observation_count += 1
        
        # Update state
        self._last_regime = current_regime
        self._last_update_ts = time.time()
        
        # Calculate and return current metrics
        return self.get_risk(current_regime)
    
    def get_risk(self, current_regime: str) -> TransitionRiskResult:
        """
        Get transition risk metrics for a regime without updating.
        
        Args:
            current_regime: Regime to check
            
        Returns:
            TransitionRiskResult
        """
        if not self.config.enabled:
            return self._default_result(current_regime)
        
        current_regime = current_regime.upper()
        
        # Check if we have enough data
        valid = self._observation_count >= self.config.min_observations_for_valid
        
        if not valid:
            return self._default_result(current_regime)
        
        # Get transition counts from current regime
        total = self._total_from.get(current_regime, 0.0)
        
        if total < 1.0:
            return self._default_result(current_regime)
        
        # Calculate transition probabilities
        trans_probs = {}
        for to_regime, count in self._transition_counts[current_regime].items():
            trans_probs[to_regime] = count / total
        
        # Stability = P(stay in current regime)
        stability = trans_probs.get(current_regime, 0.0)
        transition_risk = 1.0 - stability
        
        # Find most likely next regime (excluding staying)
        most_likely_next = current_regime
        max_prob = stability
        for to_regime, prob in trans_probs.items():
            if to_regime != current_regime and prob > max_prob * 0.5:
                if prob > trans_probs.get(most_likely_next, 0):
                    most_likely_next = to_regime
        
        return TransitionRiskResult(
            stability=stability,
            transition_risk=transition_risk,
            current_regime=current_regime,
            valid=True,
            observations=self._observation_count,
            most_likely_next=most_likely_next,
            transition_probs=trans_probs,
        )
    
    def _apply_decay(self):
        """Apply exponential decay to all counts."""
        decay = self.config.decay_factor
        
        for from_regime in list(self._transition_counts.keys()):
            for to_regime in list(self._transition_counts[from_regime].keys()):
                self._transition_counts[from_regime][to_regime] *= decay
            
            if from_regime in self._total_from:
                self._total_from[from_regime] *= decay
    
    def _default_result(self, regime: str) -> TransitionRiskResult:
        """Return safe default result."""
        return TransitionRiskResult(
            stability=self.config.default_stability,
            transition_risk=self.config.default_transition_risk,
            current_regime=regime.upper(),
            valid=False,
            observations=self._observation_count,
            most_likely_next=regime.upper(),
            transition_probs={regime.upper(): self.config.default_stability},
        )
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            "observation_count": self._observation_count,
            "tracked_regimes": len(self._transition_counts),
            "last_regime": self._last_regime,
        }
    
    def reset(self):
        """Reset all state."""
        self._transition_counts = defaultdict(lambda: defaultdict(float))
        self._total_from = defaultdict(float)
        self._observation_count = 0
        self._last_regime = None
        self._last_update_ts = 0.0
        
        # Re-initialize known regimes
        for regime in self.config.known_regimes:
            self._transition_counts[regime]
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get the full transition probability matrix.
        
        Returns:
            Dict[from_regime -> Dict[to_regime -> probability]]
        """
        matrix = {}
        
        for from_regime in self._transition_counts:
            total = self._total_from.get(from_regime, 0.0)
            if total > 0:
                matrix[from_regime] = {}
                for to_regime, count in self._transition_counts[from_regime].items():
                    if count > 0.01:  # Filter noise
                        matrix[from_regime][to_regime] = count / total
        
        return matrix


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_tracker_instance: Optional[RegimeTransitionRisk] = None


def get_regime_transition_risk(config: RegimeTransitionConfig = None) -> RegimeTransitionRisk:
    """Get or create the singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = RegimeTransitionRisk(config)
    elif config is not None and _tracker_instance.config != config:
        logger.info("RegimeTransitionRisk config changed; refreshing singleton instance")
        _tracker_instance = RegimeTransitionRisk(config)
    return _tracker_instance


def reset_regime_transition_risk():
    """Reset the singleton instance."""
    global _tracker_instance
    _tracker_instance = None

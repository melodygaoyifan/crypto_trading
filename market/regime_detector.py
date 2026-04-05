"""
HMATS v3.2 - Enhanced Market Regime
====================================
Purpose: Centralized market state with explicit NO_TRADE and CRACK window.

Philosophy:
- Master decision frequency = 4H
- NO_TRADE is a HARD state (not just reduced exposure)
- CRACK window allows aggressive early entries during structural breaks
- Hysteresis for stability, but CRACK can bypass some requirements

Regime States:
- NO_TRADE: Hard FLAT, do not trade
- STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR: Directional bias
- VOLATILE_CHOP: Reduced confidence, smaller positions
- CRACK_OPPORTUNITY: Structural break window, aggressive entries allowed
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timezone, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MarketRegimeState(Enum):
    """
    Core market regime states.
    
    NO_TRADE is the FIRST state - it takes absolute priority.
    """
    NO_TRADE = 0        # HARD FLAT - do not trade
    STRONG_BEAR = 1
    BEAR = 2
    NEUTRAL = 3
    VOLATILE_CHOP = 4
    BULL = 5
    STRONG_BULL = 6


class OpportunityMode(Enum):
    """
    Opportunity mode - overlays on top of base regime.
    
    CRACK = Structural break window, aggressive allowed
    SENTIMENT_TRIGGER = Extreme sentiment, opportunity window
    FLOW_TRIGGER = Strong flow signal, opportunity window
    """
    NONE = auto()
    CRACK = auto()              # Structural break / cascade
    SENTIMENT_TRIGGER = auto()  # Extreme sentiment spike
    FLOW_TRIGGER = auto()       # Strong flow divergence


@dataclass
class RegimeConfidence:
    """Confidence metrics for current regime."""
    probability: float = 0.5      # GMM/HMM probability
    stability: float = 0.5        # How stable is this regime
    consistency: float = 0.5      # Cross-timeframe consistency
    time_in_regime_hours: float = 0.0
    
    def overall(self) -> float:
        """Weighted overall confidence."""
        return (
            0.4 * self.probability +
            0.3 * self.stability +
            0.3 * self.consistency
        )


@dataclass
class CrackWindow:
    """
    CRACK window state - structural break opportunity.
    
    When a structural break is detected, we enter a CRACK window
    that allows aggressive entries without full persistence requirements.
    """
    is_active: bool = False
    direction: str = "none"  # "long", "short", "none"
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    trigger_reason: str = ""
    
    # Break characteristics
    break_magnitude: float = 0.0  # How significant is the break
    volume_confirmation: bool = False
    structure_level_broken: float = 0.0  # Price level that was broken
    
    # Decay
    remaining_strength: float = 1.0  # Decays over time
    
    def is_valid(self) -> bool:
        """Check if CRACK window is still valid."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return self.remaining_strength > 0.2
    
    def update_decay(self, decay_per_4h: float = 0.3):
        """Update decay based on time passed."""
        if not self.is_active or not self.triggered_at:
            return
        
        hours_passed = (datetime.now(timezone.utc) - self.triggered_at).total_seconds() / 3600
        periods_passed = hours_passed / 4  # 4H periods
        self.remaining_strength = max(0, 1.0 - (periods_passed * decay_per_4h))
        
        if self.remaining_strength <= 0.2:
            self.is_active = False
            logger.info(f"CRACK window expired: {self.trigger_reason}")


@dataclass
class OpportunityState:
    """
    Opportunity mode state - aggressive trading windows.
    
    This is NOT a multiplier - it's a MODE SWITCH.
    When active, authority rules change to favor aggressive entries.
    """
    mode: OpportunityMode = OpportunityMode.NONE
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    trigger_reason: str = ""
    
    # Direction hint (if any)
    directional_bias: str = "none"  # "long", "short", "none"
    bias_strength: float = 0.0
    
    # Decay
    decay_window_hours: float = 4.0  # Default 4H decay
    remaining_strength: float = 1.0
    
    def is_active(self) -> bool:
        if self.mode == OpportunityMode.NONE:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return self.remaining_strength > 0.2
    
    def update_decay(self):
        """Update decay based on time passed."""
        if not self.triggered_at or self.mode == OpportunityMode.NONE:
            return
        
        hours_passed = (datetime.now(timezone.utc) - self.triggered_at).total_seconds() / 3600
        self.remaining_strength = max(0, 1.0 - (hours_passed / self.decay_window_hours))
        
        if self.remaining_strength <= 0.2:
            self.mode = OpportunityMode.NONE
            logger.info(f"Opportunity mode expired: {self.trigger_reason}")


@dataclass
class EnhancedRegimeState:
    """
    Complete regime state - the master state object.
    
    Updated on 4H ticks only (master frequency).
    """
    # Core regime
    regime: MarketRegimeState = MarketRegimeState.NEUTRAL
    confidence: RegimeConfidence = field(default_factory=RegimeConfidence)
    
    # Opportunity overlays
    opportunity: OpportunityState = field(default_factory=OpportunityState)
    crack_window: CrackWindow = field(default_factory=CrackWindow)
    
    # Directional metrics
    trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    trend_strength: float = 0.0       # 0-1
    
    # Cross-asset
    beta_sol_btc: float = 1.0
    beta_eth_btc: float = 1.0
    correlation_btc_eth_sol: float = 0.8
    
    # Volatility
    volatility_regime: str = "normal"  # "low", "normal", "high", "extreme"
    vol_of_vol: float = 0.0
    dvol_zscore: float = 0.0
    
    # Liquidity
    liquidity_regime: str = "moderate"  # "deep", "moderate", "thin", "critical"
    
    # Timestamps
    last_update: datetime = field(default_factory=datetime.utcnow)
    regime_started_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_no_trade(self) -> bool:
        """Check if we're in NO_TRADE state."""
        return self.regime == MarketRegimeState.NO_TRADE
    
    def is_opportunity_active(self) -> bool:
        """Check if any opportunity mode is active."""
        return self.opportunity.is_active() or self.crack_window.is_valid()
    
    def get_effective_regime_name(self) -> str:
        """Get human-readable regime name including opportunity."""
        base = self.regime.name
        
        if self.crack_window.is_valid():
            return f"{base}+CRACK({self.crack_window.direction})"
        elif self.opportunity.is_active():
            return f"{base}+{self.opportunity.mode.name}"
        return base
    
    def get_directional_bias(self) -> Tuple[str, float]:
        """
        Get overall directional bias.
        
        Returns: (direction, strength) where direction is "long"/"short"/"none"
        """
        # CRACK window has highest priority for direction
        if self.crack_window.is_valid():
            return self.crack_window.direction, self.crack_window.remaining_strength
        
        # Opportunity mode bias
        if self.opportunity.is_active() and self.opportunity.directional_bias != "none":
            return self.opportunity.directional_bias, self.opportunity.bias_strength
        
        # Regime-based bias
        if self.regime in [MarketRegimeState.STRONG_BULL, MarketRegimeState.BULL]:
            return "long", self.trend_strength
        elif self.regime in [MarketRegimeState.STRONG_BEAR, MarketRegimeState.BEAR]:
            return "short", self.trend_strength
        
        return "none", 0.0
    
    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.name,
            "confidence": self.confidence.overall(),
            "opportunity_mode": self.opportunity.mode.name,
            "crack_window_active": self.crack_window.is_valid(),
            "effective_regime": self.get_effective_regime_name(),
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "volatility_regime": self.volatility_regime,
            "liquidity_regime": self.liquidity_regime,
            "beta_sol_btc": self.beta_sol_btc,
            "dvol_zscore": self.dvol_zscore
        }


class RegimeTransitionRules:
    """
    Rules for regime transitions.
    
    Hysteresis for normal transitions, but CRACK can bypass.
    """
    
    # Persistence requirements (4H periods)
    NORMAL_PERSISTENCE = 3
    CRACK_BYPASS_PERSISTENCE = 1
    
    # Thresholds
    NO_TRADE_TRIGGERS = {
        "data_invalid": True,
        "dvol_zscore_extreme": 5.0,
        "liquidity_critical": True,
        "all_signals_conflict": True
    }
    
    CRACK_TRIGGERS = {
        "structure_break_magnitude": 0.02,  # 2% price move through structure
        "volume_spike_ratio": 2.5,          # 2.5x normal volume
        "vpin_extreme": 0.85                 # VPIN > 85%
    }
    
    @staticmethod
    def should_enter_no_trade(
        data_valid: bool,
        dvol_zscore: float,
        liquidity_critical: bool,
        signals_in_conflict: bool
    ) -> Tuple[bool, str]:
        """Check if we should enter NO_TRADE state."""
        
        if not data_valid:
            return True, "Data invalid or stale"
        
        if dvol_zscore >= RegimeTransitionRules.NO_TRADE_TRIGGERS["dvol_zscore_extreme"]:
            return True, f"DVOL extreme: zscore={dvol_zscore:.1f}"
        
        if liquidity_critical:
            return True, "Liquidity critical"
        
        if signals_in_conflict:
            return True, "All signals in directional conflict"
        
        return False, ""
    
    @staticmethod
    def should_trigger_crack(
        price_break_pct: float,
        volume_ratio: float,
        vpin: float,
        structure_level: float,
        current_price: float
    ) -> Tuple[bool, str, str]:
        """
        Check if we should trigger CRACK window.
        
        Returns: (should_trigger, direction, reason)
        """
        magnitude_ok = abs(price_break_pct) >= RegimeTransitionRules.CRACK_TRIGGERS["structure_break_magnitude"]
        volume_ok = volume_ratio >= RegimeTransitionRules.CRACK_TRIGGERS["volume_spike_ratio"]
        
        if magnitude_ok and volume_ok:
            direction = "long" if current_price > structure_level else "short"
            return True, direction, f"Structure break: {price_break_pct:.1%} on {volume_ratio:.1f}x volume"
        
        if vpin >= RegimeTransitionRules.CRACK_TRIGGERS["vpin_extreme"]:
            # VPIN extreme suggests toxic flow - but which direction?
            direction = "short"  # Assume toxic = selling pressure
            return True, direction, f"VPIN extreme: {vpin:.1%}"
        
        return False, "none", ""


class EnhancedRegimeNavigator:
    """
    Master regime navigator - 4H frequency.
    
    Integrates:
    - GMM/HMM regime detection
    - Vol-of-vol calculation
    - CRACK window detection
    - Opportunity mode management
    """
    
    def __init__(self):
        self.state = EnhancedRegimeState()
        self.transition_rules = RegimeTransitionRules()
        
        # Persistence tracking
        self.pending_regime: Optional[MarketRegimeState] = None
        self.persistence_count: int = 0
        
        # History for analysis
        self.regime_history: List[Tuple[datetime, MarketRegimeState]] = []
    
    def update_4h_tick(
        self,
        features: Dict,
        macro_context: Dict,
        sentiment_state: Dict,
        lead_lag_packet: Optional[Dict] = None
    ) -> EnhancedRegimeState:
        """
        Master update on 4H tick.
        
        This is the ONLY place regime state changes (except emergency NO_TRADE).
        """
        logger.info("4H Regime Update starting...")
        
        # 1. Check for NO_TRADE conditions first
        no_trade, reason = self._check_no_trade_conditions(features, sentiment_state)
        if no_trade:
            self._enter_no_trade(reason)
            return self.state
        
        # 2. Detect base regime from features
        detected_regime, confidence = self._detect_regime(features)
        
        # 3. Check for CRACK window trigger
        self._check_crack_trigger(features, lead_lag_packet)
        
        # 4. Check for opportunity mode from sentiment/flow
        self._check_opportunity_trigger(sentiment_state, lead_lag_packet)
        
        # 5. Apply hysteresis for regime change
        self._apply_hysteresis(detected_regime, confidence)
        
        # 6. Update cross-asset metrics
        self._update_cross_asset_metrics(features)
        
        # 7. Update volatility metrics
        self._update_volatility_metrics(features)
        
        # 8. Decay opportunity states
        self.state.opportunity.update_decay()
        self.state.crack_window.update_decay()
        
        self.state.last_update = datetime.now(timezone.utc)
        
        logger.info(f"4H Regime: {self.state.get_effective_regime_name()} | "
                   f"confidence={self.state.confidence.overall():.2f}")
        
        return self.state
    
    def emergency_no_trade(self, reason: str):
        """Emergency NO_TRADE - can be called outside 4H tick."""
        logger.warning(f"EMERGENCY NO_TRADE: {reason}")
        self._enter_no_trade(f"EMERGENCY: {reason}")
    
    def _enter_no_trade(self, reason: str):
        """Enter NO_TRADE state."""
        if self.state.regime != MarketRegimeState.NO_TRADE:
            self.regime_history.append((datetime.now(timezone.utc), self.state.regime))
        
        self.state.regime = MarketRegimeState.NO_TRADE
        self.state.regime_started_at = datetime.now(timezone.utc)
        self.state.opportunity.mode = OpportunityMode.NONE
        self.state.crack_window.is_active = False
        
        logger.warning(f"Entered NO_TRADE: {reason}")
    
    def _check_no_trade_conditions(
        self,
        features: Dict,
        sentiment_state: Dict
    ) -> Tuple[bool, str]:
        """Check if NO_TRADE conditions are met."""
        
        # Data validity
        data_age_s = features.get('data_age_seconds', 0)
        if data_age_s > 2.0:
            return True, f"Stale data: {data_age_s:.1f}s"
        
        # DVOL extreme
        dvol_zscore = features.get('dvol_zscore', 0)
        if dvol_zscore >= 5.0:
            return True, f"DVOL extreme: {dvol_zscore:.1f}"
        
        # Liquidity critical
        if features.get('liquidity_regime') == 'critical':
            return True, "Liquidity critical"
        
        # All signals in conflict (checked by Fusion, passed in sentiment_state)
        if sentiment_state.get('all_signals_conflict', False):
            return True, "All signals in directional conflict"
        
        return False, ""
    
    def _detect_regime(self, features: Dict) -> Tuple[MarketRegimeState, RegimeConfidence]:
        """Detect regime from features using GMM/HMM."""
        
        # Get price features
        returns_4h = features.get('returns_4h', 0)
        volatility = features.get('volatility_4h', 0.02)
        trend_strength = features.get('trend_strength', 0)
        
        # Simple regime detection (replace with actual GMM/HMM)
        if volatility > 0.05 and abs(returns_4h) < 0.01:
            regime = MarketRegimeState.VOLATILE_CHOP
            confidence = RegimeConfidence(probability=0.6, stability=0.3, consistency=0.4)
        elif returns_4h > 0.03 and trend_strength > 0.7:
            regime = MarketRegimeState.STRONG_BULL
            confidence = RegimeConfidence(probability=0.8, stability=0.7, consistency=0.8)
        elif returns_4h > 0.01 and trend_strength > 0.3:
            regime = MarketRegimeState.BULL
            confidence = RegimeConfidence(probability=0.7, stability=0.6, consistency=0.6)
        elif returns_4h < -0.03 and trend_strength > 0.7:
            regime = MarketRegimeState.STRONG_BEAR
            confidence = RegimeConfidence(probability=0.8, stability=0.7, consistency=0.8)
        elif returns_4h < -0.01 and trend_strength > 0.3:
            regime = MarketRegimeState.BEAR
            confidence = RegimeConfidence(probability=0.7, stability=0.6, consistency=0.6)
        else:
            regime = MarketRegimeState.NEUTRAL
            confidence = RegimeConfidence(probability=0.5, stability=0.5, consistency=0.5)
        
        return regime, confidence
    
    def _check_crack_trigger(self, features: Dict, lead_lag_packet: Optional[Dict]):
        """Check if CRACK window should be triggered."""
        
        if self.state.crack_window.is_valid():
            return  # Already in CRACK
        
        price_break = features.get('structure_break_pct', 0)
        volume_ratio = features.get('volume_ratio_vs_avg', 1.0)
        vpin = features.get('vpin', 0.35)
        structure_level = features.get('structure_level', 0)
        current_price = features.get('current_price', 0)
        
        should_crack, direction, reason = self.transition_rules.should_trigger_crack(
            price_break, volume_ratio, vpin, structure_level, current_price
        )
        
        if should_crack:
            self.state.crack_window = CrackWindow(
                is_active=True,
                direction=direction,
                triggered_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=8),  # 2x 4H periods
                trigger_reason=reason,
                break_magnitude=abs(price_break),
                volume_confirmation=volume_ratio >= 2.0,
                structure_level_broken=structure_level,
                remaining_strength=1.0
            )
            logger.info(f"CRACK window triggered: {direction} | {reason}")
    
    def _check_opportunity_trigger(self, sentiment_state: Dict, lead_lag_packet: Optional[Dict]):
        """
        Check if opportunity mode should be triggered from sentiment/flow.
        
        Gap 3 Fix: Handle both 'sentiment_shock' and 'shock_magnitude' field names.
        """
        
        if self.state.opportunity.is_active():
            return  # Already in opportunity mode
        
        # Gap 3: Standardize sentiment shock field name
        # Support both 'sentiment_shock' and 'shock_magnitude'
        sentiment_shock = sentiment_state.get('sentiment_shock', 
                         sentiment_state.get('shock_magnitude', 0))
        
        if abs(sentiment_shock) > 2.0:  # 2 sigma shock
            direction = "long" if sentiment_shock > 0 else "short"
            self.state.opportunity = OpportunityState(
                mode=OpportunityMode.SENTIMENT_TRIGGER,
                triggered_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
                trigger_reason=f"Sentiment shock: {sentiment_shock:.1f}σ",
                directional_bias=direction,
                bias_strength=min(1.0, abs(sentiment_shock) / 3),
                decay_window_hours=4.0,
                remaining_strength=1.0
            )
            logger.info(f"[Gap 3] Opportunity mode: SENTIMENT_TRIGGER | {direction} | shock={sentiment_shock:.1f}σ")
            return
        
        # Flow trigger from lead-lag
        if lead_lag_packet:
            flow_strength = lead_lag_packet.get('flow_alignment', 0)
            if abs(flow_strength) > 0.7:
                direction = "long" if flow_strength > 0 else "short"
                self.state.opportunity = OpportunityState(
                    mode=OpportunityMode.FLOW_TRIGGER,
                    triggered_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
                    trigger_reason=f"Flow alignment: {flow_strength:.2f}",
                    directional_bias=direction,
                    bias_strength=abs(flow_strength),
                    decay_window_hours=2.0,
                    remaining_strength=1.0
                )
                logger.info(f"Opportunity mode: FLOW_TRIGGER | {direction}")
    
    def _apply_hysteresis(self, detected_regime: MarketRegimeState, confidence: RegimeConfidence):
        """Apply hysteresis for regime transitions."""
        
        current_regime = self.state.regime
        
        # CRACK window can bypass persistence for aligned direction
        persistence_required = self.transition_rules.NORMAL_PERSISTENCE
        if self.state.crack_window.is_valid():
            persistence_required = self.transition_rules.CRACK_BYPASS_PERSISTENCE
        
        if detected_regime == current_regime:
            # Same regime - reset pending
            self.pending_regime = None
            self.persistence_count = 0
            self.state.confidence = confidence
            return
        
        if detected_regime == self.pending_regime:
            # Counting persistence
            self.persistence_count += 1
            if self.persistence_count >= persistence_required:
                # Transition confirmed
                self.regime_history.append((datetime.now(timezone.utc), current_regime))
                self.state.regime = detected_regime
                self.state.regime_started_at = datetime.now(timezone.utc)
                self.state.confidence = confidence
                self.pending_regime = None
                self.persistence_count = 0
                logger.info(f"Regime transition: {current_regime.name} -> {detected_regime.name}")
        else:
            # New candidate
            self.pending_regime = detected_regime
            self.persistence_count = 1
    
    def _update_cross_asset_metrics(self, features: Dict):
        """Update cross-asset beta metrics."""
        self.state.beta_sol_btc = features.get('beta_sol_btc', 1.0)
        self.state.beta_eth_btc = features.get('beta_eth_btc', 1.0)
        self.state.correlation_btc_eth_sol = features.get('correlation_3asset', 0.8)
    
    def _update_volatility_metrics(self, features: Dict):
        """Update volatility metrics."""
        vol = features.get('volatility_4h', 0.02)
        
        if vol < 0.01:
            self.state.volatility_regime = "low"
        elif vol < 0.03:
            self.state.volatility_regime = "normal"
        elif vol < 0.06:
            self.state.volatility_regime = "high"
        else:
            self.state.volatility_regime = "extreme"
        
        self.state.vol_of_vol = features.get('vol_of_vol', 0)
        self.state.dvol_zscore = features.get('dvol_zscore', 0)
        
        # Update trend
        trend_strength = features.get('trend_strength', 0)
        self.state.trend_strength = trend_strength
        
        if self.state.regime in [MarketRegimeState.BULL, MarketRegimeState.STRONG_BULL]:
            self.state.trend_direction = "bullish"
        elif self.state.regime in [MarketRegimeState.BEAR, MarketRegimeState.STRONG_BEAR]:
            self.state.trend_direction = "bearish"
        else:
            self.state.trend_direction = "neutral"
    
    def get_state(self) -> EnhancedRegimeState:
        """Get current regime state."""
        return self.state


# Singleton instance
_regime_navigator = EnhancedRegimeNavigator()

def get_regime_navigator() -> EnhancedRegimeNavigator:
    return _regime_navigator

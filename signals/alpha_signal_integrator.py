"""
================================================================================
HMATS v6.1.2 - Alpha Signal Integrator
================================================================================
Purpose: Integrate high-priority alpha modules into the main decision flow.
Integrated modules:
1. VolatilityAlphaAgent (volatility bias and intensity)
2. OnChainSentimentAlphaEngine (onchain + sentiment alpha)
3. NoTradeTriggerChecker (Constitution S3 guard)
4. OpportunityTriggerChecker (Constitution S2 trigger)
5. AuthorityMatrix (agent permission checks)
6. ExitAlphaManager (phase-aware exits)
Author: HMATS Alpha Integration
Version: 6.1.2
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AlphaSignalSource(Enum):
    """Alpha signal sources."""
    VOLATILITY = "volatility_alpha"
    ONCHAIN_SENTIMENT = "onchain_sentiment"
    OPPORTUNITY = "opportunity_trigger"
    NO_TRADE = "no_trade_trigger"
    EXIT_ALPHA = "exit_alpha"


@dataclass
class AlphaSignal:
    """Normalized alpha signal payload."""
    source: AlphaSignalSource
    direction: float = 0.0       # -1 to 1
    confidence: float = 0.0      # 0 to 1
    intensity: float = 0.0       # 0 to 1
    is_active: bool = False
    reason: str = ""
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "intensity": self.intensity,
            "is_active": self.is_active,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlphaState:
    """Aggregated alpha state."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Volatility Alpha
    vol_bias: str = "neutral"  # increase/neutral/suppress
    vol_intensity: float = 0.0
    
    # OnChain Sentiment
    onchain_direction: float = 0.0
    onchain_confidence: float = 0.0
    fear_greed_index: float = 50.0
    
    # Triggers
    no_trade_active: bool = False
    no_trade_reason: str = ""
    opportunity_active: bool = False
    opportunity_type: str = ""
    opportunity_direction: str = "none"
    
    # Exit Alpha
    exit_signal_active: bool = False
    scale_out_pct: float = 0.0
    
    # Combined
    combined_alpha_direction: float = 0.0
    combined_alpha_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "volatility": {
                "bias": self.vol_bias,
                "intensity": self.vol_intensity,
            },
            "onchain_sentiment": {
                "direction": self.onchain_direction,
                "confidence": self.onchain_confidence,
                "fear_greed": self.fear_greed_index,
            },
            "triggers": {
                "no_trade_active": self.no_trade_active,
                "no_trade_reason": self.no_trade_reason,
                "opportunity_active": self.opportunity_active,
                "opportunity_type": self.opportunity_type,
                "opportunity_direction": self.opportunity_direction,
            },
            "exit": {
                "signal_active": self.exit_signal_active,
                "scale_out_pct": self.scale_out_pct,
            },
            "combined": {
                "direction": self.combined_alpha_direction,
                "confidence": self.combined_alpha_confidence,
            },
        }


@dataclass
class AlphaIntegratorConfig:
    """Configuration for alpha signal integrator."""
    # Weights
    vol_alpha_weight: float = 0.15
    onchain_sentiment_weight: float = 0.20
    opportunity_weight: float = 0.15
    
    # Thresholds
    vol_intensity_threshold: float = 0.3
    opportunity_confidence_threshold: float = 0.5
    
    # Feature flags
    enable_volatility_alpha: bool = True
    enable_onchain_sentiment: bool = True
    enable_triggers: bool = True
    enable_exit_alpha: bool = True
    enable_authority_matrix: bool = True


class AlphaSignalIntegrator:
    """
    High-priority alpha integrator.

    Integrates volatility alpha, onchain sentiment alpha, trigger modules,
    authority checks, and exit alpha into one state object for fusion.
    """
    
    def __init__(self, config: AlphaIntegratorConfig = None):
        self.config = config or AlphaIntegratorConfig()
        
        # Components
        self._vol_alpha = None
        self._onchain_sentiment = None
        self._no_trade_checker = None
        self._opportunity_checker = None
        self._authority_matrix = None
        self._exit_alpha = None
        
        # State
        self._current_state = AlphaState()
        self._signals: Dict[str, AlphaSignal] = {}
        
        # Initialize
        self._init_components()
        
        logger.info("AlphaSignalIntegrator initialized")
    
    def _init_components(self):
        """Initialize optional alpha sub-components."""
        
        # 1. VolatilityAlphaAgent
        if self.config.enable_volatility_alpha:
            try:
                from agents.volatility_alpha_agent import get_volatility_alpha_agent
                self._vol_alpha = get_volatility_alpha_agent()
                logger.info("[OK] VolatilityAlphaAgent connected")
            except ImportError as e:
                logger.warning(f"VolatilityAlphaAgent not available: {e}")
            except Exception as e:
                logger.warning(f"VolatilityAlphaAgent init failed: {e}")
        
        # 2. OnChainSentimentAlphaEngine
        if self.config.enable_onchain_sentiment:
            try:
                from agents.onchain_sentiment_fusion import get_alpha_engine
                self._onchain_sentiment = get_alpha_engine()
                logger.info("[OK] OnChainSentimentAlphaEngine connected")
            except ImportError as e:
                logger.warning(f"OnChainSentimentAlphaEngine not available: {e}")
            except Exception as e:
                logger.warning(f"OnChainSentimentAlphaEngine init failed: {e}")
        
        # 3. NoTradeTriggerChecker
        if self.config.enable_triggers:
            try:
                from signals.no_trade_triggers import NoTradeTriggerChecker
                self._no_trade_checker = NoTradeTriggerChecker()
                logger.info("[OK] NoTradeTriggerChecker connected")
            except ImportError as e:
                logger.warning(f"NoTradeTriggerChecker not available: {e}")
            except Exception as e:
                logger.warning(f"NoTradeTriggerChecker init failed: {e}")
        
        # 4. OpportunityTriggerChecker
        if self.config.enable_triggers:
            try:
                from signals.opportunity_triggers import OpportunityTriggerChecker
                self._opportunity_checker = OpportunityTriggerChecker()
                logger.info("[OK] OpportunityTriggerChecker connected")
            except ImportError as e:
                logger.warning(f"OpportunityTriggerChecker not available: {e}")
            except Exception as e:
                logger.warning(f"OpportunityTriggerChecker init failed: {e}")
        
        # 5. AuthorityMatrix
        if self.config.enable_authority_matrix:
            try:
                from signals.authority_matrix import get_authority_matrix
                self._authority_matrix = get_authority_matrix()
                logger.info("[OK] AuthorityMatrix connected")
            except ImportError as e:
                logger.warning(f"AuthorityMatrix not available: {e}")
            except Exception as e:
                logger.warning(f"AuthorityMatrix init failed: {e}")
        
        # 6. ExitAlphaManager
        if self.config.enable_exit_alpha:
            try:
                from execution.exit_alpha import ExitAlphaManager
                self._exit_alpha = ExitAlphaManager()
                logger.info("[OK] ExitAlphaManager connected")
            except ImportError as e:
                logger.warning(f"ExitAlphaManager not available: {e}")
            except Exception as e:
                logger.warning(f"ExitAlphaManager init failed: {e}")
    
    def update(self, market_data: Dict = None, position_data: Dict = None) -> AlphaState:
        """
        Update all alpha signals.
        
        Args:
            market_data: Market data (price, volume, volatility, etc.)
            position_data: Position data (for exit alpha)
        
        Returns:
            Updated AlphaState
        """
        market_data = market_data or {}
        position_data = position_data or {}
        
        now = datetime.now()
        state = AlphaState(timestamp=now)
        signals = {}
        
        # 1. Volatility Alpha
        if self._vol_alpha:
            try:
                asset = market_data.get("symbol", "BTC")
                intent = self._vol_alpha.evaluate(asset)
                
                state.vol_bias = intent.vol_bias.value if hasattr(intent, 'vol_bias') else "neutral"
                state.vol_intensity = intent.intensity if hasattr(intent, 'intensity') else 0.0
                
                # Convert to direction signal
                vol_direction = 0.0
                if state.vol_bias == "increase":
                    vol_direction = 0.5  # Volatility expansion might favor momentum
                elif state.vol_bias == "suppress":
                    vol_direction = -0.3  # Suppress volatility exposure
                
                signals["volatility"] = AlphaSignal(
                    source=AlphaSignalSource.VOLATILITY,
                    direction=vol_direction,
                    confidence=min(state.vol_intensity, 1.0),
                    intensity=state.vol_intensity,
                    is_active=state.vol_intensity > self.config.vol_intensity_threshold,
                    reason=f"vol_bias={state.vol_bias}",
                    metadata={"bias": state.vol_bias, "rationale": getattr(intent, 'rationale_tags', [])},
                )
            except Exception as e:
                logger.debug(f"VolatilityAlpha update failed: {e}")
        
        # 2. OnChain Sentiment
        if self._onchain_sentiment:
            try:
                token = market_data.get("symbol", "SOL")
                fusion_signal = self._onchain_sentiment.get_signal_for_fusion(token)
                
                state.onchain_direction = fusion_signal.get("direction", 0.0)
                state.onchain_confidence = fusion_signal.get("confidence", 0.0)
                
                # Get fear/greed if available
                try:
                    stats = self._onchain_sentiment.get_stats()
                    state.fear_greed_index = stats.get("fear_greed_index", 50.0)
                except:
                    pass
                
                signals["onchain_sentiment"] = AlphaSignal(
                    source=AlphaSignalSource.ONCHAIN_SENTIMENT,
                    direction=state.onchain_direction,
                    confidence=state.onchain_confidence,
                    intensity=abs(state.onchain_direction) * state.onchain_confidence,
                    is_active=state.onchain_confidence > 0.3,
                    reason=fusion_signal.get("trade_bias", "neutral"),
                    metadata={"fear_greed": state.fear_greed_index},
                )
            except Exception as e:
                logger.debug(f"OnChainSentiment update failed: {e}")
        
        # 3. No Trade Triggers
        if self._no_trade_checker:
            try:
                trigger_result = self._no_trade_checker.check_all_triggers(market_data)
                
                state.no_trade_active = trigger_result.triggered if hasattr(trigger_result, 'triggered') else False
                state.no_trade_reason = trigger_result.reason if hasattr(trigger_result, 'reason') else ""
                
                if state.no_trade_active:
                    signals["no_trade"] = AlphaSignal(
                        source=AlphaSignalSource.NO_TRADE,
                        direction=0.0,
                        confidence=1.0,
                        intensity=1.0,
                        is_active=True,
                        reason=state.no_trade_reason,
                        metadata={"category": getattr(trigger_result, 'category', 'unknown')},
                    )
            except Exception as e:
                logger.debug(f"NoTradeTrigger check failed: {e}")
        
        # 4. Opportunity Triggers
        if self._opportunity_checker:
            try:
                opp_result = self._opportunity_checker.check_all_triggers(market_data)
                
                state.opportunity_active = opp_result.triggered if hasattr(opp_result, 'triggered') else False
                state.opportunity_type = opp_result.group.value if hasattr(opp_result, 'group') and opp_result.group else ""
                state.opportunity_direction = opp_result.direction if hasattr(opp_result, 'direction') else "none"
                
                if state.opportunity_active:
                    opp_direction = 1.0 if state.opportunity_direction == "long" else (-1.0 if state.opportunity_direction == "short" else 0.0)
                    
                    signals["opportunity"] = AlphaSignal(
                        source=AlphaSignalSource.OPPORTUNITY,
                        direction=opp_direction,
                        confidence=opp_result.confidence if hasattr(opp_result, 'confidence') else 0.5,
                        intensity=1.0,
                        is_active=True,
                        reason=opp_result.reason if hasattr(opp_result, 'reason') else state.opportunity_type,
                        metadata={"type": state.opportunity_type, "ttl_hours": getattr(opp_result, 'ttl_hours', 0)},
                    )
            except Exception as e:
                logger.debug(f"OpportunityTrigger check failed: {e}")
        
        # 5. Exit Alpha
        if self._exit_alpha and position_data:
            try:
                exit_result = self._exit_alpha.evaluate_scale_out(
                    position=position_data,
                    market_data=market_data,
                )
                
                if hasattr(exit_result, 'should_scale_out') and exit_result.should_scale_out:
                    state.exit_signal_active = True
                    state.scale_out_pct = getattr(exit_result, 'scale_out_pct', 0.25)
                    
                    signals["exit_alpha"] = AlphaSignal(
                        source=AlphaSignalSource.EXIT_ALPHA,
                        direction=-1.0 if position_data.get("side") == "long" else 1.0,
                        confidence=0.8,
                        intensity=state.scale_out_pct,
                        is_active=True,
                        reason=getattr(exit_result, 'trigger', 'unknown'),
                        metadata={"scale_out_pct": state.scale_out_pct},
                    )
            except Exception as e:
                logger.debug(f"ExitAlpha evaluation failed: {e}")
        
        # 6. Calculate combined signal
        if signals:
            weighted_direction = 0.0
            total_weight = 0.0
            confidences = []
            
            weight_map = {
                "volatility": self.config.vol_alpha_weight,
                "onchain_sentiment": self.config.onchain_sentiment_weight,
                "opportunity": self.config.opportunity_weight,
            }
            
            for name, sig in signals.items():
                if sig.is_active and name in weight_map:
                    w = weight_map[name]
                    weighted_direction += sig.direction * w * sig.confidence
                    total_weight += w * sig.confidence
                    confidences.append(sig.confidence)
            
            if total_weight > 0:
                state.combined_alpha_direction = weighted_direction / total_weight
            
            if confidences:
                state.combined_alpha_confidence = sum(confidences) / len(confidences)
        
        # Update internal state
        self._current_state = state
        self._signals = signals
        
        return state
    
    def get_trading_signal(self, symbol: str) -> Tuple[float, float]:
        """
        Get the fused signal for downstream decision fusion.
        Args:
            symbol: Trading symbol.
        
        Returns:
            (direction, confidence)
        """
        # Check no-trade first
        if self._current_state.no_trade_active:
            return 0.0, 0.0
        
        return self._current_state.combined_alpha_direction, self._current_state.combined_alpha_confidence
    
    def should_block_trade(self) -> Tuple[bool, str]:
        """
        Check whether NO_TRADE should block trading.
        Returns:
            (should_block, reason)
        """
        if self._current_state.no_trade_active:
            return True, self._current_state.no_trade_reason
        return False, ""
    
    def is_opportunity_active(self) -> Tuple[bool, str, str]:
        """
        Check whether an opportunity trigger is active.

        Returns:
            (active, type, direction)
        """
        return (
            self._current_state.opportunity_active,
            self._current_state.opportunity_type,
            self._current_state.opportunity_direction,
        )
    
    def get_exit_signal(self) -> Tuple[bool, float]:
        """
        Get exit-alpha recommendation.
        Returns:
            (should_exit, scale_out_pct)
        """
        return self._current_state.exit_signal_active, self._current_state.scale_out_pct
    
    def check_authority(self, agent: str, action: str, mode: str = "NORMAL") -> bool:
        """
        Check whether an agent is allowed to perform an action.
        Args:
            agent: Agent name.
            action: Action type (VETO/DECIDE/TRIGGER).
            mode: System mode.
        
        Returns:
            True when action is permitted.
        """
        if not self._authority_matrix:
            return True  # Permissive by default
        
        try:
            from signals.authority_matrix import Agent, SystemMode, AuthorityRole
            
            agent_enum = Agent[agent.upper()] if hasattr(Agent, agent.upper()) else None
            mode_enum = SystemMode[mode.upper()] if hasattr(SystemMode, mode.upper()) else SystemMode.NORMAL
            
            if agent_enum:
                roles = self._authority_matrix.get_roles(agent_enum, mode_enum)
                action_role = AuthorityRole[action.upper()] if hasattr(AuthorityRole, action.upper()) else None
                return action_role in roles if action_role else False
        except Exception as e:
            logger.debug(f"Authority check failed: {e}")
        
        return True
    
    def get_state(self) -> AlphaState:
        """Get current alpha state."""
        return self._current_state
    
    def get_signals(self) -> Dict[str, AlphaSignal]:
        """Get latest raw alpha signals."""
        return self._signals
    
    def get_status(self) -> Dict[str, Any]:
        """Get component availability and signal summary."""
        return {
            "components": {
                "vol_alpha": self._vol_alpha is not None,
                "onchain_sentiment": self._onchain_sentiment is not None,
                "no_trade_checker": self._no_trade_checker is not None,
                "opportunity_checker": self._opportunity_checker is not None,
                "authority_matrix": self._authority_matrix is not None,
                "exit_alpha": self._exit_alpha is not None,
            },
            "state": self._current_state.to_dict(),
            "active_signals": [s.source.value for s in self._signals.values() if s.is_active],
        }


# =============================================================================
# SINGLETON
# =============================================================================

_alpha_integrator: Optional[AlphaSignalIntegrator] = None


def get_alpha_integrator(config: AlphaIntegratorConfig = None) -> AlphaSignalIntegrator:
    """Get or create alpha signal integrator singleton."""
    global _alpha_integrator
    if _alpha_integrator is None:
        _alpha_integrator = AlphaSignalIntegrator(config)
    elif config is not None and _alpha_integrator.config != config:
        logger.info("[ALPHA_INTEGRATOR] Config changed; refreshing singleton instance")
        _alpha_integrator = AlphaSignalIntegrator(config)
    return _alpha_integrator


def reset_alpha_integrator():
    """Reset singleton."""
    global _alpha_integrator
    _alpha_integrator = None


__all__ = [
    "AlphaSignalSource",
    "AlphaSignal",
    "AlphaState",
    "AlphaIntegratorConfig",
    "AlphaSignalIntegrator",
    "get_alpha_integrator",
    "reset_alpha_integrator",
]

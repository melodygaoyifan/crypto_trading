"""
HMATS v3.2 - Idle Module Integrator
====================================
Purpose: Wire up previously idle modules with correct semantics.

Integrated Modules:
- macro_integrator.get_regime_weight_adjustment() -> risk cap / leverage
- ConsensusState from vllm_inference_wrapper.py -> trigger/veto/mode-switch
- global_context_informer.py -> context aggregation
- state_vector_augmentor.py -> DRL state enhancement

Philosophy:
- Sentiment/Macro are TRIGGER/VETO/MODE-SWITCH, NOT linear multipliers
- Sentiment shock -> OpportunityMode
- Chaos + conflict -> NO_TRADE
- Macro adjusts risk cap / leverage multiplier, NEVER flips direction
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging
import numpy as np

from integration.core.market_regime_v32 import EnhancedRegimeState, OpportunityMode, OpportunityState
from signals.authority_fusion import SentimentSignal, MacroContext

logger = logging.getLogger(__name__)


@dataclass
class SentimentState:
    """
    Aggregated sentiment state from vLLM/Consensus.
    
    This is NOT a linear multiplier.
    It produces TRIGGERS and VETOS.
    """
    # Core sentiment
    direction: float = 0.0  # -1 to 1 (bearish to bullish)
    confidence: float = 0.5  # 0 to 1
    
    # Shock detection (sigma from baseline)
    shock_magnitude: float = 0.0
    shock_direction: str = "none"  # "bullish", "bearish", "none"
    
    # Consensus tracking
    consensus_divergence: float = 0.0  # How much this diverges from consensus
    consensus_stability: float = 0.5   # How stable is consensus
    
    # Trigger outputs (NOT multipliers)
    trigger_opportunity: bool = False
    trigger_veto: bool = False
    veto_direction: str = "none"  # Which direction to veto
    
    # Mode switch
    mode_switch: Optional[str] = None  # "RISK_ON", "RISK_OFF", "DEFENSIVE"
    
    def is_shock(self) -> bool:
        """Check if sentiment shock detected."""
        return abs(self.shock_magnitude) > 2.0
    
    def should_trigger_opportunity(self) -> Tuple[bool, str, str]:
        """
        Check if sentiment should trigger opportunity mode.
        
        P1 FIX: Mock sentiment cannot trigger OPPORTUNITY.
        
        Returns: (trigger, direction, reason)
        """
        # P1: Check if sentiment is allowed to trigger opportunity
        try:
            from .vllm_inference_wrapper import sentiment_can_trigger_opportunity, is_sentiment_mock
            
            if not sentiment_can_trigger_opportunity():
                if is_sentiment_mock():
                    return False, "none", "Mock sentiment - OPPORTUNITY trigger disabled"
                return False, "none", "Sentiment OPPORTUNITY trigger disabled"
        except ImportError:
            pass  # vllm_inference_wrapper not available
        
        if not self.is_shock():
            return False, "none", ""
        
        direction = "long" if self.shock_magnitude > 0 else "short"
        return True, direction, f"Sentiment shock: {self.shock_magnitude:.1f}σ"
    
    def should_veto_direction(self, intended_direction: str) -> Tuple[bool, str]:
        """
        Check if sentiment should veto a direction.
        
        Returns: (should_veto, reason)
        """
        if not self.is_shock():
            return False, ""
        
        # Extreme bearish shock -> veto longs
        if self.shock_magnitude < -3.0 and intended_direction == "long":
            return True, f"Extreme bearish sentiment ({self.shock_magnitude:.1f}σ) vetoes long"
        
        # Extreme bullish shock -> veto shorts
        if self.shock_magnitude > 3.0 and intended_direction == "short":
            return True, f"Extreme bullish sentiment ({self.shock_magnitude:.1f}σ) vetoes short"
        
        return False, ""


class SentimentIntegrator:
    """
    Integrates sentiment from vLLM/Consensus into the system.
    
    CRITICAL: This produces TRIGGERS and VETOS, not multipliers.
    """
    
    # Thresholds
    SHOCK_THRESHOLD_SIGMA = 2.0      # 2σ = shock
    EXTREME_SHOCK_SIGMA = 3.0        # 3σ = extreme, can veto
    CHAOS_THRESHOLD = 0.8            # High chaos + conflict = NO_TRADE
    
    def __init__(self):
        self.history: list = []
        self.baseline_direction: float = 0.0
        self.baseline_volatility: float = 0.3
    
    def process_consensus(
        self,
        raw_sentiment: Dict
    ) -> SentimentState:
        """
        Process raw consensus data into SentimentState.
        
        Expected raw_sentiment format:
        {
            "direction": float (-1 to 1),
            "confidence": float (0 to 1),
            "sources": [{"name": str, "direction": float, "weight": float}],
            "volatility": float
        }
        """
        
        direction = raw_sentiment.get("direction", 0.0)
        confidence = raw_sentiment.get("confidence", 0.5)
        volatility = raw_sentiment.get("volatility", self.baseline_volatility)
        
        # Calculate shock (deviation from baseline in sigma)
        deviation = direction - self.baseline_direction
        sigma = volatility if volatility > 0 else 0.3
        shock_magnitude = deviation / sigma
        
        shock_direction = "none"
        if shock_magnitude > self.SHOCK_THRESHOLD_SIGMA:
            shock_direction = "bullish"
        elif shock_magnitude < -self.SHOCK_THRESHOLD_SIGMA:
            shock_direction = "bearish"
        
        # Calculate divergence from source consensus
        sources = raw_sentiment.get("sources", [])
        divergence = self._calculate_divergence(sources)
        
        # Update baseline (slow adaptation)
        self.baseline_direction = 0.95 * self.baseline_direction + 0.05 * direction
        self.baseline_volatility = 0.95 * self.baseline_volatility + 0.05 * volatility
        
        # Determine triggers
        state = SentimentState(
            direction=direction,
            confidence=confidence,
            shock_magnitude=shock_magnitude,
            shock_direction=shock_direction,
            consensus_divergence=divergence,
            consensus_stability=1 - divergence
        )
        
        # Set trigger flags
        trigger, trigger_dir, _ = state.should_trigger_opportunity()
        state.trigger_opportunity = trigger
        
        # Check for chaos (high divergence + conflict)
        if divergence > self.CHAOS_THRESHOLD:
            state.mode_switch = "DEFENSIVE"
        
        self.history.append(state)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return state
    
    def _calculate_divergence(self, sources: list) -> float:
        """Calculate divergence among sentiment sources."""
        if len(sources) < 2:
            return 0.0
        
        directions = [s.get("direction", 0) for s in sources]
        if not directions:
            return 0.0
        
        return float(np.std(directions))
    
    def to_signal(self, state: SentimentState) -> SentimentSignal:
        """Convert SentimentState to SentimentSignal for Fusion."""
        return SentimentSignal(
            direction=state.direction,
            shock_magnitude=state.shock_magnitude,
            divergence=state.consensus_divergence,
            reasoning=f"Sentiment: dir={state.direction:.2f}, shock={state.shock_magnitude:.1f}σ"
        )


@dataclass
class MacroState:
    """
    Macro state from macro_integrator.
    
    CRITICAL: Macro adjusts risk cap and leverage, NEVER flips direction.
    """
    # Risk metrics
    risk_appetite: float = 0.5  # 0 to 1 (risk-off to risk-on)
    leverage_multiplier: float = 1.0  # 0.5 to 1.5
    
    # Individual signals
    dxy_signal: float = 0.0       # DXY impact
    us10y_signal: float = 0.0     # Treasury impact
    vix_signal: float = 0.0       # VIX impact
    spx_signal: float = 0.0       # S&P impact
    
    # ETF flows
    btc_etf_flow: str = "NEUTRAL"  # STRONG_INFLOW, MODERATE_INFLOW, etc.
    eth_etf_flow: str = "NEUTRAL"
    combined_etf_signal: float = 0.0
    
    # Stablecoin
    stablecoin_liquidity: str = "MODERATE"  # HIGH, MODERATE, LOW
    stablecoin_signal: float = 0.0
    
    # Regime suggestion
    suggested_regime_adjustment: str = "none"  # "RISK_ON", "RISK_OFF", "CRISIS"
    
    def get_risk_cap_multiplier(self) -> float:
        """
        Get risk cap multiplier (NOT direction multiplier).
        
        This only affects position SIZE, not DIRECTION.
        """
        if self.risk_appetite < 0.3:
            return 0.5  # Reduce max position by 50%
        elif self.risk_appetite < 0.5:
            return 0.75
        elif self.risk_appetite > 0.7:
            return 1.2
        return 1.0
    
    def get_leverage_cap(self) -> float:
        """Get max leverage allowed by macro conditions."""
        return max(0.5, min(1.5, self.leverage_multiplier))


class MacroIntegratorWrapper:
    """
    Wrapper for macro_integrator that enforces correct semantics.
    
    CRITICAL RULE: Macro NEVER flips direction, only adjusts risk/leverage.
    """
    
    def __init__(self):
        # Import the actual macro integrator
        try:
            from ..defense.macro_integrator import MacroIntegrator
            self.integrator = MacroIntegrator()
        except ImportError:
            logger.warning("MacroIntegrator not found, using defaults")
            self.integrator = None
    
    def get_macro_state(self) -> MacroState:
        """Get current macro state."""
        
        if self.integrator is None:
            return MacroState()
        
        # Get raw adjustment from integrator
        try:
            adjustment = self.integrator.get_regime_weight_adjustment()
        except Exception as e:
            logger.error(f"Failed to get macro adjustment: {e}")
            return MacroState()
        
        # Parse adjustment into MacroState
        state = MacroState(
            risk_appetite=adjustment.get("risk_appetite", 0.5),
            leverage_multiplier=adjustment.get("position_scale", 1.0),
            btc_etf_flow=adjustment.get("btc_etf_signal", "NEUTRAL"),
            eth_etf_flow=adjustment.get("eth_etf_signal", "NEUTRAL"),
            stablecoin_liquidity=adjustment.get("stablecoin_signal", "MODERATE"),
            suggested_regime_adjustment=adjustment.get("regime_suggestion", "none")
        )
        
        # Calculate combined signals
        state.combined_etf_signal = self._parse_etf_signal(state.btc_etf_flow, state.eth_etf_flow)
        state.stablecoin_signal = self._parse_stablecoin_signal(state.stablecoin_liquidity)
        
        return state
    
    def _parse_etf_signal(self, btc_flow: str, eth_flow: str) -> float:
        """Parse ETF flow into numeric signal."""
        signal_map = {
            "STRONG_INFLOW": 0.5,
            "MODERATE_INFLOW": 0.25,
            "NEUTRAL": 0.0,
            "MODERATE_OUTFLOW": -0.25,
            "STRONG_OUTFLOW": -0.5
        }
        btc_sig = signal_map.get(btc_flow, 0.0)
        eth_sig = signal_map.get(eth_flow, 0.0)
        return (btc_sig + eth_sig) / 2
    
    def _parse_stablecoin_signal(self, liquidity: str) -> float:
        """Parse stablecoin liquidity into numeric signal."""
        signal_map = {
            "HIGH": 0.3,
            "MODERATE": 0.0,
            "LOW": -0.3
        }
        return signal_map.get(liquidity, 0.0)
    
    def to_context(self, state: MacroState) -> MacroContext:
        """Convert MacroState to MacroContext for Fusion."""
        return MacroContext(
            risk_appetite=state.risk_appetite,
            leverage_multiplier=state.leverage_multiplier,
            etf_flow_signal=state.btc_etf_flow,
            stablecoin_signal=state.stablecoin_liquidity
        )


class GlobalContextAggregator:
    """
    Aggregates context from multiple sources.
    
    Integrates:
    - global_context_informer
    - state_vector_augmentor
    - macro state
    - sentiment state
    """
    
    def __init__(self):
        self.sentiment_integrator = SentimentIntegrator()
        self.macro_wrapper = MacroIntegratorWrapper()
        
        # Import global context informer
        try:
            from .global_context_informer import GlobalContextInformer
            self.context_informer = GlobalContextInformer()
        except ImportError:
            logger.warning("GlobalContextInformer not found")
            self.context_informer = None
        
        # Import state vector augmentor
        try:
            from .state_vector_augmentor import StateVectorAugmentor
            self.state_augmentor = StateVectorAugmentor()
        except ImportError:
            logger.warning("StateVectorAugmentor not found")
            self.state_augmentor = None
    
    def aggregate(
        self,
        raw_sentiment: Optional[Dict] = None,
        base_features: Optional[Dict] = None
    ) -> Dict:
        """
        Aggregate all context into a unified dict.
        
        Returns context dict with:
        - macro: MacroState
        - sentiment: SentimentState
        - global_context: from informer
        - augmented_state: for DRL
        - triggers: any triggered events
        """
        
        # Get macro state
        macro_state = self.macro_wrapper.get_macro_state()
        
        # Process sentiment
        if raw_sentiment:
            sentiment_state = self.sentiment_integrator.process_consensus(raw_sentiment)
        else:
            sentiment_state = SentimentState()
        
        # Get global context
        global_context = {}
        if self.context_informer:
            try:
                global_context = self.context_informer.get_context()
            except Exception as e:
                logger.error(f"Failed to get global context: {e}")
        
        # Get augmented state for DRL
        augmented_state = None
        if self.state_augmentor and base_features:
            try:
                # Convert base_features dict to state vector
                if isinstance(base_features, dict):
                    base_state = np.array([
                        base_features.get('price_change', 0),
                        base_features.get('volatility', 0),
                        base_features.get('volume_ratio', 1),
                        base_features.get('rsi', 50) / 100,
                        base_features.get('macd', 0),
                    ] + [0] * 59, dtype=np.float32)[:64]
                elif isinstance(base_features, np.ndarray):
                    base_state = base_features[:64] if len(base_features) >= 64 else np.pad(base_features, (0, 64 - len(base_features)))
                else:
                    base_state = np.zeros(64, dtype=np.float32)
                
                augmented_state = self.state_augmentor.get_state_vector(
                    symbol="COMBINED",
                    base_state=base_state,
                    context=global_context
                )
            except Exception as e:
                logger.error(f"Failed to augment state: {e}")
        
        # Aggregate triggers
        triggers = self._aggregate_triggers(sentiment_state, macro_state)
        
        return {
            "macro": macro_state,
            "macro_context": self.macro_wrapper.to_context(macro_state),
            "sentiment": sentiment_state,
            "sentiment_signal": self.sentiment_integrator.to_signal(sentiment_state),
            "global_context": global_context,
            "augmented_state": augmented_state,
            "triggers": triggers,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _aggregate_triggers(
        self,
        sentiment: SentimentState,
        macro: MacroState
    ) -> Dict:
        """Aggregate triggers from all sources."""
        
        triggers = {
            "opportunity": False,
            "opportunity_direction": "none",
            "opportunity_reason": "",
            "no_trade": False,
            "no_trade_reason": "",
            "mode_switch": None,
            "veto_direction": None
        }
        
        # Sentiment triggers
        if sentiment.trigger_opportunity:
            triggers["opportunity"] = True
            opp_trigger, opp_dir, opp_reason = sentiment.should_trigger_opportunity()
            triggers["opportunity_direction"] = opp_dir
            triggers["opportunity_reason"] = opp_reason
        
        if sentiment.trigger_veto:
            triggers["veto_direction"] = sentiment.veto_direction
        
        if sentiment.mode_switch:
            triggers["mode_switch"] = sentiment.mode_switch
        
        # Macro triggers
        if macro.suggested_regime_adjustment == "CRISIS":
            triggers["mode_switch"] = "DEFENSIVE"
        
        # Chaos detection -> NO_TRADE
        if sentiment.consensus_divergence > 0.9 and sentiment.confidence < 0.3:
            triggers["no_trade"] = True
            triggers["no_trade_reason"] = "Sentiment chaos: high divergence, low confidence"
        
        return triggers


# Singleton instances
_sentiment_integrator = SentimentIntegrator()
_macro_wrapper = MacroIntegratorWrapper()
_context_aggregator = GlobalContextAggregator()

def get_sentiment_integrator() -> SentimentIntegrator:
    return _sentiment_integrator

def get_macro_wrapper() -> MacroIntegratorWrapper:
    return _macro_wrapper

def get_context_aggregator() -> GlobalContextAggregator:
    return _context_aggregator

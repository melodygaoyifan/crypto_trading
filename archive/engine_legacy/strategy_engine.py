"""
HMATS v3.2 - Strategy Engine
=============================
Integrates all Strategy Constitution components into a single engine.

Components:
- SystemModeManager (Section 1)
- OpportunityTriggerChecker (Section 2)
- NoTradeTriggerChecker (Section 3)
- AlphaThresholdCalculator (Section 4)
- TrancheManager (Section 5)
- SOLExposureManager (Section 6)
- AuthorityMatrix (Section 7)

This is the main entry point for strategy decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from orchestration.system_mode import SystemModeManager, SystemMode, get_mode_manager
from .opportunity_triggers import OpportunityTriggerChecker, OpportunityState, get_opportunity_checker
from .no_trade_triggers import NoTradeTriggerChecker, NoTradeState, get_no_trade_checker
from market.alpha_threshold import AlphaThresholdCalculator, AlphaEstimate, get_alpha_calculator
from risk.tranche_manager import TrancheManager, TrancheDecision, get_tranche_manager
from execution.sol_exposure_manager import SOLExposureManager, ExposureMode, get_sol_exposure_manager
from .authority_matrix import AuthorityMatrix, AuthorityDecision, Agent, AgentSignal, get_authority_matrix
from execution.trade_intent import TradeIntent, IntentFactory, IntentAuthority, IntentUrgency

logger = logging.getLogger(__name__)


@dataclass
class StrategyDecision:
    """Complete strategy decision output."""
    # Mode
    mode: SystemMode
    mode_reason: str
    
    # Authority
    authority_decision: AuthorityDecision
    
    # Alpha
    alpha_estimate: AlphaEstimate
    should_execute: bool
    
    # Tranches
    tranche_decision: TrancheDecision
    
    # Exposure
    final_exposures: Dict[str, float]
    exposure_mode: ExposureMode
    
    # Intent
    intents: Dict[str, TradeIntent]
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    opportunity_state: Optional[OpportunityState] = None
    no_trade_state: Optional[NoTradeState] = None
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.name,
            "mode_reason": self.mode_reason,
            "authority": self.authority_decision.to_dict(),
            "alpha": self.alpha_estimate.to_dict(),
            "should_execute": self.should_execute,
            "tranche": self.tranche_decision.to_dict(),
            "exposures": self.final_exposures,
            "exposure_mode": self.exposure_mode.name,
            "timestamp": self.timestamp.isoformat()
        }


class StrategyEngine:
    """
    Main strategy engine implementing the Strategy Constitution.
    
    This class orchestrates all strategy components and produces
    final trading decisions.
    """
    
    def __init__(self):
        # Initialize all components
        self.mode_manager = get_mode_manager()
        self.opportunity_checker = get_opportunity_checker()
        self.no_trade_checker = get_no_trade_checker()
        self.alpha_calculator = get_alpha_calculator()
        self.tranche_manager = get_tranche_manager()
        self.sol_manager = get_sol_exposure_manager()
        self.authority_matrix = get_authority_matrix()
        
        logger.info("Strategy Engine initialized with all constitution components")
    
    def process_4h_tick(
        self,
        market_data: Dict,
        lead_lag_data: Dict,
        sentiment_data: Dict,
        sol_flow_data: Dict,
        signal_data: Dict,
        integrity_data: Dict,
        execution_data: Dict,
        is_4h_bar_close: bool = True
    ) -> StrategyDecision:
        """
        Process a 4H tick and produce strategy decision.
        
        This is the main entry point for strategy logic.
        
        Args:
            market_data: Price, volume, DVOL, etc.
            lead_lag_data: Binance-Kraken edge data
            sentiment_data: Sentiment shock, sources
            sol_flow_data: DEX, on-chain flow
            signal_data: Quant, DRL signals
            integrity_data: CRC32, ShadowLedger status
            execution_data: Execution failure count
            is_4h_bar_close: Whether this is a 4H bar close
        
        Returns:
            StrategyDecision with complete decision output
        """
        
        logger.info("=" * 60)
        logger.info("STRATEGY ENGINE: 4H TICK PROCESSING")
        logger.info("=" * 60)
        
        # === Step 1: Check NO_TRADE triggers ===
        no_trade_state = self.no_trade_checker.check_all_triggers(
            market_data=market_data,
            signal_data=signal_data,
            integrity_data=integrity_data,
            execution_data=execution_data
        )
        
        # === Step 2: Check OPPORTUNITY triggers ===
        opportunity_state = self.opportunity_checker.check_all_triggers(
            market_data=market_data,
            lead_lag_data=lead_lag_data,
            sentiment_data=sentiment_data,
            sol_flow_data=sol_flow_data
        )
        
        # === Step 3: Build trigger dicts for mode manager ===
        no_trade_triggers = {
            'data_integrity_fail': any(c.trigger_type.name == 'DATA_INTEGRITY_FAIL' 
                                       for c in no_trade_state.active_conditions),
            'execution_blocked': any(c.trigger_type.name == 'EXECUTION_BLOCKED' 
                                     for c in no_trade_state.active_conditions),
            'stale_data': any(c.trigger_type.name == 'STALE_DATA' 
                              for c in no_trade_state.active_conditions),
            'flash_crash': any(c.trigger_type.name == 'FLASH_CRASH' 
                               for c in no_trade_state.active_conditions),
            'extreme_dvol': any(c.trigger_type.name == 'EXTREME_DVOL' 
                                for c in no_trade_state.active_conditions),
            'all_conflict_flat': any(c.trigger_type.name == 'ALL_CONFLICT_FLAT' 
                                     for c in no_trade_state.active_conditions),
            'correlation_collapse_risk': any(c.trigger_type.name == 'CORRELATION_COLLAPSE' 
                                             for c in no_trade_state.active_conditions),
            'liquidity_critical': any(c.trigger_type.name == 'LIQUIDITY_CRITICAL' 
                                      for c in no_trade_state.active_conditions),
            'feed_disagreement': any(c.trigger_type.name == 'FEED_DISAGREEMENT' 
                                     for c in no_trade_state.active_conditions)
        }
        
        opportunity_triggers = {
            'crack_window': any(t.group.value == 'B' for t in opportunity_state.active_triggers) if opportunity_state.is_active else False,
            'lead_lag_edge': any(t.group.value == 'A' for t in opportunity_state.active_triggers) if opportunity_state.is_active else False,
            'volatility_expansion': any(t.group.value == 'C' for t in opportunity_state.active_triggers) if opportunity_state.is_active else False,
            'sentiment_shock': any(t.group.value == 'D' for t in opportunity_state.active_triggers) if opportunity_state.is_active else False,
            'sol_flow_surge': any(t.group.value == 'E' for t in opportunity_state.active_triggers) if opportunity_state.is_active else False
        }
        
        # === Step 4: Update mode ===
        lead_lag_edge_bps = lead_lag_data.get('net_edge_bps', 0)
        mode, mode_reason = self.mode_manager.check_and_update(
            no_trade_triggers=no_trade_triggers,
            opportunity_triggers=opportunity_triggers,
            lead_lag_edge_bps=lead_lag_edge_bps,
            is_4h_bar=is_4h_bar_close
        )
        
        if mode_reason:
            mode_reason_str = mode_reason.name
        else:
            mode_reason_str = "Default"
        
        logger.info(f"Mode: {mode.name} (reason: {mode_reason_str})")
        
        # === Step 5: Build agent signals ===
        agent_signals = self._build_agent_signals(signal_data, sentiment_data, market_data)
        
        # === Step 6: Resolve authority ===
        crack_direction = None
        if opportunity_state.is_active and opportunity_state.primary_trigger:
            if opportunity_state.primary_trigger.can_set_direction:
                crack_direction = opportunity_state.primary_trigger.direction
        
        authority_decision = self.authority_matrix.resolve(
            mode=mode,
            signals=agent_signals,
            regime_name=market_data.get('regime_name', 'NEUTRAL'),
            crack_direction=crack_direction
        )
        
        # === Step 7: Check alpha threshold ===
        lead_lag_confirmed = opportunity_triggers.get('lead_lag_edge', False)
        crack_active = opportunity_triggers.get('crack_window', False)
        
        alpha_estimate = self.alpha_calculator.should_execute(
            signal_strength=abs(authority_decision.direction),
            regime_confidence=market_data.get('regime_confidence', 0.5),
            mode=mode,
            is_maker=False,  # Assume taker for now
            lead_lag_edge_confirmed=lead_lag_confirmed,
            crack_window_active=crack_active
        )
        
        should_execute = alpha_estimate.meets_threshold and mode != SystemMode.NO_TRADE
        
        logger.info(f"Alpha: {alpha_estimate.estimated_alpha_bps:.0f}bps vs {alpha_estimate.threshold_bps:.0f}bps threshold")
        logger.info(f"Should execute: {should_execute}")
        
        # === Step 8: Calculate exposures ===
        if mode == SystemMode.NO_TRADE or not should_execute:
            final_exposures = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
            exposure_mode = ExposureMode.CONSERVATIVE
        else:
            # Get target exposures per asset
            target_direction = authority_decision.direction
            target_exposure = authority_decision.exposure
            
            # Create dummy intents for exposure manager
            from execution.trade_intent import IntentFactory, IntentAuthority
            
            btc_intent = IntentFactory.create_directional_intent(
                asset="BTC",
                target_exposure=target_exposure * 0.25,  # BTC baseline
                authority=IntentAuthority.AUTHORITY_QUANT,
                confidence=authority_decision.confidence,
                reasoning="Strategy engine"
            )
            eth_intent = IntentFactory.create_directional_intent(
                asset="ETH",
                target_exposure=target_exposure * 0.25,  # ETH baseline
                authority=IntentAuthority.AUTHORITY_QUANT,
                confidence=authority_decision.confidence,
                reasoning="Strategy engine"
            )
            sol_intent = IntentFactory.create_directional_intent(
                asset="SOL",
                target_exposure=target_exposure * 0.50,  # SOL baseline
                authority=IntentAuthority.AUTHORITY_QUANT,
                confidence=authority_decision.confidence,
                reasoning="Strategy engine"
            )
            
            # Get regime state for SOL manager
            from integration.core.market_regime_v32 import EnhancedRegimeState, MarketRegimeState, RegimeConfidence
            regime_state = EnhancedRegimeState(
                regime=MarketRegimeState.BULL if target_direction > 0 else MarketRegimeState.BEAR,
                confidence=RegimeConfidence(
                    probability=market_data.get('regime_confidence', 0.5),
                    stability=0.5,
                    consistency=0.5
                ),
                beta_sol_btc=market_data.get('beta_sol_btc', 1.0),
                correlation_btc_eth_sol=market_data.get('correlation', 0.8)
            )
            
            # Check if CRACK or opportunity active
            if opportunity_state.is_active:
                from integration.core.market_regime_v32 import CrackWindow, OpportunityState as OppState, OpportunityMode
                if crack_active:
                    regime_state.crack_window = CrackWindow(
                        is_active=True,
                        direction=crack_direction or "long",
                        remaining_strength=1.0
                    )
                else:
                    regime_state.opportunity = OppState(
                        mode=OpportunityMode.SENTIMENT_TRIGGER,
                        remaining_strength=1.0
                    )
            
            flow_alignment = lead_lag_data.get('cvd_divergence', 0)
            
            final_exposures = self.sol_manager.calculate_target_exposures(
                btc_intent=btc_intent,
                eth_intent=eth_intent,
                sol_intent=sol_intent,
                regime_state=regime_state,
                flow_alignment=flow_alignment
            )
            exposure_mode = self.sol_manager.state.exposure_mode
        
        logger.info(f"Exposure mode: {exposure_mode.name}")
        logger.info(f"Final exposures: BTC={final_exposures.get('BTC', 0):.2f}, "
                   f"ETH={final_exposures.get('ETH', 0):.2f}, SOL={final_exposures.get('SOL', 0):.2f}")
        
        # === Step 9: Manage tranches ===
        # For simplicity, just check SOL
        tranche_decision = self.tranche_manager.decide(
            asset="SOL",
            mode=mode,
            target_exposure=final_exposures.get("SOL", 0),
            current_price=market_data.get('sol_price', market_data.get('current_price', 0)),
            market_data=market_data,
            signal_data=signal_data,
            is_4h_bar_close=is_4h_bar_close
        )
        
        logger.info(f"Tranche: {tranche_decision.action.name} -> {tranche_decision.new_exposure:.2f}")
        
        # === Step 10: Create intents ===
        intents = self._create_intents(
            final_exposures=final_exposures,
            mode=mode,
            authority_decision=authority_decision,
            market_data=market_data
        )
        
        # === Build final decision ===
        decision = StrategyDecision(
            mode=mode,
            mode_reason=mode_reason_str,
            authority_decision=authority_decision,
            alpha_estimate=alpha_estimate,
            should_execute=should_execute,
            tranche_decision=tranche_decision,
            final_exposures=final_exposures,
            exposure_mode=exposure_mode,
            intents=intents,
            opportunity_state=opportunity_state,
            no_trade_state=no_trade_state
        )
        
        logger.info("=" * 60)
        logger.info("STRATEGY ENGINE: DECISION COMPLETE")
        logger.info("=" * 60)
        
        return decision
    
    def _build_agent_signals(
        self,
        signal_data: Dict,
        sentiment_data: Dict,
        market_data: Dict
    ) -> Dict[Agent, AgentSignal]:
        """Build agent signals from raw data."""
        
        signals = {}
        
        # Quant Base
        signals[Agent.QUANT_BASE] = AgentSignal(
            agent=Agent.QUANT_BASE,
            direction=signal_data.get('quant_direction', 0),
            confidence=signal_data.get('quant_confidence', 0.5)
        )
        
        # Quant Aggressive
        signals[Agent.QUANT_AGGRESSIVE] = AgentSignal(
            agent=Agent.QUANT_AGGRESSIVE,
            direction=signal_data.get('quant_aggressive_direction', signal_data.get('quant_direction', 0)),
            confidence=signal_data.get('quant_aggressive_confidence', 0.5)
        )
        
        # DRL
        signals[Agent.DRL] = AgentSignal(
            agent=Agent.DRL,
            direction=signal_data.get('drl_direction', 0),
            confidence=signal_data.get('drl_confidence', 0.5),
            entropy=signal_data.get('drl_entropy', 0.5)
        )
        
        # Lead-Lag (no direction authority)
        signals[Agent.LEAD_LAG] = AgentSignal(
            agent=Agent.LEAD_LAG,
            direction=0,  # Lead-Lag never sets direction
            confidence=0
        )
        
        # Sentiment
        signals[Agent.SENTIMENT] = AgentSignal(
            agent=Agent.SENTIMENT,
            direction=sentiment_data.get('sentiment_direction', 0),
            confidence=sentiment_data.get('sentiment_confidence', 0.5),
            shock_sigma=sentiment_data.get('sentiment_shock_sigma', 0)
        )
        
        # Macro
        signals[Agent.MACRO] = AgentSignal(
            agent=Agent.MACRO,
            direction=0,  # Macro doesn't set direction
            confidence=0,
            risk_appetite=market_data.get('macro_risk_appetite', 0.5),
            leverage_cap=market_data.get('macro_leverage_cap', 1.0)
        )
        
        return signals
    
    def _create_intents(
        self,
        final_exposures: Dict[str, float],
        mode: SystemMode,
        authority_decision: AuthorityDecision,
        market_data: Dict
    ) -> Dict[str, TradeIntent]:
        """Create trade intents from exposures."""
        
        intents = {}
        
        # Determine urgency
        if mode == SystemMode.NO_TRADE:
            urgency = IntentUrgency.IMMEDIATE
            authority = IntentAuthority.HARD_STATE
        elif mode == SystemMode.OPPORTUNITY:
            urgency = IntentUrgency.IMMEDIATE
            authority = IntentAuthority.OPPORTUNITY_MODE
        else:
            urgency = IntentUrgency.NORMAL
            authority = IntentAuthority.AUTHORITY_QUANT
        
        for asset, exposure in final_exposures.items():
            if mode == SystemMode.NO_TRADE:
                intent = IntentFactory.create_flat_intent(
                    asset=asset,
                    authority=authority,
                    reasoning=f"NO_TRADE mode",
                    urgency=urgency
                )
            else:
                intent = IntentFactory.create_directional_intent(
                    asset=asset,
                    target_exposure=exposure,
                    authority=authority,
                    confidence=authority_decision.confidence,
                    reasoning=f"Strategy engine: {authority_decision.deciding_agent.value}",
                    regime_name=market_data.get('regime_name', 'UNKNOWN'),
                    urgency=urgency
                )
            
            intents[asset] = intent
        
        return intents
    
    def get_state_summary(self) -> Dict:
        """Get summary of current strategy state."""
        return {
            "mode": self.mode_manager.get_mode().name,
            "mode_state": {
                "decay_strength": self.mode_manager.state.decay_strength,
                "time_in_mode_hours": self.mode_manager.state.time_in_mode_hours()
            },
            "alpha_thresholds": self.alpha_calculator.get_thresholds_summary(),
            "authority_matrix": self.authority_matrix.get_matrix_summary(),
            "sol_exposure_state": self.sol_manager.get_state().to_dict()
        }


# Singleton instance
_strategy_engine = StrategyEngine()

def get_strategy_engine() -> StrategyEngine:
    return _strategy_engine

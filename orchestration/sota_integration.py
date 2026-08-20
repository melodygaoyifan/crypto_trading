"""
================================================================================
HMATS v5.2.0 - Unified SOTA Integration Layer
================================================================================
Purpose: Unified integration of all SOTA modules into cohesive system

This module integrates:
1. Risk Management (SOTA Risk Controller, Correlation Sizer)
2. Execution Engine (Level2 Analyzer, RL Fallback, SOTA Scheduler)
3. Monitoring (Enhanced Dashboard, Alert Manager)
4. Strategy (Adaptive Weighting, Sentiment, Microstructure, OnChain)
5. Event Replay (Formalized replay with fault injection)
6. Explainability (LLM Decision Explainer)
7. Architecture (Plugin Registry)

Author: HMATS SOTA Upgrade
Version: 5.2.0
================================================================================
"""

import logging
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE IMPORTS WITH FALLBACKS
# =============================================================================

def safe_import(module_path: str, class_name: str = None,
                archived: str = ""):
    """Safely import a module or class.

    [P336] `archived` names where a module went when it was deliberately
    retired. Two of the 21 targets here (`infra.event_replay`,
    `core.plugin_registry`) live only under `archive/`, so every boot logged
    two WARNINGs about modules nobody intends to have — an alert whose only
    resolutions are theatre or ignoring it, which is how a real CRITICAL stops
    being read (P202/P303). An expected absence is reported ONCE at INFO and
    names the archive path; anything else still WARNs, so this cannot become a
    blanket mute (a guard weakened to admit a known case stops catching the
    unknown ones — P248).
    """
    try:
        module = __import__(module_path, fromlist=[class_name] if class_name else [])
        if class_name:
            return getattr(module, class_name, None)
        return module
    except ImportError as e:
        if archived:
            logger.info(
                "%s is ARCHIVED by decision (%s); the feature it backs is "
                "inert and its config flag cannot take effect", module_path,
                archived)
        else:
            logger.warning(f"Failed to import {module_path}: {e}")
        return None


# Risk Management
SOTARiskController = safe_import("risk.sota_risk_controller", "SOTARiskController")
CorrelationAwarePositionSizer = safe_import("risk.correlation_position_sizer", "CorrelationAwarePositionSizer")

# Execution Engine
Level2Analyzer = safe_import("execution.level2_analyzer", "Level2Analyzer")
RLExecutionFallback = safe_import("execution.rl_fallback", "RLExecutionFallback")
SOTAExecutionScheduler = safe_import("execution.sota_scheduler", "SOTAExecutionScheduler")

# Monitoring
AlertManager = safe_import("infra.alert_manager", "AlertManager")

# Signals & Agents
# [v3.3-B11] Fix adaptive weighting import: old path was signals.adaptive_weighting (missing)
_RealWeightManager = safe_import("signals.adaptive_weight_v521", "AdaptiveWeightManagerV521")

if _RealWeightManager is not None:
    class AdaptiveAgentWeighting(_RealWeightManager):
        """Thin adapter: maps legacy call-site names -> v521 methods."""
        def get_current_weights(self):
            return self.get_all_weights()
        def get_performance_summary(self):
            return self.get_dashboard_data()
else:
    AdaptiveAgentWeighting = None
SentimentLLMAgent = safe_import("agents.sentiment_llm_agent", "SentimentLLMAgent")
MicrostructureArbitrageAgent = safe_import("agents.microstructure_agent", "MicrostructureArbitrageAgent")
OnChainSolanaAgent = safe_import("agents.onchain_solana_agent", "OnChainSolanaAgent")

# Event Replay
EventReplayManager = safe_import("infra.event_replay", "EventReplayManager",
                                archived="archive/infra/event_replay.py")

# Explainability
LLMDecisionExplainer = safe_import("analytics.llm_explainer", "LLMDecisionExplainer")

# Plugin Registry
PluginRegistry = safe_import("core.plugin_registry", "PluginRegistry",
                             archived="archive/core/plugin_registry.py")

# Flow Signal Integrator (?
FlowSignalIntegrator = safe_import("signals.flow_signal_integrator", "FlowSignalIntegrator")

# Alpha Signal Integrator ( Alpha )
AlphaSignalIntegrator = safe_import("signals.alpha_signal_integrator", "AlphaSignalIntegrator")

# ===  (v6.1.2-INTEGRATED) ===

# Volatility Alpha Agent (?
VolatilityAlphaAgent = safe_import("agents.volatility_alpha_agent", "VolatilityAlphaAgent")

# OnChain Sentiment Fusion ()
OnChainSentimentAlphaEngine = safe_import("agents.onchain_sentiment_fusion", "OnChainSentimentAlphaEngine")

# Strategy Constitution 
NoTradeTriggerChecker = safe_import("signals.no_trade_triggers", "NoTradeTriggerChecker")
get_no_trade_checker = safe_import("signals.no_trade_triggers", "get_no_trade_checker")

OpportunityTriggerChecker = safe_import("signals.opportunity_triggers", "OpportunityTriggerChecker")
get_opportunity_checker = safe_import("signals.opportunity_triggers", "get_opportunity_checker")

AuthorityMatrix = safe_import("signals.authority_matrix", "AuthorityMatrix")
get_authority_matrix = safe_import("signals.authority_matrix", "get_authority_matrix")

# Exit Alpha (?
ExitAlphaManager = safe_import("execution.exit_alpha", "ExitAlphaManager")
get_exit_manager = safe_import("execution.exit_alpha", "get_exit_manager")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SOTAIntegrationConfig:
    """Configuration for SOTA integration."""
    
    # Feature flags
    enable_risk_controller: bool = True
    enable_correlation_sizing: bool = True
    enable_level2_analysis: bool = True
    enable_rl_fallback: bool = True
    enable_execution_scheduler: bool = True
    enable_alerts: bool = True
    enable_adaptive_weighting: bool = True
    enable_sentiment_agent: bool = True
    enable_microstructure_agent: bool = True
    enable_onchain_agent: bool = True
    # [P336] INERT: infra.event_replay is archived, and the consumer is
    # guarded by `and EventReplayManager`, so this flag reads as enabled
    # and can never take effect. Kept (removing a config field is a
    # contract change) but annotated, per the P307 no-effect-switch roster.
    enable_event_replay: bool = True
    enable_explainability: bool = True
    enable_plugins: bool = True
    enable_flow_integrator: bool = True  # ?
    enable_alpha_integrator: bool = True  #  Alpha 
    
    # ===  (v6.1.2-INTEGRATED) ===
    enable_volatility_alpha: bool = True       # ?
    enable_onchain_sentiment_fusion: bool = True  # 
    enable_no_trade_triggers: bool = True      # Strategy Constitution 3
    enable_opportunity_triggers: bool = True   # Strategy Constitution 2
    enable_authority_matrix: bool = True       # 
    enable_exit_alpha: bool = True             # ?
    
    # Risk settings
    risk_config: Dict = field(default_factory=dict)
    
    # Execution settings
    execution_config: Dict = field(default_factory=dict)
    
    # Alert settings
    alert_email_enabled: bool = False
    # [P307b] NO READER: declared, never consulted. `enable_alerts`
    # beside it IS read; this one is not, so a popup cannot be
    # disabled (or enabled) through it.
    alert_popup_enabled: bool = True
    
    # Agent weights (v6.1.2-INTEGRATED:  volatility, onchain_fusion)
    agent_base_weights: Dict[str, float] = field(default_factory=lambda: {
        "quant": 0.18,              # 
        "drl": 0.08,                #  (shadow only)
        "sentiment": 0.08,          # 
        "microstructure": 0.10,     #  (VPIN/OFI)
        "onchain": 0.06,            #  (SOL )
        "flow": 0.12,               # ?
        "volatility": 0.10,         # ?
        "onchain_fusion": 0.08,     # 
        "alpha": 0.12,              #  Alpha  (trigger/exit/authority)
        "risk": 0.08                # 
    })
    
    # Plugin directories
    plugin_dirs: List[str] = field(default_factory=lambda: [
        "plugins/strategies",
        "plugins/risk",
        "plugins/execution"
    ])


# =============================================================================
# INTEGRATION LAYER
# =============================================================================

class SOTAIntegration:
    """
    Master integration layer for all SOTA modules.
    
    Coordinates:
    - Risk management decisions
    - Execution optimization
    - Multi-agent signal fusion
    - System monitoring and alerts
    - Decision explainability
    """
    
    def __init__(self, config: SOTAIntegrationConfig = None):
        self.config = config or SOTAIntegrationConfig()
        
        # Initialize components
        self.components = {}
        self._init_components()
        
        # State
        self.is_running = False
        self.last_decision: Optional[Dict] = None
        self.decision_history: List[Dict] = []
        
        logger.info(f"SOTAIntegration initialized with {len(self.components)} components")
    
    def _init_components(self):
        """Initialize all SOTA components."""
        
        # Risk Controller
        if self.config.enable_risk_controller and SOTARiskController:
            try:
                self.components["risk_controller"] = SOTARiskController()
                logger.info("[OK] SOTA Risk Controller initialized")
            except Exception as e:
                logger.error(f"Failed to init risk controller: {e}")
        
        # Correlation Position Sizer
        if self.config.enable_correlation_sizing and CorrelationAwarePositionSizer:
            try:
                self.components["correlation_sizer"] = CorrelationAwarePositionSizer()
                logger.info("[OK] Correlation Position Sizer initialized")
            except Exception as e:
                logger.error(f"Failed to init correlation sizer: {e}")
        
        # Level 2 Analyzer
        if self.config.enable_level2_analysis and Level2Analyzer:
            try:
                self.components["level2_analyzer"] = Level2Analyzer()
                logger.info("[OK] Level 2 Analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to init level2 analyzer: {e}")
        
        # RL Fallback
        if self.config.enable_rl_fallback and RLExecutionFallback:
            try:
                self.components["rl_fallback"] = RLExecutionFallback()
                logger.info("[OK] RL Execution Fallback initialized")
            except Exception as e:
                logger.error(f"Failed to init rl fallback: {e}")
        
        # Execution Scheduler
        if self.config.enable_execution_scheduler and SOTAExecutionScheduler:
            try:
                self.components["execution_scheduler"] = SOTAExecutionScheduler()
                logger.info("[OK] SOTA Execution Scheduler initialized")
            except Exception as e:
                logger.error(f"Failed to init execution scheduler: {e}")
        
        # Alert Manager
        if self.config.enable_alerts and AlertManager:
            try:
                self.components["alert_manager"] = AlertManager()
                logger.info("[OK] Alert Manager initialized")
            except Exception as e:
                logger.error(f"Failed to init alert manager: {e}")
        
        # Adaptive Weighting
        if self.config.enable_adaptive_weighting and AdaptiveAgentWeighting:
            try:
                self.components["adaptive_weighting"] = AdaptiveAgentWeighting()
                logger.info("[OK] Adaptive Agent Weighting initialized")
            except Exception as e:
                logger.error(f"Failed to init adaptive weighting: {e}")
        
        # Sentiment Agent
        if self.config.enable_sentiment_agent and SentimentLLMAgent:
            try:
                self.components["sentiment_agent"] = SentimentLLMAgent()
                logger.info("[OK] Sentiment LLM Agent initialized")
            except Exception as e:
                logger.error(f"Failed to init sentiment agent: {e}")
        
        # Microstructure Agent
        if self.config.enable_microstructure_agent and MicrostructureArbitrageAgent:
            try:
                self.components["microstructure_agent"] = MicrostructureArbitrageAgent()
                logger.info("[OK] Microstructure Arbitrage Agent initialized")
            except Exception as e:
                logger.error(f"Failed to init microstructure agent: {e}")
        
        # OnChain Agent
        if self.config.enable_onchain_agent and OnChainSolanaAgent:
            try:
                self.components["onchain_agent"] = OnChainSolanaAgent()
                logger.info("[OK] OnChain Solana Agent initialized")
            except Exception as e:
                logger.error(f"Failed to init onchain agent: {e}")
        
        # Event Replay
        if self.config.enable_event_replay and EventReplayManager:
            try:
                self.components["event_replay"] = EventReplayManager()
                logger.info("[OK] Event Replay Manager initialized")
            except Exception as e:
                logger.error(f"Failed to init event replay: {e}")
        
        # LLM Explainer
        if self.config.enable_explainability and LLMDecisionExplainer:
            try:
                self.components["llm_explainer"] = LLMDecisionExplainer()
                logger.info("[OK] LLM Decision Explainer initialized")
            except Exception as e:
                logger.error(f"Failed to init llm explainer: {e}")
        
        # Plugin Registry
        if self.config.enable_plugins and PluginRegistry:
            try:
                self.components["plugin_registry"] = PluginRegistry()
                logger.info("[OK] Plugin Registry initialized")
            except Exception as e:
                logger.error(f"Failed to init plugin registry: {e}")
        
        # Flow Signal Integrator (?
        if self.config.enable_flow_integrator and FlowSignalIntegrator:
            try:
                self.components["flow_integrator"] = FlowSignalIntegrator()
                logger.info("[OK] Flow Signal Integrator initialized (flow monitor)")
            except Exception as e:
                logger.error(f"Failed to init flow integrator: {e}")
        
        # Alpha Signal Integrator ( Alpha )
        if self.config.enable_alpha_integrator and AlphaSignalIntegrator:
            try:
                self.components["alpha_integrator"] = AlphaSignalIntegrator()
                logger.info("[OK] Alpha Signal Integrator initialized (Vol/Trigger/Exit Alpha)")
            except Exception as e:
                logger.error(f"Failed to init alpha integrator: {e}")
        
        # === ?(v6.1.2-INTEGRATED) ===
        
        # Volatility Alpha Agent (?
        if self.config.enable_volatility_alpha and VolatilityAlphaAgent:
            try:
                self.components["volatility_alpha"] = VolatilityAlphaAgent()
                logger.info("[OK] Volatility Alpha Agent initialized")
            except Exception as e:
                logger.error(f"Failed to init volatility alpha: {e}")
        
        # OnChain Sentiment Fusion ()
        if self.config.enable_onchain_sentiment_fusion and OnChainSentimentAlphaEngine:
            try:
                self.components["onchain_fusion"] = OnChainSentimentAlphaEngine()
                logger.info("[OK] OnChain Sentiment Fusion initialized")
            except Exception as e:
                logger.error(f"Failed to init onchain sentiment fusion: {e}")
        
        # NoTrade Trigger Checker (Strategy Constitution 3)
        if self.config.enable_no_trade_triggers and get_no_trade_checker:
            try:
                self.components["no_trade_checker"] = get_no_trade_checker()
                logger.info("[OK] NoTrade Trigger Checker initialized (Constitution S3)")
            except Exception as e:
                logger.error(f"Failed to init no_trade checker: {e}")
        
        # Opportunity Trigger Checker (Strategy Constitution 2)
        if self.config.enable_opportunity_triggers and get_opportunity_checker:
            try:
                self.components["opportunity_checker"] = get_opportunity_checker()
                logger.info("[OK] Opportunity Trigger Checker initialized (Constitution S2)")
            except Exception as e:
                logger.error(f"Failed to init opportunity checker: {e}")
        
        # Authority Matrix ()
        if self.config.enable_authority_matrix and get_authority_matrix:
            try:
                self.components["authority_matrix"] = get_authority_matrix()
                logger.info("[OK] Authority Matrix initialized")
            except Exception as e:
                logger.error(f"Failed to init authority matrix: {e}")
        
        # Exit Alpha Manager (?
        if self.config.enable_exit_alpha and get_exit_manager:
            try:
                self.components["exit_alpha"] = get_exit_manager()
                logger.info("[OK] Exit Alpha Manager initialized")
            except Exception as e:
                logger.error(f"Failed to init exit alpha: {e}")
    
    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    
    def check_trade_allowed(self, symbol: str, side: str, size: float,
                            portfolio: Dict[str, float] = None,
                            market_data: Dict = None) -> Dict:
        """
        Check if trade is allowed through all risk gates.
        
        Returns:
            Dict with allowed, reason, adjustments
        """
        result = {
            "allowed": True,
            "reason": "",
            "adjusted_size": size,
            "risk_flags": [],
            "correlation_adjustment": 1.0,
            "opportunity_triggered": False,
            "opportunity_group": None
        }
        
        market_data = market_data or {}
        
        # === NoTrade Trigger Check (Strategy Constitution 3) - HIGHEST PRIORITY ===
        no_trade_checker = self.components.get("no_trade_checker")
        if no_trade_checker:
            try:
                # ?
                signal_data = market_data.get("signals", {})
                integrity_data = market_data.get("integrity", {
                    "data_stale": False,
                    "feed_healthy": True,
                    "latency_ok": True
                })
                execution_data = market_data.get("execution", {
                    "blocked": False,
                    "rate_limited": False
                })
                
                no_trade_state = no_trade_checker.check_all_triggers(
                    market_data=market_data,
                    signal_data=signal_data,
                    integrity_data=integrity_data,
                    execution_data=execution_data
                )
                if no_trade_state.should_no_trade:
                    result["allowed"] = False
                    trigger_name = no_trade_state.primary_reason.name if no_trade_state.primary_reason else "UNKNOWN"
                    reason = no_trade_state.active_conditions[0].reason if no_trade_state.active_conditions else "NO_TRADE active"
                    result["reason"] = f"NO_TRADE [{trigger_name}]: {reason}"
                    result["risk_flags"].append(f"NO_TRADE: {trigger_name}")
                    logger.warning(f"[CONSTITUTION 3] Trade blocked: {reason}")
                    return result
            except Exception as e:
                logger.error(f"NoTrade trigger check failed: {e}")
        
        # === Opportunity Trigger Check (Strategy Constitution 2) ===
        opportunity_checker = self.components.get("opportunity_checker")
        if opportunity_checker:
            try:
                # ?
                lead_lag_data = market_data.get("lead_lag", {})
                sentiment_data = market_data.get("sentiment", {})
                sol_flow_data = market_data.get("sol_flow", {})
                
                opp_state = opportunity_checker.check_all_triggers(
                    market_data=market_data,
                    lead_lag_data=lead_lag_data,
                    sentiment_data=sentiment_data,
                    sol_flow_data=sol_flow_data
                )
                if opp_state.is_active:
                    result["opportunity_triggered"] = True
                    if opp_state.primary_trigger:
                        result["opportunity_group"] = opp_state.primary_trigger.group.value if opp_state.primary_trigger.group else None
                        result["risk_flags"].append(f"OPPORTUNITY: {opp_state.primary_trigger.group.value if opp_state.primary_trigger.group else 'UNKNOWN'}")
                        logger.info(f"[CONSTITUTION 2] Opportunity triggered: {opp_state.primary_trigger.reason}")
            except Exception as e:
                logger.error(f"Opportunity trigger check failed: {e}")
        
        # SOTA Risk Controller check
        risk_controller = self.components.get("risk_controller")
        if risk_controller:
            try:
                can_trade, reason = risk_controller.can_open_position(symbol, side, size)
                if not can_trade:
                    result["allowed"] = False
                    result["reason"] = reason
                    result["risk_flags"].append(f"RISK_CONTROLLER: {reason}")
                    self._send_alert("TRADE_BLOCKED", f"Trade blocked by risk controller: {reason}")
                    return result
            except Exception as e:
                logger.error(f"Risk controller check failed: {e}")
        
        # Correlation sizing adjustment
        # [CORR-3] PATH B DEAD CODE: This computes adjusted_size and correlation_adjustment,
        # but main.py only reads "allowed" and "opportunity_triggered" from the result.
        # Additionally, the sizer's method is `calculate_adjusted_size` but this calls
        # `compute_adjusted_size` - AttributeError at runtime, so this block never executes.
        # The ACTIVE correlation path is Path A (CorrelationRealtimeController).
        correlation_sizer = self.components.get("correlation_sizer")
        if correlation_sizer and portfolio:
            try:
                sizing_result = correlation_sizer.compute_adjusted_size(
                    symbol=symbol,
                    base_size=size,
                    current_positions=portfolio
                )
                if sizing_result:
                    result["adjusted_size"] = sizing_result.adjusted_size
                    result["correlation_adjustment"] = sizing_result.reduction_factor
                    if sizing_result.reduction_factor < 1.0:
                        result["risk_flags"].append(
                            f"CORRELATION: Size reduced by {(1-sizing_result.reduction_factor)*100:.0f}%"
                        )
            except Exception as e:
                logger.error(f"Correlation sizer check failed: {e}")
        
        return result
    
    def update_risk_state(self, equity: float, realized_pnl: float,
                          prices: Dict[str, float] = None):
        """Update risk management state."""
        risk_controller = self.components.get("risk_controller")
        if risk_controller:
            try:
                risk_controller.update_equity(equity=equity, realized_pnl_today=realized_pnl)
                if prices:
                    risk_controller.update_prices(prices)
            except Exception as e:
                logger.error(f"Failed to update risk state: {e}")
    
    def check_exit_signal(self, symbol: str, position: Dict, market_data: Dict = None) -> Dict:
        """
        Check if position should be exited or scaled out.
        Uses ExitAlphaManager for phase-aware exit decisions.
        
        Args:
            symbol: Trading symbol
            position: Current position dict with entry_price, current_price, size, unrealized_pnl
            market_data: Market data including regime phase
            
        Returns:
            Dict with should_exit, scale_out_pct, trigger, reason
        """
        result = {
            "should_exit": False,
            "should_scale_out": False,
            "scale_out_pct": 0.0,
            "trigger": None,
            "reason": "",
            "runner_action": None
        }
        
        exit_manager = self.components.get("exit_alpha")
        if not exit_manager:
            return result
        
        market_data = market_data or {}
        
        try:
            # 
            entry_price = position.get("entry_price", 0)
            current_price = position.get("current_price", entry_price)
            position_size = position.get("size", 0)
            bars_held = position.get("bars_held", 0)
            
            #  regime phase ()
            phase_result = market_data.get("phase_result")
            if phase_result is None:
                #  phase result
                try:
                    from market.phase_detector import RegimePhase, RegimePhaseResult
                    phase_result = RegimePhaseResult(phase=RegimePhase.UNDEFINED)
                except ImportError:
                    phase_result = None
            
            if phase_result is None:
                return result
                
            crack_weight = market_data.get("crack_weight", 0.0)
            momentum = market_data.get("momentum", 0.0)
            
            # ?
            exit_signal = exit_manager.evaluate_scale_out(
                asset=symbol,
                current_price=current_price,
                entry_price=entry_price,
                position_size=position_size,
                bars_held=bars_held,
                phase_result=phase_result,
                crack_weight=crack_weight,
                momentum=momentum,
                drl_output=None
            )
            
            if exit_signal:
                if exit_signal.should_scale_out:
                    result["should_scale_out"] = True
                    result["scale_out_pct"] = exit_signal.scale_out_pct
                    result["trigger"] = exit_signal.trigger.name if exit_signal.trigger else "UNKNOWN"
                    result["reason"] = exit_signal.reason
                    logger.info(f"[EXIT_ALPHA] Scale out {symbol}: {exit_signal.scale_out_pct*100:.0f}% - {exit_signal.reason}")
                
        except Exception as e:
            logger.error(f"Exit signal check failed: {e}")
        
        return result
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    def plan_execution(self, symbol: str, side: str, size: float,
                       urgency: str = "medium", orderbook: Dict = None) -> Dict:
        """
        Plan optimal execution strategy.
        
        Returns:
            Dict with execution plan
        """
        plan = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "strategy": "immediate",
            "slices": [{"size": size, "type": "market"}],
            "use_rl": False,
            "level2_recommendation": None
        }
        
        # Level 2 analysis
        level2 = self.components.get("level2_analyzer")
        if level2 and orderbook:
            try:
                analysis = level2.analyze(orderbook)
                if analysis:
                    plan["level2_recommendation"] = analysis.get("recommendation")
                    
                    # Adjust strategy based on liquidity
                    if analysis.get("liquidity_level") == "thin":
                        plan["strategy"] = "twap"
                        plan["slices"] = self._create_twap_slices(size, 10)
            except Exception as e:
                logger.error(f"Level 2 analysis failed: {e}")
        
        # RL vs Fallback decision
        rl_fallback = self.components.get("rl_fallback")
        if rl_fallback:
            try:
                decision = rl_fallback.decide(symbol, side, size)
                plan["use_rl"] = decision.get("use_rl", False)
                if not plan["use_rl"]:
                    plan["fallback_reason"] = decision.get("reason", "uncertainty")
            except Exception as e:
                logger.error(f"RL fallback decision failed: {e}")
        
        return plan
    
    def _create_twap_slices(self, total_size: float, num_slices: int) -> List[Dict]:
        """Create TWAP execution slices."""
        slice_size = total_size / num_slices
        return [{"size": slice_size, "type": "limit", "delay_ms": i * 60000} 
                for i in range(num_slices)]
    
    # =========================================================================
    # SIGNAL FUSION
    # =========================================================================
    
    def get_fused_signal(self, symbol: str, market_data: Dict) -> Dict:
        """
        Get fused trading signal from all agents.
        
        Returns:
            Dict with direction, confidence, components
        """
        signals = {}
        weights = self.config.agent_base_weights.copy()
        
        # Get adaptive weights if available
        weighting = self.components.get("adaptive_weighting")
        if weighting:
            try:
                weights = weighting.get_current_weights()
            except Exception:
                pass
        
        # Sentiment signal
        sentiment = self.components.get("sentiment_agent")
        if sentiment:
            try:
                bias, conf = sentiment.get_current_bias(symbol)
                signals["sentiment"] = {"direction": bias, "confidence": conf}
            except Exception as e:
                logger.debug(f"Sentiment signal failed: {e}")
        
        # Microstructure signal
        micro = self.components.get("microstructure_agent")
        if micro:
            try:
                bias, conf = micro.get_microstructure_bias(symbol)
                signals["microstructure"] = {"direction": bias, "confidence": conf}
            except Exception as e:
                logger.debug(f"Microstructure signal failed: {e}")
        
        # OnChain signal (SOL only)
        onchain = self.components.get("onchain_agent")
        if onchain and symbol == "SOL":
            try:
                signal = onchain.get_current_signal()
                if signal:
                    signals["onchain"] = {
                        "direction": signal.direction,
                        "confidence": signal.confidence
                    }
            except Exception as e:
                logger.debug(f"OnChain signal failed: {e}")
        
        # Flow signal (?- whale, ETF, exchange flow)
        flow_integrator = self.components.get("flow_integrator")
        if flow_integrator:
            try:
                # Update flow state
                flow_integrator.update()
                # Get signal for symbol
                direction, confidence = flow_integrator.get_signal_for_symbol(symbol)
                if confidence > 0.1:  # Only include if meaningful
                    signals["flow"] = {
                        "direction": direction,
                        "confidence": confidence
                    }
                    # Log whale alerts
                    state = flow_integrator.get_state()
                    if state.whale_alert:
                        logger.info(f"[FLOW] Whale alert for {symbol}: direction={direction:.2f}")
                    if state.etf_outflow_caution:
                        logger.warning(f"[FLOW] ETF outflow caution active")
            except Exception as e:
                logger.debug(f"Flow signal failed: {e}")
        
        # ===  (v6.1.2-INTEGRATED) ===
        
        # Volatility Alpha signal (?
        vol_alpha = self.components.get("volatility_alpha")
        if vol_alpha:
            try:
                # ?
                vol_intent = vol_alpha.evaluate(asset=symbol)
                if vol_intent and vol_intent.intensity > 0.1:
                    # vol_bias: INCREASE=1, NEUTRAL=0, SUPPRESS=-1
                    bias_map = {"INCREASE": 0.5, "NEUTRAL": 0.0, "SUPPRESS": -0.5}
                    direction = bias_map.get(vol_intent.bias.name, 0.0) * vol_intent.intensity
                    signals["volatility"] = {
                        "direction": direction,
                        "confidence": vol_intent.confidence,
                        "bias": vol_intent.bias.name,
                        "horizon": vol_intent.horizon.name if vol_intent.horizon else "SHORT"
                    }
            except Exception as e:
                logger.debug(f"Volatility alpha signal failed: {e}")
        
        # OnChain Sentiment Fusion signal ()
        onchain_fusion = self.components.get("onchain_fusion")
        if onchain_fusion:
            try:
                # 
                fused_signal = onchain_fusion.generate_signals(symbol)
                if fused_signal and fused_signal.confidence > 0.1:
                    signals["onchain_fusion"] = {
                        "direction": fused_signal.direction,
                        "confidence": fused_signal.confidence,
                        "onchain_score": fused_signal.onchain_score,
                        "sentiment_score": fused_signal.sentiment_score
                    }
            except Exception as e:
                logger.debug(f"OnChain fusion signal failed: {e}")
        
        # Alpha Signal Integrator ( Alpha: trigger/exit/authority)
        alpha_integrator = self.components.get("alpha_integrator")
        if alpha_integrator:
            try:
                # Update alpha state
                market_data_for_alpha = {"symbol": symbol, **market_data}
                alpha_state = alpha_integrator.update(market_data=market_data_for_alpha)
                
                # Check for NO_TRADE block
                should_block, block_reason = alpha_integrator.should_block_trade()
                if should_block:
                    logger.warning(f"[ALPHA] NO_TRADE triggered: {block_reason}")
                    return {
                        "direction": 0.0,
                        "confidence": 0.0,
                        "components": {},
                        "blocked": True,
                        "block_reason": block_reason
                    }
                
                # Check for OPPORTUNITY trigger
                opp_active, opp_type, opp_dir = alpha_integrator.is_opportunity_active()
                if opp_active:
                    logger.info(f"[ALPHA] OPPORTUNITY triggered: {opp_type} direction={opp_dir}")
                
                # Get combined alpha signal
                alpha_direction, alpha_confidence = alpha_integrator.get_trading_signal(symbol)
                if alpha_confidence > 0.1:
                    signals["alpha"] = {
                        "direction": alpha_direction,
                        "confidence": alpha_confidence,
                        "opportunity_active": opp_active,
                        "opportunity_type": opp_type,
                        "vol_bias": alpha_state.vol_bias
                    }
            except Exception as e:
                logger.debug(f"Alpha integrator signal failed: {e}")
        
        # Fuse signals with weights
        if not signals:
            return {"direction": 0.0, "confidence": 0.0, "components": {}}
        
        weighted_direction = 0.0
        total_weight = 0.0
        confidences = []
        
        for agent, sig in signals.items():
            weight = weights.get(agent, 0.1)
            weighted_direction += sig["direction"] * weight * sig["confidence"]
            total_weight += weight * sig["confidence"]
            confidences.append(sig["confidence"])
        
        if total_weight > 0:
            fused_direction = weighted_direction / total_weight
        else:
            fused_direction = 0.0
        
        fused_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "direction": fused_direction,
            "confidence": fused_confidence,
            "components": signals,
            "weights_used": weights
        }
    
    # =========================================================================
    # ALERTS
    # =========================================================================
    
    def _send_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Send alert through alert manager."""
        alert_manager = self.components.get("alert_manager")
        if alert_manager:
            try:
                alert_manager.send(
                    alert_type=alert_type,
                    message=message,
                    severity=severity
                )
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        else:
            # Fallback to console
            logger.warning(f"[ALERT] {alert_type}: {message}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts."""
        alert_manager = self.components.get("alert_manager")
        if alert_manager:
            alert_manager.register_callback(callback)
    
    # =========================================================================
    # EXPLAINABILITY
    # =========================================================================
    
    def explain_decision(self, decision: Dict) -> str:
        """
        Get natural language explanation for a trading decision.
        
        Returns:
            Human-readable explanation string
        """
        explainer = self.components.get("llm_explainer")
        if explainer:
            try:
                return explainer.explain(decision)
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
        
        # Fallback: template-based explanation
        return self._template_explanation(decision)
    
    def _template_explanation(self, decision: Dict) -> str:
        """Generate template-based explanation."""
        parts = []
        
        action = decision.get("action", "hold")
        symbol = decision.get("symbol", "unknown")
        
        if action == "buy":
            parts.append(f"Decision: BUY {symbol}")
        elif action == "sell":
            parts.append(f"Decision: SELL {symbol}")
        else:
            parts.append(f"Decision: HOLD {symbol}")
        
        # Add signal components
        components = decision.get("signal_components", {})
        for agent, sig in components.items():
            direction = "bullish" if sig.get("direction", 0) > 0 else "bearish"
            conf = sig.get("confidence", 0) * 100
            parts.append(f"- {agent}: {direction} ({conf:.0f}% confidence)")
        
        # Add risk flags
        risk_flags = decision.get("risk_flags", [])
        if risk_flags:
            parts.append("Risk considerations:")
            for flag in risk_flags:
                parts.append(f"- {flag}")
        
        return "\n".join(parts)
    
    def ask_question(self, question: str) -> str:
        """
        Interactive Q&A about system state.
        
        Returns:
            Answer to the question
        """
        explainer = self.components.get("llm_explainer")
        if explainer:
            try:
                return explainer.answer_question(question, self.get_system_state())
            except Exception as e:
                logger.error(f"LLM Q&A failed: {e}")
                return f"Failed to answer: {e}"
        
        return "LLM explainer not available"
    
    # =========================================================================
    # SYSTEM STATE
    # =========================================================================
    
    def get_system_state(self) -> Dict:
        """Get comprehensive system state."""
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_running": self.is_running,
            "components_active": list(self.components.keys()),
            "last_decision": self.last_decision,
            "component_status": {}
        }
        
        # Get status from each component
        for name, component in self.components.items():
            try:
                if hasattr(component, "get_status"):
                    state["component_status"][name] = component.get_status()
                else:
                    state["component_status"][name] = {"active": True}
            except Exception as e:
                state["component_status"][name] = {"error": str(e)}
        
        return state
    
    def get_dashboard_data(self) -> Dict:
        """Get data formatted for dashboard display."""
        data = {
            "execution_metrics": {},
            "risk_state": {},
            "agent_performance": {},
            "alerts": [],
            "system_health": {}
        }
        
        # Risk state
        risk_controller = self.components.get("risk_controller")
        if risk_controller:
            try:
                flags = risk_controller.get_dashboard_flags()
                data["risk_state"] = {
                    "flags": flags,
                    "state": risk_controller.risk_state.value if hasattr(risk_controller, "risk_state") else "unknown"
                }
            except Exception:
                pass
        
        # Execution metrics
        scheduler = self.components.get("execution_scheduler")
        if scheduler:
            try:
                data["execution_metrics"] = scheduler.get_metrics()
            except Exception:
                pass
        
        # Agent weights
        weighting = self.components.get("adaptive_weighting")
        if weighting:
            try:
                data["agent_performance"] = weighting.get_performance_summary()
            except Exception:
                pass
        
        return data
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    async def start(self):
        """Start the integration layer."""
        self.is_running = True
        logger.info("SOTAIntegration started")
        
        # Start async components
        scheduler = self.components.get("execution_scheduler")
        if scheduler and hasattr(scheduler, "start"):
            await scheduler.start()
    
    async def stop(self):
        """Stop the integration layer."""
        self.is_running = False
        
        # Stop async components
        scheduler = self.components.get("execution_scheduler")
        if scheduler and hasattr(scheduler, "stop"):
            await scheduler.stop()
        
        logger.info("SOTAIntegration stopped")
    
    def shutdown(self):
        """Clean shutdown of all components."""
        for name, component in self.components.items():
            try:
                if hasattr(component, "shutdown"):
                    component.shutdown()
                logger.debug(f"Shutdown {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown {name}: {e}")
        
        self.components.clear()
        logger.info("SOTAIntegration shutdown complete")


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_sota_integration: Optional[SOTAIntegration] = None


def get_sota_integration(config: SOTAIntegrationConfig = None) -> SOTAIntegration:
    """Get or create SOTA integration singleton."""
    global _sota_integration
    if _sota_integration is None:
        _sota_integration = SOTAIntegration(config)
    elif config is not None and _sota_integration.config != config:
        logger.info("SOTAIntegration config changed; refreshing singleton instance")
        _sota_integration.shutdown()
        _sota_integration = SOTAIntegration(config)
    return _sota_integration


def reset_sota_integration():
    """Reset SOTA integration singleton."""
    global _sota_integration
    if _sota_integration:
        _sota_integration.shutdown()
    _sota_integration = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create integration
    config = SOTAIntegrationConfig(
        enable_sentiment_agent=True,
        enable_microstructure_agent=True
    )
    integration = SOTAIntegration(config)
    
    # Check system state
    state = integration.get_system_state()
    print(f"\nSystem State:")
    print(f"  Components: {state['components_active']}")
    
    # Test risk check
    result = integration.check_trade_allowed(
        symbol="BTC",
        side="buy",
        size=0.5,
        portfolio={"ETH": 0.3, "SOL": 0.2}
    )
    print(f"\nRisk Check: {result}")
    
    # Test signal fusion
    signal = integration.get_fused_signal("BTC", {})
    print(f"\nFused Signal: {signal}")
    
    # Test explanation
    decision = {
        "action": "buy",
        "symbol": "BTC",
        "signal_components": signal.get("components", {})
    }
    explanation = integration.explain_decision(decision)
    print(f"\nExplanation:\n{explanation}")
    
    # Cleanup
    integration.shutdown()

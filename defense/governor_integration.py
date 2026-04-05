"""
================================================================================
HMATS v6.1.3 - Governor Integration Layer
================================================================================
Purpose: 将三个 Long-Term Survival Governors 接入 Canonical Flow

集成的 Governors:
    1. OpportunityBudgetGovernor - OP 机会预算限制
    2. RegimeTransitionBuffer - BEAR -> BULL 过渡保护
    3. CascadeExhaustionGovernor - 清算级联削峰

接入点:
    Strategy Intent / TwoStage Output
        ↓
    GovernorIntegration.check_all()  ← 新增
        ↓
    TradeGate / SafetyIntegrator

所有 veto 都写入 Shadow Ledger，状态写入 RuntimeState

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class GovernorVetoSource(Enum):
    """Veto sources."""
    OPPORTUNITY_BUDGET = "OPPORTUNITY_BUDGET"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    CASCADE_EXHAUSTION = "CASCADE_EXHAUSTION"


@dataclass
class GovernorCheckResult:
    """Combined result from all governors."""
    allowed: bool = True
    vetoed_by: Optional[GovernorVetoSource] = None
    reason: str = "OK"
    modifications: Dict[str, Any] = field(default_factory=dict)
    
    # Individual results
    opportunity_result: Optional[Dict] = None
    transition_result: Optional[Dict] = None
    cascade_result: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "allowed": self.allowed,
            "vetoed_by": self.vetoed_by.value if self.vetoed_by else None,
            "reason": self.reason,
            "modifications": self.modifications,
            "opportunity_result": self.opportunity_result,
            "transition_result": self.transition_result,
            "cascade_result": self.cascade_result,
        }


@dataclass
class GovernorIntegrationConfig:
    """Configuration for governor integration."""
    enable_opportunity_budget: bool = True
    enable_regime_transition: bool = True
    enable_cascade_exhaustion: bool = True
    
    # Shadow ledger integration
    enable_shadow_ledger: bool = True
    
    # RuntimeState integration
    enable_runtime_state: bool = True


class GovernorIntegration:
    """
    Governor Integration Layer
    
    提供单一入口点检查所有 governors，用于 canonical flow 集成。
    """
    
    def __init__(self, config: GovernorIntegrationConfig = None):
        self.config = config or GovernorIntegrationConfig()
        
        # Load flags
        self._load_flags()
        
        # Governor references
        self._opportunity_governor = None
        self._transition_buffer = None
        self._cascade_governor = None
        
        # Initialize governors
        self._init_governors()
        
        # Statistics
        self._total_checks = 0
        self._total_vetoes = 0
        self._veto_breakdown = {
            GovernorVetoSource.OPPORTUNITY_BUDGET: 0,
            GovernorVetoSource.REGIME_TRANSITION: 0,
            GovernorVetoSource.CASCADE_EXHAUSTION: 0,
        }
        
        logger.info("[GOVERNOR] Integration layer initialized")
    
    def _load_flags(self):
        """Load flags from sota_flags."""
        try:
            from configs.sota_flags import get_sota_flags
            flags = get_sota_flags()
            
            self.config.enable_opportunity_budget = getattr(
                flags, 'ENABLE_OPPORTUNITY_BUDGET_GOVERNOR', True
            )
            self.config.enable_regime_transition = getattr(
                flags, 'ENABLE_REGIME_TRANSITION_BUFFER', True
            )
            self.config.enable_cascade_exhaustion = getattr(
                flags, 'ENABLE_CASCADE_EXHAUSTION_GOVERNOR', True
            )
        except ImportError:
            pass
    
    def _init_governors(self):
        """Initialize governor instances."""
        # Opportunity Budget Governor
        if self.config.enable_opportunity_budget:
            try:
                from risk.opportunity_budget_governor import get_opportunity_budget_governor
                self._opportunity_governor = get_opportunity_budget_governor()
                logger.info("[GOVERNOR] ✓ OpportunityBudgetGovernor loaded")
            except ImportError as e:
                logger.warning(f"[GOVERNOR] OpportunityBudgetGovernor not available: {e}")
        
        # Regime Transition Buffer
        if self.config.enable_regime_transition:
            try:
                from risk.regime_transition_buffer import get_regime_transition_buffer
                self._transition_buffer = get_regime_transition_buffer()
                logger.info("[GOVERNOR] ✓ RegimeTransitionBuffer loaded")
            except ImportError as e:
                logger.warning(f"[GOVERNOR] RegimeTransitionBuffer not available: {e}")
        
        # Cascade Exhaustion Governor
        if self.config.enable_cascade_exhaustion:
            try:
                from risk.cascade_exhaustion_governor import get_cascade_exhaustion_governor
                self._cascade_governor = get_cascade_exhaustion_governor()
                logger.info("[GOVERNOR] ✓ CascadeExhaustionGovernor loaded")
            except ImportError as e:
                logger.warning(f"[GOVERNOR] CascadeExhaustionGovernor not available: {e}")
    
    def check_all(
        self,
        symbol: str,
        direction: str,
        intent_type: str = "trade",
        is_opportunity: bool = False,
        opportunity_type: str = "generic",
        is_naked_short: bool = False,
        strategy_name: str = "",
        tranche: int = 1,
        confidence: float = 0.0,
    ) -> GovernorCheckResult:
        """
        Check all governors and return combined result.
        
        Args:
            symbol: Trading symbol (BTC, ETH, SOL)
            direction: Trade direction (long, short)
            intent_type: Type of intent (trade, add, exit)
            is_opportunity: Whether this is an OPPORTUNITY mode trade
            opportunity_type: Type of opportunity signal
            is_naked_short: Whether this is a naked short
            strategy_name: Strategy name (e.g., "kalman", "funding_divergence")
            tranche: Tranche level for position (1-4)
            confidence: Signal confidence (0-1)
            
        Returns:
            GovernorCheckResult with combined decision
        """
        self._total_checks += 1
        
        result = GovernorCheckResult()
        
        # 1. Check Opportunity Budget (only for OPPORTUNITY mode)
        if self._opportunity_governor and is_opportunity:
            op_result = self._opportunity_governor.check(
                opportunity_type=opportunity_type,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
            )
            result.opportunity_result = op_result.to_dict()
            
            if not op_result.allowed:
                result.allowed = False
                result.vetoed_by = GovernorVetoSource.OPPORTUNITY_BUDGET
                result.reason = op_result.reason
                self._record_veto(GovernorVetoSource.OPPORTUNITY_BUDGET)
                return result
        
        # 2. Check Regime Transition Buffer (for shorts)
        if self._transition_buffer and direction == "short":
            trans_result = self._transition_buffer.check(
                intent_type=intent_type,
                symbol=symbol,
                direction=direction,
                is_naked_short=is_naked_short,
                is_opportunity=is_opportunity,
                strategy_name=strategy_name,
            )
            result.transition_result = trans_result.to_dict()
            
            if trans_result.action == "BLOCK":
                result.allowed = False
                result.vetoed_by = GovernorVetoSource.REGIME_TRANSITION
                result.reason = trans_result.reason
                self._record_veto(GovernorVetoSource.REGIME_TRANSITION)
                return result
            
            # Apply modifications
            if trans_result.modifications:
                result.modifications.update(trans_result.modifications)
        
        # 3. Check Cascade Exhaustion (for adding to positions)
        if self._cascade_governor and tranche > 1:
            cascade_result = self._cascade_governor.check_tranche(
                symbol=symbol,
                tranche=tranche,
                direction=direction,
            )
            result.cascade_result = cascade_result.to_dict()
            
            if not cascade_result.allowed:
                result.allowed = False
                result.vetoed_by = GovernorVetoSource.CASCADE_EXHAUSTION
                result.reason = cascade_result.reason
                self._record_veto(GovernorVetoSource.CASCADE_EXHAUSTION)
                return result
            
            # Add recommended action to modifications
            if cascade_result.recommended_action != "PROCEED":
                result.modifications["cascade_action"] = cascade_result.recommended_action
        
        return result
    
    def _record_veto(self, source: GovernorVetoSource):
        """Record veto statistics."""
        self._total_vetoes += 1
        self._veto_breakdown[source] += 1
        logger.warning(f"[GOVERNOR] Veto by {source.value}")
    
    def update_market_data(
        self,
        # Regime transition data
        btc_ma50: float = None,
        btc_ma200: float = None,
        sol_btc_strength_14d: float = None,
        funding_7d_avg: float = None,
        oi_change_pct_7d: float = None,
        liq_change_pct_7d: float = None,
        btc_price_change_7d: float = None,
        # Cascade data
        liquidation_volume_1h: float = None,
        liquidation_volume_4h: float = None,
        price_change_1h_pct: float = None,
        price_change_4h_pct: float = None,
        volume_spike_ratio: float = None,
    ):
        """
        Update market data for all governors.
        
        Call this periodically (e.g., every tick or every minute) to keep
        governors updated with current market conditions.
        """
        # Update Regime Transition Buffer
        if self._transition_buffer:
            self._transition_buffer.update_market_data(
                btc_ma50=btc_ma50,
                btc_ma200=btc_ma200,
                sol_btc_strength_14d=sol_btc_strength_14d,
                funding_7d_avg=funding_7d_avg,
                oi_change_pct_7d=oi_change_pct_7d,
                liq_change_pct_7d=liq_change_pct_7d,
                btc_price_change_7d=btc_price_change_7d,
            )
        
        # Update Cascade Exhaustion Governor
        if self._cascade_governor:
            if liquidation_volume_1h is not None:
                self._cascade_governor.update_metrics(
                    liquidation_volume_1h=liquidation_volume_1h,
                    liquidation_volume_4h=liquidation_volume_4h or 0,
                    price_change_1h_pct=price_change_1h_pct or 0,
                    price_change_4h_pct=price_change_4h_pct or 0,
                    volume_spike_ratio=volume_spike_ratio or 1.0,
                )
    
    def release_opportunity(self, opportunity_id: str) -> bool:
        """Release an opportunity slot."""
        if self._opportunity_governor:
            return self._opportunity_governor.release(opportunity_id)
        return False
    
    def release_opportunities_for_symbol(self, symbol: str) -> int:
        """Release all opportunities for a symbol."""
        if self._opportunity_governor:
            return self._opportunity_governor.release_for_symbol(symbol)
        return 0
    
    def get_status(self) -> Dict:
        """Get combined status from all governors."""
        status = {
            "enabled": True,
            "governors": {
                "opportunity_budget": {
                    "enabled": self._opportunity_governor is not None,
                    "status": self._opportunity_governor.get_status() if self._opportunity_governor else None,
                },
                "regime_transition": {
                    "enabled": self._transition_buffer is not None,
                    "status": self._transition_buffer.get_status() if self._transition_buffer else None,
                },
                "cascade_exhaustion": {
                    "enabled": self._cascade_governor is not None,
                    "status": self._cascade_governor.get_status() if self._cascade_governor else None,
                },
            },
            "statistics": {
                "total_checks": self._total_checks,
                "total_vetoes": self._total_vetoes,
                "veto_rate": self._total_vetoes / self._total_checks if self._total_checks > 0 else 0,
                "veto_breakdown": {k.value: v for k, v in self._veto_breakdown.items()},
            },
        }
        return status
    
    def reset(self):
        """Reset all governors."""
        if self._opportunity_governor:
            self._opportunity_governor.reset()
        if self._transition_buffer:
            self._transition_buffer.reset()
        if self._cascade_governor:
            self._cascade_governor.reset()
        
        self._total_checks = 0
        self._total_vetoes = 0
        self._veto_breakdown = {s: 0 for s in GovernorVetoSource}
        
        logger.info("[GOVERNOR] All governors reset")


# =============================================================================
# SINGLETON
# =============================================================================

_governor_integration: Optional[GovernorIntegration] = None


def get_governor_integration(
    config: GovernorIntegrationConfig = None
) -> GovernorIntegration:
    """Get or create singleton instance."""
    global _governor_integration
    if _governor_integration is None:
        _governor_integration = GovernorIntegration(config)
    elif config is not None and _governor_integration.config != config:
        logger.info("[GOVERNOR] Config changed; refreshing singleton instance")
        _governor_integration.reset()
        _governor_integration = GovernorIntegration(config)
    return _governor_integration


def reset_governor_integration():
    """Reset singleton."""
    global _governor_integration
    if _governor_integration:
        _governor_integration.reset()
    _governor_integration = None


# =============================================================================
# CONVENIENCE FUNCTIONS FOR CANONICAL FLOW
# =============================================================================

def check_governors(
    symbol: str,
    direction: str,
    intent_type: str = "trade",
    is_opportunity: bool = False,
    opportunity_type: str = "generic",
    is_naked_short: bool = False,
    strategy_name: str = "",
    tranche: int = 1,
    confidence: float = 0.0,
) -> Tuple[bool, str, Dict]:
    """
    Convenience function for canonical flow integration.
    
    Returns:
        Tuple of (allowed, reason, modifications)
    """
    integration = get_governor_integration()
    result = integration.check_all(
        symbol=symbol,
        direction=direction,
        intent_type=intent_type,
        is_opportunity=is_opportunity,
        opportunity_type=opportunity_type,
        is_naked_short=is_naked_short,
        strategy_name=strategy_name,
        tranche=tranche,
        confidence=confidence,
    )
    return result.allowed, result.reason, result.modifications


def update_governor_market_data(**kwargs):
    """Update market data for governors."""
    integration = get_governor_integration()
    integration.update_market_data(**kwargs)


__all__ = [
    "GovernorVetoSource",
    "GovernorCheckResult",
    "GovernorIntegrationConfig",
    "GovernorIntegration",
    "get_governor_integration",
    "reset_governor_integration",
    "check_governors",
    "update_governor_market_data",
]

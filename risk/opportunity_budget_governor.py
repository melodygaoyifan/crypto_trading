"""
================================================================================
HMATS v6.1.3 - Opportunity Budget Governor
================================================================================
Purpose: 限制同时活跃的 OPPORTUNITY 机会数量，防止极端行情中隐含高相关性敞口

背景:
    当前 OPPORTUNITY 模式允许多个高胜率机会同时激活，
    在极端行情中会造成隐含高相关性敞口。

功能:
    1. 全局机会预算闸门
    2. 限制同时活跃的 OP 机会数量
    3. 被拒绝的机会写入 Shadow Ledger
    4. 当前 budget 使用情况写入 RuntimeState

接线点:
    Strategy Intent / TwoStage Output
        ↓
    OpportunityBudgetGovernor.check()
        ↓
    TradeGate / SafetyIntegrator

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Opportunity signal types with budget costs."""
    LIQUIDATION_CASCADE = "liquidation_cascade"
    FUNDING_DIVERGENCE = "funding_divergence"
    STRUCTURAL_BREAKOUT = "structural_breakout"
    ONCHAIN_FLOW = "onchain_flow"
    LEAD_LAG_EDGE = "lead_lag_edge"
    CRACK_WINDOW = "crack_window"
    VOLATILITY_EXPANSION = "volatility_expansion"
    SENTIMENT_SHOCK = "sentiment_shock"
    SOL_FLOW_SURGE = "sol_flow_surge"
    GENERIC = "generic"


# Budget costs per opportunity type
OPPORTUNITY_BUDGET_COSTS: Dict[OpportunityType, float] = {
    OpportunityType.LIQUIDATION_CASCADE: 0.5,
    OpportunityType.FUNDING_DIVERGENCE: 0.4,
    OpportunityType.STRUCTURAL_BREAKOUT: 0.6,
    OpportunityType.ONCHAIN_FLOW: 0.5,
    OpportunityType.LEAD_LAG_EDGE: 0.4,
    OpportunityType.CRACK_WINDOW: 0.5,
    OpportunityType.VOLATILITY_EXPANSION: 0.5,
    OpportunityType.SENTIMENT_SHOCK: 0.4,
    OpportunityType.SOL_FLOW_SURGE: 0.5,
    OpportunityType.GENERIC: 0.3,
}

# Global budget limit
GLOBAL_OPPORTUNITY_BUDGET = 1.0


@dataclass
class OpportunitySlot:
    """Active opportunity slot."""
    opportunity_id: str
    opportunity_type: OpportunityType
    symbol: str
    budget_cost: float
    activated_at: datetime
    ttl_hours: float = 4.0  # Default 4 hours
    direction: str = "none"
    confidence: float = 0.0
    
    @property
    def expires_at(self) -> datetime:
        return self.activated_at + timedelta(hours=self.ttl_hours)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict:
        return {
            "opportunity_id": self.opportunity_id,
            "opportunity_type": self.opportunity_type.value,
            "symbol": self.symbol,
            "budget_cost": self.budget_cost,
            "activated_at": self.activated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "direction": self.direction,
            "confidence": self.confidence,
            "is_expired": self.is_expired,
        }


@dataclass
class BudgetCheckResult:
    """Result of budget check."""
    allowed: bool
    reason: str
    used_budget: float
    available_budget: float
    requesting_cost: float
    active_opportunities: int
    
    def to_dict(self) -> Dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "used_budget": self.used_budget,
            "available_budget": self.available_budget,
            "requesting_cost": self.requesting_cost,
            "active_opportunities": self.active_opportunities,
        }


@dataclass
class OpportunityBudgetConfig:
    """Configuration for budget governor."""
    global_budget: float = GLOBAL_OPPORTUNITY_BUDGET
    budget_costs: Dict[str, float] = field(default_factory=lambda: {
        k.value: v for k, v in OPPORTUNITY_BUDGET_COSTS.items()
    })
    default_ttl_hours: float = 4.0
    max_opportunities_per_symbol: int = 1
    enable_shadow_ledger: bool = True
    enable_runtime_state: bool = True


class OpportunityBudgetGovernor:
    """
    全局机会预算闘门 - 限制同时活跃的 OPPORTUNITY 机会数量
    
    功能:
    1. 维护活跃机会列表
    2. 计算当前 budget 使用
    3. 检查新机会是否允许
    4. 自动清理过期机会
    5. 写入 Shadow Ledger 和 RuntimeState
    """
    
    def __init__(self, config: OpportunityBudgetConfig = None):
        self.config = config or OpportunityBudgetConfig()
        
        # Active opportunities
        self._active_slots: Dict[str, OpportunitySlot] = {}
        
        # Statistics
        self._total_allowed = 0
        self._total_rejected = 0
        self._rejection_log: List[Dict] = []
        
        # Shadow ledger reference
        self._shadow_ledger = None
        self._runtime_state = None
        
        self._init_integrations()
        
        logger.info(f"[OP_BUDGET] Governor initialized | budget={self.config.global_budget}")
    
    def _init_integrations(self):
        """Initialize integrations with Shadow Ledger and RuntimeState."""
        if self.config.enable_shadow_ledger:
            try:
                from data_mgmt.shadow_ledger_manager import get_shadow_ledger
                self._shadow_ledger = get_shadow_ledger()
                logger.info("[OP_BUDGET] Shadow Ledger connected")
            except ImportError:
                logger.debug("[OP_BUDGET] Shadow Ledger not available")
            except Exception as e:
                logger.warning(f"[OP_BUDGET] Shadow Ledger init failed: {e}")
        
        if self.config.enable_runtime_state:
            try:
                from core.runtime_state import get_runtime_state
                self._runtime_state = get_runtime_state()
                logger.info("[OP_BUDGET] RuntimeState connected")
            except ImportError:
                logger.debug("[OP_BUDGET] RuntimeState not available")
            except Exception as e:
                logger.warning(f"[OP_BUDGET] RuntimeState init failed: {e}")
    
    def _cleanup_expired(self):
        """Remove expired opportunity slots."""
        now = datetime.now(timezone.utc)
        expired_ids = [
            oid for oid, slot in self._active_slots.items()
            if slot.is_expired
        ]
        
        for oid in expired_ids:
            slot = self._active_slots.pop(oid)
            logger.debug(f"[OP_BUDGET] Expired: {slot.opportunity_type.value} for {slot.symbol}")
    
    def _get_used_budget(self) -> float:
        """Calculate current budget usage."""
        self._cleanup_expired()
        return sum(slot.budget_cost for slot in self._active_slots.values())
    
    def _count_symbol_opportunities(self, symbol: str) -> int:
        """Count active opportunities for a symbol."""
        return sum(1 for slot in self._active_slots.values() if slot.symbol == symbol)
    
    def _get_budget_cost(self, opportunity_type: OpportunityType) -> float:
        """Get budget cost for opportunity type."""
        type_key = opportunity_type.value
        return self.config.budget_costs.get(
            type_key, 
            OPPORTUNITY_BUDGET_COSTS.get(opportunity_type, 0.3)
        )
    
    def check(
        self,
        opportunity_type: str,
        symbol: str,
        direction: str = "none",
        confidence: float = 0.0,
        ttl_hours: float = None,
    ) -> BudgetCheckResult:
        """
        Check if a new opportunity is allowed within budget.
        
        Args:
            opportunity_type: Type of opportunity (string or OpportunityType)
            symbol: Trading symbol (BTC, ETH, SOL)
            direction: Trade direction (long, short, none)
            confidence: Signal confidence (0-1)
            ttl_hours: Time-to-live in hours
            
        Returns:
            BudgetCheckResult with allowed status and details
        """
        # Parse opportunity type
        try:
            op_type = OpportunityType(opportunity_type) if isinstance(opportunity_type, str) else opportunity_type
        except ValueError:
            op_type = OpportunityType.GENERIC
        
        # Get budget cost
        budget_cost = self._get_budget_cost(op_type)
        
        # Calculate current usage
        used_budget = self._get_used_budget()
        available_budget = self.config.global_budget - used_budget
        
        # Check per-symbol limit
        symbol_count = self._count_symbol_opportunities(symbol)
        
        # Default result
        result = BudgetCheckResult(
            allowed=True,
            reason="OK",
            used_budget=used_budget,
            available_budget=available_budget,
            requesting_cost=budget_cost,
            active_opportunities=len(self._active_slots),
        )
        
        # Check budget limit
        if budget_cost > available_budget:
            result.allowed = False
            result.reason = f"OPPORTUNITY_BUDGET_EXCEEDED: need {budget_cost:.2f}, have {available_budget:.2f}"
            self._handle_rejection(op_type, symbol, direction, result)
            return result
        
        # Check per-symbol limit
        if symbol_count >= self.config.max_opportunities_per_symbol:
            result.allowed = False
            result.reason = f"MAX_OPPORTUNITIES_PER_SYMBOL: {symbol} already has {symbol_count}"
            self._handle_rejection(op_type, symbol, direction, result)
            return result
        
        # Allowed - activate the slot
        opportunity_id = f"{op_type.value}_{symbol}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
        
        slot = OpportunitySlot(
            opportunity_id=opportunity_id,
            opportunity_type=op_type,
            symbol=symbol,
            budget_cost=budget_cost,
            activated_at=datetime.now(timezone.utc),
            ttl_hours=ttl_hours or self.config.default_ttl_hours,
            direction=direction,
            confidence=confidence,
        )
        
        self._active_slots[opportunity_id] = slot
        self._total_allowed += 1
        
        logger.info(
            f"[OP_BUDGET] ALLOWED: {op_type.value} for {symbol} | "
            f"cost={budget_cost:.2f} | used={used_budget + budget_cost:.2f}/{self.config.global_budget}"
        )
        
        # Update runtime state
        self._update_runtime_state()
        
        return result
    
    def _handle_rejection(
        self,
        op_type: OpportunityType,
        symbol: str,
        direction: str,
        result: BudgetCheckResult,
    ):
        """Handle rejected opportunity."""
        self._total_rejected += 1
        
        rejection_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "opportunity_type": op_type.value,
            "symbol": symbol,
            "direction": direction,
            "reason": result.reason,
            "used_budget": result.used_budget,
            "requesting_cost": result.requesting_cost,
        }
        
        self._rejection_log.append(rejection_record)
        
        # Keep only last 100 rejections
        if len(self._rejection_log) > 100:
            self._rejection_log = self._rejection_log[-100:]
        
        logger.warning(f"[OP_BUDGET] REJECTED: {op_type.value} for {symbol} | {result.reason}")
        
        # Write to Shadow Ledger
        if self._shadow_ledger:
            try:
                self._shadow_ledger.log_veto(
                    source="OpportunityBudgetGovernor",
                    reason="OPPORTUNITY_BUDGET_EXCEEDED",
                    details=rejection_record,
                )
            except Exception as e:
                # [P96 2026-04-26] Promoted DEBUG → WARNING. Same P64
                # silent-failure pattern as P92 exit_drl_promotion_gate.
                # Operator running at INFO level previously couldn't see
                # that the audit trail wasn't being written.
                logger.warning(
                    f"[OP_BUDGET] Shadow ledger write FAILED "
                    f"({type(e).__name__}: {e}); rejection record NOT "
                    f"persisted to audit trail."
                )
        
        # Update runtime state
        self._update_runtime_state()
    
    def _update_runtime_state(self):
        """Update RuntimeState with current budget status."""
        if self._runtime_state:
            try:
                state = self.get_status()
                self._runtime_state.set("opportunity_budget", state)
            except Exception as e:
                logger.debug(f"[OP_BUDGET] RuntimeState update failed: {e}")
    
    def release(self, opportunity_id: str) -> bool:
        """
        Manually release an opportunity slot.
        
        Args:
            opportunity_id: ID of opportunity to release
            
        Returns:
            True if released, False if not found
        """
        if opportunity_id in self._active_slots:
            slot = self._active_slots.pop(opportunity_id)
            logger.info(f"[OP_BUDGET] Released: {slot.opportunity_type.value} for {slot.symbol}")
            self._update_runtime_state()
            return True
        return False
    
    def release_for_symbol(self, symbol: str) -> int:
        """
        Release all opportunities for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Number of released slots
        """
        to_release = [
            oid for oid, slot in self._active_slots.items()
            if slot.symbol == symbol
        ]
        
        for oid in to_release:
            self._active_slots.pop(oid)
        
        if to_release:
            logger.info(f"[OP_BUDGET] Released {len(to_release)} opportunities for {symbol}")
            self._update_runtime_state()
        
        return len(to_release)
    
    def get_status(self) -> Dict:
        """Get current budget status."""
        self._cleanup_expired()
        
        used_budget = self._get_used_budget()
        
        return {
            "enabled": True,
            "global_budget": self.config.global_budget,
            "used_budget": used_budget,
            "available_budget": self.config.global_budget - used_budget,
            "utilization_pct": (used_budget / self.config.global_budget) * 100 if self.config.global_budget > 0 else 0,
            "active_opportunities": len(self._active_slots),
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
            "active_slots": [slot.to_dict() for slot in self._active_slots.values()],
            "recent_rejections": self._rejection_log[-10:],
        }
    
    def reset(self):
        """Reset all state (for testing)."""
        self._active_slots.clear()
        self._total_allowed = 0
        self._total_rejected = 0
        self._rejection_log.clear()
        logger.info("[OP_BUDGET] Governor reset")

    def to_dict(self) -> Dict:
        """Serialize governor state for restart persistence."""
        return {
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
            "active_slots": {
                k: slot.to_dict() for k, slot in self._active_slots.items()
            },
        }

    def from_dict(self, data: Dict):
        """Restore governor state from persistence."""
        self._total_allowed = data.get("total_allowed", 0)
        self._total_rejected = data.get("total_rejected", 0)
        # Note: active_slots not restored (they expire quickly anyway)
        logger.info(
            f"[OP_BUDGET] State restored: allowed={self._total_allowed}, "
            f"rejected={self._total_rejected}"
        )


# =============================================================================
# SINGLETON
# =============================================================================

_opportunity_budget_governor: Optional[OpportunityBudgetGovernor] = None


def get_opportunity_budget_governor(
    config: OpportunityBudgetConfig = None
) -> OpportunityBudgetGovernor:
    """Get or create singleton instance."""
    global _opportunity_budget_governor
    if _opportunity_budget_governor is None:
        _opportunity_budget_governor = OpportunityBudgetGovernor(config)
    elif config is not None and _opportunity_budget_governor.config != config:
        logger.info("[OPP_BUDGET] Config changed; refreshing singleton instance")
        _opportunity_budget_governor = OpportunityBudgetGovernor(config)
    return _opportunity_budget_governor


def reset_opportunity_budget_governor():
    """Reset singleton."""
    global _opportunity_budget_governor
    if _opportunity_budget_governor:
        _opportunity_budget_governor.reset()
    _opportunity_budget_governor = None


# =============================================================================
# CONVENIENCE FUNCTION FOR CANONICAL FLOW
# =============================================================================

def check_opportunity_budget(
    opportunity_type: str,
    symbol: str,
    direction: str = "none",
    confidence: float = 0.0,
) -> Tuple[bool, str]:
    """
    Convenience function for canonical flow integration.
    
    Args:
        opportunity_type: Type of opportunity
        symbol: Trading symbol
        direction: Trade direction
        confidence: Signal confidence
        
    Returns:
        Tuple of (allowed, reason)
    """
    # Check if enabled via flags
    try:
        from configs.sota_flags import get_sota_flags
        flags = get_sota_flags()
        if not getattr(flags, 'ENABLE_OPPORTUNITY_BUDGET_GOVERNOR', True):
            return True, "Governor disabled"
    except ImportError:
        pass
    
    governor = get_opportunity_budget_governor()
    result = governor.check(opportunity_type, symbol, direction, confidence)
    return result.allowed, result.reason


__all__ = [
    "OpportunityType",
    "OpportunitySlot",
    "BudgetCheckResult",
    "OpportunityBudgetConfig",
    "OpportunityBudgetGovernor",
    "get_opportunity_budget_governor",
    "reset_opportunity_budget_governor",
    "check_opportunity_budget",
    "OPPORTUNITY_BUDGET_COSTS",
    "GLOBAL_OPPORTUNITY_BUDGET",
]

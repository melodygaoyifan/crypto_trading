"""
================================================================================
HMATS v6.1.3 - Regime Transition Buffer
================================================================================
Purpose: BEAR -> BULL 过渡缓冲层，降低牛市初期 short 风险

背景:
    系统已有 BEAR / BULL / VOLATILE + RegimeShortFilter，
    但缺少 BEAR -> BULL 的过渡缓冲，牛市初期 short 风险偏高。

功能:
    1. 检测 TRANSITION 状态 (任意 2 条触发)
    2. 作为额外 veto / modifier (不修改原 RegimeClassifier)
    3. TRANSITION 行为只影响 aggressiveness

TRANSITION 触发条件 (任意 2 条):
    - BTC 周线 MA50 > MA200
    - SOL/BTC 相对强度 14D > 0
    - Funding 连续 7 天 > 0 且价格未明显回落
    - OI 上升但清算量下降

TRANSITION 行为:
    - 裸 short: OFF 禁止
    - Kalman: 只允许对冲
    - Funding 背离: 权重 × 0.5
    - long_bias: 最低提升到 0.4
    - OP short: OFF 禁止

接线点:
    在 RegimeShortFilter 之前
    只作用于 short / 加仓 / aggressiveness

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class TransitionState(Enum):
    """Regime transition states."""
    INACTIVE = "INACTIVE"           # No transition detected
    POTENTIAL = "POTENTIAL"         # 1 condition met
    ACTIVE = "ACTIVE"               # 2+ conditions met, TRANSITION active
    CONFIRMED = "CONFIRMED"         # Sustained transition (5+ days)


class TransitionCondition(Enum):
    """Conditions that trigger TRANSITION state."""
    MA_GOLDEN_CROSS = "MA_GOLDEN_CROSS"           # BTC MA50 > MA200
    SOL_RELATIVE_STRENGTH = "SOL_RELATIVE_STRENGTH"  # SOL/BTC 14D > 0
    FUNDING_POSITIVE_STREAK = "FUNDING_POSITIVE_STREAK"  # Funding 7D > 0
    OI_UP_LIQ_DOWN = "OI_UP_LIQ_DOWN"             # OI rising, liquidations falling


@dataclass
class ConditionStatus:
    """Status of a single condition."""
    condition: TransitionCondition
    triggered: bool = False
    value: float = 0.0
    threshold: float = 0.0
    last_checked: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "condition": self.condition.value,
            "triggered": self.triggered,
            "value": self.value,
            "threshold": self.threshold,
            "last_checked": self.last_checked.isoformat(),
            "details": self.details,
        }


@dataclass
class TransitionCheckResult:
    """Result of transition buffer check."""
    in_transition: bool
    state: TransitionState
    conditions_met: int
    action: str  # "ALLOW", "BLOCK", "MODIFY"
    reason: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "in_transition": self.in_transition,
            "state": self.state.value,
            "conditions_met": self.conditions_met,
            "action": self.action,
            "reason": self.reason,
            "modifications": self.modifications,
        }


@dataclass
class RegimeTransitionConfig:
    """Configuration for transition buffer."""
    # Condition thresholds
    ma_period_short: int = 50
    ma_period_long: int = 200
    sol_strength_period: int = 14
    funding_streak_days: int = 7
    oi_lookback_days: int = 7
    
    # Trigger threshold
    min_conditions_for_transition: int = 2
    
    # Behavior modifications
    min_long_bias_in_transition: float = 0.4
    funding_divergence_weight_multiplier: float = 0.5
    
    # Duration
    transition_duration_days: int = 14  # Auto-expire after
    confirm_days: int = 5  # Days to confirm sustained transition
    
    # Integration
    enable_shadow_ledger: bool = True
    enable_runtime_state: bool = True


class RegimeTransitionBuffer:
    """
    Regime Transition Buffer - BEAR -> BULL 过渡保护层
    
    功能:
    1. 检测 TRANSITION 状态
    2. 作为 RegimeShortFilter 的前置层
    3. 阻止/修改危险的 short 操作
    4. 写入 Shadow Ledger 和 RuntimeState
    """
    
    def __init__(self, config: RegimeTransitionConfig = None):
        self.config = config or RegimeTransitionConfig()
        
        # State
        self._state = TransitionState.INACTIVE
        self._conditions: Dict[TransitionCondition, ConditionStatus] = {}
        self._transition_start: Optional[datetime] = None
        
        # Market data cache
        self._btc_ma50: float = 0.0
        self._btc_ma200: float = 0.0
        self._sol_btc_strength: float = 0.0
        self._funding_streak: int = 0
        self._oi_change_pct: float = 0.0
        self._liq_change_pct: float = 0.0
        
        # Statistics
        self._blocks_count = 0
        self._modifies_count = 0
        self._allows_count = 0
        self._block_log: List[Dict] = []
        
        # Integrations
        self._shadow_ledger = None
        self._runtime_state = None
        
        self._init_conditions()
        self._init_integrations()
        
        logger.info(f"[TRANSITION] Buffer initialized | min_conditions={self.config.min_conditions_for_transition}")
    
    def _init_conditions(self):
        """Initialize condition trackers."""
        for cond in TransitionCondition:
            self._conditions[cond] = ConditionStatus(condition=cond)
    
    def _init_integrations(self):
        """Initialize integrations."""
        if self.config.enable_shadow_ledger:
            try:
                from data_mgmt.shadow_ledger_manager import get_shadow_ledger
                self._shadow_ledger = get_shadow_ledger()
            except (ImportError, Exception):
                pass
        
        if self.config.enable_runtime_state:
            try:
                from core.runtime_state import get_runtime_state
                self._runtime_state = get_runtime_state()
            except (ImportError, Exception):
                pass
    
    def update_market_data(
        self,
        btc_ma50: float = None,
        btc_ma200: float = None,
        sol_btc_strength_14d: float = None,
        funding_7d_avg: float = None,
        oi_change_pct_7d: float = None,
        liq_change_pct_7d: float = None,
        btc_price_change_7d: float = None,
    ):
        """
        Update market data for condition checking.
        
        Args:
            btc_ma50: BTC weekly MA50
            btc_ma200: BTC weekly MA200
            sol_btc_strength_14d: SOL/BTC relative strength (14D)
            funding_7d_avg: Average funding rate over 7 days
            oi_change_pct_7d: OI change % over 7 days
            liq_change_pct_7d: Liquidation change % over 7 days
            btc_price_change_7d: BTC price change % over 7 days
        """
        now = datetime.now(timezone.utc)
        
        # Update MA data
        if btc_ma50 is not None:
            self._btc_ma50 = btc_ma50
        if btc_ma200 is not None:
            self._btc_ma200 = btc_ma200
        
        # Check MA Golden Cross
        if self._btc_ma50 > 0 and self._btc_ma200 > 0:
            triggered = self._btc_ma50 > self._btc_ma200
            self._conditions[TransitionCondition.MA_GOLDEN_CROSS] = ConditionStatus(
                condition=TransitionCondition.MA_GOLDEN_CROSS,
                triggered=triggered,
                value=self._btc_ma50 / self._btc_ma200 if self._btc_ma200 > 0 else 0,
                threshold=1.0,
                last_checked=now,
                details=f"MA50={self._btc_ma50:.2f}, MA200={self._btc_ma200:.2f}",
            )
        
        # Update SOL/BTC strength
        if sol_btc_strength_14d is not None:
            self._sol_btc_strength = sol_btc_strength_14d
            self._conditions[TransitionCondition.SOL_RELATIVE_STRENGTH] = ConditionStatus(
                condition=TransitionCondition.SOL_RELATIVE_STRENGTH,
                triggered=sol_btc_strength_14d > 0,
                value=sol_btc_strength_14d,
                threshold=0.0,
                last_checked=now,
                details=f"SOL/BTC 14D strength={sol_btc_strength_14d:.4f}",
            )
        
        # Update funding streak
        if funding_7d_avg is not None:
            # Funding positive AND price not crashing
            price_stable = (btc_price_change_7d is None or btc_price_change_7d > -0.1)
            triggered = funding_7d_avg > 0 and price_stable
            self._conditions[TransitionCondition.FUNDING_POSITIVE_STREAK] = ConditionStatus(
                condition=TransitionCondition.FUNDING_POSITIVE_STREAK,
                triggered=triggered,
                value=funding_7d_avg,
                threshold=0.0,
                last_checked=now,
                details=f"Funding 7D avg={funding_7d_avg:.4f}, price_stable={price_stable}",
            )
        
        # Update OI vs Liquidation
        if oi_change_pct_7d is not None and liq_change_pct_7d is not None:
            self._oi_change_pct = oi_change_pct_7d
            self._liq_change_pct = liq_change_pct_7d
            # OI rising but liquidations falling = bullish transition
            triggered = oi_change_pct_7d > 0 and liq_change_pct_7d < 0
            self._conditions[TransitionCondition.OI_UP_LIQ_DOWN] = ConditionStatus(
                condition=TransitionCondition.OI_UP_LIQ_DOWN,
                triggered=triggered,
                value=oi_change_pct_7d - liq_change_pct_7d,
                threshold=0.0,
                last_checked=now,
                details=f"OI change={oi_change_pct_7d:.2%}, Liq change={liq_change_pct_7d:.2%}",
            )
        
        # Update transition state
        self._update_transition_state()
    
    def _update_transition_state(self):
        """Update overall transition state based on conditions."""
        conditions_met = sum(1 for c in self._conditions.values() if c.triggered)
        
        old_state = self._state
        
        if conditions_met >= self.config.min_conditions_for_transition:
            if self._state == TransitionState.INACTIVE:
                self._state = TransitionState.ACTIVE
                self._transition_start = datetime.now(timezone.utc)
                logger.warning(f"[TRANSITION] ACTIVATED | conditions_met={conditions_met}")
            elif self._state == TransitionState.ACTIVE:
                # Check if confirmed
                if self._transition_start:
                    days_active = (datetime.now(timezone.utc) - self._transition_start).days
                    if days_active >= self.config.confirm_days:
                        self._state = TransitionState.CONFIRMED
                        logger.warning(f"[TRANSITION] CONFIRMED | active for {days_active} days")
        elif conditions_met == 1:
            self._state = TransitionState.POTENTIAL
        else:
            if self._state in (TransitionState.ACTIVE, TransitionState.CONFIRMED):
                logger.info("[TRANSITION] Deactivated - conditions no longer met")
            self._state = TransitionState.INACTIVE
            self._transition_start = None
        
        # Check expiration
        if self._transition_start:
            days_active = (datetime.now(timezone.utc) - self._transition_start).days
            if days_active >= self.config.transition_duration_days:
                logger.info(f"[TRANSITION] Expired after {days_active} days")
                self._state = TransitionState.INACTIVE
                self._transition_start = None
        
        # Update runtime state
        self._update_runtime_state()
    
    def check(
        self,
        intent_type: str,
        symbol: str,
        direction: str,
        is_naked_short: bool = False,
        is_opportunity: bool = False,
        strategy_name: str = "",
    ) -> TransitionCheckResult:
        """
        Check if trade intent is allowed/modified during transition.
        
        Args:
            intent_type: Type of trade intent
            symbol: Trading symbol
            direction: "long" or "short"
            is_naked_short: Whether this is a naked short (no hedge)
            is_opportunity: Whether this is an OPPORTUNITY mode trade
            strategy_name: Name of strategy (e.g., "kalman", "funding_divergence")
            
        Returns:
            TransitionCheckResult with action and modifications
        """
        # Default: allow everything
        result = TransitionCheckResult(
            in_transition=self._state in (TransitionState.ACTIVE, TransitionState.CONFIRMED),
            state=self._state,
            conditions_met=sum(1 for c in self._conditions.values() if c.triggered),
            action="ALLOW",
            reason="OK",
        )
        
        # If not in transition, allow everything
        if self._state not in (TransitionState.ACTIVE, TransitionState.CONFIRMED):
            self._allows_count += 1
            return result
        
        # === TRANSITION ACTIVE: Apply restrictions ===
        
        # 1. Block naked shorts
        if direction == "short" and is_naked_short:
            result.action = "BLOCK"
            result.reason = "REGIME_TRANSITION_BLOCK: Naked shorts prohibited during TRANSITION"
            self._handle_block(intent_type, symbol, direction, result.reason)
            return result
        
        # 2. Block OPPORTUNITY shorts
        if direction == "short" and is_opportunity:
            result.action = "BLOCK"
            result.reason = "REGIME_TRANSITION_BLOCK: OP shorts prohibited during TRANSITION"
            self._handle_block(intent_type, symbol, direction, result.reason)
            return result
        
        # 3. Kalman: only allow as hedge
        if strategy_name.lower() == "kalman" and direction == "short":
            # This would need position context to check if it's a hedge
            # For now, we'll mark it for modification
            result.action = "MODIFY"
            result.reason = "TRANSITION: Kalman short restricted to hedge-only"
            result.modifications["require_hedge"] = True
            self._modifies_count += 1
        
        # 4. Funding divergence: reduce weight
        if "funding" in strategy_name.lower() and direction == "short":
            result.action = "MODIFY"
            result.reason = "TRANSITION: Funding divergence weight reduced"
            result.modifications["weight_multiplier"] = self.config.funding_divergence_weight_multiplier
            self._modifies_count += 1
        
        # 5. Enforce minimum long bias
        if direction == "long":
            result.modifications["min_long_bias"] = self.config.min_long_bias_in_transition
        
        self._allows_count += 1
        return result
    
    def _handle_block(self, intent_type: str, symbol: str, direction: str, reason: str):
        """Handle blocked trade."""
        self._blocks_count += 1
        
        block_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent_type": intent_type,
            "symbol": symbol,
            "direction": direction,
            "reason": reason,
            "state": self._state.value,
        }
        
        self._block_log.append(block_record)
        if len(self._block_log) > 100:
            self._block_log = self._block_log[-100:]
        
        logger.warning(f"[TRANSITION] BLOCKED: {direction} {symbol} | {reason}")
        
        # Write to Shadow Ledger
        if self._shadow_ledger:
            try:
                self._shadow_ledger.log_veto(
                    source="RegimeTransitionBuffer",
                    reason="REGIME_TRANSITION_BLOCK",
                    details=block_record,
                )
            except Exception:
                pass
    
    def _update_runtime_state(self):
        """Update RuntimeState."""
        if self._runtime_state:
            try:
                self._runtime_state.set("regime_transition", self.get_status())
            except Exception:
                pass
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            "enabled": True,
            "state": self._state.value,
            "in_transition": self._state in (TransitionState.ACTIVE, TransitionState.CONFIRMED),
            "conditions_met": sum(1 for c in self._conditions.values() if c.triggered),
            "min_conditions": self.config.min_conditions_for_transition,
            "transition_start": self._transition_start.isoformat() if self._transition_start else None,
            "days_active": (datetime.now(timezone.utc) - self._transition_start).days if self._transition_start else 0,
            "conditions": {c.value: s.to_dict() for c, s in self._conditions.items()},
            "statistics": {
                "blocks": self._blocks_count,
                "modifies": self._modifies_count,
                "allows": self._allows_count,
            },
            "recent_blocks": self._block_log[-10:],
        }
    
    def reset(self):
        """Reset state."""
        self._state = TransitionState.INACTIVE
        self._transition_start = None
        self._init_conditions()
        self._blocks_count = 0
        self._modifies_count = 0
        self._allows_count = 0
        self._block_log.clear()
        logger.info("[TRANSITION] Buffer reset")


# =============================================================================
# SINGLETON
# =============================================================================

_regime_transition_buffer: Optional[RegimeTransitionBuffer] = None


def get_regime_transition_buffer(
    config: RegimeTransitionConfig = None
) -> RegimeTransitionBuffer:
    """Get or create singleton instance."""
    global _regime_transition_buffer
    if _regime_transition_buffer is None:
        _regime_transition_buffer = RegimeTransitionBuffer(config)
    elif config is not None and _regime_transition_buffer.config != config:
        logger.info("[REGIME_BUFFER] Config changed; refreshing singleton instance")
        _regime_transition_buffer = RegimeTransitionBuffer(config)
    return _regime_transition_buffer


def reset_regime_transition_buffer():
    """Reset singleton."""
    global _regime_transition_buffer
    if _regime_transition_buffer:
        _regime_transition_buffer.reset()
    _regime_transition_buffer = None


# =============================================================================
# CONVENIENCE FUNCTION FOR CANONICAL FLOW
# =============================================================================

def check_regime_transition(
    intent_type: str,
    symbol: str,
    direction: str,
    is_naked_short: bool = False,
    is_opportunity: bool = False,
    strategy_name: str = "",
) -> Tuple[bool, str, Dict]:
    """
    Convenience function for canonical flow integration.
    
    Returns:
        Tuple of (allowed, reason, modifications)
    """
    # Check if enabled via flags
    try:
        from configs.sota_flags import get_sota_flags
        flags = get_sota_flags()
        if not getattr(flags, 'ENABLE_REGIME_TRANSITION_BUFFER', True):
            return True, "Buffer disabled", {}
    except ImportError:
        pass
    
    buffer = get_regime_transition_buffer()
    result = buffer.check(
        intent_type=intent_type,
        symbol=symbol,
        direction=direction,
        is_naked_short=is_naked_short,
        is_opportunity=is_opportunity,
        strategy_name=strategy_name,
    )
    
    allowed = result.action != "BLOCK"
    return allowed, result.reason, result.modifications


__all__ = [
    "TransitionState",
    "TransitionCondition",
    "ConditionStatus",
    "TransitionCheckResult",
    "RegimeTransitionConfig",
    "RegimeTransitionBuffer",
    "get_regime_transition_buffer",
    "reset_regime_transition_buffer",
    "check_regime_transition",
]

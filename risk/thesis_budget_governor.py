"""
HMATS v6.4 - Thesis Budget Governor
====================================

PURPOSE: Prevent repeated re-entry on the same thesis after consecutive losses.
This is the "anti-self-convincing mechanism" for high-risk systems.

DESIGN:
- thesis_key = f"{symbol}:{side}:{strategy_id}"
- Tracks realized_pnl, loss_streak, cooldown per thesis
- Returns VETO if budget exhausted or in cooldown

FAIL-CLOSED: Any exception -> veto (no-trade)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
from enum import Enum, auto

logger = logging.getLogger("ThesisBudgetGovernor")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ThesisBudgetConfig:
    """Configuration for thesis budget tracking."""
    
    # Master switch
    enabled: bool = True
    
    # Budget limits
    loss_budget_pct_nav: float = 0.008  # 0.8% NAV per thesis
    max_reentry_after_budget_hit: int = 0  # No re-entry after budget exhausted
    
    # Cooldown
    cooldown_bars: int = 6  # 6 * 4h = 24h
    bar_duration_seconds: int = 14400  # 4 hours
    
    # Loss streak
    loss_streak_limit: int = 3  # Max consecutive losses before cooldown
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ThesisBudgetConfig':
        """Load from config dict with safe defaults."""
        thesis_budget = data.get("thesis_budget", {})
        return cls(
            enabled=thesis_budget.get("enabled", True),
            loss_budget_pct_nav=thesis_budget.get("loss_budget_pct_nav", 0.008),
            max_reentry_after_budget_hit=thesis_budget.get("max_reentry_after_budget_hit", 0),
            cooldown_bars=thesis_budget.get("cooldown_bars", 6),
            bar_duration_seconds=thesis_budget.get("bar_duration_seconds", 14400),
            loss_streak_limit=thesis_budget.get("loss_streak_limit", 3),
        )


# =============================================================================
# VETO RESULT
# =============================================================================

class ThesisBudgetVetoReason(Enum):
    """Reasons for thesis budget veto."""
    NONE = auto()
    BUDGET_EXHAUSTED = "THESIS_BUDGET_HIT"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    LOSS_STREAK_LIMIT = "LOSS_STREAK_LIMIT"
    REENTRY_USED_UP = "REENTRY_USED_UP"


@dataclass
class ThesisBudgetResult:
    """Result of thesis budget evaluation."""
    allow: bool
    veto_reason: ThesisBudgetVetoReason = ThesisBudgetVetoReason.NONE
    veto_message: str = ""
    thesis_key: str = ""
    current_pnl: float = 0.0
    budget_remaining: float = 0.0
    loss_streak: int = 0
    cooldown_remaining_seconds: float = 0.0
    # P1-02: Re-entry tracking
    reentry_used: int = 0
    reentry_max: int = 0
    reentry_size_multiplier: float = 1.0  # <1.0 when in reentry mode
    # P1-02: Config snapshot for proof log
    config_loss_budget_pct_nav: float = 0.0
    config_loss_streak_limit: int = 0
    config_cooldown_bars: int = 0
    config_max_reentry: int = 0


# =============================================================================
# THESIS STATE TRACKING
# =============================================================================

@dataclass
class ThesisState:
    """State tracking for a single thesis."""
    thesis_key: str
    realized_pnl: float = 0.0  # Cumulative realized PnL
    loss_streak: int = 0       # Consecutive losses
    trade_count: int = 0       # Total trades
    reentry_count: int = 0     # Re-entries after budget hit
    cooldown_until_ts: float = 0.0  # Cooldown expiry timestamp
    budget_hit_ts: float = 0.0      # When budget was first exhausted
    last_trade_ts: float = 0.0      # Last trade timestamp
    
    def to_dict(self) -> Dict:
        return {
            "thesis_key": self.thesis_key,
            "realized_pnl": self.realized_pnl,
            "loss_streak": self.loss_streak,
            "trade_count": self.trade_count,
            "reentry_count": self.reentry_count,
            "cooldown_until_ts": self.cooldown_until_ts,
            "budget_hit_ts": self.budget_hit_ts,
            "last_trade_ts": self.last_trade_ts,
        }


# =============================================================================
# GOVERNOR IMPLEMENTATION
# =============================================================================

class ThesisBudgetGovernor:
    """
    Governor that tracks and enforces thesis-level loss budgets.
    
    CRITICAL: This is FAIL-CLOSED. Any exception -> veto.
    """

    def __init__(self, config: ThesisBudgetConfig = None, nav: float = 10_000.0):  # [P0-FIX] was 100_000.0
        """
        Initialize governor.

        Args:
            config: Configuration (uses defaults if None)
            nav: Net Asset Value for budget calculation
        """
        self.config = config or ThesisBudgetConfig()
        self.nav = nav
        
        # In-memory state tracking
        self._thesis_states: Dict[str, ThesisState] = {}
        
        # Calculated budget per thesis
        self._budget_per_thesis = self.nav * self.config.loss_budget_pct_nav
        
        logger.info(
            f"[THESIS_BUDGET] Initialized: enabled={self.config.enabled}, "
            f"budget_per_thesis=${self._budget_per_thesis:.2f} "
            f"({self.config.loss_budget_pct_nav:.2%} of ${self.nav:,.0f})"
        )
    
    def update_nav(self, nav: float):
        """Update NAV and recalculate budgets."""
        self.nav = nav
        self._budget_per_thesis = self.nav * self.config.loss_budget_pct_nav
    
    @staticmethod
    def build_thesis_key(
        symbol: str,
        side: str,
        strategy_id: str = None,
        intent_source: str = None
    ) -> str:
        """
        Build thesis key from trade parameters.
        
        Args:
            symbol: Trading symbol (e.g., "SOL")
            side: "long" or "short"
            strategy_id: Strategy identifier (preferred)
            intent_source: Fallback source name
            
        Returns:
            Thesis key string
        """
        strategy = strategy_id or intent_source or "unknown"
        return f"{symbol.upper()}:{side.lower()}:{strategy}"
    
    def evaluate(
        self,
        intent: Any,
        context: Dict[str, Any] = None
    ) -> ThesisBudgetResult:
        """
        Evaluate whether a trade intent should be allowed.
        
        CRITICAL: FAIL-CLOSED - exceptions return veto.
        
        Args:
            intent: TradeIntentV36 or similar
            context: Additional context (nav, position_state, etc.)
            
        Returns:
            ThesisBudgetResult with allow/veto decision
        """
        try:
            # Disabled -> allow
            if not self.config.enabled:
                return ThesisBudgetResult(allow=True)
            
            # Extract intent fields safely
            symbol = getattr(intent, 'asset', 'UNKNOWN')
            direction = getattr(intent, 'direction', 0.0)
            strategy_id = getattr(intent, 'quant_strategy_id', None)
            intent_source = getattr(intent, 'source', None)
            
            # No direction -> no trade -> allow (nothing to budget)
            if abs(direction) < 0.01:
                return ThesisBudgetResult(allow=True)
            
            # Determine side
            side = "long" if direction > 0 else "short"
            
            # Build thesis key
            thesis_key = self.build_thesis_key(
                symbol=symbol,
                side=side,
                strategy_id=strategy_id,
                intent_source=intent_source
            )
            
            # Get or create state
            state = self._get_or_create_state(thesis_key)
            
            # Update NAV from context if provided
            if context and 'nav' in context:
                self.update_nav(context['nav'])
            
            # Calculate budget remaining
            budget_remaining = self._budget_per_thesis + state.realized_pnl  # pnl is negative for losses
            
            now = time.time()

            # Common config snapshot for all results (P1-02)
            _cfg_snap = dict(
                config_loss_budget_pct_nav=self.config.loss_budget_pct_nav,
                config_loss_streak_limit=self.config.loss_streak_limit,
                config_cooldown_bars=self.config.cooldown_bars,
                config_max_reentry=self.config.max_reentry_after_budget_hit,
                reentry_used=state.reentry_count,
                reentry_max=self.config.max_reentry_after_budget_hit,
            )

            # Check 1: Cooldown active?
            if state.cooldown_until_ts > now:
                remaining = state.cooldown_until_ts - now
                logger.warning(
                    f"[THESIS_BUDGET] VETO: {thesis_key} in cooldown "
                    f"({remaining/3600:.1f}h remaining)"
                )
                return ThesisBudgetResult(
                    allow=False,
                    veto_reason=ThesisBudgetVetoReason.COOLDOWN_ACTIVE,
                    veto_message=f"Thesis {thesis_key} in cooldown ({remaining/3600:.1f}h remaining)",
                    thesis_key=thesis_key,
                    current_pnl=state.realized_pnl,
                    budget_remaining=budget_remaining,
                    loss_streak=state.loss_streak,
                    cooldown_remaining_seconds=remaining,
                    **_cfg_snap,
                )

            # Check 2: Budget exhausted?
            if budget_remaining <= 0:
                max_re = self.config.max_reentry_after_budget_hit
                if max_re > 0 and state.reentry_count < max_re:
                    # P1-02: Allow re-entry with reduced size
                    state.reentry_count += 1
                    _mult = 0.5  # conservative re-entry
                    logger.info(
                        f"[THESIS_BUDGET] REENTRY #{state.reentry_count}/{max_re}: "
                        f"{thesis_key} budget exhausted but reentry allowed "
                        f"(size_mult={_mult})"
                    )
                    return ThesisBudgetResult(
                        allow=True,
                        thesis_key=thesis_key,
                        current_pnl=state.realized_pnl,
                        budget_remaining=0.0,
                        loss_streak=state.loss_streak,
                        reentry_size_multiplier=_mult,
                        **{**_cfg_snap, "reentry_used": state.reentry_count},
                    )
                else:
                    # No re-entries left (or max_reentry=0)
                    _reason = (
                        ThesisBudgetVetoReason.REENTRY_USED_UP
                        if max_re > 0
                        else ThesisBudgetVetoReason.BUDGET_EXHAUSTED
                    )
                    logger.warning(
                        f"[THESIS_BUDGET] VETO: {thesis_key} budget exhausted "
                        f"(pnl=${state.realized_pnl:.2f}, reentry={state.reentry_count}/{max_re})"
                    )
                    return ThesisBudgetResult(
                        allow=False,
                        veto_reason=_reason,
                        veto_message=(
                            f"Thesis {thesis_key} budget exhausted "
                            f"(loss=${-state.realized_pnl:.2f}, reentry={state.reentry_count}/{max_re})"
                        ),
                        thesis_key=thesis_key,
                        current_pnl=state.realized_pnl,
                        budget_remaining=0.0,
                        loss_streak=state.loss_streak,
                        **_cfg_snap,
                    )

            # Check 3: Loss streak limit?
            if state.loss_streak >= self.config.loss_streak_limit:
                # Trigger cooldown
                cooldown_seconds = self.config.cooldown_bars * self.config.bar_duration_seconds
                state.cooldown_until_ts = now + cooldown_seconds
                logger.warning(
                    f"[THESIS_BUDGET] VETO: {thesis_key} loss streak limit "
                    f"({state.loss_streak} consecutive losses) -> {cooldown_seconds/3600:.1f}h cooldown"
                )
                return ThesisBudgetResult(
                    allow=False,
                    veto_reason=ThesisBudgetVetoReason.LOSS_STREAK_LIMIT,
                    veto_message=f"Thesis {thesis_key} hit {state.loss_streak} consecutive losses",
                    thesis_key=thesis_key,
                    current_pnl=state.realized_pnl,
                    budget_remaining=budget_remaining,
                    loss_streak=state.loss_streak,
                    cooldown_remaining_seconds=cooldown_seconds,
                    **_cfg_snap,
                )

            # All checks passed -> allow
            logger.debug(
                f"[THESIS_BUDGET] ALLOW: {thesis_key} "
                f"(budget_remaining=${budget_remaining:.2f}, streak={state.loss_streak})"
            )
            return ThesisBudgetResult(
                allow=True,
                thesis_key=thesis_key,
                current_pnl=state.realized_pnl,
                budget_remaining=budget_remaining,
                loss_streak=state.loss_streak,
                **_cfg_snap,
            )
            
        except Exception as e:
            # FAIL-CLOSED: Exception -> veto
            logger.error(f"[THESIS_BUDGET] Exception during evaluation: {e} -> VETO")
            return ThesisBudgetResult(
                allow=False,
                veto_reason=ThesisBudgetVetoReason.BUDGET_EXHAUSTED,
                veto_message=f"Evaluation error: {e}",
            )
    
    def record_fill(
        self,
        thesis_key: str,
        realized_pnl: float,
        is_win: bool = None
    ):
        """
        Record a trade fill and update thesis state.
        
        Args:
            thesis_key: Thesis identifier
            realized_pnl: Realized PnL from this trade (can be negative)
            is_win: Whether trade was profitable (auto-detected if None)
        """
        try:
            state = self._get_or_create_state(thesis_key)
            
            # Update PnL
            old_pnl = state.realized_pnl
            state.realized_pnl += realized_pnl
            state.trade_count += 1
            state.last_trade_ts = time.time()
            
            # Determine win/loss
            if is_win is None:
                is_win = realized_pnl >= 0
            
            # Update loss streak
            if is_win:
                state.loss_streak = 0  # Reset on win
            else:
                state.loss_streak += 1
            
            # Check if budget just hit
            budget_remaining = self._budget_per_thesis + state.realized_pnl
            if budget_remaining <= 0 and state.budget_hit_ts == 0:
                state.budget_hit_ts = time.time()
                logger.warning(
                    f"[THESIS_BUDGET] Budget exhausted for {thesis_key}: "
                    f"pnl=${state.realized_pnl:.2f}"
                )
            
            logger.info(
                f"[THESIS_BUDGET] FILL: {thesis_key} pnl=${realized_pnl:+.2f} "
                f"(total=${state.realized_pnl:.2f}, streak={state.loss_streak})"
            )
            
        except Exception as e:
            logger.error(f"[THESIS_BUDGET] Error recording fill: {e}")
    
    def record_approximate_risk(
        self,
        thesis_key: str,
        risk_amount: float
    ):
        """
        Record approximate risk exposure for budget tracking only.

        NOTE: This only deducts from the thesis budget - it does NOT count
        as a win or loss for loss_streak purposes.  Loss streak is updated
        exclusively by record_fill() when actual realized PnL is known.

        Args:
            thesis_key: Thesis identifier
            risk_amount: Expected risk/loss amount (positive number)
        """
        try:
            state = self._get_or_create_state(thesis_key)
            state.realized_pnl -= abs(risk_amount)
            state.trade_count += 1
            state.last_trade_ts = time.time()
            # loss_streak intentionally NOT touched - only record_fill() changes it
            logger.debug(
                f"[THESIS_BUDGET] Risk exposure recorded for {thesis_key}: "
                f"-${abs(risk_amount):.2f} (budget only, no streak impact)"
            )
        except Exception as e:
            logger.error(f"[THESIS_BUDGET] Error recording risk: {e}")
    
    def _get_or_create_state(self, thesis_key: str) -> ThesisState:
        """Get or create thesis state."""
        if thesis_key not in self._thesis_states:
            self._thesis_states[thesis_key] = ThesisState(thesis_key=thesis_key)
        return self._thesis_states[thesis_key]
    
    def reset(self, thesis_key: str = None):
        """
        Reset thesis state.
        
        Args:
            thesis_key: Specific thesis to reset, or None for all
        """
        if thesis_key:
            if thesis_key in self._thesis_states:
                del self._thesis_states[thesis_key]
                logger.info(f"[THESIS_BUDGET] Reset: {thesis_key}")
        else:
            self._thesis_states.clear()
            logger.info("[THESIS_BUDGET] Reset: ALL theses cleared")
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Get snapshot of all thesis states for debugging/verification.

        Returns:
            Dict with all thesis states and config
        """
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize governor state for persistence."""
        return {
            "config": {
                "enabled": self.config.enabled,
                "loss_budget_pct_nav": self.config.loss_budget_pct_nav,
                "max_reentry_after_budget_hit": self.config.max_reentry_after_budget_hit,
                "cooldown_bars": self.config.cooldown_bars,
                "bar_duration_seconds": self.config.bar_duration_seconds,
                "loss_streak_limit": self.config.loss_streak_limit,
            },
            "nav": self.nav,
            "budget_per_thesis": self._budget_per_thesis,
            "thesis_count": len(self._thesis_states),
            "theses": {
                k: v.to_dict() for k, v in self._thesis_states.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThesisBudgetGovernor':
        """Restore governor from persisted dict."""
        config_data = data.get("config", {})
        config = ThesisBudgetConfig(
            enabled=config_data.get("enabled", True),
            loss_budget_pct_nav=config_data.get("loss_budget_pct_nav", 0.008),
            max_reentry_after_budget_hit=config_data.get("max_reentry_after_budget_hit", 0),
            cooldown_bars=config_data.get("cooldown_bars", 6),
            bar_duration_seconds=config_data.get("bar_duration_seconds", 14400),
            loss_streak_limit=config_data.get("loss_streak_limit", 3),
        )
        nav = data.get("nav", 10_000.0)  # [P0-FIX] was 100_000.0
        gov = cls(config=config, nav=nav)

        for key, thesis_data in data.get("theses", {}).items():
            gov._thesis_states[key] = ThesisState(**thesis_data)

        logger.info(
            f"[THESIS_BUDGET] Restored from persisted state: "
            f"{len(gov._thesis_states)} theses"
        )
        return gov


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_thesis_budget_governor(
    config_dict: Dict = None,
    nav: float = 10_000.0  # [P0-FIX] was 100_000.0
) -> ThesisBudgetGovernor:
    """
    Factory function to create ThesisBudgetGovernor.
    
    Args:
        config_dict: Full config dict containing "thesis_budget" section
        nav: Initial NAV
        
    Returns:
        Configured ThesisBudgetGovernor
    """
    config = ThesisBudgetConfig.from_dict(config_dict or {})
    return ThesisBudgetGovernor(config=config, nav=nav)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test
    gov = create_thesis_budget_governor(nav=10_000.0)  # [P0-FIX] was 100_000.0
    
    # Mock intent
    class MockIntent:
        asset = "SOL"
        direction = 1.0
        quant_strategy_id = "momentum"
        source = "test"
    
    intent = MockIntent()
    
    # Test evaluate
    result = gov.evaluate(intent)
    print(f"Evaluate result: {result}")
    
    # Test record losses
    thesis_key = "SOL:long:momentum"
    gov.record_fill(thesis_key, -200.0)  # Loss
    gov.record_fill(thesis_key, -300.0)  # Loss
    gov.record_fill(thesis_key, -400.0)  # Loss - should hit streak limit
    
    # Test again
    result = gov.evaluate(intent)
    print(f"After losses: {result}")
    
    # Snapshot
    print(f"Snapshot: {gov.snapshot()}")

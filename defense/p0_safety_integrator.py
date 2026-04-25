"""
================================================================================
HMATS v6.1.0 - P0 SAFETY INTEGRATOR
================================================================================
Purpose: Unified integration layer for all P0 safety modules

Modules Integrated:
    1. SOTARiskController - Kill switch, drawdown, correlation collapse
    2. TradeGate - Pre-execution gate (NO_TRADE/EXIT_ONLY/ALLOW)
    3. ExecutionGuards - Order clipping, staleness check, DVOL override
    4. RateLimitManager - API rate limiting for Kraken
    5. HumanOverride - Manual override mechanism (file-based)
    6. ShadowLedger - Audit trail for all intents/orders/fills

Integration Points (in canonical loop):
    1. Pre-tick: human_override.check(), risk_controller.update()
    2. Pre-decide: (none - pure decision)
    3. Post-decide/Pre-execute: trade_gate.check(), execution_guards.clip()
    4. Execute: rate_limit.acquire(), shadow_ledger.record_order()
    5. Post-execute: shadow_ledger.record_fill(), risk_controller.record_pnl()

================================================================================
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS (with fallbacks)
# =============================================================================

try:
    from configs.sota_flags import get_sota_flags, SOTAFlags
except ImportError:
    # Fallback: create minimal flags
    class SOTAFlags:
        ENABLE_SOTA_RISK_CONTROLLER = True
        ENABLE_TRADE_GATE = True
        ENABLE_EXECUTION_GUARDS = True
        ENABLE_RATE_LIMIT_MANAGER = True
        ENABLE_HUMAN_OVERRIDE = True
        ENABLE_SHADOW_LEDGER = True
    def get_sota_flags():
        return SOTAFlags()


# SOTA Risk Controller
SOTA_RISK_AVAILABLE = False
try:
    from risk.sota_risk_controller import (
        SOTARiskController,
        SOTARiskConfig,
        RiskState,
        KillSwitchReason,
    )
    SOTA_RISK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SOTARiskController not available: {e}")


# Trade Gate
TRADE_GATE_AVAILABLE = False
try:
    from defense.trade_gate import (
        TradeGate,
        TradeGateConfig,
        GateDecision,
        GateResult,
        TradeProposal,
    )
    TRADE_GATE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TradeGate not available: {e}")


# Execution Guards
EXECUTION_GUARDS_AVAILABLE = False
try:
    from defense.execution_guards import (
        StaleDataKillSwitch,
        DVOLOverrideController,
        DataFreshness,
    )
    EXECUTION_GUARDS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ExecutionGuards not available: {e}")


# Rate Limit Manager
RATE_LIMIT_AVAILABLE = False
try:
    from defense.rate_limit_manager import (
        KrakenRateLimitManager,
        RateLimitConfig,
    )
    RATE_LIMIT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RateLimitManager not available: {e}")


# Human Override
HUMAN_OVERRIDE_AVAILABLE = False
try:
    from defense.human_override import (
        HumanOverrideProtocol,
        HumanOverride,
        OverrideType,
        OverrideEffects,
    )
    HUMAN_OVERRIDE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HumanOverride not available: {e}")


# Shadow Ledger
SHADOW_LEDGER_AVAILABLE = False
try:
    from defense.shadow_ledger_jsonl import (
        ShadowLedgerWriter,
    )
    SHADOW_LEDGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ShadowLedger not available: {e}")


# =============================================================================
# INTEGRATION RESULT TYPES
# =============================================================================

class P0CheckResult(Enum):
    """Result of P0 safety check."""
    ALLOW = "ALLOW"              # Proceed with trade
    BLOCK_NO_TRADE = "BLOCK_NO_TRADE"  # NO_TRADE mode active
    BLOCK_EXIT_ONLY = "BLOCK_EXIT_ONLY"  # EXIT_ONLY mode (can exit, can't open)
    BLOCK_KILL_SWITCH = "BLOCK_KILL_SWITCH"  # Kill switch active
    BLOCK_HUMAN_OVERRIDE = "BLOCK_HUMAN_OVERRIDE"  # Human intervention
    BLOCK_STALE_DATA = "BLOCK_STALE_DATA"  # Data too old
    BLOCK_RATE_LIMIT = "BLOCK_RATE_LIMIT"  # Rate limited
    CLIPPED = "CLIPPED"          # Size clipped but allowed


@dataclass
class P0SafetyResult:
    """Result from P0 safety check."""
    result: P0CheckResult
    allow_trade: bool = True
    allow_exit: bool = True
    
    # Clipping
    original_size: float = 0.0
    clipped_size: float = 0.0
    
    # Details
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # State snapshot
    kill_switch_active: bool = False
    risk_state: str = "UNKNOWN"
    gate_mode: str = "UNKNOWN"
    human_override_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result.value,
            "allow_trade": self.allow_trade,
            "allow_exit": self.allow_exit,
            "original_size": self.original_size,
            "clipped_size": self.clipped_size,
            "reason": self.reason,
            "details": self.details,
            "kill_switch_active": self.kill_switch_active,
            "risk_state": self.risk_state,
            "gate_mode": self.gate_mode,
            "human_override_active": self.human_override_active,
        }


# =============================================================================
# P0 SAFETY INTEGRATOR
# =============================================================================

class P0SafetyIntegrator:
    """
    Unified P0 Safety Layer
    
    Coordinates all P0 safety modules in the correct order:
    1. Human Override (can override everything)
    2. Kill Switch / Risk State
    3. Trade Gate (NO_TRADE/EXIT_ONLY/ALLOW)
    4. Execution Guards (stale data, DVOL)
    5. Rate Limit
    6. Shadow Ledger (recording)
    
    Usage:
        integrator = P0SafetyIntegrator()
        
        # Pre-tick check
        integrator.pre_tick_update(equity=100000, prices={'BTC': 50000})
        
        # Pre-execution check
        result = integrator.check_pre_execution(
            asset='SOL',
            direction=-1,
            size=0.5,
            is_entry=True,
        )
        
        if result.allow_trade:
            # Execute with clipped size
            execute_order(size=result.clipped_size)
            
            # Record to shadow ledger
            integrator.record_order(...)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flags = get_sota_flags()
        
        # Initialize modules
        self._init_modules()
        
        # State
        self._last_check_result: Optional[P0SafetyResult] = None
        self._tick_count = 0
        
        logger.info("[P0 SAFETY] Integrator initialized")
        self._log_module_status()
    
    def _init_modules(self):
        """Initialize all P0 modules."""
        
        # 1. SOTA Risk Controller
        self.risk_controller: Optional[SOTARiskController] = None
        if self.flags.ENABLE_SOTA_RISK_CONTROLLER and SOTA_RISK_AVAILABLE:
            try:
                config = SOTARiskConfig(
                    kill_switch_drawdown=self.flags.RISK_MAX_DRAWDOWN_PCT,
                    kill_switch_daily_loss=self.flags.RISK_DAILY_LOSS_HALT_PCT,
                    collapse_threshold=self.flags.RISK_CORRELATION_COLLAPSE_THRESHOLD,
                    # [T17] Pass profile thresholds (was using code defaults only)
                    reduce_threshold=self.config.get("reduce_at_drawdown", 0.15),
                    halt_threshold=self.config.get("hard_drawdown_halt", 0.25),
                    critical_threshold=self.config.get("critical_drawdown", 0.35),
                    max_daily_loss=self.config.get("daily_loss_limit", 0.08),
                    spike_threshold=self.config.get("correlation_crisis", 0.98),
                )
                self.risk_controller = SOTARiskController(config)
                logger.info("  ✓ SOTARiskController: ACTIVE")
            except Exception as e:
                logger.error(f"  ✗ SOTARiskController init failed: {e}")
        
        # 2. Trade Gate
        self.trade_gate: Optional[TradeGate] = None
        if self.flags.ENABLE_TRADE_GATE and TRADE_GATE_AVAILABLE:
            try:
                self.trade_gate = TradeGate(TradeGateConfig())
                logger.info("  ✓ TradeGate: ACTIVE")
            except Exception as e:
                logger.error(f"  ✗ TradeGate init failed: {e}")
        
        # 3. Execution Guards
        self.stale_guard: Optional[StaleDataKillSwitch] = None
        self.dvol_controller: Optional[DVOLOverrideController] = None
        if self.flags.ENABLE_EXECUTION_GUARDS and EXECUTION_GUARDS_AVAILABLE:
            try:
                self.stale_guard = StaleDataKillSwitch()
                self.dvol_controller = DVOLOverrideController()
                logger.info("  ✓ ExecutionGuards: ACTIVE")
            except Exception as e:
                logger.error(f"  ✗ ExecutionGuards init failed: {e}")
        
        # 4. Rate Limit Manager
        self.rate_limiter: Optional[KrakenRateLimitManager] = None
        if self.flags.ENABLE_RATE_LIMIT_MANAGER and RATE_LIMIT_AVAILABLE:
            try:
                # Use default config - RateLimitConfig is a dataclass
                self.rate_limiter = KrakenRateLimitManager()
                logger.info("  ✓ RateLimitManager: ACTIVE")
            except Exception as e:
                logger.error(f"  ✗ RateLimitManager init failed: {e}")
        
        # 5. Human Override
        self.human_override: Optional[HumanOverrideProtocol] = None
        if self.flags.ENABLE_HUMAN_OVERRIDE and HUMAN_OVERRIDE_AVAILABLE:
            try:
                self.human_override = HumanOverrideProtocol()
                logger.info("  ✓ HumanOverride: ACTIVE")
            except Exception as e:
                logger.error(f"  ✗ HumanOverride init failed: {e}")
        
        # 6. Shadow Ledger
        self.shadow_ledger: Optional[ShadowLedgerWriter] = None
        if self.flags.ENABLE_SHADOW_LEDGER and SHADOW_LEDGER_AVAILABLE:
            try:
                self.shadow_ledger = ShadowLedgerWriter(output_dir="data/shadow_ledger")
                logger.info("  ✓ ShadowLedger: ACTIVE")
            except Exception as e:
                logger.error(f"  ✗ ShadowLedger init failed: {e}")
    
    def _log_module_status(self):
        """Log status of all modules."""
        logger.info("[P0 SAFETY] Module Status:")
        modules = [
            ("SOTARiskController", self.risk_controller),
            ("TradeGate", self.trade_gate),
            ("StaleDataGuard", self.stale_guard),
            ("DVOLController", self.dvol_controller),
            ("RateLimiter", self.rate_limiter),
            ("HumanOverride", self.human_override),
            ("ShadowLedger", self.shadow_ledger),
        ]
        for name, module in modules:
            status = "ACTIVE" if module else "INACTIVE"
            logger.info(f"  {name}: {status}")
    
    # =========================================================================
    # PRE-TICK UPDATE
    # =========================================================================
    
    def pre_tick_update(
        self,
        equity: float,
        prices: Dict[str, float],
        realized_pnl_today: float = 0.0,
        regime: str = "",
    ):
        """
        Update P0 modules at the start of each tick.

        Call this BEFORE processing any decisions.

        Args:
            equity: Current account equity
            prices: Dict of asset prices
            realized_pnl_today: Today's realized P&L
            regime: R6 - current regime name for DD threshold scaling
        """
        self._tick_count += 1

        # 1. Update risk controller with equity + regime
        if self.risk_controller:
            self.risk_controller.update_equity(equity, realized_pnl_today, regime=regime)
            self.risk_controller.update_prices(prices)
        
        # 2. Update stale data guard timestamps
        if self.stale_guard:
            for source in ["kraken_ws", "kraken_rest"]:
                self.stale_guard.update_timestamp(source)
        
        # 3. Check human override
        if self.human_override:
            self.human_override.check()  # Reads override file
    
    # =========================================================================
    # PRE-EXECUTION CHECK
    # =========================================================================
    
    def check_pre_execution(
        self,
        asset: str,
        direction: int,  # -1 short, 1 long, 0 close
        size: float,
        is_entry: bool,
        confidence: float = 0.5,
        regime: str = "UNKNOWN",
        dvol: float = 0.0,
        # [BUGFIX AUDIT-A2] Pass actual market context (was hardcoded)
        expected_alpha_bps: float = 60.0,
        drl_weight: float = 0.0,
        regime_confidence: float = 0.5,
        volume_ratio: float = 1.0,
        strategy: str = "",  # [W11] Best-of-N selected strategy for signal quality tracking
        market_data_age_seconds: Optional[float] = None,
        market_data_fetched_at_ts: Optional[float] = None,
        orderbook_stale: bool = False,
        orderbook_fallback_reason: str = "",
        orderbook_cache_age_seconds: Optional[float] = None,
        price_direction: str = "flat",
    ) -> P0SafetyResult:
        """
        Check all P0 safety conditions before execution.
        
        Call this AFTER decide() but BEFORE sending to exchange.
        
        Returns P0SafetyResult with:
        - allow_trade: Can proceed with trade
        - allow_exit: Can exit (even if can't open)
        - clipped_size: Size after clipping
        - reason: Why blocked (if blocked)
        """
        result = P0SafetyResult(
            result=P0CheckResult.ALLOW,
            original_size=size,
            clipped_size=size,
        )
        
        # Record to ledger (intent)
        if self.shadow_ledger:
            try:
                self.shadow_ledger.record_intent(
                    asset=asset,
                    direction=direction,
                    size=size,
                    is_entry=is_entry,
                    confidence=confidence,
                    regime=regime,
                    mode="PAPER",
                    extra={"strategy": strategy} if strategy else None,
                )
            except Exception as e:
                logger.error(f"[P0] Shadow ledger record_intent FAILED: {e}")
        
        # =====================================================================
        # CHECK 1: Human Override
        # =====================================================================
        
        if self.human_override:
            override_effects = self.human_override.get_effects()
            result.human_override_active = override_effects.any_override_active
            
            if override_effects.any_override_active:
                # Check for mode suppressions that affect trading
                if "NO_TRADE" in override_effects.suppressed_modes:
                    result.result = P0CheckResult.BLOCK_HUMAN_OVERRIDE
                    result.allow_trade = False
                    result.allow_exit = False
                    result.reason = "Human override: NO_TRADE mode active"
                    logger.warning(f"[P0] HUMAN OVERRIDE: NO_TRADE mode active")
                    return result
                
                if "EXIT_ONLY" in override_effects.suppressed_modes or "OPPORTUNITY" in override_effects.suppressed_modes:
                    result.result = P0CheckResult.BLOCK_EXIT_ONLY
                    result.allow_trade = False
                    result.allow_exit = True
                    result.reason = "Human override: EXIT_ONLY mode"
                    
                    if is_entry:
                        logger.warning(f"[P0] HUMAN OVERRIDE EXIT_ONLY: blocking entry")
                        return result
                
                # Check for asset restrictions
                if asset in override_effects.asset_max_exposure:
                    max_exposure = override_effects.asset_max_exposure[asset]
                    if max_exposure <= 0:
                        result.result = P0CheckResult.BLOCK_HUMAN_OVERRIDE
                        result.allow_trade = False
                        result.allow_exit = True
                        result.reason = f"Human override: {asset} restricted"
                        if is_entry:
                            logger.warning(f"[P0] HUMAN OVERRIDE: {asset} restricted to 0 exposure")
                            return result
        
        # =====================================================================
        # CHECK 2: Kill Switch / Risk State
        # =====================================================================
        
        if self.risk_controller:
            # Check kill switch
            if self.risk_controller.kill_switch_active:
                result.result = P0CheckResult.BLOCK_KILL_SWITCH
                result.allow_trade = False
                result.allow_exit = True  # Always allow exits
                result.kill_switch_active = True
                result.reason = f"Kill switch active: {self.risk_controller.kill_switch_reason}"
                
                if is_entry:
                    logger.critical(f"[P0] KILL SWITCH: {result.reason}")
                    return result
            
            # Check position allowed
            side = "short" if direction < 0 else "long"
            can_open, reason = self.risk_controller.can_open_position(asset, side, size)
            result.risk_state = self.risk_controller.risk_state.value
            
            if not can_open and is_entry:
                result.result = P0CheckResult.BLOCK_NO_TRADE
                result.allow_trade = False
                result.reason = reason
                logger.warning(f"[P0] RISK BLOCK: {reason}")
                return result
        
        # =====================================================================
        # CHECK 3: Trade Gate
        # =====================================================================
        
        if self.trade_gate:
            try:
                gate_result = self.trade_gate.check(
                    asset=asset,
                    side="short" if direction < 0 else "long",
                    size=size,
                    confidence=confidence,
                    regime=regime,
                    # [BUGFIX AUDIT-A2] Use caller-provided values, not hardcoded
                    drl_weight=drl_weight,
                    regime_confidence=regime_confidence,
                    expected_alpha_bps=expected_alpha_bps,
                    volume_ratio=volume_ratio,
                    price_direction=price_direction,
                    market_data_age_seconds=market_data_age_seconds,
                    market_data_fetched_at_ts=market_data_fetched_at_ts,
                    orderbook_stale=orderbook_stale,
                    orderbook_fallback_reason=orderbook_fallback_reason,
                    orderbook_cache_age_seconds=orderbook_cache_age_seconds,
                    alpha_gate_passed=True,  # [VC-1] Constitution alpha gate already ran
                )
                # [P48 2026-04-25 BUG-6] Was `.value` (returns int from auto()),
                # making gate_mode field in shadow ledger an unreadable integer.
                # `.name` returns symbolic name (e.g., "REJECT") for forensics.
                result.gate_mode = gate_result.decision.name

                # [P48 2026-04-25 BUG-4] Plumb gate_result.details into result.details
                # so the shadow ledger forensic tools (P41-P45) can see WHY this gate
                # rejected, not just THAT it did. Without this, p0_integrator-routed
                # rejections lost all gate-internal state (DVOL z-score, freshness
                # sources, edge bps, etc.) downstream.
                _gate_details = getattr(gate_result, "details", None) or {}

                if gate_result.decision == GateDecision.REJECT:
                    result.result = P0CheckResult.BLOCK_NO_TRADE
                    result.allow_trade = False
                    result.allow_exit = False
                    result.reason = f"Trade gate reject: {gate_result.reason.value}"
                    result.details.update(_gate_details)
                    logger.warning(f"[P0] TRADE GATE REJECT: {result.reason}")
                    return result

                elif gate_result.decision == GateDecision.EXIT_ONLY:
                    result.result = P0CheckResult.BLOCK_EXIT_ONLY
                    result.allow_trade = False
                    result.allow_exit = True
                    result.reason = f"Trade gate EXIT_ONLY: {gate_result.reason.value}"
                    result.details.update(_gate_details)

                    if is_entry:
                        logger.warning(f"[P0] TRADE GATE EXIT_ONLY: blocking entry")
                        return result

                elif gate_result.decision == GateDecision.CLIP:
                    # Clip size
                    if hasattr(gate_result, 'clipped_size'):
                        result.clipped_size = gate_result.clipped_size
                        result.result = P0CheckResult.CLIPPED
                        result.details.update(_gate_details)
                        logger.info(f"[P0] Size clipped: {size:.4f} -> {result.clipped_size:.4f}")

            except Exception as e:
                # [P48 2026-04-25 BUG-1] Was: log error, fall through with default
                # allow_trade=True. A gate crash was being silently treated as a PASS,
                # which violates the fail-closed contract for safety gates.
                # Now: fail-closed — if gate crashes, block the trade.
                logger.error(f"[P0] Trade gate error (FAIL-CLOSED): {e}")
                result.result = P0CheckResult.BLOCK_NO_TRADE
                result.allow_trade = False
                result.allow_exit = True  # exits still allowed (risk reduction)
                result.reason = f"Trade gate exception: {type(e).__name__}: {e}"
                return result
        
        # =====================================================================
        # CHECK 4: Stale Data Guard
        # =====================================================================
        
        if self.stale_guard:
            can_exec, reason, details = self.stale_guard.can_execute()
            
            if not can_exec:
                result.result = P0CheckResult.BLOCK_STALE_DATA
                result.allow_trade = False
                result.allow_exit = True  # Allow exits even with stale data
                result.reason = f"Stale data: {reason}"
                result.details["stale_sources"] = details
                
                if is_entry:
                    logger.warning(f"[P0] STALE DATA: {reason}")
                    return result
        
        # =====================================================================
        # CHECK 5: DVOL Override (can modify size)
        # =====================================================================
        
        if self.dvol_controller and dvol > 0:
            self.dvol_controller.update(dvol)
            mode = self.dvol_controller.get_execution_mode()
            
            if mode.value == "HALT":
                result.result = P0CheckResult.BLOCK_NO_TRADE
                result.allow_trade = False
                result.allow_exit = True
                result.reason = f"DVOL halt: dvol={dvol:.1f}"
                logger.warning(f"[P0] DVOL HALT: {result.reason}")
                return result
            
            elif mode.value == "REDUCED":
                # Clip to 50%
                result.clipped_size = min(result.clipped_size, size * 0.5)
                result.result = P0CheckResult.CLIPPED
                logger.info(f"[P0] DVOL REDUCED: size clipped to {result.clipped_size:.4f}")
        
        # =====================================================================
        # CHECK 6: Rate Limit
        # =====================================================================
        
        if self.rate_limiter:
            if not self.rate_limiter.can_proceed():
                result.result = P0CheckResult.BLOCK_RATE_LIMIT
                result.allow_trade = False
                result.allow_exit = True  # Allow exits
                result.reason = "Rate limited"
                
                if is_entry:
                    logger.warning("[P0] RATE LIMIT: blocking request")
                    return result
        
        # =====================================================================
        # ALL CHECKS PASSED
        # =====================================================================
        
        self._last_check_result = result
        return result
    
    # =========================================================================
    # POST-EXECUTION RECORDING
    # =========================================================================
    
    def record_order(
        self,
        asset: str,
        direction: int,
        size: float,
        order_id: str,
        order_type: str = "LIMIT",
        price: float = 0.0,
    ):
        """Record order to shadow ledger."""
        if self.shadow_ledger:
            try:
                self.shadow_ledger.record_order(
                    asset=asset,
                    direction=direction,
                    size=size,
                    order_id=order_id,
                    order_type=order_type,
                    price=price,
                )
            except Exception as e:
                logger.error(f"[P0] Shadow ledger record_order FAILED: {e}")
        
        # Consume rate limit token
        if self.rate_limiter:
            self.rate_limiter.consume()
    
    def record_fill(
        self,
        asset: str,
        side: str,
        size: float,
        price: float,
        order_id: str = "",
        fee: float = 0.0,
        realized_pnl: float = 0.0,
        extra: dict = None,  # [P0-FIX] pass-through metadata (exit_trigger, etc.)
    ) -> bool:
        """Record fill to shadow ledger and update risk controller."""
        recorded = False
        if self.shadow_ledger:
            try:
                self.shadow_ledger.record_fill(
                    asset=asset,
                    order_id=order_id,
                    fill_id=f"{order_id}_fill",
                    side=side,
                    size=size,
                    price=price,
                    fee=fee,
                    realized_pnl=realized_pnl,
                    extra=extra,  # [P0-FIX]
                )
                recorded = True
            except Exception as e:
                logger.error(f"[P0] Shadow ledger record_fill FAILED: {e}")

        # Update risk controller with realized PnL
        if self.risk_controller and realized_pnl != 0:
            self.risk_controller.daily_pnl += realized_pnl
        
        return recorded
    
    def record_position_change(
        self,
        asset: str,
        old_position: float,
        new_position: float,
        reason: str = "",
        old_direction: int = 0,
        new_direction: int = 0,
        realized_pnl: float = 0.0,
    ):
        """Record position change to shadow ledger."""
        if self.shadow_ledger:
            try:
                self.shadow_ledger.record_position_change(
                    asset=asset,
                    old_size=old_position,
                    new_size=new_position,
                    old_direction=old_direction,
                    new_direction=new_direction,
                    realized_pnl=realized_pnl,
                    reason=reason,
                )
            except Exception as e:
                logger.error(f"[P0] Shadow ledger record_position_change FAILED: {e}")
    
    # =========================================================================
    # STATE GETTERS
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current P0 safety status."""
        status = {
            "tick_count": self._tick_count,
            "modules": {},
            "last_check": None,
        }
        
        # Risk controller status
        if self.risk_controller:
            status["modules"]["risk_controller"] = {
                "active": True,
                "kill_switch": self.risk_controller.kill_switch_active,
                "risk_state": self.risk_controller.risk_state.value,
                "daily_pnl_pct": getattr(self.risk_controller, '_daily_pnl_pct', 0.0),
                "max_drawdown_pct": getattr(self.risk_controller, '_max_drawdown_pct', 0.0),
            }
        
        # Trade gate status
        if self.trade_gate:
            status["modules"]["trade_gate"] = {
                "active": True,
            }
        
        # Human override status
        if self.human_override:
            effects = self.human_override.get_effects()
            status["modules"]["human_override"] = {
                "active": True,
                "override_active": effects.any_override_active,
                "override_count": effects.override_count,
                "suppressed_modes": list(effects.suppressed_modes) if effects.suppressed_modes else [],
            }
        
        # Shadow ledger status
        if self.shadow_ledger:
            status["modules"]["shadow_ledger"] = {
                "active": True,
                "entry_count": getattr(self.shadow_ledger, '_entry_count', 0),
            }
        
        # Last check result
        if self._last_check_result:
            status["last_check"] = self._last_check_result.to_dict()
        
        return status
    
    def get_kill_switch_status(self) -> Tuple[bool, str]:
        """Get kill switch status."""
        if self.risk_controller:
            return (
                self.risk_controller.kill_switch_active,
                str(self.risk_controller.kill_switch_reason) if self.risk_controller.kill_switch_reason else ""
            )
        return False, ""
    
    def get_ledger_entries(self, n: int = 10) -> List[Dict]:
        """Get recent ledger entries."""
        if self.shadow_ledger and hasattr(self.shadow_ledger, 'get_recent'):
            return self.shadow_ledger.get_recent(n)
        return []


# =============================================================================
# SINGLETON
# =============================================================================

_p0_integrator: Optional[P0SafetyIntegrator] = None


def get_p0_integrator(config: Optional[Dict[str, Any]] = None) -> P0SafetyIntegrator:
    """Get singleton P0 safety integrator.

    Refresh the singleton when a different config is explicitly supplied later
    in the same process.
    """
    global _p0_integrator
    if _p0_integrator is None:
        _p0_integrator = P0SafetyIntegrator(config=config)
    elif config is not None and (_p0_integrator.config or {}) != (config or {}):
        logger.info("P0SafetyIntegrator config changed; refreshing singleton instance")
        _p0_integrator = P0SafetyIntegrator(config=config)
    return _p0_integrator


def reset_p0_integrator():
    """Reset singleton (for testing)."""
    global _p0_integrator
    _p0_integrator = None

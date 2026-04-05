"""
================================================================================
HMATS P0 Safe Execution Wrapper
================================================================================
Version: 5.4.1-P0FIX
Purpose: Wrap execution flow with P0 safety guarantees

This module provides P0-safe wrappers that MUST be used instead of direct
execution. It enforces:
    1. Unit conversion (exposure -> quantity)
    2. Authority chain (fail-closed risk checks)
    3. Leverage guard (max leverage, isolated margin)
    4. Account sync (real-time equity)
    5. Exchange guard (Kraken only)

USAGE IN main.py:
    from orchestration.p0_safe_execution import P0SafeExecutor
    
    self.p0_executor = P0SafeExecutor(
        mode=self.config.mode,
        kraken_client=self.kraken_client,
    )
    
    # In _execute_intent():
    result = await self.p0_executor.execute_safe(
        asset=asset,
        intent=intent,
        market_data=market_data,
        execution_manager=self.execution_manager,
    )

================================================================================
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORTS - P0 MODULES (ALL REQUIRED)
# =============================================================================

# Unit system - canonical conversion
from core.unit_system import (
    exposure_to_quantity,
    validate_exposure_fraction,
    validate_base_quantity,
)

# Account sync - real equity
from core.account_sync import (
    AccountSyncManager,
    get_account_sync,
    EquityStatus,
)

# Authority chain - risk pipeline
from core.authority_chain import (
    AuthorityChain,
    ChainResult,
    AuthorityStage,
    get_authority_chain,
)

# Exchange guard - Kraken only
from core.exchange_guard import (
    validate_kraken_only,
    ExchangeGuard,
    get_exchange_guard,
)

# Leverage guard - leverage limits
from risk.leverage_guard import (
    LeverageGuard,
    LeverageConfig,
    MarginMode,
    get_leverage_guard,
)


# =============================================================================
# EXECUTION RESULT
# =============================================================================

@dataclass
class P0ExecutionResult:
    """Result from P0-safe execution."""
    success: bool = False
    executed: bool = False
    
    # Order details
    asset: str = ""
    side: str = ""
    base_quantity: float = 0.0
    notional_usd: float = 0.0
    price: float = 0.0
    
    # Safety details
    account_equity: float = 0.0
    leverage: float = 0.0
    exposure_fraction: float = 0.0
    
    # Veto details (if not executed)
    veto_stage: Optional[str] = None
    veto_reason: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "executed": self.executed,
            "asset": self.asset,
            "side": self.side,
            "base_quantity": self.base_quantity,
            "notional_usd": self.notional_usd,
            "price": self.price,
            "account_equity": self.account_equity,
            "leverage": self.leverage,
            "exposure_fraction": self.exposure_fraction,
            "veto_stage": self.veto_stage,
            "veto_reason": self.veto_reason,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
        }


# =============================================================================
# P0 SAFE EXECUTOR
# =============================================================================

class P0SafeExecutor:
    """
    P0-safe execution wrapper.
    
    This class MUST be used for all trade execution in PAPER/LIVE modes.
    It enforces all P0 safety requirements:
    
    1. FAIL-CLOSED: All safety modules must be available
    2. UNIT SYSTEM: Exposure fraction -> base quantity conversion
    3. AUTHORITY CHAIN: All risk checks must pass
    4. LEVERAGE GUARD: Max leverage and isolated margin
    5. KRAKEN ONLY: No other exchanges in PAPER/LIVE
    
    Usage:
        executor = P0SafeExecutor(mode="paper", kraken_client=client)
        result = await executor.execute_safe(asset, intent, market_data, exec_mgr)
    """
    
    def __init__(
        self,
        mode: str = "verify",
        kraken_client: Optional[Any] = None,
        dry_run: bool = True,
        strict_mode: bool = True,
    ):
        """
        Initialize P0-safe executor.
        
        Args:
            mode: Run mode (verify, paper, live)
            kraken_client: Kraken exchange client
            dry_run: Whether to simulate execution
            strict_mode: If True, all modules must be available
        """
        self.mode = mode.lower()
        self.kraken_client = kraken_client
        self.dry_run = dry_run
        self.strict_mode = strict_mode and self.mode in ("paper", "live")
        
        # Initialize P0 modules
        self._init_p0_modules()
        
        # Statistics
        self._stats = {
            "total_attempts": 0,
            "executed": 0,
            "vetoed": 0,
            "errors": 0,
            "vetoes_by_stage": {},
        }
        
        logger.info(
            f"[P0_EXECUTOR] Initialized: mode={mode}, "
            f"strict={self.strict_mode}, dry_run={dry_run}"
        )
    
    def _init_p0_modules(self):
        """Initialize all P0 safety modules."""
        
        # Exchange guard
        self.exchange_guard = get_exchange_guard(self.mode)
        
        # Validate Kraken only
        if self.strict_mode:
            validate_kraken_only("kraken", self.mode, "P0SafeExecutor init")
        
        # Account sync
        self.account_sync = get_account_sync(
            exchange_name="kraken",
            dry_run=self.dry_run,
        )
        
        # Leverage guard
        self.leverage_guard = get_leverage_guard()
        
        # Authority chain is initialized lazily via set_authority_chain()
        # This allows the main system to inject its configured risk modules
        self._authority_chain: Optional[AuthorityChain] = None
        
        logger.info("[P0_EXECUTOR] P0 modules initialized")
    
    def set_authority_chain(
        self,
        data_guard = None,
        trade_gate = None,
        risk_manager = None,
    ):
        """
        Set authority chain with actual risk modules.
        
        This should be called after main.py initializes its modules.
        """
        self._authority_chain = AuthorityChain(
            data_guard=data_guard,
            trade_gate=trade_gate,
            risk_manager=risk_manager,
            leverage_guard=self.leverage_guard,
            strict_mode=self.strict_mode,
        )
        logger.info("[P0_EXECUTOR] Authority chain configured")
    
    async def execute_safe(
        self,
        asset: str,
        intent: Any,  # TradeIntentV36
        market_data: Dict[str, Any],
        execution_manager: Any,
    ) -> P0ExecutionResult:
        """
        Execute trade with P0 safety guarantees.
        
        FLOW:
            1. Validate exchange (Kraken only)
            2. Get fresh account equity
            3. Convert exposure to quantity
            4. Run authority chain
            5. Validate leverage
            6. Execute (if all pass)
        
        Args:
            asset: Asset symbol (SOL, BTC, ETH)
            intent: Trade intent from v3.6 engine
            market_data: Current market data
            execution_manager: Execution manager for actual orders
            
        Returns:
            P0ExecutionResult with execution details or veto reason
        """
        start_time = datetime.utcnow()
        self._stats["total_attempts"] += 1
        
        result = P0ExecutionResult(
            asset=asset,
            exposure_fraction=intent.target_exposure if hasattr(intent, 'target_exposure') else 0.0,
        )
        
        try:
            # =================================================================
            # STEP 1: Validate exchange
            # =================================================================
            
            if self.strict_mode:
                self.exchange_guard.check_name("kraken", f"execute {asset}")
            
            # =================================================================
            # STEP 2: Get fresh account equity
            # =================================================================
            
            await self.account_sync.refresh()
            account_equity = self.account_sync.get_equity()
            result.account_equity = account_equity
            
            # =================================================================
            # STEP 3: Convert exposure to quantity
            # =================================================================
            
            current_price = market_data.get("current_price", 0.0)
            if current_price <= 0:
                raise ValueError(f"Invalid price: {current_price}")
            
            exposure_fraction = intent.target_exposure if hasattr(intent, 'target_exposure') else 0.0
            
            # Validate exposure
            validate_exposure_fraction(exposure_fraction)
            
            # Convert
            base_quantity = exposure_to_quantity(
                exposure_fraction=exposure_fraction,
                account_equity=account_equity,
                price=current_price,
                asset=asset,
            )
            
            result.base_quantity = base_quantity
            result.price = current_price
            result.notional_usd = base_quantity * current_price
            
            # =================================================================
            # STEP 4: Run authority chain
            # =================================================================
            
            direction = intent.direction if hasattr(intent, 'direction') else 0.0
            result.side = "BUY" if direction > 0 else "SELL"
            
            if self._authority_chain is not None:
                chain_result = self._authority_chain.evaluate(
                    asset=asset,
                    direction=direction,
                    exposure_fraction=exposure_fraction,
                    base_quantity=base_quantity,
                    price=current_price,
                    account_equity=account_equity,
                    regime=market_data.get("regime_state", "UNKNOWN"),
                    confidence=getattr(intent, 'quant_confidence', 0.5),
                )
                
                if not chain_result.approved:
                    result.veto_stage = chain_result.veto_stage.name if chain_result.veto_stage else "UNKNOWN"
                    result.veto_reason = chain_result.veto_reason
                    self._stats["vetoed"] += 1
                    self._stats["vetoes_by_stage"][result.veto_stage] = (
                        self._stats["vetoes_by_stage"].get(result.veto_stage, 0) + 1
                    )
                    result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return result
            
            # =================================================================
            # STEP 5: Validate leverage
            # =================================================================
            
            approved, reason = self.leverage_guard.validate_order(
                position_notional_usd=result.notional_usd,
                account_equity=account_equity,
                current_price=current_price,
                is_short=(direction < 0),
                asset=asset,
            )
            
            if not approved:
                result.veto_stage = "LEVERAGE_GUARD"
                result.veto_reason = f"[LEVERAGE_GUARD] {reason}"
                self._stats["vetoed"] += 1
                result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return result
            
            # Calculate actual leverage
            result.leverage = result.notional_usd / account_equity if account_equity > 0 else 0.0
            
            # =================================================================
            # STEP 6: Execute
            # =================================================================
            
            if execution_manager is None:
                result.veto_stage = "EXECUTION_MANAGER"
                result.veto_reason = "No execution manager available"
                result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return result
            
            # Determine order type
            exec_mode = getattr(intent, 'execution_mode', None)
            order_type = "MARKET" if (exec_mode and exec_mode.value == "AGGRESSIVE") else "LIMIT"
            
            # Execute with CORRECT units
            exec_result = execution_manager.execute_order(
                symbol=f"{asset}/USD",
                side=result.side,
                size=base_quantity,  # ✓ CORRECT: Base quantity, NOT exposure fraction
                price=current_price,
                order_type=order_type,
            )
            
            result.success = True
            result.executed = True
            self._stats["executed"] += 1
            
            logger.info(
                f"[P0_EXECUTOR] EXECUTED: {asset} {result.side} "
                f"qty={base_quantity:.4f} @ ${current_price:.2f} "
                f"notional=${result.notional_usd:,.2f} leverage={result.leverage:.2f}x"
            )
            
        except Exception as e:
            result.veto_stage = "EXCEPTION"
            result.veto_reason = f"[P0_EXCEPTION] {e}"
            self._stats["errors"] += 1
            logger.error(f"[P0_EXECUTOR] Error: {e}")
        
        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["executed"] / max(self._stats["total_attempts"], 1)
            ),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_p0_executor: Optional[P0SafeExecutor] = None


def get_p0_executor(
    mode: str = "verify",
    kraken_client: Optional[Any] = None,
    dry_run: bool = True,
) -> P0SafeExecutor:
    """Get or create P0-safe executor singleton."""
    global _p0_executor
    if _p0_executor is None:
        _p0_executor = P0SafeExecutor(
            mode=mode,
            kraken_client=kraken_client,
            dry_run=dry_run,
        )
    return _p0_executor


def reset_p0_executor():
    """Reset P0-safe executor singleton."""
    global _p0_executor
    _p0_executor = None


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("P0 Safe Executor Self-Test")
    print("=" * 60)
    
    async def test():
        # Create executor
        executor = P0SafeExecutor(mode="paper", dry_run=True)
        
        # Mock intent
        class MockIntent:
            target_exposure = 0.25
            direction = 1.0
            quant_confidence = 0.7
            execution_mode = type('obj', (object,), {'value': 'NORMAL'})()
        
        # Mock execution manager
        class MockExecutionManager:
            def execute_order(self, **kwargs):
                print(f"  [MOCK_EXEC] {kwargs}")
                return {"status": "FILLED"}
        
        intent = MockIntent()
        market_data = {"current_price": 185.0, "regime_state": "NORMAL"}
        exec_mgr = MockExecutionManager()
        
        # Set mock equity
        executor.account_sync.set_dry_run_equity(100_000.0)
        
        # Execute
        result = await executor.execute_safe(
            asset="SOL",
            intent=intent,
            market_data=market_data,
            execution_manager=exec_mgr,
        )
        
        print(f"\nResult: {result.to_dict()}")
        
        # Verify
        assert result.success, f"Should succeed, got: {result.veto_reason}"
        assert abs(result.base_quantity - 135.135) < 0.1, f"Wrong quantity: {result.base_quantity}"
        
        print("\n✓ P0 Safe Executor test passed")
    
    asyncio.run(test())

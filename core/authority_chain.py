"""
================================================================================
HMATS Authority Chain - Risk Check Pipeline with One-Veto-Kill Semantics
================================================================================
Version: 5.4.1-P0FIX
Purpose: Enforce strict risk check order where ANY veto terminates execution

AUTHORITY CHAIN ORDER (IMMUTABLE):
    1. DataIntegrity / StaleDataKillSwitch
    2. TradeGate
    3. RiskManager  
    4. LeverageGuard
    5. -> Execution (only if ALL pass)

CRITICAL RULES:
    - ANY stage veto -> immediate termination
    - Later stages CANNOT override earlier vetoes
    - Every veto MUST have non-empty veto_reason
    - Logging level ≥ INFO (never DEBUG-only)
    - Exceptions in any check -> automatic veto (fail-closed)

================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum, auto
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# VETO TYPES AND STAGES
# =============================================================================

class AuthorityStage(Enum):
    """Stages in the authority chain (execution order)."""
    DATA_INTEGRITY = 1
    TRADE_GATE = 2
    RISK_MANAGER = 3
    LEVERAGE_GUARD = 4
    EXECUTION = 5  # Final stage, only reached if all pass


class VetoSeverity(Enum):
    """Severity of veto."""
    HARD = "HARD"       # Non-negotiable, immediate halt
    SOFT = "SOFT"       # Reduces/delays but may proceed with adjustment
    EXCEPTION = "EXCEPTION"  # Veto due to exception (fail-closed)


@dataclass
class StageResult:
    """Result from a single authority stage."""
    stage: AuthorityStage
    passed: bool
    veto_reason: str = ""
    severity: VetoSeverity = VetoSeverity.HARD
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Adjustments (only valid if passed=True)
    size_multiplier: float = 1.0
    delay_seconds: int = 0


@dataclass
class ChainResult:
    """Result from full authority chain evaluation."""
    approved: bool = False
    final_stage: AuthorityStage = AuthorityStage.DATA_INTEGRITY
    veto_reason: str = ""
    veto_stage: Optional[AuthorityStage] = None
    veto_severity: Optional[VetoSeverity] = None
    
    # Aggregated adjustments
    size_multiplier: float = 1.0
    delay_seconds: int = 0
    
    # Full audit trail
    stage_results: List[StageResult] = field(default_factory=list)
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    def total_time_ms(self) -> float:
        """Get total chain evaluation time in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    def to_log(self) -> str:
        """Generate log line."""
        if self.approved:
            return f"[AUTHORITY_CHAIN] APPROVED: size_mult={self.size_multiplier:.2f}"
        else:
            return f"[AUTHORITY_CHAIN] VETOED at {self.veto_stage.name}: {self.veto_reason}"


# =============================================================================
# AUTHORITY CHAIN
# =============================================================================

class AuthorityChain:
    """
    Risk check pipeline with strict ordering and one-veto-kill semantics.
    
    This is the SINGLE POINT OF TRUTH for trade authorization.
    All risk checks MUST go through this chain.
    
    Usage:
        chain = AuthorityChain(
            data_guard=stale_data_guard,
            trade_gate=trade_gate,
            risk_manager=risk_manager,
            leverage_guard=leverage_guard,
        )
        
        result = chain.evaluate(
            asset="SOL",
            direction=1.0,
            exposure_fraction=0.3,
            base_quantity=162.16,
            price=185.0,
            account_equity=10000.0,
        )
        
        if not result.approved:
            raise OrderRejected(result.veto_reason)
    """
    
    def __init__(
        self,
        data_guard = None,       # StaleDataKillSwitch or similar
        trade_gate = None,       # TradeGate
        risk_manager = None,     # RiskManager
        leverage_guard = None,   # LeverageGuard
        strict_mode: bool = True # If True, missing modules = fail-closed
    ):
        """
        Initialize authority chain.
        
        Args:
            data_guard: Data integrity checker
            trade_gate: Trade gate
            risk_manager: Risk manager
            leverage_guard: Leverage guard
            strict_mode: If True, missing modules cause rejection
        """
        self.data_guard = data_guard
        self.trade_gate = trade_gate
        self.risk_manager = risk_manager
        self.leverage_guard = leverage_guard
        self.strict_mode = strict_mode
        
        # Statistics
        self._stats = {
            "total_evaluations": 0,
            "approved": 0,
            "rejected": 0,
            "rejections_by_stage": {stage.name: 0 for stage in AuthorityStage},
        }
        
        # Validate modules in strict mode
        if strict_mode:
            self._validate_modules()
        
        logger.info(
            f"[AUTHORITY_CHAIN] Initialized: strict_mode={strict_mode}, "
            f"modules=[data_guard={'OK' if data_guard else 'MISSING'}, "
            f"trade_gate={'OK' if trade_gate else 'MISSING'}, "
            f"risk_manager={'OK' if risk_manager else 'MISSING'}, "
            f"leverage_guard={'OK' if leverage_guard else 'MISSING'}]"
        )
    
    def _validate_modules(self):
        """Validate all required modules are present."""
        missing = []
        if self.data_guard is None:
            missing.append("data_guard")
        if self.trade_gate is None:
            missing.append("trade_gate")
        if self.risk_manager is None:
            missing.append("risk_manager")
        if self.leverage_guard is None:
            missing.append("leverage_guard")
        
        if missing:
            raise RuntimeError(
                f"FAIL-CLOSED: Authority chain missing required modules: {missing}. "
                f"All modules are MANDATORY in strict_mode."
            )
    
    def evaluate(
        self,
        asset: str,
        direction: float,
        exposure_fraction: float,
        base_quantity: float,
        price: float,
        account_equity: float,
        **context
    ) -> ChainResult:
        """
        Evaluate trade through full authority chain.
        
        ONE-VETO-KILL: Any stage veto immediately terminates evaluation.
        
        Args:
            asset: Asset symbol (SOL, BTC, ETH)
            direction: Trade direction (-1.0 to 1.0)
            exposure_fraction: Target exposure as fraction of account
            base_quantity: Calculated base quantity to trade
            price: Current asset price
            account_equity: Current account equity
            **context: Additional context passed to all stages
            
        Returns:
            ChainResult with approval status and audit trail
        """
        self._stats["total_evaluations"] += 1
        
        result = ChainResult(
            start_time=time.time(),
            approved=False,
        )
        
        # Build context for all stages
        ctx = {
            "asset": asset,
            "direction": direction,
            "exposure_fraction": exposure_fraction,
            "base_quantity": base_quantity,
            "price": price,
            "account_equity": account_equity,
            "position_notional_usd": base_quantity * price,
            **context
        }
        
        # Stage 1: Data Integrity
        stage_result = self._check_data_integrity(ctx)
        result.stage_results.append(stage_result)
        result.final_stage = AuthorityStage.DATA_INTEGRITY
        
        if not stage_result.passed:
            return self._finalize_rejection(result, stage_result)
        
        # Stage 2: Trade Gate
        stage_result = self._check_trade_gate(ctx)
        result.stage_results.append(stage_result)
        result.final_stage = AuthorityStage.TRADE_GATE
        
        if not stage_result.passed:
            return self._finalize_rejection(result, stage_result)
        
        # Apply adjustments from trade gate
        result.size_multiplier *= stage_result.size_multiplier
        result.delay_seconds = max(result.delay_seconds, stage_result.delay_seconds)
        
        # Stage 3: Risk Manager
        stage_result = self._check_risk_manager(ctx)
        result.stage_results.append(stage_result)
        result.final_stage = AuthorityStage.RISK_MANAGER
        
        if not stage_result.passed:
            return self._finalize_rejection(result, stage_result)
        
        # Apply adjustments from risk manager
        result.size_multiplier *= stage_result.size_multiplier
        
        # Stage 4: Leverage Guard
        stage_result = self._check_leverage_guard(ctx)
        result.stage_results.append(stage_result)
        result.final_stage = AuthorityStage.LEVERAGE_GUARD
        
        if not stage_result.passed:
            return self._finalize_rejection(result, stage_result)
        
        # All stages passed
        result.approved = True
        result.final_stage = AuthorityStage.EXECUTION
        result.end_time = time.time()
        
        self._stats["approved"] += 1
        
        logger.info(
            f"[AUTHORITY_CHAIN] APPROVED: {asset} "
            f"direction={direction:+.2f} qty={base_quantity:.4f} "
            f"size_mult={result.size_multiplier:.2f} "
            f"time={result.total_time_ms():.1f}ms"
        )
        
        return result
    
    def _check_data_integrity(self, ctx: Dict) -> StageResult:
        """Check data integrity / stale data."""
        stage = AuthorityStage.DATA_INTEGRITY
        
        if self.data_guard is None:
            if self.strict_mode:
                return StageResult(
                    stage=stage,
                    passed=False,
                    veto_reason="DATA_GUARD_UNAVAILABLE",
                    severity=VetoSeverity.HARD,
                )
            else:
                return StageResult(stage=stage, passed=True)
        
        try:
            # Try different interfaces
            if hasattr(self.data_guard, 'can_execute'):
                can_exec, reason, details = self.data_guard.can_execute()
                if not can_exec:
                    return StageResult(
                        stage=stage,
                        passed=False,
                        veto_reason=f"STALE_DATA: {reason}",
                        severity=VetoSeverity.HARD,
                        details=details,
                    )
            elif hasattr(self.data_guard, 'check_freshness'):
                is_fresh, details = self.data_guard.check_freshness()
                if not is_fresh:
                    return StageResult(
                        stage=stage,
                        passed=False,
                        veto_reason=f"STALE_DATA: {details.get('stale_sources', [])}",
                        severity=VetoSeverity.HARD,
                        details=details,
                    )
            
            return StageResult(stage=stage, passed=True)
            
        except Exception as e:
            logger.error(f"[AUTHORITY_CHAIN] Data integrity exception: {e}")
            return StageResult(
                stage=stage,
                passed=False,
                veto_reason=f"DATA_INTEGRITY_EXCEPTION: {e}",
                severity=VetoSeverity.EXCEPTION,
            )
    
    def _check_trade_gate(self, ctx: Dict) -> StageResult:
        """Check trade gate."""
        stage = AuthorityStage.TRADE_GATE
        
        if self.trade_gate is None:
            if self.strict_mode:
                return StageResult(
                    stage=stage,
                    passed=False,
                    veto_reason="TRADE_GATE_UNAVAILABLE",
                    severity=VetoSeverity.HARD,
                )
            else:
                return StageResult(stage=stage, passed=True)
        
        try:
            # Call trade gate check
            side = "long" if ctx["direction"] > 0 else "short"
            
            if hasattr(self.trade_gate, 'check'):
                gate_result = self.trade_gate.check(
                    asset=ctx["asset"],
                    side=side,
                    size=ctx["exposure_fraction"],
                    confidence=ctx.get("confidence", 0.5),
                    regime=ctx.get("regime", "UNKNOWN"),
                    drl_weight=ctx.get("drl_weight", 0.0),
                    regime_confidence=ctx.get("regime_confidence", 0.5),
                    expected_alpha_bps=ctx.get("expected_alpha_bps", 60.0),
                    alpha_gate_passed=ctx.get("alpha_gate_passed", False),
                    volume_ratio=ctx.get("volume_ratio", 1.0),
                    market_data_age_seconds=ctx.get("market_data_age_seconds"),
                    market_data_fetched_at_ts=ctx.get("market_data_fetched_at_ts"),
                    orderbook_stale=ctx.get("orderbook_stale", False),
                    orderbook_fallback_reason=ctx.get("orderbook_fallback_reason", ""),
                    orderbook_cache_age_seconds=ctx.get("orderbook_cache_age_seconds"),
                    # [P50 2026-04-25] P47/P48 plumbing extended to the third
                    # trade_gate.check call site (CLAUDE.md non-negotiable rule
                    # #6). Without these, DVOL gate (Gate 2) and price-direction
                    # logic are dead code on the authority_chain path. AuthorityChain
                    # is currently orphan-instantiated (main.py:2698 never calls
                    # .evaluate()), so this is defensive, but if the path is
                    # activated, gate inputs match the other two sites.
                    dvol_zscore=float(ctx.get("dvol_zscore", 0.0) or 0.0),
                    dvol_current=float(ctx.get("dvol_current", 0.0) or 0.0),
                    price_direction=ctx.get("price_direction", "flat"),
                )
                
                # Check decision
                decision = gate_result.decision.name if hasattr(gate_result.decision, 'name') else str(gate_result.decision)
                
                if decision not in ("ALLOW", "REDUCE"):
                    reason = gate_result.reason.name if hasattr(gate_result.reason, 'name') else str(gate_result.reason)
                    return StageResult(
                        stage=stage,
                        passed=False,
                        veto_reason=f"TRADE_GATE_{decision}: {reason}",
                        severity=VetoSeverity.HARD,
                        details=gate_result.details if hasattr(gate_result, 'details') else {},
                    )
                
                # Get size adjustment
                size_mult = 1.0
                if decision == "REDUCE" and hasattr(gate_result, 'adjusted_size'):
                    original = ctx["base_quantity"]
                    adjusted = float(gate_result.adjusted_size)
                    if original > 0:
                        size_mult = adjusted / original
                
                return StageResult(
                    stage=stage,
                    passed=True,
                    size_multiplier=size_mult,
                )
            
            return StageResult(stage=stage, passed=True)
            
        except Exception as e:
            logger.error(f"[AUTHORITY_CHAIN] Trade gate exception: {e}")
            return StageResult(
                stage=stage,
                passed=False,
                veto_reason=f"TRADE_GATE_EXCEPTION: {e}",
                severity=VetoSeverity.EXCEPTION,
            )
    
    def _check_risk_manager(self, ctx: Dict) -> StageResult:
        """Check risk manager."""
        stage = AuthorityStage.RISK_MANAGER
        
        if self.risk_manager is None:
            if self.strict_mode:
                return StageResult(
                    stage=stage,
                    passed=False,
                    veto_reason="RISK_MANAGER_UNAVAILABLE",
                    severity=VetoSeverity.HARD,
                )
            else:
                return StageResult(stage=stage, passed=True)
        
        try:
            # Try check_trade_allowed interface
            if hasattr(self.risk_manager, 'check_trade_allowed'):
                result = self.risk_manager.check_trade_allowed(
                    symbol=ctx["asset"],
                    direction=ctx["direction"],
                    size=ctx["exposure_fraction"],
                )
                
                # Default to False if not explicitly approved
                approved = result.get("approved", False)
                
                if not approved:
                    return StageResult(
                        stage=stage,
                        passed=False,
                        veto_reason=f"RISK_MANAGER: {result.get('reason', 'Not approved')}",
                        severity=VetoSeverity.HARD,
                        details=result,
                    )
                
                # Check for size adjustment
                size_mult = 1.0
                if "adjusted_size" in result:
                    original = ctx["exposure_fraction"]
                    if original != 0:
                        size_mult = abs(result["adjusted_size"]) / abs(original)
                
                return StageResult(
                    stage=stage,
                    passed=True,
                    size_multiplier=size_mult,
                )
            
            # Try can_trade interface
            elif hasattr(self.risk_manager, 'can_trade'):
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    return StageResult(
                        stage=stage,
                        passed=False,
                        veto_reason=f"RISK_MANAGER: {reason}",
                        severity=VetoSeverity.HARD,
                    )
            
            return StageResult(stage=stage, passed=True)
            
        except Exception as e:
            logger.error(f"[AUTHORITY_CHAIN] Risk manager exception: {e}")
            return StageResult(
                stage=stage,
                passed=False,
                veto_reason=f"RISK_MANAGER_EXCEPTION: {e}",
                severity=VetoSeverity.EXCEPTION,
            )
    
    def _check_leverage_guard(self, ctx: Dict) -> StageResult:
        """Check leverage guard."""
        stage = AuthorityStage.LEVERAGE_GUARD
        
        if self.leverage_guard is None:
            if self.strict_mode:
                return StageResult(
                    stage=stage,
                    passed=False,
                    veto_reason="LEVERAGE_GUARD_UNAVAILABLE",
                    severity=VetoSeverity.HARD,
                )
            else:
                return StageResult(stage=stage, passed=True)
        
        try:
            # Call leverage guard
            approved, reason = self.leverage_guard.validate_order(
                position_notional_usd=ctx["position_notional_usd"],
                account_equity=ctx["account_equity"],
                current_price=ctx["price"],
                is_short=(ctx["direction"] < 0),
                asset=ctx["asset"],
            )
            
            if not approved:
                return StageResult(
                    stage=stage,
                    passed=False,
                    veto_reason=f"LEVERAGE_GUARD: {reason}",
                    severity=VetoSeverity.HARD,
                )
            
            return StageResult(stage=stage, passed=True)
            
        except Exception as e:
            logger.error(f"[AUTHORITY_CHAIN] Leverage guard exception: {e}")
            return StageResult(
                stage=stage,
                passed=False,
                veto_reason=f"LEVERAGE_GUARD_EXCEPTION: {e}",
                severity=VetoSeverity.EXCEPTION,
            )
    
    def _finalize_rejection(self, result: ChainResult, stage_result: StageResult) -> ChainResult:
        """Finalize a rejection result."""
        result.approved = False
        result.veto_stage = stage_result.stage
        result.veto_reason = stage_result.veto_reason
        result.veto_severity = stage_result.severity
        result.end_time = time.time()
        
        self._stats["rejected"] += 1
        self._stats["rejections_by_stage"][stage_result.stage.name] += 1
        
        # Log at INFO level (never DEBUG-only)
        logger.info(
            f"[AUTHORITY_CHAIN] VETOED at {stage_result.stage.name}: "
            f"{stage_result.veto_reason} "
            f"(severity={stage_result.severity.value})"
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        total = self._stats["total_evaluations"]
        return {
            **self._stats,
            "approval_rate": self._stats["approved"] / max(total, 1),
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_authority_chain: Optional[AuthorityChain] = None
_authority_chain_signature: Optional[tuple] = None


def _normalize_authority_value(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return ("id", id(value))


def get_authority_chain(
    strict_mode: bool = False,  # Default to non-strict for dry run compatibility
    auto_load_modules: bool = True,
    **kwargs
) -> AuthorityChain:
    """Get or create authority chain singleton.
    
    Args:
        strict_mode: If True, missing modules cause rejection. Default False for dry run.
        auto_load_modules: If True, try to load available modules automatically.
        **kwargs: Additional arguments passed to AuthorityChain constructor.
    """
    global _authority_chain, _authority_chain_signature

    resolved_kwargs = dict(kwargs)

    if auto_load_modules:
        if 'leverage_guard' not in resolved_kwargs:
            try:
                from risk.leverage_guard import get_leverage_guard
                resolved_kwargs['leverage_guard'] = get_leverage_guard()
            except ImportError:
                pass

        if 'trade_gate' not in resolved_kwargs:
            try:
                from defense.trade_gate import get_trade_gate
                resolved_kwargs['trade_gate'] = get_trade_gate()
            except ImportError:
                pass

    resolved_kwargs['strict_mode'] = strict_mode
    signature = (
        strict_mode,
        auto_load_modules,
        tuple(
            sorted((key, _normalize_authority_value(value)) for key, value in resolved_kwargs.items())
        ),
    )

    if _authority_chain is None:
        _authority_chain = AuthorityChain(**resolved_kwargs)
        _authority_chain_signature = signature
    elif _authority_chain_signature != signature:
        logger.info("[AUTHORITY_CHAIN] Constructor inputs changed; refreshing singleton instance")
        _authority_chain = AuthorityChain(**resolved_kwargs)
        _authority_chain_signature = signature
    return _authority_chain


def reset_authority_chain():
    """Reset authority chain singleton."""
    global _authority_chain, _authority_chain_signature
    _authority_chain = None
    _authority_chain_signature = None


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Authority Chain Self-Test")
    print("=" * 60)
    
    # Mock modules
    class MockDataGuard:
        def can_execute(self):
            return True, "OK", {}
    
    class MockTradeGate:
        def check(self, **kwargs):
            class Result:
                decision = type('obj', (object,), {'name': 'ALLOW'})()
                reason = type('obj', (object,), {'name': 'NONE'})()
                details = {}
            return Result()
    
    class MockRiskManager:
        def check_trade_allowed(self, **kwargs):
            return {"approved": True, "reason": "OK"}
    
    class MockLeverageGuard:
        def validate_order(self, **kwargs):
            if kwargs.get("position_notional_usd", 0) > 30000:
                return False, "Exceeds max position"
            return True, "Approved"
    
    # Test 1: All pass
    print("\n[Test 1] All stages pass")
    chain = AuthorityChain(
        data_guard=MockDataGuard(),
        trade_gate=MockTradeGate(),
        risk_manager=MockRiskManager(),
        leverage_guard=MockLeverageGuard(),
        strict_mode=True,
    )
    
    result = chain.evaluate(
        asset="SOL",
        direction=1.0,
        exposure_fraction=0.2,
        base_quantity=108.0,
        price=185.0,
        account_equity=10000.0,
    )
    print(f"  Approved: {result.approved}")
    print(f"  Stages passed: {len(result.stage_results)}")
    assert result.approved, "Should be approved"
    
    # Test 2: Leverage guard veto
    print("\n[Test 2] Leverage guard veto")
    result = chain.evaluate(
        asset="SOL",
        direction=1.0,
        exposure_fraction=0.5,
        base_quantity=270.0,
        price=185.0,
        account_equity=10000.0,
    )
    print(f"  Approved: {result.approved}")
    print(f"  Veto stage: {result.veto_stage.name if result.veto_stage else 'None'}")
    print(f"  Veto reason: {result.veto_reason}")
    assert not result.approved, "Should be rejected"
    assert result.veto_stage == AuthorityStage.LEVERAGE_GUARD
    
    # Test 3: Missing module in strict mode
    print("\n[Test 3] Missing module rejection")
    try:
        AuthorityChain(
            data_guard=MockDataGuard(),
            trade_gate=None,  # Missing!
            risk_manager=MockRiskManager(),
            leverage_guard=MockLeverageGuard(),
            strict_mode=True,
        )
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"  ✓ Correctly rejected: {e}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Chain Statistics:")
    for k, v in chain.get_stats().items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

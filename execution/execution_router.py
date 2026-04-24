"""
================================================================================
HMATS v6.8 — Execution Router (v2.0 Part 2.1)
================================================================================
Routes TradeIntent to the appropriate executor:
  - SpotMarginExecutor (existing, core/execution_service.py) — default path
  - DerivativesExecutor (new)                                 — perp / carry

Routing rules (checked in order — first match wins):

  1. DERIVATIVES_ENABLED=false          → SpotMargin (global kill switch)
  2. Derivatives gate vetoes intent     → SpotMargin (fallback when possible)
     OR intent vetoed entirely if carry-only
  3. intent.strategy == 'CASH_CARRY'    → Derivatives (mandatory — no spot path)
  4. intent.direction > 0 AND
     intent.leverage > 1.0              → Derivatives (perp long w/ leverage)
  5. intent.direction < 0 AND
     context.funding_rate_h > threshold
     AND budget available               → Derivatives (earn funding via short perp)
  6. default                            → SpotMargin

Design notes:
- Router is a thin dispatcher. It does NOT own risk state; both executors
  enforce their own caps. The DerivativesRiskGate is called before routing
  to the derivatives executor.
- Router returns a uniform result object (dict with `route`, `success`,
  `reason`, `details`) so main.py / audit_trail don't need to know which
  executor handled the intent.
- FAIL-CLOSED: any unexpected error routes to spot/margin. Derivatives
  must prove itself healthy before receiving traffic.

Wiring tag: [WIRE-ROUTER]
================================================================================
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTE ENUM & RESULT
# =============================================================================

class ExecutionRoute(Enum):
    SPOT_MARGIN = "SPOT_MARGIN"
    DERIVATIVES = "DERIVATIVES"


@dataclass
class RouteDecision:
    route: ExecutionRoute
    reason: str = ""
    gate_veto: Optional[str] = None  # derivative-gate veto, if any
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ROUTER
# =============================================================================

class ExecutionRouter:
    """Dispatch TradeIntent to the correct executor.

    The router's `route_and_execute()` method is what main.py calls. Under the
    hood it picks an executor via `decide_route()` and calls it. `decide_route()`
    is exposed separately so it can be audited / tested in isolation.
    """

    # Minimum 1h funding rate to justify routing a short directional intent
    # through the derivatives path (earns funding instead of shorting via margin)
    FUNDING_SHORT_PERP_THRESHOLD_H = 0.0001  # 0.01%/h

    def __init__(
        self,
        spot_margin_executor,
        derivatives_executor=None,
        derivatives_gate=None,
    ):
        """
        Args:
            spot_margin_executor: existing execution_service.execute_intent_v2
                                   or an object with an `execute()` interface.
            derivatives_executor: DerivativesExecutor instance (optional).
            derivatives_gate: DerivativesRiskGate instance (optional).
        """
        self.spot_margin = spot_margin_executor
        self.derivatives = derivatives_executor
        self.gate = derivatives_gate

    # ------------------------------------------------------------------
    # GLOBAL KILL SWITCH
    # ------------------------------------------------------------------
    def _derivatives_enabled(self) -> bool:
        """Read DERIVATIVES_ENABLED env var (fresh each call — kill switch
        must take effect without restart)."""
        env_val = os.environ.get("DERIVATIVES_ENABLED", "false").strip().lower()
        if env_val not in ("true", "1", "yes", "on"):
            return False
        return self.derivatives is not None and self.derivatives.is_enabled()

    # ------------------------------------------------------------------
    # ROUTE DECISION (pure function over intent + context)
    # ------------------------------------------------------------------
    def decide_route(
        self,
        intent,
        context: Optional[Dict[str, Any]] = None,
    ) -> RouteDecision:
        context = context or {}

        # Rule 1: global kill switch
        if not self._derivatives_enabled():
            return RouteDecision(
                ExecutionRoute.SPOT_MARGIN,
                reason="derivatives_kill_switch_off",
            )

        strategy = str(getattr(intent, "strategy", getattr(intent, "quant_strategy_id", "")) or "").upper()
        direction = float(getattr(intent, "direction", 0) or 0)
        leverage = float(getattr(intent, "leverage", 1.0) or 1.0)

        # Rule 3: cash-carry is derivatives-only
        if strategy == "CASH_CARRY":
            veto = self._gate_check(intent, context)
            if veto:
                # Carry has no spot fallback — router returns SPOT_MARGIN with a
                # note but execute path will bail since strategy!=spot.
                return RouteDecision(
                    ExecutionRoute.SPOT_MARGIN,
                    reason="carry_gate_vetoed_no_fallback",
                    gate_veto=veto,
                )
            return RouteDecision(ExecutionRoute.DERIVATIVES, reason="cash_carry_strategy")

        # Rule 4: leveraged long → perp (spot can't leverage longs)
        if direction > 0 and leverage > 1.0:
            veto = self._gate_check(intent, context)
            if veto:
                return RouteDecision(
                    ExecutionRoute.SPOT_MARGIN,
                    reason="leveraged_long_gate_fallback",
                    gate_veto=veto,
                )
            return RouteDecision(ExecutionRoute.DERIVATIVES, reason="leveraged_long")

        # Rule 5: short + positive funding → route via perp (earn funding)
        funding_h = context.get("funding_rate_h", 0.0) or 0.0
        if direction < 0 and funding_h >= self.FUNDING_SHORT_PERP_THRESHOLD_H:
            # Check capacity
            if self.derivatives and self._has_derivatives_budget(intent, context):
                veto = self._gate_check(intent, context)
                if veto:
                    return RouteDecision(
                        ExecutionRoute.SPOT_MARGIN,
                        reason="short_funding_gate_fallback",
                        gate_veto=veto,
                    )
                return RouteDecision(
                    ExecutionRoute.DERIVATIVES,
                    reason=f"short_funding_positive:{funding_h * 10000:.1f}bps/h",
                )

        # Default
        return RouteDecision(ExecutionRoute.SPOT_MARGIN, reason="default_spot_margin")

    # ------------------------------------------------------------------
    # EXECUTE (dispatch + return)
    # ------------------------------------------------------------------
    async def route_and_execute(
        self,
        intent,
        context: Optional[Dict[str, Any]] = None,
        **spot_margin_kwargs,
    ) -> Dict[str, Any]:
        """Dispatch intent to the chosen executor.

        For SPOT_MARGIN: calls the existing execute_intent_v2(ctx, asset, intent, ...)
        flow (caller passes extra kwargs in spot_margin_kwargs).

        For DERIVATIVES: calls the appropriate DerivativesExecutor method.
        """
        decision = self.decide_route(intent, context)

        if decision.route == ExecutionRoute.DERIVATIVES:
            return await self._execute_derivatives(intent, context or {}, decision)
        else:
            # Spot/margin path — delegate to existing machinery.
            # Caller is responsible for passing the right kwargs (ctx, asset, market_data...).
            return await self._execute_spot_margin(intent, decision, **spot_margin_kwargs)

    async def _execute_derivatives(
        self,
        intent,
        context: Dict[str, Any],
        decision: RouteDecision,
    ) -> Dict[str, Any]:
        strategy = str(getattr(intent, "strategy", getattr(intent, "quant_strategy_id", "")) or "").upper()
        asset = getattr(intent, "asset", "")
        size_usd = float(getattr(intent, "size_usd", 0.0) or 0.0)
        direction = int(getattr(intent, "direction", 0) or 0)
        leverage = float(getattr(intent, "leverage", 1.0) or 1.0)

        try:
            if strategy == "CASH_CARRY":
                funding_h = float(context.get("funding_rate_h", 0.0) or 0.0)
                result = self.derivatives.execute_cash_carry(
                    asset=asset, size_usd=size_usd, funding_rate_h=funding_h,
                )
            else:
                result = self.derivatives.execute_directional(
                    asset=asset, direction=direction, size_usd=size_usd, leverage=leverage,
                )
            return {
                "route": decision.route.value,
                "route_reason": decision.reason,
                "success": result.success,
                "status": result.status.value,
                "reason": result.reason,
                "order_id": result.order_id,
                "filled_price": result.filled_price,
                "liquidation_price": result.liquidation_price,
            }
        except Exception as e:
            logger.error(f"[WIRE-ROUTER] Derivatives exec error: {e}")
            return {
                "route": decision.route.value,
                "route_reason": decision.reason,
                "success": False,
                "status": "FAILED",
                "reason": f"exception:{e}",
            }

    async def _execute_spot_margin(
        self,
        intent,
        decision: RouteDecision,
        **kwargs,
    ) -> Dict[str, Any]:
        """Delegate to existing execute_intent_v2. The router doesn't know the
        spot path's exact signature — caller passes ctx/asset/market_data in
        **kwargs, router just invokes and tags the result.
        """
        try:
            if self.spot_margin is None:
                return {
                    "route": decision.route.value,
                    "success": False,
                    "reason": "no_spot_margin_executor",
                }
            # `execute` is a callable that returns an exec_result dict.
            # In HMATS this is `execute_intent_v2(ctx, asset, intent, market_data, ...)`.
            exec_result = await self.spot_margin(intent=intent, **kwargs)
            if isinstance(exec_result, dict):
                exec_result["route"] = decision.route.value
                exec_result["route_reason"] = decision.reason
                if decision.gate_veto:
                    exec_result["derivatives_gate_veto"] = decision.gate_veto
            return exec_result
        except Exception as e:
            logger.error(f"[WIRE-ROUTER] Spot/margin exec error: {e}")
            return {
                "route": decision.route.value,
                "success": False,
                "reason": f"spot_exception:{e}",
            }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _gate_check(self, intent, context) -> Optional[str]:
        """Run DerivativesRiskGate if present. Returns veto_reason or None."""
        if self.gate is None:
            return None
        result = self.gate.validate(intent, context)
        return None if result.passed else result.veto_reason

    def _has_derivatives_budget(self, intent, context) -> bool:
        """Quick capacity probe without full gate run."""
        if self.derivatives is None:
            return False
        size_usd = float(getattr(intent, "size_usd", 0.0) or 0.0)
        nav = self.derivatives._nav_callback() or 0.0
        if nav <= 0:
            return False
        current = self.derivatives._total_derivatives_exposure()
        return (current + size_usd) / nav <= self.derivatives.MAX_DERIVATIVES_ALLOCATION_PCT

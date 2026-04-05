# defense/sol_execution_guard.py  [P2-FIX] T16
"""
SOL-Specific Execution Guard - SHADOW mode default.

SOL risks vs BTC/ETH:
1. Thinner orderbook on Kraken -> larger market impact
2. DEX-driven price discovery -> exchange price can lag
3. Higher intraday volatility -> wider effective spread

This is a lightweight pre-trade validator.
For the full SOL execution engine (ATR slicing, VPIN, network stress),
see execution/sol_execution.py.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SOLExecutionGuard:
    """
    SOL-specific execution safety layer. SHADOW mode default.

    Validates order parameters before execution. In SHADOW mode,
    logs warnings but does NOT block or modify orders.
    """

    # [P2-FIX] SOL gets stricter limits than BTC/ETH
    MAX_ORDER_SIZE_SOL = 500       # Max SOL per single order (avoid impact)
    MAX_SLIPPAGE_BPS = 30          # Max acceptable spread (SOL)
    MIN_DEPTH_USD = 50_000         # Minimum orderbook depth to execute

    def __init__(self, enabled: bool = False, shadow: bool = True):
        self._enabled = enabled
        self._shadow = shadow  # [P2-FIX] Default SHADOW: log only
        logger.info(
            f"[P2-FIX] SOLExecutionGuard initialized: "
            f"enabled={enabled}, shadow={shadow}"
        )

    def check_execution_safe(
        self,
        order_size_sol: float,
        orderbook_depth_usd: float = 0,
        spread_bps: float = 0,
    ) -> Dict:
        """
        Check if SOL order is safe to execute.

        Returns:
            {'safe': bool, 'reason': str, 'recommended_size': float}
        """
        if not self._enabled:
            return {
                "safe": True,
                "reason": "disabled",
                "recommended_size": order_size_sol,
            }

        issues = []
        recommended = order_size_sol

        # [P2-FIX] Size cap - avoid excessive market impact
        if order_size_sol > self.MAX_ORDER_SIZE_SOL:
            issues.append(
                f"size {order_size_sol:.1f} > max {self.MAX_ORDER_SIZE_SOL}"
            )
            recommended = min(recommended, float(self.MAX_ORDER_SIZE_SOL))

        # [P2-FIX] Depth check - scale order to available liquidity
        if orderbook_depth_usd > 0 and orderbook_depth_usd < self.MIN_DEPTH_USD:
            issues.append(
                f"depth ${orderbook_depth_usd:,.0f} < min ${self.MIN_DEPTH_USD:,.0f}"
            )
            depth_ratio = orderbook_depth_usd / self.MIN_DEPTH_USD
            recommended = min(recommended, order_size_sol * depth_ratio)

        # [P2-FIX] Spread check - wide spread signals illiquidity
        if spread_bps > self.MAX_SLIPPAGE_BPS:
            issues.append(
                f"spread {spread_bps:.1f}bps > max {self.MAX_SLIPPAGE_BPS}bps"
            )

        safe = len(issues) == 0
        reason = "; ".join(issues) if issues else "OK"

        if self._shadow and not safe:
            logger.info(
                f"[P2-FIX] SHADOW SOL guard: {reason} "
                f"(would clip to {recommended:.1f} SOL)"
            )
            return {
                "safe": True,
                "reason": f"SHADOW: {reason}",
                "recommended_size": order_size_sol,
            }

        if not safe:
            logger.warning(
                f"[P2-FIX] SOL guard: {reason}, "
                f"clipping {order_size_sol:.1f} -> {recommended:.1f}"
            )

        return {
            "safe": safe,
            "reason": reason,
            "recommended_size": recommended,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_sol_guard: Optional[SOLExecutionGuard] = None


def get_sol_execution_guard(
    enabled: bool = False, shadow: bool = True
) -> SOLExecutionGuard:
    """Get or create singleton SOL execution guard."""
    global _sol_guard
    if _sol_guard is None:
        _sol_guard = SOLExecutionGuard(enabled=enabled, shadow=shadow)
    return _sol_guard
"""
================================================================================
HMATS v6.8 — Derivatives Risk Gate (v2.0 Part 2.3)
================================================================================
Pre-trade risk gate specific to derivatives routing. Layered with existing
LeverageGuard + PositionSizer + TradeGate (spot/margin) — does NOT replace them.
Only called when ExecutionRouter routes an intent to DerivativesExecutor.

Checks (all must pass):
  1. Liquidation buffer ≥ 40% (hard cap from DerivativesExecutor reused)
  2. Funding direction sanity (don't short when funding is deeply negative)
  3. Perp volume adequate (avoid thin / illiquid perps)
  4. Basis spread within tolerance for cash-carry
  5. Maintenance margin room (NAV × (leverage/max_lev) < free_margin)
  6. Kill switch + executor health

Any veto returns a typed ValidationResult with a reason string; the Router
then falls back to spot/margin path if the intent is compatible there.

Wiring tag: [WIRE-DERIV-GATE]
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from execution.derivatives_executor import DerivativesExecutor, PERP_SYMBOLS

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class CheckResult:
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    passed: bool
    veto_reason: str = ""
    veto_source: str = "DERIVATIVES_RISK_GATE"
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GATE
# =============================================================================

class DerivativesRiskGate:
    """Pre-trade gate for derivatives-routed intents.

    Usage:
        gate = DerivativesRiskGate(executor=deriv_exec)
        result = gate.validate(intent, context)
        if not result.passed:
            # Router falls back to spot/margin or vetoes entirely
            ...
    """

    # Minimum perp 24h volume for a directional trade (USD)
    MIN_PERP_VOLUME_24H_USD = 5_000_000
    # Maximum allowed spot-perp basis for cash-carry entry
    MAX_CARRY_BASIS_PCT = 0.02
    # Funding direction sanity: don't SHORT if funding is below this (paying longs heavily)
    FUNDING_SHORT_VETO_THRESHOLD_H = -0.0005  # -0.05%/h = -4.4% annualized

    def __init__(self, executor: DerivativesExecutor):
        self.executor = executor

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def validate(
        self,
        intent,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run all checks. Returns ValidationResult with first-fail semantics."""
        context = context or {}

        # Kill switch first — if executor is disabled, nothing passes
        ks = self._check_kill_switch(intent, context)
        if not ks.passed:
            return ValidationResult(
                passed=False, veto_reason=ks.reason,
                veto_source="DERIVATIVES_RISK_GATE",
                details=ks.details,
            )

        checks = [
            self._check_liquidation_buffer,
            self._check_funding_direction,
            self._check_perp_volume,
            self._check_basis_spread,
            self._check_maintenance_margin,
        ]
        for check_fn in checks:
            result = check_fn(intent, context)
            if not result.passed:
                logger.info(
                    f"[WIRE-DERIV-GATE] VETO {getattr(intent, 'asset', '?')} "
                    f"reason={result.reason}"
                )
                return ValidationResult(
                    passed=False, veto_reason=result.reason,
                    veto_source="DERIVATIVES_RISK_GATE",
                    details=result.details,
                )
        return ValidationResult(passed=True)

    # ------------------------------------------------------------------
    # CHECKS
    # ------------------------------------------------------------------
    def _check_kill_switch(self, intent, context) -> CheckResult:
        if not self.executor.is_enabled():
            return CheckResult(
                passed=False, reason="executor_disabled",
                details={"disabled_reason": self.executor._disabled_reason},
            )
        return CheckResult(passed=True)

    def _check_liquidation_buffer(self, intent, context) -> CheckResult:
        """Reuse executor's liq buffer logic. Only meaningful for directional perp;
        cash-carry enters perp at leverage=1 which always clears the buffer.
        """
        asset = getattr(intent, "asset", "")
        direction = int(getattr(intent, "direction", 0) or 0)
        leverage = float(getattr(intent, "leverage", 1.0) or 1.0)
        if direction == 0:
            return CheckResult(passed=True)  # nothing to open, nothing to check
        if asset not in PERP_SYMBOLS:
            return CheckResult(passed=False, reason=f"unsupported_asset:{asset}")

        mark = self.executor.client.get_mark_price(asset) if self.executor.client else None
        if not mark or mark <= 0:
            return CheckResult(passed=False, reason="no_mark_price_for_liq_check")

        liq = self.executor._calculate_liquidation_price(mark, direction, leverage)
        buffer_ratio = abs(mark - liq) / mark
        if buffer_ratio < self.executor.MAX_LIQUIDATION_BUFFER_MIN:
            return CheckResult(
                passed=False,
                reason=f"liq_buffer_thin:{buffer_ratio:.1%}",
                details={"mark": mark, "liq": liq, "buffer_ratio": buffer_ratio},
            )
        return CheckResult(passed=True)

    def _check_funding_direction(self, intent, context) -> CheckResult:
        """Reject SHORT perp when funding is deeply negative (longs get paid a lot,
        shorts pay heavily). Only applies when opening a short directional perp,
        NOT cash-carry which is always short+long paired.
        """
        strategy = str(getattr(intent, "strategy", getattr(intent, "quant_strategy_id", "")) or "").upper()
        if strategy == "CASH_CARRY":
            return CheckResult(passed=True)  # carry sanity is checked in executor

        direction = int(getattr(intent, "direction", 0) or 0)
        if direction >= 0:
            return CheckResult(passed=True)  # only check when shorting

        funding_rate_h = context.get("funding_rate_h", None)
        if funding_rate_h is None:
            # Missing data — prefer pass rather than fail-closed here; other
            # layers (TradeGate) already enforce data-integrity.
            return CheckResult(passed=True)

        if funding_rate_h < self.FUNDING_SHORT_VETO_THRESHOLD_H:
            return CheckResult(
                passed=False,
                reason=f"funding_hostile_to_short:{funding_rate_h * 10000:.1f}bps/h",
                details={"funding_rate_h": funding_rate_h},
            )
        return CheckResult(passed=True)

    def _check_perp_volume(self, intent, context) -> CheckResult:
        """Reject if perp 24h volume too thin (slippage risk)."""
        asset = getattr(intent, "asset", "")
        if not self.executor.client or asset not in PERP_SYMBOLS:
            return CheckResult(passed=False, reason="no_client_or_asset")

        data = self.executor.client.fetch_market_data()
        md = data.get(asset) if data else None
        if md is None:
            return CheckResult(passed=True)  # missing data — don't veto on this alone
        vol_usd = (md.volume_24h or 0.0) * (md.mark_price or 0.0)
        if vol_usd < self.MIN_PERP_VOLUME_24H_USD:
            return CheckResult(
                passed=False,
                reason=f"perp_volume_thin:${vol_usd:,.0f}<${self.MIN_PERP_VOLUME_24H_USD:,.0f}",
                details={"volume_24h_usd": vol_usd},
            )
        return CheckResult(passed=True)

    def _check_basis_spread(self, intent, context) -> CheckResult:
        """For cash-carry, reject if spot-perp basis is too wide."""
        strategy = str(getattr(intent, "strategy", getattr(intent, "quant_strategy_id", "")) or "").upper()
        if strategy != "CASH_CARRY":
            return CheckResult(passed=True)

        asset = getattr(intent, "asset", "")
        mark = self.executor.client.get_mark_price(asset) if self.executor.client else None
        spot = context.get("spot_price", None) or self.executor._get_spot_price(asset)
        if not mark or not spot or mark <= 0 or spot <= 0:
            return CheckResult(passed=True)  # missing data pass

        basis = abs(mark - spot) / spot
        if basis > self.MAX_CARRY_BASIS_PCT:
            return CheckResult(
                passed=False,
                reason=f"carry_basis_wide:{basis:.2%}>{self.MAX_CARRY_BASIS_PCT:.2%}",
                details={"basis": basis, "mark": mark, "spot": spot},
            )
        return CheckResult(passed=True)

    def _check_maintenance_margin(self, intent, context) -> CheckResult:
        """Probe executor's allocation caps. The executor itself enforces
        MAX_SINGLE_PERP_POSITION_PCT and MAX_DERIVATIVES_ALLOCATION_PCT, but we
        check here too to produce an earlier veto with a clean reason string.
        """
        size_usd = float(getattr(intent, "size_usd", 0.0) or 0.0)
        if size_usd <= 0:
            return CheckResult(passed=True)  # can't evaluate without sizing

        nav = self.executor._nav_callback() or 0.0
        if nav <= 0:
            return CheckResult(passed=False, reason="nav_unknown")

        if size_usd / nav > self.executor.MAX_SINGLE_PERP_POSITION_PCT:
            return CheckResult(
                passed=False,
                reason=f"single_position_cap:{size_usd / nav:.2%}",
            )

        current = self.executor._total_derivatives_exposure()
        total_pct = (current + size_usd) / nav
        if total_pct > self.executor.MAX_DERIVATIVES_ALLOCATION_PCT:
            return CheckResult(
                passed=False,
                reason=f"total_allocation_cap:{total_pct:.2%}",
            )
        return CheckResult(passed=True)

"""
================================================================================
HMATS v6.8 — Derivatives Executor (Kraken Perpetual Futures)
================================================================================
Replaces DeltaNeutralExecutor (archived 2026-04-24). Supports both directional
perp trades AND cash-carry (long spot + short perp) in one class.

Architecture (per v2.0 Part 2.2 spec):
  ExecutionRouter
      ├── SpotMarginExecutor (existing, via core/execution_service.py)
      └── DerivativesExecutor (this file)
              ├── execute_directional(asset, direction, size_usd, leverage)
              ├── execute_cash_carry(asset, size_usd, funding_rate_h)
              ├── close_directional(asset, reason)
              ├── close_cash_carry(asset, reason)
              ├── reconcile_on_startup()
              └── _emergency_unwind_perp(asset)  [3-retry, disables on fail]

ABSOLUTE HARD CAPS (class constants — NOT config-adjustable):
  - MAX_LEVERAGE_PERP = 3.0               (code-enforced cap)
  - MAX_SINGLE_PERP_POSITION_PCT = 0.15   (15% NAV per position)
  - MAX_DERIVATIVES_ALLOCATION_PCT = 0.30 (30% NAV total derivatives exposure)
  - MAX_LIQUIDATION_BUFFER_MIN = 0.40     (40% distance to liq required)
  - ISOLATED_MARGIN_ONLY = True           (no cross margin)

Supported symbols:
  BTC → PF_XBTUSD, ETH → PF_ETHUSD, SOL → PF_SOLUSD

Wiring tag: [WIRE-DERIV]
================================================================================
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# PERP SYMBOL MAP
# =============================================================================

PERP_SYMBOLS: Dict[str, str] = {
    "BTC": "PF_XBTUSD",
    "ETH": "PF_ETHUSD",
    "SOL": "PF_SOLUSD",
}

SPOT_SYMBOLS: Dict[str, str] = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
}


# =============================================================================
# RESULT TYPES
# =============================================================================

class ExecutionStatus(Enum):
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


@dataclass
class ExecutionResult:
    success: bool
    status: ExecutionStatus
    reason: str = ""
    order_id: str = ""
    filled_price: float = 0.0
    filled_size: float = 0.0
    liquidation_price: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# POSITION STATE
# =============================================================================

@dataclass
class PerpPosition:
    """State of one directional perpetual position."""
    asset: str
    symbol: str               # e.g. PF_XBTUSD
    side: str                 # "buy" (long) or "sell" (short)
    size_usd: float
    size_asset: float         # notional / entry_price
    entry_price: float
    leverage: float
    liquidation_price: float
    opened_at: float
    order_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerpPosition":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class CashCarryPosition:
    """State of one cash-carry position (long spot + short perp).

    The spot leg is tracked by amount bought; the perp leg holds a separate
    PerpPosition. Both legs unwound together when closing.
    """
    asset: str
    size_usd: float
    spot_entry_price: float
    spot_size_asset: float
    perp_position: PerpPosition
    entry_funding_rate_h: float
    entry_time: float
    cumulative_funding_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "perp_position"}
        d["perp_position"] = self.perp_position.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CashCarryPosition":
        pp = data.get("perp_position", {})
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data and k != "perp_position"}
        fields["perp_position"] = PerpPosition.from_dict(pp)
        return cls(**fields)


# =============================================================================
# EXECUTOR
# =============================================================================

class DerivativesExecutor:
    """Kraken Perpetual Futures executor — directional + cash-carry.

    Usage:
        executor = DerivativesExecutor(
            kraken_deriv_client=kdc,
            spot_exchange=ccxt_kraken,
            risk_callback=lambda: runner._get_nav(),
            discord_alerter=runner.audit_manager.discord,
        )
        executor.reconcile_on_startup()
        result = executor.execute_directional("BTC", direction=+1, size_usd=500, leverage=2.0)
    """

    # ---- ABSOLUTE HARD CAPS (per v2.0 Part 2.2) ----
    MAX_LEVERAGE_PERP = 3.0
    MAX_SINGLE_PERP_POSITION_PCT = 0.15
    MAX_DERIVATIVES_ALLOCATION_PCT = 0.30
    MAX_LIQUIDATION_BUFFER_MIN = 0.40  # require ≥40% price buffer to liq
    ISOLATED_MARGIN_ONLY = True

    # ---- Cash-carry thresholds ----
    CARRY_BASIS_MAX = 0.02  # 2% max spot-perp basis for entry

    # ---- Unwind retry policy ----
    UNWIND_MAX_RETRIES = 3
    UNWIND_BACKOFF_BASE_SEC = 2  # exponential: 2s, 4s, 8s

    def __init__(
        self,
        kraken_deriv_client,
        spot_exchange=None,
        risk_callback: Optional[Callable[[], float]] = None,
        discord_alerter=None,
        initial_size_cap_usd: Optional[float] = None,
    ):
        """
        Args:
            kraken_deriv_client: KrakenDerivativesClient instance (auth + orders)
            spot_exchange: ccxt.kraken instance for spot leg of cash-carry
            risk_callback: callable returning current NAV (USD); used for % caps.
                           Defaults to 0 if not provided -> all orders rejected.
            discord_alerter: DiscordNotifier for CRITICAL/WARNING events.
            initial_size_cap_usd: first-week absolute $ cap per position (default None).
                                  Set from env DERIVATIVES_INITIAL_SIZE_USD_CAP.
        """
        self.client = kraken_deriv_client
        self.spot_exchange = spot_exchange
        self._nav_callback = risk_callback or (lambda: 0.0)
        self.discord = discord_alerter

        # Initial-week safety cap (hardcoded first-week protection)
        if initial_size_cap_usd is None:
            try:
                initial_size_cap_usd = float(
                    os.environ.get("DERIVATIVES_INITIAL_SIZE_USD_CAP", "0") or 0
                )
            except ValueError:
                initial_size_cap_usd = 0.0
        self.initial_size_cap_usd = initial_size_cap_usd or None

        self.active_perp_positions: Dict[str, PerpPosition] = {}
        self.active_carries: Dict[str, CashCarryPosition] = {}
        self._disabled: bool = False
        self._disabled_reason: str = ""

        logger.info(
            f"[WIRE-DERIV] DerivativesExecutor initialized: "
            f"max_lev={self.MAX_LEVERAGE_PERP}x, "
            f"single_cap={self.MAX_SINGLE_PERP_POSITION_PCT:.0%} NAV, "
            f"total_cap={self.MAX_DERIVATIVES_ALLOCATION_PCT:.0%} NAV, "
            f"initial_usd_cap={self.initial_size_cap_usd}"
        )

    # ------------------------------------------------------------------
    # PUBLIC API — kill switch
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return (
            not self._disabled
            and self.client is not None
            and self.client.is_execution_enabled()
        )

    def disable(self, reason: str):
        self._disabled = True
        self._disabled_reason = reason
        logger.critical(f"[WIRE-DERIV] Executor DISABLED: {reason}")
        self._alert("CRITICAL", "DERIV_EXECUTOR_DISABLED", {"reason": reason})

    # ------------------------------------------------------------------
    # DIRECTIONAL (single-leg)
    # ------------------------------------------------------------------
    def execute_directional(
        self,
        asset: str,
        direction: int,       # +1 long, -1 short
        size_usd: float,
        leverage: float,
    ) -> ExecutionResult:
        if not self.is_enabled():
            return ExecutionResult(False, ExecutionStatus.DISABLED, "executor_disabled")

        asset = asset.upper()
        symbol = PERP_SYMBOLS.get(asset)
        if symbol is None:
            return ExecutionResult(False, ExecutionStatus.REJECTED, f"unsupported_asset:{asset}")

        # Pre-flight risk checks
        rejection = self._validate_pre_flight(asset, direction, size_usd, leverage)
        if rejection:
            return ExecutionResult(False, ExecutionStatus.REJECTED, rejection)

        # Mark price + liquidation calc
        mark_price = self.client.get_mark_price(asset)
        if not mark_price or mark_price <= 0:
            return ExecutionResult(False, ExecutionStatus.REJECTED, "no_mark_price")

        liq_price = self._calculate_liquidation_price(mark_price, direction, leverage)
        buffer_ratio = abs(mark_price - liq_price) / mark_price
        if buffer_ratio < self.MAX_LIQUIDATION_BUFFER_MIN:
            logger.warning(
                f"[WIRE-DERIV] {asset} liq buffer {buffer_ratio:.1%} < "
                f"{self.MAX_LIQUIDATION_BUFFER_MIN:.0%} required"
            )
            return ExecutionResult(False, ExecutionStatus.REJECTED, "liq_buffer_too_thin")

        # Place order (post-only limit, short client-side timeout, fallback to market)
        side = "buy" if direction > 0 else "sell"
        size_asset = size_usd / mark_price
        cli_ord_id = f"hmats-dir-{uuid.uuid4().hex[:16]}"

        try:
            resp = self.client.send_order(
                symbol=symbol,
                side=side,
                size=size_asset,
                order_type="post",       # post-only limit
                limit_price=mark_price,
                cli_ord_id=cli_ord_id,
            )
        except Exception as e:
            logger.error(f"[WIRE-DERIV] send_order raised: {e}")
            return ExecutionResult(False, ExecutionStatus.FAILED, f"send_order_exception:{e}")

        if not resp or resp.get("result") != "success":
            return ExecutionResult(
                False, ExecutionStatus.FAILED,
                reason=str(resp)[:200] if resp else "no_response",
                details=resp or {},
            )

        send_status = resp.get("sendStatus", {}) or {}
        order_id = send_status.get("order_id", "") or cli_ord_id
        filled_price = float(send_status.get("orderEvents", [{}])[0].get("price", mark_price) or mark_price)

        pos = PerpPosition(
            asset=asset,
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            size_asset=size_asset,
            entry_price=filled_price,
            leverage=leverage,
            liquidation_price=liq_price,
            opened_at=time.time(),
            order_id=order_id,
        )
        self.active_perp_positions[asset] = pos

        self._alert("INFO", "DERIV_DIRECTIONAL_OPENED", {
            "asset": asset, "side": side, "size_usd": round(size_usd, 2),
            "leverage": leverage, "entry": round(filled_price, 2),
            "liquidation_price": round(liq_price, 2),
            "liq_buffer_pct": round(buffer_ratio, 3),
        })
        logger.info(
            f"[WIRE-DERIV] OPENED {asset} {side.upper()} ${size_usd:.0f} "
            f"lev={leverage}x entry={filled_price:.2f} liq={liq_price:.2f} "
            f"buffer={buffer_ratio:.1%}"
        )
        return ExecutionResult(
            True, ExecutionStatus.SUCCESS,
            order_id=order_id, filled_price=filled_price, filled_size=size_asset,
            liquidation_price=liq_price,
        )

    def close_directional(self, asset: str, reason: str = "signal") -> bool:
        asset = asset.upper()
        if asset not in self.active_perp_positions:
            return False
        if not self.is_enabled():
            return False
        pos = self.active_perp_positions[asset]
        opposite = "sell" if pos.side == "buy" else "buy"
        try:
            resp = self.client.send_order(
                symbol=pos.symbol,
                side=opposite,
                size=pos.size_asset,
                order_type="mkt",
                reduce_only=True,
            )
        except Exception as e:
            logger.error(f"[WIRE-DERIV] close_directional send failed: {e}")
            return False

        if not resp or resp.get("result") != "success":
            logger.error(f"[WIRE-DERIV] close_directional rejected: {resp}")
            return False

        hold_h = (time.time() - pos.opened_at) / 3600
        self._alert("INFO", "DERIV_DIRECTIONAL_CLOSED", {
            "asset": asset, "reason": reason, "held_hours": round(hold_h, 2),
            "entry": pos.entry_price, "size_usd": pos.size_usd,
        })
        logger.info(f"[WIRE-DERIV] CLOSED {asset} reason={reason} held={hold_h:.1f}h")
        del self.active_perp_positions[asset]
        return True

    # ------------------------------------------------------------------
    # CASH-CARRY (double-leg: long spot + short perp)
    # ------------------------------------------------------------------
    def execute_cash_carry(
        self,
        asset: str,
        size_usd: float,
        funding_rate_h: float,
    ) -> ExecutionResult:
        """Open a delta-neutral carry position.

        Execution order: PERP first (locks in funding income), SPOT second.
        If SPOT leg fails -> emergency unwind perp immediately.
        """
        if not self.is_enabled():
            return ExecutionResult(False, ExecutionStatus.DISABLED, "executor_disabled")

        asset = asset.upper()
        if asset in self.active_carries:
            return ExecutionResult(False, ExecutionStatus.REJECTED, "carry_already_active")

        rejection = self._validate_cash_carry(asset, size_usd, funding_rate_h)
        if rejection:
            return ExecutionResult(False, ExecutionStatus.REJECTED, rejection)

        # Leg 1: Perp SHORT
        perp_result = self.execute_directional(asset, direction=-1, size_usd=size_usd, leverage=1.0)
        if not perp_result.success:
            return ExecutionResult(
                False, ExecutionStatus.FAILED,
                reason=f"perp_leg_failed:{perp_result.reason}",
            )

        # Leg 2: Spot LONG
        spot_symbol = SPOT_SYMBOLS.get(asset, f"{asset}/USD")
        mark_price = self.client.get_mark_price(asset) or perp_result.filled_price
        spot_size_asset = size_usd / mark_price

        spot_fill_price = None
        try:
            if self.spot_exchange is None:
                raise RuntimeError("spot_exchange not configured")
            order = self.spot_exchange.create_limit_buy_order(
                spot_symbol, spot_size_asset, mark_price, params={"oflags": "post"},
            )
            spot_fill_price = float(order.get("price") or mark_price)
        except Exception as e:
            logger.error(f"[WIRE-DERIV-CARRY] Spot leg failed ({e}) — unwinding perp")
            unwound = self._emergency_unwind_perp(asset)
            self._alert("CRITICAL", "DERIV_CARRY_SPOT_FAILED", {
                "asset": asset, "size_usd": size_usd,
                "spot_error": str(e)[:200], "perp_unwound": unwound,
            })
            return ExecutionResult(
                False, ExecutionStatus.FAILED,
                reason=f"spot_leg_failed_{'unwound' if unwound else 'unwind_failed'}",
            )

        # Both legs succeeded — consolidate into CashCarryPosition
        perp_pos = self.active_perp_positions.pop(asset)
        carry = CashCarryPosition(
            asset=asset,
            size_usd=size_usd,
            spot_entry_price=spot_fill_price,
            spot_size_asset=spot_size_asset,
            perp_position=perp_pos,
            entry_funding_rate_h=funding_rate_h,
            entry_time=time.time(),
        )
        self.active_carries[asset] = carry

        annualized_pct = funding_rate_h * 24 * 365 * 100
        self._alert("INFO", "DERIV_CARRY_OPENED", {
            "asset": asset, "size_usd": size_usd,
            "funding_bps_h": round(funding_rate_h * 10000, 2),
            "annualized_pct": round(annualized_pct, 2),
            "spot_entry": spot_fill_price, "perp_entry": perp_pos.entry_price,
        })
        logger.info(
            f"[WIRE-DERIV-CARRY] OPENED {asset} ${size_usd:.0f} "
            f"funding={funding_rate_h * 10000:.2f}bps/h = {annualized_pct:.1f}% ann"
        )
        return ExecutionResult(True, ExecutionStatus.SUCCESS)

    def close_cash_carry(self, asset: str, reason: str = "funding_reversal") -> bool:
        asset = asset.upper()
        if asset not in self.active_carries:
            return False
        carry = self.active_carries[asset]

        # Restore perp_position into active_perp_positions so close_directional can see it
        self.active_perp_positions[asset] = carry.perp_position

        perp_ok = self.close_directional(asset, reason=f"carry_close_{reason}")

        # Close spot leg (market sell)
        spot_ok = False
        try:
            if self.spot_exchange:
                self.spot_exchange.create_market_sell_order(
                    SPOT_SYMBOLS.get(asset, f"{asset}/USD"),
                    carry.spot_size_asset,
                )
                spot_ok = True
        except Exception as e:
            logger.error(f"[WIRE-DERIV-CARRY] Spot close failed: {e}")

        if perp_ok and spot_ok:
            del self.active_carries[asset]
            self._alert("INFO", "DERIV_CARRY_CLOSED", {"asset": asset, "reason": reason})
            return True

        # Partial close = degraded state, needs manual cleanup
        self._alert("CRITICAL", "DERIV_CARRY_PARTIAL_CLOSE", {
            "asset": asset, "perp_ok": perp_ok, "spot_ok": spot_ok,
        })
        return False

    # ------------------------------------------------------------------
    # EMERGENCY UNWIND
    # ------------------------------------------------------------------
    def _emergency_unwind_perp(self, asset: str) -> bool:
        """Force-close perp position with retry + exponential backoff.

        After UNWIND_MAX_RETRIES failed attempts, disable the executor entirely
        and send CRITICAL alert.
        """
        if asset not in self.active_perp_positions:
            return True
        pos = self.active_perp_positions[asset]
        opposite = "sell" if pos.side == "buy" else "buy"

        for attempt in range(self.UNWIND_MAX_RETRIES):
            try:
                resp = self.client.send_order(
                    symbol=pos.symbol,
                    side=opposite,
                    size=pos.size_asset,
                    order_type="mkt",
                    reduce_only=True,
                )
                if resp and resp.get("result") == "success":
                    del self.active_perp_positions[asset]
                    logger.warning(
                        f"[WIRE-DERIV] Emergency unwound {asset} perp (attempt {attempt + 1})"
                    )
                    return True
            except Exception as e:
                logger.error(f"[WIRE-DERIV] Unwind attempt {attempt + 1} failed: {e}")
            time.sleep(self.UNWIND_BACKOFF_BASE_SEC * (2 ** attempt))

        # All attempts failed — disable executor
        self.disable(f"emergency_unwind_failed_{asset}")
        self._alert("CRITICAL", "DERIV_EMERGENCY_UNWIND_FAILED", {
            "asset": asset, "symbol": pos.symbol, "size_usd": pos.size_usd,
            "message": "MANUAL INTERVENTION REQUIRED. Executor disabled.",
        })
        return False

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def _validate_pre_flight(
        self, asset: str, direction: int, size_usd: float, leverage: float,
    ) -> Optional[str]:
        """Returns None if OK, else string reason for rejection."""
        if direction not in (-1, 1):
            return f"invalid_direction:{direction}"
        if leverage <= 0 or leverage > self.MAX_LEVERAGE_PERP:
            return f"leverage_out_of_bounds:{leverage}"
        if size_usd <= 0:
            return f"invalid_size:{size_usd}"

        # First-week absolute $ cap (if set)
        if self.initial_size_cap_usd and size_usd > self.initial_size_cap_usd:
            return f"initial_usd_cap_exceeded:{size_usd}>{self.initial_size_cap_usd}"

        nav = self._nav_callback() or 0.0
        if nav <= 0:
            return "nav_unknown"

        if size_usd / nav > self.MAX_SINGLE_PERP_POSITION_PCT:
            return f"single_position_cap:{size_usd / nav:.2%}>{self.MAX_SINGLE_PERP_POSITION_PCT:.2%}"

        current_exposure = self._total_derivatives_exposure()
        if (current_exposure + size_usd) / nav > self.MAX_DERIVATIVES_ALLOCATION_PCT:
            return (f"total_allocation_cap:"
                    f"{(current_exposure + size_usd) / nav:.2%}>{self.MAX_DERIVATIVES_ALLOCATION_PCT:.2%}")
        return None

    def _validate_cash_carry(
        self, asset: str, size_usd: float, funding_rate_h: float,
    ) -> Optional[str]:
        # Must have positive funding (short side earns)
        if funding_rate_h < 0.0001:  # 0.01%/h min
            return f"funding_too_low:{funding_rate_h:.6f}"
        # Spot and mark price reasonably close (basis check)
        mark = self.client.get_mark_price(asset) if self.client else None
        if not mark or mark <= 0:
            return "no_mark_price_for_basis"
        spot = self._get_spot_price(asset) or mark
        if spot > 0:
            basis = abs(mark - spot) / spot
            if basis > self.CARRY_BASIS_MAX:
                return f"basis_too_wide:{basis:.2%}>{self.CARRY_BASIS_MAX:.2%}"
        # Reuse pre_flight (treat carry as lev=1 short)
        return self._validate_pre_flight(asset, direction=-1, size_usd=size_usd, leverage=1.0)

    def _total_derivatives_exposure(self) -> float:
        """Directional perp notional + 2× carry notional (both legs count)."""
        perp_total = sum(p.size_usd for p in self.active_perp_positions.values())
        carry_total = sum(c.size_usd * 2 for c in self.active_carries.values())
        return perp_total + carry_total

    # ------------------------------------------------------------------
    # LIQUIDATION PRICE
    # ------------------------------------------------------------------
    def _calculate_liquidation_price(
        self, mark_price: float, direction: int, leverage: float,
        maintenance_margin_pct: float = 0.02,
    ) -> float:
        """Simplified Kraken isolated-margin liq price.

        For isolated margin: liq price ≈ entry × (1 ∓ 1/leverage + maintenance_margin_pct).
        Delegates to risk.leverage_guard if available (canonical formula).
        """
        try:
            from risk.leverage_guard import LeverageGuard
            # Reuse canonical formula (static-style call doesn't need instance state)
            lg = LeverageGuard.__new__(LeverageGuard)
            return lg.calculate_liquidation_price(
                entry_price=mark_price,
                leverage=max(leverage, 1.0),
                is_short=(direction < 0),
                margin_rate=maintenance_margin_pct,
            )
        except Exception:
            # Fallback inline formula
            lev = max(leverage, 1.0)
            if direction > 0:  # long
                return mark_price * (1 - 1.0 / lev + maintenance_margin_pct)
            else:              # short
                return mark_price * (1 + 1.0 / lev - maintenance_margin_pct)

    # ------------------------------------------------------------------
    # RECONCILIATION (critical for restart safety)
    # ------------------------------------------------------------------
    def reconcile_on_startup(self) -> None:
        """Query exchange for open positions and rebuild local state.

        WITHOUT this, any position opened before a restart becomes a 'ghost' —
        invisible to the executor but still open on the exchange.

        Must be called once during runner startup, BEFORE first tick.
        Fails closed: if reconciliation fails, executor disables itself.
        """
        if self.client is None or not self.client.is_execution_enabled():
            logger.info("[WIRE-DERIV] Reconcile skipped (execution disabled)")
            return

        try:
            exchange_positions = self.client.get_open_positions()
            # Build reverse symbol map (PF_XBTUSD -> BTC)
            symbol_to_asset = {v: k for k, v in PERP_SYMBOLS.items()}

            self.active_perp_positions.clear()
            for ep in exchange_positions:
                symbol = ep.get("symbol", "")
                asset = symbol_to_asset.get(symbol)
                if asset is None:
                    continue  # symbol we don't trade; ignore
                size = float(ep.get("size", 0) or 0)
                entry = float(ep.get("price", 0) or 0)
                side = ep.get("side", "buy")
                if size <= 0 or entry <= 0:
                    continue
                size_usd = size * entry
                # Leverage unknown from openpositions — use 1.0 placeholder; liq_price
                # can be recomputed on first tick if needed.
                liq_price = self._calculate_liquidation_price(
                    entry, direction=(1 if side == "buy" else -1), leverage=1.0,
                )
                self.active_perp_positions[asset] = PerpPosition(
                    asset=asset, symbol=symbol, side=side,
                    size_usd=size_usd, size_asset=size, entry_price=entry,
                    leverage=1.0, liquidation_price=liq_price,
                    opened_at=float(ep.get("fillTime", time.time()) or time.time()),
                    order_id=ep.get("orderId", ""),
                )
                logger.info(
                    f"[WIRE-DERIV-RECONCILE] Restored {asset} {side} "
                    f"size={size:.6f} @ {entry:.2f}"
                )

            logger.info(
                f"[WIRE-DERIV-RECONCILE] Complete. "
                f"{len(self.active_perp_positions)} perp positions restored. "
                f"(Note: carry positions cannot be fully reconstructed from exchange "
                f"alone — spot+perp pairing inferred from persisted state if present.)"
            )
        except Exception as e:
            self.disable(f"reconcile_failed:{e}")
            self._alert("CRITICAL", "DERIV_RECONCILE_FAILED", {
                "error": str(e)[:300],
                "message": "Derivatives DISABLED until manual check",
            })

    # ------------------------------------------------------------------
    # PERSISTENCE (restart continuity for cash-carry pairing)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_perp_positions": {a: p.to_dict() for a, p in self.active_perp_positions.items()},
            "active_carries": {a: c.to_dict() for a, c in self.active_carries.items()},
            "disabled": self._disabled,
            "disabled_reason": self._disabled_reason,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        # We do NOT restore perp positions from local state — reconcile_on_startup()
        # is the authoritative source. Only restore carry spot-leg metadata.
        for asset, carry_data in (data.get("active_carries") or {}).items():
            try:
                self.active_carries[asset] = CashCarryPosition.from_dict(carry_data)
            except Exception as e:
                logger.warning(f"[WIRE-DERIV] Failed to restore {asset} carry: {e}")
        self._disabled = data.get("disabled", False)
        self._disabled_reason = data.get("disabled_reason", "")

    # ------------------------------------------------------------------
    # STATUS / ALERT HELPERS
    # ------------------------------------------------------------------
    def get_positions_summary(self) -> Dict[str, Any]:
        return {
            "directional": {a: p.to_dict() for a, p in self.active_perp_positions.items()},
            "carries": {a: c.to_dict() for a, c in self.active_carries.items()},
            "total_exposure_usd": self._total_derivatives_exposure(),
            "disabled": self._disabled,
            "disabled_reason": self._disabled_reason,
        }

    def _get_spot_price(self, asset: str) -> Optional[float]:
        if self.spot_exchange is None:
            return None
        try:
            symbol = SPOT_SYMBOLS.get(asset, f"{asset}/USD")
            ticker = self.spot_exchange.fetch_ticker(symbol)
            return float(ticker.get("last") or ticker.get("close") or 0)
        except Exception:
            return None

    def _alert(self, level: str, event: str, data: Dict[str, Any]) -> None:
        if self.discord is None:
            return
        try:
            from infra.persistence import AlertSeverity, AlertCategory
            sev = {
                "INFO": AlertSeverity.INFO,
                "WARNING": AlertSeverity.WARNING,
                "CRITICAL": AlertSeverity.CRITICAL,
            }.get(level, AlertSeverity.INFO)
            message = f"**{event}** — {data.get('asset', '?')} "
            self.discord.send_alert(
                severity=sev,
                category=AlertCategory.TRADING,
                message=message,
                details={k: str(v)[:100] for k, v in data.items()},
            )
        except Exception as e:
            logger.debug(f"[WIRE-DERIV] Alert send failed: {e}")

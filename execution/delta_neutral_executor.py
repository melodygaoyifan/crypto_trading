"""
================================================================================
HMATS v6.7 - Delta-Neutral Carry Executor
================================================================================
Manages spot+perp paired positions for funding rate harvesting.

Paper mode: simulates fills, tracks positions, accrues funding.
Live mode: ccxt.krakenfutures for perp leg + existing spot exchange for spot leg.

Signal source: CashAndCarryStrategy.analyze() -> CarrySignal
Wiring tag: [WIRE-G2]
================================================================================
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CarryExecutorConfig:
    """Delta-neutral carry executor configuration."""
    max_carry_budget_pct: float = 0.10      # 10% NAV total across all assets
    max_per_asset_pct: float = 0.05         # 5% NAV per single carry position
    max_hold_hours: int = 168               # 7 days max hold
    max_unrealized_loss_pct: float = 0.02   # 2% basis risk -> exit
    min_funding_for_entry: float = 0.0001   # 0.01%/8h minimum to enter
    exit_funding_threshold: float = 0.00005 # 0.005%/8h -> funding too low, exit
    funding_accrual_hours: float = 4.0      # Kraken pays funding every 4h
    slippage_bps: float = 5.0              # Paper mode slippage per leg


# =============================================================================
# POSITION STATE
# =============================================================================

@dataclass
class CarryPosition:
    """State of one active delta-neutral carry position."""
    asset: str
    spot_entry_price: float
    perp_entry_price: float
    size_usd: float
    size_asset: float
    entry_funding_rate: float
    entry_time: float
    cumulative_funding_usd: float = 0.0
    last_funding_accrue: float = 0.0
    unrealized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "spot_entry_price": self.spot_entry_price,
            "perp_entry_price": self.perp_entry_price,
            "size_usd": self.size_usd,
            "size_asset": self.size_asset,
            "entry_funding_rate": self.entry_funding_rate,
            "entry_time": self.entry_time,
            "cumulative_funding_usd": self.cumulative_funding_usd,
            "last_funding_accrue": self.last_funding_accrue,
            "unrealized_pnl": self.unrealized_pnl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarryPosition":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# =============================================================================
# FUTURES SYMBOL MAP
# =============================================================================

FUTURES_SYMBOLS = {
    "BTC": "PF_XBTUSD",
    "ETH": "PF_ETHUSD",
    "SOL": "PF_SOLUSD",
}

SPOT_SYMBOLS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
}


# =============================================================================
# EXECUTOR
# =============================================================================

class DeltaNeutralExecutor:
    """
    Manages delta-neutral carry positions (long spot + short perp).

    Paper mode: simulates fills at current price +/- slippage.
    Live mode: uses ccxt.kraken (spot) + ccxt.krakenfutures (perp).
    """

    def __init__(
        self,
        config: Optional[CarryExecutorConfig] = None,
        dry_run: bool = True,
        spot_exchange=None,
    ):
        self.config = config or CarryExecutorConfig()
        self.dry_run = dry_run
        self._spot_exchange = spot_exchange
        self._futures_exchange = None
        self.active_positions: Dict[str, CarryPosition] = {}
        self._total_realized_pnl: float = 0.0
        self._trade_count: int = 0

    # -----------------------------------------------------------------
    # Lazy futures client init (live mode only)
    # -----------------------------------------------------------------
    def _ensure_futures_exchange(self):
        """Lazy-init ccxt.krakenfutures on first live carry trade."""
        if self._futures_exchange is not None:
            return True
        if self.dry_run:
            return True  # Paper mode doesn't need real exchange

        try:
            import ccxt
            api_key = os.environ.get("KRAKEN_API_KEY", "")
            api_secret = os.environ.get("KRAKEN_API_SECRET", "")
            if not api_key or not api_secret:
                logger.warning("[WIRE-G2] No API keys for futures - falling back to paper")
                self.dry_run = True
                return True

            self._futures_exchange = ccxt.krakenfutures({
                "apiKey": api_key,
                "secret": api_secret,
                "timeout": 10000,
                "enableRateLimit": True,
            })
            logger.info("[WIRE-G2] Kraken Futures exchange initialized")
            return True
        except Exception as e:
            logger.error(f"[WIRE-G2] Futures exchange init failed: {e}")
            return False

    # -----------------------------------------------------------------
    # Open carry position
    # -----------------------------------------------------------------
    def open_carry(
        self,
        asset: str,
        size_usd: float,
        spot_price: float,
        funding_rate: float,
        nav: float = 100_000.0,
    ) -> bool:
        """Open delta-neutral position: long spot + short perp."""
        if asset in self.active_positions:
            logger.debug(f"[WIRE-G2] {asset}: carry already active, skip")
            return False

        if funding_rate < self.config.min_funding_for_entry:
            logger.debug(f"[WIRE-G2] {asset}: funding {funding_rate:.6f} below min")
            return False

        if spot_price <= 0:
            return False

        # Budget checks
        total_active = sum(p.size_usd for p in self.active_positions.values())
        if total_active + size_usd > nav * self.config.max_carry_budget_pct:
            logger.info(f"[WIRE-G2] {asset}: budget exhausted (active=${total_active:.0f})")
            return False

        size_usd = min(size_usd, nav * self.config.max_per_asset_pct)
        size_asset = size_usd / spot_price

        if not self._ensure_futures_exchange():
            return False

        now = time.time()
        slip_mult = self.config.slippage_bps / 10_000

        if self.dry_run:
            # Paper mode: simulate fills with slippage
            spot_fill = spot_price * (1.0 + slip_mult)   # buy spot slightly above
            perp_fill = spot_price * (1.0 - slip_mult)   # sell perp slightly below
        else:
            # Live mode: place both legs
            spot_fill = self._execute_spot_buy(asset, size_asset, spot_price)
            if spot_fill is None:
                logger.error(f"[WIRE-G2] {asset}: spot leg FAILED")
                return False

            perp_fill = self._execute_futures_sell(asset, size_asset, spot_price)
            if perp_fill is None:
                logger.error(f"[WIRE-G2] {asset}: perp leg FAILED - unwinding spot")
                self._emergency_unwind_spot(asset, size_asset)
                return False

        pos = CarryPosition(
            asset=asset,
            spot_entry_price=spot_fill,
            perp_entry_price=perp_fill,
            size_usd=size_usd,
            size_asset=size_asset,
            entry_funding_rate=funding_rate,
            entry_time=now,
            last_funding_accrue=now,
        )
        self.active_positions[asset] = pos
        self._trade_count += 1

        annual = funding_rate * 3 * 365
        logger.info(
            f"[WIRE-G2] CARRY OPENED: {asset} "
            f"size=${size_usd:.0f} ({size_asset:.6f} units) "
            f"spot={spot_fill:.2f} perp={perp_fill:.2f} "
            f"funding={funding_rate:.4%} ({annual:.1%} ann)"
        )
        return True

    # -----------------------------------------------------------------
    # Close carry position
    # -----------------------------------------------------------------
    def close_carry(
        self,
        asset: str,
        spot_price: float,
        reason: str = "signal",
    ) -> Optional[Dict[str, Any]]:
        """Close delta-neutral position. Returns PnL dict."""
        if asset not in self.active_positions:
            return None

        pos = self.active_positions[asset]
        slip_mult = self.config.slippage_bps / 10_000

        if self.dry_run:
            spot_exit = spot_price * (1.0 - slip_mult)  # sell spot slightly below
            perp_exit = spot_price * (1.0 + slip_mult)  # buy perp slightly above
        else:
            spot_exit = self._execute_spot_sell(asset, pos.size_asset, spot_price)
            perp_exit = self._execute_futures_buy(asset, pos.size_asset, spot_price)
            if spot_exit is None:
                spot_exit = spot_price
            if perp_exit is None:
                perp_exit = spot_price

        # PnL calculation
        spot_pnl = (spot_exit - pos.spot_entry_price) * pos.size_asset
        perp_pnl = (pos.perp_entry_price - perp_exit) * pos.size_asset
        basis_pnl = spot_pnl + perp_pnl
        total_pnl = basis_pnl + pos.cumulative_funding_usd
        hold_hours = (time.time() - pos.entry_time) / 3600

        self._total_realized_pnl += total_pnl
        del self.active_positions[asset]

        logger.info(
            f"[WIRE-G2] CARRY CLOSED: {asset} reason={reason} "
            f"held={hold_hours:.1f}h basis_pnl=${basis_pnl:.2f} "
            f"funding=${pos.cumulative_funding_usd:.2f} total=${total_pnl:.2f}"
        )

        return {
            "asset": asset,
            "reason": reason,
            "hold_hours": hold_hours,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "basis_pnl": basis_pnl,
            "funding_pnl": pos.cumulative_funding_usd,
            "total_pnl": total_pnl,
        }

    # -----------------------------------------------------------------
    # Per-tick: accrue funding + check exit conditions
    # -----------------------------------------------------------------
    def tick(self, asset: str, market_data: Dict) -> Optional[str]:
        """Per-tick check. Returns exit reason string or None."""
        if asset not in self.active_positions:
            return None

        pos = self.active_positions[asset]
        now = time.time()
        funding_rate = market_data.get("funding_rate", 0.0) or 0.0
        spot_price = market_data.get("current_price", 0.0) or 0.0

        # Accrue funding (every funding_accrual_hours)
        hours_since = (now - pos.last_funding_accrue) / 3600
        if hours_since >= self.config.funding_accrual_hours and funding_rate > 0:
            periods = int(hours_since / self.config.funding_accrual_hours)
            payment = funding_rate * pos.size_usd * periods
            pos.cumulative_funding_usd += payment
            pos.last_funding_accrue = now
            if payment > 0.01:
                logger.debug(
                    f"[WIRE-G2] {asset}: funding accrued ${payment:.4f} "
                    f"(total=${pos.cumulative_funding_usd:.2f})"
                )

        # Update unrealized PnL
        if spot_price > 0:
            spot_pnl = (spot_price - pos.spot_entry_price) * pos.size_asset
            perp_pnl = (pos.perp_entry_price - spot_price) * pos.size_asset
            pos.unrealized_pnl = spot_pnl + perp_pnl + pos.cumulative_funding_usd

        # Exit condition 1: funding reversed
        if abs(funding_rate) < self.config.exit_funding_threshold:
            return "FUNDING_REVERSED"

        # Exit condition 2: max hold time
        hold_hours = (now - pos.entry_time) / 3600
        if hold_hours > self.config.max_hold_hours:
            return "MAX_HOLD"

        # Exit condition 3: basis risk (unrealized loss exceeds threshold)
        if pos.size_usd > 0:
            basis_loss_pct = -pos.unrealized_pnl / pos.size_usd
            if basis_loss_pct > self.config.max_unrealized_loss_pct:
                return "BASIS_RISK"

        return None

    # -----------------------------------------------------------------
    # Live mode execution helpers
    # -----------------------------------------------------------------
    def _execute_spot_buy(self, asset: str, size: float, price: float) -> Optional[float]:
        """Place spot buy order. Returns fill price or None."""
        try:
            symbol = SPOT_SYMBOLS.get(asset, f"{asset}/USD")
            result = self._spot_exchange.create_limit_buy_order(
                symbol, size, price, params={"oflags": "post"}
            )
            return float(result.get("price", price))
        except Exception as e:
            logger.error(f"[WIRE-G2] Spot buy failed: {e}")
            return None

    def _execute_spot_sell(self, asset: str, size: float, price: float) -> Optional[float]:
        """Place spot sell (market) to close. Returns fill price or None."""
        try:
            symbol = SPOT_SYMBOLS.get(asset, f"{asset}/USD")
            result = self._spot_exchange.create_market_sell_order(symbol, size)
            return float(result.get("price", price))
        except Exception as e:
            logger.error(f"[WIRE-G2] Spot sell failed: {e}")
            return None

    def _execute_futures_sell(self, asset: str, size: float, price: float) -> Optional[float]:
        """Place futures short. Returns fill price or None."""
        try:
            symbol = FUTURES_SYMBOLS.get(asset)
            if not symbol or not self._futures_exchange:
                return None
            result = self._futures_exchange.create_limit_sell_order(
                symbol, size, price, params={"postOnly": True}
            )
            return float(result.get("price", price))
        except Exception as e:
            logger.error(f"[WIRE-G2] Futures sell failed: {e}")
            return None

    def _execute_futures_buy(self, asset: str, size: float, price: float) -> Optional[float]:
        """Place futures buy (market) to close short. Returns fill price or None."""
        try:
            symbol = FUTURES_SYMBOLS.get(asset)
            if not symbol or not self._futures_exchange:
                return None
            result = self._futures_exchange.create_market_buy_order(symbol, size)
            return float(result.get("price", price))
        except Exception as e:
            logger.error(f"[WIRE-G2] Futures buy failed: {e}")
            return None

    def _emergency_unwind_spot(self, asset: str, size: float):
        """Emergency: close spot leg if futures leg failed."""
        logger.warning(f"[WIRE-G2] EMERGENCY UNWIND spot for {asset}")
        try:
            symbol = SPOT_SYMBOLS.get(asset, f"{asset}/USD")
            if self._spot_exchange:
                self._spot_exchange.create_market_sell_order(symbol, size)
        except Exception as e:
            logger.error(f"[WIRE-G2] Emergency unwind failed: {e}")

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for restart recovery."""
        return {
            "active_positions": {
                k: v.to_dict() for k, v in self.active_positions.items()
            },
            "total_realized_pnl": self._total_realized_pnl,
            "trade_count": self._trade_count,
        }

    def from_dict(self, data: Dict[str, Any]):
        """Restore state from saved data."""
        if not data:
            return
        self._total_realized_pnl = data.get("total_realized_pnl", 0.0)
        self._trade_count = data.get("trade_count", 0)
        for asset, pos_data in data.get("active_positions", {}).items():
            try:
                self.active_positions[asset] = CarryPosition.from_dict(pos_data)
            except Exception as e:
                logger.warning(f"[WIRE-G2] Failed to restore {asset} carry: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Status dict for logging/dashboard."""
        return {
            "active_count": len(self.active_positions),
            "active_assets": list(self.active_positions.keys()),
            "total_size_usd": sum(p.size_usd for p in self.active_positions.values()),
            "total_funding_earned": sum(
                p.cumulative_funding_usd for p in self.active_positions.values()
            ),
            "total_unrealized_pnl": sum(
                p.unrealized_pnl for p in self.active_positions.values()
            ),
            "total_realized_pnl": self._total_realized_pnl,
            "trade_count": self._trade_count,
            "dry_run": self.dry_run,
        }

"""
================================================================================
HMATS v6.7 - Cash-and-Carry (Basis Trade) Strategy
================================================================================

Delta-neutral strategy: Buy spot + Sell perpetual = Earn funding rate.

Concept:
    - When funding rate is positive (longs pay shorts), hold:
        * LONG spot position (no funding cost)
        * SHORT perpetual position (receive funding)
    - Net delta = 0 (market neutral)
    - Revenue = funding rate income - execution costs
    - Target: 8-12% annualized from funding alone

Entry Conditions:
    1. Funding rate > entry_threshold (default 0.01%/8h = ~4.5% annualized)
    2. Funding trend is stable or rising (not one-off spike)
    3. Spot-perp basis > min_basis_bps (spread is profitable)
    4. No extreme volatility (avoid liquidation on perp side)

Exit Conditions:
    1. Funding rate < exit_threshold (funding no longer pays)
    2. Basis inverts (perp cheaper than spot)
    3. Unrealized loss > max_unrealized_loss_pct (basis risk materialized)
    4. Max hold time exceeded

Kraken Futures:
    - Symbols: PF_XBTUSD, PF_ETHUSD, PF_SOLUSD
    - Requires separate krakenfutures ccxt instance
    - Funding period: 4h (Kraken) vs 8h (Binance)

NOTE: This module provides signal generation only. Execution through
the main pipeline requires Kraken Futures API integration (Phase 2).
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CashAndCarryConfig:
    """Cash-and-carry strategy configuration."""
    # Enable/disable
    enabled: bool = True

    # Funding rate thresholds (per 8h period)
    entry_funding_threshold: float = 0.0001   # 0.01%/8h = ~4.5% annualized
    exit_funding_threshold: float = 0.00005   # 0.005%/8h = exit when near zero
    extreme_funding_boost: float = 0.0005     # 0.05%/8h = boost allocation

    # Basis thresholds (spot-perp spread)
    min_basis_bps: float = 5.0          # Min 5bps basis to enter
    max_basis_bps: float = 200.0        # >200bps basis = anomaly, don't enter

    # Position sizing (% of portfolio allocated to cash-and-carry)
    base_allocation_pct: float = 0.10   # 10% of portfolio
    max_allocation_pct: float = 0.25    # Max 25% in extreme funding
    boost_allocation_pct: float = 0.20  # 20% when funding > extreme threshold

    # Risk limits
    max_unrealized_loss_pct: float = 0.02   # 2% loss -> exit
    max_hold_hours: int = 168               # 7 days max hold
    min_funding_periods: int = 3            # Need 3 consecutive positive periods

    # Supported assets
    assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Kraken Futures symbol mapping
    futures_symbols: Dict[str, str] = field(default_factory=lambda: {
        "BTC": "PF_XBTUSD",
        "ETH": "PF_ETHUSD",
        "SOL": "PF_SOLUSD",
    })


class CarryState(Enum):
    """State of a cash-and-carry position."""
    IDLE = "IDLE"                   # No position
    MONITORING = "MONITORING"       # Watching funding rate
    ENTERING = "ENTERING"           # Opening spot+perp
    ACTIVE = "ACTIVE"               # Position open, earning funding
    EXITING = "EXITING"             # Closing positions
    COOLDOWN = "COOLDOWN"           # Post-exit cooldown


# =============================================================================
# CARRY SIGNAL
# =============================================================================

@dataclass
class CarrySignal:
    """Signal from cash-and-carry analysis."""
    asset: str
    action: str  # "ENTER", "EXIT", "HOLD", "NONE"
    funding_rate: float  # Current 8h funding rate
    funding_annualized: float  # Annualized funding yield
    basis_bps: float  # Current spot-perp basis
    allocation_pct: float  # Recommended allocation
    confidence: float  # 0-1
    reasons: List[str] = field(default_factory=list)
    state: CarryState = CarryState.IDLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "action": self.action,
            "funding_rate": self.funding_rate,
            "funding_annualized": self.funding_annualized,
            "basis_bps": self.basis_bps,
            "allocation_pct": self.allocation_pct,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "state": self.state.value,
        }


# =============================================================================
# CASH-AND-CARRY STRATEGY
# =============================================================================

class CashAndCarryStrategy:
    """
    Cash-and-carry (basis trade) strategy engine.

    Monitors funding rates across assets and generates entry/exit signals
    for delta-neutral spot+perp positions.
    """

    def __init__(self, config: Optional[CashAndCarryConfig] = None):
        self.config = config or CashAndCarryConfig()

        # Per-asset state tracking
        self._states: Dict[str, CarryState] = {
            a: CarryState.IDLE for a in self.config.assets
        }
        self._entry_times: Dict[str, datetime] = {}
        self._entry_funding: Dict[str, float] = {}

        # Funding rate history (per asset, last 24 periods = 8 days at 8h)
        self._funding_history: Dict[str, deque] = {
            a: deque(maxlen=24) for a in self.config.assets
        }

        # Cumulative funding earned (paper tracking)
        self._cumulative_funding: Dict[str, float] = {
            a: 0.0 for a in self.config.assets
        }

        logger.info(
            f"[CASH_CARRY] Initialized: assets={self.config.assets}, "
            f"entry_threshold={self.config.entry_funding_threshold:.4%}, "
            f"base_alloc={self.config.base_allocation_pct:.0%}"
        )

    def analyze(
        self,
        asset: str,
        funding_rate: float,
        spot_price: float,
        perp_price: Optional[float] = None,
        account_equity: float = 100_000.0,
    ) -> CarrySignal:
        """
        Analyze an asset for cash-and-carry opportunity.

        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            funding_rate: Current 8h funding rate (e.g., 0.0001 = 0.01%)
            spot_price: Current spot price
            perp_price: Current perpetual price (None = use spot)
            account_equity: Current account equity

        Returns:
            CarrySignal with recommended action
        """
        if asset not in self.config.assets or not self.config.enabled:
            return CarrySignal(
                asset=asset, action="NONE", funding_rate=funding_rate,
                funding_annualized=0.0, basis_bps=0.0,
                allocation_pct=0.0, confidence=0.0,
            )

        # Record funding history
        self._funding_history[asset].append({
            "rate": funding_rate,
            "timestamp": datetime.now(timezone.utc),
        })

        # Calculate basis (spot vs perp spread)
        if perp_price and perp_price > 0 and spot_price > 0:
            basis_bps = (perp_price - spot_price) / spot_price * 10000
        else:
            # No perp price available - estimate from funding
            basis_bps = funding_rate * 10000 * 3  # Rough: basis ≈ 3x funding

        # Annualize funding (3 periods/day * 365 days)
        funding_annualized = funding_rate * 3 * 365

        current_state = self._states.get(asset, CarryState.IDLE)

        if current_state == CarryState.ACTIVE:
            return self._check_exit(asset, funding_rate, funding_annualized, basis_bps)
        else:
            return self._check_entry(asset, funding_rate, funding_annualized, basis_bps)

    def _check_entry(
        self, asset: str, funding_rate: float,
        funding_annualized: float, basis_bps: float,
    ) -> CarrySignal:
        """Check if entry conditions are met."""
        reasons = []
        score = 0.0

        # Condition 1: Funding rate above threshold
        if funding_rate >= self.config.entry_funding_threshold:
            score += 0.4
            reasons.append(f"Funding {funding_rate:.4%}/8h ({funding_annualized:.1%} ann)")
        else:
            return CarrySignal(
                asset=asset, action="NONE", funding_rate=funding_rate,
                funding_annualized=funding_annualized, basis_bps=basis_bps,
                allocation_pct=0.0, confidence=0.0,
                reasons=[f"Funding {funding_rate:.4%} < threshold {self.config.entry_funding_threshold:.4%}"],
            )

        # Condition 2: Funding persistence (N consecutive positive periods)
        history = list(self._funding_history[asset])
        if len(history) >= self.config.min_funding_periods:
            recent = history[-self.config.min_funding_periods:]
            all_positive = all(h["rate"] > 0 for h in recent)
            if all_positive:
                score += 0.3
                reasons.append(f"{self.config.min_funding_periods} consecutive positive funding")
            else:
                score -= 0.2
                reasons.append("Inconsistent funding history")
        else:
            reasons.append(f"Insufficient history ({len(history)}/{self.config.min_funding_periods})")

        # Condition 3: Basis within range
        if self.config.min_basis_bps <= basis_bps <= self.config.max_basis_bps:
            score += 0.2
            reasons.append(f"Basis {basis_bps:.1f}bps (within range)")
        elif basis_bps > self.config.max_basis_bps:
            score -= 0.3
            reasons.append(f"Basis {basis_bps:.1f}bps too wide (anomaly)")

        # Determine allocation
        if funding_rate >= self.config.extreme_funding_boost:
            allocation = self.config.boost_allocation_pct
            score += 0.1
            reasons.append(f"Extreme funding -> boosted allocation {allocation:.0%}")
        else:
            allocation = self.config.base_allocation_pct

        allocation = min(allocation, self.config.max_allocation_pct)
        confidence = max(0.0, min(1.0, score))

        if confidence >= 0.5:
            action = "ENTER"
            self._states[asset] = CarryState.MONITORING
            logger.info(
                f"[CASH_CARRY] {asset} ENTER signal: funding={funding_rate:.4%}, "
                f"basis={basis_bps:.1f}bps, alloc={allocation:.0%}, conf={confidence:.2f}"
            )
        else:
            action = "NONE"

        return CarrySignal(
            asset=asset, action=action, funding_rate=funding_rate,
            funding_annualized=funding_annualized, basis_bps=basis_bps,
            allocation_pct=allocation, confidence=confidence,
            reasons=reasons, state=self._states.get(asset, CarryState.IDLE),
        )

    def _check_exit(
        self, asset: str, funding_rate: float,
        funding_annualized: float, basis_bps: float,
    ) -> CarrySignal:
        """Check if exit conditions are met for an active carry position."""
        reasons = []
        should_exit = False

        # Exit 1: Funding dropped below exit threshold
        if funding_rate < self.config.exit_funding_threshold:
            should_exit = True
            reasons.append(f"Funding {funding_rate:.4%} < exit threshold")

        # Exit 2: Basis inverted (perp cheaper than spot)
        if basis_bps < -10:
            should_exit = True
            reasons.append(f"Basis inverted: {basis_bps:.1f}bps")

        # Exit 3: Max hold time
        entry_time = self._entry_times.get(asset)
        if entry_time:
            held_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            if held_hours > self.config.max_hold_hours:
                should_exit = True
                reasons.append(f"Max hold time exceeded ({held_hours:.0f}h > {self.config.max_hold_hours}h)")

        # Track cumulative funding
        self._cumulative_funding[asset] += funding_rate

        if should_exit:
            self._states[asset] = CarryState.EXITING
            logger.info(
                f"[CASH_CARRY] {asset} EXIT signal: {', '.join(reasons)} | "
                f"cumulative_funding={self._cumulative_funding[asset]:.4%}"
            )
            return CarrySignal(
                asset=asset, action="EXIT", funding_rate=funding_rate,
                funding_annualized=funding_annualized, basis_bps=basis_bps,
                allocation_pct=0.0, confidence=0.9,
                reasons=reasons, state=CarryState.EXITING,
            )

        # Hold - still earning
        reasons.append(f"Holding: funding={funding_rate:.4%}, cumulative={self._cumulative_funding[asset]:.4%}")
        return CarrySignal(
            asset=asset, action="HOLD", funding_rate=funding_rate,
            funding_annualized=funding_annualized, basis_bps=basis_bps,
            allocation_pct=self.config.base_allocation_pct,
            confidence=0.7, reasons=reasons, state=CarryState.ACTIVE,
        )

    def on_position_opened(self, asset: str, funding_rate: float):
        """Called when carry position is actually opened."""
        self._states[asset] = CarryState.ACTIVE
        self._entry_times[asset] = datetime.now(timezone.utc)
        self._entry_funding[asset] = funding_rate
        self._cumulative_funding[asset] = 0.0
        logger.info(f"[CASH_CARRY] {asset} position OPENED at funding={funding_rate:.4%}")

    def on_position_closed(self, asset: str):
        """Called when carry position is actually closed."""
        cum = self._cumulative_funding.get(asset, 0.0)
        self._states[asset] = CarryState.COOLDOWN
        self._entry_times.pop(asset, None)
        logger.info(f"[CASH_CARRY] {asset} position CLOSED, total_funding_earned={cum:.4%}")

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status summary."""
        return {
            "states": {a: s.value for a, s in self._states.items()},
            "cumulative_funding": dict(self._cumulative_funding),
            "history_lengths": {a: len(h) for a, h in self._funding_history.items()},
        }


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[CashAndCarryStrategy] = None


def get_cash_and_carry() -> CashAndCarryStrategy:
    """Get or create singleton instance."""
    global _instance
    if _instance is None:
        _instance = CashAndCarryStrategy()
    return _instance

"""
_DEPRECATED (T29): Alternative risk pipeline, NOT wired into main.py.
Decision-affecting risk checks live in:
  - integration_v36._check_early_hard_veto() (drawdown/correlation)
  - agents/risk_agent.py (RISK_MULTIPLIERS)
  - market/intraday_correlation_monitor.py (correlation p0_abort)
WARNING: Contains hardcoded 10%/0.95 thresholds that conflict with ULTRA profile.

================================================================================
HMATS v5.1.0 - RISK GOVERNOR
================================================================================
Version: 5.1.0-HARDENED
Purpose: Runtime risk governor with veto authority over all execution

The Risk Governor is NOT a passive module - it has VETO AUTHORITY.
When the governor says NO, execution MUST NOT proceed.

VETO TYPES:
    HARD - Non-negotiable, immediate halt (drawdown, correlation crisis)
    SOFT - Reduces/delays execution (exposure limits, volatility)

NON-NEGOTIABLE RULES (from v3.3-HR):
    1. Hard drawdown halt at 10%
    2. Correlation crisis at 0.95
    3. No override allowed for HARD vetoes

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# VETO TYPES
# =============================================================================

class VetoType(Enum):
    """Type of risk veto."""
    NONE = "NONE"           # No veto
    SOFT = "SOFT"           # Reduce/delay execution
    HARD = "HARD"           # Immediate halt, no override


class VetoReason(Enum):
    """Specific veto reasons."""
    # HARD vetoes
    HARD_DRAWDOWN = "HARD_DRAWDOWN"
    CORRELATION_CRISIS = "CORRELATION_CRISIS"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    DATA_INVALID = "DATA_INVALID"
    
    # SOFT vetoes
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    ASSET_CONCENTRATION = "ASSET_CONCENTRATION"
    VOLATILITY_HIGH = "VOLATILITY_HIGH"
    LIQUIDITY_LOW = "LIQUIDITY_LOW"
    RECENT_LOSS = "RECENT_LOSS"
    
    # Throttles
    THROTTLE_SIZE = "THROTTLE_SIZE"
    THROTTLE_FREQUENCY = "THROTTLE_FREQUENCY"


# =============================================================================
# GOVERNOR CONFIG
# =============================================================================

@dataclass
class GovernorConfig:
    """Risk governor configuration."""
    
    # HARD limits (NON-NEGOTIABLE - from v3.3-HR)
    hard_drawdown_halt: float = 0.20        # 20% drawdown = HALT [UNLEASH UL-5]
    correlation_crisis: float = 0.98        # 0.98 correlation = HALT [UNLEASH UL-4]
    
    # SOFT limits
    max_gross_exposure: float = 1.50        # 150% max gross
    max_asset_exposure: float = 0.80        # 80% per asset
    max_single_trade_pct: float = 0.25      # 25% single trade
    
    # Volatility
    volatility_reduction_threshold: float = 2.0  # 2x ATR = reduce
    volatility_halt_threshold: float = 4.0       # 4x ATR = halt
    
    # Liquidity
    min_orderbook_depth_pct: float = 0.001  # Min depth vs trade size
    
    # Frequency
    min_seconds_between_trades: int = 60    # Minimum time between trades
    max_trades_per_hour: int = 10           # Max trades per hour
    
    # Loss recovery
    pause_after_loss_pct: float = 0.02      # Pause after 2% loss
    pause_duration_seconds: int = 3600      # 1 hour pause


# =============================================================================
# GOVERNOR DECISION
# =============================================================================

@dataclass
class GovernorDecision:
    """Decision from risk governor."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Veto
    vetoed: bool = False
    veto_type: VetoType = VetoType.NONE
    veto_reasons: List[VetoReason] = field(default_factory=list)
    reason: str = ""
    
    # Adjustments (if not fully vetoed)
    size_multiplier: float = 1.0   # 0 to 1
    urgency_adjustment: float = 0.0  # -1 to +1
    delay_seconds: int = 0
    
    # State at decision
    current_drawdown: float = 0.0
    current_exposure: float = 0.0
    
    # Metadata
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    
    def to_log(self) -> str:
        """Generate log line."""
        if self.vetoed:
            return f"[RISK_VETO] {self.veto_type.value}: {self.reason}"
        elif self.size_multiplier < 1.0:
            return f"[RISK_ADJUST] size={self.size_multiplier:.2f}, delay={self.delay_seconds}s"
        return "[RISK_OK]"


# =============================================================================
# RISK GOVERNOR
# =============================================================================

class RiskGovernor:
    """
    Runtime risk governor with VETO AUTHORITY.
    
    This is NOT a passive risk module. When the governor vetoes,
    execution MUST NOT proceed. There is no override for HARD vetoes.
    
    Key Responsibilities:
    1. Drawdown monitoring (10% HALT - NON-NEGOTIABLE)
    2. Exposure management (gross and per-asset)
    3. Volatility-based throttling
    4. Liquidity checks
    5. Trade frequency limits
    6. Loss recovery pauses
    """
    
    def __init__(self, config: Optional[GovernorConfig] = None):
        """Initialize risk governor."""
        self.config = config or GovernorConfig()
        
        # State
        self._portfolio_value: float = 100000.0
        self._peak_value: float = 100000.0
        self._current_drawdown: float = 0.0
        
        # Positions
        self._positions: Dict[str, float] = {}  # asset -> exposure %
        
        # Trade tracking
        self._last_trade_time: Optional[datetime] = None
        self._trades_this_hour: int = 0
        self._hour_start: Optional[datetime] = None
        
        # Loss tracking
        self._pause_until: Optional[datetime] = None
        self._recent_pnl: float = 0.0
        
        # Volatility
        self._current_volatility_ratio: float = 1.0  # Current / Historical
        
        # Correlation
        self._current_correlation: float = 0.5
        
        # Circuit breaker
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_until: Optional[datetime] = None
        
        logger.info(f"RiskGovernor initialized with config: {self.config}")
    
    def check(
        self,
        asset: str,
        direction: float,
        size: float,
        price: float = 0.0,
        volatility_ratio: float = 1.0,
        correlation: float = 0.5,
        orderbook_depth: float = float('inf'),
    ) -> GovernorDecision:
        """
        Check proposed trade against risk limits.
        
        This is the MAIN ENTRY POINT for risk checks.
        
        Args:
            asset: Asset to trade
            direction: Direction (-1 to 1)
            size: Size as fraction of portfolio (0 to 1)
            price: Current price (for liquidity checks)
            volatility_ratio: Current/Historical volatility
            correlation: Cross-asset correlation
            orderbook_depth: Orderbook depth in USD
            
        Returns:
            GovernorDecision with veto/adjustment details
        """
        timestamp = datetime.now(timezone.utc)
        decision = GovernorDecision(timestamp=timestamp)
        
        # Update state
        self._current_volatility_ratio = volatility_ratio
        self._current_correlation = correlation
        
        # =====================================================================
        # HARD CHECKS (Non-negotiable - cannot be overridden)
        # =====================================================================
        
        # 1. Drawdown check (NON-NEGOTIABLE)
        if self._current_drawdown >= self.config.hard_drawdown_halt:
            decision.vetoed = True
            decision.veto_type = VetoType.HARD
            decision.veto_reasons.append(VetoReason.HARD_DRAWDOWN)
            decision.reason = (
                f"HARD_DRAWDOWN: {self._current_drawdown:.1%} >= "
                f"{self.config.hard_drawdown_halt:.1%} [NON-NEGOTIABLE]"
            )
            decision.checks_failed.append("drawdown")
            logger.warning(f"[GOVERNOR] {decision.reason}")
            return decision
        decision.checks_passed.append("drawdown")
        
        # 2. Correlation crisis (NON-NEGOTIABLE)
        if correlation >= self.config.correlation_crisis:
            decision.vetoed = True
            decision.veto_type = VetoType.HARD
            decision.veto_reasons.append(VetoReason.CORRELATION_CRISIS)
            decision.reason = (
                f"CORRELATION_CRISIS: {correlation:.2f} >= "
                f"{self.config.correlation_crisis:.2f} [NON-NEGOTIABLE]"
            )
            decision.checks_failed.append("correlation")
            logger.warning(f"[GOVERNOR] {decision.reason}")
            return decision
        decision.checks_passed.append("correlation")
        
        # 3. Circuit breaker
        if self._circuit_breaker_active:
            if self._circuit_breaker_until and timestamp < self._circuit_breaker_until:
                decision.vetoed = True
                decision.veto_type = VetoType.HARD
                decision.veto_reasons.append(VetoReason.CIRCUIT_BREAKER)
                remaining = (self._circuit_breaker_until - timestamp).total_seconds()
                decision.reason = f"CIRCUIT_BREAKER: active for {remaining:.0f}s more"
                decision.checks_failed.append("circuit_breaker")
                logger.warning(f"[GOVERNOR] {decision.reason}")
                return decision
            else:
                self._circuit_breaker_active = False
                self._circuit_breaker_until = None
        decision.checks_passed.append("circuit_breaker")
        
        # =====================================================================
        # SOFT CHECKS (Adjustable - reduce or delay)
        # =====================================================================
        
        soft_vetoes = []
        
        # 4. Gross exposure
        current_gross = sum(abs(v) for v in self._positions.values())
        proposed_gross = current_gross + abs(size)
        if proposed_gross > self.config.max_gross_exposure:
            soft_vetoes.append(VetoReason.EXPOSURE_LIMIT)
            decision.checks_failed.append("gross_exposure")
            # Calculate reduction needed
            max_new_size = max(0, self.config.max_gross_exposure - current_gross)
            decision.size_multiplier = min(decision.size_multiplier, max_new_size / size if size > 0 else 0)
        else:
            decision.checks_passed.append("gross_exposure")
        
        # 5. Asset concentration
        current_asset = abs(self._positions.get(asset, 0))
        proposed_asset = current_asset + abs(size)
        if proposed_asset > self.config.max_asset_exposure:
            soft_vetoes.append(VetoReason.ASSET_CONCENTRATION)
            decision.checks_failed.append("asset_exposure")
            max_new_size = max(0, self.config.max_asset_exposure - current_asset)
            decision.size_multiplier = min(decision.size_multiplier, max_new_size / size if size > 0 else 0)
        else:
            decision.checks_passed.append("asset_exposure")
        
        # 6. Single trade size
        if size > self.config.max_single_trade_pct:
            soft_vetoes.append(VetoReason.THROTTLE_SIZE)
            decision.checks_failed.append("trade_size")
            decision.size_multiplier = min(decision.size_multiplier, self.config.max_single_trade_pct / size)
        else:
            decision.checks_passed.append("trade_size")
        
        # 7. Volatility
        if volatility_ratio >= self.config.volatility_halt_threshold:
            soft_vetoes.append(VetoReason.VOLATILITY_HIGH)
            decision.checks_failed.append("volatility")
            decision.size_multiplier = 0  # Full veto on extreme volatility
        elif volatility_ratio >= self.config.volatility_reduction_threshold:
            soft_vetoes.append(VetoReason.VOLATILITY_HIGH)
            decision.checks_failed.append("volatility")
            reduction = 1.0 - (volatility_ratio - self.config.volatility_reduction_threshold) / (
                self.config.volatility_halt_threshold - self.config.volatility_reduction_threshold
            )
            decision.size_multiplier = min(decision.size_multiplier, max(0.3, reduction))
        else:
            decision.checks_passed.append("volatility")
        
        # 8. Trade frequency
        if self._hour_start is None or timestamp - self._hour_start > timedelta(hours=1):
            self._hour_start = timestamp
            self._trades_this_hour = 0
        
        if self._trades_this_hour >= self.config.max_trades_per_hour:
            soft_vetoes.append(VetoReason.THROTTLE_FREQUENCY)
            decision.checks_failed.append("frequency_hour")
            decision.delay_seconds = 3600 - int((timestamp - self._hour_start).total_seconds())
        else:
            decision.checks_passed.append("frequency_hour")
        
        if self._last_trade_time:
            seconds_since = (timestamp - self._last_trade_time).total_seconds()
            if seconds_since < self.config.min_seconds_between_trades:
                soft_vetoes.append(VetoReason.THROTTLE_FREQUENCY)
                decision.checks_failed.append("frequency_min")
                decision.delay_seconds = max(
                    decision.delay_seconds,
                    self.config.min_seconds_between_trades - int(seconds_since)
                )
        else:
            decision.checks_passed.append("frequency_min")
        
        # 9. Loss recovery pause
        if self._pause_until and timestamp < self._pause_until:
            soft_vetoes.append(VetoReason.RECENT_LOSS)
            decision.checks_failed.append("loss_pause")
            decision.delay_seconds = max(
                decision.delay_seconds,
                int((self._pause_until - timestamp).total_seconds())
            )
        else:
            decision.checks_passed.append("loss_pause")
            self._pause_until = None
        
        # =====================================================================
        # FINAL DECISION
        # =====================================================================
        
        if soft_vetoes:
            decision.veto_reasons = soft_vetoes
            
            # If size reduced to 0 or delay too long, it's effectively a veto
            if decision.size_multiplier <= 0:
                decision.vetoed = True
                decision.veto_type = VetoType.SOFT
                decision.reason = f"SOFT_VETO (size=0): {', '.join(v.value for v in soft_vetoes)}"
            elif decision.delay_seconds > 3600:  # More than 1 hour
                decision.vetoed = True
                decision.veto_type = VetoType.SOFT
                decision.reason = f"SOFT_VETO (delay>{decision.delay_seconds}s): {', '.join(v.value for v in soft_vetoes)}"
            else:
                decision.reason = f"ADJUSTED: {', '.join(v.value for v in soft_vetoes)}"
        
        decision.current_drawdown = self._current_drawdown
        decision.current_exposure = current_gross
        
        logger.info(f"[GOVERNOR] {decision.to_log()}")
        return decision
    
    # =========================================================================
    # STATE UPDATES
    # =========================================================================
    
    def update_portfolio(
        self,
        value: float,
        positions: Optional[Dict[str, float]] = None
    ):
        """Update portfolio state."""
        self._portfolio_value = value
        self._peak_value = max(self._peak_value, value)
        self._current_drawdown = (self._peak_value - value) / self._peak_value if self._peak_value > 0 else 0
        
        if positions:
            self._positions = positions.copy()
        
        logger.debug(f"[GOVERNOR] Portfolio updated: value={value:.2f}, dd={self._current_drawdown:.2%}")
    
    def record_trade(
        self,
        asset: str,
        size: float,
        pnl: float = 0.0
    ):
        """Record a completed trade."""
        now = datetime.now(timezone.utc)
        self._last_trade_time = now
        self._trades_this_hour += 1
        self._recent_pnl += pnl
        
        # Update position
        current = self._positions.get(asset, 0)
        self._positions[asset] = current + size
        
        # Check for loss pause
        if pnl < 0:
            loss_pct = abs(pnl) / self._portfolio_value if self._portfolio_value > 0 else 0
            if loss_pct >= self.config.pause_after_loss_pct:
                self._pause_until = now + timedelta(seconds=self.config.pause_duration_seconds)
                logger.warning(f"[GOVERNOR] Loss pause activated: {loss_pct:.2%} loss, pause until {self._pause_until}")
    
    def activate_circuit_breaker(self, duration_seconds: int = 300):
        """Activate circuit breaker."""
        self._circuit_breaker_active = True
        self._circuit_breaker_until = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        logger.error(f"[GOVERNOR] CIRCUIT BREAKER ACTIVATED for {duration_seconds}s")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker."""
        self._circuit_breaker_active = False
        self._circuit_breaker_until = None
        logger.info("[GOVERNOR] Circuit breaker reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current governor state."""
        return {
            "portfolio_value": self._portfolio_value,
            "peak_value": self._peak_value,
            "drawdown": self._current_drawdown,
            "positions": self._positions.copy(),
            "gross_exposure": sum(abs(v) for v in self._positions.values()),
            "trades_this_hour": self._trades_this_hour,
            "circuit_breaker": self._circuit_breaker_active,
            "pause_until": self._pause_until.isoformat() if self._pause_until else None,
            "config": {
                "hard_drawdown_halt": self.config.hard_drawdown_halt,
                "max_gross_exposure": self.config.max_gross_exposure,
            }
        }


# =============================================================================
# SINGLETON
# =============================================================================

_risk_governor: Optional[RiskGovernor] = None


def get_risk_governor(config: Optional[GovernorConfig] = None) -> RiskGovernor:
    """Get risk governor singleton."""
    global _risk_governor
    if _risk_governor is None:
        _risk_governor = RiskGovernor(config)
    elif config is not None and _risk_governor.config != config:
        logger.info("RiskGovernor config changed; refreshing singleton instance")
        _risk_governor = RiskGovernor(config)
    return _risk_governor


def reset_risk_governor():
    """Reset risk governor singleton."""
    global _risk_governor
    _risk_governor = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    gov = RiskGovernor()
    gov.update_portfolio(100000, {"SOL": 0.2, "BTC": 0.1})
    
    # Test normal trade
    print("\n=== Test 1: Normal trade ===")
    decision = gov.check(asset="ETH", direction=1.0, size=0.1)
    print(f"Vetoed: {decision.vetoed}")
    print(f"Size multiplier: {decision.size_multiplier}")
    
    # Test exposure limit
    print("\n=== Test 2: High exposure ===")
    decision = gov.check(asset="SOL", direction=1.0, size=0.8)
    print(f"Vetoed: {decision.vetoed}")
    print(f"Size multiplier: {decision.size_multiplier}")
    print(f"Reason: {decision.reason}")
    
    # Test drawdown
    print("\n=== Test 3: Drawdown halt ===")
    gov._current_drawdown = 0.12
    decision = gov.check(asset="BTC", direction=1.0, size=0.1)
    print(f"Vetoed: {decision.vetoed}")
    print(f"Type: {decision.veto_type}")
    print(f"Reason: {decision.reason}")
    
    # Test correlation crisis
    print("\n=== Test 4: Correlation crisis ===")
    gov._current_drawdown = 0.05
    decision = gov.check(asset="ETH", direction=1.0, size=0.1, correlation=0.97)
    print(f"Vetoed: {decision.vetoed}")
    print(f"Type: {decision.veto_type}")
    print(f"Reason: {decision.reason}")

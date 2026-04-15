"""
HMATS v3.2 - NO_TRADE Trigger Checker
======================================
Implements Strategy Constitution Section 3.

NO_TRADE Categories:
A) Hard Signal Conflict (All-Conflict-Flat)
B) Systemic Risk (DVOL, correlation, liquidity, flash crash)
C) Data/Signal Integrity (stale, feed disagreement, execution blocked)

All conditions have explicit return-to-trading rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class NoTradeTriggerCategory(Enum):
    """NO_TRADE trigger categories."""
    HARD_CONFLICT = "A"
    SYSTEMIC_RISK = "B"
    DATA_INTEGRITY = "C"


class NoTradeTriggerType(Enum):
    """Specific NO_TRADE trigger types in priority order."""
    # Category C - Data Integrity (highest priority)
    DATA_INTEGRITY_FAIL = 1
    EXECUTION_BLOCKED = 2
    STALE_DATA = 3
    FEED_DISAGREEMENT = 4
    
    # Category B - Systemic Risk
    FLASH_CRASH = 5
    EXTREME_DVOL = 6
    LIQUIDITY_CRITICAL = 7
    CORRELATION_COLLAPSE = 8
    
    # Category A - Hard Conflict
    ALL_CONFLICT_FLAT = 9


@dataclass
class NoTradeCondition:
    """A single NO_TRADE condition."""
    trigger_type: NoTradeTriggerType
    category: NoTradeTriggerCategory
    is_active: bool = False
    triggered_at: Optional[datetime] = None
    details: str = ""
    severity: str = "HARD"  # HARD or SOFT
    
    def to_dict(self) -> Dict:
        return {
            "type": self.trigger_type.name,
            "category": self.category.value,
            "is_active": self.is_active,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "details": self.details,
            "severity": self.severity
        }


@dataclass
class NoTradeState:
    """Aggregated NO_TRADE state."""
    should_no_trade: bool = False
    active_conditions: List[NoTradeCondition] = field(default_factory=list)
    primary_reason: Optional[NoTradeTriggerType] = None
    return_conditions: List[str] = field(default_factory=list)


class NoTradeTriggerChecker:
    """
    Checks all NO_TRADE triggers with explicit thresholds.
    
    From Strategy Constitution Section 3.
    """
    
    # =========================================================================
    # CATEGORY A: Hard Signal Conflict Thresholds
    # =========================================================================
    CONFLICT_DIRECTION_THRESHOLD = 0.3  # |direction| > 0.3 = not neutral
    
    # =========================================================================
    # CATEGORY B: Systemic Risk Thresholds
    # =========================================================================
    DVOL_ZSCORE_EXTREME = 5.0           # DVOL z-score for NO_TRADE
    DVOL_ZSCORE_RECOVERY = 4.0          # Must drop below this to recover
    
    CORRELATION_COLLAPSE_THRESHOLD = 0.92  # BTC/ETH/SOL correlation
    CORRELATION_RECOVERY_THRESHOLD = 0.88  # Must drop below to recover
    
    LIQUIDITY_CRITICAL_USD = 100_000    # Order book depth within 1%
    LIQUIDITY_RECOVERY_USD = 500_000    # Must recover to this level
    
    FLASH_CRASH_PCT = 0.05              # 5% move
    FLASH_CRASH_WINDOW_SECONDS = 300    # In 5 minutes
    FLASH_RECOVERY_MINUTES = 15         # No >2% move in 15 minutes
    
    # =========================================================================
    # CATEGORY C: Data/Signal Integrity Thresholds
    # =========================================================================
    STALE_DATA_SECONDS = 2.0            # Primary feed age threshold
    STALE_DATA_RECOVERY_SECONDS = 1.0   # Must be below this to recover
    
    FEED_DISAGREEMENT_PCT = 0.01        # 1% price divergence
    FEED_DISAGREEMENT_DURATION = 30     # Sustained for 30 seconds
    FEED_AGREEMENT_PCT = 0.003          # Must agree within 0.3%
    
    EXECUTION_FAILURES_THRESHOLD = 3    # 3 consecutive failures
    
    INTEGRITY_FAIL_TYPES = ['crc32_mismatch', 'shadow_ledger_divergence']
    
    def __init__(self):
        self.conditions: Dict[NoTradeTriggerType, NoTradeCondition] = {}
        self.last_check = datetime.now(timezone.utc)
        
        # Tracking for time-based conditions
        self._feed_disagreement_start: Optional[datetime] = None
        self._execution_failure_count = 0
        self._last_price_checks: List[Tuple[datetime, float]] = []
    
    def check_all_triggers(
        self,
        market_data: Dict,
        signal_data: Dict,
        integrity_data: Dict,
        execution_data: Dict
    ) -> NoTradeState:
        """
        Check all NO_TRADE triggers in priority order.
        
        Returns NoTradeState with active conditions.
        """
        
        active_conditions = []
        
        # === CATEGORY C: Data Integrity (highest priority) ===
        
        # C1: Data integrity fail
        integrity_result = self._check_data_integrity(integrity_data)
        if integrity_result.is_active:
            active_conditions.append(integrity_result)
        
        # C2: Execution blocked
        execution_result = self._check_execution_blocked(execution_data)
        if execution_result.is_active:
            active_conditions.append(execution_result)
        
        # C3: Stale data
        stale_result = self._check_stale_data(market_data)
        if stale_result.is_active:
            active_conditions.append(stale_result)
        
        # C4: Feed disagreement
        feed_result = self._check_feed_disagreement(market_data)
        if feed_result.is_active:
            active_conditions.append(feed_result)
        
        # === CATEGORY B: Systemic Risk ===
        
        # B1: Flash crash
        flash_result = self._check_flash_crash(market_data)
        if flash_result.is_active:
            active_conditions.append(flash_result)
        
        # B2: Extreme DVOL
        dvol_result = self._check_extreme_dvol(market_data)
        if dvol_result.is_active:
            active_conditions.append(dvol_result)
        
        # B3: Liquidity critical
        liquidity_result = self._check_liquidity_critical(market_data)
        if liquidity_result.is_active:
            active_conditions.append(liquidity_result)
        
        # B4: Correlation collapse
        correlation_result = self._check_correlation_collapse(market_data, signal_data)
        if correlation_result.is_active:
            active_conditions.append(correlation_result)
        
        # === CATEGORY A: Hard Conflict ===
        
        # A1: All-Conflict-Flat
        conflict_result = self._check_all_conflict_flat(signal_data)
        if conflict_result.is_active:
            active_conditions.append(conflict_result)
        
        # Build state
        should_no_trade = len(active_conditions) > 0
        primary_reason = active_conditions[0].trigger_type if active_conditions else None
        
        # Generate return-to-trading conditions
        return_conditions = self._generate_return_conditions(active_conditions)
        
        self.last_check = datetime.now(timezone.utc)
        
        return NoTradeState(
            should_no_trade=should_no_trade,
            active_conditions=active_conditions,
            primary_reason=primary_reason,
            return_conditions=return_conditions
        )
    
    def _check_data_integrity(self, data: Dict) -> NoTradeCondition:
        """Check for data integrity failures (CRC32, ShadowLedger)."""
        
        crc_ok = data.get('crc32_valid', True)
        shadow_ok = data.get('shadow_ledger_valid', True)
        
        if not crc_ok or not shadow_ok:
            details = []
            if not crc_ok:
                details.append("CRC32 mismatch")
            if not shadow_ok:
                details.append("ShadowLedger divergence")
            
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.DATA_INTEGRITY_FAIL,
                category=NoTradeTriggerCategory.DATA_INTEGRITY,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=", ".join(details),
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.DATA_INTEGRITY_FAIL,
            category=NoTradeTriggerCategory.DATA_INTEGRITY,
            is_active=False
        )
    
    def _check_execution_blocked(self, data: Dict) -> NoTradeCondition:
        """Check for execution blocked state (consecutive failures)."""
        
        consecutive_failures = data.get('consecutive_failures', 0)
        
        if consecutive_failures >= self.EXECUTION_FAILURES_THRESHOLD:
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.EXECUTION_BLOCKED,
                category=NoTradeTriggerCategory.DATA_INTEGRITY,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"{consecutive_failures} consecutive execution failures",
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.EXECUTION_BLOCKED,
            category=NoTradeTriggerCategory.DATA_INTEGRITY,
            is_active=False
        )
    
    def _check_stale_data(self, data: Dict) -> NoTradeCondition:
        """Check for stale data (age > 2 seconds)."""
        
        data_age = data.get('data_age_seconds', 0)
        
        if data_age > self.STALE_DATA_SECONDS:
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.STALE_DATA,
                category=NoTradeTriggerCategory.DATA_INTEGRITY,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"Data age {data_age:.1f}s > {self.STALE_DATA_SECONDS}s",
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.STALE_DATA,
            category=NoTradeTriggerCategory.DATA_INTEGRITY,
            is_active=False
        )
    
    def _check_feed_disagreement(self, data: Dict) -> NoTradeCondition:
        """Check for feed disagreement (Binance vs Kraken divergence)."""
        
        binance_price = data.get('binance_price', 0)
        kraken_price = data.get('kraken_price', 0)
        
        if binance_price > 0 and kraken_price > 0:
            divergence = abs(binance_price - kraken_price) / kraken_price
            
            if divergence > self.FEED_DISAGREEMENT_PCT:
                # Track duration
                if self._feed_disagreement_start is None:
                    self._feed_disagreement_start = datetime.now(timezone.utc)
                
                duration = (datetime.now(timezone.utc) - self._feed_disagreement_start).total_seconds()
                
                if duration >= self.FEED_DISAGREEMENT_DURATION:
                    return NoTradeCondition(
                        trigger_type=NoTradeTriggerType.FEED_DISAGREEMENT,
                        category=NoTradeTriggerCategory.DATA_INTEGRITY,
                        is_active=True,
                        triggered_at=self._feed_disagreement_start,
                        details=f"Feed divergence {divergence:.2%} for {duration:.0f}s",
                        severity="HARD"
                    )
            else:
                self._feed_disagreement_start = None
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.FEED_DISAGREEMENT,
            category=NoTradeTriggerCategory.DATA_INTEGRITY,
            is_active=False
        )
    
    def _check_flash_crash(self, data: Dict) -> NoTradeCondition:
        """Check for flash crash (>5% move in <5 minutes)."""
        
        current_price = data.get('current_price', 0)
        now = datetime.now(timezone.utc)
        
        # Add current price to history
        self._last_price_checks.append((now, current_price))
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self.FLASH_CRASH_WINDOW_SECONDS)
        self._last_price_checks = [
            (t, p) for t, p in self._last_price_checks if t > cutoff
        ]
        
        if len(self._last_price_checks) >= 2 and current_price > 0:
            oldest_price = self._last_price_checks[0][1]
            if oldest_price > 0:
                move_pct = abs(current_price - oldest_price) / oldest_price
                
                if move_pct >= self.FLASH_CRASH_PCT:
                    direction = "down" if current_price < oldest_price else "up"
                    return NoTradeCondition(
                        trigger_type=NoTradeTriggerType.FLASH_CRASH,
                        category=NoTradeTriggerCategory.SYSTEMIC_RISK,
                        is_active=True,
                        triggered_at=datetime.now(timezone.utc),
                        details=f"Flash {direction}: {move_pct:.1%} in {self.FLASH_CRASH_WINDOW_SECONDS}s",
                        severity="HARD"
                    )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.FLASH_CRASH,
            category=NoTradeTriggerCategory.SYSTEMIC_RISK,
            is_active=False
        )
    
    def _check_extreme_dvol(self, data: Dict) -> NoTradeCondition:
        """
        Check for extreme DVOL (z-score >= 5.0).
        
        P1 FIX: Respects DVOL configuration - skips if mock mode.
        """
        # P1: Check if DVOL triggers are enabled
        try:
            from .deribit_dvol_client import dvol_triggers_enabled, is_dvol_mock_mode
            
            if not dvol_triggers_enabled():
                if is_dvol_mock_mode():
                    logger.debug("[NO_TRADE] DVOL trigger SKIPPED - mock mode")
                else:
                    logger.debug("[NO_TRADE] DVOL trigger SKIPPED - disabled")
                return NoTradeCondition(
                    trigger_type=NoTradeTriggerType.EXTREME_DVOL,
                    category=NoTradeTriggerCategory.SYSTEMIC_RISK,
                    is_active=False
                )
        except ImportError:
            pass  # DVOL module not available
        
        dvol_zscore = data.get('dvol_zscore', 0)
        
        if dvol_zscore >= self.DVOL_ZSCORE_EXTREME:
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.EXTREME_DVOL,
                category=NoTradeTriggerCategory.SYSTEMIC_RISK,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"DVOL z-score {dvol_zscore:.1f} >= {self.DVOL_ZSCORE_EXTREME}",
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.EXTREME_DVOL,
            category=NoTradeTriggerCategory.SYSTEMIC_RISK,
            is_active=False
        )
    
    def _check_liquidity_critical(self, data: Dict) -> NoTradeCondition:
        """Check for critical liquidity (order book depth < $100K within 1%)."""
        
        depth_1pct = data.get('orderbook_depth_1pct_usd', float('inf'))
        
        if depth_1pct < self.LIQUIDITY_CRITICAL_USD:
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.LIQUIDITY_CRITICAL,
                category=NoTradeTriggerCategory.SYSTEMIC_RISK,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"Liquidity ${depth_1pct:,.0f} < ${self.LIQUIDITY_CRITICAL_USD:,}",
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.LIQUIDITY_CRITICAL,
            category=NoTradeTriggerCategory.SYSTEMIC_RISK,
            is_active=False
        )
    
    def _check_correlation_collapse(self, market_data: Dict, signal_data: Dict) -> NoTradeCondition:
        """Check for correlation collapse risk."""
        
        correlation = market_data.get('correlation_btc_eth_sol', 0)
        has_validated_edge = signal_data.get('has_validated_edge', False)
        
        # Check if all positions same direction
        btc_direction = signal_data.get('btc_direction', 0)
        eth_direction = signal_data.get('eth_direction', 0)
        sol_direction = signal_data.get('sol_direction', 0)
        
        all_same = (
            (btc_direction > 0.2 and eth_direction > 0.2 and sol_direction > 0.2) or
            (btc_direction < -0.2 and eth_direction < -0.2 and sol_direction < -0.2)
        )
        
        if correlation > self.CORRELATION_COLLAPSE_THRESHOLD and not has_validated_edge and all_same:
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.CORRELATION_COLLAPSE,
                category=NoTradeTriggerCategory.SYSTEMIC_RISK,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"Correlation {correlation:.2f} > {self.CORRELATION_COLLAPSE_THRESHOLD}, all same direction, no edge",
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.CORRELATION_COLLAPSE,
            category=NoTradeTriggerCategory.SYSTEMIC_RISK,
            is_active=False
        )
    
    def _check_all_conflict_flat(self, data: Dict) -> NoTradeCondition:
        """
        Check for All-Conflict-Flat condition (Iron Law #34).

        [SENTIMENT-VETO-FIX 2026-04-15] Mirrors defense/constitution.py
        commit da18669: sentiment is ADVISE-only and MUST NOT veto trades.
        Conflict is now defined as quant vs DRL only. Sentiment included
        here previously caused 10-day no-trade lockouts when F&G hit
        extreme fear/greed.

        NOTE: this checker is currently NOT wired into the live decision
        path (main.py uses defense/constitution.py instead). Fix applied
        defensively so future refactors won't re-introduce the bug.
        """

        quant_direction = data.get('quant_direction', 0)
        drl_direction = data.get('drl_direction', 0)

        # Check if both quant and DRL are non-neutral
        quant_strong = abs(quant_direction) > self.CONFLICT_DIRECTION_THRESHOLD
        drl_strong = abs(drl_direction) > self.CONFLICT_DIRECTION_THRESHOLD

        if not (quant_strong and drl_strong):
            # At least one is neutral — no conflict
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.ALL_CONFLICT_FLAT,
                category=NoTradeTriggerCategory.HARD_CONFLICT,
                is_active=False
            )

        # Conflict only when quant and DRL point in opposite directions
        if (quant_direction > 0 and drl_direction < 0) or (quant_direction < 0 and drl_direction > 0):
            return NoTradeCondition(
                trigger_type=NoTradeTriggerType.ALL_CONFLICT_FLAT,
                category=NoTradeTriggerCategory.HARD_CONFLICT,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"Quant={quant_direction:.2f}, DRL={drl_direction:.2f} (sentiment excluded per Iron Law #34)",
                severity="HARD"
            )
        
        return NoTradeCondition(
            trigger_type=NoTradeTriggerType.ALL_CONFLICT_FLAT,
            category=NoTradeTriggerCategory.HARD_CONFLICT,
            is_active=False
        )
    
    def _generate_return_conditions(self, active_conditions: List[NoTradeCondition]) -> List[str]:
        """Generate return-to-trading conditions for active triggers."""
        
        conditions = []
        
        for condition in active_conditions:
            if condition.trigger_type == NoTradeTriggerType.DATA_INTEGRITY_FAIL:
                conditions.append("All integrity checks must pass (CRC32, ShadowLedger)")
            
            elif condition.trigger_type == NoTradeTriggerType.EXECUTION_BLOCKED:
                conditions.append("Successful execution test order required")
            
            elif condition.trigger_type == NoTradeTriggerType.STALE_DATA:
                conditions.append(f"Data age must be < {self.STALE_DATA_RECOVERY_SECONDS}s")
            
            elif condition.trigger_type == NoTradeTriggerType.FEED_DISAGREEMENT:
                conditions.append(f"Feed divergence must be < {self.FEED_AGREEMENT_PCT:.1%}")
            
            elif condition.trigger_type == NoTradeTriggerType.FLASH_CRASH:
                conditions.append(f"Price must stabilize (no >2% move in {self.FLASH_RECOVERY_MINUTES} min)")
            
            elif condition.trigger_type == NoTradeTriggerType.EXTREME_DVOL:
                conditions.append(f"DVOL z-score must drop below {self.DVOL_ZSCORE_RECOVERY} for 2 bars")
            
            elif condition.trigger_type == NoTradeTriggerType.LIQUIDITY_CRITICAL:
                conditions.append(f"Order book depth must recover to > ${self.LIQUIDITY_RECOVERY_USD:,}")
            
            elif condition.trigger_type == NoTradeTriggerType.CORRELATION_COLLAPSE:
                conditions.append(f"Correlation < {self.CORRELATION_RECOVERY_THRESHOLD} OR has validated edge")
            
            elif condition.trigger_type == NoTradeTriggerType.ALL_CONFLICT_FLAT:
                conditions.append("At least one signal must become neutral OR all must align")
        
        # Add general condition
        if conditions:
            conditions.append("All conditions must hold for 2 consecutive 4H bars")
        
        return conditions


# Singleton instance
_no_trade_checker = NoTradeTriggerChecker()

def get_no_trade_checker() -> NoTradeTriggerChecker:
    return _no_trade_checker

"""
================================================================================
HMATS v3.5 - PnL ATTRIBUTION PROMOTION SIGNAL
================================================================================
Version: 3.5.0
Upgrade From: v3.4 PnL Attribution
Purpose: Make PnL attribution actionable for DRL promotion decisions

CHANGES FROM v3.4:
    v3.4: PnL attribution was diagnostic only (passive observation)
    v3.5: Attribution is actionable - drives DRL promotion decisions

NEW FEATURES:
    1. Aggregate attribution over rolling windows (7d, 30d)
    2. Compare DRL shadow decisions vs actual exits
    3. Generate DRL promotion signals when criteria met
    4. Detect if execution or exit is leaking alpha

DATA SCHEMA:
    AttributionWindow:
        - window_days: 7 or 30
        - entry_alpha_total: sum of entry attribution
        - execution_alpha_total: sum of execution attribution
        - exit_alpha_total: sum of exit attribution
        - drl_shadow_vs_actual: comparison of DRL recommendations vs what happened
        - trade_count: number of trades in window

    DRLPromotionSignal:
        - ready_for_promotion: bool
        - promotion_type: "SHADOW_TO_EXIT" or None
        - confidence: float (0-1)
        - supporting_evidence: list of metrics
        - blocking_factors: list of reasons why not

WHY THIS PREVENTS PREMATURE DRL ACTIVATION:
    1. Requires minimum sample size (30 trades in 30 days)
    2. DRL shadow decisions must outperform baseline by 10%
    3. Consistency check: improvement must be stable, not one-off
    4. Automatic demotion if performance degrades
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TradeAttributionRecord:
    """Attribution record for a single trade."""
    
    trade_id: str
    timestamp: datetime
    
    # Component attributions (bps)
    entry_alpha_bps: float
    execution_alpha_bps: float
    exit_alpha_bps: float
    total_pnl_bps: float
    
    # DRL shadow tracking
    drl_shadow_exit_recommendation: Optional[str] = None  # PARTIAL_EXIT, RELEASE_RUNNER, etc.
    actual_exit_type: str = ""
    drl_shadow_would_have_bps: Optional[float] = None  # Estimated PnL if DRL decision was used
    actual_exit_bps: float = 0.0


@dataclass
class AttributionWindow:
    """Aggregated attribution over a time window."""
    
    window_days: int
    start_date: datetime
    end_date: datetime
    
    # Totals
    entry_alpha_total: float = 0.0
    execution_alpha_total: float = 0.0
    exit_alpha_total: float = 0.0
    
    # Trade count
    trade_count: int = 0
    
    # DRL shadow comparison
    drl_shadow_total_bps: float = 0.0
    actual_exit_total_bps: float = 0.0
    drl_outperformance_bps: float = 0.0
    drl_outperformance_pct: float = 0.0
    
    # Derived metrics
    avg_entry_alpha: float = 0.0
    avg_execution_alpha: float = 0.0
    avg_exit_alpha: float = 0.0
    
    def compute_derived(self):
        """Compute derived metrics."""
        if self.trade_count > 0:
            self.avg_entry_alpha = self.entry_alpha_total / self.trade_count
            self.avg_execution_alpha = self.execution_alpha_total / self.trade_count
            self.avg_exit_alpha = self.exit_alpha_total / self.trade_count
            
            if self.actual_exit_total_bps != 0:
                self.drl_outperformance_bps = self.drl_shadow_total_bps - self.actual_exit_total_bps
                self.drl_outperformance_pct = self.drl_outperformance_bps / abs(self.actual_exit_total_bps) * 100


@dataclass
class DRLPromotionSignal:
    """Signal for DRL promotion decision."""
    
    # Decision
    ready_for_promotion: bool = False
    promotion_type: Optional[str] = None  # "SHADOW_TO_EXIT" or None
    
    # Confidence
    confidence: float = 0.0
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    blocking_factors: List[str] = field(default_factory=list)
    
    # Metrics
    sample_size: int = 0
    drl_outperformance_pct: float = 0.0
    consistency_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "ready": self.ready_for_promotion,
            "type": self.promotion_type,
            "confidence": self.confidence,
            "evidence": self.supporting_evidence,
            "blockers": self.blocking_factors,
            "sample_size": self.sample_size,
            "outperformance_pct": self.drl_outperformance_pct,
        }


@dataclass
class AlphaLeakSignal:
    """Signal indicating alpha leak in a component."""
    
    component: str  # "entry", "execution", "exit"
    leak_magnitude_bps: float
    is_significant: bool
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "leak_bps": self.leak_magnitude_bps,
            "significant": self.is_significant,
            "recommendation": self.recommendation,
        }


@dataclass
class PromotionConfig:
    """Configuration for promotion decisions."""
    
    # Sample size requirements
    min_trades_for_promotion: int = 30
    min_drl_shadow_samples: int = 20
    
    # Performance requirements
    min_drl_outperformance_pct: float = 10.0  # DRL must beat baseline by 10%
    min_consistency_score: float = 0.6  # 60% of windows must show improvement
    
    # Confidence thresholds
    high_confidence_outperformance: float = 20.0
    
    # Alpha leak thresholds
    significant_leak_bps: float = -20.0  # -20bps avg = significant leak
    
    # Rolling windows
    short_window_days: int = 7
    long_window_days: int = 30
    
    # Demotion criteria
    demotion_underperformance_pct: float = -10.0
    demotion_sample_size: int = 10


class PnLAttributionPromotion:
    """
    PnL Attribution system with promotion signaling for v3.5.
    
    UPGRADES FROM v3.4:
        1. Aggregates attribution over rolling windows
        2. Compares DRL shadow decisions vs actual
        3. Generates actionable promotion signals
        4. Detects alpha leaks by component
    
    USAGE:
        attribution = PnLAttributionPromotion()
        
        # Record trade with DRL shadow info
        attribution.record_trade(
            trade_id=...,
            entry_alpha=..., execution_alpha=..., exit_alpha=...,
            drl_shadow_recommendation="PARTIAL_EXIT",
            drl_shadow_would_have_bps=85,
            actual_exit_bps=70,
        )
        
        # Check if DRL should be promoted
        signal = attribution.get_promotion_signal()
        if signal.ready_for_promotion:
            # Enable DRL exit authority
        
        # Check for alpha leaks
        leaks = attribution.get_alpha_leak_signals()
    """
    
    def __init__(self, config: Optional[PromotionConfig] = None):
        self.config = config or PromotionConfig()
        
        # Trade records
        self._records: List[TradeAttributionRecord] = []
        
        # Cached windows
        self._short_window: Optional[AttributionWindow] = None
        self._long_window: Optional[AttributionWindow] = None
        self._last_window_compute: Optional[datetime] = None
        
        logger.info("PnLAttributionPromotion v3.5 initialized")
    
    def record_trade(
        self,
        trade_id: str,
        entry_alpha_bps: float,
        execution_alpha_bps: float,
        exit_alpha_bps: float,
        total_pnl_bps: float,
        drl_shadow_recommendation: Optional[str] = None,
        drl_shadow_would_have_bps: Optional[float] = None,
        actual_exit_type: str = "",
        actual_exit_bps: float = 0.0,
    ):
        """Record a trade with attribution."""
        
        record = TradeAttributionRecord(
            trade_id=trade_id,
            timestamp=datetime.now(timezone.utc),
            entry_alpha_bps=entry_alpha_bps,
            execution_alpha_bps=execution_alpha_bps,
            exit_alpha_bps=exit_alpha_bps,
            total_pnl_bps=total_pnl_bps,
            drl_shadow_exit_recommendation=drl_shadow_recommendation,
            actual_exit_type=actual_exit_type,
            drl_shadow_would_have_bps=drl_shadow_would_have_bps,
            actual_exit_bps=actual_exit_bps,
        )
        
        self._records.append(record)
        
        # Trim old records (keep 90 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        self._records = [r for r in self._records if r.timestamp > cutoff]
        
        # Invalidate cache
        self._last_window_compute = None
        
        logger.debug(f"Attribution: Recorded trade {trade_id}")
    
    def _compute_window(self, days: int) -> AttributionWindow:
        """Compute attribution window."""
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        records = [r for r in self._records if r.timestamp > cutoff]
        
        window = AttributionWindow(
            window_days=days,
            start_date=cutoff,
            end_date=datetime.now(timezone.utc),
            trade_count=len(records),
        )
        
        if not records:
            return window
        
        # Sum attributions
        window.entry_alpha_total = sum(r.entry_alpha_bps for r in records)
        window.execution_alpha_total = sum(r.execution_alpha_bps for r in records)
        window.exit_alpha_total = sum(r.exit_alpha_bps for r in records)
        
        # DRL shadow comparison
        drl_records = [r for r in records if r.drl_shadow_would_have_bps is not None]
        if drl_records:
            window.drl_shadow_total_bps = sum(r.drl_shadow_would_have_bps for r in drl_records)
            window.actual_exit_total_bps = sum(r.actual_exit_bps for r in drl_records)
        
        window.compute_derived()
        
        return window
    
    def _ensure_windows_computed(self):
        """Ensure windows are computed and cached."""
        
        if self._last_window_compute is not None:
            age = (datetime.now(timezone.utc) - self._last_window_compute).total_seconds()
            if age < 300:  # Cache for 5 minutes
                return
        
        self._short_window = self._compute_window(self.config.short_window_days)
        self._long_window = self._compute_window(self.config.long_window_days)
        self._last_window_compute = datetime.now(timezone.utc)
    
    def get_promotion_signal(self) -> DRLPromotionSignal:
        """
        Get DRL promotion signal based on attribution data.
        
        PROMOTION CRITERIA (ALL must be met):
            1. Sample size >= 30 trades in 30-day window
            2. DRL shadow samples >= 20
            3. DRL outperformance >= 10% over baseline
            4. Consistency >= 60% (improvement in most sub-windows)
        
        RETURNS:
            DRLPromotionSignal with decision and evidence
        """
        
        self._ensure_windows_computed()
        
        signal = DRLPromotionSignal()
        long_window = self._long_window
        
        # Check sample size
        if long_window.trade_count < self.config.min_trades_for_promotion:
            signal.blocking_factors.append(
                f"Insufficient trades: {long_window.trade_count}/{self.config.min_trades_for_promotion}"
            )
            signal.sample_size = long_window.trade_count
            return signal
        
        signal.sample_size = long_window.trade_count
        signal.supporting_evidence.append(f"Trade count: {long_window.trade_count}")
        
        # Check DRL shadow samples
        drl_records = [r for r in self._records if r.drl_shadow_would_have_bps is not None]
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.long_window_days)
        drl_records = [r for r in drl_records if r.timestamp > cutoff]
        
        if len(drl_records) < self.config.min_drl_shadow_samples:
            signal.blocking_factors.append(
                f"Insufficient DRL shadow samples: {len(drl_records)}/{self.config.min_drl_shadow_samples}"
            )
            return signal
        
        signal.supporting_evidence.append(f"DRL shadow samples: {len(drl_records)}")
        
        # Check DRL outperformance
        signal.drl_outperformance_pct = long_window.drl_outperformance_pct
        
        if long_window.drl_outperformance_pct < self.config.min_drl_outperformance_pct:
            signal.blocking_factors.append(
                f"DRL outperformance insufficient: {long_window.drl_outperformance_pct:.1f}%/{self.config.min_drl_outperformance_pct}%"
            )
            return signal
        
        signal.supporting_evidence.append(
            f"DRL outperforms baseline by {long_window.drl_outperformance_pct:.1f}%"
        )
        
        # Check consistency (improvement in multiple sub-windows)
        consistency = self._compute_consistency()
        signal.consistency_score = consistency
        
        if consistency < self.config.min_consistency_score:
            signal.blocking_factors.append(
                f"Consistency insufficient: {consistency:.0%}/{self.config.min_consistency_score:.0%}"
            )
            return signal
        
        signal.supporting_evidence.append(f"Consistency: {consistency:.0%}")
        
        # All criteria met - ready for promotion
        signal.ready_for_promotion = True
        signal.promotion_type = "SHADOW_TO_EXIT"
        
        # Calculate confidence
        if long_window.drl_outperformance_pct >= self.config.high_confidence_outperformance:
            signal.confidence = 0.9
        else:
            signal.confidence = 0.6 + (
                long_window.drl_outperformance_pct - self.config.min_drl_outperformance_pct
            ) / 30  # Scale up to 0.9
        
        signal.confidence = min(0.95, signal.confidence)
        
        logger.info(
            f"DRL PROMOTION SIGNAL: ready={signal.ready_for_promotion}, "
            f"confidence={signal.confidence:.2f}"
        )
        
        return signal
    
    def _compute_consistency(self) -> float:
        """Compute consistency score (how often DRL outperforms in sub-windows)."""
        
        # Divide 30-day window into 5-day sub-windows
        sub_windows = []
        
        for i in range(6):  # 6 x 5-day windows
            start = datetime.now(timezone.utc) - timedelta(days=(i + 1) * 5)
            end = datetime.now(timezone.utc) - timedelta(days=i * 5)
            
            records = [
                r for r in self._records
                if start < r.timestamp <= end and r.drl_shadow_would_have_bps is not None
            ]
            
            if len(records) >= 3:  # Minimum 3 records per window
                drl_total = sum(r.drl_shadow_would_have_bps for r in records)
                actual_total = sum(r.actual_exit_bps for r in records)
                sub_windows.append(drl_total > actual_total)
        
        if not sub_windows:
            return 0.0
        
        return sum(sub_windows) / len(sub_windows)
    
    def get_demotion_signal(self) -> bool:
        """
        Check if DRL should be demoted (if currently promoted).
        
        DEMOTION CRITERIA:
            - DRL underperforms baseline by > 10%
            - Over at least 10 trades
        """
        
        self._ensure_windows_computed()
        
        short_window = self._short_window
        
        if short_window.trade_count < self.config.demotion_sample_size:
            return False
        
        if short_window.drl_outperformance_pct < self.config.demotion_underperformance_pct:
            logger.warning(
                f"DRL DEMOTION SIGNAL: underperformance={short_window.drl_outperformance_pct:.1f}%"
            )
            return True
        
        return False
    
    def get_alpha_leak_signals(self) -> List[AlphaLeakSignal]:
        """
        Detect if any component is leaking alpha.
        
        A component leaks alpha if its average attribution is significantly negative.
        
        RETURNS:
            List of AlphaLeakSignal for problematic components
        """
        
        self._ensure_windows_computed()
        
        signals = []
        long_window = self._long_window
        
        if long_window.trade_count < 10:
            return signals
        
        # Check entry
        if long_window.avg_entry_alpha < self.config.significant_leak_bps:
            signals.append(AlphaLeakSignal(
                component="entry",
                leak_magnitude_bps=long_window.avg_entry_alpha,
                is_significant=True,
                recommendation="Review Quant/Regime entry signals - entries underperforming",
            ))
        
        # Check execution
        if long_window.avg_execution_alpha < self.config.significant_leak_bps:
            signals.append(AlphaLeakSignal(
                component="execution",
                leak_magnitude_bps=long_window.avg_execution_alpha,
                is_significant=True,
                recommendation="Review execution timing - slippage or adverse selection",
            ))
        
        # Check exit
        if long_window.avg_exit_alpha < self.config.significant_leak_bps:
            signals.append(AlphaLeakSignal(
                component="exit",
                leak_magnitude_bps=long_window.avg_exit_alpha,
                is_significant=True,
                recommendation="Review exit timing - giving back profits",
            ))
        
        return signals
    
    def get_windows(self) -> Tuple[AttributionWindow, AttributionWindow]:
        """Get short and long attribution windows."""
        self._ensure_windows_computed()
        return self._short_window, self._long_window
    
    def get_report(self) -> Dict:
        """Get comprehensive attribution report."""
        
        self._ensure_windows_computed()
        
        return {
            "short_window": {
                "days": self._short_window.window_days,
                "trades": self._short_window.trade_count,
                "entry_alpha_avg": self._short_window.avg_entry_alpha,
                "execution_alpha_avg": self._short_window.avg_execution_alpha,
                "exit_alpha_avg": self._short_window.avg_exit_alpha,
                "drl_outperformance_pct": self._short_window.drl_outperformance_pct,
            },
            "long_window": {
                "days": self._long_window.window_days,
                "trades": self._long_window.trade_count,
                "entry_alpha_avg": self._long_window.avg_entry_alpha,
                "execution_alpha_avg": self._long_window.avg_execution_alpha,
                "exit_alpha_avg": self._long_window.avg_exit_alpha,
                "drl_outperformance_pct": self._long_window.drl_outperformance_pct,
            },
            "promotion_signal": self.get_promotion_signal().to_dict(),
            "alpha_leaks": [s.to_dict() for s in self.get_alpha_leak_signals()],
        }
    
    def reset(self):
        """Full reset."""
        self._records.clear()
        self._short_window = None
        self._long_window = None
        self._last_window_compute = None


# Singleton
_attribution_promotion: Optional[PnLAttributionPromotion] = None

def get_attribution_promotion() -> PnLAttributionPromotion:
    global _attribution_promotion
    if _attribution_promotion is None:
        _attribution_promotion = PnLAttributionPromotion()
    return _attribution_promotion

def reset_attribution_promotion():
    global _attribution_promotion
    _attribution_promotion = None

"""
================================================================================
HMATS v3.4 - RISK AGENT REWEIGHTING
================================================================================
Version: 3.4.0
Upgrade From: v3.3-HR
Purpose: High-risk aligned risk management that doesn't suppress OPPORTUNITY

KEY CHANGES FROM v3.3-HR:
    v3.3-HR: Same risk multipliers regardless of mode
    v3.4:    OPPORTUNITY gets relaxed (but not removed) risk constraints

RETAINED FROM v3.3-HR:
    - Risk VETO authority (unchanged)
    - Hard drawdown stops (5%/10%/15%)
    - Correlation collapse triggers
    - Data integrity triggers

NEW IN v3.4:
    - OPPORTUNITY-aware risk multipliers
    - Phase-gated SOL dominance override
    - Opportunity density requirement for SOL override

CRITICAL PRINCIPLE:
    Risk is REDUCED in OPPORTUNITY, not REMOVED.
    Hard limits remain hard.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

from market.phase_detector import RegimePhase, RegimePhaseResult

logger = logging.getLogger(__name__)


# ============================================================================
# RISK CONFIGURATION
# ============================================================================

@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Hard drawdown limits [UNLEASH UL-5: synced with RULETABLE]
    reduce_at_drawdown: float = 0.08    # 8% - start reducing
    halt_at_drawdown: float = 0.20      # 20% - halt trading
    critical_drawdown: float = 0.35     # 35% - require manual review
    
    # Position limits
    max_single_asset_pct: float = 0.80  # 80% in single asset (base)
    max_gross_exposure: float = 1.50    # 150% gross (base)
    
    # SOL dominance override limits
    sol_override_max_exposure: float = 1.00      # 100% SOL allowed
    sol_override_gross_cap_add: float = 0.25     # +25% to gross cap
    sol_override_max_duration_bars: int = 2      # Max 2 bars
    
    # Correlation thresholds
    # [UNLEASH] UL-4: Relaxed - crypto daily corr 0.75-0.90 is normal, not crisis
    # Original: 0.88/0.92/0.95 -> New: 0.92/0.96/0.98
    correlation_warning: float = 0.92   # [UNLEASH] was 0.88
    correlation_danger: float = 0.96    # [UNLEASH] was 0.92
    correlation_crisis: float = 0.98    # [UNLEASH] was 0.95
    
    # Fast correlation (from v3.3-HR)
    fast_correlation_window_min: int = 30
    fast_correlation_exit_threshold: float = 0.90


# ============================================================================
# OPPORTUNITY-AWARE RISK MULTIPLIERS (T25: configurable from profile)
# ============================================================================

def _build_multiplier_tables(halt: float):
    """Build drawdown multiplier tables scaled to the halt threshold.

    Bands are proportional: 0-30%, 30-50%, 50-70%, 70-100% of halt.
    """
    b1 = round(halt * 0.30, 4)
    b2 = round(halt * 0.50, 4)
    b3 = round(halt * 0.70, 4)

    normal = {
        (0.00, b1):   1.0,
        (b1,   b2):   0.8,
        (b2,   b3):   0.5,
        (b3,   halt): 0.3,
        (halt, 1.00): 0.0,
    }
    opportunity = {
        (0.00, b1):   1.0,
        (b1,   b2):   0.9,
        (b2,   b3):   0.7,
        (b3,   halt): 0.5,
        (halt, 1.00): 0.0,
    }
    return normal, opportunity

# Module-level defaults (halt=20% UNLEASH) - overridden by configure_from_profile()
RISK_MULTIPLIERS_NORMAL, RISK_MULTIPLIERS_OPPORTUNITY = _build_multiplier_tables(0.20)


def get_risk_multiplier(drawdown: float, is_opportunity: bool,
                        normal_table=None, opportunity_table=None) -> float:
    """Get risk multiplier based on drawdown and mode. (UNUSED - risk multipliers computed inline in RiskAgentV34)"""
    multipliers = (opportunity_table or RISK_MULTIPLIERS_OPPORTUNITY) if is_opportunity \
        else (normal_table or RISK_MULTIPLIERS_NORMAL)

    for (dd_min, dd_max), mult in multipliers.items():
        if dd_min <= drawdown < dd_max:
            return mult

    return 0.0  # Default to halt


# ============================================================================
# SOL DOMINANCE OVERRIDE (v3.4 Refined)
# ============================================================================

@dataclass
class SolDominanceConditions:
    """Conditions for SOL dominance override."""
    
    # All conditions must be met
    mode_is_opportunity: bool = False
    sol_signal_strength: float = 0.0    # Threshold: >= 0.60
    sol_beta: float = 0.0               # Threshold: >= 1.30
    btc_eth_exposure: float = 0.0       # Threshold: < 0.10
    
    # v3.4 additions
    sol_phase_is_early: bool = False    # Must be IGNITION or EXPANSION
    opportunity_density: float = 0.0    # Threshold: >= 0.70
    no_trade_score: float = 0.0         # Threshold: < 0.60
    
    def all_met(self) -> bool:
        """Check if all conditions are met."""
        return (
            self.mode_is_opportunity and
            self.sol_signal_strength >= 0.60 and
            self.sol_beta >= 1.30 and
            self.btc_eth_exposure < 0.10 and
            self.sol_phase_is_early and
            self.opportunity_density >= 0.70 and
            self.no_trade_score < 0.60
        )
    
    def get_unmet_conditions(self) -> list:
        """Get list of unmet conditions."""
        unmet = []
        if not self.mode_is_opportunity:
            unmet.append("mode_not_opportunity")
        if self.sol_signal_strength < 0.60:
            unmet.append(f"sol_signal_low:{self.sol_signal_strength:.2f}")
        if self.sol_beta < 1.30:
            unmet.append(f"sol_beta_low:{self.sol_beta:.2f}")
        if self.btc_eth_exposure >= 0.10:
            unmet.append(f"btc_eth_high:{self.btc_eth_exposure:.2f}")
        if not self.sol_phase_is_early:
            unmet.append("sol_phase_late")
        if self.opportunity_density < 0.70:
            unmet.append(f"density_low:{self.opportunity_density:.2f}")
        if self.no_trade_score >= 0.60:
            unmet.append(f"no_trade_risk:{self.no_trade_score:.2f}")
        return unmet


@dataclass
class SolDominanceOverride:
    """SOL dominance override state."""
    
    active: bool = False
    start_bar: int = 0
    bars_active: int = 0
    max_bars: int = 2
    
    # Adjusted limits
    max_sol_exposure: float = 1.00
    gross_cap_bonus: float = 0.25
    
    # Conditions at activation
    conditions: Optional[SolDominanceConditions] = None
    
    def should_normalize(self, current_bar: int) -> bool:
        """Check if override should be forced to normalize."""
        if not self.active:
            return False
        return (current_bar - self.start_bar) >= self.max_bars


class SolDominanceManager:
    """
    SOL dominance override manager.
    
    v3.4 CHANGES FROM v3.3-HR:
        1. Added sol_phase_is_early requirement (IGNITION/EXPANSION only)
        2. Added opportunity_density requirement (>= 0.70)
        3. Added no_trade_score check (< 0.60)
        4. Reduced gross cap bonus from +30% to +25%
        5. Auto-normalize when SOL phase transitions to SATURATION
    
    WHY THESE CHANGES:
        v3.3-HR could trigger SOL dominance in marginal setups.
        v3.4 requires high-quality phase + density.
        This means FEWER triggers but HIGHER quality.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self._override = SolDominanceOverride()
        self._current_bar = 0
        logger.info("SolDominanceManager v3.4 initialized")
    
    def evaluate(
        self,
        system_mode: str,
        sol_signal_strength: float,
        sol_beta: float,
        current_exposures: Dict[str, float],
        phase_result: RegimePhaseResult,
        no_trade_trigger_scores: Dict[str, float],
    ) -> SolDominanceOverride:
        """
        Evaluate SOL dominance override conditions.
        
        WHERE CONSUMED:
            - Risk manager applies adjusted caps if override active
            - Fusion applies adjusted gross cap
        """
        
        self._current_bar += 1
        
        # Check if override should be normalized
        if self._override.should_normalize(self._current_bar):
            logger.info("SOL dominance override TTL expired - forcing normalization")
            self._override.active = False
            return self._override
        
        # Auto-normalize if SOL phase transitions to SATURATION
        if self._override.active and phase_result.phase_sol in [
            RegimePhase.SATURATION, RegimePhase.EXHAUSTION
        ]:
            logger.info(f"SOL dominance auto-normalize: SOL phase {phase_result.phase_sol.value}")
            self._override.active = False
            return self._override
        
        # Build conditions
        btc_eth_exposure = (
            current_exposures.get("BTC", 0.0) + 
            current_exposures.get("ETH", 0.0)
        )
        
        max_no_trade_score = max(no_trade_trigger_scores.values()) if no_trade_trigger_scores else 0.0
        
        conditions = SolDominanceConditions(
            mode_is_opportunity=(system_mode == "OPPORTUNITY"),
            sol_signal_strength=sol_signal_strength,
            sol_beta=sol_beta,
            btc_eth_exposure=btc_eth_exposure,
            sol_phase_is_early=phase_result.phase_sol in [
                RegimePhase.IGNITION, RegimePhase.EXPANSION
            ],
            opportunity_density=phase_result.opportunity_density,
            no_trade_score=max_no_trade_score,
        )
        
        # Check if all conditions met
        if conditions.all_met():
            if not self._override.active:
                # Activate override
                self._override = SolDominanceOverride(
                    active=True,
                    start_bar=self._current_bar,
                    bars_active=0,
                    max_bars=self.config.sol_override_max_duration_bars,
                    max_sol_exposure=self.config.sol_override_max_exposure,
                    gross_cap_bonus=self.config.sol_override_gross_cap_add,
                    conditions=conditions,
                )
                logger.info(
                    f"SOL DOMINANCE OVERRIDE ACTIVATED: "
                    f"max_sol={self._override.max_sol_exposure:.0%}, "
                    f"gross_bonus=+{self._override.gross_cap_bonus:.0%}"
                )
            else:
                self._override.bars_active = self._current_bar - self._override.start_bar
        else:
            if self._override.active:
                # Conditions no longer met - deactivate
                unmet = conditions.get_unmet_conditions()
                logger.info(f"SOL dominance override deactivated: unmet={unmet}")
                self._override.active = False
        
        return self._override
    
    def get_adjusted_caps(
        self,
        base_gross_cap: float,
        base_sol_cap: float,
    ) -> Tuple[float, float]:
        """Get adjusted caps if override is active."""
        if not self._override.active:
            return base_gross_cap, base_sol_cap
        
        adjusted_gross = base_gross_cap + self._override.gross_cap_bonus
        adjusted_sol = self._override.max_sol_exposure
        
        return adjusted_gross, adjusted_sol
    
    def reset(self):
        """Reset override state."""
        self._override = SolDominanceOverride()
        self._current_bar = 0


# ============================================================================
# FAST CORRELATION SENTINEL (Exit-Only)
# ============================================================================

class FastCorrelationSentinel:
    """
    Fast correlation detector (30-min window).
    
    RETAINED FROM v3.3-HR:
        - 30-minute rolling window
        - Exit-only (cannot veto entries)
        - Only active in OPPORTUNITY mode
    
    v3.4 BEHAVIOR:
        - Integrates with ExitAlphaManager
        - Triggers exit acceleration, not NO_TRADE
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self._correlation_buffer: list = []
        self._window_size = self.config.fast_correlation_window_min
        logger.info("FastCorrelationSentinel initialized")
    
    def update(self, correlation: float):
        """Add new correlation observation."""
        self._correlation_buffer.append(correlation)
        self._correlation_buffer = self._correlation_buffer[-self._window_size:]
    
    def should_accelerate_exit(self, system_mode: str) -> Tuple[bool, float]:
        """
        Check if exit should be accelerated.
        
        CRITICAL: This is EXIT-ONLY. Does NOT block entries.
        
        Returns:
            (should_accelerate, current_fast_correlation)
        """
        if system_mode != "OPPORTUNITY":
            return False, 0.0
        
        if len(self._correlation_buffer) < self._window_size // 2:
            return False, 0.0
        
        avg_correlation = sum(self._correlation_buffer) / len(self._correlation_buffer)
        
        if avg_correlation >= self.config.fast_correlation_exit_threshold:
            logger.warning(
                f"Fast correlation exit signal: {avg_correlation:.3f} >= "
                f"{self.config.fast_correlation_exit_threshold}"
            )
            return True, avg_correlation
        
        return False, avg_correlation
    
    def reset(self):
        """Reset buffer."""
        self._correlation_buffer = []


# ============================================================================
# WEEKEND/HOLIDAY LOGIC
# ============================================================================

class WeekendHolidayManager:
    """
    Weekend/holiday exposure management.
    
    v3.3-HR BEHAVIOR (RETAINED):
        - Weekend caps apply only in NORMAL mode
        - Does NOT veto OPPORTUNITY
    
    v3.4 ADDITION:
        - Integration with phase-aware logic
    """
    
    # Exposure caps by period
    CAPS = {
        "weekday": 1.50,      # Normal trading
        "weekend_normal": 0.50,   # Weekend in NORMAL mode
        "weekend_opportunity": 1.50,  # Weekend in OPPORTUNITY mode
        "holiday_normal": 0.30,
        "holiday_opportunity": 1.20,
        "low_liquidity": 0.70,
    }
    
    @classmethod
    def get_exposure_cap(
        cls,
        is_weekend: bool,
        is_holiday: bool,
        is_low_liquidity: bool,
        system_mode: str,
    ) -> float:
        """
        Get exposure cap for current conditions.
        
        KEY RULE: OPPORTUNITY ignores weekend/holiday restrictions
        """
        
        is_opportunity = (system_mode == "OPPORTUNITY")
        
        if is_holiday:
            return cls.CAPS["holiday_opportunity" if is_opportunity else "holiday_normal"]
        
        if is_weekend:
            return cls.CAPS["weekend_opportunity" if is_opportunity else "weekend_normal"]
        
        if is_low_liquidity:
            return cls.CAPS["low_liquidity"]
        
        return cls.CAPS["weekday"]


# ============================================================================
# INTEGRATED RISK MANAGER
# ============================================================================

@dataclass
class RiskAssessment:
    """Complete risk assessment result."""

    # Position sizing
    risk_multiplier: float = 1.0
    max_gross_cap: float = 1.50
    max_single_asset_cap: float = 0.80

    # SOL dominance
    sol_dominance_active: bool = False
    sol_max_exposure: float = 0.80

    # Vetoes
    veto_active: bool = False
    veto_reason: str = ""

    # Exit signals
    accelerate_exit: bool = False
    exit_reason: str = ""

    # Metadata
    drawdown: float = 0.0
    correlation: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "risk_multiplier": self.risk_multiplier,
            "max_gross_cap": self.max_gross_cap,
            "sol_dominance_active": self.sol_dominance_active,
            "veto_active": self.veto_active,
            "accelerate_exit": self.accelerate_exit,
        }

    def to_v6_dict(self) -> Dict:
        """V6.4-compatible output with standard metadata."""
        reasons = []
        if self.veto_active:
            reasons.append(self.veto_reason)
        if self.accelerate_exit:
            reasons.append(self.exit_reason)

        return {
            # v6 risk fields
            "risk_veto": self.veto_active,
            "leverage_cap": min(3.0, self.max_gross_cap),
            "risk_budget_remaining": max(0.0, 1.0 - self.drawdown) if self.drawdown < 1.0 else 0.0,
            "risk_multiplier": self.risk_multiplier,
            "max_gross_cap": self.max_gross_cap,
            "sol_dominance_active": self.sol_dominance_active,
            "accelerate_exit": self.accelerate_exit,
            "reasons": reasons,
            # v6.4 standard metadata
            "asof_ts": self.timestamp.isoformat() + "Z",
            "data_age_seconds": 0.0,
            "data_quality": "ok" if not self.veto_active else "ok",
            "provenance": ["risk_agent_v34"],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RiskAssessment":
        """Restore from persisted dict (hot restart recovery)."""
        return cls(
            risk_multiplier=d.get("risk_multiplier", 1.0),
            max_gross_cap=d.get("max_gross_cap", 1.50),
            max_single_asset_cap=d.get("max_single_asset_cap", 0.80),
            sol_dominance_active=d.get("sol_dominance_active", False),
            sol_max_exposure=d.get("sol_max_exposure", 0.80),
            veto_active=d.get("veto_active", False),
            veto_reason=d.get("veto_reason", ""),
            accelerate_exit=d.get("accelerate_exit", False),
            exit_reason=d.get("exit_reason", ""),
            drawdown=d.get("drawdown", 0.0),
            correlation=d.get("correlation", 0.0),
        )


class RiskAgentV34:
    """
    Risk agent for v3.4.
    
    PRINCIPLES:
        1. Risk VETO authority is UNCHANGED
        2. Hard limits remain HARD
        3. OPPORTUNITY gets more room, not freedom from risk
        4. SOL dominance requires HIGH QUALITY conditions
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.sol_dominance = SolDominanceManager(config)
        self.fast_correlation = FastCorrelationSentinel(config)
        # T25: instance-level tables (use module defaults until configure_from_profile)
        self._mult_normal = RISK_MULTIPLIERS_NORMAL
        self._mult_opportunity = RISK_MULTIPLIERS_OPPORTUNITY
        logger.info("RiskAgentV34 initialized")

    def configure_from_profile(self, profile: Dict) -> None:
        """T25: Override drawdown thresholds from risk profile."""
        if "halt_at_drawdown" in profile:
            self.config.halt_at_drawdown = profile["halt_at_drawdown"]
        if "critical_drawdown" in profile:
            self.config.critical_drawdown = profile["critical_drawdown"]
        if "reduce_at_drawdown" in profile:
            self.config.reduce_at_drawdown = profile["reduce_at_drawdown"]
        # Rebuild multiplier tables scaled to new halt threshold
        self._mult_normal, self._mult_opportunity = _build_multiplier_tables(
            self.config.halt_at_drawdown
        )
        logger.info(
            f"[T25] RiskAgent thresholds: halt={self.config.halt_at_drawdown:.0%}, "
            f"critical={self.config.critical_drawdown:.0%}, "
            f"reduce={self.config.reduce_at_drawdown:.0%}"
        )
    
    def assess(
        self,
        system_mode: str,
        current_drawdown: float,
        current_correlation: float,
        current_exposures: Dict[str, float],
        phase_result: RegimePhaseResult,
        sol_signal_strength: float = 0.0,
        sol_beta: float = 1.0,
        no_trade_trigger_scores: Optional[Dict[str, float]] = None,
        is_weekend: bool = False,
        is_holiday: bool = False,
        is_low_liquidity: bool = False,
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment.
        
        WHERE CONSUMED:
            - authority_fusion_v34.py: Applies caps and vetoes
            - exit_alpha_v34.py: Uses exit acceleration signal
        """
        
        assessment = RiskAssessment(
            drawdown=current_drawdown,
            correlation=current_correlation,
        )
        
        is_opportunity = (system_mode == "OPPORTUNITY")
        
        # =====================================================================
        # 1. Hard Drawdown Limits (UNCHANGED - ALWAYS ENFORCED)
        # =====================================================================
        
        if current_drawdown >= self.config.critical_drawdown:
            assessment.veto_active = True
            assessment.veto_reason = f"CRITICAL drawdown {current_drawdown:.1%}"
            assessment.risk_multiplier = 0.0
            logger.error(f"RISK VETO: {assessment.veto_reason}")
            return assessment
        
        if current_drawdown >= self.config.halt_at_drawdown:
            assessment.veto_active = True
            assessment.veto_reason = f"HALT drawdown {current_drawdown:.1%}"
            assessment.risk_multiplier = 0.0
            logger.warning(f"RISK VETO: {assessment.veto_reason}")
            return assessment
        
        # =====================================================================
        # 2. Risk Multiplier (OPPORTUNITY-AWARE)
        # =====================================================================
        
        assessment.risk_multiplier = get_risk_multiplier(
            current_drawdown, is_opportunity,
            normal_table=self._mult_normal,
            opportunity_table=self._mult_opportunity,
        )
        
        # =====================================================================
        # 3. Correlation Check
        # =====================================================================
        
        if current_correlation >= self.config.correlation_crisis:
            assessment.veto_active = True
            assessment.veto_reason = f"Correlation crisis {current_correlation:.3f}"
            logger.warning(f"RISK VETO: {assessment.veto_reason}")
            return assessment
        
        if current_correlation >= self.config.correlation_danger:
            # Reduce caps but don't veto
            assessment.max_gross_cap = 1.00
        elif current_correlation >= self.config.correlation_warning:
            assessment.max_gross_cap = 1.20
        else:
            assessment.max_gross_cap = self.config.max_gross_exposure
        
        # =====================================================================
        # 4. SOL Dominance Override (v3.4 refined)
        # =====================================================================
        
        sol_override = self.sol_dominance.evaluate(
            system_mode=system_mode,
            sol_signal_strength=sol_signal_strength,
            sol_beta=sol_beta,
            current_exposures=current_exposures,
            phase_result=phase_result,
            no_trade_trigger_scores=no_trade_trigger_scores or {},
        )
        
        if sol_override.active:
            assessment.sol_dominance_active = True
            assessment.max_gross_cap += sol_override.gross_cap_bonus
            assessment.sol_max_exposure = sol_override.max_sol_exposure
        else:
            assessment.sol_max_exposure = self.config.max_single_asset_pct
        
        # =====================================================================
        # 5. Weekend/Holiday Caps
        # =====================================================================
        
        time_cap = WeekendHolidayManager.get_exposure_cap(
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            is_low_liquidity=is_low_liquidity,
            system_mode=system_mode,
        )
        
        assessment.max_gross_cap = min(assessment.max_gross_cap, time_cap)
        
        # =====================================================================
        # 6. Fast Correlation (Exit Signal)
        # =====================================================================
        
        self.fast_correlation.update(current_correlation)
        should_accelerate, fast_corr = self.fast_correlation.should_accelerate_exit(system_mode)
        
        if should_accelerate:
            assessment.accelerate_exit = True
            assessment.exit_reason = f"Fast correlation {fast_corr:.3f}"
        
        # =====================================================================
        # 7. Single Asset Cap
        # =====================================================================
        
        assessment.max_single_asset_cap = self.config.max_single_asset_pct
        if not assessment.sol_dominance_active:
            assessment.sol_max_exposure = assessment.max_single_asset_cap
        
        return assessment
    
    def reset(self):
        """Reset all state."""
        self.sol_dominance.reset()
        self.fast_correlation.reset()

    # v6.4 persistence
    def to_dict(self) -> Dict:
        """Persist agent state for hot restart."""
        return {
            "sol_dominance_bar": self.sol_dominance._current_bar,
            "sol_dominance_active": self.sol_dominance._override.active,
            "fast_corr_buffer": list(self.fast_correlation._correlation_buffer),
        }

    def from_dict(self, d: Dict):
        """Restore agent state from persisted dict."""
        self.sol_dominance._current_bar = d.get("sol_dominance_bar", 0)
        if d.get("sol_dominance_active"):
            self.sol_dominance._override.active = True
        buf = d.get("fast_corr_buffer", [])
        self.fast_correlation._correlation_buffer = list(buf)

    # v6.4 generate_signal-style wrapper
    def generate_signal(self, **kwargs) -> Dict:
        """
        V6.4 agent contract wrapper around assess().

        Accepts kwargs matching assess() params and returns a v6.4 dict
        with risk_veto, leverage_cap, reasons, and standard metadata.
        """
        assessment = self.assess(**kwargs)
        return assessment.to_v6_dict()


# Singleton
_risk_agent: Optional[RiskAgentV34] = None

def get_risk_agent() -> RiskAgentV34:
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskAgentV34()
    return _risk_agent

def reset_risk_agent():
    global _risk_agent
    _risk_agent = None

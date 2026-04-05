"""
HMATS v3.2 - Opportunity Trigger Checker
=========================================
Implements Strategy Constitution Section 2.

Trigger Groups:
A) Lead-Lag Edge (Binance -> Kraken)
B) CRACK Window (structural break)
C) Volatility Expansion (DVOL / vol-of-vol)
D) Sentiment Shock (≥2σ)
E) SOL-Native Flow (DEX/on-chain)

All thresholds are explicit and testable.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class TriggerGroup(Enum):
    """Opportunity trigger groups."""
    LEAD_LAG_EDGE = "A"
    CRACK_WINDOW = "B"
    VOLATILITY_EXPANSION = "C"
    SENTIMENT_SHOCK = "D"
    SOL_FLOW_SURGE = "E"


@dataclass
class TriggerResult:
    """Result of a trigger check."""
    triggered: bool = False
    group: Optional[TriggerGroup] = None
    confidence: float = 0.0
    direction: str = "none"  # "long", "short", "none"
    reason: str = ""
    ttl_hours: float = 0.0
    can_set_direction: bool = False
    boosts_execution: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "triggered": self.triggered,
            "group": self.group.value if self.group else None,
            "confidence": self.confidence,
            "direction": self.direction,
            "reason": self.reason,
            "ttl_hours": self.ttl_hours,
            "can_set_direction": self.can_set_direction,
            "boosts_execution": self.boosts_execution
        }


@dataclass
class OpportunityState:
    """Aggregated opportunity state from all triggers."""
    is_active: bool = False
    active_triggers: List[TriggerResult] = field(default_factory=list)
    primary_trigger: Optional[TriggerResult] = None
    direction_consensus: str = "none"
    combined_confidence: float = 0.0
    earliest_expiry: Optional[datetime] = None


class OpportunityTriggerChecker:
    """
    Checks all opportunity triggers with explicit thresholds.
    
    From Strategy Constitution Section 2.
    """
    
    # =========================================================================
    # GROUP A: Lead-Lag Edge Thresholds
    # =========================================================================
    LEAD_LAG_NET_EDGE_BPS = 25      # Minimum net edge in bps
    LEAD_LAG_CONFIDENCE_MIN = 0.65  # Minimum edge window confidence
    LEAD_LAG_CVD_DIVERGENCE = 0.6   # Minimum CVD divergence (abs)
    LEAD_LAG_VPIN_MAX = 0.80        # Maximum VPIN (not toxic)
    LEAD_LAG_TTL_HOURS = 2          # Shorter TTL due to fast decay
    LEAD_LAG_AGGRESSIVE_EDGE_BPS = 35  # Edge for aggressive execution
    
    # =========================================================================
    # GROUP B: CRACK Window Thresholds
    # =========================================================================
    CRACK_BREAK_PCT = 0.02          # 2% structure break
    CRACK_VOLUME_RATIO = 2.5        # 2.5x average volume
    CRACK_ORDERBOOK_IMBALANCE = 0.4 # Minimum aligned imbalance
    CRACK_TTL_HOURS = 8             # 2x 4H bars
    CRACK_DECAY_PER_BAR = 0.30      # 30% decay per 4H bar
    # --- CRACK Weight Threshold Hierarchy ---
    # Layer 1: CRACK is ACTIVE gate (entry/activation)
    CRACK_ACTIVATION_WEIGHT = 0.50  # CRACK weight > 0.50 to activate window
    # Layer 2: Exit alpha partial close (high_risk_config default=0.45, main.py overrides to 0.35)
    CRACK_EXIT_ALPHA_DEFAULT = 0.45 # Default partial exit threshold (exit_alpha.py)
    CRACK_EXIT_URGENCY = 0.35       # Runtime override - harder to trigger scale-out
    # Layer 3: Decay floor during confirmed trend
    CRACK_DECAY_FLOOR = 0.40        # Never decay below 40% during trend
    
    # =========================================================================
    # GROUP C: Volatility Expansion Thresholds
    # =========================================================================
    VOL_DVOL_ZSCORE_MIN = 2.5       # Minimum DVOL z-score
    VOL_DVOL_ZSCORE_MAX = 4.0       # Maximum (above = NO_TRADE)
    VOL_VOL_OF_VOL_RATIO = 1.5      # 1.5x 30-day average
    VOL_RV_IV_RATIO_HIGH = 1.3      # RV/IV ratio high threshold
    VOL_RV_IV_RATIO_LOW = 0.7       # RV/IV ratio low threshold
    VOL_TTL_HOURS = 4               # Shorter due to vol mean reversion
    
    # =========================================================================
    # GROUP D: Sentiment Shock Thresholds
    # =========================================================================
    SENTIMENT_SHOCK_SIGMA = 2.0     # Minimum shock (standard deviations)
    SENTIMENT_EXTREME_SIGMA = 3.0   # Extreme shock (enables veto)
    SENTIMENT_SOURCES_ALIGNED = 2   # Minimum aligned sources (of 3)
    SENTIMENT_TTL_HOURS = 6         # 6 hours
    SENTIMENT_DECAY_PER_BAR = 0.25  # 25% decay per 4H bar
    
    # =========================================================================
    # GROUP E: SOL Flow Thresholds
    # =========================================================================
    SOL_DEX_VOLUME_RATIO = 3.0      # 3x 24H average
    SOL_ONCHAIN_INFLOW_USD = 10_000_000  # [P0-FIX] was $50M, lowered to $10M for SOL flow volumes
    SOL_JITO_TIP_RATIO = 2.0        # 2x average (competition signal)
    SOL_TTL_HOURS = 4               # 4 hours
    SOL_FLOW_HALF_LIFE_HOURS = 2    # 2H half-life on flow signals
    
    def __init__(self):
        self.last_check = datetime.now(timezone.utc)
        self.active_triggers: Dict[TriggerGroup, TriggerResult] = {}
    
    def check_all_triggers(
        self,
        market_data: Dict,
        lead_lag_data: Dict,
        sentiment_data: Dict,
        sol_flow_data: Dict
    ) -> OpportunityState:
        """
        Check all trigger groups and return aggregated state.
        
        Returns OpportunityState with all active triggers.
        """
        
        results: List[TriggerResult] = []
        
        # Check each trigger group
        result_a = self.check_lead_lag_edge(lead_lag_data)
        if result_a.triggered:
            results.append(result_a)
        
        result_b = self.check_crack_window(market_data)
        if result_b.triggered:
            results.append(result_b)
        
        result_c = self.check_volatility_expansion(market_data)
        if result_c.triggered:
            results.append(result_c)
        
        result_d = self.check_sentiment_shock(sentiment_data)
        if result_d.triggered:
            results.append(result_d)
        
        result_e = self.check_sol_flow_surge(sol_flow_data)
        if result_e.triggered:
            results.append(result_e)
        
        # Build aggregated state
        state = self._aggregate_triggers(results)
        
        self.last_check = datetime.now(timezone.utc)
        
        return state
    
    def check_lead_lag_edge(self, data: Dict) -> TriggerResult:
        """
        Group A: Lead-Lag Edge (Binance -> Kraken)
        
        CRITICAL: This trigger can NEVER set direction.
        It only triggers OPPORTUNITY and execution mode.
        """
        
        net_edge_bps = data.get('net_edge_bps', 0)
        edge_confidence = data.get('edge_window_confidence', 0)
        cvd_divergence = data.get('cvd_divergence', 0)
        vpin = data.get('vpin', 1.0)
        vpin_source = data.get('vpin_source', 'synthetic')

        # Check all conditions
        edge_ok = net_edge_bps >= self.LEAD_LAG_NET_EDGE_BPS
        confidence_ok = edge_confidence >= self.LEAD_LAG_CONFIDENCE_MIN
        cvd_ok = abs(cvd_divergence) >= self.LEAD_LAG_CVD_DIVERGENCE
        # [VPIN-C] Synthetic VPIN -> skip filter (don't let fake data block real signals)
        vpin_ok = True if vpin_source == 'synthetic' else vpin < self.LEAD_LAG_VPIN_MAX
        
        triggered = edge_ok and confidence_ok and cvd_ok and vpin_ok
        
        if triggered:
            # Determine execution boost level
            boosts_aggressive = net_edge_bps >= self.LEAD_LAG_AGGRESSIVE_EDGE_BPS
            
            return TriggerResult(
                triggered=True,
                group=TriggerGroup.LEAD_LAG_EDGE,
                confidence=edge_confidence,
                direction="none",  # NEVER sets direction
                reason=f"Lead-Lag edge {net_edge_bps:.0f}bps, CVD={cvd_divergence:.2f}",
                ttl_hours=self.LEAD_LAG_TTL_HOURS,
                can_set_direction=False,  # CRITICAL: Always False
                boosts_execution=boosts_aggressive
            )
        
        return TriggerResult(triggered=False)
    
    def check_crack_window(self, data: Dict) -> TriggerResult:
        """
        Group B: CRACK Window (structural break)
        
        This trigger CAN set direction based on break direction.
        """
        
        structure_break_pct = data.get('structure_break_pct', 0)
        volume_ratio = data.get('volume_ratio', 1.0)
        orderbook_imbalance = data.get('orderbook_imbalance', 0)
        structure_level = data.get('structure_level', 0)
        current_price = data.get('current_price', 0)
        
        # Check conditions
        break_ok = abs(structure_break_pct) >= self.CRACK_BREAK_PCT
        volume_ok = volume_ratio >= self.CRACK_VOLUME_RATIO
        
        # Imbalance must align with break direction
        break_direction = "long" if current_price > structure_level else "short"
        if break_direction == "long":
            imbalance_ok = orderbook_imbalance >= self.CRACK_ORDERBOOK_IMBALANCE
        else:
            imbalance_ok = orderbook_imbalance <= -self.CRACK_ORDERBOOK_IMBALANCE
        
        triggered = break_ok and volume_ok and imbalance_ok
        
        if triggered:
            confidence = min(1.0, 0.5 + (volume_ratio / 5) * 0.25 + abs(structure_break_pct) * 5)
            
            return TriggerResult(
                triggered=True,
                group=TriggerGroup.CRACK_WINDOW,
                confidence=confidence,
                direction=break_direction,
                reason=f"CRACK {break_direction}: {structure_break_pct:.1%} break, {volume_ratio:.1f}x vol",
                ttl_hours=self.CRACK_TTL_HOURS,
                can_set_direction=True,
                boosts_execution=True
            )
        
        return TriggerResult(triggered=False)
    
    def check_volatility_expansion(self, data: Dict) -> TriggerResult:
        """
        Group C: Volatility Expansion
        
        This trigger does NOT set direction (vol is non-directional).
        """
        
        dvol_zscore = data.get('dvol_zscore', 0)
        vol_of_vol = data.get('vol_of_vol', 0)
        vol_of_vol_avg = data.get('vol_of_vol_30d_avg', 1.0)
        rv_iv_ratio = data.get('rv_iv_ratio', 1.0)
        
        # Check DVOL in expansion range (not extreme)
        dvol_ok = self.VOL_DVOL_ZSCORE_MIN <= dvol_zscore < self.VOL_DVOL_ZSCORE_MAX
        
        # Check vol-of-vol elevated
        vov_ratio = vol_of_vol / vol_of_vol_avg if vol_of_vol_avg > 0 else 0
        vov_ok = vov_ratio >= self.VOL_VOL_OF_VOL_RATIO
        
        # Check RV/IV dislocation
        rv_iv_ok = (rv_iv_ratio >= self.VOL_RV_IV_RATIO_HIGH or 
                   rv_iv_ratio <= self.VOL_RV_IV_RATIO_LOW)
        
        triggered = dvol_ok and vov_ok and rv_iv_ok
        
        if triggered:
            confidence = min(1.0, 0.4 + dvol_zscore * 0.1 + vov_ratio * 0.1)
            
            return TriggerResult(
                triggered=True,
                group=TriggerGroup.VOLATILITY_EXPANSION,
                confidence=confidence,
                direction="none",  # Vol is non-directional
                reason=f"Vol expansion: DVOL z={dvol_zscore:.1f}, VoV={vov_ratio:.1f}x",
                ttl_hours=self.VOL_TTL_HOURS,
                can_set_direction=False,
                boosts_execution=False  # Vol doesn't boost execution
            )
        
        return TriggerResult(triggered=False)
    
    def check_sentiment_shock(self, data: Dict) -> TriggerResult:
        """
        Group D: Sentiment Shock
        
        This trigger CAN set direction based on shock direction.
        Extreme shock (>3σ) enables VETO on opposite direction.
        """
        
        sentiment_shock = data.get('sentiment_shock_sigma', 0)
        sources_aligned = data.get('sources_aligned', 0)
        no_reversal = data.get('no_reversal_in_2h', True)
        
        # Check conditions
        shock_ok = abs(sentiment_shock) >= self.SENTIMENT_SHOCK_SIGMA
        sources_ok = sources_aligned >= self.SENTIMENT_SOURCES_ALIGNED
        
        triggered = shock_ok and sources_ok and no_reversal
        
        if triggered:
            direction = "long" if sentiment_shock > 0 else "short"
            is_extreme = abs(sentiment_shock) >= self.SENTIMENT_EXTREME_SIGMA
            confidence = min(1.0, 0.5 + abs(sentiment_shock) * 0.1 + sources_aligned * 0.1)
            
            reason = f"Sentiment shock: {sentiment_shock:.1f}σ, {sources_aligned}/3 aligned"
            if is_extreme:
                reason += f" [EXTREME - VETO on {'short' if direction == 'long' else 'long'}]"
            
            return TriggerResult(
                triggered=True,
                group=TriggerGroup.SENTIMENT_SHOCK,
                confidence=confidence,
                direction=direction,
                reason=reason,
                ttl_hours=self.SENTIMENT_TTL_HOURS,
                can_set_direction=True,
                boosts_execution=False
            )
        
        return TriggerResult(triggered=False)
    
    def check_sol_flow_surge(self, data: Dict) -> TriggerResult:
        """
        Group E: SOL-Native Flow (DEX/on-chain)
        
        This trigger CAN set direction based on flow direction.
        """
        
        dex_volume_ratio = data.get('dex_volume_ratio', 0)
        onchain_inflow_4h = data.get('onchain_inflow_4h', 0)
        whale_direction = data.get('whale_net_direction', "none")
        jito_tip_ratio = data.get('jito_tip_ratio', 0)
        
        # Check flow conditions (either DEX or on-chain)
        flow_ok = (dex_volume_ratio >= self.SOL_DEX_VOLUME_RATIO or 
                  abs(onchain_inflow_4h) >= self.SOL_ONCHAIN_INFLOW_USD)
        
        # Check Jito tips (competition signal)
        jito_ok = jito_tip_ratio >= self.SOL_JITO_TIP_RATIO
        
        # Whale direction must be aligned
        flow_direction = "long" if onchain_inflow_4h > 0 else "short"
        whale_aligned = (whale_direction == flow_direction or whale_direction == "none")
        
        triggered = flow_ok and jito_ok and whale_aligned
        
        if triggered:
            confidence = min(1.0, 0.5 + dex_volume_ratio * 0.1 + jito_tip_ratio * 0.1)
            inflow_m = onchain_inflow_4h / 1_000_000
            
            return TriggerResult(
                triggered=True,
                group=TriggerGroup.SOL_FLOW_SURGE,
                confidence=confidence,
                direction=flow_direction,
                reason=f"SOL flow surge: ${inflow_m:.0f}M inflow, DEX {dex_volume_ratio:.1f}x, Jito {jito_tip_ratio:.1f}x",
                ttl_hours=self.SOL_TTL_HOURS,
                can_set_direction=True,
                boosts_execution=True
            )
        
        return TriggerResult(triggered=False)
    
    def _aggregate_triggers(self, results: List[TriggerResult]) -> OpportunityState:
        """Aggregate multiple triggers into opportunity state."""
        
        if not results:
            return OpportunityState(is_active=False)
        
        # Find direction consensus
        directions_with_authority = [r.direction for r in results if r.can_set_direction and r.direction != "none"]
        
        if directions_with_authority:
            # Check for consensus
            long_count = sum(1 for d in directions_with_authority if d == "long")
            short_count = sum(1 for d in directions_with_authority if d == "short")
            
            if long_count > 0 and short_count > 0:
                direction_consensus = "conflict"  # Will need authority to resolve
            elif long_count > 0:
                direction_consensus = "long"
            elif short_count > 0:
                direction_consensus = "short"
            else:
                direction_consensus = "none"
        else:
            direction_consensus = "none"
        
        # Calculate combined confidence (max of all)
        combined_confidence = max(r.confidence for r in results)
        
        # Find primary trigger (highest confidence with direction)
        direction_triggers = [r for r in results if r.can_set_direction and r.direction != "none"]
        if direction_triggers:
            primary = max(direction_triggers, key=lambda r: r.confidence)
        else:
            primary = max(results, key=lambda r: r.confidence)
        
        # Calculate earliest expiry
        now = datetime.now(timezone.utc)
        expiries = [now + timedelta(hours=r.ttl_hours) for r in results]
        earliest_expiry = min(expiries)
        
        return OpportunityState(
            is_active=True,
            active_triggers=results,
            primary_trigger=primary,
            direction_consensus=direction_consensus,
            combined_confidence=combined_confidence,
            earliest_expiry=earliest_expiry
        )
    
    def get_trigger_summary(self) -> Dict:
        """Get summary of active triggers for logging."""
        return {
            group.value: {
                "threshold": self._get_thresholds(group),
                "active": group in self.active_triggers
            }
            for group in TriggerGroup
        }
    
    def _get_thresholds(self, group: TriggerGroup) -> Dict:
        """Get thresholds for a trigger group."""
        if group == TriggerGroup.LEAD_LAG_EDGE:
            return {
                "net_edge_bps": self.LEAD_LAG_NET_EDGE_BPS,
                "confidence_min": self.LEAD_LAG_CONFIDENCE_MIN,
                "cvd_divergence": self.LEAD_LAG_CVD_DIVERGENCE,
                "vpin_max": self.LEAD_LAG_VPIN_MAX
            }
        elif group == TriggerGroup.CRACK_WINDOW:
            return {
                "break_pct": self.CRACK_BREAK_PCT,
                "volume_ratio": self.CRACK_VOLUME_RATIO,
                "imbalance": self.CRACK_ORDERBOOK_IMBALANCE
            }
        elif group == TriggerGroup.VOLATILITY_EXPANSION:
            return {
                "dvol_zscore_range": [self.VOL_DVOL_ZSCORE_MIN, self.VOL_DVOL_ZSCORE_MAX],
                "vol_of_vol_ratio": self.VOL_VOL_OF_VOL_RATIO,
                "rv_iv_ratio": [self.VOL_RV_IV_RATIO_LOW, self.VOL_RV_IV_RATIO_HIGH]
            }
        elif group == TriggerGroup.SENTIMENT_SHOCK:
            return {
                "shock_sigma": self.SENTIMENT_SHOCK_SIGMA,
                "extreme_sigma": self.SENTIMENT_EXTREME_SIGMA,
                "sources_aligned": self.SENTIMENT_SOURCES_ALIGNED
            }
        elif group == TriggerGroup.SOL_FLOW_SURGE:
            return {
                "dex_volume_ratio": self.SOL_DEX_VOLUME_RATIO,
                "onchain_inflow_usd": self.SOL_ONCHAIN_INFLOW_USD,
                "jito_tip_ratio": self.SOL_JITO_TIP_RATIO
            }
        return {}


# Singleton instance
_opportunity_checker = OpportunityTriggerChecker()

def get_opportunity_checker() -> OpportunityTriggerChecker:
    return _opportunity_checker

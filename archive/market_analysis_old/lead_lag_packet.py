"""
HMATS v3.2 - Lead-Lag Decision Packet
======================================
Purpose: Encapsulate lead-lag signals with STRICTLY LIMITED powers.

Philosophy:
- Lead-Lag operates at 200ms half-life (execution frequency)
- It CANNOT directly modify target exposure
- It CAN ONLY:
  1. Trigger OpportunityMode
  2. Set execution_mode / cancel passives
  3. Provide flow evidence to Authority layer

The packet is ADVISORY to execution, not DIRECTIVE to strategy.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """
    Execution mode advised by lead-lag.
    
    This controls HOW we execute, not WHAT we execute.
    """
    PASSIVE_ONLY = auto()       # Only maker orders, no urgency
    PASSIVE_PREFERRED = auto()  # Prefer maker, take if needed
    AGGRESSIVE_TAKER = auto()   # Take immediately, opportunity window


class FlowDirection(Enum):
    """Observed flow direction from CVD/VPIN analysis."""
    STRONG_BUY = auto()
    MODERATE_BUY = auto()
    NEUTRAL = auto()
    MODERATE_SELL = auto()
    STRONG_SELL = auto()


@dataclass
class LeadLagDecisionPacket:
    """
    Lead-Lag advisory packet.
    
    CRITICAL CONSTRAINT: This packet CANNOT modify target_exposure.
    It can only influence execution mechanics and trigger opportunity mode.
    
    Created at ~200ms frequency, expires quickly.
    """
    
    # Identity & expiry
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at_ms: float = 200.0  # Half-life in ms
    packet_id: str = ""
    
    # Edge calculation (Binance -> Kraken)
    edge_bps: float = 0.0              # Observed price edge in bps
    friction_bps: float = 33.0         # Expected execution friction
    net_edge_bps: float = 0.0          # edge - friction
    edge_window_confidence: float = 0.0 # Confidence this edge is real
    
    # Execution advice (NOT strategy advice)
    execution_mode: ExecutionMode = ExecutionMode.PASSIVE_PREFERRED
    cancel_passive_orders_now: bool = False
    urgency_score: float = 0.0  # 0-1, how urgent to execute
    
    # Flow metrics (evidence, not directive)
    cvd_divergence_score: float = 0.0   # -1 to 1
    vpin_toxicity: float = 0.0          # 0 to 1
    flow_direction: FlowDirection = FlowDirection.NEUTRAL
    flow_alignment: float = 0.0          # -1 to 1, how aligned is flow
    
    # Spread & liquidity
    spread_bps: float = 0.0
    orderbook_imbalance: float = 0.0  # -1 to 1
    
    # Trigger recommendations
    should_trigger_opportunity: bool = False
    opportunity_direction: str = "none"  # "long", "short", "none"
    opportunity_reason: str = ""
    
    def is_expired(self) -> bool:
        """Check if packet has expired."""
        age_ms = (datetime.utcnow() - self.created_at).total_seconds() * 1000
        return age_ms > self.expires_at_ms
    
    def remaining_life_pct(self) -> float:
        """Get remaining life as percentage."""
        age_ms = (datetime.utcnow() - self.created_at).total_seconds() * 1000
        return max(0, 1.0 - (age_ms / self.expires_at_ms))
    
    def has_actionable_edge(self) -> bool:
        """Check if there's actionable edge after friction."""
        return (
            self.net_edge_bps > 10 and  # At least 10bps net
            self.edge_window_confidence > 0.6 and
            not self.is_expired()
        )
    
    def to_dict(self) -> Dict:
        return {
            "created_at": self.created_at.isoformat(),
            "expires_at_ms": self.expires_at_ms,
            "remaining_life_pct": self.remaining_life_pct(),
            "edge_bps": self.edge_bps,
            "friction_bps": self.friction_bps,
            "net_edge_bps": self.net_edge_bps,
            "edge_window_confidence": self.edge_window_confidence,
            "execution_mode": self.execution_mode.name,
            "cancel_passive_orders_now": self.cancel_passive_orders_now,
            "urgency_score": self.urgency_score,
            "cvd_divergence_score": self.cvd_divergence_score,
            "vpin_toxicity": self.vpin_toxicity,
            "flow_direction": self.flow_direction.name,
            "flow_alignment": self.flow_alignment,
            "should_trigger_opportunity": self.should_trigger_opportunity,
            "opportunity_direction": self.opportunity_direction
        }


class LeadLagPacketFactory:
    """
    Factory for creating lead-lag packets from raw signals.
    
    Enforces the constraint that packets are ADVISORY only.
    """
    
    # Friction constants
    KRAKEN_TAKER_FEE_BPS = 26.0
    ESTIMATED_SLIPPAGE_BPS = 5.0
    LATENCY_COST_BPS = 2.0
    
    # Thresholds
    OPPORTUNITY_TRIGGER_CVD = 0.7   # CVD divergence to trigger opportunity
    OPPORTUNITY_TRIGGER_VPIN = 0.8  # VPIN to trigger opportunity
    TOXIC_FLOW_THRESHOLD = 0.85     # VPIN threshold for toxic flow
    
    @classmethod
    def create(
        cls,
        binance_price: float,
        kraken_price: float,
        binance_volume: float,
        kraken_volume: float,
        cvd_delta: float,
        vpin: float,
        orderbook_imbalance: float,
        latency_ms: float
    ) -> LeadLagDecisionPacket:
        """
        Create a lead-lag packet from raw market data.
        
        CRITICAL: This creates an ADVISORY packet only.
        It CANNOT and SHOULD NOT modify strategy target exposure.
        """
        
        # Calculate edge
        if kraken_price > 0:
            edge_bps = ((binance_price - kraken_price) / kraken_price) * 10000
        else:
            edge_bps = 0.0
        
        # Calculate friction
        friction_bps = (
            cls.KRAKEN_TAKER_FEE_BPS +
            cls.ESTIMATED_SLIPPAGE_BPS +
            cls.LATENCY_COST_BPS
        )
        
        net_edge_bps = edge_bps - friction_bps
        
        # Calculate confidence based on volume and latency
        volume_ratio = binance_volume / max(kraken_volume, 1)
        latency_decay = max(0, 1 - latency_ms / 200)  # Decay over 200ms
        edge_confidence = min(1.0, volume_ratio * 0.3 + latency_decay * 0.7)
        
        # Determine flow direction
        flow_direction = cls._calculate_flow_direction(cvd_delta, vpin)
        flow_alignment = cvd_delta  # -1 to 1
        
        # Determine execution mode
        execution_mode, cancel_passives = cls._determine_execution_mode(
            net_edge_bps, vpin, edge_confidence
        )
        
        # Calculate urgency
        urgency = cls._calculate_urgency(net_edge_bps, vpin, latency_decay)
        
        # Check for opportunity trigger
        should_trigger, opp_direction, opp_reason = cls._check_opportunity_trigger(
            cvd_delta, vpin, net_edge_bps, edge_confidence
        )
        
        # Calculate half-life based on latency
        half_life = max(100, 200 - latency_ms)  # Faster expiry with higher latency
        
        return LeadLagDecisionPacket(
            created_at=datetime.utcnow(),
            expires_at_ms=half_life,
            packet_id=f"ll_{datetime.utcnow().strftime('%H%M%S%f')[:12]}",
            edge_bps=edge_bps,
            friction_bps=friction_bps,
            net_edge_bps=net_edge_bps,
            edge_window_confidence=edge_confidence,
            execution_mode=execution_mode,
            cancel_passive_orders_now=cancel_passives,
            urgency_score=urgency,
            cvd_divergence_score=cvd_delta,
            vpin_toxicity=vpin,
            flow_direction=flow_direction,
            flow_alignment=flow_alignment,
            spread_bps=abs(edge_bps) * 0.3,  # Rough estimate
            orderbook_imbalance=orderbook_imbalance,
            should_trigger_opportunity=should_trigger,
            opportunity_direction=opp_direction,
            opportunity_reason=opp_reason
        )
    
    @classmethod
    def _calculate_flow_direction(cls, cvd_delta: float, vpin: float) -> FlowDirection:
        """Determine flow direction from CVD and VPIN."""
        
        if cvd_delta > 0.5:
            return FlowDirection.STRONG_BUY
        elif cvd_delta > 0.2:
            return FlowDirection.MODERATE_BUY
        elif cvd_delta < -0.5:
            return FlowDirection.STRONG_SELL
        elif cvd_delta < -0.2:
            return FlowDirection.MODERATE_SELL
        else:
            return FlowDirection.NEUTRAL
    
    @classmethod
    def _determine_execution_mode(
        cls,
        net_edge_bps: float,
        vpin: float,
        confidence: float
    ) -> tuple:
        """
        Determine execution mode and cancel flag.
        
        Returns: (ExecutionMode, cancel_passives)
        """
        
        # High VPIN = toxic flow, go passive
        if vpin >= cls.TOXIC_FLOW_THRESHOLD:
            return ExecutionMode.PASSIVE_ONLY, True
        
        # Strong edge with confidence = aggressive
        if net_edge_bps > 20 and confidence > 0.7:
            return ExecutionMode.AGGRESSIVE_TAKER, True
        
        # Moderate edge = passive preferred
        if net_edge_bps > 10:
            return ExecutionMode.PASSIVE_PREFERRED, False
        
        # Default = passive only
        return ExecutionMode.PASSIVE_ONLY, False
    
    @classmethod
    def _calculate_urgency(
        cls,
        net_edge_bps: float,
        vpin: float,
        latency_decay: float
    ) -> float:
        """Calculate execution urgency score (0-1)."""
        
        edge_urgency = min(1.0, max(0, net_edge_bps) / 50)
        flow_urgency = vpin if vpin < cls.TOXIC_FLOW_THRESHOLD else 0
        
        return min(1.0, edge_urgency * 0.5 + flow_urgency * 0.3 + latency_decay * 0.2)
    
    @classmethod
    def _check_opportunity_trigger(
        cls,
        cvd_delta: float,
        vpin: float,
        net_edge_bps: float,
        confidence: float
    ) -> tuple:
        """
        Check if this packet should trigger opportunity mode.
        
        Returns: (should_trigger, direction, reason)
        """
        
        # Strong CVD divergence
        if abs(cvd_delta) >= cls.OPPORTUNITY_TRIGGER_CVD and confidence > 0.6:
            direction = "long" if cvd_delta > 0 else "short"
            return True, direction, f"CVD divergence: {cvd_delta:.2f}"
        
        # Extreme VPIN with edge
        if vpin >= cls.OPPORTUNITY_TRIGGER_VPIN and net_edge_bps > 15:
            return True, "short", f"VPIN extreme with edge: {vpin:.2f}"
        
        return False, "none", ""


class LeadLagIntegrator:
    """
    Integrates lead-lag packets into the system.
    
    CRITICAL: This integrator respects the 3-power limit:
    1. Trigger opportunity mode
    2. Set execution mode / cancel passives
    3. Provide flow evidence
    
    It NEVER modifies target exposure directly.
    """
    
    def __init__(self):
        self.current_packet: Optional[LeadLagDecisionPacket] = None
        self.packet_history: list = []
        self.max_history = 1000
    
    def process_packet(self, packet: LeadLagDecisionPacket) -> Dict:
        """
        Process a lead-lag packet and return integration results.
        
        Returns dict with:
        - execution_advice: How to execute current intent
        - opportunity_trigger: Whether to trigger opportunity mode
        - flow_evidence: Evidence for authority layer
        """
        
        self.current_packet = packet
        self.packet_history.append(packet)
        
        # Trim history
        if len(self.packet_history) > self.max_history:
            self.packet_history = self.packet_history[-self.max_history:]
        
        result = {
            # Power 1: Execution advice (mode + cancel)
            "execution_mode": packet.execution_mode.name,
            "cancel_passive_orders": packet.cancel_passive_orders_now,
            "urgency": packet.urgency_score,
            
            # Power 2: Opportunity trigger
            "trigger_opportunity": packet.should_trigger_opportunity,
            "opportunity_direction": packet.opportunity_direction,
            "opportunity_reason": packet.opportunity_reason,
            
            # Power 3: Flow evidence (for authority layer)
            "flow_evidence": {
                "cvd_divergence": packet.cvd_divergence_score,
                "vpin_toxicity": packet.vpin_toxicity,
                "flow_direction": packet.flow_direction.name,
                "flow_alignment": packet.flow_alignment,
                "orderbook_imbalance": packet.orderbook_imbalance,
                "net_edge_bps": packet.net_edge_bps,
                "confidence": packet.edge_window_confidence
            },
            
            # Meta
            "packet_id": packet.packet_id,
            "remaining_life_pct": packet.remaining_life_pct(),
            "has_actionable_edge": packet.has_actionable_edge()
        }
        
        if packet.should_trigger_opportunity:
            logger.info(f"Lead-Lag OPPORTUNITY trigger: {packet.opportunity_direction} | "
                       f"{packet.opportunity_reason}")
        
        return result
    
    def get_recent_flow_summary(self, lookback_packets: int = 50) -> Dict:
        """Get summary of recent flow for authority layer."""
        
        recent = self.packet_history[-lookback_packets:]
        
        if not recent:
            return {
                "avg_cvd": 0.0,
                "avg_vpin": 0.5,
                "dominant_direction": "neutral",
                "consistency": 0.0
            }
        
        cvd_values = [p.cvd_divergence_score for p in recent]
        vpin_values = [p.vpin_toxicity for p in recent]
        flow_values = [p.flow_alignment for p in recent]
        
        avg_cvd = sum(cvd_values) / len(cvd_values)
        avg_vpin = sum(vpin_values) / len(vpin_values)
        avg_flow = sum(flow_values) / len(flow_values)
        
        # Check consistency
        same_direction = sum(1 for f in flow_values if f * avg_flow > 0)
        consistency = same_direction / len(flow_values) if flow_values else 0
        
        if avg_flow > 0.3:
            direction = "bullish"
        elif avg_flow < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return {
            "avg_cvd": avg_cvd,
            "avg_vpin": avg_vpin,
            "avg_flow_alignment": avg_flow,
            "dominant_direction": direction,
            "consistency": consistency
        }


# Singleton integrator
_lead_lag_integrator = LeadLagIntegrator()

def get_lead_lag_integrator() -> LeadLagIntegrator:
    return _lead_lag_integrator

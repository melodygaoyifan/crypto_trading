"""
================================================================================
SOL DEFENSE MODULE - Solana-Specific Risk Management
================================================================================
Version: 3.2.0
Purpose: Comprehensive SOL defense for 50% portfolio weight (Part 6 compliance)

Features:
- Jito Tips tracking for MEV activity
- On-chain inflow monitoring (Kraken deposit detection)
- Network congestion tracking
- SOL-specific risk override
- DEX arbitrage detection
================================================================================
"""

import time
import logging
import asyncio
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import threading

logger = logging.getLogger("HMATS.SOLDefense")


class SOLRiskLevel(Enum):
    """SOL-specific risk levels."""
    LOW = auto()
    NORMAL = auto()
    ELEVATED = auto()
    HIGH = auto()
    CRITICAL = auto()


class NetworkStatus(Enum):
    """Solana network status."""
    HEALTHY = auto()
    DEGRADED = auto()
    CONGESTED = auto()
    HALTED = auto()


@dataclass
class JitoMetrics:
    """Jito MEV metrics."""
    avg_tip_lamports: int = 0
    tip_90th_percentile: int = 0
    bundle_success_rate: float = 0.0
    landed_tips_per_minute: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class NetworkMetrics:
    """Solana network metrics."""
    tps: float = 0.0                    # Transactions per second
    slot_time_ms: float = 400.0         # Current slot time
    priority_fee_estimate: int = 0      # Recommended priority fee (lamports)
    compute_units_consumed: float = 0.0  # CU usage %
    vote_transaction_pct: float = 0.0   # Vote tx percentage
    skip_rate: float = 0.0              # Leader skip rate
    status: NetworkStatus = NetworkStatus.HEALTHY


@dataclass
class OnChainFlow:
    """On-chain flow data."""
    exchange_inflow_sol: float = 0.0
    exchange_outflow_sol: float = 0.0
    net_flow_sol: float = 0.0
    large_transfers: int = 0  # Transfers > 1000 SOL
    whale_movements: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SOLDefenseConfig:
    """SOL defense configuration."""
    # Portfolio weight
    sol_target_weight: float = 0.50
    
    # Network thresholds
    tps_warning_threshold: float = 2000.0
    tps_critical_threshold: float = 1000.0
    slot_time_warning_ms: float = 500.0
    slot_time_critical_ms: float = 800.0
    skip_rate_warning: float = 0.05
    skip_rate_critical: float = 0.15
    
    # Jito thresholds
    jito_tip_spike_multiplier: float = 3.0  # Tips > 3x average = spike
    
    # On-chain thresholds
    large_inflow_threshold_sol: float = 50000.0  # 50k SOL
    whale_transfer_threshold_sol: float = 1000.0
    
    # Risk adjustments
    position_reduction_degraded: float = 0.75
    position_reduction_congested: float = 0.50
    position_reduction_critical: float = 0.0  # No new positions


class JitoTipsTracker:
    """
    Track Jito Tips for MEV activity detection.
    
    High Jito tips indicate:
    - High MEV activity
    - Potential front-running
    - Market manipulation
    """
    
    def __init__(self, config: SOLDefenseConfig):
        self.config = config
        
        self.tip_history: deque = deque(maxlen=1000)
        self.current_metrics = JitoMetrics()
        
        self._baseline_tip = 1000  # 1000 lamports baseline
        self._lock = threading.Lock()
    
    def add_tip(self, tip_lamports: int, bundle_landed: bool = True):
        """Record a Jito tip observation."""
        with self._lock:
            self.tip_history.append({
                'tip': tip_lamports,
                'landed': bundle_landed,
                'timestamp': time.time()
            })
            
            self._update_metrics()
    
    def _update_metrics(self):
        """Update tip metrics."""
        if len(self.tip_history) < 10:
            return
        
        import numpy as np
        
        tips = [t['tip'] for t in self.tip_history]
        landed = [t['landed'] for t in self.tip_history]
        
        self.current_metrics.avg_tip_lamports = int(np.mean(tips))
        self.current_metrics.tip_90th_percentile = int(np.percentile(tips, 90))
        self.current_metrics.bundle_success_rate = np.mean(landed)
        
        # Calculate tips per minute
        recent = [t for t in self.tip_history if time.time() - t['timestamp'] < 60]
        self.current_metrics.landed_tips_per_minute = len([t for t in recent if t['landed']])
        self.current_metrics.timestamp = time.time()
    
    def is_tip_spike(self) -> Tuple[bool, float]:
        """
        Check if current tips indicate a spike.
        
        Returns:
            (is_spike, spike_multiplier)
        """
        with self._lock:
            if self.current_metrics.avg_tip_lamports == 0:
                return False, 1.0
            
            current = self.current_metrics.tip_90th_percentile
            baseline = max(self._baseline_tip, self.current_metrics.avg_tip_lamports)
            
            multiplier = current / baseline
            is_spike = multiplier >= self.config.jito_tip_spike_multiplier
            
            return is_spike, multiplier
    
    def get_metrics(self) -> JitoMetrics:
        """Get current Jito metrics."""
        with self._lock:
            return self.current_metrics


class OnChainInflowMonitor:
    """
    Monitor SOL exchange inflows.
    
    Large inflows to Kraken often precede selling pressure.
    """
    
    # Kraken SOL deposit addresses - loaded from config or environment
    # In production: Set KRAKEN_SOL_DEPOSIT_ADDRESSES env var (comma-separated)
    # Or provide via SOLDefenseConfig.kraken_deposit_addresses
    DEFAULT_KRAKEN_ADDRESSES: List[str] = []  # Empty by default - must be configured
    
    def __init__(self, config: SOLDefenseConfig):
        self.config = config
        
        # Load Kraken deposit addresses from config or environment
        self.KRAKEN_DEPOSIT_ADDRESSES = self._load_deposit_addresses()
        
        self.flow_history: deque = deque(maxlen=100)
        self.current_flow = OnChainFlow()
        self.whale_alerts: List[Dict] = []
        
        self._lock = threading.Lock()
    
    def _load_deposit_addresses(self) -> List[str]:
        """Load Kraken deposit addresses from config or environment."""
        import os
        
        # Try config first
        if hasattr(self.config, 'kraken_deposit_addresses') and self.config.kraken_deposit_addresses:
            return self.config.kraken_deposit_addresses
        
        # Try environment variable
        env_addresses = os.getenv('KRAKEN_SOL_DEPOSIT_ADDRESSES', '')
        if env_addresses:
            return [addr.strip() for addr in env_addresses.split(',') if addr.strip()]
        
        # No addresses configured - inflow detection will rely on is_exchange_deposit flag
        logger.warning(
            "[OnChainInflowMonitor] No Kraken deposit addresses configured. "
            "Set KRAKEN_SOL_DEPOSIT_ADDRESSES env var or config.kraken_deposit_addresses"
        )
        return []
    
    def record_transfer(
        self,
        amount_sol: float,
        from_address: str,
        to_address: str,
        is_exchange_deposit: bool = False
    ):
        """Record a SOL transfer."""
        with self._lock:
            timestamp = time.time()
            
            # Update flow
            if is_exchange_deposit or to_address in self.KRAKEN_DEPOSIT_ADDRESSES:
                self.current_flow.exchange_inflow_sol += amount_sol
            elif from_address in self.KRAKEN_DEPOSIT_ADDRESSES:
                self.current_flow.exchange_outflow_sol += amount_sol
            
            self.current_flow.net_flow_sol = (
                self.current_flow.exchange_inflow_sol -
                self.current_flow.exchange_outflow_sol
            )
            
            # Track large transfers
            if amount_sol >= self.config.whale_transfer_threshold_sol:
                self.current_flow.large_transfers += 1
                whale_movement = {
                    'amount_sol': amount_sol,
                    'from': from_address[:8] + '...',
                    'to': to_address[:8] + '...',
                    'timestamp': timestamp,
                    'is_exchange_deposit': is_exchange_deposit
                }
                self.current_flow.whale_movements.append(whale_movement)
                
                # Alert on very large deposits
                if is_exchange_deposit and amount_sol >= self.config.large_inflow_threshold_sol:
                    self.whale_alerts.append(whale_movement)
                    logger.warning(f"Large SOL exchange deposit: {amount_sol:.0f} SOL")
            
            self.current_flow.timestamp = timestamp
    
    def get_flow(self) -> OnChainFlow:
        """Get current flow data."""
        with self._lock:
            return self.current_flow
    
    def get_inflow_pressure(self) -> Tuple[bool, float]:
        """
        Check if inflow pressure is elevated.
        
        Returns:
            (is_elevated, net_flow_sol)
        """
        with self._lock:
            net = self.current_flow.net_flow_sol
            is_elevated = net >= self.config.large_inflow_threshold_sol
            return is_elevated, net
    
    def reset_period(self):
        """Reset for new tracking period."""
        with self._lock:
            self.flow_history.append(self.current_flow)
            self.current_flow = OnChainFlow()


class NetworkCongestionMonitor:
    """
    Monitor Solana network congestion.
    
    Congestion affects execution quality and timing.
    """
    
    def __init__(self, config: SOLDefenseConfig):
        self.config = config
        
        self.current_metrics = NetworkMetrics()
        self.metrics_history: deque = deque(maxlen=100)
        
        self._lock = threading.Lock()
    
    def update(
        self,
        tps: float = None,
        slot_time_ms: float = None,
        priority_fee: int = None,
        skip_rate: float = None
    ):
        """Update network metrics."""
        with self._lock:
            if tps is not None:
                self.current_metrics.tps = tps
            if slot_time_ms is not None:
                self.current_metrics.slot_time_ms = slot_time_ms
            if priority_fee is not None:
                self.current_metrics.priority_fee_estimate = priority_fee
            if skip_rate is not None:
                self.current_metrics.skip_rate = skip_rate
            
            # Determine status
            self.current_metrics.status = self._calculate_status()
    
    def _calculate_status(self) -> NetworkStatus:
        """Calculate network status from metrics."""
        m = self.current_metrics
        c = self.config
        
        # Check for halt
        if m.tps < 100:
            return NetworkStatus.HALTED
        
        # Check critical thresholds
        if (m.tps < c.tps_critical_threshold or
            m.slot_time_ms > c.slot_time_critical_ms or
            m.skip_rate > c.skip_rate_critical):
            return NetworkStatus.CONGESTED
        
        # Check warning thresholds
        if (m.tps < c.tps_warning_threshold or
            m.slot_time_ms > c.slot_time_warning_ms or
            m.skip_rate > c.skip_rate_warning):
            return NetworkStatus.DEGRADED
        
        return NetworkStatus.HEALTHY
    
    def get_status(self) -> NetworkStatus:
        """Get current network status."""
        with self._lock:
            return self.current_metrics.status
    
    def get_metrics(self) -> NetworkMetrics:
        """Get current network metrics."""
        with self._lock:
            return self.current_metrics


class SOLDefenseModule:
    """
    Main SOL defense coordinator.
    
    Integrates:
    - Network monitoring
    - Jito tips tracking
    - On-chain inflow monitoring
    - Risk override decisions
    """
    
    def __init__(self, config: SOLDefenseConfig = None):
        self.config = config or SOLDefenseConfig()
        
        # Sub-modules
        self.network_monitor = NetworkCongestionMonitor(self.config)
        self.jito_tracker = JitoTipsTracker(self.config)
        self.inflow_monitor = OnChainInflowMonitor(self.config)
        
        # Current risk state
        self.risk_level = SOLRiskLevel.NORMAL
        self.position_multiplier = 1.0
        self.override_active = False
        self.override_reason = ""
        
        self._lock = threading.Lock()
        
        logger.info("SOLDefenseModule initialized")
    
    def evaluate_risk(self) -> Tuple[SOLRiskLevel, float, str]:
        """
        Evaluate current SOL risk level.
        
        Returns:
            (risk_level, position_multiplier, reason)
        """
        with self._lock:
            reasons = []
            
            # Check network status
            network_status = self.network_monitor.get_status()
            
            if network_status == NetworkStatus.HALTED:
                self.risk_level = SOLRiskLevel.CRITICAL
                self.position_multiplier = 0.0
                return self.risk_level, self.position_multiplier, "NETWORK_HALTED"
            
            if network_status == NetworkStatus.CONGESTED:
                reasons.append("NETWORK_CONGESTED")
                self.position_multiplier = min(
                    self.position_multiplier,
                    self.config.position_reduction_congested
                )
            
            if network_status == NetworkStatus.DEGRADED:
                reasons.append("NETWORK_DEGRADED")
                self.position_multiplier = min(
                    self.position_multiplier,
                    self.config.position_reduction_degraded
                )
            
            # Check Jito tips
            tip_spike, tip_multiplier = self.jito_tracker.is_tip_spike()
            if tip_spike:
                reasons.append(f"JITO_SPIKE_{tip_multiplier:.1f}x")
                self.position_multiplier *= 0.8
            
            # Check inflow pressure
            inflow_elevated, net_flow = self.inflow_monitor.get_inflow_pressure()
            if inflow_elevated:
                reasons.append(f"INFLOW_PRESSURE_{net_flow:.0f}SOL")
                self.position_multiplier *= 0.7
            
            # Determine risk level
            if self.position_multiplier <= 0.0:
                self.risk_level = SOLRiskLevel.CRITICAL
            elif self.position_multiplier <= 0.3:
                self.risk_level = SOLRiskLevel.HIGH
            elif self.position_multiplier <= 0.6:
                self.risk_level = SOLRiskLevel.ELEVATED
            elif self.position_multiplier < 1.0:
                self.risk_level = SOLRiskLevel.NORMAL
            else:
                self.risk_level = SOLRiskLevel.LOW
            
            reason = ", ".join(reasons) if reasons else "NORMAL"
            
            return self.risk_level, self.position_multiplier, reason
    
    def should_override_signal(self, signal_direction: str) -> Tuple[bool, str]:
        """
        Check if SOL signal should be overridden.
        
        Args:
            signal_direction: 'long' or 'short'
            
        Returns:
            (should_override, reason)
        """
        with self._lock:
            risk_level, multiplier, reason = self.evaluate_risk()
            
            # Critical risk = override all
            if risk_level == SOLRiskLevel.CRITICAL:
                self.override_active = True
                self.override_reason = f"CRITICAL: {reason}"
                return True, self.override_reason
            
            # High risk = override longs only
            if risk_level == SOLRiskLevel.HIGH and signal_direction == 'long':
                self.override_active = True
                self.override_reason = f"HIGH_RISK_LONG: {reason}"
                return True, self.override_reason
            
            # Elevated + large inflow = override longs
            if risk_level == SOLRiskLevel.ELEVATED:
                elevated, net_flow = self.inflow_monitor.get_inflow_pressure()
                if elevated and signal_direction == 'long':
                    self.override_active = True
                    self.override_reason = f"INFLOW_OVERRIDE: {net_flow:.0f}SOL"
                    return True, self.override_reason
            
            self.override_active = False
            self.override_reason = ""
            return False, "OK"
    
    def get_adjusted_size(self, proposed_size: float) -> float:
        """Get position size adjusted for SOL risk."""
        with self._lock:
            return proposed_size * self.position_multiplier
    
    def get_status(self) -> Dict:
        """Get complete SOL defense status."""
        with self._lock:
            return {
                'risk_level': self.risk_level.name,
                'position_multiplier': self.position_multiplier,
                'override_active': self.override_active,
                'override_reason': self.override_reason,
                'network': {
                    'status': self.network_monitor.get_status().name,
                    'tps': self.network_monitor.current_metrics.tps,
                    'slot_time_ms': self.network_monitor.current_metrics.slot_time_ms
                },
                'jito': {
                    'avg_tip': self.jito_tracker.current_metrics.avg_tip_lamports,
                    'is_spike': self.jito_tracker.is_tip_spike()[0]
                },
                'onchain': {
                    'net_flow_sol': self.inflow_monitor.current_flow.net_flow_sol,
                    'large_transfers': self.inflow_monitor.current_flow.large_transfers
                }
            }

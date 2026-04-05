"""
HMATS v3.2 - Consolidated Proof Log System
==========================================

TASK 5: Single consolidated proof log per 4H tick showing:
- Data source (REAL/DISABLED)
- Quant generator (module.class)
- DRL status (ENABLED/DISABLED)
- DVOL status (REAL/DISABLED)
- Sentiment status (REAL/DISABLED)
- Mode (NORMAL/OPPORTUNITY/NO_TRADE)
- Intent (exposures + tranche)

This module aggregates status from all components for audit trail.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of each component."""
    REAL = "REAL"
    DISABLED = "DISABLED"
    MOCK = "MOCK"  # Should never appear in production
    ERROR = "ERROR"


@dataclass
class ProofLogEntry:
    """Single proof log entry for one 4H tick."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Data source
    data_status: ComponentStatus = ComponentStatus.DISABLED
    data_source: str = "none"
    data_age_seconds: float = 0.0
    
    # Quant
    quant_status: ComponentStatus = ComponentStatus.DISABLED
    quant_module: str = "none"
    quant_class: str = "none"
    quant_mode: str = "none"
    
    # DRL
    drl_status: ComponentStatus = ComponentStatus.DISABLED
    drl_checkpoint: str = "none"
    
    # DVOL
    dvol_status: ComponentStatus = ComponentStatus.DISABLED
    dvol_reason: str = ""
    
    # Sentiment
    sentiment_status: ComponentStatus = ComponentStatus.DISABLED
    sentiment_reason: str = ""
    
    # Mode
    system_mode: str = "UNKNOWN"
    mode_trigger: str = "none"
    
    # Signals
    signals: Dict[str, Dict] = field(default_factory=dict)
    
    # Intent
    intent_exposures: Dict[str, float] = field(default_factory=dict)
    intent_tranche: int = 0
    
    def to_log_string(self) -> str:
        """Generate formatted proof log string."""
        lines = [
            "=" * 60,
            "4H TICK PROOF LOG",
            "=" * 60,
            f"  Timestamp: {self.timestamp.isoformat()}",
            f"  Data: {self.data_status.value} source={self.data_source} age={self.data_age_seconds:.1f}s",
            f"  Quant: {self.quant_module}.{self.quant_class} ({self.quant_status.value})",
            f"  DRL: {self.drl_status.value}" + (f" checkpoint={self.drl_checkpoint}" if self.drl_status == ComponentStatus.REAL else ""),
            f"  DVOL: {self.dvol_status.value}" + (f" ({self.dvol_reason})" if self.dvol_reason else ""),
            f"  Sentiment: {self.sentiment_status.value}" + (f" ({self.sentiment_reason})" if self.sentiment_reason else ""),
            f"  Mode: {self.system_mode} | Trigger: {self.mode_trigger}",
        ]
        
        # Add signals
        if self.signals:
            for asset, sig in self.signals.items():
                lines.append(f"  Signals [{asset}]: dir={sig.get('direction', 0):.3f} conf={sig.get('confidence', 0):.3f}")
        
        # Add intent
        if self.intent_exposures:
            exp_str = ", ".join(f"{k}={v:.1%}" for k, v in self.intent_exposures.items())
            lines.append(f"  Intent: {exp_str} | Tranche: {self.intent_tranche}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def is_production_safe(self) -> bool:
        """Check if this tick would be safe for production trading."""
        # Data must be real
        if self.data_status != ComponentStatus.REAL:
            return False
        
        # Quant must be real
        if self.quant_status != ComponentStatus.REAL:
            return False
        
        # Mock components must not influence decisions
        # (DISABLED is OK, MOCK is not)
        if self.dvol_status == ComponentStatus.MOCK:
            return False
        if self.sentiment_status == ComponentStatus.MOCK:
            return False
        
        return True


class ProofLogCollector:
    """
    Collects status from all components and generates proof log.
    """
    
    def __init__(self):
        self._current_entry: Optional[ProofLogEntry] = None
        self._history: List[ProofLogEntry] = []
    
    def start_tick(self):
        """Start collecting for a new tick."""
        self._current_entry = ProofLogEntry()
    
    def set_data_status(
        self,
        status: ComponentStatus,
        source: str,
        age_seconds: float = 0.0
    ):
        """Set data source status."""
        if self._current_entry:
            self._current_entry.data_status = status
            self._current_entry.data_source = source
            self._current_entry.data_age_seconds = age_seconds
    
    def set_quant_status(
        self,
        status: ComponentStatus,
        module: str,
        class_name: str,
        mode: str = "real"
    ):
        """Set quant generator status."""
        if self._current_entry:
            self._current_entry.quant_status = status
            self._current_entry.quant_module = module
            self._current_entry.quant_class = class_name
            self._current_entry.quant_mode = mode
    
    def set_drl_status(
        self,
        status: ComponentStatus,
        checkpoint: str = "none"
    ):
        """Set DRL status."""
        if self._current_entry:
            self._current_entry.drl_status = status
            self._current_entry.drl_checkpoint = checkpoint
    
    def set_dvol_status(
        self,
        status: ComponentStatus,
        reason: str = ""
    ):
        """Set DVOL status."""
        if self._current_entry:
            self._current_entry.dvol_status = status
            self._current_entry.dvol_reason = reason
    
    def set_sentiment_status(
        self,
        status: ComponentStatus,
        reason: str = ""
    ):
        """Set sentiment status."""
        if self._current_entry:
            self._current_entry.sentiment_status = status
            self._current_entry.sentiment_reason = reason
    
    def set_mode(self, mode: str, trigger: str = "none"):
        """Set system mode."""
        if self._current_entry:
            self._current_entry.system_mode = mode
            self._current_entry.mode_trigger = trigger
    
    def add_signal(self, asset: str, direction: float, confidence: float):
        """Add signal for asset."""
        if self._current_entry:
            self._current_entry.signals[asset] = {
                "direction": direction,
                "confidence": confidence
            }
    
    def set_intent(self, exposures: Dict[str, float], tranche: int):
        """Set trade intent."""
        if self._current_entry:
            self._current_entry.intent_exposures = exposures
            self._current_entry.intent_tranche = tranche
    
    def finalize_tick(self) -> ProofLogEntry:
        """Finalize and log the current tick."""
        if not self._current_entry:
            self._current_entry = ProofLogEntry()
        
        entry = self._current_entry
        self._history.append(entry)
        
        # Log the proof
        logger.info(entry.to_log_string())
        
        # Check production safety
        if not entry.is_production_safe():
            logger.warning("[PROOF LOG] [WARN]️ This tick is NOT production-safe")
        
        self._current_entry = None
        return entry
    
    def get_history(self) -> List[ProofLogEntry]:
        """Get proof log history."""
        return list(self._history)


# Singleton
_proof_collector: Optional[ProofLogCollector] = None


def get_proof_collector() -> ProofLogCollector:
    """Get or create proof log collector singleton."""
    global _proof_collector
    if _proof_collector is None:
        _proof_collector = ProofLogCollector()
    return _proof_collector


def collect_component_statuses() -> Dict[str, ComponentStatus]:
    """
    Collect current status of all components.
    
    Returns dict with component -> status mapping.
    """
    statuses = {}
    
    # Data
    try:
        from core.market_data_provider import get_market_data_provider
        provider = get_market_data_provider()
        if provider.config.ENABLE_LIVE_DATA:
            statuses['data'] = ComponentStatus.REAL
        else:
            statuses['data'] = ComponentStatus.DISABLED
    except Exception:
        statuses['data'] = ComponentStatus.ERROR
    
    # Quant
    try:
        from core.signal_generators_v32 import get_signal_config
        config = get_signal_config()
        if config.QUANT_MODE == "real":
            statuses['quant'] = ComponentStatus.REAL
        else:
            statuses['quant'] = ComponentStatus.DISABLED
    except Exception:
        statuses['quant'] = ComponentStatus.ERROR
    
    # DRL
    try:
        from core.signal_generators_v32 import get_signal_config
        config = get_signal_config()
        if config.ENABLE_DRL:
            statuses['drl'] = ComponentStatus.REAL
        else:
            statuses['drl'] = ComponentStatus.DISABLED
    except Exception:
        statuses['drl'] = ComponentStatus.ERROR
    
    # DVOL
    try:
        from core.deribit_dvol_client import dvol_triggers_enabled, is_dvol_mock
        if is_dvol_mock():
            statuses['dvol'] = ComponentStatus.DISABLED
        elif dvol_triggers_enabled():
            statuses['dvol'] = ComponentStatus.REAL
        else:
            statuses['dvol'] = ComponentStatus.DISABLED
    except Exception:
        statuses['dvol'] = ComponentStatus.DISABLED
    
    # Sentiment
    try:
        from core.vllm_inference_wrapper import VLLM_AVAILABLE, TRANSFORMERS_AVAILABLE
        if VLLM_AVAILABLE or TRANSFORMERS_AVAILABLE:
            statuses['sentiment'] = ComponentStatus.REAL
        else:
            statuses['sentiment'] = ComponentStatus.DISABLED
    except Exception:
        statuses['sentiment'] = ComponentStatus.DISABLED
    
    return statuses


def print_component_status_summary():
    """Print current component status summary."""
    statuses = collect_component_statuses()
    
    print("\n" + "=" * 50)
    print("HMATS v3.2 COMPONENT STATUS")
    print("=" * 50)
    
    for component, status in statuses.items():
        emoji = "✓" if status == ComponentStatus.REAL else ("✗" if status == ComponentStatus.ERROR else "○")
        print(f"  {emoji} {component.upper()}: {status.value}")
    
    print("=" * 50)
    
    # Production readiness
    real_count = sum(1 for s in statuses.values() if s == ComponentStatus.REAL)
    disabled_count = sum(1 for s in statuses.values() if s == ComponentStatus.DISABLED)
    error_count = sum(1 for s in statuses.values() if s == ComponentStatus.ERROR)
    
    if error_count > 0:
        print("[WARN]️ ERRORS detected - NOT production ready")
    elif statuses.get('data') != ComponentStatus.REAL:
        print("[WARN]️ Data not REAL - NOT production ready")
    elif statuses.get('quant') != ComponentStatus.REAL:
        print("[WARN]️ Quant not REAL - NOT production ready")
    else:
        print(f"✓ Production ready ({real_count} REAL, {disabled_count} DISABLED)")
    
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print_component_status_summary()
    
    # Demo proof log
    collector = get_proof_collector()
    collector.start_tick()
    
    collector.set_data_status(ComponentStatus.REAL, "kraken_ccxt", 1.5)
    collector.set_quant_status(ComponentStatus.REAL, "core.signal_generators_real", "QuantSignalGenerator", "real")
    collector.set_drl_status(ComponentStatus.DISABLED)
    collector.set_dvol_status(ComponentStatus.DISABLED, "mock mode")
    collector.set_sentiment_status(ComponentStatus.DISABLED, "no vLLM")
    collector.set_mode("NORMAL", "none")
    collector.add_signal("BTC", 0.15, 0.72)
    collector.add_signal("SOL", 0.28, 0.68)
    collector.set_intent({"BTC": 0.1, "SOL": 0.15}, 1)
    
    entry = collector.finalize_tick()
    
    print(f"\nProduction safe: {entry.is_production_safe()}")

"""
================================================================================
HOT RESTART STATE PERSISTENCE - Complete State Recovery
================================================================================
Version: 3.3.0
Purpose: Resolves HIGH impact item #6 from CTO Audit

PROBLEM SOLVED:
- What survives restart was not fully specified
- Positions could be lost or orphaned after restart
- Intents and tranche state not persisted
- No cross-validation between proof log and SQLite

SOLUTION:
- Define EXACTLY what state is persisted:
  1. POSITIONS: All open positions with entry prices, sizes, stop levels
  2. TRANCHES: Current tranche level per position
  3. INTENTS: Pending trade intents not yet executed
  4. MODE: System mode at shutdown (NO_TRADE/OPPORTUNITY/NORMAL)
  5. STOPS: All active stop orders (soft and hard)
  6. METRICS: Running drawdown, peak equity, daily P&L
  
- State is written to SQLite on every significant change
- Proof log provides independent verification
- On restart, state is loaded and validated against exchange

PERSISTENCE TRIGGERS:
- Position opened/closed
- Tranche escalated
- Stop modified
- Mode changed
- Every 5 minutes (heartbeat)
================================================================================
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class PersistenceEvent(Enum):
    """Events that trigger state persistence."""
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    TRANCHE_ESCALATED = "TRANCHE_ESCALATED"
    STOP_MODIFIED = "STOP_MODIFIED"
    MODE_CHANGED = "MODE_CHANGED"
    HEARTBEAT = "HEARTBEAT"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class PersistedPosition:
    """Position state to persist."""
    asset: str
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    avg_price: float
    tranche_level: int  # 0-4
    soft_stop: float
    hard_stop: float
    unrealized_pnl: float
    opened_at: str  # ISO timestamp
    last_updated: str


@dataclass
class PersistedIntent:
    """Trade intent to persist."""
    intent_id: str
    asset: str
    direction: str
    target_exposure: float
    tranche_target: int
    created_at: str
    expires_at: str
    status: str  # "PENDING", "EXECUTING", "COMPLETED", "EXPIRED"


@dataclass
class PersistedStops:
    """Stop orders to persist."""
    asset: str
    soft_stop_price: Optional[float]
    soft_stop_active: bool
    hard_stop_price: float
    hard_stop_exchange_id: Optional[str]


@dataclass
class PersistedMetrics:
    """Trading metrics to persist."""
    account_balance: float
    peak_equity: float
    current_drawdown: float
    drawdown_status: str
    daily_pnl: float
    daily_trades: int
    last_trade_time: Optional[str]


@dataclass
class SystemState:
    """Complete system state for persistence."""
    
    # Core state
    system_mode: str  # NO_TRADE, OPPORTUNITY, NORMAL
    opportunity_trigger: Optional[str]  # Which trigger if OPPORTUNITY
    
    # Positions
    positions: Dict[str, PersistedPosition] = field(default_factory=dict)
    
    # Pending intents
    intents: Dict[str, PersistedIntent] = field(default_factory=dict)
    
    # Stops
    stops: Dict[str, PersistedStops] = field(default_factory=dict)
    
    # Metrics
    metrics: PersistedMetrics = field(default_factory=lambda: PersistedMetrics(
        account_balance=0.0,
        peak_equity=0.0,
        current_drawdown=0.0,
        drawdown_status="NORMAL",
        daily_pnl=0.0,
        daily_trades=0,
        last_trade_time=None,
    ))
    
    # Thesis budget governor state
    thesis_budget_state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    version: str = "3.3.0"
    persisted_at: str = ""
    persist_event: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "system_mode": self.system_mode,
            "opportunity_trigger": self.opportunity_trigger,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "intents": {k: asdict(v) for k, v in self.intents.items()},
            "stops": {k: asdict(v) for k, v in self.stops.items()},
            "metrics": asdict(self.metrics),
            "thesis_budget_state": self.thesis_budget_state,
            "version": self.version,
            "persisted_at": self.persisted_at,
            "persist_event": self.persist_event,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemState':
        state = cls(
            system_mode=data.get("system_mode", "NORMAL"),
            opportunity_trigger=data.get("opportunity_trigger"),
        )
        
        # Positions
        for asset, pos_data in data.get("positions", {}).items():
            state.positions[asset] = PersistedPosition(**pos_data)
        
        # Intents
        for intent_id, intent_data in data.get("intents", {}).items():
            state.intents[intent_id] = PersistedIntent(**intent_data)
        
        # Stops
        for asset, stop_data in data.get("stops", {}).items():
            state.stops[asset] = PersistedStops(**stop_data)
        
        # Metrics
        if "metrics" in data:
            state.metrics = PersistedMetrics(**data["metrics"])
        
        # Thesis budget governor
        state.thesis_budget_state = data.get("thesis_budget_state", {})

        state.version = data.get("version", "3.3.0")
        state.persisted_at = data.get("persisted_at", "")
        state.persist_event = data.get("persist_event", "")

        return state


@dataclass
class ValidationResult:
    """Result of state validation against exchange."""
    is_valid: bool
    position_mismatches: List[str] = field(default_factory=list)
    orphan_positions: List[str] = field(default_factory=list)  # On exchange but not in state
    missing_positions: List[str] = field(default_factory=list)  # In state but not on exchange
    stop_mismatches: List[str] = field(default_factory=list)


class HotRestartPersistence:
    """
    Manages state persistence for hot restart.
    
    USAGE:
        persistence = HotRestartPersistence("/path/to/state.db")
        
        # On state change
        persistence.persist(current_state, PersistenceEvent.POSITION_OPENED)
        
        # On restart
        state = persistence.load()
        validation = persistence.validate_against_exchange(exchange_state)
        
        if validation.is_valid:
            # Resume from persisted state
            pass
        else:
            # Reconcile differences
            pass
    """
    
    def __init__(self, db_path: str = "hmats_state.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._thesis_budget_governor = None

        # Initialize database
        self._init_db()

        logger.info(f"HotRestartPersistence initialized: {self.db_path}")

    def set_thesis_budget_governor(self, governor) -> None:
        """Register governor so persist() can capture its state."""
        self._thesis_budget_governor = governor
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    persist_event TEXT NOT NULL,
                    version TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_json TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    persist_event TEXT NOT NULL,
                    version TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def persist(self, state: SystemState, event: PersistenceEvent):
        """
        Persist current state to database.
        
        Args:
            state: Current system state
            event: What triggered this persistence
        """
        now = datetime.now(timezone.utc).isoformat()
        state.persisted_at = now
        state.persist_event = event.value

        if self._thesis_budget_governor is not None:
            try:
                state.thesis_budget_state = self._thesis_budget_governor.to_dict()
                logger.info(
                    f"[THESIS_BUDGET] Persisted governor state: "
                    f"{state.thesis_budget_state.get('thesis_count', 0)} theses"
                )
            except Exception as e:
                logger.warning(f"[THESIS_BUDGET] Failed to persist governor state: {e}")

        state_json = json.dumps(state.to_dict())
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Update current state (single row)
                conn.execute("DELETE FROM system_state")
                conn.execute(
                    "INSERT INTO system_state (id, state_json, persisted_at, persist_event, version) VALUES (1, ?, ?, ?, ?)",
                    (state_json, now, event.value, state.version)
                )
                
                # Add to history
                conn.execute(
                    "INSERT INTO state_history (state_json, persisted_at, persist_event, version) VALUES (?, ?, ?, ?)",
                    (state_json, now, event.value, state.version)
                )
                
                conn.commit()
        
        logger.debug(f"State persisted: event={event.value}")
    
    def load(self) -> Optional[SystemState]:
        """
        Load persisted state.
        
        Returns None if no state exists.
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT state_json FROM system_state WHERE id = 1"
                )
                row = cursor.fetchone()
                
                if not row:
                    logger.info("No persisted state found")
                    return None
                
                try:
                    data = json.loads(row[0])
                    state = SystemState.from_dict(data)
                    logger.info(f"State loaded: mode={state.system_mode}, positions={len(state.positions)}")
                    if state.thesis_budget_state:
                        logger.info(
                            f"[THESIS_BUDGET] Restored state available: "
                            f"{state.thesis_budget_state.get('thesis_count', 0)} theses"
                        )
                    return state
                except Exception as e:
                    logger.error(f"Failed to load state: {e}")
                    return None
    
    def validate_against_exchange(
        self,
        exchange_positions: Dict[str, Dict],
        exchange_stops: Dict[str, Dict],
    ) -> ValidationResult:
        """
        Validate persisted state against actual exchange state.
        
        Args:
            exchange_positions: Actual positions from exchange
            exchange_stops: Actual stop orders from exchange
        
        Returns:
            ValidationResult with any mismatches
        """
        state = self.load()
        if not state:
            # No persisted state - just use exchange state
            return ValidationResult(
                is_valid=True,
                orphan_positions=list(exchange_positions.keys()),
            )
        
        result = ValidationResult(is_valid=True)
        
        # Check for orphan positions (on exchange but not in state)
        for asset in exchange_positions:
            if asset not in state.positions:
                result.orphan_positions.append(asset)
                result.is_valid = False
        
        # Check for missing positions (in state but not on exchange)
        for asset in state.positions:
            if asset not in exchange_positions:
                result.missing_positions.append(asset)
                result.is_valid = False
        
        # Check position details match
        for asset, ex_pos in exchange_positions.items():
            if asset in state.positions:
                saved_pos = state.positions[asset]
                
                # Check size matches (within tolerance)
                ex_size = ex_pos.get("size", 0)
                if abs(saved_pos.size - ex_size) / max(saved_pos.size, 1e-8) > 0.01:
                    result.position_mismatches.append(
                        f"{asset}: size mismatch (saved={saved_pos.size}, exchange={ex_size})"
                    )
                    result.is_valid = False
                
                # Check direction matches
                ex_dir = ex_pos.get("direction", "")
                if saved_pos.direction != ex_dir:
                    result.position_mismatches.append(
                        f"{asset}: direction mismatch (saved={saved_pos.direction}, exchange={ex_dir})"
                    )
                    result.is_valid = False
        
        # Check stop orders
        for asset, ex_stop in exchange_stops.items():
            if asset in state.stops:
                saved_stop = state.stops[asset]
                
                # Check hard stop exists on exchange
                if saved_stop.hard_stop_exchange_id:
                    if saved_stop.hard_stop_exchange_id != ex_stop.get("order_id"):
                        result.stop_mismatches.append(
                            f"{asset}: hard stop order ID mismatch"
                        )
        
        logger.info(
            f"State validation: valid={result.is_valid}, "
            f"orphans={len(result.orphan_positions)}, "
            f"missing={len(result.missing_positions)}, "
            f"mismatches={len(result.position_mismatches)}"
        )
        
        return result
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get state history for debugging."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT state_json, persisted_at, persist_event FROM state_history ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                
                return [
                    {
                        "state": json.loads(row[0]),
                        "persisted_at": row[1],
                        "event": row[2],
                    }
                    for row in cursor.fetchall()
                ]
    
    def clear(self):
        """Clear all persisted state (for testing)."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM system_state")
                conn.execute("DELETE FROM state_history")
                conn.commit()
        logger.info("State cleared")


# Singleton instance
_persistence: Optional[HotRestartPersistence] = None


def get_persistence(db_path: str = "hmats_state.db") -> HotRestartPersistence:
    """Get or create singleton persistence manager."""
    global _persistence
    if _persistence is None:
        _persistence = HotRestartPersistence(db_path)
    return _persistence


def reset_persistence():
    """Reset singleton (for testing)."""
    global _persistence
    _persistence = None

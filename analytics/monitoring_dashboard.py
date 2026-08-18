#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - Lightweight Monitoring Dashboard
================================================================================
Purpose: Real-time monitoring of trading system health and performance

P3 SOTA Requirement:
- "监控面板（轻量即可）"
- "必须至少可视化: PnL、仓位、风险状态、gate 拒单原因、数据延迟、WS 重连次数"

Dashboard Metrics:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      HMATS Monitor                               │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
    │  │   PnL    │  │ Position │  │   Risk   │  │  System  │        │
    │  │  Panel   │  │  Panel   │  │  Panel   │  │  Panel   │        │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
    │                                                                  │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │                    Event Log                              │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

Implementation:
- Streamlit-based web UI (lightweight, Python-native)
- JSON-based data export for external dashboards
- Console-based text dashboard for terminal monitoring

Author: HMATS v5.1 P3 Fix
Version: 5.1.0
================================================================================
"""

import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from collections import deque
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC TYPES
# =============================================================================

class MetricType(Enum):
    """Types of monitored metrics."""
    PNL = "pnl"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"
    DATA = "data"
    EXECUTION = "execution"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


# =============================================================================
# METRIC DATA STRUCTURES
# =============================================================================

@dataclass
class PnLMetrics:
    """PnL metrics."""
    total_pnl_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    daily_pnl_usd: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PositionMetrics:
    """Position metrics."""
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> size
    position_values: Dict[str, float] = field(default_factory=dict)  # symbol -> USD value
    total_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0  # Long - Short
    gross_exposure_usd: float = 0.0  # |Long| + |Short|
    leverage: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskMetrics:
    """Risk metrics."""
    # Vetoes
    total_vetoes_today: int = 0
    hard_vetoes_today: int = 0
    soft_vetoes_today: int = 0
    veto_reasons: List[str] = field(default_factory=list)
    last_veto_time: Optional[datetime] = None
    
    # Gate status
    gate_status: Dict[str, bool] = field(default_factory=dict)
    trades_blocked: int = 0
    trades_allowed: int = 0
    
    # Exposure limits
    exposure_utilization: float = 0.0  # Current / Max
    drawdown_utilization: float = 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["last_veto_time"] = self.last_veto_time.isoformat() if self.last_veto_time else None
        return d


@dataclass
class SystemMetrics:
    """System health metrics."""
    # Status
    status: HealthStatus = HealthStatus.HEALTHY
    uptime_seconds: float = 0.0
    mode: str = "NORMAL"
    regime: str = "unknown"
    
    # Data feed
    ws_connected: bool = False
    ws_reconnect_count: int = 0
    last_data_timestamp: Optional[datetime] = None
    data_latency_ms: float = 0.0
    
    # Components
    components_healthy: Dict[str, bool] = field(default_factory=dict)
    
    # Performance
    tick_processing_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["last_data_timestamp"] = self.last_data_timestamp.isoformat() if self.last_data_timestamp else None
        return d


@dataclass
class DashboardSnapshot:
    """Complete dashboard snapshot."""
    timestamp: datetime
    pnl: PnLMetrics
    positions: PositionMetrics
    risk: RiskMetrics
    system: SystemMetrics
    events: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "pnl": self.pnl.to_dict(),
            "positions": self.positions.to_dict(),
            "risk": self.risk.to_dict(),
            "system": self.system.to_dict(),
            "events": self.events[:50]  # Last 50 events
        }


# =============================================================================
# METRIC COLLECTOR
# =============================================================================

class MetricCollector:
    """
    Collects and aggregates metrics from system components.
    """
    
    def __init__(self):
        self._start_time = datetime.now(timezone.utc)
        
        # Current metrics
        self._pnl = PnLMetrics()
        self._positions = PositionMetrics()
        self._risk = RiskMetrics()
        self._system = SystemMetrics()
        
        # Event log
        self._events: deque = deque(maxlen=1000)
        
        # Subscribers
        self._subscribers: List[Callable[[DashboardSnapshot], None]] = []
        
        # Update lock
        self._lock = threading.Lock()
        
    def update_pnl(
        self,
        total_pnl: Optional[float] = None,
        realized: Optional[float] = None,
        unrealized: Optional[float] = None,
        drawdown_pct: Optional[float] = None
    ):
        """Update PnL metrics."""
        with self._lock:
            if total_pnl is not None:
                self._pnl.total_pnl_usd = total_pnl
            if realized is not None:
                self._pnl.realized_pnl_usd = realized
            if unrealized is not None:
                self._pnl.unrealized_pnl_usd = unrealized
            if drawdown_pct is not None:
                self._pnl.current_drawdown_pct = drawdown_pct
                self._pnl.max_drawdown_pct = max(self._pnl.max_drawdown_pct, drawdown_pct)
                
    def update_positions(
        self,
        positions: Optional[Dict[str, float]] = None,
        values: Optional[Dict[str, float]] = None
    ):
        """Update position metrics."""
        with self._lock:
            if positions is not None:
                self._positions.positions = positions.copy()
            if values is not None:
                self._positions.position_values = values.copy()
                self._positions.total_exposure_usd = sum(abs(v) for v in values.values())
                self._positions.net_exposure_usd = sum(values.values())
                self._positions.gross_exposure_usd = sum(abs(v) for v in values.values())
                
    def record_veto(self, veto_type: str, reason: str):
        """Record a risk veto."""
        with self._lock:
            self._risk.total_vetoes_today += 1
            if veto_type == "HARD":
                self._risk.hard_vetoes_today += 1
            else:
                self._risk.soft_vetoes_today += 1
            self._risk.veto_reasons.append(f"[{veto_type}] {reason}")
            self._risk.veto_reasons = self._risk.veto_reasons[-20:]  # Keep last 20
            self._risk.last_veto_time = datetime.now(timezone.utc)
            
            self._add_event("risk_veto", {"type": veto_type, "reason": reason})
            
    def update_gate_status(self, gate_name: str, allowed: bool):
        """Update trade gate status."""
        with self._lock:
            self._risk.gate_status[gate_name] = allowed
            if allowed:
                self._risk.trades_allowed += 1
            else:
                self._risk.trades_blocked += 1
                
    def update_system_status(
        self,
        mode: Optional[str] = None,
        regime: Optional[str] = None,
        ws_connected: Optional[bool] = None,
        data_latency_ms: Optional[float] = None
    ):
        """Update system status."""
        with self._lock:
            if mode is not None:
                self._system.mode = mode
            if regime is not None:
                self._system.regime = regime
            if ws_connected is not None:
                if not ws_connected and self._system.ws_connected:
                    self._system.ws_reconnect_count += 1
                self._system.ws_connected = ws_connected
            if data_latency_ms is not None:
                self._system.data_latency_ms = data_latency_ms
                self._system.last_data_timestamp = datetime.now(timezone.utc)
                
            # Update uptime
            self._system.uptime_seconds = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            
            # Determine health status
            self._system.status = self._determine_health()
            
    def _determine_health(self) -> HealthStatus:
        """Determine overall system health."""
        issues = 0
        
        # Check WebSocket
        if not self._system.ws_connected:
            issues += 2
            
        # Check data latency
        if self._system.data_latency_ms > 1000:
            issues += 1
        elif self._system.data_latency_ms > 5000:
            issues += 2
            
        # Check reconnects
        if self._system.ws_reconnect_count > 10:
            issues += 1
            
        # Check drawdown
        if self._risk.hard_vetoes_today > 5:
            issues += 1
            
        if issues == 0:
            return HealthStatus.HEALTHY
        elif issues <= 2:
            return HealthStatus.DEGRADED
        elif issues <= 4:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
            
    def _add_event(self, event_type: str, data: Dict):
        """Add event to log."""
        self._events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "data": data
        })
        
    def add_event(self, event_type: str, data: Dict):
        """Public method to add event."""
        with self._lock:
            self._add_event(event_type, data)
            
    def get_snapshot(self) -> DashboardSnapshot:
        """Get current dashboard snapshot."""
        with self._lock:
            return DashboardSnapshot(
                timestamp=datetime.now(timezone.utc),
                pnl=PnLMetrics(**asdict(self._pnl)),
                positions=PositionMetrics(**asdict(self._positions)),
                risk=RiskMetrics(**asdict(self._risk)),
                system=SystemMetrics(**asdict(self._system)),
                events=list(self._events)
            )
            
    def subscribe(self, callback: Callable[[DashboardSnapshot], None]):
        """Subscribe to dashboard updates."""
        self._subscribers.append(callback)
        
    def notify_subscribers(self):
        """Notify all subscribers with current snapshot."""
        snapshot = self.get_snapshot()
        for callback in self._subscribers:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"[Monitor] Subscriber error: {e}")


# =============================================================================
# CONSOLE DASHBOARD
# =============================================================================

class ConsoleDashboard:
    """
    Text-based console dashboard.
    
    Displays metrics in terminal with color coding.
    """
    
    # ANSI colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def __init__(self, collector: MetricCollector):
        self.collector = collector
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def display(self, snapshot: Optional[DashboardSnapshot] = None):
        """Display current metrics."""
        if snapshot is None:
            snapshot = self.collector.get_snapshot()
            
        self._clear_screen()
        self._print_header(snapshot)
        self._print_pnl(snapshot.pnl)
        self._print_positions(snapshot.positions)
        self._print_risk(snapshot.risk)
        self._print_system(snapshot.system)
        self._print_events(snapshot.events)
        
    def _clear_screen(self):
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")
        
    def _print_header(self, snapshot: DashboardSnapshot):
        """Print dashboard header."""
        status_color = {
            HealthStatus.HEALTHY: self.GREEN,
            HealthStatus.DEGRADED: self.YELLOW,
            HealthStatus.UNHEALTHY: self.RED,
            HealthStatus.CRITICAL: self.RED + self.BOLD
        }.get(snapshot.system.status, self.RESET)
        
        print(f"{self.BOLD}═" * 70)
        print(f"  HMATS MONITOR  │  {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Status: {status_color}{snapshot.system.status.value.upper()}{self.RESET}  │  "
              f"Mode: {snapshot.system.mode}  │  Regime: {snapshot.system.regime}")
        print(f"═" * 70 + self.RESET)
        print()
        
    def _print_pnl(self, pnl: PnLMetrics):
        """Print PnL section."""
        color = self.GREEN if pnl.total_pnl_usd >= 0 else self.RED
        
        print(f"{self.BLUE}┌─ PnL ──────────────────────────────────────────────────────────┐{self.RESET}")
        print(f"  Total PnL:      {color}${pnl.total_pnl_usd:>12,.2f}{self.RESET}")
        print(f"  Realized:       ${pnl.realized_pnl_usd:>12,.2f}")
        print(f"  Unrealized:     ${pnl.unrealized_pnl_usd:>12,.2f}")
        print(f"  Daily:          {color}${pnl.daily_pnl_usd:>12,.2f} ({pnl.daily_pnl_pct:+.2f}%){self.RESET}")
        
        dd_color = self.RED if pnl.current_drawdown_pct > 3 else self.YELLOW if pnl.current_drawdown_pct > 1 else self.RESET
        print(f"  Drawdown:       {dd_color}{pnl.current_drawdown_pct:>12.2f}% (max: {pnl.max_drawdown_pct:.2f}%){self.RESET}")
        print(f"{self.BLUE}└───────────────────────────────────────────────────────────────┘{self.RESET}")
        print()
        
    def _print_positions(self, pos: PositionMetrics):
        """Print positions section."""
        print(f"{self.BLUE}┌─ Positions ────────────────────────────────────────────────────┐{self.RESET}")
        print(f"  Total Exposure: ${pos.total_exposure_usd:>12,.2f}")
        print(f"  Net Exposure:   ${pos.net_exposure_usd:>12,.2f}")
        
        for symbol, size in list(pos.positions.items())[:5]:
            value = pos.position_values.get(symbol, 0)
            color = self.GREEN if size > 0 else self.RED if size < 0 else self.RESET
            print(f"  {symbol:<12}  {color}{size:>10.4f}  (${value:>10,.2f}){self.RESET}")
            
        print(f"{self.BLUE}└───────────────────────────────────────────────────────────────┘{self.RESET}")
        print()
        
    def _print_risk(self, risk: RiskMetrics):
        """Print risk section."""
        print(f"{self.BLUE}┌─ Risk ─────────────────────────────────────────────────────────┐{self.RESET}")
        
        veto_color = self.RED if risk.hard_vetoes_today > 0 else self.RESET
        print(f"  Vetoes Today:   {veto_color}{risk.total_vetoes_today} (Hard: {risk.hard_vetoes_today}, Soft: {risk.soft_vetoes_today}){self.RESET}")
        print(f"  Trades:         Allowed: {risk.trades_allowed}  │  Blocked: {risk.trades_blocked}")
        
        # Recent veto reasons
        if risk.veto_reasons:
            print(f"  Recent Vetoes:")
            for reason in risk.veto_reasons[-3:]:
                print(f"    {self.YELLOW}-> {reason[:50]}{self.RESET}")
                
        print(f"{self.BLUE}└───────────────────────────────────────────────────────────────┘{self.RESET}")
        print()
        
    def _print_system(self, sys: SystemMetrics):
        """Print system section."""
        print(f"{self.BLUE}┌─ System ───────────────────────────────────────────────────────┐{self.RESET}")
        
        ws_color = self.GREEN if sys.ws_connected else self.RED
        print(f"  WebSocket:      {ws_color}{'Connected' if sys.ws_connected else 'Disconnected'}{self.RESET}  "
              f"(reconnects: {sys.ws_reconnect_count})")
        
        latency_color = self.GREEN if sys.data_latency_ms < 100 else self.YELLOW if sys.data_latency_ms < 500 else self.RED
        print(f"  Data Latency:   {latency_color}{sys.data_latency_ms:.0f}ms{self.RESET}")
        
        uptime_hours = sys.uptime_seconds / 3600
        print(f"  Uptime:         {uptime_hours:.1f} hours")
        
        print(f"{self.BLUE}└───────────────────────────────────────────────────────────────┘{self.RESET}")
        print()
        
    def _print_events(self, events: List[Dict]):
        """Print recent events."""
        print(f"{self.BLUE}┌─ Recent Events ────────────────────────────────────────────────┐{self.RESET}")
        
        for event in events[-5:]:
            ts = event["timestamp"].split("T")[1][:8] if "T" in event["timestamp"] else event["timestamp"]
            print(f"  [{ts}] {event['type']}: {str(event['data'])[:45]}")
            
        print(f"{self.BLUE}└───────────────────────────────────────────────────────────────┘{self.RESET}")
        
    def start_live(self, refresh_interval: float = 1.0):
        """Start live updating display."""
        self._running = True
        
        def _update_loop():
            while self._running:
                self.display()
                time.sleep(refresh_interval)
                
        self._thread = threading.Thread(target=_update_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop live display."""
        self._running = False


# =============================================================================
# JSON EXPORTER
# =============================================================================

class JSONExporter:
    """
    Export dashboard data as JSON for external dashboards.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("logs/dashboard")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_snapshot(self, snapshot: DashboardSnapshot) -> Path:
        """Export snapshot to JSON file."""
        filename = f"dashboard_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)
            
        return filepath
        
    def export_latest(self, snapshot: DashboardSnapshot):
        """Export to latest.json (overwrites)."""
        filepath = self.output_dir / "latest.json"
        
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)


# =============================================================================
# STREAMLIT DASHBOARD (Optional)
# =============================================================================

def create_streamlit_dashboard(collector: MetricCollector) -> str:
    """
    Generate Streamlit dashboard code.
    
    Returns Python code that can be saved and run with `streamlit run`.
    """
    code = '''
import streamlit as st
import json
from pathlib import Path
import time

st.set_page_config(page_title="HMATS Monitor", layout="wide")

# Load latest data
data_path = Path("logs/dashboard/latest.json")

def load_data():
    if data_path.exists():
        with open(data_path, encoding="utf-8") as f:
            return json.load(f)
    return None

# Auto-refresh
st.experimental_rerun_interval = 1000  # 1 second

data = load_data()
if not data:
    st.error("No dashboard data available")
    st.stop()

# Header
st.title("🤖 HMATS Trading Monitor")
st.text(f"Last Update: {data['timestamp']}")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total PnL", f"${data['pnl']['total_pnl_usd']:,.2f}")
    
with col2:
    st.metric("Exposure", f"${data['positions']['total_exposure_usd']:,.2f}")
    
with col3:
    st.metric("Drawdown", f"{data['pnl']['current_drawdown_pct']:.2f}%")
    
with col4:
    st.metric("Status", data['system']['status'].upper())

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["PnL", "Positions", "Risk", "System"])

with tab1:
    st.subheader("PnL Metrics")
    st.json(data['pnl'])
    
with tab2:
    st.subheader("Positions")
    st.json(data['positions'])
    
with tab3:
    st.subheader("Risk Status")
    st.write(f"Vetoes Today: {data['risk']['total_vetoes_today']}")
    st.write(f"Trades Blocked: {data['risk']['trades_blocked']}")
    if data['risk']['veto_reasons']:
        st.write("Recent Vetoes:")
        for reason in data['risk']['veto_reasons'][-5:]:
            st.write(f"  • {reason}")
            
with tab4:
    st.subheader("System Health")
    st.write(f"WebSocket: {'🟢 Connected' if data['system']['ws_connected'] else '🔴 Disconnected'}")
    st.write(f"Reconnects: {data['system']['ws_reconnect_count']}")
    st.write(f"Data Latency: {data['system']['data_latency_ms']:.0f}ms")
    
# Events
st.subheader("Recent Events")
for event in data['events'][-10:]:
    st.text(f"[{event['timestamp'][:19]}] {event['type']}: {event['data']}")
'''
    return code


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_collector: Optional[MetricCollector] = None
_console: Optional[ConsoleDashboard] = None
_exporter: Optional[JSONExporter] = None


def get_metric_collector() -> MetricCollector:
    """Get singleton MetricCollector instance."""
    global _collector
    if _collector is None:
        _collector = MetricCollector()
    return _collector


def get_console_dashboard() -> ConsoleDashboard:
    """Get singleton ConsoleDashboard instance."""
    global _console, _collector
    if _console is None:
        if _collector is None:
            _collector = MetricCollector()
        _console = ConsoleDashboard(_collector)
    return _console


def get_json_exporter(output_dir: Optional[Path] = None) -> JSONExporter:
    """Get singleton JSONExporter instance."""
    global _exporter
    if _exporter is None:
        _exporter = JSONExporter(output_dir)
    return _exporter


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MetricCollector',
    'ConsoleDashboard',
    'JSONExporter',
    'DashboardSnapshot',
    'PnLMetrics',
    'PositionMetrics',
    'RiskMetrics',
    'SystemMetrics',
    'MetricType',
    'HealthStatus',
    'get_metric_collector',
    'get_console_dashboard',
    'get_json_exporter',
    'create_streamlit_dashboard',
]

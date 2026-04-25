"""
PersistenceAndAlerting - Discord Notifications & SQLite Trade Logging
======================================================================
Production-grade persistence and alerting for HMATS.

Features:
- SQLite database for trade logging and state recovery
- Discord webhook integration for real-time alerts
- Hot-restart state recovery
- Audit trail with slippage/fee tracking
- Performance metrics persistence

Alert Categories:
- CRITICAL: Circuit breaker trips, system failures
- WARNING: Watchdog restarts, degraded connections
- INFO: Trade executions, daily summaries
- DEBUG: Detailed operational logs

Database Tables:
- trades: All trade intents and fills
- orders: Order lifecycle tracking
- metrics: Periodic performance snapshots
- events: System events and alerts
- state: System state for hot-restart

Version: 4.0 (Discord)
"""

import os
import sys
import time
import json
import asyncio
import logging
import sqlite3
import threading
import hashlib
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from collections import deque
from pathlib import Path
import queue
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('PersistenceAndAlerting')


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class AlertCategory(Enum):
    """Alert categories"""
    SYSTEM = auto()       # System events (start, stop, restart)
    TRADING = auto()      # Trade executions
    RISK = auto()         # Risk events (circuit breaker, limits)
    PERFORMANCE = auto()  # Performance metrics
    HARDWARE = auto()     # GPU/CPU/RAM warnings


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class TradeIntent:
    """Trade intent record"""
    intent_id: str
    timestamp: float
    asset: str
    side: str
    target_quantity: float
    target_price: float
    signal_source: str
    signal_strength: float
    regime: str
    vpin: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeExecution:
    """Trade execution record"""
    execution_id: str
    intent_id: str
    timestamp: float
    asset: str
    side: str
    quantity: float
    price: float
    fee: float
    fee_currency: str = "USD"
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    exchange_order_id: str = ""
    status: str = "FILLED"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric snapshot"""
    timestamp: float
    equity: float
    cash: float
    positions: Dict[str, float]
    daily_pnl: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_fees: float


@dataclass
class SystemEvent:
    """System event record"""
    event_id: str
    timestamp: float
    severity: str
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SQLITE TRADE LOGGER
# =============================================================================

class TradeLoggerDB:
    """
    SQLite-based trade logging with hot-restart support.

    Provides persistent storage for:
    - Trade intents and executions
    - Order lifecycle
    - Performance metrics
    - System events
    - State snapshots for recovery
    """

    DEFAULT_DB_PATH = "hmats_trades.db"

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS trade_intents (
        intent_id TEXT PRIMARY KEY,
        timestamp REAL NOT NULL,
        asset TEXT NOT NULL,
        side TEXT NOT NULL,
        target_quantity REAL NOT NULL,
        target_price REAL NOT NULL,
        signal_source TEXT,
        signal_strength REAL,
        regime TEXT,
        vpin REAL,
        metadata TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS trade_executions (
        execution_id TEXT PRIMARY KEY,
        intent_id TEXT,
        timestamp REAL NOT NULL,
        asset TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        fee REAL DEFAULT 0,
        fee_currency TEXT,
        slippage_bps REAL,
        latency_ms REAL,
        exchange_order_id TEXT,
        status TEXT,
        metadata TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (intent_id) REFERENCES trade_intents(intent_id)
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        intent_id TEXT,
        timestamp REAL NOT NULL,
        asset TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL,
        status TEXT NOT NULL,
        exchange_order_id TEXT,
        filled_quantity REAL DEFAULT 0,
        avg_fill_price REAL,
        total_fee REAL DEFAULT 0,
        updated_at REAL,
        metadata TEXT,
        FOREIGN KEY (intent_id) REFERENCES trade_intents(intent_id)
    );

    CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        equity REAL NOT NULL,
        cash REAL NOT NULL,
        positions TEXT,
        daily_pnl REAL,
        total_pnl REAL,
        sharpe_ratio REAL,
        max_drawdown REAL,
        win_rate REAL,
        total_trades INTEGER,
        total_fees REAL
    );

    CREATE TABLE IF NOT EXISTS system_events (
        event_id TEXT PRIMARY KEY,
        timestamp REAL NOT NULL,
        severity TEXT NOT NULL,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT
    );

    CREATE TABLE IF NOT EXISTS system_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_intents_timestamp ON trade_intents(timestamp);
    CREATE INDEX IF NOT EXISTS idx_intents_asset ON trade_intents(asset);
    CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON trade_executions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_executions_asset ON trade_executions(asset);
    CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events(severity);
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialize_db()
        logger.info(f"TradeLoggerDB initialized: {self.db_path}")

    def _initialize_db(self):
        with self._lock:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()

    def _generate_id(self, prefix: str = "") -> str:
        import uuid
        return f"{prefix}{uuid.uuid4().hex[:12]}"

    def log_intent(self, intent: TradeIntent) -> str:
        with self._lock:
            self._conn.execute("""
                INSERT INTO trade_intents
                (intent_id, timestamp, asset, side, target_quantity, target_price,
                 signal_source, signal_strength, regime, vpin, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intent.intent_id, intent.timestamp, intent.asset, intent.side,
                intent.target_quantity, intent.target_price, intent.signal_source,
                intent.signal_strength, intent.regime, intent.vpin,
                json.dumps(intent.metadata)
            ))
            self._conn.commit()
        logger.debug(f"Logged intent: {intent.intent_id}")
        return intent.intent_id

    def get_intent(self, intent_id: str) -> Optional[Dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM trade_intents WHERE intent_id = ?", (intent_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['metadata'] = json.loads(result['metadata'] or '{}')
                return result
        return None

    def log_execution(self, execution: TradeExecution) -> str:
        with self._lock:
            self._conn.execute("""
                INSERT INTO trade_executions
                (execution_id, intent_id, timestamp, asset, side, quantity, price,
                 fee, fee_currency, slippage_bps, latency_ms, exchange_order_id,
                 status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution.execution_id, execution.intent_id, execution.timestamp,
                execution.asset, execution.side, execution.quantity, execution.price,
                execution.fee, execution.fee_currency, execution.slippage_bps,
                execution.latency_ms, execution.exchange_order_id, execution.status,
                json.dumps(execution.metadata)
            ))
            self._conn.commit()
        logger.debug(f"Logged execution: {execution.execution_id}")
        return execution.execution_id

    def get_executions(self, intent_id=None, asset=None, start_time=None, end_time=None, limit=100):
        query = "SELECT * FROM trade_executions WHERE 1=1"
        params = []
        if intent_id:
            query += " AND intent_id = ?"; params.append(intent_id)
        if asset:
            query += " AND asset = ?"; params.append(asset)
        if start_time:
            query += " AND timestamp >= ?"; params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"; params.append(end_time)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        with self._lock:
            cursor = self._conn.execute(query, params)
            return [dict(r) | {'metadata': json.loads(dict(r).get('metadata') or '{}')} for r in cursor.fetchall()]

    def log_metrics(self, metrics: PerformanceMetric):
        with self._lock:
            self._conn.execute("""
                INSERT INTO performance_metrics
                (timestamp, equity, cash, positions, daily_pnl, total_pnl,
                 sharpe_ratio, max_drawdown, win_rate, total_trades, total_fees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.equity, metrics.cash,
                json.dumps(metrics.positions), metrics.daily_pnl, metrics.total_pnl,
                metrics.sharpe_ratio, metrics.max_drawdown, metrics.win_rate,
                metrics.total_trades, metrics.total_fees
            ))
            self._conn.commit()

    def get_latest_metrics(self) -> Optional[Dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['positions'] = json.loads(result['positions'] or '{}')
                return result
        return None

    def log_event(self, event: SystemEvent) -> str:
        with self._lock:
            self._conn.execute("""
                INSERT INTO system_events
                (event_id, timestamp, severity, category, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.timestamp, event.severity,
                event.category, event.message, json.dumps(event.details)
            ))
            self._conn.commit()
        return event.event_id

    def get_recent_events(self, severity=None, category=None, limit=100):
        query = "SELECT * FROM system_events WHERE 1=1"
        params = []
        if severity:
            query += " AND severity = ?"; params.append(severity)
        if category:
            query += " AND category = ?"; params.append(category)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        with self._lock:
            cursor = self._conn.execute(query, params)
            return [dict(r) | {'details': json.loads(dict(r).get('details') or '{}')} for r in cursor.fetchall()]

    def save_state(self, key: str, value: Any):
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), time.time()))
            self._conn.commit()

    def load_state(self, key: str) -> Optional[Any]:
        with self._lock:
            cursor = self._conn.execute("SELECT value FROM system_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def load_all_state(self) -> Dict[str, Any]:
        with self._lock:
            cursor = self._conn.execute("SELECT key, value FROM system_state")
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def get_trade_summary(self, start_time=None, end_time=None) -> Dict[str, Any]:
        if start_time is None:
            start_time = time.time() - 86400
        if end_time is None:
            end_time = time.time()
        with self._lock:
            cursor = self._conn.execute("""
                SELECT COUNT(*) as total_trades, SUM(quantity * price) as total_volume,
                       SUM(fee) as total_fees, AVG(slippage_bps) as avg_slippage,
                       AVG(latency_ms) as avg_latency
                FROM trade_executions WHERE timestamp BETWEEN ? AND ?
            """, (start_time, end_time))
            row = cursor.fetchone()
            cursor = self._conn.execute("""
                SELECT asset, COUNT(*) as trades, SUM(quantity * price) as volume, SUM(fee) as fees
                FROM trade_executions WHERE timestamp BETWEEN ? AND ? GROUP BY asset
            """, (start_time, end_time))
            by_asset = {r['asset']: dict(r) for r in cursor.fetchall()}
            return {
                'period_start': start_time, 'period_end': end_time,
                'total_trades': row['total_trades'] or 0,
                'total_volume': row['total_volume'] or 0,
                'total_fees': row['total_fees'] or 0,
                'avg_slippage_bps': row['avg_slippage'] or 0,
                'avg_latency_ms': row['avg_latency'] or 0,
                'by_asset': by_asset,
            }

    def close(self):
        if self._conn:
            self._conn.close()


# =============================================================================
# DISCORD WEBHOOK NOTIFIER
# =============================================================================

class DiscordNotifier:
    """
    Discord webhook integration for real-time alerts.

    Uses Discord embeds for rich formatting with color-coded severity.
    No external library required - stdlib urllib only.

    Features:
    - Async message queue with background worker thread
    - Rate limiting (30 msgs/min, Discord webhook limit)
    - Color-coded embeds by severity
    - Daily summary reports
    - Circuit breaker / hardware alerts
    """

    RATE_LIMIT_MSGS_PER_MIN = 30

    # Discord embed colors (decimal)
    SEVERITY_COLOR = {
        AlertSeverity.DEBUG:    0x95A5A6,  # grey
        AlertSeverity.INFO:     0x3498DB,  # blue
        AlertSeverity.WARNING:  0xF39C12,  # amber
        AlertSeverity.CRITICAL: 0xE74C3C,  # red
    }

    SEVERITY_EMOJI = {
        AlertSeverity.DEBUG:    "🔍",
        AlertSeverity.INFO:     "ℹ️",
        AlertSeverity.WARNING:  "[WARN]️",
        AlertSeverity.CRITICAL: "🚨",
    }

    CATEGORY_EMOJI = {
        AlertCategory.SYSTEM:      "🖥️",
        AlertCategory.TRADING:     "📈",
        AlertCategory.RISK:        "🛡️",
        AlertCategory.PERFORMANCE: "📊",
        AlertCategory.HARDWARE:    "🔧",
    }

    def __init__(
        self,
        webhook_url: str = None,
        username: str = "HMATS",
        enable_async: bool = True,
    ):
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        self.username = username
        self.enable_async = enable_async

        self._message_queue: queue.Queue = queue.Queue()
        self._message_timestamps: deque = deque(maxlen=100)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # [P24 2026-04-24] Circuit breaker: previously, a permanently-dead
        # webhook (revoked URL → 401/403) would generate an error log per
        # queued message forever. Now we suppress sends after consecutive
        # failures and re-probe after a cooldown.
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0  # epoch seconds
        self._circuit_permanently_disabled = False

        self.min_severity = AlertSeverity.INFO

        if self.webhook_url:
            logger.info("DiscordNotifier initialized")
        else:
            logger.warning("Discord not configured (set DISCORD_WEBHOOK_URL)")

    def start(self):
        if not self.enable_async or not self.webhook_url:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._message_worker, name="DiscordWorker", daemon=True,
        )
        self._worker_thread.start()

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def _message_worker(self):
        while self._running:
            try:
                try:
                    payload = self._message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                self._wait_for_rate_limit()
                self._post_webhook(payload)
            except Exception as e:
                logger.error(f"Discord worker error: {e}")

    def _wait_for_rate_limit(self):
        now = time.time()
        cutoff = now - 60
        with self._lock:
            while self._message_timestamps and self._message_timestamps[0] < cutoff:
                self._message_timestamps.popleft()
            if len(self._message_timestamps) >= self.RATE_LIMIT_MSGS_PER_MIN:
                wait = self._message_timestamps[0] + 60 - now
                if wait > 0:
                    logger.debug(f"Discord rate limited, waiting {wait:.1f}s")
                    time.sleep(wait)

    def _post_webhook(self, payload: dict):
        if not self.webhook_url:
            return
        # [P24 2026-04-24] Circuit breaker check.
        if self._circuit_permanently_disabled:
            return
        now = time.time()
        if now < self._circuit_open_until:
            return  # circuit open, drop silently (cooldown in effect)

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "HMATS/6.8 (https://github.com)"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 204):
                    logger.warning(f"Discord webhook returned {resp.status}")
            with self._lock:
                self._message_timestamps.append(time.time())
            # Success → reset circuit.
            if self._consecutive_failures > 0:
                logger.info(
                    f"[DISCORD] webhook recovered after {self._consecutive_failures} failures"
                )
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            logger.error(f"Discord webhook HTTP {e.code}: {body}")
            self._consecutive_failures += 1
            if e.code in (401, 403, 404):
                # Auth error / revoked webhook — permanent disable.
                self._circuit_permanently_disabled = True
                logger.error(
                    f"[DISCORD] HTTP {e.code} — webhook permanently disabled "
                    f"(URL likely revoked or wrong)"
                )
            elif e.code == 429:
                # Rate limit — honor Retry-After if present, else 60s.
                retry_after = 60.0
                try:
                    raw = e.headers.get("Retry-After", "") if e.headers else ""
                    if raw:
                        retry_after = min(300.0, max(1.0, float(raw)))
                except (ValueError, TypeError):
                    pass
                self._circuit_open_until = time.time() + retry_after
                logger.warning(
                    f"[DISCORD] 429 rate limit, circuit open {retry_after:.0f}s"
                )
            else:
                self._maybe_open_circuit()
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            self._consecutive_failures += 1
            self._maybe_open_circuit()

    def _maybe_open_circuit(self):
        """Exponential backoff: 5 consecutive failures → 60s cooldown,
        10 → 300s, 15+ → 1800s. Reset on first success."""
        n = self._consecutive_failures
        if n >= 15:
            cooldown = 1800.0
        elif n >= 10:
            cooldown = 300.0
        elif n >= 5:
            cooldown = 60.0
        else:
            return
        self._circuit_open_until = time.time() + cooldown
        logger.warning(
            f"[DISCORD] circuit open {cooldown:.0f}s "
            f"after {n} consecutive failures"
        )

    def _enqueue_or_send(self, payload: dict):
        if self.enable_async and self._running:
            self._message_queue.put(payload)
        else:
            self._post_webhook(payload)

    def _build_embed(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        message: str,
        details: Dict[str, Any] = None,
    ) -> dict:
        sev_emoji = self.SEVERITY_EMOJI.get(severity, "")
        cat_emoji = self.CATEGORY_EMOJI.get(category, "")

        embed: Dict[str, Any] = {
            "title": f"{sev_emoji} {severity.name} - {cat_emoji} {category.name}",
            "description": message,
            "color": self.SEVERITY_COLOR.get(severity, 0x95A5A6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if details:
            embed["fields"] = [
                {"name": str(k), "value": str(v), "inline": True}
                for k, v in details.items()
            ]

        return {
            "username": self.username,
            "embeds": [embed],
        }

    def send_alert(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        message: str,
        details: Dict[str, Any] = None,
    ):
        if severity.value < self.min_severity.value:
            return
        if not self.webhook_url:
            return
        payload = self._build_embed(severity, category, message, details)
        self._enqueue_or_send(payload)

    def send_trade_alert(
        self,
        asset: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        slippage_bps: float,
    ):
        side_emoji = "🟢" if side == "BUY" else "🔴"
        self.send_alert(
            severity=AlertSeverity.INFO,
            category=AlertCategory.TRADING,
            message=f"{side_emoji} **{side}** {quantity:.6f} **{asset}**",
            details={
                "Price": f"${price:,.2f}",
                "Fee": f"${fee:.4f}",
                "Slippage": f"{slippage_bps:.2f} bps",
            },
        )

    def send_daily_summary(self, summary: Dict[str, Any]):
        if not self.webhook_url:
            return

        embed = {
            "title": "📊 Daily Summary",
            "color": 0x2ECC71 if summary.get("daily_pnl", 0) >= 0 else 0xE74C3C,
            "fields": [
                {"name": "💰 Equity",    "value": f"${summary.get('equity', 0):,.2f}",           "inline": True},
                {"name": "📈 Daily P&L", "value": f"${summary.get('daily_pnl', 0):+,.2f}",      "inline": True},
                {"name": "📈 Total P&L", "value": f"${summary.get('total_pnl', 0):+,.2f}",      "inline": True},
                {"name": "🔄 Trades",    "value": str(summary.get("total_trades", 0)),           "inline": True},
                {"name": "💸 Volume",    "value": f"${summary.get('volume', 0):,.2f}",           "inline": True},
                {"name": "💸 Fees",      "value": f"${summary.get('total_fees', 0):.2f}",        "inline": True},
                {"name": "📉 Sharpe",    "value": f"{summary.get('sharpe_ratio', 0):.2f}",       "inline": True},
                {"name": "📉 Max DD",    "value": f"{summary.get('max_drawdown', 0):.2%}",       "inline": True},
                {"name": "🎯 Win Rate",  "value": f"{summary.get('win_rate', 0):.1%}",           "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"username": self.username, "embeds": [embed]}
        self._enqueue_or_send(payload)

    def send_circuit_breaker_alert(self, reason: str, details: Dict = None):
        self.send_alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            message=f"🛑 **CIRCUIT BREAKER TRIPPED**\n{reason}",
            details=details,
        )

    def send_hardware_alert(self, component: str, metric: str, value: float, threshold: float):
        self.send_alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.HARDWARE,
            message=f"**{component}** {metric} exceeded threshold",
            details={"Current": f"{value:.1f}", "Threshold": f"{threshold:.1f}"},
        )


# =============================================================================
# DISCORD LOG HANDLER — forward ERROR/CRITICAL to Discord
# =============================================================================

class DiscordLogHandler(logging.Handler):
    """Forward ERROR/CRITICAL logs to Discord webhook with 5-min dedup."""

    _DEDUP_WINDOW = 300

    def __init__(self, notifier: 'DiscordNotifier', min_level=logging.ERROR):
        super().__init__(level=min_level)
        self._notifier = notifier
        self._dedup: Dict[str, float] = {}

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Dedup by call site (pathname:lineno:levelno), not by message text.
            # Tick-loop except blocks like "SOTA integration error: {e}" produce
            # messages whose tails vary per tick (embedded object reprs,
            # timestamps, per-asset identifiers). Keying on message would let
            # the same recurring crash spam Discord every tick. The log file
            # still has the full message detail.
            _key = f"{record.pathname}:{record.lineno}:{record.levelno}"
            _now = time.time()
            if _key in self._dedup and _now - self._dedup[_key] < self._DEDUP_WINDOW:
                return
            self._dedup[_key] = _now
            if len(self._dedup) > 200:
                cutoff = _now - self._DEDUP_WINDOW
                self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}
            severity = AlertSeverity.CRITICAL if record.levelno >= logging.CRITICAL else AlertSeverity.WARNING
            self._notifier.send_alert(severity, AlertCategory.SYSTEM, msg[:1900])
        except Exception:
            pass


# =============================================================================
# UNIFIED PERSISTENCE & ALERTING MANAGER
# =============================================================================

class PersistenceAndAlertingManager:
    """
    Unified manager for persistence and alerting.

    Combines:
    - SQLite trade logging
    - Discord webhook notifications
    - State recovery
    - Audit trail
    """

    def __init__(
        self,
        db_path: str = None,
        discord_webhook_url: str = None,
    ):
        self.db = TradeLoggerDB(db_path)
        self.discord = DiscordNotifier(webhook_url=discord_webhook_url)

        # Backward compat alias - code referencing .telegram still works
        self.telegram = self.discord

        self._event_handlers: List[Callable[[SystemEvent], None]] = []
        logger.info("PersistenceAndAlertingManager initialized (Discord)")

    def start(self):
        self.discord.start()
        self.log_event(AlertSeverity.INFO, AlertCategory.SYSTEM, "HMATS started")

    def stop(self):
        self.log_event(AlertSeverity.INFO, AlertCategory.SYSTEM, "HMATS stopped")
        self.discord.stop()
        self.db.close()

    def add_event_handler(self, handler: Callable[[SystemEvent], None]):
        self._event_handlers.append(handler)

    def log_event(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        message: str,
        details: Dict[str, Any] = None,
        send_notification: bool = True,
    ) -> str:
        event = SystemEvent(
            event_id=f"evt_{int(time.time()*1000)}_{hash(message) % 10000:04d}",
            timestamp=time.time(),
            severity=severity.name,
            category=category.name,
            message=message,
            details=details or {},
        )
        self.db.log_event(event)
        if send_notification:
            self.discord.send_alert(severity, category, message, details)
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        return event.event_id

    def log_trade_intent(self, intent: TradeIntent) -> str:
        return self.db.log_intent(intent)

    def log_trade_execution(
        self,
        execution: TradeExecution,
        send_notification: bool = True,
    ) -> str:
        execution_id = self.db.log_execution(execution)
        if send_notification:
            self.discord.send_trade_alert(
                asset=execution.asset, side=execution.side,
                quantity=execution.quantity, price=execution.price,
                fee=execution.fee, slippage_bps=execution.slippage_bps,
            )
        return execution_id

    def log_metrics(self, metrics: PerformanceMetric):
        self.db.log_metrics(metrics)

    def save_state(self, key: str, value: Any):
        self.db.save_state(key, value)

    def load_state(self, key: str) -> Optional[Any]:
        return self.db.load_state(key)

    def recover_state(self) -> Dict[str, Any]:
        state = self.db.load_all_state()
        self.log_event(
            AlertSeverity.INFO, AlertCategory.SYSTEM,
            "State recovered from database",
            details={"keys": list(state.keys())},
        )
        return state

    def get_trade_summary(self, hours: float = 24) -> Dict[str, Any]:
        start_time = time.time() - hours * 3600
        return self.db.get_trade_summary(start_time)

    def send_daily_summary(self):
        summary = self.get_trade_summary(24)
        metrics = self.db.get_latest_metrics()
        if metrics:
            summary.update({
                "equity": metrics["equity"],
                "daily_pnl": metrics["daily_pnl"],
                "total_pnl": metrics["total_pnl"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
            })
        self.discord.send_daily_summary(summary)
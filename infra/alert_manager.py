"""
================================================================================
HMATS v5.2.0 - User Alert System
================================================================================
Purpose: Multi-channel alert system for critical trading events

Features:
1. Console alerts (always on)
2. Email notifications (SMTP)
3. Desktop popup notifications (optional)
4. Alert history and deduplication
5. Alert severity levels and throttling

Supported Events:
- Kill switch activation
- Drawdown breach
- Regime change
- Prolonged inactivity
- System errors
- Trade execution alerts

Author: HMATS v5.2 Upgrade
Version: 5.2.0
================================================================================
"""

import logging
import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


class AlertChannel(Enum):
    """Available alert channels."""
    CONSOLE = "console"
    EMAIL = "email"
    POPUP = "popup"
    WEBHOOK = "webhook"
    FILE = "file"


class AlertType(Enum):
    """Types of system alerts."""
    # Risk alerts
    KILL_SWITCH = "kill_switch"
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_BREACH = "drawdown_breach"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CORRELATION_CRISIS = "correlation_crisis"
    
    # Regime alerts
    REGIME_CHANGE = "regime_change"
    PROLONGED_INACTIVITY = "prolonged_inactivity"
    NO_TRADE_MODE = "no_trade_mode"
    
    # Execution alerts
    TRADE_EXECUTED = "trade_executed"
    TRADE_BLOCKED = "trade_blocked"
    ORDER_REJECTED = "order_rejected"
    SLIPPAGE_HIGH = "slippage_high"
    
    # System alerts
    SYSTEM_ERROR = "system_error"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    LATENCY_SPIKE = "latency_spike"
    
    # Position alerts
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_HIT = "stop_loss_hit"


@dataclass
class AlertConfig:
    """Alert system configuration."""
    # Email settings
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Throttling
    min_interval_seconds: Dict[str, int] = field(default_factory=lambda: {
        "kill_switch": 0,           # Always send immediately
        "drawdown_breach": 60,      # Max once per minute
        "regime_change": 300,       # Max once per 5 minutes
        "trade_executed": 10,       # Max once per 10 seconds
        "default": 30               # Default: 30 seconds
    })
    
    # Severity routing
    email_min_severity: AlertSeverity = AlertSeverity.ERROR
    popup_min_severity: AlertSeverity = AlertSeverity.WARNING
    
    # Feature flags
    enable_email: bool = False
    enable_popup: bool = True
    enable_console: bool = True
    enable_file: bool = True
    
    # File logging
    alert_log_file: str = "logs/alerts.jsonl"
    max_history: int = 1000


@dataclass
class Alert:
    """A single alert."""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    channels_sent: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.alert_type.value,
            "severity": self.severity.name,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "channels_sent": [c.value for c in self.channels_sent],
            "acknowledged": self.acknowledged
        }
    
    def format_console(self) -> str:
        """Format for console output."""
        severity_icons = {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "[WARN]️",
            AlertSeverity.ERROR: "OFF",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        icon = severity_icons.get(self.severity, "•")
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"{icon} [{time_str}] {self.severity.name}: {self.title}\n   {self.message}"
    
    def format_email_html(self) -> str:
        """Format as HTML email."""
        severity_colors = {
            AlertSeverity.DEBUG: "#6c757d",
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.EMERGENCY: "#721c24"
        }
        color = severity_colors.get(self.severity, "#000000")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px;">
                <h2 style="margin: 0;">{self.title}</h2>
                <p style="margin: 5px 0 0 0;">{self.severity.name} | {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
                <p style="font-size: 16px;">{self.message}</p>
            </div>
        """
        
        if self.data:
            html += """
            <div style="padding: 15px; background-color: #e9ecef; border-radius: 5px; margin-top: 10px;">
                <h4>Additional Data:</h4>
                <pre style="font-size: 12px;">"""
            html += json.dumps(self.data, indent=2)
            html += "</pre></div>"
        
        html += """
            <div style="margin-top: 20px; color: #6c757d; font-size: 12px;">
                <p>HMATS Trading System - Automated Alert</p>
            </div>
        </body>
        </html>
        """
        return html


class AlertManager:
    """
    Central alert management system.
    
    Features:
    - Multi-channel dispatch (console, email, popup)
    - Alert throttling and deduplication
    - Severity-based routing
    - History tracking
    """
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        
        # Alert history
        self.history: deque = deque(maxlen=self.config.max_history)
        self.last_sent: Dict[str, datetime] = {}
        
        # Alert queue for async processing
        self.alert_queue: queue.Queue = queue.Queue()
        
        # Callbacks for custom handlers
        self._callbacks: List[Callable[[Alert], None]] = []
        
        # Start background worker
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        
        # Counter for alert IDs
        self._alert_counter = 0
        
        logger.info(f"[ALERT] AlertManager initialized. Email={self.config.enable_email}, "
                   f"Popup={self.config.enable_popup}")
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """Register custom alert handler."""
        self._callbacks.append(callback)
    
    # =========================================================================
    # ALERT CREATION
    # =========================================================================
    
    def send_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        data: Dict = None,
        force: bool = False
    ) -> Optional[Alert]:
        """
        Send an alert through configured channels.
        
        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            severity: Severity level
            data: Additional data to include
            force: Skip throttling
            
        Returns:
            Alert object if sent, None if throttled
        """
        # Check throttling
        if not force and not self._should_send(alert_type):
            logger.debug(f"[ALERT] Throttled: {alert_type.value}")
            return None
        
        # Create alert
        self._alert_counter += 1
        alert = Alert(
            id=f"ALERT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._alert_counter}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc),
            data=data or {}
        )
        
        # Update throttle tracker
        self.last_sent[alert_type.value] = datetime.now(timezone.utc)
        
        # Add to queue for async processing
        self.alert_queue.put(alert)
        
        # Add to history
        self.history.append(alert)
        
        return alert
    
    def _should_send(self, alert_type: AlertType) -> bool:
        """Check if alert should be sent (throttling)."""
        type_key = alert_type.value
        last_time = self.last_sent.get(type_key)
        
        if last_time is None:
            return True
        
        min_interval = self.config.min_interval_seconds.get(
            type_key,
            self.config.min_interval_seconds.get("default", 30)
        )
        
        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        return elapsed >= min_interval
    
    # =========================================================================
    # ASYNC PROCESSING
    # =========================================================================
    
    def _process_queue(self):
        """Background worker to process alert queue."""
        while self._running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                self._dispatch_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[ALERT] Queue processing error: {e}")
    
    def _dispatch_alert(self, alert: Alert):
        """Dispatch alert to all configured channels."""
        # Console (always)
        if self.config.enable_console:
            self._send_console(alert)
        
        # File log
        if self.config.enable_file:
            self._send_file(alert)
        
        # Email (severity check)
        if (self.config.enable_email and 
            alert.severity.value >= self.config.email_min_severity.value):
            self._send_email(alert)
        
        # Popup (severity check)
        if (self.config.enable_popup and 
            alert.severity.value >= self.config.popup_min_severity.value):
            self._send_popup(alert)
        
        # Custom callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"[ALERT] Callback error: {e}")
    
    # =========================================================================
    # CHANNEL IMPLEMENTATIONS
    # =========================================================================
    
    def _send_console(self, alert: Alert):
        """Send alert to console."""
        print("\n" + "=" * 60)
        print(alert.format_console())
        print("=" * 60 + "\n")
        alert.channels_sent.append(AlertChannel.CONSOLE)
    
    def _send_file(self, alert: Alert):
        """Log alert to file."""
        try:
            log_dir = os.path.dirname(self.config.alert_log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            with open(self.config.alert_log_file, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
            
            alert.channels_sent.append(AlertChannel.FILE)
        except Exception as e:
            logger.error(f"[ALERT] File logging error: {e}")
    
    def _send_email(self, alert: Alert):
        """Send alert via email."""
        if not self.config.smtp_host or not self.config.email_to:
            logger.debug("[ALERT] Email not configured")
            return
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[HMATS {alert.severity.name}] {alert.title}"
            msg['From'] = self.config.email_from
            msg['To'] = ", ".join(self.config.email_to)
            
            # Plain text version
            text_part = MIMEText(f"{alert.title}\n\n{alert.message}", 'plain')
            msg.attach(text_part)
            
            # HTML version
            html_part = MIMEText(alert.format_email_html(), 'html')
            msg.attach(html_part)
            
            # Send
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(
                    self.config.email_from,
                    self.config.email_to,
                    msg.as_string()
                )
            
            alert.channels_sent.append(AlertChannel.EMAIL)
            logger.info(f"[ALERT] Email sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"[ALERT] Email error: {e}")
    
    def _send_popup(self, alert: Alert):
        """Send desktop popup notification."""
        try:
            # Try different notification methods based on platform
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f"""
                    osascript -e 'display notification "{alert.message}" with title "HMATS: {alert.title}"'
                """)
                alert.channels_sent.append(AlertChannel.POPUP)
                
            elif system == "Linux":
                # Try notify-send
                os.system(f"""
                    notify-send "HMATS: {alert.title}" "{alert.message}" -u critical
                """)
                alert.channels_sent.append(AlertChannel.POPUP)
                
            elif system == "Windows":
                # Try Windows toast notification
                try:
                    from win10toast import ToastNotifier
                    toaster = ToastNotifier()
                    toaster.show_toast(
                        f"HMATS: {alert.title}",
                        alert.message,
                        duration=10
                    )
                    alert.channels_sent.append(AlertChannel.POPUP)
                except ImportError:
                    logger.debug("[ALERT] win10toast not installed")
                    
        except Exception as e:
            logger.debug(f"[ALERT] Popup notification error: {e}")
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def alert_kill_switch(self, reason: str, data: Dict = None):
        """Alert for kill switch activation."""
        self.send_alert(
            AlertType.KILL_SWITCH,
            "🚨 KILL SWITCH ACTIVATED",
            f"Trading has been halted. Reason: {reason}",
            AlertSeverity.EMERGENCY,
            data,
            force=True
        )
    
    def alert_drawdown(self, current_dd: float, threshold: float, data: Dict = None):
        """Alert for drawdown breach."""
        severity = AlertSeverity.CRITICAL if current_dd >= threshold else AlertSeverity.WARNING
        alert_type = AlertType.DRAWDOWN_BREACH if current_dd >= threshold else AlertType.DRAWDOWN_WARNING
        
        self.send_alert(
            alert_type,
            f"Drawdown Alert: {current_dd:.1%}",
            f"Current drawdown {current_dd:.1%} has {'exceeded' if current_dd >= threshold else 'approached'} "
            f"threshold {threshold:.1%}",
            severity,
            data
        )
    
    def alert_regime_change(self, old_regime: str, new_regime: str, data: Dict = None):
        """Alert for regime change."""
        self.send_alert(
            AlertType.REGIME_CHANGE,
            f"Regime Change: {old_regime} -> {new_regime}",
            f"Market regime has changed from {old_regime} to {new_regime}",
            AlertSeverity.WARNING,
            data
        )
    
    def alert_inactivity(self, hours_inactive: float, data: Dict = None):
        """Alert for prolonged inactivity."""
        self.send_alert(
            AlertType.PROLONGED_INACTIVITY,
            f"System Inactive for {hours_inactive:.1f} hours",
            f"No trades have been executed for {hours_inactive:.1f} hours. "
            "This may indicate market conditions unsuitable for trading or a system issue.",
            AlertSeverity.WARNING,
            data
        )
    
    def alert_trade(self, symbol: str, side: str, size: float, price: float, data: Dict = None):
        """Alert for trade execution."""
        self.send_alert(
            AlertType.TRADE_EXECUTED,
            f"Trade: {side.upper()} {size} {symbol} @ {price}",
            f"Executed {side.upper()} {size} {symbol} at {price}",
            AlertSeverity.INFO,
            data
        )
    
    def alert_error(self, error: str, data: Dict = None):
        """Alert for system error."""
        self.send_alert(
            AlertType.SYSTEM_ERROR,
            "System Error",
            error,
            AlertSeverity.ERROR,
            data
        )
    
    # =========================================================================
    # HISTORY & STATUS
    # =========================================================================
    
    def get_recent_alerts(self, count: int = 10, severity_min: AlertSeverity = None) -> List[Dict]:
        """Get recent alerts."""
        alerts = list(self.history)
        
        if severity_min:
            alerts = [a for a in alerts if a.severity.value >= severity_min.value]
        
        return [a.to_dict() for a in alerts[-count:]]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark alert as acknowledged."""
        for alert in self.history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def shutdown(self):
        """Shutdown alert manager."""
        self._running = False
        self._worker_thread.join(timeout=2.0)
        logger.info("[ALERT] AlertManager shutdown")


# =============================================================================
# SINGLETON
# =============================================================================

_alert_manager: Optional[AlertManager] = None


def get_alert_manager(config: AlertConfig = None) -> AlertManager:
    """Get or create the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(config)
    return _alert_manager


def reset_alert_manager():
    """Reset the global alert manager."""
    global _alert_manager
    if _alert_manager:
        _alert_manager.shutdown()
    _alert_manager = None


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def alert(
    alert_type: AlertType,
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    data: Dict = None
) -> Optional[Alert]:
    """Quick function to send an alert."""
    return get_alert_manager().send_alert(alert_type, title, message, severity, data)


def alert_critical(title: str, message: str, data: Dict = None):
    """Send a critical alert."""
    return alert(AlertType.SYSTEM_ERROR, title, message, AlertSeverity.CRITICAL, data)


def alert_warning(title: str, message: str, data: Dict = None):
    """Send a warning alert."""
    return alert(AlertType.SYSTEM_ERROR, title, message, AlertSeverity.WARNING, data)

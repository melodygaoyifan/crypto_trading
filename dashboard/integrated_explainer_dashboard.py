"""
HMATS integrated explainer dashboard.

This module provides a light-weight decision explainer, alert center,
and Streamlit UI wiring for interactive monitoring.
"""

import json
import logging
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DecisionRecord:
    """Decision record for dashboard and explainer."""

    timestamp: datetime
    decision_id: str
    action: str
    symbol: str
    direction: float
    confidence: float
    mode: str
    regime: str
    signal_components: Dict[str, float]
    risk_checks: Dict[str, bool]
    vetoes: List[str]
    executed: bool = False
    execution_details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    explanation_detailed: str = ""


@dataclass
class AlertRecord:
    """Alert record used in dashboard display."""

    timestamp: datetime
    alert_id: str
    severity: AlertSeverity
    source: str
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class DecisionExplainer:
    """Rule-based explainer with lightweight Q&A support."""

    ACTION_TEMPLATES = {
        "BUY": "System decided to BUY {symbol}",
        "SELL": "System decided to SELL {symbol}",
        "HOLD": "System decided to HOLD current position on {symbol}",
        "NO_TRADE": "System decided not to open a new trade on {symbol}",
        "EXIT": "System decided to EXIT {symbol}",
    }

    MODE_EXPLANATIONS = {
        "OPPORTUNITY": "Mode=OPPORTUNITY, entry conditions can be more aggressive",
        "NORMAL": "Mode=NORMAL, standard risk and execution gates apply",
        "NO_TRADE": "Mode=NO_TRADE, new entries are blocked",
    }

    REGIME_EXPLANATIONS = {
        "bull": "Market regime suggests upside pressure",
        "bear": "Market regime suggests downside pressure",
        "volatile": "Market regime is volatile and risk limits are tighter",
        "ranging": "Market regime is range-bound",
    }

    def __init__(self):
        self.history: deque[DecisionRecord] = deque(maxlen=200)

    def explain_decision(self, record: DecisionRecord) -> str:
        """Generate one-line explanation."""
        parts: List[str] = []

        action_text = self.ACTION_TEMPLATES.get(
            record.action,
            f"System executed action={record.action} on {record.symbol}",
        ).format(symbol=record.symbol)
        parts.append(action_text)

        mode_text = self.MODE_EXPLANATIONS.get(record.mode, "")
        if mode_text:
            parts.append(mode_text)

        regime_text = self.REGIME_EXPLANATIONS.get(record.regime, "")
        if regime_text:
            parts.append(regime_text)

        signal_text = self._explain_signals(record.signal_components)
        if signal_text:
            parts.append(signal_text)

        conf_pct = record.confidence * 100.0
        if conf_pct >= 80:
            parts.append(f"confidence is high ({conf_pct:.0f}%)")
        elif conf_pct >= 60:
            parts.append(f"confidence is medium ({conf_pct:.0f}%)")
        else:
            parts.append(f"confidence is low ({conf_pct:.0f}%)")

        if record.vetoes:
            parts.append(f"vetoes: {', '.join(record.vetoes)}")

        explanation = ". ".join(parts) + "."
        record.explanation = explanation
        self.history.append(record)
        return explanation

    def _explain_signals(self, components: Dict[str, float]) -> str:
        if not components:
            return ""

        sorted_signals = sorted(
            components.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        bullish = [self._translate_signal_name(k) for k, v in sorted_signals if v > 0.1][:3]
        bearish = [self._translate_signal_name(k) for k, v in sorted_signals if v < -0.1][:3]

        parts: List[str] = []
        if bullish:
            parts.append(f"bullish drivers: {', '.join(bullish)}")
        if bearish:
            parts.append(f"bearish drivers: {', '.join(bearish)}")
        return "; ".join(parts)

    @staticmethod
    def _translate_signal_name(name: str) -> str:
        mapping = {
            "trend_following": "trend_following",
            "momentum": "momentum",
            "mean_reversion": "mean_reversion",
            "sentiment": "sentiment",
            "onchain": "onchain",
            "microstructure": "microstructure",
            "quant": "quant",
            "drl": "drl",
        }
        return mapping.get(name, name)

    def explain_detailed(self, record: DecisionRecord) -> str:
        """Generate detailed markdown explanation."""
        rows: List[str] = []
        rows.append(f"# {record.action} {record.symbol} Decision")
        rows.append(f"Time: {record.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        rows.append("")

        rows.append("## Context")
        rows.append(f"- Mode: {record.mode}")
        rows.append(f"- Regime: {record.regime}")
        rows.append(f"- Direction: {record.direction:+.3f}")
        rows.append(f"- Confidence: {record.confidence:.1%}")
        rows.append("")

        rows.append("## Signals")
        if record.signal_components:
            for name, value in sorted(
                record.signal_components.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            ):
                direction = "+" if value > 0 else "-" if value < 0 else "0"
                rows.append(f"- {self._translate_signal_name(name)}: {direction} {value:+.3f}")
        else:
            rows.append("- no component signals")
        rows.append("")

        rows.append("## Risk Checks")
        if record.risk_checks:
            for check_name, passed in record.risk_checks.items():
                status = "PASS" if passed else "FAIL"
                rows.append(f"- {check_name}: {status}")
        else:
            rows.append("- no risk checks")

        if record.vetoes:
            rows.append("")
            rows.append("## Vetoes")
            for veto in record.vetoes:
                rows.append(f"- {veto}")

        if record.executed and record.execution_details:
            rows.append("")
            rows.append("## Execution")
            for key, value in record.execution_details.items():
                rows.append(f"- {key}: {value}")

        record.explanation_detailed = "\n".join(rows)
        return record.explanation_detailed

    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """Answer common dashboard questions."""
        q = (question or "").lower()

        if "why" in q and ("buy" in q or "sell" in q or "trade" in q):
            return self._answer_why_trade(context)
        if "exit" in q or "close" in q:
            return self._answer_why_exit(context)
        if "risk" in q:
            return self._answer_risk_status(context)
        if "performance" in q or "pnl" in q or "profit" in q:
            return self._answer_performance(context)

        return "I can answer questions about trade rationale, risk state, and performance."

    def _answer_why_trade(self, context: Dict[str, Any]) -> str:
        decision = context.get("last_decision")
        if decision:
            return self.explain_decision(decision)
        return "No recent decision is available."

    @staticmethod
    def _answer_why_exit(context: Dict[str, Any]) -> str:
        _ = context
        return (
            "Exit decisions usually come from stop loss, take profit, signal reversal, "
            "or risk limit controls."
        )

    @staticmethod
    def _answer_risk_status(context: Dict[str, Any]) -> str:
        risk_state = context.get("risk_state", {})
        if not risk_state:
            return "Risk system appears normal."

        pieces = [f"state={risk_state.get('state', 'NORMAL')}"]
        if "drawdown" in risk_state:
            pieces.append(f"drawdown={risk_state['drawdown']:.1%}")
        if risk_state.get("alerts"):
            pieces.append(f"active_alerts={len(risk_state['alerts'])}")
        return " | ".join(pieces)

    @staticmethod
    def _answer_performance(context: Dict[str, Any]) -> str:
        perf = context.get("performance", {})
        if not perf:
            return "Performance data is not available yet."
        return (
            f"daily_pnl={perf.get('daily_pnl', 0):+.2%} | "
            f"win_rate={perf.get('win_rate', 0):.0%} | "
            f"trades={perf.get('trades', 0)}"
        )


class AlertDisplayManager:
    """Alert collector used by the dashboard panel."""

    def __init__(self):
        self.alerts: deque[AlertRecord] = deque(maxlen=100)
        self.unacknowledged_count = 0

    def add_alert(
        self,
        severity: AlertSeverity,
        source: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AlertRecord:
        alert = AlertRecord(
            timestamp=datetime.now(timezone.utc),
            alert_id=f"ALERT_{len(self.alerts)}_{int(datetime.now(timezone.utc).timestamp())}",
            severity=severity,
            source=source,
            title=title,
            message=message,
            data=data or {},
        )
        self.alerts.append(alert)
        self.unacknowledged_count += 1
        return alert

    def acknowledge(self, alert_id: str) -> None:
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                self.unacknowledged_count = max(self.unacknowledged_count - 1, 0)
                return

    def acknowledge_all(self) -> None:
        for alert in self.alerts:
            alert.acknowledged = True
        self.unacknowledged_count = 0

    def get_recent(self, n: int = 10, unacked_only: bool = False) -> List[AlertRecord]:
        alerts = list(self.alerts)
        if unacked_only:
            alerts = [a for a in alerts if not a.acknowledged]
        return alerts[-n:]

    def get_by_severity(self, severity: AlertSeverity) -> List[AlertRecord]:
        return [a for a in self.alerts if a.severity == severity]


class IntegratedDashboardProvider:
    """Provider that combines decisions, alerts, and dashboard Q&A."""

    def __init__(self):
        self.explainer = DecisionExplainer()
        self.alert_manager = AlertDisplayManager()
        self.decision_history: deque[DecisionRecord] = deque(maxlen=200)
        self.qa_history: List[Dict[str, str]] = []
        self.data_sources: Dict[str, Callable[[], Any]] = {}

    def register_data_source(self, name: str, provider: Callable[[], Any]) -> None:
        self.data_sources[name] = provider

    def record_decision(self, decision: DecisionRecord) -> None:
        self.explainer.explain_decision(decision)
        self.explainer.explain_detailed(decision)
        self.decision_history.append(decision)
        self._check_decision_alerts(decision)

    def _check_decision_alerts(self, decision: DecisionRecord) -> None:
        if decision.vetoes:
            self.alert_manager.add_alert(
                AlertSeverity.WARNING,
                "DecisionEngine",
                "Trade vetoed",
                f"{decision.action} {decision.symbol} vetoed: {', '.join(decision.vetoes)}",
                {"decision_id": decision.decision_id},
            )

        if decision.confidence < 0.30 and decision.action in ("BUY", "SELL"):
            self.alert_manager.add_alert(
                AlertSeverity.INFO,
                "DecisionEngine",
                "Low-confidence trade",
                f"{decision.action} {decision.symbol} confidence={decision.confidence:.0%}",
                {"decision_id": decision.decision_id},
            )

    def add_risk_alert(self, title: str, message: str, severity: str = "warning") -> None:
        sev = AlertSeverity.__members__.get(severity.upper(), AlertSeverity.WARNING)
        self.alert_manager.add_alert(sev, "RiskManager", title, message)

    def ask_question(self, question: str) -> str:
        context: Dict[str, Any] = {
            "last_decision": self.decision_history[-1] if self.decision_history else None,
            "decision_count": len(self.decision_history),
            "alert_count": len(self.alert_manager.alerts),
        }
        for name, provider in self.data_sources.items():
            try:
                context[name] = provider()
            except Exception:
                logger.debug("data source failed: %s", name, exc_info=True)

        answer = self.explainer.answer_question(question, context)
        self.qa_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "answer": answer,
            }
        )
        return answer

    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decisions": {
                "total": len(self.decision_history),
                "recent": [
                    {
                        "time": d.timestamp.strftime("%H:%M:%S"),
                        "action": d.action,
                        "symbol": d.symbol,
                        "confidence": d.confidence,
                        "explanation": d.explanation,
                    }
                    for d in list(self.decision_history)[-5:]
                ],
            },
            "alerts": {
                "unacknowledged": self.alert_manager.unacknowledged_count,
                "critical": len(self.alert_manager.get_by_severity(AlertSeverity.CRITICAL)),
                "recent": [
                    {
                        "id": a.alert_id,
                        "severity": a.severity.value,
                        "title": a.title,
                        "message": a.message,
                        "time": a.timestamp.strftime("%H:%M:%S"),
                        "acked": a.acknowledged,
                    }
                    for a in self.alert_manager.get_recent(10)
                ],
            },
            "qa": {
                "history_count": len(self.qa_history),
                "recent": self.qa_history[-5:],
            },
        }


def _rerun() -> None:
    if not STREAMLIT_AVAILABLE:
        return
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_explainer_dashboard() -> None:
    """Render the integrated explainer dashboard page."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available")
        return

    st.set_page_config(page_title="HMATS Explainer", page_icon="HM", layout="wide")
    st.title("HMATS Explainer Center")
    st.caption("Decision explanations, alert triage, and interactive Q&A")

    if "provider" not in st.session_state:
        st.session_state.provider = IntegratedDashboardProvider()
        _add_sample_data(st.session_state.provider)

    provider: IntegratedDashboardProvider = st.session_state.provider

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Recent Decisions")
        decisions = list(provider.decision_history)[-10:]
        if decisions:
            for d in reversed(decisions):
                with st.expander(
                    f"{d.action} {d.symbol} @ {d.timestamp.strftime('%H:%M:%S')} [{d.confidence:.0%}]",
                    expanded=False,
                ):
                    st.markdown(f"**Summary:** {d.explanation}")
                    st.markdown(d.explanation_detailed)
        else:
            st.info("No decision records yet")

        st.subheader("Q&A")
        question = st.text_input("Ask", placeholder="why did we buy BTC?")
        if st.button("Ask") and question.strip():
            st.success(provider.ask_question(question.strip()))

    with col_right:
        st.subheader("Alerts")
        if provider.alert_manager.unacknowledged_count > 0:
            st.error(f"Unacknowledged alerts: {provider.alert_manager.unacknowledged_count}")
        if st.button("Acknowledge all"):
            provider.alert_manager.acknowledge_all()
            _rerun()

        for alert in reversed(provider.alert_manager.get_recent(10)):
            badge = alert.severity.value.upper()
            ack = "ACK" if alert.acknowledged else "OPEN"
            st.markdown(f"**[{badge}] {alert.title}** ({ack})")
            st.caption(f"{alert.message} | {alert.timestamp.strftime('%H:%M:%S')} | {alert.source}")
            if not alert.acknowledged and st.button("Ack", key=f"ack_{alert.alert_id}"):
                provider.alert_manager.acknowledge(alert.alert_id)
                _rerun()

        st.subheader("Signal Attribution")
        if decisions and PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
            latest = decisions[-1]
            if latest.signal_components:
                df = pd.DataFrame(
                    [
                        {"signal": k, "contribution": v}
                        for k, v in latest.signal_components.items()
                    ]
                )
                colors = ["green" if v > 0 else "red" for v in df["contribution"]]
                fig = go.Figure(
                    go.Bar(
                        x=df["contribution"],
                        y=df["signal"],
                        orientation="h",
                        marker_color=colors,
                    )
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, width="stretch")

    if st.button("Refresh"):
        _rerun()


def _add_sample_data(provider: IntegratedDashboardProvider) -> None:
    actions = ["BUY", "SELL", "HOLD"]
    symbols = ["BTC", "ETH", "SOL"]
    regimes = ["bull", "bear", "volatile"]

    for i in range(5):
        decision = DecisionRecord(
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=(30 - i * 5)),
            decision_id=f"DEC_{i}",
            action=random.choice(actions),
            symbol=random.choice(symbols),
            direction=random.uniform(-1.0, 1.0),
            confidence=random.uniform(0.5, 0.95),
            mode="NORMAL",
            regime=random.choice(regimes),
            signal_components={
                "trend_following": random.uniform(-0.5, 0.5),
                "momentum": random.uniform(-0.4, 0.4),
                "sentiment": random.uniform(-0.3, 0.3),
                "onchain": random.uniform(-0.2, 0.2),
            },
            risk_checks={"drawdown": True, "correlation": True, "kill_switch": True},
            vetoes=[],
        )
        provider.record_decision(decision)

    provider.alert_manager.add_alert(
        AlertSeverity.WARNING,
        "CorrelationMonitor",
        "Correlation elevated",
        "BTC-ETH correlation rose to 0.92",
    )
    provider.alert_manager.add_alert(
        AlertSeverity.INFO,
        "ExecutionEngine",
        "Order fill complete",
        "BUY 0.5 BTC @ 100500",
    )


_dashboard_provider: Optional[IntegratedDashboardProvider] = None


def get_dashboard_provider() -> IntegratedDashboardProvider:
    """Get singleton dashboard provider."""
    global _dashboard_provider
    if _dashboard_provider is None:
        _dashboard_provider = IntegratedDashboardProvider()
    return _dashboard_provider


def reset_dashboard_provider() -> None:
    """Reset singleton dashboard provider."""
    global _dashboard_provider
    _dashboard_provider = None


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        render_explainer_dashboard()
    else:
        logging.basicConfig(level=logging.INFO)
        provider = IntegratedDashboardProvider()
        _add_sample_data(provider)
        print("\n=== Dashboard Data ===")
        print(json.dumps(provider.get_dashboard_data(), indent=2, default=str))
        print("\n=== Q&A Test ===")
        for q in [
            "why buy?",
            "risk status?",
            "performance?",
        ]:
            print(f"Q: {q}")
            print(f"A: {provider.ask_question(q)}")

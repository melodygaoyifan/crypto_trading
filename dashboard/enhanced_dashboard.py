#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1.0 - Enhanced SOTA Dashboard
================================================================================
Purpose: Comprehensive monitoring dashboard with all SOTA metrics

SOTA Requirements:
- Add latency, fill rate, and slippage stats to the dashboard
- Include execution queue status and order rejection/veto count
- Add alert module for major system events
- Interactive controls and explainability

Features:
1. Real-time P&L and position tracking
2. Execution metrics (latency, fill rate, slippage)
3. Risk monitoring with alerts
4. Agent performance visualization
5. Event replay controls
6. LLM decision explainer integration

Usage:
    streamlit run dashboard/enhanced_dashboard.py

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
================================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# DATA PROVIDERS (Mock for standalone)
# =============================================================================

class EnhancedDataProvider:
    """Provides all data needed for dashboard."""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        np.random.seed(42)
    
    def get_execution_metrics(self) -> Dict:
        """Get execution performance metrics."""
        return {
            "avg_latency_ms": 45 + np.random.randint(-10, 20),
            "p99_latency_ms": 120 + np.random.randint(-20, 40),
            "fill_rate": 0.87 + np.random.uniform(-0.05, 0.05),
            "avg_slippage_bps": 3.2 + np.random.uniform(-1, 2),
            "orders_today": 12,
            "fills_today": 10,
            "partial_fills": 2,
            "rejects_today": 1,
            "veto_count": 3,
            "queue_depth": 2,
            "pending_orders": [
                {"id": "ORD-001", "asset": "BTC", "side": "buy", "status": "pending", "age_ms": 150},
                {"id": "ORD-002", "asset": "ETH", "side": "sell", "status": "partial", "age_ms": 2300}
            ]
        }

    # [P2-FIX] Read real execution quality data from JSONL logs
    def get_real_execution_quality(self) -> Optional[Dict]:
        """Load real execution quality data from fill_quality.jsonl."""
        try:
            fills_path = Path("logs/fill_quality.jsonl")
            if not fills_path.exists():
                return None
            records = []
            with open(fills_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
            if not records:
                return None
            return {"records": records}
        except Exception:
            return None

    def get_latency_history(self) -> List[Dict]:
        """Get latency history for chart."""
        history = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(60):
            history.append({
                "time": base_time + timedelta(minutes=i),
                "latency_ms": 40 + np.random.randint(0, 80),
                "slippage_bps": 2 + np.random.uniform(0, 5)
            })
        return history
    
    def get_agent_performance(self) -> Dict:
        """Get agent performance metrics."""
        return {
            "quant_agent": {
                "weight": 0.45,
                "recent_sharpe": 1.2,
                "hit_rate": 0.62,
                "signals_today": 8
            },
            "drl_agent": {
                "weight": 0.30,
                "recent_sharpe": 0.8,
                "hit_rate": 0.55,
                "status": "SUPERVISED"
            },
            "risk_agent": {
                "weight": 0.25,
                "vetos_today": 3,
                "interventions": 5
            },
            "onchain_agent": {
                "weight": 0.0,
                "status": "DISABLED",
                "last_signal": None
            }
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics."""
        return {
            "current_drawdown": 0.032,
            "max_drawdown_today": 0.045,
            "daily_pnl": 523.50,
            "risk_state": "NORMAL",
            "correlation_state": "NORMAL",
            "kill_switch_active": False,
            "position_limit_usage": 0.65,
            "correlation_btc_eth": 0.87,
            "correlation_btc_sol": 0.72,
            "regime": "bull_trend",
            "regime_confidence": 0.78
        }
    
    def get_positions(self) -> Dict:
        """Get current positions."""
        return {
            "BTC": {"size": 0.5, "entry": 42000, "current": 42500, "pnl": 250, "side": "long", "exposure_pct": 0.21},
            "ETH": {"size": 5.0, "entry": 2200, "current": 2180, "pnl": -100, "side": "long", "exposure_pct": 0.11},
            "SOL": {"size": 100, "entry": 98, "current": 102, "pnl": 400, "side": "long", "exposure_pct": 0.10}
        }
    
    def get_recent_alerts(self) -> List[Dict]:
        """Get recent system alerts."""
        return [
            {"time": "14:32:15", "type": "INFO", "message": "Order filled: BUY 0.5 BTC @ $42,000"},
            {"time": "14:15:42", "type": "WARNING", "message": "High correlation detected (0.92)"},
            {"time": "13:45:00", "type": "INFO", "message": "Regime changed: sideways �?bull_trend"},
            {"time": "12:30:00", "type": "ERROR", "message": "Order rejected: insufficient margin"},
            {"time": "11:00:00", "type": "INFO", "message": "System started in NORMAL mode"}
        ]
    
    def get_replay_status(self) -> Dict:
        """Get event replay status."""
        return {
            "mode": "LIVE",  # "LIVE" or "REPLAY"
            "replay_speed": None,
            "events_loaded": 0,
            "current_event": 0,
            "simulated_time": None
        }


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_execution_metrics(metrics: Dict):
    """Render execution performance metrics."""
    st.subheader("�?Execution Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latency = metrics["avg_latency_ms"]
        color = "normal" if latency < 100 else "inverse"
        st.metric("Avg Latency", f"{latency}ms", delta_color=color)
    
    with col2:
        fill_rate = metrics["fill_rate"]
        st.metric("Fill Rate", f"{fill_rate:.1%}")
    
    with col3:
        slippage = metrics["avg_slippage_bps"]
        color = "normal" if slippage < 5 else "inverse"
        st.metric("Avg Slippage", f"{slippage:.1f} bps", delta_color=color)
    
    with col4:
        st.metric("Orders Today", f"{metrics['orders_today']} ({metrics['fills_today']} filled)")
    
    # Second row: Queue and rejections
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queue Depth", metrics["queue_depth"])
    
    with col2:
        st.metric("Rejections", metrics["rejects_today"], delta_color="inverse")
    
    with col3:
        st.metric("Risk Vetos", metrics["veto_count"])
    
    with col4:
        st.metric("P99 Latency", f"{metrics['p99_latency_ms']}ms")
    
    # Pending orders table
    if metrics.get("pending_orders"):
        st.write("**Pending Orders:**")
        pending_df = pd.DataFrame(metrics["pending_orders"])
        st.dataframe(pending_df, width="stretch")


def render_latency_chart(history: List[Dict]):
    """Render latency and slippage chart."""
    if not PLOTLY_AVAILABLE or not history:
        return
    
    df = pd.DataFrame(history)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["latency_ms"], name="Latency (ms)",
                  line=dict(color="cyan", width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["slippage_bps"], name="Slippage (bps)",
                  line=dict(color="orange", width=2)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Execution Quality (1h)",
        template="plotly_dark",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_yaxes(title_text="Latency (ms)", secondary_y=False)
    fig.update_yaxes(title_text="Slippage (bps)", secondary_y=True)
    
    st.plotly_chart(fig, width="stretch")


# [P2-FIX] Real execution quality panel from fill_quality.jsonl
def render_real_execution_quality(data: Optional[Dict]):
    """Render execution quality from real fill data."""
    st.subheader("Execution Quality (Real Data)")

    if data is None or not data.get("records"):
        st.info("No execution data yet. Waiting for first fills.")
        return

    if not PANDAS_AVAILABLE:
        st.warning("pandas not available for execution quality panel.")
        return

    df = pd.DataFrame(data["records"])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Fills", len(df))
    if "slippage_bps" in df.columns:
        col2.metric(
            "Avg Slippage",
            f"{df['slippage_bps'].mean():+.1f} bps",
        )
        _favorable = (df["slippage_bps"] < 0).mean() * 100
        col3.metric("Favorable %", f"{_favorable:.0f}%")
    if "fill_ratio" in df.columns:
        col4.metric(
            "Avg Fill Rate",
            f"{df['fill_ratio'].mean():.0%}",
        )

    # Per-asset breakdown
    if "asset" in df.columns and "slippage_bps" in df.columns:
        st.write("**Per-Asset Breakdown:**")
        _agg = (
            df.groupby("asset")["slippage_bps"]
            .agg(["count", "mean", "median"])
            .round(2)
        )
        _agg.columns = ["Fills", "Mean Slip (bps)", "Median Slip (bps)"]
        st.dataframe(_agg, width="stretch")


def render_agent_performance(agents: Dict):
    """Render agent performance panel."""
    st.subheader("🤖 Agent Performance")
    
    # Create columns for each agent
    cols = st.columns(len(agents))
    
    for i, (agent_name, data) in enumerate(agents.items()):
        with cols[i]:
            st.write(f"**{agent_name.replace('_', ' ').title()}**")
            
            weight = data.get("weight", 0)
            st.progress(weight, text=f"Weight: {weight:.0%}")
            
            if "recent_sharpe" in data:
                sharpe = data["recent_sharpe"]
                color = "🟢" if sharpe > 1 else ("🟡" if sharpe > 0 else "🔴")
                st.write(f"{color} Sharpe: {sharpe:.2f}")
            
            if "hit_rate" in data:
                st.write(f"Hit Rate: {data['hit_rate']:.0%}")
            
            if "status" in data:
                st.write(f"Status: {data['status']}")
            
            if "vetos_today" in data:
                st.write(f"Vetos: {data['vetos_today']}")


def render_risk_panel(risk: Dict):
    """Render comprehensive risk panel."""
    st.subheader("🛡�?Risk Management")
    
    # Kill switch warning
    if risk.get("kill_switch_active"):
        st.error("🚨 KILL-SWITCH ACTIVE - All trading halted")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dd = risk["current_drawdown"]
        color = "🟢" if dd < 0.03 else ("🟡" if dd < 0.05 else "🔴")
        st.metric("Drawdown", f"{color} {dd:.1%}")
    
    with col2:
        st.metric("Risk State", risk["risk_state"])
    
    with col3:
        st.metric("Regime", risk["regime"].replace("_", " ").title())
    
    with col4:
        st.metric("Regime Confidence", f"{risk['regime_confidence']:.0%}")
    
    # Correlation panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        corr = risk["correlation_btc_eth"]
        color = "🟢" if corr < 0.85 else ("🟡" if corr < 0.95 else "🔴")
        st.metric("BTC-ETH Corr", f"{color} {corr:.2f}")
    
    with col2:
        corr = risk["correlation_btc_sol"]
        st.metric("BTC-SOL Corr", f"{corr:.2f}")
    
    with col3:
        st.metric("Position Usage", f"{risk['position_limit_usage']:.0%}")


def render_positions(positions: Dict):
    """Render positions panel."""
    st.subheader("📊 Positions")
    
    if not positions:
        st.info("No open positions")
        return
    
    data = []
    total_pnl = 0
    total_exposure = 0
    
    for symbol, pos in positions.items():
        pnl = pos["pnl"]
        total_pnl += pnl
        total_exposure += pos["exposure_pct"]
        
        data.append({
            "Symbol": symbol,
            "Side": pos["side"].upper(),
            "Size": pos["size"],
            "Entry": f"${pos['entry']:,.2f}",
            "Current": f"${pos['current']:,.2f}",
            "P&L": f"${pnl:+,.2f}",
            "Exposure": f"{pos['exposure_pct']:.1%}"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, width="stretch")
    
    # Totals
    col1, col2 = st.columns(2)
    with col1:
        color = "normal" if total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"${total_pnl:+,.2f}", delta_color=color)
    with col2:
        st.metric("Total Exposure", f"{total_exposure:.1%}")


def render_alerts(alerts: List[Dict]):
    """Render alerts panel."""
    st.subheader("🔔 Recent Alerts")
    
    for alert in alerts[:10]:
        alert_type = alert["type"]
        
        if alert_type == "ERROR":
            st.error(f"**{alert['time']}** - {alert['message']}")
        elif alert_type == "WARNING":
            st.warning(f"**{alert['time']}** - {alert['message']}")
        else:
            st.info(f"**{alert['time']}** - {alert['message']}")


def render_controls():
    """Render interactive controls."""
    st.subheader("🎮 Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⏸️ Pause Trading", width="stretch"):
            st.session_state.trading_paused = True
            st.success("Trading paused")
    
    with col2:
        if st.button("▶️ Resume Trading", width="stretch"):
            st.session_state.trading_paused = False
            st.success("Trading resumed")
    
    with col3:
        if st.button("🚨 Emergency Flat", width="stretch", type="secondary"):
            st.warning("Confirm emergency flat?")
    
    with col4:
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()


def render_explainer():
    """Render LLM explainer panel."""
    st.subheader("💬 Decision Explainer")
    
    question = st.text_input("Ask about trading decisions:", 
                            placeholder="e.g., Why did we exit ETH?")
    
    if question:
        # Mock response - would call actual explainer
        if "exit" in question.lower():
            st.write("📝 **Explanation:**")
            st.write("The ETH position was exited due to signal reversal. "
                    "The quant agent detected weakening momentum while correlation "
                    "with BTC increased to 0.92, triggering the risk agent's "
                    "correlation-based exit rule.")
        elif "buy" in question.lower() or "entry" in question.lower():
            st.write("📝 **Explanation:**")
            st.write("The position was entered based on OPPORTUNITY mode signals. "
                    "Multiple agents aligned on bullish direction with high confidence.")
        else:
            st.write("📝 **Current State:**")
            st.write("System is in NORMAL mode. Quant agent has primary authority. "
                    "No active risk vetoes.")


def render_replay_controls(status: Dict):
    """Render event replay controls."""
    st.subheader("🔁 Event Replay")
    
    mode = status.get("mode", "LIVE")
    
    if mode == "LIVE":
        st.success("🟢 LIVE MODE")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Historical Events", width="stretch"):
                st.info("Select date range to load events")
        with col2:
            if st.button("Start Replay", width="stretch", disabled=True):
                pass
    else:
        st.warning(f"🔄 REPLAY MODE ({status.get('replay_speed', 'FAST')})")
        
        progress = status["current_event"] / max(status["events_loaded"], 1)
        st.progress(progress, text=f"Event {status['current_event']}/{status['events_loaded']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("⏸️ Pause", width="stretch")
        with col2:
            st.button("⏹️ Stop", width="stretch")
        with col3:
            st.button("�?Speed Up", width="stretch")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main dashboard application."""
    
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available")
        return
    
    # Page config
    st.set_page_config(
        page_title="HMATS Enhanced Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    if "trading_paused" not in st.session_state:
        st.session_state.trading_paused = False
    
    provider = EnhancedDataProvider()
    
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("📈 HMATS v5.1.0 Enhanced Dashboard")
    with col2:
        status = "🔴 PAUSED" if st.session_state.trading_paused else "🟢 ACTIVE"
        st.metric("System", status)
    with col3:
        st.write(f"�?{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    
    st.divider()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trading", "�?Execution", "🛡�?Risk", "🔧 Tools"])
    
    with tab1:
        # Positions and P&L
        col1, col2 = st.columns([2, 1])
        with col1:
            render_positions(provider.get_positions())
        with col2:
            render_alerts(provider.get_recent_alerts())
    
    with tab2:
        # Execution metrics
        render_execution_metrics(provider.get_execution_metrics())
        render_latency_chart(provider.get_latency_history())
        # [P2-FIX] Real execution quality from fill_quality.jsonl
        render_real_execution_quality(provider.get_real_execution_quality())
        render_agent_performance(provider.get_agent_performance())
    
    with tab3:
        # Risk management
        render_risk_panel(provider.get_risk_metrics())
        
        st.divider()
        
        # Explainer
        render_explainer()
    
    with tab4:
        # Tools and controls
        render_controls()
        
        st.divider()
        
        render_replay_controls(provider.get_replay_status())
    
    # Auto-refresh hint
    st.caption("Dashboard auto-refreshes every 5 seconds in production mode.")


if __name__ == "__main__":
    main()


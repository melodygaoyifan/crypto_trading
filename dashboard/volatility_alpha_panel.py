"""
Volatility Alpha dashboard panel.

This module stores VolAlpha snapshots and renders a Streamlit panel
with current status, signal breakdown, timeline, and summary statistics.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

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
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class VolAlphaSnapshot:
    """Single VolAlpha inference snapshot for dashboard use."""

    timestamp: datetime
    asset: str
    vol_bias: str
    intensity: float
    rationale_tags: List[str]
    breakout_signal: float
    liquidation_signal: float
    compression_signal: float
    data_quality: float
    fusion_accepted: bool
    fusion_action: str

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "vol_bias": self.vol_bias,
            "intensity": self.intensity,
            "rationale_tags": list(self.rationale_tags),
            "breakout": self.breakout_signal,
            "liquidation": self.liquidation_signal,
            "compression": self.compression_signal,
            "data_quality": self.data_quality,
            "fusion_accepted": self.fusion_accepted,
            "fusion_action": self.fusion_action,
        }


class VolAlphaDashboardState:
    """In-memory state used by VolAlpha dashboard widgets."""

    def __init__(self, max_history: int = 500):
        self._history: deque[VolAlphaSnapshot] = deque(maxlen=max_history)
        self._current_snapshot: Optional[VolAlphaSnapshot] = None
        self._bear_market_timeline: List[VolAlphaSnapshot] = []
        self._in_bear_market: bool = False
        self._stats = {
            "total_snapshots": 0,
            "increase_count": 0,
            "suppress_count": 0,
            "neutral_count": 0,
            "fusion_accepted_count": 0,
            "fusion_rejected_count": 0,
        }

    def add_snapshot(self, snapshot: VolAlphaSnapshot) -> None:
        self._history.append(snapshot)
        self._current_snapshot = snapshot
        self._stats["total_snapshots"] += 1

        if snapshot.vol_bias == "increase":
            self._stats["increase_count"] += 1
        elif snapshot.vol_bias == "suppress":
            self._stats["suppress_count"] += 1
        else:
            self._stats["neutral_count"] += 1

        if snapshot.fusion_accepted:
            self._stats["fusion_accepted_count"] += 1
        else:
            self._stats["fusion_rejected_count"] += 1

        if self._in_bear_market:
            self._bear_market_timeline.append(snapshot)

    def set_bear_market(self, is_bear: bool) -> None:
        if is_bear and not self._in_bear_market:
            self._bear_market_timeline = []
        self._in_bear_market = is_bear

    def get_current(self) -> Optional[VolAlphaSnapshot]:
        return self._current_snapshot

    def get_history(self, limit: int = 100) -> List[VolAlphaSnapshot]:
        return list(self._history)[-limit:]

    def get_bear_timeline(self) -> List[VolAlphaSnapshot]:
        return list(self._bear_market_timeline)

    def get_stats(self) -> Dict:
        return dict(self._stats)


_dashboard_state: Optional[VolAlphaDashboardState] = None


def get_volalpha_dashboard_state() -> VolAlphaDashboardState:
    global _dashboard_state
    if _dashboard_state is None:
        _dashboard_state = VolAlphaDashboardState()
    return _dashboard_state


class VolAlphaPanelRenderer:
    """Render VolAlpha panel blocks in Streamlit."""

    def __init__(self, state: VolAlphaDashboardState):
        self.state = state

    def render_current_status(self) -> None:
        if not STREAMLIT_AVAILABLE:
            return

        current = self.state.get_current()
        st.subheader("Volatility Alpha Status")

        if current is None:
            st.info("No VolAlpha data available yet")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            bias_color = {"increase": "INC", "suppress": "SUP", "neutral": "NEU"}
            st.metric("Vol Bias", f"{bias_color.get(current.vol_bias, 'NEU')} {current.vol_bias.upper()}")

        with col2:
            st.metric("Intensity", f"{current.intensity:.1%}")

        with col3:
            st.metric("Data Quality", f"{current.data_quality:.1%}")

        with col4:
            fusion_status = "Accepted" if current.fusion_accepted else "Rejected"
            st.metric("Fusion", fusion_status)

        if current.rationale_tags:
            st.write("**Rationale Tags:**")
            st.markdown(", ".join([f"`{tag}`" for tag in current.rationale_tags]))

    def render_sub_signals(self) -> None:
        if not STREAMLIT_AVAILABLE or not PLOTLY_AVAILABLE:
            return

        current = self.state.get_current()
        if current is None:
            return

        st.subheader("Sub-Signal Breakdown")
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Breakout", "Liquidation", "Compression"],
                    y=[
                        current.breakout_signal,
                        current.liquidation_signal,
                        current.compression_signal,
                    ],
                    marker_color=["#FF6B6B", "#4ECDC4", "#FFE66D"],
                    text=[
                        f"{current.breakout_signal:.2f}",
                        f"{current.liquidation_signal:.2f}",
                        f"{current.compression_signal:.2f}",
                    ],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title="Sub-Signal Contributions",
            yaxis_title="Signal Intensity",
            yaxis_range=[0, 1],
            height=300,
        )
        st.plotly_chart(fig, width="stretch")

    def render_timeline(self, limit: int = 50) -> None:
        if not STREAMLIT_AVAILABLE or not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            return

        history = self.state.get_history(limit)
        if not history:
            return

        st.subheader("Signal Timeline")
        df = pd.DataFrame(
            [
                {
                    "timestamp": s.timestamp,
                    "intensity": s.intensity,
                    "vol_bias": s.vol_bias,
                    "breakout": s.breakout_signal,
                    "liquidation": s.liquidation_signal,
                    "compression": s.compression_signal,
                }
                for s in history
            ]
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Total Intensity", "Sub-Signals"),
        )

        colors = [
            "red" if b == "increase" else "green" if b == "suppress" else "gray"
            for b in df["vol_bias"]
        ]

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["intensity"],
                mode="lines+markers",
                name="Intensity",
                marker=dict(color=colors),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["breakout"], name="Breakout", line=dict(color="#FF6B6B")),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["liquidation"], name="Liquidation", line=dict(color="#4ECDC4")),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["compression"], name="Compression", line=dict(color="#FFE66D")),
            row=2,
            col=1,
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")

    def render_bear_market_timeline(self) -> None:
        if not STREAMLIT_AVAILABLE:
            return

        timeline = self.state.get_bear_timeline()
        st.subheader("Bear Market Volatility Timeline")

        if not timeline:
            st.info("No bear market timeline data available")
            return

        if not PANDAS_AVAILABLE:
            for snapshot in timeline[-20:]:
                st.text(f"{snapshot.timestamp}: {snapshot.vol_bias} ({snapshot.intensity:.1%})")
            return

        df = pd.DataFrame([s.to_dict() for s in timeline[-50:]])
        st.dataframe(
            df[["timestamp", "vol_bias", "intensity", "rationale_tags", "fusion_action"]],
            width="stretch",
        )

    def render_statistics(self) -> None:
        if not STREAMLIT_AVAILABLE:
            return

        stats = self.state.get_stats()
        st.subheader("Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Signals", stats["total_snapshots"])
            st.metric("Increase", stats["increase_count"])
        with col2:
            st.metric("Suppress", stats["suppress_count"])
            st.metric("Neutral", stats["neutral_count"])
        with col3:
            acceptance_rate = (
                stats["fusion_accepted_count"] / max(stats["total_snapshots"], 1) * 100.0
            )
            st.metric("Fusion Acceptance", f"{acceptance_rate:.1f}%")

    def render_full_panel(self) -> None:
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit not available, cannot render panel")
            return

        st.header("Volatility Alpha Agent")
        with st.container():
            self.render_current_status()

        col1, col2 = st.columns(2)
        with col1:
            self.render_sub_signals()
        with col2:
            self.render_statistics()

        self.render_timeline()
        with st.expander("Bear Market Timeline"):
            self.render_bear_market_timeline()


class VolAlphaDashboardIntegration:
    """Integration facade between dashboard and VolatilityAlphaAgent."""

    def __init__(self):
        self._state = get_volalpha_dashboard_state()
        self._renderer = VolAlphaPanelRenderer(self._state)
        self._vol_alpha_agent = None

    def connect_to_vol_alpha_agent(self) -> bool:
        try:
            from agents.volatility_alpha_agent import get_volatility_alpha_agent

            self._vol_alpha_agent = get_volatility_alpha_agent()
            logger.info("Dashboard connected to VolAlpha agent")
            return True
        except Exception as exc:
            logger.warning("Failed to connect to VolAlpha agent: %s", exc)
            return False

    def update_from_agent(self) -> None:
        if not self._vol_alpha_agent:
            return

        try:
            status = self._vol_alpha_agent.get_status()
            last_intent = status.get("last_intent")
            if not last_intent:
                return

            snapshot = VolAlphaSnapshot(
                timestamp=datetime.fromisoformat(last_intent["timestamp"]),
                asset=last_intent.get("asset", "UNKNOWN"),
                vol_bias=last_intent.get("vol_bias", "neutral"),
                intensity=float(last_intent.get("intensity", 0.0)),
                rationale_tags=list(last_intent.get("rationale_tags", [])),
                breakout_signal=float(last_intent.get("sub_signals", {}).get("breakout", 0.0)),
                liquidation_signal=float(last_intent.get("sub_signals", {}).get("liquidation", 0.0)),
                compression_signal=float(last_intent.get("sub_signals", {}).get("compression", 0.0)),
                data_quality=float(last_intent.get("data_quality", 1.0)),
                fusion_accepted=bool(last_intent.get("fusion_accepted", False)),
                fusion_action=str(last_intent.get("fusion_action", "NONE")),
            )
            self._state.add_snapshot(snapshot)
        except Exception as exc:
            logger.warning("Failed to update VolAlpha panel state: %s", exc)

    def render(self) -> None:
        self._renderer.render_full_panel()


_dashboard_integration: Optional[VolAlphaDashboardIntegration] = None


def get_volalpha_dashboard_integration() -> VolAlphaDashboardIntegration:
    global _dashboard_integration
    if _dashboard_integration is None:
        _dashboard_integration = VolAlphaDashboardIntegration()
    return _dashboard_integration


__all__ = [
    "VolAlphaSnapshot",
    "VolAlphaDashboardState",
    "get_volalpha_dashboard_state",
    "VolAlphaPanelRenderer",
    "VolAlphaDashboardIntegration",
    "get_volalpha_dashboard_integration",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    state = get_volalpha_dashboard_state()
    state.add_snapshot(
        VolAlphaSnapshot(
            timestamp=datetime.now(),
            asset="BTC",
            vol_bias="increase",
            intensity=0.72,
            rationale_tags=["spread_widening", "liquidation_spike"],
            breakout_signal=0.67,
            liquidation_signal=0.74,
            compression_signal=0.31,
            data_quality=0.95,
            fusion_accepted=True,
            fusion_action="AMPLIFY",
        )
    )
    print("VolAlpha state initialized with one sample snapshot")

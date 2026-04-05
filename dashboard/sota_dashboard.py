#!/usr/bin/env python3
"""
HMATS v6.8 - Paper Trading Dashboard
=====================================
Reads live data from file-based state exports:
  - data/dashboard_state.json  (per-round snapshot from main.py)
  - data/paper_positions.json  (position persistence)
  - data/shadow_ledger/*.jsonl (trade history)
  - data/paper_run.pid         (process status)

Usage:
    streamlit run dashboard/sota_dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any

import streamlit as st

try:
    import pandas as pd
    import numpy as np
except ImportError:
    st.error("Missing pandas/numpy. Run: pip install pandas numpy")
    st.stop()

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# =============================================================================
# FILE-BASED DATA PROVIDER
# =============================================================================

DATA_DIR = ROOT / "data"
DASHBOARD_STATE = DATA_DIR / "dashboard_state.json"
PAPER_POSITIONS = DATA_DIR / "paper_positions.json"
PID_FILE = DATA_DIR / "paper_run.pid"
SHADOW_LEDGER_DIR = DATA_DIR / "shadow_ledger"
STDERR_LOG = ROOT / "logs" / "paper_run_stderr.log"


def _read_json(path: Path) -> dict:
    """Safely read a JSON file. Returns {} on any error."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _read_jsonl_tail(directory: Path, max_lines: int = 200) -> List[dict]:
    """Read last N lines from the most recent .jsonl file."""
    try:
        files = sorted(directory.glob("ledger_*.jsonl"), key=lambda f: f.name, reverse=True)
        if not files:
            return []
        lines = files[0].read_text(encoding="utf-8", errors="replace").strip().split("\n")
        result = []
        for line in lines[-max_lines:]:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result
    except Exception:
        return []


def _process_alive(pid: int) -> bool:
    """Check if PID is alive (Windows)."""
    try:
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
    except Exception:
        pass
    return False


def get_process_status() -> dict:
    """Get paper run process status."""
    pid_data = _read_json(PID_FILE)
    pid = pid_data.get("pid")
    if not pid:
        return {"running": False, "pid": None, "started_at": None}
    alive = _process_alive(pid)
    return {
        "running": alive,
        "pid": pid,
        "started_at": pid_data.get("started_at"),
    }


def get_dashboard_state() -> dict:
    """Read the main dashboard state exported by main.py."""
    return _read_json(DASHBOARD_STATE)


def get_positions() -> dict:
    """Read paper positions."""
    data = _read_json(PAPER_POSITIONS)
    return data.get("positions", {})


def get_trades(max_trades: int = 50) -> List[dict]:
    """Read recent trades from shadow ledger."""
    entries = _read_jsonl_tail(SHADOW_LEDGER_DIR, max_lines=300)
    fills = [e for e in entries if e.get("entry_type") == "FILL"]
    return fills[-max_trades:]


def get_pnl_history() -> List[dict]:
    """Extract PnL entries from shadow ledger."""
    entries = _read_jsonl_tail(SHADOW_LEDGER_DIR, max_lines=500)
    return [e for e in entries if e.get("entry_type") in ("FILL", "POSITION")]


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_header(proc_status: dict, state: dict):
    """Top bar: title, process status, last update."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        st.title("HMATS v6.8 Paper Dashboard")

    with col2:
        if proc_status["running"]:
            st.metric("Process", f"PID {proc_status['pid']}")
        else:
            st.metric("Process", "STOPPED")

    with col3:
        updated = state.get("updated_at", "")
        if updated:
            try:
                dt = datetime.fromisoformat(updated.rstrip("Z"))
                age = (datetime.now(timezone.utc) - dt).total_seconds()
                if age < 300:
                    st.metric("Data Age", f"{age:.0f}s")
                else:
                    st.metric("Data Age", f"{age/3600:.1f}h")
            except Exception:
                st.metric("Data Age", "?")
        else:
            st.metric("Data Age", "No data")

    with col4:
        rnd = state.get("round", 0)
        ticks = state.get("tick_count", 0)
        st.metric("Round / Ticks", f"{rnd} / {ticks}")


def render_equity_and_risk(state: dict, positions: dict):
    """Equity, exposure, risk metrics."""
    st.subheader("Portfolio Overview")

    equity = state.get("equity", 10_000)  # [CFG-4] was 100K, actual $10K
    total_notional = sum(p.get("notional", 0) for p in positions.values())
    total_exposure_pct = sum(abs(p.get("exposure", 0)) for p in positions.values()) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Equity", f"${equity:,.0f}")
    with col2:
        st.metric("Total Notional", f"${total_notional:,.0f}")
    with col3:
        st.metric("Gross Exposure", f"{total_exposure_pct:.1f}%")
    with col4:
        n_pos = sum(1 for p in positions.values() if abs(p.get("exposure", 0)) > 0.001)
        st.metric("Open Positions", str(n_pos))


def render_positions(positions: dict, asset_data: dict):
    """Position table with per-asset details."""
    st.subheader("Positions & Signals")

    if not positions:
        st.info("No open positions")
        return

    rows = []
    for asset, pos in positions.items():
        ad = asset_data.get(asset, {})
        direction = pos.get("direction", 0)
        side = "SHORT" if direction < 0 else ("LONG" if direction > 0 else "FLAT")
        exposure = abs(pos.get("exposure", 0)) * 100
        entry = pos.get("entry_price", 0)
        current = ad.get("price", entry)
        notional = pos.get("notional", 0)

        # Unrealized PnL estimate
        if entry > 0 and current > 0 and notional > 0:
            if direction < 0:
                pnl_pct = (entry - current) / entry * 100
            else:
                pnl_pct = (current - entry) / entry * 100
            pnl_usd = notional * pnl_pct / 100
        else:
            pnl_pct = 0
            pnl_usd = 0

        rows.append({
            "Asset": asset,
            "Side": side,
            "Exposure": f"{exposure:.1f}%",
            "Entry": f"${entry:,.2f}",
            "Current": f"${current:,.2f}",
            "PnL $": f"${pnl_usd:,.0f}",
            "PnL %": f"{pnl_pct:+.2f}%",
            "Regime": ad.get("regime", "?"),
            # [FIX] Read strategy/mode from position (entry-time), not asset_data (current tick)
            "Strategy": pos.get("strategy", ad.get("strategy", "?")),
            "Alpha": f"{ad.get('alpha_bps', 0):.0f}bps",
            "Mode": pos.get("mode_at_entry", ad.get("system_mode", "?")),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_regime_panel(asset_data: dict):
    """Regime classification per asset."""
    st.subheader("Market Regime")

    if not asset_data:
        st.info("Waiting for first tick...")
        return

    cols = st.columns(len(asset_data))
    regime_icons = {
        "MOMENTUM_RALLY": "📈",
        "PANIC_SELLOFF": "📉",
        "QUIET_ACCUMULATION": "😴",
        "WEAK_CONSOLIDATION": "➡️",
        "VOLATILE_CHOP": "🌊",
        "EXTREME_VOLATILITY": "[VOL]",
    }

    for col, (asset, ad) in zip(cols, asset_data.items()):
        with col:
            regime = ad.get("regime", "UNKNOWN")
            conf = ad.get("regime_confidence", 0.5)
            icon = regime_icons.get(regime, "[?]")
            st.metric(asset, f"{icon} {regime.replace('_', ' ').title()}")
            st.progress(min(conf, 1.0), text=f"Confidence: {conf:.0%}")

            alpha = ad.get("alpha_bps", 0)
            threshold = ad.get("alpha_threshold_bps", 0)
            passed = ad.get("alpha_gate_passed", False)
            gate_str = "PASS" if passed else "FAIL"
            st.caption(
                f"Alpha: {alpha:.0f} / {threshold:.0f} bps ({gate_str}) | "
                f"Strategy: {ad.get('strategy', '?')}"
            )


def render_signals(asset_data: dict):
    """Technical signal details per asset."""
    st.subheader("Signal Details")

    if not asset_data:
        return

    rows = []
    for asset, ad in asset_data.items():
        # [FIX] target_exposure in asset_data can be the raw fusion direction
        # (0.52 = 52%) when no position exists. Clamp to realistic range.
        raw_exp = ad.get('target_exposure', 0)
        display_exp = min(abs(raw_exp), 0.30) * 100  # Cap at 30% for display
        rows.append({
            "Asset": asset,
            "Direction": f"{ad.get('direction', 0):+.3f}",
            "Target Exp": f"{display_exp:.1f}%",
            "Strategy": ad.get("strategy", "?"),
            "RSI": f"{ad.get('rsi', 0):.1f}",
            "ADX": f"{ad.get('adx', 0):.1f}",
            "Vol Ratio": f"{ad.get('volume_ratio', 0):.2f}",
            "Spread": f"{ad.get('spread_bps', 0):.1f}bps",
            "Veto": "YES" if ad.get("veto_active") else "No",
            "Actionable": "YES" if ad.get("actionable") else "No",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_trade_history(trades: List[dict]):
    """Recent trades from shadow ledger."""
    st.subheader("Recent Trades")

    if not trades:
        st.info("No trades recorded yet")
        return

    rows = []
    for t in reversed(trades[-20:]):
        data = t.get("data", {})
        ts = t.get("timestamp", "")
        try:
            ts_short = ts[11:19] if len(ts) > 19 else ts
        except Exception:
            ts_short = ts

        rows.append({
            "Time": ts_short,
            "Asset": t.get("asset", "?"),
            "Side": data.get("side", "?"),
            "Qty": f"{data.get('size', 0):.4f}",
            "Price": f"${data.get('price', 0):,.2f}",
            "Notional": f"${data.get('notional', 0):,.0f}",
            "Fee": f"${data.get('fee', 0):.2f}",
            "PnL": f"${data.get('realized_pnl', 0):,.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_pnl_chart(trades: List[dict]):
    """Cumulative PnL chart from fills."""
    if not go:
        return

    st.subheader("Cumulative Fees & Notional")

    if not trades:
        return

    # Build cumulative series
    timestamps = []
    cum_fees = []
    cum_notional = []
    total_fee = 0
    total_notional = 0

    for t in trades:
        data = t.get("data", {})
        ts = t.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        total_fee += data.get("fee", 0)
        total_notional += data.get("notional", 0)
        timestamps.append(dt)
        cum_fees.append(total_fee)
        cum_notional.append(total_notional)

    if not timestamps:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_notional,
        mode="lines+markers", name="Cumulative Notional",
        line=dict(color="cyan", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_fees,
        mode="lines+markers", name="Cumulative Fees",
        line=dict(color="orange", width=1),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        yaxis=dict(title="Notional ($)"),
        yaxis2=dict(title="Fees ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, width="stretch")


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title="HMATS Paper Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # -- Load all data --
    proc_status = get_process_status()
    state = get_dashboard_state()
    positions = get_positions()
    asset_data = state.get("assets", {})
    trades = get_trades()

    # -- Sidebar --
    with st.sidebar:
        if proc_status["running"]:
            st.success(f"Paper run active (PID {proc_status['pid']})")
        else:
            st.error("Paper run NOT running")

        if state:
            st.caption(f"Last update: {state.get('updated_at', '?')}")
            st.caption(f"Round {state.get('round', 0)} | {state.get('tick_count', 0)} ticks")
        else:
            st.warning("No dashboard_state.json �?run a paper tick first")

        st.divider()
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)

    # -- Header --
    render_header(proc_status, state)
    st.divider()

    # -- Main content --
    render_equity_and_risk(state, positions)
    st.divider()

    render_positions(positions, asset_data)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_regime_panel(asset_data)
    with col2:
        render_signals(asset_data)

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        render_trade_history(trades)
    with col2:
        render_pnl_chart(trades)

    # -- Auto-refresh --
    if auto_refresh:
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()


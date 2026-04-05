#!/usr/bin/env python3
"""
HMATS v6.7 - Trading Dashboard
Real-time monitoring dashboard for HMATS trading system.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import gzip
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import logging

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
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SHADOW_LEDGER_DIR = PROJECT_ROOT / "data" / "shadow_ledger"
EVENTS_DIR = PROJECT_ROOT / "logs" / "events"
LOGS_DIR = PROJECT_ROOT / "logs"
PAPER_POSITIONS_FILE = PROJECT_ROOT / "data" / "paper_positions.json"
DASHBOARD_STATE_FILE = PROJECT_ROOT / "data" / "dashboard_state.json"
LEDGER_FRESHNESS_HOURS = 8
RUNTIME_STATE_FRESHNESS_HOURS = 4


# =============================================================================
# DATA LOADERS
# =============================================================================

class DashboardDataLoader:
    """Load data for dashboard display from real sources."""

    def __init__(self):
        self._ledger_cache = None
        self._ledger_mtime = 0

    # ---- Paper Positions Fallback ----

    def _load_paper_positions_fallback(self) -> Dict[str, Dict]:
        """Load positions from paper_positions.json as fallback."""
        for fpath in [PAPER_POSITIONS_FILE, DASHBOARD_STATE_FILE]:
            if not fpath.exists():
                continue
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                positions = data.get("positions", {})
                if positions:
                    return positions
            except Exception as e:
                logger.debug(f"Error reading {fpath}: {e}")
        return {}

    @staticmethod
    def _is_active_position(position: Dict[str, Any]) -> bool:
        try:
            exposure = abs(float(position.get("exposure", 0.0) or 0.0))
            direction = float(position.get("direction", 0.0) or 0.0)
            notional = abs(float(position.get("notional", 0.0) or 0.0))
        except Exception:
            return False
        return exposure > 1e-4 and abs(direction) > 0.1 and notional >= 1.0

    def _read_json_file(self, fpath: Path) -> Dict[str, Any]:
        """Read a JSON file safely and return a dict."""
        if not fpath.exists():
            return {}
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.debug(f"Error reading {fpath}: {e}")
            return {}

    def _runtime_state(self) -> Dict[str, Dict[str, Any]]:
        """Return latest dashboard/paper state snapshots."""
        return {
            "dashboard": self._read_json_file(DASHBOARD_STATE_FILE),
            "paper": self._read_json_file(PAPER_POSITIONS_FILE),
        }

    def _latest_session_meta(self) -> Dict[str, Any]:
        """Return freshness metadata for the newest shadow-ledger session."""
        records = self._read_all_ledgers()
        if not records:
            return {"records": [], "latest_ts": None, "session_id": "", "fresh": False}

        latest_ts = None
        for rec in reversed(records):
            latest_ts = self._parse_event_timestamp(str(rec.get("timestamp", "")))
            if latest_ts is not None:
                break
        if latest_ts is None:
            return {"records": [], "latest_ts": None, "session_id": "", "fresh": False}

        latest_sid = records[-1].get("session_id", "")
        session_records = [r for r in records if r.get("session_id") == latest_sid] if latest_sid else []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=LEDGER_FRESHNESS_HOURS)
        fresh = latest_ts >= cutoff
        return {
            "records": session_records if fresh else [],
            "latest_ts": latest_ts,
            "session_id": latest_sid,
            "fresh": fresh,
        }

    # ---- Shadow Ledger ----

    def _read_all_ledgers(self) -> List[Dict]:
        """Read all records from all shadow ledger files."""
        if not SHADOW_LEDGER_DIR.exists():
            return []

        files = sorted(SHADOW_LEDGER_DIR.glob("ledger_*.jsonl"))
        if not files:
            return []

        # Check if latest file changed
        latest_mtime = max(f.stat().st_mtime for f in files)
        if latest_mtime == self._ledger_mtime and self._ledger_cache is not None:
            return self._ledger_cache

        records = []
        for fpath in files:
            try:
                for line in fpath.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Error reading ledger {fpath}: {e}")

        self._ledger_cache = records
        self._ledger_mtime = latest_mtime
        return records

    def _latest_session_records(self) -> List[Dict]:
        """Get records from the most recent session only."""
        return self._latest_session_meta()["records"]

    @staticmethod
    def _parse_event_timestamp(ts_str: str) -> datetime | None:
        """Parse event timestamp and normalize to UTC-aware datetime."""
        if not ts_str:
            return None

        normalized = str(ts_str).strip()
        # Accept both "...Z" and malformed "...+00:00Z".
        if normalized.endswith("Z"):
            if normalized.endswith("+00:00Z") or normalized.endswith("-00:00Z"):
                normalized = normalized[:-1]
            else:
                normalized = normalized[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def load_positions(self) -> Dict[str, Any]:
        """Load current positions - paper_positions.json is ground truth."""
        positions = {}
        state = self._runtime_state()

        # Primary source: paper_positions.json (actual held positions)
        paper_pos = self._load_paper_positions_fallback()

        # Supplementary: latest FILL per asset from shadow ledger
        records = self._latest_session_records()
        fill_by_asset: Dict[str, Dict] = {}
        for r in records:
            if r.get("entry_type") == "FILL":
                fill_by_asset[r.get("asset", "")] = r

        for asset in ["BTC", "ETH", "SOL"]:
            symbol = f"{asset}/USD"
            pp = paper_pos.get(asset, {})
            pp_dir = float(pp.get("direction", 0) or 0)
            asset_state = state.get("dashboard", {}).get("assets", {}).get(asset, {})
            mark_price = float(asset_state.get("price", 0.0) or 0.0)

            if self._is_active_position(pp):
                side = "SHORT" if pp_dir < 0 else "LONG"
                fill_rec = fill_by_asset.get(asset)
                last_fill_price = fill_rec["data"].get("price", 0) if fill_rec else 0
                entry_price = float(pp.get("entry_price", 0) or 0.0)
                notional = float(pp.get("notional", 0) or 0.0)
                if mark_price > 0 and entry_price > 0 and notional > 0:
                    unrealized_pnl = ((mark_price - entry_price) / entry_price) * pp_dir * notional
                else:
                    unrealized_pnl = 0.0
                positions[symbol] = {
                    "size": abs(pp.get("exposure", 0)),
                    "side": side,
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized_pnl,
                    "notional": notional,
                    "tranche": pp.get("tranche", 0),
                    "last_fill_price": last_fill_price,
                }
            else:
                positions[symbol] = {
                    "size": 0, "side": "FLAT", "entry_price": 0,
                    "unrealized_pnl": 0, "notional": 0, "tranche": 0,
                }

        return positions

    def load_risk_metrics(self) -> Dict[str, float]:
        """Load risk metrics from shadow ledger."""
        records = self._latest_session_records()
        state = self._runtime_state()
        dashboard_state = state.get("dashboard", {})
        paper_state = state.get("paper", {})
        equity = float(dashboard_state.get("equity", 10_000.0) or 10_000.0)
        if equity <= 0:
            equity = 10_000.0

        if not records:
            positions = self.load_positions()
            current_notional = sum(p.get("notional", 0) for p in positions.values())
            cumulative_pnl = float(paper_state.get("cumulative_pnl", 0.0) or 0.0)
            peak_equity = float(paper_state.get("peak_equity", equity) or equity)
            max_drawdown = (
                max((peak_equity - equity) / peak_equity, 0.0)
                if peak_equity > 0 else 0.0
            )
            return {
                "exposure_pct": current_notional / equity if equity > 0 else 0.0,
                "session_pnl": cumulative_pnl,
                "max_drawdown": max_drawdown,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
            }

        total_notional = 0.0
        total_realized = 0.0
        n_fills = 0
        n_wins = 0

        for r in records:
            if r["entry_type"] == "FILL":
                total_notional += abs(r["data"].get("notional", 0))
                rpnl = r["data"].get("realized_pnl", 0)
                total_realized += rpnl
                n_fills += 1
                if rpnl > 0:
                    n_wins += 1

        session_pnl = float(paper_state.get("cumulative_pnl", total_realized) or total_realized)

        # Current exposure: sum of latest position notionals
        positions = self.load_positions()
        current_notional = sum(p.get("notional", 0) for p in positions.values())
        exposure = current_notional / equity if equity > 0 else 0
        peak_equity = float(paper_state.get("peak_equity", equity) or equity)
        max_drawdown = (
            max((peak_equity - equity) / peak_equity, 0.0)
            if peak_equity > 0 else 0.0
        )

        return {
            "exposure_pct": exposure,
            "session_pnl": session_pnl,
            "max_drawdown": max_drawdown,
            "win_rate": n_wins / n_fills if n_fills > 0 else 0,
            "sharpe_ratio": 0.0,
        }

    def load_pnl_history(self) -> List[Dict]:
        """Load cumulative P&L from latest session; fallback to runtime snapshot."""
        records = self._latest_session_records()
        points = []
        running_pnl = 0.0

        for r in records:
            if r["entry_type"] == "FILL":
                rpnl = r["data"].get("realized_pnl", 0)
                running_pnl += rpnl
                points.append({
                    "timestamp": r["timestamp"],
                    "pnl": running_pnl,
                    "asset": r.get("asset", "?"),
                })

        state = self._runtime_state()
        dashboard_state = state.get("dashboard", {})
        paper_state = state.get("paper", {})
        cumulative_pnl = paper_state.get("cumulative_pnl")
        raw_ts = (
            str(dashboard_state.get("updated_at", ""))
            or str(paper_state.get("saved_at", ""))
        )
        ts = self._parse_event_timestamp(raw_ts)
        ts_out = ts.isoformat() if ts else datetime.now(timezone.utc).isoformat()
        if cumulative_pnl is not None:
            snapshot_point = {
                "timestamp": ts_out,
                "pnl": float(cumulative_pnl),
                "asset": "PORTFOLIO",
            }
            if points:
                if abs(points[-1]["pnl"] - snapshot_point["pnl"]) > 1e-6:
                    points.append(snapshot_point)
                return points
            return [snapshot_point]

        return points

    def load_trade_records(self, hours: int = 24) -> List[Dict]:
        """Load trade records from fresh ledger, then fallback to event bus fills."""
        records = self._latest_session_records()
        if records:
            return records

        fills = []
        for e in self.load_recent_events(hours=hours):
            if e.get("event_type") != "ORDER_SUBMITTED":
                continue

            payload = e.get("payload", {}) or {}
            result = payload.get("result", {}) or {}
            filled_size = float(result.get("filled_size", result.get("requested_size", 0.0)) or 0.0)
            filled_price = float(result.get("filled_price", result.get("requested_price", 0.0)) or 0.0)
            notional = float(
                result.get(
                    "notional_usd",
                    result.get("notional", payload.get("notional", filled_size * filled_price)),
                ) or 0.0
            )
            fee = float(result.get("fee", 0.0) or 0.0)
            realized_pnl = float(result.get("realized_pnl", 0.0) or 0.0)
            side = str(result.get("side", payload.get("side", "?")))

            fills.append({
                "timestamp": e.get("timestamp", ""),
                "entry_type": "FILL",
                "asset": payload.get("asset", "-"),
                "data": {
                    "side": side,
                    "size": filled_size,
                    "price": filled_price,
                    "notional": notional,
                    "fee": fee,
                    "realized_pnl": realized_pnl,
                },
            })

        return fills

    # ---- Events ----

    def load_recent_events(self, hours: int = 24) -> List[Dict]:
        """Load recent events from event bus JSONL files."""
        if not EVENTS_DIR.exists():
            return []

        events = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        for log_file in sorted(EVENTS_DIR.glob("events_*.jsonl*")):
            try:
                if log_file.suffix == ".gz":
                    with gzip.open(log_file, "rt", encoding="utf-8") as f:
                        lines = f.readlines()
                else:
                    lines = log_file.read_text(encoding="utf-8").splitlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    ts_str = event.get("timestamp", "2000-01-01")
                    event_time = self._parse_event_timestamp(ts_str)
                    if event_time is None:
                        continue
                    if event_time >= cutoff:
                        events.append(event)
            except Exception as e:
                logger.debug(f"Error loading {log_file}: {e}")

        return events

    # ---- System Status ----

    def load_system_status(self) -> Dict[str, Any]:
        """Load system status from events, ledger, and stderr."""
        events = self.load_recent_events(hours=1)
        ledger_meta = self._latest_session_meta()
        records = ledger_meta["records"]
        runtime = self._runtime_state().get("dashboard", {})

        # From latest INTENT
        regime = "UNKNOWN"
        run_mode = "PAPER"
        for r in reversed(records):
            if r["entry_type"] == "INTENT":
                regime = r["data"].get("regime", "UNKNOWN")
                run_mode = r["data"].get("mode", "PAPER")
                break
        if regime == "UNKNOWN":
            assets = runtime.get("assets", {})
            if isinstance(assets, dict) and assets:
                first_asset = next(iter(assets.values()), {})
                if isinstance(first_asset, dict):
                    regime = first_asset.get("regime", regime)
            run_mode = runtime.get("mode", run_mode)

        # Last heartbeat from events
        last_heartbeat = "N/A"
        for e in reversed(events):
            if e.get("event_type") == "HEARTBEAT":
                last_heartbeat = e.get("timestamp", "")[:19]
                break
        if last_heartbeat == "N/A":
            runtime_ts = str(runtime.get("updated_at", ""))
            if runtime_ts:
                last_heartbeat = runtime_ts[:19]

        # System mode from stderr tail
        system_mode = "NORMAL"
        stderr_path = LOGS_DIR / "paper_run_stderr.log"
        if stderr_path.exists():
            try:
                raw = stderr_path.read_bytes()[-5000:]
                content = raw.decode("utf-8", errors="replace")
                for line in reversed(content.splitlines()):
                    if "Mode=" in line:
                        idx = line.find("Mode=")
                        mode_str = line[idx + 5:].split()[0].split("|")[0].strip()
                        if mode_str in ("NORMAL", "OPPORTUNITY", "NO_TRADE"):
                            system_mode = mode_str
                            break
            except Exception:
                pass

        runtime_updated_at = self._parse_event_timestamp(str(runtime.get("updated_at", "")))
        runtime_cutoff = datetime.now(timezone.utc) - timedelta(hours=RUNTIME_STATE_FRESHNESS_HOURS)
        runtime_fresh = runtime_updated_at is not None and runtime_updated_at >= runtime_cutoff
        runtime_source = "fresh-ledger" if ledger_meta["fresh"] else ("runtime-fallback" if runtime_fresh else "stale")
        drl_state = runtime.get("drl", {}) if isinstance(runtime.get("drl"), dict) else {}
        sentiment_state = runtime.get("sentiment", {}) if isinstance(runtime.get("sentiment"), dict) else {}
        runtime_assets = runtime.get("assets", {}) if isinstance(runtime.get("assets"), dict) else {}
        learned_exec_state = {}
        composite_toxicity_state = {}
        g6_state = {}
        aggressive_allocator_state = {}
        lead_lag_state = {}
        model_alpha_state = {}
        for asset, payload in runtime_assets.items():
            if not isinstance(payload, dict):
                continue
            learned_exec_state[asset] = {
                "authority": payload.get("learned_exec_authority", "DISABLED"),
                "mode": payload.get("learned_exec_mode", "DISABLED"),
                "action": payload.get("learned_exec_action", "none"),
                "confidence": float(payload.get("learned_exec_confidence", 0.0) or 0.0),
                "aggressiveness": float(payload.get("learned_exec_aggressiveness", 0.0) or 0.0),
                "offset_bps": float(payload.get("learned_exec_limit_offset_bps", 0.0) or 0.0),
                "applied": bool(payload.get("learned_exec_applied", False)),
            }
            composite_toxicity_state[asset] = {
                "authority": payload.get("composite_toxicity_authority", "DISABLED"),
                "score": float(payload.get("composite_toxicity_score", 0.0) or 0.0),
                "warn": bool(payload.get("composite_toxicity_warn", False)),
                "dominant": payload.get("composite_toxicity_dominant", "none"),
                "applied": bool(payload.get("composite_toxicity_applied", False)),
            }
            g6_state[asset] = {
                "authority": payload.get("g6_authority", "DISABLED"),
                "mode": payload.get("g6_mode", "DISABLED"),
                "weight": float(payload.get("g6_weight", 1.0 / 3.0) or (1.0 / 3.0)),
                "multiplier": float(payload.get("g6_multiplier", 1.0) or 1.0),
                "applied": bool(payload.get("g6_applied", False)),
            }
            aggressive_allocator_state[asset] = {
                "authority": payload.get("aggressive_allocator_authority", "DISABLED"),
                "score": float(payload.get("aggressive_allocator_score", 0.0) or 0.0),
                "cap": float(payload.get("aggressive_allocator_cap", 0.0) or 0.0),
                "tier": payload.get("aggressive_allocator_tier", "NONE"),
                "applied": bool(payload.get("aggressive_allocator_applied", False)),
            }
            lead_lag_state[asset] = {
                "authority": payload.get("lead_lag_authority", "DISABLED"),
                "edge": float(payload.get("lead_lag_edge", 0.0) or 0.0),
                "confidence": float(payload.get("lead_lag_confidence", 0.0) or 0.0),
                "multiplier": float(payload.get("lead_lag_confidence_multiplier", 1.0) or 1.0),
                "applied": bool(payload.get("lead_lag_applied", False)),
            }
            model_alpha_state[asset] = {
                "model_type": payload.get("model_alpha_model_type", "none"),
                "model_version": payload.get("model_alpha_model_version", "none"),
                "confidence": float(payload.get("model_alpha_confidence", 0.0) or 0.0),
                "weight": float(payload.get("model_alpha_weight", 0.0) or 0.0),
                "delta": float(payload.get("model_alpha_confidence_delta", 0.0) or 0.0),
                "applied": bool(payload.get("model_alpha_applied", False)),
            }

        components = {
            "EventBus": "OK" if events else ("IDLE" if runtime_fresh else "NO_DATA"),
            "RiskManager": "OK",
            "Execution": "OK" if any(
                e.get("event_type") == "ORDER_SUBMITTED" for e in events
            ) else "IDLE",
            "DataFeed": "OK" if any(
                e.get("event_type") == "HEARTBEAT" for e in events
            ) else ("OK" if runtime_fresh else "NO_DATA"),
        }

        return {
            "mode": system_mode,
            "regime": regime,
            "phase": run_mode,
            "components": components,
            "last_heartbeat": last_heartbeat,
            "data_source": runtime_source,
            "ledger_session_id": ledger_meta.get("session_id", ""),
            "ledger_fresh": ledger_meta.get("fresh", False),
            "runtime_updated_at": runtime.get("updated_at", ""),
            "drl": drl_state,
            "sentiment": sentiment_state,
            "learned_exec": learned_exec_state,
            "composite_toxicity": composite_toxicity_state,
            "g6": g6_state,
            "aggressive_allocator": aggressive_allocator_state,
            "lead_lag": lead_lag_state,
            "model_alpha": model_alpha_state,
            "status_note": (
                "using runtime fallback state" if runtime_source == "runtime-fallback"
                else ("ledger/runtime state stale" if runtime_source == "stale" else "fresh ledger session")
            ),
        }


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_header():
    st.set_page_config(
        page_title="HMATS Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🚀 HMATS Trading Dashboard")
    st.markdown("*Hierarchical Multi-Agent Trading System - Real-time Monitoring*")
    st.divider()


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")
        refresh_rate = st.selectbox(
            "Refresh Rate", options=[5, 10, 30, 60], index=1,
            format_func=lambda x: f"{x} seconds",
        )
        time_range = st.selectbox(
            "Time Range", options=[1, 4, 12, 24, 48], index=3,
            format_func=lambda x: f"Last {x} hours",
        )
        st.divider()
        st.markdown("### About\nHMATS v6.7\nPaper Trading Dashboard")
    return {"refresh_rate": refresh_rate, "time_range": time_range}


def render_position_panel(positions: Dict[str, Any]):
    st.subheader("📊 Current Positions")
    for symbol, pos in positions.items():
        side = pos.get("side", "FLAT")
        pnl = pos.get("unrealized_pnl", 0)
        st.metric(label=symbol, value=side, delta=f"${pnl:,.2f}")
        if pos.get("size", 0) != 0:
            st.caption(
                f"Size: {pos['size']:.6f} | "
                f"Entry: ${pos.get('entry_price', 0):,.2f} | "
                f"Notional: ${pos.get('notional', 0):,.0f} | "
                f"T{pos.get('tranche', 0)}"
            )
            if pos.get("last_fill_price"):
                st.caption(f"Last fill: ${pos['last_fill_price']:,.2f}")


def render_risk_panel(metrics: Dict[str, float]):
    st.subheader("[WARN]️ Risk Metrics")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Exposure", f"{metrics.get('exposure_pct', 0):.1%}")
    with cols[1]:
        pnl = metrics.get("session_pnl", 0)
        st.metric("Session P&L", f"${pnl:,.2f}", delta=f"{pnl:+,.2f}")
    with cols[2]:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    with cols[3]:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.0%}")
    with cols[4]:
        st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")


def render_regime_panel(status: Dict[str, Any]):
    st.subheader("🌊 Market Regime")
    mode = status.get("mode", "NORMAL")
    mode_icon = {"NORMAL": "🟢", "OPPORTUNITY": "🟡", "NO_TRADE": "🔴"}.get(mode, "⚪")
    regime = status.get("regime", "Unknown")
    run_mode = status.get("phase", "Unknown")
    hb = str(status.get("last_heartbeat", "N/A"))
    hb = hb[:19] if len(hb) > 19 else hb

    st.markdown(
        f"| | |\n|---|---|\n"
        f"| **System Mode** | {mode_icon} {mode} |\n"
        f"| **Regime** | {regime} |\n"
        f"| **Run Mode** | {run_mode} |\n"
        f"| **Last Heartbeat** | {hb} |"
    )


def render_system_health(components: Dict[str, str]):
    st.subheader("🏥 System Health")
    cols = st.columns(len(components))
    for i, (name, status) in enumerate(components.items()):
        with cols[i]:
            emoji = "ON" if status == "OK" else ("⏳" if status == "IDLE" else "OFF")
            st.metric(name, f"{emoji} {status}")


def render_events_table(events: List[Dict]):
    st.subheader("📋 Recent Events")
    if not events:
        st.info("No recent events")
        return

    rows = []
    for e in events[-50:]:
        payload = e.get("payload", {})
        event_type = e.get("event_type", "Unknown")
        asset = payload.get("asset", "-")

        if event_type == "ORDER_SUBMITTED":
            result = payload.get("result", {})
            details = (
                f"{result.get('side', '?')} {result.get('filled_size', 0):.4f} "
                f"@ ${result.get('filled_price', 0):,.2f}"
            )
        elif event_type == "HEARTBEAT":
            details = f"tick={payload.get('tick_id', '?')}"
        else:
            details = str(payload)[:80]

        rows.append({
            "Time": e.get("timestamp", "")[:19],
            "Type": event_type,
            "Asset": asset,
            "Details": details,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", height=300)


def render_pnl_chart(pnl_history: List[Dict]):
    st.subheader("💰 Cumulative P&L (net of fees)")
    if not pnl_history:
        st.info("No P&L data available yet")
        return

    df = pd.DataFrame(pnl_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    last_pnl = df["pnl"].iloc[-1]
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["pnl"],
        fill="tozeroy",
        fillcolor="rgba(0,204,150,0.3)" if last_pnl >= 0 else "rgba(239,83,80,0.3)",
        line=dict(color="#00CC96" if last_pnl >= 0 else "#EF5350", width=2),
        name="P&L",
        text=[f"{a}: ${p:,.2f}" for a, p in zip(df["asset"], df["pnl"])],
    ))
    fig.update_layout(
        height=300, template="plotly_dark", showlegend=False,
        margin=dict(l=50, r=50, t=10, b=30),
        yaxis_title="USD", xaxis_title="Time",
    )
    st.plotly_chart(fig, width="stretch")


# [P2-FIX] Execution Quality panel
EXECUTION_QUALITY_DIRS = [
    PROJECT_ROOT / "logs" / "execution_quality",
    PROJECT_ROOT / "data" / "execution_logs",
]


def render_execution_quality():
    """Render execution quality metrics from fill records."""
    st.subheader("Execution Quality")

    eq_dir = next((p for p in EXECUTION_QUALITY_DIRS if p.exists()), None)
    if eq_dir is None:
        st.info("No execution quality data yet. Waiting for first fills.")
        return

    jsonl_files = sorted(eq_dir.glob("execution_quality_*.jsonl"))
    if not jsonl_files:
        st.info("No execution quality data yet. Waiting for first fills.")
        return

    records = []
    for fpath in jsonl_files[-7:]:  # Last 7 days max
        try:
            for line in fpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception:
            continue

    if not records or not PANDAS_AVAILABLE:
        st.info("No execution quality records found.")
        return

    df = pd.DataFrame(records)

    if "realized_slippage_bps" not in df.columns:
        st.info("Execution records missing slippage data.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Fills", len(df))
    col2.metric("Avg Slippage", f"{df['realized_slippage_bps'].mean():+.1f} bps")
    col3.metric("Median Slippage", f"{df['realized_slippage_bps'].median():+.1f} bps")
    favorable_pct = (df["realized_slippage_bps"] < 0).mean() * 100
    col4.metric("Favorable %", f"{favorable_pct:.0f}%")

    # Per-asset breakdown
    if "asset" in df.columns:
        st.dataframe(
            df.groupby("asset")["realized_slippage_bps"]
            .agg(["count", "mean", "median"])
            .round(2)
            .rename(columns={"count": "fills", "mean": "avg_slip_bps", "median": "med_slip_bps"}),
            width="stretch",
        )


def render_trade_log(records: List[Dict]):
    st.subheader("📜 Trade Log (Latest Session)")
    fills = [r for r in records if r["entry_type"] == "FILL"]
    if not fills:
        st.info("No trades in current session")
        return

    rows = []
    for f in fills:
        d = f["data"]
        rows.append({
            "Time": f["timestamp"][:19],
            "Asset": f["asset"],
            "Side": d.get("side", "?"),
            "Size": f"{d.get('size', 0):.6f}",
            "Price": f"${d.get('price', 0):,.2f}",
            "Notional": f"${d.get('notional', 0):,.0f}",
            "Fee": f"${d.get('fee', 0):.2f}",
            "PnL": f"${d.get('realized_pnl', 0):.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", height=250)


def render_regime_panel(status: Dict[str, Any]):
    """Explicitly surface runtime fallback/stale state instead of silent zeros."""
    st.subheader("Market Regime")
    mode = status.get("mode", "NORMAL")
    mode_icon = {"NORMAL": "OK", "OPPORTUNITY": "WARN", "NO_TRADE": "STOP"}.get(mode, "?")
    regime = status.get("regime", "Unknown")
    run_mode = status.get("phase", "Unknown")
    hb = str(status.get("last_heartbeat", "N/A"))
    hb = hb[:19] if len(hb) > 19 else hb
    data_source = status.get("data_source", "unknown")
    session_id = status.get("ledger_session_id", "")
    status_note = status.get("status_note", "")
    drl_state = status.get("drl", {}) or {}
    sentiment_state = status.get("sentiment", {}) or {}
    learned_exec_state = status.get("learned_exec", {}) or {}
    composite_toxicity_state = status.get("composite_toxicity", {}) or {}
    g6_state = status.get("g6", {}) or {}
    aggressive_allocator_state = status.get("aggressive_allocator", {}) or {}
    lead_lag_state = status.get("lead_lag", {}) or {}
    model_alpha_state = status.get("model_alpha", {}) or {}
    sentiment_assets = sentiment_state.get("assets", {}) if isinstance(sentiment_state, dict) else {}
    sentiment_summary = ", ".join(
        f"{asset}:{payload.get('source', 'none')}/{payload.get('headline_count', 0)}/{payload.get('source_status', 'unavailable')}{'*' if payload.get('tradeable_alpha', False) else ''}"
        for asset, payload in sentiment_assets.items()
        if isinstance(payload, dict)
    ) or "none"
    learned_exec_summary = ", ".join(
        f"{asset}:{payload.get('mode', 'DISABLED')}/{payload.get('action', 'none')} "
        f"conf={payload.get('confidence', 0.0):.2f} aggr={payload.get('aggressiveness', 0.0):.2f} "
        f"off={payload.get('offset_bps', 0.0):.1f}bps{'*' if payload.get('applied', False) else ''}"
        for asset, payload in learned_exec_state.items()
        if isinstance(payload, dict)
    ) or "none"
    composite_toxicity_summary = ", ".join(
        f"{asset}:{payload.get('authority', 'DISABLED')} score={payload.get('score', 0.0):.2f} "
        f"{payload.get('dominant', 'none')}{'*' if payload.get('applied', False) else ''}"
        for asset, payload in composite_toxicity_state.items()
        if isinstance(payload, dict)
    ) or "none"
    g6_summary = ", ".join(
        f"{asset}:{payload.get('mode', 'DISABLED')} w={payload.get('weight', 0.0):.1%} "
        f"x{payload.get('multiplier', 1.0):.2f}{'*' if payload.get('applied', False) else ''}"
        for asset, payload in g6_state.items()
        if isinstance(payload, dict)
    ) or "none"
    aggressive_allocator_summary = ", ".join(
        f"{asset}:{payload.get('authority', 'DISABLED')} tier={payload.get('tier', 'NONE')} "
        f"score={payload.get('score', 0.0):.3f} cap={payload.get('cap', 0.0):.1%}{'*' if payload.get('applied', False) else ''}"
        for asset, payload in aggressive_allocator_state.items()
        if isinstance(payload, dict)
    ) or "none"
    lead_lag_summary = ", ".join(
        f"{asset}:{payload.get('authority', 'DISABLED')} edge={payload.get('edge', 0.0):+.1f}bps "
        f"conf={payload.get('confidence', 0.0):.2f} x{payload.get('multiplier', 1.0):.2f}{'*' if payload.get('applied', False) else ''}"
        for asset, payload in lead_lag_state.items()
        if isinstance(payload, dict)
    ) or "none"
    model_alpha_summary = ", ".join(
        f"{asset}:{payload.get('model_type', 'none')} conf={payload.get('confidence', 0.0):.2f} "
        f"w={payload.get('weight', 0.0):.2f} d={payload.get('delta', 0.0):+.3f}{'*' if payload.get('applied', False) else ''}"
        for asset, payload in model_alpha_state.items()
        if isinstance(payload, dict)
    ) or "none"
    drl_summary = (
        f"gate={drl_state.get('gate_level', 'DISABLED')} | "
        f"infer={drl_state.get('inference_mode', 'DISABLED')} | "
        f"promo={drl_state.get('promotion_level', 'DISABLED')} | "
        f"exec={drl_state.get('execution_authority', 'OFF')} | "
        f"models={drl_state.get('models_ready', 0)} | "
        f"impact={drl_state.get('trade_impact', 'NONE')} | "
        f"pnl={'YES' if drl_state.get('pnl_driver') else 'NO'} | "
        f"bootstrap={'YES' if drl_state.get('bootstrap_applied') else 'NO'}"
    )

    st.markdown(
        f"| | |\n|---|---|\n"
        f"| **System Mode** | {mode_icon} {mode} |\n"
        f"| **Regime** | {regime} |\n"
        f"| **Run Mode** | {run_mode} |\n"
        f"| **Last Heartbeat** | {hb} |\n"
        f"| **State Source** | {data_source} |\n"
        f"| **Ledger Session** | {session_id or '-'} |\n"
        f"| **DRL Runtime** | {drl_summary} |\n"
        f"| **Sentiment** | {sentiment_summary} |\n"
        f"| **Learned Exec** | {learned_exec_summary} |\n"
        f"| **Composite Toxicity** | {composite_toxicity_summary} |\n"
        f"| **G6 Sizing** | {g6_summary} |\n"
        f"| **Aggressive Alloc** | {aggressive_allocator_summary} |\n"
        f"| **Lead-Lag** | {lead_lag_summary} |\n"
        f"| **Model Alpha** | {model_alpha_summary} |\n"
        f"| **Status Note** | {status_note or '-'} |"
    )


# =============================================================================
# ASCII UI OVERRIDES
# =============================================================================

def render_header():
    st.set_page_config(
        page_title="HMATS Trading Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("HMATS Trading Dashboard")
    st.markdown("*Hierarchical Multi-Agent Trading System - Real-time Monitoring*")
    st.divider()


def render_sidebar():
    with st.sidebar:
        st.header("Controls")
        refresh_rate = st.selectbox(
            "Refresh Rate", options=[5, 10, 30, 60], index=1,
            format_func=lambda x: f"{x} seconds",
        )
        time_range = st.selectbox(
            "Time Range", options=[1, 4, 12, 24, 48], index=3,
            format_func=lambda x: f"Last {x} hours",
        )
        st.divider()
        st.markdown("### About\nHMATS v6.7\nPaper Trading Dashboard")
    return {"refresh_rate": refresh_rate, "time_range": time_range}


def render_position_panel(positions: Dict[str, Any]):
    st.subheader("Current Positions")
    for symbol, pos in positions.items():
        side = pos.get("side", "FLAT")
        pnl = pos.get("unrealized_pnl", 0)
        st.metric(label=symbol, value=side, delta=f"${pnl:,.2f}")
        if pos.get("size", 0) != 0:
            st.caption(
                f"Size: {pos['size']:.6f} | "
                f"Entry: ${pos.get('entry_price', 0):,.2f} | "
                f"Notional: ${pos.get('notional', 0):,.0f} | "
                f"T{pos.get('tranche', 0)}"
            )
            if pos.get("last_fill_price"):
                st.caption(f"Last fill: ${pos['last_fill_price']:,.2f}")


def render_risk_panel(metrics: Dict[str, float]):
    st.subheader("Risk Metrics")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Exposure", f"{metrics.get('exposure_pct', 0):.1%}")
    with cols[1]:
        pnl = metrics.get("session_pnl", 0)
        st.metric("Session P&L", f"${pnl:,.2f}", delta=f"{pnl:+,.2f}")
    with cols[2]:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    with cols[3]:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.0%}")
    with cols[4]:
        st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")


def render_system_health(components: Dict[str, str]):
    st.subheader("System Health")
    cols = st.columns(len(components))
    for i, (name, status) in enumerate(components.items()):
        with cols[i]:
            indicator = "OK" if status == "OK" else ("IDLE" if status == "IDLE" else "OFF")
            st.metric(name, indicator)


def render_events_table(events: List[Dict]):
    st.subheader("Recent Events")
    if not events:
        st.info("No recent events")
        return

    rows = []
    for e in events[-50:]:
        payload = e.get("payload", {})
        event_type = e.get("event_type", "Unknown")
        asset = payload.get("asset", "-")

        if event_type == "ORDER_SUBMITTED":
            result = payload.get("result", {})
            details = (
                f"{result.get('side', '?')} {result.get('filled_size', 0):.4f} "
                f"@ ${result.get('filled_price', 0):,.2f}"
            )
        elif event_type == "HEARTBEAT":
            details = f"tick={payload.get('tick_id', '?')}"
        else:
            details = str(payload)[:80]

        rows.append({
            "Time": e.get("timestamp", "")[:19],
            "Type": event_type,
            "Asset": asset,
            "Details": details,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", height=300)


def render_pnl_chart(pnl_history: List[Dict]):
    st.subheader("Cumulative P&L (net of fees)")
    if not pnl_history:
        st.info("No P&L data available yet")
        return

    df = pd.DataFrame(pnl_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    last_pnl = df["pnl"].iloc[-1]
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["pnl"],
        fill="tozeroy",
        fillcolor="rgba(0,204,150,0.3)" if last_pnl >= 0 else "rgba(239,83,80,0.3)",
        line=dict(color="#00CC96" if last_pnl >= 0 else "#EF5350", width=2),
        name="P&L",
        text=[f"{a}: ${p:,.2f}" for a, p in zip(df["asset"], df["pnl"])],
    ))
    fig.update_layout(
        height=300, template="plotly_dark", showlegend=False,
        margin=dict(l=50, r=50, t=10, b=30),
        yaxis_title="USD", xaxis_title="Time",
    )
    st.plotly_chart(fig, width="stretch")


def render_trade_log(records: List[Dict]):
    st.subheader("Trade Log (Latest Session)")
    fills = [r for r in records if r["entry_type"] == "FILL"]
    if not fills:
        st.info("No trades in current session")
        return

    rows = []
    for f in fills:
        d = f["data"]
        rows.append({
            "Time": f["timestamp"][:19],
            "Asset": f["asset"],
            "Side": d.get("side", "?"),
            "Size": f"{d.get('size', 0):.6f}",
            "Price": f"${d.get('price', 0):,.2f}",
            "Notional": f"${d.get('notional', 0):,.0f}",
            "Fee": f"${d.get('fee', 0):.2f}",
            "PnL": f"${d.get('realized_pnl', 0):.2f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", height=250)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    if not STREAMLIT_AVAILABLE:
        print("Install: pip install streamlit plotly pandas")
        return

    render_header()
    controls = render_sidebar()

    loader = DashboardDataLoader()
    positions = loader.load_positions()
    risk_metrics = loader.load_risk_metrics()
    system_status = loader.load_system_status()
    events = loader.load_recent_events(hours=controls.get("time_range", 24))
    pnl_history = loader.load_pnl_history()
    session_records = loader.load_trade_records(hours=controls.get("time_range", 24))

    # Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        render_pnl_chart(pnl_history)
    with col2:
        render_position_panel(positions)
        render_regime_panel(system_status)
        render_system_health(system_status.get("components", {}))

    st.divider()
    render_risk_panel(risk_metrics)

    st.divider()
    render_trade_log(session_records)

    # [P2-FIX] Execution quality panel (data from ExecutionQualityLogger JSONL)
    st.divider()
    render_execution_quality()

    st.divider()
    render_events_table(events)

    # Auto-refresh
    refresh_rate = controls.get("refresh_rate", 10)
    st.markdown(f"*Auto-refresh in {refresh_rate} seconds*")
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        main()
    else:
        print("Install: pip install streamlit plotly pandas")
        print("Run: streamlit run dashboard/app.py")

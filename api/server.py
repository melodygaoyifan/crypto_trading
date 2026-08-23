"""
HMATS v6.8.0 — Read-Only Operational API
=========================================
Internal status API for monitoring. No trade-execution endpoints.

Reads from:
  - data/dashboard_state.json  (engine writes each tick)
  - data/paper_positions.json  (engine writes on fill / periodic)
  - logs/                      (structured logs)

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8080
"""

import json
import math
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = Path(os.environ.get("HMATS_BASE_DIR", Path(__file__).resolve().parent.parent))
_DATA = Path(os.environ.get("HMATS_STATE_DIR", _BASE / "data"))
_LOGS = Path(os.environ.get("HMATS_LOG_DIR", _BASE / "logs"))

DASHBOARD_STATE = _DATA / "dashboard_state.json"
PAPER_POSITIONS = _DATA / "paper_positions.json"
STEP15_STATUS = _DATA / "step15_status.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_cache: Dict[str, Any] = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL = 5  # seconds


def _read_json(path: Path, cache_key: str) -> Dict:
    """Read JSON with 5-second memory cache."""
    now = time.time()
    if cache_key in _cache and (now - _cache_ts.get(cache_key, 0)) < CACHE_TTL:
        return _cache[cache_key]
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _cache[cache_key] = data
        _cache_ts[cache_key] = now
        return data
    except (json.JSONDecodeError, OSError):
        return _cache.get(cache_key, {})


def _dashboard() -> Dict:
    return _read_json(DASHBOARD_STATE, "dashboard")


def _positions_file() -> Dict:
    return _read_json(PAPER_POSITIONS, "positions")


def _is_fresh(state: Dict, max_age_seconds: int = 600) -> bool:
    """Check if state was updated within max_age_seconds."""
    ts = state.get("updated_at") or state.get("saved_at", "")
    if not ts:
        return False
    try:
        if isinstance(ts, str):
            # [P323] Tolerate the malformed "...+00:00Z" (double tz marker) a
            # writer can emit via `isoformat() + "Z"`. The naive
            # replace("Z", "+00:00") turns it into "+00:00+00:00", which
            # fromisoformat rejects — so freshness read False FOREVER rather
            # than reporting a stale file. Strip a trailing Z only when the
            # string already carries an explicit offset. Belt-and-braces with
            # the writer fix (main.py): files already on the volume keep the
            # old shape until they are next rewritten.
            ts = ts.strip()
            if ts.endswith("Z") and ("+00:00" in ts[:-1] or "+" in ts[:-1]):
                ts = ts[:-1]
            ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
        else:
            return False
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return age < max_age_seconds
    except Exception as _e:  # noqa: silent-swallow
        # [P105 2026-04-27] Was bare except: return False — health
        # endpoint reported "stale" silently on any parse failure
        # (timestamp shape change, naive vs aware, etc.). Surface so
        # operator can see a parse-failure pattern even if the answer
        # (False=stale) is correct conservatively.
        import logging as _log
        _log.getLogger(__name__).warning(
            f"[API] _is_fresh parse failed ({type(_e).__name__}: {_e}); "
            f"returning stale=True conservatively"
        )
        return False


# ---------------------------------------------------------------------------
# [P383] Equity / drawdown / positions views over dashboard_state.json
# ---------------------------------------------------------------------------
# The export contract (main.py::_export_dashboard_state, P383):
#   equity          COMBINED kraken + sleeve, same denomination as peak_equity
#                   (an unreadable sleeve half serves the last-known COMBINED
#                   value — P261's rule — never the partial Kraken figure)
#   kraken_equity   float
#   sleeve_equity   float | null
#   equity_valid    bool — both halves fresh
#   equity_basis    "combined" | "kraken_only" (the latter only on a first-ever
#                   boot with no combined reading)
#   sleeve_positions   list of {asset, venue:"coinbase", signed_contracts,
#                      entry_vwap, current_price} from the RECONCILED book
#   sleeve_reconcile_ok  bool
#
# Before P383 the file carried `equity` = account_sync.get_equity(), which is
# KRAKEN-ONLY (~$0.40 since the June flatten) while `peak_equity` was the
# COMBINED peak (~$10.9k, P351) — so /pnl/summary computed a ~99.99% drawdown
# from two numbers in different denominations, /status served equity ~$0.40,
# and /positions/current counted 0 positions while the sleeve held several.
# Every helper below keys on the PRESENCE of the new keys: a pre-P383 file
# must read as "unverified", never as a fabricated figure (P2).
_P383_LEGACY_NOTE = "equity/peak denomination unverified (pre-P383 export)"


def _num(v: Any) -> Optional[float]:
    """A finite number or None. Never coerces absence/garbage to 0 (P2)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _equity_view(state: Dict) -> Dict[str, Any]:
    """equity + its provenance, straight off the export, absence = null."""
    basis = state.get("equity_basis")
    has_contract = "equity_basis" in state
    view: Dict[str, Any] = {
        "equity": _num(state.get("equity")),
        "kraken_equity": _num(state.get("kraken_equity")),
        "sleeve_equity": _num(state.get("sleeve_equity")),
        "equity_valid": state.get("equity_valid") if isinstance(
            state.get("equity_valid"), bool) else None,
        "equity_basis": basis if isinstance(basis, str) else None,
        "peak_equity": _num(state.get("peak_equity")),
    }
    notes: List[str] = []
    if not has_contract:
        notes.append(_P383_LEGACY_NOTE + "; `equity` may be the Kraken-only "
                     "figure, not the combined book")
    elif basis != "combined":
        notes.append(f"equity_basis={basis!r}: no combined reading yet")
    if view["equity_valid"] is False:
        notes.append("equity_valid=false: one half is stale — equity is the "
                     "last-known combined value")
    view["note"] = "; ".join(notes) if notes else None
    return view


def _drawdown_view(state: Dict) -> Tuple[Optional[float], Optional[str]]:
    """(drawdown_pct, note). Computed ONLY from a same-denomination pair —
    `equity_basis == "combined"` — else null with the reason. The pre-P383
    arithmetic produced ~99.99% from Kraken-only equity vs the combined peak;
    a null with a note is the honest reading of that file, a 99% is not."""
    if "equity_basis" not in state:
        return None, _P383_LEGACY_NOTE
    basis = state.get("equity_basis")
    if basis != "combined":
        return None, (f"equity_basis={basis!r}: equity and peak_equity are "
                      f"not a same-denomination pair")
    eq = _num(state.get("equity"))
    peak = _num(state.get("peak_equity"))
    if eq is None or peak is None or peak <= 0:
        return None, "equity or peak_equity unreadable"
    dd = round(max((peak - eq) / peak * 100.0, 0.0), 2)
    note = None
    if state.get("equity_valid") is False:
        note = ("equity_valid=false: drawdown from the last-known combined "
                "equity, one half is stale")
    return dd, note


def _sleeve_positions_view(state: Dict) -> Tuple[Optional[Dict[str, Dict]],
                                                 Optional[bool], Optional[str]]:
    """(positions-by-asset | None, sleeve_reconcile_ok | None, note).
    None (not {}) when the export carries no `sleeve_positions` key — an
    absent book is not an empty book."""
    if "sleeve_positions" not in state:
        return None, None, ("sleeve_positions absent from export (pre-P383); "
                            "the Coinbase sleeve book is NOT counted here")
    raw = state.get("sleeve_positions")
    rec_ok = state.get("sleeve_reconcile_ok")
    rec_ok = rec_ok if isinstance(rec_ok, bool) else None
    if not isinstance(raw, list):
        # present but null/garbage = the sleeve book could not be read this
        # tick; that is not an empty book either (P2)
        return None, rec_ok, ("sleeve_positions unreadable in export; the "
                              "Coinbase sleeve book is NOT counted here")
    out: Dict[str, Dict] = {}
    for sp in raw:
        if not isinstance(sp, dict):
            continue
        asset = sp.get("asset")
        if not isinstance(asset, str) or not asset:
            continue
        ct = _num(sp.get("signed_contracts"))
        if ct is not None and abs(ct) < 1e-9:
            continue  # flat is not a position
        out[asset] = {
            "venue": sp.get("venue") or "coinbase",
            "direction": ("UNKNOWN" if ct is None
                          else ("LONG" if ct > 0 else "SHORT")),
            "signed_contracts": ct,
            "entry_vwap": _num(sp.get("entry_vwap")),
            "current_price": _num(sp.get("current_price")),
        }
    note = None
    if rec_ok is False:
        note = ("sleeve_reconcile_ok=false: sleeve positions are the "
                "last-known snapshot, not a fresh venue read")
    elif rec_ok is None:
        note = "sleeve_reconcile_ok absent from export"
    return out, rec_ok, note


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HMATS Operational API",
    version="6.8.0",
    description="Read-only internal monitoring API. No trade execution.",
    docs_url="/docs",
    redoc_url=None,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness + freshness check.

    [FIX 2026-04-22] max_age_seconds raised 600 -> 15000. Engine writes
    dashboard_state.json once per 4H tick (14400s cadence), so a 600s
    threshold guaranteed "degraded" between ticks, spamming 503s. 15000s
    = 4H + 10min buffer lets us detect truly-stuck ticks without false
    alarms during normal inter-tick waits.
    """
    state = _dashboard()
    fresh = _is_fresh(state, max_age_seconds=15000)
    positions = _positions_file()
    pos_fresh = _is_fresh(positions, max_age_seconds=15000)

    status = "healthy" if (fresh or pos_fresh) else "degraded"
    code = 200 if status == "healthy" else 503

    return JSONResponse(
        status_code=code,
        content={
            "status": status,
            "engine_state_fresh": fresh,
            "positions_fresh": pos_fresh,
            "updated_at": state.get("updated_at", ""),
            "tick_count": state.get("tick_count", 0),
            "round": state.get("round", 0),
            "mode": state.get("mode", "UNKNOWN"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.get("/decision-trace")
def decision_trace(hours: int = 24):
    """[P111 Tier1#3 2026-04-27] Gate-rejection histogram + abstain
    detection over the last N hours. Wraps decision_trace_watcher.py
    so dashboards/operators can poll a single endpoint instead of
    SSH+grep workflow.

    Surfaces:
      - DECIDE_ABSTAIN streak length + percentage
      - per-agent abstain rate (catches "kraken_quant 100% dead")
      - top veto-reason histogram
      - PERMANENT_FAILURE outcome count
      - operator-actionable warnings list
    """
    try:
        # Import here to avoid hard dependency at module-load time;
        # the watcher script is the single source of truth for analysis logic.
        import sys as _sys
        from pathlib import Path as _Path
        _scripts_dir = _Path(__file__).resolve().parents[1] / "scripts"
        if str(_scripts_dir) not in _sys.path:
            _sys.path.insert(0, str(_scripts_dir))
        from decision_trace_watcher import (  # type: ignore[import-not-found]
            _resolve_attr_dir, _iter_recent_records,
            analyze_signals, analyze_outcomes, detect_anomalies,
        )
    except Exception as _e:
        return {
            "error": "decision_trace_watcher unavailable",
            "detail": f"{type(_e).__name__}: {_e}",
        }
    try:
        attr_dir = _resolve_attr_dir()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        signals = _iter_recent_records(attr_dir, "signals_*.jsonl", cutoff, 5000)
        outcomes = _iter_recent_records(attr_dir, "outcomes_*.jsonl", cutoff, 5000)
        sig_summary = analyze_signals(signals)
        out_summary = analyze_outcomes(outcomes)
        warnings = detect_anomalies(sig_summary, out_summary, hours)
        return {
            "hours_lookback": hours,
            "signals": sig_summary,
            "outcomes": out_summary,
            "warnings": warnings,
            "warning_count": len(warnings),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as _e:
        import logging as _log
        _log.getLogger(__name__).warning(
            f"[API] /decision-trace failed ({type(_e).__name__}: {_e})"
        )
        return {"error": str(_e)}


@app.get("/status")
def status():
    """Full system status summary.

    [P383] equity is reported with its provenance (kraken / sleeve halves,
    validity, basis). Pre-P383 exports carry none of the new keys: those read
    as null with a note, never as a fabricated zero.
    """
    state = _dashboard()
    ev = _equity_view(state)
    sleeve_pos, sleeve_ok, _ = _sleeve_positions_view(state)
    return {
        "mode": state.get("mode", "UNKNOWN"),
        "risk_profile": state.get("risk_profile", ""),
        "round": state.get("round", 0),
        "tick_count": state.get("tick_count", 0),
        "equity": ev["equity"],
        "kraken_equity": ev["kraken_equity"],
        "sleeve_equity": ev["sleeve_equity"],
        "equity_valid": ev["equity_valid"],
        "equity_basis": ev["equity_basis"],
        "cumulative_pnl": state.get("cumulative_pnl", 0),
        "peak_equity": ev["peak_equity"],
        # Kraken runtime count as before; the sleeve's is reported beside it.
        "position_count": state.get("position_count", 0),
        "sleeve_position_count": (None if sleeve_pos is None
                                  else len(sleeve_pos)),
        "sleeve_reconcile_ok": sleeve_ok,
        "note": ev["note"],
        "updated_at": state.get("updated_at", ""),
        "fresh": _is_fresh(state),
    }


@app.get("/signals/latest")
def signals_latest():
    """Latest per-asset signal state from dashboard export."""
    state = _dashboard()
    assets = state.get("assets", {})
    result = {}
    for asset, data in assets.items():
        if not isinstance(data, dict):
            continue
        result[asset] = {
            "regime": data.get("regime", ""),
            "regime_confidence": data.get("regime_confidence", 0),
            "quant_direction": data.get("quant_direction", 0),
            "quant_confidence": data.get("quant_confidence", 0),
            "strategy": data.get("strategy", ""),
            "alpha_est_bps": data.get("alpha_est_bps", 0),
            "alpha_gate_passed": data.get("alpha_gate_passed", False),
            "vpin": data.get("vpin", 0),
            "price": data.get("price", 0),
        }
    return {"assets": result, "updated_at": state.get("updated_at", "")}


@app.get("/regime/current")
def regime_current():
    """Current regime per asset."""
    state = _dashboard()
    assets = state.get("assets", {})
    result = {}
    for asset, data in assets.items():
        if not isinstance(data, dict):
            continue
        result[asset] = {
            "regime": data.get("regime", "UNKNOWN"),
            "confidence": data.get("regime_confidence", 0),
            "power": data.get("regime_power", 0),
            "allow_short": data.get("allow_short", True),
        }
    return {"regimes": result, "updated_at": state.get("updated_at", "")}


@app.get("/positions/current")
def positions_current():
    """Current open positions — Kraken book + Coinbase sleeve book.

    [P383] `positions` is the Kraken runtime book as before (each entry now
    tagged venue="kraken"); `sleeve_positions` is the sleeve's reconciled book
    from the dashboard export (venue="coinbase"); `count` covers BOTH. A
    pre-P383 export has no sleeve key: `sleeve_positions` is null, `count` is
    Kraken-only and `count_includes_sleeve` says so.
    """
    raw = _positions_file()
    positions = raw.get("positions", {})
    result = {}
    for asset, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        exposure = float(pos.get("exposure", 0) or 0)
        if abs(exposure) < 1e-6:
            continue
        result[asset] = {
            "venue": "kraken",
            "direction": "LONG" if pos.get("direction", 0) > 0 else "SHORT",
            "exposure_pct": round(exposure * 100, 2),
            "entry_price": pos.get("entry_price", 0),
            "notional": pos.get("notional", 0),
            "tranche": pos.get("tranche_level", pos.get("tranche", 0)),
            "strategy": pos.get("strategy", ""),
            "entry_time": pos.get("entry_time", ""),
        }

    # Supplement with dashboard state for mark prices
    state = _dashboard()
    dashboard_positions = state.get("positions", {})
    if isinstance(dashboard_positions, dict):
        for asset, dp in dashboard_positions.items():
            if asset in result and isinstance(dp, dict):
                result[asset]["mark_price"] = dp.get("mark_price", 0)

    sleeve_pos, sleeve_ok, sleeve_note = _sleeve_positions_view(state)
    includes_sleeve = sleeve_pos is not None
    return {
        "positions": result,
        "sleeve_positions": sleeve_pos,
        "count": len(result) + (len(sleeve_pos) if includes_sleeve else 0),
        "kraken_count": len(result),
        "sleeve_count": (len(sleeve_pos) if includes_sleeve else None),
        "count_includes_sleeve": includes_sleeve,
        "sleeve_reconcile_ok": sleeve_ok,
        "note": sleeve_note,
        "saved_at": raw.get("saved_at", ""),
        "cumulative_pnl": raw.get("cumulative_pnl", 0),
        "peak_equity": raw.get("peak_equity", 0),
    }


@app.get("/pnl/summary")
def pnl_summary():
    """PnL and performance summary.

    [P383] drawdown_pct is computed ONLY when the export says `equity` and
    `peak_equity` share a denomination (`equity_basis == "combined"`). The old
    arithmetic divided Kraken-only equity (~$0.40) by the combined peak and
    reported ~99.99%; a pre-P383 file now yields null + drawdown_note.
    """
    state = _dashboard()
    raw = _positions_file()
    realized = state.get("realized_pnl", {})
    if not isinstance(realized, dict):
        realized = {}
    ev = _equity_view(state)
    dd, dd_note = _drawdown_view(state)
    peak = ev["peak_equity"]
    if peak is None and "peak_equity" not in state:
        peak = _num(raw.get("peak_equity"))
    return {
        "equity": ev["equity"],
        "kraken_equity": ev["kraken_equity"],
        "sleeve_equity": ev["sleeve_equity"],
        "equity_valid": ev["equity_valid"],
        "equity_basis": ev["equity_basis"],
        "cumulative_pnl": state.get("cumulative_pnl", raw.get("cumulative_pnl", 0)),
        "peak_equity": peak,
        "drawdown_pct": dd,
        "drawdown_note": dd_note,
        "note": ev["note"],
        "total_trades": realized.get("total_trades", 0),
        "wins": realized.get("wins", 0),
        "losses": realized.get("losses", 0),
        "win_rate": realized.get("win_rate", 0),
        "avg_win": realized.get("avg_win", 0),
        "avg_loss": realized.get("avg_loss", 0),
        "updated_at": state.get("updated_at", ""),
    }


@app.get("/logs/recent")
def logs_recent(lines: int = 50):
    """Recent engine log lines (capped at 200)."""
    lines = min(lines, 200)
    log_paths = [
        _LOGS / "hmats_stderr.log",
        _LOGS / "paper_run_stderr.log",
        _LOGS / "hmats.log",
    ]
    for lp in log_paths:
        if lp.exists():
            try:
                raw = lp.read_bytes()[-50000:]
                content = raw.decode("utf-8", errors="replace")
                all_lines = content.splitlines()
                return {"file": lp.name, "lines": all_lines[-lines:]}
            except Exception as _e:  # noqa: silent-swallow
                # [P105 2026-04-27] Was bare except: continue — read
                # errors silently fell through to next log file. Now
                # logged so operator can see permission/encoding issues.
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"[API] /logs read of {lp.name} failed "
                    f"({type(_e).__name__}: {_e}); trying next log file"
                )
                continue
    return {"file": "", "lines": []}


@app.get("/drl/status")
def drl_status():
    """DRL agent runtime status."""
    state = _dashboard()
    drl = state.get("drl", {})
    if not isinstance(drl, dict):
        drl = {}
    return {
        "gate": drl.get("gate", "UNKNOWN"),
        "inference": drl.get("inference", "DISABLED"),
        "promotion": drl.get("promotion", ""),
        "execution": drl.get("execution", ""),
        "models_ready": drl.get("models_ready", {}),
        "trade_impact": drl.get("trade_impact", {}),
        "updated_at": state.get("updated_at", ""),
    }


@app.get("/sentiment/latest")
def sentiment_latest():
    """Sentiment L1 state."""
    state = _dashboard()
    sentiment = state.get("sentiment", {})
    if not isinstance(sentiment, dict):
        sentiment = {}
    # Also extract per-asset sentiment from assets
    assets = state.get("assets", {})
    per_asset = {}
    for asset, data in assets.items():
        if not isinstance(data, dict):
            continue
        per_asset[asset] = {
            "fear_greed": data.get("fear_greed", None),
            "sentiment_direction": data.get("sentiment_direction", 0),
            "sentiment_zscore": data.get("sentiment_zscore", 0),
        }
    return {
        "global": sentiment,
        "per_asset": per_asset,
        "updated_at": state.get("updated_at", ""),
    }

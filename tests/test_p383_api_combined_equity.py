"""
[P383] The dashboard API reported Kraken-only equity against the COMBINED peak.

`main.py::_export_dashboard_state` wrote `equity` = account_sync.get_equity()
— KRAKEN-ONLY, ~$0.40 since the June flatten — while `peak_equity` is the
COMBINED Kraken+sleeve peak (~$10.9k, P351). So api/server.py:

  /pnl/summary   drawdown_pct = (peak - equity)/peak  ≈ 99.99%   (fiction)
  /status        equity ≈ $0.40                                    (the wrong book)
  /positions/current  count 0 while the Coinbase sleeve held several

This file pins the consumer side of the P383 export contract:
  equity (COMBINED, same denomination as peak_equity), kraken_equity,
  sleeve_equity, equity_valid, equity_basis ("combined"|"kraken_only"),
  sleeve_positions (list of {asset, venue, signed_contracts, entry_vwap,
  current_price}), sleeve_reconcile_ok.

Three exports are driven through the REAL endpoint handlers:
  (i)   a post-P383 file            -> combined numbers, a real drawdown
  (ii)  a pre-P383 file (no new keys) -> drawdown null + note, NO 99% figure
  (iii) a sleeve-unreadable file     -> last-known combined, flagged, no zero

The handlers are compiled OUT OF THE AST (the P323 pattern) because fastapi is
absent in CI, and a test that skips where it matters is not a guard (P194).
"""
from __future__ import annotations

import ast
import io
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SERVER = REPO / "api" / "server.py"

_HELPERS = ("_num", "_equity_view", "_drawdown_view",
            "_sleeve_positions_view", "_is_fresh")
_ENDPOINTS = ("status", "positions_current", "pnl_summary", "health")


def _load_server(dashboard: Dict, positions_file: Dict) -> Dict[str, Any]:
    """Compile the helpers + endpoint handlers out of api/server.py without
    importing it. Decorators (@app.get) are stripped; the file readers are
    replaced by closures over the supplied dicts."""
    tree = ast.parse(io.open(SERVER, encoding="utf-8").read())
    body: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_P383_LEGACY_NOTE"
                for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in (
                _HELPERS + _ENDPOINTS):
            node.decorator_list = []
            body.append(node)
    found = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    missing = set(_HELPERS + _ENDPOINTS) - found
    assert not missing, f"api/server.py no longer defines {sorted(missing)}"
    ns: Dict[str, Any] = {
        "math": math, "datetime": datetime, "timezone": timezone,
        "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple,
        "Any": Any,
        "_dashboard": lambda: dashboard,
        "_positions_file": lambda: positions_file,
        # /health builds a JSONResponse; give it a stand-in that records.
        "JSONResponse": lambda status_code, content: {"status_code": status_code,
                                                      "content": content},
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "<server>", "exec"), ns)
    return ns


_NOW = datetime.now(timezone.utc).isoformat()


def _post_p383_state(**over) -> Dict:
    st = {
        "updated_at": _NOW, "mode": "LIVE", "tick_count": 12, "round": 3,
        "equity": 10_874.25,
        "kraken_equity": 0.40,
        "sleeve_equity": 10_873.85,
        "equity_valid": True,
        "equity_basis": "combined",
        "peak_equity": 10_927.00,
        "cumulative_pnl": -52.75,
        "position_count": 0,
        "positions": {},
        "sleeve_positions": [
            {"asset": "ETH", "venue": "coinbase", "signed_contracts": 4.0,
             "entry_vwap": 2251.3, "current_price": 2276.18},
            {"asset": "SOL", "venue": "coinbase", "signed_contracts": 1.0,
             "entry_vwap": 91.2, "current_price": 94.94},
        ],
        "sleeve_reconcile_ok": True,
        "realized_pnl": {"total_trades": 39, "wins": 20, "losses": 19},
    }
    st.update(over)
    return st


def _pre_p383_state() -> Dict:
    """Exactly what the live file carried before P383: Kraken-only `equity`
    against the combined peak, no provenance keys, no sleeve book."""
    return {
        "updated_at": _NOW, "mode": "LIVE", "tick_count": 12, "round": 3,
        "equity": 0.398431766369,
        "peak_equity": 10_927.00,
        "cumulative_pnl": -52.75,
        "position_count": 0,
        "positions": {},
        "realized_pnl": {"total_trades": 39},
    }


def _sleeve_unreadable_state() -> Dict:
    """The sleeve half could not be read this tick: per P261 the export keeps
    the last-known COMBINED equity, marks it invalid, and the sleeve book is
    the last-known snapshot (reconcile_ok False)."""
    return _post_p383_state(
        sleeve_equity=None, equity_valid=False, equity_basis="combined",
        sleeve_reconcile_ok=False,
    )


_EMPTY_POSITIONS_FILE = {"saved_at": _NOW, "positions": {},
                         "cumulative_pnl": -52.75, "peak_equity": 10_927.00}


# ---------------------------------------------------------------------------
# (i) post-P383 export
# ---------------------------------------------------------------------------
class TestPostP383Export:

    def test_pnl_summary_drawdown_is_the_same_denomination_pair(self):
        ns = _load_server(_post_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["pnl_summary"]()
        assert out["equity"] == pytest.approx(10_874.25)
        assert out["peak_equity"] == pytest.approx(10_927.00)
        assert out["equity_basis"] == "combined"
        # (10927 - 10874.25) / 10927 * 100 = 0.4828..
        assert out["drawdown_pct"] == pytest.approx(0.48, abs=0.01)
        assert out["drawdown_note"] is None
        assert out["note"] is None

    def test_status_reports_every_equity_half_and_its_basis(self):
        ns = _load_server(_post_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["status"]()
        assert out["equity"] == pytest.approx(10_874.25)
        assert out["kraken_equity"] == pytest.approx(0.40)
        assert out["sleeve_equity"] == pytest.approx(10_873.85)
        assert out["equity_valid"] is True
        assert out["equity_basis"] == "combined"
        assert out["sleeve_position_count"] == 2
        assert out["sleeve_reconcile_ok"] is True
        assert out["note"] is None
        assert out["fresh"] is True

    def test_positions_current_counts_both_books_and_tags_venue(self):
        pos_file = dict(_EMPTY_POSITIONS_FILE)
        pos_file["positions"] = {
            "BTC": {"exposure": 0.1, "direction": 1, "entry_price": 64_000,
                    "notional": 640, "strategy": "regimebook"},
        }
        ns = _load_server(_post_p383_state(), pos_file)
        out = ns["positions_current"]()
        assert out["positions"]["BTC"]["venue"] == "kraken"
        assert set(out["sleeve_positions"]) == {"ETH", "SOL"}
        eth = out["sleeve_positions"]["ETH"]
        assert eth == {"venue": "coinbase", "direction": "LONG",
                       "signed_contracts": 4.0, "entry_vwap": 2251.3,
                       "current_price": 2276.18}
        assert out["count"] == 3
        assert out["kraken_count"] == 1
        assert out["sleeve_count"] == 2
        assert out["count_includes_sleeve"] is True
        assert out["sleeve_reconcile_ok"] is True
        assert out["note"] is None

    def test_a_short_sleeve_position_and_a_flat_row_are_classified(self):
        st = _post_p383_state(sleeve_positions=[
            {"asset": "BTC", "venue": "coinbase", "signed_contracts": -2,
             "entry_vwap": 64_100.0, "current_price": 63_900.0},
            {"asset": "ETH", "venue": "coinbase", "signed_contracts": 0.0,
             "entry_vwap": None, "current_price": 2276.0},
        ])
        ns = _load_server(st, _EMPTY_POSITIONS_FILE)
        out = ns["positions_current"]()
        assert list(out["sleeve_positions"]) == ["BTC"], \
            "a 0-contract row is flat, not a position"
        assert out["sleeve_positions"]["BTC"]["direction"] == "SHORT"
        assert out["count"] == 1


# ---------------------------------------------------------------------------
# (ii) pre-P383 export — the file as it was when the defect was live
# ---------------------------------------------------------------------------
class TestPreP383ExportIsNotTurnedIntoAFigure:

    def test_pnl_summary_refuses_the_denomination_mismatch(self):
        ns = _load_server(_pre_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["pnl_summary"]()
        assert out["drawdown_pct"] is None
        assert out["drawdown_note"] == (
            "equity/peak denomination unverified (pre-P383 export)")
        assert out["equity_basis"] is None
        assert out["kraken_equity"] is None
        assert out["sleeve_equity"] is None
        assert out["equity_valid"] is None
        assert "pre-P383" in out["note"]

    def test_no_99_percent_figure_anywhere_in_the_summary(self):
        """The defect's signature: (10927 - 0.40)/10927 = 99.996%. It must
        not be reachable from a pre-P383 file through ANY field."""
        ns = _load_server(_pre_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["pnl_summary"]()
        for k, v in out.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                assert not (99.0 <= float(v) <= 100.0), (
                    f"{k}={v}: the fabricated drawdown is back")

    def test_status_nulls_the_missing_keys_and_says_why(self):
        ns = _load_server(_pre_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["status"]()
        # the file's own equity is still reported — it is what the file says —
        # but every provenance key is null and the note names the ambiguity
        assert out["equity"] == pytest.approx(0.398431766369)
        assert out["kraken_equity"] is None
        assert out["sleeve_equity"] is None
        assert out["equity_valid"] is None
        assert out["equity_basis"] is None
        assert out["sleeve_position_count"] is None
        assert out["sleeve_reconcile_ok"] is None
        assert "pre-P383" in out["note"] and "Kraken-only" in out["note"]

    def test_positions_current_does_not_fabricate_an_empty_sleeve(self):
        ns = _load_server(_pre_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["positions_current"]()
        assert out["sleeve_positions"] is None, \
            "an absent book must not read as an empty book (P2)"
        assert out["sleeve_count"] is None
        assert out["count_includes_sleeve"] is False
        assert out["count"] == 0
        assert out["sleeve_reconcile_ok"] is None
        assert "pre-P383" in out["note"]

    def test_an_absent_equity_is_null_not_zero(self):
        st = _pre_p383_state()
        del st["equity"]
        ns = _load_server(st, _EMPTY_POSITIONS_FILE)
        assert ns["status"]()["equity"] is None
        assert ns["pnl_summary"]()["equity"] is None


# ---------------------------------------------------------------------------
# (iii) sleeve unreadable this tick
# ---------------------------------------------------------------------------
class TestSleeveUnreadableExport:

    def test_equity_is_the_last_known_combined_and_flagged(self):
        ns = _load_server(_sleeve_unreadable_state(), _EMPTY_POSITIONS_FILE)
        out = ns["status"]()
        assert out["equity"] == pytest.approx(10_874.25), \
            "must serve the last-known COMBINED value, never the Kraken half"
        assert out["sleeve_equity"] is None
        assert out["equity_valid"] is False
        assert out["equity_basis"] == "combined"
        assert "equity_valid=false" in out["note"]

    def test_drawdown_is_still_same_denomination_but_annotated(self):
        ns = _load_server(_sleeve_unreadable_state(), _EMPTY_POSITIONS_FILE)
        out = ns["pnl_summary"]()
        assert out["drawdown_pct"] == pytest.approx(0.48, abs=0.01)
        assert "equity_valid=false" in out["drawdown_note"]
        assert out["drawdown_pct"] < 1.0

    def test_a_null_sleeve_book_is_unreadable_not_empty(self):
        """The parent's export may carry `sleeve_positions: null` if the
        sleeve half raised — present-but-null is NOT an empty book."""
        st = _post_p383_state(sleeve_positions=None, sleeve_reconcile_ok=False)
        ns = _load_server(st, _EMPTY_POSITIONS_FILE)
        out = ns["positions_current"]()
        assert out["sleeve_positions"] is None
        assert out["count_includes_sleeve"] is False
        assert out["sleeve_count"] is None
        assert "unreadable" in out["note"]

    @pytest.mark.parametrize("basis", ["kraken_only", "notional_fallback"])
    def test_a_non_combined_basis_never_yields_a_drawdown(self, basis):
        """The export's two non-combined bases (first-ever boot; both halves
        unreadable) have no same-denomination peak to compare against."""
        st = _post_p383_state(equity_basis=basis, equity=0.40,
                              sleeve_equity=None, equity_valid=False)
        ns = _load_server(st, _EMPTY_POSITIONS_FILE)
        out = ns["pnl_summary"]()
        assert out["drawdown_pct"] is None
        assert basis in out["drawdown_note"]
        assert out["equity_basis"] == basis
        assert basis in ns["status"]()["note"]

    def test_sleeve_positions_are_served_as_a_stale_snapshot(self):
        ns = _load_server(_sleeve_unreadable_state(), _EMPTY_POSITIONS_FILE)
        out = ns["positions_current"]()
        assert set(out["sleeve_positions"]) == {"ETH", "SOL"}
        assert out["count"] == 2
        assert out["sleeve_reconcile_ok"] is False
        assert "sleeve_reconcile_ok=false" in out["note"]


# ---------------------------------------------------------------------------
# the contract's edges, on the pure helpers
# ---------------------------------------------------------------------------
class TestDrawdownViewEdges:

    def test_kraken_only_basis_never_produces_a_drawdown(self):
        """First-ever boot: equity_basis="kraken_only" means no combined
        reading exists, so equity vs peak_equity is not a pair."""
        ns = _load_server({}, {})
        dd, note = ns["_drawdown_view"](
            {"equity_basis": "kraken_only", "equity": 0.40,
             "peak_equity": 10_927.0})
        assert dd is None
        assert "kraken_only" in note

    @pytest.mark.parametrize("peak", [0, -1, None, float("nan"), "10927"])
    def test_unreadable_peak_is_null_not_zero_or_inf(self, peak):
        ns = _load_server({}, {})
        dd, note = ns["_drawdown_view"](
            {"equity_basis": "combined", "equity": 100.0, "peak_equity": peak})
        assert dd is None and note

    def test_drawdown_floors_at_zero_at_a_new_high(self):
        ns = _load_server({}, {})
        dd, _ = ns["_drawdown_view"](
            {"equity_basis": "combined", "equity": 200.0, "peak_equity": 100.0})
        assert dd == 0.0

    def test_num_rejects_bools_strings_and_nonfinite(self):
        ns = _load_server({}, {})
        n = ns["_num"]
        assert n(True) is None and n("3") is None and n(None) is None
        assert n(float("inf")) is None
        assert n(3) == 3.0 and n(2.5) == 2.5


# ---------------------------------------------------------------------------
# /health and _is_fresh are untouched
# ---------------------------------------------------------------------------
class TestHealthIsUntouched:

    def test_health_does_not_consume_the_new_views(self):
        tree = ast.parse(io.open(SERVER, encoding="utf-8").read())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "health")
        called = {c.func.id for c in ast.walk(fn)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert not ({"_equity_view", "_drawdown_view",
                     "_sleeve_positions_view"} & called)

    def test_health_still_healthy_on_either_fresh_file(self):
        ns = _load_server(_pre_p383_state(), _EMPTY_POSITIONS_FILE)
        out = ns["health"]()
        assert out["status_code"] == 200
        assert out["content"]["status"] == "healthy"

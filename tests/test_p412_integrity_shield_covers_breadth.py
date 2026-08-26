"""[P412] The Kraken integrity shield must cover the breadth perp (XRP) it now
manages, or the P0 integrity check aborts XRP's tick every cycle
(get_orderbook(XRP/USD) -> None) — fail-safe (XRP stays inert) but XRP could
never trade even once routed. The shield's symbol set is now DERIVED from
config.assets so any asset in the tradeable universe is data-integrity-shielded
like the home trio. Surfaced live: after the P412 config deploy, XRP logged
`[P0 ABORT] [INTEGRITY] Data integrity check failed for XRP/USD: unreadable`
every tick because the shield was constructed with a hardcoded 3-symbol list.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _normalize():
    # _normalize_kraken_pair is a @staticmethod on the runner; import it without
    # constructing the (heavy) runner.
    import main
    return main.HMATSProductionRunner._normalize_kraken_pair


def test_home_trio_and_xrp_map_to_the_expected_kraken_pairs():
    n = _normalize()
    assert n("SOL") == "SOL/USDT"   # [P135] SOL on USDT (USD pair dead)
    assert n("BTC") == "BTC/USD"
    assert n("ETH") == "ETH/USD"
    assert n("XRP") == "XRP/USD"    # [P412] the breadth perp


def test_shield_constructed_from_config_assets_covers_xrp():
    """The behavioural fix: a shield built from the live config.assets tracks
    XRP/USD, so feeding it an XRP snapshot yields a non-None orderbook and the
    P0 check does not abort."""
    from defense.kraken_integrity_shield import KrakenIntegrityShield
    n = _normalize()
    prof = json.loads((REPO / "configs" / "live_high_risk.json")
                      .read_text(encoding="utf-8-sig"))
    symbols = sorted({n(a) for a in prof["assets"]})
    assert "XRP/USD" in symbols, "config.assets no longer yields XRP/USD"
    shield = KrakenIntegrityShield(symbols=symbols)
    # XRP/USD is a KNOWN symbol now (feed_rest_snapshot refuses unknowns)
    ok, reason = shield.feed_rest_snapshot(
        "XRP/USD", bids=[(1.42, 1000.0)], asks=[(1.43, 1000.0)], ts=time.time())
    assert ok, f"valid XRP snapshot rejected: {reason}"
    assert shield.is_fed() is True
    # the P0 check reads get_orderbook -> must be non-None (else it aborts)
    ob = shield.get_orderbook("XRP/USD")
    assert ob is not None, "XRP/USD orderbook is None — the P0 abort would fire"
    # and the home trio is still covered (unchanged, P133/P135)
    for home in ("SOL/USDT", "BTC/USD", "ETH/USD"):
        assert home in shield.orderbooks, f"{home} lost shield coverage"


def test_an_unknown_symbol_is_still_refused_not_grown():
    """The fix must not weaken the P384 rule: a mis-mapped pair surfaces as a
    defect (unknown_symbol), never silently grows the shield."""
    from defense.kraken_integrity_shield import KrakenIntegrityShield
    shield = KrakenIntegrityShield(symbols=["BTC/USD"])
    ok, reason = shield.feed_rest_snapshot(
        "DOGE/USD", bids=[(0.1, 10.0)], asks=[(0.11, 10.0)], ts=time.time())
    assert ok is False and reason == "unknown_symbol"


def test_main_shield_construction_is_config_derived_not_hardcoded():
    """Regression guard: main.py must build the shield's symbols from
    config.assets, not the old hardcoded 3-symbol literal (which is what left
    XRP uncovered)."""
    src = (REPO / "main.py").read_text(encoding="utf-8", errors="replace")
    # the PRIMARY shield (self.integrity_shield) is the one the P0 check reads;
    # scope the guard to its construction window (KrakenLink and the inert
    # secondary shadow shield, P383, keep their own literals and do not abort).
    i = src.index("self.integrity_shield = KrakenIntegrityShield(")
    window = src[max(0, i - 500):i + 200]
    assert "_normalize_kraken_pair(a) for a in self.config.assets" in window, (
        "the primary integrity shield is no longer config-derived — a hardcoded "
        "symbol list leaves any breadth asset uncovered and P0-aborting every tick")
    # the old hardcode must be gone from the primary's construction
    assert "symbols=['SOL/USDT', 'BTC/USD', 'ETH/USD']" not in window

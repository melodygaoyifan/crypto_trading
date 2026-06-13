"""
HMATS v5.1 Phase 2 - Coinbase Sleeve (separate-sleeve state tracking)
=====================================================================

The H6 decision (docs/COINBASE_ENGINE_INTEGRATION_PLAN.md): Coinbase runs as a
SEPARATE sleeve with its own position state, deliberately isolated from the
Kraken-shaped `_paper_positions`/tranche/fill machinery. This avoids the
cross-venue state-contamination class that caused P139/P140.

Core anti-P139 invariant: Coinbase position state is **reconciled from the
venue** (`list_futures_positions`), NEVER inferred from our own fills. The
exchange is the source of truth; we read it, we don't reconstruct it.

This module is INERT until Phase B wires order routing — it only *reads*.
Sync (reuses the adapter's RESTClient); fail-soft (never raises to the caller).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from exchange.symbol_mapping import from_venue_symbol, to_venue_symbol

logger = logging.getLogger(__name__)


def _g(o: Any, k: str, d: Any = None) -> Any:
    return (o.get(k, d) if isinstance(o, dict) else getattr(o, k, d))


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


class CoinbaseSleeve:
    """Read-only Coinbase position/equity view, reconciled from the venue.

    Args:
        adapter: a connected CoinbaseAdapter (shares its RESTClient).
        assets: HMATS-canonical assets this sleeve covers.
    """

    def __init__(self, adapter, assets=("BTC", "ETH", "SOL")) -> None:
        self._adapter = adapter
        self._assets = tuple(assets)
        # product_id -> HMATS asset, for mapping venue positions back
        self._pid_to_asset = {}
        for a in self._assets:
            try:
                self._pid_to_asset[to_venue_symbol(a, "coinbase", "perp")] = a
            except KeyError:
                pass
        self._last_positions: Dict[str, Dict[str, Any]] = {}
        self._last_buying_power_usd: float = 0.0

    def is_ready(self) -> bool:
        return bool(self._adapter and self._adapter.is_connected())

    # ----- reconciliation (authoritative, anti-P139) ----------------------

    def reconcile_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return {asset: {product_id, side, contracts, ...}} read straight from
        the venue. On any error, returns the last-known snapshot (never raises).
        """
        if not self.is_ready():
            return dict(self._last_positions)
        try:
            resp = self._adapter._client.list_futures_positions()
            raw = _g(resp, "positions") or []
            out: Dict[str, Dict[str, Any]] = {}
            for pos in raw:
                pid = str(_g(pos, "product_id") or "")
                asset = self._pid_to_asset.get(pid)
                if asset is None:
                    # map by product_id suffix if not in our set
                    try:
                        asset = from_venue_symbol(pid, "coinbase", "perp")
                    except KeyError:
                        continue
                side = str(_g(pos, "side") or "").upper()
                contracts = _f(_g(pos, "number_of_contracts") or _g(pos, "net_size"))
                signed = contracts if side == "LONG" else -contracts
                out[asset] = {
                    "product_id": pid,
                    "side": side,
                    "contracts": contracts,
                    "signed_contracts": signed,
                    "entry_vwap": _f(_g(pos, "entry_vwap"), 0.0) or None,
                    "unrealized_pnl": _f(_g(pos, "unrealized_pnl"), 0.0),
                    "venue": "coinbase",
                }
            self._last_positions = out
            return out
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] reconcile failed: {type(e).__name__}: {e}; "
                           f"returning last snapshot")
            return dict(self._last_positions)

    def position(self, asset: str) -> Optional[Dict[str, Any]]:
        return self._last_positions.get(asset)

    def signed_contracts(self, asset: str) -> float:
        p = self._last_positions.get(asset)
        return float(p["signed_contracts"]) if p else 0.0

    # ----- equity / margin -------------------------------------------------

    def buying_power_usd(self) -> float:
        """USDC-backed futures buying power (read from the venue). Falls back to
        the last value on error."""
        if not self.is_ready():
            return self._last_buying_power_usd
        try:
            fb = self._adapter._client.get_futures_balance_summary()
            bs = _g(fb, "balance_summary") or fb
            bp = _g(bs, "futures_buying_power") or {}
            self._last_buying_power_usd = _f(_g(bp, "value"), self._last_buying_power_usd)
            return self._last_buying_power_usd
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] buying_power failed: {type(e).__name__}: {e}")
            return self._last_buying_power_usd

    def snapshot(self) -> Dict[str, Any]:
        """Combined read for reporting/heartbeat: positions + buying power."""
        return {
            "venue": "coinbase",
            "positions": self.reconcile_positions(),
            "buying_power_usd": self.buying_power_usd(),
        }

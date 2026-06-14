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

import json
import logging
import os
import time
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

    def __init__(self, adapter, assets=("BTC", "ETH", "SOL"),
                 max_sleeve_drawdown_pct: float = 0.15,
                 max_contracts_per_asset: int = 1) -> None:
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
        self._last_equity_usd: float = 0.0
        # True only after a SUCCESSFUL venue reconcile this call. Autonomous
        # management refuses to act when this is False (don't trade on a stale
        # snapshot after an API timeout). See manage_to_signal.
        self._reconcile_ok: bool = False
        # --- isolated sleeve risk guard (the Kraken existence-fuse equivalent,
        # scoped to Coinbase ONLY; never touches the global fuse) ---
        self._max_sleeve_drawdown_pct = float(max_sleeve_drawdown_pct)
        self._max_contracts_per_asset = int(max_contracts_per_asset)
        self._sleeve_start_equity: Optional[float] = None
        self._halted: bool = False
        self._halt_reason: str = ""
        # [P150] persist the risk baseline + halt across restarts. Without this
        # the 15% drawdown halt re-anchors to current (lower) equity on every
        # container restart — same in-memory-baseline-resets-on-restart class as
        # P148 (DRL buffer) and P140/B2 (_peak_equity). The "loss is capped"
        # guarantee is only real if the baseline survives a restart.
        self._restore_state()

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
            self._reconcile_ok = True
            return out
        except Exception as e:
            self._reconcile_ok = False
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

    def sleeve_equity_usd(self) -> float:
        """True NET LIQUIDATION VALUE of the futures sleeve = total USD collateral
        in the portfolio + unrealized PnL. This is the correct base for the
        drawdown anchor + PnL.

        [P151] MUST NOT use futures_buying_power — that is leverage-adjusted
        (collateral x max-leverage, ~8x) and overstates losable capital by ~8x,
        which made the 15% halt anchor on ~$3,560 when the account only holds
        ~$439 (it would liquidate near $376 long before a 15%-of-$3560 loss).
        Used ONLY for the sleeve's own risk guard + reporting — never fed into the
        Kraken existence-fuse/drawdown."""
        if not self.is_ready():
            return self._last_equity_usd
        try:
            fb = self._adapter._client.get_futures_balance_summary()
            bs = _g(fb, "balance_summary") or fb

            def _field(name: str) -> float:
                return _f(_g(_g(bs, name) or {}, "value"), 0.0)

            total = _field("total_usd_balance")
            if total <= 0:  # fallback: available + posted margin = total collateral
                total = _field("available_margin") + _field("initial_margin")
            upnl = _field("unrealized_pnl")
            if upnl == 0.0:  # summary had none -> sum position-level uPnL
                upnl = sum(_f(p.get("unrealized_pnl")) for p in self._last_positions.values())
            eq = (total + upnl) if total > 0 else self._last_equity_usd
            self._last_equity_usd = eq
            return eq
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] equity fetch failed: {type(e).__name__}: {e}")
            return self._last_equity_usd

    # ----- isolated sleeve risk guard -------------------------------------

    def update_risk(self) -> Dict[str, Any]:
        """Refresh sleeve drawdown + halt state. Call each tick. Sets the
        baseline on first call. HALT is sticky until manually reset (mirrors the
        existence-fuse 'manual recovery only' rule), scoped to Coinbase only."""
        eq = self.sleeve_equity_usd()
        if self._sleeve_start_equity is None and eq > 0:
            self._sleeve_start_equity = eq
            logger.info(f"[COINBASE_SLEEVE] risk baseline set: ${eq:,.2f}")
            self._persist_state()  # [P150] anchor survives restart
        dd = 0.0
        if self._sleeve_start_equity and self._sleeve_start_equity > 0:
            dd = (self._sleeve_start_equity - eq) / self._sleeve_start_equity
        if dd >= self._max_sleeve_drawdown_pct and not self._halted:
            self._halted = True
            self._halt_reason = (f"sleeve drawdown {dd:.1%} >= "
                                 f"{self._max_sleeve_drawdown_pct:.0%}")
            logger.error(f"[COINBASE_SLEEVE] HALTED: {self._halt_reason}")
            self._persist_state()  # [P150] sticky halt survives restart
        return {"equity_usd": eq, "drawdown_pct": dd,
                "halted": self._halted, "halt_reason": self._halt_reason}

    def can_trade(self, asset: str, intended_signed_contracts: float) -> tuple:
        """(allowed, reason). Gate consulted by the order-routing branch BEFORE
        any Coinbase order. Isolated to Coinbase — never blocks Kraken."""
        if self._halted:
            return False, f"coinbase_sleeve_halted: {self._halt_reason}"
        # resulting position size after this order (current + intended)
        cur = self.signed_contracts(asset)
        resulting = abs(cur + intended_signed_contracts)
        if resulting > self._max_contracts_per_asset:
            return False, (f"coinbase_contract_cap: {resulting:.0f} > "
                           f"{self._max_contracts_per_asset} for {asset}")
        return True, "ok"

    def reset_halt(self) -> None:
        """Manual recovery (operator action), Coinbase-sleeve only."""
        self._halted = False
        self._halt_reason = ""
        self._persist_state()  # [P150] clear the persisted halt too
        logger.warning("[COINBASE_SLEEVE] halt manually reset")

    # ----- [P150] persistence: baseline survives restart + forward PnL log ----
    # [P151] bump when the equity FORMULA changes -> stale-baseline files are
    # discarded on restore instead of producing a false drawdown reading.
    _BASE_VERSION = "net_liq_v2"

    def _data_dir(self) -> str:
        return os.environ.get("HMATS_DATA_DIR", "data")

    def _state_path(self) -> str:
        return os.path.join(self._data_dir(), "coinbase_sleeve_state.json")

    def _pnl_path(self) -> str:
        return os.path.join(self._data_dir(), "coinbase_sleeve_pnl.jsonl")

    def _restore_state(self) -> None:
        """Restore the risk baseline + sticky halt from disk so the drawdown cap
        is anchored to inception, not to post-restart equity. Best-effort."""
        try:
            p = self._state_path()
            if not os.path.exists(p):
                return
            with open(p, "r", encoding="utf-8") as fh:
                st = json.load(fh)
            # [P151] the baseline UNIT changed (buying_power -> net-liq equity).
            # Discard any state written under the old formula so the cap
            # re-anchors to true equity instead of false-halting (a ~$3,561
            # baseline vs ~$439 equity would read as an 87% drawdown).
            if st.get("base_version") != self._BASE_VERSION:
                logger.warning(
                    f"[COINBASE_SLEEVE] state base_version="
                    f"{st.get('base_version')!r} != {self._BASE_VERSION!r} "
                    f"-> discarding stale baseline, will re-anchor to true equity")
                return
            se = st.get("sleeve_start_equity")
            self._sleeve_start_equity = float(se) if se is not None else None
            self._halted = bool(st.get("halted", False))
            self._halt_reason = str(st.get("halt_reason", "") or "")
            logger.info(
                f"[COINBASE_SLEEVE] restored state: baseline="
                f"${(self._sleeve_start_equity or 0):,.2f} halted={self._halted}"
                + (f" ({self._halt_reason})" if self._halted else "")
            )
        except Exception as e:  # noqa: silent-swallow — bad/old state file; fall back to fresh baseline + log
            logger.warning(f"[COINBASE_SLEEVE] state restore failed: {type(e).__name__}: {e}")

    def _persist_state(self) -> None:
        """Atomically write the baseline + halt. Best-effort; never raises."""
        try:
            p = self._state_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "sleeve_start_equity": self._sleeve_start_equity,
                    "halted": self._halted,
                    "halt_reason": self._halt_reason,
                    "base_version": self._BASE_VERSION,
                    "saved_ts": time.time(),
                }, fh)
            os.replace(tmp, p)
        except Exception as e:  # noqa: silent-swallow — persistence is best-effort, never break the tick
            logger.debug(f"[COINBASE_SLEEVE] state persist failed: {type(e).__name__}: {e}")

    def log_pnl_point(self) -> Dict[str, Any]:
        """[P150] Append one forward-PnL record to a JSONL so the sleeve's live
        edge can be JUDGED on real data (not a backtest) over days/weeks. Returns
        the record. Call once per tick. Best-effort; never raises to the caller."""
        eq = self.sleeve_equity_usd()
        start = self._sleeve_start_equity
        pnl_usd = (eq - start) if (start is not None) else 0.0
        pnl_pct = (pnl_usd / start) if (start and start > 0) else 0.0
        rec = {
            "ts": time.time(),
            "equity_usd": round(eq, 2),
            "start_equity_usd": round(start, 2) if start is not None else None,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 4),
            "buying_power_usd": round(self._last_buying_power_usd, 2),
            "positions": {a: p.get("signed_contracts")
                          for a, p in self._last_positions.items()},
            "unrealized_pnl_usd": round(
                sum(_f(p.get("unrealized_pnl")) for p in self._last_positions.values()), 2),
            "halted": self._halted,
        }
        try:
            p = self._pnl_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: silent-swallow — logging is best-effort, never break the tick
            logger.debug(f"[COINBASE_SLEEVE] pnl log failed: {type(e).__name__}: {e}")
        return rec

    @staticmethod
    def target_for_signal(direction: float, threshold: float = 0.15) -> int:
        """Map a fused per-asset direction to a target signed contract count.
        |direction| below the threshold -> 0 (FLATTEN). This is what makes the
        sleeve EXIT on hold/neutral, not just open."""
        d = float(direction or 0.0)
        if d >= threshold:
            return 1
        if d <= -threshold:
            return -1
        return 0

    async def manage_to_signal(self, asset: str, direction: float,
                               threshold: float = 0.15) -> Dict[str, Any]:
        """Per-tick driver: move `asset` to the contract target implied by the
        fused direction (incl. flatten on hold). The SOLE Coinbase order path —
        called every tick for routed assets so positions are actively managed
        (opened, flipped, AND closed), closing the exit gap.

        Resilience: refuses to act on a STALE snapshot. A fresh reconcile must
        succeed this call, else SKIP (don't trade off last-known state after an
        API timeout)."""
        self.reconcile_positions()
        if not self._reconcile_ok:
            logger.warning(f"[COINBASE_SLEEVE] manage_to_signal {asset}: skip "
                           f"(reconcile failed; not acting on stale snapshot)")
            return {"status": "SKIPPED_STALE", "asset": asset,
                    "reason": "reconcile_failed"}
        target = self.target_for_signal(direction, threshold)
        return await self.execute_target(asset, target)

    async def execute_target(self, asset: str, target_signed_contracts: int,
                             order_type: str = "LIMIT") -> Dict[str, Any]:
        """Move `asset` to a target signed contract count (e.g. +1 long, -1
        short, 0 flat) via a single marketable order. Risk-gated by can_trade.
        This is the isolated Coinbase execution primitive the engine fork calls.

        Returns a dict {status, ...}. fail-closed: never raises.
        """
        try:
            from exchange.adapter import OrderRequest
        except Exception as e:
            return {"status": "ERROR", "reason": f"import: {e}"}
        if not self.is_ready():
            return {"status": "NOT_READY", "asset": asset}
        self.reconcile_positions()
        cur = self.signed_contracts(asset)
        delta = int(round(target_signed_contracts - cur))
        if delta == 0:
            return {"status": "NOOP", "asset": asset, "contracts": cur}
        ok, reason = self.can_trade(asset, delta)
        if not ok:
            logger.warning(f"[COINBASE_SLEEVE] execute_target {asset} blocked: {reason}")
            return {"status": "BLOCKED", "asset": asset, "reason": reason}
        side = "BUY" if delta > 0 else "SELL"
        n_contracts = abs(delta)
        try:
            pid = self._adapter.to_venue_symbol(asset, "perp")
            cs = self._adapter._contract_size(pid) or 1.0
            base_size = n_contracts * cs  # adapter converts base->contracts
            # marketable limit: cross slightly so it fills; adapter rounds to tick
            prod = self._adapter._client.get_product(product_id=pid)
            mid = _f(_g(prod, "mid_market_price") or _g(prod, "price"))
            px = mid * (1.002 if side == "BUY" else 0.998)
            req = OrderRequest(symbol=pid, side=side, size=base_size,
                               order_type=order_type, price=px, post_only=False)
            res = await self._adapter.place_order(req)
            self.reconcile_positions()
            logger.info(f"[COINBASE_SLEEVE] execute_target {asset} {side} "
                        f"{n_contracts}ct -> success={res.success} "
                        f"now={self.signed_contracts(asset)}ct")
            return {"status": "OK" if res.success else "FAILED", "asset": asset,
                    "side": side, "contracts": n_contracts, "order": res,
                    "position_after": self.signed_contracts(asset)}
        except Exception as e:
            logger.error(f"[COINBASE_SLEEVE] execute_target {asset} error: "
                         f"{type(e).__name__}: {e}")
            return {"status": "ERROR", "asset": asset, "reason": str(e)}

    def snapshot(self) -> Dict[str, Any]:
        """Combined read for reporting/heartbeat: positions + buying power + risk."""
        positions = self.reconcile_positions()
        return {
            "venue": "coinbase",
            "positions": positions,
            "buying_power_usd": self.buying_power_usd(),
            "risk": self.update_risk(),
        }

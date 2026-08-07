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
                 max_contracts_per_asset: int = 1,
                 protective_stop_pct: float = 0.0,
                 protective_stop_assets=None,
                 flip_persist_ticks: int = 0) -> None:
        self._adapter = adapter
        self._assets = tuple(assets)
        # [P198] Flip-persistence churn control, mirroring the Kraken-side P142
        # guard which never reached this path: the sleeve driver reads
        # _last_quant_directions raw, so the sleeve inherited NONE of the
        # Layer-2 churn controls and flipped BTC 29 times in 54 days while the
        # sleeve lost 5.6% (measured 2026-08-06, coinbase_sleeve_pnl.jsonl).
        # A sign-FLIP of a live position must persist this many CONSECUTIVE
        # ticks before it executes; a single-tick reversal holds the position.
        # <=1 disables. Entries from flat, adds, reduces and flattens are NEVER
        # deferred — only direction flips (same asymmetry as the P195 halt:
        # exits must stay instant). In-memory streak; a restart resets it,
        # which only DELAYS a flip — the conservative side.
        self._flip_persist_ticks = max(0, int(flip_persist_ticks or 0))
        self._flip_pending: Dict[str, Any] = {}  # asset -> (want_sign, streak)
        # [P197] Server-side protective stop. `pct` <= 0 DISABLES the feature
        # entirely — a single knob, so "enabled with a 0% stop" is unexpressible.
        # `protective_stop_assets=None` means every sleeve asset; pass a subset to
        # roll out one asset at a time (P141: activation is a deliberate,
        # operator-watched step, never momentum).
        self._protective_stop_pct = float(protective_stop_pct or 0.0)
        self._protective_stop_assets = (
            tuple(protective_stop_assets) if protective_stop_assets else None)
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
        self._cb_portfolio_uuid: Optional[str] = None  # [P153] cached Default portfolio uuid
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
                    # [P197] The venue returns `avg_entry_price` for CDE futures;
                    # `entry_vwap` is never present, so this read was silently
                    # None for every position since the sleeve was written. It
                    # went unnoticed because nothing consumed it — the protective
                    # stop is its first consumer, and it would have anchored to
                    # the MARK instead of to entry without ever saying so.
                    # Textbook P2: reader and writer never agreed on the key.
                    # Keep the dict key stable; only the source changes.
                    "entry_vwap": _f(_g(pos, "avg_entry_price")
                                     or _g(pos, "entry_vwap"), 0.0) or None,
                    "current_price": _f(_g(pos, "current_price"), 0.0) or None,
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

    def _portfolio_uuid(self) -> Optional[str]:
        """Default portfolio uuid (cached) — needed to read TRUE account equity."""
        if self._cb_portfolio_uuid is not None:
            return self._cb_portfolio_uuid
        try:
            ports = self._adapter._client.get_portfolios()
            pl = _g(ports, "portfolios") or []
            for p in pl:
                if str(_g(p, "type") or "").upper() == "DEFAULT":
                    self._cb_portfolio_uuid = str(_g(p, "uuid") or "")
                    return self._cb_portfolio_uuid
            if pl:
                self._cb_portfolio_uuid = str(_g(pl[0], "uuid") or "")
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] portfolio uuid fetch failed: {type(e).__name__}: {e}")
        return self._cb_portfolio_uuid

    def sleeve_equity_usd(self) -> float:
        """True account equity (net liquidation value) of the Coinbase sleeve.

        [P153] = the Default PORTFOLIO's `total_balance` (cash USDC collateral +
        futures uPnL). The Coinbase US perp product cross-collateralizes futures
        against the spot USDC wallet, so the real equity is the PORTFOLIO total
        (~$4,000), NOT `futures_balance_summary.total_usd_balance` (~$439, an
        FCM-only subset) and NOT `futures_buying_power`. P151 used the $439 figure
        and wrongly concluded the account was thinly margined near liquidation;
        the portfolio breakdown shows ~$4,000 USDC backing ~$2,050 notional
        (~0.5x leverage). Falls back to the futures-summary estimate on error.
        Used ONLY for the sleeve's own risk guard + reporting — never fed into the
        Kraken existence-fuse/drawdown."""
        if not self.is_ready():
            return self._last_equity_usd
        # primary: real portfolio total_balance (the cross-collateralized equity)
        try:
            uuid = self._portfolio_uuid()
            if uuid:
                bd = self._adapter._client.get_portfolio_breakdown(portfolio_uuid=uuid)
                d = _g(bd, "breakdown") or bd
                pb = _g(d, "portfolio_balances") or {}
                tb = _f(_g(_g(pb, "total_balance") or {}, "value"), 0.0)
                if tb > 0:
                    self._last_equity_usd = tb
                    return tb
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] portfolio equity fetch failed: {type(e).__name__}: {e}")
        # fallback (degraded): futures-summary collateral + uPnL
        try:
            fb = self._adapter._client.get_futures_balance_summary()
            bs = _g(fb, "balance_summary") or fb

            def _field(name: str) -> float:
                return _f(_g(_g(bs, name) or {}, "value"), 0.0)

            total = _field("total_usd_balance") or (_field("available_margin") + _field("initial_margin"))
            upnl = _field("unrealized_pnl") or sum(
                _f(p.get("unrealized_pnl")) for p in self._last_positions.values())
            eq = (total + upnl) if total > 0 else self._last_equity_usd
            self._last_equity_usd = eq
            return eq
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] equity fallback failed: {type(e).__name__}: {e}")
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
        any Coinbase order. Isolated to Coinbase — never blocks Kraken.

        [P195] The halt stops OPENING; it must never stop EXITING. Previously
        `if self._halted: return False` was the first statement, so a tripped
        drawdown halt blocked every order including a flatten — the control
        meant to cap losses prevented the exit that realises the cap:
          - manage_to_signal(asset, 0.0) -> execute_target(asset, 0) -> BLOCKED,
            so a halted sleeve could not flatten on a hold signal;
          - scripts/coinbase_flatten.py builds a fresh sleeve, which restores
            `halted` from disk (P150), so the documented emergency flatten was
            blocked too until an operator called reset_halt().
        P150 made the halt sticky across restarts, which is right for a loss cap
        and compounding for a trade block. The two were conflated; they are now
        separated.
        """
        # Position math first — the halt decision needs it.
        cur = self.signed_contracts(asset)
        resulting_signed = cur + intended_signed_contracts
        resulting = abs(resulting_signed)

        if self._halted:
            # Allow only orders that STRICTLY reduce absolute exposure.
            # `abs(resulting) < abs(cur)` is the deliberate predicate: a flatten
            # (1 -> 0) and a partial reduce pass, while a FLIP (+1 -> -1, abs
            # 1 -> 1) does not — a halted sleeve must not open new directional
            # risk in the opposite direction.
            if resulting < abs(cur):
                logger.warning(
                    f"[COINBASE_SLEEVE] {asset}: halted but ALLOWING risk-reducing "
                    f"order ({cur:+.0f} -> {resulting_signed:+.0f} contracts); "
                    f"halt blocks opening, never exiting. reason={self._halt_reason}"
                )
                return True, "halted_but_reducing"
            return False, f"coinbase_sleeve_halted: {self._halt_reason}"

        # [P195] Same shape as the halt above, found while testing it: the cap
        # must not block getting UNDER the cap. Only gate orders that INCREASE
        # absolute exposure, so an over-cap position (venue drift, a lowered
        # limit, a manual fill) can always be trimmed back down.
        if resulting > self._max_contracts_per_asset and resulting > abs(cur):
            return False, (f"coinbase_contract_cap: {resulting:.0f} > "
                           f"{self._max_contracts_per_asset} for {asset}")
        return True, "ok"

    def reset_halt(self) -> None:
        """Manual recovery (operator action), Coinbase-sleeve only.

        [P195] Resetting is only needed to resume OPENING. Exiting/reducing an
        existing position never requires a reset — see can_trade().
        """
        self._halted = False
        self._halt_reason = ""
        self._persist_state()  # [P150] clear the persisted halt too
        logger.warning("[COINBASE_SLEEVE] halt manually reset")

    # ----- [P150] persistence: baseline survives restart + forward PnL log ----
    # [P151/P153] bump when the equity FORMULA changes -> stale-baseline files are
    # discarded on restore instead of producing a false drawdown reading.
    # v2 = futures-summary total_usd_balance (~$439, WRONG subset);
    # v3 = portfolio total_balance (~$4,000, true cross-collateralized equity).
    _BASE_VERSION = "portfolio_total_v3"

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
            # Deliberately leaves any flip-persistence streak untouched: a
            # stale tick neither confirms nor refutes the opposing signal, so
            # the streak pauses rather than resets.
            return {"status": "SKIPPED_STALE", "asset": asset,
                    "reason": "reconcile_failed"}
        target = self.target_for_signal(direction, threshold)
        # [P198] Flip-persistence: an opposing target against a LIVE position
        # must persist `_flip_persist_ticks` consecutive ticks before the flip
        # executes; until then, hold the current position (no close, no
        # reverse — the P142 semantics). Never applies to entries from flat,
        # flattens (target 0), or same-direction targets, so exits stay
        # instant (P195 principle) and the deadband flatten is unaffected.
        cur = self.signed_contracts(asset)
        if (self._flip_persist_ticks > 1 and cur != 0 and target != 0
                and (target > 0) != (cur > 0)):
            want = 1 if target > 0 else -1
            pend_sign, streak = self._flip_pending.get(asset, (0, 0))
            streak = streak + 1 if pend_sign == want else 1
            self._flip_pending[asset] = (want, streak)
            if streak < self._flip_persist_ticks:
                logger.info(
                    f"[COINBASE_SLEEVE] {asset}: FLIP DEFERRED "
                    f"({streak}/{self._flip_persist_ticks} consecutive opposing "
                    f"ticks; cur={cur:+.0f}ct target={target:+d}ct) — holding")
                return {"status": "FLIP_DEFERRED", "asset": asset,
                        "streak": streak, "need": self._flip_persist_ticks,
                        "current": cur, "target": target}
            self._flip_pending.pop(asset, None)
        else:
            # Same-direction, flat, or flatten: any pending flip streak is
            # broken — a flip must be CONSECUTIVE opposing ticks.
            self._flip_pending.pop(asset, None)
        return await self.execute_target(asset, target)

    async def _cancel_resting_orders(self, pid: str, asset: str) -> int:
        """[P195] Cancel our own resting orders for `pid` before placing a new one.

        execute_target places a marketable GTC LIMIT, and nothing ever cancelled
        it. `cancel_order`/`fetch_open_orders` existed on the adapter with zero
        callers anywhere in main.py, core/, exchange/ or scripts/. So an unfilled
        limit rested indefinitely and could fill AFTER the engine died — making
        the sleeve a risk-ADDER on process death rather than merely unprotected.
        It also let orders stack across ticks.

        Fail-soft by design: a cancel failure must never raise into the tick, and
        must not stop the new order (the venue-authoritative reconcile on the next
        pass is the backstop). Returns the number cancelled.
        """
        cancelled = 0
        try:
            open_orders = await self._adapter.fetch_open_orders(pid)
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] {asset}: could not list resting "
                           f"orders ({type(e).__name__}: {e}); proceeding")
            return 0
        for o in open_orders or []:
            oid = _g(o, "order_id") or _g(o, "id")
            if not oid:
                continue
            try:
                if await self._adapter.cancel_order(str(oid), pid):
                    cancelled += 1
            except Exception as e:
                logger.warning(f"[COINBASE_SLEEVE] {asset}: cancel {oid} failed "
                               f"({type(e).__name__}: {e}); proceeding")
        if cancelled:
            logger.info(f"[COINBASE_SLEEVE] {asset}: cancelled {cancelled} stale "
                        f"resting order(s) before new target")
        return cancelled

    # ----- [P197] server-side protective stop -----------------------------
    #
    # The sleeve had NO server-side protection: every exit was a client-side API
    # call on the 4H tick, so a dead process left BTC/ETH/SOL perp exposure with
    # nothing resting at the venue to close it. Preview-verified 2026-08-07 that
    # CDE accepts stop-limits on all three contracts (errs: [], and
    # order_margin_total = 0, i.e. treated as position-REDUCING).
    #
    # THE HAZARD THIS CODE EXISTS TO CONTAIN: CDE rejects `reduce_only`
    # (coinbase_adapter.py:206). A resting stop is therefore a PLAIN order — if
    # the position it guards disappears and the stop is still live, triggering it
    # OPENS an opposite position. So the stop is reconciled to desired-state every
    # tick and cancelled the moment the asset is flat. It is never fire-and-forget.

    def _stop_enabled_for(self, asset: str) -> bool:
        if self._protective_stop_pct <= 0:
            return False
        if self._protective_stop_assets is None:
            return True
        return asset in self._protective_stop_assets

    @staticmethod
    def _order_config(o) -> Dict[str, Any]:
        """order_configuration as a plain mapping, whatever shape it arrives in.

        [P197-fix] The adapter now normalises this at the boundary, but this
        stays defensive on purpose: the first version required a dict, the SDK
        hands back an `OrderConfiguration` object, and the result was that a
        stop-limit demonstrably resting at the venue read as "no stop found".
        Downstream that is not a cosmetic miss — it would have placed a second
        stop every tick, and skipped cancelling orphans when flat.
        """
        cfg = _g(o, "order_configuration")
        if isinstance(cfg, dict):
            return cfg
        inner = getattr(cfg, "__dict__", None)
        return inner if isinstance(inner, dict) else {}

    @staticmethod
    def _is_stop_order(o) -> bool:
        """A Coinbase order is a stop iff its order_configuration says so."""
        keys = list(CoinbaseSleeve._order_config(o).keys())
        return any("stop" in str(k).lower() or "bracket" in str(k).lower()
                   for k in keys)

    def desired_stop_price(self, asset: str) -> Optional[float]:
        """Stop anchored to ENTRY, not to the current mark.

        Anchoring to entry makes this a fixed-risk stop-loss. Anchoring to the
        mark would silently make it a trailing stop that ratchets on every tick
        and re-places orders forever. Falls back to the mark only when the venue
        gives us no entry_vwap.
        """
        pos = self._last_positions.get(asset) or {}
        cur = float(pos.get("signed_contracts") or 0.0)
        if cur == 0:
            return None
        anchor = pos.get("entry_vwap")
        if not anchor:
            try:
                pid = self._adapter.to_venue_symbol(asset, "perp")
                prod = self._adapter._client.get_product(product_id=pid)
                anchor = _f(_g(prod, "mid_market_price") or _g(prod, "price"))
            except Exception:
                return None
        if not anchor:
            return None
        pct = self._protective_stop_pct
        return float(anchor) * ((1.0 - pct) if cur > 0 else (1.0 + pct))

    async def ensure_protective_stop(self, asset: str) -> Dict[str, Any]:
        """Reconcile the resting stop for `asset` to the desired state.

        Called every tick AFTER manage_to_signal. Never raises.
        """
        if not self._stop_enabled_for(asset):
            return {"status": "DISABLED", "asset": asset}
        if not self.is_ready():
            return {"status": "NOT_READY", "asset": asset}
        # Never act on a stale snapshot — same rule as manage_to_signal. Acting
        # on last-known state here could cancel a live stop we cannot see.
        if not self._reconcile_ok:
            return {"status": "SKIPPED_STALE", "asset": asset}
        try:
            from exchange.adapter import OrderRequest
            pid = self._adapter.to_venue_symbol(asset, "perp")
            cur = self.signed_contracts(asset)
            resting = [o for o in (await self._adapter.fetch_open_orders(pid) or [])
                       if self._is_stop_order(o)]

            # FLAT: any surviving stop is an ORPHAN that could open a position.
            # Cancelling it is the single most important thing in this method.
            if cur == 0:
                n = 0
                for o in resting:
                    oid = _g(o, "order_id") or _g(o, "id")
                    if oid and await self._adapter.cancel_order(str(oid), pid):
                        n += 1
                if n:
                    logger.info(f"[COINBASE_STOP] {asset}: flat -> cancelled {n} "
                                f"orphan stop(s) (no reduce_only on CDE, so a live "
                                f"stop here would OPEN a position)")
                return {"status": "FLAT_CANCELLED" if n else "FLAT_NONE",
                        "asset": asset, "cancelled": n}

            cs = self._adapter._contract_size(pid) or 1.0
            want_side = "SELL" if cur > 0 else "BUY"
            want_base = abs(cur) * cs

            # Correct stop already resting? Leave it — re-placing every tick would
            # churn the venue and reset the anchor.
            def _matches(o) -> bool:
                if str(_g(o, "side") or "").upper() != want_side:
                    return False
                cfg = self._order_config(o)
                inner = next(iter(cfg.values()), {}) if cfg else {}
                bs = _f(_g(inner, "base_size"), 0.0)
                # base_size is in CONTRACTS at the venue (base_increment=1)
                return abs(bs - abs(cur)) < 1e-9

            good = [o for o in resting if _matches(o)]
            if len(good) == 1 and len(resting) == 1:
                return {"status": "OK_EXISTS", "asset": asset,
                        "contracts": cur, "side": want_side}

            # Otherwise: clear whatever is there and place one correct stop.
            for o in resting:
                oid = _g(o, "order_id") or _g(o, "id")
                if oid:
                    await self._adapter.cancel_order(str(oid), pid)

            stop_px = self.desired_stop_price(asset)
            if not stop_px:
                return {"status": "NO_ANCHOR", "asset": asset}
            req = OrderRequest(symbol=pid, side=want_side, size=want_base,
                               order_type="STOP", stop_price=stop_px,
                               post_only=False)
            res = await self._adapter.place_order(req)
            if res.success:
                logger.info(f"[COINBASE_STOP] {asset}: placed protective "
                            f"{want_side} stop @ {stop_px:.4f} for {cur:+.0f}ct "
                            f"({self._protective_stop_pct:.1%} from entry)")
                return {"status": "PLACED", "asset": asset, "side": want_side,
                        "stop_price": stop_px, "contracts": cur}
            logger.warning(f"[COINBASE_STOP] {asset}: stop placement FAILED: "
                           f"{res.error_code}: {res.error_message} — position is "
                           f"UNPROTECTED at the venue this tick")
            return {"status": "FAILED", "asset": asset,
                    "reason": f"{res.error_code}: {res.error_message}"}
        except Exception as e:
            logger.warning(f"[COINBASE_STOP] {asset}: ensure failed "
                           f"({type(e).__name__}: {e}); position may be unprotected")
            return {"status": "ERROR", "asset": asset, "reason": str(e)}

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
            await self._cancel_resting_orders(pid, asset)
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

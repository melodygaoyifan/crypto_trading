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

[STATUS 2026-08-07 — LIVE, the SOLE directional order path.] The "INERT until
Phase B" line that used to sit here went stale the day Phase B activated
(2026-06-13) and cost diagnosis time (P155 layer 1: docs said the opposite of
runtime truth). Current reality: `manage_to_signal` is called every 4H tick
from main.py's heartbeat for every routed asset and OPENS, FLIPS (subject to
[P198] flip-persistence), and FLATTENS real Coinbase perp positions; [P197]
`ensure_protective_stop` rests a real venue-side stop. Kraken is structurally
flat (P152), so this sleeve is the only thing taking directional risk.
Sync (reuses the adapter's RESTClient); fail-soft (never raises to the caller).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
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
    """LIVE Coinbase perp sleeve: venue-reconciled state + the sole order
    driver (`manage_to_signal`/`execute_target`), with its own risk guard
    (15% drawdown halt, contract cap, [P197] protective stop, [P198]
    flip-persistence). Position state is always reconciled FROM the venue,
    never inferred from our own fills (anti-P139).

    Args:
        adapter: a connected CoinbaseAdapter (shares its RESTClient).
        assets: HMATS-canonical assets this sleeve covers.
    """

    def __init__(self, adapter, assets=("BTC", "ETH", "SOL"),
                 max_sleeve_drawdown_pct: float = 0.15,
                 max_contracts_per_asset: int = 1,
                 protective_stop_pct: float = 0.0,
                 protective_stop_assets=None,
                 flip_persist_ticks: int = 0,
                 max_net_exposure: Optional[float] = None,
                 max_asset_exposure: Optional[Dict[str, float]] = None,
                 maker_first: bool = False,
                 maker_wait_sec: float = 45.0) -> None:
        self._adapter = adapter
        # [P270] Maker-first execution. CDE charges 0bps maker vs 3bps taker
        # (probed 2026-08-15: post_only is genuinely ENFORCED — a crossing
        # post-only previews PREVIEW_INVALID_LIMIT_PRICE_POST_ONLY, so this
        # cannot silently degrade into taker wearing maker's name, the P169
        # shape). When on, execute_target tries a post-only order joining the
        # touch, polls up to maker_wait_sec INSIDE the same call, cancels,
        # and crosses the remainder — an entry is NEVER left resting across
        # ticks (the P265 dead-signal-fill class). Default OFF: enabling
        # places differently-shaped real orders (P141, operator-watched).
        self._maker_first = bool(maker_first)
        self._maker_wait_sec = max(5.0, float(maker_wait_sec or 45.0))
        # [P210] Per-asset gross cap as a fraction of SLEEVE equity. Wired from
        # config post_leverage_caps — the same policy the Kraken path applies,
        # against the equity that actually backs these positions. See can_trade
        # for why intent.target_exposure is deliberately not reconnected.
        self._max_asset_exposure: Dict[str, float] = dict(max_asset_exposure or {})
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
        # [P208] NET exposure budget, enforced on THIS sleeve's own book.
        # P144 added `max_net_exposure` because the book ran +0.54 net-long into
        # a -23% market — about half the Apr-Jun loss — and gross caps do not
        # constrain net direction. But its only enforcement site is
        # `core/execution_service.py`, past the P152 early return, reading
        # Kraken-shaped `_paper_positions` which has been `{}` since June. So on
        # the only venue that trades, the cap has never once been evaluated
        # (P201). The 1-contract-per-asset cap does not substitute: it is
        # per-asset with no aggregation, so all-three-long is ~+0.5x net —
        # precisely what P144 was written to prevent.
        # Deliberately NOT wired through GlobalExposureCapManager: that carries
        # Kraken-shaped state, and feeding Coinbase positions into it is the
        # cross-venue contamination P139/P140 came from. Same policy number,
        # enforced locally, on a book read from the venue.
        # None disables.
        self._max_net_exposure = (float(max_net_exposure)
                                  if max_net_exposure else None)
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
        self._last_dd_pct: float = 0.0  # [P265] last real dd; served when equity is unknown
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
                # [P253] Magnitude and sign are derived SEPARATELY, and an
                # unrecognized side REFUSES rather than guesses. The old
                # `contracts if side == "LONG" else -contracts` turned ANY
                # unexpected side string (an SDK enum rename, a missing key,
                # "FUTURES_POSITION_SIDE_LONG") into a phantom SHORT — the
                # same reader/writer class as the entry_vwap bug documented
                # below. abs() on the magnitude also defuses a signed
                # `net_size` doubling the sign (SHORT + net_size=-2 used to
                # read as +2). Raising here lands in the except below ->
                # _reconcile_ok=False -> callers refuse to act on this
                # snapshot, which is the correct failure direction: a refusal
                # is recoverable, a fabricated position is not.
                contracts = abs(_f(_g(pos, "number_of_contracts")
                                   or _g(pos, "net_size")))
                if contracts == 0:
                    # [P265] A ZERO-magnitude row (settled/residual entries
                    # arrive with side "" or FUTURES_POSITION_SIDE_UNKNOWN)
                    # has no sign to guess — the P253 raise on it poisoned
                    # the ENTIRE snapshot: _reconcile_ok=False made every
                    # asset's manage/execute/stop-reconcile (flattens
                    # included) return SKIPPED_STALE for as long as the row
                    # persisted. Skipping a zero row preserves the invariant
                    # exactly (never guess a NONZERO position's sign).
                    continue
                if "SHORT" in side:
                    signed = -contracts
                elif "LONG" in side:
                    signed = contracts
                else:
                    raise ValueError(
                        f"unrecognized position side {side!r} for {pid} — "
                        f"refusing to guess a sign")
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
        """Default portfolio uuid (cached) — needed to read TRUE account equity.

        [P265] Only a NON-EMPTY uuid is ever cached. The old code cached
        `str(_g(p, "uuid") or "")`, so one malformed get_portfolios response
        (row without a uuid key) stored "" permanently — `if uuid:` in
        sleeve_equity_usd is falsy, so the PRIMARY equity path was disabled
        for the process lifetime with no retry and no warning, pinning every
        tick on the degraded fallback."""
        if self._cb_portfolio_uuid:
            return self._cb_portfolio_uuid
        try:
            ports = self._adapter._client.get_portfolios()
            pl = _g(ports, "portfolios") or []
            for p in pl:
                if str(_g(p, "type") or "").upper() == "DEFAULT":
                    _uid = str(_g(p, "uuid") or "")
                    if _uid:
                        self._cb_portfolio_uuid = _uid
                        return _uid
                    break  # DEFAULT row without uuid: fall to first-row probe
            for p in pl:
                _uid = str(_g(p, "uuid") or "")
                if _uid:
                    self._cb_portfolio_uuid = _uid
                    return _uid
            if pl:
                logger.warning("[COINBASE_SLEEVE] get_portfolios returned rows "
                               "but none carried a uuid — will retry next call")
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

        [P209] This value IS now fed into the existence fuse (main.py, tagged
        [FUSE-FEED]) as a per-tick equity delta, because the fuse's own three
        record_pnl() sites sit past the P152 early return and so never fire for a
        routed asset — leaving Non-Negotiable Rule #3 with an empty pnl_history
        while this sleeve carried all of the directional risk. The earlier
        "never fed into the Kraken existence-fuse/drawdown" note described a
        separation that made sense when Kraken still traded; keeping it after
        Phase B just meant nothing measured the book that was actually at risk.

        NOTE the failure mode if you reuse this: on an API error it returns the
        last KNOWN equity, so a caller computing a delta must check
        `_reconcile_ok` first or a stale read silently becomes "no change"."""
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
                else:
                    logger.warning(
                        "[COINBASE_SLEEVE] portfolio breakdown returned "
                        f"total_balance={tb!r} (not > 0) — treating equity as "
                        "unknown this pass")
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] portfolio equity fetch failed: {type(e).__name__}: {e}")
        # [P265] The futures-summary figure is an FCM-ONLY SUBSET of the real
        # cross-collateralized equity (~$439 vs ~$4,000 — the exact P153
        # confusion). It is a WRONG-DENOMINATION number, not a degraded copy of
        # the right one, and it used to be RETURNED here unmarked: one partial
        # outage tick (portfolio endpoint down, futures endpoint up) fed the
        # 15% halt dd≈88% (sticky, persisted), fed the existence fuse a phantom
        # ~-$3.4k realized loss (suspension = force-flatten, manual recovery
        # only), and re-created P261's phantom daily-loss arithmetic through a
        # door its known/unknown guard cannot see. The subset is now DIAGNOSTIC
        # ONLY: log it for the operator, serve the last-known TRUE equity
        # (0.0 on a never-priced process — every consumer already treats <=0
        # as "unknown" and holds/skips).
        try:
            fb = self._adapter._client.get_futures_balance_summary()
            bs = _g(fb, "balance_summary") or fb

            def _field(name: str) -> float:
                return _f(_g(_g(bs, name) or {}, "value"), 0.0)

            total = _field("total_usd_balance") or (_field("available_margin") + _field("initial_margin"))
            upnl = _field("unrealized_pnl") or sum(
                _f(p.get("unrealized_pnl")) for p in self._last_positions.values())
            if total > 0:
                logger.warning(
                    "[COINBASE_SLEEVE] portfolio equity UNAVAILABLE; futures "
                    f"summary shows FCM-subset ${total + upnl:,.2f} (WRONG "
                    "denomination, P153/P265 — NOT substituted). Serving "
                    f"last-known true equity ${self._last_equity_usd:,.2f}")
            return self._last_equity_usd
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] equity fallback failed: {type(e).__name__}: {e}")
            return self._last_equity_usd

    # ----- isolated sleeve risk guard -------------------------------------

    def update_risk(self) -> Dict[str, Any]:
        """Refresh sleeve drawdown + halt state. Call each tick. Sets the
        baseline on first call. HALT is sticky until manually reset (mirrors the
        existence-fuse 'manual recovery only' rule), scoped to Coinbase only."""
        eq = self.sleeve_equity_usd()
        # [P265] Unknown is not zero dollars. On a cold boot with the venue
        # down, eq is 0.0 while the RESTORED baseline is real — the old code
        # computed dd = (start - 0)/start = 100% and tripped the sticky,
        # PERSISTED halt on an API artifact. P263 made the first update_risk
        # run at startup construction, i.e. exactly when a boot-time outage
        # yields 0. Skip the evaluation until a real equity read exists; the
        # halt state itself is untouched (a tripped halt stays tripped).
        if eq <= 0:
            logger.warning(
                "[COINBASE_SLEEVE] update_risk: equity unknown (<=0) — "
                "skipping drawdown evaluation this pass (unknown != $0)")
            return {"equity_usd": eq,
                    "drawdown_pct": self._last_dd_pct,
                    "halted": self._halted, "halt_reason": self._halt_reason,
                    "degraded": True}
        if self._sleeve_start_equity is None:
            self._sleeve_start_equity = eq
            logger.info(f"[COINBASE_SLEEVE] risk baseline set: ${eq:,.2f}")
            self._persist_state()  # [P150] anchor survives restart
        dd = 0.0
        if self._sleeve_start_equity and self._sleeve_start_equity > 0:
            dd = (self._sleeve_start_equity - eq) / self._sleeve_start_equity
        self._last_dd_pct = dd
        if dd >= self._max_sleeve_drawdown_pct and not self._halted:
            self._halted = True
            self._halt_reason = (f"sleeve drawdown {dd:.1%} >= "
                                 f"{self._max_sleeve_drawdown_pct:.0%}")
            logger.error(f"[COINBASE_SLEEVE] HALTED: {self._halt_reason}")
            self._persist_state()  # [P150] sticky halt survives restart
        return {"equity_usd": eq, "drawdown_pct": dd,
                "halted": self._halted, "halt_reason": self._halt_reason}

    def _notional_usd(self, asset: str, signed_contracts: float) -> float:
        """Signed USD notional of a contract count for `asset`.

        Price preference: the venue's `current_price` for a live position, then
        its entry vwap, then the product mid. Returns 0.0 when no price can be
        established — a position we cannot price must not silently count as
        zero exposure, so callers treat 0.0 as "unknown" (see sleeve_exposure).
        """
        if not signed_contracts:
            return 0.0
        pos = self._last_positions.get(asset) or {}
        px = _f(pos.get("current_price"), 0.0) or _f(pos.get("entry_vwap"), 0.0)
        try:
            pid = self._adapter.to_venue_symbol(asset, "perp")
            if not px:
                prod = self._adapter._client.get_product(product_id=pid)
                px = _f(_g(prod, "mid_market_price") or _g(prod, "price"), 0.0)
            cs = self._adapter._contract_size(pid) or 0.0
        except Exception:
            return 0.0
        if not px or not cs:
            return 0.0
        return float(signed_contracts) * float(cs) * float(px)

    def sleeve_exposure(self, overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """[P208] Net and gross exposure of THIS sleeve, as a fraction of its
        own equity, computed from the venue-reconciled book.

        `overrides` replaces an asset's signed contract count, so a proposed
        order can be evaluated before it is sent.

        `priced_ok` is False when any non-zero position could not be priced —
        callers must fail OPEN on that (never block a de-risking order because a
        price lookup failed) while still refusing to certify an increase.
        """
        eq = float(self._last_equity_usd or 0.0)
        book = {a: float((p or {}).get("signed_contracts") or 0.0)
                for a, p in self._last_positions.items()}
        _ov: Dict[str, float] = dict(overrides or {})
        for a, c in _ov.items():
            book[a] = float(c)
        net = gross = 0.0
        priced_ok = True
        for a, c in book.items():
            n = self._notional_usd(a, c)
            if c and not n:
                priced_ok = False
                continue
            net += n
            gross += abs(n)
        return {
            "equity_usd": eq, "net_usd": net, "gross_usd": gross,
            "net_pct": (net / eq) if eq > 0 else 0.0,
            "gross_pct": (gross / eq) if eq > 0 else 0.0,
            "priced_ok": priced_ok and eq > 0,
        }

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

        # [P208] NET exposure budget across the whole sleeve. The per-asset
        # contract cap above does not aggregate, so all-three-long is ~+0.5x net
        # while every asset is individually "within cap" — exactly the +0.54
        # net-long P144 was written to prevent, on the venue P144 cannot see.
        # getattr, not self._max_net_exposure: this is the live order path, and
        # an AttributeError here refuses every order. P85's rule — a new
        # attribute read defends itself. Caught immediately by pre-existing
        # tests whose sleeves were built before this field existed; a partially
        # constructed sleeve in production would have had the same shape.
        _net_budget = getattr(self, "_max_net_exposure", None)
        if _net_budget:
            _before = self.sleeve_exposure()
            _after = self.sleeve_exposure(
                overrides={asset: cur + intended_signed_contracts})
            _n_before, _n_after = abs(_before["net_pct"]), abs(_after["net_pct"])
            # Only ever block an order that INCREASES |net| — de-risking must
            # always be free (the P144 rule, and the P195 lesson about controls
            # that trap you in a position).
            if _n_after > _n_before and _n_after > _net_budget:
                if not _after["priced_ok"]:
                    # Fail OPEN on a pricing failure rather than block on a
                    # number we could not compute. A risk control that fires on
                    # missing data is a data outage that stops trading.
                    logger.warning(
                        f"[COINBASE_SLEEVE] {asset}: net-exposure check "
                        f"INCONCLUSIVE (could not price the book); allowing")
                else:
                    return False, (
                        f"coinbase_net_exposure_cap: |net| {_n_after:.1%} > "
                        f"{_net_budget:.0%} (was {_n_before:.1%}) "
                        f"on ${_after['equity_usd']:,.0f} equity")

        # [P210] PER-ASSET gross exposure, as a fraction of SLEEVE equity.
        #
        # This is the control `intent.target_exposure` was supposed to provide
        # and structurally cannot. That number is converted to a size at
        # core/unit_system.py:232 as `target_exposure * account_equity`, where
        # account_equity is Kraken NAV (account_sync is built exchange_name=
        # "kraken"). So the risk stack sizes Coinbase positions out of KRAKEN
        # capital: at target_exposure=0.25 on ~$9.8k Kraken NAV that is ~$2,457,
        # against the ~$643 this sleeve actually holds. "Reconnecting" it would
        # roughly 4x the book AND denominate one venue's risk in another's
        # capital — the cross-venue contamination P139/P140 came from. So the
        # chain is deliberately NOT reconnected; the same policy numbers
        # (config post_leverage_caps: BTC/ETH 0.25, SOL 0.20) are enforced here
        # instead, against the equity that actually backs these positions.
        #
        # Not vacuous today, which is the test this file's controls have to
        # pass: contract granularity is FIXED while equity moves. One BTC nano
        # is ~17% of a $3.8k sleeve but ~34% of a $1.9k one, and nothing else
        # notices — the contract cap counts contracts, and P208's net budget
        # aggregates and so passes a single concentrated asset. This binds as
        # the account shrinks, saying "one contract is now too large for this
        # account" rather than quietly taking an oversized position.
        _asset_caps = getattr(self, "_max_asset_exposure", None) or {}
        _asset_budget = _asset_caps.get(asset)
        if _asset_budget:
            _eq = float(self._last_equity_usd or 0.0)
            _n_cur = abs(self._notional_usd(asset, cur))
            _n_res = abs(self._notional_usd(asset, resulting_signed))
            # Unpriceable => fail OPEN, exactly as the net check above does.
            if _eq > 0 and (_n_res or not resulting):
                _p_cur = _n_cur / _eq
                _p_res = _n_res / _eq
                # Increases only — never block getting back UNDER the cap.
                if _p_res > _p_cur and _p_res > _asset_budget:
                    return False, (
                        f"coinbase_asset_exposure_cap: {asset} {_p_res:.1%} > "
                        f"{_asset_budget:.0%} of ${_eq:,.0f} sleeve equity "
                        f"(was {_p_cur:.1%})")
            elif resulting:
                logger.warning(
                    f"[COINBASE_SLEEVE] {asset}: per-asset exposure check "
                    f"INCONCLUSIVE (eq=${_eq:,.0f}, could not price "
                    f"{resulting_signed:+.0f}ct); allowing")
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
        except Exception as e:  # noqa: silent-swallow — persistence must never break the tick, but it must be LOUD
            # [P265] This file carries the STICKY HALT and the drawdown
            # baseline. A persist failure at DEBUG (dropped at production log
            # level) meant a tripped halt that silently failed to write would
            # CLEAR on the next restart and the sleeve would resume opening —
            # the P160 shape ("except: logger.debug in a writer is a silent
            # failure") on load-bearing safety state.
            logger.error(
                f"[COINBASE_SLEEVE] state persist FAILED "
                f"({type(e).__name__}: {e}) — halted={self._halted}: if the "
                f"halt is tripped it will NOT survive a restart")

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

    async def _cancel_stale_entry_orders(self, pid: str, asset: str) -> int:
        """[P265] Cancel resting NON-STOP orders on ticks that place nothing.

        `_cancel_resting_orders` runs only on the order-placing path, AFTER the
        NOOP/BLOCKED early returns. So an unfilled marketable entry limit (the
        P207-observed non-fill window) survived every subsequent hold tick —
        target 0 -> delta 0 -> NOOP before any cancel — and could fill hours or
        days later against a dead signal, with no protective stop resting at
        fill time. Worse: a limit placed just before the drawdown halt tripped
        survived the halt (BLOCKED also preceded the cancel), so the halt's
        "no new risk" guarantee was durably bypassed by an order it never saw.

        Stops are deliberately NOT touched here: the protective stop is the
        one resting order that SHOULD survive a hold tick, and cancelling it
        every NOOP would churn the venue and reset the anchor (the exact churn
        `ensure_protective_stop`'s match-and-leave logic exists to avoid).
        Fail-soft; a cancel is always de-risking. Returns the number cancelled.
        """
        cancelled = 0
        try:
            open_orders = await self._adapter.fetch_open_orders(pid)
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] {asset}: could not list resting "
                           f"orders on hold tick ({type(e).__name__}: {e})")
            return 0
        for o in open_orders or []:
            if self._is_stop_order(o):
                continue
            oid = _g(o, "order_id") or _g(o, "id")
            if not oid:
                continue
            try:
                if await self._adapter.cancel_order(str(oid), pid):
                    cancelled += 1
            except Exception as e:
                logger.warning(f"[COINBASE_SLEEVE] {asset}: stale-entry cancel "
                               f"{oid} failed ({type(e).__name__}: {e})")
        if cancelled:
            logger.warning(
                f"[COINBASE_SLEEVE] {asset}: cancelled {cancelled} stale ENTRY "
                f"order(s) on a no-order tick — an unfilled limit from a "
                f"previous tick would otherwise rest until it filled against "
                f"a dead signal (P265)")
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

    async def ensure_protective_stop(self, asset: str,
                                     intended_target: Optional[float] = None
                                     ) -> Dict[str, Any]:
        """Reconcile the resting stop for `asset` to the desired state.

        Called every tick AFTER manage_to_signal. Never raises.

        `intended_target` is this tick's target contract count, when the caller
        knows it. It exists because of a REAL orphan observed live on
        2026-08-07 (P207): `execute_target` flattens with a marketable LIMIT,
        which had not yet FILLED when this method ran, so `reconcile_positions`
        still reported the old position and a protective stop was placed on an
        asset that went flat moments later. CDE rejects `reduce_only`, so that
        resting stop is a PLAIN order — it would have OPENED a position when
        touched, and the next reconcile was 4 hours away.

        Reconciling to the venue is right in the steady state but wrong in the
        instant between "order accepted" and "order filled". When the caller
        knows the intent, intent wins: `intended_target == 0` means treat the
        asset as flat and cancel, whatever the snapshot currently says.
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
            # [P207] Intent beats the snapshot when the caller supplied it. A
            # flatten placed a moment ago may not have filled yet, so `cur` can
            # still show the position we just closed.
            if intended_target is not None and abs(float(intended_target)) < 1e-9:
                cur = 0.0
            # [P265] FLIP IN FLIGHT: a nonzero intent whose SIGN disagrees with
            # a nonzero snapshot means the flip order was accepted but has not
            # filled at reconcile time (the same P207 non-fill window, on the
            # flip leg the ==0 carve-out cannot see). A stop for EITHER side is
            # dangerous here: anchored to the snapshot, it is wrong-side the
            # moment the flip fills — triggering it DOUBLES the new position
            # (no reduce_only on CDE, no can_trade gate on venue-triggered
            # fills); anchored to the intent, it is wrong-side if the flip
            # strands. Place NOTHING this tick — execute_target already swept
            # all resting orders before the flip, so nothing is left guarding
            # the old side either — and let the next tick's reconcile place
            # the stop for whichever position is actually real.
            elif (intended_target is not None
                    and abs(cur) > 1e-9
                    and (float(intended_target) > 0) != (cur > 0)):
                logger.warning(
                    f"[COINBASE_STOP] {asset}: flip in flight (snapshot "
                    f"{cur:+.0f}ct vs intended {float(intended_target):+.0f}ct)"
                    f" — placing NO stop this tick; either side could double "
                    f"the final position (P265)")
                return {"status": "FLIP_IN_TRANSITION", "asset": asset,
                        "snapshot": cur, "intended": float(intended_target)}
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

            # [P265] Same refusal as execute_target: an `or 1.0` fallback
            # here sized the protective stop 10x for ETH.
            cs = self._adapter._contract_size(pid)
            if not cs or cs <= 0:
                logger.error(f"[COINBASE_STOP] {asset}: contract size UNKNOWN "
                             f"for {pid} — refusing to size a stop with a "
                             f"fabricated unit")
                return {"status": "NO_CONTRACT_SIZE", "asset": asset}
            want_side = "SELL" if cur > 0 else "BUY"
            want_base = abs(cur) * cs
            _want_px = self.desired_stop_price(asset)

            # Correct stop already resting? Leave it — re-placing every tick would
            # churn the venue and reset the anchor.
            def _matches(o) -> bool:
                if str(_g(o, "side") or "").upper() != want_side:
                    return False
                cfg = self._order_config(o)
                inner: Any = next(iter(cfg.values()), {}) if cfg else {}
                bs = _f(_g(inner, "base_size"), 0.0)
                # base_size is in CONTRACTS at the venue (base_increment=1)
                if abs(bs - abs(cur)) >= 1e-9:
                    return False
                # [P265] The match must include the PRICE. Side+size alone
                # certified ANY same-shaped stop as "the protection" forever:
                # a config pct change never re-priced resting stops, a
                # mark-anchored fallback stop was never re-anchored to entry,
                # and a manually placed stop at an absurd trigger was silently
                # adopted. 0.5% relative tolerance: generous vs tick rounding
                # (BTC's tick of 5 on ~$64k is 0.008%), tight vs a config
                # change (10% -> 15% moves the trigger ~5%).
                if _want_px:
                    sp = _f(_g(inner, "stop_price"), 0.0)
                    if not sp or abs(sp - _want_px) / _want_px > 0.005:
                        return False
                return True

            good = [o for o in resting if _matches(o)]
            if len(good) == 1 and len(resting) == 1:
                return {"status": "OK_EXISTS", "asset": asset,
                        "contracts": cur, "side": want_side}

            # Otherwise: clear whatever is there and place one correct stop.
            # [P265] Count FAILED cancels — the old loop discarded the result
            # (the FLAT path five branches up counts it) and placed a new stop
            # unconditionally, so a failed cancel yielded TWO plain stops: the
            # first close the position, the second OPENS the opposite side
            # when it fires. Refuse to place while anything we tried to cancel
            # may still be live; next tick retries.
            _cancel_failed = 0
            for o in resting:
                oid = _g(o, "order_id") or _g(o, "id")
                if oid and not await self._adapter.cancel_order(str(oid), pid):
                    _cancel_failed += 1
            if _cancel_failed:
                logger.warning(
                    f"[COINBASE_STOP] {asset}: {_cancel_failed} stop cancel(s) "
                    f"FAILED — refusing to place a replacement (two live plain "
                    f"stops = the second one OPENS an opposite position when "
                    f"it fires, P265). Retrying next tick.")
                return {"status": "CANCEL_FAILED", "asset": asset,
                        "failed_cancels": _cancel_failed}

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

    async def _maker_first_attempt(self, pid: str, asset: str, side: str,
                                   base_size: float) -> str:
        """[P270] One post-only attempt joining the touch, resolved INSIDE
        this call. Returns:
          "DONE"          — order left the book (filled or gone); caller
                            recomputes the remaining delta from reconcile
          "FALLBACK"      — no maker fill possible (no bid/ask, placement
                            rejected, e.g. price crossed between quote and
                            place); caller proceeds straight to the cross
          "CANCEL_FAILED" — timeout AND the cancel failed: the maker order
                            may still be live, so the caller must NOT place
                            the cross (two live orders for one delta = the
                            P265 double-order class). Next tick's stale-entry
                            sweep retries.
        Never raises."""
        try:
            from exchange.adapter import OrderRequest
            # Join the touch: post-only BUY at the best bid / SELL at the
            # best ask. Mid-based pricing is NOT safe here — on a 1-tick
            # spread, mid rounds onto the opposite touch and the venue
            # rejects the post-only (correctly).
            try:
                bb = self._adapter._client.get_best_bid_ask(product_ids=[pid])
                books = _g(bb, "pricebooks") or []
                book = books[0] if books else None
                bids = _g(book, "bids") or []
                asks = _g(book, "asks") or []
                best_bid = _f(_g(bids[0], "price")) if bids else 0.0
                best_ask = _f(_g(asks[0], "price")) if asks else 0.0
            except Exception as e:
                self._maker_log_once(asset, f"no best_bid_ask "
                                     f"({type(e).__name__}) — taker fallback")
                return "FALLBACK"
            px = best_bid if side == "BUY" else best_ask
            if not (px and math.isfinite(px) and px > 0):
                self._maker_log_once(asset, "empty book side — taker fallback")
                return "FALLBACK"
            req = OrderRequest(symbol=pid, side=side, size=base_size,
                               order_type="LIMIT", price=px, post_only=True)
            res = await self._adapter.place_order(req)
            if not res.success:
                # includes INVALID_LIMIT_PRICE_POST_ONLY when the touch moved
                # through our price between quote and place — cross instead
                logger.info(f"[COINBASE-MAKER] {asset}: post-only rejected "
                            f"({res.error_code}) — taker fallback")
                return "FALLBACK"
            oid = str(res.order_id or "")
            if not oid:
                # accepted but no id to poll/cancel by: treat as DONE and let
                # the caller's reconcile+remainder logic settle the truth —
                # never place a blind second order beside an untracked one.
                logger.warning(f"[COINBASE-MAKER] {asset}: accepted post-only "
                               f"carried no order_id — resolving via "
                               f"reconcile only")
                return "DONE"
            deadline = time.monotonic() + self._maker_wait_sec
            while time.monotonic() < deadline:
                await asyncio.sleep(5.0)
                try:
                    open_orders = await self._adapter.fetch_open_orders(pid)
                except Exception:
                    continue  # transient list failure — keep waiting
                still_open = any(
                    str(_g(o, "order_id") or _g(o, "id") or "") == oid
                    for o in (open_orders or []))
                if not still_open:
                    logger.info(f"[COINBASE-MAKER] {asset}: post-only left "
                                f"the book within the window (filled at "
                                f"0bps maker)")
                    return "DONE"
            if await self._adapter.cancel_order(oid, pid):
                logger.info(f"[COINBASE-MAKER] {asset}: unfilled after "
                            f"{self._maker_wait_sec:.0f}s — cancelled, "
                            f"crossing the remainder")
                return "DONE"
            logger.warning(f"[COINBASE-MAKER] {asset}: timeout AND cancel "
                           f"FAILED — order {oid} may be live; refusing to "
                           f"place the cross (double-order risk, P265). "
                           f"Next tick retries.")
            return "CANCEL_FAILED"
        except Exception as e:  # noqa: silent-swallow — a maker attempt must never break execution; fallback is the stated consequence (logged)
            logger.warning(f"[COINBASE-MAKER] {asset}: attempt error "
                           f"({type(e).__name__}: {e}) — taker fallback")
            return "FALLBACK"

    def _maker_log_once(self, asset: str, msg: str) -> None:
        key = f"maker:{asset}:{msg}"
        if not hasattr(self, "_maker_warned"):
            self._maker_warned: set = set()
        if key not in self._maker_warned:
            self._maker_warned.add(key)
            logger.info(f"[COINBASE-MAKER] {asset}: {msg}")

    async def execute_target(self, asset: str, target_signed_contracts: int,
                             order_type: str = "LIMIT",
                             urgent: bool = False) -> Dict[str, Any]:
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
        # [P253] Same stale-snapshot rule as manage_to_signal. execute_target
        # is also called DIRECTLY (FORCE_FLAT, sleeve_fast_risk_action), and
        # sizing `delta` off a failed reconcile's last-known snapshot can
        # overshoot into an OPPOSITE position on a venue with no reduce_only.
        # P141: never trade on last-known state — refuse and retry next pass.
        if not self._reconcile_ok:
            logger.warning(f"[COINBASE_SLEEVE] execute_target {asset}: skip "
                           f"(reconcile failed; not acting on stale snapshot)")
            return {"status": "SKIPPED_STALE", "asset": asset,
                    "reason": "reconcile_failed"}
        cur = self.signed_contracts(asset)
        delta = int(round(target_signed_contracts - cur))
        # [P265] The no-order paths must still sweep stale ENTRY limits. An
        # unfilled marketable limit from a previous tick otherwise rests
        # through every NOOP/BLOCKED tick and can fill against a dead signal
        # — including UNDER HALT, opening a position the halt never saw.
        if delta == 0:
            try:
                _pid_noop = self._adapter.to_venue_symbol(asset, "perp")
                await self._cancel_stale_entry_orders(_pid_noop, asset)
            except Exception as e:  # noqa: silent-swallow — the sweep must never break the NOOP path; failures are logged inside the helper
                logger.warning(f"[COINBASE_SLEEVE] {asset}: NOOP-tick stale "
                               f"order sweep failed: {type(e).__name__}: {e}")
            return {"status": "NOOP", "asset": asset, "contracts": cur}
        ok, reason = self.can_trade(asset, delta)
        if not ok:
            logger.warning(f"[COINBASE_SLEEVE] execute_target {asset} blocked: {reason}")
            try:
                _pid_blk = self._adapter.to_venue_symbol(asset, "perp")
                await self._cancel_stale_entry_orders(_pid_blk, asset)
            except Exception as e:  # noqa: silent-swallow — same rule as the NOOP sweep above
                logger.warning(f"[COINBASE_SLEEVE] {asset}: BLOCKED-tick stale "
                               f"order sweep failed: {type(e).__name__}: {e}")
            return {"status": "BLOCKED", "asset": asset, "reason": reason}
        side = "BUY" if delta > 0 else "SELL"
        n_contracts = abs(delta)
        try:
            pid = self._adapter.to_venue_symbol(asset, "perp")
            await self._cancel_resting_orders(pid, asset)
            # [P265] REFUSE on an unknown contract size — never `or 1.0`.
            # The fallback fabricated a UNIT: base_size became a contract
            # count wearing base-unit clothes, and if the adapter's own
            # lookup then succeeded (the two calls are ms apart), it computed
            # int(round(1/0.1)) = 10 contracts for ETH — a 10x order that
            # fills inside buying power, with can_trade having gated only the
            # intended 1-contract delta. (Latent while all three pids sit in
            # the fallback table; armed the day the 20DEC30 contract rolls.)
            cs = self._adapter._contract_size(pid)
            if not cs or cs <= 0:
                logger.error(f"[COINBASE_SLEEVE] execute_target {asset}: "
                             f"contract size UNKNOWN for {pid} — refusing to "
                             f"fabricate a unit (a wrong unit is a 10-100x "
                             f"order)")
                return {"status": "ERROR", "asset": asset,
                        "reason": f"no_contract_size:{pid}"}
            base_size = n_contracts * cs  # adapter converts base->contracts
            # marketable limit: cross slightly so it fills; adapter rounds to tick
            prod = self._adapter._client.get_product(product_id=pid)
            mid = _f(_g(prod, "mid_market_price") or _g(prod, "price"))
            # [P253] A failed/empty get_product yields mid=0.0, and a SELL
            # limit priced at ~0 is "sell at any price". _notional_usd guards
            # exactly this case; the ORDER path did not. No price -> no order.
            if not (mid and math.isfinite(mid) and mid > 0):
                logger.error(f"[COINBASE_SLEEVE] execute_target {asset}: no "
                             f"usable price from venue (mid={mid!r}) — "
                             f"refusing to place a priceless order")
                return {"status": "ERROR", "asset": asset,
                        "reason": f"no_price:mid={mid!r}"}
            # [P270] Maker-first ladder: post-only at the touch (0bps) ->
            # poll inside this call -> cancel -> cross whatever remains at
            # taker (3bps). Skipped when urgent (FORCE_FLAT and the fast-risk
            # watchdog need immediacy, not fee savings) or for non-LIMIT
            # callers. Resolved remainder from a fresh reconcile so a partial
            # maker fill is crossed exactly once, never double-ordered.
            maker_note = None
            if self._maker_first and not urgent and order_type == "LIMIT":
                _mk = await self._maker_first_attempt(pid, asset, side,
                                                      base_size)
                if _mk == "CANCEL_FAILED":
                    return {"status": "FAILED", "asset": asset, "side": side,
                            "reason": "maker_cancel_failed_no_cross"}
                if _mk == "DONE":
                    self.reconcile_positions()
                    if not self._reconcile_ok:
                        # cannot see the book after the maker attempt —
                        # crossing blind can double-fill (P141/P253 rule)
                        return {"status": "SKIPPED_STALE", "asset": asset,
                                "reason": "post_maker_reconcile_failed"}
                    cur = self.signed_contracts(asset)
                    delta = int(round(target_signed_contracts - cur))
                    if delta == 0:
                        logger.info(f"[COINBASE_SLEEVE] execute_target "
                                    f"{asset} {side} filled as MAKER (0bps) "
                                    f"-> now={cur}ct")
                        return {"status": "OK", "asset": asset, "side": side,
                                "contracts": n_contracts, "maker": True,
                                "position_after": cur}
                    # partial (or none) filled as maker — cross the remainder
                    side = "BUY" if delta > 0 else "SELL"
                    n_contracts = abs(delta)
                    base_size = n_contracts * cs
                    maker_note = "partial_maker"
                    # refresh the price for the cross: the maker window may
                    # have been most of a minute, and the old mid is what the
                    # priceless-order guard above certified, not this one
                    prod = self._adapter._client.get_product(product_id=pid)
                    mid = _f(_g(prod, "mid_market_price") or _g(prod, "price"))
                    if not (mid and math.isfinite(mid) and mid > 0):
                        return {"status": "ERROR", "asset": asset,
                                "reason": f"no_price_post_maker:mid={mid!r}"}
                # _mk == "FALLBACK": fall through to the marketable cross
            px = mid * (1.002 if side == "BUY" else 0.998)
            req = OrderRequest(symbol=pid, side=side, size=base_size,
                               order_type=order_type, price=px, post_only=False)
            res = await self._adapter.place_order(req)
            self.reconcile_positions()
            logger.info(f"[COINBASE_SLEEVE] execute_target {asset} {side} "
                        f"{n_contracts}ct -> success={res.success} "
                        f"now={self.signed_contracts(asset)}ct"
                        + (f" ({maker_note})" if maker_note else ""))
            return {"status": "OK" if res.success else "FAILED", "asset": asset,
                    "side": side, "contracts": n_contracts, "order": res,
                    "maker_note": maker_note,
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

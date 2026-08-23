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
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from exchange.symbol_mapping import from_venue_symbol, to_venue_symbol

# [P378] How long the sleeve may be unable to READ the venue before it says so
# at a severity the operator must act on. reconcile_positions() is attempted at
# least every ~30s by the FastRiskTick loop, so 15 min is ~30 failed attempts —
# far past any plausible blip, and short enough to matter while positions are
# open and unmanageable.
RECONCILE_BLIND_ALERT_SEC = 15 * 60.0
RECONCILE_BLIND_REALERT_SEC = 60 * 60.0

_HTTP_STATUS_RE = re.compile(r"\b([45]\d\d)\b")


def _http_status(exc: Exception) -> Optional[int]:
    """Best-effort HTTP status for an SDK exception.

    [P378] `requests.HTTPError` carries `.response.status_code`; the Coinbase
    SDK re-raises shapes that sometimes only have it in the message text.
    Returns None when neither is available, and None classifies TRANSIENT —
    the safe direction, because the only thing the class changes here is how
    loudly we speak (never whether we retry).
    """
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int):
        return code
    m = _HTTP_STATUS_RE.search(str(exc))
    return int(m.group(1)) if m else None

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

    # [P383] The order type an URGENT exit (FORCE_FLAT kill switch, the
    # FastRiskTick EXIT_ONLY/REDUCE_50 watchdog — every `urgent=True` caller)
    # is sent as. ONE named decision, not a literal at the call site.
    #
    # WHY MARKET and not the 0.2%-through GTC LIMIT the non-urgent cross
    # uses: an urgent exit fires in exactly the condition — a fast move —
    # where a limit 0.2% through a stale mid can be LEFT RESTING UNFILLED,
    # and execute_target has already SWEPT the protective stop by the time it
    # places the cross (P197/P207 ordering). An emergency exit that can rest
    # unfilled beside a swept stop is not an exit; the P382 stop follow-up
    # bounds the stranded window but does not remove it. MARKET is IOC at
    # the venue (market_market_ioc): it fills or it dies, it never rests.
    # The fee difference (taker either way) is nil; what is bought is
    # certainty of NOT holding through the move the watchdog fired on.
    # Non-urgent behaviour is untouched: maker ladder, then the LIMIT cross.
    URGENT_ORDER_TYPE = "MARKET"

    def __init__(self, adapter, assets=("BTC", "ETH", "SOL"),
                 max_sleeve_drawdown_pct: float = 0.15,
                 max_contracts_per_asset: int = 1,
                 protective_stop_pct: float = 0.0,
                 protective_stop_assets=None,
                 flip_persist_ticks: int = 0,
                 max_net_exposure: Optional[float] = None,
                 max_asset_exposure: Optional[Dict[str, float]] = None,
                 maker_first: bool = False,
                 maker_wait_sec: float = 45.0,
                 maker_reprice: bool = False,
                 max_contracts_by_asset: Optional[Dict[str, int]] = None,
                 target_fraction_by_asset: Optional[Dict[str, float]] = None) -> None:
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
        # [P291] ONE reprice step at the half-way mark of an unfilled maker
        # window. Both live orders on 2026-08-16 posted at the touch, sat the
        # full 45s and crossed as taker: a passive order that is no longer at
        # the FRONT of the book will not fill, and waiting the window out buys
        # nothing but staleness. When on, the window is SPLIT (never extended
        # — a longer wait is a staler signal, the P265 dead-signal-fill
        # class): poll to wait/2, and if the touch has moved past our resting
        # price, cancel and re-post ONCE at the new touch. A FAILED cancel
        # refuses BOTH the re-post and the cross (P287 double-order rule), so
        # at most one post-only is ever live at a time. Default OFF: flag-off
        # is byte-identical to the P270 single-post ladder.
        self._maker_reprice = bool(maker_reprice)
        # [P272] Per-asset contract caps — the vol-parity PREP (flat +/-1 is
        # accidentally ~2.4x more BTC dollar-risk than ETH per contract; the
        # 2025-26 vol-targeting evidence says size inversely to vol). An
        # asset absent from the dict falls back to the scalar cap, so an
        # empty dict is byte-identical to today. Raising any value is the
        # September sizing activation (own P-entry + operator flip) — note
        # the driver's target_for_signal still emits +/-1 targets, so caps
        # alone change nothing until that activation also sizes targets.
        self._max_contracts_by_asset: Dict[str, int] = {
            k: int(v) for k, v in (max_contracts_by_asset or {}).items()}
        # [P274] equity-scaled target fractions (of sleeve equity, per
        # asset). Empty = fixed-contract sizing only. Values are clamped to
        # [0, 0.25] — a fat-fingered 1.5 must not size a 10x book; 0.25 is
        # the largest per-asset gross cap in the standing policy (P210).
        self._target_fraction_by_asset: Dict[str, float] = {
            k: min(0.25, max(0.0, float(v)))
            for k, v in (target_fraction_by_asset or {}).items()}
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
        # [P287] When the TRUE (portfolio) equity was last actually read from
        # the venue. `_last_equity_usd` serves last-known-forever on a
        # portfolio-endpoint outage, and before this timestamp existed no
        # consumer could tell fresh from days-stale: update_risk computed a
        # live-looking drawdown from a frozen number (the 15% halt could not
        # fire on a real drawdown), and _sized_contracts sized real orders
        # off stale equity. The P156 rule — every stored reference needs a
        # staleness bound — applied to equity. None = never read this
        # process (in production that coexists with equity 0.0, which every
        # consumer already treats as unknown).
        self._last_equity_ts: Optional[float] = None
        self._cb_portfolio_uuid: Optional[str] = None  # [P153] cached Default portfolio uuid
        # True only after a SUCCESSFUL venue reconcile this call. Autonomous
        # management refuses to act when this is False (don't trade on a stale
        # snapshot after an API timeout). See manage_to_signal.
        self._reconcile_ok: bool = False
        # [P378] Blind-since tracking is a TIMESTAMP, not a count: reconcile is
        # called from three loops at different cadences (30s FastRiskTick, the
        # 4H manage driver, snapshot), so "N consecutive failures" would mean a
        # different duration depending on which one happened to call.
        self._reconcile_blind_since: Optional[float] = None
        # [P378] None means NEVER ALERTED. 0.0 would be a timestamp
        # meaning 1970, and `now - 0.0 >= interval` is a different question
        # from "have we alerted yet" — the missing-vs-neutral collapse (P2),
        # which is exactly what the first draft got wrong here.
        self._reconcile_last_alert_at: Optional[float] = None
        # --- isolated sleeve risk guard (the Kraken existence-fuse equivalent,
        # scoped to Coinbase ONLY; never touches the global fuse) ---
        self._max_sleeve_drawdown_pct = float(max_sleeve_drawdown_pct)
        self._last_dd_pct: float = 0.0  # [P265] last real dd; served when equity is unknown
        self._max_contracts_per_asset = int(max_contracts_per_asset)
        self._sleeve_start_equity: Optional[float] = None
        # [P293h] cumulative deposits/withdrawals, excluded from PnL
        self._external_flow_usd: float = 0.0
        self._last_equity_for_flow: Optional[float] = None
        self._last_equity_for_flow_ts: Optional[float] = None  # [P294]
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
            # [P378] Announce recovery and RESET, or a long-lived process
            # drifts into a permanent alert state that no longer describes the
            # present (P265f/P329). Silent on the normal path: this only fires
            # when the sleeve had actually gone blind.
            _since = getattr(self, "_reconcile_blind_since", None)
            if _since is not None:
                _blind_min = (time.time() - _since) / 60.0
                logger.warning(
                    f"[COINBASE_SLEEVE] reconcile RECOVERED after "
                    f"{_blind_min:.1f} min blind — the venue is readable again "
                    f"and the sleeve can manage positions")
                self._reconcile_blind_since = None
                self._reconcile_last_alert_at = None
            return out
        except Exception as e:
            self._reconcile_ok = False
            # [P331] Same treatment as the equity path below. P304 stripped the
            # venue's HTML error page from the SDK logger and from the equity
            # warning, and left THIS one — so a 502 still pasted a full
            # Cloudflare page into the log (observed 2026-08-20 03:09:07, ~30
            # lines of markup around one useful status code). A mitigation
            # applied to one instance of a class is not applied to the class
            # (P171 for the BOM, P226 for the scanners, P323 for timestamps).
            try:
                from infra.vendor_log_hygiene import strip_html_body as _shb
            except Exception:  # noqa: silent-swallow — hygiene helper is optional; fall back to raw text
                def _shb(msg: str) -> str:
                    # signature must MATCH strip_html_body, or mypy flags the
                    # conditional variants [misc] — and a fallback that does
                    # not type like the thing it replaces is a latent bug too.
                    return msg
            # [P378] CLASSIFY THE FAILURE — FOR SEVERITY ONLY, NEVER FOR
            # SUPPRESSION, and that distinction is the whole of this fix.
            #
            # P345's contract says a PERMANENT class (401/403/404/422)
            # suppresses for the PROCESS. Wiring that here would be the P329
            # mistake mirrored: the live 403 that produced this entry
            # (2026-08-22 06:12:20, PERMISSION_DENIED "User does not have
            # access to portfolio") RECOVERED on the very next cycle, so
            # suppressing would have converted a one-cycle blip into a
            # permanently blind sleeve holding ETH+SOL. A failure to READ is
            # never evidence that the next read fails, and this reader guards
            # live positions — so it retries on its normal cadence, always.
            # Pinned by test.
            _status = _http_status(e)
            try:
                from infra.failure_policy import classify_external_failure
                _cls = classify_external_failure(
                    status=_status, message=str(e)).failure_class.name
            except Exception:  # noqa: silent-swallow — classification is advisory only
                _cls = "UNCLASSIFIED"

            _now = time.time()
            _blind_since = getattr(self, "_reconcile_blind_since", None)
            if _blind_since is None:
                _blind_since = _now
                self._reconcile_blind_since = _now
            _blind_sec = _now - _blind_since
            _held = {a: p.get("signed_contracts")
                     for a, p in self._last_positions.items()
                     if p.get("signed_contracts")}

            _base = (f"[COINBASE_SLEEVE] reconcile failed: "
                     f"{type(e).__name__}: {_shb(str(e))}; "
                     f"returning last snapshot")

            # A single blip must stay noise, or the alert becomes wallpaper and
            # gets ignored (P202/P303) — today's 403 was exactly one cycle.
            # What is actionable is SUSTAINED blindness, and it is far more
            # actionable while the sleeve holds positions it cannot see, exit
            # or flip: manage_to_signal returns SKIPPED_STALE for every one of
            # them, at INFO, forever, and nothing else escalates unless
            # FastRiskTick happens to trigger (P329 only fires on an action).
            _last_alert = getattr(self, "_reconcile_last_alert_at", None)
            if (_blind_sec >= RECONCILE_BLIND_ALERT_SEC
                    and (_last_alert is None
                         or _now - _last_alert >= RECONCILE_BLIND_REALERT_SEC)):
                self._reconcile_last_alert_at = _now
                _hint = ("credentials/permissions — this does NOT self-heal "
                         "and needs an operator"
                         if _cls == "PERMANENT"
                         else "venue-side; may clear on its own")
                if _held:
                    logger.critical(
                        f"{_base}. BLIND {_blind_sec / 60.0:.0f} min while "
                        f"HOLDING {_held} — signal exits and flips are NOT "
                        f"happening (every manage returns SKIPPED_STALE); only "
                        f"the venue-resting stop still protects these. "
                        f"class={_cls} status={_status} ({_hint})")
                else:
                    logger.error(
                        f"{_base}. BLIND {_blind_sec / 60.0:.0f} min; the book "
                        f"is FLAT so nothing is at risk, but no entry can be "
                        f"taken until it clears. class={_cls} status={_status} "
                        f"({_hint})")
            else:
                logger.warning(f"{_base} [class={_cls} status={_status}]")
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
                    self._last_equity_ts = time.time()  # [P287] fresh TRUE read
                    return tb
                else:
                    logger.warning(
                        "[COINBASE_SLEEVE] portfolio breakdown returned "
                        f"total_balance={tb!r} (not > 0) — treating equity as "
                        "unknown this pass")
        except Exception as e:
            # [P304] Truncated: the vendor attaches its full HTML error page
            # to the exception, and this line is what an operator actually
            # reads. The status code and reason survive; the CSS does not.
            try:
                from infra.vendor_log_hygiene import strip_html_body as _shb
            except Exception:  # noqa: silent-swallow — hygiene helper is optional; fall back to raw text
                def _shb(msg: str) -> str:
                    # signature must MATCH strip_html_body, or mypy flags the
                    # conditional variants [misc] — and a fallback that does
                    # not type like the thing it replaces is a latent bug too.
                    return msg
            logger.warning(
                f"[COINBASE_SLEEVE] portfolio equity fetch failed: "
                f"{type(e).__name__}: {_shb(str(e))}")
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

    # [P287] Staleness bound on the stored equity (P156 rule). 2x the 4H tick,
    # matching the FastRiskTick anchor bound's reasoning: one late refresh is
    # tolerated, two consecutive misses are not.
    _EQUITY_MAX_AGE_SEC = 8 * 3600.0

    def sleeve_equity_age_sec(self) -> Optional[float]:
        """Seconds since the last SUCCESSFUL true-equity read, or None if no
        fresh read has been stamped this process (in production that state
        coexists with equity 0.0 = unknown; tests that monkeypatch
        sleeve_equity_usd legitimately never stamp it). Exposed via snapshot()
        so main.py's fuse feed can bound its delta on the same clock."""
        ts = getattr(self, "_last_equity_ts", None)
        return (time.time() - ts) if ts else None

    def _equity_is_stale(self) -> bool:
        """True only when a stamp EXISTS and is over the bound. A missing
        stamp is deliberately NOT stale: in production it always accompanies
        equity<=0 (already refused everywhere), and treating it as stale
        would flip every monkeypatched-equity test fixture — and any future
        caller that fakes equity without faking the clock — into the
        degraded path."""
        age = self.sleeve_equity_age_sec()
        return age is not None and age > self._EQUITY_MAX_AGE_SEC

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
                    "degraded": True,
                    "equity_age_sec": self.sleeve_equity_age_sec()}
        # [P287] Stale is not fresh. sleeve_equity_usd() above serves the
        # LAST-KNOWN value on a portfolio-endpoint outage, so `eq > 0` alone
        # proved nothing about freshness — a multi-day outage froze the
        # drawdown at its pre-outage value while positions bled (the halt
        # could not fire), and would also have anchored a first-call baseline
        # to a stale number. HOLD the last real drawdown; never recompute
        # from stale equity, never set the baseline from it, and (as with
        # eq<=0) never touch the halt state — a tripped halt stays tripped.
        if self._equity_is_stale():
            _age = self.sleeve_equity_age_sec() or 0.0
            logger.warning(
                f"[COINBASE_SLEEVE] update_risk: equity is STALE "
                f"({_age/3600.0:.1f}h old > "
                f"{self._EQUITY_MAX_AGE_SEC/3600.0:.0f}h bound) — holding the "
                f"last drawdown reading; the 15% halt cannot be evaluated on "
                f"a frozen number (P287/P156)")
            return {"equity_usd": eq,
                    "drawdown_pct": self._last_dd_pct,
                    "halted": self._halted, "halt_reason": self._halt_reason,
                    "degraded": True,
                    "equity_age_sec": _age}
        if self._sleeve_start_equity is None:
            self._sleeve_start_equity = eq
            logger.info(f"[COINBASE_SLEEVE] risk baseline set: ${eq:,.2f}")
            self._persist_state()  # [P150] anchor survives restart
        # [P294] Drawdown is measured against INVESTED CAPITAL, not the
        # inception anchor alone. A deposit raises equity without earning it,
        # so against a fixed anchor the ratio goes NEGATIVE and the loss cap
        # silently loosens: measured live 2026-08-18, after the 2026-08-16
        # deposit of $7,074.27 the anchor was $3,997.75 against equity of
        # $10,847.88, so `dd` read -171.3% and the 15% halt could not fire
        # until equity fell to $3,398 — a 68.7% loss of the capital actually
        # at risk. The control designed to cap losses at 15% was disarmed by
        # a bank transfer.
        #
        # P274 recorded the deposit direction as "conservative" and it IS for
        # the P0 fuse (a deposit reads as a gain there); for THIS control it
        # is the opposite. P287's note below covers the withdrawal direction
        # and prescribed a manual re-anchor — netting the flow out fixes both
        # ends at once, and the withdrawal case stops needing an operator
        # step at all once the P293h detector has seen it.
        #
        # With no recorded flows (`_external_flow_usd == 0.0`, every sleeve
        # that has never seen a transfer) this is byte-identical to the old
        # arithmetic.
        _flows = float(getattr(self, "_external_flow_usd", 0.0) or 0.0)
        _basis = (self._sleeve_start_equity or 0.0) + _flows
        dd = 0.0
        if self._sleeve_start_equity and self._sleeve_start_equity > 0:
            if _basis > 0:
                dd = (_basis - eq) / _basis
            else:
                # A withdrawal at or beyond the whole invested base leaves no
                # denominator. Hold the last real drawdown rather than divide
                # into nonsense, and never touch the halt state (same rule as
                # the stale/unknown-equity branches above).
                logger.warning(
                    "[COINBASE_SLEEVE] drawdown basis <= 0 "
                    "(start=%.2f flows=%+.2f) — holding last dd %.4f",
                    self._sleeve_start_equity or 0.0, _flows, self._last_dd_pct)
                dd = self._last_dd_pct
        self._last_dd_pct = dd
        # [P287] WITHDRAWAL direction, recorded (P274 documented deposits
        # only): a capital withdrawal from the perp portfolio lowers eq
        # against the inception anchor and reads as drawdown here — a >15%
        # withdrawal trips the sticky, persisted halt with one bank transfer.
        # That is deliberately NOT auto-corrected (the halt must stay
        # conservative; distinguishing a withdrawal from a loss needs venue
        # transfer records this code does not read). Operator procedure after
        # any withdrawal: reset_halt() + a recorded re-anchor decision.
        if dd >= self._max_sleeve_drawdown_pct and not self._halted:
            self._halted = True
            self._halt_reason = (f"sleeve drawdown {dd:.1%} >= "
                                 f"{self._max_sleeve_drawdown_pct:.0%}")
            logger.error(f"[COINBASE_SLEEVE] HALTED: {self._halt_reason}")
            self._persist_state()  # [P150] sticky halt survives restart
        return {"equity_usd": eq, "drawdown_pct": dd,
                "halted": self._halted, "halt_reason": self._halt_reason,
                "equity_age_sec": self.sleeve_equity_age_sec()}

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
        # [P287] The exposure fractions keep their fail-OPEN convention on a
        # stale denominator (a risk control that fires on missing data is a
        # data outage that stops trading — P208), but staleness is no longer
        # silent: warn once per staleness episode with the age.
        if self._equity_is_stale():
            if not getattr(self, "_exposure_stale_warned", False):
                self._exposure_stale_warned = True
                _age = self.sleeve_equity_age_sec() or 0.0
                logger.warning(
                    f"[COINBASE_SLEEVE] sleeve_exposure: equity denominator is "
                    f"STALE ({_age/3600.0:.1f}h old) — exposure fractions are "
                    f"computed against a frozen ${eq:,.2f} (fail-open kept, "
                    f"P287)")
        else:
            self._exposure_stale_warned = False
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
            "equity_age_sec": self.sleeve_equity_age_sec(),  # [P287]
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
        # [P272] per-asset cap wins when configured; scalar is the fallback.
        # getattr-defended: this is the live order path (P85) — pre-P272
        # pickles/fixtures without the dict must not refuse every order.
        # [P274] a fraction-sized asset's cap IS its equity-scaled size
        # (cap == target at any equity) — without this, the fixed {1,3,1}
        # caps would block the scaled targets the moment capital arrives.
        # An unpriceable tick returns None here and the FIXED cap binds
        # (staleness resolves toward the smaller, known-safe book).
        try:
            _sized_cap = self._sized_contracts(asset)
        except Exception:
            _sized_cap = None
        _cap = _sized_cap if _sized_cap is not None else int(
            getattr(self, "_max_contracts_by_asset", {}).get(
                asset, self._max_contracts_per_asset))
        if resulting > _cap and resulting > abs(cur):
            return False, (f"coinbase_contract_cap: {resulting:.0f} > "
                           f"{_cap} for {asset}")

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

    def _fill_quality_path(self) -> str:
        return os.path.join(self._data_dir(), "fill_quality.jsonl")

    def _record_fill_quality(self, asset: str, order_id: Optional[str],
                             side: str, contracts: Optional[int],
                             liquidity: str, urgent: bool,
                             decision_mid: Optional[float] = None,
                             decision_bid: Optional[float] = None,
                             decision_ask: Optional[float] = None,
                             order_payload: Optional[Dict[str, Any]] = None,
                             ) -> None:
        """[P290] Append one realized-fill record to data/fill_quality.jsonl.

        The instrument P289 found missing: no realized-fill cost measurement
        existed for CDE (the P169 [FILL_VS_MID] line lives on the dormant
        Kraken path), so the CDE spread table is certified by order-book
        probes only. Every field is measurement-or-absent (P2): a fill whose
        order payload cannot be read records status="unresolved" with NO
        fill/slippage fields — absence must never read as a measurement.
        Sign convention: realized_slippage_bps is positive when the fill was
        WORSE than the decision mid for our side (BUY above mid / SELL below
        mid both read positive = paid).

        Observation-only (Iron Law 7): this method never raises — a ledger
        failure logs a WARNING and the order path continues byte-identically.
        """
        try:
            now = time.time()
            rec: Dict[str, Any] = {
                "ts": now,
                "iso": datetime.now(timezone.utc).isoformat(),
                "asset": asset,
                "order_id": order_id,
                "side": side,
                "contracts": contracts,
                "liquidity": liquidity,
                "urgent": bool(urgent),
                "decision_mid": decision_mid,
                "decision_bid": decision_bid,
                "decision_ask": decision_ask,
                "decision_spread_bps": None,
                "status": "unresolved",
                "fill_avg_price": None,
                "realized_slippage_bps": None,
                "fees_usd": None,
                "filled_size_raw": None,
            }
            try:
                if (decision_bid and decision_ask and decision_bid > 0
                        and decision_ask >= decision_bid):
                    _m = (decision_bid + decision_ask) / 2.0
                    rec["decision_spread_bps"] = round(
                        (decision_ask - decision_bid) / _m * 1e4, 3)
            except Exception:  # noqa: silent-swallow — spread context is optional; the record stays honest without it
                pass
            op = order_payload or {}
            _status = str(op.get("status") or "").upper()
            _fill = None
            for k in ("average_filled_price", "avg_price",
                      "average_fill_price"):
                v = op.get(k)
                try:
                    if v is not None and float(v) > 0:
                        _fill = float(v)
                        break
                except (TypeError, ValueError):
                    continue
            # filled_size recorded VERBATIM — no unit guess (P219: silent
            # unit calibration is how mislabels survive; the reader sees raw).
            rec["filled_size_raw"] = op.get("filled_size")
            for k in ("total_fees", "fee", "total_fees_usd"):
                v = op.get(k)
                try:
                    if v is not None:
                        rec["fees_usd"] = float(v)
                        break
                except (TypeError, ValueError):
                    continue
            _has_fill_size = False
            try:
                _has_fill_size = float(op.get("filled_size") or 0) > 0
            except (TypeError, ValueError):
                _has_fill_size = False
            if _fill is not None and (_status == "FILLED" or _has_fill_size):
                rec["status"] = "filled"
                rec["fill_avg_price"] = _fill
                if decision_mid and decision_mid > 0:
                    _dir = 1.0 if side == "BUY" else -1.0
                    rec["realized_slippage_bps"] = round(
                        (_fill - decision_mid) / decision_mid * 1e4 * _dir, 3)
            path = self._fill_quality_path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info(
                f"[FILL-QUALITY] {asset} {side} {liquidity} "
                f"status={rec['status']} fill={rec['fill_avg_price']} "
                f"mid={decision_mid} slip={rec['realized_slippage_bps']}bps "
                f"spread={rec['decision_spread_bps']}bps oid={order_id}")
        except Exception as e:  # noqa: silent-swallow — logs; an observation ledger must never touch the order path (Iron Law 7)
            logger.warning(f"[FILL-QUALITY] {asset}: record failed "
                           f"({type(e).__name__}: {e}) — order path "
                           f"unaffected")

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
            try:  # [P293h] cumulative external transfers
                self._external_flow_usd = float(
                    st.get("external_flow_usd") or 0.0)
            except (TypeError, ValueError):  # noqa: silent-swallow
                self._external_flow_usd = 0.0
            try:  # [P294] flow reference across restarts (see _detect_external_flow)
                _lef = st.get("last_equity_for_flow")
                self._last_equity_for_flow = (
                    float(_lef) if _lef is not None else None)
                _lts = st.get("last_equity_for_flow_ts")
                self._last_equity_for_flow_ts = (
                    float(_lts) if _lts is not None else None)
            except (TypeError, ValueError):  # noqa: silent-swallow
                # An unreadable reference means "no reference": the next tick
                # takes the prev-is-None path and simply re-anchors. Never
                # fabricate one — a wrong reference invents a transfer.
                self._last_equity_for_flow = None
                self._last_equity_for_flow_ts = None
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
                    # [P293h] without this a restart forgets the
                    # transfer and re-books it as profit
                    "external_flow_usd": getattr(
                        self, "_external_flow_usd", 0.0),
                    # [P294] Persisting the flow reference is what makes a
                    # transfer during DOWNTIME detectable. Without it a fresh
                    # process always took the `prev is None` early return, so
                    # the common operational sequence (wire the funds, then
                    # restart) was exactly the one the detector was blind to.
                    "last_equity_for_flow": getattr(
                        self, "_last_equity_for_flow", None),
                    "last_equity_for_flow_ts": getattr(
                        self, "_last_equity_for_flow_ts", None),
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

    # [P293h] A per-tick equity step larger than this multiple of the current
    # position notional cannot be trading PnL on a few nano contracts. Set
    # deliberately high (a 50% adverse move on the whole book in one 4H tick
    # is already extreme) so ONLY unmistakable transfers are caught.
    FLOW_DETECT_NOTIONAL_MULT = 0.5
    # Floor for the flat/near-flat case: with no position, ANY material
    # equity change is a transfer or a fee, and fees are cents.
    FLOW_DETECT_MIN_USD = 200.0

    def _position_notional_usd(self) -> float:
        """Best-effort USD notional of the current book. 0.0 if unknown.

        Uses the adapter's contract size — the same source `_sizing_inputs`
        uses — so the detector and the sizer agree on what a contract is
        worth (P172). An unpriceable leg contributes 0, which makes the
        detector MORE conservative: a smaller notional means a smaller
        allowed PnL step, so an ambiguous case is more likely to be left
        classified as PnL rather than silently adjusted away.
        """
        total = 0.0
        for a, p in (self._last_positions or {}).items():
            try:
                ct = abs(_f(p.get("signed_contracts")))
                px = _f(p.get("current_price")) or _f(p.get("entry_vwap"))
                if not (ct and px):
                    continue
                cs = 0.0
                try:
                    from exchange.symbol_mapping import to_venue_symbol
                    _pid = to_venue_symbol(a, "coinbase", "perp")
                    cs = _f(self._adapter._contract_size(_pid))
                except Exception:  # noqa: silent-swallow — see docstring
                    cs = 0.0
                if cs:
                    total += ct * px * cs
            except Exception:  # noqa: silent-swallow — see docstring
                continue
        return total

    # [P294] The reference equity is compared across whatever gap actually
    # elapsed, including a restart. A longer gap can accumulate more honest
    # PnL, so the allowed step scales with elapsed 4H periods — capped, or a
    # long outage would widen the bound until nothing is ever detected.
    FLOW_DETECT_MAX_PERIODS = 12.0   # ~2 days of 4H bars

    def _detect_external_flow(self, eq: float, now: Optional[float] = None) -> float:
        """[P293h] Return the size of an unmistakable capital transfer, else 0.

        Returns 0.0 whenever the step is explainable as trading PnL, or when
        there is no previous equity to compare against. Never guesses.

        [P294] The reference is PERSISTED, so a transfer that lands while the
        process is DOWN is detected on the next start. That is not an edge
        case: the wire is a manual operator step (P274) and deploys are
        frequent, so "deposit, then restart" is the NORMAL sequence — and it
        was the one sequence the original detector could not see, because
        `_last_equity_for_flow` lived only in RAM and a fresh process always
        took the `prev is None` early return.

        The tolerance widens with the elapsed gap (see FLOW_DETECT_MAX_PERIODS)
        because a multi-day outage can accumulate more mark-to-market than one
        4H tick. Widening is the CONSERVATIVE direction here: the failure mode
        it protects is "silently reclassify real PnL as a transfer", which
        would hide a loss.
        """
        prev = getattr(self, "_last_equity_for_flow", None)
        if prev is None or not (prev > 0) or not (eq > 0):
            return 0.0
        step = eq - prev
        notional = self._position_notional_usd()
        periods = 1.0
        _prev_ts = getattr(self, "_last_equity_for_flow_ts", None)
        if _prev_ts:
            try:
                _elapsed_h = max(0.0, ((now or time.time()) - float(_prev_ts))) / 3600.0
                periods = min(self.FLOW_DETECT_MAX_PERIODS,
                              max(1.0, _elapsed_h / 4.0))
            except (TypeError, ValueError):  # noqa: silent-swallow
                periods = 1.0  # [P294] unreadable stamp -> the tight bound
        limit = max(self.FLOW_DETECT_MIN_USD,
                    notional * self.FLOW_DETECT_NOTIONAL_MULT * periods)
        if abs(step) <= limit:
            return 0.0
        return step

    def log_pnl_point(self) -> Dict[str, Any]:
        """[P150] Append one forward-PnL record to a JSONL so the sleeve's live
        edge can be JUDGED on real data (not a backtest) over days/weeks. Returns
        the record. Call once per tick. Best-effort; never raises to the caller."""
        eq = self.sleeve_equity_usd()
        start = self._sleeve_start_equity

        # [P293h] SUBTRACT EXTERNAL CAPITAL FLOWS. `pnl = eq - start` counts a
        # DEPOSIT as profit. Measured on the live ledger 2026-08-17: after the
        # 2026-08-16 deposit of $7,074 this reported **+$6,850 (+171%)** while
        # actual trading PnL over the same 65 days was **-$225 (-6.3%)** — a
        # 177-percentage-point misreport, in the flattering direction, on the
        # one number that answers "is this strategy making money".
        #
        # P274/P287 documented the WITHDRAWAL direction (it reads as drawdown
        # against the sticky halt anchor) and prescribed a manual re-anchor;
        # the deposit direction was never handled, and the manual step was not
        # performed for this deposit. So the anchor is stale and the ledger
        # has been wrong since.
        #
        # Detection is bounded, not clever: this sleeve holds at most a few
        # nano contracts, so the largest per-tick PnL physically possible is a
        # fraction of the position notional. An equity step far exceeding that
        # cannot be trading PnL and is a transfer. Ambiguous steps are NEVER
        # classified as flows — the failure direction is "keep counting it as
        # PnL", i.e. the honest-but-noisy reading, never a silent adjustment
        # that could hide a real loss.
        _flow = self._detect_external_flow(eq)
        if _flow:
            self._external_flow_usd = getattr(self, "_external_flow_usd", 0.0) + _flow
            logger.warning(
                "[P293h] EXTERNAL CAPITAL FLOW detected: %+,.2f USD "
                "(equity %,.2f -> %,.2f). Excluded from PnL; cumulative "
                "flow now %+,.2f. Trading PnL is measured NET of transfers.",
                _flow, self._last_equity_for_flow or 0.0, eq,
                self._external_flow_usd)
            self._persist_state()
        self._last_equity_for_flow = eq
        self._last_equity_for_flow_ts = time.time()
        if not _flow:
            # [P294] Keep the reference fresh on ordinary ticks too. Persisting
            # only when a flow fires would leave a stale stamp behind, and the
            # elapsed-gap tolerance is computed FROM that stamp — an old one
            # widens the bound and blinds the next detection. One small atomic
            # write per 4H tick; the flow branch above already persisted.
            self._persist_state()

        _flows = float(getattr(self, "_external_flow_usd", 0.0) or 0.0)
        pnl_usd = (eq - start - _flows) if (start is not None) else 0.0
        # Percent is against invested capital (start + net deposits), not the
        # original anchor — otherwise a deposit shrinks the denominator's
        # meaning and the ratio stops being comparable over time.
        _basis = (start + _flows) if start is not None else 0.0
        pnl_pct = (pnl_usd / _basis) if _basis > 0 else 0.0
        rec = {
            "ts": time.time(),
            "equity_usd": round(eq, 2),
            "start_equity_usd": round(start, 2) if start is not None else None,
            "external_flow_usd": round(_flows, 2),
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

    def target_for(self, asset: str, direction: float,
                   threshold: float = 0.15,
                   conviction: float = 1.0) -> int:
        """[P273] SIZED per-asset target: sign from target_for_signal x the
        configured per-asset size. The configured cap IS the target — the
        book runs at full configured size or flat/reversed; partial sizes
        would need a conviction-magnitude input the +/-1-era driver never
        had. Default size 1 = the historical behavior exactly. Sizing above
        1 was activated 2026-08-16 by explicit operator instruction ("use
        as much money as possible"), bounded by the STANDING policy caps —
        can_trade's per-asset cap, the P208 net budget (0.50) and the P210
        gross caps all still gate every order independently. getattr-
        defended: live order path (P85).

        [P293d option C] `conviction` is the "conviction-magnitude input the
        +/-1-era driver never had" this docstring names above. It is the ONLY
        channel by which the 22 non-DECIDE agents can reach an order: their
        entire output is exposure modulation, which is discarded twice
        downstream (the tranche overwrite at integration_v36.py:1678, then
        this method's sign-based sizing). Default 1.0 is byte-identical to
        every prior release; it only ever REDUCES magnitude (advisors may
        express doubt, never leverage) and never changes the sign."""
        sign = self.target_for_signal(direction, threshold)
        if sign == 0:
            return 0
        # [P274] EQUITY-SCALED sizing (operator-instructed: deploy the idle
        # capital). When a target fraction is configured for the asset, the
        # size is floor(fraction x sleeve equity / 1-contract notional) —
        # so a deposit deploys itself under the standing caps with no
        # config change, instead of a fixed contract count leaving new
        # capital idle (the exact under-utilization P273 fixed at ONE
        # equity level). At the 2026-08-16 equity ($3,775) the live
        # fractions (0.15 each) reproduce the P273 book {BTC:1, ETH:3,
        # SOL:1} EXACTLY — behavior-identical until capital moves.
        # Fail-safe: unpriced contract or unreadable equity falls back to
        # the FIXED size — sizing must never be fabricated from missing
        # data (P2/P208), and the fixed path is the known-safe book.
        # Sub-1-contract budgets floor to 1 (the historical minimum);
        # can_trade's fraction caps still gate the order independently.
        si = self._sizing_inputs(asset)
        if si is not None:
            fr, eq, one_ct = si
            sized = max(1, int(fr * eq / one_ct))
            # [P287] Boundary hysteresis — see _dampen_boundary_step. The
            # damping lives HERE (not in _sized_contracts) so every
            # target_for caller — the driver AND main.py's three
            # driver-adjacent reads — sees the same dampened book (P2), while
            # can_trade's cap keeps the raw sized count (cap >= target holds
            # in every branch of the damping truth table).
            try:
                _cur = float(self.signed_contracts(asset) or 0.0)
            except Exception:  # noqa: silent-swallow — an unreadable current position just skips damping; the raw sized target is the pre-P287 behavior
                _cur = 0.0
            sized = self._apply_conviction(sized, conviction)
            if sized == 0:
                return 0
            return self._dampen_boundary_step(_cur, sign * sized, fr, eq,
                                              one_ct)
        fixed = int(getattr(self, "_max_contracts_by_asset", {}).get(
            asset, 1) or 1)
        fixed = self._apply_conviction(max(1, fixed), conviction)
        return sign * fixed

    @staticmethod
    def _apply_conviction(size: int, conviction: float) -> int:
        """[P293d option C] Scale a POSITIVE contract count by conviction.

        Fail direction is deliberate and one-way: anything unusable
        (non-numeric, NaN, inf, negative) returns the size UNCHANGED rather
        than 0, because a fabricated 0 reads as "every advisor said flat"
        — a very different claim from "the ratio was unreadable" (P2).
        Values above 1.0 are clamped: this channel expresses doubt, not
        leverage.
        """
        try:
            c = float(conviction)
        except (TypeError, ValueError):  # noqa: silent-swallow
            return int(size)
        if c != c or c in (float("inf"), float("-inf")) or c < 0.0:
            return int(size)
        c = min(1.0, c)
        return max(0, min(int(size), int(round(int(size) * c))))

    @staticmethod
    def _dampen_boundary_step(cur: float, proposed: int, fr: float,
                              eq: float, one_ct: float) -> int:
        """[P287] Hysteresis at the floor() contract boundary.

        floor(fraction x equity / notional) is discontinuous while equity
        (incl. uPnL) and the contract price move every tick — equity hovering
        at a boundary emits a same-direction 1-contract add/reduce round trip
        each 4H tick, uncontrolled churn of the P142/P198 family that
        flip-persistence deliberately never defers (it only defers FLIPS).

        Rule: a same-direction +1 ADD must also hold under equity x 0.97 (a
        3% buffer) before it is emitted, else the current size is held — so
        re-adding after a boundary reduce needs a real >3% equity move, not
        noise. Everything else passes untouched, each for a doctrinal reason:
        entries from flat and flattens are signal changes, not sizing noise;
        flips belong to flip-persistence; REDUCES are de-risking and must
        always be free (P195 — damping a reduce would hold a LARGER position
        than target); and a >=2-contract change is a real equity move by
        construction (a 1-ct boundary wobble cannot produce it).
        Pure function; truth-tabled in tests/test_p287_exchange.py.
        """
        cur_i = int(round(float(cur or 0.0)))
        p = int(proposed)
        if p == 0 or cur_i == 0:
            return p
        if (p > 0) != (cur_i > 0):
            return p
        if abs(p) - abs(cur_i) != 1:
            return p
        if max(1, int(fr * eq * 0.97 / one_ct)) >= abs(p):
            return p
        return cur_i

    def _sizing_inputs(self, asset: str):
        """[P287] The (fraction, equity, one-contract-notional) triple behind
        equity-scaled sizing, or None when sizing must fall back to the FIXED
        book: no fraction configured, price/equity unreadable, or — new —
        equity STALE (>8h since the last true read). Sizing real orders off a
        frozen equity is fabrication-from-missing-data one step removed
        (P2/P208); the fixed book is the smaller, known-safe fallback (P274's
        own fail direction). Single source for both _sized_contracts and
        target_for's damping so the cap and the target cannot drift (P172)."""
        fr = float(getattr(self, "_target_fraction_by_asset", {}).get(
            asset, 0.0) or 0.0)
        if fr <= 0:
            return None
        try:
            eq = float(self.sleeve_equity_usd() or 0.0)
            if self._equity_is_stale():
                warned = getattr(self, "_sizing_stale_warned", None)
                if warned is None:
                    warned = set()
                    self._sizing_stale_warned = warned
                if asset not in warned:
                    warned.add(asset)
                    _age = self.sleeve_equity_age_sec() or 0.0
                    logger.warning(
                        f"[COINBASE_SLEEVE] {asset}: equity STALE "
                        f"({_age/3600.0:.1f}h) — equity-scaled sizing falls "
                        f"back to the FIXED contract book (the smaller "
                        f"known-safe size, P287)")
                return None
            getattr(self, "_sizing_stale_warned", set()).discard(asset)
            one_ct = abs(self._notional_usd(asset, 1))
            if eq > 0 and one_ct > 0:
                return (fr, eq, one_ct)
        except Exception:  # noqa: silent-swallow — pricing helpers log their own failures; None = fall back to the fixed known-safe size
            pass
        return None

    def _sized_contracts(self, asset: str):
        """[P274] Equity-scaled contract count for `asset`, or None when no
        fraction is configured / the price or equity is unreadable or stale
        (callers fall back to the FIXED size — sizing must never be
        fabricated from missing data, P2/P208). This one function is BOTH the
        target and the per-asset contract cap (cap == target, the P273
        design, now at any equity) — two implementations would drift (P172).
        [P287] Inputs come from _sizing_inputs (shared with target_for's
        boundary damping); this returns the RAW (undampened) count, which is
        what the cap must be: raw >= dampened target in every damping branch."""
        si = self._sizing_inputs(asset)
        if si is None:
            return None
        fr, eq, one_ct = si
        return max(1, int(fr * eq / one_ct))

    async def manage_to_signal(self, asset: str, direction: float,
                               threshold: float = 0.15,
                               conviction: float = 1.0) -> Dict[str, Any]:
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
        # [P273] sized; [P293d] conviction defaults to 1.0 = unchanged
        target = self.target_for(asset, direction, threshold, conviction)
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
        _res = await self.execute_target(asset, target)
        # [P382] Report the target this call actually DROVE TO (post-sizing,
        # post-conviction, post-boundary-damping) so the caller's stop
        # reconcile can use the same number. The driver used to recompute
        # `target_for(asset, dir)` WITHOUT conviction — a raw 6 against a
        # dampened 4 — which read as a size mismatch and armed a follow-up
        # for a fill lag that did not exist.
        if isinstance(_res, dict) and "target" not in _res:
            _res["target"] = target
        return _res

    async def _cancel_resting_orders(self, pid: str, asset: str) -> Optional[int]:
        """[P195] Cancel our own resting orders for `pid` before placing a new one.

        execute_target places a marketable GTC LIMIT, and nothing ever cancelled
        it. `cancel_order`/`fetch_open_orders` existed on the adapter with zero
        callers anywhere in main.py, core/, exchange/ or scripts/. So an unfilled
        limit rested indefinitely and could fill AFTER the engine died — making
        the sleeve a risk-ADDER on process death rather than merely unprotected.
        It also let orders stack across ticks.

        Never raises into the tick. [P287] Returns the number cancelled, or
        **None when the book of resting orders could not be VERIFIED clear**
        (listing failed, or any cancel was unconfirmed) — the caller must then
        REFUSE to place the new order this tick. The old contract ("warn and
        proceed") fail-OPENED the P265 double-order class: an order we cannot
        see, or could not kill, plus a new one is two live orders for one
        delta. The next tick's reconcile + sweep is the retry path.
        """
        cancelled = 0
        failed = 0
        try:
            open_orders = await self._adapter.fetch_open_orders(pid)
        except Exception as e:
            # [P366] both refusal paths feed the streak: an unreadable book
            # and an unconfirmed cancel both mean "this asset cannot trade".
            self._note_cancel_refusal(
                asset, f"could not list resting orders "
                       f"({type(e).__name__}: {e})")
            logger.warning(
                f"[COINBASE_SLEEVE] {asset}: could not list resting orders "
                f"({type(e).__name__}: {e}) — REFUSING to place a new order "
                f"beside a book we cannot see (P287; 'could not read the "
                f"book' must never read as 'book is clean')")
            return None
        for o in open_orders or []:
            oid = _g(o, "order_id") or _g(o, "id")
            if not oid:
                continue
            try:
                if await self._adapter.cancel_order(str(oid), pid):
                    cancelled += 1
                else:
                    failed += 1
                    logger.warning(f"[COINBASE_SLEEVE] {asset}: cancel {oid} "
                                   f"unconfirmed")
            except Exception as e:
                failed += 1
                logger.warning(f"[COINBASE_SLEEVE] {asset}: cancel {oid} failed "
                               f"({type(e).__name__}: {e})")
        if failed:
            self._note_cancel_refusal(
                asset, f"{failed} resting-order cancel(s) unconfirmed")
            logger.warning(
                f"[COINBASE_SLEEVE] {asset}: {failed} resting-order cancel(s) "
                f"unconfirmed — refusing to place a new order this tick "
                f"(double-order class, P265/P287); next tick retries")
            return None
        # [P366] the book is verified clear — this asset can trade again.
        self._note_cancel_ok(asset)
        if cancelled:
            logger.info(f"[COINBASE_SLEEVE] {asset}: cancelled {cancelled} stale "
                        f"resting order(s) before new target")
        return cancelled

    # [P366] How many CONSECUTIVE refusals before this stops being a blip.
    # 6 matches P329's sustained bar for the sibling watchdog escalation; at
    # the observed ~35-45s retry cadence that is ~4 minutes.
    _CANCEL_REFUSE_SUSTAINED = 6

    def _note_cancel_refusal(self, asset: str, why: str) -> None:
        """[P366] Count consecutive 'cannot verify the book is clear' refusals.

        The refusal itself is correct and stays (P265/P287: an order we cannot
        see plus a new one is two live orders for one delta). What was missing
        is that a routed asset can be refused indefinitely and the operator
        sees only a repeating WARNING — 28 consecutive refusals on one ETH
        order during the 2026-08-21 venue outage rendered identically to a
        single blip, and WARNING is not forwarded to Discord.

        P329 built exactly this warn-isolated / escalate-sustained split for
        FastRiskTick and it was never applied to this sibling refusal path
        (P171/P226/P323/P355/P356: a mitigation applied to one instance of a
        class is not applied to the class).

        Escalates ONCE per streak, never per tick — an ERROR every 35s is the
        wallpaper P202 is about, and it would bury the one line that matters.
        """
        try:
            streaks = getattr(self, "_cancel_refuse_streak", None)
            if streaks is None:
                streaks = {}
                self._cancel_refuse_streak = streaks
            n = int(streaks.get(asset, 0)) + 1
            streaks[asset] = n
            if n == self._CANCEL_REFUSE_SUSTAINED:
                logger.error(
                    f"[COINBASE_SLEEVE] {asset}: SUSTAINED — {n} consecutive "
                    f"refusals to place an order because the resting-order "
                    f"book could not be verified clear ({why}). This asset "
                    f"has been unable to open, flip or flatten for the whole "
                    f"streak, INCLUDING emergency exits from FastRiskTick; "
                    f"the venue-resting protective stop is the only "
                    f"protection until it clears. Usually a venue outage and "
                    f"self-resolving — if it persists, check for a stuck "
                    f"resting order at the venue.")
        except Exception as e:  # noqa: silent-swallow — logged; an alert counter must never break the order path
            logger.debug("[COINBASE_SLEEVE] cancel-refusal counter skipped "
                         "(%s: %s)", type(e).__name__, e)

    def _note_cancel_ok(self, asset: str) -> None:
        """[P366] The book was verified clear — clear any refusal streak.

        Without the reset a long-lived process accumulates isolated blips into
        a permanent 'SUSTAINED' that no longer describes the present, which is
        the P303/P265f lesson P329 recorded for the watchdog streak.
        """
        try:
            streaks = getattr(self, "_cancel_refuse_streak", None)
            if not streaks:
                return
            n = int(streaks.pop(asset, 0) or 0)
            if n >= self._CANCEL_REFUSE_SUSTAINED:
                logger.warning(
                    f"[COINBASE_SLEEVE] {asset}: resting-order book verified "
                    f"clear again after {n} consecutive refusals — the asset "
                    f"can trade; the streak is reset.")
        except Exception as e:  # noqa: silent-swallow — logged; see _note_cancel_refusal
            logger.debug("[COINBASE_SLEEVE] cancel-refusal reset skipped "
                         "(%s: %s)", type(e).__name__, e)

    async def _cancel_stale_entry_orders(self, pid: str, asset: str) -> Optional[int]:
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
        Fail-soft; a cancel is always de-risking. Returns the number
        cancelled, or None when the book could not be listed at all ([P287]:
        "could not read" must stay distinguishable from "clean"; the sweep's
        callers place nothing either way, so None only carries honesty here).
        """
        cancelled = 0
        try:
            open_orders = await self._adapter.fetch_open_orders(pid)
        except Exception as e:
            logger.warning(f"[COINBASE_SLEEVE] {asset}: could not list resting "
                           f"orders on hold tick ({type(e).__name__}: {e}) — "
                           f"sweep result UNKNOWN, not clean (P287)")
            return None
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

        [P287] The mark fallback itself needed the same hysteresis: the P265
        price-match tolerance is 0.5%, and a LIVE mark drifts past that in
        ordinary hours — so a mark-anchored stop failed the match every tick,
        cancel+replaced forever, each replacement a brief unprotected window
        through the cancel-confirmation path. The mark anchor is now STORED
        per asset and reused while the mark stays within 2% of it; only a
        real >2% move (or a side change) re-anchors. Entry-anchored behavior
        is untouched. The anchor BASIS is recorded so the PLACED log stops
        claiming "from entry" for a mark-anchored stop.
        """
        pos = self._last_positions.get(asset) or {}
        cur = float(pos.get("signed_contracts") or 0.0)
        # getattr-defended lazy stores: this runs on the live order path and
        # test fixtures build sleeves via object.__new__ (P85).
        anchors = getattr(self, "_stop_mark_anchor", None)
        if anchors is None:
            anchors = {}
            self._stop_mark_anchor = anchors
        bases = getattr(self, "_stop_anchor_basis", None)
        if bases is None:
            bases = {}
            self._stop_anchor_basis = bases
        if cur == 0:
            anchors.pop(asset, None)
            bases.pop(asset, None)
            return None
        anchor = pos.get("entry_vwap")
        basis = "entry"
        if not anchor:
            basis = "mark"
            try:
                pid = self._adapter.to_venue_symbol(asset, "perp")
                prod = self._adapter._client.get_product(product_id=pid)
                mark = _f(_g(prod, "mid_market_price") or _g(prod, "price"))
            except Exception:
                return None
            if not mark:
                return None
            side_sign = 1 if cur > 0 else -1
            prev = anchors.get(asset)
            if (prev and prev[1] == side_sign and prev[0] > 0
                    and abs(mark - prev[0]) / prev[0] <= 0.02):
                anchor = prev[0]  # [P287] reuse: <2% drift is not a re-anchor
            else:
                anchor = mark
                anchors[asset] = (float(mark), side_sign)
                logger.info(
                    f"[COINBASE_STOP] {asset}: stop anchored to MARK "
                    f"{float(mark):.4f} (no entry_vwap from the venue; "
                    f"re-anchors only on a >2% mark move, P287)")
        else:
            anchors.pop(asset, None)  # entry is back: mark anchor obsolete
        bases[asset] = basis
        if not anchor:
            return None
        pct = self._protective_stop_pct
        return float(anchor) * ((1.0 - pct) if cur > 0 else (1.0 + pct))

    # [P382] The stop FOLLOW-UP: the protective stop is reconciled on the 4H
    # tick, but the venue's position read lags a fill by up to a few seconds
    # (observed: every taker-cross leg on 2026-08-22 read back the PRE-trade
    # size). Whenever ensure_protective_stop sees intent and snapshot
    # disagree (entry from flat, same-sign resize, flip, flatten) it records
    # the intent here, and the 30s loop calls followup_protective_stop until
    # the venue agrees with the intent (then the stop is sized from truth)
    # or a bounded wait expires (then the stop is sized from the snapshot
    # and the disagreement is logged). Closes the up-to-4h windows P207/P265
    # left: an oversized stop on a reduced position, an unprotected entry.
    STOP_FOLLOWUP_MAX_SEC = 600.0
    # annotation only — instances built via object.__new__ (tests, operator
    # scripts) may lack it, so every reader goes through getattr (P85)
    _stop_followup: Dict[str, Tuple[float, float]]

    def _request_stop_followup(self, asset: str, intended: float) -> None:
        m: Optional[Dict[str, Tuple[float, float]]] = getattr(self, "_stop_followup", None)
        if m is None:
            m = {}
            self._stop_followup = m
        # keep the FIRST request's clock; re-stamp only the intent
        _prev = m.get(asset)
        m[asset] = (float(intended),
                    _prev[1] if _prev else time.time())

    def request_stop_followup(self, asset: str, intended: float) -> None:
        """Public entry for callers that just changed a position (the 30s
        watchdog's exit/reduce): ask the next loop pass to re-check the stop
        against the venue rather than waiting for the 4H tick."""
        try:
            self._request_stop_followup(asset, intended)
        except Exception:  # noqa: silent-swallow — bookkeeping for a re-check; must never raise into an order path
            pass

    def stop_followup_pending(self) -> Dict[str, float]:
        m: Dict[str, Tuple[float, float]] = getattr(self, "_stop_followup", None) or {}
        return {a: v[0] for a, v in m.items()}

    async def followup_protective_stop(self, asset: str) -> Optional[Dict[str, Any]]:
        """Re-reconcile a pending stop. Returns None when nothing is pending
        for `asset`, else the ensure_protective_stop result (or a PENDING /
        SKIPPED_STALE marker). Never raises."""
        m: Dict[str, Tuple[float, float]] = getattr(self, "_stop_followup", None) or {}
        if asset not in m:
            return None
        intended, since = m[asset]
        try:
            if not self._stop_enabled_for(asset):
                m.pop(asset, None)
                return {"status": "DISABLED", "asset": asset}
            self.reconcile_positions()
            if not getattr(self, "_reconcile_ok", False):
                return {"status": "SKIPPED_STALE", "asset": asset}
            cur = self.signed_contracts(asset)
            agrees = ((abs(cur) <= 1e-9 and abs(intended) <= 1e-9) or
                      (abs(cur) > 1e-9 and abs(intended) > 1e-9
                       and (cur > 0) == (intended > 0)
                       and abs(abs(cur) - abs(intended)) < 1e-9))
            waited = time.time() - since
            if agrees:
                res = await self.ensure_protective_stop(
                    asset, intended_target=intended)
                if res.get("status") in ("OK_EXISTS", "PLACED", "FLAT_NONE",
                                         "FLAT_CANCELLED", "DISABLED"):
                    m.pop(asset, None)
                    logger.info(f"[COINBASE_STOP] {asset}: follow-up settled "
                                f"({res.get('status')}) {waited:.0f}s after "
                                f"the order (P382)")
                return res
            if waited > self.STOP_FOLLOWUP_MAX_SEC:
                # the venue never showed the intended position — size the stop
                # to what the venue DOES show and stop waiting; say so, because
                # intent and reality disagreeing for this long is a finding
                m.pop(asset, None)
                logger.warning(
                    f"[COINBASE_STOP] {asset}: follow-up gave up after "
                    f"{waited:.0f}s — venue shows {cur:+.0f}ct vs intended "
                    f"{intended:+.0f}ct; sizing the stop to the VENUE (P382)")
                return await self.ensure_protective_stop(asset)
            return {"status": "PENDING", "asset": asset, "snapshot": cur,
                    "intended": intended, "waited_sec": waited}
        except Exception as e:
            logger.warning(f"[COINBASE_STOP] {asset}: follow-up failed "
                           f"({type(e).__name__}: {e}); retrying next pass")
            return {"status": "ERROR", "asset": asset, "reason": str(e)}

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
                if abs(cur) > 1e-9:
                    # [P382] the flatten may STRAND; re-check within the
                    # 30s loop rather than at the next 4H tick
                    self._request_stop_followup(asset, 0.0)
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
                self._request_stop_followup(asset, float(intended_target))
                return {"status": "FLIP_IN_TRANSITION", "asset": asset,
                        "snapshot": cur, "intended": float(intended_target)}
            # [P382] THE THIRD CASE, observed live 2026-08-22 and the one the
            # two carve-outs above cannot see: a nonzero intent with the SAME
            # sign as a nonzero snapshot but a DIFFERENT size. The venue's
            # position read lags the fill on the taker-cross leg (4 of 4 cross
            # legs that day logged `now=` equal to the PRE-trade size), so a
            # reduce 5->3 ran this method against a snapshot of 5 and placed a
            # 5ct SELL stop on a 3ct long; a touch would have closed the long
            # and OPENED a 2ct short (no reduce_only on CDE, no can_trade gate
            # on a venue-triggered fill). Venue-verified the same afternoon:
            # SOL 1ct long guarded by a 2ct stop. An oversized stop opens the
            # opposite side; an undersized one under-protects until the
            # follow-up below re-sizes it seconds later. So size by the SMALLER
            # of snapshot and intent (never the larger), keep the snapshot's
            # sign, and ask the 30s loop to re-reconcile.
            elif (intended_target is not None
                    and abs(cur) > 1e-9
                    and abs(float(intended_target)) > 1e-9
                    and abs(abs(float(intended_target)) - abs(cur)) >= 1e-9):
                _sized = math.copysign(
                    min(abs(cur), abs(float(intended_target))), cur)
                logger.info(
                    f"[COINBASE_STOP] {asset}: snapshot {cur:+.0f}ct vs "
                    f"intended {float(intended_target):+.0f}ct — sizing the "
                    f"stop to the SMALLER ({_sized:+.0f}ct; an oversized stop "
                    f"opens the opposite side, P382) and re-checking within "
                    f"the 30s loop")
                self._request_stop_followup(asset, float(intended_target))
                cur = _sized
            # [P382] ENTRY FROM FLAT whose fill the snapshot has not seen yet:
            # the old path fell into the FLAT branch below, returned an
            # unlogged FLAT_NONE, and left the new position with NO venue stop
            # until the next 4H tick. One fresh reconcile first (the cross leg
            # places then reconciles immediately, so the cached snapshot is
            # the stale one); if the venue still shows flat, place NOTHING
            # (a stop sized to the intent is the P207 orphan if the entry
            # strands) and let the follow-up re-check within ~34s.
            elif (intended_target is not None
                    and abs(float(intended_target)) > 1e-9
                    and abs(cur) <= 1e-9):
                try:
                    self.reconcile_positions()
                    if getattr(self, "_reconcile_ok", False):
                        cur = self.signed_contracts(asset)
                except Exception:  # noqa: silent-swallow — a failed re-read keeps the cached snapshot; the follow-up below retries
                    pass
                if abs(cur) <= 1e-9:
                    self._request_stop_followup(asset, float(intended_target))
                    logger.warning(
                        f"[COINBASE_STOP] {asset}: entry in transition "
                        f"(intended {float(intended_target):+.0f}ct, venue "
                        f"still shows flat) — placing NO stop this pass; the "
                        f"30s loop re-checks and places it once the fill is "
                        f"visible (P382)")
                    return {"status": "ENTRY_IN_TRANSITION", "asset": asset,
                            "intended": float(intended_target)}
                elif (float(intended_target) > 0) != (cur > 0):
                    # the fresh read shows the OPPOSITE side — that is the
                    # flip window above, by a different route
                    self._request_stop_followup(asset, float(intended_target))
                    return {"status": "FLIP_IN_TRANSITION", "asset": asset,
                            "snapshot": cur,
                            "intended": float(intended_target)}
                elif abs(abs(float(intended_target)) - abs(cur)) >= 1e-9:
                    self._request_stop_followup(asset, float(intended_target))
                    cur = math.copysign(
                        min(abs(cur), abs(float(intended_target))), cur)
            # [P287] Listing failure is handled HERE, explicitly, not by the
            # generic ERROR catch below: with the adapter's old swallow this
            # read `resting=[]` on a venue error and fell into clear-and-place
            # — a SECOND stop beside the invisible real one (two plain stops
            # on a no-reduce_only venue: the first closes the position, the
            # second OPENS the opposite side when touched). Refuse for the
            # tick, keep the last resting state, retry next tick.
            try:
                _orders = await self._adapter.fetch_open_orders(pid) or []
            except Exception as e:
                logger.warning(
                    f"[COINBASE_STOP] {asset}: cannot list resting orders "
                    f"({type(e).__name__}: {e}) — refusing to reconcile the "
                    f"stop this tick (clear-and-place on an unreadable book "
                    f"is the double-stop door, P287)")
                return {"status": "ORDERS_UNREADABLE", "asset": asset}
            resting = [o for o in _orders if self._is_stop_order(o)]

            # FLAT: any surviving stop is an ORPHAN that could open a position.
            # Cancelling it is the single most important thing in this method.
            if cur == 0:
                # [P287] a flat asset's stored mark anchor is obsolete (the
                # early-return above skips desired_stop_price's own cleanup).
                getattr(self, "_stop_mark_anchor", {}).pop(asset, None)
                n = 0
                _flat_failed = []
                for o in resting:
                    oid = _g(o, "order_id") or _g(o, "id")
                    if not oid:
                        continue
                    if await self._adapter.cancel_order(str(oid), pid):
                        n += 1
                    else:
                        _flat_failed.append(str(oid))
                if _flat_failed:
                    # [P287] The old loop counted successes only: a failed
                    # cancel produced no log and returned FLAT_NONE —
                    # byte-identical to "no orphans existed", while a plain
                    # stop that OPENS a position when touched stayed live.
                    logger.warning(
                        f"[COINBASE_STOP] {asset}: {len(_flat_failed)} "
                        f"orphan-stop cancel(s) FAILED "
                        f"({', '.join(_flat_failed)}) — a live stop on a "
                        f"flat asset OPENS a position when touched (no "
                        f"reduce_only on CDE); retrying next tick (P287)")
                    return {"status": "FLAT_CANCEL_FAILED", "asset": asset,
                            "cancelled": n,
                            "failed_cancels": len(_flat_failed)}
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
                # [P287] say which anchor actually priced this stop — the old
                # literal "from entry" was a lie for a mark-anchored fallback
                _basis = getattr(self, "_stop_anchor_basis", {}).get(
                    asset, "entry")
                logger.info(f"[COINBASE_STOP] {asset}: placed protective "
                            f"{want_side} stop @ {stop_px:.4f} for {cur:+.0f}ct "
                            f"({self._protective_stop_pct:.1%} from {_basis})")
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

    async def _maker_reprice_step(self, pid: str, asset: str, side: str,
                                  base_size: float, oid: str, px: float,
                                  best_bid: float, best_ask: float,
                                  intended_contracts: Optional[int] = None,
                                  ) -> Dict[str, Any]:
        """[P291] The ONE reprice decision, taken at the half-way mark of an
        unfilled maker window. Never raises. Returns either a state update the
        caller applies and keeps polling, or a verdict it returns verbatim:

          {"used": False}                 — the book could not be read; the
                                            opportunity is NOT consumed (a
                                            read failure is not information),
                                            the original post keeps working
          {"used": True}                  — still at the front of the book:
                                            NO reprice (a reprice that fires
                                            when the touch has not moved is
                                            pure churn — two venue calls
                                            buying nothing)
          {"used": True, "verdict": ...}  — terminal; see below
          {"used": True, "oid": ..., ...} — repriced; poll the new order

        Verdicts: CANCEL_FAILED (cancel unconfirmed — the order may be live,
        so refuse BOTH the re-post and the cross, P287/P265); DONE (the
        cancelled order's fill state is unreadable or shows a PARTIAL — hand
        back so the caller's venue reconcile sizes the remainder, anti-P139;
        filled_size is deliberately never arithmetic'd here, P219);
        REPRICE_UNKNOWN (the re-post raised — an order MAY rest, so the cross
        is refused, same class as CANCEL_FAILED); FALLBACK (re-post rejected
        with the cancel CONFIRMED, so nothing rests and the caller crosses,
        exactly as it would have at the window's end); DONE_UNTRACKED
        (re-post accepted with no order_id — caller sweeps first, P287).

        The move test is "are we still at the FRONT of the queue": our resting
        BUY *is* the best bid unless someone outbid us, so a new best bid
        ABOVE our price means we are behind the queue and will not fill.
        """
        try:
            bb = self._adapter._client.get_best_bid_ask(product_ids=[pid])
            books = _g(bb, "pricebooks") or []
            book = books[0] if books else None
            bids = _g(book, "bids") or []
            asks = _g(book, "asks") or []
            new_bid = _f(_g(bids[0], "price")) if bids else 0.0
            new_ask = _f(_g(asks[0], "price")) if asks else 0.0
        except Exception as e:
            self._maker_log_once(asset, f"reprice book read failed "
                                 f"({type(e).__name__}) — holding the post")
            return {"used": False}
        new_px = new_bid if side == "BUY" else new_ask
        if not (new_px and math.isfinite(new_px) and new_px > 0):
            return {"used": False}
        moved = (new_px > px) if side == "BUY" else (new_px < px)
        if not moved:
            logger.info(f"[COINBASE-MAKER] {asset}: still at the front of the "
                        f"book at {px} — no reprice")
            return {"used": True}
        # One at a time: the cancel must be CONFIRMED before anything new is
        # placed, or a re-post beside a live post-only doubles the order.
        if not await self._adapter.cancel_order(oid, pid):
            logger.warning(f"[COINBASE-MAKER] {asset}: reprice cancel FAILED "
                           f"— order {oid} may be live; refusing both the "
                           f"re-post and the cross (P265 double-order class). "
                           f"Next tick's stale-entry sweep retries.")
            return {"used": True, "verdict": "CANCEL_FAILED"}
        _op = None
        try:
            _op = await self._adapter.get_order(oid)
        except Exception:  # noqa: silent-swallow — unknown fill state is handled as the DONE hand-back below, which is logged
            _op = None
        if _op is None:
            # [P2] absence is not zero: we cannot certify the cancelled order
            # was unfilled, and re-posting full size beside an unknown partial
            # would overshoot the target.
            logger.info(f"[COINBASE-MAKER] {asset}: reprice cancelled {oid} "
                        f"but its fill state is unreadable — handing back so "
                        f"the caller reconciles and crosses the remainder")
            return {"used": True, "verdict": "DONE"}
        try:
            _filled = float((_op or {}).get("filled_size") or 0)
        except (TypeError, ValueError, AttributeError):
            logger.info(f"[COINBASE-MAKER] {asset}: reprice could not parse "
                        f"the cancelled order's fill state — handing back")
            return {"used": True, "verdict": "DONE"}
        if _filled > 0:
            try:
                self._record_fill_quality(
                    asset=asset, order_id=oid, side=side,
                    contracts=intended_contracts, liquidity="maker",
                    urgent=False,
                    decision_mid=((best_bid + best_ask) / 2.0
                                  if best_bid and best_ask else None),
                    decision_bid=best_bid or None,
                    decision_ask=best_ask or None,
                    order_payload=_op)
            except Exception as e:  # noqa: silent-swallow — logs; observation must never affect the ladder (Iron Law 7)
                logger.warning(f"[FILL-QUALITY] {asset}: reprice partial "
                               f"record failed ({type(e).__name__})")
            logger.info(f"[COINBASE-MAKER] {asset}: reprice found a PARTIAL "
                        f"maker fill — handing back rather than re-posting "
                        f"full size (overshoot class)")
            return {"used": True, "verdict": "DONE"}
        from exchange.adapter import OrderRequest
        req = OrderRequest(symbol=pid, side=side, size=base_size,
                           order_type="LIMIT", price=new_px, post_only=True)
        try:
            res = await self._adapter.place_order(req)
        except Exception as e:
            logger.warning(f"[COINBASE-MAKER] {asset}: reprice placement "
                           f"raised ({type(e).__name__}: {e}) — an order MAY "
                           f"rest; refusing the cross (P265 double-order "
                           f"class). Next tick's sweep retries.")
            return {"used": True, "verdict": "REPRICE_UNKNOWN"}
        if not res.success:
            logger.info(f"[COINBASE-MAKER] {asset}: reprice post-only "
                        f"rejected ({res.error_code}) — taker fallback")
            return {"used": True, "verdict": "FALLBACK"}
        new_oid = str(res.order_id or "")
        if not new_oid:
            logger.warning(f"[COINBASE-MAKER] {asset}: repriced post-only "
                           f"carried no order_id — caller must sweep the book "
                           f"before crossing (P287)")
            return {"used": True, "verdict": "DONE_UNTRACKED"}
        logger.info(f"[COINBASE-MAKER] {asset}: touch moved {px} -> {new_px} "
                    f"(no longer at the front) — repriced once, order "
                    f"{new_oid}")
        return {"used": True, "oid": new_oid, "px": new_px,
                "best_bid": new_bid, "best_ask": new_ask}

    async def _maker_first_attempt(self, pid: str, asset: str, side: str,
                                   base_size: float,
                                   intended_contracts: Optional[int] = None,
                                   ) -> str:
        """[P270] One post-only attempt joining the touch, resolved INSIDE
        this call. Returns:
          "DONE"           — order left the book (filled or gone); caller
                             recomputes the remaining delta from reconcile
          "DONE_UNTRACKED" — [P287] order ACCEPTED but the venue returned no
                             order_id, so it can be neither polled nor
                             cancelled by id; the caller must SWEEP resting
                             orders and verify the book is clear before any
                             cross (a cross beside the untracked resting
                             post-only is the P265 double-order class)
          "FALLBACK"       — no maker fill possible (no bid/ask, placement
                             rejected, e.g. price crossed between quote and
                             place); caller proceeds straight to the cross
          "CANCEL_FAILED"  — timeout AND the cancel failed: the maker order
                             may still be live, so the caller must NOT place
                             the cross (two live orders for one delta = the
                             P265 double-order class). Next tick's stale-entry
                             sweep retries.
          "REPRICE_UNKNOWN" — [P291] the reprice re-post raised, so an order
                             MAY rest and its id is unknown; the caller must
                             NOT cross (same class as CANCEL_FAILED). Only
                             reachable with maker_reprice on.
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
                # [P287] accepted but no id to poll/cancel by. The old "DONE"
                # here contradicted its own comment: the caller's DONE
                # handling reconciles, sees the unfilled delta, and CROSSES —
                # a blind second order beside the untracked resting post-only
                # (which can then fill and overshoot the target). Return a
                # DISTINCT verdict so the caller sweeps + verifies the book
                # is clear of the untracked order before any cross.
                logger.warning(f"[COINBASE-MAKER] {asset}: accepted post-only "
                               f"carried no order_id — caller must sweep the "
                               f"book before crossing (P287)")
                return "DONE_UNTRACKED"
            # [P291] The window is SPLIT, never extended: the deadline below
            # is computed once and never reassigned, so the reprice buys a
            # second queue position out of the SAME budget (a longer wait
            # means a staler signal — P265 dead-signal-fill class).
            deadline = time.monotonic() + self._maker_wait_sec
            # [P85] getattr-defended: this read sits on the LIVE order path
            # and the ctor is not the only way a sleeve exists (operator
            # scripts and every test fixture build one via object.__new__).
            # An undefended read raises AttributeError, which the outer
            # handler swallows into "FALLBACK" — silently crossing at taker
            # for every such instance. Missing attribute = OFF = the P270
            # single-post ladder, the conservative direction.
            _reprice_at = (deadline - self._maker_wait_sec / 2.0
                           if getattr(self, "_maker_reprice", False)
                           else None)
            _reprice_used = False
            while time.monotonic() < deadline:
                await asyncio.sleep(5.0)
                try:
                    open_orders = await self._adapter.fetch_open_orders(pid)
                except Exception:
                    # [P287] transient list failure = order state UNKNOWN,
                    # never "left the book": keep waiting; the timeout+cancel
                    # path below settles it. (This branch was dead code while
                    # the adapter swallowed errors into [] — an outage then
                    # read as "filled at 0bps maker" and the caller crossed
                    # the full remainder beside a still-resting post-only.)
                    continue
                still_open = any(
                    str(_g(o, "order_id") or _g(o, "id") or "") == oid
                    for o in (open_orders or []))
                if not still_open:
                    logger.info(f"[COINBASE-MAKER] {asset}: post-only left "
                                f"the book within the window (filled at "
                                f"0bps maker)")
                    # [P290] "left the book" is resolved to fill-truth by ONE
                    # get_order read; unreadable -> unresolved record, never
                    # a fabricated fill. Fully fail-soft: the ladder's return
                    # value is decided above this and cannot change.
                    try:
                        _op = await self._adapter.get_order(oid)
                        self._record_fill_quality(
                            asset=asset, order_id=oid, side=side,
                            contracts=intended_contracts, liquidity="maker",
                            urgent=False,
                            decision_mid=((best_bid + best_ask) / 2.0
                                          if best_bid and best_ask else None),
                            decision_bid=best_bid or None,
                            decision_ask=best_ask or None,
                            order_payload=_op)
                    except Exception as e:  # noqa: silent-swallow — logs; the observation must never affect the ladder (Iron Law 7)
                        logger.warning(f"[FILL-QUALITY] {asset}: maker-fill "
                                       f"record failed ({type(e).__name__})")
                    return "DONE"
                # [P291] Half-way mark: the post is still resting. If the
                # touch has moved past it we are behind the queue and will
                # not fill — cancel and take a fresh queue position ONCE.
                # Evaluated at most once per call; a book-read failure does
                # not consume the opportunity.
                if (_reprice_at is not None and not _reprice_used
                        and time.monotonic() >= _reprice_at):
                    _rp = await self._maker_reprice_step(
                        pid, asset, side, base_size, oid, px,
                        best_bid, best_ask,
                        intended_contracts=intended_contracts)
                    _reprice_used = bool(_rp.get("used", True))
                    _verdict = _rp.get("verdict")
                    if _verdict:
                        return str(_verdict)
                    if _rp.get("oid"):
                        oid = str(_rp["oid"])
                        px = float(_rp["px"])
                        best_bid = float(_rp.get("best_bid") or 0.0)
                        best_ask = float(_rp.get("best_ask") or 0.0)
            if await self._adapter.cancel_order(oid, pid):
                logger.info(f"[COINBASE-MAKER] {asset}: unfilled after "
                            f"{self._maker_wait_sec:.0f}s — cancelled, "
                            f"crossing the remainder")
                # [P290] A cancelled post-only can still carry a PARTIAL
                # maker fill; record it only when the payload shows one.
                try:
                    _op = await self._adapter.get_order(oid)
                    _pf = 0.0
                    try:
                        _pf = float((_op or {}).get("filled_size") or 0)
                    except (TypeError, ValueError):
                        _pf = 0.0
                    if _pf > 0:
                        self._record_fill_quality(
                            asset=asset, order_id=oid, side=side,
                            contracts=intended_contracts, liquidity="maker",
                            urgent=False,
                            decision_mid=((best_bid + best_ask) / 2.0
                                          if best_bid and best_ask else None),
                            decision_bid=best_bid or None,
                            decision_ask=best_ask or None,
                            order_payload=_op)
                except Exception as e:  # noqa: silent-swallow — logs; same Iron Law 7 rule as the fill branch
                    logger.warning(f"[FILL-QUALITY] {asset}: partial-maker "
                                   f"record failed ({type(e).__name__})")
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

        `urgent=True` (FORCE_FLAT, the FastRiskTick watchdog) skips the maker
        ladder AND [P383] sends the cross as URGENT_ORDER_TYPE (MARKET, IOC)
        instead of the 0.2%-through GTC limit — an emergency exit must never
        be left resting unfilled beside a swept protective stop. Every
        refusal (stale reconcile, unverifiable sweep, unknown contract size,
        can_trade) applies to urgent and non-urgent alike.

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
            # [P287] the sweep now returns None when the book of resting
            # orders could not be verified clear (listing failed / a cancel
            # unconfirmed). Placing a new order beside an order we cannot see
            # or could not kill is the P265 double-order class — refuse this
            # tick; the venue stop keeps protecting and the next tick retries.
            _swept = await self._cancel_resting_orders(pid, asset)
            if _swept is None:
                return {"status": "SKIPPED_STALE", "asset": asset,
                        "reason": "resting_orders_unverifiable"}
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
            # [P383] ...for a LIMIT. An URGENT exit is sent as MARKET (no
            # limit price exists to be ~0), so an unreadable mid must NOT
            # block the emergency exit — reconcile just succeeded, so the
            # venue is up; what failed is one product read. The mid is still
            # read here as the decision reference for the fill-quality
            # ledger; when unusable the ledger records it ABSENT (P2), never
            # a fabricated 0.0.
            _mid_ok = bool(mid and math.isfinite(mid) and mid > 0)
            if not _mid_ok and not urgent:
                logger.error(f"[COINBASE_SLEEVE] execute_target {asset}: no "
                             f"usable price from venue (mid={mid!r}) — "
                             f"refusing to place a priceless order")
                return {"status": "ERROR", "asset": asset,
                        "reason": f"no_price:mid={mid!r}"}
            _fq_mid: Optional[float] = mid if _mid_ok else None
            if not _mid_ok:
                logger.warning(f"[COINBASE_SLEEVE] execute_target {asset}: "
                               f"URGENT exit with no usable decision mid "
                               f"(mid={mid!r}) — MARKET order proceeds; the "
                               f"fill-quality row records mid=None")
            # [P290] ONE decision-time book read for the fill-quality ledger
            # (spread context). Fail-soft to None fields — never a fabricated
            # book, and never a second attempt: this is observation, not a
            # guard.
            _fq_bid = _fq_ask = None
            try:
                _fq_bk = self._adapter._client.get_product_book(
                    product_id=pid, limit=1)
                _fq_pb = _g(_fq_bk, "pricebook") or _fq_bk
                _fq_bids = _g(_fq_pb, "bids") or []
                _fq_asks = _g(_fq_pb, "asks") or []
                if _fq_bids and _fq_asks:
                    _b = _f(_g(_fq_bids[0], "price"))
                    _a = _f(_g(_fq_asks[0], "price"))
                    if _b > 0 and _a >= _b:
                        _fq_bid, _fq_ask = _b, _a
            except Exception:  # noqa: silent-swallow — spread context is optional; the ledger records None (absence stays absent, P2)
                pass
            # [P270] Maker-first ladder: post-only at the touch (0bps) ->
            # poll inside this call -> cancel -> cross whatever remains at
            # taker (3bps). Skipped when urgent (FORCE_FLAT and the fast-risk
            # watchdog need immediacy, not fee savings) or for non-LIMIT
            # callers. Resolved remainder from a fresh reconcile so a partial
            # maker fill is crossed exactly once, never double-ordered.
            maker_note = None
            _maker_ran = bool(self._maker_first and not urgent
                              and order_type == "LIMIT")
            if _maker_ran:
                _mk = await self._maker_first_attempt(
                    pid, asset, side, base_size,
                    intended_contracts=n_contracts)
                if _mk == "CANCEL_FAILED":
                    return {"status": "FAILED", "asset": asset, "side": side,
                            "reason": "maker_cancel_failed_no_cross"}
                if _mk == "REPRICE_UNKNOWN":
                    # [P291] The reprice placement raised: an order may rest
                    # under an id we never learned. Same refusal as an
                    # unconfirmed cancel — crossing beside it is the P265
                    # double-order class; the next tick's sweep resolves it.
                    return {"status": "FAILED", "asset": asset, "side": side,
                            "reason": "maker_reprice_unknown_no_cross"}
                if _mk == "DONE_UNTRACKED":
                    # [P287] The accepted post-only has no id: before any
                    # cross, sweep the book and VERIFY no non-stop order
                    # still rests. If the sweep cannot certify a clear book,
                    # refuse the cross outright (same class as
                    # maker_cancel_failed_no_cross) — crossing beside the
                    # untracked resting post-only overshoots the target the
                    # moment it fills.
                    _usw = await self._cancel_resting_orders(pid, asset)
                    _residual = None
                    if _usw is not None:
                        try:
                            _residual = [
                                o for o in
                                (await self._adapter.fetch_open_orders(pid)
                                 or [])
                                if not self._is_stop_order(o)]
                        except Exception:  # noqa: silent-swallow — verification failure = refuse below; the refusal itself is logged
                            _residual = None
                    if _usw is None or _residual is None or _residual:
                        logger.warning(
                            f"[COINBASE-MAKER] {asset}: could not verify the "
                            f"untracked post-only is gone — refusing the "
                            f"cross (P287; double-order class)")
                        return {"status": "FAILED", "asset": asset,
                                "side": side,
                                "reason": "maker_untracked_no_cross"}
                    _mk = "DONE"
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
                    _fq_mid = mid
                # _mk == "FALLBACK": fall through to the marketable cross
            if urgent:
                # [P383] URGENT -> MARKET (IOC at the venue): fills or dies,
                # never rests. See URGENT_ORDER_TYPE for why a 0.2%-through
                # GTC limit is the wrong order for an emergency exit. No
                # price is sent — the adapter's MARKET branch takes base_size
                # only (contracts = size / contract_size, P195).
                req = OrderRequest(symbol=pid, side=side, size=base_size,
                                   order_type=self.URGENT_ORDER_TYPE,
                                   price=None, post_only=False)
            else:
                # non-urgent: byte-identical to pre-P383 — marketable GTC
                # limit 0.2% through the mid (adapter rounds to tick)
                px = mid * (1.002 if side == "BUY" else 0.998)
                req = OrderRequest(symbol=pid, side=side, size=base_size,
                                   order_type=order_type, price=px,
                                   post_only=False)
            res = await self._adapter.place_order(req)
            self.reconcile_positions()
            # [P290] Realized-fill record for the cross/direct leg: ONE
            # immediate get_order (no polling); an unfilled/unreadable order
            # records status="unresolved" with no fill fields. Fully
            # fail-soft — the return value below is composed from `res` and
            # the reconcile exactly as before.
            if res.success:
                try:
                    _fq_oid = str(res.order_id or "") or None
                    _fq_op = (await self._adapter.get_order(_fq_oid)
                              if _fq_oid else None)
                    # [P383] urgent MARKET legs carry their own liquidity
                    # class so the maker/taker review (P375) can separate
                    # emergency exits from the non-urgent cross.
                    self._record_fill_quality(
                        asset=asset, order_id=_fq_oid, side=side,
                        contracts=n_contracts,
                        liquidity=("market_urgent" if urgent
                                   else "taker_cross" if _maker_ran
                                   else "direct"),
                        urgent=urgent, decision_mid=_fq_mid,
                        decision_bid=_fq_bid, decision_ask=_fq_ask,
                        order_payload=_fq_op)
                except Exception as e:  # noqa: silent-swallow — logs; observation must never affect the order path (Iron Law 7)
                    logger.warning(f"[FILL-QUALITY] {asset}: cross-fill "
                                   f"record failed ({type(e).__name__})")
            logger.info(f"[COINBASE_SLEEVE] execute_target {asset} {side} "
                        f"{n_contracts}ct {req.order_type}"
                        f"{' URGENT' if urgent else ''} -> "
                        f"success={res.success} "
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
            # [P287] freshness of the TRUE-equity read, for consumers whose
            # own guards need it (the P209 fuse feed gates on _reconcile_ok,
            # which certifies POSITIONS, not equity — a partial outage keeps
            # it True while equity freezes; main.py can bound on this age).
            "equity_age_sec": self.sleeve_equity_age_sec(),
        }

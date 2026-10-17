"""
================================================================================
HMATS [P315] - CDE fees are PER CONTRACT, not a percentage of notional
================================================================================

WHY THIS EXISTS

    `core/paper_fee_service.VENUE_FEE_STD` models Coinbase as
    ``{"maker": 0.0000, "taker": 0.0003}`` — 0bps maker / 3bps taker. Measured
    against the venue's own reported fees in `data/fill_quality.jsonl` (the
    ledger P290 built for exactly this purpose):

        asset  liquidity     ct   notional     fee$   fee/ct   REALIZED bps
        BTC    maker          2   $1,282.10    1.202   0.601      9.37
        BTC    taker_cross    2   $1,286.60    1.269   0.635      9.87
        BTC    maker          2   $1,286.10    1.205   0.602      9.37
        BTC    maker          2   $1,288.70    1.207   0.603      9.36
        BTC    maker          2   $1,286.90    1.205   0.603      9.36
        ETH    maker          6   $1,149.90    1.582   0.264     13.76

    The model is wrong by ~3x on the taker leg and by INFINITY on the maker leg
    (it charges zero for something that costs ~9.4bps). The cause is
    structural, which is why no bps constant can fix it: CDE charges a FLAT FEE
    PER CONTRACT (~$0.60 BTC, ~$0.26 ETH), so the cost in bps is
    ``fee_per_contract / (contract_size * price)`` — it moves inversely with
    price and is largest exactly where the contract is smallest. A percentage
    model cannot represent that at any calibration.

WHAT IT COSTS

    Live gate line, 2026-08-19:
        [VENUE-FEE] BTC: friction priced for COINBASE (taker=3.0bps maker=0.0bps)
        Alpha gate: passes=True, alpha=22bps, threshold=18bps, net=8bps -> ALLOW

    P167 charges friction ROUND TRIP (2 legs) with a 1.1x multiplier. Adding
    the ~6.9bps/leg the model omits moves that threshold to ~33bps against
    22bps of asserted alpha: net ~-11bps, i.e. the trade should be REJECTED.
    Every entry has been taken on economics that understate cost ~3x.

FAIL DIRECTION (P167: overcharging costs opportunity, undercharging spends
money — so every unknown resolves to the EXPENSIVE side)

    * An asset with no measured fee falls back to the most expensive measured
      per-contract fee, and says so via `provenance`.
    * An unusable price (<=0, NaN) returns None — the caller keeps its existing
      (Kraken-tier) number rather than being handed a fabricated one. Absence
      is not zero (P2).
    * Where maker was measured but taker was not, taker is scaled up by the
      measured BTC taker/maker ratio, never set equal to maker.

PRE-COMMITTED REVISION RULE (the P290 pattern)

    These constants may be RAISED on any evidence. They may only be LOWERED on
    >= 20 filled legs for that asset in `fill_quality.jsonl` AND a new P-entry
    recording the measurement. A fee constant that drifts down without evidence
    re-creates the exact defect this module exists to correct.
================================================================================
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("HMATS.CDEFees")

# --- Measured per-contract fees, USD. Provenance in the module docstring. ----
# BTC: 5 fills (3 maker @ .601/.602/.603, 1 taker @ .635) 2026-08-18/19.
# ETH: 1 fill  (maker @ .264) 2026-08-18.
# SOL: NO FILL WITH A REPORTED FEE EXISTS. Assumed at the most expensive
#      measured per-contract fee (P167). One measured SOL fill closes this.
_BTC_TAKER_OVER_MAKER = 0.635 / 0.603          # 1.0531, measured

# [P327] MEASURED constant — registered in core.calibration_registry, which
# carries the producer command, the source data, the staleness horizon and the
# revision rule. P315 found this model wrong for the life of the system, and
# nothing recorded when it had last been checked against a fill.
_MEASURED_ON = "2026-08-26"
_MEASURED_BY = ("data/fill_quality.jsonl (fills, P290/P315) + "
                "scripts/coinbase_probe_stop_support.py venue previews "
                "(P334, which refuted the flat-per-contract model); "
                "[P412] XRP added from a read-only venue preview 2026-08-26 "
                "(commission $0.64 on $708.68 notional -> 9.03bps), with a "
                "same-run BTC control reproducing the recorded ~9.67bps quote")


# [P334] CORRECTED MODEL: per-asset BPS OF NOTIONAL, not flat dollars.
#
# P315 concluded the fee is a flat ~$0.60/contract. That rested on 5 BTC fills
# and 1 ETH fill spanning almost NO price range (BTC 64.1k-64.3k), which cannot
# distinguish a flat dollar fee from a percentage. With ~8% of price variation
# now available the flat model FAILS and the percentage model fits:
#
#     asset  source                px        notional   fee$    fee_bps
#     BTC    fill  (maker)         64,105      641.05   0.601      9.38
#     BTC    fill  (taker)         64,330      643.30   0.635      9.87
#     BTC    PREVIEW 2026-08-20    69,280      692.80   0.670      9.67
#     ETH    fill  (maker)          1,916      191.65   0.264     13.78
#     ETH    PREVIEW 2026-08-20     2,252      225.23   0.300     13.32
#     SOL    PREVIEW 2026-08-20        84.44   422.20   0.460     10.90
#     XRP    PREVIEW 2026-08-26         1.417   708.68   0.640      9.03
#     BNB    PREVIEW 2026-08-26       701.600   701.60   0.610      8.69
#
# The discriminator is the only within-asset price move we have: BTC price
# +7.7%, fee +5.5%. A flat fee predicts +0.0%; a percentage predicts +7.7%.
# bps are stable per asset across that move; dollars per contract are not.
CDE_FEE_BPS: Dict[str, Dict[str, float]] = {
    # highest measured value for each asset/side (P167)
    "BTC": {"maker": 9.38, "taker": 9.87},
    "ETH": {"maker": 13.78, "taker": 13.78 * _BTC_TAKER_OVER_MAKER},
    # [P412] XRP: maker MEASURED from a read-only venue preview 2026-08-26
    # ($0.64/$708.68 = 9.03bps; a same-run BTC control reproduced its recorded
    # ~9.67bps quote, so the preview method is validated). taker DERIVED via the
    # measured fill ratio (the ETH pattern — real fills show taker>maker). This
    # PRICES XRP from a preview rather than a fill (provenance measured_venue_
    # preview, below) to break the chicken-and-egg on a newly-activated asset:
    # without a priceable fee the P321b interlock perma-gates XRP (it can never
    # fill, so can never earn a fill-measured fee). SOL keeps its assumed-worst
    # below because SOL trades regardless (edge >> friction); XRP does not.
    "XRP": {"maker": 9.03, "taker": round(9.03 * _BTC_TAKER_OVER_MAKER, 2)},
    # [P412] BNB (2nd breadth, P197): maker MEASURED via preview 2026-08-26
    # ($0.61/$701.60 = 8.69bps). taker derived via the measured fill ratio.
    # The only breadth asset with all 3 eras positive (seat_alpha, P412).
    "BNB": {"maker": 8.69, "taker": round(8.69 * _BTC_TAKER_OVER_MAKER, 2)},
    # SOL still has NO FILL. A venue PREVIEW quotes 10.90bps, which is why the
    # assumption below is now known-conservative rather than arbitrary — but a
    # quote is not a fill, so the value stays at the worst measured figure and
    # the provenance still says ASSUMED (P315's revision rule: LOWER only on
    # >=20 filled legs).
    "SOL": {"maker": 0.0, "taker": 0.0},      # filled in below
}

# Assets whose fee is ASSUMED rather than measured — surfaced, not hidden.
CDE_FEE_ASSUMED = frozenset({"SOL"})

# [P412] Assets priced from a validated venue PREVIEW rather than a fill —
# honest provenance (a preview is the venue fee schedule; the BTC control
# confirms previews match fills; what a preview cannot see is realized
# slippage, which is tracked separately, not part of the fee).
CDE_FEE_PREVIEW = frozenset({"XRP", "BNB"})

# The most expensive measured fee in bps, used for anything unknown.
_WORST_BPS = max(max(v["maker"], v["taker"])
                 for a, v in CDE_FEE_BPS.items() if a not in CDE_FEE_ASSUMED)
for _a in CDE_FEE_ASSUMED:
    CDE_FEE_BPS[_a] = {"maker": _WORST_BPS, "taker": _WORST_BPS}

# The venue's own preview quote for SOL, recorded so the assumption above is
# auditable as conservative rather than merely large. NOT used for pricing.
CDE_FEE_BPS_VENUE_QUOTE: Dict[str, float] = {"SOL": 10.90, "XRP": 9.03, "BNB": 8.69}


# =============================================================================
# [P1-1] VENUE FEE SCHEDULE — the structure read off the venue's own ledger
# =============================================================================
#
# Phase 0 (reports/HMATS_PHASE0_COST_AUDIT.md) read every fill's
# `commission_detail_total`: commission = client_commission (a % of notional,
# set by the 30-day-volume tier) + venue $0.10/ct + clearing $0.01-0.05/ct.
# BOTH earlier models were half right: P315's flat-per-contract and P334's
# pure percentage. The table above (CDE_FEE_BPS) is the P334 percentage-only
# model and, measured against all 211 fills, was +10.3% overall but -13.8%
# at the CURRENT tier (BTC 9.87bps taker modelled vs 11.35 actual), because it
# has no fixed term and was calibrated at a cheaper tier.
#
# The schedule is loaded from configs/cost/venue_fee_schedule_coinbase.json, which
# scripts/export_venue_fee_schedule.py writes from phase0_params.json — no fee
# number is typed in code. Absent/unreadable file -> None, and cde_fee_bps
# falls back to CDE_FEE_BPS above (the pre-P1-1 behavior), logged once.
import json as _json
import os as _os
from dataclasses import dataclass as _dataclass

_SCHEDULE_MEASURED_ON = "2026-10-18"
_SCHEDULE_MEASURED_BY = ("scripts/export_venue_fee_schedule.py <- reports/phase0_params.json + data/phase0/fills_raw.parquet (tiers, fixed components) "
                         "(Phase 0: 211 venue fills + get_transaction_summary)")
_SCHEDULE_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "configs", "cost", "venue_fee_schedule_coinbase.json")


@_dataclass(frozen=True)
class VenueFeeSchedule:
    venue: str
    rate_bps_maker: float
    rate_bps_taker: float
    fixed_per_contract_usd: float
    min_fee_per_contract_usd: float
    contract_size: Dict[str, float]
    # [P1R-1] Volume tiers ((min_30d_volume_usd, maker_bps, taker_bps), ...)
    # ascending by volume, and the per-asset fixed components
    # {asset: {"venue": $, "clearing": $, "regulatory": $}} per contract.
    # Both come from the producer (scripts/export_venue_fee_schedule.py),
    # derived from the venue's own fills; empty -> the flat P1-1 fields.
    tiers: Tuple[Tuple[float, float, float], ...] = ()
    fixed_components_by_asset: Optional[Dict[str, Dict[str, float]]] = None

    def rate_bps(self, is_maker: bool,
                 rolling_30d_volume_usd: Optional[float] = None) -> float:
        """The pct-of-notional rate for the tier the 30-day volume falls in.

        Unknown volume (None/NaN/negative) -> the MOST EXPENSIVE tier: an
        unreadable volume must never buy a cheaper fee (P167 safe side)."""
        if not self.tiers:
            return self.rate_bps_maker if is_maker else self.rate_bps_taker
        col = 1 if is_maker else 2
        v = rolling_30d_volume_usd
        if v is None or v != v or v < 0:
            return max(t[col] for t in self.tiers)
        rate = self.tiers[0][col]
        for t in self.tiers:
            if v >= t[0]:
                rate = t[col]
        return rate

    def fixed_usd_per_contract(self, asset: str) -> float:
        comps = (self.fixed_components_by_asset or {}).get(str(asset).upper())
        if comps:
            return float(sum(float(x) for x in comps.values()))
        return self.fixed_per_contract_usd

    def fee_usd(self, asset: str, contracts: float, price: float,
                is_maker: bool,
                rolling_30d_volume_usd: Optional[float] = None) -> float:
        cs = self.contract_size[str(asset).upper()]
        n = abs(float(contracts))
        notional = n * cs * float(price)
        if self.tiers:
            rate = self.rate_bps(is_maker, rolling_30d_volume_usd)
        else:
            rate = self.rate_bps_maker if is_maker else self.rate_bps_taker
        fee = notional * rate / 1e4 + self.fixed_usd_per_contract(asset) * n
        return max(fee, self.min_fee_per_contract_usd * n)

    def fee_bps(self, asset: str, contracts: float, price: float,
                is_maker: bool,
                rolling_30d_volume_usd: Optional[float] = None) -> float:
        n = abs(float(contracts))
        notional = n * self.contract_size[str(asset).upper()] * float(price)
        if not notional > 0:
            return float("inf")
        return self.fee_usd(asset, n, price, is_maker,
                            rolling_30d_volume_usd) / notional * 1e4

    def covers(self, asset: str) -> bool:
        return str(asset).upper() in self.contract_size


_SCHEDULE_CACHE: Dict[str, Optional[VenueFeeSchedule]] = {}
_SCHEDULE_WARNED = {"missing": False}


def load_venue_fee_schedule(path: Optional[str] = None,
                            *, reload: bool = False) -> Optional[VenueFeeSchedule]:
    """The Phase-0-derived schedule, or None (callers keep CDE_FEE_BPS)."""
    p = path or _os.environ.get("HMATS_VENUE_FEE_SCHEDULE") or _SCHEDULE_PATH
    if not reload and p in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[p]
    sched: Optional[VenueFeeSchedule] = None
    try:
        with open(p, encoding="utf-8") as fh:
            d = _json.load(fh)
        sched = VenueFeeSchedule(
            venue=str(d["venue"]),
            rate_bps_maker=float(d["rate_bps_maker"]),
            rate_bps_taker=float(d["rate_bps_taker"]),
            fixed_per_contract_usd=float(d["fixed_per_contract_usd"]),
            min_fee_per_contract_usd=float(d.get("min_fee_per_contract_usd", 0.0)),
            contract_size={str(k).upper(): float(v)
                           for k, v in dict(d["contract_size"]).items()},
            tiers=tuple(sorted(
                (float(t["min_30d_volume_usd"]), float(t["maker_bps"]),
                 float(t["taker_bps"])) for t in (d.get("tiers") or []))),
            fixed_components_by_asset=(
                {str(k).upper(): {c: float(x) for c, x in dict(v).items()}
                 for k, v in dict(d["fixed_components_by_asset"]).items()}
                if d.get("fixed_components_by_asset") else None),
        )
        if not (sched.rate_bps_taker > 0 and sched.rate_bps_maker > 0):
            raise ValueError("non-positive rate")
    except Exception as e:  # noqa: silent-swallow — logged once; caller falls back to CDE_FEE_BPS
        if not _SCHEDULE_WARNED["missing"]:
            _SCHEDULE_WARNED["missing"] = True
            logger.warning("[P1-1] venue fee schedule unavailable at %s (%s: %s) — "
                           "pricing falls back to the CDE_FEE_BPS percentage table",
                           p, type(e).__name__, e)
        sched = None
    _SCHEDULE_CACHE[p] = sched
    return sched


def _contract_sizes() -> Dict[str, float]:
    """asset -> CDE contract size, DERIVED from the two existing sources.

    [P310] Imported, never restated: the product id comes from
    `exchange.symbol_mapping.SYMBOL_MAP` and the size from
    `CoinbaseAdapter._CONTRACT_SIZE_FALLBACK`. A local copy of either would
    drift the first time a contract rolls. [P1R-8] The product ids carry
    "20DEC30" but the venue API reports contract_expiry 2089-12-30T16:00Z
    for all three (Phase 0 product_specs) — the API is authoritative; no
    code depends on either date.
    """
    out: Dict[str, float] = {}
    try:
        from exchange.symbol_mapping import SYMBOL_MAP
        from exchange.coinbase_adapter import CoinbaseAdapter
        # Read the maps by LOOKUP rather than through to_venue_symbol(), which
        # raises on an unmapped asset: a per-asset try/except here would be an
        # unlogged early exit, and "this asset has no perp" is an ordinary
        # absence, not an exception.
        pids = SYMBOL_MAP.get("coinbase", {}).get("perp", {})
        table = CoinbaseAdapter._CONTRACT_SIZE_FALLBACK
        for asset in CDE_FEE_BPS:
            cs = table.get(pids.get(asset, ""))
            if cs:
                out[asset] = float(cs)
    except Exception as e:  # noqa: silent-swallow — logged; caller degrades
        logger.warning("[P315] contract-size resolution unavailable (%s: %s); "
                       "per-contract fee pricing will return None and callers "
                       "keep their existing figure", type(e).__name__, e)
    return out


def cde_fee_bps(asset: str, price: float, *, is_maker: bool,
                contract_size: Optional[float] = None) -> Optional[Tuple[float, str]]:
    """Per-LEG CDE fee in bps of notional, or None when it cannot be priced.

    Returns ``(bps, provenance)``. `provenance` names whether the underlying
    per-contract fee was measured for THIS asset or assumed, so a caller (and
    a reader of the logs) can tell a measurement from a fallback (P169).
    """
    a = str(asset or "").upper().strip()
    try:
        px = float(price)
    except (TypeError, ValueError):  # noqa: silent-swallow — see below
        # Deliberately silent HERE: this is called per-asset per-tick, and the
        # CALLER logs the outcome once ("per-contract fee not priceable —
        # keeping the modeled figure"). Logging in both places would make a
        # routine absence look like two faults (P202). Returning None rather
        # than 0.0 is the load-bearing part: a fabricated free trade is what
        # this module exists to stop (P2/P167).
        return None
    if not (px > 0) or px != px:            # <=0, NaN
        return None

    cs = contract_size
    if cs is None:
        cs = _contract_sizes().get(a)
    if not cs or cs <= 0:
        return None

    notional_per_contract = cs * px
    if notional_per_contract <= 0:
        return None
    # [P334] The price and contract size no longer enter the ARITHMETIC — the
    # fee is a percentage — but they are still REQUIRED, and deliberately so:
    # `core.seat_alpha.resolve_seat_edge` uses "can this asset be priced at
    # all" as its effect-interlock (P321b), refusing the calibrated alpha
    # whenever the fee side cannot price the asset. Dropping the price
    # dependency here would silently make that interlock inert — a flag-based
    # check that cannot tell "armed" from "took effect", which is the exact
    # failure P321b exists to prevent. So this stays a liveness check.
    sched = CDE_FEE_BPS.get(a)
    if sched is None:
        return _WORST_BPS, f"assumed_worst_unknown_asset:{a}"
    prov = ("assumed_no_measured_fill" if a in CDE_FEE_ASSUMED
            else "measured_venue_preview" if a in CDE_FEE_PREVIEW
            else "measured_fill_quality")
    return sched["maker" if is_maker else "taker"], prov


def venue_fee_bps(asset: str, price: float, *, is_maker: bool,
                  contract_size: Optional[float] = None,
                  rolling_30d_volume_usd: Optional[float] = None) -> Optional[Tuple[float, str]]:
    """[P1-1] Per-LEG fee in bps from the Phase-0 VENUE SCHEDULE.

    Live pricing entry point (alpha gate, seat-alpha interlock). Kept separate
    from `cde_fee_bps` on purpose: training/funding_legs_lab.py calls
    `cde_fee_bps(asset, 1.0, contract_size=1.0)` to read a constant per-asset
    RATE, and the schedule's fixed per-contract term would turn that probe
    into ~1,100bps/leg. training/ is off-limits in Phase 1, so the legacy
    percentage table stays where the lab reads it.

    Per-contract bps is independent of the contract COUNT (both fee terms are
    linear in contracts), so one contract prices any order size exactly.
    Falls back to `cde_fee_bps` (table) for assets the schedule does not
    cover or when it cannot load; None when the price is unusable (P2).
    """
    a = str(asset or "").upper().strip()
    try:
        px = float(price)
    except (TypeError, ValueError):  # noqa: silent-swallow — caller logs the unpriceable tick (P202)
        return None
    if not (px > 0) or px != px:
        return None
    vfs = load_venue_fee_schedule()
    if vfs is None or not vfs.covers(a):
        return cde_fee_bps(a, px, is_maker=is_maker, contract_size=contract_size)
    cs = contract_size if contract_size else vfs.contract_size[a]
    notional = float(cs) * px
    if not notional > 0:
        return None
    vol = rolling_30d_volume_usd
    if vol is None and vfs.tiers:
        vol = rolling_30d_volume_usd_from_ledger()
    rate = vfs.rate_bps(is_maker, vol) if vfs.tiers else (
        vfs.rate_bps_maker if is_maker else vfs.rate_bps_taker)
    bps = rate + vfs.fixed_usd_per_contract(a) / notional * 1e4
    bps = max(bps, vfs.min_fee_per_contract_usd / notional * 1e4)
    return bps, "venue_fee_schedule_phase0"


# [P1R-1] Rolling 30-day traded volume from the sleeve's own fill ledger
# (data/fill_quality.jsonl, P290), so the tier is DERIVED from what we
# actually traded rather than typed in. Only `status == "filled"` rows with a
# fill price count; unresolved rows are excluded, which can only UNDERSTATE
# volume and so price a more expensive tier (the safe direction). Unreadable
# ledger -> None -> the most expensive tier (VenueFeeSchedule.rate_bps).
_VOL_CACHE: Dict[str, Tuple[float, Optional[float]]] = {}
_VOL_TTL_SEC = 3600.0


def rolling_30d_volume_usd_from_ledger(path: Optional[str] = None,
                                       now: Optional[float] = None,
                                       *, use_cache: bool = True) -> Optional[float]:
    import time as _time
    t_now = float(now) if now is not None else _time.time()
    p = path or _os.path.join(
        _os.environ.get("HMATS_DATA_DIR") or _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"),
        "fill_quality.jsonl")
    if use_cache and now is None:
        hit = _VOL_CACHE.get(p)
        if hit and t_now - hit[0] < _VOL_TTL_SEC:
            return hit[1]
    vfs = load_venue_fee_schedule()
    sizes = dict(vfs.contract_size) if vfs else {}
    sizes.update({k: v for k, v in _contract_sizes().items() if k not in sizes})
    total: Optional[float] = None
    try:
        with open(p, encoding="utf-8") as fh:
            total = 0.0
            for line in fh:
                try:
                    r = _json.loads(line)
                    if not isinstance(r, dict) or r.get("status") != "filled":
                        continue
                    ts = float(r.get("ts"))
                    if not (t_now - 30 * 86400.0 <= ts <= t_now):
                        continue
                    cs = sizes.get(str(r.get("asset") or "").upper())
                    px = float(r.get("fill_avg_price"))
                    n = abs(float(r.get("contracts")))
                    if cs and px > 0:
                        total += n * cs * px
                except (TypeError, ValueError):  # noqa: silent-swallow — a malformed row adds no volume (understates -> dearer tier)
                    continue
    except FileNotFoundError:  # noqa: silent-swallow — absence returns None and the schedule prices the dearest tier
        total = None
    if use_cache and now is None:
        _VOL_CACHE[p] = (t_now, total)
    return total


def schedule_pct_fraction(is_maker: bool) -> Optional[float]:
    """[P1R-1] The schedule's pct-of-notional rate as a FRACTION, for the
    fraction-based fallback models (core.paper_fee_service.VENUE_FEE_STD,
    core.execution_service._COINBASE_*_BPS) that used to hardcode the refuted
    0/3bps Coinbase model. Tier from the ledger's rolling 30d volume; the fixed
    per-contract term is NOT representable as a fraction and is added by
    venue_fee_bps where a price is known. None when the schedule is absent."""
    vfs = load_venue_fee_schedule()
    if vfs is None:
        return None
    vol = rolling_30d_volume_usd_from_ledger() if vfs.tiers else None
    return vfs.rate_bps(is_maker, vol) / 1e4

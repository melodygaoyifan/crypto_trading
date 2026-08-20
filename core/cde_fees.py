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
_MEASURED_ON = "2026-08-19"
_MEASURED_BY = "data/fill_quality.jsonl via scripts/fill_quality_review.py (P290/P315)"


CDE_FEE_PER_CONTRACT_USD: Dict[str, Dict[str, float]] = {
    "BTC": {"maker": 0.603, "taker": 0.635},
    "ETH": {"maker": 0.264, "taker": 0.264 * _BTC_TAKER_OVER_MAKER},
    "SOL": {"maker": 0.635, "taker": 0.635},
}

# Assets whose fee is ASSUMED rather than measured — surfaced, not hidden.
CDE_FEE_ASSUMED = frozenset({"SOL"})

# The most expensive measured per-contract fee, used for anything unknown.
_WORST_PER_CONTRACT = max(
    max(v["maker"], v["taker"]) for v in CDE_FEE_PER_CONTRACT_USD.values())


def _contract_sizes() -> Dict[str, float]:
    """asset -> CDE contract size, DERIVED from the two existing sources.

    [P310] Imported, never restated: the product id comes from
    `exchange.symbol_mapping.SYMBOL_MAP` and the size from
    `CoinbaseAdapter._CONTRACT_SIZE_FALLBACK`. A local copy of either would
    drift the first time a contract rolls (they expire 2030-12-20).
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
        for asset in CDE_FEE_PER_CONTRACT_USD:
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

    sched = CDE_FEE_PER_CONTRACT_USD.get(a)
    if sched is None:
        per_ct, prov = _WORST_PER_CONTRACT, f"assumed_worst_unknown_asset:{a}"
    else:
        per_ct = sched["maker" if is_maker else "taker"]
        prov = ("assumed_no_measured_fill" if a in CDE_FEE_ASSUMED
                else "measured_fill_quality")

    notional_per_contract = cs * px
    if notional_per_contract <= 0:
        return None
    return (per_ct / notional_per_contract) * 1e4, prov

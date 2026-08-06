from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# [P169] Per-leg fee schedules by venue. `fee_std` used to be hardcoded to
# Kraken (0.0016/0.0026) regardless of where the order actually executed, which
# after the 2026-06-13 Coinbase cutover priced every routed fill on the wrong
# exchange by ~8x. These are FALLBACKS — the venue's own reported fee wins
# whenever it is available.
VENUE_FEE_STD: Dict[str, Dict[str, float]] = {
    "kraken": {"maker": 0.0016, "taker": 0.0026},
    "coinbase": {"maker": 0.0000, "taker": 0.0003},
}
_DEFAULT_VENUE = "kraken"

# [P169] Currencies a reported fee may be denominated in and still be taken at
# face value as USD. A fee of "0.0031" means something very different in ETH
# than in USD, so anything outside this set falls back to the model rather than
# being silently misread by three orders of magnitude.
USD_EQUIVALENT_CURRENCIES = frozenset({"USD", "ZUSD", "USDT", "USDC", "USDG", ""})


def venue_fee_std(venue: str, is_maker: bool) -> float:
    """[P169] Per-leg fee fraction for a venue. Unknown venue -> Kraken, the
    more expensive of the two, because over-charging fails toward not trading.
    """
    schedule = VENUE_FEE_STD.get(str(venue or "").strip().lower(),
                                 VENUE_FEE_STD[_DEFAULT_VENUE])
    return schedule["maker" if is_maker else "taker"]


def resolve_trade_fee_usd(
    *,
    executed_notional_usd: float,
    venue_fee_usd: Optional[float],
    venue_fee_currency: str,
    modelled_fee_usd: float,
) -> Dict[str, Any]:
    """[P169] Prefer the fee the venue actually charged over the modelled one.

    `OrderResult.fee` is parsed straight out of the ccxt order response
    (`execution_manager.py:1462`) and was then thrown away: every fee in
    `data/trade_attribution.jsonl` is modelled, none measured. Measured over
    the 52 closed trades there, 32/52 entry legs recorded *exactly* 16.0bps —
    Kraken's maker rate — because the model asks `order_type == "LIMIT"` and
    calls that a maker fill.

    Returns the fee plus a `fee_source` of "venue" or "model" and, when the
    model is used, `fee_source_reason` saying why. The two are never conflated:
    a modelled fee that happens to be right is still not an observation.
    """
    notional = max(0.0, float(executed_notional_usd or 0.0))
    out: Dict[str, Any] = {
        "trade_fee_usd": float(modelled_fee_usd),
        "fee_source": "model",
        "fee_source_reason": "",
        "venue_fee_usd": None,
        "venue_fee_currency": str(venue_fee_currency or ""),
    }

    if venue_fee_usd is None:
        out["fee_source_reason"] = "venue reported no fee"
        return out
    # The handler below looks like a silent swallow but is not: every rejection
    # path in this function records WHY in `fee_source_reason` and leaves
    # `fee_source` at "model", so a caller can always tell a measured fee from a
    # modelled one and read the cause. Logging would fire on every fill of a
    # venue that simply does not report fees; the reason field is the durable
    # record.
    try:
        fee = float(venue_fee_usd)
    except (TypeError, ValueError):  # noqa: silent-swallow — reason is recorded, see above
        out["fee_source_reason"] = f"venue fee not numeric: {venue_fee_usd!r}"
        return out
    if fee != fee or fee in (float("inf"), float("-inf")):  # NaN / inf
        out["fee_source_reason"] = "venue fee not finite"
        return out
    if fee < 0:
        out["fee_source_reason"] = f"venue fee negative ({fee})"
        return out

    currency = str(venue_fee_currency or "").strip().upper()
    if currency not in USD_EQUIVALENT_CURRENCIES:
        # Do NOT convert here — a wrong conversion rate is worse than a
        # modelled fee, because it looks like an observation.
        out["fee_source_reason"] = f"venue fee denominated in {currency}, not USD"
        return out

    if fee == 0.0 and notional > 0:
        # A genuine 0 is possible (Coinbase maker, Kraken+ free tier), but so is
        # "the field was absent and defaulted to 0.0" — OrderResult.fee is
        # `float((order_status.get('fee') or {}).get('cost', 0) or 0)`, which
        # collapses missing and zero into the same value. Unresolvable here, so
        # keep the model and say so rather than book a free trade.
        out["fee_source_reason"] = "venue fee 0.0 — indistinguishable from absent"
        return out

    out["trade_fee_usd"] = fee
    out["fee_source"] = "venue"
    out["venue_fee_usd"] = fee
    return out


def paper_trade_requires_margin(
    *,
    execution_direction: float,
    regime_leverage: float,
    existing_position: Optional[Dict[str, Any]] = None,
) -> bool:
    """Paper shorts and leveraged trades must use margin-style fee semantics."""
    existing_position = existing_position or {}
    existing_direction = float(existing_position.get("direction", 0.0) or 0.0)
    existing_leverage = float(existing_position.get("regime_leverage", 1.0) or 1.0)
    return bool(
        float(regime_leverage or 1.0) > 1.0
        or existing_leverage > 1.0
        or float(execution_direction or 0.0) < 0.0
        or existing_direction < 0.0
    )


def get_paper_margin_opening_fee_bps(
    asset: str,
    *,
    default_bps: float,
    margin_fee_map: Optional[Dict[str, Any]] = None,
) -> float:
    asset_key = str(asset or "").upper().replace("/USD", "").replace("USD", "")
    margin_fee_map = dict(margin_fee_map or {})
    try:
        return float(margin_fee_map.get(asset_key, default_bps) or default_bps)
    except Exception:
        return float(default_bps)


def build_execution_fee_result(
    *,
    asset: str,
    executed_notional_usd: float,
    order_type: str,
    execution_direction: float,
    regime_leverage: float,
    existing_position: Optional[Dict[str, Any]] = None,
    fee_blending_enabled: bool,
    default_margin_opening_fee_bps: float,
    margin_fee_map: Optional[Dict[str, Any]] = None,
    fee_record_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    venue: str = _DEFAULT_VENUE,
    venue_fee_usd: Optional[float] = None,
    venue_fee_currency: str = "",
) -> Dict[str, Any]:
    """Build fee result for an executed order without mutating portfolio state."""
    executed_notional_usd = max(0.0, float(executed_notional_usd or 0.0))
    # [P169] `is_maker` here is the ORDER TYPE, not the fill type. A limit order
    # that crosses the spread is filled as a taker and charged accordingly. This
    # is the same conflation P166 found in review_aggregator's `maker_fee_ratio`
    # (n_limit / n_classified). It is kept as the model's best guess but is now
    # labelled an assumption, and the venue's reported fee overrides it.
    is_maker = str(order_type or "").upper() == "LIMIT"
    fee_std = venue_fee_std(venue, is_maker)
    requires_margin = paper_trade_requires_margin(
        execution_direction=execution_direction,
        regime_leverage=regime_leverage,
        existing_position=existing_position,
    )
    is_spot_trade = not requires_margin

    if not fee_blending_enabled:
        return {
            "fee_usd": 0.0,
            "trade_fee_usd": 0.0,
            "fee_effective": 0.0,
            "fee_std": float(fee_std),
            "is_maker": bool(is_maker),
            "is_spot": bool(is_spot_trade),
            "requires_margin": bool(requires_margin),
            "executed_notional_usd": float(executed_notional_usd),
            "margin_opening_fee_bps": 0.0,
            "margin_opening_fee_usd": 0.0,
            "total_fee_usd": 0.0,
            "blending_applied": False,
            "fill_tracked": False,
            "reason": "Fee model disabled",
            # [P169] Explicitly not "model" — nothing was priced at all. A
            # reader must be able to tell $0.00-because-disabled from
            # $0.00-because-the-venue-charged-nothing.
            "fee_source": "disabled",
            "fee_source_reason": "fee model disabled",
            "venue": str(venue or _DEFAULT_VENUE).lower(),
            "is_maker_assumed": True,
        }

    fee_result: Dict[str, Any] = {}
    if executed_notional_usd > 0 and fee_record_fn is not None:
        try:
            fee_result = dict(
                fee_record_fn(
                    notional_usd=executed_notional_usd,
                    fee_std=fee_std,
                    symbol=f"{asset}/USD",
                    is_spot=is_spot_trade,
                    # [P169] Was hardcoded "kraken". The Kraken+ blender gates on
                    # `exchange.lower() != "kraken"` (kraken_plus_fee_blender.py:454)
                    # to decide whether tier blending applies, so a hardcoded
                    # "kraken" fed Coinbase-routed fills into Kraken's monthly
                    # volume tracker and handed them free-tier discounts they
                    # never earned. Pass the venue the order actually hit.
                    exchange=str(venue or _DEFAULT_VENUE).strip().lower(),
                    is_maker=is_maker,
                )
                or {}
            )
        except Exception as exc:
            fee_result = {"error": str(exc)}

    modelled_fee_usd = float(
        fee_result.get("fee_usd", executed_notional_usd * fee_std)
        or (executed_notional_usd * fee_std)
    )
    fee_effective = float(fee_result.get("fee_effective", fee_std) or fee_std)

    # [P169] Measured beats modelled.
    _resolved = resolve_trade_fee_usd(
        executed_notional_usd=executed_notional_usd,
        venue_fee_usd=venue_fee_usd,
        venue_fee_currency=venue_fee_currency,
        modelled_fee_usd=modelled_fee_usd,
    )
    trade_fee_usd = float(_resolved["trade_fee_usd"])
    if _resolved["fee_source"] == "venue" and executed_notional_usd > 0:
        # Report what was actually paid, not what the schedule says.
        fee_effective = trade_fee_usd / executed_notional_usd
    margin_opening_fee_bps = (
        get_paper_margin_opening_fee_bps(
            asset,
            default_bps=default_margin_opening_fee_bps,
            margin_fee_map=margin_fee_map,
        )
        if requires_margin
        else 0.0
    )

    result = dict(fee_result)
    result.update(
        {
            "fee_usd": trade_fee_usd,
            "trade_fee_usd": trade_fee_usd,
            "fee_effective": fee_effective,
            "fee_std": float(fee_result.get("fee_std", fee_std) or fee_std),
            "is_maker": bool(is_maker),
            "is_spot": bool(is_spot_trade),
            "requires_margin": bool(requires_margin),
            "executed_notional_usd": float(executed_notional_usd),
            "margin_opening_fee_bps": float(margin_opening_fee_bps),
            "margin_opening_fee_usd": 0.0,
            "total_fee_usd": trade_fee_usd,
            # [P169] Provenance travels with the number.
            "fee_source": _resolved["fee_source"],
            "fee_source_reason": _resolved["fee_source_reason"],
            "venue": str(venue or _DEFAULT_VENUE).lower(),
            "modelled_fee_usd": modelled_fee_usd,
            "venue_fee_usd": _resolved["venue_fee_usd"],
            "venue_fee_currency": _resolved["venue_fee_currency"],
            # is_maker is derived from ORDER TYPE, never observed. Say so.
            "is_maker_assumed": True,
        }
    )
    return result


def compute_margin_opening_fee_usd(
    fee_result: Optional[Dict[str, Any]],
    *,
    opening_notional_usd: float,
) -> float:
    opening_notional_usd = max(0.0, float(opening_notional_usd or 0.0))
    if opening_notional_usd <= 0:
        return 0.0
    fee_result = fee_result or {}
    margin_bps = float(fee_result.get("margin_opening_fee_bps", 0.0) or 0.0)
    return opening_notional_usd * margin_bps / 10000.0


def split_trade_fee_usd(
    *,
    trade_fee_usd: float,
    total_notional_usd: float,
    close_notional_usd: float,
) -> Dict[str, float]:
    trade_fee_usd = max(0.0, float(trade_fee_usd or 0.0))
    total_notional_usd = max(0.0, float(total_notional_usd or 0.0))
    close_notional_usd = max(0.0, float(close_notional_usd or 0.0))
    if trade_fee_usd <= 0.0 or total_notional_usd <= 0.0:
        return {
            "close_fee_usd": 0.0,
            "open_fee_usd": 0.0,
            "close_fraction": 0.0,
        }
    close_fraction = min(1.0, max(0.0, close_notional_usd / total_notional_usd))
    close_fee_usd = trade_fee_usd * close_fraction
    open_fee_usd = max(trade_fee_usd - close_fee_usd, 0.0)
    return {
        "close_fee_usd": close_fee_usd,
        "open_fee_usd": open_fee_usd,
        "close_fraction": close_fraction,
    }

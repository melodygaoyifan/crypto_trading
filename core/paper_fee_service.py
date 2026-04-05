from __future__ import annotations

from typing import Any, Callable, Dict, Optional


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
) -> Dict[str, Any]:
    """Build fee result for an executed order without mutating portfolio state."""
    executed_notional_usd = max(0.0, float(executed_notional_usd or 0.0))
    is_maker = str(order_type or "").upper() == "LIMIT"
    fee_std = 0.0016 if is_maker else 0.0026
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
                    exchange="kraken",
                    is_maker=is_maker,
                )
                or {}
            )
        except Exception as exc:
            fee_result = {"error": str(exc)}

    trade_fee_usd = float(
        fee_result.get("fee_usd", executed_notional_usd * fee_std)
        or (executed_notional_usd * fee_std)
    )
    fee_effective = float(fee_result.get("fee_effective", fee_std) or fee_std)
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

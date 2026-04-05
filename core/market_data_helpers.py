from typing import Any, Mapping


def effective_volume_ratio(
    market_data: Mapping[str, Any],
    *,
    is_4h_bar_close: bool = False,
    fallback: float = 1.0,
) -> float:
    """Return pace-adjusted volume ratio for partial 4H bars when available."""
    raw_ratio = float(market_data.get("volume_ratio", fallback) or 0.0)
    if is_4h_bar_close:
        return raw_ratio
    try:
        effective_ratio = float(
            market_data.get("volume_ratio_effective", raw_ratio) or raw_ratio
        )
    except Exception:
        effective_ratio = raw_ratio
    return effective_ratio if effective_ratio >= 0.0 else raw_ratio

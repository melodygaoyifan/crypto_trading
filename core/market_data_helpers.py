from typing import Any, Mapping, Optional


def signal_value(
    key: str,
    agent_signals: Optional[Mapping[str, Any]],
    market_data: Optional[Mapping[str, Any]],
    default: Any = None,
) -> Any:
    """[P173] Read a signal key from whichever dict actually carries it.

    `agent_signals` first, then `market_data`, then the default. Use this at
    any site where the producer and the consumer live in different modules.

    This exists because reading the wrong dict is the single most repeated bug
    in this codebase — P2, P15, P16, P23, P85, P138, P139, P140, P147, P152,
    P155d, P170, P171, and the three sites P173 fixes. The failure is always
    silent: `.get()` returns the default, the default was picked to look
    reasonable, and a constant flows downstream forever wearing the name of a
    measurement.

    `None` is treated as absent, so a producer that writes an explicit `None`
    falls through rather than pinning the value. That is deliberate: every
    caller here wants "the measurement, if anyone made one".

    `tools/lint_orphan_signal_reads.py` (P171) is the CI half of this; the
    helper is the code half. Neither is sufficient alone — the scanner cannot
    see through a helper it does not know about, which is why the sites below
    keep their explicit key names.
    """
    for src in (agent_signals, market_data):
        if not isinstance(src, Mapping):
            continue
        val = src.get(key, None)
        if val is not None:
            return val
    return default


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

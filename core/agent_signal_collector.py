"""
Agent Signal Collector - extracted from main.py Phase 3.

Collects signals from supplementary ADVISE agents into agent_signals dict.
Each agent follows a try/except pattern with graceful degradation.

This module provides the `collect_supplementary_signals` function that
replaces ~400 lines of repetitive agent polling in _process_4h_tick_inner.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_call(
    name: str,
    fn: Callable,
    *args,
    **kwargs,
) -> Optional[Any]:
    """Call a function with exception handling, returning None on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"[WIRE] {name} skipped: {e}")
        return None


async def _safe_async_call(
    name: str,
    fn: Callable,
    *args,
    **kwargs,
) -> Optional[Any]:
    """Call an async function with exception handling, returning None on failure."""
    try:
        return await fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"[WIRE] {name} skipped: {e}")
        return None


def collect_short_bias_signal(
    agent: Any,
    asset: str,
    market_data: Dict[str, Any],
    agent_signals: Dict[str, Any],
    *,
    regime: str,
    skip_regimes: frozenset,
    p0_abort_tick: bool,
) -> None:
    """Collect short bias agent signal into agent_signals."""
    if agent is None or p0_abort_tick:
        return
    if regime in skip_regimes:
        return
    try:
        signal = agent.analyze(
            asset=asset,
            market_data=market_data,
            regime=regime,
        )
        if signal:
            agent_signals["short_bias_direction"] = signal.direction
            agent_signals["short_bias_confidence"] = signal.confidence
            agent_signals["short_bias_funding_rate"] = getattr(signal, "funding_rate", 0.0)
            agent_signals["short_bias_funding_bias"] = getattr(signal, "funding_bias", 0.0)
            agent_signals["short_bias_penalty_active"] = getattr(signal, "penalty_active", False)
            agent_signals["short_bias_penalty_factor"] = getattr(signal, "penalty_factor", 1.0)
            agent_signals["short_bias_authority"] = "ADVISE"
            if abs(signal.direction) > 0.1:
                logger.info(
                    f"[V6.2.3] ShortBias {asset}: dir={signal.direction:+.2f} "
                    f"conf={signal.confidence:.2f} "
                    f"funding_bias={getattr(signal, 'funding_bias', 0):.3f} "
                    f"penalty={getattr(signal, 'penalty_factor', 1.0):.2f}"
                )
    except Exception as e:
        logger.debug(f"[V6.2.3] ShortBias skipped for {asset}: {e}")


def collect_funding_rate_signal(
    agent: Any,
    asset: str,
    market_data: Dict[str, Any],
    agent_signals: Dict[str, Any],
    *,
    p0_abort_tick: bool,
) -> None:
    """Collect funding rate agent signal into agent_signals."""
    if agent is None or p0_abort_tick:
        return
    symbol = asset.replace("/USD", "").upper()
    try:
        signal = agent.generate_signal(
            symbol=symbol,
            funding_rate=market_data.get("funding_rate", 0.0),
            funding_bias=market_data.get("funding_bias", 0.0),
            open_interest=market_data.get("open_interest", 0.0),
            oi_change_1h_pct=market_data.get("oi_change_1h_pct", 0.0),
        )
        if signal:
            agent_signals["funding_rate_direction"] = signal.direction
            agent_signals["funding_rate_confidence"] = signal.confidence
            agent_signals["funding_rate_carry_bias"] = signal.carry_bias
            agent_signals["funding_rate_authority"] = "ADVISE"
            if abs(signal.direction) > 0.1:
                logger.info(
                    f"[V10] FundingRate {asset}: dir={signal.direction:+.2f} "
                    f"conf={signal.confidence:.2f} carry={signal.carry_bias:+.3f}"
                )
    except Exception as e:
        logger.debug(f"[V10] FundingRateAgent skipped: {e}")


async def collect_options_sentiment_signal(
    agent: Any,
    asset: str,
    market_data: Dict[str, Any],
    agent_signals: Dict[str, Any],
    *,
    p0_abort_tick: bool,
) -> None:
    """Collect options sentiment agent signal into agent_signals."""
    if agent is None or p0_abort_tick:
        return
    try:
        signal = await agent.generate_signal(asset, market_data)
        if signal and signal.get("valid", False):
            for k, v in signal.items():
                if k != "valid":
                    agent_signals[f"options_{k}"] = v
            agent_signals["options_authority"] = "ADVISE"
            pcr = signal.get("put_call_ratio", 0)
            if pcr > 0:
                logger.info(
                    f"[WIRE] OptionsSentiment {asset}: pcr={pcr:.2f}, "
                    f"max_pain=${signal.get('max_pain', 0):,.0f}, "
                    f"skew={signal.get('put_call_skew', 0):.3f}"
                )
    except Exception as e:
        logger.debug(f"[WIRE] OptionsSentimentAgent skipped: {e}")


def collect_squeeze_signal(
    agent: Any,
    asset: str,
    market_data: Dict[str, Any],
    agent_signals: Dict[str, Any],
    *,
    p0_abort_tick: bool,
) -> None:
    """Collect squeeze detector signal into agent_signals."""
    if agent is None or p0_abort_tick:
        return
    try:
        signal = agent.generate_signal(asset, market_data)
        if signal and signal.get("valid", False):
            for k, v in signal.items():
                if k != "valid":
                    agent_signals[f"squeeze_{k}"] = v
            agent_signals["squeeze_authority"] = "ADVISE"
            if signal.get("squeeze_level", 0) > 0:
                logger.info(
                    f"[WIRE] SqueezeDetector {asset}: level={signal['squeeze_level']}, "
                    f"risk={signal.get('squeeze_risk', 0):.2f}"
                )
    except Exception as e:
        logger.debug(f"[WIRE] SqueezeDetectorAgent skipped: {e}")

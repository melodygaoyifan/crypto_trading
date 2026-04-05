"""
================================================================================
HMATS Kraken Exchange - Kraken连接
================================================================================
Version: 5.1.0-CLOUD
Status: ACTIVE (唯一执行交易所)

Kraken API连接和管理。

核心组件:
- KrakenTokenBucket: API限流
- KrakenDefensiveLink: 防御性连接
- KrakenQuantAgent: 量化代理

================================================================================
"""

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# EXPORTS
# =============================================================================

try:
    from exchange.kraken.kraken_token_bucket import KrakenTokenBucket
except ImportError as e:
    logger.warning(f"KrakenTokenBucket import failed: {e}")
    KrakenTokenBucket = None

try:
    from exchange.kraken.kraken_defensive_link import KrakenDefensiveLink
except ImportError as e:
    logger.warning(f"KrakenDefensiveLink import failed: {e}")
    KrakenDefensiveLink = None

try:
    from exchange.kraken.kraken_quant_agent import KrakenQuantAgent
except ImportError as e:
    logger.warning(f"KrakenQuantAgent import failed: {e}")
    KrakenQuantAgent = None

# Singleton accessors from infra.kraken_link
try:
    from infra.kraken_link import get_kraken_link, reset_kraken_link
except ImportError as e:
    logger.warning(f"kraken_link singleton accessors import failed: {e}")
    get_kraken_link = None
    reset_kraken_link = None

__all__ = [
    "KrakenTokenBucket",
    "KrakenDefensiveLink",
    "KrakenQuantAgent",
    "get_kraken_link",
    "reset_kraken_link",
]

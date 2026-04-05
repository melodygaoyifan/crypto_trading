"""
WarmupTracker - tracks per-asset tick warmup state.

Extracted from main.py (Phase 4A) to reduce God Object attributes.
First N ticks per asset are flagged as warmup (proof log prefix, cold-start guards).
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class WarmupTracker:
    """Track per-asset warmup ticks. First `threshold` ticks are warmup."""

    def __init__(self, threshold: int = 2):
        self._ticks: Dict[str, int] = {}
        self._threshold = threshold

    def record_tick(self, asset: str) -> bool:
        """Increment tick count for asset. Returns True if still in warmup."""
        self._ticks[asset] = self._ticks.get(asset, 0) + 1
        return self._ticks[asset] <= self._threshold

    def get_tick_count(self, asset: str) -> int:
        """Return current tick count for asset."""
        return self._ticks.get(asset, 0)

    @property
    def threshold(self) -> int:
        return self._threshold

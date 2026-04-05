"""
================================================================================
HMATS v6 - RL Execution State Builder
================================================================================
Builds observation vectors from orderbook snapshots for RL execution agents.

Pipeline position: execution layer (pre-decision state encoding).
Does NOT place orders - purely transforms orderbook -> feature vector.
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RLStateConfig:
    """Configuration for RL execution state builder."""

    # Number of orderbook levels to encode
    n_levels: int = 5

    # Features per level: spread_bps, imbalance, cumulative_depth
    features_per_level: int = 3

    # Additional global features: mid_price, total_bid_depth, total_ask_depth,
    # depth_pressure, volatility
    n_global_features: int = 5

    # Whether to include extended features (level-wise cumulative depth)
    use_extended: bool = True

    @property
    def state_dim(self) -> int:
        """Total observation vector dimension."""
        if self.use_extended:
            return self.n_levels * self.features_per_level + self.n_global_features
        return self.n_levels * 2 + self.n_global_features  # spread + imbalance only


# =============================================================================
# STATE BUILDER
# =============================================================================

class RLExecutionStateBuilder:
    """
    Converts raw orderbook snapshots into fixed-size observation vectors
    suitable for RL execution agents.

    Thread-safe: no mutable shared state between calls.
    """

    def __init__(self, config: Optional[RLStateConfig] = None):
        self.config = config or RLStateConfig()

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    def build_state(
        self,
        bid_prices: List[float],
        bid_quantities: List[float],
        ask_prices: List[float],
        ask_quantities: List[float],
        volatility: float = 0.0,
    ) -> np.ndarray:
        """
        Build a fixed-size state vector from orderbook data.

        Returns:
            np.ndarray of shape (state_dim,) with values clipped to [-10, 10].
        """
        n = self.config.n_levels
        features = []

        # Pad to n_levels if needed
        bp = list(bid_prices[:n]) + [0.0] * max(0, n - len(bid_prices))
        bq = list(bid_quantities[:n]) + [0.0] * max(0, n - len(bid_quantities))
        ap = list(ask_prices[:n]) + [0.0] * max(0, n - len(ask_prices))
        aq = list(ask_quantities[:n]) + [0.0] * max(0, n - len(ask_quantities))

        mid = (bp[0] + ap[0]) / 2.0 if bp[0] > 0 and ap[0] > 0 else 0.0

        # Per-level features
        for i in range(n):
            # Spread in bps at level i
            if mid > 0 and ap[i] > 0 and bp[i] > 0:
                spread_bps = (ap[i] - bp[i]) / mid * 10000.0
            else:
                spread_bps = 0.0
            features.append(spread_bps)

            # Imbalance at level i
            total = bq[i] + aq[i]
            imbalance = (bq[i] - aq[i]) / total if total > 0 else 0.0
            features.append(imbalance)

            if self.config.use_extended:
                # Cumulative depth ratio (bid/ask) up to level i
                cum_bid = sum(bq[: i + 1])
                cum_ask = sum(aq[: i + 1])
                cum_total = cum_bid + cum_ask
                cum_ratio = (cum_bid - cum_ask) / cum_total if cum_total > 0 else 0.0
                features.append(cum_ratio)

        # Global features
        total_bid = sum(bq)
        total_ask = sum(aq)
        depth_total = total_bid + total_ask
        depth_pressure = (total_bid - total_ask) / depth_total if depth_total > 0 else 0.0

        features.append(mid / 1000.0 if mid > 0 else 0.0)  # normalized mid
        features.append(total_bid / 1e6)  # bid depth in M
        features.append(total_ask / 1e6)  # ask depth in M
        features.append(depth_pressure)
        features.append(volatility)

        state = np.array(features, dtype=np.float32)
        return np.clip(state, -10.0, 10.0)

    def process_orderbook(self, orderbook: Dict[str, Any]) -> np.ndarray:
        """
        Process a raw orderbook dict into a state vector.

        Expects keys: bid_prices, bid_quantities, ask_prices, ask_quantities.
        """
        return self.build_state(
            bid_prices=orderbook.get("bid_prices", []),
            bid_quantities=orderbook.get("bid_quantities", []),
            ask_prices=orderbook.get("ask_prices", []),
            ask_quantities=orderbook.get("ask_quantities", []),
            volatility=orderbook.get("volatility", 0.0),
        )

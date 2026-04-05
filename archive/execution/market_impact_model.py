"""
================================================================================
HMATS v6 - Enhanced Market Impact Model with Impact Curves
================================================================================
Extends the base market_impact.py with:
  - Empirical impact curve construction from recorded trades
  - Curve-adjusted impact estimation (blends model + historical)
  - Enhanced execution plan generation with order-type recommendations

Pipeline position: execution layer - called by ExecutionManager to size/split
orders.  Does NOT place orders directly.

Backward compatibility: ``MarketImpactModel`` alias wraps the enhanced model
with sensible defaults.
================================================================================
"""

import logging
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OrderbookLevel:
    """Single price level in an orderbook."""
    price: float
    quantity: float


@dataclass
class OrderbookSnapshot:
    """Multi-level orderbook snapshot."""
    timestamp: datetime
    symbol: str
    bids: List[OrderbookLevel] = field(default_factory=list)
    asks: List[OrderbookLevel] = field(default_factory=list)

    # Derived
    mid_price: float = 0.0
    spread_bps: float = 0.0
    total_bid_depth: float = 0.0
    total_ask_depth: float = 0.0

    def calculate_metrics(self) -> None:
        """Compute derived metrics in-place."""
        if self.bids and self.asks:
            self.mid_price = (self.bids[0].price + self.asks[0].price) / 2.0
            self.spread_bps = (
                (self.asks[0].price - self.bids[0].price) / self.mid_price * 10000.0
            )
        self.total_bid_depth = sum(l.quantity for l in self.bids)
        self.total_ask_depth = sum(l.quantity for l in self.asks)


@dataclass
class ImpactResult:
    """Result of an impact estimation."""
    total_impact_bps: float
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    model_used: str = "sqrt"
    should_split: bool = False
    recommended_slices: int = 1
    recommended_limit_offset_bps: float = 0.0
    recommended_depth_levels: int = 1
    curve_adjusted: bool = False
    curve_confidence: float = 0.0


@dataclass
class ImpactCurve:
    """Empirical impact curve built from recorded trades."""
    symbol: str
    total_samples: int = 0
    avg_impact_bps: float = 0.0
    model_rmse: float = 0.0
    points: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ExecutionSlice:
    """A single slice in an execution plan."""
    slice_idx: int
    quantity: float
    expected_impact_bps: float
    recommended_order_type: str = "limit"
    limit_offset_bps: float = 0.0
    delay_seconds: float = 0.0


@dataclass
class ExecutionPlan:
    """Full execution plan for a large order."""
    order_id: str
    symbol: str
    side: str
    total_quantity: float
    slices: List[ExecutionSlice] = field(default_factory=list)
    recommended_strategy: str = "twap"
    expected_vwap: float = 0.0
    total_expected_impact_bps: float = 0.0


# =============================================================================
# IMPACT RECORD (for curve building)
# =============================================================================

@dataclass
class _ImpactRecord:
    """Internal record of an actual trade impact observation."""
    timestamp: datetime
    side: str
    order_size: float
    daily_volume: float
    actual_impact_bps: float
    volatility: float
    levels_consumed: int
    fill_rate: float
    execution_time_ms: float
    participation_rate: float = 0.0


# =============================================================================
# ENHANCED MARKET IMPACT MODEL
# =============================================================================

class EnhancedMarketImpactModel:
    """
    Market impact model with empirical curve learning.

    Blends Almgren-Chriss square-root model with an empirical impact curve
    built from recorded trade observations.

    Args:
        use_impact_curve: Enable curve-adjusted estimation.
        curve_weight: Weight given to curve estimate (0-1).  Rest goes to model.
        eta: Temporary impact coefficient.
        gamma: Permanent impact coefficient.
        max_history: Max impact observations to retain per symbol.
    """

    def __init__(
        self,
        use_impact_curve: bool = True,
        curve_weight: float = 0.3,
        eta: float = 0.0001,
        gamma: float = 0.0001,
        max_history: int = 1000,
    ):
        self._use_curve = use_impact_curve
        self._curve_weight = min(max(curve_weight, 0.0), 1.0)
        self._eta = eta
        self._gamma = gamma
        self._max_history = max_history

        # Per-symbol state
        self._orderbooks: Dict[str, OrderbookSnapshot] = {}
        self._history: Dict[str, deque] = {}
        self._curves: Dict[str, ImpactCurve] = {}

    # -----------------------------------------------------------------
    # ORDERBOOK
    # -----------------------------------------------------------------

    def update_orderbook(self, symbol: str, ob: OrderbookSnapshot) -> None:
        """Store latest orderbook for a symbol."""
        self._orderbooks[symbol] = ob

    # -----------------------------------------------------------------
    # IMPACT ESTIMATION
    # -----------------------------------------------------------------

    def estimate_impact(
        self,
        symbol: str,
        order_quantity: float,
        daily_volume: float,
        volatility: float,
        side: str,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> ImpactResult:
        """
        Estimate total market impact for an order.

        Blends model-based and curve-based estimates when curve data exists.
        """
        ob = orderbook or self._orderbooks.get(symbol)
        participation = order_quantity / daily_volume if daily_volume > 0 else 0.0

        # --- Square-root model ---
        temp = self._eta * volatility * math.sqrt(participation) * 10000.0
        perm = self._gamma * volatility * math.sqrt(participation) * 10000.0

        spread_cost = 0.0
        if ob and ob.spread_bps > 0:
            spread_cost = ob.spread_bps * 0.5
        elif ob and ob.asks and ob.bids:
            mid = (ob.asks[0].price + ob.bids[0].price) / 2.0
            spread_cost = (ob.asks[0].price - ob.bids[0].price) / mid * 5000.0

        model_impact = temp + perm + spread_cost

        # --- Curve adjustment ---
        curve_adjusted = False
        curve_confidence = 0.0
        curve = self._curves.get(symbol)

        if self._use_curve and curve and curve.total_samples >= 20:
            # Interpolate curve
            curve_est = self._interpolate_curve(curve, participation)
            if curve_est is not None:
                conf = min(curve.total_samples / 100.0, 1.0)
                w = self._curve_weight * conf
                model_impact = model_impact * (1.0 - w) + curve_est * w
                curve_adjusted = True
                curve_confidence = round(conf, 3)

        # --- Recommendations ---
        levels_needed = 1
        if ob:
            levels = ob.asks if side == "buy" else ob.bids
            remaining = order_quantity
            for i, lvl in enumerate(levels):
                remaining -= lvl.quantity
                levels_needed = i + 1
                if remaining <= 0:
                    break

        should_split = order_quantity > daily_volume * 0.01 or levels_needed > 2
        offset = max(spread_cost * 0.8, 1.0)

        return ImpactResult(
            total_impact_bps=round(max(model_impact, 0.01), 4),
            temporary_impact_bps=round(temp, 4),
            permanent_impact_bps=round(perm, 4),
            spread_cost_bps=round(spread_cost, 4),
            model_used="sqrt+curve" if curve_adjusted else "sqrt",
            should_split=should_split,
            recommended_slices=max(1, levels_needed),
            recommended_limit_offset_bps=round(offset, 1),
            recommended_depth_levels=levels_needed,
            curve_adjusted=curve_adjusted,
            curve_confidence=curve_confidence,
        )

    # -----------------------------------------------------------------
    # IMPACT RECORDING & CURVE BUILDING
    # -----------------------------------------------------------------

    def record_actual_impact(
        self,
        symbol: str,
        side: str,
        order_size: float,
        daily_volume: float,
        actual_impact_bps: float,
        volatility: float = 0.0,
        orderbook: Optional[OrderbookSnapshot] = None,
        levels_consumed: int = 1,
        fill_rate: float = 1.0,
        execution_time_ms: float = 0.0,
    ) -> None:
        """Record an observed trade impact for curve building."""
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._max_history)

        participation = order_size / daily_volume if daily_volume > 0 else 0.0

        self._history[symbol].append(
            _ImpactRecord(
                timestamp=datetime.utcnow(),
                side=side,
                order_size=order_size,
                daily_volume=daily_volume,
                actual_impact_bps=actual_impact_bps,
                volatility=volatility,
                levels_consumed=levels_consumed,
                fill_rate=fill_rate,
                execution_time_ms=execution_time_ms,
                participation_rate=participation,
            )
        )

        # Rebuild curve periodically
        if len(self._history[symbol]) % 10 == 0:
            self._build_curve(symbol)

    def _build_curve(self, symbol: str) -> None:
        """Build/rebuild empirical impact curve from recorded observations."""
        records = list(self._history.get(symbol, []))
        if len(records) < 20:
            return

        impacts = np.array([r.actual_impact_bps for r in records])
        participations = np.array([r.participation_rate for r in records])

        # Sort by participation rate and bin
        order = np.argsort(participations)
        impacts = impacts[order]
        participations = participations[order]

        n_bins = min(10, len(records) // 5)
        points = []
        for i in range(n_bins):
            start = i * len(records) // n_bins
            end = (i + 1) * len(records) // n_bins
            if end <= start:
                continue
            points.append(
                {
                    "participation_rate": float(np.mean(participations[start:end])),
                    "avg_impact_bps": float(np.mean(impacts[start:end])),
                    "std_impact_bps": float(np.std(impacts[start:end])),
                    "count": end - start,
                }
            )

        # Simple RMSE of sqrt model vs actual
        predicted = self._eta * np.sqrt(participations) * 10000.0
        rmse = float(np.sqrt(np.mean((predicted - impacts) ** 2)))

        self._curves[symbol] = ImpactCurve(
            symbol=symbol,
            total_samples=len(records),
            avg_impact_bps=float(np.mean(impacts)),
            model_rmse=rmse,
            points=points,
        )

    def _interpolate_curve(self, curve: ImpactCurve, participation: float) -> Optional[float]:
        """Linear interpolation on the empirical curve."""
        if not curve.points:
            return None

        pts = curve.points
        # Below range -> extrapolate from first point
        if participation <= pts[0]["participation_rate"]:
            return pts[0]["avg_impact_bps"]
        # Above range -> extrapolate from last point
        if participation >= pts[-1]["participation_rate"]:
            return pts[-1]["avg_impact_bps"]

        # Interpolate
        for i in range(len(pts) - 1):
            x0 = pts[i]["participation_rate"]
            x1 = pts[i + 1]["participation_rate"]
            if x0 <= participation <= x1:
                t = (participation - x0) / (x1 - x0) if x1 > x0 else 0.5
                y0 = pts[i]["avg_impact_bps"]
                y1 = pts[i + 1]["avg_impact_bps"]
                return y0 + t * (y1 - y0)

        return curve.avg_impact_bps

    def get_impact_curve(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the impact curve for a symbol as a dict, or None."""
        curve = self._curves.get(symbol)
        if curve is None:
            return None
        return {
            "symbol": curve.symbol,
            "total_samples": curve.total_samples,
            "avg_impact_bps": curve.avg_impact_bps,
            "model_rmse": curve.model_rmse,
            "points": curve.points,
        }

    # -----------------------------------------------------------------
    # EXECUTION PLAN
    # -----------------------------------------------------------------

    def generate_execution_plan(
        self,
        order_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        daily_volume: float,
        volatility: float,
        current_price: float,
        urgency: float = 0.5,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> ExecutionPlan:
        """
        Generate a multi-slice execution plan.

        Args:
            order_id: Unique order identifier.
            symbol: Trading pair.
            side: "buy" or "sell".
            total_quantity: Total order size in base units.
            daily_volume: Estimated daily volume.
            volatility: Current volatility.
            current_price: Current mid/last price.
            urgency: 0 (patient) to 1 (immediate).
            orderbook: Optional current orderbook.

        Returns:
            ExecutionPlan with slices and strategy recommendation.
        """
        participation = total_quantity / daily_volume if daily_volume > 0 else 0.0

        # Strategy selection
        if urgency > 0.8 or participation < 0.001:
            strategy = "aggressive"
            n_slices = 1
        elif urgency > 0.5:
            strategy = "adaptive"
            n_slices = max(2, min(5, int(participation * 200)))
        elif volatility > 0.5:
            strategy = "vwap"
            n_slices = max(3, min(8, int(participation * 300)))
        else:
            strategy = "twap"
            n_slices = max(2, min(6, int(participation * 250)))

        slice_qty = total_quantity / n_slices
        slices = []
        total_impact = 0.0

        for i in range(n_slices):
            impact = self.estimate_impact(
                symbol=symbol,
                order_quantity=slice_qty,
                daily_volume=daily_volume,
                volatility=volatility,
                side=side,
                orderbook=orderbook,
            )
            order_type = "limit" if urgency < 0.7 else "market"
            offset = impact.recommended_limit_offset_bps if order_type == "limit" else 0.0

            slices.append(
                ExecutionSlice(
                    slice_idx=i,
                    quantity=slice_qty,
                    expected_impact_bps=impact.total_impact_bps,
                    recommended_order_type=order_type,
                    limit_offset_bps=round(offset, 1),
                    delay_seconds=i * 30.0 if strategy in ("twap", "vwap") else 0.0,
                )
            )
            total_impact += impact.total_impact_bps

        avg_impact = total_impact / n_slices if n_slices > 0 else 0.0
        expected_vwap = current_price * (1.0 + avg_impact / 10000.0) if side == "buy" else current_price * (1.0 - avg_impact / 10000.0)

        return ExecutionPlan(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            slices=slices,
            recommended_strategy=strategy,
            expected_vwap=round(expected_vwap, 6),
            total_expected_impact_bps=round(avg_impact, 4),
        )


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

class MarketImpactModel(EnhancedMarketImpactModel):
    """Legacy alias - wraps EnhancedMarketImpactModel with defaults."""

    def __init__(self, **kwargs):
        super().__init__(use_impact_curve=False, **kwargs)

    def estimate_impact(self, symbol: str, order_quantity: float,
                        daily_volume: float, volatility: float, side: str,
                        orderbook: Optional[OrderbookSnapshot] = None,
                        **kwargs) -> ImpactResult:
        result = super().estimate_impact(
            symbol=symbol, order_quantity=order_quantity,
            daily_volume=daily_volume, volatility=volatility,
            side=side, orderbook=orderbook,
        )
        return result

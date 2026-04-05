"""
G6: Portfolio Brain Offensive - Alpha-weighted dynamic allocation.

Input: Per-asset rolling PnL history (from 4H bar results)
Output: Asset-level weights Dict[str, float] (BTC/ETH/SOL)

Algorithm:
  1. Compute per-asset rolling Sortino (past 42 bars ~ 7 days)
  2. IR-Softmax normalization (tau=0.5, moderate concentration)
  3. Weight decay: new = 0.7 * computed + 0.3 * previous
  4. Safety constraint: per-asset <= existing Cross-Asset cap
  5. Minimum floor: each asset >= 10% (avoid total neglect)

Notes:
  - ASSET-level weights, not strategy-level (3 assets only)
  - Output consumed by sizing path: multiplied against base exposure
  - Cross-Asset cap remains hard upper limit; PortfolioBrain cannot exceed it
"""
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration constants
LOOKBACK_BARS = 42          # 7 days x 6 bars/day
MIN_BARS_FOR_CALC = 12      # At least 2 days before computing
TAU = 0.5                   # Softmax temperature: lower = more concentrated
DECAY = 0.3                 # 30% from previous period
MIN_WEIGHT = 0.10           # 10% floor per asset
REBALANCE_INTERVAL = 6      # Every 6 bars (= 24H) recompute

# Defensive caps (match main.py exposure_caps)
DEFENSIVE_CAPS = {
    "BTC": 0.60,
    "ETH": 0.50,
    "SOL": 0.95,
}


@dataclass
class AssetPerformance:
    """Rolling performance tracker for a single asset."""
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=LOOKBACK_BARS))
    trade_count: int = 0
    last_update: float = 0.0

    def record_pnl(self, pnl_pct: float):
        self.pnl_history.append(pnl_pct)
        self.trade_count += 1
        self.last_update = time.time()

    def rolling_sortino(self) -> float:
        if len(self.pnl_history) < MIN_BARS_FOR_CALC:
            return 0.0
        returns = list(self.pnl_history)
        mean_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            # All positive: use mean-proportional score to preserve ordering
            # +0.3%/bar -> 0.6, +2%/bar -> 4.0
            return min(10.0, max(0.0, mean_r * 200))
        downside_std = (sum(d ** 2 for d in downside) / len(downside)) ** 0.5
        if downside_std < 1e-8:
            return min(10.0, max(0.0, mean_r * 200))
        sortino = mean_r / downside_std
        return max(-10.0, min(10.0, sortino))


@dataclass
class AssetLiveContext:
    expected_net_alpha_bps: Optional[float] = None
    toxicity_score: Optional[float] = None
    side_avg_realized_pnl_bps: Optional[float] = None
    side_total_realized_pnl_usd: Optional[float] = None
    crowding_alignment_score: Optional[float] = None
    panic_score: Optional[float] = None
    extreme_crowding: Optional[bool] = None
    exit_avg_retention_pct: Optional[float] = None
    exit_avg_peak_to_exit_bps: Optional[float] = None
    updated_at: float = 0.0


class PortfolioBrainOffensive:
    """
    Alpha-weighted dynamic asset allocation.

    Usage:
        brain = PortfolioBrainOffensive(assets=["BTC", "ETH", "SOL"])

        # After each 4H bar:
        brain.record_bar_pnl("BTC", pnl_pct=0.012)
        brain.record_bar_pnl("ETH", pnl_pct=-0.005)
        brain.record_bar_pnl("SOL", pnl_pct=0.031)

        # Get weights:
        weights = brain.get_weights()
        # -> {"BTC": 0.35, "ETH": 0.15, "SOL": 0.50}
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        tau: float = TAU,
        decay: float = DECAY,
        min_weight: float = MIN_WEIGHT,
        rebalance_interval: int = REBALANCE_INTERVAL,
        defensive_caps: Optional[Dict[str, float]] = None,
        shadow_mode: bool = True,
        cold_start_prior_enabled: bool = True,
        cold_start_prior_floor: float = MIN_WEIGHT,
        cold_start_prior_max_multiplier: float = 1.40,
    ):
        self.assets = assets or ["BTC", "ETH", "SOL"]
        self.tau = tau
        self.decay = decay
        self.min_weight = min_weight
        self.rebalance_interval = rebalance_interval
        self.defensive_caps = defensive_caps or DEFENSIVE_CAPS
        self.shadow_mode = shadow_mode
        self.cold_start_prior_enabled = bool(cold_start_prior_enabled)
        self.cold_start_prior_floor = max(0.0, float(cold_start_prior_floor))
        self.cold_start_prior_max_multiplier = max(1.0, float(cold_start_prior_max_multiplier))

        # Per-asset performance tracking
        self.perf: Dict[str, AssetPerformance] = {
            a: AssetPerformance() for a in self.assets
        }

        # Weight state
        n = len(self.assets)
        equal = 1.0 / n
        self.current_weights: Dict[str, float] = {a: equal for a in self.assets}
        self.previous_weights: Dict[str, float] = {a: equal for a in self.assets}
        self.live_context: Dict[str, AssetLiveContext] = {
            a: AssetLiveContext() for a in self.assets
        }

        # Rebalance counter
        self._bar_count = 0
        self._last_rebalance = 0
        self._cold_start_prior_active = False

        logger.info(
            f"[G6] PortfolioBrainOffensive init: assets={self.assets}, "
            f"tau={tau}, decay={decay}, shadow={shadow_mode}, "
            f"cold_start_prior={self.cold_start_prior_enabled}"
        )

    # --- Data input ---

    def record_bar_pnl(self, asset: str, pnl_pct: float):
        """Record asset bar PnL (percentage, e.g. 0.012 = +1.2%)."""
        if asset not in self.perf:
            return
        self.perf[asset].record_pnl(pnl_pct)

    def on_bar_close(self):
        """Called after each 4H bar. Checks if rebalance needed."""
        self._bar_count += 1
        if self._bar_count - self._last_rebalance >= self.rebalance_interval:
            self._rebalance()
            self._last_rebalance = self._bar_count

    # --- Weight output ---

    def get_weights(self) -> Dict[str, float]:
        """Get current asset weights. Shadow mode returns equal weights."""
        if self.shadow_mode:
            n = len(self.assets)
            return {a: 1.0 / n for a in self.assets}
        return self._apply_live_context_overlay(self.current_weights)

    def get_shadow_weights(self) -> Dict[str, float]:
        """Get computed weights regardless of shadow mode. For logging."""
        return self._apply_live_context_overlay(self.current_weights)

    def update_live_context(
        self,
        asset: str,
        *,
        expected_net_alpha_bps: Optional[float] = None,
        toxicity_score: Optional[float] = None,
        side_avg_realized_pnl_bps: Optional[float] = None,
        side_total_realized_pnl_usd: Optional[float] = None,
        crowding_alignment_score: Optional[float] = None,
        panic_score: Optional[float] = None,
        extreme_crowding: Optional[bool] = None,
        exit_avg_retention_pct: Optional[float] = None,
        exit_avg_peak_to_exit_bps: Optional[float] = None,
    ) -> None:
        asset_key = str(asset or "").upper().replace("/USD", "").replace("USD", "")
        if asset_key not in self.live_context:
            return
        self.live_context[asset_key] = AssetLiveContext(
            expected_net_alpha_bps=(
                float(expected_net_alpha_bps or 0.0) if expected_net_alpha_bps is not None else None
            ),
            toxicity_score=(
                float(toxicity_score or 0.0) if toxicity_score is not None else None
            ),
            side_avg_realized_pnl_bps=(
                float(side_avg_realized_pnl_bps or 0.0) if side_avg_realized_pnl_bps is not None else None
            ),
            side_total_realized_pnl_usd=(
                float(side_total_realized_pnl_usd or 0.0) if side_total_realized_pnl_usd is not None else None
            ),
            crowding_alignment_score=(
                float(crowding_alignment_score or 0.0)
                if crowding_alignment_score is not None
                else None
            ),
            panic_score=(
                float(panic_score or 0.0) if panic_score is not None else None
            ),
            extreme_crowding=(
                bool(extreme_crowding) if extreme_crowding is not None else None
            ),
            exit_avg_retention_pct=(
                float(exit_avg_retention_pct or 0.0)
                if exit_avg_retention_pct is not None
                else None
            ),
            exit_avg_peak_to_exit_bps=(
                float(exit_avg_peak_to_exit_bps or 0.0)
                if exit_avg_peak_to_exit_bps is not None
                else None
            ),
            updated_at=time.time(),
        )

    # --- Core algorithm ---

    def _rebalance(self):
        """Rebalance: Sortino -> IR-Softmax -> Decay -> Cap."""
        # Step 1: Compute per-asset Sortino
        sortinos = {}
        for asset in self.assets:
            sortinos[asset] = self.perf[asset].rolling_sortino()

        # Step 2: Cold-start check
        has_data = any(
            len(self.perf[a].pnl_history) >= MIN_BARS_FOR_CALC
            for a in self.assets
        )
        if not has_data:
            cold_start_prior = self._compute_cold_start_prior()
            if cold_start_prior is not None:
                self.previous_weights = dict(self.current_weights)
                self.current_weights = cold_start_prior
                self._cold_start_prior_active = True
                mode = "SHADOW" if self.shadow_mode else "LIVE"
                logger.info(
                    f"[G6] Cold start prior [{mode}] "
                    f"Weights={{{', '.join(f'{a}:{w:.1%}' for a, w in cold_start_prior.items())}}}"
                )
            else:
                self._cold_start_prior_active = False
                logger.debug("[G6] Cold start - keeping equal weights")
            return
        self._cold_start_prior_active = False

        # Step 3: IR-Softmax
        raw_weights = self._ir_softmax(sortinos)

        # Step 4: Weight decay
        decayed = {}
        for a in self.assets:
            decayed[a] = (1 - self.decay) * raw_weights[a] + self.decay * self.previous_weights[a]

        # Step 5: Minimum floor
        for a in self.assets:
            decayed[a] = max(decayed[a], self.min_weight)

        # Step 6: Normalize
        total = sum(decayed.values())
        if total > 0:
            for a in self.assets:
                decayed[a] /= total

        # Step 7: Defensive cap
        capped = self._apply_caps(decayed)

        # Step 8: Update state
        self.previous_weights = dict(self.current_weights)
        self.current_weights = capped

        # Step 9: Log
        mode = "SHADOW" if self.shadow_mode else "LIVE"
        logger.info(
            f"[G6] Rebalance [{mode}] "
            f"Sortinos={{{', '.join(f'{a}:{s:.2f}' for a, s in sortinos.items())}}} "
            f"-> Weights={{{', '.join(f'{a}:{w:.1%}' for a, w in capped.items())}}}"
        )

    def _ir_softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """IR-Softmax: exp(score/tau) normalized."""
        max_score = max(scores.values())
        exps = {}
        for a, s in scores.items():
            exps[a] = math.exp((s - max_score) / self.tau)
        total = sum(exps.values())
        if total < 1e-10:
            n = len(self.assets)
            return {a: 1.0 / n for a in self.assets}
        return {a: v / total for a, v in exps.items()}

    def _apply_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply defensive caps, redistribute overflow to uncapped assets."""
        result = dict(weights)
        overflow = 0.0
        uncapped = []

        for a in self.assets:
            cap = self.defensive_caps.get(a, 1.0)
            if result[a] > cap:
                overflow += result[a] - cap
                result[a] = cap
            else:
                uncapped.append(a)

        if overflow > 0 and uncapped:
            share = overflow / len(uncapped)
            for a in uncapped:
                result[a] += share
                cap = self.defensive_caps.get(a, 1.0)
                result[a] = min(result[a], cap)

        # Final normalize (caps may cause sum != 1)
        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            for a in self.assets:
                result[a] /= total

        return result

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _context_multiplier(self, asset: str) -> float:
        ctx = self.live_context.get(asset)
        if ctx is None:
            return 1.0

        alpha_factor = 1.0
        if ctx.expected_net_alpha_bps is not None:
            alpha_factor = self._clip(0.85 + min(max(ctx.expected_net_alpha_bps, 0.0), 20.0) / 66.6667, 0.85, 1.15)

        realized_factor = 1.0
        if ctx.side_avg_realized_pnl_bps is not None or ctx.side_total_realized_pnl_usd is not None:
            avg_norm = self._clip(float(ctx.side_avg_realized_pnl_bps or 0.0) / 25.0, -1.0, 1.0)
            total_norm = self._clip(float(ctx.side_total_realized_pnl_usd or 0.0) / 100.0, -1.0, 1.0)
            realized_factor = self._clip(1.0 + 0.08 * avg_norm + 0.04 * total_norm, 0.85, 1.12)

        toxicity_factor = 1.0
        if ctx.toxicity_score is not None:
            toxicity_factor = self._clip(
                1.05 - 0.30 * self._clip(float(ctx.toxicity_score or 0.0), 0.0, 1.0),
                0.75,
                1.05,
            )

        crowding_factor = 1.0
        if ctx.crowding_alignment_score is not None:
            align = self._clip(float(ctx.crowding_alignment_score or 0.0), -1.0, 1.0)
            crowding_factor = self._clip(1.0 + 0.08 * align, 0.88, 1.10)
            if bool(ctx.extreme_crowding):
                crowding_factor = self._clip(
                    crowding_factor + (0.03 if align > 0 else -0.05),
                    0.82,
                    1.12,
                )

        panic_factor = 1.0
        if ctx.panic_score is not None:
            panic_factor = self._clip(
                1.02 - 0.10 * self._clip(float(ctx.panic_score or 0.0), 0.0, 1.0),
                0.90,
                1.02,
            )

        exit_factor = 1.0
        if (
            ctx.exit_avg_retention_pct is not None
            or ctx.exit_avg_peak_to_exit_bps is not None
        ):
            retention_norm = self._clip(
                ((float(ctx.exit_avg_retention_pct or 0.0) - 50.0) / 50.0),
                -1.0,
                1.0,
            )
            giveback_norm = self._clip(
                float(ctx.exit_avg_peak_to_exit_bps or 0.0) / 100.0,
                0.0,
                1.0,
            )
            exit_factor = self._clip(
                1.0 + 0.10 * retention_norm - 0.08 * giveback_norm,
                0.88,
                1.12,
            )

        return self._clip(
            alpha_factor
            * realized_factor
            * toxicity_factor
            * crowding_factor
            * panic_factor
            * exit_factor,
            0.70,
            1.30,
        )

    def _apply_live_context_overlay(self, weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return {}

        adjusted = {}
        for asset in self.assets:
            base_weight = float(weights.get(asset, 0.0) or 0.0)
            adjusted[asset] = base_weight * self._context_multiplier(asset)

        total = sum(adjusted.values())
        if total <= 0:
            return dict(weights)
        return {asset: adjusted[asset] / total for asset in self.assets}

    def _compute_cold_start_prior(self) -> Optional[Dict[str, float]]:
        if not self.cold_start_prior_enabled:
            return None

        equal = 1.0 / max(len(self.assets), 1)
        raw = {}
        has_signal = False
        for asset in self.assets:
            multiplier = self._clip(
                self._context_multiplier(asset),
                self.cold_start_prior_floor,
                self.cold_start_prior_max_multiplier,
            )
            ctx = self.live_context.get(asset)
            if ctx and any(
                value is not None and abs(float(value or 0.0)) > 1e-9
                for value in (
                    ctx.expected_net_alpha_bps,
                    ctx.toxicity_score,
                    ctx.side_avg_realized_pnl_bps,
                    ctx.side_total_realized_pnl_usd,
                    ctx.crowding_alignment_score,
                    ctx.panic_score,
                    ctx.exit_avg_retention_pct,
                    ctx.exit_avg_peak_to_exit_bps,
                )
            ):
                has_signal = True
            raw[asset] = equal * multiplier

        if not has_signal:
            return None

        for asset in self.assets:
            raw[asset] = max(raw[asset], self.cold_start_prior_floor)

        total = sum(raw.values())
        if total <= 0:
            return None
        normalized = {asset: raw[asset] / total for asset in self.assets}
        return self._apply_caps(normalized)

    # --- Promotion control ---

    def promote_to_live(self):
        self.shadow_mode = False
        logger.warning("[G6] PortfolioBrain promoted to LIVE mode")

    def demote_to_shadow(self):
        self.shadow_mode = True
        n = len(self.assets)
        self.current_weights = {a: 1.0 / n for a in self.assets}
        logger.warning("[G6] PortfolioBrain demoted to SHADOW mode")

    # --- State snapshot ---

    def snapshot(self) -> Dict:
        effective_weights = self._apply_live_context_overlay(self.current_weights)
        return {
            "mode": "SHADOW" if self.shadow_mode else "LIVE",
            "weights": dict(effective_weights),
            "raw_weights": dict(self.current_weights),
            "context_multipliers": {
                a: round(self._context_multiplier(a), 4) for a in self.assets
            },
            "cold_start_prior_active": bool(self._cold_start_prior_active),
            "sortinos": {
                a: round(self.perf[a].rolling_sortino(), 3) for a in self.assets
            },
            "bar_counts": {
                a: len(self.perf[a].pnl_history) for a in self.assets
            },
            "total_bars": self._bar_count,
            "rebalances": self._bar_count // max(self.rebalance_interval, 1),
        }

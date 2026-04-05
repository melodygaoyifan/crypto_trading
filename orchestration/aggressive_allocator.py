"""
orchestration/aggressive_allocator.py - P4-01 AggressiveAllocation

Dynamic concentration allocation for ULTRA_AGGRESSIVE_5Y profile.
Ranks assets by signal quality and concentrates exposure on the strongest.

Score formula (interpretable):
    alpha_bps = abs(quant_direction) * 200
    score     = (alpha_bps / 200) * quant_confidence * regime_confidence
              = abs(quant_direction) * quant_confidence * regime_confidence

Allocation tiers (configurable):
    STRONG  top_score > 0.80 -> [1.0, 0.0, 0.0]
    MID     top_score > 0.65 -> [0.7, 0.3, 0.0]
    BASE    else             -> [0.5, 0.3, 0.2]

Cap-aware normalization prevents churn by pre-clamping to exposure_caps.
Anti-churn guard: effective_cap = max(alloc_cap, current_exposure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AggressiveAllocationConfig:
    """Configuration for aggressive allocation. Disabled by default."""
    enabled: bool = False
    strong_threshold: float = 0.80
    mid_threshold: float = 0.65
    strong_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0]
    )
    mid_weights: List[float] = field(
        default_factory=lambda: [0.7, 0.3, 0.0]
    )
    base_weights: List[float] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]
    )
    # [S10] Correlation-aware allocation guard
    correlation_merge_threshold: float = 0.85   # above -> keep only higher-score asset
    correlation_reduce_threshold: float = 0.70   # above -> second asset allocation *= reduce_factor
    correlation_reduce_factor: float = 0.50      # discount for second asset

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> AggressiveAllocationConfig:
        if not d:
            return cls()
        return cls(
            enabled=d.get("enabled", False),
            strong_threshold=d.get("strong_threshold", 0.80),
            mid_threshold=d.get("mid_threshold", 0.65),
            strong_weights=d.get("strong_weights", [1.0, 0.0, 0.0]),
            mid_weights=d.get("mid_weights", [0.7, 0.3, 0.0]),
            base_weights=d.get("base_weights", [0.5, 0.3, 0.2]),
            correlation_merge_threshold=d.get("correlation_merge_threshold", 0.85),
            correlation_reduce_threshold=d.get("correlation_reduce_threshold", 0.70),
            correlation_reduce_factor=d.get("correlation_reduce_factor", 0.50),
        )


# ---------------------------------------------------------------------------
# AggressiveAllocator
# ---------------------------------------------------------------------------

class AggressiveAllocator:
    """
    Ranks assets by signal quality and produces cap-aware allocation weights.

    Usage from main.py tick loop:
        # BEFORE engine.decide() - accumulate score
        allocator.update_score(asset, agent_signals)

        # AFTER engine.decide(), BEFORE soft cap - get allocation
        alloc_cap = allocator.get_allocation(
            asset, exposure_caps, current_exposures)
        portfolio_recommended_exposure = alloc_cap
    """

    def __init__(self, config: AggressiveAllocationConfig):
        self.config = config
        self._scores: Dict[str, float] = {}
        self._score_components: Dict[str, Dict[str, float]] = {}
        self._last_allocation: Optional[Dict] = None

    # -- Score computation ---------------------------------------------------

    @staticmethod
    def compute_score(
        quant_direction: float,
        quant_confidence: float,
        regime_confidence: float,
        *,
        expected_net_alpha_bps: Optional[float] = None,
        toxicity_score: Optional[float] = None,
        side_avg_realized_pnl_bps: Optional[float] = None,
        side_total_realized_pnl_usd: Optional[float] = None,
        crowding_alignment_score: Optional[float] = None,
        panic_score: Optional[float] = None,
        exit_avg_retention_pct: Optional[float] = None,
        exit_avg_peak_to_exit_bps: Optional[float] = None,
    ) -> float:
        """
        Compute interpretable allocation score.

        Components:
            alpha_bps = abs(quant_direction) * 200  (proxy for estimated alpha)
            score = (alpha_bps / 200) * quant_confidence * regime_confidence
                  = abs(quant_direction) * quant_confidence * regime_confidence

        Optional profitability modifiers preserve the old formula when omitted:
        - expected_net_alpha_bps: boosts strong net-edge setups, penalizes negative edge
        - side realized PnL: rewards sides with constructive realized evidence
        - toxicity_score: penalizes execution-hostile assets
        - crowding_alignment_score: boosts setups aligned with crowding, penalizes headwinds
        - panic_score: penalizes execution-hostile panic conditions
        - exit alpha: rewards assets where exits retain more alpha and give back less

        Range: [0, 1]
        """
        base_score = abs(float(quant_direction or 0.0)) * float(quant_confidence or 0.0) * float(regime_confidence or 0.0)

        net_alpha_factor = 1.0
        if expected_net_alpha_bps is not None:
            net_alpha_bps = float(expected_net_alpha_bps or 0.0)
            if net_alpha_bps <= 0:
                net_alpha_factor = 0.75
            else:
                net_alpha_factor = min(1.25, 0.85 + min(net_alpha_bps, 20.0) / 50.0)

        realized_factor = 1.0
        if side_avg_realized_pnl_bps is not None or side_total_realized_pnl_usd is not None:
            avg_norm = max(-1.0, min(1.0, float(side_avg_realized_pnl_bps or 0.0) / 25.0))
            total_norm = max(-1.0, min(1.0, float(side_total_realized_pnl_usd or 0.0) / 100.0))
            realized_factor = max(0.75, min(1.15, 1.0 + 0.10 * avg_norm + 0.05 * total_norm))

        toxicity_factor = 1.0
        if toxicity_score is not None:
            toxicity_factor = max(0.70, 1.0 - 0.35 * max(0.0, min(1.0, float(toxicity_score or 0.0))))

        crowding_factor = 1.0
        if crowding_alignment_score is not None:
            align = max(-1.0, min(1.0, float(crowding_alignment_score or 0.0)))
            crowding_factor = max(0.82, min(1.12, 1.0 + 0.10 * align))

        panic_factor = 1.0
        if panic_score is not None:
            panic_factor = max(0.88, min(1.02, 1.02 - 0.12 * max(0.0, min(1.0, float(panic_score or 0.0)))))

        exit_factor = 1.0
        if exit_avg_retention_pct is not None or exit_avg_peak_to_exit_bps is not None:
            retention_norm = max(
                -1.0,
                min(1.0, (float(exit_avg_retention_pct or 0.0) - 50.0) / 50.0),
            )
            giveback_norm = max(
                0.0,
                min(1.0, float(exit_avg_peak_to_exit_bps or 0.0) / 100.0),
            )
            exit_factor = max(0.85, min(1.15, 1.0 + 0.12 * retention_norm - 0.08 * giveback_norm))

        return max(
            0.0,
            min(
                1.0,
                base_score
                * net_alpha_factor
                * realized_factor
                * toxicity_factor
                * crowding_factor
                * panic_factor
                * exit_factor,
            ),
        )

    def update_score(self, asset: str, agent_signals: Dict) -> float:
        """Extract signals from agent_signals, compute score, store it."""
        qd = agent_signals.get("quant_direction", 0.0)
        qc = agent_signals.get("quant_confidence", 0.5)
        rc = agent_signals.get("regime_confidence", 0.5)
        expected_net_alpha_bps = agent_signals.get("aggressive_allocator_expected_net_alpha_bps")
        toxicity_score = agent_signals.get("aggressive_allocator_toxicity_score")
        side_avg_realized_pnl_bps = agent_signals.get("aggressive_allocator_side_avg_realized_pnl_bps")
        side_total_realized_pnl_usd = agent_signals.get("aggressive_allocator_side_total_realized_pnl_usd")
        crowding_alignment_score = agent_signals.get("aggressive_allocator_crowding_alignment_score")
        panic_score = agent_signals.get("aggressive_allocator_panic_score")
        exit_avg_retention_pct = agent_signals.get("aggressive_allocator_exit_avg_retention_pct")
        exit_avg_peak_to_exit_bps = agent_signals.get("aggressive_allocator_exit_avg_peak_to_exit_bps")
        score = self.compute_score(
            qd,
            qc,
            rc,
            expected_net_alpha_bps=expected_net_alpha_bps,
            toxicity_score=toxicity_score,
            side_avg_realized_pnl_bps=side_avg_realized_pnl_bps,
            side_total_realized_pnl_usd=side_total_realized_pnl_usd,
            crowding_alignment_score=crowding_alignment_score,
            panic_score=panic_score,
            exit_avg_retention_pct=exit_avg_retention_pct,
            exit_avg_peak_to_exit_bps=exit_avg_peak_to_exit_bps,
        )
        self._scores[asset] = score
        self._score_components[asset] = {
            "base_signal_score": abs(float(qd or 0.0)) * float(qc or 0.0) * float(rc or 0.0),
            "expected_net_alpha_bps": float(expected_net_alpha_bps or 0.0) if expected_net_alpha_bps is not None else 0.0,
            "toxicity_score": float(toxicity_score or 0.0) if toxicity_score is not None else 0.0,
            "side_avg_realized_pnl_bps": float(side_avg_realized_pnl_bps or 0.0) if side_avg_realized_pnl_bps is not None else 0.0,
            "side_total_realized_pnl_usd": float(side_total_realized_pnl_usd or 0.0) if side_total_realized_pnl_usd is not None else 0.0,
            "crowding_alignment_score": float(crowding_alignment_score or 0.0) if crowding_alignment_score is not None else 0.0,
            "panic_score": float(panic_score or 0.0) if panic_score is not None else 0.0,
            "exit_avg_retention_pct": float(exit_avg_retention_pct or 0.0) if exit_avg_retention_pct is not None else 0.0,
            "exit_avg_peak_to_exit_bps": float(exit_avg_peak_to_exit_bps or 0.0) if exit_avg_peak_to_exit_bps is not None else 0.0,
        }
        return score

    # -- Allocation ----------------------------------------------------------

    def get_allocation(
        self,
        asset: str,
        exposure_caps: Dict[str, float],
        current_exposures: Optional[Dict[str, float]] = None,
        correlations: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute allocation and return the effective exposure cap for `asset`.

        Returns the cap for the requested asset (float).
        Fail-open: if no scores, returns exposure_caps[asset].
        """
        if not self._scores:
            return exposure_caps.get(asset, 0.30)

        # Rank assets by score descending
        ranked = sorted(
            self._scores.keys(), key=lambda a: self._scores[a], reverse=True
        )

        top_score = self._scores[ranked[0]]

        # Select tier weights
        weights_template = self._select_tier(top_score)

        # Map weights to ranked assets
        raw_weights: Dict[str, float] = {}
        for i, a in enumerate(ranked):
            raw_weights[a] = weights_template[i] if i < len(weights_template) else 0.0

        # Include assets that have caps but no score yet (fail-open)
        for a in exposure_caps:
            if a not in raw_weights:
                raw_weights[a] = weights_template[-1] if weights_template else 0.2

        # Cap-aware normalization
        alloc = self._cap_aware_normalize(raw_weights, exposure_caps)

        # [S10] Correlation-aware guard: reduce/merge highly-correlated bets
        if correlations:
            alloc = self._apply_correlation_guard(alloc, correlations, ranked)

        # Anti-churn guard: don't force-close existing positions
        if current_exposures:
            for a in alloc:
                cur = current_exposures.get(a, 0.0)
                if cur > alloc[a]:
                    alloc[a] = cur

        # Determine tier name for proof log
        if top_score > self.config.strong_threshold:
            tier_name = "STRONG"
        elif top_score > self.config.mid_threshold:
            tier_name = "MID"
        else:
            tier_name = "BASE"

        # Cache for proof log
        self._last_allocation = {
            "scores": {a: round(self._scores.get(a, 0), 4) for a in ranked},
            "score_components": {
                a: {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in self._score_components.get(a, {}).items()
                }
                for a in ranked
            },
            "tier": tier_name,
            "top_score": round(top_score, 4),
            "raw_weights": {a: round(w, 3) for a, w in raw_weights.items()},
            "final_alloc": {a: round(v, 4) for a, v in alloc.items()},
        }

        return alloc.get(asset, exposure_caps.get(asset, 0.30))

    # -- [S10] Correlation guard -----------------------------------------------

    def _apply_correlation_guard(
        self,
        alloc: Dict[str, float],
        correlations: Dict[str, float],
        ranked: List[str],
    ) -> Dict[str, float]:
        """Reduce allocation for highly-correlated asset pairs.

        Args:
            alloc: Current allocation weights per asset.
            correlations: Pairwise correlations, e.g. {"BTC-ETH": 0.87, ...}.
                          Also accepts lowercase/underscore keys like "btc_eth".
            ranked: Assets sorted by score descending (highest score first).

        Returns:
            Adjusted allocation dict.
        """
        if not correlations or len(ranked) < 2:
            return alloc

        result = dict(alloc)

        def _get_corr(a1: str, a2: str) -> Optional[float]:
            """Look up pairwise correlation with flexible key format."""
            for fmt in (
                f"{a1}-{a2}", f"{a2}-{a1}",
                f"{a1}_{a2}".lower(), f"{a2}_{a1}".lower(),
                f"{a1.lower()}_{a2.lower()}", f"{a2.lower()}_{a1.lower()}",
            ):
                if fmt in correlations:
                    return correlations[fmt]
            return None

        merged_assets = set()

        # Walk ranked pairs - higher-score asset takes priority
        for i in range(len(ranked)):
            a_hi = ranked[i]
            if a_hi in merged_assets:
                continue
            for j in range(i + 1, len(ranked)):
                a_lo = ranked[j]
                if a_lo in merged_assets:
                    continue

                corr = _get_corr(a_hi, a_lo)
                if corr is None:
                    continue

                if corr > self.config.correlation_merge_threshold:
                    # Merge: zero out lower-score asset
                    logger.info(
                        f"[CORR_GUARD] {a_hi}-{a_lo} corr={corr:.2f} > "
                        f"{self.config.correlation_merge_threshold}, merged to {a_hi}-only"
                    )
                    result[a_lo] = 0.0
                    merged_assets.add(a_lo)
                elif corr > self.config.correlation_reduce_threshold:
                    # Reduce: discount lower-score asset
                    old_val = result[a_lo]
                    result[a_lo] *= self.config.correlation_reduce_factor
                    logger.info(
                        f"[CORR_GUARD] {a_hi}-{a_lo} corr={corr:.2f} > "
                        f"{self.config.correlation_reduce_threshold}, "
                        f"{a_lo} alloc {old_val:.3f}->{result[a_lo]:.3f}"
                    )

        # Re-normalize so sum <= 1.0 (don't inflate, just deflate)
        total = sum(result.values())
        if total > 1.0:
            for a in result:
                result[a] /= total

        return result

    def _select_tier(self, top_score: float) -> List[float]:
        """Select weight template based on top score."""
        if top_score > self.config.strong_threshold:
            return list(self.config.strong_weights)
        elif top_score > self.config.mid_threshold:
            return list(self.config.mid_weights)
        else:
            return list(self.config.base_weights)

    @staticmethod
    def _cap_aware_normalize(
        raw_weights: Dict[str, float],
        caps: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize weights respecting per-asset caps.

        Algorithm:
        1. Clamp each weight to its cap
        2. Accumulate overflow from clamped assets
        3. Redistribute overflow proportionally to uncapped assets
        4. Repeat up to 3 times (handles cascading overflow)
        """
        alloc = dict(raw_weights)

        for _ in range(3):  # max 3 redistribution rounds
            overflow = 0.0
            uncapped_total = 0.0
            capped_assets = set()

            for a in alloc:
                cap = caps.get(a, 1.0)
                if alloc[a] > cap:
                    overflow += alloc[a] - cap
                    alloc[a] = cap
                    capped_assets.add(a)

            if overflow < 1e-6:
                break  # No overflow - done

            # Sum weights of uncapped assets for proportional redistribution
            for a in alloc:
                if a not in capped_assets:
                    uncapped_total += alloc[a]

            if uncapped_total < 1e-6:
                # All assets capped - distribute overflow equally
                uncapped = [a for a in alloc if a not in capped_assets]
                if not uncapped:
                    break  # Everything at cap, can't redistribute
                per_asset = overflow / len(uncapped)
                for a in uncapped:
                    alloc[a] += per_asset
            else:
                # Proportional redistribution
                for a in alloc:
                    if a not in capped_assets and uncapped_total > 0:
                        share = alloc[a] / uncapped_total
                        alloc[a] += overflow * share

        # Final clamp (safety)
        for a in alloc:
            cap = caps.get(a, 1.0)
            alloc[a] = min(alloc[a], cap)

        return alloc

    # -- Proof log -----------------------------------------------------------

    def get_proof_snapshot(self) -> Optional[Dict]:
        """Return the last allocation decision for proof logging."""
        return self._last_allocation

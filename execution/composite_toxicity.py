"""
Composite Toxicity Score - Multi-dimensional order flow toxicity filter.

Combines 4 toxicity dimensions into a weighted score:
  1. Adverse selection (fill quality vs arrival price)
  2. VPIN (Volume-Synchronized Probability of Informed Trading)
  3. Order flow imbalance (buy/sell volume asymmetry)
  4. Price reversal (post-fill adverse moves)

Score > VETO_THRESHOLD -> MicrostructureAgent VETO (block execution).
Score > WARN_THRESHOLD -> log warning + reduce urgency.
"""
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ToxicityComponents:
    """Individual toxicity dimension scores (each 0-1)."""
    adverse_selection: float = 0.0
    vpin: float = 0.0
    flow_imbalance: float = 0.0
    price_reversal: float = 0.0


@dataclass
class CompositeToxicityResult:
    """Output from composite toxicity evaluation."""
    score: float = 0.0          # Weighted composite (0-1)
    should_veto: bool = False
    should_warn: bool = False
    components: Optional[ToxicityComponents] = None
    dominant_dimension: str = ""  # Which dimension contributed most
    profile_key: str = "DEFAULT:FLAT"
    warn_threshold: float = 0.60
    veto_threshold: float = 0.85


@dataclass(frozen=True)
class CompositeToxicityProfile:
    """Asset/side-specific toxicity calibration."""
    weights: Dict[str, float]
    warn_threshold: float
    veto_threshold: float


class CompositeToxicityFilter:
    """4-dimensional toxicity scorer with configurable weights and thresholds."""

    WEIGHTS = {
        "adverse_selection": 0.35,
        "vpin": 0.30,
        "flow_imbalance": 0.20,
        "price_reversal": 0.15,
    }

    VETO_THRESHOLD = 0.85    # > 0.85 -> block execution
    WARN_THRESHOLD = 0.60    # > 0.60 -> log warning, reduce urgency
    HISTORY_SIZE = 100       # Rolling history for calibration
    PROFILE_DEFAULTS: Dict[Tuple[str, str], CompositeToxicityProfile] = {
        ("DEFAULT", "flat"): CompositeToxicityProfile(
            weights=WEIGHTS,
            warn_threshold=WARN_THRESHOLD,
            veto_threshold=VETO_THRESHOLD,
        ),
        ("BTC", "long"): CompositeToxicityProfile(
            weights={"adverse_selection": 0.35, "vpin": 0.30, "flow_imbalance": 0.20, "price_reversal": 0.15},
            warn_threshold=0.60,
            veto_threshold=0.84,
        ),
        ("BTC", "short"): CompositeToxicityProfile(
            weights={"adverse_selection": 0.40, "vpin": 0.20, "flow_imbalance": 0.15, "price_reversal": 0.25},
            warn_threshold=0.66,
            veto_threshold=0.88,
        ),
        ("ETH", "long"): CompositeToxicityProfile(
            weights={"adverse_selection": 0.35, "vpin": 0.28, "flow_imbalance": 0.18, "price_reversal": 0.19},
            warn_threshold=0.60,
            veto_threshold=0.84,
        ),
        ("ETH", "short"): CompositeToxicityProfile(
            weights={"adverse_selection": 0.38, "vpin": 0.22, "flow_imbalance": 0.18, "price_reversal": 0.22},
            warn_threshold=0.64,
            veto_threshold=0.86,
        ),
        ("SOL", "long"): CompositeToxicityProfile(
            weights={"adverse_selection": 0.38, "vpin": 0.24, "flow_imbalance": 0.18, "price_reversal": 0.20},
            warn_threshold=0.63,
            veto_threshold=0.87,
        ),
        ("SOL", "short"): CompositeToxicityProfile(
            weights={"adverse_selection": 0.42, "vpin": 0.18, "flow_imbalance": 0.15, "price_reversal": 0.25},
            warn_threshold=0.68,
            veto_threshold=0.90,
        ),
    }

    def __init__(self):
        self._score_history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._profiles: Dict[Tuple[str, str], CompositeToxicityProfile] = dict(self.PROFILE_DEFAULTS)

    @staticmethod
    def _normalize_asset(asset: str) -> str:
        return str(asset or "").upper().replace("/USD", "").replace("USD", "") or "DEFAULT"

    @staticmethod
    def _normalize_side(side: str) -> str:
        side_key = str(side or "flat").lower()
        return side_key if side_key in {"long", "short", "flat"} else "flat"

    def configure_profile(
        self,
        *,
        asset: str,
        side: str,
        weights: Optional[Dict[str, float]] = None,
        warn_threshold: Optional[float] = None,
        veto_threshold: Optional[float] = None,
    ) -> None:
        asset_key = self._normalize_asset(asset)
        side_key = self._normalize_side(side)
        base = self._profiles.get(
            (asset_key, side_key),
            self._profiles.get((asset_key, "flat"), self._profiles[("DEFAULT", "flat")]),
        )
        merged_weights = dict(base.weights)
        if isinstance(weights, dict):
            merged_weights.update({
                key: float(value)
                for key, value in weights.items()
                if key in {"adverse_selection", "vpin", "flow_imbalance", "price_reversal"}
            })
        weight_total = sum(merged_weights.values()) or 1.0
        normalized_weights = {key: value / weight_total for key, value in merged_weights.items()}
        self._profiles[(asset_key, side_key)] = CompositeToxicityProfile(
            weights=normalized_weights,
            warn_threshold=float(warn_threshold if warn_threshold is not None else base.warn_threshold),
            veto_threshold=float(veto_threshold if veto_threshold is not None else base.veto_threshold),
        )

    def _resolve_profile(self, asset: str, side: str) -> Tuple[str, CompositeToxicityProfile]:
        asset_key = self._normalize_asset(asset)
        side_key = self._normalize_side(side)
        profile = self._profiles.get((asset_key, side_key))
        if profile is not None:
            return f"{asset_key}:{side_key.upper()}", profile
        profile = self._profiles.get((asset_key, "flat"))
        if profile is not None:
            return f"{asset_key}:FLAT", profile
        return "DEFAULT:FLAT", self._profiles[("DEFAULT", "flat")]

    def compute(
        self,
        adverse_selection: float = 0.0,
        vpin: float = 0.0,
        flow_imbalance: float = 0.0,
        price_reversal: float = 0.0,
        asset: str = "",
        side: str = "flat",
    ) -> CompositeToxicityResult:
        """Compute weighted composite toxicity score.

        Args:
            adverse_selection: 0-1, fraction of fills at worse-than-mid prices.
            vpin: 0-1, probability of informed trading (from VPIN module).
            flow_imbalance: 0-1, |buy_vol - sell_vol| / total_vol.
            price_reversal: 0-1, fraction of fills followed by adverse 1-min move.

        Returns:
            CompositeToxicityResult with score and veto/warn flags.
        """
        components = ToxicityComponents(
            adverse_selection=np.clip(adverse_selection, 0, 1),
            vpin=np.clip(vpin, 0, 1),
            flow_imbalance=np.clip(flow_imbalance, 0, 1),
            price_reversal=np.clip(price_reversal, 0, 1),
        )
        profile_key, profile = self._resolve_profile(asset, side)
        weights = profile.weights

        score = (
            weights["adverse_selection"] * components.adverse_selection
            + weights["vpin"] * components.vpin
            + weights["flow_imbalance"] * components.flow_imbalance
            + weights["price_reversal"] * components.price_reversal
        )
        score = min(1.0, max(0.0, score))

        # Track dominant dimension
        dim_scores = {
            "adverse_selection": components.adverse_selection * weights["adverse_selection"],
            "vpin": components.vpin * weights["vpin"],
            "flow_imbalance": components.flow_imbalance * weights["flow_imbalance"],
            "price_reversal": components.price_reversal * weights["price_reversal"],
        }
        dominant = max(dim_scores, key=dim_scores.get)

        self._score_history.append(score)

        should_veto = bool(score > profile.veto_threshold)
        should_warn = bool(score > profile.warn_threshold)

        if should_veto:
            logger.warning(
                f"[TOXICITY] VETO score={score:.3f} "
                f"(as={components.adverse_selection:.2f} "
                f"vpin={components.vpin:.2f} "
                f"fi={components.flow_imbalance:.2f} "
                f"pr={components.price_reversal:.2f}) "
                f"dominant={dominant} profile={profile_key}"
            )
        elif should_warn:
            logger.info(
                f"[TOXICITY] WARN score={score:.3f} dominant={dominant} profile={profile_key}"
            )

        return CompositeToxicityResult(
            score=score,
            should_veto=should_veto,
            should_warn=should_warn,
            components=components,
            dominant_dimension=dominant,
            profile_key=profile_key,
            warn_threshold=profile.warn_threshold,
            veto_threshold=profile.veto_threshold,
        )

    @property
    def rolling_mean(self) -> float:
        """Rolling average toxicity score."""
        if not self._score_history:
            return 0.0
        return float(np.mean(self._score_history))


# Module-level shared instance kept only for explicit opt-in callers.
_filter: Optional[CompositeToxicityFilter] = None


def get_composite_toxicity_filter(*, shared: bool = False) -> CompositeToxicityFilter:
    """Get a CompositeToxicityFilter instance.

    The default is a fresh instance so per-engine calibration and tests do not
    leak thresholds or history through a process-wide singleton. Callers that
    intentionally want a shared rolling-history instance must opt in.
    """
    global _filter
    if not shared:
        return CompositeToxicityFilter()
    if _filter is None:
        _filter = CompositeToxicityFilter()
    return _filter


def reset_composite_toxicity_filter() -> None:
    """Reset the optional shared CompositeToxicityFilter instance."""
    global _filter
    _filter = None

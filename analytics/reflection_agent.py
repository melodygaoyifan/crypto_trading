"""
HMATS v3.2 - Weekly Reflection Agent
Pattern: Periodic aggregation of StrategyConfidenceScorer data.
Mode: ADVISORY (logs to Decision Log, does not affect fusion weights).

[v3.2-C1] Inspired by CryptoTrade (EMNLP 2024) reflection mechanism.
Every 42 ticks (7 days at 4H), aggregates signal outcomes by (strategy, regime)
and surfaces underperformers, alpha leaks, and regime mismatches.

Usage:
    from analytics.confidence_scorer import get_confidence_scorer
    from core.component_lifecycle import ComponentLifecycle

    scorer = get_confidence_scorer()
    lifecycle = ComponentLifecycle("ReflectionAgent")
    lifecycle.promote("ADVISORY")

    agent = ReflectionAgent(scorer=scorer, lifecycle=lifecycle)

    # In tick loop:
    if agent.should_run(tick_count):
        report = agent.run_weekly()
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analytics.confidence_scorer import StrategyConfidenceScorer
    from core.component_lifecycle import ComponentLifecycle

logger = logging.getLogger(__name__)

# Minimum signals per (strategy, regime) cell before scoring
MIN_CELL_SAMPLES = 5

# Multiplier bounds
MULTIPLIER_MIN = 0.5
MULTIPLIER_MAX = 1.2

# Weekly = 42 ticks at 4H frequency
WEEKLY_TICKS = 42


@dataclass
class ReflectionReport:
    """Weekly reflection output."""
    reliability_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_multipliers: Dict[str, float] = field(default_factory=dict)
    alpha_leaks: List[str] = field(default_factory=list)
    regime_mismatches: List[str] = field(default_factory=list)
    total_signals_evaluated: int = 0
    generated_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "reliability_scores": self.reliability_scores,
            "confidence_multipliers": self.confidence_multipliers,
            "alpha_leaks": self.alpha_leaks,
            "regime_mismatches": self.regime_mismatches,
            "total_signals": self.total_signals_evaluated,
            "generated_at": self.generated_at,
        }


class ReflectionAgent:
    """
    Weekly reflection agent - aggregates StrategyConfidenceScorer data
    into per-(strategy, regime) reliability scores.

    ADVISORY mode: computes and logs multipliers but does NOT apply them.
    """

    def __init__(self, scorer: 'StrategyConfidenceScorer', lifecycle: 'ComponentLifecycle'):
        self.scorer = scorer
        self.lifecycle = lifecycle
        self._last_report: Optional[ReflectionReport] = None
        logger.info("[ReflectionAgent] Initialized (reads from ConfidenceScorer singleton)")

    def should_run(self, tick_count: int) -> bool:
        """Check if weekly reflection should fire."""
        if self.lifecycle.is_disabled():
            return False
        return tick_count > 0 and tick_count % WEEKLY_TICKS == 0

    def run_weekly(self) -> Optional[ReflectionReport]:
        """Execute weekly reflection. Returns report or None on error."""
        try:
            # Gather all completed signals from scorer
            all_completed = []
            for strategy_name, signals in self.scorer._signals.items():
                for sig in signals:
                    if sig.has_outcome:
                        all_completed.append(sig)

            if not all_completed:
                logger.info("[REFLECTION][ADVISORY] No completed signals to reflect on")
                return None

            reliability = self._compute_reliability(all_completed)
            multipliers = self._compute_multipliers(reliability)
            alpha_leaks = self._detect_alpha_leaks(reliability)
            regime_mismatches = self._detect_regime_mismatches(reliability)

            report = ReflectionReport(
                reliability_scores=reliability,
                confidence_multipliers=multipliers,
                alpha_leaks=alpha_leaks,
                regime_mismatches=regime_mismatches,
                total_signals_evaluated=len(all_completed),
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

            self._log_report(report)
            self._last_report = report
            self.lifecycle.record_success()
            return report

        except Exception as e:
            logger.warning(f"[REFLECTION][ADVISORY] Weekly reflection failed: {e}")
            self.lifecycle.record_error()
            return None

    def _compute_reliability(self, signals) -> Dict[str, Dict[str, float]]:
        """Group signals by (strategy, regime) and compute reliability scores."""
        # Group by (strategy, regime)
        groups: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for sig in signals:
            groups[sig.strategy_name][sig.regime_at_signal].append(sig)

        reliability: Dict[str, Dict[str, float]] = {}
        for strategy, regimes in groups.items():
            reliability[strategy] = {}
            for regime, sigs in regimes.items():
                if len(sigs) < MIN_CELL_SAMPLES:
                    reliability[strategy][regime] = 1.0  # neutral
                    continue

                hit_rate = sum(1 for s in sigs if s.direction_correct) / len(sigs)
                avg_pnl = statistics.mean([s.realized_pnl_bps for s in sigs])
                # Normalize PnL score: -100bps -> 0.0, 0 -> 0.5, +100bps -> 1.0
                pnl_score = max(0.0, min(1.0, (avg_pnl + 100) / 200))

                score = 0.5 + hit_rate * 0.4 + pnl_score * 0.3
                reliability[strategy][regime] = round(max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, score)), 3)

        return reliability

    def _compute_multipliers(self, reliability: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute per-strategy multiplier (average across regimes)."""
        multipliers = {}
        for strategy, regimes in reliability.items():
            if regimes:
                avg = statistics.mean(regimes.values())
                multipliers[strategy] = round(max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, avg)), 3)
            else:
                multipliers[strategy] = 1.0
        return multipliers

    def _detect_alpha_leaks(self, reliability: Dict[str, Dict[str, float]]) -> List[str]:
        """Find strategies with good direction but poor PnL capture."""
        leaks = []
        for strategy, regimes in self.scorer._signals.items():
            completed = [s for s in regimes if s.has_outcome]
            if len(completed) < MIN_CELL_SAMPLES:
                continue
            hit_rate = sum(1 for s in completed if s.direction_correct) / len(completed)
            avg_pnl_vs_exp = statistics.mean([
                (s.realized_pnl_bps - s.expected_pnl_bps) for s in completed
            ])
            # Good direction (>60%) but underperforming expectations
            if hit_rate > 0.6 and avg_pnl_vs_exp < -20:
                leaks.append(f"{strategy} (hit={hit_rate:.0%}, pnl_gap={avg_pnl_vs_exp:+.0f}bps)")
        return leaks

    def _detect_regime_mismatches(self, reliability: Dict[str, Dict[str, float]]) -> List[str]:
        """Find strategies performing poorly in their 'home' regime."""
        from analytics.confidence_scorer import STRATEGY_GROUPS, REGIME_STRATEGY_MAP

        # Invert: strategy -> group
        strategy_to_group = {}
        for group, strategies in STRATEGY_GROUPS.items():
            for s in strategies:
                strategy_to_group[s] = group

        # Invert: group -> home regimes
        group_to_regimes = defaultdict(list)
        for regime, group in REGIME_STRATEGY_MAP.items():
            group_to_regimes[group].append(regime)

        mismatches = []
        for strategy, regimes in reliability.items():
            group = strategy_to_group.get(strategy)
            if not group or group == "universal":
                continue
            home_regimes = group_to_regimes.get(group, [])
            for regime in home_regimes:
                score = regimes.get(regime)
                if score is not None and score < 0.7:
                    mismatches.append(f"{strategy} in {regime} (score={score:.2f})")
        return mismatches

    def _log_report(self, report: ReflectionReport):
        """Log the weekly reflection report."""
        logger.info(
            f"[REFLECTION][ADVISORY] Weekly report: "
            f"{report.total_signals_evaluated} signals evaluated"
        )

        for strategy, mult in sorted(report.confidence_multipliers.items()):
            tag = ""
            if mult < 0.7:
                tag = " [WARN] UNDERPERFORMING"
            elif mult > 1.0:
                tag = " ★ OUTPERFORMING"
            logger.info(f"[REFLECTION][ADVISORY]   {strategy}: multiplier={mult:.3f}{tag}")

        if report.alpha_leaks:
            logger.warning(
                f"[REFLECTION][ADVISORY] Alpha leaks detected: {report.alpha_leaks}"
            )

        if report.regime_mismatches:
            logger.warning(
                f"[REFLECTION][ADVISORY] Regime mismatches: {report.regime_mismatches}"
            )

    @property
    def last_report(self) -> Optional[ReflectionReport]:
        return self._last_report

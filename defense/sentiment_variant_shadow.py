"""
================================================================================
HMATS [P293e] - Sentiment interpretation exam: three claims, one ledger
================================================================================

The P293d audit found that the live Fear & Greed signal is THREE stacked
assertions, none of them measured, and that two of them contradict each other:

  1. `main.py` maps F&G by MOMENTUM  -- (fg-50)/50*3, so fear reads BEARISH.
     This is the value that reaches fusion (F&G=31 -> z=-1.14 ->
     sentiment_direction=-1.0, confidence 0.38).
  2. `signals/deterministic_sentiment.py` -- the module CLAUDE.md lists as
     "Sentiment L1 (F&G) ACTIVE" -- maps it CONTRARIAN (greed -0.6, extreme
     fear 0.0 "don't chase shorts"). Its output is logged and never becomes
     `sentiment_direction`.
  3. The "z-score" is not a z-score at all (P293): it is a fixed linear
     rescale, because the feed asks for limit=1. The real trailing
     distribution is free from the same endpoint.

They disagree in SIGN whenever the market is greedy, and the 90-day live IC
(+0.001 at 4h, +0.049 at 16h -- both insignificant) cannot tell which is
right. An argument cannot settle this; an exam can.

So all three claims are recorded side by side, every tick, and judged by the
SAME P166 cost-aware gate. Zero live risk: this module places no orders,
touches no signal, and is wired loop-level fail-soft.

WHY THE CONTRARIAN VARIANT IS NOT JUST `-momentum`
    A pure negation would be statistically empty -- its IC is exactly the
    negative of the momentum form, so scoring it separately would add no
    information. The contrarian claim recorded here is the DETERMINISTIC
    ENGINE'S OWN asymmetric mapping (greed penalised hard, extreme fear
    deliberately NEUTRAL rather than bullish), which is a genuinely
    different signal shape -- and it is the mapping the codebase already
    calls its sentiment engine. That asymmetry is the whole point: the
    engine can never emit a bullish value, and this ledger will show
    whether that one-sidedness costs or saves.

Ledger:  data/strategy_shadow/sentvariant_{ASSET}.jsonl
Scored:  analytics/shadow_ic/compute_shadow_ic.py (prefix `sentvariant`,
         grouped by the record's `strategy` field -- three series)
================================================================================
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HMATS.SentimentVariant")

# The live band: |z| > 1.0 becomes directional (~F&G outside 33-67). Reused
# for every variant so the three differ ONLY in interpretation, never in
# how decisively they are expressed (P172 -- one comparison, three inputs).
DIRECTIONAL_BAND = 1.0

STRATEGY_MOMENTUM_LINEAR = "sent_momentum_linear"
STRATEGY_MOMENTUM_HIST = "sent_momentum_hist"
STRATEGY_CONTRARIAN = "sent_contrarian"

# [P310] SINGLE SOURCE for the names this module writes into a record's
# `strategy` field. Consumers (analytics/shadow_ic) must not restate
# them: P309 keyed its allowlists on LEDGER-FILE PREFIXES instead, so
# two families were silently never pooled and an archive section never
# rendered. A conformance test asserts every consumer name is one of
# these, and that every one of these is classified by a consumer.
SHADOW_STRATEGY_NAMES = frozenset({
    STRATEGY_MOMENTUM_LINEAR, STRATEGY_MOMENTUM_HIST, STRATEGY_CONTRARIAN,
})


def momentum_direction(zscore: Optional[float]) -> float:
    """The LIVE rule, verbatim: |z| > 1.0 -> sign(z), else flat.

    Mirrors main.py's `sentiment_direction` derivation so the ledger's
    momentum claim is the signal actually being traded, not a re-derivation
    that could drift from it.
    """
    if zscore is None:
        return 0.0
    try:
        z = float(zscore)
    except (TypeError, ValueError):  # noqa: silent-swallow
        return 0.0
    if z != z:  # NaN
        return 0.0
    if abs(z) > DIRECTIONAL_BAND:
        return 1.0 if z > 0 else -1.0
    return 0.0


def contrarian_signal(fg_value: Optional[float]) -> float:
    """The deterministic engine's asymmetric contrarian mapping, verbatim.

    Thresholds copied from `signals/deterministic_sentiment.py` (the F&G
    branch). Note the deliberate asymmetry the audit surfaced: NO branch is
    positive, so this can never emit a bullish claim, and extreme fear maps
    to 0.0 ("don't chase shorts") rather than to a buy.

    Returns the engine's raw signal in [-0.6, 0.0].
    """
    if fg_value is None:
        return 0.0
    try:
        fg = float(fg_value)
    except (TypeError, ValueError):  # noqa: silent-swallow
        return 0.0
    if fg != fg:
        return 0.0
    if fg > 75:
        return -0.6
    if fg > 55:
        return -0.3
    if fg < 25:
        return 0.0      # extreme fear -> neutral, NOT bullish
    if fg < 45:
        return -0.2
    return 0.0


def contrarian_direction(fg_value: Optional[float]) -> float:
    """Discretised contrarian claim: sign of the engine's signal."""
    s = contrarian_signal(fg_value)
    if s > 1e-9:
        return 1.0
    if s < -1e-9:
        return -1.0
    return 0.0


class SentimentVariantShadow:
    """Records the three competing interpretations, observation-only."""

    def __init__(self, data_dir: Optional[str] = None):
        import os
        base = data_dir or os.environ.get("HMATS_DATA_DIR", "data")
        self._dir = Path(base) / "strategy_shadow"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("[SENTVARIANT] ledger dir unavailable: %s", e)

    def record_tick(
        self,
        asset: str,
        fg_value: Optional[float],
        z_linear: Optional[float],
        z_historical: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Append one row per interpretation. Never raises.

        A variant whose input is MISSING is skipped entirely rather than
        written as a flat row: an absent historical z (thin window) is not
        the same claim as "the historical reading says flat", and conflating
        them would let a starved variant look like a confident neutral (P2).
        """
        out: List[Dict[str, Any]] = []
        try:
            if fg_value is None:
                return out

            now = time.time()
            iso = datetime.now(timezone.utc).isoformat()

            variants = [
                (STRATEGY_MOMENTUM_LINEAR, momentum_direction(z_linear),
                 z_linear, "live_rule"),
                (STRATEGY_MOMENTUM_HIST, momentum_direction(z_historical),
                 z_historical, "historical_z"),
                (STRATEGY_CONTRARIAN, contrarian_direction(fg_value),
                 contrarian_signal(fg_value), "deterministic_engine"),
            ]

            path = self._dir / f"sentvariant_{asset}.jsonl"
            lines = []
            for name, direction, basis, kind in variants:
                if basis is None:
                    continue    # input absent -> no claim, not a flat claim
                rec = {
                    "ts": now,
                    "iso": iso,
                    "strategy": name,
                    "asset": asset,
                    "direction": float(direction),
                    # scorer multiplies direction x confidence (P236/P224):
                    # a flat row must contribute zero, never a saturated
                    # claim, so confidence IS |direction|.
                    "confidence": abs(float(direction)),
                    "fg_value": float(fg_value),
                    "basis": None if basis is None else float(basis),
                    "basis_kind": kind,
                }
                lines.append(json.dumps(rec))
                out.append(rec)

            if lines:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
        except Exception as e:  # noqa: silent-swallow — ledger-stale-this-tick
            logger.warning(
                "[SENTVARIANT] %s record failed: %s — ledger stale this tick",
                asset, type(e).__name__)
        return out


_instance: Optional[SentimentVariantShadow] = None


def get_sentiment_variant_shadow(**kwargs) -> SentimentVariantShadow:
    global _instance
    if _instance is None:
        _instance = SentimentVariantShadow(**kwargs)
    return _instance


def reset_sentiment_variant_shadow():
    global _instance
    _instance = None

"""
Regime-aware IC-weighted signal fusion (Layers 1+2 of the alpha/beta end-to-end
fix; see docs/HMATS_IMPLEMENTATION_PROMPT_ALPHA_BETA_ENDTOEND.md).

WHY (forensic, CLAUDE.md P143): per-agent live IC showed the DECIDE `quant` signal
is aggregate noise (IC -0.018) but a REGIME SIGN-FLIP (+0.17 weak-consol / -0.41
uptrend); applied sign-blind it cancels to ~0. Other agents are inverted (model_
alpha -0.16, llm_sentiment -0.05) or dead. The current fusion is authority-bucketed
and regime-naive on signs, so real per-regime alpha is destroyed before it is used.

WHAT this does (research-grounded, all SHADOW-first / unvalidated on crypto 4H):
  - Layer 1 (R1, Wang/Lin/Mikhelson 2020): apply each signal with a per-(agent,
    regime) SIGN learned from rolling realized IC. SELECTIVE — only where the sign
    is stable out-of-sample. Internal OOS proved this recovers ~+10bps on quant /
    +9bps on llm_sentiment but HURTS stable-sign sources (whale, drl) -> those stay
    blind.
  - Layer 2 (R2, Garleanu-Pedersen 2013 / Kakushadze-Yu 2016): weight each signal
    by |rolling IC| (edge-weighting), so a zero-IC (dead/noise) agent gets ~zero
    weight automatically — no hardcoded demotion.

IRON LAWS (this module enforces them structurally):
  - It NEVER decides a live trade. `shadow_fuse()` returns a direction for LOGGING
    ONLY. There is no enforce path in this module.
  - Fail-open: any error -> returns None (caller logs and ignores), never raises.
  - Confidence is NOT a feature (measured anti-predictive).
"""

from __future__ import annotations

import logging
from collections import deque, defaultdict
from typing import Dict, Deque, Optional, Tuple, Any, List

logger = logging.getLogger(__name__)


def _sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


class RegimeICFusion:
    """Rolling per-(agent, regime) IC tracker + regime-aware IC-weighted shadow fuse.

    Stable-sign agents (a configurable allowlist) are used BLIND (no sign flip) —
    the OOS evidence showed regime-sign-conditioning hurts them.
    """

    # Agents whose sign is stable across regimes (OOS): use blind, do NOT regime-flip.
    # [P265] "funding_rate", not "funding": the only caller (main.py's P232
    # wiring) names the agent "funding_rate", so the old member matched
    # NOTHING and funding was regime-sign-flipped in shadow_fuse — the
    # [RIC-SHADOW] promotion-evidence stream was measuring a different fusion
    # than the one the OOS evidence designed (P214 class). Pinned by test
    # against the caller's roster.
    DEFAULT_STABLE_SIGN = frozenset({"whale", "drl", "funding_rate"})

    def __init__(
        self,
        window: int = 60,
        min_samples: int = 20,
        stable_sign_agents: Optional[frozenset] = None,
    ) -> None:
        self._window = int(window)
        self._min_samples = int(min_samples)
        self._stable = stable_sign_agents if stable_sign_agents is not None else self.DEFAULT_STABLE_SIGN
        # (agent, regime) -> deque[(signed_dir, fwd_ret)]
        self._hist: Dict[Tuple[str, str], Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    # ---- learning -----------------------------------------------------------
    def record_outcome(self, regime: str, agent_dirs: Dict[str, float], fwd_ret: float) -> None:
        """Record one bar's outcome: each agent's direction vs the realized forward
        return, bucketed by the regime that was active when the signal was emitted."""
        if regime is None or fwd_ret is None:
            return
        for ag, d in agent_dirs.items():
            try:
                d = float(d)
            except (TypeError, ValueError):  # noqa: silent-swallow — non-numeric agent dir, skip (logging each would spam)
                continue
            if abs(d) < 1e-9:
                continue
            self._hist[(ag, regime)].append((d, float(fwd_ret)))

    def ic(self, agent: str, regime: str) -> Tuple[float, int]:
        """Realized IC proxy = mean(sign(dir) * fwd_ret) over the rolling window,
        normalized to a unit-ish scale. Returns (ic, n). Cheap, no scipy."""
        h = self._hist.get((agent, regime))
        if not h:
            return 0.0, 0
        n = len(h)
        if n == 0:
            return 0.0, 0
        edge = sum(_sgn(d) * r for d, r in h) / n  # avg signed forward return
        return edge, n

    # ---- shadow fusion (LOG ONLY) ------------------------------------------
    def shadow_fuse(self, regime: str, agent_dirs: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Return a regime-aware IC-weighted direction for LOGGING ONLY (never a
        live decision). None if not enough learned history yet (fail-open)."""
        try:
            num = 0.0
            den = 0.0
            used: List[str] = []
            for ag, d in agent_dirs.items():
                try:
                    d = float(d)
                except (TypeError, ValueError):  # noqa: silent-swallow — non-numeric agent dir, skip (logging each would spam)
                    continue
                if abs(d) < 1e-9:
                    continue
                edge, n = self.ic(ag, regime)
                if n < self._min_samples or abs(edge) < 1e-9:
                    continue  # WARMUP / no edge -> contributes nothing
                weight = abs(edge)  # Layer 2: IC-magnitude weighting
                if ag in self._stable:
                    # Layer 1 carve-out: stable-sign agent used blind (no regime flip)
                    contrib = _sgn(d)
                else:
                    # Layer 1: regime-conditional sign = signal sign * sign(regime IC)
                    contrib = _sgn(d) * _sgn(edge)
                num += contrib * weight
                den += weight
                used.append(ag)
            if den <= 0:
                return None
            direction = max(-1.0, min(1.0, num / den))
            return {"direction": direction, "weight_sum": den, "agents_used": used}
        except Exception as e:  # fail-open, never raise into the tick
            logger.warning(f"[REGIME-IC] shadow_fuse error: {type(e).__name__}: {e}")
            return None

    # ---- persistence --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self._window,
            "min_samples": self._min_samples,
            "hist": {f"{a}|{r}": list(dq) for (a, r), dq in self._hist.items() if dq},
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        try:
            for key, items in (data.get("hist") or {}).items():
                if "|" not in key:
                    continue
                a, r = key.split("|", 1)
                self._hist[(a, r)] = deque(
                    ((float(d), float(v)) for d, v in items), maxlen=self._window
                )
        except Exception as e:
            logger.warning(f"[REGIME-IC] from_dict error: {type(e).__name__}: {e}")

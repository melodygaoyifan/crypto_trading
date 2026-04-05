"""
================================================================================
HMATS Alpha Boost — P0 CAGR Improvements (V2: Hardened)
================================================================================
Version: 2.0.0
Purpose: Three targeted upgrades to improve realized annualized return:
  P0.1 — Regime-transition selective aggression
  P0.2 — Quant-strategy adaptive confidence (NOT full multi-expert gating)
  P0.3 — Execution-to-alpha feedback loop

V2 Hardening (2026-04-05):
  - Raised sample thresholds (expert_gating_min_trades: 25, exec_feedback_min_fills: 15)
  - Bayesian shrinkage: low-sample weights blend toward 1.0 (neutral)
  - Global clamps: combined gate_mult in [0.80, 1.15], combined size_mult in [0.85, 1.25]
  - Mode-isolated state: paper / live / verify keep separate adaptive data
  - Honest scope: P0.2 is quant-confidence-only, documented as such
  - All clamp/shrinkage events logged for auditability

Design constraints:
  - Does NOT create new authority levels or override vetoes
  - All effects are multiplicative modifiers with bounded ranges
  - Everything is logged to proof logs for auditability
  - Everything is config-gated and reversible
  - Paper state cannot contaminate live adaptation

Integration:
  Called from main.py's _process_4h_tick_inner, between agent signals
  and engine.decide(). Outputs are injected into agent_signals for
  downstream consumption by existing regime_power / aggression / fusion.
================================================================================
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from collections import deque

logger = logging.getLogger("HMATS.AlphaBoost")


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AlphaBoostConfig:
    """Configuration for all P0 CAGR improvements. All flags default OFF."""

    # --- P0.1: Regime-transition selective aggression ---
    transition_aggression_enabled: bool = False
    transition_alpha_gate_mult: float = 0.80
    transition_size_boost: float = 1.20
    favorable_transitions: Dict[str, list] = field(default_factory=lambda: {
        "QUIET_ACCUMULATION": ["MOMENTUM_RALLY", "STEADY_UPTREND", "STRONG_TREND"],
        "WEAK_CONSOLIDATION": ["MOMENTUM_RALLY", "STEADY_UPTREND"],
        "NEUTRAL_DRIFT": ["MOMENTUM_RALLY", "STEADY_UPTREND"],
        "VOLATILE_CHOP": ["MOMENTUM_RALLY", "PANIC_SELLOFF"],
    })
    transition_min_phase_confidence: float = 0.40
    transition_max_age_bars: int = 2
    transition_require_volume_confirm: bool = True

    # --- P0.2: Quant-strategy adaptive confidence ---
    # NOTE: This is quant-confidence-only. It does NOT gate sentiment,
    # microstructure, squeeze, or other auxiliary experts. Broader multi-expert
    # gating is deferred to a future version with more live data.
    expert_gating_enabled: bool = False
    expert_gating_min_trades: int = 25          # V2: raised from 8 (need ~6 weeks of 4H data)
    expert_gating_max_boost: float = 1.20       # V2: reduced from 1.30 (more conservative)
    expert_gating_min_weight: float = 0.60      # V2: raised from 0.50 (less punishing)
    expert_gating_shrinkage_strength: float = 0.6  # V2: Bayesian shrinkage toward neutral
    # shrinkage formula: effective_weight = neutral + (raw_weight - neutral) * shrinkage_factor
    # where shrinkage_factor = min(1.0, n_trades / expert_gating_min_trades) ^ shrinkage_strength
    expert_strategy_map: Dict[str, str] = field(default_factory=lambda: {
        "momentum": "trend_following",
        "mean_revert": "mean_reversion",
        "volume_breakout": "range_breakout",
        "vrp": "volatility_breakout",
    })

    # --- P0.3: Execution-to-alpha feedback ---
    exec_feedback_enabled: bool = False
    exec_feedback_min_fills: int = 15           # V2: raised from 5 (need ~1 week stable data)
    exec_feedback_slippage_penalty_bps: float = 3.0
    exec_feedback_maker_bonus_threshold: float = 0.70
    exec_feedback_maker_bonus: float = 1.03     # V2: reduced from 1.05
    exec_feedback_gate_mult_range: tuple = (0.93, 1.07)  # V2: tightened from (0.90, 1.10)

    # --- V2: Global clamps (applied AFTER all component multipliers combine) ---
    global_gate_mult_clamp: tuple = (0.80, 1.15)    # Combined alpha gate effect
    global_size_mult_clamp: tuple = (0.85, 1.25)     # Combined size effect
    global_confidence_mult_clamp: tuple = (0.75, 1.25) # Combined quant confidence effect


# =============================================================================
# STATE TRACKING
# =============================================================================

@dataclass
class TransitionState:
    """Per-asset regime transition tracking."""
    previous_regime: str = ""
    current_regime: str = ""
    transition_tick: int = 0
    transition_confirmed: bool = False
    volume_confirmed: bool = False
    phase_confidence: float = 0.0


@dataclass
class ExecFeedbackState:
    """Per-asset execution quality tracking."""
    fills: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def avg_slippage_bps(self) -> float:
        if not self.fills:
            return 0.0
        return sum(f["slippage_bps"] for f in self.fills) / len(self.fills)

    @property
    def maker_rate(self) -> float:
        if not self.fills:
            return 0.5
        return sum(1 for f in self.fills if f["is_maker"]) / len(self.fills)

    @property
    def avg_fill_pct(self) -> float:
        if not self.fills:
            return 1.0
        return sum(f["fill_pct"] for f in self.fills) / len(self.fills)

    @property
    def n_fills(self) -> int:
        return len(self.fills)


# =============================================================================
# MAIN CLASS
# =============================================================================

class AlphaBoostController:
    """
    Centralized controller for P0 CAGR improvements (V2 hardened).

    Output keys in agent_signals:
      _ab_transition_active: bool
      _ab_gate_mult: float           — alpha gate multiplier (P0.1 + P0.3), globally clamped
      _ab_size_mult: float           — position size multiplier (P0.1), globally clamped
      _ab_expert_weights: dict       — per-quant-strategy confidence modifiers (P0.2)
      _ab_exec_confidence: float     — execution quality confidence modifier (P0.3)
      _ab_proof: str                 — human-readable proof log entry

    V2 guarantees:
      - Combined gate_mult always in [global_gate_mult_clamp]
      - Combined size_mult always in [global_size_mult_clamp]
      - Expert weights shrink toward 1.0 with low samples
      - Paper/live/verify state is isolated by mode prefix
    """

    def __init__(self, config: Optional[AlphaBoostConfig] = None, mode: str = "paper"):
        self.config = config or AlphaBoostConfig()
        self._mode = mode  # V2.3: mode isolation
        self._transition_state: Dict[str, TransitionState] = {}
        self._exec_state: Dict[str, ExecFeedbackState] = {}
        self._tick_count: Dict[str, int] = {}

    # -----------------------------------------------------------------
    # MAIN ENTRY
    # -----------------------------------------------------------------

    def compute(
        self,
        asset: str,
        agent_signals: Dict[str, Any],
        market_data: Dict[str, Any],
        phase_result: Any,
        confidence_scorer: Any,
    ) -> Dict[str, Any]:
        self._tick_count[asset] = self._tick_count.get(asset, 0) + 1

        result = {
            "_ab_transition_active": False,
            "_ab_gate_mult": 1.0,
            "_ab_size_mult": 1.0,
            "_ab_expert_weights": {},
            "_ab_exec_confidence": 1.0,
            "_ab_proof": "",
        }

        proofs = []

        # --- P0.1: Regime-transition selective aggression ---
        if self.config.transition_aggression_enabled:
            t = self._compute_transition(asset, agent_signals, market_data, phase_result)
            if t["active"]:
                result["_ab_transition_active"] = True
                result["_ab_gate_mult"] *= t["gate_mult"]
                result["_ab_size_mult"] *= t["size_mult"]
                proofs.append(t["proof"])

        # --- P0.2: Quant-strategy adaptive confidence ---
        if self.config.expert_gating_enabled and confidence_scorer:
            eg = self._compute_expert_gating(asset, agent_signals, confidence_scorer)
            result["_ab_expert_weights"] = eg["weights"]
            if eg["proof"]:
                proofs.append(eg["proof"])

        # --- P0.3: Execution-to-alpha feedback ---
        if self.config.exec_feedback_enabled:
            ef = self._compute_exec_feedback(asset)
            result["_ab_gate_mult"] *= ef["gate_mult"]
            result["_ab_exec_confidence"] = ef["confidence_mult"]
            if ef["proof"]:
                proofs.append(ef["proof"])

        # --- V2.2: Global clamps ---
        result = self._apply_global_clamps(result, proofs)

        result["_ab_proof"] = " | ".join(proofs) if proofs else "inactive"
        return result

    # -----------------------------------------------------------------
    # V2.2: Global clamps
    # -----------------------------------------------------------------

    def _apply_global_clamps(self, result: dict, proofs: list) -> dict:
        """Apply global bounds on combined multipliers. Prevents hidden aggressiveness."""
        cfg = self.config

        # Gate clamp
        g_lo, g_hi = cfg.global_gate_mult_clamp
        raw_gate = result["_ab_gate_mult"]
        result["_ab_gate_mult"] = max(g_lo, min(g_hi, raw_gate))
        if result["_ab_gate_mult"] != raw_gate:
            proofs.append(f"V2-CLAMP gate {raw_gate:.3f}->{result['_ab_gate_mult']:.3f}")

        # Size clamp
        s_lo, s_hi = cfg.global_size_mult_clamp
        raw_size = result["_ab_size_mult"]
        result["_ab_size_mult"] = max(s_lo, min(s_hi, raw_size))
        if result["_ab_size_mult"] != raw_size:
            proofs.append(f"V2-CLAMP size {raw_size:.3f}->{result['_ab_size_mult']:.3f}")

        # Confidence clamp (applied to expert weights individually)
        c_lo, c_hi = cfg.global_confidence_mult_clamp
        for k, w in result["_ab_expert_weights"].items():
            clamped = max(c_lo, min(c_hi, w))
            if clamped != w:
                proofs.append(f"V2-CLAMP conf {k} {w:.3f}->{clamped:.3f}")
                result["_ab_expert_weights"][k] = clamped

        # Exec confidence clamp
        raw_ec = result["_ab_exec_confidence"]
        result["_ab_exec_confidence"] = max(c_lo, min(c_hi, raw_ec))
        if result["_ab_exec_confidence"] != raw_ec:
            proofs.append(f"V2-CLAMP exec_conf {raw_ec:.3f}->{result['_ab_exec_confidence']:.3f}")

        return result

    # -----------------------------------------------------------------
    # P0.1: Regime-transition selective aggression
    # -----------------------------------------------------------------

    def _compute_transition(
        self, asset: str, agent_signals: Dict[str, Any],
        market_data: Dict[str, Any], phase_result: Any,
    ) -> Dict[str, Any]:
        cfg = self.config
        regime = agent_signals.get("regime_state", "UNKNOWN").upper()
        tick = self._tick_count.get(asset, 0)

        if asset not in self._transition_state:
            self._transition_state[asset] = TransitionState(
                current_regime=regime, previous_regime=regime, transition_tick=tick
            )
            return {"active": False, "gate_mult": 1.0, "size_mult": 1.0, "proof": ""}

        ts = self._transition_state[asset]

        if regime != ts.current_regime:
            ts.previous_regime = ts.current_regime
            ts.current_regime = regime
            ts.transition_tick = tick
            ts.transition_confirmed = False
            ts.volume_confirmed = False
            ts.phase_confidence = 0.0

        favorable_targets = cfg.favorable_transitions.get(ts.previous_regime, [])
        if regime not in favorable_targets:
            return {"active": False, "gate_mult": 1.0, "size_mult": 1.0, "proof": ""}

        bars_since = tick - ts.transition_tick
        if bars_since > cfg.transition_max_age_bars:
            return {"active": False, "gate_mult": 1.0, "size_mult": 1.0, "proof": ""}

        phase_conf = 0.5
        if phase_result and hasattr(phase_result, 'phase_confidence'):
            phase_conf = phase_result.phase_confidence
        ts.phase_confidence = phase_conf

        if phase_conf < cfg.transition_min_phase_confidence:
            return {
                "active": False, "gate_mult": 1.0, "size_mult": 1.0,
                "proof": f"transition {ts.previous_regime}->{regime} low_conf={phase_conf:.2f}"
            }

        vol_ok = True
        if cfg.transition_require_volume_confirm:
            vol_ratio = market_data.get("volume_ratio_20", 1.0)
            vol_ok = vol_ratio > 1.2
            ts.volume_confirmed = vol_ok
            if not vol_ok:
                return {
                    "active": False, "gate_mult": 1.0, "size_mult": 1.0,
                    "proof": f"transition {ts.previous_regime}->{regime} vol_unconfirmed={vol_ratio:.2f}"
                }

        ts.transition_confirmed = True
        decay = max(0.5, 1.0 - (bars_since * 0.25))
        gate_mult = 1.0 + (cfg.transition_alpha_gate_mult - 1.0) * decay
        size_mult = 1.0 + (cfg.transition_size_boost - 1.0) * decay

        proof = (
            f"P0.1 TRANSITION {ts.previous_regime}->{regime} "
            f"bars={bars_since} decay={decay:.2f} "
            f"gate={gate_mult:.3f} size={size_mult:.3f} "
            f"phase_conf={phase_conf:.2f} vol_ok={vol_ok}"
        )
        return {"active": True, "gate_mult": gate_mult, "size_mult": size_mult, "proof": proof}

    # -----------------------------------------------------------------
    # P0.2: Quant-strategy adaptive confidence (V2: with shrinkage)
    # -----------------------------------------------------------------

    def _compute_expert_gating(
        self, asset: str, agent_signals: Dict[str, Any], confidence_scorer: Any,
    ) -> Dict[str, Any]:
        """
        Quant-strategy confidence adjustment based on realized utility.

        V2 hardening: Bayesian shrinkage toward neutral (1.0).
        At low sample sizes, the weight stays close to 1.0 regardless of
        the confidence score. Only with sufficient data does the weight
        diverge meaningfully.

        Shrinkage factor = (n / min_trades) ^ shrinkage_strength, capped at 1.0
        effective_weight = 1.0 + (raw_weight - 1.0) * shrinkage_factor

        NOTE: This is QUANT-ONLY gating. It adjusts how much the active
        Best-of-N strategy's confidence is trusted. It does NOT gate
        sentiment, microstructure, squeeze, or other auxiliary experts.
        """
        cfg = self.config
        weights = {}
        proofs = []
        regime = agent_signals.get("regime_state", "UNKNOWN").upper()

        for quant_name, scorer_name in cfg.expert_strategy_map.items():
            try:
                sc = confidence_scorer.get_strategy_confidence(scorer_name)
                if sc is None:
                    continue

                conf = sc.confidence  # 0.3 to 1.0

                # Raw weight: linear map [0.3, 1.0] -> [min_weight, max_boost] with 0.65 = neutral
                if conf >= 0.65:
                    raw_w = 1.0 + (conf - 0.65) / (1.0 - 0.65) * (cfg.expert_gating_max_boost - 1.0)
                else:
                    raw_w = cfg.expert_gating_min_weight + (conf - 0.3) / (0.65 - 0.3) * (1.0 - cfg.expert_gating_min_weight)
                raw_w = max(cfg.expert_gating_min_weight, min(cfg.expert_gating_max_boost, raw_w))

                # V2.1: Bayesian shrinkage toward neutral
                # Get approximate trade count from confidence scorer internals
                n_trades = 0
                if hasattr(sc, 'n_signals'):
                    n_trades = sc.n_signals
                elif hasattr(sc, 'total_signals'):
                    n_trades = sc.total_signals
                else:
                    # Fallback: check if confidence_scorer has a way to get count
                    try:
                        report = confidence_scorer.get_confidence_report()
                        strat_report = report.get("strategies", {}).get(scorer_name, {})
                        n_trades = strat_report.get("total_signals", 0)
                    except Exception:
                        n_trades = 0

                min_n = cfg.expert_gating_min_trades
                if n_trades < min_n:
                    # Shrinkage: blend toward 1.0 based on data sufficiency
                    ratio = n_trades / max(min_n, 1)
                    shrinkage_factor = ratio ** cfg.expert_gating_shrinkage_strength
                    effective_w = 1.0 + (raw_w - 1.0) * shrinkage_factor
                    effective_w = max(cfg.expert_gating_min_weight, min(cfg.expert_gating_max_boost, effective_w))

                    if abs(effective_w - 1.0) > 0.02:
                        proofs.append(
                            f"{quant_name}={effective_w:.3f}(raw={raw_w:.2f},n={n_trades}/{min_n},shrink={shrinkage_factor:.2f})"
                        )
                    else:
                        proofs.append(f"{quant_name}=~1.0(n={n_trades}/{min_n},insufficient)")
                    weights[quant_name] = effective_w
                else:
                    weights[quant_name] = raw_w
                    if abs(raw_w - 1.0) > 0.05:
                        proofs.append(f"{quant_name}={raw_w:.3f}(sc={conf:.2f},n={n_trades})")

            except Exception:
                continue

        proof_str = ""
        if proofs:
            proof_str = f"P0.2 QUANT_GATE {regime}: " + ", ".join(proofs)

        return {"weights": weights, "proof": proof_str}

    # -----------------------------------------------------------------
    # P0.3: Execution-to-alpha feedback (V2: with smoothing)
    # -----------------------------------------------------------------

    def record_fill(
        self, asset: str, slippage_bps: float, is_maker: bool, fill_pct: float,
    ) -> None:
        """Record a fill outcome. Called from _execute_intent."""
        if asset not in self._exec_state:
            self._exec_state[asset] = ExecFeedbackState()
        self._exec_state[asset].fills.append({
            "ts": time.time(),
            "slippage_bps": abs(slippage_bps),
            "is_maker": is_maker,
            "fill_pct": fill_pct,
        })

    def _compute_exec_feedback(self, asset: str) -> Dict[str, Any]:
        """
        V2: execution feedback with sample-aware smoothing.

        When n_fills < min_fills, returns neutral. When n_fills >= min_fills,
        adjustments scale linearly from 0% effect at min_fills to 100% at 2*min_fills.
        """
        cfg = self.config
        if asset not in self._exec_state:
            return {"gate_mult": 1.0, "confidence_mult": 1.0, "proof": ""}

        es = self._exec_state[asset]
        if es.n_fills < cfg.exec_feedback_min_fills:
            if es.n_fills > 0:
                return {
                    "gate_mult": 1.0, "confidence_mult": 1.0,
                    "proof": f"P0.3 {asset}: n={es.n_fills}/{cfg.exec_feedback_min_fills} insufficient"
                }
            return {"gate_mult": 1.0, "confidence_mult": 1.0, "proof": ""}

        # V2.1: Ramp-up factor — effects scale in gradually past min_fills
        ramp = min(1.0, (es.n_fills - cfg.exec_feedback_min_fills) / max(cfg.exec_feedback_min_fills, 1))

        gate_mult = 1.0
        conf_mult = 1.0
        parts = []

        # Slippage penalty
        avg_slip = es.avg_slippage_bps
        if avg_slip > cfg.exec_feedback_slippage_penalty_bps:
            excess = avg_slip - cfg.exec_feedback_slippage_penalty_bps
            raw_penalty = min(0.07, excess * 0.01)
            penalty = raw_penalty * ramp  # V2: ramp in
            gate_mult *= (1.0 + penalty)
            conf_mult *= max(0.90, 1.0 - penalty)
            parts.append(f"slip={avg_slip:.1f}bps(+{penalty:.3f}gate,ramp={ramp:.2f})")

        # Maker bonus
        maker_rate = es.maker_rate
        if maker_rate > cfg.exec_feedback_maker_bonus_threshold:
            raw_gate_bonus = 0.03
            gate_mult *= (1.0 - raw_gate_bonus * ramp)
            conf_mult *= (1.0 + (cfg.exec_feedback_maker_bonus - 1.0) * ramp)
            parts.append(f"maker={maker_rate:.0%}(bonus,ramp={ramp:.2f})")
        elif maker_rate < 0.40:
            gate_mult *= (1.0 + 0.02 * ramp)
            parts.append(f"maker={maker_rate:.0%}(penalty,ramp={ramp:.2f})")

        # Fill completeness
        avg_fill = es.avg_fill_pct
        if avg_fill < 0.70:
            conf_mult *= max(0.85, 1.0 - (1.0 - avg_fill) * 0.3 * ramp)
            parts.append(f"fill={avg_fill:.0%}(low)")

        # Component clamp
        lo, hi = cfg.exec_feedback_gate_mult_range
        gate_mult = max(lo, min(hi, gate_mult))
        conf_mult = max(0.85, min(1.10, conf_mult))

        proof = ""
        if parts:
            proof = f"P0.3 EXEC_FB {asset}: " + ", ".join(parts) + f" ->gate={gate_mult:.3f} conf={conf_mult:.3f} n={es.n_fills}"

        return {"gate_mult": gate_mult, "confidence_mult": conf_mult, "proof": proof}

    # -----------------------------------------------------------------
    # V2.3: Mode-isolated persistence
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize state with mode prefix for isolation."""
        return {
            "mode": self._mode,
            "version": "2.0",
            "transition_state": {
                k: {
                    "previous_regime": v.previous_regime,
                    "current_regime": v.current_regime,
                    "transition_tick": v.transition_tick,
                }
                for k, v in self._transition_state.items()
            },
            "exec_state": {
                k: {"fills": list(v.fills)}
                for k, v in self._exec_state.items()
            },
            "tick_count": dict(self._tick_count),
        }

    def from_dict(self, data: dict) -> None:
        """Restore state only if mode matches. V2.3: mode isolation."""
        if not data:
            return

        saved_mode = data.get("mode", "paper")
        if saved_mode != self._mode:
            logger.info(
                f"[AlphaBoost] Ignoring saved state: mode mismatch "
                f"(saved={saved_mode}, current={self._mode})"
            )
            return

        for k, v in data.get("transition_state", {}).items():
            self._transition_state[k] = TransitionState(
                previous_regime=v.get("previous_regime", ""),
                current_regime=v.get("current_regime", ""),
                transition_tick=v.get("transition_tick", 0),
            )
        for k, v in data.get("exec_state", {}).items():
            es = ExecFeedbackState()
            for f in v.get("fills", []):
                es.fills.append(f)
            self._exec_state[k] = es
        self._tick_count = data.get("tick_count", {})


# =============================================================================
# SINGLETON (mode-aware)
# =============================================================================

_instances: Dict[str, AlphaBoostController] = {}


def get_alpha_boost(
    config: Optional[AlphaBoostConfig] = None, mode: str = "paper",
) -> AlphaBoostController:
    """Get mode-isolated AlphaBoost instance."""
    global _instances
    if mode not in _instances:
        _instances[mode] = AlphaBoostController(config, mode=mode)
    return _instances[mode]


def reset_alpha_boost() -> None:
    global _instances
    _instances = {}

"""
================================================================================
HMATS v3.4 - AUTHORITY-BASED FUSION ENGINE
================================================================================
Version: 3.4.0
Upgrade From: v3.3-HR (which used weighted/consensus with overrides)
Purpose: Replace weighted fusion with pure authority-based decision making

KEY CHANGE FROM v3.3-HR:
    v3.3-HR: Weighted fusion with conditional overrides
    v3.4:    Pure authority-based with no weights
    
    Weights DILUTE conviction. Authority is BINARY.

AUTHORITY TYPES:
    DECIDE   - Has final say on this aspect
    CONFIRM  - Must align, else reduce size
    ADVISE   - Provides input but no veto power
    VETO     - Can block decisions
    CAP      - Sets upper limits only
    TRIGGER  - Can amplify but not reduce
    EXECUTE  - Controls execution only

PROFITABILITY IMPACT:
    - Full conviction when appropriate (no dilution)
    - OPPORTUNITY truly aggressive (Regime+Lead-Lag own timing)
    - DRL cannot veto entry (only advises on tranche timing)
    - Expected: +20% larger average position in winning trades
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from enum import Enum

from market.phase_detector import RegimePhase, RegimePhaseResult
from core.canonical_enums import (
    SystemMode,
    Authority,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """Signal from an agent."""
    direction: float = 0.0      # -1 to 1
    confidence: float = 0.0     # 0 to 1
    veto_active: bool = False
    veto_direction: Optional[str] = None  # "long" or "short"
    leverage_cap: float = 1.0
    exit_pressure: float = 0.0  # 0 to 1
    tranche_advice: Optional[str] = None  # ESCALATE, HOLD, REDUCE
    asof_timestamp: float = 0.0  # [v3.3-B9] Unix timestamp when signal was generated

    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "veto_active": self.veto_active,
            "leverage_cap": self.leverage_cap,
        }


@dataclass
class FusionContext:
    """Context for fusion decision."""
    mode: SystemMode
    regime_phase: RegimePhase
    phase_result: Optional[RegimePhaseResult] = None
    crack_active: bool = False
    crack_weight: float = 0.0
    data_valid: bool = True
    drl_enabled: bool = False
    lead_lag_confident: bool = False
    lead_lag_edge: float = 0.0
    regime: str = "UNKNOWN"  # [FIX-32] GMM regime label for ADVISE influence layer (was regime_name)
    htf_trend_direction: int = 0  # [S11] 1D trend: -1/0/+1 (0 = no data)


@dataclass
class FusionResult:
    """Result of authority fusion."""
    
    # Decision
    direction: float = 0.0          # -1 to 1
    target_exposure: float = 0.0    # 0 to 1 (of max)
    execution_mode: str = "PASSIVE_PREFERRED"
    
    # Tranche guidance
    tranche_target: int = 0         # 0-4
    allow_escalation: bool = False
    max_tranche_tier: int = 4       # Maximum tranche tier allowed (1-4)
    
    # Execution
    urgency: float = 0.5
    delay_allowed: bool = True

    # [FIX-4] Fusion confidence - independent of caps/vetoes
    confidence: float = 0.5
    
    # Authority tracking
    decider_agent: str = ""
    authority_matrix_used: str = ""
    # [FIX 2026-04-22] Primary agent for per-agent PnL attribution.
    # Populated by consensus() when multiple DECIDE agents contribute.
    # Empty string = single-decider or no decider.
    primary_agent: str = ""
    
    # Vetoes/Caps applied
    vetoes_active: List[str] = field(default_factory=list)
    caps_applied: Dict[str, float] = field(default_factory=dict)
    
    # v6.2.3: Partial Consensus Entry
    is_partial_consensus: bool = False
    partial_consensus_scale: float = 1.0
    partial_consensus_aligned: List[str] = field(default_factory=list)
    partial_consensus_flags: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "target_exposure": self.target_exposure,
            "confidence": self.confidence,
            "execution_mode": self.execution_mode,
            "tranche_target": self.tranche_target,
            "allow_escalation": self.allow_escalation,
            "max_tranche_tier": self.max_tranche_tier,
            "decider_agent": self.decider_agent,
            "vetoes_active": self.vetoes_active,
            "is_partial_consensus": self.is_partial_consensus,
            "partial_consensus_scale": self.partial_consensus_scale,
            "partial_consensus_aligned": self.partial_consensus_aligned,
        }


# ============================================================================
# AUTHORITY MATRICES BY MODE
# ============================================================================

AUTHORITY_MATRIX_NORMAL = {
    "quant": Authority.DECIDE,      # Primary direction authority
    "regime": Authority.CONFIRM,    # Must align
    "drl": Authority.ADVISE,        # Default: advisory only (upgraded to DECIDE when ACTIVE)
    "sentiment": Authority.ADVISE,   # [FIX-H1] Was CONFIRM (halved exposure on misalign). Crypto sentiment too noisy for 50% penalty. Moves to ADVISE influence layer (±15% max).
    "macro": Authority.CAP,         # Leverage ceiling
    "lead_lag": Authority.EXECUTE,  # Execution mode only
    "risk": Authority.VETO,         # Hard override
    # model_alpha: wired as ADVISE (line 165), promoted from CONTEXT 2026-04-11
    "two_stage": Authority.CONFIRM,   # V6: Two-Stage prior must confirm
    "short_bias": Authority.ADVISE,   # v7.0: Soft advisory (was CONFIRM, halving longs - too aggressive per PENALIZE×0.7 intent)
    # [WIRE-DERIV 2026-04-24] ShortBias applies to ALL execution routes (spot,
    # margin, derivatives). v2.0 Part 2.4 proposed route-specific penalty
    # (skip on DERIVATIVES), but HMATS runs fusion BEFORE routing, so penalty
    # is already baked in by the time ExecutionRouter decides. Given ADVISE
    # authority (soft ±15% confidence modulation, not veto), no functional
    # restoration is needed for derivatives longs — the penalty is small enough
    # that perp longs with sufficient edge still fire.
    "funding_rate": Authority.ADVISE,  # [v9-PATCH-8] Carry-bias from funding rates
    "onchain": Authority.ADVISE,       # OnChain intelligence (Helius/Birdeye)
    "llm_sentiment": Authority.ADVISE, # LLM sentiment (Haiku)
    "flow": Authority.ADVISE,          # Whale/exchange/ETF flow
    "structure": Authority.CONFIRM,     # [FIX-L1-07] OFI/orderbook structure confirmation
    "squeeze": Authority.ADVISE,        # [FIX-L1-07] Squeeze risk detection
    "cvd": Authority.ADVISE,            # [FIX-L1-07] CVD divergence signal
    "risk_appetite": Authority.ADVISE,  # [FIX-L1-07] Macro risk appetite
    # [FIX-GHOST] Previously ghost agents now wired
    # [PROMOTE 2026-04-22] kraken_quant ADVISE -> DECIDE. The 12-strategy institutional
    # matrix (Hurst, Shannon entropy, Kalman/ETF-spot cointegration, OB imbalance,
    # Ornstein-Uhlenbeck, dark-pool volume, delta-neutral funding, etc.) is
    # structurally more sophisticated than the 5 TA-based Best-of-N strategies
    # powering the legacy "quant" DECIDE slot. Prior ADVISE + fusion 0.5 dampening
    # meant 14d of kraken_quant output never actually swayed direction. Both
    # restrictions removed simultaneously today.
    "kraken_quant": Authority.DECIDE,
    "microstructure": Authority.ADVISE, # OB imbalance + taker flow
    "model_alpha": Authority.ADVISE,    # LSTM/GRU ensemble
    "onchain_graph": Authority.ADVISE,  # Whale/cluster on-chain
    "options": Authority.ADVISE,        # Put/call + max pain
    "vol_alpha": Authority.ADVISE,      # Vol compression/expansion
    "whale": Authority.ADVISE,          # Large order detection
    # [SOLDEX-AUTHORITY 2026-04-15] Promoted from SHADOW: SolDex monitor exposes
    # soldex_arb_direction / soldex_arb_strength / soldex_liquidity_score / soldex_arb_active
    # via main.py:6022-6051. Without this matrix entry, fusion silently dropped them.
    # ADVISE-only: SolDex DEX-CEX arb is opportunistic, never overrides quant/DRL.
    "soldex": Authority.ADVISE,
}

AUTHORITY_MATRIX_OPPORTUNITY = {
    "quant": Authority.CONFIRM,     # Direction alignment only
    "regime": Authority.DECIDE,     # Entry timing authority
    "drl": Authority.ADVISE,        # Default: advisory (upgraded to DECIDE when ACTIVE)
    "sentiment": Authority.TRIGGER, # Can amplify, not reduce
    "macro": Authority.CAP,         # Relaxed ceiling
    "lead_lag": Authority.EXECUTE,  # Execution timing only; never direction authority
    "risk": Authority.VETO,         # Still hard override
    # "model_alpha": Authority.CONFIRM,  # NOT WIRED - agent exists but not initialized in main.py
    "two_stage": Authority.TRIGGER,   # V6: Two-Stage can amplify in OPPORTUNITY
    "short_bias": Authority.TRIGGER,  # V6.2.3: Short-bias can amplify SHORT, still vetoes LONG
    "funding_rate": Authority.CONFIRM,  # [v9-PATCH-8] Carry-bias confirms in OPPORTUNITY
    "onchain": Authority.ADVISE,       # OnChain intelligence
    "llm_sentiment": Authority.ADVISE, # LLM sentiment
    "flow": Authority.ADVISE,          # Whale/exchange/ETF flow
    "structure": Authority.CONFIRM,     # [FIX-L1-07] OFI/orderbook structure confirmation
    "squeeze": Authority.ADVISE,        # [FIX-L1-07] Squeeze risk detection
    "cvd": Authority.ADVISE,            # [FIX-L1-07] CVD divergence signal
    "risk_appetite": Authority.ADVISE,  # [FIX-L1-07] Macro risk appetite
    # [FIX-GHOST] Previously ghost agents now wired
    # [PROMOTE 2026-04-22] kraken_quant DECIDE in OPPORTUNITY too (consistency)
    "kraken_quant": Authority.DECIDE,
    "microstructure": Authority.ADVISE, # OB imbalance + taker flow
    "model_alpha": Authority.ADVISE,    # LSTM/GRU ensemble
    "onchain_graph": Authority.ADVISE,  # Whale/cluster on-chain
    "options": Authority.ADVISE,        # Put/call + max pain
    "vol_alpha": Authority.ADVISE,      # Vol compression/expansion
    "whale": Authority.ADVISE,          # Large order detection
    # [SOLDEX-AUTHORITY 2026-04-15] OPPORTUNITY mode: SolDex can TRIGGER
    # opportunity entries (DEX-CEX arb is by definition an opportunity).
    "soldex": Authority.TRIGGER,
}

# v6.7: DRL authority level (module-level, set by main.py at startup)
_drl_authority_level = "DISABLED"

def set_drl_authority_level(level: str):
    """Set DRL authority level. Called from main.py when promotion gate updates."""
    global _drl_authority_level
    _drl_authority_level = level

# Escalation profile (module-level, set by main.py at startup)
_escalation_profile = "default"  # "default" | "aggressive"

def set_escalation_profile(profile: str):
    """Set escalation profile. Called from main.py on startup."""
    global _escalation_profile
    _escalation_profile = profile

def get_authority_matrix(mode: str) -> dict:
    """Get authority matrix with DRL authority applied dynamically."""
    if mode == "NO_TRADE":
        return dict(AUTHORITY_MATRIX_NO_TRADE)
    elif mode == "OPPORTUNITY":
        matrix = dict(AUTHORITY_MATRIX_OPPORTUNITY)
    else:
        matrix = dict(AUTHORITY_MATRIX_NORMAL)

    # v6.7: Upgrade DRL authority when ACTIVE
    if _drl_authority_level == "ACTIVE":
        matrix["drl"] = Authority.DECIDE
    return matrix

# --- [v9-PATCH-11a] High-volatility authority matrix ---
AUTHORITY_MATRIX_HIGH_VOL = {
    "quant": Authority.ADVISE,       # downgrade from DECIDE in high vol
    "regime": Authority.DECIDE,      # regime drives decisions
    "drl": Authority.ADVISE,         # advisory only
    "sentiment": Authority.ADVISE,   # advisory only
    "macro": Authority.CAP,          # still cap leverage
    "lead_lag": Authority.EXECUTE,   # execution mode only
    "risk": Authority.VETO,          # hard override
    # model_alpha: wired as ADVISE (line 165), promoted from CONTEXT 2026-04-11
    "two_stage": Authority.ADVISE,
    "short_bias": Authority.ADVISE,
    "funding_rate": Authority.ADVISE,
    "onchain": Authority.ADVISE,
    "llm_sentiment": Authority.ADVISE,
    "flow": Authority.ADVISE,
    "structure": Authority.ADVISE,      # [FIX-L1-07] downgraded in high vol
    "squeeze": Authority.ADVISE,        # [FIX-L1-07]
    "cvd": Authority.ADVISE,            # [FIX-L1-07]
    "risk_appetite": Authority.ADVISE,  # [FIX-L1-07]
    # [FIX-GHOST]
    "kraken_quant": Authority.ADVISE,
    "microstructure": Authority.ADVISE,
    "model_alpha": Authority.ADVISE,
    "onchain_graph": Authority.ADVISE,
    "options": Authority.ADVISE,
    "vol_alpha": Authority.ADVISE,
    "whale": Authority.ADVISE,
}
# --- end [v9-PATCH-11a] ---

AUTHORITY_MATRIX_NO_TRADE = {
    "quant": Authority.NONE,
    "regime": Authority.NONE,
    "drl": Authority.NONE,
    "sentiment": Authority.NONE,
    "macro": Authority.NONE,
    "lead_lag": Authority.NONE,
    "risk": Authority.ABSOLUTE,     # All decisions
    # "model_alpha": Authority.NONE,  # NOT WIRED - agent exists but not initialized in main.py
    "two_stage": Authority.NONE,      # V6: Two-Stage suspended in NO_TRADE
    "short_bias": Authority.NONE,     # V6.2.3: Short-bias suspended in NO_TRADE
    "funding_rate": Authority.ADVISE,  # [v9-PATCH-8] Advisory only in NO_TRADE
    "onchain": Authority.NONE,
    "llm_sentiment": Authority.NONE,
    "flow": Authority.NONE,
    "structure": Authority.NONE,        # [FIX-L1-07]
    "squeeze": Authority.NONE,          # [FIX-L1-07]
    "cvd": Authority.NONE,              # [FIX-L1-07]
    "risk_appetite": Authority.NONE,    # [FIX-L1-07]
}


# ============================================================================
# REGIME-CONDITIONAL ADVISE INFLUENCE WEIGHTS (v6.7-P1)
# ============================================================================
# ADVISE agents collectively modulate base_exposure by up to ±MAX_ADVISE_INFLUENCE.
# Each agent's contribution = weight × confidence × direction_alignment.
# Weights are regime-conditional: different regimes trust different ADVISE agents.
#
# NOTE: onchain weight only applies to SOL. For BTC/ETH it's redistributed.

MAX_ADVISE_INFLUENCE = 0.20  # [FIX-M6] Was 0.15. Raised to ±20% after sentiment moved from CONFIRM to ADVISE (FIX-H1). Consensus boost can take it to ±25%, hard cap 0.35.

ADVISE_WEIGHTS_BY_REGIME = {
    # [FIX-M6] Added "sentiment" key (moved from CONFIRM to ADVISE per FIX-H1).
    # Weights redistributed: sentiment gets 0.10 base, taken from llm_sentiment.
    "default": {
        "short_bias": 0.25,
        "funding_rate": 0.20,
        "onchain": 0.15,
        "llm_sentiment": 0.10,
        "sentiment": 0.10,
        "flow": 0.20,
    },
    "PANIC_SELLOFF": {
        "short_bias": 0.35,
        "funding_rate": 0.15,
        "onchain": 0.20,
        "llm_sentiment": 0.05,
        "sentiment": 0.10,
        "flow": 0.15,
    },
    "MOMENTUM_RALLY": {
        "short_bias": 0.15,
        "funding_rate": 0.20,
        "onchain": 0.15,
        "llm_sentiment": 0.15,
        "sentiment": 0.10,
        "flow": 0.25,
    },
    "VOLATILE_CHOP": {
        "short_bias": 0.15,
        "funding_rate": 0.15,
        "onchain": 0.15,
        "llm_sentiment": 0.10,
        "sentiment": 0.05,
        "flow": 0.40,
    },
    "EXTREME_VOLATILITY": {
        "short_bias": 0.20,
        "funding_rate": 0.25,
        "onchain": 0.10,
        "llm_sentiment": 0.05,
        "sentiment": 0.10,
        "flow": 0.30,
    },
}


class AuthorityFusionEngine:
    """
    Authority-based fusion engine.

    KEY DESIGN: NO WEIGHTS
        v3.3-HR: final = 0.5×quant + 0.3×regime + 0.2×drl
        v3.4:    if mode == NORMAL: decider = quant
                 if mode == OPPORTUNITY: decider = regime
    
    WHY THIS IMPROVES PROFITABILITY:
        1. Full conviction on strong signals (no dilution)
        2. OPPORTUNITY is truly aggressive (regime owns timing)
        3. Clear decision responsibility (no confusion)
    """
    
    def __init__(self):
        self._momentum_memory = MomentumMemory()
        # --- [v9-PATCH-6] Confidence scorer for reliability injection ---
        self._confidence_scorer = None
        # --- end [v9-PATCH-6] ---
        logger.info("AuthorityFusionEngine v3.4 initialized (authority-based, no weights)")

    # --- [v9-PATCH-6] Setter for confidence scorer ---
    def set_confidence_scorer(self, scorer):
        """Set confidence scorer for reliability-based authority adjustment."""
        self._confidence_scorer = scorer
        logger.info("[v9-PATCH-6] confidence_scorer connected to AuthorityFusionEngine")
    # --- end [v9-PATCH-6] ---
    
    def fuse(
        self,
        signals: Dict[str, AgentSignal],
        context: FusionContext,
    ) -> FusionResult:
        """
        Fuse agent signals using authority matrix.
        
        FLOW:
            1. Get authority matrix for current mode
            2. Check VETO/ABSOLUTE authorities
            3. Get DECIDE authority signal
            4. Check CONFIRM alignments
            5. Apply CAP limits
            6. Return final decision
        """
        
        # Get authority matrix for current mode
        matrix = self._get_authority_matrix(context.mode)
        matrix_name = f"MATRIX_{context.mode.name}"
        
        # =====================================================================
        # LAYER 1: HARD STATE CHECK
        # =====================================================================
        
        # NO_TRADE mode -> Risk has ABSOLUTE authority
        if context.mode == SystemMode.NO_TRADE:
            logger.warning("NO_TRADE mode: Risk has ABSOLUTE authority")
            return FusionResult(
                direction=0.0,
                target_exposure=0.0,
                decider_agent="risk",
                authority_matrix_used=matrix_name,
                vetoes_active=["NO_TRADE_MODE"],
            )
        
        # Data validity check
        if not context.data_valid:
            logger.warning("Data invalid: Holding current position")
            return FusionResult(
                direction=0.0,
                target_exposure=0.0,  # Hold, don't add
                decider_agent="system",
                authority_matrix_used=matrix_name,
                vetoes_active=["DATA_INVALID"],
            )
        
        # =====================================================================
        # LAYER 2: VETO CHECK
        # =====================================================================
        
        result = FusionResult(authority_matrix_used=matrix_name)
        vetoes = []
        
        for agent, authority in matrix.items():
            if authority == Authority.VETO:
                if agent not in signals:
                    # [FIX-C1] Missing VETO agent = fail-closed. Log critical warning.
                    logger.critical(
                        f"[FUSION] VETO agent '{agent}' MISSING from signals — "
                        f"fail-closed: treating as active veto"
                    )
                    return FusionResult(
                        direction=0.0,
                        target_exposure=0.0,
                        decider_agent=f"missing_veto_{agent}",
                        authority_matrix_used=matrix_name,
                        vetoes_active=[f"missing_veto:{agent}"],
                    )
                signal = signals[agent]
                if signal.veto_active:
                    vetoes.append(f"{agent}:{signal.veto_direction}")
                    
                    # Risk VETO is absolute
                    if agent == "risk":
                        logger.warning(f"RISK VETO: {signal.veto_direction}")
                        return FusionResult(
                            direction=0.0,
                            target_exposure=0.0,
                            decider_agent="risk",
                            authority_matrix_used=matrix_name,
                            vetoes_active=[f"risk_veto:{signal.veto_direction}"],
                        )
        
        result.vetoes_active = vetoes
        
        # =====================================================================
        # LAYER 3: GET DECIDER SIGNAL (with multi-DECIDE conflict resolution)
        # =====================================================================

        decide_agents = [
            (agent, signals.get(agent, AgentSignal()))
            for agent, authority in matrix.items()
            if authority == Authority.DECIDE
        ]

        # [DIAG P21] Periodic visibility into DECIDE pool membership. Every 10th call
        # logs which agents are currently in DECIDE, so we can tell from a live log
        # whether DRL is being upgraded correctly.
        if not hasattr(self, "_decide_pool_diag_counter"):
            self._decide_pool_diag_counter = 0
        self._decide_pool_diag_counter += 1
        if self._decide_pool_diag_counter % 10 == 1:
            _drl_in = any(a == "drl" for a, _ in decide_agents)
            logger.info(
                f"[DECIDE_POOL] matrix={matrix_name} drl_authority={_drl_authority_level} "
                f"agents={[a for a, _ in decide_agents]} (drl_in_pool={_drl_in})"
            )

        if len(decide_agents) == 0:
            # [FIX-H1] Fallback to quant — but validate it exists
            decider_agent = "quant"
            decider_signal = signals.get("quant", AgentSignal())
            if abs(decider_signal.direction) < 0.001 and decider_signal.confidence < 0.01:
                logger.warning(
                    f"[FUSION] No DECIDE agents and quant signal absent/zero — "
                    f"returning NO_TRADE (fail-closed)"
                )
                return FusionResult(
                    direction=0.0,
                    target_exposure=0.0,
                    decider_agent="no_decider",
                    authority_matrix_used=matrix_name,
                )
        elif len(decide_agents) == 1:
            # Single decider - no conflict
            decider_agent = decide_agents[0][0]
            decider_signal = decide_agents[0][1]
        else:
            # [L3-04 + FUSION-V2 2026-04-25] Multiple DECIDE agents.
            # Resolution combines three changes from CLAUDE.md P30 (Bayesian
            # Model Averaging alignment):
            #
            #   FIX-1: ACTIVE-ONLY AGREEMENT — agents with |dir|<0.01 are
            #     treated as ABSTAINING, not as DISAGREEING. They drop from
            #     the agreement denominator entirely. Aligns with Black-
            #     Litterman: "no view = no contribution".
            #   FIX-2: SOLO-CONVICTION PASSTHROUGH — when exactly 1 agent
            #     votes among the active pool AND its confidence >= 0.5,
            #     treat as consensus rather than minority conflict. Protects
            #     the case where only DRL has regime-relevant info during
            #     transitions where TA hasn't caught up yet.
            #   FIX-3: CONVICTION-SQUARED WEIGHTING — w = confidence² instead
            #     of confidence. Inverse-variance weighting per BMA / Kalman
            #     optimum: under Gaussian noise, weight ∝ 1/var ∝ confidence².
            #     High-conviction agents dominate more decisively (a 0.9-conf
            #     agent contributes ~9× a 0.3-conf one, vs. 3× under linear).
            _SOLO_CONVICTION_THRESHOLD = 0.5
            total_weight = 0.0
            weighted_direction = 0.0
            best_agent = decide_agents[0][0]
            best_conf = -1.0

            for agent_name, sig in decide_agents:
                # FIX-3: confidence-squared weighting (inverse-variance optimum)
                _conf_floor = max(sig.confidence, 0.01)  # floor to avoid div-by-zero
                w = _conf_floor * _conf_floor
                weighted_direction += sig.direction * w
                total_weight += w
                if sig.confidence > best_conf:
                    best_conf = sig.confidence
                    best_agent = agent_name

            if total_weight > 0:
                final_direction = weighted_direction / total_weight
            else:
                final_direction = 0.0

            # [FIX-H2] Clamp direction to [-1, 1]
            final_direction = max(-1.0, min(1.0, final_direction))

            # [FIX-H2 + FIX-1 2026-04-25] Measure agreement among ACTIVE
            # agents only (those with non-trivial direction). Abstainers
            # don't dilute the agreement ratio — they have no view, so
            # they don't count for or against. Avoids the failure mode
            # where DRL=-0.93 + 2 abstainers gets dampened to ~10%
            # confidence as if it were a 1-of-3 minority opinion.
            _directions = [sig.direction for _, sig in decide_agents]
            _n_pos = sum(1 for d in _directions if d > 0.01)
            _n_neg = sum(1 for d in _directions if d < -0.01)
            _n_active = _n_pos + _n_neg
            if _n_active > 0:
                direction_agreement = max(_n_pos, _n_neg) / _n_active
            else:
                direction_agreement = 0.0  # nobody voted; no consensus to measure
            avg_conf = total_weight / len(decide_agents)
            # avg_conf is built from squared weights so equals avg(confidence²).
            # Take its sqrt to get a comparable "average confidence" for the
            # downstream alpha gate (which expects [0, 1] confidence units).
            try:
                avg_conf = avg_conf ** 0.5
            except Exception:
                avg_conf = 0.0

            # FIX-2: solo-conviction passthrough
            _solo_conviction = (
                _n_active == 1
                and best_conf >= _SOLO_CONVICTION_THRESHOLD
            )
            if _solo_conviction:
                merged_confidence = best_conf
                logger.info(
                    f"[DECIDE_SOLO] {best_agent}: solo high-conviction signal "
                    f"(dir={final_direction:+.3f} conf={best_conf:.3f}) — "
                    f"treated as consensus among {len(decide_agents)} agents "
                    f"({_n_active} active, {len(decide_agents) - _n_active} abstaining)"
                )
            elif _n_active == 0:
                # All agents abstain — no signal at all.
                merged_confidence = 0.0
                logger.info(
                    f"[DECIDE_ABSTAIN] all {len(decide_agents)} DECIDE agents "
                    f"abstain (|dir|<0.01) — fused confidence=0"
                )
            elif direction_agreement < 0.6:
                # Genuine disagreement among active agents — dampen confidence.
                merged_confidence = avg_conf * direction_agreement
                logger.info(
                    f"[DECIDE_CONFLICT] {len(decide_agents)} agents "
                    f"({_n_active} active), low agreement "
                    f"({direction_agreement:.2f}), confidence={merged_confidence:.3f} - "
                    f"agents: {[a for a, _ in decide_agents]}"
                )
            else:
                merged_confidence = avg_conf

            # Build merged signal
            decider_agent = f"consensus({','.join(a for a, _ in decide_agents)})"
            decider_signal = AgentSignal(
                direction=final_direction,
                confidence=merged_confidence,
            )
            logger.info(
                f"[DECIDE_CONSENSUS] {decider_agent}: dir={final_direction:+.3f} "
                f"conf={merged_confidence:.3f} (primary={best_agent}, "
                f"active={_n_active}/{len(decide_agents)}, agreement={direction_agreement:.2f})"
            )

        result.decider_agent = decider_agent
        # [FIX 2026-04-22] Propagate primary agent for PnL attribution
        result.primary_agent = best_agent if 'best_agent' in locals() else decider_agent
        result.direction = decider_signal.direction
        
        # [FIX-L1] Base exposure from confidence, with configurable scale factor
        _conf_to_exp_factor = getattr(self, 'confidence_to_exposure_factor', 1.0)
        base_exposure = decider_signal.confidence * _conf_to_exp_factor

        # --- [v9-PATCH-6] Reliability-based authority downgrade ---
        _PATCH_6_ACTIVE = True  # [FIX-L1-03] Activated: confidence-based authority adjustment
        try:
            if self._confidence_scorer:
                _strat_conf = self._confidence_scorer.get_all_confidences()
                # Try to get strategy name from quant signal
                _strategy_name = getattr(signals.get("quant", AgentSignal()), 'strategy_name', None)
                if _strategy_name and _strategy_name in _strat_conf:
                    _current_conf = _strat_conf[_strategy_name]

                    if _current_conf < 0.35:
                        # [FIX-H5] Was 0.3x (70% reduction) — too nuclear. Reduced to 0.6x (40% max).
                        logger.info(
                            f"[v9-PATCH-6] RELIABILITY_DOWNGRADE: {_strategy_name} "
                            f"conf={_current_conf:.2f} -> exposure x0.60"
                        )
                        if _PATCH_6_ACTIVE:
                            base_exposure *= 0.6
                    elif _current_conf < 0.50:
                        # [FIX-H5] Was ×conf (up to 50% reduction). Now max 15% reduction.
                        _p6_mult = 0.85 + (_current_conf * 0.30)  # 0.85-1.0x range
                        logger.info(
                            f"[v9-PATCH-6] RELIABILITY_DAMPEN: {_strategy_name} "
                            f"conf={_current_conf:.2f} -> exposure x{_p6_mult:.2f}"
                        )
                        if _PATCH_6_ACTIVE:
                            base_exposure *= _p6_mult
        except Exception as e:
            logger.debug(f"[v9-PATCH-6] Reliability injection skipped: {e}")
        # --- end [v9-PATCH-6] ---

        # =====================================================================
        # LAYER 4: CONFIRM CHECK
        # =====================================================================
        
        confirm_misaligned = False  # Track if opposition detected
        
        for agent, authority in matrix.items():
            if authority == Authority.CONFIRM and agent in signals:
                confirm_signal = signals[agent]

                # Check direction alignment
                if decider_signal.direction != 0 and confirm_signal.direction != 0:
                    same_direction = (
                        (decider_signal.direction > 0 and confirm_signal.direction > 0) or
                        (decider_signal.direction < 0 and confirm_signal.direction < 0)
                    )

                    if not same_direction:
                        # Reduce size by 50% if CONFIRM doesn't align
                        base_exposure *= 0.5
                        confirm_misaligned = True
                        logger.info(f"{agent} CONFIRM misaligned: reducing exposure 50%")
                elif decider_signal.direction != 0 and confirm_signal.direction == 0:
                    # [FIX-H3] CONFIRM neutral: confidence-weighted penalty (was fixed 25%).
                    # High-confidence neutral (confident "no opinion") should penalize less.
                    _neutral_conf = getattr(confirm_signal, 'confidence', 0.5)
                    _neutral_penalty = 0.85 + (0.15 * (1.0 - _neutral_conf))  # 0.85-1.0x
                    base_exposure *= _neutral_penalty
                    logger.info(f"{agent} CONFIRM neutral (dir=0, conf={_neutral_conf:.2f}): exposure x{_neutral_penalty:.2f}")
        
        # =====================================================================
        # LAYER 4.25: HTF CONFIRM MODIFIER (S11)
        # =====================================================================
        # If 4H signal direction opposes 1D trend, dampen confirm_strength.
        # If aligned, slight boost.  Fail-closed: no 1D data -> no effect.

        try:
            _htf_dir = self._get_htf_direction(context)
            if _htf_dir != 0 and decider_signal.direction != 0:
                if _htf_dir * decider_signal.direction < 0:
                    # [FIX-H7] Regime-aware HTF dampen. Choppy regimes = 1D trend is meaningless.
                    _is_choppy = getattr(context, 'regime', '') in (
                        "VOLATILE_CHOP", "EXTREME_VOLATILITY", "WEAK_CONSOLIDATION",
                    )
                    _htf_dampen = 0.85 if _is_choppy else 0.70
                    base_exposure *= _htf_dampen
                    logger.info(
                        f"[HTF] direction={_htf_dir:+d}, signal={decider_signal.direction:+.2f}, "
                        f"MISALIGN -> exposure *= {_htf_dampen} (choppy={_is_choppy})"
                    )
                elif _htf_dir * decider_signal.direction > 0:
                    # Aligned with daily trend -> slight boost
                    base_exposure = min(base_exposure * 1.1, 1.0)
                    logger.info(
                        f"[HTF] direction={_htf_dir:+d}, signal={decider_signal.direction:+.2f}, "
                        f"ALIGNED -> exposure *= 1.1"
                    )
        except Exception as _htf_err:
            logger.debug(f"[HTF] Skipped: {_htf_err}")

        # =====================================================================
        # LAYER 4.5: PARTIAL CONSENSUS CHECK (v6.2.3)
        # =====================================================================
        # Only applies when:
        # - No opposition (LAYER 4 didn't trigger)
        # - Some CONFIRM agents are neutral (direction≈0)
        # - At least one core agent aligned
        # - Disable conditions not active
        
        if not confirm_misaligned and abs(decider_signal.direction) > 0.1:
            try:
                from signals.partial_consensus import (
                    get_partial_consensus_checker,
                    DisableConditions,
                )
                
                checker = get_partial_consensus_checker()
                
                # Build disable conditions from actual system state (fail-closed)
                disable_conditions = DisableConditions()
                
                # Check correlation control state
                try:
                    from risk.global_exposure_cap import get_exposure_cap_manager
                    exposure_cap = get_exposure_cap_manager()
                    cap_level = exposure_cap.state.cap_level
                    disable_conditions.correlation_cap_level = cap_level.value if hasattr(cap_level, 'value') else str(cap_level)
                except (ImportError, Exception):
                    pass  # Use default NORMAL
                
                # Check cascade exhaustion state
                try:
                    from risk.cascade_exhaustion_governor import get_cascade_exhaustion_governor
                    cascade_gov = get_cascade_exhaustion_governor()
                    status = cascade_gov.get_status()
                    disable_conditions.cascade_phase = status.get("phase", "NONE")
                except (ImportError, Exception):
                    pass  # Use default NONE
                
                # Check DVOL state (from trade gate)
                try:
                    from defense.trade_gate import get_trade_gate
                    trade_gate = get_trade_gate()
                    if trade_gate and trade_gate._dvol_module:
                        disable_conditions.dvol_zscore = trade_gate._dvol_module.dvol_zscore
                        disable_conditions.dvol_warning_threshold = trade_gate.config.dvol_zscore_warning
                except (ImportError, Exception):
                    pass  # Use default 0.0
                
                # Check partial consensus
                pc_result = checker.check(
                    decider_agent=decider_agent,
                    decider_direction=decider_signal.direction,
                    decider_confidence=decider_signal.confidence,
                    signals=signals,
                    authority_matrix=matrix,
                    disable_conditions=disable_conditions,
                )
                
                if pc_result.is_partial_consensus:
                    # Apply partial consensus scale
                    base_exposure *= pc_result.scale_factor
                    
                    # Update result fields
                    result.is_partial_consensus = True
                    result.partial_consensus_scale = pc_result.scale_factor
                    result.partial_consensus_aligned = pc_result.aligned_agents
                    result.partial_consensus_flags = pc_result.flags
                    
                    # [FIX-H6] Was unconditional block. Now only block on severe fragmentation.
                    # Moderate partial consensus (scale ≥ 0.50) can still escalate if regime confirms later.
                    if pc_result.scale_factor < 0.50:
                        result.allow_escalation = False
                    # else: allow_escalation stays True (default), enabling recovery-path pyramiding
                    
                    logger.info(
                        f"[PARTIAL_CONSENSUS] Applied: scale={pc_result.scale_factor:.2f}, "
                        f"aligned={pc_result.aligned_agents}, exposure->{base_exposure:.2f}"
                    )
                
            except ImportError:
                pass  # Partial consensus module not available
            except Exception as e:
                logger.debug(f"Partial consensus check skipped: {e}")

        # =====================================================================
        # LAYER 4.6: CONSENSUS SNAPSHOT - Cross-Agent Awareness (S15)
        # =====================================================================
        # Aggregate ADVISE agent directions to detect unanimity.
        # When 4/5+ ADVISE agents agree, grant extra influence budget to L4.75.

        _consensus_boost = 0.0
        try:
            _advise_names = ["short_bias", "funding_rate", "onchain", "llm_sentiment", "flow"]
            _c_dirs = []
            _c_confs = []
            _n_bearish = 0
            _n_bullish = 0

            for _ca in _advise_names:
                _ca_sig = signals.get(_ca)
                if _ca_sig and hasattr(_ca_sig, "direction") and hasattr(_ca_sig, "confidence"):
                    if abs(_ca_sig.direction) < 0.01 or _ca_sig.confidence < 0.05:
                        continue
                    _c_dirs.append(_ca_sig.direction)
                    _c_confs.append(_ca_sig.confidence)
                    if _ca_sig.direction < -0.3:
                        _n_bearish += 1
                    elif _ca_sig.direction > 0.3:
                        _n_bullish += 1

            if _c_dirs:
                _max_side = max(_n_bearish, _n_bullish)
                _unanimity = _max_side / len(_c_dirs)
                _avg_conf = sum(_c_confs) / len(_c_confs)

                if _unanimity > 0.8 and _avg_conf > 0.5:
                    _consensus_boost = 0.05
                elif _unanimity > 0.6:
                    _consensus_boost = 0.02

                if _consensus_boost > 0:
                    logger.info(
                        f"[CONSENSUS] n_bear={_n_bearish}, n_bull={_n_bullish}, "
                        f"unanimity={_unanimity:.2f}, avg_conf={_avg_conf:.2f}, "
                        f"boost={_consensus_boost}"
                    )
        except Exception as _cons_err:
            logger.debug(f"[CONSENSUS] Skipped: {_cons_err}")

        # =====================================================================
        # LAYER 4.75: ADVISE INFLUENCE (Regime-Conditional)
        # =====================================================================
        # ADVISE agents collectively nudge exposure up/down based on alignment
        # with the decider direction.  Influence is capped at ±MAX_ADVISE_INFLUENCE.

        try:
            regime_name = getattr(context, "regime", "UNKNOWN") or "UNKNOWN"  # [FIX-32]
            advise_weights = ADVISE_WEIGHTS_BY_REGIME.get(
                regime_name, ADVISE_WEIGHTS_BY_REGIME["default"]
            )

            weighted_alignment = 0.0
            total_weight = 0.0
            advise_details = []

            for agent, authority in matrix.items():
                if authority != Authority.ADVISE or agent not in signals:
                    continue
                w = advise_weights.get(agent, 0.0)
                if w <= 0:
                    continue
                sig = signals[agent]
                if sig.confidence < 0.05 or abs(sig.direction) < 0.01:
                    continue

                # alignment: +1 if same direction as decider, -1 if opposite
                if decider_signal.direction != 0.0:
                    alignment = 1.0 if (sig.direction * decider_signal.direction > 0) else -1.0
                else:
                    alignment = 0.0

                contribution = w * sig.confidence * alignment
                weighted_alignment += contribution
                total_weight += w
                advise_details.append(f"{agent}={alignment:+.0f}*{sig.confidence:.2f}*{w:.2f}")

            if total_weight > 0:
                # Normalize by total weight and scale to MAX_ADVISE_INFLUENCE
                # [S15] consensus_boost widens the influence band when agents agree
                _effective_max = min(MAX_ADVISE_INFLUENCE + _consensus_boost, 0.35)
                normalized = weighted_alignment / total_weight
                influence = normalized * _effective_max
                influence = max(-_effective_max, min(_effective_max, influence))

                old_exposure = base_exposure
                base_exposure *= (1.0 + influence)
                base_exposure = max(0.0, min(1.0, base_exposure))

                if abs(influence) > 0.01:
                    logger.info(
                        f"[ADVISE_INFLUENCE] regime={regime_name}, "
                        f"influence={influence:+.3f}, "
                        f"exposure {old_exposure:.3f}->{base_exposure:.3f}, "
                        f"agents=[{', '.join(advise_details)}]"
                    )
        except Exception as e:
            logger.debug(f"ADVISE influence skipped: {e}")

        # =====================================================================
        # LAYER 5: TRIGGER AMPLIFICATION (OPPORTUNITY only)
        # =====================================================================
        
        if context.mode == SystemMode.OPPORTUNITY:
            for agent, authority in matrix.items():
                if authority == Authority.TRIGGER and agent in signals:
                    trigger_signal = signals[agent]
                    
                    # TRIGGER can only amplify, never reduce
                    if trigger_signal.confidence > 0.7:
                        base_exposure = min(base_exposure * 1.2, 1.0)
                        logger.info(f"{agent} TRIGGER amplifying exposure")
        
        # =====================================================================
        # LAYER 6: CAP APPLICATION
        # =====================================================================
        
        caps_applied = {}
        
        for agent, authority in matrix.items():
            if authority == Authority.CAP and agent in signals:
                cap_signal = signals[agent]
                leverage_cap = cap_signal.leverage_cap

                if leverage_cap < 1.0:
                    # [SHORT-OPT SB-2] Direction-aware macro cap
                    # Short positions benefit from risk-off/crisis macro environments
                    if result.direction < 0:  # Short - macro headwind benefits shorts
                        adjusted_cap = min(leverage_cap * 1.5, 0.75)
                    else:  # Long or flat - full macro restriction
                        adjusted_cap = leverage_cap * 0.6
                    base_exposure = min(base_exposure, adjusted_cap)
                    caps_applied[agent] = adjusted_cap
        
        result.caps_applied = caps_applied
        result.target_exposure = base_exposure

        # [FIX-4] Compute fusion confidence: decider conf × confirm penalty
        _fusion_conf = decider_signal.confidence
        if confirm_misaligned:
            _fusion_conf *= 0.5
        result.confidence = max(0.0, min(1.0, _fusion_conf))

        # =====================================================================
        # LAYER 7: VETO DIRECTION APPLICATION
        # =====================================================================
        
        # Apply direction vetoes from Sentiment
        for veto in vetoes:
            parts = veto.split(":")
            if len(parts) == 2:
                agent, veto_dir = parts
                if agent == "sentiment":
                    # Sentiment can only veto the OPPOSITE direction
                    if veto_dir == "long" and result.direction > 0:
                        result.direction = 0.0
                        result.target_exposure = 0.0
                        logger.info("Sentiment veto blocked LONG direction")
                    elif veto_dir == "short" and result.direction < 0:
                        result.direction = 0.0
                        result.target_exposure = 0.0
                        logger.info("Sentiment veto blocked SHORT direction")
        
        # =====================================================================
        # LAYER 8: EXECUTION MODE (from Lead-Lag)
        # =====================================================================
        
        lead_lag_signal = signals.get("lead_lag", AgentSignal())
        
        if context.mode == SystemMode.OPPORTUNITY and context.lead_lag_confident:
            result.execution_mode = "AGGRESSIVE_TAKER"
            result.urgency = 0.8
            result.delay_allowed = False
        else:
            result.execution_mode = "PASSIVE_PREFERRED"
            result.urgency = 0.5
            result.delay_allowed = True
        
        # =====================================================================
        # LAYER 9: TRANCHE GUIDANCE (from DRL if enabled)
        # =====================================================================
        
        drl_signal = signals.get("drl", AgentSignal())
        
        if context.drl_enabled and drl_signal.tranche_advice:
            if drl_signal.tranche_advice == "ESCALATE":
                result.allow_escalation = True
            elif drl_signal.tranche_advice == "HOLD":
                result.allow_escalation = False
        else:
            # Default: allow escalation in early phases
            result.allow_escalation = context.regime_phase in [
                RegimePhase.IGNITION, RegimePhase.EXPANSION
            ]
        
        # Phase-based tranche gating
        if context.regime_phase == RegimePhase.EXHAUSTION:
            result.allow_escalation = False
            logger.info("Phase EXHAUSTION: blocking tranche escalation")
        elif context.regime_phase == RegimePhase.SATURATION:
            if _escalation_profile == "aggressive":
                result.allow_escalation = True
                result.max_tranche_tier = min(result.max_tranche_tier, 3)
                logger.info("Phase SATURATION + aggressive profile: escalation allowed (T3 cap)")
            else:
                result.allow_escalation = False
                logger.info("Phase SATURATION: blocking tranche escalation")
        
        # =====================================================================
        # LAYER 10: MOMENTUM MEMORY INTEGRATION
        # =====================================================================
        
        # Update momentum memory
        self._momentum_memory.update(
            lead_lag_edge=context.lead_lag_edge,
            regime_confidence=context.phase_result.phase_confidence if context.phase_result else 0.5,
        )
        
        # In OPPORTUNITY, momentum memory can increase action probability
        if context.mode == SystemMode.OPPORTUNITY:
            memory_boost = self._momentum_memory.get_action_boost()
            if memory_boost > 0:
                result.target_exposure = min(result.target_exposure * (1 + memory_boost), 1.0)
                result.urgency = min(result.urgency + memory_boost * 0.3, 1.0)
        
        return result
    
    @staticmethod
    def _get_htf_direction(context: FusionContext) -> int:
        """Return daily trend direction from context. 0 = no data / neutral."""
        return getattr(context, "htf_trend_direction", 0)

    def _get_authority_matrix(self, mode: SystemMode) -> Dict[str, Authority]:
        """Get authority matrix for current mode with DRL authority applied.

        [FIX-DRL-AUTHORITY] Previously returned raw matrices without DRL ACTIVE
        upgrade. The module-level get_authority_matrix() applied the upgrade but
        this instance method did not — so DRL stayed at ADVISE in all actual
        fusion decisions despite logs showing ACTIVE. Now delegates to the
        module-level function which correctly upgrades DRL to DECIDE when ACTIVE.
        """
        return get_authority_matrix(mode.name if hasattr(mode, 'name') else str(mode))
    
    def get_authority_for_agent(
        self,
        agent: str,
        mode: SystemMode,
    ) -> Authority:
        """Get authority for a specific agent in a mode."""
        matrix = self._get_authority_matrix(mode)
        return matrix.get(agent, Authority.NONE)


# ============================================================================
# MOMENTUM MEMORY (v3.4 addition)
# ============================================================================

class MomentumMemory:
    """
    Short-term memory for fusion decisions.
    
    PURPOSE:
        - Accumulate weak lead-lag confirmations over time
        - Track regime confidence trend
        - Increase action probability in OPPORTUNITY (not smooth it away)
    
    WHY THIS IMPROVES PROFITABILITY:
        Single weak signals get missed. Accumulated weak signals
        that point the same direction should increase confidence.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
        # Memory buffers
        self._lead_lag_edges: List[float] = []
        self._regime_confidences: List[float] = []
        
        # Decay factor per update
        self.decay = 0.85
    
    def update(
        self,
        lead_lag_edge: float = 0.0,
        regime_confidence: float = 0.5,
    ):
        """Update memory with new observations."""
        
        # Decay existing values
        self._lead_lag_edges = [v * self.decay for v in self._lead_lag_edges]
        self._regime_confidences = [v * self.decay for v in self._regime_confidences]
        
        # Add new values
        self._lead_lag_edges.append(lead_lag_edge)
        self._regime_confidences.append(regime_confidence)
        
        # Trim to window
        self._lead_lag_edges = self._lead_lag_edges[-self.window_size:]
        self._regime_confidences = self._regime_confidences[-self.window_size:]
    
    def get_accumulated_edge(self) -> float:
        """Get accumulated lead-lag edge."""
        if not self._lead_lag_edges:
            return 0.0
        return sum(self._lead_lag_edges)
    
    def get_confidence_trend(self) -> float:
        """Get regime confidence trend (-1 to 1)."""
        if len(self._regime_confidences) < 3:
            return 0.0
        
        recent = self._regime_confidences[-3:]
        older = self._regime_confidences[:-3] if len(self._regime_confidences) > 3 else [0.5]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        return recent_avg - older_avg
    
    def get_action_boost(self) -> float:
        """
        Get boost to action probability.
        
        Returns 0.0 to 0.3 boost based on accumulated evidence.
        """
        edge = self.get_accumulated_edge()
        trend = self.get_confidence_trend()
        
        # Both must be positive for boost
        if edge <= 0 or trend <= 0:
            return 0.0
        
        # Scale boost
        boost = min(edge * 0.1, 0.15) + min(trend * 0.5, 0.15)
        return min(boost, 0.3)
    
    def reset(self):
        """Reset memory."""
        self._lead_lag_edges = []
        self._regime_confidences = []


# ============================================================================
# TIME-SCALE OVERRIDE RULES
# ============================================================================

"""
TIME-SCALE OVERRIDE RULES (v3.4):

4H LAYER (Master):
    - Owns: Direction, target exposure, tranche level
    - Authority: Quant (NORMAL) or Regime (OPPORTUNITY)
    - CANNOT be modified by 200ms layer

200ms LAYER (Execution):
    - Owns: Order timing, slice size, maker/taker mode
    - Authority: Lead-Lag + Execution Agent
    - CANNOT modify: Direction, target exposure, stops

OVERRIDE RULES:
    1. 200ms can DELAY 4H intent (up to 2 bars max)
    2. 200ms can ACCELERATE 4H intent (within same bar)
    3. 200ms CANNOT change target exposure
    4. 200ms CANNOT change direction
    5. 200ms CANNOT skip a tranche level

These rules are enforced in execution_timing_v34.py
"""


# Singleton
_fusion_engine: Optional[AuthorityFusionEngine] = None

def get_fusion_engine() -> AuthorityFusionEngine:
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = AuthorityFusionEngine()
    return _fusion_engine

def reset_fusion_engine():
    global _fusion_engine
    _fusion_engine = None


# ============================================================================
# Compatibility aliases for master_pipeline.py integration
# ============================================================================
QuantSignal = AgentSignal  # master_pipeline uses QuantSignal name
DRLSignal = AgentSignal    # DRL signals use same structure  
SentimentSignal = AgentSignal  # Sentiment signals use same structure
MacroContext = FusionContext   # Macro context maps to fusion context
AuthorityFusionModule = AuthorityFusionEngine  # Old name compatibility
get_authority_fusion = get_fusion_engine  # Old function name

__all__ = [
    'Authority', 'SystemMode', 'AgentSignal', 'FusionContext', 'FusionResult',
    'AuthorityFusionEngine', 'MomentumMemory', 'get_fusion_engine', 'reset_fusion_engine',
    # Compatibility aliases
    'QuantSignal', 'DRLSignal', 'SentimentSignal', 'MacroContext',
    'AuthorityFusionModule', 'get_authority_fusion'
]

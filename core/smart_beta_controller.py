"""
Smart Beta Context Layer V1 — runtime-only, non-invasive, bounded.

Provides market environment context to guide alpha behavior:
  - TrendMomentumContext: directional background + trend quality
  - VolatilityRegimeContext: compression/expansion/toxicity
  - LiquidityFundingContext: crowding/funding heat/liquidation risk

Does NOT:
  - Decide trade direction
  - Veto trades
  - Change DRL obs_dim or training contracts
  - Add new DECIDE/VETO authority

Does:
  - Modulate alpha_gate_mult (bounded)
  - Modulate position_size_mult (bounded)
  - Modulate confidence_mult (bounded)
  - Control allow_scale_in
  - Restrict short exposure multiplier
  - Log proof-level diagnostics

Modes:
  enabled=false        → strict no-op
  mode=observe_only    → computes + logs, zero decision impact
  mode=bounded_influence → limited runtime modulation through existing paths

Reuses existing code:
  - market_data_pipeline: Hurst, vol_regime, vol_z_score, convexity, BSS, MTF momentum
  - deterministic_sentiment: crowding_score, funding signals
  - GMM regime: regime_state, regime_confidence
  - phase_detector: RegimePhase (IGNITION/EXPANSION/SATURATION/EXHAUSTION)
  - alpha_boost pattern: multiplicative injection into agent_signals
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SmartBetaConfig:
    """Configuration for Smart Beta Controller."""
    enabled: bool = False
    mode: str = "observe_only"  # observe_only | bounded_influence

    # Context enables
    trend_enabled: bool = True
    volatility_enabled: bool = True
    liquidity_enabled: bool = True

    # Bounded output clamps
    alpha_gate_mult_min: float = 0.85
    alpha_gate_mult_max: float = 1.20
    size_mult_min: float = 0.70
    size_mult_max: float = 1.15
    confidence_mult_min: float = 0.80
    confidence_mult_max: float = 1.10
    short_restriction_mult_min: float = 0.20
    short_restriction_mult_max: float = 1.00

    # Feature controls
    allow_scale_in_modulation: bool = True
    allow_direction_override: bool = False  # MUST stay False
    allow_veto: bool = False  # MUST stay False

    log_proofs: bool = True


@dataclass
class SmartBetaState:
    """Output contract for Smart Beta Controller."""
    # Trend context
    trend_dir: float = 0.0           # [-1, 1]
    trend_strength: float = 0.0      # [0, 1]
    trend_quality: float = 0.0       # [0, 1]

    # Volatility context
    vol_regime_score: float = 0.5    # [0, 1] 0=compressed, 1=expanded
    vol_expansion_score: float = 0.0 # [0, 1]
    toxicity_score: float = 0.0      # [0, 1]

    # Liquidity/Funding context
    crowding_score: float = 0.0      # [0, 1]
    funding_heat: float = 0.0        # [-1, 1]
    liquidity_risk: float = 0.0      # [0, 1]
    liquidation_risk: float = 0.0    # [0, 1]

    # Neutral drift score
    neutral_drift_score: float = 0.0 # [0, 1] how "drifty/directionless" the market is

    # Recommended modulations (bounded)
    recommended_alpha_gate_mult: float = 1.0
    recommended_size_mult: float = 1.0
    recommended_confidence_mult: float = 1.0
    allow_scale_in: bool = True
    short_restriction_mult: float = 1.0

    # Diagnostics
    explanation_tags: list = field(default_factory=list)
    mode: str = "observe_only"


class SmartBetaController:
    """
    Lightweight orchestrator composing existing market data into bounded modulations.

    Reuses: market_data dict (Hurst, vol_regime, convexity, BSS, funding, OI, liq,
    vpin, dvol_zscore, MTF momentum, GMM regime, phase detector, beta_to_btc)
    """

    def __init__(self, config: Optional[SmartBetaConfig] = None):
        self.config = config or SmartBetaConfig()

    def compute(
        self,
        asset: str,
        market_data: Dict[str, Any],
        agent_signals: Dict[str, Any],
    ) -> SmartBetaState:
        """Compute SmartBetaState from existing market_data and agent_signals.

        Returns:
            SmartBetaState with bounded recommendations.
            If disabled, returns neutral state (all mults = 1.0).
        """
        state = SmartBetaState(mode=self.config.mode)

        if not self.config.enabled:
            return state

        cfg = self.config
        tags = []

        # =====================================================================
        # TREND MOMENTUM CONTEXT (reuses: GMM regime, phase, Hurst, MTF momentum)
        # =====================================================================
        if cfg.trend_enabled:
            regime = str(market_data.get("regime_state", "")).upper()
            hurst = float(market_data.get("hurst_exponent", 0.5))
            mtf_mom = float(market_data.get("mtf_momentum", 0.0))
            phase = str(market_data.get("phase", agent_signals.get("_phase", "UNKNOWN"))).upper()
            adx = float(market_data.get("adx", 20.0))

            # Trend direction from MTF momentum + regime
            _BULLISH_REGIMES = {"MOMENTUM_RALLY", "STEADY_UPTREND"}
            _BEARISH_REGIMES = {"PANIC_SELLOFF", "VOLATILE_CHOP"}
            _NEUTRAL_REGIMES = {"QUIET_ACCUMULATION", "WEAK_CONSOLIDATION", "NEUTRAL_DRIFT"}

            if regime in _BULLISH_REGIMES:
                state.trend_dir = min(0.5 + abs(mtf_mom) * 5, 1.0)
            elif regime in _BEARISH_REGIMES:
                state.trend_dir = max(-0.5 - abs(mtf_mom) * 5, -1.0)
            else:
                state.trend_dir = float(np.clip(mtf_mom * 5, -0.3, 0.3)) if 'np' in dir() else max(-0.3, min(0.3, mtf_mom * 5))

            # Trend strength: ADX-based
            state.trend_strength = min(max(adx - 15, 0) / 35.0, 1.0)

            # Trend quality: Hurst > 0.55 = trending quality
            state.trend_quality = min(max(hurst - 0.4, 0) / 0.3, 1.0)

            # Neutral drift score: high when regime is directionless + Hurst near 0.5 + low ADX
            if regime in _NEUTRAL_REGIMES:
                _nd_regime = 0.6
                _nd_hurst = max(0, 1.0 - abs(hurst - 0.5) * 4)  # peaks at H=0.5
                _nd_adx = max(0, 1.0 - (adx - 10) / 25)  # peaks at ADX=10
                state.neutral_drift_score = min(_nd_regime * 0.4 + _nd_hurst * 0.3 + _nd_adx * 0.3, 1.0)
            else:
                state.neutral_drift_score = 0.0

            # Tags
            if regime in _NEUTRAL_REGIMES:
                tags.append("NEUTRAL_DRIFT")
            elif regime in _BULLISH_REGIMES and phase in ("IGNITION", "EXPANSION"):
                tags.append("TREND_STRONG")
            elif regime in _BEARISH_REGIMES:
                tags.append("BEARISH_REGIME")

            if cfg.log_proofs:
                logger.debug(
                    f"[BETA_TREND] {asset}: dir={state.trend_dir:+.2f} str={state.trend_strength:.2f} "
                    f"qual={state.trend_quality:.2f} drift={state.neutral_drift_score:.2f} "
                    f"regime={regime} hurst={hurst:.2f} adx={adx:.0f}"
                )

        # =====================================================================
        # VOLATILITY REGIME CONTEXT (reuses: vol_regime, vol_z_score, VPIN, kurtosis)
        # =====================================================================
        if cfg.volatility_enabled:
            vol_regime = float(market_data.get("vol_regime", 1.0))
            vol_z = float(market_data.get("vol_z_score", 0.0))
            vpin = float(market_data.get("vpin", 0.35))
            kurtosis = float(market_data.get("kurtosis", 3.0))
            bss = int(market_data.get("black_swan_sentinel", 3))

            # Vol regime score: 0 = compressed, 0.5 = normal, 1 = expanded
            state.vol_regime_score = min(max(vol_regime - 0.5, 0) / 1.5, 1.0)

            # Vol expansion
            state.vol_expansion_score = min(max(vol_z, 0) / 3.0, 1.0)

            # Toxicity: VPIN + kurtosis
            vpin_tox = max(0, (vpin - 0.5) / 0.5)
            kurt_tox = max(0, (kurtosis - 5) / 10)
            state.toxicity_score = min(vpin_tox * 0.6 + kurt_tox * 0.4, 1.0)

            if vol_regime < 0.7:
                tags.append("VOL_COMPRESSED")
            elif vol_regime > 1.5:
                tags.append("VOL_EXPANDED")
            if state.toxicity_score > 0.5:
                tags.append("HIGH_TOXICITY")
            if bss <= 1:
                tags.append("BLACK_SWAN_RISK")

            if cfg.log_proofs:
                logger.debug(
                    f"[BETA_VOL] {asset}: regime={vol_regime:.2f} z={vol_z:.2f} "
                    f"tox={state.toxicity_score:.2f} bss={bss} "
                    f"expansion={state.vol_expansion_score:.2f}"
                )

        # =====================================================================
        # LIQUIDITY FUNDING CONTEXT (reuses: funding, OI, liq, crowding, F&G)
        # =====================================================================
        if cfg.liquidity_enabled:
            funding = float(market_data.get("funding_rate", 0.0))
            oi_chg = float(market_data.get("oi_change_24h_pct", 0.0))
            liq_imb = float(market_data.get("liquidation_imbalance", 0.0))
            fg = market_data.get("_fear_greed_value", 50)
            fg = int(fg) if isinstance(fg, (int, float)) else 50
            crowding = float(market_data.get("_ssc_crowding", 0.0))

            # Funding heat: abs(funding) > threshold
            state.funding_heat = float(max(-1, min(1, funding * 500)))

            # Crowding
            state.crowding_score = min(abs(crowding), 1.0)

            # Liquidity risk from OI spike + F&G extreme
            oi_risk = min(max(oi_chg - 3, 0) / 10, 1.0)
            fg_risk = max(0, (80 - fg) / 80) if fg > 70 else max(0, (fg - 20) / -20) if fg < 30 else 0
            # Correct: extreme fear OR greed = risk
            fg_risk = 1.0 - min(abs(fg - 50) / 50, 1.0) if abs(fg - 50) > 25 else 0.0
            state.liquidity_risk = min(oi_risk * 0.5 + abs(state.funding_heat) * 0.3 + fg_risk * 0.2, 1.0)

            # Liquidation risk
            state.liquidation_risk = min(abs(liq_imb), 1.0)

            if abs(state.funding_heat) > 0.5:
                tags.append("FUNDING_EXTREME")
            # Carry opportunity: negative funding = LONGs earn carry income
            # This is free alpha when holding LONG positions in negative funding environment
            if state.funding_heat < -0.3:
                tags.append("CARRY_OPPORTUNITY_LONG")
            elif state.funding_heat > 0.3:
                tags.append("CARRY_OPPORTUNITY_SHORT")
            if state.crowding_score > 0.6:
                tags.append("HIGH_CROWDING")
            if state.liquidation_risk > 0.5:
                tags.append("LIQUIDATION_RISK")

            if cfg.log_proofs:
                logger.debug(
                    f"[BETA_LIQ] {asset}: funding_heat={state.funding_heat:+.2f} "
                    f"crowd={state.crowding_score:.2f} liq_risk={state.liquidity_risk:.2f} "
                    f"liq_imb={state.liquidation_risk:.2f} fg={fg}"
                )

        # =====================================================================
        # COMPUTE BOUNDED RECOMMENDATIONS
        # =====================================================================
        gate_mult = 1.0
        size_mult = 1.0
        conf_mult = 1.0
        allow_si = True
        short_mult = 1.0

        # Trend: neutral drift → tighter gate, smaller size, block scale-in
        if "NEUTRAL_DRIFT" in tags:
            gate_mult *= 1.10   # higher bar to enter
            size_mult *= 0.85   # smaller positions
            allow_si = False
        elif "TREND_STRONG" in tags:
            gate_mult *= 0.90   # lower bar (let alpha participate)
            size_mult *= 1.10   # slightly larger

        # Volatility: compression → tighter, expansion+toxic → reduce
        if "VOL_COMPRESSED" in tags:
            gate_mult *= 1.05
            size_mult *= 0.90
            allow_si = False
        if "HIGH_TOXICITY" in tags:
            size_mult *= 0.80
            conf_mult *= 0.85
        if "BLACK_SWAN_RISK" in tags:
            size_mult *= 0.70
            conf_mult *= 0.80
            allow_si = False
            short_mult *= 0.50

        # Liquidity: overheating → restrict; carry opportunity → encourage
        if "FUNDING_EXTREME" in tags:
            size_mult *= 0.85
            short_mult *= 0.70
        # Carry opportunity: negative funding = LONGs earn carry income
        # Relax gate for LONG entries and boost size — this is "free" alpha
        if "CARRY_OPPORTUNITY_LONG" in tags:
            gate_mult *= 0.92   # lower bar for LONG entries
            size_mult *= 1.08   # slightly larger LONG positions (earn carry)
            short_mult *= 0.80  # discourage shorts (they PAY funding)
        elif "CARRY_OPPORTUNITY_SHORT" in tags:
            short_mult *= 1.00  # don't penalize shorts when they earn carry
        if "HIGH_CROWDING" in tags:
            size_mult *= 0.90
            allow_si = False
        if "LIQUIDATION_RISK" in tags:
            size_mult *= 0.85
            conf_mult *= 0.90

        # Clamp to bounds
        state.recommended_alpha_gate_mult = max(cfg.alpha_gate_mult_min,
                                                 min(cfg.alpha_gate_mult_max, gate_mult))
        state.recommended_size_mult = max(cfg.size_mult_min,
                                           min(cfg.size_mult_max, size_mult))
        state.recommended_confidence_mult = max(cfg.confidence_mult_min,
                                                 min(cfg.confidence_mult_max, conf_mult))
        state.short_restriction_mult = max(cfg.short_restriction_mult_min,
                                            min(cfg.short_restriction_mult_max, short_mult))
        if cfg.allow_scale_in_modulation:
            state.allow_scale_in = allow_si

        state.explanation_tags = tags
        return state

    def apply_to_agent_signals(
        self,
        state: SmartBetaState,
        agent_signals: Dict[str, Any],
        asset: str,
    ) -> bool:
        """Apply SmartBetaState to agent_signals through existing modulation paths.

        Returns True if any modulation was applied.
        """
        if not self.config.enabled:
            if self.config.log_proofs:
                logger.debug(f"[SMART_BETA_NOOP] {asset}: disabled")
            return False

        # Store state for downstream logging
        agent_signals["_smart_beta_state"] = {
            "trend_dir": state.trend_dir,
            "trend_strength": state.trend_strength,
            "vol_regime_score": state.vol_regime_score,
            "toxicity_score": state.toxicity_score,
            "crowding_score": state.crowding_score,
            "funding_heat": state.funding_heat,
            "tags": state.explanation_tags,
            "gate_mult": state.recommended_alpha_gate_mult,
            "size_mult": state.recommended_size_mult,
            "conf_mult": state.recommended_confidence_mult,
            "allow_scale_in": state.allow_scale_in,
            "short_mult": state.short_restriction_mult,
            "mode": state.mode,
        }

        if state.mode == "observe_only":
            if self.config.log_proofs:
                logger.info(
                    f"[SMART_BETA_STATE] {asset}: "
                    f"tags={state.explanation_tags} "
                    f"gate={state.recommended_alpha_gate_mult:.2f} "
                    f"size={state.recommended_size_mult:.2f} "
                    f"conf={state.recommended_confidence_mult:.2f} "
                    f"scale_in={state.allow_scale_in} "
                    f"short={state.short_restriction_mult:.2f} "
                    f"[OBSERVE_ONLY — no decision change]"
                )
            return False

        if state.mode != "bounded_influence":
            return False

        applied = False

        # Multiply into EXISTING modulation paths (same pattern as alpha_boost)
        _cur_gate = float(agent_signals.get("_regime_alpha_gate_mult", 1.0))
        _new_gate = _cur_gate * state.recommended_alpha_gate_mult
        if abs(_new_gate - _cur_gate) > 0.001:
            agent_signals["_regime_alpha_gate_mult"] = _new_gate
            agent_signals["_regime_alpha_gate_mult_short"] = float(
                agent_signals.get("_regime_alpha_gate_mult_short", _cur_gate)
            ) * state.recommended_alpha_gate_mult
            applied = True

        _cur_size = float(agent_signals.get("_regime_position_size_mult", 1.0))
        _new_size = _cur_size * state.recommended_size_mult
        if abs(_new_size - _cur_size) > 0.001:
            agent_signals["_regime_position_size_mult"] = _new_size
            applied = True

        # Scale-in modulation
        if not state.allow_scale_in and self.config.allow_scale_in_modulation:
            agent_signals["_smart_beta_block_scale_in"] = True
            applied = True

        if self.config.log_proofs:
            if applied:
                logger.info(
                    f"[SMART_BETA_APPLY] {asset}: "
                    f"gate {_cur_gate:.2f}→{agent_signals.get('_regime_alpha_gate_mult', 1.0):.2f} "
                    f"size {_cur_size:.2f}→{agent_signals.get('_regime_position_size_mult', 1.0):.2f} "
                    f"tags={state.explanation_tags}"
                )
            else:
                logger.info(
                    f"[SMART_BETA_NOOP] {asset}: no modulation needed "
                    f"(all mults ~1.0, tags={state.explanation_tags})"
                )

        return applied


# Singleton
_controller: Optional[SmartBetaController] = None


def get_smart_beta_controller(config: Optional[Dict] = None) -> SmartBetaController:
    """Get or create SmartBetaController singleton."""
    global _controller
    if _controller is None:
        cfg = SmartBetaConfig()
        if config:
            for k, v in config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        _controller = SmartBetaController(cfg)
    return _controller

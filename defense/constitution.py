"""
================================================================================
HMATS v3.6 - CONSTITUTION GUARANTEES MODULE
================================================================================
Version: 3.6.1
Purpose: Restore v3.3-STABILIZED internal guarantees to v3.6

RESTORED GUARANTEES (from v3.3-STABILIZED):
    1. NO_TRADE trigger computation (internal, not caller-provided)
    2. OPPORTUNITY trigger computation (internal, with TTL/decay)
    3. Alpha threshold gating (friction-aware)
    4. Tranche scheduling + abort conditions
    5. Signal conflict detection (all-conflict-flat)
    6. Strategy constitution enforcement (rule disables)
    7. Market data validation (canonical schema)

CRITICAL: These checks run INSIDE v3.6 runtime, not delegated to caller.
================================================================================
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from enum import Enum, auto

from core.market_data_helpers import effective_volume_ratio
from configs.canonical_config import (  # [FIX-33] single source for fee constants
    KRAKEN_FREE_TIER_USD as _CC_FREE_TIER,
    KRAKEN_TAKER_BPS as _CC_TAKER_BPS,
    KRAKEN_MAKER_BPS as _CC_MAKER_BPS,
)

logger = logging.getLogger(__name__)

# [P225] Terminal bounds for the COMPOSED regime/smart-beta alpha-gate
# multiplier (see check_alpha_gate). Wider than any single writer's own bounds
# on purpose — this is a runaway-stacking backstop, not a tuning knob.
REGIME_GATE_MULT_MIN = 0.5
REGIME_GATE_MULT_MAX = 2.0


# =============================================================================
# MARKET DATA VALIDATION (v3.3-STABILIZED guarantee #7)
# =============================================================================

# CANONICAL SCHEMA KEY MAPPING (Patch 1)
CANONICAL_SCHEMA = {
    # === REQUIRED KEYS (missing -> NO_TRADE) ===
    "required": {
        "current_price": {"type": float, "min": 0.0001, "max": 1_000_000_000},
        "volume_24h": {"type": float, "min": 0},
        "data_age_seconds": {"type": float, "min": 0, "max": 60.0},  # [P22 2026-04-24] aligned with MAX_DATA_AGE_SECONDS class constant; 4H tick cycle
    },
    # === CRITICAL KEYS (missing -> NO_TRADE in NORMAL, warning in OPPORTUNITY) ===
    "critical": {
        "dvol_zscore": {"type": float, "default": 0.0},
        "vpin": {"type": float, "default": 0.5, "min": 0, "max": 1},
        "correlation_btc_eth_sol": {"type": float, "default": 0.0, "min": -1, "max": 1},  # 0.0 = unknown (fail-safe)
        "orderbook_depth_1pct_usd": {"type": float, "default": 500_000.0, "min": 0},  # [T10] was inf
    },
    # === OPTIONAL KEYS (missing -> use default, log warning) ===
    "optional": {
        "binance_price": {"type": float, "default": 0.0},
        "kraken_price": {"type": float, "default": 0.0},
        "spread_bps": {"type": float, "default": 10.0},
        "structure_break_pct": {"type": float, "default": 0.0},
        "volume_ratio": {"type": float, "default": 1.0},
        "volume_ratio_effective": {"type": float, "default": 1.0},
        "orderbook_imbalance": {"type": float, "default": 0.0},
        "structure_level": {"type": float, "default": 0.0},
        "dvol_1h_change": {"type": float, "default": 0.0},
    },
    # === ALIASES (map legacy keys to canonical keys) ===
    "aliases": {
        "price": "current_price",
        "vol_24h": "volume_24h",
        "age_seconds": "data_age_seconds",
        "dvol": "dvol_zscore",
        "depth_1pct": "orderbook_depth_1pct_usd",
    },
}


@dataclass
class MarketDataValidationResult:
    """Result of market data validation."""
    is_valid: bool = True
    error: str = ""
    canonical_data: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_critical: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    proof_log: str = ""


class MarketDataValidator:
    """
    Canonical market data validation with schema key mapping.
    
    From v3.3-STABILIZED core/market_data_schema.py
    Enhanced with Patch 1: key mapping, validation, proof logging.
    """
    
    REQUIRED_FIELDS = list(CANONICAL_SCHEMA["required"].keys())
    CRITICAL_FIELDS = list(CANONICAL_SCHEMA["critical"].keys())
    OPTIONAL_FIELDS = list(CANONICAL_SCHEMA["optional"].keys())
    ALIASES = CANONICAL_SCHEMA["aliases"]
    
    MAX_DATA_AGE_SECONDS = 60.0  # [T4] was 10.0; unified with NoTradeTrigger for 4H tick cycle
    MIN_PRICE = 0.0001
    MAX_PRICE = 1_000_000_000
    
    def validate(self, market_data: Dict, mode: str = "NORMAL") -> MarketDataValidationResult:
        """
        Validate market data against canonical schema.
        
        Patch 1: Enhanced with key mapping, missing key detection, proof logging.
        """
        
        warnings = []
        missing_required = []
        missing_critical = []
        missing_optional = []
        
        # Apply aliases first
        canonical_data = self._apply_aliases(market_data)
        
        # Check required fields (missing -> always NO_TRADE)
        for field, spec in CANONICAL_SCHEMA["required"].items():
            if field not in canonical_data:
                missing_required.append(field)
        
        if missing_required:
            proof_log = self._generate_proof_log(
                valid=False, 
                reason=f"MISSING_REQUIRED: {missing_required}",
                canonical_data=canonical_data
            )
            logger.warning(f"[SCHEMA] {proof_log}")
            return MarketDataValidationResult(
                is_valid=False,
                error=f"Missing required fields: {missing_required}",
                missing_required=missing_required,
                proof_log=proof_log
            )
        
        # Check critical fields (missing -> NO_TRADE in NORMAL, warning in OPPORTUNITY)
        # [BUGFIX AUDIT-C3] Detect missing fields FIRST, apply defaults ONLY after validity check
        for field, spec in CANONICAL_SCHEMA["critical"].items():
            if field not in canonical_data:
                missing_critical.append(field)

        if missing_critical and mode == "NORMAL":
            # Return WITHOUT applying defaults - canonical_data stays clean
            proof_log = self._generate_proof_log(
                valid=False,
                reason=f"MISSING_CRITICAL_IN_NORMAL: {missing_critical}",
                canonical_data=canonical_data
            )
            logger.warning(f"[SCHEMA] {proof_log}")
            return MarketDataValidationResult(
                is_valid=False,
                error=f"Missing critical fields in NORMAL mode: {missing_critical}",
                missing_critical=missing_critical,
                proof_log=proof_log
            )
        elif missing_critical:
            # OPPORTUNITY mode: apply defaults only when proceeding
            for field in missing_critical:
                canonical_data[field] = CANONICAL_SCHEMA["critical"][field].get("default", 0.0)
            warnings.append(f"Missing critical fields (defaults applied): {missing_critical}")
        
        # Check optional fields (missing -> use default, log warning)
        for field, spec in CANONICAL_SCHEMA["optional"].items():
            if field not in canonical_data:
                missing_optional.append(field)
                canonical_data[field] = spec.get("default", 0.0)
        
        if missing_optional:
            warnings.append(f"Missing optional fields (defaults applied): {missing_optional}")
        
        # Validate price range
        price = canonical_data.get('current_price', 0)
        if not (self.MIN_PRICE < price < self.MAX_PRICE):
            proof_log = self._generate_proof_log(
                valid=False,
                reason=f"INVALID_PRICE: {price}",
                canonical_data=canonical_data
            )
            return MarketDataValidationResult(
                is_valid=False,
                error=f"Invalid price: {price}",
                proof_log=proof_log
            )
        
        # Validate data age
        data_age = canonical_data.get('data_age_seconds', 0)
        if data_age > self.MAX_DATA_AGE_SECONDS:
            proof_log = self._generate_proof_log(
                valid=False,
                reason=f"STALE_DATA: {data_age:.1f}s",
                canonical_data=canonical_data
            )
            return MarketDataValidationResult(
                is_valid=False,
                error=f"Stale data: {data_age:.1f}s > {self.MAX_DATA_AGE_SECONDS}s",
                proof_log=proof_log
            )
        
        # Check for NaN/inf
        for key, value in canonical_data.items():
            if isinstance(value, float):
                if value != value:  # NaN check
                    proof_log = self._generate_proof_log(
                        valid=False,
                        reason=f"NAN_VALUE: {key}",
                        canonical_data=canonical_data
                    )
                    return MarketDataValidationResult(
                        is_valid=False,
                        error=f"NaN value in field: {key}",
                        proof_log=proof_log
                    )
        
        # Success - generate proof log
        proof_log = self._generate_proof_log(
            valid=True,
            reason="VALID",
            canonical_data=canonical_data
        )
        
        return MarketDataValidationResult(
            is_valid=True,
            canonical_data=canonical_data,
            warnings=warnings,
            missing_critical=missing_critical,
            missing_optional=missing_optional,
            proof_log=proof_log
        )
    
    def _apply_aliases(self, data: Dict) -> Dict:
        """Apply key aliases to map legacy keys to canonical keys."""
        result = dict(data)
        for alias, canonical in self.ALIASES.items():
            if alias in result and canonical not in result:
                result[canonical] = result[alias]
        return result
    
    def _generate_proof_log(self, valid: bool, reason: str, canonical_data: Dict) -> str:
        """Generate structured proof log for schema validation."""
        from datetime import datetime
        
        keys_present = list(canonical_data.keys()) if canonical_data else []
        required_present = [k for k in self.REQUIRED_FIELDS if k in keys_present]
        critical_present = [k for k in self.CRITICAL_FIELDS if k in keys_present]
        
        return (
            f"[SCHEMA_PROOF] {datetime.now(timezone.utc).isoformat()} | "
            f"valid={valid} | reason={reason} | "
            f"required={len(required_present)}/{len(self.REQUIRED_FIELDS)} | "
            f"critical={len(critical_present)}/{len(self.CRITICAL_FIELDS)} | "
            f"keys={len(keys_present)}"
        )


# =============================================================================
# NO_TRADE TRIGGER COMPUTATION (v3.3-STABILIZED guarantee #1)
# =============================================================================

class NoTradeTriggerType(Enum):
    """NO_TRADE trigger types in priority order."""
    DATA_INTEGRITY_FAIL = 1
    EXECUTION_BLOCKED = 2
    STALE_DATA = 3
    FEED_DISAGREEMENT = 4
    FLASH_CRASH = 5
    EXTREME_DVOL = 6
    LIQUIDITY_CRITICAL = 7
    CORRELATION_COLLAPSE = 8
    ALL_CONFLICT_FLAT = 9


@dataclass
class NoTradeCondition:
    """A single NO_TRADE condition."""
    trigger_type: NoTradeTriggerType
    is_active: bool = False
    triggered_at: Optional[datetime] = None
    details: str = ""
    severity: str = "HARD"


@dataclass
class NoTradeState:
    """Aggregated NO_TRADE state."""
    should_no_trade: bool = False
    active_conditions: List[NoTradeCondition] = field(default_factory=list)
    primary_reason: Optional[NoTradeTriggerType] = None
    return_conditions: List[str] = field(default_factory=list)
    trigger_scores: Dict[str, float] = field(default_factory=dict)


class NoTradeTriggerChecker:
    """
    Internal NO_TRADE trigger computation.
    
    From v3.3-STABILIZED core/no_trade_triggers.py
    
    CRITICAL: These thresholds are computed internally, not provided by caller.
    """
    
    # === CATEGORY B: Systemic Risk Thresholds ===
    DVOL_ZSCORE_EXTREME = 5.0
    DVOL_ZSCORE_RECOVERY = 4.0
    
    CORRELATION_COLLAPSE_THRESHOLD = 0.92
    CORRELATION_RECOVERY_THRESHOLD = 0.88
    
    LIQUIDITY_CRITICAL_USD = 100_000
    LIQUIDITY_RECOVERY_USD = 500_000
    
    FLASH_CRASH_PCT = 0.05  # 5% move
    FLASH_CRASH_WINDOW_SECONDS = 300  # 5 minutes
    
    # === CATEGORY C: Data Integrity Thresholds ===
    STALE_DATA_SECONDS = 60.0  # [T4] was 2.0; REST latency 4-5s, 4H tick cycle
    FEED_DISAGREEMENT_PCT = 0.01  # 1%
    # [P287] DELIBERATELY UNUSED (with `_feed_disagreement_start` below):
    # disagreement fires INSTANTLY, which is the fail-safe direction — a
    # duration debounce would let a 1% feed split trade for 30s. Kept only as
    # documentation of the considered-and-rejected debounce; do not wire it
    # in without its own P-entry.
    FEED_DISAGREEMENT_DURATION = 30  # seconds
    
    # === CATEGORY A: Signal Conflict ===
    CONFLICT_DIRECTION_THRESHOLD = 0.3
    
    def __init__(self):
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}  # per-asset
        # [P287] deliberately unused — see FEED_DISAGREEMENT_DURATION note.
        self._feed_disagreement_start: Optional[datetime] = None
    
    def compute_triggers(
        self,
        market_data: Dict,
        signal_data: Dict,
        integrity_data: Optional[Dict] = None,
    ) -> NoTradeState:
        """
        Compute all NO_TRADE triggers INTERNALLY.
        
        Returns NoTradeState with computed trigger scores.
        """
        
        active_conditions = []
        trigger_scores = {}
        
        # === DATA INTEGRITY CHECKS ===
        
        # Stale data
        data_age = market_data.get('data_age_seconds', 0)
        stale_score = min(1.0, data_age / self.STALE_DATA_SECONDS)
        trigger_scores['stale_data'] = stale_score
        if data_age > self.STALE_DATA_SECONDS:
            active_conditions.append(NoTradeCondition(
                trigger_type=NoTradeTriggerType.STALE_DATA,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"Data age {data_age:.1f}s > {self.STALE_DATA_SECONDS}s"
            ))
        
        # Feed disagreement
        binance_price = market_data.get('binance_price', 0)
        kraken_price = market_data.get('kraken_price', 0)
        if binance_price > 0 and kraken_price > 0:
            divergence = abs(binance_price - kraken_price) / kraken_price
            trigger_scores['feed_disagreement'] = min(1.0, divergence / self.FEED_DISAGREEMENT_PCT)
            if divergence > self.FEED_DISAGREEMENT_PCT:
                active_conditions.append(NoTradeCondition(
                    trigger_type=NoTradeTriggerType.FEED_DISAGREEMENT,
                    is_active=True,
                    triggered_at=datetime.now(timezone.utc),
                    details=f"Feed divergence {divergence:.2%}"
                ))
        
        # === SYSTEMIC RISK CHECKS ===
        
        # Flash crash detection
        current_price = market_data.get('current_price', 0)
        asset = market_data.get('asset', 'UNKNOWN')
        volume_ratio = effective_volume_ratio(market_data)
        flash_score = self._check_flash_crash(current_price, asset, volume_ratio)
        trigger_scores['flash_crash'] = flash_score
        if flash_score >= 1.0:
            active_conditions.append(NoTradeCondition(
                trigger_type=NoTradeTriggerType.FLASH_CRASH,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"Flash crash detected (>= {self.FLASH_CRASH_PCT:.0%} in {self.FLASH_CRASH_WINDOW_SECONDS}s)"
            ))
        
        # DVOL extreme
        dvol_zscore = market_data.get('dvol_zscore', 0)
        # [P306] Clamped at 0. EXTREME_DVOL is one-sided by construction
        # (only a vol EXPANSION is the hazard), so a calm market must
        # contribute 0 activation, not a negative one. Behaviour-identical
        # today because both consumers take max(trigger_scores.values()),
        # but P306 is the first change that makes this value genuinely
        # negative (live z is about -1.4 BTC / -2.0 ETH), and a future
        # consumer that sums or averages would read calm as risk RELIEF.
        dvol_score = max(0.0, min(1.0, dvol_zscore / self.DVOL_ZSCORE_EXTREME))
        trigger_scores['extreme_dvol'] = dvol_score
        if dvol_zscore >= self.DVOL_ZSCORE_EXTREME:
            active_conditions.append(NoTradeCondition(
                trigger_type=NoTradeTriggerType.EXTREME_DVOL,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=f"DVOL z-score {dvol_zscore:.1f} >= {self.DVOL_ZSCORE_EXTREME}"
            ))
        
        # Liquidity critical
        depth = market_data.get('orderbook_depth_1pct_usd', float('inf'))
        if depth < float('inf'):
            liquidity_score = max(0, 1.0 - (depth / self.LIQUIDITY_CRITICAL_USD))
            trigger_scores['liquidity_critical'] = liquidity_score
            if depth < self.LIQUIDITY_CRITICAL_USD:
                active_conditions.append(NoTradeCondition(
                    trigger_type=NoTradeTriggerType.LIQUIDITY_CRITICAL,
                    is_active=True,
                    triggered_at=datetime.now(timezone.utc),
                    details=f"Liquidity ${depth:,.0f} < ${self.LIQUIDITY_CRITICAL_USD:,}"
                ))
        
        # Correlation collapse — danger is when correlation EXCEEDS threshold
        # (all assets moving in lockstep = systemic risk / contagion).
        # Score should be 0 when correlation is normal (<0.92), 1.0 when extreme (>=0.92).
        correlation = market_data.get('correlation_btc_eth_sol', 0)
        if correlation > 0:
            if correlation >= self.CORRELATION_COLLAPSE_THRESHOLD:
                # Above threshold: scale 0→1 for the excess
                corr_score = min(1.0, (correlation - self.CORRELATION_COLLAPSE_THRESHOLD)
                                 / (1.0 - self.CORRELATION_COLLAPSE_THRESHOLD + 1e-9))
            else:
                corr_score = 0.0  # Normal correlation — no danger
            trigger_scores['correlation_collapse'] = corr_score
            if correlation >= self.CORRELATION_COLLAPSE_THRESHOLD:  # [FIX-C2] was > (inconsistent with score check at line 422)
                active_conditions.append(NoTradeCondition(
                    trigger_type=NoTradeTriggerType.CORRELATION_COLLAPSE,
                    is_active=True,
                    triggered_at=datetime.now(timezone.utc),
                    details=f"Correlation {correlation:.2f} >= {self.CORRELATION_COLLAPSE_THRESHOLD}"
                ))
        
        # === SIGNAL CONFLICT CHECK ===
        
        conflict_result = self._check_all_conflict_flat(signal_data)
        trigger_scores['signal_conflict'] = conflict_result[1]
        # [FIX] Quant vs DRL conflict alone (score=0.7) should NOT trigger NO_TRADE.
        # Only escalate to NO_TRADE if conflict score >= 0.9 (reserved for 3+ agent conflict).
        # Pure 2-agent conflict reduces confidence via fusion, not via NO_TRADE.
        if conflict_result[0] and conflict_result[1] >= 0.9:
            active_conditions.append(NoTradeCondition(
                trigger_type=NoTradeTriggerType.ALL_CONFLICT_FLAT,
                is_active=True,
                triggered_at=datetime.now(timezone.utc),
                details=conflict_result[2]
            ))
        
        # === BUILD STATE ===
        
        should_no_trade = len(active_conditions) > 0
        primary_reason = active_conditions[0].trigger_type if active_conditions else None
        
        return NoTradeState(
            should_no_trade=should_no_trade,
            active_conditions=active_conditions,
            primary_reason=primary_reason,
            trigger_scores=trigger_scores
        )
    
    def _check_flash_crash(self, current_price: float, asset: str = "UNKNOWN", volume_ratio: float = 1.0) -> float:
        """Check for flash crash using per-asset price history.

        P1: 急跌缩量=洗盘 - if volume is below average during crash,
        reduce score (sharp drop + low volume = shakeout, not real crash).

        [P287] EXPLICITLY DISABLED (the P265 empty-interval pattern): this
        checker is STRUCTURALLY UNREACHABLE at the live cadence and has never
        fired. It keeps a 300-second window (FLASH_CRASH_WINDOW_SECONDS) but
        its only caller is compute_no_trade_triggers, once per 4H tick per
        asset — each sample evicts the previous one (4h > 300s), so
        len(history) < 2 always and the score was 0.0 forever. The 3-tier
        volume/washout logic below is KEPT for a future re-derivation (a
        faster caller would need a re-derived window, P141 — not a silent
        re-arm), but the honest state is a hard 0.0 return, not a check that
        reads as protection. REAL flash protection is the pipeline's
        bar-over-bar `flash_crash_active` (market_data_pipeline) feeding the
        early hard veto in integration_v36 — that path is live and pinned.
        """
        return 0.0  # [P287] see docstring — unreachable-by-cadence, disabled

        if current_price <= 0:
            return 0.0

        now = datetime.now(timezone.utc)

        # Per-asset price history to avoid cross-asset false positives
        if asset not in self._price_history:
            self._price_history[asset] = []
        history = self._price_history[asset]

        # Add current price
        history.append((now, current_price))

        # Clean old entries
        cutoff = now - timedelta(seconds=self.FLASH_CRASH_WINDOW_SECONDS)
        self._price_history[asset] = [(t, p) for t, p in history if t > cutoff]
        history = self._price_history[asset]

        if len(history) < 2:
            return 0.0

        oldest_price = history[0][1]
        if oldest_price <= 0:
            return 0.0

        move_pct = abs(current_price - oldest_price) / oldest_price
        score = min(1.0, move_pct / self.FLASH_CRASH_PCT)

        # P1: 急跌缩量=洗盘 - 3-tier volume classification
        if score >= 1.0:  # >= 5% price move triggered
            if volume_ratio < 0.5:
                # 洗盘 (washout): sharp drop + very low volume = shakeout, NOT a real crash
                logger.info(
                    f"[FLASH_CRASH] {asset}: WASHOUT - price_move={move_pct:.1%}, "
                    f"vol_ratio={volume_ratio:.2f} < 0.5 -> NOT triggered (opportunity)"
                )
                return 0.0  # Don't trigger - possible entry opportunity
            elif volume_ratio >= 1.5:
                # 真崩盘 (real crash): sharp drop + high volume = genuine crash
                logger.info(
                    f"[FLASH_CRASH] {asset}: REAL CRASH - price_move={move_pct:.1%}, "
                    f"vol_ratio={volume_ratio:.2f} >= 1.5 -> TRIGGERED"
                )
            else:
                # 模糊区域: vol between 0.5 and 1.5, conservative - still trigger
                logger.info(
                    f"[FLASH_CRASH] {asset}: AMBIGUOUS - price_move={move_pct:.1%}, "
                    f"vol_ratio={volume_ratio:.2f} (0.5~1.5, conservative trigger)"
                )

        return score
    
    def _check_all_conflict_flat(self, signal_data: Dict) -> Tuple[bool, float, str]:
        """Check for all-conflict-flat condition.

        [FIX] Iron Law #34: Sentiment NEVER vetoes trades. Previously sentiment_direction
        was included in conflict detection, causing NO_TRADE whenever F&G was extreme
        (sentiment_dir=-1.0) and quant/DRL disagreed. This effectively gave sentiment
        veto power through the conflict backdoor, blocking ALL trades for 10+ days.

        Now: only quant vs DRL conflict triggers NO_TRADE. Sentiment is excluded.
        """

        quant_dir = signal_data.get('quant_direction', 0)
        drl_dir = signal_data.get('drl_direction', 0)
        # [FIX] Sentiment excluded from conflict detection per Iron Law #34
        # sentiment_dir was -1.0 during F&G=16 (extreme fear), creating permanent
        # NO_TRADE when quant and DRL had any disagreement.

        # Only check quant vs DRL conflict (the two DECIDE agents)
        quant_strong = abs(quant_dir) > self.CONFLICT_DIRECTION_THRESHOLD
        drl_strong = abs(drl_dir) > self.CONFLICT_DIRECTION_THRESHOLD

        if not (quant_strong and drl_strong):
            return (False, 0.0, "")

        # Check for opposing directions between DECIDE agents only
        quant_sign = 1 if quant_dir > 0 else -1
        drl_sign = 1 if drl_dir > 0 else -1

        if quant_sign != drl_sign:
            details = f"Quant={quant_dir:.2f} vs DRL={drl_dir:.2f} (opposing DECIDE agents)"
            # Return score 0.7 instead of 1.0 — conflict but not extreme
            # (only two agents disagree, not three)
            return (True, 0.7, details)

        return (False, 0.0, "")


# =============================================================================
# OPPORTUNITY TRIGGER COMPUTATION (v3.3-STABILIZED guarantee #2)
# =============================================================================

class OpportunityTriggerGroup(Enum):
    """Opportunity trigger groups."""
    LEAD_LAG_EDGE = "A"
    CRACK_WINDOW = "B"
    VOLATILITY_EXPANSION = "C"
    SENTIMENT_SHOCK = "D"
    SOL_FLOW_SURGE = "E"


@dataclass
class OpportunityTriggerResult:
    """Result of an opportunity trigger check."""
    triggered: bool = False
    group: Optional[OpportunityTriggerGroup] = None
    confidence: float = 0.0
    direction: str = "none"
    reason: str = ""
    ttl_hours: float = 0.0
    can_set_direction: bool = False
    boosts_execution: bool = False


@dataclass
class OpportunityState:
    """Aggregated opportunity state."""
    is_active: bool = False
    active_triggers: List[OpportunityTriggerResult] = field(default_factory=list)
    primary_trigger: Optional[OpportunityTriggerResult] = None
    direction_consensus: str = "none"
    combined_confidence: float = 0.0
    earliest_expiry: Optional[datetime] = None
    ttl_remaining_bars: int = 0


class OpportunityTriggerChecker:
    """
    Internal OPPORTUNITY trigger computation with TTL/decay.
    
    From v3.3-STABILIZED core/opportunity_triggers.py
    """
    
    # === GROUP A: Lead-Lag Edge ===
    LEAD_LAG_NET_EDGE_BPS = 25
    LEAD_LAG_CONFIDENCE_MIN = 0.65
    LEAD_LAG_CVD_DIVERGENCE = 0.6
    LEAD_LAG_VPIN_MAX = 0.80
    LEAD_LAG_TTL_HOURS = 2
    LEAD_LAG_AGGRESSIVE_EDGE_BPS = 35
    
    # === GROUP B: CRACK Window ===
    CRACK_BREAK_PCT = 0.02  # 2%
    CRACK_VOLUME_RATIO = 2.5
    CRACK_ORDERBOOK_IMBALANCE = 0.4
    CRACK_TTL_HOURS = 8
    CRACK_DECAY_PER_BAR = 0.30
    
    # === GROUP C: Volatility Expansion ===
    VOL_DVOL_ZSCORE_MIN = 2.5
    VOL_DVOL_ZSCORE_MAX = 4.0
    VOL_TTL_HOURS = 4
    
    # === GROUP D: Sentiment Shock ===
    SENTIMENT_SHOCK_SIGMA = 2.0
    SENTIMENT_TTL_HOURS = 6
    
    # === GROUP E: SOL Flow ===
    SOL_DEX_VOLUME_RATIO = 3.0
    SOL_ONCHAIN_INFLOW_USD = 10_000_000  # [P0-FIX] was 50_000_000, too high for SOL flow volumes
    SOL_TTL_HOURS = 4
    
    def __init__(self):
        self._active_triggers: Dict[OpportunityTriggerGroup, Tuple[datetime, float]] = {}
    
    def compute_triggers(
        self,
        market_data: Dict,
        lead_lag_data: Dict,
        sentiment_data: Optional[Dict] = None,
        sol_flow_data: Optional[Dict] = None,
    ) -> OpportunityState:
        """
        Compute all opportunity triggers INTERNALLY.
        
        Includes TTL tracking and decay management.
        """
        
        results: List[OpportunityTriggerResult] = []
        now = datetime.now(timezone.utc)
        
        # === GROUP A: Lead-Lag Edge ===
        lead_lag_result = self._check_lead_lag_edge(lead_lag_data)
        if lead_lag_result.triggered:
            results.append(lead_lag_result)
            self._active_triggers[OpportunityTriggerGroup.LEAD_LAG_EDGE] = (now, lead_lag_result.confidence)
        
        # === GROUP B: CRACK Window ===
        crack_result = self._check_crack_window(market_data)
        if crack_result.triggered:
            results.append(crack_result)
            self._active_triggers[OpportunityTriggerGroup.CRACK_WINDOW] = (now, crack_result.confidence)
        
        # === GROUP C: Volatility Expansion ===
        vol_result = self._check_volatility_expansion(market_data)
        if vol_result.triggered:
            results.append(vol_result)
        
        # === GROUP D: Sentiment Shock ===
        if sentiment_data:
            sentiment_result = self._check_sentiment_shock(sentiment_data)
            if sentiment_result.triggered:
                results.append(sentiment_result)
        
        # === GROUP E: SOL Flow ===
        if sol_flow_data:
            sol_result = self._check_sol_flow(sol_flow_data)
            if sol_result.triggered:
                results.append(sol_result)
        
        # Check for decayed triggers still within TTL
        for group, (start_time, initial_conf) in list(self._active_triggers.items()):
            ttl_hours = self._get_ttl_hours(group)
            if (now - start_time).total_seconds() < ttl_hours * 3600:
                # Apply decay
                elapsed_bars = (now - start_time).total_seconds() / (4 * 3600)
                decayed_conf = initial_conf * (1 - self.CRACK_DECAY_PER_BAR * elapsed_bars)
                if decayed_conf > 0.3:  # Still significant
                    # Keep active even if not re-triggered
                    pass
            else:
                # TTL expired
                del self._active_triggers[group]
        
        # Aggregate results
        return self._aggregate_triggers(results)
    
    def _check_lead_lag_edge(self, data: Dict) -> OpportunityTriggerResult:
        """Check lead-lag edge trigger."""
        
        net_edge_bps = data.get('net_edge_bps', 0)
        edge_confidence = data.get('edge_window_confidence', 0)
        cvd_divergence = data.get('cvd_divergence', 0)
        vpin = data.get('vpin', 1.0)
        
        edge_ok = net_edge_bps >= self.LEAD_LAG_NET_EDGE_BPS
        confidence_ok = edge_confidence >= self.LEAD_LAG_CONFIDENCE_MIN
        cvd_ok = abs(cvd_divergence) >= self.LEAD_LAG_CVD_DIVERGENCE
        vpin_ok = vpin < self.LEAD_LAG_VPIN_MAX
        
        triggered = edge_ok and confidence_ok and cvd_ok and vpin_ok
        
        if triggered:
            boosts_aggressive = net_edge_bps >= self.LEAD_LAG_AGGRESSIVE_EDGE_BPS
            return OpportunityTriggerResult(
                triggered=True,
                group=OpportunityTriggerGroup.LEAD_LAG_EDGE,
                confidence=edge_confidence,
                direction="none",  # NEVER sets direction
                reason=f"Lead-Lag edge {net_edge_bps:.0f}bps",
                ttl_hours=self.LEAD_LAG_TTL_HOURS,
                can_set_direction=False,
                boosts_execution=boosts_aggressive
            )
        
        return OpportunityTriggerResult(triggered=False)
    
    def _check_crack_window(self, data: Dict) -> OpportunityTriggerResult:
        """Check CRACK window trigger."""
        
        structure_break_pct = data.get('structure_break_pct', 0)
        volume_ratio = effective_volume_ratio(data)
        orderbook_imbalance = data.get('orderbook_imbalance', data.get('order_book_imbalance', 0))
        structure_level = data.get('structure_level', 0)
        current_price = data.get('current_price', 0)
        
        break_ok = abs(structure_break_pct) >= self.CRACK_BREAK_PCT
        volume_ok = volume_ratio >= self.CRACK_VOLUME_RATIO
        
        # Determine break direction
        break_direction = "long" if current_price > structure_level else "short"
        
        # Imbalance must align
        if break_direction == "long":
            imbalance_ok = orderbook_imbalance >= self.CRACK_ORDERBOOK_IMBALANCE
        else:
            imbalance_ok = orderbook_imbalance <= -self.CRACK_ORDERBOOK_IMBALANCE
        
        triggered = break_ok and volume_ok and imbalance_ok
        
        if triggered:
            confidence = min(1.0, 0.5 + (volume_ratio / 5) * 0.25)
            return OpportunityTriggerResult(
                triggered=True,
                group=OpportunityTriggerGroup.CRACK_WINDOW,
                confidence=confidence,
                direction=break_direction,
                reason=f"CRACK {break_direction}: {structure_break_pct:.1%}",
                ttl_hours=self.CRACK_TTL_HOURS,
                can_set_direction=True,
                boosts_execution=True
            )
        
        return OpportunityTriggerResult(triggered=False)
    
    def _check_volatility_expansion(self, data: Dict) -> OpportunityTriggerResult:
        """Check volatility expansion trigger."""
        
        dvol_zscore = data.get('dvol_zscore', 0)
        
        if self.VOL_DVOL_ZSCORE_MIN <= dvol_zscore <= self.VOL_DVOL_ZSCORE_MAX:
            return OpportunityTriggerResult(
                triggered=True,
                group=OpportunityTriggerGroup.VOLATILITY_EXPANSION,
                confidence=0.6,
                direction="none",
                reason=f"DVOL expansion z={dvol_zscore:.1f}",
                ttl_hours=self.VOL_TTL_HOURS,
                can_set_direction=False,
                boosts_execution=False
            )
        
        return OpportunityTriggerResult(triggered=False)
    
    def _check_sentiment_shock(self, data: Dict) -> OpportunityTriggerResult:
        """Check sentiment shock trigger."""
        
        shock_sigma = data.get('shock_sigma', 0)
        shock_direction = data.get('shock_direction', 'none')
        
        if abs(shock_sigma) >= self.SENTIMENT_SHOCK_SIGMA:
            return OpportunityTriggerResult(
                triggered=True,
                group=OpportunityTriggerGroup.SENTIMENT_SHOCK,
                confidence=min(1.0, abs(shock_sigma) / 3.0),
                direction=shock_direction,
                reason=f"Sentiment shock {shock_sigma:.1f}σ",
                ttl_hours=self.SENTIMENT_TTL_HOURS,
                can_set_direction=True,
                boosts_execution=True
            )
        
        return OpportunityTriggerResult(triggered=False)
    
    def _check_sol_flow(self, data: Dict) -> OpportunityTriggerResult:
        """Check SOL flow surge trigger."""

        dex_volume_ratio = data.get('dex_volume_ratio', 1.0)
        onchain_inflow = data.get('onchain_inflow_4h', 0)
        whale_dir = data.get('whale_net_direction', 'none')

        flow_ok = dex_volume_ratio >= self.SOL_DEX_VOLUME_RATIO
        inflow_ok = abs(onchain_inflow) >= self.SOL_ONCHAIN_INFLOW_USD
        whale_active = whale_dir in ("buy", "sell")

        if flow_ok or inflow_ok or whale_active:
            # Direction: prefer inflow signal, fallback to whale direction
            if abs(onchain_inflow) > 0:
                direction = "long" if onchain_inflow > 0 else "short"
            elif whale_active:
                direction = "long" if whale_dir == "buy" else "short"
            else:
                direction = "long"  # DEX volume surge default
            # Confidence: base 0.6, boosted by confirming signals
            conf = 0.6
            if flow_ok:
                conf += 0.1
            if inflow_ok:
                conf += 0.1
            if whale_active:
                conf += 0.1
            conf = min(conf, 1.0)
            parts = []
            if flow_ok:
                parts.append(f"DEX {dex_volume_ratio:.1f}x")
            if inflow_ok:
                parts.append(f"inflow ${abs(onchain_inflow):,.0f}")
            if whale_active:
                parts.append(f"whale={whale_dir}")
            return OpportunityTriggerResult(
                triggered=True,
                group=OpportunityTriggerGroup.SOL_FLOW_SURGE,
                confidence=conf,
                direction=direction,
                reason=f"SOL flow: {', '.join(parts)}",
                ttl_hours=self.SOL_TTL_HOURS,
                can_set_direction=True,
                boosts_execution=True
            )

        return OpportunityTriggerResult(triggered=False)
    
    def _get_ttl_hours(self, group: OpportunityTriggerGroup) -> float:
        """Get TTL for trigger group."""
        ttl_map = {
            OpportunityTriggerGroup.LEAD_LAG_EDGE: self.LEAD_LAG_TTL_HOURS,
            OpportunityTriggerGroup.CRACK_WINDOW: self.CRACK_TTL_HOURS,
            OpportunityTriggerGroup.VOLATILITY_EXPANSION: self.VOL_TTL_HOURS,
            OpportunityTriggerGroup.SENTIMENT_SHOCK: self.SENTIMENT_TTL_HOURS,
            OpportunityTriggerGroup.SOL_FLOW_SURGE: self.SOL_TTL_HOURS,
        }
        return ttl_map.get(group, 4.0)
    
    def _aggregate_triggers(self, results: List[OpportunityTriggerResult]) -> OpportunityState:
        """Aggregate triggers into opportunity state."""
        
        if not results:
            return OpportunityState(is_active=False)
        
        # Find direction consensus
        directions_with_authority = [
            r.direction for r in results 
            if r.can_set_direction and r.direction != "none"
        ]
        
        if directions_with_authority:
            long_count = sum(1 for d in directions_with_authority if d == "long")
            short_count = sum(1 for d in directions_with_authority if d == "short")
            
            if long_count > 0 and short_count > 0:
                direction_consensus = "conflict"
            elif long_count > 0:
                direction_consensus = "long"
            elif short_count > 0:
                direction_consensus = "short"
            else:
                direction_consensus = "none"
        else:
            direction_consensus = "none"
        
        combined_confidence = max(r.confidence for r in results)
        
        # Find primary trigger
        direction_triggers = [r for r in results if r.can_set_direction and r.direction != "none"]
        if direction_triggers:
            primary = max(direction_triggers, key=lambda r: r.confidence)
        else:
            primary = max(results, key=lambda r: r.confidence)
        
        # Calculate TTL remaining
        max_ttl = max(r.ttl_hours for r in results)
        ttl_remaining_bars = int(max_ttl / 4)  # 4H bars
        
        return OpportunityState(
            is_active=True,
            active_triggers=results,
            primary_trigger=primary,
            direction_consensus=direction_consensus,
            combined_confidence=combined_confidence,
            ttl_remaining_bars=ttl_remaining_bars
        )


# =============================================================================
# ALPHA THRESHOLD GATING (v3.3-STABILIZED guarantee #3)
# =============================================================================

# [P327] MEASURED constant — registered in core.calibration_registry.
# Provenance lives there (producer command, source data, staleness horizon and
# revision rule) rather than in prose, because three separate incidents came
# from a measurement whose derivation could not be re-run (P326), whose age was
# invisible (P315), or whose source could not be questioned (P316).
_MEASURED_ON = "2026-08-16"
_MEASURED_BY = "scripts/coinbase_probe_stop_support.py (read-only CDE book probe, P289)"

# [P289] FULL measured CDE spread, rounded UP (P167: a lookup failure or a
# rounding choice must overcharge, never undercharge). The taker cost is the
# HALF-spread; these are the full figures the friction model halves.
CDE_SPREAD_BPS_MEASURED = {"BTC": 2.0, "ETH": 5.5, "SOL": 4.0}


@dataclass
class FrictionComponents:
    """
    Friction components for alpha calculation.

    Volume-aware: Below $10K/month (Kraken+ free tier) fees are 0.
    Above $10K: taker=26bps, maker=16bps (Kraken standard).
    [T2] Margin-aware: leveraged positions incur opening + rollover fees.
    """
    taker_fee_bps: float = 0.0    # Dynamic - updated per tick via update_for_volume()
    maker_fee_bps: float = 0.0    # Dynamic
    slippage_bps: float = 5.0     # Default; overridden per-asset by update_for_asset()
    latency_cost_bps: float = 2.0
    # [FIX-SPREAD] Per-asset spread based on observed market microstructure
    # BTC: deep book, 3bps typical. ETH: 5bps. SOL: thin book, 10bps.
    ASSET_SPREAD_BPS: dict = None  # Set in __post_init__
    # [P289] CDE (Coinbase derivatives) spread table + per-asset venue memory;
    # table populated in __post_init__ (default_factory, not the legacy
    # `dict = None` shape — that pattern is a standing mypy finding).
    # See the provenance comment in __post_init__.
    CDE_SPREAD_BPS: dict = field(default_factory=dict)
    _spread_venue_by_asset: dict = field(default_factory=dict)
    # [T2] Kraken margin fees (dynamic - set when leveraged)
    margin_opening_fee_bps: float = 0.0
    margin_rollover_bps_per_4h: float = 0.0
    expected_hold_periods_4h: float = 6.0  # Default 24h = 6 x 4h ticks
    # [P291] Venue-true HOLD cost. The three fields below exist because the
    # funding reading is DESTROYED by update_funding_rate's max() the moment
    # the Kraken margin rollover is larger (it always is at live funding
    # levels), so venue-true mode cannot recover it from
    # margin_rollover_bps_per_4h. `_funding_bps_per_4h = None` is the
    # NEVER-READ sentinel and is deliberately distinct from a real 0.0
    # reading: unknown funding must fall back to the Kraken charge, never
    # to a free hold (P2 absence-is-not-zero, P167 fail-expensive).
    # `_funding_asset` guards the cross-asset leak — FRICTION is ONE shared
    # object walked per asset each tick, so asset B must never be charged
    # asset A's funding (the P225 class).
    _funding_bps_per_4h: Optional[float] = None
    _funding_asset: Optional[str] = None
    _venue_true_hold_asset: Optional[str] = None
    # [P291] INDEPENDENT gate, default OFF. The venue memory this reuses is
    # shared with P289's spread pricing, which is already LIVE — but the
    # hold correction has a materially different consequence (it moves ETH
    # and SOL BELOW the 30bps asserted-alpha ceiling, i.e. it OPENS two
    # assets that are arithmetically locked out today), so it must not ride
    # the spread flag. Enabling is an operator decision (P141) with the
    # P237 tripwire question attached; until then this stays False and the
    # Kraken arithmetic is byte-identical to today.
    venue_true_hold_enabled: bool = False

    # Kraken fee schedule - [FIX-33] sourced from canonical_config
    KRAKEN_FREE_TIER_USD: float = _CC_FREE_TIER
    KRAKEN_TAKER_BPS: float = _CC_TAKER_BPS
    KRAKEN_MAKER_BPS: float = _CC_MAKER_BPS

    # [T2] Kraken margin fee schedule (bps per event)
    MARGIN_FEE_BPS: dict = None  # Set in __post_init__

    def __post_init__(self):
        if self.MARGIN_FEE_BPS is None:
            self.MARGIN_FEE_BPS = {"BTC": 1.0, "ETH": 2.0, "SOL": 2.0}
        if self.ASSET_SPREAD_BPS is None:
            self.ASSET_SPREAD_BPS = {"BTC": 3.0, "ETH": 5.0, "SOL": 10.0}
        # [P289] Venue-true spreads for Coinbase-routed assets. The Kraken-era
        # ASSET_SPREAD_BPS above was applied to CDE nano contracts for the
        # sleeve's whole life; a live read-only order-book probe (2026-08-16,
        # WEEKEND book = worst-case liquidity, 6 samples/contract) measured
        # median FULL spreads BTC 1.58 / ETH 5.26 / SOL 2.65 bps. The true
        # taker cost per leg is the HALF-spread; this table charges the FULL
        # spread rounded conservatively (a ~2x buffer, P167: overcharging is
        # the safe direction). Note ETH ROSE 5.0 -> 5.5 — the measurement is
        # honest in both directions, not a loosening-by-fiat. Consequence at
        # current constants: BTC threshold ~29 -> ~26bps (the 1-2bp epsilon
        # pass becomes robust); SOL ~55 -> ~40bps and ETH ~43bps — BOTH STILL
        # ABOVE the 30bps asserted-alpha ceiling, so this re-pricing does NOT
        # re-open ETH/SOL; only the P237 tripwire question can.
        if not self.CDE_SPREAD_BPS:
            # [P327] one home for the measured values (P172). Copied, not
            # aliased: this instance may be mutated per-venue at runtime and
            # must never write back into the module-level measurement.
            self.CDE_SPREAD_BPS = dict(CDE_SPREAD_BPS_MEASURED)

    def set_spread_venue(self, asset: str, venue: Optional[str]):
        """[P289] Remember which venue's book prices this asset's slippage.

        Called from the per-tick [VENUE-FEE] block (flag-gated in main.py).
        Absent/unknown venue -> Kraken table (the more expensive one — a
        lookup failure must overcharge, never undercharge, P167/P172).
        """
        asset_key = asset.upper().replace("/USD", "").replace("USD", "")
        v = (venue or "").strip().lower()
        if v == "coinbase":
            self._spread_venue_by_asset[asset_key] = "coinbase"
        else:
            self._spread_venue_by_asset.pop(asset_key, None)

    def update_for_asset(self, asset: str):
        """[FIX-SPREAD] Set slippage_bps to per-asset spread from market
        microstructure. [P289] Venue-aware: assets whose spread venue was set
        to coinbase (via set_spread_venue) price the CDE book instead of the
        Kraken-era constants; everything else keeps the Kraken table."""
        asset_key = asset.upper().replace("/USD", "").replace("USD", "")
        if self._spread_venue_by_asset.get(asset_key) == "coinbase":
            self.slippage_bps = self.CDE_SPREAD_BPS.get(
                asset_key, self.ASSET_SPREAD_BPS.get(asset_key, 5.0))
            # [P291] The SAME venue memory governs the hold term. Recording
            # which asset is being priced right now is what lets
            # _margin_cost_bps refuse a funding reading taken for a
            # different asset.
            self._venue_true_hold_asset = asset_key
        else:
            self.slippage_bps = self.ASSET_SPREAD_BPS.get(asset_key, 5.0)
            self._venue_true_hold_asset = None

    def update_for_volume(self, monthly_volume_usd: float):
        """Update fee components based on current monthly volume.

        Uses continuous linear blending matching KrakenPlusFeeBlender:
        - V <= FREE_TIER ($10K): fees = 0
        - FREE_TIER < V < FREE_TIER + BLEND_BAND ($12K): linear interpolation
        - V >= FREE_TIER + BLEND_BAND: full standard fees
        """
        _blend_band = 2000.0  # Match KrakenPlusFeeBlender.blend_band_usd
        if monthly_volume_usd <= self.KRAKEN_FREE_TIER_USD:
            self.taker_fee_bps = 0.0
            self.maker_fee_bps = 0.0
        elif monthly_volume_usd >= self.KRAKEN_FREE_TIER_USD + _blend_band:
            self.taker_fee_bps = self.KRAKEN_TAKER_BPS
            self.maker_fee_bps = self.KRAKEN_MAKER_BPS
        else:
            _weight = (monthly_volume_usd - self.KRAKEN_FREE_TIER_USD) / _blend_band
            self.taker_fee_bps = self.KRAKEN_TAKER_BPS * _weight
            self.maker_fee_bps = self.KRAKEN_MAKER_BPS * _weight

    def update_fee_bps(self, *, taker_fee_bps: float, maker_fee_bps: float):
        """Override fee bps from the shared execution fee model."""
        self.taker_fee_bps = max(0.0, float(taker_fee_bps or 0.0))
        self.maker_fee_bps = max(0.0, float(maker_fee_bps or 0.0))

    def update_for_leverage(self, asset: str, leverage: float):
        """[T2] Update margin friction when leveraged."""
        asset_key = asset.upper().replace("/USD", "").replace("USD", "")
        if leverage > 1.0:
            base_bps = self.MARGIN_FEE_BPS.get(asset_key, 2.0)
            self.margin_opening_fee_bps = base_bps
            self.margin_rollover_bps_per_4h = base_bps
        else:
            self.margin_opening_fee_bps = 0.0
            self.margin_rollover_bps_per_4h = 0.0

    def update_funding_rate(self, funding_rate_8h: float,
                            asset: Optional[str] = None):
        """Ingest live perpetual funding rate as holding cost.

        funding_rate_8h is the per-8h rate for the venue that will hold the
        position (P277 routes routed assets' own venue rate here).
        Converts to bps and adds to rollover cost so the alpha gate
        accounts for real funding drag on leveraged/perpetual positions.
        Consumed by cost model via _margin_cost_bps -> taker_friction_bps.

        [P291] `asset` is OPTIONAL and additive: passing it is what arms
        venue-true hold pricing for that asset (see _margin_cost_bps). A
        caller that omits it keeps today's behavior exactly — the
        venue-true path then cannot confirm which asset the reading belongs
        to and falls back to the Kraken charge, so forgetting to wire it
        OVERCHARGES rather than undercharges (P167).
        """
        # funding_rate_8h is a ratio (e.g. 0.0001 = 1bps per 8h)
        # Convert to bps: 0.0001 * 10000 = 1.0 bps
        # A 4H position pays ~half an 8H funding period
        funding_bps_per_4h = abs(funding_rate_8h) * 10000.0 * 0.5
        # [P291] abs() is kept deliberately: NEGATIVE funding means the book
        # would RECEIVE carry, and crediting that here would turn the hold
        # term into a gate subsidy. Charging |funding| is the conservative
        # reading of an uncertain future carry.
        self._funding_bps_per_4h = funding_bps_per_4h
        self._funding_asset = (
            asset.upper().replace("/USD", "").replace("USD", "")
            if asset else None)
        self.margin_rollover_bps_per_4h = max(
            self.margin_rollover_bps_per_4h,
            funding_bps_per_4h,
        )

    @property
    def _margin_cost_bps(self) -> float:
        """Total holding cost: opening fee + expected rollover fees.

        [P291] VENUE-TRUE branch. The Kraken terms above are a SPOT-MARGIN
        BORROW schedule (an opening fee plus a per-4h rollover on borrowed
        funds). CDE perpetual-style futures do not levy either: margin
        there is posted COLLATERAL (P275 probed
        INTRADAY_MARGIN_SETTING_STANDARD), and the venue's actual carry is
        FUNDING — already fed here by P277.

        Measured before this branch existed: with the live profile's
        `regime_leverage` overlay putting the two dominant regimes
        (WEAK_CONSOLIDATION / QUIET_ACCUMULATION, ~93% of ticks) at 2.0x,
        update_for_leverage took its leverage>1 branch and produced
        BTC 1.0 + 1.0x6 = 7.0bps and ETH/SOL 2.0 + 2.0x6 = 14.0bps — an
        exact match to the live gate lines' "margin=7" / "14.0bps hold".
        The funding P277 wired was simultaneously INERT on this path,
        because max() picks the Kraken rollover (1.0-2.0) over live funding
        (~0.04-1.9 bps/4h) every time. So routed assets were charged a fee
        the venue never levies, while the fee it does levy was masked.

        Conditions for the venue-true branch (ALL must hold, else the
        Kraken arithmetic stands unchanged — every fallback overcharges):
          * update_for_asset resolved this asset to the coinbase venue, AND
          * a funding reading exists (None = never read, NOT zero), AND
          * that reading was tagged with THIS asset (no cross-asset leak).
        """
        _vt_asset = self._venue_true_hold_asset
        if (self.venue_true_hold_enabled
                and _vt_asset is not None
                and self._funding_bps_per_4h is not None
                and self._funding_asset == _vt_asset):
            # Clamp at zero: funding is stored as |rate| above, but a
            # negative hold cost would be a gate SUBSIDY, so the floor is
            # asserted here too rather than relying on the caller.
            return max(0.0, float(self._funding_bps_per_4h)
                       * self.expected_hold_periods_4h)
        return self.margin_opening_fee_bps + (
            self.margin_rollover_bps_per_4h * self.expected_hold_periods_4h
        )

    @property
    def total_taker(self) -> float:
        """ONE LEG of friction. See `round_trip_bps` — a position pays this twice."""
        return self.taker_fee_bps + self.slippage_bps + self.latency_cost_bps + self._margin_cost_bps

    @property
    def total_maker(self) -> float:
        """ONE LEG of friction. See `round_trip_bps` — a position pays this twice."""
        return self.maker_fee_bps + self.slippage_bps + self.latency_cost_bps + self._margin_cost_bps

    def per_leg_bps(self, is_maker: bool = False) -> float:
        """[P167] Costs paid once per ORDER: exchange fee, spread crossed,
        latency slip. Excludes margin/funding, which is charged per HOLD."""
        fee = self.maker_fee_bps if is_maker else self.taker_fee_bps
        return fee + self.slippage_bps + self.latency_cost_bps

    def round_trip_bps(self, is_maker: bool = False, legs: float = 2.0) -> float:
        """[P167] What a position actually costs, entry through exit.

        The alpha gate compared a ROUND-TRIP alpha estimate against ONE LEG of
        friction. `signal_edge_bps = |quant_dir| * 65` was calibrated (see
        `data_mgmt/market_data_pipeline.py:1318`) from realized per-trade alpha
        — "+41bps avg win, 50% win rate -> effective ~20bps" — which is an
        entry-to-exit number. But `friction` summed a single fee and a single
        spread crossing, so the exit leg was free.

        Margin/funding is deliberately NOT multiplied: `_margin_cost_bps` is
        already opening_fee + rollover x expected_hold_periods_4h, i.e. the
        whole holding cost. Doubling it would overcharge.
        """
        return float(legs) * self.per_leg_bps(is_maker) + self._margin_cost_bps


@dataclass
class AlphaGatingResult:
    """Result of alpha threshold check."""
    passes_threshold: bool = False
    estimated_alpha_bps: float = 0.0
    threshold_bps: float = 0.0
    margin_bps: float = 0.0
    multiplier_used: float = 3.0
    reason: str = ""
    # P1-2A: Friction breakdown for proof log
    friction_fee_bps: float = 0.0
    friction_slippage_bps: float = 0.0
    friction_latency_bps: float = 0.0
    friction_margin_bps: float = 0.0
    friction_total_bps: float = 0.0
    # [P167] How many order legs `friction_total_bps` charges for. 2.0 = entry
    # and exit, which is what a round-trip alpha estimate must be compared
    # against. Defaults to 0.0, not 1.0, so the early-return paths (NO_TRADE,
    # QUIET_ACCUMULATION) read as "no friction was priced here" rather than
    # "one leg was priced" — absent must not look like a value.
    friction_legs: float = 0.0
    min_alpha_bps_used: float = 0.0
    ev_threshold_bps: float = 0.0       # friction * multiplier (before min_alpha floor)
    gate_decision: str = ""              # ALLOW / REJECT_MIN_ALPHA / REJECT_EV
    expected_net_alpha_bps: float = 0.0
    required_net_alpha_bps: float = 0.0


class AlphaThresholdCalculator:
    """
    Friction-aware alpha threshold gating.
    
    From v3.3-STABILIZED core/alpha_threshold.py
    
    CRITICAL: Blocks intent creation if alpha doesn't cover friction.
    """
    
    # Mode multipliers
    # v6.5.1 CTO audit: lowered from 3.0/2.0/2.5
    # [UNLEASH] UL-3: further lowered - 4H system should trade more freely
    # Original: 2.0/1.15/1.15/1.15. New: 1.5/1.0/1.0/1.0
    NORMAL_MULTIPLIER = 1.10            # was 1.25 -> 1.10 (2026-04-08: 1.25 blocked 95% signals in paper run, 4H system alpha typically 5-30bps)
    OPPORTUNITY_LEAD_LAG_MULTIPLIER = 1.0  # [UNLEASH] was 1.15 -> 1.0 (7bps = just friction)
    OPPORTUNITY_CRACK_MULTIPLIER = 1.0     # [UNLEASH] was 1.15 -> 1.0
    OPPORTUNITY_OTHER_MULTIPLIER = 1.0     # [UNLEASH] was 1.15 -> 1.0
    
    # Alpha estimation
    MAX_ALPHA_BPS = 200

    # [P167] A position is opened AND closed. Both legs pay fee + spread +
    # latency, so a round-trip alpha estimate must be charged for both.
    ROUND_TRIP_LEGS = 2.0
    
    def __init__(self):
        self._rolling_hit_rate = 0.5
        self.FRICTION = FrictionComponents()
        # [ALPHA-FEEDBACK 2026-06-13] Connect the realized-alpha feedback loop to
        # the override path (see check_alpha_gate). Default ON; disable with
        # HMATS_ALPHA_FEEDBACK=0 in .env + restart. Reversible escape hatch.
        self._alpha_feedback_enabled = os.environ.get("HMATS_ALPHA_FEEDBACK", "1") != "0"
        # [P167] Charge entry AND exit friction. Default ON: this TIGHTENS the
        # gate, and undercharging cost is the failure being corrected. The
        # escape hatch exists to restore the old arithmetic in one restart if
        # the tightening proves to have starved trading, not because the old
        # arithmetic was defensible.
        self._round_trip_friction_enabled = (
            os.environ.get("HMATS_ROUND_TRIP_FRICTION", "1") != "0"
        )
    
    def check_alpha_gate(
        self,
        signal_strength: float,
        regime_confidence: float,
        mode: str,  # "NO_TRADE", "OPPORTUNITY", "NORMAL"
        is_maker: bool = False,
        lead_lag_edge_confirmed: bool = False,
        crack_window_active: bool = False,
        min_alpha_bps: float = 0.0,
        direction: float = 0.0,  # [UNLEASH] UL-2b: short friction discount
        regime_alpha_gate_mult: float = 1.0,  # [v6.7-P1] regime-conditional threshold scaler
        quant_confidence: float = 0.5,  # [FIX-25] strategy signal confidence (primary)
        asset: str = "",         # [FIX-SPREAD] per-asset spread
        regime: str = "",        # [FIX-QA] regime for QUIET_ACCUMULATION filter
        quiet_accum_min_direction: float = 0.35,  # [P1] paper-baseline override
        quiet_accum_min_direction_short: Optional[float] = None,
        borderline_pass_epsilon_bps: float = 0.0,
        borderline_pass_epsilon_bps_short: Optional[float] = None,
        estimated_alpha_override: Optional[float] = None,  # [UNIFIED-ALPHA] pre-computed value from pipeline
    ) -> AlphaGatingResult:
        """
        Check if trade passes alpha threshold.

        BLOCKS execution if alpha < threshold.

        P1-2A: Two-layer gate:
          1. min_alpha_bps - configurable floor (profile-gated)
          2. EV gate - friction × multiplier (always active)
          effective_threshold = max(min_alpha_bps, friction × multiplier)
        """

        # NO_TRADE = infinite threshold
        if mode == "NO_TRADE":
            return AlphaGatingResult(
                passes_threshold=False,
                threshold_bps=float('inf'),
                reason="NO_TRADE mode - execution blocked",
                gate_decision="REJECT_NO_TRADE",
            )

        # [FIX-SPREAD] Update friction to per-asset spread before calculation
        if asset:
            self.FRICTION.update_for_asset(asset)

        # [FIX-QA] QUIET_ACCUMULATION + low direction = reject
        # Data: 52% of QUIET_ACCUMULATION fills had alpha < friction,
        # 100% of those had direction < 0.35. Low-direction signals in
        # low-volatility regimes have no edge after friction.
        _quiet_direction_floor = float(quiet_accum_min_direction or 0.0)
        if direction < 0 and quiet_accum_min_direction_short is not None:
            try:
                _quiet_direction_floor = float(quiet_accum_min_direction_short or 0.0)
            except Exception:
                _quiet_direction_floor = float(quiet_accum_min_direction or 0.0)
        if (
            regime in ("QUIET_ACCUMULATION", "REGIME_0")
            and _quiet_direction_floor > 0
            and abs(direction) < _quiet_direction_floor
        ):
            return AlphaGatingResult(
                passes_threshold=False,
                estimated_alpha_bps=abs(signal_strength) * self.MAX_ALPHA_BPS * 0.5,
                threshold_bps=float('inf'),
                reason=(
                    f"QUIET_ACCUMULATION + direction {abs(direction):.2f} "
                    f"< {_quiet_direction_floor:.2f} hard filter"
                ),
                gate_decision="REJECT_LOW_DIRECTION_QUIET",
            )

        # Determine multiplier
        if mode == "OPPORTUNITY":
            if lead_lag_edge_confirmed:
                multiplier = self.OPPORTUNITY_LEAD_LAG_MULTIPLIER
            elif crack_window_active:
                multiplier = self.OPPORTUNITY_CRACK_MULTIPLIER
            else:
                multiplier = self.OPPORTUNITY_OTHER_MULTIPLIER
        else:  # NORMAL
            multiplier = self.NORMAL_MULTIPLIER

        # Calculate friction components
        if is_maker:
            fee_bps = self.FRICTION.maker_fee_bps
        else:
            fee_bps = self.FRICTION.taker_fee_bps
        slippage_bps = self.FRICTION.slippage_bps
        latency_bps = self.FRICTION.latency_cost_bps
        margin_bps = float(self.FRICTION._margin_cost_bps or 0.0)
        # [P167] Charge BOTH legs. `estimated_alpha` is a round-trip number
        # (see FrictionComponents.round_trip_bps), so pricing a single fee and
        # a single spread crossing let every trade in on the assumption that
        # exiting is free. On SOL (10bps spread) that understated the true cost
        # by ~13bps against a threshold of 1.10x friction — the gate demanded
        # 1.10 legs where the position pays 2. Measured over the 85 closed
        # trades in data/trade_attribution.jsonl: gross_alpha -$179.51 vs fees
        # $689.77. Escape hatch: HMATS_ROUND_TRIP_FRICTION=0 (+ restart).
        friction_legs = self.ROUND_TRIP_LEGS if self._round_trip_friction_enabled else 1.0
        # [P253c] Route through FrictionComponents.per_leg_bps instead of
        # re-inlining fee+slippage+latency: the helpers existed with no
        # production caller while this line carried a hand-written copy of
        # the same arithmetic — two copies of the number that gates every
        # trade is one drift away from a silent P167 regression. NOT
        # round_trip_bps: that helper already ADDS _margin_cost_bps, so
        # calling it and adding margin_bps again double-charges margin —
        # the exact trap its own docstring warns about, caught by
        # test_margin_is_charged_once_not_twice on the first attempt.
        # margin_bps keeps its `or 0.0` guard (the helper's raw field read
        # would TypeError on None). The fee/slippage/latency locals above
        # are kept for the reason strings below.
        friction = (float(friction_legs) * self.FRICTION.per_leg_bps(is_maker)
                    + margin_bps)

        # EV gate threshold = friction × multiplier
        ev_threshold_bps = friction * multiplier

        # [UNLEASH] UL-2b: Short friction discount 20%
        # Crypto long-term upward bias -> shorts need lower threshold to compensate headwind
        if direction < 0:
            ev_threshold_bps *= 0.80

        # [v6.7-P1] Regime-conditional alpha gate multiplier
        # <1.0 = more permissive (PANIC_SELLOFF), >1.0 = more selective (MOMENTUM_RALLY)
        # [P225] Terminal clamp at the single consumption point. Six writers
        # stack multiplicatively into agent_signals["_regime_alpha_gate_mult"]
        # (RegimeAggressor assign, then SmartBeta / AlphaBoost /
        # EXTERNAL-COMPOSITE / EC-ORPHAN each `*=`, main.py) and each bounds
        # only its OWN factor — the size-path twin has had a [0.2, 2.0] clamp
        # at main.py's WIRE-REGIME-SIZE for months while this gate path, the
        # one that produces vetoes and (post-P206) sleeve flattens, had none.
        # Bounds chosen to be behavior-preserving for every composition
        # observed live (0.99–1.39): they only bite on runaway stacking.
        # A non-finite product fails toward NEUTRAL (1.0), not toward either
        # extreme — a poisoned multiplier must not become a silent trading
        # stop (mult→inf) or a disabled gate (mult→0).
        if regime_alpha_gate_mult != 1.0:
            _gm = float(regime_alpha_gate_mult)
            if not math.isfinite(_gm):
                logger.warning(
                    f"[P225-GATE-MULT] non-finite regime_alpha_gate_mult "
                    f"({regime_alpha_gate_mult!r}) — using neutral 1.0"
                )
                _gm = 1.0
            elif not (REGIME_GATE_MULT_MIN <= _gm <= REGIME_GATE_MULT_MAX):
                _clamped = min(REGIME_GATE_MULT_MAX, max(REGIME_GATE_MULT_MIN, _gm))
                logger.warning(
                    f"[P225-GATE-MULT] composed regime_alpha_gate_mult {_gm:.3f} "
                    f"outside [{REGIME_GATE_MULT_MIN}, {REGIME_GATE_MULT_MAX}] — "
                    f"clamped to {_clamped:.3f} (check the stacking writers in main.py)"
                )
                _gm = _clamped
            ev_threshold_bps *= _gm

        # P1-2A: Effective threshold = max(min_alpha floor, EV gate)
        effective_threshold = max(min_alpha_bps, ev_threshold_bps)

        # [UNIFIED-ALPHA] Single alpha estimation path.
        # When caller provides estimated_alpha_override (from market_data_pipeline's
        # calibrated signal_edge_bps), use it directly. This eliminates the dual-path
        # problem where Path 1 (×65, calibrated) and Path 2 (×200×conf×perf, uncalibrated)
        # produced different values. The override is calibrated against paper run data.
        if estimated_alpha_override is not None and estimated_alpha_override > 0:
            estimated_alpha = estimated_alpha_override
            # [ALPHA-FEEDBACK 2026-06-13] Close the realized-alpha feedback loop on
            # the override path. The override (signal_edge_bps = |quant_dir|*65) is
            # driven by the quant signal, which live-IC analysis showed is NOISE
            # (IC -0.018); the 65x was calibrated once from stale paper data and
            # never re-checked against realized. The system ALREADY tracks realized
            # wins via update_hit_rate() -> _rolling_hit_rate, but only the fallback
            # path consumed it (the override path threw it away). Apply the same
            # performance_factor here so the estimate self-corrects toward realized:
            # factor in [0.5, 1.0] — 1.0 when winning, shrinks toward 0.5 when the
            # signal stops working -> gate tightens on no-edge regimes. Floor 0.5
            # prevents over-tightening to zero (won't starve trading). The system's
            # measured live alpha is negative (-62%/yr), so a haircut here is the
            # correct direction. Reversible: HMATS_ALPHA_FEEDBACK=0.
            if getattr(self, "_alpha_feedback_enabled", True):
                _perf_factor = 0.5 + (self._rolling_hit_rate * 0.5)
                _alpha_pre = estimated_alpha
                estimated_alpha = estimated_alpha * _perf_factor
                logger.info(
                    f"[ALPHA-FEEDBACK] est {_alpha_pre:.1f}->{estimated_alpha:.1f}bps "
                    f"(perf_factor={_perf_factor:.2f}, hit_rate={self._rolling_hit_rate:.2f})"
                )
        else:
            # Fallback: internal calculation (kept for backward compatibility)
            effective_confidence = 0.7 * quant_confidence + 0.3 * regime_confidence
            base_alpha = abs(signal_strength) * self.MAX_ALPHA_BPS
            adjusted_alpha = base_alpha * effective_confidence
            performance_factor = 0.5 + (self._rolling_hit_rate * 0.5)
            estimated_alpha = adjusted_alpha * performance_factor

        # Net alpha view: compare expected net alpha against required net alpha.
        expected_net_alpha_bps = estimated_alpha - friction
        required_net_alpha_bps = effective_threshold - friction
        _borderline_epsilon = float(borderline_pass_epsilon_bps or 0.0)
        if direction < 0 and borderline_pass_epsilon_bps_short is not None:
            try:
                _borderline_epsilon = float(borderline_pass_epsilon_bps_short or 0.0)
            except Exception:
                _borderline_epsilon = float(borderline_pass_epsilon_bps or 0.0)
        passes = (expected_net_alpha_bps + _borderline_epsilon) >= required_net_alpha_bps
        margin = expected_net_alpha_bps - required_net_alpha_bps

        reason = ""
        gate_decision = "ALLOW"
        if not passes:
            if min_alpha_bps > ev_threshold_bps and estimated_alpha < min_alpha_bps:
                gate_decision = "REJECT_MIN_ALPHA"
                reason = (f"Alpha {estimated_alpha:.0f}bps < min_alpha_floor "
                          f"{min_alpha_bps:.0f}bps (EV threshold={ev_threshold_bps:.0f}bps)")
            else:
                gate_decision = "REJECT_EV"
                # [P167] Name the leg count: a rejection at 15bps reads very
                # differently depending on whether the exit was priced.
                reason = (f"Alpha {estimated_alpha:.0f}bps < threshold "
                          f"{effective_threshold:.0f}bps ({multiplier}x friction "
                          f"{friction:.1f}bps = {friction_legs:.0f}x"
                          f"{fee_bps + slippage_bps + latency_bps:.1f}bps/leg"
                          f"{f' + {margin_bps:.1f}bps hold' if margin_bps else ''})")
        elif margin < 0 and _borderline_epsilon > 0:
            gate_decision = "ALLOW_EPSILON"

        return AlphaGatingResult(
            passes_threshold=passes,
            estimated_alpha_bps=estimated_alpha,
            threshold_bps=effective_threshold,
            margin_bps=margin,
            multiplier_used=multiplier,
            reason=reason,
            friction_fee_bps=fee_bps,
            friction_slippage_bps=slippage_bps,
            friction_latency_bps=latency_bps,
            friction_margin_bps=margin_bps,
            friction_total_bps=friction,
            friction_legs=friction_legs,  # [P167]
            min_alpha_bps_used=min_alpha_bps,
            ev_threshold_bps=ev_threshold_bps,
            gate_decision=gate_decision,
            expected_net_alpha_bps=expected_net_alpha_bps,
            required_net_alpha_bps=required_net_alpha_bps,
        )
    
    def update_hit_rate(self, won: bool):
        """Update rolling hit rate."""
        # Simple EMA update
        alpha = 0.1
        result = 1.0 if won else 0.0
        self._rolling_hit_rate = alpha * result + (1 - alpha) * self._rolling_hit_rate


# =============================================================================
# TRANCHE SCHEDULING + ABORT (v3.3-STABILIZED guarantee #4)
# =============================================================================

from core.canonical_enums import TrancheLevel  # Single source of truth


@dataclass
class TranchePosition:
    """Current tranche position state."""
    asset: str
    level: TrancheLevel = TrancheLevel.NONE
    direction: str = "none"
    target_exposure: float = 0.0
    current_exposure: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl_bps: float = 0.0
    entered_at: Optional[datetime] = None
    last_escalation: Optional[datetime] = None


@dataclass
class AbortConditions:
    """Abort condition state."""
    vpin_spike: bool = False
    dvol_shock: bool = False
    signal_conflict: bool = False
    price_reversal: bool = False
    volume_collapse: bool = False
    no_trade_triggered: bool = False
    
    def any_triggered(self) -> bool:
        return any([
            self.vpin_spike, self.dvol_shock, self.signal_conflict,
            self.price_reversal, self.volume_collapse, self.no_trade_triggered
        ])

    def has_hard_trigger(self) -> bool:
        """Hard triggers justify FLATTEN even for existing positions.
        Volume collapse alone = HOLD (flattening in low volume = worse slippage)."""
        return any([
            self.vpin_spike, self.dvol_shock, self.price_reversal,
            self.no_trade_triggered
        ])
    
    def triggered_reasons(self) -> List[str]:
        reasons = []
        if self.vpin_spike: reasons.append("VPIN > 0.85")
        if self.dvol_shock: reasons.append("DVOL shock")
        if self.signal_conflict: reasons.append("Signal conflict")
        if self.price_reversal: reasons.append("Price reversal > threshold")
        if self.volume_collapse: reasons.append("Volume collapse")
        if self.no_trade_triggered: reasons.append("NO_TRADE mode")
        return reasons


@dataclass
class TrancheDecision:
    """Tranche scheduling decision."""
    action: str  # HOLD, ENTER_T1, ESCALATE, REDUCE, FLATTEN
    new_level: TrancheLevel = TrancheLevel.NONE
    target_exposure: float = 0.0
    exposure_delta: float = 0.0
    reason: str = ""
    abort_active: bool = False


class TrancheScheduler:
    """
    Tranche-based pyramiding with abort conditions.

    From v3.3-STABILIZED core/tranche_manager.py

    Schedule (default): 20% -> 50% -> 80% -> 100% (cumulative)

    P1-03: Accepts optional tranche_config dict for profile-gated overrides.
    """

    # Default tranche sizing (HIGH_RISK)
    _DEFAULT_TRANCHE_SIZES = {
        TrancheLevel.NONE: 0.0,
        TrancheLevel.TRANCHE_1: 0.35,  # [UNLEASH] UL-8d: was 0.20 -> 0.35 (heavier T1 entry)
        TrancheLevel.TRANCHE_2: 0.65,  # [UNLEASH] was 0.50 -> 0.65 (cumulative: T2 adds 30%)
        TrancheLevel.TRANCHE_3: 0.90,  # [UNLEASH] was 0.80 -> 0.90 (cumulative: T3 adds 25%)
        TrancheLevel.TRANCHE_4: 1.00,  # unchanged (cumulative: T4 adds 10%)
    }

    # Abort thresholds
    VPIN_SPIKE_THRESHOLD = 0.85
    DVOL_SHOCK_THRESHOLD = 1.5
    # Bug #43: Per-regime price reversal thresholds.
    # 1% fixed was too aggressive for 4H bars (BTC median 4H vol ~1.2-1.5%, SOL ~2-3%).
    # Enter->abort->flatten loop on almost every tick.
    PRICE_REVERSAL_THRESHOLDS = {
        "QUIET_ACCUMULATION":   0.02,   # Low vol regime, 2% is meaningful
        "WEAK_CONSOLIDATION":   0.02,
        "MOMENTUM_RALLY":       0.025,  # Trending, allow more room
        "VOLATILE_CHOP":        0.03,   # High vol, 3% needed
        "EXTREME_VOLATILITY":   0.04,   # Very high vol
        "PANIC_SELLOFF":        0.05,   # Extreme moves expected
    }
    PRICE_REVERSAL_THRESHOLD_DEFAULT = 0.025  # Fallback for unknown regimes
    VOLUME_COLLAPSE_THRESHOLD = 0.05  # Real liquidity crisis < 0.02. Weekend/Asia often 0.05-0.15

    # Default escalation requirements (HIGH_RISK)
    _DEFAULT_T3_MIN_PROFIT_BPS = 0
    _DEFAULT_T4_MIN_PROFIT_BPS = 50
    _DEFAULT_T4_MIN_TIME_HOURS = 2
    _DEFAULT_T2_REQUIRE_4H_CLOSE = True
    _DEFAULT_T3_REQUIRE_REGIME_ALIGNED = True
    _DEFAULT_ENTRY_VOLUME_COLLAPSE_THRESHOLD = 0.05
    _DEFAULT_ESCALATION_MIN_BARS = {"t1_to_t2": 1, "t2_to_t3": 0, "t3_to_t4": 0}
    _BAR_DURATION_SECONDS = 14400  # 4H

    def __init__(self, tranche_config: Optional[Dict] = None):
        self.positions: Dict[str, TranchePosition] = {}

        # P1-03: Apply profile-gated config overrides
        cfg = tranche_config or {}

        # Tranche sizing (cumulative) from config
        sizes_cfg = cfg.get("sizes_cumulative", {})
        self.TRANCHE_SIZES = dict(self._DEFAULT_TRANCHE_SIZES)
        if sizes_cfg:
            _level_map = {
                "TRANCHE_1": TrancheLevel.TRANCHE_1,
                "TRANCHE_2": TrancheLevel.TRANCHE_2,
                "TRANCHE_3": TrancheLevel.TRANCHE_3,
                "TRANCHE_4": TrancheLevel.TRANCHE_4,
            }
            for name, level in _level_map.items():
                if name in sizes_cfg:
                    self.TRANCHE_SIZES[level] = float(sizes_cfg[name])

        # Escalation parameters
        self.TRANCHE_3_MIN_PROFIT_BPS = cfg.get("t3_min_profit_bps", self._DEFAULT_T3_MIN_PROFIT_BPS)
        self.TRANCHE_4_MIN_PROFIT_BPS = cfg.get("t4_min_profit_bps", self._DEFAULT_T4_MIN_PROFIT_BPS)
        self.TRANCHE_4_MIN_TIME_HOURS = cfg.get("t4_min_time_hours", self._DEFAULT_T4_MIN_TIME_HOURS)
        self.t2_require_4h_close = cfg.get("t2_require_4h_close", self._DEFAULT_T2_REQUIRE_4H_CLOSE)
        self.t3_require_regime_aligned = cfg.get("t3_require_regime_aligned", self._DEFAULT_T3_REQUIRE_REGIME_ALIGNED)
        self.entry_volume_collapse_threshold = float(
            cfg.get(
                "entry_volume_collapse_threshold",
                self._DEFAULT_ENTRY_VOLUME_COLLAPSE_THRESHOLD,
            )
        )
        _entry_volume_collapse_threshold_by_asset = cfg.get(
            "entry_volume_collapse_threshold_by_asset",
            {},
        )
        self.entry_volume_collapse_threshold_by_asset = {}
        if isinstance(_entry_volume_collapse_threshold_by_asset, dict):
            for _asset, _threshold in _entry_volume_collapse_threshold_by_asset.items():
                try:
                    self.entry_volume_collapse_threshold_by_asset[str(_asset).upper()] = float(
                        _threshold
                    )
                except Exception:
                    continue

        # Minimum bars between escalations
        esc_bars = cfg.get("escalation_min_bars", self._DEFAULT_ESCALATION_MIN_BARS)
        self._min_bars_t1_to_t2 = esc_bars.get("t1_to_t2", 1)
        self._min_bars_t2_to_t3 = esc_bars.get("t2_to_t3", 0)
        self._min_bars_t3_to_t4 = esc_bars.get("t3_to_t4", 0)
    
    def _bars_since_last_escalation(self, position: TranchePosition, now: datetime) -> int:
        """Return number of full bars elapsed since last escalation."""
        if not position.last_escalation:
            return 999  # No prior escalation -> always pass
        elapsed_s = (now - position.last_escalation).total_seconds()
        return int(elapsed_s // self._BAR_DURATION_SECONDS)

    @staticmethod
    def _effective_volume_ratio(market_data: Dict, is_4h_bar_close: bool = False) -> float:
        """Use pace-adjusted volume on partial 4H bars to avoid false collapse reads."""
        return effective_volume_ratio(
            market_data,
            is_4h_bar_close=is_4h_bar_close,
        )

    def get_position(self, asset: str) -> TranchePosition:
        """Get or create position."""
        if asset not in self.positions:
            self.positions[asset] = TranchePosition(asset=asset)
        return self.positions[asset]

    def _get_entry_volume_collapse_threshold(self, asset: str) -> float:
        asset_key = str(asset or "").upper()
        override = self.entry_volume_collapse_threshold_by_asset.get(asset_key)
        if override is None:
            return self.entry_volume_collapse_threshold
        try:
            return float(override)
        except Exception:
            return self.entry_volume_collapse_threshold
    
    def decide(
        self,
        asset: str,
        mode: str,  # "NO_TRADE", "OPPORTUNITY", "NORMAL"
        target_direction: float,
        market_data: Dict,
        signal_data: Dict,
        is_4h_bar_close: bool = False,
    ) -> TrancheDecision:
        """
        Decide tranche action.
        
        Returns decision with abort handling.
        """
        
        position = self.get_position(asset)

        # v6.5.2 FIX: Skip abort checks when no position exists.
        # You cannot "abort" a position that doesn't exist.
        # Without this, volume_collapse (vol_ratio < 0.3) blocks initial entry forever.
        #
        # v6.5.2b FIX: Also skip abort for freshly-entered positions (< 4h / 1 bar).
        # Without this, cold-start enters T1 on tick N, then tick N+1 aborts on
        # partial-bar volume_ratio (~0.01), creating an enter->abort->flatten loop.
        # NO_TRADE always forces flatten/block, regardless of position age
        if mode == "NO_TRADE":
            if position.level != TrancheLevel.NONE:
                abort = AbortConditions()
                abort.no_trade_triggered = True
                return self._handle_abort(position, abort)
            else:
                return TrancheDecision(
                    action="FLATTEN",
                    new_level=TrancheLevel.NONE,
                    target_exposure=0.0,
                    reason="NO_TRADE mode: no entry permitted"
                )

        # v6.5.2b FIX: Skip market-based abort checks for freshly-entered positions (< 4h / 1 bar).
        # Without this, cold-start enters T1 on tick N, then tick N+1 aborts on
        # partial-bar volume_ratio (~0.01), creating an enter->abort->flatten loop.
        _position_age_hours = 0.0
        if position.level != TrancheLevel.NONE and position.entered_at:
            _entered = position.entered_at
            if _entered.tzinfo is None:
                _entered = _entered.replace(tzinfo=timezone.utc)
            _position_age_hours = (datetime.now(timezone.utc) - _entered).total_seconds() / 3600.0

        if position.level != TrancheLevel.NONE and _position_age_hours >= 4.0:
            # Check abort conditions (only for existing positions held >= 1 bar)
            abort = self._check_abort_conditions(position, market_data, signal_data)
            if abort.any_triggered():
                # Soft triggers (volume_collapse, signal_conflict) alone -> HOLD existing positions.
                # Flattening in low volume creates worse slippage than holding.
                # Only hard triggers (VPIN spike, DVOL shock, price reversal) justify FLATTEN.
                if abort.has_hard_trigger():
                    return self._handle_abort(position, abort)
                else:
                    logger.info(
                        f"[TRANCHE] {asset}: soft abort ({', '.join(abort.triggered_reasons())}) "
                        f"- HOLD existing T{position.level.value} (no FLATTEN in low volume)"
                    )
                    # Fall through to normal HOLD/OPPORTUNITY logic below
        
        # Update P&L
        current_price = market_data.get('current_price', 0)
        if position.entry_price > 0 and current_price > 0:
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            if position.direction == "short":
                pnl_pct = -pnl_pct
            position.unrealized_pnl_bps = pnl_pct * 10000
        
        # NORMAL mode: no early entry
        if mode == "NORMAL":
            # [FIX-NORMAL-OVERRIDE] If position already exists at T1+ (entered in
            # OPPORTUNITY mode), do NOT call _execute_full_position which would jump
            # straight to T4, bypassing T3 profit check (pnl>=0bps), T4 profit check
            # (pnl>=50bps), T4 time check (>=2h), and all bars_since requirements.
            # Instead, fall through to _manage_opportunity_tranches for safe escalation.
            if position.level == TrancheLevel.NONE:
                if is_4h_bar_close:
                    return self._execute_full_position(position, target_direction)
                else:
                    return TrancheDecision(
                        action="HOLD",
                        new_level=position.level,
                        target_exposure=position.current_exposure,
                        reason="NORMAL mode: waiting for 4H bar close"
                    )
            # else: existing position — fall through to OPPORTUNITY escalation logic

        # OPPORTUNITY mode (or NORMAL with existing position): pyramiding allowed
        return self._manage_opportunity_tranches(
            position, target_direction, market_data, signal_data, is_4h_bar_close
        )
    
    def _check_abort_conditions(
        self,
        position: TranchePosition,
        market_data: Dict,
        signal_data: Dict,
    ) -> AbortConditions:
        """Check all abort conditions."""

        abort = AbortConditions()

        # VPIN spike
        vpin = market_data.get('vpin', 0)
        if vpin > self.VPIN_SPIKE_THRESHOLD:
            abort.vpin_spike = True

        # DVOL shock (rapid increase)
        dvol_change = market_data.get('dvol_1h_change', 0)
        if dvol_change > self.DVOL_SHOCK_THRESHOLD:
            abort.dvol_shock = True

        # Price reversal - per-regime dynamic threshold (Bug #43)
        regime = market_data.get('regime_state', '')
        reversal_threshold = self.PRICE_REVERSAL_THRESHOLDS.get(
            regime, self.PRICE_REVERSAL_THRESHOLD_DEFAULT
        )
        if position.entry_price > 0:
            current_price = market_data.get('current_price', 0)
            if current_price > 0:
                reversal = abs(current_price - position.entry_price) / position.entry_price
                if position.direction == "long" and current_price < position.entry_price:
                    if reversal > reversal_threshold:
                        abort.price_reversal = True
                elif position.direction == "short" and current_price > position.entry_price:
                    if reversal > reversal_threshold:
                        abort.price_reversal = True
        
        # Volume collapse - with validity guard for partial bars
        # If volume_ratio < 0.05, likely a partial (currently-forming) bar, not real collapse.
        # Real volume collapse means vol dropped 70% from 20-bar average - vol_ratio ~0.1-0.3.
        # Partial bars on Kraken show vol_ratio ~0.00-0.02 which is data artifact, not market signal.
        volume_ratio = self._effective_volume_ratio(market_data, is_4h_bar_close=False)
        # [P265] DELIBERATELY DISABLED, made explicit. The old condition was
        # `< VOLUME_COLLAPSE_THRESHOLD and >= 0.05` — with the threshold AT
        # 0.05 that is an EMPTY INTERVAL, i.e. this trigger has never fired,
        # silently. Making it fire is not a bugfix: the comment above records
        # that sub-0.05 readings are partial-bar DATA ARTIFACTS (exactly what
        # `0 < ratio < 0.05` would fire on) while a real collapse reads
        # ~0.1-0.3 — ABOVE the threshold — so the trigger's calibration is
        # incoherent as designed. Re-arming needs a re-derived threshold from
        # measured volume-ratio distributions (an activation decision, P141),
        # not a predicate flip. (abort.volume_collapse simply keeps its
        # False default — no code fires it.)
        
        # Signal conflict
        if signal_data.get('signal_conflict', False):
            abort.signal_conflict = True
        
        return abort
    
    def _handle_abort(
        self,
        position: TranchePosition,
        abort: AbortConditions,
    ) -> TrancheDecision:
        """Handle abort condition - reduce or flatten."""
        
        reasons = abort.triggered_reasons()
        
        # If at T3 or T4, reduce to T1
        if position.level in [TrancheLevel.TRANCHE_3, TrancheLevel.TRANCHE_4]:
            # [FIX-ABORT-EXP] Scale T1 target by position magnitude, not raw TRANCHE_SIZES.
            # Raw TRANCHE_SIZES[T1]=0.35 would INCREASE a 15% position to 35%.
            # Correct: T1_exposure = (T1_size / T4_size) * current_exposure
            _t_current_size = self.TRANCHE_SIZES.get(position.level, 1.0)
            _magnitude = position.current_exposure / _t_current_size if _t_current_size > 0 else position.current_exposure
            new_exposure = self.TRANCHE_SIZES[TrancheLevel.TRANCHE_1] * _magnitude
            return TrancheDecision(
                action="REDUCE",
                new_level=TrancheLevel.TRANCHE_1,
                target_exposure=new_exposure,
                exposure_delta=new_exposure - position.current_exposure,
                reason=f"Abort: {', '.join(reasons)} - reducing to T1",
                abort_active=True
            )
        
        # Otherwise flatten
        return TrancheDecision(
            action="FLATTEN",
            new_level=TrancheLevel.NONE,
            target_exposure=0.0,
            exposure_delta=-position.current_exposure,
            reason=f"Abort: {', '.join(reasons)} - flattening",
            abort_active=True
        )
    
    def _execute_full_position(
        self,
        position: TranchePosition,
        target_direction: float,
    ) -> TrancheDecision:
        """Execute full position at 4H close (NORMAL mode)."""
        
        target_exposure = abs(target_direction)
        direction = "long" if target_direction > 0 else "short"
        
        position.level = TrancheLevel.TRANCHE_4
        position.direction = direction
        position.target_exposure = target_exposure
        position.current_exposure = target_exposure
        position.entered_at = datetime.now(timezone.utc)
        
        return TrancheDecision(
            action="ENTER_FULL",
            new_level=TrancheLevel.TRANCHE_4,
            target_exposure=target_exposure,
            exposure_delta=target_exposure,
            reason="NORMAL mode: full position at 4H close"
        )
    
    def _manage_opportunity_tranches(
        self,
        position: TranchePosition,
        target_direction: float,
        market_data: Dict,
        signal_data: Dict,
        is_4h_bar_close: bool,
    ) -> TrancheDecision:
        """Manage tranches in OPPORTUNITY mode."""
        
        direction = "long" if target_direction > 0 else "short"
        now = datetime.now(timezone.utc)
        
        # Not in position - enter T1
        if position.level == TrancheLevel.NONE:
            # Guard: refuse entry on synthetic/fallback data (stub prices would corrupt entry_price)
            entry_price = market_data.get('current_price', 0)
            if entry_price <= 0 or market_data.get('_source') == 'synthetic_fallback' or not market_data.get('data_valid', True):
                logger.warning(f"[TRANCHE] {position.asset}: refusing T1 entry on invalid data (price={entry_price}, source={market_data.get('_source', 'unknown')})")
                return TrancheDecision(
                    action="HOLD",
                    new_level=TrancheLevel.NONE,
                    target_exposure=0.0,
                    exposure_delta=0.0,
                    reason="HOLD: invalid market data, cannot enter"
                )

            # Guard: refuse NEW entry during volume collapse (but existing positions HOLD, see decide())
            #
            # [P334] DELIBERATELY does NOT pass `is_4h_bar_close`, and that is
            # the whole fix. The flag has NO PRODUCER (P173: nothing writes the
            # key, so `market_data.get("is_4h_bar_close", True)` in main.py is
            # always True), and `effective_volume_ratio` returns the RAW ratio
            # when it is True — so this call site, alone of eleven in the tree,
            # threw away the partial-bar correction its own producer had just
            # computed.
            #
            # Why that inverts the guard. `volume_ratio` is
            # current-bar-volume / SMA(FULL-bar volumes), and the 4H tick fires
            # ~10 min into a new bar, so bar_progress is ~0.04 EVERY tick. With
            # true volume pace k the raw ratio is ~k*0.04:
            #     k=1.0 (normal)          raw 0.040 -> REFUSED (false positive)
            #     k=0.5                   raw 0.020 -> REFUSED
            #     k=0.2 (severe collapse) raw 0.008 -> ALLOWED (below the
            #                                          >=0.01 artifact clause)
            # i.e. normal volume was blocked and the worse the collapse the
            # more likely it passed. Observed live 2026-08-20 04:11:44: ETH
            # refused at raw=0.021 while its certified book was long and had
            # cleared the alpha gate (66bps vs 55).
            #
            # The producer's correction fixes both directions (normal 0.040 ->
            # 0.200 allow; severe 0.008 -> 0.040 refuse) and is SELF-DISABLING
            # on a complete bar (`bar_progress >= 1.0` -> effective == raw), so
            # the flag was redundant with information market_data already
            # carries — which is why the other ten call sites never pass it.
            _vol_ratio = self._effective_volume_ratio(market_data)
            _entry_volume_threshold = self._get_entry_volume_collapse_threshold(position.asset)
            if _vol_ratio < _entry_volume_threshold and _vol_ratio >= 0.01:
                _raw_vol_ratio = float(market_data.get("volume_ratio", _vol_ratio) or 0.0)
                _bar_progress = float(market_data.get("bar_progress_4h", 1.0) or 1.0)
                logger.warning(
                    f"[TRANCHE] {position.asset}: refusing T1 entry in volume collapse "
                    f"(effective={_vol_ratio:.3f}, raw={_raw_vol_ratio:.3f}, "
                    f"bar_progress={_bar_progress:.2f})"
                )
                return TrancheDecision(
                    action="HOLD",
                    new_level=TrancheLevel.NONE,
                    target_exposure=0.0,
                    exposure_delta=0.0,
                    reason=(
                        f"HOLD: volume collapse (effective={_vol_ratio:.3f}, "
                        f"raw={_raw_vol_ratio:.3f}, bar_progress={_bar_progress:.2f}, "
                        f"threshold={_entry_volume_threshold:.3f})"
                    )
                )

            target_exposure = self.TRANCHE_SIZES[TrancheLevel.TRANCHE_1] * abs(target_direction)

            position.level = TrancheLevel.TRANCHE_1
            position.direction = direction
            position.target_exposure = target_exposure
            position.current_exposure = target_exposure
            position.entry_price = entry_price
            position.entered_at = now
            position.last_escalation = now

            t1_pct = int(self.TRANCHE_SIZES[TrancheLevel.TRANCHE_1] * 100)
            return TrancheDecision(
                action="ENTER_T1",
                new_level=TrancheLevel.TRANCHE_1,
                target_exposure=target_exposure,
                exposure_delta=target_exposure,
                reason=f"OPPORTUNITY: entering T1 ({t1_pct}%)"
            )

        # At T1 - escalate to T2
        # P1-03: t2_require_4h_close=False skips 4H close gate; min_bars gate replaces it
        if position.level == TrancheLevel.TRANCHE_1:
            bars_since = self._bars_since_last_escalation(position, now)
            structure_confirmed = signal_data.get('structure_confirmed', False)

            gate_ok = False
            if self.t2_require_4h_close:
                # HIGH_RISK default: need 4H close or structure confirmation
                gate_ok = is_4h_bar_close or structure_confirmed
            else:
                # ULTRA: no 4H close needed, just min bars
                gate_ok = True

            if gate_ok and bars_since >= self._min_bars_t1_to_t2:
                t2_pct = int(self.TRANCHE_SIZES[TrancheLevel.TRANCHE_2] * 100)
                target_exposure = self.TRANCHE_SIZES[TrancheLevel.TRANCHE_2] * abs(target_direction)
                delta = target_exposure - position.current_exposure

                position.level = TrancheLevel.TRANCHE_2
                position.current_exposure = target_exposure
                position.last_escalation = now

                return TrancheDecision(
                    action="ESCALATE_T2",
                    new_level=TrancheLevel.TRANCHE_2,
                    target_exposure=target_exposure,
                    exposure_delta=delta,
                    reason=f"OPPORTUNITY: escalating to T2 ({t2_pct}%)"
                )

        # At T2 - escalate to T3 if profitable
        # P1-03: t3_require_regime_aligned=False skips regime gate; min_bars gate added
        if position.level == TrancheLevel.TRANCHE_2:
            bars_since = self._bars_since_last_escalation(position, now)
            profit_ok = position.unrealized_pnl_bps >= self.TRANCHE_3_MIN_PROFIT_BPS
            regime_ok = True
            if self.t3_require_regime_aligned:
                regime_ok = signal_data.get('regime_aligned', True)

            if profit_ok and regime_ok and bars_since >= self._min_bars_t2_to_t3:
                t3_pct = int(self.TRANCHE_SIZES[TrancheLevel.TRANCHE_3] * 100)
                target_exposure = self.TRANCHE_SIZES[TrancheLevel.TRANCHE_3] * abs(target_direction)
                delta = target_exposure - position.current_exposure

                position.level = TrancheLevel.TRANCHE_3
                position.current_exposure = target_exposure
                position.last_escalation = now

                return TrancheDecision(
                    action="ESCALATE_T3",
                    new_level=TrancheLevel.TRANCHE_3,
                    target_exposure=target_exposure,
                    exposure_delta=delta,
                    reason=f"OPPORTUNITY: escalating to T3 ({t3_pct}%), PnL={position.unrealized_pnl_bps:.0f}bps"
                )

        # At T3 - escalate to T4 if profitable and time/bar requirement met
        # P1-03: min_bars gate replaces fixed time requirement
        if position.level == TrancheLevel.TRANCHE_3:
            bars_since = self._bars_since_last_escalation(position, now)
            time_in_position = (now - position.entered_at).total_seconds() / 3600 if position.entered_at else 0

            if (position.unrealized_pnl_bps >= self.TRANCHE_4_MIN_PROFIT_BPS and
                time_in_position >= self.TRANCHE_4_MIN_TIME_HOURS and
                bars_since >= self._min_bars_t3_to_t4):
                t4_pct = int(self.TRANCHE_SIZES[TrancheLevel.TRANCHE_4] * 100)
                target_exposure = self.TRANCHE_SIZES[TrancheLevel.TRANCHE_4] * abs(target_direction)
                delta = target_exposure - position.current_exposure

                position.level = TrancheLevel.TRANCHE_4
                position.current_exposure = target_exposure
                position.last_escalation = now

                return TrancheDecision(
                    action="ESCALATE_T4",
                    new_level=TrancheLevel.TRANCHE_4,
                    target_exposure=target_exposure,
                    exposure_delta=delta,
                    reason=f"OPPORTUNITY: escalating to T4 ({t4_pct}%), PnL={position.unrealized_pnl_bps:.0f}bps"
                )
        
        # Default: hold current position
        return TrancheDecision(
            action="HOLD",
            new_level=position.level,
            target_exposure=position.current_exposure,
            reason=f"Holding at {position.level.name}"
        )


# =============================================================================
# STRATEGY CONSTITUTION ENFORCEMENT (v3.3-STABILIZED guarantee #6)
# =============================================================================

class ConstitutionRule(Enum):
    """Rules that can be disabled/softened."""
    SNR_FILTER = auto()
    REGIME_PERSISTENCE = auto()
    DRL_ENTROPY_THRESHOLD = auto()
    CONFIRMATION_WAIT = auto()
    BTC_ETH_ANCHORS = auto()
    POSITION_CAP = auto()


@dataclass
class RuleStatus:
    """Status of a constitution rule."""
    rule: ConstitutionRule
    is_disabled: bool = False
    is_softened: bool = False
    original_value: Any = None
    softened_value: Any = None
    reason: str = ""


class StrategyConstitutionEnforcer:
    """
    Enforces which rules are disabled/softened per mode.
    
    From v3.3-STABILIZED core/strategy_enforcement.py
    """
    
    # Rules disabled in OPPORTUNITY mode
    DISABLED_IN_OPPORTUNITY = {
        ConstitutionRule.SNR_FILTER,
        ConstitutionRule.REGIME_PERSISTENCE,
        ConstitutionRule.CONFIRMATION_WAIT,
        ConstitutionRule.BTC_ETH_ANCHORS,
    }
    
    # Rules softened in OPPORTUNITY mode
    SOFTENED_IN_OPPORTUNITY = {
        ConstitutionRule.DRL_ENTROPY_THRESHOLD: (0.7, 0.5),  # (original, softened)
        ConstitutionRule.POSITION_CAP: (0.8, 1.0),
    }
    
    def get_rule_status(
        self,
        rule: ConstitutionRule,
        mode: str,
    ) -> RuleStatus:
        """Get status of a rule for current mode."""
        
        if mode == "OPPORTUNITY":
            if rule in self.DISABLED_IN_OPPORTUNITY:
                return RuleStatus(
                    rule=rule,
                    is_disabled=True,
                    reason=f"{rule.name} disabled in OPPORTUNITY mode"
                )
            
            if rule in self.SOFTENED_IN_OPPORTUNITY:
                original, softened = self.SOFTENED_IN_OPPORTUNITY[rule]
                return RuleStatus(
                    rule=rule,
                    is_softened=True,
                    original_value=original,
                    softened_value=softened,
                    reason=f"{rule.name} softened in OPPORTUNITY: {original} -> {softened}"
                )
        
        return RuleStatus(rule=rule)
    
    def is_rule_active(self, rule: ConstitutionRule, mode: str) -> bool:
        """Check if a rule is active (not disabled)."""
        status = self.get_rule_status(rule, mode)
        return not status.is_disabled
    
    def get_param_value(self, rule: ConstitutionRule, mode: str) -> Any:
        """Get current parameter value (original or softened)."""
        status = self.get_rule_status(rule, mode)
        if status.is_softened:
            return status.softened_value
        if rule in self.SOFTENED_IN_OPPORTUNITY:
            return self.SOFTENED_IN_OPPORTUNITY[rule][0]  # Original
        return None


# =============================================================================
# UNIFIED GUARANTEES MODULE
# =============================================================================

class ConstitutionGuaranteesModule:
    """
    Unified module providing all v3.3-STABILIZED guarantees.
    
    Wire this into HMATSv36Engine.decide() for internal computation.
    """
    
    def __init__(self, tranche_config: Optional[Dict] = None):
        self.data_validator = MarketDataValidator()
        self.no_trade_checker = NoTradeTriggerChecker()
        self.opportunity_checker = OpportunityTriggerChecker()
        self.alpha_calculator = AlphaThresholdCalculator()
        self.tranche_scheduler = TrancheScheduler(tranche_config=tranche_config)
        self.constitution_enforcer = StrategyConstitutionEnforcer()

        logger.info("[v3.6.1] ConstitutionGuaranteesModule initialized with all v3.3-STABILIZED guarantees")
    
    def validate_market_data(self, market_data: Dict) -> MarketDataValidationResult:
        """Validate market data before processing."""
        return self.data_validator.validate(market_data)
    
    def compute_no_trade_triggers(
        self,
        market_data: Dict,
        signal_data: Dict,
    ) -> NoTradeState:
        """Compute NO_TRADE triggers internally."""
        return self.no_trade_checker.compute_triggers(market_data, signal_data)
    
    def compute_opportunity_triggers(
        self,
        market_data: Dict,
        lead_lag_data: Dict,
        sentiment_data: Optional[Dict] = None,
        sol_flow_data: Optional[Dict] = None,
    ) -> OpportunityState:
        """Compute OPPORTUNITY triggers internally."""
        return self.opportunity_checker.compute_triggers(
            market_data, lead_lag_data, sentiment_data, sol_flow_data
        )
    
    def check_alpha_gate(
        self,
        signal_strength: float,
        regime_confidence: float,
        mode: str,
        **kwargs,
    ) -> AlphaGatingResult:
        """Check if trade passes alpha threshold."""
        return self.alpha_calculator.check_alpha_gate(
            signal_strength, regime_confidence, mode,
            quant_confidence=kwargs.pop("quant_confidence", 0.5),  # [FIX-25]
            **kwargs,
        )
    
    def schedule_tranche(
        self,
        asset: str,
        mode: str,
        target_direction: float,
        market_data: Dict,
        signal_data: Dict,
        is_4h_bar_close: bool = False,
    ) -> TrancheDecision:
        """Get tranche scheduling decision."""
        return self.tranche_scheduler.decide(
            asset, mode, target_direction, market_data, signal_data, is_4h_bar_close
        )
    
    def is_rule_active(self, rule: ConstitutionRule, mode: str) -> bool:
        """Check if constitution rule is active."""
        return self.constitution_enforcer.is_rule_active(rule, mode)
    
    def get_proof_summary(self) -> Dict:
        """Get summary for proof logging."""
        return {
            "no_trade_triggers_computed": True,
            "opportunity_triggers_computed": True,
            "alpha_gating_active": True,
            "tranche_scheduling_active": True,
            "constitution_enforcement_active": True,
            "data_validation_active": True,
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_guarantees_module: Optional[ConstitutionGuaranteesModule] = None
_guarantees_tranche_config: Optional[Dict] = None


def get_constitution_guarantees(tranche_config: Optional[Dict] = None) -> ConstitutionGuaranteesModule:
    """Get singleton guarantees module.

    Refresh the singleton when a different tranche_config is explicitly
    supplied later in the same process.
    """
    global _guarantees_module, _guarantees_tranche_config
    normalized_config = dict(tranche_config or {})
    if _guarantees_module is None:
        _guarantees_module = ConstitutionGuaranteesModule(tranche_config=tranche_config)
        _guarantees_tranche_config = normalized_config
    elif tranche_config is not None and (_guarantees_tranche_config or {}) != normalized_config:
        logger.info("ConstitutionGuarantees tranche_config changed; refreshing singleton instance")
        _guarantees_module = ConstitutionGuaranteesModule(tranche_config=tranche_config)
        _guarantees_tranche_config = normalized_config
    return _guarantees_module


def reset_constitution_guarantees() -> None:
    """Reset singleton guarantees module."""
    global _guarantees_module, _guarantees_tranche_config
    _guarantees_module = None
    _guarantees_tranche_config = None

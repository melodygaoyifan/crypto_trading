"""
================================================================================
HMATS v6.1.0 - SOTA FLAGS (Unified Configuration)
================================================================================
Purpose: Centralized feature flags for all SOTA modules

All flags can be overridden by environment variables:
    export HMATS_ENABLE_SOTA_RISK_CONTROLLER=true
    
Or by config file: configs/sota_flags.json

SAFETY PRINCIPLE:
    - P0 flags default to True (fail-closed safety)
    - P1/P2/P3 flags can be True but must be toggleable
    
================================================================================
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(f"HMATS_{key}", "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    return default


@dataclass
class SOTAFlags:
    """
    SOTA Module Flags - Centralized Control
    
    P0 (Safety Critical - Default True):
        - ENABLE_SOTA_RISK_CONTROLLER: Kill switch + drawdown + correlation collapse
        - ENABLE_TRADE_GATE: Pre-execution gate (NO_TRADE/EXIT_ONLY)
        - ENABLE_EXECUTION_GUARDS: Order clipping + staleness check
        - ENABLE_RATE_LIMIT_MANAGER: API rate limiting for Kraken
        - ENABLE_HUMAN_OVERRIDE: Manual override mechanism
        - ENABLE_SHADOW_LEDGER: Audit trail for all intents/orders/fills
    
    P1 (Strategy Enhancement - Default True, Can Disable):
        - ENABLE_TWO_STAGE_INTELLIGENCE: Heavy/Light model for direction
        - ENABLE_STRATEGY_WEIGHTING: Adaptive strategy weights
        - ENABLE_REGIME_SHORT_FILTER: Block shorts in bull regimes
    
    P2 (Verification - Default True for verify mode):
        - ENABLE_WALKFORWARD_VERIFY: Walk-forward validation
        - ENABLE_STATISTICAL_GATE: Statistical significance gate
        - ENABLE_EXTREME_VERIFY_SUITE: Extreme scenario testing
    
    P3 (Integration Layer - Default True):
        - ENABLE_SOTA_INTEGRATION_LAYER: Full SOTA v5.2.1 integration
        - ENABLE_DASHBOARD_LIVE_STATE: Dashboard shows real runtime state
    """
    
    # =========================================================================
    # P0: SAFETY CRITICAL (Default True - Fail-Closed)
    # =========================================================================
    
    ENABLE_SOTA_RISK_CONTROLLER: bool = True
    ENABLE_TRADE_GATE: bool = True
    ENABLE_EXECUTION_GUARDS: bool = True
    ENABLE_RATE_LIMIT_MANAGER: bool = True
    ENABLE_HUMAN_OVERRIDE: bool = True
    ENABLE_SHADOW_LEDGER: bool = True
    
    # =========================================================================
    # P1: STRATEGY ENHANCEMENT (Default True, Can Disable)
    # =========================================================================
    
    ENABLE_TWO_STAGE_INTELLIGENCE: bool = True
    ENABLE_STRATEGY_WEIGHTING: bool = True
    ENABLE_REGIME_SHORT_FILTER: bool = True
    
    # =========================================================================
    # P1.5: LONG-TERM SURVIVAL GOVERNORS (v6.1.3)
    # =========================================================================
    
    # Opportunity Budget Governor — [ACTIVATE] prevents overconcentration in OPPORTUNITY windows
    ENABLE_OPPORTUNITY_BUDGET_GOVERNOR: bool = True

    # Regime Transition Buffer 鈥?DISABLED: 1 tick delay is too slow for 4H bars
    ENABLE_REGIME_TRANSITION_BUFFER: bool = False

    # Cascade Exhaustion Governor - keep enabled as safety net
    ENABLE_CASCADE_EXHAUSTION_GOVERNOR: bool = True
    
    # =========================================================================
    # P1.6: SHADOW MODE OBSERVERS (v6.1.4)
    # =========================================================================
    # Shadow 妯″潡鍙厑璁? 璁＄畻 鈫?璁板綍 鈫?灞曠ず
    # 鉂?涓嶈兘瑙﹀彂 entry / veto / 鏀瑰彉浠撲綅
    
    # LeadLag Shadow - L0 瑙傚療绾?(Observation-only)
    ENABLE_LEAD_LAG_SHADOW: bool = True

    # WhaleDetector - Exchange-level large order detection (搂20)
    # 3-dim detection (ABSOLUTE + RELATIVE_SIZE + DEPTH_IMPACT) + 4 behavioral patterns
    # Applies soft confidence penalty (脳0.70) when whale pattern opposes trade direction
    ENABLE_WHALE_DETECTOR: bool = True

    # =========================================================================
    # P1.7: HIGH-RISK GAMBLER MODE (v6.2.1)
    # =========================================================================
    # 鈿狅笍 WARNING: This mode increases risk exposure significantly
    # 鉂?Does NOT modify: core signal logic, final stop loss, governors
    # 鉁?ONLY modifies: entry timing, anti-martingale speed, position limits
    
    # Master switch - [UNLEASH] UL-9a: ENABLED (was False)
    # Prerequisite: C4 BUGFIX verified (60% hard ceiling present)
    ENABLE_HIGH_RISK_GAMBLER_MODE: bool = True  # [UNLEASH] UL-9a: was False 鈫?True
    
    # =========================================================================
    # P1.8: KRAKEN+ FEE BLENDING (v6.2.2)
    # =========================================================================
    # Kraken+ spot trading fee blending based on monthly volume
    # Only applies to Kraken Spot trades (not Margin/Futures)
    
    # Enable/disable fee blending
    ENABLE_KRAKEN_PLUS_FEE_BLENDING: bool = True
    
    # Free tier threshold: no fees below this monthly volume (USD)
    KRAKEN_PLUS_FREE_TIER_USD: float = 10_000.0
    
    # Blend band: transition zone from free_tier to full fees (USD)
    # Full fees apply at: free_tier + blend_band = 12,000 USD
    KRAKEN_PLUS_BLEND_BAND_USD: float = 2_000.0
    
    # [DEAD-FLAG-CLEANUP 2026-04-15] removed: ENABLE_PASSIVE_AGGRESSIVE_SHADOW (0 refs)
    
    # SolDex Monitor Shadow - L1 鐗瑰緛婧?(Feature-only)
    ENABLE_SOLDEX_MONITOR_SHADOW: bool = True
    
    # =========================================================================
    # P1.9: PARTIAL CONSENSUS ENTRY (v6.2.3)
    # =========================================================================
    # Allows reduced-size T1 entries when partial (non-conflicting) consensus exists
    # 鉂?Does NOT bypass: veto logic, risk checks, trade gate, stop logic
    # 鉂?Does NOT enable: Gambler Mode, leverage increase, T2/T3/T4 escalation
    # 鉁?ONLY allows: Smaller T1 position when no directional conflict
    
    # Master switch - Default FALSE for safety (conservative)
    ENABLE_PARTIAL_CONSENSUS_ENTRY: bool = False
    
    # Maximum scale factor for partial consensus (relative to normal T1)
    # 0.5 = 50% of normal T1 size maximum
    PARTIAL_CONSENSUS_MAX_SCALE: float = 0.5
    
    # Minimum scale factor (below this, don't bother trading)
    PARTIAL_CONSENSUS_MIN_SCALE: float = 0.3
    
    # Minimum confidence from aligned agent to allow partial consensus
    PARTIAL_CONSENSUS_MIN_CONFIDENCE: float = 0.6
    
    # =========================================================================
    # P1.10: SHORT-BIAS AGENT (v6.2.3)
    # =========================================================================
    # Specialized agent that ONLY outputs SHORT signals
    # - Identifies: Parabolic tops, extreme funding, liquidation cascades, extreme greed
    # - direction: Always -1 or 0 (NEVER positive)
    # - veto_direction: Always "long" (blocks all LONG signals when active)
    # - Participates in Authority Fusion with CONFIRM authority
    
    # [DEAD-FLAG-CLEANUP 2026-04-15] removed: ENABLE_SHORT_BIAS_AGENT (0 refs)
    
    # Maximum confidence output (prevents over-conviction)
    SHORT_BIAS_MAX_CONFIDENCE: float = 0.85
    
    # Minimum confidence to emit signal
    SHORT_BIAS_MIN_CONFIDENCE: float = 0.40
    
    # Signal cooldown period (minutes)
    SHORT_BIAS_COOLDOWN_MINUTES: int = 30
    
    # Maximum consecutive short signals before pause
    SHORT_BIAS_MAX_CONSECUTIVE: int = 3
    
    # =========================================================================
    # P1.11: BACKFILL MODULES (v6.2.3 Patch)
    # =========================================================================
    # Restored modules from earlier versions, gated behind feature flags.
    # These provide enhanced regime detection, structure analysis, and
    # optional strategy allocation.
    
    # Enhanced Regime Navigator - Advanced regime switching with squeeze risk
    # Provides: regime_label, regime_confidence, squeeze_risk, risk_multiplier_hint
    # Safe to enable - no external dependencies
    ENABLE_ENHANCED_REGIME_NAVIGATOR: bool = True
    
    # Structure Analyzer - Microstructure / liquidity / sweep risk signals
    # Provides: liquidity_risk, sweep_risk, slippage_hint, structure_bias
    # Safe to enable - derives metrics from OHLCV data
    ENABLE_STRUCTURE_ANALYZER: bool = True
    
    # Strategy Allocator - Risk budget distribution across agents/signals
    # [ACTIVATE] Enable for risk budget distribution
    ENABLE_STRATEGY_ALLOCATOR: bool = True
    
    # On-chain Graph Alpha Agent - Solana on-chain analysis
    # Default OFF - requires on-chain data providers
    ENABLE_ONCHAIN_GRAPH_ALPHA: bool = True
    
    # BitBeast Rules Layer - Enhanced signal filtering rules
    # Default OFF - extends BitBeast engine with additional rules
    ENABLE_BITBEAST_RULES: bool = True

    # Microstructure Agent 鈥?cross-exchange lag, OB imbalance, taker flow
    # Default OFF 鈥?ADVISE authority, enriches market_data + agent_signals
    ENABLE_MICROSTRUCTURE_AGENT: bool = True

    # Solana On-Chain Agent 鈥?DEX flow, whale activity (Birdeye/Solscan)
    # Default OFF 鈥?SOL-only ADVISE, requires API keys
    ENABLE_ONCHAIN_SOLANA_AGENT: bool = True

    # Kraken Integrity Shield 鈥?CRC32 orderbook checksum + REST fallback
    # Default OFF 鈥?SHADOW monitor, needs WS orderbook feed for full validation
    ENABLE_INTEGRITY_SHIELD: bool = True
    
    # =========================================================================
    # P2: VERIFICATION (Default True for verify/backtest modes)
    # =========================================================================
    
    # [DEAD-FLAG-CLEANUP 2026-04-15] removed: ENABLE_EXTREME_VERIFY_SUITE (0 refs)
    
    # =========================================================================
    # P3: INTEGRATION & DASHBOARD (Default True)
    # =========================================================================
    
    # [DEAD-FLAG-CLEANUP 2026-04-15] removed: ENABLE_DASHBOARD_LIVE_STATE (0 refs)
    
    # =========================================================================
    # ADDITIONAL RUNTIME OPTIONS
    # =========================================================================
    
    # Shadow ledger output path
    SHADOW_LEDGER_PATH: str = "data/shadow_ledger.jsonl"
    
    # Human override file path (check this file for override commands)
    HUMAN_OVERRIDE_PATH: str = "data/human_override.json"
    
    # Verify suite output path
    VERIFY_OUTPUT_PATH: str = "reports/verify_output"
    
    # Rate limit config
    RATE_LIMIT_REQUESTS_PER_SECOND: float = 1.0
    RATE_LIMIT_BURST_SIZE: int = 5
    
    # SOTA Risk Controller thresholds
    RISK_MAX_DRAWDOWN_PCT: float = 0.35    # 35% (was 15%) 鈥?profit maximization
    RISK_DAILY_LOSS_HALT_PCT: float = 0.10  # 10% (was 5%)
    RISK_CORRELATION_COLLAPSE_THRESHOLD: float = 0.3

    # 鈺愨晲鈺?Leverage (Single Source of Truth) 鈺愨晲鈺?    MAX_LEVERAGE: float = 3.0              # Hard cap, all assets
    REGIME_LEVERAGE_VOLATILE_CHOP: float = 3.0
    REGIME_LEVERAGE_MOMENTUM_RALLY: float = 2.0
    REGIME_LEVERAGE_PANIC_SELLOFF: float = 2.0
    REGIME_LEVERAGE_DEFAULT: float = 1.0

    # 鈺愨晲鈺?Alpha Gate (Single Source of Truth) 鈺愨晲鈺?    ALPHA_GATE_NORMAL_MULTIPLIER: float = 1.5     # NORMAL mode friction multiplier
    ALPHA_GATE_OPPORTUNITY_MULTIPLIER: float = 1.0 # OPPORTUNITY mode friction multiplier
    ALPHA_GATE_FREE_TIER_NORMAL_BPS: float = 14.0  # Free tier NORMAL threshold (bps)
    ALPHA_GATE_FREE_TIER_OPP_BPS: float = 8.0      # Free tier OPPORTUNITY threshold (bps)
    
    def __post_init__(self):
        """Override from environment variables."""
        for field_name in self.__dataclass_fields__:
            if field_name.startswith("ENABLE_"):
                env_val = _env_bool(field_name, getattr(self, field_name))
                setattr(self, field_name, env_val)
    
    @classmethod
    def from_file(cls, path: str = "configs/sota_flags.json") -> 'SOTAFlags':
        """Load flags from JSON file."""
        flags = cls()
        
        config_path = Path(path)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                
                for key, value in data.items():
                    if hasattr(flags, key):
                        setattr(flags, key, value)
                
                logger.info(f"Loaded SOTA flags from {path}")
            except Exception as e:
                logger.warning(f"Failed to load SOTA flags from {path}: {e}")
        
        return flags
    
    def to_dict(self) -> Dict[str, Any]:
        """Export flags as dictionary."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
    
    def save(self, path: str = "configs/sota_flags.json"):
        """Save current flags to file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved SOTA flags to {path}")
    
    def log_status(self):
        """Log all flag values."""
        logger.info("=" * 60)
        logger.info("SOTA FLAGS STATUS")
        logger.info("=" * 60)
        
        categories = {
            "P0 (Safety)": ["ENABLE_SOTA_RISK_CONTROLLER", "ENABLE_TRADE_GATE", 
                          "ENABLE_EXECUTION_GUARDS", "ENABLE_RATE_LIMIT_MANAGER",
                          "ENABLE_HUMAN_OVERRIDE", "ENABLE_SHADOW_LEDGER"],
            "P1 (Strategy)": ["ENABLE_TWO_STAGE_INTELLIGENCE", "ENABLE_STRATEGY_WEIGHTING",
                             "ENABLE_REGIME_SHORT_FILTER"],
            "P1.5 (Governors)": ["ENABLE_OPPORTUNITY_BUDGET_GOVERNOR", 
                                "ENABLE_REGIME_TRANSITION_BUFFER",
                                "ENABLE_CASCADE_EXHAUSTION_GOVERNOR"],
            "P1.6 (Shadow)": ["ENABLE_LEAD_LAG_SHADOW",
                             "ENABLE_SOLDEX_MONITOR_SHADOW"],
            "P1.7 (High-Risk)": ["ENABLE_HIGH_RISK_GAMBLER_MODE"],
            "P1.8 (Fee Blending)": ["ENABLE_KRAKEN_PLUS_FEE_BLENDING"],
            "P1.9 (Partial Consensus)": ["ENABLE_PARTIAL_CONSENSUS_ENTRY"],
            # [DEAD-FLAG-CLEANUP 2026-04-15] removed P2 (Verify) + P3 (Integration)
            # categories — flags ENABLE_WALKFORWARD_VERIFY, ENABLE_STATISTICAL_GATE,
            # ENABLE_EXTREME_VERIFY_SUITE, ENABLE_SOTA_INTEGRATION_LAYER,
            # ENABLE_DASHBOARD_LIVE_STATE all removed (0 refs across codebase).
        }
        
        for category, flags in categories.items():
            logger.info(f"\n{category}:")
            for flag in flags:
                value = getattr(self, flag)
                status = "ON" if value else "OFF"
                logger.info(f"  {flag}: {status}")
        
        logger.info("=" * 60)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_sota_flags: Optional[SOTAFlags] = None


def get_sota_flags() -> SOTAFlags:
    """Get singleton SOTA flags instance."""
    global _sota_flags
    if _sota_flags is None:
        _sota_flags = SOTAFlags.from_file()
    return _sota_flags


def reset_sota_flags():
    """Reset singleton (for testing)."""
    global _sota_flags
    _sota_flags = None


def set_flag(flag_name: str, value: bool) -> bool:
    """Set a specific flag at runtime."""
    flags = get_sota_flags()
    if hasattr(flags, flag_name):
        setattr(flags, flag_name, value)
        logger.info(f"[SOTA FLAGS] {flag_name} = {value}")
        return True
    return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_p0_enabled() -> bool:
    """Check if all P0 safety flags are enabled."""
    flags = get_sota_flags()
    return all([
        flags.ENABLE_SOTA_RISK_CONTROLLER,
        flags.ENABLE_TRADE_GATE,
        flags.ENABLE_EXECUTION_GUARDS,
        flags.ENABLE_SHADOW_LEDGER,
    ])


def is_p1_enabled() -> bool:
    """Check if all P1 strategy flags are enabled."""
    flags = get_sota_flags()
    return all([
        flags.ENABLE_TWO_STAGE_INTELLIGENCE,
        flags.ENABLE_STRATEGY_WEIGHTING,
        flags.ENABLE_REGIME_SHORT_FILTER,
    ])


def disable_all_trading():
    """Emergency: disable all new trades."""
    flags = get_sota_flags()
    # We don't disable flags, we just ensure trade gate blocks
    logger.critical("[EMERGENCY] All trading disabled via flags")
    # This would be handled by human_override or kill switch


if __name__ == "__main__":
    # Test/debug: print all flags
    flags = get_sota_flags()
    flags.log_status()
    
    # Save default config
    flags.save()
    print("\nDefault config saved to configs/sota_flags.json")

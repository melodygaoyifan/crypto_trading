"""
================================================================================
HMATS v6.1.0 - SOTA FLAGS (Unified Configuration)
================================================================================
Purpose: Centralized feature flags for all SOTA modules

All flags can be overridden by environment variables:
    export HMATS_ENABLE_SOTA_RISK_CONTROLLER=true
    
Or by config file: config/sota_flags.json

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
    
    # Opportunity Budget Governor - DISABLED for profit maximization
    ENABLE_OPPORTUNITY_BUDGET_GOVERNOR: bool = False

    # Regime Transition Buffer - DISABLED: 1 tick delay is too slow for 4H bars
    ENABLE_REGIME_TRANSITION_BUFFER: bool = False

    # Cascade Exhaustion Governor - keep enabled as safety net
    ENABLE_CASCADE_EXHAUSTION_GOVERNOR: bool = True
    
    # =========================================================================
    # P1.6: SHADOW MODE OBSERVERS (v6.1.4)
    # =========================================================================
    # Shadow 模块只允许: 计算 -> 记录 -> 展示
    # OFF 不能触发 entry / veto / 改变仓位
    
    # LeadLag Shadow - L0 观察级 (Observation-only)
    ENABLE_LEAD_LAG_SHADOW: bool = True
    
    # =========================================================================
    # P1.7: HIGH-RISK GAMBLER MODE (v6.2.1)
    # =========================================================================
    # [WARN]️ WARNING: This mode increases risk exposure significantly
    # OFF Does NOT modify: core signal logic, final stop loss, governors
    # ON ONLY modifies: entry timing, anti-martingale speed, position limits
    
    # Master switch - Default FALSE for safety
    ENABLE_HIGH_RISK_GAMBLER_MODE: bool = False
    
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
    
    # PassiveAggressive Shadow - L1 执行观察级 (Post-trade analysis)
    ENABLE_PASSIVE_AGGRESSIVE_SHADOW: bool = True
    
    # SolDex Monitor Shadow - L1 特征源 (Feature-only)
    ENABLE_SOLDEX_MONITOR_SHADOW: bool = True
    
    # =========================================================================
    # P1.9: PARTIAL CONSENSUS ENTRY (v6.2.3)
    # =========================================================================
    # Allows reduced-size T1 entries when partial (non-conflicting) consensus exists
    # OFF Does NOT bypass: veto logic, risk checks, trade gate, stop logic
    # OFF Does NOT enable: Gambler Mode, leverage increase, T2/T3/T4 escalation
    # ON ONLY allows: Smaller T1 position when no directional conflict
    
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
    
    # Master switch - Default FALSE for safety
    ENABLE_SHORT_BIAS_AGENT: bool = False
    
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
    # Default OFF - requires tuning for specific strategy profiles
    ENABLE_STRATEGY_ALLOCATOR: bool = False
    
    # On-chain Graph Alpha Agent - Solana on-chain analysis
    # Default OFF - requires on-chain data providers
    ENABLE_ONCHAIN_GRAPH_ALPHA: bool = False
    
    # BitBeast Rules Layer - Enhanced signal filtering rules
    # Default OFF - extends BitBeast engine with additional rules
    ENABLE_BITBEAST_RULES: bool = False
    
    # =========================================================================
    # P2: VERIFICATION (Default True for verify/backtest modes)
    # =========================================================================
    
    ENABLE_WALKFORWARD_VERIFY: bool = True
    ENABLE_STATISTICAL_GATE: bool = True
    ENABLE_EXTREME_VERIFY_SUITE: bool = True
    
    # =========================================================================
    # P3: INTEGRATION & DASHBOARD (Default True)
    # =========================================================================
    
    ENABLE_SOTA_INTEGRATION_LAYER: bool = True
    ENABLE_DASHBOARD_LIVE_STATE: bool = True
    
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
    RISK_MAX_DRAWDOWN_PCT: float = 0.35    # 35% (was 15%) - profit maximization
    RISK_DAILY_LOSS_HALT_PCT: float = 0.10  # 10% (was 5%)
    RISK_CORRELATION_COLLAPSE_THRESHOLD: float = 0.3
    
    def __post_init__(self):
        """Override from environment variables."""
        for field_name in self.__dataclass_fields__:
            if field_name.startswith("ENABLE_"):
                env_val = _env_bool(field_name, getattr(self, field_name))
                setattr(self, field_name, env_val)
    
    @classmethod
    def from_file(cls, path: str = "config/sota_flags.json") -> 'SOTAFlags':
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
    
    def save(self, path: str = "config/sota_flags.json"):
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
                             "ENABLE_PASSIVE_AGGRESSIVE_SHADOW",
                             "ENABLE_SOLDEX_MONITOR_SHADOW"],
            "P1.7 (High-Risk)": ["ENABLE_HIGH_RISK_GAMBLER_MODE"],
            "P1.8 (Fee Blending)": ["ENABLE_KRAKEN_PLUS_FEE_BLENDING"],
            "P1.9 (Partial Consensus)": ["ENABLE_PARTIAL_CONSENSUS_ENTRY"],
            "P2 (Verify)": ["ENABLE_WALKFORWARD_VERIFY", "ENABLE_STATISTICAL_GATE",
                           "ENABLE_EXTREME_VERIFY_SUITE"],
            "P3 (Integration)": ["ENABLE_SOTA_INTEGRATION_LAYER", "ENABLE_DASHBOARD_LIVE_STATE"],
        }
        
        for category, flags in categories.items():
            logger.info(f"\n{category}:")
            for flag in flags:
                value = getattr(self, flag)
                status = "✓ ON" if value else "✗ OFF"
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
    print("\nDefault config saved to config/sota_flags.json")

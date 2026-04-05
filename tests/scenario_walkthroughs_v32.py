#!/usr/bin/env python3
"""
================================================================================
HMATS v3.2 - SCENARIO WALKTHROUGHS
================================================================================
Two required scenarios demonstrating mode changes, tranche rules, and TradeIntent.

Scenario 1: Fast SOL Bullish Move (OPPORTUNITY long, SOL dominance)
Scenario 2: Fast SOL Bearish Move (OPPORTUNITY short, SOL dominance)
================================================================================
"""

import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s'
)
logger = logging.getLogger("HMATS.Scenario")


def scenario_1_sol_bullish():
    """
    SCENARIO 1: Fast SOL Bullish Move
    
    Setup:
    - SOL breaks above key resistance ($200)
    - Volume surge (3× average)
    - CVD strongly positive (retail buying)
    - Sentiment shock +2.5σ
    
    Expected Flow:
    1. CRACK window triggers (structure break)
    2. Mode -> OPPORTUNITY
    3. Strategy -> bull_aggressive (promoted)
    4. Tranche-1 entry (20% of target)
    5. After 4H close -> escalate to Tranche-2 (50%)
    6. SOL dominance allowed (up to 0.9 exposure)
    """
    print("=" * 70)
    print("SCENARIO 1: FAST SOL BULLISH MOVE")
    print("=" * 70)
    
    from core.market_regime_v32 import EnhancedRegimeNavigator, MarketRegimeState
    from core.signal_generators_v32 import (
        get_quant_generator, get_drl_generator, compute_signal_conflict
    )
    from core.tranche_manager import get_tranche_manager, TrancheLevel
    from core.sol_exposure_manager import get_sol_exposure_manager, ExposureMode
    from core.system_mode import SystemMode
    from core.trade_intent import IntentFactory, IntentAuthority, IntentUrgency
    import numpy as np
    
    print("\n--- Initial State ---")
    
    regime_nav = EnhancedRegimeNavigator()
    quant_gen = get_quant_generator()
    drl_gen = get_drl_generator()
    tranche_mgr = get_tranche_manager()
    sol_mgr = get_sol_exposure_manager()
    
    print(f"Regime: {regime_nav.get_state().regime.name}")
    print(f"Opportunity: {regime_nav.get_state().opportunity.mode.name}")
    
    # --- T=0: Pre-breakout ---
    print("\n--- T=0: Pre-Breakout (SOL at $195) ---")
    
    market_data = {
        'current_price': 195,
        'btc_price': 95000, 'btc_price_prev': 94800,
        'sol_price': 195, 'sol_price_prev': 193,
        'returns_4h': 0.01,
        'volatility_4h': 0.03,
        'volume_ratio': 1.2,
        'vpin': 0.45,
        'structure_break_pct': 0,
        'data_age_seconds': 0.5,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Generate signals (pre-breakout)
    quant_output = quant_gen.generate(market_data, "NEUTRAL", is_opportunity_mode=False)
    state_vector = np.array([0.01, 0.03, 0.45, 0.5, 0.6, 0.0])
    drl_output = drl_gen.generate(state_vector, "NEUTRAL", is_opportunity_mode=False)
    
    print(f"Quant: {quant_output.direction:.3f} ({quant_output.strategy_name})")
    print(f"DRL: {drl_output.direction:.3f}")
    print(f"Mode: NORMAL")
    
    # --- T=1: BREAKOUT! Structure break detected ---
    print("\n--- T=1: BREAKOUT! SOL breaks $200 resistance ---")
    
    market_data_breakout = {
        'current_price': 208,
        'btc_price': 95200, 'btc_price_prev': 95000,
        'sol_price': 208, 'sol_price_prev': 195,
        'returns_4h': 0.067,  # 6.7% move
        'volatility_4h': 0.05,
        'volume_ratio': 3.2,  # 3× average
        'vpin': 0.55,
        'structure_break_pct': 0.04,  # 4% break above level
        'structure_level': 200,
        'data_age_seconds': 0.3,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Trigger CRACK window
    regime_nav._check_crack_trigger(market_data_breakout, None)
    
    # Trigger sentiment OPPORTUNITY
    sentiment_state = {'sentiment_shock': 2.5, 'shock_magnitude': 2.5}
    regime_nav._check_opportunity_trigger(sentiment_state, None)
    
    state = regime_nav.get_state()
    print(f"CRACK window: {state.crack_window.is_active} (dir={state.crack_window.direction})")
    print(f"Opportunity: {state.opportunity.mode.name}")
    
    # Mode changes to OPPORTUNITY
    current_mode = SystemMode.OPPORTUNITY
    
    # Signals in OPPORTUNITY mode
    quant_output_opp = quant_gen.generate(market_data_breakout, "STRONG_BULL", is_opportunity_mode=True)
    state_vector_opp = np.array([0.067, 0.05, 0.55, 0.8, 0.8, 1.0])
    drl_output_opp = drl_gen.generate(state_vector_opp, "STRONG_BULL", is_opportunity_mode=True)
    
    print(f"\nSignals in OPPORTUNITY mode:")
    print(f"  Quant: {quant_output_opp.direction:.3f} ({quant_output_opp.strategy_name}) ← AGGRESSIVE")
    print(f"  DRL: {drl_output_opp.direction:.3f}")
    
    # Check no conflict
    conflict = compute_signal_conflict(
        quant_output_opp.direction, quant_output_opp.confidence,
        drl_output_opp.direction, drl_output_opp.confidence,
        0.6, 0.7  # Sentiment bullish
    )
    print(f"  Conflict: {conflict.all_signals_conflict}")
    
    # --- T=2: Tranche Entry (pre-4H close) ---
    print("\n--- T=2: Tranche Entry (before 4H bar close) ---")
    
    target_exposure_sol = 0.9  # SOL dominance target
    
    tranche_decision = tranche_mgr.decide(
        asset="SOL",
        mode=current_mode,
        target_exposure=target_exposure_sol,
        current_price=208,
        market_data=market_data_breakout,
        signal_data={'regime_aligned': True},
        is_4h_bar_close=False  # Not at close yet
    )
    
    print(f"Tranche decision: {tranche_decision.action.name}")
    print(f"New exposure: {tranche_decision.new_exposure:.2f} (capped at Tranche-1)")
    print(f"Reason: {tranche_decision.reason}")
    
    # Create TradeIntent
    intent_t1 = IntentFactory.create_directional_intent(
        asset="SOL",
        target_exposure=tranche_decision.new_exposure,
        authority=IntentAuthority.OPPORTUNITY_MODE,
        confidence=0.85,
        reasoning="CRACK window breakout, Tranche-1 entry",
        regime_name="STRONG_BULL_OPPORTUNITY",
        urgency=IntentUrgency.IMMEDIATE
    )
    
    print(f"\nTradeIntent (Tranche-1):")
    print(f"  Type: {intent_t1.intent_type.name}")
    print(f"  Target: {intent_t1.target_exposure:.2f}")
    print(f"  Authority: {intent_t1.authority.name}")
    print(f"  Urgency: {intent_t1.urgency.name}")
    
    # --- T=3: 4H Bar Close - Escalate to Tranche-2 ---
    print("\n--- T=3: 4H Bar Close - Escalate ---")
    
    # Update position to reflect T1 execution
    tranche_mgr.update_executed("SOL", tranche_decision.new_exposure, 208)
    
    # Now at 4H close
    tranche_decision_t2 = tranche_mgr.decide(
        asset="SOL",
        mode=current_mode,
        target_exposure=target_exposure_sol,
        current_price=215,
        market_data={'vpin': 0.5, 'volume_ratio': 2.5},
        signal_data={'regime_aligned': True},
        is_4h_bar_close=True
    )
    
    print(f"Tranche decision (4H close): {tranche_decision_t2.action.name}")
    print(f"New exposure: {tranche_decision_t2.new_exposure:.2f}")
    
    # Create TradeIntent for Tranche-2
    intent_t2 = IntentFactory.create_directional_intent(
        asset="SOL",
        target_exposure=tranche_decision_t2.new_exposure,
        authority=IntentAuthority.OPPORTUNITY_MODE,
        confidence=0.85,
        reasoning="4H close confirmed, escalating to Tranche-2",
        regime_name="STRONG_BULL_OPPORTUNITY",
        urgency=IntentUrgency.NORMAL
    )
    
    print(f"\nTradeIntent (Tranche-2):")
    print(f"  Type: {intent_t2.intent_type.name}")
    print(f"  Target: {intent_t2.target_exposure:.2f}")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SCENARIO 1 SUMMARY")
    print("=" * 70)
    print(f"Mode changes: NORMAL -> OPPORTUNITY (CRACK + Sentiment)")
    print(f"Strategy: bull_base -> bull_aggressive (OPPORTUNITY promotes)")
    print(f"Tranche flow: NONE -> T1 (20%) -> T2 (50%)")
    print(f"SOL dominance: Allowed up to 0.9 exposure")
    print(f"Final position: {tranche_decision_t2.new_exposure:.2f} of target")
    
    # Cleanup
    tranche_mgr.reset_position("SOL")
    
    return True


def scenario_2_sol_bearish():
    """
    SCENARIO 2: Fast SOL Bearish Move
    
    Setup:
    - SOL breaks below key support ($180)
    - Volume surge (2.5× average)
    - CVD strongly negative (retail panic)
    - Sentiment shock -3σ
    
    Expected Flow:
    1. CRACK window triggers (structure break DOWN)
    2. Mode -> OPPORTUNITY (short direction)
    3. Strategy -> bear_aggressive (promoted)
    4. Tranche-1 SHORT entry (20% of target)
    5. After 4H close -> escalate to Tranche-2 (50%)
    6. SOL dominance allowed (up to -0.9 exposure = short)
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: FAST SOL BEARISH MOVE")
    print("=" * 70)
    
    from core.market_regime_v32 import EnhancedRegimeNavigator, MarketRegimeState
    from core.signal_generators_v32 import (
        get_quant_generator, get_drl_generator, compute_signal_conflict
    )
    from core.tranche_manager import get_tranche_manager, TrancheLevel
    from core.sol_exposure_manager import get_sol_exposure_manager, ExposureMode
    from core.system_mode import SystemMode
    from core.trade_intent import IntentFactory, IntentAuthority, IntentUrgency
    import numpy as np
    
    print("\n--- Initial State ---")
    
    regime_nav = EnhancedRegimeNavigator()
    quant_gen = get_quant_generator()
    drl_gen = get_drl_generator()
    tranche_mgr = get_tranche_manager()
    
    print(f"Regime: {regime_nav.get_state().regime.name}")
    
    # --- T=0: Pre-breakdown ---
    print("\n--- T=0: Pre-Breakdown (SOL at $185) ---")
    
    market_data = {
        'current_price': 185,
        'btc_price': 94000, 'btc_price_prev': 94500,
        'sol_price': 185, 'sol_price_prev': 187,
        'returns_4h': -0.01,
        'volatility_4h': 0.04,
        'volume_ratio': 1.5,
        'vpin': 0.55,
        'structure_break_pct': 0,
        'data_age_seconds': 0.5,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    quant_output = quant_gen.generate(market_data, "BEAR", is_opportunity_mode=False)
    state_vector = np.array([-0.01, 0.04, 0.55, 0.5, 0.6, 0.0])
    drl_output = drl_gen.generate(state_vector, "BEAR", is_opportunity_mode=False)
    
    print(f"Quant: {quant_output.direction:.3f} ({quant_output.strategy_name})")
    print(f"DRL: {drl_output.direction:.3f}")
    print(f"Mode: NORMAL")
    
    # --- T=1: BREAKDOWN! Structure break detected ---
    print("\n--- T=1: BREAKDOWN! SOL breaks $180 support ---")
    
    market_data_breakdown = {
        'current_price': 168,
        'btc_price': 93000, 'btc_price_prev': 94000,
        'sol_price': 168, 'sol_price_prev': 185,
        'returns_4h': -0.092,  # -9.2% move
        'volatility_4h': 0.08,
        'volume_ratio': 2.8,  # 2.8× average
        'vpin': 0.72,  # Elevated toxicity
        'structure_break_pct': -0.067,  # 6.7% break below level
        'structure_level': 180,
        'data_age_seconds': 0.3,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Trigger CRACK window (short direction)
    regime_nav._check_crack_trigger(market_data_breakdown, None)
    
    # Trigger sentiment OPPORTUNITY (bearish shock)
    sentiment_state = {'sentiment_shock': -3.0, 'shock_magnitude': -3.0}
    regime_nav._check_opportunity_trigger(sentiment_state, None)
    
    state = regime_nav.get_state()
    print(f"CRACK window: {state.crack_window.is_active} (dir={state.crack_window.direction})")
    print(f"Opportunity: {state.opportunity.mode.name} (dir={state.opportunity.directional_bias})")
    
    # Mode changes to OPPORTUNITY
    current_mode = SystemMode.OPPORTUNITY
    
    # Signals in OPPORTUNITY mode (bearish)
    quant_output_opp = quant_gen.generate(market_data_breakdown, "STRONG_BEAR", is_opportunity_mode=True)
    state_vector_opp = np.array([-0.092, 0.08, 0.72, 0.8, 0.75, 1.0])
    drl_output_opp = drl_gen.generate(state_vector_opp, "STRONG_BEAR", is_opportunity_mode=True)
    
    print(f"\nSignals in OPPORTUNITY mode:")
    print(f"  Quant: {quant_output_opp.direction:.3f} ({quant_output_opp.strategy_name}) ← AGGRESSIVE")
    print(f"  DRL: {drl_output_opp.direction:.3f}")
    
    # Check no conflict (both bearish)
    conflict = compute_signal_conflict(
        quant_output_opp.direction, quant_output_opp.confidence,
        drl_output_opp.direction, drl_output_opp.confidence,
        -0.7, 0.8  # Sentiment bearish
    )
    print(f"  Conflict: {conflict.all_signals_conflict}")
    
    # --- T=2: Tranche Entry SHORT (pre-4H close) ---
    print("\n--- T=2: Tranche Entry SHORT (before 4H bar close) ---")
    
    target_exposure_sol = -0.9  # SOL dominance SHORT target
    
    tranche_decision = tranche_mgr.decide(
        asset="SOL",
        mode=current_mode,
        target_exposure=target_exposure_sol,
        current_price=168,
        market_data=market_data_breakdown,
        signal_data={'regime_aligned': True},
        is_4h_bar_close=False  # Not at close yet
    )
    
    print(f"Tranche decision: {tranche_decision.action.name}")
    print(f"New exposure: {tranche_decision.new_exposure:.2f} (capped at Tranche-1 SHORT)")
    print(f"Reason: {tranche_decision.reason}")
    
    # Create TradeIntent for SHORT
    intent_t1 = IntentFactory.create_directional_intent(
        asset="SOL",
        target_exposure=tranche_decision.new_exposure,
        authority=IntentAuthority.OPPORTUNITY_MODE,
        confidence=0.82,
        reasoning="CRACK window breakdown, Tranche-1 SHORT entry",
        regime_name="STRONG_BEAR_OPPORTUNITY",
        urgency=IntentUrgency.IMMEDIATE
    )
    
    print(f"\nTradeIntent (Tranche-1 SHORT):")
    print(f"  Type: {intent_t1.intent_type.name}")
    print(f"  Target: {intent_t1.target_exposure:.2f} (SHORT)")
    print(f"  Authority: {intent_t1.authority.name}")
    print(f"  Urgency: {intent_t1.urgency.name}")
    
    # --- T=3: 4H Bar Close - Escalate SHORT ---
    print("\n--- T=3: 4H Bar Close - Escalate SHORT ---")
    
    # Update position to reflect T1 execution
    tranche_mgr.update_executed("SOL", tranche_decision.new_exposure, 168)
    
    # Now at 4H close, price drops more
    tranche_decision_t2 = tranche_mgr.decide(
        asset="SOL",
        mode=current_mode,
        target_exposure=target_exposure_sol,
        current_price=158,  # Further drop
        market_data={'vpin': 0.65, 'volume_ratio': 2.2},
        signal_data={'regime_aligned': True},
        is_4h_bar_close=True
    )
    
    print(f"Tranche decision (4H close): {tranche_decision_t2.action.name}")
    print(f"New exposure: {tranche_decision_t2.new_exposure:.2f}")
    
    # Create TradeIntent for Tranche-2 SHORT
    intent_t2 = IntentFactory.create_directional_intent(
        asset="SOL",
        target_exposure=tranche_decision_t2.new_exposure,
        authority=IntentAuthority.OPPORTUNITY_MODE,
        confidence=0.82,
        reasoning="4H close confirmed, escalating SHORT to Tranche-2",
        regime_name="STRONG_BEAR_OPPORTUNITY",
        urgency=IntentUrgency.NORMAL
    )
    
    print(f"\nTradeIntent (Tranche-2 SHORT):")
    print(f"  Type: {intent_t2.intent_type.name}")
    print(f"  Target: {intent_t2.target_exposure:.2f} (SHORT)")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SCENARIO 2 SUMMARY")
    print("=" * 70)
    print(f"Mode changes: NORMAL -> OPPORTUNITY (CRACK + Sentiment -3σ)")
    print(f"Strategy: bear_base -> bear_aggressive (OPPORTUNITY promotes)")
    print(f"Tranche flow: NONE -> T1 SHORT (20%) -> T2 SHORT (50%)")
    print(f"SOL dominance: Allowed up to -0.9 exposure (SHORT)")
    print(f"Final position: {tranche_decision_t2.new_exposure:.2f} of target")
    
    # Cleanup
    tranche_mgr.reset_position("SOL")
    
    return True


def run_scenarios():
    """Run both scenario walkthroughs."""
    print("=" * 70)
    print("HMATS v3.2 - SCENARIO WALKTHROUGHS")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    
    scenario_1_sol_bullish()
    scenario_2_sol_bearish()
    
    print("\n" + "=" * 70)
    print("SCENARIO WALKTHROUGHS COMPLETE")
    print("=" * 70)
    print("\nKey Behaviors Demonstrated:")
    print("  ✓ CRACK window triggers OPPORTUNITY mode")
    print("  ✓ Sentiment shock triggers OPPORTUNITY mode")
    print("  ✓ OPPORTUNITY promotes AGGRESSIVE strategies")
    print("  ✓ Tranche-1 gating enforced (20% pre-4H close)")
    print("  ✓ Tranche escalation at 4H close")
    print("  ✓ SOL dominance (0.9 exposure) allowed")
    print("  ✓ SHORT positions supported")
    print("  ✓ TradeIntent boundary respected")


if __name__ == "__main__":
    run_scenarios()

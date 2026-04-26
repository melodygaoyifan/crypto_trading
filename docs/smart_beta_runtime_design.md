# Smart Beta V1 — Runtime Design

## Architecture
```
market_data (existing pipeline) ──→ SmartBetaController.compute()
                                         ├── TrendMomentumContext
                                         ├── VolatilityRegimeContext
                                         └── LiquidityFundingContext
                                              ↓
                                    SmartBetaState (bounded output)
                                              ↓
                                    apply_to_agent_signals()
                                              ↓
                         Multiply into _regime_alpha_gate_mult / _regime_position_size_mult
                         (SAME pattern as alpha_boost, BEFORE engine.decide())
```

## 3 Context Semantics

### TrendMomentumContext
- Inputs: GMM regime_state, Hurst, MTF momentum, ADX, RegimePhase
- Outputs: trend_dir, trend_strength, trend_quality, neutral_drift_score
- Effect: NEUTRAL_DRIFT → gate ×1.10, size ×0.85, block scale-in. TREND_STRONG → gate ×0.90, size ×1.10

### VolatilityRegimeContext
- Inputs: vol_regime, vol_z_score, VPIN, kurtosis, BSS
- Outputs: vol_regime_score, vol_expansion_score, toxicity_score
- Effect: VOL_COMPRESSED → gate ×1.05, size ×0.90. HIGH_TOXICITY → size ×0.80, conf ×0.85. BLACK_SWAN → size ×0.70

### LiquidityFundingContext
- Inputs: funding_rate, OI change, liquidation_imbalance, F&G, SSC crowding
- Outputs: crowding_score, funding_heat, liquidity_risk, liquidation_risk
- Effect: FUNDING_EXTREME → size ×0.85, short ×0.70. HIGH_CROWDING → size ×0.90, block scale-in

## Bounded Output Contract (SmartBetaState)
All multipliers clamped: gate [0.85, 1.20], size [0.70, 1.15], conf [0.80, 1.10], short [0.20, 1.00]

## What Smart Beta Does NOT Do
- Does NOT change trade direction
- Does NOT add DECIDE or VETO authority
- Does NOT modify DRL obs_dim or training contracts
- Does NOT bypass existing risk/safety layers
- Does NOT duplicate existing multiplier paths

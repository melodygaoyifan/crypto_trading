# Smart Beta V1 — Wiring Map

## Hot Path Injection Point
```
_process_4h_tick_inner()
  → market_data_pipeline.prepare_market_data() [existing]
  → Regime aggression params set [existing, L7768-7770]
  → ★ SmartBetaController.compute() + apply() ★  [NEW, ~L7795]
  → AlphaBoost.compute() [existing, multiplies on top]
  → engine.decide() (integration_v36) [existing]
  → Exit triggers [existing]
  → Execution [existing]
```

## Injection Mechanism
Multiplicative into EXISTING agent_signals fields:
- `_regime_alpha_gate_mult` (same as alpha_boost pattern)
- `_regime_position_size_mult` (same as alpha_boost pattern)
- `_smart_beta_block_scale_in` (new flag, read by downstream)

## No-Op Proof
- `enabled=false` → `compute()` returns neutral SmartBetaState (all mults=1.0)
- `apply_to_agent_signals()` returns False, no dict mutation
- Verified by `test_disabled_returns_neutral` and `test_disabled_apply_is_noop`

## Bounded Proof
- All outputs clamped: gate [0.85, 1.20], size [0.70, 1.15], conf [0.80, 1.10]
- Verified by `test_all_clamps_respected`

## What Smart Beta Affects
| Path | Mechanism | Bounded |
|------|-----------|---------|
| Alpha gate threshold | `_regime_alpha_gate_mult` × smart_beta_gate_mult | [0.85, 1.20] |
| Position size | `_regime_position_size_mult` × smart_beta_size_mult | [0.70, 1.15] |
| Scale-in permission | `_smart_beta_block_scale_in` flag | bool |

## What Smart Beta Does NOT Affect
- quant_direction (Best-of-N strategy selection)
- drl_direction (DRL action)
- sentiment_zscore
- veto_active
- any Authority level
- obs_dim or feature_manifest

# Smart Beta V1 → V1.1 Patch Plan

## Files Modified
| File | Change | Why Not Rewrite |
|------|--------|----------------|
| analytics/beta_exposure.py | Added compute_downside_beta(), downside_beta in BetaReport | Extension of existing class |
| core/smart_beta_controller.py | Added neutral_drift_score, [BETA_TREND/VOL/LIQ] proof logs | Extension of existing SmartBetaState |
| tests/test_smart_beta.py | Added 7 new tests (data guard, obs contract, drift, downside) | Extension of existing test file |
| main.py | SmartBeta init + injection (12 lines added) | Minimal insertion at alpha_boost injection point |

## Files Created
| File | Why New (not wrapper) | Can Be Deleted Safely? |
|------|----------------------|----------------------|
| core/smart_beta_controller.py | No existing Smart Beta orchestrator existed | Yes (enabled=false = no-op) |
| analytics/beta_exposure.py | No existing beta measurement module existed | Yes (standalone analytics) |
| scripts/run_beta_audit.py | No existing beta audit CLI existed | Yes (standalone script) |
| tests/test_smart_beta.py | No existing Smart Beta tests existed | Yes |
| tests/test_beta_exposure.py | No existing beta exposure tests existed | Yes |
| 5 docs in docs/ | Documentation | Yes |

## Rollback Plan
1. Set `smart_beta_config.enabled = false` in config → immediate no-op
2. If deeper rollback needed: revert the 12-line main.py insertion
3. All new files are additive — deleting them breaks nothing

## What Was NOT Changed
- obs_dim (126 dims)
- feature_manifest.json
- train_drl_full.py
- Any model weights
- Authority matrix
- Risk/veto chain
- DRL training contracts
- Sentiment L1/L2/L3
- Short Bias Agent
- Any existing test (only additions)

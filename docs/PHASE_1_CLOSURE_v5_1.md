# HMATS v5.1 — Phase 1 Closure (Strategy Library Archive)

**Status:** COMPLETE — ready for Phase 4 (microstructure expansion).
**Generated:** 2026-04-29

## What changed

| File | Change |
|---|---|
| `configs/strategy_v5_1_decisions.json` | NEW — 12-strategy decision matrix (4 ARCHIVE, 8 KEEP). Hot-revert: set `archived: false` + restart. |
| `agents/kraken_quant_agent.py` | +73 lines: `_load_archive_decisions()`, `_enforce_iron_law_6()`, archive gate in `process_tick()`, `_strategy_archived_skips` counter, JSON/Path/Set imports. |

JSON, not YAML — codebase has zero `import yaml` in source tree (only venv libraries). Path corrected from prompt's `config/` → `configs/`.

## Archive set

**ARCHIVED (4):** HurstExponentStrategy, ShannonEntropyStrategy, OrnsteinUhlenbeckStrategy, DeltaNeutralFundingStrategy.
**KEEP (8):** LiquidationCascadeHunter, VarianceRiskPremiumStrategy, FundingDivergenceStrategy, ETFSpotCointegration, RelativeStrengthStrategy, OrderBookImbalance, KalmanCointegration_SOL_ETH, DarkPoolVolumeStrategy.

Iron Law 6 (≥3 active) satisfied with margin (8 ≥ 3).

## Architecture

Archive gate lives in `StrategyAllocator.process_tick()` at the dispatch loop (per pre-confirmation issue #4). The 12 strategies are nested classes without a shared `self.config` dict; the prompt's `if self.config.get('archived', False)` shape was unworkable. The chosen route reads `configs/strategy_v5_1_decisions.json` once at allocator `__init__`, builds `Set[str]`, skips matching strategy names in the per-tick dispatch loop. Equivalent to returning Signal.NEUTRAL — Iron Law 4 fail-closed (skip = no vote, no fire).

```
process_tick(regime, market_data):
    for strategy in strategies[regime]:
        if strategy.name in self._archived_strategies:
            self._strategy_archived_skips[strategy.name] += 1
            continue                      # <- gate
        signal = strategy.update(...)     # only KEEP strategies reach here
```

A new counter `_strategy_archived_skips` records the skip telemetry separately from the existing `_strategy_attempts`/`_strategy_fires` counters so operator can see archive coverage in `data/kq_firing_stats.json`.

## Fail-safety verification (3/3 PASS)

1. **Missing JSON file** → empty archive set (all 12 active). Logged INFO. ✓
2. **Malformed JSON** → empty archive set + WARN log with exception type. ✓
3. **Iron Law 6 violation** (only 2 active configured) → CRITICAL log + reset to all-active. ✓

Pattern matches CLAUDE.md P15/P85 lessons: defensive `getattr`-style + WARN log on degradation, never silently violate a non-negotiable invariant.

## Test verification

| Suite | Result |
|---|---|
| `tests/test_kraken_quant_agent.py` | **33/33 PASS** |
| `tests/test_strategy_selection.py` | 15/15 PASS |
| `tests/test_alpha_gate.py` | 9/9 PASS |
| `tests/test_black_swan_hold.py` | 23/23 PASS |
| `tests/test_authority_fusion.py` | PASS (incl. 20 datetime warns, pre-existing) |
| `tests/test_authority_chain_freshness_context.py` | PASS |
| `tests/test_drl_promotion_gate.py` | PASS |
| `tests/test_drl_authority_punchthrough.py` | PASS |
| `tests/test_drl_agent.py` | PASS |
| `tests/test_constitution_core.py` | PASS |
| **Cross-cutting cumulative** | **225/225 PASS** |

Broader regression: 1931 pass / 78 fail / 6 skip across the full suite. The 78 failures are pre-existing tech debt (passive_aggressive, realized_pnl_exit, sentiment_llm, step15_status_export, stop_order_retry, ultra_aggressive_profile, ultra_tranche, etc.) — none of them import or touch `agents/kraken_quant_agent.py`, `analytics/ic/compute_ic.py`, or `configs/strategy_v5_1_decisions.json`. `tests/v36_scenarios/test_production_patches.py` fails at collection due to a stale import path (`integration.core.production_reliability_patches`), also pre-existing.

## Singleton smoke

`get_kraken_quant_agent()` (the entry point used by `main.py`) returns a `KrakenQuantAgentV6` whose `.allocator._archived_strategies` carries the 4 archived names through. Production singleton picks up the gate without any code changes in main.py.

## Iron Law check

| Law | Status |
|---|---|
| 1. obs_dim=126 | UNCHANGED — Phase 1 does not touch features |
| 2. constitution.py | UNCHANGED |
| 3. training/ | UNCHANGED |
| 4. fail-closed | HELD — archive gate skips → Signal.NEUTRAL semantics |
| 5. DRL ACTIVE floor | UNCHANGED — kraken_quant changes don't touch DRL authority |
| 6. ≥3 active strategies | HELD (8 active); enforced by `_enforce_iron_law_6()` |
| 7. shadow ≥30d before promotion | N/A — Phase 1 archives, doesn't promote new strategies |
| 8. DRL ACTIVE during cutover | N/A — Phase 2 not started |
| 9. Maker-first defensive default | UNCHANGED — `OrderConfig.post_only=True` invariant intact |

## Outstanding [PARAMETER]

- [PARAMETER 3] V4.3 cutover mode — still pending operator (Phase 2 not yet triggered, Day 14).
- [PARAMETER 4] V8 DRL retrain — default N, unchanged.

## Deploy step (operator action)

Phase 1 changes are ready to deploy when operator confirms. Recommended:
1. `git add agents/kraken_quant_agent.py configs/strategy_v5_1_decisions.json analytics/ic/compute_ic.py docs/PHASE_BASELINE_v5_1.md docs/PHASE_1_CLOSURE_v5_1.md`
2. Commit message:
   ```
   v5.1 Phase 0+1: IC re-baseline + strategy archive gate (4 archived)

   Phase 0: post-P19 IC baseline (insufficient samples → Branch Y), 12-strategy
   bucket categorization (strategy_id=9 = KalmanCointegration_SOL_ETH was the
   v5-doc miss), pre-v5 30d Sharpe 0.72 unreliable, fee/alpha 11.5%, Iron
   Laws 1-9 verified intact.

   Phase 1: kraken_quant_agent reads configs/strategy_v5_1_decisions.json at
   allocator init, archive gate at process_tick dispatch. ARCHIVED 4
   (HurstExponent, ShannonEntropy, OrnsteinUhlenbeck — Bucket A pure-technical;
   DeltaNeutralFunding — cash-carry deferred v6). KEEP 8 strategies satisfies
   Iron Law 6 (≥3) with margin. Fail-safety: missing/malformed JSON or Iron
   Law 6 violation → fail-open all-active with WARN/CRITICAL log.

   compute_ic.py: 3 sites of pd.to_datetime patched format=ISO8601 to handle
   mixed-format IC log timestamps (tooling unblock for Phase 0 IC re-run).

   225/225 cross-cutting tests pass (kraken_quant_agent + strategy_selection +
   alpha_gate + black_swan_hold + authority + DRL + constitution). Broader
   78-test failure set is pre-existing, none touch the Phase 1 surface.
   ```
3. Deploy via existing `bash scripts/hetzner_deploy.sh hmats` workflow.

## Phase 2 readiness

Phase 1 closure does not unblock Phase 2 — that remains gated on [PARAMETER 3] cutover-mode operator answer. Phase 4 (microstructure expansion, Days 5-7) is next on Branch Y per the v5.1 timeline and CAN start without operator answer.

# Reader Consumer Options — design candidates for `get_weight_modifiers()`

**Date**: 2026-04-28
**Mode**: READ-ONLY design scoping per v3.5 Phase 1.2.
**Decision deferred**: this doc lists candidates + tradeoffs; does NOT pick one.

---

## Context

`analytics/strategy_aging.py:get_weight_modifiers()` returns `Dict[str, float]` mapping strategy_name → weight modifier in [0.5, 1.2]. Reader trace (`READER_TRACE_get_weight_modifiers_2026-04-28.md`) showed the dict goes nowhere; root cause (`READER_DEAD_ROOT_CAUSE_2026-04-28.md`) is Type 3 / unfinished design from initial commit.

Three candidate consumer designs evaluated below.

---

## Candidate A — Best-of-N selection scorer

**Insertion site**: `data_mgmt/market_data_pipeline.py:1239`

```python
# Current
best_name = max(strategies, key=lambda k: strategies[k]["strength"])

# With Candidate A
best_name = max(strategies, key=lambda k:
    strategies[k]["strength"] * weight_modifiers.get(k, 1.0))
```

**Mechanism**: aged strategy's strength is scaled down by its modifier BEFORE best-of-N picks the winner.

**Effort**: ~0.5 person-day code change. Plus readiness of `weight_modifiers` requires plumbing the dict from `_strategy_aging` → `market_data_pipeline` (currently the pipeline doesn't have a reference).

**Risk surface**:
- **HIGH** — best-of-N output flows directly into `quant_direction` (DECIDE-authority) signal. A noisy modifier produces noisy strategy switching.
- Strategy decay/recovery cycles could cause oscillation: strategy A loses → modifier 0.6 → not selected → no new outcomes → modifier never recovers → permanent demotion.
- Risks the only HEALTHY strategy (mean_revert per v1 §1.2) being demoted on a bad streak.

**Expected impact**:
- 30-day: low (with min_signals_for_assessment=20 and 17 trades/30d total, modifiers stay at 1.0 default for foreseeable future — wiring delivers zero behavior change).
- Post-aging-warmup (would take 60-90 days at current trade rate to populate windows): moderate. Aged strategies stop winning best-of-N → reduced loss attribution to known-bad strategies.

**Failure modes**:
1. Modifier oscillation → strategy churn → fee drag.
2. Mean_revert (the one healthy strategy per v1) over-demoted on small sample.
3. `min_signals_for_assessment=20` never reached → modifier stays 1.0 forever → wiring is silently dead loop again (different shape).

---

## Candidate B — Authority-fusion weight multiplier

**Insertion site**: `signals/authority_fusion.py` per-agent weighting (likely the DECIDE / CONFIRM layer aggregation in `_build_fusion_signals`).

**Mechanism**: each agent's contribution to fusion is multiplied by its modifier (or by an aggregate of its strategies' modifiers).

**Effort**: 1-2 person-days. Authority fusion is intricate (12 agent paths, regime-dependent logic, multiple gates); modifier injection needs careful placement to avoid double-weighting.

**Risk surface**:
- **CONFLICTS WITH 1.4 FINDING**. The correlation report (`agent_correlation_matrix_2026-04-28.json`) shows fusion has already collapsed to 2-3 effective independent sources across BTC/ETH/SOL. Adding a per-agent weight modifier inside a 2-3-rank system has mathematically near-zero effect — multiplying noise by a scalar still gives noise.
- Authority matrix is load-bearing for VETO / DECIDE semantics (CLAUDE.md non-negotiable rule #1). Touching it has high blast radius.

**Expected impact**:
- Effectively zero given current factor collapse. Wiring would technically work but the modifier value would not differentiate signal from noise in the post-fusion output.

**Verdict**: weakest of the three. Don't pick unless 1.4 finding turns out to be artefact (per Phase 2.3 sensitivity check).

---

## Candidate C — Sizing modifier (post-decision)

**Insertion site**: After best-of-N decides (`data_mgmt/market_data_pipeline.py:1239` consumer) AND alpha gate passes; before final position size is multiplied by `target_exposure`. Likely in `core/execution_service.py` or in the pre-execution sizing flow (`risk/unified_position_sizer.py`).

**Mechanism**: trade direction is unchanged; trade SIZE is scaled by the winner's modifier.

```python
final_size = target_exposure * weight_modifiers.get(strategy_name, 1.0)
```

**Effort**: ~0.5 person-day code change + plumbing.

**Risk surface**:
- **LOW** — fail-soft semantics. If modifier=0.5, position is smaller but still in the right direction. If modifier=1.0 (default for under-sample case), behavior is unchanged.
- Modifier floor at 0.5 means worst case is half-size; multiplier floor 0.15 (canonical config) still applies on top so no zero-size execution risk.
- Doesn't change strategy SELECTION → no oscillation/churn risk. The strategy that WON best-of-N still trades; it just trades smaller when aging says it's been weak.

**Expected impact**:
- 30-day: minimal (same `min_signals_for_assessment=20` gate; modifier stays 1.0).
- Post-warmup: reduced loss tail. Aged strategies still execute but with smaller bites, so single bad trade caps at 0.5x exposure.

**Failure modes**:
1. Sizing reduction is bounded by other floors (multiplier 0.15, kelly cap, position limits) — modifier=0.5 may have no effect if other floors already binding.
2. Reduces upside symmetrically: a strategy in recovery (modifier=0.7) places 70% bets even when its first post-aging trade is correct.

---

## Comparison table

| Dimension | A: Best-of-N | B: Fusion weight | C: Sizing modifier |
|---|---|---|---|
| Effort | 0.5d | 1-2d | 0.5d |
| Risk | HIGH (selection churn) | HIGH (touches authority matrix; near-zero impact post-1.4) | LOW (fail-soft) |
| Expected 30d impact | ~0 (sample gate) | ~0 (factor collapse) | ~0 (sample gate) |
| Expected 90d impact | Moderate | Near-zero | Modest |
| Composes with 1.4 finding | Neutral | Bad | Neutral |
| Iron Law compliance | PASS | PASS but touches authority surface | PASS (cleanest) |
| Reversibility | Easy (revert one line) | Hard (fusion logic intricate) | Easy |
| 14-day kill criterion | Strategy churn rate > 2x baseline | (B not recommended) | Avg position size 30% smaller than counterfactual + no PnL improvement |

---

## Recommendation framing (NOT a decision — operator picks)

If the goal is **"close the dead loop minimally"** → **C** (lowest risk, easiest revert).

If the goal is **"actually let aging affect strategy selection"** → **A** (more impact, more risk; needs the 60-90 day warmup before showing effect).

If the goal is **"defer until orthogonal-factor work resolves the 1.4 finding"** → **stay DEAD** (path P4 in v3.5 §3.1) and file as v4 P0 contingent on factor work.

**Path B is dominated** — it's higher effort + higher risk + lower expected impact than A or C given the 1.4 fusion-collapse finding.

---

## What needs operator input

1. Which path (A / C / DEAD-pending-v4)?
2. If A or C: should `min_signals_for_assessment=20` be lowered alongside (e.g. to 5) so wiring delivers signal in current low-trade regime, OR kept at 20 (= wiring is technically live but no behavior change for ~60d)?
3. If staying DEAD: should we add a startup CRITICAL log declaring "strategy_aging is INSTRUMENTATION ONLY, does not affect decisions" so the operator's mental model matches reality?

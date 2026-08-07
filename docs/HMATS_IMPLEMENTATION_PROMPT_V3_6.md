# HMATS Implementation Prompt v3.6 — DRAFT (post v3.5 reorder)

**Status**: DRAFT for operator review. Do NOT execute as-is.
**Triggered by**: v3.5 outputs (READER_DEAD_ROOT_CAUSE / READER_CONSUMER_OPTIONS / ORTHOGONAL_FACTOR_CANDIDATES).
**Date**: 2026-04-28

---

## What v3.5 changed in the v3 plan

Three blocking findings:

1. **`get_weight_modifiers()` is Type-3 dead** (LOGGED_ONLY since initial commit, no historical consumer to revert from). Reader fix is a DESIGN choice not a bug fix.
2. **12-agent fusion collapses to 2-3 effective independent sources** (BTC=3, ETH=2, SOL=2). Sensitivity check at PCA thresholds 0.5/1.0/1.5/2.0 — finding holds; even at 0.5 no asset reaches 6 sources.
3. **IQL doesn't fix factor collapse**. Same `obs_dim=126` input → same factor ceiling regardless of RL architecture.

These reorder both Track A (which item to tackle next) and Track B (what migration sequence makes sense).

---

## Operator decisions required BEFORE this prompt becomes executable

### Decision D1 — Reader fix path

| Path | Description | Effort | Risk |
|---|---|---|---|
| **P1** | Wire Candidate A (best-of-N selection scorer) | 0.5d | HIGH (selection churn) |
| **P2** | Wire Candidate C (sizing modifier post-decision) | 0.5d | LOW (fail-soft) |
| **P3** | Archive `analytics/strategy_aging.py` entirely | 0.25d | LOW (delete dead code) |
| **P4** | Stay DEAD pending v4; add startup CRITICAL log "INSTRUMENTATION ONLY" | 0.1d | NONE |

**Recommendation framing**: P2 (sizing modifier) or P4 (declare dead, defer wiring). P1 is the most ambitious but has the highest blast radius given the 1.4 finding. P3 is reasonable if operator agrees the producer side was always speculative.

### Decision D2 — `min_signals_for_assessment` threshold

If D1 = P1 or P2:
- Keep at 20 → wiring is technically live but no behavior change for ~60d (need 17→60 trades to populate).
- Lower to 5 → wiring delivers signal in ~10d but on noisier estimates.
- Lower to 10 → middle ground.

### Decision D3 — Track B sequencing

| Item | v3 priority | Recommended v3.6 priority |
|---|---|---|
| 2.1 Coinbase eligibility check (manual) | high | **HIGH (do today)** |
| 2.2 Coinbase migration scope | high | **HIGHEST** — bundles funding-rate orthogonal factor |
| 2.3 IQL replacement scope | medium | **LOW (defer to v5)** |
| 2.4 (NEW) On-chain feed re-enable | — | **MEDIUM** — cheap orthogonal factor |
| 2.5 (NEW) Orthogonal factor research broader | — | filed for v4 audit cycle |

---

## Provisional v3.6 execution order (assumes D1=P2, D2=keep 20, D3=as above)

```
Day 1 AM:
  - 1.5 P0-4 RSI sign-flip diagnostic (paused from v3, ~1d)

Day 1 PM:
  - 1.0' Reader fix Candidate C (sizing modifier) (~0.5d)
    - Add weight_modifier multiplier in pre-execution sizing
    - Defensive default 1.0 when missing
    - Test + smoke + deploy

Day 2 AM:
  - 1.6 P1-5 IC cron wiring (~0.5d)
  - 2.1 Coinbase eligibility check (operator manual, ~10min)

Day 2 PM:
  - 2.4 (NEW) On-chain feed re-enable scoping (~0.5d, READ-ONLY)
    - Verify enabled=False is intentional vs accidental
    - List config + integration changes needed for re-enable
    - File as v4 GO if scoping clean

Day 3:
  - 2.2 Coinbase migration scope (~1d)
    - Include section: "funding rate as orthogonal factor"
    - Include leverage analysis (existing 5x cap stays; Coinbase 10x available)

Day 4-7:
  - Passive: 1.7 P0-1 kraken_quant 7-day capture, re-eval 2026-05-05
  - Operator: review Coinbase scope, decide migration go/no-go
```

**Total committed engineer time**: ~3 person-days for items above the line.

---

## Items DROPPED from v3 / deferred

- **Reader Candidate A (best-of-N)** — deferred to v4 contingent on operator pick. Higher blast radius than v3.6 should ship.
- **Reader Candidate B (authority fusion weight)** — DROPPED. Per ORTHOGONAL_FACTOR_CANDIDATES.md analysis: dominated by A and C, conflicts with 1.4 finding.
- **IQL replacement scope (v3 2.3)** — deferred to v5. Solving factor collapse first.
- **Per-kraken_quant-strategy correlation** — deferred to v4 with longitudinal `kq_firing_stats.jsonl` data (P128 capture window must accumulate).

---

## v4 trigger conditions (informational, not in v3.6 scope)

v4 audit launches when ANY of:
- 1.5 RSI diagnostic complete + 14d post-fix data captured
- Reader fix (D1 path) shipped + 30d behavior change observable
- 2026-05-05 P0-1 kraken_quant 7-day data review
- Coinbase migration decision made
- Operator-triggered ad-hoc

v4 P0 candidates already filed:
- Per-kraken_quant-strategy correlation (deeper than v3.5 1.4)
- `signals/adaptive_weight_v521.py` parallel-track aging audit (may be DEAD too)
- ETH cvd↔lead_lag rho=+0.75 — data leakage investigation
- Strategy lifecycle redesign (if D1=P3 picked)

---

## Iron Laws (sustained)

All v3 Iron Laws carry forward verbatim. Specifically:
- READ-ONLY for any scoping work.
- `obs_dim=126` / `defense/constitution.py` / `training/` untouchable.
- Every implementation item is fail-closed.
- No destructive migrations; old code preserved next to new.

---

## Failure modes for v3.6 (to be enforced when this prompt becomes executable)

1. If D1 picked but operator skips D2 (threshold) → STOP, demand D2 before code.
2. If 1.5 RSI diagnostic discovers Hypothesis B (regime-conditional, not 1-line bug) → ship NOTHING, file v4 task.
3. If 2.1 Coinbase eligibility = INELIGIBLE → drop 2.2 from v3.6, run 2.4 standalone.
4. If reader fix Candidate C shipped but 14-day data shows position sizing is unbounded by other floors (i.e. modifier never has effect) → file as expected; declare wiring is observability-only and re-evaluate at v4.

---

## Open questions for operator review

1. **D1**: P2 (sizing) or P4 (declare dead)? Other paths have higher risk than the marginal value.
2. **D2**: lower `min_signals_for_assessment` from 20 → 10? Affects how soon reader-wiring shows behavior delta.
3. **D3**: agreed on Coinbase HIGHEST + IQL defer + on-chain re-enable MEDIUM? Or different?
4. **Scope creep check**: are we OK adding "on-chain re-enable" (item 2.4 NEW) to v3.6 or should it stand alone in a separate prompt?
5. **1.5 RSI diagnostic** — proceed in v3.6 even though v3.5 paused before it ran? It's the next item in the original v3 queue and didn't depend on the paused decisions.

---

## Output checklist for v3.6 execution (when ready)

If v3.6 ships per provisional order above:
- [ ] `agents/kraken_quant_agent.py` or wherever RSI signal lives — Hypothesis-A 1-line fix (only if 1.5 confirms A)
- [ ] `core/execution_service.py` or `risk/unified_position_sizer.py` — modifier multiplier insertion (if D1=P2)
- [ ] `tests/test_reader_fix_p132.py` — regression covering modifier behavior
- [ ] `analytics/ic/cron_compute_ic.sh` or equivalent — IC cron wrapper (P1-5)
- [ ] `docs/COINBASE_ELIGIBILITY_2026-04-28.md` (operator manual deliverable)
- [ ] `docs/COINBASE_MIGRATION_SCOPE_2026-04-28.md`
- [ ] `docs/ONCHAIN_REENABLE_SCOPE_2026-04-28.md` (if 2.4 NEW included)

NOT in v3.6:
- IQL prototype (v5+)
- Best-of-N reader fix Candidate A (deferred to v4)
- Authority fusion weight modifier Candidate B (DROPPED)
- Per-strategy correlation matrix (v4)

---

*Generated 2026-04-28 from v3.5 Phase 3 outputs.*
*Awaiting operator review of D1/D2/D3 before this prompt becomes executable as v3.6.*

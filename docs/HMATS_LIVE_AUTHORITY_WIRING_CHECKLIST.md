# HMATS Live Authority Wiring Checklist

> **⚠️ HISTORICAL — SUPERSEDED (banner added 2026-08-07).** March-2026 snapshot
> of a paper-trading system. Since then: the system went live, lost −25%
> (Apr–Jun), cut over to Coinbase perps (2026-06-13), and both DRLs were demoted
> to SHADOW (2026-08-07, P198/P200). The promotion framework below is replaced
> by the **P200-LADDER** (`docs/HMATS_TRAINING_GUIDE_V2.md` + CLAUDE.md). This
> doc also contradicts itself on DRL (§3 says promote-to-DECIDE, §5.5 says
> EXIT_ONLY first). Every `main.py:NNNN` line reference is from a March snapshot
> of a file that has since roughly doubled — treat all of them as unverified.

Date: 2026-03-09
Scope: runtime module promotion for live paper-trading authority

## 1. Core framing

The first distinction is non-negotiable:

- `direction alpha` answers: do we already have a live directional edge?
- `realized alpha` answers: did that edge survive execution and become fills / PnL?

Not every profitable module creates `direction alpha`.
Many modules improve:

- sizing
- execution
- cost control
- risk control
- feedback

Those modules improve net profit, but they are not primary directional alpha.

Current interpretation for HMATS:

- Base directional alpha already exists.
- Realized alpha path is not yet stable enough to use as a promotion baseline.
- Do not promote more amplifiers before the first stable `INTENT -> FILL -> POSITION` cycle is repeatable.

## 2. Verified current state

As of the 2026-03-09 paper session:

- `data/shadow_ledger/ledger_20260309.jsonl` records `INTENT` for SOL, BTC, ETH, but no `FILL` or `POSITION` entries.
- `data/dashboard_state.json` records `alpha_gate_passed=true` for all three assets.
- `logs/paper_run_stderr.log` shows first-tick `AC-0` post-restart entry blocking, then re-enable.
- `agents/model_alpha_agent.py` fail-closes to neutral when no model is found.

Operational conclusion:

- We already have live directional approval from the base quant core.
- We do not yet have a clean realized-alpha baseline for measuring additional module contribution.
- `dashboard_state.json` equity / cumulative PnL should not be treated as proof of newly realized alpha unless the same session also contains `FILL` records in the shadow ledger.

## 3. Promotion order

| Phase | Module | Authority | Promote now? | Reason |
|---|---|---|---|---|
| 0 | Fill path baseline | N/A | Yes | First prove `INTENT -> FILL -> POSITION` without extra amplification |
| 1 | LeadLag | `CONTEXT` | Yes, after baseline fill | Confidence modulation only, lowest directional risk |
| 2 | CompositeToxicityFilter | `EXECUTION` | Yes, after baseline fill | Execution-style control, not entry alpha |
| 3 | LearnedExecutionPolicy | `EXECUTION` | Yes, after baseline fill | Advisory execution only |
| 4 | PortfolioBrain / G6 | `SIZING` | Yes, after stable fills | Portfolio weights, not direction |
| 5 | DRL | `ACTIVE/DECIDE` | Yes, 30 shadow trades passed | v6.8: FIX-DRL-AUTHORITY wired DECIDE in fusion. Entry+exit authority via confidence-weighted consensus |
| 6 | aggressive_allocator | `SIZING` | Later | Pure amplifier; should not precede stable fills |
| 7 | ModelAlphaAgent | none yet | No | Missing model artifact means neutral output |

## 4. Global promotion gates

Every module promotion should satisfy all of these:

- A baseline paper session exists with actual `FILL` records.
- The module changes only its intended authority surface.
- Dashboard export exposes whether the module was merely called vs actually consumed.
- Proof log or shadow ledger can show whether the module influenced the final action.
- The module has a clear demotion path.

Recommended minimum evidence before each phase:

| Phase | Suggested evidence |
|---|---|
| Phase 0 complete | At least 1 repeatable paper `FILL` after restart |
| Execution modules | At least 10 fills with stable execution telemetry |
| Sizing modules | At least 10 fills across all assets plus exposure telemetry |
| Amplifiers | At least 20 fills and positive execution / sizing delta vs baseline |
| DRL `ACTIVE/DECIDE` | 30 shadow trades passed, OOD/drift guard live, FIX-DRL-AUTHORITY verified in fusion (v6.8) |

## 5. Module checklists

### 5.1 LeadLag -> `CONTEXT`

Current code touchpoints:

- Load: `main.py:313-317`
- Runtime signal injection: `main.py:6056-6060`, `main.py:6425-6448`
- Diagnostics: `main.py:6869-6880`
- Live consumption: `main.py:9222-9266`
- Dashboard export: `main.py:7170-7174`, `main.py:14908-14912`

Required behavior:

- Keep `lead_lag_authority = "CONTEXT"`.
- Only modulate confidence.
- Never write directly to `intent.direction`.
- Never cast a veto by itself.

Checklist:

- [ ] `lead_lag_edge` and `lead_lag_confidence` are present in `agent_signals`
- [ ] Only `quant_confidence` changes in `LEAD_LAG_AMPLIFIER`
- [ ] Dashboard shows `lead_lag_applied=true` only when multiplier actually changed
- [ ] Attribution can compare baseline fill quality before and after LeadLag promotion

Do not do:

- [ ] Do not move LeadLag into `DECIDE`
- [ ] Do not let LeadLag create standalone entries

### 5.2 CompositeToxicityFilter -> `EXECUTION`

Current code touchpoints:

- Load: `main.py:779-783`
- Init: `main.py:3932-3940`
- Diagnostics: `main.py:6902-6915`
- Pre-execution application: `main.py:11993-12030`
- Dashboard export: `main.py:12008-12021`, `main.py:14885-14890`
- Scoring logic: `execution/composite_toxicity.py:65-120`

Important limitation in current wiring:

- `adverse_selection` is still hardcoded to `0.0`
- `price_reversal` is still hardcoded to `0.0`

That means the current runtime version is only partially productized.

Required behavior:

- Stay in `EXECUTION` authority.
- Only change urgency / prefer-limit behavior.
- Do not veto entry direction just because toxicity is high.

Checklist:

- [ ] Wire real `adverse_selection` from fill history or FillSlopeMonitor
- [ ] Wire real `price_reversal` from post-fill tracking
- [ ] Keep current behavior as advisory until the two missing inputs are live
- [ ] Dashboard exposes `score`, `warn`, `dominant`, `applied`

Do not do:

- [ ] Do not route toxicity into `DECIDE`
- [ ] Do not let toxicity flip long/short direction

### 5.3 LearnedExecutionPolicy -> `EXECUTION`

Current code touchpoints:

- Load: `main.py:1214-1226`
- Init: `main.py:3500-3507`
- Diagnostics: `main.py:6917-6932`
- Pre-execution application: `main.py:12041-12097`
- Dashboard export: `main.py:12063-12081`, `main.py:14891-14897`
- Mode contract: `execution/learned_execution_policy.py:13-19`, `execution/learned_execution_policy.py:47`

Required behavior:

- Remain advisory-only.
- Only affect order type, urgency, limit offset.
- Never bypass market impact logic.

Checklist:

- [ ] `recommended_action` only changes execution style
- [ ] `learned_exec_applied` is exported when advice was consumed
- [ ] `active` mode remains unused
- [ ] Bear-market passiveness stays a style hint, not a direction vote

Do not do:

- [ ] Do not let LEP veto entry
- [ ] Do not let LEP create standalone alpha claims

### 5.4 PortfolioBrain / G6 -> `SIZING`

Current code touchpoints:

- Import: `main.py:957-963`
- Init: `main.py:2927-2945`
- Bar-PnL feedback: `main.py:15190-15203`
- Sizing application: `main.py:9544-9574`
- Dashboard export: `main.py:14898-14902`
- Promotion hook: `portfolio/portfolio_brain_offensive.py:250-258`

Important note:

- Current init path can run in live sizing mode unless `HMATS_G6_SHADOW=true`.
- That makes G6 more advanced than the other modules from a wiring perspective.

Required behavior:

- Only scale target exposure.
- Never decide direction.
- Keep a visible `shadow` vs `live` mode distinction.

Checklist:

- [ ] Confirm intended runtime mode: `SIZING_SHADOW` or `SIZING`
- [ ] `g6_multiplier` is capped and exported
- [ ] G6 reads bar PnL continuously before trusting live weights
- [ ] Promotion uses `promote_to_live()` instead of ad hoc flag drift

Do not do:

- [ ] Do not use G6 as alpha confirmation
- [ ] Do not let G6 bypass portfolio caps

### 5.5 DRL -> `EXIT_ONLY`

Current code touchpoints:

- Authority contract: `main.py:27-31`
- Promotion lifecycle: `main.py:4200-4233`
- Runtime summary: `main.py:15757-15762`
- Integrity check: `main.py:16477-16480`
- v36 gate behavior: `integration/integration_v36.py:1711-1719`
- Agent mode contract: `agents/drl_agent.py:79-99`, `agents/drl_agent.py:941-942`

Required behavior:

- Promote only to `EXIT_ONLY` first.
- DRL may reduce or hold exits, never become entry authority.
- Keep OOD / drift demotion path live.

Checklist:

- [ ] Promotion gate never bootstraps straight to entry authority
- [ ] `trade_impact` remains `EXIT_ONLY` before any higher promotion
- [ ] OOD / drift can demote back from `EXIT_ONLY`
- [ ] PnL attribution is available before arguing for higher authority

Do not do:

- [ ] Do not give DRL entry alpha authority
- [ ] Do not evaluate DRL only by unrealized dashboard PnL

### 5.6 aggressive_allocator -> `SIZING`

Current code touchpoints:

- Init: `main.py:2258-2273`
- Application: `main.py:10037-10091`
- Diagnostics: `main.py:10193-10208`
- Dashboard export: `main.py:14903-14907`

Required behavior:

- Treat as a sizing amplifier only.
- Use after baseline fill path and after G6 sizing is already stable.
- Keep the tighter cap semantics intact.

Checklist:

- [ ] Confirm allocation config is intentionally enabled
- [ ] Confirm dashboard shows `aggressive_allocator_applied`
- [ ] Confirm allocator only narrows cap or amplifies within explicit risk budget
- [ ] Compare fill quality before and after enabling

Do not do:

- [ ] Do not enable before stable fills exist
- [ ] Do not use allocator to compensate for missing alpha

### 5.7 ModelAlphaAgent -> blocked until model exists

Current code touchpoints:

- Init path: `main.py:3169-3173`
- Signal capture: `main.py:6106-6124`
- Current role: `main.py:7110-7145`
- Dashboard export: `main.py:7175-7185`, `main.py:14913-14923`
- Fail-closed neutral behavior: `agents/model_alpha_agent.py:463-464`

Current status:

- The runtime role is already `CONTEXT` / advisory.
- Without a model artifact it emits neutral signals.
- Therefore it currently adds no usable live alpha.

Checklist:

- [ ] Model artifact exists and loads without mock fallback
- [ ] Non-neutral outputs appear in shadow runs
- [ ] Offline validation proves incremental value over base quant
- [ ] Only after that revisit whether it should stay `CONTEXT` or become stronger

Do not do:

- [ ] Do not promote a neutral agent just because the wiring exists
- [ ] Do not count missing-model zeroes as live alpha evidence

## 6. What to do next

Recommended next execution sequence:

1. Stabilize the fill path and capture the first clean `FILL`.
2. Promote `LeadLag` as `CONTEXT` only.
3. Keep `CompositeToxicity` and `LearnedExecutionPolicy` in `EXECUTION` advisory mode and measure fill-quality delta.
4. Confirm whether G6 is intentionally in shadow or live sizing mode.
5. Only after stable fills exist, decide whether `aggressive_allocator` should remain disabled.
6. Keep DRL capped at `EXIT_ONLY`.
7. Do nothing with `ModelAlphaAgent` until the model artifact exists.

## 7. One-line rule

HMATS does not currently lack directional alpha.
It lacks a clean realized-alpha baseline for measuring additional module promotion.

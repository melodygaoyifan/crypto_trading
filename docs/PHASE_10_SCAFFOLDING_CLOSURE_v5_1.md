# HMATS v5.1 — Phase 10 Scaffolding Closure (Promotion Gate Plan, ADVISORY)

**Status:** SCAFFOLD COMPLETE — actual flip-to-live remains operator-gated.
**Generated:** 2026-04-29

## What this delivers

Phase 10 in the v5.1 prompt (Day 57+) is the promotion gate that converts shadow strategies + sleeves into live participants after their 30d shadow window has produced sufficient evidence. The full Phase 10 has two halves:

1. **Plan generator** (this delivery) — read `shadow_ic` + `sleeve_pnl` reports, decide what would promote, write a plan JSON.
2. **Plan applier** (deferred) — read the plan, flip strategies into fusion / wire allocator → UnifiedPositionSizer. Requires operator approval to proceed; not built yet because it's the only step that mutates production runtime.

Pulling Phase 10 plan-generator forward into the [PARAMETER 3 / 5] gap means: the moment Tier 1 + Pre-6 + 6.0prep ledgers accumulate 30d of post-deploy data, the operator can run `promotion_plan.py` and get a precise list of what's safe to flip — no analysis paralysis at the Day 57+ checkpoint.

## What changed

| File | Change |
|---|---|
| `analytics/promotion_gate/__init__.py` | NEW — package marker |
| `analytics/promotion_gate/promotion_plan.py` | NEW — plan generator with 4 strategy actions + 4 sleeve actions, `build_plan()` orchestrator, CLI |
| `tests/test_promotion_plan.py` | NEW — 26 unit tests |

## Architecture

**Strategy decision matrix:**

| `shadow_ic` verdict | window_days | Plan action |
|---|---|---|
| PROMOTE | ≥30 | `PROMOTE_TO_FUSION` |
| PROMOTE | <30 | `HOLD_SHADOW` (Iron Law 7 enforcement) |
| KILL | any | `ARCHIVE` |
| INSUFFICIENT_SAMPLES | any | `EXTEND_SHADOW` |
| HOLD / unknown | any | `HOLD_SHADOW` |

**Sleeve decision matrix:**

| Condition | Plan action |
|---|---|
| `name == "unknown"` | `SKIP_NO_DATA` |
| `n_days < 7` | `SKIP_NO_DATA` |
| `attribution_coverage < 0.95` | `DEFER_INSUFFICIENT_COVERAGE` |
| `realized_vol == 0` | `SKIP_NO_DATA` |
| `\|realized−estimated\| / estimated > 0.50` | `FLAG_FOR_OPERATOR_REVIEW` |
| else | `UPDATE_ALLOCATOR_REALIZED_VOL` |

The drift gate (50%) flags cases where the estimated vol used by the Phase 7 allocator is materially out of date — operator decides whether to re-estimate or accept the realized number.

## Plan output schema

```json
{
  "generated_at": "2026-04-29T...",
  "inputs": {
    "shadow_ic_report": ".../shadow_ic_20260513_063000.json",
    "sleeve_pnl_report": ".../sleeve_pnl_20260513_063500.json"
  },
  "thresholds": {
    "min_shadow_days_for_promotion": 30,
    "min_attribution_coverage": 0.95,
    "max_vol_drift_pct": 0.50
  },
  "shadow_strategy_actions": [
    {"strategy": "ofi", "asset": "BTC", "action": "PROMOTE_TO_FUSION",
     "reason": "verdict=PROMOTE + window >= 30d", "verdict": "PROMOTE",
     "ic_per_horizon": {"4": 0.06, "12": 0.07, "24": 0.08},
     "n_records": 180, "annualized_sharpe": 0.7},
    ...
  ],
  "sleeve_actions": [
    {"sleeve": "directional_short", "action": "UPDATE_ALLOCATOR_REALIZED_VOL",
     "reason": "coverage OK + drift OK; feed realized_vol=42.30%",
     "n_days": 32, "n_fills": 280, "realized_vol": 0.423,
     "estimated_vol": 0.45, "annualized_sharpe": 0.61, "total_net_pnl": 245.00},
    ...
  ],
  "summary": {
    "n_strategy_promote": 1, "n_strategy_kill": 1, "n_strategy_hold": 0,
    "n_strategy_extend": 1, "n_sleeve_update": 1, "n_sleeve_defer": 0,
    "n_sleeve_flag": 0,
    "blockers": []
  },
  "advisory_only": true
}
```

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | reads JSON reports only |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | UNCHANGED | analytics/ work, not training/ |
| 4. fail-closed | HELD | missing reports → blocker list, no auto-promotion; malformed JSON → load_report returns None; unknown verdict → HOLD_SHADOW |
| 5. DRL ACTIVE floor | UNCHANGED | offline tooling |
| 6. ≥3 active strategies | UNCHANGED | plan does not auto-flip; operator can refuse any PROMOTE_TO_FUSION |
| 7. ≥30d shadow before promotion | **HELD via test** `test_decide_strategy_promote_blocked_when_short_window` — PROMOTE downgraded to HOLD when `window_days < 30` |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not started |
| 9. post-only default | UNCHANGED | execution layer untouched |
| 10 (NEW phrasing) — ZERO production runtime side-effect | **HELD via static check** `test_promotion_plan_does_not_import_runtime_state` — reads source and asserts no import of unified_position_sizer / sleeve_allocator_v5_1 / authority_fusion / main |

## CLI smoke

```bash
$ venv/Scripts/python.exe -X utf8 analytics/promotion_gate/promotion_plan.py
========================================================================================
  PROMOTION GATE PLAN  (advisory_only=True — operator must apply manually)
========================================================================================
  generated_at: 2026-04-29T04:53:07
  shadow_ic_report:  None
  sleeve_pnl_report: .../sleeve_pnl_20260429_042747.json

  BLOCKERS:
    - no shadow_ic report found

  STRATEGY ACTIONS:
    (none — shadow_ic report missing or empty)

  SLEEVE ACTIONS:
    SKIP_NO_DATA  unknown  realized=15.27% estimated=n/a  reason=unknown sleeve bucket
========================================================================================
```

Expected per local state: no shadow_ic report exists locally (Phase 4/8 ledgers don't write outside engine ticks). Sleeve report shows only "unknown" bucket (pre-P25 attribution era). Both correctly route to non-promote actions; CLI exits 1 due to blocker. Once Tier 1 ships and live ledgers accumulate, the same CLI will produce a real plan.

## Test results

```
tests/test_promotion_plan.py                       26/26 PASS
─────────────────────────────────────────────────────────────
Cumulative cross-cutting (Phase 0+1+4+8+7+Pre-6+6.0prep+10) 330/330 PASS
```

Test breakdown:
- `decide_strategy_action`: 6 cases (all 4 verdict types + window-gate + unknown)
- `decide_sleeve_action`: 7 cases (unknown / few_days / low_cov / zero_vol / drift / clean / no-estimate)
- `build_strategy_actions` + `build_sleeve_actions`: 3
- `build_plan` end-to-end: 5 (happy path / both missing / only shadow / low coverage / short window)
- `latest_report` + `load_report` fail-closed: 4
- Iron Law 10 static check: 1

## What is intentionally NOT built

- **Plan applier** (`apply_promotion_plan.py`) — reads a plan JSON and actually flips strategies into fusion + wires allocator → UnifiedPositionSizer. This is the only step that mutates production runtime. Iron Law 7 + 10 require operator approval; will be built when operator signs off on a specific plan + at least one strategy clears the 30d gate.
- **Periodic plan-generation cron** — would auto-run `compute_shadow_ic` → `compute_sleeve_pnl` → `promotion_plan` daily. Phase 11 (60-day review) will scaffold this if the manual workflow proves fragile.
- **Audit trail of applied plans** — once the applier exists, every plan that mutated production state should be archived with the operator signature for compliance.

## Cumulative session totals (8 phases delivered)

| Phase | Tests added |
|---|---|
| 0 — Pre-flight + IC re-baseline | 0 (tooling) |
| 1 — Strategy archive gate | 0 (reused 33) |
| 4 — Microstructure shadow | +20 |
| 8 — Cascade shadow | +18 |
| 7 — Sleeve allocator advisory | +25 |
| Pre-6 — Backtest + Shadow IC framework | +20 |
| 6.0 prep — Per-sleeve PnL slicer | +30 |
| 10 scaffold — Promotion gate plan | +26 |
| **Total v5.1 work this session** | **+139 new tests** |

Cross-cutting cumulative: **330/330 PASS**. Iron Laws 1, 2, 3, 4, 5, 6, 7, 9 verified intact across all phases. Iron Law 8 N/A until Phase 2. Iron Law 10 (zero runtime side-effect) introduced + verified in Phase 10.

## Pending operator decisions before deploy

8 closure docs ready for review/commit:
- `docs/PHASE_BASELINE_v5_1.md` (Phase 0)
- `docs/PHASE_1_CLOSURE_v5_1.md`
- `docs/PHASE_4_CLOSURE_v5_1.md`
- `docs/PHASE_8_CLOSURE_v5_1.md`
- `docs/PHASE_7_CLOSURE_v5_1.md`
- `docs/PHASE_PRE6_CLOSURE_v5_1.md`
- `docs/PHASE_6_0_PREP_CLOSURE_v5_1.md`
- `docs/PHASE_10_SCAFFOLDING_CLOSURE_v5_1.md`

Each with prepared commit message. None committed/pushed/deployed autonomously.

## Outstanding [PARAMETER]s

| # | Parameter | Status | Phase |
|---|---|---|---|
| 1 | Branch X or Y | RESOLVED Y | — |
| 2 | 12-strategy buckets | RESOLVED | — |
| 3 | V4.3 cutover mode | **PENDING OPERATOR** | Phase 2 |
| 4 | V8 DRL retrain Y/N | DEFAULT N | Phase 2 |
| 5 | Phase 6 constitutional override | **PENDING OPERATOR** | Phase 6 |
| 6 (NEW) | Plan applier autonomy boundary | **PENDING OPERATOR** | Phase 10 applier |

[PARAMETER 6] is the new question raised by Phase 10 scaffold completion: when a plan says PROMOTE_TO_FUSION on a strategy, who flips the switch? Recommended default: operator runs `apply_promotion_plan.py --plan path.json --confirm` interactively; the applier never auto-runs. This codifies Iron Law 10 enforcement at the human-protocol layer.

## Deploy step

Phase 10 scaffold is offline tooling. Recommended commit message:
```
v5.1 Phase 10 scaffold: promotion gate plan generator (advisory)

analytics/promotion_gate/promotion_plan.py reads latest shadow_ic + sleeve_pnl
reports and emits a "would-promote" plan JSON. Strategy actions:
PROMOTE_TO_FUSION (verdict=PROMOTE + window >= 30d), HOLD_SHADOW,
ARCHIVE (verdict=KILL), EXTEND_SHADOW (insufficient samples). Sleeve actions:
UPDATE_ALLOCATOR_REALIZED_VOL, DEFER_INSUFFICIENT_COVERAGE (coverage < 95%),
FLAG_FOR_OPERATOR_REVIEW (vol drift > 50%), SKIP_NO_DATA.

Iron Law 7 enforcement: PROMOTE downgraded to HOLD when shadow window < 30d.
Iron Law 10 (NEW): zero production runtime side-effect — static check via
test_promotion_plan_does_not_import_runtime_state. Plan applier deliberately
NOT built; operator must run apply_promotion_plan.py --confirm interactively
when one is built post-30d-shadow-maturity.

330/330 cross-cutting tests pass (Phase 0+1+4+8+7+Pre-6+6.0prep+10 union).
26 new tests cover all 4 strategy actions, all 4 sleeve actions, end-to-end
build_plan with missing reports + low coverage + short window, latest_report +
load_report fail-closed, Iron Law 10 static check.
```

## What's next

Two more pure-tooling work items can proceed without operator unblock:

**Path C — Phase 11 60-day review framework**: synthesizes across all v5.1 reports (shadow_ic, sleeve_pnl, promotion plans, equity_history) into a single dashboard showing whether the v5.1 stack hit its quantitative targets. Pure aggregation, zero runtime side-effect.

**Path D — Plan applier scaffold (dry-run only)**: builds `apply_promotion_plan.py --dry-run` that prints what it WOULD do but never mutates state. Establishes the exact mutation API surface so operator review is concrete. Live mode (`--confirm`) deliberately not built without operator sign-off.

Both are independent and can ship together or separately. Or pause here for operator review of the 8 closure docs accumulated so far — this is the natural Tier 2-prep checkpoint.

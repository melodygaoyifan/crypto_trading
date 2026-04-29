# HMATS v5.1 — Phase 6.0 Prep Closure (Per-Sleeve PnL Slicer)

**Status:** COMPLETE — Phase 6 main work blocked on operator constitutional override.
**Generated:** 2026-04-29

## Why this doc exists

Phase 6 (ML factor extraction, v5.1 prompt Day 42-56) requires an operator-signed constitutional override for `training/factor_extraction/` per Iron Law 3. I cannot self-sign on the operator's behalf.

Two off-critical-path work items can proceed without that signature: per-sleeve PnL slicer (this doc) and Phase 10 promotion-gate scaffolding. I picked the slicer first because it directly feeds Phase 7's `SleeveAllocator.update_realized_vol()` mechanism — measurable improvement the moment Tier 1 deploys, not future-tense scaffold.

## What changed

| File | Change |
|---|---|
| `analytics/sleeve_attribution/__init__.py` | NEW — package marker |
| `analytics/sleeve_attribution/compute_sleeve_pnl.py` | NEW — `load_fills`, `aggregate_daily_sleeve_pnl`, `compute_vol_and_sharpe`, `build_report`, `update_allocator_from_report`, CLI |
| `tests/test_sleeve_pnl_slicer.py` | NEW — 30 unit tests |

## Architecture

**Sleeve mapping** (`map_agent_to_sleeve`):
- 19 directional agents → `directional_short` (existing live system)
- `ofi`, `vpin_spike`, `kyle_lambda`, `phase4_*` → `microstructure`
- `cascade_anticipation`, `stop_hunt_defense`, `phase8_*` → `cascade`
- empty / unrecognized → `unknown` (debug-only, NOT promoted)

**Pipeline:**
1. `load_fills(ledger_dir, since)` — reads `data/shadow_ledger/ledger_*.jsonl`, filters by timestamp, recovers naive timestamps as UTC (CLAUDE.md P39/P40 family), parses FILL records only, attaches sleeve label.
2. `aggregate_daily_sleeve_pnl(fills)` — buckets by `(sleeve, YYYY-MM-DD)`, net PnL = `realized_pnl − fee` per fill (entries have `realized_pnl=0`, closes have realized PnL).
3. `compute_vol_and_sharpe(daily_pnls)` — converts to daily returns against running equity, then annualized vol + Sharpe over 365 trading days/yr (crypto 24/7).
4. `update_allocator_from_report(allocator, report)` — feeds `annualized_vol` per sleeve into `SleeveAllocator.update_realized_vol()`. Skips unknown sleeves not registered in the allocator. Iron Law 7: caller responsible for ensuring the allocator is advisory, NOT one wired into UnifiedPositionSizer (verified by `test_slicer_does_not_import_unified_position_sizer`).

**Report schema:**
```json
{
  "generated_at": "2026-04-29T...",
  "window_days": 30,
  "since": "2026-03-30T...",
  "n_total_fills": 211,
  "n_unknown_sleeve_fills": 211,
  "attribution_coverage": 0.0,
  "initial_capital": 10000.0,
  "sleeves": [
    {"name": "directional_short", "n_days": 32, "n_fills": 211,
     "total_net_pnl": -1731.89, "annualized_vol": 0.1527,
     "annualized_sharpe": -14.09}
  ]
}
```

`attribution_coverage` is the new Phase-10-gate input: % of fills that mapped to a known sleeve. Coverage < 95% should block allocator updates.

## Live smoke results

Ran against locally-mirrored `data/shadow_ledger/` (211 FILL records spanning 2026-02-11 → 2026-04-25, naive→UTC recovered):

```
SLEEVE PNL REPORT  window=90d  fills=211  attribution_coverage=0.0%
sleeve                  days  fills    net_pnl    ann_vol   sharpe
unknown                   32    211   -1731.89    15.27%   -14.09

WARNING: attribution coverage 0.0% < 95% — 211 fills lack primary_agent
(likely pre-P25 era). Phase 10 promotion gate should require coverage >= 95%
before feeding allocator.
```

Two findings from the smoke that informed slicer design:

1. **Naive-timestamp tolerance:** initial run failed on 11 older ledger files (`TypeError: can't compare offset-naive and offset-aware datetimes`). CLAUDE.md P39/P40 family — pre-2026-04-24 records were written with `datetime.utcnow()` (naive). Slicer now treats naive as UTC; recovers full window (211 fills vs initial 106).

2. **All 211 fills attributed to "unknown":** P25 fix (primary_agent population) didn't land until 2026-04-24, so this entire local sample is pre-P25 and lacks `primary_agent`. The slicer correctly reports `attribution_coverage=0.0%` and emits a WARNING. Once Tier 1 + Pre-6 commits ship and live engine accumulates post-P25 fills, coverage will climb. **Phase 10 promotion gate should require coverage ≥ 95% before feeding the allocator** (recommendation now in the closure doc; implementation deferred to Phase 10).

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | slicer reads ledgers + reports; no feature/manifest touch |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | UNCHANGED | slicer lives in `analytics/sleeve_attribution/`, not `training/` |
| 4. fail-closed | HELD | malformed JSONL skipped + WARN; non-FILL records ignored; NaN PnL coerced to 0; missing `primary_agent` → "unknown" sleeve (NOT inflated into known sleeves); `update_allocator_from_report` exception per-call → counted-as-failure, loop continues |
| 5. DRL ACTIVE floor | UNCHANGED | offline tooling; runtime DRL authority untouched |
| 6. ≥3 active strategies | UNCHANGED | no strategy archive change |
| 7. ≥30d shadow before promotion | HELD | `test_slicer_does_not_import_unified_position_sizer` reads source and asserts no import of `risk.unified_position_sizer`; `update_allocator_from_report` is opt-in by caller; no auto-feed |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not started |
| 9. post-only default | UNCHANGED | execution layer untouched |

## Test results

```
tests/test_sleeve_pnl_slicer.py                30/30 PASS
─────────────────────────────────────────────────────────
Cumulative cross-cutting (Phase 0+1+4+8+7+Pre-6+6.0prep) 304/304 PASS
```

Test coverage:
- `map_agent_to_sleeve` parametrized: 15 cases (case-insensitivity, prefix matching, unknown handling)
- `load_fills`: missing dir, since-filter, malformed-line skip, non-FILL ignore, NaN-PnL coercion (5)
- `aggregate_daily_sleeve_pnl`: per-sleeve sum, day-boundary split (2)
- `compute_vol_and_sharpe`: empty / single-day / realistic / constant-zero (4)
- `build_report`: 5-day end-to-end (1)
- `update_allocator_from_report`: known sleeves fed correctly, unknown sleeves skipped, allocator exception isolated (2)
- Iron Law 7 static check: 1

## What does NOT happen yet

- **Per-sleeve PnL slicer is NOT scheduled** — designed to be invoked manually or on-demand. Phase 10 will add a periodic call (e.g. nightly cron or on-tick after each close) that runs the slicer and feeds the allocator.
- **Realized vol does not yet flow into the live `SleeveAllocator` instance in main.py** — Phase 10 promotion gate will do this after operator approves the auto-feed. For now the allocator's per-tick advisory record continues to use `estimated_vol_pct` until realized history accumulates.
- **Microstructure / cascade sleeve PnL is empty** — those sleeves are shadow-only at the runtime layer (Phase 4/8); they don't generate FILL records yet. Phase 10 promotion converts them to live participants, at which point the slicer will start accumulating their realized vol.

## Phase 6 readiness

The two non-`training/` Phase 6 prerequisites identified during Pre-6 planning are now met:
- ✓ Backtest framework (Pre-6.1)
- ✓ Per-sleeve PnL slicer (this doc)

Phase 6 main work (autoencoder factor extraction in `training/factor_extraction/`) remains blocked on **[PARAMETER 5 NEW]** — operator constitutional override sign per Iron Law 3.

## Outstanding [PARAMETER]s

| # | Parameter | Status | Phase |
|---|---|---|---|
| 1 | Branch X or Y | RESOLVED Y | — |
| 2 | 12-strategy buckets | RESOLVED | — |
| 3 | V4.3 cutover mode | **PENDING OPERATOR** | Phase 2 (Day 14) |
| 4 | V8 DRL retrain Y/N | DEFAULT N | Phase 2 |
| 5 | Phase 6 constitutional override | **PENDING OPERATOR** | Phase 6 (Day 42) |

## Deploy step (operator action)

Phase 6.0 prep ships entirely as new files + tests. Recommended commit message:
```
v5.1 Phase 6.0 prep: per-sleeve PnL slicer + realized-vol feeder

analytics/sleeve_attribution/compute_sleeve_pnl.py: reads
data/shadow_ledger/ledger_*.jsonl FILL records, attributes each fill's net PnL
(realized_pnl - fee) to a sleeve via primary_agent → sleeve mapping (19
directional agents -> directional_short; ofi/vpin/kyle -> microstructure;
cascade_* -> cascade; unknown -> "unknown" debug bucket). Computes daily
PnL series, annualized vol + Sharpe (365 days/yr crypto), exports report
JSON. update_allocator_from_report() is the Phase 10 hook that feeds
realized vol into SleeveAllocator.update_realized_vol().

Naive timestamp recovery (CLAUDE.md P39/P40 family): pre-2026-04-24 records
written with datetime.utcnow() are recovered as UTC. Live smoke went from
106 → 211 fills after the fix on the local mirror.

attribution_coverage metric in the report: % of fills that mapped to a
known sleeve. Phase 10 promotion gate should require coverage >= 95%
before feeding the allocator. Local smoke shows 0% (pre-P25 era data;
post-P25 data on Hetzner will populate primary_agent properly).

304/304 cross-cutting tests pass (Phase 0+1+4+8+7+Pre-6+6.0prep union).
30 new tests cover sleeve mapping (15 parametrized), load_fills (5 fail-
closed paths), daily PnL aggregation, vol/Sharpe computation, end-to-end
report shape, allocator-feed correctness + exception isolation, Iron Law 7
static check (no UnifiedPositionSizer import).
```

## Cumulative phase status

| Phase | Status | Tests added |
|---|---|---|
| 0 — Pre-flight + IC re-baseline | DONE | 0 (tooling unblock) |
| 1 — Strategy archive gate | DONE | 0 (reused existing) |
| 4 — Microstructure shadow | DONE | +20 |
| 8 — Cascade shadow | DONE | +18 |
| 7 — Sleeve allocator advisory | DONE | +25 |
| Pre-6 — Backtest + Shadow IC framework | DONE | +20 |
| 6.0 prep — Per-sleeve PnL slicer | DONE | +30 |
| **Total v5.1 work this session** | **7 phases** | **+113 tests** |

Cross-cutting cumulative: **304/304 PASS**. Iron Laws 1-7, 9 verified intact across all phases. Iron Law 8 N/A until Phase 2 begins.

## Pending operator decisions before deploy

7 closure docs ready for review/commit:
- `docs/PHASE_BASELINE_v5_1.md` (Phase 0)
- `docs/PHASE_1_CLOSURE_v5_1.md`
- `docs/PHASE_4_CLOSURE_v5_1.md`
- `docs/PHASE_8_CLOSURE_v5_1.md`
- `docs/PHASE_7_CLOSURE_v5_1.md`
- `docs/PHASE_PRE6_CLOSURE_v5_1.md`
- `docs/PHASE_6_0_PREP_CLOSURE_v5_1.md`

Each has its own prepared commit message. Can ship as one combined Tier 1+ commit or 7 separate. None committed/pushed/deployed autonomously.

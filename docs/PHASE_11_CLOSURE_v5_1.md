# HMATS v5.1 — Phase 11 Closure (60-Day Post-Deploy Review Aggregator)

**Status:** SCAFFOLD COMPLETE — runs immediately, becomes meaningful after 60d post-Tier-1 deploy.
**Generated:** 2026-04-29

## What this delivers

The v5.1 prompt's Phase 11 specifies a 60-day post-deploy gate against 11 quantitative targets (net Sharpe, max DD, DRL ACTIVE, ≥3 strategies, maker fee ratio, per-sleeve Sharpes, fee/alpha ratio, Coinbase API uptime). Phase 11 ships an aggregator that synthesizes across all v5.1 advisory layers + live runtime signals into a single PASS/FAIL/INSUFFICIENT_DATA dashboard.

Built as offline tooling (Iron Law 10 — zero runtime side-effect). Operator runs it manually after Day 60 of post-Tier-1 deploy and gets a pass/fail decision per target. If any target FAILs, operator decides whether to roll forward, hold, or initiate the Phase 11 retreat plan.

## What changed

| File | Change |
|---|---|
| `analytics/sixty_day_review/__init__.py` | NEW |
| `analytics/sixty_day_review/review_aggregator.py` | NEW — `build_review`, 11 check builders, CLI |
| `tests/test_sixty_day_review.py` | NEW — 27 unit tests |

## Targets codified (per v5.1 prompt Phase 11)

| Target | Threshold | Source |
|---|---|---|
| net_sharpe | ≥ 1.5 | equity_history daily closes, annualized 365 |
| max_drawdown | < 25% | equity_history peak-to-trough |
| maker_fee_ratio | ≥ 95% | shadow_ledger FILL.data.order_type *(known gap — see below)* |
| fee_alpha_ratio | < 5% | shadow_ledger fees / Σ\|realized_pnl\| |
| active_strategies_min_3 | ≥ 3 | configs/strategy_v5_1_decisions.json |
| sleeve_sharpe_directional_short | ≥ 0.8 | sleeve_pnl report annualized_sharpe |
| sleeve_sharpe_microstructure | ≥ 1.0 | sleeve_pnl report |
| sleeve_sharpe_cascade | ≥ 0.8 | sleeve_pnl report |
| sleeve_sharpe_funding | ≥ 1.5 | sleeve_pnl report (Phase 3 dependency) |
| sleeve_sharpe_ml_factor | ≥ 1.0 | sleeve_pnl report (Phase 6 dependency) |
| coinbase_api_uptime | ≥ 99% | NOT_APPLICABLE until Phase 2 ships, then `--coinbase-migration-done` flag enables stub |

## Status semantics

```
PASS              — observed meets threshold
FAIL              — observed below threshold; v5.1 prompt's failure mode triggers
INSUFFICIENT_DATA — couldn't measure (window too short, source missing, etc.)
NOT_APPLICABLE    — gated by an unfinished phase (e.g. Coinbase pre-Phase-2)
```

Overall:
- `PASS` only if zero FAIL and zero INSUFFICIENT_DATA
- `FAIL` if any FAIL
- `INSUFFICIENT_DATA` otherwise

This means a partial-data run never accidentally claims success — Iron Law 4 fail-closed.

## Live CLI smoke (against current Hetzner equity_history + local ledger mirror)

```
HMATS v5.1  60-DAY REVIEW   window=90d   OVERALL: FAIL
inputs.n_equity_points=131  n_fills=211  n_market_orders=0

  net_sharpe                       FAIL              1.5    0.82
      detail: computed over 9 daily closes
  max_drawdown                     PASS              <0.25  0.1902
  maker_fee_ratio                  INSUFFICIENT_DATA >=0.95 n/a
      detail: 211 fills lack order_type field (lives in PLACE_ORDER log or
      fill_quality.jsonl, NOT shadow_ledger). Wire those sources for live
      measurement; V15 production logs show 98.7% maker.
  fee_alpha_ratio                  FAIL              <0.05  2.0956
  active_strategies_min_3          PASS              >=3    8
  sleeve_sharpe_*                  INSUFFICIENT_DATA (5 sleeves) — Phase 7 sleeves still shadow + Phase 3/6 not yet shipped
  coinbase_api_uptime              NOT_APPLICABLE   >=0.99  n/a

  SUMMARY: PASS=2  FAIL=2  INSUFFICIENT_DATA=6  N/A=1  (total 11)
```

Honest baseline read pre-Tier-1:
- **net_sharpe = 0.82** (vs target 1.5) — pre-v5.1 baseline; expected to improve as v5.1 stack ships
- **max_drawdown = 19.02%** PASS — within 25% Iron Law
- **fee/alpha = 209%** FAIL — matches CLAUDE.md P64-B finding (fees > realized PnL because pre-P25-era records have realized_pnl=0; expect coverage to climb post-Tier-1)
- **active_strategies = 8** PASS — Phase 1 archive applied
- **5 sleeve Sharpes INSUFFICIENT_DATA** — Phase 7 sleeves still shadow-only; funding/ml_factor pending Phase 3/6
- **Coinbase N/A** — Phase 2 not started

## Issue caught by live smoke (and fixed)

Initial run produced `maker_fee_ratio = FAIL = 0.0` because FILL records in shadow_ledger don't carry `order_type` field — that lives in PLACE_ORDER logs or `fill_quality.jsonl`. Misleading FAIL on a target the system actually clears at 98.7% per V15 evidence.

Fixed by tightening: `compute_fee_metrics` now returns `n_classified` count; `build_checks` routes to `INSUFFICIENT_DATA` with explicit detail if `n_classified == 0`. Operator sees the gap honestly. Future improvement (deferred): wire the maker check to read `data/fill_quality.jsonl` or grep `[FILL-QUALITY]` from production logs — both have the LIMIT/MARKET signal.

Test added: `test_compute_fee_metrics_no_order_type_classified_zero` exercises the no-order_type path on a synthetic FILL stream that mirrors real production schema.

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | reads JSON + JSONL only |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | UNCHANGED | analytics/, not training/ |
| 4. fail-closed | HELD | missing equity → INSUFFICIENT_DATA on Sharpe + DD; missing/malformed FILLs skipped + WARN; no-classification → INSUFFICIENT_DATA (not 0); missing reports → blocker; overall PASS only when zero fail+insufficient |
| 5. DRL ACTIVE floor | UNCHANGED | offline tooling |
| 6. ≥3 active strategies | CHECKED — `active_strategies_min_3` is one of the 11 targets |
| 7. ≥30d shadow before promotion | N/A (this is the review, not the gate) |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not started |
| 9. post-only default | CHECKED — `maker_fee_ratio` target |
| 10. zero runtime side-effect | **HELD via static check** `test_review_does_not_import_runtime_state` — reads source and asserts no import of unified_position_sizer / sleeve_allocator_v5_1 / authority_fusion / main |

## Test results

```
tests/test_sixty_day_review.py    27/27 PASS
─────────────────────────────────────────────────────────────────────
Cumulative cross-cutting (Phase 0+1+4+8+7+Pre-6+6.0prep+10+11)  357/357 PASS
```

Test breakdown:
- `compute_sharpe_and_dd`: empty / single / uptrend / drawdown-detect / daily-close (5)
- `load_equity_history`: missing / since-filter / malformed / naive-ts-recovery (4)
- `compute_fee_metrics`: empty / maker-dom / no-order-type / no-realized (4)
- `extract_sleeve_sharpes`: skip-unknown / handle-None (2)
- `count_active_strategies`: real config (=8) / missing file (=−1) (2)
- `build_checks`: insufficient-data / below-min-strategies / per-sleeve-missing / per-sleeve-pass / maker-fail-when-low / maker-insufficient-when-no-classification / coinbase-done-stub (7)
- `build_review` end-to-end: no-data → not-PASS / synthetic complete → PASS where applicable (2)
- Iron Law 10 static check (1)

## CLI usage

```bash
# Standard 60-day review (after Phase 11 deadline)
venv/Scripts/python.exe -X utf8 analytics/sixty_day_review/review_aggregator.py

# After Coinbase migration ships (Phase 2)
venv/Scripts/python.exe -X utf8 analytics/sixty_day_review/review_aggregator.py \
    --coinbase-migration-done

# Custom window (e.g. weekly checkpoint)
venv/Scripts/python.exe -X utf8 analytics/sixty_day_review/review_aggregator.py \
    --window-days 7
```

Exit codes:
- 0 = OVERALL PASS
- 1 = OVERALL FAIL  (any target FAIL → trigger v5.1 retreat plan per prompt's failure mode 10)
- 2 = OVERALL INSUFFICIENT_DATA (rerun after more data accumulates)

## Phase 11 follow-up — fill_quality.jsonl wire-up (2026-04-29 same session)

The initial Phase 11 ship reported `maker_fee_ratio = INSUFFICIENT_DATA` because shadow_ledger FILL records lack `order_type`. Same-session follow-up wires `data/fill_quality.jsonl` (the canonical maker/taker source per V15 evidence) into the aggregator:

- New helper `classify_maker_taker_from_fill_quality(path, since)` returns `(n_limit, n_market, n_classified)`. Recognized maker types: `LIMIT, POST, POSTONLY, POST-ONLY, POST_ONLY`. `MARKET` = taker. Other / unknown / missing order_type → not classified.
- New CLI flag `--fill-quality data/fill_quality.jsonl` (defaults to that path)
- `build_review` now uses fill_quality classification when available and falls back to shadow_ledger order_type only if fill_quality has no records (legacy path)
- Report inputs gain `n_classified_via_fill_quality` field for Phase 11 audit trail
- 5 new tests: `classify_maker_taker_*` (missing file / counts / since-filter / malformed-skip / unknown-order-type-skip)

**Live re-smoke after wire-up (Hetzner fill_quality.jsonl, 173 records):**
```
maker_fee_ratio       PASS    >=0.95   observed=0.994   detail: 173/211 fills classified
```

99.4% maker — exceeds 95% target. Validates V15's 98.7% finding with current data showing even higher maker ratio. Iron Law 9 holding cleanly.

Updated overall: **PASS=3 / FAIL=2 / INSUFFICIENT_DATA=5 / N/A=1**.

## What still does NOT happen yet
- **Coinbase API uptime** — stub ready at `coinbase_api_uptime`. Implementation needs the actual probe (e.g. `gh api` to a status page, or scrape Hetzner monitoring). Built when Phase 2 ships.
- **DRL ACTIVE check** — Iron Law 5 already enforced runtime-side. Could add a check that reads `drl_promotion_state.json` mtime + content; deferred (the actual runtime invariant is more important than the review-time read).
- **Retreat plan trigger** — when overall=FAIL, operator should run the retreat path described in v5.1 prompt's failure mode 10 ("60d Sharpe < 0.5 → 全部回滚 v3.6 状态"). Phase 11 reports the FAIL; the rollback workflow is operator-driven not auto-applied.

## Cumulative session totals (9 phases delivered)

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
| 11 — 60-day review aggregator | +27 |
| 11 follow-up — fill_quality wire-up | +5 |
| **Total** | **+171 new tests** |

Cross-cutting cumulative: **362/362 PASS**. Iron Laws 1, 2, 3, 4, 5, 6, 7, 9, 10 verified intact across all phases. Iron Law 8 N/A until Phase 2.

## Pending operator decisions before deploy

9 closure docs ready for review/commit:
- `PHASE_BASELINE_v5_1.md` (Phase 0)
- `PHASE_1_CLOSURE_v5_1.md`
- `PHASE_4_CLOSURE_v5_1.md`
- `PHASE_8_CLOSURE_v5_1.md`
- `PHASE_7_CLOSURE_v5_1.md`
- `PHASE_PRE6_CLOSURE_v5_1.md`
- `PHASE_6_0_PREP_CLOSURE_v5_1.md`
- `PHASE_10_SCAFFOLDING_CLOSURE_v5_1.md`
- `PHASE_11_CLOSURE_v5_1.md`

Each with prepared commit message. None committed/pushed/deployed autonomously.

## Outstanding [PARAMETER]s

| # | Parameter | Status | Phase |
|---|---|---|---|
| 1 | Branch X or Y | RESOLVED Y | — |
| 2 | 12-strategy buckets | RESOLVED | — |
| 3 | V4.3 cutover mode | **PENDING OPERATOR** | Phase 2 |
| 4 | V8 DRL retrain Y/N | DEFAULT N | Phase 2 |
| 5 | Phase 6 constitutional override | **PENDING OPERATOR** | Phase 6 |
| 6 | Plan applier autonomy boundary | **PENDING OPERATOR** | Phase 10 applier |

## Deploy step (operator action)

Phase 11 ships entirely as new files + tests (offline tooling). Recommended commit message:
```
v5.1 Phase 11: 60-day post-deploy review aggregator (advisory)

analytics/sixty_day_review/review_aggregator.py synthesizes across
equity_history (Sharpe + max DD), shadow_ledger (fee/alpha + maker ratio),
sleeve_pnl report (per-sleeve Sharpes), strategy decisions JSON (active count)
into 11 PASS/FAIL/INSUFFICIENT_DATA/NOT_APPLICABLE checks against v5.1's
Phase 11 quantitative targets.

Iron Law 4 fail-closed: any missing input maps to INSUFFICIENT_DATA, never
PASS. Overall=PASS only when zero FAIL + zero INSUFFICIENT_DATA. Iron Law 10
static check verifies no import of runtime sizer/fusion/main.

Live smoke against Hetzner equity_history + local shadow_ledger mirror caught
a code-correctness issue: FILL records lack order_type field (lives in
fill_quality.jsonl / PLACE_ORDER log), so naive maker classification produced
0.0/100. Tightened: returns n_classified count; check routes to
INSUFFICIENT_DATA with explicit detail when classification is empty. V15
production-log evidence shows 98.7% maker; future Phase 11 follow-up wires
fill_quality.jsonl as the classification source.

357/357 cross-cutting tests pass (Phase 0+1+4+8+7+Pre-6+6.0prep+10+11
union). 27 new tests cover Sharpe + drawdown math, equity loader fail-closed
+ naive-ts recovery, fee metrics with classified vs unclassified records,
sleeve Sharpe extraction, real-config strategy count, all 7 check-builder
status branches, Iron Law 10 static check.
```

## v5.1 closure status

The v5.1 prompt's failure-mode #10 says "60d Sharpe < 0.5 → 全部回滚 v3.6 状态". Phase 11 implements the *measurement* of that condition. The *retreat* itself is intentionally manual — Iron Law 10 forbids auto-rollback of production state.

This brings session-delivered v5.1 phases to 9. The **only** remaining v5.1 phases that require new code are:
- **Phase 2** (Coinbase migration) — blocked on [PARAMETER 3]
- **Phase 3** (Funding strategies) — depends on Phase 2
- **Phase 6 main** (ML factor extraction) — blocked on [PARAMETER 5]
- **Phase 10 applier** — operator-protocol decision per [PARAMETER 6]
- **Phase 12** (stabilize) — entirely operator-driven

Everything else has been delivered as advisory tooling. Operator can:
1. Review the 9 closure docs
2. Approve commits + deploy Tier 1 + 6.0prep + 10-scaffold + Pre-6 + Phase 11 aggregator
3. Wait 30+ days for shadow ledgers to mature
4. Run promotion_plan + sixty_day_review to drive Phase 10 + 11 decisions

That sequence proceeds without further v5.1 code from me until [PARAMETER 3] / [PARAMETER 5] / [PARAMETER 6] are resolved.

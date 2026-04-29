# HMATS v5.1 — CLAUDE.md Pattern Audit

**Status:** v5.1 deliverables clean against CLAUDE.md anti-patterns.
**Generated:** 2026-04-29

## Audit scope

9 v5.1 deliverable files (excluding tests + docs):
1. `strategies/microstructure_v5_1.py`
2. `strategies/liquidation_cascade_v5_1.py`
3. `defense/strategy_shadow_v5_1.py`
4. `risk/sleeve_allocator_v5_1.py`
5. `training/backtest_framework/backtest_engine.py`
6. `analytics/shadow_ic/compute_shadow_ic.py`
7. `analytics/sleeve_attribution/compute_sleeve_pnl.py`
8. `analytics/promotion_gate/promotion_plan.py`
9. `analytics/sixty_day_review/review_aggregator.py`

Plus `configs/strategy_v5_1_decisions.json` (data, not code).

## Patterns checked

| CLAUDE.md anti-pattern | v5.1 status |
|---|---|
| **P39/P40** — `datetime.utcnow()` / `datetime.now()` mixed naive/aware | ✅ **CLEAN** — zero hits in all 9 files (`grep -rn datetime\.utcnow\|datetime\.now\(\) v5_1_files` empty) |
| **P22** — archive imports / stale module references | ✅ **CLEAN** — zero `from archive` / `import archive.` |
| **P87** — same-class method collision (`def foo` redefined) | ✅ **CLEAN** — duplicate def names exist but across DIFFERENT classes (`SleeveAllocator.__init__` vs `SleeveAdvisorySink.__init__`; 3 `evaluate` methods on 3 different shadow strategies) |
| **P15/P85** — silent reader/writer mismatch (read attr that writer doesn't set) | ✅ **HARDENED** — `update_allocator_from_report` defends with `getattr(allocator, "_sleeves", {})` per P85 norm; the bare `except: known = set()` is now logged |
| **P25/P64/P72** — silent `except: pass` / `logger.debug` swallows | ⚠️→✅ **20 findings → 0 unannotated** after fixes |
| **P3/P8** — agent attribution wiring 4-place rule | N/A — v5.1 strategies are shadow-only (no fusion wire), Iron Law 7 enforces no agent_signals write |
| **P1** — Docker volume name mismatch | N/A — no compose changes |
| **P4** — BEST_FOLDS hardcode drift | N/A — no DRL training touch |
| **P50** — fail-closed contracts | ✅ Held: every loader returns empty `[]` / `{}` / `None` on miss; never auto-promotes; `decide_strategy_action` downgrades PROMOTE → HOLD if window < 30d |

## P25/P64/P72 silent-swallow fixes

`tools/lint_silent_swallow.py` (CLAUDE.md P72 lint) initially reported **20 unannotated silent swallows** across 7 v5.1 files. Categorized + fixed in 4 patterns:

### Pattern A — Per-line/per-record JSONL skip with batch-level WARN at end of loop (12 sites)

Files: `compute_shadow_ic.py:88,93,103,147`, `compute_sleeve_pnl.py:123,134`, `review_aggregator.py:123,130,139,244,251,305,314`.

Each loader iterates a JSONL file and skips malformed lines with `except json.JSONDecodeError: skipped += 1; continue`. The batch counter emits `logger.warning("[X] {skipped} malformed records skipped")` at end of load. **Per-line silence is intentional** — annotated with `# noqa: silent-swallow` + reason pointing at the batch WARN.

`load_equity_history`, `compute_fee_metrics`, `classify_maker_taker_from_fill_quality` in `review_aggregator.py` were missing the batch counter — added in this audit.

### Pattern B — Per-field type coercion helper (5 sites)

Files: `microstructure_v5_1.py:81 (_get_float)`, `liquidation_cascade_v5_1.py:86 (_get_float)`, `sleeve_allocator_v5_1.py:156 (update_realized_vol)`, `sleeve_allocator_v5_1.py:239 (quarter_kelly)`, `review_aggregator.py:398 (extract_sleeve_sharpes)`.

Each is `try: float(x) except (TypeError, ValueError): return default`. The default value IS the documented contract — bad input maps to a documented sentinel. **Annotated `# noqa: silent-swallow`** with reason explaining the fallback contract.

### Pattern C — fsync best-effort fallback (2 sites)

Files: `sleeve_allocator_v5_1.py:328`, `strategy_shadow_v5_1.py:151`.

Each is `f.flush(); try: os.fsync(f.fileno()); except: pass`. fsync isn't supported on all platform/fs combos (Windows + network mounts). `f.flush()` already gives OS-level durability; fsync is the crash-safety upgrade. **Annotated `# noqa: silent-swallow`** with reason.

### Pattern D — Optional dependency / batch-level strategy errors (3 sites, escalated)

- `compute_shadow_ic.py:147 except ImportError` (pandas missing) — was silent return None. **Promoted to `logger.warning`** with explicit detail about IC compute degradation, since pandas is required for the join.
- `backtest_engine.py:248 except Exception (per-bar strategy error)` — was `logger.debug` (below operator visibility per P64). **Refactored** to track `n_strategy_errors` + `last_strategy_error`, emits batch WARN at end of run with total count + last error type. Avoids spamming per-bar but keeps operator visibility.
- `compute_sleeve_pnl.py:313 except` (allocator `_sleeves` access) — was silent fallback to empty set. **Promoted to `logger.warning`** since reaching this means allocator is in weird state (P85-shape).
- `review_aggregator.py:398 except` in `count_active_strategies` JSON parse path — was silent `return -1`. **Promoted to `logger.warning`** for the parse-fail path; the missing-file path stays silent (intentional, caller routes to INSUFFICIENT_DATA).

## Final lint verification

```
$ python -X utf8 tools/lint_silent_swallow.py <9 v5.1 files>
[lint_silent_swallow] OK — scanned 9 files, zero unannotated silent swallows.
```

## Test verification

```
v5.1 surface (171/171 PASS)
+ cross-cutting (kraken_quant_agent, authority_fusion, DRL gates, constitution): 191
─────────────────────────────────────────────────────────────────────────
Cumulative cross-cutting after audit fixes: 362/362 PASS
```

No test logic changed — only annotations, batch-WARN emissions, and 4 logger.warning promotions. The smoke tests already exercised the silent paths; the lint annotations don't change runtime behavior, just operator visibility.

## What this audit did NOT find

- Zero datetime naive/aware bugs in v5.1 code (P39/P40 lessons learned)
- Zero method-collision bugs (P87)
- Zero archive-module imports (P22)
- Zero P15-shape attr reads (the few cross-module attr reads use defensive `getattr` + sentinel)
- Zero auto-flip-to-live behavior in any of the advisory-tooling phases (Iron Law 7 + 10)

## Standing recommendation per CLAUDE.md P72

The repo's `tools/scanner_baselines/silent_swallow_baseline.json` is the ratchet. After Tier 1 + Pre-6 + Phase 6.0 prep + Phase 10 scaffold + Phase 11 commits land, the baseline should be re-snapshot via `tools/ci_check_invariants.py` so the new annotations register as "expected" rather than as additions. Until then, CI may flag the new noqa-annotated sites as net-zero changes (they're inside fresh files — count delta 0; per-file delta = +20 lines containing `noqa`).

## Audit closure

v5.1 deliverables comply with CLAUDE.md's documented failure-pattern catalogue. Specifically:
- 0 silent swallows unannotated
- 0 datetime mixing
- 0 archive imports
- 0 method collisions
- 0 silent attr-read bugs (P15/P85 patterns)
- All advisory tooling honors Iron Law 7 + 10 (no runtime side-effect)

The 9 closure docs accumulated this session remain accurate; this audit doc is supplementary evidence that the v5.1 code itself meets the CLAUDE.md hardening discipline established by P9-P109.

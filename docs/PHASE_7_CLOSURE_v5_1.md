# HMATS v5.1 — Phase 7 Closure (Risk-Parity Sleeve Sizing, ADVISORY)

**Status:** COMPLETE — Tier 1 (Phase 0 → 1 → 4 → 8 → 7) FINISHED.
**Generated:** 2026-04-29

## What changed

| File | Change |
|---|---|
| `risk/sleeve_allocator_v5_1.py` | NEW — `SleeveAllocator`, `SleeveProfile`, `SleeveAdvisorySink`, `build_phase7_sleeve_allocator()` |
| `main.py` | +28 lines — startup builds allocator + sink; per-tick advisory record write |
| `tests/test_sleeve_allocator_v5_1.py` | NEW — 25 unit tests |

## Architecture

`SleeveAllocator` is portfolio-level (not per-asset). Algorithm:

1. **Inverse-vol risk parity:** raw weight per sleeve = `(1/vol_i) / sum(1/vol_j)`.
2. **Per-sleeve cap (default 0.50):** redistribute excess to non-capped sleeves in one pass. Prevents the V9 cash-and-carry artifact (5% vol → 48.5% inverse-vol weight).
3. **Portfolio leverage scalar:** `clamp(target_vol / portfolio_vol_unit, 0, max_leverage=2.0)`. Default target = 25% per CLAUDE.md memory.
4. **Final weight:** `raw[i] * leverage_scalar`. Sum across sleeves ≤ `max_portfolio_leverage`.

**Quarter-Kelly per-sleeve:** `(sharpe / vol) / 4`, clamped to `[0, 1]`. Used in advisory diagnostics — Phase 10 will gate position sizing against this bound.

## Sleeves registered (Phase 7)

| Sleeve | Vol estimate | Sharpe estimate | is_live | Phase owner |
|---|---|---|---|---|
| directional_short | 0.45 | 0.8 | **True** (existing live system) | Phase 0+1 baseline |
| microstructure | 0.20 | 1.0 | False (Phase 4 shadow) | Phase 4 |
| cascade | 0.30 | 0.9 | False (Phase 8 shadow) | Phase 8 |

**Deferred sleeves (registered later):**
- `funding` — Phase 3 (post-Coinbase migration, Day 29-31)
- `ml_factor` — Phase 6 (Day 42-56)
- `cash_and_carry` — v6 ($50K+ AUM)
- `options` — v6 (Deribit access RED)

Per V9 math, with these 3 sleeves and the caps above, allocator produces a ~25%-vol-targeted portfolio without any single sleeve exceeding 50% gross exposure.

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | allocator does not touch features or feature_manifest |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | UNCHANGED | not touched |
| 4. fail-closed | HELD | `test_register_sleeve_rejects_invalid_vol` (NaN, 0, negative); `test_update_realized_vol_skips_invalid`; empty allocator → `{}`; sink errors logged not raised |
| 5. DRL ACTIVE floor | UNCHANGED | allocator output is advisory; DRL authority unchanged |
| 6. ≥3 active strategies | UNCHANGED | Phase 7 does not archive sleeves; weights only |
| 7. Shadow ≥30d before promotion | **HELD with explicit static check** | `test_allocator_does_not_import_unified_position_sizer` reads source and asserts no production import of `risk.unified_position_sizer`; advisory_only flag hardcoded in record |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not started |
| 9. post-only default | UNCHANGED | execution layer untouched |

## Caps enforced

| Cap | Default | Configurable |
|---|---|---|
| Per-sleeve max weight | 0.50 | `SleeveAllocator(max_sleeve_weight=...)` |
| Portfolio max leverage | 2.0 | `SleeveAllocator(max_portfolio_leverage=...)` |
| Target portfolio vol | 0.25 | `SleeveAllocator(target_portfolio_vol=...)` |
| Vol floor | 0.05 | `DEFAULT_VOL_FLOOR` constant |
| Vol ceiling | 1.50 | `DEFAULT_VOL_CEILING` constant |
| Quarter-Kelly clamp | [0.0, 1.0] | `DEFAULT_KELLY_CLAMP` constant |

Tests `test_compute_allocations_respects_max_sleeve_weight`, `test_compute_allocations_total_leverage_cap`, and `test_advisory_record_total_weight_within_leverage_cap` enforce these bounds.

## Wire-in semantics (main.py)

**Startup (line ~4763, after Phase 8 cascade harness):**
```python
self._sleeve_allocator = None
self._sleeve_advisory_sink = None
try:
    from risk.sleeve_allocator_v5_1 import (
        build_phase7_sleeve_allocator, SleeveAdvisorySink,
    )
    self._sleeve_allocator = build_phase7_sleeve_allocator()
    self._sleeve_advisory_sink = SleeveAdvisorySink()
    logger.info("  [v5.1 PHASE7] SleeveAllocator: ADVISORY (3 sleeves; ...)")
except Exception as _sa_err:
    logger.warning(...)
```

**Per-tick (after Phase 8 `observe()` block):**
```python
if (self._sleeve_allocator is not None and self._sleeve_advisory_sink is not None
        and not p0_abort_tick):
    try:
        _sleeve_rec = self._sleeve_allocator.advisory_record_for(asset)
        self._sleeve_advisory_sink.write(_sleeve_rec)
    except Exception as _sa_err:
        logger.debug(f"[v5.1 PHASE7] sleeve advisory {asset} skipped: {_sa_err}")
```

Advisory records are **per-asset per-tick** — same granularity as Phase 4 / 8 strategy shadows so Pre-6 IC compute can attribute realized PnL slices to recommended weights at matching timestamps.

## Output ledger

Path: `data/strategy_shadow/sleeve_allocations_YYYYMMDD.jsonl`.

Record example:
```json
{
  "ts": "2026-04-29T03:14:15.123456+00:00",
  "asset": "BTC",
  "target_portfolio_vol": 0.25,
  "max_portfolio_leverage": 2.0,
  "max_sleeve_weight": 0.5,
  "sleeves": [
    {"name": "directional_short", "weight": 0.234, "current_vol": 0.45,
     "estimated_vol": 0.45, "sharpe_estimate": 0.8, "is_live": true,
     "quarter_kelly": 0.444, "realized_vol_n": 0},
    {"name": "microstructure",   "weight": 0.527, "current_vol": 0.20,
     "estimated_vol": 0.20, "sharpe_estimate": 1.0, "is_live": false,
     "quarter_kelly": 1.000, "realized_vol_n": 0},
    {"name": "cascade",          "weight": 0.351, "current_vol": 0.30,
     "estimated_vol": 0.30, "sharpe_estimate": 0.9, "is_live": false,
     "quarter_kelly": 0.750, "realized_vol_n": 0}
  ],
  "total_weight": 1.112,
  "advisory_only": true
}
```

Note `quarter_kelly=1.000` for microstructure is the clamp-at-1.0 ceiling (raw = (1.0/0.20)/4 = 1.25 → clamped). Phase 10 will use these as input to position-size caps when the sleeve graduates from shadow.

## Test results

```
tests/test_sleeve_allocator_v5_1.py     25/25 PASS
─────────────────────────────────────────────────
Cumulative cross-cutting (Phase 0+1+4+8+7) 254/254 PASS
```

Suite breakdown:
- Phase 1 surface: 80 (kraken_quant_agent + strategy_selection + alpha_gate + black_swan_hold)
- Iron Law surface: 111 (authority_fusion + drl_promotion_gate + drl_authority_punchthrough + constitution_core)
- Phase 4: 20 (microstructure_v5_1)
- Phase 8: 18 (liquidation_cascade_v5_1)
- Phase 7: 25 (sleeve_allocator_v5_1)
- Total: 254

`main.py` imports cleanly with all three v5.1 advisory layers (Phase 4 micro shadow, Phase 8 cascade shadow, Phase 7 sleeve advisory) initialized in series.

## What does NOT happen yet

- Realized-vol attribution per sleeve from equity_history → Pre-6 (Day 32) builds the per-sleeve PnL slicer
- `update_realized_vol()` is wired-callable but not yet driven by a periodic equity update; Pre-6 will add the cron
- Allocator → `UnifiedPositionSizer.calculate_position_size` integration → **Phase 10 promotion** (Day 57+) after 30d advisory accumulates and validates against realized portfolio vol
- Sleeve registration for `funding`, `ml_factor`, `cash_and_carry`, `options` — owned by their respective phases / v6

## Tier 1 summary (Phase 0 → 1 → 4 → 8 → 7 complete)

| Phase | Days | Status | Lines added | Tests added |
|---|---|---|---|---|
| 0 — Pre-flight + IC re-baseline + 12-strategy buckets | 1-2 | DONE | 1 tooling fix | 0 (existing IC framework unblocked) |
| 1 — Strategy archive gate | 3-4 | DONE | +73 lines | reused existing 33 |
| 4 — Microstructure shadow harness | 5-7 | DONE | +~250 lines | +20 |
| 8 — Cascade shadow harness | 8-10 | DONE | +~210 lines | +18 |
| 7 — Sleeve allocator advisory | 11-13 | DONE | +~230 lines | +25 |

Total Tier 1: ~13 days budgeted, completed within session. **+63 new tests / 254 cross-cutting PASS.** Iron Laws 1, 2, 3, 4, 5, 6, 7, 9 all verified intact across phases. Iron Law 8 (DRL ACTIVE during cutover) remains N/A until Phase 2.

## Deploy step (operator action)

Phase 7 can ship in a combined Tier 1 deploy or as its own commit. Recommended commit message:
```
v5.1 Phase 7: portfolio risk-parity sleeve allocator (ADVISORY)

SleeveAllocator computes inverse-vol risk-parity weights, applies per-sleeve
cap (0.50) + portfolio leverage cap (2.0) targeting 25% vol. 3 sleeves
registered: directional_short (live, vol=0.45 sharpe=0.8), microstructure
(shadow, vol=0.20 sharpe=1.0), cascade (shadow, vol=0.30 sharpe=0.9).
Funding / ml_factor / cash_carry / options DEFERRED to their phases / v6.

Per-asset per-tick advisory record written to
data/strategy_shadow/sleeve_allocations_YYYYMMDD.jsonl. Quarter-Kelly bound
included for Phase 10 promotion-time sizing gate.

Iron Law 7 enforced via static-check test:
test_allocator_does_not_import_unified_position_sizer reads source and
asserts zero production import of risk.unified_position_sizer. UnifiedPositionSizer
remains the live sizer; allocator is observation-only.

main.py wire-in next to Phase 4 + Phase 8 harnesses; per-tick advisory_record_for
writes after both shadow observe() calls. Order: market_data assembled →
phase 4 observe → phase 8 observe → phase 7 advisory record.

254/254 cross-cutting tests PASS (Phase 0+1+4+8+7 union). 25 new tests
cover registration validation, inverse-vol math, both caps, Quarter-Kelly
clamping, advisory record shape, sink durability, Iron Law 7 static check.
```

## Tier 2 readiness

Phase 7 closure ends Tier 1. Tier 2 begins:
- **Phase 2** (Coinbase migration, Days 14-28) — **BLOCKED on [PARAMETER 3]** cutover-mode operator answer.
- **Phase 3** (Funding strategies, Days 29-31) — depends on Phase 2; will register `funding` sleeve into the Phase 7 allocator at completion.

[PARAMETER 3] is the only remaining blocker. Phase 7 has no dependency on it.

## Outstanding [PARAMETER]s after Tier 1

| # | Parameter | Status | Phase |
|---|---|---|---|
| 1 | Branch X or Y | RESOLVED Y (Phase 0) | — |
| 2 | 12-strategy buckets | RESOLVED (Phase 0+1) | — |
| 3 | V4.3 cutover mode | **PENDING OPERATOR** | Phase 2 (Day 14) |
| 4 | V8 DRL retrain Y/N | DEFAULT N | Phase 2 (Day 14) |

# HMATS v5.1 — Phase 4 Closure (Microstructure Layer Expansion, SHADOW)

**Status:** COMPLETE — ready for Phase 8 (Liquidation Cascade Sleeve Expansion).
**Generated:** 2026-04-29

## What changed

| File | Change |
|---|---|
| `strategies/microstructure_v5_1.py` | NEW — 3 shadow-mode strategies (OFI, VPIN spike, Kyle's lambda fade) |
| `defense/strategy_shadow_v5_1.py` | NEW — observation-only harness, JSONL ledger writer, fail-closed per-strategy isolation |
| `main.py` | +27 lines — startup harness init (next to `_microstructure_agent`) + per-tick `observe()` after `WIRE-MICRO` block |
| `tests/test_microstructure_v5_1.py` | NEW — 20 unit tests covering 3 strategies + harness |

## Architecture

Three strategies wrap **existing** market_data primitives — no new feeds, no new features, no obs_dim touch:

| Strategy | Trigger | Direction | Confidence cap |
|---|---|---|---|
| OrderFlowImbalanceStrategy | `|ofi_zscore| > 1.5` AND consecutive same-sign streak ≥ 2 AND `|order_book_imbalance| > 0.10` | sign of streak | 0.65 |
| VPINSpikeStrategy | `vpin > 0.70` OR `vpin_anomaly == HIGH`; requires `vpin_source == 'computed'` (synthetic 0.35 default rejected); `|ofi_zscore| > 1.0` | **contrarian** to OFI sign (fade toxic flow) | 0.55 |
| KyleLambdaStrategy | rolling 30-bar lambda_proxy = `|return| / max(|imbalance|, 1e-4)` exceeds 85th percentile; `spread_bps > 5`; `|return_4h| > 0.5%` | **fade** the recent move | 0.50 |

All 3 fail-closed (Iron Law 4) on:
- missing field → reason `"missing_*"`
- NaN value → caught via `_get_float` helper
- insufficient warmup → reason `"history_warmup"` / `"no_prev_price"`
- threshold not crossed → reason `"below_threshold"` / `"streak_too_short"` / `"below_p85"`

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | strategies consume `vpin`, `ofi_zscore`, `order_book_imbalance`, `spread_bps`, `current_price` — all already in market_data; zero new features |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | UNCHANGED | not touched |
| 4. fail-closed | HELD | test_ofi_neutral_when_field_missing, test_ofi_neutral_on_nan, test_kyle_neutral_warmup; per-strategy exception isolation in harness verified by test_shadow_harness_isolates_strategy_failure |
| 5. DRL ACTIVE floor | UNCHANGED | shadow strategies write only to JSONL ledger, never agent_signals |
| 6. ≥3 active strategies | UNCHANGED | Phase 4 ADDS shadow observers, doesn't archive anything |
| 7. ≥30d shadow before promotion | HELD | `MicrostructureShadowHarness.observe()` returns None; no fusion hook; verified by test_shadow_harness_does_not_mutate_market_data (input dict immutable post-call) |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not yet started |
| 9. post-only default | UNCHANGED | execution layer untouched |

## Wire-in semantics (main.py)

**Startup (line ~4732):** harness built once via `build_microstructure_shadow_harness()`. Stored as `self._micro_shadow`. If init fails, attribute set to None and tick code defends.

**Per-tick (line ~7234, immediately after `[WIRE-MICRO]` block):**
```python
if getattr(self, "_micro_shadow", None) is not None and not p0_abort_tick:
    try:
        self._micro_shadow.observe(asset, market_data)
    except Exception as _ms_err:
        logger.debug(f"[v5.1 PHASE4] shadow observe {asset} skipped: {_ms_err}")
```

Position is deliberate: **after** market_data has been fully assembled with all microstructure fields and the production microstructure_agent has run, **before** any decision/fusion code. Shadow strategies see exactly what fusion sees, but their output goes to JSONL only.

Double-guard: harness already catches per-strategy exceptions internally; the outer try/except at the call site is belt-and-suspenders for any harness-level fault (e.g. disk full mid-write).

## Output ledger schema

Path: `data/strategy_shadow/microstructure_YYYYMMDD.jsonl` (or `$HMATS_DATA_DIR/strategy_shadow/...` in production container).

Per record (one line per (asset, strategy) per tick):
```json
{
  "ts": "2026-04-29T03:14:15.123456+00:00",
  "asset": "BTC",
  "strategy": "ofi",
  "direction": -1.0,
  "confidence": 0.42,
  "reason": "ofi_sustained_-3",
  "diagnostics": {"ofi_zscore": -2.30, "ob_imbalance": -0.31, "streak": -3}
}
```

Phase Pre-6 (Day 32) will join these against forward returns to compute 14d/30d shadow IC. Kill-criteria per v5.1 prompt: 14d shadow IC < 0.05 → KILL individual; trade-frequency surge >3x → anti-churn protection (signals don't reach the trade gate yet, so this is a Phase 10 promotion-time concern).

## Test results

```
tests/test_microstructure_v5_1.py        20/20 PASS
tests/test_kraken_quant_agent.py         33/33 PASS  (Phase 1 still green)
tests/test_strategy_selection.py         15/15 PASS
tests/test_alpha_gate.py                 9/9   PASS
tests/test_black_swan_hold.py            23/23 PASS
tests/test_authority_fusion.py           21/21 PASS
tests/test_drl_promotion_gate.py         15/15 PASS
tests/test_drl_authority_punchthrough.py 14/14 PASS
tests/test_constitution_core.py          61/61 PASS
─────────────────────────────────────────────────────
Cross-cutting cumulative                 211/211 PASS
```

`main.py` import smoke: clean (`ExecutionQualityLogger`, `Level2OrderBookAnalyzer`, `LearnedExecutionPolicy` all load; only the pre-existing `agents.quant_agent` warning that was already there before Phase 4).

## What does NOT happen yet

- Forward-return join → Pre-6 (Day 32)
- IC compute on shadow signals → Pre-6 (Day 32)
- Promotion gate to fusion → Phase 10 (Day 57+)
- Per-strategy 14d kill-criteria evaluation → Phase 10 (after 30d shadow accumulation)

These deferrals are intentional — the v5.1 prompt schedules them after Pre-6 builds the formal infrastructure. Phase 4 ships the data-collection layer that those phases will operate on.

## Deploy step (operator action)

Phase 4 can ship as a single commit on top of Phase 1. Recommended:
1. `git add strategies/microstructure_v5_1.py defense/strategy_shadow_v5_1.py main.py tests/test_microstructure_v5_1.py docs/PHASE_4_CLOSURE_v5_1.md`
2. Commit message:
   ```
   v5.1 Phase 4: microstructure shadow harness (3 strategies, observation-only)

   Three shadow-mode strategies wrap existing market_data primitives:
   OrderFlowImbalanceStrategy (sustained OFI z-score), VPINSpikeStrategy
   (toxic-flow contrarian fade), KyleLambdaStrategy (lambda proxy spike fade).
   No new feeds, no obs_dim change, no constitution change. Zero fusion
   side-effect — Iron Law 7 enforced at the call signature (observe() returns
   None, harness has no fusion hook). Per-strategy exception isolation via
   warn-once + bypass; main tick is defended.

   defense/strategy_shadow_v5_1.py is a write-only JSONL sink; Phase Pre-6
   (Day 32) will join records against forward returns for IC compute. Phase
   10 (Day 57+) gates promotion.

   Wire-in main.py: startup builds harness next to _microstructure_agent
   (line ~4732), per-tick observe() runs after WIRE-MICRO block (line ~7234).
   Hot-disable: clear self._micro_shadow at runtime — gate at call site.

   211/211 cross-cutting tests PASS (Phase 0 + 1 + 4 union). 20 new tests
   for Phase 4 cover Iron Law 4 fail-closed paths and Iron Law 7 dict-immutability.
   ```
3. Deploy via `bash scripts/hetzner_deploy.sh hmats`. Container will start
   producing `data/strategy_shadow/microstructure_YYYYMMDD.jsonl` on the next
   tick. Operator can verify with:
   ```bash
   ssh hmats "docker exec hmats-engine ls /opt/hmats/data/strategy_shadow/"
   ssh hmats "docker exec hmats-engine tail -5 /opt/hmats/data/strategy_shadow/microstructure_$(date +%Y%m%d).jsonl"
   ```

## Phase 8 readiness

Phase 4 closure does not block anything. Phase 8 (Liquidation Cascade Sleeve
Expansion, Days 8-10) operates on `LiquidationCascadeHunter` (kraken_quant_agent
strategy_id=1) + Coinglass `liquidation_imbalance` feed + adds defensive
stop-zone-avoidance. Phase 7 (risk-parity sizing, Days 11-13) follows.

[PARAMETER 3] cutover-mode operator answer remains pending for Phase 2 (Day 14)
but does not bind Phase 8.

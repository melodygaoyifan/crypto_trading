# HMATS v5.1 — Phase 8 Closure (Liquidation Cascade Sleeve Expansion, SHADOW)

**Status:** COMPLETE — ready for Phase 7 (Risk-Parity Sleeve Sizing).
**Generated:** 2026-04-29

## What changed

| File | Change |
|---|---|
| `strategies/liquidation_cascade_v5_1.py` | NEW — 2 cascade shadow strategies |
| `defense/strategy_shadow_v5_1.py` | +21 lines — `build_cascade_shadow_harness()` (prefix='cascade') |
| `main.py` | +21 lines — startup builds `_cascade_shadow`; per-tick `observe()` after Phase 4 |
| `tests/test_liquidation_cascade_v5_1.py` | NEW — 18 unit tests |

## Architecture

Two strategies wrap **existing** Coinglass-derived market_data fields:

| Strategy | Trigger | Direction | Confidence cap |
|---|---|---|---|
| CascadeAnticipationStrategy | `\|liq_imbalance\| ≥ 0.6` AND `total_liq_24h ≥ $100M` AND price_change_4h in cascade window (between trigger_drop=−1.5% and cap_floor=−10%); symmetric for short squeeze | sign-of-cascade (SHORT for long-liq, LONG for short-squeeze) | 0.60 |
| StopHuntDefenseStrategy | composite_risk = avg(cascade_score, toxic_score, spread_score, move_score). Always emits; quiet=baseline 25bps offset, elevated=up to 100bps | **0.0 (non-directional)** — advisory `recommended_stop_offset_bps` in diagnostics | 0.85 |

**Asymmetry asymmetry note:** Per Sep 2025 cited research ($1.6B / $1.7B daily liquidations long-side), the long-liq branch is the higher-information asymmetry. Symmetric short-squeeze branch handled for completeness but expected to fire less.

**Capitulation cap:** when `|price_change_4h| > 10%`, the cascade is considered exhausted → NEUTRAL with reason `likely_capitulation`. Avoids re-entering after the move is done.

**Stop-hunt advisory:** Phase 8 ships StopHuntDefense as **observation-only**. Phase 10 promotion gate will wire `recommended_stop_offset_bps` into `execution/execution_manager.py:place_stop_loss` (line 2094) via a callback if shadow validates.

## Iron Law verification

| Law | Status | Evidence |
|---|---|---|
| 1. obs_dim=126 | UNCHANGED | strategies consume `liquidation_imbalance`, `total_liquidations_24h`, `price_change_4h_pct`, `price_change_1h_pct`, `spread_bps`, `vpin_anomaly` — all already in market_data |
| 2. constitution.py | UNCHANGED | not touched |
| 3. training/ | UNCHANGED | not touched |
| 4. fail-closed | HELD | `test_cascade_neutral_when_fields_missing`, `test_cascade_neutral_on_nan`, `test_stophunt_neutral_when_spread_missing`; per-strategy exception isolation reused from Phase 4 harness |
| 5. DRL ACTIVE floor | UNCHANGED | shadow strategies write only to JSONL; no agent_signals touch |
| 6. ≥3 active strategies | UNCHANGED | Phase 8 ADDS observers, doesn't archive |
| 7. ≥30d shadow before promotion | HELD | `test_cascade_harness_does_not_mutate_market_data` — input dict immutable post-call; `place_stop_loss()` untouched |
| 8. DRL ACTIVE during cutover | N/A | Phase 2 not started |
| 9. post-only default | UNCHANGED | execution layer untouched |

## Wire-in semantics (main.py)

**Startup (line ~4747, immediately after Phase 4 harness init):**
```python
self._cascade_shadow = None
try:
    from defense.strategy_shadow_v5_1 import build_cascade_shadow_harness
    self._cascade_shadow = build_cascade_shadow_harness()
    logger.info("  [v5.1 PHASE8] CascadeShadowHarness: ACTIVE (2 strategies, observation-only)")
except Exception as _cs_err:
    logger.warning(f"  [v5.1 PHASE8] CascadeShadowHarness init failed: ...")
```

**Per-tick (immediately after Phase 4 `observe()` block):**
```python
if getattr(self, "_cascade_shadow", None) is not None and not p0_abort_tick:
    try:
        self._cascade_shadow.observe(asset, market_data)
    except Exception as _cs_err:
        logger.debug(f"[v5.1 PHASE8] cascade observe {asset} skipped: {_cs_err}")
```

Position is deliberate: **after** Phase 4 microstructure observation (so both shadow ledgers see the same fully-assembled market_data), **before** any decision/fusion code. Both Phase 4 and Phase 8 strategies see identical inputs.

## Output ledger

Path: `data/strategy_shadow/cascade_YYYYMMDD.jsonl` (separate from `microstructure_*.jsonl`).

Cascade-anticipation record example:
```json
{
  "ts": "2026-04-29T03:14:15.123456+00:00",
  "asset": "BTC",
  "strategy": "cascade_anticipation",
  "direction": -1.0,
  "confidence": 0.48,
  "reason": "long_liq_cascade_anticipation(imb=+0.78,ret4h=-3.20%)",
  "diagnostics": {
    "liquidation_imbalance": 0.78,
    "total_liquidations_24h": 520000000.0,
    "price_change_4h_pct": -0.032,
    "vpin_anomaly": "HIGH",
    "side": "long_liq"
  }
}
```

Stop-hunt-defense record example:
```json
{
  "ts": "2026-04-29T03:14:15.123456+00:00",
  "asset": "BTC",
  "strategy": "stop_hunt_defense",
  "direction": 0.0,
  "confidence": 0.72,
  "reason": "elevated_stop_hunt_risk(composite=0.78)",
  "diagnostics": {
    "composite_risk": 0.78,
    "recommended_stop_offset_bps": 83.5,
    "cascade_score": 0.85, "toxic_score": 1.0,
    "spread_score": 0.50, "move_score": 0.78,
    "advisory_only": true
  }
}
```

Pre-6 (Day 32) will join cascade direction signals against forward returns. StopHunt advisory is non-directional — Pre-6 will instead measure whether stops placed at `recommended_offset` would have survived clusters that hit the baseline-offset stops, using historical fill_quality + stop-execution data.

## Test results

```
tests/test_liquidation_cascade_v5_1.py    18/18 PASS
─────────────────────────────────────────────
Cumulative cross-cutting (Phase 0+1+4+8)  229/229 PASS
```

Suite breakdown:
- kraken_quant_agent 33 + strategy_selection 15 + alpha_gate 9 + black_swan_hold 23 = 80 (Phase 1 surface)
- authority_fusion 21 + drl_promotion_gate 15 + drl_authority_punchthrough 14 + constitution_core 61 = 111 (cross-cutting Iron Law surface)
- microstructure_v5_1 20 (Phase 4)
- liquidation_cascade_v5_1 18 (Phase 8)
- Total: 229

`main.py` imports cleanly with both shadow harnesses initialized in series.

## What does NOT happen yet

- Cascade-anticipation forward-return IC compute → Pre-6 (Day 32)
- StopHunt advisory → real `place_stop_loss` integration → Phase 10 promotion (Day 57+) after 30d shadow validates the recommendations
- Cascade kill-criteria evaluation (FP > 50% → revisit; IC < 0.04 over 30d → KILL) → Phase 10
- Per-price-level cluster heatmap → DEFERRED. Coinglass `liquidation_map` endpoint not currently subscribed; current implementation uses 24h aggregates as proxy. v6 candidate.

## Deploy step (operator action)

Phase 8 can ship on top of Phase 4 in a single combined commit, OR independently. Recommended commit message:
```
v5.1 Phase 8: liquidation cascade shadow expansion (2 strategies, observation-only)

CascadeAnticipationStrategy — short-bias asymmetry, anticipates next leg of
long-liquidation cascade when imb >= 0.6 + price_change_4h in cascade window
(-1.5% to -10%, capitulation cap). Symmetric short-squeeze branch.

StopHuntDefenseStrategy — non-directional advisory. Composite risk over
cascade/toxic/spread/move components → recommended_stop_offset_bps. Phase 10
gate will wire into place_stop_loss after 30d shadow validates.

Reuses Phase 4 MicrostructureShadowHarness with prefix='cascade'. Output
ledger: data/strategy_shadow/cascade_YYYYMMDD.jsonl (segregated from Phase 4
microstructure ledger). main.py wire-in next to Phase 4 harness — both
observe per-tick after market_data is assembled, neither touches fusion.

229/229 cross-cutting tests pass (Phase 0 + 1 + 4 + 8 union). 18 new tests
cover Iron Law 4 fail-closed paths, capitulation gate, vpin-confirm gate,
synthetic-vpin rejection, ledger segregation, dict-immutability.

Per-price-level cluster heatmap (v5.1 prompt's nearest_long.distance_pct
shape) is DEFERRED — Coinglass liquidation_map endpoint not subscribed.
Current implementation uses 24h aggregates as proxy. v6 candidate.
```

Verify on Hetzner after deploy:
```bash
ssh hmats "docker exec hmats-engine ls /opt/hmats/data/strategy_shadow/"
ssh hmats "docker exec hmats-engine tail -3 /opt/hmats/data/strategy_shadow/cascade_$(date +%Y%m%d).jsonl"
```

## Phase 7 readiness

Phase 8 closure does not block Phase 7. Phase 7 (risk-parity sleeve sizing,
Days 11-13) builds the sleeve allocator. The 3 v5.1 sleeves currently active
(directional_short / microstructure-shadow / cascade-shadow) plus the upcoming
funding sleeve (Phase 3, post-Coinbase) will feed it. Phase 7 ships with 1
real sleeve (directional_short, the existing live system) + 2 shadow sleeves
counted only in observation; promotion to allocator participation happens at
Phase 10 (Day 57+).

[PARAMETER 3] cutover-mode operator answer remains pending for Phase 2 (Day 14)
but does not bind Phase 7.

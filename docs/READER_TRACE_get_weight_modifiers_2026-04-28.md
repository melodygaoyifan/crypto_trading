# Reader Trace: `strategy_aging.get_weight_modifiers()`

**Date**: 2026-04-28
**Mode**: READ-ONLY trace per v3 prompt item 1.2 (GO #4).
**Outcome**: **DEAD LOOP confirmed.** Priority inversion warranted — see Recommendation.

---

## Method

Found every consumer of `strategy_aging.get_weight_modifiers()` across the live tree (excluding `__pycache__`, `archive/`, `tests/`, and `analytics/strategy_aging.py` itself which is the producer).

```bash
grep -rn "get_weight_modifiers\|strategy_aging.get_weight\|aging.get_weight" \
  --include="*.py" | filter
```

**Result**: Exactly **one consumer site**.

---

## The single consumer

**Location**: `core/execution_service.py:2469`

**Context**: Inside the `record_outcome()` block at line 2440-2483. Fires after every position close (BRANCH A full-exit path).

**Code**:
```python
_c12_mods = ctx.strategy_aging.get_weight_modifiers()
for _c12_sname, _c12_wmod in _c12_mods.items():
    if _c12_wmod < 0.7:
        logger.critical(
            f"[STRATEGY_AGING] {_c12_sname} degraded: "
            f"weight modifier={_c12_wmod:.2f}. "
            f"Consider reviewing strategy parameters."
        )
    elif _c12_wmod > 1.1:
        logger.info(
            f"[STRATEGY_AGING] {_c12_sname} strong: "
            f"weight modifier={_c12_wmod:.2f}"
        )
```

**Classification**: **LOGGED_ONLY**

The return value `_c12_mods` is:
1. Iterated to extract per-strategy modifiers
2. Compared against thresholds (0.7 / 1.1)
3. Used SOLELY to fire `logger.critical` or `logger.info` messages

It is **NEVER**:
- Applied to position sizing
- Applied to strategy selection in best-of-N
- Multiplied into any weight elsewhere in fusion / authority / execution
- Stored back to any state file
- Read by any downstream module

---

## Per-caller classification (v3 prompt schema)

| Caller | Type | Evidence |
|---|---|---|
| `core/execution_service.py:2469` | **LOGGED_ONLY** | Return value used only for log emission (lines 2472, 2478). No decision branch reads it. |

| Other callers | None found |

---

## What this means for the system

The full strategy_aging chain looks like this:

```
record_signal()  ──[per tick]──▶  _strategy_signals[name].append(...)
                                              │
                                              │  (in-memory buffer)
                                              ▼
record_outcome() ──[per close]──▶  _evaluate_strategy(name)
                                              │
                                              │  (computes modifier 0.5–1.2)
                                              ▼
get_weight_modifiers() ─────────▶  _modifiers dict
                                              │
                                              │  (returned to caller)
                                              ▼
        execution_service.py:2469 ─────────▶  logger.critical / logger.info
                                              │
                                              ▼
                                          ☠ DEAD END ☠
                                  (no decision consumes the value)
```

Every step in the producer side is wired correctly:
- ✓ `record_signal()` IS called per tick (`main.py:8280`)
- ✓ `record_outcome()` IS called on close (`execution_service.py:2446`, P129 promoted its swallow to WARNING)
- ✓ `_evaluate_strategy()` runs every `evaluation_period_hours` (default 24h)
- ✓ `get_weight_modifiers()` returns a populated dict

**But the dict goes nowhere.** Strategy weights in best-of-N selection (`data_mgmt/market_data_pipeline.py` REGIME_STRATEGY_FIT) are static config values. The 12-strategy kraken_quant matrix uses regime-fit thresholds that are also static. Neither layer reads `_c12_mods`.

This means:
- Even if every record_outcome call succeeds with perfect data
- Even if the IC framework computes accurate per-strategy modifiers
- Even if weights of 0.5 (max decay) are computed for the worst-performing strategy

**Nothing changes in production.** The strategy that lost $300 last week gets selected with the same weight next tick.

---

## Implication for v1's recommendations

v1's H3 ("system lacks self-learning") was rated **PARTIAL**. v1 said the gap was "wire one missing callback (`record_outcome`)". v2 caught that the callback IS already wired. v3's reader trace shows the callback was wiring up to a **dead end** all along.

The actual self-learning gap is **the read side, not the write side**:

| What v1 thought was missing | What's actually missing |
|---|---|
| `record_outcome()` not called | A consumer that APPLIES `get_weight_modifiers()` output to decisions |
| → "wire 1 callback, ~3 days" | → wire weight application into best-of-N + kraken_quant selection, ~3-5 days |

Per v3 prompt's decision logic for this trace:

> 全部 LOGGED_ONLY / DEAD / ABSENT → strategy_aging IS dead-loop, 任何 record_outcome 上游修复都是浪费. 优先级反转: 修 reader 才是真 P0.

---

## Priority inversion recommendation

**File for v4 P0**: Wire `get_weight_modifiers()` output into:

1. **Best-of-N strategy selection** (`data_mgmt/market_data_pipeline.py` strategy scoring loop). Multiply each strategy's `strength` by its weight modifier before selecting the best.

2. **kraken_quant 12-strategy weight** (`agents/kraken_quant_agent.py` strategy fire decision). If a strategy's modifier < 0.7, gate it out for the tick.

3. **Position sizing** (optional, lower priority): if downselected to one strategy, scale position by modifier (modifier=1.2 → 1.2× notional, etc.).

**Effort estimate**: 3–5 person-days. Risk: medium (touches selection logic in two places). Pre-mortem required before implementation:
- What if a healthy strategy temporarily underperforms and gets gated out → loses recovery alpha?
- What if `_evaluate_strategy()` doesn't have enough samples (`min_signals_for_assessment=20`) → modifier defaults to 1.0, no effect, dead-loop persists
- Need 30+ trades per strategy for confidence; with current 17 trades / 30 days, system needs months of data first

**This is NOT in v3 Track A scope.** Filed for v4 audit.

---

## What v3 Track A items are now ALSO affected

- **P0-1 (kraken_quant 0% firing)** — Even if we diagnose the firing rate and unlock 12 strategies, the aging system can't learn from their outcomes until the reader is wired. Doesn't block P0-1 directly but caps its long-term ROI.
- **P0-2 (window extension to 6h)** — v2 already NO-GO'd this. Reader trace confirms NO-GO is correct: extending the window captures more outcomes, but they fed into a dead-end logger. Window-extension fix without reader-fix changes nothing.

---

## What v3 Track A items are NOT affected

- **P0-3 (regime_at_entry)** — orthogonal observability fix, already shipped (P129).
- **P0-4 (RSI sign-flip)** — orthogonal signal-quality fix, still on day-2 schedule.
- **P1-5 / P1-6 / P1-7** — observability + analysis, orthogonal.

---

## Calibrated confidence

- **~95%** that the trace is exhaustive (single grep across live tree, exact match).
- **~85%** that there's no other consumer hidden in dynamic dispatch / lambda / config-driven code (didn't trace `eval()` or string-keyed dispatch tables; manual review of `signals/` and `data_mgmt/` for any "aging" or "modifier" key search came up empty).
- **~99%** that the existing call site is LOGGED_ONLY (read the source directly).

---

*Generated 2026-04-28. READ-ONLY trace per v3 prompt item 1.2.*
*Filed: v4 P0 candidate — wire `get_weight_modifiers()` consumer into best-of-N + kraken_quant selection.*

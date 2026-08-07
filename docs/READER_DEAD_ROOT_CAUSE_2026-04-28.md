# Reader DEAD Root Cause — `strategy_aging.get_weight_modifiers()`

**Date**: 2026-04-28
**Mode**: READ-ONLY git archaeology + cross-reference scan.
**Triggered by**: v3.5 Phase 1.1.

---

## Question

The reader trace report (`READER_TRACE_get_weight_modifiers_2026-04-28.md`) classified the single call site as LOGGED_ONLY. v3.5 asks: which of these patterns?

- **Type 1 — Never wired**: producer + LOGGED_ONLY caller exist; real consumer was never built (unfinished design).
- **Type 2 — Wired then unwired**: a real consumer once existed, was removed by some refactor (revert reason matters).
- **Type 3 — Logged-only by design**: the LOGGED_ONLY caller IS the intended end consumer (telemetry, not control loop).

---

## Evidence

### Git history of the symbol

```
$ git log --all --oneline -S "get_weight_modifiers" -- '*.py'
8753484 Wire Exit DRL Discrete-SAC to EXIT_ONLY for BTC/ETH/SOL via accelerated path
7b70907 Initial commit: HMATS v6.8.0 — Hierarchical Multi-Agent Trading System

$ git log --all --oneline -S "weight_modifier" -- '*.py'
8753484 Wire Exit DRL Discrete-SAC to EXIT_ONLY for BTC/ETH/SOL via accelerated path
7b70907 Initial commit: HMATS v6.8.0 — Hierarchical Multi-Agent Trading System
```

Only two commits ever touched the symbol:
1. The **initial commit** (`7b70907`) where the producer + LOGGED_ONLY caller were both introduced together.
2. Commit `8753484` (Exit DRL wiring, 2026-04-24) — incidental, only touched the surrounding scalar-drift fix at execution_service.py:2460–2468 (`_last_aging_check` runner-attribute persistence). Did NOT modify the call site or its consumers.

### Cross-reference scan

```
core/execution_context.py:242         ctx.strategy_aging = getattr(runner, '_strategy_aging', None)
core/execution_service.py:2446        ctx.strategy_aging.record_outcome(...)         # P0-2 wiring
core/execution_service.py:2469        _c12_mods = ctx.strategy_aging.get_weight_modifiers()   # LOGGED_ONLY consumer
main.py:3157                          self._strategy_aging = None
main.py:3161                          self._strategy_aging = get_aging_manager()     # construction
main.py:8278/8280                     self._strategy_aging.record_signal(...)        # producer hot-path
signals/adaptive_weight_v521.py:23    docstring mention only
signals/adaptive_weight_v521.py:734   def get_agent_weight_modifier()               # SEPARATE module, own method
```

`signals/adaptive_weight_v521.py:734` defines `get_agent_weight_modifier()` (singular) on its own `AdaptiveWeightManagerV521` class — **not** a consumer of `analytics/strategy_aging.get_weight_modifiers()`. It's a parallel-track aging system that may itself be DEAD (filed for separate audit). The line 23 docstring is forward-looking ("增强 ... strategy_aging.py" = "to enhance alongside strategy_aging.py"), not a use site.

---

## Verdict

**Type 3 — Logged-only by design (since initial commit).**

There is **no git evidence** of a previously-existing consumer that was removed. The LOGGED_ONLY pattern at `execution_service.py:2469` has been in place since the initial commit. The call site emits `logger.critical` for degraded strategies (modifier < 0.7) and `logger.info` for strong strategies (> 1.1) — pure telemetry, not control.

But: **"by design" likely means "by partial design"**. The producer side of strategy_aging is sophisticated (60-day rolling window, direction accuracy, PnL contribution, signal quality, weight modifier 0.5–1.2, daily re-evaluation). Building all that infrastructure to fire log messages doesn't pass smell test. More plausible interpretation:

> Initial commit ALSO never finished the consumer wiring. The system was shipped with `record_signal` + `record_outcome` + `get_weight_modifiers` all implemented, with the intent to add a real consumer before going live. That second step was never taken. The LOGGED_ONLY caller is a placeholder, not the design endpoint.

This is a **Type 1-via-Type 3 hybrid**: technically logged-only by current design, but the design was incomplete from day one rather than deliberately telemetry-only.

---

## Implications for path selection

- **NOT Type 2** → no historical revert to study; no operator who already-rejected a wiring attempt.
- Safe to wire a real consumer; we are not undoing someone's intentional decision.
- BUT: also not safe to assume the producer side is fully production-ready — its 60-day rolling window, 1-hour matching, `min_signals_for_assessment=20` thresholds were chosen WITHOUT live consumption feedback. Any wiring should treat the modifier values with skepticism for the first N weeks.

---

## Filed observations (NOT actions in this prompt)

1. **`signals/adaptive_weight_v521.py`** is a parallel-track aging system. May also be a dead loop. Filed for separate audit pass.
2. **`min_signals_for_assessment=20`** with current 17 trades / 30 days means weight modifiers stay at 1.0 for foreseeable future even WITH a wired consumer. Sample-size threshold may need tuning before consumer-wiring delivers any real signal change.

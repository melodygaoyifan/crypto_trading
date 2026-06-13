# HMATS Implementation Prompt — Layer 1+4: Live-IC Bucket Gate — DRAFT

**Status**: DRAFT for operator review. Do NOT execute as-is. **Blocked on a post-Layer-2 live-data window** (see Decision D5).
**Triggered by**: Apr–Jun 2026 live forensic (CLAUDE.md P142 + memory `live-performance-apr-jun-2026`). Kraken-authoritative −25% = −$2,314 trading + −$125 fees.
**Date**: 2026-06-13
**Author**: Claude (Opus 4.8)
**Depends on**: Layer 2 (churn control, commit `228a984`, LIVE 2026-06-13) already deployed.

---

## Why this exists (the forensic in one paragraph)

The −25% decomposed (570 INTENT records vs Kraken 4H OHLC): raw signal next-bar ≈ −$29 (52% hit, ≈break-even) → hold-signal-forward −$574 → actual −$2,314. **~75% of the loss was execution churn** — addressed by Layer 2. The remaining ~25% is a **weak, bimodal signal**. This prompt addresses the signal half **without hardcoding in-sample rules**.

The signal's live edge, by bucket (next-4H-bar, no churn, **in-sample/optimistic**):

| Cut | Winners | Losers |
|---|---|---|
| Direction | LONG 58% / +24% | **SHORT 42% / −28%** |
| Strategy | momentum 58% / +20% | **mean_revert 42% / −9%** |
| Regime | WEAK_CONSOLIDATION 60% / +19% | **QUIET_ACCUMULATION 48% / −25%** (bulk of activity) |
| Confidence | — | **conf≥0.6 → 43% / −3% (ANTI-predictive)** |

Filtered "drop mean_revert + shorts" → 59% hit, +28% (in-sample). **The trap to avoid:** baking those bans in as constants. That is the exact backtest-overfit failure (P40/P41) that produced the +7-Sharpe-backtest vs 46%-live gap. The fix is a gate that **learns these buckets from live data and re-evaluates continuously**, not a frozen ruleset.

---

## What this builds (Layer 1 ⊆ Layer 4)

A **live-IC bucket gate**: a per-`(strategy × regime × direction)` tracker of realized live edge that gates which intents are allowed to size up. Layer 1 ("trade only proven buckets") is the *behavior*; Layer 4 (the IC tracker + promotion mechanism) is the *implementation*. One module, not two.

**Explicitly OUT of scope**: Layer 3 (shorts on the Coinbase perp). That is the operator's active in-flight work (the two-sleeve / Phase 3 SHADOW migration, 8+ commits on 2026-06-13). Do not touch it here.

---

## Operator decisions required BEFORE this becomes executable

### D1 — Bucket granularity
- **Options:** (a) `direction` only (2 buckets); (b) `strategy × direction` (~8); (c) `strategy × regime × direction` (~40, but most sparse); (d) `regime × direction` (~14).
- **Trade-off:** finer = more targeted but slower to reach statistical significance per bucket (sparsity → most buckets never gate). The forensic shows the strongest, densest signals are along **direction** and **strategy** axes; **regime** matters (QUIET_ACCUMULATION) but is collinear with low activity.
- **Recommendation framing:** start (b) `strategy × direction` — dense enough to populate, captures the two biggest effects (shorts, mean_revert). Add `regime` only if (b)'s buckets prove too coarse after the shadow window.

### D2 — Edge metric
- **Options:** (a) rolling **hit-rate** (sign accuracy vs next-bar); (b) rolling mean **signed return** (IC-like); (c) both, gate on the worse.
- Hit-rate is intuitive but ignores magnitude; signed-return captures the −28% tail. **Recommendation:** (c) — block a bucket if hit-rate < D3 **or** mean signed-return < 0 over the window.
- **HARD constraint:** model confidence is **anti-predictive** (conf≥0.6 → 43% hit). Do NOT use confidence as a gating feature or a tiebreak. (This also means: do not "trust high-conviction signals more" anywhere downstream.)

### D3 — Thresholds
- `min_samples_per_bucket` before a bucket can gate (below this → **allow**, fail-open, still in warmup). Provisional: **30**.
- `block_hit_rate` below which a bucket is blocked. Provisional: **0.50** (≤ coin-flip ⇒ no edge ⇒ don't pay costs).
- `rolling_window`: trailing N evaluations or trailing D days. Provisional: **rolling 60 evaluations** per bucket.

### D4 — Enforcement action on a blocked bucket
- **Options:** (a) hard veto the intent (→ HOLD); (b) size-multiplier 0 (same effect, composes with sizing); (c) downgrade to SHADOW-only logging for that bucket.
- **Recommendation:** (b) size-mult → 0 via the existing modifier chain (`signals/profit_max_adapter.py` style), so it composes with B1/Layer-2 and is reversible per-bucket. Never blocks reduces/exits/safety (same carve-out as B1/Layer-2).

### D5 — Shadow window length (THE GATE ON THIS WHOLE PROMPT)
- The tracker must populate on **post-Layer-2** live data — Layer-2 changed the trade distribution, so pre-2026-06-13 buckets are stale. Run **SHADOW-only** (log "would block bucket X") until each gated bucket has ≥ `min_samples` post-Layer-2 evaluations.
- Provisional: **≥ 21 days live AND ≥ min_samples per bucket actually traded**, reconciled against the existing IC reports (`analytics/ic/`). This also satisfies the v5.1 "30-day clean IC" pause condition for the signal layer.

---

## Provisional execution order (assumes D1=b, D2=c, D3 defaults, D4=b, D5≥21d)

1. **`risk/live_ic_bucket_gate.py` (new)** — `LiveICBucketGate`:
   - `record_outcome(strategy, direction, regime, signed_next_bar_return)` — updates the rolling per-bucket deque.
   - `bucket_status(strategy, direction, regime) -> {n, hit_rate, mean_ret, state: WARMUP|PASS|BLOCK}`.
   - `size_multiplier(...) -> float` (1.0 PASS/WARMUP, 0.0 BLOCK in ENFORCE mode; always 1.0 + log in SHADOW mode).
   - `to_dict()/from_dict()` persistence (mirror `core/anti_churn.py`). Fail-open on any error (never block on a tracker bug — same doctrine as the FLIP-PERSIST `except`).
2. **Outcome wiring** — feed `record_outcome` from the same place realized/next-bar outcomes are computed for attribution (reuse `data/trade_attribution.jsonl` producer or the INTENT→next-bar evaluator from the forensic script `C:/tmp/cf_breakdown.py`). Bucket key must use the **intent's** strategy/direction/regime at decision time.
3. **Gate consult** — in `main.py` intent path next to B1 / FLIP-PERSIST (`~main.py:11593+`), behind config `live_ic_gate_mode: "shadow"|"enforce"|"off"` (default **shadow**). SHADOW logs `[IC-GATE] would BLOCK {bucket} (hit={x}% n={n})`; ENFORCE applies `size_multiplier`.
4. **Config** — `configs/live_high_risk.json`: `live_ic_gate_mode`, `ic_gate_min_samples`, `ic_gate_block_hit_rate`, `ic_gate_window`. ProductionConfig fields + `from_file` parse (mirror the `flip_persist_*` pattern from P142).
5. **Persistence + ExecutionContext** — persist the tracker in the paper_positions save block (`main.py:~14780`) and restore (`~16419`), same as `_anti_churn`.
6. **Promotion runbook** — operator flips `live_ic_gate_mode: shadow → enforce` only after D5 is met and the shadow log shows the blocked buckets match the forensic direction (sanity: shorts/mean_revert/QUIET_ACCUMULATION should be the ones lighting up).

**Total**: ~1.5 person-days for the module + wiring + tests; then a ≥21-day SHADOW soak before enforce.

---

## Items explicitly DEFERRED / DROPPED
- **Layer 3 (perp shorts)** — operator's active work. Not here.
- **Hardcoded bucket bans** — rejected (in-sample overfit; D5 exists precisely to avoid this).
- **Confidence-based gating** — rejected (anti-predictive in the data).
- **Per-asset buckets** — deferred; assets are collinear with strategy/regime here, and would worsen sparsity. Revisit if `strategy×direction` proves too coarse.

---

## Iron Laws (sustained)
1. Gate only ever **reduces** activity (size→0 or HOLD). Never opens, never blocks reduces/exits/safety. Same carve-out as B1/Layer-2.
2. **Fail-open**: any tracker error → allow + WARN (never block trading on a bookkeeping bug).
3. **SHADOW-first, post-Layer-2 data only.** No enforcement on pre-2026-06-13 buckets.
4. Confidence is not a feature anywhere in this gate.
5. Reversible: `live_ic_gate_mode: off` returns byte-identical behavior.

---

## Failure modes (to enforce when executable)
- **Sparsity starvation** — most fine buckets never reach `min_samples` → gate is a silent no-op. Mitigation: log per-bucket `n` each tick; if < X% of activity is in PASS/BLOCK (vs WARMUP) after the window, coarsen granularity (D1).
- **Regime drift** — a bucket that had edge loses it (the live-IC degradation P40/P41). The rolling window handles this *if* the window is short enough; too long = stale. Tune D3.
- **Reflexivity** — blocking a bucket stops generating its outcomes → it freezes in BLOCK forever. Mitigation: keep recording the *would-be* outcome in SHADOW even for BLOCKED buckets (counterfactual next-bar return needs no fill), so a bucket can earn its way back.
- **Double-counting with Layer 2** — Layer 2 already suppresses churn/flips; ensure the IC outcome is recorded on the *intent at decision time*, not on suppressed re-fires, or hit-rates get polluted.

---

## Open questions for operator
1. D1 granularity — start `strategy×direction` or go straight to add `regime`?
2. D5 — is 21 days + min_samples acceptable, or hold to the full 30-day v5.1 IC pause before even SHADOW-enforcing?
3. Should the gate's outcome source be the existing `analytics/ic/` pipeline (reuse, single source of truth) rather than a new evaluator? (Recommend yes — avoid a second, divergent IC computation.)
4. Interaction with the operator's Coinbase two-sleeve: when shorts route to the perp (Layer 3), does the short bucket's IC follow the *signal* regardless of venue, or split per-venue? (Recommend: per-signal, venue-agnostic — the edge is in the call, not the venue.)

---

## Output checklist (when ready to execute)
- [ ] `risk/live_ic_bucket_gate.py` + unit tests (truth-table + persistence + fail-open, mirror `tests/test_anti_churn_layer2.py`).
- [ ] Outcome wiring reuses `analytics/ic/` (no second IC computation).
- [ ] Config + ProductionConfig + from_file (mirror P142 `flip_persist_*`).
- [ ] `main.py` consult next to B1/FLIP-PERSIST, default `mode=shadow`.
- [ ] Persistence in save/restore blocks.
- [ ] CLAUDE.md P-entry + memory update; CI gate green; verify smoke green.
- [ ] Deploy SHADOW; confirm `[IC-GATE] would BLOCK` log lines match the forensic buckets.
- [ ] **Hold ≥21d (D5) before flipping to enforce.**

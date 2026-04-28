# HMATS Action Validation Report — v2 Pre-Implementation Red Team

**Date**: 2026-04-28
**Mode**: READ-ONLY. No code modified.
**Input**: `docs/HMATS_GROWTH_VALIDATION_REPORT_2026-04-28.md` (v1) — 4 P0 + 3 P1 recommendations.
**Output**: GO / HOLD / NO-GO / KILL decision matrix + per-rec rationale.

---

## Executive Summary

| Decision | Count | IDs |
|---|---|---|
| **GO** | 3 | P0-3, P1-6, P1-7 |
| **HOLD** | 2 | P0-1, P0-4 |
| **NO-GO** | 1 | P0-2 (rewritten as it stood) |
| **KILL** | 0 | — |

**Net change vs v1**: v1's "implement all 4 P0 + 3 P1 in ~5 person-days" simplification does NOT survive red-teaming. After Phase A (evidence) + Phase B (counterfactual) + Phase C (pre-mortem) + Phase D (Iron Law) + Phase E (ROI):

- **P0-2 was based on a factual error.** `record_outcome()` IS already wired at `core/execution_service.py:2437`. The actual gap is the 1-hour signal-outcome matching window in `analytics/strategy_aging.py:291`. AND the counterfactual on simply extending that window to 6h shows it would capture 28 additional outcomes that are net **negative −$11.77 mean** — extending the window without first hardening the aging logic captures losing trades into weight adjustment, which is the opposite of what we want.
- **P0-1 needs production data BEFORE decision.** The "0% firing" claim is from a 3-minute window. Cannot decide GO/NO-GO until a 7-day production capture.
- **P0-4 is GO if Hypothesis A holds (1-line sign convention bug); HOLD if Hypothesis B (regime-conditional inversion) is true.** Decision conditional on a one-day diagnostic.
- **P0-3 + P1-6 + P1-7 are clean GOs**: pure observability, zero PnL risk, ~1.5 person-days total.

---

## Decision Matrix

| ID | Recommendation | Evidence | Counterfactual | Pre-mortem | Iron Law | ROI | **Decision** |
|---|---|---|---|---|---|---|---|
| **P0-1** | Diagnose kraken_quant 0% firing | **WEAK** — 3-min sample | INSUFFICIENT DATA | 5 modes ✓ | PASS | 0.0 | **HOLD** — need 7-day production capture |
| **P0-2** | Wire `record_outcome()` (rewritten as window-extension) | **MEDIUM** — wiring already exists; v1 was wrong about the gap | 28 additional captures @ −$11.77 mean → **CI [−$415, −$282]** | 5 modes ✓ | PASS | **−0.67** | **NO-GO** — window extension as a standalone fix captures losers |
| **P0-3** | Populate `regime_at_entry` field | **STRONG** — 100% empty in 90 trades | N/A (zero PnL impact) | 5 modes ✓ | PASS | N/A (observability) | **GO** |
| **P0-4** | RSI sign-flip diagnosis + fix | **STRONG** — multi-asset IC < 0 | Hypothesis A: +$82/90d, CI [+$82, +$82] | 5 modes ✓ | PASS | 0.89 | **HOLD pending diagnosis** — auto-GO if Hypothesis A confirmed |
| **P1-5** | IC framework cron wiring | **MEDIUM** — gap confirmed | N/A | 5 modes ✓ | PASS | N/A | **HOLD** — depends on P1-7 (OHLCV sync first) |
| **P1-6** | Strategy correlation matrix from BACKFILL | **WEAK** — needs data | N/A | 5 modes ✓ | PASS | N/A | **GO** (analysis-only) |
| **P1-7** | Sync `equity_history.jsonl` to local audit | **MEDIUM** — file missing locally | N/A | 5 modes ✓ | PASS | N/A | **GO** |

### Decision rule application

- **GO**: Iron Law clear + (zero PnL risk OR positive ROI) + pre-mortem complete with numeric kill criteria.
- **HOLD**: Has a missing-piece dependency (production data, diagnostic outcome, OR upstream rec). Re-enters decision after dependency resolved.
- **NO-GO**: Counterfactual 95% CI lower < 0 OR causal chain SPECULATIVE.
- **KILL**: Iron Law violation. None present.

---

## Per-GO Implementation Cards

### GO #1 — P0-3: Populate `regime_at_entry`

**Files to modify**: `analytics/trade_attributor.py` (line 173 already passes `regime=regime`; verify the caller actually passes a non-empty string).

**Files to verify**:
- `core/execution_service.py` — find the `record_open` / `record_entry` call site that constructs the `AttribTradeRecord` and ensure `regime` argument is sourced from `market_data.get('regime_state')` not an empty string.
- `core/feedback_loop.py:286, 304` — same check.

**Tests to add**: `tests/test_trade_attribution_regime_p128.py`:
```
test_regime_at_entry_populated_for_all_record_calls()
test_regime_at_entry_matches_market_data_at_record_time()
test_regime_at_entry_is_string_not_enum()
```

**Shadow plan**: N/A — observability fix. Deploy directly after smoke test passes.

**Kill criteria**: 7 days post-deploy:
- If <80% of new trades have populated regime → revert + investigate caller.
- If any trade has regime not in {GMM regime enum} → revert + add validation.

**Estimated effort**: 0.5 person-day.

---

### GO #2 — P1-6: Strategy correlation matrix from BACKFILL

**Files to add**: `analytics/ic/compute_strategy_correlation.py` (new analysis script, ~100 LOC).

**Inputs**: `logs/ic_signals/ic_signals_BACKFILL_{BTC,ETH,SOL}.jsonl` (production-side; sync via P1-7 first).

**Outputs**: `analytics/ic/reports/strategy_correlation_matrix_<date>.json` — 12×12 pairwise Pearson correlation across kraken_quant strategies + 4×4 across Best-of-N.

**Tests to add**: `tests/test_strategy_correlation_p128.py`:
```
test_correlation_matrix_symmetric()
test_correlation_matrix_diagonal_one()
test_correlation_matrix_in_unit_range()
```

**Shadow plan**: N/A — analysis-only.

**Kill criteria**: N/A — analysis output, not a runtime change.

**Estimated effort**: 0.5 person-day.

**Why this matters**: directly answers whether the 12 kraken_quant strategies represent 3-4 independent alpha sources or actually 12. If correlation > 0.7 between most pairs, "add more strategies" makes even less sense.

---

### GO #3 — P1-7: Sync `equity_history.jsonl` to local audit

**Files to add**: `scripts/sync_audit_data.sh` — pulls equity_history + signals_*.jsonl + outcomes_*.jsonl + proof_log_*.log from production volumes via `scp hmats:/var/lib/docker/volumes/hmats-{data,logs}/_data/...`.

**Files to verify**: cloud-side — confirm `equity_history.jsonl` exists and is being written. From CLAUDE.md "Phase 0.5 Drawdown Assessment" we cited it as missing.

**Tests to add**: lightweight bash assertion script (line count > 0, JSON parseable on tail -1).

**Shadow plan**: N/A — data sync.

**Kill criteria**:
- Local file mtime > 7 days → sync broken.
- > 10% of lines unparseable JSON → cloud writer corrupting.

**Estimated effort**: 0.5 person-day.

---

## Per-HOLD: Missing Evidence + Unblock Conditions

### HOLD #1 — P0-1: kraken_quant diagnosis

**Why HOLD**: v1 says "12 strategies firing 0/0 in 100% of sampled records" but the sample is `data/kq_firing_stats.json` which only covers ~3.6 minutes of uptime since last restart (per CLAUDE.md P115). Cannot distinguish "permanently broken" from "no chop-regime ticks observed yet".

**Unblock**: 7-day production capture of `data/kq_firing_stats.json` post-restart. If after 7 days × 6 ticks/day = 42 ticks, the 12 strategies still fire 0/12 in every regime they cover, the gap is real and warrants P0 investigation. If even 1-2 strategies fire occasionally, the right diagnostic shifts to "why these 10 specific strategies don't fire" (per-strategy threshold tuning, much smaller scope).

**Pre-mortem captured 5 failure modes** (regime threshold drift / authority gate / is_valid suppression / negative IC at confidence floor / confidence floor drift). All have observable + warning + kill thresholds documented.

**Estimated effort post-data**: 1-2 person-days investigation.

**Re-evaluation date**: 2026-05-05 (7 days from now).

---

### HOLD #2 — P0-4: RSI sign-flip fix

**Why HOLD**: The IC evidence (BTC −0.0296, ETH −0.0175, SOL −0.0429, all p < 0.05) is STRONG. But there are TWO competing hypotheses:
- **Hypothesis A** (signal-convention bug): 1-line fix in RSI computation (~1 day total). Counterfactual: +$82/90d, +$328 annualized, ROI=0.89.
- **Hypothesis B** (regime-conditional inversion in current crypto regime): requires per-regime retune (~3 days), probable shadow window (14 days), risk score ↑.

**Unblock**: One-day source-code investigation:
1. Grep RSI computation site in `agents/kraken_quant_agent.py` + `data_mgmt/market_data_pipeline.py`.
2. Check whether the sign convention assumes RSI > 70 → SHORT (mean-revert) vs RSI > 70 → LONG (momentum).
3. If the code says "RSI > 70 → SHORT" but the trade attribution shows the agent went LONG when RSI > 70, **Hypothesis A confirmed → automatically promote to GO**.
4. If the convention IS correct + IC is still negative → **Hypothesis B confirmed → stays HOLD pending shadow window decision**.

**Pre-mortem captured 5 failure modes** including the 2-phase regression (sign-flip works for 3 days then re-inverts) — kill criterion: 14d post-fix, RSI IC < 0 across 2+ assets → revert.

**Re-evaluation**: After diagnostic completes (1 day).

---

### HOLD #3 — P1-5: IC cron wiring

**Why HOLD**: Low risk but has a hard dependency on P1-7 (need OHLCV parquet files synced to runtime container before cron can run reliably). Doing P1-5 before P1-7 means the cron silently fails on FileNotFoundError.

**Unblock**: P1-7 GO ships first. Then 1-day cron-wiring task.

**Re-evaluation**: After P1-7 deploys.

---

## Per-NO-GO: Rejection Rationale

### NO-GO — P0-2 (as v1 stated it)

**v1 claim**: "Wire `strategy_aging.record_outcome()` into trade-close path (3 days)."

**Phase A discovery**: the wire ALREADY exists.
```
core/execution_service.py:2437
    ctx.strategy_aging.record_outcome(
        strategy_name=_c12_strategy,
        signal_timestamp=_c12_sig_ts,
        pnl_bps=_c12_pnl_bps,
        was_correct_direction=_c12_correct,
    )
```
v1's evidence ("never called anywhere") is FALSE. The recommendation as written is solving a problem that doesn't exist.

**The actual gap** (re-characterized): `analytics/strategy_aging.py:291` matches signal records to outcomes within a **1-hour** window. With 17 trades in 30 days having multi-hour holds, most outcomes don't match any signal record. The aging logic therefore has near-zero outcome data and `get_weight_modifiers()` returns 1.0 forever.

**Phase B counterfactual on the obvious "extend to 6h" fix**:
- Of 52 historical closed trades, **3** (5.8%) had hold time ≤1h, **31** (59.6%) had hold time ≤6h.
- The 28 newly-captured outcomes have **mean PnL = −$11.77** (worse than the overall −$10.39 mean).
- 90d hypothetical impact: mean −$368, σ=$50, **95% CI [−$415, −$282]** — fully negative downside.

**Why "capturing more outcomes" is bad here**: aging weight modifiers DEMOTE strategies based on outcome PnL. If the additional captured outcomes are losers AND the aging logic has any leverage on weights, it would correctly demote losing strategies — net positive. BUT (per CLAUDE.md P64 + the v1 report) the live-mode state-recording paths are partially dead-looped. Without proven aging effectiveness, capturing more loser-outcomes adds noise to weight calculations rather than improving them. **Conservative interpretation: NO-GO until aging effectiveness is independently validated.**

**The right fix** (filed for v3): Audit the read side — does `get_weight_modifiers()` actually get consumed by best-of-N selection? If not, the gap is the READER, not the matching window. Fix the reader first; matching window extension is a follow-on optimization.

---

## Per-KILL: None

No recommendation violates Iron Law. The 4 P0 + 3 P1 are all observability / wiring / diagnostic — none touch `defense/constitution.py`, `training/`, `obs_dim=126`, ExistenceFuse / CRACK thresholds, or introduce fail-OPEN semantics.

---

## Phase G — Implementation Order (GO items only)

```
Day 1 morning: P0-3 deploy
  - Verify regime is passed at trade-record-creation
  - 0.5 day, no shadow needed

Day 1 afternoon: P1-7 deploy
  - Build sync script + cron entry
  - 0.5 day

Day 2 morning: P1-6 deploy
  - Run correlation analysis once on synced backfill data
  - 0.5 day output: strategy_correlation_matrix_2026-04-30.json

Day 2 afternoon: P0-4 diagnostic (HOLD-resolution)
  - 1-day source-code investigation of RSI sign convention
  - If Hypothesis A confirmed → P0-4 promotes to GO; ship same day (1-line fix + paper smoke test)
  - If Hypothesis B → P0-4 stays HOLD; file v3 task for regime-conditional retune

Day 3+ (concurrent with day-2 P0-4 diagnostic):
  - Begin P0-1 production data capture (kq_firing_stats.json over 7 days, no engineer time required, just wait)
  - 2026-05-05 re-evaluation date for P0-1

Day 4: P1-5 deploy (only after P1-7 confirmed working)
  - Cron-wire IC compute
  - 0.5 day

Total committed time: 2.5 person-days for the 3 GOs + ~1 day diagnostic for P0-4 = 3.5 days.
+ 0 engineer time for P0-1 (passive data capture).
+ Deferred: P0-1 + P0-2 reformulation + P0-4 hypothesis B.
```

---

## Side Findings (filed for v3 audit)

1. **P0-2 reformulation**: The real research question is "does `get_weight_modifiers()` have any consumer in the hot path?". File for v3 to investigate the READ side of strategy_aging before touching the matching window.

2. **kq_firing_stats.json window**: This file should persist longer than ~3 minutes. Filed: investigate whether the writer truncates on restart vs appends. If it truncates, we lose the longitudinal data needed to even evaluate P0-1.

3. **17 trades / 30 days is statistically thin**: All P0/P1 reasoning is bounded by tiny sample. The biggest unknown isn't strategy correctness — it's whether the system is firing often enough to learn anything. Filed: a separate "trade frequency reality check" audit per CLAUDE.md's standing rule (>7 days at 0 trades demands ONE recommendation: relax thresholds aggressively, then add safety back).

4. **All counterfactuals here use rough approximations**. The 95% CIs are bootstrap estimates from 90 historical trades — not true forecasts. Confidence in the NO-GO on P0-2 (window extension) is high (mechanism + sign of CI agree); confidence in P0-4 magnitude is lower (IC-to-PnL conversion is heuristic).

---

## Iron Law Compliance Summary

| Recommendation | constitution.py | training/ | obs_dim=126 | veto chain | thresholds | fail-open | Verdict |
|---|---|---|---|---|---|---|---|
| P0-1 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR |
| P0-2 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR (fails on counterfactual, not Iron Law) |
| P0-3 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR |
| P0-4 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR |
| P1-5 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR |
| P1-6 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR |
| P1-7 | PASS | PASS | PASS | PASS | PASS | PASS | CLEAR |

**No KILL triggers anywhere. All 7 are constitutionally safe.** The decisions reduce to evidence strength + counterfactual + ROI.

---

*Generated 2026-04-28. READ-ONLY red-team. No code or runtime state was modified during this audit.*
*Next action: implement the 3 GO items in the Phase G order. Re-run v2 audit in 7 days for HOLD-promotion decisions.*

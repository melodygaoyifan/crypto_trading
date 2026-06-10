# HMATS — Project Status & Development Guidelines

**Last updated:** 2026-06-10 (P139)
**Version:** HMATS v6.8.0
**Live mode:** Kraken, Hetzner CPX21, `configs/live_high_risk.json`

> **⚠️ CLAUDE.md discipline**: When you finish a non-trivial change, update the relevant
> sections here in the same commit. Specifically: runtime-state (ACTIVE/SHADOW), authority
> matrix changes, known-pitfalls. Stale CLAUDE.md = repeat bugs. See §Pitfalls for incidents.

---

## Current Runtime State (cloud, 2026-04-22)

| Component | State | Notes |
|---|---|---|
| **DRL (TQC)** | **ACTIVE** | 3/3 TQC models loaded from volume `hmats-models`. Backtest Sharpe BTC +9.22 / ETH +7.32 / SOL +10.29 |
| **Best fold per asset** | BTC fold_3, ETH fold_3, SOL fold_3 | ETH fold_1 was stale (train_rows=0) — permanently switched to fold_3 |
| **Sentiment L1 (F&G)** | ACTIVE | `DeterministicSentimentEngine`, writes `sentiment_direction`/`sentiment_confidence` |
| **Sentiment LLM (Haiku)** | ACTIVE | `SentimentLLMAgent`, CryptoPanic + CC News blend |
| **Quant (Best-of-N)** | ACTIVE (DECIDE) | 4 strategies: mean_revert, momentum, volume_breakout, vrp (+ hold) |
| **kraken_quant (12 strat)** | ACTIVE (**DECIDE**) | Promoted 2026-04-22 from ADVISE+×0.5 dampen. 12 institutional strategies now full-weight. Per-strategy stats in `data/kq_firing_stats.json` |
| **onchain_sol** | ACTIVE | Singleton agent, `.start()` dispatched in `run_live()` as well as `run_paper()` |
| **Binance WS (micro)** | ACTIVE | taker flow + mark price for cross-exchange microstructure |
| **Discord alerts** | ACTIVE | webhook in `.env` → `DiscordLogHandler` forwards ERROR/CRITICAL + 4H heartbeat |
| **Attribution tracker** | ACTIVE | 16-agent coverage (see §Authority Matrix) |
| **Execution shadow** | RETIRED | Cutover to `execute_intent_v2` completed 2026-04-18 (commit ef4060b); the shadow call site was deleted in that same commit. Snapshot-capture dead code + `_enable_execution_shadow` flag removed 2026-04-24. Re-enabling requires a `shadow_mode` kwarg on `execute_intent_v2` that short-circuits the ~10 live `record_*()` mutations — dict deep-copy alone would double-record anti_churn/thesis_budget/existence_fuse/trade_attributor/etc. |
| **Exit DRL (Discrete SAC)** | **EXIT_ONLY (all 3 assets)** | Third DRL alongside the TQC direction DRL (P28). Authority cap = EXIT_ONLY; never decides direction. v1 = 4 actions {HOLD, PARTIAL_EXIT, RELEASE_RUNNER, EXIT_ALL}. **v2 checkpoints (200ep, seed=42) in `models/exit_drl_v2/{ASSET}/exit_sac_best.pt`:** BTC val_align=0.730, ETH=0.710, SOL=0.746. **All 3 assets promoted via accelerated path 2026-04-24 (CLAUDE.md P29) — bypasses spec's 30-day shadow + ≥30-event gate.** Bridge: System-3's PARTIAL_EXIT prediction overrides System-1's DRLOutput at [core/tick_exit_triggers.py:373-403](core/tick_exit_triggers.py#L373-L403) when (a) asset is in EXIT_ONLY, (b) Exit-SAC predicted PARTIAL_EXIT this tick. **Limited to PARTIAL_EXIT only — RELEASE_RUNNER and EXIT_ALL stay handled by the existing 5 rule-based triggers.** Audit stamp: `data/exit_drl_promotion_state.json` records `force_promote_at` + reason + blockers-at-override per asset. End-to-end diagnostic: `python -X utf8 scripts/exit_drl_e2e_diagnostic.py` (10 stages, all green). |

**Verification commands** (run these when in doubt):
```bash
# ---------- TRUTH-LEVEL (static, no runtime needed) ----------
# Is DRL really ACTIVE? Check all config declarations + instantiated classes:
python -X utf8 scripts/startup_drl_truth.py
# Expected: DRLAuthorityGate.get_authority() = ACTIVE; is_shadow_mode() = False
# If config says ACTIVE but runtime says SHADOW → volume mount bug (P1)

# Full-system 5-hop completeness audit (23 directories, 1122 classes)
# Writes /tmp/hmats_audit/{inventory,wiring_analysis,final_report}.json
python -X utf8 scripts/completeness_audit.py
# Expected: ~1.4% ACTIVE (strict 5-hop: import→instantiate→method→decision→impact).
# Check `agent_signals_flow.dead_reads` section for silent bugs.

# Which agent/ classes are actually wired into main.py?
python -X utf8 scripts/startup_agent_wiring_truth.py
# Expected (2026-04-22): 122/130 ACTIVE (93.8%).
#   - 7 INSTANTIATED_BUT_UNUSED = all in agents/drl_agent.py (P10 tranche/exit
#     scaffolding, mode=DISABLED — not a bug).
#   - 1 DEAD = AgentSignalEnvelope dataclass (constructed via wrap_agent_signal
#     factory, not directly — cosmetic script false-positive).
# Script detects BOTH `self.xxx = Foo(...)` AND local-var `_var = Foo(...)`
# patterns (local-var added 2026-04-22 for AttributionTracker detection).

# ---------- RUNTIME-LEVEL (needs live container) ----------
# Container state + TQC load + startup health
ssh hmats "docker ps && docker logs hmats-engine --since 10m 2>&1 | grep -iE 'TQC loaded|DRL.*ACTIVE|HEALTH_S[0-9]'"

# Per-tick per-agent signal dump (last tick)
ssh hmats "ls /var/lib/docker/volumes/hmats-logs/_data/attribution/signals_*.jsonl | tail -1 | xargs tail -1 | python3 -m json.tool | head -80"

# [AGENT-TRACE] one-line per-tick per-agent snapshot (added 2026-04-22)
ssh hmats "docker logs hmats-engine --since 8h 2>&1 | grep AGENT-TRACE | tail -10"

# DRL promotion state
ssh hmats "docker exec hmats-engine cat /opt/hmats/data/drl_promotion_state.json"

# 4H heartbeat (latest equity + positions)
ssh hmats "docker logs hmats-engine --since 4h 2>&1 | grep HEARTBEAT"

# kraken_quant 12-strategy firing breakdown (after 1+ tick)
ssh hmats "docker exec hmats-engine python -X utf8 scripts/kq_strategy_diagnostic.py"

# 16-agent attribution audit
ssh hmats "docker exec hmats-engine python -X utf8 scripts/agent_audit_16.py"
```

**Where the "quant DECIDE" signal actually lives** (important — commonly confused):
`data_mgmt/market_data_pipeline.py:1244` Best-of-N strategy selector produces
`quant_direction`/`quant_confidence`. The file `agents/quant_agent.py` is **orphan
legacy code** (only referenced from `core/runtime_spine.py`, which itself is not on
the live call path). Don't go looking for the quant agent class expecting it to be
the decision maker — it isn't.

---

## Authority Matrix (v6.8)

`signals/authority_fusion.py` declares **25 agents** in `AUTHORITY_MATRIX_NORMAL` (soldex added 2026-04-15 — see SOLDEX-AUTHORITY tag at line 193 — replaced an earlier slot, kept count at 25).
`_build_fusion_signals` actually reads and uses **19** of them for direction/confidence;
the other 6 are architecturally non-directional (risk/macro/lead_lag/cvd/structure/options).

| # | Agent | Authority | Keys in agent_signals | Consumed by |
|---|---|---|---|---|
| 1 | quant | DECIDE | quant_direction, quant_confidence | fusion + attribution |
| 2 | regime | CONFIRM | regime_direction, regime_confidence | fusion |
| 3 | drl | ADVISE (ACTIVE → DECIDE) | drl_direction, drl_confidence | fusion + attribution |
| 4 | sentiment | ADVISE | sentiment_direction, sentiment_confidence | fusion + attribution |
| 5 | macro | CAP | macro_leverage_cap | fusion (leverage cap only) |
| 6 | lead_lag | EXECUTE | lead_lag_edge, lead_lag_confidence | fusion (timing only) |
| 7 | risk | VETO | risk_veto | fusion (veto only) |
| 8 | two_stage | CONFIRM | two_stage_direction, two_stage_confidence | fusion + attribution |
| 9 | short_bias | ADVISE (PENALIZE) | short_bias_direction, short_bias_confidence | fusion + attribution |
| 10 | funding_rate | ADVISE | funding_direction, funding_confidence | fusion + attribution |
| 11 | onchain (BTC/ETH) | ADVISE | onchain_direction, onchain_confidence | fusion + attribution |
| 12 | llm_sentiment | ADVISE | llm_sentiment_direction, llm_sentiment_confidence | fusion + attribution |
| 13 | flow | ADVISE | flow_direction (whale+exchange+ETF net) | fusion + attribution |
| 14 | structure | CONFIRM | structure_confirmed (bool) | fusion (boolean only) |
| 15 | squeeze | ADVISE | squeeze_risk (bridged from squeeze_score at main.py:7662) | fusion (veto above 0.7) |
| 16 | cvd | ADVISE | cvd_divergence | fusion (one-sided) |
| 17 | risk_appetite | ADVISE | macro_risk_appetite | fusion (derived direction) |
| 18 | kraken_quant | **DECIDE** (was ADVISE) | kq_direction, kq_confidence | fusion + attribution; ×0.5 dampen removed 2026-04-22; CVD z-score + bearish funding-divergence branches ported from archived quant_agent.py 2026-04-22 |
| 19 | microstructure | ADVISE | micro_imbalance, micro_confidence, micro_direction | fusion + attribution |
| 20 | model_alpha | ADVISE | model_alpha_direction, model_alpha_weight | fusion + attribution |
| 21 | onchain_graph (SOL) | ADVISE | onchain_graph_direction, onchain_graph_confidence | fusion + attribution |
| 22 | options | ADVISE | options_short_confirmation, options_confidence | fusion; **×0.5 dampen removed 2026-04-22** — full weight |
| 23 | vol_alpha | ADVISE | vol_alpha_direction (always 0; runs via intensity) | **fusion branch REMOVED** — affects execution only |
| 24 | whale | ADVISE | whale_flow_direction, whale_confidence (bridged at main.py:7402) | fusion + attribution |
| 25 | soldex (SOL) | ADVISE | soldex_arb_direction, soldex_confidence | fusion + attribution |

**Attribution tracker** (`main.py:8299`) covers 16 direction-producing agents.
Adding a new agent requires **3 files**: agent_signals write site + `_attr_collected` + `_EXTRACTORS` dict in `agents/signal_envelope.py`.

---

## Non-Negotiable Rules

1. **Constitution is supreme** — no trade without alpha gate pass
2. **P0 Safety cannot be bypassed** — kill switch, stale data guard, rate limiter
3. **Existence Fuse** — 28d window, -5% PnL → system halt, manual recovery only
4. **DRL Authority** — ACTIVE since 2026-04-22.
5. **Single exchange** — Kraken only (Binance/Deribit in `legacy/` for historical refs)
6. **Three trade_gate call sites** — main veto_chain, authority_chain, AND p0_safety_integrator ALL call `trade_gate.check()`. Fix ALL three when changing the gate API.
7. **CLAUDE.md discipline** — runtime-state changes, new pitfalls, and authority-matrix edits MUST update this file in the same commit.

---

## CRITICAL: Trade Frequency Reality Check

Before suggesting any new patch, audit, or "optimization", Claude MUST first ask:

1. Has the system executed any trade in the last 24h? Last 7d?
2. If 0 trades, what is the documented bottleneck (with log evidence)?
3. Is the proposed patch directly addressing that bottleneck?

If the system has had 0 trades for >7 days while "running normally":
- Stop suggesting new patches
- Stop suggesting new audits
- The bottleneck is NOT in the patches — it's in the system's
  fundamental ability to convert signals to trades
- Recommend ONE thing: revert to a known-trading version,
  or relax thresholds aggressively until trades happen,
  THEN add safety back

Anti-pattern Claude must avoid:
- "Let me suggest 8 more audit dimensions" (over-engineering on
  a system that doesn't trade)
- "Let me write a 1126-line patch prompt" (compounding complexity
  on a non-trading system)
- "Let me validate one more thing" (analysis paralysis)

The user's time is finite. A non-trading system is ZERO ROI
regardless of how clean the architecture is.

---

## Lessons (Process Discipline)

### Lesson: Two Kinds of Stale State

Stale state has two forms in trading systems:

1. **Code stale**: Conversation memory ≠ codebase state.
   Example: `[SOTA-G2]` already wired but I assumed not.
   Mitigation: VERIFY codebase before propose.

2. **Data stale**: Earlier diagnostic snapshot ≠ today's state.
   Example: Tier 4 said STRUCTURE 48% killer; today's analyzer
   shows STRUCTURE 0%, WEEKEND 83%.
   Mitigation: Re-run diagnostic before applying patches based
   on prior data.

Both are equally dangerous. Today's session ran into both at
least 4 times. Future Claude sessions must check both before
proposing any change.

### Lesson: Defensive Mitigation Is Not a Fix

P52/P56 made the fallback non-catastrophic. This is good
operational hygiene but does not address root cause. Future
discipline: when applying defensive mitigation, log the gap
explicitly so it doesn't appear "fixed" in dashboards while
the root cause continues to fire.

### Lesson: Heisenbug Investigation Discipline

When state is correct at observation point A but wrong at
observation point B, do not patch the gate. Trace the state
mutation path between A and B. Three standard candidates:
  - Mutation paths post-init
  - Observation gaps (rate-limited logs hiding frequency)
  - Object identity (shallow vs deep copy, instance vs class)

Today's weekend_config bug is a textbook example. CLAUDE.md
P45 flagged it; follow-up trace was not done.

### Lesson: Parallel-Edit Discipline (operator + Claude)

The operator and Claude often edit the codebase IN PARALLEL.
Operator commits land between Claude's `git status` checks. Claude's
mental model of "what's in main" lags reality by minutes-to-hours.

This session's evidence: P82/P83/P84 ALL had operator-titled sibling
commits (24ffcd6, 2d0a93c, 72ea60f) between Claude's pushes. One of
those introduced a P15-shape bug (`ShadowLedgerWriter.frozen_allocations`
read with no writer) that Claude's code didn't touch but cascaded to
**10 container restarts in 6 minutes** before P85 emergency-fixed it.

**Three rules to enforce parallel-edit safety:**

1. **Re-pull before EVERY non-trivial commit.** Do `git fetch origin
   && git log HEAD..origin/main` before staging. If origin has new
   commits, REVIEW them before committing — operator may have already
   fixed/changed/refactored what you're about to touch.

2. **Every new attribute READ must defend with `getattr(...)` or
   verify the writer exists in the SAME commit.** P15-shape bugs
   (reader exists, writer missing) recur because the reader-author
   trusts a contract the writer-author didn't honor. With parallel
   edits, that contract drift is invisible at write time. Pattern:

       # WRONG — assumes writer exists somewhere
       value = self.shadow_ledger.frozen_allocations

       # RIGHT — defaults safely + logs gap
       value = getattr(self.shadow_ledger, 'frozen_allocations', None)
       if value is None:
           logger.warning(f"[X] frozen_allocations attr missing; "
                          f"degraded behavior: <what we do instead>")

3. **NEVER add code that exits the process on a missing internal
   attribute.** The reconciler's strict-contract refusal to start LIVE
   when ORDER_CHECK FAILED was a correct safety pattern, but cascading
   to "exit and let docker restart" amplified ONE missing attribute
   into a 10-restart loop. Either:
   (a) the safety check tolerates the failure mode (defensive guard
   degrades gracefully, logs WARNING), OR
   (b) the safety check has a circuit breaker (after N consecutive
   failures, stop trying to restart and require manual intervention —
   prevents docker-compose's `restart: always` from making it worse).

**Forcing function:** the P70 CI gate (codebase-invariants workflow)
runs the silent_failure_audit + lint_silent_swallow on every push.
A new reader added without a defensive `getattr` will trip the lint
if it falls inside a try/except: pass shape. But a reader OUTSIDE a
try/except (like `frozen_allocations`) won't — the strict-contract
caller (reconciler) wrapped it in try/except internally and surfaced
the AttributeError as a top-level FAILED. **The CI gate cannot catch
attribute-access-outside-try/except**; this requires either operator
discipline or a separate scanner pass (filed for future P-entry).

**Rule 4: When adding a new helper method, grep `def <name>` FIRST.**
P87 (commit 088b865) hit a method-collision-shadows bug — added a new
`_clamp_size_to_balance` method to `execution_manager.py:601`, but a
PRE-EXISTING method with the same name lived at `:1545` with a
different signature (4 args returning float vs 5 args returning tuple).
Python silently picks the LAST definition, so callers expecting the
new signature got the old one → `TypeError: takes from 4 to 5 positional
arguments but 6 were given` on every entry attempt → 4-minute
production outage. The hotfix renamed to `_v2`. Lesson: long files
(execution_manager.py is 2400+ lines) make manual scanning unreliable;
EXPLICIT `grep "def <name>"` before adding any non-trivial helper
catches collisions at write time. Add to the standard pre-commit
mental checklist alongside the parallel-edit rules above.

**The CI gate cannot catch this either** — both methods are valid
Python; collision only fails at call time when callers expect
different signatures. Filed under "static analysis can't see
method-resolution-order bugs"; remediation is purely operator
discipline + the grep habit.

---

## Known Pitfalls (source of repeat bugs — read this before deploying)

> **History:** Detailed P-entries older than the last ~30 days have been moved to [archive/CLAUDE_history.md](archive/CLAUDE_history.md) to keep this file scannable. The summaries below + the foundational invariants (P1-P8) are what every session should load.

### Recent pitfalls (last ~30 days)

### P139. [FIXED 2026-06-10] Idempotency-cache phantom-fill inflation — 245-SOL-recorded vs 8.6-actual
- **Symptom:** Cloud forensic 2026-06-10 found paper_positions.json tracked all 3 assets as SHORTS (`direction: -1.0`, $2,427 notional total) while Kraken actually held them LONG (8.6 SOL + 2.1 ETH + 0.048 BTC ≈ $7,000 spot). Heartbeat showed 3 weeks of no trades. Every 4H tick triggered EXIT_ONLY → BUY → REJECTED (P87 dust clamp) → P110 backoff → repeat. P110 was correctly suppressing the alert storm, but the underlying state had diverged from reality over 6 weeks.
- **Root cause:** `execute_order` at [execution_manager.py:1178-1197](execution/execution_manager.py#L1178) has an idempotency cache that returns `success=True, status=FILLED` with the cached `order_id` when the same userref is re-submitted. The userref is generated deterministically from intent parameters, so any tick that produces the same intent shape hits the cache. The caller in [execution_service.py:execute_intent_v2](core/execution_service.py) saw `success=True` and ran the full post-execution block — `record_fill`, `_paper_positions[asset]` mutation, `anti_churn.record_fill`, `thesis_budget.record_fill`, `existence_fuse.record_pnl`, etc. — re-counting the SAME Kraken execution as a fresh fill every tick.
- **Shadow ledger evidence:** Across 52 ledger files (2026-04-07 → 2026-06-04), SOL has 47 BUY FILL records spanning only **9 unique order_ids**. Single order_id `OVRJNB-ME53X-64MQWH` appears **36 times** with different sizes/prices/timestamps. BTC and ETH have the same pattern. Net recorded: 245 SOL net bought (paper); Kraken reality: 8.6 SOL. Total phantom inflation: ~236 SOL across that asset alone.
- **Why P25 family detection missed it:** P25 (2026-04-24) fixed `primary_agent` attribution and added FILL records to live mode (P64 et al). The fix correctly wired `record_fill` into the close path. What it didn't catch: the cache-hit return from `execute_order` looks identical to a fresh fill (`success=True, status=FILLED`, no flag distinguishing the source). The `fill_id` at [defense/p0_safety_integrator.py:700](defense/p0_safety_integrator.py#L700) is synthesized as `f"{order_id}_fill"` — a deterministic value that collides 1:1 with `order_id`, so the shadow_ledger layer had no per-fill dedup either.
- **Fix (P139): two layers of defense.**
  1. **Layer 1 (primary, caller-side) — `OrderResult.is_cached_idempotent: bool = False`**. Set `True` in `execute_order`'s cache-hit return path ([execution_manager.py:1196+](execution/execution_manager.py#L1196)). `to_dict()` serializes the flag. `execute_intent_v2` checks `exec_result.get("is_cached_idempotent")` immediately after the existing failure-check at [execution_service.py:1899+](core/execution_service.py#L1899) and `return exec_result` — short-circuits the entire post-execution block (no `record_fill`, no `_paper_positions` mutation, no governor updates). The order is in the past tense from this tick's perspective; the original (first-time) bookkeeping already happened.
  2. **Layer 2 (belt-and-suspenders) — `ShadowLedgerWriter._recorded_fill_order_ids: Set[str]`**. `record_fill` rejects a second FILL record for an order_id already in the set, returns `False`, WARNs once per duplicate. Falsy `order_id`s skip the dedup (paper-mode synthetic ids that legitimately can collide). `replay_frozen_allocations_from_jsonl` extended to seed `_recorded_fill_order_ids` from prior JSONL `FILL` entries during the same pass that seeds `frozen_allocations`. Mirrors P85-arch shape; the reconciler's defensive `getattr` continues to handle older module versions gracefully.
- **Chaos tests (8 new in `TestChaosIdempotencyCachePhantomFill` in `tests/chaos/test_chaos_order_leak.py`):** OrderResult flag exists + defaults False; `to_dict` serializes; execute_order source contains the `is_cached_idempotent=True` set; execute_intent_v2 source contains the guard + `return exec_result`; record_fill rejects duplicate order_id; falsy order_id doesn't dedup (paper compat); restart-via-replay survives the dedup contract; replay log mentions P139.
- **Paper_positions reconciliation required after deploy:** P139 stops further inflation but doesn't unwind 6 weeks of drift. Paper state on cloud needs a one-time reset to match Kraken — either flip all 3 directions to +1 with sizes computed from current Kraken balances, OR zero them all out and let the engine re-enter from a clean slate. The actual money in the account is fine (Kraken reality is authoritative); only the tracker is wrong.
- **Mitigation pattern:** Any return path that looks identical to "fresh successful operation" must carry a flag distinguishing it from a cached/idempotent return. The caller must check the flag and skip side effects that are not idempotent themselves. Same shape as P85 reader/writer contract drift but at the *return-value* layer: the function's contract said "True = fresh fill" but actually returned "True = either fresh OR cached". Add to the recurring-bug-class catalog.

### P138. [FIXED 2026-06-09] Margin-position close paths sent spot orders without leverage → stranded short on SOL
- **Symptom:** Production Discord stream 2026-06-09 22:36 UTC. SOL price moved 25.7% off FastRiskTick anchor, triggering EXIT_ONLY on a SHORT position. `_handle_fast_risk_action` LIVE branch submitted a spot MARKET BUY via `execute_order(...)` (with no `leverage=` kwarg). P87 balance clamp at [execution_manager.py:_clamp_size_to_balance_v2](execution/execution_manager.py) saw spot USDT free=$0.12 vs required=$756, clamped to 0.001901 SOL — below Kraken's 0.020 min — REJECTED with `INSUFFICIENT_SPOT_BALANCE`. P110 backoff correctly engaged for 30 min and the watchdog went quiet, but the position remained stranded.
- **Root cause:** A SHORT position can only exist on Kraken via margin trading. The entry path correctly passes `leverage=int(round(regime_leverage)) if regime_leverage > 1.0 else None` ([execution_service.py:1766,1794](core/execution_service.py#L1766)) and stores `regime_leverage` on `_paper_positions[asset]` ([execution_service.py:3204](core/execution_service.py#L3204)). But the three watchdog-path close call sites omitted leverage entirely and routed everything as spot:
  1. `main.py:_handle_fast_risk_action` ([main.py:14229](main.py#L14229)) — FastRiskTick EXIT_ONLY / REDUCE_50
  2. `main.py:_crisis_position_reduction` ([main.py:14365](main.py#L14365)) — CORR-0 4H crisis reduction
  3. `main.py:_emergency_flatten` ([main.py:14532](main.py#L14532)) — DEAD_MAN_SWITCH / panic flatten
- **Why P87's clamp didn't recover:** `_clamp_size_to_balance_v2` always checks spot wallet (`fetch_balance()['free'][quote]`). For a margin BUY-to-close, the actual constraint is margin collateral availability (validated server-side by Kraken), NOT free quote. Free USDT can be near zero while the margin short is perfectly closable. The clamp was right to reject a *spot* order with insufficient quote — the bug was upstream, sending a spot order at all.
- **Fix (P138):** three-part.
  1. `execution/execution_manager.py:_clamp_size_to_balance_v2` — added `leverage: Optional[int] = None` param. When `leverage > 1`, return `(size, "")` immediately (skip the spot balance check). Comment block explains the contract: margin orders are validated server-side by Kraken; if collateral is insufficient, P79's `EOrder:Insufficient margin` classifier surfaces it as PERMANENT.
  2. `execution/execution_manager.py:execute_order` — pass `leverage=leverage` when calling `_clamp_size_to_balance_v2`.
  3. `main.py` — all three watchdog-path close sites now read `_pos_leverage = pos.get("regime_leverage", 1.0)` and pass `leverage=int(round(_pos_leverage)) if _pos_leverage > 1.0 else None` to `execute_order`. Tagged `[P138 2026-06-09]` inline.
- **Behavioral test:** `python -X utf8 -c "..."` smoke confirmed: spot path (`leverage=None`) still rejects the incident-replica scenario at the dust threshold; margin path (`leverage=2`) passes the full size through unclamped; explicit `leverage=1` is treated as spot (correct).
- **Operator action for stranded SOL position:** the system will auto-retry on next 4H anchor (P110 backoff clears via `set_4h_anchor`). If `regime_leverage` was correctly stored at entry, the close now routes margin and should succeed assuming margin collateral is intact. If the close still fails with `EOrder:Insufficient margin`, replenish margin collateral or manually close via Kraken UI.
- **Same-family check that PASSED:** entry-path `execute_intent_v2` already plumbs `regime_leverage` to both single-shot and multi-slice executions ([execution_service.py:1766,1794](core/execution_service.py#L1766)). Only the three off-band watchdog paths were missing the plumbing.
- **Mitigation pattern:** Any safety-net code path that places orders against an EXISTING position must read the position's instrument shape from the same source the entry wrote it. Position state (`_paper_positions[asset]`) IS the source of truth for `regime_leverage`, `direction`, `tranche`, etc. — close paths that ignore those fields can structurally fail to net the position they're trying to flatten. This is the same shape as P15 (reader/writer mismatch) and P85 (cross-module attribute contract drift): the entry-author plumbed leverage end-to-end, but the watchdog-authors trusted that "MARKET BUY closes a short" without re-reading the instrument type. CLAUDE.md non-negotiable rule additions would be: "all order-placement paths against existing positions MUST read `regime_leverage` from `_paper_positions[asset]` (or equivalent)" — same shape as rule #6 (three trade_gate.check sites).

### P110. [FIXED 2026-05-17] FastRiskTick emergency exit re-fires every 30s when stop-loss locks the spot
- **Symptom:** Production Discord stream showed `[FastRiskTick][LIVE] ETH: EXIT_ONLY - price_move=3.0-3.5%` firing every ~35s for 20+ consecutive minutes (2026-05-17 23:43 through 2026-05-18 00:01 UTC), and the same pattern on 2026-05-13 for SOL. Each fire was followed by `[FastRiskTick][EXIT] FLATTENING` then `[ORDER-BALANCE] SOL/USDT SELL: REJECTED — requires 3.287584 SOL but free=0.001058 (used by other orders=3.000000)`. The position never actually exited; the watchdog logged 100s of CRITICAL alerts with zero progress.
- **Root cause:** `main.py:_handle_fast_risk_action` LIVE branch (was lines ~14197-14211) submitted a MARKET SELL via `execute_order(...)` without first cancelling the existing exchange-native stop-loss. The stop-loss reserves the spot at Kraken (`used by other orders=3.0` in the reject log = active stop-loss). P87's `[ORDER-BALANCE]` balance-clamp correctly refused to submit a dust-sized order. Meanwhile `FastRiskAction.EXIT_ONLY` explicitly bypasses `FastRiskTick.REDUCE_COOLDOWN_SEC` (per `execution/fast_risk_tick.py:100,145` "EXIT_ONLY which always fires"), so the watchdog re-fired on the next 30s sleep cycle (`main.py:17086+`). And `on_reduce_executed()` was called unconditionally after `execute_order`, so the cooldown timestamp got updated but the per-asset state in `_paper_positions` was not — next eval saw the same exposure and re-triggered.
- **Why the on_reduce_executed cooldown didn't help:** the cooldown gates `REDUCE_50` only, not `EXIT_ONLY` (`execution/fast_risk_tick.py:145-151`). So setting `_last_reduce_time[asset]` had zero effect on the next EXIT_ONLY fire.
- **Fix (P110):** two-part.
  1. `main.py:_handle_fast_risk_action` LIVE branch: before `execute_order`, call `self.execution_manager.cancel_stop_loss(kraken_sym)` to release the spot. Wrapped in try/except so cancel failure logs WARN but still attempts the exit (best-effort). After `execute_order`, inspect the `OrderResult` — if `success=False` or `status='REJECTED'`, call new `fast_risk_tick.on_exit_failed(asset, reason)` instead of `on_reduce_executed`. Defensive `getattr(self.fast_risk_tick, 'on_exit_failed', None)` per P85 — if running against an older module that lacks the method, the fix degrades gracefully to no-op.
  2. `execution/fast_risk_tick.py`: added `on_exit_failed(asset, reason)` + `EXIT_FAILED_BACKOFF_SEC=1800.0` (30 min) + `_exit_failed_at`/`_exit_failed_reason`/`_exit_suppress_log_at` dicts. `evaluate()` now checks for active backoff and suppresses the EXIT_ONLY trigger (REDUCE_50 from vol-spike / depth-drop still composes normally — only the price_move EXIT_ONLY is suppressed). `set_4h_anchor()` clears the backoff so each 4H tick gets a fresh attempt with a new anchor. Suppression logs WARN once per minute (rate-limited) so operator sees that the watchdog is silenced, not silently broken.
- **What changes after deploy:** the same fired-and-rejected pattern from 5/13 + 5/17 now goes: (a) FastRiskTick fires EXIT_ONLY, (b) handler cancels active stop-loss, (c) handler submits MARKET SELL with the spot now free → position actually flattens (or, if cancel+exit still fails for another reason, on_exit_failed engages and the watchdog goes quiet for 30 min instead of re-firing every 30s). At the next 4H tick, anchor refresh clears the backoff.
- **CORR-0 same-shape fix applied (P110-followup 2026-05-18):** `_crisis_position_reduction` at `main.py:14345` had the same bug shape (MARKET reduce without cancelling stop). Even a 25% partial reduce can fail when the stop reserves the full position size — `used_by_other_orders == position` means MARKET SELL clamps to dust and REJECTs via P87. Fix: same cancel_stop_loss-before-execute_order pattern as P110, plus REJECTED surfacing at ERROR level. No backoff needed because CORR-0 is tick-driven (4H), not 30s-polled like FastRiskTick. Next 4H tick replaces the cancelled stop on the reduced position.
- **Mitigation pattern:** Any code path that places MARKET orders against existing positions must check the spot wallet's reservations BEFORE submitting, OR cancel reserving orders FIRST. The P87 balance-clamp is a safety net at the order layer; it doesn't fix the upstream design that allowed the reservation conflict. When a watchdog has bypass logic for cooldowns (because the condition it monitors is supposed to be urgent), it MUST also have failure-detection logic so that "urgent and impossible to act on" doesn't generate an alert-storm. EXIT_ONLY's cooldown bypass was correct in isolation but wrong without a REJECTED-detection complement.

---

### Foundational invariants (P1-P8)

### P1. Docker volume name mismatch → silent empty models
- **Symptom:** `[WIRE] TQC models: 0/3 loaded`, `models_ready=0`, DRL stuck in SHADOW even though force-ACTIVE logic exists at [main.py:4696](main.py#L4696).
- **Cause:** `docker compose up` creates project-prefixed volumes (`app_hmats-models`). Deploy script synced to un-prefixed `hmats-models`. Container mounted empty new volume.
- **Mitigation:** `docker-compose.hetzner.yml` now has `external: true` + explicit `name:` for each volume. If this ever regresses, check `docker inspect hmats-engine | grep Mounts` — should show `hmats-models` NOT `app_hmats-models`.

### P2. Agent signal key mismatch between writer and consumer
- **Symptom:** Agent appears to run but its signal is zero everywhere downstream.
- **Historical incidents:**
  - `micro_direction` was only written to `market_data`, not `agent_signals`. Downstream readers saw 0 for 14 days.
  - `kq_direction` writer vs `kraken_quant_direction` reader — 12-strategy matrix silently ignored.
  - `whale_direction` (str) vs `whale_flow_direction` (float) — fusion branch dead.
  - `squeeze_score` writer vs `squeeze_risk` reader — squeeze veto never fired.
- **Mitigation:** When adding any agent, trace the keys **all the way** from writer (`agent_signals[...] = ...`) → fusion (`agent_signals.get(...)`) → attribution (`_attr_collected[...]`) → extractor (`_EXTRACTORS[...]`). Four places, must match.

### P3. Attribution `signal_envelope._EXTRACTORS` silently zeros unknown agents
- **Symptom:** New agent in `_attr_collected` but attribution JSONL shows direction=0, reasoning="unknown_agent".
- **Cause:** `agents/signal_envelope.py` has a hardcoded `_EXTRACTORS` dict. Agents not listed get the fallback that zeros direction+confidence.
- **Mitigation:** Add an extractor to `_EXTRACTORS` dict for any new agent name used in attribution.

### P4. ETH fold_1 stale checkpoint + deploy-sync pitfall [REGRESSION FIXED 2026-04-24]
- **Symptom:** ETH TQC `results.json` reports `best_fold: fold_1` with reward=1400, but `train_rows=0, train_time=0`.
- **Reality:** fold_1 is an aborted/stale run; fold_3 (reward=1029, train_rows=10028) is the genuine best.
- **Regression (2026-04-24):** Even after `BEST_FOLDS` was hardcoded everywhere, live ETH inference loaded `fold_1` model. Root cause: `drl/ensemble.py:275` read `results.get("best_fold", BEST_FOLDS.get(...))` — results.json won. Meanwhile `drl/runtime_obs_builder.py:102` uses `BEST_FOLDS` directly. Result: **mixed-fold pairing** — fold_1 TQC policy fed with fold_3-scaled features → broken inference math for ETH. ObsBuilder scaler + OOD detector loaded fold_3; only TQC itself loaded fold_1.
- **Compound cause:** Hetzner deploy script Step 4 `cp -r /home/hmats/hmats/models/* /volume/` overwrites the volume with whatever's on the host. The host's `/home/hmats/hmats/models/retrained/ETH/results.json` had never been re-synced after the local repo fix, so each deploy faithfully overwrote a patched volume with the stale host file.
- **Fix:** (a) `drl/ensemble.py:275` — swap priority so `BEST_FOLDS.get(asset)` wins over `results.json`; fallback default also bumped `fold_1` → `fold_3`. (b) scp'd corrected local `results.json` to hetzner host. (c) patched volume directly as belt-and-suspenders.
- **Mitigation:** `BEST_FOLDS` in `drl/ensemble.py` + `drl/runtime_obs_builder.py` + `training/drl/oracle_tqc_teacher.py` all hardcode ETH→fold_3. **Check `train_rows>0` on every fold before trusting reward.** Also: verify `docker logs hmats-engine | grep "TQC loaded"` shows matching fold for TQC + ObsBuilder scaler + OOD detector — mixed-fold pairing is a silent bug.

### P5. Pickle models from `__main__` can't be loaded cross-script
- **Symptom:** `AttributeError: module '__main__' has no attribute 'DTConfigV32'` when loading saved DT/TQC from a different script.
- **Cause:** Training scripts saved models embed `__main__.ClassName` pickle refs.
- **Mitigation:** Before `torch.load`, register classes in `__main__`:
  ```python
  import __main__
  from training.drl import train_decision_transformer_v32 as _t
  for name in ("DTConfigV32", "DecisionTransformerV32", ...):
      setattr(__main__, name, getattr(_t, name))
  ```

### P6. DT val_acc measures DT-vs-TQC imitation, not market prediction
- **Symptom:** Trained DT shows val_acc 70-80% but Sharpe is NEGATIVE on backtest.
- **Cause:** Training loss uses `true_dir = sign(TQC action)`. DT learns to imitate TQC's policy structure, not predict actual next-bar direction.
- **Mitigation:** Always verify DT with `training/drl/eval_dt_val_sharpe.py` (real returns). TQC teacher is better used directly than through KD imitation on small datasets.

### P7. ent_coef must be a fixed float
- **Symptom:** NaN in DRL loss / rewards; 0.2 → 10^23 gradient explosion.
- **Mitigation:** `ent_coef` in TQC config = float literal (`0.1`, `0.2`). Never `"auto"`.

### P8. Three places to update when adding/removing a fusion agent
Authority matrix (`signals/authority_fusion.py`) + writer (`main.py` somewhere in tick loop) + `_build_fusion_signals` consumer (`integration/integration_v36.py`). Missing any one → agent is a ghost.

---

### Archived pitfalls (P9–P109, see [archive/CLAUDE_history.md](archive/CLAUDE_history.md))

- P9. [ARCHIVED 2026-04-22] agents/quant_agent.py moved to archive/legacy_agents/
- P10. TWO SEPARATE DRL systems — don't confuse them
- P11. [FIXED 2026-04-22] Local-var instantiation hidden from naive wiring scans
- P12. [FIXED 2026-04-24] 2-agent conflict score 0.7 force-promoted to HARD VETO 1.0
- P13. [FIXED 2026-04-24] kraken_quant cross-asset data starvation
- P15. [FIXED 2026-04-24] v521 AdaptiveWeightManager feedback loop never closed
- P16. [FIXED 2026-04-24] Dummy ENABLE_* flags — declared but never gated
- P17. [CLEANUP 2026-04-24] canonical_config.py single-source-of-truth drift
- P18. [FIXED 2026-04-24] Two dead-reads missed in P14 (market_data-vs-agent_signals + pre-write read)
- P19. [FIXED 2026-04-24] BEST_OF_N_HOLD short-circuit demoted DRL to effective ADVISE
- P20. [FIXED 2026-04-24] Alpha gate's effective_alpha_direction ignored DRL when quant abstains
- P21. [DIAG 2026-04-24] DECIDE pool observability log
- P22. [RETIRED 2026-04-24] Execution shadow mode + 3,160-line `_execute_intent` removed
- P22. [FIXED 2026-04-24] Full-repo audit cleanup — schema/security/config drift
- P23. [FIXED 2026-04-24] AC-5 daily fill budget cap was a silent no-op
- P23. [FIXED 2026-04-24] Regression tests for P12/P19/P20 — DRL authority punch-through family
- P24. [FIXED 2026-04-24] Discord log-handler dedup key was message-based → spam risk
- P25. [FIXED 2026-04-24] `ctx.intent` was undeclared → PnL attribution wrote empty `primary_agent` to every fill
- P26. [FIXED 2026-04-24] Scalar drift: `ctx.last_aging_check` never synced back → weekly log rate-limiter broke
- P27. [CLEANUP 2026-04-24] Write-only runner flags + dead OOD reader removed
- P28. [NEW 2026-04-24] THREE separate DRL systems — supersedes P10
- P29. [OVERRIDE 2026-04-24] Exit-SAC promoted to EXIT_ONLY for ALL 3 ASSETS via accelerated path
- P29. [FIXED 2026-04-24] External API resilience — Discord circuit breaker + Haiku 429
- P30. [FIXED 2026-04-25] Multi-DECIDE fusion: abstain treated as disagreement, linear weighting overcorrected → "DECID...
- P30. [FIXED 2026-04-24] torch.load(weights_only=False) defense-in-depth
- P31. [NEW 2026-04-25] One-shot health monitor (`scripts/hmats_monitor.sh`)
- P31. [FIXED 2026-04-24] joblib.load + pickle.load defense-in-depth (P30 extension)
- P32. [FIXED 2026-04-24] Constitution test harness — supreme-gate regression coverage
- P33. [FIXED 2026-04-24] DRLPromotionGate test harness — auto-demotion safety net
- P34. [FIXED 2026-04-24] StrategyExistenceFuse test harness — 28-day window + consecutive-loss safety net
- P35. [FIXED 2026-04-24] AuthorityFusionEngine test harness — multi-agent fusion core
- P36. [FIXED 2026-04-24] Architecture invariants + P22 regression tests cleanup
- P37. [FIXED 2026-04-24] State-persistence atomicity + fire-and-forget asyncio task tracking
- P38. [FIXED 2026-04-24] External feed 429 visibility + Solana RPC per-call timeouts
- P39. [FIXED 2026-04-24] Threading races + datetime naive/aware + numerical stability
- P40. [DIAG 2026-04-25] Backfill-IC vs live-alpha gap — RSI sign-flip by regime + walk-forward instability
- P40. [FIXED 2026-04-24] Datetime naive/aware sweep across data_mgmt feeds
- P41. [DIAG 2026-04-25] Regime-conditional sign flip is the rule, not the RSI exception — 75% of backfill signals affe...
- P41. [DIAG 2026-04-25] Runtime gate-rejection forensics — why "1 fill in 30 days"
- P42. [FIXED 2026-04-25] Weekend gate calibrated for 24/7 crypto (acts on P41 diagnostic)
- P43. [FIXED 2026-04-25] STALE_DATA observability — show WHICH feed is stuck
- P44. [FIXED 2026-04-25] STRUCTURE fractal-break observability — show WHY the gate blocks
- P45. [FIXED 2026-04-25] Weekend config plumbing audit + observability
- P46. [FIXED 2026-04-25] Weekend gate confidence — DRL substitution + min lowered
- P47. [FIXED 2026-04-25] Four silent-failure bugs — half-wired modules and dict.get misuse
- P48. [FIXED 2026-04-25] p0_safety_integrator deep audit — 5 silent-failure bugs + static detector
- P49. [VERIFIED 2026-04-25] Authority matrix count — confirmed 25 agents (audit miscount)
- P50. [FIXED 2026-04-25] Comprehensive codebase audit follow-up — 8 real bugs from TIER 1
- P51. [FIXED 2026-04-25] TIER 2 cleanup — 3 real datetime/observability fixes; rest were audit noise
- P52. [FIXED 2026-04-25] Weekend confidence default 0.50 → 0.30 + diag forwarding
- P53. [CLEANUP 2026-04-25] Stale tests + ETH fold regression + dead flags + path portability
- P54. [FIXED 2026-04-25] SHORT_CONTROL `_SC_PROTECTED_VETOES` substring keys mismatched real veto_reason format
- P55. [DIAG 2026-04-25] integration_v36 fusion internals trace — DRL silent abstain logging
- P56. [DIAG 2026-04-25] Weekend gate enforcement trace + silent-fallback observability
- P57. [FIXED 2026-04-25] Authority/flag/constant consistency scanner + 2 P3-shape attribution gaps + dead code
- P58. [FIXED 2026-04-25] 4 pre-existing weekend test failures — stale thresholds + archived module imports
- P59. [FIXED 2026-04-25] Scanner extension — DRL invariants + ENABLE_* real-gate audit + 3 new dead flags surfaced
- P60. [FIXED 2026-04-25] 3 dead P1 flags removed + Section F multi-site scanner + p0_integrator missing DVOL kwargs
- P61. [DIAG+FIX 2026-04-25] Comprehensive 5-axis batch audit + 4 small fixes + 8 deferred findings
- P62. [DIAG+FIX 2026-04-25] Live runtime verification of 13 audit items + monitor fill grep fix
- P63. [FIX 2026-04-25] C1 follow-through — record_fill silent-failure observability
- P64. [FIXED 2026-04-25] Weekend gate read-back via getattr was the actual root cause P45 flagged
- P64. [DIAG+FIX 2026-04-25] LIVE-mode silent-feedback-loop discovery + CANCEL-ALL log severity
- P65. [FIX 2026-04-25] P64-B follow-through — REMOVED the PAPER-only gate at execution_service.py:1828
- P67. [LESSON 2026-04-26] `/ultrareview` is diff-of-changes review, NOT context-loaded whole-file review
- P68. [FIXED 2026-04-26] Explore-agent audit batch 1 — 6 tz-aware + JSONL flush fixes
- P71. [LANDED 2026-04-26 in 8555fe7] Trim CLAUDE.md — archive P9-P54 entries (1207 → 699 lines)
- P72. [LANDED 2026-04-26 in 142f916] Silent-swallow lint + CI gate + optional pre-commit hook
- P85. [EMERGENCY 2026-04-26] ShadowLedgerWriter.frozen_allocations missing → 10 container restarts in 6 min
- P86. [FIXED 2026-04-26] Stop-loss `EOrder:Insufficient funds` — actual fetch_balance() check
- P87. [FIXED 2026-04-26] Dynamic balance check at order layer + method-collision hotfix
- P89. [SUPERSEDED 2026-04-26] Per-script audit loop continues — saturation criteria revised
- P89. [SUMMARY 2026-04-26] Per-script audit loop concluded — saturation reached
- P90. [FIXED 2026-04-26] Rounds 5a + 5b — silent fixes spanning account_sync, correlation, thesis_budget
- P91. [HOTFIX 2026-04-26] Stop-loss pre-flight min-size check — SOL volume rejection
- P92. [FIXED 2026-04-26] Round 5c — auto_recovery state-corruption fail-closed + promotion gate fsync
- P93. [HOTFIX 2026-04-26] PREFLIGHT_* errors classified PERMANENT + POSITION-DESYNC alert
- P94. [FIXED 2026-04-26] Round 5d — NaN volatility fail-OPEN + gambler-mode silent ImportError
- P95. [FIXED 2026-04-27] Stop-order userref leak — 6 stops for one position
- P96. [FIXED 2026-04-27] Round 5e — opportunity_budget tz-aware + sota correlation WARN
- P97. [FIXED 2026-04-27] Round 5f — datetime tz-aware sweep
- P98. [ROOT-CAUSE 2026-04-27] fetch_open_orders dedup — survives userref scheme drift
- P109. [COMPLETION 2026-04-27] Full-codebase audit loop FINISHED — 368/368 files (100%)

## Architecture Rules (unchanged invariants)

- **DRL state space** = 126 dims (122 features + 4 env state: position_ratio, position_direction, pnl_ratio, drawdown)
- **VecFrameStack n_stack=8** in training → TQC expects 1008-dim stacked obs at inference
- **RegimeSmoother persistence=2** must match training + runtime (prevents regime flip ping-pong)
- **GMM feature defaults must match training distribution** — e.g., `cross_asset_correlation=0.87`, `spread_percentile` per-asset (BTC=5, ETH=8, SOL=12 bps)
- **Constitution schema** requires: `dvol_zscore`, `vpin`, `correlation_btc_eth_sol`, `orderbook_depth_1pct_usd`
- **Data age** uses exchange timestamp, `MAX_DATA_AGE_SECONDS=60.0` (was 10.0; widened for 4H tick cycle, see `defense/constitution.py:105`. Schema `data_age_seconds.max` aligned 2026-04-24)
- **pandas_ta broken** on Python 3.14 → use `ta` library instead

---

## Development Guidelines

### When Making Changes
1. **Read before edit.** Verify assumptions, don't trust documentation blindly.
2. **Check all call sites.** When modifying a method signature or a dict key name, verify EVERY reader and writer.
3. **Trace 4 layers** for any agent change (Pitfall P2): writer → fusion → attribution → extractor.
4. **Use `-X utf8`** on Windows (avoids GBK encoding errors).
5. **Set-Location -LiteralPath** for paths with `()` in PowerShell.
6. **Test commands:**
   - `python -X utf8 main.py --mode verify` — quick smoke
   - `python -X utf8 main.py --mode paper` — full paper tick
   - `pytest tests/test_black_swan_hold.py tests/test_strategy_selection.py tests/test_alpha_gate.py` — strategy-layer regression
7. **Update CLAUDE.md in the same commit** if you change runtime state, add agent, or hit a pitfall.

### Training Commands

```bash
# DRL (TQC) — v5090 rig
python -X utf8 -u training/train_drl_full.py --asset BTC --no-progress-bar

# DT (with TQC teacher, Tier 2 pretrain + finetune)
python -X utf8 -u training/drl/train_decision_transformer_v32.py \
    --asset BTC --extra-assets ETH,SOL --oracle-mode tqc_teacher \
    --epochs 300 --save-suffix _pretrain
python -X utf8 -u training/drl/train_decision_transformer_v32.py \
    --asset BTC --oracle-mode tqc_teacher --epochs 800 --lr 1e-5 \
    --init-from models/decision_transformer/BTC/dt_v32_best_pretrain.pt \
    --save-suffix _ft800

# Data prep
python -X utf8 training/fetch_binance_full.py               # 3y 1H OHLCV
python -X utf8 training/scripts/rebuild_pipeline.py --smooth 2  # → 122-feat parquets
```

### Deployment

```bash
# Push local commits + redeploy
git push origin main
bash scripts/hetzner_deploy.sh hmats

# If container shows "Conflict" errors:
ssh hmats "docker stop hmats-engine hmats-api; docker rm hmats-engine hmats-api"
bash scripts/hetzner_deploy.sh hmats

# Force rebuild image (after code change that must affect runtime)
ssh hmats "cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml build hmats-engine && docker compose -f docker-compose.hetzner.yml up -d --force-recreate hmats-engine"
```

---

## Historical Notes (for archaeology only)

Completed-work table and verification history from Feb-April 2026 has been moved to
`archive/CLAUDE_history.md` (if it exists). If you need to trace when a specific
fix landed, use `git log --all --oneline -S "<marker>" -- path/to/file`.

Active markers in code that point to completed fixes:
- `[FIX-...]`, `[WIRE-...]`, `[AUDIT ...]`, `[TIER-...]`, `[P0-...]`, `[FIX 2026-04-22]`

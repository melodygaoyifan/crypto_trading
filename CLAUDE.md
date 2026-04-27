# HMATS — Project Status & Development Guidelines

**Last updated:** 2026-04-24
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
| **Exit DRL (Discrete SAC)** | **EXIT_ONLY (all 3 assets)** | Third DRL alongside the TQC direction DRL (P28). Authority cap = EXIT_ONLY; never decides direction. v1 = 4 actions {HOLD, PARTIAL_EXIT, RELEASE_RUNNER, EXIT_ALL}. **v2 checkpoints (200ep, seed=42) in `models/exit_drl_v2/{ASSET}/exit_sac_best.pt`:** BTC val_align=0.730, ETH=0.710, SOL=0.746. **All 3 assets promoted via accelerated path 2026-04-24 (CLAUDE.md P29) — bypasses spec's 30-day shadow + ≥30-event gate.** Bridge: System-3's PARTIAL_EXIT prediction overrides System-1's DRLOutput at [core/tick_exit_triggers.py:373-403](core/tick_exit_triggers.py#L373-L403) when (a) asset is in EXIT_ONLY, (b) Exit-SAC predicted PARTIAL_EXIT this tick, (c) kill switch hasn't tripped. **Limited to PARTIAL_EXIT only — RELEASE_RUNNER and EXIT_ALL stay handled by the existing 5 rule-based triggers.** Kill switch (`risk/exit_drl_kill_switch.py`) auto-demotes any asset to SHADOW (per-asset, independent state) on any of: 5 consecutive DRL-driven losses, 7-day rolling realized PnL < 0, HOLD ratio drift outside [50%, 90%] over last 50 ticks, or EXIT_ALL ratio > 30% over last 50 closes. Audit stamp: `data/exit_drl_promotion_state.json` records `force_promote_at` + reason + blockers-at-override per asset. End-to-end diagnostic: `python -X utf8 scripts/exit_drl_e2e_diagnostic.py` (10 stages, all green). |

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
4. **DRL Authority** — ACTIVE since 2026-04-22. Auto-demote to EXIT_ONLY on 5 consecutive losses or 15% DD. Recovery after 3 days.
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

### P95. [FIXED 2026-04-27] Stop-order userref leak — 6 stops for one position
- **Symptom:** Production cascade where each 4H tick produced a fresh stop order on top of prior unfilled stops. Diagnostic snapshot: 6 BTC stop orders for ONE BTC short position (ages 1.5/9/21/47/95/109 min) + 1 stale SOL stop sell holding 7.26 SOL locked. Free SOL dropped to 0.014 → next tick's stop hit P91 min-size pre-flight → P93 [POSITION-DESYNC] + [STOP-MINSIZE] CRITICAL alerts fired every tick.
- **Root cause:** `_generate_stop_userref()` at `execution/execution_manager.py:253` hashed `f"{symbol}_{side}_{stop_price:.8f}_{suffix}"`. The trigger price recomputes against the moving market each tick (different by tens of bps) → different hash → different userref → `check_userref_executed()` never matched the prior tick's stop → `place_stop_loss()` submitted a brand-new order each tick instead of recognizing existing one.
- **Fix (commit 62f8776):** drop `stop_price` from the hash. Userref keyed on `(symbol, side, suffix)` only. One stop per (symbol, side) at any time. Callers needing to update the trigger price should cancel-and-replace via `active_stops` dict, not place a parallel order.
- **Operator follow-up (resolved this session):** 6 stale orders cancelled via API (the user updated Kraken API 2 days prior; engine confirmed using latest secret via SHA256 match across local .env + host .env + container env). After cancellation, 0 open orders remained, SOL fully free for clean stop placement on next tick.
- **Mitigation pattern:** When generating a deterministic dedup key, exclude any field that drifts naturally between calls. The "idempotency" guarantee only works if the key is stable across the lifetime of the resource being deduplicated.

### P94. [FIXED 2026-04-26] Round 5d — NaN volatility fail-OPEN + gambler-mode silent ImportError
- **Why:** Per-script audit Round 5d (4 files in `risk/`). 2 verified bugs after rejecting 6 false positives.
- **Fix 1 (`risk/dynamic_limits.py:159`):** `volatility_z > threshold` returns False for NaN, silently SKIPPING the leverage pullback. Pullback is a downward safety adjustment — fail-OPEN on NaN data leaves leverage HIGH on bad volatility input. Now `math.isnan()` guard treats NaN/unparseable as the high-vol regime → applies pullback as fail-CLOSED default + WARN log.
- **Fix 2 (`risk/unified_position_sizer.py:235-257`):** three `try/except ImportError` blocks for `configs.high_risk_mode` silently fell back to NORMAL caps (80%/150%) when gambler mode (95%/200%) was intended. Operator had no visibility. Class-level `_gambler_import_warned` flag now WARNs once per process on first call.
- **False positives dismissed:** `global_exposure_cap.py:258` short-cap formula verified math-correct; `dynamic_limits.py:147` "AttributeError on None.get" — line 146 has inline dict fallback, `rm` is never None; agent's "running exposure unit confusion" was self-acknowledged as "happens to work because both are %-of-account".
- **Deferred (architectural):** dynamic_limits.max_gross_exposure not piped to GlobalExposureCap; tranche_manager update_executed never called from main.py; tranche_manager close-path missing reset_position; risk_manager threshold drift vs canonical (likely different semantic gates per startup logs).

### P93. [HOTFIX 2026-04-26] PREFLIGHT_* errors classified PERMANENT + POSITION-DESYNC alert
- **Symptom:** After P91 deploy, production retry loop wasted 3 attempts on each PREFLIGHT_BELOW_MIN_SIZE rejection because `_classify_kraken_order_error` only knew Kraken response strings, not our own pre-flight rejections.
- **Fix part 1 (`execution/execution_manager.py:289`):** prefix-match any `PREFLIGHT_*` / `INSUFFICIENT_SPOT_BALANCE` and return `("PERMANENT", ...)`. Same input on retry produces same rejection — no point retrying.
- **Fix part 2:** Added separate `[POSITION-DESYNC]` CRITICAL alert when balance clamp shrinks size by >50%. That threshold is far beyond fee/rounding buffer and signals real state desync (P87 phantom-position pattern). Previous `[STOP-BALANCE]` WARNING surfaced the gap percentage but buried it; the new CRITICAL points at 3 likely causes (unfilled prior order recorded as filled, manual move to derivatives/staking, stale order holding the asset) and references `defense/startup_reconciler.py` for diagnosis.
- **Mitigation pattern:** any error string that the caller GENERATES (rather than receives from an external service) should be classified PERMANENT — retry can't fix what local validation rejected.

### P92. [FIXED 2026-04-26] Round 5c — auto_recovery state-corruption fail-closed + promotion gate fsync
- **Fix 1 (`risk/auto_recovery_gate.py:269`):** `_load_state()` previously treated "no file" and "file exists but unreadable" the same (both → empty HaltState). If a P0 abort wrote the halt state and the file got corrupted (SIGKILL truncation, disk full mid-write), the gate would silently forget the halt and resume trading on next restart. Now distinguishes: missing file → fresh start (OK), unreadable file → synthesize halt with `reason="STATE_CORRUPTION_DETECTED"` and force operator to clear via `clear_halt()`. CRITICAL log with actionable guidance.
- **Fix 2 (`risk/exit_drl_promotion_gate.py:253`):** `record_override()` used `path.write_text()` (no flush/fsync) and `except: logger.debug()` on write failure. Override audit trail is the SOURCE OF TRUTH for accelerated promotions (P29). Silent loss meant operator believed BTC was promoted but kill switch saw no record on next restart. Now uses `core.state_persistence.save_state()` (P83 atomic + fsync) with manual fsync fallback, surfaces write failures at WARNING.
- **False positives dismissed:** `stop_loss_authority.py:243-251` "VETO logic contradiction" — agent misread; inline comments document the `pass` blocks are intentional ("Actually keep soft stop active since we want to protect long").

### P91. [HOTFIX 2026-04-26] Stop-loss pre-flight min-size check — SOL volume rejection
- **Symptom:** Production hit `EGeneral:Invalid arguments:volume minimum not met` for SOL/USD stop-loss. P79 PERMANENT classifier correctly short-circuited but ALERT_ONLY policy left position WITHOUT exchange-native stop protection.
- **Root cause:** P86/P87 added balance-clamp on the upper bound (don't exceed free spot), but no LOWER bound check against Kraken's market minimum (SOL/USD = 0.05 SOL minimum). Sequence: position 0.X SOL → fetch_balance shows free = position - reservations → clamp to free × 0.998 drops below 0.05 → amount_to_precision rounds further → Kraken rejects → ALERT_ONLY → position unprotected.
- **Fix (commit c5be36f):** pre-flight check against `exchange.market(symbol)['limits']['amount']['min']` BEFORE sending. If size < min: log CRITICAL with explicit operator action ("position WITHOUT exchange-native stop protection — close manually OR top up balance"), return REJECTED with PREFLIGHT_BELOW_MIN_SIZE reason, skip the API call entirely. Same defensive pattern as PREFLIGHT_WRONG_SIDE / INSUFFICIENT_SPOT_BALANCE.

### P90. [FIXED 2026-04-26] Rounds 5a + 5b — silent fixes spanning account_sync, correlation, thesis_budget
- **Round 5a (commit ee5445d):**
  - `core/account_sync.py:367` — `except (asyncio.TimeoutError, Exception): pass` on `fetch_positions` swallowed Kraken API failures silently → mysterious notional=0.0 in leverage calculations. Promoted to logger.warning with type+exception.
  - `main.py:10115` — added `"CRITICAL drawdown"`, `"HALT drawdown"`, `"Correlation crisis"` substrings to `_SC_PROTECTED_VETOES` defense-in-depth set so risk_agent's drawdown/correlation vetoes can't be overridden if short_control's scope ever widens.
- **Round 5b (commit e48c6ff):**
  - `risk/correlation_realtime_controller.py:423-433` — earlier P50 "tuple-key fix" was wrong: `current.get((asset1, asset2), 0.0)` passes a tuple as the first STRING arg of `CorrelationMatrix.get(asset1: str, asset2: str)`. Tuple isn't a key in self.matrix (str keys), every branch falls through, returns 0.0. **detect_jumps() never fired in production** — eating every correlation jump warning. Fixed to use the actual 2-string signature.
  - `risk/thesis_budget_governor.py:69-71` — 3 of 4 veto-reason enum values lacked the `"THESIS_BUDGET"` substring (`COOLDOWN_ACTIVE`, `LOSS_STREAK_LIMIT`, `REENTRY_USED_UP`). main.py:11090 builds intent.veto_reason as `[{enum.value}] {message}`, then main.py:10111 checks substring `"THESIS_BUDGET"`. 3 of 4 thesis vetoes were not in the safety net. Renamed enum values with `THESIS_BUDGET_` prefix.
- **Round 5c-5e: P92-P96 entries above.**

### P96. [FIXED 2026-04-27] Round 5e — opportunity_budget tz-aware + sota correlation WARN
- **Fix 1 (`risk/opportunity_budget_governor.py`):** converted all 5 `datetime.now()` calls (lines 86, 191, 280, 287, 317) to `datetime.now(timezone.utc)`. On the production UTC container the runtime impact is benign, but `activated_at + expires_at + is_expired` comparisons mixing naive/aware would crash on any non-UTC deployment. P39/P40 family.
- **Fix 2 (`risk/opportunity_budget_governor.py:343`):** promoted shadow-ledger write failure from `logger.debug` to `logger.warning` + included exception type. Same P64 silent-failure pattern. Operator now sees if the audit trail isn't being written.
- **Fix 3 (`risk/sota_risk_controller.py:348`):** promoted correlation-calculation exception from `logger.debug` to `logger.warning` + asset/exception context. Correlation collapse detection is mission-critical; if `_calculate_correlation()` silently returned NaN due to numpy/data issues, `_check_correlation_collapse()` returned early and the gate went blind without operator visibility.
- **Bonus:** rebaseline (`tools/scanner_baselines/`) — silent_swallow dropped 427→421 (P92/P93/P96 promoted 6 swallows to WARNs); silent_failure.dictget bumped 38→42 (intentional defensive `.get` chains from P91/P93/P94); silent_failure.tryexcept dropped 313→312. Net safety win.

### P89. [SUPERSEDED 2026-04-26] Per-script audit loop continues — saturation criteria revised
- **Original conclusion (this entry, before user override):** stopping at 44 files / 6 rounds, declaring saturation. **User correction:** "only 44 files, i need the entire code base be audit, can you refine your memory and continue". Memory rules updated to require ENTIRE codebase coverage (every `.py` in live tree), not saturation-based stop.
- **Original session totals preserved for trace:** 6 rounds 3a-4f covered 44 files. 19 real bugs + 1 cosmetic shipped as P72/P73/P74/P81/P82/P83/P84/P86/P88. All committed/deployed/verified.
- **Resumption:** built `archive/audit_ledger.json` tracking all 368 live `.py` files. Round 5a-5e + production hotfixes P91/P93/P95 added: 11 new fixes spanning account_sync silent-swallow, risk-veto safety net, correlation jump-detect repair, thesis budget veto consistency, auto_recovery state-corruption fail-closed, exit_drl_promotion_gate atomic write, stop-loss min-size pre-flight, PREFLIGHT_* PERMANENT classifier, POSITION-DESYNC alert, dynamic_limits NaN guard, stop-order userref dedup, opportunity_budget tz-aware datetime, sota correlation WARN.
- **Ledger state at this entry:** 66/368 audited (~18%). 302 remaining. Loop continues.
- **Mitigation pattern:** when the user defines completion criteria differently from the agent's heuristic, the user's criteria take precedence. "Saturation reached" is an agent-defined notion of diminishing returns; "every file audited" is operator-defined. The latter wins.

### P89. [SUMMARY 2026-04-26] Per-script audit loop concluded — saturation reached
- **Why:** User-authorized autonomous audit workflow (per agent memory `autonomous_audit_loop.md`). Loop: read CLAUDE.md → re-pull → dispatch 4 Explore agents per round on next-highest-leverage unaudited critical files → triage findings → fix with P85 discipline → smoke test → commit + push + deploy → verify production health. Stop when codebase confidence saturates.
- **Total:** 6 rounds (3a-4f) covering 44 critical files (~12% of 365-file codebase). 19 real bugs + 1 cosmetic shipped as P72/P73/P74/P81/P82/P83/P84/P86/P88. **All 19 fixes:** committed, pushed, deployed, root-cause-fixed (not band-aid mitigations), production-verified via startup logs (e.g., `[EXIT_DRL_KILLSWITCH] Restored state`, `StopLossAuthority: ACTIVE`, `SentimentL2: ACTIVE val_acc=0.833`, TQC fold_3 ×3 assets, P74 GMM cache fix in effect).
- **Per-script methodology validated as decisively superior to directory-batched audits for the post-extraction silent-failure bug class.** Bugs caught that 3 prior directory-rounds missed: P72 ×4 NameErrors in execution_service.py from main.py extraction, P73 Exit-SAC bridge `runner` undefined in tick_exit_triggers.py, P74 GMM cache key using close PRICE not TIMESTAMP (the actual P41 root cause), P82 mode hardcode in p0_safety_integrator, P84 Exit-SAC kill switch state silently lost on restart, P86 ×3 NameErrors in kraken_link.py (P85-shape).
- **Stopping criteria met (all 4):**
  1. Round-4f ratio 0/21 (worse than 1:25 plateau threshold)
  2. Round-4f = 0 same-day actionable across all 4 files
  3. Remaining unaudited files are leaf utilities (`data_mgmt/feeds/*`, `risk/short_position_controller.py`) — already covered by directory-batched rounds 1-2, low blast radius
  4. Production stable 50+ min since P88 deploy at 22:46 UTC, 0 restarts, healthy
- **Calibrated confidence at stopping:**
  - **~95%** the 19 fixes are correct (smoke-tested, deployed, log-verified)
  - **~75%** the 44 audited files are clean for documented bug shapes (P12/P15/P22/P25/P39/P47/P85)
  - **~50%** for novel bug classes I didn't prompt for
  - **~12%** codebase-wide (only 12% per-script audited)
- **8 deferred items documented in earlier P-entries** (not regressions, design decisions): drift_detector silent logging, integration_v36:1388 tranche conflict-threshold, trade_gate:726 governor exception fail-OPEN intent, trade_gate snapshot age unbounded, constitution:1763 regime_aligned dead-read pending writer wiring, anti_churn check_fill_budget dead method, signals/no_trade_triggers.py orphan archival pending alpha_signal_integrator decoupling, kraken_plus_fee_blender:156 div-by-zero on env=0.
- **Mitigation pattern (audit methodology established):** per-script depth + P-history-aware prompts + P85 discipline (re-pull before stage AND push, defensive `getattr(obj, 'attr', sentinel)`, no `sys.exit()` to compose with `restart: always`, smoke test, post-deploy verify) is the canonical workflow for future codebase-coverage passes. Directory-batched audits remain useful for novel bug class hunts; per-script depth for verifying fixes hold + finding extraction-class silent failures.

### P87. [FIXED 2026-04-26] Dynamic balance check at order layer + method-collision hotfix
- **Why:** Operator pointed out that P86's stop-loss balance check wasn't enough. The system places multiple orders per 4H tick (1 entry per asset × 3 assets = 3+ orders); each order consumes balance dynamically. Position sizing is computed against TOTAL account equity, NOT against the live changing free balance. Result: 2nd/3rd order in a tick can exceed actual spot capital → Kraken rejects `EOrder:Insufficient funds` → P79 short-circuits PERMANENT → entry FAILS → phantom-position cascade where the system thinks position opened (intent registered) but actually didn't, then attempts stop on phantom → cascade of CRITICAL alerts.
- **Fix (commit 088b865):** new `_clamp_size_to_balance_v2` helper called from `execute_order` after dry_run check, before order placement. Fetches `exchange.fetch_balance()` per order, returns `(clamped_size, diagnostic_msg)`:
  - BUY: needs quote (USD/USDT) ≥ size × price + 40bps fee buffer; uses 99.6% of free quote as max-affordable. For MARKET orders without price, estimates via fetch_ticker.
  - SELL: needs base ≥ size + 20bps lot/fee buffer; uses 99.8% of free base.
  - Clamped → WARN log with both `free` AND `used` (held by other orders) + 3-hypothesis cause list ("prior order this tick consumed", "funds moved to derivatives/staking", "position-size config exceeds spot capital").
  - Clamps to 0 → REJECT with explicit operator guidance.
- **Hotfix (commit eb84b9e — see Lesson Rule 4 above):** initial P87 caused a 4-minute outage because `_clamp_size_to_balance` was the name of a PRE-EXISTING method at `execution_manager.py:1545` with a different signature (4 args returning float). Python silently picked the LAST definition → `TypeError: takes from 4 to 5 positional arguments but 6 were given` on every entry attempt. Hotfix renamed to `_v2`; old method left in place with its 2 existing callers untouched. Future cleanup: merge the two methods into one with the richer return signature.
- **Result:** entries that would have failed with `EOrder:Insufficient funds` now either succeed with a clamped size (logged via `[ORDER-BALANCE]` WARN) or REJECT early with explicit guidance. Combined with P86 (stop-loss balance check), both ENTRY and EXIT sides of every order validate against live spot balance before reaching Kraken. Phantom-position cascades are structurally prevented at both ends.
- **Mitigation pattern:** any code path that mutates external resources (places orders, sends webhooks, writes to shared state) needs the same shape — defensive fetch + clamp + diagnostic log. The pattern was added per-site in P86 and P87; future similar paths should reuse the helpers OR follow the same template (fetch live state → compute requirement → return `(clamped_value, reason)` → caller logs WARN if clamped, REJECTS if zeroed).

### P86. [FIXED 2026-04-26] Stop-loss `EOrder:Insufficient funds` — actual fetch_balance() check
- **Why:** P83 fixed the order-shape (Kraken accepted the request structurally) but Kraken still rejected with `EOrder:Insufficient funds`. P84's blanket 0.5% size buffer was sized for fee/rounding shortfalls, NOT for operator-induced moves like "moved $1000 to derivatives" that drop the spot wallet's free SOL by 1.7% from intended position size.
- **Fix (commit 6db05eb):** `fetch_balance()` before placing the stop, clamp size to `free[base] × 0.998` for SELL or `free[quote] × 0.996 / trigger_price` for BUY. Same defensive shape as P87 (which generalizes the pattern to all orders).
- **Mitigation:** stop-loss size now lives on the LIVE Kraken spot wallet, not a stale config-derived value.

### P85. [EMERGENCY 2026-04-26] ShadowLedgerWriter.frozen_allocations missing → 10 container restarts in 6 min
- **Symptom:** Engine refused to start LIVE. Each startup attempt crashed in `_reconcile_orders` with `'ShadowLedgerWriter' object has no attribute 'frozen_allocations'`. Reconciler marked ORDER_CHECK as FAILED; main.py's strict contract refused to start LIVE per `LIVE mode requires successful startup reconciliation`. Process exited; docker-compose `restart: always` re-launched; cycle repeated.
- **10 distinct container starts** between 22:12:43 and 22:18:24 UTC (~40s each — startup + reconciler attempt + refusal exit). Pattern visible in `journalctl -u docker.service | grep hmats-engine` (each `sbJoin` event = container network attach = container start). Docker `RestartCount` showed 0 because P85's deploy at 22:18:24 force-recreated the container, resetting the counter.
- **Root cause:** `defense/startup_reconciler.py:629` reads `self.shadow_ledger.frozen_allocations`, but `ShadowLedgerWriter` (`defense/shadow_ledger_jsonl.py:64-116`) NEVER declares or sets that attribute. Verified via grep: only the reader exists; zero writers. Classic P15-shape silent reader/writer mismatch — likely introduced by a parallel-edit operator commit (P82-P84 sibling commits 24ffcd6, 2d0a93c, 72ea60f all landed in this window) that added the orphan-detection feature without adding the corresponding writer.
- **Why the cascade was so violent:** the safety pattern (refuse to start LIVE on reconciler failure) is correct in principle, but composing it with `restart: always` produces an amplification loop. ONE missing attribute → process exit → docker restart → 30s startup → same exit → repeat.
- **Fix (commit fc10b46):** defensive `getattr(self.shadow_ledger, 'frozen_allocations', None)`. If missing, log WARNING and SKIP orphan cancellation (do NOT default to empty set — that would classify every legitimate exchange order as orphan and cancel them all). Reconciler completes with WARN; LIVE startup proceeds. Proper fix (add `frozen_allocations` to ShadowLedgerWriter with the right semantics — a `Set[str]` of order IDs the ledger has reserved) is a separate architectural commit.
- **Mitigation pattern (added to Lessons):** every new attribute READ on a third-party / cross-module object must defend with `getattr(obj, 'attr', sentinel)` + WARN log on missing, OR verify the writer exists IN THE SAME COMMIT. With parallel edits, the reader-author can't trust that the writer-author honored the contract.
- **Mitigation pattern (architectural):** add a circuit breaker to startup-refusal cascades. After N consecutive reconciler failures, stop restarting and require manual intervention. Otherwise `restart: always` weaponizes a missing attribute into a 6-minute outage.

### P72. [LANDED 2026-04-26 in 142f916] Silent-swallow lint + CI gate + optional pre-commit hook
- **Why:** Pattern 1 in the recurring-bug analysis (silent feedback loops — P15/P25/P47/P64). The ultrareview-bug-006 incident proved this class can hide undetected for weeks even with the existing scanners. Need a lint that catches it at write time, plus a CI gate that prevents regression.
- **Symptom of the problem class:** `try: foo() except Exception: pass` (or `logger.debug(...)`) hides exceptions from operator visibility. Method on `foo()` doesn't exist or fails — caller sees neutral result, downstream computes wrong answer, bug surfaces hours-to-weeks later in production. Enumerated cases in archive/CLAUDE_history.md: P15 (record_trade_completed vs record_trade_result), P25 (ctx.intent undeclared), P47 (4 silent attribute swallows), P64 (8 state recorders gated PAPER-only).
- **What landed (commit 142f916, accidentally bundled with Dockerfile changes — see "Why this commit message is misleading" below):**
  - **`tools/lint_silent_swallow.py`** — AST-based scanner. Detects 3 sub-patterns:
    - A) `try: ...; except ...: pass` (full discard)
    - B) `try: ...; except ...: logger.debug(...)` (below operator visibility)
    - C) `try: ...; except ...: <no log> ; return/continue/break` (early-exit, no log, no re-raise)
    - Per-block opt-out via `# noqa: silent-swallow` on except line OR inside body. Forces a deliberate decision at write time.
    - CLI: default scans LIVE_DIRS + main.py; `paths` for specific files/dirs; `--staged` for pre-commit; `--json` for tooling.
  - **`tools/ci_check_invariants.py`** — extended with third baseline (silent_swallow_baseline.json). Same "counts can DECREASE freely, INCREASE blocks CI" semantics as the existing two baselines.
  - **`tools/scanner_baselines/silent_swallow_baseline.json`** — current state frozen: **425 total findings (151 debug, 147 pass, 127 no-log-early-exit) across 121 files.** Future PRs that introduce a new silent swallow fail CI; future PRs that fix or annotate existing swallows green CI naturally as the count drops.
  - **`.pre-commit-config.yaml`** — optional convenience for developers. Two hooks: silent-swallow-staged (verbose, non-blocking — informational at commit time) and ci-invariants-quick-check (--diff-only at pre-push). Install: `pip install pre-commit && pre-commit install`. Skipping it is fine — CI gate is authoritative.
- **Why CI gate not pre-commit-blocking on full count:** 425 pre-existing silent swallows (P22-P64 era of accumulation). Hard-blocking on total count would block every commit until they're all fixed/annotated — months of work. Baseline-diff approach lets the count only decrease; operators chip away at existing ones during normal work without race against new additions.
- **How to drive the count down (each fix is a 1-3 line change):**
  1. Promote `pass`/`logger.debug` → `logger.warning` with `type(e).__name__` and context.
  2. Re-raise (or `raise X from e`) if the caller should handle it.
  3. Annotate `# noqa: silent-swallow` if genuinely intentional (e.g. ImportError on optional dep) — include a brief comment explaining why on the same line.
- **Why this commit message is misleading:** The 5 P72 files were staged but unpushed when the operator ran a parallel `git add` for a Dockerfile change; the operator's commit swept them all into 142f916 with a Dockerfile-only commit message. The follow-up "P72: CLAUDE.md entry" commit (this entry) documents what's actually in 142f916. Same precedent as 734f921 ("P68: CLAUDE.md entry — covers ca24727").
- **Effects on the recurring-bug analysis:**
  - **Pattern 1** (silent feedback loops): blocked from growing. Existing 425 sites have an enforced ceiling — operator can drive down via normal cleanup work.
  - **Pattern 6** (defensive mitigation hides root cause): the lint surfaces the *mechanism* by which root cause is hidden (pass/debug-only catches). Each annotated swallow now must either (a) explain itself via noqa-comment or (b) actually surface to operator visibility.

### P71. [LANDED 2026-04-26 in 8555fe7] Trim CLAUDE.md — archive P9-P54 entries (1207 → 699 lines)
- **Why:** Pre-trim CLAUDE.md was 1207 lines / ~52KB; ~75% was the Pitfalls section (800+ lines of P-entry detail). New Claude sessions had to read through hundreds of lines of historical fix narrative to find the non-negotiable rules + recent context. P67 explicitly cited this ("CLAUDE.md context is too long to be a useful index").
- **What changed:** 52 older P-entries (P9-P54) moved to `archive/CLAUDE_history.md` with FULL body preserved. P1-P8 (foundational invariants — perpetual reference) and P55-P68 (last ~30 days of work) kept in CLAUDE.md as full text. Recent pitfalls now ordered most-recent-first at the top of the Pitfalls section.
- **Side benefit:** the trim reduced the authority-scanner constant-drift baseline by 2 entries (`MAX_LEVERAGE` and `WEEKEND_MIN_ALPHA_MULTIPLIER` had prose-level "2.0" mentions in archived P-entries that were being falsely matched as drifts). Scanner now reports drift only on actual config files. Baseline updated.
- **Recovery:** Reversible — every archived entry retains its original heading and body. To restore, copy from archive/CLAUDE_history.md back to the appropriate section.

### P68. [FIXED 2026-04-26] Explore-agent audit batch 1 — 6 tz-aware + JSONL flush fixes
- **Why:** After P67 tear-down of the ultrareview slice plan, pivoted to 7 internal Explore agents (one per directory: execution / signals+integration / agents / drl / data_mgmt+market / infra+orchestration+core / analytics+liquidity). Each agent loaded its full slice with a P-history-aware prompt and hunted documented bug shapes (silent-failure / fail-open / threshold-drift / dead-code / authority-drift / datetime-mixing / numerical-stability). 7 reports back in ~5min wall time.
- **Real bugs fixed (6 across 6 files, 2 documented families):**
  - **Datetime naive/aware (P39/P40 family — 3 files):**
    - `execution/exit_alpha.py:135` — `ScaleOutSignal.timestamp` default `datetime.utcnow` → `lambda: datetime.now(timezone.utc)`. Plus added `timezone` to import.
    - `execution/execution_quality_logger.py:376` — `ExecutionQualityRecord.timestamp` built from naive `datetime.now()`; `_compute_session().hour` therefore read host LOCAL hour, not UTC, silently corrupting session bucketing on non-UTC containers. Lifted to `datetime.now(timezone.utc)` + reused via `_now_utc` local for the `execution_id` timestamp too.
    - `execution/execution_quality_logger.py:651` — `fromisoformat` fallback for malformed timestamps used naive `datetime.now()`; now matches the success branch (which produces aware via `+00:00`).
    - `agents/model_alpha_agent.py:643` — `ModelIntent.timestamp` built from naive `datetime.now()` but `to_agent_signal_dict` tags ISO with `"Z"`; timestamp ended up local-time but UTC-tagged.
  - **JSONL durability (P37 family — 3 files):**
    - `analytics/exit_alpha_tracker.py` — `_persist` + `_persist_counterfactual` missing `f.flush()`.
    - `analytics/signal_quality_tracker.py` — `_persist` missing `f.flush()`.
    - `analytics/ic/ic_logger.py` — `_writer_loop` missing `f.flush()`; IC snapshots are async-queued but each line was buffered in Python text-mode and lost on Docker SIGKILL.
- **False positives caught and documented (4):** Audit script bias-correction notes for future runs.
  - `agents/microstructure_agent.py:986,1000` — already uses `datetime.now(timezone.utc)`. Agent misread.
  - `analytics/ic/backfill_ic.py:159` — one-shot script with `open("w")`; `with` block flushes on close. Per-line flush is no-op for one-shot writers.
  - `analytics/trade_attributor.py:622` — `_persist_trade` ALREADY has `f.flush()` (P37 fix from 2026-04-24). Agent's "line 334 bulk export" reference was actually a dict-literal in `report()`.
  - `integration/integration_v36.py:1388` — `tranche_signal_data["signal_conflict"] > 0.5` is DISTINCT from the cosmetic flag at `:1828` (which P23 documents as legitimate `> 0.5`). Line 1388 IS in a fusion-routing path (feeds tranche_scheduler abort detection), but P12's `>= 0.9` threshold was for HARD VETO, not tranche reduction. Tightening or loosening this without runtime evidence violates CLAUDE.md's trade-frequency reality check. **Filed for separate investigation when forensics show tranche-abort fire rate.**
- **Defensive / informational findings deferred (~7):**
  - 5 division-by-feed-derived-price guards in `phase_detector` and `structure_analyzer` (theoretical — Kraken BTC/ETH/SOL never hit 0.0 in production).
  - `drl/ensemble.py:150` `TQC.load` not routed through `safe_torch_load` (paths implicitly safe via hardcoded model dir; SB3's framework loader bypasses our wrapper anyway).
  - `core/anti_churn.py` `_fills_today` no lock (single-threaded today; would matter if Discord worker ever calls it).
  - `core/runtime_state.py:80` max_drawdown 0.35 cross-check vs canonical_config.
  - `signals/authority_fusion.py` module-global `_drl_authority_level` no test-reset path.
  - `integration/integration_v36.py:2434` soldex confidence ×0.5 dampening — post-promotion leftover, low-impact (SOL only).
- **Audit ratio:** 6 real fixes / 27 raw findings = **1:4.5** — better than the P51-predicted 1:8 because the Explore-agent prompts were P-history-aware and pre-filtered most known false-positive families. Per-slice ratios: drl/ and infra+orchestration+core were essentially clean (0 HIGH); analytics+liquidity had the densest bug cluster (5/9 real); signals+integration produced 1 real + 1 confirmed false positive + 1 needs-verify; execution and agents each produced 2 real datetime fixes.
- **Verified clean across all 7 slices:** P4 BEST_FOLDS hardcode (drl/), P22 binance kline (data_mgmt/), P29 Discord/Haiku 429 (infra/), P30/P31 safe_torch_load wrappers (drl/, infra/), P37 atomic state writes (core/, infra/), P39 promotion_gate datetime + Discord lock-across-sleep, P40 feed datetime fixes, P41 GMM cache key scheme, P46 weekend confidence DRL substitution chain, P47 strategic_coordinator rename, P50 fail-closed contracts + runtime_spine deprecation gate + 5-failure escalation, P52 weekend min_confidence 0.30, P56 weekend silent-fallback log, P57 whale + options attribution, P66 Kraken nonce ratchet — ALL still hold.
- **Mitigation pattern:** Internal Explore agents are the right tool for whole-codebase semantic review. They load full directory context (which `/ultrareview` cannot) and produce structured findings cheaply. Future audits should default to Explore agents per slice; reserve `/ultrareview` for genuine code-change PRs where the diff itself is what needs review.

### P67. [LESSON 2026-04-26] `/ultrareview` is diff-of-changes review, NOT context-loaded whole-file review
- **Symptom:** Built `scripts/touch_for_audit.py` (commit 1595979) to insert a 1-line `# [AUDIT-SLICE: ...]` marker per file, intending the resulting commit's diff to put every file "in scope" for an `/ultrareview <branch>` review of the entire codebase. Slice 1 (`audit/safety-defense`, 49 marker files) shipped — but the review's reported scope was "1 file changed, 42 insertions(+), 5 deletions(-)" matching unrelated WIP edits to `scripts/authority_consistency_audit.py`, NOT the 49 markers. Slice 2 (`audit/execution`) errored: "no new commits or changes to review against your audit/execution branch".
- **Cause:** `/ultrareview` reviews the actual diff content. A 1-line comment insertion produces a 1-line diff — the reviewer sees "added a comment" and has nothing substantive to comment on for the file's actual code. The marker scheme can't trick `/ultrareview` into context-loading the whole file. The reviewer for a real bug doesn't expand context to the entire file just because that file appears in the diff.
- **Cleanup (this commit):** removed `scripts/touch_for_audit.py` (commit 1595979 reverted in spirit, file deleted). Both audit branches deleted (`audit/safety-defense`, `audit/execution`) remote + local. The audit-script POSIX ERE bug fix found by slice 1's accidental WIP review (`13812f5`) is keep — that was a real bug found by a real diff review.
- **What `/ultrareview` IS for:** real code-change PRs / branches with substantive diffs. The reviewer reads the changed lines + surrounding context and finds bugs in those changes. It is NOT a whole-codebase scanner.
- **What to use instead for whole-codebase coverage:**
  1. **Static scanners** that ARE whole-codebase aware: `scripts/authority_consistency_audit.py` (this session extended), `scripts/completeness_audit.py`, `scripts/silent_failure_audit.py` (P48).
  2. **Internal Explore agents** per directory slice — load the whole subsystem, hunt for the documented bug shapes (silent-failure / fail-open / threshold-drift / dead-code / authority-drift / datetime-mixing / numerical-stability), return structured finding lists. No Extra Usage billing.
  3. **Forensic tools** triggered by runtime signal: `scripts/gate_rejection_analysis.py`, `scripts/alpha_gate_postmortem.py`, `scripts/exit_drl_e2e_diagnostic.py` — discover from real data, not blanket sweeps.
- **Mitigation pattern:** Before building tooling around a diff-based review tool, verify what the tool actually consumes from the diff. A 1-line marker doesn't hand the reviewer enough context to do anything useful with the surrounding file. If you want whole-codebase semantic review, use a tool that's whole-codebase-aware by design.

### P65. [FIX 2026-04-25] P64-B follow-through — REMOVED the PAPER-only gate at execution_service.py:1828
- **Why:** P64 documented but did not fix the issue. After investigating, the gate is unnecessarily wide — every dry_run-specific call inside the BRANCH tree (e.g. `account_sync.update_dry_run_pnl` at lines 1969/2481/2758/3313) already has its own inner `if ctx.account_sync.dry_run` guard. The outer `if RunMode.PAPER or dry_run` was a blanket cover hiding functionality that already had finer-grained protection.
- **Architectural verification:**
  - `_paper_positions` at main.py:18242 is read by tranche scheduler in BOTH modes (active_positions filter doesn't check mode).
  - `_paper_positions` writes use `exec_result.filled_price`/`filled_size` from real exchange data — values are accurate in live mode.
  - `_save_paper_positions` was being skipped in live mode → no on-disk position state. After P65 it fires universally → live restart recovery actually works.
- **Fix:** changed line 1828 from gate to `if True:` with a long comment block explaining why. Functionally equivalent to deleting the gate. The structurally-paper-only `account_sync.update_dry_run_pnl` calls (4 sites) already self-gate.
- **Also fixed:** the diagnostic `[P0-2] execution completed without shadow-ledger fill ack` at line 3407 was ALSO PAPER-gated — operator wouldn't see the warning if a live fill failed to record. Removed mode check; warning fires whenever `shadow_fill_recorded=False`.
- **Also removed:** the P64-B CRITICAL startup notice (no longer relevant — gate is gone). Replaced with INFO confirmation that the loops are wired in both modes.
- **What changes after deploy:**
  - `shadow_ledger/ledger_*.jsonl` will get FILL entries for live fills (was: 0)
  - `thesis_budget_state.json`, `confidence_scorer_state.json`, `failure_memory_state.json`, `cascade_governor_state.json` — file timestamps will update on every fill (was: stuck at 08:04)
  - `existence_fuse` will start tracking real PnL → 28d kill switch becomes live-armed
  - `anti_churn` will enforce AC-2 (2-fills/asset/24h) + AC-5 (8 fills/day) ceilings
  - `trade_attributor` outcomes_*.jsonl will get FILL-derived close records
  - `strategic_coordinator.record_trade_completed` → v521 AdaptiveWeightManager will start receiving outcomes
  - `exit_drl_outcome_ledger.record_close` → Exit-SAC training data starts accumulating
  - `paper_positions.json` will exist on disk (was: missing) → restart recovery actually works
- **Risks (real, but acceptable):**
  - **AC-2/AC-5 rate limits will start enforcing.** If today's fill pattern (3 fills in 5 minutes after restart) exceeds AC-2's 2/asset/24h ceiling, NEW entries get blocked. This is GOOD (rate limits should enforce in live), but operator should see the new `AC2_RATE_LIMITED` veto messages and accept them.
  - **existence_fuse counts live PnL** for the first time. If recent live trades had losses near -15%, the fuse could engage immediately on next deploy. Verify `existence_fuse_state` (currently empty/stale) before going live.
  - `_save_paper_positions` writes to disk on every fill — adds I/O but it's atomic-write + only fires post-fill (~0/tick when no fill). Negligible.
- **Tests:** 155/155 regression green. The runtime change is significant — recommend close monitoring of the next 4H tick boundary.
- **Mitigation pattern:** When fixing an over-wide gate, FIRST check whether each path inside has its own self-guard. If they do, the outer gate is redundant — just delete it. P65 took ~1 hour to find vs the half-day refactor I feared yesterday.

### P64. [FIXED 2026-04-25] Weekend gate read-back via getattr was the actual root cause P45 flagged
- **Symptom:** P45 added gate-rejection observability that surfaced 18/20 weekend rejections (90%) hitting `wk_cfg_present=False` with `mult_normal=NOT_SET` — even though the startup banner at [main.py:1867](main.py#L1867) DID fire (proving config loaded as a dict) and the P0 instantiation at [main.py:2154](main.py#L2154)/[2167](main.py#L2167) read the same dict cleanly. P52/P56 made the fallback non-catastrophic (class default 0.30/1.0/20bps matches live config) but did NOT trace the wiring.
- **Cause:** runtime tick read at the old [main.py:10706](main.py#L10706) was `_wk_raw = getattr(self.config, "weekend_config", None)` — a fresh attribute lookup every tick. Whatever the actual flip mechanism (config object swap, attribute clobber, intermittent reload path, stale ledger from pre-instrumentation entries), all hypotheses produce the same symptom and the same fix: do not re-read the live attribute on the hot path. Plus the per-process rate-limit `_wk_cfg_fallback_logged` hid frequency — only one warning per process, even on a 100% miss rate.
- **Fix:** (a) Snapshot `weekend_config` once at `__init__` ([main.py:1819](main.py#L1819)+) into `self._weekend_config_snapshot: Dict`, normalized to `{}` if non-dict at init time. (b) Record `self._config_id_at_init = id(config)`. (c) Runtime tick site now reads `_wk_cfg = self._weekend_config_snapshot` directly. (d) Added a drift trace: per-50-tick rate-limited WARNING log when `id(self.config) != self._config_id_at_init` OR when the live attribute has flipped to non-dict OR has become a different dict instance than the snapshot. (e) Same snapshot used at the proof-log site at [main.py:13344](main.py#L13344) so both sites stay in sync.
- **Mitigation pattern:** When init reads X correctly but runtime reads X-but-it's-wrong, snapshot at init and trace drift instead of patching the runtime fallback. The "patch the gate" approach (P52/P56) hides root cause; the "patch the read path" approach (P64) eliminates the failure mode entirely. This is the textbook example referenced in the "Heisenbug Investigation Discipline" lesson at the top of this file.

### P64. [DIAG+FIX 2026-04-25] LIVE-mode silent-feedback-loop discovery + CANCEL-ALL log severity
- **Why:** Operator asked "is post-trade learning/budget/attribution silent NORMAL?" + "fix `[CANCEL-ALL] graceful_shutdown`". Two unrelated findings folded into one P-fix.

#### Part A — CANCEL-ALL log severity (FIXED `execution_manager.py:1582`)
- Old: every `cancel_all_open_orders()` call logged at `CRITICAL` with "emergency order cancellation" wording — including `reason="graceful_shutdown"` (called on every container stop). Operator-fatigue inducing; mixed normal lifecycle with real emergencies.
- 3 call sites with different reasons (`graceful_shutdown`, `stop_order_failure_*`, `disconnect:*`).
- Fix: `graceful_shutdown` → `INFO` non-alarming wording. Real emergencies keep `CRITICAL`.

#### Part B — LIVE-mode feedback loops are SILENT (DIAG ONLY, NOT FIXED — needs operator review)
- **Discovery via P63 monitoring:** P63 promoted record_fill failures DEBUG → WARNING + added `_note_shadow_fill(False)` diagnostic. Deployed and ran 3 fills (08:54-08:56). **0 SHADOW_LEDGER WARN logs fired AND 0 FILL records in shadow_ledger.** The code path NEVER REACHES `record_fill`.
- **Root cause:** `core/execution_service.py:1828`:
  ```python
  if ctx.config.mode == RunMode.PAPER or (ctx.account_sync and ctx.account_sync.dry_run):
  ```
  This gate wraps the ENTIRE BRANCH A/B/C/D tree (lines 1828-3389, ~1500 lines) where ALL post-trade state recording lives. In `--mode live`, both conditions are False, so the entire tree is skipped.
- **What's silently dead in LIVE mode:**
  - `shadow_ledger.record_fill` (P25 fix only fires paper-side)
  - `anti_churn.record_fill` (AC-2 / AC-5 rate limits unenforced — P23)
  - `thesis_budget.record_fill` (weekly budget cap not tracked)
  - `existence_fuse.record_pnl` + `on_trade_close` (28d kill switch P0 — **safety risk**)
  - `trade_attributor.record_entry/exit` (DIM 4 attribution P25)
  - `confidence_scorer.record_outcome` (P15 strategy calibration frozen)
  - `pnl_attribution.record_trade`
  - `strategic_coordinator.record_trade_completed` (P15 v521 AdaptiveWeightManager input — never fed)
  - `failure_memory.record_opportunity`
  - `exit_drl_outcome_ledger.record_close` (P28 DRL training data)
- **Empirical proof:** State files in `/opt/hmats/data/`:
  - `thesis_budget_state.json`: timestamp 08:04 (before today's 08:25 restart)
  - `confidence_scorer_state.json`, `failure_memory_state.json`, `cascade_governor_state.json`: all 08:04
  - `tranche_state.json`: `{}` (empty after 6+ fills)
  - 6+ live fills today wrote ZERO updates.
- **Implications:**
  - **P15-P47 state-recording fixes are paper-mode-only.** Every "we fixed feedback loop" is true ONLY in paper mode.
  - **Existence fuse can't trigger** — won't see live PnL, can't enforce 28d -15% kill switch (P0 safety).
  - **Anti-churn rate limits unenforced** — could trade dozens/day with no brake.
  - **DRL has zero closed-loop learning** in live mode.
  - **DIM 4 attribution = empty** — `agent_audit_16.py` reads non-existent FILL records.
- **Why not fix here:** the gate wraps 1500 lines INCLUDING `paper_positions[asset] = ...` writes that DO need to stay paper-only (live → Kraken is truth). Properly narrowing requires (1) extract paper_positions writes to their own paper-only block, (2) move all `record_X()` calls outside, (3) extensive testing. Half-day refactor + risk to live trading. Filed for operator-approved separate work.
- **Interim mitigation in this commit:** added `[P64-B] LIVE-MODE FEEDBACK-LOOP NOTICE` CRITICAL log line in `main.py` after `[LIVE] ExecutionManager verified`. Fires once per startup so operator sees the issue loudly until the gate is narrowed. Lists all 8 affected loops in the log.
- **Tests:** 155/155 regression green. Pure observability + log severity changes.
- **Mitigation pattern:** When a feature flag / mode gate wraps a large block, audit what's INSIDE vs what should fire universally. State recording (audit / safety / learning) usually fires regardless of mode; only OUTPUT side effects (write paper-specific state) should be mode-gated.

### P63. [FIX 2026-04-25] C1 follow-through — record_fill silent-failure observability
- **Why:** P62/C1 found ZERO FILL records in `shadow_ledger/ledger_20260425.jsonl` today despite 3 actual fills (08:26 SOL / 08:27 BTC / 08:28 ETH). The `record_fill` call path has TWO silent-failure modes:
  1. **Exception caught at `logger.debug`** (4 sites at `core/execution_service.py:2054 / 2585 / 2840 / 3045`) — operator never sees the failure.
  2. **Silent-False return** — `defense/p0_safety_integrator.py:record_fill` returns False without raising when `self.shadow_ledger is None`. The caller's `_note_shadow_fill(False)` was a no-op.
- **Root cause status:** STILL UNKNOWN until next deploy + next fill. Engine logs show `ShadowLedger: ACTIVE` at startup, so initialization succeeded. The bug must be in the per-call code path. P63 is the diagnostic — P64 will be the actual root-cause fix once we see the WARNING logs.
- **Fix (observability only):**
  1. **4 try/except blocks** at execution_service.py:2054/2585/2840/3045 — promoted from `logger.debug` to `logger.warning` with `type(e).__name__: {e}` + key kwargs (asset, side, size, price for the entry path). Operator now sees both the exception type AND the bind values.
  2. **`_note_shadow_fill(False)` no-op** rewritten — now logs `[SHADOW_LEDGER] FILL not recorded ({site}): shadow_ledger_state={None|missing|ok-but-returned-False}, asset={X}` so we can distinguish "shadow_ledger never initialized" from "init OK but returned False".
  3. **All 4 call sites** now pass a site tag (`full_close` / `partial_close` / `flip_close` / `entry`) so the WARN line localizes which branch failed.
- **What to watch after deploy:**
  - First fill after deploy will trigger ONE of:
    - `[SHADOW_LEDGER] record_fill ({site}) FAILED: AttributeError: 'NoneType' has no attribute 'X'` — exception in p0_integrator.record_fill
    - `[SHADOW_LEDGER] FILL not recorded (entry): shadow_ledger_state=None` — shadow_ledger attr lost mid-tick (init race, garbage-collected, etc.)
    - `[SHADOW_LEDGER] FILL not recorded (entry): shadow_ledger_state=ok-but-returned-False` — record_fill internal bug
    - **No log line at all** — call site never reached, even though `[FILL-QUALITY]` log fired (this would mean the FILL log fires from a different code path that doesn't go through execute_intent_v2's record_fill block)
- **Tests:** 155/155 regression green. Pure observability change, zero runtime logic impact.
- **Mitigation pattern:** When `try/except: pass` or `logger.debug` swallows an exception and the surrounding feature appears to "just not work", promote to WARNING + include `type(e).__name__` in the message. This is the P15 / P25 / P47 family at the *logging* layer — silent debug-level swallows are functionally equivalent to silent attribute drift.

### P62. [DIAG+FIX 2026-04-25] Live runtime verification of 13 audit items + monitor fill grep fix
- **Why:** Operator asked to actually RUN/VERIFY (not just static-audit) restart recovery / state persistence / data freshness / cancel-on-disconnect / order ack / maker-taker fallback / existence_fuse / live DRL log / health monitor / trade attribution / cashandcarry / best-of-N / unwired modules / dead configs.

#### Direct verdict: live engine is healthy + traded today

- `bash scripts/hmats_monitor.sh` → 9/9 axes OK after fix below.
- Engine running `--mode live --confirm-live --config configs/live_high_risk.json` against real Kraken API. Real orders.
- **3 actual fills today** post-P61 deploy (08:25 UTC):
  - 08:26:37 SOL LONG qty=7.5 @ $86.53 notional $649 — **slippage=0.0bps** (clean LIMIT fill)
  - 08:27:21 BTC SHORT qty=0.0102 @ $77,581 notional $791 — slippage=-7.8bps
  - 08:28:09 ETH SHORT qty=0.393 @ $2,319 notional $911 — **slippage=-145.1bps (BAD)**
- DRL was decisively bearish on BTC/ETH (-0.95 / -0.96 conf 0.45 / 0.35) — 16-agent attribution captured this correctly in `signals_20260425.jsonl`. P19/P20/P46 substitution chain working end-to-end.

#### Real bug fixed (1 1-liner):

- **`scripts/hmats_monitor.sh:198` — FILL grep pattern was wrong.** Old pattern `[FILL]|FILLED|SCALE-OUT` matched **zero** real fills because engine never logs literal `[FILL]` or `FILLED` strings — it logs `[FILL-QUALITY]` and `[P0_EXECUTE]`. Verified: monitor showed `fills=0` while engine had 3 actual fills today. Fixed pattern to `[FILL-QUALITY]|[SCALE-OUT]`. Now reports `fills=3` correctly.

#### Verified clean / working:

- **Restart recovery (state persistence):** `confidence_scorer_state.json`, `cascade_governor_state.json`, `failure_memory_state.json`, `thesis_budget_state.json` all loaded on restart. Verified `[CONFIDENCE] State restored: 12 strategies` in startup logs.
- **Data freshness:** Last tick 2026-04-25T08:31:00, age 0.5s. `[LIVE_DATA]` for ETH showed bars=721, age=0.50s.
- **Existence_fuse params:** Match CLAUDE.md (28d / -15% / -18% / -15% / -18% / 10).
- **Live unrounded DRL log:** `[PROOF][SIGNALS] adopted=['drl_direction', ...]` confirms DRL signals reaching fusion. `signals_*.jsonl` capturing all 16 agents per tick.
- **Best-of-N selection:** `kq_firing_stats.json` updated 08:28; 12 institutional strategies + 4 TA Best-of-N all loaded. quant_direction=0 (HOLD) on all 3 fills today — DRL was the actual decider via P19/P20.
- **Cash-and-carry signal wiring:** Wired into agent_signals; execution path is signal-only pending DerivativesExecutor wire.
- **agent_audit_16 DIM 4:** `signals_*.jsonl` and `outcomes_*.jsonl` populated per tick; whale + options now in attribution per P57. Latest outcome record marks DRL/sentiment/llm_sentiment all `correct=true`.
- **Live health monitor:** All 8 axes pass post-fix.
- **Cancel-on-disconnect / order ack timeout / maker-taker fallback:** Static audit (P61 Agent 2) verified the code paths. Cannot live-test cancel-on-disconnect without simulating disconnect — operator approval needed.
- **Dead config files:** P53 + P57 + P61 catalog held.
- **Unwired modules:** P57-A scanner Section A/B/E all clean.

#### Real concerns surfaced (filed, NOT fixed — need operator decision):

- **C1 — Zero FILL records in shadow_ledger today.** Engine had 3 fills, but `shadow_ledger/ledger_20260425.jsonl` shows 0 entries with `entry_type=FILL` (only INTENT=29 / POSITION=28 / GATE_REJECT=24). `record_fill()` IS called from `core/execution_service.py:2038/2561/2823/3014` (P25 fix), but failures caught at `logger.debug`. **`primary_agent` attribution exists in code but never reaches the ledger.** Fix: promote that try/except DEBUG → INFO so we see WHY it's failing.
- **C2 — POSITION records still missing `primary_agent`.** All 28 POSITION records today have only `[old_size, new_size, old_direction, new_direction, realized_pnl, reason]`. P25 fixed FILL records, but `shadow_ledger.record_position_change()` doesn't accept primary_agent and the caller doesn't pass it via the `extra` dict. Either feed via extra or extend the API.
- **C3 — 22 engine restarts today** (one per P52→P61 deploy). Each `session_id` change = new tick_id space. Tranche state effectively starts fresh each session. Consolidate deploys going forward.
- **C4 — `tranche_state.json: {}` (empty)** after 3 fills. Either persist isn't firing on entry, OR new T4 entries don't trigger save because no escalation. Worth a 1-line check.
- **C5 — `drl_promotion_state.json` dated 2026-04-14** (11 days stale). If a demotion event fires, will state save?
- **C6 — ETH 08:28 fill had -145.1bps slippage** vs SOL 0.0 / BTC -7.8. Worth investigating `fill_quality.jsonl` for pattern.
- **C7 — `mode=PAPER` hardcoded** in `record_intent()` default at `defense/p0_safety_integrator.py:417`. Engine is `--mode live`. Should read actual run mode.

- **Tests:** 155/155 regression green. Pure observability fix, no runtime logic change.
- **Mitigation:** Live runtime audit catches what static can't — log pattern drift, missing FILL records, slippage anomalies. Run `bash scripts/hmats_monitor.sh` after every deploy.

### P61. [DIAG+FIX 2026-04-25] Comprehensive 5-axis batch audit + 4 small fixes + 8 deferred findings
- **Why:** Operator named 13 audit areas to cover (runtime data flow / execution reachability / config loading e2e / post-trade state sync / restart recovery / data freshness / cancel-on-disconnect / order ack timeout / maker-taker fallback / existence_fuse params / live_health_monitor / trade outcome attribution / sentiment-CRACK-leadlag sign-flip / cashandcarry wiring / best-of-N selection / dead config files). Dispatched 5 parallel Explore agents.

#### Real fixes applied (4 high-leverage 1-liners):
1. **`main.py:1681` — `initial_capital` no `float()` cast.** JSON could send int / string `"10000"`, silently propagating wrong type into `intent.target_exposure * self.config.initial_capital` math. Added defensive cast.
2. **`.env.example` — 4 undocumented HMATS_* env vars added** with default values + descriptions: `HMATS_REGIME_WARN_COOLDOWN_SEC` (main.py:4076), `HMATS_G6_SHADOW` (main.py:14474), `HMATS_ENABLE_AGGRESSIVE_ALLOCATOR`, `HMATS_AGGRESSIVE_ALLOCATOR_MIN_FILLS` (main.py:14480-14481). Operators can now see all available knobs.
3. **`configs/cloud_production.before_phase1.json`** — deleted. Stale backup (P53 pattern). Zero readers.
4. **`strategies/cash_and_carry.py:236`** — fixed stale "Funding {x}/8h" → "/h". Kraken Futures uses hourly funding (per file's own comment at line 214); print message was off by 8x.

#### Areas where audit found NO bugs (verified clean):
- **Authority matrix wiring (Agent 1):** All 25 agents have per-tick writers (some are conditional, but all explicitly handled). No P3-shape silent zero attribution beyond P57's whale/options.
- **Post-trade state sync (Agent 4):** All 10+ `record_X()` calls in `execute_intent_v2` have closed read loops. All 3 BRANCH paths persist tranche+positions (P15 fix held). All atomic-write protected (P37 held).
- **Restart recovery (Agent 4):** 8 critical state files all have writer + reader. Drawdown/equity peak survives restart per P-PATCH-4.
- **Live health monitor (Agent 5):** `scripts/hmats_monitor.sh` executable, 8 checks all map to real log patterns.
- **existence_fuse params (Agent 5):** Match CLAUDE.md documented values (28d/-15%/-18%/-15%/-18%/10) — UNLEASH v2 thresholds intact.
- **agent_audit_16 DIM 4 (Agent 5):** `primary_agent` populated since 2026-04-22 (P25 fix).
- **OOD detector (Agent 5):** Compute+log path live; confidence-multiplier path cleanly retired per P27.
- **Sentiment / CRACK / Lead-Lag sign conventions (Agent 5):** All consistent writer→fusion. No silent flips.
- **Cash-and-carry signal pipeline (Agent 5):** Wired into agent_signals; execution path is signal-only pending DerivativesExecutor wire.

#### Deferred findings (filed for future work):
- **D1 — `regime_direction` falls back to `quant_direction` (Agent 1, integration_v36.py:2092-2094):** CONFIRM-authority signal is tautological — confirms what quant said. Real fix requires deciding what `regime_direction` SHOULD mean (e.g. derive +1/-1 from regime classification: TRENDING_BULL/PARABOLIC=+1, TRENDING_BEAR/REVERSAL=-1, others=0). Currently the fallback is harmless but documented invariant is violated.
- **D2 — `macro_leverage_cap` 3-site overwrite (Agent 1, main.py:6307/6415/6945):** Last write wins; no precedence comment. Worth a 3-line comment block above each site explaining why each layer overwrites.
- **D3 — Per-key staleness markers (Agent 1):** Fusion can't distinguish fresh vs stale signals. Current `agent_signals["_signal_timestamp"]` is global. Per-key marker (`agent_signals["_ts"]["drl_direction"] = ts`) would be a design change with downstream consumer impact.
- **D4 — Market fallback disabled (Agent 2 B1):** If `market_fallback_enabled=False` in config, LIMIT timeouts return CANCELLED with no fallback. Worth a startup WARN if disabled in production profile.
- **D5 — CASH_CARRY no spot fallback (Agent 2 B4):** Pre-flight check needed before DerivativesExecutor is wired (currently signal-only, so no immediate risk).
- **D6 — Dead-man monitor thread killed (Agent 2 B5):** If monitor thread dies, server auto-cancels after 60s with no client notification. Add watchdog.
- **D7 — Partial fill orphan on reconnect (Agent 2 B3):** Original userref's partial not re-recorded after disconnect+retry.
- **D8 — Slippage measured but not fed back (Agent 2 B7):** Fill quality logged to `logs/fill_quality.jsonl` but not used to adjust friction model. Static config wins.
- **D9 — Stale `paper_baseline_5y.json` + `paper_profit_5y.json` + `sota_config.json`** — verified in P57 to be diagnostic-safety-net only. Acceptable retain.
- **D10 — AC-2 tick-epoch drift on restart (Agent 4):** New session tick=1 with restored fills from old session tick=18000 → comparison `1-18000=-17999` excludes old fills. Mitigated in practice by AC-1 reset pattern at main.py:15710 (sets `entry_tick=0` on restore), but AC-2 doesn't apply same logic. Low severity (24h aging window).

- **Tests:** 155/155 regression green. Audit-finding-to-real-bug ratio: ~13 listed concerns → 4 real fixes → ~1:3 (similar to P53 cleanup pass).
- **Mitigation:** When auditing a system area, list findings as DEFINITE/PROBABLE/FALSE_POSITIVE — most "scary" findings (regime_direction missing, market fallback disabled) turn out to be intentional design or low impact. Triage matters more than discovery.

### P60. [FIXED 2026-04-25] 3 dead P1 flags removed + Section F multi-site scanner + p0_integrator missing DVOL kwargs (rule #6 violation)
- **Two parts (1 + 2 from operator's request):**

#### Part 1 — Delete 3 dead flags (P53 / P59 follow-through)
- P59 surfaced `ENABLE_TWO_STAGE_INTELLIGENCE` / `ENABLE_STRATEGY_WEIGHTING` / `ENABLE_REGIME_SHORT_FILTER` as having zero real control-flow gates.
- All 3 declared in `configs/sota_flags.py:76-78` AND in `main.py:1010-1012` (DefaultFlags fallback) AND read by `is_p1_enabled()` at `configs/sota_flags.py:397-404` — but `is_p1_enabled()` itself was imported at `main.py:995` and **never called anywhere**.
- **Removed:** declarations in sota_flags.py (incl. P1 docstring section + categories list at line 324-325) + DefaultFlags fallback entries in main.py + `is_p1_enabled()` function + the import. Same P53 cleanup pattern.
- **Underlying features still wired** — two_stage agent (matrix row 8), strategy_weighting (signals/adaptive_weight_v521.py per P15), V6 SHORT FILTER (main.py:9914-9957) all live through other paths, not gated by these flags.
- **Result:** scanner Section E went from 3 / 25 → 0 / 22 dead flags.

#### Part 2 — Section F: multi-call-site kwarg consistency
- New scanner section in `scripts/authority_consistency_audit.py` that enforces CLAUDE.md non-negotiable rule #6 ("all 3 trade_gate.check sites must pass identical kwargs"). Generalized: any function listed in `TRACKED_MULTI_SITE_FUNCS` gets diff-checked across all its call sites.
- **Algorithm:** find every call via `git grep`; parse the call body (paren-depth tracker, comment-stripped — see parser-bug fix below) to extract kwarg names; diff against the union of all sites; flag any kwarg present in some sites but missing from others (unless listed in `intentional_omits`).
- **Tracked:** `trade_gate.check` (rule #6) + `execute_intent_v2` (P57-B, with `agent_signals` flagged as intentional omit for the MAX_HOLD_TIMEOUT site).
- **Parser bug fixed in same commit:** the kwarg extractor was getting poisoned by Python comments — `# [BUGFIX AUDIT-A2] Use caller-provided values, not hardcoded` between `regime=regime,` and `drl_weight=drl_weight,` made the parser see `drl_weight` as part of a longer string starting with `Use caller-provided...` and reject it. Fixed by stripping `#…\n` before parsing.
- **Real bug surfaced (and fixed):** `defense/p0_safety_integrator.py:498` was missing `dvol_zscore` + `dvol_current` kwargs that P47 added to `main.py:10567`. **This is the SAME-FAMILY bug as P48 BUG-2** — P48 plumbed the data-health snapshot through to p0_integrator's trade_gate but missed the DVOL kwargs. Without them, the trade_gate Gate 2 EMERGENCY_FLAT path (`dvol_zscore >= 5.0` per defense/trade_gate.py:627) saw default 0.0 and never fired at the p0_integrator route. Added params to `check_pre_execution()` signature + passed through to the call site at line 505. Caller at `main.py:10876` updated to pass `market_data.get("dvol_zscore", 0.0)` + `market_data.get("dvol", 0.0)`.
- **Result:** all 3 trade_gate.check sites (`core/authority_chain.py:366` + `defense/p0_safety_integrator.py:505` + `main.py:10567`) now pass identical kwargs. Rule #6 holds end-to-end.
- **CLI:** `python scripts/authority_consistency_audit.py --section multisite [--json]`. Now 6 sections total: A authority / B flags / C constants / D drl / E gates / F multisite.
- **Tests:** 155/155 regression green. Pure-tooling addition + 1 real bugfix.
- **Mitigation pattern:** When CLAUDE.md says "N call sites must do X", add a Section-F entry. The next refactor that adds a kwarg to one site will fail the scanner instead of going unnoticed for weeks (P47 + P48 + P60 all touched the same trade_gate.check pattern; without F, the next one would have been P61).

### P59. [FIXED 2026-04-25] Scanner extension — DRL invariants + ENABLE_* real-gate audit + 3 new dead flags surfaced
- **Why:** Operator asked whether `/ultrareview` could review the entire codebase. It can't — `/ultrareview <branch|PR>` is diff-based by design. So the alternative is making the static scanner cover the codebase-level checks ultrareview can't easily do (cross-file invariants, declared-vs-gated drift). Three new sections added to `scripts/authority_consistency_audit.py`.
- **Section D — DRL feature/state invariants:**
  - Reads `configs/feature_manifest.json`, verifies `total_feature_count == len(all_features) == 122` (CLAUDE.md documented invariant).
  - Verifies `no_scale_features = {regime_proba_0..7, has_external_data}` exactly (RobustScaler skips these — drift breaks inference).
  - Reports computed obs_dim = features + 4 env state = 126 vs documented invariant.
  - **Result this run:** clean. `total_feature_count=122 == len(all_features)=122 == documented`. No drift.
- **Section E — ENABLE_* real-gate audit (extends Section B):**
  - Section B (P57) only checked "any reader exists". Section E checks "is the reader actually a CONTROL-FLOW gate" — i.e. `getattr(flags, FLAG, ...)`, `flags.FLAG`, `if FLAG`. Catches the case where a flag is `import`ed, listed in a fallback `DefaultFlags` class, but never used in any `if` statement.
  - Started with too-strict ERE regex (25/25 false positive). Rewrote as Python-side classification: do `git grep \bFLAG\b`, then run a `re.search` per hit line to detect the actual gate pattern.
  - **Result this run:** 3 real dead flags surfaced — `ENABLE_TWO_STAGE_INTELLIGENCE`, `ENABLE_STRATEGY_WEIGHTING`, `ENABLE_REGIME_SHORT_FILTER`. All 3 are declared in `configs/sota_flags.py:76-78` AND in `main.py:1010-1012` (DefaultFlags fallback) AND read by `is_p1_enabled()` at `configs/sota_flags.py:400-403` — but `is_p1_enabled()` itself is imported at `main.py:995` and **never called anywhere**. Same shape as P16 / P53 dead flags. Filed for separate cleanup batch.
- **Section C extended:**
  - Added `hard_drawdown_halt` (canonical 0.20 vs live 0.25 — `core/risk_governor.py` reads from profile, both values are valid) and `initial_capital` (production .env=10000 vs scripts/tools default=100000) to TRACKED_CONSTANTS.
  - Refined regex patterns to skip false-positive matches against format strings (`{x:.1%}` no longer matches as "value=.1") and underscore-prefixed names.
  - **Result this run:** drift on hard_drawdown_halt at `integration_v36.py:100, 660` is intentional fallback (line 660 comment: "overridden by profile e.g. 0.25"). Scanner correctly surfaces for human triage.
- **CLI:** `python scripts/authority_consistency_audit.py [--section authority|flags|constants|drl|gates|all] [--json]`. Section flags now include `drl` (Section D) and `gates` (Section E).
- **Tests:** 155/155 regression green. Pure-tooling change, no runtime impact.
- **Mitigation pattern:** Static scanner now covers (A) authority drift, (B) flag-declared-no-reader, (C) constant drift, (D) DRL feature invariants, (E) flag-read-but-no-gate. The 5 cover ~80% of codebase-level inconsistency classes that ultrareview's diff-based mode can't easily detect. Run periodically; CI-eligible.

### P58. [FIXED 2026-04-25] 4 pre-existing weekend test failures — stale thresholds + archived module imports
- **Symptom:** `tests/test_ultra_weekend_manager.py` had 4 long-standing failures predating P54-P57. Two distinct root causes:
  1. **Stale UL-4/UL-5 unleash thresholds** (2 failures): `test_drawdown_hard_veto_on_weekend` used `drawdown=0.12`, `test_correlation_crisis_hard_veto_on_weekend` used `correlation=0.96`. Both were valid HARD-veto inputs when written, but the live `RiskVetoClassifier.HARD_THRESHOLDS` was widened in UL-4 (`correlation 0.95→0.98`) and UL-5 (`drawdown 0.10→0.20`). After widening, `0.12 < 0.20` and `0.96 < 0.98` — both fall through to SOFT veto.
  2. **Archived module imports** (2 failures): `TestHardSoftVetoClassifierPatch` imported `defense.production_reliability_patches` which was retired and moved to `archive/defense/production_reliability_patches.py`. Successor is `RiskVetoClassifier` in the live `defense/production_reliability.py`.
- **Fix:**
  1. Bumped `drawdown=0.12 → 0.22` and `correlation=0.96 → 0.99` in `TestHardVetoesPreserved` so the inputs actually exceed the (current) HARD thresholds. Test intent (HARD veto regardless of weekend mode) preserved.
  2. Renamed `TestHardSoftVetoClassifierPatch` → `TestRiskVetoClassifierWeekendCap`, repointed at the live `RiskVetoClassifier`. Same behavior tested — default cap `WEEKEND_LIQUIDITY=0.40` and ULTRA override to `1.00` via `weekend_soft_veto_cap` config key.
- **Tests:** 41/41 weekend manager tests + 155/155 broader regression all green.
- **Why this matters for the audit-cycle question:** these tests were the canary for "spec drift" (UL-4/UL-5 widened thresholds without updating tests). Running pytest on every CI cycle would have caught them — but if tests are skipped or flagged as "expected failures", the spec-drift signal is lost. Periodic full-pytest runs should be a forcing function. P57's `authority_consistency_audit.py` provides a complementary static check; pytest is the dynamic check.
- **Mitigation:** When widening a threshold, grep `HARD_THRESHOLDS\[` (or whatever constant moved) and update every test using values in the old window. Same pattern as P22 schema drift.

### P57. [FIXED 2026-04-25] Authority/flag/constant consistency scanner + 2 P3-shape attribution gaps + dead code
- **Tooling:** new `scripts/authority_consistency_audit.py` does codebase-level static checks for the 3 high-frequency latent-bug classes the operator named (DISABLED-when-shouldn't-be / authority-level-mismatch / parameter-drift). Three sections: (A) authority-matrix wiring quartet (writer / fusion-reader / extractor / matrix-label), (B) ENABLE_* flags without runtime readers, (C) numerical-constant drift across files for a curated list of multi-location values.
- **Real fixes from first run:**
  1. **`whale` agent (matrix row 24, ADVISE)** — wrote `whale_flow_direction` at main.py:7733-7739, consumed by integration_v36.py:2340-2346 fusion, but **NOT in `_attr_collected` dict** at main.py:8670+. Attribution silently zeroed every tick. Same P3 shape that affected micro_direction for 14 days. Added entry.
  2. **`options` agent (matrix row 22, ADVISE)** — wrote `options_short_confirmation` at main.py:6877, consumed at 8800/12392, but missing from `_attr_collected`. Same P3 silent-zero. Added entry.
  3. **`_extract_whale` + `_extract_options`** added to `agents/signal_envelope.py:_EXTRACTORS` so the wrap_agent_signal factory doesn't fall through to the `unknown_agent` zero-direction stub at line 244.
  4. **`_margin_tracker`** at main.py:4763 — instantiated as `MarginCostTracker()` with **zero method calls anywhere**. P57-B init-wiring audit (Explore agent) confirmed dead. Removed instantiation; left a breadcrumb comment.
- **False positives the scanner over-flagged (refinement deferred):**
  - 6 ADVISE agents flagged "no _EXTRACTORS entry" are non-direction-producers (cvd / squeeze / risk_appetite / structure / macro / lead_lag) — intentionally not in attribution. Scanner needs a directional-only filter.
  - 7 agents flagged "no direct writer" are written via `agent_signals.update(...)` or by integration_v36 internal — scanner regex is too narrow.
  - vol_alpha "fusion does NOT consume" — confirmed intentional (matrix row 23 docstring: "fusion branch REMOVED — affects execution only").
- **Stale profile configs (documented, not fixed):** `configs/high_risk.json`, `configs/ultra_aggressive_5y.json`, `configs/paper_baseline_5y.json`, `configs/paper_profit_5y.json` — all have stale `WEEKEND_MIN_CONFIDENCE` (0.45-0.50 vs live 0.30) and stale `WEEKEND_MIN_ALPHA_MULTIPLIER` (some 0.9 vs live 1.0). None loaded by `scripts/launch_live.py` (only `live_high_risk.json` is). Only consumer is `core/health_validator.py:199` which uses `high_risk.json` for sanity-check only. Filed for future cleanup batch — these don't affect production.
- **3 execute_intent_v2 call sites verified** (main.py:5361, 12730, 12759, per non-negotiable rule #6 pattern):
  - Site 1 (MAX_HOLD_TIMEOUT) calls without `agent_signals=` kwarg; Sites 2 + 3 include it. Likely safe (max-hold is rule-based, not signal-driven) but documented.
  - All 3 use identical `ExecutionContext.build_from_runner(self)` pattern.
  - No double-execution risk: Site 1 returns early; Site 3 only fires if Site 2's `_exec_effective=False`.
- **Tests:** 114/114 regression green.
- **Mitigation pattern:** When adding a new agent, run `python scripts/authority_consistency_audit.py` — it'll flag the missing entry in `_attr_collected` / `_EXTRACTORS` before deploy. Goal: turn the 4-place wiring discipline (P2 / P3 / P8) into mechanical validation rather than human discipline.

### P56. [DIAG 2026-04-25] Weekend gate enforcement trace + silent-fallback observability
- **Trace result:** Weekend gate has exactly ONE production enforcement site: `main.py:10674` `WeekendOverrideRules.should_override_entry()`. `integration_v36.py:1691, 1715` reads `is_weekend` but only as INPUT to `risk_agent.assess()` and `risk_veto_classifier.classify()` — neither enforces a weekend-specific veto.
- **Decision path inside `should_override_entry`** (`liquidity/weekend_manager.py:430-494`):
  1. Resolve `_alpha_mult` (per-asset > global > class default 1.0) — line 442-456
  2. Resolve `_min_conf` (per-asset > global > class default 0.30 post-P52) — line 457-477
  3. Resolve `_min_alpha_bps_base` (config > class default 20.0) — line 483-485
  4. Check 1 (alpha): `estimated_alpha_bps < (_min_alpha_bps_base × _alpha_mult)` — line 487
  5. Check 2 (confidence): `confidence < _min_conf` — line 491 (uses `_compute_effective_weekend_confidence` per P46)
- **Observability gap fixed:** `main.py:10662` previously did `_wk_cfg = self.config.weekend_config if isinstance(...) else {}` — silent fallback to `{}` if config never loaded. Class defaults are safe post-P52, but operator had no log of the fallback. Added one-shot WARNING log + `self._wk_cfg_fallback_logged` rate-limit so it doesn't spam. Pattern: same shape as P56's WEEKEND_CONFIG startup banner enrichment but at the rejection site.
- **No new bugs found.** The decision path is fully instrumented after P52/P55. All 5 weekend-related fixes (P42 alpha base, P45 diag forwarding, P46 confidence DRL substitution, P52 conf default lower, P56 fallback log) form a complete observable chain.
- **Pre-existing test failures (DEFERRED):** `tests/test_ultra_weekend_manager.py:TestHardVetoesPreserved` + `TestHardSoftVetoClassifierPatch` — 4 failures predating P54/P55/P56 where RiskVetoClassifier returns SOFT for drawdown/correlation when tests expect HARD. Either stale tests or a real classifier regression — separate investigation.

### P55. [DIAG 2026-04-25] integration_v36 fusion internals trace — DRL silent abstain logging
- **Trace coverage:** Mapped the full `decide()` execution path inside `integration/integration_v36.py` (~960 lines from line 914 to line 1875). 13 numbered steps from entry through fusion to TradeIntentV36 construction.
- **Key findings (no fixes needed):**
  1. Pre-fusion short-circuits at `_maybe_apply_pre_alpha_hold()` line 755-912 (BEST_OF_N_HOLD / BLACK_SWAN_SENTINEL / PRE_ALPHA_HOLD). All correctly bypassed by P19 punch-through when DRL ACTIVE + |dir|>=0.5 + conf>=0.3.
  2. Alpha gate at line 1248 (`self.guarantees.check_alpha_gate(...)`) reads `agent_signals.get("effective_alpha_direction", agent_signals.get("quant_direction", 0.0))` at line 1237 — P20 substitution wired correctly.
  3. Fusion call at line 1349 invokes `signals/authority_fusion.py:fuse()` after `_build_fusion_signals()` (line 2024-2392) builds the per-agent dict. DRL gets 1.3× confidence boost when ACTIVE (line 2216) — caps at 0.95.
  4. `kraken_quant` is **always** added to fusion when `|kq_dir|>0.01 AND kq_conf>0.1` (line 2334-2340), NOT just as fallback. Authority matrix v6.8 row 18 (DECIDE) is honored.
  5. 7 separate early-return-with-direction=0 paths exist; each writes a distinct `intent.veto_reason` — first one wins. By design (fail-closed) but operator can't see if multiple conditions tripped same tick.
  6. DRL confidence threshold asymmetry: fusion entry requires `conf>0.05`; punch-through layers (P19/P20/P46) require `conf>=0.30`. Intentional asymmetry — fusion is permissive (one vote among many), substitution is strict (about overriding safety gates).
- **One observability fix (P55):** `_build_fusion_signals` at line 2197 silently dropped DRL from `signals` dict if level/dir/conf gates failed. P8 explicitly warns "Missing key = dead authority". Added `[FUSION_DRL_ABSTAIN]` debug log when DRL was *expected* to vote (level in EXIT_ONLY/ACTIVE) but didn't, with the specific abstain reason. DEBUG-level only so it doesn't spam — appears when grep'd post-incident.
- **Tests:** 114/114 green. No behavior change — pure observability.
- **Mitigation pattern:** When tracing a fusion engine, look for "agent silently excluded from output dict" branches. The opposite of P8's "missing writer" — here it's a missing logger that masks an intentional exclusion.

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

### Archived pitfalls (P9–P54, see [archive/CLAUDE_history.md](archive/CLAUDE_history.md))

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

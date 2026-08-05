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
| **Coinbase US Perp (Phase 2)** | **Phase A SHADOW ACTIVE / Phase B DUAL_VENUE LIVE (BTC+ETH+SOL)** | v5.1 derivatives venue = Coinbase Derivatives Exchange US Perpetual-Style Futures (`BTC=BIP-20DEC30-CDE`, `ETH=ETP-20DEC30-CDE`, `SOL=SLP-20DEC30-CDE`; nano contracts 0.01/0.1/5.0; INTX `-PERP-INTX` is US-restricted, do NOT use). **Phase A (read-only, LIVE):** flag `coinbase_routing_enabled=true`; 4H heartbeat logs `[COINBASE-SHADOW]` parity + `[COINBASE-SLEEVE]` positions/buying-power. Trade key on volume `/opt/hmats/data/.coinbase_key.json` (`COINBASE_KEY_FILE`), `coinbase-advanced-py==1.8.3`. **Phase B (order routing): ACTIVATED ALL-3 2026-06-13** — `data/coinbase_routing_state.json` = {phase:DUAL_VENUE, coinbase_assets:[BTC,ETH,SOL]} → `core/execution_service._coinbase_routed`=True for all 3. **Two-sleeve, NOT a fork**: the `execute_intent_v2` Coinbase fork is a NO-OP per P141. ⚠️ **CORRECTED 2026-08-04 (P155): Kraken does NOT "still trade all 3."** P152 (landed later the same day as this sentence was written) skips every NEW Kraken entry for a Coinbase-routed asset that is flat, and P140-A1 had flattened all 3 on 2026-06-12 — so with all 3 routed, Kraken can only ever *unwind legacy spot*, of which there is none. Kraken directional trading has been structurally zero since ~2026-06-13, **by design**; the Coinbase sleeve is the sole directional driver; the Coinbase sleeve runs in parallel as the SOLE driver of its own positions. Isolated separate-sleeve `exchange/coinbase_sleeve.py`: venue-authoritative reconcile, 1-contract/asset cap + 15% sleeve-drawdown halt, per-tick `manage_to_signal` driver (opens/flips/**flattens-on-hold when |dir|<0.15**). **Activation validated live 2026-06-13:** reconcile OK, risk baseline $3,805, all-3 routed True, SOL short auto-flattened to 0 on dir=-0.10 (independent read-only reconcile confirmed POSITIONS={} — no orphan, P141 exit-mgmt working). **Toggle coverage (operator-run via `!`):** `scripts/coinbase_set_assets.sh {BTC,ETH,SOL|SOL|""}` writes routing state + restarts; `""` → inert. **Risk caveat:** zero PnL evidence on the directional sleeve yet — widen with eyes open, revert to SOL-only or "" if BTC/ETH misbehave. Operator-run scripts (auto-mode blocks the agent from live orders/activation): `scripts/coinbase_{probe,shadow_compare,test_order,sleeve_validate,manage_validate,flatten,set_assets}.py/.sh`. See P141 + `docs/COINBASE_MIGRATION_PREP.md` + `docs/COINBASE_ENGINE_INTEGRATION_PLAN.md`. |

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

### P155. [FIXED 2026-08-04] ~7.5 weeks of zero trades stayed undiagnosed because every alert on the path was either misdirecting, silenced, or measuring the wrong venue
- **Trigger:** operator asked why there had been no trade since 2026-06-12. Live evidence: `[HEALTH_T3] CRITICAL: SOL intent actionable — BLOCKED 312 consecutive ticks — system cannot trade! Check VETO_CHAIN logs for root cause.`
- **What 312 proves.** `_t3_intent_actionable` resets the streak whenever the intent is actionable **OR** `quant_strategy_id == 'hold'`. So 312 consecutive blocked ticks means SOL produced a **real non-hold directional signal every single tick** and the intent never became actionable — this is NOT a "no signal" problem. And because `_consecutive_blocked` is in-memory, 312 × 4h ⇒ **~52 days of unbroken uptime** (see the P154 update above).
- **Layer 1 — Kraken is structurally dead by design, and CLAUDE.md said otherwise.** P140-A1 flattened all Kraken positions 06-12; Phase B routed all 3 assets to Coinbase 06-13; P152 then skipped every new Kraken entry for a routed+flat asset. Net: Kraken can only unwind legacy spot, of which there is none. The Coinbase sleeve is the sole directional driver. **The header table still claimed "Kraken spot path is byte-identical and still trades all 3"** — written earlier the same day, falsified by P152 hours later, and never corrected. Anyone reading the docs to explain the silence was told the opposite of the truth. Corrected in the table above.
- **Layer 2 — the alert named the wrong subsystem.** `TradeIntentV36.is_actionable` is a 3-clause conjunction (`not veto_active AND |direction| > dir_thresh AND target_exposure > exp_thresh`), but T3's message said only *"Check VETO_CHAIN logs"*. When the real blocker is a collapsed `target_exposure` or a sub-threshold direction, **the veto chain is completely silent** — so the operator greps a clean log, finds nothing, and concludes the alert is noise. 312 ticks of that.
- **Layer 3 — the trade counter measures a venue that cannot trade.** The 4H Discord heartbeat's `Trades: N` counts only `_dashboard_asset_runtime[asset]["execution_status"] == "FILLED"`, and that field has **exactly one writer** — the Kraken `execute_intent_v2` path. The Coinbase sleeve never sets it. Post-Phase-B this number is structurally 0 forever, and it was being read as "the system traded 0 times."
- **Layer 4 — the sole order path was silenced three ways.** `[COINBASE-MANAGE]` logged only non-`NOOP`/`NOT_READY` results, so a permanently idle sleeve (halted / stale reconcile / target already met) emitted **nothing at all** — silence identical to healthy. Its `except` logged at DEBUG. And the whole block sits inside the heartbeat `try`, whose handler said **`"[HEARTBEAT] Discord push failed"`** — so a crash in the only Coinbase order path was reported as a Discord problem.
- **Layer 5 — a real state bug feeding it (`main.py` ~L6377).** `if abs(_qd) > 0.05: self._last_quant_directions[asset] = _qd` made the dict a **high-water mark, not a current reading**: once an asset printed a strong direction the entry was frozen there forever, because decay below 0.05 skipped the write instead of updating it. Two consumers read it as live — `_compute_crack_weight` cross-asset alignment (a long-dead direction kept voting) and the Coinbase sleeve driver (`~L18039`), which converts it straight into a position target. Now written unconditionally; safe because CRACK-3 applies its own `|d|>0.1` filter downstream, so widening cannot manufacture alignment — it only stops expired directions persisting.
- **Fix (P155), diagnostics + one state bug; no trading-threshold changes.** (a) `PerTickInvariantChecker._actionable_blocker()` names **every** failing clause with its value and the veto reason (`VETO_ACTIVE` / `WEAK_DIRECTION` / `ZERO_EXPOSURE`), thresholds read off the intent itself so it cannot drift from `is_actionable`; wired into both the CRITICAL and the WARN tier. (b) heartbeat counter relabelled `Kraken trades:` and a `Coinbase sleeve` field added (marked *prev tick* — the manage driver runs after the message is composed). (c) `[COINBASE-MANAGE]` emits an unconditional per-tick summary incl. `NOOP`, and a WARNING when **no** routed asset was managed. (d) its `except`, the not-connected branch, and the outer heartbeat handler are all WARNING with correct attribution and `exc_info`.
- **Tests:** `tests/test_health_t3_blocker_diagnosis.py` (13, all pass) — each clause named, all failing clauses reported not just the first, OPPORTUNITY/short/custom threshold overrides honoured, a grid assertion that no blocked intent ever reports `UNEXPLAINED` (drift guard vs `is_actionable`), `hold` resets the streak (pins the reading of the live 312 line), and never raises on a malformed intent.
- **Follow-up (P155b) — read the blocker NOW, without waiting 4H: `scripts/why_no_trade.py`.** `_process_4h_tick_inner` writes `data/diagnostics/diag_<asset>_<tick>.json` **unconditionally every tick**, and its `engine_decide` probe already records the three `is_actionable` clause inputs — so the answer is on the volume for ticks that ran *before* the fix was deployed. Run `docker exec hmats-engine python -X utf8 scripts/why_no_trade.py`. It also prints routing phase + sleeve halt state (a sticky halt needs a manual `reset_halt()`). Parsing note: `_diag_record` stores `repr(output)[:200]` — **not** JSON; it embeds enum reprs like `<SystemMode.NORMAL: 'NORMAL'>` that break `ast.literal_eval`, and truncates at 200 chars, so the script reads per-key by regex. Tests: `tests/test_why_no_trade.py` (15).
- **Follow-up (P155c) — the live dashboard was as blind as the alert.** `run_paper` exported the full intent decomposition per asset (`main.py:17049`, ~180 fields), but `run_live` exported **`{price, regime}` only** — so in the mode that actually matters `dashboard_state.json` carried no `direction` / `target_exposure` / `veto_active` / `actionable`, and the dashboard could not answer "why is nothing trading?" either. The live export now carries the clause inputs plus a `blocked_by` string (from the same `_actionable_blocker`, one source of truth). Diagnostics only — nothing reads these back into a decision. `_export_dashboard_state`'s `merged_asset_data` was also a full **replace** despite its name; now a field-level `update`, so an asset skipped on a prefetch failure carries forward its last real reading (flagged `intent_missing`) instead of being blanked to zeros that read as "the signal died".
- **Follow-up (P155d) — Iron Law 8 was defined but never called (P152 class).** `RoutingPolicy.advance_phase()` does enforce "DRL must be ACTIVE to advance the cutover", **but nothing in production calls `advance_phase`**: `core/execution_service._coinbase_get_routing()` assigns `rp.phase` straight from `data/coinbase_routing_state.json`. So the cutover could be advanced by editing a JSON file with DRL demoted to SHADOW and the guard never ran — while **three** docstrings (`exchange/routing.py`, `exchange/adapter.py`, `exchange/cutover.py`) claimed a *"continuous check"*, one of them naming a module it never even imported. All three corrected. `_coinbase_check_iron_law_8()` now runs on the live routing path and logs CRITICAL once per process. **It observes, it does not block** — fail-closing there routes the asset back to Kraken, which is structurally flat post-Phase-B, i.e. it would convert a DRL-authority problem into a silent total trading stop, the exact failure mode P155 exists to end. `cutover_invariants()` / `validate_obs_dim` / `validate_maker_first` still have **no** production caller: Iron Laws 1 and 9 are *not* enforced at runtime, whatever the module header used to imply.
- **Follow-up (P155e) — the "0/3bps Coinbase fees" migration item is NOT implemented, and it is a live suspect.** `FRICTION.update_fee_bps` (`main.py:4419`) is **global with no venue dimension** and is fed from Kraken's fee-tier API. Post-Phase-B every routed asset is therefore priced at Kraken's ~26/16bps while it actually executes on Coinbase at ~3/0bps. Friction is subtracted from expected alpha *before* sizing, so this systematically shrinks `target_exposure` — a leading candidate for a `ZERO_EXPOSURE` blocker. `_coinbase_fee_model_warning()` now reports the mismatch and its magnitude once per process. **Deliberately NOT auto-corrected:** making the gate venue-aware *loosens* it, and loosening a risk gate blind on a system that has not traded since 2026-06-12 — before `why_no_trade.py` has confirmed the blocker — is an operator decision, not a side-effect of a diagnostics commit.
- **Open, operator decision — config/changelog mismatch.** Commit `795ecc4` is titled "FULL PROMOTION … live ADVISE", but `configs/live_high_risk.json:14` has `v5_1_strategies_live: false`, and that file is the production config (`docker-compose.hetzner.yml:27`). The v5.1 strategy set has therefore never been enabled in production. Not flipped here — that is a risk decision with no PnL evidence behind it.
- **Still NOT established — this fix does not itself resume trading.** Which of `VETO_ACTIVE` / `WEAK_DIRECTION` / `ZERO_EXPOSURE` was blocking SOL for 312 ticks is unknown until `scripts/why_no_trade.py` is run on the server (or the patched build runs one tick); the old message never recorded it and the logs are unreachable from the operator's laptop. Related open lead: `[FastRiskTick][LIVE] BTC: REDUCE_50 - depth_drop=82%(3x)` implies BTC exposure to reduce and a sustained 82% depth collapse — if that depth reading is a degraded feed rather than a real market event, it is a plausible common cause of collapsed `target_exposure` across assets.
- **Mitigation pattern:** an alert that names a subsystem must name it from the *data*, not from a guess baked into the format string — "check X's logs" is worse than useless when X is innocent and silent. And a health counter must measure the venue that is actually authorised to trade; after any routing change, re-derive what the counter counts.

### P154. [FIXED 2026-08-04] CryptoPanic rate-limit state was in-memory → every restart re-spent quota and forgot the 429 backoff
- **Trigger:** operator asked why CryptoPanic billed ~$400 in a month against a Growth plan ($15/mo, 3000 req/mo — see commit c064b1f). **Static call-path audit says the scheduled path cannot cost that.** Only one live consumer exists (`main.py:8272` `_process_4h_tick_inner` → `fetch_headlines_with_meta`; `analyze_many` is dead, `feed.start()` is never called, `hmats-api` mounts volumes `:ro` and holds no key). `_fetch_real` issues **1 request per currency = 3 per refresh**; the agent refreshes only above 1h staleness and the feed throttles below 300s; the live loop is 4H-aligned and per-tick only the FIRST asset trips the refresh. ⇒ **~18 req/day ≈ 540/month ≈ 18% of quota.**
- **Root cause (the multiplier is restarts, not the schedule):** `run_live()` runs a full tick **immediately on entering `while self._running:`**, before any sleep — so each process start = 1 unscheduled tick = 3 requests. Worse, all three rate-limit fields were **in-memory only**: `_last_fetch_time` (300s throttle), `_backoff_until` (429 circuit breaker, default 900s) and `_last_data` (the cache the agent's staleness gate reads). A restart wiped all three → cache empty so `needs_refresh` is True, throttle disarmed, and **an active 429 backoff forgotten, so the engine immediately re-hammered an API that had just said "wait 15 minutes."** A P85-shape crash loop (10 restarts in 6 min) burns 30 requests in 6 minutes — ~a quarter of a normal day's usage per minute; sustained at 1 restart/min that is ~130k req/month. Same in-memory-baseline-is-not-a-control-across-restarts class as **P150 / P148 / P140-B2**, applied to an API rate limiter.
- **Fix (P154), additive, confined to `data_mgmt/feeds/cryptopanic_feed.py`:** persist `{last_fetch_time, backoff_until, data}` to `data/cryptopanic_state.json` (atomic `os.replace`, `_STATE_VERSION="cp_cache_v1"` so shape changes discard stale files per the P153 pattern). Restored in `__init__` via `_restore_state()` **before the first tick**; written at the end of `_fetch_real` AND **immediately on the 429 branch in `_fetch_posts`** (a checkpoint only at end-of-fetch would still lose the backoff if the process dies between the 429 and the return). Cache older than 24h is dropped but the backoff is still restored. Mock mode neither reads nor writes the file. All parses force tz-aware UTC (P40/P97 — a naive datetime would raise on every comparison in `fetch()`). Serialization uses a full round-trip helper, **not** `NewsItem.to_dict()` (that is the lossy 5-field display shape and cannot rebuild the item).
- **This also subsumes the "skip the refresh on the startup tick" fix:** with the cache restored, a restart within the last fetch's 1h window costs **0 requests**, and one outside it refetches legitimately — better than blinding the first real tick.
- **Path note:** `HMATS_DATA_DIR` is unset in the container, but `WORKDIR /opt/hmats` makes the `"data"` default resolve onto the mounted `hmats-data` volume — same resolution P150 verified live.
- **Tests:** `tests/test_cryptopanic_persistence.py` (12, all pass) — backoff survives restart and blocks the startup fetch, expired backoff doesn't block, throttle survives yet still expires (persistence must not wedge the feed shut), news cache round-trips fully, ancient cache dropped while backoff kept, corrupt/version-mismatch/missing file degrade to a cold start, mock mode can't poison the file, no `.tmp` left behind.
- **NOT established at the time:** the restart count itself — `ssh hmats` does not resolve from the operator's laptop and cryptopanic.com 403s WebFetch, so the bill was never tied to a measured request count. Settle it with `docker inspect hmats-engine --format '{{.RestartCount}}'`, `docker logs hmats-engine --since 720h | grep -c '\[CRYPTOPANIC\] Initialized'` (= process starts), and the 429 count.
- **[UPDATED 2026-08-04 via P155 evidence] Restarts are in fact RARE, so the restart multiplier is almost certainly NOT the cause of the bill.** The live line `[HEALTH_T3] ... BLOCKED 312 consecutive ticks` is decisive: `PerTickInvariantChecker._consecutive_blocked` is an **in-memory** dict with no persistence, so a streak of 312 4H ticks means **~52 days of unbroken process uptime** (312 × 4h = 1248h), landing exactly on the 2026-06-13 change window. A crash-looping engine cannot accumulate that streak. ⇒ real usage is ~540 req/mo ≈ 18% of quota, and **the ~$400 is a subscription/tier/overage-pricing question, not traffic — check the CryptoPanic billing page, not the engine.** P154 is still correct and worth having (it removes a real failure mode and costs nothing), but it should not be expected to reduce the bill.
- **Two open items, deliberately not fixed here:** (a) the plan tier is baked into the URL (`BASE_URL = ".../api/growth/v2"`), so downgrading to cut the bill 404s the feed and it silently returns `[]` → falls through to CC News/mock — the exact failure c064b1f fixed for `developer`; (b) `.env.example:47,156` and `docs/HMATS_E2E_TRAINING_GUIDE.md:1691` still describe CryptoPanic as free tier / "100 requests/hour (免费)", which is stale and actively misleading for cost decisions.
- **Mitigation pattern:** an API rate limiter (throttle, backoff, quota counter) that lives only in RAM is not a control — it re-arms on every restart, and the restart-heavy failure modes are exactly when you most need it to hold. Any loop that fires work immediately on startup must read its throttle from disk, not from a fresh object.

### P153. [FIXED 2026-06-14] Coinbase sleeve equity must be PORTFOLIO total_balance (~$4,000), not the futures-summary subset (~$439) — corrects P151
- **Investigation (operator asked "investigate the futures wallet"):** `get_portfolio_breakdown(Default)` shows `total_balance=$3,997`, `total_cash_equivalent_balance=$4,000` (USDC in the spot wallet), `total_futures_balance=-$2.90` (just uPnL). The ~$4,000 the operator expected IS there — it's USDC in the Default portfolio's spot wallet, **cross-collateralizing** the Coinbase US perp futures. The dedicated futures (CFM) wallet holds ~$0; no sweep configured. All 3 perps short 1 nano.
- **Bug (in my own P151):** P151 set `sleeve_equity_usd()` to `futures_balance_summary.total_usd_balance` (~$439) believing that was the real equity. It is an **FCM-only subset**, NOT the cross-collateralized portfolio equity. So P151 wrongly concluded the account was ~5x leveraged near liquidation and the 15% halt was moot. **Reality: ~$4,000 backs ~$2,050 notional = ~0.5x leverage, liquidation far away.** `futures_buying_power=$3,560` (≈ the USDC) was actually closer to the truth than the $439 I trusted — the operator's "3561 is possible" was right.
- **Fix (P153):** `sleeve_equity_usd()` now reads the Default portfolio's `total_balance` via `get_portfolio_breakdown` (cached uuid in `_cb_portfolio_uuid`), with the futures-summary estimate as a degraded fallback. `_BASE_VERSION` bumped `net_liq_v2 -> portfolio_total_v3` so the stale $439 baseline is discarded and re-anchors. **Verified live: `risk baseline set: $3,997.75`.** The 15% halt now fires at ~$3,398 (a real ~$600 stop), not the moot ~$373.
- **Mitigation pattern:** on a cross-collateralized venue, "account equity" is the PORTFOLIO net liquidation value, not any single sub-account/wallet figure. When an exchange exposes multiple balance views (FCM total_usd_balance vs portfolio total_balance vs buying_power), verify which one actually moves with PnL + backs the positions before using it as a risk anchor. Don't trust the first plausibly-named field. Same "wrong-but-plausible field" family as P2.

### P152. [FIXED 2026-06-14] Kraken spot path opened doomed short entries for Coinbase-routed assets (`_coinbase_routed` defined but never wired)
- **Symptom (live CRITICAL):** SOL (Coinbase-routed, DUAL_VENUE) generated a NEW short entry on the Kraken spot path → `[P0_EXECUTE] SOL SELL 6.3 @ leverage=1.0x` → `INSUFFICIENT_SPOT_BALANCE: requires 6.3 SOL, free=0` every tick it signalled short. The short was ALREADY correctly placed on the Coinbase perp sleeve; Kraken was redundantly trying (and failing) to short spot.
- **Root cause (two parts):** (1) the two-sleeve design (`core/execution_service.py:269-276`) deliberately has Kraken trade all 3, relying on **B1** (P140) to block spot-shorts — but B1 is leverage-gated and missed this: `regime_leverage` read 2x at the B1 check (B2's QUIET_ACCUMULATION=2.0) yet the order executed at 1x spot. (2) `_coinbase_routed(ctx, asset)` (the helper that knows an asset is Coinbase-driven) **existed and was unit-tested but was NEVER called** in `execute_intent_v2` — the P141 "Kraken no-op for routed assets" was never actually wired.
- **Fix (P152):** wire `_coinbase_routed` as an ENTRY guard near the top of `execute_intent_v2` (right after the "No active position to close" skip). For a Coinbase-routed asset that is FLAT on Kraken and not a full-exit request → `return {"status":"SKIPPED","reason":"coinbase_routed_no_kraken_entry"}`. **Skips NEW entries only** (both directions: short=impossible on spot, long=would double the book); exits/reduces of a REAL Kraken spot holding still execute (so any legacy position can be unwound). Verified live: `[P152] SOL: Coinbase-routed -> Kraken spot ENTRY skipped`, 0 spot rejects after deploy.
- **Future basis/carry note:** this intentionally stops the engine opening a parallel Kraken spot leg for routed assets. A future carry module (long spot / short perp) must place its spot leg via its own path/flag.
- **Mitigation pattern:** a guard helper that is defined + unit-tested but never CALLED is invisible to both the tests (they test the helper in isolation) and the CI gate. When wiring a "skip path", grep that the predicate is actually invoked in the hot path — same family as P2 (writer/reader exist but aren't connected). Tests: `tests/test_coinbase_routing_fork_v5_1.py` (+3, 8 pass).

### P150. [FIXED 2026-06-14] Coinbase sleeve 15% drawdown-halt baseline was in-memory → loss-cap re-anchored on every restart + no forward-PnL evidence
- **Symptom (latent, not yet fired):** The Coinbase perp sleeve's isolated risk guard (`exchange/coinbase_sleeve.py`) halts at `max_sleeve_drawdown_pct=0.15`, measured from `self._sleeve_start_equity` — but that field was **in-memory only**. Every container restart (deploys are frequent) re-set the baseline to *current* equity, so a sleeve down 14% + restart could lose another 15% before halting. The "loss is capped at 15%" guarantee silently didn't survive a restart. Also: nothing persisted a PnL time-series, so the "live PnL is the real test" claim had no data to evaluate.
- **Root cause:** Same in-memory-baseline-resets-on-restart class as **P148** (DRL frame buffer) and **P140/B2** (`_peak_equity` re-inits → DD-reducer sees ~0%). A risk baseline that lives only in RAM is not a risk control across restarts.
- **Fix (P150): additive, does NOT touch the operator's reconcile/execute/routing logic.**
  1. **Persist baseline + sticky halt** → `data/coinbase_sleeve_state.json` (atomic `os.replace`), restored in `__init__` via `_restore_state()`; written in `update_risk()` when the baseline is first set and when the halt trips, and in `reset_halt()`. The cap is now measured from inception across restarts. Verified live: `[COINBASE_SLEEVE] restored state: baseline=$3,561.42`.
  2. **Forward-PnL log** → `log_pnl_point()` appends one record/tick to `data/coinbase_sleeve_pnl.jsonl` (equity, pnl $/%, positions, uPnL, halt). `main.py` (~17979) writes it after the sleeve snapshot + emits a `[COINBASE-PNL]` heartbeat line.
  3. **Review readout** → `scripts/coinbase_sleeve_review.py` reads the JSONL and prints a sample-size-aware EARN/LOSE/TOO-EARLY/HALTED verdict (cumulative PnL, max DD vs 15% cap, naive annualized Sharpe). The scale-or-stop decision is now data-driven, not a vibe.
- **Tests:** `tests/test_coinbase_sleeve_persistence.py` (7, all pass) — baseline + sticky halt survive restart, halt fires from the restored baseline, reset clears persisted state, PnL log appends, corrupt/missing state degrades to a fresh baseline.
- **Verify live:** `ssh hmats "docker logs hmats-engine --since 8h 2>&1 | grep COINBASE-PNL"` and `docker exec hmats-engine python3 /tmp/coinbase_sleeve_review.py --file /opt/hmats/data/coinbase_sleeve_pnl.jsonl` (scp the script in first — scripts/ isn't baked into the image).
- **Mitigation pattern:** Any risk baseline / high-water-mark / drawdown anchor MUST be persisted and restored, or it is not a control across restarts — it silently re-anchors to the post-restart (often worse) state. Add to the recurring-bug catalog alongside P148/P140-B2.

### P147. [FIXED 2026-06-13] Phase-10 shadow harnesses were DEAD for 6 weeks — 6188/6190 records direction=0 (P2 key-mismatch at scale) + v5.1 promotion demoted
- **Discovery:** validating the Phase-2 cutover + Phase-10 promotion gate, pulled 41 days of shadow ledgers (`data/strategy_shadow/{microstructure,cascade,funding,ml_factor}_*.jsonl`, 6190 records 2026-04-30→06-14) and ran IC locally against fresh Kraken 4H OHLC. **Exactly 2 of 6190 records had a nonzero signal (0.03%).** Every one of the 6 v5.1 strategies returns INSUFFICIENT — not because they're bad, but because **they never computed a signal.** The promotion gate would return INSUFFICIENT_SAMPLES forever; the literature-IC comparison (OFI/VPIN 0.05-0.10, funding 0.10+) was untestable.
- **Root cause (textbook P2 at scale):** the harness calls `strat.evaluate(asset, market_data)`, but the strategies read keys the engine populates under DIFFERENT names: funding strategies read `market_data["funding_rate_8h"]` while the engine stores the 8h value under `funding_rate` (`main.py:5898`); `kyle_lambda` reads `current_price` while the engine has `mark_price`; `vpin_spike` reads `vpin`/`vpin_source` which are **never threaded into** the per-asset dict; `ml_factor` needs `models/factor_extraction/{ASSET}/factor_autoencoder.pt` which were **never trained**. Reason-code histogram nailed each one (`missing_funding_rate_8h` ×1238, `model_missing:...autoencoder.pt` ×619, `no_prev_price`, `quiet(composite=0.00)`, `z_below_threshold(+0.00)`).
- **Fix (P-SHADOW-WIRE, `main.py:~7394`):** build a **shadow-only enriched copy** `_shadow_md = {**market_data, **{funding_rate_8h, current_price}}` before the observe calls — re-keys funding (8h value) + kyle (mark_price→current_price). **NEVER mutates the shared `market_data`** (Iron Law 7: shadow cannot affect fusion). Verified: `funding_mean_reversion`/`funding_post_etf` went `missing_funding_rate_8h`→`history_warmup(N/12)` — now receiving input + warming up. Real signals accumulate from next deploy.
- **Follow-ups — ALL RESOLVED 2026-06-13 (P147-c, commits f412ef1→59664e1, deployed):**
  - `funding_extreme`: `funding_sustained_hours` threaded (stateful wall-clock tracker, |funding_8h|>=0.03%). Quiet now because funding isn't extreme — correct.
  - `vpin_spike`: NO fix needed — runtime VPIN is **computed** (1000 trades/tick), emits `no_spike(vpin=0.2-0.6)`; fires when VPIN spikes. My initial "synthetic" call was from stale data — wrong.
  - `cascade`: `price_change_4h_pct` threaded (rolling per-asset price buffer). Still rare-event by design.
  - `ml_factor`: **fully revived.** (1) Trained 3 autoencoders (clip+grad-clip fixed divergence; |IC| up to 0.116 ETH/BTC; SOL no edge), seed=42, IC-table embedded in the .pt. (2) Inlined the encoder in the agent (runtime image excludes `training/`). (3) Feature threading was the hard part: `_ohlcv_df` is RAW OHLCV — the 122 features = 102 base computed on-the-fly by `_feature_engineer.compute_features()` + 5 denoised/7 external from CURRENT market_data (externals land at :7656, after the :7402 `_shadow_md` snapshot) + 8 regime_proba from `market_data["_gmm_probs"]`, assembled exactly like `build_obs`. Tolerant extraction (zero-fill <=10% missing). **Live-verified: BTC dir-1.0, ETH dir-1.0 (mag 16.2), SOL NEUTRAL.** Models deploy to the `hmats-models` volume (gitignored), via `tar | ssh ... -C /var/lib/docker/volumes/hmats-models/_data/`.
  - **Lesson:** a "shadow input" can be dead at many layers — key-mismatch, raw-vs-computed features, snapshot staleness, runtime-image module exclusion, a swallowed `bool(DataFrame)`. Each looked identical (`missing_features`) until the count diagnostic (`N/122`) localized it.
- **Companion (P147-b):** flipped `configs/live_high_risk.json:v5_1_strategies_live` **true→false**. The 2026-06-13 "full promotion" had wired these into LIVE ADVISE fusion — i.e. **promoting pure noise** (the shadow proves zero measured signal). Re-enable only after `compute_shadow_ic` shows IC>0.05 over 30d on the now-live signals. **Lesson: "promote without validation" is exactly what the shadow layer exists to prevent — and the shadow itself can be silently dead. Always verify the shadow is PRODUCING before trusting (or overriding) its verdict.** Same reader/writer-contract-drift family as P2/P15/P85/P138/P139/P140, at the shadow-input layer.

### P144. [LANDED 2026-06-14] NET (signed) exposure cap — the +0.54 net-long that caused half the loss had NO control
- **Forensic:** the −25% decomposed ~half BETA (system ran **+0.54 net-long into a −23% market**) + ~half negative alpha. The risk layer **computed** `net_exposure` (`risk/global_exposure_cap.py:102` = btc+eth+sol signed) but **only enforced GROSS caps** — a 100% net-long book (all 3 assets long) passes every gross/per-asset cap. The single biggest risk-control gap, and a computed-but-unenforced dead read.
- **Fix:** `max_net_exposure` budget in `ExposureCapConfig`, enforced in `validate_new_exposure` (after the gross check). Clamps a request that would push `|net|` beyond budget **in the dominant direction**; **only reduces requests that INCREASE |net|** — de-risking/hedging (anything that reduces |net|) is never blocked. Pure risk control (never adds exposure). Wired ProductionConfig + from_file + live JSON `risk.max_net_exposure=0.50`. Reversible: set `null`.
- **Why this and not the alpha fixes:** deep research (2 passes, memory `crypto-quant-research-2026-06`) + OOS validation showed the alpha side has NO validated quick fix — cross-sectional needs 100s of coins, stat-arb needs HFT, ensemble directional timing is a dead-end, funding carry is real but currently evaporated (live funding ~0), DRL is overfit (PSR 21%; OOD clean 30d so retraining-as-is won't help). Viable pivot = trend-following + regime-gated carry, ~Sharpe-1, a strategy rewrite that is SHADOW/spec-gated, NOT shippable tonight. The net cap is the one fix that's high-value (half the loss), safe (risk-only), and needs no prediction. See `docs/HMATS_IMPLEMENTATION_PROMPT_ALPHA_BETA_ENDTOEND.md`.
- Tests: `tests/test_net_exposure_cap.py` (7). Commit 2671690.

### P143. [LANDED 2026-06-13] Alpha/beta forensic + alpha-estimate feedback loop reconnected
- **Investigation (memory `live-performance-apr-jun-2026`):** decomposed the −25% into alpha vs beta. **Per-agent live IC** (signal direction vs forward 4H return, 1456 attribution records): **quant IC −0.018 (NOISE)** — and quant is what drives the alpha estimate; **model_alpha −0.160 + llm_sentiment −0.053 (INVERTED, anti-predictive)**; **drl +0.052 (barely break-even, 51% hit)** — the +9 backtest Sharpe does NOT translate live (P40/P41 overfit); only **whale (+10bps, 55%)** + funding showed plausible edge. ~10 agents emit **zero direction** in attribution (dead or P3-extractor gaps — verify before trusting). **Beta regression:** system carried **+0.54 net-long beta** into a −23% market; daily ALPHA intercept **−24.8bps/day (−62%/yr)**; R²=0.09 (only 9% explained by market — the rest is churn); system daily vol 382bps vs market 214bps (1.8× excess from churn). Loss ≈ half beta (long a falling market), half negative alpha.
- **Root issue fixed:** `data_mgmt/market_data_pipeline.py:1318` sets `signal_edge_bps = |quant_dir| * 65` — a fixed multiplier on the NOISE quant signal, calibrated once from stale paper data. The gate consumed it via `estimated_alpha_override` and **ignored the realized-hit-rate feedback the system already tracks** (`update_hit_rate()` IS fed on every close/partial/flip at `execution_service.py:2405/2940/3140` → `_rolling_hit_rate`), because only the FALLBACK path applied `performance_factor`; the override path threw it away.
- **Fix (`defense/constitution.py` check_alpha_gate override branch, `[ALPHA-FEEDBACK]`):** apply `performance_factor = 0.5 + 0.5*_rolling_hit_rate` (range [0.5,1.0]) to the override estimate too. Self-corrects toward realized: 1.0 when winning, →0.5 when the signal stops working (gate tightens on no-edge regimes). Floor 0.5 won't starve trading. **Correct direction for a negative-alpha system** (trade the noise less). Verified live in smoke: `est 55.0→41.2bps (perf_factor=0.75)`. Reversible: `HMATS_ALPHA_FEEDBACK=0` + restart. Tests: `tests/test_alpha_feedback.py`. **This touches the SUPREME gate (rule #1) — change is multiply-by-[0.5,1.0] on one path, floored, env-revertible.**
- **NOT fixed (needs the live-IC validation window, see `docs/HMATS_IMPLEMENTATION_PROMPT_LAYER1_4_IC_GATE.md`):** demoting the inverted agents (model_alpha/llm_sentiment), the +0.54 unwanted beta (no beta cap), and the dead-direction agents. Do these data-driven via the IC gate, NOT hardcoded.

### P142. [LANDED 2026-06-13] Layer-2 churn control — over-trading was ~75% of the Apr-Jun −25% loss
- **Forensic (Kraken-authoritative, [[live-performance-apr-jun-2026]] memory):** the −25% reconciles to −$2,314 realized trading + −$125 fees. Decomposed via 570 INTENT records vs Kraken 4H OHLC: raw signal next-bar ≈ −$29 (52% hit, ≈break-even) → hold-signal-forward = −$574 → ACTUAL −$2,314. **~75% of the loss (~$1,740) is execution churn** (median 12h holds on a 4H clock, ~45 direction-flips/asset, whipsaw out of correct theses then flip at local tops). Signal is bimodal: LONG 58%/+24% vs SHORT 42%/−28%; momentum +20% vs mean_revert −9%; QUIET_ACCUMULATION −25% (bulk of activity); **model confidence is anti-predictive (don't gate on it).**
- **Change 1 — AC thresholds tightened** (`main.py:~2105` constructor literals, tagged `[L2-CHURN]`): `ac1_min_hold_ticks 1→3` (4h→12h), `ac2_max_global 6→3`, `ac5_max_per_day 8→4`. **Wiring note (dead-code trap):** `AntiChurnManager.check_min_hold/check_fill_budget` are DEAD; the live path enforces via `ctx.AC1_MIN_HOLD_TICKS` etc. (`core/execution_service.py:791,820`), sourced `AntiChurnManager → runner._AC*_ mirrors → ExecutionContext.build_from_runner` (`core/execution_context.py:284`). The constructor literals DO propagate through that chain. `ac1_flip_min_hold_ticks` is unwired (flips gated by the alpha-based FLIP_GATE at `execution_service.py:3015`).
- **Change 2 — flip-persistence whipsaw guard** (`main.py`, intent level next to B1, tagged `[L2-CHURN]`): a direction FLIP (opposing a live position) is suppressed until the opposing signal persists `flip_persist_ticks` (default 2) CONSECUTIVE ticks; single-tick reversals just hold the position (no close, no reverse). **Estimate-independent** — unlike the FLIP_GATE which trusts the alpha estimate that proved unreliable (read 40-66bps while realized was negative). Per-asset counter `self._flip_persist` (in-memory; resets on restart = conservative). Never blocks adds/reduces/exits/safety. Config-gated `flip_persistence_enabled` + `flip_persist_ticks` in `configs/live_high_risk.json`. Sim on live history: persist=2 cut flips ~59%. Tests: `tests/test_anti_churn_layer2.py`.
- **Scope/honesty:** Layer 2 attacks the 75% (execution); it does NOT fix the weak signal (the remaining 25%). It only ever PREVENTS trades — strictly safer. Layers 1/3/4 (regime/strategy selection, perp venue, live-IC gate) are the follow-ups. Reversible via the JSON flags + reverting the 3 literals.

### P141. [FIXED 2026-06-13] Coinbase fork opened positions the engine could not exit (orphaned position on rollback)
- **Symptom:** During a ~30s test activation of Coinbase DUAL_VENUE (all 3 assets), the engine immediately opened a real Coinbase BTC 1-contract LONG. Rolling routing back to inert then **orphaned** the position — the engine no longer routed BTC to Coinbase, so it would never manage/close it. Operator flattened via `scripts/coinbase_flatten.py` ($0.45 cost).
- **Root cause:** the first-cut `execute_intent_v2` fork PLACED Coinbase orders on the engine's entry intents, but the engine decides intents from its **Kraken-shaped `_paper_positions`**, which the isolated fork (correctly) never updates. So the engine always thinks Coinbase-routed assets are FLAT → it generates ENTRIES but **never EXITS** for them. A Coinbase position could open/flip on signals but never flatten on hold, bounded only by the sleeve drawdown halt. Same class as P139/P140: a state machine acting on a view that doesn't reflect reality.
- **Fix (P141): single-driver exit-management.**
  1. The `execute_intent_v2` fork is now a pure **NO-OP** for Coinbase-routed assets (`_execute_coinbase_intent_noop` — skips the Kraken path, places nothing).
  2. The SOLE Coinbase order driver is a per-tick step in the 4H heartbeat (`main.py`): for each routed asset it calls `sleeve.manage_to_signal(asset, fused_direction)` which drives the position to `target_for_signal(dir)` — opens/flips AND **flattens when `|dir| < 0.15` (hold)**. Runs every tick regardless of engine intents, so exits always happen.
  3. Resilience: `manage_to_signal` refuses to act on a STALE snapshot (`_reconcile_ok` False after an API timeout → `SKIPPED_STALE`, no order) — don't trade off last-known state.
- **Live-validated:** `scripts/coinbase_manage_validate.py` — dir=+0.50 → opens +1 ETH; dir=+0.05 (hold) → flattens to FLAT. PASSED.
- **Mitigation pattern + LESSON:** (a) any isolated execution path that OPENS positions must also be the thing that CLOSES them, driven by a signal that runs every tick (not only on the host state machine's intents). (b) **Do not activate autonomous live trading on "continue" momentum** — the incident happened precisely because activation was rushed; re-activation must be a deliberate, operator-watched step (one asset first, watch a full open→flatten cycle live). (c) the auto-mode classifier correctly blocks the agent from placing/enabling live orders — operator runs the `coinbase_*` scripts via `!`.

### P140. [FIXED 2026-06-12] Short-biased strategy ran on SPOT (regime_leverage=1) → 6wk spot-long churn, tracker showed phantom shorts, −25% equity
- **Symptom:** Live forensic 2026-06-12. Kraken held ~$7,200 of spot LONGS (0.0489 BTC + 2.09 ETH + 8.61 SOL) with $0.12 USDT free, while `paper_positions.json` recorded BTC/ETH SHORTS. `[P0_EXECUTE] ETH SELL … leverage=0.01x (regime_leverage=1.0x)`; **0 `[MARGIN]` orders in 24h**. "Short" positions were stuck underwater (ETH −277bps) with exits blocked by anti-churn (`net_bps=-318.9 < min 5.0`). Equity drifted ~$9,600 → ~$7,180.
- **Root cause:** `configs/live_high_risk.json:regime_leverage` maps WEAK_CONSOLIDATION / QUIET_ACCUMULATION / NEUTRAL / EXTREME_VOLATILITY → **1.0**. Code at `core/execution_service.py:1766` `leverage = int(round(regime_leverage)) if regime_leverage > 1.0 else None` → `None` → **spot order** (P138: leverage=1 is spot). The market sat in these regimes for weeks, so every order was spot. A spot account cannot hold a short: "short" intents (`direction<0`) became spot SELLs of held coin, "cover" intents became spot BUYs → directionless churn that net-accumulated real spot longs (amplified by the pre-P139 phantom BUY loop). The tracker recorded **intent direction** (short), not the spot effect (long) → total divergence.
- **Fix (P140): two parts.**
  1. **A1 money reconcile** — `scripts/reconcile_flatten_2026_06_12.py` flattened the spot longs to ~$7,190 USD (spot market sells, no leverage) and reset `paper_positions.json` to flat (existence_fuse preserved; backup written).
  2. **B1 root-cause guard** — `main.py` `block_short_entry_on_spot` config flag (default ON). After `regime_leverage` is finalized (~main.py:11556), a NEW short ENTRY (`intent.direction<0`, `is_actionable`, asset currently **flat** via `exposure<1e-9`) at effective leverage ≤ 1 is converted to hold (`direction=0, is_actionable=False, veto_active=True, veto_reason+=B1_SPOT_SHORT_BLOCK`). **Gates entries only — never reduces/exits**, which also carry `direction<0` and must stay executable. Reversible via `"block_short_entry_on_spot": false` in the live JSON profile.
- **What changes after deploy:** in leverage-1 regimes the bot HOLDS instead of churning spot. Real shorts only fire in margin regimes (VOLATILE_CHOP 3x / MOMENTUM_RALLY 2x / PANIC_SELLOFF 2x). The margin-vs-Coinbase-perp decision is deferred to the v5.1/Coinbase plan (Phase 2) on clean state.
- **[B2 2026-06-13 — operator-chosen] regime_leverage WEAK_CONSOLIDATION/QUIET_ACCUMULATION 1.0 → 2.0** in `configs/live_high_risk.json`. Restores real margin shorts in the two dominant regimes (previously spot-only, where B1 blocked them). EXTREME_VOLATILITY kept at 1.0 (B1 still guards it). Composes with B1: shorts allowed where leverage ≥ 2 (real margin), blocked where leverage = 1 (spot). Viable only post-A1 because the flatten freed ~$7,178 USD collateral. **Risk note:** adds leverage on a −25% account with no clean IC yet; `_peak_equity` re-inits on restart so the DD-adaptive reducer (`main.py:11521`, DD>22%→force 1x) sees ~0% now and won't auto-throttle — watch DD as positions build. Reversible: set the two regimes back to 1.0.
- **[B1-EXT 2026-06-13] B1 broadened from flat-only to not-reducing-a-long** (`main.py:~11583`). The original B1 only converted a short to HOLD when the asset was *exactly* flat; a short ADD on an already-short position at spot leverage slipped through to the order layer and dust-rejected (`[ORDER-BALANCE] … REJECTED`). B1-EXT now keys off `_b1_reducing_long = (cur_dir>0 and cur_exp>1e-9)` and fires the block whenever `regime_leverage<=1 and not _b1_reducing_long` — covering flat AND already-short, while preserving the P140 invariant that a SELL reducing a real LONG stays executable. **0 live occurrences post-flatten** (B1+B2 already killed the churn — verified 0 `ORDER-BALANCE` rejects/24h); this is preventive hardening for spot-leverage regimes (EXTREME_VOLATILITY=1.0x, or the DD-adaptive reducer forcing 1x). Regression: `tests/test_b1_spot_short_guard.py` (truth-table + source guard). Reversible via the same `block_short_entry_on_spot` flag.
- **Mitigation pattern:** an order-placement path must express the position the strategy *intends* in an instrument that can *hold* it. A short-capable strategy on a spot-only routing is structurally incoherent — the tracker will record intent while the exchange records the opposite spot effect. Same reader/writer-contract-drift family as P15/P85/P138/P139, but at the **instrument-semantics** layer: intent said "short," the venue could only do "sell spot." See `docs/LIVE_ROOT_CAUSE_2026-06-12.md`.

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

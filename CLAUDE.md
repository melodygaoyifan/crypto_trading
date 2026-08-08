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
| **DRL (TQC)** | **SHADOW (demoted 2026-08-07, P198)** | 3/3 TQC models still load and run inference every tick (signals logged for IC monitoring) but are EXCLUDED from fusion (`integration_v36` admits DRL only at EXIT_ONLY/ACTIVE). Demoted on live evidence: IC +0.052 Apr–Jun, +0.019 (4h, ns) / **−0.081 (16h, t=−2.78)** Jun–Aug — no stable edge in two independent windows; the header's old "Backtest Sharpe +9.22/+7.32/+10.29" numbers are P164-leak artifacts. Demotion holds across restarts only because `drl.force_active=false` in the live config (P198) — before that, `[DRL_FORCE_ACTIVE]` re-promoted ANY persisted level to ACTIVE at every boot. Iron Law 8 logs ONE expected `[CUTOVER-IRON-LAW-8]` CRITICAL per process start (observe-only; that alert is the demotion being seen, not a fault). RE-PROMOTE only via clean retrain (causal pipeline) passing P182 baselines + P166 cost-aware gate on forward data. |
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
| **Exit DRL (Discrete SAC)** | **SHADOW (all 3, demoted 2026-08-07 P200)** | Third DRL alongside the TQC direction DRL (P28); never decides direction. v2 checkpoints (`models/exit_drl_v2/{ASSET}/exit_sac_best.pt`, epochs 138/85/122) still load and log predictions to `data/exit_drl_shadow.jsonl`, but the PARTIAL_EXIT bridge at [core/tick_exit_triggers.py](core/tick_exit_triggers.py#L381) requires EXIT_ONLY and is therefore inert. **The P29 accelerated promotion (2026-04-24) was withdrawn on forensic review (P200):** the "+50%/+83%/+91% Sharpe lift" was negative→less-negative gross of fees vs a strawman baseline (min_profit 100bps vs live 20) in a simulator whose remaining_size goes negative past 4 partials; val_align 0.730/0.710/0.746 is imitation accuracy vs a future-peeking oracle on the checkpoint-selection slice (constant-HOLD scores ~0.74); the kill switch cited as the override's safety net has had `should_demote() -> None` unconditionally since 2026-04-30; 11/40 state features are P164-leaked; lifetime live record = 27 closed events, mean −38.8bps (never reached the ≥30-event gate it bypassed). Demotion changes zero live behavior — Exit-SAC only acts on Kraken positions (structurally empty since P152) and the Coinbase sleeve never consults it. RE-PROMOTE only after a clean retrain (real-entry trajectories, cost-aware validation vs the real exit_alpha config, fixed simulator) passes forward evidence with a working kill switch. |
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

# trend_regime_gate FORWARD evidence — the ONLY basis for promoting it to
# "enforce". P198's regime split is IN-SAMPLE (measured on the loss window);
# do not promote on those numbers. [P213] now runs in-container.
ssh hmats "docker exec hmats-engine python -X utf8 scripts/trend_regime_review.py"

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

`signals/authority_fusion.py` declares **26 agents** in `AUTHORITY_MATRIX_NORMAL` (soldex added 2026-04-15 — see SOLDEX-AUTHORITY tag at line 193 — replaced an earlier slot; **`v5_1_strats` added 2026-06-13 by commit `795ecc4` took the count 25 → 26 and this table was not updated at the time — see P165**).
`_build_fusion_signals` actually reads and uses **20** of them for direction/confidence;
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
| 21 | **v5_1_strats** | ADVISE | v5_1_strats_direction, v5_1_strats_confidence | fusion (`integration_v36.py:2425`) + attribution. Writer `main.py:7546`, gated by `v5_1_strategies_live` |
| 22 | onchain_graph (SOL) | ADVISE | onchain_graph_direction, onchain_graph_confidence | fusion + attribution |
| 23 | options | ADVISE | options_short_confirmation, options_confidence | fusion; **×0.5 dampen removed 2026-04-22** — full weight |
| 24 | vol_alpha | ADVISE | vol_alpha_direction (always 0; runs via intensity) | **fusion branch REMOVED** — affects execution only |
| 25 | whale | ADVISE | whale_flow_direction, whale_confidence (bridged at main.py:7402) | fusion + attribution |
| 26 | soldex (SOL) | ADVISE | soldex_arb_direction, soldex_confidence | fusion + attribution |

**Attribution tracker** (`main.py:8299`) covers 16 direction-producing agents.
Adding a new agent requires **3 files**: agent_signals write site + `_attr_collected` + `_EXTRACTORS` dict in `agents/signal_envelope.py`.

---

## Non-Negotiable Rules

1. **Constitution is supreme** — no trade without alpha gate pass
2. **P0 Safety cannot be bypassed** — kill switch, stale data guard, rate limiter
3. **Existence Fuse** — 28d window, -5% PnL → system halt, manual recovery only
4. **DRL Authority** — SHADOW since 2026-08-07 (P198; was ACTIVE 2026-04-22 → 2026-08-07). Re-promotion requires a clean retrain (P164/P179–P184-fixed pipeline) that passes the P182 baselines and the P166 cost-aware gate on FORWARD data — never on backtest alone.
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

### P226. [FIXED 2026-08-08] Two scanners had been silently skipping main.py — the largest file in the repo — for their entire existence
- **Found by accident, which is the point.** An unrelated edit of mine stripped `main.py`'s UTF-8 BOM and the deploy gate went red with **+308 / +272** findings at once. The tempting move — restore the BOM and go green — would have re-hidden them.
- **Cause:** `scripts/silent_failure_audit.py` and `tools/lint_silent_swallow.py` read with `encoding="utf-8"`. `main.py` carries a **BOM**, so `ast.parse` raised `SyntaxError: invalid non-printable character U+FEFF` and the file was dropped — **not loudly**: a parse failure was indistinguishable from a clean file. The committed baselines simply never included main.py.
  ```
  silent_failure  tryexcept  337 -> 645   (+308, ALL main.py symbols)
  silent_swallow  total      416 -> 688   (+272)
  ```
- **This is P171 EXACTLY** — same file, same BOM, same SyntaxError, same "a check that cannot read the code reports what a check that found nothing reports". P171 fixed it in `lint_orphan_signal_reads.py` (utf-8-sig + a PARSE_FAILURES refusal) and **it was never applied to the other two**. A mitigation applied to one instance of a class is not applied to the class.
- **Attributed BEFORE re-baselining (P175):** every one of the 308 new entries is a main.py symbol (`ExecutionContext.build_from_runner`, `MahalanobisOODDetector.load`, `*Config.from_dict` …), and both counts are now **identical with and without the BOM** — so the rise is **coverage, not regression**. Both baselines carry that attribution inline, or the next reader assumes 272 defects were introduced.
- **`main.py` is left WITHOUT the BOM.** It is an encoding artefact that has now caused this twice; removing it is safe and removes the trap.
- **P171's own test predicted this and had to be corrected.** It asserted main.py *starts with* a BOM, with a message saying "no longer starts with a BOM — good, but this test is the record of why the scanner reads utf-8-sig". That day arrived, so the assertion now pins the **durable contract** (the scanner reads BOM-safely and main.py is genuinely scanned) rather than an incidental property of the file, which was always going to rot.
- Tests: `tests/test_scanner_bom_blindspot_p226.py` (9). The load-bearing one is **behavioural** — it runs the linter and asserts main.py contributes findings — because a source scan can pass while the file is skipped for some other reason. My first version asserted "no bare `encoding=\"utf-8\"` anywhere" and failed on those files' legitimate JSON reads.
- **Mitigation pattern:** when a defect is fixed, grep for every other instance of the SHAPE, not just the symptom — `encoding="utf-8"` + `ast.parse` was findable the day P171 landed. And a scanner must treat "I could not read this file" as a distinct, loud outcome from "I found nothing"; a silent skip understates every number it reports.

### P225. [FIXED 2026-08-08] SmartBeta is LIVE on the gate path with no terminal clamp, it read the previous asset's phase, and five of its seven outputs are decorative — the smart-beta workstream audited for the first time since April
Smart beta shipped 2026-04 and appears in **zero** P-entries since — the one workstream that quietly stayed in production unaudited across the June loss forensic, the cutover, and the P198 investigation. Full audit + fixes:
- **Verified live first (not from the repo):** `SmartBeta V1: ACTIVE (mode=bounded_influence)` on the server, applying **every tick** — `gate 0.99→1.14 size 0.85→0.59`. Note 0.59 is already **below the module's own documented size floor of 0.70**: composition escapes per-writer bounds in practice, tonight, not hypothetically. (Local logs all say DISABLED — the local default config lacks `smart_beta_config`; only `live_high_risk.json` has it. Do not diagnose smart beta from laptop logs.)
- **Channel census — five of seven outputs are inert:** the gate mult is **LIVE-EFFECTIVE** (→ `constitution.check_alpha_gate` → veto → P206 `veto_flat` → **sleeve flatten**, and it binds every tick because the gate's existing-position escape hatch reads Kraken-shaped `_paper_positions`, `{}` since June); the size mult is **decorative** (the ±1-contract sleeve discards magnitude — P210's lesson); `short_restriction_mult` and `recommended_confidence_mult` are computed+clamped and **never written to any consumed key**; `_smart_beta_block_scale_in` has **zero readers** repo-wide; the `CARRY_OPPORTUNITY_*` tags (c9a014b) need `|funding| > 6bps/8h` vs live readings 1–2 orders of magnitude smaller — **the shipping commit's own justifying example does not clear its own threshold** — and if it ever fired, the "LONG-only" gate relaxation multiplies into BOTH `_regime_alpha_gate_mult` and `_..._short` (`smart_beta_controller.py:412-415`) while the compensating short restriction is unwired: a symmetric loosening wearing a directional label.
- **FIX 1 — terminal clamp [0.5, 2.0] on the composed gate multiplier** at its single consumption point (`defense/constitution.py` `[P225-GATE-MULT]`). Six writers stack `*=` into `_regime_alpha_gate_mult` (RegimeAggressor, SmartBeta, AlphaBoost, EXTERNAL-COMPOSITE, EC-ORPHAN), each bounding only its own factor; the size path has had `[0.2,2.0]` at WIRE-REGIME-SIZE for months while the gate path — the one that produces vetoes and sleeve flattens — had nothing. Bounds are behavior-preserving for every composition observed live (0.99–1.39): a runaway backstop, not a retune. **Non-finite fails toward NEUTRAL 1.0**, never either extreme — a poisoned multiplier must not become a silent trading stop (`inf` → permanent veto) or a disabled gate (`nan`).
- **FIX 2 — per-asset phase store.** `agent_signals['phase']` is written from the engine-global `_last_phase_result` **before either phase writer runs in the tick**, so on every asset after the first, SmartBeta's `TREND_STRONG` branch — the long-side gate/size loosening that P173 armed on 2026-08-05, gated on `phase ∈ {IGNITION, EXPANSION}` — was evaluated against **the previous asset's phase**. Now `_last_phase_by_asset[asset]`: same-asset one-tick-stale beats cross-asset fresh (the P206 rule). Landed in `c13de72` via the shared-working-tree sweep (see mechanics note below); guards in `tests/test_gate_mult_clamp_p225.py`.
- **Exoneration recorded so nobody re-litigates it:** smart beta was **not** a plausible contributor to the Apr–Jun +0.54 net-long — 93.4% of ticks got `NEUTRAL_DRIFT` (gate ×1.10 tighter, size ×0.85 smaller) and the only long-boosting branch was dead code (P173) for the entire window. It was a mild brake in the correct direction. **But** P173's fix newly armed that branch with no forward validation, on a book whose net-long problem is still unfixed (P201) — watch `TREND_STRONG` in `[SMART_BETA_APPLY]` logs now that STEADY_UPTREND also routes to the BULL bucket (P217).
- **Also closed: the engine/ + shadow/ shells.** Empty `__init__`-only packages (real code in `archive/` since forever, every importer guarded by `try/except ImportError`) deleted — **and both Dockerfiles COPYed them**, so for ~20 minutes HEAD had the deletion without the COPY-line removal and was **unbuildable** (P192's shape, produced this time by two sessions sharing one working tree: the parallel session's commit swept this session's staged `git rm` but not its unstaged Dockerfile edits). `TestDeadShellsStayDead` now pins dirs-gone and no-COPY **together** so the halves cannot separate again.
- **Parallel-session mechanics worth keeping:** `git add <file>` from a shared tree commits *whatever is in the file at that moment*, including another session's half-landed work — a staged deletion and its unstaged counterpart can be split across two sessions' commits, leaving an intermediate broken state on origin. When two sessions are active, treat "my staged changes" as already-shared state: commit the two halves of any build-coupled change (tree + Dockerfile, manifest + packaging) in ONE action, immediately.
- **NOT fixed (recorded, operator/later):** the double-application of the size mult (`main.py:11992` VC-4 `max()` then `:12632` re-multiply — latent while the sleeve ignores magnitude); the four fail-quiet CoinGlass/sentiment inputs (`oi_change_24h_pct`, `liquidation_imbalance`, `_fear_greed_value`, `_ssc_crowding`) where feed-outage and calm-market are byte-identical (P199/P216 shape) despite `LIQUIDATION_RISK` being the most common live tag; the unwired carry/short/confidence channels (wiring them is a live behavior change, P141); and `scripts/run_beta_audit.py`, which reads a file that does not exist, returns `[]` silently, and last produced a report **before** the loss it existed to measure.
- **Tests:** `tests/test_gate_mult_clamp_p225.py` (23) — clamp binds both ways and demonstrably changes the outcome; every live-observed composition passes through exactly; non-finite → neutral + warns; the phase read-site-precedes-write-site guard (retire the fix if that ordering ever inverts); dead shells pinned in tree and Dockerfiles together. Red-on-regression by construction (pre-fix the module lacks `REGIME_GATE_MULT_*` → import failure).
- **Mitigation patterns:** (a) when N writers compose multiplicatively into one key, per-writer bounds bound nothing — clamp at the consumption point, and check *both* twin paths (the size twin had a clamp; the gate twin, the live one, did not); (b) an audit that "cleared" a module in April says nothing in August — any module still writing into live keys needs re-auditing after every architecture shift under it (Phase B silently converted smart beta's size channel to a no-op and left its gate channel as a flatten trigger, a redistribution of power nobody decided).

### P208. [FIXED 2026-08-07] P144's net-exposure cap has never been evaluated on the venue that holds the risk — enforced it on the sleeve's own book
- **The gap:** P144 added `max_net_exposure` (live **0.50**) because the book ran **+0.54 net-long into a −23% market** — roughly half the Apr–Jun loss — and gross caps do not constrain net *direction*. Its only enforcement site is `core/execution_service.py`, which sits **past the P152 early return** and reads Kraken-shaped `_paper_positions`, `{}` since the June flatten. So since Phase B the cap has run zero times on the only venue that trades (P201).
- **The per-asset contract cap is not a substitute.** `max_contracts_per_asset=1` is per-asset with **no aggregation**: all-three-long is ≈ **+0.5× net** on ~$4,000 while every asset is individually "within cap" — precisely the shape P144 exists to prevent, invisible to both controls.
- **Fix:** `sleeve_exposure()` computes net/gross as a fraction of sleeve equity from the **venue-reconciled** book (using the venue's `current_price`, then entry vwap, then product mid), and `can_trade` enforces the same policy number. Two properties matter more than the threshold:
  - **De-risking is always free.** The gate fires only when an order both *increases* `|net|` **and** breaches the budget. An over-budget book can always be trimmed — the P144 rule, and the P195 lesson about a control that traps you in the position it was meant to limit. Falsification-checked: dropping the "increases" clause immediately fails the de-risk test.
  - **A pricing failure fails OPEN.** `priced_ok` is False if any non-zero position could not be priced, and the gate then allows and warns. A risk control that fires on missing data is a data outage that halts trading.
- **Deliberately NOT routed through `GlobalExposureCapManager`.** That object carries Kraken-shaped state; feeding Coinbase positions into it is the cross-venue contamination P139/P140 came from. Same policy number, enforced locally, on a book read from the venue — consistent with the separate-sleeve design.
- **[P85 again, caught by the suite]** The new `self._max_net_exposure` read sits on the **live order path**, and pre-existing P195 fixtures built sleeves without it → `AttributeError` in `can_trade`, which would refuse *every* order. Now `getattr(self, "_max_net_exposure", None)`. Third time this session that a new attribute read on a hot path needed defending (P193, P201, here).
- **Scope, honestly:** this connects the **net-exposure** control. The existence fuse still never sees sleeve PnL (`pnl_history: []`), and `target_exposure`-based sizing is still discarded by the ±1-contract sleeve — the contract cap, not sizing, is what binds today. Those remain open.
- **Tests:** `tests/test_sleeve_net_exposure_p208.py` (14), including the aggregate-binds-where-the-per-asset-cap-cannot case, hedging orders that reduce `|net|` staying free, and composition with the P195 halt (neither control traps a position).

### P207. [FIXED 2026-08-07] A flatten left an orphan stop on a flat asset — venue-authoritative reconcile is blind for exactly one order-lifetime
- **Observed live**, minutes after the P206 activation. The alpha gate refused ETH and SOL, the sleeve flattened both, and the venue then showed:
  ```
  POSITIONS:   BTC LONG 1                (ETH and SOL flat)
  OPEN ORDERS: BTC SELL stop 57955       (correct)
               SOL BUY  stop 80.82       <- ORPHAN on a flat asset
               ETH SELL stop 1725.5      <- ORPHAN on a flat asset
  ```
- **Mechanism:** `execute_target` flattens with a **marketable LIMIT**, which had not FILLED when `ensure_protective_stop` ran in the same tick. `reconcile_positions` therefore still reported the old position, so a protective stop was **placed on an asset that went flat seconds later**. CDE rejects `reduce_only`, so those are PLAIN orders — SOL touching 80.82 would have **OPENED a long** — and the next reconcile was **4 hours away**.
- **The existing orphan-cancel logic was correct and simply never fired**, because the snapshot said "still holding". This is not a missing guard; it is a guard reading a source that is briefly wrong. Venue-authoritative reconcile (the anti-P139 invariant) is right in the steady state and blind in the window between *order accepted* and *order filled*.
- **Fix:** `ensure_protective_stop(asset, intended_target=None)`. When the caller knows this tick's target, **intent beats the snapshot**: `intended_target == 0` means treat the asset as flat and cancel, whatever reconcile says. Deliberately narrow — a nonzero intent still reconciles to the venue, omitting the argument preserves the old behaviour exactly, and the HOLD path passes nothing because HOLD means the position did not change and the snapshot IS the truth there.
- **Verified live:** after deploy, `SOL: flat -> cancelled 1 orphan stop(s)` / `ETH: FLAT_CANCELLED`, and an independent venue query showed BTC long with exactly one resting order and zero orphans. This also closed the last unobserved leg of P197 (`PLACED → OK_EXISTS → FLAT_CANCELLED`).
- **Tests:** 4 added to `tests/test_coinbase_protective_stop.py` (34 in the file).
- **Mitigation pattern:** any code that acts on a snapshot taken **after it submitted an order** must decide explicitly whether the snapshot or its own intent is authoritative in that window. "Read the venue, never infer" (P139) is the right default and needs this one carve-out; otherwise the system races itself.

### P206. [BUILT 2026-08-07, DEFAULT OFF] Drive the Coinbase sleeve from the GATED intent — and why the obvious two-line version liquidates the book
- **The gap (P201):** the sleeve is the only venue that places orders, and it reads `_last_quant_directions` — written at `main.py:6480` and `:7834`, **both before `engine.decide()` at `:9737`**. So it trades a **pre-gate snapshot** and no risk control binds on the venue holding the risk. `grep -rli coinbase risk/ defense/` returns nothing.
- **The plumbing is trivial and that is the trap.** `_live_intents[asset] = intent` (`main.py:18237`) and the sleeve driver are in the same function body, same tick — the gated intent is already in scope. The 2-line swap "read `_live_intents` instead of the dict" is **wrong**: the gate stack speaks **order** semantics ("should I send an order?"), the sleeve speaks **position-target** semantics ("what position should exist?").
- **Five ways the naive version fails**, each pinned by a test — falsification-checked: the naive version fails **9 of 15**.
  1. **A veto does NOT imply `direction` was zeroed.** Real emitted record: `direction=-0.3327 target_exposure=0.2495 veto_active=True` (`[WEEKEND] alpha 10bps < min 20bps`). Passing `direction` through **opens a position the gate just refused**.
  2. **Some vetoes mean HOLD, not FLAT.** `EXPOSURE_DELTA_BELOW_THRESHOLD` (anti-churn) means *already at target, send nothing*. Under position-target semantics, mapping it to 0 means **flatten — on every tick where the position is already correct.** This is the one that liquidates the book.
  3. **Some vetoes are venue-inapplicable.** B1 blocks short entries because **Kraken spot** cannot hold a short; the sleeve trades **perps** and can. B1 also zeroes `direction`, so the signal is unrecoverable from the intent — the pre-gate value is used for that case alone, and only when B1 is the *sole* veto. (Verified: B1 is **not currently firing** — `regime_leverage` is 2.0 in both dominant regimes. Preventive.)
  4. **`direction` is overloaded as a CLOSE instruction.** The existence fuse and deadlock abort encode close as `direction = −current, target_exposure = 0, veto_active = False`. Fed to `target_for_signal` that **opens the opposite side**. `target_exposure == 0` is the discriminator and `manage_to_signal` never sees it, so it is resolved in the translator. Latent today only because `_paper_positions` is `{}`.
  5. **A missing intent means HOLD, not 0.** A prefetch failure `continue`s the asset so no intent exists, while the sleeve loop iterates all assets. Defaulting to `0.0` turns a data outage into an **unintended liquidation**. (Today's behaviour is already a bug: the asset is managed on a *stale* pre-gate value.)
- **Implementation:** module-level `sleeve_direction_from_intent(intent, fallback_dir) -> (target|SLEEVE_HOLD, reason)` in `main.py`, unit-testable without constructing the runner. The driver honours `SLEEVE_HOLD` by skipping `manage_to_signal` entirely (still reconciling the protective stop) and **never** silently substitutes the ungated value — that substitution is the gap being closed. `[COINBASE-MANAGE]` now logs `src=` naming which branch fired.
- **Config:** `coinbase_use_gated_intent`, **default false**, declared on `ProductionConfig` **and** parsed in `from_file` (P201 had just fixed two flags read by `getattr` and never parsed — do not add a third).
- **What enabling does, from live values:** the alpha gate — Non-Negotiable Rule #1 — currently refuses **ETH** (`Alpha 30bps < threshold 47bps`) and **SOL** (`Alpha 10bps < threshold 59bps`), zeroing both; **BTC** passes (`dir=+0.9998`). So enabling **flattens ETH and SOL and keeps BTC long**. That is Rule #1 binding on the venue that trades, for the first time since Phase B.
- **What this does NOT do, and must not be claimed:** P144's net cap, the existence fuse and the sizing stack stay inert. They measure `_paper_positions` (permanently `{}`, so `target_exposure` is sized against a phantom flat book) and express themselves as `target_exposure`, which the ±1-contract sleeve discards. **This reconnects the signal, not the book.** Making those real needs the sleeve's reconciled positions fed into the exposure-cap state and notional sizing in `execute_target` — separate, larger work.
- **Tests:** `tests/test_sleeve_gated_intent_p206.py` (15).

### P202. [FIXED 2026-08-07] Iron Law 8's DRL clause was a CRITICAL nobody could act on — retired and replaced with the condition that actually protects a routed asset
- **Symptom:** `[CUTOVER-IRON-LAW-8] Iron Law 5/8 violation: DRL authority is 'SHADOW', not ACTIVE` at **CRITICAL**, once per process start, forwarded to Discord. Operator raised it three times; the answer each time was "expected, ignore it" — which is the wrong answer to give about a CRITICAL.
- **Why the clause was vacuous.** It asserts "DRL must be ACTIVE during cutover". Literally true, substantively meaningless: **DRL cannot influence a single live order.** The sleeve trades `_last_quant_directions`, written at `main.py:6480` and `:7834`; `drl_direction` is written at `:7902` — **after both** — and no path connects DRL to that dict or to `market_data["quant_direction"]`. Re-promoting DRL to ACTIVE would satisfy the law and change **zero orders**. So the alert instructed the operator to fix a non-cause, and its only possible resolutions were theatre (re-promote a model measured at IC_16h −0.081) or ignoring it.
- **Why that is a real defect, not a cosmetic one.** A standing CRITICAL that cannot be acted on trains everyone to ignore the channel — the same reasoning that retired the always-red auto-deploy in **P196**, and the mechanism by which **P192**'s broken image build hid for weeks. Telling the operator to ignore it was inconsistent with both.
- **Retired, not downgraded.** The clause guarded an assumption — DRL decides, so demoting it endangers the cutover — that died at Phase B. `exchange/cutover.py::validate_drl_active` is left intact for `advance_phase()` (no production callers); a test now fails if it is re-wired to the live path. `_resolve_drl_authority_level` (P193's fix for this clause) is **deleted** as dead code; the P193 *lesson* survives in the replacement, which reads only `ctx.config` — a field both ctx shapes carry — so the dual-shape hazard cannot recur.
- **Replacement — `_coinbase_check_cutover_guards(ctx, rp, asset)`:** reports a routed asset trading with **no venue-resting protective stop**. Post-Phase-B the sleeve carries 100% of the directional risk and bypasses the alpha gate, veto chain, net-exposure cap and existence fuse (P201), so the stop is the only control that survives the process dying. The message names the fix (`coinbase_protective_stop_assets`). **Self-extinguishing** — unlike the DRL clause, acting on it silences it. Latched **per asset** (the old latch was a single global bool, so the first asset consumed the one shot for all of them), and only runs for assets actually routed (the old clause ran unconditionally and fired for Kraken-bound assets too). Still OBSERVES, never blocks: blocking would route back to a structurally-flat Kraken, converting a protection gap into a silent total trading stop (P155).
- **Expected live output** at the current SOL-only rollout: one CRITICAL each for **BTC** and **ETH**, naming them as unprotected. That is correct and actionable — it disappears when the stop is widened.
- **Tests:** `tests/test_cutover_iron_law_8_wiring.py` rewritten (15). Note the retirement guard matches an **import or call**, not a bare substring — a substring guard fires on the docstring that deliberately names the retired function, which is exactly the P192 `_emergency_flatten` mistake. Falsification-checked: adding a real `from exchange.cutover import validate_drl_active` fails it.
- **Mitigation pattern:** an invariant is only worth alerting on while the thing it constrains can still affect an outcome. When an architecture changes underneath a safety check, the check does not become wrong — it becomes *vacuous*, which is harder to notice because it still reports something true. Ask of any standing alert: *what action would resolve this, and would that action change anything?* If the honest answer is "nothing", retire it or repoint it.

### P201. [FIXED 2026-08-07] Three live risk controls that did not exist, guarded nothing, or measured the wrong book — found by tracing the live order path end to end
**Context.** Since Phase B (2026-06-13) the Coinbase sleeve carries **100% of the directional risk** and Kraken is structurally flat (P152). The sleeve is driven from `_last_quant_directions`, whose only two writers are `main.py:6460` and `:7814` — both **before** `engine.decide()` (`:9737`), i.e. upstream of the alpha gate and every veto — and it is read at `:18391` to place a real order in the same tick. So the sleeve trades a **pre-gate snapshot**, and `grep -rli coinbase risk/ defense/` returns **nothing**: not one risk or defense module knows it exists. Every control below was written against the Kraken book.

**1. Two P198 config keys were inert.** `trend_regime_gate` and `coinbase_flip_persist_ticks` were read via `getattr(self.config, ..., default)` (`main.py:7802`, `:18313`) but were never `ProductionConfig` fields and were never parsed in `from_file` — loading the live profile returned `<<ABSENT>>` for both. The getattr defaults happened to equal the JSON values, which is exactly why it looked correct. Two live consequences: setting `trend_regime_gate: "enforce"` would **silently no-op**, so P198's entire promotion path was dead on arrival; and the config's documented `REVERT: set 0 and redeploy` for flip-persistence **would not have reverted anything**. Same shape as P16 (ENABLE_* flags declared but never gated) and P152 (helper defined but never called). Fixed by declaring + parsing both; defaults deliberately match the old getattr defaults so current behaviour is unchanged.

**2. LIVE had no drawdown halt at all.** `hard_drawdown_halt` (live 0.25) is compared in exactly one place — `main.py:17655`, inside **`run_paper`**. `run_live` computed the drawdown, logged `[NAV-LIVE]`, and never checked it. The only live circuit breaker in the entire system was the sleeve's own 15%. Added the comparison in `run_live`, mirroring the paper halt. It stops **new orders** (`_running = False`) and deliberately does **not** force-close: an unattended forced exit at whatever price a halt lands on is the failure P141 exists to prevent.

**2b. …and the halt would have been unfireable anyway.** `_update_drawdown_snapshot` reads `account_sync.get_equity()`, and `account_sync` is built `exchange_name="kraken"` (`main.py:2759`). Kraken equity has not moved since the 2026-06-13 flatten, so **P163 fixed the writer while the value it wrote stayed blind to the only book that moves**. A halt on a static number cannot fire — worse than no halt, because it reads as protection. Now adds the sleeve's equity. Fail-safe: if the sleeve exists but its equity is unreadable, **hold** the last known drawdown rather than measure a partial book — against a peak that included the sleeve, omitting it understates equity and would fire a *spurious* halt.

**3. The emergency kill switch abandoned the real positions.** `_check_and_execute_force_flat` iterated `self._paper_positions` — `{}` since the June flatten — and sent Kraken orders. So `FORCE_FLAT` closed **nothing**, halted the loop, and walked away leaving the live perps open; with only SOL carrying a P197 venue-resting stop, BTC and ETH would have been left completely unmanaged. It now flattens the sleeve via `execute_target(asset, 0)`, fail-soft per asset so one venue error cannot abort the rest. Note this works even when the sleeve is halted, because P195 permits a reduce while halted.

- **Tests:** `tests/test_live_risk_controls_p201.py` (14). All falsification-checked: un-wiring the config keys fails 4; dropping the sleeve from the drawdown fails 3; removing the FORCE_FLAT sleeve section fails the source guard.
- **NOT fixed (architectural, operator decision):** the sleeve still reads a pre-gate snapshot, so the alpha gate (Non-Negotiable #1), veto chain, P144 net cap, existence fuse, anti-churn and thesis budget remain inert for the only venue that trades — several of them are literally unreachable code past the P152 early return. The existence fuse has never seen a dollar of the sleeve's −5.6% (`pnl_history: []`). The 1-contract-per-asset cap has no net aggregation, so all-three-long ≈ **+0.5× net** — precisely what P144 was written to prevent, and P144 cannot see it. Choose: reconnect the sleeve to the gated intent, or accept trend-only and delete the decorative layers so the system stops appearing safer than it is.
- **Mitigation pattern:** when an execution path is added beside an existing one, every control written against the old path silently becomes decorative. The controls still run, still log, still write to an object — which is why nothing looks broken. Ask of each one: *which book does this measure, and can it reach the orders that actually exist?*

### P199. [FIXED 2026-08-07] The shadow-IC gate had never once produced a verdict — its price series was a frozen TRAINING artifact with zero overlap with the data it was judging
- **Symptom:** every run of `analytics/shadow_ic/compute_shadow_ic.py` reported `INSUFFICIENT_SAMPLES` / `N=0` for every strategy. Indistinguishable from "the strategies have no signal", which is exactly what it was read as. The P147/P165 re-enable criterion for `v5_1_strategies_live` therefore could never be evaluated — the config note said "VERIFY FIRST on the server", and that was impossible in three separate ways at once.
- **Three stacked faults, each sufficient alone:**
  1. **The price series was a training artifact.** `load_ohlcv` read `training/training_data/drl_training/{ASSET}_4H_full.parquet` — the **130-column DRL training set**, regenerated only by a full `rebuild_pipeline` run and frozen at **2026-03-31**. The shadow ledgers start **2026-04-30**. **Zero overlap**, so no forward return could be computed for a single record. Coupling an analytics price series to a training artifact is the root cause; they have different refresh cadences and different owners.
  2. **It read the wrong files.** `load_shadow_ledgers` defaulted to `prefixes=("microstructure","cascade")` and never opened `funding_*.jsonl` or `ml_factor_*.jsonl` — the **only** families emitting signal. The gate meant to validate the promotion could not see the strategies it was judging.
  3. **It cannot run on the server at all** — `training/` is dockerignored, so in-container it fails `ohlcv_missing`, which is where the config note told the operator to run it.
- **Fix (P199):** new `training/scripts/refresh_ohlcv_4h.py` writes `{ASSET}_4H_ohlcv.parquet` (OHLCV only) resampled from `training/training_data/raw/{ASSET}_60m.parquet`, which `training/fetch_binance_full.py` refreshes (it merges, so re-running is cheap). `load_ohlcv` now **prefers** that file, keeping the training parquet as fallback; `prefixes` now defaults to all four families. **Deliberately does NOT append to the training parquet** — OHLCV-only rows would leave 122 feature columns NaN and silently corrupt the DRL training input.
- **Validation:** the resample reproduces all **6,525** overlapping bars of the existing 4H parquet to **0.000000%** — it is exactly the transform that built it. `origin="start_day"` pins bins to 00/04/08/12/16/20 UTC; any other origin shifts every bar and that check would not hold.
- **FIRST REAL VERDICT (30d window, 2026-08-07)** — and it is damning for the promotion: **6 of 9 strategies KILL with `n_directional = 0`** (cascade_anticipation, funding_extreme, kyle_lambda, ofi, stop_hunt_defense, vpin_spike — they emit literally nothing). `funding_mean_reversion` / `funding_post_etf_regime` HOLD on n=4–14 signals, i.e. statistically meaningless, several with negative IC. The one real result is **`ml_factor` BTC: PROMOTE** — n=107, IC +0.181/+0.312/+0.201 across 4/12/24 bars, Sharpe +2.49. `ml_factor` ETH HOLD (Sharpe −4.04), SOL KILL (dead).
  - **Caveat before acting on the PROMOTE:** the ml_factor autoencoders were trained ~2026-06-13, so part of this window is plausibly in-sample for the model. Treat it as a candidate for a forward test, not a validated edge.
- **Coverage note:** Binance publishes MONTHLY archives, so the series lags by up to ~31 days (currently ends 2026-07-31). The refresh script reports the lag explicitly rather than leaving it to be inferred. Parquets are gitignored (`training/training_data/`), so this data is operator-local — CI and the container do not have it.
- **Mitigation pattern:** when a check needs two inputs, "no result" must distinguish *no signal* from *no data*. This one conflated them for months and the wrong conclusion (the strategies are dead) happened to be partly true, which made it even less likely anyone would look. Same family as P175/P187/P188 — a check that cannot run, reporting as though it ran.

### P200-LADDER. [2026-08-07, operator: "if we want to promote DRL, make a plan and act"] The DRL re-promotion ladder — edge probe found a real 16h-horizon signal, and every rung has a kill criterion
The P200 measurement killed the old formulation (every-bar 4h decisions, NOT PROMOTABLE ×3 folds, friction ≥ the loss). Promotion is therefore a **re-formulation ladder**, each rung gated, cheapest evidence first:

**Rung 0 — supervised edge probe (DONE, PASSED).** `training/scripts/edge_probe.py`: walk-forward Ridge/HGB per feature-group per horizon on the clean parquets, after-cost at 6bps RT, against the P166-derived required-IC bar; exit 1 on NO_EDGE so pipelines can gate on it. **Result: EDGE_CANDIDATE on all 3 assets, at the 16h horizon** — and the strict falsification rerun (`--min-train 7200`, every prediction past the split-aware GMM fit boundary, 2023-11→2026-07) got STRONGER, not weaker: BTC 16h ALL/ridge IC 0.079 t=6.0 **+24.4bps/trade after cost** (n=5,747); ETH +34.7bps; SOL external-group +40.8bps. The 4h horizon mostly fails — the edge lives at the slower cadence, exactly where the churn-death evidence pointed. Caveats recorded, not waved off: ~72 cells tested (expect 2-3 false positives at t≥2 — the survivors are at t 3-6, above that, but selection pressure is real), and the **external group's clear rests on only 180 days of Coinglass data** (API depth; 8% of bars) — treat external as a candidate needing longer history, not a proven basis. Probe reports in the session scratchpad (`edge_probe_report.json`, `edge_probe_strict.json`).
**Rung 1 — reformulate to the horizon where signal exists (BUILT).** `--decision-interval N` in `train_drl_full.py`: the policy acts every N bars, positions HOLD in between — N=4 aligns the action cadence to 16h and structurally caps turnover at ¼ of the old policy's. Recorded in `training_friction.json` (a di=1 and di=4 model are indistinguishable by value — record the source). Tests: `tests/test_decision_interval.py` (9; note the alternating-action trap — an even/odd pattern cannot test a period-4 interval because every decision bar is even). The execution prerequisite (sleeve consuming gated fused intent) is covered by the parallel session's **P206** `coinbase_use_gated_intent` — do not duplicate it.
**Rung 2 — train the reformulated candidate (LAUNCHED).** `--asset BTC --decision-interval 4 --tag p200_di4 --venue coinbase --fee-side taker`. Judge ONLY on the in-run P182 gate (beat B&H + SMA after cost, Sharpe CI excluding zero). KILL: gate fails → the DRL formulation is dead even at the right horizon; the supervised ridge signal itself (Rung 0) then becomes the promotion candidate instead, through the same remaining rungs — the edge does not need to be an RL policy to be traded.
**Rung 3 — 30-day live shadow.** The demoted DRL already runs inference + logs signals every tick; a Rung-2 pass means deploying the new checkpoints in SHADOW (unchanged authority) and measuring live IC via the attribution logs + `compute_shadow_ic`'s cost-aware P166 gate on FORWARD data only. KILL: forward IC below the cost bar → back to Rung 1 with a different basis.
**Rung 4 — promotion, three deliberate flips, in order.** (a) gate promote via `data/drl_promotion_state.json` (force_active stays FALSE — the gate's persisted level is now authoritative, that is the point of P200); (b) fusion re-admits DRL automatically at ACTIVE; (c) sleeve consumes it only when `coinbase_use_gated_intent` (P206) is flipped — a separate, operator-watched step (P141: activation is never momentum). Iron Law 8's CRITICAL stops firing at (a).
- **Mitigation pattern:** before spending GPU on RL, ask the supervised question first — RL cannot conjure edge from features that carry none, and a $0.02 ridge probe answers in minutes what a training run answers in days. And when a probe result improves under the stricter test, that is the signature worth trusting; when it only survives the lenient one, it was the lenience.

### P200. [SHIPPED 2026-08-07] The pre-retrain forensics: both DRLs' selection evidence dissolves on inspection, the P164 GMM leak was never fixed in the pipeline that mattered, and the Exit-SAC was demoted to SHADOW
Operator asked for a full investigation of previous training and model selection before any retrain. Two forensic passes (TQC + Exit-SAC), full reports in session memory `drl-training-forensics-2026-08`. (Renumbered from P199 — a parallel session claimed P199 for the shadow-IC price-series fix minutes earlier.) What was found, and what shipped:

**TQC selection was circular end to end.** `EvalCallback` saved "best_model" as the argmax of shaped reward over 112–164 evaluations ON the validation window, and `results.json` reports that same maximum as the model's validation score (identical to 6dp on all three assets) — no held-out data anywhere in the selection chain. The shaped reward included the then-unconditional regime-alignment bonus (up to +0.20/bar for agreeing with the GMM label — the size of a real 0.2% move). `training_friction.json` proves fee_bps=0 (13/11.9/20bps total = slip+impact only, vs 26bps Kraken taker alone). Optuna tuned BTC only (19 completed trials, 91min, objective "+736% NAV" at 0 fee with 5-identical-episode eval); ETH/SOL inherited it untuned; two `optuna_winner.json` values (net_arch, buffer_size) never reached the model (lstm_film_a path replaces net_arch; buffer capped at 500K). Training early-stopped at ~⅓ budget (780K/820K/560K of 2.5–3M steps).

**Provenance is broken.** The 18,316/18,305/11,719-row parquets that trained the deployed models were overwritten 2026-04-22 with 6,525-row files — the deployed runs are unreproducible. Two `results.json` fold_1 entries (BTC 806.34, ETH 1400.12, both `train_rows=0`) match NO surviving artifact. **On the current short parquets, folds 2/3 skip entirely (`train_end < 5000`) — fold_3, the deployed fold, is unreachable without more data.**

**The Exit-SAC promotion evidence dissolved.** The "+50%/+83%/+91% Sharpe lift" that justified P29's accelerated override is negative→less-negative (−0.042→−0.021 etc.), gross of fees, with win rates 26–41% WORSE, against a strawman baseline (`min_profit_bps=100` vs the live exit_alpha's 20; live PROFIT_SCHEDULE omitted), in a simulator whose `remaining_size` goes negative after 4 partials and books a phantom short leg — inflating exactly the DRL's signature action (PARTIAL_EXIT on 26–31% of bars vs the baseline's ~1%). `val_alignment` 0.730/0.710/0.746 is imitation accuracy against a **future-peeking oracle** on the same slice used to select the checkpoint; ~74% of labels are HOLD, so constant-HOLD scores ≈ the same. The oracle's trajectories are synthetic entries every 6 bars in both directions (not HMATS's position distribution); RELEASE_RUNNER has 22–39 examples in ~20k (an untrained action the balanced sampler replicates into ~25% of every batch). 11/40 state features come from the leaked pipeline. **And the kill switch cited as the override's safety net has had `should_demote() → None` unconditionally since 2026-04-30** — README/main.py claimed 4 auto-demote conditions that did not exist. Lifetime live record: 27 closed events (< the 30 the bypassed gate required), mean −38.8bps, idle since 2026-06-13.

**Shipped:**
1. **`rebuild_pipeline.py` GMM leak FIXED (the one P164 missed).** P164 hardened `train_per_asset_gmm.py`, but Step 4 of `rebuild_pipeline.py` — the script that actually generated the deployed parquets — still fit scaler+GMM+BIC-k+cluster-names on 100% of history and deployed the result. Now fits on the first `n·(1−3×0.15)−42` valid rows (the STRICTEST fold boundary, derived from `_get_fold_splits`' own arithmetic rather than looked up from the manifest, which is generated FROM these parquets — circular). Note fitting to fold_1's boundary (what `train_per_asset_gmm`'s `fold=1` default does) still leaks folds 2/3's val windows — they sit inside fold_1's train range. `--gmm-no-split` is the explicit leaky opt-in; `gmm_config.json` records `fit_policy` (a leaky and a clean GMM are indistinguishable by value — record the source, P179). Tests: `tests/test_rebuild_pipeline_gmm_split.py` (8, incl. arithmetic parity vs the trainer and a NaN-shift guard).
2. **`run_training.py` now passes `--extractor lstm_film_a`** — without it the orchestrator produced a 126-dim ULTIMATE model the runtime's hardcoded 1008-dim input cannot consume; the Makefile and the orchestrator silently disagreed (P189 family).
3. **Exit-SAC demoted EXIT_ONLY → SHADOW, all 3 assets** (`main.py` `_per_asset_modes`). Changes zero live behavior (Kraken-only path, structurally empty; sleeve never consults it). The per-boot `record_promotion`/`record_override` re-stamping loop is gone (it overwrote `force_promote_at` with every restart — an audit stamp that erased itself). README's false kill-switch claims corrected. Tests rewritten to pin the demotion (`tests/test_exit_drl_promotion_active.py`).
4. **`fetch_binance_full.py` default 3y → 6y** with a floor warning — 3 years is exactly what produced the fold-skipping parquets.
- **Retrain sequence from here:** fetch 6y → `rebuild_pipeline.py` (causal wavelet + split-aware GMM) → `generate_split_manifest.py` → `train_drl_full.py --extractor lstm_film_a --venue coinbase --fee-side taker` → judge ONLY on the P182 baselines + bootstrap CI + P166 forward gate. Expect the honest number to be near zero (two live windows price the signal class at IC ≈ 0); the retrain is a measurement, not a fix.
- **[MEASURED 2026-08-07 13:32] The measurement ran and the answer is NO EDGE — all three BTC folds FAIL.** Clean parquets (13,095 bars), 3bps Coinbase taker, honest harness, ~16h on the 5090, tag `p200_clean`: fold_1 (val incl. the live period) **−$154,661**, Sharpe −0.93 [−5.31,+1.38] vs B&H −$49,864; fold_2 (rising market) **−$63,650**, Sharpe **−3.11 [−5.27,−0.96]** — significantly negative — vs B&H **+$59,431**/Sharpe +1.69; fold_3 **−$62,045**, Sharpe −1.46 vs B&H +$66,150/+1.82. ~1,900 trades/fold with friction ≥ the loss in every fold; early stops at 355K/395K/~400K steps. Final verdict line from the trainer: *"NOT PROMOTABLE … Deploying this model would be worse than holding."* The old +9..+17 Sharpes were the P164 leak in their entirety. **Do not run ETH/SOL — the strategy-class verdict is settled.** Further DRL investment requires a different signal basis (trend/carry class, whale tilt), not more runs of this design. Full numbers: `models/retrained/BTC/p200_clean/results.json`. Launch gotcha recorded there too: without a fresh `--tag` the trainer restores old folds from cache and reports the stale numbers.
- **Mitigation patterns:** (a) "best on validation" must never mean "max over N draws on the window that is then reported as the score" — selection and reporting need different data; (b) an imitation-accuracy metric against a lookahead oracle is not evidence of economic value, and any headline "lift" must state the SIGN of both arms; (c) when a promotion's safety net is disabled later, the promotion's justification is retroactively void — revisit the decision, don't just note the disablement; (d) a fix to a leak must be verified in EVERY pipeline that carries the leak, not the one where it was first noticed.

### P198. [SHIPPED 2026-08-07] The DRL investigation: the fused decision layer drives nothing, trend-following owns the live loss, and the DRL was demoted to SHADOW — plus the sleeve churn control the P142 fix never reached
Operator asked whether the DRL is the right bet or the failure lives elsewhere, and whether to retrain. Fresh evidence pulled from the server (63 days of attribution signals 2026-06→08, `coinbase_sleeve_pnl.jsonl`, per-tick diag regimes) settled all three questions. Full working notes in the session memory `drl-investigation-2026-08`.

**Finding 1 — the fused decision layer is structurally decorative.** The live book's ONLY driver is `_last_quant_directions` (`main.py:18351` → `sleeve.manage_to_signal`), which under `trend_following_mode=enforce` is the trend signal (P149 inject at `main.py:~7783`). The fused intent — where DRL held DECIDE, where v5_1 was re-enabled (P165), where the alpha gate got venue-aware fees — routes only to Kraken, which P152 keeps structurally flat. **26 agents + fusion + gates currently compute an intent that trades nowhere.** Tuning any of it changes no live position until either the sleeve consumes fused direction or Kraken trades again. This is the P155 lesson one level up: after a routing change, re-derive not just what the health counters count but what the *decision layer decides for*.

**Finding 2 — trend-following owns the current loss, and its execution path had no churn control.** Sleeve forward PnL (the P150 "real test"): **−$225.36 (−5.64%) over 53.9 days, daily Sharpe −4.47**, maxDD −6.79%, fees only ~$16 — the loss is direction, not friction. Market was flat-to-up (BTC −0.2%, ETH +13.5%, SOL +6.6%); trend was short ETH 40% / SOL 50% of the time and flipped BTC **29 times in 54 days** (trend live IC: 4h −0.018, 16h −0.064 t=−2.10; BTC 4h −0.084). The P142 flip-persistence guard acts on the Kraken *intent* and never reached the sleeve driver. **Fix: sleeve-side flip persistence** (`exchange/coinbase_sleeve.py` `flip_persist_ticks`, config `coinbase_flip_persist_ticks=2`): a sign-FLIP of a live position must persist 2 consecutive 4H ticks; single-tick reversals hold. Entries-from-flat, adds, reduces and flattens are NEVER deferred (P195 asymmetry: exits stay instant). Streak is in-memory; a restart only delays a flip (conservative). Tightening-only; revert = set 0.

**Finding 3 — regime decomposition kills the naive chop-gate and motivates a narrow one, SHADOW-first.** Per-tick GMM regimes (from the diag files) × trend signed next-4H-bar return: **WEAK_CONSOLIDATION n=364, hit 45.3%, −9.1bps/tick (negative on all 3 assets); NEUTRAL_DRIFT n=11, −68.9bps; QUIET_ACCUMULATION n=663, +2.6bps (mildly POSITIVE)**. Gating "all chop" would block 93% of ticks = system off; gating only {WEAK_CONSOLIDATION, NEUTRAL_DRIFT} blocks ~34%. That split is IN-SAMPLE (measured on the loss window), so the gate shipped as **`trend_regime_gate: "shadow"`** — logs raw signal + regime + would-gate verdict to `data/trend_regime_shadow.jsonl` every tick; `scripts/trend_regime_review.py` (operator laptop, stdlib-only) prints the forward verdict. Promote to `"enforce"` (zeroes trend in gated regimes → sleeve flattens there) only if ≥3-4 weeks of FORWARD accumulation shows blocked ticks losing and kept ticks not. Do not promote on the 06-14→08-06 numbers — they are the hypothesis, not the evidence.

**Finding 4 — DRL demoted ACTIVE → SHADOW, and the demotion previously could not have stuck.** Live IC across two independent windows: Apr–Jun +0.052 (P143), Jun–Aug **+0.019 (4h, t=0.65, ns) / −0.081 (16h, t=−2.78, significantly NEGATIVE)**, 46% sign-agreement with trend (random). The +9..+19 backtest Sharpes are P164-leak artifacts; OOD clean 30d ⇒ overfit/no-edge, not drift. Demotion mechanics mattered more than the demotion: **`[DRL_FORCE_ACTIVE]` (main.py, 2026-04-11) re-promoted ANY persisted gate level to ACTIVE on every boot** whenever models loaded — the promotion gate's entire demotion machinery (rule #4, auto-demote, manual demote) was cosmetic across restarts. Now gated on `drl.force_active` (default True = historical behavior; live config sets **false**). At SHADOW the DRL still runs inference and logs signals every tick (IC monitoring continues) but `integration_v36.py:~2330` excludes it from fusion (admits only EXIT_ONLY/ACTIVE). **Expected alert: Iron Law 8 logs ONE `[CUTOVER-IRON-LAW-8]` CRITICAL per process start** (observe-only, P155d/P193) — that is the demotion being *seen*, not a fault; do not "fix" it by re-promoting. Exit-SAC (System 3) is governed by its own gate and is untouched.

**Retrain verdict (operator question answered):** retraining now would change nothing live (Finding 1) and would re-measure a signal class that two live windows price at zero. If run at all, run it as a *measurement* under the now-honest harness (P179 venue fees, P180 after-cost selection, P182 baselines, P164 causal pipeline — parquets must be rebuilt server-side first) and expect the honest number to be near zero. The only signal positive in BOTH live windows is **whale** (+10bps Apr–Jun, +5.3bps Jun–Aug; ETH IC +0.122) — too weak to trade alone at this cost structure, but the first candidate as a filter/tilt on whatever drives the book. model_alpha (−0.160→+0.127) and llm_sentiment (−0.053→+0.026) flipped sign between windows: that instability is what noise looks like — do not chase either on one window's numbers.

- **Tests:** `tests/test_coinbase_sleeve_flip_persistence.py` (13 — incl. the oscillation case that reproduces the live bleed, the never-defer asymmetry, stale-pause semantics, and a P152-shape wiring check that main.py actually passes the config), `tests/test_trend_regime_gate_shadow.py` (14 — shadow never alters the signal, enforce zeroes only gated regimes, raw signal still logged under enforce so evidence keeps accumulating, unknown regime fails toward trading, P160 write-failure warning), `tests/test_drl_force_active_gate.py` (5 — force block conditioned on the flag, default preserves history, live config pins false, fusion admission pinned).
- **Mitigation patterns:** (a) a churn/risk control added at one layer does not exist at another — after any new order path, re-derive which controls actually sit on it (this is P142's control missing from the P141 path, found only by counting flips in the PnL log); (b) a demotion mechanism that a startup hook silently reverses is not a mechanism — grep for force-promotes before trusting any persisted authority level; (c) an in-sample regime split is a hypothesis to shadow, never a gate to enforce.

### P197. [BUILT 2026-08-07, DEFAULT OFF] Server-side protective stop on the Coinbase sleeve — plus the dead `entry_vwap` read it immediately exposed
- **Gap it closes:** the sleeve had **no venue-resting protection**. Every exit was a client-side call on the 4H tick, so a dead process left BTC/ETH/SOL perp exposure with nothing at the venue to close it. Capability confirmed by preview first (P195 probe): CDE accepts stop-limits on all three contracts, `errs: []` with `order_margin_total = 0` — the venue treats them as position-REDUCING.
- **Blocker 1 — the adapter had no STOP branch.** `order_type="STOP"` fell through the `else` into a plain GTC limit at `price`, ignoring `stop_price` entirely: an order that reads as protection in the code and is not one at the venue. (`OrderRequest.stop_price` had been carried unused since `exchange/adapter.py:48`.) Added a real branch → `stop_limit_order_gtc_{sell,buy}`, with SELL⇒`STOP_DOWN` (protects a long), BUY⇒`STOP_UP` (protects a short), limit priced *through* the stop so it can fill, and prices via `_round_to_tick` (which correctly rounds to a **multiple** of the tick — BTC's is 5).
- **Blocker 2 — `reduce_only` is rejected by CDE**, so a resting stop is a PLAIN order: if the position it guards disappears while the stop is live, triggering it **OPENS an opposite position**. Contained by reconciling to desired-state every tick in `ensure_protective_stop()`: flat ⇒ cancel every resting stop (the single most important branch); wrong side/size ⇒ cancel and replace; correct one already resting ⇒ leave it (no churn). Refuses to act on a stale snapshot, same rule as `manage_to_signal` — cancelling against last-known state could remove a live stop we cannot see.
- **Ordering that matters:** `execute_target` cancels ALL resting orders (including the stop) before changing a position, so `ensure_protective_stop` runs **after** `manage_to_signal` each tick, against the new position. A NOOP tick never calls the cancel path, so a good stop survives untouched.
- **[FOUND WHILE BUILDING] `entry_vwap` was a dead read.** `reconcile_positions` read `entry_vwap`, but CDE returns **`avg_entry_price`** — the key was silently `None` for **every position since the sleeve was written**. Invisible because nothing consumed it; the stop is its first consumer and would have anchored to the **mark** instead of to entry without ever saying so. Textbook P2 reader/writer key mismatch. Fixed (`avg_entry_price` with `entry_vwap` as fallback, plus `current_price`). Verified live: entries now read BTC 64395 / ETH 1917 / SOL 73.47 where all three were `None` an hour earlier.
- **Anchored to ENTRY, not the mark** — a fixed-risk stop. Anchoring to the mark would quietly make it a trailing stop that ratchets every tick.
- **DEFAULT OFF and it must stay that way until deliberately enabled.** One knob: `coinbase_protective_stop_pct <= 0` disables the feature entirely ("on with a 0% stop" is unexpressible). `coinbase_protective_stop_assets` restricts it to a subset. Enabling places **real resting orders on a live account** — P141: activation is a deliberate, operator-watched step, never momentum.
- **Rollout:** `python -X utf8 scripts/coinbase_stop_validate.py --pct 0.10` (read-only; previews the exact payload per asset, places nothing) → set `coinbase_protective_stop_pct: 0.10` + `coinbase_protective_stop_assets: ["SOL"]` → deploy → watch a full cycle: `[COINBASE-STOP] SOL: PLACED @ …` then `tick summary: SOL=OK_EXISTS` (persists, no churn) then `SOL: FLAT_CANCELLED` → only then widen by emptying the assets list.
- **Observability:** the tick summary logs unconditionally whenever the feature is on, so "no stop resting" can never be silent — a position sitting unprotected is the failure that matters, and silence must not be indistinguishable from protected (P155's lesson).
- **[LIVE BUG, caught by verifying at the venue rather than trusting the log — 2026-08-07]** After activation the engine logged `[COINBASE-STOP] SOL: PLACED @ 80.8170` and the order really was resting. But querying it back showed `_is_stop_order() -> **False**`: `_is_stop_order` required `order_configuration` to be a **dict**, the SDK returns an `OrderConfiguration` **object**, and `fetch_open_orders` did only a **one-level** `__dict__` conversion so the nested object survived. The reconciler could not recognise its own stop. Two live-order consequences, not cosmetic: (1) the next tick would see zero stops and place a **second** one — a new real order every 4H, stacking; (2) on going flat the orphan-cancel branch would iterate an empty list and cancel **nothing**, leaving exactly the orphan the design exists to prevent on a venue with no `reduce_only`. Fixed at the boundary (`_plain()` deep-converts SDK responses to plain dicts, depth-bounded against self-reference) **and** defensively in the sleeve (`_order_config()` accepts either shape). Re-verified against the live resting order: now `True`.
  - **Why the tests passed anyway:** every fake in the test file used a convenient plain `dict`. The venue does not. Added `_SdkOrderConfiguration` (data in `__dict__`, not a dict) and three tests over the real shape — recognised, not duplicated, orphan cancelled — all falsification-checked.
- **Tests:** `tests/test_coinbase_protective_stop.py` (30). Falsification-checked: removing the adapter STOP branch fails 7; removing the flat-cancel branch fails the orphan test; reverting the object-shape handling fails 3.
- **Mitigation pattern:** when a venue lacks `reduce_only`, a "protective" order is only protective while the thing it protects exists — its lifecycle must be bound to the position's, not fired and forgotten. And a field that no one reads is not verified by anything: `entry_vwap` looked fine for months because being wrong had no consequence yet.

### P196. [DECISION 2026-08-07] auto-deploy is `workflow_dispatch`-only on purpose — deploys are manual. Do not "fix" it by adding the secrets.
- **What changed:** `.github/workflows/auto-deploy.yml` no longer triggers on push. Deploys are run from the operator's machine: `bash scripts/hetzner_deploy.sh hmats` (~90 seconds).
- **Why not just add the four `HMATS_SSH_*` secrets:** they hold an SSH private key with **root on the trading server**, and **this repository is PUBLIC**. Actions secrets are available to any workflow that runs on `main`, so configuring them makes *push access to the repo* equivalent to *root on the box that holds the money*. That widening of the blast radius was declined deliberately — not overlooked, and not blocked on anything.
- **Why disable rather than leave it failing:** with the secrets absent it failed on **every** commit since 2026-06-14. A permanently-red check on main is exactly how P192 hid a completely broken image build for weeks — a gate that always fails teaches you to ignore it, so it stops being a gate. Removing a check that cannot pass is better than keeping a red badge nobody reads.
- **Kept as `workflow_dispatch`** rather than deleted, so the capability is one click away if a **restricted deploy user** is ever set up: a dedicated non-root keypair plus a narrow `sudoers` rule for the `su - hmats -c` steps in `hetzner_deploy.sh`. That is the right way to re-enable this. Dispatching it without secrets still fails at the verify step, which is correct.
- **Where the values live if it is ever re-enabled:** `~/.ssh/config` host `hmats` → `HostName 178.104.63.196`, `User root`, `IdentityFile ~/.ssh/hetzner_hmats`; known-hosts from `ssh-keyscan -t rsa,ecdsa,ed25519 <host>` (fingerprints verified against the local `known_hosts` on 2026-08-07 — RSA/ECDSA/ED25519 all match, no MITM).
- **Mitigation pattern:** CI convenience that requires handing production credentials to a public repo is not a neutral default. And when a check cannot pass in the current setup, either fix it or remove it — never leave it red, because the next real failure will look identical to the standing one.

### P195. [FIXED 2026-08-06] The Coinbase drawdown halt blocked the exit it exists to force; the DMS heartbeat gave away its own safety margin; and the LIVE emergency-flatten was dead three ways over
Four defects, found while investigating a Kraken alert that turned out **not** to be our bug. Grouped because they are one shape: **a safety control that does not do what its name says.**

**1. The 15% sleeve halt trapped you in the losing position (the P0).**
- `exchange/coinbase_sleeve.py:can_trade` tested `_halted` as its **first statement** and returned False for *every* order — including a flatten. The control that caps losses prevented the exit that realises the cap. Live consequences, both real: `manage_to_signal(asset, 0.0)` → `execute_target(asset, 0)` → **BLOCKED**, so a halted sleeve could not flatten on a hold signal; and `scripts/coinbase_flatten.py` builds a fresh `CoinbaseSleeve`, which restores `halted` from disk (P150), so **the documented emergency flatten was blocked too** until an operator called `reset_halt()`. It trips at ~$3,398 against equity of $3,772 at the time — roughly 10% away, and reachable.
- **Root cause:** P150 made the halt *sticky across restarts*. That is right for a **loss cap** and compounding for a **trade block**; the two were conflated in one boolean.
- **Fix:** halt now blocks only orders that *increase* absolute exposure. Predicate is `abs(resulting) < abs(cur)` — strictly reducing. Chosen deliberately so a **flip** (`+1 → −1`, abs 1 → 1) is **not** treated as a reduction: a halted sleeve must not open fresh directional risk in the opposite direction.
- **Same shape, found by the test written for the above:** the `max_contracts_per_asset` cap *also* blocked reducing from an over-cap position — a cap preventing you from getting *under* the cap. Now gated on `resulting > abs(cur)` too.

**2. Resting Coinbase orders were never cancelled.** `execute_target` places a marketable **GTC LIMIT** (crosses 0.2%), and `cancel_order`/`fetch_open_orders` existed on the adapter (`exchange/coinbase_adapter.py:252,266`) with **zero callers** anywhere in `main.py`, `core/`, `exchange/` or `scripts/`. An unfilled limit rested indefinitely and could fill **after the engine died** — making the sleeve a risk-*adder* on process death, not merely unprotected. It also let orders stack across ticks. Fix: `_cancel_resting_orders()` runs before each new target, fail-soft.

**3. The DMS heartbeat handed back its margin exactly when it mattered.** `DeadManSwitchMonitor.run()` ended every cycle with `self._stop_event.wait(self._interval_sec)` — **fixed-delay**, so the period was `refresh_duration + interval` and the REST retry budget was charged *on top of* the heartbeat:

| state | refresh cost | period | margin vs 60s timer |
|---|---|---|---|
| healthy | ~0.3s | ~24s | 2.5× |
| failing | 3×10s timeouts + 2×0.5s backoff ≈ 31s | **~55s** | **1.09× (5s)** |

The 24s interval is `min(30, max(5, int(timeout*0.4)))` precisely to buy 2.5×. Confirmed empirically: the 2026-08-06 failures are spaced **exactly 55s apart**. The real danger is the *marginal* case — Kraken slow but answering on attempt 3 — where a refresh "succeeds" yet lands after the server timer already lapsed, silently. Fix: schedule at a **fixed rate** (`wait(max(floor, interval − elapsed))`), with the floor **capped at the interval** (a floor above the cadence would slow a healthy monitor and starve sub-second callers — caught when a flat 5s floor broke three existing tests).

**4. The LIVE escalation was dead three independent ways** (`main.py`, `[FIX-DMS-HALT]`): `refresh()` catches internally and returns `False` rather than raising, and its return was discarded — so the `except` never fired and the counter stayed 0 forever; it then called an underscore-prefixed emergency-flatten method **that does not exist** on the class (the real one is `trigger_emergency_flatten`); and that `AttributeError` was swallowed by its own `except`. **Deleted, not repaired** — deliberately. Flatten-on-DMS-failure is *not* the policy we want: the incident was a Kraken **private-endpoint** outage with the public API healthy, and that must not liquidate the book. Fail-closing there converts an API problem into a forced exit at whatever price the outage leaves — the shape P141 exists to prevent. Fix #3 is the actual protection.

**The Kraken alert itself: DIAGNOSED, NOT A BUG — do not re-investigate.** One contiguous 13-minute window, 2026-08-06 07:01:47–07:14:51 UTC, 17 failures, **zero on any other retained day**. `ReadTimeout (read timeout=10.0)` escalating to Kraken's own `{"error":["EService:Unavailable"]}`, and **only on `/0/private/CancelAllOrdersAfter`** — `[LIVE_DATA]` public fetches kept succeeding mid-window (07:01:52, 07:01:53, with `vol24h` changing, so genuinely fresh). That rules out a local network partition and a Kraken-wide outage. Our retry logic worked and recovered unaided. Harm was zero: 0 Kraken orders, positions and trades all day. **Note `docker logs` is wiped by container recreation — use `/var/lib/docker/volumes/hmats-logs/_data/hmats.log` for incident history.**

- **Tests:** `tests/test_coinbase_sleeve_halt_allows_exit.py` (14, truth-table incl. the flip case), `tests/test_dms_heartbeat_margin.py` (10, drives the real `run()` loop with a fake clock + a source guard that the dead escalation stays removed). All falsification-checked by reverting each fix: 5, 4 and 1 failures respectively.
- **Explicitly accepted open gaps** (decided, not overlooked): no consumer of DMS health — `get_status()`/`consecutive_failures` are read by nothing, and the pre-execution `refresh()` at `core/execution_service.py:570` still discards its bool; Kraken places no orders today (P152 routes all three assets to Coinbase), so an entry-guard would gate a venue that does not trade. Sleeve drawdown is sampled only on the **4H tick** (`update_risk()` is reached solely via `snapshot()`). No server-side Coinbase protection exists at all — residual risk accepted at 0.32× gross / 0.12× net, where liquidation is unreachable.
- **[PROBE RESULT 2026-08-07] Coinbase DOES support server-side protective stops — so building them is now a choice, not a blocked one.** `scripts/coinbase_probe_stop_support.py` (preview-only; creates no orders). A protective stop-limit previews clean on all three contracts — `errs: []` and **`order_margin_total = 0`**, i.e. the venue treats them as position-REDUCING, not as new exposure: BTC SELL 1ct (fee $0.64, lev 3.3), ETH SELL 1ct ($0.27, 3.0), SOL BUY 1ct ($0.42, 2.7). Product metadata does **not** expose supported order configurations, so this cannot be settled by GET alone — the SDK's `preview_*` endpoint validates a payload without creating an order.
  - **Two payload facts, each of which cost a wrong answer first.** (1) `base_size` is in **CONTRACTS**, not base currency — `base_increment = base_min_size = 1` on all three, with `contract_size` (0.01/0.1/5) as *separate* metadata; passing 0.01 for BTC gives `PREVIEW_INVALID_BASE_SIZE_TOO_SMALL`. `CoinbaseAdapter.place_order` already converts correctly (`contracts = int(round(size / cs))`) — the error only appears if you bypass the adapter. (2) Prices must be a **multiple of `price_increment`** (BTC 5, ETH 0.5, SOL 0.01), not merely rounded to its decimal places: `Decimal.quantize(Decimal("5"))` rounds to whole numbers, *not* to multiples of 5, and yields `PREVIEW_INVALID_PRICE_PRECISION`. Use `(v/inc).to_integral_value()*inc` or `CoinbaseAdapter._round_to_tick`.
  - **Blockers before any stop actually ships:** the adapter has **no STOP branch** — `order_type="STOP"` falls through the `else` into a plain GTC limit, so it would silently place the wrong order while looking correct (`OrderRequest` even carries an unused `stop_price`, `exchange/adapter.py:48-50`). And `reduce_only` is rejected by CDE, so a resting stop cannot be reduce-only and could *open* a position if the underlying one is already gone — it must be cancelled whenever the position changes (the P195 `_cancel_resting_orders` hook is the natural place).
  - Also worth knowing: these contracts report `contract_expiry_type = EXPIRING` (dated `20DEC30`) despite the `BTC PERP` display name — "perpetual-style", not true perps.
- **Mitigation pattern:** a risk control must be checked against the action it is meant to *permit*, not only the action it is meant to block. "Halt", "cap", "freeze" and "kill switch" all read as unambiguously safe and can each trap you in the position they were added to protect. Ask of every gate: *what does this prevent me from doing when it fires?* Same family as P144 (computed but unenforced) and P177 (loaded but never called) — here the control is wired and firing, and points the wrong way.

### P194. [FIXED 2026-08-06] The suite was green in CI and red on any machine that had config, models, or logs — 12 failures that CI structurally could not see
- **Symptom:** `pytest tests/` on a developer machine: **13 failed, 3234 passed**. The same commit was **green** on GitHub Actions (`test-suite` success on `0f3cc5d`). Not flakiness — every failure was deterministic, and every one passed in CI for the same underlying reason: **CI has no `.env`, no `models/`, and no `logs/`**, because all three are gitignored. The suite was implicitly asserting "this machine has no configuration."
- **Why this matters more than tidiness:** a local suite that is red for environmental reasons is a suite nobody can read. Real breakage hides in the noise, which is precisely how P188 (every CI job dying at collection, red for months on main) went unnoticed. A test that passes only where the file is missing is not testing the fallback — it is testing the absence.
- **The four defects:**
  1. **`test_derivatives_executor.py` (9 tests) — ambient env leak.** `_build_executor()` passes `initial_size_cap_usd=None`; the constructor then falls back to `os.environ["DERIVATIVES_INITIAL_SIZE_USD_CAP"]` ([derivatives_executor.py:194](execution/derivatives_executor.py#L194)). The repo's `.env` sets it to **250**, and `main.py` `load_dotenv()`s at import — so once any earlier test imported main, a $250 cap fired *before* the cap each test was asserting (`initial_usd_cap_exceeded:2000.0>250.0` instead of `single_position_cap`). Passed in isolation, failed in the full suite: a **test-ordering dependency created by a config file**. Fixed with an autouse `monkeypatch.delenv` so the file is hermetic.
  2. **`test_model_alpha_agent.py::TestTorchFallback` — tested the real model, not the fallback.** `model_path` only selects the DecisionTransformer; the sequence model resolves from `self.sequence_model_path`, which `__init__` **hardcodes** to `models/model_alpha` and does not accept as a kwarg. With the checkpoint present, `_make_agent()` returned a `RealSequenceAlphaModel` — which wants 122 features, got 13, raised, and returned `(None, None)` per `[FIX-AG7]`. Worse, the sibling `test_mock_model_loaded` **passed for the wrong reason**: `_model_loaded` is True when the real model loads too, so it asserted nothing about the mock. Both now patch `_resolve_sequence_model_checkpoint → None` (reproducing CI's state exactly) and assert `isinstance(_model, MockDecisionTransformer)`.
  3. **`test_health_validator.py::test_w3_log_growing` — asserted the machine, not the checker.** Accepted only `("PASS","SKIP")`, but `check_log_growing()` returns **WARN** for a stale log ([live_watchdog.py:279](scripts/live_watchdog.py#L279)) — the check *working*. The sibling `w4` test already accepted WARN. As written it asserted "this box has a freshly-written `live_stderr.log`": true on the live server, SKIP in CI, WARN on any dev machine that ever ran the engine.
  4. **`test_training_orchestrator_scripts.py` — asserted the platform.** Compared a literal `"no/such/trainer.py"` against a log line carrying an OS-native resolved path; on Windows that reads `...\no\such\trainer.py`. Green on Linux CI, red on Windows. Now normalises separators before comparing.
- **Result: 13 → 1.** The one remaining is `test_mypy_gate_is_live::test_the_stamp_matches_the_installed_analyzer`, and it is **correct**: the baseline is stamped mypy `2.3.0`, a local venv has `1.19.1`, and the test's job is to say the local gate would be skipped. CI installs `mypy==` the version read from the baseline (P187), so it passes there. Its own message says *"Do not re-baseline to make this test pass"* — left alone deliberately.
- **Mitigation pattern:** if a test's outcome depends on a **gitignored** path (`.env`, `models/`, `logs/`), CI is structurally blind to it and the local suite carries the whole signal. Tests that exercise a "file missing" fallback must **construct** that state (monkeypatch the resolver, delete the env var), never inherit it from the checkout. And when a fallback test asserts only `loaded is True`, check it cannot pass with the non-fallback object — same "passes for the wrong reason" family as P174/P176.

### P193. [FIXED 2026-08-06] Iron Law 8 cried wolf on the runner call path, and its once-per-process latch meant the false alarm permanently silenced the real check
- **Symptom (found by deploying P192 and reading the first live logs):** `CRITICAL [CUTOVER-IRON-LAW-8] Iron Law 5/8 violation: DRL authority is '', not ACTIVE` at 23:04:12 — while the same process logged `DRLPromotionGate: ACTIVE` at 23:03:30, `[DECIDE_POOL] drl_authority=ACTIVE`, `DRLRuntime={gate=ACTIVE|infer=ACTIVE|promo=ACTIVE}`, and `data/drl_promotion_state.json` said `"authority_level": "ACTIVE"`. DRL was ACTIVE by every other measure. Not a warmup race — the authority was set 42s *before* the alarm.
- **Root cause:** `_coinbase_routed(ctx, asset)` is called with **two different object shapes**. `execute_intent_v2` passes an `ExecutionContext`, which exposes `drl_authority_level` ([core/execution_context.py:106](core/execution_context.py#L106)). `main.py:8562` — the P172 venue-fee resolution — passes **the runner itself** (`_coinbase_routed(self, asset)`), and `HMATSProductionRunner` names the same value `_drl_authority_level` ([main.py:5021](main.py#L5021)). The check only read the first name, so every runner-path call resolved `""`, which `validate_drl_active` correctly fail-closes on. The timestamps line up exactly: the CRITICAL and `[VENUE-FEE] SOL` both fired at 23:04:12.
- **Why it mattered more than one noisy line:** the check latches (`_CB_IRON_LAW_8_WARNED`) on its **first failing** observation, by design, to avoid per-tick spam. The false alarm spent that one shot during warmup — so a genuine DRL demotion later in the same process would have been **reported nowhere at all**. A safety check that fires once, on the wrong thing, is worse than one that never fires: it looks like it is working.
- **Fix (P193):** `_resolve_drl_authority_level(ctx)` reads `drl_authority_level` then falls back to `_drl_authority_level`, returning `""` only when neither carries a value — preserving the deliberate fail-closed-to-reporting behaviour for a genuinely undeterminable level (`test_missing_authority_attribute_fails_closed_to_reporting`). One-line change at the call site; no routing/order behaviour touched.
- **Why the P155 tests missed it:** every fixture in `test_cutover_iron_law_8_wiring.py` built the `ExecutionContext` shape (`_ctx()`). The runner shape was never constructed, so the only call path that actually fires in production was the one path never tested.
- **Tests:** 4 added — runner-shape ACTIVE is silent; runner-shape SHADOW still reported (so the fix isn't just "make it quiet"); an empty `drl_authority_level` falls through to the runner name; and the latch test asserting **which** violation survives. That last one initially asserted only `len(critical) == 1`, which passed with *and* without the fix — a check that could not fail (P174/P176 shape), caught by falsification and rewritten to assert the surviving alert names `'SHADOW'`. All 4 verified to fail with the fix reverted.
- **Mitigation pattern:** when a helper is deliberately shape-tolerant (this one is even tested with `object()`), every attribute it reads must be resolved for **each** shape its real callers pass — grep the call sites, don't trust the type hint. Same reader/writer contract-drift family as P2/P15/P85/P138/P139/P140, but the drift is between two *caller shapes* rather than two modules. And a one-shot alert latch must not be consumable by a condition the check cannot distinguish from "not yet known".

### P192. [FIXED 2026-08-06] P190's allowlist COPY could never build — `.dockerignore` excludes `scripts/` — and auto-deploy had been failing on missing secrets since 2026-06-14, so nobody found out
- **Symptom:** `bash scripts/hetzner_deploy.sh hmats` died in Step 3: `Dockerfile.engine:94 ... failed to compute cache key: "/scripts/why_no_trade.py": not found`. The file exists in the repo AND on the server — it is simply not in the build context.
- **Root cause (two independent faults, stacked):**
  1. **The build was broken by P190.** `.dockerignore:18` has excluded `scripts/` since e92ceef. P190 added `COPY scripts/why_no_trade.py scripts/kq_strategy_diagnostic.py scripts/agent_audit_16.py` (Dockerfile.engine:94) with no matching re-include, so the engine image **has never built since 70ab962**. The P190 gate (`test_the_engine_image_copies_the_diagnostics_the_docs_exec_into_it`) asserted the scripts appear *in the Dockerfile* and passed — membership in one file, while the neighbouring file silently removed them. Same shape as P170/P176: a check whose subject was already gone.
  2. **Nothing surfaced it.** `auto-deploy.yml` fails at its first real step, **"Verify required secrets are present"** — `HMATS_SSH_PRIVATE_KEY` / `HMATS_SSH_HOST` / `HMATS_SSH_USER` / `HMATS_SSH_KNOWN_HOSTS` are not configured as repo secrets. Every auto-deploy run since **2026-06-14 (bc24a40)** has failed there. It never reached `docker build`, so the broken image was invisible. Meanwhile `test-suite` and `codebase-invariants` were **green on main** — main's *code* gates passed while main was **undeployable**.
- **Consequence:** the live engine ran commit `57eefce` (P153) for **7 weeks** while 18 commits (P154–P191, 127 files) sat undeployed — including **P163**, which means every drawdown-scaled risk control in LIVE was reading a permanent `0.0` that entire time.
- **Fix (P192):** `.dockerignore` re-includes exactly the three allowlisted files (`!scripts/why_no_trade.py` etc.) placed **after** the `scripts/` line — Docker takes the last matching pattern. The blanket `scripts/` exclusion **stays**: it is what keeps `launch_live.py` / `coinbase_test_order.py` / `coinbase_flatten.py` out of the live image (P141). Precedent for the pattern already existed in this same file (`training/sentiment/` + `!training/sentiment/train_sentiment_agent_v22.py`).
- **Gate:** `test_every_allowlisted_script_survives_dockerignore` in `tests/test_ops_docs_reference_real_commands.py`. Asserts each COPYed script has a matching negation, that a negation never names a non-existent file (dead config), and — **asserted, not branched on** — that `scripts/` is still excluded, so relaxing the P141 property fails loudly rather than making the test vacuous. Falsification-checked: with the three negations removed it fails naming all three files.
- **RESOLVED 2026-08-07 (P196) — by dropping auto-deploy, not by adding the secrets.** Deploys are run from the operator's machine: `bash scripts/hetzner_deploy.sh hmats` (~90s). `auto-deploy.yml` is now `workflow_dispatch`-only, so main no longer carries a permanently-red check. Note the deploy script's Step 0 runs `ci_check_invariants.py` via bare `python`; if that interpreter lacks mypy the type gate silently SKIPs (P175) — prefer an interpreter that has it (mypy 2.3.0, the version stamped in the baseline).
- **Mitigation pattern:** "the file is named in the build recipe" is not "the file is in the build context." Any allowlist that spans two files (`Dockerfile` + `.dockerignore`, or manifest + packaging config) needs a gate that reads **both**, or the two drift and only a real build reveals it. And a deploy pipeline that fails before its build step gives no signal about whether the build works — a green `test-suite` badge on main says nothing about deployability.

### P191. [FIXED 2026-08-06] The deployment guide built a different system than the one we deploy, and drove it with systemd

P190 fixed the operations runbook and left this as "needs verification against
the box". That was wrong: `scripts/hetzner_deploy.sh` is in the tree and **is**
the authority on what gets deployed. Everything below was checkable locally.

`docs/hetzner_deployment_guide.md` §4 方式 A "Docker 部署（推荐）" walked the
operator through `docker build -t hmats:6.8.0 .` — the **root** `Dockerfile`,
v5.1.0 layout — then `docker run --name hmats-paper` with host mounts at
`/var/log/hmats` and `/var/lib/hmats`. It never named
`docker-compose.hetzner.yml`, `Dockerfile.engine`, `hmats-engine`, or
`hetzner_deploy.sh`. Every other operational doc says `docker exec hmats-engine`;
following the build guide produces a container by a different name, from a
different Dockerfile, with state in directories nothing else reads. Mounting the
old paths also leaves engine state inside the container layer — lost on the next
`up -d`, since the real state lives in the named volumes `hmats-data`/`hmats-logs`
at `/opt/hmats/data`, `/opt/hmats/logs`.

Then five separate places drove the engine through systemd — §7 (the recipe
itself), §8.1 (the cron health check), §9.1 更新代码, §9.2 更新模型, 快速参考, and
Paper→Live. `deploy/systemd/hmats.service` is a v5.1.0 artifact still launching
`main.py --mode paper`; there is no such unit in production. Two consequences
worth naming separately:

- **§8.1 fails silently in the wrong direction.** `systemctl is-active --quiet
  hmats` on a nonexistent unit returns non-zero forever, so a health check copied
  from the guide alerts "HMATS is DOWN!" every 5 minutes while the engine is
  perfectly healthy. Same class as P155-L5/P174: a check whose result carries no
  information.
- **§9.2 更新模型 was wrong even after the systemd lines were fixed.** The engine
  reads models from the **named volume** `hmats-models` (`:ro` at
  `/opt/hmats/models`), not from `~/hmats/models`. `scp`-ing to the host dir
  without syncing into the volume leaves the engine on the old models — which is
  exactly the 2026-04-22 `models_ready=0` / DRL-stuck-in-SHADOW incident recorded
  in the compose file's own comment. The guide now carries
  `hetzner_deploy.sh`'s step-4 `docker run --rm -v hmats-models:/models ...` line.

Also fixed: `docs/HMATS_Architecture_Part4_Execution_DRL_v10.md:424`, the model
promotion checklist, step 6 `systemctl restart hmats` → the compose equivalent.

§7 is kept, retitled 历史路径，线上未使用, with a legacy banner and a
systemd→docker equivalence table, because the non-Docker install is still a
legitimate path and an operator who inherits one needs it.

Gate: three tests in `tests/test_ops_docs_reference_real_commands.py` — the guide
must name what `hetzner_deploy.sh` actually deploys (and clone into the `APP_DIR`
the script `cd`s to, read out of the script, not hardcoded); no doc may drive
`hmats` through systemd outside §7; §7 must still carry its banner (a
falsification guard — the exemption is only safe while the label is there). All
3 red against the pre-fix guide.

**Two gate bugs found while writing this, both worth remembering:**

1. The confinement check first scanned only ```bash fences. Part4:424 sits in an
   **untagged** fence next to a directory tree, so the check could not have
   caught the one line it was written for. Added `_all_fenced_lines()`.
2. It first exempted the deployment guide **as a whole file**. That hid the four
   other systemd sites in that same file (§8.1, §9.1, §9.2, 快速参考, Paper→Live)
   — they only surfaced after the exemption was narrowed to §7's line range. An
   exemption the width of a file is not a carve-out, it is a blind spot. If you
   exempt something, exempt the smallest region that needs it.
3. It scanned only `docs/*.md` + this file, leaving root-level markdown
   uncovered — including `README_DEPLOY_HETZNER.md`, a deployment doc, i.e.
   exactly the class of file that rotted here. Widened to `REPO_ROOT/*.md`;
   those files are clean today, which is the point of covering them now rather
   than after they aren't.

### P190. [FIXED 2026-08-06] The operations runbook documented 14 scripts that never existed — including the emergency-flatten procedure

P186 found `make drl` pointing at a trainer that was never in the tree; P189
found the same in `run_training.py`. This is the third instance, and it is on
the incident-response path.

`docs/HMATS_Architecture_Part5_Operations_v10.md` instructed the operator to run
14 files under `/opt/hmats/scripts/`: `reconcile_positions.py`,
`check_drawdown.py`, `check_fuse_status.py`, `check_bull_transition.py`,
`check_positions.py`, `check_drl_drift.py`, `fill_quality_weekly.py`,
`weekly_performance.py`, `drl_promotion_gate.py`, `pnl_attribution.py`,
`fee_analysis.py`, `fuse_reset.py`, `emergency_flatten.py`,
`remote_emergency_flatten.py`. **`git log --all --diff-filter=A` finds no commit
that ever added any of them.** The first line of 紧急程序 1 was
`python /opt/hmats/scripts/emergency_flatten.py --confirm`.

Two more defects in the same section:

- **The deployment is not systemd.** `sudo systemctl stop hmats` /
  `journalctl -u hmats` describe the v5.1.0 venv install
  (`deploy/systemd/hmats.service`, which still launches `main.py --mode paper`).
  Live is `docker-compose.hetzner.yml` — `hmats-engine` v6.8.0 + `hmats-api`,
  deployed by `scripts/hetzner_deploy.sh` into `/home/hmats/hmats/app`. So the
  stop-the-engine step of an emergency flatten stopped nothing either.
- **`scripts/` is not in the engine image.** `Dockerfile.engine` copies 20-odd
  packages and no `scripts/`, yet lines 66–75 of this file document
  `docker exec hmats-engine python -X utf8 scripts/<x>.py`. Line 1118 had
  already noticed in passing ("scp the script in first — scripts/ isn't baked
  into the image") without the other call sites being corrected.

Fixed:
- Runbook rewritten against what exists. Capabilities with no implementation are
  marked **[未实现]** rather than left as commands that read as working — there
  is no general emergency flatten; the real paths are Kraken Pro UI,
  `scripts/coinbase_flatten.py` (Coinbase sleeve only), and
  `scripts/reconcile_flatten_2026_06_12.py` (Kraken **spot longs** only,
  dry-run by default, does not close margin shorts).
- `Dockerfile.engine` now copies `why_no_trade.py`, `kq_strategy_diagnostic.py`
  and `agent_audit_16.py` — an **allowlist**, not `COPY scripts/`. `scripts/`
  also holds `launch_live.py`, `coinbase_test_order.py` and
  `coinbase_flatten.py`; baking those into the live trading container is exactly
  what P141 exists to prevent. The three copied files are stdlib-only and read
  data/logs. **These `docker exec` commands only start working after the image
  is rebuilt and redeployed.**

Gate: `tests/test_ops_docs_reference_real_commands.py` reads only `bash` fenced
blocks (prose describing a historical bug is not an instruction) and asserts
every documented command's script exists, no command points at
`/opt/hmats/scripts/`, every `docker exec` target is in the image allowlist, the
allowlist has not become a blanket copy, and the runbook does not drive the
system through systemd. Falsified against the pre-fix files: 3 red; plus 2 red
on injected bad commands.

**Follow-up:** the two remaining systemd-era documents named here
(`docs/hetzner_deployment_guide.md`, `docs/HMATS_Architecture_Part4_Execution_DRL_v10.md:424`)
were fixed in **P191**.

### P189. [FIXED 2026-08-05] The training orchestrator invoked two scripts that do not exist, and exited 0 when it failed

P186 found `make drl` pointing at a trainer that was never in the tree. This is
the same defect one layer up, in `training/run_training.py` — the module
`make all`, `make quick` and `make gmm` all delegate to:

| step | invoked | reality |
|---|---|---|
| `run_gmm` | `<root>/scripts/retrain_gmm.py` | not in the tree; the only copy is `archive/gmm_research/retrain_gmm.py`, and it trains the **global** 6-component model that `main.py:3552` treats as the legacy fallback |
| `run_drl` | `<root>/train_drl_full.py` | off by one directory — the file is `training/train_drl_full.py` |

So the documented full pipeline could not complete. `make all` died in step 1;
`make quick` would have died in step 3 after the DT stage had already run.

**And it exited 0 anyway.** `main()` discarded every stage's return value and
returned `None`, so the process reported success whether it trained anything or
not. That is the same shape as P187/P188 — a run that cannot fail says nothing
about what it did — except here it was the *pipeline* rather than the gate.

Which GMM trainer is correct was settled by reading the runtime, not by
picking: `main.py:3520` tries `models/gmm/<ASSET>/gmm_model.pkl` first
(`# Try per-asset models first (v7)`) and only then the global model
(`# Fallback: try global model (legacy)`). So the target is
`training/scripts/train_per_asset_gmm.py` — the leak-free per-asset trainer
P164 fixed. Had the archived path existed, `make all` would have refreshed the
fallback and left the models the runtime actually loads untouched.

Fixed:
- One `SCRIPTS` map on `TrainingOrchestrator`, resolved through `_script()`.
  Every stage goes through it, so `preflight()` checks the paths that are used.
- `preflight()` verifies all four scripts exist **before** step 1, and names the
  key and the resolved path. A pipeline that discovers a bad path in step 3 has
  already spent the cost of steps 1 and 2.
- `main()` propagates failure: `sys.exit(main())`, `0` only if every stage
  returned truthy.
- `--venue` / `--fee-side` added and threaded into the DRL command (the P179 gap
  the Makefile had already closed for `make drl`), and into the Makefile's
  `all`/`quick` recipes — otherwise `make all DRL_VENUE=coinbase` and
  `make drl DRL_VENUE=coinbase` charge different fees with nothing saying so.
- Makefile help said "DRL v5.5" while the trainer self-identifies as "HMATS v7".
  Renamed to v7. (`--output models/drl_v55` left alone: nothing reads it, and
  renaming a live output dir is a separate change.)

Gate: `tests/test_training_orchestrator_scripts.py` — every script in `SCRIPTS`
exists, every flag passed is one the target's argparse defines, `preflight`
fails on an injected bad path, and `main()` returns 1 on both a preflight
failure and a stage failure. Falsified by reintroducing both wrong paths and
dropping `--venue`: 6 of 11 go red.

### P187–P188. [FIXED 2026-08-05] Neither CI workflow was checking anything: the type gate had no analyzer, and the test suite had never run a single test

The previous ~90 P-entries were all defended the same way — write a gate, add it
to CI, move on. This entry is what happened when someone finally looked at CI
itself. Both workflows were reporting on work they were not doing.

**P187 — the mypy gate had no mypy.** `.github/workflows/codebase-invariants.yml`
had no install step at all; the rationale in the file was "Scanners depend only
on stdlib + git", which is true of seven of the eight scanners and false of the
type gate. So on every CI run `ci_check_invariants.py` printed
`mypy check SKIPPED (mypy not installed)` and returned 0, and the job went
green. P159, P161 and P175 each made that skip *louder* — a banner, a version
stamp, a carry-forward — and not one of them made it *fail*. **A warning in the
log of a green job is not a gate.** The 1076-finding type baseline had never once
been compared against a PR.

Fixed in two halves, because either alone would have left the hole open:
- `tools/ci_check_invariants.py` gained `--require-all-gates`, which turns "this
  gate could not run" into exit 1 with a banner naming the missing analyzer.
  Local runs still warn, so a dev without mypy is not blocked; CI passes the flag.
- The workflow installs the mypy release **read out of
  `tools/scanner_baselines/mypy_baseline.json`**, not pinned in the YAML. Per-code
  counts fingerprint the analyzer version, so a YAML pin that drifts from the
  baseline re-creates P161 exactly. One source of truth, not two.

**P188 — `test-suite` had been red on every push, including main, for months.**
`gh run list` showed conclusion `failure` on every run. `gh run view --log-failed`
gave the cause in three lines: `ImportError while loading conftest` →
`tests/conftest.py:10: import numpy as np` → `ModuleNotFoundError`. The install
step was `pip install pytest pytest-asyncio pydantic mypy hypothesis`. numpy was
not on that list, pytest aborts the whole session on a conftest ImportError, and
the job exited 4 at collection. **Not one test in this repository had ever run in
CI.** The suite is the enforcement mechanism for ~90 P-entries; it was enforcing
none of them, and the red X had been on the board long enough to stop being read.

A second, independent fault sat behind the first: the job named ~13 test files
in hand-written steps, a list frozen at the P113 era. Every gate from P125 on —
including every falsifiability test written this week — was absent from it. Had
numpy been installed, CI would still have run about a third of the suite and
called it a pass.

Fixed: `requirements-test.txt` (numpy comes via `requirements.txt`, plus pyarrow,
which pandas needs for `to_parquet` and which no requirements file listed); the
13 steps replaced by one `pytest tests/`; `timeout-minutes` 5 → 20, since the
suite is ~95s plus install and a timeout kill reads as a test failure.

And because "0 tests ran" is invisible in a `-q` summary — that is precisely how
this survived — a second step asserts a floor on the collected count (3247 as of
today, floor 2500) and fails loudly below it. Falsified both ways: 21 collected
from a single file and 0 from a bad path each fire the floor.

**Running the suite for the first time immediately paid for itself.** Three
real defects that had been sitting in the tree, found by tests that existed and
had simply never executed:
- `MockDecisionTransformer.predict` returned `np.float32` despite its
  `-> Tuple[float, float]` annotation (`signal += features[i] * w` promotes, and
  the clamp propagates). `json.dumps` cannot serialize `np.float32`, and that
  value lands in persisted agent-signal dicts — so the failure would have
  surfaced at write time in an unrelated module. The real model already cast;
  this branch did not. `tests/test_model_alpha_agent.py` had asserted it all along.
- `test_account_sync_stale_detection` aged the state by *exactly*
  `MAX_EQUITY_AGE_SECONDS` and the predicate is `age > MAX`, so it passed only on
  the microseconds between two statements surviving float64 rounding at epoch
  magnitude (~240ns resolution near 1.7e9). It failed about one full-suite run in
  two. The literal was `120` under a comment calling it "2 minutes old" against a
  limit the code comment still claimed was 60s; the constant is 120.0. Now
  `MAX_EQUITY_AGE_SECONDS * 2`, read from the module.
- The parquet backtest tests could not run at all — pyarrow was in no
  requirements file.

**Then the first real CI run failed, and that was the point.** Both jobs went red
on the push that fixed them — locally everything was green. Three more defects,
none of which any amount of local testing would have surfaced:

- **The scanner baseline fingerprints mypy's *environment*, not just its
  version.** P161 established that the baseline is version-specific and stamped
  the version. This is that lesson one level deeper: with numpy and pandas
  absent, `--ignore-missing-imports` turns them into `Any` and mypy reports a
  *different set of errors on identical code*. The gate installed bare mypy and
  failed with `no-redef 7 → 8` against a baseline built on a dev box. Verified by
  reproducing all three environments locally: bare mypy diverges, `requirements.txt`
  + mypy 2.3.0 matches the baseline exactly. **`--update` would have "fixed" this
  by baking a CI-shaped number into a file that dev boxes then disagree with.**
  The workflow now installs `requirements.txt`, and the divergent finding was
  fixed at the source instead.
- `infra/safe_torch_load.py` did `import torch as torch_module` — rebinding its
  own parameter with an import statement, which mypy flags as `no-redef` *only
  where torch is missing*. Same pattern in the joblib and pickle wrappers. Fixed
  all three; both environments now agree.
- `training/exit_drl/validate_against_baseline.py` called `sys.exit(1)` at module
  scope when torch was missing. `SystemExit` is not an `Exception` subclass, so
  it tears straight through `try/except ImportError` guards; two tests that only
  import dataclasses from that module died with it. Availability is now recorded
  at import and enforced at use. That in turn required `STATE_DIM`/`ACTION_DIM`
  to move to the torch-free module — they had been *restated* in
  `train_exit_sac.py` under a comment reading "Must match
  generate_expert_trajectories.py", which nothing checked. Another instance of
  the drift class, found only because the module had to become importable.

**The lesson is one step further out than the usual one.** The recurring finding
in this file is "a check that cannot fail is indistinguishable from a check that
passed." P187/P188 are its parent case: *the harness that runs the checks is
itself a check, and nobody had falsified it.* Every gate in this document was
green in CI for the same reason a deleted test file is green.

And the corollary, which cost two round trips here: **a gate verified only on
your own machine has been verified in one environment, which is not the one it
runs in.** Local green told me nothing about numpy-less mypy or torch-less
imports. Push it, watch it fail, read `gh run view --log-failed`, fix the real
cause — and specifically *do not* reach for `--update` when a baseline gate goes
red in CI. That converts an environment discrepancy into a permanently wrong
number.

### P185–P186. [FIXED 2026-08-05] Two ledgers and one build target that had been failing quietly for months

Follow-on work from the P179–P184 pass. Each of these was found by a number
that a *previous* fix had forced into the open — which is the argument for
printing coverage next to every aggregate.

**P185 — 38 of 90 closed trades had no recorded entry, and the ledger
understated its own costs because of it.** The P183 coverage line surfaced the
count; the cause was not the one the code had been blaming. `TradeAttributor`
persisted only CLOSED trades: `_open_trades` lived in memory and
`_load_persisted` restored `_closed_trades` alone, so every position held
across a process restart lost its entry record. The exit then fell into
`record_exit`'s orphan branch with `entry_price=0.0`, `direction=0`,
`strategy=""` and — the part that moved money-shaped numbers —
`entry_fee_usd=0.0`. `net_pnl_usd` subtracts that field, so each
restart-straddling trade understated its cost by roughly one taker fee (26bps
at Kraken) and the ledger read more profitable than the account.

A 2026-04-28 comment at `core/execution_service.py:3743` had diagnosed this as
a swallowed exception in `record_entry` and added logging to catch it. Wrong
suspect: the call succeeded, its result was never written down. *Instrumenting
the caller cannot find a bug in the writer.*

Fixes: an atomic `data/trade_attribution_open.json` sidecar written on every
mutation of `_open_trades` (entry, funding, exit, backfill) and restored in
`__init__`; `TradeRecord.entry_recorded: bool` so an orphan is *marked* rather
than inferred from `entry_time == ""` (a legal value — the same shape as P170,
in a second file); `entry_coverage()` returning the ratio, and a WARNING at
construction naming it. Two smaller losses fixed in passing: `record_entry`'s
force-close branch closed a trade and never persisted it, so the in-memory
report and the JSONL disagreed about the set of closed trades; and
`_load_persisted` carried its own hand-written field list that had already
dropped `funding_payments` — both readers now share `_record_from_dict`.

The DRL counterfactual filters on the flag and prints orphans separately from
unparseable timestamps, because those have different causes and different
fixes.

**P186 — `make drl` invoked a script that does not exist.** `training/Makefile`
advertised `make drl   DRL v5.5 (~6-12h)` and ran `drl/train_drl_v55.py`, which
is not in the tree. The documented path to retrain DRL had been failing with
"can't open file" for as long as anyone ran it. It also passed only
`SOL_60m.parquet` while v5.5 is documented as Cross-Asset BTC/ETH/SOL, and
`make check` in the same file verifies data for all three — so the two targets
disagreed about the asset set and neither said so. Rewritten against
`train_drl_full.py` (the only DRL trainer in the tree, and the one carrying the
P179–P184 fixes), looping `DRL_ASSETS`, with `--venue`/`--fee-side` stated
explicitly: after P179 the env charges real fees, and a model trained at Kraken
26bps is not interchangeable with one trained for the Coinbase nano sleeve at
3bps.

**Also in this pass — the test suite was writing into the repo.** 36 audit
records had accumulated in `analytics/promotion_gate/applied/`, one per run of
`test_main_confirm_executes_atomic_archive`, which drove `main(--confirm)`
without redirecting `APPLIED_DIR`. Every one was a valid-looking operator audit
log; you could only tell them from real applications by reading
`input_plan_path` and noticing it pointed at `/var/folders`. Fixed with an
autouse fixture, plus a `pytest_sessionfinish` guard in `tests/conftest.py` that
fails the run if anything appears in `configs/` or the applied directory —
`data/` and `logs/` are deliberately excluded because the live system writes
there and a guard that names the wrong culprit gets disabled. Three
early-bound `= DECISIONS_PATH` defaults in `apply_promotion_plan.py` were made
late-binding: monkeypatching the module global did nothing, so a test that
omitted `decisions_path` would have archived strategies in the live config.

**Tests:** `tests/test_trade_attributor_open_durability.py` (15),
`tests/test_training_makefile_targets.py` (7). 10 gates verified
red-on-regression before being trusted, including the repo-write guard, which
was confirmed to exit non-zero — the first measurement said exit 0 because the
shell reported `tail`'s status through a pipeline, not pytest's.

**Consequence to carry forward:** realized-PnL totals taken from
`data/trade_attribution.jsonl` for the period before this fix are biased
favourably by the missing entry fees on the 38 orphans, and every per-strategy
or per-regime figure derived from that file covers 52 of 90 trades. Do not
compare a post-fix number to a pre-fix one without saying which is which.

### P179–P184. [FIXED 2026-08-05] The DRL was trained and selected against five numbers that were not measurements — and every model currently deployed was validated that way

Operator instruction was five specific items in the training harness. All five
were real; a sixth (P184) fell out of probing the second. They share one shape,
the same one as P155-L5/P156/P158–P160/P164/P166/P169–P171/P174/P175/P177/P178:
**a quantity that looked like an observation and was structurally incapable of
being one.** Taken together they mean the reported per-fold validation Sharpes
(header table: BTC +9.22 / ETH +7.32 / SOL +10.29) are not evidence of edge —
this compounds P164, which already established the features were leaked.

**Nothing here retrains anything.** The parquets and the training data are
server-side. These fixes stop the *next* run from being meaningless; the
deployed models are still the ones selected under all six defects.

#### P179. `KRAKEN_PRO_FEES` was defined, never read, and the fee it described was hardcoded to zero
`TradingEnvFull._compute_trade_cost_bps` had `fee_bps = 0.0  # within free tier
($10K/mo)`. `grep -rn KRAKEN_PRO_FEES` returned **one line — its own
definition**; the constant that was supposed to price the trade had no reader,
and the comment justifying the zero cited the free-tier branch of the table
nobody was reading. So the DRL was validated on slippage and impact alone (3–10
bps/side) against a live Kraken taker fee of **26 bps** — roughly half the real
friction, on a strategy whose measured defect (P142) is over-trading.

Fixed with `VENUE_FEES_BPS`, venue-aware and sourced from the live table, plus
`--venue` / `--fee-side` / `--assume-free-tier`. The free-tier assumption is now
an explicit opt-in, because it holds only below $10K monthly volume and nothing
in the trainer ever checked that. Unknown venue or bad fee side **raises**
rather than defaulting to zero. `KRAKEN_PRO_FEES` was deleted: a second unread
fee table is how the first one got ignored.

**The fix shipped with the same bug and the test caught it.** `_load_venue_fees`
imported `_VENUE_FEES`; the live symbol is `VENUE_FEE_STD`. `except Exception`
swallowed the ImportError and returned the hardcoded fallback — and because the
fallback is a *correct copy*, the numbers were right and the source was wrong,
which is undetectable by value. The loader now returns `(table, source)` and
logs loudly on fallback. **When a fallback is a correct copy of the thing it
falls back from, agreement proves nothing; you must record which one you read.**

#### P180. Fold selection ran on the shaped reward, which paid for agreeing with the GMM
`best_fold = max(mean_reward)`. `mean_reward` included a bonus for holding a
position aligned with `POSITION_BIAS[regime]` **whether or not it made money** —
a reward for agreeing with the regime label, not for being right. It is
unfalsifiable inside the training loop (the agent can raise it without earning
anything), and it was then the selection metric, so the choice of fold was
partly a statement about the reward function. Worse in the "classic" branch,
which added it to the raw reward with no `quality_weight`, so a 0.5 alignment
bonus outweighed a 0.4% bar move.

The bonus is now `regime_alignment_bonus=False` by default in **both** copies
(`EnhancedRewardCalculator.compute` and `step()`); `bull_mult`/`bear_mult` stay,
because those scale *realized* pnl, so a wrong label scales a loss. Selection is
`mean_pnl_after_cost`, with `selection_metric` recorded in the summary. The
mean_reward path survives only as a guarded fallback for folds restored from
cache with no after-cost figure, and it logs that the choice is not grounded in
realized money.

#### P181. `std_reward` was exactly 0.0 on every fold, of every asset, of every run
`_evaluate` ran ten `deterministic=True` rollouts over one fixed validation
window. A deterministic policy on a fixed window is a pure function — all ten
episodes were byte-identical by construction. **A dispersion figure that cannot
be non-zero is not a measurement**, and it sat next to a Sharpe as if it
qualified it. Replaced by `evaluate_policy_full`: stochastic rollouts, after-cost
PnL, an annualized Sharpe off the **median** episode (bootstrapping the best of N
would put a confidence interval around a max-order statistic), and a percentile
bootstrap CI. `degenerate_spread` is still computed from the actual spread, so
the old condition remains *observable* rather than merely absent. Optuna's
`_eval_nav` had the identical defect — 5 identical episodes averaged into one
number that ranked trials — and is fixed the same way.

#### P182. "Under-performs buy-and-hold" was not an outcome the harness could produce
No baseline had ever been run through this environment, so the promotion
criterion was `Sharpe > X` — an absolute number with nothing to be worse than.
Added `buy_and_hold` and `sma_200bar` as `ScriptedPolicy` objects that duck-type
`model.predict`, so they run the **same** eval path at the **same** fees on the
**same** fold. Promotion now needs two gates: beat **every** baseline on
after-cost PnL, and a bootstrap Sharpe CI excluding zero. **No baselines means
NO PASS** — `_evaluate_baselines` returns `{}` rather than a partial dict on any
failure, because a model that "beat" one of two baselines because the other
crashed is exactly the reassuring-looking artifact this replaces. An unmeasured
comparison is not a favourable one.

#### P183. The DRL counterfactual read nothing and reported that DRL contributed nothing
`analytics/drl_realized/drl_counterfactual_sharpe.py` globbed `logs/attribution/`
for `signals_*.jsonl`. The directory is not in the repo, and the path was
hardcoded to the repo root while the container writes to the `hmats-logs` volume
(`HMATS_LOG_DIR`). With no files to read, every trade fell into `no_signal`, the
aligned/opposed/silent rows printed `(empty)`, and the **ALL TRADES row still
printed real numbers underneath** — so the report looked like it ran and looked
like it found no DRL contribution. It found nothing because it read nothing.
Now honours `HMATS_LOG_DIR`, `logs/attribution/.gitkeep` is tracked, and the
script **exits non-zero** on a missing/empty/all-empty directory rather than
producing a report. It also prints a COVERAGE line, which immediately surfaced
something invisible before: of 90 closed trades, **38 have no usable
`entry_time`** and were being silently dropped before bucketing.

#### P184. A regime id read as a float disabled every regime-conditional reward term
Found while probing P180: with the bonus flag on and off, the reward was
*identical*. Not a bug in the flag — `_regime_to_name` was returning the string
`"2.0"`. `_get_regime` reads the regime out of `df.iloc[step]`, which builds a
Series over the whole row; if every column in that row is numeric, pandas upcasts
the int64 regime to float64, `isinstance(2.0, np.integer)` is False, and the
name comes back as `"2.0"`. That string is not in `POSITION_BIAS`, not in
`BULL_REGIMES`, not in `BEAR_REGIMES`, not in `regime_weights` — so every
regime-conditional term quietly took its no-op branch and the environment
trained **regime-blind**, with nothing raised.

Verified directly:

    df[timestamp, close, regime].iloc[2]["regime"] -> np.int64(2)
    df[close, regime].iloc[2]["regime"]            -> np.float64(2.0)

**Latent today, not live**: the production parquets carry a `timestamp`
datetime64 column, which forces the row Series to dtype object and preserves the
int. So the correctness of the reward function depends on an unrelated column
being present — drop `timestamp` in any preprocessing step and the regime logic
silently evaporates. `_regime_to_name` now accepts integral floats, and
`_assert_regimes_resolve()` warns at construction if any id fails to map. A test
pins the pandas upcast itself, so if that behaviour ever changes the explanation
in the code is flagged rather than left quietly wrong.

#### Tests and method
`tests/test_drl_cost_realism.py` (36). Two layers, deliberately: source-level
gates that run everywhere, and behavioural tests gated on
`pytest.importorskip("gymnasium")` — the training stack is not installed on most
machines, so behavioural tests alone would skip silently and guard nothing.
**All 13 source gates were verified red-on-regression by injecting the
corresponding defect into a scratch copy** (never the shared working tree) and
confirming each predicate flips. Behavioural verification ran against minimal
gymnasium/sb3 stubs. Measured after the fixes: kraken/taker `fee_bps=26.00`
(coinbase `3.00`, free tier `0.00`); deterministic `std_reward=0.000000
degenerate=True` vs stochastic `260.670062 degenerate=False`; `buy_and_hold
pnl=-$46,245.68`, `sma_200bar pnl=-$23,977.47`.

`tests/_source_scan.py` is new and shared. P177's scanner blanked `#` comments
but kept string literals, which is correct for asserting a *log line* is gone
and wrong for asserting a *statement* is gone — the P179 fix documents the
removed `fee_bps = 0.0` inside a docstring, so a comments-only scan matched its
own explanation. Hence `strip_docstrings`. It returns the raw source unchanged
on a parse failure, because a scanner that returns `""` would make every
"X is absent" assertion pass vacuously.

#### Consequence to carry forward
**Every DRL run predating this is invalid for promotion purposes** — validated
at roughly half the real friction, selected on a shaped reward that paid for
agreeing with a label, with an error bar that was zero by construction, against
no baseline, on an environment that was regime-blind whenever the frame was
all-numeric. Retraining is server-side and is a prerequisite for any promotion
decision, not an optimisation.

### P178. [FIXED 2026-08-05] The DRL scored an all-zero state vector and returned it as a signal; and main.py named a DEPRECATED file as the canonical main loop

`DRLAgent.generate_signal` takes 8 args; the 3 that carry state
(`position_state`, `market_data`, `agent_signals`) default to `None`. A caller
passing only asset/price/regime reached `build_state()` with three empty dicts.
`build_state()` does not object — it fills the vector with zeros and defaults,
`get_action()` scores it, and the payload came back `is_valid=True` with a real
direction and confidence. The only trace was `data_quality=0.20` on a field with
no consumer outside `to_dict()`. Measured, mode=SHADOW, no state args:

    is_valid=True  data_quality=0.20
    issues=['position_state_empty','regime_result_none',
            'market_data_empty','agent_signals_empty']

`core/runtime_spine.py:~878` is that call verbatim — 3 of 8 arguments.

**It is latent, not live, and the reason matters.** `RuntimeSpine` has no
production constructor (only its own factory and
`tests/test_runtime_singleton_refresh_advanced.py`); main.py owns the tick; and
production DRL is the TQC ensemble in `main.py` (`_drl_ensembles`, :3697/:7795),
not this wrapper — `agents/drl_agent.py:172` already calls itself legacy. So the
spine's DRL path is dead code calling a legacy wrapper.

Fixed by refusing, not by rewiring: `generate_signal` now returns
`is_valid=False, direction=0.0, reason="no_state_inputs"` when **all three**
state dicts are absent, carrying `price`/`regime`/`issues` for diagnosis.
Zeroing `direction` is the load-bearing half — `is_valid` has no consumer, so
flagging alone would have changed nothing. Partial input is still permitted and
merely lowers `data_quality`, so this cannot misfire on a degraded real tick.
The call site was deliberately NOT corrected: supplying the dicts would put a
never-exercised DRL path into a file marked NOT USED, on a live system.

Second half: `main.py:11` declared `CANONICAL_MAIN_LOOP: core/runtime_spine.py`
while that file's own line 2 says `_DEPRECATED (T29) ... NOT used. Actual tick
processing lives in main.py._process_4h_tick_inner()`. The banner at :19884
repeated it as `CANONICAL_SPINE`. Both now name
`main.py::_process_4h_tick_inner`. A reader could not previously tell which of
the two contradicting claims to believe, which is how the dead spine kept
looking like something worth maintaining.

Four existing tests in `tests/test_drl_agent.py` called `generate_signal` with
no state at all — they test the action→direction mapping and used the empty
path as a convenience. Two were given a minimal `market_data`; the other two
pass because the refusal now carries `price`/`regime`. Worth noting the suite
had encoded the empty-input call as legitimate.

Pinned by `tests/test_drl_refuses_empty_state.py` (11 tests), including
`test_the_spine_has_no_production_constructor` — if `RuntimeSpine` ever gains
one, the latent path is live and that test says so.

### P177. [FIXED 2026-08-05] A risk controller imported, logged as "loaded", and never once called

`main.py` imported 7 symbols from `risk/short_position_controller.py` and
`analytics/sota_metrics_calculator.py`, used **none** of them (each name
appeared exactly once — on its own import line), set `V6_MODULES_AVAILABLE`
which is read nowhere, and logged on every boot:

    [OK]V6 SOTA modules loaded (short risk + metrics)

`get_short_controller()` has no call site anywhere in the repo, so
`assess_risk`, `check_stop_loss` and `get_position_size_multiplier` — a
stop-loss, a daily-loss halt and a squeeze-risk sizer — have never executed in
production. The log line was literally true and materially false: it reads as
an assurance that short-side risk is governed.

Live short risk is `defense/short_control.py`, which IS invoked at
`main.py:10577` under `intent.direction < 0` and reported at :11540.
`risk/short_position_controller.py` is a second, parallel implementation of the
same job that lost the race and was never unplugged.

**Deliberately not wired in.** Enabling three untested risk *actuators* on a
live account to fix a cosmetic log line trades a false reassurance for a real
hazard, and two controllers clamping the same exposure independently is worse
than one. The import block and banner were removed; the module keeps a NOT
WIRED header and its own unit tests.

`tests/test_dead_risk_controller.py` (7 tests) pins all three directions: the
banner does not return, the dead controller does not silently acquire a caller,
**and the live path stays live** — the third is the one that matters, since the
first two would also pass on a system with no short risk control at all. Both
failure modes were probe-verified by injection.

Method note: the first draft of those tests failed against the fix's own
comment block, because the comment quotes the removed log line and names
`get_short_controller()`. A source scanner that cannot distinguish code from
prose about code is not measuring what it claims to; the helper now blanks `#`
comments in place (joining tokens with separators instead silently broke every
regex by turning `self._short_control.evaluate(` into spaced tokens).

### P176. [FIXED 2026-08-05] The gate's headline metric was 100% false positives and 0% recall on the bug it is named for

- **The finding, both halves.** P174 shipped `misrouted_hot_count: 10` as the number a reviewer should read first. **All ten were wrong**: four `market_data.get("data_valid", True)` and six `market_data.get("vpin_source", "synthetic")`, both keys genuinely produced by `data_mgmt/market_data_pipeline.py` and returned into `market_data`. They were flagged only because the classifier tested `written_other` (MISROUTED) *before* `produced_elsewhere`, so a key that is correctly produced **and** relayed into `system_state` (`main.py:6755`) fell into the wrong bucket. Then, checking recall by reconstructing P170's shape synthetically — `agent_signals.get("quant_data_quality", 1.0)` where the pipeline produces the key into *market_data* and nothing copies it across — it came out **PRODUCED_ELSEWHERE**, which is reported but never gated. **The scanner built to catch P170 could not catch P170.** A metric can be noise and blind at the same time; measuring only one of those tells you nothing about the other.
- **Precision and recall had the same root cause: production was tracked without a destination.** P174 credited "somebody builds this key and returns it" tree-wide, deliberately refusing to guess *which* dict it becomes. That single set cannot separate "produced into the dict you are reading" (correct) from "produced into a different one" (P170). The fix is `PRODUCER_MODULES`, keyed `(module, function) -> dict`, plus a `PRODUCED_HERE` verdict that outranks MISROUTED **only for the dict that producer actually fills**.
- **The tempting one-line repair is the dangerous one.** Plain "produced beats misrouted" clears all ten false positives — and silences P170, P173's `drl_confidence` and `phase`, and every bug the scanner exists for. The `dname` check is the whole design. `tests/test_producer_attribution.py::test_destination_is_what_distinguishes_the_two` pins it with two reads of the same key with the same default that must return *different* verdicts.
- **Module-level attribution was too coarse and was caught mid-fix.** Crediting all of `main.py` to one dict handed `market_data` every key main.py ever returns. Function-level precision was needed: `main.py::_get_effective_position_state` (`main.py:6741`) is what makes 11 correct `position_state.get("current_exposure")` reads stop being noise.
- **Correction to P174, which was itself a correction.** P174's docstring said P171 "blamed the pipeline fills `raw`… that was a guess, and it was wrong about the mechanism." **P171 was right.** `fetch_and_prepare` builds a local literally named `raw` (80 keys) and returns it. The correction was the error, and it sat in a docstring being cited as settled fact. Verify before overturning.
- **Result:** MISROUTED 26 -> 2, HOT 10 -> 0, PRODUCED_HERE 268. The 2 survivors are real: `agents/drl_agent.py:617` and `risk/short_position_controller.py:222` (the dead controller, see P174).
- **Latent finding, deliberately NOT fixed — needs an operator decision.** `core/runtime_spine.py:878` calls `DRLAgent.generate_signal(asset, price, regime)` and passes **none** of `position_state`, `market_data`, `agent_signals`. All three default to `None` -> `{}`, so the whole DRL state vector is built from empty dicts and `_validate_build_state_inputs` scores **0.20** every call (`position_state_empty`, `regime_result_none`, `market_data_empty`, `agent_signals_empty`). **This is not the live path** — `runtime_spine` is marked DEPRECATED at `main.py:14439`, has no production constructor call, and live DRL runs through `main.py` (which sets `drl_data_quality` at `:7824` and gates fusion at `integration_v36.py:2300`). Two things still want attention: the module header at `main.py:11` calls `core/runtime_spine.py` the `CANONICAL_MAIN_LOOP` while line 14439 calls it deprecated, and if that path is ever revived it will run a DRL policy on constant inputs.

### P175. [FIXED 2026-08-05] The mypy gate — the largest check in CI — had been skipping every run since P161, under a green OK line

- **The finding.** P161 (2026-08-04) correctly made the mypy baseline analyzer-version-aware: per-code counts are a fingerprint of the mypy release, so diffing across versions reports phantom regressions. On a version mismatch the guard carries the old baseline forward and prints a SKIPPED warning. But the committed baseline was written **2026-06-13** and carried no `mypy_version` key at all, so "no stamp" read as "mismatch" — and from the moment P161 landed, `ci_check_invariants.py` verified **zero type errors across the entire tree** while still printing `OK — no new findings vs baseline` and exiting 0.
- **Twelfth sighting of the P174 class, and the most expensive one.** The warning *was* printed. It scrolled past on every run, immediately above a green summary, and every session in between read the exit code instead. A check that is skipping and a check that is passing must not produce the same exit code and a near-identical screen. **If a check can self-disable, the disabled state must be loud enough to stop you, or it is not a check.**
- **Do not re-baseline to make a dead gate green — attribute the delta first.** Baseline said 1080 errors; mypy 2.3.0 said 1076. The total *fell*, which looks harmless, but five per-code counts **rose**: `arg-type` +2, `float` +2, `index` +2, `operator` +1, `var-annotated` +3. `_diff()` only fails on increases, so a blind re-baseline would have permanently accepted those ten as the new floor — laundering real regressions through a version bump. Attribution used `git archive HEAD | tar -x` into scratch (**never `git stash` on this repo**), mypy against both trees with separate cache dirs, line numbers stripped, sets compared:
  ```
  HEAD errors: 1139   CURRENT errors: 1139
  errors present NOW but not at HEAD:   (none)
  errors present at HEAD but fixed NOW: (none)
  ```
  Zero new type errors from the working tree, so the whole delta is the analyzer. Only then was the baseline re-stamped at mypy 2.3.0 / 1076.
- **Then prove the restored check can fail** (the P174 lesson, applied rather than quoted). A deliberate `x: int = "not an int"` dropped into `core/` produced:
  ```
  + mypy.by_code.assignment: count INCREASED 453 -> 454 (+1)
  + mypy.total_count:        count INCREASED 1076 -> 1077 (+1)
  ```
  Probe removed, gate back to exit 0. A restored check is assumed dead until you have watched it fail.
- **`tests/test_mypy_gate_is_live.py` (new, 8 tests)** pins both halves: the committed baseline must carry a `mypy_version` stamp *and* it must match the installed mypy (a mismatch means the gate is skipping **right now**), and the scanner must still count a known error and score clean code zero. Every one was verified to fail when perturbed.
- **Never run `ci_check_invariants.py --update` to fix this.** `--update` deliberately bypasses the version-carry-forward guard, so it silently re-stamps the mypy baseline along with everything else — the exact blind re-baseline described above. Write the individual baseline file directly.
- **Standing cost of the fix:** the new baseline accepts the +10 version-shift across those five codes. That was justified by the HEAD-vs-working-tree evidence above, not by the totals looking better.

### P174. [FIXED 2026-08-05] The scanner written to catch "a check that cannot fail" shipped with a check that cannot fail

- **The finding.** P171 gated CI on `orphan_count: 0` and treated the zero as a clean bill of health. It was arithmetically forced. `main.py` copies signal keys in loops (`for k, v in ...: agent_signals[k] = v`), which marks `agent_signals`, `market_data` and `position_state` permanently *dynamic*; every unmatched read of those three is downgraded to `UNPROVABLE` **before** it can be counted. Measured: the ORPHAN check adjudicated **0 of 458** unmatched reads. `system_state` was the only dict it could judge, and it had zero unmatched reads. The gate could not have failed under any code change.
- **This is the eleventh sighting of the class it was built to detect** (P155-L5, P156, P158, P159, P160, P164, P166, P169, P170, P171). It shipped *inside the tool*, one day after the tool was written, by the same author who wrote the docstring warning about it. Treat that as settled evidence: **vigilance does not fix this class — only a falsifiability test does.** Before baselining any metric, construct the input that makes it fail. If you cannot, the metric is decoration.
- **The soundness limit is real.** With a dynamic copy in the tree, "nobody writes key K" is genuinely unprovable. So the fix is NOT to make ORPHAN work — that would trade a vacuous check for an unsound one. The fix is to score what the scanner can prove, and to keep the vacuity visible where it remains.
- **What the rework changed (`tools/lint_orphan_signal_reads.py`):**
  - `_is_null_coalesce` — `x = x or {}` writes no keys, but was treated as an opaque alias. Eight sites used the idiom and each poisoned a whole dict. Removing that false dynamism took `UNPROVABLE` from **427 → 77**.
  - `collect_produced_keys` — the real `market_data` producer is `data_mgmt/market_data_pipeline.py`, which builds 51/55/89 keys into a *local* and returns it. 2686 produced keys tree-wide were invisible. P171 had asserted this gap existed and blamed "the pipeline fills `raw`" — **that was a guess, and it was wrong about the mechanism** (the producers build by subscript assignment, not under a variable called `raw`). It excused the right findings for the wrong reason, which is worse than being wrong, because nothing forces it to be checked.
  - `COPY_ONLY` — a key only ever *relayed* between signal dicts, with no producer anywhere. A copy is downstream of a producer, not a substitute for one; crediting relay writes as evidence hid the exact shape the scanner exists to find. Coercion wrappers are transparent (`int(market_data.get(k, 0))` is still a copy) — missing that cost the detector both its findings on the first run.
  - `FALLBACK_CHAIN` — `a.get(k, b.get(k, d))` reads both dicts for the same key. It is the hand-rolled `signal_value`, not a misroute. Flagging it would have sent a reviewer to "fix" `main.py:12086`, which is already correct.
- **The gate now scores** `copy_only_count`, `misrouted_count`, `misrouted_hot_count`, `dynamic_site_count`, `orphan_count`, `parse_failure_count` — each with a test proving it is reachable from clean (`tests/test_orphan_gate_is_falsifiable.py::TestTheGateCanActuallyFail`). **`dynamic_site_count` is not re-baselineable**: every new computed-key write makes more of the tree unprovable, so without it the other counts can always be driven to zero by making the code less analyzable.
- **`orphan_coverage_lost` is emitted but INERT** — coverage is 0, so it sits at its floor and cannot rise. It is documented and asserted as inert rather than described as protection. Recording a pinned metric as a live guard would repeat this entire pitfall one level up.
- **MISROUTED is now gated**, reversing P171's call. Crediting hidden producers shrank it to a hand-triaged list, and it is the only metric here that has ever caught a real bug (P170, and all three P173 sites).
- **What triage of the new output found:** `COPY_ONLY` independently rediscovered `is_4h_bar_close` — the key P173 had triaged *by hand* and deliberately left alone. That agreement is the evidence the detector works. Its other member, `htf_trend_direction`, has **no producer anywhere in the tree**: `main.py:9374` copies `market_data.get(...)` (never written) into `agent_signals`, `integration_v36.py:1410` reads it back, and the `[S11]` authority-fusion input is permanently `0` — which `signals/authority_fusion.py:81` documents as "no data", the fail-safe value. A dead feature, not a loss. **Not fixed**: wiring a real HTF producer is a feature, not a bugfix.
- **Also surfaced, deliberately not fixed:** `risk/short_position_controller.py` — squeeze protection, funding-rate gates, force-flatten — is **imported but never invoked**. `assess_risk` and `get_short_controller` have zero live callers; the import exists only to set `V6_MODULES_AVAILABLE = True`, which logs `"[OK] V6 SOTA modules loaded (short risk + metrics)"` at startup. **The startup line asserts a risk control that is not running.** Wiring it in would enable an untested risk path in a live system — an operator decision, not an agent one.

### P173. [FIXED 2026-08-05] Triaging P171's "too noisy to gate" list found three more constants wearing the name of a measurement

- **Why triage a metric you chose not to gate.** P171's scanner reported `ORPHAN=0` but `MISROUTED=34`, left ungated because name-based write-tracking under-counts producers (the pipeline fills `raw`, returns it as `market_data`). **"Too noisy to gate" is not "all false positives."** Hand-triaging all 34 found three real bugs — and in all three the *correct* read was sitting a few lines away in the same file.

- **1. `core/execution_service.py:541` — the DRL guard's input was the constant 0.5.** `market_data.get("drl_confidence", 0.5)`, but the producer is `agent_signals['drl_confidence']` (`main.py:7817`) — which **the same function** reads correctly ~3000 lines below when stamping `latest_drl_confidence` onto the position. `ExecutionGuard.can_drl_trade` compares it against `min_confidence_volatile = 0.7`, so in every VOLATILE regime it failed and stamped `drl_blocked_reason` onto every execution. That branch *records* rather than blocks, so the cost is diagnostic, not blocking — **but a diagnostic that always fires is exactly as uninformative as P170's guard that never fired.** Note the `[BUGFIX M7]` comment on the very next line: someone already fixed the *weight* fallback from 0.5 to 0.0 for this reason and left the *confidence* beside it untouched.

- **2. `core/execution_service.py:3606` — every position ever opened recorded `phase_at_entry="UNKNOWN"`.** Read off `market_data`, which nobody writes `phase` into; the producer is `agent_signals['phase']` (`main.py:8522`), the same dict the three `latest_drl_*` fields three lines below read correctly. Any analysis of "which market phase do our winners come from" has been reading a constant.

- **3. `core/smart_beta_controller.py:144` — a leading-underscore typo made a whole branch unreachable.** `market_data.get("phase", agent_signals.get("_phase", "UNKNOWN"))`. *Neither* key exists — `_phase` appears **nowhere else in the tree**. So `phase` was always `"UNKNOWN"` and the `TREND_STRONG` tag (line ~182, requires `phase in ("IGNITION","EXPANSION")`) could never be emitted: in a confirmed bullish regime the controller never applied its `gate_mult 0.90 / size_mult 1.10` trend-participation boost. `smart_beta_config.enabled` is `true` in `configs/live_high_risk.json`, so this one has real behavioural effect — **and fixing it LOOSENS the gate**, within the configured `alpha_gate_mult_min` / `size_mult_max` bounds.

- **Fix: one shared resolver.** `core.market_data_helpers.signal_value(key, agent_signals, market_data, default)` — agent_signals first, then market_data, then the default; `None` counts as absent, falsy values do not (`0.0` is a measurement). This is the code half of P171's CI half; neither is sufficient alone, since the scanner cannot see through a helper it does not know about. **The `ImportError` fallback stub in `execution_service.py` must read both dicts too** — a stub that quietly reads only `market_data` would restore the exact bug, on the one path nobody tests.

- **Triaged and deliberately NOT changed: `is_4h_bar_close`.** `main.py:6759` defaults it to `True` with nothing writing the key, which permanently satisfies the T1→T2 tranche escalation gate (`risk/tranche_manager.py:297`, `defense/constitution.py:1817`). That default is load-bearing and documented in place: `_process_4h_tick_inner` is reached only from loops that sleep to the 4H candle boundary, so the tick IS a bar close by construction. **Fragile, not broken** — a new caller on a faster cadence would silently unlock the gate, so `tests/test_signal_value_resolution.py` pins the caller count at 4.

- **Tests:** `tests/test_signal_value_resolution.py` (32).

### P172. [FIXED 2026-08-05] The alpha gate priced the asset at 3bps and everything downstream priced the same asset at 26bps, on the same tick

- **Two blocks, sixty lines apart, pricing the same friction.** P155e made the alpha-gate friction block (`main.py:8534`) venue-aware behind `coinbase_venue_aware_fees`, and P165 **turned that flag ON in `configs/live_high_risk.json` on 2026-08-04** by explicit operator instruction. But the `_fee_context` dict built ~60 lines later in the *same method* still hardcoded `_fc.blender.apply(0.0016/0.0026, monthly_vol)` and stamped itself `"fee_source": "kraken_plus_fee_blender"`. So for a Coinbase-routed asset the gate charged 0/3bps while every `fee_context` consumer charged 16/26bps.

- **Three consumers, and the worst is the telemetry.** `main.py:12793` is a *second* pre-trade veto (`alpha_estimated_bps < friction * 1.5`) — it kept blocking on Kraken pricing after the alpha gate had already cleared the trade on Coinbase pricing, so half the P165 loosening never took effect. `main.py:18997` accrues the paper exit fee at the wrong venue's rate. And `main.py:15827` sets `friction_fee_bps` from `fee_context`, **overriding `alpha_result.friction_fee_bps`** — meaning the dashboard reported Kraken friction for a decision that was actually made on Coinbase friction. Diagnosing from that export would have pointed at the wrong number.

- **Fix.** One pure resolver, `core/execution_service.resolve_venue_fee_bps(...) -> (maker_bps, taker_bps, venue, fee_source)`, called **once per tick** in the alpha-gate block; the `_fee_context` builder reuses the result instead of re-deriving it. `fee_context` now also carries `"venue"`, and the friction export carries `friction_venue`.

- **Two things to keep right when touching this:**
  - **The fallback direction is deliberate and asymmetric.** Flag off, RoutingPolicy says Kraken, or any exception → Kraken tier. Over-charging friction costs opportunity; under-charging spends money. Never "simplify" this into a symmetric default.
  - **It is a per-tick local (`_venue_fee_resolved`), not `self.`** `asset` is a *parameter* of `_process_4h_tick_inner`, so an instance field would carry one asset's venue pricing into the next asset's tick whenever the guarded block is skipped. Initialise it to `None` *above* the `if self._fee_blending_enabled and hasattr(...)` guard — the fee_context builder is reachable when that guard is false.

- **Missing is not Kraken.** `friction_venue` defaults to `"UNKNOWN"`, not `"kraken"`, when no `fee_context` was built. Collapsing those two is the same missing-vs-neutral mistake as P170/P171.

- **Tests:** `tests/test_venue_fee_context.py` (23). The load-bearing ones are negative: no Kraken number may ever be labelled `coinbase_venue_schedule` and vice versa, and no fallback may price below the Kraken tier.

### P171. [FIXED 2026-08-05] The reader/writer-drift class now has a scanner — and the scanner's first version reproduced the exact bug it was written to detect

- **Why a scanner.** P170 was the twelfth sighting of one bug: a consumer reads a key off one signal dict, the producer writes it into a different one, and the `.get()` default — always chosen to look reassuring — becomes the only value that key ever holds. P2, P15, P16, P23, P85, P138, P139, P140, P147, P152, P155d, P170. Twelve hand-fixes and no gate. `lint_signal_freshness.py` (P120) **cannot** catch it: it inventories agent_signals *writes* and classifies their freshness guards, so a key with no writer at all is structurally invisible to a writer census. `tools/lint_orphan_signal_reads.py` is the complement — reads with no writer anywhere in the tree.

- **Severities, and why the distinction matters.** `ORPHAN-HOT` is a read with a *non-falsy* default: the key never arrives, so the default IS the value on every call, and it asserts something positive (healthy / confident / large) that nobody measured. Both P170 defaults were this shape (`quant_data_quality` → `1.0`, `signal_edge_bps` → `50.0`). `ORPHAN-COLD` is a falsy default — still drift, but absence degrades to "nothing", usually the fail-safe direction.

- **The scanner caught itself first.** Its first run reported `ORPHAN=36` with confident-looking findings. They were all false. `main.py` starts with a UTF-8 BOM, so `read_text(encoding="utf-8")` → `ast.parse` raised `SyntaxError: invalid non-printable character U+FEFF`, and the handler was `except (SyntaxError, OSError): return [], set(), set()`. **The tree's dominant producer vanished from the scan, and a parse failure was indistinguishable from a clean file** — the same "a check that cannot fail looks exactly like a check that passed" shape as P155-L5/P156/P158/P159/P160/P164/P166/P169/P170. Fixed twice over: `encoding="utf-8-sig"` (a BOM is an encoding detail, not a syntax error), **and** a `PARSE_FAILURES` list that makes the scanner *refuse to report* (exit 2) rather than emit findings computed from a partial parse. After the fix: `ORPHAN=0`, `parse_failures=[]`. **If you add a scanner to this repo, make "I could not read the code" a distinct, loud outcome from "I found nothing."**

- **The live bug it found.** `agents/model_alpha_agent.py` read `lob_imbalance` and `spread_bps` straight off `agent_signals`. Both keys only ever exist in `market_data` (written at `market_data_pipeline.py:1888/1900`; nothing copies them across), so both resolved to `0.0` on every call — **a perfectly balanced book and a zero spread, i.e. free trading**, fed into the alpha model. Worse, because they bypassed the module's `_get` helper they never landed in `missing` either, so the coverage instrumentation built to catch exactly this reported full coverage. `main.py:7407` already carries a `[PATCH-6] Bridge micro key mismatch` comment for a neighbouring key; these two were left behind. Fixed with a `_get_either` helper: `market_data` first, then `agent_signals`, and **record a miss when neither has it**.

- **What the gate scores, and what it deliberately does not.** `ORPHAN_READS_BASELINE` locks `orphan_count`, `orphan_hot_count`, `parse_failure_count`. `MISROUTED` (written to a *different* signal dict) and `UNPROVABLE` (the dict has computed-key writes in some file, so absence cannot be proven) are **reported but not gated**. Reason: the scanner tracks writes by variable *name*, and producers legitimately build these dicts under other names — the pipeline fills `raw` and returns it as `market_data` — so most MISROUTED entries are that naming gap rather than a defect. Gating a noisy metric trains people to re-baseline it, which is how a check stops being a check. `UNPROVABLE` is surfaced in `dynamic_write_sites` so the blind spot stays visible instead of silently shrinking the finding count. **A rise in `parse_failure_count` is never re-baselineable** — it means the scan could not read part of the tree.

- **Sharp edge found while wiring this up:** `ci_check_invariants.py --update` re-seeds *all seven* baselines, and the mypy version-mismatch carry-forward is explicitly disabled under `--update` (`tools/ci_check_invariants.py:332`). So running `--update` to seed one new scanner also silently re-arms the mypy gate with the local mypy release's numbers. Check `git diff tools/scanner_baselines/` after every `--update` and revert anything you did not mean to move.

- **Tests:** `tests/test_orphan_signal_reads.py` (43). The load-bearing ones are the parse-failure tests: an unparseable file must make the scanner exit 2, a BOM must not be a parse failure, and the real `main.py` must contribute >50 writes to a scan.

### P170. [FIXED 2026-08-05] P126's staleness guard has never fired once — the producer never wrote the key, and the consumer's default said "healthy"

- **A guard with no producer.** `integration_v36.decide()` read `agent_signals.get("quant_data_quality", 1.0)` and zeroed quant confidence when it fell below 0.5. The pipeline dutifully sets `quant_data_quality` on *every* path — `setdefault(0.0)` at `market_data_pipeline.py:664` covers early returns, `1.0` at `:1314` on Best-of-N success. But it sets it in **`market_data`**, and the consumer reads **`agent_signals`** — a separate dict built as a literal at `main.py:6420` that never copied the key across. So the read always missed, the default always won, and the default was `1.0`: healthy. **P126 was written 2026-04-27 and has not excluded a single stale quant signal since.**

- **Why this keeps happening.** The guard did not fail. It *could not* fail — and from the logs those are indistinguishable, because a check that always passes emits exactly what a healthy system emits. Same shape as P155-L5, P156, P158, P159, P160, P164, P166, and the `_coinbase_fee_model_warning` in P169. The reader/writer half is the P2/P15/P16/P23/P85/P138/P139/P140/P147/P152/P155d family: **two dicts, one key, nobody checking that the writer and the reader agree.**

- **The second half: a fabricated constant on an unexercised path.** Three deadlock call sites used `agent_signals.get("signal_edge_bps", 50.0)`. Unlike the above, this key *is* always present on the live path (`main.py:6435` copies it from market_data), so **the 50.0 has not been firing and did not cause the losses under investigation** — an earlier read of mine that said otherwise was wrong. What makes it worth fixing is that it is a loaded default sitting where nothing tests it. Under the pipeline's own calibration (`signal_edge_bps = abs(quant_dir) * 65`, `market_data_pipeline.py:1318`, whose comment records avg `|quant_dir| ≈ 0.3` → ~19.5bps typical), 50.0bps implies `|quant_dir| = 0.77`. In `TrancheAwareDeadlockResolver` (`MIN_EDGE_FOR_FORCE = 1.5`, `EDGE_DECAY_PER_BAR = 0.15`) at T1 after two stuck bars against 15bps friction:

  | edge | decayed | edge/friction | resolution |
  |---|---|---|---|
  | 50.0 (fabricated) | 35.0 | 2.33x | **FORCE_AGGRESSIVE** |
  | 19.5 (typical real) | 13.6 | 0.91x | **ABORT_OPPORTUNITY** |

  Break-even is `|quant_dir| ≥ 0.494`. `FORCE_AGGRESSIVE` sets `force_execution` and switches to taker execution, so the constant would buy its way out of the patience the system was trying to exercise — on the first caller that ever builds `agent_signals` differently.

- **Fix.**
  - `main.py:6420` now propagates `quant_data_quality` into `agent_signals`, defaulting to **0.0** (unverified) rather than 1.0.
  - The consumer distinguishes **absent** from **failed** and fails closed on both: if nobody told us the quant signal was fresh, we do not assume it was. Non-numeric values fail closed too.
  - `resolve_signal_edge_bps(agent_signals, market_data)` resolves the edge **once**, with provenance (`agent_signals` | `market_data` | `ABSENT`), and never fabricates — an unresolvable edge is `0.0`, which can never clear `MIN_EDGE_FOR_FORCE`, so absence declines to force. A malformed value logs rather than falling through in silence.
  - `0.0` (flat signal — a real observation) stays distinguishable from `ABSENT` (no data) via the source field, not the number.

- **This is a behaviour change in both directions, not a pure tightening.** More `ABORT_OPPORTUNITY` is not simply "do less": that branch force-closes an existing position aggressively when `direction != 0` and `target_exposure > 0.001` (`integration_v36.py`, `ABORT+CLOSE`). Failing the dq guard closed also means a pipeline hiccup now zeroes quant confidence instead of passing it through as fresh — correct, but it will reduce exposure on ticks that previously traded. Watch `[P170]` and `[P126]` log lines after deploy to see how often either path is actually reached; if `quant_data_quality` turns out to be 0.0 more often than expected, the bug is in the pipeline's selector, and it has been invisible until now.

- **Tests.** `tests/test_signal_absence_provenance.py`, 34 tests. Two of them read `main.py` and `integration_v36.py` as text to assert the producer still emits the key and that neither fabricated default (`50.0`, `1.0`) has resurfaced — the reader/writer contract is what breaks, so the contract is what gets pinned.

### P169. [FIXED 2026-08-05] The venue told us what it charged and we threw the number away — every fee in the attribution log is modelled, and the model was priced on the wrong exchange

- **The fee column was never an observation.** `execution/execution_manager.py:1462` parses the fee straight out of the ccxt order response into `OrderResult.fee`. Nothing downstream read it. `paper_fee_service.build_execution_fee_result` computed its own number from a schedule and wrote *that* into `data/trade_attribution.jsonl`. So the file that answers "what did trading actually cost us" contained no measured cost at all.

- **The fingerprint.** A modelled constant does not look like real fills, and the data says so plainly:

  | leg | records | median | at *exactly* 16.0bps |
  |---|---|---|---|
  | entry | 52 | 16.0bps | 32 / 52 (62%) |
  | exit | 90 | 16.0bps | 59 / 90 (66%) |

  Round trip 32.0bps. Real fills scatter; two thirds of them landing on the same round number is a constant wearing a measurement's clothes.

- **Why 16.0 specifically — two bugs multiplying.**
  1. `fee_std` was hardcoded to Kraken's `0.0016 / 0.0026` regardless of where the order executed, and the Kraken+ blender was called with `exchange="kraken"` hardcoded too. Post-cutover (2026-06-13) that prices Coinbase-routed fills on the wrong exchange — Coinbase nano perps are 0/3bps against Kraken's 16/26. It also fed Coinbase fills into Kraken's monthly-volume tracker, earning them free-tier discounts they never qualified for (the blender gates on `exchange.lower() != "kraken"` at `kraken_plus_fee_blender.py:454`, so passing the real venue makes it correctly decline).
  2. `is_maker = order_type == "LIMIT"` — that is the **order** type, not the **fill** type. A limit order that crosses the spread is filled as a taker and charged as one. **This is the third sighting of this exact conflation**: P166 found it in `review_aggregator.maker_fee_ratio` (`n_limit / n_classified`), and it is the same shape as the reader/writer drift class. An intent is not an outcome.

  Together: every LIMIT order booked Kraken's *maker* rate, 16.0bps, whichever venue it hit and whether or not it actually made.

- **What could not be established from the laptop, and is not claimed.** Over those trades: gross_alpha **−$220.74**, recorded fees **$708.57**, net **−$929.31**. At Coinbase's 3bps/leg the same volume would be ~$81.76. That ~8.7x is **conditional on those fills having been Coinbase-routed, which cannot be determined from this machine** — `data/coinbase_sleeve_pnl.jsonl` is server-side only. What *is* established is that $708.57 is modelled rather than measured. Note also that **gross_alpha is negative before any fee at all**, so this changes the magnitude of the loss, not its sign, and does not weaken P166's or P167's conclusions.

- **The pre-existing warning nobody acted on.** `_coinbase_fee_model_warning()` in `core/execution_service.py` already printed that the Kraken fee model over-charges Coinbase-routed assets, ending with "NOT auto-corrected". A warning that fires into a log and changes no behaviour is the same failure mode as P155-L5/P156/P158/P160/P164/P166: a check that cannot act is indistinguishable from no check.

- **Fix.**
  - `OrderResult.to_dict()` now emits `fee_currency`. A fee without its denomination is not a fee — `0.0031` means something very different in USD and in ETH.
  - New `resolve_trade_fee_usd()`: the venue's number wins when usable, and every rejection records **why** in `fee_source_reason`. Rejected: `None`, non-numeric, NaN/inf, negative, non-USD denomination (deliberately **not** converted — a wrong FX rate is worse than a model, because it carries the authority of an observation), and **`0.0`**. That last one matters: `OrderResult.fee` is `float((order_status.get('fee') or {}).get('cost', 0) or 0)`, which collapses *missing* and *genuinely zero* into the same value, so a zero cannot be trusted to mean a free trade. Missing-vs-neutral again (P2/P15/P23/P138/P147/P152).
  - `venue_fee_std(venue, is_maker)` replaces the hardcode. Unknown venue falls back to **Kraken**, the expensive one, because over-charging fails toward not trading.
  - Provenance travels with the number: `fee_source` ∈ `venue` | `model` | `disabled`, plus `modelled_fee_usd` kept alongside the measured one so model error becomes measurable after the fact. `disabled` is distinct from `model` so that $0.00-because-off is readable apart from $0.00-because-free.
  - `is_maker_assumed: True` is set unconditionally. The assumption is retained as the model's best guess but is no longer passed off as fact.
  - New `[FILL_VS_MID]` log: realised slippage (fill vs decision price, signed by direction) against the assumed spread. When the friction object is unreachable it prints `assumed_spread=UNAVAILABLE` rather than substituting a default — a substituted constant is how the original bug got its authority.

- **Still wrong, deliberately out of scope.** `main.py:8593` builds `fee_context` from the same hardcoded `0.0016/0.0026` and labels it `fee_source: "kraken_plus_fee_blender"`. That context feeds the **exit** leg and the alpha gate's friction, so both are still Kraken-priced on every venue. Making it venue-aware needs the routing decision, which happens after `fee_context` is built — an architectural change, not a bug fix, and not something to bundle into a fee-provenance pass. It fails toward over-charging (fewer trades), so it is safe to leave pending.

- **Tests.** `tests/test_venue_fee_provenance.py`, 82 tests. The load-bearing ones are negative: a modelled number must never be labelled `fee_source="venue"`, no rejected value may be smuggled into `venue_fee_usd`, and no venue may price *above* the pre-P169 Kraken model (so this fix cannot manufacture losses that were not already booked).

### P168. [FIXED 2026-08-05] The rebuild cooldown exempted direction flips — it waived the churn that costs the most, on the path it fired most often
- **The carve-out.** `execute_intent_v2` (`core/execution_service.py`) set an 8h cooldown after every close and blocked new entries during it, *except* an entry opposite to the closed position:
  ```
  # Cooldown is designed to prevent same-direction re-entry churn.
  # Opposite-direction entry is a signal-aligned reversal, allow it.
  ```
- **It has the cost backwards.** A reversal pays a full round trip to close and commits another to open (**P167**). Inside a cooldown window it is the *most* expensive thing that can happen, not the natural exception.
- **It swallowed its own dominant case.** A close is usually *caused* by the signal turning, so the next entry is opposite **by construction** — that is what "reversal" means. The exemption therefore fired on the common path and left the cooldown binding only when the signal reversed and then reversed *back* inside 8h. Measured over the 52 timestamped closed trades in `data/trade_attribution.jsonl` (total net **−$540.07**):

  | re-entry after a close | inside 8h | outside |
  |---|---|---|
  | FLIP (was exempt) | **10** | 17 |
  | SAME-DIR (was blocked) | 5 | 17 |

  **10/15 = 67%** of in-window re-entries took the exemption. Those 10 flips: **8 losers, net −$94.45**, mean −$9.45. **Six of the 10 opened at 0.0h** — the exit and its reversal landed in the same 4H bar, two round trips of friction inside one candle. One ETH sequence ran +$8.46 → flip +$2.18 → flip **−$31.54**.
- **A narrower carve-out is not supported either.** Splitting by whether the *closed* trade won: after-loser **7 trades, 7 losers, −$67.75**; after-winner **3 trades, −$26.70**. Neither subgroup is profitable, and n=3 is not a rule. So the exemption is **off entirely** (`REBUILD_COOLDOWN_EXEMPT_FLIP = False`) rather than conditioned. Restoring it is a one-line config change; the add-on exemption (`REBUILD_COOLDOWN_EXEMPT_ADDON`) is **untouched**, since a pyramid adds to a position that is already winning and is not a re-entry at all.
- **This delays a reversal, it does not forbid one** — and it can never block an exit, because the cooldown check is gated on `is_new_entry or is_adding`. The worst case is staying flat for up to 8h. 17 of the 27 observed flips were already outside the window and are unaffected.
- **Why it was never caught: it was unreachable from a test.** The decision sat inline in a ~2000-line async function needing a full runner, positions, market data and an event loop. Extracted to a pure `rebuild_cooldown_decision(...)`; `tests/test_rebuild_cooldown_flip.py` (92 tests) now pins the grid, including a property test that the change **only ever tightens** and that every old-vs-new disagreement is exactly an in-window flip. **If a branch has no reachable test, assume nobody has ever checked whether it is right.**
- **Missing direction must block, not exempt.** Cooldown entries written before the closed-direction field existed are 2-tuples; absent direction reads as `0`, which cannot be a flip and therefore blocks. Same family as P2/P15/P138/P152 — *do not let unknown collapse into permissive*.
- **Latent bug preserved on the legacy path, deliberately.** The exempt branch does `del ctx.rebuild_cooldown[asset]` on a **check** path, before the trade is known to execute. A flip later rejected by the AC-0 restart guard has still consumed the cooldown, so the next tick's same-direction entry sails through. Left as-is because that branch exists to reproduce pre-P168 behaviour exactly — and it is a further reason not to switch it back on.

### P167. [FIXED 2026-08-04] The alpha gate charged **one leg** of friction against a **round-trip** alpha estimate — it could not reject a trade whose only problem was that it has to be closed
- **The arithmetic.** `AlphaThresholdCalculator.check_alpha_gate` (`defense/constitution.py`) computed `friction = fee + slippage + latency + margin` and required `alpha >= friction × multiplier`. With `NORMAL_MULTIPLIER = 1.10` that demands **1.10 legs** of cost from a position that pays **2** — entry and exit. Every trade the system has ever placed was priced as if it were half a trade.
- **What it let through.** At Coinbase taker (3bps) with the live pipeline's own calibration (`data_mgmt/market_data_pipeline.py:1318` sets `signal_edge_bps = |quant_dir| × 65`, and the comment next to it puts average `|quant_dir|` at **~0.3** → 19.5bps raw, **14.6bps** after the 0.75 ALPHA-FEEDBACK haircut):

  | asset | old friction / threshold / passes | new friction / threshold / passes |
  |---|---|---|
  | BTC | 8.0 / 8.8 / **True** | 16.0 / 17.6 / False |
  | ETH | 10.0 / 11.0 / **True** | 20.0 / 22.0 / False |
  | SOL | 15.0 / 16.5 / False | 30.0 / 33.0 / False |

  The *typical* live signal cleared the BTC and ETH gates while being a guaranteed loser. This is a direct arithmetic explanation for the P166 attribution numbers — gross_alpha **−$179.51** against **$689.77** of fees over 85 closed trades. The gate was not leaking; it was correctly enforcing the wrong inequality.
- **New minimum `|quant_dir|` to clear the gate:** BTC **0.37**, ETH **0.46**, SOL **0.68** (was 0.18 / 0.23 / 0.34).
- **The fix.** `FrictionComponents` gained `per_leg_bps(is_maker)` and `round_trip_bps(is_maker, legs=2.0)`; `check_alpha_gate` now charges `ROUND_TRIP_LEGS × (fee + spread + latency) + margin`. `AlphaGatingResult.friction_legs` reports which arithmetic produced the decision — it defaults to **0.0**, not 1.0, so the early-return paths read as "no friction was priced" rather than silently claiming one leg. The REJECT_EV reason string now spells out `2x8.0bps/leg`.
- **Margin/funding is deliberately NOT doubled.** `_margin_cost_bps` is `opening_fee + rollover × expected_hold_periods_4h` — it is a per-HOLD cost already integrated over the hold, not a per-ORDER cost. Doubling it would charge a 24h position as if it were held 48h: a second, unrelated bug wearing this fix's clothes. `tests/test_round_trip_friction.py::test_margin_is_charged_once_not_twice` pins this.
- **Kill switch:** `HMATS_ROUND_TRIP_FRICTION=0` restores the one-leg arithmetic. Default is **ON** because the safe failure direction is charging too much, not too little; parsing is fail-safe (only the literal `"0"` disables it) and it is read at construction, so it is not a hot toggle.
- **Why the existing tests did not catch it.** Sixteen tests encoded the one-leg number as a literal (`7.7`, `36.3`, `friction 7.0bps`) — they pinned the bug rather than the contract. Updating them required *rescaling inputs*, not relaxing assertions: several `min_alpha_bps` floors (10.0) had been chosen to sit above a one-leg EV gate and now sat below the two-leg one, so those tests would have kept passing while silently no longer exercising the branch they were named after. **When a threshold moves, check that each test's inputs still reach the code path in its own name.**
- **What this does not fix.** The gate is only as good as `signal_edge_bps`, which is still a hand-calibrated `|quant_dir| × 65` rather than a measured forward return. And `integration/integration_v36.py:1254` never passes `is_maker`, so every gate decision assumes taker — correct today (P155/P156 confirmed the venue is taker-dominated), but it is an assumption, not an observation.

### P166. [FIXED 2026-08-04] The shadow promotion gate had no cost term, no significance term, and `abs()` on the promote branch — its pass mark sat *below* break-even
- **The gate was one line:** `if min(|IC|) > 0.05 and sharpe > 0.5: return PROMOTE` (`analytics/shadow_ic/compute_shadow_ic.py`). `promotion_gate/promotion_plan.py:127` maps PROMOTE straight to `PROMOTE_TO_FUSION`. So this line decides what gets to trade real money, and each of the three defects below is on its own sufficient to promote a strategy that is arithmetically certain to lose.
- **Defect 1 — no cost term, at all.** IC is dimensionless; fees are in bps; the function never converted between them, so `IC > 0.05` could not possibly know what it was clearing. Priced: expected edge = `E|z| · r_pearson · sigma_fwd` = `0.7979 · 2sin(pi·rho/6) · sigma`. At the ~107bps of 16h forward vol these assets show, **IC 0.05 is worth 4.5bps** — against **6bps** of Coinbase taker fee (3bps × 2 sides) before any spread. Break-even needs **IC 0.134**. The old bar was 2.7× too low at the shortest horizon.
- **The cost number the system believed was ~400× wrong.** `training/backtest_framework.FeeSchedule` defaults to `maker_pct=0.987`/`maker_bps=0.0`, giving `round_trip_bps() = 0.078`. Measured over the 85 closed trades in `data/trade_attribution.jsonl`: **median 31.1bps, mean 33.0bps** round trip (min 16.0, max 184.7). Sum gross_alpha **−$179.51** vs sum fees **$689.77**. And the `maker_fee_ratio = 0.994` "PASS" in `sixty_day_review/review_aggregator.py:376` does **not** support the 98.7% assumption — it is `n_limit / n_classified`, i.e. **order type, not fill type**. A limit order that crosses the spread pays taker and still counts as maker here. The new gate assumes **100% taker**, deliberately.
- **Defect 2 — no significance term, and `min_samples=30` made it worse.** `SE(IC) ≈ 1/sqrt(n−1)`, so at n=30 an IC of 0.05 is **0.27 standard errors from zero**. Thirty 4H samples is about five days. The gate could not distinguish an edge from a coin flip and was reachable within one shadow week. Getting IC 0.05 to |t| ≥ 2 needs **n ≈ 1600**.
- **Defect 3 — `abs()` on the promote branch, with nothing downstream to flip the sign.** `valid_ics = [abs(ic_per_h[h]) ...]` fed the promote comparison, and `decide_strategy_action` has no sign handling anywhere. P143 measured `model_alpha` at IC **−0.160** and `llm_sentiment` at **−0.053**. Under the old gate a strongly anti-predictive strategy was *more* promotable than a weak one, and fusion would then have traded it in the direction it predicts against.
- **The fix.** New `assess_promotion()` requires, at **every** horizon with enough samples: IC positive; `|IC| > 0.05` floor; `|t| = |IC|·sqrt(n−1) >= 2.0`; and `expected_edge_bps >= 6.0bps × 2.0 margin` priced off the **measured** forward-return volatility. The margin covers spread/impact (absent from the fee number) and the optimism of the linear edge model. `determine_verdict()` is kept as a thin wrapper so existing callers are unchanged.
- **Fail closed on the new bar.** `compute_per_strategy_ic` now emits `fwd_vol_bps_per_horizon`, measured from the *same* joined pairs the IC is computed on. A horizon with <2 pairs gets **no entry at all** rather than `0.0` — and a missing/zero/NaN vol is a **refusal to promote**, not a skipped check. Same lesson as P159/P164: a check that could not run must never read as a check that passed. Reports written before this change have no vol key, so they degrade to "cannot verify", never to "verified".
- **KILL and INSUFFICIENT_SAMPLES semantics are byte-identical to the old gate**, and KILL still uses `|IC|` (a strongly negative IC is informative, not weak). Every new condition only ever *removes* a PROMOTE, so the gate cannot have become looser — `tests/test_promotion_gate_cost_aware.py` pins that property parametrically against the old implementation.
- **Two call sites had been deriving the verdict independently** (`render_summary` and the JSON report each called `determine_verdict` with their own arguments) — a console PROMOTE and a report HOLD from one run was a live possibility. Both now route through `assess_record()`. The summary also prints the arithmetic per horizon (`edge= 5.54bps need= 12.00bps vol= 107.0bps IC=+0.0620 (req 0.1343) t=1.84`) and every blocker, because a HOLD that does not say which bar was missed cannot tell an operator whether to wait or to archive.
- **What this does not do:** it does not create edge. Applied to the ICs actually observed, essentially everything currently in shadow now returns HOLD with an explicit shortfall. That is the correct reading, and it is the point — the previous gate was reporting these same strategies as promotable.

### P165. [FIXED 2026-08-04] `core.canonical_imports` has never been importable, so `activate_runtime_mode()` never ran — plus a stale-test sweep in which one test was silently issuing real billed Anthropic API calls and another was corrupting `place_stop_loss` for every test after it
- **The load-bearing one: an import that could never succeed, swallowed by an `except ImportError` five lines below it.** `core/canonical_imports.py:229` imported the four sentiment-contract symbols from `engine.compute.vllm_inference_wrapper`, a module that lives in `archive/engine/compute/` and is not on the production path. So the import raised `ModuleNotFoundError` and made **the entire `core.canonical_imports` module unimportable** — for the whole history of this repo. `main.py:19861` wraps its runtime-protection block in `except ImportError: logger.warning(...)`, so **`activate_runtime_mode()` has never run in any process**: the canonical-import enforcement it turns on was silently off in production, and the `[STARTUP] SENTIMENT_MODE=` line an operator greps for was never emitted. Same family as **P152** (a guard defined, unit-tested, and never called) and **P155d** (an Iron Law with no production caller) — except here the swallow sat five lines from the only statement that could raise.
- **Fix:** new `core/sentiment_config.py` is the live home for the contract, and `canonical_imports` points at it. The semantics of `is_sentiment_mock` changed **deliberately**: the archived `IS_MOCK` was `not (VLLM_AVAILABLE or TRANSFORMERS_AVAILABLE)`, describing a local-vLLM inference architecture this system no longer has. Live LLM sentiment is Haiku over an HTTP API, so mock-ness is now a property of whether an API key is configured.
- **Authority-matrix drift (rule #7 violation, now corrected).** Commit `795ecc4` added `v5_1_strats` to `AUTHORITY_MATRIX_NORMAL` on 2026-06-13, taking the declared count **25 → 26** and the fusion-consuming count **19 → 20**, without touching the §Authority Matrix table here. `tests/test_authority_fusion.py` and `tests/test_docs_runtime_parity.py` both pinned 25, so the drift read as a *test failure* rather than as the doc bug it was. The table now lists all 26 with `v5_1_strats` at row 21; the parity test asserts the table against the live matrix rather than a literal, so the next addition fails loudly on the doc and not on an arbitrary number.
- **A test was issuing real, billed Anthropic API calls on every run.** `LLMSentimentAgent(api_key="")` does **not** disable Haiku: the constructor is `api_key or os.environ.get("ANTHROPIC_API_KEY", "")` (`sentiment_llm_agent.py:466`) and `""` is falsy — so every test in `tests/test_sentiment_llm_agent.py` that passed `api_key=""` to mean "no Haiku, exercise the fallback" instead picked up the developer's ambient key, took the Haiku branch, and **hit the paid API**. It also made three tests fail, because they asserted the fallback source (`f&g`) and got `haiku`. An autouse `monkeypatch.delenv("ANTHROPIC_API_KEY")` fixture fixes all three at once. **Generalise this:** a constructor that falls back to an ambient credential turns "I passed nothing" into "use the operator's real key" — in a test suite that is both a correctness bug and a billing one, and it is invisible on a machine that has no key set.
- **A test's own teardown broke the code under test for the rest of the process.** `tests/test_mutation_audit_p122.py` saved `original = ExecutionManager._generate_stop_userref` before monkeypatching it. Attribute access **unwraps** a `staticmethod` and returns a plain function, so the `finally` restored it as an *instance* method. Every later call to `place_stop_loss` (`execution_manager.py:2298`) then bound `self` to `symbol` and died with `TypeError: got multiple values for argument 'suffix'`. All 8 tests in `tests/test_stop_order_retry_policy.py` passed in isolation and **7 failed in a full-suite run** — the classic signature of leaked global state, and it had been misfiled as flakiness. Fix: capture the descriptor via `ExecutionManager.__dict__["_generate_stop_userref"]`. **When monkeypatching a `staticmethod`/`classmethod` on a class, save and restore from `__dict__`, never from attribute access.**
- **Maker-reprice KPI was dropped in exactly the case it exists to measure.** `[MARKET-FALLBACK 2026-04-15]` re-enabled a market fallback when the reprice loop exhausts with <10% filled, but `_execute_market_order` builds a fresh `OrderResult` — so `maker_reprice_attempts`/`maker_reprice_cancel_count` were reset to 0 on the maker-**starved** path. "Tried 3 times and gave up" and "reprice never ran" produced identical telemetry. The counters are now carried forward. The companion `[REPRICE] DEFERRED … - no taker fallback` log was also **false** from that same date — it was routinely followed one line later by "falling back to market", so an operator grepping the old wording concluded no taker fee had been paid. It now states what the maker path decided and defers the fallback question to the caller.
- **Doc-vs-code drift in the PA executor.** `PassiveAggressiveExecutor`'s class docstring said "SHADOW (default)" long after `PAExecutorConfig.mode` and `from_dict` were both promoted to `"ACTIVE"` (`passive_aggressive.py:472/481`). Three tests were written against the docstring rather than the field. Docstring corrected; note `shadow_mode = (mode != "ACTIVE")`, so an unrecognised mode string degrades to SHADOW, which is the safe side.
- **The rest of the sweep: ~44 tests whose *premise* had silently stopped holding.** These are not the same as tests that broke because behaviour regressed — in each case the code was right and the test had quietly stopped exercising the thing it was named after:
  - **Two epsilon tests could no longer fail.** `test_min_alpha_bps_staging.py` hardcoded signals chosen when the EV multiplier was 1.25×; at 1.10× they clear the bar unaided, so the gate returned a plain `ALLOW` and the tests asserted `ALLOW_EPSILON` on inputs that need no epsilon. Replaced with a helper that *searches* for a genuinely EV-rejected signal and measures the shortfall, then asserts the epsilon admits it and half the epsilon does not. Same shape: `test_progressively_tighter_prices` used a book so narrow that all three prices clamped to the same cap, so its ordering assertion held trivially.
  - **One test was self-contradictory** — it built a `TradeIntentV36` with `direction=-0.018` against its own `opportunity_actionable_direction_threshold_short=0.04`; since `is_actionable` is a conjunction, it could not pass on any exposure setting.
  - **Several depended on machine state**: `test_health_validator.py` needed `data/drl_promotion_state.json` (runtime state, absent in a checkout, so the watchdog returned SKIP); `test_model_alpha_agent.py` asserted `_model_loaded is True` while `models/` is gitignored.
  - **Several pinned config literals that were legitimately retuned** (`_cache_ttl` 1h→4h, epsilon 0.25→1.5, short direction floors, the staged min-alpha schedules, `regime_alpha_gate_relax_short` 0.80→1.1 — which despite its name is now a *tightening* multiplier).
  - **Two pinned behaviour that was deliberately removed**: a `CRITICAL` DRL-authority severity retired on 2026-04-30, and the pre-`[MAKER-PRICE-FIX 2026-04-15]` post-only pricing whose "go deeper below the bid" semantics guaranteed a 0% fill rate.
- **The rule applied throughout: rewrite a rotted literal into the invariant it was reaching for, not into a fresh literal.** Parity against the canonical source (docs vs the live matrix, `from_dict` vs the dataclass), or a structural property (monotone staged schedules, short-leg never looser than long, a post-only price strictly inside the opposite touch). A re-pinned constant fails again on the next legitimate retune and trains the reader to update tests without reading them.
- **`tests/test_min_alpha_bps_staging.py` was committed with only the `[P165]` hunks.** A concurrent session's `[P167]` round-trip-friction expectations are interleaved through the same file line-for-line; those assert against a `defense/constitution.py` change that is not in this commit, so shipping them here would leave the tree failing its own suite. The 15 `[P167]` hunks were reverted in the committed blob only — they remain intact in the working tree for P167 to commit. Verified: the exact committed tree runs **2551 passed / 0 failed**.
- **Result: 2748 passed / 0 failed, stable under both `-p no:randomly` and randomised ordering.**
- **Live-risk flags flipped in the same change, by explicit operator instruction ("loosen live risk"), and both are OVERRIDES rather than validation passes** — each is annotated in `configs/live_high_risk.json` with its evidence gap and its revert:
  - `v5_1_strategies_live` **false → true**. Resolves the open operator decision left by P155 (commit `795ecc4` was titled "FULL PROMOTION … live ADVISE" while the production config had it off). The P147 re-enable criterion — `compute_shadow_ic` IC>0.05 over 30d — has **not** been met and cannot be evaluated from the operator laptop (the shadow ledgers live on the container volumes). Note P166 moved that bar: under the new cost-aware gate, IC 0.05 is *below* break-even. Bounded by ADVISE authority, the 0.50 net cap (P144), the 20%/asset size cap and the −5%/28d existence fuse.
  - `coinbase_venue_aware_fees` **absent → true** (it had no entry in the live profile, so it took the `False` default). Post-Phase-B every routed asset executes on Coinbase (~3/0bps) while alpha-gate friction was priced off Kraken's fee tier (~26/16bps), systematically shrinking `target_exposure` — P155e's leading candidate for the `ZERO_EXPOSURE` blocker. This makes the gate price the venue that actually executes, which is the *correct* number, but it is still a real loosening and it was **not** confirmed as the blocker first (`scripts/why_no_trade.py` needs the server). Wrong-way failures fall back to the Kraken tier and log `[VENUE-FEE]`.
- **Mitigation pattern (the through-line of P158/P159/P160/P162/P164 and this entry):** a check that cannot fail, a tool that is not installed, a writer that stopped writing, and a test whose premise no longer holds all produce output *indistinguishable from success*. Before trusting a green test, confirm it can still go red — the fastest way is to break the thing it claims to guard and watch it fail.

### P164. [FIXED 2026-08-04] Two lookahead leaks in the training pipeline — the wavelet denoise was non-causal, and the GMM fit on 100% of history because of a one-character path typo
- **Why this is the most consequential entry in the recent set:** it means the DRL's reported per-fold validation Sharpe (BTC +9.22 / ETH +7.32 / SOL +10.29, header table) is **not evidence of edge**. Measured directly: applying `wavelet_denoise` to a whole column of a **pure random walk** — zero predictability by construction — yields **IC +0.41 vs the next-bar return** (Sharpe ~+16). The reported backtest Sharpes sit *inside the range the leak produces on noise*, against a live DRL IC of **+0.052** (P143). This is the mechanism behind the P40/P41 "backtest-IC vs live-alpha gap", and it explains why CSCV-PBO reported ROBUST_SELECTION while the account lost money: **no date-based split removes it, because the contamination is in every row.**
- **Leak 1 — `training/scripts/wavelet_denoise.py` is not causal.** VisuShrink computes `sigma = median(|coeffs[-1]|)/0.6745` and `threshold = sigma*sqrt(2*ln(N))` over the **whole array**, and the inverse transform reconstructs every sample from every coefficient. `rebuild_pipeline.py` called it on the full history, so each training row was a function of all future rows. **Live does something different**: `data_mgmt/market_data_pipeline.py:853-866` applies it to a trailing 256-bar deque and takes the last value. Two different transforms, silently — a train/serve skew on top of the leak.
- **Fix:** added `wavelet_denoise_causal()` (rolling `RUNTIME_WINDOW=256` / `RUNTIME_MIN_SAMPLES=8`, output[i] depends only on signal[:i+1]) and pointed `rebuild_pipeline.py` at it. The leaky function is **kept, not deleted** — it is the right transform for offline visualisation — with a loud docstring warning. ~11s to rebuild, so there is no performance argument for reverting.
- **Leak 2 — `train_per_asset_gmm.py:load_split_manifest` read `config/split_manifest.json`; `generate_split_manifest.py` writes `configs/` (plural).** `config/` exists (it holds `optuna_winner.json`), so the path resolved without error and simply never matched. The loader then **returned `{}`**, `train_end` arrived as `None`, and `train_gmm_for_asset` logged "Using ALL data for GMM fit" — fitting the scaler, the GaussianMixture, the BIC k-selection and the cluster naming on 100% of history, then emitting `regime_proba_0..7` for every bar. **Iron Rule #12 has never been enforced by this script.** Eight contaminated features on every run it ever had.
- **The typo is the small half. The dangerous half is the fallback** that treated "I could not find the boundaries" as "proceed without boundaries". `load_split_manifest` now **fails closed** (`FileNotFoundError` / `ValueError`), which is safe because `--no-split` already exists as the explicit way to ask for a full-sample fit.
- **`scripts/runtime_parity_check.py` was supposed to cover exactly this and only asserted the five denoised column *names* exist in the manifest** — a shape check reading as a value check. Same family as P158 (a pattern that matched nothing) and P159 (a missing tool recorded as a pass): **a check that cannot fail is indistinguishable from a check that passed.**
- **Tests:** `tests/test_wavelet_causality.py` (10) — asserts causality *directly* by mutating the future and requiring the past not to move; pins the old transform's leak as a characterisation test; asserts the causal form reproduces the live deque recurrence **exactly**, bar for bar; guards the call site, not just the function. `tests/test_gmm_split_manifest_failclosed.py` (6).
- **NOT done here (server-side, needs `training/training_data/`):** rebuilding the parquets and retraining. **Until that happens the deployed models are still the contaminated ones** — this fix only stops the next build from being poisoned. Expect the honest Sharpe to be far below the reported one; that is the point.
- **Mitigation pattern:** any feature transform must be verified causal by *construction test*, not by inspection — perturb a future sample and assert nothing earlier moves. And when training and serving compute "the same" feature through different code paths, assert the two agree numerically on a shared series, not that the column names match.

### P163. [FIXED 2026-08-04] `_current_drawdown_pct` had exactly one writer and it was inside `run_paper` — every drawdown-scaled risk control read a permanent 0.0 in LIVE
- **The whole de-risking ladder was disarmed in the only mode that risks real money.** Consumers: `main.py:11797` regime-leverage reducer (DD>22% → force 1x), the DD halt, and `main.py:19448` the **DRL's own observation vector** (`drawdown` is one of the 4 env-state dims of the 126-dim space). All three read `getattr(self, '_current_drawdown_pct', 0.0)`. `run_live` never assigned it. So the system believed it was at its all-time high no matter how far equity had fallen — and the DRL was served a state it was never trained on.
- **Invisible by construction, which is why it survived:** a defaulted `getattr` makes "never written" and "zero drawdown" the *same reading*. Nothing could observe the gap. Same missing-vs-neutral collapse as P2/P15/P16/P23/P85/P138/P139/P140/P147/P152/P155d — the most common bug class in this repo.
- **Fix:** extracted the inline `run_paper` block into `HMATSProductionRunner._update_drawdown_snapshot()` (one writer, both callers), and called it in `run_live` **at the top of the tick, before any decision** — a post-trade update would gate the *next* tick, not the one it is meant to gate. Equity-fetch failure **holds the last known drawdown** rather than recomputing from `initial_capital`: falling back to notional would report "no drawdown" precisely when the venue is unhealthy, i.e. switch de-risking off at the worst moment. **A stale drawdown is conservative; a fabricated zero is not.**
- **Tests:** `tests/test_live_drawdown_tracking.py` (11) — peak ratchets and never falls back, pipeline kept in sync, the fetch-failure hold, and three wiring tests that are the actual regression: both loops call the snapshot (parametrized over `run_live`/`run_paper` via `inspect.getsource`), LIVE calls it *before* `process_4h_tick`, and a regex count asserting **exactly one** writer of `_current_drawdown_pct` so a second inline copy cannot reappear and drift.
- **Mitigation pattern:** when a value is read with `getattr(self, X, <neutral>)` in three places, grep for its **writers** before trusting any of the readers. If there is exactly one and it sits inside a mode-specific branch, the other modes are running on the default — silently, forever.

### P162. [FIXED 2026-08-04] The phantom "312 consecutive blocked ticks" was manufactured by the alert's own code path — a P152 routing skip was being stamped as a veto *after* execution
- **Closes the loop on P155.** `_process_4h_tick_inner` treated **any** non-fill from `execute_intent_v2` as a veto: `veto_active=True`, `veto_reason="[EXECUTION] …"`, `target_exposure=0.0`. Since the 2026-06-13 cutover every asset is Coinbase-routed and Kraken-flat, so P152 makes `execute_intent_v2` return `{"status":"SKIPPED","reason":"coinbase_routed_no_kraken_entry"}` on **every tick by design**. The stamp therefore flipped an intent that had passed all seven gate layers into a non-actionable one, and `PerTickInvariantChecker._t3_intent_actionable` dutifully counted it as blocked. **312/312.** HEALTH_T3 was measuring a retired code path, and P155/P155b/P155c were three rounds of diagnostics spent on a blocker that did not exist.
- **The distinction the fix encodes: "this path had nothing to do" ≠ "this trade was blocked."** `core/execution_service.is_benign_exec_skip()` + `BENIGN_EXEC_SKIP_REASONS` (exactly two: `coinbase_routed_no_kraken_entry`, `No active position to close`). Benign → record `TradeIntentV36.execution_skip_reason` for observability and **do not** touch `veto_active`/`target_exposure`; it deliberately feeds neither `is_actionable` nor any health counter. Everything else still latches, still zeroes exposure — the fix must not disarm a genuine safety response.
- **Deliberately placed in the producing module, not in `main.py`.** The classifier lives next to the three `return {"status":"SKIPPED"}` sites it classifies, so the reason strings have one home. Duplicating them at the consumer is exactly the reader/writer drift that produces P2-family bugs.
- **The real veto branch is now `logger.warning("[EXECUTION-VETO] …")`.** The old line was `INFO` and tagged `[EXECUTION]` — indistinguishable from routine execution logging, which is a large part of why a 312-tick streak went unexplained for ~7.5 weeks. A genuine execution veto is now greppable.
- **Tests:** `tests/test_benign_exec_skip_not_veto.py` (12) — the classifier fails **closed** (a benign reason on a non-`SKIPPED` status is still a real failure), the benign set is pinned so it cannot quietly grow into blanket suppression, and two end-to-end streak tests against the real `PerTickInvariantChecker`: 312 routing skips leave the streak at **0**, while 12 real vetoes still escalate to **CRITICAL**.
- **Mitigation pattern:** a health counter must be derived from state that exists *before* the thing it measures. Stamping a decision field after execution and then alarming on that field means the alarm can only ever describe its own side effect.

### P161. [FIXED 2026-08-04] Installing the missing tools revealed two more "the check never ran" bugs — the mypy baseline is analyzer-version-specific, and ~30 async tests were failing purely because a declared dev dep wasn't installed
- **Direct consequence of P159.** Once `pip install mypy` restored the check, the deploy gate immediately failed with **10 "NEW findings"** — `arg-type +2`, `float +2`, `index +2`, `operator +1`, `var-annotated +3` — with **no code change behind any of them**. The total had gone *DOWN* (1080 → 1073); mypy 2.3.0 simply **reclassifies errors between codes** versus the 1.x that produced the baseline, and `_diff` only flags increases, so the redistribution surfaced as pure phantom regressions.
- **The baseline is a fingerprint of the analyzer, not only of the code.** A cross-version comparison is neither a pass nor a fail — it is a check that *cannot be made*. Fixed by stamping `mypy_version` into the baseline payload (`tools/lint_mypy_baseline.py:mypy_version`) and having `ci_check_invariants` carry the baseline forward + print a loud `mypy check SKIPPED (analyzer version differs)` banner on mismatch, the same shape as the P159 unavailable path. **[Corrected P188 2026-08-05]** This paragraph used to end "the committed baseline is currently `<unstamped>`, so every machine gets that banner." That has not been true since the re-baseline: `tools/scanner_baselines/mypy_baseline.json` carries `"mypy_version": "2.3.0"` alongside its 1076 findings, and CI now installs exactly that release by reading it out of the baseline file (P187). A machine on a different mypy still gets the version-mismatch banner, which is the honest state rather than a silent pass — and under `--require-all-gates` that banner is now a failure, not a note.
- **`mypy>=1.5.0` (`requirements-train.txt:59`) is too loose for a count-locked baseline.** Pin it to whatever version the baseline is stamped with, or the gate oscillates between real and skipped as environments drift.
- **The mypy check has never gated anything in CI either.** `.github/workflows/codebase-invariants.yml:40` states *"Scanners depend only on stdlib + git. No requirements install needed"* — so mypy is absent in the job that runs the scanners, and P159's SKIPPED path fires there on every run. `test-suite.yml:41` installs mypy but unpinned, and does not run the gate. Enabling it for real means adding a **pinned** mypy to the invariants workflow; not done here because it cannot be validated from this machine.
- **Second finding, same family: `pytest-asyncio` is declared (`requirements-train.txt:55`) but was not installed**, so every `@pytest.mark.asyncio` test errored with *"async def functions are not natively supported"*. That accounted for **~30 of the 98 local failures** — `test_onchain_solana_agent.py` went 16 failed → 1, `test_sentiment_llm_agent.py` 19 → 4, and `test_http_retry_and_manifest.py` / `test_concurrent_stress.py` went fully green. Full suite: **98 → 51 failures, 2532 passed**. These were never code defects.
- **Mitigation pattern (generalising P158/P159):** before treating scanner or test output as evidence about the *code*, confirm the **tool that produced it was present and is the same tool the baseline was recorded with**. A missing tool, an unsupported regex engine, and a different analyzer version all produce output that is indistinguishable from a real result. Every declared dev dependency should be installed before any "N failures" number is quoted.

### P160. [FIXED 2026-08-04] `dashboard_state.json` could stop updating forever and say nothing — the export's only failure report was a `logger.debug`
- `HMATSProductionRunner._export_dashboard_state` (`main.py:~16457`) is the **sole writer** of `data/dashboard_state.json` — the file the API serves and an operator reads to answer "is the engine alive?". Its blanket `except Exception` logged at **DEBUG**, which every production log level drops. So one bad attribute access froze the dashboard at its last good values with **nothing said anywhere**, and a stale-but-plausible dashboard is indistinguishable from a live one.
- Same shape as **P155 Layer 5** (`_last_quant_directions` high-water mark), **P156** (frozen FastRiskTick anchor) and **P158** (a scanner matching nothing): **state that reads as live but silently stopped updating.** This is now the most common bug class in this codebase — when a value looks stale, ask what *writes* it and whether that writer can fail quietly.
- **How it was found:** two tests in `tests/test_dashboard_state_incremental_export.py` (and seven in `tests/test_step15_status_export.py`) were failing on a *missing file*. The real cause — their `SimpleNamespace` fixture had no `mode`, which `_export_dashboard_state` reads (`ProductionConfig.mode`, `main.py:1361`) — was swallowed by that same DEBUG handler. The tests could not report the reason they failed. Both fixtures fixed.
- **Fix:** keep the swallow (a diagnostics writer must never kill the tick) but escalate to `logger.error` naming the exception type **and the consequence** ("dashboard_state.json is STALE"), rate-limited to the 1st / 10th / every 100th consecutive failure so a sustained outage is loud without burying a 4H log, `exc_info` on the first only. A `logger.warning` announces recovery and resets `_dashboard_export_fail_count`. The happy path stays silent.
- **Tests:** `tests/test_dashboard_export_failure_visible.py` (5) — error-not-debug with cause + consequence in the message, never propagates, exactly 3 logs over 120 failures, recovery announced + counter reset, healthy path silent.
- **Mitigation pattern:** `except: logger.debug(...)` in a **writer** is a silent failure, not an observability choice — the reader has no way to tell "not updated" from "updated to the same value". Diagnostics code may swallow, but it must swallow **loudly**, and the message must name what is now stale.

### P159. [FIXED 2026-08-04] "mypy is not installed" was recorded as "mypy found 0 errors", and `--update` then zeroed the baseline
- `tools/lint_mypy_baseline.py:run_mypy` shells out to `sys.executable -m mypy`. When mypy is absent that command **exits 1 and prints to stderr** — it does **not** raise `FileNotFoundError`, so the `except FileNotFoundError` guard was unreachable (**P152 shape**: guard defined but never called). `parse_errors()` then found no `error: … [code]` lines and reported `total_count: 0`.
- Two consequences, second one worse: (1) in check mode the count only ever went *down* vs baseline and the gate only flags *increases*, so the mypy check **passed silently without ever running**; (2) `ci_check_invariants --update` on such a machine **rewrote `mypy_baseline.json` from 1080 → 0**, which would then fail the gate with **+1080** on every machine that does have mypy. This machine (`/Users/yifangao/miniconda3/bin/python`) has no mypy, so any local rebaseline was one commit away from breaking CI for everyone.
- Fixed: `run_mypy` raises `MypyUnavailable`; the scanner emits `{"unavailable": …}` as **data** rather than a zero count; `ci_check_invariants` carries the previous baseline forward and prints a loud `mypy check SKIPPED` banner. **A missing tool is a broken check, never a passing one.** Tests: `tests/test_mypy_baseline_unavailable.py`.
- **mypy is a declared dev dependency** (`requirements-train.txt:59`) — `pip install mypy` restores the check.

### P158. [FIXED 2026-08-04] The authority audit silently matched nothing — 20 phantom "no direct writer" issues, 22 phantom dead flags, and one check that had never run in the project's history
- Every pattern in `scripts/authority_consistency_audit.py` is authored in **Python `re` syntax** (`\s`, `\b`, `\d`) but executed by **`git grep`**, whose engine depends on how git was *built*. An unsupported escape is **not a syntax error**: the pattern compiles, matches nothing, and git grep exits **1 = "no matches"** — indistinguishable from the wiring genuinely being absent. The 2026-04-25 hardening only caught exit > 1.
- **Engine matrix — this is the load-bearing detail:**
  | escape | glibc regcomp (Linux CI) | BSD regcomp (macOS, Apple git 2.39) | PCRE (`git grep -P`) |
  |---|---|---|---|
  | `\s` `\b` `\w` | ✅ GNU extensions | ❌ | ✅ |
  | `\d` | ❌ **not implemented** | ❌ | ✅ |
- So on macOS the whole audit went dark (Section A reported "no direct writer" for **all 20** agents; Section B reported **22 dead `ENABLE_*` flags**), and on Linux CI `\d` never worked either — meaning **`DRL_PUNCH_THROUGH_CONF` was never once evaluated**, which is why the baseline had no entry for it despite the drift predating the baseline commit. `BEST_FOLDS_ETH` (the only other `\d` pattern) happens to be clean.
- Fixed: `_detect_grep_mode()` probes the engine **once** against a known canary line using `\s` and `\b`, prefers `-P`, and **raises** if no available engine honours the escapes rather than emitting a wall of false findings. Tests: `tests/test_authority_audit_regex_engine.py` pins the engine contract, not any particular finding.
- **When adding a tracked constant/pattern here, verify it actually matches something.** A pattern that matches nothing is an unevaluated check that reads in the baseline exactly like a clean one — `test_every_git_grep_pattern_in_tracked_constants_matches_something` now enforces this.
- Accepted into the baseline (**not** a regression, and deliberately **not** "fixed" by editing live thresholds): `DRL_PUNCH_THROUGH_CONF` observes 5 values at 5 *semantically distinct* decision points — `0.3` entry punch-through (`integration_v36.py:784`, `main.py:5304/5344`), `0.35` DRL-vs-position conflict (`execution_service.py:2682/2734`, `integration_v36.py:2250/2255`), `0.4` exit alignment (`tick_exit_triggers.py:144`), `0.55` (`integration_v36.py:2253`) and `0.6` strong-conviction exit (`tick_exit_triggers.py:496`).

### P157. [FIXED 2026-08-04] Phase 1 audit buckets read as 7 ad-hoc labels instead of a taxonomy
- `configs/strategy_v5_1_decisions.json` classified 12 strategies into buckets meant to be *class letter + descriptive suffix*, but two entries (`derivatives_cascade`, `vol_options_derived`) were written with no letter — so the set looked like 7 unrelated labels and the "5-bucket categorization" the phase was described as never existed in any form. Renamed to `E_derivatives_cascade` / `F_vol_options_derived`; the real taxonomy is **six** classes A–F, now documented in `_meta.bucket_taxonomy`.
- **No decision changed** (`KEEP_IMPROVE` 4 / `KEEP_AS_IS` 4 / `ARCHIVE` 3 / `DEFER_v6` 1), and this is safe because `bucket` has **no production consumer** — the engine reads only `archived` and `decision` (`agents/kraken_quant_agent.py:2303`, `analytics/sixty_day_review/review_aggregator.py:445`).

### P156. [FIXED 2026-08-04] FastRiskTick acted on an unboundedly stale 4H anchor — a REDUCE_50 that could fire forever
- **Every** trigger in `FastRiskTick._evaluate` (price move, vol spike, depth drop) compares a live reading against a reference captured by `set_4h_anchor()`. That call is the **last statement** of the 4H decision path (`main.py:10166`), so every early return before it skips it — notably the **P0 ABORT at `main.py:7998`**. Nothing bounded how old the reference could get, while the evaluator kept running every 30s against it.
- **Failure shape:** a depth baseline anchored during a healthy orderbook makes ordinary depth look like an 80%+ collapse *indefinitely* → `REDUCE_50` every cooldown → exposure ratcheted toward zero and never allowed back up (this evaluator can only reduce). This is the same class as P155 Layer 5's `_last_quant_directions` high-water mark: **state that reads as live but silently stopped updating.** It is a concrete candidate explanation for the observed `[FastRiskTick][LIVE] BTC: REDUCE_50 - depth_drop=82%(3x)` and, through it, for a collapsed `target_exposure`.
- **Fix:** `set_4h_anchor()` records `_anchor_set_at[asset]`; `_evaluate` refuses to act when the anchor is older than `ANCHOR_MAX_AGE_SEC` (6h = 1.5 × the 4H tick, so one late tick is tolerated and two consecutive misses are not), resets the confirm streak, and logs a rate-limited WARNING naming `set_4h_anchor` as the thing to check. **Fail-SAFE:** the suppressed action is always a reduce, so refusing to act is the conservative side.
- **Mitigation pattern:** any guard that compares "now" to a stored reference needs a **staleness bound on the reference**, not just correctness of the comparison. Prefer refreshing such state on a path that cannot be short-circuited; where that is impractical, bound its age and fail toward inaction.
- **Tests:** `tests/test_fast_risk_anchor_staleness.py` (10) — the real 82% collapse still fires on a fresh anchor (the guard must not disarm what it protects), stale suppresses both REDUCE and the cooldown-bypassing EXIT_ONLY, one missed tick tolerated, streak reset while stale, log rate-limited, per-asset independence.

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
- **Follow-up (P155e) — the "0/3bps Coinbase fees" migration item is NOT implemented, and it is a live suspect.** `FRICTION.update_fee_bps` (`main.py:4419`) is **global with no venue dimension** and is fed from Kraken's fee-tier API. Post-Phase-B every routed asset is therefore priced at Kraken's ~26/16bps while it actually executes on Coinbase at ~3/0bps. Friction is subtracted from expected alpha *before* sizing, so this systematically shrinks `target_exposure` — a leading candidate for a `ZERO_EXPOSURE` blocker. `_coinbase_fee_model_warning()` reports the mismatch and its magnitude once per process. **[IMPLEMENTED 2026-08-04, DEFAULT OFF]** `coinbase_venue_aware_fees` (`configs/*.json`) prices alpha-gate friction for the venue the asset will actually execute on. The per-tick friction sync (`main.py:~8505`) already runs **per asset**, so this needed no change to the alpha calculator. It stays OFF by default because enabling it **loosens a risk gate**: confirm the blocker with `scripts/why_no_trade.py` first, then flip it. Wrong-way failures fall back to the Kraken tier (the conservative side) and log `[VENUE-FEE]`.
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
- **Two open items, deliberately not fixed here:** (a) the plan tier is baked into the URL (`BASE_URL = ".../api/growth/v2"`), so downgrading to cut the bill 404s the feed and it silently returns `[]` → falls through to CC News/mock — the exact failure c064b1f fixed for `developer`; (b) `.env.example:47,156` still describes CryptoPanic as free tier, which is stale and actively misleading for cost decisions. (The training guide's copy of this error was retired with the guide itself — archived 2026-08-07 to `docs/archive/HMATS_E2E_TRAINING_GUIDE_2026-02.md`; the live guide is `docs/HMATS_TRAINING_GUIDE_V2.md`.)
- **Mitigation pattern:** an API rate limiter (throttle, backoff, quota counter) that lives only in RAM is not a control — it re-arms on every restart, and the restart-heavy failure modes are exactly when you most need it to hold. Any loop that fires work immediately on startup must read its throttle from disk, not from a fresh object.

### P215. [2026-08-07] Cross-validating P214 against the P200 retrain surfaced the regime-vocabulary parity gap — THREE GMM vocabularies now coexist, and Rung 3 must ship the split-aware GMM with the checkpoints. Plus: the flow-v2 feature build replicates the paid external group on free full-history data
Operator asked to cross-validate the parallel session's regime-classifier-adjacent work (P214) against the P200 retrain workstream. Verified, nuanced, and one new blocker found:

**P214 claims verified/nuanced against P200 artifacts:**
- *"Per-asset GMMs take 12 features, none denoised — regime classification unaffected"* — **VERIFIED** (GMM_FEATURE_COLS inspected: 12 features, zero `*_denoised`).
- *"Would have silently skewed the retrain now in flight"* — **nuance**: the retrain itself is self-consistent (trains on parquets whose denoised columns are causal); what the image gap would have skewed is the **Rung-3 live shadow evaluation**, where the checkpoints would have been fed raw features against a causal-denoised training set. P214's ship (wavelet module + pywt in the image) plus P200's causal parquets **close the wavelet half of train/serve parity** — `tests/test_wavelet_causality.py` pins that the causal transform equals the live deque recurrence.
- *"Live DRL IC (+0.019/−0.081) was measured on a model fed features that disagree with its training set"* — **fair caveat, recorded**: the skew touches 5/122 inputs and both live windows were measured under the SAME skew (so the cross-window sign-instability stands), and the no-edge verdict rests decisively on the clean retrain leg, which is skew-free by construction. But post-P214, future shadow IC is clean where past IC was not — do not compare across that boundary without saying so.

**The new blocker — three regime vocabularies coexist (found BY this cross-validation):**
| artifact | k | fit | names |
|---|---|---|---|
| `models/regime_classifier/{ASSET}` (what the RUNTIME serves) | 8 | Feb full-sample = P164-leaked, on the overwritten 18k series | incl. STEADY_UPTREND, NEUTRAL_DRIFT |
| `training/training_data/gmm_models/{ASSET}` (what the P200 parquets + p200_di4b retrain use) | 7 | 2026-08-07 split-aware | incl. **REGIME_1 (unmapped)** |
| `models/regime_classifier/gmm_config.json` (global legacy fallback) | 6 | 2026-02-14 | 6 names |
Consequences: (a) any Rung-3 shadow deployment of retrained checkpoints without ALSO deploying the split-aware per-asset GMMs reproduces the exact P214 skew shape on the 8 `regime_proba` features + the regime label — **the split-aware GMMs must ship WITH the checkpoints, atomically** (P4's mixed-fold lesson at the artifact-set level); (b) cluster `REGIME_1` covers **2,582 bars = 20% of BTC** and appears in none of POSITION_BIAS/BULL_REGIMES/BEAR_REGIMES/regime_weights — it silently takes neutral values in every regime-conditional term, and the P184 guard does NOT fire because REGIME_1 *is* a resolvable name, just an unmapped one. The rebuild's `name_clusters` needs a naming pass (or the bias tables need the new vocabulary) before any promotion-grade run; (c) the rebuild did NOT deploy the new GMMs to `models/regime_classifier` because the pre-training checklist failed on `has_external_data` — deploy-gating worked as designed, but it means the runtime/training divergence persists silently until a deliberate joint deploy.

**Feature build (operator: "more important is the features"):** three converging evidence lines — the probe's strongest group was external flow/positioning (180d Coinglass depth only), live whale+funding were the only both-window-positive agents, and the literature (order flow >> 40 price characteristics without it). The Binance kline archive carries `taker_buy_base/quote`, `count`, `quote_volume` for the FULL history — and the fetch was discarding them. Shipped: `fetch_binance_full.py` keeps the flow columns (`keep='last'` dedup so old flow-less rows lose); `training/scripts/build_flow_features.py` builds **13 causal `fv2_*` features** (taker-imbalance z + momentum, trade-count z, avg-trade-size z, Amihud z, quote-vol z, taker-quote share, hour/dow seasonality, cross-asset rel-strength + LAGGED reference return) as EXTRA parquet columns — NOT manifest members, obs_dim stays 126; admission to the DRL manifest is a separate deliberate obs-contract change. `tests/test_flow_features_causal.py` (5) includes the P164 construction test (perturb the future violently, assert zero movement in the past) and a lead-lag contemporaneity check. **Probe result (strict window, 2.7y OOS): flow_v2 standalone CLEARS on BTC 16h (IC 0.033, t=2.5, +10.9bps) and SOL 4h+16h (+20.2bps 16h, both models); ETH does not clear (honest). ALL-group with fv2: BTC +24.4→+28.1bps (t 6.0→6.3), SOL bps up, ETH neutral.** The free full-history family replicates the paid 180d external group on 2 of 3 assets with 15× the history depth.
- **[ADDENDUM 2026-08-07, answering the venue-risk session's degeneracy question] The old GMM's saturation is DISTRIBUTION SHIFT, the concentration is the MARKET — and the split-aware refit already fixes the half that is broken.** The parallel session observed the live GMMs emitting 6 of 7-8 clusters in four months, two clusters 92% of the time, posteriors saturated at ~0.99997, and asked whether that is degeneracy needing a retrain. Measured against the fresh split-aware fits (last 720 bars = ~4 months): **concentration persists** (top-2 share 80%/64%/95% BTC/ETH/SOL) — so it is the market (2026 has genuinely been chop; P198 measured the same 93% from the live diags) — **but saturation does not**: mean max-posterior drops to 0.94/0.89/0.97 with only 23%/12%/42% of bars >0.999, vs ~0.99997-every-tick from the Feb model. The old runtime GMM was fit on the overwritten 2017-2026 series with its own scaler; 2026 features sit far from ALL of its centroids in that stale space, so nearest-cluster wins with certainty — one-hot output, zero marginal information in `regime_proba` and a CONSTANT `regime_confidence` feature. The refit produces calibrated soft posteriors. Conclusion for the ladder: the "retrain" the degeneracy calls for **is the split-aware refit that already exists**; what remains is (a) the Rung-3 atomic deploy (GMMs + checkpoints together, above), and (b) the naming pass — BTC has unmapped `REGIME_1` AND SOL has unmapped `REGIME_7`; both take silent neutral values in every name-keyed table. Do not patch the old model's lookup table — its posteriors are the artifact, not its names.
- **Mitigation patterns:** (d) after any refit that can change k or cluster identity, treat {model artifacts, parquets, checkpoints} as ONE versioned set — a regime feature is only meaningful relative to the GMM that produced it; (e) when two sessions work adjacent surfaces, cross-validate CLAIMS against ARTIFACTS, not summaries — P214's "regime unaffected" was true for its finding (wavelet) and false as a general statement about regime parity, which its author had no reason to check.

### P221. [FIXED 2026-08-07] GMM feature audit (operator: "are we sure the GMM features are correctly built?") — causally clean, but `vol_percentile` was a train/serve skew and two inputs are constants held by a load-bearing ordering accident
(Committed under the title P216 before the parallel session's P216-P220 landed — code markers renumbered to P221.) All 12 GMM inputs audited on both sides (`rebuild_pipeline.compute_gmm_features_for_bar` vs `market_data_pipeline._predict_gmm_regime`):
- **Causality: CLEAN.** Every window trails; no lookahead anywhere. Pinned by a P164-style construction test (perturb bars > i, features at i must be bit-identical).
- **`vol_percentile` was a REAL train/serve skew (fixed).** Training ranked the bar's volume against the expanding 6-year history; runtime ranks within its fetched ~1024-bar frame. With volume's secular drift those distributions differ materially — one of the GMM's 10 effective inputs, the same family as P214's wavelet skew, and a contributor to the old GMM's shift-driven saturation (P215 addendum). Training now uses trailing `GMM_VOL_PCT_WINDOW = 1024`. **Requires one more parquet rebuild + GMM refit** — scheduled after the in-flight p200_di4b measurement completes (its TQC-vs-ridge verdict is on a fixed feature set and is not invalidated).
- **`cross_asset_correlation` matches by ACCIDENT.** Training hardcodes 0.87; runtime reads `raw.get("cross_asset_correlation", 0.87)` — and the live value is written into `raw` at :1437, AFTER `_predict_gmm_regime` reads it at :950, so the GMM always sees the default. A load-bearing ordering accident (P173 `is_4h_bar_close` shape): pinned — `tests/test_gmm_feature_parity.py` fails if the write ever moves before the predict.
- **`fear_index` is the 100−RSI(14) proxy on BOTH sides** — not the real F&G index its name suggests. Consistent, therefore safe; pinned so one side cannot drift. `spread_percentile` = per-asset constants, equal both sides, pinned.
- **Net: the GMM effectively runs on 10 features, not 12** (two zero-variance constants). Tests: `tests/test_gmm_feature_parity.py` (7).
- **Mitigation pattern:** "matches runtime" in a comment is a claim, not a property — parity between a training constant and a runtime `.get()` default can rest on execution ORDER, and only a pin that reads both sides keeps it true.

### P224. [FIXED 2026-08-08] A confidence I saturated myself, and the one agent that would not say why it was silent
- **1. `flow` confidence pinned at 1.00 with direction 0.00 — a defect I introduced in P223.** The attribution proxy is `min(1.0, |whale_flow| / 1e7)`, calibrated for CryptoCompare's `large_tx_count × avg_tx_value` (millions). P223 swapped that source for Blockchair's **24h settlement value (~$5.7e10)**, so the proxy saturated and attribution began reporting flow as a **maximally-confident non-signal** — strictly worse than the `0.00/0.00` it replaced, because the IC/attribution layer will weight it. A confidence attached to a **zero direction has no meaning**: nothing downstream can act on "certainly no opinion". The whale feed is a MAGNITUDE with no sign (P223), so until a signed source exists this is honestly zero. **Same shape as the P219 confidence bug — twice in one session: changing what a number MEASURES without re-checking the normalisers calibrated to the old scale. A magnitude swap is never local.**
- **2. `micro` — the one agent the P216 sweep left "not traced" — now says why.** It records a cause in `diagnostics.reason` (`no_exchange_data`, `stale_snapshot`, insufficient samples, or `error` on the exception path) and **nothing surfaced it**, so all four presented identically as `micro=+0.00/0.00`. Notably the Binance LOB feed **is** live and directional (BTC `taker_buy=159.8` vs `taker_sell=69.8`, a 2.3:1 skew), so "no data" is not the obvious answer and guessing would have repeated the Helius mistake. Now logged **once per reason CHANGE, per asset** — a per-tick line for a steady-state condition becomes wallpaper (P202), and a single global latch would let the first asset consume the one report for all of them (also P202). Reads `error` as well as `reason`, or the crash case — the worst of the four — would stay silent.
- Tests: `tests/test_flow_confidence_and_micro_reason_p224.py` (12).
- **Numbering:** shipped as P221/P222 and renumbered to P223/P224 — a parallel session claimed both minutes earlier (its `[P222]` tags in `main.py:8830/:10144` and `defense/constitution.py` are ITS work and were left alone). Same convention as P200 and the GMM audit's own renumber.
- **Mitigation pattern:** a confidence, weight or normaliser is calibrated against the SCALE of its input, so replacing a data source silently invalidates every one of them downstream. Grep the consumers of any quantity whose source you change — and prefer reporting zero confidence over a saturated one, because a confident non-signal is harder to notice than a missing one.

### P223. [FIXED 2026-08-07] The CryptoCompare plan cannot be upgraded — replaced its on-chain feed with Blockchair (free, keyless, unmetered)
- **Constraint:** CryptoCompare is capped at **100 calls/MONTH** (P220) and the plan **cannot be upgraded**. `/blockchain/latest` was the only source of BTC/ETH on-chain metrics, and its absence is why `flow` and `onchain` read 0.00 every tick.
- **Alternative, probed before it was written** (twice burned this session by guessing — the options "tier limit" and Helius): Blockchair `/{chain}/stats` needs **no API key**, covers **both chains**, is **unmetered**, and returns real numbers — BTC **$53.6B** 24h on-chain volume with a largest single transfer of **$1.31B**; ETH **$4.7B** / **$306M**.
- **It does NOT fabricate `large_transaction_count`.** Blockchair does not publish one, and inventing a number to satisfy the composite's `> 0` gate is exactly the defect class this codebase keeps finding. It publishes the honest quantity instead — **24h settlement value in USD** — which is arguably a *better* whale magnitude than CryptoCompare's `large_tx_count × avg_tx_value`, itself an estimate whose own comment admitted *"we don't have price here, so use count as the signal"*.
- **Still a MAGNITUDE, not a direction — unchanged.** The whale proxy's *sign* has always come from CoinGlass OI + funding (`main.py:7263`), never from the on-chain feed. Swapping the size source loses nothing directional, and a test pins that no direction is claimed.
- **Wired as a FALLBACK, not a replacement:** runs only when CryptoCompare produced nothing, so restoring that quota silently takes precedence again. Fail-soft; a free fallback must never break a tick. 1h TTL (24h rolling stats — faster just re-reads an unmoved number), disable via `HMATS_DISABLE_BLOCKCHAIR=1`.
- **Units handled at the boundary:** Blockchair returns satoshi/wei **and** `market_price_usd` in the same payload, so USD conversion happens in the feed rather than leaving a caller to guess the denomination (P169: a value without its unit is not a measurement). ETH's volume arrives as a **string** because wei overflows JSON's safe integer range.
- **[BUG I SHIPPED FIRST, caught by printing the number]** `largest_transaction_24h` is reported **already in USD** under `value_usd`. My first version read `value` — the natural guess — and silently produced **0.0 on both chains**, i.e. a whale metric that is always zero. Same reader/writer key mismatch as P2 and P197's `entry_vwap`. Pinned by a test that reads the source and asserts the wrong key is absent.
- **Remaining CryptoCompare exposure:** news only, and that is redundant — CryptoPanic already serves `llm_sentiment` live (`src=cryptopanic_metrics`).
- Tests: `tests/test_blockchair_onchain_p221.py` (18). Baseline: `silent.tryexcept_count` 336 → 337 by hand for the numeric-coercion helper (annotated `noqa`, but that counter comes from `silent_failure_audit`, which is not noqa-aware).
- **Mitigation pattern:** when a vendor becomes unavailable, the replacement should publish what it *actually has* and let the gap be visible, not shim a same-named field so downstream checks keep passing. A fabricated field is worse than a missing one, because nothing downstream can tell.

### P220. [FIXED 2026-08-07] Two CryptoCompare feeds, one 100-calls/MONTH account, and no shared accounting — demand was 5.4x the cap
- **Measured from the account's own `/stats/rate/limit`** (note: that call itself costs quota):
  ```
  calls_made  hour 3 · day 43 · month 283 · total 39,223
  max_calls   hour 100 · day 100 · month 100
  ```
  The binding limit is the **MONTH — 100 calls, ~3/day**, and 283 were already spent. Not a rate-limit problem: a hard cap.
- **Demand before:** `cc_news` 1 call/fetch @ 5min TTL ⇒ ~180/mo, `cc_onchain` 2 calls/fetch (BTC+ETH) @ 15min TTL ⇒ ~360/mo. **~540/month against 100.** (Pre-P219 news was 3 calls/fetch, so ~900.)
- **The structural fault was not the TTLs — it was that neither feed knew the other existed.** Both key off the same `CRYPTOCOMPARE_API_KEY`; each had its own independent backoff; so one could exhaust the month while the other kept calling, and both would then sit in separate 15-minute backoffs while the real constraint (the month) was already blown. **A per-feed rate limiter cannot express a per-ACCOUNT budget.**
- **Fix:** `data_mgmt/feeds/_cc_quota.py` — a shared, **persisted** (P154), per-calendar-month budget that both feeds **reserve against BEFORE** calling. Reserve-then-call so a refusal is free; counting afterwards would let a burst spend the month before anything noticed. All-or-nothing per fetch (half an on-chain fetch spends quota for a partial picture). Spend is attributed per caller, so "which feed ate the month" is answerable. Exhaustion warns **once per period**, names the spend, and says *"this is a BUDGET decision, not a market condition"* — silent exhaustion is indistinguishable from "no news", the conflation this codebase keeps producing (P199/P216/P218). Callers degrade to **cached** data, never raise.
- **Budget defaults to 90, below the real 100** — the provider counts calls this process cannot see (ad-hoc probes, another process on the key), so a budget equal to the cap always discovers the difference as a 429. `HMATS_CC_MONTHLY_BUDGET` overrides.
- **TTLs re-derived to fit ON the budget, not above it:** news 5min → **12h** (60/mo), on-chain 15min → **48h** (30/mo) ⇒ **90 total = budget < 100 cap**. 48h rather than 24h deliberately: at 24h demand is 120 and the guard would bind every month around the ¾ mark, and **a control that fires routinely stops being read**. The guard should be a backstop for the unforeseen, not the thing that paces us.
- **Answer to "do we have sufficient calls": no, and now it degrades honestly instead of hammering.** At 100/month these feeds cannot serve a per-tick system at live cadence — 12h/48h refresh is the most they support. The primary consumers already have replacements: sentiment via **CryptoPanic** (live, `src=cryptopanic_metrics`) and flow via **P219's CoinGlass** shadow.
- Tests: `tests/test_cc_quota_p220.py` (18), incl. an arithmetic pin that fails if a future TTL edit pushes demand back over.
- **Mitigation pattern:** when several callers share one credential, the quota is a property of the **account**, not of any caller — put the budget where the credential is. And prefer to fit demand under a limit by design, so the guard stays exceptional; a limiter that trips every cycle is indistinguishable from normal operation.

### P219. [SHADOW 2026-08-07] A flow signal from data we already pay for — CoinGlass liquidations, shadowed not promoted
- **Why:** the `flow` agent emits 0.00 every tick because its whale proxy needs CryptoCompare's `large_transaction_count`, and that account is hard-capped at **100 API calls/month** (measured via `/stats/rate/limit`: **283 used**, 39,223 lifetime, *"please upgrade your account"*). The upgrade is blocked. **No new vendor is needed** — CoinGlass is already paid for and already fetched every tick: `liquidation_imbalance` was live at **BTC −0.53 / ETH −0.50 / SOL +0.06** on **$83M / $64M / $7.4M** of 24h liquidations.
- **It is arguably the better input anyway.** CryptoCompare's metric counts large **on-chain transfers**; this book trades **perpetual futures**. Liquidation imbalance is derivatives positioning being forcibly unwound — a more direct read on what moves perp prices.
- **SIGN VERIFIED, NOT ASSUMED.** `coinglass_feed.py:650` computes `(long − short)/total`, "positive = more longs liquidated". A **1h** probe appeared to contradict the live value; re-probing at the feed's own **24h** range resolved it (long $8.5M vs short $31.7M → −0.576, matching `market_data`). Pinned by a test — getting this backwards inverts the whole signal.
- **TWO OPPOSITE STRATEGIES ON PURPOSE.** Forced liquidation is directionally ambiguous, so both readings are emitted and the **P166 cost-aware IC gate decides**: `liquidation_squeeze` (momentum — shorts liquidated ⇒ forced buying ⇒ continues) and `liquidation_exhaustion` (reversion — the cascade IS capitulation ⇒ fades). Exact negations, so at most one can be right and *"both are noise"* is a possible, informative outcome. Replaces a guess with a measurement at zero extra cost.
- **Observation-only (Iron Law 7).** Ledger `data/strategy_shadow/derivflow_*.jsonl`, registered with `compute_shadow_ic.py` so the existing validated harness scores it — **not** a new bespoke review tool. Gated on a minimum liquidation size and a minimum imbalance so background noise does not pollute the IC. **Adds ZERO API calls** — it reads `market_data` the CoinGlass feed already populates (`main.py:6024-6030`).
- **[MISTAKE, caught by my own test]** I first inserted the init block **between the funding harness's `try` and its `except`**, which silently replaced that handler with `pass` and relabelled derivflow failures as *"FundingShadowHarness init failed"*. Two tests now pin that the funding handler survived intact and that the derivflow handler names its own harness.
- **CC News: 3 calls → 1.** The request asked per-asset (`categories=BTC`, then ETH, then SOL) for the same headline pool. Now one combined call fans out to every tracked asset's cache, with uncategorised rows kept for all of them (dropping them would shrink the corpus and push `headline_count` back toward 0 — the condition being lifted). ~18 calls/day → ~6.
- **Do NOT promote on these numbers.** Promotion requires the P166 gate on **forward** data: positive IC at every horizon, |t| ≥ 2, and edge ≥ 2× round-trip cost. Run `python -X utf8 analytics/shadow_ic/compute_shadow_ic.py --window-days 30` (operator-local, P213) once weeks of ledger exist.
- Tests: `tests/test_derivflow_shadow_p219.py` (25).
- **Mitigation pattern:** when a vendor blocks you, inventory what you already pay for before buying — and when the direction of a signal is genuinely arguable, ship **both** readings to the measurement harness instead of picking one and calling it a design decision.

### P218. [FIXED 2026-08-07] Options endpoints are 404 (NOT a tier limit), and funding is priced on the wrong venue — the third cross-venue leftover
- **OPTIONS — I guessed "CoinGlass tier limit" and was WRONG.** Probed live: the three paths the agent calls all return **HTTP 404** (`/option/info/max-pain`, `/option/info/oi`, `/option/info/volume`), while the parent `/option/info` returns **real data**, and the same key serves funding/OI/liquidations fine. So the v3 paths no longer exist; the subscription is not the problem. The handler returned `None` on any non-200 and the caller fell back to `put_call_ratio = 1.0`, so **"this URL is gone" and "the options market is perfectly balanced" produced byte-identical output** — the same shape as P199's `INSUFFICIENT_SAMPLES` and P216's `pcr=1.0`.
  - **Fixing the URLs is deliberately NOT attempted:** `/option/info` returns per-exchange OI/volume aggregates with **no put/call split**, so PCR cannot be reconstructed from it. Inventing a replacement options signal is a strategy change, not a bugfix.
  - **What shipped:** a one-shot-per-path warning naming the consequence ("will fall back to NEUTRAL forever") and explicitly recording **"NOT a tier limit"**, so the wrong diagnosis is not re-derived. Still fails soft.
- **FUNDING VENUE — third instance of P172/P210's shape.** The book trades **Coinbase** perps; `market_data["funding_rate"]` is overwritten at `main.py:6086` with **Kraken futures**. Measured live (8h): BTC **−0.000077** vs **+0.000040**, ETH **−0.000015** vs **−0.000008**, SOL **−0.000378** vs **+0.000168**. Both venues sit far below every downstream threshold (the short-bias whale proxy needs `|funding| > 0.0002`), so this is **behaviourally inert today** — but the **SIGN differs on BTC and SOL**, so if funding ever becomes material we would trade the wrong one.
  - `coinbase_venue_aware_funding`, **default OFF**, declared **and** parsed (P201), routed-assets only, wrong-way failure keeps the Kraken rate (P172's conservative direction). The value is cached by the Phase-3 shadow block that already fetches it rather than adding three product calls to the hot path — CDE funding updates hourly against a 4H loop, so the one-tick lag is immaterial.
  - Note `data_mgmt/feeds/coinbase_funding_feed.py` has **zero consumers** — a whole feed built for this and never wired.
- **SHORT_BIAS — the risk I flagged is smaller than I said.** Its whale-flow proxy is gated on `|funding| > 0.0002`; live funding is 1–2 orders of magnitude below that on BTC/ETH. So even with the regime gate opened, its primary input is **structurally zero** and it would mostly emit nothing. Enabling it is therefore closer to a no-op than to a live risk change — but that also means opening the gate would not buy a signal.
- Tests: `tests/test_venue_funding_and_options_p218.py` (19), incl. pinning the 2bps threshold that the "inert" argument depends on.
- **Mitigation pattern:** probe the endpoint before blaming the plan. A 404 on a hardcoded URL is a broken integration, not a market condition or a billing tier — and any handler that converts a transport failure into a *neutral domain value* makes the two indistinguishable. Fail soft, but never fail **quiet**.

### P217. [FIXED 2026-08-07] STEADY_UPTREND spent 4 months routed to the MEAN-REVERSION bucket — the 12 kraken_quant strategies are bucketed by REGIME, not asset
- **The question that found it:** "why are only 2 of 12 strategies reachable when we trade BTC, ETH and SOL?" The premise is the key: the 12 are bucketed by **regime**, not by asset — all three assets share one set, so trading three assets widens nothing. Reachability is decided entirely by `_map_regime`.
- **Measured over 2,545 per-tick diagnostics (2026-04-08 -> 08-07):**
  | regime | n | % | mapped to |
  |---|---|---|---|
  | QUIET_ACCUMULATION | 1489 | 58.5% | SIDEWAYS (correct) |
  | WEAK_CONSOLIDATION | 859 | 33.8% | SIDEWAYS (correct) |
  | **STEADY_UPTREND** | **93** | **3.7%** | **SIDEWAYS — WRONG, a trending-up regime handed to the mean-reversion bucket** |
  | EXTREME_VOLATILITY | 41 | 1.6% | BEAR (correct) |
  | NEUTRAL_DRIFT | 27 | 1.1% | SIDEWAYS (right, but by accident via the default) |
  | VOLATILE_CHOP | 17 | 0.7% | SIDEWAYS (correct) |
- **`STEADY_UPTREND` and `NEUTRAL_DRIFT` were simply absent from `_REGIME_MAP`**, and `_map_regime` returned its SIDEWAYS default **in silence**. **`MOMENTUM_RALLY` — the only BULL name that WAS mapped — has never occurred once in four months**, so the four BULL strategies (none of them archived) had 93 ticks of their own regime and never saw one. `PANIC_SELLOFF` likewise never fired.
- **Root shape:** a GMM regime name is **data** (cluster names come out of the model), `_REGIME_MAP` is a **hardcoded mirror** of it, and the two drift the instant a model is retrained with different cluster names. Same family as P215 (the diagnostic's hardcoded strategy-name list) and P2 generally.
- **The durable fix is not the two new entries — it is that an unmapped name is now LOUD.** `_map_regime` warns once per unseen name, naming the consequence ("the BEAR and BULL buckets cannot fire in this regime") and the fix. `REGIME_0..7` stays quiet: that is the expected unnamed-cluster shape, not drift.
- **Live effect, stated plainly:** ~3.7% of ticks now route to the BULL bucket, making **4 previously-unreachable strategies reachable** at kraken_quant's DECIDE authority. They remain subject to the alpha gate, veto chain and every P208/P210 cap. Revert = delete one map entry.
- **Worth noting separately:** the GMMs carry 7-8 clusters but emitted only **6 distinct regimes in four months**, two of them 92% of the time, with posteriors saturated at ~0.99997. That is a degenerate-looking classifier and is a retrain question, not a mapping one.
- Tests: `tests/test_kq_regime_map_p217.py` (17). The load-bearing one asserts each GMM-emitted name is an **explicit entry**, not merely "resolves to something" — which is always true because of the default, and is exactly how this hid.
- **Mitigation pattern:** when a hardcoded map has a catch-all default, "unmapped" and "deliberately mapped to the default" are indistinguishable — and the silent case is always the one that rots. Make the default observable, and pin the map against what the producer actually emits rather than against a list someone typed.

### P216. [FIXED 2026-08-07] The idle-agent sweep: the layer is STARVED, not broken — 3 of 12 causes were code
- **Taxonomy from the per-tick `[DIAG]` component audit + agent diagnostics (evidence, not inference).** Of the 12 agents emitting zero direction (P215), only **three** were fixable in code. **Do not go looking for 12 latent P2 bugs — they are not there.**
  | class | agents | cause |
  |---|---|---|
  | Starved (inputs all zero, credentials VALID) | llm_sentiment, flow, onchain, options, micro | `headlines=0`; `whale=0/exchange=0/etf=0`; `quality='N/A'`; `pcr=1.0` (neutral default) |
  | Correct behaviour | funding, kraken_quant | funding genuinely ~0 (Kraken 8h **−0.000047**); kraken_quant has **2 of 12** strategies reachable (P215) |
  | Skipped by design | short_bias | regime gate, see below |
  | Conditional writers | two_stage, soldex, onchain_graph | gated on validity/prior flags (all reached via dict-update — a `agent_signals["k"] =` grep MISSES these and reports "0 writers", which is how I first mis-called `onchain_graph` dead) |
  | **Real code bug** | **vol_alpha** | extractor key mismatch |
- **1. CC News 429 storm (the one with consequences).** On a 429 the feed returned `[]` with **no backoff recorded**, and `Retry-After` was parsed straight into the log line and discarded (**computed-but-unenforced, P144 shape**). The 5-min cache is written only on SUCCESS, so a rate-limited feed retried at full rate — **3 assets × every tick, forever** — and the state was in RAM, so every restart resumed hammering (**P154's lesson, fixed for CryptoPanic and never applied here**). Consequence: `headline_count=0` → `_c3_live` False → `main.py:8529` **deliberately** zeroes `llm_sentiment_direction`. An ADVISE agent was dark **because of our own request pattern**, with a valid 64-char key. Fixed: honour `Retry-After` (**with a fallback — the live 429s carry `Retry-After=None`, so without it the fix would do nothing in the exact case that produced it**), persist the backoff, and **serve cached headlines while backed off** rather than `[]` (serving `[]` would keep the agent dark for the whole backoff). **Verified live: 3 × 429/tick → 1, then silence.**
- **2. vol_alpha extractor mismatch (measurement only).** `_extract_vol_alpha` reads `vol_alpha_implied_direction`/`vol_alpha_intensity`; `_attr_collected` passed `vol_alpha_direction`/`vol_alpha_bias`. **The key sets did not intersect**, so attribution read `0.0` whatever the agent emitted — a P3 bug *independent* of vol_alpha being directionally dead by design. Post-fix it still reads `0.00/dq0.0`, which is now **honestly** zero rather than structurally zero.
- **3. short_bias regime gate — made expressible, NOT changed.** `_SHORT_BIAS_SKIP_REGIMES` was a code literal covering **~93% of live ticks**, so the agent was off almost always and nothing recorded that as a decision. Now `short_bias_skip_regimes`, declared **and** parsed (P201), **default unchanged**; missing key → historical set, `[]` → never skip. Collapsing those two would silently enable a short-signal agent everywhere. The live profile is untouched and a test asserts it — enabling it is an operator call (P141).
- **Baseline:** `silent.tryexcept_count` 334 → 336 by hand for the two fail-soft guards (both log, both carry `noqa: silent-swallow`, but that counter comes from `silent_failure_audit`, which counts try/except generally and is **not** noqa-aware). Updated the single counter rather than `--update`, which re-seeds all seven including mypy (P171); verified only that file moved.
- Tests: `tests/test_idle_agents_p216.py` (17), falsification-checked.
- **Mitigation pattern:** when many components read zero, separate *starved* from *broken* before writing any code — the per-tick diagnostics already record `called` and the raw inputs, and they said "inputs are zero", not "output is zero". And a rate limiter that only *logs* `Retry-After` is not a rate limiter; the retry budget it was given must be spent, persisted, and shared across every caller.

### P215. [FIXED 2026-08-07] 12 of 18 agents emit ZERO direction — and the diagnostic built to explain kraken_quant could only ever print zeros
- **Measured, not inferred.** 21 consecutive ticks × 3 assets over 18h from `[AGENT-TRACE]` (persistent log, since `docker logs` is wiped on container recreation):

  | | agents |
  |---|---|
  | **Fires every tick** | quant, sentiment, drl |
  | **Partial** | model_alpha 19/63 (**BTC 0/21**), whale 19/63 (**SOL 0/21**), funding 5/63 (SOL only) |
  | **NEVER fires (0/63)** | short_bias, micro, **kraken_quant**, vol_alpha, two_stage, llm_sentiment, flow, options, onchain, onchain_sol, soldex, onchain_graph |

  This quantifies P143's "~10 agents emit zero direction". **`kraken_quant` holds DECIDE authority** (matrix #18) and contributed nothing for 18h.
- **The diagnostic was structurally incapable of reporting anything else.** `get_firing_stats()` keyed `regime_ticks` AND `by_regime` by the Regime enum's **VALUE** (`'chop'`); `scripts/kq_strategy_diagnostic.py` looks them up by **NAME** (`'SIDEWAYS'`). Every lookup missed → `by_regime.get('SIDEWAYS')` returned `[]` → **all 12 strategies printed `attempts=0 fires=0` regardless of what actually happened**, and the status column read *"never-active (regime not seen)"* for SIDEWAYS while the header of the same report showed `chop=3 (100%)`. The method's own docstring declared the contract as names — **the writer violated a contract written five lines above it** (P2 family; P174 for a metric that cannot vary).
- **The false line is the dangerous part.** *"regime not seen"* sends an operator to fix the GMM→bucket mapping, which is **correct** (`QUIET_ACCUMULATION`/`WEAK_CONSOLIDATION` → `Regime.SIDEWAYS`, verified). Exactly P155's lesson: naming a subsystem from a guess rather than from the data is worse than silence, because the named subsystem is innocent. **I read the false output myself and drew the wrong conclusion from it before checking the keys.**
- **What is actually true of kraken_quant** (from the archive log + bucket table, not the broken tool): the market has been **100% SIDEWAYS**, that bucket holds **4 of the 12** strategies, and **2 of those 4 are archived** (`OrnsteinUhlenbeck`, `DeltaNeutralFunding` — P157 decisions, `[KQ_ARCHIVE] 4 archived, 8 active`). So **only 2 of 12 strategies are reachable at all** — `KalmanCointegration_SOL_ETH` and `DarkPoolVolume` — and neither fired in 18h. A DECIDE agent whose reachable surface is 2 strategies is a config finding, not a bug.
- **Fixed:** writer keys by `regime.name` (matching its docstring); reader keeps a `_VALUE_TO_NAME` map so pre-fix snapshots still report what they recorded rather than reading as "everything dead"; `archived` exposed in the stats and labelled distinctly (archived and dead were both `attempts=0`); summary now reports **reachable-in-regimes-actually-seen** instead of a flat `x/12`.
- Tests: `tests/test_kq_diagnostic_keys_p215.py` (11), incl. a round-trip that every `Regime` the agent can emit is a key the reader's `CANONICAL` map holds, and an end-to-end render of the OLD on-disk shape. Falsification-checked. One assertion was initially too broad and failed on the tool telling the truth (BEAR/BULL genuinely were not seen) — scoped to the SIDEWAYS section.
- **NOT fixed (deliberate):** the 12 silent agents. Several are ADVISE-only and some are legitimately quiet; distinguishing "no signal" from "dead wiring" per agent is a real investigation, and bulk-enabling agents on a live account is not a bugfix. The `conf>0 while dir==0` pairs (`llm_sentiment` 0.42, `soldex` 1.00) are the strongest P2/P3 candidates and are where to start.
- **Mitigation pattern:** before believing any diagnostic, check that its reader and its writer agree on keys — a lookup miss degrades to a plausible zero, and zero reads as a finding. And when a report contradicts itself (header `chop=100%` vs bucket `regime_ticks=0`), the tool is broken; do not reconcile it in your head.

### P214. [FIXED 2026-08-07] Three runtime modules imported `training.*` code that was never in the image — swallowed ImportErrors whose fallback looked like ordinary output
- **Found by reading the live WARNING histogram**, not by any test. `Dockerfile.engine` copies runtime packages wholesale but only an **allowlist** of `training/` files; nothing checked that the allowlist covered what runtime code actually imports.
- **1. `training.scripts.wavelet_denoise` — a TRAIN/SERVE SKEW on 5 of the 122 model features.** Imported **every tick** by `data_mgmt/market_data_pipeline.py:855`. Never copied → `ModuleNotFoundError` → the `except` fell back to **RAW** values for `rsi_14/macd_12_26/bb_width_20/atr_14/vol_ratio_s`, while training used denoised. **Blast radius, measured not assumed:** the per-asset GMMs take **12 features, none denoised** (`models/regime_classifier/{ASSET}/gmm_config.json`), so REGIME classification — and everything it drives — is **unaffected**. The 5 land in the DRL observation vector, which is SHADOW, so **no live orders were affected**. But they **confound the live DRL IC** (P198's +0.019/−0.081) used to decide re-promotion, and would have silently skewed any retrained model too. Needed `PyWavelets` in `requirements-runtime.txt` as well — shipping the module without the library just moves the failure to `import pywt`.
- **2. `training.model_alpha.sequence_alpha_model` — an ADVISE agent 2/3 dead.** The **checkpoints are in the models volume**; the classes that unpickle them were not. Live: `No module named 'training.model_alpha'` for BTC and SOL (ETH happened to load), so `model_alpha` emitted **`+0.00/0.00`** for two of three assets — while `HEALTH_S7` reported *"model_alpha loaded"*. Any per-asset IC attributed to model_alpha over this period is measuring a degraded agent.
- **3. `training.regime.regime_classifier` — DELIBERATELY NOT SHIPPED.** `orchestration/strategic_coordinator.py:200`; live logs *"Ensemble Regime Classifier not available"*, so its consumer at `:595` has **never run**. It only tightens (reduces `max_short_exposure`, blocks new shorts in bullish regimes) and is stdlib-only, but switching on a never-executed decision path on a live account is an **operator call**, not a dependency fix (P141/P177). Recorded as an exemption **with its reason** in the gate.
- **Gate:** `tests/test_runtime_training_imports_p214.py` (15) scans runtime packages for `training.*` imports and asserts each is COPYed — plus that packages carry their `__init__.py`, that no COPYed path is dockerignored (P192), that the scanner finds something (anti-vacuity, P174), and that **every exemption is still real** (naming a module that is still imported and still unshipped), so it cannot become a parking spot. Falsification-checked.
- **Mitigation pattern:** an `except ImportError` whose fallback produces *plausible output* is invisible — the failure looks like a feature with no signal, not like a failure. Any cross-boundary allowlist (Dockerfile vs imports, Dockerfile vs `.dockerignore`, manifest vs packaging) needs a gate that reads **both sides**; otherwise they drift and only production tells you. Same family as P192 and P165.

### P213. [FIXED 2026-08-07] The shadow-IC gate ships in the image without its data — it now refuses instead of reporting `ohlcv_missing` as a result
- **The trap:** `.dockerignore` excludes `training/training_data/` (line 41) but NOT `analytics/`, so `compute_shadow_ic.py` **is** in the engine image and its 4H price parquets are not. Run it in-container and every strategy returns `ohlcv_missing`, a full table prints, and a report is written — output indistinguishable from *"the strategies have no signal"*. That exact conflation is what hid **P199** for months.
- **Decision (recorded, not left implicit):** this stays **OPERATOR-LOCAL**. Shipping the parquets into the image or mounting them costs real size for an occasional analysis tool whose data is refreshed from Binance monthly archives on the operator's box. The module docstring now has a `WHERE THIS RUNS` section stating that, with the reason and the two commands (`refresh_ohlcv_4h.py` then the gate).
- **Fix:** when **no** asset has a price series, `main` prints `REFUSING TO REPORT` naming the cause (dockerignore) and the fix, and returns **2** — before any report is written. Deliberately **ALL, not ANY**: one missing asset is a genuine data gap that should still produce a per-strategy report. Exit codes are distinct on purpose — **1** = no signals to score, **2** = no prices to score them against; collapsing them would recreate the no-signal/no-data conflation at the shell level.
- **Companion, opposite call — `scripts/trend_regime_review.py` ADDED to the P190 image allowlist.** Unlike the IC gate, its evidence (`data/trend_regime_shadow.jsonl`) is written to the **hmats-data volume**, so the data is already on the server and only the reader was missing — previously you had to `scp` the ledger to a laptop to read the promotion evidence. Stdlib-only, reads the ledger + Kraken's PUBLIC OHLC endpoint, places no orders (the P141 criterion). Needed BOTH the `Dockerfile.engine` COPY and a `.dockerignore` negation placed **after** `scripts/` — P192's lesson that naming a file in the build recipe is not putting it in the build context.
- Tests: `tests/test_shadow_ic_operator_local_p213.py` (13), incl. an end-to-end run that asserts the process really exits 2, and guards that the docstring's dockerignore rationale stays **true** (if the parquets stop being excluded, the operator-local justification is void and the test says so).
- **Mitigation pattern:** "where can this run?" is part of a tool's contract. When code and its data are governed by different exclusion rules, the tool ships somewhere it cannot work — and the failure looks like a *finding*. Decide server-side vs operator-local explicitly, write it in the docstring, and make the wrong environment a named refusal rather than an empty result.

### P212. [FIXED 2026-08-07] `[FEE-MODEL-MISMATCH]` was stale, vacuous AND backwards — retired and repointed at the condition that matters
- **The alert claimed** the alpha gate prices every asset with Kraken's tier and ended *"NOT auto-corrected"*. **P172 corrected it** (`resolve_venue_fee_bps`, resolved once per tick and reused by the `_fee_context` builder) and **P165 enabled `coinbase_venue_aware_fees` in the live profile**. Verified live 2026-08-07: all three assets log `[VENUE-FEE] … priced for COINBASE (taker=3.0bps maker=0.0bps)`.
- **Wrong three ways at once**, which is why rewording was not enough: (a) **stale** — told the operator to fix something already fixed; (b) **vacuous** — Kraken's tier is currently the FREE tier (0.0/0.0bps, monthly volume ~0 since Kraken stopped trading), so the "over-charged" magnitude it printed was `max(0, 0−3) = **0bps**`; (c) **backwards** — with Kraken at 0.0 and Coinbase at 3.0, the uncorrected model would UNDER-charge, so an operator acting on it would have loosened a gate that was already correct.
- **Fix:** fires only when an asset is routed **while `coinbase_venue_aware_fees` is OFF** — the case where the gate really is venue-blind, and at the free tier that means charging ~0bps for a venue that charges 3bps taker. Renamed `[FEE-MODEL-VENUE-BLIND]`, names the resolving action, **self-extinguishing**. A correctly-configured pass is silent and **does not consume the latch**, so turning the flag off later still warns (the P193 latch bug).
- **Retirement guard matches EMITTED output, not source** — the docstring deliberately names the retired string to explain the retirement, and a substring source-scan would fire on its own explanation (the P192 `_emergency_flatten` mistake).
- Tests: `tests/test_cutover_iron_law_8_wiring.py` (18).
- **Mitigation pattern:** same as P202 — when a fix lands elsewhere, the alert that motivated it does not become harmless, it becomes *misleading*, and a standing alert nobody can act on trains everyone to ignore the channel. Before trusting any long-standing warning, check whether the thing it describes is still true; a warning whose own printed magnitude is zero is proof it is not.

### P211. [FIXED 2026-08-07] LIVE persisted governor state but never restored it — and the first fix duplicated the restore instead of parameterising it
- **Found by reading the live log after deploying P209** (the only reason it was caught): the first fuse record came back with `cumulative_pnl: 0.0` and `history: 1`, i.e. prior state had **not** been read back. `_load_paper_positions()` is called from exactly one place — `run_paper():17375`. **`run_live()` never restored anything**, so live persistence was **write-only** and the fuse's 28d window still reset on every deploy. Persisting without restoring is the same non-control as not persisting at all.
- **Fix:** `_load_paper_positions(restore_positions: bool = True)`; `run_live` calls it with **False**. Restores the governors (existence fuse, cascade, failure memory, confidence scorer, opportunity budget, gambler, regime smoother, AC-5 counters, peak equity) and leaves `_paper_positions` untouched.
- **Why positions are deliberately NOT restored in live:** repopulating a Kraken book from a file is the **P139/P140** failure shape (a state machine acting on a view that does not reflect the venue), `_paper_positions` being empty is **load-bearing for P152/P206**, and the startup reconciler runs against the exchange a few lines later — it is the authority on that book, not this file.
- **Non-fatal in live, unlike the `run_paper` caller which fails closed.** Refusing to start LIVE because a diagnostics file is malformed turns a corrupt file into an outage, and `restart: always` turns that into **P85's** 10-restart loop. The lost history is announced at ERROR — a silent fresh start is indistinguishable from a healthy one.
- **The first attempt duplicated the restore inline in `run_live`** — a second hand-written reader of the same file, i.e. the reader/writer contract drift this codebase keeps producing (P2/P15/P85/P138/P139/P193). Replaced with the parameter so the two cannot drift; a test asserts there is only one `_load_paper_positions` and no second `from_dict` path.
- Tests: `tests/test_fuse_sleeve_feed_p209.py::TestLiveRestore` (7).
- **Mitigation pattern:** persistence is two halves and shipping one reads as done. After wiring a save, grep for the **load** and confirm the mode you care about calls it — then verify on a real restart, because the state file's mtime looks identical either way.

### P210. [FIXED 2026-08-07] `intent.target_exposure` is denominated in KRAKEN NAV — reconnecting it to the Coinbase sleeve would ~4× the book, not tighten it
- **The trap:** the last "gap" looked like an obvious two-line win — the risk stack sizes each asset and the sleeve throws it away, taking a flat ±1 contract. **Do not reconnect it.** `core/unit_system.py:232` computes `usd_notional = abs(exposure_fraction) * account_equity`, and `account_equity` comes from `account_sync`, constructed `exchange_name="kraken"` (`main.py:2866`) → it is **Kraken NAV (~$9.8k)**, not sleeve equity (~$3.8k). At `target_exposure=0.25` that sizes **~$2,457 against the ~$643 the sleeve holds** — ~4× the book, on a strategy measured at Sharpe −4.5 — and denominates one venue's risk in another venue's capital (the P139/P140 cross-venue-contamination family).
- **Therefore:** the ±1-contract cap is currently the **tighter** control. "Reconnecting the risk stack" here would have been a risk *increase* wearing the language of a risk fix. Entry sites: `core/execution_service.py:990` (entry/resize) and `:976` (full-exit fallback); sliced execution divides an already-computed `base_quantity` (`:2100`) so it shares the denominator.
- **What shipped instead (P210):** the control `target_exposure` was *supposed* to provide, denominated correctly — the **same policy numbers** the Kraken path uses (`post_leverage_caps`: BTC/ETH 0.25, SOL 0.20) enforced in `can_trade` against **sleeve** equity. Wired `main.py` → `CoinbaseSleeve(max_asset_exposure=...)`.
- **Why it is not a control that can never fire:** contract granularity is **fixed** while equity moves — one BTC nano is ~17% of a $3.8k sleeve but **~34% of a $1.9k one**. The contract cap counts *contracts* and cannot see it; P208's net budget *aggregates* and so passes a single concentrated asset. This binds as the account shrinks, saying "one contract is now too large for this account" instead of quietly holding an oversized position. Tests pin that both existing caps would have allowed the blocked case.
- **Invariants carried from P195/P208:** gates only orders that INCREASE absolute exposure (de-risking always free, over-cap positions always trimmable); pricing failure fails **OPEN**; halt takes precedence; `getattr`-defended attribute read (P85).
- Tests: `tests/test_sleeve_asset_exposure_p210.py` (22).
- **Mitigation pattern:** before reconnecting any "computed but discarded" value, **trace its denominator**. A fraction is meaningless without knowing what it is a fraction *of* — and in a multi-venue system the plausible denominator (this venue's equity) is often not the actual one. Same "wrong-but-plausible field" family as P2 and P153.

### P209. [FIXED 2026-08-07] Existence fuse (Non-Negotiable Rule #3) had an EMPTY pnl_history since Phase B — and `run_live()` never persisted state at all
- **Symptom (latent, never fired):** the live fuse's persisted `pnl_history` held **1 record, from 2026-06-12**, while the Coinbase sleeve carried 100% of the directional risk. The 28d-window loss halt could not fire because nothing fed it.
- **Root cause A (input):** all three `existence_fuse.record_pnl()` call sites are in `core/execution_service.py` (:2657, :3178, :3463) — **past the P152 early return**, so for a Coinbase-routed asset they never execute. Same "the control lives on the dead Kraken path" shape as P201 (drawdown halt / kill switch) and P208 (net cap).
- **Root cause B (persistence — the one that made everything else moot):** **`run_live()` never calls `_save_paper_positions()` anywhere.** The three per-tick calls (:17800, :18046, :18057) are all inside `run_paper()`; the live-reachable ones (MAX_HOLD_TIMEOUT, FastRiskTick, CORR-0) each require a Kraken `_paper_positions` entry, and that dict has been `{}` since the 2026-06-13 flatten. Live `paper_positions.json` was last written **2026-06-13T02:17Z** — 8 weeks stale. So the fuse history, cascade governor, failure memory, confidence scorer, opportunity budget, regime smoother, AC-5 daily fill counters **and `peak_equity`** all reset on every deploy. This is why P140/B2 observed `_peak_equity` re-initialising. Same in-memory-baseline class as **P150** (sleeve DD baseline) and **P148** (DRL frame buffer). `run_live` is a partial copy of `run_paper` and keeps losing things — cf. the note at `main.py:18215`.
- **Root cause C (would have silently undone the fix):** the restore's capital-regime guard resets the fuse when `max(saved_starting_equity, initial_capital)/min(...) > 2.0`. Sleeve ~$3.8k vs `initial_capital` $10k → ratio **2.65** → the history would be discarded on **every** restart. Note the natural alternative (compare against *current* equity) is worse: it resets hardest exactly when a drawdown is deepest.
- **Fix (P209):** per-tick in `run_live`, feed `record_pnl(realized_pnl=<equity delta vs persisted anchor>, current_equity=<sleeve equity>, trade_count=0)`; persist immediately after via `_save_paper_positions(force=True)`; persist/restore the anchor + an `existence_fuse_equity_basis` marker; skip the capital-regime guard (loudly) for a `coinbase_sleeve`-denominated series.
- **Deliberate design choices, each a trap avoided:** (a) **`on_trade_close()` is NOT called** — it counts a consecutive-LOSS streak and suspends at 10, and a 4H mark-to-market tick is not a trade (10 red ticks ≈ 1.7 days of drift would halt the system); (b) **not retroactive** — the first point anchors with delta 0, so the sleeve's inception-to-date −5.6% does not suspend on tick one for PnL the fuse could never have acted on; (c) **a stale read is skipped, not recorded as 0** — `sleeve_equity_usd()` returns the last KNOWN value on API error, so an unguarded delta would enter the window as "no loss"; the feed requires `_reconcile_ok`; (d) denominated in **sleeve** equity, not Kraken NAV or the combined book — a total-book denominator would make it ~3.6× less sensitive.
- **Output half was already wired (verified, not assumed):** a suspended fuse sets `veto_reason=[STRATEGY_SUSPENDED]`, which P206's translator classifies as neither a HOLD veto nor venue-inapplicable → falls through to `veto_flat` → the sleeve flattens.
- **Live thresholds now armed** (note CLAUDE.md's old "−5%" is stale — UNLEASH widened these): 28d window −15% PnL or −18% equity, weekly −15%, monthly −18%, needing 2 consecutive 4H evaluations; `min_data_points=5` ≈ 20h before it can evaluate at all.
- Tests: `tests/test_fuse_sleeve_feed_p209.py` (23) — including `test_a_sustained_loss_actually_suspends`, asserting the control **can fire** rather than inferring it from wiring.
- **Mitigation pattern:** a risk control needs THREE things, and checking only the first two is how this family keeps recurring — an input on the live path, an output that binds, and **persistence** across restarts. Any halt/baseline/high-water-mark that lives only in RAM is not a control. When a persist helper exists, verify the *live* loop calls it; `mtime` on the state file is the cheapest possible check and would have caught this in one command.

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

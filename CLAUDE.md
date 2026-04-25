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

## Known Pitfalls (source of repeat bugs — read this before deploying)

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

### P25. [FIXED 2026-04-24] `ctx.intent` was undeclared → PnL attribution wrote empty `primary_agent` to every fill
- **Symptom:** `shadow_ledger` FILL records all carried `primary_agent=""` despite the 2026-04-22 fix that was supposed to populate it for `agent_audit_16.py DIM 4`. Agent attribution quality metric was permanently 0 across the fleet.
- **Cause:** `execute_intent_v2` writes `"primary_agent": getattr(ctx.intent, "primary_agent", "") or ""` at 4 sites (entry record + 3 exit/flip record paths). But `ctx.intent` was never assigned anywhere — `ExecutionContext` doesn't declare it as a dataclass field, `build_from_runner` doesn't set it, and the function receives `intent` as a parameter, not through ctx. So `getattr(None, "primary_agent", "")` returned `""` every call.
- **Fix:** Replaced all 4 sites with `getattr(intent, "primary_agent", "") or ""` — reads the function parameter instead of the ghost ctx attribute.
- **Mitigation:** When migrating from `self.X` to `ctx.X` during a god-object extraction, diff the parameter list — anything that arrives as a function argument must NOT be accessed via ctx. Grep `ctx\.(asset|intent|market_data|agent_signals)` after a migration.

### P26. [FIXED 2026-04-24] Scalar drift: `ctx.last_aging_check` never synced back → weekly log rate-limiter broke
- **Symptom:** The "weekly strategy-aging" critical log (warning when a strategy's weight modifier < 0.7) was supposed to fire at most once per 168 hours per ctx lifetime, but was firing every trade close where a strategy had record_outcome called.
- **Cause:** `ExecutionContext.sync_scalars_back()` was defined but never called. `execute_intent_v2` mutates `ctx.last_aging_check = _c12_now` (a datetime), but the mutation lives only on the ctx, which is rebuilt each call via `build_from_runner`. Next call reads `runner._last_aging_check`, which stayed `None` forever (only set to None once at init), so the "is None" guard always re-triggered the log.
- **Fix:** (a) Added `ExecutionContext._runner_ref` field, populated by `build_from_runner` with a live runner handle. (b) Rewrote the aging-check branch in execution_service.py to read/write `ctx._runner_ref._last_aging_check` directly. (c) Retired the `last_aging_check: float = 0.0` dataclass field (type was wrong — always held a datetime when set) and the never-called `sync_scalars_back()` method.
- **Mitigation:** When adding a scalar to `ExecutionContext`, ask "how does this write get back to the runner?" Three acceptable answers: (1) it doesn't need to (tick-local, read-only, or piped through a shared component reference like `anti_churn._fills_today`); (2) a dedicated `fn_sync_*` callback writes it; (3) it writes through `ctx._runner_ref`. A raw `ctx.X = value` with no plan is scalar drift.

### P27. [CLEANUP 2026-04-24] Write-only runner flags + dead OOD reader removed
- **Symptom:** Three runner attributes (`self._ac0_requires_entry_block`, `self._paper_restore_failed`, `self._resolved_config`) were assigned at multiple sites but had **zero readers** anywhere in the repo. They would pass `hasattr` checks, show up in IDE autocomplete, and tempt a future reader to wire them into a gate or status export — creating phantom guard paths.
- **Cause (specific):**
  - `_ac0_requires_entry_block` (6 writes): the real AC-0 entry block uses per-asset `_ac0_restored_assets` via `_get_ac0_entry_block_reason()`; the boolean mirror was never consumed.
  - `_paper_restore_failed` (5 writes): restore failures are already logged at their failure site and `_load_paper_positions()` returns `False`, which is what callers actually check.
  - `_resolved_config` (2 writes): `ConfigResolver.resolve_and_log()` does its logging inside the method; the returned dict had no consumer.
- **Also cleaned:** dead `_ood_confidence_mult` reader at `integration_v36.py:2182`. The OOD confidence-multiplier path was deliberately retired 2026-04-11 (decision: detect+log OOD but don't penalize DRL confidence — insufficient live data to tell "model wrong" from "regime shift"). The reader stayed on the books with a default of `1.0` (harmless no-op) but was misleading about whether OOD still influenced DRL weights.
- **Also cleaned:** `ctx.current_drawdown_pct` was assigned in `build_from_runner` via `__dict__` (not declared on the dataclass). Now properly declared as `current_drawdown_pct: float = 0.0`.
- **Fix:** All write sites removed (main.py inits + 3 load paths + 2 test-file mock setups); dead OOD reader deleted; current_drawdown_pct field added to ExecutionContext dataclass. Left breadcrumb comments at each removal site pointing at the live replacement.
- **Mitigation:** When an attribute survives only as writes, delete it in the same pass. A write-only flag **is** a bug — it advertises a contract that nothing enforces. Run the `reads_by_attr` audit from this session's scripts periodically.

### P23. [FIXED 2026-04-24] AC-5 daily fill budget cap was a silent no-op
- **Symptom:** The "max 8 fills/day" hard cap (AC-5) never blocked anything. No `AC5_BUDGET_EXHAUSTED` log line ever appeared in production, even on days where fill count exceeded the budget.
- **Cause:** Two duplicate counters. `AntiChurnManager._fills_today` (core/anti_churn.py) is the canonical one — incremented by `record_fill()` at the end of every execution, persisted via `to_dict()`/`from_dict()`. BUT the execution path's AC-5 gate at `execution_service.py:670-685` was reading `ctx.ac5_fills_today`, which was sourced from `runner._ac5_fills_today`. That runner attribute was only ever assigned `0` once at startup ([main.py:2068](main.py#L2068) with the comment "Proxied through _anti_churn") and never incremented. The gate check was always `0 >= 8 = False`. The canonical `anti_churn.check_fill_budget()` method exists and is correct — just has zero callers.
- **Fix:** Rewrote the execution_service AC-5 gate to read `ctx.anti_churn._fills_today` / `_fills_date` directly. Removed the dead `ac5_fills_today`/`ac5_fills_date` mirror fields from `ExecutionContext`.
- **Expected behavioral change:** On high-volume days (more than 8 fills), trades past the 8th will now return `AC5_BUDGET_EXHAUSTED`. This is the intended governor; the system's been running without it since the Phase 4B refactor.
- **Mitigation:** When there are two places holding the same state (a dedicated manager + scalar mirror on the god-object), grep `record_*`/`increment_*` to see which one is actually being updated. Prefer reading from the authoritative manager; delete the mirror.

### P24. [FIXED 2026-04-24] Discord log-handler dedup key was message-based → spam risk
- **Symptom:** Not yet observed in production (DiscordLogHandler only went live 2026-04-24). Would have manifested as Discord alert spam under persistent tick-loop errors like `SOTA integration error: {e}` where the exception `repr` includes a counter/timestamp.
- **Cause:** Dedup key was `f"{record.name}:{record.getMessage()[:100]}"`. Tick-loop catch-all `except` blocks produce messages whose tails vary per tick (embedded timestamps, per-asset identifiers). Same code path → different dedup keys → one alert per tick → Discord rate limit or user fatigue.
- **Fix:** Keyed on call site `f"{record.pathname}:{record.lineno}:{record.levelno}"` instead. The log file still has the full variant detail; Discord just gets one alert every 5 min per recurring code location.
- **Mitigation:** When building a dedup key for an alerting handler, key on the *origin* of the alert (file+line), not the *content* of the current instance. Content-based dedup assumes messages are stable, which isn't true for error handlers interpolating exceptions.

### P28. [NEW 2026-04-24] THREE separate DRL systems — supersedes P10
- **Status:** P10 originally documented "TWO SEPARATE DRL systems". A third was added 2026-04-24 in `training/exit_drl/`. Update P10's mental model accordingly.
- **System 1 — TQC direction DRL** (`drl/ensemble.py` + `models/retrained/{ASSET}/fold_3/.../best_model.zip`): predicts direction + confidence; feeds `agent_signals["drl_direction"]`/`drl_confidence`; authority ACTIVE; Sharpe +9 on val backtest. **The main DRL.**
- **System 2 — `agents/drl_agent.py` DRLAgent** (DISABLED): tranche/exit optimization scaffolding. Per its own docstring: "DRL DOES NOT DECIDE direction". Phase-2 stub, no trained model, returns neutral. Not dead — dormant.
- **System 3 — Exit DRL (Discrete SAC)** (`training/exit_drl/` + `agents/exit_drl_agent.py`, ACTIVE-SHADOW 2026-04-24): exit-timing-only discrete SAC over {HOLD, PARTIAL_EXIT, RELEASE_RUNNER, EXIT_ALL}. Trained from oracle-labeled trajectories on the same `{ASSET}_4H_full.parquet` as TQC (per-asset val alignment 0.676–0.707). Authority cap = EXIT_ONLY. Stage 3 v1 SHADOW landed — runs `.predict()` per tick on every active position, logs to `data/exit_drl_shadow.jsonl`, **does not influence trading decisions**. `execution/exit_alpha.py` triggers (phase / CRACK / momentum / DRL PARTIAL_EXIT / drawdown) keep operating exactly as before.
- **Differences that matter** (don't conflate):
  - TQC is continuous control of position size; Exit-SAC is discrete classification of exit action. Different action spaces, different libraries.
  - TQC's 122-feature obs is per-bar market state; Exit-SAC's 40-d state is **position-aware** (includes bars_held, unrealized_pnl, drawdown_from_peak, is_runner — features TQC never sees).
  - TQC predicts; Exit-SAC reacts to a position. Calling TQC "the exit DRL" is wrong even though it can output 0 (close).
- **What lives where:**
  - Trajectory generator: `training/exit_drl/generate_expert_trajectories.py` (oracle uses 48-bar lookahead; output: `training/exit_drl/data/{ASSET}_exit_trajectories.npz`).
  - Trainer: `training/exit_drl/train_exit_sac.py` (Discrete SAC + CQL regularizer + balanced per-action sampler).
  - Models: `models/exit_drl/{ASSET}/exit_sac_best.pt` (saved by trainer).
  - Runtime agent: `agents/exit_drl_agent.py` — `ExitDRLAgent` + `get_exit_drl_agent()` singleton + `ExitDRLMode` enum {DISABLED, SHADOW, EXIT_ONLY}. Loads checkpoints through `infra/safe_torch_load.safe_torch_load` (path-allowlisted).
  - Wiring: instantiated in `_init_components` after DRLShadowDiagnostics; per-tick `.predict()` is invoked from the agent_signals tick block (right after the TQC DRL signals get cached on `_tracked_pos`).
  - Shadow log: `data/exit_drl_shadow.jsonl` — one JSONL line per (asset, tick-with-active-position) with action/probs/state-summary/inference_ms.
  - Tests: `tests/test_exit_drl_trajectory_gen.py` + `tests/test_exit_drl_sac.py` + `tests/test_exit_drl_agent.py` + `tests/test_exit_drl_promotion_gate.py` — **31 tests total, all passing.**
- **Phase A (retraining at 200 epochs, 2026-04-24):** Initial 50-epoch checkpoints (`models/exit_drl/`) had val_align 0.676/0.677/0.707 — only SOL cleared the spec's 0.70 minimum. Retrained all 3 at 200 epochs (seed=42, same balanced sampler + CQL_alpha=1.0): **BTC 0.730 @ epoch 138, ETH 0.710 @ epoch 85, SOL 0.746 @ epoch 122 — all clear.** Saved to `models/exit_drl_v2/`; runtime defaults to v2 with v1 fallback. Beyond ~150 epochs alignment plateaus and oscillates ±0.05; not worth more compute.
- **Phase B (validation harness + outcome ledger + gate, 2026-04-24):** all built and tested.
  - `training/exit_drl/validate_against_baseline.py` — replays held-out 20% of timeline through Exit-SAC + a pure-Python mirror of `execution/exit_alpha.py`'s rule-based triggers (phase/CRACK/momentum/drawdown/stop-out). Per-asset Sharpe lift: **BTC +50.0%, ETH +83.0%, SOL +91.3%** (vs +10% threshold — all pass). **Caveat: absolute Sharpe is *negative* for both actors on the held-out window. DRL "loses less", doesn't generate alpha.** Held-out window covers a tough crypto period (mostly drawdowns); DRL holds longer and avoids stop-out cascades the rule-based actor triggers. In a trending market the comparison may invert. Validator now reports BOTH a `final_action_dist` (terminal action — useful for spotting EXIT_ALL panic) AND a `step_action_pct` (every-bar policy histogram — the honest picture). On the v2 checkpoints: DRL HOLD 57-63% / PARTIAL_EXIT 26-31% / RELEASE_RUNNER 0.2% / EXIT_ALL 11-12% per step; baseline HOLD 69-74% / PARTIAL ~1% / EXIT 17-19%. DRL fires PARTIAL_EXIT ~15× more often than the rule-based mirror — that's where its Sharpe lift comes from. All 3 assets sit inside the gate's HOLD ∈ [50%, 90%] + EXIT_ALL ≤ 30% guards. Validator output: `data/exit_drl_validation/{ASSET}_validation.json`.
  - `agents/exit_drl_outcome_ledger.py` — per-trade outcome ledger writer. `record_open(asset, entry_price, direction)` on position open, `record_prediction(asset, action_name, confidence, unrealized_pnl, bars_held)` on every Exit-SAC prediction during the trade, `record_close(asset, exit_price, exit_reason, realized_pnl_bps)` on close. Flushes one JSONL line per closed trade to `data/exit_drl_outcome_ledger.jsonl`. **Both halves of the loop wired:** open + prediction at [main.py:7457](main.py#L7457) (per-tick predict block); close at three real sites — BRANCH A full-exit ([core/execution_service.py:1973](core/execution_service.py#L1973)), BRANCH C flip-close ([core/execution_service.py:2724](core/execution_service.py#L2724)), and paper-mode emergency flatten ([main.py:13750](main.py#L13750)). BRANCH B (partial exit) intentionally skips `record_close` — the position remains open. Static guards in [tests/test_exit_drl_close_integration.py](tests/test_exit_drl_close_integration.py) verify each close site + the partial-skip invariant. **This prevents the P15-style half-wired feedback loop** (read side wired, write side never fed) — caught in this session before deploy.
  - `risk/exit_drl_promotion_gate.py` — read-only gate. `evaluate(asset)` returns `{would_promote, blockers, evidence, thresholds}`. Thresholds: ≥30 shadow days, ≥30 closed exit events, +10% Sharpe lift, HOLD ratio ∈ [50%, 90%], EXIT_ALL ratio ≤ 30%. **Current state (offline):** Sharpe lift threshold passes for all 3 assets; remaining blockers are shadow days (0 < 30) and exit events (0 < 30) — both accumulate after deploy.
- **Promotion path remaining (post-shadow):**
  1. Deploy and let the agent shadow-run for 30+ days, accumulating ≥30 closed exit events per asset in `data/exit_drl_outcome_ledger.jsonl`.
  2. Re-run `validate_against_baseline.py` on the *most recent* held-out window (recompute lift on a market regime closer to live conditions — the current +50/+83/+91% lift was on the 2024-2025 drawdown window).
  3. Run `ExitDRLPromotionGate.evaluate_all()` — confirm `would_promote=True` for the asset(s) you want to promote.
  4. Only then: wire `ExitDRLAgent.predict()` into `execution/exit_alpha.py`'s TRIGGER 4 (`DRL_ACTION`) by exposing a `DRLOutput`-compatible bridge (System 3's prediction stands in for the dormant System 2). Flip `ExitDRLMode.SHADOW` → `ExitDRLMode.EXIT_ONLY` only for the asset(s) the gate clears.
  5. Add a kill switch: same flip in reverse on any of `(consecutive_losses ≥ 5)`, `(7-day Sharpe vs baseline drops below +0%)`, or `(action distribution drifts outside HOLD ∈ [50%, 90%])`.
- **Mitigation:** When `startup_drl_truth.py` is extended for a fourth DRL family, add a System-N row here. Sharing a single gate across two DRLs would create cross-coupling. **Don't promote on offline-only validation alone** — the live shadow phase is what catches "DRL learned a regime that doesn't exist in production" failures (a known offline-RL failure mode per the spec's Stage 4 pitfall #6).

### P41. [DIAG 2026-04-25] Regime-conditional sign flip is the rule, not the RSI exception — 75% of backfill signals affected
- **Source script:** `analytics/ic/strategy_sign_flip_audit.py` — extends P40's RSI finding to all 8 backfill signal families.
- **Result:** 18/24 backfilled signal columns (BTC 6/8, ETH 8/8, SOL 4/8) show statistically significant IC with **opposite signs** across regimes. The user's prediction was 5-6 of 12; actual is 75% of measurable signals.
- **Universal sign-flippers (3/3 assets):** bb_backfill.direction/position, meanrev_backfill.direction/z_score. The whole mean-reversion family inverts between QUIET_ACCUMULATION and REVERSAL/TRENDING regimes.
- **2/3-asset sign-flippers:** quant_backfill.direction/rsi (positive IC in WEAK_CONSOLIDATION/QUIET, negative in trending and chop).
- **ETH worst case:** 100% sign-flip rate. Every backfilled signal inverts somewhere across the 8 regimes.
- **Implication:** the Best-of-N selector at `data_mgmt/market_data_pipeline.py:1244` aggregates strategy outputs as if regime-independent. When two sub-signals have opposite signs (one trending-correct, one consolidation-correct), they cancel → str < threshold → quant.direction = 0. This is why live quant routinely lands at +0.000 in the current WEAK_CONSOLIDATION regime — not because there's no signal, but because the static rule throws away regime-conditional alpha.
- **What's NOT proven:** that a regime-conditional rewrite would yield $X of additional alpha. The 75% sign-flip rate confirms the *information exists*; it doesn't confirm the *fusion architecture can capture it*. Backfill sign-flip rate also includes some walk-forward noise (per P40 Task 2). Need 7+ days of live IC accumulation to validate the pattern out-of-sample before acting.
- **Caveat on the 12 kraken_quant strategies:** can't audit them directly (cointegration / Hurst / ETF basis need live cross-asset data). By analogy with the 75% rate on price-derivable signals, the same pattern is plausible. Will become measurable from `analytics/ic/compute_ic.py` after 7-day live accumulation.
- **Recommended action (deferred per P40 caveat):** implement regime-conditional Best-of-N weights → for each strategy, weight `sign(IC_for_current_regime) × |IC_for_current_regime|`. Strategies with negative IC contribute with flipped sign (not zero). Strategies without significant IC in current regime weight 0. **Wait for 7-day live IC validation before implementing** to mitigate in-sample-fit risk.
- **Mitigation:** Re-run `python analytics/ic/strategy_sign_flip_audit.py --all-assets` monthly to detect regime-calibration drift across the price-driven signal family.

### P40. [DIAG 2026-04-25] Backfill-IC vs live-alpha gap — RSI sign-flip by regime + walk-forward instability
- **Source script:** `analytics/ic/diagnostic_ic.py` — runs three independent diagnostics on the same 365-day backfill and live engine logs.
- **Task 1 (regime-conditional IC, BTC h=12b):** RSI's information coefficient flips sign by regime, not magnitude. Contrarian (negative IC) in TRENDING_BULL (-0.137 p=0.015), VOLATILE_CHOP (-0.154 p=0.007), REVERSAL (-0.216 p=0.017). Momentum-aligned (positive IC) in WEAK_CONSOLIDATION (+0.096 p=0.085) and QUIET_ACCUMULATION (+0.501 p=0.004 small N). The static `<30 buy / >70 sell` Best-of-N rule is **calibrated for trending markets** and silently mis-fires in consolidation. Fortunately the live system rarely fires RSI as a non-zero direction in WEAK_CONSOLIDATION — the strength threshold gates it to 0 — so no realized loss, just uncaptured alpha. ETH and SOL show the same pattern.
- **Task 2 (walk-forward IC, 5 chunks × 10 months):** All three assets show std ≈ mean (UNSTABLE / MIXED). Per-chunk quant_backfill.direction IC for BTC: -0.010, +0.096, -0.112, +0.039, +0.237. The aggregate +0.05 average **is carried by 1-2 favorable chunks**, not a persistent property. Chunks 3-4 (Aug 2025 - Jan 2026, regime-change zone) had near-zero or negative IC. Implication: backfill IC numbers are partially in-sample artifacts; **don't act on aggregate IC < 0.10 without 7+ days of live shadow data**.
- **Task 3 (live RSI signal flow):** Live `[QUANT_SIGNAL]` log lines show RSI sub-component currently contributes ±0.01 to ±0.05 to quant.direction. With Best-of-N picking `hold` strategy in WEAK_CONSOLIDATION (sig=0.000), `quant.direction` lands at 0 → RSI's contribution to fused alpha = 0 bps. DRL provides the only non-zero direction (-0.69 BTC, -0.96 ETH today); fusion correctly defers via DECIDE_SOLO passthrough (P30 fix).
- **Synthesis:** All three diagnostic explanations contribute. RSI is regime-conditional (not regime-useless), backfill IC is unstable, and live fusion correctly handles the resulting noise. **Combined diagnosis: the system isn't broken; it's correctly silencing a miscalibrated signal in the current regime.**
- **Recommended actions (NOT yet taken):**
  1. Don't deprecate RSI — it has IC in TRENDING regimes which the system isn't currently in.
  2. Implement regime-conditional Best-of-N strategy weights — RSI weight ∝ regime in {TRENDING_BULL, TRENDING_BEAR, VOLATILE_CHOP, REVERSAL}, zero out in {WEAK_CONSOLIDATION, QUIET_ACCUMULATION}. ~1-day implementation.
  3. Wait for live IC accumulation (7+ days) before acting on per-agent IC numbers — task 2 instability is a strong reason to NOT trust aggregate backfill IC.
- **Mitigation:** Run `python analytics/ic/diagnostic_ic.py --all-assets --task all` monthly to re-check whether backfilled signals still meaningfully predict, and per-regime to detect calibration drift.

### P31. [NEW 2026-04-25] One-shot health monitor (`scripts/hmats_monitor.sh`)
- **What it does:** Single command audits 8 axes of the live engine. Cron-friendly (non-zero exit on FAIL), watch-mode for live observation.
  ```bash
  bash scripts/hmats_monitor.sh           # snapshot
  bash scripts/hmats_monitor.sh --watch   # 60s refresh
  HMATS_HOST=staging bash scripts/hmats_monitor.sh
  ```
- **What it checks:**
  1. Container state (engine + api + dashboard up + healthy)
  2. Engine healthcheck (status + failing_streak + restarts)
  3. Recent fatal-level log entries (CRITICAL / FATAL / Traceback)
  4. Tick liveness (last LIVE_DATA / AGENT-TRACE within 15min)
  5. Fusion behavior (CONSENSUS / CONFLICT / SOLO / ABSTAIN counts in last 1h — directly observes whether P30 is working)
  6. Exit-DRL state (per-asset modes + kill-switch demotion count + EXIT_DRL_BRIDGE fire count)
  7. Trade activity (fills, alpha-gate passes, gate rejects in last 24h)
  8. Resource usage (cpu / mem / mempct per container, FAIL above 90% mem)
- **FAIL conditions** (non-zero exit, suitable for paging via cron + Discord):
  - Container stopped/unhealthy/restarting
  - Healthcheck unhealthy
  - ≥5 critical log entries in 30min
  - ≥1 Exit-DRL kill-switch demotion in last 1h
  - Exit-DRL `ready=NONE` (checkpoints failed)
- **WARN conditions** (informational, exit code 0):
  - 1-4 critical entries (often false positives like `[HEALTH_STARTUP] ... 0 CRITICAL` summary text)
  - No tick activity in 15min (could be normal between 4H candles)
  - Fusion 100% conflict (agents persistently disagree)
  - Alpha gate passing but no fills (downstream blocker like AC-2 / weekend filter / sizing — see WeekendManager investigation 2026-04-25)
  - Memory > 90%
- **Mitigation:** Run after every deploy. Add to cron @5min for paging. The script's stage-by-stage output is the standard verification tool going forward.

### P30. [FIXED 2026-04-25] Multi-DECIDE fusion: abstain treated as disagreement, linear weighting overcorrected → "DECIDE_CONFLICT" on solo strong signals
- **Symptom:** With DRL promoted to ACTIVE (and now also Exit-SAC EXIT_ONLY for all 3 assets), the most common fusion log line was `[DECIDE_CONFLICT] 3 agents, low agreement (0.33), confidence=0.10`. Observed live on BTC at 2026-04-25 05:09: DRL=-0.93/0.44 with quant=0 and kraken_quant=0 (both abstaining) → fusion treated DRL as a 1-of-3 minority signal, dampened confidence to 10%, alpha gate blocked. System held instead of acting on a strong DRL signal in a regime where TA had no view.
- **Cause:** `signals/authority_fusion.py:548-568` had three issues vs Bayesian Model Averaging best practice:
  1. Agreement metric used `_n_total = len(decide_agents)` as the denominator, counting abstainers (`|dir|<0.01`) as disagreement. Aligns with neither Black-Litterman ("no view = no contribution") nor BMA (zero-information models contribute zero weight, not zero direction).
  2. No solo-conviction passthrough — a single high-conviction agent surrounded by abstainers always landed in the conflict path even though there was nothing to disagree with.
  3. Confidence-linear weighting (`w = confidence`) understates the BMA inverse-variance optimum (`w ∝ 1/var ∝ confidence²`). High-conviction agents got 3× the weight of low-conviction ones rather than the optimal 9×.
- **Fix:** Three changes at `signals/authority_fusion.py:521-625`:
  - **FIX-1: active-only agreement** — `_n_total = _n_pos + _n_neg` (exclude abstainers). BTC's solo-DRL case now shows `agreement=1.00` instead of `0.33`. ETH's genuine DRL-vs-model_alpha disagreement still shows `agreement=0.5` and gets correctly flagged.
  - **FIX-2: solo-conviction passthrough** — when exactly 1 active agent has `confidence ≥ 0.5`, log as `[DECIDE_SOLO]` and preserve full confidence rather than running through the conflict path.
  - **FIX-3: confidence-squared weighting** — `w = confidence²`. avg_conf reported back as `sqrt(avg(confidence²))` so the alpha gate still receives a comparable [0, 1] confidence number. Direction is sharper toward high-conviction agents.
- **Tests:** `tests/test_authority_fusion.py` — 6 new tests (`TestFusionV2_*`) covering: BTC solo-DRL case (full signal preserved), ETH genuine-disagreement case (still flagged), all-abstain case (zero confidence), solo high-conviction passthrough, solo low-conviction does NOT passthrough, conviction-squared dominates linear. **27/27 fusion tests pass; 83/83 across the wider DRL test surface pass — no regression.**
- **Research basis:** Black & Litterman 1992 (no-view contribution); Hoeting et al. 1999 (Bayesian Model Averaging, inverse-variance weighting); López de Prado 2018 (conviction-weighted aggregation in finance). The pre-fix logic was overcorrecting against the P12 "rogue agent" failure mode; FIX-1 is the principled correction (abstain ≠ disagree), FIX-2 is the conservative path that preserves the same protection while honoring solo conviction, FIX-3 is the BMA optimum.
- **Mitigation:** When fusing N heterogeneous agents, "abstain" must NOT be counted as a vote against. Use `n_active = n_pos + n_neg` for the agreement denominator, and weight by `confidence²` (BMA inverse-variance) rather than linear confidence.

### P29. [OVERRIDE 2026-04-24] Exit-SAC promoted to EXIT_ONLY for ALL 3 ASSETS via accelerated path
- **Status:** explicit deviation from the spec's 30-day shadow + ≥30-real-exit-events gate. Started with BTC-only, then promoted ETH+SOL same session after end-to-end diagnostic confirmed wire-up healthy. Approved by user under "go with the accelerated path" + "i want all three be active, since we have monitor". Records: `data/exit_drl_promotion_state.json` (force_promote_at + reason + blockers-at-override per asset).
- **Why all 3 instead of phased rollout:** the codebase has structurally sound per-asset isolation. The kill switch tracks `consecutive_losses`, `pnl_history`, `recent_step_actions`, `recent_close_actions` independently per asset; the agent's `_asset_modes` dict is per-asset; the bridge's `should_act_on(asset)` consults the per-asset mode. End-to-end diagnostic ([scripts/exit_drl_e2e_diagnostic.py](scripts/exit_drl_e2e_diagnostic.py)) proved Stage-6 (kill switch) trips on BTC at 5 losses while ETH stays clean — isolation works. The remaining argument for phased rollout was "watch for production-only bugs in the bridge code that the test suite missed" — user accepted that risk on the basis of the kill-switch monitor.
- **Pre-override evidence:**
  - Per-asset val alignment (200ep): BTC 0.730, ETH 0.710, SOL 0.746 — all ≥0.70 spec floor.
  - Offline Sharpe lift vs rule-based mirror: BTC +50%, ETH +83%, SOL +91% (vs +10% threshold). **Caveat: absolute Sharpe is negative for both actors on the held-out window — DRL loses less, doesn't generate alpha.**
  - Per-step action distribution: BTC HOLD 62.6% / PARTIAL 26.6% / EXIT 10.6% — inside the gate's HOLD ∈ [50%, 90%] + EXIT_ALL ≤ 30% bands.
  - Promotion-gate evidence at override: 0/30 shadow days, 0/30 exit events, +50% Sharpe lift.
- **Five required guardrails (all landed):**
  1. **Kill switch** ([risk/exit_drl_kill_switch.py](risk/exit_drl_kill_switch.py)) — 4 trip conditions: 5 consecutive DRL-driven losses, 7-day rolling realized PnL < 0, HOLD ratio drift outside [50%, 90%] over last 50 ticks, EXIT_ALL ratio > 30% over last 50 closes. Trips evaluated on every tick before consulting Exit-SAC; auto-demotes the offending asset to SHADOW. State persisted to `data/exit_drl_kill_switch_state.jsonl`.
  2. **Bridge limited to PARTIAL_EXIT only** ([core/tick_exit_triggers.py:373-403](core/tick_exit_triggers.py#L373)) — System-3 cannot fire RELEASE_RUNNER or EXIT_ALL via the bridge. The existing 5 rule-based triggers (phase / CRACK / momentum / drawdown / stop) keep authority for those terminal actions. Static guard test: `test_bridge_only_acts_on_partial_exit_action`.
  3. **Per-asset isolation.** `agents/exit_drl_agent.py` now carries a `_asset_modes: Dict[str, ExitDRLMode]` and `mode_for_asset(asset)` / `set_mode_for_asset(asset, mode)` / `should_act_on(asset)` accessors. BTC=EXIT_ONLY at startup; ETH+SOL=SHADOW. A bad regime on BTC cannot poison ETH/SOL.
  4. **Override audit stamp** — `risk/exit_drl_promotion_gate.py:record_override()` writes `force_promote_at` + reason + blockers-at-override to `data/exit_drl_promotion_state.json`. main.py init records this for every asset promoted at startup.
  5. **CLAUDE.md runtime-state row updated** to reflect partial promotion (BTC=EXIT_ONLY, ETH+SOL=SHADOW), and this P29 entry exists for future readers to find via `git blame` / `grep`.
- **Test coverage:** `tests/test_exit_drl_promotion_active.py` — 18 tests covering the 4 trip conditions, per-asset mode helpers, override-stamp persistence, and static guards on the bridge code + main.py promotion config. Combined with the prior 38 Exit-DRL tests: **56/56 passing.**
- **What we expect on first deploy:** BTC's `latest_exit_drl_action` gets stamped on `_paper_positions['BTC']` every tick. When System-3 says PARTIAL_EXIT (~26-31% of bars per validator), the existing exit_alpha TRIGGER 4 fires a 25% scale-out (already wired pre-promotion; the difference is now the trigger consults System 3 instead of System 1's TQC direction). Existing 4 triggers (phase / CRACK / momentum / drawdown) keep firing independently — System 3 only adds, never blocks.
- **What to watch:**
  - `data/exit_drl_kill_switch_state.jsonl` — last entry's `BTC.consecutive_losses` and `last_demote_reason`. If `last_demote_reason` is non-null after a deploy, BTC has been auto-demoted and you'll find why in the runtime log + `[EXIT_DRL_KILLSWITCH] BTC: DEMOTED to SHADOW` warning.
  - `data/exit_drl_outcome_ledger.jsonl` — `predicted_action_at_close` distribution for BTC vs ETH/SOL. ETH+SOL keeps showing the SHADOW counterfactual; BTC's predictions are now actually consulted.
  - `[EXIT_DRL_BRIDGE] BTC: System-3 PARTIAL_EXIT override` log lines — frequency tells you how often System 3 actually fires vs the rule-based triggers.
- **Rollback (per asset):** flip `_per_asset_modes["<ASSET>"] = ExitDRLMode.SHADOW` at [main.py:3275](main.py#L3275) and redeploy. The kill switch already does this automatically per asset on any of its 4 trip conditions; manual rollback is for when you want to demote despite no kill-switch trip.
- **Rollback (global):** set all 3 to `ExitDRLMode.SHADOW` in `_per_asset_modes`. Reverts to the SHADOW baseline that ran before promotion — System 3 still logs predictions but doesn't influence trades.
- **Phased-rollout policy (relaxed 2026-04-24):** prior version of P29 required single-asset rollout for the first 30 days. User overrode this on the basis that per-asset isolation is structurally sound and the kill switch is the actual safety mechanism. The relaxation only applies given (a) clean e2e diagnostic, (b) per-asset kill-switch state, (c) per-asset mode flag. If those three properties ever regress, single-asset rollout becomes mandatory again.
- **Mitigation pattern:** When the spec says "wait N days for live data" and you override that constraint, the override needs (a) a kill switch that's strictly stricter than the gate, (b) per-asset isolation so one asset's failure can't poison the others, (c) an auditable stamp with the blockers that were active at override time, (d) a CLAUDE.md entry quoting the user direction. The gate stays in place for future promotions — the override is *one-asset, one-direction* (PARTIAL_EXIT), not a global authority change.

### P22. [RETIRED 2026-04-24] Execution shadow mode + 3,160-line `_execute_intent` removed
- **Symptom 1 (shadow):** CLAUDE.md claimed "Execution shadow: ACTIVE". `_enable_execution_shadow = True` was set and a `_shadow_state_snapshot` dict was built every tick at what-was-then [main.py:12341-12356](main.py#L12341). But nothing consumed it — the cutover commit ef4060b (2026-04-18) had deleted the `run_shadow_execution()` invocation. `data/shadow_exec_comparison.jsonl` has had no writes since 2026-04-15. The "CRITICAL MISMATCH: AC2_RATE_LIMITED" log lines being attributed to a current AC-2 snapshot bug were actually pre-cutover log lines.
- **Symptom 2 (_execute_intent):** The cutover commit's message said the old 3,160-line `_execute_intent` was "dead code (can be removed in a follow-up cleanup)". Two live callers remained: `MAX_HOLD_TIMEOUT` inline exit at [main.py:5237](main.py#L5237) and `FLIP_BLOCKED` recovery at [main.py:12429](main.py#L12429). Parallel-maintained divergent implementations for the same asset-close transition.
- **Hidden risk if shadow had been re-enabled:** dict-level deep-copy (paper_positions, ac2_fill_ticks, etc.) is not enough. `execute_intent_v2` mutates ~10 live singletons per fill — `anti_churn.record_fill`, `thesis_budget_governor.record_fill`, `existence_fuse.record_pnl` + `on_trade_close`, `trade_attributor.record_entry/exit`, `confidence_scorer.record_outcome/signal`, `pnl_attribution.record_trade`, `strategy_aging.record_outcome`, `failure_memory.record_opportunity`, `account_sync.update_dry_run_pnl`, `strategic_coordinator.record_trade_completed`. Running the new path a second time against live components would double-record every one of those, corrupting rate limiters, budget windows, the 28-day existence fuse PnL, agent attribution, confidence calibration, and v521 adaptive weights.
- **Fix:** (a) Removed shadow snapshot-capture block + `_enable_execution_shadow` flag from main.py; CLAUDE.md runtime-state table updated to RETIRED. (b) Migrated both remaining `_execute_intent` callers to `execute_intent_v2(ctx, asset, intent, market_data, ...)` with a fresh `ExecutionContext.build_from_runner(self)`. (c) Deleted the 3,160-line `_execute_intent` body; left a 5-line breadcrumb comment pointing at execution_service.
- **Mitigation:** If dual-path validation is ever needed again, do NOT simulate it by deep-copying dicts and re-running the same function. Add a `shadow_mode: bool = False` kwarg to `execute_intent_v2` that short-circuits every `.record_*()`, `account_sync.update_*()`, `risk_manager.update_balance()`, `fn_save_paper_positions()`, `fn_persist_tranche_state()`, and `anti_churn.record_fill()` call before re-wiring the invocation.

### P9. [ARCHIVED 2026-04-22] agents/quant_agent.py moved to archive/legacy_agents/
- **Historical symptom:** Prior readers finding `agents/quant_agent.py` assumed it was the DECIDE signal source and tried to edit it — zero runtime effect.
- **Resolution:** File moved to `archive/legacy_agents/quant_agent.py`. Its two unique signals (CVD z-score cascade confirmation + predicted-funding bearish divergence) were ported into `agents/kraken_quant_agent.py` (LiquidationCascadeHunter and FundingDivergenceStrategy respectively). See commit 540167d.
- **Where quant DECIDE lives now:**
  - TA-based Best-of-N (mean_revert/momentum/volume_breakout/vrp/hold) → `data_mgmt/market_data_pipeline.py:1244` → `quant_direction`
  - 12 institutional stat-arb strategies → `agents/kraken_quant_agent.py` → `kq_direction`
  - Both are DECIDE authority in fusion, alongside DRL (TQC) when ACTIVE.

### P15. [FIXED 2026-04-24] v521 AdaptiveWeightManager feedback loop never closed
- **Symptom:** `signals/adaptive_weight_v521.py` (915 lines, self-labeled "CANONICAL v5.4.0") appeared DEAD in completeness_audit (0/5 hops). Deeper inspection showed it IS wired via `main.py → strategic_coordinator.pre_decision_check → v521.get_adjusted_weights → agent_signals["v6_adjusted_weights"] → integration_v36 fusion`. But the INPUT side (trade outcomes) was never fed.
- **Cause:** `strategic_coordinator.record_trade_completed()` defined but zero callers. Without trade data, `StrategyMetrics.sufficient_data=False`, `compute_weight()` returns 1.0 neutral (line 504-505). Every strategy got multiplier=1.0 forever → `v6_adjusted_weights == base_weights` → 915 lines of Sharpe/Calmar/WinRate math was permanent no-op.
- **Fix (commit db029e6):** Added `strategic_coordinator` to `ExecutionContext`; after full-exit `thesis_budget.record_fill()` (execution_service.py:2107), also call `ctx.strategic_coordinator.record_trade_completed(strategy, pnl, pnl_pct, duration_hours)`.
- **Mitigation:** When auditing claims like "v5.4.0 CANONICAL", verify BOTH the read side (who reads the outputs) AND the write side (who feeds the feedback signal). A half-wired feedback loop looks connected at a glance but produces neutral outputs.
- **Symptom:** `scripts/completeness_audit.py` flagged 27 `agent_signals.get(KEY)` calls with no matching writer. 23/27 false positives (dynamic dict-unpack like `for k,v in sig.items(): agent_signals[k]=v` bypasses regex), but 4 are real silent bugs.
- **Fixed in commit 1d72baf:**
  1. `quant_strategy` — 3 readers expected it; only `primary_strategy` exists. Max-pain strategy gate silently always-off.
  2. `_ood_score` — reader expected normalized [0,1]; OOD detector writes `_ood_distance`. Bridge added.
  3. `_vr_bounce_pct` — local var in `_diag_record()` never mirrored to `agent_signals`; reader in different method always got 0. V-reversal SHORT-override guard disabled.
  4. `cross_asset_divergence` — CHAOS NO_TRADE veto (>0.9 triggers safety halt) never populated. Derived from `cross_asset_correlation` as `1 - |corr|`.
- **Mitigation:** Run `python scripts/completeness_audit.py` monthly. Compare the `agent_signals_flow.dead_reads` list — verify each real miss with `grep -rn "agent_signals\[['\"]KEY['\"]\] ="` across whole codebase (not just main.py).

### P12. [FIXED 2026-04-24] 2-agent conflict score 0.7 force-promoted to HARD VETO 1.0
- **Symptom:** After DRL went ACTIVE on 2026-04-22, cloud produced zero fills for 48 hours. Shadow ledger full of `[PROD] HARD VETO: ALL_CONFLICT_FLAT` on every tick.
- **Cause:** `integration_v36.py:1679` promoted ANY `signal_conflict > 0.5` to `1.0` before calling the risk classifier, which treats `>= 1.0` as HARD VETO. But `constitution.py:441-444` explicitly says 2-agent conflict (score 0.7) should only reduce confidence via fusion — not veto — and only 3-agent conflict (score 0.9+) is NO_TRADE-worthy. DRL ACTIVE with +0.9 signals frequently disagreed with quant Best-of-N, generating score 0.7 every tick → auto-promoted to 1.0 → HARD VETO → no trades.
- **Fix:** Threshold aligned with constitution.py (commit 607ab10). Only `>= 0.9` (true 3-agent conflict) escalates.
- **Mitigation:** When adding a DECIDE agent, check cross-agent conflict handling — the existence of an additional DECIDE voter can trigger this kind of interaction bug.

### P16. [FIXED 2026-04-24] Dummy ENABLE_* flags — declared but never gated
- **Symptom:** `configs/sota_flags.py` declared `ENABLE_STRUCTURE_ANALYZER`, `ENABLE_ENHANCED_REGIME_NAVIGATOR`, `ENABLE_SOLDEX_MONITOR_SHADOW` all defaulting to `True`. Flipping any of them to `False` produced zero runtime effect — operators thought they had kill switches they did not have.
- **Cause:** Instantiation sites in main.py guarded only on `_AVAILABLE` imports (e.g. `if STRUCTURE_ANALYZER_AVAILABLE: self.structure_analyzer = ...`), never on the flag itself. `ENABLE_SOLDEX_MONITOR_SHADOW`'s only real consumer was `archive/shadow/shadow_observers.py` — off the live path since the 2026-04-15 soldex SHADOW→ACTIVE promotion.
- **Fix:** Wrapped StructureBreakAnalyzer ([main.py:3817-3838](main.py#L3817-L3838)) and EnhancedMarketRegimeNavigator ([main.py:3311-3327](main.py#L3311-L3327)) with real `getattr(flags, ENABLE_X, True)` checks. Dropped `ENABLE_SOLDEX_MONITOR_SHADOW` entirely (dead-flag cleanup pattern per 2026-04-15 `ENABLE_PASSIVE_AGGRESSIVE_SHADOW` precedent).
- **Mitigation:** New `ENABLE_*` flag must include (a) real runtime check at every instantiation/call site, (b) `DISABLED` log message branch so operators can verify the kill switch engaged. If the flag has no live consumer, delete it; do not leave "future-proofing" stubs.

### P17. [CLEANUP 2026-04-24] canonical_config.py single-source-of-truth drift
- **Symptom:** `configs/canonical_config.py` self-labeled "Single Source of Truth" but 13 of its constants (`CORRELATION_COLLAPSE`, `REGIME_LEVERAGE_MAP`, `MASTER_TICK_SECONDS`, `EXECUTION_LOOP_MS`, `DMS_TIMEOUT_SECONDS`, `SCALE_OUT_PCT`, `SCALE_OUT_MIN_PROFIT_BPS`, `SCALE_OUT_MIN_BARS`, `RUNNER_INITIAL_TRAIL_PCT`, `RUNNER_TIGHT_TRAIL_PCT`, `STOP_ATR_MULTIPLIER`, `STOP_MIN_PCT`, `MAX_HOLDING_HOURS`) had zero importers. Authoritative values lived in `configs/high_risk_mode.py` dict literals, `defense/constitution.py` class attributes, or local module constants — while canonical_config's copies silently rotted.
- **Fix:** Removed the 13 dead constants. Left breadcrumb comments pointing to where the live values actually live (e.g. `SCALE_OUT_PCT` → `configs/high_risk_mode.py`).
- **Mitigation:** Before tweaking any config value, run `grep -rn "from configs.canonical_config import"` to confirm the constant is actually on the hot path. If the module is supposed to be authoritative, new additions need at least one real importer — otherwise they are documentation masquerading as config.

### P18. [FIXED 2026-04-24] Two dead-reads missed in P14 (market_data-vs-agent_signals + pre-write read)
- **Symptom:** After P14 fixed 4 dead-reads, a deeper trace of the remaining 25 audit false-positives turned up **2 more real silent bugs**:
  1. `cascade_phase` — writer at `main.py:5459` writes to **market_data**, reader at `main.py:12099` (PartialConsensus `DisableConditions`) reads from **agent_signals`. Namespace mismatch → reader always got `'NONE'` → cascade-driven disable of partial-consensus entries never fired. Dormant today because `ENABLE_PARTIAL_CONSENSUS_ENTRY=False`, but activates the moment PC flips on.
  2. `quant_strategy_id` — reader at `main.py:7155` (StrategyAllocator) reads `agent_signals.get('quant_strategy_id', 'momentum')` BEFORE intent is built. The key is set as `intent.quant_strategy_id` at `integration_v36.py:1167`, never mirrored to agent_signals. Allocator always looked up 'momentum' weight regardless of actual selected strategy. Impact limited because `allocator_authority='SHADOW'` (advisory), but still wrong.
- **Fix:** Both readers bridged. `cascade_phase` falls back to `market_data.get('cascade_phase')`; `quant_strategy_id` falls back to `agent_signals.get('primary_strategy')`.
- **Mitigation:** The audit's dead-read list is worth tracing by hand even when most are false positives — among 25 "definitely-false-positive" keys, 2 turned out real. Dict-literal writes and dict-unpack writes are invisible to the audit regex, but so are market_data→agent_signals namespace mismatches. Whenever a `.get()` reader lives far from the writer, spot-check the namespace explicitly.

### P19. [FIXED 2026-04-24] BEST_OF_N_HOLD short-circuit demoted DRL to effective ADVISE
- **Symptom:** DRL promoted to ACTIVE (DECIDE authority) 2026-04-22 per CLAUDE.md P10, but live attribution JSONL showed `"drl": authority="ADVISE"` and DRL signals had zero impact on trades. On 2026-04-24 post-deploy: DRL=-0.95 to -0.96 on BTC/ETH (strong bearish), SOL=-0.62, but intent.direction=+0.00 across all 3 assets. No trades fired despite clear signal.
- **Cause:** `integration_v36.py:769-803` `_maybe_apply_pre_alpha_hold()` — when TA-based Best-of-N picked `hold` strategy (score < 0.05 on all 4 TA strategies in quiet/consolidation regimes), the code hard-set `intent.direction = 0.0` and returned True, short-circuiting BEFORE fusion. DRL's DECIDE-authority signal was never consumed. Comment note from 2026-04-08 explained it as "prevents forced entries that produced 14.3% SOL win rate and -$60 SHORT losses" — valid at the time (DRL was SHADOW) but structurally incompatible with DRL=DECIDE.
- **Secondary cosmetic bug:** `main.py:8434` `_ATTR_AUTHORITY["drl"] = "ADVISE"` hardcoded regardless of runtime promotion state. Authority label in JSONL didn't follow the fusion-layer upgrade.
- **Fix:** Added DRL ACTIVE punch-through in `_maybe_apply_pre_alpha_hold`: when `drl_authority_level == "ACTIVE"` AND `|drl_direction| >= 0.5` AND `drl_confidence >= 0.3`, skip the hold short-circuit and let fusion+alpha gate process the intent with DRL's direction. Thresholds are conservative enough that routine DRL noise still respects hold. Also: mirrored `drl_authority_level` into `agent_signals` at the TQC inference site; fixed `_ATTR_AUTHORITY["drl"]` to read runtime authority.
- **Mitigation:** When promoting an agent from ADVISE→DECIDE, audit ALL short-circuits that set `intent.direction = 0` before fusion runs. The authority matrix upgrade doesn't backfill into pre-fusion guards; each guard needs explicit review.

### P20. [FIXED 2026-04-24] Alpha gate's effective_alpha_direction ignored DRL when quant abstains
- **Symptom:** 30-day live run produced 1 fill, 241 gate rejections. Top blockers in shadow_ledger were (a) `ALL_CONFLICT_FLAT` (103 occurrences — fixed by P12) and (b) `Alpha gate: QUIET_ACCUMULATION + direction 0.00 < 0.XX hard filter` + `Alpha 0bps < threshold N bps` (~65 combined). After P19 deployed 2026-04-24, BEST_OF_N_HOLD_OVERRIDE fired correctly on all 3 assets with DRL=-0.66 to -0.96, but intent still landed at alpha gate with `direction=+0.00` and was rejected for `alpha=0bps`.
- **Cause:** `integration_v36.py:1237-1274` runs alpha gate check BEFORE `fusion_engine.fuse()`. Alpha gate uses `_alpha_input_direction = agent_signals.get("effective_alpha_direction", agent_signals.get("quant_direction", 0.0))`. `effective_alpha_direction` is computed by `main.py:4952 _compute_effective_alpha_direction()` which early-exits at line 4973 with direction=`_quant_dir` (=0 when quant=hold) without considering DRL. DRL's DECIDE-authority signal was structurally invisible to alpha gate.
- **Fix:** Added DRL substitution in `_compute_effective_alpha_direction`: when quant abstains (|quant_dir|<0.03) AND DRL is ACTIVE AND |drl_dir|>=0.5 AND drl_conf>=0.3, use DRL's direction as the effective alpha input. Alpha gate then evaluates alpha against DRL's direction, and QUIET_ACCUMULATION hard filter sees |direction|~0.9 instead of 0. Both blockers resolved by one change.
- **Mitigation:** Alpha gate is a pre-fusion short-circuit. Any time an agent is promoted to DECIDE, check what `effective_alpha_direction` feeds into alpha gate — if the agent isn't a contributor there, its DECIDE authority is nullified in quiet/consolidation regimes.

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

### P54. [FIXED 2026-04-25] SHORT_CONTROL `_SC_PROTECTED_VETOES` substring keys mismatched real veto_reason format
- **Symptom:** Decision-path trace surfaced `main.py:9979 _SC_PROTECTED_VETOES` — a defense-in-depth set that prevents `defense/short_control.py:override_veto=True` from clearing system-level vetoes. Audit found 3 of 8 keys would NEVER match a real `intent.veto_reason`:
  - `EXISTENCE_FUSE` — real string is `"[STRATEGY_SUSPENDED] ..."` (main.py:10601)
  - `CIRCUIT_BREAKER` — never written to veto_reason (only `risk_governor.py:264` sets a `VetoReason` enum, distinct path)
  - `DEAD_MAN` — dead-man switch calls `_emergency_flatten()` directly, doesn't go through veto_reason
- **Why this isn't an active runtime bug today:** `defense/short_control.py:168-171` source-level scope only flips `override_veto=True` when `"SHORT BLOCK" in veto_reason` OR `"SHORT FILTER" in veto_reason`. So the 3 broken substring keys don't actually let anything bypass — the override never proposes itself for those reasons in the first place.
- **Why it's still a bug:** the protected list is a code-review hazard. A reader sees "EXISTENCE_FUSE is protected" and trusts the defense-in-depth. If a future refactor widens `_is_short_veto` (e.g. to accept any veto containing `"SHORT"`), Layer 1 would propose overrides for `"[STRATEGY_SUSPENDED] ..."` cases too — and Layer 2 substring match would silently FAIL because `"EXISTENCE_FUSE"` doesn't match `"[STRATEGY_SUSPENDED]"`. Same shape as P15 half-wired feedback loop.
- **Fix:**
  1. **`main.py:9979`** — aligned 8 keys to real veto_reason format. Replaced `"EXISTENCE_FUSE"` → `"STRATEGY_SUSPENDED"`. Removed dead `"CIRCUIT_BREAKER"` + `"DEAD_MAN"`. Added `"THESIS_BUDGET"` + `"TRADE_GATE"` + `"WEEKEND"` + `"REGIME_POWER_NO_TRADE"` (all real strings observed in 30-day shadow ledger). Added cross-reference comments pointing to where each is set.
  2. **`tests/test_short_control_ultra.py`** — added 3 new test classes (14 tests):
     - `TestP54SourceLevelScope` (12 parametrized) — Layer 1 must NOT propose override for any system-level veto string. Pins down `defense/short_control.py:168-171` scope.
     - `TestP54ProtectedVetosSubstring` — bidirectional: every PROTECTED key matches at least one real veto, AND every real veto has a matching PROTECTED key.
     - `test_dead_substrings_no_longer_present` — explicit guard that the 3 broken keys don't regress.
- **Tests:** 114/114 (48 short_control + 66 prior regression). Layer-1 + Layer-2 now both have explicit coverage; future refactors will fail loud.
- **Mitigation pattern:** When adding a defense-in-depth substring match against another module's output, write a roundtrip test in the same commit that proves the substring actually appears in real output. Don't trust string literals — the real veto_reason format drifts when modules get renamed or the format changes (P15 family).

### P53. [CLEANUP 2026-04-25] Stale tests + ETH fold regression + dead flags + path portability
- **Symptom:** Cleanup pass on shallow-audited areas (`tests/`, `scripts/`, `training/model_alpha/`, `configs/sota_flags.py`, `.env.example`). Six findings, three with real-bug shape.
- **Real bugs:**
  1. **`tests/test_quant_agent.py` + `tests/test_modular_integration.py`** — both imported `agents.quant_agent` which was archived in P9 (commit 540167d). Both files failed at pytest collection (`ModuleNotFoundError: agents.quant_agent`). DELETED — coverage already provided by `tests/test_kraken_quant_agent.py` etc. (the ported strategies live there now).
  2. **`training/model_alpha/train_sequence_alpha_v1.py:35`** — `BEST_FOLDS = {..., "ETH": "fold_1", ...}` was the same P4 stale-fold regression that fired in `drl/ensemble.py` 2026-04-24 but in a SECOND file. If the sequence-alpha training was rerun, scalers would be loaded from `models/retrained/ETH/fold_1/` (train_rows=0) — silent train-test mismatch since runtime `runtime_obs_builder.py` always loads fold_3 for ETH. Aligned to fold_3.
  3. **`scripts/_gen_report.py`** — orphan one-shot generator (writes `trade_report.py`). Zero callers anywhere. DELETED.
- **Cleanup:**
  - `configs/sota_flags.py:57-64` — removed dead docstring section listing P2/P3 flags that were already deleted 2026-04-15. Breadcrumb comment retained.
  - `.env.example:120-128` — removed dead `HMATS_ENABLE_DRL` / `HMATS_ENABLE_SENTIMENT` / `HMATS_ENABLE_FALSE_BREAKOUT` / `HMATS_ENABLE_SIGNAL_QUALITY` block. Verified 0 readers across non-archive code.
  - `scripts/analyze_shadow_ledger.py`, `scripts/trade_report.py`, `scripts/shadow_ledger_report.py` — replaced 4 hardcoded `c:/Users/melod/...` paths with `Path(__file__).resolve().parents[1]` / repo-relative pattern. Operator scripts now portable.
- **Verified safe to retain (no action):**
  - `configs/default.json` + `configs/sota_config.json` — both marked `_DEPRECATED` and only referenced by `config_resolver.py:_check_stale_configs()` which is the safety net warning anyone tries to use them. Removing the files would also require removing the safety-net code.
  - `configs/live_phase1.json:23 max_leverage=2.0` — intentional Phase-1 conservative profile (described in `_description` field), not stale. Loaded by `scripts/launch_live.py` via `phase1` alias.
- **Audit ratio:** 4 real cleanup wins from ~12 findings (~1:3 — higher than P51's 1:8 because the cleanup-pass surface area was different, mostly orphans rather than safety bugs).
- **Tests:** 66/66 regression suite green. Stale-tests removal reduces collection errors from 1 → 0 on `pytest tests/`.
- **Mitigation:** When a module is archived (P9 pattern), grep `git grep "from <archived_module>"` AND `git grep "<archived_module> import"` AND check for test files importing it. The 540167d commit moved quant_agent.py to archive but missed the two tests. Cleanup-audit rounds catch these but only if the audit specifically looks at test collection errors. Now: pytest will fail loud on a stale test rather than skip it.

### P52. [FIXED 2026-04-25] Weekend confidence default 0.50 → 0.30 + diag forwarding
- **Symptom:** Post-P46 deploy, runtime forensics showed weekend rejections still firing at "min 50%" even though `live_high_risk.json` had `min_confidence_weekend: 0.30`. Container's JSON verified correct via SSH; post-P45 ledger entries showed `weekend_cfg_present: True` and `weekend_mult_normal: 1.0` (config IS loading); but rejection text reads "min 50%" — which is exactly `cls.WEEKEND_MIN_CONFIDENCE` class default. Two BTC/ETH rejections at 05:24 UTC: drl_conf=0.444/0.355, mode=NORMAL, regime=WEAK_CONSOLIDATION.
- **Root cause:** ambiguous — could be a config read path that bypasses the JSON value (callsite reads class default directly), OR a stale path that uses the cls.* fallback. Without a runtime field showing exactly what threshold the gate saw at rejection time, we can't tell. **Two-part fix:**
  1. **Defensive (immediate):** lower class default `WEEKEND_MIN_CONFIDENCE 0.50 → 0.30` in `liquidity/weekend_manager.py`. Closes the gap regardless of root cause; aligns with `live_high_risk.json` and the P19/P20/P46 substitution-chain threshold (drl_conf>=0.3). Same pattern as P42 lowering MAX_LEVERAGE class default.
  2. **Diagnostic (root-cause):** forward 3 new fields to the shadow ledger `gate_details` dict at `main.py:13057`: `weekend_min_confidence` (the `min_confidence_weekend` value the gate saw), `weekend_cfg_keys` (the actual key list of `_wk_cfg` at the rejection site), `weekend_system_mode`. Plus extend `scripts/gate_rejection_analysis.py` with two new pareto sections: "By min_confidence_weekend seen at rejection" and "Whether min_confidence_weekend is in the loaded config dict". After 24-48h of fresh ledger entries, the analyzer will reveal whether JSON value flows in (`min_conf=0.3`) or class default fallback is in effect (`min_conf=NOT_SET`).
- **What to verify after deploy:** new ledger entries show `weekend_min_confidence: 0.3` (JSON wired) or `weekend_min_confidence: null` + `weekend_cfg_keys: [..., "min_confidence_weekend", ...]` (key present but reader broken). Either way, the next forensic run nails the root cause.
- **Tests:** 66/66 regression suite green (test_black_swan_hold + test_strategy_selection + test_alpha_gate + test_drl_authority_punchthrough).

### P51. [FIXED 2026-04-25] TIER 2 cleanup — 3 real datetime/observability fixes; rest were audit noise
- **Symptom:** Started TIER 2 batch (audit's "silent-veto-without-logging" + datetime/threading category, ~25 sites). After triage, **most "silent veto" findings were false positives** — query-style returns `(bool, reason_string)` whose callers log, NOT actual veto sites. Adding `logger.warning()` to every False return would spam logs.
- **Real fixes (3 sites):**
  1. **`risk/gambler_entry_exit.py:65, 73, 81`** — Three `except ImportError: pass`/`return False/None` paths gave operator no visibility into whether gambler mode was OFF by config or DEAD by missing import. Added `logger.warning()` per site.
  2. **`risk/cascade_exhaustion_governor.py:558, 563`** — `datetime.fromisoformat()` deserializes to aware if persisted ISO carries `+00:00`; runtime uses naive `datetime.now()`. Same shape as P39/P40. Local `_strip_tz()` helper now normalizes loaded timestamps on the way in.
  3. **`infra/event_bus.py:126`** — `timestamp: datetime = field(default_factory=datetime.utcnow)` was naive. Changed to `lambda: datetime.now(timezone.utc)` to match the codebase post-P39/P40/P50.
- **Audit false positives (saved triage time):**
  - `risk/stop_loss_authority.py:410, 414`, `risk/correlation_position_sizer.py:362, 372`, `risk/short_position_controller.py:298, 379, 505`, `risk/auto_recovery_gate.py:163-197` — all query-style `(bool, reason)` returns; not vetoes.
  - `risk/regime_transition_buffer.py` — no `fromisoformat()` in file; all `datetime.now()` calls pair internally; no naive/aware mismatch.
  - `analytics/failure_memory.py:206` — theoretical race; all callers single-threaded async; defer lock until evidence of thread exposure.
- **What this means for the audit-cycle question:** the trend is clear. Each round of comprehensive audit produces fewer real bugs. Audit-finding-to-real-bug ratio: P22 = 1:4, P47 = 1:6, P50 = 1:8, **P51 = 1:8 again**. We're approaching the point where additional audits cost more than the bugs they catch. **Recommend the runtime forensics tools (P41-P45 + the static detector P48) become the primary discovery loop going forward** — focused audits triggered by surprise rejection patterns, not calendar.
- **Tests:** 245/245 safety-critical regression tests pass.

### P50. [FIXED 2026-04-25] Comprehensive codebase audit follow-up — 8 real bugs from TIER 1
- **Symptom:** After dispatching 6 parallel audit agents to comprehensively cover ~250 files (the directories not yet deeply audited: risk/, signals/, core/, execution/, agents/, analytics/+market/+orchestration/+remaining), then a follow-up pass with main.py deep audit + market_data_pipeline + Multi-DECIDE fusion review + cross-file invariants + configs sweep, ~95 findings surfaced. Many were audit false positives (matrix count miscount, profit_calibration's `or {}` already correctly nested, sentiment_gate's quant_dir=0 guard already present). After triage, **8 real TIER-1 silent-failure bugs** in the same shape as P15/P47/P48.
- **Bugs fixed:**
  1. **`risk/correlation_realtime_controller.py:418-419, 431-432`** — Three `dict.get(asset1, asset2)` sites with the P47-Bug-2 shape. `asset2` was being passed as the default value (a string), not a second key. Live correlation lookups always returned the string and the surrounding try/except: pass swallowed the downstream TypeError. Fixed to `(asset1, asset2)` tuple key.
  2. **`defense/human_override.py:84`** — `created_at` default factory was `datetime.utcnow` (naive); downstream comparisons at lines 104/113/430 use `timezone.utc` aware. P39/P40 same shape — restart with persisted state would TypeError silently. Changed to `lambda: datetime.now(timezone.utc)`.
  3. **`core/authority_chain.py:366`** — Third `trade_gate.check()` call site (CLAUDE.md non-negotiable rule #6) was missing the P47/P48 plumbing. Added `dvol_zscore`, `dvol_current`, `price_direction` kwargs from ctx so the gate receives the same input shape as the other two sites. AuthorityChain is currently orphan-instantiated (main.py:2698 never calls .evaluate()) so this is defensive — when it's wired in, the inputs match.
  4. **`infra/kraken_derivatives_client.py:102`** — `KRAKEN_DERIVS_MAX_LEVERAGE` env var default was `"5.0"` while spot `MAX_LEVERAGE = 3.0` everywhere else. Cross-file invariant FAIL — derivatives client could silently exceed spot cap. Lowered default to `"3.0"`. Operator can still override via env var.
  5. **`execution/dead_man_switch.py:71, 169-170, 191-192`** — `self._active` was read by `DeadManSwitchMonitor` background thread + written by `start()`/`stop()` on main thread without lock. Added `self._lock = threading.RLock()`; guarded reads in `is_active` and writes in `stop()`. P39 same shape (Discord lock-across-sleep family).
  6. **`core/feedback_loop.py:460-466`** — Subscriber callback errors logged at WARNING and silently continued — same shape as P15 (record_trade_completed AttributeError swallowed). Now tracks consecutive failures per callback (by `__qualname__`); after 5 in a row, escalates to ERROR with explicit guidance: "Likely AttributeError from method-name mismatch (P15-shape). Check the callback's target type for the expected method." This makes silent broken-feedback-loops loud.
  7. **`execution/derivatives_executor.py:539-541`** — `nav = self._nav_callback() or 0.0` would propagate exceptions from the callback. If callback raised (dependency unwired, network glitch), the entire risk-cap math block was skipped and the position was either approved by an unrelated path or fail-opened. Now wrapped in try/except → returns `nav_callback_error:<type>` reject, fail-closed. Plus explicit `float()` coercion to avoid string-NAV silent failure.
  8. **`core/runtime_spine.py:351`** — File header marks DEPRECATED but `RuntimeSpine.__init__` was fully functional. Per audit, hardcodes 10% DD halt that conflicts with ULTRA profile (35%). Added `os.environ.get("HMATS_ALLOW_RUNTIME_SPINE")` gate at instantiation — raises RuntimeError unless explicitly enabled. `core/canonical_imports.py:186` still imports the TYPES (which is fine; gate fires only on instantiation).
- **Audit false positives identified (kept for reference):**
  - **AUTHORITY_MATRIX_NORMAL count** — audit said 26, actual is 25. Existing test correct, no change needed.
  - **`profit_calibration_service.py:115-119` dict.get chain** — `(by_side.get("long", {}) or {}).get(...)` is correctly nested. The `or {}` IS outside the inner .get(), so None is converted to {} before the next access. Audit miscounted parens.
  - **`sentiment_gate.py:77-83` Iron Law violation** — code has explicit `if quant_sign == 0.0: return neutral` guard at lines 80-81. Audit missed this guard.
- **Triaged for separate sessions:**
  - **`market_data_pipeline.py:1873-1879` GMM cache freeze** — root cause of P41's 110 STALE_DATA rejections with identical alpha=66.36. Real bug but requires careful refactor of cache invalidation logic; defer to a focused session.
  - **`risk/cascade_exhaustion_governor.py`, `risk/regime_transition_buffer.py`** datetime naive/aware — same P39/P40 pattern, mechanical sweep when next visiting those files.
  - TIER 2/TIER 3 (~25 silent-veto-without-logging in risk/* files, calibration tweaks) — defer to weekly sweep driven by `gate_rejection_analysis.py` data.
- **Tests:** All 282 safety-critical regression tests still pass (4 pre-existing failures unrelated to this commit — missing `defense.production_reliability_patches` module).
- **Static detector pattern reinforced:** P22-P50 sequence shows the static silent-failure detector (`scripts/silent_failure_audit.py`, P48) doesn't catch every semantic bug — but combined with focused parallel-agent audits, the surface area shrinks each round. **Estimated ratio of audit findings → real bugs is now ~1 in 8** (was ~1 in 6 in P47, ~1 in 4 in P22). Diminishing returns curve confirms the codebase is approaching a clean state for the silent-failure family.

### P49. [VERIFIED 2026-04-25] Authority matrix count — confirmed 25 agents (audit miscount)
- **Symptom:** Comprehensive audit agent reported `AUTHORITY_MATRIX_NORMAL` had 26 entries, claiming `tests/test_authority_fusion.py:80` would FAIL on next run. Verified by direct count.
- **Reality:** Live matrix has exactly 25 entries (audit agent included a duplicate). Test passes. Updated test docstring to clarify soldex (added 2026-04-15) replaced an earlier slot, keeping count at 25. Updated CLAUDE.md authority matrix paragraph with same clarification.
- **Lesson:** Audit agent findings need verification before blind action. Reading the actual code is faster than triaging false-positive fixes.

### P48. [FIXED 2026-04-25] p0_safety_integrator deep audit — 5 silent-failure bugs + static detector
- **Symptom:** Built `scripts/silent_failure_audit.py` to programmatically detect the SHAPE of P15/P47 bugs across the repo (3 patterns: silent try/except method calls, `dict.get(a,b)` shape, orphan ENABLE_ flags). Static detector found ZERO new actionable bugs from its top hits — confirming P22-P47 cleaned the surface area substantially. **But a focused dive on `defense/p0_safety_integrator.py` (one of three trade_gate.check() sites per non-negotiable rule #6) found 5 real bugs the static detector couldn't see** — they're semantic, not pattern-based.
- **Bug 1 — Trade gate exception → silent allow (fail-OPEN violation)**: `defense/p0_safety_integrator.py:544-545` had `try: gate_result = self.trade_gate.check(...) except Exception as e: logger.error(...)`. When the gate raises (data race, numpy crash, import error), code logs and falls through with `result.allow_trade=True` (default at line 154). **A gate crash silently became a PASS.** This violates the fail-closed contract for safety gates. **Fix**: wrap exception → set `allow_trade=False`, populate reason, return immediately.
- **Bug 2 — P47 fix incomplete: data-health snapshot only wired to `self.trade_gate`, not `p0_integrator.trade_gate`**: P47 plumbed snapshot at `main.py:5922` into `self.trade_gate`. But `p0_integrator` has its OWN trade_gate at `self.p0_integrator.trade_gate` — that's a separate instance per CLAUDE.md non-negotiable rule #6 (three call sites). **P47's data-health Gate 0 was only fixed for one of three call sites.** **Fix**: added `self.p0_integrator.trade_gate.set_data_health_snapshot(_dh_snap)` immediately after the P47 line.
- **Bug 3 — `gate_result.details` not propagated to `result.details`**: When p0_integrator's trade_gate rejects, `result.reason` carries only the reason string, but `result.details` stays empty. **Shadow ledger forensics (P41-P45) lose all gate-internal state for p0-routed rejections** (DVOL z-score, freshness sources, edge bps, etc.). **Fix**: `result.details.update(gate_result.details or {})` on each REJECT/EXIT_ONLY/CLIP path.
- **Bug 4 — `gate_result.decision.value` (int) vs `.name` (string)**: `result.gate_mode = gate_result.decision.value` returns the auto() integer (1, 2, 3...) not the symbolic name. Shadow ledger field `gate_mode` was unreadable JSON like `{"gate_mode": 3}` instead of `"REJECT"`. **Fix**: changed to `.name`.
- **Bug 5 — `CRITICAL_SOURCES` mismatch makes stale-data kill switch dead**: `defense/execution_guards.py:61` had `CRITICAL_SOURCES = ['orderbook', 'trades', 'position']`. But the only production caller (`p0_safety_integrator.py:357`) only updates timestamps for `'kraken_ws'` and `'kraken_rest'` — names didn't match. `can_execute()` iterates `data_timestamps.items()` (which only contains `kraken_ws`/`kraken_rest` after first registration) and checks `if source in CRITICAL_SOURCES` — never True. **Net: critical_stale path NEVER fires; stale-data kill switch in p0_integrator effectively dead.** **Fix**: extended `CRITICAL_SOURCES` to `['kraken_ws', 'kraken_rest', 'orderbook', 'trades', 'position']` so the actually-tracked sources are also CRITICAL.
- **Skipped (not a bug):** the audit's "Bug 3" (`daily_pnl += vs =` inconsistency in `record_fill`). `record_fill()` is never called from main.py per grep, so the `+=` path is dormant. Defer.
- **Static detector — `scripts/silent_failure_audit.py`**: scans 244+ files in live dirs for 3 P15/P47-shape patterns. After P22-P47 cleanup, current top hits all verify clean. Becomes a CI tool: re-run after every refactor to catch regressions of the silent-failure pattern. Pattern 1 found 1134 method-call sites in silent try/except (290 unique pairs after filtering noise — operator triages); Pattern 2 found 39 `dict.get(a,b)` sites (all intentional aliasing fallbacks); Pattern 3 found 24 ENABLE_* flags (all have ≥1 reader, all readers are real instantiation gates).
- **What changes operationally after P48 deploy:**
  - p0_integrator's data health Gate 0 will start firing when CRITICAL/EXIT_ONLY data states actually occur (was dead).
  - p0_integrator's stale-data kill switch will start firing when `kraken_ws` / `kraken_rest` go stale (was dead).
  - Shadow ledger entries from p0_integrator-routed rejections will carry `gate_mode` as readable string (e.g. `"REJECT"`) and full `gate_details` for forensic analysis.
  - A trade_gate crash will now block trades instead of silently allowing them.
- **Mitigation pattern**: P47 + P48 establish a reusable workflow: (1) static detector flags silent-failure shapes; (2) operator does a focused semantic audit on the highest-leverage file; (3) each finding gets a 1-3 line surgical fix. The static detector is fast (seconds) and catches the SHAPE of regressions; the human audit catches the semantic bugs (input plumbing, source-name mismatches, fail-open violations) that pattern matching misses.

### P47. [FIXED 2026-04-25] Four silent-failure bugs — half-wired modules and dict.get misuse
- **Symptom:** Focused audit of high-leverage files (`risk/risk_manager.py`, `defense/trade_gate.py`, `signals/no_trade_triggers.py`, `signals/adaptive_weight_v521.py`, `execution/loop_controller.py`, `data_mgmt/market_data_pipeline.py`) surfaced 4 silent-failure bugs that exemplify the audit-cycle pain: code that "looks wired" in static review but isn't doing anything at runtime.
- **Bug 1 — P15 silently broken since 2026-04-24**: `core/execution_service.py:2171` calls `ctx.strategic_coordinator.record_trade_completed(...)` but `orchestration/strategic_coordinator.py:527` defined the method as `record_trade_result`. The `AttributeError` was caught by the surrounding try/except at line 2177 and logged at debug level only. **Result**: the 915-line `signals/adaptive_weight_v521.py` AdaptiveWeightManager has been receiving zero trade records since deploy — `compute_weight()` returns 1.0 for every strategy forever (the same bug P15 was supposed to fix). **Fix**: renamed `record_trade_result` → `record_trade_completed` at the definition site (1-line change).
- **Bug 2 — `risk/risk_manager.py:506` dict.get signature misuse**: `matrix.get(base_a, base_b)` — `dict.get(key, default)` takes 2 args; `base_b` was being passed as the *default value*, not as a second key. Live correlation matrix lookup ALWAYS returned `base_b` (a string like "ETH"), and the surrounding `try/except: pass` at line 512 swallowed the downstream TypeError. **Net**: live correlation has been silently broken since this code was written. **Fix**: changed to `matrix.get((base_a, base_b))` (tuple-keyed) plus an explicit `float()` coercion guard.
- **Bug 3 — DVOL gate (Gate 2 in `defense/trade_gate.py:627-637`) has been dead code**: gate condition `dvol_zscore >= 5.0 → EMERGENCY_FLAT`, but `main.py:10492` (the only `trade_gate.check()` call site) didn't pass `dvol_zscore` or `dvol_current`. Defaults in `TradeProposal` (lines 84-85) are `0.0` → never crosses 5.0 threshold → gate never fires. Same shape as P15 half-wiring: code present, input absent. **Fix**: added `dvol_zscore=...` and `dvol_current=...` to the `check()` kwargs at `main.py:10492`, sourced from `market_data`.
- **Bug 4 — Data Health snapshot never injected (Gate 0 in `trade_gate.py:590-618`)**: `trade_gate.set_data_health_snapshot()` is defined and called from many test files, but `main.py` never invoked it on the live tick. Gate 0 always saw `_data_health_snapshot=None` → checks at lines 591-618 were dead code. Even if `DataHealthMonitor` reported CRITICAL, gate allowed full new entries. **Fix**: added `self.trade_gate.set_data_health_snapshot(_dh_snap)` immediately after the existing snapshot fetch at `main.py:5922`.
- **Investigation skipped (legitimate but not bugs):** `signals/no_trade_triggers.py` is confirmed orphaned with a STALE_DATA threshold drift (2.0s vs constitution's 60.0s); recommend deletion in a future cleanup pass — not a current runtime bug since nothing imports it from the live path. `agent_signals["v6_adjusted_weights"]` IS partially consumed at `integration/integration_v36.py:2042-2043` (the `quant_agent` key) — the audit's "zero readers" claim was wrong, integration is half-wired but Bug 1's fix is sufficient to validate the loop is alive. `execution/loop_controller.py` tick_counter race remains dormant (CPython GIL + atomic int assignment).
- **Why these matter for the "1 fill in 30 days" question**: P15's broken claim is the most damning. The whole point of v5.4.0 AdaptiveWeightManager is to learn from realized trades and adjust strategy weights. With the input side silently broken, that learning loop has been off since promotion. **No amount of P19/P20/P42/P46 punch-through tuning fixes a learning system that never learns.** Fix 1 alone may be the single highest-impact change in this session.
- **Mitigation pattern**: Every silent `try/except` block around an attribute call is a candidate for this bug. Recommend a future audit pass that grep's for `try: ctx.X.method(...)\nexcept Exception` patterns and verifies each method actually exists on `X`. Also: `dict.get(a, b)` where both `a` and `b` look like keys — search for this idiom across the codebase.

### P46. [FIXED 2026-04-25] Weekend gate confidence — DRL substitution + min lowered
- **Symptom:** Post-P42 deploy verification on 2026-04-25 showed the alpha gate now passes (alpha_est 68-79 bps vs threshold 35-39 = 1.9-2.0× margin), but rejections moved to a SECOND weekend gate: `[WEEKEND] Weekend confidence 33% < min 50%`. DRL was at full conviction (`drl_action=-0.93`, `drl_confidence=0.44`) but the gate only consults `intent.quant_confidence`, which the HOLD-strategy path clamps to [0.40, 0.70] (`market_data_pipeline.py:1235-1236`) and downstream multipliers (`main.py:6203/6407/8798/9429`) damp to 0.33-0.42.
- **Root cause:** Same shape as P20 (alpha-direction substitution) but at the confidence layer. When DRL ACTIVE punches through HOLD via P19, intent direction comes from DRL but `intent.quant_confidence` stays the HOLD-clamp result. Weekend gate sees 0.33-0.42, rejects. DRL's confidence is invisible to the gate.
- **Fix (two layers, same as P42's structure):**
  1. **Code substitution** — new `HMATSProductionRunner._compute_effective_weekend_confidence(intent, agent_signals, asset)` helper (sibling of `_compute_effective_alpha_direction`). Returns `max(quant_conf, drl_conf)` when DRL is ACTIVE AND `|drl_dir|>=0.5` AND `drl_conf>=0.3` AND `drl_conf > quant_conf`. Logs `[WEEKEND_CONF_DRL_SUB]` when substitution fires. Wired at `main.py:10583` (replaced inline `confidence=getattr(intent, 'quant_confidence', 0.5)`).
  2. **Config** — `live_high_risk.json:152` `min_confidence_weekend` lowered `0.50 → 0.30` to match the threshold P19/P20/P46 substitution chain uses (drl_conf>=0.3). Without this, even with the substitution in place, mid-conviction DRL signals (0.30-0.49 conf) would still be blocked.
- **Tests:** 6 new tests in `tests/test_drl_authority_punchthrough.py:TestP46WeekendDrlConfSubstitution`: DISABLED returns quant_conf; ACTIVE+strong+higher-drl substitutes; weak direction doesn't substitute; low conf doesn't substitute; drl_conf < quant_conf doesn't substitute (substitution should never LOWER confidence); ADVISE authority doesn't substitute. Direct test of the helper, not source-level guard — same pattern as P19/P20 tests.
- **Net effect on weekend gate path:**
  - **Pre-P46**: alpha=68 / threshold=35 PASSES → `confidence=0.33 < 0.50` REJECTS
  - **Post-P46**: alpha=68 / threshold=35 PASSES → `confidence=0.44 (DRL sub) >= 0.30` PASSES → trade enters
- **What to verify after deploy:** new `[WEEKEND_CONF_DRL_SUB]` log lines in engine logs (means the substitution IS firing). Plus `gate_rejection_analysis.py --days 2` should show weekend rejections drop sharply or shift to a different reason.

### P45. [FIXED 2026-04-25] Weekend config plumbing audit + observability
- **Symptom:** P41 forensic surfaced 69 `[WEEKEND]` rejections + an additional ~164 "alpha gate" rejections that were actually weekend-gate rejections in disguise. Some samples showed `min 66` (mult=2.0 × hardcoded 33) despite operator's `live_high_risk.json:151` setting `min_alpha_multiplier_weekend: 1.0` — implying the JSON config wasn't always reaching `should_override_entry()`. P42 made the class-default fallback safe (1.0 × 20 = 20bps) but did NOT diagnose the root cause: where exactly does the config not flow through?
- **Fix:** Three changes — startup banner enrichment, runtime call-site diag, and analyzer extension. Same surgical pattern as P43/P44.
  1. **`main.py:1860-1872`** (startup banner): the existing `[WEEKEND_CONFIG] Profile-gated weekend overrides` log was missing the multiplier and base-bps fields. Now logs `alpha_mult_normal`, `alpha_mult_opp`, `alpha_base_bps` so operator can verify at boot whether values are coming from JSON or class default fallback.
  2. **`main.py:10584-10605`** (rejection call site): when the weekend gate vetoes, stash `intent._weekend_block_details` with the actual `_wk_cfg` keys present at that moment, the multiplier values it found, the base bps, and `wk_cfg_present` boolean. Plus a structured INFO log line on the rejection: `[WEEKEND] Entry blocked: ... (wk_cfg_present=True/False mult_normal=1.0/CLASS_DEFAULT mult_opp=0.45/CLASS_DEFAULT)`. Now operators see in the log AT THE MOMENT OF REJECTION whether config was loaded.
  3. **`main.py:12889-12895`** (shadow ledger gate_details builder): added 4 new weekend fields — `weekend_cfg_present`, `weekend_mult_normal`, `weekend_mult_opp`, `weekend_base_bps`.
  4. **`scripts/gate_rejection_analysis.py`** new "WEEKEND-BLOCK BREAKDOWN" section: pareto by config-loading status (`config_loaded` vs `NO_CONFIG (CLASS_DEFAULT fallback)`), by observed multiplier values (NORMAL + OPPORTUNITY), by observed base bps. After 24-48h of fresh data, this immediately surfaces whether the wiring is intact (`config_loaded: 100%`) or broken (`NO_CONFIG: N rejections`) — without operator having to grep production logs by hand.
- **Diagnostic verdict:** the existing 69 weekend rejections in the ledger all show `NO_CONFIG (CLASS_DEFAULT fallback)` — but this is because they're pre-P45 entries with no diag fields, not because config was actually missing. The new ledger entries (post-deploy) will show the real picture. **Hypothesis: config IS loading at startup correctly but possibly being lost on a config-reload code path.** The P45 instrumentation is what proves this either way.
- **Why the runtime-forensics pattern is now mature:** P41/P42/P43/P44/P45 form a complete loop. P41 added the analyzer; P42 fixed a config-tuning issue surfaced by it; P43-P45 added gate-specific observability so the next round of analysis is data-driven, not speculative. Each P-numbered observability commit follows the same template (10-15 minutes per gate): stash `intent._<gate>_block_details`, plumb fields into the shadow ledger gate_details dict, add a section to `gate_rejection_analysis.py` that surfaces the relevant pareto. Operators can replicate this pattern for any new top-blocker.

### P44. [FIXED 2026-04-25] STRUCTURE fractal-break observability — show WHY the gate blocks
- **Symptom:** P41 forensics surfaced 148 rejections (21.5% of all 30-day rejects) from the structure fractal-break gate — second-largest blocker after the weekend gate. Reasons in the ledger: `[STRUCTURE] LONG blocked -no fractal high break` / `[STRUCTURE] SHORT blocked -no fractal low break`. Like STALE_DATA before P43, the rejection lacks any context about HOW close to the fractal we were, what the configured threshold is, what edge we had, or which strategy was attempting the entry. Operator can't decide whether to loosen / convert-to-dampener / leave-alone without that data.
- **Fix (same surgical pattern as P43):** Two main.py edits + analyzer extension.
  1. **`main.py:10148-10172`** (LONG block site) and **`main.py:10269-10295`** (SHORT block site): when the gate vetoes, build `intent._structure_block_details` with `gap_bps`, `gap_limit_bps`, `edge_bps`, `min_edge_bps`, intent direction, fractal levels (resistance/support), regime, mode, strategy, existing exposure, soft-override-enabled flag. Replaced the opaque veto_reason with a structured one that includes the actual gap and threshold so even the log line is useful.
  2. **`main.py:12842-12891`** (shadow ledger gate_details builder): added 8 new structure fields (`structure_side`, `structure_gap_bps`, `structure_gap_limit_bps`, `structure_edge_bps`, `structure_min_edge_bps`, `structure_strategy_id`, `structure_soft_override_enabled`, `structure_existing_exposure`).
  3. **`scripts/gate_rejection_analysis.py`** new "STRUCTURE-BLOCK BREAKDOWN" section: pareto by side (LONG vs SHORT), by gap distance bucket (near/mid/far/very-far relative to the configured limit), by strategy that triggered the entry, by edge-vs-min-edge ratio. Pre-P44 entries show `n_with_diag_fields_p44: 0` so the boundary between old and new ledger format is visible.
- **What the new data will reveal:** running `gate_rejection_analysis.py --days 2` after P44 deploys will show the actual distribution of structure rejections. Three likely shapes:
  - **"By gap distance: most are 'near (< 1.5×limit)'"** → loosen the gap_limit_long / gap_limit_short config; we're rejecting trades that are barely outside the threshold.
  - **"By edge_vs_min: most have edge >= min_edge"** → the gap is the issue, not insufficient alpha; same conclusion as above.
  - **"By strategy: 90% momentum"** → the gate is hitting one strategy disproportionately; check whether momentum entries should bypass structure (they're directional-by-design and don't need confirmation from the same fractal logic that supports mean-revert).
- **Status:** instrumentation deployed; the data-collection cycle starts now. Operator runs analyzer in 24-48h to see the breakdown and decide which knob to turn.
- **Pattern note:** P41/P43/P44 together establish a repeatable runtime-forensics pattern: (a) static audit found nothing actionable; (b) ledger forensic shows the bucket; (c) bucket lacks enrichment, so we add diag fields + analyzer section; (d) wait for fresh data, run analyzer, decide. Each subsequent gate that turns out to be a top blocker can be wired up the same way in ~10 minutes.

### P43. [FIXED 2026-04-25] STALE_DATA observability — show WHICH feed is stuck
- **Symptom:** P41 forensics surfaced 110 STALE_DATA rejections all showing identical `alpha_estimated_bps=66.36` across many ticks — clear evidence the data pipeline is freezing during MOMENTUM_RALLY regime. But the shadow ledger only recorded `[TRADE_GATE] STALE_DATA` as the rejection reason, with no information about WHICH feed froze. Operator was blocked from acting on this finding because the bug is in runtime data flow and the ledger didn't capture enough context to diagnose it.
- **Fix:** Two surgical edits to main.py + analyzer extension.
  1. **`main.py:10416-10438`** (TRADE_GATE veto recording): when `gate_result.reason == STALE_DATA`, build a richer `intent.veto_reason` that includes `stale_sources`, `orderbook_stale`, `orderbook_fallback_reason`, `orderbook_cache_age_seconds`. Stash the full freshness diag dict on `intent._stale_freshness_details` for downstream pickup.
  2. **`main.py:12842-12868`** (shadow ledger gate_details builder): added 7 new fields to the gate_details dict — `data_age_seconds`, `orderbook_stale`, `orderbook_fallback_reason`, `orderbook_cache_age_seconds`, `stale_sources`, `freshness_mode`, `decision_lag_seconds`. The data IS being computed in `defense/trade_gate.py:_check_freshness_with_context` (line 487+); it just wasn't being plumbed through to the ledger.
  3. **`scripts/gate_rejection_analysis.py`** new "STALE_DATA BREAKDOWN" section: pareto by stale source (price/orderbook/vpin), by data_age bucket (0-60s / 60-300s / 5-30min / 30min-2h / >2h), by orderbook status, by freshness mode (direct vs tick_grace). Pre-P43 ledger entries are flagged explicitly as `<unknown — no diag fields>` so it's obvious which entries are old and which are new-format.
- **What happens next:** new ledger entries (post-deploy) will carry the diag fields. After 24-48 hours of live data, run `python -X utf8 scripts/gate_rejection_analysis.py --days 2` and the STALE_DATA BREAKDOWN section will show exactly which feed is stuck (e.g. "By stale source: price 84, orderbook 23, vpin 3" → it's the price feed, not vpin or orderbook). That pareto becomes the next actionable change.
- **Why this is the right pattern:** the audit-and-fix loop kept failing because static code review can't see WHICH feed went stale at 14:32 UTC last Tuesday. The diagnostic for any runtime data-flow bug is "make the ledger entry rich enough to debug from" — then a single forensic command reveals the cause without needing more code review. P41 added the analysis tool; P43 adds the data so the tool can do its job.
- **Data: existing 110 stale rejections in the ledger** all show `<unknown — no diag fields, pre-P43 ledger>` — this is the BASELINE confirming the gap. The next 110 (after P43 deploys) will be diagnosable.

### P42. [FIXED 2026-04-25] Weekend gate calibrated for 24/7 crypto (acts on P41 diagnostic)
- **Symptom:** P41's runtime forensics showed weekend gate blocking ~25% of all signals on Sat/Sun, even when alpha was 2-3× the regular threshold and DRL was strongly directional. Reason: `liquidity/weekend_manager.py:WEEKEND_MIN_ALPHA_MULTIPLIER = 2.0` doubled an already-hardcoded `33 * mult` floor → 66 bps minimum on weekends. Calibration was inherited from equities-style weekend illiquidity that doesn't apply to Kraken (24/7 market). Operator's live config (`live_high_risk.json:151`) already had `min_alpha_multiplier_weekend: 1.0`, but rejected samples in the ledger showed `min 66` reasons — meaning the config wasn't always reaching `should_override_entry()` and the class default was kicking in.
- **Fix (two layers):**
  1. **Class default lowered to match live config**: `WEEKEND_MIN_ALPHA_MULTIPLIER 2.0 → 1.0`, `WEEKEND_MIN_CONFIDENCE 0.75 → 0.50` in both `weekend_manager.py:WeekendOverrideRules` and `integrated_manager.py:Config`. This means a code path that fails to thread the config through (the actual root cause of "min 66" rejections at runtime) now falls back to the live-config default instead of equities-era 2.0.
  2. **Hardcoded `33` base now configurable**: new `WEEKEND_MIN_ALPHA_BPS = 20.0` class constant; new `weekend_min_alpha_bps` config key. Formula changed from `min_alpha = 33 * _alpha_mult` to `min_alpha = (config.weekend_min_alpha_bps or 20) * _alpha_mult`. Same change in `integrated_manager.py:Config.weekend_min_alpha_bps`. Operators can now tune via JSON without editing constants. Default 20 bps better matches the regular alpha threshold range (10-30 bps observed in P41 data).
- **Net effect on weekend min**:
  - Before (with class default if config not loaded): `33 * 2.0 = 66 bps` minimum
  - After (with class default): `20 * 1.0 = 20 bps` minimum
  - With operator's live config (`mult=1.0` already set): `20 * 1.0 = 20 bps`. Was `33 * 1.0 = 33 bps` before this commit.
- **Tests:** `tests/test_ultra_weekend_manager.py` already had 39 weekend tests; updated 2 (`test_default_caps` to assert new defaults, `test_opportunity_alpha_multiplier_falls_back_to_global_when_asset_missing` to use new bps math). Added 2 new tests: `test_weekend_min_alpha_bps_config_overrides_base` (operator override path), `test_weekend_default_floor_is_20_bps` (default floor in NORMAL mode).
- **Why this is the right shape**: P41 forensic showed alpha estimates clustering 10-50 bps in live trades. With base=20 and mult=1.0, weekend behavior is "slightly stricter than regular threshold in low-vol regimes (~10-15 bps regular → 20 weekend), neutral or looser in high-vol regimes (30 bps regular → 20 weekend)". Operator can still tighten via config (`weekend_min_alpha_bps`/`min_alpha_multiplier_weekend`) per regime / market condition without redeploying code.
- **What still needs operator action (P0 from P41 diagnostic):**
  - **Verify config loading**: the "min 66" rejections seen in ledger imply the weekend_config dict isn't always reaching `should_override_entry()`. Audit the call site (likely in main.py or trade_gate) to confirm `live_high_risk.json:weekend_manager` is passed through. The new defaults make this less catastrophic if it regresses, but it's still a real wiring bug.
  - **STALE_DATA pipeline freeze**: 110 rejections with identical alpha=66.36 — separate issue, not addressed here. Add `data_age_seconds` + feed source name to `[TRADE_GATE] STALE_DATA` rejection's `gate_details` so the next forensic run shows WHICH feed went stale.
  - **STRUCTURE break gate** (148 rejections, 21.5%) — fractal-break filter still blocking ~1/5 of all entries. Either lower the confirmation threshold or convert to a confidence dampener.

### P41. [DIAG 2026-04-25] Runtime gate-rejection forensics — why "1 fill in 30 days"
- **Symptom:** Operator reported persistent issue: ~10 rounds of static code audits found "fixes" but production keeps producing 1-50 fills per 30 days. Same modules / authorities / data pipelines flagged-then-fixed-then-re-broken. Root cause was that **static audits look at code, runtime bugs hide in data flow**. Solution: stop adding more audits, start adding runtime forensics.
- **Tools added:**
  - `scripts/gate_rejection_analysis.py` — reads `data/shadow_ledger/ledger_*.jsonl` and produces a Pareto of WHICH gate is killing trades, plus per-asset/regime/mode/strategy breakdown of the top blocker, plus a DRL contribution audit ("are strong DRL signals being honored?"), plus the "fill anatomy" comparing each successful fill to its 5 nearest rejects. Run it with `python -X utf8 scripts/gate_rejection_analysis.py --days 30`.
  - `scripts/alpha_gate_postmortem.py` — drills into alpha-related rejections: bucketed DRL direction/confidence distributions, alpha-vs-threshold gap distribution, P20 substitution evidence ("did DRL fall through the alpha gate's substitution branch?"), plus N raw sample rejections with full gate_details for eyeballing. Run with `--n-samples 10`.
- **What the runtime forensics revealed (2026-04-25 run):**
  1. **Of 687 gate rejects, 636 (92.6%) had |drl_direction| ≥ 0.5.** DRL is producing strong signals, but they're being filtered DOWNSTREAM of fusion. P19/P20/P23 are working at the alpha-gate layer — the kill happens at WEEKEND_GATE / STRUCTURE / STALE_DATA layers above it.
  2. **23.9% of rejections (164) bucketed as "ALPHA_GATE" are actually WEEKEND_GATE in disguise.** Reason text "Weekend alpha 55bps < min 66bps" — alpha is 55 bps, regular threshold is 30 bps (PASSING), but `liquidity/weekend_manager.py:WEEKEND_MIN_ALPHA_MULTIPLIER = 2.0` doubles the bar to 66 bps. Crypto trades 24/7 on Kraken; this multiplier is calibrated for equities-style weekend illiquidity that doesn't apply.
  3. **STALE_DATA fires 110× (16% of rejects) ALL with identical `alpha_estimated_bps=66.36` across many ticks.** That value being constant across many rejections means the alpha calc is feeding from frozen data — the data pipeline IS getting stuck, not just briefly slow.
  4. **STRUCTURE break gate blocks 148 (21.5%)** — fractal-break filter rejects "no fractal high break" 110 times for LONG and 38 for SHORT.
  5. **All 50 "fills" in the 30-day window had `strategy=None/?`** — they're from before strategy tagging was added, not real recent trades. **Properly-tagged data shows ~zero fills.**
- **Diagnostic verdict — three actionable items, in priority order:**
  - **P0 (LIKELY-BIGGEST WIN): WEEKEND_MIN_ALPHA_MULTIPLIER = 2.0** is too aggressive for 24/7 crypto. Either reduce to 1.2-1.3 OR remove the `33 * mult` floor entirely and rely on the regular alpha threshold (which already incorporates regime/friction). Backtest Sharpe (BTC=+9.22) was computed including weekends — current multiplier is bleeding alpha there.
  - **P0: STALE_DATA at 16% with identical alpha values is a real pipeline bug**, not a config tuning issue. Diagnose by adding `data_age_seconds` and a feed-source name to the STALE_DATA rejection's `gate_details` so the next forensic run shows WHICH feed went stale.
  - **P1: STRUCTURE fractal-break gate is over-strict** — 21.5% rejection rate means it's blocking the majority of momentum entries. Consider lowering the fractal-break confirmation requirement, or making it a confidence dampener rather than a hard veto.
- **Why this matters for the audit-cycle question:** static audits found wiring "ACTIVE" because the code IS active. The runtime forensics shows DRL signals reaching fusion AND surviving alpha gate AND still getting blocked downstream. No amount of grep-based wiring audit would have surfaced "weekend gate multiplier is mis-calibrated for 24/7 crypto" — that requires looking at WHY rejections happen, with the actual data. Recommend running `gate_rejection_analysis.py` weekly and using its output to drive the next change, not another code audit pass.

### P40. [FIXED 2026-04-24] Datetime naive/aware sweep across data_mgmt feeds
- **Symptom:** P39 fixed promotion_gate's naive/aware mixing, but the same pattern existed in 4 sibling feed files (`macro_feed.py`, `lob_feed.py`, `sentiment_feed.py`, `onchain_feed.py`). Each computes `staleness_sec = (datetime.now() - tick.timestamp).total_seconds()` where `tick.timestamp` came from `datetime.fromisoformat()` — aware OR naive depending on whether the persisted ISO string carried a tz marker. After a state file is written by a future tz-aware version, reload + staleness calc raises `TypeError` per tick, breaking ALL feed health checks.
- **Fix:** Added `strip_tz(dt)` helper to `data_mgmt/feeds/_http.py` (mirrors the pattern in `defense/strategy_existence_fuse.py` and `drl/promotion_gate.py:_strip_tz`). Wrapped 8 staleness-calc sites across the 4 feeds (2 per file: one in fetch path, one in `get_latest`). Each site now does `(datetime.now() - strip_tz(tick.timestamp)).total_seconds()`.
- **Tests:** Added 4 new tests in `tests/test_http_retry_and_manifest.py:TestStripTz`: None passthrough, naive passthrough (returns same instance), aware → tzinfo stripped, aware-past minus naive-now subtraction works (the actual bug case).
- **Mitigation:** When a module mixes `datetime.now()` (naive) with timestamps that flow through `fromisoformat()` (variable), normalize on the read side using `strip_tz()`. The helper is idempotent and cheap. Don't bet on persisted ISO strings being naive forever — old state files are naive, but any tz-aware writer (e.g., a `datetime.now(timezone.utc).isoformat()` introduced later) flips them to aware and breaks the comparison the next time the gate restarts and reloads state.

### P39. [FIXED 2026-04-24] Threading races + datetime naive/aware + numerical stability
- **Symptom:** A 3-lens audit hunt (numerical stability, threading concurrency, datetime naive/aware) surfaced 35+ findings including 5 HIGH bugs in code I introduced earlier this session:
  1. **Discord circuit breaker fields not lock-protected** (P29 regression). `_consecutive_failures`, `_circuit_open_until`, `_circuit_permanently_disabled` were read by callers and written by the worker thread without `self._lock`. Lost-update races could leave the breaker stuck or trip falsely.
  2. **`_wait_for_rate_limit` held `self._lock` across `time.sleep(wait)`**. A single rate-limit hit could block ALL Discord operations for 60+ seconds — hard deadlock during the wait window.
  3. **promotion_gate datetime mixing** (P33-related). `datetime.fromisoformat()` returns aware OR naive depending on the persisted ISO string's tz marker; the gate's runtime code uses naive `datetime.now()` everywhere; an aware-loaded `_demoted_at` compared to naive `datetime.now()` raises `TypeError` on EVERY tick after restart, blocking `get_authority_level()`.
  4. **`agents/squeeze_detector_agent.py:213`** — `short_liq / avg` after only `if avg <= 0: return` guard. Tiny positive `avg` (1e-15 from float averaging on illiquid alts) made the ratio explode and poison the squeeze score.
  5. **`agents/kraken_quant_agent.py:1654`** — `np.log(price)` with no `> 0` guard. Feed glitch sending price=0 → `np.log(0)=-inf` poisons the Kalman spread estimator until restart.
  6. **`market/phase_detector.py:266`** — `(prices_arr[-1] / prices_arr[-2] - 1) * 100` with no guard. NaN/0 historical price → momentum = NaN → propagates to phase detection → all downstream fusion signals inherit NaN.
- **Fix:**
  - **`infra/persistence.py`**: `_post_webhook` now acquires `self._lock` for ALL circuit-breaker reads/writes; logging happens outside the lock. `_wait_for_rate_limit` computes wait under the lock, releases, then sleeps. `_maybe_open_circuit` renamed to `_maybe_open_circuit_locked` to make the caller-holds-lock contract explicit.
  - **`drl/promotion_gate.py`**: added module-level `_strip_tz()` helper. `_load_state` strips tzinfo from `_demoted_at` on load; the two `fromisoformat()` reads in `_demote` and `get_status` route through the helper. New regression test `test_aware_demoted_at_loaded_as_naive_no_typeerror` writes a state file with `+00:00` ISO marker and asserts the load + a tick-style operation (get_status) don't raise.
  - **Numerical guards**: squeeze_detector tightened `if avg <= 0` → `if avg < 1e-6`; kraken_quant Kalman returns None when either price ≤ 0; phase_detector momentum guards both `_p1 > 0` and `not np.isnan(_p1)` per leg.
- **Other findings from this round (deferred or skipped)**:
  - 4 more datetime sites in feeds (`macro_feed`, `lob_feed`, `sentiment_feed`, `onchain_feed`) — same pattern, different files. Defer to next pass.
  - `loop_controller.py` tick_counter race (MED) — Python ints + GIL, dormant in practice.
  - `runtime_state.py` nested RLock fragility (MED) — RLock means it doesn't deadlock today, but caller-holds-lock contract should be made explicit. Defer.
  - 12 more numerical findings (MED/LOW) in `whale_detector`, `microstructure_agent`, `intraday_correlation_monitor`, `learned_execution_policy`, etc. Defer batch.
- **Mitigation:** When introducing a new field that's read by multiple threads (Discord-style background worker, websocket reader, etc.), lock-protect ALL reads + writes from day 1. When mixing `datetime.now()` with `fromisoformat()`-parsed timestamps, normalize tz-handling on the load side immediately — don't rely on naive code being naive forever. When dividing by a float quantity from a feed, guard with an explicit epsilon (`x < 1e-6`) not just `x <= 0` — float averaging produces near-zero values that survive the latter check.

### P38. [FIXED 2026-04-24] External feed 429 visibility + Solana RPC per-call timeouts
- **Symptom:** Diagnostic pass on the audit's deferred items. Two real issues, three "skip — already mitigated":
  1. **Solana RPC + Jito had no per-call timeouts** — `solana_onchain.py:128, 159` used `session.post`/`session.get` without `timeout=`, falling back to the session-level 15s. For RPC + Jito specifically (low-quota, fast-path), 10s per call is the right budget — keeps the live tick from stalling on a slow RPC node.
  2. **Coinglass / CryptoCompare 429s silently dropped** — 5 direct `session.get` sites in `coinglass_feed.py` (×3), `cryptocompare_onchain.py`, `cryptocompare_news_feed.py` checked `resp.status != 200` and returned None / empty list / continue. 429 was indistinguishable from any other 4xx in logs, hiding rate-limit issues that operators couldn't diagnose.
- **Fix:** (1) Added `timeout=aiohttp.ClientTimeout(total=10)` to both Solana RPC and Jito calls. (2) Added `parse_retry_after()` helper to `data_mgmt/feeds/_http.py` (extracted from `fetch_with_retry`'s 429 path) so direct-call clients can honor `Retry-After` without routing through the full retry loop. Wired into all 5 sites: each now logs a distinct `WARNING ... rate-limited (429), Retry-After=Ns` line so operators see rate limits in heartbeat logs.
- **Diagnostic verdicts on other deferred items (no action needed):**
  - **FastAPI 0.0.0.0 bind** — `docker-compose.hetzner.yml:72` already binds `127.0.0.1:8080:8080`. The internal 0.0.0.0 is irrelevant because Docker is the security boundary. Audit was a false positive at the deployment layer. **SKIP.**
  - **BLACK_SWAN_SENTINEL DRL punch-through** — DRL trained on data that doesn't include any specific crisis; trusting its prediction during a regime its training distribution didn't cover is exactly when it's least reliable. The OOD detector exists for this reason. Adding a punch-through here defeats the intent of the safety gate. **SKIP.**
  - **AUTHORITY_MATRIX_NO_TRADE DRL=NONE** — NO_TRADE is the strongest fail-closed safety state. Adding DRL override defeats its purpose. **SKIP.**
  - **`weights_only=True` end-state refactor** — 6 save sites + 4 load sites across 4 training pipelines, ~half-day. P30 wrapper provides defense-in-depth so this isn't urgent.
  - **3 fire-and-forget tasks in main.py** (lines 15839, 15846, 15856 + their run_live mirrors at 16651, 16657, 16663) — same P37 pattern applies; blocked on operator's main.py WIP being committed.
- **Tests:** Added 6 new tests (`TestParseRetryAfter` class in `tests/test_http_retry_and_manifest.py`): None input, empty/whitespace-only input, integer-seconds form, negative-seconds clamp to 0, unparseable returns None, HTTP-date form parses to seconds-from-now within tolerance.
- **Mitigation:** When adding a new `session.get`/`session.post` call to a boundary client, default to either (a) `fetch_with_retry` from `_http.py` if the call fits the generic-JSON-with-retry pattern, or (b) handle `resp.status == 429` specifically with `parse_retry_after(resp.headers.get("Retry-After"))` — never lump 429 with other 4xx in a generic warn-and-drop branch.

### P37. [FIXED 2026-04-24] State-persistence atomicity + fire-and-forget asyncio task tracking
- **Symptom:** A new round of audit lenses (state persistence, fire-and-forget asyncio, mutable defaults, boundary clients) surfaced 3 patterns of crash-unsafe and silent-failure code:
  1. **Non-atomic state writes** (5 sites): `open(path, 'w'); json.dump(...)` → corrupt on crash mid-write. Worst was `drl/promotion_gate.py:276` which writes the live DRL authority level — a crash on Hetzner OOM / Docker restart could leave the system in inconsistent demoted state next boot.
  2. **Fire-and-forget `asyncio.create_task(coro)` with discarded result** (3 sites in tracked code): Python can GC the unreferenced task mid-execution, AND there's no way to cancel them on shutdown. `lead_lag_engine.start()` was the worst — both Binance and Deribit WebSocket message loops untracked, so `stop()` couldn't cancel them, leading to hung graceful shutdowns and zombie connections.
  3. **Missing JSONL `flush()`** (2 sites): `pnl_attribution._persist_trade` and `trade_attributor._persist_trade` wrote without flushing — last few records lost on Docker kill.
- **Fix:**
  - **Atomic writes**: routed through existing `core.state_persistence.save_state()` (tempfile + `os.replace`). Sites: `drl/promotion_gate.py:_save_state`, `core/runtime_state.py:export_to_file`, `analytics/pnl_attribution.py:persist_to_file`, `execution/impact_calibration.py`, `execution/fees/kraken_plus_fee_blender.py:_save`.
  - **JSONL flush**: added `f.flush()` after each line in `pnl_attribution._persist_trade` and `trade_attributor._persist_trade` (matches the existing pattern in `attribution_tracker.py`, `trade_explainer.py`, `live_experience_buffer.py`, `shadow_ledger_jsonl.py`).
  - **Task tracking**: instance attrs + cancel branch. `market/lead_lag_engine.py` — `_binance_task`, `_deribit_task` cancelled in `stop()`; `infra/kraken_link.py` — `_reconnect_task` stored + duplicate-spawn guard; `execution/sota_scheduler.py:reset_execution_scheduler` — `add_done_callback` logs exceptions instead of swallowing.
- **Not fixed in this commit (deferred)**:
  - 3 fire-and-forget tasks in `main.py` (~lines 15831, 15838, 15848 — OnChainFeed/LeadLagAlphaEngine/SolanaOnChainAgent `.start()`) — main.py is in operator WIP, will get the same pattern when next committed.
  - Coinglass / CryptoCompare direct `session.get` calls bypass `_http.py:fetch_with_retry` 429 path — moderate refactor.
  - Solana RPC + Jito per-call timeouts.
- **Mitigation:** When introducing new state writes, default to `from core.state_persistence import save_state` rather than raw `open(w) + json.dump`. When spawning a long-lived task with `asyncio.create_task`, always assign to an instance attribute and add a cancel branch to the corresponding `stop()`/`close()` method. JSONL appenders need explicit `f.flush()` after every line — Python's text-mode buffer doesn't survive Docker SIGKILL.

### P36. [FIXED 2026-04-24] Architecture invariants + P22 regression tests cleanup
- **Symptom:** Two architecture invariants flagged FAIL by the audit but missed by P22:
  1. `scripts/runtime_parity_check.py:47` — `BEST_FOLDS = {"BTC": "fold_3", "ETH": "fold_1", "SOL": "fold_3"}` — ETH still pointed at the stale fold_1 (P4). The parity-check script would silently use the wrong fold for ETH validation, masking mixed-fold pairing bugs.
  2. `training/train_drl_full.py:1182-1188` — `SubprocVecEnv` path was reachable despite the file's own header invariant ("DummyVecEnv ONLY — SubprocVecEnv deadlocks on Windows, no speedup on WSL2"). Setting `vec_env_type="subproc"` would silently produce a hung run.
- **Fix:** (1) `runtime_parity_check.py` ETH → `fold_3` aligned with `drl/ensemble.py` + `drl/runtime_obs_builder.py` + `training/drl/oracle_tqc_teacher.py`. (2) `train_drl_full.py` raises `RuntimeError` if `vec_env_type=="subproc"` is requested, with explicit `HMATS_ALLOW_SUBPROC_VEC_ENV=1` env-var override for operators who really know what they're doing.
- **Tests added** (`tests/test_http_retry_and_manifest.py`, 9 tests): closes regression-test gaps for two earlier P22 hunks that shipped without coverage:
  - **fetch_with_retry HTTP 429** (6 tests): 429 with Retry-After retries successfully, Retry-After is capped at 30s (defense against malicious "wait 99999s"), 4xx-other-than-429 still drops immediately (no regression on existing 4xx behavior), 5xx exponential backoff retries succeed, 429 without Retry-After header falls back to `backoff_base`, max_retries exhausted returns None.
  - **Manifest path traversal** (3 tests): relative weights under manifest dir pass; `../../etc/passwd` traversal raises `ValueError` from `relative_to()`; `./../weights/model.pt` 1-level escape detected.
- **Mitigation:** Architecture invariants from CLAUDE.md (BEST_FOLDS in 4 places now, DummyVecEnv-only, ent_coef=float, etc.) should be grepped on every audit. The `BEST_FOLDS` constant is now consistent across all 4 hot files; if a 5th file introduces it, this CLAUDE.md entry is the search anchor. The SubprocVecEnv guard is opt-in-only via env var, so accidental reactivation is loud (RuntimeError) instead of silent (deadlock).

### P35. [FIXED 2026-04-24] AuthorityFusionEngine test harness — multi-agent fusion core
- **Symptom:** Audit's last "no tests" gap. `signals/authority_fusion.py` is the multi-agent fusion core that consumes all 25 authority-matrix agents and produces the fused direction/confidence/exposure. It's also where DRL's authority upgrade lands (`set_drl_authority_level("ACTIVE")` mutates the matrix), where multi-DECIDE conflict resolution lives (FIX-H2 sign-agreement fix), and where mode-specific matrices (NORMAL / OPPORTUNITY / NO_TRADE) gate every trade. P12, P19, P20 all touched downstream consumers of this engine.
- **Fix:** Added `tests/test_authority_fusion.py` (21 tests across 6 classes): authority matrix lookup (4 — 25-agent count invariant, NO_TRADE/NORMAL/OPPORTUNITY structure); DRL authority upgrade (5 — DISABLED stays ADVISE, ACTIVE upgrades in NORMAL+OPPORTUNITY but NOT NO_TRADE, SHADOW does not upgrade); fuse() Layer 1 hard-state (2 — NO_TRADE/data_invalid → direction=0); fuse() Layer 2 VETO (3 — risk veto blocks, missing risk fail-closed FIX-C1, inactive risk allows); fuse() Layer 3 DECIDE pool (6 — quant dominates when kraken_quant silent, empty DECIDE fail-closed, DRL ACTIVE joins pool, direction clamped [-1,1] FIX-H2, opposing signals reduce confidence via sign-agreement fraction, 3-agent 2/3 majority keeps full conf); ADVISE doesn't decide (1 — even strong opposing ADVISE signal cannot flip direction).
- **Tests caught a real surprise:** my initial "single DECIDE agent decides" test assumed only quant was in the DECIDE pool in NORMAL. Actual matrix has BOTH quant AND kraken_quant as DECIDE (per authority matrix v6.8 row 18). So `fuse()` ALWAYS runs the multi-DECIDE consensus path in NORMAL — the "single decider" branch is essentially dead. Tests rewritten with explicit `kraken_quant` signals in fixtures. The `_drl_authority_level` is module-global, so an autouse fixture resets it between tests to prevent cross-test leakage.
- **Mitigation:** When adding/removing a DECIDE agent in the authority matrix, run this whole file. The multi-DECIDE consensus is sensitive to which agents have non-zero confidence — silent DECIDE agents (zero confidence) effectively cede the decision to non-zero peers, which is the intended behavior but easy to miss when reasoning about contributions.

### P34. [FIXED 2026-04-24] StrategyExistenceFuse test harness — 28-day window + consecutive-loss safety net
- **Symptom:** Audit found `defense/strategy_existence_fuse.py` (the rolling-window suspension fuse — "have we been losing money for too long? halt entries") had zero direct tests despite multiple recent fixes (FIX-FUSE-AUTORECOVERY, FIX-DA2, FIX-H3, UL-6a, UL-6b). The module is responsible for halting all trading after weekly/monthly/streak loss limits, then refusing to resume without manual confirmation. A regression here = silent loss of safety, or worse, a fuse that won't reset and blocks legitimate trading for days.
- **Fix:** Added `tests/test_strategy_existence_fuse.py` (32 tests across 9 classes):
  - **Initialization (8)**: 28d window, -15% PnL threshold, -18% equity DD, -15% weekly limit, -18% monthly limit, 10 consecutive-loss limit, default `allow_auto_recovery=False` (FIX-DA2 — matches documented contract).
  - **record_pnl (3)**: starting-equity initialization, history append, cumulative accumulation.
  - **Weekly limit (2)**: 5% loss → no suspend, 20% loss → suspend with "Weekly" reason.
  - **Monthly limit (1)**: spread 20% loss over 30 days → trips structural limit.
  - **min_data_points (1)**: 4 records (< min 5) skip evaluation even with massive loss.
  - **Consecutive trade losses (5)**: 9 → no suspend, 10 → suspend, winner resets streak, **[FIX-H3] breakeven (pnl=0) does NOT reset** (rounding/fees shouldn't break a streak), 0/positive initial keeps counter at 0.
  - **is_entry_allowed (3)**: ACTIVE allows, SUSPENDED blocks, `is_suspended()` truth table.
  - **Manual unsuspend (5)**: refuses when not suspended, refuses without `confirm=True`, refuses during cooldown, succeeds after cooldown, **resets `_consecutive_trade_losses` on unsuspend** (recent fix — without reset, single new loss would re-trip immediately).
  - **Time-based recovery (3)**: consecutive-loss suspension recovers after cooldown when `allow_auto_recovery=True`; structural (weekly/monthly) suspensions do NOT auto-recover via this path; `allow_auto_recovery=False` (default) blocks all auto-recovery.
  - **force_suspend (1)**: emergency-suspend test path.
- **Tests caught a real surprise:** my initial test for `test_winner_resets_streak` assumed `on_trade_close` only INCREMENTS on negative pnl and ignored positive — but the [FIX-H3] reset path does `self._consecutive_trade_losses = 0` on `pnl > 0`. Test corrected; an additional `test_breakeven_does_not_reset_streak` test guards the FIX-H3 invariant (pnl==0 must NOT reset, only pnl>0).
- **Mitigation:** When changing any threshold (weekly/monthly limits, consecutive-loss pause, cooldown hours), run this whole file. The time-based recovery path (FIX-FUSE-AUTORECOVERY) is intentionally narrow — only "Consecutive trade losses" reasons recover. Structural reasons require either window improvement OR manual unsuspend, by design.

### P33. [FIXED 2026-04-24] DRLPromotionGate test harness — auto-demotion safety net
- **Symptom:** Audit found `drl/promotion_gate.py` (the auto-demotion safety net per CLAUDE.md non-negotiable rule #4) had zero direct tests. With DRL ACTIVE on the live tick, this is the module that auto-demotes to EXIT_ONLY on 5 consecutive losses or 15% drawdown, and to DISABLED after 3 demotions in 14 days. Multiple subtle paths (P-PATCH-4 zero-peak demotion, FIX-DA3 EXIT_ONLY-also-demotes, recovery-doesn't-lift-DISABLED, demotion-window aging) had never been exercised by a test.
- **Fix:** Added `tests/test_drl_promotion_gate.py` (34 tests across 8 classes): initialization defaults; manual promote (valid + invalid); trade recording (drl_contributed=True/False); consecutive loss demotion (5 → demote, 4 → no, streak broken → no, EXIT_ONLY also demotes); drawdown demotion (15% → demote, 10% → no, **zero-peak-with-loss → demote** per P-PATCH-4); auto-recovery (cooldown elapsed → ACTIVE, DISABLED stays DISABLED, equity resets); max demotions disable (3 in window → DISABLED, stale demotions outside window don't count); state persistence (save/load round-trip + demotion_history); authority enum shim (`get_authority()`, `has_exit_authority()`, `is_shadow_mode()`).
- **Tests caught a real surprise:** writing the tests revealed that the zero-peak demotion path (P-PATCH-4) is more aggressive than the consecutive-loss path — fires on the very FIRST loss when peak is still 0. This is the intended behavior (closing the silent-bypass bug), but tests for the consecutive-loss path must build positive equity FIRST (peak>0) for the consecutive counter to ever reach 5. Added a comment in the test class explaining this so future test authors don't trip on it.
- **Mitigation:** Three parallel auto-demotion paths (zero-peak vs drawdown vs consecutive-loss) interact subtly. When tweaking any threshold or adding a new demotion trigger, run this whole file — most assertions are tight enough to catch off-by-one drifts.

### P32. [FIXED 2026-04-24] Constitution test harness — supreme-gate regression coverage
- **Symptom:** The audit found `defense/constitution.py` (86KB, supreme authority gate per non-negotiable rule #1) had no direct tests. Existing tests covered `AlphaThresholdCalculator` (test_alpha_gate.py) and `TrancheScheduler` (test_ultra_tranche.py) but the schema validator and the NO_TRADE trigger checker — both of which decide whether ANY trade can happen — were untested. P14 fixed silent dead-reads in this file; P12 fixed a conflict-score escalation bug here; P22 fixed a schema-vs-runtime drift here. Each prior fix shipped without a regression guard.
- **Fix:** Added `tests/test_constitution_core.py` (30 tests) covering:
  - `MarketDataValidator` (12 tests): valid passthrough, missing required → reject, missing critical in NORMAL → reject vs. OPPORTUNITY → default-applied, missing optional → default-applied, alias resolution, price range (negative/zero/excessive), data age threshold (P22 invariant: 30s passes, 120s rejects), NaN rejection.
  - `NoTradeTriggerChecker` (18 tests across 6 classes): stale data threshold, feed disagreement >1%, DVOL z-score extreme, liquidity critical, correlation collapse at 0.92 threshold, signal conflict (P12 invariant — 2-agent opposing scores 0.7 but does NOT activate ALL_CONFLICT_FLAT), Iron Law #34 (sentiment never enters conflict), aggregation (clean → no_trade=False, any trigger → no_trade=True, multi-trigger primary_reason).
- **Mitigation:** When changing a constitution threshold or adding a new trigger type, the matching test in `test_constitution_core.py` should fail first (TDD-style). The P12 invariant test (`test_2agent_opposing_scores_07_but_no_active_condition`) is the source-of-truth guard — paired with the integration_v36.py source-level guard from P23, the 2-agent conflict over-promotion bug now has two independent regression tracks.

### P31. [FIXED 2026-04-24] joblib.load + pickle.load defense-in-depth (P30 extension)
- **Symptom:** Same RCE-by-pickle vector as P30 but for `joblib.load` (~30 sites repo-wide; 4 on the live trading hot path) and `pickle.load` (1 live site in `defense/drift_detector.py:530`). joblib uses pickle internally; loading from an attacker-influenced path = arbitrary code execution.
- **Live sites wrapped (committed):** `agents/model_alpha_agent.py:1075` (sequence-alpha scaler), `drl/runtime_obs_builder.py:112` (TQC observation scaler), `defense/drift_detector.py:530` (drift checkpoint pickle — `checkpoint_dir` defaults to `/tmp/hmats_checkpoints`, wrapper called with `extra_allowed_roots=[checkpoint_dir]` to allow that one specific dir while still rejecting path-injection attempts).
- **Sites wrapped in working tree but committed via separate main.py work:** `main.py:3373` (per-asset GMM model load), `main.py:3402` (legacy global GMM fallback). Edits sit in the working tree alongside other pending main.py changes; they ship when the next main.py commit lands.
- **Fix:** Extended `infra/safe_torch_load.py` with `safe_joblib_load` and `safe_pickle_load` — same path-prefix validation as `safe_torch_load`, just delegating to a different deserializer.
- **Tests:** 5 new tests in `tests/test_safe_torch_load.py` (14 total): joblib allowed/rejected paths, pickle allowed/rejected paths, `extra_allowed_roots` widens for pickle.
- **Mitigation:** When introducing a new pickle/joblib site, use the wrapper. Out-of-tree training/archive scripts deliberately not wrapped (operator-controlled, not on hot path).

### P30. [FIXED 2026-04-24] torch.load(weights_only=False) defense-in-depth
- **Symptom:** Audit identified 4 `torch.load(..., weights_only=False)` sites in live agent code (`exit_drl_agent.py:146`, `sentiment_deberta.py:63`, `model_alpha_agent.py:1017`, `model_alpha_agent.py:1176`). All load checkpoints that contain pickled custom config classes (`SentimentConfigV22`, `DTConfigV32`, `SequenceAlphaConfig`, exit-SAC dict). Pickle = arbitrary code execution at load time. If an attacker can write to a path that the agent loads from, that's RCE.
- **Constraint:** Can't simply switch to `weights_only=True` — that would reject the pickled config dataclasses. Proper fix is splitting `state_dict` from `config` in the training save path, which is a multi-script refactor.
- **Fix:** Added `infra/safe_torch_load.py` — wrapper around `torch.load` that resolves the path and rejects anything outside an allowlist of model-root directories. Default allowlist: repo `models/`, `/opt/hmats/models` (in-container), `/var/lib/docker/volumes/hmats-models` (Docker host bind). Override via `HMATS_TORCH_LOAD_ALLOWED_ROOTS` env var (`os.pathsep`-separated) or `extra_allowed_roots=[...]` kwarg per-call. All 4 live sites now route through the wrapper. Defense-in-depth: even if the path string is attacker-controlled, the load fails before unpickling unless the path is under a blessed root.
- **Tests:** `tests/test_safe_torch_load.py` (9 tests): allowed path passes; outside-allowlist rejected; `..` traversal rejected; per-call extra root extends allowlist; env var extends allowlist; no-roots-resolved refuses; `safe_torch_load` calls `torch.load` for allowed paths; `safe_torch_load` does NOT call `torch.load` for rejected paths (critical — pickle never runs); extra kwargs forwarded.
- **Mitigation:** When introducing a new `torch.load` / `pickle.load` / `joblib.load` call site, route through `infra.safe_torch_load.safe_torch_load` (or a sibling helper for joblib/pickle that doesn't exist yet — TODO). The full-strength fix (`weights_only=True` everywhere) requires saving config separately from state_dict in `training/sentiment/`, `training/drl/`, `training/model_alpha/`, `training/exit_drl/`. Not done in this commit; wrapper is the immediate-mitigation defense layer.

### P29. [FIXED 2026-04-24] External API resilience — Discord circuit breaker + Haiku 429
- **Symptom:** Two boundary error-handling gaps surfaced by the audit:
  1. **Discord webhook**: `_post_webhook` had no failure tracking. A revoked webhook URL (HTTP 401/403) generated an `ERROR Discord webhook HTTP 401` log line for every queued message forever — log spam, no recovery, no alerting that the channel was permanently dead. Rate limit (429) was logged but not honored — next message hit immediately, burning quota.
  2. **Haiku LLM sentiment**: `_classify_haiku_error` lumped 429 into `transient_error`. Each per-asset tick fired another request; under sustained 429, this hammered the Anthropic API and silently fell back to the deterministic heuristic without operator visibility.
- **Fix:** (1) Added circuit breaker to `infra/persistence.py:DiscordNotifier`: `_consecutive_failures` counter, `_circuit_open_until` timestamp, `_circuit_permanently_disabled` flag. 401/403/404 → permanent disable + ERROR log once. 429 → cooldown driven by `Retry-After` header (capped 1-300s). Other failures → exponential backoff: 5 consecutive → 60s, 10 → 5min, 15+ → 30min. First success resets all state. (2) Added 429-specific path to `agents/sentiment_llm_agent.py:_classify_haiku_error` — parses `Retry-After` from `err.response.headers`, returns `(429, non_retryable=True, "rate_limit_429_retry_after=N")`. Caller routes to `_open_hard_disable(cooldown_sec=retry_after)` with a short cooldown (default 30s) so the Haiku circuit reopens quickly for transient rate limits while still suppressing the busy-loop.
- **Tests:** `tests/test_external_api_resilience.py` (15 tests): Discord 401/403 permanent disable, 429 with/without Retry-After, retry-after capping at 300s, 5-failure threshold, sub-threshold no-circuit, open-circuit message drops, permanent-disable message drops; Haiku 429 by status code, 429 by message text, "rate limit" text, Retry-After header parsing, 403 vs 429 distinction, 500 still transient.
- **Mitigation:** When wrapping a network call: (a) status-code-by-status-code error classification, not "all errors equal"; (b) circuit breaker on consecutive failures; (c) honor `Retry-After`; (d) distinguish auth (permanent) from rate-limit (transient cooldown) from network (exponential backoff). The pattern in `infra/persistence.py:DiscordNotifier._maybe_open_circuit` is reusable for any other webhook/API client.

### P23. [FIXED 2026-04-24] Regression tests for P12/P19/P20 — DRL authority punch-through family
- **Symptom:** Three recent silent-bug fixes (P12 conflict-score over-promotion, P19 BEST_OF_N_HOLD short-circuit, P20 effective_alpha_direction substitution) had no test coverage. A future refactor that removes any of the punch-through branches, or drifts the |dir|>=0.5 / conf>=0.3 thresholds, would silently regress DRL's DECIDE authority back to effective ADVISE — exactly the failure mode that produced 1 fill / 241 rejections over 30 days. Same family as P4 (ETH fold_1 stale checkpoint), where the absence of a regression guard let a fixed bug return.
- **Fix:** Added `tests/test_drl_authority_punchthrough.py` (13 tests):
  - **P19** — 6 tests covering `_maybe_apply_pre_alpha_hold`: DRL DISABLED applies hold; ACTIVE+strong (LONG and SHORT) punches through; weak direction (|dir|=0.4) still holds; low confidence (conf=0.2) still holds; ADVISE authority does not punch through.
  - **P20** — 6 tests covering `_compute_effective_alpha_direction`: DRL DISABLED returns 0; ACTIVE+strong substitutes; weak direction does not substitute; low confidence does not substitute; non-abstaining quant doesn't get DRL override; ADVISE authority does not substitute.
  - **P12** — 1 source-level guard: the `signal_conflict_score=1.0` line that feeds `risk_veto_classifier.classify` must require `>= 0.9`. Catches accidental drift back to `> 0.5` (HARD VETO) without false-positives on the cosmetic `intent.signal_conflict_detected` audit flag (which legitimately uses `> 0.5`).
- **Mitigation:** When fixing a silent agent-routing bug, write the regression test in the same commit. Without it, the fix is one careless refactor away from returning. Threshold values (`|dir|>=0.5`, `conf>=0.3`) are explicit in tests so a future drop to `0.3 / 0.2` etc. fails CI immediately rather than degrading DRL behavior in production.

### P22. [FIXED 2026-04-24] Full-repo audit cleanup — schema/security/config drift
- **Symptom:** Multi-lens repo audit surfaced 6 unrelated issues that each looked harmless in isolation:
  1. `defense/constitution.py:105` `MAX_DATA_AGE_SECONDS = 60.0` but schema at line 47 capped `data_age_seconds` at `10.0`. A 30s data-age sample would pass the runtime constant but fail schema validation → spurious NO_TRADE rejections.
  2. `infra/alert_manager.py:435,442` used `os.system(f"...{alert.message}...")` for desktop popups. `alert.message` is built from regime descriptions / error text — anything containing `; rm -rf /`, `` ` ``, `$(...)`, etc. would shell-inject. Live code (imported at `main.py:1289`).
  3. `orchestration/strategic_coordinator.py:798` hardcoded `MAX_LEVERAGE = 2.0` inside an inactive `_PATCH_4_ACTIVE = False` block, while `configs/canonical_config.py:95` and `configs/sota_flags.py:266` and `main.py:1061` all say `3.0`. If the patch flag is ever flipped on, position sizing silently undersizes by 33%.
  4. `data_mgmt/feeds/_http.py:44-46` `fetch_with_retry` lumped `429` with all `4xx` and dropped without retry. CryptoPanic / Coinglass / Anthropic burned quota silently under load — no `Retry-After` parsing.
  5. `execution/learned_execution_policy.py:507-511` resolved manifest-relative weights paths via `.resolve()` only, no escape check. A manifest claiming `weights_path="../../../etc/x.pt"` would resolve outside the manifest dir.
  6. `data_mgmt/feeds/binance_ticker.py:80-81` `try: ... except Exception: pass` on Binance kline fetch silently returned `taker_buy=taker_sell=0` as valid data, feeding the microstructure agent zeroed flow on every kline failure (network blip, rate limit, schema change).
- **Fix:** (1) Schema `data_age_seconds.max` aligned to `60.0`. (2) Replaced `os.system` with `subprocess.run([list], shell=False)` + AppleScript quote escaping; added 5s timeout. (3) `strategic_coordinator.py` now imports `MAX_LEVERAGE` from `configs.canonical_config` with fallback to `3.0`. (4) `fetch_with_retry` retries on 429, parses `Retry-After` (seconds + HTTP-date), caps wait at 30s. (5) Manifest weights resolution now requires `weights_path.relative_to(manifest_root)` — escape attempts raise. (6) Binance kline failures now log at debug level so partial failures are visible in heartbeat.
- **Mitigation:** When a constant has multiple definitions, grep ALL of them on every change. Schema range vs runtime constant is a common source of drift. `os.system(f"...")` with any interpolated string is a code smell — use `subprocess.run([list], shell=False)` always. `except Exception: pass` on a network call that returns numeric data feeds zeros downstream — at minimum log it.

### P21. [DIAG 2026-04-24] DECIDE pool observability log
- **Symptom:** 30-day audit found zero `DECIDE_CONSENSUS` or `DECIDE_CONFLICT` log lines. Either fusion was being bypassed by earlier rejections, OR DRL was never in DECIDE pool despite promotion — couldn't tell from logs.
- **Mitigation:** Added periodic `[DECIDE_POOL]` diagnostic at `authority_fusion.py:~492` that logs every 10th call: matrix name, DRL authority level, and agents in DECIDE pool. Lets us verify promotion actually landed in fusion.

### P13. [FIXED 2026-04-24] kraken_quant cross-asset data starvation
- **Symptom:** kraken_quant 12-strategy matrix wired as DECIDE and ACTIVE, but 30 ticks over 39h all showed `○kraken_quant=+0.00/0.00/dq1.0` in AGENT-TRACE. Looked like "ACTIVE but never fires".
- **Cause:** main.py's `market_data` dict is built per-asset (one tick = one call per asset). kraken_quant's 12 strategies (Kalman cointegration, ETF-spot cointegration, relative strength, Hurst, etc.) ALL need BTC+ETH+SOL prices/OI/funding simultaneously for cross-asset stat-arb. `_convert_market_data` expected `price_btc`/`price_eth`/`price_sol` suffixed keys but only got flat `market_data["price"]` for the current asset → 2/3 assets see zeros → strategies bail.
- **Fix:** Added `self._kq_xasset_cache: Dict[str, Dict]` in main.py. Each per-asset tick updates the cache; before calling `kraken_quant.generate_signal`, inject `price_{asset}`, `open_interest_{asset}`, `funding_rate_{asset}`, `liquidation_volume_{asset}`, `taker_ratio_{asset}`, `bid_depth_{asset}`, `ask_depth_{asset}` from all 3 cached asset snapshots. Cross-asset data is ≤1-tick stale (acceptable for 4H stat-arb).
- **Mitigation:** When a multi-asset agent produces no signals despite being ACTIVE, check its input-dict expectations against the per-asset loop structure.

### P11. [FIXED 2026-04-22] Local-var instantiation hidden from naive wiring scans
- **Historical symptom:** `AttributionTracker` and `AgentScorecard` appeared as `SOMETHING_CREATED` in `startup_agent_wiring_truth.py`, suggesting "imported, instantiated, but methods never called". Led to mistakenly concluding 2 agents were half-wired.
- **Reality:** `_attr_tracker = get_attribution_tracker()` (local var at main.py:8298) + `.record_signals()` / `.resolve_outcome()` / `.get_decay_alerts()` calls via that local var. Not a `self.xxx` attribute, so the old regex missed both the L3 assignment and all L4 method calls.
- **Resolution:** Script updated (commit 540167d) to match **both** `self.xxx = Foo(...)` AND indented `^\s+_var = Foo(...)` local-var patterns, plus method calls on locals. Wiring score 84.5% → 93.8% ACTIVE as a result. If another SOMETHING_CREATED verdict appears, check whether the class is consumed via a local var in a long function before assuming it's really unused.

### P10. TWO SEPARATE DRL systems — don't confuse them
- **`drl/ensemble.py` + `models/retrained/{ASSET}/fold_3/.../best_model.zip`** — the **TQC direction DRL** we activated 2026-04-22. Predicts direction+confidence, feeds `agent_signals["drl_direction"]`/`drl_confidence`, authority ACTIVE, Sharpe +9 on val backtest. **This is the main DRL.**
- **`agents/drl_agent.py` DRLAgent class** — a completely separate **tranche/exit optimization DRL** designed for local execution timing (T2→T3 escalation, exit pressure, runner hold/release). Per the file's own docstring: "DRL DOES NOT DECIDE direction". Requires a DIFFERENT trained model (env var `HMATS_DRL_MODEL_PATH`) which we don't have, so it runs with `mode=DISABLED` and returns neutral. **Not dead code — dormant Phase-2 scaffolding.**
- **Symptom of confusion:** `startup_agent_wiring_truth.py` flags `DRLAgent` as INSTANTIATED_BUT_UNUSED. That's accurate for this class (no model → no methods called), but does NOT mean "DRL is off". The TQC direction DRL is separately wired via `drl/ensemble.py` and is ACTIVE.
- **Mitigation:** Run `python scripts/startup_drl_truth.py` to see both systems at once. The `DRLAuthorityGate` (from `drl/promotion_gate.py`) is the authority for the TQC direction signal — that's the "is DRL ACTIVE" question.

---

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

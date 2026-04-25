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
| **Exit DRL (Discrete SAC)** | **SHADOW** | Third DRL alongside the TQC direction DRL (P28). Authority cap = EXIT_ONLY; never decides direction. v1 = 4 actions {HOLD, PARTIAL_EXIT, RELEASE_RUNNER, EXIT_ALL}. **v2 checkpoints (200ep, seed=42) in `models/exit_drl_v2/{ASSET}/exit_sac_best.pt` — all clear 0.70 spec target:** BTC val_align=0.730, ETH=0.710, SOL=0.746. Runtime: `agents/exit_drl_agent.py` + `agents/exit_drl_outcome_ledger.py`. Per-tick `.predict()` from agent_signals tick block on active positions; logs to `data/exit_drl_shadow.jsonl` + per-trade outcomes to `data/exit_drl_outcome_ledger.jsonl`. SHADOW mode never influences `execution/exit_alpha.py` triggers. Offline validator (`training/exit_drl/validate_against_baseline.py`) reports Sharpe lift vs rule-based baseline → **BTC +50.0%, ETH +83.0%, SOL +91.3% (vs +10% threshold — all clear, but absolute Sharpe is negative for both actors → DRL "loses less" rather than wins).** Promotion gate `risk/exit_drl_promotion_gate.py` (read-only): two remaining blockers before EXIT_ONLY → (a) ≥30 shadow days, (b) ≥30 closed exit events. Currently 0/30 on both — accumulates after deploy. |

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

`signals/authority_fusion.py` declares **25 agents** in `AUTHORITY_MATRIX_NORMAL`.
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
  - `training/exit_drl/validate_against_baseline.py` — replays held-out 20% of timeline through Exit-SAC + a pure-Python mirror of `execution/exit_alpha.py`'s rule-based triggers (phase/CRACK/momentum/drawdown/stop-out). Per-asset Sharpe lift: **BTC +50.0%, ETH +83.0%, SOL +91.3%** (vs +10% threshold — all pass). **Caveat: absolute Sharpe is *negative* for both actors on the held-out window. DRL "loses less", doesn't generate alpha.** Held-out window covers a tough crypto period (mostly drawdowns); DRL holds longer and avoids stop-out cascades the rule-based actor triggers. In a trending market the comparison may invert. Validator output: `data/exit_drl_validation/{ASSET}_validation.json`.
  - `agents/exit_drl_outcome_ledger.py` — per-trade outcome ledger writer. `record_open(asset, entry_price, direction)` on position open, `record_prediction(asset, action_name, confidence, unrealized_pnl, bars_held)` on every Exit-SAC prediction during the trade, `record_close(asset, exit_price, exit_reason, realized_pnl_bps)` on close. Flushes one JSONL line per closed trade to `data/exit_drl_outcome_ledger.jsonl`. Wired into the per-tick predict block at [main.py:7457](main.py#L7457).
  - `risk/exit_drl_promotion_gate.py` — read-only gate. `evaluate(asset)` returns `{would_promote, blockers, evidence, thresholds}`. Thresholds: ≥30 shadow days, ≥30 closed exit events, +10% Sharpe lift, HOLD ratio ∈ [50%, 90%], EXIT_ALL ratio ≤ 30%. **Current state (offline):** Sharpe lift threshold passes for all 3 assets; remaining blockers are shadow days (0 < 30) and exit events (0 < 30) — both accumulate after deploy.
- **Promotion path remaining (post-shadow):**
  1. Deploy and let the agent shadow-run for 30+ days, accumulating ≥30 closed exit events per asset in `data/exit_drl_outcome_ledger.jsonl`.
  2. Re-run `validate_against_baseline.py` on the *most recent* held-out window (recompute lift on a market regime closer to live conditions — the current +50/+83/+91% lift was on the 2024-2025 drawdown window).
  3. Run `ExitDRLPromotionGate.evaluate_all()` — confirm `would_promote=True` for the asset(s) you want to promote.
  4. Only then: wire `ExitDRLAgent.predict()` into `execution/exit_alpha.py`'s TRIGGER 4 (`DRL_ACTION`) by exposing a `DRLOutput`-compatible bridge (System 3's prediction stands in for the dormant System 2). Flip `ExitDRLMode.SHADOW` → `ExitDRLMode.EXIT_ONLY` only for the asset(s) the gate clears.
  5. Add a kill switch: same flip in reverse on any of `(consecutive_losses ≥ 5)`, `(7-day Sharpe vs baseline drops below +0%)`, or `(action distribution drifts outside HOLD ∈ [50%, 90%])`.
- **Mitigation:** When `startup_drl_truth.py` is extended for a fourth DRL family, add a System-N row here. Sharing a single gate across two DRLs would create cross-coupling. **Don't promote on offline-only validation alone** — the live shadow phase is what catches "DRL learned a regime that doesn't exist in production" failures (a known offline-RL failure mode per the spec's Stage 4 pitfall #6).

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

# HMATS — Project Status & Development Guidelines

**Last updated:** 2026-04-22
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
| **Execution shadow** | ACTIVE | Re-enabled 2026-04-24 after self→ctx bugs fixed + AC-2 snapshot. Watch `data/shadow_exec_comparison.jsonl` for CRITICAL MISMATCH rate before cutover |

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

### P4. ETH fold_1 stale checkpoint
- **Symptom:** ETH TQC `results.json` reports `best_fold: fold_1` with reward=1400, but `train_rows=0, train_time=0`.
- **Reality:** fold_1 is an aborted/stale run; fold_3 (reward=1029, train_rows=10028) is the genuine best.
- **Mitigation:** `BEST_FOLDS` in `drl/ensemble.py` + `drl/runtime_obs_builder.py` + `training/drl/oracle_tqc_teacher.py` all hardcode ETH→fold_3. `results.json` updated to best_fold=fold_3. **Check `train_rows>0` on every fold before trusting reward.**

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
- **Data age** uses exchange timestamp, `MAX_DATA_AGE_SECONDS=10.0`
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

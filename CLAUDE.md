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
| **kraken_quant (12 strat)** | ACTIVE (ADVISE) | Per-strategy stats in `data/kq_firing_stats.json` |
| **onchain_sol** | ACTIVE | Singleton agent, `.start()` dispatched in `run_live()` as well as `run_paper()` |
| **Binance WS (micro)** | ACTIVE | taker flow + mark price for cross-exchange microstructure |
| **Discord alerts** | ACTIVE | webhook in `.env` → `DiscordLogHandler` forwards ERROR/CRITICAL + 4H heartbeat |
| **Attribution tracker** | ACTIVE | 16-agent coverage (see §Authority Matrix) |

**Verification commands** (run these when in doubt):
```bash
# ---------- TRUTH-LEVEL (static, no runtime needed) ----------
# Is DRL really ACTIVE? Check all config declarations + instantiated classes:
python -X utf8 scripts/startup_drl_truth.py
# Expected: DRLAuthorityGate.get_authority() = ACTIVE; is_shadow_mode() = False
# If config says ACTIVE but runtime says SHADOW → volume mount bug (P1)

# Which agent/ classes are actually wired into main.py?
python -X utf8 scripts/startup_agent_wiring_truth.py
# Expected: ~120/142 ACTIVE. Dead classes = legacy code in agents/quant_agent.py
# (real quant decision is in data_mgmt/market_data_pipeline.py Best-of-N).

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
| 18 | kraken_quant | ADVISE | kq_direction, kq_confidence | fusion + attribution |
| 19 | microstructure | ADVISE | micro_imbalance, micro_confidence, micro_direction | fusion + attribution |
| 20 | model_alpha | ADVISE | model_alpha_direction, model_alpha_weight | fusion + attribution |
| 21 | onchain_graph (SOL) | ADVISE | onchain_graph_direction, onchain_graph_confidence | fusion + attribution |
| 22 | options | ADVISE | options_short_confirmation, options_confidence | fusion |
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

### P9. The quant DECIDE agent is NOT in agents/quant_agent.py
- **Symptom:** Looking for the "quant agent" class, finding `agents/quant_agent.py`, assuming it's the DECIDE signal source. Trying to edit its behavior has zero runtime effect.
- **Reality:** `agents/quant_agent.py` is orphan legacy code (only referenced by `core/runtime_spine.py`, which itself is not on the live call path). The actual quant DECIDE signal comes from the Best-of-N strategy selector in `data_mgmt/market_data_pipeline.py:1244`.
- **Mitigation:** When you need to modify quant behavior, edit `data_mgmt/market_data_pipeline.py` (strategy fitness, confidence formula, alpha gate). `startup_agent_wiring_truth.py` will correctly flag quant_agent.py as DEAD.

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

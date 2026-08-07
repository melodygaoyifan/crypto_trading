# HMATS — Hierarchical Multi-Agent Trading System

[![test-suite](https://github.com/melodygaoyifan/crypto_trading/actions/workflows/test-suite.yml/badge.svg)](https://github.com/melodygaoyifan/crypto_trading/actions/workflows/test-suite.yml)
[![codebase-invariants](https://github.com/melodygaoyifan/crypto_trading/actions/workflows/codebase-invariants.yml/badge.svg)](https://github.com/melodygaoyifan/crypto_trading/actions/workflows/codebase-invariants.yml)

Production live-trading system for **BTC / ETH / SOL on Kraken**. Multi-agent decision pipeline with GMM regime detection, three independent DRL systems, 25-agent authority fusion, and seven layers of regression-protected safety.

> **Project status**: live on Hetzner CPX21, single-operator deployment. Not a general-purpose library — private codebase, conventions are HMATS-specific.

---

## Table of Contents

- [What this is](#what-this-is)
- [Architecture at a glance](#architecture-at-a-glance)
- [Runtime state](#runtime-state)
- [Decision pipeline](#decision-pipeline)
- [Three DRL systems](#three-drl-systems)
- [Safety layers](#safety-layers)
- [Test + CI infrastructure](#test--ci-infrastructure)
- [Repository discipline](#repository-discipline)
- [Quick start](#quick-start)
- [Training](#training)
- [Deployment](#deployment)
- [Operating runbook](#operating-runbook)

---

## What this is

A single-asset-class (crypto spot) execution system that:

- Subscribes to **Kraken REST + WebSocket** for OHLCV, orderbook, taker flow.
- Runs ~25 specialised agents per 4H tick (quant Best-of-N, three DRL systems, sentiment LLM, on-chain, options, microstructure, volatility-alpha, etc.).
- Fuses signals through a **non-weighted authority matrix** (DECIDE / ADVISE / CONFIRM / VETO / CAP / EXECUTE) — see [`signals/authority_fusion.py`](signals/authority_fusion.py).
- Enforces seven gate layers (Constitution alpha gate, P0 Safety, Trade Gate, Existence Fuse, Risk Veto, Cascade Governor, Dead-Man Switch) before any order leaves the box.
- Self-monitors via 89+ regression tests, 7 baseline-diff CI scanners, and a chaos harness that replays prior production incidents.

If you're trying to learn how the system works, **start with [`CLAUDE.md`](CLAUDE.md)** — it's the canonical operational reference, kept current with every non-trivial change.

---

## Architecture at a glance

```
                                          ┌─────────────────┐
                                          │  Discord Alerts │
                                          └────────▲────────┘
                                                   │ CRITICAL/ERROR + 4H heartbeat
┌──────────────┐   ┌────────┐   ┌──────────────┐   │   ┌────────────────┐
│  Kraken WS   │──▶│ Market │──▶│  ~25 Agents  │───┼──▶│  Fusion Engine │
│  + REST      │   │  Data  │   │  (per asset, │   │   │  (authority    │
│  + Coinglass │   │ Pipe   │   │   per tick)  │   │   │   matrix)      │
│  + onchain   │   └────────┘   └──────┬───────┘   │   └────────┬───────┘
└──────────────┘                       │           │            │
                                       │ data_quality           │ TradeIntent
                                       │ markers (P126)         ▼
                                       │           │   ┌────────────────┐
                                       │           │   │  7 Gate Layers │
                                       │           │   │  (Constitution │
                                       │           │   │   → Existence  │
                                       │           │   │   Fuse → Trade │
                                       │           │   │   Gate → ...)  │
                                       │           │   └────────┬───────┘
                                       ▼           │            ▼
                              ┌────────────────┐   │   ┌────────────────┐
                              │  decision_     │   │   │  Execution     │
                              │  trace_watcher │◀──┘   │  Manager       │
                              │  (anomaly      │       │  (slicer +     │
                              │   alerts)      │       │   pre-flight   │
                              └────────────────┘       │   min-size +   │
                                                       │   bal-clamp)   │
                                                       └────────┬───────┘
                                                                ▼
                                                       ┌────────────────┐
                                                       │  Kraken        │
                                                       │  (real orders) │
                                                       └────────────────┘
```

Decision tick: every 4H bar. Per-tick latency: ~7s for the agent fan-in, ~1s for fusion + gates, ~0.5–2s per order placement (slicer-aware).

---

## Runtime state

| Component | State | Notes |
|---|---|---|
| **DRL (TQC)** | ACTIVE | Best-fold per asset (`fold_3` for all three); `models/retrained/{ASSET}/fold_3/`. |
| **Sentiment L1 (Fear & Greed)** | ACTIVE | Deterministic engine. |
| **Sentiment LLM (Haiku)** | ACTIVE | CryptoPanic + CC News blend. |
| **Quant Best-of-N** | DECIDE | 4 strategies: mean-revert / momentum / volume-breakout / vrp + hold. |
| **kraken_quant (12 strats)** | DECIDE | Promoted full-weight 2026-04-22; per-strategy stats in `data/kq_firing_stats.json`. |
| **Exit DRL (Discrete SAC)** | SHADOW (demoted 2026-08-07, P199) | Predictions logged only. The old "kill switch auto-demotes" claim had been false since 2026-04-30 (should_demote returns None unconditionally); the promotion evidence was a negative-Sharpe lift vs a strawman baseline. |
| **OnChain (BTC/ETH)** | DISABLED by config | `OnChainSentimentAlphaEngine.enabled=False`. |
| **Soldex (SOL DEX arb)** | ACTIVE | 100% confidence emitter, 0% direction in normal regime. |
| **Discord alerts** | ACTIVE | Webhook in `.env`. |

For exhaustive runtime state + agent-by-agent authority breakdown, see **[CLAUDE.md → Authority Matrix](CLAUDE.md#authority-matrix-v68)**.

---

## Decision pipeline

```
agent_signals dict   ──▶   _build_fusion_signals (integration_v36.py:2024)
                              │
                              ▼
                          AuthorityFusionEngine.fuse (signals/authority_fusion.py:402)
                              │
                              ├── Layer 1: NO_TRADE / DATA_INVALID short-circuits
                              ├── Layer 2: VETO check (Risk, BitBeast)
                              ├── Layer 3: DECIDE (Quant + DRL + kraken_quant)
                              ├── Layer 4: CONFIRM (Regime, two_stage, structure)
                              ├── Layer 5: ADVISE (Sentiment, LLM, Whale, OnChain, ...)
                              └── Layer 6: CAP (Macro, Risk leverage)
                              │
                              ▼
                          FusionResult (direction, target_exposure, vetoes)
                              │
                              ▼
                          7-Gate Chain → TradeIntentV36
```

**Key invariant**: signal keys must match across writer / fusion / attribution / extractor (P2/P3/P8). Adding an agent requires touching all four sites or the agent silently zero-fires. See [`scripts/agent_attribution_validate.py`](scripts/agent_attribution_validate.py) for the structural validator.

---

## Three DRL systems

HMATS runs three independent DRL components — easy to confuse, important not to.

| System | Module | Authority | Purpose |
|---|---|---|---|
| **TQC direction** | `drl/ensemble.py` | DECIDE (ACTIVE) | Per-asset 122-feature → 126-dim obs → quantile critic; primary directional signal alongside Quant. |
| **DRL Agent (legacy)** | `agents/drl_agent.py` | DISABLED | P10 tranche/exit scaffolding; instantiated but `mode=DISABLED`. Kept for future enable-path. |
| **Exit-SAC** | `models/exit_drl_v2/` + `core/tick_exit_triggers.py` | SHADOW (all 3, P199) | Discrete SAC; inference + shadow logging only. NOTE: the kill switch's should_demote() has returned None unconditionally since 2026-04-30 — there is NO auto-demotion; re-promotion requires a clean retrain + forward evidence. |

The TQC and Exit-SAC use independent observation spaces, model files, and promotion gates. Don't conflate them when grepping for "DRL".

---

## Safety layers

```
1. Constitution alpha gate         — no trade without alpha > 1.5×(fee+slip+latency)
2. P0 Safety Integrator            — kill switch, stale data guard, rate limiter
3. Trade Gate (defense/trade_gate) — 7 sub-gates (data health, freshness, DVOL,
                                      DRL constraints, volume, structure, governors)
4. Existence Fuse                  — 28d rolling window; -15% / -18% / -15% / -18% /
                                      10-loss-streak triggers system halt + manual recovery
5. Risk Veto Classifier            — HARD threshold (DD ≥ 20%, corr ≥ 0.98, dvol ≥ 5σ)
                                      always blocks; SOFT caps exposure in OPPORTUNITY mode
6. Authority Fusion VETO layer     — Risk + BitBeast can override DECIDE agents
7. Dead-Man Switch                 — Kraken CancelAllOrdersAfter, 60s timeout
```

Plus order-layer pre-flights:
- **PREFLIGHT_BELOW_MIN_SIZE** (P91 + P127) — every market/limit/stop order checked against `exchange.market(symbol)['limits']['amount']['min']` before sending.
- **PREFLIGHT_WRONG_SIDE** — stop-loss SELL price must be below market; BUY above.
- **INSUFFICIENT_SPOT_BALANCE** — every order checked against `fetch_balance()['free']` (P86/P87) before placement.
- **STOP-DEDUP** — `fetch_open_orders` ground-truth dedup before placing new stop (P98).

All four return `PERMANENT` from the error classifier — no retry storm on local validation failures (P79/P93).

---

## Test + CI infrastructure

The codebase has two CI workflows + 89+ tests + 7 baseline-diff scanners.

### Tests (`.github/workflows/test-suite.yml`)

| Suite | Purpose |
|---|---|
| `tests/test_invariants_p111.py` | Property-style invariants for every P-fix shipped (Sharpe, Kelly, tranche, NaN, userref, PREFLIGHT, authority matrix). |
| `tests/chaos/` | Chaos harness — replays the actual production cascades (P85 restart loop, P95 userref drift, P92 state corruption, etc.). |
| `tests/test_property_invariants.py` | Hypothesis property tests (~1500 random cases) against `_classify_kraken_order_error`, `_compute_effective_weekend_confidence`, `RiskVetoClassifier.classify`, `AuthorityFusionEngine.fuse`. |
| `tests/test_replay_fusion.py` | Golden-trace replay — 217 frozen production attribution records → fusion engine → snapshot diff on any output change. |
| `tests/test_external_api_fuzz.py` | Hypothesis fuzz against the 5 `_parse_raw_data` feed parsers (malformed JSON / wrong types / NaN). |
| `tests/test_trade_gate_coverage_p121.py` + `_p124.py` | Behavioral test for every concretely-firing `RejectReason`. |
| `tests/test_mutation_audit_p113.py` + `_p122.py` | Manual mutation audit — re-introduce documented bug shapes, assert tests catch them. |
| `tests/test_concurrent_stress.py` | RLock + asyncio.gather + atomic-write contention tests. |
| `tests/test_market_minsize_p127.py` | P127 regression — pre-flight min-size on market+limit + slicer cap. |

Full suite runs in **~6 seconds** on CI.

### Scanners (`.github/workflows/codebase-invariants.yml`)

Seven baseline-diff scanners under `tools/`:

| Scanner | Catches | Baseline file |
|---|---|---|
| `authority_consistency_audit.py` | Authority matrix drift (writer/fusion/attribution/extractor mismatch). | `authority_consistency_baseline.json` |
| `silent_failure_audit.py` | New silent-failure patterns (try/except: pass, dict.get without check). | `silent_failure_baseline.json` |
| `lint_silent_swallow.py` | New `try/except: pass` or `: logger.debug` blocks. | `silent_swallow_baseline.json` |
| `lint_naive_datetime.py` | New `datetime.utcnow` or bare `datetime.now()` in dataclass defaults. | `naive_datetime_baseline.json` |
| `lint_self_config_undefined.py` | Classes that read `self.config` without setting it. | `self_config_undefined_baseline.json` |
| `lint_mypy_baseline.py` | New mypy errors by error code. | `mypy_baseline.json` |
| `lint_signal_freshness.py` | New BLIND `agent_signals[X] = ...` writers (no per-key freshness marker). | `signal_freshness_baseline.json` |

**Semantics**: counts can DECREASE freely; INCREASE blocks CI. Operators chip away at existing findings during normal work; new code can't add silent regressions without an explicit baseline bump.

### Coverage gap analyzer

[`tools/decision_trace_coverage.py`](tools/decision_trace_coverage.py) — walks the call graph from `defense/trade_gate.py` and reports which `RejectReason` outcomes are reachable in source but not asserted in any test.

---

## Repository discipline

This repo follows three forcing functions documented in [CLAUDE.md](CLAUDE.md):

1. **Re-pull before every non-trivial commit.** The operator and Claude often edit in parallel; CLAUDE.md P85 documents a 10-restart cascade caused by a parallel-edit collision.
2. **Defensive `getattr(obj, 'attr', sentinel)` on every cross-module attribute read.** P85 lesson: a missing attribute on a third-party object should not exit the process — `restart: always` weaponises a single AttributeError into a 6-minute outage.
3. **`grep "def <name>"` before adding any non-trivial helper method.** P87 lesson: 2,400-line modules silently shadow methods; collisions surface only at call time as `TypeError: takes from N to M positional arguments`.

The deploy script ([`scripts/hetzner_deploy.sh`](scripts/hetzner_deploy.sh)) runs the local CI gate (Step 0) before pushing — catches scanner regressions before 30s of container churn.

---

## Quick start

```bash
# Verify mode — proof logs only, no orders
python -X utf8 main.py --mode verify

# Paper trading (simulated execution against live data)
python -X utf8 main.py --mode paper

# Live trading (REQUIRES explicit confirmation flag)
python -X utf8 main.py --mode live --confirm-live --config configs/live_high_risk.json
```

### Verification commands

```bash
# Truth-level: is DRL really ACTIVE? (no runtime needed)
python -X utf8 scripts/startup_drl_truth.py

# Per-agent attribution validator (requires live JSONL)
python -X utf8 scripts/agent_attribution_validate.py <signals_YYYYMMDD.jsonl>

# 9-axis production health monitor (one-shot)
bash scripts/hmats_monitor.sh
```

### Run the test suite locally

```bash
python -X utf8 -m pytest tests/ -v
python -X utf8 tools/ci_check_invariants.py
```

---

## Training

```bash
# DRL (TQC) — per asset, ULTIMATE preset (~6 hours per fold per asset on a 5090)
python -X utf8 -u training/train_drl_full.py --asset BTC --no-progress-bar
python -X utf8 -u training/train_drl_full.py --asset ETH --no-progress-bar
python -X utf8 -u training/train_drl_full.py --asset SOL --no-progress-bar

# Decision Transformer (TQC teacher pretrain + finetune)
python -X utf8 -u training/drl/train_decision_transformer_v32.py \
    --asset BTC --extra-assets ETH,SOL --oracle-mode tqc_teacher \
    --epochs 300 --save-suffix _pretrain

# Data prep
python -X utf8 training/fetch_binance_full.py
python -X utf8 training/scripts/rebuild_pipeline.py --smooth 2
```

Iron rules for DRL training (see [`memory/drl_training_rules.md`](https://example.invalid/) — local file): `ent_coef` must be a fixed float (auto causes gradient explosion); `n_quantiles=24` for TQC; SubprocVecEnv unreliable on Windows, use `DummyVecEnv`.

---

## Deployment

Hetzner CPX21 + Docker Compose. Models on a separate volume (`hmats-models`), state on `hmats-data`, logs on `hmats-logs`.

```bash
# Deploy from local — runs local CI gate (Step 0), builds image, restarts containers
bash scripts/hetzner_deploy.sh hmats

# Health check after deploy
ssh hmats 'docker ps && docker logs hmats-engine --since 10m | grep -E "TQC loaded|HEALTH_S|HEARTBEAT"'

# Force rebuild (after a code change that must affect runtime)
ssh hmats 'cd /home/hmats/hmats/app && \
    docker compose -f docker-compose.hetzner.yml build hmats-engine && \
    docker compose -f docker-compose.hetzner.yml up -d --force-recreate hmats-engine'
```

**Volume gotcha (P1)**: `docker-compose.hetzner.yml` declares volumes as `external: true` with explicit `name:` so Docker doesn't create project-prefixed copies. If `docker inspect hmats-engine` shows `app_hmats-models` instead of `hmats-models`, the deploy will silently load empty model checkpoints.

For full deploy procedure see [`README_DEPLOY_HETZNER.md`](README_DEPLOY_HETZNER.md).

---

## Operating runbook

| Symptom | First check | Doc reference |
|---|---|---|
| `[STOP-RETRY] PERMANENT` log spam | Free spot balance vs Kraken min-size | CLAUDE.md P91 / P127 |
| `[POSITION-DESYNC]` CRITICAL alert | `fetch_positions` vs `paper_positions.json` | CLAUDE.md P93 |
| `STATE_CORRUPTION_DETECTED` halt | `risk/auto_recovery_gate.py` halt-state file integrity | CLAUDE.md P92 |
| 0 trades in 7+ days, system "running normally" | Re-run gate-rejection forensics; **don't add patches** | CLAUDE.md "Trade Frequency Reality Check" |
| Restart loop, RestartCount > 0 | `journalctl -u docker.service` for cascade pattern; check parallel-edit conflicts | CLAUDE.md P85 |
| Replay snapshot diff in CI | Re-baseline if intentional: `python -X utf8 tests/test_replay_fusion.py --update-snapshot` | CLAUDE.md P116 |
| Scanner baseline INCREASE blocking deploy | Either fix the new finding or re-baseline: `python -X utf8 tools/ci_check_invariants.py --update` | CLAUDE.md P72 |

---

## Environment

```bash
cp .env.example .env

# Required for any mode
KRAKEN_API_KEY=
KRAKEN_API_SECRET=

# Required for full feature set
FRED_API_KEY=
COINGLASS_API_KEY=
LUNARCRUSH_API_KEY=
CRYPTOPANIC_API_KEY=
HF_TOKEN=

# Optional knobs
HMATS_INITIAL_CAPITAL=10000
HMATS_REGIME_WARN_COOLDOWN_SEC=
HMATS_G6_SHADOW=
HMATS_ENABLE_AGGRESSIVE_ALLOCATOR=
HMATS_AGGRESSIVE_ALLOCATOR_MIN_FILLS=
HMATS_ALLOW_TEST_SIMULATORS=

# Discord alerts
DISCORD_WEBHOOK_URL=
```

`main.py` calls `dotenv` at import time so values in `.env` override config file defaults.

---

## Tech stack

- **Python 3.12+**
- **ccxt** for Kraken REST/WS, **pandas / numpy / ta** for features
- **stable-baselines3 (TQC)** for direction DRL, **PyTorch** for SAC + DT + Sentiment LLM
- **scikit-learn** GMM (per-asset, BIC-optimized k=7-8)
- **Hugging Face transformers** (DeBERTa v3-base for SentimentL2)
- **pydantic v2** for config schema
- **pytest + hypothesis** for tests
- **Streamlit** dashboard
- **Docker Compose** on Hetzner CPX21

---

## License

Private / All rights reserved. Code is published for transparency and CI; no permission granted for re-use.

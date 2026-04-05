# HMATS: Hierarchical Multi-Agent Trading System
## White Paper v6.8.0

**System**: HMATS — Hierarchical Multi-Agent Trading System
**Assets**: BTC, ETH, SOL on Kraken
**Decision Frequency**: 4-hour candles
**Account Target**: $10K seed → $100K growth

---

## 1. Executive Summary

HMATS is a fully autonomous cryptocurrency trading system that combines deep reinforcement learning (DRL) with rule-based quantitative strategies through a hierarchical multi-agent architecture. The system trades BTC, ETH, and SOL on Kraken using a 4-hour decision cycle, with a philosophy of **"Aggressive Alpha, Defensive Shell"** — pursuing strong alpha in the signal layer while maintaining strict risk controls in the defense layer.

The system features:
- **5 specialized agents** with authority-based fusion (DECIDE / VETO / ADVISE / PENALIZE)
- **Per-asset GMM regime classification** (BTC k=8, ETH k=7, SOL k=7)
- **TQC deep reinforcement learning** with FiLM-conditioned LSTM (126-dim obs, 8-frame stacking)
- **8 independent veto sources** (one-veto-kill architecture)
- **4-tier drawdown control** with existence fuse (system self-halt)
- **Dead-man switch** (server-side Kraken CancelAllOrdersAfter API)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HMATS v6.8.0 Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              Data Layer (6 Sources)                        │   │
│   │   Kraken API / Coinglass / Fear & Greed / CryptoCompare  │   │
│   │   Solana RPC / Jito MEV                                   │   │
│   │   Contract & Data Health Gate (Fail-Closed, MAX_AGE=60s) │   │
│   └─────────────────────────┬─────────────────────────────────┘   │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              Analysis Layer                                │   │
│   │   Per-Asset GMM Regime Classifier (6-8 regimes)           │   │
│   │   Phase Detector (IGNITION → EXPANSION → SATURATION →    │   │
│   │                    EXHAUSTION)                             │   │
│   │   Regime Smoother (persistence=2, hysteresis)             │   │
│   │   Lead-Lag Engine (Binance → Kraken, 2-tier dampening)   │   │
│   │   Bull Transition Detector (4-condition state machine)    │   │
│   └─────────────────────────┬─────────────────────────────────┘   │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              Agent Layer (5 Agents)                        │   │
│   │   Quant Agent (DECIDE) — Best-of-N: 4 strategies         │   │
│   │   DRL Agent  (DECIDE) — TQC LSTM-FiLM, 126-dim obs      │   │
│   │   Sentiment  (ADVISE) — F&G L1 + Haiku L3                │   │
│   │   Short Bias (PENALIZE) — Funding rate weighted           │   │
│   │   Risk Agent (VETO) — One-veto-kill authority             │   │
│   └─────────────────────────┬─────────────────────────────────┘   │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              Fusion Layer                                  │   │
│   │   Authority Fusion (5-agent weighted consensus)           │   │
│   │   Reliability Injection (per-strategy confidence)         │   │
│   │   Deadlock Resolution (ALL_CONFLICT → NO_TRADE)          │   │
│   └─────────────────────────┬─────────────────────────────────┘   │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              Risk Layer (8 Veto Sources)                   │   │
│   │   Constitution / Risk Manager / Dead-Man Switch           │   │
│   │   Squeeze Protection (3-tier) / Leverage Guard (3x max)  │   │
│   │   Drawdown Control (4-level: 10/15/25/35%)               │   │
│   │   Correlation Crisis (5-state controller)                 │   │
│   │   Existence Fuse (weekly/monthly/consecutive)             │   │
│   └─────────────────────────┬─────────────────────────────────┘   │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              Execution Layer                               │   │
│   │   Passive-Aggressive Executor (post-only → market)       │   │
│   │   Timing Engine → Dynamic Slicer → Impact Calibration    │   │
│   │   Fill Rate Logging (slippage/fill time/repricing)       │   │
│   └───────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Signal Generation

### 3.1 Quant Agent — Best-of-N Strategy Selection

Four independent strategies compete per tick. The **strongest signal wins** based on regime-conditional fitness weights:

| Strategy | Best Regime | Signal Source |
|----------|-------------|---------------|
| **Mean Revert** | QUIET_ACCUMULATION (1.3x) | RSI extremes + Bollinger Band position |
| **Momentum** | MOMENTUM_RALLY (1.3x) | EMA cross + MACD with ADX amplification |
| **Volume Breakout** | PANIC_SELLOFF (1.2x) | MACD + volume confirmation |
| **VRP** | EXTREME_VOLATILITY (1.3x) | RSI + volume + BB compression |

In consolidation regimes (QUIET_ACCUMULATION, WEAK_CONSOLIDATION), momentum strength is capped to prevent lagging EMA_200 signals from dominating. EMA weight is reduced from 60% to 30% in these regimes, with MACD (a leading indicator) receiving 70%.

### 3.2 DRL Agent — TQC with FiLM-Conditioned LSTM

The DRL agent uses Truncated Quantile Critics (TQC) with a custom feature extractor:

| Parameter | Value |
|-----------|-------|
| Algorithm | TQC (stable-baselines3-contrib) |
| Extractor | FiLM Position A (166K params) |
| Net Architecture | [384, 384, 256] (~1.03M total params) |
| Observation Space | 126 dims (122 features + 4 env state) |
| Frame Stacking | n_stack=8 (32h temporal context) |
| Action Space | Continuous Box(-1, 1) |
| Training | 3-fold time-series CV, 2.5M-3M steps/fold |

**Feature composition** (122 features):
- 102 base TA features (OHLCV-derived)
- 8 GMM regime probabilities (per-asset, zero-padded)
- 7 external features (Coinglass funding/OI/liquidations)
- 5 wavelet-denoised features (Coiflet-4 soft thresholding)

**4 environment state features**: position_ratio, position_direction, pnl_ratio, drawdown

**FiLM conditioning**: Regime probabilities modulate LSTM features via Feature-wise Linear Modulation (γ·x + β), allowing the policy to adapt behavior per regime without separate models.

**Training configuration** (Config 1 Optuna winner, 3-fold validated):
- ent_coef = 0.1 (fixed — "auto" causes gradient explosion)
- learning_rate = 1.5255e-5
- buffer_size = 500K (n_stack=8 memory constraint)
- n_quantiles = 24, reward_clip = 20
- Real transaction costs embedded in env.step() and reward

### 3.3 Sentiment Agent — Three-Layer Architecture

| Layer | Component | Status | Output |
|-------|-----------|--------|--------|
| L1 | Deterministic Engine (6 signals) | Active | Weighted composite score |
| L2 | DeBERTa v2.2 | Pending retrain | Per-asset direction |
| L3 | Haiku LLM (CryptoPanic headlines) | Active | Per-asset narrative sentiment |

**L1 signal weights**: Funding Rate (25%), Long/Short Ratio (20%), Fear & Greed (15%), OI Change (15%), Liquidations (15%), DVOL+VPIN (10%).

**Iron Rule #34**: Sentiment never vetoes trades, never flips direction. Maximum modulation: ±10% sizing, ±30% urgency. Sentiment is a modulator, not a decision-maker.

**Fallback chain**: L3 Haiku → L1 F&G → 0.0 (never blocks the tick).

---

## 4. Authority Fusion

The system uses an **authority matrix** instead of traditional weighted averaging:

| Agent | Authority | Behavior |
|-------|-----------|----------|
| Quant Agent | **DECIDE** | Primary signal source, ~45% fusion weight |
| DRL Agent | **DECIDE** | Secondary signal, ~55% fusion weight (1.3x confidence boost) |
| Sentiment | **CONFIRM** | Modulates confidence (halves exposure if misaligned) |
| Short Bias | **PENALIZE** | Soft ×0.7 penalty on longs (funding arb: >0.24%/8h → short +15%) |
| Risk Agent | **VETO** | One-veto-kill (can block but not increase exposure) |

When multiple DECIDE agents disagree, the system uses **confidence-weighted consensus**. If agreement < 0.3, confidence is reduced proportionally. Complete conflict → NO_TRADE (never hangs in deadlock).

---

## 5. Risk Management

### 5.1 Eight Veto Sources (One-Veto-Kill)

| Veto Source | Trigger | Action |
|-------------|---------|--------|
| Constitution | Parameter violation | Block trade |
| Risk Manager | Risk limits exceeded | Block trade |
| Dead-Man Switch | Heartbeat timeout (60s) | Cancel all orders |
| Squeeze Protection | Score ≥ 0.50/0.70/0.80 | Warn / Reduce 50% / Flatten |
| Leverage Guard | > 3.0x leverage | Reject / Clip |
| Drawdown Control | 4 levels | 10%→reduce, 15%→halt, 25%→pause, 35%→kill |
| Correlation Crisis | 5-state controller | SPIKING→-25%, CRISIS→-50% |
| Existence Fuse | Multi-layer protection | See below |

### 5.2 Existence Fuse (System Self-Halt)

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Consecutive losses | 5 trades | Pause 24h |
| Weekly loss | ≥ 8% | HALT (manual recovery) |
| Monthly loss | ≥ 10% | KILL (system shutdown) |
| Max drawdown | ≥ 35% | System-level kill switch |

### 5.3 CRACK Thesis Exit (Tiered)

When the CRACK weight (cross-asset alignment × kinetic momentum × volume confirmation) was active (>0.50) and decays:

| Tier | Trigger | Exit |
|------|---------|------|
| 1 | crack < 0.50 | 25% partial exit |
| 2 | crack < 0.45 | 50% partial exit |
| 3 | crack < 0.35 | 100% full exit |

### 5.4 Bull Transition Detector

4-condition state machine to prevent slow bleeding during mild bull markets:

**Conditions**: BTC Golden Cross (MA50 > MA200), SOL/BTC relative strength > 0, 7-day positive funding streak, rising OI + falling liquidations.

| State | Short Restriction |
|-------|-------------------|
| INACTIVE (0 conditions) | None |
| POTENTIAL (1 condition) | None |
| ACTIVE (2+ conditions) | Short exposure ×0.2-0.5 |
| CONFIRMED (5+ days) | **Block all naked shorts** |

### 5.5 Dead-Man Switch

Server-side Kraken `CancelAllOrdersAfter` API with 60-second timeout. Heartbeat refreshed every 24 seconds. If the system crashes or loses connectivity, Kraken automatically cancels all open orders after 60 seconds.

---

## 6. Execution

### 6.1 Dual Time-Scale Control

- **4H decision loop**: `process_4h_tick()` — sole trade decision entry point
- **200ms execution sub-loop**: PA executor repricing (reserved, no direction changes)

### 6.2 Passive-Aggressive Executor

| Mode | Behavior |
|------|----------|
| PASSIVE | Post-only limit order, wait for fill |
| AGGRESSIVE | 120s timeout → cancel & market order |
| ABORT | Cancel, do not execute |

Post-only orders (`oflags='post'`) avoid taker fees. Partial fills ≥50% accepted. Maker reprice up to 3 attempts with improving spread.

### 6.3 Position Sizing Pipeline

Exposure passes through 10+ sequential multipliers with a **minimum viable floor** (0.5% of equity). If stacked multipliers reduce position below this floor, the trade is rejected as uneconomical. Force-execution exits (stops, gambler, abort) bypass the floor.

---

## 7. Adaptive Feedback

### 7.1 Failure Memory

Tracks OPPORTUNITY mode trade outcomes. Consecutive losses → caution mode (density_boost increases entry threshold, tranche_delay slows escalation). Learns from both full and partial exits.

### 7.2 Confidence Scorer

Per-strategy × per-regime confidence tracking:
- Direction accuracy (35%)
- PnL vs expected (35%)
- Regime accuracy (30%)

Low confidence (< 0.35) → conviction multiplied by 0.3 (soft downgrade). Persisted across restarts.

### 7.3 Drift Detection

5-source model drift monitoring:
- Feature distribution (z-score)
- Latent space (distance threshold)
- GMM regime (Jensen-Shannon divergence)
- Execution slippage (z-score)
- DRL action (mean shift)

Drift weight multiplier applied to DRL confidence in fusion. Critical drift → DRL demoted to EXIT_ONLY.

### 7.4 OOD Detection

Mahalanobis distance detector fitted on training features. When observations fall outside the training distribution:
- **Soft degradation**: confidence_mult applied (0.1-1.0)
- **Hard switch**: ≥6 consecutive OOD steps → DRL to EXIT_ONLY

---

## 8. Regime Classification

### 8.1 Per-Asset GMM

| Asset | GMM k | Regimes |
|-------|-------|---------|
| BTC | k=8 | STEADY_UPTREND, NEUTRAL_DRIFT, QUIET_ACCUMULATION, WEAK_CONSOLIDATION, MOMENTUM_RALLY, PANIC_SELLOFF, VOLATILE_CHOP, EXTREME_VOLATILITY |
| ETH | k=7 | Similar, zero-padded to 8 |
| SOL | k=7 | Similar, zero-padded to 8 |

12 features per model, full covariance, reg_covar=1e-2. Confidence cap removed — only distribution-shift guard (>30% features |z|>3σ → ADX fallback).

### 8.2 Regime Smoother

Hysteresis threshold = 3, min persistence = 2 bars. Reduces regime flip rate by ~30%. Applied identically in training and runtime.

### 8.3 Regime-Conditional Parameters

| Regime | Leverage | Position Size | Alpha Gate | Strategy Fitness |
|--------|----------|---------------|------------|-----------------|
| MOMENTUM_RALLY | 2x | 1.0x | Permissive | Momentum 1.3x |
| PANIC_SELLOFF | 2x | 1.0x | Permissive | Vol Breakout 1.2x |
| VOLATILE_CHOP | 3x | 1.0x | Standard | Vol Breakout 1.1x |
| QUIET_ACCUMULATION | 1x | 0.8x | Selective | Mean Revert 1.3x |
| WEAK_CONSOLIDATION | 1x | 0.6x | Selective | Mean Revert 0.7x |
| EXTREME_VOLATILITY | 1x | 0.5x | Restrictive | VRP 1.3x |

---

## 9. Alpha Gate (Fee-Aware)

Every trade must pass a fee-aware profitability check:

```
Effective threshold = max(min_alpha_floor, friction × multiplier)

Where friction = fee_bps + slippage_bps + latency_bps + margin_bps
```

Monthly volume tracking with Kraken Pro free tier ($10K/month):
- Below $10K: zero effective fees
- $10K-$12K: linear blend
- Above $12K: full Kraken taker (26 bps) / maker (16 bps)

| Mode | Free Tier Threshold |
|------|---------------------|
| NORMAL | 14 bps |
| OPPORTUNITY | 8 bps |

---

## 10. Training Pipeline

### 10.1 End-to-End Stages

```
Stage 0-6:   Feature engineering + Architecture search    ✅
Stage 7:     Reward mode selection (classic NAV%)         ✅
Stage 8A/B:  Optuna hyperparameter optimization           ✅
Stage 9:     Real friction + OOD integration              ✅
Stage 10:    Full TQC training (3 assets × 3 folds)       ✅
Stage 11-12: DT v3.2 + Ensemble (DT in SHADOW mode)      ✅
Stage 13-15: Validation + Runtime parity + Deployment     ✅
Stage 16:    Paper trading baseline                        In progress
Stage 17-18: Live deployment + DRL promotion              Pending
```

### 10.2 Iron Rules (35 Total)

Critical constraints that must never be violated:

- **#5**: ent_coef = 0.1 (fixed float, NEVER "auto")
- **#7**: batch=256 + grad_steps=4 (validated combination)
- **#9**: DummyVecEnv only (SubprocVecEnv deadlocks at ~1.47M steps)
- **#10**: Deploy best_model.zip (EvalCallback), not final_model
- **#11**: window_size = 10 (not 96)
- **#17**: GMM scaler (12-dim) independent from Feature scaler (~114-dim)
- **#29**: Transaction costs in both env.step() (NAV) and reward (learning signal)
- **#34**: Sentiment only modulates, never vetoes or flips direction (±10%/±30%)
- **#35**: Online DRL only in shadow mode; promotion requires Sharpe>+0.1, DD<+2%, p<0.05

---

## 11. Deployment Architecture

### 11.1 Live Deployment Phases

| Phase | Timeline | Allocation | Leverage | Key Constraint |
|-------|----------|------------|----------|----------------|
| Phase 1 | Day 7 | 50% max exposure | 2x | DMS 60s, daily loss 3% |
| Phase 2 | Day 14 | 100% max exposure | 3x | Auto-demotion on 3 consecutive losses |

### 11.2 Safety Infrastructure

- **Dead-Man Switch**: Kraken CancelAllOrdersAfter, 60s timeout, heartbeat every 24s
- **Startup Reconciler**: Position/order/balance verification before trading (fail-closed in LIVE mode)
- **Shadow Ledger**: Every intent, fill, and position change recorded to JSONL audit trail
- **Health Monitor**: 10-source data quality tracking with automatic degradation policies

### 11.3 Hardware Requirements

| Component | Runtime | Training |
|-----------|---------|----------|
| CPU | 8+ cores | 8+ cores |
| RAM | 16GB+ | 128GB (n_stack=8 buffer) |
| GPU | Not required | RTX 5090 (24GB VRAM) |
| Storage | 100GB SSD | 200GB SSD |
| OS | Windows 11 (native) | Windows 11 (native) |

---

## 12. Key Metrics & Parameters

| Parameter | Value |
|-----------|-------|
| Initial Capital | $10,000 |
| Decision Frequency | 4 hours |
| Max Leverage | 3.0x |
| Hard Drawdown Kill | 35% |
| DRL Observation Dims | 126 (122 features + 4 env) |
| DRL Frame Stack | 8 (32h temporal context) |
| DRL ent_coef | 0.1 (fixed) |
| GMM Regimes | BTC:8, ETH:7, SOL:7 |
| Regime Smoother Persistence | 2 bars |
| Alpha Gate Floor | 5-14 bps (mode-dependent) |
| Maker / Taker Fee | 16 / 26 bps |
| Dead-Man Timeout | 60 seconds |
| Existence Fuse Window | 28 days |

---

## 13. Competitive Advantages

1. **Authority-based fusion** replaces naive weighted averaging — agents have explicit permissions (DECIDE/VETO/ADVISE), preventing catastrophic consensus failures.

2. **Per-asset regime classification** with GMM instead of a single global regime — each asset has its own regime model trained on its specific market microstructure.

3. **FiLM-conditioned DRL** adapts policy behavior per regime without separate models — a single 1.03M-parameter network serves all 6-8 regimes per asset.

4. **Real friction in training** — transaction costs embedded in both environment dynamics and reward signal, producing models that learn optimal trade timing rather than frequency.

5. **Multi-layer existence protection** — the system can halt itself at 5 different severity levels, from 24h pause to permanent shutdown, before human intervention is needed.

6. **Tiered CRACK thesis exit** — structured position unwinding based on thesis strength decay, not just stop-losses. Three tiers (25%/50%/100%) preserve capital while allowing thesis recovery.

---

*HMATS v6.8.0 — Hierarchical Multi-Agent Trading System*
*Built for Kraken Pro | BTC/ETH/SOL | 4H Decision Cycle*

# HMATS v6.8.0 — Hierarchical Multi-Agent Trading System

Algorithmic crypto trading system for **BTC, ETH, SOL** on Kraken. Multi-agent architecture with GMM regime detection, DRL (TQC) reinforcement learning, Best-of-N strategy selection, and layered risk management.

## Architecture

```
DATA → REGIME → AGENTS → FUSION → EXECUTION → RISK → FEEDBACK
  │       │        │        │          │         │        │
  ▼       ▼        ▼        ▼          ▼         ▼        ▼
Kraken  GMM      Quant   Authority   Order    Governor   PnL
OHLCV   6-regime Best-N  Chain       Router   (Veto)     Attribution
Futures DRL(TQC) Risk    Fusion      Maker    Fuse       Reflection
OnChain Smoother Sentiment          Post-only
```

## Quick Start

```bash
# Verification (proof logs only, no trades)
python -X utf8 main.py --mode verify

# Paper trading (simulated execution)
python -X utf8 main.py --mode paper

# Live trading (REQUIRES explicit confirmation)
python -X utf8 main.py --mode live --confirm-live
```

## Key Components

| Layer | Module | Role |
|-------|--------|------|
| Entry point | `main.py` | Tick loop, agent orchestration |
| Decision spine | `integration/integration_v36.py` | Authority fusion, mode resolution |
| Config | `configs/cloud_production.json` | Production parameters |
| Agents | `agents/` | Quant, DRL, Risk, Sentiment, Short Bias, Volatility Alpha, Whale Detector |
| Core | `core/` | Execution, risk governor, authority chain, account sync |
| Analytics | `analytics/` | PnL attribution, failure memory, confidence scoring |
| Defense | `defense/` | Constitution, trade gate, P0 safety |
| Training | `training/` | DRL (TQC), Decision Transformer, Sentiment, GMM |
| Data | `data_mgmt/` | Market data pipeline, feeds (Kraken, Coinglass, CryptoCompare, etc.) |

## Agent Authority Matrix

| Agent | Authority | Notes |
|-------|-----------|-------|
| Quant (Best-of-N) | DECIDE | Primary signal — regime-fitted strategy selection |
| DRL (TQC) | DECIDE | Entry + exit via fusion weights when ACTIVE |
| Risk | VETO | Can reduce/block, never increase exposure |
| Sentiment | ADVISE | Modulates confidence only |
| Short Bias | PENALIZE | Soft ×0.7 penalty for longs |

## Safety

- **Constitution** — No trade without alpha gate pass
- **P0 Safety** — Kill switch, stale data guard, rate limiter
- **Existence Fuse** — 28d window, -5% PnL → system halt
- **Dead-man Switch** — Kraken CancelAllOrdersAfter, 60s timeout
- **DRL Auto-demotion** — 5 consecutive losses or 15% DD → EXIT_ONLY for 3 days

## Training

```bash
# DRL (TQC) — per asset, ULTIMATE preset
python -X utf8 -u training/train_drl_full.py --asset BTC --no-progress-bar
python -X utf8 -u training/train_drl_full.py --asset ETH --no-progress-bar
python -X utf8 -u training/train_drl_full.py --asset SOL --no-progress-bar
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

- `KRAKEN_API_KEY` / `KRAKEN_API_SECRET` (required)
- `FRED_API_KEY`, `COINGLASS_API_KEY`, `LUNARCRUSH_API_KEY`, `CRYPTOPANIC_API_KEY` (required for full feature set)
- `HMATS_INITIAL_CAPITAL` (default: 10000)

## Tech Stack

- Python 3.12+
- ccxt (Kraken), stable-baselines3, PyTorch
- GMM regime classification (per-asset, BIC-optimized)
- TQC (Truncated Quantile Critics) for DRL
- Streamlit dashboard

## License

Private / All rights reserved.

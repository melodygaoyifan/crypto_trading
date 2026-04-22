# HMATS — Project Status & Development Guidelines

## Project Overview
HMATS v6.8.0 — Hierarchical Multi-Agent Trading System for SOL, ETH, BTC on Kraken.

**Entry point:** `main.py`
**Decision spine:** `integration/integration_v36.py`
**Config:** `configs/cloud_production.json`

---

## Paper Trading Progress (Feb 11)

- **GMM Retrained (Feb 8):** 6 regimes, 721 bars/asset, reg_covar=1e-2. Confidence cap REMOVED — only distribution-shift guard (>30% features |z|>3σ → ADX fallback)
- **RegimeSmoother:** persistence=2, flip rate 30%→14%, eliminates ping-pong. Applied identically in training + runtime
- **Best-of-N Strategy:** 4 independent strategies (mean_revert, momentum, volume_breakout, vrp). Strongest signal wins per regime fitness
- **Alpha Gate:** Dynamic volume-aware thresholds. Free tier (<$10K/mo): NORMAL=14bps, OPPORTUNITY=8bps. Standard: NORMAL=66bps, OPPORTUNITY=38bps
- **DRL Ultimate Training (Feb 11):** In progress — BTC 6.8%, ETH 5.8%, SOL 4.9%. ULTIMATE preset: ent_coef=0.1 fixed, 4-fold CV, position_direction feature (+1 dim = 126 total)
- **v6.7 Deployment:** Regime leverage, maker orders, sentiment L1, cash-and-carry, funding arb all active in paper run

---

## Safety Architecture

### DRL Authority Policy
- **Mode: FULL** — DRL participates in both entry and exit decisions via Authority Fusion when in ACTIVE mode
- **Auto-demotion:** Consecutive loss threshold (5) or drawdown trigger (15%) → demotes to EXIT_ONLY for 3 days
- **Current status (2026-04-22):** **ACTIVE.** TQC val-period backtest showed Sharpe +9.22 (BTC), +7.32 (ETH), +10.29 (SOL) — validated that deterministic TQC policy generalizes to truly-unseen post-2026-02-27 window. Promoted via one-shot manual `promote("ACTIVE")` after verifying `models_ready=3` in the production container.
- **DO NOT downgrade to SHADOW again** unless auto-demotion fires. Running without DRL ACTIVE throws away the Sharpe-+9 alpha source we already trained.
- **Docker-volume gotcha:** TQC models are in external volume `hmats-models`. Compose-managed `app_hmats-models` is incorrect and was silently empty — that's why DRL kept booting at SHADOW (`models_ready=0`). `docker-compose.hetzner.yml` now declares volumes as `external: true` to prevent the regression.
- **Config:**
  ```
  DRL_AUTO_DEMOTE = {
      'enable': True,
      'consecutive_loss_threshold': 5,
      'drawdown_trigger_pct': 0.15,
      'demote_to': 'EXIT_ONLY',
      'recovery_period_days': 3,
  }
  ```

### Authority Matrix
| Agent | Authority | Notes |
|-------|-----------|-------|
| Quant (Best-of-N) | DECIDE | Primary signal source via regime-fitted strategy selection |
| DRL Agent | DECIDE | Entry + exit signals via fusion weights (FULL authority when ACTIVE) |
| Risk Agent | VETO | Can reduce/block but not increase exposure |
| Sentiment Agent | ADVISE | Modulates confidence, does not generate independent signals |
| Short Bias Agent | PENALIZE | Soft penalty ×0.7 for longs (was hard veto, converted v6.5.2) |

### Non-Negotiable Rules
1. **Constitution is supreme** — No trade without alpha gate pass
2. **P0 Safety cannot be bypassed** — kill switch, stale data guard, rate limiter
3. **Existence Fuse** — 28d window, -5% PnL → system halt, manual recovery only
4. **DRL Authority:** DRL participates in both entry and exit decisions when in ACTIVE mode. Auto-demotion to EXIT_ONLY triggers on consecutive losses or drawdown threshold
5. **Single exchange** — Kraken only (Binance/Deribit in legacy/)

---

## Completed Work

| # | Item | Status |
|---|------|--------|
| 1 | GMM Retrained | 6 regimes, 721 bars/asset, cross_asset_correlation default fixed (0.65→0.87) |
| 2 | RegimeSmoother | persistence=2, flip rate 30%→14%, wired in training + runtime |
| 3 | GMM Confidence Cap | REMOVED hard cap. Only distribution-shift guard (>30% features |z|>3σ → ADX fallback) |
| 4 | Best-of-N Strategy | Replaced fixed-weight composite with 4 independent strategies |
| 5 | Alpha Gate Calibration | Dynamic volume-aware: NORMAL ×2.0, OPPORTUNITY ×1.15. Free tier: 14/8bps |
| 6 | Dead-man switch | Kraken CancelAllOrdersAfter API, 60s timeout, refreshed each tick |
| 7 | Cancel-on-disconnect | HeartbeatWatchdog → handle_disconnect() → cancel_all |
| 8 | Tick exception handling | process_4h_tick split into wrapper + _process_4h_tick_inner |
| 9 | Disconnect race | Early exit at tick start + mid-tick check before execution |
| 10 | Proof log memory cap | `List[str]` → `deque(maxlen=1000)` |
| 11 | Reconnect sync | handle_reconnect() reconciles positions with exchange |
| 12 | MEAN_REVERT power | 0.0→0.3 (BB/RSI extremes work best in mean-revert) |
| 13 | Size cap per asset | MAX_EXPOSURE_FRACTION: BTC/ETH 25%, SOL 20% |
| 14 | Warmup logging | First 2 ticks per asset flagged in proof log |
| 15 | Cache age enforcement | BTC cache expires after 300s |
| 16 | Crowd strictness | Pre-computed before false breakout detection |
| 17 | Tranche cold-start | constitution.py skips abort check when position.level == NONE |
| 18 | DeadlockResolution enum | Fixed CONTINUE→NONE in integration_v36.py |
| 19 | Profit max adapter wiring | sentiment/macro/crowd params now passed from main.py |
| 20 | GMM regime→regime_state | _predict_gmm_regime() stores regime in raw["regime_state"] |
| 21 | Per-asset price history | Fixed FLASH_CRASH false positive (was shared across assets) |
| 22 | DRL training infra | ent_coef fixed (TQC_1=0.2, TQC_2=0.1), GPU params, new regime names |
| 23 | Risk Governor daily loss | Fixed false veto: RiskManager.initialize(balance) + guard |
| 24 | Volume Collapse abort | Fixed enter→abort→flatten loop: position-age + vol validity guard |
| 25 | Short-bias → soft penalty | veto_direction removed, longs penalized ×0.7 (not blocked). Funding arb override ×0.95 |
| 26 | Regime-conditional leverage ⏳ | VOLATILE_CHOP=3x, MOMENTUM_RALLY/PANIC_SELLOFF=2x, others=1x. Kraken isolated margin |
| 27 | Maker orders (post-only) ⏳ | oflags='post' + postOnly=True, 120s timeout, partial fill ≥50% accepted |
| 28 | Dynamic alpha gate ⏳ | Volume-aware friction: <$10K free tier (0 fees), >$10K standard Kraken fees |
| 29 | Compounding ⏳ | account_sync.update_dry_run_pnl() wired after paper trades |
| 30 | Sentiment L1 ⏳ | SentimentFeed(alternative.me) + DeterministicSentimentEngine in tick loop |
| 31 | Cash-and-carry ⏳ | Delta-neutral module, signal-only Phase 1 (Kraken Futures not wired) |
| 32 | Funding rate arb ⏳ | ShortBiasAgent: funding>0.24%/8h → short +15%, funding<-0.16%/8h → long ×0.95 |
| 33 | Deployment configs ⏳ | live_phase1.json (Day 7, half pos, 2x lev) + live_phase2.json (Day 14, full, 3x) |
| 34 | DRL training ⏳ | training/train_drl_full.py: ULTIMATE preset, 3-fold, position_direction feature, 126 dims, early stopping, Optuna |

---

## Development Guidelines

### When Making Changes

1. **Read before edit.** Always read existing code before modifying.
2. **Match patterns.** Follow the existing code style and naming conventions.
3. **Check call sites.** When modifying method signatures, verify ALL callers pass the new params.
4. **Three trade_gate paths.** main veto_chain, authority_chain, AND p0_safety_integrator ALL call trade_gate.check() — fix ALL three.
5. **Test with verify mode.** `python -X utf8 main.py --mode verify` for quick smoke test.
6. **Test with paper mode.** `python -X utf8 main.py --mode paper` for live data validation.
7. **Use `-X utf8`** on Windows to avoid GBK encoding issues.
8. **Quoted paths.** Use `Set-Location -LiteralPath` for paths with parentheses (e.g., `training`).
9. **Track verification status.** Use these markers in Completed Work:
   - (no marker) = verified in production
   - ⏳ = code deployed, awaiting production verification

### Key Architecture Rules

- **ent_coef must be fixed float** — "auto" causes gradient explosion (0.2→10^23→NaN)
- **TQC_1 = 0.2, TQC_2 = 0.1** — proven stable for RTX 5090. ULTIMATE preset uses 0.1
- **RegimeSmoother must match** between training (train_tqc.py) and runtime (main.py)
- **GMM feature defaults must match training distribution** — e.g., cross_asset_correlation=0.87, spread_percentile per-asset (BTC=5, ETH=8, SOL=12 bps)
- **Constitution schema** requires `dvol_zscore`, `vpin`, `correlation_btc_eth_sol`, `orderbook_depth_1pct_usd`
- **Data age** uses exchange timestamp, MAX_DATA_AGE_SECONDS=10.0
- **pandas_ta broken** on Python 3.14 — use `ta` library instead
- **DRL state space** = 126 dims (122 features + 4 env state including position_direction)

### Training Commands (Ultimate)

```bash
# All assets use training/train_drl_full.py with ULTIMATE preset
python -X utf8 -u training/train_drl_full.py --asset BTC --no-progress-bar
python -X utf8 -u training/train_drl_full.py --asset ETH --no-progress-bar
python -X utf8 -u training/train_drl_full.py --asset SOL --no-progress-bar
```

### Monitoring

- **Health monitor:** `python scripts/paper_run_monitor.py`
- **Training:** Watch for NaN in loss/rewards, verify ent_coef stays fixed (0.1)
- **Paper run logs:** `logs/proof_log_*.log`, `data/shadow_ledger/*.jsonl`

---

## Pre-Live Checklist

- [ ] DRL Ultimate training completes (all 3 assets, 4 folds each) — BTC fold_2 in progress
- [ ] DRL models deployed + 30 shadow trades for promotion
- [x] Paper run produces trades on tick 2+ consistently — verified Feb 13
- [x] Shadow ledger shows correct PnL tracking — FILL records with price/fee/realized_pnl, verified Feb 13
- [ ] Health monitor shows no CRITICAL alerts for 24h
- [x] Dead-man switch verified with Kraken API — SET/REFRESH/DISABLE all pass, verified Feb 13
- [x] StartupReconciler sub-flags enabled — paper: balance=True, live: all=True, verified Feb 13
- [x] Regime leverage verified in proof logs — MOMENTUM_RALLY=2x visible in proof logs
- [x] Sentiment L1 scores visible in proof logs — F&G=50 (neutral) visible
- [x] live_phase1.json tested with Kraken API — 29/29 tests pass, balance=$10,565, all tickers OK, verified Feb 13

---

## Known Non-Critical Errors (Paper Run)

- ~~`GateDecision.EXIT_ONLY` attribute missing~~ — **RESOLVED**: member exists in trade_gate.py:35
- ~~`KrakenRateLimitManager.can_proceed` missing~~ — **RESOLVED**: method exists in rate_limit_manager.py:338
- ~~`ReconciliationResult.__init__() missing 'status'`~~ — **FIXED**: added default `status=PENDING` in startup_reconciler.py

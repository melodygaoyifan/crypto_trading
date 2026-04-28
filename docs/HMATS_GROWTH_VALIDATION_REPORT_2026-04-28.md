# HMATS Growth Validation Report — 2026-04-28

**Mode**: READ-ONLY diagnostic. No code modified.
**Scope**: 30-day live performance + IC framework + capacity + self-learning components.
**Data sources**: `data/trade_attribution.jsonl`, `analytics/ic/reports/`, `logs/ic_signals/`, `data/kraken_plus_monthly_volume.json`, source code grep.

---

## Executive Summary

| Hypothesis | Verdict | One-line evidence |
|---|---|---|
| **H1** "We need more strategies" | **REJECTED** | The 12 kraken_quant strategies we already have are firing direction=0 / confidence=0 in 100% of sampled signals — adding more strategies before fixing the silent ones is premature. |
| **H2** "We should add more coins" | **REJECTED** | Volume utilization is **22.6% of the $10K free tier** AND all 3 existing assets are loss-making (ETH +$45 alpha, BTC −$151, SOL −$115 over 90 lifetime trades). Adding a 4th coin (8.5–12.5 person-days, high refactor risk) doesn't fix the alpha problem; it dilutes attention. |
| **H3** "System lacks self-learning" | **PARTIAL** | Most components are ALIVE. The single REAL gap is `strategy_aging.py`: `record_signal()` IS called per tick but `record_outcome()` is NEVER called → weight modifiers stuck at 1.0 → strategy effectiveness silently never evaluated. This is a "wire one existing callback" fix, not "build new self-learning system". |

**Bottom line**: All three architectural changes the user is considering are **the wrong response to the data**. The right response is **fix-first**: close 4 specific dead loops (~5 person-days total) that would surface root cause for the current losses and unblock the 12 dormant strategies before adding any new code.

---

## Phase 0 — Live Trading Baseline (last 30 days)

### 0.1 Filled trades + per-asset breakdown

| Asset | Trades (30d) | Net PnL | Avg/Trade | Notional | Win rate |
|---|---|---|---|---|---|
| BTC | 5 | −$79.27 | −$15.85 | $4,973 | 20% |
| ETH | 5 | +$27.29 | +$5.46 | $3,646 | 60% |
| SOL | 7 | −$55.01 | −$7.86 | $3,585 | 14% |
| **TOTAL** | **17** | **−$106.99** | **−$6.29** | **$12,204** | **23.5%** |

Source: `data/trade_attribution.jsonl` (17 closed trades in window).

### 0.2 Performance metrics

- **Win rate**: 23.5% (4/17)
- **Avg win**: $7.46 | **Avg loss**: −$10.53
- **Win:Loss ratio**: 1 : 1.4 (unfavorable — losses are 40% larger than wins)
- **Fee drag**: $36.34 fees / $106.99 net loss = **34% of loss attributable to fees**
- **Max drawdown** (% basis): **INSUFFICIENT DATA** — `equity_history.jsonl` not available locally; need production sync

### 0.3 Regime distribution

`regime_at_entry` field is **empty string in all 17 trade records** → cannot validate per-regime PnL. **OBSERVABILITY GAP filed as P0 fix below.**

### 0.4 Volume capacity

- 30-day notional: **$2,260.66** (per `data/kraken_plus_monthly_volume.json`)
- Kraken free tier: **$10,000** (`defense/trade_gate.py:418`)
- **Utilization: 22.6%** — plenty of headroom on existing 3 assets. Free-tier ceiling is NOT the bottleneck.

### 0.5 Signal-to-trade conversion

**INSUFFICIENT DATA** — Production `proof_log_*.log` not synced locally. Visible from in-window data:
- 2,000+ signal rows sampled in `ic_signals_*.jsonl`
- Quant + DRL signals present + firing
- **kraken_quant signals: 0% firing rate** (direction=0, confidence=0 on all sampled rows)
- 17 actual fills → ~0.85% conversion if every signal counted; real number requires production veto-log access

---

## Phase 1 — Strategy Library Health

### 1.1 IC Heatmap (24-bar / 96h horizon)

Source: `analytics/ic/reports/ic_report_20260425_0623.json`

| Asset | quant.direction IC | RSI IC | Verdict |
|---|---|---|---|
| BTC | 0.0127 (p=0.128) | **−0.0296 (p=0.042)** | Quant noise; **RSI inverted** |
| ETH | **0.0207 (p=0.016)** | **−0.0175 (p=0.014)** | Quant marginal-pass; **RSI inverted** |
| SOL | 0.0303 (p=0.039) | **−0.0429 (p=0.028)** | Quant weak-pass; **RSI inverted** |

**Critical finding: RSI shows pervasive negative IC across ALL THREE ASSETS.** This is the same P40/P41 sign-flip family from CLAUDE.md history — except the existing fix may not cover all paths since signal quality remains WEAK.

### 1.2 Strategy classification

#### kraken_quant (12 strategies)

| Class | Count | Strategies |
|---|---|---|
| Healthy | 0 | — |
| Sign-flipped | 0 | — |
| Aged out | 0 | — |
| **Untested** | **12** | All 12 (BEAR / BULL / SIDEWAYS × 4 each) |

**Evidence**: `ic_signals_*.jsonl` shows `kraken_quant.direction=0.0` + `kraken_quant.confidence=0.0` on **every sampled record**. The 12-strategy library is wired (matrix row 18 = DECIDE) but not producing signals.

#### Best-of-N (4 core + HOLD)

| Strategy | Status | Evidence |
|---|---|---|
| mean_revert | **HEALTHY** | SOL Bollinger IC=0.051; recent 2026-04-22 tuning boosted MOMENTUM_RALLY fit 0.5→0.9 |
| momentum | **SIGN-FLIPPED** | 0/98 hit rate in MOMENTUM_RALLY → fit reduced 1.3→0.7 (`market_data_pipeline.py:104-106`); lagging EMA/MACD buys the top |
| volume_breakout | **UNTESTED** | No dedicated IC signal in report |
| vrp | **UNTESTED** | No dedicated IC signal in report |
| hold | **ACTIVE** | 85% paper win rate when ADX<15 + RSI neutral |

### 1.3 Strategy correlation

**INSUFFICIENT DATA** — pairwise correlation matrix not computed. `analytics/ic/` has per-signal records; would need a one-time analysis pass over the BACKFILL files.

### 1.4 Inference

**The right interpretation is NOT "we need more strategies".** It is:

1. **12 strategies already exist and are dormant** — find out why kraken_quant signals are 0/0 across the board. Could be: signal-key wiring (P2/P3), authority gate, regime detection, or threshold misconfiguration.
2. **RSI sign-flip across all assets** — same family as P40/P41 from CLAUDE.md but apparently unresolved at the IC level.
3. **Best-of-N has only ONE healthy strategy** (mean_revert). Momentum is documented as sign-flipped + reduced. The remaining 2 are untested. **Fix the wiring/sign-flip/untested-coverage before considering library expansion.**

---

## Phase 2 — Coin Expansion

### 2.1 Capacity utilization

**22.6% of $10K free tier** ($2,260 / $10,000 monthly notional). Runway at current pace: ~132 days to hit cap.

### 2.2 Per-coin lifetime alpha (90 trades)

| Asset | Trades | Win rate | Net PnL | Gross alpha | Notional | Alpha/$1 |
|---|---|---|---|---|---|---|
| BTC | 32 | 12.5% | −$259.91 | −$151.17 | $23,157 | −$0.0065 |
| ETH | 31 | 29.0% | −$300.90 | **+$45.43** | $62,597 | **+$0.0007** |
| SOL | 27 | 7.4% | −$368.50 | −$115.00 | $36,803 | −$0.0031 |

**ETH is the only asset with positive gross alpha contribution.** SOL has the worst win rate (7.4%) and largest avg loss. **No asset is dominant on PnL; all three are losing.**

### 2.3 Cost of adding a 4th coin

| Area | Effort (person-days) | Risk |
|---|---|---|
| GMM retrain (per-asset model) | 1.0–1.5 | Low |
| obs_dim=126 (no impact, per-asset scaler) | 0.5 | Low |
| Cross-asset correlation recalibration | 1.5–2.0 | Medium |
| Microstructure thresholds (DVOL/VPIN) | 0.5–1.0 | Low–Medium |
| DRL retrain (4th TQC, ~8h wall + tuning) | 3.0–4.0 | Medium–High |
| Veto chain coverage (~99 hardcoded `["BTC", "ETH", "SOL"]` sites across ~40-50 files) | 2.0–3.0 | **High** |
| **TOTAL** | **8.5–12.5 person-days** | **Medium–High** |

### 2.4 Inference

**Adding a 4th coin solves zero current problems**:
- We're not capacity-constrained (22.6% utilization)
- Existing assets are losing → adding a 4th distributes the loss across 4 books, doesn't fix it
- 99 hardcoded asset locations → adding without refactoring first creates 40+ silent regression risks
- 8–12 days of high-risk work for *negative expected value* given the current loss rate

**The right response: fix existing-coin alpha first.** ETH is closest to positive (gross alpha +$45 over 31 trades) — if we figure out why ETH works and BTC/SOL don't, that's the highest-leverage signal.

---

## Phase 3 — Self-Learning Component Status

### 3.1 Component status table

| Component | Code | Hot path | Output consumed | Loop closed | Verdict |
|---|---|---|---|---|---|
| **DT (Decision Transformer)** | Y | N | N | N | **DEAD** (intentional — grid search showed no value vs TQC) |
| **TQC + StatisticalPromotionGate** | Y | Y (`main.py:7401` per tick) | Y (`integration_v36.py:780,2189` fusion) | Y (authority level cascades to `agent_signals['drl_authority_level']`) | **ALIVE** |
| **strategy_aging.py** | Y | Y (`main.py:8280` records signals) | **N** (`get_weight_modifiers()` never called from hot path) | **N** (`record_outcome()` NEVER called → effectiveness never computed) | **SHADOW (DEAD LOOP)** |
| **IC Framework** | Y | Y (live logging to `logs/ic_signals/`) | Partial (logs, but no continuous compute → fusion) | N (analytics-only by design per `ic_logger.py:28-31`) | **SHADOW** |
| **GMM regime classifier** | Y | Y (`market_data_pipeline.py:950` per tick per asset) | Y (TQC `regime=` param + REGIME_STRATEGY_FIT lookup) | Y (RegimeSmoother persistence=2) | **ALIVE** |

### 3.2 The one real dead loop

**strategy_aging.py is the smoking gun.**

- `record_signal()` IS called per tick at `main.py:8280` ✓
- `record_outcome()` is **NEVER called anywhere in the codebase** ✗
- Without outcomes, `_evaluate_strategy()` never crosses `min_signals_for_assessment=20` threshold
- Without evaluation, `get_weight_modifiers()` returns `1.0` for every strategy forever
- Without modifiers, the system has **no automatic feedback** from PnL to strategy weights

**This is exactly the "feels like missing self-learning, but it's actually one missing callback" gap** the prompt warned about. The component was built; the closing wire was never connected.

### 3.3 Gap classification

| Gap | Type | Cheapest fix | Cost |
|---|---|---|---|
| strategy_aging outcome feedback | **Reader-writer mismatch** (P15-shape) | Wire `aging.record_outcome(...)` into trade-close path in `core/execution_service.py` (or wherever `record_trade_completed` fires) | ~3 days |
| IC framework feedback | **Scheduling gap** | Cron `analytics/ic/compute_ic.py --all-assets` hourly + wire output into agent confidence discount | ~1 day |
| DT shadow → promotion | **By design** (grid search showed no value) | No fix — intentionally DEAD | 0 |

**None of these are "missing algorithm" gaps. All three are wiring/scheduling.**

---

## Recommendations (priority by ROI)

### P0 — fix this week (~5 person-days)

1. **Diagnose why kraken_quant 12 strategies aren't firing** (1–2 days investigation, READ-ONLY first).
   - Grep the kraken_quant agent invocation site + `is_valid` flag + signal-key wiring (P2/P3 family).
   - Per CLAUDE.md P115, kraken_quant was classified `SILENT_DATA_QUIET` with confirmed-active wiring + 0/12 strategy fires in chop regime over a SHORT window. If 30+ days still shows 0 fires, the strategies' regime-fit thresholds are likely wrong, not the wiring.
   - **Outcome**: either 12 dormant strategies become active (large alpha unlock) OR we know they're genuinely dead and can decide whether to retune thresholds vs. retire.

2. **Wire strategy_aging.record_outcome()** (~3 days).
   - Find `record_trade_completed` / `pnl_attribution` close path in `core/execution_service.py` or `analytics/trade_attributor.py`.
   - Add `aging.record_outcome(strategy_name, was_correct_direction, pnl_bps)` call.
   - Add `weight_modifiers = aging.get_weight_modifiers()` reader in best-of-N + kraken_quant strategy selectors.
   - **Outcome**: closes the real self-learning loop the operator was sensing.

3. **Add `regime_at_entry` field population in trade_attribution** (~0.5 day).
   - Currently empty string on all 17 trades.
   - Without this, future audits cannot answer "are we losing in specific regimes?".
   - **Outcome**: future audits become DATA-DRIVEN instead of INSUFFICIENT DATA.

4. **Investigate RSI sign-flip across all 3 assets** (~1 day).
   - IC report shows persistently negative RSI IC. Either: signal generator has wrong sign convention, OR RSI is genuinely inverted in current crypto regime (mean-revert dominance).
   - If sign convention bug: 1-line fix unlocks 3 assets.
   - **Outcome**: mid-cost diagnostic with potentially huge upside.

### P1 — next sprint (~3 person-days)

5. **IC framework cron wiring** (1 day) — close the IC feedback loop.
6. **Compute strategy correlation matrix from BACKFILL files** (0.5 day) — answers the "12 strategies vs N independent alpha sources" question.
7. **Add `equity_history.jsonl` sync to local audit script** (0.5 day) — observability.

### Don't do (with reasons)

| Action | Why not |
|---|---|
| **Add new strategies** | 12 already exist + are dormant. Adding more adds debugging surface without unlocking alpha. |
| **Add a 4th coin** | 22.6% capacity utilization + all 3 existing coins losing + 8–12 days of high-risk refactoring. Negative EV. |
| **Build a new self-learning component** | The existing `strategy_aging.py` is wired up to the writer side; only the reader side is missing. Building new from scratch repeats existing work. |
| **Touch `obs_dim=126` / `defense/constitution.py` / `training/`** | Iron law — these are the load-bearing invariants that previous P-fixes verified. Any change here without proven need creates large blast radius. |

---

## Appendix — Data quality + audit limitations

- **Production data NOT synced locally**: equity_history, proof_log, signals_* full series, outcomes_* full series. Recommend pulling these via `scp hmats:/var/lib/docker/volumes/hmats-{data,logs}/_data/...` before next audit.
- **Regime labels**: `regime_at_entry` is empty in all trade records → cannot validate H1's per-regime claim quantitatively. Fix #3 above resolves this for future audits.
- **Strategy correlation**: not computed in this pass; would require ~30 min standalone analysis on the BACKFILL JSONLs.
- **kraken_quant per-strategy fire counts**: `kq_firing_stats.json` exists locally but only reflects ~3.6 minutes of uptime since last restart (per CLAUDE.md P115). Need a multi-day production capture.

**Audit confidence**: ~80% on H1+H2 verdicts (data is sufficient to reject), ~70% on H3 (strategy_aging dead-loop verified by source-grep, but full closure of "no other dead loops" needs the production proof_log).

---

*Generated 2026-04-28. READ-ONLY diagnostic. No code or runtime state was modified during this audit.*

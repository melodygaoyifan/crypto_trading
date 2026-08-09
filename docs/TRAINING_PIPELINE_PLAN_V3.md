# Training Pipeline Plan V3 — the operator-spec quant pipeline

**Drafted:** 2026-08-09 (P246 era). Operator direction: one clear pipeline —
EDA → train/test split → model training simple→deep (early stopping, GPU) →
evaluation (overfitting vs underfitting) — per asset, up to SIX models
(three market trends × derivative/spot instrument), goal = find alpha and
beta, research-first, learning from the last three months of live data.
Supersedes the per-experiment ad-hoc scripts as the single reference design.

---

## 1. Objective: alpha AND beta, separately accounted

This is a quant project; realized gain decomposes into two products and the
pipeline must pursue and ACCOUNT for them separately:

- **BETA (exposure timing):** being long the market in regimes that pay for
  it, flat/short when they don't. Everything measured so far says this is
  where the recoverable money is at 4H frequency: bull-regime drift is the
  only significant unconditional return (BTC +13bps/16h t=2.7, SOL +26bps
  t=2.8), and the only two era-stable survivors of five experiment
  families are beta-shaped (trend filter; BTC hold-bull assembly).
- **ALPHA (market-neutral):** funding/basis carry, cross-sectional spreads,
  microstructure. Perp funding carry is REAL cash flow (wired in P245) but
  the funding-DIRECTION signal inverted across eras; carry-as-income while
  holding beta positions is bankable, carry-as-direction is not (yet).

Every shadow ledger and results artifact reports `price_pnl`, `carry_pnl`,
and `cost` as separate columns so alpha and beta claims are auditable.

## 2. What the last three months of LIVE data say (measured 2026-08-09)

- **Sleeve equity:** −5.5% official since 2026-06-14 (anchor $3,997.75 →
  $3,777); July −4.2%, last 14 days −2.6%. The bleed continued after the
  P197–P240 control hardening — the controls bound the risk, they did not
  create edge.
- **90-day per-agent live IC (in-container review, n up to 1,588):** every
  agent ≈ 0. drl 16h −0.073 (t=−1.44); quant −0.042; model_alpha +0.034
  (t=0.38 — its earlier +0.289 window fully decayed); whale +0.033
  (t=0.83); sentiment/funding/llm all ns. **No live signal has 90-day
  predictive value.** This is the live confirmation of the lab's
  era-instability finding, and the reason this plan does not bet on any
  existing signal being "on".
- **Trend regime gate:** forward evidence INVERTED vs its in-sample
  hypothesis (weekly cron, P235) — regime gating promoted on in-sample
  splits would have been wrong; the shadow-first discipline worked.

## 3. Verdict inventory from the five prior experiment families

| family | verdict |
|---|---|
| TQC RL (2 clean campaigns) | 0/9 folds promotable; loses to its own ridge baseline on BTC/ETH |
| supervised single models (protocol + DS pipeline + 196-trial search) | big searches overfit; ridge lockbox PASS later weakened (era-favorable window) |
| bull/bear composite | falsified on pre-design era (only the trend filter survived) |
| regime lab funding cells | CV-excellent, validation-catastrophic (−87%/−162%); small CV gap ≠ era stability |
| **carry-aware BTC assembly (P245)** | **validation +44.1% vs B&H +20.6%, no fragility flags — sole full-pipeline survivor** (window twice-read; forward gate is its exam) |

## 4. Pipeline stages (the one clear pipeline)

**Stage 0 — Data + provenance.** 6y 4H parquets (+ flow cols, full-history
funding), content-hashed; every artifact stamps git commit + data hashes
(`training/provenance.py`). Data-quality validation (row counts, gaps, NaN
budgets) runs before anything trains.

**Stage 1 — EDA (standing, per data refresh).** Per asset × per CELL:
target distributions, per-cell feature IC rankings, funding conditioning,
durations/transitions (`regime_model_lab.py --stage eda` + per-cell
extension). EDA output PRESCRIBES each cell's candidate list — measured
structure in, guessed structure out.

**Stage 2 — Splits (one module: `training/splits.py`).**
- DESIGN ERA [3000, 9100): ALL selection, tuning, early stopping.
- VALIDATION ERA [9100, n): single recorded shot per assembled system;
  window-usage ledger counts every read and surfaces prior spend.
- FORWARD (live shadow, 30d, P166): the only unread window; final arbiter.
- Purged K-fold (embargo 42, horizon 4) inside the design era for CV.

**Stage 3 — The 6-cell matrix per asset.** {bull, bear, peace} ×
{PERP, SPOT}, causal a-priori regime labels (SMA200 × 90d momentum):

| | PERP cell | SPOT cell |
|---|---|---|
| positions | ±1, funding carry credited/charged | long/flat ONLY (no shorts, no carry) |
| costs | 3bps taker (Coinbase CDE) + slip | 16/26bps maker/taker (Kraken) + slip |
| role | full directional + carry alpha | cheap-to-hold beta vehicle; high costs mean LOW-TURNOVER candidates only |

Cells are independent: different model families per cell are expected, not
exceptional. `flat` is a universal candidate and the floor rule stands: a
cell whose best candidate has negative design-era CV realized gain deploys
flat.

**Stage 4 — Model ladder, simple → deep (the capacity ladder IS the
under/overfit diagnosis).** Every cell climbs in order; each rung must beat
the previous rung's CV realized gain to justify its capacity:
1. baselines: flat, hold
2. rules: funding threshold, mean-reversion, dip-buy (few params, grid-CV)
3. linear: ridge family (adaptive refit; the proven generalizer)
4. trees: LightGBM (small: depth ≤3, ≤200 iters)
5. shallow nets: MLP (≤2 layers, early stopping)
6. sequence DL on GPU: GRU / TCN on 8-frame windows — **early stopping on a
   purged validation tail of the design era (patience 8 evals), mixed
   precision, weight decay + dropout, parameter budget ≤ n_cell/10**
7. (only if rung 6 beats rung 5): small transformer

A rung that fails to beat its simpler predecessor STOPS the ladder for that
cell — that is the underfit/overfit verdict in action: if linear beats deep,
the data said the capacity buys overfit (already measured true for forecast
IC on all three assets); if hold beats linear, the cell's alpha is beta.

**Stage 5 — Evaluation (every candidate, no exceptions).**
- standard row: train + CV + (when spent) validation, with the OVERFIT GAP
  printed (`training/eval_report.py`);
- learning curve per DL candidate (train/val loss vs epochs — underfit =
  both high, overfit = divergence; saved as artifact);
- robustness battery: ERA / PARAM / COST fragility flags;
- alpha/beta decomposition: price vs carry vs cost columns.

**Stage 6 — Assembly + validation shot.** Per asset per instrument,
cell winners assemble into the regime-switched book; ONE ledger-recorded
validation shot vs B&H + SMA; battery attached.

**Stage 7 — Forward.** Survivors → in-engine shadow harness (GP2, P219
ledger pattern, parity-tested) → 30d P166 forward gate → operator flip
(P141). Nothing promotes from backtest alone — five families of evidence
say backtest windows cannot certify at this data size.

## 5. GPU plan (RTX 5090)

Dozens of SMALL models, not one big one: batch cells × assets as parallel
processes (models are <1M params; the 5090 is bottlenecked by Python, not
FLOPs — run 4-6 cell-trainings concurrently); mixed precision for the
sequence models; early stopping caps wasted epochs; CPU remains correct for
ridge/LGBM rungs (GPU only enters at rung 5+). Full 6-cell ladder for all
3 assets estimated < 1 GPU-day — vs ~16h per TQC fold under the old design.

## 6. Standing decisions & open items

- **Gate design (operator, still open):** per-fold CI certifies nothing at
  1,964 bars (no baseline passes either); this plan certifies on the
  validation-era shot + forward gate instead. Formal blessing pending.
- **SOL RL:** churn-tier Optuna KILLED 2026-08-09 per operator (2/24 trials
  done; study `hmats_v8_sol_churn` resumes from sqlite if ever revived).
  SOL re-enters through this pipeline like every other asset; RL is rung 8
  at best, and only with a new signal basis.
- **Kraken spot cells** are model-ready but venue-dormant (P152 structural
  flatness); building them keeps the option real without deploying.
- **Research brief: MERGED** (full text:
  `docs/research/METHODOLOGY_BRIEF_2026-08-09.md`). Binding adjustments:
  1. **External corroboration of the funding inversion** — market-wide
     crypto carry Sharpe fell 6.45 → 4.06 (2024) → negative (2025), and
     the ETF-era basis trade compressed to ~zero (BIS WP1087 + 2025/26
     survey). The P244 era-fragility was the market re-pricing, not a lab
     artifact. **Funding = income always, signal never** (until forward
     data says otherwise).
  2. **The alpha bar, arithmetically:** break-even IC ≈ 0.13 per decision
     at 16h vol and 6–13bps/side. No measured in-house IC clears it
     unconditionally — directional research targets only regime-cells that
     could clear it, turnover-reduced expressions, or trend-filter on/off
     conditioning. Sub-breakeven searches are not funded.
  3. **Linear-first is externally settled** (Grinsztajn 2022: trees beat
     nets at ~10k tabular samples; in-house even trees lose to ridge).
     **TCN** is the designated sequence architecture if a net is ever
     justified; transformers are not candidates at this n. Params ≤
     ~n_cell/10, dropout+weight-decay, patience 5–10 with best-weights
     restore, early stopping on POOLED multi-segment validation loss (a
     single fixed val window re-creates P243 inside the training loop).
  4. **Regime conditioning: coarse cells, policy-class switching, and the
     jump-model switch** (persistence-penalty clustering, CV-tuned, judged
     on ONLINE-inferred labels with trade delay — arXiv:2402.05272) as the
     upgrade for SMA200's turn lag. Forecasts prefer regime-as-FEATURE
     pooled models; hard switches only where the policy CLASS differs.
  5. **Model reports gain three missing artifacts:** learning curve,
     capacity-sweep curve (report the plateau, never the argmax), and a
     **DSR/trial-count line** — the pipeline logs every configuration
     evaluated.
  6. **GPU pattern:** `torch.vmap` + stacked module states for batched
     small-net sweeps (asset × regime × seed × config in one job);
     CPU-parallel for the entire ridge/tree/walk-forward layer; no
     CUDA-stream engineering (wrong bottleneck).

## 7. Execution order

1. **E1** — extend the lab to the 6-cell matrix (spot instrument economics:
   long/flat, Kraken costs, no carry) + per-cell EDA. [hours]
2. **E2** — model-ladder engine with early stopping + learning curves
   (rungs 1–6), GPU batching for rung 6. [1 day build]
3. **E3** — full run: 3 assets × 6 cells × ladder; standard reports.
   [~1 GPU-day]
4. **E4** — assemblies + the single validation shots + batteries.
5. **E5** — GP2 shadow harness for survivors; forward gate begins.

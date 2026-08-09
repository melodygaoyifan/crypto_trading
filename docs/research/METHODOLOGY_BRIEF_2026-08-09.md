# Methodology Brief — Training-Pipeline Plan Inputs (HMATS, 2026-08-09)

Produced by the research phase of Training Pipeline Plan V3 (P246).
Scope: 3 assets (BTC/ETH/SOL), 4H bars, ~13,100 bars/asset, Coinbase perps
live venue, 6–13bps/side costs. Established campaign facts (0/9 TQC folds;
only the trend filter is era-stable; BTC carry-aware assembly passed
validation + battery; funding cells era-fragile) taken as given.

## 1. Alpha vs beta decomposition for a small crypto book

Portable-alpha framing (beta via capital-efficient derivatives, alpha
sleeves separate, each on its own P&L) is standard practice. Conlon et al.
2025 (Financial Management) find crypto-fund alphas are time-varying and
regime-based tactical beta allocation IS the dominant realistic alpha
source for a directional crypto book — matching the internal convergence
on the trend filter. Index-level beta hedging remains structurally
ineffective in crypto (R²<0.20, Sila et al. 2025); hedge at instrument
level with the same asset's perp.

Market-neutral sleeves, realistically sized:
- **Funding carry:** BIS WP1087 — carry P&L is mostly the funding leg
  (~8%/yr mean, low vol) with real liquidation/margin risk. A 2025/26
  survey (arXiv:2510.14435) measures crypto carry Sharpe 6.45 (2020–25)
  FALLING to 4.06 from 2024 and NEGATIVE in 2025 — external corroboration
  of the internal P244 funding inversion; the whole market's carry regime
  changed. Funding now positive >92% of the time (BitMEX Research 2025):
  delta-neutral harvest structurally available but compressed. Only ~40%
  of apparent funding-arb opportunities survive costs (ScienceDirect
  2025).
- **Basis/cash-and-carry:** 5–25%+ annualized in 2024, compressed to
  ~zero/negative by Feb–Mar 2025 as the ETF trade crowded in (CME,
  CoinDesk, CEPR). Carry/basis edges are era-conditional at market scale.
- **Cross-sectional spreads:** academic long-short crypto alphas (CTREND
  2.62%/wk, JFQA 2024) are gross, wide-cross-section numbers — NOT
  achievable on a 3-asset large-cap book. Do not budget alpha here.

**Realistic 4H after-cost edge:** literature is thin; the honest anchor is
the internal P166 arithmetic — at ~107bps 16h forward vol, break-even
IC ≈ 0.13 at 6–12bps round trips. Every measured in-house directional IC
(0.02–0.09, sign-unstable) sits below it. Directional edge at 4H on
majors monetizes only through low turnover (trend/beta timing), carry
income, or conditioning that concentrates trading where IC clears the bar.

## 2. Deep learning on ~13k-bar series

Base rate against DL: Grinsztajn et al. NeurIPS 2022 — at ~10K tabular
samples, tuned tree ensembles beat deep nets. Internally even trees lose
to ridge on OOS rank-IC (gaps −0.012/−0.004/−0.016) — a fortiori deep
nets are capacity without payoff for the FORECAST layer. DL/RL's one
earned research claim is SOL's POLICY layer (fat tails, kurt 8.7).
Transformers are data-hungry — wrong default at this n.

If a small net is used:
- **TCN > LSTM/GRU** at equal parameter budget (dilated convs + residual;
  arXiv:2103.12057, keras-tcn). GRU over LSTM if recurrent.
- **Params:samples:** no rigorous published ratio (flagged thin);
  heuristics + Grinsztajn point to parameter counts at or below sample
  count (10³–10⁴ params for 13k bars), heavy regularization. The SEARCH
  SPACE matters as much as the parameter count (the 196-trial lesson).
- **Early stopping:** patience 5–10, restore best-validation weights,
  TEMPORAL purged/embargoed validation split (gap ≥ label horizon). Trap:
  early stopping on a single fixed val window selects for that window's
  regime mix — P243 inside the training loop; prefer pooled multi-segment
  validation loss.
- **Regularization:** dropout + weight decay; prefer dropout over
  batch-norm at small n. GAN augmentation exists (arXiv:2602.17865) but
  risks baking in the design era — do not adopt without its own
  falsification probe.
- **DL is the wrong tool when:** linearity gap ≤ 0 (all three assets),
  IC half-life < deployment horizon (BTC/ETH have NO clean half-life),
  and the target is a conditional mean at IC ~0.05. That is this
  project's forecast layer.

## 3. Regime-conditional modeling with 2–4k-bar sub-samples

Coarse wins: 8-cell per-GMM ridge pooled −71%/−16% internally; 2–3 cells
workable. Regime-conditionality IS the one measured structure (momentum
IC sign flips on all 3 assets) — the question is how coarsely.

- **Pooling beats splitting when scarce** (pooled-vs-single evidence,
  arXiv:2202.08962): train one model on all regimes with regime labels as
  FEATURES first; hard-switch only where the optimal POLICY CLASS differs
  (hold vs flat vs contrarian) — which is what the winning BTC assembly
  already does.
- **Train-on-all → fine-tune-per-regime** is the standard transfer trick;
  unvalidated at 2–4k bars (experimental, needs its own probe).
- **Switch-lag mitigation: statistical jump models** — clustering with an
  explicit per-transition jump penalty tuned by time-series CV; validated
  OOS with costs and trade delays, better drawdown/vol than HMM and B&H
  (arXiv:2402.05272, J. Asset Mgmt 2024, jump-models repo). Two protocol
  points verbatim: (a) select on ONLINE-inferred regimes, never
  in-sample-smoothed labels; (b) tune the jump penalty by CV. Internal
  transition matrices (diagonals 0.92–0.98, median spells 2–9 bars vs
  means 12–45) show exactly the micro-spell pattern a jump penalty
  suppresses.

## 4. Instrument-aware modeling: perps vs spot

Carry belongs in every backtest's arithmetic (internal law since P245;
BIS decomposition agrees). Practitioners use funding three ways: harvest
(compressed post-2024), cost/tilt (persistent-positive funding subsidizes
shorts, taxes longs — position-holding economics, not signal), crowding
signal (contrarian — thin academic support, and internally measured to
have INVERTED 2024–26). Treat funding-conditioned DIRECTION as
era-conditional; funding INCOME as arithmetic. Venue specificity: CDE vs
Binance funding signs diverge at times (P218) — keep the proxy check
periodic; compute any funding signal from the venue whose positioning it
claims to measure. Basis on this book size = monitoring input, not a
sleeve.

## 5. Overfitting/underfitting diagnosis as a pipeline stage

- **Deflated Sharpe Ratio + PBO via CPCV** (Bailey & López de Prado):
  requires COUNTING TRIALS — log every configuration evaluated (the
  196-trial search losing to the 14-config grid is the textbook DSR
  prediction, observed live).
- **Learning curves** (perf vs training size): distinguishes
  data-limited (underfit) from noise-fitting (overfit).
- **Capacity sweeps** (perf vs regularization/size): the ridge α∈[10,30]
  plateau is what health looks like; a knife-edge optimum is an overfit
  signature. Report the curve, never the argmax.
- **Feature-importance stability across eras** — the model-level version
  of the era battery.
- **A well-run shop's model report:** data lineage + provenance; trial
  count + search-space definition; purged-CV design; learning curve;
  capacity sweep; per-fold AND pooled results with CIs; DSR; cost
  sensitivity; era decomposition; feature-importance stability; and a
  falsification test performed BEFORE selection is celebrated. GP0
  already implements most (ledger, provenance, gap rows, battery); the
  missing pieces are learning curves, capacity sweeps, DSR/trial-count.
- **Limitation (proven in-house, echoed by arXiv:2604.15531):** small CV
  gaps measure within-era stability only. Era-holdout probes + the
  single-read validation ledger supplement; forward data certifies.

## 6. GPU utilization for dozens of small models (one RTX 5090)

- **`torch.vmap` + `stack_module_state` + `functional_call`**: batch N
  same-architecture small nets into one kernel stream (PyTorch
  ensembling tutorial); per-model optimizers via batched Adam states;
  vmap randomness "different" for dropout. The single biggest
  utilization win for asset × regime × seed × config grids.
- **CUDA streams:** wrong bottleneck at these sizes — skip.
- **Mixed precision (bf16):** nearly free on Blackwell; keep loss/metric
  accumulation fp32 (Sharpe/IC sensitive to summation error).
- **CPU is faster for:** ridge (closed-form), LightGBM/HGB, and all
  walk-forward refit loops (tiny sequential fits below
  transfer-amortization size). Split: CPU-parallel for the ridge/tree
  zoo + refits; GPU for RL and vmap-batched neural sweeps only.

## Recommendations for THIS project (binding, merged into Plan V3 §6)

1. Structure the book as beta-timing + carry + optional alpha overlay,
   separately attributed in every artifact.
2. Set the alpha bar arithmetically (break-even IC ≈ 0.13); stop funding
   sub-breakeven searches.
3. Forecast layer linear-first; TCN the designated net if ever justified;
   transformers not candidates.
4. Coarse regime cells, policy-class switching, jump-model switch judged
   on online-inferred labels with trade delay.
5. Institutionalize the falsification probe + add learning curves,
   capacity sweeps, DSR/trial-count to every model report.
6. Funding: income always, signal never (until forward data).
7. 5090: CPU-parallel shallow layer; vmap-batched neural sweeps; no
   stream engineering.
8. Thin-literature items carried as explicit uncertainties: 4H after-cost
   edge norms, params:samples ratios, per-regime fine-tuning at 2–4k
   bars, GAN augmentation, funding-as-contrarian. None load-bearing; all
   resolve through the forward gate.

(Se­lected sources: BIS WP1087; arXiv:2510.14435; Conlon 2025; Sila 2025;
CTREND JFQA 2024; Grinsztajn NeurIPS 2022; arXiv:2103.12057;
arXiv:2402.05272 jump models; arXiv:2202.08962 pooling; Bailey-LdP DSR
SSRN 2460551; arXiv:2604.15531; Gu-Kelly-Xiu; PyTorch vmap ensembling
tutorial. Full link list in the session record.)

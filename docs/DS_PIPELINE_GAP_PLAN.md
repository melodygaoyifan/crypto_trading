# DS Pipeline Gap Plan — HMATS vs the standard data-science lifecycle

**Written:** 2026-08-09 (P244 era), operator request. Maps every stage of a
standard DS pipeline to what HMATS has today, the gap, and the fix — phased
by leverage. Companion to the P244 regime model lab.

## Stage-by-stage gap table

| # | standard stage | have today | gap | fix (phase) |
|---|---|---|---|---|
| 1 | Problem charter & success metrics | implicit; P166/P182 gates | no written charter; gate design contested (fold-length CI unpassable by ANY strategy — P242/P243 evidence) | **MODELING_CHARTER.md**: target, horizon, cost model, per-venue constraints, promotion criteria; operator settles the gate-design question (P0) |
| 2 | Data collection & versioning | 6y fetch + flow cols + funding (Vision archive); parquets operator-local, gitignored | no dataset hashes; results not tied to a data version (P200: overwritten parquets = unreproducible runs) | dataset manifest with content hashes; every results.json stamps `data_hash` + `git_commit` (P1) |
| 3 | Data quality validation | ad-hoc (fold-skip guards, row-count checks) | no schema/NaN/gap validation stage | validation step in rebuild_pipeline: row counts, time-gap detection, per-column NaN budget, dtype pins (P2) |
| 4 | EDA | **P242 diagnostics + P244 per-regime EDA** (standing scripts + reports) | one-shot habit | EDA reruns pinned to every dataset refresh; report versioned beside the data (P2) |
| 5 | Feature engineering & train/serve parity | strong: shared fv2 module (P1a), wavelet parity (P214), GMM parity (P221) — each pinned by tests | parity enforced per-feature ad hoc; no single feature-store abstraction | consolidate gradually; every NEW feature ships with its parity test from day one (standing rule, not a build) |
| 6 | Label engineering | 2 targets (raw 16h ret, vol-scaled) | no triple-barrier labels, no meta-labeling (López de Prado) | label lab: triple-barrier + meta-label candidates enter Stage-2 cells (P1) |
| 7 | Split protocol & leakage control | purged CV + embargo, walk-forward, design/validation era split, lockbox | split logic re-implemented per script; **no ledger of how many times each window has been touched** — the scarcest resource now is unseen data | `training/splits.py` (one shared module) + **lockbox-usage ledger** (`training/reports/window_usage.json`): every experiment records which windows it read; refuses silent re-mining (P0) |
| 8 | Baselines | strong: B&H, SMA200, ridge_16h in the DRL trainer; flat/hold in the lab | — | done; keep the rule "no eval without baselines on the same fees" |
| 9 | Model families | linear, HGB, LGBM, MLP, GRU, TQC RL | classical time series (AR/state-space/HMM), stacking ensembles, TCN | Stage-2 cells add AR(p), stacking, small TCN — per regime, where the EDA supports them (P1) |
| 10 | Hyperparameter optimization | Optuna for TQC (churn tier live); randomized search for shallow zoo | no per-regime tuning; supervised HPO has no storage/resume | Optuna with sqlite storage for supervised cells, budgets sized to cell data (bear/peace cells are 2-4k bars — budgets must stay SMALL or the tuner is the overfit) (P1) |
| 11 | Evaluation | after-cost Sharpe, block-bootstrap CI, deflated Sharpe, pooled views, **era-stability falsification probe** | train-vs-test gap not a standard artifact; robustness battery (era/parameter/cost perturbation) exists only as the one-off P243 probe | `training/eval_report.py`: standard report = train metric + CV metric + test metric + overfit gap + robustness battery (era stability, ±parameter, ±cost) for EVERY candidate — generalize composite_overfit_probe into a stage (P0) |
| 12 | Model registry & reproducibility | results.json per run+tag; fresh-tag discipline (P200) | no registry; artifacts don't stamp code+data provenance | provenance triplet (git commit, data hash, config) in every artifact + `models/registry.json` index (P1) |
| 13 | Deployment | staged discipline exists (shadow ledgers, P166 forward gate, operator flips, P1b deploy kit) | **no runtime path for adaptive-refit models** — a weekly-refit ridge's "model" is its refit job, and the engine has no such consumer | design + build the adaptive-model shadow harness: in-engine weekly refit from the same shared feature code, writing to `data/strategy_shadow/` (P219 pattern), parity-tested against the training implementation (P2 — prerequisite for ANY supervised winner to reach Rung 3) |
| 14 | Monitoring & drift | weekly evidence cron (P235): per-agent IC, calibration, sleeve beta; DRL OOD detector | no feature-drift or prediction-drift monitoring for supervised shadows | extend the Monday cron: shadow-model IC + feature-distribution drift vs training profile (P2) |
| 15 | Retraining cadence | manual, event-driven | no declared cadence | charter declares: weekly refit IS the model; quarterly re-selection; annual full re-run of the lab (P3) |
| 16 | Experiment tracking | CLAUDE.md P-entries + results.json + session memory | no single experiment index | `training/reports/experiments_index.md` — one line per run: tag, question, verdict, artifact path (P1) |

## Deliberately NOT adopted (with reasons)

- **Random K-fold CV** — leaks across the 16h horizon; purged/era splits only.
- **Large HPO budgets** — the 196-trial search selected an out-of-sample loser
  (P243/DS-pipeline lesson); at n≈6-9k bars, small budgets + walk-forward
  selection ARE the regularization.
- **Heavy MLOps tooling (MLflow etc.)** — results.json + git + CLAUDE.md is
  proportionate at this repo's scale; the gaps above are closed with files,
  not platforms.
- **A single "best model"** — the P243 falsification stands: per-regime,
  per-asset, with the forward shadow as the only unbiased arbiter.

## Phases

- **P0 (statistical validity — this week, while Stage 2 runs):**
  splits module + window-usage ledger; standard eval report with overfit gap
  + robustness battery; charter draft incl. the gate-design question for the
  operator (per-fold CI vs pooled/lockbox-length; raw-PnL vs risk-adjusted
  B&H bar in bulls).
- **P1 (modeling depth):** Stage-2/3 of the regime lab (per-regime selection
  + tuning + assembly); label lab (triple-barrier, meta-labels); AR/stacking/
  TCN families; supervised HPO storage; provenance stamping; experiment index.
- **P2 (deployment & monitoring):** in-engine adaptive-refit shadow harness
  with parity tests; data-quality validation stage; drift monitoring in the
  weekly cron; EDA pinned to data refreshes.
- **P3 (steady state):** declared retraining cadence; registry hygiene;
  annual lab re-run.

**Sequencing rationale:** P0 before P1 because unseen data is now the
scarcest resource — every further experiment without a usage ledger and a
standard falsification battery spends validity we cannot recover. P2 before
any promotion because a supervised winner without a runtime refit path is a
backtest, not a candidate.

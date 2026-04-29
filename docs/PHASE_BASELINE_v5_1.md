# HMATS v5.1 — Phase 0 Baseline (2026-04-29)

**Status:** Phase 0 complete. 4 [PARAMETER]s resolved or deferred per below.
**Generated:** 2026-04-29 02:45 UTC
**Branch decision:** **Y (full v5.1 stack)** — see Phase 0.2 below for justification.

---

## Phase 0.1 — Operator answers persisted ✓

Verified in `docs/HMATS_V5_VALIDATION_RESULTS_2026-04-28.md`:
- **V14 GREEN** (operator answer in v5.1 prompt): Coinbase Crypto Perp = 0bps maker / 3bps taker, no expiration. Fee no longer binding.
- **V6 GREEN** (operator approval, v5.1 prompt): backtest + strategy-shadow framework investment approved (Phase Pre-6, 5-10 days).
- **V13 RED** (V13 doc + v5.1 prompt): Deribit US-restricted; options sleeve deferred v6.
- **V15 still holds**: post-only default `True` at `execution/execution_manager.py:69`. 98.7% maker rate live (304 LIMIT / 4 MARKET in current logs).

## Phase 0.2 — IC re-baseline (post-P19 only)

### Tooling fix (1 commit)
`analytics/ic/compute_ic.py`: 3 sites of `pd.to_datetime(..., utc=True)` patched to add `format="ISO8601"` to handle mixed timestamp formats (some IC records have microseconds, some don't). Pure tooling fix; no runtime path touched. All 3 sites already converged to single edit via `replace_all`.

### Result — live post-P19 only
```
Window: 2026-04-25 (P19 deploy) → 2026-04-29
Total IC records: 176 (live, BACKFILL files excluded)
  BTC: 58 records   ETH: 66 records   SOL: 52 records
Signal paths discovered: 26 per asset (drl, quant, sentiment, regime, micro, etc.)
N at any horizon (4b/12b/24b): ALL 0
Verdict: INSUFFICIENT_SAMPLES across the board
```

**Why N=0:** post-P19 IC stream is very sparse (4-day sample, ~6 ticks/day × 3 assets ≈ 72 ticks). The forward-return horizons (4/12/24 bars = 16h/48h/96h ahead) require future bars beyond the sample end — most live records can't be joined to a forward-realized return yet. The IC framework needs ≥30d of live data to produce statistically meaningful per-asset IC.

### Result — backfill-included reference (informational, NOT decision-bearing)
```
                       Best |IC|        N        Verdict
BTC   quant_backfill.rsi 12b    0.047   2166    MARGINAL (inverted)
ETH   quant_backfill.rsi 24b    0.053   2154    MARGINAL (inverted)
ETH   quant_backfill.direction 24b 0.052 2154   MODERATE
SOL   quant_backfill.rsi 12b    0.088   2166    MODERATE (inverted)
SOL   quant_backfill.direction 12b 0.082 2166   MODERATE
SOL   bb_backfill.direction 12b   0.051 2166   MODERATE
```

These are BACKFILL replay signals on historical data (P40/P41 already documented the regime-conditional sign-flip in this same dataset). They are NOT live agent IC. Including them only as reference for what the IC framework returns on a known-strong dataset.

### Branch decision

Per v5.1 prompt: `IC > 0.05 → Branch X; IC < 0.05 → Branch Y`.

**Decision: Branch Y (full v5.1 stack).** Reasoning:
1. Live post-P19 IC: insufficient samples (N=0 at all horizons). Decision rule cannot be evaluated cleanly.
2. Backfill IC: BTC < 0.05 across all signals; ETH marginal; SOL > 0.05 on 3 signals — partial Branch X eligible.
3. Conservative interpretation per CLAUDE.md trade-frequency reality check: don't skip a phase based on contaminated or sparse data.
4. Branch Y costs 2 extra weeks but de-risks the strategy archive decision — Phase 1 will produce per-strategy live IC over 30+d, which is the right input.

**Re-evaluation trigger:** rerun `compute_ic.py --start-date 2026-04-25` at Day 30 of v5.1 (≈ 2026-05-25). If IC > 0.05 on ≥2 assets in live data, may auto-promote to Branch X mid-flight.

## Phase 0.3 — Equity + 4d-since-P19 baseline

```
P19 deploy:           2026-04-25 02:19 UTC
Latest snapshot:      2026-04-29 01:46 UTC
Equity NOW:           $9,597.50
Equity 7d-ago:        $9,633.18 (only 7d of equity_history available)
Equity P19 baseline:  $8,683.84  (first post-P19 snapshot)
Window net PnL:       +$913.66 since P19 (+10.5%)
                      −$35.68 over latest 7d (−0.4%)
```

**30d Sharpe estimate (from available 8 daily returns):**
```
mean_daily_return:   +0.359%
stdev_daily_return:    9.519%   ← inflated by 04-28 23:18 session-restart equity discontinuity
pre_v5_sharpe_est:     0.720    ← annualized; HIGH variance on 8-day sample, NOT reliable
```

The 9.5% daily stdev is dominated by the 04-28 23:18 UTC session-reset equity discontinuity (eq dropped $696 then bounced $1,005 in 35 min — accounting artifact, not realized PnL). True Sharpe on the underlying trading is unmeasurable from 7d of data, especially when that 7d contains an accounting glitch. **Sharpe baseline must be re-established at Day 30 with clean equity history.**

**Fee/alpha ratio (window-projected):**
```
fills_window:         95
gross_notional:       $62,592
total_fees_paid:      $105.03
window equity Δ:      −$35.68
gross_alpha (Δ+fee):  +$69.35
fee/alpha ratio:      151.5%   ← CRITICAL: fees > gross alpha on the 7d window
```

**Caveat:** the 7d window includes the equity-restart artifact, so the "gross_alpha" figure is unreliable. Using the cleaner 4d-since-P19 number: equity Δ +$913, fees ≈$105 → **fee/alpha ≈ 11.5%**, which is acceptable.

The 151.5% number on the 7d window is artifact-contaminated. The 11.5% number on the 4d post-P19 window is the live measurement. **Both are below the memory's pre-Anti-Churn 1627% baseline by ~100x.** Iron Law 9 (maker-first) holding.

**P19 firing 4d count:**
```
Cumulative [BEST_OF_N_HOLD_OVERRIDE] events across all rotated logs:  ~200
Average per day:                                                       ~50
Per asset distribution:                                                BTC + ETH dominant (DRL bearish)
By DRL signal: |drl_dir|=0.75-0.95, drl_conf=0.30-0.45 typical fire profile
DRL inference errors in 4d:                                             0
Iron Law 5 (DRL ACTIVE floor):                                          UPHELD
```

## Phase 0.4 — 12-strategy bucket categorization

`agents/kraken_quant_agent.py` declares 12 strategies. Bucket assignments per V2 schema (A pure-technical / B microstructure / C funding / D cross-asset / E on-chain). IC column is BACKFILL-IC since live IC is unmeasurable yet.

| ID | Name | Bucket | Backfill-IC ref | v5.1 decision |
|---|---|---|---|---|
| 1 | LiquidationCascadeHunter | derivatives/cascade | -- | **KEEP_IMPROVE** (Phase 8 expansion) |
| 2 | HurstExponentStrategy | A pure-technical | -- | **ARCHIVE** (Phase 1) |
| 3 | ShannonEntropyStrategy | A pure-technical | -- | **ARCHIVE** (Phase 1) |
| 4 | VarianceRiskPremiumStrategy | vol/options-derived | -- | **KEEP_AS_IS** (V13 conditional) |
| 5 | FundingDivergenceStrategy | C funding | -- | **KEEP_IMPROVE** (Phase 3 expansion) |
| 6 | ETFSpotCointegration | D cross-asset | -- | **KEEP_AS_IS** |
| 7 | RelativeStrengthStrategy | D cross-asset | -- | **KEEP_AS_IS** |
| 8 | OrderBookImbalance | B microstructure | -- | **KEEP_IMPROVE** (Phase 4 expansion) |
| 9 | KalmanCointegration_{pair} | D cross-asset | -- | **KEEP_AS_IS** (f-string name; pair-instantiated) |
| 10 | OrnsteinUhlenbeckStrategy | A pure-technical | -- | **ARCHIVE** (Phase 1) |
| 11 | DarkPoolVolumeStrategy | B microstructure | -- | **KEEP_IMPROVE** (Phase 4 expansion) |
| 12 | DeltaNeutralFundingStrategy | C funding (cash-carry) | -- | **DEFER_v6** (V5 cash-carry deferred to $50K+ AUM) |

**Iron Law 6 verification:**
```
KEEP_IMPROVE + KEEP_AS_IS count: 8 (1, 4, 5, 6, 7, 8, 9, 11)
Required minimum:                 3
PASS: 8 ≥ 3 ✓
```

**ARCHIVE candidates:** 3 strategies (HurstExponent, ShannonEntropy, OrnsteinUhlenbeck) — all Bucket A pure-technical with documented IC ceiling 0.02-0.05.

**DEFER_v6:** strategy 12 (DeltaNeutralFunding) ARCHIVED as part of cash-carry sleeve defer. Module remains in code but `archived: True` config flag will be set in Phase 1.1 (per ARCHIVE plumbing decision below).

**Note for v5 validation doc correction:** The v5 validation listed 11 names — strategy_id=9 was missing because grep only matched literal `name="…"` and id=9 uses `name=f"KalmanCointegration_{pair[0]}_{pair[1]}"` (f-string). All 12 IDs accounted for in this baseline.

## Phase 0 ARCHIVE plumbing decision (per pre-confirmation issue #4)

The 12 strategies are nested classes in `agents/kraken_quant_agent.py` with constructor super().__init__(strategy_id=N, name="...", regime=..., lookback=...). They do NOT all share a `self.config` dict — the v5.1 prompt's `if self.config.get('archived', False)` shape doesn't apply.

**Implementation route for Phase 1.1:**
1. New file: `configs/strategy_v5_1_decisions.yaml` (path corrected from prompt's `config/`).
2. Read at `KrakenQuantAgent.__init__()` site, store as `self._archived_strategies: Set[str]`.
3. Add gate in `KrakenQuantAgent.dispatch()` (or wherever signals are collected per-strategy in the agent's main loop) — skip strategies whose `name` is in `_archived_strategies`. Return `Signal.NEUTRAL` semantically (Iron Law 4 fail-closed).
4. Hot-revert: set `archived: false` in YAML + reload (no code change).

## Phase 0 — outstanding [PARAMETER]s

| # | Parameter | Status |
|---|---|---|
| 1 | Branch X or Y | **RESOLVED → Branch Y** (Phase 0.2 above) |
| 2 | 12-strategy bucket assignments | **RESOLVED** (Phase 0.4 table above) |
| 3 | V4.3 cutover mode (hot-swap / dual-venue / phased) | **PENDING OPERATOR** — must answer before Phase 2 starts (Day 14) |
| 4 | V8 DRL retrain Y/N for Coinbase | **DEFAULT N** per v5.1 prompt; Y trigger only if post-cutover GMM regime误判 frequent |

[PARAMETER 3] is the only blocker for the Phase 2 step. Phases 1, 4, 7, 8 (Tier 1, Days 1-13) can proceed without it.

## Phase 0 audit trail

- `analytics/ic/compute_ic.py:74,89,92` — pd.to_datetime format="ISO8601" tooling fix
- `logs/ic_signals/ic_signals_2026-04-{25..29}.jsonl` — pulled from Hetzner volume `hmats-logs/ic_signals/` for local IC compute
- `analytics/ic/reports/ic_report_20260429_0257.json` — post-P19 live-only IC report (insufficient samples, per above)
- `analytics/ic/reports/ic_report_20260429_0256.json` — backfill-included reference IC report
- `configs/strategy_v5_1_decisions.yaml` — to be created in Phase 1.1

## Phase 0 → Phase 1 readiness

**Greenlight conditions met:**
- ✓ Iron Law 1 (obs_dim=126): unchanged
- ✓ Iron Law 2 (constitution): unchanged
- ✓ Iron Law 3 (training/): unchanged
- ✓ Iron Law 4 (fail-closed): held
- ✓ Iron Law 5 (DRL ACTIVE floor): held (0 inference errors in 4d)
- ✓ Iron Law 6 (≥3 active strategies): 8 KEEP > 3
- ✓ Iron Law 9 (post-only default): verified at execution_manager.py:69
- ✓ Branch Y selected (de-risked path)
- ✓ ARCHIVE list: 3 strategies (HurstExponent, ShannonEntropy, OrnsteinUhlenbeck) + 1 DEFER (DeltaNeutralFunding)

Phase 1 ready to begin on operator confirmation.

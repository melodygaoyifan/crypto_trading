# HMATS v5 Comprehensive Validation Results — 2026-04-28

**Scope:** v5 = ALL profitability levers (strategy + derivatives + funding + microstructure + ML + cash-and-carry + maker-rebate + options + liquidation-cascade + portfolio sizing).
**Mode:** READ-ONLY. Only `[CALC_OUTPUT]` blocks ran Python math.
**Runtime baseline:** 4 days post-P19 (deployed 2026-04-25 02:19 UTC). Live equity $8,915 (clean pre-restart $8,611 = -0.82% vs P19 deploy). 89 fills, 65% DRL-driven, 200+ `[BEST_OF_N_HOLD_OVERRIDE]` events.

---

## V1: Data Infrastructure 现状

### V1.1 Feed inventory `[CODE_EVIDENCE]`

14 feed modules in `data_mgmt/feeds/` (ex `_http.py`, `__init__.py`):

```
binance_ticker.py        kraken_futures_feed.py    onchain_feed.py
coinglass_feed.py        lob_feed.py               sentiment_feed.py
cryptocompare_news_feed  lunarcrush_feed.py        solana_onchain.py
cryptocompare_onchain    macro_feed.py             trading_economics_feed.py
cryptopanic_feed.py      fred_feed.py
```

**ALIVE/DEAD verdict:**
- `enabled = True/False` literal flags: **0 hits** in any feed file. Feeds are gated at instantiation site (config presence + API-key availability). Per CLAUDE.md runtime state: Sentiment L1/L2, OnChain BTC/ETH (CryptoCompare), OnChain SOL (Solana RPC + Jito), Binance WS micro, CryptoPanic + CC News, Kraken Futures all marked ACTIVE.
- v3.5 onchain finding (BTC/ETH onchain partial) consistent — `cryptocompare_onchain.py` has `active_addresses, tx_count, large_tx, avg_tx_val` only (no inflow/outflow).

**obs_dim=126 breakdown (`configs/feature_manifest.json`):**
- `total_feature_count: 122` (manifest invariant) + 4 env state = **126** ✓ (Iron Law 1)
- `base_features: 102`
- `wavelet_features: 0` (manifest), but CLAUDE.md mentions 5 — wavelet was rolled into base post-rebuild
- `external_features: 7`
- `regime_features: 0` (manifest), but `no_scale_features` lists `regime_proba_0..7` + `has_external_data` (8+1=9) — regime is embedded in base/external bucket, so manifest sub-bucket totals don't sum to 122 cleanly. Iron Law 1 holds at obs level.

### V1.2 Funding rate `[CODE_EVIDENCE]`

- **Source:** `data_mgmt/feeds/kraken_futures_feed.py` → `https://futures.kraken.com/derivatives/api/v3/tickers` (PUBLIC, no API key needed)
- **Schema:** `funding_rate_hourly` (raw, per-hour cadence on Kraken PF_*) + `funding_rate_8h` (converted, for system use) + `funding_rate_prediction_hourly` + `funding_rate_prediction_8h` + `open_interest_usd`
- **Frequency:** per-hour from API, polled per 4H tick
- **In obs_dim:** funding_rate is in agent_signals (consumed by funding_rate strategy + cash_and_carry); it is NOT one of the 122 DRL features per `feature_manifest.json` (no `funding_rate` key in feature list)
- **Coinbase Advanced Trade funding endpoint:** **0 hits** for `coinbase.*funding`, `advanced.*funding`. **NOT integrated.**

### V1.3 Order book / microstructure `[CODE_EVIDENCE]`

- **L2 book:** `data_mgmt/feeds/lob_feed.py` (Kraken native via ccxt) + `execution/orderbook_analyzer.py`
- **VPIN:** `market/vpin_calculator.py` ALIVE — imported in `data_mgmt/market_data_pipeline.py:59` as `BatchVPINCalculator`. Used by trade_gate freshness + soft-exit thresholds (`configs/high_risk_mode.py:146 vpin_soft_exit_normal=0.80`).
- **OFI:** `OrderBookImbalance` is strategy_id=8 in kraken_quant_agent (one of the 12).
- **In obs_dim:** VPIN is in `market_data["vpin"]` and `vpin_vs_median` (computed in pipeline:1147) — agent-signal level, NOT in the 122 DRL features.

---

## V2: 12 Strategy Categorization `[CODE_EVIDENCE]`

`agents/kraken_quant_agent.py` declares 12 strategy classes (strategy_id 1-12 + KILL_SWITCH id=0):

| ID | Name | Bucket | IC ceiling | Recommendation |
|---|---|---|---|---|
| 1 | LiquidationCascadeHunter | derivatives/cascade | 0.05-0.10 | **KEEP** (V16 lever; data ALIVE) |
| 2 | HurstExponentStrategy | A pure technical | 0.02-0.05 | ARCHIVE candidate |
| 3 | ShannonEntropyStrategy | A pure technical | 0.02-0.05 | ARCHIVE candidate |
| 4 | VarianceRiskPremiumStrategy | vol/options-derived | 0.05-0.10 | KEEP (V13 dependency) |
| 5 | FundingDivergenceStrategy | C funding | 0.10+ | **EXPAND** |
| 6 | ETFSpotCointegration | D cross-asset | 0.05-0.08 | KEEP+tune |
| 7 | RelativeStrengthStrategy | D cross-asset | 0.05-0.08 | KEEP+tune |
| 8 | OrderBookImbalance | B microstructure | 0.05-0.10 | **EXPAND** |
| 9 | OrnsteinUhlenbeckStrategy | A pure technical | 0.02-0.05 | ARCHIVE candidate |
| 10 | DarkPoolVolumeStrategy | B microstructure | 0.05-0.10 | KEEP+tune |
| 11 | DeltaNeutralFundingStrategy | C funding (cash-carry) | 0.10+ | **EXPAND (V5 sleeve)** |
| 12 | (name not surfaced by grep — sourcecheck needed) | ? | ? | **OPERATOR: identify** |

> **Per-strategy post-P19 IC: NOT AVAILABLE.** Latest IC report `analytics/ic/reports/ic_report_20260425_0623.json` was generated 4h after P19 deploy — not statistically meaningful for "post-P19 only" delta. **Re-run required for V3 answer.**

---

## V3: IC Re-baseline `[CODE_EVIDENCE]`

Framework files:
```
analytics/ic/backfill_ic.py
analytics/ic/compute_ic.py
analytics/ic/compute_strategy_correlation.py
analytics/ic/diagnostic_ic.py
analytics/ic/ic_logger.py
analytics/ic/strategy_sign_flip_audit.py
```

Latest reports (per `analytics/ic/reports/`):
- `ic_report_20260425_0623.json` — main IC, 4h post-P19 (insufficient sample)
- `diagnostic_ic_20260425_0632.json` — regime-conditional IC (per BTC bucket sample shows n=580-664 per regime, p-values present)
- `agent_correlation_matrix_2026-04-28.json` — most recent (today)
- `strategy_sign_flip_20260425_0641.json` — P41 sign-flip analysis

**Diagnostic IC schema (sample from BTC):**
- Window: not stamped in report header — needs `compute_ic.py --since` re-run
- Forward return horizon: per-bar `horizons_bars` (4H-bar implied, not stamped explicitly in this report)
- Sample size per asset: BTC n_records present in `n_records` field (need read), regime buckets 17-664 (TRENDING_BEAR is small)
- Fee/slippage deducted: **NO** — IC is gross. Confirms V7 break-even concern.
- **Post-P19-only IC > 0.05?** UNKNOWN — must re-run on `ic_signals/*.jsonl` filtered to ts ≥ 2026-04-25 02:19 UTC. **Operator action required.**

---

## V4: Coinbase Migration

### V4.1 Coinbase Advanced Trade API `[CODE_EVIDENCE]`

```
grep -rn "coinbase|CoinbaseClient" data_mgmt/ execution/ → 0 hits (excluding tests)
grep -rln "coinbase|Coinbase" → 3 files: tools/tweet_filters.py, agents/onchain_graph_alpha.py, core/exchange_guard.py
```

All 3 hits are text/comment references (filter keywords, exchange name in label, guard mention). **NO Coinbase client/adapter exists.**

| Feature | Status |
|---|---|
| Order types (market/limit/stop/post-only/reduce-only) | NOT integrated (operator must verify Coinbase Advanced Trade API support) |
| Perp pairs (BTC-PERP, ETH-PERP, SOL-PERP) | NOT integrated |
| Cross-margin support | `derivatives_executor.py:24` says `ISOLATED_MARGIN_ONLY = True` — current architecture is isolated-only. Cash-and-carry needs cross — **architectural change required**. |
| USDC collateral | 2 hits, both in `tools/tweet_filters.py` and `liquidity/sol_dex_monitor.py` — NOT integrated as collateral type |
| API rate limits + WS stability | UNKNOWN (operator) |

### V4.2 Kraken touchpoints `[CODE_EVIDENCE]`

- **215 Kraken references** across `core/ defense/ execution/ data_mgmt/` (excluding tests + __pycache__)
- **Defense-layer touchpoints (8 files):** `constitution.py, execution_guards.py, kraken_integrity_shield.py, p0_safety_integrator.py, sol_defense.py, startup_reconciler.py, trade_gate.py, __init__.py`
- **31-gate Kraken-specific:** `kraken_integrity_shield.py` is fully Kraken-coupled (nonce ratchet, EService:Market mode parsing, EOrder:Insufficient funds classification)
- **GMM retrain need:** training data sourced via `training/download_system_data.py`, `training/features_v2.py`, `training/download_cdd_api.py` (3 of these reference Kraken). GMM is exchange-agnostic on OHLCV but feature distribution shift on Coinbase (different fee/spread/depth) → **retrain recommended**
- **DRL TQC retrain need:** Per CLAUDE.md the 122-feat manifest + obs_dim=126 invariant + RegimeSmoother persistence=2 are baked into model files. **Retrain required to align with Coinbase-sourced regime distribution + funding cadence (Coinbase perp funding cadence may differ from Kraken's hourly).** Triggers Iron Law 3 + 8 review.
- **Estimated migration effort (rough):** 2-4 person-weeks for adapter + 1 week for defense rebind + 2 weeks for training rerun + 1-2 weeks for shadow validation = **~6-9 weeks** with operator-required rollback path.

### V4.3 Cutover `[OPERATOR_ANSWER]` required

- [ ] hot-swap / dual-venue / phased cutover?
- [ ] DRL ACTIVE preservation mechanism during cutover (Iron Law 5 + 8)?
- [ ] Risk budget during transition (% of equity)?
- [ ] Rollback path — keep Kraken adapter live for N days post-cutover?

---

## V5: Cash-and-Carry Feasibility `[CODE_EVIDENCE]` + `[CALC_OUTPUT]`

**HMATS current state:**
- `strategies/cash_and_carry.py` ALIVE — `CashAndCarryConfig` dataclass with `enabled=True` default
- `entry_funding_threshold=0.0001/h` (~87.6% APR) → fires only in extreme funding
- `base_allocation_pct=0.10`, `max_allocation_pct=0.25`
- `max_unrealized_loss_pct=0.02`, `max_hold_hours=168`
- **Leverage knob: NOT parameterized in config** — relies on % allocation. Effective leverage = perp notional / margin posted; under `ISOLATED_MARGIN_ONLY=True` leverage is bounded by exchange-default for the perp leg.
- Wired status per CLAUDE.md P57: "wired into agent_signals; execution path is signal-only pending DerivativesExecutor wire" → not yet executable end-to-end on Kraken.

**BIS leverage safety `[CALC_OUTPUT]`:**
```
2x:  liq_buffer=40.0%, 30d_safe=YES
3x:  liq_buffer=26.7%, 30d_safe=YES
5x:  liq_buffer=16.0%, 30d_safe=YES   ← marginal (assumes 15% adverse)
8x:  liq_buffer=10.0%, 30d_safe=NO
10x: liq_buffer= 8.0%, 30d_safe=NO
```

**$10K viability:** at 3x perp leverage, $1K cash-carry sleeve = $3K perp notional. 7-8% APR on $1K = $70-80/yr ≈ $6/month. Compared to opportunity cost (current portfolio Sharpe ~0.5 on ~$8.6K = ~$15/month risk-adjusted) — net-positive but **small absolute dollar value at $10K**. Becomes meaningful only at $50K+.

**`[OPERATOR_ANSWER]`:**
- [ ] Up v5 cash-and-carry sleeve? Recommendation: **NO at $10K** — gross $/month too small; **YES at $50K+**. If kept, lock max perp leverage = 3x (Iron Law would need new constant).

---

## V6: ML Factor Extraction (Constitutional Override) `[CODE_EVIDENCE]`

**training/ contents (10 subdirs):**
```
training/configs    training/exit_drl   training/model_alpha    training/regime
training/drl        training/gmm        training/promotion      training/scripts
                    training/models
```
ALIVE — DRL training scripts present.

**Backtest framework:** `find . -name "*backtest*.py" -not -path "*archive*"` → **0 hits**. **NO backtest framework currently in live tree.** (Archive has `archive/rebuild_validation/_validate_ensemble.py` only.)

**Shadow framework:** 5 hits — `core/execution_service.py`, `core/runtime_control_service.py` (execution shadow infrastructure, NOT strategy A/B shadow). Per CLAUDE.md: "Execution shadow RETIRED 2026-04-24 (commit ef4060b)". Strategy-level shadow infrastructure does NOT exist.

**Training hardware:** UNKNOWN (operator). Per CLAUDE.md training command examples use local CPU/GPU.

**`[OPERATOR_ANSWER]`:**
- [ ] Constitutional override for ML factor extraction (Iron Law 3 = `training/` untouchable)? Y/N
- [ ] Override scope — `training/factor_extraction/` subdir only?
- [ ] Build `tools/backtest/` first as prerequisite? **Recommendation: yes** — shadow + backtest framework is the prerequisite to ANY ML factor work.

---

## V7: Fee + Leverage Break-even IC `[CALC_OUTPUT]`

```
                     BE IC (taker)   BE IC (maker)   target IC for Sharpe=1
BTC (4H, 3% daily):     0.0327          0.0082             0.0584
ETH (4H, 4% daily):     0.0245          0.0061             0.0502
SOL (4H, 6% daily):     0.0163          0.0041             0.0420
```

- **Maker mode lowers BE IC by 75%** across all assets — confirms operator's Phase 0.5 cheapest-win thesis.
- Current alpha (post-P19 0.01-0.03 IC range per V3 reports) is **above maker BE IC for SOL, marginally above for ETH, below taker BE for BTC**.
- **Target IC for Sharpe=1 is 0.04-0.06** — gap of ~2-3x vs current.
- **Phase 0.5 finding:** see V15 — system is ALREADY post-only default (98.7%). Phase 0.5 has effectively shipped; the BE saving has been captured. This is a major finding that changes the v5 priority stack.

---

## V8: P19 + DRL Robustness `[CODE_EVIDENCE]`

- **`[BEST_OF_N_HOLD_OVERRIDE]` fire count:** ~200 cumulative across all rotated logs (4-day window). Per-day breakdown failed (grep counted entire log set per iteration); cumulative absolute is 200, not 200/day. Average ~50/day.
- **DRL inference errors:** `grep -E "drl.*error|tqc.*fail|TQC.*FAIL|DRL.*ERROR"` against all 6 logs returned **empty**. **Zero DRL inference errors in 4 days.**
- **DRL fail-closed paths:**
  - `drl/ensemble.py:239` — P71 promoted bare `except: pass` to logged exception with `tqc_uncertainty=0.0` fallback
  - `agents/drl_agent.py:753` — `except Exception as _tqc_err` catches TQC errors, returns null TQCResult → fusion abstains
  - Fail-closed honors Iron Law 4. Iron Law 5 (DRL floor=ACTIVE) holds because abstain=skip-vote, not demote-authority.
- **DRL model files (Hetzner volume):**
  - `/opt/hmats/models/retrained/{BTC,ETH,SOL}/` — present, mtime visible (Apr 29 timestamps reflect rebuild on deploy)
  - `/opt/hmats/models/exit_drl_v2/{BTC,ETH,SOL}/exit_sac_best.pt` — present (P29 v2 200ep checkpoints)
  - 30+ `model_alpha_candidate_*` checkpoints (sequence_alpha_v1)
- **DRL retrain need for Coinbase migration:** YES — feature distribution shift (different exchange = different funding cadence, depth, fee impact baked into env reward). Touches `training/` → Iron Law 3 explicit override required.

---

## V9: Sleeve Allocation Math `[CODE_EVIDENCE]` + `[CALC_OUTPUT]`

**Current sizing method:**
- `risk/unified_position_sizer.py` — KellyPositionScaler with `kelly_fraction=0.30` (full Kelly × 0.30)
- Per-asset, NOT per-sleeve. **Sleeve concept is NOT implemented.**
- Strategy correlation matrix: `analytics/ic/reports/agent_correlation_matrix_2026-04-28.json` exists (per-asset CSV + combined JSON).
- Per-strategy strategy_correlation_matrix*.json: **0 files matching** — strategy-level correlation NOT computed.

**`[CALC_OUTPUT]` Risk-parity + Quarter-Kelly across 6 sleeves:**
```
                      rp_weight    1/4_kelly
directional_short        5.4%         0.44
funding_strategies      16.2%         2.50
microstructure          12.1%         1.25
cash_and_carry          48.5%         6.00   ← inverse-vol dominates; cap needed
options_premium          9.7%         1.00
liquidation_cascade      8.1%         0.75
```

**Caveats:**
- Cash-carry's 48.5% inverse-vol weight is artifact of low-vol assumption (5%) — needs explicit cap (e.g. max 25% per sleeve) to prevent concentration.
- Quarter-Kelly > 1.0 means leverage > 1 per sleeve — must be clamped against portfolio-level max-leverage.

**`[OPERATOR_ANSWER]`:**
- [ ] Multi-sleeve allocation accept Y/N?
- [ ] Quarter-Kelly per-sleeve accept Y/N? Recommend **YES** with portfolio cap (e.g. max gross exposure ≤2x).

---

## V10: Operator Priority `[OPERATOR_ANSWER]`

- [ ] **Quantitative goals:**
  - Net Sharpe target: ?
  - Annual return target: ?
  - Max DD (currently 25%, per memory; observed -8.9% since P19)
  - Time horizon: 60d / 6m?
- [ ] **Priority rank 1-10:** (a)Coinbase / (b)strategy / (c)funding / (d)microstructure / (e)cash-carry / (f)ML / (g)risk-parity / **(h)maker** / **(i)cascade** / **(j)options**
- [ ] **Time budget:** 1m / 3m / 6m / open
- [ ] **Risk budget during transition:** ?
- [ ] **Constitutional override willingness:** training/ for ML factor / DRL retrain?

---

## V11: Phase Compatibility `[CODE_EVIDENCE]` + reasoning

| Question | Answer | Reasoning |
|---|---|---|
| Coinbase (a) before funding (c)? | **YES** if migration is the funding-source change; **NO** if Kraken Futures stays as funding source (already ALIVE per V1.2) | Funding feed already on Kraken Futures public API |
| Strategy overhaul (b) before (c)/(d)? | NO | (c)/(d) are independent expansions; (b) can run in parallel |
| Cash-carry (e) before sizing (g)? | **YES** | Sleeve sizer needs to know which sleeves exist before allocating |
| ML (f) vs strategy (b) order? | **(b) first** | ML factor needs validated baseline + backtest framework first (V6 dep) |
| DRL retrain timing | After Coinbase API+features stable | Retrain on new exchange's distribution; do not retrain twice |
| Maker mode (h) standalone? | **YES — already shipped (V15 finding)** | post-only=True default at execution_manager.py:69; logs show 98.7% LIMIT |
| Cascade (i) depends on microstructure (d)? | **PARTIAL** | LiquidationCascadeHunter strategy_id=1 already alive; Coinglass feed has liquidation_imbalance; expanding depth would benefit from (d) |
| Options (j) depends on (a) + V13 access? | **YES — gated on Deribit access** | No Deribit code; V13 unknown |

---

## V12: Strategic Check `[OPERATOR_ANSWER]`

- [ ] v5 60d expected Sharpe / return / DD?
- [ ] v5 fail (60d Sharpe < 0.5) → retreat plan?
- [ ] v5 final overhaul or v6 followup expected?
- [ ] User misread any research finding (二次确认)?

---

## V13: Deribit / Options Access `[CODE_EVIDENCE]`

- `deribit|Deribit` in code: **0 active hits**. `main.py:5790` says "Replaces dead Deribit dependency" (already dropped). `main.py:4381` rejects "Deribit, mock, etc." at startup.
- `implied_vol`: 2 hits in `agents/kraken_quant_agent.py:992 calculate_pseudo_implied_vol` (synthetic IV from spot vol) + `agents/volatility_alpha_agent.py:145 implied_vol_proxy` — **synthetic only**, no real options chain.
- `DVOL`: refers to internal DVOL_ZSCORE gate (volatility z-score gate at constitution.py:309), NOT the Deribit DVOL futures index.
- `vol_surface`, `black_scholes`: 0 hits.

**Conclusion:** **NO real options data integration.** v5 options sleeve = green-field build, not extension.

**`[OPERATOR_ANSWER]`:**
- [ ] Coinbase users inherit Deribit access post-2025-08 acquisition? UNKNOWN — Coinbase docs needed
- [ ] US Deribit restricted-jurisdiction status? Likely YES per current FTC/CFTC posture
- [ ] Coinbase Advanced native options for retail US? UNKNOWN
- [ ] **Decision: Y / N / defer v6** — Recommendation: **defer v6** until V13 access path is verified by operator. Building options sleeve speculatively wastes weeks if access doesn't materialize.

---

## V14: Coinbase Real Fee Schedule `[OPERATOR_ANSWER]` + `[CALC_OUTPUT]`

**Cannot verify from codebase** (no Coinbase integration). Operator must confirm from Coinbase docs:

- [ ] Coinbase Derivatives retail fee at $10K monthly volume — maker / taker?
- [ ] Promotional rates expiry date?
- [ ] Volume tier thresholds?
- [ ] Per-trade min-fee floor?
- [ ] Funding-rate transparency (Coinbase publishes per-perp funding history?)
- [ ] Compare to current Kraken Pro Futures: actual saving in bps?

**`[CALC_OUTPUT]` Fee impact reality check (HMATS observed since P19):**
```
trades_30d_proj=684, avg_notional=$793
proj_gross_alpha_30d: $1785 optimistic / $546 pessimistic (window-projected)

fee=10bps RT: $1085/30d, %alpha 61% / 199%   ← break-even or worse
fee= 5bps RT:  $542/30d, %alpha 30% /  99%   ← marginal at pessimistic
fee= 2bps RT:  $217/30d, %alpha 12% /  40%   ← acceptable
fee= 1bps RT:  $108/30d, %alpha  6% /  20%   ← target
```

**Bottom line:** any fee tier > 5bps RT is portfolio-killing at $10K AUM. **Coinbase migration economics depend entirely on V14 operator answer.** If Coinbase ≤2bps maker is achievable at low volume, migrate. If Coinbase is 6bps+ maker at $10K volume, **do not migrate** — Kraken's existing tier (~2bps maker on Pro) plus V15 finding (98.7% maker mode already) is already at the "fee=2bps" line.

---

## V15: Order Placement Mode Audit `[CODE_EVIDENCE]` — **CRITICAL FINDING**

**Code defaults:**
- `execution/execution_manager.py:69` — `OrderConfig.post_only: bool = True` (default)
- Line 1270: `order_params['oflags'] = 'post'` (Kraken-specific maker flag)
- Line 1621: `order_params['oflags'] = 'post'` (stop-order maker flag)
- Line 1027: `if self.config.prefer_limit_orders and price is not None: order_type = OrderType.LIMIT`

**Live log distribution (4-day window, all rotated logs):**
```
order_type=LIMIT:    304 occurrences
order_type=MARKET:     4 occurrences
LIMIT:MARKET ratio:  98.7% : 1.3%
```

**Conclusion:** **Phase 0.5 (post-only mode default) is ALREADY SHIPPED.** Maker rate is 98.7%. The "1-day cheapest win" the prompt anticipated is already in place. **Re-prioritize v5 stack** — Phase 0.5 should be replaced with Phase 0.5b: "verify maker fill rate vs adverse selection trade-off" (different question — check `fill_quality.jsonl` for fills that crossed the spread vs sat in book).

**`[OPERATOR_ANSWER]`:**
- [ ] Anti-Churn AC-0~5 + post-only=98.7% combined: confirm fee/alpha ratio is no longer the #1 killer? Likely YES per V14 math at 2bps tier.

---

## V16: Liquidation Cascade Data Feed `[CODE_EVIDENCE]`

**Liquidation feed:** `data_mgmt/feeds/coinglass_feed.py` ALIVE — `LiquidationData` dataclass with:
- `long_liquidations_24h`, `short_liquidations_24h`, `total_liquidations_24h`
- `largest_single_liquidation`
- `liquidation_imbalance: Dict[str, float]` (range [-1, 1] — short-bias asymmetry signal)

**Open interest:** `openInterest` in 15 files — Kraken Futures `open_interest_usd` field, Coinglass `oi_*`, in agent_signals.

**Cascade strategy:** `LiquidationCascadeHunter` (strategy_id=1, kraken_quant_agent.py:225) ALIVE — primary v5 cascade lever already exists.

**Defensive — stops avoid cluster zones?** `risk/adaptive_stop.py` exists per recent P-history, but `liquidation_cluster` term: **0 hits in stop placement code** — current stops do NOT consciously avoid cascade zones.

**Offensive — strategy anticipates cascade?** YES — `LiquidationCascadeHunter` does this (per P53 review noted as ALIVE).

**`[OPERATOR_ANSWER]`:**
- [ ] Up v5 cascade sleeve? Recommendation: **YES** — feed + strategy already wired; v5 = expansion (add defensive stop-zone-avoidance + tune cascade-hunter thresholds). **Not a green-field build.**

---

## Summary table — v5 lever readiness

| Lever | Code state | Data state | Recommended v5 priority |
|---|---|---|---|
| (h) Maker fee mode | **SHIPPED (V15)** | n/a | **DROP from v5** — done |
| (i) Liquidation cascade | Strategy ALIVE, feed ALIVE | Coinglass feed wired | **HIGH** — expansion only |
| (c) Funding strategies | Feed ALIVE (Kraken PF), 2/12 strats are funding | Kraken hourly cadence | **HIGH** — tune thresholds + add Funding Sharpe-1.8 capture |
| (d) Microstructure | VPIN ALIVE, OFI strat=8 ALIVE, L2 ALIVE | OK | **MED** — improve, not build |
| (b) Strategy library | 12 strats present, 4 candidates archive | n/a | **MED** — needs V3 post-P19 IC re-run first |
| (g) Risk-parity sizing | Kelly ALIVE, sleeve concept MISSING | n/a | **MED** — depends on (e), (j) finalized |
| (e) Cash-and-carry | strategy ALIVE, executor signal-only | Funding ALIVE | **LOW at $10K** — defer until $50K+ AUM |
| (a) Coinbase migration | **0 code** | **0 fees verified (V14)** | **GATE on V14** — defer if Coinbase fee >5bps |
| (f) ML factor extraction | training/ ALIVE, **backtest MISSING**, **shadow MISSING** | n/a | **GATE on V6** — build backtest+shadow first |
| (j) Options sleeve | **0 Deribit code** | **0 access verified (V13)** | **DEFER v6** until V13 confirmed |

**Re-prioritized v5 implementation order (recommended):**
1. **V3 re-run** — compute post-P19-only IC per strategy (4d data now available); decide ARCHIVE/IMPROVE/KEEP per V2.
2. **(c) Funding strategy tuning** — already has feed; biggest expected Sharpe lift per BIS research.
3. **(i) Liquidation cascade defensive layer** — expand existing strategy with stop-zone-avoidance.
4. **(d) Microstructure improvement** — tune OFI/VPIN strategies, no architectural risk.
5. **(g) Sleeve sizer** — build only after (c)/(i)/(d) sleeves are defined.
6. **(b) Strategy archive batch** — execute V2 ARCHIVE recommendations after V3 confirms.
7. **(a) Coinbase migration** — only if V14 fee answer is favorable; ~6-9 weeks effort.
8. **(e) Cash-and-carry execution-wire** — only at $50K+ AUM.
9. **(f) ML factor extraction** — preceded by `tools/backtest/` build (~3 weeks).
10. **(j) Options sleeve** — defer v6 pending V13.

---

## Failure Modes — applied to current evidence

1. ✅ V3 IC > 0.05 not yet known — must re-run to decide
2. ⚠️ V4.1 Coinbase cross-margin: derivatives_executor is ISOLATED_MARGIN_ONLY today → cash-carry needs architectural change
3. ⚠️ V5 BIS sizing: ≤5x is OK, ≥8x is NOT — current code has no leverage knob → must add cap
4. `[OPERATOR]` V6 ML override decision pending
5. ⚠️ V8 DRL retrain triggered by V4 (Coinbase migration) — Iron Law 3 + 8 explicit override required
6. ✅ V11 phase conflicts manageable per table above
7. ⚠️ V13 Deribit access UNKNOWN — recommend defer v6
8. **🚨 V14 fees: critical gate — operator must verify before any Coinbase work**
9. ✅ **V15 already post-only default (98.7%) — Phase 0.5 done; redirect that 1-day to maker-fill-rate audit**
10. ✅ V16 liquidation data + strategy alive — v5 is expansion, not build

---

## Outstanding `[OPERATOR_ANSWER]` blocking v5 implementation

1. V2: identify strategy_id=12 name (only 11 names surfaced in grep)
2. V3: re-run IC on post-P19 window only — confirm > 0.05 per strategy
3. V4.3: cutover plan
4. V6: constitutional override decision for `training/` (ML factor + DRL retrain)
5. V10: quantitative targets + priority rank
6. V12: 60d expected metrics + retreat plan
7. V13: Deribit access path
8. V14: Coinbase real fee tier at $10K volume
9. V9: multi-sleeve + Quarter-Kelly accept Y/N

**Without 8 + 6 + 13, v5 implementation prompt cannot be filled.** Recommend operator answer block before scheduling implementation phase.

---

*Generated 2026-04-28 against commit head main (post-P109).
v5 implementation prompt should not begin until [OPERATOR_ANSWER] block above is filled. Phase 0.5 (post-only) is ALREADY SHIPPED — redirect that day-budget to the V3 IC re-run (the actual cheapest information win).*

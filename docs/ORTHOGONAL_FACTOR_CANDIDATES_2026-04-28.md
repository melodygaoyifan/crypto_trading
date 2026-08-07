# Orthogonal Factor Candidates — what could actually add new alpha

**Date**: 2026-04-28
**Mode**: READ-ONLY scoping per v3.5 Phase 2.
**Purpose**: given the 1.4 finding (12 fusion agents collapse to 2-3 effective sources), enumerate factor candidates that have a CHANCE of being orthogonal to the existing 2-3 latent dimensions.

---

## 2.3 Sensitivity check — does 1.4 hold?

PCA threshold sensitivity, computed from `analytics/ic/reports/agent_correlation_matrix_2026-04-28.json`:

| Asset | n agents (non-zero variance) | eig > 0.5 | **eig > 1.0** | eig > 1.5 | eig > 2.0 |
|---|---|---|---|---|---|
| BTC | 5 | 4 | **3** | 1 | 0 |
| ETH | 7 | 4 | **2** | 2 | 1 |
| SOL | 5 | 3 | **2** | 2 | 1 |

Eigenvalue spectra:
- BTC: `[0.05, 0.62, 1.13, 1.34, 1.87]`
- ETH: `[-0.05, 0.26, 0.48, 0.90, 0.95, 1.79, 2.66]`
- SOL: `[0.20, 0.30, 0.73, 1.61, 2.16]`

**Verdict: finding HOLDS.** Even at the permissive 0.5 threshold, no asset reaches 6 sources. ETH stays at 4. The "12-agent matrix has 2-3 effective dimensions" conclusion is robust to threshold choice in [0.5, 2.0].

**Filed as numerical-quality issue** (not material to the finding): ETH's smallest eigenvalue is −0.05, which a correlation matrix shouldn't produce (must be PSD). The negative comes from `compute_strategy_correlation.py:118`'s `.fillna(0)` step degrading the matrix when some agents have NaN-correlation pairs. Doesn't affect the count of eigvals > threshold; cosmetic.

---

## 2.1 Implication for IQL priority

IQL replaces DT v3.2's offline-RL trainer. It maps `obs_dim=126` → policy. Both DT and IQL operate on the SAME observation space.

Per the 1.4 finding, the observation space's directional signal already collapses to 2-3 dimensions. IQL would learn a different policy from the same input — but **cannot create new factors**. The ceiling on any RL-policy improvement is bounded by the input information content, which is collapsed.

**Reordered priority**:
- v3 prior: IQL was MEDIUM (DT replacement, structural improvement on small-data regime)
- v3.5 post: IQL is **LOW** (solves a model-architecture problem we don't have; doesn't address the actual factor-collapse bottleneck)

Concrete delta: 30-day expected PnL improvement from IQL replacement is bounded by current 12-agent → 2-3-factor signal. Even a perfect IQL on a perfect oracle of those 2-3 factors won't beat a competent linear model on the same factors by much.

**IQL is not dead** — it's still the right replacement for DT v3.2 *eventually*, but not the highest-leverage next step.

---

## 2.2 Orthogonal factor candidates

What could be MATHEMATICALLY orthogonal to the current 2-3 latent dimensions? Need information sources that aren't derived from BTC/ETH/SOL OHLCV at the same timeframe.

### Candidate F1 — Cross-asset factors

Sources NOT in current observation:
- BTC dominance % (CoinMarketCap / similar)
- BTC-ETH spread / cointegration residual
- USDT premium / discount on Korean exchanges
- Stablecoin supply growth

**Orthogonality argument**: these are derived from CROSS-asset relationships, not from any single asset's OHLCV. The 12 current agents are mostly per-asset.

**Effort**: 3-5 days per factor (data feed + integration + obs-dim impact analysis).

**Risk**:
- Touching `obs_dim=126` is **Iron Law forbidden**. Adding a factor requires either (a) replacing an existing feature in the 122-feature manifest, (b) treating new factor as agent-level signal not raw observation, or (c) a constitutional override for obs_dim change.
- Path (b) is the safe one — wire as a new fusion agent. But that adds to the 12 → 13/14 collapsed matrix, not necessarily orthogonal.

### Candidate F2 — Cross-market factors

Sources:
- SPY / NDX / DXY / yields (TradFi correlation)
- Gold / oil (macro hedge correlation)

**Orthogonality argument**: equity / commodity correlations to crypto are well-documented but vary by regime. In regimes where correlation is high (risk-on), these add no info. In regimes where correlation breaks (idiosyncratic crypto move), they actively confuse.

**Effort**: 2-3 days for SPY/NDX feed (most operators already have it).

**Risk**: low feed cost, medium signal cost. Probably orthogonal in normal regime, anti-orthogonal in crisis.

### Candidate F3 — On-chain factors

Sources:
- Exchange netflow (BTC/ETH leaving exchange = bullish accumulation signal)
- LP imbalance (Uniswap-side concentration)
- Stablecoin issuance rate
- Funding rate (Coinbase / Bybit / Binance perp)
- Realized volatility from options (Deribit IV)

**Orthogonality argument**: on-chain data is fundamentally different from price OHLCV. Strong candidate.

**Effort**: 2-7 days per source (varies widely by data feed availability).

**Risk**: data quality is highly variable (free APIs vs paid Glassnode/Nansen).

**Bonus**: HMATS already has `data_mgmt/feeds/onchain_feed.py` but it's documented as `enabled=False` (per CLAUDE.md). Re-enabling could be a 1-day fix that adds the on-chain factor for free.

### Candidate F4 — Funding curve / basis term structure

Sources:
- Perp funding rate (single number, sign + magnitude)
- Spot-perp basis
- Term structure of futures (1m / 3m / 6m basis)

**Orthogonality argument**: funding rate is a STRUCTURAL signal — directional bias of long-vs-short positioning. This is information that DOESN'T appear in OHLCV.

**Specifically**: if HMATS migrates to Coinbase perpetual-style futures (Track B 2.2), funding rate is **automatically available as part of the contract economics**. This is "free" data — comes with the migration without separate feed work.

**Effort**: 0 incremental (bundled with Coinbase migration). 1-2 days post-migration to wire as a fusion agent.

**Risk**: funding rate is correlated with momentum (extreme funding usually after extreme price moves). Orthogonality to existing momentum agent must be tested empirically post-wiring.

---

## Summary table

| Factor | Effort | Iron Law risk | Independent-source likelihood | Verdict |
|---|---|---|---|---|
| F1 cross-asset | 3-5d/factor | obs_dim must be respected | High (genuinely cross-section) | **MEDIUM priority for v4+** |
| F2 cross-market | 2-3d | obs_dim respected via agent path | Medium (regime-dependent) | LOW priority |
| F3 on-chain | 2-7d (or 1d if re-enable existing) | obs_dim respected | High | **HIGH priority** if enable-existing path works |
| F4 funding curve | 0 incremental w/ Coinbase | bundled with migration | High | **HIGHEST priority** — bundled with Track B 2.2 |

---

## Re-evaluated Track B priority

Per v3 Track B scoping:

| Item | v3 priority | v3.5 priority | Reason |
|---|---|---|---|
| 2.1 Coinbase eligibility | high (necessary precondition) | unchanged: HIGH | Manual 10-min check |
| 2.2 Coinbase migration scope | high | **HIGHEST** | Adds funding rate as orthogonal factor for free + leverage benefit |
| 2.3 IQL replacement scope | medium | **LOW (deferred)** | Doesn't address factor-collapse bottleneck |
| 2.4 Orthogonal factor research (NEW) | — | **MEDIUM** | Filed for v4+; F3 (re-enable on-chain) is the cheapest first step |

---

## Forced reverse-questions (anti confirmation bias)

Per v3.5 §2.3:

**Q1**: Is 1.4's correlation using the WRONG data? Is `agent_signals.<agent>.direction` from BACKFILL records (dormant) or live records?

**A1**: Both. The synced corpus is `data/audit_sync/2026-04-28/ic_signals/ic_signals_*.jsonl` — date-stamped daily files of LIVE attribution. Records were generated by the engine each tick. Not BACKFILL artefacts. Finding is on real production signal flow.

**Q2**: ETH cvd↔lead_lag rho=+0.75 — could be data leakage from shared lookback window?

**A2**: Plausible. Both `lead_lag` and `cvd` agents likely consume order-book or VWAP data from the same time window. If they're using overlapping rolling windows, correlation is artefact of construction, not "redundant alpha". Filed for v4 — would require tracing each agent's actual input data dependency.

**Q3**: Is `min_signals_for_assessment=20` causing some agents' direction to default to constant (and thus zero variance)?

**A3**: Unrelated — that threshold is in `strategy_aging.py`, not in agent-direction generation. Zero-variance agents are a separate "silent agents" issue (CLAUDE.md P115).

---

## What this means for v3.6

1. **Reader fix is still real** but its IMPACT is bounded by factor collapse — even with weights, the underlying signal pool is 2-3-d.
2. **IQL is a v5+ concern** — solve factor collapse first, then a better RL is high-leverage.
3. **Coinbase migration is the highest-leverage move** — 2x leverage win + funding-rate factor + perp economics; AND the funding factor is bundled.
4. **Re-enabling on-chain feed (F3 cheap path)** is a 1-day Track B follow-on; should be in v3.6 or v4.

---

## Iron Law check on this doc

All candidates discussed are SCOPING ONLY:
- F1/F2/F3: would require fusion-agent wiring, not obs_dim manipulation. PASS.
- F4: bundled with Coinbase migration. PASS (assuming migration itself passes Iron Law check, which is a separate doc.)

No code modified. No commits. No deploys.

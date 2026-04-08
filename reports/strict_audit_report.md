# HMATS vs SOTA Strict Audit Report

**Auditor role**: Quant / Systems Auditor (no recommendations, no code, no roadmap)
**Date**: 2026-04-08
**Document A**: CLAUDE.md (HMATS v6.8.0 system documentation)
**Document B**: deep-research-report.md (SOTA multi-agent crypto trading baseline)
**Document C**: Runtime logs, configs, code evidence from live deployment session

---

## PARAGRAPH NUMBERING

### Document A (CLAUDE.md)
- A-001: Project overview (v6.8.0, SOL/ETH/BTC, Kraken)
- A-002: Paper trading progress (GMM, RegimeSmoother, Best-of-N, Alpha Gate, DRL)
- A-003: DRL Authority Policy (FULL mode, auto-demotion config)
- A-004: Authority Matrix (Quant=DECIDE, DRL=DECIDE, Risk=VETO, Sentiment=ADVISE, ShortBias=PENALIZE)
- A-005: Non-negotiable rules (Constitution, P0 Safety, Existence Fuse, DRL Authority, Single Exchange)
- A-006: Completed work items 1-25 (verified)
- A-007: Completed work items 26-34 (awaiting verification, marked ⏳)
- A-008: Development guidelines (testing, code patterns)
- A-009: Key architecture rules (ent_coef, RegimeSmoother, GMM features, Constitution schema, data age)
- A-010: Training commands (ULTIMATE preset)
- A-011: Monitoring (health monitor, training watch, proof logs)
- A-012: Pre-live checklist (partially completed)
- A-013: Known non-critical errors (all resolved)

### Document B (SOTA Research Report)
- B-001: Executive summary (derivatives priority, data/API constraints, strategy pool, multi-agent, risk budget, execution/cost, backtest protocol, public cases, MVP timeline)
- B-002: Assumptions table (assets, exchange, frequency specified; capital, leverage, margin, short, perpetual UNSPECIFIED)
- B-003: Spot fee structure (maker/taker 0.25%/0.40% at lowest tier, post-only option)
- B-004: Derivatives fee structure (maker/taker 0.0200%/0.0500%, order-of-magnitude lower)
- B-005: Margin spot extra costs (opening fee 0.01%-0.05%, 4H rollover, regular trade fees on open/close)
- B-006: Spot OHLC/WebSocket (720 bar limit, WS ohlc interval=240, since parameter limitation)
- B-007: Historical tick-level trade data (CSV/ZIP downloads for long-term backtest + order-flow)
- B-008: Derivatives market data API (tickers, candles, historical-funding-rates)
- B-009: Spot API rate limits (public/private/trading tiers, counter system)
- B-010: Derivatives REST rate limits (token pool mechanism, sendorder=10 tokens)
- B-011: Derivatives margin/leverage (up to 50x, Class A-G, IM/MM, equity protection)
- B-012: Perpetual funding rate mechanics (no expiry, funding alignment, holding cost)
- B-013: Agent types (Market Data, Signal Agents, Risk Agents, Portfolio Allocator, Execution Agents, Ops Agent)
- B-014: Communication mechanism (message bus: JetStream or Redis Streams with durable storage, replay, consumer groups)
- B-015: Priority and arbitration rules (hard risk > margin health > normal rebalance > research)
- B-016: State management (target-position as center, event sourcing, append-only, rebalance_id, client_order_id, deterministic replay)
- B-017: Architecture diagram (mermaid flowchart)
- B-018: Strategy comparison table (TSM, relative strength, risk-managed momentum, CTREND, carry/funding, mean-reversion, order-flow, on-chain)
- B-019: Strategy sketch 1 (4H trend breakout + vol-target leverage)
- B-020: Strategy sketch 2 (3-asset relative strength rotation + risk-managed momentum)
- B-021: Strategy sketch 3 (funding/carry risk premium + trend direction switch)
- B-022: Risk budget example table (TSM 45%, rotation 25%, carry 15%, mean-revert 10%, order-flow/on-chain 5%)
- B-023: Correlation correction (total leverage cap, per-coin cap, correlation penalty, portfolio vol target)
- B-024: Capacity estimation (ADV constraint, orderbook depth, impact cost model, cost stress test)
- B-025: Execution strategy (maker post-only priority, taker for risk exits, slicing, TWAP/VWAP, rate-limit budget)
- B-026: Multi-layer risk framework (vol-target, per-agent drawdown, soft/hard stop, regime kill-switch, exchange/ops circuit breaker)
- B-027: Monitoring metrics (data latency, order/fill stats, risk metrics, cost metrics)
- B-028: Non-standard errors in crypto research (data source, sample processing, portfolio construction biases)
- B-029: Data and simulation requirements (point-in-time, fee simulation, slippage model, funding/rollover simulation)
- B-030: Walk-forward + purged CV + OOS steps (A through G)
- B-031: Public multi-agent cases (5+ systems: LLM portfolio, adaptive BTC, zero-shot, TradingAgents, CryptoTrade, Temporal-based)
- B-032: Open-source tools (CCXT, Hummingbot, Freqtrade, Backtrader, Ray, Temporal, NATS JetStream, Redis Streams)
- B-033: Data source checklist (Kraken Spot, Derivatives, on-chain)
- B-034: Key module pseudocode (orchestrator + execution loop)
- B-035: MVP deployment timeline (7 weeks)

### Document C (Runtime Evidence)
- C-001: Live container status (hmats-live, healthy, 0 restarts, DRL=ACTIVE, 3/3 models)
- C-002: Account sync ($9,400 equity, 0 positions)
- C-003: Fee tier log (P0-FIX: blended taker=0.0bps, maker=0.0bps at $0 volume)
- C-004: PA_PROOF (friction=50bps pre-fix, edge insufficient)
- C-005: DMS (dedicated client, 60s timeout, heartbeat 24s)
- C-006: Veto chain logs (ALPHA_GATE pass/block, STRUCTURE bypass, REGIME_POWER)
- C-007: Sentiment L1 (F&G=11 extreme_fear, zscore=-2.34, adopted in fusion)
- C-008: DRL shadow output (action=+0.94, conf=0.42, all assets strongly positive)
- C-009: Regime detection (GMM conf 0.85-1.00, VOLATILE_CHOP/NEUTRAL_DRIFT/MOMENTUM_RALLY)
- C-010: Execution failure (EOrder:Insufficient funds on all 4 slices, pre-fix)
- C-011: cloud_production.json (risk params, alpha_boost, gambler, DRL config, sentiment_gate)
- C-012: Anti-churn defaults (AC-1=8h, AC-2=2/asset, AC-5=8/day)
- C-013: Constitution (MAX_ALPHA_BPS=200, friction components)
- C-014: OOD detector (122-dim, thresholds 17-18, scalers 3/3 loaded after joblib fix)
- C-015: No dashboard_state.json written in LIVE mode (pre-fix, now fixed)
- C-016: Zero executed trades in 20+ hours of live operation

---

# COVERAGE MATRIX

| Module | Baseline Reqs | Matched | Partial | Missing | Weaker | Contradicts | Unverifiable | Out_of_scope | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1. Assumptions & Boundaries | 8 | 3 | 2 | 1 | 1 | 0 | 1 | 0 | Product type unresolved |
| 2. Kraken Exchange Constraints | 12 | 5 | 3 | 2 | 1 | 1 | 0 | 0 | Derivatives not used |
| 3. Data Layer | 9 | 4 | 2 | 2 | 0 | 0 | 1 | 0 | Historical depth weak |
| 4. Multi-Agent Architecture | 8 | 4 | 2 | 2 | 0 | 0 | 0 | 0 | No durable bus, no replay |
| 5. Alpha / Strategy Pool | 9 | 5 | 2 | 1 | 1 | 0 | 0 | 0 | Mostly covered |
| 6. Portfolio & Risk Budget | 7 | 3 | 2 | 1 | 1 | 0 | 0 | 0 | Vol target weak |
| 7. Execution & Cost Model | 10 | 4 | 2 | 2 | 1 | 1 | 0 | 0 | PA friction blocker |
| 8. Risk & Circuit Breakers | 8 | 5 | 2 | 1 | 0 | 0 | 0 | 0 | Strong overall |
| 9. Backtest & Validation | 8 | 1 | 3 | 3 | 1 | 0 | 0 | 0 | Weakest module |
| 10. Ops & Monitoring | 7 | 4 | 2 | 0 | 0 | 0 | 1 | 0 | Adequate |
| 11. Tool Stack & Implementability | 6 | 4 | 1 | 0 | 0 | 0 | 1 | 0 | Mostly landed |
| **TOTAL** | **92** | **42** | **23** | **15** | **6** | **2** | **4** | **0** | |

---

# GAP LIST

## GAP-001
- module: 1. Assumptions & Boundaries
- submodule: Trading product type
- baseline_requirement: System should evaluate derivatives (perpetual/futures) as primary venue for 4H high-risk trading due to 10x lower fees
- status: CONTRADICTS_BASELINE
- primary_gap_type: engineering
- secondary_tags: exchange_constraints, fees, assumptions
- severity: S1
- priority: P0
- baseline_evidence:
  - [B-001] "若你的目标确实是'高风险高收益 + 小资金'，则'衍生品（永续/期货）+ 严格风控'的可行性非常高于'现货频繁换手'"
  - [B-004] "Derivatives maker/taker约为 0.0200% / 0.0500%"
  - [B-003] "Spot maker/taker约为 0.25% / 0.40%"
- our_system_evidence:
  - [A-005] "Single exchange — Kraken only (Binance/Deribit in legacy/)"
  - [A-007] "#31 Cash-and-carry ⏳ Delta-neutral module, signal-only Phase 1 (Kraken Futures not wired)"
  - [C-003] "Fee tier: FREE ($0 volume), taker_allowed=True" — currently on free tier but will exceed $10K
- gap_statement: System operates spot-only. Baseline explicitly states derivatives are strongly preferred for 4H high-risk trading due to 10x fee advantage. Cash-and-carry module exists but marked ⏳ (not wired to Kraken Futures).
- why_it_matters: Once monthly volume exceeds $10K free tier, spot taker fee of 40bps vs derivatives 5bps creates a 35bps/trade cost disadvantage that erodes most mid-frequency alpha.
- hidden_assumption_risk: high — System assumes free-tier fees persist; this breaks at any meaningful trading volume
- overfitting_risk: low
- cost_underestimation_risk: high — All alpha estimates assume free-tier or low fees; post-free-tier cost structure invalidates current alpha gate calibration

## GAP-002
- module: 1. Assumptions & Boundaries
- submodule: Leverage boundary
- baseline_requirement: Leverage range and dynamic adjustment mechanism should be explicitly specified with vol-target scaling
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: leverage, risk, assumptions
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-002] "默认建议 Lmax=3–8（高风险但较小仓位）"
  - [B-019] "杠杆：lev_t = min(Lmax, σ*/σ_hat)"
- our_system_evidence:
  - [A-007] "#26 Regime-conditional leverage ⏳ VOLATILE_CHOP=3x, MOMENTUM_RALLY/PANIC_SELLOFF=2x, others=1x"
  - [C-006] "VOLATILE_CHOP -> 3.0x requested, then SOFT veto -> clamped to 1x"
- gap_statement: Leverage exists but is static per-regime (not vol-target driven). Runtime evidence shows 3x was requested but vetoed to 1x. Baseline recommends dynamic vol-target-driven leverage lev_t = min(Lmax, σ*/σ_hat).
- why_it_matters: Static leverage map does not adapt to changing volatility within a regime. High leverage in suddenly-volatile conditions increases liquidation risk.
- hidden_assumption_risk: medium — Assumes regime label correctly captures current vol level
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-003
- module: 1. Assumptions & Boundaries
- submodule: Capital size and capacity constraints
- baseline_requirement: Capital size assumptions and their impact on strategy viability should be explicit
- status: UNVERIFIABLE
- primary_gap_type: documentation
- secondary_tags: assumptions, capacity
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-002] "初始资金规模：未指定"
  - [B-024] "容量估算是严谨回测的一部分"
- our_system_evidence:
  - [C-011] "initial_capital_default: 10000.0"
  - [C-002] "Live equity: $9,400.36"
- gap_statement: $10K capital is specified in config but no formal analysis of capacity constraints, minimum viable capital, or ADV impact exists in documentation.
- why_it_matters: At $10K, capacity is not a binding constraint, but there is no documented analysis proving this.
- hidden_assumption_risk: low — $10K is genuinely small enough that capacity is unlikely to matter
- overfitting_risk: low
- cost_underestimation_risk: low
- audit_note: This is primarily a documentation gap. The small capital actually makes capacity irrelevant, but this assumption should be stated explicitly.

## GAP-004
- module: 2. Kraken Exchange Constraints
- submodule: Margin/rollover cost modeling
- baseline_requirement: Margin opening fee and 4H rollover fee must be modeled if margin spot is used
- status: MISSING
- primary_gap_type: engineering
- secondary_tags: fees, margin, funding
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-005] "保证金现货在常规交易费之外，还会收取'开仓费 + rollover费（每4小时）'"
  - [B-005] "开仓 0.01%–0.05%，rollover每4小时周期"
- our_system_evidence:
  - [A-007] "#26 Regime-conditional leverage ⏳ ... Kraken isolated margin"
  - No evidence of rollover fee modeling in A or C
- gap_statement: System references Kraken isolated margin for leverage but no evidence of margin opening fee or 4H rollover fee modeling in cost calculations, alpha gate friction, or backtest simulations.
- why_it_matters: If margin spot is used for leverage, unmodeled rollover fees at 4H granularity could significantly erode returns, especially for positions held >4H.
- hidden_assumption_risk: high — Using margin without modeling its specific costs
- overfitting_risk: low
- cost_underestimation_risk: high — Rollover fees compound per 4H period, directly impacting the 4H decision cycle

## GAP-005
- module: 2. Kraken Exchange Constraints
- submodule: Derivatives API integration
- baseline_requirement: Derivatives market data and execution APIs should be integrated for cost-optimal trading
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: exchange_constraints, execution, fees
- severity: S1
- priority: P1
- baseline_evidence:
  - [B-008] "Kraken Futures/Derivatives API提供公开市场数据"
  - [B-004] "衍生品 maker/taker约为 0.0200% / 0.0500%"
- our_system_evidence:
  - [A-007] "#31 Cash-and-carry ⏳ ... Kraken Futures not wired"
  - [C-009] "Kraken Futures feed loaded (public API, no key needed)" — market data only, no execution
- gap_statement: Kraken Futures public data feed is connected (funding rates, OI). But execution via Derivatives is not wired. All trades execute on spot.
- why_it_matters: Derivatives execution would reduce per-trade fees by ~10x and enable native short selling without margin spot rollover costs.
- hidden_assumption_risk: medium
- overfitting_risk: low
- cost_underestimation_risk: high

## GAP-006
- module: 2. Kraken Exchange Constraints
- submodule: Rate limit budget as first-class resource
- baseline_requirement: Rate limit budget should be managed with priority routing (risk actions > normal rebalance)
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: rate_limits, execution, architecture
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-009] "trading（下单/撤单）端点'按账户 + 交易对'并基于订单与订单簿交互的'点数系统'限流"
  - [B-010] "Derivatives REST端点使用'token pool'机制"
  - [B-025] "执行代理必须把'限流预算'当作一等公民（first-class resource）"
- our_system_evidence:
  - [A-005] "P0 Safety cannot be bypassed — kill switch, stale data guard, rate limiter"
  - [C-010] "Kraken rate_limit=15" in config
- gap_statement: Rate limiter exists but no evidence of priority routing where risk/cancel actions get higher priority than normal rebalance orders. The rate limit is a flat cap (15), not a budgeted resource with tiered access.
- why_it_matters: During high-volatility periods, risk exits competing with normal rebalance for rate limit tokens could delay critical protective actions.
- hidden_assumption_risk: medium
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-007
- module: 2. Kraken Exchange Constraints
- submodule: Fee tier progression modeling
- baseline_requirement: Fee tier changes based on 30-day trading volume should be modeled for cost projections
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: fees, cost
- severity: S2
- priority: P2
- baseline_evidence:
  - [B-003] "最近30天交易量"分档费率
  - [B-029] "按Kraken费率表逐笔扣费，并做费率分档敏感性分析"
- our_system_evidence:
  - [A-002] "Alpha Gate: Dynamic volume-aware thresholds. Free tier (<$10K/mo)"
  - [C-003] "KrakenPlusFeeBlender initialized: enabled=True, free_tier=$10,000, blend_band=$2,000"
- gap_statement: Free-tier blending exists for the $0-$12K band. But no sensitivity analysis or projection of fee impact as volume scales beyond free tier.
- why_it_matters: Alpha gate thresholds calibrated for free-tier may not hold when fees increase 40x at $10K+ volume.
- hidden_assumption_risk: medium
- overfitting_risk: low
- cost_underestimation_risk: high

## GAP-008
- module: 3. Data Layer
- submodule: Historical data for long-term backtest
- baseline_requirement: Multi-year historical data (2020-2026) including tick-level trades for order-flow features and long-term backtest
- status: MISSING
- primary_gap_type: research
- secondary_tags: data, backtest, data_quality
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-006] "一次最多返回720条且无法通过since拉取更旧数据"
  - [B-007] "Kraken提供'交易对的完整逐笔交易历史CSV/ZIP下载'"
- our_system_evidence:
  - [A-002] "GMM Retrained: 721 bars/asset"
  - No evidence of historical tick-level data download or multi-year backtest dataset
- gap_statement: GMM trained on 721 bars (~120 days at 4H). No evidence of multi-year backtest data. Baseline identifies 720-bar REST limit and recommends historical CSV/ZIP downloads for 2020-2026 coverage.
- why_it_matters: 721 bars covers ~4 months, missing major bear markets (2022), regime transitions, and black swan events. Strategy validation on 4 months of data has high overfitting risk.
- hidden_assumption_risk: medium
- overfitting_risk: high — Only 4 months of regime data may not capture full regime diversity
- cost_underestimation_risk: low

## GAP-009
- module: 3. Data Layer
- submodule: Point-in-time data guarantee
- baseline_requirement: All external/on-chain indicators must have point-in-time timestamps and correction rules to prevent lookahead bias
- status: UNVERIFIABLE
- primary_gap_type: validation
- secondary_tags: data_quality, leakage
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-029] "任何链上/外部指标必须带'当时可见'的时间戳与修正规则"
  - [B-028] "数据源、样本处理、组合构造等'看似无害的选择'会导致组合表现出现巨大差异"
- our_system_evidence:
  - [A-009] "Data age uses exchange timestamp, MAX_DATA_AGE_SECONDS=10.0"
  - No evidence of point-in-time enforcement for on-chain features or sentiment data in backtesting
- gap_statement: Runtime data age enforcement exists (10s max). But no evidence of point-in-time data discipline for on-chain signals (CryptoCompare, Solana RPC), sentiment (F&G), or Coinglass data in the backtest/training pipeline.
- why_it_matters: On-chain and sentiment data often has reporting delays or retroactive corrections. Without point-in-time enforcement in training, models may have been trained with lookahead bias.
- hidden_assumption_risk: high
- overfitting_risk: high — Backtest/training may use data that was not available at decision time
- cost_underestimation_risk: low

## GAP-010
- module: 3. Data Layer
- submodule: Data versioning and reproducibility
- baseline_requirement: Data versions should be locked and reproducible for backtest audit
- status: MISSING
- primary_gap_type: engineering
- secondary_tags: data, data_quality, backtest
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-016] "锁定数据版本"
  - [B-030] "步骤A：构建4H主数据集...并锁定数据版本"
- our_system_evidence:
  - No evidence found in A or C of data versioning
- gap_statement: No data versioning or snapshot mechanism documented. Training data files exist but no version control, checksums, or reproducibility guarantee.
- why_it_matters: Without data versioning, model training cannot be reproduced or audited. Changes to data sources may silently alter model behavior.
- hidden_assumption_risk: medium
- overfitting_risk: medium
- cost_underestimation_risk: low

## GAP-011
- module: 4. Multi-Agent Architecture
- submodule: Durable message bus with replay
- baseline_requirement: Inter-agent communication via durable, replayable message stream (not ephemeral pub/sub)
- status: MISSING
- primary_gap_type: engineering
- secondary_tags: architecture, replay, deployment
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-014] "用可持久化消息系统（如JetStream/Redis Streams）提供'可重放的事件日志'"
  - [B-016] "任何'状态写入'必须是追加式（append-only）并可追踪版本"
- our_system_evidence:
  - [C-001] EventBus exists (in-memory pub/sub)
  - Shadow ledger provides append-only fill records
  - archive/infra/event_replay.py exists but is ARCHIVED
- gap_statement: EventBus is ephemeral in-memory pub/sub. Baseline requires durable message stream with replay capability. Shadow ledger partially covers fill audit trail but not the full event stream (signals, targets, risk events).
- why_it_matters: Without durable event stream, post-incident analysis cannot replay the exact sequence of signals, decisions, and executions that led to a loss.
- hidden_assumption_risk: low
- overfitting_risk: low
- cost_underestimation_risk: low
- audit_note: For a single-process $10K system, ephemeral EventBus is functionally adequate. The gap is real vs SOTA but severity is lower at current scale.

## GAP-012
- module: 4. Multi-Agent Architecture
- submodule: Target-position state model with event sourcing
- baseline_requirement: System state should be organized as target_position[t] -> execution reconciliation, with event sourcing for audit
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: architecture, idempotency, replay
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-016] "系统状态按两个层次：Target State + Execution State"
  - [B-034] "Orchestrator: every 4H tick produces rebalance_id and target_position event"
- our_system_evidence:
  - [A-004] "Authority Matrix" defines signal -> fusion -> intent flow
  - TradeIntentV36 serves as target intent object
  - No rebalance_id, no formal target vs actual reconciliation loop
- gap_statement: TradeIntentV36 is functionally similar to a target position, but there is no explicit rebalance_id, no formal target-vs-actual reconciliation loop, and no event-sourced audit trail of target -> execution mapping.
- why_it_matters: Without explicit target-vs-actual reconciliation, it is harder to detect and diagnose execution drift, partial fills, or position mismatches.
- hidden_assumption_risk: low
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-013
- module: 5. Alpha / Strategy Pool
- submodule: Risk-managed momentum (vol-scaling overlay)
- baseline_requirement: Momentum strategies should have explicit vol-target scaling overlay (not just regime power multiplier)
- status: WEAKER_THAN_BASELINE
- primary_gap_type: research
- secondary_tags: alpha, risk, portfolio
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-018] "风险管理动量可显著改善年化Sharpe（1.12→1.42）并提高周收益（3.18%→3.47%）"
  - [B-020] "对每条腿应用 lev_t = min(Lmax, σ*/σ_hat) 的风险缩放"
- our_system_evidence:
  - [A-002] "4 independent strategies (mean_revert, momentum, volume_breakout, vrp)"
  - Regime power multiplier applies blanket scaling but not per-strategy vol-target
  - VolatilityTargetingSizer exists in risk layer but operates at portfolio level, not strategy level
- gap_statement: Momentum strategy exists but lacks dedicated vol-target scaling. Baseline cites concrete evidence that vol-managed momentum improves Sharpe from 1.12 to 1.42. System uses regime power (blunt) instead of continuous vol-scaling (precise).
- why_it_matters: Regime power is a discrete multiplier (e.g., 1.3 for MOMENTUM_RALLY). Vol-target is continuous and adapts within regimes. The discrete approach misses intra-regime volatility changes.
- hidden_assumption_risk: medium — Regime label may not capture current vol accurately
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-014
- module: 5. Alpha / Strategy Pool
- submodule: Relative strength rotation as independent strategy
- baseline_requirement: Cross-sectional relative strength (rank-based rotation among 3 assets) as distinct alpha source
- status: MISSING
- primary_gap_type: research
- secondary_tags: alpha, portfolio
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-020] "横截面相对强弱：r_i = log(close_i/close_i[-k])，排序 long winner / short loser"
  - [B-022] "相对强弱轮动 25% 风险预算"
- our_system_evidence:
  - [A-002] "4 independent strategies (mean_revert, momentum, volume_breakout, vrp)"
  - No cross-sectional ranking strategy found
- gap_statement: Best-of-N selects the strongest strategy per asset independently. There is no cross-asset relative strength rotation (rank assets against each other, overweight winner, underweight loser).
- why_it_matters: With only 3 assets, cross-sectional alpha is limited, but the signal is orthogonal to time-series momentum and could add diversification.
- hidden_assumption_risk: low — 3-asset cross-section has limited statistical power
- overfitting_risk: medium — Small cross-section prone to overfitting
- cost_underestimation_risk: low

## GAP-015
- module: 6. Portfolio & Risk Budget
- submodule: Volatility target level for high-risk mandate
- baseline_requirement: Portfolio-level vol target should be 60-120% annualized for high-risk high-return mandate
- status: WEAKER_THAN_BASELINE
- primary_gap_type: engineering
- secondary_tags: risk, leverage, portfolio
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-019] "目标年化波动：σ* = 60%–120%（高风险）"
  - [B-026] "组合年化波动目标 σ*"
- our_system_evidence:
  - [C-001] "VolAlphaRisk: portfolio_vol=2.35% utilization=117.4% ... status=CRITICAL"
  - Vol target appears to be ~32% annual based on code analysis
- gap_statement: System vol target (~32% annual) is less than half the baseline minimum (60%) for a high-risk mandate. Runtime shows portfolio vol at 2.35% (annualized ~45%) flagged as CRITICAL, suggesting the system considers even moderate vol as dangerous.
- why_it_matters: Low vol target directly limits position sizes and utilization, reducing achievable returns. The system's stated goal of "high risk high return" contradicts its conservative vol targeting.
- hidden_assumption_risk: medium — Assumes conservative vol = safe; may actually cap returns below meaningful levels
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-016
- module: 6. Portfolio & Risk Budget
- submodule: Formal per-agent risk budget allocation
- baseline_requirement: Explicit risk budget allocation across strategy agents (e.g., TSM 45%, rotation 25%, carry 15%, etc.)
- status: MISSING
- primary_gap_type: engineering
- secondary_tags: portfolio, risk, architecture
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-022] Risk budget table: TSM 45%, rotation 25%, carry 15%, mean-revert 10%, order-flow 5%
- our_system_evidence:
  - [A-004] Authority Matrix defines roles (DECIDE/VETO/ADVISE) but no risk budget allocation
  - Best-of-N is winner-take-all, not weighted allocation
  - [C-011] "thesis_budget=$200 per thesis (2% of $10K)"
- gap_statement: No explicit per-strategy risk budget. Best-of-N is winner-take-all: the strongest strategy gets all allocation, others get zero. Thesis budget limits per-thesis capital but does not allocate across strategy types.
- why_it_matters: Winner-take-all concentration means the portfolio has zero diversification across alpha sources in any given tick. If the winning strategy is wrong, there is no hedging from other strategies.
- hidden_assumption_risk: medium
- overfitting_risk: medium — Concentrated bets amplify strategy selection errors
- cost_underestimation_risk: low

## GAP-017
- module: 7. Execution & Cost Model
- submodule: PA executor edge multiplier calibration
- baseline_requirement: Execution edge validation should be calibrated to actual fee tier, not hardcoded multiplier
- status: CONTRADICTS_BASELINE
- primary_gap_type: engineering
- secondary_tags: execution, fees, cost
- severity: S0
- priority: P0
- baseline_evidence:
  - [B-025] "当信号不是紧急时...优先使用限价并启用post-only"
  - [B-004] "Derivatives费率数量级差异使得策略在衍生品上更具可实现性"
- our_system_evidence:
  - [C-004] "PA_PROOF: friction=50.0bps, edge insufficient (32.3 bps < 3.0× 50.0 bps)"
  - Pre-fix: MIN_EDGE_MULTIPLIER=3.0, KRAKEN_TAKER_FEE_BPS overridden to 40bps by exchange query
  - Post-fix: MIN_EDGE_MULTIPLIER=1.5, fees blended to 0bps for free tier
- gap_statement: PA executor was requiring alpha > 3x friction (150bps) to trade, blocking ALL signals. This directly contradicted the alpha gate which passed at 32bps > 11bps. Two parallel cost systems (constitution vs PA executor) used different fee sources, creating an irreconcilable gate. Fixed post-audit but the architectural duplication remains.
- why_it_matters: This was the root cause of zero live trades in 20+ hours. The system generated valid signals, approved them through the alpha gate, then killed them at the execution layer with a different cost model.
- hidden_assumption_risk: high — Two independent fee paths can silently diverge again
- overfitting_risk: low
- cost_underestimation_risk: low — The opposite problem: cost was OVER-estimated, blocking trades
- audit_note: Fix has been deployed (C-003 shows taker=0.0bps post-fix). But the architectural duplication (constitution friction vs PA friction) is an ongoing fragility.

## GAP-018
- module: 7. Execution & Cost Model
- submodule: Pre-order balance validation
- baseline_requirement: Order sizing must validate against available balance before submission
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: execution, risk
- severity: S1
- priority: P0
- baseline_evidence:
  - [B-025] "处理限流预算，并对每一个订单/执行批次做幂等与可追踪"
- our_system_evidence:
  - [C-010] "EOrder:Insufficient funds on all 4 slices"
  - Post-fix: _clamp_size_to_balance() added
- gap_statement: Orders were submitted without checking available balance, causing Kraken to reject with "Insufficient funds". The equity figure ($9,400) included non-USD assets (BABY, BTC), but available USD for margin may have been much less. Fix deployed but the root cause (equity vs available balance confusion) indicates a broader account state modeling gap.
- why_it_matters: Every rejected order consumes rate limit tokens, wastes execution time, and leaves the system unable to take positions during valid signal windows.
- hidden_assumption_risk: high — System assumes equity ≈ available trading capital
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-019
- module: 7. Execution & Cost Model
- submodule: VWAP / advanced execution algorithms
- baseline_requirement: VWAP execution using real-time trade stream volume profile
- status: MISSING
- primary_gap_type: engineering
- secondary_tags: execution, slippage
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-025] "以交易量为权重的VWAP（需要实时交易量信息）"
- our_system_evidence:
  - TWAP exists as enum type in execution/level2_analyzer.py
  - Maker reprice loop (3 attempts, 20s each) serves as simplified TWAP
  - No VWAP implementation found
- gap_statement: No VWAP implementation. Maker reprice loop provides basic execution improvement but does not incorporate volume profile.
- why_it_matters: At $10K account size, market impact is negligible. VWAP becomes relevant only at scale.
- hidden_assumption_risk: low
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-020
- module: 7. Execution & Cost Model
- submodule: Funding rate injection in cost model
- baseline_requirement: Perpetual funding rates must be injected into backtest cost simulation
- status: PARTIAL
- primary_gap_type: validation
- secondary_tags: funding, backtest, cost
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-029] "永续：使用历史资金费率序列...注入现金流"
- our_system_evidence:
  - [A-007] "#32 Funding rate arb ⏳ ShortBiasAgent"
  - [C-009] "BTC funding=0.003127" fetched at runtime
  - Funding data is used for signal generation but no evidence of funding cost injection in backtest/training
- gap_statement: Funding rates are fetched and used as trading signals (carry/short-bias). But no evidence that funding costs are injected as holding costs in backtest simulations or DRL training reward functions.
- why_it_matters: If DRL or strategies train without funding cost, they may learn to hold positions that look profitable in backtest but are eroded by real funding payments.
- hidden_assumption_risk: high — Training without funding costs creates optimistic bias
- overfitting_risk: high — Models may learn to hold positions that are unprofitable after funding
- cost_underestimation_risk: high

## GAP-021
- module: 8. Risk & Circuit Breakers
- submodule: Margin health / liquidation awareness
- baseline_requirement: Real-time margin health monitoring with liquidation price awareness and pre-emptive deleveraging
- status: MISSING
- primary_gap_type: engineering
- secondary_tags: margin, risk, kill_switch
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-011] "Derivatives提供'维持保证金/初始保证金'与'防止账户为负'的净值保护流程"
  - [B-026] "hard：账户净值或保证金健康度接近清算阈值时，强制reduce-only平仓"
- our_system_evidence:
  - [A-005] "Existence Fuse — 28d window, -5% PnL → system halt"
  - [C-011] "hard_drawdown_halt: 0.25, max_drawdown: 0.35"
  - No evidence of margin health / liquidation price monitoring
- gap_statement: Drawdown limits and existence fuse exist. But no real-time margin health monitoring (IM/MM ratio), liquidation price tracking, or pre-emptive deleveraging based on proximity to liquidation.
- why_it_matters: With leverage up to 3x, a sudden 20-30% price move could bring the account near liquidation faster than the 4H tick cycle can respond.
- hidden_assumption_risk: high — 4H tick cycle may be too slow to respond to liquidation-proximity events
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-022
- module: 9. Backtest & Validation
- submodule: Walk-forward validation protocol
- baseline_requirement: Formal walk-forward with rolling/expanding windows (e.g., 12mo train, 3mo validate, 3mo test)
- status: PARTIAL
- primary_gap_type: validation
- secondary_tags: backtest, oos, leakage
- severity: S1
- priority: P0
- baseline_evidence:
  - [B-030] "步骤D：Walk-forward分割：训练12个月→验证3个月→测试3个月（滚动推进或expanding window）"
- our_system_evidence:
  - [A-010] "3-fold, position_direction feature, 126 dims, early stopping, Optuna"
  - DRL training uses 3-fold time-series split with gap=42 bars
  - GMM trained on 721 bars (~4 months)
  - No evidence of formal walk-forward protocol with the specified window sizes
- gap_statement: DRL uses 3-fold time-series CV with 42-bar gap (1 week purge). But no evidence of formal walk-forward with 12mo/3mo/3mo windows. GMM training window (721 bars ≈ 4 months) is too short for walk-forward with the recommended windows.
- why_it_matters: Without walk-forward, there is no evidence that strategy parameters generalize beyond the training period. 3-fold CV with a 4-month dataset means each fold sees ~1.3 months of test data — insufficient to capture regime diversity.
- hidden_assumption_risk: medium
- overfitting_risk: high — Short training window + Optuna hyperparameter search = high overfitting risk
- cost_underestimation_risk: low

## GAP-023
- module: 9. Backtest & Validation
- submodule: Purged cross-validation with embargo
- baseline_requirement: When using ML/optimization, purged CV with embargo is required to prevent information leakage
- status: PARTIAL
- primary_gap_type: validation
- secondary_tags: backtest, leakage
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-030] "步骤E：Purged CV...purge掉与标签形成期重叠的训练样本，并加embargo（例如1–2个4H bar）"
- our_system_evidence:
  - [A-010] "3-fold, ... early stopping, Optuna"
  - DRL training has gap=42 bars between train/val
  - WalkForwardValidator exists in training/utils/improvements.py
  - No formal PurgedKFold implementation
- gap_statement: gap=42 bars provides basic purging (1 week). But no formal PurgedKFold implementation with embargo exists. Baseline recommends combinatorial purged CV which is not implemented.
- why_it_matters: With Optuna hyperparameter optimization, information leakage between folds would make validation metrics unreliable. The 42-bar gap helps but is not a formal purged CV implementation.
- hidden_assumption_risk: medium
- overfitting_risk: high — Optuna + non-purged CV can select hyperparameters that exploit leaked information
- cost_underestimation_risk: low

## GAP-024
- module: 9. Backtest & Validation
- submodule: Cost stress testing
- baseline_requirement: Backtests must include cost stress tests (slippage ×2/×3, fee tier sensitivity)
- status: MISSING
- primary_gap_type: validation
- secondary_tags: backtest, cost, slippage
- severity: S2
- priority: P1
- baseline_evidence:
  - [B-030] "步骤F：压力测试：成本×2/×3"
  - [B-024] "应强制做'cost stress test'（例如滑点×2、×3）"
- our_system_evidence:
  - No evidence of cost stress testing in A or C
- gap_statement: No cost stress testing (slippage ×2/×3, fee sensitivity) documented or implemented in the training or validation pipeline.
- why_it_matters: Strategies that are marginally profitable at base costs may become unprofitable under stress conditions. Without stress testing, there is no evidence of cost robustness.
- hidden_assumption_risk: high — Profitability assumes best-case execution costs
- overfitting_risk: medium
- cost_underestimation_risk: high — No robustness test against cost increases

## GAP-025
- module: 9. Backtest & Validation
- submodule: Paper trading minimum duration
- baseline_requirement: Minimum 4-8 weeks of paper trading before live
- status: WEAKER_THAN_BASELINE
- primary_gap_type: validation
- secondary_tags: backtest, oos
- severity: S1
- priority: P0
- baseline_evidence:
  - [B-030] "步骤G：paper trading（实盘仿真）：至少4–8周，以4H周期运行"
- our_system_evidence:
  - [A-012] Pre-live checklist shows paper run verified Feb 13
  - [C-016] Zero executed trades in live; paper run had 0 closed trades before going live
  - System went from paper to live without completing paper trading validation
- gap_statement: System went live with zero closed paper trades. Baseline requires 4-8 weeks minimum paper trading. The pre-live checklist (A-012) shows several items still unchecked including "Health monitor shows no CRITICAL alerts for 24h" and "DRL models deployed + 30 shadow trades for promotion".
- why_it_matters: Going live without paper validation means there is no evidence that the signal-to-execution-to-PnL pipeline works end-to-end in realistic conditions.
- hidden_assumption_risk: high — No evidence the system can profitably close trades
- overfitting_risk: high — No OOS performance evidence
- cost_underestimation_risk: high — No real execution cost evidence

## GAP-026
- module: 9. Backtest & Validation
- submodule: Non-standard error awareness in crypto research
- baseline_requirement: System should document awareness and mitigation of non-standard errors (data source bias, sample processing bias, portfolio construction bias)
- status: MISSING
- primary_gap_type: documentation
- secondary_tags: backtest, leakage, assumptions
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-028] "数据源、样本处理、组合构造等'看似无害的选择'会导致组合表现出现巨大差异"
- our_system_evidence:
  - No evidence found in A of non-standard error documentation
- gap_statement: No explicit documentation of potential non-standard errors or their mitigations. With only 3 assets and limited history, the system is especially susceptible to data source and sample processing biases.
- why_it_matters: Without awareness, the system may unknowingly incorporate biases that inflate backtest performance.
- hidden_assumption_risk: high
- overfitting_risk: high
- cost_underestimation_risk: low

## GAP-027
- module: 10. Ops & Monitoring
- submodule: Order audit trail completeness
- baseline_requirement: Complete order-level audit trail with order_id, intent, placement, fill, slippage, fee per trade
- status: PARTIAL
- primary_gap_type: engineering
- secondary_tags: monitoring, execution, replay
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-027] "订单/成交：订单拒绝率、部分成交率、平均滑点（bps）、maker占比"
- our_system_evidence:
  - Shadow ledger records fills with price/fee/realized_pnl
  - Proof logs record intent and decision chain
  - No unified order-level audit trail linking intent -> order -> fill -> PnL
- gap_statement: Shadow ledger captures fills. Proof logs capture intents. But there is no unified audit trail linking a specific proof log entry to its corresponding order placement, partial fills, reprices, and final PnL.
- why_it_matters: Post-trade analysis requires linking decisions to outcomes. Separated logs make this correlation manual and error-prone.
- hidden_assumption_risk: low
- overfitting_risk: low
- cost_underestimation_risk: low

## GAP-028
- module: 10. Ops & Monitoring
- submodule: Replayable state for post-incident analysis
- baseline_requirement: System state should be replayable for post-incident forensics
- status: UNVERIFIABLE
- primary_gap_type: engineering
- secondary_tags: replay, monitoring, architecture
- severity: S3
- priority: P2
- baseline_evidence:
  - [B-016] "映射过程可被replay（重放），以便审计与定位问题"
- our_system_evidence:
  - archive/infra/event_replay.py exists but is ARCHIVED
  - Proof logs provide human-readable decision trail
  - No evidence of automated replay capability
- gap_statement: Event replay module exists in archive but is not in the live code path. Proof logs provide a narrative but are not machine-replayable for automated forensics.
- why_it_matters: After a significant loss event, replaying the exact state and decisions is critical for root cause analysis.
- hidden_assumption_risk: low
- overfitting_risk: low
- cost_underestimation_risk: low

---

# TOP BLOCKERS

- **GAP-017 (S0/P0)** — PA executor edge multiplier (3x) and fee source mismatch blocked ALL trades for 20+ hours. Fix deployed but architectural duplication (two independent fee paths) remains fragile.
- **GAP-018 (S1/P0)** — Orders submitted without balance validation, all rejected by exchange. Fix deployed but equity-vs-available-balance confusion indicates incomplete account state modeling.
- **GAP-022 (S1/P0)** — No formal walk-forward validation with recommended window sizes. DRL trained on ~4 months with 3-fold CV and Optuna, creating high overfitting risk with no long-term OOS evidence.
- **GAP-025 (S1/P0)** — System went live with zero closed paper trades. Baseline requires 4-8 weeks minimum paper trading. No evidence the pipeline can profitably close trades.
- **GAP-001 (S1/P0)** — Spot-only execution contradicts baseline's strong recommendation for derivatives-first at 4H frequency. Fee disadvantage of up to 35bps/trade once free tier is exceeded.

---

# UNVERIFIABLE CLAIMS

| Claim | Evidence Seen | Why Unverifiable |
|-------|---------------|------------------|
| DRL models produce actionable exits | DRL shadow output: action=+0.94 for all assets [C-008] | No closed trades attributable to DRL. All DRL actions are uniformly +0.93-0.94 (possible directional bias). No evidence of actual EXIT_ONLY or ACTIVE contribution to realized outcomes. |
| Alpha gate calibration is valid for live | Alpha gate passes at 32bps > 11bps [C-006] | No closed trades to verify estimated vs realized alpha. Gate thresholds (14/8bps) were calibrated from paper run but paper run had 0 closed trades. |
| Regime detection improves trade outcomes | GMM conf 0.85-1.00 [C-009] | No evidence linking regime classification to realized PnL. GMM trained on 721 bars which may not capture regime diversity. |
| Sentiment switching improves strategy selection | F&G=11 boosts mean_revert by +40% | No trades executed with sentiment switching active. Effect on realized returns is unknown. |
| Cost model accurately reflects real trading costs | Constitution friction components computed per tick | PA executor used different fee source than constitution, creating contradictory gates. Even after fix, two fee paths exist. |

---

# NO-GAP / MATCHED AREAS

The following areas have clear evidence of implementation matching or exceeding baseline requirements:

1. **Dead-man switch** [A-006 #6, C-005]: Kraken CancelAllOrdersAfter with dedicated API client, 60s timeout, heartbeat monitoring, emergency flatten — exceeds baseline [B-026] circuit breaker requirement.

2. **VPIN / Microstructure signals** [C-006, C-009]: Per-asset VPIN computation, whale detection, composite toxicity filter (AS+VPIN+OBI+reversal) — exceeds baseline [B-018] order-flow requirement for 4H frequency.

3. **On-chain data integration** [C-009]: CryptoCompare (BTC/ETH active addresses, tx count), Solana RPC (TPS, congestion, MEV pressure) — matches baseline [B-018] on-chain signal category.

4. **Funding rate data** [C-009]: Kraken Futures funding + Coinglass funding/OI/liquidation data fetched and used in decision path — matches baseline [B-008, B-012].

5. **DRL auto-demotion safety** [A-003]: 5 consecutive losses or 15% drawdown triggers automatic demotion to EXIT_ONLY with 3-day recovery — matches baseline [B-026] per-agent drawdown limit.

6. **Authority matrix with separation of concerns** [A-004]: DECIDE/VETO/ADVISE/PENALIZE hierarchy with clear agent roles — matches baseline [B-013] agent type separation.

7. **Startup position reconciliation** [A-006 #11, C-001]: StartupReconciler with balance/position/order reconciliation on restart — matches baseline [B-016] state recovery requirement.

8. **Maker order preference** [A-007 #27]: Post-only orders with 120s timeout, partial fill handling, maker reprice — matches baseline [B-025] maker-preferred execution.

9. **Regime detection** [A-002]: 6-regime GMM per-asset with RegimeSmoother persistence=2 — matches baseline [B-018] regime filter requirement.

10. **Existence fuse** [A-005]: 28-day rolling window, -5% PnL triggers system halt — exceeds baseline [B-026] soft/hard stop requirement for system-level protection.

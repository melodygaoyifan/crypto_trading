# HMATS 系统架构文档 - Part 2
# 数据流与决策链
# ═══════════════════════════════════════════════════════════════
# 版本: v10.0-POSTAUDIT
# 日期: 2026年2月28日
# 审计状态: 7/7 端到端数据流 INTACT
# ═══════════════════════════════════════════════════════════════

## 本部分目录

1. [完整数据流：市场数据→执行](#完整数据流市场数据执行)
2. [决策链：10步 process_4h_tick()](#决策链10步-process_4h_tick)
3. [Authority-Based Fusion架构 (5-Agent)](#authority-based-fusion架构-5-agent)
4. [8 Veto 源验证链](#8-veto-源验证链)
5. [4H Tick vs 200ms执行循环](#4h-tick-vs-200ms执行循环)
6. [7条端到端数据流 (全部验证INTACT)](#7条端到端数据流)

---

## 完整数据流：市场数据→执行

```
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 1: 外部数据源 (6个)                                                │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Kraken     │  │  Coinglass   │  │ Alternative  │                  │
│  │  WS V2 + REST│  │  OI+清算     │  │  .me (F&G)   │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                            │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐                  │
│  │ CryptoCompare│  │  Solana RPC  │  │   Jito API   │                  │
│  │  链上指标    │  │  SOL 网络    │  │   MEV 指标   │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                            │
│  ┌──────┴─────────────────┴─────────────────┴───────┐                   │
│  │  Contract & Data Health Gate (Fail-Closed)       │                   │
│  │  ┌─────────────────────────────────────────┐    │                   │
│  │  │ Required (缺失 → NO_TRADE):              │    │                   │
│  │  │ - BTC/ETH/SOL 任一资产 OHLCV 缺失       │    │                   │
│  │  │ - data_age > MAX_DATA_AGE (10s)          │    │                   │
│  │  └─────────────────────────────────────────┘    │                   │
│  │  ┌─────────────────────────────────────────┐    │                   │
│  │  │ Degraded (缺失 → 保持 None, 不猜测):     │    │                   │
│  │  │ - vpin, dvol_zscore (可选安全字段)        │    │                   │
│  │  │ - microstructure (时效 > 2s → 降级)      │    │                   │
│  │  └─────────────────────────────────────────┘    │                   │
│  └──────┬────────────────────────────────────────────┘                   │
└─────────┼────────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 2: 市场分析 (market/)                                               │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  RegimeNavigator (Per-Asset GMM)                             │       │
│  │  ┌────────────────────────────────────────────────────┐     │       │
│  │  │ Per-Asset GMM:                                      │     │       │
│  │  │   BTC k=8, ETH k=7, SOL k=7                        │     │       │
│  │  │ RegimeSmoother:                                     │     │       │
│  │  │   hysteresis_threshold=3, min_persistence=2         │     │       │
│  │  │ ADX Fallback:                                       │     │       │
│  │  │   distribution-shift guard (z>3σ + >30% 偏移)      │     │       │
│  │  │ 冷启动保护:                                         │     │       │
│  │  │   per-asset _warmup_ticks, 前2 tick = degraded     │     │       │
│  │  │ cross_asset_correlation: 0.87 (统一)                │     │       │
│  │  └────────────────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  PhaseDetector                                               │       │
│  │  4阶段: IGNITION → EXPANSION → SATURATION → EXHAUSTION     │       │
│  │  影响: Alpha Gate / Exit Alpha / Leverage / CRACK 权重      │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  LeadLagEngine                                               │       │
│  │  - Binance→Kraken 领先信号                                  │       │
│  │  - 2-tier dampening                                          │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  ★ BullTransitionDetector [v10]                              │       │
│  │  4条件: Golden Cross + SOL/BTC RS + Funding + OI            │       │
│  │  4状态: INACTIVE → POTENTIAL → ACTIVE → CONFIRMED           │       │
│  │  ACTIVE → short ×0.5, CONFIRMED → BLOCK_NAKED_SHORT        │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  输出: regime_label, phase, bull_transition_state                       │
└─────────┬───────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 3: Agent层 - 信号生成 (5 Agent)                                    │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │ QuantAgent (Authority: DECIDE)                       │               │
│  │ - Best-of-N 4策略: mean_revert / momentum /          │               │
│  │   volume_breakout / vrp                               │               │
│  │ - Winner-take-all 选择 (非 IR-Softmax)               │               │
│  │ - Alpha formula: |sig| × 200 × conf × perf          │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │ DRLAgent (Authority: DECIDE, SHADOW mode)            │               │
│  │ - TQC, obs_dim=126, ent_coef=0.1 (铁律不碰)         │               │
│  │ - FiLM Conditioning (regime 条件化)                   │               │
│  │ - OOD Detector (Mahalanobis distance)                │               │
│  │ - 晋升: DISABLED → SHADOW → EXIT_ONLY → FULL        │               │
│  │ - StatisticalPromotionGate: 30天+Sharpe>1+win>48%   │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │ SentimentAgent (Authority: ADVISE)                   │               │
│  │ - Fear & Greed Index (L1 量化)                       │               │
│  │ - Haiku LLM Agent (L3 语义)                          │               │
│  │ - sentiment_zscore = (fg_value - 50) / 50.0 × 3.0   │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │ ShortBiasAgent (Authority: PENALIZE)                 │               │
│  │ - 做多信号 → soft penalty ×0.7                       │               │
│  │ - funding > 0.24%/8h → short conviction +15%         │               │
│  │ - 不否决, 只惩罚                                     │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────┐               │
│  │ RiskAgent (Authority: VETO)                          │               │
│  │ - 一票否决权                                          │               │
│  │ - 实时风险信号生成                                    │               │
│  │ - 可单方面阻止任何决策                                │               │
│  └──────────────────────────────────────────────────────┘               │
│                                                                          │
│  输出: 5 × AgentSignal(direction, confidence, authority)                │
└─────────┬───────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 4: 信号融合 (signals/authority_fusion.py)                          │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  5-Agent Authority Fusion                                    │       │
│  │                                                               │       │
│  │  融合流程:                                                    │       │
│  │  1. VETO检查: RiskAgent 否决 → 立即中止                      │       │
│  │  2. DECIDE agents (Quant + DRL) → 方向/确信                  │       │
│  │  3. PENALIZE: ShortBias → 做多 ×0.7                          │       │
│  │  4. ADVISE: Sentiment → 确信度微调                           │       │
│  │  5. Reliability Injection:                                    │       │
│  │     ConfidenceScorer 实时置信度                               │       │
│  │     confidence < 0.35 → conviction × 0.3                     │       │
│  │  6. Deadlock: ALL_CONFLICT → NO_TRADE (不挂起)               │       │
│  │                                                               │       │
│  │  Multiplier Floor: 0.15 (VC-5 修复, 防叠乘致4.6%)           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  输出: FusionResult(direction, conviction, mode)                        │
└─────────┬───────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 5: 风险验证 — 8 Veto 源 (One-Veto-Kill)                           │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  8 个 Veto 源, 任一否决 → 交易取消:                                     │
│                                                                          │
│  1. Constitution          → 参数违规检查                                │
│  2. RiskManager           → 中央风控协调                                │
│  3. DeadManSwitch         → 心跳超时 → 撤单 (refresh in try/except)    │
│  4. SqueezeProtection     → score ≥ 0.50/0.70/0.80                     │
│  5. LeverageGuard         → > 3.0× → 拒绝/削减                        │
│  6. DrawdownControl       → 4级: 10%→减仓, 15%→大减, 25%→暂停, 35%→kill│
│  7. CorrelationCrisis     → 5状态: SPIKING→减仓, CRISIS→停止新仓       │
│  8. ExistenceFuse         → weekly-8%/monthly-10%/consecutive-5         │
│                                                                          │
│  ★ BullTransition Override [v10]:                                       │
│     ACTIVE → short ×0.5                                                 │
│     CONFIRMED → BLOCK_NAKED_SHORT (只允许对冲)                          │
│                                                                          │
│  如果所有8个通过 → 继续                                                  │
│  如果任一否决 → 中止并记录原因                                           │
└─────────┬───────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 6: Profit-Max 优化                                                 │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  FalseBreakoutDetector — 唯一硬否决                          │       │
│  │  SignalQualityScorer — conviction 乘数                       │       │
│  │                                                               │       │
│  │  Alpha Gate (Fee-Aware, Volume-Aware):                       │       │
│  │  alpha > friction × 1.5 → 正常执行                           │       │
│  │  friction < alpha < friction × 1.5 → 降速拆单               │       │
│  │  alpha < friction → VETO: FRICTION_EXCEEDS_EDGE             │       │
│  │  Free Tier: NORMAL 14bps, OPPORTUNITY 8bps                  │       │
│  │                                                               │       │
│  │  CRACK 阈值 (集中定义):                                      │       │
│  │  FULL_EXIT = 0.50, PARTIAL = 0.45, URGENCY = 0.35          │       │
│  │                                                               │       │
│  │  ★ HPLV Filter [v10]:                                       │       │
│  │  price ≥ 90th + volume < 60% avg → short ×0.5              │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────┬───────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 7: Portfolio Brain 协调                                            │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  StrategicCoordinator                                        │       │
│  │  - CorrelationController → 5状态 → per-asset exposure 约束  │       │
│  │  - recommended_exposure = base × scale_factor × adjustment  │       │
│  │  - portfolio_allocation (归一化权重)                          │       │
│  │                                                               │       │
│  │  ★ AssetAlphaTilt [v10]:                                    │       │
│  │  - Rolling 30D Sortino per asset                             │       │
│  │  - Sortino ≥ 2.0 → ×1.5, ≥ 1.0 → ×1.3, ≥ 0 → ×1.0      │       │
│  │  - Sortino ≥ -0.5 → ×0.7, < -0.5 → ×0.5                  │       │
│  │  - 更新频率: 每24h (每6个4H tick)                            │       │
│  │  - 平滑: α=0.3, 冷启动 (<5笔) → mult=1.0                  │       │
│  │                                                               │       │
│  │  Cross-Asset Rules:                                          │       │
│  │  BTC 25% max, ETH 25% max, SOL 20% max                     │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────┬───────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 8: 执行层 (execution/)                                             │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  PA Executor (passive_aggressive.py)                         │       │
│  │  - PASSIVE: post_only limit order, 等待成交                  │       │
│  │  - AGGRESSIVE: 120s timeout → cancel & market               │       │
│  │  - ABORT: 取消, 不执行                                       │       │
│  │  - oflags='post', postOnly=True                              │       │
│  │  - TimingEngine → timing_mode/score → PA behavior (已接线)  │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  Market Impact (ImpactCalibration → ProductionMarketImpact) │       │
│  │  - CalibrationBridge: bucket-level params → Almgren-Chriss  │       │
│  │  - 有置信度 (conf>0.3) → bucket 参数                        │       │
│  │  - 无置信度 → fallback per-symbol 自标定                     │       │
│  │  - record_execution() 双写                                   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  Dynamic Slicer (ATR-based) + Fill-Slope Monitor            │       │
│  │  - adverse selection 检测                                    │       │
│  │  - 成交率监控                                                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  ★ Fill Rate Logging [v10]                                  │       │
│  │  - fill_ratio, time_to_fill_s, slippage_bps                 │       │
│  │  - was_repriced, order_type, final_action                   │       │
│  │  - 输出: logs/fill_quality.jsonl (周报手动 review)          │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  输出: OrderResult(success, fills, slippage, fees)                      │
└─────────┬────────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 9: 反馈记录                                                        │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  PnL / ShadowLedger / FailureMemory                         │       │
│  │  - ShadowLedger: tick_id + fill_quality 字段                │       │
│  │  - FailureMemory: per-asset × per-regime 失败记忆           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  ConfidenceScorer (已接通)                                   │       │
│  │  - record_signal(strategy, direction, confidence, regime)   │       │
│  │  - record_outcome(strategy, timestamp, pnl, correct)        │       │
│  │  - 三维度: 方向准确率35% + PnL35% + Regime30%               │       │
│  │  - 输出 → Reliability Injection (fusion 降权)               │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  DriftDetector (5源)                                        │       │
│  │  - Feature distribution (z-score, z>3.0)                    │       │
│  │  - Latent space (distance)                                   │       │
│  │  - GMM Regime (JSD > 0.15)                                  │       │
│  │  - 执行滑点 (z > 2.0)                                       │       │
│  │  - DRL Action (mean_z > 2.5 / std > 2.0)                   │       │
│  │  → Severity: NONE/MINOR/MODERATE/SEVERE/CRITICAL            │       │
│  │  → SEVERE+ → DRL EXIT_ONLY 降级                             │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  ExistenceFuse.on_trade_close()                             │       │
│  │  - 更新 consecutive_loss 计数                               │       │
│  │  - 检查 weekly/monthly 损失                                 │       │
│  │  - 返回 (NONE/OBSERVE/HALT/KILL, reason)                   │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────┬────────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 10: 每日任务 (每6个4H tick = 24h)                                  │
│  ────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  ★ AssetAlphaTilt.update() → 重算 per-asset Sortino multiplier         │
│  WeekendManager 检查 → 周末限制                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 决策链：10步 process_4h_tick()

main.py 中的规范主循环, 唯一决策入口:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  process_4h_tick() — 10步决策流程 (v10, 已审计验证)                       │
└─────────────────────────────────────────────────────────────────────────┘

[1] 获取最新市场数据 → Contract & Data Health Gate
    ├─ 6 数据源拉取
    ├─ OHLCV 时效检查 (MAX_AGE=10s)
    ├─ 必需资产检查 (BTC/ETH/SOL 缺一 → NO_TRADE)
    └─ 可选字段降级 (vpin/dvol → None, 不猜测)

[2] 分析市场状态
    ├─ Per-Asset GMM → regime_label (BTC k=8, ETH k=7, SOL k=7)
    ├─ PhaseDetector → IGNITION/EXPANSION/SATURATION/EXHAUSTION
    ├─ RegimeSmoother → hysteresis (threshold=3, persistence=2)
    ├─ ADX Fallback → distribution-shift guard
    └─ ★ BullTransitionDetector [v10]
        ├─ 4条件评估 (Golden Cross/RS/Funding/OI)
        └─ 状态更新: INACTIVE/POTENTIAL/ACTIVE/CONFIRMED

[3] 各智能体生成信号 (5 agent)
    ├─ QuantAgent (DECIDE) → Best-of-N 4策略
    ├─ DRLAgent (DECIDE, SHADOW) → TQC obs_dim=126
    ├─ SentimentAgent (ADVISE) → F&G L1 + Haiku L3
    ├─ ShortBiasAgent (PENALIZE) → 做多 ×0.7, funding weighted
    └─ RiskAgent (VETO) → 风控信号

[4] Authority Fusion 信号融合
    ├─ 5-agent 权限矩阵
    ├─ VETO → 立即中止
    ├─ DECIDE → 方向/确信
    ├─ PENALIZE → soft penalty (不否决)
    ├─ ADVISE → 确信微调
    ├─ Reliability Injection → confidence<0.35 → conviction×0.3
    ├─ Deadlock → NO_TRADE (不挂起)
    └─ Multiplier floor = 0.15 (防叠乘 → 4.6%)

[5] 风险检查 — 8 Veto 源 (One-Veto-Kill)
    ├─ Constitution / RiskManager / DMS / Squeeze
    ├─ LeverageGuard(3.0×) / DrawdownControl(4级) / CorrelationCrisis
    ├─ ExistenceFuse (weekly-8%/monthly-10%/consecutive-5)
    └─ ★ BullTransition Override [v10]:
        ├─ ACTIVE → short ×0.5
        └─ CONFIRMED → BLOCK_NAKED_SHORT

[6] Profit-Max 优化
    ├─ FalseBreakoutDetector (唯一硬否决)
    ├─ SignalQualityScorer → conviction 乘数
    ├─ Alpha Gate → friction vs edge (NORMAL 14bps, OPP 8bps)
    ├─ CRACK → 0.50/0.45/0.35
    └─ ★ HPLV Filter [v10] → price≥90th + vol<60% → short×0.5

[7] Portfolio Brain 协调
    ├─ CorrelationController → per-asset exposure 约束
    ├─ ★ AssetAlphaTilt [v10] → Sortino-weighted ×0.5~1.5
    └─ portfolio_allocation (归一化)

[8] 执行
    ├─ PA Executor (passive→aggressive, post_only)
    ├─ TimingEngine → timing_mode/score → PA behavior
    ├─ Dynamic Slicer (ATR-based)
    ├─ ImpactCalibration → bucket-level 校准
    └─ ★ Fill Rate Logging [v10] → jsonl

[9] 反馈记录
    ├─ PnL / ShadowLedger / FailureMemory
    ├─ ConfidenceScorer.record_signal() + record_outcome()
    ├─ DriftDetector 更新 (5源)
    └─ ExistenceFuse.on_trade_close() → consecutive_loss

[10] 每日任务 (每6 tick = 24h)
    ├─ ★ AssetAlphaTilt.update() → Sortino multiplier
    └─ WeekendManager 检查
```

---

## Authority-Based Fusion架构 (5-Agent)

### 为什么用 Authority-Based 而非加权融合

```python
# 传统加权融合的问题:
signal = w1*agent1 + w2*agent2 + w3*agent3
# 50% long + 50% short = 0% (瘫痪)
# 稀释确信度, 没有明确决策权

# HMATS Authority-Based (v3.4+):
if risk_agent.veto:
    return NO_TRADE  # 一票否决

direction = quant_agent.direction  # DECIDE authority
conviction = quant_agent.confidence

if short_bias.penalizes(direction):
    conviction *= 0.7  # PENALIZE, 不否决

if sentiment.advises(direction):
    conviction += adjustment  # ADVISE

if confidence_scorer.low(quant):
    conviction *= 0.3  # Reliability Injection

# 明确的权限, 不稀释
return Trade(direction, conviction)
```

### 5-Agent 权限矩阵

| Agent | Authority | 可否决? | 可决定方向? | 功能 |
|-------|-----------|--------|-----------|------|
| **QuantAgent** | DECIDE | 否 | **是** | Best-of-N 4策略, 方向性确信 |
| **DRLAgent** | DECIDE | 否 | **是** (SHADOW时仅记录) | TQC, 渐进晋升 |
| **SentimentAgent** | ADVISE | 否 | 否 | F&G + Haiku, 确信微调 |
| **ShortBiasAgent** | PENALIZE | 否 | 否 | 做多 ×0.7, 不否决 |
| **RiskAgent** | VETO | **是** | 否 | 一票否决 |

### Authority 流程示例

```
场景: BTC 做空信号

1. VETO 检查:
   RiskAgent: veto = False ✓ (drawdown OK, leverage OK)

2. DECIDE Authority:
   QuantAgent: direction=SHORT, confidence=0.72
   DRLAgent (SHADOW): 仅记录, 不参与决策 ✓

3. PENALIZE 检查:
   ShortBiasAgent: direction=SHORT → 不惩罚 ✓
   (如果是 LONG → conviction ×0.7)

4. ADVISE 整合:
   SentimentAgent: F&G=28 (恐惧) → +0.05 确信
   → 调整确信 = 0.72 + 0.05 = 0.77

5. Reliability Injection:
   ConfidenceScorer.get(quant, QUIET_ACCUM) = 0.85
   → confidence > 0.35, 不降权 ✓

6. 最终:
   direction = SHORT, conviction = 0.77
   → 继续到风险检查...
```

### Deadlock Resolution

```
ALL_CONFLICT 场景: Quant=SHORT, DRL=LONG
  旧行为: 可能挂起
  v10行为: ALL_CONFLICT_FLAT → NO_TRADE (不挂起)
  CONTINUE→NONE 映射已修复
```

---

## 8 Veto 源验证链

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HMATS v10.0 风险验证链 — 8 Veto 源                                     │
│  规则: One-Veto-Kill (任一否决 → 交易取消)                                │
│  审计: 实际 31 gates (18 hard + 23 soft, floor=0.15)                    │
└─────────────────────────────────────────────────────────────────────────┘

Veto 1: Constitution
├─ 参数违规检查
├─ Schema 验证
└─ 失败 → 阻止交易

Veto 2: RiskManager
├─ 中央风控协调
├─ 聚合所有风险信号
└─ 失败 → 阻止交易

Veto 3: DeadManSwitch
├─ 心跳监控 (refresh in try/except)
├─ 超时 → 撤单
└─ 生产中无法禁用

Veto 4: SqueezeProtection (3级)
├─ score ≥ 0.50 → WARN (记录, 继续)
├─ score ≥ 0.70 → REDUCE (减仓)
└─ score ≥ 0.80 → FLATTEN (全部平仓)

Veto 5: LeverageGuard
├─ 硬性限制: 3.0×
├─ > 3.0× → 拒绝新仓位
└─ 超限 → 强制削减到 3.0×

Veto 6: DrawdownControl (4级梯度)
├─ 10% → 减仓 (×0.85)
├─ 15% → 大幅减仓 (×0.65)
├─ 25% → 暂停交易
└─ 35% → KILL (系统停机)

Veto 7: CorrelationCrisis (5状态)
├─ STABLE → 正常
├─ ELEVATED → 观察
├─ SPIKING → 减仓
├─ CRISIS → 停止新开仓
└─ COLLAPSING → 观察

Veto 8: ExistenceFuse (v10 多层)
├─ 5 笔连续亏损 → 暂停 24h
├─ 周损失 ≥ 8% → HALT (暂停)
├─ 月损失 ≥ 8% → OBSERVE (半仓)
├─ 月损失 ≥ 10% → KILL
└─ DRL PromotionGate: 5-loss → EXIT_ONLY (独立计数)

★ BullTransition Override [v10] (在 veto 链之后):
├─ ACTIVE → short ×0.5
└─ CONFIRMED → BLOCK_NAKED_SHORT

★ HPLV Filter [v10] (在 ProfitMax 中):
└─ price ≥ 90th + volume < 60% → short ×0.5

Soft Multiplier 叠乘保护 (VC-5):
├─ 23 个 soft multipliers 可叠加
├─ 最差理论值: 4.6%
├─ Floor: 0.15 (15%, 不低于此值)
└─ 3 个重复检查已去重 (VC-3)
```

---

## 4H Tick vs 200ms执行循环

```
┌─────────────────────────────────────────────────────────────────────┐
│  4H TICK (14,400秒) — 战略决策 (WHAT to trade)                       │
│  ─────────────────────────────────────────────────────────────────── │
│  - 完整 10步决策周期                                                  │
│  - 6 数据源拉取 + 验证                                                │
│  - Per-Asset GMM regime 检测                                          │
│  - 5 agent 信号生成                                                    │
│  - Authority Fusion + Reliability Injection                           │
│  - 8 Veto 源风险检查                                                   │
│  - Portfolio Brain 协调                                                │
│  - 创建新交易意图 → 执行                                               │
│  - 反馈: ConfidenceScorer + DriftDetector + Fuse                     │
│  - ★ BullTransition 评估 [v10]                                       │
│  - 每6 tick: AssetAlphaTilt 更新                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  200MS执行循环 (reserved, 文档化) — 战术执行 (HOW to execute)         │
│  ─────────────────────────────────────────────────────────────────── │
│  - PA Executor reprice interval                                       │
│  - 订单簿深度监控                                                     │
│  - Fill-Slope Monitor (adverse selection)                             │
│  - 无新决策、无融合、无 regime 检测                                    │
│  - 不可改变方向或目标敞口                                              │
└─────────────────────────────────────────────────────────────────────┘

时机对比:
├─ 4H Tick: 每 14,400秒 一次完整决策
├─ 200ms Loop: 每 0.2秒 一次执行调整
└─ 比率: 72,000× 更频繁
```

---

## 7条端到端数据流

v10 审计验证的 7 条端到端数据流, 全部 INTACT:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DF-01: Signal → Fusion → Direction                           ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  QuantAgent.signal → AuthorityFusion.fuse() → final_direction         │
│  验证: direction 正确传播, DECIDE 权限生效                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  DF-02: Direction → Sizing → Order                            ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  final_direction → PositionSizer → TradeIntent → PA Executor → Order  │
│  验证: sizing 受所有约束 (leverage, exposure, correlation)             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  DF-03: MarketData → Regime → Decision Gates                  ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  OHLCV → GMM → regime_label → PhaseDetector → Alpha Gate / CRACK     │
│  验证: regime 正确影响 gate 阈值和行为                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  DF-04: Risk Checks → Veto Chain → Execution Guard            ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  8 Veto 源 → any_veto → abort_trade + log_reason                     │
│  验证: One-Veto-Kill 正确执行, 无遗漏                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  DF-05: BullDetector → BLOCK/REDUCE → Override                ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  BullTransitionDetector.evaluate() → state → direction override       │
│  验证: ACTIVE→×0.5, CONFIRMED→BLOCK 正确触发                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  DF-06: Trade Result → Fuse Check → HALT/KILL                 ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  trade.close() → ExistenceFuse.on_trade_close() → (action, reason)   │
│  验证: consecutive_loss / weekly / monthly 正确计数和触发              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  DF-07: TimingEngine → timing_mode → PA Executor              ✅ INTACT│
│  ──────────────────────────────────────────────────────────────────────│
│  TimingEngine.get_score() → timing_mode/score → PA executor behavior  │
│  验证: timing_mode 正确影响 passive/aggressive 选择                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**文档第2部分结束**

继续阅读：
- Part 3: 风险管理与状态机
- Part 4: 执行层与DRL
- Part 5: 运维与附录

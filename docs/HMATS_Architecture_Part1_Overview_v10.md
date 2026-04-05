# HMATS 系统架构文档 - Part 1
# 概览与系统总览
# ═══════════════════════════════════════════════════════════════
# 版本: v10.1-POSTAUDIT (v6.8.0 sync)
# 日期: 2026年3月27日 (updated from Feb 28)
# 前版本: v10.0-POSTAUDIT / v6.5.1 (2026年2月21日)
# 状态: 审计通过 + v6.8.0 18个bug修复, pre-live hardening完成
# 审计链: 1909行审计 → 84项自检 → 11修复 → 91项再审计 → 7/7数据流 → v6.8.0 18修复
# ═══════════════════════════════════════════════════════════════

## 目录

**Part 1 - 概览与系统总览**（本文档）
- 执行摘要
- 系统概览
- 核心模块清单
- 关键统计数据

**Part 2 - 数据流与决策链**
- 完整数据流图
- 10步决策流程
- Authority-Based Fusion (5-agent)
- 风险管理决策链

**Part 3 - 风险管理与状态机**
- 8 Veto 源 + One-Veto-Kill
- 4级 Drawdown 梯度
- BullTransitionDetector 状态机
- Existence Fuse 多层保护

**Part 4 - 执行层与DRL**
- PA Executor + TimingEngine
- DRL训练流程（20 Stages, 35 Iron Laws）
- Fill Rate Logging

**Part 5 - 运维与附录**
- 运维手册
- 配置参考 (sota_flags.py 集中)
- 术语表
- 代码组织图

---

## 执行摘要

**HMATS (Hierarchical Multi-Agent Trading System)** 是一个生产级算法交易系统，专为加密货币市场（BTC、ETH、SOL）在Kraken交易所设计。系统结合了量化策略、深度强化学习、情绪分析和链上数据，通过多层风险管理生成交易信号。**主做空, 高风险偏好**。

### 关键统计数据

```
代码库规模:
├─ 核心入口: main.py (~13,000 行)
├─ 总代码库: 300+ Python 文件, 200K+ 行
├─ Agent数量: 5 个权限化 Agent (Quant/DRL/Sentiment/ShortBias/Risk)
├─ 风险层级: 8 个 Veto 源 (One-Veto-Kill)
├─ 数据源: 6 个 (Kraken/Coinglass/F&G/CryptoCompare/Solana RPC/Jito)
├─ 审计覆盖: 91 checks, 97.7% GREEN, 7/7 data flows INTACT
└─ SOTA 覆盖: ~96% (语境过滤后)

交易参数:
├─ 交易频率: 4H主Tick (14,400s), 200ms执行子循环 (reserved)
├─ 风险偏好: 高风险, 主做空
├─ 资产: BTC (25% cap), ETH (25% cap), SOL (20% cap)
├─ 交易所: Kraken ONLY (单交易所模式, 强制执行)
├─ 账户: $10K → $100K (盈利后扩展)
├─ 会员: Kraken Pro ($10K/月免费交易额度)
└─ 部署方式: Systemd服务, Ubuntu/Debian

核心参数 (sota_flags.py 集中管理):
├─ 最大杠杆: 3.0× (硬性限制)
├─ Drawdown: 4级梯度 (10%→减仓, 15%→大减, 25%→暂停, 35%→kill)
├─ Existence Fuse: weekly-8% / monthly-10% / consecutive-5
├─ cross_asset_correlation: 0.87 (统一)
├─ CRACK 阈值: 0.50 / 0.45 / 0.35 (集中定义)
├─ DRL obs_dim: 126 (铁律, 不碰)
├─ DRL ent_coef: 0.1 (铁律, 不碰)
├─ Alpha Gate: NORMAL 14bps, OPPORTUNITY 8bps
└─ Maker/Taker: 16/26 bps (Kraken Pro)
```

### 核心理念

> **「Aggressive Alpha, Defensive Shell」**

> **「错失高置信度机会，比承受可控损失更糟糕」** (v3.3-HR)

> **「Don't kill trades, modulate them.」** (v6.4.1)

> **「Wire it or it doesn't exist.」** (v9.0)

> **「Code exists ≠ code works — verify end-to-end.」** (v10.0)

---

## 系统概览

### 设计原则

```
1. 纵深防御 (Defense in Depth)
   - 8 个 Veto 源, 任何一个都可否决 (One-Veto-Kill)
   - 失败默认拒绝, 安全优先于盈利

2. 基于权威的融合 (Authority-Based Fusion)
   - 5-agent 权限矩阵 (DECIDE/VETO/ADVISE/PENALIZE)
   - 非加权稀释, 明确的权限层级

3. 单一事实来源 (Single Source of Truth)
   - process_4h_tick() 是唯一决策入口
   - sota_flags.py 集中管理关键参数
   - 禁止孤立并行流水线

4. 失败闭合 (Fail-Closed)
   - 异常默认拒绝, 安全默认值
   - 数据缺失 → NO_TRADE (不猜测)
   - MAX_DATA_AGE = 10s

5. 完整审计追踪 (Complete Audit Trail)
   - ShadowLedger (含 tick_id + fill_quality)
   - 100% 可追踪的因果链

6. 端到端验证 (End-to-End Verification) [v10 新增]
   - 91 项功能检查 + 7 条数据流验证
   - 代码存在 ≠ 代码工作, 必须验证集成

7. 热重启能力 (Hot-Restart Capable)
   - 状态持久化 + Startup Reconciler (重启对账)
   - Cancel-on-Disconnect (断线撤单)
```

### 模式层次结构（已锁定）

```
优先级 0: NO_TRADE        ← 最高优先级, 硬性FLAT
         ↑
优先级 1: OPPORTUNITY     ← 激进窗口, 降低阈值 (Alpha Gate 8bps)
         ↑
优先级 2: NORMAL         ← 默认状态 (Alpha Gate 14bps)

规则：NO_TRADE > OPPORTUNITY > NORMAL
任何时候只能处于一种模式
```

### 资产配置策略

```
BTC:  25% max exposure  (流动性最高, 稳定锚)
      └─ Golden Cross 用于 BullTransition 检测

ETH:  25% max exposure  (中等流动性, 生态系统敞口)
      └─ 相关性多样化

SOL:  20% max exposure  (高 beta, 做空收益放大器)
      ├─ SOL Dominance Mode (高确信度时)
      ├─ 6 数据源中 2 个 SOL 特有 (Solana RPC, Jito)
      └─ 预期 1-2 年后进入牛市 → 准备翻转机制

全局:
├─ AssetAlphaTilt: per-asset Sortino-weighted 倾斜 (0.5~1.5×)
├─ CorrelationController: 5 状态动态缩放
└─ BullTransition: CONFIRMED → 禁止裸空
```

### 硬件要求

```
开发/训练环境:
├─ GPU: RTX 5090 - DRL训练 (7-25h per run)
├─ CPU: 8核+ (SubprocVecEnv 4 parallel)
├─ RAM: 32GB+
└─ 网络: 低延迟到 Kraken

生产环境:
├─ 服务器: Linux (Ubuntu 24)
├─ RAM: 16GB+
├─ CPU: 8核+
├─ GPU: 不需要 (推理 <10ms)
├─ 网络: 稳定低延迟
└─ 存储: 100GB SSD
```

---

## 完整模块清单

### 核心模块 (按功能分组)

#### 1. 主入口 — main.py (~13,000 行)

```
⭐⭐⭐⭐⭐ main.py
├─ process_4h_tick() — 唯一决策入口
├─ 10步决策流程 (Data→Analysis→Agents→Fusion→Risk→ProfitMax→Portfolio→Execute→Feedback→Daily)
├─ 所有模块初始化和接线
├─ BullTransitionDetector 评估 (L3645-3660)
├─ HPLV Filter (L9521-9544)
├─ AssetAlphaTilt 每日更新
└─ Fill Rate Logging
```

#### 2. core/ — 运行时核心

```
sota_flags.py        — 集中参数 (drawdown/leverage/alpha_gate)
config_resolver.py   — 启动审计, 冲突检测, 参数完整性
regime_smoother.py   — hysteresis + min_persistence
exchange_guard.py    — Kraken-only 保护
★ asset_alpha_tilt.py — Sortino-weighted 动态倾斜 [v10]
```

#### 3. risk/ — 风险管理 (8 Veto 源)

```
risk_manager.py                     — 中央风控协调器
★ bull_transition_detector.py       — 4条件牛市检测 [v10]
★ strategy_existence_fuse.py        — 多层熔断 (weekly/monthly/consecutive) [v10增强]
thesis_budget_governor.py           — Thesis 预算 (is_win=pnl≥0)
leverage_guard.py                   — 3.0× 硬性限制
correlation_realtime_controller.py  — 5状态相关性 (1,098行)
drawdown_controller.py              — 4级梯度 (10/15/25/35%)
squeeze_protection.py               — 3级 (warn/reduce/flatten)
```

#### 4. defense/ — 防御与宪法

```
constitution.py                  — 参数验证 + veto
p0_safety_integrator.py          — 6层安全链
shadow_ledger.py                 — 不可变审计追踪 (含 fill_quality)
sol_defense.py                   — SOL 特有防护
kraken_integrity_shield.py       — Kraken 完整性护盾
dead_man_switch.py               — 心跳超时 → 撤单
```

#### 5. agents/ — 5 Agent 权限矩阵

```
kraken_quant_agent.py   — Best-of-N 4策略 (mean_revert/momentum/vol_breakout/vrp)
                           Authority: DECIDE
drl_agent.py            — TQC, obs_dim=126, ent_coef=0.1
                           Authority: DECIDE (SHADOW/EXIT_ONLY/FULL)
                           FiLM Conditioning, OOD Detector
sentiment_agent.py      — F&G L1 + Haiku L3
                           Authority: ADVISE
short_bias_agent.py     — 做多 ×0.7 penalty, funding>0.24%→short+15%
                           Authority: PENALIZE
risk_agent.py           — 风控否决
                           Authority: VETO (一票否决)
```

#### 6. signals/ — 信号融合

```
authority_fusion.py              — 5-agent 融合 + Reliability Injection
opportunity_triggers.py          — 5组 trigger + CRACK 集中阈值
★ high_position_low_volume_filter.py — 卖盘衰竭检测 [v10]
signal_quality_scorer.py         — 信号质量评分
profit_max_adapter.py            — FalseBreakout + Alpha Gate
partial_consensus.py             — SHADOW 模式部分共识
fusion_patience.py               — 耐心等待更好信号
```

#### 7. execution/ — 执行层

```
passive_aggressive.py            — PA Executor (passive→aggressive, post_only)
timing_engine.py                 — timing_mode/score → PA behavior (v10 已接线)
production_market_impact.py      — Almgren-Chriss + CalibrationBridge
impact_calibration.py            — bucket-level 精细标定
dynamic_slicer.py                — ATR-based 拆单
fill_slope_monitor.py            — adverse selection 检测
★ fill_quality_logger.py         — 成交质量 jsonl 记录 [v10]
```

#### 8. analytics/ — 分析与反馈

```
confidence_scorer.py             — per-strategy × per-regime 置信度 (已接通)
drift_detector.py                — 5源漂移检测 (feature/latent/GMM/slippage/DRL)
★ monte_carlo_validator.py       — 策略鲁棒性验证 (1000 shuffle) [v10]
```

#### 9. market/ — 市场分析

```
regime_navigator.py              — Per-Asset GMM (BTC k=8, ETH k=7, SOL k=7)
phase_detector.py                — 4阶段 (IGNITION/EXPANSION/SATURATION/EXHAUSTION)
lead_lag_engine.py               — Binance→Kraken 2-tier dampening
```

#### 10. orchestration/ — 编排

```
strategic_coordinator.py         — Portfolio Brain
                                   CorrelationController → per-asset exposure
                                   ★ AssetAlphaTilt integration [v10]
```

#### 11. training/ — DRL 训练 (不碰)

```
20-stage pipeline, 35 iron laws
RTX 5090, SubprocVecEnv (4 parallel), 7-25h per run
obs_dim=126, ent_coef=0.1 — 铁律
Per-asset GMM: BTC k=8, ETH k=7, SOL k=7
4-fold cross-validation
StatisticalPromotionGate: 30天 shadow + Sharpe>1.0 + win>48%
```

---

## 系统特征总结

### 已锁定的设计决策

```
✅ 单一交易所模式 — Kraken ONLY
✅ 4H主决策频率 — 14,400秒
✅ 200ms执行子循环 — reserved for PA executor
✅ 模式层次 — NO_TRADE > OPPORTUNITY > NORMAL
✅ DRL入场权限 — SHADOW→EXIT_ONLY→FULL 渐进晋升
✅ One-Veto-Kill — 8 veto 源, 任一否决
✅ 主做空 + BullTransition 保护 — CONFIRMED 时禁止裸空
✅ Existence Fuse 多层 — weekly/monthly/consecutive
✅ 杠杆 3.0× 硬限 + Drawdown 4级梯度
✅ cross_asset_correlation 0.87 统一
✅ CRACK 阈值 0.50/0.45/0.35 集中定义
✅ DRL obs_dim=126, ent_coef=0.1 铁律不碰
✅ Authority-based 融合 — 5-agent, 非加权稀释
✅ ShadowLedger — 完整审计追踪
```

### 关键性能指标

| 指标 | 目标 | 状态 |
|------|------|------|
| Sharpe比率 | >1.0 | DRL晋升标准 |
| 胜率 | >48% | DRL晋升标准 |
| Drawdown | 4级梯度 | ✅ 10%/15%/25%/35% |
| 杠杆 | <3.0× | ✅ 硬性限制 |
| Maker成交率 | >70% | Fill Rate Logging 追踪 |
| 系统集成率 | >95% | ✅ 97.7% (91 checks) |
| 数据流完整性 | 7/7 | ✅ 100% INTACT |
| SOTA覆盖度 | >90% | ✅ ~96% (过滤后) |

### 生产就绪状态

```
✅ 审计通过 (97.7% GREEN):
├─ 91 功能检查, 80 LIVE
├─ 7/7 端到端数据流 INTACT
├─ 0 REGRESSION
├─ All 11 residual fixes PASS
├─ SOTA 覆盖 ~96%
└─ 4 PARTIAL (all LOW cosmetic) + 6 Final Polish 执行中

✅ 完全就绪 (高置信度):
├─ Kraken集成 (WS V2 + CRC32 + 自愈 + Cancel-on-Disconnect)
├─ 风险管理 (8 veto, 4级 drawdown, BullTransition, Fuse 多层)
├─ Authority Fusion (5-agent matrix + Reliability Injection)
├─ 执行层 (PA Executor + TimingEngine 已接线)
├─ Constitution + P0 Safety
├─ ShadowLedger (审计追踪)
├─ Anti-Churn 已修复 (AC-0~5)
├─ Veto Chain 已修复 (VC-0~9, floor 0.15)
└─ 6 数据源全部 LIVE

🔄 进行中:
├─ Final Polish (6 items: 4 PARTIAL + 2 SOTA-lite)
├─ 24h Paper Run Validation
├─ DRL TQC Training (Stage 8+)
└─ Exit Alpha Audit (prompt #10)

📋 计划中:
├─ Live Deployment (paper → live)
├─ DRL SHADOW→EXIT_ONLY 晋升 (30天)
├─ Account Scaling $10K → $100K
└─ Stage 21 Meta-Learner
```

### HMATS 超越 SOTA 的能力 (做空场景)

```
⭐ CRACK System          — 结构突破信号, SOTA 无对应
⭐ Phase-Aware Exit      — 4阶段精确退出
⭐ BullTransitionDetector — 做空系统最关键保护
⭐ SOL Dominance Mode    — 高 beta 做空放大器
⭐ Existence Fuse 多层    — 做空亏损无上限, 这个保命
⭐ One-Veto-Kill         — 挤空时硬否决救命
⭐ Squeeze Protection 3T  — 专门针对做空最大风险
⭐ Lead-Lag Alpha        — Binance 领先信号
⭐ FiLM Conditioning     — DRL regime 调制
⭐ 15 条做空纪律         — SOTA 没有的做空规则集
⭐ Startup Reconciler    — 重启对账 (消除 churn)
⭐ Cancel-on-Disconnect  — 断线撤单
```

---

**文档第1部分结束**

继续阅读：
- Part 2: 数据流与决策链
- Part 3: 风险管理与状态机
- Part 4: 执行层与DRL
- Part 5: 运维与附录

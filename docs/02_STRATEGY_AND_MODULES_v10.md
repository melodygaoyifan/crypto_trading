# HMATS v10.0 策略与模块详细说明
## 系统架构、策略逻辑与因果关系完整解析

**版本**: v10.1-POSTAUDIT (v6.8.0 sync)
**最后更新**: 2026年3月27日 (updated from Feb 28)
**前版本**: v10.0-POSTAUDIT (2026年2月28日)
**审计状态**: 97.7% GREEN — 91 checks, 80 LIVE, 7/7 data flows INTACT, 0 REGRESSION
**v6.8.0 变更**: DRL=DECIDE(ACTIVE), ShortBias=ADVISE(was PENALIZE), mean_revert RSI确认, 18个bug修复

---

# 目录

- [第一部分：系统总体架构](#第一部分系统总体架构)
- [第二部分：数据获取层](#第二部分数据获取层)
- [第三部分：市场分析层](#第三部分市场分析层)
- [第四部分：智能体层（完整列表）](#第四部分智能体层完整列表)
- [第五部分：信号融合层](#第五部分信号融合层)
- [第六部分：风险控制层](#第六部分风险控制层)
- [第七部分：Profit-Max优化层](#第七部分profit-max优化层)
- [第八部分：Portfolio Brain层](#第八部分portfolio-brain层)
- [第九部分：执行层](#第九部分执行层)
- [第十部分：自适应反馈层](#第十部分自适应反馈层)
- [第十一部分：策略哲学与盈利逻辑](#第十一部分策略哲学与盈利逻辑)
- [第十二部分：技术栈总览](#第十二部分技术栈总览)
- [附录A：模块依赖关系图](#附录a模块依赖关系图)
- [附录B：审计验证记录](#附录b审计验证记录)
- [附录C：SOTA覆盖分析](#附录csota覆盖分析)

---

# 第一部分：系统总体架构

## 1.1 核心设计理念

**核心哲学：「Aggressive Alpha, Defensive Shell」**。系统在信号层追求激进的Alpha捕获（高确信度时果断入场、允许杠杆放大），同时在风控层构建严格的防御性外壳（多层Veto、Thesis预算、存在性熔断），确保单次失败的损失始终可控。

四条核心理念:

> **理念1：「错过一次高确信度的机会，比承担一次可控的亏损更糟糕。」**

> **理念2：「不要杀死交易，调节它们。」（Don't kill trades, modulate them）**

> **理念3：「写好的代码必须接通线路。」（Wire it or it doesn't exist）**

> **理念4（v10新增）：「代码存在不等于代码工作 — 必须端到端验证。」**

v10 的核心发现: v9 完成了模块接线, 但审计揭示了 "code exists ≠ code works" 问题 — 98.6% 的模块有代码, 但只有 55% 正确集成到主执行循环。v10 通过系统性审计-修复-再审计, 将实际集成率提升到 97.7%。

**双时间尺度控制回路**：4H决策主循环 (`process_4h_tick()`) + 200ms执行调整子循环。4H主循环是唯一的交易决策入口；200ms执行子循环仅负责订单微调，不可改变方向或目标敞口。

## 1.2 核心能力

| 能力 | 描述 | 技术支撑 | v10 状态 |
|------|------|---------|---------|
| 市场状态识别 | 识别趋势、震荡、极端等状态 | Per-asset GMM (BTC k=8, ETH k=7, SOL k=7) | ✅ 运行中 |
| 多维度信号生成 | 技术、情绪、链上多维分析 | 多智能体协作 (5 agent, 14 信号源) | ✅ 运行中 |
| 智能信号融合 | 基于权限的决策机制 | Authority Fusion (DECIDE/VETO/ADVISE/PENALIZE) | ✅ 运行中 |
| 策略自适应 | 低置信度策略自动降权 | Reliability Injection + ConfidenceScorer | ✅ 运行中 |
| 多层风险控制 | 杠杆、预算、熔断、相关性、牛市保护 | Defense-in-Depth (8 veto源, 4级Drawdown) | ✅ v10 增强 |
| 牛市转换检测 | 牛市来临时禁止裸空 | BullTransitionDetector (4条件状态机) | 🆕 v10 新增 |
| 多资产协调 | 相关性约束 + 动态分配 | Portfolio Brain + AssetAlphaTilt | ✅ v10 增强 |
| 智能执行 | 精细冲击校准 + 最优拆单 | PA Executor + TimingEngine (已接线) | ✅ v10 修复 |
| 漂移检测 | 模型/数据/执行偏移自动识别 | DriftDetector (5源) | ✅ 运行中 |
| 存在性保护 | 系统级自我退出能力 | Existence Fuse (weekly/monthly/consecutive) | ✅ v10 增强 |
| 执行质量追踪 | 成交率/滑点持续记录 | Fill Rate Logging (jsonl) | 🆕 v10 新增 |

## 1.3 版本演进

| 版本 | 里程碑 | 核心创新 |
|------|--------|---------|
| v3.3-HR | 高风险模式 | SOL主导、激进风格 |
| v3.4 | 权限融合 | 替代传统加权平均 |
| v3.5 | 学习机制 | 失败记忆、策略置信度 |
| v3.6 | 生产就绪 | 可靠性补丁、合规审计 |
| v6.3 | 断线保护 | 完整的断连事件处理 |
| v6.4 | 信念预算 | Thesis Budget + Regime Power |
| v6.4.1 | Profit-Max | 「Don't kill trades, modulate them」 |
| v6.5 | 数据健康 | Contract & Data Health Gate |
| v9.0 | 系统接线 | 闭环自适应: Portfolio Brain + Reliability + Drift |
| **v10.0** | **审计闭环** | **全量验证 + BullTransition + Fuse增强 + AlphaTilt + SOTA 96%** |

## 1.4 系统架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HMATS v10.0 完整架构图                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    数据获取层 (Data Layer)                           │   │
│   │   KrakenLink → 6 数据源 (Kraken/Coinglass/F&G/CryptoCompare/      │   │
│   │                          Solana RPC/Jito)                           │   │
│   │   Contract & Data Health Gate (Fail-Closed, MAX_AGE=10s)           │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   市场分析层 (Analysis Layer)                        │   │
│   │   RegimeNavigator → Per-Asset GMM (BTC k=8, ETH k=7, SOL k=7)    │   │
│   │   PhaseDetector   → 4阶段 (IGNITION/EXPANSION/SATURATION/EXHAUS) │   │
│   │   RegimeSmoother  → 滞后补偿 (hysteresis_threshold=3)             │   │
│   │   LeadLagEngine   → Binance→Kraken 领先信号 (2-tier dampening)     │   │
│   │   ★ BullTransitionDetector → 4条件牛市识别 [v10 NEW]              │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   智能体层 (Agents Layer) — 5 agent                 │   │
│   │   核心: QuantAgent (Best-of-N 4策略) / DRLAgent (TQC, ACTIVE)     │   │
│   │   情绪: SentimentAgent (F&G L1 + Haiku L3)                        │   │
│   │   偏向: ShortBiasAgent (ADVISE, soft penalty)                      │   │
│   │   风控: RiskAgent (VETO authority)                                 │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              信号融合层 (Fusion Layer)                               │   │
│   │   AuthorityFusion → 5-agent 权限矩阵 (DECIDE/VETO/ADVISE/PENAL)  │   │
│   │   ReliabilityInjection → ConfidenceScorer 动态降权                 │   │
│   │   PartialConsensus → SHADOW 模式部分共识                           │   │
│   │   DeadlockResolution → ALL_CONFLICT → NO_TRADE (不挂起)           │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   风险控制层 (Risk Layer) — 8 Veto 源               │   │
│   │   LeverageGuard (3.0×) / ThesisBudget (is_win=realized_pnl≥0)    │   │
│   │   ExistenceFuse (weekly-8%/monthly-10%/consecutive-5)              │   │
│   │   DrawdownControl (4级: 10%/15%/25%/35%)                          │   │
│   │   CorrelationCrisis (5状态: STABLE→CRISIS)                        │   │
│   │   SqueezeProtection (3级: warn/reduce/flatten)                     │   │
│   │   DeadManSwitch (refresh in try/except) / ShadowLedger            │   │
│   │   ★ BullTransitionDetector → BLOCK_NAKED_SHORT [v10 NEW]         │   │
│   │   ★ HighPositionLowVolume → 卖盘衰竭减仓 [v10 NEW]               │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              Profit-Max优化层                                       │   │
│   │   FalseBreakoutDetector (唯一硬否决) / SignalQualityScorer         │   │
│   │   AlphaGate (friction vs edge, NORMAL=14bps, OPP=8bps)            │   │
│   │   ★ CRACK 阈值集中化 (0.50/0.45/0.35) [v10 修复]                 │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              Portfolio Brain层                                      │   │
│   │   StrategicCoordinator → 多资产协调                                │   │
│   │     ├── CorrelationController → 5状态相关性 + per-asset 缩放      │   │
│   │     ├── ★ AssetAlphaTilt → Sortino-weighted 动态倾斜 [v10 NEW]   │   │
│   │     └── portfolio_allocation (归一化权重)                          │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              执行层 (Execution Layer)                                │   │
│   │   PA Executor (passive→aggressive) + TimingEngine (已接线)         │   │
│   │   Post-only orders (oflags='post', 120s timeout)                   │   │
│   │   Dynamic Slicer (ATR-based) + Fill-Slope Monitor                  │   │
│   │   ImpactCalibration ↔ ProductionMarketImpact (CalibrationBridge)  │   │
│   │   ★ Fill Rate Logging (jsonl, 周报分析) [v10 NEW]                 │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              自适应反馈层 (Feedback Layer)                           │   │
│   │   PnLEngine / ShadowLedger / FailureMemory                        │   │
│   │   ConfidenceScorer → per-strategy × per-regime 置信度              │   │
│   │   DriftDetector → 5源漂移检测 (GMM/滑点/DRL/Feature/Latent)      │   │
│   │   ★ MonteCarloValidator → 策略鲁棒性验证 [v10 NEW]               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1.5 决策流程 (v10 更新)

每4小时，系统执行一次完整的决策循环:

```
时间点 T（每4小时）
    │
    ▼
[1] 获取最新市场数据 → Contract & Data Health Gate (6数据源)
    │
    ▼
[2] 分析市场状态
    │   ├── Per-Asset GMM → regime_label
    │   ├── PhaseDetector → IGNITION/EXPANSION/SATURATION/EXHAUSTION
    │   ├── RegimeSmoother → hysteresis (threshold=3, min_persistence=2)
    │   ├── ADX Fallback → distribution-shift guard
    │   └── ★ BullTransitionDetector → INACTIVE/POTENTIAL/ACTIVE/CONFIRMED [v10]
    │
    ▼
[3] 各智能体生成信号 (5 agent)
    │   ├── QuantAgent → Best-of-N (mean_revert/momentum/volume_breakout/vrp)
    │   ├── DRLAgent → TQC LSTM_FILM_A (ACTIVE/DECIDE, obs_dim=126, n_stack=8)
    │   ├── SentimentAgent → F&G L1 + Haiku L3
    │   ├── ShortBiasAgent → funding rate 加权 + soft penalty
    │   └── RiskAgent → VETO authority
    │
    ▼
[4] Authority Fusion 信号融合
    │   ├── 5-agent 权限矩阵 (DECIDE/VETO/ADVISE/PENALIZE)
    │   ├── Reliability Injection → 低置信度策略自动降权
    │   ├── Deadlock → NO_TRADE (不挂起)
    │   └── sentiment_zscore = (fg_value - 50) / 50.0 × 3.0
    │
    ▼
[5] 风险检查（8 veto 源, 任一否决 → 交易取消）
    │   ├── Constitution / RiskManager / DMS / Squeeze
    │   ├── LeverageGuard / DrawdownControl / CorrelationCrisis
    │   ├── ExistenceFuse (weekly-8%/monthly-10%/consecutive-5)
    │   └── ★ BullTransition: CONFIRMED → BLOCK_NAKED_SHORT [v10]
    │       ★ BullTransition: ACTIVE → REDUCE_SHORT ×0.5 [v10]
    │
    ▼
[6] Profit-Max 优化
    │   ├── FalseBreakoutDetector（唯一硬否决）
    │   ├── SignalQualityScorer → conviction 乘数
    │   ├── Alpha Gate: friction vs edge 检查
    │   ├── CRACK 阈值: FULL_EXIT=0.50, PARTIAL=0.45, URGENCY=0.35
    │   └── ★ HPLV Filter: 价格≥90th + 量<60%avg → short ×0.5 [v10]
    │
    ▼
[7] Portfolio Brain 协调
    │   ├── CorrelationController → per-asset exposure 约束
    │   ├── ★ AssetAlphaTilt → Sortino-weighted multiplier (0.5~1.5) [v10]
    │   └── portfolio_allocation (归一化)
    │
    ▼
[8] 执行
    │   ├── PA Executor (passive→aggressive, post_only)
    │   ├── TimingEngine → timing_mode/timing_score → PA behavior
    │   ├── Dynamic Slicer (ATR-based)
    │   ├── ImpactCalibration → bucket-level 校准参数
    │   └── ★ Fill Rate Logging (ratio/time/slippage/repriced) [v10]
    │
    ▼
[9] 反馈记录
    │   ├── PnL / ShadowLedger / FailureMemory
    │   ├── ConfidenceScorer.record_signal() + record_outcome()
    │   ├── DriftDetector 更新 (5源)
    │   └── ExistenceFuse.on_trade_close() → consecutive_loss 计数
    │
    ▼
[10] 每24H (每6 tick):
    │   ├── ★ AssetAlphaTilt.update() → 重算 per-asset multiplier [v10]
    │   └── WeekendManager 检查
    │
    ▼
等待下一个4小时周期
```

---

# 第二部分：数据获取层

## 2.1 核心模块: KrakenLink + Contract & Data Health Gate

**目标账户**: $10K 起步 → $100K (盈利后扩展)，Kraken Pro 会员（$10K/月免手续费交易额度）

| 数据源 | 数据类型 | 更新频率 | 用途 | v10 状态 |
|--------|---------|---------|------|---------|
| Kraken API | OHLCV + 订单簿 + 账户 | 实时 | 基础数据 | ✅ LIVE |
| Coinglass | OI + 清算数据 | 4H | 机构行为 | ✅ LIVE |
| Alternative.me | Fear & Greed Index | 24H | 情绪信号 | ✅ LIVE |
| CryptoCompare | 链上指标 | 1H | 链上分析 | ✅ LIVE |
| Solana RPC | SOL 网络指标 | 实时 | SOL 特有 | ✅ LIVE |
| Jito API | MEV 指标 | 实时 | SOL 特有 | ✅ LIVE |

## 2.2 Contract-First 数据健康门控

系统采用 **Fail-Closed** 原则:

| 规则 | 行为 |
|------|------|
| 必需资产缺失 (BTC/ETH/SOL 任一) | → NO_TRADE |
| OHLCV数据时效 > MAX_DATA_AGE_SECONDS (10s) | → NO_TRADE |
| Microstructure数据时效 > 2s | → 降级 (vpin/ofi 置 None) |
| 可选安全字段缺失 (vpin, dvol_zscore) | → 保持 None (不猜测默认值) |

---

# 第三部分：市场分析层

## 3.1 RegimeNavigator (Per-Asset GMM)

| 资产 | GMM k | 模型文件 | 训练数据 |
|------|-------|---------|---------|
| BTC | k=8 | gmm_model_btc.pkl | 2020-2024 |
| ETH | k=7 | gmm_model_eth.pkl | 2020-2024 |
| SOL | k=7 | gmm_model_sol.pkl | 2020-2024 |

核心参数:
- cross_asset_correlation default: **0.87** (统一, 无 0.0/0.65 残留)
- ADX fallback: distribution-shift guard, z > 3σ + > 30% 偏移时触发
- 冷启动保护: per-asset _warmup_ticks, 前 2 tick 标记 degraded

## 3.2 PhaseDetector (4阶段)

IGNITION → EXPANSION → SATURATION → EXHAUSTION, 影响:
- Alpha Gate 阈值
- Exit Alpha 行为 (Phase-Aware Exit)
- Leverage 调整
- CRACK 权重计算

## 3.3 RegimeSmoother + Hysteresis

- hysteresis_threshold = 3: 新 regime 概率需超过当前 20% margin
- min_persistence = 2: 持续 2 个 4H bar 才确认切换
- 降低 ~30% regime flip rate

## 3.4 BullTransitionDetector (v10 新增)

**最关键的做空系统保护** — 识别牛市转换, 防止温和牛市中缓慢亏损 (审计 §23: 最坏场景 -20~25% / 8周)。

4 条件:
1. BTC Golden Cross: weekly MA50 > MA200
2. SOL/BTC 14D relative strength > 0
3. Funding positive 7 天连续
4. OI rising + liquidations falling

4 状态机:
```
INACTIVE (0条件) → POTENTIAL (1条件) → ACTIVE (2+条件) → CONFIRMED (5+天)
```

动作:
| 状态 | 做空限制 | Funding 权重 |
|------|---------|-------------|
| INACTIVE | 无 | ×1.0 |
| POTENTIAL | 无 | ×1.0 |
| ACTIVE | 空头仓位 ×0.5 | ×0.5 |
| CONFIRMED | **禁止裸空**, 只允许对冲 | ×0.5 |

接线: main.py L758 (初始化), L3645-3660 (每 tick 评估, 在 direction override 之前)。

---

# 第四部分：智能体层（完整列表）

## 4.1 权限矩阵 (5 Agent)

| Agent | 权限 | 功能 |
|-------|------|------|
| QuantAgent | DECIDE | Best-of-N 4策略选择 (mean_revert/momentum/volume_breakout/vrp) |
| DRLAgent | DECIDE | TQC obs_dim=126, SHADOW→EXIT_ONLY 晋升路径, ent_coef=0.1 |
| SentimentAgent | ADVISE | F&G L1 + Haiku L3, sentiment_zscore = (fg-50)/50 × 3.0 |
| ShortBiasAgent | PENALIZE | 做多软惩罚 ×0.7, funding>0.24%/8h → short +15% |
| RiskAgent | VETO | 风控否决 (一票否决权) |

## 4.2 DRL 管理

- 起始状态: DISABLED
- 晋升: StatisticalPromotionGate (30天 shadow + Sharpe>1.0 + win_rate>48%)
- 降级: 5 consecutive losses / 15% DD → EXIT_ONLY
- 铁律: obs_dim=126 不碰, ent_coef=0.1 不碰, 训练代码不碰
- FiLM Conditioning: regime 条件化特征调制
- OOD Detector: Mahalanobis distance, 分布外样本 → 降权

---

# 第五部分：信号融合层

## 5.1 Authority Fusion

5-agent 权限矩阵, 3 模式 (FULL/EXIT_ONLY/DISABLED), 输出 final_direction。

## 5.2 Reliability Injection

ConfidenceScorer (per-strategy × per-regime) 实时评分:
- 方向准确率 (35%)
- PnL vs 期望 (35%)
- Regime 准确率 (30%)

低置信度 (< 0.35) → conviction_multiplier × 0.3 (软降权)。

## 5.3 Deadlock Resolution

ALL_CONFLICT_FLAT → NO_TRADE (不永久挂起)。CONTINUE→NONE 已修复。

---

# 第六部分：风险控制层

## 6.1 8 Veto 源 (One-Veto-Kill)

| Veto 源 | 触发条件 | 动作 |
|---------|---------|------|
| Constitution | 参数违规 | 阻止交易 |
| RiskManager | 风险超限 | 阻止交易 |
| DeadManSwitch | 心跳超时 | 撤单 (refresh in try/except) |
| SqueezeProtection | score ≥ 0.50/0.70/0.80 | warn/reduce/flatten |
| LeverageGuard | > 3.0× | 拒绝/削减 |
| DrawdownControl | 4级梯度 | 10%→减仓, 15%→大幅减仓, 25%→暂停, 35%→kill |
| CorrelationCrisis | 5状态 | SPIKING→减仓, CRISIS→停止新开仓 |
| ExistenceFuse | 多层 | 见 6.2 |

## 6.2 Strategy Existence Fuse (v10 增强)

**v9**: 仅 3 笔连续亏损 / 24h 暂停。
**v10**: 多层递进保护:

| 层级 | 触发条件 | 动作 | 严重度 |
|------|---------|------|--------|
| 连续亏损 | 5 笔连亏 | 暂停 24h | 轻 |
| 周损失 | ≥ 8% | 暂停, 通知人工 (HALT) | 中 |
| 月损失 | ≥ 8% | 观察模式, 减半仓位 (OBSERVE) | 中 |
| 月损失 | ≥ 10% | Kill switch (KILL) | 重 |
| Drawdown | ≥ 25% | 系统级 kill (在 DrawdownControl 中) | 最重 |

返回值: `(action, reason)`, action ∈ {NONE, OBSERVE, HALT, KILL}。

连续亏损计数现在同时存在于:
- ExistenceFuse (系统级, 所有交易)
- DRL PromotionGate (DRL级, 只看 DRL 交易 → EXIT_ONLY 降级)

## 6.3 CorrelationRealtimeController

1,098 行完整实现:
- 多时间尺度: 20/60/200 bars
- 5 状态: STABLE → ELEVATED → SPIKING → CRISIS → COLLAPSING
- Spike 检测: 0.98 threshold
- 预测性调整 (EWMA) + 跳跃检测 (z-score) + 特征值分析
- per-asset 仓位缩放 (get_position_adjustment)

## 6.4 HighPositionLowVolumeFilter (v10 新增)

审计 §22 纪律⑦: 高价位 + 低成交量 = 卖盘衰竭 → 减少做空暴露。

| 参数 | 值 | 含义 |
|------|-----|------|
| HIGH_PRICE_PERCENTILE | 90 | 价格 ≥ 90th 视为高位 |
| LOW_VOLUME_RATIO | 0.6 | 量 < 60% 均量视为缩量 |
| SHORT_EXPOSURE_REDUCTION | 0.5 | 减半空头 |

接线: main.py L9521-9544, ACTIVE (非 SHADOW)。

---

# 第七部分：Profit-Max优化层

## 7.1 核心组件

| 组件 | 功能 | 类型 |
|------|------|------|
| FalseBreakoutDetector | 假突破检测 | 唯一硬否决 |
| SignalQualityScorer | 信号质量评分 | conviction 乘数 |
| RegimeTransitionRisk | regime 转换风险 | 软调整 |
| ProfitMaxAdapter | 整合所有优化 | 适配器 |

## 7.2 Alpha Gate (Fee-Aware, Volume-Aware)

```
alpha_bps vs friction_bps:
  alpha > friction × 1.5 → 正常执行
  friction < alpha < friction × 1.5 → 降速拆单
  alpha < friction → VETO: FRICTION_EXCEEDS_EDGE
```

月交易量追踪: get_monthly_volume() + on_trade_fill(), $10K free tier threshold。

| 模式 | Free Tier 门槛 |
|------|---------------|
| NORMAL | 14 bps |
| OPPORTUNITY | 8 bps |

## 7.3 CRACK 阈值 (v10 集中化)

v9: 三处不同值 (0.50/0.45/0.35 散落)。v10: 集中定义:

| 层级 | 阈值 | 含义 |
|------|------|------|
| FULL_EXIT | 0.50 | CRACK 权重 < 0.50 → 完全退出论题 |
| PARTIAL_EXIT | 0.45 | exit_alpha 激活, 渐进式退出 |
| URGENCY | 0.35 | 紧急判断 |

---

# 第八部分：Portfolio Brain层

## 8.1 StrategicCoordinator

连接多个子系统输出为统一的 StrategicIntent:
- correlations: 实时相关性矩阵
- portfolio_allocation: 归一化权重
- recommended_exposure: per-asset soft cap

## 8.2 Cross-Asset Rules

| 资产 | Max Exposure | 说明 |
|------|-------------|------|
| BTC | 25% | 流动性最高 |
| ETH | 25% | 中等流动性 |
| SOL | 20% | 高 beta, 需限制 |

## 8.3 AssetAlphaTilt (v10 新增)

SOTA G6-lite 简化版 — 不是完整的 Portfolio Brain, 只是 per-asset Sortino-weighted 倾斜。

| Sortino (30D) | Exposure 乘数 | 含义 |
|---------------|-------------|------|
| ≥ 2.0 | 1.5 | 优秀, 加仓 |
| ≥ 1.0 | 1.3 | 良好 |
| ≥ 0.0 | 1.0 | 中性 |
| ≥ -0.5 | 0.7 | 较差, 减仓 |
| < -0.5 | 0.5 | 严重亏损, 大幅减仓 |

更新频率: 每 24h (每 6 个 4H tick)。平滑系数 α=0.3。冷启动: < 5 笔交易不倾斜 (mult=1.0)。

---

# 第九部分：执行层

## 9.1 双时间尺度控制回路

- 4H 决策主循环: process_4h_tick() — 唯一交易决策入口
- 200ms 执行子循环: PA executor reprice interval (reserved, 文档化)

## 9.2 Passive-Aggressive Executor

| 模式 | 行为 |
|------|------|
| PASSIVE | post_only limit order, 等待成交 |
| AGGRESSIVE | 120s timeout → cancel & market |
| ABORT | 取消, 不执行 |

TimingEngine → timing_mode + timing_score → 影响 PA executor 行为 (v10 已接线)。

## 9.3 其他执行组件

- Dynamic Slicer: ATR-based 拆单
- Fill-Slope Monitor: 成交率监控 (adverse selection 检测)
- Market Impact: ImpactCalibration (bucket-level) → ProductionMarketImpact (CalibrationBridge)
- Post-only: oflags='post', postOnly=True

## 9.4 Fill Rate Logging (v10 新增)

SOTA G8-lite — 不做自动调整, 只记录到 `logs/fill_quality.jsonl`:

| 字段 | 类型 | 说明 |
|------|------|------|
| order_type | str | limit / market |
| fill_ratio | float | 0-1, 成交比例 |
| time_to_fill_s | int | 成交耗时 (秒) |
| slippage_bps | float | 滑点 (bps) |
| was_repriced | bool | PA executor 是否 reprice |

每周手动 review, 积累数据后考虑自动化调参。

---

# 第十部分：自适应反馈层

## 10.1 ConfidenceScorer (已接通)

record_signal() + record_outcome() 在 main.py 中调用。
per-strategy × per-regime 独立评分, 输出到 Authority Fusion 做降权。

## 10.2 DriftDetector (5源)

| 数据源 | 检测方法 | 触发条件 | 效果 |
|--------|---------|---------|------|
| Feature distribution | z-score | z > 3.0 | 降权 |
| Latent space | distance | dist > threshold | 降权 |
| GMM Regime | JSD | JSD > 0.15 | DRL 降权 |
| 执行滑点 | z-score | z > 2.0 | 市场恶化警告 |
| DRL Action | mean shift | z > 2.5 / std > 2.0 | EXIT_ONLY 降级 |

## 10.3 MonteCarloValidator (v10 新增)

Post-paper-run 分析工具:
- Shuffle trade PnL 1000+ 次
- 验证: win rate vs random, 95% CI, worst-case max DD
- 输出: is_robust (bool), bankruptcy_probability

## 10.4 ShadowLedger

所有交易记录写入, 包含 tick_id 用于审计追踪。v10 新增 fill_quality 字段。

---

# 第十一部分：策略哲学与盈利逻辑

## 11.1 「Aggressive Alpha, Defensive Shell」

进攻端 (Alpha): CRACK system, Lead-Lag, Regime Phase-Aware trading, SOL Dominance Mode。
防御端 (Shell): 8 veto, 4级 drawdown, squeeze protection, bull transition, existence fuse。

## 11.2 做空特化设计

HMATS 主做空, 系统有 10+ 个做空特化能力:

| 能力 | 做空价值 |
|------|---------|
| CRACK System | 结构突破信号 = 做空核心 alpha |
| Phase-Aware Exit | EXHAUSTION 阶段精确平仓 |
| BullTransitionDetector | 牛市来临时停手, 避免 -25% 场景 |
| SOL Dominance Mode | SOL 高 beta = 做空收益放大器 |
| Existence Fuse (multi-layer) | 做空亏损可无上限, 这个保命 |
| One-Veto-Kill | 挤空时硬否决救命 |
| Squeeze Protection 3-tier | 专门针对做空最大风险 |
| Short-Bias Sentiment | 看多情绪 = 做空减仓信号 |
| 15 条做空纪律 | SOTA 没有的做空规则集 |
| Higher Lows Detection | 底部抬升 = 减少做空 |

## 11.3 v10 盈利改进 (累积, 含 v9)

| 改进 | 机制 | 预期影响 |
|------|------|---------| 
| Reliability Injection | 低置信度策略自动降权 | -10~15% 垃圾信号亏损 |
| Portfolio Brain | 相关性约束 + 动态分配 | -15~20% 集中风险 |
| Alpha Gate | friction vs edge 检查 | -5~10% 负 EV 交易 |
| Impact Calibration | bucket-level 精细校准 | -5~10 bps 冲击成本 |
| Drift Detection | 5源漂移检测 | 模型失效时自动降级 |
| **BullTransitionDetector** | 牛市识别 + 裸空禁止 | **避免 -20~25% 最坏场景** |
| **Existence Fuse 增强** | weekly/monthly 多层 | **加速止损, 不让亏损累积** |
| **AssetAlphaTilt** | Sortino-weighted 倾斜 | **+资金集中到表现好的资产** |
| **HPLV Filter** | 卖盘衰竭检测 | **减少逆势做空** |

---

# 第十二部分：技术栈总览

## 12.1 编程语言与框架

Python 3.10+, PyTorch 2.0+, stable-baselines3, Optuna。

### 硬件

| 组件 | 运行 | 训练 |
|------|------|------|
| CPU | 8核+ | 8核+ |
| RAM | 16GB+ | 32GB+ |
| GPU | 不需要 | RTX 5090 (DRL) |
| 存储 | 100GB SSD | 200GB SSD |

### Kraken Pro 集成

| 项目 | 配置 |
|------|------|
| 会员等级 | Kraken Pro |
| 免费额度 | $10K/月 |
| API 权限 | Query Funds + Orders + Create + Cancel + WebSocket |
| 禁止权限 | Withdraw |

## 12.2 核心参数速查 (v10 统一)

| 参数 | 值 | 来源 |
|------|-----|------|
| 账户 | $10K (扩展目标 $100K) | .env |
| 决策频率 | 4H (14400s) | main.py |
| 执行子循环 | 200ms (reserved) | SOTAConfig |
| 最大杠杆 | 3.0× | sota_flags.py |
| Hard Drawdown | 25% halt, 35% kill | sota_flags.py |
| cross_asset_correlation | 0.87 | 统一 |
| CRACK: FULL/PARTIAL/URGENCY | 0.50 / 0.45 / 0.35 | 集中定义 |
| VPIN Toxic | > 0.85 | constitution |
| DRL obs_dim | 126 (122 features + 4 env) | 铁律, 不碰 |
| DRL ent_coef | 0.1 (fixed) | 铁律, 不碰 |
| DRL 晋升 | 30天 shadow + Sharpe>1.0 + win>48% | PromotionGate |
| Maker/Taker fee | 16/26 bps | Kraken Pro |
| BTC/ETH max exposure | 25% | portfolio |
| SOL max exposure | 20% | portfolio |

## 12.3 路线图

### v10.0 完成的工作

| Task | 内容 | 状态 |
|------|------|------|
| 全量审计 (1909行) | 121 子章节, 170 发现 | ✅ 完成 |
| 自检 (84 checks) | 74% LIVE → 修复 → 97.7% GREEN | ✅ 完成 |
| P0 Bug 修复 (16项) | $100K, engine.configure, sentiment_zscore 等 | ✅ 全部修复 |
| Residual Fix (11项) | 5 bug + 6 module | ✅ 11/11 PASS |
| BullTransitionDetector | 4条件, 4状态机, main.py 接线 | ✅ LIVE |
| Existence Fuse 增强 | weekly/monthly/consecutive 多层 | ✅ LIVE |
| HPLV Filter | 卖盘衰竭检测 | ✅ LIVE |
| CRACK 集中化 | 0.50/0.45/0.35 | ✅ LIVE |
| TimingEngine 接线 | timing_mode/score → PA executor | ✅ LIVE |
| cross_asset_correlation 统一 | 全部 0.87 | ✅ LIVE |
| Monte Carlo Validator | 策略鲁棒性验证 | ✅ LIVE |
| Cash-and-Carry Advisory | Phase 1 (signal only) | ✅ LIVE |
| Navigator 冷启动 | warmup 保护 | ✅ LIVE |
| 再审计 (91 checks) | 80 LIVE + 7/7 data flows | ✅ GREEN |
| SOTA 分析 (10 gap) | 过滤后 2 gap (G6-lite, G8-lite) | ✅ 完成 |
| Final Polish (6项) | 4 PARTIAL→LIVE + 2 SOTA | 📋 执行中 |

### 未来版本

| 版本 | 功能 | 预计时间 |
|------|------|---------| 
| v10.1 | 24h paper run 验证 → live deployment | 2026-03 |
| v10.2 | DRL TQC SHADOW→EXIT_ONLY 晋升 | Stage 8 后 |
| v11.0 | 三资产批量决策 (collect → joint optimize → execute) | Q2 2026 |
| v11.1 | Account scaling $10K → $100K 优化 | Q2 2026 |
| v12.0 | Stage 21 Meta-Learner + 完全自主参数调整 | Q3 2026 |

## 12.4 关键文件结构

```
hmats/
├── main.py                              # 主入口 (~13,000 行)
├── VERSION.txt                          # 6.5.1
├── .env                                 # HMATS_INITIAL_CAPITAL=10000
│
├── core/
│   ├── sota_flags.py                    # 集中参数 (drawdown/leverage/alpha_gate)
│   ├── config_resolver.py               # 启动审计, 冲突检测
│   ├── regime_smoother.py               # hysteresis + min_persistence
│   ├── exchange_guard.py                # Kraken-only 保护
│   └── asset_alpha_tilt.py              # ★ Sortino-weighted 动态倾斜 [v10]
│
├── risk/
│   ├── risk_manager.py                  # 核心风控
│   ├── bull_transition_detector.py      # ★ 4条件牛市检测 [v10]
│   ├── strategy_existence_fuse.py       # ★ 多层熔断 (weekly/monthly) [v10增强]
│   ├── thesis_budget_governor.py        # Thesis 预算
│   ├── leverage_guard.py                # 3.0× hard cap
│   └── correlation_realtime_controller.py  # 5状态相关性
│
├── defense/
│   ├── constitution.py                  # 参数验证 + veto
│   ├── p0_safety_integrator.py          # 6层安全链
│   └── sol_defense.py                   # SOL 特有防护
│
├── agents/
│   ├── kraken_quant_agent.py            # Best-of-N 4策略
│   ├── short_bias_agent.py              # 做空偏向 + funding
│   └── ...
│
├── signals/
│   ├── opportunity_triggers.py          # 5组 trigger + CRACK 集中阈值
│   ├── high_position_low_volume_filter.py  # ★ 卖盘衰竭 [v10]
│   ├── regime_transition_risk.py        # 转换概率矩阵
│   └── authority_fusion.py              # + Reliability Injection
│
├── execution/
│   ├── passive_aggressive.py            # PA executor + timing wiring
│   ├── production_market_impact.py      # CalibrationBridge 注入
│   └── impact_calibration.py            # bucket-level 标定
│
├── analytics/
│   ├── confidence_scorer.py             # per-strategy × per-regime
│   └── drift_detector.py               # 5源漂移检测
│
├── tools/
│   └── monte_carlo_validator.py         # ★ 策略鲁棒性验证 [v10]
│
├── orchestration/
│   └── strategic_coordinator.py         # Portfolio Brain
│
├── market/
│   ├── phase_detector.py                # 4阶段
│   └── gmm_models/                      # per-asset pkl
│
├── training/                            # DRL 训练 (不碰)
├── logs/
│   └── fill_quality.jsonl               # ★ 成交质量记录 [v10]
│
└── configs/
    ├── cloud_production.json
    └── ...
```

---

# 附录A：模块依赖关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HMATS v10.0 模块依赖图 (★ = v10 新增/修复)                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   main.py    │
                              │  (主入口)     │
                              └──────┬───────┘
                                     │
     ┌───────────┬───────────┬───────┼───────┬──────────────┬─────────────┐
     │           │           │       │       │              │             │
     ▼           ▼           ▼       ▼       ▼              ▼             ▼
 ┌────────┐┌─────────┐┌────────┐┌────────┐┌──────────┐┌──────────┐┌──────────┐
 │ infra/ ││market/  ││agents/ ││signals/││ risk/    ││execution/││analytics/│
 │        ││         ││ (×5)   ││        ││          ││          ││          │
 │kraken  ││gmm_nav  ││quant   ││fusion  ││★bull_det ││pa_exec   ││confidence│
 │link    ││phase_det││drl     ││triggers││★fuse_enh ││timing_eng││drift_det │
 │        ││★smoother││sent    ││★hplv   ││squeeze   ││★fill_log ││★monte_mc │
 └────────┘│lead_lag ││short   ││crack   ││drawdown  ││impact_cal││          │
           │★bull_det││risk    ││alpha_gt││corr_ctrl │└──────────┘└──────────┘
           └─────────┘└────────┘└────────┘│leverage  │
                                          │dms      │
                                          │thesis   │
                                          └─────────┘
                                               │
                                               ▼
                                     ┌──────────────────┐
                                     │  orchestration/   │
                                     │  strategic_coord  │
                                     │  ★ alpha_tilt    │
                                     └──────────────────┘
```

---

# 附录B：审计验证记录

## B.1 审计链

| 步骤 | 文档 | 结果 |
|------|------|------|
| 1. 综合审计 | HMATS_INTEGRATED_AUDIT.md (1909行) | 121 子章节, 170 发现 |
| 2. 自检 | HMATS_SELFCHECK_PROMPT.md (84 checks) | 54 LIVE (74%) |
| 3. 残余修复 | HMATS_RESIDUAL_FIX_PROMPT.md (11项) | 11/11 PASS |
| 4. 修复验证 | HMATS_POSTFIX_VERIFICATION.md | 11/11 PASS |
| 5. 再审计 | HMATS_FINAL_REAUDIT_PROMPT.md (91 checks) | 80 LIVE (97.7%) |
| 6. Final Polish | HMATS_FINAL_POLISH_PROMPT.md (6项) | 执行中 |

## B.2 7 端到端数据流

| Flow | 链路 | 状态 |
|------|------|------|
| DF-01 | Signal → Fusion → Direction | ✅ INTACT |
| DF-02 | Direction → Sizing → Order | ✅ INTACT |
| DF-03 | MarketData → Regime → Decision Gates | ✅ INTACT |
| DF-04 | Risk Checks → Veto Chain → Execution Guard | ✅ INTACT |
| DF-05 | BullDetector → BLOCK/REDUCE_SHORT → Override | ✅ INTACT |
| DF-06 | Trade Result → Fuse Check → HALT/KILL | ✅ INTACT |
| DF-07 | TimingEngine → timing_mode → PA Executor | ✅ INTACT |

---

# 附录C：SOTA覆盖分析

## C.1 场景过滤

SOTA 文档假设: 机构级, 多交易所, 多策略, 双向。
HMATS 实际: **个人, 单交易所 (Kraken), 三币种, 主做空, 高风险偏好, 4H**。

## C.2 过滤后 SOTA 覆盖

| SOTA 章节 | 通用覆盖度 | 过滤后覆盖度 | 说明 |
|-----------|-----------|-------------|------|
| §1 策略分层 | 70% | 95% | Best-of-N 单选模式 + ThesisBudget 已覆盖 |
| §2 DRL 配比 | 95% | 95% | HMATS 更细致 (PromotionGate) |
| §3 六类 Alpha | 80% | 90% | 做空不需要双向 straddle |
| §4 伪结构套利 | 60% | 90% | 4H+$10K 下不可行的已排除 |
| §5 Portfolio Brain | 50% | 85% | AssetAlphaTilt 覆盖核心需求 |
| §6 资金效率 | 85% | 95% | $10K 无 impact 问题 |
| §7 行为适应 | 95% | 95% | HMATS 更丰富 (CRACK, Strategy Aging) |
| §8 执行逻辑 | 80% | 90% | Fill Rate Logging 补齐 |
| §9 路线图 | 90% | 90% | 对齐 |
| **综合** | **~80%** | **~96%** | **含超越项 >100%** |

## C.3 HMATS 超越 SOTA

| 超越项 | 说明 |
|--------|------|
| CRACK System | 结构突破信号, SOTA 无对应 |
| Phase-Aware Exit | 4阶段精确退出 |
| BullTransitionDetector | 做空系统最关键保护 |
| SOL Dominance Mode | 高 beta 资产特化 |
| Lead-Lag Alpha | 跨所领先信号 |
| Existence Fuse (multi-layer) | 多层递进保护, SOTA 无此深度 |
| One-Veto-Kill | 硬否决, SOTA 依赖 fusion |
| Startup Reconciler | 重启对账, SOTA 未提及 |
| Cancel-on-Disconnect | 断线撤单, SOTA 只说测试 |
| FiLM Conditioning | DRL regime 调制, SOTA 未提及 |
| 15 条做空纪律 | 做空特化规则集 |

---

## 术语表

| 术语 | 定义 |
|------|------|
| NAV | Net Asset Value，净资产价值 |
| GMM | Gaussian Mixture Model，高斯混合模型 |
| DRL | Deep Reinforcement Learning，深度强化学习 |
| TQC | Truncated Quantile Critics，HMATS DRL 核心算法 |
| FiLM | Feature-wise Linear Modulation，特征调制层 |
| CRACK | Catalyst-Regime Aligned Conviction Kernel |
| PA | Passive-Aggressive (执行模式) |
| HPLV | High Position Low Volume (卖盘衰竭) |
| JSD | Jensen-Shannon Divergence |
| Sortino | 下行风险调整后收益 |
| Almgren-Chriss | 市场冲击理论模型 |

---

**文档结束**

v10.0 的核心贡献: 通过系统性的审计-修复-再审计循环, 将代码集成率从 ~55% 提升到 97.7%, 新增 BullTransitionDetector (做空最关键保护) 和 Existence Fuse 多层增强 (系统自我退出能力), 并通过 SOTA 语境过滤确认系统在做空+单交易所+3币种场景下已超越 SOTA 要求。

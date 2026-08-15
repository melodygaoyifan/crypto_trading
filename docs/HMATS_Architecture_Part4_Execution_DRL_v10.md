# HMATS 系统架构文档 - Part 4
# 执行层与DRL训练
# ═══════════════════════════════════════════════════════════════
# ⚠️ HISTORICAL DOCUMENT (banner added 2026-08-14, P269). 本文档描述的是
# 2026年3月的系统状态，多处已被后续事实推翻，只作考古参考：
#   - "DRL now DECIDE in fusion" 已失效 — DRL 于 2026-08-07 降级为 SHADOW
#     (P198)，且 P200/P241/P258 三轮干净重训均 0/9 fold 通过；
#   - 本文的 DRL 训练流程/性能数字系 P164 泄漏时代的产物 (P179–P184)；
#   - 训练的权威文档是 docs/HMATS_TRAINING_GUIDE_V2.md，运行时状态的
#     权威文档是 CLAUDE.md。
# 版本: v10.1-POSTAUDIT (v6.8.0 sync)
# 日期: 2026年3月27日 (updated from Feb 28)
# 审计状态: TimingEngine→PA Executor 接线验证 INTACT (DF-07)
# v6.8.0 变更 (当时): FIX-DRL-AUTHORITY, FIX-MARKET-FALLBACK,
#              FIX-RECONNECT-ORDER, FIX-DRL-VALIDATE (smoke test), FIX-FEE-TIER
# ═══════════════════════════════════════════════════════════════

## 本部分目录

1. [执行层架构](#执行层架构)
2. [PA Executor 详解](#pa-executor-详解)
3. [市场冲击建模](#市场冲击建模)
4. [Fill Rate Logging (v10)](#fill-rate-logging-v10)
5. [DRL训练流程: 20 Stages, 35 Iron Laws](#drl训练流程-20-stages-35-iron-laws)
6. [DRL 生产部署](#drl-生产部署)
7. [模型版本管理](#模型版本管理)

---

## 执行层架构

执行层将**战略意图**转化为**已执行订单**, 具有滑点控制和费用优化。

### Intent → Order 解耦

```
┌─────────────────────────────────────────────────────────────────┐
│  Trade Intent (策略想要什么)                                    │
│  - 创建后不可变                                                 │
│  - 包含: direction, size, confidence, urgency                   │
│  - 创建者: process_4h_tick() 步骤 [7]                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  执行层 (如何实现)                                              │
│  - 可以失败而不使策略无效                                       │
│  - 单独记录执行质量 (fill_quality.jsonl)                       │
│  - TimingEngine → timing_mode/score → PA behavior              │
└─────────────────────────────────────────────────────────────────┘
```

### 执行层组件总览

```
┌─────────────────────────────────────────────────────────────────┐
│  PA Executor (passive_aggressive.py)                            │
│  ├─ PASSIVE: post_only limit order, 等待成交                    │
│  ├─ AGGRESSIVE: 120s timeout → cancel & market                 │
│  ├─ ABORT: 取消, 不执行                                        │
│  └─ oflags='post', postOnly=True                               │
│                                                                 │
│  TimingEngine (timing_engine.py)                                │
│  ├─ timing_mode: DELAY/PASSIVE_ONLY/PASSIVE_PREFERRED/AGGRESSIVE│
│  ├─ timing_score: 0.0~1.0                                      │
│  └─ v10: 已接线到 PA Executor (DF-07 验证 INTACT)             │
│                                                                 │
│  ImpactCalibration → ProductionMarketImpact                    │
│  ├─ CalibrationBridge: bucket-level params → Almgren-Chriss    │
│  ├─ 有置信度 (conf>0.3) → bucket 参数                          │
│  ├─ 无置信度 → fallback per-symbol 自标定                      │
│  └─ record_execution() 双写                                    │
│                                                                 │
│  DynamicSlicer (ATR-based 拆单)                                │
│  ├─ 大订单 → 3-10 个切片                                       │
│  └─ 基于 ATR 和订单簿深度                                       │
│                                                                 │
│  FillSlopeMonitor (adverse selection 检测)                     │
│  ├─ 成交时间 <500ms → 有毒流量                                  │
│  └─ 取消并重新评估                                              │
│                                                                 │
│  ★ FillQualityLogger [v10]                                     │
│  ├─ fill_ratio, time_to_fill_s, slippage_bps                  │
│  ├─ was_repriced, order_type, final_action                     │
│  └─ 输出: logs/fill_quality.jsonl                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## PA Executor 详解

核心执行引擎, 实现 Passive-Aggressive 策略:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PA Executor — Passive-Aggressive 策略                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  默认: PASSIVE (Maker偏好)                                               │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ 在最佳 bid/ask 提交限价单                                    │       │
│  │ ├─ 16bps 费用 (Kraken Pro maker)                             │       │
│  │ ├─ oflags='post', postOnly=True (强制 maker)                │       │
│  │ ├─ 耐心成交 (最多 120 秒超时)                                │       │
│  │ └─ 价格: mid ± (spread/4) 取决于方向                         │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  AGGRESSIVE 触发 (切换到 TAKER):                                        │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ 1. 120s 限价单超时 → cancel & market                        │       │
│  │ 2. Urgency = IMMEDIATE (CRACK, 止损)                        │       │
│  │ 3. TimingEngine score > 0.7 → AGGRESSIVE_TAKER              │       │
│  │ 4. VPIN > 0.75 (有毒订单流, 需要速度)                       │       │
│  │ 26bps taker 费用                                             │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ABORT:                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ TimingEngine score < 0.3 → DELAY (不执行)                   │       │
│  │ FillSlope 检测到有毒 → cancel & re-evaluate                 │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  TimingEngine 接线 (v10 已验证 INTACT):                                  │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ timing_score < 0.3  → DELAY (不执行)                        │       │
│  │ timing_score 0.3-0.5 → PASSIVE_ONLY (仅 maker)             │       │
│  │ timing_score 0.5-0.7 → PASSIVE_PREFERRED (maker + fallback)│       │
│  │ timing_score > 0.7  → AGGRESSIVE_TAKER (速度关键)          │       │
│  │                                                               │       │
│  │ 考虑因素:                                                    │       │
│  │ ├─ Spread 宽度 (更窄 = 更好)                                │       │
│  │ ├─ 订单簿深度 (更深 = 更好)                                 │       │
│  │ ├─ VPIN 水平 (更低 = 更好)                                  │       │
│  │ ├─ Lead-lag 信号强度                                        │       │
│  │ └─ 时间/流动性 regime                                       │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  费用优化:                                                               │
│  ├─ Maker: 16bps, Taker: 26bps (Kraken Pro)                            │
│  ├─ 差额: 10bps/trade                                                   │
│  ├─ 目标: >70% maker 成交率                                             │
│  └─ $10K/月 免费交易额度 (Kraken Pro 会员)                               │
│                                                                          │
│  Startup Reconciler (Anti-Churn):                                        │
│  ├─ 重启时对账 → 不重复开仓                                             │
│  ├─ Cancel-on-Disconnect (断线撤单)                                      │
│  └─ AC-0~5 修复: 消除 restart churn (#1 费用杀手)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 市场冲击建模

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ImpactCalibration → ProductionMarketImpact                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CalibrationBridge:                                                      │
│  ├─ bucket-level 精细标定 (按价格区间)                                   │
│  ├─ 有置信度 (conf > 0.3) → 使用 bucket 参数                            │
│  └─ 无置信度 → fallback per-symbol 自标定                               │
│                                                                          │
│  Almgren-Chriss 扩展模型:                                                │
│  ├─ 永久冲击: I_perm = γ × (Q/V)^α                                     │
│  ├─ 临时冲击: I_temp = η × (dQ/dt)                                     │
│  └─ 最优切片: N = sqrt(Q × λ / η)                                      │
│                                                                          │
│  record_execution() 双写:                                                │
│  ├─ 写入 ImpactCalibration (更新 bucket 参数)                           │
│  └─ 写入 ProductionMarketImpact (全局统计)                              │
│                                                                          │
│  $10K 账户下的影响:                                                      │
│  ├─ 订单规模 < 深度 1% → 冲击可忽略                                     │
│  ├─ 主要关注: 费用 > 冲击                                                │
│  └─ 扩展到 $100K 时需要重新校准                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Fill Rate Logging (v10)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ★ FillQualityLogger [v10 新增]                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  记录字段:                                                               │
│  ├─ fill_ratio: 实际成交量 / 目标量                                      │
│  ├─ time_to_fill_s: 从下单到成交的时间 (秒)                             │
│  ├─ slippage_bps: 实际价格 vs 预期价格 (bps)                            │
│  ├─ was_repriced: 是否经过 reprice                                       │
│  ├─ order_type: limit / market / post_only                               │
│  ├─ final_action: filled / cancelled / timeout_market                   │
│  ├─ timing_mode: PA executor 使用的模式                                  │
│  └─ asset, direction, timestamp                                          │
│                                                                          │
│  输出: logs/fill_quality.jsonl                                           │
│  ├─ 每笔交易一行 JSON                                                   │
│  ├─ 周报手动 review                                                     │
│  └─ 不自动触发动作 (v10 lite 实现)                                       │
│                                                                          │
│  目标指标:                                                               │
│  ├─ Maker 成交率 > 70%                                                  │
│  ├─ 平均滑点 < 15 bps                                                   │
│  ├─ 限价单超时率 < 5%                                                   │
│  └─ 逆向选择率 < 2% (<500ms 成交)                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## DRL训练流程: 20 Stages, 35 Iron Laws

### ⚠️ 铁律: 不碰训练代码

```
DRL 训练是 20-stage pipeline, 35 条铁律。
obs_dim = 126 — 铁律, 不碰
ent_coef = 0.1 — 铁律, 不碰
训练代码独立于交易系统, 不在审计/修复范围内。
```

### 当前训练状态

```
✅ Stage 0-8A: 全部完成
   FiLM Position A (166K params) 锁定
   classic reward 锁定
   Optuna 50 trials 完成
   ent_coef=0.1 固定 (auto 模式不稳定)

🔄 Stage 8B+: 进行中
   Top 3 超参数 × 4-fold cross-validation
   RTX 5090, SubprocVecEnv (4 parallel)
   每次运行: 7-25 小时

📋 待完成:
   Stage 9-20 + Stage 21 (Meta-Learner)
```

### 20 Stage 路径图

```
┌──────────────────────────────────────────────────────────────┐
│  STAGE 0-6: 基础 ✅                                          │
│  数据准备 → 环境验证 → 基线 → 超参搜索 →                    │
│  架构搜索 → 特征消融 → Extractor 比较                        │
│  结果: FiLM Position A, obs_dim=126                          │
├──────────────────────────────────────────────────────────────┤
│  STAGE 7: 奖励模式选择 ✅                                     │
│  5 个模式测试 → classic 锁定                                  │
├──────────────────────────────────────────────────────────────┤
│  STAGE 8A: Optuna 超参优化 ✅                                 │
│  50 trials → ent_coef=0.1, buffer=1M, reward_clip=20        │
├──────────────────────────────────────────────────────────────┤
│  STAGE 8B: Top 3 完整训练 🔄                                  │
│  3 configs × 4 folds = 12 runs                               │
│  每 run: 7-25h on RTX 5090                                   │
├──────────────────────────────────────────────────────────────┤
│  STAGE 9: 真实摩擦环境                                       │
│  Kraken 费用 (16/26bps) + 滑点 + 延迟                       │
├──────────────────────────────────────────────────────────────┤
│  STAGE 10: TQC 完整训练 (摩擦感知)                            │
│  5M+ 步, 最优超参, 最优 reward                               │
├──────────────────────────────────────────────────────────────┤
│  STAGE 11: Decision Transformer V3.2                          │
│  从 TQC 最佳 episodes 离线学习                                │
├──────────────────────────────────────────────────────────────┤
│  STAGE 12: TQC + DT Ensemble                                 │
├──────────────────────────────────────────────────────────────┤
│  STAGE 13: 离线验证 (hold-out)                                │
│  Sharpe>1.0, Win>48%, DD<15%                                 │
├──────────────────────────────────────────────────────────────┤
│  STAGE 14: 运行时 Parity 检查                                 │
│  推理延迟 <10ms, 输入/输出形状验证                           │
├──────────────────────────────────────────────────────────────┤
│  STAGE 15: 模型部署                                           │
│  → DRL Authority: SHADOW                                     │
├──────────────────────────────────────────────────────────────┤
│  STAGE 16: Paper Trading (24h 干净运行)                       │
├──────────────────────────────────────────────────────────────┤
│  STAGE 17: Live — SHADOW (30天)                               │
│  StatisticalPromotionGate: Sharpe>1+Win>48%+无 drift        │
├──────────────────────────────────────────────────────────────┤
│  STAGE 18: Live — EXIT_ONLY                                   │
│  DRL 可以建议退出, 不能入场 (永久禁止)                       │
├──────────────────────────────────────────────────────────────┤
│  STAGE 19: 监控与维护 (持续)                                  │
│  Drift 检测 → SEVERE+ → 降级到 SHADOW                       │
├──────────────────────────────────────────────────────────────┤
│  STAGE 20: 在线学习 (增量更新)                                │
│  每周增量, A/B 测试新模型版本                                │
├──────────────────────────────────────────────────────────────┤
│  STAGE 21: Meta-Learner (计划中)                              │
│  多策略协调, 自适应权重                                       │
└──────────────────────────────────────────────────────────────┘
```

### 关键训练参数 (铁律)

```
算法:         TQC (Truncated Quantile Critics)
obs_dim:      126 (铁律 #1, 不碰)
ent_coef:     0.1 (铁律, auto 模式不稳定)
Extractor:    FiLM Position A (166K params)
VecFrameStack: 8 (时间上下文)
SubprocVecEnv: 4 parallel (比 DummyVecEnv 快 18-24×)
GPU:          RTX 5090
训练时间:     7-25h per run

Per-Asset GMM:
├─ BTC k=8
├─ ETH k=7
└─ SOL k=7

Cross-Validation: 4-fold
Reward Mode:  classic (纯 PnL)
Buffer Size:  1M
Reward Clip:  20
```

### DRL Authority 晋升

```
DISABLED → SHADOW (30天) → EXIT_ONLY → FULL

StatisticalPromotionGate:
├─ 30 天 shadow 干净运行
├─ Sharpe > 1.0
├─ Win rate > 48%
└─ 无 drift 检测 (DriftDetector SEVERE+)

ExistenceFuse (DRL 独立):
├─ DRL 5 笔连续亏损 → EXIT_ONLY 降级
└─ 与系统级 Fuse 独立计数

永久禁止: DRL 入场权限
├─ EXIT_ONLY 是最高实际权限
├─ FULL 模式理论存在但实践中不使用入场
└─ 架构决策: DRL 只帮助退出, 不帮助入场
```

---

## DRL 生产部署

### 部署拓扑

```
┌─────────────────────────────────────────────────────────────────────────┐
│  开发/训练环境                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  RTX 5090 + SubprocVecEnv(4)                                 │       │
│  │  DRL 训练 (Stage 0-15)                                       │       │
│  │  Optuna 超参优化                                             │       │
│  │  Walk-forward 验证                                           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                  ↓ (模型导出)                           │
│                                                                          │
│  Paper Trading 验证                                                      │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  实时 Kraken 数据, 模拟执行                                  │       │
│  │  DRL: SHADOW (记录决策, 无执行)                              │       │
│  │  验证: 24h 无崩溃 + 所有 4H ticks 执行                      │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                  ↓ (验证通过)                           │
│                                                                          │
│  生产环境 (Linux, systemd)                                               │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  /opt/hmats/           — 应用代码                            │       │
│  │  /opt/hmats/models/    — DRL 模型检查点                      │       │
│  │  /var/log/hmats/       — 日志 (events/, proof/, fill_quality)│       │
│  │  /var/lib/hmats/       — 状态持久化 (SQLite)                │       │
│  │  /etc/hmats/           — 配置 + API keys (600 perms)        │       │
│  │                                                               │       │
│  │  systemd 服务:                                               │       │
│  │  ├─ User: hmats (专用, 无 shell 登录)                        │       │
│  │  ├─ Restart: on-failure, max 5/5min                          │       │
│  │  ├─ 安全: NoNewPrivileges, ProtectSystem=strict             │       │
│  │  └─ Mode: --mode live --confirm-live                        │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

### DRL 推理要求

```
推理延迟: < 10ms (目标)
├─ 不需要 GPU (推理用 CPU)
├─ FiLM Position A: 166K params (轻量)
└─ VecFrameStack(8): 内存中维护

OOD Detector:
├─ Mahalanobis distance
├─ 超出分布 → DRL 信号标记为低置信度
└─ 不自动降级 (仅标记)
```

---

## 模型版本管理

```
/opt/hmats/models/
├── current/                    # 活跃模型
│   ├── tqc_btc.zip
│   ├── tqc_eth.zip
│   └── tqc_sol.zip
│
├── backup/                     # 最后稳定
│   └── last_stable/
│
├── archive/                    # 历史版本
│   └── YYYY-MM/
│       ├── tqc_*.zip
│       └── metadata.json
│
└── shadow/                     # Shadow 模式测试
    ├── tqc_*_candidate.zip
    └── performance_log.json

部署流程:
1. 训练完成 → 导出到 shadow/
2. SHADOW 30 天验证
3. StatisticalPromotionGate 通过
4. backup/ ← current/ (备份)
5. current/ ← shadow/ (部署)
6. [P190] docker compose -f docker-compose.hetzner.yml restart hmats-engine
   (原文是 systemd 命令 — 线上不是 systemd 部署，见部署指南第 7 节)
7. 监控 docker logs -f hmats-engine
```

---

### 执行层 vs DRL 关系总结

```
┌─────────────────────────────────────────────────────────────┐
│  执行层: 独立于 DRL, 即使 DRL=DISABLED 也正常工作          │
│                                                              │
│  DRL DISABLED:                                              │
│  └─ PA Executor 使用 TimingEngine 独立决策                  │
│                                                              │
│  DRL SHADOW:                                                │
│  └─ DRL 记录信号到 shadow ledger, 不影响执行                │
│                                                              │
│  DRL EXIT_ONLY:                                             │
│  └─ DRL 可以建议退出 → 影响 PA Executor 的 urgency         │
│     但不能建议入场 (永久禁止)                               │
│                                                              │
│  关键: 执行层是 DRL 的下游, 不是上游                        │
│  DRL 影响 what/when, PA Executor 决定 how                   │
└─────────────────────────────────────────────────────────────┘
```

---

**文档第4部分结束**

继续阅读：
- Part 5: 运维与附录

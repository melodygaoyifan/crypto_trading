# HMATS E2E Training Guide — 优化补丁
# 基于 2026-02-22 ~ 2026-02-25 对话全回溯
# 应用于: HMATS_E2E_TRAINING_GUIDE.md (1733 lines, Stage 9.7 版本)

---

## 一、需要修正的不一致项

### 1.1 V8 Prompt 硬件部分 (line ~74)

```
❌ 旧: 128GB RAM → buffer_size=2M 全模型通用（含时序模型 ~15GB buffer）
✅ 新: 128GB RAM → buffer_size=500K (n_stack=8 内存放大: 500K×8×126×4bytes×2 ≈ 3.75GB)
```

**原因**: Stage 9.7 segfault 确认 1M buffer + n_stack=8 需要 ~7.5GB 连续内存分配，Windows 内存碎片化导致失败。

### 1.2 SOL GMM k 值 (V8 Prompt 部分引用)

```
❌ V8 某些地方写: SOL k=8
✅ 正确值: SOL k=7
```

E2E guide preflight 写的 "k values: BTC=8, ETH=7, SOL=7" 是正确的。V8 prompt 的错误不影响训练（regime_dim 固定=8，zero-padded），但文档应统一。

### 1.3 Phase A/B 历史记录中的 buffer=1M

以下位置保留 1M 作为**历史记录**，不改：
- Phase A Trial #5 参数 (buffer_size: 1,000,000) — 这是 Optuna 搜到的值
- Phase B Config 1 命令行 (--buffer-size 1000000) — 这是已执行的命令
- Phase A/B 对比表中的数值

唯一需要确保的是 **FINAL_CONFIG** 和 **Stage 10 执行命令** 用 500K。当前版本已正确。

---

## 二、已在 Guide 中更新（验证通过）

以下改动已在上一轮对话中完成，验证无遗漏：

| 改动 | 位置 | 状态 |
|------|------|------|
| Stage 9.5 结果 bake in (+631%/+318%) | Step 7 | ✅ |
| Stage 9.7 Go/No-Go ✅ GO | Step 8 | ✅ |
| FINAL_CONFIG buffer_size=500K | Step 4 / Step 9 | ✅ |
| Fold 间清理代码 + RSS assert | Step 9.2 | ✅ |
| Preflight 新增 3 项检查 | Step 9.1 | ✅ |
| 时间预估 96-224h | Step 9.3 + 路径图 | ✅ |
| 状态快照更新到 Stage 10 | 文档头部 | ✅ |
| 训练 vs Runtime 数据源表 | 新增部分 | ✅ |

---

## 三、新增内容（尚未写入 Guide）

### 3.1 Step 18: Sentiment Signal Wiring (Stage 19, Post-deployment)

**位置**: 在 Step 17 (Live Deployment) 之后新增

```markdown
## Step 18: Sentiment Signal A/B (Stage 19) [~1 week 对比]

**前置**: Paper Run 稳定运行 ≥ 1 week（积累 baseline 数据）

### 18.1 SimpleSentimentCalculator

6 个信号，加权合成，纯确定性（无 LLM，无额外 API key）：

| 信号 | 权重 | 数据源 | 短偏好解读 |
|------|------|--------|-----------|
| Funding Rate | 25% | Kraken Futures | 正=多头拥挤→做空; 负=空头拥挤→谨慎 |
| Long/Short Ratio | 20% | Coinglass | >1.5=多头过多→做空确认; <0.7=减仓 |
| Fear & Greed Index | 15% | alternative.me | >75贪婪→逆向做空; <25恐惧→中性 |
| OI Change | 15% | Kraken Futures | OI↑+价格↑=趋势强→做空; OI↓=观望 |
| Liquidations | 15% | Coinglass | 多头爆仓>空头=级联做空机会 |
| DVOL + VPIN | 10% | Pipeline | 高vol/毒性→降低全部方向信心 |

**输出**: composite_score (-1~+1), direction, strength (0~1), crowding_score (0~1)

### 18.2 注入决策循环

注入点: Quant signal → VolAlpha → ★ Sentiment ★ → Risk check

**三个效应**:
- [SENT-EFFECT-1] 方向对齐: bearish + short → +10%; bullish + short → -5% (NOT veto)
- [SENT-EFFECT-2] 拥挤: crowding > 0.7 → urgency ×0.80, size ×0.70
- [SENT-EFFECT-3] 低强度: strength < 0.2 → urgency ×0.85

**⚠️ 绝对红线**:
- Sentiment 绝不 veto 交易
- Sentiment 绝不翻转方向
- Sentiment 是调节器，不是决策器

### 18.3 A/B 验证

```
Week 1: 裸跑 baseline（已有 Paper Run 数据）
Week 2: 开启 Sentiment，记录每笔 [SENT] 调节
Week 3: 对比 Total PnL, Win Rate, Avg Trade Size, Max DD
```

判定: PnL 改善 > 5% → 保留; 无改善或退化 → enabled=False

### 18.4 Gate
- [ ] SimpleSentimentCalculator 编码完成
- [ ] 6 个数据源可用（Kraken Futures / Coinglass / alternative.me / Pipeline）
- [ ] main.py 注入点正确（signal_strength 之后, risk check 之前）
- [ ] 3 week A/B 对比完成
- [ ] PnL 改善 > 5% 或关闭
```

### 3.2 Step 19: Online DRL Framework (Stage 20, Continuous)

**位置**: 在 Step 18 之后新增

```markdown
## Step 19: Online DRL (Stage 20) [持续]

**前置**: 100+ live experiences 积累（通常 ~2-4 weeks live trading）

### 19.1 架构

```
LIVE TRADING (每笔交易)
    ↓ record_decision() + record_outcome()
LiveExperienceBuffer (circular, 10K/asset, JSONL 持久化)
    ↓ 每 24H 或每 50 experiences (先到者)
PeriodicFinetuner (50 steps, lr=1e-5, grad_clip=0.5)
    ↓ 只更新 SHADOW model
ShadowModelValidator (50 bars A/B, ~8天)
    ↓ shadow_sharpe > prod_sharpe + 0.1 AND p < 0.05
ModelPromoter → swap shadow → production
```

### 19.2 安全规则

**⚠️ 绝对红线**:
1. PeriodicFinetuner 永不写入 production model path
2. Promote 需 3 条件: Sharpe > +0.1, Max DD 增加 < 2%, p < 0.05
3. Fine-tune lr = 1e-5 (原训练 1.53e-5 的 ~65%)
4. Gradient norm clipping = 0.5
5. Min 100 experiences 才触发
6. Buffer 硬上限 10K/asset (circular)
7. 系统 restart 后从 JSONL 恢复

### 19.3 Paper Run 期间: 被动收集

Paper Run 启动前嵌入 main.py:
- LiveExperienceBuffer enabled=True, finetuner 不初始化
- 每次 decision → record_decision()
- Position close → record_outcome()
- 只写 JSONL，零决策影响

目的: Paper Run + Live Week 1-2 默默积累数据，Stage 20 启用时已有数百条可用。

### 19.4 关键文件

```
drl/live_experience_buffer.py          (~200 lines)
training/online_finetuner.py           (~150 lines)
training/online_model_promoter.py      (~100 lines)
```

### 19.5 Gate
- [ ] 3 个文件编码完成
- [ ] main.py 被动收集嵌入完成
- [ ] Paper Run 期间无决策影响验证
- [ ] 100+ experiences 积累
- [ ] 首次 fine-tune shadow 完成
- [ ] 50-bar A/B 通过 3 条件
- [ ] 首次 promotion 成功
```

### 3.3 SOTA Gap Validation 参考 (新增附录)

**位置**: 风险和回退计划之后，作为附录

```markdown
# ═══════════════════════════════════════════════════════
# 附录 A: SOTA Gap Validation (2026-02-25)
# ═══════════════════════════════════════════════════════

7 个 SOTA 领域逐项验证，结论: 无阻塞性缺口。

| # | SOTA 领域 | 覆盖状态 | 不做原因 |
|---|-----------|---------|---------|
| 1 | Reward 工程 (Self-Rewarding DRL) | ✅ reward_clip=20 + dd_penalty=2.82 | Stage 7: 任何降低 PnL 权重(1.0→0.4)都拖慢学习 |
| 2 | Regime Detection HMM+DRL | ✅ per-asset GMM + FiLM PosA γ·h+β | TQC 无 ε-greedy, gamma=0.99 不敏感 |
| 3 | Meta-Strategy Selection | ⏳ Stage 21 可选 | TQC+DT 不同范式 ensemble 更好 |
| 4 | Funding Rate Alpha | ✅ obs 内 + Runtime Sentiment | 4H 决策频率下小时级 funding 信息量有限 |
| 5 | 预测-决策两阶段 | ❌ 不做 | Phase B +715.5% 证明 TQC 收敛良好 |
| 6 | 训练稳定性 (PER/FinRL/Turbulence) | ✅ TQC 分布 + GMM + OOD detector | PER +30% 采样开销，CPU-bound 不划算 |
| 7 | 多分辨率 CNN + 链上数据 | ✅ VecFrameStack(8) + TA 指标 | CNN 参数量远超 500K 铁律 |

**关键不一致修正**: SOL GMM k=7 (非 k=8)。E2E guide preflight 正确，V8 prompt 需修正。

**提醒**: 到 Stage 21 做 meta-learner 时，先用 regime 标签做分层评估。
如果 TQC 和 DT 在某些 regime 下表现差异不显著，动态权重增益可能有限。
```

---

## 四、路径图和快速参考更新

### 4.1 路径图更新

当前版本路径图指向 Stage 10，需在 Stage 17 后补充 18-19:

```
Stage 17 Live Deployment                              [Week 1+]
    ↓
Stage 18 Sentiment A/B                                [~3 weeks]    ← 新增
    ↓
Stage 19 Online DRL                                   [持续]        ← 新增
    ↓
Stage 20 监控 + Drift Detection                       [持续]
    ↓
Stage 21 Meta-Learner (可选, TQC+DT 分层评估后决定)    [如果需要]    ← 新增
```

### 4.2 快速参考更新

```
现在 — Stage 10 Full Training:
  □ 更新 config/optuna_winner.json: buffer_size=500000
  □ 确认 train_drl_full.py 有 fold 间清理 (del model + gc.collect)
  □ 跑 Preflight checklist (Step 9.1)
  □ 启动 BTC 3 folds × 2.5M (~60-70h, early stopping 可能 ~30-45h)

然后:
  □ ETH 3 folds × 2.5M + SOL 3 folds × 3.0M (~132-154h)
  □ Step 10-13: Ensemble + Validation (5h)
  □ Step 14-15: Deploy + Paper Run (48h, 含 Experience Buffer 被动收集)
  □ Step 16: Live Week 1-2 (Conservative → Standard)
  □ Step 17: 持续监控
  □ Step 18: Sentiment A/B (Paper Run 后 ~3 week 对比)      ← 新增
  □ Step 19: Online DRL (100+ experiences 后, 持续)          ← 新增
```

---

## 五、铁律更新

### 5.1 现有铁律修正

```
❌ 旧 #6: buffer_size = 1M (optimal from Optuna)
✅ 新 #6: buffer_size = 500K (Optuna 选 1M, 但 n_stack=8 内存放大: 1M→7.5GB segfault, 修正为 500K)
```

### 5.2 新增铁律 (#33-35)

```
#33 Fold 间清理: 每个 fold 结束后 del model + gc.collect() + torch.cuda.empty_cache(),
    下一个 fold 启动前 assert RSS < 4.0 GB
    (原因: 两个 buffer 短暂共存 = 3.75GB × 2 = 7.5GB 瞬时峰值)

#34 Sentiment 只调不决: Sentiment 绝不 veto/翻转交易，最大调节幅度 ±10%/±30%
    (原因: PROFIT-MAX 原则, 调节器不是决策器)

#35 Online DRL 只 shadow: PeriodicFinetuner 永不写入 production path,
    promote 需 Sharpe>+0.1 + DD<+2% + p<0.05 三条件全通过
    (原因: 生产安全, 避免灾难性遗忘)
```

---

## 六、训练 vs Runtime 数据源表 (已添加，此处做完整性参考)

```
进 DRL obs (122 features, 训练可见):
  Kraken OHLCV + TA:           102 features
  GMM per-asset:               8 regime_proba (BTC k=8, ETH k=7, SOL k=7)
  Coinglass (衍生品):          6 (funding_zscore, oi_change, liq_imbalance, ...)
  Coinglass (flag):            1 has_external_data
  Wavelet 去噪:               5 *_denoised
  Env state:                   4 (position_*, pnl, dd)
  Total: 126 obs_dim

只在 Runtime 消费 (不进 obs, 改动不需重训):
  Funding Rate (raw)     → Short Filter, Sentiment
  OI/OI Change (raw)     → Short Bias, Sentiment
  Liquidations (raw)     → Cascade Governor, Sentiment
  Long/Short Ratio       → Crowding Score, Sentiment
  Whale Flow             → Alpha Engine
  Active Addresses       → Alpha Engine
  DEX Volume / TVL       → Alpha Engine
  Exchange Inflow (SOL)  → SOL Defense
  Fear & Greed Index     → Sentiment
  DVOL z-score           → Sentiment, Risk
  VPIN                   → Sentiment, Toxicity
```

**关键**: 只有 obs 内的 122 features 变更才需要重训 (改 obs_dim → 全部模型作废)。
Runtime 数据源改动只影响 veto/sizing/urgency/sentiment，零训练成本。

---

## 七、应用清单

按顺序执行：

```
[ ] 1. config/optuna_winner.json: 确认 buffer_size=500000
[ ] 2. train_drl_full.py: 确认 fold 间清理代码存在
[ ] 3. V8 Prompt 硬件部分: buffer_size=2M → 500K (备注)
[ ] 4. V8 Prompt SOL GMM: 确认 k=7 (非 k=8)
[ ] 5. E2E Guide: 铁律 #6 修正, 新增 #33-35
[ ] 6. E2E Guide: Step 18 (Sentiment) 写入
[ ] 7. E2E Guide: Step 19 (Online DRL) 写入
[ ] 8. E2E Guide: 附录 A (SOTA Gap) 写入
[ ] 9. E2E Guide: 路径图和快速参考补充 Stage 18-21
[ ] 10. 启动 Stage 10 BTC training
```

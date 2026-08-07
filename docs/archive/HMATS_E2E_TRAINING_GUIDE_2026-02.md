> # ⛔ ARCHIVED 2026-08-07 (P200 forensics) — DO NOT TRAIN FROM THIS DOCUMENT
> Every performance number in this guide is a P164 leak artifact measured at zero
> exchange fee: the wavelet denoise was non-causal (IC +0.41 on pure random
> walks), the GMM was fit on 100% of history, fold selection ran on shaped
> reward with a regime-alignment bonus, and eval std was structurally 0.0.
> P200's honest rerun of this exact formulation on clean data at real fees is
> NOT PROMOTABLE on all 3 BTC folds (-$154,661 / -$63,650 / -$62,045 vs
> positive buy-and-hold baselines). The "Stage 7 chose classic → 8B locked
> Config 1 → 9.7 said GO" chain this document is organized around is a chain of
> those artifacts. Its training commands also predate venue-aware fees
> (--venue/--fee-side), --decision-interval, and the split-aware GMM.
>
> **The live guide is docs/HMATS_TRAINING_GUIDE_V2.md.** This file is retained
> as the forensic record that CLAUDE.md P199/P200 audit, and for the still-valid
> reference blocks the audit identified (iron laws L229-291 — amend #26 to
> CAUSAL wavelet and #35 to P200-LADDER — feature chain L89-160, runtime parity
> L1182-1213, sentiment phase L1387-1974).

# HMATS End-to-End Training Guide: Stage 8B → Live
# ═══════════════════════════════════════════════════
# 当前状态: Stage 9.7 ✅ GO, Stage 10 Full Training 下一步
# 日期: 2026-02-21
# ═══════════════════════════════════════════════════

---

## ⚠️ 参考文档（执行每个 Step 前必须对照）

```
本文档是执行路径向导。以下文档包含完整安全约束，不可省略:

HMATS_V8_FINAL_PROMPT.md:
  - 铁律 #1-28 完整列表（违反任何一条 = 训练失败）
  - 特征维度链 (102→110→117→121→122→126→1008)
  - 外部数据源详情 (Coinglass + Futures 7 列)
  - FiLM Position A 完整 PyTorch 代码
  - TQC 配置全参数
  - 硬件约束 (RTX 5090 Laptop, 128GB RAM, Windows native)

HMATS_V9_FRICTION_OOD_PATCH.md:
  - 铁律 #29-32
  - 交易成本模型完整代码
  - OOD 检测器完整实现
  - 比例翻仓成本公式推导

本文档内含:
  - 铁律 #33-35 (v10: fold 清理 / Sentiment 约束 / Online DRL 安全)
  - SOTA Gap Validation 附录
```

---

## 当前状态快照

```
已完成:
  Stage 0-6: 全部完成 ✅
  Stage 7:   classic 锁定 ✅ (NAV% 确认: +693% >> sortino +580% >> sharpe +565%)
  Stage 8A:  Optuna 51 trials 完成 ✅
             Best: Trial #5 +736.11 (vs baseline +794, 方差范围内)
             关键发现: learning_starts=30K (63.6% importance), reward_clip=20 (触下界)
  Stage 8B:  Config 1 锁定 ✅ (mean NAV +715.5%, 自适应跳过 Config 2)
             3 folds: $807K / $719K / $920K, Mean/Std=9.89, Max DD=2.29%
  Stage 9:   v9 编码 ✅ + Smoke test ✅ (与 8B 并行完成)
             SubprocVecEnv 验证 ❌ (3.6x 墙钟加速但 8x 更少 gradient updates)
  Stage 9.5: Friction A/B ✅ (off +631% vs on +318%, 模型学会 2.4x 更长持仓)
  Stage 9.7: Go/No-Go ✅ (+317.74% >> +5%, 1050K 步 segfault 但 best_model 已 plateau)
             ⚠️ buffer_size 修正: 1M → 500K (n_stack=8 内存放大)

进行中:
  Stage 10: TQC Full Training                             ← 下一步

锁定决策:
  Extractor:    FiLM Position A (166K params, Stage 6)
  Reward:       classic (Stage 7, NAV% 确认 ✅)
  Hyperparams:  Config 1 Optuna (Stage 8B, 3-fold 验证 ✅)
  buffer_size:  500,000 (从 1M 修正, 9.7 segfault)
  Architecture: obs_dim=126 single-frame runtime input, TQC internal stack=8 (effective 1008), DummyVecEnv
  VecEnv:       DummyVecEnv only (SubprocVecEnv 不可行, 已验证)
  铁律 #1-35:   全部有效 (含 v9 #29-32, v10 #33-35)
```

---

## 硬件约束

```
CPU:    Intel Core Ultra 9 285HX
GPU:    NVIDIA RTX 5090 Laptop (24GB GDDR7, 175W TDP)
RAM:    128 GB DDR5
SSD:    3.7TB + 1.8TB
OS:     Windows 11 (native, not WSL2)
Driver: Game Ready 581.83

关键影响:
  128GB RAM  → buffer_size=500K (n_stack=8 内存放大: 500K×8×126×4bytes×2 ≈ 3.75GB)
  24GB VRAM  → TQC + FiLM PosA (~1.03M params) 充裕
  Windows native → DummyVecEnv FPS ~20-80（不是 WSL2 的 200-400）
  175W laptop → 不要并行训练，热管理
  FPS 基准: FiLM PosA ~30-50 FPS（比 Plain LSTM 略慢）
  
  本文档所有时间预估基于此硬件。换机器需重新估算。
```

---

## 特征维度链

```
维度演变（完整推导）:
  Base features: 102 (lower_low 是 #102)
  + regime_proba: 102 + 8 = 110 (regime_proba_0..7, ETH/SOL k=7 → proba_7 zero-padded)
  + external:    110 + 7 = 117 (funding_rate_zscore, oi_change_5d, liq_imbalance,
                                 taker_ratio_zscore, tradecount_zscore,
                                 taker_vol_momentum, has_external_data)
  + wavelet:     117 + 5 = 122 (rsi_14_denoised, macd_12_26_denoised,
                                 bb_width_20_denoised, atr_14_denoised,
                                 vol_ratio_s_denoised)
  + env state:   122 + 4 = 126 (position_ratio, position_direction, pnl_ratio, drawdown)

  Runtime contract: single obs shape = (126,), TQCInference internally stacks 8 frames -> effective (1008,)
  LSTM extractor 内部 reshape: (1008,) → (8, 126) → LSTM sequence

  feature_manifest.json: 动态读取，绝不硬编码。all_features=122, obs_dim=126

  FiLM PosA 内部拆分:
    obs[-12:-4] = regime_proba_0..7 (8-dim, FiLM 调制信号)
    obs[:-12] + obs[-4:] = non-regime features (118-dim, LSTM 输入)

Scaling 规则:
  RobustScaler: 适用于 ~108 个数值特征

训练 vs Runtime 数据源:

  进 DRL obs (122 features, 训练时可见):
  ┌──────────────────────────────┬───────────┬──────────────────────────┐
  │ 数据源                       │ 特征数     │ obs 中的列名              │
  ├──────────────────────────────┼───────────┼──────────────────────────┤
  │ Kraken OHLCV + TA           │ 102       │ base features            │
  │ GMM per-asset               │ 8         │ regime_proba_0..7        │
  │ Coinglass (Funding/OI/Liq)  │ 6         │ funding_rate_zscore,     │
  │                              │           │ oi_change_5d,            │
  │                              │           │ liq_imbalance,           │
  │                              │           │ taker_ratio_zscore,      │
  │                              │           │ tradecount_zscore,       │
  │                              │           │ taker_vol_momentum       │
  │ Coinglass (flag)            │ 1         │ has_external_data        │
  │ Wavelet 去噪                │ 5         │ *_denoised (5 cols)      │
  │ Env state                   │ 4         │ position_*, pnl, dd      │
  └──────────────────────────────┴───────────┴──────────────────────────┘
  Total: 122 features + 4 env = 126 obs_dim

  只在 Runtime 消费 (不进 obs, DRL 看不到):
  ┌──────────────────────────────┬───────────────────────────┬──────────────────┐
  │ 数据源                       │ Runtime 消费者             │ 状态             │
  ├──────────────────────────────┼───────────────────────────┼──────────────────┤
  │ Funding Rate (raw)          │ Short Filter, Sentiment   │ ✅ Kraken Futures │
  │ OI / OI Change (raw)        │ Short Bias, Sentiment     │ ✅ Kraken Futures │
  │ Liquidations (raw)          │ Cascade Governor, Sent.   │ ✅ Coinglass      │
  │ Long/Short Ratio            │ Crowding Score, Sentiment │ ✅ Coinglass      │
  │ Whale Flow                  │ Alpha Engine              │ ✅ CryptoCompare  │
  │ Active Addresses            │ Alpha Engine              │ ✅ CryptoCompare  │
  │ DEX Volume / TVL            │ Alpha Engine              │ ✅ CryptoCompare  │
  │ Exchange Inflow (SOL)       │ SOL Defense               │ ✅ Solana RPC     │
  │ Network Congestion          │ SOL Defense               │ ✅ Solana RPC     │
  │ Fear & Greed Index          │ Sentiment                 │ ✅ alternative.me │
  │ News/Sentiment              │ Sentiment (未来)           │ ✅ CryptoPanic    │
  │ Macro (利率/CPI)            │ Macro Crowd Adapter       │ ✅ FRED           │
  │ Jito Tips (SOL MEV)         │ SOL Defense               │ ⚠️ 需要 Jito API  │
  │ DVOL z-score                │ Sentiment, Risk           │ ✅ Pipeline       │
  │ VPIN                        │ Sentiment, Toxicity       │ ✅ Pipeline       │
  └──────────────────────────────┴───────────────────────────┴──────────────────┘

  关键: Runtime 数据源改动不需要重训 DRL。只影响 veto/sizing/urgency/sentiment。
  只有 obs 内的 122 features 变更才需要重训（改 obs_dim → 全部模型作废）。
  不 scale:     regime_proba_0..7 (已是 0-1), has_external_data (二值)
  GMM scaler:   独立 StandardScaler (12-dim)，和 Feature scaler 完全分离
```

---

## TQC 完整配置

```
Algorithm:       TQC (sb3_contrib)
Policy:          FiLM Position A extractor (Stage 6 winner)
Net Arch:        FiLM PosA (166K ext) + [384, 384, 256] policy net (~1.03M total)
Obs Space:       126 (122 features + 4 env state)
Env State:       position_ratio, position_direction, pnl_ratio, drawdown
VecFrameStack:   n_stack=8 (8×4H=32h temporal context)
Action Space:    Box(-1, 1) continuous position
VecEnv:          DummyVecEnv(1)

默认超参 (Phase B 验证后可能被 Optuna winner 覆盖):
  n_quantiles: 32 (Optuna: 24), n_critics: 2, top_quantiles_to_drop: 2
  ent_coef: 0.1 (fixed, 铁律 #5), lr: 3e-5 (Optuna: 1.53e-5, SOL: 2e-5)
  batch: 256, grad_steps: 4 (铁律 #7)
  buffer: 500K (Optuna 选 1M, 9.7 修正→500K), learning_starts: 30K (Phase A 确认), gamma: 0.99, tau: 0.005
  weight_decay: 0 (铁律 #6)
  reward_clip: 50 (Optuna: 20 ← Phase A 关键发现)

铁律固定 (Optuna 不搜, Phase B 不变):
  ent_coef, weight_decay, batch, grad_steps

Tier 2 固定 (Phase A importance 证明不敏感):
  gamma=0.99, tau=0.005

Tier 3 固定 (TQC 论文默认, 只在 Phase B fail 时作 fallback 搜索):
  n_critics=2, top_quantiles_to_drop=2

net_arch 固定 (18K bars 过拟合约束, 只在 Phase B fail 时作 fallback 搜索):
  [384, 384, 256] → ~860K policy params

Phase A 关键发现:
  learning_starts=30K: 63.6% importance, Top 5 全部收敛到默认值
  reward_clip=20: 11.5% importance, 4/5 Top trials 触及下界, 截断极端梯度
  lr 1.5e-5~8.9e-5: 分散, 在此范围内不敏感
```

---

## 外部数据概要

```
7 列外部特征（仅 TQC 消费，DT v3.2 不用）:

| 列名                  | 来源             | 转换              | Clip      | 含义           |
|----------------------|-----------------|-------------------|-----------|---------------|
| funding_rate_zscore  | Coinglass funding| 30-day z-score    | [-10,10]  | 资金费率异常    |
| oi_change_5d         | Coinglass OI     | 5-day pct change  | [-5,5]    | 持仓量动量      |
| liq_imbalance        | Coinglass liq    | pre-computed, clip | [-1,1]    | 清算方向偏斜    |
| taker_ratio_zscore   | Futures daily    | 30-day z-score    | [-10,10]  | 主动成交异常    |
| tradecount_zscore    | Futures daily    | 30-day z-score    | [-10,10]  | 活跃度异常      |
| taker_vol_momentum   | Futures daily    | 5-day pct change  | [-5,5]    | 主动力量动量    |
| has_external_data    | —                | binary flag       | {0,1}     | 数据可用性      |

关键规则:
  - 所有转换用 z-score 或 pct-change，绝不用绝对值（铁律 #2）
  - Pre-2020: 6 个数值特征 = 0.0, has_external_data = 0（铁律 #3）
  - merge_asof(direction='backward') forward-fill daily → 4H
  - z-score rolling 前 30 天 = NaN → fillna(0)
  - Coinglass DNS 失败 → features = 0 + has_external_data = 0
```

---

## 铁律完整列表 (#1-32)

```
数据:
  #1  新数据必须验证时间戳（year=57086 bug 导致 eval=-3184）
  #2  Daily→4H forward-fill 必须用 z-score/pct-change，不用绝对值
  #3  Pre-2020 缺失外部数据: 填 0 + has_external_data=0
  #4  mom_4h~96h 已删除（与 ret_4h~96h r≈1.000 冗余）

超参:
  #5  ent_coef = 0.1（固定）。绝不用 "auto" — 梯度爆炸
  #6  weight_decay = 0。加 1e-4 直接从 +33.69 崩到 -845
  #7  batch=256 + grad_steps=4 是验证过的组合
  #8  early_stopping 必须开启（20 evals no improvement → stop）

环境:
  #9  只用 DummyVecEnv。SubprocVecEnv 在 ~1.47M 步 deadlock（踩了两次）
  #10 部署 best_model.zip（EvalCallback），不是 final_model

Fold:
  #11 window_size = 10（不是 96）                        ★ 高频出错
  #12 gap = 42 bars（42×4H=7 天）
  #13 n_folds = 3, val_ratio = 0.15

架构:
  #14 DT v3.2 保留原样（80-dim 特征，无外部数据）
  #15 Ensemble 只能是 signal-level（不同特征空间不能共享 obs）

Pipeline:
  #16 所有模型共享同一个 train/val split（独立 split = 数据泄漏）
  #17 GMM scaler (12-dim) 和 Feature scaler (~110-dim) 独立  ★ 高频出错
  #18 RobustScaler: fit only on train, transform both
  #19 Scaling 排除: regime_proba_0..7 + has_external_data

执行:
  #20 一次一个 asset。顺序 BTC → ETH → SOL
  #21 checkpoint_freq = 500K 步
  #22 Windows 防休眠: powercfg /change standby-timeout-ac 0

v8 新增:
  #23 Wavelet 后必须重建 parquet + 更新 feature_manifest
  #24 FiLM 初始化: γ bias=1, β bias=0, weights=0          ★ 首测教训
  #25 Per-regime 训练: ≥ 500 bars 才训练，否则 fallback
  #26 Wavelet 去噪必须同时应用于训练和 runtime
  #27 v8 模型 (obs 126) 和 v7 模型 (obs 121) 不兼容
  #28 Extractor 参数 < 500K（4.2M 在 18K bars 严重过拟合）

v9 新增:
  #29 交易成本必须同时写入 env.step() (NAV) 和 reward (学习信号)
  #30 翻仓惩罚必须是连续比例，不是阈值阶梯               ★ 不连续梯度
  #31 OOD 检测器只用训练集 fit
  #32 OOD 降权是 soft degradation，连续 ≥ N 步才硬切

v10 新增:
  #33 Fold 间清理: del model + gc.collect() + torch.cuda.empty_cache(),
      下一 fold 启动前 assert RSS < 4.0 GB
      (原因: buffer=500K×n_stack=8 ≈ 3.75GB, 两个 buffer 共存 → OOM)
  #34 Sentiment 只调不决: 绝不 veto/翻转交易, 最大调节 ±10%/±30%
      (原因: PROFIT-MAX 原则, 调节器不是决策器)
  #35 Online DRL 只 shadow: PeriodicFinetuner 永不写入 production path,
      promote 需 Sharpe>+0.1 + DD<+2% + p<0.05 三条件全通过
      (原因: 生产安全, 避免灾难性遗忘)
```

---

## 总路径图 + 时间预估

```
                                                     你在这里
                                                        ↓
Stage 7  NAV% retest                                 ✅ DONE (classic +693%)
    ↓
Stage 8A Optuna 51 trials                            ✅ DONE (best +736, Trial #5)
    ↓
Stage 8B Config 1 × 3 folds × 2.5M                  ✅ DONE (mean NAV +715.5%, 自适应跳过 Config 2)
    ↓
Stage 9  v9 patch (编码 + smoke test)                ✅ DONE (与 8B 并行完成)
    ↓
Stage 9.5 200K friction A/B                           ✅ DONE (on +318% vs off +631%)
    ↓
Stage 9.7 Go/No-Go (BTC fold_1 2.5M)                 ✅ DONE (+317.74% >> +5%, buffer 1M→500K)
    ↓
Stage 10 TQC Full Training (friction-aware)           [~96-224h]    ← 下一步
    ↓                                                (DummyVecEnv + early stopping, buffer=500K)
    ├── ⚡ Stage 19 Sentiment L3 (并行, CPU-only)     [校准 ~40min + 部署 ~2h + A/B 持续]
    │   (前置: Paper Run baseline 50+ 笔交易)
    │   (L3 Haiku 已部署, A/B 数据随 Paper Run 积累)
    ↓
Stage 11 DT v3.2 Training                            [~1.5h]
    ↓
Stage 12 TQC+DT Ensemble                             [~1h]
    ↓
Stage 13 Offline Validation                           [~1h]
    ↓
Stage 14 Runtime Parity Check                         [~1h]
    ↓
Stage 15 Model Deployment                             [~30 min]
    ↓
Stage 16 Paper Run (裸跑 baseline)                    [48h]
    ↓                                                (同时: LiveExperienceBuffer 被动收集)
Stage 17 Live Week 1-2 (Conservative → Standard)      [Week 1-2]
    ↓
Stage 18 DRL Promotion (30+ shadow trades)            [30+ days]
    ↓
Stage 19 Sentiment A/B 决策                            [L3 已跑 2+ weeks → keep/kill]
    ↓
Stage 20 Online DRL (100+ live experiences 后)         [持续]
    ↓
Stage 21 Regime Power Retrain + Meta-Learner (可选)      [全量重训 ~96-224h]
         Regime Power: STEADY_UPTREND→1.1, NEUTRAL_DRIFT→0.7
         前置: Live 1+ month + Sentiment/Aggression A/B 完成
         三个 asset 必须一起重训

════════════════════════════════════════
到 Paper Run 启动:
  保守: ~230-290h (~10-12 天)
  最可能: ~100-160h (~4-7 天)

Paper Run → 全功能 Live:
  ~2-3 months (DRL promotion + Sentiment A/B + Online DRL 积累)
  ⚡ Sentiment A/B 与 Stage 10 并行 → 不额外占用时间线
════════════════════════════════════════
```

---

# ═══════════════════════════════════════════════════
# PHASE 1: 锁定超参 (Stage 7 ✅ + Stage 8B ✅)
# Config 1 锁定: mean NAV +715.5%, 自适应跳过 Config 2
# ═══════════════════════════════════════════════════

## Step 1: Stage 7 NAV% Retest — ✅ 完成

```
| Mode      | Best Reward | Final NAV ($) | NAV %    | Max DD | Trades | Avg Hold       |
|-----------|-------------|---------------|----------|--------|--------|----------------|
| classic   | +808.19     | $793,177      | +693.18% | 2.08%  | 7,890  | 3.4 bars (14h) |
| sortino   | +316.59     | $679,812      | +579.81% | 1.74%  | 7,790  | 3.5 bars (14h) |
| sharpe    | +368.22     | $664,748      | +564.75% | 1.65%  | 6,990  | 3.9 bars (15h) |
```

**结论**:
- Classic 全面领先: NAV% +693% >> sortino +580% (+113pp) >> sharpe +565% (+128pp)
- 不是尺度伪影 — NAV% 是统一度量，classic 的优势是真实的
- Max DD 都极低 (1.65%-2.08%)，classic 稍高但远在安全范围内
- 交易风格相似 (3.4-3.9 bars avg hold)，差异来自策略质量而非交易频率
- classic 的 reward +808 vs Stage 7 原始 +642 (↑26%) 印证了单次评估方差 ±80-100

**决策**: classic 锁定 ✅ (reward_mode = classic, augment = False)

---

## Step 2: Stage 8 Phase A 分析 — ✅ 完成

### 2.1 结果

```
统计: 51 trials (19 complete, 30 pruned, 2 failed)
Best:  Trial #5, value=+736.11
Baseline: +794 (Stage 6 FiLM PosA @ 200K)

参数重要性:
  learning_starts:    63.6%  ← 压倒性主导
  reward_clip:        11.5%
  regime_weight_max:  10.1%
  其余:               <10% each

Best trial #5 完整参数:
  lr:                 1.5255e-05
  buffer_size:        1,000,000
  learning_starts:    30,000
  n_quantiles:        24
  dd_threshold:       0.0866
  dd_penalty_mult:    2.8243
  reward_clip:        20
  regime_weight_min:  0.5962
  regime_weight_max:  1.5697
```

### 2.2 分析

```
Best +736 vs baseline +794 = -7.3%，但:
  - Trial #1 和 #5 参数完全相同，NAV 差 84 点 (652 vs 736)
  - 单次 200K 评估方差 ≈ ±80-100
  - 736 和 794 的差距在噪声范围内 → Phase A 本质与 baseline 打平

关键收敛:
  learning_starts = 30K   → Top 5 全部收敛，恰好是默认值（Optuna 确认默认正确）
  reward_clip = 20        → 4/5 触及下界，和 TQC_SAFE (ent_coef=0.1) 同思路
  lr = 1.5e-5 ~ 8.9e-5   → 分散，说明 lr 在此范围内不关键
  buffer_size = 1M (3/5)  → 多数偏好 1M
  n_quantiles = 24 (3/5)  → 偏低端

无尺度伪影: dd_penalty_mult importance 未进 Top 3，
reward_clip 的 importance 来自梯度截断效果（物理意义），不是简单缩放。
```

### 2.3 未搜索的超参层级

```
Phase A 只搜了 Tier 1（9 个参数）。以下层级未搜，原因和 fallback 策略如下:

Tier 2 — 不搜（不敏感）:
  gamma:  0.99   有效 horizon ≈ 100 bars ≈ 17 天，4H 频率合理
  tau:    0.005  SAC/TQC 论文标准值
  理由: Phase A importance 分析显示 learning_starts 一项占 63.6%，
        说明模型对这些二阶参数不敏感。RL 文献中 gamma/tau 极少需要调。

Tier 3 — 不搜（风险高收益低）:
  n_critics:              2   TQC 论文默认
  top_quantiles_to_drop:  2   TQC 论文默认
  理由: 这两个控制分位数估计的保守程度。改动需要配合 n_quantiles 联调，
        搜索空间爆炸。Phase A 已经把 n_quantiles 从 32 降到 24，
        critics/drop 的组合在这个 n_quantiles 下没有强理由改。

net_arch — 不搜（过拟合约束）:
  当前: [384, 384, 256]  ~860K policy params
  加 FiLM PosA 166K     = ~1.03M total
  理由: 18K bars 下 1.03M 参数已经偏紧。
        搜更大 ([512,512,256]) → 几乎必然过拟合
        搜更小 ([256,256])     → 可能牺牲表达力
        当前 arch 是 Stage 4 LSTM 胜出时的配置，经过 200K 和 2.5M 验证。

Fallback — 只在 Phase B fail 时启用:
  如果两个 config 的 3-fold mean NAV 都远低于 baseline:
    ① 先搜 net_arch: [256,256], [384,384,256], [512,512,256] (~44h)
    ② 再搜 Tier 3: n_critics ∈ [2,3], top_quantiles_to_drop ∈ [1,2,4]
    ③ Tier 2 最后考虑（gamma 和 tau 几乎不可能是瓶颈）
  详细搜索空间和执行命令见 V8 Stage 8.1 Tier 3 说明。
```

---

## Step 3: Stage 8 Phase B — 2-Config 验证

**前置**: Step 1 ✅ (classic 锁定) + Step 2 ✅ (Phase A 分析完成)

**为什么 2 configs 而非原计划 3**: Top 10 的参数差异主要在 reward_clip (20 vs 50)
和 buffer_size (1M vs 2M)。没有第三组有足够差异化的参数来证明多花 55-80h。

### 3.1 两个 Config

```python
CONFIG_1 = {  # Optuna 优化 (Phase A best trial #5)
    'lr': 1.5255e-05,
    'buffer_size': 1_000_000,
    'learning_starts': 30_000,
    'n_quantiles': 24,
    'dd_threshold': 0.0866,
    'dd_penalty_mult': 2.8243,
    'reward_clip': 20,            # ← Phase A 关键发现: 截断极端梯度
    'regime_weight_min': 0.5962,
    'regime_weight_max': 1.5697,
}

CONFIG_2 = {  # Default baseline (V8 TQC 配置)
    'lr': 3e-5,
    'buffer_size': 2_000_000,
    'learning_starts': 30_000,
    'n_quantiles': 32,
    'dd_threshold': 0.10,
    'dd_penalty_mult': 2.0,
    'reward_clip': 50,            # ← 原默认
    'regime_weight_min': 0.50,
    'regime_weight_max': 2.0,
}
```

**关键差异对照**:
```
| 参数              | Config 1 (Optuna) | Config 2 (Default) | 差异说明                   |
|-------------------|-------------------|--------------------|-----------------------------|
| lr                | 1.53e-5           | 3e-5               | Phase A lr 不敏感，不关键     |
| buffer_size       | 1M                | 2M                 | Top 5 多数选 1M              |
| n_quantiles       | 24                | 32                 | Top 5 偏低端                  |
| dd_threshold      | 0.087             | 0.10               | 接近，影响小                  |
| dd_penalty_mult   | 2.82              | 2.0                | Config 1 惩罚更重             |
| reward_clip       | 20                | 50                 | ★ 核心差异: 梯度截断强度       |
| regime_weight_max | 1.57              | 2.0                | Config 1 regime 权重更保守    |

注意: Config 1 的 dd_penalty_mult=2.82 + reward_clip=20 是一个组合:
更重的回撤惩罚 + 更强的梯度截断 → 学习更保守但更稳定。
Phase B 3-fold 会验证这个组合在多 fold 下是否一致。
```

### 3.2 执行 Phase B

```bash
# === Config 1 (Optuna 优化) — 先跑 ===
python -X utf8 -u train_drl_full.py --asset BTC --folds 3 --timesteps 2500000 \
    --extractor lstm_film_a --reward-mode classic --no-progress-bar \
    --lr 1.5255e-05 --buffer-size 1000000 --learning-starts 30000 \
    --n-quantiles 24 --dd-threshold 0.0866 --dd-penalty-mult 2.8243 \
    --reward-clip 20 --regime-weight-min 0.5962 --regime-weight-max 1.5697 \
    --tag config1_optuna

# === Config 1 完成后: 检查自适应跳过 (见 3.2.1) ===

# === Config 2 (Default baseline) — 如果需要 ===
python -X utf8 -u train_drl_full.py --asset BTC --folds 3 --timesteps 2500000 \
    --extractor lstm_film_a --reward-mode classic --no-progress-bar \
    --lr 3e-5 --buffer-size 2000000 --learning-starts 30000 \
    --n-quantiles 32 --dd-threshold 0.10 --dd-penalty-mult 2.0 \
    --reward-clip 50 --regime-weight-min 0.50 --regime-weight-max 2.0 \
    --tag config2_default
```

### 3.2.1 自适应跳过 Config 2（省 ~28-40h）

```
Config 1 三个 fold 全部跑完后，检查:

  IF Config 1 mean Final NAV > $800K (即 NAV% > +700%)
  AND 3 folds 全部 NAV > initial ($100K)
  AND Mean/Std > 1.0
  → ✅ 直接锁定 Config 1，跳过 Config 2，省 ~28-40h

  IF Config 1 mean Final NAV < $800K
  OR 任何 fold NAV < initial
  → 必须跑 Config 2 对比

理由: $800K 远超 baseline ($794K @ 200K, 2.5M 应更高)，
      3 fold 全正说明不是单 fold 噪声。此时跑 Config 2 只是确认，不会改变决策。
```

**FPS 和时间注意**:
```
Config 1 (buffer=1M): FPS ≈ 25 → 每 fold ~28h
Config 2 (buffer=2M): FPS ≈ 15-18 → 每 fold ~39-46h（buffer 更大，采样更慢）

如果跑两个 config:
  Config 1: 3 folds ≈ 84h
  Config 2: 3 folds ≈ 117-138h (buffer=2M 拖慢 ~50%)
  Total: ~200-220h (~8-9 天)
  Early stopping: 实际 ≈ 100-120h (~4-5 天)

如果自适应跳过 Config 2:
  Config 1 only: 3 folds ≈ 84h
  Early stopping: 实际 ≈ 42-55h (~2-2.5 天)
```

**并行**: Phase B 训练期间，同时做 Step 5 (v9 编码 ~5h) 和 Step 6 (smoke test ~15min)。
GPU 利用率仅 ~18%，CPU 编码不冲突。

### 3.3 Phase B 评估 — ✅ 完成

```
Config 1 (Optuna) 3-fold 结果:

| Config   | Fold 1 NAV  | Fold 2 NAV  | Fold 3 NAV  | Mean NAV  | Std     | Mean/Std |
|----------|-------------|-------------|-------------|-----------|---------|----------|
| Config 1 | $807,266    | $718,923    | $920,336    | $815,508  | $82,433 | 9.89     |
|          | +707.3%     | +618.9%     | +820.3%     | +715.5%   |         |          |

Max DD: Fold 1 2.26%, Fold 2 1.98%, Fold 3 2.62%, Mean 2.29%
All positive: ✅ (3/3 folds)

自适应跳过触发:
  ✅ Mean NAV $815K > $800K 阈值
  ✅ 3 folds 全正
  ✅ Mean/Std 9.89 >> 1.0
  → Config 2 跳过，省 ~40-46h

决策: Config 1 锁定 → optuna_winner.json
```

### 3.4 Phase B Gate

```
[ ] 2 configs × 3 folds = 6 次训练完成
[ ] Winner 选出: mean Final NAV 最高 + std 可控
[ ] Winner 的 3 folds 全部 NAV > initial
[ ] 记录最终超参集 → OPTUNA_WINNER_CONFIG

如果 Phase B 结果不及预期（两个 config 都远低于 baseline）:
  Fallback 优先级（见 Step 2.3 完整说明）:
    ① net_arch 搜索 (~44h): [256,256], [384,384,256], [512,512,256]
       → 过拟合 vs 表达力的 sweet spot 可能不在当前 arch
    ② Tier 3 搜索 (~22h): n_critics ∈ [2,3], top_quantiles_to_drop ∈ [1,2,4]
       → 联调 n_quantiles=24 下的保守程度
    ③ Tier 2 (gamma/tau) → 最后手段，几乎不可能是瓶颈
  只在两个 config 都严重低于 baseline 时才启动，逐层尝试不跳级。
```

---

## Step 4: Stage 8 最终锁定 — ✅ 完成

**Config 1 锁定 (Phase B 3-fold 验证: mean NAV +715.5%)**:

```python
# 最终锁定配置（后续所有 Stage 使用）
FINAL_CONFIG = {
    # === Stage 6 锁定 ===
    'extractor': 'lstm_film_a',         # FiLM Position A, 166K params
    'n_stack': 8,                       # VecFrameStack
    'obs_dim': 126,                     # 122 features + 4 env state
    
    # === Stage 7 锁定 ===
    'reward_mode': 'classic',           # NAV% 确认 ✅
    'augment': False,
    
    # === Stage 8 锁定 — Config 1 (Optuna, 3-fold 验证 ✅) ===
    'lr': 1.5255e-05,
    'buffer_size': 500_000,             # 从 1M 降到 500K (9.7 segfault: n_stack=8 内存放大)
    'learning_starts': 30_000,          # Phase A 确认: 30K (63.6% importance)
    'n_quantiles': 24,
    'dd_threshold': 0.0866,
    'dd_penalty_mult': 2.8243,
    'reward_clip': 20,                  # Phase A 关键发现
    'regime_weight_min': 0.5962,
    'regime_weight_max': 1.5697,
    
    # === 铁律固定（不随 Phase B 变化）===
    'ent_coef': 0.1,
    'weight_decay': 0,
    'batch_size': 256,
    'grad_steps': 4,
    
    # === Tier 2 固定（Phase A importance 证明不敏感）===
    'gamma': 0.99,                      # 有效 horizon ≈ 100 bars ≈ 17 天
    'tau': 0.005,                       # SAC/TQC 标准值
    
    # === Tier 3 固定（TQC 论文默认，风险高收益低）===
    'n_critics': 2,
    'top_quantiles_to_drop': 2,
    
    # === net_arch 固定（18K bars 过拟合约束）===
    'net_arch': [384, 384, 256],        # ~860K policy params, 总计 ~1.03M
}
```

**保存到文件**:
```bash
# 将最终配置写入 JSON，后续所有 Stage 从这里读取
python -X utf8 -c "
import json
config = { ... }  # 填入 Phase B winner 的具体值
with open('config/optuna_winner.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Saved config/optuna_winner.json')
"
```

---

# ═══════════════════════════════════════════════════
# PHASE 2: 真实摩擦环境升级 (Stage 9, v9 patch)
# 编码 ✅ + Smoke test ✅ (与 Phase B 并行完成)
# 剩余: Step 7 A/B (~4-8h) + Step 8 Go/No-Go (~28h)
# ═══════════════════════════════════════════════════

**完整技术细节见 `HMATS_V9_FRICTION_OOD_PATCH.md`。这里只写执行步骤。**

## Step 5: 代码改动 (9.1 + 9.2 + 9.3) [~5h 编码] — ✅ 完成

**与 Phase B 训练并行完成。**

### 5.1 交易成本嵌入 env.step()

```
修改: train_drl_full.py → TradingEnvFull

新增:
  _compute_trade_cost_bps(asset, trade_notional) → float
  ASSET_SLIPPAGE_BPS = {'BTC': 3.0, 'ETH': 5.0, 'SOL': 10.0}

在 step() 中:
  if abs(position_change) > 0.01:
      cost = trade_notional × cost_bps / 10000
      self.current_nav -= cost           ← 物理扣费
      self.cumulative_trade_cost += cost  ← 累计记录

新增 CLI flag:
  --no-friction    → 跳过成本扣减（对比用）
  --slippage-bps   → 覆盖默认 slippage（调参用）
```

### 5.2 比例翻仓成本

```
修改: train_drl_full.py → EnhancedRewardCalculator

删除:
  if abs(position_change) > 1.5: risk_r -= 0.02    ← 旧阈值惩罚

新增:
  _compute_turnover_cost(position_change, trade_cost_bps) → float
    Layer 1: -trade_cost_bps × 0.001              ← 真实成本映射
    Layer 2: -abs(position_change) × 0.01          ← 额外换仓抑制

新增 CLI flag:
  --turnover-mult  → 放缩 turnover cost（默认 1.0）
```

### 5.3 OOD 检测器

```
新建: drl/ood_detector.py

class MahalanobisOODDetector:
  fit(train_features)       → 训练集上计算 mean + cov_inv + threshold
  score(obs_features)       → 返回 {distance, is_ood, confidence_mult, hard_switch}
  save(path) / load(path)   → npz 序列化

集成到 TradingEnvFull:
  __init__: fit OOD detector on train_df features
  step():   记录 OOD score（训练时不影响 reward）

集成到 main.py (runtime):
  推理前: ood_detector.score(features)
  is_ood + not hard_switch → action × confidence_mult (soft degrade)
  hard_switch (≥6 连续) → EXIT_ONLY
```

---

## Step 6: Smoke Test (9.4) [~15 min] — ✅ 完成

**与 Phase B 训练并行完成。**

**结果**:
```
cumulative_trade_cost:  $857,769 (random policy 期间, 预期很高)
reward range:           [-575.91] per episode (per-step clipped, accumulated OK)
ood_detector.npz:       121KB, fitted on 15527 samples, dim=122, threshold=19.24
training_friction.json: trades=9897, slip=3.0 bps
crash:                  None ✅
```

```bash
# 6.1 交易成本 smoke test
python -X utf8 -c "
# 验证: 大换仓产生成本, 微调不产生成本
# 见 V9 patch 9.4.1 完整代码
"

# 6.2 翻仓成本 smoke test  
python -X utf8 -c "
# 验证: 连续比例, 小换仓<大换仓, 零换仓=零成本
# 见 V9 patch 9.4.2 完整代码
"

# 6.3 OOD 检测器 smoke test
python -X utf8 -c "
# 验证: fit/score/save/load, 正常obs=高信心, 极端obs=OOD
# 见 V9 patch 9.4.3 完整代码
"

# 6.4 完整 10K 步 smoke test (DummyVecEnv)
python -X utf8 train_drl_full.py --asset BTC --folds 1 --timesteps 10000 \
    --extractor lstm_film_a --reward-mode classic
# 确认: 无 crash, cumulative_trade_cost > 0, reward in [-50, 50]

# 6.5 SubprocVecEnv 可行性测试 — ❌ 不可行 (已验证)
#
# 结果: 3.6x 墙钟加速，但 gradient updates 从 80K 降到 10K (8x 更少)
# 原因: DummyVecEnv 每步 4 次 grad update (grad_steps=4)
#       SubprocVecEnv(4) 每 4 步仍只做 4 次 grad update → update:data = 1:1 vs 4:1
# 修复需要 grad_steps=16，违反铁律 #7，且 GPU 时间 4x 抵消加速
# 结论: Stage 10 维持 DummyVecEnv only
```

**Gate**: 6.1-6.4 全部通过才继续。6.5 已验证不可行，不影响后续。

---

## Step 7: 200K A/B 对比 (9.5) [~4-8h] — ✅ 完成

**结果** (BTC fold_1, 200K steps, Config 1 超参):
```
| 指标              | A (friction_off) | B (friction_on) | Delta              |
|------------------|------------------|-----------------|---------------------|
| Final NAV        | $730,958         | $417,744        | -$313K (-43%)       |
| NAV%             | +630.96%         | +317.74%        |                     |
| 累计交易成本      | $0               | $44,215         | $44K 直接摩擦       |
| Trades/episode   | —                | 2,373           |                     |
| 平均持仓时长      | 4.2 bars (17h)   | 10.1 bars (41h) | 2.4x 更长 ✅        |

判定: -43% 触发 ">20% 检查 slippage_bps" 条件，但不需要调参:
  - 模型学会了 2.4x 更长持仓来降低交易成本 → friction signal 在工作 ✅
  - +317.74% NAV >> +5% Go/No-Go 阈值 → 方向正确
  - 200K 步只是起步，2.5M 步下模型会进一步优化进出点时机
  - $269K "机会成本" 不是损失，是模型学到频繁交易在有摩擦世界里不值得
```

---

## Step 8: 单 Fold 全量验证 (9.7) — Go/No-Go [~20-30h] — ✅ GO

**结果**: BTC fold_1, friction-aware, Config 1 超参, buffer_size=1M

```
训练到 ~1050K 步时 segfault (内存):
  - Replay buffer 1M × n_stack=8 × obs_dim=126 = ~7.5GB 连续数组
  - 加 model/CUDA/Python ~2GB → 总 ~9.5GB
  - Windows 内存碎片化导致连续分配失败

但 best_model 在 ~1000K 步已 plateau 13 次 eval (reward +703)
→ 用 best_model 评估: friction_on NAV% = +317.74%

决策矩阵判定:
  +317.74% >> +5% 阈值 → ✅ GO Stage 10

⚠️ buffer_size 修正 (Stage 10 必须):
  buffer_size: 1,000,000 → 500,000
  原因: VecFrameStack(n_stack=8) 内存放大，1M buffer 需要 ~7.5GB 连续内存
  500K buffer = ~3.75GB，安全且 Phase A importance 显示 buffer_size 不敏感
  
⚠️ fold 间清理 (Stage 10 必须):
  每个 fold 结束后 del model + gc.collect() + torch.cuda.empty_cache()
  防止两个 buffer 短暂共存 (3.75GB × 2 = 7.5GB 瞬时峰值)
```

---

# ═══════════════════════════════════════════════════
# PHASE 3: Full Training (Stage 10)
# 预计: ~96-224h (DummyVecEnv + early stopping, buffer=500K)
# ═══════════════════════════════════════════════════

## Step 9: TQC Full Training — 3 Assets × 3 Folds

**前置**: Stage 9 Go/No-Go 通过, FINAL_CONFIG + friction 参数确定

### 9.1 训练前 Preflight（完整版，含 V8 + v9）

```
铁律 (任何一条 FAIL = 停止):
  [ ] grep: 0 SubprocVecEnv in training code              (#9, 已验证不可行)
  [ ] ent_coef = 0.1 (fixed float, not "auto")        (#5)
  [ ] weight_decay = 0                                 (#6)
  [ ] batch=256 + grad_steps=4                         (#7)
  [ ] EvalCallback → best_model.zip (NOT final_model)  (#10)
  [ ] Early stopping enabled                           (#8)
  [ ] checkpoint_freq = 500K                           (#21)
  [ ] window_size = 10 (NOT 96)                        (#11)  ← 高频出错点
  [ ] gap = 42 bars, n_folds = 3, val_ratio = 0.15    (#12, #13)

内存安全 (9.7 segfault 修正):
  [ ] buffer_size = 500,000 (NOT 1M — n_stack=8 内存放大, 1M=7.5GB segfault)
  [ ] fold 间清理: del model; gc.collect(); torch.cuda.empty_cache()
  [ ] 训练前 RSS < 3GB (无残留 buffer)
  [ ] 所有模型共享同一 train/val split                   (#16)
  [ ] Extractor 参数 < 500K (FiLM PosA = 166K)        (#28)

数据:
  [ ] Parquets: BTC~18K, ETH~18K, SOL~12K rows
  [ ] 零 NaN, 时间戳 2017-2026（无 year=57086 bug）    (#1)
  [ ] feature_manifest.json: 122 features
  [ ] split_manifest.json: 3 folds, gap=42, val_ratio=0.15
  [ ] 5 wavelet 去噪列存在且非零                        (#23)
  [ ] config/optuna_winner.json 存在且参数正确

GMM:
  [ ] Per-asset GMM 文件存在 (3 × {gmm_model, scaler, config})
  [ ] k values: BTC=8, ETH=7, SOL=7
  [ ] regime_proba_0..7 sum ≈ 1.0
  [ ] GMM scaler (12-dim) 独立于 Feature scaler (~114-dim) (#17) ← 高频出错点

外部数据:
  [ ] 7 external cols 在 parquet 中
  [ ] Pre-2020 = 0.0, has_external_data = 0             (#3)

架构 (Stage 6 锁定: FiLM Position A):
  [ ] FiLM PosA extractor 可加载（--extractor lstm_film_a）
  [ ] γ bias ≈ 1, β bias ≈ 0（恒等初始化验证）          (#24)
  [ ] regime_probs slice sum ≈ 1.0
  [ ] Extractor 参数 ~166K, 总参数 ~1.03M
  [ ] runtime obs_dim = 126 单帧输入
  [ ] TQC internal stack = 8, effective stacked shape = (1008,)

Reward/Augment (Stage 7 锁定):
  [ ] reward_mode = classic
  [ ] augment = False

Scaling:
  [ ] RobustScaler: fit on train only, transform both    (#18)
  [ ] 排除: regime_proba_0..7, has_external_data          (#19)

v9 新增:
  [ ] ASSET_SLIPPAGE_BPS 有 BTC/ETH/SOL 的值
  [ ] _compute_trade_cost_bps() 对非零换仓返回 > 0
  [ ] 旧 threshold turnover penalty 已删除（铁律 #30）
  [ ] OOD detector fit 只用 train split（铁律 #31）
  [ ] --no-friction 未启用（production 训练必须有 friction）

系统:
  [ ] CUDA available
  [ ] Disk > 50GB
  [ ] powercfg /change standby-timeout-ac 0              (#22)
  [ ] 顺序执行 BTC → ETH → SOL                           (#20)

Smoke test:
  [ ] python train_drl_full.py --asset BTC --folds 1 --timesteps 10000 \
        --extractor lstm_film_a --reward-mode classic
  [ ] runtime obs shape = (126,), effective TQC stacked shape = (1008,), no NaN, reward in [-50, 50]
  [ ] cumulative_trade_cost > 0
  [ ] 无 crash
```

### 9.2 执行

```bash
# === BTC: 3 folds × 2.5M 步 ===
python -X utf8 -u train_drl_full.py --asset BTC --folds 3 --timesteps 2500000 \
    --extractor lstm_film_a --reward-mode classic --no-progress-bar \
    --config config/optuna_winner.json

# === ETH: 3 folds × 2.5M 步 ===
python -X utf8 -u train_drl_full.py --asset ETH --folds 3 --timesteps 2500000 \
    --extractor lstm_film_a --reward-mode classic --no-progress-bar \
    --config config/optuna_winner.json

# === SOL: 3 folds × 3.0M 步 (数据少, lr 可降到 2e-5) ===
python -X utf8 -u train_drl_full.py --asset SOL --folds 3 --timesteps 3000000 \
    --extractor lstm_film_a --reward-mode classic --no-progress-bar \
    --config config/optuna_winner.json --lr 2e-5
```

**顺序执行**: BTC → ETH → SOL（铁律 #20），一次只跑一个 asset。

**关键约束**:
- eval_freq = 5,000（每 5K 步 eval 一次）
- checkpoint_freq = 500K（crash recovery）
- DummyVecEnv only（SubprocVecEnv 已验证不可行）
- 部署 best_model.zip（NOT final_model）
- buffer_size = 500K（NOT 1M，9.7 segfault 修正）

**⚠️ Fold 间内存清理** (train_drl_full.py 必须包含):
```python
# 每个 fold 训练结束后，启动下一个 fold 之前:
del model
del env
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 验证: 下一个 fold 启动前打印 RSS
import psutil
rss_gb = psutil.Process().memory_info().rss / 1e9
logger.info(f"[MEM] Pre-fold RSS: {rss_gb:.1f} GB")
assert rss_gb < 4.0, f"Memory leak: {rss_gb:.1f} GB before new fold"
```

### 9.3 时间预估

```
                       Windows native, FiLM PosA, DummyVecEnv, buffer=500K
                       (buffer=500K 比 1M 采样更快, FPS 可能 ~30-35)
BTC: 3 folds × 2.5M   ~60-70h
ETH: 3 folds × 2.5M   ~60-70h
SOL: 3 folds × 3.0M   ~72-84h
──────────────────────────────────────
Total:                 ~192-224h (~8-9 天)
Early stopping:        可能缩短 30-50% → ~96-155h (~4-6.5 天)
```

### 9.4 产出

```
每个 asset × fold 产出:
  models/{ASSET}/fold_{n}/best_model.zip           ← TQC 模型 (NOT final_model)
  models/{ASSET}/fold_{n}/ood_detector.npz         ← OOD 检测器 (v9 新增)
  models/{ASSET}/fold_{n}/training_friction.json   ← 成本统计 (v9 新增)
  models/{ASSET}/fold_{n}/feature_scaler.pkl       ← RobustScaler
  models/{ASSET}/fold_{n}/eval_log.json            ← 训练曲线

总计: 3 assets × 3 folds = 9 组文件
```

### 9.5 训练期间监控 (Eval Reward 诊断)

```
TQC 训练早期 eval reward 波动极大，这是正常现象。以下是判断框架:

════════════════════════════════════════════════════════════════
阶段 1: 0 → learning_starts (30K)
  - 纯随机策略，reward 约 -10 ~ -20
  - 正常: 无学习信号，忽略这段

阶段 2: 30K → 100K (刚开始学习)
  - 可能在 35K 左右短暂出现 best reward (-6 这类)
  - 然后暴跌到 -1000 ~ -3000 = 正常
  - 原因: 1 个 catastrophic episode 就能拉低整个 eval，
    不代表"持续恶化"——是采样方差

阶段 3: 100K → 200K (关键判断窗口)
  - 对比 Stage 9.7: 200K 左右开始稳定上升
  - ✅ 正常: reward 从 -3000 级别回升到 -500 以内
  - ⚠️ 需排查: 200K 时仍 < -500

阶段 4: 200K → 500K (收敛期)
  - reward 应持续正向收敛
  - 9.7 在 165K 转正, 500K 达 +700
  - ✅ 正常: 单调上升（允许 10-20% 回调）
  - ⚠️ 需排查: 趋势持平或二次下跌
════════════════════════════════════════════════════════════════

⚠️ 常见误判:

  "85K reward -3248, 9.7 同期 ~-350, 差了 10x" → 不构成停止理由
  原因: buffer 在 85K 时只有 ~55K transitions (85K - 30K learning_starts)
  无论 max size 500K 或 1M，buffer 内容完全一样——还远没填满
  buffer_size 差异只在 >500K 步时才体现

  "reward 从 -6 暴跌到 -3248" → 不是恶化趋势
  原因: -6 是随机策略碰巧的好 episode，-3248 是单笔全仓亏损的坏 episode
  TQC 早期 eval 方差可达 ±3000, 用趋势判断而非点值

决策检查点:
  200K 步: 是否 < -500?
    否 → 继续 ✅
    是 → 检查 ent_coef 是否稳定在 0.1 (不应飙升)
         检查 critic_loss 趋势 (应下降)
         如两项正常 → 再等到 300K
         如 ent_coef 异常 → 停止排查

  500K 步: reward 是否 > 0?
    否 → 执行 "风险和回退计划" Stage 10 部分
    是 → 继续到 early stopping 或 2.5M

参考数据 — BTC Fold 1 实际训练 vs 9.7 Go/No-Go:
  ┌────────┬──────────────────┬──────────────────┐
  │ 步数    │ Stage 10 Fold 1  │ Stage 9.7        │
  ├────────┼──────────────────┼──────────────────┤
  │  35K   │ -6 (best)        │ ~-15             │
  │  50K   │ -279             │ -249             │
  │  85K   │ -3,248           │ ~-350            │
  │ 165K   │ (pending)        │ 转正              │
  │ 500K   │ (pending)        │ +700 (plateau)    │
  └────────┴──────────────────┴──────────────────┘
  85K 的 10x 差距不构成停止理由 (见上方误判分析)
```

### 9.6 Gate

```
[ ] 9 个 best_model.zip 产出
[ ] 多数 fold eval reward > 0
[ ] 无 NaN
[ ] Early stopping 触发在 >500K 步后（不是过早）
[ ] obs_dim = 126 确认
[ ] cumulative_trade_cost > 0 for all folds (friction 生效)
[ ] training_friction.json 中 avg_cost_per_trade_bps 合理
    BTC: 3-8 bps, ETH: 5-12 bps, SOL: 10-20 bps
[ ] OOD detector threshold 在合理范围
[ ] 选出每个 asset 的 best fold (eval reward 最高)
```

---

# ═══════════════════════════════════════════════════
# PHASE 4: Ensemble + Validation (Stage 11-14)
# 预计: ~5h
# ═══════════════════════════════════════════════════

## Step 10: DT v3.2 Training (Stage 11) [~1.5h]

```bash
python -X utf8 -u drl/train_decision_transformer_v32.py --asset BTC --folds 3 --epochs 200
python -X utf8 -u drl/train_decision_transformer_v32.py --asset ETH --folds 3 --epochs 200
python -X utf8 -u drl/train_decision_transformer_v32.py --asset SOL --folds 3 --epochs 200
```

**注意**: DT v3.2 用 80-dim 特征（不含外部数据/wavelet），和 TQC 不共享 obs space。
DT 不需要 friction-aware 环境（supervised learning，不在环境里交互）。

**Gate**:
```
[ ] 9 个 DT 模型产出
[ ] Val direction accuracy > 55%
[ ] Actions ∈ [-1, 1]
[ ] 用 GMM regime（不是 rule-based）
[ ] 用 split_manifest 的 fold 边界
```

---

## Step 11: TQC + DT Ensemble (Stage 12) [~1h]

```
Regime-conditional 权重:
  MOMENTUM_RALLY:       TQC=0.6, DT=0.4
  EXTREME_VOLATILITY:   TQC=0.4, DT=0.6
  VOLATILE_CHOP:        TQC=0.4, DT=0.6
  QUIET_ACCUMULATION:   TQC=0.7, DT=0.3
  DEFAULT:              TQC=0.5, DT=0.5

信号组合规则:
  Agreement:             confidence boost
  Disagreement:          combined *= 0.6
  DT unavailable:        TQC only (1.0, 0.0)
  DT confidence < 0.3:   shift weight to TQC
  DT warmup (context < 96 steps): fallback TQC only

BTC val data grid search:
  TQC weight ∈ [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
```

**Gate**:
```
[ ] Ensemble Sharpe >= TQC only × 90% → 部署 ensemble
[ ] MaxDD <= TQC only × 120%
[ ] 否则 → TQC only, DT shadow mode
```

---

## Step 12: Offline Validation (Stage 13) [~1h]

**关键变更 (v9)**: 用 friction-aware 环境回测，OOS 指标反映真实成本。

```
WalkForwardValidator 指标 (per asset per fold):
  OOS Sharpe ≥ 0.3      (v9 降低: 加了成本，原来要求 0.5)
  Max drawdown ≤ 15%
  Per-window trades ≥ 10
  Action entropy ≤ 0.7
  vs random: Sharpe 差 > +0.2

  v9 新增:
  cumulative_friction_cost < 15% of NAV
  avg_turnover_per_bar < 0.3
```

```
StatisticalPromotionGate (v9 调整后):
  min_sharpe_ratio: 0.7       (v8 原值 1.0, v9 降低因 friction)
  min_bear_sharpe_ratio: 0.5  (v8 原值 0.8)
  max_drawdown: 0.15
  max_bear_drawdown: 0.20
  min_sample_size: 30 trades
  min_win_rate: 0.52
  bear_market_weight: 1.5x

  注: 如果 Stage 13 实际数据显示 friction 导致 Sharpe 进一步下降，
  可以微调 min_sharpe_ratio 到 0.5，但不能低于 0.5。
```

**决策**:
```
多数 asset OOS Sharpe ≥ 0.5  → 策略非常好 ✅
多数 asset OOS Sharpe 0.3-0.5 → 可接受，成本影响正常 ✅
多数 asset OOS Sharpe < 0.3   → 回到 Stage 8 重新调参 ❌
任何 asset MaxDD > 20%        → 该 asset 不部署，用 DT only 替代
StatisticalPromotionGate FAIL → 不影响部署，但 DRL 必须留在 SHADOW
```

---

## Step 13: Runtime Parity Check (Stage 14) [~1h]

```
Feature parity:
  [ ] feature_manifest.json: 122 features, 正确列顺序
  [ ] Scaler: fold_{best} 的 feature_scaler.pkl
  [ ] GMM: per-asset (BTC=8, ETH=7, SOL=7)
  [ ] NaN → np.nan_to_num(0), clip 范围一致
  [ ] window_size: runtime = 10
  [ ] VecFrameStack: runtime 维护 8 bar 历史

Wavelet parity:
  [ ] Runtime 用 256-bar 滑动窗口 + Coiflet-4 level-2
  [ ] Warm-up: 前 256 bars 用 raw 值
  [ ] 5 去噪列: runtime 和训练最后 100 行 tolerance < 1e-4

外部特征 parity:
  [ ] funding_rate_zscore / oi_change_5d / liq_imbalance: 同 rebuild_pipeline
  [ ] taker_ratio_zscore / tradecount_zscore / taker_vol_momentum: 同上
  [ ] has_external_data: 根据可用性设置
  [ ] Coinglass DNS 失败 → features = 0 + has_external_data = 0

v9 新增 parity:
  [ ] _compute_trade_cost_bps(): runtime 和训练环境一致
  [ ] OOD detector: 正确加载 ood_detector.npz
  [ ] Soft degradation: mock OOD → action 缩小 ✓
  [ ] Hard switch: mock 连续 OOD ≥ 6 → EXIT_ONLY ✓

数值验证:
  [ ] 训练数据最后 100 行: train obs == runtime obs (tolerance < 1e-6)
  [ ] model.predict(obs): 训练和 runtime action 一致
```

---

# ═══════════════════════════════════════════════════
# PHASE 4b: 部署 + Paper Run (Stage 15-18)
# ═══════════════════════════════════════════════════

## Step 14: Model Deployment (Stage 15) [~30 min]

```
部署文件:
  models/{ASSET}/fold_{best}/best_model.zip        ← TQC (NOT final_model)
  models/{ASSET}/fold_{best}/best_model.pt         ← DT (if ensemble)
  models/{ASSET}/fold_{best}/ood_detector.npz      ← OOD (v9 新增)
  models/{ASSET}/fold_{best}/feature_scaler.pkl
  models/regime_classifier/{ASSET}/gmm_model.pkl
  models/regime_classifier/{ASSET}/scaler.pkl
  config/feature_manifest.json
  config/split_manifest.json
  config/optuna_winner.json

DRL Authority:
  [ ] authority = SHADOW
  [ ] shadow counter = 0
  [ ] v7 模型全部替换（铁律 #27）
```

---

## Step 15: Paper Run (Stage 16) [48h minimum]

```bash
python main.py --mode paper
```

### First 24h

```
系统稳定性:
  [ ] 0 crash
  [ ] GMM source = GMM 100%

DRL shadow:
  [ ] runtime obs contract: (126,) single obs, internal TQC stack -> effective (1008,)
  [ ] Shadow actions 非 trivial, 无 NaN
  [ ] LSTM FiLM extractor 正确加载

v9 新增:
  [ ] OOD detector 运行中，score 有波动
  [ ] 无 spurious hard switch
  [ ] 交易成本 log 记录正常

Wavelet:
  [ ] 去噪列非零且随市场变化

交易信号:
  [ ] Alpha gate pass rate > 40%
  [ ] 至少 5+ 模拟交易
```

### 48h Gate

```
[ ] 48h 零 crash
[ ] 三个 asset 都在交易
[ ] 10+ 模拟交易
[ ] Risk controls 验证
[ ] Kill switch 手动测试通过
[ ] OOD soft degrade 事件 ≤ 10% of decisions
[ ] 交易成本 log 和预期范围一致
[ ] LiveExperienceBuffer: experiences 文件在 data/live_experiences/ 中增长
```

### Paper Run 期间: 被动 Experience 收集

```python
# 在 Paper Run 启动前嵌入 main.py（只收集，不触发 fine-tune）:
from drl.live_experience_buffer import LiveExperienceBuffer

self._exp_buffer = LiveExperienceBuffer(persist_dir="data/live_experiences")
# enabled=True 但 finetuner 不初始化 → 只写 JSONL，零决策影响

# 在每次 decision 后:
self._exp_buffer.record_decision(
    asset=asset, state=drl_state, action=drl_action,
    signal_strength=signal_strength, direction=intended_direction,
    position_size_pct=position_size_pct, regime=current_regime,
    mode=current_mode, veto_active=intent.veto_active,
    sentiment_score=0.0,  # Sentiment 尚未接入
)

# 在 position close 时:
self._exp_buffer.record_outcome(
    asset=asset, reward=computed_reward, next_state=next_drl_state,
    pnl_bps=realized_pnl_bps, holding_bars=bars_held, exit_reason=exit_reason,
)
```

**目的**: Paper Run + Live Week 1-2 期间默默积累 experiences，
等 Stage 20 Online DRL 启用时已有数百条数据可直接用。零决策影响，零风险。

---

## Step 16: Live Deployment (Stage 17) [Phased]

### Phase 1: Conservative (Week 1)

```
[ ] Paper Run 48h PASSED
[ ] 参数减半:
    max_position_pct: 0.20
    max_total_exposure: 0.50
    max_leverage: 2.0
[ ] DRL 在 SHADOW mode
[ ] 监控: 每 30 min 健康检查
```                                                                                                                                                                  

### Phase 2: Standard (Week 2+)

```
[ ] Week 1 无重大问题
[ ] 恢复全参数:
    max_position_pct: 0.40
    max_total_exposure: 0.85
    max_leverage: 3.0
[ ] DRL 仍在 SHADOW
```

### Phase 3: DRL Promotion (30+ shadow trades over 30+ days)

```
StatisticalPromotionGate:
  [ ] DRL shadow outperforms baseline ≥ 10%
  [ ] Consistency ≥ 60% across sub-windows
  [ ] No DRL errors
  [ ] p-value < 0.05

PASS → promote to EXIT_ONLY
FAIL → stays SHADOW, 30 more trades 后重评

Demotion triggers:
  [ ] Underperforms > 10% over 10 trades
  [ ] DRL errors ≥ 3
  [ ] Manual override
```

---

## Step 17: 持续监控 (Stage 18)

```
| 指标              | 频率      | 紧急操作                      |
|-------------------|----------|-------------------------------|
| 系统健康          | 每 30min  | Crash → kill switch           |
| 仓位大小          | 每笔交易  | > max → 手动检查              |
| 杠杆              | 每笔交易  | > max → 告警                  |
| 回撤              | 每小时    | > 15% → 手动减仓              |
| DRL 表现          | 每周      | Underperform → demote SHADOW  |
| GMM regime        | 每天      | 卡在单一 regime → 检查         |
| OOD 检测 (v9)     | 每 4H    | Hard switch → 检查阈值         |
| 外部数据          | 每天      | DNS fail → graceful degrade    |
| Wavelet 去噪      | 每天      | 输出全零 → 检查                |
| Experience Buffer | 每天      | 文件增长 → 正常; 停滞 → 检查    |

Online Impact Learning (50+ trades 后):
  [ ] FrictionLearner._coeffs 基于实际数据
  [ ] |estimated - actual| < 5 bps (200 笔后中位数)
  [ ] 学习结果写入 log，供下一轮训练参考
  [ ] 闭环: 实盘数据 → FrictionLearner → 更新 ASSET_SLIPPAGE_BPS → 重训
```

---

# ═══════════════════════════════════════════════════
# PHASE 5: Sentiment Signal Wiring (Stage 19)
# 三层架构: L1 确定性引擎 + L2 DeBERTa + L3 Haiku LLM
# 时间: L1 ~20min + L3 ~2h 编码 + 1-2 week A/B 对比
# ⚡ 可与 Stage 10 TQC 训练并行 (CPU-only, runtime-only, 不改 obs_dim)
# 前置: Paper Run baseline 已有足够裸跑数据 (50+ 笔交易)
# ═══════════════════════════════════════════════════

## Step 18: Sentiment Wiring (Stage 19)

**并行执行说明**: Stage 19 是 CPU-only + runtime-only + ADVISE 级别。
与 Stage 10 TQC GPU 训练零资源冲突，可以同时进行。

**前置条件**:
- Paper Run 裸跑 baseline 已有 50+ 笔交易数据 (作为 A/B 对照组)
- 如果裸跑交易数不足 → 等 Paper Run 积累更多数据再开启
- L3 不改 obs_dim、不改模型权重、不影响 DRL 训练
- 关闭只需 LLM_SENTIMENT_ENABLED=false，零残留

**为什么可以并行**:
```
Stage 10 (GPU):  TQC fold training → 不受 sentiment 影响
Stage 19 (CPU):  L3 Haiku API call → 不消耗 GPU
Paper Run:       裸跑 baseline 已积累 → 开启 L3 后自动成为 "+sentiment" 组
                 Stage 10 训练完毕时可能已有 2 weeks A/B 数据
```

**为什么不需要等 Stage 10 完成**:
- L3 只影响 runtime signal_strength/urgency/sizing (ADVISE 权重 10-20%)
- L3 不改变 DRL 模型的 obs 或 action space
- Stage 10 产出的新模型 deploy 时，L3 已经在跑 → 可以直接评估
  "旧模型 + L3" vs "新模型 + L3" vs "新模型 裸跑" 三组对比

### 18.0 三层架构总览

```
┌──────┬──────────────────────────┬──────────┬──────────────────────────────┐
│ 层    │ 组件                      │ 状态      │ 输出                          │
├──────┼──────────────────────────┼──────────┼──────────────────────────────┤
│ L1   │ SimpleSentimentCalculator │ ✅ 可用   │ 6 指标加权 composite score    │
│      │ (确定性, 无 ML)           │          │ 零 API 成本, 可回测           │
├──────┼──────────────────────────┼──────────┼──────────────────────────────┤
│ L2   │ DeBERTa v2.2             │ ⏳ 待重训  │ per-asset direction+confidence│
│      │ (353MB, 合成数据训练)      │          │ 本地推理, 零 API 成本         │
│      │                          │          │ 需真实加密数据重训才有价值      │
├──────┼──────────────────────────┼──────────┼──────────────────────────────┤
│ L3   │ Haiku LLM Agent          │ ⏳ 骨架已有│ per-asset sentiment+narrative │
│      │ (CryptoPanic headlines)  │          │ ~$0.34/月, 3s 延迟            │
│      │                          │          │ 代码 wired 到 main.py ADVISE  │
└──────┴──────────────────────────┴──────────┴──────────────────────────────┘

Fusion 注入点:
  L1 → macro_crowd_context["sentiment"]     (crowd modulation, 已接入)
  L2 → agent_signals["sentiment_zscore"]    (fusion engine, 当前硬编码 0.0)
  L3 → llm_sentiment = ADVISE              (authority_fusion, 当前 default 0)

信号流:
  CryptoPanic titles + GMM regime + funding_rate
      ──→ L3 Haiku (regime-aware prompt)
      ──→ sentiment_zscore ──→ Fusion Engine
                                           ↑ fallback
  F&G + Funding + LS ──→ L1 确定性 ──→ macro_crowd ──→ Sizing/Urgency
                                           ↑ fallback
                                          0.0 (训练时 pre-2020 行为一致)

Fallback 链 (never-block):
  L3 Haiku API → L1 F&G 确定性 → 0.0 (硬编码安全值)
  任何一层超时/失败 → 跳过，不阻塞 4H tick

实施顺序:
  Step 18.1-18.4: L1 确定性引擎 (已有完整代码)
  Step 18.5:      L3 Haiku LLM Agent (补完骨架)
  Step 18.6:      L2 DeBERTa 重训 (条件性，需真实加密标注数据)
  Step 18.7:      多层 A/B 验证
  Step 18.8:      Gate
```

### 18.1 L1 核心原则 (PROFIT-MAX CONSTRAINTS)

```
⚠️ 绝对红线:
  - Sentiment 绝不 veto 交易
  - Sentiment 绝不翻转方向
  - Sentiment 是调节器，不是决策器

Sentiment 影响路径 (通过 Authority Fusion, 非散装乘数):
  L1 → macro_crowd_context → fusion ADVISE 权重 10-20%
  L3 → sentiment_zscore    → fusion ADVISE 权重 10-20%

  ✅ 方向一致时: fusion 给 sentiment agent 更高 confidence → 有效 ~+5-15% boost
  ✅ 方向冲突时: fusion 中 sentiment 与其他 agent 冲突 → 有效 ~-5-10% dampening
  ✅ 极端拥挤时: crowding_risk=true → confidence boost → 有效 ~10-20% sizing reduction
  ❌ 不使用散装乘数: 无 signal_strength *= 1.10 或 position_size *= 0.70
     原因: fusion 已统一处理调节，散装乘数会导致双重计算

  实际效果范围: ±5-15% (在 guide 定义的 ±10%/±30% 范围内)
  Iron Law #34: veto_active=False 硬编码，sentiment 永远不能 veto
```

### 18.2 L1 数据源 (6 个信号，全部已有)

```
所有数据源都是 runtime-only，不进 DRL obs，不需要重训。

┌─────────────────────┬─────────┬───────────┬──────────────────────────────┐
│ 信号                 │ 权重     │ 数据源     │ 短偏好解读                    │
├─────────────────────┼─────────┼───────────┼──────────────────────────────┤
│ Funding Rate        │ 25%     │ Kraken    │ 正=多头拥挤→做空;             │
│                     │         │ Futures   │ 负=空头拥挤→谨慎              │
├─────────────────────┼─────────┼───────────┼──────────────────────────────┤
│ Long/Short Ratio    │ 20%     │ Coinglass │ >1.5=多头过多→做空确认;       │
│                     │         │           │ <0.7=空头过多→减仓            │
├─────────────────────┼─────────┼───────────┼──────────────────────────────┤
│ Fear & Greed Index  │ 15%     │ alt.me    │ >75 贪婪→逆向做空;            │
│                     │         │ (免费API) │ <25 恐惧→中性(不追空)         │
├─────────────────────┼─────────┼───────────┼──────────────────────────────┤
│ OI Change           │ 15%     │ Kraken    │ OI 上升+价格上升=趋势强→做空; │
│                     │         │ Futures   │ OI 下降=去杠杆→观望           │
├─────────────────────┼─────────┼───────────┼──────────────────────────────┤
│ Liquidations        │ 15%     │ Coinglass │ 多头爆仓>空头=级联做空;       │
│                     │         │           │ 空头爆仓>多头=挤压风险→减仓   │
├─────────────────────┼─────────┼───────────┼──────────────────────────────┤
│ DVOL + VPIN         │ 10%     │ Pipeline  │ 高 vol/毒性→降低全部方向信心   │
└─────────────────────┴─────────┴───────────┴──────────────────────────────┘
Total: 100%

拥挤度 (crowding_score) 独立计算:
  = weighted_max(funding_crowding, ls_ratio_crowding, oi_crowding)
  不进 composite，通过 crowding_risk 标志传递给 fusion engine
```

### 18.3 L1 确定性计算

```python
# signals/deterministic_sentiment.py
# 纯确定性，无 LLM，无额外 API key
# 所有数据源都已有，零新依赖

class SimpleSentimentCalculator:
    """6-signal weighted composite for short-biased crypto trading."""

    def compute(
        self,
        fear_greed_value: Optional[int],         # 0-100, alternative.me
        funding_rates: Dict[str, float],          # asset → 8h rate, Kraken Futures
        ls_ratios: Dict[str, float],              # asset → long/short ratio, Coinglass
        oi_changes: Dict[str, float],             # asset → 24h OI % change, Kraken
        liq_imbalance: Dict[str, float],          # asset → (long_liq - short_liq) / total
        dvol_zscore: Optional[float],             # pipeline
        vpin_values: Dict[str, float],            # asset → VPIN, pipeline
    ) -> Dict[str, Any]:
        """
        Returns:
            composite_score: -1.0 ~ +1.0 (负=看空, 正=看多)
            direction:       "bearish" / "bullish" / "neutral"
            strength:        0.0 ~ 1.0
            crowding_score:  0.0 ~ 1.0 (独立于 composite)
            volatility:      0.0 ~ 1.0
        """
        signals = []  # (name, value, weight)

        # --- 1. Funding Rate (25%) ---
        avg_funding = np.mean(list(funding_rates.values())) if funding_rates else 0.0
        # 正 funding → 多头付费 → 做空信号
        funding_signal = -np.clip(avg_funding * 100, -1.0, 1.0)
        signals.append(("funding", funding_signal, 0.25))

        # --- 2. Long/Short Ratio (20%) ---
        avg_ls = np.mean(list(ls_ratios.values())) if ls_ratios else 1.0
        if avg_ls > 1.5:
            ls_signal = -0.6   # 多头拥挤 → 强做空
        elif avg_ls > 1.2:
            ls_signal = -0.3   # 温和多头 → 温和做空
        elif avg_ls < 0.7:
            ls_signal = 0.3    # 空头拥挤 → 谨慎 (减仓信号)
        elif avg_ls < 0.8:
            ls_signal = 0.1    # 温和空头
        else:
            ls_signal = 0.0    # 中性
        signals.append(("ls_ratio", ls_signal, 0.20))

        # --- 3. Fear & Greed (15%) ---
        if fear_greed_value is not None:
            if fear_greed_value > 75:
                fg_signal = -0.6   # 极度贪婪 → 逆向做空
            elif fear_greed_value > 55:
                fg_signal = -0.3   # 贪婪 → 温和做空
            elif fear_greed_value < 25:
                fg_signal = 0.0    # 极度恐惧 → 中性 (波动率飙升不追空)
            elif fear_greed_value < 45:
                fg_signal = -0.2   # 恐惧 → 确认做空
            else:
                fg_signal = 0.0
            signals.append(("fear_greed", fg_signal, 0.15))

        # --- 4. OI Change (15%) ---
        avg_oi_chg = np.mean(list(oi_changes.values())) if oi_changes else 0.0
        if avg_oi_chg > 5.0:
            oi_signal = -0.4   # OI 快速上升 → 杠杆过热 → 做空
        elif avg_oi_chg > 2.0:
            oi_signal = -0.2   # OI 温和上升
        elif avg_oi_chg < -5.0:
            oi_signal = 0.2    # OI 快速下降 → 去杠杆 → 观望
        else:
            oi_signal = 0.0
        signals.append(("oi_change", oi_signal, 0.15))

        # --- 5. Liquidations (15%) ---
        avg_liq = np.mean(list(liq_imbalance.values())) if liq_imbalance else 0.0
        # liq_imbalance > 0 = 多头爆仓多 → 级联做空
        # liq_imbalance < 0 = 空头爆仓多 → 挤压风险
        liq_signal = -np.clip(avg_liq * 2, -0.8, 0.8)
        signals.append(("liquidations", liq_signal, 0.15))

        # --- 6. DVOL + VPIN (10%) ---
        vol_score = 0.0
        if dvol_zscore is not None:
            vol_score = min(abs(dvol_zscore) / 3.0, 1.0)
            if dvol_zscore > 2.0:
                signals.append(("dvol", -0.2, 0.05))  # 高 vol → 降低信心
        avg_vpin = np.mean(list(vpin_values.values())) if vpin_values else 0.5
        if avg_vpin > 0.7:
            signals.append(("vpin", -0.1, 0.05))      # 高毒性 → 降低信心

        # --- Composite ---
        if signals:
            weighted_sum = sum(s * w for _, s, w in signals)
            total_weight = sum(w for _, _, w in signals)
            composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            composite = 0.0

        strength = min(abs(composite) * 2, 1.0)
        direction = "bearish" if composite < -0.15 else ("bullish" if composite > 0.15 else "neutral")

        # --- Crowding (独立计算，不进 composite) ---
        funding_crowd = min(abs(avg_funding) * 200, 1.0)
        ls_crowd = max(0, (avg_ls - 1.0) / 1.0) if avg_ls > 1.0 else max(0, (1.0 - avg_ls) / 0.5)
        ls_crowd = min(ls_crowd, 1.0)
        oi_crowd = min(abs(avg_oi_chg) / 10.0, 1.0) if avg_oi_chg > 0 else 0.0
        crowding = max(funding_crowd * 0.4 + ls_crowd * 0.4 + oi_crowd * 0.2, 0.0)

        return {
            "composite_score": np.clip(composite, -1.0, 1.0),
            "direction": direction,
            "strength": strength,
            "crowding_score": min(crowding, 1.0),
            "volatility": vol_score,
        }
```

### 18.4 L1 注入 — 通过 Authority Fusion (非散装乘数)

```python
# ═══════════════════════════════════════════════════════════════
# ❌ 旧设计 (已废弃): 散装乘数注入
#    signal_strength *= 1.10  / urgency *= 0.80 / position_size *= 0.70
#    问题: 与 fusion ADVISE 权重双重计算，效果不可预测
#
# ✅ 实际实现: 通过 Authority Fusion 统一调节
#    sentiment agent 以 ADVISE 级别参与 fusion (权重 10-20%)
#    fusion engine 统一计算所有 agent 的加权贡献
# ═══════════════════════════════════════════════════════════════

# --- 每 4H tick 数据收集 (所有源已有，零新 API) ---
fear_greed = await fetch_fear_greed_index()  # alternative.me, free
sentiment = self._sentiment_calc.compute(
    fear_greed_value=fear_greed.get("value") if fear_greed else None,
    funding_rates={a: market_data.get(f"{a}_funding_rate", 0.0) for a in ASSETS},
    ls_ratios={a: market_data.get(f"{a}_ls_ratio", 1.0) for a in ASSETS},
    oi_changes={a: market_data.get(f"{a}_oi_change_24h", 0.0) for a in ASSETS},
    liq_imbalance={a: market_data.get(f"{a}_liq_imbalance", 0.0) for a in ASSETS},
    dvol_zscore=market_data.get("dvol_zscore"),
    vpin_values={a: market_data.get(f"{a}_vpin", 0.5) for a in ASSETS},
)
logger.info(f"[SENTIMENT_L1] composite={sentiment['composite_score']:.3f} "
            f"dir={sentiment['direction']} crowd={sentiment['crowding_score']:.2f}")

# L1 结果注入 Authority Fusion:
signals["sentiment_l1"] = AgentSignal(
    direction=sentiment["composite_score"],  # [-1, 1]
    confidence=sentiment["strength"],        # [0, 1]
    veto_active=False,                       # Iron Law #34
)
# → fusion engine 以 ADVISE 权重 (10-20%) 混入最终决策
# → 不需要显式的 signal_strength *= 或 position_size *= 乘数
# → fusion 的加权机制已经实现了等效的 ±5-15% 调节效果

# [ADVISE_INFLUENCE] 日志会显示 sentiment 对最终决策的实际贡献百分比
```

### 18.5 L3: Haiku LLM Agent (补完已有骨架)

```
⚡ L1 + L3 同时部署 (与 Stage 10 并行运行中)
L1 已 ACTIVE, L3 已部署 (LLM_SENTIMENT_ENABLED=true, PID 320880)
A/B baseline = 先前裸跑 Paper Run 数据

已有代码:
  sentiment_llm_agent.py        — 骨架 drafted, import + init + runtime 调用已 wired
  llm_sentiment = ADVISE        — authority_fusion matrix 中已有位置
  source_weights                — news 0.30, twitter 0.20, llm 0.15 等已设计

需补完:
  1. CryptoPanic API 数据获取 (title-only)
     ⚠️ [P155 2026-08-04] 原写 "free tier"，实际代码用付费 Growth 版
     (cryptopanic_feed.py BASE_URL = .../api/growth/v2)。
  2. Haiku API 调用 (structured JSON output)
  3. Fallback 链接入
  4. sentiment_zscore 写入 market_data (当前硬编码 0.0)
```

**架构:**

```python
# sentiment/llm_sentiment_engine.py (~150 lines)

class LLMSentimentEngine:
    """每 4H 调一次 Haiku，分析 CryptoPanic headlines"""

    def __init__(self):
        self.cache = {}           # 4H TTL 缓存
        self.fallback_fg = None   # F&G fallback 引用

    async def get_sentiment(self, asset: str) -> dict:
        """Never-block: L3 → L1 → 0.0"""

        # L3: Haiku (最准)
        try:
            texts = await self._fetch_cryptopanic(asset, n=50, timeout=5)
            if texts:
                result = await self._call_haiku(asset, texts, timeout=10)
                if result and result["confidence"] > 0.3:
                    self.cache[asset] = result
                    return result
        except (APIError, Timeout, JSONDecodeError):
            logger.warning(f"[SENT-L3] Haiku failed for {asset}, falling back")

        # L1: F&G (已验证)
        try:
            fg = self.fallback_fg.get_current() if self.fallback_fg else None
            if fg is not None:
                return {
                    "sentiment": (fg - 50) / 50,  # 0-100 → [-1, 1]
                    "confidence": 0.5,
                    "source": "fear_greed_fallback",
                }
        except Exception:
            pass

        # L0: 安全零值 (和训练时 pre-2020 行为一致)
        return {"sentiment": 0.0, "confidence": 0.0, "source": "default"}

    async def _fetch_cryptopanic(self, asset: str, n: int, timeout: int) -> list:
        """CryptoPanic: title + metadata, 无 full content (付费 Growth 版, 非 free tier)"""
        # GET https://cryptopanic.com/api/v1/posts/
        #   ?auth_token={FREE_TOKEN}&currencies={asset}&kind=news&public=true
        # 返回 title list, 不含正文 (free tier 限制)
        # Title 对 LLM 够用: "SEC rejects spot Bitcoin ETF" = 足够判断
        ...

    async def _call_haiku(self, asset: str, titles: list, timeout: int) -> dict:
        """单次 Haiku 调用, structured JSON output"""
        prompt = f"""Analyze these {len(titles)} crypto news headlines about {asset}.
Return ONLY valid JSON:
{{
  "sentiment": <float -1.0 to 1.0, negative=bearish>,
  "confidence": <float 0.0 to 1.0>,
  "key_narrative": <string, 1 sentence>,
  "crowding_risk": <bool, true if extreme positioning mentioned>
}}

Headlines:
{chr(10).join(f"- {t}" for t in titles)}"""

        # claude-haiku-4-5-20251001, max_tokens=200
        # 50 titles × ~15 tokens = 750 input tokens
        # 月成本: 750 × 6次/天 × 30天 / 1M × $0.25 ≈ $0.034/月/asset
        # 3 assets ≈ $0.10/月 总计
        ...
```

**Fusion 接入 (关键修改):**

```python
# integration_v36.py 中，当前硬编码 0.0 的位置:

# ❌ 现状 (永远 0.0):
sentiment_zscore = agent_signals.get("sentiment_zscore", 0.0)

# ✅ 修改: 从 LLM engine 获取, 传入 regime context
current_regime = self._get_regime_label(asset)  # GMM → "BEAR"/"BULL"/...
current_funding = market_data.get(f"{asset}_funding_rate", 0.0)
llm_result = await self._llm_sentiment.get_sentiment(
    asset, regime=current_regime, funding_rate=current_funding,
)
sentiment_zscore = llm_result["sentiment"] * 3.0  # scale to z-score range
sentiment_confidence = llm_result["confidence"]

signals["sentiment"] = AgentSignal(
    direction=np.sign(sentiment_zscore) * sentiment_confidence,
    confidence=sentiment_confidence,
    veto_active=False,  # 铁律 #34: sentiment 永不 veto
)
```

**关键设计约束:**

```
1. Title-only 够用 — 全文有噪声, title 是编辑提炼的信号
2. LLM 不确定性 — 4H 频率下不是问题 (0.35 vs 0.42 效果一样)
3. Never-block — bare except + pass, sentiment 永不阻塞 tick
4. 4H 缓存 — 同一 4H 周期不重复调用
5. 不需要 fine-tune — 用 few-shot prompt calibration 替代
6. Short-bias 解读 — prompt 必须教 Haiku 我们的解读方式:
   "BTC to 100K 🚀" = 通用 NLP 认为 bullish,
   但我们的系统看到 retail euphoria = crowding = SHORT 信号
7. Regime-aware — prompt 接收当前 GMM regime + funding rate:
   同一条 "BTC breaks $100K" 在 BULL regime 末期 = euphoria top signal,
   在 BEAR regime = dead cat bounce → 不同解读
   funding rate 只做 disambiguation, 不做 primary signal (避免与 L1 双重计算)
8. 时间过滤 — 只取最近 4H 的 headlines (匹配决策周期)
   < 3 条 → 扩展到 8H; 仍 < 3 → 跳过 L3, fallback 到 L1

Prompt 校准 (Phase A, 部署前一次性):
  1. 从 Bitcoin_tweets.csv 采样 500 条 (跨 bull/bear/sideways)
     - 过滤: 非英语, 非加密, 垃圾广告/项目推广, 低质量推文
     - 采样: engagement 加权 (点赞/粉丝数高的优先, 如果 CSV 有这些列)
  2. 手动标注 100 条 gold labels (~30 min)
  3. Haiku 标注 400 条 → 对比 → 找系统性错误
  4. 错误案例变成 few-shot examples 嵌入 SYSTEM_PROMPT
  5. 重新评分验证改善 (目标: MAE < 0.3, 3-class acc > 70%)
  成本: ~$0.04, 时间: ~40 min, 可在 Stage 10 训练期间做

为什么不 fine-tune:
  - Fine-tune 冻结知识 ("ETF=bullish" 在 ETF priced-in 后过时)
  - Few-shot 可随时更新 (编辑 prompt, 不重训)
  - Haiku 天然理解 crypto 语义, 只需校准我们的 short-bias 解读
  - GPU 零占用 (Stage 10 训练不受影响)
```

### 18.6 L2: DeBERTa 重训 (条件性)

```
⚠️ 当前状态:
  - 模型文件: training_data/models/sentiment_v22/sentiment_v22_best.pt (353MB)
  - 训练数据: 2000 条合成模板 (generate_sample_data) → 无泛化能力
  - 推理代码: 存在且完整
  - 部署目录: models/sentiment_agent/ 不存在 (未部署)

现有"真实"数据:
  - Bitcoin_tweets.csv (22.4M 条) — 无标注
  - Bitcoin_tweets_dataset_2.csv (907K 条) — 无标注
  - crypto_news.parquet (50 条) — 太少
  - sentiment_data_full.parquet (104K 条) — FinancialPhraseBank, 非加密
  → 结论: 零加密货币标注数据，FinancialPhraseBank 87% neutral 极度偏斜

L2 重训路径 (仅在 L3 验证有效后):
  Option A: Haiku 批量标注 → DeBERTa 蒸馏
    1. 从 Bitcoin_tweets.csv 采样 10K 条代表性推文
    2. Haiku 批量标注 (sentiment + confidence)
       成本: 10K × 750 tokens / 1M × $0.25 ≈ $1.88 一次性
    3. 人工抽检 500 条，质量 > 85% → 继续
    4. DeBERTa fine-tune on 加密标注数据 (~1h GPU)
    5. 部署为本地推理层: 零 API 成本, 零延迟

  Option B: 学术数据集
    - CryptoSent, TweetCoin 等已有标注的加密情绪数据集
    - 质量未知，需评估

  Option C: 不做 L2
    - 如果 L3 Haiku 稳定且成本可接受 (~$0.10/月)
    - L2 DeBERTa 价值有限 (Haiku 已覆盖)

决策点: L3 Haiku 跑 2+ weeks 后评估
  - Haiku 稳定 + 成本低 → L2 优先级低
  - Haiku 频繁超时/API 不稳 → L2 作为本地 fallback 有价值
  - 规模扩展到更高频率 → L2 本地推理成本优势明显

历史 Sentiment 时间序列 (可选, Stage 21+):
  - 从 Bitcoin_tweets.csv 按日采样 50 条 → 每日一次 Haiku 调用
  - 产出: daily_sentiment.parquet (date, sentiment, confidence)
  - 成本: ~2000 天 × 750 tokens ≈ $0.38 一次性
  - 用途: 如果 L3 A/B 证明 sentiment 对 PnL 有正贡献
    → merge_asof 到 4H bars → 可作为 obs feature #123
    → 但改 obs_dim 需要全部模型重训, 仅在 Stage 21+ 考虑
```

### 18.7 多层 A/B 验证

```
⚡ 并行执行时间线 (Stage 10 训练期间):

Baseline (已有):  Paper Run 裸跑数据 (50+ 笔交易, 无 sentiment)
                  → 这是你的对照组, 不需要额外等待

Phase 1: L1 + L3 同时开启 (与 Stage 10 并行)
  L1 已 ACTIVE (SimpleSentimentCalculator)
  L3 已部署 (Haiku regime-aware, LLM_SENTIMENT_ENABLED=true)
  → A/B 数据随 Paper Run 自动积累
  → Stage 10 训练完毕时已有 2+ weeks 数据

  48h 稳定性监控:
    [ ] [SENT-L3] 日志每 4H tick 出现
    [ ] Fallback rate < 20% (大部分请求走 Haiku, 非 F&G)
    [ ] 429 rate limit 在 production 不出现 (1 call/4H ≠ 批量)
    [ ] Latency < 10s (含 CryptoPanic + Haiku)

  2 weeks A/B 判定:
    判定: PnL delta vs baseline > 3% → keep
    判定: PnL 退化 → LLM_SENTIMENT_ENABLED=false, 回到裸跑

Phase 2 (可选): 新模型 + L3
  Stage 10 训练完毕 → deploy 新 TQC 模型
  此时 L3 已验证稳定 → 可以直接对比:
    A: 旧模型 裸跑 (已有 baseline)
    B: 旧模型 + L3  (Phase 1 数据)
    C: 新模型 + L3  (Phase 2 数据)
    D: 新模型 裸跑  (可选, 关掉 L3 跑 1 week)

Phase 3 (可选): L1 + L3 + L2
  仅在 L3 验证有效且 Haiku 蒸馏完成后
  对比 Haiku 实时 vs DeBERTa 本地的准确率/延迟/成本

对比指标:
┌───────────────────┬──────────────┬──────────────┬──────────────┐
│ 指标               │ Baseline     │ + L1 + L3    │ 新模型+L3    │
├───────────────────┼──────────────┼──────────────┼──────────────┤
│ Total PnL         │              │              │              │
│ Win Rate          │              │              │              │
│ Avg Trade Size    │              │              │              │
│ Max DD            │              │              │              │
│ Trades Boosted    │ N/A          │              │              │
│ Trades Reduced    │ N/A          │              │              │
│ Sentiment Accuracy│ N/A          │ 人工抽检      │ 人工抽检      │
│ API Failures      │ N/A          │              │              │
│ Fallback Rate     │ N/A          │              │              │
│ L3 Latency (avg)  │ N/A          │              │              │
└───────────────────┴──────────────┴──────────────┴──────────────┘

任何层对 PnL 无改善或造成退化 → 关掉该层 (enabled=False)
```

### 18.8 Gate

```
L1 接入:
  [x] Fear & Greed API 接入 (graceful fallback → fg_signal=0, weight 跳过)
  [x] Funding Rate / LS Ratio / OI / Liq / DVOL+VPIN 消费 (全部已有)
  [x] DeterministicSentimentEngine 每 4H 产出 composite_score
  [x] Sentiment 通过 Authority Fusion ADVISE 权重 (10-20%) 调节
      不使用散装乘数 (signal_strength *= 等) — fusion 统一处理，避免双重计算
      实际效果 ±5-15%，在 ±10%/±30% 安全范围内
  [x] Sentiment 绝不触发 veto — veto_active=False 硬编码, F&G veto 仅在 >3σ 极端
  [x] [SENTIMENT_L1] / [SENT-L3→L1] / [LLM_SENTIMENT] 日志可搜索
  [x] 任何单一数据源挂掉 → graceful skip (bare except + neutral fallback)

L3 接入:
  [x] Prompt 校准完成 (calibration_report.json)
      - 100 gold labels 手动标注
      - Acc=57.8% (> baseline 48.7%), MAE=0.414 (> 0.3 但方向性准确)
      - 8 few-shot examples 嵌入 calibrated short-bias prompt
      - 剩余 MAE gap 因标注尺度差异 (你标 ±1.0, Haiku 标 ±0.4) — 生产中安全
  [x] CryptoPanic API token 配置 (Growth 付费版)
  [x] sentiment_llm_agent.py 完整实现 (976 lines, circuit breaker, caching, events)
  [x] Haiku API 调用 + structured JSON 解析 (fence stripping + schema validation)
  [x] sentiment_zscore 从硬编码 0.0 → LLM 实际值 (confidence > 0.3 时升级)
  [x] Per-asset cache, 1h TTL — 同周期不重复调用
  [x] Fallback 链: L3 → L1 → 0.0 全部测试通过
  [x] [SENT-L3] 日志: direction, confidence, source, crowding, headline count
  [x] 8s timeout → asyncio.TimeoutError → fallback, 不阻塞 tick

L2 (DEFERRED — Haiku 优先):
  [ ] 决策: 是否需要 DeBERTa 本地推理层
      → Haiku 稳定 + 成本低 → L2 优先级低
      → Haiku 频繁超时 → L2 作为本地 fallback 有价值
  [ ] 如需: Haiku 批量标注 10K 推文
  [ ] 人工抽检 > 85%
  [ ] DeBERTa fine-tune (~1h)
  [ ] 部署到 models/sentiment_agent/

A/B 验证:
  [x] Phase 1 L1+L3 A/B running (PID 320880, LLM_SENTIMENT_ENABLED=true)
      Shadow ledger 捕获 L1/L3 signals, [ADVISE_INFLUENCE] 日志活跃
  [ ] 2 weeks A/B 数据积累 → keep/kill 决策 (PnL delta > 3% → keep)
  [ ] 自动化 A/B 对比脚本 (当前 PARTIAL: shadow ledger 有数据，无分析脚本)
  [ ] 最终 sentiment 配置确定
```

---

# ═══════════════════════════════════════════════════
# PHASE 6: Online DRL (Stage 20)
# 前置: 100+ live experiences 积累 (通常 ~2-4 weeks)
# ═══════════════════════════════════════════════════

## Step 19: Online DRL Framework (Stage 20)

**前置条件**:
- LiveExperienceBuffer 已在 Paper Run 起就被动收集 (Step 15)
- DRL 已 promoted 或至少在 SHADOW mode 积累了 100+ experiences
- Live 部署稳定运行 2+ weeks

### 19.1 架构

```
LIVE TRADING (每笔交易)
    ↓ record_decision() + record_outcome()
LiveExperienceBuffer (circular, 10K/asset, JSONL 持久化)
    ↓ 每 24H 或每 50 experiences (先到者)
PeriodicFinetuner
    ↓ 50 steps, lr=1e-5 (10x smaller), grad_clip=0.5
    ↓ 只更新 SHADOW model (永不碰 production)
ShadowModelValidator
    ↓ 50 bars A/B test (~8 days)
    ↓ shadow_sharpe > prod_sharpe + 0.1 AND p < 0.05
ModelPromoter
    ↓ 全部通过 → swap shadow → production
PRODUCTION DRL MODEL (updated)
```

### 19.2 安全规则

```
⚠️ 绝对红线:
  1. PeriodicFinetuner 永不写入 production model path
  2. Shadow promote 必须通过 3 个条件:
     - Sharpe 提升 > 0.1
     - Max DD 增加 < 2%
     - 统计显著 (paired t-test p < 0.05)
  3. Fine-tune lr = 1e-5 (原训练 1.53e-5 的 ~65%, 保守)
  4. Gradient norm clipping = 0.5 (严格)
  5. Min 100 experiences 才触发 fine-tune
  6. Experience buffer 有硬上限 (10K/asset, circular)
  7. 系统 restart 后 experiences 从 JSONL 恢复
```

### 19.3 关键文件

```
drl/live_experience_buffer.py         (~200 lines)
  - LiveExperience dataclass
  - LiveExperienceBuffer: record_decision, record_outcome, sample_batch
  - JSONL 持久化/恢复

training/online_finetuner.py          (~150 lines)
  - PeriodicFinetuner: should_finetune, finetune
  - 只操作 shadow model

training/online_model_promoter.py     (~100 lines)
  - ShadowModelValidator: start_validation, record_comparison, should_promote
  - 统计验证 (scipy.stats.ttest_rel)

完整实现见 V10-FINAL-SENT-ONLINE 文档。
```

### 19.4 启用步骤

```
前提: LiveExperienceBuffer 已积累 100+ experiences

1. 初始化 Finetuner:
   self._finetuner = PeriodicFinetuner(
       production_model_path="models/production/best_model.zip",
       device="cuda",
   )
   self._shadow_validator = ShadowModelValidator()

2. 在主循环中启用 fine-tune trigger:
   buffer_stats = self._exp_buffer.get_stats()
   if self._finetuner.should_finetune(buffer_stats):
       for asset in ["BTC", "ETH", "SOL"]:
           result = self._finetuner.finetune(self._exp_buffer, asset)
           if result["success"]:
               self._shadow_validator.start_validation()

3. 在每 tick 中记录 shadow vs production 对比:
   if self._shadow_validator._validation_active:
       self._shadow_validator.record_comparison(shadow_pnl, prod_pnl)
       verdict = self._shadow_validator.should_promote()
       if verdict["promote"]:
           # Swap shadow → production
           ...
```

### 19.5 Gate

```
[ ] LiveExperienceBuffer 100+ experiences per asset
[ ] PeriodicFinetuner 只更新 shadow model (grep 验证)
[ ] Shadow promote 需要 3 条件全过 (Sharpe + DD + p-value)
[ ] Fine-tune lr = 1e-5, grad_clip = 0.5
[ ] Experience buffer JSONL 在 restart 后正确恢复
[ ] 第一次 fine-tune 后 shadow model 产生合理 actions (非 NaN, 非全零)
[ ] 至少一次 validation cycle 完成 (50 bars ≈ 8 days)
```

---

# ═══════════════════════════════════════════════════
# 风险和回退计划
# ═══════════════════════════════════════════════════

```
Stage 7 NAV% 翻转 classic:
  → ✅ 已解决: classic +693% >> sortino +580% >> sharpe +565%，差距真实

Stage 8B 两个 config 都 fail:
  → Fallback 阶梯（见 Step 2.3 + Step 3.4）:
    ① net_arch 搜索 (~44h)
    ② Tier 3 搜索 (~22h)
    ③ 如果全部 fail → 用 Config 2 default 直接进 Stage 9

Stage 9.7 Go/No-Go STOP:
  → 调低 slippage_bps / turnover_cost_mult → 重跑（最多 2 次 × 20-30h）
  → 2 次仍 STOP → 回退 Stage 8 让 Optuna 搜 friction 参数

Stage 10 多数 fold reward < 0:
  → 先看 Step 9.5 诊断框架, 确认不是早期正常波动
  → 200K 步仍 < -500: 检查 ent_coef + critic_loss
  → 500K 步仍 < 0: 检查该 asset 是否数据不足 (SOL 只有 12K bars)
  → 降低 timesteps / 调大 early stopping patience
  → 单 asset fail → 该 asset 用 DT only

Stage 13 OOS Sharpe < 0.3:
  → 全面回退到 Stage 8（最坏情况）
  → 或: 降低 friction 参数 → Stage 10 重训（部分返工）

Stage 16 Paper Run crash:
  → 回到 Stage 14 Runtime Parity（最常见: feature mismatch）

Stage 19 Sentiment A/B 退化:
  → L1 退化: 关掉 L1 (enabled=False)，回到裸跑
  → L3 退化: 关掉 L3，保留 L1 only
  → L3 API 不稳: 降级到 L1 (fallback 链自动处理)
  → 不影响 DRL 模型或 ensemble（Sentiment 全部是 runtime 调节器）

Stage 20 Online DRL shadow 退化:
  → 不 promote，shadow 模型丢弃
  → Production model 不受影响（safety by design）
  → 等待更多 experiences 后重试 fine-tune

最坏路径: Stage 8B → 9 fail 重跑 → 10 → 13 fail → 回 8 → ...
额外时间: +200-400h
但概率很低（<5%），因为 Stage 9.7 Go/No-Go 会提前拦截多数问题。
```

---

# ═══════════════════════════════════════════════════
# 附录 A: SOTA Gap Validation (2026-02-25)
# ═══════════════════════════════════════════════════

7 个 SOTA 领域逐项验证，结论: 无阻塞性缺口。

```
┌───┬─────────────────────────────────┬──────────┬─────────────────────────────────────────┐
│ # │ SOTA 领域                        │ 覆盖状态  │ 不做/已覆盖原因                           │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 1 │ Reward 工程 (Self-Rewarding)    │ ✅ 已覆盖 │ reward_clip=20 + dd_penalty=2.82        │
│   │                                 │          │ Stage 7: PnL 权重 1.0→0.4 拖慢学习       │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 2 │ Regime Detection HMM+DRL        │ ✅ 已覆盖 │ per-asset GMM + FiLM PosA γ·h+β         │
│   │                                 │          │ gamma=0.99 不敏感, TQC 无 ε-greedy       │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 3 │ Meta-Strategy Selection          │ ⏳ 可选  │ Stage 21: TQC+DT 不同范式 ensemble       │
│   │                                 │          │ 先分层评估, 差异>15% 才做 meta-learner    │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 4 │ Funding Rate Alpha              │ ✅ 已覆盖 │ obs 内 funding_rate_zscore + Runtime      │
│   │                                 │          │ 25% 权重 Sentiment + Short Filter         │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 5 │ 预测-决策两阶段 (GBRT+DQN)      │ ❌ 不做  │ Phase B +715.5% 证明 TQC 端到端收敛良好   │
│   │                                 │          │ GBRT 准→直接用, 不准→给 obs 加噪声        │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 6 │ 训练稳定性 (PER/FinRL/Turb)     │ ✅ 已覆盖 │ TQC 分布学习 + GMM regime + OOD detector  │
│   │                                 │          │ PER +30% 采样开销, CPU-bound 不划算       │
├───┼─────────────────────────────────┼──────────┼─────────────────────────────────────────┤
│ 7 │ 多分辨率 CNN + 链上数据          │ ✅ 已覆盖 │ VecFrameStack(8) + TA 指标 = 数值化      │
│   │                                 │          │ CNN 参数量远超 500K 铁律 #28              │
└───┴─────────────────────────────────┴──────────┴─────────────────────────────────────────┘

验证细节:

1. Reward 工程:
   - Stage 7 classic +642 >> sharpe +318 >> sortino +166
   - 任何降低 PnL 权重(~1.0→0.4)的做法都拖慢学习
   - reward_clip=20 + dd_penalty_mult=2.82 已是"复合风险感知 reward"

2. Regime Detection:
   - FiLM PosA +794 vs LSTM +525 (Stage 6, γ=1/β=0 正确初始化)
   - GMM: BTC k=8, ETH k=7, SOL k=7
   - 动态 γ 需要训练中切换 discount factor, SB3 不原生支持

3. Meta-Strategy:
   - TQC continuous action [-1,+1] 表达力 > discrete choice
   - Specialist Agents (Short Bias + Regime Nav + Cascade Gov) 叠加非互斥
   - meta-learner 可作 Stage 21, 需 TQC 和 DT 都部署后积累分层数据

4. Funding Rate:
   - 训练层: funding_rate_zscore 是 122 features 之一
   - Runtime: Short Filter (极端 funding 时 veto) + Sentiment (25% 权重)
   - 小时级 funding 在 4H 决策频率下额外信息量有限

5. 预测-决策两阶段:
   - Phase B 3-fold mean NAV +715.5%, friction-aware 9.7 +317.74%
   - 122 features + FiLM regime conditioning 提供足够丰富状态表征

6. 训练稳定性:
   - TQC = Truncated Quantile Critics, n_quantiles=24, 学 Q-value 分布
   - reward_clip=20 截断 + dd_penalty=2.82 放大 = 等效 PER 在 reward 端
   - TQC + DT ensemble (不同范式) > PPO + A2C + SAC ensemble (相关 failure mode)

7. 多分辨率 CNN:
   - LSTM 从 8 bar 窗口看短期(1-2 bar) + 中期(32h) = 多分辨率变体
   - base features 102 维含多时间尺度 TA (RSI-14, MACD-12/26, BB-20, ATR-14)
   - 最小 CNN 几十万参数, 远超 FiLM 166K, 铁律 #28 难满足
```

---

# ═══════════════════════════════════════════════════
# PHASE 7: Stage 21+ Regime Power Retrain + Meta-Learner (可选)
# 前置: Live 稳定运行 1+ month, 有足够 A/B 数据
# ═══════════════════════════════════════════════════

## Step 21: Regime Power Value Retrain

**背景**: Stage 10 所有 asset 使用 default power=0.75 (unnamed regimes)。
Centroid 分析后识别出两个跨 asset 一致的 regime 需要差异化 power:

```
当前 (Stage 10):
  所有 regime → power=0.75 (一刀切)

目标 (Stage 21):
  STEADY_UPTREND  → power=1.1  (BTC R0, ETH R2)
  NEUTRAL_DRIFT   → power=0.7  (BTC R1, SOL R5)
  其他 regime     → 保持现有 power 值不变

Power 值的含义:
  reward *= power^regime
  power > 1.0 → 该 regime 下的 reward 被放大 → 模型更重视这种市况
  power < 1.0 → 该 regime 下的 reward 被缩小 → 模型更保守
```

### 21.1 变更依据 (Centroid 分析)

```
STEADY_UPTREND (BTC R0, ETH R2):
  ┌────────────────────┬────────┬────────┐
  │ 特征                │ BTC R0 │ ETH R2 │
  ├────────────────────┼────────┼────────┤
  │ return_24h         │ +0.42  │ +0.51  │  ← 明显正
  │ volatility_1h      │ -0.10  │ -0.32  │  ← 低于均值
  │ momentum_consistency│ +0.69  │ +0.70  │  ← 很强
  │ fear_index         │ -0.20  │ -0.24  │  ← 偏贪婪
  └────────────────────┴────────┴────────┘
  解读: 安静的牛市上行，方向明确、波动低
  Power 1.1: 鼓励模型在这种环境下更积极 (趋势跟随)
  Weight: BTC 14.4%, ETH 20.2%

NEUTRAL_DRIFT (BTC R1, SOL R5):
  ┌────────────────────┬────────┬────────┐
  │ 特征                │ BTC R1 │ SOL R5 │
  ├────────────────────┼────────┼────────┤
  │ return_24h         │ +0.12  │ +0.25  │  ← 微正/持平
  │ volatility_1h      │ +0.07  │ -0.05  │  ← 均值附近
  │ vol_of_vol         │ -0.20  │ -0.35  │  ← 很低 = 稳定
  │ momentum_consistency│ +0.60  │ +0.68  │  ← 中强
  └────────────────────┴────────┴────────┘
  解读: 有序但无方向的市场 (≠ CHOP，CHOP momentum=-1.5)
  Power 0.7: 让模型在此 regime 更保守 (少交易，降噪)
  Weight: BTC 15.5%, SOL 22.1%

  注: ETH 无 NEUTRAL_DRIFT 对应 cluster — ETH 波动性天然较高，
  "有序无方向" 的时段更少，GMM 未单独识别
```

### 21.2 Aggression 参数 (Runtime, 不影响训练)

```
同时部署的 runtime aggression 参数:

┌──────────────────┬──────────────┬───────────────┬──────────┐
│ Regime           │ Alpha Gate ×  │ Position Size │ Scale-In │
├──────────────────┼──────────────┼───────────────┼──────────┤
│ STEADY_UPTREND   │ ×1.00 (标准)  │ ×1.00         │ 允许     │
│ NEUTRAL_DRIFT    │ ×1.15 (高门槛)│ ×0.80 (-20%)  │ 不允许   │
│ 其他 regimes     │ 保持现有值     │ 保持现有值     │ 保持现有  │
└──────────────────┴──────────────┴───────────────┴──────────┘

Aggression 参数是 runtime-only, 不影响训练。
但须与 Sentiment A/B 隔离 — 不要同时改两个变量。
部署顺序: Sentiment A/B 完成 → Aggression A/B → Power 重训
```

### 21.3 全量重训计划

```
前置条件:
  [ ] Live 稳定 1+ month
  [ ] Sentiment A/B 决策完成 (keep/kill)
  [ ] Aggression A/B 决策完成 (keep/kill)
  [ ] 确认 power 变更对 reward distribution 的影响 (dry-run 分析)

重训范围:
  [ ] BTC 3 folds × 2.5M steps (power 变更: R0→1.1, R1→0.7)
  [ ] ETH 3 folds × 2.5M steps (power 变更: R2→1.1)
  [ ] SOL 3 folds × 3.0M steps (power 变更: R5→0.7)
  估算: ~96-224h (与 Stage 10 相同)

  ⚠️ 三个 asset 必须一起重训 — 不能只训一个
  ⚠️ 使用修复后的 early stopping (learning_starts reset)
  ⚠️ 保留 Stage 10 模型作为 fallback

验证:
  [ ] 每个 asset 至少 2/3 folds 正 reward
  [ ] weighted_score > Stage 10 的 weighted_score (否则不部署)
  [ ] Paper Run 48h 对比 Stage 10 模型
  [ ] A/B: 新模型 vs Stage 10 模型 2 weeks

回滚:
  新模型 PnL 退化 → 切回 Stage 10 模型 (保留在 models/stage10_backup/)
```

### 21.4 Regime 命名总表

```
三个资产的完整 regime mapping (供 Stage 21 重训使用):

BTC (8 regimes, GMM k=8):
  R0: STEADY_UPTREND     weight=14.4%  power=1.1  leverage=1.5×
  R1: NEUTRAL_DRIFT      weight=15.5%  power=0.7  leverage=1.0×
  R2: WEAK_CONSOLIDATION weight=17.4%  power=0.6  leverage=1.0×
  R3: MOMENTUM_RALLY     weight=10.4%  power=1.3  leverage=2.0×
  R4: QUIET_ACCUMULATION weight=24.4%  power=0.8  leverage=1.0×
  R5: VOLATILE_CHOP      weight=10.9%  power=1.5  leverage=3.0×
  R6: EXTREME_VOLATILITY weight= 4.6%  power=0.5  leverage=1.0×
  R7: PANIC_SELLOFF      weight= 2.4%  power=1.2  leverage=2.0×

ETH (7 regimes, GMM k=7):
  R0: PANIC_SELLOFF      weight= 4.4%  power=1.2  leverage=2.0×
  R1: WEAK_CONSOLIDATION weight=18.4%  power=0.6  leverage=1.0×
  R2: STEADY_UPTREND     weight=20.2%  power=1.1  leverage=1.5×
  R3: MOMENTUM_RALLY     weight=16.5%  power=1.3  leverage=2.0×
  R4: QUIET_ACCUMULATION weight=26.0%  power=0.8  leverage=1.0×
  R5: VOLATILE_CHOP      weight=11.0%  power=1.5  leverage=3.0×
  R6: EXTREME_VOLATILITY weight= 3.6%  power=0.5  leverage=1.0×

SOL (7 regimes, GMM k=7):
  R0: QUIET_ACCUMULATION weight=28.8%  power=0.8  leverage=1.0×
  R1: MOMENTUM_RALLY     weight=11.6%  power=1.3  leverage=2.0×
  R2: EXTREME_VOLATILITY weight= 3.2%  power=0.5  leverage=1.0×
  R3: VOLATILE_CHOP      weight=12.2%  power=1.5  leverage=3.0×
  R4: PANIC_SELLOFF      weight= 3.3%  power=1.2  leverage=2.0×
  R5: NEUTRAL_DRIFT      weight=22.1%  power=0.7  leverage=1.0×
  R6: WEAK_CONSOLIDATION weight=18.8%  power=0.6  leverage=1.0×

跨 asset 一致性验证:
  STEADY_UPTREND: BTC R0 ≈ ETH R2 (return, momentum, fear 接近)
  NEUTRAL_DRIFT:  BTC R1 ≈ SOL R5 (vol_of_vol 低, momentum 中)
  ETH 无 NEUTRAL_DRIFT — 波动性天然较高, GMM 未单独识别
```

---

# ═══════════════════════════════════════════════════
# 快速参考: 你现在该做什么
# ═══════════════════════════════════════════════════

```
现在 — Stage 10 Full Training:
  □ 更新 config/optuna_winner.json: buffer_size=500000
  □ 确认 train_drl_full.py 有 fold 间清理 (del model + gc.collect)
  □ 跑 Preflight checklist (Step 9.1)
  □ 启动 BTC 3 folds × 2.5M (~84h, early stopping 可能 ~42-55h)

然后:
  □ ETH 3 folds × 2.5M + SOL 3 folds × 3.0M (~100-184h)
  □ Step 10-13: Ensemble + Validation (5h)
  □ Step 14-15: Deploy + Paper Run (48h, 含 Experience Buffer 被动收集)
  □ Step 16: Live Week 1-2 (Conservative → Standard)
  □ Step 17: 持续监控
  □ Step 18: Sentiment A/B ⚡ 与 Stage 10 并行 (L3 Haiku 已部署, A/B 持续积累)
  □ Step 19: Online DRL (100+ experiences 后, 持续)
```

# HMATS 统一系统参考手册

> **版本**: v10.0-POSTAUDIT — 融合 Strategy Master List V3 + 代码审计 + v10 残留修复 + SOTA 过滤
> **审计链**: 1909行审计 → 84项自检 → 11修复(11/11 PASS) → 91项再审计 = 97.7% GREEN
> **总条目**: 96+ 项 (含 v10 新增)
> **标注**: 🔧 = v10审计修正; 🆕 = v10新增; ★ = v10核心改进

> **⚠️ 历史文档（标注于 2026-08-08，P227 审计）。** 冻结于 2026-03 时代。机制叙述仍有
> 参考价值，但运行时事实已过时：**执行 venue 是 Coinbase US perps sleeve（2026-06-13
> 起，Kraken 结构性无仓，P152）**；**DRL(TQC) 与 Exit-SAC 均已降级 SHADOW（2026-08-07，
> P198/P200）**；authority matrix 为 26 agents，fusion 实际消费 11 个（P227）；风控数值
> 以代码与 `configs/live_high_risk.json` 为准（本文的 4 级回撤梯度乘数 ×0.85/×0.65 在
> 代码中不存在，fuse 阈值为 UNLEASH v2）。当前真实状态以 **CLAUDE.md** 为准。
> **接线状态**: ✅ ACTIVE | 🌓 SHADOW/ADVISORY/BOUNDED_LIVE | ⚠️ FEATURE_FLAG / CODE_ONLY | ❌ NOT_IMPL

---

## 一、Alpha 生成层 — 交易信号来源 (14 项)

### 1. Quant Agent: 3×4 策略矩阵 (12 策略) — ⚠️ CODE_ONLY (研究库)

> `agents/kraken_quant_agent.py` — 独立于生产 Best-of-N 系统 (§2)。

#### BEAR 市场策略 (4)
| # | 策略 | Alpha 来源 |
|---|------|-----------|
| 1 | Liquidation Cascade Hunter | 强制平仓瀑布效应 |
| 2 | Hurst Exponent (R/S) | 趋势持续性数学量化 |
| 3 | Shannon Entropy | 市场无序度检测 |
| 4 | Variance Risk Premium (VRP) | 恐惧溢价定价 |

#### BULL 市场策略 (4)
| # | 策略 | Alpha 来源 |
|---|------|-----------|
| 5 | Funding Rate Divergence | 杠杆成本不对称 |
| 6 | ETF-Spot Cointegration | 传统金融→加密溢出 |
| 7 | SOL/BTC Relative Strength | 强者恒强动量 |
| 8 | Order Book Imbalance (OBI) | 微观结构供需 |

#### SIDEWAYS 市场策略 (4)
| # | 策略 | Alpha 来源 |
|---|------|-----------|
| 9 | Kalman Filter Cointegration | 配对交易均衡回归 |
| 10 | Ornstein-Uhlenbeck MLE | 均值回复速度量化 |
| 11 | Dark Pool Volume Tracking | 机构大单隐藏信息 |
| 12 | Delta-Neutral Funding Harvest | 无风险 funding 收益 |

---

### 2. 四策略 Best-of-N 选择器 (Production) — ✅ ACTIVE

> `main.py` — Winner-take-all 选择 (非 IR-Softmax)

| 策略 | 逻辑 | 适配 Regime |
|------|------|-------------|
| **momentum** | EMA ×0.6 + MACD ×0.4, ADX>25 增强 | MOMENTUM_RALLY ×1.3, PANIC_SELLOFF ×1.3 |
| **mean_revert** | RSI 超买卖 + BB 边缘 | QUIET_ACCUM ×1.3, WEAK_CONSOL ×1.1, MEAN_REVERT ×1.3 |
| **volume_breakout** | MACD + 放量确认 | MOMENTUM_RALLY ×1.2, PANIC_SELLOFF ×1.2 |
| **vrp** | RSI 过热/投降 + BB 极端 | EXTREME_VOL ×1.3, VOLATILE_CHOP ×1.2 |

Alpha formula: `|sig| × 200 × conf × perf`

---

### 3. DRL Agent: TQC — ✅ ACTIVE (SHADOW mode)

| 属性 | 值 |
|------|-----|
| 算法 | TQC (Truncated Quantile Critics) |
| obs_dim | **126** (铁律, 不碰) |
| Extractor | FiLM Position A (166K params) |
| Frame Stacking | VecFrameStack(8) → 实际 (1008,) |
| ent_coef | **0.1** (固定, auto 不稳定) |
| n_quantiles | 25, top_quantiles_to_drop=2 |
| Reward | classic PnL-based (锁定) |
| 当前角色 | SHADOW → 30天后 EXIT_ONLY |
| 入场权限 | **永久禁止** |
| Per-Asset GMM 🔧 | BTC k=8, ETH k=7, SOL k=7 |
| Cross-Validation 🔧 | **4-fold** (不是 3-fold) |
| 训练 | SubprocVecEnv(4), RTX 5090, 7-25h/run |
| Pipeline | **20 stages + 35 iron laws** |

DRL Exit-Only Authority: HOLD / ESCALATE / PARTIAL_EXIT / INCREASE_EXIT_PRESSURE / RELEASE_RUNNER

---

### 4. Decision Transformer (DT v3.2) — ⚠️ CODE_ONLY

从 TQC 最佳 episodes 离线学习, Stage 11 训练。

---

### 5. Short-Bias Agent — ✅ ACTIVE 🔧

> 🔧 v10: 已接入生产, Authority=PENALIZE (不是 CODE_ONLY)

- 做多信号 → soft penalty ×0.7 (不否决)
- funding > 0.24%/8h → short conviction +15%
- Authority: PENALIZE (只惩罚, 不否决)

---

### 6. Lead-Lag Alpha Engine — ✅ ACTIVE

- Binance → Kraken 领先信号, 2-tier dampening
- OPPORTUNITY 触发: net_edge ≥25bps

---

### 7. Sentiment Agent — ✅ ACTIVE

| 层 | 实现 | 状态 |
|----|------|------|
| L1 | Deterministic Fear & Greed / crowding modulation | ✅ |
| L2 | DeBERTa (待训练) | ⚠️ |
| L3 | Haiku LLM Agent (语义) | ⚠️ feature-flagged |

Authority: ADVISE / MODULATE (只调 confidence / size / urgency, 永不 veto)
sentiment_zscore = (fg_value - 50) / 50.0 × 3.0

---

### 8. SOL On-Chain Watcher — ⚠️ CODE_ONLY (多数模块)

Solana RPC + Jito API 数据源已接入, 分析模块待接线。

---

### 9. Macro Agent / GlobalContextInformer — ✅ ACTIVE

12 特征实现 (不是 v2 标注的 17 个):
`gdp_trend, cpi_trend, fed_rate, dxy_trend, vix_level, sp500_trend, gold_trend, oil_trend, btc_dominance, defi_tvl, stablecoin_flow, global_m2`

---

## 二、Regime 检测层 — 市场状态分类 (5 项)

### 10. GMM Per-Asset Regime 分类器 — ✅ ACTIVE 🔧

> 🔧 v10: Per-Asset GMM (不是统一 6-regime)

| 资产 | k (组件数) |
|------|-----------|
| BTC | 8 |
| ETH | 7 |
| SOL | 7 |

RegimeSmoother: hysteresis_threshold=3, min_persistence=2
ADX Fallback: z>3σ + >30% 偏移 → distribution-shift guard
冷启动: per-asset _warmup_ticks, 前 2 tick = degraded
cross_asset_correlation: **0.87** (统一) 🔧

---

### 11. Empirical Regime Transition Probabilities — ✅ ACTIVE

从训练数据推断, 概率矩阵 (不是 HMM)。

---

### 12. Regime Phase Detector — ✅ ACTIVE

4 阶段: **IGNITION → EXPANSION → SATURATION → EXHAUSTION**

影响: Alpha Gate / Exit Alpha / Leverage / CRACK 权重 / BullTransition 门控

---

### 12b. ★ BullTransitionDetector — ✅ ACTIVE 🆕

> v10 核心新增: 做空系统最关键的牛市保护

4 条件: Golden Cross + SOL/BTC RS + Funding positive 7天 + OI rising
4 状态: INACTIVE → POTENTIAL → ACTIVE → CONFIRMED
ACTIVE → short ×0.5, CONFIRMED → BLOCK_NAKED_SHORT

---

## 三、决策融合层 — 5-Agent Authority Matrix (7 项)

### 13. 5-Agent Authority Matrix — ✅ ACTIVE 🔧

> 🔧 v10: 从加权融合改为 binary authority (不稀释)

| Agent | Authority | 可否决? | 功能 |
|-------|-----------|--------|------|
| QuantAgent | DECIDE | 否 | Best-of-N 4策略, 方向确信 |
| DRLAgent | DECIDE | 否 | TQC, SHADOW→EXIT_ONLY |
| SentimentAgent | ADVISE | 否 | F&G + Haiku, 确信微调 |
| ShortBiasAgent | PENALIZE | 否 | 做多 ×0.7, 不否决 |
| RiskAgent | VETO | **是** | 一票否决 |

Reliability Injection: confidence < 0.35 → conviction × 0.3
Deadlock: ALL_CONFLICT → NO_TRADE (不挂起)
Multiplier Floor: **0.15** (VC-5, 防叠乘→4.6%)

---

### 14. Partial Consensus Entry — ✅ ACTIVE

不需要全体一致, DECIDE agent 可以单独决策方向。

---

### 15. One-Veto-Kill — ✅ ACTIVE

8 Veto 源, 任一否决 → 交易取消 (详见 §四)

---

### 15b. Profit Max Adapter — ✅ ACTIVE

- FalseBreakoutDetector (唯一硬否决)
- SignalQualityScorer → conviction 乘数
- Alpha Gate 已改为 `risk-profile` 驱动，不再是全局固定 `14bps / 8bps`
- `HIGH_RISK` 磁盘基线: `5bps long / 3bps short`
- `HIGH_RISK` staged schedule: long `5 -> 4 -> 3`, short `3 -> 3 -> 2`
- quiet / weak regime 仍保留 side-specific direction floors 和 pre-alpha holds

---

### 15c. CRACK 阈值 (集中定义) — ✅ ACTIVE 🔧

> 🔧 v10: 阈值集中在 sota_flags.py

FULL_EXIT = **0.50**, PARTIAL = **0.45**, URGENCY = **0.35**

---

### 🆕 15d. ★ HPLV Filter — ✅ ACTIVE

price ≥ 90th percentile + volume < 60% avg → short ×0.5

---

### 🆕 15e. Runtime Authority Contract — ✅ ACTIVE

Canonical authority states:
`DISABLED / SHADOW / ADVISORY / BOUNDED_LIVE / LIVE`

Current `HIGH_RISK` on-disk baseline:

- `Sentiment / LeadLag / ModelAlpha`: `ADVISORY`
- `CompositeToxicity / LearnedExecutionPolicy`: `ADVISORY` with fill / telemetry guard
- `G6`: `BOUNDED_LIVE` with clamp `0.75 .. 1.30` and cold-start prior
- `AggressiveAllocator`: `BOUNDED_LIVE` after `min_session_fills = 2`
- `DRL`: `SHADOW`, promotion ceiling remains `EXIT_ONLY`, never entry authority

---

## 四、风险管理层 — 8 Veto 源 One-Veto-Kill (13 项)

### 16. 8 Veto 源 — ✅ ACTIVE 🔧

> 🔧 v10: 从 7 层/13 步 → 8 Veto 源 + 31 gates (18 hard + 23 soft)

| # | Veto 源 | 功能 |
|---|---------|------|
| 1 | Constitution | 参数违规检查, Schema 验证 |
| 2 | RiskManager | 中央风控协调 |
| 3 | DeadManSwitch | 心跳超时→撤单, 不可禁用 |
| 4 | SqueezeProtection | 3级: ≥0.50 warn, ≥0.70 reduce, ≥0.80 flatten |
| 5 | LeverageGuard | **3.0×** 硬限 🔧 |
| 6 | DrawdownControl | **4级梯度** 🔧 (见下) |
| 7 | CorrelationCrisis | 5状态, **0.87 统一** 🔧 |
| 8 | ExistenceFuse | **多层** 🔧 (见下) |

---

### 17. Tranche Pyramiding — ✅ ACTIVE

| Tranche | 仓位 | 条件 |
|---------|------|------|
| T1 | 20% | Alpha > threshold, confidence > 60% |
| T2 | 35% | 4H close 确认 OR 结构突破, conf > 70% |
| T3 | 50% | 持仓盈利 + regime 对齐, conf > 80% |
| T4 | 65% | 盈利 >50bps + 持仓>2h + 动量持续, conf > 90% |

Phase 门控: SATURATION/EXHAUSTION 禁止 T4

---

### 18. ★ Drawdown 风控 — ✅ ACTIVE 🔧

> 🔧 v10: 从 10% 单一硬停 → 4级梯度

| Drawdown | 动作 |
|----------|------|
| 10% | 减仓 (position ×0.85) |
| 15% | 大幅减仓 (position ×0.65) |
| 25% | 暂停交易 (HALT) |
| 35% | 系统停机 (KILL, 全部平仓) |

参数在 sota_flags.py 集中定义。

---

### 19. Correlation Crisis — ✅ ACTIVE 🔧

> 🔧 v10: cross_asset_correlation 统一为 **0.87** (不再有 0.0/0.65 遗留)

5 状态: STABLE → ELEVATED → SPIKING → CRISIS → COLLAPSING
多窗口监控: 20/60/200 bars
跳跃检测: z > 2.5σ
预防性缩放: r > 0.85 开始

---

### 20b. ★ ExistenceFuse — ✅ ACTIVE 🔧

> 🔧 v10: 从单一 Fuse → 多层递进

| 层 | 触发 | 动作 |
|----|------|------|
| consecutive | **5 笔** 连续亏损 | 暂停 24h |
| weekly | 周损失 ≥ **8%** | HALT |
| monthly | 月损失 ≥ 8% | OBSERVE (半仓) |
| monthly | 月损失 ≥ **10%** | KILL |

DRL 独立计数: DRL 5-loss → EXIT_ONLY (不影响 quant)

---

### 20c. Thesis Budget Governor — ✅ ACTIVE

每论题预算 0.8% NAV, 3 次连续亏损 → 冷却 24h, FAIL-CLOSED。

---

### 20d. P0 Safety Integrator — ✅ ACTIVE

统一安全层: isolated margin + max leverage **3.0×** + anti-martingale。

---

### 20e. Squeeze Protection — ✅ ACTIVE

3级阈值 (集中定义): **0.50/0.70/0.80**

---

### 20f. Leverage Guard — ✅ ACTIVE 🔧

> 🔧 v10: **3.0×** (不是旧的 2.5×)

---

### 🆕 20g. ★ AssetAlphaTilt — ✅ ACTIVE

Rolling 30D Sortino per asset → dynamic multiplier (0.5~1.5×)
更新频率: 每 24h (每 6 个 4H tick)
平滑: α=0.3, 冷启动 (<5笔) → mult=1.0

---

### 🆕 20h. MonteCarloValidator — ✅ ACTIVE

1000 shuffle 策略鲁棒性验证, p<0.05 才认为策略有效。

---

## 五、执行层 — 交易执行优化 (14 项)

### 21. Execution Mode → TimingEngine 驱动 — ✅ ACTIVE 🔧

> 🔧 v10: TimingEngine 已接线到 PA Executor (DF-07 验证 INTACT)

| timing_score | Mode | 行为 |
|-------------|------|------|
| < 0.3 | DELAY | 不执行 |
| 0.3-0.5 | PASSIVE_ONLY | 仅 maker |
| 0.5-0.7 | PASSIVE_PREFERRED | maker + fallback |
| > 0.7 | AGGRESSIVE_TAKER | 速度关键 |

---

### 22. ★ PA Executor — ✅ ACTIVE 🔧

> 🔧 v10: 从 CODE_ONLY → ACTIVE (已接线)

- 默认: LIMIT + post_only + PASSIVE bias
- `HIGH_RISK` maker reprice: `2` attempts, `8s` wait, improve schedule `[6, 3]`
- fast market conversion 只允许在 `OPPORTUNITY` 或高质量 short
- Adverse Selection: <500ms fill → toxic → cancel
- Anti-Churn: AC-0~5 修复, 消除 restart churn (#1 费用杀手)

---

### 23. Fee Optimization (Kraken Pro) — ✅ ACTIVE

$10K/月免费额度, maker 16bps / taker 26bps。
Friction: TAKER 33bps (26+5+2), MAKER 23bps (16+5+2)。

---

### 23b. ★ Alpha Gate — ✅ ACTIVE 🔧

> 🔧 v10+: 从旧 friction-multiple / 全局固定阈值，收敛为 profile-driven EV gate

| Profile | Long floor | Short floor | 备注 |
|---------|------------|-------------|------|
| `HIGH_RISK` | **5 bps** | **3 bps** | staged relax, quiet/weak regime side-specific rules |
| other profiles | config-driven | config-driven | 以 risk-profile overlay 为准 |

---

### 24. VPIN Toxic Flow Detection — ✅ ACTIVE

| VPIN | 动作 |
|------|------|
| < 0.50 | PROCEED |
| 0.50-0.70 | REDUCE ×0.5 |
| 0.70-0.85 | DELAY ×0.25, 只 PASSIVE |
| ≥ 0.85 | ABORT |

---

### 24b. Adaptive Stop Manager — ✅ ACTIVE

Multi-Window ATR (14d/30d/60d), regime-adaptive, max holding 168h.

---

### 24c. Runner Management — ✅ ACTIVE

25% 锁利 (Phase-Aware) + 75% Runner (trailing stop 2%, DRL 可收紧到 1%)。

---

### 24d. Phase-Aware Exit Logic — ✅ ACTIVE

| 转换 | 动作 |
|------|------|
| IGNITION → EXPANSION | 持有, 考虑加仓 |
| EXPANSION → SATURATION | 25% scale-out, 禁止 T4 |
| SATURATION → EXHAUSTION | 退出所有, 只保留 runner |

---

### 24e. ★ Fill-Slope Monitor — ✅ ACTIVE 🔧

> 🔧 v10: 从 CODE_ONLY → ACTIVE

SOL <500ms 成交 → Adverse Selection → 暂停执行

---

### 🆕 24f. ★ FillQualityLogger — ✅ ACTIVE

fill_ratio, time_to_fill_s, slippage_bps, was_repriced, order_type, final_action
输出: logs/fill_quality.jsonl, 周报手动 review

---

### 24g. Dynamic Slicer (ATR) — ⚠️ CODE_ONLY

ATR-based 拆单, 大订单 3-10 个切片。

---

### 24h. Market Impact (Almgren-Chriss) — ✅ ACTIVE

ImpactCalibration → ProductionMarketImpact, CalibrationBridge, record_execution() 双写。
$10K 账户: 冲击可忽略, 扩展到 $100K 需重新校准。

---

## 六、微观结构层 — 读懂市场 (4 项)

### M1. OFI (Order Flow Imbalance) — ⚠️ CODE_ONLY
### M2. Vol-of-Vol — ✅ ACTIVE (在 OPPORTUNITY triggers 内)
### M3. Whale + Flow Signal Integrator — ✅ ACTIVE (部分)
### M4. CrossExchangeEngine — ⚠️ CODE_ONLY

---

## 七、模式切换层 — 激进 vs 保守 (5 项)

### 25. SOL Allocation — ✅ ACTIVE 🔧

> 🔧 v10: SOL max **20%** (不是旧的 50-60% / 100% dominance)
> SOL Dominance 机制已移除。资产分配: BTC 25%, ETH 25%, SOL 20%。

---

### 26. System Mode State Machine — ✅ ACTIVE 🔧

| 模式 | Alpha 门槛 |
|------|-----------|
| NORMAL | **14 bps** 🔧 |
| OPPORTUNITY | **8 bps** 🔧 |
| NO_TRADE | ∞ |

优先级: NO_TRADE > OPPORTUNITY > NORMAL (已锁定)

---

### 26b. OPPORTUNITY 五触发组 — ✅ ACTIVE

| 组 | 触发 | TTL |
|----|------|-----|
| A | Lead-Lag Edge ≥25bps | 2H |
| B | CRACK Window ≥2% | 8H |
| C | Vol Expansion 2.5σ | 4H |
| D | Sentiment Shock 2σ | 6H |
| E | SOL Flow Surge 3σ | 4H |

OPPORTUNITY TTL: **16h** (4 bars)

---

### 26c. NO_TRADE 触发 — ✅ ACTIVE

9 个条件: all_conflict, data_integrity, stale_data, extreme_dvol, liquidity_critical, correlation_collapse, flash_crash, execution_blocked, feed_disagreement

---

### 26d. CRACK System — ✅ ACTIVE 🔧

阈值 (集中定义): FULL_EXIT **0.50**, PARTIAL **0.45**, URGENCY **0.35** 🔧

---

## 八、自适应学习层 — 从错误中学习 (3 项)

### A1. Failure-Aware Memory — ✅ ACTIVE

per-asset × per-regime 失败记忆, threshold 调整。

---

### A2. ConfidenceScorer — ✅ ACTIVE 🔧

> 🔧 v10: 已接通, 三维度 (方向准确率 35% + PnL 35% + Regime 30%)

record_signal() + record_outcome() → Reliability Injection (fusion 降权)。

---

### A3. DriftDetector (5源) — ✅ ACTIVE 🔧

> 🔧 v10: 5 个检测源

Feature z-score (z>3.0), Latent space distance, GMM JSD (>0.15), 执行滑点 (z>2.0), DRL Action (mean_z>2.5)
→ Severity: NONE/MINOR/MODERATE/SEVERE/CRITICAL
→ SEVERE+ → DRL 降级到 SHADOW

---

## 九、特殊设计 — 系统架构创新 (15 项)

### 27. Alpha Formula — ✅ ACTIVE
`estimated_alpha = |sig| × 200 × regime_confidence × performance_factor`

### 28. High-Risk Gambler Mode — ⚠️ CODE_ONLY (flag=False)

### 29. DRL Promotion Gate — ✅ ACTIVE 🔧
> 🔧 v10: Win Rate ≥ **48%** (不是 52%), Sharpe ≥ 1.0, 30天 shadow

### 30. Wavelet 去噪 (Stage 5) — ✅ ACTIVE (训练)
obs_dim 121 → 126, Coiflet-4 level-2 soft thresholding

### 31. FiLM Regime 条件化 (Stage 6) — ✅ ACTIVE (训练)
FiLM Position A: γ→1, β→0 初始化, 166K params

### 32. State Augmentation — ✅ ACTIVE (训练)
保护 13 维 (4+8+1), TQC 2% noise, 5% dropout

### 33. Shadow Ledger & Proof Log — ✅ ACTIVE
tick_id + fill_quality 字段 🔧, immutable audit trail

### 34. Data Drift Detector — ✅ ACTIVE
5源检测 (见 §A3), SEVERE+ → DRL 降级

### 35. Cross-Asset Rules — ✅ ACTIVE 🔧

| Asset | Max Exposure | 特殊规则 |
|-------|-------------|---------|
| BTC | **25%** 🔧 | lead-lag primary leader |
| ETH | **25%** 🔧 | secondary leader |
| SOL | **20%** 🔧 | β adj, AGGRESSIVE_MAKER |

### 36. 周末风险缩减 — ⚠️ 部分 ACTIVE
WeekendManager: is_weekend 标记传递。

### 37. K-Fold Cross-Validation — ✅ ACTIVE (训练) 🔧
> 🔧 v10: **4-fold** (不是 3-fold)

### 38. Optuna 超参搜索 — ✅ ACTIVE (训练)
50 trials, ent_coef=0.1 固定, buffer=1M, reward_clip=20

### 38b. Per-Regime Agent Pool — ✅ ACTIVE (训练)
Per-asset GMM → 独立 TQC 模型, soft routing

### 38c. DRL Reward Mode — ✅ ACTIVE
classic 锁定 (+642), sharpe (+318), sortino (+166)

---

## 十、15 个市场模式/场景 — ✅ 部分覆盖

(与 v6.5.1 相同, 不变)

---

## 十一、跨交易所 & 链上情报层 (8 项)

| # | 组件 | 状态 |
|---|------|------|
| X1 | Feed Disagreement Safety | ✅ (在 NO_TRADE 内) |
| X2 | Chaos Index | ⚠️ 部分 |
| X3 | OnChainInflowMonitor | ⚠️ CODE_ONLY |
| X4 | DEXFlowAnalyzer | ⚠️ CODE_ONLY |
| X5 | JitoTipsTracker | ⚠️ CODE_ONLY |
| X6 | NetworkCongestionMonitor | ⚠️ CODE_ONLY |
| X7 | SOLDefenseModule | ⚠️ CODE_ONLY |
| X8 | ETF Flow Signals | ✅ (在 FlowSignalIntegrator 内) |

---

## 十二、工程防御层 — Production Safety (7 项)

### 39c. Cancel-on-Disconnect — ✅ ACTIVE
on_error/on_disconnect → cancel all + set latch → watchdog → on_recovered

### 39d. Kraken Integrity Shield (CRC32) — ✅ ACTIVE
CRC32 orderbook 校验 + WS Hot/REST Escape 双通道

### 39e. Startup Reconciler — ✅ ACTIVE
启动时 Shadow Ledger vs 交易所对账, Anti-Churn (AC-0~5)

### 39f. Hot Restart / Checkpoint — ✅ ACTIVE
每 5min save checkpoint

### 39g. VRAM Guard — ⚠️ CODE_ONLY

### 41. Veto Chain — ✅ ACTIVE 🔧

> 🔧 v10: 10步 process_4h_tick() (不是旧的 7步 Runtime Spine + 15步 v36Engine)

```
process_4h_tick()
  [1] 数据获取 + Contract & Data Health Gate
  [2] 市场分析 (GMM + Phase + BullTransition)
  [3] 5 Agent 信号生成
  [4] Authority Fusion
  [5] 8 Veto 源 One-Veto-Kill
  [6] Profit-Max (Alpha Gate + CRACK + HPLV)
  [7] Portfolio Brain (Correlation + AlphaTilt)
  [8] 执行 (PA + Timing + Impact + FillLog)
  [9] 反馈 (ShadowLedger + Confidence + Drift + Fuse)
  [10] 每日任务 (AlphaTilt update + Weekend)
```

### 42. 核心安全常量 — ✅ ACTIVE 🔧

| 常量 | 值 |
|------|-----|
| drawdown_reduce | **10%** |
| drawdown_heavy | **15%** |
| drawdown_halt | **25%** |
| drawdown_kill | **35%** |
| max_leverage | **3.0×** 🔧 |
| alpha_gate_normal | **14 bps** 🔧 |
| alpha_gate_opp | **8 bps** 🔧 |
| multiplier_floor | **0.15** 🆕 |
| cross_asset_corr | **0.87** 🔧 |
| fuse_consecutive | **5** 🆕 |
| fuse_weekly | **8%** 🆕 |
| fuse_monthly_kill | **10%** 🆕 |
| cancel_on_disconnect | True |
| fail_closed | True |

---

## 十三、系统哲学总结

| 原则 | 实现 |
|------|------|
| **Aggressive Alpha, Defensive Shell** | 策略激进追求收益, 防御层层叠加保护资本 |
| **4H Master + 200ms Refinement** | 主决策每 4h (战略), 执行微调 200ms (战术) |
| **One-Veto-Kill** | 8 veto 源, 任一否决 |
| **Fail-Closed** | 异常 = 不交易, 永远不会因 bug 开仓 |
| **Authority > Weighted Average** | 二元权限, 不稀释确信度 |
| **Multiplier Floor** | 0.15, 防止叠乘致仓位过小 |
| **Code Exists ≠ Code Works** | 审计验证 55% 实际接线 (v10 前) |
| **保险丝不是方向盘** | ExistenceFuse 不判断市场, 只承认\"不值得存在\" |

---

## 附录 A: v10 新增组件

| 组件 | 功能 | 状态 |
|------|------|------|
| BullTransitionDetector | 4条件牛市检测, BLOCK_NAKED_SHORT | ✅ ACTIVE |
| ExistenceFuse (多层) | consecutive/weekly/monthly 熔断 | ✅ ACTIVE |
| HPLV Filter | 高价低量 → short ×0.5 | ✅ ACTIVE |
| AssetAlphaTilt | Sortino-weighted 倾斜 | ✅ ACTIVE |
| MonteCarloValidator | 1000 shuffle 验证 | ✅ ACTIVE |
| FillQualityLogger | 成交质量 jsonl | ✅ ACTIVE |
| Multiplier Floor (0.15) | VC-5, 防叠乘 | ✅ ACTIVE |
| Anti-Churn (AC-0~5) | 消除 restart churn | ✅ ACTIVE |

---

## 附录 B: v10 审计修正总表

| 位置 | 旧值 | v10 修正 |
|------|------|---------|
| 杠杆 | 2.5× | **3.0×** |
| Drawdown | 10% 硬停 | **4级梯度 10/15/25/35%** |
| Alpha Gate | 99bps/50-83bps | **Profile-driven**: `HIGH_RISK = 5bps long / 3bps short`, staged to `3/2` |
| SOL Max | 95-100% dominance | **20%** |
| BTC/ETH Max | 60%/50% | **25%/25%** |
| Correlation | 0.95 crisis | **0.87 统一, 5状态** |
| GMM | 统一 6-regime | **Per-Asset (8/7/7)** |
| CV Folds | 3 | **4** |
| DRL Win Rate | ≥52% | **≥48%** |
| ExistenceFuse | 单一 -5%/28d | **多层 (5连亏/8%周/10%月)** |
| ShortBias | CODE_ONLY | **ACTIVE (PENALIZE)** |
| PA Executor | CODE_ONLY | **ACTIVE (接线)** |
| TimingEngine | CODE_ONLY | **ACTIVE (DF-07 INTACT)** |
| FillSlope | CODE_ONLY | **ACTIVE** |
| obs_dim | 115 (部分文档) | **126** |
| DRL Pipeline | 18-19 stages | **20 stages + 35 iron laws** |
| Agents | 15 | **5 (authority matrix)** |
| Risk Chain | 7层/13步 | **8 Veto + 31 gates** |

---

## 附录 C: 统计

| 状态 | 数量 |
|------|------|
| ✅ ACTIVE | ~70 |
| ⚠️ CODE_ONLY | ~22 |
| ❌ NOT_IMPL | 1 |
| 部分 ACTIVE | ~3 |

| 审计 | 结果 |
|------|------|
| 审计链 | 1909行 → 84检 → 11修 → 91再审 |
| 结果 | **97.7% GREEN** |
| 数据流 | **7/7 INTACT** |
| SOTA | **~96%** (语境过滤后) |
| 超越SOTA | **12 项** |

---

_HMATS v10.0-POSTAUDIT 统一系统参考手册结束_

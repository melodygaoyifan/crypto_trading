# HMATS 系统架构文档 - Part 3
# 风险管理与状态机
# ═══════════════════════════════════════════════════════════════
# 版本: v10.1-POSTAUDIT (v6.8.0 sync)
# 日期: 2026年3月27日 (updated from Feb 28)
# 审计状态: 8 Veto 源验证 INTACT, 31 gates 已审计
# v6.8.0 变更: UNLEASH v2 阈值 + FIX-FUSE-AUTORECOVERY + Drawdown 5级梯度
# ═══════════════════════════════════════════════════════════════

## 本部分目录

1. [风险管理哲学](#风险管理哲学)
2. [8 Veto 源 — One-Veto-Kill](#8-veto-源--one-veto-kill)
3. [31 Gates 详解](#31-gates-详解)
4. [状态机协调 (6个)](#状态机协调-6个)
5. [状态机交互示例](#状态机交互示例)

---

## 风险管理哲学

```
1. 失败闭合 (Fail-Closed)
   - 异常或不确定时, 默认拒绝
   - 数据缺失 → NO_TRADE (不猜测)

2. 一票否决 (One-Veto-Kill)
   - 8 个 Veto 源, 任一否决终止决策
   - 无例外, 无覆盖

3. 叠乘保护 (Multiplier Floor)
   - 23 个 soft multipliers 可叠加
   - 最差理论值: 4.6%
   - Floor = 0.15 (15%), 防止过度削减

4. 多层递进 (Graduated Response)
   - Drawdown: 5级梯度 (8%→×0.75, 15%→×0.50, 20%→×0.25, 25%→×0.10, 35%→kill)
   - Squeeze: 3级 (warn→reduce→flatten)
   - Fuse: 3层 (10 consecutive→-15% weekly→-18% monthly) [UNLEASH v2]

5. 做空特化 (Short-Biased Protection)
   - 做空亏损无上限 → ExistenceFuse 保命
   - BullTransitionDetector → 牛市来临时停手
   - SqueezeProtection 3级 → 挤空时硬否决
```

---

## 8 Veto 源 — One-Veto-Kill

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    HMATS v10.0 风险管理金字塔                              ║
║                    8 Veto 源 + One-Veto-Kill                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║                         ┌──────────┐                                      ║
║                         │ 手动停止 │  Level 0                              ║
║                         └────┬─────┘                                      ║
║                              │                                            ║
║                    ┌─────────▼─────────┐                                  ║
║                    │  ① ExistenceFuse  │  Level 1                          ║
║                    │  weekly-8%        │                                   ║
║                    │  monthly-10%      │  ★ v10 多层增强                   ║
║                    │  consecutive-5    │                                   ║
║                    └─────────┬─────────┘                                  ║
║                              │                                            ║
║               ┌──────────────▼──────────────┐                             ║
║               │  ② DrawdownControl (4级)    │  Level 2                     ║
║               │  10%→×0.85  15%→×0.65       │                             ║
║               │  25%→HALT   35%→KILL        │                             ║
║               └──────────────┬──────────────┘                             ║
║                              │                                            ║
║          ┌───────────────────▼───────────────────┐                        ║
║          │  ③ LeverageGuard (3.0× 硬限)          │  Level 3               ║
║          │  Leverage = (long + |short|) / NAV    │                        ║
║          └───────────────────┬───────────────────┘                        ║
║                              │                                            ║
║     ┌────────────────────────▼────────────────────────┐                   ║
║     │  ④ Constitution + ⑤ RiskManager                 │  Level 4          ║
║     │  参数验证 + 中央风控协调                          │                   ║
║     └────────────────────────┬────────────────────────┘                   ║
║                              │                                            ║
║     ┌────────────────────────▼────────────────────────┐                   ║
║     │  ⑥ CorrelationCrisis (5状态)                    │  Level 5          ║
║     │  STABLE→ELEVATED→SPIKING→CRISIS→COLLAPSING     │                   ║
║     │  cross_asset_correlation = 0.87 (统一)          │                   ║
║     └────────────────────────┬────────────────────────┘                   ║
║                              │                                            ║
║     ┌────────────────────────▼────────────────────────┐                   ║
║     │  ⑦ SqueezeProtection (3级)                      │  Level 6          ║
║     │  score≥0.50→WARN  ≥0.70→REDUCE  ≥0.80→FLATTEN  │                   ║
║     └────────────────────────┬────────────────────────┘                   ║
║                              │                                            ║
║  ┌───────────────────────────▼───────────────────────────┐                ║
║  │  ⑧ DeadManSwitch                                      │  Level 7       ║
║  │  心跳超时 → 撤单 (refresh in try/except)              │                ║
║  │  生产中无法禁用                                        │                ║
║  └───────────────────────────────────────────────────────┘                ║
║                                                                           ║
║  ★ BullTransition Override (在 veto 链之后) [v10]:                       ║
║  │  ACTIVE → short ×0.5                                                  ║
║  │  CONFIRMED → BLOCK_NAKED_SHORT (只允许对冲)                           ║
║  │                                                                        ║
║  ★ HPLV Filter (在 ProfitMax 中) [v10]:                                 ║
║  │  price≥90th + volume<60% → short ×0.5                                 ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 各 Veto 源详解

#### ① ExistenceFuse (v10 多层增强)

做空系统最关键的保护 — 做空亏损理论上无上限。

```
┌─────────────────────────────────────────────────────────────┐
│  ExistenceFuse — 多层递进保护                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: 连续亏损 (Consecutive Loss)                       │
│  ├─ 触发: 5 笔连续亏损                                      │
│  ├─ 动作: 暂停 24h                                          │
│  └─ 恢复: 24h 后自动恢复                                    │
│                                                              │
│  Layer 2: 周损失 (Weekly Loss)                              │
│  ├─ 触发: 周损失 ≥ 8%                                       │
│  ├─ 动作: HALT (暂停交易)                                   │
│  └─ 恢复: 下周自动恢复                                      │
│                                                              │
│  Layer 3: 月损失 (Monthly Loss)                             │
│  ├─ 触发 1: 月损失 ≥ 8% → OBSERVE (半仓)                   │
│  ├─ 触发 2: 月损失 ≥ 10% → KILL (系统停机)                 │
│  └─ 恢复: 需要手动批准                                      │
│                                                              │
│  DRL 独立计数:                                              │
│  ├─ DRL 5 笔连续亏损 → EXIT_ONLY (不影响 quant)            │
│  └─ 与系统级 Fuse 独立运行                                  │
│                                                              │
│  API:                                                        │
│  ├─ on_trade_close(pnl) → (action, reason)                 │
│  ├─ action ∈ {NONE, OBSERVE, HALT, KILL}                   │
│  └─ 返回 reason 用于 ShadowLedger 记录                     │
└─────────────────────────────────────────────────────────────┘
```

#### ② DrawdownControl (4级梯度)

```
┌─────────────────────────────────────────────────────────────┐
│  DrawdownControl — 4级梯度 (v10 统一)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Drawdown 10% → 减仓 (position ×0.85)                      │
│  Drawdown 15% → 大幅减仓 (position ×0.65)                  │
│  Drawdown 25% → 暂停交易 (HALT, 不开新仓)                  │
│  Drawdown 35% → 系统停机 (KILL, 全部平仓)                  │
│                                                              │
│  注意: 25% 是硬暂停 (不是旧版的 10% 或 20%)                │
│  参数在 sota_flags.py 集中定义                              │
└─────────────────────────────────────────────────────────────┘
```

#### ③ LeverageGuard

```
硬性限制: 3.0× (不可覆盖)
计算: Leverage = (long_value + |short_value|) / NAV
超限: > 3.0× → 拒绝新仓位, 强制削减到 3.0×
参数来源: sota_flags.py MAX_LEVERAGE = 3.0
```

#### ④ Constitution

```
参数验证 + Schema 检查
Required keys 缺失 → NO_TRADE
Direction float→int 已修复 (FIX-P2)
fail-closed 已修复 (FIX-H5)
```

#### ⑤ RiskManager

```
中央风控协调器
聚合所有子控制器信号
任何子控制器否决 → 总否决
```

#### ⑥ CorrelationCrisis (5状态)

```
5 状态:
├─ STABLE     → 正常
├─ ELEVATED   → 观察 (scale_factor ~0.85)
├─ SPIKING    → 减仓 (scale_factor ~0.65)
├─ CRISIS     → 停止新开仓 (scale_factor ~0.30)
└─ COLLAPSING → 观察

cross_asset_correlation: 0.87 (统一, 不再有 0.0/0.65 遗留)
1,098 行完整实现 (已审计 LIVE)
```

#### ⑦ SqueezeProtection (3级)

```
做空最大风险: 挤空

score ≥ 0.50 → WARN (记录, 继续)
score ≥ 0.70 → REDUCE (削减空头仓位)
score ≥ 0.80 → FLATTEN (全部平仓)

检测: >5% pump + volume spike + OI surge
```

#### ⑧ DeadManSwitch

```
心跳监控 (refresh 在 try/except 内)
超时 → 撤单
生产中无法禁用
Cancel-on-Disconnect (断线撤单)
```

---

## 31 Gates 详解

v10 Veto Chain 审计发现: 实际有 31 个 kill/modulation 点, 不是表面上的 8 个。

```
┌─────────────────────────────────────────────────────────────┐
│  31 Gates 分解                                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  18 Hard Kills (任一触发 → 交易取消):                       │
│  ├─ 8 个 One-Veto-Kill 源 (上述)                            │
│  ├─ Alpha Gate (friction > edge)                            │
│  ├─ FalseBreakoutDetector (唯一 ProfitMax 硬否决)          │
│  ├─ NO_TRADE mode (9 个触发条件)                            │
│  └─ BullTransition CONFIRMED → BLOCK_NAKED_SHORT           │
│                                                              │
│  23 Soft Multipliers (叠乘效应):                            │
│  ├─ RegimePower (0~1.5×)                                    │
│  ├─ ShortBias penalty (×0.7)                                │
│  ├─ Reliability Injection (×0.3 if conf<0.35)              │
│  ├─ DrawdownGradient (×0.85/0.65)                           │
│  ├─ CorrelationScale (0.3~1.0)                              │
│  ├─ AssetAlphaTilt (0.5~1.5×)                               │
│  ├─ BullTransition ACTIVE (×0.5)                            │
│  ├─ HPLV Filter (×0.5)                                      │
│  ├─ SignalQualityScorer                                     │
│  └─ ... (其他 regime/phase/confidence multipliers)          │
│                                                              │
│  叠乘保护 (VC-5 修复):                                     │
│  ├─ 最差理论: 23 个 soft 全触发 → 仓位 4.6%               │
│  ├─ Floor = 0.15 (15%)                                      │
│  └─ 应用位置: sizing 最终阶段                               │
│                                                              │
│  3 个重复检查 (已去重, VC-3):                               │
│  ├─ V6 Short Filter ∩ P0 Short Block                        │
│  ├─ Risk Governor ∩ P0 Safety                               │
│  └─ Trade Gate ∩ Constitution (overlap)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 状态机协调 (6个)

HMATS 协调 **6 个主要状态机**:

### 1. 系统模式状态机 (3状态)

```
NO_TRADE > OPPORTUNITY > NORMAL (优先级已锁定)

→ NO_TRADE (触发条件):
  - all_conflict_flat
  - data_integrity_fail / stale_data (>10s)
  - extreme_dvol (>3σ)
  - liquidity_critical
  - flash_crash
  - ExistenceFuse HALT/KILL

→ OPPORTUNITY (触发条件):
  - CRACK window (清算级联)
  - lead_lag_edge
  - volatility_expansion (>2σ)
  - sentiment_shock (>2σ)
  - TTL: 16h (4 bars)
  - Alpha Gate: 8bps (vs NORMAL 14bps)

→ NORMAL:
  - 默认状态
  - OPP TTL 过期
  - NO_TRADE 条件解除
```

### 2. Regime Phase 状态机 (4阶段)

```
IGNITION → EXPANSION → SATURATION → EXHAUSTION → (repeat)

IGNITION (趋势点火):
├─ 早期趋势形成
├─ 仓位: 保守
└─ 止损: 宽 (ATR-based)

EXPANSION (趋势扩张):
├─ 动量建立, 趋势确认
├─ 仓位: 可以加仓
└─ 最佳交易阶段

SATURATION (趋势饱和):
├─ 接近极值
├─ 仓位: 减仓, 不开新仓
└─ Phase-Aware Exit 触发区域

EXHAUSTION (趋势耗尽):
├─ 反转即将来临
├─ 仓位: 平仓
└─ 为相反方向准备
```

### 3. ★ BullTransitionDetector 状态机 (4状态) [v10]

```
做空系统最关键的新增保护。

INACTIVE → POTENTIAL → ACTIVE → CONFIRMED

4 条件 (每个独立评估):
├─ C1: BTC Golden Cross (weekly MA50 > MA200)
├─ C2: SOL/BTC relative strength > 0
├─ C3: Funding positive 持续 7天
└─ C4: OI rising + liquidations falling

状态转换:
├─ 0 条件满足 → INACTIVE (正常做空)
├─ 1 条件满足 → POTENTIAL (记录, 不干预)
├─ 2 条件满足 → ACTIVE (short ×0.5)
└─ 3-4 条件满足 → CONFIRMED (BLOCK_NAKED_SHORT)

CONFIRMED 行为:
├─ 禁止新的裸空仓位
├─ 只允许对冲空头
├─ 现有空头: 加速止损
└─ 预期避免: -20~25% 最坏场景 (温和牛市 8 周)
```

### 4. ★ ExistenceFuse 状态机 (4状态) [v10]

```
NORMAL → OBSERVE → HALT → KILL

NORMAL:
├─ 正常交易
└─ on_trade_close() 更新计数

OBSERVE (月损失 ≥ 8%):
├─ 半仓交易
└─ 加强监控

HALT (周损失 ≥ 8% 或 连续 5 亏损):
├─ 暂停交易
├─ 不开新仓, 允许平仓
└─ 24h 或下周恢复

KILL (月损失 ≥ 10%):
├─ 系统停机
├─ 全部平仓
└─ 需要手动批准重启
```

### 5. DRL Authority 状态机 (4级别)

```
DISABLED → SHADOW → EXIT_ONLY → FULL

DISABLED: DRL 完全关闭, Authority=NONE
SHADOW:   DRL 运行但不采取行动, 信号记录到 shadow ledger
EXIT_ONLY: DRL 可以建议退出, 不能入场 (永久禁止入场)
FULL:     DRL 完全参与 (目标状态, 未达到)

晋升: StatisticalPromotionGate
├─ 30 天 shadow 干净运行
├─ Sharpe > 1.0
├─ Win rate > 48%
└─ 无重大 drift 检测

降级 (Drift → SHADOW):
├─ DriftDetector SEVERE+ → 自动降级
├─ ExistenceFuse 5-loss (DRL独立计数) → EXIT_ONLY
└─ 恢复: 30 天冷却 + 统计验证

永久禁止: DRL 入场权限
├─ 不能从 EXIT_ONLY 升级到有入场权限
├─ 架构决策: 防止 DRL 生成新风险
└─ DRL 只能帮助退出, 不能帮助入场
```

### 6. Correlation 状态机 (5状态)

```
STABLE → ELEVATED → SPIKING → CRISIS → COLLAPSING

STABLE:     r < 0.70, scale=1.0
ELEVATED:   r ∈ [0.70, 0.85), scale~0.85
SPIKING:    r ∈ [0.85, 0.92), scale~0.65
CRISIS:     r ≥ 0.92, scale~0.30, 停止新开仓
COLLAPSING: 相关性从高位快速下降, 观察

cross_asset_correlation 统一: 0.87
冷却期: 危机后 15 分钟
```

---

## 状态机交互示例

### 场景 1: 正常做空交易

```
当前状态:
├─ 系统模式: NORMAL
├─ Regime Phase: EXPANSION (ETH)
├─ BullTransition: INACTIVE (0 条件)
├─ ExistenceFuse: NORMAL
├─ DRL: SHADOW
└─ Correlation: STABLE

决策流程:
1. QuantAgent: SHORT ETH, confidence=0.75
2. ShortBias: SHORT → 不惩罚 ✓
3. Sentiment: F&G=32 (恐惧) → +0.05
4. Reliability: ConfidenceScorer=0.82 → 不降权 ✓
5. BullTransition: INACTIVE → 不干预 ✓
6. 8 Veto 源: 全部通过 ✓
7. Alpha Gate: alpha=38bps > 14bps → 通过 ✓
8. HPLV: price < 90th → 不触发 ✓
9. AssetAlphaTilt: ETH Sortino=1.2 → ×1.3
10. PA Executor: post_only limit order

结果: SHORT ETH, conviction=0.80×1.3=1.04 (cap 1.0)
```

### 场景 2: 牛市转换保护触发

```
当前状态:
├─ 系统模式: NORMAL
├─ BullTransition: ACTIVE (2条件: Golden Cross + Funding positive)
├─ ExistenceFuse: NORMAL
└─ 持有: SHORT BTC 15% NAV

决策流程:
1. QuantAgent: SHORT BTC (追加做空), confidence=0.68
2. BullTransition: ACTIVE → short ×0.5
3. 调整后 conviction: 0.68 × 0.5 = 0.34
4. Alpha Gate: alpha=34bps×0.5=17bps > 14bps → 勉强通过
5. 最终仓位: 大幅缩减

如果 BullTransition → CONFIRMED (3 条件):
→ BLOCK_NAKED_SHORT: 新空仓被完全阻止
→ 现有空头: 加速止损
→ 保护: 避免 -20~25% 最坏场景
```

### 场景 3: ExistenceFuse 级联触发

```
当前状态:
├─ 本周已亏损 7.5%
├─ 最近 4 笔连续亏损
└─ ExistenceFuse: NORMAL

事件: 第 10 笔亏损, 本周累计达到 15%

ExistenceFuse 响应:
1. consecutive_loss = 10 → SUSPEND (v6.8: auto-recovery after 24h cooldown)
   [UNLEASH v2] UL-6a: 从5放宽到10, 短期3-5连亏在牛市中正常
   [FIX-FUSE-AUTORECOVERY] v6.8: 从manual恢复改为24h自动恢复
2. weekly_loss = 15% → HALT (暂停到下周)
   [UNLEASH v2] 从8%放宽到15%
3. monthly_loss 检查: 假设 = 18% → KILL
   [UNLEASH v2] 从10%放宽到18%

取最严格: HALT/SUSPEND
├─ 暂停所有新交易
├─ 允许平仓现有仓位
├─ ShadowLedger 记录原因
└─ 24h cooldown后自动恢复 (仅consecutive_loss触发的暂停)
```

### 场景 4: 叠乘效应 (VC-5 修复前后对比)

```
假设多个 soft multiplier 同时触发:

├─ RegimePower: ×0.7 (VOLATILE_CHOP)
├─ ShortBias: ×0.7 (做多信号)
├─ DrawdownGradient: ×0.85 (drawdown 10%)
├─ CorrelationScale: ×0.65 (SPIKING)
├─ Reliability: ×0.3 (confidence < 0.35)
└─ HPLV: ×0.5

修复前 (VC-5):
position = base × 0.7 × 0.7 × 0.85 × 0.65 × 0.3 × 0.5
         = base × 0.040 (4.0%)
→ 几乎无法交易

修复后 (floor = 0.15):
position = max(base × 0.040, base × 0.15)
         = base × 0.15 (15%)
→ 仍然大幅削减, 但不至于无法交易
```

---

**文档第3部分结束**

继续阅读：
- Part 4: 执行层与DRL
- Part 5: 运维与附录

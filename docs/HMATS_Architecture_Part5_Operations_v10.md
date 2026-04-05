# HMATS 系统架构文档 - Part 5
# 运维手册与附录（最终部分）
# ═══════════════════════════════════════════════════════════════
# 版本: v10.1-POSTAUDIT (v6.8.0 sync)
# 日期: 2026年3月27日 (updated from Feb 28)
# 审计状态: 97.7% GREEN, 7/7 data flows INTACT
# v6.8.0 变更: FIX-API-VALIDATE, FIX-DMS-HALT, FIX-NAN-GUARD, FIX-RECONCILE-LIVE
# ═══════════════════════════════════════════════════════════════

## 本部分目录

1. [日常运维](#日常运维)
2. [紧急程序](#紧急程序)
3. [配置参考 (sota_flags.py 集中)](#配置参考)
4. [术语表 (v10 更新)](#术语表)
5. [总结](#总结)

---

## 日常运维

### 每日检查清单

```bash
# 1. 检查服务状态
sudo systemctl status hmats

# 2. 审查过夜日志
sudo journalctl -u hmats --since "24 hours ago" | grep -i error

# 3. 检查 ShadowLedger 报告
cat /var/log/hmats/shadow_ledger/$(date +%Y-%m-%d).json

# 4. 验证仓位与交易所匹配
python /opt/hmats/scripts/reconcile_positions.py

# 5. 检查 Drawdown 级别 (4级梯度)
python /opt/hmats/scripts/check_drawdown.py

# 6. 检查 ExistenceFuse 状态
python /opt/hmats/scripts/check_fuse_status.py
# → consecutive_loss, weekly_loss, monthly_loss

# 7. 检查 BullTransition 状态
python /opt/hmats/scripts/check_bull_transition.py
# → INACTIVE/POTENTIAL/ACTIVE/CONFIRMED + 4条件状态

# 8. 审查 Fill Quality (v10)
tail -20 /var/log/hmats/fill_quality.jsonl | jq .
```

### 周度运维

```bash
# 1. Fill Rate 周报 (v10)
python /opt/hmats/scripts/fill_quality_weekly.py
# → maker_ratio, avg_slippage, timeout_rate

# 2. 审查周度性能
python /opt/hmats/scripts/weekly_performance.py

# 3. 检查 DRL drift (如果 DRL 活跃)
python /opt/hmats/scripts/check_drl_drift.py --window 7d

# 4. 清理旧日志 (保留 30 天)
find /var/log/hmats -type f -mtime +30 -delete

# 5. 数据库维护
sqlite3 /var/lib/hmats/state.db "VACUUM; ANALYZE;"

# 6. 重启服务 (如果无开放仓位)
python /opt/hmats/scripts/check_positions.py
# 如果 flat:
sudo systemctl restart hmats
```

### 月度运维

```bash
# 1. DRL 晋升审查 (30天 shadow 后)
python /opt/hmats/scripts/drl_promotion_gate.py --review

# 2. 策略 PnL 归因
python /opt/hmats/scripts/pnl_attribution.py --month $(date +%Y-%m)

# 3. 费用效率分析
python /opt/hmats/scripts/fee_analysis.py --month $(date +%Y-%m)
# → 检查 $10K/月 Kraken Pro 免费额度使用情况

# 4. AssetAlphaTilt 审查 (v10)
# → 检查 per-asset Sortino multipliers 是否合理

# 5. ExistenceFuse 月度重置确认
# → monthly_loss 计数器自动重置

# 6. 归档旧备份
tar -czf /var/lib/hmats/archives/$(date +%Y-%m).tar.gz \
    /var/lib/hmats/backups/state_$(date +%Y%m)*.db
```

---

## 紧急程序

### 程序 1: 紧急平仓

```bash
# 方法1: 通过系统命令 (首选)
sudo systemctl stop hmats
python /opt/hmats/scripts/emergency_flatten.py --confirm

# 方法2: 通过 Kraken UI (系统无响应时)
# 1. 登录 Kraken Pro
# 2. 市价卖出所有 BTC/ETH/SOL 仓位
# 3. API 验证 flat

# 方法3: REST API 脚本 (SSH 不可用时)
python /opt/hmats/scripts/remote_emergency_flatten.py --api-key $KEY
```

### 程序 2: Dead Man Switch 触发

```bash
# DMS 检测到心跳超时 → 自动撤单
# 1. 检查原因
sudo journalctl -u hmats --since "10 min ago" | grep -i "dead man"
# 2. 验证仓位已平仓
python /opt/hmats/scripts/check_positions.py
# 3. 调查根本原因 (网络/OOM/异常)
# 4. 修复后重启
sudo systemctl restart hmats
```

### 程序 3: ExistenceFuse HALT/KILL 触发

```bash
# Fuse 触发: consecutive-5 或 weekly-8% 或 monthly-10%
# 1. 检查 Fuse 状态
python /opt/hmats/scripts/check_fuse_status.py --detailed
# 2. HALT: 暂停交易, 允许平仓, 自动恢复
# 3. KILL: 系统停机, 全部平仓, 需要手动批准重启
# 4. 审查亏损原因
cat /var/log/hmats/shadow_ledger/recent_trades.json | jq '.[] | select(.pnl < 0)'
# 5. KILL 恢复: 确认问题已解决后
python /opt/hmats/scripts/fuse_reset.py --confirm --reason "问题已修复"
```

### 程序 4: BullTransition CONFIRMED

```bash
# BullTransition 达到 CONFIRMED → BLOCK_NAKED_SHORT
# 1. 检查 4 条件状态
python /opt/hmats/scripts/check_bull_transition.py --detailed
# → Golden Cross, SOL/BTC RS, Funding, OI
# 2. 现有空头: 加速止损 (自动)
# 3. 新空仓: 被阻止 (自动)
# 4. 考虑: 是否需要手动切换到做多策略
# 5. 注意: BullTransition 是保护机制, 不要手动覆盖
```

### 程序 5: 交易所 API 中断

```bash
# Kraken API 不可达
# 1. 检查 Kraken 状态
curl https://api.kraken.com/0/public/SystemStatus
# 2. 系统自动进入 NO_TRADE
# 3. Cancel-on-Disconnect 已激活 (断线撤单)
# 4. 现有仓位维持
# 5. 恢复后自动对账 (Startup Reconciler)
```

---

## 配置参考

### 集中参数管理 (sota_flags.py)

v10 的关键改进: **所有核心参数集中在 sota_flags.py**, 消除了 v6.5 时代参数散落多个文件的问题。

```python
# sota_flags.py — 集中参数 (v10)

# ═══ Drawdown 4级梯度 ═══
DRAWDOWN_REDUCE = 0.10      # 10% → position ×0.85
DRAWDOWN_HEAVY_REDUCE = 0.15 # 15% → position ×0.65
DRAWDOWN_HALT = 0.25         # 25% → 暂停交易
DRAWDOWN_KILL = 0.35         # 35% → 系统停机

# ═══ 杠杆 ═══
MAX_LEVERAGE = 3.0           # 硬性限制, 不可覆盖

# ═══ Alpha Gate ═══
ALPHA_GATE_NORMAL = 14       # bps, NORMAL 模式
ALPHA_GATE_OPPORTUNITY = 8   # bps, OPPORTUNITY 模式

# ═══ Maker/Taker 费用 ═══
MAKER_FEE_BPS = 16           # Kraken Pro maker
TAKER_FEE_BPS = 26           # Kraken Pro taker

# ═══ CRACK 阈值 (集中定义) ═══
CRACK_FULL_EXIT = 0.50
CRACK_PARTIAL = 0.45
CRACK_URGENCY = 0.35

# ═══ 相关性 ═══
CROSS_ASSET_CORRELATION = 0.87  # 统一, 不再有 0.0/0.65 遗留

# ═══ ExistenceFuse ═══
FUSE_CONSECUTIVE_LOSS = 5    # 连续亏损暂停阈值
FUSE_WEEKLY_LOSS = 0.08      # 8% 周损失 → HALT
FUSE_MONTHLY_OBSERVE = 0.08  # 8% 月损失 → OBSERVE (半仓)
FUSE_MONTHLY_KILL = 0.10     # 10% 月损失 → KILL

# ═══ 资产配置 ═══
BTC_MAX_EXPOSURE = 0.25      # 25%
ETH_MAX_EXPOSURE = 0.25      # 25%
SOL_MAX_EXPOSURE = 0.20      # 20%

# ═══ DRL 铁律 ═══
DRL_OBS_DIM = 126            # 不碰
DRL_ENT_COEF = 0.1           # 不碰

# ═══ Per-Asset GMM ═══
GMM_K_BTC = 8
GMM_K_ETH = 7
GMM_K_SOL = 7

# ═══ Squeeze Protection ═══
SQUEEZE_WARN = 0.50
SQUEEZE_REDUCE = 0.70
SQUEEZE_FLATTEN = 0.80

# ═══ Soft Multiplier Floor ═══
MULTIPLIER_FLOOR = 0.15      # VC-5 修复
```

### 环境变量

```bash
# /etc/hmats/hmats.env (600 perms)
KRAKEN_API_KEY=your_api_key_here
KRAKEN_SECRET_KEY=your_secret_key_here
COINGLASS_API_KEY=optional
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 目录结构

```
/opt/hmats/              — 应用代码 + main.py (~13,000行)
/opt/hmats/models/       — DRL 模型检查点
/var/log/hmats/          — 日志
  ├─ events/             — 事件日志
  ├─ proof/              — 决策 proof 日志
  └─ fill_quality.jsonl  — 成交质量记录 (v10)
/var/lib/hmats/          — 状态持久化 (SQLite)
/etc/hmats/              — 配置 + API keys
```

---

## 术语表 (v10 更新)

| 术语 | 定义 |
|------|------|
| **4H Tick** | 主决策周期 (14,400秒), process_4h_tick() 唯一入口 |
| **Alpha Gate** | 摩擦感知阈值: NORMAL 14bps, OPPORTUNITY 8bps |
| **AssetAlphaTilt** | ★v10: Per-asset Sortino-weighted 倾斜 (0.5~1.5×) |
| **Authority Fusion** | 5-agent 权限矩阵 (DECIDE/VETO/ADVISE/PENALIZE) |
| **BullTransitionDetector** | ★v10: 4条件牛市检测, CONFIRMED→BLOCK_NAKED_SHORT |
| **CRACK** | 清算级联触发器, 阈值 0.50/0.45/0.35 (集中定义) |
| **Drawdown 4级** | ★v10: 10%→减仓, 15%→大减, 25%→暂停, 35%→kill |
| **DRL** | TQC, obs_dim=126, ent_coef=0.1, FiLM Position A |
| **ExistenceFuse** | ★v10: 多层熔断 (consecutive-5/weekly-8%/monthly-10%) |
| **Fill Rate Logging** | ★v10: 成交质量 jsonl 记录 (周报 review) |
| **HPLV Filter** | ★v10: price≥90th + vol<60% → short×0.5 |
| **MonteCarloValidator** | ★v10: 1000 shuffle 策略鲁棒性验证 |
| **Multiplier Floor** | ★v10: 0.15, 防止 soft multiplier 叠乘致 4.6% |
| **One-Veto-Kill** | 8 veto 源, 任一否决→交易取消 |
| **PA Executor** | Passive-Aggressive, post_only→aggressive (120s timeout) |
| **process_4h_tick()** | 10步决策流程, main.py 唯一决策入口 |
| **ShadowLedger** | 不可变审计追踪 (含 tick_id + fill_quality) |
| **sota_flags.py** | 集中参数管理 (drawdown/leverage/alpha_gate/...) |
| **Startup Reconciler** | 重启对账, 消除 churn (AC-0~5) |

---

## 总结

### HMATS v10.0-POSTAUDIT 系统概述

HMATS 是**生产级多 Agent 加密货币交易系统**, 300+ Python 文件, main.py ~13,000 行。系统实现 **「Aggressive Alpha, Defensive Shell」** 哲学, 主做空, 高风险偏好。

### 审计状态

```
审计链: 1909行审计 → 84项自检 → 11修复(11/11 PASS) → 91项再审计
结果:   97.7% GREEN, 7/7 数据流 INTACT, 0 REGRESSION
SOTA:   ~96% 覆盖 (语境过滤后), 12 项超越 SOTA
```

### v10 关键能力

```
✅ 8 Veto 源 One-Veto-Kill + 31 gates (18 hard + 23 soft)
✅ 4级 Drawdown 梯度 (10/15/25/35%)
✅ ExistenceFuse 多层 (consecutive/weekly/monthly)
✅ BullTransitionDetector 4状态机
✅ 5-Agent Authority Fusion + Reliability Injection
✅ PA Executor + TimingEngine 已接线
✅ Fill Rate Logging
✅ AssetAlphaTilt (Sortino-weighted)
✅ HPLV Filter + MonteCarloValidator
✅ Anti-Churn (AC-0~5) + Cancel-on-Disconnect
✅ Veto Chain 修复 (VC-0~9, floor 0.15)
✅ sota_flags.py 集中参数管理
```

### 当前状态 (2026年2月28日)

```
✅ 完成:
├─ v10.0-POSTAUDIT 审计通过
├─ Paper trading 首批盈利 (+$27.08)
├─ 所有数据源 LIVE
├─ Anti-Churn + Veto Chain 修复

🔄 进行中:
├─ Final Polish (6 items)
├─ 24h Paper Run Validation
├─ DRL TQC Training (Stage 8B+)

📋 Next Steps:
├─ 24h 干净 paper run → live deployment
├─ DRL SHADOW→EXIT_ONLY (30天)
├─ Account scaling $10K → $100K
└─ Stage 21 Meta-Learner
```

---

## 完整文档系列

| Part | 标题 | 版本 |
|------|------|------|
| Part 1 | 概览与系统总览 | v10.0 |
| Part 2 | 数据流与决策链 | v10.0 |
| Part 3 | 风险管理与状态机 | v10.0 |
| Part 4 | 执行层与DRL训练 | v10.0 |
| Part 5 | 运维手册与附录 (本文档) | v10.0 |

**配套文档**:
- 02_STRATEGY_AND_MODULES_v10.md — 策略文档
- 03_WHITE_PAPER_v10.md — 白皮书
- HMATS_CONVERSATION_SUMMARY_v10.md — 对话摘要
- HMATS_SOTA_FILTERED.md — SOTA 差距分析

---

_HMATS v10.0-POSTAUDIT 系统架构文档结束_

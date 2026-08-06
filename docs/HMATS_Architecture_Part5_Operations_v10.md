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

> **[P190 2026-08-05] 本节此前记录的命令没有一条可以运行。**
> 两个独立的问题:
>
> 1. **14 个 `/opt/hmats/scripts/*.py` 脚本从未存在过** —— `git log --all
>    --diff-filter=A` 在任何提交中都找不到它们。日常/周度/月度清单、
>    紧急平仓程序、Fuse 复位程序全部指向不存在的文件。
> 2. **部署不是 systemd** —— `sudo systemctl stop hmats` /
>    `journalctl -u hmats` 描述的是 v5.1.0 时代的 venv 部署
>    (`deploy/systemd/hmats.service`, 跑的还是 `main.py --mode paper`)。
>    实际线上是 `docker-compose.hetzner.yml`: 容器 `hmats-engine` (v6.8.0)
>    + `hmats-api`。日志/状态在 docker volume 里, 不在 `/var/log/hmats`
>    和 `/var/lib/hmats`。
>
> 下面的命令是按仓库里**实际存在**的东西重写的。没有实现的能力标为
> **[未实现]** —— 事故当中发现一条命令不存在, 比一开始就知道它不存在要贵。
> 这与 [P186] (`make drl` 指向不存在的脚本) 和 [P189]
> (`run_training.py` 同样的毛病) 是同一类缺陷。

### 每日检查清单

```bash
# 1. 检查服务状态
ssh hmats "cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml ps"

# 2. 审查过夜日志
ssh hmats "docker logs hmats-engine --since 24h 2>&1 | grep -iE 'error|traceback'"

# 3. 权益 / 回撤 / 仓位 —— 都在 dashboard_state.json 里
#    (main.py:_export_dashboard_state 是它唯一的写入者)
ssh hmats "docker exec hmats-engine python -X utf8 -c \"
import json;d=json.load(open('/opt/hmats/data/dashboard_state.json'));
eq,pk=d['equity'],d['peak_equity'];
print('updated_at', d['updated_at'], 'mode', d['mode']);
print('equity %.2f peak %.2f drawdown %.2f%%' % (eq, pk, 100*(1-eq/pk) if pk else 0));
print('positions', d['position_count'], d['positions'])\""
# → 回撤 4 级梯度阈值见下方\"配置参考\": 10% / 15% / 25% / 35%

# 4. 为什么没交易 (阻塞原因, 不用等 4H tick)
ssh hmats "docker exec hmats-engine python -X utf8 scripts/why_no_trade.py"

# 5. 审查 Fill Quality
ssh hmats "docker exec hmats-engine tail -20 /opt/hmats/logs/fill_quality.jsonl" | jq .

# 6. ExistenceFuse 状态 (consecutive/weekly/monthly loss)   [未实现]
# 7. BullTransition 4 条件状态                              [未实现]
#    两者都没有 CLI。现状只能读引擎日志:
ssh hmats "docker logs hmats-engine --since 24h 2>&1 | grep -iE 'fuse|bulltransition'"
```

### 周度运维

```bash
# 1. 执行质量周报 (maker_ratio, slippage, timeout)
python -X utf8 scripts/execution_quality_report.py

# 2. 周度性能回顾
python -X utf8 scripts/reflect_weekly.py
python -X utf8 scripts/weekly_agent_report.py

# 3. Agent 归因审计 (16 agent 信号是否真的被消费)
ssh hmats "docker exec hmats-engine python -X utf8 scripts/agent_audit_16.py"

# 4. 策略诊断
ssh hmats "docker exec hmats-engine python -X utf8 scripts/kq_strategy_diagnostic.py"

# 5. 清理旧日志 (docker 已做 rotation: json-file, max-size 50m x 5)
#    见 docker-compose.hetzner.yml 的 logging 段, 无需手工 find -delete

# 6. 重启 (仅在 flat 时): position_count 见每日检查第 3 项
ssh hmats "cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml restart hmats-engine"

# DRL drift 检查                                            [未实现]
#   最接近的是 scripts/validate_drl_oos.py (离线 OOS 验证, 不是 drift 监控)
```

### 月度运维

```bash
# 1. DRL 晋升状态 (晋升逻辑在 risk/ 的 promotion gate 里, 由引擎自己写状态)
ssh hmats "docker exec hmats-engine cat /opt/hmats/data/drl_promotion_state.json"
ssh hmats "docker exec hmats-engine cat /opt/hmats/data/exit_drl_promotion_state.json"

# 2. 策略 PnL 归因
python -X utf8 scripts/agent_attribution_validate.py

# 3. 费用效率分析 (现货 vs 永续 两个 sleeve 的费率)
python -X utf8 scripts/futures_vs_spot_fee_analysis.py
# → Kraken Pro $10K/月免费额度: data/kraken_plus_monthly_volume.json

# 4. AssetAlphaTilt 审查
# → 检查 per-asset Sortino multipliers 是否合理 (无 CLI, 读配置 + 日志)

# 5. ExistenceFuse 月度重置确认
# → monthly_loss 计数器自动重置

# 6. 备份 docker volume
ssh hmats "docker run --rm -v hmats-data:/d -v ~/backups:/b alpine \
    tar -czf /b/hmats-data-\$(date +%Y-%m).tar.gz -C /d ."
```

---

## 紧急程序

> **[P190] 本节的 5 个 `emergency_flatten.py` / `remote_emergency_flatten.py`
> / `check_positions.py` / `check_fuse_status.py` / `fuse_reset.py` /
> `check_bull_transition.py` 全部不存在。** 事故当中才发现首选平仓命令是
> "can't open file", 是这份文档能造成的最贵的一种错误。下面按实际存在的
> 东西重写, 并明确标出没有实现的部分。

### 程序 1: 紧急平仓

**没有通用的一键平仓命令。** 现有的三条路径:

```bash
# 方法1 (首选): 先停引擎, 再从 Kraken Pro UI 手工市价平掉 BTC/ETH/SOL
ssh hmats "cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml stop hmats-engine"
# 1. 登录 Kraken Pro, 市价平掉所有 BTC/ETH/SOL 仓位 (含保证金空头)
# 2. 回到本地验证 flat:
#    python -X utf8 scripts/kraken_credentials_check.py

# 方法2: Coinbase sleeve 平仓 (仅 Coinbase nano 永续, 不含 Kraken)
python -X utf8 scripts/coinbase_flatten.py          # 先看 dry-run 输出

# 方法3: Kraken 现货多头平仓 —— scripts/reconcile_flatten_2026_06_12.py
#   这是 2026-06-12 事故时写的一次性脚本, 默认 dry-run, 只 SELL
#   BTC/ETH/SOL 现货, 从不买入、从不用杠杆。它**不平保证金空头**。
#   scripts/ 没有打进镜像 (见下方"已知缺口"), 需要先 scp 进去:
#   scp scripts/reconcile_flatten_2026_06_12.py hmats:/tmp/
#   ssh hmats "docker cp /tmp/reconcile_flatten_2026_06_12.py hmats-engine:/tmp/"
#   ssh hmats "docker exec hmats-engine python3 -X utf8 /tmp/reconcile_flatten_2026_06_12.py"
#   加 --execute 才会真的下单。

# [未实现] 统一的 emergency_flatten (现货 + 保证金 + Coinbase, 一条命令)。
# [未实现] SSH 不可用时的 REST 兜底 (remote_emergency_flatten)。
# 引擎内部的 trigger_emergency_flatten() (main.py:14906) 只由 DEAD_MAN_SWITCH
# 触发, 没有对外的 CLI 入口。
```

### 程序 2: Dead Man Switch 触发

```bash
# DMS 检测到心跳超时 → 自动撤单
# 1. 检查原因
ssh hmats "docker logs hmats-engine --since 10m 2>&1 | grep -i 'dead man'"
# 2. 验证仓位已平仓 (position_count 见"每日检查清单"第 3 项)
# 3. 调查根本原因 (网络/OOM/异常)
ssh hmats "docker inspect hmats-engine --format '{{.State.OOMKilled}} {{.State.ExitCode}}'"
# 4. 修复后重启
ssh hmats "cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml restart hmats-engine"
```

### 程序 3: ExistenceFuse HALT/KILL 触发

```bash
# Fuse 触发: consecutive-5 或 weekly-8% 或 monthly-10%
# 1. 检查 Fuse 状态  [未实现 —— 没有 check_fuse_status CLI]
ssh hmats "docker logs hmats-engine --since 24h 2>&1 | grep -iE 'existencefuse|fuse.*(halt|kill)'"
# 2. HALT: 暂停交易, 允许平仓, 自动恢复
# 3. KILL: 系统停机, 全部平仓, 需要手动批准重启
# 4. 审查亏损原因
python -X utf8 scripts/analyze_shadow_ledger.py
python -X utf8 scripts/shadow_ledger_report.py
# 5. KILL 恢复  [未实现 —— 没有 fuse_reset CLI]
#    目前只能停容器、确认原因、再 up -d 重启。
```

### 程序 4: BullTransition CONFIRMED

```bash
# BullTransition 达到 CONFIRMED → BLOCK_NAKED_SHORT
# 1. 检查 4 条件状态  [未实现 —— 没有 check_bull_transition CLI]
#    → Golden Cross, SOL/BTC RS, Funding, OI; 只能读日志:
ssh hmats "docker logs hmats-engine --since 24h 2>&1 | grep -i bulltransition"
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

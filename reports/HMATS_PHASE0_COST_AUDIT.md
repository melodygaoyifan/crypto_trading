# HMATS Phase 0 — 真实成本审计（只读）

生成：2026-10-18 · 数据源：Coinbase Advanced Trade API（`get_fills` / `list_orders` / `get_transaction_summary` / `get_product` / `list_futures_positions`，均为 GET，在 `hmats-engine` 容器内经 stdin 运行，未复制任何文件进容器）+ 服务器 `data/fill_quality.jsonl`（决策时 mid）+ `data/cde_funding_history.jsonl`（funding **费率**）+ `data/diagnostics/`（4,979 个 tick 的 GMM regime）。
本审计未改动任何代码、配置或参数。机器可读结果：`reports/phase0_params.json`；原始数据：`data/phase0/{fills_raw,funding_raw,roundtrips}.parquet`、`product_specs.json`、`fee_tier.json`。

---

## 1. 结论表

| 资产 | n_RT | 中位 all-in 成本 (bps/RT) | 中位 gross alpha (bps/RT) | 中位 net | 判定 |
|---|---|---|---|---|---|
| BTC | 41 | **21.8** | **−23.9** | −47.8 | **STRUCTURAL_DEFICIT** |
| ETH | 23 | **32.1** | **−24.0** | −53.1 | **STRUCTURAL_DEFICIT** |
| SOL | 24 | **26.6** | **−40.1** | −64.4 | **STRUCTURAL_DEFICIT** |

**核心事实：90 个已平仓 RT 的 gross PnL 合计 −$5.85，手续费合计 $149.02。** 也就是说，即使手续费为零，这本账也几乎没赚钱。三个资产的**中位 gross alpha 在扣任何成本之前就是负的**。"每笔收益覆盖不了手续费"的说法方向对，但不够：问题首先是信号，其次才是费用。费用把一个 ≈0 的 gross 变成了 −$155 的 net。

**样本口径警告（必须读）：** 88/90 个 RT 开仓于 2026-06 ~ 08（06: 19, 07: 38, 08: 31, 09: 2），即趋势 seat ±1 翻转期 + regimebook/whale/skew seat 交替期。当前配置（skew seat on BTC、regimebook、resize/flip persistence、P356 去掉两个 entry filter 之后）只贡献了 2 个已平仓 RT + 1 个未平仓（BTC −2ct，自 2026-10-13）。**这个表衡量的是过去的配置，不是现在的。**

## 2. 费用：系统 model vs 交易所真实 commission

### 2.1 真实费率结构（来自每个 fill 的 `commission_detail_total`，不是拟合）

```
commission = client_commission (名义的百分比，按 30 天量分层)
           + venue_commission  $0.10 / 合约
           + clearing_commission $0.01–0.05 / 合约
```

- **当前 tier（`get_transaction_summary`）：`Intro`，taker 10.0 bps / maker 9.5 bps，`has_cost_plus_commission=true`，`has_promo_fee=false`，30 天量 $9,600.65。**
- 固定部分是**逐合约相加**，不是"最低费"：只有 1/211 个 fill（0.47%）的固定部分大于百分比部分。固定部分占总手续费 **28.2%**。
- 百分比部分随 30 天量变化，历史中位（bps）：06 taker 9.0 · 07 taker 8.0 · 08 maker 6.5 / taker 8.0 · 09 maker 6.5 / taker 7.0 · **10 maker 9.5 / taker 10.0**。**现在处在最贵的 tier**，因为交易量下降了——大部分历史 fill 比今天便宜。
- 拟合 `commission ≈ a·notional + b·contracts`（跨 tier 合并，因此有误差）：a = 8.29 bps，b = $0.098/ct，MAPE 9.05%（maker 5.5% / taker 7.9%）。精确结构以 breakdown 为准。
- 单合约每腿实际费率（当前 tier，taker）：BTC $813 名义 → 10 + 1.35 = **11.35 bps**；ETH $241 → 10 + 4.55 = **14.55 bps**；SOL $519 → 10 + 2.12 = **12.1 bps**。

### 2.2 Maker 是否有费用优惠？—— 有，但很小

| | 同月 (2026-08) 百分比部分 | 当前 tier |
|---|---|---|
| maker | 6.5 bps | 9.5 bps |
| taker | 8.0 bps | 10.0 bps |
| 差 | **1.5 bps** | **0.5 bps** |

另外 maker fill 赚到了 spread（相对 mid 中位 −0.6 ~ −1.6 bps），taker fill 付出 +0.4 ~ +4 bps。maker 每腿总共省 ~1.5–4.5 bps。**在这个 venue 上 maker 率主要省 spread，费用优惠在当前 tier 几乎为零**——这会改变 Phase 1 对 maker 比例的权重。（注：实际 maker 成交比例 = **73/211 = 34.6%**（按 venue `liquidity_indicator`），**不是** 98.7%；98.7% 是 `training/backtest_framework.FeeSchedule` 的默认假设，P166 已指出对应的 review 指标度量的是订单类型而非成交类型。）

### 2.3 系统 fee model 误差

| 模型（对全部 211 fills，Σcommission = $154.48） | Σmodel / Σcommission − 1 |
|---|---|
| 当前 `core.cde_fees.cde_fee_bps`（全部 fill 重算） | **+10.3%**（总体高估） |
| fill 当时生效的模型（2026-08-19 前 VENUE_FEE_STD 0/3 bps，之后 cde_fee_bps） | **−32.6%**（低估） |
| `VENUE_FEE_STD` coinbase 0/3 bps（全部） | **−80.4%** |
| 当前 model，**仅当前 tier 的 10 月 fill**（n=6） | **−13.8%**（低估） |

结论：当前 model 的总体高估来自它把 SOL 设成"假设最贵"的 14.51 bps 以及历史上 tier 更便宜；**但在今天的 tier 上它对 BTC 是低估的**（model 9.87 taker / 9.38 maker vs 实际 11.35 / 10.8），因为 model 是纯百分比、没有 $0.11/ct 固定项，并且是在更便宜的 tier 上校准的。

## 3. 0.1 代码状态（grep 当前 codebase）

- **gate 用的 fee**：`main.py:12821` `resolve_venue_fee_bps`（`core/execution_service.py:349`，返回 Coinbase 0/3 bps）→ `main.py:12840-12858` 在 `coinbase_per_contract_fees`（容器内配置 = `true`）下用 `core/cde_fees.py:181 cde_fee_bps` **max()** 抬高到 BTC 9.38/9.87、ETH 13.78/14.51、SOL 14.51/14.51（`core/cde_fees.py:106-145`）。纯百分比，**无 per-contract 最低费/固定项**。live 日志 2026-10-17 16:02 确认：`[P315-FEE] BTC: maker=9.38bps taker=9.87bps`。
- **friction 是 round-trip**：`defense/constitution.py:1539-1553` `friction = 2 × per_leg_bps + margin_bps`；`per_leg_bps = fee + slippage + latency`（`:1350`）。slippage 用 `CDE_SPREAD_BPS_MEASURED` 的**完整 spread**（BTC 2.0 / ETH 5.5 / SOL 4.0，`:1076`）按每腿收取；latency 2.0 bps/腿；hold = venue-true funding（`coinbase_venue_true_hold=true`）× 6 个 4H 期。门槛 = friction × 1.10。
  - gate 的 BTC 每 RT 成本 ≈ 2×(9.87+2.0+2.0) = 27.7 + hold；ETH ≈ 2×(14.51+5.5+2.0) = 44.0 + hold。**实测 all-in 中位：BTC 21.8 / ETH 32.1 / SOL 26.6**——gate 在 spread+latency 上比实测更保守（实测 taker 半 spread ~1 bps/腿，不是 2–5.5），在今天 tier 的 fee 上对 BTC 偏低。
- **hot path**：`cde_fee_bps` 只在 alpha gate（`main.py:12851`）和离线 labs（`training/funding_legs_lab.py:139`）上。sleeve 实际 PnL/费用全部来自 venue（`fill_quality.jsonl` 的 `fees_usd` 来自 `get_order`），没有 model 化的 paper fee；`core/paper_fee_service.VENUE_FEE_STD` 只在 Kraken 路径（P152 后结构性不执行）。
- **合约规格 vs API**：contract_size BTC 0.01 / ETH 0.1 / SOL 5 / XRP 500 / BNB 1 与 `exchange/coinbase_adapter.py:117` `_CONTRACT_SIZE_FALLBACK` **一致**；tick 0.01；最小 1 合约；**funding_interval 3600s（每小时）**。差异：API 报 `contract_expiry = 2089-12-30T16:00Z`，而 CLAUDE.md / adapter 注释写 2030-12-20（产品 ID 仍是 `20DEC30`）。仅记录。
- **funding 记录口径**：`main.py:24934/24952` 存的 `funding_8h = 每小时费率 × 8`，与 3600s interval 一致。

## 4. Round-trip 重建（FIFO，0.4）

- 211 fills / 186 orders / **90 个已平仓 RT** + 1 个未平仓（BTC −2ct，入场名义 $3,294）。**FIFO 终点 = BTC −2，其余为 0，与 venue `list_futures_positions`（BTC SHORT 2）完全一致**——起点为平仓的假设和账本完整性被验证。
- XRP 2 个 RT（XRP 已于 2026-10-13 de-route）单独列出不进主表；BNB 无成交。

| 资产 | n | 中位 all-in | 均值 all-in | 中位 fee | 中位 spread | 中位 funding* | 中位 α | 均值 α | 胜率(gross) | 中位 hold (bars) | fills/RT | Σgross $ | Σfee $ | fee_share** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 41 | 21.8 | 21.4 | 20.3 | 1.17 | 0.52 | −23.9 | −41.2 | 24% | 2.4 | 2.46 | −71.25 | 70.77 | 0.30 |
| ETH | 23 | 32.1 | 31.5 | 31.4 | 2.61 | 2.58 | −24.0 | −7.0 | 22% | 3.0 | 3.78 | −2.75 | 45.12 | 0.19 |
| SOL | 24 | 26.6 | 25.5 | 24.6 | 1.36 | 1.49 | −40.1 | −16.5 | 33% | 3.0 | 2.67 | +140.10 | 30.58 | 0.09 |

\* funding 只对 17/90 个 RT 可估（见 §8）。\*\* fee_share = Σfee / Σ|gross_pnl|。
SOL Σgross 为正（+$140）而中位 α 为负：由少数大赢家驱动（SOL 胜率 33%），中位不支持"SOL 有可靠 edge"。

**21 个 RT 持仓 < 1 个 4H bar**（7 个 < 1 小时）：gross −$20.25，fee $29.13。

## 5. Alpha vs 持仓时间（0.5）

| 资产 | ≤1 | 2 | 3–4 | 5–8 | 9+ |
|---|---|---|---|---|---|
| BTC n / α / cost | 7 / −7.7 / 20.2 | **11 / −41.9 / 20.9** | 8 / −13.7 / 21.1 | 7 / −150.5 / 24.8 | 8 / +19.4 / 22.9 |
| ETH | 8 / −19.1 / 39.2 | 2 / −80.2 / 30.8 | 4 / −134.8 / 34.8 | 3 / +32.2 / 30.0 | 6 / +103.5 / 31.6 |
| SOL | 6 / −43.6 / 28.3 | 4 / −96.2 / 25.1 | 4 / −21.0 / 28.3 | 2 / +81.3 / 25.9 | 8 / +4.3 / 26.0 |

只有 BTC 的 "2 bars" 桶 n ≥ 10，且为 NEGATIVE。**`break_even_hold_bars`：BTC = NONE（在样本充足的桶中），ETH = INSUFFICIENT，SOL = INSUFFICIENT。**
指示性（不可作为参数）：长持仓桶（BTC/ETH 9+、ETH/SOL 5–8）中位 α 为正，与仓库既有结论"edge 在多日 regime 持仓、不在 4H 翻转"一致；但 n=2–8，按铁律 4 不外推。

## 6. Fill 构成（0.6；按订单分类，181/186 由仓位变化 INFERRED，5 个与 `fill_quality.position_before` 交叉验证一致）

| 类别 | 订单 | fills | Σfee $ | Σ已实现 gross $ |
|---|---|---|---|---|
| NEW_ENTRY | 33 | 36 | 28.16 | — |
| ESCALATE | 28 | 37 | 19.61 | — |
| PARTIAL_EXIT | 32 | 32 | 19.55 | +81.42 |
| FULL_EXIT | 32 | 37 | 26.96 | +107.50 |
| **FLIP** | **58** | **66** | **59.36** | **−175.45** |
| REBALANCE（同向、\|Δ\|<25%） | 3 | 3 | 0.84 | −2.69 |
| RESTART_RECONCILE / UNKNOWN | — | — | — | 无法从 venue 账本区分（UNAVAILABLE） |

- **FLIP 是损失和费用的最大来源**：58 个翻向订单付了 40% 的手续费，且实现 gross −$175（它们关闭的仓位是亏损的）。2026-07 的 43 个订单中 38 个是 FLIP（趋势 seat ±1 翻转期，P198 记录的 BTC 54 天翻 29 次）。
- 1 张合约的 fill：**120 / 211**，Σcommission $59.43。
- PARTIAL_EXIT：平均 1.5 合约，平均 fee 9.90 bps/腿——与整笔退出无差别（费用逐合约线性）。
- **同一 4H tick 同一资产 ≥2 个订单：20 次，其中 18 次方向相反**（netting 失败形态；例：2026-08-18 08:00 BTC SELL 2 / BUY 2 / SELL 2，fee $3.68；2026-06-13 20:00 ETH、SOL 各 BUY/SELL/BUY/SELL）。集中在 06-13 首日与 08-16 ~ 08-28（P232/P370/P416 修复之前的 watchdog/resize 抖动期）；09 月以后为 0。
- REBALANCE |Δ| 分布：n=3，P25/P50/P75 = 0.18/0.20/0.20（INSUFFICIENT）。所有同向调仓 n=63：0.50/0.50/1.00——仓位只有 1–4 合约，任何一次调仓至少 25%–100%。

## 7. 退出碎片化（0.7）

- `execution/exit_alpha.py`：`min_profit_bps` 默认 20；**`main.py:6117-6120` 构造时 `scale_out_pct=0.10`、`min_profit_bps=500`、`min_bars_held=3`**。`REBUILD_COOLDOWN_EXEMPT_FLIP=False`（`core/execution_service.py:137`）。
- **可达性：以上全部在 Kraken 执行路径上**（P152 后对 Coinbase-routed 资产结构性不执行；P275 已确认 exit stack 读 Kraken 簿，对 sleeve 永不触发）。`data/exit_alpha_tracking.jsonl` 最后写入 2026-05-22，**ExitAlphaTracker 对 sleeve 无数据 → T1–T14 计数、peak→realized 保留率 UNAVAILABLE**。
- sleeve 上实际可达的：`coinbase_reentry_cooldown_ticks=2`、`coinbase_flip_persist_ticks=2`、`coinbase_resize_persist_ticks=2`、`alpha_gate_hold_ratio=0.65`（容器内配置读回）。sleeve 没有 scale-out 逻辑（P276：测量后为 non-gap）。
- 每 RT exit fills：1 个 = 81 RT，2 = 4，3/6/7/9/12 各 1（多 exit fill 的 RT 是开仓后多次 resize 下调）。

## 8. 显式列出的 UNAVAILABLE / INSUFFICIENT / UNKNOWN

| 项 | 状态 | 原因 |
|---|---|---|
| Funding **实际支付** | **UNAVAILABLE** | Advanced Trade API 无 funding 结算账本；`list_futures_sweeps` 为空。仅能用费率 × 仓位估算（`ESTIMATE_FROM_RATES`） |
| Funding 估算覆盖 | 17/90 RT | 费率历史 `cde_funding_history.jsonl` 始于 2026-08-16；之前的 73 个 RT 标 `UNAVAILABLE_NO_RATE_HISTORY`，其 all_in_cost 只含 fee+spread |
| Spread 的 mid 来源 | 15 RT decision_mid / 74 RT 1m-bar (H+L)/2 / 1 混合 | 2026-08-18 之前没有决策时 bid/ask 记录；1m bar 中价是噪声代理，可能把成交本身包含进去 |
| `exit_reasons` | **UNKNOWN（126/126 退出腿）** | `fill_quality.cause` 只从 P423（2026-10-13）开始，之后无已平仓 RT；服务器日志只保留到 ~2026-10-12 |
| `regime_at_entry` | 全部可得 | 来自 tick diagnostics（最近 ≤8h 的 tick） |
| `break_even_hold_bars` | BTC NONE / ETH、SOL INSUFFICIENT | §5 |
| per_asset_regime | 只有 BTC/ETH/SOL × QUIET_ACCUMULATION（n=31/18/17）充足，均 **NEGATIVE**（中位 α −23.9 / −42.8 / −26.2 vs cost 21.6 / 33.4 / 26.9）；其余格 INSUFFICIENT | `phase0_params.json:per_asset_regime` |
| RESTART_RECONCILE 分类 | UNAVAILABLE | venue 账本无法区分重启对账单 |
| fill 分类 | 181/186 INFERRED | 只有 5 个订单有 `position_before` 可交叉验证 |
| `fill_quality.target_contracts` | 记录值等于方向（−0.9998），不是合约数 | 仅记录，非本审计范围 |

## 9. Phase 1 参数（推导规则 → 结果）

| 参数 | 规则 | 结果 |
|---|---|---|
| `NO_TRADE_BAND_FRAC` | REBALANCE \|Δ\| P75，clamp [0.15, 0.35] | **null — INSUFFICIENT**（n=3，P75=0.20）。所有调仓 \|Δ\| P25=0.50：1–4 合约下 [0.15, 0.35] 的 band 不会生效 |
| `MIN_HOLD_BARS` | 各资产 break_even 中位（忽略 NONE），clamp [1, 3] | **null — INSUFFICIENT**（无可计算的 break_even） |
| `MIN_EXIT_CONTRACTS` | 单张 partial exit fee_bps ≤ 20% all_in_cost 的最小张数 | **null — NOT_ACHIEVABLE**：费用两部分都逐合约计，fee_bps 与张数无关（~10–15 bps/腿），任何张数都达不到 all-in 的 20%（~4–6 bps） |
| `SCALE_OUT_TRANCHES` | （spec 未给规则） | **null — 无费用依据**：无单笔最低费，分批不增加每合约费用 |
| `friction_rt_bps` | 中位 all-in（spread 不可得时 fee+funding） | **BTC 21.8 / ETH 32.1 / SOL 26.6**（含 1m-bar spread 代理；funding 仅部分 RT 有估算） |
| （附）当前 tier 下的投影 | 2×(10 bps + $0.11/(cs·price)) + 实测 spread | BTC 23.9 / ETH 31.7 / SOL 25.6 |

**对 Phase 1 最重要的一条：** 在费用侧，唯一显著可调的项是 **30 天交易量 tier**（10 bps vs 6.5 bps 是 3.5 bps/腿）和 **ETH 的固定项**（$0.11 在 $241 名义上是 4.55 bps/腿）。但三个资产的中位 gross alpha 都低于 −20 bps，在零费用下仍为负——在当前样本上，没有任何 Phase 1 范围内的费用/退出参数能把这三行变成 COVERS。

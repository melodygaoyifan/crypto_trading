# 🚀 HMATS Training Pipeline

完整的 HMATS 模型训练系统。

| 组件 | 版本 | 性能 | 文件 |
|------|------|------|------|
| **GMM** | 6-Regime | - | `gmm/gmm_pretrain.py` |
| **Decision Transformer** | v3.2 | 78-82% 准确率 | `drl/train_decision_transformer_v32.py` |
| **DRL** | v5.5 | Sharpe 1.5-2.2 | `drl/train_drl_v55.py` |
| **Sentiment** | v2.2 | 92-95% 准确率 | `sentiment/train_sentiment_agent_v22.py` |

---

## 📁 目录结构

```
training/
├── run_training.py              # 主编排脚本
├── Makefile                     # 统一 Makefile
├── README.md
├── requirements.txt
│
├── drl/                         # DRL 模型
│   ├── train_decision_transformer_v32.py  # DT v3.2 (1267行)
│   ├── train_drl_v55.py         # DRL v5.5 训练
│   ├── model.py                 # DRL v5.5 模型 (6-Regime MoE)
│   ├── features.py              # 特征工程
│   └── DT_V32_README.md
│
├── sentiment/                   # Sentiment Agent
│   ├── train_sentiment_agent_v22.py  # v2.2 (1480行)
│   └── SENTIMENT_V22_README.md
│
├── gmm/                         # GMM 预训练
│   └── gmm_pretrain.py          # 6-Regime GMM
│
├── configs/
│   └── config.py
│
└── scripts/
    └── collect_data.py
```

---

## ⚡ 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```
training_data/
├── raw/
│   ├── BTC_60m.parquet
│   ├── ETH_60m.parquet
│   └── SOL_60m.parquet
└── processed/
    └── gmm_features.parquet  # 可选
```

### 3. 训练

```bash
# 完整流程
make all

# 快速测试
make quick

# 或
python run_training.py --all
python run_training.py --quick
```

---

## 🎯 组件详情

### GMM 6-Regime

```
STRONG_BEAR (0)   → bias = -0.25
MODERATE_BEAR (1) → bias = -0.10
NEUTRAL (2)       → bias = 0.00
VOLATILE_CHOP (3) → bias = 0.00
MODERATE_BULL (4) → bias = +0.05
STRONG_BULL (5)   → bias = +0.15
```

### DT v3.2 (78-82% 准确率)

- ✅ Regime 感知智能专家 (+3-5%)
- ✅ 智能 Oracle (+2-3%)
- ✅ OHEM + R-Drop + FGM
- ✅ AMP 混合精度

### DRL v5.5 (Sharpe 1.5-2.2)

- ✅ RTG-Free 模式
- ✅ 6-Regime MoE (6专家)
- ✅ Cross-Asset (BTC/ETH)
- ✅ 3模型集成

### Sentiment v2.2 (92-95% 准确率)

- ✅ DeBERTa-v3 + SupCon (+2-3%)
- ✅ 修复 R-Drop / FGM
- ✅ OHEM + Focal Loss
- ✅ TTA 测试时增强

---

## 🔧 Make 命令

| 命令 | 说明 |
|------|------|
| `make all` | 完整训练 |
| `make quick` | 快速测试 |
| `make gmm` | GMM 预训练 |
| `make dt` | DT v3.2 |
| `make drl` | DRL v5.5 |
| `make sentiment` | Sentiment v2.2 |
| `make check` | 检查数据 |
| `make collect` | 收集数据 |

---

## 📊 预期性能

| 配置 | DT 准确率 | DRL Sharpe | Sentiment | 时间 |
|------|-----------|------------|-----------|------|
| Quick | 70-75% | 1.0-1.5 | 85-88% | ~2h |
| Default | 78-82% | 1.5-2.0 | 90-92% | ~12h |
| Full | 80-85% | 1.8-2.2 | 93-95% | ~24h |

---

## 📝 数据要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 时间 | 6个月 | 1-2年 |
| 资产 | SOL | BTC/ETH/SOL |
| 频率 | 1H | 1H |

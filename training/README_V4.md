# 🚀 HMATS Training v4.0 - End-to-End Diagnostics

> ⚠️ **HISTORICAL (banner added 2026-08-14, P269).** 描述的 zip 分发/
> auto_fix 工作流早已不存在。训练的权威文档：
> `docs/HMATS_TRAINING_GUIDE_V2.md` + `training/Makefile`。

完整的诊断 + 训练系统，自动检测问题并提供修复。

## 📦 快速开始

```bash
# 1. 解压
unzip hmats_training_v4.zip -d hmats_training
cd hmats_training

# 2. 自动修复
python auto_fix.py

# 3. 诊断
python diagnose.py --all

# 4. 训练
python train_v4.py --all
```

---

## 🔧 命令参考

### 诊断

```bash
# 完整诊断
python diagnose.py --all

# 仅 GPU
python diagnose.py --gpu

# 仅数据
python diagnose.py --data

# 仅模块
python diagnose.py --modules

# 计算推荐参数
python diagnose.py --params
```

### 训练

```bash
# 完整诊断 + 训练
python train_v4.py --all

# 仅诊断（不训练）
python train_v4.py --diagnose-only

# 快速模式（减少训练量，用于测试）
python train_v4.py --all --quick

# 训练特定组件
python train_v4.py --components dt,tqc

# 自定义参数
python train_v4.py --all \
    --dt-batch-size 64 \
    --dt-epochs 100 \
    --tqc-timesteps 1000000
```

---

## 📊 组件说明

| 组件 | 脚本 | 用途 |
|------|------|------|
| **GMM** | `gmm/gmm_pretrain.py` | 6-Regime 市场状态分类 |
| **DT v3.2** | `drl/train_decision_transformer_v32.py` | Decision Transformer 策略 |
| **TQC** | `train_tqc.py` | Truncated Quantile Critics |
| **Sentiment** | `sentiment/train_sentiment_agent_v22.py` | 情感分析 |

---

## 📁 目录结构

```
hmats_training/
├── train_v4.py              # 🆕 v4 主训练脚本
├── diagnose.py              # 🆕 详细诊断工具
├── auto_fix.py              # 🆕 自动修复脚本
│
├── training_data/
│   ├── raw/                 # OHLCV 数据
│   │   ├── SOL_60m.parquet
│   │   ├── BTC_60m.parquet
│   │   └── ETH_60m.parquet
│   └── sentiment/           # 情感数据
│
├── models/                  # 训练输出
│   ├── gmm_regime_6.pkl
│   ├── dt_v32_best.pt
│   └── tqc_*/
│
├── drl/
│   ├── train_decision_transformer_v32.py
│   ├── train_drl_v55.py
│   ├── model.py
│   └── features.py
│
├── gmm/
│   └── gmm_pretrain.py
│
├── sentiment/
│   └── train_sentiment_agent_v22.py
│
├── configs/
│   └── config.py
│
├── get_data.py              # 数据获取
├── train_tqc.py             # TQC 训练
└── train_kfold.py           # K-Fold 训练
```

---

## ⚠️ 常见问题

### Q: DT 训练 Val Acc = 0%

**原因**: 验证集太小 + batch_size 太大 + drop_last=True

**修复**: v4 已自动处理
```bash
python train_v4.py --all --dt-batch-size 64
```

### Q: 数据不足

```bash
# 获取数据
python get_data.py --all --years 2

# 或生成示例数据测试
python get_data.py --generate-sample
```

### Q: 模块导入失败

```bash
# 自动修复
python auto_fix.py

# 或手动安装依赖
pip install torch transformers stable-baselines3 sb3-contrib
```

### Q: CUDA 不可用

```bash
# 安装 CUDA 版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 训练流程

```
1. 诊断
   └── 环境检查 → 数据检查 → 模块检查

2. 训练
   └── GMM → DT → TQC → Sentiment

3. 验证
   └── 模型文件检查 → 指标验证
```

---

## 🎯 v4 新特性

- ✅ **端到端诊断** - 训练前自动检测问题
- ✅ **自动修复** - 一键修复常见问题
- ✅ **参数推荐** - 根据硬件自动推荐参数
- ✅ **批量大小自适应** - 根据数据量自动调整
- ✅ **详细日志** - 完整的错误追踪
- ✅ **快速模式** - 快速验证训练流程

---

## 📞 故障排除

如果遇到问题:

1. 运行 `python diagnose.py --all` 查看详细诊断
2. 运行 `python auto_fix.py` 尝试自动修复
3. 检查 `logs/` 目录下的日志文件

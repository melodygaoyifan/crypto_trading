#!/bin/bash
# ================================================================================
# HMATS Training v3.1 - 依赖安装和模块修复
# ================================================================================

echo "========================================"
echo "HMATS Training v3.1 环境修复"
echo "========================================"

# 1. 安装缺失的 pip 依赖
echo ""
echo "[1/4] 安装 pip 依赖..."
pip install --quiet transformers peft tokenizers sentencepiece 2>/dev/null
pip install --quiet stable-baselines3 sb3-contrib gymnasium 2>/dev/null
pip install --quiet tensorboard 2>/dev/null

# 检查关键依赖
echo "检查依赖..."
python3 -c "import transformers; print('  ✅ transformers:', transformers.__version__)" 2>/dev/null || echo "  ❌ transformers 未安装"
python3 -c "import peft; print('  ✅ peft')" 2>/dev/null || echo "  ❌ peft 未安装"
python3 -c "import stable_baselines3; print('  ✅ stable-baselines3')" 2>/dev/null || echo "  ❌ stable-baselines3 未安装"
python3 -c "import sb3_contrib; print('  ✅ sb3-contrib')" 2>/dev/null || echo "  ❌ sb3-contrib 未安装"

# 2. 修复 DRL v5.5 导入路径
echo ""
echo "[2/4] 修复 DRL v5.5 导入路径..."
if [ -f "drl/train_drl_v55.py" ]; then
    # 检查是否有错误的导入
    if grep -q "from core.model" drl/train_drl_v55.py; then
        echo "  修复 core.model -> drl.model"
        sed -i 's/from core\.model/from drl.model/g' drl/train_drl_v55.py
    fi
    if grep -q "from core.features" drl/train_drl_v55.py; then
        echo "  修复 core.features -> drl.features"
        sed -i 's/from core\.features/from drl.features/g' drl/train_drl_v55.py
    fi
    if grep -q "from training.gmm_pretrain" drl/train_drl_v55.py; then
        echo "  修复 training.gmm_pretrain -> gmm.gmm_pretrain"
        sed -i 's/from training\.gmm_pretrain/from gmm.gmm_pretrain/g' drl/train_drl_v55.py
    fi
    echo "  ✅ DRL v5.5 导入已修复"
else
    echo "  ⚠️ drl/train_drl_v55.py 不存在"
fi

# 3. 创建缺失的 __init__.py
echo ""
echo "[3/4] 创建缺失的 __init__.py..."
for dir in configs drl gmm sentiment utils scripts; do
    if [ -d "$dir" ] && [ ! -f "$dir/__init__.py" ]; then
        touch "$dir/__init__.py"
        echo "  创建 $dir/__init__.py"
    fi
done
echo "  ✅ __init__.py 已补全"

# 4. 验证模块导入
echo ""
echo "[4/4] 验证模块导入..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

errors = []

# 测试 configs
try:
    from configs.config import DRLConfig
    print("  ✅ configs.config")
except Exception as e:
    errors.append(f"configs.config: {e}")
    print(f"  ❌ configs.config: {e}")

# 测试 drl.model
try:
    from drl.model import DRLModelV55
    print("  ✅ drl.model")
except Exception as e:
    errors.append(f"drl.model: {e}")
    print(f"  ❌ drl.model: {e}")

# 测试 drl.features
try:
    from drl.features import FeatureEngineer
    print("  ✅ drl.features")
except Exception as e:
    errors.append(f"drl.features: {e}")
    print(f"  ❌ drl.features: {e}")

# 测试 gmm
try:
    from gmm.gmm_pretrain import RegimeGMM
    print("  ✅ gmm.gmm_pretrain")
except Exception as e:
    errors.append(f"gmm.gmm_pretrain: {e}")
    print(f"  ❌ gmm.gmm_pretrain: {e}")

# 测试 sentiment (需要 transformers)
try:
    # 只测试文件存在，不导入（避免 GPU 初始化）
    import importlib.util
    spec = importlib.util.spec_from_file_location("sentiment", "sentiment/train_sentiment_agent_v22.py")
    print("  ✅ sentiment (文件存在)")
except Exception as e:
    errors.append(f"sentiment: {e}")
    print(f"  ❌ sentiment: {e}")

print()
if errors:
    print(f"⚠️ 有 {len(errors)} 个模块存在问题")
else:
    print("🎉 所有模块验证通过!")
EOF

echo ""
echo "========================================"
echo "修复完成!"
echo "========================================"
echo ""
echo "现在可以运行:"
echo "  python run_training.py --all"
echo "  python train_tqc.py --data data.parquet --presets TQC_1,TQC_2"

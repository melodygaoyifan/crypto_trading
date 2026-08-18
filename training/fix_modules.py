#!/usr/bin/env python3
"""
================================================================================
HMATS Training v3.1 - 模块修复脚本
================================================================================

修复内容:
1. 安装缺失的 pip 依赖
2. 修复 DRL v5.5 导入路径
3. 创建缺失的 __init__.py
4. 验证所有模块

用法:
    python fix_modules.py
    python fix_modules.py --install-deps
    python fix_modules.py --verify-only

================================================================================
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def install_dependencies():
    """安装缺失的依赖"""
    print("\n[1/4] 安装 pip 依赖...")
    
    packages = [
        "transformers",
        "peft", 
        "tokenizers",
        "sentencepiece",
        "stable-baselines3",
        "sb3-contrib",
        "gymnasium",
        "tensorboard",
    ]
    
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ON {pkg} 已安装")
        except ImportError:
            print(f"  📦 安装 {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], 
                          capture_output=True)


def fix_drl_imports():
    """修复 DRL v5.5 导入路径"""
    print("\n[2/4] 修复 DRL v5.5 导入路径...")
    
    drl_file = Path("drl/train_drl_v55.py")
    if not drl_file.exists():
        print("  [WARN]️ drl/train_drl_v55.py 不存在")
        return
    
    content = drl_file.read_text(encoding="utf-8")
    original = content
    
    replacements = [
        ("from core.model", "from drl.model"),
        ("from core.features", "from drl.features"),
        ("from training.gmm_pretrain", "from gmm.gmm_pretrain"),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  修复: {old} -> {new}")
    
    if content != original:
        drl_file.write_text(content, encoding="utf-8")
        print("  ON DRL v5.5 导入已修复")
    else:
        print("  ON DRL v5.5 导入已正确")


def create_init_files():
    """创建缺失的 __init__.py"""
    print("\n[3/4] 创建缺失的 __init__.py...")
    
    dirs = ["configs", "drl", "gmm", "sentiment", "utils", "scripts"]
    
    for d in dirs:
        dir_path = Path(d)
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  创建 {init_file}")
    
    print("  ON __init__.py 已补全")


def verify_modules():
    """验证模块导入"""
    print("\n[4/4] 验证模块导入...")
    
    sys.path.insert(0, ".")
    errors = []
    
    # 测试列表
    tests = [
        ("configs.config", "DRLConfig"),
        ("drl.model", "DRLModelV55"),
        ("drl.features", "FeatureEngineer"),
        ("gmm.gmm_pretrain", "RegimeGMM"),
    ]
    
    for module, cls in tests:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"  ON {module}.{cls}")
        except Exception as e:
            errors.append((module, str(e)))
            print(f"  OFF {module}: {e}")
    
    # 检查 sentiment (只检查文件)
    if Path("sentiment/train_sentiment_agent_v22.py").exists():
        print("  ON sentiment/train_sentiment_agent_v22.py (文件存在)")
    else:
        errors.append(("sentiment", "文件不存在"))
        print("  OFF sentiment: 文件不存在")
    
    # 检查 transformers
    try:
        import transformers
        print(f"  ON transformers {transformers.__version__}")
    except ImportError:
        errors.append(("transformers", "未安装"))
        print("  OFF transformers: 未安装")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="HMATS v3.1 模块修复")
    parser.add_argument("--install-deps", action="store_true", help="安装依赖")
    parser.add_argument("--verify-only", action="store_true", help="仅验证")
    args = parser.parse_args()
    
    print("=" * 50)
    print("HMATS Training v3.1 - 模块修复")
    print("=" * 50)
    
    if args.verify_only:
        errors = verify_modules()
    else:
        if args.install_deps:
            install_dependencies()
        fix_drl_imports()
        create_init_files()
        errors = verify_modules()
    
    print("\n" + "=" * 50)
    if errors:
        print(f"[WARN]️ 有 {len(errors)} 个问题需要手动解决:")
        for module, error in errors:
            print(f"  - {module}: {error}")
        print("\n建议:")
        print("  pip install transformers peft stable-baselines3 sb3-contrib")
    else:
        print("🎉 所有模块验证通过!")
        print("\n现在可以运行:")
        print("  python run_training.py --all")
        print("  python train_tqc.py --data data.parquet")
    print("=" * 50)


if __name__ == "__main__":
    main()

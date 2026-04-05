#!/usr/bin/env python3
"""
================================================================================
HMATS v4 - 自动修复脚本
================================================================================

自动检测并修复常见问题。

用法:
    python auto_fix.py

================================================================================
"""

import os
import sys
import subprocess
from pathlib import Path


def print_step(step: int, total: int, message: str):
    print(f"\n[{step}/{total}] {message}")
    print("-" * 50)


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           HMATS v4 - 自动修复                                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    total_steps = 5
    
    # 1. 创建必要目录
    print_step(1, total_steps, "创建目录结构")
    
    dirs = [
        "training_data/raw",
        "training_data/sentiment",
        "training_data/processed",
        "models",
        "logs",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    
    # 2. 创建 __init__.py
    print_step(2, total_steps, "创建 __init__.py")
    
    packages = ["configs", "drl", "gmm", "sentiment", "utils", "scripts"]
    
    for pkg in packages:
        pkg_dir = Path(pkg)
        if pkg_dir.exists():
            init_file = pkg_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  ✓ 创建 {init_file}")
            else:
                print(f"  - {init_file} 已存在")
    
    # 3. 修复导入路径
    print_step(3, total_steps, "修复导入路径")
    
    files_to_fix = [
        ("drl/train_drl_v55.py", [
            ("from core.model", "from drl.model"),
            ("from core.features", "from drl.features"),
            ("from training.gmm_pretrain", "from gmm.gmm_pretrain"),
        ]),
    ]
    
    for file_path, replacements in files_to_fix:
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            modified = False
            
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    print(f"  ✓ {file_path}: {old} -> {new}")
                    modified = True
            
            if modified:
                path.write_text(content)
            else:
                print(f"  - {file_path}: 无需修复")
        else:
            print(f"  [WARN] {file_path}: 文件不存在")
    
    # 4. 检查依赖
    print_step(4, total_steps, "检查依赖")
    
    required = {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
    }
    
    optional = {
        "transformers": "transformers",
        "peft": "peft",
        "stable_baselines3": "stable-baselines3",
        "sb3_contrib": "sb3-contrib",
    }
    
    missing_required = []
    missing_optional = []
    
    for module, pkg in required.items():
        try:
            __import__(module)
            print(f"  ✓ {pkg}")
        except ImportError:
            missing_required.append(pkg)
            print(f"  ✗ {pkg} (必需)")
    
    for module, pkg in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {pkg}")
        except ImportError:
            missing_optional.append(pkg)
            print(f"  [WARN] {pkg} (可选)")
    
    if missing_required:
        print(f"\n  安装必需依赖:")
        print(f"    pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print(f"\n  安装可选依赖 (完整功能):")
        print(f"    pip install {' '.join(missing_optional)}")
    
    # 5. 验证模块
    print_step(5, total_steps, "验证模块")
    
    sys.path.insert(0, '.')
    
    modules = [
        ("configs.config", "DRLConfig"),
        ("drl.model", "DRLModelV55"),
        ("drl.features", "FeatureEngineer"),
        ("gmm.gmm_pretrain", "RegimeGMM"),
    ]
    
    all_ok = True
    for module_path, cls_name in modules:
        try:
            module = __import__(module_path, fromlist=[cls_name])
            getattr(module, cls_name)
            print(f"  ✓ {module_path}.{cls_name}")
        except Exception as e:
            print(f"  ✗ {module_path}.{cls_name}: {e}")
            all_ok = False
    
    # 总结
    print("\n" + "="*60)
    if all_ok and not missing_required:
        print("ON 修复完成! 可以开始训练:")
        print("   python train_v4.py --all")
    else:
        print("[WARN]️ 部分问题需要手动解决")
        if missing_required:
            print(f"   pip install {' '.join(missing_required)}")
    print("="*60)


if __name__ == "__main__":
    main()

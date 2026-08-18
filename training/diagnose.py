#!/usr/bin/env python3
"""
================================================================================
HMATS v4 - 详细诊断工具
================================================================================

深度检查各个组件的问题并提供修复建议。

用法:
    python diagnose.py --all
    python diagnose.py --data
    python diagnose.py --modules
    python diagnose.py --gpu

================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_result(name: str, status: str, message: str, indent: int = 0):
    icon = {"ok": "ON", "warn": "[WARN]️", "error": "OFF", "info": "ℹ️"}[status]
    prefix = "  " * indent
    print(f"{prefix}{icon} {name}: {message}")


# =============================================================================
# GPU 诊断
# =============================================================================

def diagnose_gpu():
    """详细 GPU 诊断"""
    print_header("GPU 诊断")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_result("CUDA", "error", "不可用")
            print("\n  可能原因:")
            print("    1. 未安装 NVIDIA 驱动")
            print("    2. 安装的是 CPU 版 PyTorch")
            print("    3. CUDA 版本不兼容")
            print("\n  修复:")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
            return
        
        print_result("CUDA", "ok", f"v{torch.version.cuda}")
        print_result("cuDNN", "ok", f"v{torch.backends.cudnn.version()}")
        
        n_gpus = torch.cuda.device_count()
        print_result("GPU 数量", "ok", str(n_gpus))
        
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1e9
            mem_used = torch.cuda.memory_allocated(i) / 1e9
            
            print(f"\n  GPU {i}: {props.name}")
            print(f"    内存: {mem_used:.1f}GB / {mem_total:.1f}GB")
            print(f"    计算能力: {props.major}.{props.minor}")
            print(f"    多处理器: {props.multi_processor_count}")
            
            # 推荐 batch_size
            if mem_total >= 24:
                rec_batch = 512
            elif mem_total >= 16:
                rec_batch = 256
            elif mem_total >= 8:
                rec_batch = 128
            else:
                rec_batch = 64
            
            print(f"    推荐 batch_size: {rec_batch}")
        
        # 测试 CUDA 运算
        print("\n  CUDA 运算测试...")
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print_result("CUDA 运算", "ok", "正常", indent=1)
        except Exception as e:
            print_result("CUDA 运算", "error", str(e), indent=1)
        
        # 测试 AMP
        print("\n  混合精度 (AMP) 测试...")
        try:
            with torch.cuda.amp.autocast():
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
            print_result("AMP", "ok", "支持", indent=1)
        except Exception as e:
            print_result("AMP", "warn", f"不支持: {e}", indent=1)
            
    except ImportError:
        print_result("PyTorch", "error", "未安装")
        print("\n  修复: pip install torch")


# =============================================================================
# 数据诊断
# =============================================================================

def diagnose_data(data_dir: str = "./training_data"):
    """详细数据诊断"""
    print_header("数据诊断")
    
    import pandas as pd
    import numpy as np
    
    data_path = Path(data_dir)
    
    # OHLCV 数据
    print("\n--- OHLCV 数据 ---")
    ohlcv_dir = data_path / "raw"
    
    if not ohlcv_dir.exists():
        print_result("OHLCV 目录", "error", f"不存在: {ohlcv_dir}")
        print("\n  修复: python get_data.py --ohlcv")
        return
    
    for asset in ["SOL", "BTC", "ETH"]:
        file_path = ohlcv_dir / f"{asset}_60m.parquet"
        if not file_path.exists():
            file_path = ohlcv_dir / f"{asset}_60m.csv"
        
        if not file_path.exists():
            print_result(f"{asset}", "error", "文件不存在")
            continue
        
        try:
            if str(file_path).endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            print(f"\n  {asset}: {file_path.name}")
            print(f"    行数: {len(df):,}")
            print(f"    列: {list(df.columns)}")
            
            # 标准化列名检查
            df.columns = df.columns.str.lower()
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                print_result("必要列", "error", f"缺少: {missing}", indent=2)
            else:
                print_result("必要列", "ok", "完整", indent=2)
            
            # NaN 检查
            nan_counts = df[required].isna().sum()
            total_nan = nan_counts.sum()
            if total_nan > 0:
                print_result("NaN", "warn", f"{total_nan} 个", indent=2)
                for col, cnt in nan_counts.items():
                    if cnt > 0:
                        print(f"        {col}: {cnt}")
            else:
                print_result("NaN", "ok", "无", indent=2)
            
            # 数据范围
            if 'timestamp' in df.columns or 'date' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                df[time_col] = pd.to_datetime(df[time_col])
                print(f"    时间范围: {df[time_col].min()} ~ {df[time_col].max()}")
            
            # 数据量评估
            if len(df) < 3000:
                print_result("数据量", "error", f"不足 (需要 3000+)", indent=2)
            elif len(df) < 5000:
                print_result("数据量", "warn", f"偏少 (建议 5000+)", indent=2)
            else:
                print_result("数据量", "ok", f"充足", indent=2)
            
            # 价格异常检查
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    zeros = (df[col] == 0).sum()
                    negatives = (df[col] < 0).sum()
                    if zeros > 0 or negatives > 0:
                        print_result(f"{col} 异常", "warn", f"零值:{zeros}, 负值:{negatives}", indent=2)
            
        except Exception as e:
            print_result(f"{asset}", "error", f"读取失败: {e}")
    
    # 情感数据
    print("\n--- 情感数据 ---")
    sentiment_dir = data_path / "sentiment"
    
    if not sentiment_dir.exists():
        print_result("情感目录", "warn", f"不存在: {sentiment_dir}")
        print("    创建: mkdir -p training_data/sentiment")
        return
    
    sentiment_files = list(sentiment_dir.glob('*.parquet')) + list(sentiment_dir.glob('*.csv'))
    
    if not sentiment_files:
        print_result("情感数据", "warn", "无数据文件")
        print("    获取: python get_data.py --sentiment")
    else:
        total = 0
        for f in sentiment_files:
            try:
                if str(f).endswith('.parquet'):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                
                print(f"\n  {f.name}: {len(df):,} 条")
                print(f"    列: {list(df.columns)}")
                
                # 检查必要列
                if 'text' not in df.columns:
                    print_result("text 列", "error", "缺少", indent=2)
                if 'label' not in df.columns:
                    print_result("label 列", "warn", "缺少 (将无法监督训练)", indent=2)
                else:
                    label_counts = df['label'].value_counts()
                    print(f"    标签分布: {dict(label_counts)}")
                
                total += len(df)
            except Exception as e:
                print_result(f.name, "error", f"读取失败: {e}")
        
        if total >= 3000:
            print_result("总数据量", "ok", f"{total:,} 条")
        else:
            print_result("总数据量", "warn", f"{total:,} 条 (建议 3000+)")


# =============================================================================
# 模块诊断
# =============================================================================

def diagnose_modules():
    """详细模块诊断"""
    print_header("模块诊断")
    
    sys.path.insert(0, '.')
    
    # 核心依赖
    print("\n--- 核心依赖 ---")
    
    core_deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
    ]
    
    for module, name in core_deps:
        try:
            m = __import__(module)
            version = getattr(m, '__version__', 'unknown')
            print_result(name, "ok", f"v{version}")
        except ImportError:
            print_result(name, "error", "未安装")
    
    # 可选依赖
    print("\n--- 可选依赖 ---")
    
    optional_deps = [
        ("transformers", "Transformers", "Sentiment 训练"),
        ("peft", "PEFT", "LoRA 微调"),
        ("stable_baselines3", "Stable-Baselines3", "TQC 训练"),
        ("sb3_contrib", "SB3-Contrib", "TQC 算法"),
        ("tensorboard", "TensorBoard", "训练监控"),
    ]
    
    for module, name, usage in optional_deps:
        try:
            m = __import__(module)
            version = getattr(m, '__version__', 'ok')
            print_result(name, "ok", f"v{version}")
        except ImportError:
            print_result(name, "warn", f"未安装 ({usage})")
    
    # 内部模块
    print("\n--- 内部模块 ---")
    
    internal_modules = [
        ("configs.config", "DRLConfig", "配置模块"),
        ("drl.model", "DRLModelV55", "DRL 模型"),
        ("drl.features", "FeatureEngineer", "特征工程"),
        ("gmm.gmm_pretrain", "RegimeGMM", "GMM 模块"),
    ]
    
    for module_path, cls_name, description in internal_modules:
        try:
            module = __import__(module_path, fromlist=[cls_name])
            cls = getattr(module, cls_name)
            print_result(description, "ok", f"{module_path}.{cls_name}")
        except ImportError as e:
            print_result(description, "error", f"导入失败: {e}")
            
            # 提供修复建议
            if "core.model" in str(e) or "training.gmm" in str(e):
                print("      修复: 导入路径错误，使用新的 v3.1+ 包")
        except Exception as e:
            print_result(description, "error", f"{e}")
    
    # 训练脚本
    print("\n--- 训练脚本 ---")
    
    scripts = [
        ("train_v4.py", "v4 主脚本"),
        ("drl/train_decision_transformer_v32.py", "DT v3.2"),
        ("train_tqc.py", "TQC 训练"),
        ("sentiment/train_sentiment_agent_v22.py", "Sentiment v2.2"),
        ("gmm/gmm_pretrain.py", "GMM 预训练"),
        ("get_data.py", "数据获取"),
    ]
    
    for script, description in scripts:
        path = Path(script)
        if path.exists():
            # 语法检查
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(path)],
                capture_output=True, text=True
            , encoding="utf-8")
            if result.returncode == 0:
                print_result(description, "ok", script)
            else:
                print_result(description, "error", f"语法错误")
                print(f"        {result.stderr[:100]}")
        else:
            print_result(description, "error", f"不存在: {script}")


# =============================================================================
# 训练参数计算器
# =============================================================================

def calculate_params():
    """计算推荐的训练参数"""
    print_header("推荐训练参数")
    
    import torch
    import pandas as pd
    
    # GPU 内存
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_mem = 0
    
    # 数据量
    data_rows = 0
    data_path = Path("./training_data/raw")
    if data_path.exists():
        for f in data_path.glob("*.parquet"):
            try:
                df = pd.read_parquet(f)
                data_rows = max(data_rows, len(df))
            except:
                pass
    
    print(f"\n  检测到:")
    print(f"    GPU 内存: {gpu_mem:.1f}GB")
    print(f"    最大数据量: {data_rows:,} 行")
    
    # 计算参数
    print(f"\n  推荐参数:")
    
    # DT batch_size
    if gpu_mem >= 24:
        dt_batch = 256
    elif gpu_mem >= 16:
        dt_batch = 128
    elif gpu_mem >= 8:
        dt_batch = 64
    else:
        dt_batch = 32
    
    # 根据数据量调整
    if data_rows > 0:
        # 预估轨迹数 (context_length=96, step=48)
        est_trajectories = (data_rows - 96 - 48) // 48
        if est_trajectories < dt_batch * 2:
            dt_batch = max(16, est_trajectories // 4)
    
    print(f"    DT batch_size: {dt_batch}")
    print(f"    DT epochs: {100 if data_rows >= 5000 else 50}")
    
    # TQC
    if data_rows >= 10000:
        tqc_steps = 2000000
    elif data_rows >= 5000:
        tqc_steps = 1000000
    else:
        tqc_steps = 500000
    
    print(f"    TQC timesteps: {tqc_steps:,}")
    
    # 命令
    print(f"\n  推荐命令:")
    print(f"    python train_v4.py --all --dt-batch-size {dt_batch} --tqc-timesteps {tqc_steps}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMATS v4 详细诊断工具')
    parser.add_argument('--all', action='store_true', help='运行所有诊断')
    parser.add_argument('--gpu', action='store_true', help='GPU 诊断')
    parser.add_argument('--data', action='store_true', help='数据诊断')
    parser.add_argument('--modules', action='store_true', help='模块诊断')
    parser.add_argument('--params', action='store_true', help='计算推荐参数')
    parser.add_argument('--data-dir', type=str, default='./training_data', help='数据目录')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           HMATS v4 - 详细诊断工具                            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.all or (not any([args.gpu, args.data, args.modules, args.params])):
        diagnose_gpu()
        diagnose_data(args.data_dir)
        diagnose_modules()
        calculate_params()
    else:
        if args.gpu:
            diagnose_gpu()
        if args.data:
            diagnose_data(args.data_dir)
        if args.modules:
            diagnose_modules()
        if args.params:
            calculate_params()
    
    print("\n" + "="*60)
    print("诊断完成!")
    print("="*60)


if __name__ == "__main__":
    main()

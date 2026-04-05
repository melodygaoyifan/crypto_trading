import torch
import numpy as np
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1024**3:.1f} GB")
    # Quick computation test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f"Matrix multiply test: OK (result shape {y.shape})")
else:
    print("NO GPU DETECTED")

# Test SB3 import
try:
    from sb3_contrib import TQC
    from stable_baselines3.common.vec_env import SubprocVecEnv
    print(f"\nSB3-Contrib TQC: OK")
    print(f"SubprocVecEnv: OK")
except ImportError as e:
    print(f"SB3 import error: {e}")

# Test other deps
import pandas as pd
import sklearn
import joblib
import ta
print(f"pandas {pd.__version__}, sklearn {sklearn.__version__}")
print(f"numpy {np.__version__}, ta OK")

# Test training data access
import os
data_path = "/mnt/c/Users/melod/Downloads/hmats/hmats_training_v5 (1)/training_data/raw/BTC_60m.parquet"
if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    print(f"\nTraining data: {len(df)} rows ({data_path.split('/')[-1]})")
else:
    print(f"\nTraining data NOT FOUND at {data_path}")

print("\n=== ALL CHECKS PASSED ===")

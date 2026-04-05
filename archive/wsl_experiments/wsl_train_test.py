"""Quick test to verify training pipeline works in WSL."""
import sys
import os

# Add project paths
project_root = "/mnt/c/Users/melod/Downloads/hmats"
training_dir = os.path.join(project_root, "hmats_training_v5 (1)")
sys.path.insert(0, project_root)
sys.path.insert(0, training_dir)

# Test 1: FeatureEngineer
from drl.features import FeatureEngineer
fe = FeatureEngineer()
print(f"[OK] FeatureEngineer: {fe.state_dim} dims")

# Test 2: Load training data and compute features
import pandas as pd
data_path = os.path.join(training_dir, "training_data/raw/BTC_60m.parquet")
df = pd.read_parquet(data_path)
print(f"[OK] BTC raw data: {len(df)} rows, cols: {list(df.columns)}")

# Rename columns for FeatureEngineer
df.columns = df.columns.str.lower()
if 'volume btc' in df.columns:
    df = df.rename(columns={'volume btc': 'volume_btc', 'volume_usdt': 'volume'})
if 'volume' not in df.columns and 'volume_usdt' in df.columns:
    df['volume'] = df['volume_usdt']

# Test feature computation on a subset
subset = df.tail(2000).copy()
features = fe.compute_features(subset)
float_cols = [c for c in features.columns if features[c].dtype in ['float64', 'float32']
              and c not in ['timestamp', 'unix_timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_btc', 'trade_count']]
print(f"[OK] Features computed: {len(float_cols)} float columns")

# Test 3: TQC import
from sb3_contrib import TQC
print(f"[OK] TQC importable")

# Test 4: GPU availability
import torch
print(f"[OK] GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB)")

# Test 5: Quick TQC env test
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(115, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(115).astype(np.float32)
        reward = float(np.random.randn())
        terminated = self.step_count >= 100
        return obs, reward, terminated, False, {}

env = DummyEnv()
model = TQC("MlpPolicy", env, verbose=0, device="cuda",
            policy_kwargs={"net_arch": [384, 384, 256], "n_quantiles": 32})
print(f"[OK] TQC model created on GPU (params: {sum(p.numel() for p in model.policy.parameters()):,})")

# Quick learn test (100 steps)
model.learn(total_timesteps=500, log_interval=None)
print(f"[OK] TQC training test: 500 steps completed")

print("\n=== WSL TRAINING PIPELINE READY ===")

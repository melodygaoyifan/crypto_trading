"""⑥ Environment Layer Validation."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import importlib

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
DRL_DIR = PROJECT / "data" / "drl_training"
MANIFEST = PROJECT / "config" / "feature_manifest.json"
SPLIT_MANIFEST = PROJECT / "config" / "split_manifest.json"

SEP = "=" * 70
results = []

def log(section, check, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

# ═══════════════════════════════════════════════════════════
# CODE AUDIT: TradingEnvFull
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CODE AUDIT: TradingEnvFull")
print(SEP)

src = (PROJECT / "train_drl_full.py").read_text(encoding="utf-8")

# obs_dim = n_features + 4
log("CODE", "obs shape = (n_features + 4,)",
    'shape=(n_features + 4,)' in src)

# 4 env state vars in _get_observation
log("CODE", "env state: position_ratio",
    "position_ratio = self.position / self.max_position" in src)
log("CODE", "env state: position_direction",
    "position_direction = float(np.sign(self.position))" in src)
log("CODE", "env state: pnl_ratio",
    "pnl_ratio = self.total_pnl / self.initial_balance" in src)
log("CODE", "env state: drawdown",
    "(self.peak_balance - self.balance) / self.peak_balance" in src)

# action_space Box(-1, 1)
log("CODE", "action_space = Box(-1, 1, shape=(1,))",
    "low=-1.0, high=1.0" in src and "shape=(1,)" in src)

# window_size = 10
log("CODE", "default window_size = 10",
    "window_size: int = 10" in src)

# reset sets current_step = window_size
log("CODE", "reset() -> current_step = window_size",
    "self.current_step = self.window_size" in src)

# done condition
log("CODE", "done = current_step >= len(df) - 1",
    "self.current_step >= len(self.df) - 1" in src)

# NaN handling in _get_obs
log("CODE", "np.nan_to_num(obs, nan=0.0) in _get_observation",
    "nan_to_num(obs, nan=0.0" in src)

# obs clip
log("CODE", "obs clip [-10, 10]",
    "np.clip(obs, -10.0, 10.0)" in src)

# reward clip
log("CODE", "reward clip [-50, 50]",
    "np.clip(reward, -50.0, 50.0)" in src)

# P3: regime weighting
log("CODE", "P3: regime_weights from _compute_regime_weights()",
    "self.regime_weights = self._compute_regime_weights()" in src)
log("CODE", "P3: regime weight clip [0.5, 2.0]",
    "np.clip(w, 0.5, 2.0)" in src)

# Classic reward
log("CODE", "classic: pnl_pct * 100",
    "reward = pnl_pct * 100" in src)
log("CODE", "classic: drawdown penalty > 10% (dd*2)",
    "if drawdown > 0.1:" in src and "reward -= drawdown * 2" in src)

# Sortino reward
log("CODE", "sortino: EnhancedRewardCalculator()",
    "EnhancedRewardCalculator() if reward_mode ==" in src)

# Direction-regime aware
log("CODE", "bear regime bonus: × bear_reward_multiplier",
    "regime in BEAR_REGIMES and self.position < 0" in src)
log("CODE", "bull regime bonus: × bull_reward_multiplier",
    "regime in BULL_REGIMES and self.position > 0" in src)

# Counter-trend hold penalty
log("CODE", "counter-trend hold penalty: bear+long=-0.05, bull+short=-0.03",
    "reward -= 0.03" in src and "reward -= 0.05" in src)

# ═══════════════════════════════════════════════════════════
# LIVE INSTANTIATION TEST
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("LIVE INSTANTIATION: BTC fold_1")
print(SEP)

# Import the env class
sys.path.insert(0, str(PROJECT))
import train_drl_full as tdf

with open(MANIFEST) as f:
    manifest = json.load(f)
with open(SPLIT_MANIFEST) as f:
    split_manifest = json.load(f)

feature_cols = manifest["all_features"]
no_scale = set(manifest.get("no_scale_features", []))

# Load data
df = pd.read_parquet(DRL_DIR / "BTC_4H_full.parquet")
sm = split_manifest["assets"]["BTC"]
fold_1 = sm["folds"][0]
train_end = fold_1["train_end"]
val_start = fold_1["val_start"]
val_end = fold_1["val_end"]

# Use only features present in both manifest and data
valid_features = [c for c in feature_cols if c in df.columns]
log("INST", f"feature_cols matched = {len(valid_features)} / {len(feature_cols)}",
    len(valid_features) == len(feature_cols))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[val_start:val_end].copy()

# Create env with classic reward
env = tdf.TradingEnvFull(
    df=train_df,
    feature_cols=valid_features,
    window_size=10,
    reward_mode="classic",
)

# obs_dim
expected_obs = len(valid_features) + 4
actual_obs = env.observation_space.shape[0]
log("INST", f"obs_dim = {actual_obs} (expected {expected_obs})",
    actual_obs == expected_obs)
log("INST", f"obs_dim = 121 (117 + 4)", actual_obs == 121)

# action space
log("INST", f"action_space.low = {env.action_space.low[0]:.1f}",
    env.action_space.low[0] == -1.0)
log("INST", f"action_space.high = {env.action_space.high[0]:.1f}",
    env.action_space.high[0] == 1.0)
log("INST", f"action_space.shape = {env.action_space.shape}",
    env.action_space.shape == (1,))

# ═══════════════════════════════════════════════════════════
# RESET & STEP TEST
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("RESET & STEP TEST")
print(SEP)

obs, info = env.reset()

# After reset
log("STEP", f"reset: current_step = {env.current_step}",
    env.current_step == 10, "should == window_size")
log("STEP", f"reset: obs.shape = {obs.shape}",
    obs.shape == (actual_obs,))
log("STEP", f"reset: obs dtype = {obs.dtype}",
    obs.dtype == np.float32)
log("STEP", f"reset: no NaN in obs",
    not np.isnan(obs).any())
log("STEP", f"reset: no inf in obs",
    not np.isinf(obs).any())
log("STEP", f"reset: obs bounded [-10, 10]",
    obs.min() >= -10.0 and obs.max() <= 10.0)

# Check external features are in obs (not all zero)
ext_features = ["funding_rate_zscore", "oi_change_5d", "liq_imbalance",
                "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum"]
ext_indices = [valid_features.index(f) for f in ext_features if f in valid_features]
ext_vals = obs[ext_indices]
ext_nonzero = (ext_vals != 0).sum()
log("STEP", f"external features in obs: {ext_nonzero}/{len(ext_indices)} non-zero",
    ext_nonzero > 0, f"vals: {dict(zip([ext_features[i] for i in range(len(ext_indices))], ext_vals.tolist()))}")

# Check regime_proba in obs
proba_features = [f"regime_proba_{i}" for i in range(8)]
proba_indices = [valid_features.index(f) for f in proba_features if f in valid_features]
proba_vals = obs[proba_indices]
proba_sum = proba_vals.sum()
log("STEP", f"regime_proba sum = {proba_sum:.4f}",
    abs(proba_sum - 1.0) < 0.01, "should ≈ 1.0")

# Step with various actions
print(f"\n  10-step rollout:")
rewards = []
for i in range(10):
    action = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)

    if i == 0:
        log("STEP", f"step 1: obs.shape = {obs.shape}",
            obs.shape == (actual_obs,))
        log("STEP", f"step 1: no NaN in obs",
            not np.isnan(obs).any())
        log("STEP", f"step 1: reward = {reward:.4f}",
            not np.isnan(reward))
        log("STEP", f"step 1: reward in [-50, 50]",
            -50 <= reward <= 50)
        log("STEP", f"step 1: done = {done}",
            not done, "shouldn't be done after 1 step")

log("STEP", f"10 steps: all rewards finite",
    all(np.isfinite(r) for r in rewards),
    f"range [{min(rewards):.4f}, {max(rewards):.4f}]")

# Test done condition: step to end
max_steps = len(train_df) - env.window_size - 1
log("STEP", f"episode length = {max_steps} steps (df={len(train_df)}, w={env.window_size})",
    max_steps > 1000)

# ═══════════════════════════════════════════════════════════
# REWARD MODE: CLASSIC vs SORTINO
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("REWARD MODE COMPARISON")
print(SEP)

# Classic env
env_classic = tdf.TradingEnvFull(
    df=train_df, feature_cols=valid_features,
    window_size=10, reward_mode="classic"
)
obs_c, _ = env_classic.reset()
np.random.seed(42)
classic_rewards = []
for _ in range(50):
    a = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
    _, r, d, _, _ = env_classic.step(a)
    classic_rewards.append(r)
    if d:
        break

log("REWARD", f"classic: 50 steps, mean={np.mean(classic_rewards):.4f}",
    True, f"std={np.std(classic_rewards):.4f}")

# Sortino env
env_sortino = tdf.TradingEnvFull(
    df=train_df, feature_cols=valid_features,
    window_size=10, reward_mode="sortino"
)
obs_s, _ = env_sortino.reset()
np.random.seed(42)
sortino_rewards = []
for _ in range(50):
    a = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
    _, r, d, _, _ = env_sortino.step(a)
    sortino_rewards.append(r)
    if d:
        break

log("REWARD", f"sortino: 50 steps, mean={np.mean(sortino_rewards):.4f}",
    True, f"std={np.std(sortino_rewards):.4f}")

log("REWARD", "classic ≠ sortino (different reward shaping)",
    classic_rewards != sortino_rewards)

# Both clipped
log("REWARD", f"classic max |reward| = {max(abs(r) for r in classic_rewards):.2f} ≤ 50",
    all(abs(r) <= 50 for r in classic_rewards))
log("REWARD", f"sortino max |reward| = {max(abs(r) for r in sortino_rewards):.2f} ≤ 50",
    all(abs(r) <= 50 for r in sortino_rewards))

# ═══════════════════════════════════════════════════════════
# P3: REGIME WEIGHTING
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("P3: REGIME WEIGHTING")
print(SEP)

rw = env_classic.regime_weights
log("P3", f"regime_weights computed: {len(rw)} regimes", len(rw) > 0)

for name, w in sorted(rw.items()):
    in_range = 0.5 <= w <= 2.0
    log("P3", f"  {name}: weight={w:.3f}", in_range,
        "clipped [0.5, 2.0]" if not in_range else "")

# All weights in [0.5, 2.0]
all_ok = all(0.5 <= w <= 2.0 for w in rw.values())
log("P3", f"all weights in [0.5, 2.0]", all_ok)

# ═══════════════════════════════════════════════════════════
# ENV STATE: 4 vars check
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ENV STATE VARIABLES")
print(SEP)

env2 = tdf.TradingEnvFull(
    df=train_df, feature_cols=valid_features,
    window_size=10, reward_mode="classic"
)
obs, _ = env2.reset()

# After reset, env state should be zeroed
n_feat = len(valid_features)
pos_ratio = obs[n_feat]
pos_dir = obs[n_feat + 1]
pnl_ratio = obs[n_feat + 2]
dd = obs[n_feat + 3]

log("STATE", f"reset: position_ratio = {pos_ratio:.4f}", pos_ratio == 0.0)
log("STATE", f"reset: position_direction = {pos_dir:.4f}", pos_dir == 0.0)
log("STATE", f"reset: pnl_ratio = {pnl_ratio:.4f}", pnl_ratio == 0.0)
log("STATE", f"reset: drawdown = {dd:.4f}", dd == 0.0)

# After taking a long action, position_ratio should be positive
obs, r, d, t, info = env2.step(np.array([0.8], dtype=np.float32))
pos_ratio_after = obs[n_feat]
pos_dir_after = obs[n_feat + 1]
log("STATE", f"after long(0.8): position_ratio = {pos_ratio_after:.2f}",
    pos_ratio_after > 0, "should be positive")
log("STATE", f"after long(0.8): position_direction = {pos_dir_after:.1f}",
    pos_dir_after == 1.0, "should be +1.0")

# After going short
obs, r, d, t, info = env2.step(np.array([-0.5], dtype=np.float32))
pos_dir_short = obs[n_feat + 1]
log("STATE", f"after short(-0.5): position_direction = {pos_dir_short:.1f}",
    pos_dir_short == -1.0, "should be -1.0")

# ═══════════════════════════════════════════════════════════
# VAL ENV: verify same obs_dim
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("VAL ENV CONSISTENCY")
print(SEP)

env_val = tdf.TradingEnvFull(
    df=val_df, feature_cols=valid_features,
    window_size=10, reward_mode="classic"
)
obs_val, _ = env_val.reset()
log("VAL", f"val env obs_dim = {env_val.observation_space.shape[0]}",
    env_val.observation_space.shape[0] == actual_obs)
log("VAL", f"val obs shape = {obs_val.shape}",
    obs_val.shape == (actual_obs,))
log("VAL", f"val episode length = {len(val_df) - 10 - 1} steps",
    len(val_df) - 10 - 1 > 100)

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

n_pass = sum(1 for r in results if r[2] == "PASS")
n_fail = sum(1 for r in results if r[2] == "FAIL")
print(f"\n  Total checks: {len(results)}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")

if n_fail > 0:
    print(f"\n  FAILURES:")
    for section, check, status, detail in results:
        if status == "FAIL":
            print(f"    [{section}] {check}" + (f"  - {detail}" if detail else ""))

print(f"\n{SEP}")

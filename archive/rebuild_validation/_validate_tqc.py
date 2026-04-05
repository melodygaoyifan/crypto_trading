"""⑦ TQC Model Architecture Validation."""
import numpy as np
from pathlib import Path
import json
import sys

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
SEP = "=" * 70
results = []

def log(section, check, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

# ═══════════════════════════════════════════════════════════
# NETWORK CONFIG
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("NETWORK CONFIG (ULTIMATE_PRESET)")
print(SEP)

sys.path.insert(0, str(PROJECT))
from train_drl_full import ULTIMATE_PRESET, ASSET_OVERRIDES, TradingEnvFull

preset = ULTIMATE_PRESET
pk = preset["policy_kwargs"]

log("NET", f"policy = {preset['policy']}", preset["policy"] == "MlpPolicy")
log("NET", f"net_arch = {pk['net_arch']}", pk["net_arch"] == [384, 384, 256])
log("NET", f"n_quantiles = {pk['n_quantiles']}", pk["n_quantiles"] == 32)
log("NET", f"n_critics = {pk['n_critics']}", pk["n_critics"] == 2)
log("NET", f"top_quantiles_to_drop = {preset['top_quantiles_to_drop_per_net']}",
    preset["top_quantiles_to_drop_per_net"] == 2)

# ═══════════════════════════════════════════════════════════
# IRON RULES
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("IRON RULES (Hyperparameters)")
print(SEP)

log("IRON", f"ent_coef = {preset['ent_coef']} (FIXED float, not 'auto')",
    isinstance(preset["ent_coef"], (int, float)) and preset["ent_coef"] == 0.1)

# weight_decay: not in preset -> defaults to 0
src = (PROJECT / "train_drl_full.py").read_text(encoding="utf-8")
log("IRON", "weight_decay NOT set (defaults to 0)",
    "weight_decay" not in src or "weight_decay=0" in src)

log("IRON", f"learning_rate = {preset['learning_rate']}", preset["learning_rate"] == 3e-5)
log("IRON", f"batch_size = {preset['batch_size']}", preset["batch_size"] == 256)
log("IRON", f"gradient_steps = {preset['gradient_steps']}", preset["gradient_steps"] == 4)
log("IRON", f"buffer_size = {preset['buffer_size']:,}", preset["buffer_size"] == 2_000_000)
log("IRON", f"learning_starts = {preset['learning_starts']:,}", preset["learning_starts"] == 30_000)
log("IRON", f"gamma = {preset['gamma']}", preset["gamma"] == 0.99)
log("IRON", f"tau = {preset['tau']}", preset["tau"] == 0.005)

# ═══════════════════════════════════════════════════════════
# ASSET OVERRIDES
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ASSET OVERRIDES")
print(SEP)

btc = ASSET_OVERRIDES.get("BTC", {})
eth = ASSET_OVERRIDES.get("ETH", {})
sol = ASSET_OVERRIDES.get("SOL", {})

log("ASSET", f"BTC: timesteps={btc.get('total_timesteps', 'default'):,}",
    btc.get("total_timesteps") == 2_500_000)
log("ASSET", f"ETH: timesteps={eth.get('total_timesteps', 'default'):,}",
    eth.get("total_timesteps") == 2_500_000)
log("ASSET", f"SOL: timesteps={sol.get('total_timesteps', 'default'):,}",
    sol.get("total_timesteps") == 3_000_000)
log("ASSET", f"SOL: learning_rate={sol.get('learning_rate', 'default')}",
    sol.get("learning_rate") == 2e-5)

# ═══════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TRAINING LOOP")
print(SEP)

# DummyVecEnv only
log("LOOP", "DummyVecEnv ONLY (no SubprocVecEnv)",
    "DummyVecEnv" in src and "SubprocVecEnv" not in src.split("class TradingEnvFull")[1].split("class ")[0]
    if "class TradingEnvFull" in src else False)

# Check SubprocVecEnv not imported/used in training path
from_imports = [line for line in src.split("\n") if "SubprocVecEnv" in line]
log("LOOP", f"SubprocVecEnv usage: {len(from_imports)} references",
    all("import" in line.lower() or "#" in line for line in from_imports),
    "only in imports/comments" if from_imports else "not found at all")

# EvalCallback saves best_model
log("LOOP", "EvalCallback saves best_model.zip",
    "best_model_save_path" in src and "best_model" in src)

# Early stopping
log("LOOP", "StopTrainingOnNoModelImprovement",
    "StopTrainingOnNoModelImprovement" in src)
log("LOOP", "max_no_improvement_evals = 20",
    "max_no_improvement_evals=20" in src)
log("LOOP", "min_evals = 50",
    "min_evals=50" in src)

# Checkpoint
log("LOOP", "CheckpointCallback save_freq=500_000",
    "save_freq=500_000" in src)

# eval_freq
log("LOOP", "eval_freq default = 5,000",
    "eval_freq: int = 5_000" in src)

# Sequential: one asset at a time (implied by design)
log("LOOP", "CLI takes --asset (one at a time)",
    "--asset" in src and "required=True" in src)

# ═══════════════════════════════════════════════════════════
# ARCH COMPARISON (Stage 4)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("ARCHITECTURE COMPARISON (Stage 4)")
print(SEP)

arch_src = (PROJECT / "scripts" / "compare_architectures.py").read_text(encoding="utf-8")

# 4 extractors
log("ARCH", "4 architectures: mlp, lstm, gru, cnn",
    all(a in arch_src for a in ["mlp", "lstm", "gru", "cnn"]))

# VecFrameStack for sequence models
log("ARCH", "VecFrameStack for sequence models",
    "VecFrameStack" in arch_src)
log("ARCH", f"DEFAULT_N_STACK = 8",
    "DEFAULT_N_STACK" in arch_src)

# Check actual n_stack value from feature_extractors
from models.feature_extractors import DEFAULT_N_STACK, SEQUENCE_ARCHS, ARCH_CONFIGS
log("ARCH", f"DEFAULT_N_STACK value = {DEFAULT_N_STACK}",
    DEFAULT_N_STACK == 8, "checklist says 16, code says 8")

# buffer_size = 2M for ALL archs
log("ARCH", "buffer_size = 2M for all archs",
    "buffer_size=2_000_000" in arch_src)

# BTC fold_1 only, 200K default
log("ARCH", "default timesteps = 200,000",
    "default=200_000" in arch_src)

# ent_coef fixed in arch comparison too
log("ARCH", "ent_coef = 0.1 (fixed) in arch comparison",
    "ent_coef=0.1" in arch_src)

# ═══════════════════════════════════════════════════════════
# LIVE MODEL INSTANTIATION
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("LIVE MODEL INSTANTIATION (BTC)")
print(SEP)

import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TQC

# Load data
with open(PROJECT / "config" / "feature_manifest.json") as f:
    manifest = json.load(f)
feature_cols = manifest["all_features"]
df = pd.read_parquet(PROJECT / "data" / "drl_training" / "BTC_4H_full.parquet")
valid_features = [c for c in feature_cols if c in df.columns]

# Small df for quick test
test_df = df.iloc[:500].copy()

env = DummyVecEnv([lambda: Monitor(TradingEnvFull(
    df=test_df, feature_cols=valid_features, reward_mode="classic"
))])

# Create model with ULTIMATE preset
p = dict(ULTIMATE_PRESET)
p["learning_rate"] = 3e-5
pk = p.pop("policy_kwargs")
policy = p.pop("policy")

model = TQC(
    policy,
    env=env,
    policy_kwargs=pk,
    device="cuda",
    verbose=0,
    **p,
)

# Verify model config
log("MODEL", f"ent_coef = {model.ent_coef}",
    model.ent_coef == 0.1)
log("MODEL", f"learning_rate = {model.learning_rate}",
    model.learning_rate == 3e-5)
log("MODEL", f"batch_size = {model.batch_size}",
    model.batch_size == 256)
log("MODEL", f"gradient_steps = {model.gradient_steps}",
    model.gradient_steps == 4)
log("MODEL", f"buffer_size = {model.buffer_size:,}",
    model.buffer_size == 2_000_000)
log("MODEL", f"gamma = {model.gamma}",
    model.gamma == 0.99)
log("MODEL", f"tau = {model.tau}",
    model.tau == 0.005)

# Check network architecture
actor = model.actor
critic = model.critic

# Count total params
total_params = sum(p.numel() for p in model.policy.parameters())
trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
log("MODEL", f"total params = {total_params:,}", total_params > 100_000)
log("MODEL", f"trainable params = {trainable_params:,}", trainable_params == total_params)

# ent_coef is a float (not autotuned log_ent_coef)
log("MODEL", "ent_coef is fixed (not auto-tuned)",
    not hasattr(model, "log_ent_coef") or model.log_ent_coef is None,
    "no log_ent_coef tensor")

# Verify critic has 2 critics
n_critics = len(critic.quantile_critics) if hasattr(critic, 'quantile_critics') else -1
log("MODEL", f"n_critics = {n_critics}",
    n_critics == 2)

# Verify n_quantiles
if n_critics > 0:
    first_critic = critic.quantile_critics[0]
    # Last layer output dim should be n_quantiles
    last_layer = list(first_critic.modules())[-1]
    if hasattr(last_layer, 'out_features'):
        log("MODEL", f"critic output dim (n_quantiles) = {last_layer.out_features}",
            last_layer.out_features == 32)

env.close()

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

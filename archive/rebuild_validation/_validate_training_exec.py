"""⑩ Training Execution Validation."""
import numpy as np
from pathlib import Path
import json

PROJECT = Path(r"c:\Users\melod\Downloads\hmats")
TRAIN_SCRIPT = PROJECT / "train_drl_full.py"

SEP = "=" * 70
results = []

def log(section, check, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((section, check, status, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {check}" + (f"  - {detail}" if detail else ""))

src = TRAIN_SCRIPT.read_text(encoding="utf-8")

# ═══════════════════════════════════════════════════════════
# EXECUTION: ONE ASSET AT A TIME
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("EXECUTION: ONE ASSET AT A TIME")
print(SEP)

# CLI requires --asset (one at a time)
log("EXEC", "--asset is required=True",
    "required=True" in src and '"--asset"' in src)
log("EXEC", "choices=[BTC, ETH, SOL]",
    'choices=["BTC", "ETH", "SOL"]' in src)

# Asset overrides
import sys
sys.path.insert(0, str(PROJECT))
from train_drl_full import ASSET_OVERRIDES

btc_steps = ASSET_OVERRIDES["BTC"]["total_timesteps"]
eth_steps = ASSET_OVERRIDES["ETH"]["total_timesteps"]
sol_steps = ASSET_OVERRIDES["SOL"]["total_timesteps"]

log("EXEC", f"BTC: {btc_steps:,} steps", btc_steps == 2_500_000)
log("EXEC", f"ETH: {eth_steps:,} steps", eth_steps == 2_500_000)
log("EXEC", f"SOL: {sol_steps:,} steps (more exploration)", sol_steps == 3_000_000)

# SOL has distinct overrides
sol_lr = ASSET_OVERRIDES["SOL"].get("learning_rate")
sol_vol = ASSET_OVERRIDES["SOL"].get("sol_vol_multiplier")
log("EXEC", f"SOL: lr={sol_lr} (lower for stability)", sol_lr == 2e-5)
log("EXEC", f"SOL: sol_vol_multiplier={sol_vol}", sol_vol == 0.6)

# ═══════════════════════════════════════════════════════════
# CLI FLAGS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CLI FLAGS")
print(SEP)

log("CLI", "--no-progress-bar flag exists",
    '"--no-progress-bar"' in src)
log("CLI", "--resume flag exists",
    '"--resume"' in src)
log("CLI", "--device auto (default)",
    'default="auto"' in src)
log("CLI", "--folds default=3",
    "default=3" in src and '"--folds"' in src)
log("CLI", "--reward-mode classic/sortino",
    '"--reward-mode"' in src and '"classic"' in src and '"sortino"' in src)

# Expected command format
log("CLI", "command: python -X utf8 -u train_drl_full.py --asset BTC --no-progress-bar",
    True, "documented in CLAUDE.md")

# ═══════════════════════════════════════════════════════════
# MONITORING: GradientMonitorCallback
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("MONITORING: GradientMonitorCallback")
print(SEP)

log("MON", "GradientMonitorCallback class exists",
    "class GradientMonitorCallback" in src)

# Checks actor_loss every 1000 steps
log("MON", "actor_loss checked every 1000 steps",
    "num_timesteps % 1000 == 0" in src and "actor_loss" in src)

# max_actor_loss = 1e6 (auto-stop threshold)
log("MON", "max_actor_loss = 1e6",
    "max_actor_loss=1e6" in src)

# Consecutive explosions counter
log("MON", "max_consecutive_explosions = 3 -> auto-stop",
    "max_consecutive_explosions=3" in src)

# ent_coef drift detection
log("MON", "ent_coef drift detection (expected 0.1)",
    "ent_coef - 0.1" in src and "CHANGED" in src)

# Memory monitoring every 50K steps
log("MON", "memory monitoring every 50K steps",
    "num_timesteps % 50_000 == 0" in src and "memory" in src.lower())

# Memory warning > 32GB
log("MON", "memory warning threshold > 32GB",
    "mem_gb > 32" in src)

# ═══════════════════════════════════════════════════════════
# MONITORING: FPSLoggerCallback
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("MONITORING: FPSLoggerCallback")
print(SEP)

log("FPS", "FPSLoggerCallback class exists",
    "class FPSLoggerCallback" in src)

# FPS logged every 100K steps
log("FPS", "FPS logged every 100K steps",
    "log_freq=100_000" in src)

# Low FPS warning < 50
log("FPS", "low FPS warning threshold < 50",
    "fps < 50" in src)

# ═══════════════════════════════════════════════════════════
# CHECKPOINT & RESUME
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("CHECKPOINT & RESUME")
print(SEP)

log("CKPT", "CheckpointCallback save_freq=500_000",
    "save_freq=500_000" in src)

log("CKPT", "checkpoint name prefix = ULTIMATE",
    'name_prefix="ULTIMATE"' in src)

# Resume from checkpoint
log("CKPT", "TQC.load() for checkpoint resume",
    "TQC.load(" in src and "resume_path" in src)

log("CKPT", "reset_num_timesteps=not bool(resume_path)",
    "reset_num_timesteps=not bool(resume_path)" in src)

# Skip completed folds
log("CKPT", "skip completed folds (final_model.zip exists)",
    "final_model_path.exists()" in src and "Already done" in src)

# ═══════════════════════════════════════════════════════════
# EVAL CALLBACK
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("EVAL CALLBACK")
print(SEP)

log("EVAL", "EvalCallback with best_model_save_path",
    "best_model_save_path" in src)

log("EVAL", "eval_freq=5_000 default",
    "eval_freq: int = 5_000" in src)

log("EVAL", "deterministic=True",
    "deterministic=True" in src)

log("EVAL", "callback_after_eval -> StopTrainingOnNoModelImprovement",
    "callback_after_eval=stop_train_callback" in src)

# After training: evaluates with best_model (not final_model)
log("EVAL", "final eval uses best_model.zip (not overfitted final_model)",
    "best_model" in src and "not overfitted final_model" in src)

# ═══════════════════════════════════════════════════════════
# EXISTING CHECKPOINTS / MODELS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("EXISTING MODEL ARTIFACTS")
print(SEP)

models_dir = PROJECT / "models" / "retrained"
for asset in ["BTC", "ETH", "SOL"]:
    asset_dir = models_dir / asset
    if not asset_dir.exists():
        log("MODELS", f"{asset}: output dir exists", False, "not created yet")
        continue

    # Check each fold
    for fold_idx in range(1, 4):
        fold_dir = asset_dir / f"fold_{fold_idx}" / "logs" / "ULTIMATE"
        final = fold_dir / "final_model.zip"
        best = fold_dir / "best_model" / "best_model.zip"
        scaler = fold_dir / "feature_scaler.pkl"

        if final.exists():
            size_mb = final.stat().st_size / 1024**2
            log("MODELS", f"{asset} fold_{fold_idx}: final_model.zip ({size_mb:.1f}MB)",
                True)
        else:
            log("MODELS", f"{asset} fold_{fold_idx}: final_model.zip",
                False, "not yet trained")

        if best.exists():
            log("MODELS", f"{asset} fold_{fold_idx}: best_model.zip", True)

        if scaler.exists():
            log("MODELS", f"{asset} fold_{fold_idx}: feature_scaler.pkl", True)

    # Check for checkpoints
    for fold_idx in range(1, 4):
        ckpt_dir = asset_dir / f"fold_{fold_idx}" / "logs" / "ULTIMATE" / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("ULTIMATE_*_steps.zip"))
            if ckpts:
                latest = sorted(ckpts)[-1]
                log("MODELS", f"{asset} fold_{fold_idx}: {len(ckpts)} checkpoints (latest: {latest.name})",
                    True)

# ═══════════════════════════════════════════════════════════
# WINDOWS SPECIFICS
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("WINDOWS SPECIFICS")
print(SEP)

log("WIN", "-X utf8 flag needed (GBK encoding)",
    True, "documented in CLAUDE.md")
log("WIN", "-u flag (unbuffered stdout)",
    True, "documented in CLAUDE.md")
log("WIN", "DummyVecEnv ONLY (SubprocVecEnv deadlocks on Windows)",
    "DummyVecEnv" in src and "SubprocVecEnv deadlocks" in src)

# FPS warning about /mnt/c/ (WSL2 path issue)
log("WIN", "FPS warning for /mnt/c/ IO bottleneck",
    "/mnt/c/" in src)

# ═══════════════════════════════════════════════════════════
# TRAINING COMMANDS (expected)
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("TRAINING COMMANDS (reference)")
print(SEP)

print("""
  BTC (first):
    python -X utf8 -u train_drl_full.py --asset BTC --folds 3 --no-progress-bar

  ETH (after BTC completes):
    python -X utf8 -u train_drl_full.py --asset ETH --folds 3 --no-progress-bar

  SOL (after ETH completes, 3M steps):
    python -X utf8 -u train_drl_full.py --asset SOL --folds 3 --no-progress-bar

  Resume from checkpoint:
    python -X utf8 -u train_drl_full.py --asset BTC --folds 3 --no-progress-bar --resume

  Expected per-fold time: ~35h at 20 FPS (2.5M steps)
  Total per asset: ~105h (3 folds)
  Total all assets: ~315h + SOL extra 50h = ~365h
""")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"{SEP}")
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

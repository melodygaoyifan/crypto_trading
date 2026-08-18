"""
Launch Optuna trial as a truly detached Windows process.
Uses DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP flags.

Usage:
    python scripts/launch_optuna_trial.py
"""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.join(ROOT, "training")
LOG_OUT = os.path.join(ROOT, "logs", "stage8_trial49.log")
LOG_ERR = os.path.join(ROOT, "logs", "stage8_trial49_err.log")

cmd = [
    sys.executable, "-X", "utf8", "-u",
    os.path.join(TRAINING, "train_drl_full.py"),
    "--asset", "BTC",
    "--optuna",
    "--optuna-trials", "51",  # 50 existing + 1 new
    "--optuna-timesteps", "200000",
    "--no-progress-bar",
]

# Windows flags for true detachment
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

with open(LOG_OUT, "w", encoding="utf-8") as f_out, open(LOG_ERR, "w", encoding="utf-8") as f_err:
    proc = subprocess.Popen(
        cmd,
        stdout=f_out,
        stderr=f_err,
        cwd=TRAINING,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )

print(f"Launched PID={proc.pid}")
print(f"Stdout -> {LOG_OUT}")
print(f"Stderr -> {LOG_ERR}")
print(f"Monitor: tail -f logs/stage8_trial49_err.log")

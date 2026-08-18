"""
Resume Trial 48: Load saved 120K model, compute NAV, patch Optuna DB.
Then run 1 fresh Optuna trial (T49) to reach 50-trial target.

Usage:
    python -X utf8 -u scripts/resume_trial48.py
"""
import sys, os, time, logging
import sqlite3
import numpy as np

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.join(ROOT, "training")
# training/models must shadow root models/ - remove ROOT and CWD, put TRAINING first
sys.path = [p for p in sys.path
            if os.path.abspath(p) not in (os.path.abspath(ROOT), os.path.abspath('.'))]
sys.path.insert(0, TRAINING)
sys.path.insert(1, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───
ASSET = "BTC"
TRIAL_NUM = 48
DB_PATH = os.path.join(ROOT, "models", "retrained", ASSET, "optuna_hmats_v8.db")
MODEL_PATH = os.path.join(ROOT, "models", "retrained", ASSET, "optuna", f"trial_{TRIAL_NUM}", "best_model")
DATA_PATH = os.path.join(TRAINING, "training_data", "drl_training", f"{ASSET}_4H_full.parquet")
MANIFEST_PATH = os.path.join(ROOT, "configs", "feature_manifest.json")


def get_trial48_params():
    """Read Trial 48's hyperparams from Optuna DB."""
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    cur.execute(
        "SELECT param_name, param_value FROM trial_params "
        "WHERE trial_id = (SELECT trial_id FROM trials WHERE number = ?)",
        (TRIAL_NUM,),
    )
    params = {row[0]: row[1] for row in cur.fetchall()}
    db.close()
    logger.info(f"Trial {TRIAL_NUM} params: {params}")
    return params


def patch_trial_complete(nav_pct: float, best_raw_reward: float, elapsed_s: float):
    """Mark Trial 48 as COMPLETE in Optuna DB with NAV value."""
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()

    # Get trial_id
    cur.execute("SELECT trial_id FROM trials WHERE number = ?", (TRIAL_NUM,))
    trial_id = cur.fetchone()[0]

    # Update state: RUNNING -> COMPLETE, set datetime_complete
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    cur.execute(
        "UPDATE trials SET state = 'COMPLETE', datetime_complete = ? WHERE trial_id = ?",
        (now, trial_id),
    )

    # Insert trial value (NAV%)
    cur.execute(
        "INSERT INTO trial_values (trial_id, objective, value, value_type) "
        "VALUES (?, 0, ?, 'FINITE')",
        (trial_id, nav_pct),
    )

    # Insert user attributes
    import json
    for key, val in [
        ("nav_pct", nav_pct),
        ("best_raw_reward", best_raw_reward),
        ("elapsed_s", elapsed_s),
        ("resumed_from_120k", True),
    ]:
        cur.execute(
            "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?)",
            (trial_id, key, json.dumps(val)),
        )

    db.commit()
    db.close()
    logger.info(f"DB patched: Trial {TRIAL_NUM} -> COMPLETE, NAV={nav_pct:+.2f}%")


def main():
    import torch
    import pandas as pd
    import json
    from sklearn.preprocessing import RobustScaler
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib import TQC

    # Import feature_extractors BEFORE train_drl_full to avoid
    # root models/ caching in sys.modules
    from models.feature_extractors import (
        get_extractor_class, get_extractor_kwargs,
        SEQUENCE_ARCHS, DEFAULT_N_STACK,
    )
    from train_drl_full import TradingEnvFull

    t_start = time.time()

    # ─── Step 1: Load data & feature cols ───
    logger.info("=" * 60)
    logger.info(f"Resume Trial {TRIAL_NUM}: Evaluate 120K checkpoint")
    logger.info("=" * 60)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    feature_cols = manifest["all_features"]
    no_scale = set(manifest.get("no_scale_features", []))

    data = pd.read_parquet(DATA_PATH)
    logger.info(f"Data: {len(data)} rows, {len(feature_cols)} features")

    # ─── Step 2: Split same as Optuna (fold_1) ───
    n = len(data)
    val_size = int(n * 0.15)
    gap = 42
    window_size = 10
    val_end = n
    val_start = val_end - val_size
    train_end = val_start - gap

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[val_start:val_end].copy()
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Gap: {gap}")

    # ─── Step 3: Scale ───
    scale_cols = [c for c in feature_cols if c not in no_scale]
    scaler = RobustScaler()
    scaler.fit(train_df[scale_cols])
    scaler.scale_ = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    train_df[scale_cols] = scaler.transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    for df_ in [train_df, val_df]:
        df_[scale_cols] = df_[scale_cols].replace([np.inf, -np.inf], 0).fillna(0)
        df_[scale_cols] = df_[scale_cols].clip(-10.0, 10.0)

    # ─── Step 4: Get T48 params ───
    params = get_trial48_params()

    env_kwargs = dict(
        feature_cols=feature_cols,
        bear_reward_multiplier=1.2,
        bull_reward_multiplier=1.1,
        sol_vol_multiplier=1.0,
        reward_mode="classic",
        dd_threshold=params["dd_threshold"],
        dd_penalty_mult=params["dd_penalty_mult"],
        reward_clip=int(params["reward_clip"]),
        regime_weight_min=params["regime_weight_min"],
        regime_weight_max=params["regime_weight_max"],
    )

    # ─── Step 5: Create eval env ───
    eval_env = DummyVecEnv([lambda: Monitor(TradingEnvFull(df=val_df, **env_kwargs))])
    eval_env = VecFrameStack(eval_env, n_stack=DEFAULT_N_STACK)
    logger.info(f"Eval env: VecFrameStack n_stack={DEFAULT_N_STACK}, obs={eval_env.observation_space.shape}")

    # ─── Step 6: Load model ───
    model_file = MODEL_PATH
    if model_file.endswith(".zip"):
        model_file = model_file[:-4]
    logger.info(f"Loading model from: {model_file}")

    model = TQC.load(model_file, env=eval_env, device="cuda")
    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Model loaded: {total_params:,} params")

    # ─── Step 7: Compute NAV ───
    logger.info("Computing NAV (5 eval episodes)...")
    initial = eval_env.get_attr("initial_balance")[0]
    balances = []
    for ep in range(5):
        obs = eval_env.reset()
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            steps += 1
        final_balance = infos[0].get("balance", initial)
        balances.append(final_balance)
        logger.info(f"  Episode {ep+1}: balance={final_balance:,.2f} ({steps} steps)")

    mean_balance = np.mean(balances)
    nav_pct = (mean_balance - initial) / initial * 100

    # Get best raw reward from evaluations.npz
    eval_npz = os.path.join(os.path.dirname(MODEL_PATH + ".zip"), "evaluations.npz")
    evals = np.load(eval_npz)
    best_raw_reward = float(max(np.mean(r) for r in evals["results"]))

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {TRIAL_NUM} (120K checkpoint) results:")
    logger.info(f"  NAV = {nav_pct:+.2f}%")
    logger.info(f"  Mean final balance = {mean_balance:,.2f}")
    logger.info(f"  Best raw reward (from training) = {best_raw_reward:.2f}")
    logger.info(f"  Eval time = {elapsed:.1f}s")
    logger.info(f"{'='*60}")

    # ─── Step 8: Patch DB ───
    patch_trial_complete(nav_pct, best_raw_reward, elapsed)

    # Cleanup
    del model
    eval_env.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    logger.info(f"\nDone. Trial {TRIAL_NUM} is now COMPLETE in DB.")
    logger.info(f"Run Optuna with --optuna --optuna-trials 1 to create Trial 49.")
    return nav_pct


if __name__ == "__main__":
    nav = main()
    print(f"\nFinal NAV: {nav:+.2f}%")

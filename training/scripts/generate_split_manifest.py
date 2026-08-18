#!/usr/bin/env python3
"""
Generate configs/split_manifest.json - shared fold boundaries for all training layers.

Iron Rule #12: All models share the same train/val split to prevent data leakage.
This manifest is the single source of truth for fold boundaries used by:
  - GMM training (scripts/rebuild_pipeline.py)
  - DT v3.2 training
  - DRL v7 TQC training (train_drl_full.py)
  - Sentiment v2.2 training
  - Feature scaler fitting

Usage:
    python scripts/generate_split_manifest.py
    python scripts/generate_split_manifest.py --folds 3 --val-ratio 0.15 --gap 42
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_TRAINING_DIR = Path(__file__).resolve().parent.parent   # training/
PROJECT_ROOT = _TRAINING_DIR.parent                      # project root
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SplitManifest")

ASSETS = ["BTC", "ETH", "SOL"]
DATA_DIR = _TRAINING_DIR / "training_data" / "drl_training"


def generate(n_folds: int = 3, val_ratio: float = 0.15, gap: int = 42,
             purge_window: int = 0, window_size: int = 10) -> dict:
    """Generate fold split boundaries for all assets."""

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "params": {
            "n_folds": n_folds,
            "val_ratio": val_ratio,
            "gap": gap,
            "purge_window": purge_window,
            "window_size": window_size,
        },
        "assets": {},
    }

    for asset in ASSETS:
        path = DATA_DIR / f"{asset}_4H_full.parquet"
        if not path.exists():
            logger.warning(f"  {asset}: parquet not found, skipping")
            continue

        df = pd.read_parquet(path, columns=["timestamp", "close"])
        n = len(df)
        val_size = int(n * val_ratio)

        ts = pd.to_datetime(df["timestamp"])
        date_range = f"{ts.iloc[0].strftime('%Y-%m-%d')} -> {ts.iloc[-1].strftime('%Y-%m-%d')}"

        folds = []
        for fold_idx in range(n_folds):
            val_end = n - fold_idx * val_size
            val_start = val_end - val_size
            train_end = val_start - gap - purge_window

            if train_end < 5000:
                logger.warning(f"  {asset} fold_{fold_idx+1}: train_end={train_end} < 5000, skipping")
                continue

            eval_steps = val_size - window_size

            fold = {
                "fold": fold_idx + 1,
                "train_start": 0,
                "train_end": train_end,
                "train_rows": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "val_rows": val_end - val_start,
                "eval_steps": eval_steps,
                "gap_total": gap + purge_window,
                "train_date_start": ts.iloc[0].strftime("%Y-%m-%d"),
                "train_date_end": ts.iloc[train_end - 1].strftime("%Y-%m-%d"),
                "val_date_start": ts.iloc[val_start].strftime("%Y-%m-%d"),
                "val_date_end": ts.iloc[val_end - 1].strftime("%Y-%m-%d"),
            }
            folds.append(fold)

        manifest["assets"][asset] = {
            "total_rows": n,
            "date_range": date_range,
            "val_size": val_size,
            "n_folds": len(folds),
            "folds": folds,
        }

        logger.info(f"  {asset}: {n} rows, {len(folds)} folds, val_size={val_size}")
        for f in folds:
            logger.info(f"    fold_{f['fold']}: train=[0:{f['train_end']}] "
                        f"({f['train_date_start']}->{f['train_date_end']}), "
                        f"val=[{f['val_start']}:{f['val_end']}] "
                        f"({f['val_date_start']}->{f['val_date_end']})")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate split manifest")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--gap", type=int, default=42)
    parser.add_argument("--purge-window", type=int, default=0)
    args = parser.parse_args()

    manifest = generate(
        n_folds=args.folds,
        val_ratio=args.val_ratio,
        gap=args.gap,
        purge_window=args.purge_window,
    )

    out_path = PROJECT_ROOT / "configs" / "split_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""
HMATS v5.1 Phase 6 - Autoencoder Factor Extraction
====================================================

Trains a simple feedforward autoencoder over the 122-feature DRL manifest
to extract a small set (default 5) of latent factors. Each latent dim is
validated against forward 24h returns; the per-dim IC informs which
latent factors are promoted into the MLFactorFusionAgent.

Architecture:
    input (122) → encoder [64, 16, latent_dim=5] → decoder [16, 64, 122]
    Trained with Adam + MSE reconstruction loss + early-stop on val MSE.

Pipeline:
    1. Load training/training_data/drl_training/{ASSET}_4H_full.parquet
    2. Sample features (drop OHLCV + timestamp + regime + regime_confidence)
    3. RobustScaler fit on train split (skip no_scale features per manifest)
    4. Train autoencoder, save to models/factor_extraction/{ASSET}/...
    5. Compute per-latent-dim IC against forward_24h_return on holdout
    6. Save IC report; latent dims with IC > 0.05 stable → promote candidates

Constitutional override per __init__.py: operator-approved 2026-04-29.

Iron Laws:
  1. obs_dim=126 unchanged at runtime — autoencoder output is a NEW agent
     signal (ml_factor_*), not added to the DRL feature set.
  3. training/factor_extraction/ subdir-only touch (override scope).
  5. Does NOT touch training/drl/*. DRL ACTIVE preserved.
  6. ML factor enters fusion as ADVISE — quant DECIDE retained.

This module ships the architecture + training entry point. Actual training
runs require GPU/CPU time; operator launches via:
    python -X utf8 training/factor_extraction/autoencoder_factor.py \
        --asset BTC --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO / "training" / "training_data" / "drl_training"
MODEL_DIR = REPO / "models" / "factor_extraction"
REPORT_DIR = REPO / "training" / "factor_extraction" / "reports"

# Default architecture
DEFAULT_LATENT_DIM = 5
DEFAULT_HIDDEN_DIMS = (64, 16)
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 256
DEFAULT_VAL_FRACTION = 0.20
DEFAULT_PATIENCE = 10
DEFAULT_FORWARD_HORIZON_BARS = 6   # 24h at 4H bars
DEFAULT_PROMOTE_IC = 0.05


def _try_import_torch():
    try:
        import torch  # noqa
        return True
    except ImportError as e:
        logger.warning(
            f"[FACTOR_EXTRACTION] torch unavailable: {type(e).__name__}: {e} "
            f"— training entry-points disabled but module shape ships"
        )
        return False


# ---------------------------------------------------------------------------
# Architecture (lazy torch import — module loads even without torch installed)
# ---------------------------------------------------------------------------

def build_autoencoder(input_dim: int = 122,
                      latent_dim: int = DEFAULT_LATENT_DIM,
                      hidden_dims: Tuple[int, ...] = DEFAULT_HIDDEN_DIMS):
    """Returns (encoder, decoder) torch modules, or (None, None) if torch absent."""
    if not _try_import_torch():
        return None, None
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, latent_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers: List[nn.Module] = []
            prev = latent_dim
            for h in reversed(hidden_dims):
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, input_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, z):
            return self.net(z)

    return Encoder(), Decoder()


# ---------------------------------------------------------------------------
# Training pipeline (operator-runnable; not auto-invoked)
# ---------------------------------------------------------------------------

def load_training_data(asset: str) -> Optional[Tuple[Any, Any]]:
    """Load (features, forward_24h_returns) DataFrames from the asset's parquet.

    Returns None if torch/pandas unavailable or parquet missing.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("[FACTOR_EXTRACTION] pandas unavailable")
        return None

    candidates = [
        PARQUET_DIR / f"{asset}_4H_full.parquet",
        PARQUET_DIR / f"{asset}_4h_full.parquet",
    ]
    parquet_path = None
    for c in candidates:
        if c.exists():
            parquet_path = c
            break
    if parquet_path is None:
        logger.warning(f"[FACTOR_EXTRACTION] no parquet for {asset} in {PARQUET_DIR}")
        return None

    df = pd.read_parquet(parquet_path)
    # Drop non-feature columns
    drop_cols = ["timestamp", "open", "high", "low", "close", "volume",
                 "regime", "regime_confidence"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    features = df[feature_cols].copy()

    # Forward 24h return target = (close[t+6] - close[t]) / close[t]
    if "close" not in df.columns:
        logger.warning(f"[FACTOR_EXTRACTION] {asset} parquet missing close column")
        return None
    horizon = DEFAULT_FORWARD_HORIZON_BARS
    forward_returns = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    # Drop trailing NaNs from forward shift
    valid = ~forward_returns.isna()
    return features[valid].reset_index(drop=True), forward_returns[valid].reset_index(drop=True)


def compute_per_latent_ic(
    encoder: Any,
    features: Any,
    forward_returns: Any,
) -> List[float]:
    """For each latent dim, compute Spearman IC against forward_returns.

    Returns list of IC values (length = latent_dim). Empty list if torch absent.
    """
    if not _try_import_torch():
        return []
    import torch
    import numpy as np

    encoder.eval()
    with torch.no_grad():
        x = torch.tensor(features.values, dtype=torch.float32)
        z = encoder(x).cpu().numpy()
    # z: (n, latent_dim)
    ic_list: List[float] = []
    target = forward_returns.values
    for k in range(z.shape[1]):
        # Spearman = Pearson on ranks
        rx = np.argsort(np.argsort(z[:, k])).astype(np.float64)
        ry = np.argsort(np.argsort(target)).astype(np.float64)
        if rx.std() <= 1e-9 or ry.std() <= 1e-9:
            ic_list.append(0.0)
            continue
        ic = float(np.corrcoef(rx, ry)[0, 1])
        ic_list.append(ic)
    return ic_list


def train_and_validate(
    asset: str,
    latent_dim: int = DEFAULT_LATENT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    patience: int = DEFAULT_PATIENCE,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full training + validation pipeline for one asset.

    Returns the training report dict (also written to disk).
    """
    if not _try_import_torch():
        return {"asset": asset, "error": "torch_unavailable",
                "promoted_dims": [], "ic_per_dim": []}

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    data = load_training_data(asset)
    if data is None:
        return {"asset": asset, "error": "data_missing",
                "promoted_dims": [], "ic_per_dim": []}
    features, forward_returns = data

    # Train/val split (chronological — no shuffle)
    n = len(features)
    n_train = int(n * (1.0 - val_fraction))
    train_x = features.iloc[:n_train]
    val_x = features.iloc[n_train:]
    val_fwd = forward_returns.iloc[n_train:]

    # Robust per-feature scaling on train
    train_arr = train_x.values.astype(np.float32)
    val_arr = val_x.values.astype(np.float32)
    median = np.median(train_arr, axis=0)
    iqr = np.percentile(train_arr, 75, axis=0) - np.percentile(train_arr, 25, axis=0)
    iqr[iqr < 1e-6] = 1.0  # protect against constant features
    train_norm = (train_arr - median) / iqr
    val_norm = (val_arr - median) / iqr

    input_dim = train_norm.shape[1]
    encoder, decoder = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    optim_ = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    no_improve = 0
    train_t = torch.tensor(train_norm, dtype=torch.float32)
    val_t = torch.tensor(val_norm, dtype=torch.float32)

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        # Mini-batch SGD
        perm = torch.randperm(train_t.shape[0])
        train_loss_sum = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            x = train_t[idx]
            optim_.zero_grad()
            x_recon = decoder(encoder(x))
            loss = loss_fn(x_recon, x)
            loss.backward()
            optim_.step()
            train_loss_sum += loss.item() * len(idx)
        train_loss = train_loss_sum / len(perm)

        # Val
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_recon = decoder(encoder(val_t))
            val_loss = loss_fn(val_recon, val_t).item()
        encoder.train()
        decoder.train()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve = 0
            # Save best
            output_dir = output_dir or (MODEL_DIR / asset)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "median": median.tolist(),
                "iqr": iqr.tolist(),
                "input_dim": input_dim,
                "latent_dim": latent_dim,
            }, output_dir / "factor_autoencoder.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[FACTOR_EXTRACTION] early stop at epoch {epoch}")
                break

    # Per-latent IC on val
    ic_per_dim = compute_per_latent_ic(encoder, val_x, val_fwd)
    promoted_dims = [k for k, ic in enumerate(ic_per_dim) if abs(ic) >= DEFAULT_PROMOTE_IC]

    report = {
        "asset": asset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "epochs_run": len(history),
        "best_val_loss": best_val_loss,
        "ic_per_dim": ic_per_dim,
        "promoted_dims": promoted_dims,
        "history_tail": history[-10:],
        "model_path": str((output_dir or (MODEL_DIR / asset)) / "factor_autoencoder.pt"),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"factor_report_{asset}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(f"[FACTOR_EXTRACTION] {asset} report saved to {report_path}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ML factor extraction (Phase 6).")
    p.add_argument("--asset", default="BTC", help="BTC / ETH / SOL")
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = p.parse_args(argv)

    report = train_and_validate(
        asset=args.asset,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())

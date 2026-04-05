#!/usr/bin/env python3
"""
Train a profitability-first supervised sequence alpha model for ModelAlphaAgent.

Uses the same 122 scaled feature space as RuntimeObsBuilder and learns
LONG / FLAT / SHORT classes from forward net edge instead of imitation actions.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from training.model_alpha.sequence_alpha_model import SequenceAlphaConfig, SequenceAlphaNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SequenceAlphaTrain")

BEST_FOLDS = {"BTC": "fold_3", "ETH": "fold_1", "SOL": "fold_3"}


class SequenceAlphaDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, values: np.ndarray):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.values = values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )


def load_manifest() -> Dict[str, List[str]]:
    with open(PROJECT_ROOT / "configs" / "feature_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_runtime_scaler(asset: str):
    import joblib

    scaler_path = (
        PROJECT_ROOT
        / "models"
        / "retrained"
        / asset
        / BEST_FOLDS[asset]
        / "logs"
        / "LSTM_FILM_A"
        / "feature_scaler.pkl"
    )
    meta = joblib.load(scaler_path)
    if not isinstance(meta, dict):
        raise ValueError(f"Unexpected scaler format at {scaler_path}")
    return scaler_path, meta


def apply_runtime_scaling(df: pd.DataFrame, feature_cols: List[str], scaler_meta: Dict[str, object]) -> pd.DataFrame:
    scaler = scaler_meta["scaler"]
    scale_cols = [c for c in scaler_meta.get("scale_cols", []) if c in feature_cols]
    scaled = df.copy()
    if scale_cols:
        scaled_vals = scaler.transform(scaled[scale_cols].astype(np.float32))
        scaled.loc[:, scale_cols] = np.clip(scaled_vals, -10.0, 10.0)
    scaled = scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return scaled


def make_targets(close: pd.Series, horizon_bars: int, fee_bps: float, neutral_bps: float, target_scale_bps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    future_close = close.shift(-horizon_bars)
    fwd_bps = ((future_close / close) - 1.0) * 10000.0
    long_edge = fwd_bps - fee_bps
    short_edge = (-fwd_bps) - fee_bps

    labels = np.full(len(close), 1, dtype=np.int64)  # 0=short,1=flat,2=long
    signed_value = np.zeros(len(close), dtype=np.float32)
    realized_edge = np.zeros(len(close), dtype=np.float32)

    long_mask = (long_edge > short_edge) & (long_edge >= neutral_bps)
    short_mask = (short_edge > long_edge) & (short_edge >= neutral_bps)

    labels[long_mask] = 2
    labels[short_mask] = 0
    realized_edge[long_mask] = long_edge[long_mask]
    realized_edge[short_mask] = short_edge[short_mask]
    signed_value[long_mask] = np.tanh(long_edge[long_mask] / target_scale_bps)
    signed_value[short_mask] = -np.tanh(short_edge[short_mask] / target_scale_bps)

    valid_mask = np.isfinite(fwd_bps.to_numpy())
    return labels, signed_value, valid_mask


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: SequenceAlphaConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, signed_value, valid_mask = make_targets(
        df["close"],
        horizon_bars=cfg.horizon_bars,
        fee_bps=cfg.fee_bps,
        neutral_bps=cfg.neutral_bps,
        target_scale_bps=cfg.target_scale_bps,
    )

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    X, y, v = [], [], []
    seq_len = cfg.seq_len
    for idx in range(seq_len - 1, len(df)):
        if not valid_mask[idx]:
            continue
        start = idx - seq_len + 1
        X.append(feats[start: idx + 1])
        y.append(labels[idx])
        v.append(signed_value[idx])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(v, dtype=np.float32)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: str, cfg: SequenceAlphaConfig) -> Dict[str, float]:
    model.eval()
    losses = []
    all_pred = []
    all_true = []
    all_value = []
    with torch.no_grad():
        for X, y, value_target in loader:
            X = X.to(device)
            y = y.to(device)
            value_target = value_target.to(device)
            out = model(X)
            ce = F.cross_entropy(out["logits"], y)
            mse = F.mse_loss(torch.tanh(out["value"]), value_target)
            losses.append(float((ce + 0.25 * mse).item()))
            all_pred.append(torch.argmax(out["logits"], dim=1).cpu().numpy())
            all_true.append(y.cpu().numpy())
            all_value.append(torch.tanh(out["value"]).cpu().numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    value_pred = np.concatenate(all_value)

    acc = float((y_pred == y_true).mean())
    nonflat_mask = y_pred != 1
    trade_rate = float(nonflat_mask.mean())
    direction_acc = float((y_pred[nonflat_mask] == y_true[nonflat_mask]).mean()) if nonflat_mask.any() else 0.0

    # Approximate profitability score from predicted signed edge
    predicted_edge_bps = np.tanh(value_pred) * cfg.target_scale_bps
    profitable = predicted_edge_bps[nonflat_mask] > 0 if nonflat_mask.any() else np.array([], dtype=bool)
    hit_rate = float(profitable.mean()) if profitable.size else 0.0
    mean_pred_edge = float(predicted_edge_bps[nonflat_mask].mean()) if nonflat_mask.any() else 0.0
    soft_edge_rate = float((np.abs(value_pred[nonflat_mask]) >= 0.03).mean()) if nonflat_mask.any() else 0.0
    hard_edge_rate = float((np.abs(value_pred[nonflat_mask]) >= 0.12).mean()) if nonflat_mask.any() else 0.0
    mean_abs_score = float(np.abs(value_pred[nonflat_mask]).mean()) if nonflat_mask.any() else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "direction_acc": direction_acc,
        "trade_rate": trade_rate,
        "hit_rate": hit_rate,
        "mean_pred_edge_bps": mean_pred_edge,
        "soft_edge_rate": soft_edge_rate,
        "hard_edge_rate": hard_edge_rate,
        "mean_abs_score": mean_abs_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Train supervised sequence alpha model")
    parser.add_argument("--asset", required=True, choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--arch", default="gru", choices=["gru", "lstm"])
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=7.0)
    parser.add_argument("--neutral-bps", type=float, default=6.0)
    parser.add_argument("--target-scale-bps", type=float, default=64.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default=str(PROJECT_ROOT / "models" / "model_alpha"))
    args = parser.parse_args()

    manifest = load_manifest()
    feature_cols = manifest["all_features"]

    data_path = PROJECT_ROOT / "training" / "training_data" / "drl_training" / f"{args.asset}_4H_full.parquet"
    df = pd.read_parquet(data_path)
    scaler_path, scaler_meta = load_runtime_scaler(args.asset)
    df = apply_runtime_scaling(df, feature_cols, scaler_meta)

    cfg = SequenceAlphaConfig(
        input_dim=len(feature_cols),
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        arch=args.arch,
        target_scale_bps=args.target_scale_bps,
        horizon_bars=args.horizon_bars,
        fee_bps=args.fee_bps,
        neutral_bps=args.neutral_bps,
    )

    n = len(df)
    val_size = int(n * 0.15)
    gap = 42
    train_end = n - val_size - gap
    val_start = train_end + gap

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:].copy()

    X_train, y_train, v_train = build_sequences(train_df, feature_cols, cfg)
    X_val, y_val, v_val = build_sequences(val_df, feature_cols, cfg)

    train_ds = SequenceAlphaDataset(X_train, y_train, v_train)
    val_ds = SequenceAlphaDataset(X_val, y_val, v_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = args.device
    model = SequenceAlphaNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = compute_class_weights(y_train).to(device)

    out_dir = Path(args.save_dir) / args.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "sequence_alpha_v1_best.pt"
    final_path = out_dir / "sequence_alpha_v1_final.pt"
    metrics_path = out_dir / "sequence_alpha_metrics.json"

    best_score = -1e9
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for X, y, value_target in train_loader:
            X = X.to(device)
            y = y.to(device)
            value_target = value_target.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(X)
            ce = F.cross_entropy(out["logits"], y, weight=class_weights)
            mse = F.mse_loss(torch.tanh(out["value"]), value_target)
            loss = ce + args.value_loss_weight * mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device, cfg)
        epoch_info = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            **val_metrics,
        }
        history.append(epoch_info)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f acc=%.3f dir_acc=%.3f trade_rate=%.3f mean_pred_edge=%.2fbps soft=%.3f hard=%.3f",
            epoch,
            epoch_info["train_loss"],
            epoch_info["loss"],
            epoch_info["acc"],
            epoch_info["direction_acc"],
            epoch_info["trade_rate"],
            epoch_info["mean_pred_edge_bps"],
            epoch_info["soft_edge_rate"],
            epoch_info["hard_edge_rate"],
        )

        profit_score = val_metrics["mean_pred_edge_bps"] * max(val_metrics["trade_rate"], 0.05)
        if profit_score > best_score:
            best_score = profit_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_dict(),
                    "feature_cols": feature_cols,
                    "runtime_scaler_path": str(scaler_path),
                    "best_profit_score": best_score,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "feature_cols": feature_cols,
            "runtime_scaler_path": str(scaler_path),
            "best_profit_score": best_score,
            "history": history,
        },
        final_path,
    )

    summary = {
        "asset": args.asset,
        "arch": args.arch,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "value_loss_weight": args.value_loss_weight,
        "duration_sec": round(time.time() - start, 2),
        "best_profit_score": best_score,
        "best_model": str(best_path),
        "final_model": str(final_path),
        "runtime_scaler_path": str(scaler_path),
        "history": history,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("done asset=%s best_profit_score=%.3f saved=%s", args.asset, best_score, best_path)


if __name__ == "__main__":
    main()

"""
Causal RegimeSmoother - single source of truth.

Used by:
  - train_drl_full.py (root training script)
  - scripts/rebuild_pipeline.py
  - scripts/prepare_drl_training_data.py
  - scripts/gmm_flip_analysis.py
  - training/train_tqc.py

IMPORTANT: This must produce identical output in training and runtime.
Do NOT modify without updating both pipelines.
"""

import pandas as pd


class RegimeSmoother:
    """Hold current regime until new one persists for N consecutive bars."""

    def __init__(self, min_persistence: int = 2):
        self.min_persistence = min_persistence

    def smooth_column(self, df: pd.DataFrame, col: str = "regime") -> pd.DataFrame:
        if col not in df.columns:
            return df
        labels = df[col].tolist()
        smoothed = self._smooth(labels)
        df = df.copy()
        df[col] = smoothed
        return df

    def _smooth(self, labels: list) -> list:
        if not labels:
            return labels
        out = []
        current = labels[0]
        pending = None
        count = 0
        for lab in labels:
            if lab == current:
                pending, count = None, 0
                out.append(current)
            elif lab == pending:
                count += 1
                if count >= self.min_persistence:
                    current = pending
                    pending, count = None, 0
                out.append(current)
            else:
                pending, count = lab, 1
                out.append(current)
        return out

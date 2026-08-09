"""[GP0] One shared split module + the window-usage ledger.

Every experiment must take its splits from here, not re-implement them —
split logic re-implemented per script is how leakage and geometry drift
happen (P164/P200 family). And every evaluation window an experiment READS
must be recorded in the usage ledger: unseen data is the scarcest resource
this project has (P243 — every window has been seen in aggregate), and a
window's statistical value decays with every look. The ledger makes that
spend visible instead of silent.

Canonical geometry (matches train_drl_full._get_fold_splits and the P244
regime lab):
    DESIGN_ERA      = [3000, 9100)   all selection + tuning happens here
    VALIDATION_ERA  = [9100, n)      touched once per assembled system
    fold geometry   = val_ratio 0.15, gap 42, 3 sequential vals from the end
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO / "training" / "reports" / "window_usage.json"

DESIGN_ERA = (3000, 9100)
VALIDATION_ERA_START = 9100
EMBARGO = 42
HORIZON = 4  # 16h label on 4H bars


def drl_fold_splits(n, n_folds=3, val_ratio=0.15, gap=42):
    """The trainer's geometry: sequential val windows walking back from the
    end; fold_1 = latest data. Returns {fold: (train_end, val_start, val_end)}."""
    val_size = int(n * val_ratio)
    out = {}
    for k in range(1, n_folds + 1):
        val_end = n - (k - 1) * val_size
        val_start = val_end - val_size
        train_end = val_start - gap
        if train_end < 3000:
            continue
        out[f"fold_{k}"] = (train_end, val_start, val_end)
    return out


def purged_folds(start, end, n_splits=5, embargo=EMBARGO, horizon=HORIZON):
    """Purged K-fold confined to [start, end): train indices exclude the val
    block plus horizon+embargo on both sides. Lopez de Prado's CV, the only
    K-fold legal on overlapping labels."""
    span = (end - start) // n_splits
    for k in range(n_splits):
        v0 = start + k * span
        v1 = v0 + span if k < n_splits - 1 else end
        tr = np.concatenate([
            np.arange(start, max(start, v0 - horizon - embargo)),
            np.arange(min(end, v1 + horizon + embargo), end),
        ])
        yield tr, np.arange(v0, v1)


# ---------------------------------------------------------------- ledger
def _load_ledger():
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"records": []}
    return {"records": []}


def record_window_usage(experiment, asset, start, end, purpose):
    """Append one usage record. purpose in {'design','validation','probe'}.
    Returns the number of PRIOR validation reads of an overlapping window —
    the caller must surface a nonzero count, because a validation window
    that has been read k times supports selection over ~k hypotheses and
    its p-values must be discounted accordingly."""
    ledger = _load_ledger()
    prior = sum(
        1 for r in ledger["records"]
        if r["asset"] == asset and r["purpose"] == "validation"
        and purpose == "validation"
        and not (end <= r["start"] or start >= r["end"])
        and r["experiment"] != experiment
    )
    ledger["records"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment, "asset": asset,
        "start": int(start), "end": int(end), "purpose": purpose,
    })
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    return prior


def validation_spend(asset):
    """How many distinct experiments have read validation windows per asset."""
    ledger = _load_ledger()
    return sorted({r["experiment"] for r in ledger["records"]
                   if r["asset"] == asset and r["purpose"] == "validation"})

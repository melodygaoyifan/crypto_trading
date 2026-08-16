#!/usr/bin/env python3
"""[P270] The volskip validation read — ONE ledgered spend, taken BEFORE any
forward-ledger wiring, per the P259b ordering rule ("spend the last unread
historical window BEFORE wiring a forward ledger, not after").

Candidate: high-vol entry-skip at causal expanding q=0.80 on the ETH and SOL
books (the two both-era winners from training/entry_filter_lab.py; BTC's
base stood and takes no read). The window [9100, n) is multiply-read for
OTHER candidates (7 prior spends) — the P260 discount applies to any
comparison against the base book, but the filtered ENTRIES' own sign does
not depend on that comparison (the P259b caveat structure).

This script records the spend in training/reports/window_usage.json and
writes its numbers to training/reports/volskip_validation_read_p270.json.
It deliberately reimplements pnl_after_cost WITHOUT the design-era assert
(the assert is correct for the lab; this is the one sanctioned read).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import book_targets, DE  # noqa: E402
from training.entry_filter_lab import (  # noqa: E402
    apply_entry_filter, _roll_vol)
from training.regime_model_lab import _ctx  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402
from training.splits import record_window_usage  # noqa: E402

Q = 0.80  # the lab-selected quantile — fixed BEFORE this read


def pnl_window(close, pos, cost_rt_bps, lo, hi):
    """Same arithmetic as mechanism_lab.pnl_after_cost, minus the design-era
    assert (this is the sanctioned validation read)."""
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = float(np.nansum(pos[seg] * r1[seg]))
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    cost = float(np.nansum(dpos) * (cost_rt_bps / 2.0) / 1e4)
    return {"gross": round(gross, 4), "cost": round(cost, 4),
            "net": round(gross - cost, 4)}


def main() -> int:
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "candidate": f"volskip_q{Q}_entry_filter",
           "note": "single sanctioned validation read (P259b ordering rule);"
                   " window multiply-read for other candidates — comparisons"
                   " vs the base book carry the P260 discount",
           "assets": {}}
    for a in ("ETH", "SOL"):
        c = _ctx(a)
        close = c["close"]
        n = len(close)
        vol = _roll_vol(close)
        thr = np.full(n, np.nan)
        thr[400:] = (pd.Series(vol).expanding(min_periods=300)
                     .quantile(Q).shift(1).values[400:])
        allow = ~np.isnan(vol) & ~np.isnan(thr) & (vol <= thr)
        raw = book_targets(a, c["lab"], c["fz"])
        filt = apply_entry_filter(raw, allow)
        lo, hi = DE, n
        base = pnl_window(close, raw, COST_BPS[a], lo, hi)
        filted = pnl_window(close, filt, COST_BPS[a], lo, hi)
        bh = pnl_window(close, np.ones(n), COST_BPS[a], lo, hi)
        record_window_usage(f"volskip_p270", a, lo, hi, "validation")
        out["assets"][a] = {
            "window": [lo, hi], "base_book": base, "volskip": filted,
            "buy_and_hold": bh,
            "increment_net": round(filted["net"] - base["net"], 4),
            # the current expanding threshold — what a live export would pin
            "expanding_thr_at_end": (None if np.isnan(thr[-1])
                                     else round(float(thr[-1]), 6)),
        }
        print(f"[volskip-VAL] {a}: base={base['net']:+.4f} "
              f"volskip={filted['net']:+.4f} "
              f"increment={out['assets'][a]['increment_net']:+.4f} "
              f"(B&H {bh['net']:+.4f})")
    rp = REPO / "training" / "reports" / "volskip_validation_read_p270.json"
    rp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"report: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

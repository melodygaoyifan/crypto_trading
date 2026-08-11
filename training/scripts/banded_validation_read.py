"""[P260] The PRESERVED reproduction of the P259b withdrawal read.

The fresh-mind review's finding #2: the numbers that withdrew the banded
overlay (BTC increment -0.212, ETH -0.618 on the validation era) existed
only in CLAUDE.md prose — the read was ad-hoc code, unpreserved, so nobody
could verify the batch's most consequential same-day decision. This script
IS that read, byte-reproducible, writing its artifact to
training/reports/banded_validation_read_p259b.json.

It does NOT re-record in the window-usage ledger: the spend was ledgered
once as `banded_overlay_p259` on 2026-08-10 and this reproduces that same
read (re-running does not re-spend an already-spent look; re-LEDGERING it
would double-count the spend statistics).

Method (identical to the original): walk-forward ridge THROUGH the
validation era (fits only ever see the past; refit cadence/gap/alpha from
banded_forecast_lab), the lab-winner band params, overlay = book where the
book is non-flat else banded, after-cost PnL with the mechanism_lab cost
convention.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import Ridge  # noqa: E402

from training.regime_model_lab import _ctx, DESIGN  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402
from training.mechanism_lab import book_targets  # noqa: E402
from training.banded_forecast_lab import (  # noqa: E402
    close_features, banded_positions, REFIT, GAP, RIDGE_ALPHA, H,
)

DS, DE = DESIGN
WINNERS = {"BTC": (2.0, 0.5, False), "ETH": (1.0, 0.25, False)}
OUT = REPO / "training" / "reports" / "banded_validation_read_p259b.json"


def _pnl(close, pos, cost_rt, lo, hi):
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = float(np.nansum(pos[seg] * r1[seg]))
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    cost = float(np.nansum(dpos) * (cost_rt / 2.0) / 1e4)
    return gross - cost, float(np.nansum(dpos))


def main() -> int:
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "reproduces": "the P259b withdrawal read (ledgered "
                            "banded_overlay_p259, 2026-08-10; this script "
                            "does NOT re-ledger the spend)"}
    for a, (te, tx, g) in WINNERS.items():
        c = _ctx(a)
        close, lab, fz = c["close"], c["lab"], c["fz"]
        n = len(close)
        X, _ = close_features(close, fz)
        y = np.full(n, np.nan)
        y[:-H] = close[H:] / close[:-H] - 1.0
        pred = np.full(n, np.nan)
        for t0 in range(DS - 500, n, REFIT):
            tr_end = t0 - GAP
            ok = ~np.isnan(X[:tr_end]).any(1) & ~np.isnan(y[:tr_end])
            if ok.sum() < 800:
                continue
            mu, sd = X[:tr_end][ok].mean(0), X[:tr_end][ok].std(0) + 1e-12
            m = Ridge(alpha=RIDGE_ALPHA).fit((X[:tr_end][ok] - mu) / sd,
                                             y[:tr_end][ok])
            hi = min(t0 + REFIT, n)
            seg = ~np.isnan(X[t0:hi]).any(1)
            p = np.full(hi - t0, np.nan)
            p[seg] = m.predict((X[t0:hi][seg] - mu) / sd)
            pred[t0:hi] = p
        pos = banded_positions(pred, lab, te, tx, g)
        book = book_targets(a, lab, fz)
        ov = np.where(book != 0.0, book, pos)
        VS, VE = 9100, n
        ov_net, ov_to = _pnl(close, ov, COST_BPS[a], VS, VE)
        bk_net, bk_to = _pnl(close, book, COST_BPS[a], VS, VE)
        report[a] = {
            "window": [VS, VE], "band": {"t_enter": te, "t_exit": tx},
            "overlay_net": round(ov_net, 4), "overlay_turnover": ov_to,
            "book_net": round(bk_net, 4), "book_turnover": bk_to,
            "increment": round(ov_net - bk_net, 4),
            "buy_and_hold": round(close[VE - 1] / close[VS] - 1.0, 4),
        }
        print(f"{a}: overlay={ov_net:+.4f} book={bk_net:+.4f} "
              f"increment={ov_net - bk_net:+.4f}")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

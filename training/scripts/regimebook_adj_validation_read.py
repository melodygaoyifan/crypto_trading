"""[P379] The one genuinely-unrun historical test: regimebook_adj's VALIDATION
era (out-of-sample). P256 selected the adj params IN-DESIGN only and hard-asserts
no validation read; its forward ledger is 12 days old. This spends the lockbox
once (like P259b for banded) to answer off->on from HISTORY instead of waiting.

TEST: run the DESIGN-SELECTED adj params (BTC ke1/kf2/mh0, ETH ke3/kf1/mh0,
SOL ke1/kf1/mh6) on the validation era [DE, n) and compare adj vs the raw book
on net-after-cost + turnover. NO re-selection on validation (that would be
cheating — use the design winners, test them out-of-sample).

VERDICT, pre-committed before the run: the adj mechanism GENERALIZES (arm-able)
iff, in the validation era, the design-selected adj config beats the raw book on
NET after honest cost on >= 2 of 3 assets AND reduces turnover on all 3 (the
churn-reduction benefit — the part that is cost-positive regardless of alpha —
must hold out-of-sample). Otherwise the per-asset params were overfit (P259b).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

from training.mechanism_lab import book_targets, apply_adjust, DS, DE  # noqa: E402
from training.regime_model_lab import _ctx  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402

# design-selected winners (P256 mechanism_lab_p256.json), NOT re-fit here
ADJ = {"BTC": (1, 2, 0), "ETH": (3, 1, 0), "SOL": (1, 1, 6)}


def net_after_cost(close, pos, cost_rt_bps, lo, hi):
    """Same arithmetic as mechanism_lab.pnl_after_cost, but scored on the
    VALIDATION window [lo, hi) — the design-only assert is deliberately not
    applied here because this IS the sanctioned one-shot validation read."""
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = float(np.nansum(pos[seg] * r1[seg]))
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    cost = float(np.nansum(dpos) * (cost_rt_bps / 2.0) / 1e4)
    turn = float(np.nansum(dpos))
    return {"gross": round(gross, 4), "cost": round(cost, 4),
            "net": round(gross - cost, 4), "turnover": round(turn, 1)}


def main():
    res = {"validation_era_start": DE, "params": ADJ, "assets": {}}
    beats = 0
    turn_down = 0
    print("=" * 88)
    print("  regimebook_adj VALIDATION-ERA read [%d, n) — out-of-sample (one-shot lockbox spend)" % DE)
    print("  BAR: adj beats raw net on >=2/3 assets AND cuts turnover on all 3")
    print("=" * 88)
    for a in ("BTC", "ETH", "SOL"):
        c = _ctx(a)
        n = len(c["close"])
        raw = book_targets(a, c["lab"], c["fz"])
        ke, kf, mh = ADJ[a]
        adj = apply_adjust(raw, ke, kf, mh)
        base = net_after_cost(c["close"], raw, COST_BPS[a], DE, n)
        av = net_after_cost(c["close"], adj, COST_BPS[a], DE, n)
        b = av["net"] > base["net"]
        td = av["turnover"] < base["turnover"]
        beats += 1 if b else 0
        turn_down += 1 if td else 0
        res["assets"][a] = {"raw": base, "adj": av, "beats_net": b, "cuts_turnover": td}
        print("\n%s (ke%d/kf%d/mh%d):" % (a, ke, kf, mh))
        print("  raw : net %+8.4f  turnover %6.1f" % (base["net"], base["turnover"]))
        print("  adj : net %+8.4f  turnover %6.1f   [net %s, turnover %s]"
              % (av["net"], av["turnover"], "BEATS" if b else "loses", "cut" if td else "up"))
    verdict = "GENERALIZES — arm-able" if (beats >= 2 and turn_down == 3) else "OVERFIT — do not arm"
    res["verdict"] = verdict
    res["net_beats"] = beats
    res["turnover_cut"] = turn_down
    (REPO / "training" / "reports" / "regimebook_adj_validation_p379.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    print("  VERDICT: %s  (net beats %d/3, turnover cut %d/3)" % (verdict, beats, turn_down))
    print("  report -> training/reports/regimebook_adj_validation_p379.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

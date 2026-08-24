"""[P396] Gated auto-probe of the accumulated FREE new-data (P395), so the payoff
of forward accumulation gets READ ON SCHEDULE rather than depending on memory
(the P230/P333 "a read nobody remembers becomes never" failure mode).

WHAT IT GATES ON: the options put/call archive (Deribit, BTC/ETH) has NO free
history (CoinGlass 404s, Deribit gives only live snapshots — verified P395/P396),
so it can only be validated after forward accumulation. This script counts the
distinct UTC days banked in newdata_snapshots.jsonl and:
  - < MIN_DAYS: prints "accumulating N/MIN" and exits 0 (nothing to do yet);
  - >= MIN_DAYS: runs the P386 hold-aware Rung-0 (walk-forward, deadband, honest
    cost) on the put/call-OI signal vs forward return built from the banked
    underlying_price, and reports EARNS/NOT_EARNED with a pre-committed rule.

PRE-COMMITTED (before any data reaches the gate): signal = put/call-OI z-score;
crowded puts (high P/C) is the contrarian-bullish reading in the options
literature, so sign = +z (high P/C -> long); band 1.0 (chosen now, not swept);
honest CDE cost on flips; OOS = second half. EARNS iff held OOS net > 0 AND >
buy-and-hold on >= 1 of 2 assets. Otherwise the free options slice is dead at
this scale (as positioning already was, P391) and the honest next step is a paid
options-chain (skew/term) purchase decision, not another free probe.

MIN_DAYS default 180 (~6 months) — enough for a ~90-day OOS second half. On a
daily cron this fires automatically the first Monday after the archive is deep
enough. Never raises; a short archive is a clean "not yet", not a failure.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SNAP = Path(os.environ.get(
    "HMATS_SNAPSHOT_DIR",
    str(REPO / "training" / "training_data" / "signal_snapshots"))) / "newdata_snapshots.jsonl"
MIN_DAYS = int(os.environ.get("HMATS_GATED_PROBE_MIN_DAYS", "180"))
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}
BAND = 1.0


def _load():
    """UTC-date-sorted rows with a usable options block."""
    rows = []
    if not SNAP.exists():
        return rows
    for ln in SNAP.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            if r.get("date") and isinstance(r.get("options"), dict):
                rows.append(r)
        except Exception:
            continue
    return sorted(rows, key=lambda r: r["date"])


def _series(rows, asset):
    """(put_call_oi, underlying_price) daily arrays for one asset; NaN where absent."""
    pcr, px = [], []
    for r in rows:
        m = (r.get("options") or {}).get(asset) or {}
        pcr.append(m.get("put_call_ratio_oi"))
        px.append(m.get("underlying_price"))
    f = lambda a: np.array([np.nan if v is None else float(v) for v in a], float)
    return f(pcr), f(px)


def _z(x, w=30):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]
        win = win[np.isfinite(win)]
        if len(win) >= 10 and win.std() > 0:
            out[i] = np.clip((x[i] - win.mean()) / win.std(), -5, 5)
    return out


def _hold_stats(px, sig, per_leg, band):
    n = len(px)
    ret = np.zeros(n)
    ret[1:] = np.where(px[:-1] > 0, px[1:] / np.where(px[:-1] == 0, np.nan, px[:-1]) - 1.0, 0.0)
    ret = np.nan_to_num(ret)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if not np.isfinite(sig[i]):
            pos[i] = cur
            continue
        if sig[i] > band:
            cur = 1.0
        elif sig[i] < -band:
            cur = -1.0
        pos[i] = cur
    dpos = np.abs(np.diff(pos, prepend=0.0))
    pnl = np.zeros(n)
    pnl[:-1] = pos[:-1] * ret[1:]
    pnl = pnl - dpos * (per_leg / 2.0)
    pnl = pnl[np.isfinite(pnl)]
    net = round(float(pnl.sum() * 100), 1) if len(pnl) else 0.0
    return net, int(dpos.sum())


def main() -> int:
    rows = _load()
    ndays = len({r["date"] for r in rows})
    print(f"[P396] gated new-data probe: {ndays}/{MIN_DAYS} days banked ({SNAP})")
    if ndays < MIN_DAYS:
        print(f"  ACCUMULATING — {MIN_DAYS - ndays} more days before the put/call Rung-0 fires. Nothing to do.")
        return 0

    print("  PRE-COMMITTED: signal = put/call-OI z (high P/C -> long, contrarian); "
          "band 1.0; OOS 2nd half; honest cost; EARNS iff held OOS net>0 AND >buy&hold on >=1/2")
    earns = 0
    for a in ("BTC", "ETH"):
        pcr, px = _series(rows, a)
        if np.isfinite(px).sum() < MIN_DAYS * 0.8:
            print(f"  {a}: insufficient price coverage")
            continue
        sig = _z(pcr)
        mid = len(px) // 2
        sig[:mid] = np.nan  # OOS = second half only
        net, tr = _hold_stats(px, sig, COST_RT[a], BAND)
        bh = np.zeros(len(px))
        bh[mid:-1] = np.where(px[mid:-1] > 0, px[mid + 1:] / px[mid:-1] - 1.0, 0.0)
        bh_net = round(float(np.nan_to_num(bh).sum() * 100), 1)
        ok = net > 0 and net > bh_net
        earns += ok
        print(f"  {a}: held OOS net {net:+.1f}% trades {tr} | buy&hold {bh_net:+.1f}% -> {'EARNS' if ok else 'no'}")
    verdict = ("EARNS — free options put/call clears; advance the P200 ladder (features->GMM->retrain->shadow)"
               if earns >= 1 else
               "NOT_EARNED — free options slice dead at this scale; paid options-chain (skew/term) is the only remaining option lead, an operator purchase decision")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

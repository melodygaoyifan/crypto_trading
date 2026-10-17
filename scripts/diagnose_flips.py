"""[P1R-4] FLIP diagnosis from the Phase 0 frames (read-only).

    python -X utf8 scripts/diagnose_flips.py

For each FLIP order (a single order that crosses the position through zero):
the flipped position's hold and its unrealized PnL at the flip (= the gross of
the round trip the flip closed), the new position's realized gross (the round
trip the flip opened), and whether the new position was itself flipped back
within 2 bars (whipsaw). Pre-P423 fills carry no decider, so the signal
source is attributed by ERA from the recorded seat history (CLAUDE.md
P149/P293j/P298/P417): trend seat (+-1 vote) until 2026-08-18, whale seat
2026-08-17..08-27, regimebook from 08-18. Writes reports/phase1r_flip_diagnosis.md.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def era(day: str) -> str:
    if day < "2026-08-17":
        return "trend_seat_pm1"
    if day < "2026-08-18":
        return "whale_seat_pm1"
    if day < "2026-08-27":
        return "regimebook+whale_seats"
    if day < "2026-10-13":
        return "regimebook (+ETF derisk)"
    return "skew_seat(BTC)+regimebook"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", default=os.path.join(REPO, "data", "phase0", "fills_raw.parquet"))
    ap.add_argument("--rts", default=os.path.join(REPO, "data", "phase0", "roundtrips.parquet"))
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "phase1r_flip_diagnosis.md"))
    a = ap.parse_args(argv)
    if not (os.path.exists(a.fills) and os.path.exists(a.rts)):
        print("REFUSING: Phase 0 frames missing"); return 2
    import pandas as pd
    F = pd.read_parquet(a.fills)
    R = pd.read_parquet(a.rts)
    R["entry_ts"] = pd.to_datetime(R.entry_iso).astype("int64") / 1e9
    R["exit_ts"] = pd.to_datetime(R.exit_iso).astype("int64") / 1e9
    fl = F[F.cls == "FLIP"]
    orders = []
    for oid, g in fl.groupby("order_id"):
        g = g.sort_values("timestamp")
        ts = float(g.timestamp.iloc[0]); asset = g.asset.iloc[0]
        closed = R[(R.asset == asset) & ((R.exit_ts - ts).abs() < 600)]
        opened = R[(R.asset == asset) & ((R.entry_ts - ts).abs() < 600)]
        c = closed.iloc[0] if len(closed) else None
        o = opened.iloc[0] if len(opened) else None
        orders.append(dict(order_id=oid, asset=asset, ts=ts, day=g.iso.iloc[0][:10],
                           pos_before=float(g.pos_before.iloc[0]), pos_after=float(g.pos_after.iloc[-1]),
                           fee_usd=float(g.commission.sum()),
                           closed_hold_bars=None if c is None else float(c.hold_bars),
                           closed_gross_bps=None if c is None else float(c.gross_alpha_bps),
                           closed_gross_usd=None if c is None else float(c.gross_pnl_usd),
                           opened_hold_bars=None if o is None else float(o.hold_bars),
                           opened_gross_usd=None if o is None else float(o.gross_pnl_usd),
                           opened_exit_ts=None if o is None else float(o.exit_ts)))
    D = pd.DataFrame(orders).sort_values("ts")
    flip_ts = {(r.asset, round(r.ts)) for r in D.itertuples()}
    D["whipsaw_2bar"] = [
        bool(r.opened_hold_bars is not None and r.opened_hold_bars <= 2.0 and any(
            x[0] == r.asset and abs(x[1] - r.opened_exit_ts) < 600 for x in flip_ts))
        for r in D.itertuples()]
    D["era"] = D.day.map(era)
    D["flip_while"] = D.closed_gross_bps.map(
        lambda v: None if v is None else ("losing" if v < 0 else "winning"))
    L = []; w = L.append
    w("# HMATS Phase 1R — R-4 FLIP diagnosis\n")
    w(f"{len(D)} FLIP orders ({int(fl.shape[0])} fills), {D.day.min()} → {D.day.max()}. "
      f"Fees on FLIP fills ${fl.commission.sum():.2f}.\n")
    w(f"- realized gross of the round trips the FLIPs CLOSED: ${D.closed_gross_usd.sum():.2f}; "
      f"of the round trips they OPENED: ${D.opened_gross_usd.sum():.2f}\n")
    w(f"- flipped while the closed position was losing: {int((D.flip_while == 'losing').sum())}, "
      f"winning: {int((D.flip_while == 'winning').sum())}\n")
    w(f"- median hold of the flipped position: {D.closed_hold_bars.median():.2f} bars; "
      f"of the opened position: {D.opened_hold_bars.median():.2f} bars\n")
    w(f"- whipsaw (opened position flipped back within <=2 bars): {int(D.whipsaw_2bar.sum())}/{len(D)}\n")
    w("\n## By asset\n")
    g = D.groupby("asset").agg(n=("order_id", "size"), fee_usd=("fee_usd", "sum"),
                               closed_gross=("closed_gross_usd", "sum"),
                               opened_gross=("opened_gross_usd", "sum"),
                               losing=("flip_while", lambda s: int((s == "losing").sum())),
                               whipsaw=("whipsaw_2bar", "sum"),
                               med_closed_hold=("closed_hold_bars", "median"))
    w("| asset | n | fee $ | closed gross $ | opened gross $ | flip-while-losing | whipsaw<=2bar | median closed hold |")
    w("|---|---|---|---|---|---|---|---|")
    for k, r in g.iterrows():
        w(f"| {k} | {r.n} | {r.fee_usd:.2f} | {r.closed_gross:.2f} | {r.opened_gross:.2f} | "
          f"{r.losing} | {int(r.whipsaw)} | {r.med_closed_hold:.2f} |")
    w("\n## By era (signal source; pre-P423 fills have no decider stamp)\n")
    for k, gg in D.groupby("era"):
        w(f"- {k}: {len(gg)} FLIP orders, {gg.day.min()} → {gg.day.max()}")
    w("\n## Per order\n")
    w("| day | asset | pos | fee $ | closed hold | closed gross bps | opened hold | opened gross $ | whipsaw |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in D.itertuples():
        def f(v, p=2):
            return "—" if v is None or v != v else f"{v:.{p}f}"
        w(f"| {r.day} | {r.asset} | {r.pos_before:+g}->{r.pos_after:+g} | {r.fee_usd:.2f} | "
          f"{f(r.closed_hold_bars)} | {f(r.closed_gross_bps, 1)} | {f(r.opened_hold_bars)} | "
          f"{f(r.opened_gross_usd)} | {'Y' if r.whipsaw_2bar else ''} |")
    md = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md[:3500])
    D.to_csv(os.path.join(os.path.dirname(a.out), "phase1r_flip_orders.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

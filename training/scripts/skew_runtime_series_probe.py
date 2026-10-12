"""[P421] First read of the P407g skew blend rule on the RUNTIME series.

THE QUESTION (the P420 open residue, brought forward): the seat's 680/1080
bps/RT calibration was measured on the Laevitas DASHBOARD-backend series; the
seat READS the apiv2 by-tenor series, and over the 67d overlap the two disagree
on ~1 decision-day in 5 (P420). This probe runs the IDENTICAL rule (the
producer's own _zseries/_positions_from_z — imported, never reimplemented,
P172) on ~14 months of the runtime series (MCP-backfilled 2025-08-17 ->
2026-06-17 + the engine's own bank 2026-06-18 -> 2026-10-11).

VERDICT RULE, PRE-COMMITTED BEFORE THE FIRST RUN (P260/P332):
  * KILL-CLASS EVIDENCE: gross bps/RT <= the CDE round-trip cost (BTC 19.7,
    ETH 29.0) on BOTH assets over the full window -> the skew family earns
    nothing on the series the seat actually reads; no Deribit replica is ever
    built and the parked seat stays parked.
  * ALIVE-CLASS: gross/RT >= 2x cost (the P167 margin) on an asset -> the
    family keeps candidate status for that asset; a replica decision would
    still need its own P-entry + forward evidence (P141).
  * Anything between is NOISE: 14 months at ~15 flips/yr is ~15-20 round
    trips -- this window can KILL or keep-alive, it cannot CERTIFY (P348/P406).
Spot = UTC daily close from the fresh {A}_4H_ohlcv.parquet (last 4H bar of the
calendar day); return convention matches the producer (pos[i] * ret[i->i+1]).

    python -X utf8 training/scripts/skew_runtime_series_probe.py \
        --hist <dir with skew_hist_{A}.jsonl> --bank <dir with laevitas_apiv2_skew_{A}.jsonl>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.skew_seat_calibration import _zseries, _positions_from_z  # P172: one rule implementation

CDE_RT_COST = {"BTC": 19.7, "ETH": 29.0}


def _load_jsonl(path: str) -> dict:
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("tenor") == 30 and r.get("skew_25d") is not None and r.get("skew_10d") is not None:
                    rows[r["date"]] = r
    return rows


def _daily_close(parquet_path: str) -> dict:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df = df.sort_values(tcol)
    df["_d"] = df[tcol].dt.strftime("%Y-%m-%d")
    g = df.groupby("_d")["close"].last()
    return {d: float(v) for d, v in g.items()}


def probe(asset: str, hist_dir: str, bank_dir: str, ohlcv_dir: str) -> None:
    hist = _load_jsonl(os.path.join(hist_dir, f"skew_hist_{asset}.jsonl"))
    bank = _load_jsonl(os.path.join(bank_dir, f"laevitas_apiv2_skew_{asset}.jsonl"))
    merged = dict(hist)
    merged.update(bank)  # bank (engine-written + gap backfill) wins on collision
    spot = _daily_close(os.path.join(ohlcv_dir, f"{asset}_4H_ohlcv.parquet"))
    days = sorted(set(merged) & set(spot))
    n = len(days)
    if n < 100:
        print(f"{asset}: REFUSED — only {n} overlapping days (<100); no verdict from missing data (P199)")
        return
    sp = [spot[d] for d in days]
    ret = [0.0] * n
    for i in range(n - 1):
        ret[i] = sp[i + 1] / sp[i] - 1.0 if sp[i] else 0.0
    z25 = _zseries([float(merged[d]["skew_25d"]) for d in days])
    z10 = _zseries([float(merged[d]["skew_10d"]) for d in days])
    pos = _positions_from_z([(z25[i] + z10[i]) / 2.0 for i in range(n)])

    def stats(idx):
        seq = [0.0] + [pos[i] for i in idx]
        flips = sum(1 for k in range(1, len(seq)) if abs(seq[k] - seq[k - 1]) > 0)
        rt = max(1, flips // 2)
        gross = sum(pos[i] * ret[i] for i in idx) * 1e4
        held = sum(1 for i in idx if pos[i] != 0.0)
        return gross, rt, held

    cost = CDE_RT_COST[asset]
    print(f"\n{asset}: runtime-series window {days[0]} .. {days[-1]}  n={n} days "
          f"(hist {len(hist)} rows + bank {len(bank)} rows, {len(set(hist) & set(bank))} overlap)")
    for label, idx in (("FULL", list(range(n))),
                       ("2025", [i for i in range(n) if days[i] < "2026-01-01"]),
                       ("2026", [i for i in range(n) if days[i] >= "2026-01-01"])):
        if len(idx) < 50:
            continue
        gross, rt, held = stats(idx)
        per_rt = gross / rt
        net_rt = per_rt - cost
        verdict = ("ALIVE-CLASS (>=2x cost)" if per_rt >= 2 * cost
                   else ("KILL-CLASS (<= cost)" if per_rt <= cost else "NOISE-BAND"))
        print(f"  {label:5s} gross {gross/100:+7.2f}%  round-trips {rt:3d}  held {held}/{len(idx)}d  "
              f"gross/RT {per_rt:+7.1f}bps vs cost {cost:.1f}  net/RT {net_rt:+7.1f}bps  -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hist", required=True, help="dir holding skew_hist_{ASSET}.jsonl (MCP backfill)")
    ap.add_argument("--bank", required=True, help="dir holding laevitas_apiv2_skew_{ASSET}.jsonl (pulled bank)")
    ap.add_argument("--ohlcv", default=os.path.join("training", "training_data", "drl_training"))
    ap.add_argument("--assets", default="BTC,ETH")
    args = ap.parse_args()
    for a in [s.strip().upper() for s in args.assets.split(",") if s.strip()]:
        probe(a, args.hist, args.bank, args.ohlcv)
    print("\nPre-committed interpretation (see docstring): this window can KILL or keep-alive; it cannot CERTIFY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

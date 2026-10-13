#!/usr/bin/env python3
"""[P422] Blocker counterfactual — what did each entry-blocking control cost?

WHY: the 2026-10-13 no-trade investigation found the regime book WANTED a
position on most ticks (SOL 145/186, ETH 60/186 over 30 days) and got 4
actionable ticks. The orders were stopped by controls that have no forward
evidence of their own: the correlation EXIT_ONLY P0 mode, the BitBeast
Structure/Volume entry vetoes, the ETF de-risk hold, and the alpha gate. A
blocker that stops a certified book's entries must be judged the way the
book was judged: on the forward return of the ticks it blocked.

METHOD: join the per-tick final intents (``data/diag_intents_pulled.csv``,
extracted from the server's diagnostics) with the regimebook ledger (the
BOOK's own direction, pulled) and the 4H closes. For every tick where the
book was directional, classify the blocker from the veto reason and compute
the signed forward return of the BOOK direction at 4/12/24 bars. A blocked
tick's forward return is the return the block FORGAVE (if positive, the block
cost money; if negative, it avoided a loss).

PRE-COMMITTED VERDICT (written before the first run), per blocker:
  * EARNS  — mean signed forward return of the ticks it blocked is NEGATIVE at
             h=12 with |t| >= 1.5 (overlap-corrected, n_eff = n/h): the block
             avoided losses.
  * COSTS  — mean signed forward return at h=12 is POSITIVE and exceeds the
             per-entry cost (CDE round trip / 2) with |t| >= 1.5: the block
             forgave money the book would have made.
  * NOISE  — anything else. NOISE is not a licence to remove a control; it
             means this window cannot decide.
The t-statistic uses n_eff = n/h because consecutive 4H ticks overlap at
every horizon > 1 bar (P231).

HONESTY: one window (Sep 13 → Oct 13, a chop/downtrend era); the book's own
forward IC in this window was NEGATIVE (P421), so "the block cost money"
cannot occur where the book itself was losing — any COSTS verdict here is
therefore conservative. Overlapping ticks; a tick is counted once per
blocker. This does not re-litigate the book; it prices the controls that
sit on top of it.

Usage (operator-local, after a september_check pull):
    python -X utf8 training/scripts/blocker_counterfactual_p422.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DIAG_CSV = REPO / "data" / "diag_intents_pulled.csv"
LEDGER_DIR = REPO / "data" / "strategy_shadow_pulled"
OHLCV_DIR = REPO / "training" / "training_data" / "drl_training"
REPORT = REPO / "training" / "reports" / "blocker_counterfactual_p422.json"

ASSETS = ("BTC", "ETH", "SOL")
HORIZONS = (4, 12, 24)
# CDE round-trip cost per asset (core.cde_fees, P334/P382 roster convention):
# a BLOCKED entry forgoes a round trip, so the per-entry bar is RT/2 per leg
# x 2 legs = the full RT. We charge the full RT so a COSTS verdict is not
# cheap to reach.
CDE_RT_BPS = {"BTC": 19.7, "ETH": 29.0, "SOL": 29.0}
T_BAR = 1.5
WINDOW_START = "2026-09-13"


def classify(reason: str, direction: float, book_dir: float, etf_dir) -> str:
    r = reason or ""
    if r == "" or r == "None":
        return "ACTIONABLE"
    if r.startswith("[P0_SAFETY] Exit-only mode"):
        return "P0_EXIT_ONLY_MODE"
    if r.startswith("[P0_SAFETY]"):
        return "P0_OTHER"
    if r.startswith("[VETO] Structure") or "; Structure" in r:
        return "BITBEAST_STRUCTURE"
    if r.startswith("[VETO] Volume") or r.startswith("[VETO] SNR"):
        return "BITBEAST_VOLUME_SNR"
    if "CORRELATION_COLLAPSE" in r:
        return "CORRELATION_COLLAPSE"
    if "VOLUME_CONTRACTING" in r or "STALE_DATA" in r:
        return "TRADE_GATE"
    if "Alpha gate" in r:
        # the book wanted a position but the gate saw direction 0 -> the
        # direction was zeroed UPSTREAM of the gate: the ETF de-risk seat on
        # BTC/ETH when its held state is -1, else unknown upstream zeroing.
        if abs(direction) < 0.1 and book_dir > 0 and etf_dir is not None and etf_dir < 0:
            return "ETF_DERISK_HOLD"
        if abs(direction) < 0.1:
            return "ALPHA_GATE_DIR_ZEROED_UPSTREAM"
        return "ALPHA_GATE_FEE_FLOOR"
    return "OTHER"


def load_ledger(asset: str, name: str):
    rows = []
    p = LEDGER_DIR / f"{name}_{asset}.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ts = r.get("ts")
        if ts is None and r.get("iso"):
            ts = datetime.fromisoformat(r["iso"].replace("Z", "+00:00")).timestamp()
        if ts is None:
            continue
        rows.append((float(ts), r))
    rows.sort(key=lambda x: x[0])
    return rows


def nearest_before(rows, ts: float, max_age: float):
    """Last ledger row at or before ts, or the first row within +10 min
    (the harness ticks ~1 min AFTER decide, so the same-tick row lands just
    after the intent)."""
    best = None
    for t, r in rows:
        if t <= ts + 600:
            best = (t, r)
        else:
            break
    if best is None or ts - best[0] > max_age:
        return None
    return best[1]


def main() -> int:
    if not DIAG_CSV.exists():
        print(f"REFUSING: {DIAG_CSV} missing — extract it from the server's "
              f"diagnostics first (see the P422 entry).", file=sys.stderr)
        return 2
    closes = {}
    for a in ASSETS:
        p = OHLCV_DIR / f"{a}_4H_ohlcv.parquet"
        if not p.exists():
            print(f"REFUSING: no OHLCV for {a} at {p}", file=sys.stderr)
            return 2
        df = pd.read_parquet(p)
        if "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"], utc=True)
        else:
            idx = pd.to_datetime(df.index, utc=True)
        closes[a] = pd.Series(df["close"].to_numpy(), index=idx).sort_index()
    books = {a: load_ledger(a, "regimebook") for a in ASSETS}
    etfs = {a: load_ledger(a, "etfflow") for a in ("BTC", "ETH")}

    per = defaultdict(lambda: defaultdict(list))  # asset -> class -> list of (fwd by h)
    counts = defaultdict(lambda: defaultdict(int))
    wanted = defaultdict(int)
    with DIAG_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = row["asset"]
            if a not in ASSETS or not row["tick_time"] or row["tick_time"][:10] < WINDOW_START:
                continue
            ts = datetime.fromisoformat(row["tick_time"].replace("Z", "+00:00")).timestamp()
            book = nearest_before(books[a], ts, max_age=5 * 3600)
            if book is None:
                continue
            book_dir = float(book.get("direction") or 0.0)
            if abs(book_dir) < 0.5:
                continue  # the book itself said flat: nothing was blocked
            wanted[a] += 1
            etf = nearest_before(etfs[a], ts, max_age=14 * 3600) if a in etfs else None
            etf_dir = float(etf.get("direction")) if etf and etf.get("direction") is not None else None
            try:
                direction = float(row["direction"] or 0.0)
            except ValueError:
                direction = 0.0
            cls = classify(row["veto_reason"], direction, book_dir, etf_dir) \
                if row["veto_active"] == "True" else "ACTIONABLE"
            counts[a][cls] += 1
            s = closes[a]
            i = s.index.searchsorted(pd.Timestamp(ts, unit="s", tz="UTC"), side="right") - 1
            if i < 0:
                continue
            fwd = {}
            for h in HORIZONS:
                if i + h < len(s):
                    fwd[h] = book_dir * (float(s.iloc[i + h]) / float(s.iloc[i]) - 1.0) * 1e4
            per[a][cls].append(fwd)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "window_start": WINDOW_START, "horizons_bars": list(HORIZONS),
              "cost_rt_bps": CDE_RT_BPS, "t_bar": T_BAR, "assets": {}}
    print(f"\nBLOCKER COUNTERFACTUAL (ticks where the BOOK was directional; signed fwd "
          f"return of the book's direction, bps)\n")
    print(f"{'asset':5} {'blocker':32} {'n':>4}  " +
          "  ".join(f"h{h}:mean(t)".rjust(14) for h in HORIZONS) + "   verdict(h12)")
    for a in ASSETS:
        report["assets"][a] = {"book_wanted_ticks": wanted[a], "blockers": {}}
        for cls, fwds in sorted(per[a].items(), key=lambda kv: -len(kv[1])):
            stats = {}
            cells = []
            for h in HORIZONS:
                xs = [f[h] for f in fwds if h in f]
                n = len(xs)
                if n == 0:
                    stats[h] = {"n": 0}
                    cells.append("n/a".rjust(14))
                    continue
                mean = sum(xs) / n
                var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
                n_eff = max(n / h, 1.0)
                t = mean / math.sqrt(var / n_eff) if var > 0 else 0.0
                stats[h] = {"n": n, "n_eff": round(n_eff, 1), "mean_bps": round(mean, 1),
                            "t": round(t, 2), "sum_bps": round(sum(xs), 1)}
                cells.append(f"{mean:+7.1f}({t:+.1f})".rjust(14))
            v = "NOISE"
            s12 = stats.get(12, {})
            if s12.get("n", 0) >= 10:
                if s12["mean_bps"] < 0 and abs(s12["t"]) >= T_BAR:
                    v = "EARNS"
                elif s12["mean_bps"] > CDE_RT_BPS[a] and abs(s12["t"]) >= T_BAR:
                    v = "COSTS"
            if cls == "ACTIONABLE":
                v = "(baseline)"
            report["assets"][a]["blockers"][cls] = {"n": len(fwds), "per_horizon": stats, "verdict": v}
            print(f"{a:5} {cls:32} {len(fwds):4d}  " + "  ".join(cells) + f"   {v}")
        print(f"      book directional on {wanted[a]} ticks")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

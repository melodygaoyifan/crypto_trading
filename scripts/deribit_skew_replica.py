#!/usr/bin/env python3
"""[P421] Deribit-direct replica of the Laevitas apiv2 by-tenor tenor-30 skew —
the FREE feed that keeps the runtime skew bank accruing after the paid key died
(2026-09-10, 403; operator declined renewal).

WHAT IT DOES AND DOES NOT DO: it computes one bank-format row per asset per run
and (with --append) merges it into data/laevitas_apiv2_skew_{ASSET}.jsonl,
EXISTING ROWS ALWAYS WINNING — it never overwrites an engine-, MCP- or
previously-banked row. It does NOT feed the live skew seat (skew_seat_mode is
"off"/PARKED per P421; re-arming the seat is its own P-entry and would also
need the seat's feed pointed here). The bank exists so the P420-residue
recalibration (`training/skew_seat_calibration.py` on the runtime series) stays
possible at $0.

THE RECIPE (the vendor's own documented computation, replicated):
  exchange mark-IV analytics from the public book summary (one request per
  currency, no key): eligible quotes have positive mark_iv and positive time
  to expiry; per expiry the ATM is the CALL nearest delta +0.50 (expiry
  ineligible when the gap exceeds 0.15), wings are the quotes nearest
  +/-0.25 and +/-0.10 delta (no max-distance test); each selected IV series is
  independently interpolated in TOTAL-VARIANCE space to the 30d tenor, flat
  outside the observed expiry range; skew = put IV - call IV (positive = puts
  richer); IVs in percentage points. Delta is Black-76 N(d1) on the
  expiry-specific forward (underlying_price), r=0 — the vendor's delta may
  differ in premium-adjustment details, which is exactly what --validate
  measures against a reference row.

FAIL DIRECTION: any fetch/parse/selection failure exits 2 and writes NOTHING —
a fabricated row in a calibration series is worse than a gap (P2/P199).
stdlib-only on purpose: the server host cron runs it with bare python3.

    python -X utf8 scripts/deribit_skew_replica.py --assets BTC,ETH            # print rows
    python -X utf8 scripts/deribit_skew_replica.py --assets BTC --append <bank.jsonl>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import urllib.request

BOOK_URL = ("https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
            "?currency={cur}&kind=option")
TENOR_DAYS = 30.0
ATM_GAP_MAX = 0.15
EXPIRY_HOUR_UTC = 8  # Deribit options expire 08:00 UTC
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
SOURCE = "deribit_replica_v1"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_instrument(name: str):
    """'BTC-12OCT26-84500-C' -> (expiry_dt, strike, is_call) or None."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    try:
        d = parts[1]
        day = int(d[:-5])
        mon = _MONTHS[d[-5:-2]]
        yr = 2000 + int(d[-2:])
        expiry = dt.datetime(yr, mon, day, EXPIRY_HOUR_UTC, tzinfo=dt.timezone.utc)
        strike = float(parts[2].replace("d", "."))
        is_call = parts[3].upper() == "C"
        return expiry, strike, is_call
    except (KeyError, ValueError, IndexError):
        return None


def call_delta(fwd: float, strike: float, iv_pct: float, t_years: float) -> float:
    """Black-76 N(d1), r=0. iv in percentage points."""
    sigma = iv_pct / 100.0
    if fwd <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return float("nan")
    d1 = (math.log(fwd / strike) + 0.5 * sigma * sigma * t_years) / (sigma * math.sqrt(t_years))
    return _norm_cdf(d1)


def tenor30_row(rows: list, now: dt.datetime) -> dict:
    """The five tenor-30 IVs from one book-summary payload. Raises ValueError
    when the chain cannot support the computation (the caller refuses)."""
    by_expiry: dict = {}
    for r in rows:
        iv = r.get("mark_iv")
        fwd = r.get("underlying_price")
        parsed = parse_instrument(str(r.get("instrument_name", "")))
        if not parsed or not iv or iv <= 0 or not fwd or fwd <= 0:
            continue
        expiry, strike, is_call = parsed
        t = (expiry - now).total_seconds() / (365.0 * 86400.0)
        if t <= 0:
            continue
        cd = call_delta(float(fwd), strike, float(iv), t)
        if not math.isfinite(cd) or cd <= 0.0 or cd >= 1.0:
            continue
        delta = cd if is_call else cd - 1.0  # put delta = N(d1) - 1
        by_expiry.setdefault(expiry, {"t": t, "calls": [], "puts": []})
        side = "calls" if is_call else "puts"
        by_expiry[expiry][side].append((delta, float(iv)))

    def nearest(quotes, target):
        if not quotes:
            return None
        d, iv = min(quotes, key=lambda q: abs(q[0] - target))
        return d, iv

    # per-expiry selections: {series: (t, iv)}
    series = {k: [] for k in ("atm", "c25", "p25", "c10", "p10")}
    for expiry in sorted(by_expiry):
        e = by_expiry[expiry]
        atm = nearest(e["calls"], 0.50)
        if atm is None or abs(atm[0] - 0.50) > ATM_GAP_MAX:
            continue  # expiry ineligible without a usable ATM (the recipe)
        series["atm"].append((e["t"], atm[1]))
        for key, side, target in (("c25", "calls", 0.25), ("p25", "puts", -0.25),
                                  ("c10", "calls", 0.10), ("p10", "puts", -0.10)):
            sel = nearest(e[side], target)
            if sel is not None:  # wings: no max-distance test (the recipe)
                series[key].append((e["t"], sel[1]))

    t30 = TENOR_DAYS / 365.0

    def interp(points):
        """Total-variance interpolation to t30; flat outside the range."""
        if not points:
            return None
        pts = sorted(points)
        if t30 <= pts[0][0]:
            return pts[0][1]
        if t30 >= pts[-1][0]:
            return pts[-1][1]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= t30 <= t1:
                w0, w1 = v0 * v0 * t0, v1 * v1 * t1
                w = w0 + (w1 - w0) * (t30 - t0) / (t1 - t0)
                return math.sqrt(max(w, 0.0) / t30)
        return None

    out = {k: interp(v) for k, v in series.items()}
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise ValueError(f"chain cannot support tenor-30 series {missing} "
                         f"({len(by_expiry)} expiries, "
                         f"{len(series['atm'])} with a usable ATM)")
    return {
        "date": now.strftime("%Y-%m-%d"), "tenor": 30,
        "skew_25d": out["p25"] - out["c25"],
        "skew_10d": out["p10"] - out["c10"],
        "call_25d_iv": out["c25"], "put_25d_iv": out["p25"],
        "atm_iv": out["atm"], "source": SOURCE,
    }


def fetch_row(asset: str, now: dt.datetime) -> dict:
    url = BOOK_URL.format(cur=asset.upper())
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
    rows = payload.get("result")
    if not rows:
        raise ValueError(f"empty book summary for {asset}")
    return tenor30_row(rows, now)


def append_row(bank_path: str, row: dict) -> str:
    """Merge-by-date, EXISTING WINS (an engine/MCP row is never overwritten).
    Returns 'appended' | 'kept_existing'."""
    rows = {}
    if os.path.exists(bank_path):
        with open(bank_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows[r["date"]] = r
    if row["date"] in rows:
        return "kept_existing"
    rows[row["date"]] = row
    tmp = bank_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for d in sorted(rows):
            fh.write(json.dumps(rows[d]) + "\n")
    os.replace(tmp, bank_path)
    return "appended"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", default="BTC,ETH")
    ap.add_argument("--append", default=None, metavar="BANK_TEMPLATE",
                    help="bank path with {ASSET} placeholder, e.g. "
                         "/data/laevitas_apiv2_skew_{ASSET}.jsonl")
    args = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    failures = 0
    for asset in [a.strip().upper() for a in args.assets.split(",") if a.strip()]:
        try:
            row = fetch_row(asset, now)
        except Exception as exc:  # refuse loudly, never fabricate (P2/P199)
            print(f"{asset}: REFUSED — {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"{asset}: {json.dumps(row)}")
        if args.append:
            path = args.append.replace("{ASSET}", asset)
            print(f"{asset}: {append_row(path, row)} -> {path}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

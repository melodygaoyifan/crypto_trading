"""[P407f] The PRODUCER for core.seat_alpha.SKEW_CONTRA_ALPHA_BY_ERA.

The skew-seat calibration (P407e) was measured by a scratch probe that is not
in the repo, so the shipped table could not be re-derived -- the exact gap
[P326] closed for the regimebook seat. This is its counterpart: a committed,
tested script that recomputes the per-year gross-per-round-trip edge of the
contrarian 25-delta-skew rule and, with --verify, compares it against the
shipped table and exits 3 on drift.

REPRODUCIBILITY IS OPERATOR-LOCAL, NOT CI (the P213 discipline). The input is
6.6y of Deribit 25d skew + spot from Laevitas deep history, which is:
  * capped at 3 months on the plain API key (P406) -- so it cannot be re-fetched
    headlessly; it was pulled once via the logged-in dashboard backend (P407),
  * proprietary Laevitas data -- so it is NOT committed (training/training_data/
    is gitignored, like every other market-data artifact, P199).
The four files live operator-local at training/training_data/laevitas_skew/:
    skew_{btc,eth}_25d.json   (field "30" = 25d skew at the 30d tenor)
    gex_{btc,eth}.json        (field "index_price" = daily spot)
If they are absent this script REFUSES (exit 2) -- "cannot reproduce" must never
read as "reproduces" (P159/P213).

THE RULE (identical to the P407 edge_calib.py probe):
  contrarian z-deadband on the 25d skew (win 30, min 8 obs, band 1.0, hold
  inside the band), traded on next-day spot return; gross bps per ROUND TRIP =
  sum(pos*ret)*1e4 / max(1, flips//2), measured per CALENDAR-YEAR era
  (2021-2026, min 50 overlapping days). The asserted value is the era-MEDIAN
  (P321: robust central estimate, not carried by one dominant era). Gross, not
  net -- matching what check_alpha_gate compares against (it prices friction
  separately). No lookahead: the z-window is strictly trailing and the return
  is next-day.

    python -X utf8 training/skew_seat_calibration.py            # print the table
    python -X utf8 training/skew_seat_calibration.py --verify   # exit 3 on drift
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys
from typing import Dict, List, Optional

# repo root on sys.path so `from core...` resolves when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_DATA_DIR = os.path.join("training", "training_data", "laevitas_skew")
_ASSETS = {"BTC": "btc", "ETH": "eth"}
_ERA_YEARS = list(range(2021, 2027))  # 2021..2026
_Z_WIN = 30
_Z_MIN = 8
_BAND = 1.0
_NAN = float("nan")


def _load(data_dir: str, fname: str) -> list:
    with open(os.path.join(data_dir, fname), encoding="utf-8") as fh:
        return json.load(fh)


def _by_day(rows: list, field: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for r in rows:
        t = r.get("date")
        v = r.get(field)
        if t is None or v is None:
            continue
        try:
            out[int(t) // 86400000] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _year(day_epoch: int) -> int:
    return (_dt.date(1970, 1, 1) + _dt.timedelta(days=int(day_epoch))).year


def _positions(sig: List[float]) -> List[float]:
    """Contrarian z-deadband, strictly trailing window, hold inside the band."""
    pos: List[float] = []
    prev = 0.0
    for i in range(len(sig)):
        window = sig[max(0, i - _Z_WIN):i]
        if len(window) < _Z_MIN:
            z = 0.0
        else:
            mu = statistics.fmean(window)
            sd = statistics.pstdev(window)
            z = 0.0 if sd == 0 else (sig[i] - mu) / sd
        contra = -z
        p = 1.0 if contra > _BAND else (-1.0 if contra < -_BAND else prev)
        pos.append(p)
        prev = p
    return pos


def calibrate(asset: str, data_dir: str = _DEFAULT_DATA_DIR
              ) -> Optional[Dict[str, float]]:
    """Per-year gross bps/round-trip + median. None if the data is absent."""
    a = _ASSETS.get(str(asset).upper())
    if a is None:
        return None
    try:
        sk = _by_day(_load(data_dir, "skew_" + a + "_25d.json"), "30")
        spot = _by_day(_load(data_dir, "gex_" + a + ".json"), "index_price")
    except FileNotFoundError:
        return None
    days = sorted(set(sk) & set(spot))
    n = len(days)
    if n < 100:
        return None
    sp = [spot[d] for d in days]
    ret = [0.0] * n
    for i in range(n - 1):
        ret[i] = sp[i + 1] / sp[i] - 1.0 if sp[i] else 0.0
    pos = _positions([sk[d] for d in days])
    yrs = [_year(d) for d in days]

    out: Dict[str, float] = {}
    per_era: List[float] = []
    for y in _ERA_YEARS:
        idx = [i for i in range(n) if yrs[i] == y]
        if len(idx) < 50:
            continue
        seq = [0.0] + [pos[i] for i in idx]
        flips = sum(1 for k in range(1, len(seq)) if abs(seq[k] - seq[k - 1]) > 0)
        rt = max(1, flips // 2)
        gross = sum(pos[i] * ret[i] for i in idx) * 1e4
        v = round(gross / rt, 1)
        out[str(y)] = v
        per_era.append(v)
    if not per_era:
        return None
    out["__median__"] = round(statistics.median(per_era), 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", default="BTC,ETH")
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    ap.add_argument("--verify", action="store_true",
                    help="compare against core.seat_alpha and exit 3 on drift")
    args = ap.parse_args(argv)
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    results: Dict[str, Dict[str, float]] = {}
    for a in assets:
        r = calibrate(a, args.data_dir)
        if r is None:
            print("REFUSING: no reproducible input for " + a + " under "
                  + repr(args.data_dir) + ". The Laevitas deep-history files are "
                  "operator-local and gitignored (see the module docstring); "
                  "'cannot reproduce' is not 'reproduces'.", file=sys.stderr)
            return 2
        results[a] = r

    print("asset  " + "  ".join(str(y) for y in _ERA_YEARS) + "     MEDIAN")
    for a in assets:
        r = results[a]
        cells = "  ".join("{:6.1f}".format(r.get(str(y), _NAN)) for y in _ERA_YEARS)
        print("{:6s} {}   {:8.1f}".format(a, cells, r["__median__"]))

    if not args.verify:
        return 0

    from core.seat_alpha import (SKEW_CONTRA_ALPHA_BY_ERA as SHIPPED_ERA,
                                 SKEW_CONTRA_ALPHA_BPS_PER_ROUND_TRIP as SHIPPED_MED)
    drift = []
    for a in assets:
        r = results[a]
        ship_era = SHIPPED_ERA.get(a, {})
        for y in _ERA_YEARS:
            got = r.get(str(y))
            exp = ship_era.get(str(y))
            if got is None or exp is None or abs(got - exp) > 0.6:
                drift.append((a, str(y), got, exp))
        got_med = r["__median__"]
        exp_med = SHIPPED_MED.get(a)
        if exp_med is None or abs(got_med - exp_med) > 0.6:
            drift.append((a, "MEDIAN", got_med, exp_med))
    if drift:
        print("\nDRIFT vs core.seat_alpha in " + str(len(drift)) + " cell(s):",
              file=sys.stderr)
        for a, era, got, exp in drift:
            print("  " + a + " " + era + ": computed " + str(got) + " vs shipped "
                  + str(exp), file=sys.stderr)
        print("Either the data changed or the table was edited by hand. Re-derive "
              "only from this lab, only the median, and never to make a trade pass "
              "(the P320 revision rule).", file=sys.stderr)
        return 3
    print("\nOK - reproduces core.seat_alpha.SKEW_CONTRA_ALPHA_BY_ERA "
          "(per-year + median) within rounding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

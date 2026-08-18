"""[P232] Shadow slope calibrator — the honest replacement for the x65/x40
alpha constants, run OFFLINE, changing nothing.

WHY (P231 research): live alpha = 40 x |trend_sig| x 0.75, where 40 is a
constant documented as chosen to clear the gate and 0.75 is a frozen
"hit rate". Measured realized slope over 60d: +1.2bps/unit (4h) / -9.1
(16h) vs the coded 65. Grinold: alpha = IC x sigma x score — a fixed
multiplier conflates IC and vol and cannot survive regime drift.

WHAT: per-asset rolling OLS slope of realized forward return (bps) on the
live signal (quant_direction from the attribution logs — post trend
injection, i.e. the number the gate actually prices), shrunk to a ZERO
prior with weight w = n_eff/(n_eff + 270), floored at 0 (a negative slope
means REFUSE, never invert), capped at 49 bps/unit (today's effective
ceiling — the calibrator may never claim MORE edge than the current
system does). Prints what the shadow calibration implies against the live
thresholds, so the gate consequence is visible before anyone flips
anything.

WHERE THIS RUNS (P213): in-container (attribution volume) —
    docker exec hmats-engine python -X utf8 \
        analytics/calibration/slope_calibrator.py
or operator-local with --log-dir. Missing data or unreachable prices =>
REFUSAL (exit 2). Cutover to a live calibrated gate requires >=4 weeks of
these reports dual-logged against the live constants AND retiring the
hit-rate factor in the same change (P231: removing it alone is a +33%
loosening).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.ic.agent_ic_review import (  # noqa: E402  (shared, P172 lesson)
    KRAKEN_PAIRS, fetch_closes, load_signal_records, resolve_log_dir, _refuse)

SHRINK_K = 270          # half-weight at ~half a 90d window of 4H bars
SLOPE_CAP = 49.0        # 65 x 0.75 — today's effective ceiling, never exceed
HORIZONS = (1, 4)       # 4h, 16h
# Live thresholds as of P230 verification (decomposition in CLAUDE.md P231);
# refreshed automatically when a newer diag value is passed via --thresholds.
# [P287] These are a SNAPSHOT of the live gate arithmetic (friction x
# smart-beta gate mult), not a constant of nature — friction moves (the
# P270 maker-first activation changed priced friction AFTER this stamp),
# and the P237 tripwire consumes the TRADEABLE/GATE-CLOSED verdict string
# these produce verbatim. Provenance is stamped and staleness warned below
# so a Sep-1 deactivation decision cannot silently ride a frozen snapshot.
# [P293k 2026-08-17] REFRESHED. The previous snapshot {BTC 28.93, ETH 42.77,
# SOL 55.34} was stamped 2026-08-08 and had been overtaken TWICE by code:
#   P289 (08-16) re-priced spreads from the Kraken-era constants
#   P291b (08-17) armed venue-true hold (CDE posts collateral, not a borrow)
# leaving the tripwire comparing against thresholds 1.5-1.9x too high
# (BTC 1.51x, ETH 1.60x, SOL 1.91x).
#
# THE DEFECT THIS EXPOSED, which matters more than the numbers: the staleness
# guard below is TIME-based (30 days), but these thresholds move when FRICTION
# CODE changes — which happened one day before the guard's window opened. A
# time-based check cannot detect a code-driven change, so it reported "verified
# 10d ago" while the values were already wrong. Any P-entry that moves friction
# (fees, spreads, hold cost, smart-beta gate mult) MUST update these and
# re-stamp, or pass --thresholds; the cron should prefer --thresholds.
#
# Immaterial to the 2026-08-17 verdict — published max alpha was 0.0/1.9/1.2
# against even the CORRECTED thresholds, so GATE-CLOSED stands either way —
# but it would silently distort the first period where slopes actually rise.
DEFAULT_THRESHOLDS = {"BTC": 19.1, "ETH": 26.7, "SOL": 29.0}
THRESHOLDS_STAMPED = "2026-08-17"   # P293k: post-P289 spreads + P291b hold
THRESHOLDS_STALE_AFTER_DAYS = 30


def default_thresholds_age_days(today=None) -> int:
    """[P287] Days since the default thresholds were verified against the
    live gate. Pure for tests."""
    t = today or datetime.now(timezone.utc).date()
    stamped = datetime.strptime(THRESHOLDS_STAMPED, "%Y-%m-%d").date()
    return (t - stamped).days


def shrunk_slope(pairs: list[tuple[float, float]], overlap: int) -> dict:
    """OLS slope (fwd bps on signed direction) + zero-prior shrinkage."""
    n = len(pairs)
    if n < 30:
        return {"n": n, "verdict": "INSUFFICIENT"}
    mx = sum(d for d, _ in pairs) / n
    my = sum(f for _, f in pairs) / n
    vx = sum((d - mx) ** 2 for d, _ in pairs)
    if vx <= 0:
        return {"n": n, "verdict": "DEGENERATE"}
    slope = sum((d - mx) * (f - my) for d, f in pairs) / vx
    resid = [f - (my + slope * (d - mx)) for d, f in pairs]
    n_eff = max(3, n // overlap)  # overlap-corrected (P231)
    se = (math.sqrt(sum(r * r for r in resid) / max(1, n - 2) / vx)
          * math.sqrt(max(1.0, n / n_eff)))
    t = slope / se if se > 0 else 0.0
    w = n_eff / (n_eff + SHRINK_K)
    shrunk = w * slope  # prior = 0
    published = max(0.0, min(SLOPE_CAP, shrunk))  # floor 0, cap 49
    return {"n": n, "n_eff": n_eff, "slope_ols": round(slope, 2),
            "t_overlap_corrected": round(t, 2), "shrink_w": round(w, 3),
            "slope_shrunk": round(shrunk, 2),
            "slope_published": round(published, 2),
            "floored": shrunk < 0, "capped": shrunk > SLOPE_CAP}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--thresholds", default=None,
                    help='JSON like {"BTC":28.9,...}; default = P230 values')
    ap.add_argument("--report-dir", default=None,
                    help="[P235] pass /opt/hmats/data/evidence_reports "
                         "in-container so weekly reports land on the "
                         "persistent volume, not the ephemeral container FS")
    args = ap.parse_args()
    thresholds = (json.loads(args.thresholds) if args.thresholds
                  else dict(DEFAULT_THRESHOLDS))

    records = load_signal_records(resolve_log_dir(args.log_dir),
                                  args.window_days)
    if not records:
        _refuse("signal files exist but hold no records in the window.")
    bars = {a: fetch_closes(a) for a in KRAKEN_PAIRS}

    pairs: dict = {a: {h: [] for h in HORIZONS} for a in KRAKEN_PAIRS}
    # [P293k] entry-time-aligned companion series (see the loop below)
    pairs_entry: dict = {a: {h: [] for h in HORIZONS}
                         for a in KRAKEN_PAIRS}
    for rec in records:
        q = next((s for s in rec.get("signals", [])
                  if s.get("agent_name") == "quant"), None)
        if not q:
            continue
        d = float(q.get("direction", 0.0) or 0.0)
        if abs(d) < 1e-9:
            continue
        ts_list, closes = bars[rec["asset"]]
        i = bisect_right(ts_list, rec["_ts"]) - 1
        for h in HORIZONS:
            if 0 <= i and i + h < len(closes):
                pairs[rec["asset"]][h].append(
                    (d, (closes[i + h] / closes[i] - 1.0) * 1e4))
            # [P293k] SECOND ALIGNMENT, reported alongside. `closes[i]` is the
            # close of the bar CONTAINING the signal, so the return above
            # starts ~4h AFTER the decision — it excludes the bar the sleeve
            # actually enters in. That is conservative against look-ahead but
            # it is NOT what the book experiences.
            #
            # Measured 2026-08-17, the difference is material and runs the
            # UNFAVOURABLE way: BTC 4h -0.74 -> -6.07, ETH +3.01 -> -7.15,
            # SOL +1.91 -> -10.89. The entry bar is where this signal loses
            # most, which is consistent with entering after the move.
            #
            # Both are reported rather than silently switching: changing the
            # basis would invalidate every prior weekly report the tripwire
            # has counted, and the look-ahead concern behind the original
            # choice is legitimate. The operator sees both numbers and the
            # tripwire keeps keying on the established one.
            if 0 <= i - 1 and (i - 1) + h < len(closes):
                pairs_entry[rec["asset"]][h].append(
                    (d, (closes[(i - 1) + h] / closes[i - 1] - 1.0) * 1e4))

    print(f"Shadow slope calibration — {args.window_days}d window, "
          f"shrink k={SHRINK_K}, prior=0, floor=0, cap={SLOPE_CAP}")
    print(f"live constants for comparison: trend 40 x 0.75 = 30.0 effective; "
          f"quant 65 x 0.75 = 48.75 effective (both bps/unit)")
    # [P287] threshold provenance — printed in every report so the verdict
    # strings the tripwire consumes carry their own freshness.
    using_defaults = args.thresholds is None
    thr_age = default_thresholds_age_days() if using_defaults else None
    if using_defaults:
        print(f"thresholds: P230 defaults, verified {THRESHOLDS_STAMPED} "
              f"({thr_age}d ago)")
        if thr_age is not None and thr_age > THRESHOLDS_STALE_AFTER_DAYS:
            print(f"!! THRESHOLD STALENESS: the default enter thresholds "
                  f"were verified {thr_age}d ago (> "
                  f"{THRESHOLDS_STALE_AFTER_DAYS}d). Friction has moved "
                  f"since (e.g. maker-first, P270) and the tripwire keys on "
                  f"the TRADEABLE/GATE-CLOSED strings below — re-derive the "
                  f"thresholds from the live gate arithmetic (CLAUDE.md "
                  f"P231 decomposition) and pass --thresholds. This tool "
                  f"cannot compute them itself (needs container state).")
    else:
        print("thresholds: operator-supplied via --thresholds")
    print()
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "window_days": args.window_days,
              "thresholds_provenance": {
                  "source": "P230_defaults" if using_defaults else "cli",
                  "stamped": THRESHOLDS_STAMPED if using_defaults else None,
                  "age_days": thr_age,
                  "stale": bool(using_defaults and thr_age is not None
                                and thr_age > THRESHOLDS_STALE_AFTER_DAYS)},
              "assets": {}}
    for a in KRAKEN_PAIRS:
        report["assets"][a] = {}
        for h in HORIZONS:
            r = shrunk_slope(pairs[a][h], overlap=h)
            # [P293k] the same statistic on the entry-aligned basis, reported
            # for comparison only — the verdict below still keys on `r`.
            r_entry = shrunk_slope(pairs_entry[a][h], overlap=h)
            r["entry_aligned_slope_ols"] = r_entry.get("slope_ols")
            r["entry_aligned_t"] = r_entry.get("t_overlap_corrected")
            report["assets"][a][f"{h*4}h"] = r
            if "slope_published" in r:
                max_alpha = r["slope_published"]  # at |dir| = 1.0
                verdict = ("TRADEABLE" if max_alpha >= thresholds.get(a, 1e9)
                           else "GATE-CLOSED under honest calibration")
                r["vs_threshold"] = (f"max alpha {max_alpha:.1f} vs enter "
                                     f"{thresholds.get(a)} -> {verdict}")
            print(f"{a} {h*4:>3}h: {json.dumps(r)}")
    rep_dir = (Path(args.report_dir) if args.report_dir
               else Path(__file__).resolve().parent / "reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    # [P253d] The P237 tripwire reads /opt/hmats/data/evidence_reports and
    # the two defaults never agreed — coupling rested entirely on the cron
    # passing --report-dir. The tripwire REFUSES on an empty dir (safe),
    # but a calibrator run that lands its report where the tripwire will
    # never look should say so at write time, not be discovered at the
    # tripwire's next refusal.
    _tw_dir = Path("/opt/hmats/data/evidence_reports")
    if not args.report_dir and rep_dir.resolve() != _tw_dir:
        print(f"NOTE: writing to {rep_dir} — the P237 tripwire reads "
              f"{_tw_dir}; pass --report-dir there (the cron does) if this "
              f"report should count toward the tripwire window.")
    out = rep_dir / ("slope_" +
                     datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                     + ".json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    print("NOTE: 'GATE-CLOSED under honest calibration' is a finding about "
          "the SIGNAL, not a malfunction — the policy choice it forces "
          "(trade on the uncalibrated constant vs barely trade) is the "
          "operator's, recorded in P231. Cutover checklist: >=4wk of these "
          "reports, retire the hit-rate factor in the SAME change, "
          "rise-rate cap 15%/refit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

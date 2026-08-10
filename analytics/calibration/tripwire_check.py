"""[P237] The calibration tripwire, made executable — checks, never acts.

THE DECISION (P237, operator-delegated 2026-08-08): trading continues on the
asserted alpha constants ONLY through the 4-report dual-log window — weekly
Monday `slope_calibrator` reports, first 2026-08-11, fourth 2026-09-01. If
report #4 still shows GATE-CLOSED for an asset (and the p221b retrain has
not produced a promotable basis — that half is judged by a human), that
asset's trend injection comes off via `trend_assets` in the live profile.

WHAT THIS DOES: reads the weekly slope reports from the evidence directory,
counts consecutive GATE-CLOSED verdicts per asset, and states the tripwire
status loudly. It NEVER edits config or touches trading — firing the
actuator is a recorded human step (P141: deactivation of a live behavior is
operator-watched, even when a tripwire demands it). Exit codes: 0 = not yet
fired; 3 = FIRED for at least one asset (grep-able in the cron log);
2 = refusal (no reports — the tripwire cannot be evaluated, which must
never read as "not fired", P199).

Runs from cron Mondays 06:25 UTC, right after the calibrator (P235).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import date, datetime
from pathlib import Path

TRIPWIRE_DATE = date(2026, 9, 1)
REPORTS_REQUIRED = 4
ASSETS = ("BTC", "ETH", "SOL")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports-dir",
                    default="/opt/hmats/data/evidence_reports")
    ap.add_argument("--today", default=None,
                    help="override for tests (YYYY-MM-DD)")
    args = ap.parse_args()
    today = (datetime.strptime(args.today, "%Y-%m-%d").date()
             if args.today else date.today())

    files = sorted(glob.glob(str(Path(args.reports_dir) / "slope_*.json")))
    if not files:
        print(f"TRIPWIRE CANNOT BE EVALUATED: no slope_*.json under "
              f"{args.reports_dir} — the weekly calibrator has not written "
              f"there yet (first cron run 2026-08-11). 'No reports' is NOT "
              f"'not fired'.", file=sys.stderr)
        return 2

    # One report per calendar day (re-runs same day supersede), newest last.
    by_day: dict = {}
    for f in files:
        try:
            rep = json.loads(Path(f).read_text(encoding="utf-8"))
            day = rep["generated"][:10]
            by_day[day] = rep
        except Exception:
            print(f"  note: unreadable report skipped: {f}", file=sys.stderr)
    days = sorted(by_day)
    window = days[-REPORTS_REQUIRED:]
    print(f"P237 tripwire status — {len(days)} report day(s), judging the "
          f"last {len(window)}: {window} (need {REPORTS_REQUIRED}; "
          f"date gate {TRIPWIRE_DATE})")

    fired_any = False
    for a in ASSETS:
        closed = 0
        nodata = 0
        for d in window:
            rep = by_day[d]
            # [P253] Only horizons that RENDERED a verdict count. An
            # INSUFFICIENT/DEGENERATE horizon has no vs_threshold key, and
            # the old `h.get("vs_threshold", "")` turned that absence into ""
            # -> "not TRADEABLE" -> the day counted as GATE-CLOSED. "Not
            # enough data" must never read as "the gate closed" — that is the
            # P199 refusal principle, and this module's own docstring states
            # it (a tripwire that can fire on missing data deactivates a live
            # asset on an outage, not on evidence).
            verdicts = [
                v for v in (
                    h.get("vs_threshold")
                    for h in rep.get("assets", {}).get(a, {}).values()
                    if isinstance(h, dict)
                ) if v
            ]
            if not verdicts:
                nodata += 1
                continue
            # GATE-CLOSED for the asset = no rendered horizon says TRADEABLE
            if not any("TRADEABLE" in v for v in verdicts):
                closed += 1
        fired = (len(window) >= REPORTS_REQUIRED
                 and closed >= REPORTS_REQUIRED
                 and today >= TRIPWIRE_DATE)
        status = ("FIRED" if fired else
                  f"armed {closed}/{len(window)} closed"
                  + (f" ({nodata} no-data day(s) NOT counted)" if nodata else ""))
        print(f"  {a}: {status}")
        if fired:
            fired_any = True
            print(f"    -> P237 TRIPWIRE FIRED for {a}: {REPORTS_REQUIRED} "
                  f"consecutive GATE-CLOSED weekly reports past "
                  f"{TRIPWIRE_DATE}. ACTION (human, recorded): unless the "
                  f"p221b retrain produced a promotable basis, remove "
                  f"'{a}' from trend_assets in configs/live_high_risk.json "
                  f"and redeploy. This checker never edits config itself.")
    return 3 if fired_any else 0


if __name__ == "__main__":
    sys.exit(main())

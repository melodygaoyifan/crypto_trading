"""[P237] The calibration tripwire — [P299] SUPERSEDED: it now REPORTS only.

WHY THE PRESCRIPTION WAS RETIRED (P299, 2026-08-18). Two things changed
under it:

  1. ITS ACTUATOR NO LONGER TARGETS THE DECIDER. It removes an asset from
     `trend_assets`, i.e. the TREND injection — but trend has not held the
     DECIDE seat since 2026-08-17 (whale_seat_mode=enforce, P293j; and
     regimebook_mode=enforce, P298). Firing it today removes the FALLBACK
     that covers whale's silent ticks (~46/57/88% of ticks on BTC/ETH/SOL),
     handing those ticks to Best-of-N — whose strategy weights are modulated
     by [SENT-SWITCH], an F&G rule that has never been validated and fires on
     ~47% of days in this regime (P293i). Removing a measured-weak signal
     into an unmeasured one is not de-risking.

  2. A BETTER INSTRUMENT EXISTS. The P295 seat controller decides the DECIDE
     slot by COMPARISON across candidates, with `flat` as one candidate among
     several rather than the only reachable outcome of a one-variable
     threshold. It reached exactly this verdict on 2026-08-18 (both scoreable
     candidates negative on both horizons -> flat wins by comparison).

WHAT SURVIVES: the evidence. A GATE-CLOSED streak is a real statement about
the calibrated alpha model and it feeds the seat decision. So this keeps
reading the weekly slope reports and counting consecutive GATE-CLOSED
verdicts per asset — it just no longer tells anyone to edit config, and it
no longer exits 3. Two tools competing to prescribe the same config edit is
how a contradiction reaches a live account.

Exit codes: 0 = reported (whatever the streak); 2 = refusal (no reports —
which must never read as "not fired", P199). Exit 3 is retired.

THE ORIGINAL DECISION, kept for the record (P237, operator-delegated
2026-08-08): trading continues on the
asserted alpha constants ONLY through the 4-report dual-log window — weekly
Monday `slope_calibrator` reports, first 2026-08-11, fourth 2026-09-01. If
report #4 still shows GATE-CLOSED for an asset (and the p221b retrain has
not produced a promotable basis — that half is judged by a human), that
asset's trend injection comes off via `trend_assets` in the live profile.

WHAT THIS DOES: reads the weekly slope reports from the evidence directory,
counts consecutive GATE-CLOSED verdicts per asset, and states the tripwire
status loudly. It NEVER edits config or touches trading (P141) — and since
P299 it never prescribes an edit either; see the header.

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
    # [P265] The P237 criterion is four consecutive WEEKLY (Monday-cron)
    # reports. The old window took the last four report DAYS, so four ad-hoc
    # calibrator runs during one debugging week satisfied it — the instrument
    # could demand the trend-injection removal on a compressed evidence
    # window. Judge only reports spaced >= 5 days apart (weekly cadence with
    # rerun slack), newest backwards; same-week reruns collapse into one slot.
    _MIN_SPACING_DAYS = 5
    from datetime import date as _date
    window: list = []
    for d in reversed(days):
        if not window:
            window.append(d)
        else:
            try:
                gap = (_date.fromisoformat(window[-1])
                       - _date.fromisoformat(d)).days
            except ValueError:  # noqa: silent-swallow — malformed day key in a report filename; the spacing filter just skips it
                continue
            if gap >= _MIN_SPACING_DAYS:
                window.append(d)
        if len(window) == REPORTS_REQUIRED:
            break
    window = sorted(window)
    print(f"P237 tripwire status — {len(days)} report day(s), judging "
          f"{len(window)} weekly-spaced (>= {_MIN_SPACING_DAYS}d apart): "
          f"{window} (need {REPORTS_REQUIRED}; date gate {TRIPWIRE_DATE})")

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
            # [P299] SUPERSEDED — this no longer prescribes an action.
            #
            # Two things changed under it. (1) Its actuator targets
            # `trend_assets`, i.e. the TREND injection — but trend has not
            # held the DECIDE seat since 2026-08-17 (whale_seat_mode=enforce,
            # P293j; regimebook_mode=enforce, P298). Removing trend now
            # removes the FALLBACK that covers whale's silent ticks
            # (~46/57/88% on BTC/ETH/SOL), handing them to Best-of-N, whose
            # weights are modulated by the never-validated [SENT-SWITCH]
            # (P293i). That is the opposite of de-risking. (2) The P295 seat
            # controller reaches this same verdict BY COMPARISON, with
            # `flat` as one candidate among several rather than the only
            # reachable outcome of a one-variable threshold — and it did
            # exactly that on 2026-08-18.
            #
            # The EVIDENCE half is kept: a GATE-CLOSED streak is a real
            # statement about the calibrated alpha model, and it feeds the
            # seat decision. Only the PRESCRIPTION is retired.
            print(f"    -> {a}: {REPORTS_REQUIRED} consecutive GATE-CLOSED "
                  f"weekly reports past {TRIPWIRE_DATE}. This is EVIDENCE, "
                  f"not an instruction: the P237 prescription (remove '{a}' "
                  f"from trend_assets) is SUPERSEDED by the P295 seat "
                  f"controller, which decides the DECIDE slot by comparison "
                  f"across candidates. Run scripts/seat_check.py for the "
                  f"action; do NOT edit trend_assets from this line.")
    if fired_any:
        print("\n  NOTE: exit code is 0. This checker no longer FIRES — it "
              "reports. The seat decision lives in scripts/seat_check.py "
              "(Mondays 06:40 UTC), which can reach 'flat' as one outcome "
              "among several instead of only ever removing a signal.")
    # [P299] Always 0. A non-zero exit here meant "act on one variable"; the
    # seat controller owns that decision now and signals it with ITS exit
    # code. Two tools competing to prescribe the same config edit is how a
    # contradiction reaches a live account.
    return 0


if __name__ == "__main__":
    sys.exit(main())

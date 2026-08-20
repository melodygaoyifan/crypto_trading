"""[P327] Report every registered calibration's age, producer and source.

A registry nobody reads is documentation, and documentation is what this
replaces. This is the instrument (P230): it runs weekly, and a constant that
has aged past the horizon its own author chose becomes a line in the evidence
log instead of a discovery someone makes a year later (P315).

    python -X utf8 scripts/calibration_check.py

Exit codes are deliberately distinct, because "I could not read it" must never
read as "it is fine" (P159/P199/P213):

    0  every registered calibration resolves and is within its horizon
    2  REFUSED — an entry does not resolve (renamed/deleted constant, or a
       module that will not import). This is a broken contract, not staleness.
    3  at least one calibration is STALE — re-run its producer, and read its
       revision_rule before changing anything.

It never edits a constant. Re-deriving a measurement is a decision with a
direction (P167), and several entries may only be moved one way on evidence.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--today", default=None,
                    help="ISO date override (testing); defaults to today UTC")
    args = ap.parse_args(argv)

    try:
        from core.calibration_registry import REGISTRY, resolve
    except Exception as e:  # noqa: silent-swallow — reported, then refused
        print(f"REFUSING: cannot import the calibration registry: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    today = (_dt.date.fromisoformat(args.today) if args.today
             else _dt.datetime.now(_dt.timezone.utc).date())

    if not REGISTRY:
        print("REFUSING: the registry is EMPTY. An empty registry reports "
              "clean, which is indistinguishable from covering everything "
              "(P174).", file=sys.stderr)
        return 2

    unresolved, stale = [], []
    print(f"{'symbol':<52}{'measured':>12}{'age':>6}{'horizon':>9}  state")
    for cal in REGISTRY:
        try:
            resolve(cal)
        except Exception as e:
            unresolved.append((cal, f"{type(e).__name__}: {e}"))
            state = "UNRESOLVED"
        else:
            state = "STALE" if cal.is_stale(today) else "ok"
            if state == "STALE":
                stale.append(cal)
        print(f"{cal.symbol:<52}{cal.measured_on:>12}"
              f"{cal.age_days(today):>5}d{cal.staleness_days:>8}d  {state}")

    if unresolved:
        print("\nREFUSING TO REPORT — a registry entry describes a symbol that "
              "no longer exists. That is a broken contract, not staleness:",
              file=sys.stderr)
        for cal, why in unresolved:
            print(f"  {cal.symbol}: {why}", file=sys.stderr)
        return 2

    if stale:
        print(f"\n{len(stale)} calibration(s) past their horizon. Re-derive "
              f"with the producer, and read revision_rule FIRST — several may "
              f"only move one way on evidence (P167):")
        for cal in stale:
            print(f"\n  {cal.symbol}  ({cal.age_days(today)}d old, horizon "
                  f"{cal.staleness_days}d)")
            print(f"    producer : {cal.producer}")
            print(f"    source   : {cal.source}")
            print(f"    rule     : {cal.revision_rule}")
        return 3

    print(f"\nOK — {len(REGISTRY)} calibration(s) resolve and are within "
          f"their horizons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

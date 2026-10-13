#!/usr/bin/env python3
"""[P423] Standing-decision check — the weekly alarm for the registry.

Reads analytics/decisions/pending_decisions.json and prints every pending
decision with its deadline. Exit codes, distinct on purpose:
  0  nothing overdue (pending items listed with days remaining)
  2  REFUSAL — the registry is missing or unparseable ("I could not read
     it" must never read as "nothing pending", P2/P199)
  3  at least one decision is OVERDUE — the operator is paged; the
     default (keep state + escalate) is already in force

It NEVER edits config and never acts: deciding is a human step (P141). What
it owes the operator is that forgetting is impossible — the 19 September
reads went unread for a month because their dates lived only in a doc
(P421). Stdlib-only; runs in-container (the registry ships with analytics/)
or operator-local. Install: Monday 06:52 UTC beside the other evidence
crons (P235, crontab-by-file).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.decisions.registry import (  # noqa: E402
    RegistryUnreadable, days_overdue, load_registry, overdue, pending,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default=None,
                    help="path to pending_decisions.json (default: the repo's)")
    ap.add_argument("--today", default=None,
                    help="YYYY-MM-DD override (tests/dry runs)")
    args = ap.parse_args(argv)
    today = (datetime.strptime(args.today, "%Y-%m-%d").date() if args.today
             else datetime.now(timezone.utc).date())
    try:
        reg = load_registry(Path(args.registry) if args.registry else None)
    except RegistryUnreadable as e:
        print(f"[decision_check] REFUSING: registry unreadable — {e}. "
              f"This is NOT 'nothing pending'.", file=sys.stderr)
        return 2
    pend = pending(reg)
    late = overdue(reg, today)
    print(f"[decision_check] {today}: {len(pend)} pending, {len(late)} overdue "
          f"(of {len(reg)} recorded)")
    for d in sorted(pend, key=lambda x: x["deadline"]):
        left = -days_overdue(d, today)
        tag = f"OVERDUE +{-left}d" if left < 0 else f"due in {left}d"
        print(f"  [{tag:>14}] {d['id']}: {d.get('title', '')}")
        if left < 0:
            print(f"                 default in force: {d.get('default', '')}")
            for o in d.get("options", []):
                print(f"                 option: {o}")
    if late:
        print(f"\n[decision_check] {len(late)} DECISION(S) OVERDUE — the live state "
              f"is unchanged by construction (never auto-de-risk, P141). Decide "
              f"by editing analytics/decisions/pending_decisions.json "
              f"(status=decided, decided_on, decision) in the SAME commit as "
              f"the config change it names.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

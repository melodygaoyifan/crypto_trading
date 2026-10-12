#!/usr/bin/env python3
"""[P421] The calbasis re-read checker — the pre-committed 2026-11-10 criterion
as a MECHANISM instead of a date in prose (P333: a decision procedure that
depends on someone remembering quietly becomes "never"; the 19 September reads
sat unread for a month for exactly that reason).

THE CANDIDATE: `calbasis` (CDE-native calendar-basis slope z, P277) — the one
near-miss of the 2026-10-11 September reads: pooled n=1374, IC +0.200/+0.265/
+0.307 at 4/12/24 bars with |t| 3.70/2.79/2.23 (significant at ALL horizons),
cost-clearing at h12 (136 vs 58bps) and h24 (214 vs 58), blocked ONLY by the
h4 edge (53.4 < 58.1) under the every-horizon rule.

THE PRE-COMMITTED CRITERION (fixed 2026-10-12, BEFORE any further number
exists — P332; do not reinterpret it after seeing the re-read):
  A. the scorer's own verdict is PROMOTE (all horizons pass the P166 gate)
       -> exit 3: draft the lab-ladder P-entry toward a seat. Promotion
          remains a human config change (P141).
  B. h12 AND h24 EACH show ic > 0, |t| >= 2.0 and edge_bps >= required_bps
     while h4 stays sub-bar
       -> exit 3: the SLOW-EXPRESSION (12-24 bar cadence) variant ledger is
          the sanctioned next build, its own P-entry. Nothing re-arms.
  C. anything else -> exit 0: keep accruing; nothing is due.
Exit codes (P185 discipline): 0 = not due / keep accruing; 2 = REFUSAL (no or
unusable evidence — "no data" must never read as a verdict, P199); 3 = a
criterion fired and the printed action is owed a human decision.

WHERE THIS RUNS: operator-local (P213) — the shadow-IC reports it reads are
produced by `python -X utf8 scripts/september_check.py --window-days <N>`
(or compute_shadow_ic --pool-assets directly) on the operator machine. The
report must span the ledger's FULL accrual (ledger start 2026-08-16): a
trailing window shorter than that misses most of the evidence — the exact
mistake the 2026-10-11 late read almost made (P421 mitigation c).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

READ_DATE = dt.date(2026, 11, 10)
LEDGER_START = dt.date(2026, 8, 16)
WINDOW_SLACK_DAYS = 3
REPORTS_GLOB = os.path.join("analytics", "shadow_ic", "reports", "shadow_ic_*.json")
STRATEGY = "calbasis"


def find_record(report: dict):
    for rec in report.get("per_strategy", []):
        if rec.get("strategy") == STRATEGY and rec.get("asset") == "POOLED":
            return rec
    return None


def horizon_passes(h: dict) -> bool:
    """One horizon's full P166 bar: positive IC, |t| >= 2, edge >= required."""
    try:
        return (float(h["ic"]) > 0.0 and abs(float(h["t_stat"])) >= 2.0
                and float(h["edge_bps"]) >= float(h["required_bps"]))
    except (KeyError, TypeError, ValueError):
        return False


def decide(rec: dict) -> tuple[int, str]:
    """(exit_code, action line) from a scoreable calbasis POOLED record.
    Pure so the criterion is testable without a report on disk (P206)."""
    if str(rec.get("verdict", "")).upper() == "PROMOTE":
        return 3, ("CRITERION A MET — the scorer's own P166 gate passes at every "
                   "horizon. Next action: draft the calbasis lab-ladder P-entry "
                   "toward a seat. Promotion is a human config change (P141); "
                   "nothing is armed by this line.")
    ph = rec.get("per_horizon", {}) or {}
    if horizon_passes(ph.get("12", {})) and horizon_passes(ph.get("24", {})):
        return 3, ("CRITERION B MET — h12 and h24 each clear the full bar while "
                   "h4 stays sub-bar. Next action: build the SLOW-EXPRESSION "
                   "(12-24 bar cadence) calbasis variant ledger, its own "
                   "P-entry. Nothing re-arms today.")
    return 0, ("keep accruing — neither criterion met; nothing is due. "
               "Do NOT reinterpret the criterion after this read (P332).")


def fmt_table(rec: dict) -> str:
    lines = []
    for h in ("4", "12", "24"):
        d = (rec.get("per_horizon") or {}).get(h)
        if not d:
            lines.append(f"  h={h:<3} (no populated horizon)")
            continue
        lines.append(
            f"  h={h:<3} ic={d.get('ic', float('nan')):+.4f}  t={d.get('t_stat', float('nan')):+.2f}  "
            f"edge={d.get('edge_bps', float('nan')):7.2f}bps vs need {d.get('required_bps', float('nan')):.2f}  "
            f"n_eff={d.get('n_eff', '?')}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", default=None,
                    help="shadow_ic report JSON from a --pool-assets run; "
                         "default = newest under analytics/shadow_ic/reports/")
    ap.add_argument("--today", default=None,
                    help="override the date gate (tests only), YYYY-MM-DD")
    args = ap.parse_args()

    today = (dt.date.fromisoformat(args.today) if args.today else dt.date.today())
    if today < READ_DATE:
        print(f"calbasis re-read not due: {(READ_DATE - today).days} day(s) left "
              f"(read date {READ_DATE}). The ledger keeps accruing; nothing to do.")
        return 0

    path = args.report
    if path is None:
        cands = sorted(glob.glob(REPORTS_GLOB), key=os.path.getmtime)
        path = cands[-1] if cands else None
    if not path or not os.path.exists(path):
        print("REFUSING: no shadow_ic report found. Produce one first:\n"
              "    python -X utf8 scripts/september_check.py --window-days "
              f"{(today - LEDGER_START).days + WINDOW_SLACK_DAYS}\n"
              "A verdict from missing evidence would be a guess wearing a "
              "measurement's name (P199).", file=sys.stderr)
        return 2
    try:
        report = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"REFUSING: report unreadable ({type(exc).__name__}: {exc}) — {path}",
              file=sys.stderr)
        return 2

    need_days = (today - LEDGER_START).days
    win = int(report.get("window_days") or 0)
    if win + WINDOW_SLACK_DAYS < need_days:
        print(f"REFUSING: report window {win}d does not span the calbasis ledger "
              f"({need_days}d since {LEDGER_START}) — a trailing window misses "
              f"most of the evidence (P421 mitigation c). Re-run:\n"
              f"    python -X utf8 scripts/september_check.py --window-days "
              f"{need_days + WINDOW_SLACK_DAYS}", file=sys.stderr)
        return 2

    rec = find_record(report)
    if rec is None or not any((rec.get("n_per_horizon") or {}).values()):
        print(f"REFUSING: no scoreable '{STRATEGY}' POOLED record in {path} — "
              "run the POOLED scorer (september_check runs both) and check the "
              "ledger pull. 'No record' is a wiring question, never a verdict "
              "(P199/P264).", file=sys.stderr)
        return 2

    code, action = decide(rec)
    print(f"calbasis re-read ({path}, window {win}d, verdict {rec.get('verdict')}):")
    print(fmt_table(rec))
    print(action)
    return code


if __name__ == "__main__":
    sys.exit(main())

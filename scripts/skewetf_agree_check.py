#!/usr/bin/env python3
"""[P425] The skew+ETF ETH agree-gate re-read checker — the pre-committed
2026-12-12 criterion as a MECHANISM (P333/P421c pattern: a decision procedure
that depends on someone remembering quietly becomes "never").

THE CANDIDATE: arming the ETH skew seat behind the ETF AGREE-GATE
(`skew_seat_agree_assets: ["ETH"]` + `skew_seat_assets: ["BTC","ETH"]`).
P407i measured skew+ETF complementary on the 1.7y overlap (skew alone NEGATIVE
OOS on ETH -223/RT; agree-gated POSITIVE on both assets), and the P421b
runtime-series probe found ETH's skew era-fragile (2025 sub-era -188.6,
kill-class) — the agree-gate is exactly the form that rescued the ETH case in
the lab. The live gate (main.py `skew_seat_agree_ok`) is decided by the SAME
`combo_directions` the `skewetf_agree` ledger records (P172), so the forward
read scores the thing that would go live.

THE LEDGER: `skewetf_agree` ETH rows are REPLICA-FED (true skew arm) only from
2026-10-13 13:16 UTC (the P424 bank-feed deploy); rows before that carry the
stale post-09-10 skew arm (P421e) and must not be scored. A report whose window
does not span LEDGER_START is REFUSED.

THE PRE-COMMITTED CRITERION (fixed 2026-10-13, BEFORE any number exists — P332;
do not reinterpret it after seeing the read):
  A. the scorer's own verdict on `skewetf_agree` ETH is PROMOTE (every horizon
     passes the P166 cost-aware gate: ic > 0, |t| >= 2, edge >= required)
       -> exit 3: the printed config change is owed a human decision
          (pending_decisions.json: skewetf_eth_agree_read). Nothing is armed
          by this script (P141).
  B. anything else -> exit 0: keep accruing; ETH stays parked. The skew-solo
     ETH record is printed beside it as CONTEXT (it is NOT a criterion: a
     solo PROMOTE would re-open the P421b era-fragility question, not this one).
Exit codes (P185): 0 = not due / keep accruing; 2 = REFUSAL (no or unusable
evidence — "no data" must never read as a verdict, P199); 3 = criterion fired.

WHERE THIS RUNS: in-container, Monday 06:56 UTC (scripts/ops/hmats.crontab),
on a PER-ASSET (not pooled) `--prefixes skewetf` report priced on Kraken
public OHLC (P423 --allow-kraken-ohlcv). Operator-local works the same way.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

READ_DATE = dt.date(2026, 12, 12)
LEDGER_START = dt.date(2026, 10, 13)
WINDOW_SLACK_DAYS = 3
REPORTS_GLOB = os.path.join("analytics", "shadow_ic", "reports", "shadow_ic_*.json")
STRATEGY = "skewetf_agree"
CONTEXT_STRATEGY = "skewetf_skew"
ASSET = "ETH"
ACTION_CONFIG = ('configs/live_high_risk.json: "skew_seat_agree_assets": ["ETH"] '
                 'AND "skew_seat_assets": ["BTC", "ETH"] (one commit with the '
                 'registry decision, P423)')


def find_record(report: dict, strategy: str = STRATEGY, asset: str = ASSET):
    for rec in report.get("per_strategy", []):
        if rec.get("strategy") == strategy and rec.get("asset") == asset:
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
    """(exit_code, action line) from a scoreable skewetf_agree ETH record.
    Pure so the criterion is testable without a report on disk (P206)."""
    if str(rec.get("verdict", "")).upper() == "PROMOTE":
        return 3, ("CRITERION A MET — the scorer's own P166 gate passes at every "
                   "horizon for skewetf_agree ETH. Next action (a HUMAN config "
                   f"change, P141; nothing is armed by this line): {ACTION_CONFIG}. "
                   "Record the decision in pending_decisions.json "
                   "(skewetf_eth_agree_read) in the same commit.")
    return 0, ("keep accruing — the agree-gated ETH record does not pass the "
               "P166 gate at every horizon; ETH stays PARKED. Do NOT reinterpret "
               "the criterion after this read (P332).")


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
                    help="shadow_ic report JSON from a PER-ASSET --prefixes skewetf run; "
                         "default = newest under analytics/shadow_ic/reports/")
    ap.add_argument("--today", default=None,
                    help="override the date gate (tests only), YYYY-MM-DD")
    args = ap.parse_args()

    today = (dt.date.fromisoformat(args.today) if args.today else dt.date.today())
    if today < READ_DATE:
        print(f"skewetf ETH agree-gate re-read not due: {(READ_DATE - today).days} day(s) left "
              f"(read date {READ_DATE}). The ledger keeps accruing; ETH stays parked.")
        return 0

    path = args.report
    if path is None:
        cands = sorted(glob.glob(REPORTS_GLOB), key=os.path.getmtime)
        path = cands[-1] if cands else None
    if not path or not os.path.exists(path):
        print("REFUSING: no shadow_ic report found. Produce one first:\n"
              "    python -X utf8 analytics/shadow_ic/compute_shadow_ic.py "
              f"--prefixes skewetf --window-days {(today - LEDGER_START).days + WINDOW_SLACK_DAYS} "
              "--allow-kraken-ohlcv\n"
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
        print(f"REFUSING: report window {win}d does not span the replica-fed skewetf "
              f"ledger ({need_days}d since {LEDGER_START}) — a trailing window misses "
              f"the evidence (P421 mitigation c). Re-run with --window-days "
              f"{need_days + WINDOW_SLACK_DAYS}.", file=sys.stderr)
        return 2

    rec = find_record(report)
    if rec is None or not any((rec.get("n_per_horizon") or {}).values()):
        print(f"REFUSING: no scoreable '{STRATEGY}' {ASSET} record in {path} — run the "
              "PER-ASSET scorer (--prefixes skewetf, no --pool-assets) and check "
              "the ledger. 'No record' is a wiring question, never a verdict "
              "(P199/P264).", file=sys.stderr)
        return 2

    code, action = decide(rec)
    print(f"skewetf ETH agree-gate re-read ({path}, window {win}d, "
          f"verdict {rec.get('verdict')}):")
    print(fmt_table(rec))
    ctx = find_record(report, CONTEXT_STRATEGY)
    if ctx is not None:
        print(f"  context — {CONTEXT_STRATEGY} {ASSET} (NOT a criterion): "
              f"verdict {ctx.get('verdict')}")
        print(fmt_table(ctx))
    print(action)
    return code


if __name__ == "__main__":
    sys.exit(main())

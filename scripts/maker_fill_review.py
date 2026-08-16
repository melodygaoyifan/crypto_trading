#!/usr/bin/env python3
"""[P282] Maker fill-rate review — the standing instrument for the P278
re-entry condition.

P278 made the entire supervised-revival path conditional on the MEASURED
effective round-trip cost: maker fills pay ~0bps/leg, taker fallbacks pay
~3bps/leg + slip. The evidence accrues in `[COINBASE-MAKER]` lines in the
persistent server log — but until this script, reading it was "a grep
someone must remember", exactly the decision-bar-without-an-instrument
shape P230 warned about (the fresh-mind review's #1 demand).

Counts, from the persistent log (docker logs are wiped on recreation, P195):
  maker-filled   : "post-only left the book within the window"
  taker-fallback : "cancelled, crossing the remainder"
                   + "post-only rejected" (crossed immediately)
  cancel-failed  : the refuse-to-cross safety path (neither fee outcome)

Effective per-leg fee = (m*MAKER_LEG + t*TAKER_LEG) / (m+t); effective RT =
2x that. VERDICT against the P278 bar (effective RT <= ~3bps unlocks the
supervised revival at measured cost). Refuses (exit 2) below MIN_N legs —
a two-order sample is an anecdote, not a fill rate (P199: no-data must
never read as a verdict).

Run (operator machine; pulls the log over ssh):
    python -X utf8 scripts/maker_fill_review.py
    python -X utf8 scripts/maker_fill_review.py --log-file <local copy>
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

MAKER_LEG_BPS = 0.5     # 0bps fee + residual micro-slip at the touch
TAKER_LEG_BPS = 3.0     # CDE taker fee (slip on a crossed nano order ~0-1
                        # extra; kept at the fee floor = the FAVORABLE
                        # reading for taker, so the verdict cannot be
                        # flattered toward "maker unlocked")
P278_RT_BAR = 3.0
MIN_N = 20

PATTERNS = {
    "maker_filled": re.compile(r"post-only left the book within the window"),
    "taker_after_timeout": re.compile(r"cancelled, crossing the remainder"),
    "taker_immediate": re.compile(r"post-only rejected"),
    "cancel_failed": re.compile(r"timeout AND cancel FAILED"),
}


def read_lines(log_file: str | None) -> list:
    if log_file:
        return open(log_file, encoding="utf-8", errors="replace").readlines()
    out = subprocess.run(
        ['ssh', 'hmats',
         'grep "COINBASE-MAKER" '
         '/var/lib/docker/volumes/hmats-logs/_data/hmats.log || true'],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        print(f"REFUSING: ssh log pull failed (rc={out.returncode}) — "
              f"no data must never read as a verdict (P199)")
        sys.exit(2)
    return out.stdout.splitlines()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()
    lines = read_lines(args.log_file)
    counts = {k: 0 for k in PATTERNS}
    for ln in lines:
        for k, pat in PATTERNS.items():
            if pat.search(ln):
                counts[k] += 1
                break
    m = counts["maker_filled"]
    t = counts["taker_after_timeout"] + counts["taker_immediate"]
    n = m + t
    print(f"[MAKER-REVIEW] maker-filled={m}  taker-fallback={t} "
          f"(timeout {counts['taker_after_timeout']} / immediate "
          f"{counts['taker_immediate']})  cancel-failed="
          f"{counts['cancel_failed']}")
    if n < MIN_N:
        print(f"REFUSING a verdict: only {n} resolved maker attempts "
              f"(< {MIN_N}) — the P278 condition stays UNMEASURED; keep "
              f"accruing. (Exit 2 = insufficient data, not a fail.)")
        return 2
    leg = (m * MAKER_LEG_BPS + t * TAKER_LEG_BPS) / n
    rt = 2 * leg
    fill_rate = m / n
    verdict = "UNLOCKED" if rt <= P278_RT_BAR else "NOT unlocked"
    print(f"[MAKER-REVIEW] n={n} legs | fill rate {fill_rate:.0%} | "
          f"effective leg {leg:.2f}bps | effective RT {rt:.2f}bps "
          f"vs P278 bar {P278_RT_BAR}bps -> supervised revival {verdict}")
    if counts["cancel_failed"]:
        print(f"  NOTE: {counts['cancel_failed']} cancel-failed refusals — "
              f"review [COINBASE-MAKER] warnings; these are safety events, "
              f"not fee outcomes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

# [P287] Ordered dict + break-on-first-match: specific patterns MUST
# precede the generic "taker fallback" catch-all ("post-only rejected ...
# — taker fallback" would otherwise double-classify).
PATTERNS = {
    "maker_filled": re.compile(r"post-only left the book within the window"),
    "taker_after_timeout": re.compile(r"cancelled, crossing the remainder"),
    "taker_immediate": re.compile(r"post-only rejected"),
    # [P287] The attempt-error path (coinbase_sleeve.py "attempt error
    # (...) — taker fallback") is a REAL crossed taker leg the caller
    # executes; pre-P287 it matched no pattern and vanished from both
    # numerator and denominator — structurally flattering the verdict
    # toward UNLOCKED, against this file's own no-flattering claim.
    "taker_error_path": re.compile(r"attempt error .*taker fallback"),
    # [P287] The remaining "— taker fallback" emitters (no best_bid_ask /
    # empty book side) go through _maker_log_once, so the log DEDUPES
    # repeats — counting what appears is a known UNDERCOUNT of taker legs
    # (still strictly less flattering than dropping them entirely).
    "taker_fallback_other": re.compile(r"taker fallback"),
    "cancel_failed": re.compile(r"timeout AND cancel FAILED"),
    # ambiguous outcome (accepted post-only with no order_id, resolved via
    # reconcile) — reported, excluded from the fee arithmetic
    "unresolved_no_id": re.compile(r"carried no order_id"),
}


def read_lines(log_file: str | None) -> list:
    if log_file:
        # [P287] an unreadable local file is a refusal, not a traceback and
        # never "zero maker attempts" (P199)
        try:
            return open(log_file, encoding="utf-8",
                        errors="replace").readlines()
        except OSError as e:
            print(f"REFUSING: cannot read {log_file} ({e}) — an unreadable "
                  f"log must never read as a verdict (P199)")
            sys.exit(2)
    # [P287] The old `|| true` collapsed "file missing/unreadable" (grep
    # rc>=2) into "zero matching lines" (grep rc 1) — the exact no-data-
    # reads-as-verdict conflation this file's docstring cites P199 against.
    # grep contract: 0=matches, 1=no matches, >=2=error. Remote maps
    # error -> exit 4; no-match -> exit 0 with empty stdout (the MIN_N
    # refusal below handles that honestly). ssh transport failure = 255.
    out = subprocess.run(
        ['ssh', 'hmats',
         'grep "COINBASE-MAKER" '
         '/var/lib/docker/volumes/hmats-logs/_data/hmats.log; ec=$?; '
         'if [ $ec -ge 2 ]; then exit 4; fi; exit 0'],
        capture_output=True, text=True, timeout=120, encoding="utf-8")
    if out.returncode == 4:
        print("REFUSING: the persistent server log could not be read "
              "(grep error — file missing/renamed/unreadable at "
              "/var/lib/docker/volumes/hmats-logs/_data/hmats.log). This is "
              "NOT 'zero maker attempts' (P199).")
        sys.exit(2)
    if out.returncode != 0:
        print(f"REFUSING: ssh log pull failed (rc={out.returncode}) — "
              f"no data must never read as a verdict (P199)")
        sys.exit(2)
    return out.stdout.splitlines()



# ---------------------------------------------------------------------------
# [P375] LEDGER MODE (authoritative). The log-parse mode above rests on the
# REFUTED percentage fee model (0.5/3.0bps) and the P278 "RT<=3bps unlocks the
# supervised revival" bar, both VOIDED by P315/P374: CDE charges a FLAT
# ~$0.60/contract, so realized fees are ~9-14bps/leg for maker AND taker alike
# and maker-first saves ~1bps of fee. What actually matters — the realized
# maker FILL RATE and the realized SLIPPAGE it captures — is recorded directly
# in data/fill_quality.jsonl (P290). This mode reports it.
# ---------------------------------------------------------------------------
# [P420] Derived, never restated: the trio literals here silently EXCLUDED
# XRP/BNB fills (routed since P412b/c) from the very fee-revision rule they
# must feed — the P315 rule lowers an ASSUMED/PREVIEW fee only on >=20 filled
# legs, and XRP/BNB carry PREVIEW fees (core.cde_fees.CDE_FEE_PREVIEW).
def _roster():
    """(contract sizes, certified per-RT edges, assets whose fee is not yet
    fill-measured) from the single sources, with the old literals as a
    logged fallback so an import error degrades rather than fabricates."""
    sizes = {"BTC": 0.01, "ETH": 0.1, "SOL": 5.0}
    edges = {"BTC": 24.1, "ETH": 88.1, "SOL": 221.7}
    unfilled = {"SOL"}
    try:
        from core.cde_fees import _contract_sizes, CDE_FEE_ASSUMED, CDE_FEE_PREVIEW
        derived = _contract_sizes()
        if derived:
            sizes = dict(derived)
        unfilled = set(CDE_FEE_ASSUMED) | set(CDE_FEE_PREVIEW)
    except Exception as e:  # noqa: silent-swallow — logged; the trio fallback is the degraded state
        print(f"  WARNING: core.cde_fees unavailable ({type(e).__name__}: {e}) — "
              f"contract sizes fall back to the trio literals (XRP/BNB fills "
              f"will be SKIPPED, P420)")
    try:
        from core.seat_alpha import REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP
        edges = dict(REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP)
    except Exception as e:  # noqa: silent-swallow — logged; the trio fallback is the degraded state
        print(f"  WARNING: core.seat_alpha unavailable ({type(e).__name__}: {e}) — "
              f"certified edges fall back to the trio literals (P420)")
    return sizes, edges, unfilled


CONTRACT_SIZE, EDGE_RT_BPS, FEE_NOT_FILL_MEASURED = _roster()
SOL_REPRICE_FLOOR = 20   # P315 revision rule: LOWER a fee only on >=20 fills
REPRICE_FLOOR = SOL_REPRICE_FLOOR   # [P420] applies to every assumed/preview fee


def report_assets(rows):
    """[P420] Every asset the fee table knows OR the ledger carries, in a
    stable order — a fill for an asset the roster forgot must still be seen."""
    seen = [r.get("asset") for r in rows if r.get("asset")]
    order = list(CONTRACT_SIZE) + sorted({a for a in seen if a not in CONTRACT_SIZE})
    out = []
    for a in order:
        if a not in out:
            out.append(a)
    return out


def read_ledger(ledger_file):
    import json
    if ledger_file:
        try:
            raw = open(ledger_file, encoding="utf-8", errors="replace").read()
        except OSError as e:
            print(f"REFUSING: cannot read {ledger_file} ({e}) — an unreadable "
                  f"ledger must never read as a verdict (P199)")
            sys.exit(2)
    else:
        out = subprocess.run(
            ['ssh', 'hmats',
             'docker exec hmats-engine cat /opt/hmats/data/fill_quality.jsonl'],
            capture_output=True, text=True, timeout=120, encoding="utf-8")
        if out.returncode != 0:
            print(f"REFUSING: could not read the live fill_quality ledger "
                  f"(rc={out.returncode}) — no data must never read as a "
                  f"verdict (P199)")
            sys.exit(2)
        raw = out.stdout
    rows = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    return rows


def ledger_report(rows):
    # CDE-format only: the P290 hooks stamp `liquidity`; pre-P290 Kraken-era
    # rows have it None and are a DIFFERENT quantity — never mix them (P255).
    cde = [r for r in rows if r.get("liquidity") is not None]
    if not cde:
        print("REFUSING a verdict: 0 CDE-format fills in the ledger (only "
              "pre-P290 rows, if any). The sleeve has not logged a CDE fill "
              "here — keep accruing (P199). NOTE: the LIVE ledger is on the "
              "container volume; a stale LOCAL copy reads as zero (P255) — "
              "run with no --ledger-file to pull it over ssh.")
        return 2
    from collections import defaultdict
    nonurg = [r for r in cde if not r.get("urgent")]
    maker = [r for r in nonurg if r["liquidity"] == "maker"]
    f = (len(maker) / len(nonurg)) if nonurg else None

    slip = defaultdict(list)
    for r in cde:
        s = r.get("realized_slippage_bps")
        if isinstance(s, (int, float)):
            slip[r["liquidity"]].append(s)
    fee = defaultdict(list)
    for r in cde:
        fu, px, ct = r.get("fees_usd"), r.get("fill_avg_price"), r.get("contracts")
        cs = CONTRACT_SIZE.get(r.get("asset"))
        if fu and px and ct and cs:
            fee[r["asset"]].append(fu / (ct * cs * px) * 1e4)

    print(f"[MAKER-REVIEW/LEDGER] {len(cde)} CDE fills | non-urgent {len(nonurg)} "
          f"| maker {len(maker)}"
          + (f" -> maker fill rate f = {f:.2f}" if f is not None else ""))
    print("  realized slippage bps by liquidity (mean/n; +=paid worse than mid):")
    for k in ("maker", "taker_cross", "direct", "market_urgent"):  # [P383] urgent exits are MARKET
        if slip.get(k):
            v = slip[k]
            print(f"    {k:12s} {sum(v)/len(v):+7.2f}  n={len(v)}")
    assets = report_assets(cde)
    unsized = sorted({r.get("asset") for r in cde
                      if r.get("asset") and r.get("asset") not in CONTRACT_SIZE})
    if unsized:
        print(f"  WARNING: fills for {unsized} carry no contract size in the "
              f"roster — their fee bps cannot be computed (P420)")
    print("  realized FEE bps/leg by asset (mean/n) vs modelled:")
    for a in assets:
        if fee.get(a):
            v = fee[a]
            print(f"    {a}: {sum(v)/len(v):5.2f}  n={len(v)}")
    # [P420] re-pricing progress for EVERY assumed/preview fee, not SOL only
    # (P315: lower an assumed/preview fee only at >=20 filled legs).
    for a in sorted(FEE_NOT_FILL_MEASURED):
        n_a = len(fee.get(a, []))
        note = ("ELIGIBLE to re-price down" if n_a >= REPRICE_FLOOR
                else f"{n_a}/{REPRICE_FLOOR} fills toward re-pricing")
        print(f"  {a} fee is NOT fill-measured (assumed/preview); {note} "
              f"(P315 revision rule)")

    # realized RT cost per asset (fee + slippage, maker-weighted at observed f)
    # vs certified edge — the honest 'does it clear' read.
    print("  realized RT cost (fee+slip) vs certified edge, per asset:")
    for a in assets:
        if not fee.get(a):
            continue
        leg_fee = sum(fee[a]) / len(fee[a])
        a_slip = [r.get("realized_slippage_bps") for r in cde
                  if r.get("asset") == a and isinstance(r.get("realized_slippage_bps"), (int, float))]
        leg_slip = (sum(a_slip) / len(a_slip)) if a_slip else 0.0
        rt = 2 * (leg_fee + leg_slip)
        edge = EDGE_RT_BPS.get(a, 0.0)
        print(f"    {a}: RT {rt:6.2f}bps vs edge {edge:6.1f} -> "
              f"net {edge - rt:+6.1f}bps")
    print("  NOTE: the gate prices TAKER fee + FULL spread + a 0.75 haircut, "
          "which is MORE conservative than realized maker economics; it admits "
          "ETH/SOL and holds BTC out. Pricing realized maker cost would admit "
          "BTC, but BTC's forward edge is ~0 (P320c/P374) and that is a P141 "
          "loosening, not a bugfix.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", default=None,
                    help="LEGACY log-parse mode (refuted fee model; kept for "
                         "back-compat). Omit to use the authoritative ledger.")
    ap.add_argument("--ledger-file", default=None,
                    help="local copy of fill_quality.jsonl; omit to pull live")
    args = ap.parse_args()
    if args.log_file is None:
        # [P375] LEDGER mode is the default + authoritative reading.
        return ledger_report(read_ledger(args.ledger_file))
    print("[MAKER-REVIEW] LEGACY log-parse mode: fee model (0.5/3.0bps) and "
          "the P278 RT<=3bps bar are SUPERSEDED by P315/P374 (flat "
          "per-contract fees). Use ledger mode (omit --log-file) for the "
          "authoritative realized-cost reading.")
    lines = read_lines(args.log_file)
    counts = {k: 0 for k in PATTERNS}
    for ln in lines:
        for k, pat in PATTERNS.items():
            if pat.search(ln):
                counts[k] += 1
                break
    m = counts["maker_filled"]
    t = (counts["taker_after_timeout"] + counts["taker_immediate"]
         + counts["taker_error_path"] + counts["taker_fallback_other"])
    n = m + t
    print(f"[MAKER-REVIEW] maker-filled={m}  taker-fallback={t} "
          f"(timeout {counts['taker_after_timeout']} / immediate "
          f"{counts['taker_immediate']} / error-path "
          f"{counts['taker_error_path']} / other "
          f"{counts['taker_fallback_other']})  cancel-failed="
          f"{counts['cancel_failed']}  unresolved={counts['unresolved_no_id']}")
    if counts["taker_fallback_other"]:
        print(f"  NOTE: {counts['taker_fallback_other']} 'other' taker "
              f"fallbacks come from log-once emitters — a known UNDERCOUNT "
              f"of taker legs (dedup); the verdict is still biased the "
              f"unflattering way, never toward UNLOCKED.")
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

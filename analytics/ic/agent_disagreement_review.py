"""[P324] Does an advisor's DISAGREEMENT predict that the decider is wrong?

This generalises the one-off measurement P236 ran for `model_alpha` (quant
earned +24.9bps/tick when model_alpha agreed and -78.9bps/tick when it
disagreed, t=-3.42) into a standing instrument. That measurement is the ONLY
evidence any sleeve entry filter has ever been armed on, and it existed as a
hand-rolled analysis nobody could re-run — the P230 shape, where a bar without
an instrument becomes "whoever re-greps by hand next".

WHY IT EXISTS NOW. `coinbase_whale_filter_enforce` is armed and vetoes entries
whenever `whale` disagrees with the sleeve target. Its own call site says the
ledger is what earns the flip; it was armed one day after that ledger began,
and whale's own IC is 16h t=0.26 (noise, P293c). Worse, P298 SUBORDINATED whale
to the regimebook as a DIRECTION source — on the grounds that the book is
certified over six years and whale is not — and left whale able to VETO that
same book's entries. One signal, two contradictory precedence rules, decided in
the same entry. This measures the veto rather than arguing about it.

THE VERDICT RULE, PRE-COMMITTED BEFORE THE FIRST RUN (a criterion chosen after
seeing the number is selection, not evidence). An entry veto keyed on
<advisor> disagreeing with <decider> is EARNED iff, POOLED across assets:

  1. the DISAGREE bucket's mean signed forward return is NEGATIVE — the veto
     must be avoiding losses, not merely skipping weaker wins; and
  2. |t| >= 2.0 on that bucket, OVERLAP-CORRECTED (n_eff = n/h, P231 — an
     h-bar return sampled every bar overlaps h times, and the uncorrected t
     is inflated by ~sqrt(h); that artifact is exactly what made
     model_alpha's 16h t read 4.4 instead of ~2.2); and
  3. AGREE mean > DISAGREE mean, on the same horizon.

Anything else is NOT EARNED, and an armed veto with no measured basis should be
off (its default) until its ledger or a counterfactual earns it.

WHAT IS MEASURED. The decider's own realized edge: sign(decider_dir) x forward
return, in bps, bucketed by whether the advisor agreed. A veto is a claim about
that quantity, not about the advisor's standalone IC — a signal can be pure
noise standalone and still mark bad entries, which is why "IC is t=0.26" does
not settle the question either way.

HONEST LIMIT, reported in the output rather than buried: the identity of
`quant` CHANGED inside any long window (trend seat -> whale seat 2026-08-17 ->
regimebook seat 2026-08-18, P293j/P298). A pooled number over 90 days is
therefore mostly a measurement of the TREND seat's disagreement behaviour. The
per-era table exists so that cannot be read past (the P320c lesson: a claim
whose premise moved is not evidence for the current configuration).

WHERE IT RUNS: in-container, like its sibling — the attribution volume is
server-side (P213).

    docker exec hmats-engine python -X utf8 \\
        analytics/ic/agent_disagreement_review.py --advisor whale

Exit codes are distinct so a shell can tell them apart (P199/P213):
  0 = measured, NOT EARNED     3 = measured, EARNED
  2 = refused (no data / too few disagreements to judge)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# [P172] single source: the loader, the price fetcher, the horizon set and the
# asset roster are IMPORTED, never restated. A second copy of fetch_closes is
# how two reviewers start disagreeing about which bar a signal belongs to (and
# this one carries the P265 in-progress-candle drop).
from analytics.ic.agent_ic_review import (  # noqa: E402
    HORIZON_BARS,
    KRAKEN_PAIRS,
    fetch_closes,
    load_signal_records,
    resolve_log_dir,
)

MIN_DISAGREE_N = 30
T_BAR = 2.0

# Seat changes inside the attribution window. The decider's identity moved, so
# a pooled figure mixes regimes of AUTHORSHIP, not just of market.
SEAT_ERAS = (
    ("trend_seat", None, "2026-08-17"),
    ("whale_seat", "2026-08-17", "2026-08-18"),
    ("book_seat", "2026-08-18", None),
)


def _refuse(msg: str) -> None:
    print(f"REFUSING TO REPORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _era(iso_day: str) -> str:
    for name, lo, hi in SEAT_ERAS:
        if (lo is None or iso_day >= lo) and (hi is None or iso_day < hi):
            return name
    return "?"


def _stats(vals: list[float], horizon_bars: int) -> dict:
    """Mean signed-return in bps plus an OVERLAP-CORRECTED t (P231)."""
    n = len(vals)
    if n < 2:
        return {"n": n, "mean_bps": None, "t": None}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    n_eff = max(1.0, n / float(horizon_bars))
    t = None
    if sd > 0 and n_eff > 1:
        t = mean / (sd / math.sqrt(n_eff - 1.0))
    return {"n": n, "mean_bps": mean, "t": t, "n_eff": round(n_eff, 1)}



def classify_bucket(decider_dir, advisor_sig):
    """(bucket, decider_dir, advisor_dir, note); bucket None = skip.

    `note` is "", "bad_decider" or "bad_advisor". It exists so the caller
    can BATCH-REPORT unreadable records: this function stays pure (it must
    not log), but a dropped record that nobody counts is a silent failure,
    and a counter that cannot rise is worse than none (P174).

    Extracted so the rules below are tested by CALLING this, not by
    asserting a substring of it — a source pin proves the code was written,
    not that it runs (P234/P307b, and it is defeated by `if False and ...`).

    A FLAT decider is skipped: there is no entry for a veto to act on.
    A missing OR zero-direction advisor is SILENT, never DISAGREEING (P2) —
    otherwise every dark agent would read as a veto with perfect coverage,
    contradicting the live filter own fail-OPEN semantics.
    """
    try:
        dd = float(decider_dir or 0.0)
    except (TypeError, ValueError):  # noqa: silent-swallow — see `note`
        # UNKNOWN, not flat: dropped from the sample and reported by the
        # caller via the note, never bucketed (P2).
        return None, 0.0, 0.0, "bad_decider"
    if abs(dd) < 1e-9:
        return None, dd, 0.0, ""
    bad = False
    try:
        ad = float((advisor_sig or {}).get("direction", 0.0) or 0.0)
    except (TypeError, ValueError):  # noqa: silent-swallow — see `note`
        # Unreadable resolves to SILENT, which makes the veto NOT fire —
        # the same fail-OPEN direction the live filter takes on a dark
        # agent — and it is counted, so it cannot pass as a quiet agent.
        ad, bad = 0.0, True
    if advisor_sig is None or bad or abs(ad) < 1e-9:
        return "advisor_silent", dd, ad, ("bad_advisor" if bad else "")
    if (ad > 0) == (dd > 0):
        return "agree", dd, ad, ""
    return "disagree", dd, ad, ""


def _contrast(agree, disagree, horizon_bars):
    """Welch t on (agree - disagree), on OVERLAP-CORRECTED sample sizes.

    [P324b] REPORTED, NOT PRE-COMMITTED. The rule in the module docstring
    tests the DISAGREE bucket LEVEL against zero. That is not the quantity a
    filter exploits: a filter skips disagree entries and keeps agree ones, so
    its claim is a CONTRAST. This statistic is computed after the fact and is
    labelled as such rather than swapped in as the verdict — moving the test
    after seeing the number is the selection sin this repo exists to prevent
    (P296/P301: state the verdict as it fell, then report the better lens).
    """
    na, nd = len(agree), len(disagree)
    if na < 2 or nd < 2:
        return {"delta_bps": None, "t": None}
    ma, md = sum(agree) / na, sum(disagree) / nd
    va = sum((v - ma) ** 2 for v in agree) / (na - 1)
    vd = sum((v - md) ** 2 for v in disagree) / (nd - 1)
    ea = max(1.0, na / float(horizon_bars))
    ed = max(1.0, nd / float(horizon_bars))
    se = math.sqrt(va / ea + vd / ed)
    return {"delta_bps": ma - md, "t": ((ma - md) / se) if se > 0 else None}

def decide_verdict(pooled, horizons):
    """The PRE-COMMITTED rule (see module docstring), as a callable.

    Returns (verdict, blocker_reasons, earned_horizons). All three
    conditions are conjoined; each alone can block.
    """
    reasons, earned = [], []
    for h in horizons:
        dg = pooled.get(("disagree", h)) or {}
        ag = pooled.get(("agree", h)) or {}
        if dg.get("n", 0) < MIN_DISAGREE_N:
            reasons.append("h%d: only %d disagreements (need %d) — cannot judge" % (h, dg.get("n", 0), MIN_DISAGREE_N))
            continue
        c1 = dg.get("mean_bps") is not None and dg["mean_bps"] < 0
        c2 = dg.get("t") is not None and abs(dg["t"]) >= T_BAR
        c3 = (ag.get("mean_bps") is not None
              and dg.get("mean_bps") is not None
              and ag["mean_bps"] > dg["mean_bps"])
        if c1 and c2 and c3:
            earned.append(h)
            continue
        miss = []
        if not c1:
            miss.append("disagree mean not negative")
        if not c2:
            miss.append("|t| %.2f < %s" % (abs(dg["t"]), T_BAR)
                        if dg.get("t") is not None else "t unavailable")
        if not c3:
            miss.append("agree not better than disagree")
        reasons.append("h%d: " % h + "; ".join(miss))
    return ("EARNED" if earned else "NOT_EARNED"), reasons, earned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decider", default="quant")
    ap.add_argument("--advisor", default="whale")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    records = load_signal_records(resolve_log_dir(args.log_dir),
                                 args.window_days)
    if not records:
        _refuse(f"no attribution records inside the {args.window_days}d "
                f"window. 'No data' is not 'the veto is unearned'.")

    bars = {a: fetch_closes(a) for a in KRAKEN_PAIRS}

    # bucket -> horizon -> list of decider signed forward returns (bps)
    buckets: dict = defaultdict(lambda: defaultdict(list))
    per_asset: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_era: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen_decider = seen_advisor = 0
    bad_decider = bad_advisor = 0
    joined = 0

    for rec in records:
        sigs = {s.get("agent_name"): s for s in rec.get("signals", [])}
        if args.decider not in sigs:
            continue
        seen_decider += 1
        if args.advisor in sigs:
            seen_advisor += 1
        bucket, dd, _ad, note = classify_bucket(
            sigs[args.decider].get("direction", 0.0), sigs.get(args.advisor))
        if note == "bad_decider":
            bad_decider += 1
        elif note == "bad_advisor":
            bad_advisor += 1
        if bucket is None:
            continue  # flat or unreadable decider: no entry for a veto

        ts_list, closes = bars[rec["asset"]]
        i = bisect_right(ts_list, rec["_ts"]) - 1
        if i < 0:
            continue
        joined += 1
        era = _era(datetime.fromtimestamp(rec["_ts"], timezone.utc)
                   .strftime("%Y-%m-%d"))
        for h in HORIZON_BARS:
            if i + h >= len(closes):
                continue  # forward bar not closed — an honest gap, not a 0
            fwd = closes[i + h] / closes[i] - 1.0
            signed = (1.0 if dd > 0 else -1.0) * fwd * 1e4
            buckets[bucket][h].append(signed)
            per_asset[rec["asset"]][bucket][h].append(signed)
            per_era[era][bucket][h].append(signed)

    print(f"decider={args.decider} advisor={args.advisor} "
          f"window={args.window_days}d joined={joined} "
          f"decider_present={seen_decider} advisor_present={seen_advisor}")
    if bad_decider or bad_advisor:
        print(f"  note: unparseable directions dropped/neutralised — "
              f"decider={bad_decider} advisor={bad_advisor}", file=sys.stderr)
    if seen_advisor == 0:
        _refuse(f"'{args.advisor}' never appears in the attribution stream — "
                f"the advisor is not being recorded, so its disagreement "
                f"cannot be measured — a wiring gap rather than a verdict.")

    print("\nPOOLED — decider's own signed forward return, by bucket")
    print(f"{'bucket':<16}{'h':>3}{'n':>6}{'n_eff':>7}"
          f"{'mean_bps':>11}{'t':>8}")
    pooled = {}
    for bucket in ("agree", "disagree", "advisor_silent"):
        for h in HORIZON_BARS:
            st = _stats(buckets[bucket][h], h)
            pooled[(bucket, h)] = st
            mb = "-" if st["mean_bps"] is None else f"{st['mean_bps']:+.1f}"
            tt = "-" if st.get("t") is None else f"{st['t']:+.2f}"
            print(f"{bucket:<16}{h:>3}{st['n']:>6}"
                  f"{st.get('n_eff', '-'):>7}{mb:>11}{tt:>8}")

    print("")
    print("CONTRAST agree-minus-disagree (reported, NOT the pre-committed "
          "rule — see _contrast)")
    contrasts = {}
    for h in HORIZON_BARS:
        c = _contrast(buckets["agree"][h], buckets["disagree"][h], h)
        contrasts[h] = c
        db = "-" if c["delta_bps"] is None else "%+.1f" % c["delta_bps"]
        tt = "-" if c["t"] is None else "%+.2f" % c["t"]
        print("  h%d: delta=%sbps  t=%s" % (h, db, tt))
    print("\nPER ASSET (disagree bucket only)")
    for a in sorted(per_asset):
        cells = []
        for h in HORIZON_BARS:
            st = _stats(per_asset[a]["disagree"][h], h)
            mb = "-" if st["mean_bps"] is None else f"{st['mean_bps']:+.0f}"
            cells.append(f"h{h}: n={st['n']} {mb}bps")
        print(f"  {a:<5} " + "   ".join(cells))

    print("\nPER SEAT ERA — the decider's identity changed mid-window "
          "(trend -> whale 08-17 -> regimebook 08-18); a pooled number is "
          "mostly the trend seat's behaviour, not the current book's.")
    for name, _lo, _hi in SEAT_ERAS:
        cells = []
        for h in HORIZON_BARS:
            ag = _stats(per_era[name]["agree"][h], h)
            dg = _stats(per_era[name]["disagree"][h], h)
            amb = "-" if ag["mean_bps"] is None else "%+.0f" % ag["mean_bps"]
            dmb = "-" if dg["mean_bps"] is None else "%+.0f" % dg["mean_bps"]
            cells.append("h%d: agree n=%d %sbps | disagree n=%d %sbps"
                         % (h, ag["n"], amb, dg["n"], dmb))
        print("  %-12s " % name + "   ".join(cells))

    # ---- the pre-committed verdict -------------------------------------
    verdict, reasons, earned_h = decide_verdict(pooled, HORIZON_BARS)

    print(f"\nVERDICT: {verdict}  (pre-committed rule: disagree mean < 0, "
          f"|t| >= {T_BAR} overlap-corrected, agree > disagree)")
    for r in reasons:
        print(f"  blocker  {r}")
    if earned_h:
        print(f"  earned at horizon(s): {earned_h}")
    else:
        print(f"  => an entry veto keyed on {args.advisor} has no measured "
              f"basis in this window. Its default is OFF; arming it is a "
              f"decision that needs this measurement to pass first.")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat()
                        .replace("+00:00", "Z"),
        "decider": args.decider,
        "advisor": args.advisor,
        "window_days": args.window_days,
        "verdict": verdict,
        "pooled": {f"{b}|h{h}": v for (b, h), v in pooled.items()},
        "blockers": reasons,
        "contrast_reported_not_precommitted": {
            ("h%d" % h): v for h, v in contrasts.items()},
    }
    print("\n" + json.dumps(out, indent=2, default=str)[:1200])
    return 3 if verdict == "EARNED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

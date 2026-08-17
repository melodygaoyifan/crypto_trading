"""[P291] The trend-rule challenger seat criteria, made executable — checks, never acts.

WHAT THIS IS FOR. P288 found the first candidates in the campaign to beat the
SMA200 incumbent under the house rules: DONCHIAN-100 and EMA-ENSEMBLE, at ~1/4
the incumbent's turnover, era-stable on BOTH lab windows. P289 wired their
forward ledgers (`donchian_*`, `emaens_*`, first live 2026-08-17). What did NOT
exist until this file: a pre-committed criterion for what a forward PASS
ENTITLES them to. A bar without an instrument becomes "whoever re-greps by
hand" (the P230 rule), and a criterion chosen after seeing the number is
selection, not evidence (P198/P243).

THE PRE-COMMITTED CRITERION (all five, per (strategy, asset) CELL; fixed
2026-08-17, BEFORE any forward evidence existed):

  1. DATE: today >= 2026-09-16 (the ledgers' own 30-day P166 window, which
     opened at the 2026-08-17 deploy).
  2. LAB PRECONDITION: the cell must be one the P288 lab actually DETHRONED
     (see LAB_DETHRONED below). A cell the lab says SMA200 won can PASS its
     forward bar and still have NO seat claim — promoting it would be
     selection on a single forward window against design-era evidence that
     already rejected it. This is the criterion's sharpest edge and the
     directive-level "ETH+SOL only" framing is not precise enough for it:
     per CELL, emaens did NOT dethrone ETH (design -0.231 vs the incumbent's
     +0.088), so `emaens/ETH` is NOT an eligible cell even though ETH is an
     "ETH+SOL" asset.
  3. LEDGER: >= 30 days of span with >= 20 directional (non-flat) records.
     These are long/flat books: a window in which the book never took a
     position made no claim distinguishable from "stay flat", and a seat is
     a claim about direction.
  4. FORWARD BAR: the scorer's own verdict on the >=30d window is PROMOTE.
     Consumed, never re-derived — `assess_record` already applies the full
     P166 gate (positive IC at every valid horizon, overlap-corrected
     |t| >= 2, edge >= 2x round-trip cost, IC floor, Sharpe). Re-deriving it
     here would be a second, looser gate wearing the same name (P166's own
     two-call-sites lesson).
  5. BEATS THE INCUMBENT ON THE SAME BARS: the cell's minimum-horizon IC must
     exceed the incumbent `regimebook_{ASSET}` ledger's minimum-horizon IC on
     the same window. A seat swap is COMPARATIVE — the P166 bar certifies a
     signal against COSTS, not against the signal already in the seat. If the
     incumbent ledger cannot be scored, the comparison is UNEVALUATED and the
     cell does NOT pass (P199: missing data never counts as passing).
     NOTE the like-for-like caveat, recorded rather than hidden: for ETH and
     SOL the regimebook ledger IS the SMA200 trend/hold book (P250: ETH is
     trend-only, SOL hold-bull-only), so the comparison is clean. BTC's
     regimebook carries funding legs and is NOT like-for-like — which costs
     nothing here, because no BTC cell is in LAB_DETHRONED anyway.

WHAT A PASS ENTITLES THEM TO — DELIBERATELY NOT AN AUTO-SWAP. The P288
virgin-era probe (`unread_era_probe_rules_p288.json`) returned **PARTIAL** for
both challengers: they pass the breadth leg 5/5 and beat SMA200 on every
out-of-selection TOTAL, but they FAIL the one property the incumbent's
certification is built on and the live book is deployed FOR — the BTC-2018
crash-dodge (DONCHIAN -0.60, EMA-ENSEMBLE -0.51 against the incumbent's -0.28
and a pre-committed -0.35 bar). Their hysteresis exits trends later: more
upside captured, ~2x deeper bear-year drawdown. That is a RISK-PREFERENCE
trade-off, and no instrument can settle a preference. So the terminal verdict
here is ELIGIBLE (exit 3), which names the trade-off and the exact config edit
and stops. It never says "fire".

EXIT CODES:
  0 = not yet — per-cell condition status printed
  3 = ELIGIBLE — at least one cell cleared all five; operator risk-preference
      decision required (grep-able)
  2 = refusal — an input that cannot be read must never read as a verdict
      (P199)

It NEVER edits config (P141). WHERE THIS RUNS: operator-local (P213) — the IC
half needs the OHLCV parquets, which are not in the engine image. Run
`scripts/september_check.py` first; it pulls fresh server evidence into the
`*_pulled` dirs this script defaults to (local `data/` is never live
evidence — P255, and the pre-P287 mlp_seat_check bug this inherits the fix of).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The ledgers' 30-day P166 window opened at the 2026-08-17 deploy (P289).
EARLIEST = date(2026, 9, 16)
MIN_SPAN_DAYS = 30.0
MIN_DIRECTIONAL = 20
INCUMBENT_PREFIX = "regimebook"
HORIZONS = (4, 12, 24)  # the scorer's standard set

# [P288] The lab's DETHRONING map, frozen with the numbers that produced it
# (training/reports/trend_rule_lab_p288.json, verified at source 2026-08-17).
# Rule: DETHRONES iff design-era net > SMA200 AND pre-design >= SMA200 - 5%.
#   BTC  SMA200 D=+0.594 P=+0.549 : donchian D=+0.582 (LOSES), emaens D=+0.467
#        (LOSES) -> SMA200 STANDS, no BTC cell is eligible.
#   ETH  SMA200 D=+0.088 P=+1.473 : donchian D=+0.177 P=+1.507 (DETHRONES),
#        emaens D=-0.231 (LOSES -> not a cell).
#   SOL  SMA200 D=+1.210 P=+3.303 : donchian D=+1.938 P=+4.188 (DETHRONES),
#        emaens D=+1.234 P=+4.985 (DETHRONES).
LAB_DETHRONED = {
    ("donchian", "ETH"): "design +0.177 vs SMA200 +0.088; pre-design +1.507 vs +1.473; turnover 39 vs 140",
    ("donchian", "SOL"): "design +1.938 vs SMA200 +1.210; pre-design +4.188 vs +3.303; turnover 34 vs 142",
    ("emaens", "SOL"): "design +1.234 vs SMA200 +1.210; pre-design +4.985 vs +3.303; turnover 33 vs 142",
}

# [P288] The measured trade-off a passing cell buys. Printed on every ELIGIBLE
# verdict so the operator decision is made against the number, not a memory.
CRASH_DODGE_CAVEAT = (
    "P288 virgin-era probe = PARTIAL for both challengers: breadth 5/5 and "
    "higher out-of-selection totals than SMA200 everywhere, but the BTC-2018 "
    "crash-dodge FAILS (donchian -0.60, emaens -0.51 vs incumbent -0.28, bar "
    "-0.35). Later exits => more upside, ~2x deeper bear-year drawdown. The "
    "incumbent keeps a certification these challengers do not have."
)


def _ic_from_stats(st: dict, horizon: int):
    """[P287] Read the scorer's SUCCESS key `ic_per_horizon`.

    `ic_per_h` exists ONLY on the `ohlcv_missing` ERROR record
    (compute_shadow_ic.py:417 vs :478). Reading `ic_per_h` alone is what made
    the P285 seat checker's kill-screen structurally unevaluated on every
    successful run — a P174-class check that could not fire, and a P2
    reader/writer key mismatch. Both shapes and both int/str keys accepted;
    None means "could not be evaluated", which never satisfies a condition."""
    ic_map = st.get("ic_per_horizon")
    if not isinstance(ic_map, dict) or not ic_map:
        ic_map = st.get("ic_per_h")
    if not isinstance(ic_map, dict):
        return None
    v = ic_map.get(horizon, ic_map.get(str(horizon)))
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _min_ic(st: dict, horizons=HORIZONS):
    """Weakest horizon IC, or None if any horizon is unreadable. Deliberately
    the MINIMUM: a seat claim should be judged on the horizon where the signal
    is weakest, not on its best one (the P166 gate takes the same view)."""
    vals = [_ic_from_stats(st, h) for h in horizons]
    if any(v is None for v in vals):
        return None
    return min(vals)


def decide(cell, today: date, span_days: float, n_directional: int,
           verdict, challenger_min_ic, incumbent_min_ic) -> tuple:
    """Pure criterion for ONE (strategy, asset) cell.

    Returns (eligible: bool, conditions: list[(name, ok, detail)]).
    Every None input means "could not be evaluated" and NEVER passes (P199)."""
    lab_ok = cell in LAB_DETHRONED
    beats = (challenger_min_ic is not None and incumbent_min_ic is not None
             and challenger_min_ic > incumbent_min_ic)
    conds = [
        ("date", today >= EARLIEST, f"{today} vs earliest {EARLIEST}"),
        ("lab_dethroned", lab_ok,
         LAB_DETHRONED.get(cell,
                           "NOT a P288-dethroned cell — design-era evidence "
                           "says SMA200 won here; a forward PASS is not a "
                           "seat claim")),
        ("ledger_span", span_days >= MIN_SPAN_DAYS,
         f"{span_days:.1f}d vs {MIN_SPAN_DAYS:.0f}d required"),
        ("n_directional", n_directional >= MIN_DIRECTIONAL,
         f"{n_directional} vs {MIN_DIRECTIONAL} required"),
        ("forward_bar_PROMOTE", verdict == "PROMOTE",
         "UNEVALUATED" if verdict is None else f"scorer verdict={verdict}"),
        ("beats_incumbent", beats,
         "UNEVALUATED" if (challenger_min_ic is None or incumbent_min_ic is None)
         else (f"min-horizon IC {challenger_min_ic:+.4f} vs incumbent "
               f"{incumbent_min_ic:+.4f}")),
    ]
    return all(ok for _, ok, _ in conds), conds


def _score(ledger_dir: Path, prefixes, window_days: int):
    """Load + score ledgers. Returns {(strategy, asset): stats} or raises."""
    from analytics.shadow_ic.compute_shadow_ic import (
        load_shadow_ledgers, compute_per_strategy_ic)
    records = load_shadow_ledgers(ledger_dir, prefixes=tuple(prefixes))
    return records, compute_per_strategy_ic(records, horizons_bars=HORIZONS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-dir",
                    default=str(REPO / "data" / "strategy_shadow_pulled"))
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--today", default=None, help="override for tests (YYYY-MM-DD)")
    args = ap.parse_args()
    today = (datetime.strptime(args.today, "%Y-%m-%d").date()
             if args.today else date.today())

    ledger_dir = Path(args.ledger_dir)
    # [P287/P255] The pulled dir not existing means september_check has never
    # run here — refuse rather than evaluate nothing.
    if not ledger_dir.exists():
        print(f"P291 CANNOT BE EVALUATED: {ledger_dir} does not exist — the "
              f"ledgers have not been pulled. Run scripts/september_check.py "
              f"first (local data/ is never live evidence, P255).",
              file=sys.stderr)
        return 2

    found = sorted(p.name for p in ledger_dir.glob("donchian_*.jsonl"))
    found += sorted(p.name for p in ledger_dir.glob("emaens_*.jsonl"))
    if not found:
        print("P291 CANNOT BE EVALUATED: no donchian_*/emaens_*.jsonl ledgers "
              "in the pulled dir. 'No ledger' is NOT 'not eligible' (P199).",
              file=sys.stderr)
        return 2

    try:
        records, stats = _score(ledger_dir, ("donchian", "emaens"),
                                args.window_days)
        _, inc_stats = _score(ledger_dir, (INCUMBENT_PREFIX,), args.window_days)
    except Exception as e:
        print(f"P291 CANNOT BE EVALUATED: scoring failed "
              f"({type(e).__name__}: {e}).", file=sys.stderr)
        return 2

    if any(v.get("error") == "ohlcv_missing" for v in stats.values()):
        print("P291 CANNOT BE EVALUATED: OHLCV parquets missing — run "
              "operator-local after refresh_ohlcv_4h.py (P213). A missing "
              "price series is not a verdict (P199).", file=sys.stderr)
        return 2

    from analytics.shadow_ic.compute_shadow_ic import assess_record

    # Span/directional counts come from the RECORDS, per cell.
    per_cell: dict = {}
    for r in records:
        key = (r.get("strategy"), r.get("asset"))
        if key[0] is None or key[1] is None:
            continue
        per_cell.setdefault(key, []).append(r)

    print(f"P291 trend-rule challenger seat criteria — window "
          f"{args.window_days}d, {len(records)} challenger records")
    print(f"  lab-dethroned cells (the only ones that can be ELIGIBLE): "
          f"{sorted(LAB_DETHRONED)}")

    eligible_cells = []
    for cell in sorted(per_cell):
        recs = per_cell[cell]
        ts = sorted(r["_parsed_ts"] for r in recs if r.get("_parsed_ts"))
        span = ((ts[-1] - ts[0]).total_seconds() / 86400.0
                if len(ts) > 1 else 0.0)
        n_dir = sum(1 for r in recs
                    if float(r.get("direction", 0.0) or 0.0) != 0.0)
        st = stats.get(cell, {})
        verdict = None
        if st and "error" not in st:
            verdict = assess_record(st, args.window_days).verdict.value
        inc = inc_stats.get((INCUMBENT_PREFIX, cell[1]), {})
        ok, conds = decide(cell, today, span, n_dir, verdict,
                           _min_ic(st), _min_ic(inc) if inc else None)
        print(f"\n  {cell[0]}/{cell[1]}  ({len(recs)} records)")
        for name, cond_ok, detail in conds:
            print(f"    [{'PASS' if cond_ok else '....'}] {name}: {detail}")
        if ok:
            eligible_cells.append(cell)

    if eligible_cells:
        print(f"\n  -> P291 ELIGIBLE: {', '.join(f'{s}/{a}' for s, a in eligible_cells)}")
        print(f"     {CRASH_DODGE_CAVEAT}")
        print("     THIS IS NOT A FIRE INSTRUCTION. A seat swap here is an "
              "OPERATOR RISK-PREFERENCE decision (upside vs bear-year "
              "drawdown), not a threshold. If taken, it is a recorded step "
              "with its own P-entry (P141): the challenger replaces the "
              "SMA200 labeler for that asset in the regimebook seat path, at "
              "UNCHANGED size, caps, stops and gates. This checker never "
              "edits config.")
        return 3

    print("\n  not yet — every condition must pass on a lab-dethroned cell "
          "(missing data never counts as passing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""[P285] The mlp early-seat criterion, made executable — checks, never acts.

THE DECISION (P285, operator-adopted 2026-08-16 "adopt the criterion and
wire up"): the P283b-certified BTC mlp_small may take the BTC direction
seat EARLY — before the full 30d P166 read — at UNCHANGED size/caps/stops,
because the incumbent trend signal has measured-NEGATIVE live forward IC
(P198) and is itself tripwire-bound (P237). The swap changes which signal
drives an already-trading, already-bounded book; it does not add risk.

THE PRE-COMMITTED CRITERION (all four, judged by this instrument — chosen
BEFORE any shadow evidence existed, the P271 rule):
  1. today >= 2026-08-28 (>= ~2 weeks of shadow ledger);
  2. the mlpshadow_BTC ledger spans >= 14 days with >= 40 scoreable
     directional records;
  3. the 16h shadow IC is NOT NEGATIVE (>= 0.0) — explicitly a WEAKER
     screen than the P166 promotion bar: a kill-screen, not a
     certification. The full P166 read at ~09-15 still governs anything
     beyond the current +/-1-contract book;
  4. the trend incumbent is still GATE-CLOSED on the latest weekly
     calibration report for BTC (if trend ever calibrates open, the swap
     question changes and goes back to the operator).

WHAT THIS DOES: evaluates the four conditions and states the verdict
loudly. It NEVER edits config (P141 — firing the actuator is a recorded
human step: set `"mlpshadow_mode": "enforce"` in configs/live_high_risk.json
and redeploy). Exit codes: 0 = not yet (per-condition status printed);
3 = FIRE (grep-able); 2 = refusal — an input that cannot be read must
never read as a verdict (P199).

WHERE THIS RUNS: operator-local (P213) — the IC half needs the OHLCV
parquets, which are not in the engine image. `scripts/september_check.py`
pulls fresh ledgers first; run that before trusting this.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

EARLIEST = date(2026, 8, 28)
MIN_SPAN_DAYS = 14.0
MIN_DIRECTIONAL = 40
ASSET = "BTC"
HORIZON_BARS = 4  # 16h at 4H bars — the certified decision horizon


def decide(today: date, span_days: float, n_directional: int,
           ic16, trend_closed_latest) -> tuple:
    """Pure criterion. Returns (fire: bool, conditions: list[(name, ok, detail)]).

    ic16 / trend_closed_latest may be None = "could not be evaluated";
    None NEVER satisfies a condition (P199 — missing is not passing)."""
    conds = [
        ("date", today >= EARLIEST, f"{today} vs earliest {EARLIEST}"),
        ("ledger_span", span_days >= MIN_SPAN_DAYS,
         f"{span_days:.1f}d vs {MIN_SPAN_DAYS:.0f}d required"),
        ("n_directional", n_directional >= MIN_DIRECTIONAL,
         f"{n_directional} vs {MIN_DIRECTIONAL} required"),
        ("ic16_not_negative", ic16 is not None and ic16 >= 0.0,
         "UNEVALUATED" if ic16 is None else f"IC={ic16:+.4f} (need >= 0)"),
        ("trend_still_gate_closed", trend_closed_latest is True,
         "UNEVALUATED" if trend_closed_latest is None
         else f"latest weekly report GATE-CLOSED={trend_closed_latest}"),
    ]
    return all(ok for _, ok, _ in conds), conds


def _trend_closed_latest(reports_dir: Path):
    """Latest weekly slope report's BTC verdict: True = GATE-CLOSED,
    False = some horizon TRADEABLE, None = cannot be evaluated."""
    import glob as _g
    import json as _j
    files = sorted(_g.glob(str(reports_dir / "slope_*.json")))
    if not files:
        return None
    by_day: dict = {}
    for f in files:
        try:
            rep = _j.loads(Path(f).read_text(encoding="utf-8"))
            by_day[rep["generated"][:10]] = rep
        except Exception:  # noqa: silent-swallow — unreadable report; absence handled below
            continue
    if not by_day:
        return None
    latest = by_day[sorted(by_day)[-1]]
    # [P253] only RENDERED verdicts count — no verdicts = cannot evaluate
    verdicts = [v for v in (
        h.get("vs_threshold")
        for h in latest.get("assets", {}).get(ASSET, {}).values()
        if isinstance(h, dict)) if v]
    if not verdicts:
        return None
    return not any("TRADEABLE" in v for v in verdicts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-dir", default=str(REPO / "data" / "strategy_shadow"))
    ap.add_argument("--reports-dir",
                    default=str(REPO / "data" / "evidence_reports"))
    ap.add_argument("--today", default=None,
                    help="override for tests (YYYY-MM-DD)")
    args = ap.parse_args()
    today = (datetime.strptime(args.today, "%Y-%m-%d").date()
             if args.today else date.today())

    ledger = Path(args.ledger_dir) / f"mlpshadow_{ASSET}.jsonl"
    if not ledger.exists():
        print(f"P285 CANNOT BE EVALUATED: {ledger} does not exist. Pull "
              f"fresh ledgers first (scripts/september_check.py). 'No "
              f"ledger' is NOT 'not fired' (P199).", file=sys.stderr)
        return 2

    from analytics.shadow_ic.compute_shadow_ic import (
        load_shadow_ledgers, compute_per_strategy_ic)
    records = load_shadow_ledgers(Path(args.ledger_dir),
                                  prefixes=("mlpshadow",))
    recs = [r for r in records if r.get("asset") == ASSET]
    if not recs:
        print("P285 CANNOT BE EVALUATED: ledger exists but holds no "
              "parseable BTC records.", file=sys.stderr)
        return 2

    ts = sorted(float(r.get("ts", 0.0) or 0.0) for r in recs)
    span_days = (ts[-1] - ts[0]) / 86400.0 if len(ts) > 1 else 0.0
    n_dir = sum(1 for r in recs
                if float(r.get("direction", 0.0) or 0.0) != 0.0)

    ic16 = None
    stats = compute_per_strategy_ic(recs, horizons_bars=(HORIZON_BARS,))
    st = stats.get(("mlpshadow", ASSET), {})
    if st.get("error") == "ohlcv_missing":
        print("P285 CANNOT BE EVALUATED: OHLCV parquets missing — run "
              "operator-local after refresh_ohlcv_4h.py (P213). A missing "
              "price series is not a verdict (P199).", file=sys.stderr)
        return 2
    ic_h = st.get("ic_per_h", {}).get(HORIZON_BARS)
    if ic_h is not None:
        ic16 = float(ic_h)

    trend_closed = _trend_closed_latest(Path(args.reports_dir))

    fire, conds = decide(today, span_days, n_dir, ic16, trend_closed)
    print(f"P285 mlp early-seat criterion — {len(recs)} ledger records:")
    for name, ok, detail in conds:
        print(f"  [{'PASS' if ok else '....'}] {name}: {detail}")
    if fire:
        print(f"\n  -> P285 CRITERION MET: the certified BTC mlp_small may "
              f"take the seat EARLY at unchanged size. ACTION (human, "
              f"recorded, P141): set '\"mlpshadow_mode\": \"enforce\"' in "
              f"configs/live_high_risk.json and redeploy; watch [MLP-SEAT] "
              f"lines on the next ticks. The ~09-15 P166 read still governs "
              f"anything beyond the current book. This checker never edits "
              f"config itself.")
        return 3
    print("\n  not yet — every condition must pass (missing data never "
          "counts as passing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

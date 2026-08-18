#!/usr/bin/env python
"""[P294] Seat controller CLI — run weekly, alongside the other evidence tools.

Gathers forward evidence for every DECIDE-slot candidate and prints which one
should hold the seat. NEVER edits config (P141): it prints the exact edit and
exits 3 so a cron/operator can see a recommendation without one being applied.

WHERE THIS RUNS (P213): in-container for the live evidence —
    docker exec hmats-engine python -X utf8 scripts/seat_check.py
or operator-local with --stats to feed numbers by hand.

EVIDENCE SOURCES (all pre-existing; this adds no new pipeline):
  * per-agent forward IC   analytics/ic/agent_ic_review.py
  * shadow ledgers         data/strategy_shadow/regimebook_*.jsonl
  * live decider           the `quant` agent series (trend, when enforce)

STRUCTURAL FACTS ENCODED HERE (inspected 2026-08-18, P294 §inspection):
  * regimebook/SOL is UNAVAILABLE, not flat — its bear-leg model was DELETED
    in P250 (it was the leak artifact) and configs/regimebook/ does not exist,
    so it emits `flat_degraded` forever. Scoring it as a flat opinion would
    let a broken candidate win by default.
  * regimebook/BTC currently expresses ONLY its funding legs (funding_short,
    funding_contrarian_short) — and P262 records those as the UNCERTIFIED
    slice of that book. Surfaced as a warning on the recommendation.
  * trend's signal is a 3-lookback vote quantized to {+-1/3, +-1}, so its
    asserted alpha is either 10bps (blocked by every threshold) or 30bps.
    It trades only on unanimity. [P295] the weak vote now reads FLAT rather
    than asserting an opinion the gate must veto.

Exit: 0 incumbent holds · 3 SWITCH recommended · 2 refusal (no usable input)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.seat.seat_controller import (  # noqa: E402
    Candidate, decide_seat, render, FLAT)

# [P294] Structural notes attached to a candidate when it wins or is blocked.
CAVEATS = {
    "regimebook": (
        "BTC's book is currently expressing ONLY its funding legs, which "
        "P262 records as the UNCERTIFIED slice; its certified trend/hold leg "
        "is flat in the present regime. SOL's book is structurally inert "
        "(model deleted in P250)."
    ),
    "trend": (
        "trend is a 3-lookback vote quantized to {+-1/3, +-1}: it asserts "
        "either 10bps (below every live threshold) or 30bps, so it trades "
        "ONLY on unanimous agreement. [P295] The weak-vote floor now ACTS "
        "(trend_min_abs_signal=0.50, live): a 2-of-3 vote reads FLAT rather "
        "than asserting 10bps for the gate to veto. The old 0.30 sat below "
        "the smallest reachable |sig| (1/3) and could never fire."
    ),
    "whale": (
        "whale clears the gate because it is BINARY (+-1 -> 30bps), not "
        "because it is better evidenced; it is silent on ~46/57/88% of ticks "
        "(BTC/ETH/SOL) and the incumbent covers those."
    ),
}


def _refuse(msg: str) -> int:
    print(f"REFUSING: {msg}", file=sys.stderr)
    print("A seat recommendation from missing evidence would be a guess "
          "wearing a measurement's name (P199).", file=sys.stderr)
    return 2


def _from_ic_report(path: Path) -> dict:
    """Parse an agent_ic_review JSON report into {agent: {h: (ic,t,n)}}."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    out: dict = {}
    rows = data.get("agents") or data.get("results") or data
    if isinstance(rows, dict):
        for agent, per_h in rows.items():
            if not isinstance(per_h, dict):
                continue
            out[agent] = {}
            for h, d in per_h.items():
                if isinstance(d, dict):
                    try:
                        out[agent][int(h)] = (
                            d.get("ic"), d.get("t"), int(d.get("n") or 0))
                    except (TypeError, ValueError):
                        continue
    return out


def availability_from_ledger(ledger_dir: Path, asset: str) -> Optional[bool]:
    """[P295] Read the regimebook ledger's own `available` flag.

    The book now stamps availability on every row (False when its version is
    degraded, e.g. SOL whose bear-leg model was deleted in P250), so the seat
    controller reads the producer's OWN statement instead of a hand-maintained
    list here that would drift the moment a model is restored.

    Returns None when the ledger cannot be read or predates the flag — which
    the caller must treat as "unknown", never as available.
    """
    p = Path(ledger_dir) / f"regimebook_{asset}.jsonl"
    try:
        last = None
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return None
        rec = json.loads(last)
    except Exception:
        return None
    if "available" not in rec:
        return None
    return bool(rec.get("available"))


def build_candidates(stats: dict) -> list:
    """stats: {name: {"ic_4h":..,"ic_16h":..,"t_4h":..,"t_16h":..,"n":..,
                      "available":bool, "in_market_rate":float, "note":str}}"""
    cands = []
    for name, d in stats.items():
        cands.append(Candidate(
            name=name,
            ic_4h=d.get("ic_4h"), ic_16h=d.get("ic_16h"),
            t_4h=d.get("t_4h"), t_16h=d.get("t_16h"),
            n=int(d.get("n") or 0),
            available=bool(d.get("available", True)),
            in_market_rate=d.get("in_market_rate"),
            note=str(d.get("note") or ""),
        ))
    return cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--incumbent", default=None,
                    help="current seat holder; default = read from the live "
                         "config profile")
    ap.add_argument("--stats", default=None,
                    help="JSON {name: {ic_4h, ic_16h, t_4h, t_16h, n, ...}} — "
                         "bypasses the report readers (operator-local use)")
    ap.add_argument("--ic-report", default=None,
                    help="path to an agent_ic_review JSON report")
    ap.add_argument("--config", default=str(REPO / "configs" / "live_high_risk.json"))
    args = ap.parse_args(argv)

    # --- incumbent -----------------------------------------------------
    incumbent = args.incumbent
    if incumbent is None:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        except Exception as e:
            return _refuse(f"cannot read live config ({type(e).__name__}) and "
                           f"no --incumbent given")
        # Precedence mirrors main.py's seat ordering: the LAST seat to run
        # wins, so whale outranks regimebook outranks trend.
        if str(cfg.get("whale_seat_mode", "off")).lower() == "enforce":
            incumbent = "whale"
        elif str(cfg.get("regimebook_mode", "off")).lower() == "enforce":
            incumbent = "regimebook"
        elif str(cfg.get("trend_following_mode", "off")).lower() == "enforce":
            incumbent = "trend"
        else:
            incumbent = FLAT

    # --- evidence ------------------------------------------------------
    if args.stats:
        try:
            stats = json.loads(args.stats)
        except Exception as e:
            return _refuse(f"--stats is not valid JSON ({type(e).__name__})")
    elif args.ic_report:
        p = Path(args.ic_report)
        if not p.exists():
            return _refuse(f"ic report not found: {p}")
        raw = _from_ic_report(p)
        if not raw:
            return _refuse(f"ic report {p} held no agent rows")
        stats = {}
        # `quant` IS the seat holder's series (trend when enforce, whale when
        # the whale seat is on) — it is the decider slot, whoever occupies it.
        for agent, series in raw.items():
            if agent not in ("quant", "whale"):
                continue
            name = "trend" if agent == "quant" else agent
            ic4, t4, n4 = series.get(1, (None, None, 0))
            ic16, t16, n16 = series.get(4, (None, None, 0))
            stats[name] = {"ic_4h": ic4, "ic_16h": ic16, "t_4h": t4,
                           "t_16h": t16, "n": max(n4, n16)}
    else:
        return _refuse("no evidence source: pass --stats or --ic-report")

    cands = build_candidates(stats)
    # [P295] Let the regimebook ledger's OWN `available` flag override any
    # hand-passed value — the producer knows whether it can take a position.
    _ldir = REPO / "data" / "strategy_shadow"
    for c in cands:
        if c.name != "regimebook":
            continue
        avail = [availability_from_ledger(_ldir, a) for a in ("BTC", "ETH", "SOL")]
        known = [a for a in avail if a is not None]
        if known and not any(known):
            c.available = False
            c.note = ("every regimebook asset reports available=false "
                      "(SOL's bear-leg model was deleted in P250)")
    if not cands:
        return _refuse("no candidates could be built from the evidence")

    decision = decide_seat(cands, incumbent)
    print(render(decision))

    _cav = CAVEATS.get(decision.winner)
    if _cav:
        print(f"\nCAVEAT on '{decision.winner}': {_cav}")

    return 3 if decision.switch else 0


if __name__ == "__main__":
    sys.exit(main())

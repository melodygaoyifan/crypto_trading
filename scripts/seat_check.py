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
  * live decider           the `quant` agent series — whoever holds the
                           DECIDE slot on each tick. [P420] That is now a
                           MIX of seats (skew on BTC/ETH, regimebook on SOL,
                           ETF de-risk, whale where the book is flat), so the
                           series is labelled by the report's
                           `primary_strategy` census when it carries one and
                           "quant (mixed seats)" otherwise — never by a
                           config-derived single name.

SEAT ARCHITECTURE ENCODED HERE [P420 — the pre-P420 text described a seat
architecture that no longer exists, and the 2026-08-24 cron run prescribed
the P299-RETIRED `trend_assets: []`]:
  * live decider PER ASSET, in the order the seats run (last to fire wins):
        skew_contra (skew_seat_mode enforce, asset in skew_seat_assets)
      > etf_flow    (etf_seat_mode enforce, asset in etf_decide_assets)
      > regimebook  (regimebook_mode enforce)
      > whale       (whale_seat_mode enforce)
      > trend       (trend_following_mode enforce)
      > flat
    read from the live profile by analytics.seat.seat_controller.live_incumbent.
  * skew_contra and etf_flow are NOT scoreable by this instrument (their
    series are feeds, not attribution agents). When one of them is the live
    decider for an asset in scope this tool REFUSES (exit 2) with the reason
    "decider not scoreable by this instrument — read skewetf_* via
    compute_shadow_ic" rather than prescribe a switch from the wrong series.
  * whale is a CANDIDATE only while whale_seat_mode is "enforce" (P417
    demoted it to a weighted ADVISE member; its series is no longer a seat).
  * the `regimebook` candidate is the POOLED home-trio book (P297's six-year
    certification was measured on BTC/ETH/SOL), so its availability check
    reads those three ledgers by design.

STRUCTURAL FACTS carried forward (P294 §inspection, P382, P299):
  * regimebook/SOL runs ETH's certified trend-only book (`v1_trend_only`,
    available=True); P250's deleted bear-leg model was never certified for
    SOL. The availability check still reads the ledger's OWN `available`
    flag — a book that genuinely degrades must not score as a flat opinion.
  * regimebook/BTC currently expresses ONLY its funding legs — P262 records
    those as the UNCERTIFIED slice of that book. Surfaced as a caveat.
  * trend's signal is a 3-lookback vote quantized to {+-1/3, +-1}, so its
    asserted alpha is either 10bps (blocked by every threshold) or 30bps.
    It trades only on unanimity. [P295] the weak vote reads FLAT.

Exit: 0 incumbent holds · 3 SWITCH recommended · 2 refusal (no usable input,
      or the live decider is not scoreable here)
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
    Candidate, decide_seat, render, FLAT,
    UNSCOREABLE_SEATS, UNSCOREABLE_REASON, live_incumbents)

# [P294] Structural notes attached to a candidate when it wins or is blocked.
CAVEATS = {
    "regimebook": (
        "BTC's book is currently expressing ONLY its funding legs, which "
        "P262 records as the UNCERTIFIED slice; its certified trend/hold leg "
        "is flat in the present regime. SOL's book is the certified "
        "trend-only form (v1_trend_only, P299 — the P250 bear-leg deletion "
        "removed a leg SOL never certified, not the book). [P420] The "
        "candidate is the POOLED home-trio book; breadth books are the "
        "separate `regimebook_breadth` family."
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
        "(BTC/ETH/SOL) and the incumbent covers those. [P417] demoted from a "
        "seat to a weighted ADVISE member — it is a candidate here only while "
        "whale_seat_mode is 'enforce'."
    ),
    "skew_contra": (
        "[P420] " + UNSCOREABLE_REASON + ". The label is a historical "
        "misnomer: the rule rides call-richness (skew_25d = put - call), it is "
        "not contrarian (defense/skew_flow_signal SIGN CONVENTION)."
    ),
    "etf_flow": "[P420] " + UNSCOREABLE_REASON + ".",
}

# [P420] What the quant series is called when the report carries no census.
QUANT_MIXED_LABEL = "quant (mixed seats)"


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
        for agent, block in rows.items():
            if not isinstance(block, dict):
                continue
            # [P295c] The real report nests per-horizon cells under "horizons"
            # (verified against the live 2026-08-17 report). The flat shape is
            # accepted too so a hand-built stats file still parses — but the
            # nested one is what agent_ic_review actually emits, and reading
            # only the flat shape silently produced n=0 for every agent, i.e.
            # a verdict computed from nothing (P264).
            per_h = block.get("horizons") if isinstance(
                block.get("horizons"), dict) else block
            out[agent] = {}
            for h, d in per_h.items():
                if isinstance(d, dict):
                    try:
                        out[agent][int(h)] = (
                            d.get("ic"), d.get("t"), int(d.get("n") or 0))
                    except (TypeError, ValueError):
                        continue
    return out


def quant_series_label(report_path: Optional[Path]) -> str:
    """[P420] Label the `quant` series by the report's `primary_strategy`
    census when it carries one ({name: share}); else say what is true —
    the series is a MIX of seats. Never a config-derived single name."""
    if report_path is None:
        return QUANT_MIXED_LABEL
    try:
        data = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: silent-swallow — an unreadable report already refused upstream; label only
        return QUANT_MIXED_LABEL
    census = data.get("primary_strategy_census") if isinstance(data, dict) else None
    if isinstance(census, dict) and census:
        parts = []
        for k, v in sorted(census.items(), key=lambda kv: -float(kv[1] or 0)):
            try:
                parts.append(f"{k} {100.0 * float(v):.0f}%")
            except (TypeError, ValueError):
                parts.append(str(k))
        return "quant (census: " + ", ".join(parts) + ")"
    return QUANT_MIXED_LABEL


def availability_from_ledger(ledger_dir: Path, asset: str) -> Optional[bool]:
    """[P295] Read the regimebook ledger's own `available` flag.

    The book now stamps availability on every row (False when its version is
    degraded — e.g. a feature-coverage gap; SOL's P250 bear-leg deletion no
    longer degrades it, P299 made SOL v1_trend_only), so the seat
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
                    help="current seat holder; default = read PER ASSET from "
                         "the live config profile (refuses when it differs "
                         "across --assets or is not scoreable here)")
    ap.add_argument("--assets", default=None,
                    help="comma list of assets in scope for the incumbent read "
                         "(default: the live profile's `assets`)")
    ap.add_argument("--stats", default=None,
                    help="JSON {name: {ic_4h, ic_16h, t_4h, t_16h, n, ...}} — "
                         "bypasses the report readers (operator-local use)")
    ap.add_argument("--ic-report", default=None,
                    help="path to an agent_ic_review JSON report")
    ap.add_argument("--config", default=str(REPO / "configs" / "live_high_risk.json"))
    args = ap.parse_args(argv)

    # --- live config (best-effort) ---------------------------------------
    # Read once: it decides the incumbent(s) AND which candidates exist.
    # [P382] An unreadable config only REFUSES when the incumbent has to come
    # from it; with --incumbent given it says nothing more than it knows.
    cfg: dict = {}
    cfg_err: Optional[Exception] = None
    try:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    except Exception as e:  # noqa: silent-swallow — surfaced below
        cfg_err = e

    def _enforce(key: str) -> bool:
        return str(cfg.get(key, "off")).lower() == "enforce"

    # --- incumbent(s) -----------------------------------------------------
    # [P420] PER ASSET, by the order the seats run. The old single-label read
    # ("regimebook > whale > trend") did not know the skew/etf seats existed
    # and scored the wrong series as the seat holder.
    incumbent = args.incumbent
    scope = ([a.strip().upper() for a in args.assets.split(",") if a.strip()]
             if args.assets else
             [str(a).upper() for a in (cfg.get("assets") or ["BTC", "ETH", "SOL"])])
    per_asset = live_incumbents(cfg, scope) if not cfg_err else {}
    if per_asset:
        print("live decider per asset (from the profile, in seat-run order):")
        for a in scope:
            flag = ("  <- NOT scoreable here" if per_asset[a] in UNSCOREABLE_SEATS
                    else "")
            print(f"   {a:5s}: {per_asset[a]}{flag}")
    if incumbent is None:
        if cfg_err is not None:
            return _refuse(f"cannot read live config "
                           f"({type(cfg_err).__name__}) and no --incumbent given")
        unscoreable = sorted(a for a, s in per_asset.items() if s in UNSCOREABLE_SEATS)
        if unscoreable:
            return _refuse(
                f"{UNSCOREABLE_REASON} (assets {unscoreable}: "
                f"{ {a: per_asset[a] for a in unscoreable} }). Pass --assets "
                f"to scope this run to the assets whose decider IS an "
                f"attribution series (e.g. --assets SOL).")
        distinct = sorted(set(per_asset.values()))
        if len(distinct) != 1:
            return _refuse(f"the live decider differs across {scope}: "
                           f"{per_asset} — one seat verdict cannot cover them; "
                           f"pass --assets to scope the run")
        incumbent = distinct[0]
    elif incumbent in UNSCOREABLE_SEATS:
        return _refuse(f"--incumbent {incumbent}: {UNSCOREABLE_REASON}")

    # --- evidence ------------------------------------------------------
    report_path: Optional[Path] = None
    if args.stats:
        try:
            stats = json.loads(args.stats)
        except Exception as e:
            return _refuse(f"--stats is not valid JSON ({type(e).__name__})")
    elif args.ic_report:
        report_path = Path(args.ic_report)
        if not report_path.exists():
            return _refuse(f"ic report not found: {report_path}")
        raw = _from_ic_report(report_path)
        if not raw:
            return _refuse(f"ic report {report_path} held no agent rows")
        stats = {}
        # `quant` IS the seat holder's series — the decider slot, whoever
        # occupied it on each tick. It is compared under the INCUMBENT's name
        # (that is what it is evidence about) and labelled honestly below.
        # [P420] `whale` is a candidate only while its seat is enforced (P417
        # demoted it to ADVISE); an explicit --incumbent whale means the
        # quant series IS whale's seat, so the agent series is not a second
        # candidate under the same name.
        whale_is_candidate = _enforce("whale_seat_mode") and incumbent != "whale"
        for agent, series in raw.items():
            if agent == "quant":
                name = incumbent
            elif agent == "whale" and whale_is_candidate:
                name = "whale"
            else:
                continue
            ic4, t4, n4 = series.get(1, (None, None, 0))
            ic16, t16, n16 = series.get(4, (None, None, 0))
            stats[name] = {"ic_4h": ic4, "ic_16h": ic16, "t_4h": t4,
                           "t_16h": t16, "n": max(n4, n16)}
        if "whale" in raw and not whale_is_candidate and incumbent != "whale":
            print("note: `whale` agent series present but whale_seat_mode is "
                  "not 'enforce' — not a seat candidate (P417).")
    else:
        return _refuse("no evidence source: pass --stats or --ic-report")

    cands = build_candidates(stats)
    # [P295] Let the regimebook ledger's OWN `available` flag override any
    # hand-passed value — the producer knows whether it can take a position.
    # [P420] HOME TRIO BY DESIGN: the `regimebook` candidate is the pooled
    # BTC/ETH/SOL book; breadth ledgers belong to `regimebook_breadth`.
    _ldir = REPO / "data" / "strategy_shadow"
    for c in cands:
        if c.name != "regimebook":
            continue
        avail = [availability_from_ledger(_ldir, a) for a in ("BTC", "ETH", "SOL")]
        known = [a for a in avail if a is not None]
        if known and not any(known):
            c.available = False
            c.note = ("every regimebook asset reports available=false "
                      "(the ledger's own flag; SOL is v1_trend_only since "
                      "P299 — P250's deleted bear leg is no longer the cause)")
    if not cands:
        return _refuse("no candidates could be built from the evidence")

    print(f"quant series label: {quant_series_label(report_path)}")
    decision = decide_seat(cands, incumbent)
    print(render(decision))

    _cav = CAVEATS.get(decision.winner)
    if _cav:
        print(f"\nCAVEAT on '{decision.winner}': {_cav}")

    if decision.refused:
        print("\nExiting 2 (refusal): no scoreable candidate. This is NOT a "
              "recommendation to go flat.", file=sys.stderr)
        return 2
    return 3 if decision.switch else 0


if __name__ == "__main__":
    sys.exit(main())

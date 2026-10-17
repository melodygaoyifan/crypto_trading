"""[P1-1] Producer for configs/cost/venue_fee_schedule_coinbase.json
(+ [P1-2] configs/cost/realized_friction_phase0.json).

Reads the Phase 0 cost audit (reports/phase0_params.json, built from the
venue's own fills + get_transaction_summary) and writes the runtime fee
schedule the engine loads via core.cde_fees.load_venue_fee_schedule().

No fee number is typed here: every value comes from phase0_params.json. The
runtime file exists because reports/ is dockerignored and configs/ ships in
the image.

Structure (Phase 0 §2.1, read off every fill's commission_detail_total, not a
fit): commission = pct-of-notional (tier-dependent) + a FIXED per-contract
venue+clearing charge. The fixed part is ADDITIVE, not a floor — Phase 0 found
only 1/211 fills where max(pct, fixed) and pct+fixed would even differ in
which term dominates, and modelling it as max() under-prices a BTC nano fill
by ~12%. `min_fee_per_contract_usd` is kept as a separate (floor) field per
the Phase 1 spec; Phase 0 observed no per-order minimum, so it is 0.0.

Usage:
    python -X utf8 scripts/export_venue_fee_schedule.py [--params reports/phase0_params.json]
Exit 2 when the params file is missing or lacks the fields (no defaults).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "configs", "cost", "venue_fee_schedule_coinbase.json")


def build(params: dict) -> dict:
    fit = params["fee_model_fit"]
    tier = fit["current_tier"]
    disc = fit.get("maker_discount_observed") or {}
    maker = float(tier["maker_bps"])
    taker = float(tier["taker_bps"])
    # spec: maker_discount_observed == false -> maker = taker
    if disc is False or (isinstance(disc, dict) and disc.get("current_tier_bps") in (0, 0.0)):
        maker = taker
    specs = params["product_specs"]
    return {
        "venue": "coinbase_cde",
        "rate_bps_maker": maker,
        "rate_bps_taker": taker,
        "fixed_per_contract_usd": float(tier["fixed_per_contract_usd"]),
        "min_fee_per_contract_usd": 0.0,
        "contract_size": {a: float(s["contract_size"]) for a, s in specs.items()},
        "provenance": {
            "source": "reports/phase0_params.json (HMATS Phase 0 cost audit)",
            "generated_from": params.get("generated_at"),
            "tier": tier.get("name"),
            "thirty_day_volume_usd": tier.get("thirty_day_volume_usd"),
            "note": ("current tier is the MOST expensive (Intro, 30d vol <$10k); "
                     "a cheaper tier only makes this schedule overcharge (P167 safe side)"),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(REPO, "reports", "phase0_params.json"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--friction-out", default=os.path.join(
        REPO, "configs", "cost", "realized_friction_phase0.json"))
    a = ap.parse_args(argv)
    if not os.path.exists(a.params):
        print(f"REFUSING: {a.params} missing — run the Phase 0 audit first (no defaults)")
        return 2
    try:
        with open(a.params, encoding="utf-8") as fh:
            sched = build(json.load(fh))
    except (KeyError, TypeError, ValueError) as e:
        print(f"REFUSING: phase0 params incomplete ({type(e).__name__}: {e})")
        return 2
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(sched, fh, indent=1)
        fh.write("\n")
    print(json.dumps(sched, indent=1))
    # [P1-2] the realized-friction baseline, same producer, same source
    with open(a.params, encoding="utf-8") as fh:
        rf = build_friction(json.load(fh))
    with open(a.friction_out, "w", encoding="utf-8") as fh:
        json.dump(rf, fh, indent=1)
        fh.write("\n")
    print(json.dumps(rf, indent=1))
    return 0


def build_friction(params: dict) -> dict:
    """[P1-2] Static Phase 0 baseline for execution/cost/realized_friction.py.
    Only the three trio assets; a value the audit marked non-numeric
    (UNAVAILABLE/INSUFFICIENT) is carried as None, never defaulted."""
    def num(x):
        return float(x) if isinstance(x, (int, float)) else None
    per = {}
    for a in ("BTC", "ETH", "SOL"):
        v = params["per_asset"][a]
        per[a] = {"rt_fee_bps": num(v.get("rt_fee_bps")),
                  "rt_spread_bps": num(v.get("rt_spread_bps")),
                  "median_hold_bars": num(v.get("median_hold_bars")),
                  "n_rt": v.get("n_rt")}
    rec = params.get("recommended", {})
    return {"per_asset": per,
            "MIN_HOLD_BARS": num(rec.get("MIN_HOLD_BARS")),
            "provenance": {"source": "reports/phase0_params.json",
                           "generated_from": params.get("generated_at"),
                           "note": "rt_fee/rt_spread are medians over Jun-Oct RTs "
                                   "(mostly cheaper tiers); the model floors the fee "
                                   "term at 2x the current venue schedule"}}


if __name__ == "__main__":
    sys.exit(main())

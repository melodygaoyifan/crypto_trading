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


def build_tiers(fills, api_tier: dict, recent_days: float = 14.0) -> dict:
    """[P1R-1] Volume tiers + per-asset fixed components FROM THE FILLS.

    fills: the Phase 0 frame (data/phase0/fills_raw.parquet) — per fill the
    venue's own commission_detail_total split, so every number here is read
    off a fill, not a fee table.

      * Tier 0 = the tier the venue reports NOW (get_transaction_summary
        fee_tier): its rates and its usd_to boundary are authoritative.
      * Cheaper tiers = clusters of the observed pct-of-notional rate
        (client_commission / notional, rounded to 0.5bps) among fills whose
        rolling 30d volume is >= tier 0's usd_to. A cluster's lower bound is
        the smallest rolling volume at which that rate was observed (the
        conservative choice: just below it, the dearer tier is charged).
        Maker and taker clusters are paired by overlapping volume range.
      * Rates observed BELOW tier 0's usd_to that differ from the API rate
        (e.g. June's 9bps Intro) are an earlier fee-schedule epoch and are
        dropped — the API's current rate wins inside its own band.
      * Fixed components: medians of venue/clearing/regulatory commission
        per contract over the last `recent_days` of fills, pooled across
        assets (they are asset-independent on every fill).
    """
    import pandas as pd
    F = fills.sort_values("timestamp").copy()
    t = F["timestamp"].to_numpy(); n = F["notional_usd"].to_numpy()
    F["vol30"] = [float(n[(t > ti - 30 * 86400) & (t < ti)].sum()) for ti in t]
    F["rate"] = (F["client_commission"] / F["notional_usd"] * 1e4 * 2).round() / 2
    lo = float(api_tier["usd_to"])
    tiers = [{"min_30d_volume_usd": 0.0,
              "maker_bps": float(api_tier["maker_fee_rate"]) * 1e4,
              "taker_bps": float(api_tier["taker_fee_rate"]) * 1e4,
              "source": f"venue get_transaction_summary fee_tier "
                        f"{api_tier.get('pricing_tier')} (usd_to={lo:g})"}]
    hi = F[F["vol30"] >= lo]
    clusters = {}
    for mk, g0 in hi.groupby("is_maker"):
        for r, g in g0.groupby("rate"):
            clusters.setdefault(bool(mk), []).append(
                (float(r), float(g["vol30"].min()), float(g["vol30"].max()), len(g)))
    for tr, tmin, tmax, tn in sorted(clusters.get(False, []), key=lambda x: x[1]):
        mk = [c for c in clusters.get(True, []) if c[1] <= tmax and c[2] >= tmin]
        mr = mk[0][0] if mk else None
        vmin = min([tmin] + [c[1] for c in mk])
        tiers.append({"min_30d_volume_usd": max(lo, vmin) if len(tiers) == 1 else vmin,
                      "maker_bps": mr if mr is not None else tr,
                      "taker_bps": tr,
                      "source": f"inferred from {tn} taker"
                                f"{' + ' + str(mk[0][3]) + ' maker' if mk else ''} fills "
                                f"at rolling 30d vol {vmin:,.0f}-{max([tmax]+[c[2] for c in mk]):,.0f}"})
    if len(tiers) > 1:
        tiers[1]["min_30d_volume_usd"] = lo
    # Pooled across assets: on every fill the three components are the same
    # per contract whatever the asset, and they DO move over time (the
    # regulatory part went 0.01-0.06 -> 0.00 in October), so the recent
    # window matters more than the per-asset split.
    recent = F[F["timestamp"] >= F["timestamp"].max() - recent_days * 86400]
    ct = recent["size_contracts"]
    pooled = {c: round(float((recent[f"{c}_commission"].fillna(0.0) / ct).median()), 4)
              for c in ("venue", "clearing", "regulatory")}
    comps = {str(a): dict(pooled) for a in sorted(set(F["asset"]))}
    return {"tiers": tiers, "fixed_components_by_asset": comps,
            "fixed_components_source": (f"median per contract over the {len(recent)} fills in the "
                                        f"last {recent_days:g} days of the ledger, pooled across assets")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(REPO, "reports", "phase0_params.json"))
    ap.add_argument("--fills", default=os.path.join(REPO, "data", "phase0", "fills_raw.parquet"))
    ap.add_argument("--fee-tier", default=os.path.join(REPO, "data", "phase0", "fee_tier.json"))
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
    # [P1R-1] tiers + fixed components from the fills — refuse rather than
    # ship a schedule whose tier table was never derived (no defaults).
    if not (os.path.exists(a.fills) and os.path.exists(a.fee_tier)):
        print(f"REFUSING: {a.fills} / {a.fee_tier} missing — the tier table is "
              "derived from the Phase 0 fills and the venue fee_tier (no defaults)")
        return 2
    try:
        import pandas as pd
        with open(a.fee_tier, encoding="utf-8") as fh:
            api_tier = json.load(fh)["fee_tier"]
        sched.update(build_tiers(pd.read_parquet(a.fills), api_tier))
    except (KeyError, TypeError, ValueError, ImportError) as e:
        print(f"REFUSING: tier derivation failed ({type(e).__name__}: {e})")
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

"""[P1-1] Replay the venue fee schedule against the venue's own commissions.

    python -X utf8 scripts/replay_fee_model.py --fills data/phase0/fills_raw.parquet

Three reads, because the tier moves with 30-day volume (Phase 0 §2.1):
  A. CURRENT-TIER fills only (the tier the schedule encodes)   -> the gate:
     sum|model - actual| / sum(actual) < 5%
  B. ALL fills, each priced at the tier rate visible in its own
     client_commission (validates the pct + fixed-per-contract STRUCTURE)
     -> also must be < 5%
  C. ALL fills at the current tier (informational: overcharge expected,
     since history was mostly at cheaper tiers)
Lists every fill whose |error| > $0.02. Exit 0 PASS / 1 FAIL / 2 refusal.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fills", default=os.path.join(REPO, "data", "phase0", "fills_raw.parquet"))
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--tol-pct", type=float, default=5.0)
    a = ap.parse_args(argv)
    try:
        import pandas as pd
    except ImportError:
        print("REFUSING: pandas unavailable"); return 2
    if not os.path.exists(a.fills):
        print(f"REFUSING: {a.fills} missing (run the Phase 0 audit)"); return 2
    from core.cde_fees import load_venue_fee_schedule
    s = load_venue_fee_schedule(a.schedule, reload=True)
    if s is None:
        print("REFUSING: venue fee schedule not loadable"); return 2
    F = pd.read_parquet(a.fills)
    F = F[F.asset.str.upper().isin(list(s.contract_size))].copy()
    if F.empty:
        print("REFUSING: no fills for assets the schedule covers"); return 2
    mk = F.liquidity_indicator.eq("MAKER")
    rate_now = mk.map({True: s.rate_bps_maker, False: s.rate_bps_taker})
    fixed = s.fixed_per_contract_usd * F.size_contracts
    F["model_current_tier"] = F.notional_usd * rate_now / 1e4 + fixed
    own_rate = F.client_commission / F.notional_usd * 1e4      # tier in force at that fill
    F["model_own_tier"] = F.notional_usd * own_rate / 1e4 + fixed
    cur = (own_rate.sub(rate_now).abs() < 1e-6)

    def report(name, sub, col):
        err = (sub[col] - sub.commission)
        rel = err.abs().sum() / sub.commission.sum() * 100
        print(f"{name}: n={len(sub)} sum_actual=${sub.commission.sum():.4f} "
              f"sum_model=${sub[col].sum():.4f} sum|err|/sum(actual)={rel:.2f}%")
        bad = sub[err.abs() > 0.02]
        print(f"   fills with |err| > $0.02: {len(bad)}"
              + (" (showing first 12)" if len(bad) > 12 else ""))
        for _, r in bad.head(12).iterrows():
            print(f"   >$0.02  {r.iso} {r.asset} {r.liquidity_indicator} ct={r.size_contracts:g} "
                  f"actual={r.commission:.4f} model={r[col]:.4f} err={r[col]-r.commission:+.4f} "
                  f"fixed_actual/ct={r.fixed_per_contract_usd:.3f}")
        return rel

    ra = report("A current-tier fills", F[cur], "model_current_tier") if cur.any() else float("nan")
    rb = report("B all fills, own-tier rate", F, "model_own_tier")
    rc = report("C all fills, current tier (informational)", F, "model_current_tier")
    ok = (ra == ra and ra < a.tol_pct) and rb < a.tol_pct
    print(("PASS" if ok else "FAIL") + f": A={ra:.2f}% B={rb:.2f}% (tol {a.tol_pct}%), C={rc:+.2f}% informational")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

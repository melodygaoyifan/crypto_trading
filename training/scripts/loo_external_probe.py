"""[P256] Leave-one-out cleanliness check on the edge probe's `external` group.

The `external` group is the only one clearing the cost bar on all three
assets at 16h (clean-parquet probe 2026-08-10). Two members carry known
concerns: `funding_rate_zscore` is the ex-P247-leak column (now causal, but
the group's clear must SURVIVE removing it entirely — if the clear collapses
without one column, that column IS the finding), and `liq_imbalance` rides
CoinGlass's ~180d history (a coverage confound over a 13k-bar window).

Runs the probe's own walk-forward machinery with the group redefined —
same code path, not a reimplementation (the P172 one-resolver rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import training.scripts.edge_probe as ep  # noqa: E402

FULL = set(ep._EXTERNAL)
VARIANTS = {
    "full": FULL,
    "no_funding": FULL - {"funding_rate_zscore"},
    "no_liq": FULL - {"liq_imbalance"},
    "no_funding_no_liq": FULL - {"funding_rate_zscore", "liq_imbalance"},
}


def main() -> int:
    out = {}
    for name, members in VARIANTS.items():
        # GROUPS["external"] is `lambda c: c in _EXTERNAL` — a closure over
        # the module GLOBAL, so reassigning it redefines the group in place.
        ep._EXTERNAL = members
        out[name] = {}
        for asset in ("BTC", "ETH", "SOL"):
            rep = ep.probe_asset(asset)
            ext = [r for r in rep["results"]
                   if r["group"] == "external" and r["model"] == "ridge"
                   and r["horizon_bars"] == 4]
            r = ext[0] if ext else {}
            out[name][asset] = {k: r.get(k) for k in
                                ("ic", "t", "bps_after_cost_q75",
                                 "required_ic", "clears_bar", "n_oos")}
            print(f"{name:>18} {asset} 16h ridge: IC={r.get('ic')} "
                  f"t={r.get('t')} bps_q75={r.get('bps_after_cost_q75')} "
                  f"clears={r.get('clears_bar')}")
    ep._EXTERNAL = FULL
    dst = REPO / "training" / "reports" / "loo_external_probe_p256.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"report: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""[P393] Macro factor lab EXTENSION — the three factors the P392 literature
stream left unmeasured that have any claim to structure. MEASUREMENT ONLY:
no live path reads this module; it writes
training/reports/macro_factor_ext_p393.json.

SERIES (FRED, observation_start=2020-01-01, key from the repo .env, never
printed; raw obs cached to training/reports/macro_factor_series_p393_ext.json
— a NEW cache file, the P392 cache is never rewritten):
  DGS2          2y Treasury yield — the policy-EXPECTATIONS proxy. The one
                lag finding in the P392 literature stream was expectations
                indices Granger-causing at 3-5 days, and FOMC-day effects
                load on 2y surprises.
  T10Y2Y        10y-2y curve slope — regime/recession context.
  BAMLH0A0HYM2  ICE BofA US High Yield OAS — credit spreads, the classic
                risk-appetite factor. Tested BOTH as the CHANGE and as a
                causal trailing LEVEL z-score (rolling 252-business-day
                window, min 120 obs, trailing-inclusive — depends only on
                values up to and including the stamped day).

EVERY convention is IMPORTED from training/scripts/macro_factor_lab.py
(P172 — one implementation, never re-typed): the B+2-business-day
publication lag (known_from), weekend/absent handling (bd_changes /
obs_to_series: ABSENT never zero-filled, P2), the alignment (align_lead),
the P166 cost bar at the measured CDE RT costs (required_ic / RT_COST_BPS),
the P231 overlap-corrected t (overlap_t via lead_cell), the era convention
(_era_dates over funding_legs_lab.ERAS), the block-bootstrap betas
(contemporaneous_cell), and the verdict function (decide_verdict).

CELLS RUN per asset (BTC/ETH/SOL):
  (a) contemporaneous beta per era for each CHANGE variant (DGS2_chg,
      T10Y2Y_chg, HY_OAS_chg) — same block-bootstrap t;
  (b) LEAD Spearman IC at 1d and 5d for every variant (the three changes
      plus HY_OAS_levelz), same publication-lag convention, BOTH bars
      (|t| >= 2 AND |IC| >= IC_req in the middle AND recent era);
  (c) DGS2 additionally at the 3-5 business-day lag the literature names:
      the change of business day B used from B + {3,4,5} business days
      against the 5d forward return (lag >= the 2-bday publication lag, so
      strictly conservative — no leak by construction).

PRE-COMMITTED VERDICT RULE (P260 — written before the first run; identical
to the P392 rule, applied to the new cells; the run reports these as they
fall, never adjusts them):
  A lead cell (asset, variant, horizon, era) PASSES iff
      n_eff >= 30  AND  |t| >= 2.0  AND  |IC| >= IC_req(era sigma).
  A variant/horizon is TRADEABLE-LEAD for an asset iff its lead cell PASSES
  in BOTH the design (middle) AND validation (recent) eras AND the IC sign
  agrees across the two eras.
  Else it is BETA-ONLY iff its CONTEMPORANEOUS beta has |t| >= 2.0 in BOTH
  design and validation eras with the same beta sign. The HY_OAS_levelz
  variant has no contemporaneous cell of its own (a level z is not a change
  over an interval); its beta context is HY_OAS_chg's — the same underlying
  series. The DGS2 lag-3/4/5 cells use DGS2_chg's beta context.
  Else NOISE.

ANTI-VACUITY (P174), run on the NEW alignment paths too:
  (a) shuffled-factor controls on DGS2_chg AND on HY_OAS_levelz (|IC| ~ 0);
  (b) the PLANTED-lead control at the standard lag (IC ~ 1) AND a planted
      lead constructed at lag_bdays=3 / horizon 5d aligned the same way —
      proving the lag-3 path does not lag a real signal away (P164 family).

Usage:  python -X utf8 training/scripts/macro_factor_ext_p393.py [--refetch]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# [P172] every convention IMPORTED, never re-typed.
from training.funding_legs_lab import ERAS, load_closes  # noqa: E402
from training.scripts.macro_factor_lab import (  # noqa: E402
    ASSETS,
    CACHE_PATH as P392_CACHE_PATH,
    EDGE_MARGIN,
    MIN_N_EFF,
    PUBLICATION_LAG_BDAYS,
    REPORT_DIR,
    RT_COST_BPS,
    T_BAR,
    _era_dates,
    align_lead,
    bd_changes,
    contemporaneous_cell,
    daily_closes,
    decide_verdict,
    fetch_fred_series,
    known_from,
    lead_cell,
    load_fred_key,
    obs_to_series,
    planted_factor,
    required_ic,
    shuffled_control,
    spearman_ic,
)

EXT_CACHE_PATH = REPORT_DIR / "macro_factor_series_p393_ext.json"
EXT_REPORT_PATH = REPORT_DIR / "macro_factor_ext_p393.json"

# series_id -> change kind (all yields/spreads: arithmetic diff in pp).
EXT_SERIES = {
    "DGS2": "diff",
    "T10Y2Y": "diff",
    "BAMLH0A0HYM2": "diff",
}

# Causal trailing level-z window (business observations).
LEVELZ_WINDOW = 252
LEVELZ_MIN_PERIODS = 120

DGS2_LIT_LAGS_BDAYS = (3, 4, 5)   # the literature's 3-5 day Granger lag
DGS2_LIT_HORIZON_D = 5            # vs the 5d forward return


def level_z(level: pd.Series, window: int = LEVELZ_WINDOW,
            min_periods: int = LEVELZ_MIN_PERIODS) -> pd.Series:
    """Causal trailing z of the LEVEL: z_t uses only observations <= t
    (rolling trailing-inclusive window). Stamped at the observation day; the
    publication lag is applied downstream by the imported align_lead exactly
    as for changes. Missing prints are already ABSENT from `level`
    (obs_to_series drops '.'), so nothing here fabricates a value (P2)."""
    lv = level.dropna()
    mu = lv.rolling(window, min_periods=min_periods).mean()
    sd = lv.rolling(window, min_periods=min_periods).std()
    z = (lv - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).dropna()


def load_or_fetch_ext(refetch: bool = False) -> tuple[dict, dict]:
    """Ext-roster fetch: same shape/logic as the lab's loader but against the
    NEW cache file — the P392 cache is never opened for writing here."""
    cache: dict = {}
    if EXT_CACHE_PATH.exists() and not refetch:
        try:
            cache = json.loads(EXT_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    key = load_fred_key()
    series, status = {}, {}
    for sid in EXT_SERIES:
        if sid in cache and cache[sid].get("observations"):
            series[sid] = cache[sid]["observations"]
            status[sid] = "cache"
            continue
        if not key:
            status[sid] = "UNFETCHABLE: FRED_API_KEY absent from .env"
            continue
        try:
            series[sid] = fetch_fred_series(sid, key)
            status[sid] = "fetched"
            cache[sid] = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
                          "observations": series[sid]}
            time.sleep(0.4)  # politeness; FRED allows 120 req/min
        except Exception as e:  # noqa: BLE001 — per-series named gap, never a silent skip (P2)
            status[sid] = f"UNFETCHABLE: {type(e).__name__}: {e}"
    if any(v == "fetched" for v in status.values()):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        EXT_CACHE_PATH.write_text(json.dumps(cache, indent=1),
                                  encoding="utf-8", newline="\n")
    return series, status


def build_variants(levels: dict) -> tuple[dict, dict]:
    """-> (variants: name -> predictor Series, beta_context: name -> the
    CHANGE-variant name whose contemporaneous betas give the variant its
    beta context in the verdict)."""
    variants: dict = {}
    beta_ctx: dict = {}
    if "DGS2" in levels:
        variants["DGS2_chg"] = bd_changes(levels["DGS2"], EXT_SERIES["DGS2"])
        beta_ctx["DGS2_chg"] = "DGS2_chg"
    if "T10Y2Y" in levels:
        variants["T10Y2Y_chg"] = bd_changes(levels["T10Y2Y"],
                                            EXT_SERIES["T10Y2Y"])
        beta_ctx["T10Y2Y_chg"] = "T10Y2Y_chg"
    if "BAMLH0A0HYM2" in levels:
        variants["HY_OAS_chg"] = bd_changes(levels["BAMLH0A0HYM2"],
                                            EXT_SERIES["BAMLH0A0HYM2"])
        beta_ctx["HY_OAS_chg"] = "HY_OAS_chg"
        variants["HY_OAS_levelz"] = level_z(levels["BAMLH0A0HYM2"])
        beta_ctx["HY_OAS_levelz"] = "HY_OAS_chg"
    return variants, beta_ctx


CHANGE_VARIANTS = ("DGS2_chg", "T10Y2Y_chg", "HY_OAS_chg")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args(argv)

    series_obs, status = load_or_fetch_ext(refetch=args.refetch)
    print("[P393] series status:")
    for sid in EXT_SERIES:
        print(f"  {sid:14s} {status.get(sid)}")

    levels = {sid: obs_to_series(series_obs[sid])
              for sid in EXT_SERIES if sid in series_obs}
    variants, beta_ctx = build_variants(levels)

    report: dict = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "publication_lag_bdays": PUBLICATION_LAG_BDAYS,
            "levelz_window_bdays": LEVELZ_WINDOW,
            "dgs2_lit_lags_bdays": list(DGS2_LIT_LAGS_BDAYS),
            "rt_cost_bps": RT_COST_BPS,
            "edge_margin": EDGE_MARGIN,
            "era_convention": "funding_legs_lab.ERAS bar bands mapped to dates per asset (imported)",
            "conventions": "imported from macro_factor_lab (P172), never re-typed",
            "verdict_rule": "pre-committed in module docstring before first run (identical to P392)",
            "coverage_notes": {
                sid: (f"history starts {s.index[0].date()} (< requested "
                      f"{'2020-01-01'}) — FRED serves a truncated window for "
                      f"this series; earlier eras are ABSENT, never inferred")
                for sid, s in
                ((k, obs_to_series(series_obs[k])) for k in EXT_SERIES
                 if k in series_obs)
                if len(s) and s.index[0] > pd.Timestamp("2020-01-15", tz="UTC")
            },
        },
        "series_status": status,
        "contemporaneous": {}, "lead": {}, "dgs2_lag": {},
        "controls": {}, "verdicts": {},
    }

    for asset in ASSETS:
        c4 = load_closes(asset)
        dc = daily_closes(c4)
        eras = _era_dates(c4)
        cost = RT_COST_BPS[asset]

        # -- (a) contemporaneous betas per era, CHANGE variants only
        cont_a: dict = {}
        for name in CHANGE_VARIANTS:
            if name not in variants:
                continue
            cont_a[name] = {}
            for era, (s, e) in eras.items():
                cont_a[name][era] = contemporaneous_cell(variants[name], dc, s, e)
        report["contemporaneous"][asset] = cont_a

        # -- (b) lead cells, 1d and 5d, all variants
        lead_a: dict = {}
        for name, pred in variants.items():
            lead_a[name] = {}
            for hz in (1, 5):
                al = align_lead(pred, dc, hz)
                lead_a[name][f"{hz}d"] = {}
                for era, (s, e) in eras.items():
                    lead_a[name][f"{hz}d"][era] = lead_cell(
                        pred, dc, hz, cost, s, e, aligned=al)
                lead_a[name][f"{hz}d"]["pooled"] = lead_cell(
                    pred, dc, hz, cost, eras["pre_design"][0], None, aligned=al)
        report["lead"][asset] = lead_a

        # -- (c) DGS2 at the literature's 3-5 bday lag vs 5d fwd
        lag_a: dict = {}
        if "DGS2_chg" in variants:
            for lag in DGS2_LIT_LAGS_BDAYS:
                al = align_lead(variants["DGS2_chg"], dc, DGS2_LIT_HORIZON_D,
                                lag_bdays=lag)
                cell_k = f"lag{lag}bd_{DGS2_LIT_HORIZON_D}d"
                lag_a[cell_k] = {}
                for era, (s, e) in eras.items():
                    lag_a[cell_k][era] = lead_cell(
                        variants["DGS2_chg"], dc, DGS2_LIT_HORIZON_D, cost,
                        s, e, lag_bdays=lag, aligned=al)
                lag_a[cell_k]["pooled"] = lead_cell(
                    variants["DGS2_chg"], dc, DGS2_LIT_HORIZON_D, cost,
                    eras["pre_design"][0], None, lag_bdays=lag, aligned=al)
        report["dgs2_lag"][asset] = lag_a

        # -- controls on the new alignment paths (P174)
        ctl: dict = {}
        pf = planted_factor(dc)
        al = align_lead(pf, dc, 1)
        ctl["planted_lead"] = {
            "n": int(len(al)),
            "ic": round(spearman_ic(al["x"].to_numpy(), al["fwd"].to_numpy()), 4)}
        pf3 = planted_factor(dc, horizon_days=DGS2_LIT_HORIZON_D, lag_bdays=3)
        al3 = align_lead(pf3, dc, DGS2_LIT_HORIZON_D, lag_bdays=3)
        ctl["planted_lead_lag3_5d"] = {
            "n": int(len(al3)),
            "ic": round(spearman_ic(al3["x"].to_numpy(), al3["fwd"].to_numpy()), 4)}
        if "DGS2_chg" in variants:
            ctl["shuffled_DGS2_chg"] = shuffled_control(variants["DGS2_chg"], dc)
        if "HY_OAS_levelz" in variants:
            ctl["shuffled_HY_OAS_levelz"] = shuffled_control(
                variants["HY_OAS_levelz"], dc)
        report["controls"][asset] = ctl
        for ck in ("planted_lead", "planted_lead_lag3_5d"):
            if ctl[ck]["ic"] < 0.9:
                print(f"[P393][WARN] {asset}: {ck} IC {ctl[ck]['ic']} < 0.9 "
                      f"— alignment suspect (P174)")

        # -- verdicts (imported pre-committed rule)
        verd_a: dict = {}
        for name in variants:
            bctx = cont_a.get(beta_ctx[name], {})
            for hz in (1, 5):
                verd_a[f"{name}_{hz}d"] = decide_verdict(
                    lead_a[name][f"{hz}d"]["design"],
                    lead_a[name][f"{hz}d"]["validation"],
                    bctx.get("design", {}), bctx.get("validation", {}))
        for cell_k, cells in lag_a.items():
            verd_a[f"DGS2_chg_{cell_k}"] = decide_verdict(
                cells["design"], cells["validation"],
                cont_a.get("DGS2_chg", {}).get("design", {}),
                cont_a.get("DGS2_chg", {}).get("validation", {}))
        report["verdicts"][asset] = verd_a

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXT_REPORT_PATH.write_text(json.dumps(report, indent=1, default=str),
                               encoding="utf-8", newline="\n")

    # ---- printed table ----
    def _f(v, w=7):
        return f"{v:>{w}}" if v is not None else " " * (w - 4) + "  — "

    print("\n[P393] CONTEMPORANEOUS betas (same-day, per era)")
    print(f"{'asset':6s}{'variant':14s}{'era':12s}{'n':>6s}{'beta':>10s}{'t':>7s}{'corr':>7s}{'R2':>6s}")
    for asset in ASSETS:
        for name in CHANGE_VARIANTS:
            if name not in report["contemporaneous"][asset]:
                continue
            for era in ERAS:
                c = report["contemporaneous"][asset][name][era]
                if "beta" not in c:
                    continue
                print(f"{asset:6s}{name:14s}{era:12s}{c['n']:>6d}"
                      f"{c['beta']:>10.4f}{c['t']:>7.2f}{c['corr']:>7.3f}{c['r2']:>6.3f}")

    print("\n[P393] LEAD cells (publication-lagged; PASS needs n_eff>=30, |t|>=2, |IC|>=IC_req)")
    print(f"{'asset':6s}{'variant':14s}{'hz':4s}{'era':12s}{'n':>5s}{'IC':>8s}{'t':>7s}"
          f"{'IC_req':>8s}{'edge':>7s}{'sigma':>7s}{'pass':>6s}")
    for asset in ASSETS:
        for name in report["lead"][asset]:
            for hz in ("1d", "5d"):
                for era in list(ERAS) + ["pooled"]:
                    c = report["lead"][asset][name][hz][era]
                    if "ic" not in c:
                        continue
                    print(f"{asset:6s}{name:14s}{hz:4s}{era:12s}{c['n']:>5d}"
                          f"{_f(c['ic'], 8)}{_f(c['t'])}{_f(c['ic_required'], 8)}"
                          f"{_f(c['implied_edge_bps'])}{c['sigma_fwd_bps']:>7.0f}"
                          f"{'PASS' if c['passes'] else '.':>6s}")

    print("\n[P393] DGS2 literature-lag cells (change of day B used from B+lag bdays, vs 5d fwd)")
    for asset in ASSETS:
        for cell_k, cells in report["dgs2_lag"][asset].items():
            for era in list(ERAS) + ["pooled"]:
                c = cells.get(era, {})
                if "ic" not in c:
                    continue
                print(f"{asset:6s}{'DGS2_chg':14s}{cell_k:12s}{era:12s}{c['n']:>5d}"
                      f"{_f(c['ic'], 8)}{_f(c['t'])}{_f(c['ic_required'], 8)}"
                      f"{'PASS' if c['passes'] else '.':>6s}")

    print("\n[P393] controls")
    for asset in ASSETS:
        print(f"  {asset}: {report['controls'][asset]}")

    print("\n[P393] VERDICTS (pre-committed rule, imported)")
    for asset in ASSETS:
        for k, v in report["verdicts"][asset].items():
            if v != "NOISE":
                print(f"  {asset} {k}: {v}")
    noise_n = sum(1 for a in ASSETS
                  for v in report["verdicts"][a].values() if v == "NOISE")
    print(f"  (NOISE cells: {noise_n})")
    print(f"\n[P393] report -> {EXT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

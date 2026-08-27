"""[P417] Historical validation of the fusion-conviction sizing channel.

THE QUESTION (operator, 2026-08-27): between (a) turning
`fusion_conviction_to_sleeve` OFF (pure direction holding) and (b) keeping it
with hysteresis/deadband, decide FROM HISTORICAL DATA. The P416 audit measured
the channel's live flap (1.09<->0.58 within minutes) as a fee leak; this lab
asks whether the channel's DE-RISK CONTENT earns more than its resize fees
over 6.6 years, on the deployed book, at honest per-leg costs.

WHAT IS RECONSTRUCTIBLE, honestly stated:
  * The MACRO component (the only validated content, P392/P393): the P393
    adverse-restricts cap map driven by the GCI caution-set classification,
    reconstructed from the P392/P393 FRED caches (VIXCLS raw for the >30
    rule; DGS10 z30>1.5 yield spike; DTWEXBGS z30>1.5 as the dollar-breakout
    PROXY -- live uses is_breakout_up, labeled approximation; HY_OAS z30>1.5
    where the ~3y series exists). ETF-outflow caution is NOT reconstructible
    pre-2024 and contributes nothing (P2 -- absence is never fabricated
    stress). Publication lag: each daily value usable from B+2 business days
    (the P392 convention).
  * The FAST flap (HTF/ADVISE alignment) is NOT reconstructible (agent
    outputs were never stored over this window; htf_trend_direction has no
    producer). It is modeled SYNTHETICALLY from the observed live
    distribution (census 2026-08-27: ~15% of ticks dip to 0.55-0.90 for one
    tick, else >=1.0 clamped) -- 10 seeds, reported as mean/min/max. The
    synthetic part bounds the FEE cost of flap under each policy; it carries
    no alpha by construction (iid, direction-independent), which is exactly
    what the measured IC of the fast contributors says (noise).

VARIANTS (sizing overlay on the DEPLOYED book direction, P172 chassis):
  A: conviction == 1 (option (a): channel OFF; pure direction holding).
  B: raw conviction applied every tick (the pre-P416 live behavior).
  C: conviction with target-level 2-tick persistence (the LIVE post-P416
     behavior: a same-direction resize needs the same quantized target on 2
     consecutive ticks; entries/exits/flips instant).
  D: deadband-0.5 + persistence (option (b) strict form: only conviction
     < 0.5 -- i.e. CRISIS 0.4 -- may de-risk, 2-tick confirmed).

Position realism: contracts quantized at the live book's base
{BTC:2, ETH:6, SOL:2}; short legs get the P393 relief min(cap*1.5, 1.0);
costs are the chassis's price-dependent per-leg CDE cost on |position
changes|; funding carry included (P245).

PRE-COMMITTED VERDICT RULE (written before the first run):
  * The channel STAYS ON (in its P416-damped form C) iff C's 3-asset summed
    NET >= A's over the full window AND C >= A in at least 2 of 3 eras.
  * If A beats C on both -> the recommendation is (a): set
    `fusion_conviction_to_sleeve: false`.
  * D is reported alongside; if D beats both A and C on the same rule, the
    recommendation is tightening to the deadband form.
Robustness: the verdict must hold on the MEAN across flap seeds; if the seed
min/max straddle the verdict, report NOT SETTLED rather than pick a side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from training.funding_legs_lab import (  # noqa: E402
    ERAS, build_positions, load_closes, load_funding_daily, pnl)

REPORT = REPO / "training" / "reports" / "conviction_channel_lab_p417.json"
CACHE_MAIN = REPO / "training" / "reports" / "macro_factor_series_p392.json"
CACHE_EXT = REPO / "training" / "reports" / "macro_factor_series_p393_ext.json"

BASE_CT = {"BTC": 2, "ETH": 6, "SOL": 2}
CAP = {"CRISIS": 0.4, "DOLLAR_BREAKOUT": 0.8, "RISK_OFF": 0.7, "CALM": 1.0}
FLAP_P = 0.15          # observed ~4/26 ticks below 1.0
FLAP_LO, FLAP_HI = 0.55, 0.90
SEEDS = list(range(10))
PERSIST = 2


def _series(cache: dict, name: str) -> pd.Series:
    obs = cache.get(name, {}).get("observations", [])
    rows = [(r["date"], r["value"]) for r in obs
            if isinstance(r, dict) and r.get("value") not in (None, ".")]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): float(v) for d, v in rows}).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _z30(s: pd.Series) -> pd.Series:
    m = s.rolling(30, min_periods=20).mean()
    sd = s.rolling(30, min_periods=20).std(ddof=1)
    return (s - m) / sd.replace(0.0, np.nan)


def build_macro_cap() -> pd.Series:
    """Daily P393 cap series from cached FRED data, B+2-lagged."""
    main = json.loads(CACHE_MAIN.read_text(encoding="utf-8"))
    ext = (json.loads(CACHE_EXT.read_text(encoding="utf-8"))
           if CACHE_EXT.exists() else {})
    vix = _series(main, "VIXCLS")
    dgs10 = _series(main, "DGS10")
    dxy = _series(main, "DTWEXBGS")
    hy = _series(ext, "BAMLH0A0HYM2")
    if vix.empty or dgs10.empty or dxy.empty:
        raise SystemExit("REFUSING: FRED cache missing core series (P199)")

    idx = vix.index.union(dgs10.index).union(dxy.index)
    yz = _z30(dgs10).reindex(idx).ffill()
    dz = _z30(dxy).reindex(idx).ffill()
    hz = _z30(hy).reindex(idx).ffill() if not hy.empty else pd.Series(
        np.nan, index=idx)
    vv = vix.reindex(idx).ffill()

    caps = {}
    for ts in idx:
        yield_c = bool(yz.get(ts, np.nan) > 1.5)
        dxy_c = bool(dz.get(ts, np.nan) > 1.5)      # PROXY for is_breakout_up
        hy_c = bool(hz.get(ts, np.nan) > 1.5)        # absent -> False (P2)
        n = int(yield_c) + int(dxy_c) + int(hy_c)    # etf caution: absent
        v = vv.get(ts, np.nan)
        if n >= 2 or (np.isfinite(v) and v > 30):
            caps[ts] = CAP["CRISIS"]
        elif dxy_c:
            caps[ts] = CAP["DOLLAR_BREAKOUT"]
        elif n >= 1:
            caps[ts] = CAP["RISK_OFF"]
        else:
            caps[ts] = CAP["CALM"]
    cap = pd.Series(caps).sort_index()
    # publication lag: value of day d usable from d + 2 business days
    cap.index = cap.index + pd.offsets.BDay(2)
    # the chassis's 4H index is tz-aware UTC; FRED dates are naive calendar
    # days -- localize so reindex can compare (P40/P97 family)
    cap.index = cap.index.tz_localize("UTC")
    return cap[~cap.index.duplicated(keep="last")]


def conviction_for(book: pd.Series, cap_at: pd.Series,
                   flap: np.ndarray | None) -> pd.Series:
    """Per-tick conviction: macro cap (long restricted, short relieved,
    P393 LAYER-6 semantics) x optional synthetic flap."""
    out = np.ones(len(book))
    caps = cap_at.to_numpy(dtype=float)
    b = book.to_numpy(dtype=float)
    for i in range(len(b)):
        c = caps[i] if np.isfinite(caps[i]) else 1.0
        if b[i] > 0:
            conv = c
        elif b[i] < 0:
            conv = min(c * 1.5, 1.0)
        else:
            conv = 1.0
        if flap is not None:
            conv = min(conv, flap[i]) if flap[i] < 1.0 else conv
        out[i] = conv
    return pd.Series(out, index=book.index)


def quantize(book: pd.Series, conv: pd.Series, base: int) -> pd.Series:
    tgt = np.round(base * book.to_numpy() * conv.to_numpy())
    return pd.Series(tgt / base, index=book.index)


def persist_targets(q: pd.Series, book: pd.Series, ticks: int) -> pd.Series:
    """Mirror the P416 sleeve rule at the lab level: a same-direction resize
    must propose the SAME quantized target on `ticks` consecutive bars.
    Entries from flat, flattens and sign flips execute immediately."""
    qv = q.to_numpy()
    out = np.empty_like(qv)
    cur = 0.0
    pend_t, streak = None, 0
    for i, t in enumerate(qv):
        if cur != 0.0 and t != 0.0 and (t > 0) == (cur > 0) and t != cur:
            if pend_t is not None and t == pend_t:
                streak += 1
            else:
                pend_t, streak = t, 1
            if streak >= ticks:
                cur = t
                pend_t, streak = None, 0
        else:
            pend_t, streak = None, 0
            cur = t
        out[i] = cur
    return pd.Series(out, index=q.index)


def era_slice(df: pd.DataFrame, pos_df: pd.DataFrame, name: str) -> float:
    lo, hi = ERAS[name]
    idx = pos_df["i"]
    mask = (idx >= lo) & ((idx < hi) if hi else True)
    return float(df.loc[mask.reindex(df.index).fillna(False), "net"].sum())


def main() -> int:
    cap_daily = build_macro_cap()
    results: dict = {"caveats": [
        "dxy_breakout reconstructed as DTWEXBGS z30>1.5 (live uses "
        "is_breakout_up) -- labeled PROXY",
        "etf_outflow caution not reconstructible pre-2024 -> contributes "
        "nothing (P2)",
        "fast flap is SYNTHETIC (iid, alpha-free by construction), bounds "
        "the FEE side only; 10 seeds",
        "additive per-bar net sums, not compounded",
    ], "assets": {}}

    grand = {"A": 0.0, "B": [], "C": [], "D": 0.0}
    era_tot: dict = {v: {e: 0.0 for e in ERAS} for v in "ACD"}
    # per-seed 3-ASSET-SUM accumulators (a flat per-(asset,seed) list would
    # average to a per-asset figure and get compared against 3-asset era
    # totals -- a x3 bias against B/C, caught on the first run)
    era_bc: dict = {v: {e: {sd: 0.0 for sd in SEEDS} for e in ERAS}
                    for v in "BC"}

    for asset in ("BTC", "ETH", "SOL"):
        closes = load_closes(asset)
        funding = load_funding_daily(asset)
        posd = build_positions(asset, closes, funding)
        # [P420] the FULL window includes the validation era; that read drove
        # the P417 flip and was unledgered. Record it (P332/P382).
        try:
            from training.splits import record_window_usage
            _prior = record_window_usage(
                "conviction_channel_lab:p417", asset, ERAS["validation"][0],
                int(posd["i"].iloc[-1]) + 1,
                "validation:conviction-channel 6.6y verdict incl. the "
                "validation era (drove the P417 channel-OFF flip)")
            if _prior:
                print(f"[WINDOW-LEDGER] {asset}: validation window already read "
                      f"by {_prior} other experiment(s) — discount (P260)")
        except Exception as e:  # noqa: silent-swallow — surfaced, never blocks the lab
            print(f"[WINDOW-LEDGER] WARNING: could not record {asset} "
                  f"({type(e).__name__}: {e})")
        book = posd["book"]
        cap_at = pd.Series(
            cap_daily.reindex(book.index, method="ffill").to_numpy(),
            index=book.index)
        base = BASE_CT[asset]

        conv_macro = conviction_for(book, cap_at, None)
        # D: deadband -- only conviction < 0.5 may act
        conv_d = conv_macro.where(conv_macro < 0.5, 1.0)

        qa = quantize(book, pd.Series(1.0, index=book.index), base)
        qd = persist_targets(quantize(book, conv_d, base), book, PERSIST)

        pnl_a = pnl(asset, qa, closes, funding)
        pnl_d = pnl(asset, qd, closes, funding)

        row = {"A_net": float(pnl_a["net"].sum()),
               "D_net": float(pnl_d["net"].sum()),
               "A_cost": float(pnl_a["cost"].sum()),
               "D_cost": float(pnl_d["cost"].sum())}
        grand["A"] += row["A_net"]
        grand["D"] += row["D_net"]
        for e in ERAS:
            era_tot["A"][e] += era_slice(pnl_a, posd, e)
            era_tot["D"][e] += era_slice(pnl_d, posd, e)

        b_nets, c_nets, b_costs, c_costs = [], [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            flap = np.where(rng.random(len(book)) < FLAP_P,
                            rng.uniform(FLAP_LO, FLAP_HI, len(book)), 1.0)
            conv = conviction_for(book, cap_at, flap)
            qb = quantize(book, conv, base)
            qc = persist_targets(qb, book, PERSIST)
            pb = pnl(asset, qb, closes, funding)
            pc = pnl(asset, qc, closes, funding)
            b_nets.append(float(pb["net"].sum()))
            c_nets.append(float(pc["net"].sum()))
            b_costs.append(float(pb["cost"].sum()))
            c_costs.append(float(pc["cost"].sum()))
            for e in ERAS:
                era_bc["B"][e][seed] += era_slice(pb, posd, e)
                era_bc["C"][e][seed] += era_slice(pc, posd, e)
        row.update({
            "B_net_mean": float(np.mean(b_nets)),
            "B_net_range": [float(min(b_nets)), float(max(b_nets))],
            "C_net_mean": float(np.mean(c_nets)),
            "C_net_range": [float(min(c_nets)), float(max(c_nets))],
            "B_cost_mean": float(np.mean(b_costs)),
            "C_cost_mean": float(np.mean(c_costs)),
        })
        grand["B"].append(b_nets)
        grand["C"].append(c_nets)
        results["assets"][asset] = row
        print(f"{asset}: A={row['A_net']:+.3f} "
              f"B={row['B_net_mean']:+.3f} C={row['C_net_mean']:+.3f} "
              f"D={row['D_net']:+.3f} | cost A={row['A_cost']:.3f} "
              f"B={row['B_cost_mean']:.3f} C={row['C_cost_mean']:.3f} "
              f"D={row['D_cost']:.3f}")

    b_sum = np.sum(np.array(grand["B"]), axis=0)   # per-seed 3-asset sums
    c_sum = np.sum(np.array(grand["C"]), axis=0)
    summary = {
        "A_total": grand["A"],
        "B_total_mean": float(b_sum.mean()),
        "B_total_range": [float(b_sum.min()), float(b_sum.max())],
        "C_total_mean": float(c_sum.mean()),
        "C_total_range": [float(c_sum.min()), float(c_sum.max())],
        "D_total": grand["D"],
        "eras_A": era_tot["A"], "eras_D": era_tot["D"],
        "eras_C_mean": {e: float(np.mean(list(era_bc["C"][e].values())))
                        for e in ERAS},
        "eras_B_mean": {e: float(np.mean(list(era_bc["B"][e].values())))
                        for e in ERAS},
    }

    # pre-committed verdict
    c_wins_full = summary["C_total_mean"] >= summary["A_total"]
    c_era_wins = sum(summary["eras_C_mean"][e] >= era_tot["A"][e]
                     for e in ERAS)
    straddle = ((c_sum.min() < grand["A"]) != (c_sum.max() < grand["A"]))
    if straddle:
        verdict = "NOT_SETTLED_BY_SEEDS"
    elif c_wins_full and c_era_wins >= 2:
        verdict = "KEEP_CHANNEL_P416_DAMPED"
    else:
        verdict = "TURN_CHANNEL_OFF"
    d_wins = (summary["D_total"] >= summary["A_total"]
              and summary["D_total"] >= summary["C_total_mean"]
              and sum(era_tot["D"][e] >= era_tot["A"][e] for e in ERAS) >= 2)
    summary["verdict"] = verdict
    summary["D_alternative_wins"] = bool(d_wins)
    results["summary"] = summary
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

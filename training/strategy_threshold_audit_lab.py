"""[P370] Strategy/decision-layer threshold audit — item 2 of the operator's
"any threshold: use backtest to find if they make sense" pass, for the layer
the risk-control audit (training/risk_control_audit_lab.py, P369) deliberately
left out. Same data (six years of hourly closes), same eras, same measured
CDE round-trip costs, same discipline: EVERY verdict criterion is written in
this docstring BEFORE the first number is read (P297/P340). A threshold judged
after seeing its result is selection, not evidence.

FOUR THRESHOLDS, FOUR PRE-COMMITTED RULES
==========================================

(a) SMA LOOKBACK = 200 (the "hold" leg: long iff close > SMA, flat otherwise)
    Sweep N in {100,150,200,250,300} 4H bars. Two expressions are scored,
    because they are NOT the same book (P307c: the book's bull cell agrees
    with close>SMA200 on only 83-84% of bars):
      * PURE      : long iff close > SMA_N                    (P262-certified rule)
      * BOOK-BULL : long iff close > SMA_N AND mom(540) > 0   (the live ETH/SOL
                    book's hold leg and BTC's bull leg — regime_book_shadow)
    Scored on net-after-cost return (additive per-4H-bar sum, % of notional),
    annualised Sharpe from 4H bars, max drawdown of the additive curve, round
    trips/yr, and per-era net (2020-22 / 2023-24 / 2025-26).
    RULE: a lookback DOMINATES 200 iff it beats 200's net in EVERY era on
    that asset (era-fragility disqualifies a winner that is carried by one
    era, P243/P244). Then:
      (i)   close to optimal  — nothing dominates 200 on that asset AND
            200's net is within 20% of the sweep's best net;
      (ii)  arbitrary but harmless — something dominates 200 but by less
            than the asset's measured round-trip cost per year of turnover
            (i.e. the gain is inside the noise a cost re-measurement moves), or
            the sweep is a plateau (best-vs-worst spread < 20% of best);
      (iii) wrong, costs money — a lookback dominates 200 on that asset by
            more than that, and the SAME lookback dominates on at least two
            of the three assets (one asset alone is a fit, P262's transfer
            rule). The number reported is the six-year net gap.

(b) ALPHA-GATE FRICTION MULTIPLIER (NORMAL_MULTIPLIER=1.10) AND
    ROUND_TRIP_LEGS=2.0 (defense/constitution.py)
    Both sides of the live gate are per-asset CONSTANTS: asserted alpha =
    seat alpha (era-median, core/seat_alpha.py) x the frozen ALPHA-FEEDBACK
    haircut 0.75 (P325 gap 6 — hit_rate stuck at its 0.5 initialiser), and
    threshold = legs x per-leg cost x multiplier x regime-gate-mult (+ hold).
    So the gate is an ASSET-LEVEL on/off switch, not a per-trade filter, and
    the only question a multiplier can answer is WHICH assets it switches off.
    Grid: multiplier in {1.0,1.10,1.25,1.5,2.0} x legs {1,2} x regime mult
    {1.0 (neutral), 1.155 (the live NEUTRAL_DRIFT x VOL_COMPRESSED stack,
    P320b)}, with and without the 0.75 haircut.
    RULE: the multiplier is DOING SOMETHING USEFUL iff, at 1.10, it switches
    OFF an asset whose replayed book (a) is net-NEGATIVE after cost in the
    majority of eras, AND switches ON every asset whose book is net-positive
    in the majority of eras — i.e. the constant agrees with the backtest's
    verdict per asset. It is JUST SUBTRACTING iff no asset's verdict changes
    anywhere in [1.0, 2.0] (the 10% is invisible). It is WRONG iff it
    excludes an asset whose book is net-positive in every era, or admits one
    that is net-negative in every era. ROUND_TRIP_LEGS=2 is CORRECT by
    construction (P167: a position is opened and closed) and is scored only
    to show what legs=1 would admit.

(c) SLEEVE TARGET FRACTION 0.15 x 3 (coinbase_target_fraction_by_asset) and
    post_leverage_caps {BTC .25, ETH .25, SOL .20}
    Vol-parity: fraction_i proportional to 1/sigma_i with the SAME total
    budget as today (0.45 = 3 x 0.15, under the 0.50 net cap). sigma is the
    annualised vol of the asset's 4H returns WHILE THE BOOK IS LONG (the risk
    the sleeve actually carries), measured over six years and per era.
    RULE: flat 0.15 is
      (i)   close to optimal  — every vol-parity fraction is within +/-15% of 0.15;
      (ii)  arbitrary but harmless — some fraction is outside +/-15% but the
            largest asset's share of total book risk is under 45% (no asset
            carries more than ~1.35x its equal share) and nothing exceeds caps;
      (iii) wrong (on a RISK basis; sizing has no "expected PnL" to cost) —
            the vol-parity fraction for some asset exceeds its post_leverage
            cap, OR one asset carries >45% of the three-asset book risk at
            flat 0.15, OR the parity fractions are era-unstable (the ranking
            of sigma changes across eras, so no static fraction is right).

(d) SEAT-ALPHA STATISTIC (era-median vs era-min vs mean; P321 chose median)
    Two tables: the RECORDED lab table (core/seat_alpha.REGIMEBOOK_ALPHA_BY_ERA,
    index-based eras on the FULL book incl. BTC funding legs) and THIS lab's
    calendar-era gross bps per round trip for the PURE and BOOK-BULL hold
    legs at SMA200 (= 2 x gross / unit turnover, an opening position is not
    turnover — the P326 convention). For each statistic: which assets pass
    the live threshold (with and without the 0.75 haircut).
    RULE: the statistic is (i) close to optimal iff the set of assets it
    admits equals the set whose replayed book is net-positive after cost in
    the majority of eras; (ii) harmless iff the sets differ only on an asset
    whose book is within one round-trip cost of zero; (iii) wrong iff it
    admits an asset negative in the majority of eras or excludes one positive
    in every era. The three statistics are judged against the SAME backtest
    so the choice becomes a measured question rather than a preference.

WHAT THE HOURLY REPLAY CANNOT SETTLE (stated before the numbers)
  * BTC's funding legs (bear funding_short / peace contrarian) are NOT
    replayed here — they need the causal daily funding z and are scored by
    training/funding_legs_lab.py. The SMA sweep therefore measures the HOLD
    leg only; changing the lookback also re-labels bear/peace cells for the
    funding legs, which this lab does not price.
  * The live threshold carries a regime-gate multiplier (~1.155 observed,
    P320b) and a venue-true hold cost (~1bp) that depend on the live GMM and
    live funding — replayed as the two constants above, not as a series.
  * The 0.75 ALPHA-FEEDBACK haircut is a frozen initialiser, not a
    measurement; it is replayed as the constant it is.
  * Execution is at the 4H close with no slippage beyond the measured RT
    cost; contract rounding (P274: floor to whole nano contracts) is ignored
    for (c) — fractions are judged in risk space, not in contract space.
  * Costs are the measured CDE round trip (P315/P334: a PERCENTAGE of
    notional, price-invariant), charged per leg on |delta position|.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.seat_alpha import (REGIMEBOOK_ALPHA_BY_ERA,          # noqa: E402
                             REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP)

RAW = REPO / "training" / "training_data" / "raw"
ASSETS = ("BTC", "ETH", "SOL")
# [P315/P334] measured CDE round trip, bps (same constants as the P369 lab)
COST_RT_BPS = {"BTC": 27.7, "ETH": 44.0, "SOL": 41.0}
ERAS = {"2020-22": ("2020-01-01", "2023-01-01"),
        "2023-24": ("2023-01-01", "2025-01-01"),
        "2025-26": ("2025-01-01", "2027-01-01")}
BARS_PER_YEAR_4H = 6 * 365.25
SMA_SWEEP = (100, 150, 200, 250, 300)
MOM_W = 540                                   # regime_book_shadow.MOM_W
LIVE_SMA = 200

# live gate constants (defense/constitution.py, core/seat_alpha.py, P320b, P325)
NORMAL_MULTIPLIER = 1.10
ROUND_TRIP_LEGS = 2.0
PERF_FACTOR_FROZEN = 0.75            # 0.5 + 0.5*hit_rate, hit_rate stuck at 0.5
REGIME_MULT_LIVE = 1.155             # NEUTRAL_DRIFT 1.10 x VOL_COMPRESSED 1.05
HOLD_COST_BPS = 1.0                  # venue-true funding hold, ~1bp today (P291b)
LIVE_FRACTION = {"BTC": 0.15, "ETH": 0.15, "SOL": 0.15}
POST_LEVERAGE_CAPS = {"BTC": 0.25, "ETH": 0.25, "SOL": 0.20}
CONTRACT_SIZE = {"BTC": 0.01, "ETH": 0.1, "SOL": 5.0}
SLEEVE_EQUITY_USD = 10_800.0         # P355/P356 [NAV-LIVE] reading


# ------------------------------------------------------------------ data --
def load_4h(asset: str) -> pd.DataFrame:
    """4H close series from the hourly parquet: the close at hours 0/4/8/...
    is the 4H decision-boundary close (what the live loop reads at its tick)."""
    df = pd.read_parquet(RAW / f"{asset}_60m.parquet")[["timestamp", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"].dt.hour % 4 == 0].reset_index(drop=True)
    df["ret"] = df["close"].pct_change().fillna(0.0)
    return df


# ---------------------------------------------------------------- engine --
def positions(df: pd.DataFrame, sma_n: int, variant: str) -> np.ndarray:
    c = df["close"]
    above = (c > c.rolling(sma_n).mean()).to_numpy()
    if variant == "pure":
        want = above
    elif variant == "book":
        mom_up = (c / c.shift(MOM_W) - 1.0 > 0).to_numpy()
        want = above & mom_up
    else:
        raise ValueError(variant)
    warm = max(sma_n, MOM_W if variant == "book" else 0)
    pos = np.where(np.arange(len(df)) >= warm, want.astype(float), 0.0)
    return pos


def book_pnl(df: pd.DataFrame, pos: np.ndarray, asset: str):
    """Per-4H-bar gross and net PnL (fraction of notional) for a target
    position series decided at bar t and held over t -> t+1."""
    ret = df["ret"].to_numpy(float)
    held = np.concatenate([[0.0], pos[:-1]])          # position over bar t
    gross = held * ret
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * (COST_RT_BPS[asset] / 2.0) * 1e-4    # per leg
    return gross, gross - cost, dpos


def metrics(df: pd.DataFrame, gross: np.ndarray, net: np.ndarray,
            dpos: np.ndarray) -> dict:
    ts = df["timestamp"]
    years = len(df) / BARS_PER_YEAR_4H
    cum = np.cumsum(net)
    dd = cum - np.maximum.accumulate(cum)
    sd = net.std()
    out = {"net_pct": round(float(net.sum() * 100), 1),
           "gross_pct": round(float(gross.sum() * 100), 1),
           "sharpe_ann": round(float(net.mean() / sd * np.sqrt(BARS_PER_YEAR_4H)), 2)
           if sd > 0 else 0.0,
           "max_dd_pct": round(float(dd.min() * 100), 1),
           "round_trips_per_yr": round(float(dpos.sum() / 2.0 / years), 1),
           "time_long_pct": None,          # filled by the caller from pos
           "eras": {}}
    for name, (a, b) in ERAS.items():
        m = ((ts >= a) & (ts < b)).to_numpy()
        if m.sum() < 6 * 30:
            continue
        g, n_, t = gross[m], net[m], dpos[m]
        turn = t.sum()
        out["eras"][name] = {
            "net_pct": round(float(n_.sum() * 100), 1),
            "gross_pct": round(float(g.sum() * 100), 1),
            "round_trips": round(float(turn / 2.0), 1),
            # P326 convention: 2 x gross per unit turnover, bps. Turnover
            # strictly inside the era, so a position standing at the era
            # open is not turnover.
            "gross_bps_per_rt": round(float(2.0 * g.sum() / turn * 1e4), 1)
            if turn > 0 else None,
        }
    return out


def era_bool(df: pd.DataFrame, name: str) -> np.ndarray:
    a, b = ERAS[name]
    return ((df["timestamp"] >= a) & (df["timestamp"] < b)).to_numpy()


# --------------------------------------------------------- (a) SMA sweep --
def audit_sma(asset: str, df: pd.DataFrame) -> dict:
    out = {"pure": {}, "book": {}}
    for variant in ("pure", "book"):
        for n in SMA_SWEEP:
            pos = positions(df, n, variant)
            g, net, dpos = book_pnl(df, pos, asset)
            m = metrics(df, g, net, dpos)
            m["time_long_pct"] = round(float(pos.mean() * 100), 1)
            out[variant][str(n)] = m
    # verdict per variant
    verdicts = {}
    for variant in ("pure", "book"):
        tab = out[variant]
        live = tab[str(LIVE_SMA)]
        best_n = max(tab, key=lambda k: tab[k]["net_pct"])
        best = tab[best_n]["net_pct"]
        worst = min(v["net_pct"] for v in tab.values())
        dominators = []
        for k, v in tab.items():
            if k == str(LIVE_SMA):
                continue
            if all(v["eras"][e]["net_pct"] > live["eras"][e]["net_pct"]
                   for e in live["eras"]):
                dominators.append({"sma": int(k), "gap_net_pct":
                                   round(v["net_pct"] - live["net_pct"], 1)})
        plateau = (best - worst) < 0.2 * abs(best) if best != 0 else True
        within20 = live["net_pct"] >= best - 0.2 * abs(best)
        # noise floor: one year of the live book's turnover at the measured
        # RT cost — a gain smaller than that is inside a cost re-measurement.
        noise_pct = live["round_trips_per_yr"] * COST_RT_BPS[asset] / 100.0
        if not dominators and within20:
            v = "(i) close to optimal"
        elif not dominators:
            v = "(ii) arbitrary but harmless (nothing dominates 200 across eras)"
        else:
            big = [d for d in dominators if d["gap_net_pct"] > noise_pct]
            v = ("(iii)-candidate: dominated across eras by SMA%s, gap %+.1f%% > "
                 "noise %.1f%%" % (big[0]["sma"], big[0]["gap_net_pct"], noise_pct)
                 if big and not plateau else
                 "(ii) arbitrary but harmless (dominated, but gap inside noise/plateau)")
        verdicts[variant] = {"verdict": v, "best_sma": int(best_n),
                             "best_net_pct": best, "live_net_pct": live["net_pct"],
                             "dominators": dominators, "plateau": plateau,
                             "noise_pct_per_yr_turnover": round(noise_pct, 1)}
    out["verdicts"] = verdicts
    return out


# ------------------------------------------------------ (b) gate multiplier --
def gate_threshold(asset: str, mult: float, legs: float, regime_mult: float) -> float:
    per_leg = COST_RT_BPS[asset] / 2.0
    return (legs * per_leg + HOLD_COST_BPS) * mult * regime_mult


def audit_gate(backtest_sign: dict) -> dict:
    """backtest_sign[asset] = {'eras_positive': k, 'eras': 3} from the book-bull
    SMA200 replay — the per-asset verdict the constant must agree with."""
    grid = {}
    for asset in ASSETS:
        alpha = REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP[asset]
        rows = []
        for haircut in (1.0, PERF_FACTOR_FROZEN):
            for rm in (1.0, REGIME_MULT_LIVE):
                for legs in (1.0, 2.0):
                    for mult in (1.0, 1.10, 1.25, 1.5, 2.0):
                        thr = gate_threshold(asset, mult, legs, rm)
                        rows.append({"haircut": haircut, "regime_mult": rm,
                                     "legs": legs, "mult": mult,
                                     "est_bps": round(alpha * haircut, 1),
                                     "thr_bps": round(thr, 1),
                                     "pass": bool(alpha * haircut >= thr)})
        # break-even multiplier (legs=2) with and without haircut/regime mult
        be = {}
        for haircut in (1.0, PERF_FACTOR_FROZEN):
            for rm in (1.0, REGIME_MULT_LIVE):
                be[f"haircut{haircut}_rm{rm}"] = round(
                    alpha * haircut / ((COST_RT_BPS[asset] + HOLD_COST_BPS) * rm), 2)
        live = [r for r in rows if r["haircut"] == PERF_FACTOR_FROZEN
                and r["regime_mult"] == REGIME_MULT_LIVE and r["legs"] == 2.0]
        at10 = next(r for r in live if r["mult"] == 1.0)["pass"]
        at11 = next(r for r in live if r["mult"] == NORMAL_MULTIPLIER)["pass"]
        grid[asset] = {"seat_alpha_median_bps": alpha, "rows": rows,
                       "verdict_decided_by_the_1.10": at10 != at11,
                       "pass_at_mult_1.0_legs2": at10,
                       "breakeven_multiplier_legs2": be,
                       "live_setting": next(r for r in live if r["mult"] == NORMAL_MULTIPLIER),
                       "verdict_flips_in_1_to_2": len({r["pass"] for r in live}) > 1}
    # agreement with the backtest
    agree = {}
    for asset in ASSETS:
        g = grid[asset]
        bt = backtest_sign[asset]
        majority_pos = bt["eras_positive"] * 2 > bt["eras"]
        all_pos = bt["eras_positive"] == bt["eras"]
        all_neg = bt["eras_positive"] == 0
        live_pass = g["live_setting"]["pass"]
        if live_pass and all_neg:
            v = "WRONG: admits a book negative in every era"
        elif (not live_pass) and all_pos:
            v = "WRONG: excludes a book positive in every era"
        elif live_pass == majority_pos:
            v = "AGREES with backtest majority-of-eras verdict"
        else:
            v = "DISAGREES with backtest majority verdict (inside one cost of zero?)"
        agree[asset] = {"gate_live_pass": live_pass, "backtest": bt, "verdict": v}
    flips = any(grid[a]["verdict_flips_in_1_to_2"] for a in ASSETS)
    decided = [a for a in ASSETS if grid[a]["verdict_decided_by_the_1.10"]]
    flip_assets = [a for a in ASSETS if grid[a]["verdict_flips_in_1_to_2"]]
    if decided:
        mv = "(iii)/(i) the 1.10 itself decides " + ",".join(decided) + " — see agreement rows"
    elif flip_assets:
        mv = ("(ii) arbitrary but harmless: the 1.10 flips NO asset vs 1.0; only " +
              ",".join(flip_assets) + " flips anywhere in [1.0,2.0] (at >=1.5 with the 0.75 haircut)")
    else:
        mv = "(ii) just subtracting: no asset's verdict changes anywhere in [1.0,2.0]"
    return {"grid": grid, "agreement": agree, "multiplier_verdict": mv,
            "multiplier_changes_any_verdict_in_1_to_2": flips,
            "legs_2_is_correct_by_construction": True}


# ------------------------------------------------------- (c) vol parity --
def audit_fraction(books: dict) -> dict:
    """books[asset] = (df, pos) for the live book-bull SMA200 expression."""
    out = {"assets": {}, "eras": {}}
    sig_all = {}
    for asset, (df, pos) in books.items():
        held = np.concatenate([[0.0], pos[:-1]]).astype(bool)
        r = df["ret"].to_numpy(float)
        s_long = r[held].std() * np.sqrt(BARS_PER_YEAR_4H) if held.sum() > 100 else np.nan
        s_all = r.std() * np.sqrt(BARS_PER_YEAR_4H)
        sig_all[asset] = s_long
        last_px = float(df["close"].iloc[-1])
        notional_at_015 = LIVE_FRACTION[asset] * SLEEVE_EQUITY_USD
        out["assets"][asset] = {
            "vol_ann_while_long": round(float(s_long), 3),
            "vol_ann_unconditional": round(float(s_all), 3),
            "vol_4h_while_long_pct": round(float(s_long / np.sqrt(BARS_PER_YEAR_4H) * 100), 2),
            "last_close": last_px,
            "contracts_at_0.15": int(np.floor(notional_at_015 / (last_px * CONTRACT_SIZE[asset]))),
            "usd_1sigma_4h_at_0.15": round(float(notional_at_015 * s_long / np.sqrt(BARS_PER_YEAR_4H)), 1),
        }
        # per era
        for e in ERAS:
            m = era_bool(df, e) & held
            if m.sum() > 100:
                out["eras"].setdefault(e, {})[asset] = round(
                    float(r[m].std() * np.sqrt(BARS_PER_YEAR_4H)), 3)
    budget = sum(LIVE_FRACTION.values())
    inv = {a: 1.0 / sig_all[a] for a in ASSETS}
    z = sum(inv.values())
    parity = {a: round(budget * inv[a] / z, 3) for a in ASSETS}
    risk_share_flat = {a: round(LIVE_FRACTION[a] * sig_all[a] /
                                sum(LIVE_FRACTION[b] * sig_all[b] for b in ASSETS), 3)
                       for a in ASSETS}
    # era ranking stability of sigma
    rankings = []
    for e, d in out["eras"].items():
        if len(d) == 3:
            rankings.append(tuple(sorted(d, key=d.get)))
    rank_stable = len(set(rankings)) == 1
    within15 = all(abs(parity[a] - 0.15) <= 0.15 * 0.15 for a in ASSETS)
    over_cap = [a for a in ASSETS if parity[a] > POST_LEVERAGE_CAPS[a]]
    max_share = max(risk_share_flat.values())
    if within15:
        v = "(i) close to optimal"
    elif over_cap or max_share > 0.45 or not rank_stable:
        v = "(iii) wrong on a risk basis"
    else:
        v = "(ii) arbitrary but harmless"
    # combined 3-asset book, flat 0.15 vs parity, same total budget: the
    # "number" for a sizing change is risk-shape, not expected PnL — but the
    # realised drawdown of the combination is measurable.
    combo = {}
    for label, fr in (("flat_0.15", LIVE_FRACTION), ("vol_parity", parity)):
        tot = None
        for a, (df, pos) in books.items():
            g, net, _ = book_pnl(df, pos, a)
            # align on timestamps via reindex onto BTC's index
            ser = pd.Series(net * fr[a], index=df["timestamp"])
            tot = ser if tot is None else tot.add(ser, fill_value=0.0)
        n = tot.to_numpy(float)
        cum = np.cumsum(n); dd = cum - np.maximum.accumulate(cum)
        combo[label] = {"net_pct_of_equity": round(float(n.sum() * 100), 1),
                        "sharpe_ann": round(float(n.mean() / n.std() * np.sqrt(BARS_PER_YEAR_4H)), 2),
                        "max_dd_pct_of_equity": round(float(dd.min() * 100), 1),
                        "worst_4h_bar_pct": round(float(n.min() * 100), 2)}
    out["combined_book"] = combo
    out.update({"budget": budget, "vol_parity_fraction": parity,
                "risk_share_at_flat_0.15": risk_share_flat,
                "parity_exceeds_cap": over_cap, "sigma_rank_stable_across_eras": rank_stable,
                "era_rankings": [list(r) for r in rankings],
                "post_leverage_caps": POST_LEVERAGE_CAPS, "verdict": v})
    return out


# --------------------------------------------------- (d) seat statistic --
def stat_table(by_era: dict) -> dict:
    vals = list(by_era.values())
    return {"min": round(float(min(vals)), 1),
            "median": round(float(np.median(vals)), 1),
            "mean": round(float(np.mean(vals)), 1)}


def audit_statistic(sma_results: dict, backtest_sign: dict) -> dict:
    thr = {a: gate_threshold(a, NORMAL_MULTIPLIER, ROUND_TRIP_LEGS, REGIME_MULT_LIVE)
           for a in ASSETS}
    out = {"threshold_live_bps": {a: round(thr[a], 1) for a in ASSETS}, "tables": {}}
    # recorded lab table (full book, index eras)
    rec = {a: stat_table(REGIMEBOOK_ALPHA_BY_ERA[a]) for a in ASSETS}
    # this lab's calendar-era tables for the two hold-leg expressions at SMA200
    mine = {}
    for variant in ("pure", "book"):
        mine[variant] = {}
        for a in ASSETS:
            eras = sma_results[a][variant][str(LIVE_SMA)]["eras"]
            vals = {e: v["gross_bps_per_rt"] for e, v in eras.items()
                    if v["gross_bps_per_rt"] is not None}
            mine[variant][a] = {"by_era": vals, **stat_table(vals)} if vals else None
    out["tables"] = {"recorded_seat_alpha_full_book": rec,
                     "replay_hold_leg_pure_sma200": mine["pure"],
                     "replay_hold_leg_book_bull_sma200": mine["book"]}
    # which assets pass under each statistic
    passes = {}
    for stat in ("min", "median", "mean"):
        for haircut in (1.0, PERF_FACTOR_FROZEN):
            key = f"{stat}_haircut{haircut}"
            passes[key] = {a: bool(rec[a][stat] * haircut >= thr[a]) for a in ASSETS}
    out["admits_recorded_table"] = passes
    # verdict per statistic vs the backtest majority-of-eras verdict (book-bull)
    verd = {}
    for stat in ("min", "median", "mean"):
        adm = {a: passes[f"{stat}_haircut{PERF_FACTOR_FROZEN}"][a] for a in ASSETS}
        bad = []
        for a in ASSETS:
            bt = backtest_sign[a]
            maj = bt["eras_positive"] * 2 > bt["eras"]
            if adm[a] and bt["eras_positive"] == 0:
                bad.append(f"{a}: admits an all-era-negative book")
            elif (not adm[a]) and bt["eras_positive"] == bt["eras"]:
                bad.append(f"{a}: excludes an all-era-positive book")
            elif adm[a] != maj:
                bad.append(f"{a}: disagrees with majority verdict")
        verd[stat] = {"admits_with_0.75_haircut": adm,
                      "verdict": "(i) agrees with backtest" if not bad else
                      ("(iii) " + "; ".join(bad) if any("all-era" in b for b in bad)
                       else "(ii) " + "; ".join(bad))}
    out["verdicts"] = verd
    return out


# ----------------------------------------------------------------- main --
def main() -> int:
    dfs = {a: load_4h(a) for a in ASSETS}
    sma = {a: audit_sma(a, dfs[a]) for a in ASSETS}

    # the per-asset backtest verdict everything else is held against: the
    # live expression (book-bull, SMA200), eras positive after cost
    backtest_sign = {}
    books = {}
    for a in ASSETS:
        eras = sma[a]["book"][str(LIVE_SMA)]["eras"]
        backtest_sign[a] = {"eras_positive": sum(1 for v in eras.values() if v["net_pct"] > 0),
                            "eras": len(eras),
                            "net_by_era": {e: v["net_pct"] for e, v in eras.items()}}
        books[a] = (dfs[a], positions(dfs[a], LIVE_SMA, "book"))

    # (a) the cross-asset half of rule (iii): a lookback is only "wrong, costs
    # money" if the SAME lookback dominates 200 across eras on >= 2 assets.
    for variant in ("pure", "book"):
        dom_by_sma = {}
        for a in ASSETS:
            for d in sma[a]["verdicts"][variant]["dominators"]:
                dom_by_sma.setdefault(d["sma"], []).append((a, d["gap_net_pct"]))
        transfer = {k: v for k, v in dom_by_sma.items() if len(v) >= 2}
        for a in ASSETS:
            vd = sma[a]["verdicts"][variant]
            if vd["verdict"].startswith("(iii)-candidate"):
                vd["verdict"] = ("(iii) wrong: " + vd["verdict"][len("(iii)-candidate: "):]
                                 if any(a in [x[0] for x in v] for v in transfer.values())
                                 else "(ii) arbitrary but harmless — dominated on THIS asset only "
                                      "(no lookback dominates 200 on >=2 assets: single-asset fit, P262 transfer rule); "
                                      + vd["verdict"][len("(iii)-candidate: "):])
        sma.setdefault("_cross_asset", {})[variant] = {
            "lookbacks_dominating_200_on_2plus_assets": {str(k): v for k, v in transfer.items()},
            "all_dominators": {str(k): v for k, v in dom_by_sma.items()}}

    gate = audit_gate(backtest_sign)
    frac = audit_fraction(books)
    stat = audit_statistic(sma, backtest_sign)

    report = {"lab": "P370 strategy_threshold_audit", "cost_rt_bps": COST_RT_BPS,
              "eras": ERAS, "constants": {
                  "NORMAL_MULTIPLIER": NORMAL_MULTIPLIER, "ROUND_TRIP_LEGS": ROUND_TRIP_LEGS,
                  "PERF_FACTOR_FROZEN": PERF_FACTOR_FROZEN, "REGIME_MULT_LIVE": REGIME_MULT_LIVE,
                  "HOLD_COST_BPS": HOLD_COST_BPS, "LIVE_FRACTION": LIVE_FRACTION,
                  "POST_LEVERAGE_CAPS": POST_LEVERAGE_CAPS, "SLEEVE_EQUITY_USD": SLEEVE_EQUITY_USD},
              "a_sma_lookback": sma, "backtest_sign_book_bull_sma200": backtest_sign,
              "b_gate_multiplier": gate, "c_target_fraction": frac,
              "d_seat_statistic": stat}
    out = REPO / "training" / "reports" / "strategy_threshold_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ print --
    W = 100
    print("=" * W)
    print("  (a) SMA LOOKBACK SWEEP — 4H bars, measured CDE RT cost, additive net % of notional")
    print("=" * W)
    for a in ASSETS:
        for variant in ("pure", "book"):
            tab = sma[a][variant]
            print(f"\n{a} [{variant:4s}]  {'sma':>4s} {'net%':>7s} {'gross%':>7s} {'sharpe':>7s} "
                  f"{'maxDD%':>7s} {'RT/yr':>6s} {'long%':>6s}   eras(net%)   eras(gross bps/RT)")
            for n in SMA_SWEEP:
                m = tab[str(n)]
                eras = " ".join(f"{v['net_pct']:+6.1f}" for v in m["eras"].values())
                alph = " ".join(f"{(v['gross_bps_per_rt'] if v['gross_bps_per_rt'] is not None else float('nan')):+7.1f}"
                                for v in m["eras"].values())
                flag = " <- live" if n == LIVE_SMA else ""
                print(f"  {'':14s}{n:4d} {m['net_pct']:+7.1f} {m['gross_pct']:+7.1f} {m['sharpe_ann']:+7.2f} "
                      f"{m['max_dd_pct']:7.1f} {m['round_trips_per_yr']:6.1f} {m['time_long_pct']:6.1f}   "
                      f"[{eras}]   [{alph}]{flag}")
            v = sma[a]["verdicts"][variant]
            print(f"  verdict: {v['verdict']}  (best={v['best_sma']} at {v['best_net_pct']:+.1f}%, "
                  f"live {v['live_net_pct']:+.1f}%, dominators={v['dominators']})")

    print("\n  cross-asset:", json.dumps(sma["_cross_asset"]))
    print("\n" + "=" * W)
    print("  (b) ALPHA-GATE MULTIPLIER — est = seat alpha (era-median) x haircut; thr = (legs x per-leg + hold) x mult x regime")
    print("=" * W)
    for a in ASSETS:
        g = gate["grid"][a]
        ls = g["live_setting"]
        print(f"\n{a}  seat alpha {g['seat_alpha_median_bps']} bps/RT   LIVE: est {ls['est_bps']} vs thr {ls['thr_bps']} "
              f"-> {'PASS' if ls['pass'] else 'FAIL'}   break-even mult (legs=2): {g['breakeven_multiplier_legs2']}")
        print(f"  {'haircut':>7s} {'regime':>6s} {'legs':>4s} | " + " ".join(f"x{m:<4}" for m in (1.0, 1.10, 1.25, 1.5, 2.0)))
        for haircut in (1.0, PERF_FACTOR_FROZEN):
            for rm in (1.0, REGIME_MULT_LIVE):
                for legs in (1.0, 2.0):
                    rs = [r for r in g["rows"] if r["haircut"] == haircut and r["regime_mult"] == rm and r["legs"] == legs]
                    cells = " ".join(f"{'PASS' if r['pass'] else 'fail'} " for r in rs)
                    print(f"  {haircut:7.2f} {rm:6.3f} {legs:4.0f} | {cells}")
        print(f"  agreement: {gate['agreement'][a]['verdict']}  backtest eras net%: "
              f"{backtest_sign[a]['net_by_era']}")
    print(f"\n  multiplier changes ANY asset's verdict anywhere in [1.0,2.0]? "
          f"{gate['multiplier_changes_any_verdict_in_1_to_2']}")
    print(f"  MULTIPLIER VERDICT: {gate['multiplier_verdict']}")

    print("\n" + "=" * W)
    print("  (c) TARGET FRACTION — vol parity on the live book's long-leg realised vol")
    print("=" * W)
    for a in ASSETS:
        d = frac["assets"][a]
        print(f"  {a}  vol_ann(long)={d['vol_ann_while_long']:.3f}  uncond={d['vol_ann_unconditional']:.3f}  "
              f"4H sigma={d['vol_4h_while_long_pct']:.2f}%  at 0.15: {d['contracts_at_0.15']}ct, "
              f"1-sigma/4H=${d['usd_1sigma_4h_at_0.15']:.0f}  parity fraction={frac['vol_parity_fraction'][a]:.3f}  "
              f"risk share@flat={frac['risk_share_at_flat_0.15'][a]:.2f}  cap={POST_LEVERAGE_CAPS[a]}")
    print(f"  per-era vol(long): {frac['eras']}")
    print(f"  sigma ranking stable across eras: {frac['sigma_rank_stable_across_eras']}  {frac['era_rankings']}")
    print(f"  combined book (same 0.45 budget): {json.dumps(frac['combined_book'])}")
    print(f"  verdict: {frac['verdict']}")

    print("\n" + "=" * W)
    print("  (d) SEAT-ALPHA STATISTIC — which assets each statistic admits (live thr, with 0.75 haircut)")
    print("=" * W)
    print(f"  live thresholds bps: {stat['threshold_live_bps']}")
    for name, tab in stat["tables"].items():
        print(f"  {name}:")
        for a in ASSETS:
            t = tab[a]
            if t is None:
                print(f"    {a}: n/a"); continue
            by = t.get("by_era", REGIMEBOOK_ALPHA_BY_ERA.get(a) if "recorded" in name else {})
            print(f"    {a}: min {t['min']:+7.1f}  median {t['median']:+7.1f}  mean {t['mean']:+7.1f}   eras={by}")
    for s, v in stat["verdicts"].items():
        print(f"  {s:6s} admits {v['admits_with_0.75_haircut']}  -> {v['verdict']}")
    print(f"\nreport -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

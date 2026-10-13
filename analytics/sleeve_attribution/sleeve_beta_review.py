"""[P230/P423] Realized sleeve beta — single-factor vs BTC, plus the
multi-factor / conditional / ex-ante views the single number cannot give.

The Apr-Jun forensic priced ~half the -25% as BETA (+0.54 net-long into a
-23% market, P143/P144), P201 records that no net-exposure control binds the
sleeve's actual book, and the only beta tool in the tree
(scripts/run_beta_audit.py) reads a Kraken-era file that does not exist — its
last report PRE-DATES the loss it existed to measure (P227b made it refuse
loudly, but it still cannot see the sleeve). Since P150 the sleeve has been
writing an equity point every 4H tick to data/coinbase_sleeve_pnl.jsonl.
This reads that series, aligns it to 4H closes from Kraken's PUBLIC OHLC
endpoint, and reports — measurement only, no orders, no state.

[P423] WHY ONE BETA IS NOT ENOUGH ON THIS BOOK. The sleeve is flat ~90% of
bars, so a full-series OLS beta is dominated by zeros (the P230 reading of
-0.005 over 329 pairs says almost nothing about the risk the book takes
WHEN it is in the market), and three correlated coins cannot be summarized
by one of them. So four views, each labelled for what it measures:
  1. SINGLE-FACTOR vs BTC on the full series (the P230 number, kept).
  2. MULTI-FACTOR: r_sleeve = a + b_BTC r_BTC + b_ETH r_ETH + b_SOL r_SOL,
     OLS with HC1 (White) standard errors — 4H non-overlapping returns need
     no Newey-West; heavy tails are handled by the SE, not assumed away.
  3. CONDITIONAL (Henriksson-Merton form on the equal-weight market):
     r_sleeve = a + b r_m + g max(0, r_m) + d*1[in-market] r_m — the
     timing/exposure split, with the in-market interaction reported as the
     "beta when exposed". The in-market-ONLY regression is also printed but
     labelled DIAGNOSTIC: it selects on the strategy's own decision.
  4. EX-ANTE exposure-implied beta from the positions the book actually
     held: beta_ea(t) = sum_a w_a(t) * beta_coin_a, with w_a = signed
     notional / equity and beta_coin_a the coin's own beta to the
     equal-weight market over the window; reported as its mean, its mean
     while in-market, and the net-long share. Ex-ante vs ex-post disagreeing
     is the signal: it says the realized book is not the intended one.

WHERE THIS RUNS (P213): in-container on the live server, where the data
volume is:
    docker exec hmats-engine python -X utf8 \
        analytics/sleeve_attribution/sleeve_beta_review.py
or operator-local with --pnl-file pointing at an scp'd copy. Missing file or
unreachable price API => REFUSAL (exit 2), never an empty report.

READ THE NUMBER HONESTLY: with the sleeve capped at a few nano contracts per
asset, beta is bounded by (gross notional / equity); SE(beta) ~ 0.15-0.25
at n~180 bars, so quote direction, not the second decimal (aim >= 500
in-market bars before a conditional beta means anything). The interesting
outputs are the SIGN persistence (net-long share), the alpha intercept, and
the ex-ante/ex-post gap — the same decomposition the June forensic did by
hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

MIN_POINTS = 12  # ~2 days of 4H ticks; below this a regression is decoration
MIN_COND_POINTS = 12  # in-market bars below which the conditional beta is withheld

# [P423] Restated locally (stdlib tool); pinned equal to agent_ic_review's
# KRAKEN_PAIRS for the home trio and to the adapter's contract sizes by
# tests/test_p423_gaps.py (P192 two-file rule).
FACTOR_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}
CONTRACT_SIZE = {"BTC": 0.01, "ETH": 0.1, "SOL": 5.0}


def _refuse(msg: str) -> None:
    print(f"REFUSING TO REPORT: {msg}", file=sys.stderr)
    sys.exit(2)


def load_pnl(path: Path, window_days: int) -> list[dict]:
    if not path.exists():
        _refuse(
            f"{path} does not exist here. The sleeve writes it on the LIVE "
            f"data volume — run in-container or scp the file and pass "
            f"--pnl-file. 'File missing' is not 'sleeve flat'."
        )
    cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if float(r["ts"]) >= cutoff and r.get("equity_usd"):
                rows.append(r)
        except Exception:  # noqa: silent-swallow — per-record JSONL
            # skip; MIN_POINTS refusal below reports the aggregate
            continue
    if len(rows) < MIN_POINTS:
        _refuse(f"only {len(rows)} usable points in {window_days}d "
                f"(need >= {MIN_POINTS}) — widen --window-days.")
    return rows


def fetch_closes(pair: str, label: str) -> tuple[list[float], list[float]]:
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        _refuse(f"Kraken public OHLC unreachable for {label}: {type(e).__name__}: {e} — "
                f"no benchmark, no beta. (Network refusal, distinct from "
                f"'no sleeve data'.)")
    if payload.get("error"):
        _refuse(f"Kraken OHLC error for {label}: {payload['error']}")
    key = next(k for k in payload["result"] if k != "last")
    rows = payload["result"][key]
    return [float(x[0]) for x in rows], [float(x[4]) for x in rows]


def fetch_btc_closes() -> tuple[list[float], list[float]]:
    return fetch_closes(FACTOR_PAIRS["BTC"], "BTC")


# ---------------------------------------------------------------- pure OLS
def _solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Gauss-Jordan with partial pivoting; None on a singular system."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [rv - f * cv for rv, cv in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def _inv(A: list[list[float]]) -> list[list[float]] | None:
    n = len(A)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        x = _solve(A, e)
        if x is None:
            return None
        cols.append(x)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ols_hc1(X: list[list[float]], y: list[float]) -> dict | None:
    """OLS with an intercept prepended; HC1 (White, small-sample scaled) SEs.
    X: rows of regressors (no intercept). Returns None if singular or n<=k."""
    n = len(y)
    if n == 0:
        return None
    Xi = [[1.0] + list(r) for r in X]
    k = len(Xi[0])
    if n <= k:
        return None
    XtX = [[sum(Xi[t][i] * Xi[t][j] for t in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(Xi[t][i] * y[t] for t in range(n)) for i in range(k)]
    beta = _solve(XtX, Xty)
    inv = _inv(XtX)
    if beta is None or inv is None:
        return None
    resid = [y[t] - sum(beta[i] * Xi[t][i] for i in range(k)) for t in range(n)]
    # meat = sum_t e_t^2 x_t x_t'
    meat = [[sum(resid[t] ** 2 * Xi[t][i] * Xi[t][j] for t in range(n))
             for j in range(k)] for i in range(k)]
    # cov = inv * meat * inv * n/(n-k)
    tmp = [[sum(inv[i][a] * meat[a][j] for a in range(k)) for j in range(k)] for i in range(k)]
    cov = [[sum(tmp[i][a] * inv[a][j] for a in range(k)) * n / (n - k)
            for j in range(k)] for i in range(k)]
    se = [math.sqrt(max(cov[i][i], 0.0)) for i in range(k)]
    ybar = sum(y) / n
    ss_tot = sum((v - ybar) ** 2 for v in y)
    ss_res = sum(e * e for e in resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"coef": beta, "se": se, "r2": r2, "n": n,
            "t": [b / s if s > 0 else float("nan") for b, s in zip(beta, se)]}


# ---------------------------------------------------------------- alignment
def _ret_between(ts_list: list[float], closes: list[float], t0: float, t1: float) -> float | None:
    i0 = bisect_right(ts_list, t0) - 1
    i1 = bisect_right(ts_list, t1) - 1
    if i0 < 0 or i1 <= i0 or i1 >= len(closes):
        return None
    return closes[i1] / closes[i0] - 1.0


def _px_at(ts_list: list[float], closes: list[float], t: float) -> float | None:
    i = bisect_right(ts_list, t) - 1
    return closes[i] if 0 <= i < len(closes) else None


def build_pairs(rows: list[dict], series: dict[str, tuple[list[float], list[float]]]) -> list[dict]:
    """Consecutive sleeve points ~one 4H bar apart, with every factor's
    return over the SAME wall-clock span and the exposure held at the start."""
    out = []
    for prev, cur in zip(rows, rows[1:]):
        dt = cur["ts"] - prev["ts"]
        if not (0.5 * 14400 <= dt <= 2.5 * 14400):
            continue  # restart gap — a stitched return would be fiction
        e0, e1 = float(prev["equity_usd"]), float(cur["equity_usd"])
        if e0 <= 0:
            continue
        # [P423] a deposit/withdrawal between two ticks is not a return
        # (P293h): skip the pair when the detected external flow moved, and
        # prefer the flow-adjusted pnl delta when the series carries it.
        f0 = float(prev.get("external_flow_usd") or 0.0)
        f1 = float(cur.get("external_flow_usd") or 0.0)
        if abs(f1 - f0) > 1e-6:
            continue
        if prev.get("pnl_usd") is not None and cur.get("pnl_usd") is not None:
            sleeve_ret = (float(cur["pnl_usd"]) - float(prev["pnl_usd"])) / e0
        else:
            sleeve_ret = e1 / e0 - 1.0
        rets = {}
        ok = True
        for a, (tsl, cl) in series.items():
            r = _ret_between(tsl, cl, prev["ts"], cur["ts"])
            if r is None:
                ok = False
                break
            rets[a] = r
        if not ok:
            continue
        pos = {a: float(v or 0) for a, v in (prev.get("positions") or {}).items()}
        expo = {}
        for a in series:
            px = _px_at(series[a][0], series[a][1], prev["ts"])
            expo[a] = (pos.get(a, 0.0) * CONTRACT_SIZE.get(a, 0.0) * px / e0
                       if px else 0.0)
        out.append({"sleeve": sleeve_ret, "rets": rets, "expo": expo,
                    "net": sum(pos.values()), "in_market": any(abs(v) > 0 for v in pos.values())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pnl-file", default="data/coinbase_sleeve_pnl.jsonl")
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--json", default=None, help="optional report path")
    args = ap.parse_args()

    rows = load_pnl(Path(args.pnl_file), args.window_days)
    series = {a: fetch_closes(p, a) for a, p in FACTOR_PAIRS.items()}
    pairs = build_pairs(rows, series)
    n = len(pairs)
    if n < MIN_POINTS:
        _refuse(f"only {n} contiguous 4H return pairs after gap filtering "
                f"(need >= {MIN_POINTS}).")

    sleeve = [p["sleeve"] for p in pairs]
    btc = [p["rets"]["BTC"] for p in pairs]
    longs = sum(1 for p in pairs if p["net"] > 0)
    inm = [p["in_market"] for p in pairs]
    n_in = sum(inm)
    report: dict = {"window_days": args.window_days, "n_pairs": n, "n_in_market": n_in,
                    "net_long_share": longs / n}

    # 1. single-factor vs BTC (the P230 number)
    sf = ols_hc1([[b] for b in btc], sleeve)
    if sf is None or sum((b - sum(btc) / n) ** 2 for b in btc) == 0:
        _refuse("benchmark variance is zero over the window.")
    beta, alpha_4h, r2 = sf["coef"][1], sf["coef"][0], sf["r2"]
    ms = sum(sleeve) / n
    vol_4h = math.sqrt(sum((s - ms) ** 2 for s in sleeve) / n)
    print(f"Sleeve beta vs BTC — {n} contiguous 4H pairs, "
          f"{args.window_days}d window")
    print(f"  beta            {beta:+.3f}  (HC1 se {sf['se'][1]:.3f})")
    print(f"  alpha           {alpha_4h * 1e4:+.1f} bps/4h "
          f"({alpha_4h * 6 * 1e4:+.0f} bps/day)")
    print(f"  R^2             {r2:.3f}   (1-R^2 = idiosyncratic/churn share)")
    print(f"  sleeve vol      {vol_4h * 1e4:.1f} bps/4h")
    print(f"  net-long share  {longs}/{n} bars ({100.0 * longs / n:.0f}%) — "
          f"the +0.54 problem was DIRECTION persistence, watch this line")
    print(f"  in-market       {n_in}/{n} bars ({100.0 * n_in / n:.0f}%) — a full-series "
          f"beta on a mostly-flat book is mostly zeros")
    print(f"  equity          ${rows[0]['equity_usd']:,.2f} -> "
          f"${rows[-1]['equity_usd']:,.2f}")
    report["single_factor_btc"] = {"beta": beta, "se": sf["se"][1], "alpha_4h": alpha_4h, "r2": r2}

    # 2. multi-factor BTC/ETH/SOL, HC1
    assets = list(FACTOR_PAIRS)
    mf = ols_hc1([[p["rets"][a] for a in assets] for p in pairs], sleeve)
    print(f"\n[P423] Multi-factor (BTC, ETH, SOL legs; HC1 se) — full series")
    if mf is None:
        print("  (singular / too few points)")
    else:
        for i, a in enumerate(assets):
            print(f"  b_{a:3}           {mf['coef'][i + 1]:+.3f}  (se {mf['se'][i + 1]:.3f}, "
                  f"t {mf['t'][i + 1]:+.1f})")
        print(f"  alpha           {mf['coef'][0] * 1e4:+.1f} bps/4h   R^2 {mf['r2']:.3f}")
        report["multi_factor"] = {a: {"beta": mf["coef"][i + 1], "se": mf["se"][i + 1]}
                                  for i, a in enumerate(assets)}
        report["multi_factor"]["alpha_4h"] = mf["coef"][0]
        report["multi_factor"]["r2"] = mf["r2"]

    # 3. conditional: HM on the equal-weight market + in-market interaction
    mkt = [sum(p["rets"][a] for a in assets) / len(assets) for p in pairs]
    hm = ols_hc1([[m, max(0.0, m), (m if im else 0.0)] for m, im in zip(mkt, inm)], sleeve)
    print(f"\n[P423] Conditional (Henriksson-Merton on the equal-weight BTC/ETH/SOL market)")
    if hm is None or n_in < MIN_COND_POINTS:
        print(f"  withheld: {n_in} in-market bars (need >= {MIN_COND_POINTS}) — "
              f"'beta when exposed' has no sample yet; this is a count, not a zero")
        report["conditional"] = {"withheld": True, "n_in_market": n_in}
    else:
        b, g, d = hm["coef"][1], hm["coef"][2], hm["coef"][3]
        print(f"  b (market)      {b:+.3f}  (se {hm['se'][1]:.3f})")
        print(f"  g (timing, max(0,r_m))  {g:+.3f}  (se {hm['se'][2]:.3f}) — "
              f"usually noise in the literature; >0 = convex/up-capture")
        print(f"  d (in-market x r_m)     {d:+.3f}  (se {hm['se'][3]:.3f}) — "
              f"beta WHEN EXPOSED = b + d = {b + d:+.3f}")
        # diagnostic: in-market-only single factor (selection on own decision)
        sub_x = [[m] for m, im in zip(mkt, inm) if im]
        sub_y = [s for s, im in zip(sleeve, inm) if im]
        sub = ols_hc1(sub_x, sub_y)
        if sub is not None:
            print(f"  DIAGNOSTIC in-market-only beta {sub['coef'][1]:+.3f} (se {sub['se'][1]:.3f}, "
                  f"n {sub['n']}) — conditional on the strategy's own decision; "
                  f"not a risk number")
        report["conditional"] = {"b": b, "g": g, "d": d, "beta_when_exposed": b + d,
                                 "se": hm["se"][1:], "n_in_market": n_in,
                                 "in_market_only_beta": (sub["coef"][1] if sub else None)}

    # 4. ex-ante exposure-implied beta
    coin_beta = {}
    for a in assets:
        cb = ols_hc1([[m] for m in mkt], [p["rets"][a] for p in pairs])
        coin_beta[a] = cb["coef"][1] if cb else float("nan")
    ea = [sum(p["expo"][a] * coin_beta[a] for a in assets) for p in pairs]
    ea_in = [v for v, im in zip(ea, inm) if im]
    mean_ea = sum(ea) / n
    mean_ea_in = sum(ea_in) / len(ea_in) if ea_in else float("nan")
    gross = [sum(abs(p["expo"][a]) for a in assets) for p in pairs]
    print(f"\n[P423] Ex-ante exposure-implied beta (positions x coin beta to the EW market)")
    print(f"  coin betas      " + ", ".join(f"{a} {coin_beta[a]:+.2f}" for a in assets))
    print(f"  mean beta_ea    {mean_ea:+.3f} (full series)   "
          f"{mean_ea_in:+.3f} (in-market bars, n {len(ea_in)})")
    print(f"  mean gross expo {sum(gross) / n:.3f}x equity   max {max(gross):.3f}x "
          f"(BTC/ETH/SOL legs only; breadth contracts count as in-market but carry no factor)")
    gap = (mf["coef"][1] + mf["coef"][2] + mf["coef"][3] if mf else float("nan")) - mean_ea
    print(f"  ex-post (sum of 3 legs) - ex-ante = {gap:+.3f} — a large gap says the "
          f"realized book is not the intended one (fills, stops, churn)")
    report["ex_ante"] = {"coin_beta": coin_beta, "mean_beta_ea": mean_ea,
                         "mean_beta_ea_in_market": mean_ea_in,
                         "mean_gross_exposure": sum(gross) / n, "ex_post_minus_ex_ante": gap}

    # stability: first vs second half
    h = n // 2
    if h >= MIN_POINTS:
        a1 = ols_hc1([[b] for b in btc[:h]], sleeve[:h])
        a2 = ols_hc1([[b] for b in btc[h:]], sleeve[h:])
        if a1 and a2:
            print(f"\n  stability (BTC beta, halves): {a1['coef'][1]:+.3f} -> {a2['coef'][1]:+.3f} "
                  f"— quote the range, not the point (betas are not stationary)")
            report["halves_btc_beta"] = [a1["coef"][1], a2["coef"][1]]

    print("\nContext: P144's 0.50 net budget is enforced by sleeve_exposure "
          "(P208); this measures what the book actually DID. Small n makes "
          "beta noisy — trend, not gospel, below ~180 pairs; the conditional "
          "beta needs ~500 in-market bars before its second decimal means anything.")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), **report},
            indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

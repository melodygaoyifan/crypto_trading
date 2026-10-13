"""[P404] Does ETF flow ADD to the certified SMA200 overlay, or overlap it?
2y history (not the thin forward window, so not the P388 leak trap). ETF = the
P402 lag-1 z-deadband signal; SMA200 = the certified long/flat overlay (P377).
Honest CDE per-leg cost (charged as dp*(pl/2.0)), OOS = 2nd half.

RESULT (2026-08-24): the two are COMPLEMENTARY, not overlapping (corr(SMA_pos,
ETF_pos) ~ -0.15 both assets). On BTC the de-risk stack (long only when SMA200
trends AND ETF is not signaling outflow) gives OOS Sharpe +1.49 / maxDD -9.7%
vs SMA200-alone +0.37 / -22.1% -- 4x Sharpe, half the drawdown. ETH: SMA200 is
the weak link, so ETF is better standalone there (asset-specific).
[P420] THOSE NUMBERS WERE LAG-2: `_stats` filled ret[:-1] and then used
ret[1:], so pos[t] earned the t+1 -> t+2 move. Fixed to lag-1 (ret[1:], the
etf_flow_probe convention); the corrected read is recorded in the P420 entry
and must be quoted beside the P404 figures, never instead of them.
CAVEATS: same recent-ETF-era window as P400 (regime-concentrated); n~300 OOS
(wide CI); the de-risk form was chosen after seeing 2 combos (the a-priori
sensible one). This strengthens the ARMING case; the live timing check
(etfflow_timing_check.py) + operator flip (P141) still gate a live seat.

[P425] RELEASE VARIANT (operator: "any improvement on profitability" ->
"do all"). The P422 finding: the live ETF de-risk HOLDS a -1 until an INFLOW
day with z > +1, so BTC/ETH could not go long for 12 days on a z that had
recovered to -0.43/-0.85. Candidate: release a held -1 to flat once z crosses
back above -release_z (default candidate 0.0 = "the outflow is over"). The
positions are built by STEPPING THE LIVE FUNCTION (etf_flow_zscore_direction,
single-source, P172) and a parity assert pins the no-release step to the
lab's own _deadband. PRE-COMMITTED RULE (fixed before the first run, P332):
adopt release_z=0.0 iff, on BOTH BTC and ETH, the de-risk combo WITH release
has OOS Sharpe >= the current combo's AND maxDD not worse; release_z=0.5 is
printed as CONTEXT only (never selected). Anything else -> record, do not arm.

Usage: python etf_sma_overlay_check.py [--cache-dir DIR with etf_btc.json/etf_eth.json]
                                        [--save-cache DIR] [--json OUT]
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from defense.etf_flow_shadow import (  # noqa: E402  [P425] single-source stepping
    ZSCORE_MIN_OBS, ZSCORE_WINDOW, etf_flow_zscore_direction)

COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4}
BAND = 1.0; LAG = 1
RELEASE_CANDIDATE = 0.0     # [P425] the pre-committed candidate
RELEASE_CONTEXT = 0.5       # printed for context, never selected
_URL = "https://open-api-v4.coinglass.com/api/etf/{a}/flow-history"
_APIN = {"BTC": "bitcoin", "ETH": "ethereum"}


def _load(asset, cache_dir):
    if cache_dir:
        p = Path(cache_dir) / f"etf_{asset.lower()}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("data", [])
    key = os.environ.get("COINGLASS_API_KEY", "")
    if not key:
        return None
    req = urllib.request.Request(_URL.format(a=_APIN[asset]), headers={"CG-API-KEY": key})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read()).get("data", [])
    except Exception as e:
        print(f"  {asset}: fetch failed ({e})"); return None


def _z(x, w=30):
    o = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w):i]; win = win[np.isfinite(win)]
        if len(win) >= 15 and win.std() > 0:
            o[i] = np.clip((x[i] - win.mean()) / win.std(), -5, 5)
    return o


def _deadband(sig, band):
    pos = np.zeros(len(sig)); cur = 0.0
    for i in range(len(sig)):
        if not np.isfinite(sig[i]): pos[i] = cur; continue
        cur = 1.0 if sig[i] > band else (-1.0 if sig[i] < -band else cur); pos[i] = cur
    return pos


def _live_positions(flow, release_band=None):
    """[P425] Positions by stepping the LIVE rule day by day: at day i the
    newest COMPLETED flow is flow[i-1] (lag-1), the trailing window the 30
    completed days before it, prev = yesterday's position. Mirrors
    etf_flow_shadow.replay_hold but records every day's position."""
    n = len(flow); pos = np.zeros(n); prev = 0.0
    for i in range(1, n):
        trailing = [float(f) for f in flow[max(0, i - 1 - ZSCORE_WINDOW):i - 1]]
        d, _z, _r = etf_flow_zscore_direction(
            float(flow[i - 1]), 0.0, trailing, prev, band=BAND,
            min_obs=ZSCORE_MIN_OBS, release_band=release_band)
        prev = d; pos[i] = d
    return pos


def _stats(px, pos, pl, mid):
    # [P420] ret[t] = the t-1 -> t return, so pos[t] * ret[t+1] earns the
    # t -> t+1 move: LAG-1, the alignment the docstring claims and the one
    # etf_flow_probe._hold uses. The pre-P420 line filled ret[:-1] (ret[t] =
    # the t -> t+1 move) and then multiplied by ret[t+1] -- pos[t] earned the
    # t+1 -> t+2 move, i.e. a lag-2 book: the P404 numbers were measured one
    # day late. Pinned by tests/test_p418_ops_labs_and_docs.py.
    ret = np.zeros(len(px)); ret[1:] = px[1:] / np.where(px[:-1] == 0, np.nan, px[:-1]) - 1
    ret = np.nan_to_num(ret); dp = np.abs(np.diff(pos, prepend=0.0))
    pnl = np.zeros(len(px)); pnl[:-1] = pos[:-1] * ret[1:]; pnl = pnl - dp * (pl / 2.0)
    p = pnl[mid:]; p = p[np.isfinite(p)]
    cum = np.cumsum(p); mdd = float((cum - np.maximum.accumulate(cum)).min()); sd = p.std()
    return (round(float(p.sum() * 100), 1),
            round(float(p.mean() / sd * np.sqrt(365.25)), 2) if sd > 0 else 0.0,
            round(mdd * 100, 1))


def _argv(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def main() -> int:
    cd = _argv("--cache-dir"); save = _argv("--save-cache"); jout = _argv("--json")
    print("[P404] ETF flow vs / + certified SMA200 overlay (2y, OOS 2nd half, after CDE cost)")
    measured = 0
    out = {"rule": "adopt release_z=%.2f iff combo+release OOS Sharpe >= combo AND maxDD not worse, BOTH assets" % RELEASE_CANDIDATE,
           "assets": {}}
    verdict_ok = []
    for a in ("BTC", "ETH"):
        d = _load(a, cd)
        if not d:
            print(f"  {a}: no data (--cache-dir or COINGLASS_API_KEY)"); continue
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            (Path(save) / f"etf_{a.lower()}.json").write_text(
                json.dumps({"data": d}), encoding="utf-8")
        d = [x for x in d if x.get("price_usd")]
        if len(d) < 300:
            print(f"  {a}: {len(d)} days only"); continue
        measured += 1
        px = np.array([float(x["price_usd"]) for x in d])
        flow = np.array([float(x.get("flow_usd", 0) or 0) for x in d])
        n = len(px); mid = n // 2; pl = COST_RT[a]
        raw = np.concatenate([[np.nan] * LAG, flow[:n - LAG]])
        etf = _deadband(_z(raw), BAND)
        sma = np.array([np.mean(px[max(0, i - 200):i + 1]) for i in range(n)])
        smapos = (px > sma).astype(float)
        combo = np.where(etf < 0, 0.0, smapos)   # ETF outflow forces flat (de-risk)
        avg = 0.5 * (smapos + etf)
        corr = float(np.corrcoef(smapos[mid:], etf[mid:])[0, 1])
        # [P425] the live rule, stepped; parity with the lab's own deadband
        # is asserted so the release variants are measured on the SAME book.
        live0 = _live_positions(flow, None)
        if not np.array_equal(live0, etf):
            print(f"  {a}: PARITY FAIL live-step vs _deadband "
                  f"({int((live0 != etf).sum())} days differ) — REFUSING")
            return 2
        rel_c = _live_positions(flow, RELEASE_CANDIDATE)
        rel_x = _live_positions(flow, RELEASE_CONTEXT)
        combo_c = np.where(rel_c < 0, 0.0, smapos)
        combo_x = np.where(rel_x < 0, 0.0, smapos)
        print(f"=== {a} (OOS n={n - mid}) ===")
        rows = {}
        cand_name = "combo +release [P425 cand z>%+.1f]" % (-RELEASE_CANDIDATE)
        for name, pos in (("SMA200 long/flat (certified)", smapos),
                          ("ETF lag1 z-deadband (P402)", etf),
                          ("SMA200 + ETF-outflow de-risk", combo),
                          ("0.5*(SMA+ETF)", avg),
                          ("ETF +release [P425 cand z>%+.1f]" % (-RELEASE_CANDIDATE), rel_c),
                          (cand_name, combo_c),
                          ("ETF +release [context z>%+.1f]" % (-RELEASE_CONTEXT), rel_x),
                          ("combo +release [context z>%+.1f]" % (-RELEASE_CONTEXT), combo_x)):
            net, sh, mdd = _stats(px, pos, pl, mid)
            rows[name] = {"net_pct": net, "sharpe": sh, "maxdd_pct": mdd,
                          "flat_share_oos": round(float((pos[mid:] == 0).mean()), 3)}
            print(f"  {name:40s}: net {net:+6.1f}%  Sharpe {sh:+.2f}  maxDD {mdd:+.1f}%  flat {rows[name]['flat_share_oos']:.0%}")
        print(f"  corr(SMA_pos, ETF_pos) OOS = {corr:+.2f} (~0 complementary, ~1 overlaps)")
        base = rows["SMA200 + ETF-outflow de-risk"]
        cand = rows[cand_name]
        ok = cand["sharpe"] >= base["sharpe"] and cand["maxdd_pct"] >= base["maxdd_pct"]
        verdict_ok.append(ok)
        print(f"  [P425] release candidate on {a}: {'EARNS' if ok else 'DOES NOT EARN'} "
              f"(Sharpe {cand['sharpe']:+.2f} vs {base['sharpe']:+.2f}, "
              f"maxDD {cand['maxdd_pct']:+.1f} vs {base['maxdd_pct']:+.1f})")
        out["assets"][a] = {"n_oos": n - mid, "corr_sma_etf": round(corr, 3), "rows": rows,
                            "candidate_earns": ok}
    if measured == 0:
        print("  REFUSED — no data"); return 2
    out["verdict"] = ("ADOPT" if measured == 2 and all(verdict_ok)
                      else ("RECORD_NOT_ARM" if measured == 2 else "INCOMPLETE"))
    print(f"[P425] VERDICT: {out['verdict']} (pre-committed rule: {out['rule']})")
    if jout:
        Path(jout).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

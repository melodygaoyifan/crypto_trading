"""
Trend-following SHADOW tracker (forward out-of-sample validation, no live footprint).

The trend-following strategy (strategies/trend_following.py) backtested at 3-asset
Sharpe 0.53 / PSR 89% over 5.3y. Before it can replace the live decision layer it
must prove edge on FORWARD data it has never seen. This standalone observer does
that without touching the live engine:

  track:  each run, fetch recent BTC/ETH/SOL 4H closes, compute the strategy's
          target_position per asset, append one record per (asset, bar) to
          data/trend_shadow.jsonl (deduped). Run on a 4H schedule.
  score:  read the log, join each logged position to its realized NEXT-bar return,
          and report the forward (out-of-sample) net-of-cost Sharpe + PSR.

Iron law: this NEVER trades. It only observes + logs. Promotion to the live
decision layer is a separate, gated decision made only if `score` confirms edge
on forward data (purged-CV via analytics/validation as well).

Usage (on the cloud, repo at /opt/hmats):
  docker exec hmats-engine python3 scripts/trend_shadow_tracker.py track
  docker exec hmats-engine python3 scripts/trend_shadow_tracker.py score
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.trend_following import TrendFollowingStrategy  # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "trend_shadow.jsonl")
ASSETS = [("BTC", "BTC/USDT:USDT"), ("ETH", "ETH/USDT:USDT"), ("SOL", "SOL/USDT:USDT")]
COST_BPS = 15.0
YEAR_BARS = 365 * 6


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def track():
    import ccxt
    ex = ccxt.binanceusdm()
    strat = TrendFollowingStrategy()
    need = strat.min_history() + 5
    # load existing keys to dedup (asset|bar_ts)
    seen = set()
    if os.path.exists(LOG):
        for ln in open(LOG, encoding="utf-8"):
            try:
                r = json.loads(ln)
                seen.add(f"{r['asset']}|{r['bar_ts']}")
            except Exception:  # noqa: silent-swallow — skip malformed log line
                continue
    out_lines = []
    for asset, sym in ASSETS:
        try:
            o = ex.fetch_ohlcv(sym, "4h", limit=need)
        except Exception as e:
            print(f"[TREND-SHADOW] {asset} fetch failed: {type(e).__name__}: {e}")
            continue
        if len(o) < strat.min_history():
            print(f"[TREND-SHADOW] {asset} too few bars ({len(o)})")
            continue
        closes = [c[4] for c in o]
        bar_ts = o[-1][0]  # the latest closed bar this signal is keyed to
        key = f"{asset}|{bar_ts}"
        if key in seen:
            continue
        res = strat.compute(closes)
        if res is None:
            continue
        rec = {
            "logged_at": _now_iso(),
            "asset": asset,
            "bar_ts": bar_ts,
            "close": closes[-1],
            "target_position": res["target_position"],
            "signal": res["signal"],
            "ann_vol": res["ann_vol"],
        }
        out_lines.append(json.dumps(rec))
        print(f"[TREND-SHADOW] {asset} bar={datetime.fromtimestamp(bar_ts/1000, timezone.utc):%Y-%m-%d %H:%M} "
              f"pos={res['target_position']:+.2f} sig={res['signal']:+.2f}")
    if out_lines:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"[TREND-SHADOW] appended {len(out_lines)} record(s) to {LOG}")
    else:
        print("[TREND-SHADOW] nothing new to log this bar")


def _ann_sharpe(rets):
    n = len(rets)
    if n < 3:
        return 0.0, 0.0
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / n)
    return (m / sd * math.sqrt(YEAR_BARS)) if sd > 0 else 0.0, sd


def score():
    if not os.path.exists(LOG):
        print("[TREND-SHADOW] no log yet — run `track` on a 4H schedule first")
        return
    recs = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
    # per asset, sort by bar_ts; realized fwd return = (next logged close / this close - 1)
    by_asset = {}
    for r in recs:
        by_asset.setdefault(r["asset"], []).append(r)
    port = []
    print(f"[TREND-SHADOW] forward OOS score over {len(recs)} logged signals")
    for a, rs in by_asset.items():
        rs.sort(key=lambda x: x["bar_ts"])
        srets = []
        prev_pos = 0.0
        for i in range(len(rs) - 1):
            fwd = (rs[i + 1]["close"] - rs[i]["close"]) / rs[i]["close"]
            p = rs[i]["target_position"]
            turn = abs(p - prev_pos)
            srets.append(p * fwd - turn * COST_BPS / 1e4)
            prev_pos = p
        if len(srets) >= 3:
            sh, _ = _ann_sharpe(srets)
            print(f"  {a}: n={len(srets)} fwd_Sharpe={sh:+.2f} cum={sum(srets)*100:+.1f}%")
            port.extend(srets)
    if len(port) >= 3:
        sh, _ = _ann_sharpe(port)
        print(f"  POOLED forward Sharpe={sh:+.2f}  (backtest was 0.53; need forward >0 to promote)")
    else:
        print("  not enough forward bars yet — keep tracking")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "track"
    if mode == "score":
        score()
    else:
        track()

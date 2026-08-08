"""[P230] Per-AGENT live-IC review — the instrument the P228 decision assumes.

P228 recorded that the 12 zero-weighted ADVISE agents stay off until an agent's
FORWARD live IC clears the P166 cost-aware bar — but no standing tool computed
per-agent IC: the P143/P198 numbers were one-off analyses, and
`compute_shadow_ic` covers the v5.1 strategy shadow ledgers, not the 26 fusion
agents. This closes that gap: it reads the attribution signal logs the tracker
already writes every tick, joins them to public 4H closes, and prints per-agent
forward IC with the same cost-aware arithmetic the promotion path names.

WHERE THIS RUNS (P213 lesson — stated, not implied):
  IN-CONTAINER on the live server, where the attribution volume is:
      docker exec hmats-engine python -X utf8 analytics/ic/agent_ic_review.py
  Or operator-local against scp'd logs:  --log-dir <dir with signals_*.jsonl>
  Prices come from Kraken's PUBLIC OHLC endpoint (no key). A missing log dir
  or an unreachable price API is a REFUSAL (exit 2), never an empty report —
  "no data" must not read as "no signal" (P199/P213/P227b).

Stdlib only, observation only: reads logs, fetches public prices, writes a
report under analytics/ic/reports/. Places no orders, mutates no state.

VERDICT SEMANTICS (mirrors analytics/shadow_ic P166 arithmetic):
  expected_edge_bps(h) = 0.7979 * 2*sin(pi*IC/6) * fwd_vol_bps(h)
  bar: IC > 0 at every horizon, |t| >= 2 (t = IC*sqrt(n-1)), and
  expected_edge >= 2.0 x 6bps (Coinbase taker round trip, safety margin 2).
  A PROMOTE-CANDIDATE here is a candidate for a WEIGHT in
  ADVISE_WEIGHTS_BY_REGIME with its own P-entry — never an automatic change.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import urllib.request
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HORIZON_BARS = (1, 4)          # 4h, 16h on the 4H clock
TAKER_RT_BPS = 6.0             # Coinbase 3bps taker x 2 legs (P166)
SAFETY_MARGIN = 2.0            # spread/impact absent from the fee number
MIN_N = 30
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}


def _refuse(msg: str) -> None:
    print(f"REFUSING TO REPORT: {msg}", file=sys.stderr)
    sys.exit(2)


def resolve_log_dir(cli: str | None) -> Path:
    if cli:
        return Path(cli)
    env = os.environ.get("HMATS_LOG_DIR")
    return Path(env, "attribution") if env else Path("logs/attribution")


def load_signal_records(log_dir: Path, window_days: int) -> list[dict]:
    files = sorted(glob.glob(str(log_dir / "signals_*.jsonl")))
    if not files:
        _refuse(
            f"no signals_*.jsonl under {log_dir}. This tool needs the "
            f"attribution volume — run in-container (docker exec hmats-engine "
            f"python -X utf8 analytics/ic/agent_ic_review.py) or pass "
            f"--log-dir pointing at scp'd logs. An empty dir here is 'no "
            f"data source', not 'no agent signal'."
        )
    cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400
    out = []
    skipped = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    ts = datetime.fromisoformat(rec["ts"]).timestamp()
                except Exception:
                    skipped += 1
                    continue
                if ts >= cutoff and rec.get("asset") in KRAKEN_PAIRS:
                    rec["_ts"] = ts
                    out.append(rec)
    if skipped:
        print(f"  note: {skipped} unparseable lines skipped", file=sys.stderr)
    return out


def fetch_closes(asset: str) -> tuple[list[float], list[float]]:
    """(bar_open_timestamps, closes) for 4H bars from Kraken public OHLC."""
    url = (f"https://api.kraken.com/0/public/OHLC?pair="
           f"{KRAKEN_PAIRS[asset]}&interval=240")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        _refuse(f"Kraken public OHLC unreachable for {asset}: "
                f"{type(e).__name__}: {e} — cannot price forward returns. "
                f"(Network refusal, distinct from 'no signals'.)")
    if payload.get("error"):
        _refuse(f"Kraken OHLC error for {asset}: {payload['error']}")
    key = next(k for k in payload["result"] if k != "last")
    rows = payload["result"][key]
    return [float(r[0]) for r in rows], [float(r[4]) for r in rows]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def required_ic(fwd_vol_bps: float) -> float | None:
    """Invert P166's edge model: IC needed for SAFETY_MARGIN x TAKER_RT."""
    need = SAFETY_MARGIN * TAKER_RT_BPS
    if fwd_vol_bps <= 0:
        return None
    x = need / (0.7979 * 2.0 * fwd_vol_bps)
    if x >= 1.0:
        return None  # unreachable at this vol
    return 6.0 / math.pi * math.asin(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--report-dir", default=None,
                    help="default: analytics/ic/reports next to this file")
    args = ap.parse_args()

    records = load_signal_records(resolve_log_dir(args.log_dir),
                                  args.window_days)
    if not records:
        _refuse(f"signal files exist but contain no records inside the "
                f"{args.window_days}d window — widen --window-days or check "
                f"the tracker is writing.")

    bars = {a: fetch_closes(a) for a in KRAKEN_PAIRS}

    # agent -> horizon -> parallel lists of (direction, fwd_return_frac)
    dirs: dict = defaultdict(lambda: defaultdict(list))
    rets: dict = defaultdict(lambda: defaultdict(list))
    vol_samples: dict = defaultdict(list)
    joined = unjoined = 0
    for rec in records:
        ts_list, closes = bars[rec["asset"]]
        # bar containing the signal = last bar whose open <= ts
        i = bisect_right(ts_list, rec["_ts"]) - 1
        if i < 0:
            unjoined += 1
            continue
        for h in HORIZON_BARS:
            if i + h >= len(closes):
                continue  # forward bar not closed yet — honest gap, not a 0
            fwd = closes[i + h] / closes[i] - 1.0
            vol_samples[h].append(fwd)
            for sig in rec.get("signals", []):
                d = float(sig.get("direction", 0.0) or 0.0)
                if abs(d) < 1e-9:
                    continue  # zero-emission ticks carry no directional info
                a = sig.get("agent_name", "?")
                dirs[a][h].append(d)
                rets[a][h].append(fwd)
        joined += 1
    print(f"records joined={joined} pre-history={unjoined} "
          f"window={args.window_days}d horizons={HORIZON_BARS} bars(4H)")

    fwd_vol = {
        h: (math.sqrt(sum(r * r for r in v) / len(v)) * 1e4 if v else 0.0)
        for h, v in vol_samples.items()
    }
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "window_days": args.window_days,
              "fwd_vol_bps": fwd_vol, "agents": {}}

    print(f"\n{'agent':<14}{'h':>3}{'n':>6}{'IC':>8}{'t':>7}"
          f"{'edge_bps':>9}{'req_IC':>8}  verdict")
    for agent in sorted(dirs):
        rows, verdict_bits = {}, []
        for h in HORIZON_BARS:
            xs, ys = dirs[agent][h], rets[agent][h]
            n = len(xs)
            ic = _pearson(xs, ys)
            if n < MIN_N or ic is None:
                rows[h] = {"n": n, "ic": ic, "verdict": "INSUFFICIENT"}
                verdict_bits.append("INSUFFICIENT")
                continue
            t = ic * math.sqrt(n - 1)
            edge = 0.7979 * 2.0 * math.sin(math.pi * ic / 6.0) * fwd_vol[h]
            req = required_ic(fwd_vol[h])
            clears = (ic > 0 and abs(t) >= 2.0
                      and req is not None and ic >= req)
            rows[h] = {"n": n, "ic": round(ic, 4), "t": round(t, 2),
                       "edge_bps": round(edge, 2),
                       "required_ic": round(req, 4) if req else None,
                       "clears_p166": clears}
            verdict_bits.append(
                "CLEARS" if clears else
                ("NEGATIVE" if (ic < 0 and abs(t) >= 2.0) else
                 ("NOISE" if abs(t) < 1.0 else "WEAK")))
        verdict = ("PROMOTE-CANDIDATE"
                   if verdict_bits and all(v == "CLEARS" for v in verdict_bits)
                   else ("NEGATIVE" if "NEGATIVE" in verdict_bits
                         else ("INSUFFICIENT" if all(
                             v == "INSUFFICIENT" for v in verdict_bits)
                             else "HOLD")))
        report["agents"][agent] = {"horizons": rows, "verdict": verdict}
        for h in HORIZON_BARS:
            r = rows[h]
            print(f"{agent:<14}{h:>3}{r['n']:>6}"
                  f"{(r.get('ic') if r.get('ic') is not None else float('nan')):>8.3f}"
                  f"{r.get('t', float('nan')):>7.2f}"
                  f"{r.get('edge_bps', float('nan')):>9.2f}"
                  f"{(r.get('required_ic') or float('nan')):>8.3f}"
                  f"  {verdict if h == HORIZON_BARS[-1] else ''}")

    rep_dir = Path(args.report_dir) if args.report_dir else (
        Path(__file__).resolve().parent / "reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = rep_dir / f"agent_ic_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    print("NOTE: a PROMOTE-CANDIDATE is a candidate for a weight in "
          "ADVISE_WEIGHTS_BY_REGIME with its own P-entry — never automatic "
          "(P228). Verdicts on <60d of data are provisional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
